"""Study 2 - the plate page: capture a whole plate, well by well.

The end goal of the project is a plate read well by well and then *measured* per
individual. This surface is the acquisition half of it, built **next to** the governed
Study pipeline rather than out of it: it walks the wells of a 6-, 24- or 96-well plate,
captures the well under the camera - a burst, or a recording that runs until it is stopped
- and lets the plate light up as it goes (`docs/requirements/SR-STUDY2.md`).

The loop *is* the surface: choose the study and the plate, then **capture** - a short
burst of the frames the stream already buffered, or a **recording** that runs from
*Start* to *Stop* and is exactly as long as the operator wants (`SR-STUDY2-013`) - and
**Next well**, moving the plate by hand in between. The recording is the half that
answers a well that drifts in and out of focus: five buffered frames are a lottery, ten
seconds of the stream are not. Nothing about it is a new storage path - a capture is an
``observation`` of a well and its frames are ``media_assets`` with the sharpest frame as
the well's image, exactly as the Acquire step writes them, so everything captured here
is readable by every other surface and by the corpus.

What it does **not** do yet, and says so on screen: measure. A size per individual in the
well - RTMPose on the well's image, or a manual line - needs the keypoint model and a
target per individual inside the well (`SR-STUDY2-006`, deliberately still open).

Gate policy: **looser than Study, stricter than Bench.** A capture - a burst *or* a
recording - needs a study (it owns the folder the images go into), a plate and a well, the
live connection and an active calibration: an image without a scale can never be measured.
There is no step order to walk, and a recording's **length** is the operator's, not a gate:
this page is a plate, not a pipeline.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QInputDialog, QWidget

from daphnia import paths, plates, settings, wells, workflow
from daphnia.db import connection, queries, repository
from daphnia.ids import new_id
from daphnia.streaming import shared_service
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import (
    ImageViewer,
    StreamView,
    ThumbnailGallery,
    WellGrid,
    WellLegend,
)
from daphnia.vision.quality import score_all, sharpness

#: Wells-per-page entries: the stored value and how it is phrased in the combo.
PAGE_ENTRIES = tuple(
    (size, wells.PAGE_SIZE_LABELS.get(size, f"{size} wells"))
    for size in wells.PAGE_SIZES
)

#: Why a recording ended. It is a constant rather than a word at the call site because the
#: sentence that reports a stored recording names the ending (`SR-STUDY2-N06`), and only
#: some of the endings ask for the operator's attention.
RECORD_STOPPED = "stopped"                  # the Stop button
RECORD_LIMIT = "limit"                      # `settings.RECORD_MAX_FRAMES`
RECORD_CONNECTION_LOST = "connection"       # the stream dropped under it
RECORD_LEFT = "left"                        # the window was closed / the page left

#: How each ending reads in a sentence: ``Recording of B02 <this>: burst 3 stored ...``.
RECORD_REASONS = {
    RECORD_STOPPED: "stopped",
    RECORD_LIMIT: "stopped at the frame limit",
    RECORD_CONNECTION_LOST: "ended by the dropped connection",
    RECORD_LEFT: "ended by the page going off screen",
}

#: The line under the record button while nothing is recorded: the one place that says what
#: the button does, in the button's own words.
RECORD_IDLE_TEXT = ("Not recording. Start writes every frame that arrives to this well's "
                    "folder; Stop stores them as one burst of it.")

#: How often the recording line is re-read, in milliseconds. The line counts frames *and*
#: seconds; a timer is what makes the seconds tick between two frames.
RECORD_TICK_MS = 250


@dataclass
class Recording:
    """A recording in progress: where its frames go, and what has been written so far.

    The frames are written **as they arrive** and only their paths and sharpness are kept
    here, because the whole point of a recording is that it may be long: one 1280x720 frame
    of this rig is 1241 kB and costs 17.2 ms to write plus 2.5 ms to score (measured
    2026-10-22, M4, offscreen), so a ten-second take is ~360 MB - holding the frames
    themselves would be that memory *and* those files. The rows are written when the
    recording **stops**, so
    a recording that is still running is invisible to every other surface instead of
    half-stored.
    """

    burst: int                    # the burst number this recording will be stored as
    directory: Path               # the observation's own folder - the burst's own folder
    observation_id: str           # the well's observation: where the rows will point
    well: str                     # the well the camera is over, for every sentence
    started: float                # ``time.monotonic()`` when Start was pressed
    #: ``(frame_index, path, sharpness)`` of every frame that reached the disk, in order.
    entries: list = field(default_factory=list)
    #: Frames ``cv2.imwrite`` refused - counted, never dropped quietly (`SR-STUDY2-N02`).
    failed: int = 0

    @property
    def frames(self) -> int:
        """Frames this recording has tried to store, written or not."""
        return len(self.entries) + self.failed

    def seconds(self) -> float:
        """How long it has been recording, in seconds (0.0 once stopped)."""
        return max(0.0, time.monotonic() - self.started)


class Study2(ShellPage):
    """The wells of one plate, lit by state, captured one burst at a time.

    The visual area is, top to bottom, the plate's **key**, the **plate** itself (one page
    of it), and a strip holding the live preview beside the frames of the well the camera
    is over - the same three questions the operator has: what do the colours mean, which
    wells are still empty, and is the camera actually over the right well.
    """

    def __init__(self, parent=None):
        super().__init__(ui_file='study2.ui', parent=parent)

        # A combo sizes itself to its longest item, and one long study name would then
        # dictate the panel's width (the trap Predict names: 653 px of combo in a 420 px
        # panel). Cap the boxes, and let the label above each of them say the rest.
        for combo in (self.combo_study, self.combo_plate, self.combo_format,
                      self.combo_page):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(16)

        self.service = shared_service()
        # The recording in progress, or ``None`` (`SR-STUDY2-013`). Set *before* the service
        # is wired: `_on_connection` reads it, and a page built while the stream is live
        # takes that path immediately below.
        self._recording = None
        # The line under the record button counts seconds as well as frames, and frames
        # arrive unevenly - the timer is what keeps the seconds honest between two of them.
        self._record_timer = QTimer(self)
        self._record_timer.setInterval(RECORD_TICK_MS)
        self._record_timer.timeout.connect(self._update_record_label)
        self.service.status.connect(self._on_connection)
        self.service.opened.connect(self._on_connection)
        self.service.closed.connect(self._on_connection)

        self.legend = self.add_visual(WellLegend())
        self.grid = self.add_visual(WellGrid(), stretch=1)
        self.grid.well_chosen.connect(self.choose_well)

        # The live preview and the well's own frames, side by side: the preview is the
        # camera, the strip is what was captured on the well - and the two answer "is this
        # the right well?" together, which neither does alone.
        strip = QWidget()
        strip_layout = QHBoxLayout(strip)
        strip_layout.setContentsMargins(0, 0, 0, 0)
        strip_layout.setSpacing(0)
        self.stream = StreamView(self.service)
        self.gallery = ThumbnailGallery(selectable=False)
        self.gallery.opened.connect(self._open_fullscreen)
        strip_layout.addWidget(self.stream)
        strip_layout.addWidget(self.gallery, 1)
        self.add_visual(strip)

        for fmt in plates.FORMATS:
            self.combo_format.addItem(self._format_label(fmt), fmt)
        for size, label in PAGE_ENTRIES:
            self.combo_page.addItem(label, size)

        self.combo_study.currentIndexChanged.connect(self._on_study_changed)
        self.combo_plate.currentIndexChanged.connect(self._on_plate_changed)
        self.combo_page.currentIndexChanged.connect(self._on_page_size_changed)
        self.btn_use_study.clicked.connect(self.use_study)
        self.btn_new_study.clicked.connect(self.new_study_dialog)
        self.btn_add_plate.clicked.connect(self.add_plate)
        self.btn_previous_page.clicked.connect(lambda: self.step_page(-1))
        self.btn_next_page.clicked.connect(lambda: self.step_page(1))
        # The picture in a well is a *view*: switching it changes nothing about the run, so
        # it re-draws the plate from the counts in hand instead of re-reading them.
        self.check_images.toggled.connect(lambda _checked: self._draw())
        self.btn_capture.clicked.connect(self.capture_burst)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_next_well.clicked.connect(self.next_well)
        self.btn_refresh.clicked.connect(self.refresh)

        self._study_id = None
        self._plates = {}          # plate_id -> row (the plates of the open study)
        self._plate = None         # the row of the plate on the bench
        self._pages = ()           # the plate, split into pages of whole rows
        self._page = 0
        self._current = None       # the well the camera is over
        self._counts = {}          # well label -> counts row
        self._progress = ()        # wells.WellProgress, in reading order

        # Read the connection once at build time: the .ui file's own wording is a
        # placeholder, and a line that says "not connected" while the camera is live is
        # worse than no line at all.
        self._update_record_label()
        self._on_connection()
        self.refresh()

    def on_shown(self):
        """Re-read everything: another surface may have captured or measured since.

        Not while a **recording** runs: it belongs to the well the camera is over, and
        re-reading could hand the page another plate in the middle of it.
        """
        if self._recording is None:
            self.refresh()

    def hideEvent(self, event):
        """A recording does not survive its page going off screen (`SR-STUDY2-N06`).

        Another window hides this one (*Go*, *Calibrate...*), and a hidden page that kept
        recording would go on filling a folder nobody is looking at. What arrived is
        stored, and the line says which ending this was.
        """
        if self._recording is not None:
            self.stop_recording(reason=RECORD_LEFT)
        super().hideEvent(event)

    # -- the state of the page, read from the database ----------------------
    def refresh(self):
        """Re-read the studies, the plate on the bench and the wells of that plate."""
        self._refresh_studies()
        self._refresh_plates()
        self._reload(keep_well=False)

    @staticmethod
    def _format_label(fmt) -> str:
        """``'96-well  (8 x 12)'`` - the format, and the shape it implies."""
        rows, columns = plates.rows_columns(fmt)
        return f"{fmt}-well  ({rows} x {columns})"

    def page_size(self) -> int:
        """The chosen wells-per-page: 0 means the whole plate on one page."""
        value = self.combo_page.currentData()
        return 0 if value is None else int(value)
    def page_size(self) -> int:
        """The chosen wells-per-page: 0 means the whole plate on one page."""
        value = self.combo_page.currentData()
        return 0 if value is None else int(value)

    # -- the study and its plates -------------------------------------------
    @staticmethod
    def _fill(combo, entries, wanted):
        """Replace a combo's entries, keeping ``wanted`` selected when it is still there.

        The signals are blocked while the entries are rebuilt: clearing the box would
        otherwise fire a change for "nothing selected" and the page would reload its whole
        state twice per refresh.
        """
        combo.blockSignals(True)
        combo.clear()
        for label, value in entries:
            combo.addItem(str(label), value)
        index = combo.findData(wanted)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _refresh_studies(self, select=None):
        """List the studies, the open project (or the last choice) selected."""
        with connection() as conn:
            rows = queries.list_studies(conn)
            current = queries.active_study(conn)
        wanted = (select or self._study_id
                  or (current["study_id"] if current is not None else None))
        self._fill(self.combo_study,
                   [(row["name"], row["study_id"]) for row in rows], wanted)
        self._study_id = self.combo_study.currentData()

    def _refresh_plates(self, select=None):
        """List the plates of the open study, the plate on the bench selected."""
        with connection() as conn:
            rows = queries.list_plates(conn, self._study_id) if self._study_id else []
        self._plates = {row["plate_id"]: row for row in rows}
        wanted = select or (self._plate["plate_id"] if self._plate is not None else None)
        self._fill(self.combo_plate,
                   [(f"{row['label']} · {row['format']}-well", row["plate_id"])
                    for row in rows], wanted)
        self._plate = self._plates.get(self.combo_plate.currentData())

    def _on_study_changed(self, *_args):
        """Another study: list *its* plates, and start its walk from the beginning."""
        self._study_id = self.combo_study.currentData()
        self._plate = None
        self._refresh_plates()
        self._reload(keep_well=False)

    def _on_plate_changed(self, *_args):
        """Another plate on the bench: a new walk, from its first empty well."""
        plate_id = self.combo_plate.currentData()
        if plate_id == (self._plate["plate_id"] if self._plate is not None else None):
            return
        self._plate = self._plates.get(plate_id)
        self._reload(keep_well=False)

    def _on_page_size_changed(self, *_args):
        """Another page size: the same plate, cut differently - the walk does not move."""
        self._reload(keep_well=True)

    def use_study(self):
        """Make the selected study the one the whole application is working on."""
        if not self._study_id:
            self.status_label.setText("Choose a study first.")
            return None
        with connection() as conn:
            repository.set_current_study(conn, self._study_id)
        self.status_label.setText(
            f"{self.combo_study.currentText()} is now the study the application works on.")
        return self._study_id

    def new_study_dialog(self):
        """Ask for a name, then create a plate study (the one dialog on this page)."""
        name, accepted = QInputDialog.getText(self, "New study", "Name of the study:")
        if not accepted or not str(name).strip():
            return None
        return self.new_study(str(name).strip())

    def new_study(self, name) -> str:
        """Create a study that runs plates, open it here; returns its id (no dialog)."""
        study_id = new_id("STUDY")
        with connection() as conn:
            repository.create_study(
                conn, name=name, destination=str(paths.study_dir(study_id)),
                study_id=study_id, target_mode="plates")
        # The study just created is the one on the bench: creating it is how the operator
        # gets to it, so it is selected (and listed) rather than left in the combo.
        self._study_id = study_id
        self._plate = None
        self.status_label.setText(
            f"Study {name} created - add a plate, then capture it well by well.")
        self.refresh()
        return study_id

    def add_plate(self, fmt=None, label=None):
        """Add a plate of the chosen format to the open study and open it here."""
        if not self._study_id:
            self.status_label.setText("Choose a study first - a plate belongs to one.")
            return None
        fmt = fmt or self.combo_format.currentData()
        name = (label if label is not None else self.edit_plate_label.text()).strip()
        with connection() as conn:
            plate_id = repository.create_plate(
                conn, study_id=self._study_id, format=fmt, label=name or None)
        self.edit_plate_label.clear()
        self._refresh_plates(select=plate_id)
        self._reload(keep_well=False)
        self.status_label.setText(
            f"Plate {self._plate['label']} added ({fmt}-well, "
            f"{plates.well_count(fmt)} wells) - take a burst on each well.")
        return plate_id

    # -- the plate on screen ------------------------------------------------
    def _reload(self, keep_well=True):
        """Read the plate's counts and draw it: the wells, the page and the panel.

        ``keep_well=False`` means "this is another plate, or another study": the well the
        camera is over is then the first one that still needs an image, which is the walk's
        own answer - never a well label remembered from the plate before it.
        """
        fmt = self._plate["format"] if self._plate is not None else None
        self._pages = wells.pages(fmt, self.page_size()) if fmt else ()
        with connection() as conn:
            self._counts = (queries.plate_well_counts(conn, self._plate["plate_id"])
                            if self._plate is not None else {})
        self._progress = wells.progress(fmt, self._counts) if fmt else ()
        # if not keep_well or self._current not in wells.by_label(self._progress):
        #     self._current = wells.next_pending(self._progress)
        self._page = (wells.page_index(self._pages, self._current)
                      if self._pages else 0)
        self._draw()

    def _draw(self):
        """Paint the page of the plate, and the lines that describe it."""
        if not self._pages:
            self.grid.set_page((), 1, images=self.check_images.isChecked())
            self.legend.set_summary(None, "No plate open.")
            self.plate_label.setText("No plate in this study yet.")
            self.page_label.setText(
                "No plate open - add one above (6, 24 or 96 wells).")
            self.progress_label.setText("-")
            self.well_label.setText("No well chosen.")
            self.gallery.load([])
            self._sync_state()
            return
        fmt = self._plate["format"]
        rows, columns = plates.rows_columns(fmt)
        labels = self._pages[self._page]
        # Each captured well draws its own frame, so the plate fills up as the operator
        # walks it - the burst appears where it was taken, not only in the strip below.
        self.grid.set_page(labels, columns, self._counts, self._current,
                           images=self.check_images.isChecked())
        counted = wells.summary(self._progress)
        self.legend.set_summary(
            counted,
            f"{self._plate['label']} · {rows} x {columns} · {counted['wells']} wells")
        self.plate_label.setText(
            f"{self._plate['label']} - {fmt}-well plate ({rows} x {columns})")
        self.page_label.setText(
            f"Page {self._page + 1} of {len(self._pages)} - wells {labels[0]} to "
            f"{labels[-1]} ({len(labels)} of {counted['wells']} wells)")
        self.progress_label.setText(
            f"{counted['done']} of {counted['wells']} wells have an image; "
            f"{counted['measured']} measured")
        self._show_current()

    def _show_current(self):
        """Show the well the camera is over: its counts, its image and its frames."""
        row = (wells.by_label(self._progress).get(self._current)
               if self._current else None)
        if row is None:
            self.well_label.setText("No well chosen.")
            self.gallery.load([])
            self._sync_state()
            return
        assets = []
        with connection() as conn:
            target = queries.find_target(conn, study_id=self._study_id,
                                         plate_id=self._plate["plate_id"],
                                         well_label=self._current)
            if target is not None:
                assets = queries.assets_for(conn, target["observation_id"])
        primary = next((asset for asset in assets if asset["is_primary"]), None)
        lines = [row.describe()]
        if primary is not None:
            lines.append(f"Image: {Path(primary['file_path']).name}"
                         " (the sharpest frame of the burst)")
        self.well_label.setText("\n".join(lines))
        self.gallery.load(assets)
        if primary is not None:
            ids = self.gallery.asset_ids()
            if primary["asset_id"] in ids:
                # The strip holds every frame of every burst; the well's *image* is the one
                # it has selected, so a later burst's frames cannot pass for it unnoticed.
                self.gallery.setCurrentRow(ids.index(primary["asset_id"]))
        self._sync_state()

    def choose_well(self, well_label):
        """Make ``well_label`` the well the camera is over - and show its page."""
        if not well_label:
            return None
        if self._recording is not None and well_label != self._current:
            # A recording belongs to **one** well: moving the camera mid-recording would
            # write the rest of it into another well's folder. The page's own controls are
            # locked while it runs; this is the refusal behind them (`SR-STUDY2-014`).
            self.status_label.setText(
                f"Recording on {self._recording.well} - press Stop to store the burst "
                f"before moving to {well_label}.")
            return None
        self._current = well_label
        page = wells.page_index(self._pages, well_label)
        if page != self._page:
            self._page = page                  # the well is on another page of the plate
            self._draw()
        else:
            self.grid.set_current(well_label)
            self._show_current()
        return well_label

    def next_well(self):
        """Choose the next well with no image yet - the operator moves the plate there."""
        label = wells.next_pending(self._progress, after=self._current)
        if label is None:
            counted = wells.summary(self._progress)
            self.status_label.setText(
                f"Every well of {self._plate['label']} has an image "
                f"({counted['wells']} wells, {counted['measured']} measured). "
                f"Measuring a well's individuals is the next step - this page does not "
                f"measure yet.")
            return None
        self.choose_well(label)
        self.status_label.setText(f"Camera over {label} - move the plate there.")
        return label

    def step_page(self, step):
        """Turn the page of the plate: the wells shown change, the plate does not."""
        if not self._pages:
            return None
        target = max(0, min(len(self._pages) - 1, self._page + int(step)))
        if target == self._page:
            return None
        self._page = target
        self._draw()
        return self._page

    # -- the camera, the burst and the recording ----------------------------
    def _on_connection(self, *_args):
        """The shared connection changed: say so, and let the capture buttons follow it.

        A **recording** cannot outlive the connection it is recording: what arrived is
        stored, and the ending is named (`SR-STUDY2-N06`) - a recording that quietly went
        on "recording" into nothing would be the worst of the two.
        """
        self.connection_label.setText(
            f"Connected: {self.service.url}" if self.service.connected
            else workflow.REASONS["connection"])
        if self._recording is not None and not self.service.connected:
            self.stop_recording(reason=RECORD_CONNECTION_LOST)
        self._sync_state()

    def _sync_state(self):
        """Enable what can act right now: a capture needs a camera, a plate and a well.

        While a **recording** runs the page is locked to its well: the plate, the study,
        the page size, the walk and *Re-read the plate* are all off, because moving the
        plate on mid-recording would write the rest of it into another well's folder
        (`SR-STUDY2-014`). The plate and its colours stay visible - they are the answer to
        "where am I", which is exactly the question a lock must not take away.
        """
        recording = self._recording is not None
        ready = (self.service.connected and self._plate is not None
                 and bool(self._current))
        self.btn_capture.setEnabled(ready and not recording)
        self.btn_record.setEnabled(recording or ready)
        self.btn_record.setText("Stop recording and store" if recording
                                else "Start recording")
        self.btn_next_well.setEnabled(self._plate is not None and not recording)
        self.btn_add_plate.setEnabled(bool(self._study_id) and not recording)
        self.btn_use_study.setEnabled(bool(self._study_id) and not recording)
        self.btn_previous_page.setEnabled(self._page > 0 and not recording)
        self.btn_next_page.setEnabled(bool(self._pages) and not recording
                                      and self._page < len(self._pages) - 1)
        self.btn_refresh.setEnabled(not recording)
        self.check_advance.setEnabled(not recording)
        for combo in (self.combo_study, self.combo_plate, self.combo_format,
                      self.combo_page):
            combo.setEnabled(not recording)
        self.grid.setEnabled(not recording)

    def _prepare_target(self):
        """Resolve where the next capture of the well the camera is over is stored.

        Returns ``(study, session_id, observation_id, burst, directory)`` - the study (it
        owns the folder), the session (it froze the scale), the observation (it *is* the
        well) and the next burst number of that observation - or ``None``, with the refusal
        on the screen. One resolver, shared by a burst and a recording, is what keeps the
        two from ever storing differently (`SR-STUDY2-N03`).
        """
        if self._plate is None or not self._current:
            self.status_label.setText("Choose a plate and a well first.")
            return None
        with connection() as conn:
            study = queries.study_by_id(conn, self._study_id)
            profile = repository.active_profile(conn)
            if study is None:
                self.status_label.setText(workflow.REASONS["study"])
                return None
            if profile is None:
                self.status_label.setText(workflow.REASONS["calibration"])
                return None

            # The session locks the scale every frame of this run is captured under, and
            # the observation is the well: both are the Acquire step's rows, not a second
            # data model beside it.
            session_id = repository.ensure_session(
                conn, study_id=study["study_id"], profile_id=profile.profile_id)
            observation_id = repository.target_observation(
                conn, session_id=session_id, plate_id=self._plate["plate_id"],
                well_label=self._current)
            burst = repository.next_burst_index(conn, observation_id)
            directory = paths.observation_dir(
                study["study_id"], session_id, observation_id)
            directory.mkdir(parents=True, exist_ok=True)
        return study, session_id, observation_id, burst, directory

    def _write_rows(self, observation_id, burst, entries):
        """Register written frames as one burst of ``observation_id``; return the count.

        ``entries`` are ``(frame_index, path, sharpness)`` in the order the frames arrived,
        and the files are **already on disk** - the burst writer and the recording both
        write as they go. Returns ``(written, best_index)``: how many rows carry the burst,
        and which frame index is the sharpest of them (``None`` when there are none).

        The sharpest frame of everything ever captured on the observation becomes the
        well's **image** (the Acquire step's rule), so a later, worse burst cannot downgrade
        a well that already has a good picture.
        """
        if not entries:
            return 0, None
        best_index = max(entries, key=lambda entry: entry[2])[0]
        with connection() as conn:
            for index, file_path, score in entries:
                repository.add_media_asset(
                    conn, observation_id=observation_id, file_path=file_path,
                    frame_index=index, burst_index=burst, sharpness=score)
            sharpest = conn.execute(
                "SELECT asset_id FROM media_assets WHERE observation_id = ?"
                " ORDER BY sharpness DESC LIMIT 1", (observation_id,)).fetchone()
            if sharpest is not None:
                repository.set_primary_asset(conn, observation_id, sharpest["asset_id"])
        return len(entries), best_index

    def capture_burst(self):
        """Store the buffered frames for the well the camera is over.

        Returns the burst number, or ``None`` when nothing was written - and every refusal
        says why, on the screen: a recording already running, no frames buffered, no study,
        no active calibration, no plate or well chosen, or a folder nothing could be
        written into. The burst number is what makes a second visit to the same well an
        *appended* burst instead of an overwrite (`SR-STUDY2-005`).

        A burst is the **short** capture: the ``DAPHNIA_BURST_SIZE`` frames the stream has
        already buffered. When five frames are a lottery - a well that drifts in and out of
        focus - the recording is the other half of the loop (`SR-STUDY2-013`).
        """
        if self._recording is not None:
            self.status_label.setText(
                f"Recording on {self._recording.well} - press Stop to store the burst "
                f"before taking another one.")
            return None
        frames = self.service.recent_frames()
        if not frames:
            self.status_label.setText(
                "No frames buffered - the camera is not delivering. The connection is "
                "shared: connect it in the bar of any window (Source), then try again.")
            return None
        target = self._prepare_target()
        if target is None:
            return None
        _study, _session_id, observation_id, burst, directory = target

        scores = score_all(frames)
        entries = []
        for index, (frame, score) in enumerate(zip(frames, scores)):
            file_path = directory / f"b{burst:02d}_f{index:02d}.png"
            if not cv2.imwrite(str(file_path), frame):
                continue        # counted below - a frame that was not written is said
            entries.append((index, file_path, score))

        written, best_index = self._write_rows(observation_id, burst, entries)
        if not written:
            self.status_label.setText(
                f"Nothing was written to {directory} - the burst is not stored.")
            return None
        well = self._current
        shortfall = "" if written == len(frames) else f" ({written} of {len(frames)} frames)"
        self.status_label.setText(
            f"Burst {burst} stored for {well}{shortfall}; sharpest frame of this burst: "
            f"f{best_index:02d}.")
        self._reload(keep_well=True)
        if self.check_advance.isChecked():
            following = wells.next_pending(self._progress, after=self._current)
            if following:
                self.choose_well(following)
                self.status_label.setText(
                    f"Burst {burst} stored for {well}{shortfall}. Next well: {following} "
                    f"- move the plate there.")
        return burst

    # -- the recording, where the length is the operator's (`SR-STUDY2-013`) --
    def toggle_recording(self):
        """The one button of the recording: it starts one, or stops the one running."""
        if self._recording is None:
            return self.start_recording()
        return self.stop_recording()

    def start_recording(self):
        """Open a recording on the well the camera is over; return its burst number.

        A recording is a burst whose **length the operator decides** (`SR-STUDY2-013`):
        every frame that arrives between Start and Stop is written to the well's own folder
        as it comes, and its rows are written when it stops. The refusals are the burst's
        refusals - a plate and a well, a study, an active scale, the connection - because a
        recording is stored by the same writer (`SR-STUDY2-N01`, `SR-STUDY2-N03`).
        """
        if self._recording is not None:
            return self._recording.burst
        if not self.service.connected:
            self.status_label.setText(workflow.REASONS["connection"])
            return None
        target = self._prepare_target()
        if target is None:
            return None
        _study, _session_id, observation_id, burst, directory = target
        self._recording = Recording(burst=burst, directory=directory,
                                    observation_id=observation_id, well=self._current,
                                    started=time.monotonic())
        # A **bound method**, never a lambda: the stream service outlives this page (the
        # trap `mlContext.md` §6 names), so PySide6 has to be able to take the slot off the
        # service when the page goes away instead of calling into a dead widget.
        self.service.frames_changed.connect(self._record_frame)
        self._record_timer.start()
        self._update_record_label()
        self._sync_state()
        waiting = "" if self.service.has_signal else (
            " No frame has arrived yet - the recording starts with the picture.")
        self.status_label.setText(
            f"Recording on {self._current}: every frame that arrives is written to "
            f"{directory.name}, and Stop stores them as one burst of this well "
            f"(limit {settings.RECORD_MAX_FRAMES} frames).{waiting}")
        return burst

    def stop_recording(self, reason=RECORD_STOPPED):
        """Store the running recording as one burst of its well; return its number.

        Returns ``None`` when nothing was running, and ``None`` - with the reason on screen
        - when no frame reached the disk: a recording that stored nothing is **not** stored,
        and says so (`SR-STUDY2-N02`). A stored one is reported with the frames written, the
        seconds it ran, and *which* ending this was (`SR-STUDY2-N06`).
        """
        recording, self._recording = self._recording, None
        if recording is None:
            return None
        self.service.frames_changed.disconnect(self._record_frame)
        self._record_timer.stop()
        written, best_index = self._write_rows(
            recording.observation_id, recording.burst, recording.entries)
        seconds = recording.seconds()
        if not written:
            self.record_label.setText(RECORD_IDLE_TEXT)
            self.status_label.setText(
                f"No frame reached {recording.directory} in {seconds:.1f} s - the "
                f"recording of {recording.well} is not stored.")
            self._sync_state()
            return None
        shortfall = (f"; {recording.failed} frames could not be written"
                     if recording.failed else "")
        self.record_label.setText(
            f"Stored: burst {recording.burst} of {recording.well}, {written} frames, "
            f"{seconds:.1f} s.")
        self.status_label.setText(
            f"Recording of {recording.well} {RECORD_REASONS[reason]}: burst "
            f"{recording.burst} stored with {written} frames in {seconds:.1f} s"
            f"{shortfall}; sharpest frame of this burst: f{best_index:02d}.")
        self._reload(keep_well=True)
        if self.check_advance.isChecked():
            following = wells.next_pending(self._progress, after=self._current)
            if following:
                self.choose_well(following)
                self.status_label.setText(
                    f"Recording of {recording.well} stored as burst {recording.burst} "
                    f"({written} frames in {seconds:.1f} s). Next well: {following} - move "
                    f"the plate there.")
        self._sync_state()
        return recording.burst

    def _record_frame(self):
        """Write the newest frame of the running recording, and count what was written.

        Connected to ``frames_changed``, so it runs once per frame that arrives - the whole
        record, not a sample of it. A frame ``cv2.imwrite`` refuses is **counted**, never
        dropped quietly: the stop line then says how many frames are on disk
        (`SR-STUDY2-N02`). Reaching `settings.RECORD_MAX_FRAMES` ends the recording and
        stores it - a guard that names itself on screen, because a recording nobody stops
        must not fill the disk in silence (`SR-STUDY2-N06`).
        """
        recording = self._recording
        frame = self.service.current_frame()
        if recording is None or frame is None:
            return
        index = recording.frames
        file_path = recording.directory / f"b{recording.burst:02d}_f{index:02d}.png"
        if not cv2.imwrite(str(file_path), frame):
            recording.failed += 1
        else:
            recording.entries.append((index, file_path, sharpness(frame)))
        if recording.frames >= settings.RECORD_MAX_FRAMES:
            self.stop_recording(reason=RECORD_LIMIT)

    def _update_record_label(self):
        """Say how the recording is going: its well, its frames, its seconds - and the limit.

        The seconds come from the recording's own clock, so the line keeps ticking between
        two frames, and the limit is part of the line because a recording that is about to
        end itself should not surprise anyone.
        """
        recording = self._recording
        if recording is None:
            self.record_label.setText(RECORD_IDLE_TEXT)
            return
        self.record_label.setText(
            f"Recording {recording.well}: {recording.frames} frames · "
            f"{recording.seconds():.1f} s (limit {settings.RECORD_MAX_FRAMES}).")

    def _open_fullscreen(self, path):
        """A frame of the strip, opened full size (the same viewer every surface uses)."""
        ImageViewer(path, parent=self).exec()

