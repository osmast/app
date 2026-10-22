"""Acquire step - capture bursts of a target.

A target is either a **well of a configured plate** or **one individual** of the
simple counter. The phone is already connected (on the Setup step) and this screen
simply observes that feed, so there is nothing to connect here. Every burst is
written into the study's folder, scored for sharpness and shown in the strip below
the stream, so you can judge on the spot whether another burst is needed.
"""

import cv2

from daphnia import paths, plates, workflow
from daphnia.db import connection, queries, repository
from daphnia.streaming import shared_service
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import ImageViewer, StreamView, ThumbnailGallery
from daphnia.vision.quality import score_all, select_best

#: data value of the combo entry that is not a plate
INDIVIDUAL = "individual"


class Acquire(ShellPage):
    """Target selection, live preview and burst capture."""

    def __init__(self, parent=None):
        super().__init__(ui_file='acquire.ui', parent=parent)

        self.service = shared_service()
        self.service.status.connect(self.status_label.setText)
        self.service.opened.connect(self._sync_connection)
        self.service.closed.connect(self._sync_connection)

        # Visual area: the stream on top (capped to its own aspect ratio), the
        # frames of the target below - so you can see whether another burst is
        # needed without leaving the screen.
        self.stream = self.add_visual(StreamView(self.service))
        self.gallery = self.add_visual(ThumbnailGallery(selectable=False), stretch=1)
        self.gallery.opened.connect(self._open_fullscreen)

        self.btn_capture.clicked.connect(self._capture_burst)
        self.btn_next_target.clicked.connect(self._next_target)
        self.combo_plate.currentIndexChanged.connect(self._on_plate_changed)
        self.combo_well.currentIndexChanged.connect(self._on_well_changed)
        self.spin_burst.valueChanged.connect(self.service.set_buffer)

        self._spec = None
        self._study_id = None
        self._plates = {}
        self._plate_id = None
        self._well_label = None
        self._individual_index = 1
        self.service.set_buffer(self.spin_burst.value())
        self.refresh_targets()

    def on_shown(self):
        self.refresh_targets()
        self._sync_connection()

    # -- connection --------------------------------------------------------
    def _sync_connection(self):
        """The capture button needs a live feed - the connection is set up on Setup."""
        connected = self.service.connected
        self.btn_capture.setEnabled(connected)
        self.connection_label.setText(
            f"Connected: {self.service.url}" if connected
            else workflow.REASONS["connection"])

    # -- targets -----------------------------------------------------------
    def refresh_targets(self):
        """Reload the study's plates and the individual counter."""
        with connection() as conn:
            spec = queries.active_study_spec(conn)
            plate_rows = queries.list_plates(conn, spec.study_id) if spec else []
            next_individual = (repository.next_individual_index(conn, spec.study_id)
                               if spec else 1)

        self._spec = spec
        self._study_id = spec.study_id if spec else None
        self._plates = {row["plate_id"]: row for row in plate_rows}
        # The study's acquisition protocol sets the default burst size.
        if spec is not None:
            self.spin_burst.setValue(spec.protocol.burst_size)
        if self._individual_index < next_individual:
            self._individual_index = next_individual

        # The plate pickers only make sense for a study whose mode runs plates.
        uses_plates = bool(spec and spec.uses_plates)
        self.combo_plate.setVisible(uses_plates)
        self.combo_well.setVisible(uses_plates)

        self.combo_plate.blockSignals(True)
        self.combo_plate.clear()
        if uses_plates:
            self.combo_plate.addItem("Individual counter", INDIVIDUAL)
            for row in plate_rows:
                self.combo_plate.addItem(
                    f"{row['label']} ({row['format']}-well)", row["plate_id"])
        self.combo_plate.blockSignals(False)
        self._on_plate_changed()

    def _on_plate_changed(self):
        plate_id = self.combo_plate.currentData()
        self._plate_id = None if plate_id in (None, INDIVIDUAL) else plate_id

        self.combo_well.blockSignals(True)
        self.combo_well.clear()
        if self._plate_id:
            row = self._plates[self._plate_id]
            self.combo_well.addItems(plates.well_labels(row["format"]))
            self.combo_well.setEnabled(True)
            self._well_label = self.combo_well.currentText()
        else:
            self.combo_well.setEnabled(False)
            self._well_label = None
        self.combo_well.blockSignals(False)
        self._refresh_progress()

    def _on_well_changed(self):
        self._well_label = self.combo_well.currentText()
        self._refresh_progress()

    def _next_target(self):
        """Step through the wells of the plate, or bump the individual counter."""
        if self._plate_id:
            index = self.combo_well.currentIndex()
            if index < self.combo_well.count() - 1:
                self.combo_well.setCurrentIndex(index + 1)
        else:
            self._individual_index += 1
            self._refresh_progress()

    def _labels(self) -> str:
        """Human description of the current target."""
        specimen = self._spec.specimen if self._spec is not None else "individual"
        if self._plate_id:
            row = self._plates[self._plate_id]
            wells = plates.well_labels(row["format"])
            position = wells.index(self._well_label) + 1
            return f"{row['label']} well {self._well_label} ({position}/{len(wells)})"
        return f"{specimen} {self._individual_index}"

    def _current_observation_id(self):
        """Observation of the current target if it was captured before."""
        if self._study_id is None:
            return None
        with connection() as conn:
            row = queries.find_target(
                conn, study_id=self._study_id, plate_id=self._plate_id,
                well_label=self._well_label, individual_index=self._individual_index)
            return row["observation_id"] if row else None

    def _refresh_progress(self):
        """Show the target and reload its frames into the strip."""
        observation_id = self._current_observation_id()
        counts = {"frames": 0, "bursts": 0}
        assets = []
        if observation_id is not None:
            with connection() as conn:
                counts = queries.frame_counts(conn, observation_id)
                assets = queries.assets_for(conn, observation_id)
        self.gallery.load(assets)
        lines = [f"{self._labels()}\nframes {counts['frames']} in {counts['bursts']} burst(s)"]
        if self._spec is not None and self._spec.uses_plates and not self._plates:
            lines.append("This study runs plates - add one on the Setup step.")
        self.progress_label.setText("\n".join(lines))

    # -- capture -----------------------------------------------------------
    def _capture_burst(self):
        frames = self.service.recent_frames()
        if not frames:
            self.status_label.setText("No frames buffered - check the connection.")
            return

        with connection() as conn:
            study = queries.active_study(conn)
            profile = repository.active_profile(conn)
            if study is None:
                self.status_label.setText(workflow.REASONS["study"])
                return
            if profile is None:
                self.status_label.setText(workflow.REASONS["calibration"])
                return

            session_id = repository.ensure_session(
                conn, study_id=study["study_id"], profile_id=profile.profile_id)
            observation_id = repository.target_observation(
                conn, session_id=session_id, plate_id=self._plate_id,
                well_label=self._well_label, individual_index=self._individual_index)
            burst_index = repository.next_burst_index(conn, observation_id)

            scores = score_all(frames)
            best_index, _ = select_best(frames)
            directory = paths.observation_dir(
                study["study_id"], session_id, observation_id)
            directory.mkdir(parents=True, exist_ok=True)

            for index, (frame, score) in enumerate(zip(frames, scores)):
                file_path = directory / f"b{burst_index:02d}_f{index:02d}.png"
                cv2.imwrite(str(file_path), frame)
                repository.add_media_asset(
                    conn,
                    observation_id=observation_id,
                    file_path=file_path,
                    frame_index=index,
                    burst_index=burst_index,
                    sharpness=score,
                )

            # The primary frame is the sharpest one of the whole target, so a
            # later, worse burst cannot downgrade it.
            sharpest = conn.execute(
                "SELECT asset_id FROM media_assets WHERE observation_id = ?"
                " ORDER BY sharpness DESC LIMIT 1", (observation_id,)).fetchone()
            if sharpest is not None:
                repository.set_primary_asset(conn, observation_id, sharpest["asset_id"])

        self.status_label.setText(
            f"Burst {burst_index} saved for {self._labels()} "
            f"(sharpest frame f{best_index:02d}).")
        self.notify_workflow()
        self._refresh_progress()

    # -- helpers -----------------------------------------------------------
    def _open_fullscreen(self, path):
        ImageViewer(path, parent=self).exec()
