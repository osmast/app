"""Predict step - watch the trained models find the animals *and* their keypoints.

The Localize step, made visible - and, once a pose model is loaded, the whole
**top-down** pipeline: the detector finds each animal, and the pose model places
`head` and `tail` inside that box, drawing the body-length line it implies. Two
sources, one detector:

* the **live feed** (the shared :class:`~daphnia.streaming.StreamService`, so the
  phone is connected once for the whole app), and
* a **video file** or a single **image** from disk, which is what a figure, a repeat
  or a quick sanity check needs.

Inference runs in an :class:`InferenceWorker` thread and always works on the
**latest** frame: a detector slower than the frame rate must drop frames, not queue
them. The frame is displayed on the ordinary
:class:`~daphnia.ui.widgets.ImageCanvas`, so zoom and pan behave exactly as they do
on every other screen, each detection is drawn as a box with its score, and each
pose as the head-to-tail line with a dot on each keypoint.

The pose model is optional and its cost is real: **Keypoints** (the tool on the
canvas, on by default when a pose model exists) can be switched off to watch the
detector alone. Lengths are reported in **pixels** and labelled unscaled, because
this surface holds no optical profile and a micrometre nobody calibrated would be a
number without its scale (``projectbrief.md``, non-negotiable 1).

Nothing here is stored (``SR-PREDICT-N01``): a prediction is not ground truth - not a
box and not a keypoint. The surface's job is to answer "do the models see the
animal?" - and to be honest about which checkpoint answered it.
"""

import time
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QComboBox, QFileDialog, QTreeWidgetItem

from daphnia import models, paths, settings
from daphnia.db import connection, queries, repository
from daphnia.streaming import shared_service
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import SOURCE_MENU, ImageCanvas, Kpi, ToolSpec
from daphnia.vision.calibration import is_stale
from daphnia.vision.tracking import Measurement, Tracker
from daphnia.vision.rtmdet.detect import (
    ModelUnavailable,
    checkpoint_path,
    load_detector,
)
from daphnia.vision.rtmpose.detect import (
    ModelUnavailable as PoseUnavailable,
    checkpoint_path as pose_checkpoint_path,
    load_pose_model,
)

#: How long the worker waits between attempts when there is nothing new to run on.
IDLE_SLEEP_MS = 5

#: Filters the file dialogs offer.
IMAGE_FILTER = ("Images (*.png *.jpg *.jpeg *.heic *.heif *.tif *.tiff *.bmp "
                "*.webp);;All files (*)")
VIDEO_FILTER = ("Video (*.mp4 *.mov *.m4v *.avi *.mkv *.ts);;All files (*)")
CHECKPOINT_FILTER = "Torch checkpoint (*.pt *.pth);;All files (*)"

#: What may be written **above** each animal, and what the average is.
#:
#: ``frame`` is this frame's prediction on its own - honest and flickery. ``average``
#: follows each animal across frames and writes the **mean of its repeats** with the
#: standard error of that mean and its ``n``, which is the number worth reading once a
#: container holds several specimens. ``off`` keeps the keypoints and drops every
#: number, for watching behaviour rather than reading a length.
LABEL_MODES = (
    ("frame", "Per frame (this prediction)"),
    ("average", "Average (tracked animals)"),
    ("off", "Hidden (no numbers)"),
)

#: The mode a session starts in: the average, because that is the measurement. The
#: per-frame value is one click away, and the spread is shown with it either way.
DEFAULT_LABEL_MODE = "average"


class InferenceWorker(QThread):
    """Runs the detector off the GUI thread, on the newest frame there is.

    Signals:
        frame_ready(object, object, float, int, object): the frame, its detections,
            the inference time in milliseconds, the frame index (0 for the live feed,
            where "which frame" is not a meaningful question), and the **poses** for
            that same frame (an empty list when none were asked for). The poses travel
            in the signal rather than on the worker so the picture and its keypoints can
            never come from two different frames.
        status(str): what the worker is doing, for the status label.
        progress(int, int): the video position and length, in frames.
        message(str): a failure the page must show rather than swallow.
    """

    frame_ready = Signal(object, object, float, int, object)
    status = Signal(str)
    progress = Signal(int, int)
    message = Signal(str)

    #: Where the frames come from.
    LIVE = "live"
    VIDEO = "video"

    def __init__(self, detector, source=LIVE, service=None, path=None, parent=None,
                 pose_model=None):
        super().__init__(parent)
        self.detector = detector
        self.source = source
        self.service = service
        self.path = Path(path) if path else None
        self.pose_model = pose_model      # optional: the second stage of top-down
        self._last_poses = []             # re-bound every pass, never mutated
        self._running = False
        self._paused = False
        self._step = False
        self._index = 0
        self._seek_to = None

    # -- control -----------------------------------------------------------
    def stop(self):
        """Ask the loop to end, and wait: a QThread must not outlive its page."""
        self._running = False
        if self.isRunning() and not self.wait(4000):
            self.terminate()
            self.wait(1000)

    def set_paused(self, paused):
        """Pause/resume the video. The live feed is never paused - it just skips."""
        self._paused = bool(paused)

    def request_step(self):
        """Detect on exactly one more frame of the video."""
        self._step = True

    def seek(self, index):
        """Jump the video to a frame index."""
        self._seek_to = int(index)

    # -- the loop ----------------------------------------------------------
    def run(self):
        self._running = True
        try:
            if self.source == self.VIDEO:
                self._run_video()
            else:
                self._run_live()
        except Exception as error:                  # noqa: BLE001 - report, never crash
            self.message.emit(f"The detector stopped: {error}")
        finally:
            self._running = False

    def _run_video(self):
        """Walk a video file, detecting on every frame it plays."""
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            self.message.emit(f"Could not open the video {self.path.name}.")
            return
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.status.emit(f"{self.path.name}: {total} frame(s).")
        try:
            while self._running:
                if self._seek_to is not None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, self._seek_to))
                    self._index = max(0, self._seek_to)
                    self._seek_to = None
                if self._paused and not self._step:
                    self.msleep(IDLE_SLEEP_MS * 4)
                    continue
                ok, frame = capture.read()
                if not ok:
                    self.status.emit("End of the video.")
                    self._running = False
                    self.progress.emit(self._index, total)
                    return
                self._step = False
                self._index += 1
                detections, elapsed = self._infer(frame)
                self.frame_ready.emit(frame, detections, elapsed, self._index,
                                      self._last_poses)
                self.progress.emit(self._index, total)
        finally:
            capture.release()

    def _run_live(self):
        """Detect on the newest frame of the shared stream, skipping the rest.

        The frame **identity** is compared, not its content: if the service has not
        buffered a new frame since the last pass there is nothing to do - so a 30 fps
        stream with a 12 fps detector shows 12 boxes a second instead of working
        through a backlog it can never clear.
        """
        last_frame = None
        while self._running:
            frame = self.service.current_frame() if self.service else None
            if frame is None or frame is last_frame:
                self.msleep(IDLE_SLEEP_MS)
                continue
            last_frame = frame
            detections, elapsed = self._infer(frame)
            self.frame_ready.emit(frame, detections, elapsed, 0, self._last_poses)

    def _infer(self, frame):
        """Time one pass of the detector - and of the pose model, if there is one.

        **Measured, not estimated** (``SR-PREDICT-N04``), and the pose stage is
        *inside* the timing on purpose: the number the operator reads is what one
        whole frame of the pipeline costs, not the detector's share of it. The poses
        are left in :attr:`_last_poses` (a fresh list each pass) so the signature
        stays the two-value one the rest of the page and its tests already use.
        """
        started = time.perf_counter()
        detections = self.detector.detect(frame)
        self._last_poses = (self.pose_model.detect_frame(frame, detections=detections)
                            if self.pose_model is not None else [])
        return detections, (time.perf_counter() - started) * 1000.0


#: How long a calibration id may be in the *Scale* combo. A profile id is a whole setup
#: key (`IPHONE_1.0X_INFINITY_20261017_164855`), and a ``QComboBox`` sizes itself to its
#: longest item - so a long id decides the width of the controls panel. The full id stays
#: in the item's tooltip, and the choice is by data (`currentData()`), never by the label.
SHORT_ID_CHARS = 22


def short_id(text) -> str:
    """A profile id shortened for a combo entry - elided with "…", never silently."""
    text = str(text or "")
    return text if len(text) <= SHORT_ID_CHARS else text[:SHORT_ID_CHARS - 1] + "…"


class Predict(ShellPage):
    """Realtime RTMDet boxes on the live feed, a video file, or one image."""

    def __init__(self, parent=None):
        super().__init__(ui_file='predict.ui', parent=parent)
        # The combo must not decide the panel's width: with the default policy it sizes
        # itself to its longest item, and one long calibration id then clipped the whole
        # panel (measured: 653 px of combo in a 420 px panel). This keeps the *panel*
        # authoritative and elides the label instead.
        self.combo_scale.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_scale.setMinimumContentsLength(18)

        self.service = shared_service()
        self.canvas = self.add_visual(
            ImageCanvas(max_points=0, point_names=[]))
        # A file the canvas cannot read is refused in the panel's status line, by name
        # and reason - this screen has no image loader of its own (see
        # ImageCanvas.set_source).
        self.canvas.source_failed.connect(self.status_label.setText)
        # The detector's view wears the same tools as every other canvas - zoom, fit,
        # pan, full screen - and nothing to measure with: a prediction is not a label.
        # Frame stepping *is* offered, but only while a video file is open: that is the
        # one source with neighbouring frames to step to (`refresh_frame_tools`).
        self.workspace = self.canvas_workspace(self.canvas)
        self.workspace.set_group("Frames")
        self.workspace.add_tool(
            ToolSpec("previous_frame", "< Frame",
                     "Detect on the previous frame of the video (Left arrow)."),
            self.previous_frame)
        self.workspace.add_tool(
            ToolSpec("next_frame", "Frame >",
                     "Detect on the next frame of the video (Right arrow)."),
            self.next_frame)
        # The second stage of the pipeline, and it lives where the work is: a toggle,
        # because the pose pass costs real time and watching the detector *alone* is a
        # question this surface must still be able to answer. It is hidden until a
        # pose model is loaded - a control that cannot act is noise.
        self.workspace.add_tool(
            ToolSpec("keypoints", "Keypoints",
                     "Place head and tail on each detected animal and draw the "
                     "length line (needs a pose model - see docs/rtmpose.md).",
                     checkable=True, checked=True))
        self.workspace.action("keypoints").toggled.connect(
            self._on_keypoints_toggled)

        # *Source*: what the surface looks at. The page's entries land in the header's
        # Source menu, under the app's own connection entry - so "connect the camera" and
        # "open a video instead" are one decision in one place (`X-N13`), and neither is a
        # sentence in the bar.
        self.workspace.set_group(SOURCE_MENU)
        self.workspace.add_tool(
            ToolSpec("live", "Use the live feed",
                     "Detect on the shared camera connection. The check mark in the "
                     "Source menu above says whether the stream is up."),
            self.use_live_feed)
        self.workspace.add_tool(
            ToolSpec("open_video", "Open a video file...",
                     "Detect on every frame of a clip, and step through it frame by "
                     "frame."),
            self._choose_video)
        self.workspace.add_tool(
            ToolSpec("open_image", "Open a single image...",
                     "One frame, one pass of the detector - the fastest answer to "
                     "\"does it see this one?\"."),
            self._choose_image)

        self.detector = None          # a Detector, once one is loaded
        self.pose_model = None        # a PoseModel, once one is loaded (optional)
        self.scale = None             # the OpticalProfile in force, or None
        self.um_per_px = None         # micrometres one pixel spans, from that scale
        self.tracker = Tracker()      # identities that outlive the frame
        self._tracks = []             # the tracks seen on the frame on screen
        self._label_mode = DEFAULT_LABEL_MODE
        self.worker = None            # the InferenceWorker, while it runs
        self._frame = None            # the frame on screen
        self._index = 0               # which frame of a video that is (0 = none yet)
        self._viewed = False          # has the canvas been fitted to this source?
        self._detections = []         # the last frame's detections
        self._poses = []              # the last frame's keypoints
        self._pose_enabled = True     # the Keypoints toggle's state
        self._pose_error = ""         # why the pose model did not load, if it did not
        self._scale_error = ""        # why the calibration list could not be read
        self._latencies = []          # a short window, for the reported average

        # Arrow keys step the video, and only while a video file is open - see
        # `refresh_frame_tools`. They are window shortcuts, so they win over the
        # canvas' own arrow-key panning (as the A/D and arrow keys already do on the
        # Dataset queue); with a live feed or a single image they are off and the
        # arrows pan the picture as they do everywhere else.
        self.shortcut_previous_frame = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_previous_frame.setContext(Qt.WindowShortcut)
        self.shortcut_previous_frame.activated.connect(self.previous_frame)
        self.shortcut_next_frame = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_next_frame.setContext(Qt.WindowShortcut)
        self.shortcut_next_frame.activated.connect(self.next_frame)

        for name in ("btn_play", "btn_stop", "btn_refresh_scale",
                     "btn_reset_tracks"):
            getattr(self, name).clicked.connect(self._on_button)
        self.spin_score.valueChanged.connect(self._on_score_changed)
        self.spin_iou.valueChanged.connect(self._on_iou_changed)
        self.slider_position.sliderMoved.connect(self._on_seek)
        self.combo_scale.currentIndexChanged.connect(self._apply_scale)
        self.spin_um_per_px.valueChanged.connect(self._apply_scale)
        for key, label in LABEL_MODES:
            self.combo_labels.addItem(label, key)
        keys = [key for key, _label in LABEL_MODES]
        self.combo_labels.setCurrentIndex(keys.index(DEFAULT_LABEL_MODE))
        self.combo_labels.currentIndexChanged.connect(self._on_label_mode_changed)
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)

        # The camera state decides whether the live feed is even an option.
        #
        # Connected to a **bound method**, never a lambda: PySide6 only
        # auto-disconnects a connection whose receiver is a QObject, and a lambda has
        # none. `shared_service()` outlives this page, so a lambda keeps firing after
        # the window is closed and reaches into freed C++ widgets
        # ("libshiboken: Internal C++ object already deleted").
        for signal in (self.service.status, self.service.opened, self.service.closed):
            signal.connect(self._refresh_state)

        self.refresh_scales()
        self._report_detector()
        self._refresh_state()

    def on_shown(self):
        self._refresh_state()
        # A calibration is usually verified in Optics *while* this window is open, so
        # the list is re-read whenever the surface comes to the front - a stale list
        # would keep measuring under a scale that is no longer in force.
        self.refresh_scales()

    # -- the models --------------------------------------------------------
    def _report_detector(self):
        """Say what the models are, or exactly how to get each one.

        Two lines, because there are two models and the second is optional: the label
        must never leave the operator wondering whether the keypoints are missing
        because there are none or because the pose model was never built.
        """
        if self.detector is not None:
            detector_text = "Detector: " + self.detector.describe()
        else:
            from daphnia.vision.rtmdet import torch_status

            path = checkpoint_path()
            if path.exists():
                detector_text = (f"Detector found at {path.name} but not loaded. "
                                 f"Press Load detector.\n{torch_status()}")
            else:
                detector_text = f"No detector at {path}.\n{torch_status()}"
        self.detector_label.setText(detector_text + "\n" + self._pose_line())
        self.refresh_keypoint_tools()

    def _pose_line(self) -> str:
        """The label's second line: the pose model, or how to build one."""
        if self._pose_error:
            return self._pose_error.splitlines()[0]
        if self.pose_model is not None:
            return "Pose: " + self.pose_model.describe() + "   (head -> tail)"
        path = pose_checkpoint_path()
        if path.exists():
            return (f"Pose found at {path.name} but not loaded. Press Load detector "
                    "to load both.")
        return (f"No pose model at {path}. Optional - the localize half still runs; "
                "`make train-pose` builds one (docs/rtmpose.md).")

    def load_detector(self, path=None) -> bool:
        """Load the detector, **and then** the pose model if there is one.

        One button for both, because this surface *is* the top-down pipeline and a
        half-loaded one is confusing: the boxes appear and the keypoints silently do
        not. The pose model is optional, so its absence never fails the load - it is
        said on the label instead.
        """
        try:
            self.detector = load_detector(
                path, score_threshold=self.spin_score.value(),
                iou_threshold=self.spin_iou.value())
        except ModelUnavailable as error:
            self.detector = None
            self.detector_label.setText(str(error))
            self.status_label.setText("No detector loaded.")
            self._refresh_state()
            return False
        self._latencies = []
        self.load_pose()
        self._report_detector()
        self.status_label.setText(
            "Detector and pose model loaded. Choose a source."
            if self.pose_model is not None
            else "Detector loaded. Choose a source. (No pose model - boxes only.)")
        self._refresh_state()
        return True

    def load_pose(self, path=None) -> bool:
        """Load the optional pose model; returns whether there is one now.

        Dialog-free, like :meth:`load_detector`, so the loading half can be driven
        directly (and tested). A missing checkpoint is **not** an error - boxes are
        what this surface must always be able to show.
        """
        self._pose_error = ""
        try:
            self.pose_model = load_pose_model(path)
        except PoseUnavailable as error:
            self.pose_model = None
            self._pose_error = str(error)
            return False
        self._poses = []
        return True

    def _refresh_state(self, *_):
        """Follow what is loaded: the tools a model unlocks, and the page's own line.

        The **source** entries are always offered (they are menu items in the *Source*
        menu now, `X-N13`): opening a video or an image before a model is loaded is a
        legitimate thing to do - the frame is shown, it simply has no boxes yet.
        """
        self.refresh_keypoint_tools()
        self._refresh_source()

    def adopt_models(self, detector=None, pose=None) -> bool:
        """Take a model the window loaded from the *Detector (RTMDet)* / *Pose (RTMPose)*
        menus; True when it was used.

        The window owns *which checkpoint is in force* (the registry); this page owns *what
        is running* (the worker reads it). This is the hand-over between them - and it
        keeps the page's own thresholds, because those are the operator's settings, not the
        file's.
        """
        used = False
        if detector is not None:
            detector.score_threshold = float(self.spin_score.value())
            detector.iou_threshold = float(self.spin_iou.value())
            self.detector = detector
            # The timing average is **this** model's: a latency belongs to a checkpoint and
            # a device, and averaging two of them would be a number with unstated
            # conditions (`mlContext.md` law 6).
            self._latencies = []
            used = True
        if pose is not None:
            self.pose_model = pose
            self._pose_error = ""
            self._poses = []
            used = True
        if not used:
            return False
        self._report_detector()
        self._refresh_state()
        # A load is a decision the operator has just made - *use this model* - so the
        # surface **applies** it and says so, instead of waiting for the next source change.
        # Two things were wrong before this (`SR-PREDICT-017`):
        #
        # * a pass that was already running kept the model it was handed, so the boxes on
        #   screen came from the *previous* checkpoint while the header named the new one;
        # * with no frame yet the load left no trace at all on screen: the header still had
        #   no facts, and the only change was a line in the panel.
        applied = self._apply_loaded_models()
        what = " and ".join(name for name, model in (("detector", detector),
                                                     ("pose model", pose))
                           if model is not None)
        self.status_label.setText(
            f"{what.capitalize()} loaded - detecting with it now." if applied
            else f"{what.capitalize()} loaded. Choose a source in the Source menu.")
        # The models *and* the line above, now: `Status` is a header fact, and so is which
        # model answered - a fact published only when the next frame arrives is a fact the
        # operator does not have.
        self._refresh_header()
        return True

    def _apply_loaded_models(self) -> bool:
        """Run what was just loaded on what is on screen; True when it ran.

        Three cases, each of which says what it did:

        * a pass is **running** - it holds the model it was handed, exactly like the
          Keypoints toggle, so it is restarted on the **same source** (a model change is not
          new animals: the identities it has averaged survive). A video therefore starts
          again from its first frame; that is the honest cost of swapping a model inside a
          reader thread, and it is the same cost the Keypoints toggle pays;
        * a **still frame** is on screen - one more pass over the frame already in memory,
          the same call :meth:`open_image` makes, so the picture answers with the new model;
        * **no source** - nothing to run, and the readout plus the header now name the model
          in force, which is what makes a load visible before the first frame exists.
        """
        if self.worker is not None:
            self._start_worker(self.worker.source, path=self.worker.path)
            return True
        if self._frame is not None:
            detections, poses, elapsed = self._time_detection(self._frame)
            self._on_frame(self._frame, detections, elapsed, self._index, poses)
            return True
        return False

    # -- keypoints ---------------------------------------------------------
    def refresh_keypoint_tools(self):
        """Offer the **Keypoints** toggle only when a pose model is loaded.

        A tool that cannot act is noise, so it appears with the model and leaves if
        the model is ever unloaded - and the pose list is dropped with it, because
        keypoints from a model that is no longer in force must not stay on a frame.
        """
        available = self.pose_model is not None
        self.workspace.set_visible("keypoints", available)
        if not available:
            self._poses = []

    def _on_keypoints_toggled(self, checked):
        """The **Keypoints** toggle: run the pose pass, or stop running it.

        The pass costs real time and lives inside the worker, so a running source is
        restarted rather than poked: rebuilding the worker is clearer (and safer) than
        swapping a thread's model from the GUI thread. With nothing running, the
        frame on screen is simply redrawn with or without its lines.
        """
        self._pose_enabled = bool(checked)
        if self.worker is not None:
            source, path = self.worker.source, self.worker.path
            self._start_worker(source, path=path)
            return
        if self._frame is not None:
            self._draw_poses()
            self._refresh_header()

    # -- the scale ---------------------------------------------------------
    def _scale_summary(self, profile) -> str:
        """``um/px`` and the calibration's own uncertainty, for a combo entry.

        Short *on purpose*: a ``QComboBox`` sizes itself to its **longest item**, so a
        sentence in here is a sentence that decides the width of the whole controls panel
        (653 px, measured, for the entry that used to spell out the profile's id and its
        verification timestamp). The rest of the story travels in the item's tooltip
        (:meth:`_scale_tooltip`).
        """
        spread = (1.0 / profile.px_per_um) if profile.px_per_um > 0 else 0.0
        return f"{spread:.2f} um/px +/- {profile.uncertainty_pct:.2f}%"

    @staticmethod
    def _scale_tooltip(profile, active=False) -> str:
        """Everything the combo entry no longer spells out, one hover away."""
        return (f"{'active calibration' if active else 'calibration'}: "
                f"{profile.profile_id}\n"
                f"{profile.px_per_um * 1000.0:.1f} px/mm "
                f"+/- {profile.uncertainty_pct:.2f}%\n"
                f"verified {profile.calibrated_at or 'unknown'}")

    def refresh_scales(self):
        """Read the calibrated profiles into the combo, keeping the choice if it lasts.

        Re-read on demand rather than once at startup: a calibration is usually
        verified in Optics *while* this window is open, and a list frozen at launch
        would silently keep measuring under the old scale. When nothing is calibrated
        yet, the manual entry is the way to a number - and the label says that it is
        not a verified calibration.
        """
        previous = self.combo_scale.currentData()
        self._scale_error = ""
        rows, active = [], None
        try:
            with connection() as conn:
                rows = list(queries.list_profiles(conn))
                active = repository.active_profile(conn)
        except Exception as error:                 # noqa: BLE001 - db without tables
            # Named, never silent: a calibration list that cannot be read is exactly
            # the situation in which a millimetre gets invented.
            self._scale_error = f"{type(error).__name__}: {error}"

        entries = [("No scale (pixels only)", ("none", None), "")]
        if active is not None:
            entries.append((f"Active: {self._scale_summary(active)}",
                            ("profile", active.profile_id),
                            self._scale_tooltip(active)))
        for row in rows:
            if active is not None and row["profile_id"] == active.profile_id:
                continue
            entries.append((f"{row['px_per_um'] * 1000.0:.1f} px/mm  "
                            f"{short_id(row['profile_id'])}",
                            ("profile", row["profile_id"]),
                            f"{row['profile_id']}\n"
                            f"{row['px_per_um'] * 1000.0:.1f} px/mm"))
        entries.append(("Manual um/px", ("manual", None), ""))

        keys = [key for _label, key, _tip in entries]
        wanted = previous if previous in keys else None
        if wanted is None:
            # First read: use the app's active calibration. It is the scale the rest
            # of the application measures under, and this surface disagreeing with the
            # labeller about the scale is how two numbers stop matching.
            wanted = (("profile", active.profile_id) if active is not None
                      else ("none", None))

        self.combo_scale.blockSignals(True)
        self.combo_scale.clear()
        for index, (label, key, tip) in enumerate(entries):
            self.combo_scale.addItem(label, key)
            self.combo_scale.setItemData(index, tip, Qt.ToolTipRole)
        self.combo_scale.setCurrentIndex(keys.index(wanted))
        self.combo_scale.blockSignals(False)
        self._apply_scale()

    def _apply_scale(self, *_):
        """Resolve the combo's choice into µm per pixel, then re-label what is shown.

        **No re-inference**: the scale converts a pixel length that is already
        measured, so changing it re-labels the frame in place. That is what makes
        "try the other calibration and look again" a one-second experiment in the lab
        instead of a re-run.
        """
        mode, profile_id = self.combo_scale.currentData() or ("none", None)
        self.scale = None
        if mode == "profile" and profile_id:
            try:
                with connection() as conn:
                    self.scale = repository.profile_by_id(conn, profile_id)
            except Exception:                      # noqa: BLE001 - see above
                self.scale = None
        if self.scale is not None and self.scale.px_per_um > 0:
            self.um_per_px = 1.0 / self.scale.px_per_um
        elif mode == "manual":
            self.um_per_px = float(self.spin_um_per_px.value()) or None
        else:
            self.um_per_px = None
        self.spin_um_per_px.setEnabled(mode == "manual")
        self._update_scale_label()
        if self._frame is not None:
            self._draw_poses()
            self._render(self._index)

    def _update_scale_label(self):
        """Say which scale is in force, how good it is, and whether to trust it.

        The warnings are the point of this label: a scale that is **stale** (the
        adapter was bumped since it was verified) or **weak** (a short reference span
        made it uncertain) still produces a number - and a number that does not carry
        that is the failure this surface must not have.
        """
        if self._scale_error:
            self.scale_label.setText(
                f"Calibrations could not be read ({self._scale_error}).  The manual "
                "um/px below still works.")
            return
        if not self.um_per_px:
            self.scale_label.setText(
                "No scale - lengths are shown in pixels.  Calibrate in Optics, then "
                "Re-read calibrations.")
            return
        if self.scale is None:
            self.scale_label.setText(
                f"Manual scale: {self.um_per_px:.4f} um/px "
                f"({1000.0 / self.um_per_px:.1f} px/mm) - NOT a verified calibration, "
                "so the number carries no scale uncertainty.")
            return
        text = (f"{self.scale.profile_id}:  {self.scale.px_per_um * 1000.0:.1f} px/mm "
                f"= {self.um_per_px:.2f} um/px   +/- "
                f"{self.scale.uncertainty_pct:.2f}%")
        notes = []
        if self.scale.uncertainty_pct >= settings.CALIBRATION_WEAK_UNCERTAINTY_PCT:
            notes.append("WEAK calibration")
        if is_stale(self.scale.calibrated_at, settings.CALIBRATION_MAX_AGE_HOURS):
            notes.append("STALE - verify the scale before trusting a length")
        if notes:
            text += "   " + "   ".join(notes)
        self.scale_label.setText(text)

    def _scaled(self, pixels):
        """``(value, unit)`` of a pixel length, in the scale in force.

        Pixels when no calibration is set, micrometres when one is - and never a
        micrometre without one, because an unscaled number dressed as SI is the one
        thing ``projectbrief.md`` forbids.
        """
        if not self.um_per_px:
            return float(pixels), "px"
        return float(pixels) * self.um_per_px, "um"

    def _length(self, pose):
        """``(value, unit)`` of a pose's head-to-tail distance, in the scale in force."""
        return self._scaled(pose.length_px())

    def _length_text(self, pose) -> str:
        """The label drawn above one animal: its length, in the unit in force."""
        value, unit = self._length(pose)
        return f"{value:.0f} {unit}" if unit == "um" else f"{value:.1f} {unit}"

    def _choose_pose_model(self):
        """Ask for a pose checkpoint, then load it (dialog-free loading half)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a pose model checkpoint", str(paths.MODELS_DIR),
            CHECKPOINT_FILTER)
        if path:
            self.load_pose(path)
            self._report_detector()
            self._refresh_header()

    # -- tracking and averages ---------------------------------------------
    def _mode_label(self) -> str:
        """The current label mode's own name, for a status line."""
        for key, label in LABEL_MODES:
            if key == self._label_mode:
                return label
        return self._label_mode

    def _on_label_mode_changed(self, index):
        """The labels mode changed: redraw what is on screen and re-run nothing.

        Switching between "this frame" and "the average" is a change of *display*: the
        tracker has been folding every frame in already, so the average is there the
        moment it is asked for.
        """
        self._label_mode = self.combo_labels.itemData(index) or DEFAULT_LABEL_MODE
        self._draw_poses()
        self._refresh_header()

    def reset_tracks(self):
        """Forget every identity and start the averages again.

        The button an operator presses after changing the container, the plate or the
        subject. Without it an average would silently span two different populations -
        the kind of number that looks fine and means nothing.
        """
        self.tracker.reset()
        self._tracks = []
        self._draw_poses()
        self._render(self._index)

    def _update_tracks(self, poses):
        """Fold this frame's poses into the tracker; returns the tracks seen now."""
        if not poses:
            self._tracks = []
            return []
        self._tracks = self.tracker.update([
            Measurement(box=pose.box, length_px=pose.length_px(), payload=pose)
            for pose in poses])
        return self._tracks

    def _track_points(self, track):
        """The keypoints of a track's **last** measurement, in the pose's name order."""
        pose = getattr(track.last, "payload", None)
        if pose is None:
            return []
        return [pose.points[name] for name in pose.names if name in pose.points]

    def _track_label(self, track) -> str:
        """A track's average label: the mean, the spread of that mean, and its ``n``.

        With a single repeat there is **no measured spread**, so none is shown - the
        label is then the frame's own value, which *is* the mean of one sample. The
        ``+/-`` is the standard error of the mean over repeats: how well *this
        animal's* length is pinned down by the frames it has been seen in. It is **not**
        the model's error against an expert, and **not** the calibration's uncertainty -
        those are separate numbers, and neither is invented here.
        """
        mean, unit = self._scaled(track.mean_px)
        if track.count < 2:
            return f"{mean:.1f} {unit}" if unit == "px" else f"{mean:.0f} {unit}"
        spread = track.stderr_px * (self.um_per_px or 1.0)
        if unit == "px":
            return f"{mean:.1f} +/-{spread:.1f} px (n={track.count})"
        return f"{mean:.0f} +/-{spread:.0f} um (n={track.count})"

    def _overlay_items(self):
        """``[(index, segments, text)]`` for what is on screen, in the mode in force.

        In **average** mode each item is a **track**: the index is the track's identity,
        so an animal keeps its colour *and* its number from frame to frame, and the text
        is the mean of its repeats. In **per frame** mode the index is the frame's order
        and the text is this frame's prediction. In **hidden** mode the lines and dots
        stay and only the numbers go.

        Before the first tracked frame in average mode the frame's own value is used - it
        is the mean of one sample, so the label is not wrong, it is simply not yet an
        average of anything.
        """
        if self._label_mode == "average" and self._tracks:
            items = []
            for track in self._tracks:
                points = self._track_points(track)
                if len(points) < 2:
                    continue
                items.append((track.track_id, [(points[0], points[1])],
                              self._track_label(track)))
            return items
        items = []
        for rank, pose in enumerate(self._poses, start=1):
            points = [pose.points[name] for name in pose.names if name in pose.points]
            if len(points) < 2:
                continue
            text = "" if self._label_mode == "off" else self._length_text(pose)
            items.append((rank, [(points[0], points[1])], text))
        return items

    def _update_average_label(self):
        """Say how many identities are alive, how many are on this frame, and the n's."""
        tracks = self.tracker.tracks()
        if not tracks:
            self.average_label.setText(
                "No animal tracked yet.  An identity and its average start with the "
                "first frame that shows an animal.")
            return
        self.average_label.setText(
            f"{len(tracks)} animal(s) tracked   {len(self._tracks)} on this frame   "
            f"repeats: up to n={max(track.count for track in tracks)}   "
            f"showing: {self._mode_label()}")

    def _refresh_source(self):
        """Describe where the frames come from - and what the camera is doing."""
        self.refresh_frame_tools()
        if self.worker is None:
            self.source_label.setText(
                f"Live feed: {self.service.state_label()}.  Open a video or an "
                "image instead if the camera is not up.")
            return
        if self.worker.source == InferenceWorker.VIDEO:
            self.source_label.setText(f"File: {self.worker.path.name}")
        else:
            self.source_label.setText(
                f"Live feed: {self.service.state_label()}")

    # -- stepping through a video file -------------------------------------
    def frame_stepping(self) -> bool:
        """True while a **video file** is the source, so frames can be stepped.

        A live feed always moves on by itself and a single image has no neighbours:
        neither has a *next frame* to ask for, so neither is offered one.
        """
        return (self.worker is not None
                and self.worker.source == InferenceWorker.VIDEO)

    def refresh_frame_tools(self):
        """Offer frame stepping only where it means something.

        The two tools and the two arrow-key shortcuts appear while a video file is
        open and leave again with it, so a live feed or a single image never shows a
        control that cannot act - and the arrow keys keep panning the canvas there,
        which is what they mean on every other screen.
        """
        video = self.frame_stepping()
        for key in ("previous_frame", "next_frame"):
            self.workspace.set_visible(key, video)
        self.shortcut_previous_frame.setEnabled(video)
        self.shortcut_next_frame.setEnabled(video)

    def next_frame(self):
        """The *Frame >* menu item and the Right arrow: detect on the next frame."""
        self._step_by(+1)

    def previous_frame(self):
        """The *< Frame* tool and the Left arrow: detect on the previous frame."""
        self._step_by(-1)

    def _step_by(self, direction):
        """Move one frame through a video file, in either direction.

        Pauses first: stepping while playing would race the loop, and the operator who
        steps is looking at one frame. Going back needs a **seek**, because the reader
        sits on the frame *after* the one on screen - so one frame back means seeking
        to the frame before it (and the step is what makes the paused loop read it).
        """
        if not self.frame_stepping():
            self.status_label.setText(
                "Frame stepping needs a video file - open one, or let the live feed "
                "run on its own.")
            self._refresh_header()
            return
        self.worker.set_paused(True)
        self.btn_play.setText("Resume")
        if direction < 0:
            self.worker.seek(max(0, self._index - 2))
        self.worker.request_step()

    # -- buttons -----------------------------------------------------------
    def _on_button(self):
        """One handler for the panel's buttons, so each action stays a named method.

        The model loaders and the source choices are **menu items in the window header**
        now (`X-N13`): what is left in the panel is the video transport and the two
        re-read actions.
        """
        sender = self.sender().objectName()
        {
            "btn_play": self.toggle_play,
                "btn_stop": self.stop,
            "btn_refresh_scale": self.refresh_scales,
            "btn_reset_tracks": self.reset_tracks,
        }[sender]()

    def _choose_checkpoint(self):
        """Ask for a checkpoint file, then load it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a detector checkpoint", str(paths.MODELS_DIR),
            CHECKPOINT_FILTER)
        if path:
            self.load_detector(path)

    def _choose_video(self):
        """Ask for a video file, then play it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a video to detect on", str(paths.RECORDINGS_DIR),
            VIDEO_FILTER)
        if path:
            self.open_video(path)

    def _choose_image(self):
        """Ask for one image, then detect on it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open an image to detect on", str(paths.DATASETS_DIR),
            IMAGE_FILTER)
        if path:
            self.open_image(path)

    # -- sources -----------------------------------------------------------
    def use_live_feed(self):
        """Detect on the shared camera connection, from now on."""
        if not self._require_detector():
            return
        if self.service.current_frame() is None:
            self.status_label.setText(
                f"No live frame.  {self.service.state_label()}  Connect in the bar "
                "below, or open a video file instead.")
        self._start_worker(InferenceWorker.LIVE)
        self.btn_play.setEnabled(False)
        self.slider_position.setEnabled(False)

    def open_video(self, path):
        """Play a video file and detect on every frame; dialog-free on purpose."""
        if not self._require_detector():
            return
        self._start_worker(InferenceWorker.VIDEO, path=Path(path))
        self.btn_play.setEnabled(True)
        self.btn_play.setText("Pause")
        self.slider_position.setEnabled(True)

    def open_image(self, path):
        """Detect on exactly one image from disk, in one pass.

        Synchronous where the video and the live feed are not: it is a single
        frame whose file was already read to build the preview, so there is no
        queue to keep up with - and the click has already waited for the file.
        """
        if not self._require_detector():
            return
        self.stop()
        # One entry point for a source: the canvas reads the file (any format the
        # machine can read) and refuses it by name when it cannot.
        frame = self.canvas.set_source(path)
        if frame is None:
            # Nothing on the canvas, so no number may be left standing: the previous
            # source's detections, poses and boxes are not this source's.
            self._frame = None
            self._detections = []
            self._poses = []
            self.canvas.set_boxes([])
            self.canvas.set_annotations([])
            self._render(0)
            return
        self._viewed = False                      # a new source: fit it once
        detections, poses, elapsed = self._time_detection(frame)
        self.source_label.setText(f"Image: {Path(path).name}")
        # The *Status* fact follows the surface: it used to keep saying "choose a source"
        # while a frame was on screen and detected.
        self.status_label.setText(f"Detecting on {Path(path).name}.")
        self.btn_play.setEnabled(False)
        self.slider_position.setEnabled(False)
        self._on_frame(frame, detections, elapsed, 0, poses)

    def _require_detector(self) -> bool:
        """True when a detector is loaded; otherwise load the one **in force**, or say how.

        The registry decides *which* checkpoint (`daphnia.models`, written by the *Detector
        (RTMDet)* / *Pose (RTMPose)* menus): this used to reach for the page's own default file, so a source could produce
        boxes from a checkpoint the bar did not name and the registry did not hold - two
        places deciding "which model" is how a surface starts lying about its numbers. With
        nothing registered, the operator is pointed at the one place that loads one
        (`SR-PREDICT-008`, law 8: no silent fallback).
        """
        if self.detector is not None:
            return True
        with connection() as conn:
            active = models.active(conn, "detector")
        if active is None:
            self.status_label.setText(
                "No detector is registered yet.  Load one from the *Detector (RTMDet)* "
                "menu - 'Load the active detector', or a checkpoint from the list below it.")
            self._refresh_header()
            return False
        self.load_detector(active.path)      # registers nothing: the menu owns the registry
        return self.detector is not None

    def _time_detection(self, frame):
        """Detect - and pose, if enabled - on one frame, timed, in the GUI thread.

        Only ever one frame at a time (a single image from disk that was already read
        to build the preview), so there is no queue to keep up with.
        """
        started = time.perf_counter()
        detections = self.detector.detect(frame)
        poses = (self.pose_model.detect_frame(frame, detections=detections)
                 if self.pose_model is not None and self._pose_enabled else [])
        return detections, poses, (time.perf_counter() - started) * 1000.0

    # -- the worker --------------------------------------------------------
    def _same_source(self, source, path) -> bool:
        """True when ``source``/``path`` is the source already running.

        The distinction that keeps an average honest: a **different** source is different
        animals and must reset the identities, while a *restart of the same source* (the
        Keypoints toggle, a resume) must not throw away what has been averaged.
        """
        if self.worker is None:
            return False
        wanted = (source, Path(path) if path else None)
        return (self.worker.source, self.worker.path) == wanted

    def _start_worker(self, source, path=None):
        """Replace any running worker with one for ``source``."""
        same_source = self._same_source(source, path)
        self.stop()
        self._viewed = False
        self._latencies = []
        worker = InferenceWorker(self.detector, source=source,
                                 service=self.service, path=path, parent=self,
                                 pose_model=(self.pose_model
                                             if self._pose_enabled else None))
        worker.frame_ready.connect(self._on_frame)
        worker.status.connect(self.status_label.setText)
        worker.progress.connect(self._on_progress)
        worker.message.connect(self._on_worker_message)
        self.worker = worker
        worker.start()
        if not same_source:
            self.reset_tracks()
        self.btn_stop.setEnabled(True)
        self.status_label.setText(
            "Watching the live feed." if source == InferenceWorker.LIVE
            else f"Playing {Path(path).name}.")
        # The source changed, so what is offered changed with it: the name shown and
        # the frame-stepping tools follow (`refresh_frame_tools`).
        self._refresh_source()

    def stop(self):
        """End the current run and leave the last frame on screen."""
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.stop()
            worker.deleteLater()
        self.btn_stop.setEnabled(False)
        self.btn_play.setEnabled(False)
        self._refresh_source()

    def toggle_play(self):
        """Pause or resume the video file."""
        if self.worker is None or self.worker.source != InferenceWorker.VIDEO:
            return
        paused = self.btn_play.text() == "Pause"
        self.worker.set_paused(paused)
        self.btn_play.setText("Resume" if paused else "Pause")

    def _on_seek(self, value):
        """The slider moved: jump the video there."""
        if self.worker is not None:
            self.worker.seek(value)

    def _on_progress(self, index, total):
        """The video position changed: show it without fighting the slider."""
        if total <= 0:
            return
        self.slider_position.blockSignals(True)
        self.slider_position.setMaximum(max(0, total - 1))
        self.slider_position.setValue(max(0, index - 1))
        self.slider_position.blockSignals(False)

    def _on_worker_message(self, message):
        """A worker failure - shown, never swallowed."""
        self.status_label.setText(message)
        self._refresh_header()

    # -- frames ------------------------------------------------------------
    def _on_frame(self, frame, detections, milliseconds, index, poses=None):
        """Show one detected frame: the picture, the boxes, the keypoints, the readout."""
        self._frame = frame
        self._index = int(index)      # which frame of a video is on screen (1-based)
        self._detections = list(detections or [])
        self._poses = list(poses or [])
        self._update_tracks(self._poses)
        self._latencies.append(float(milliseconds))
        self._latencies = self._latencies[-30:]

        # Fit once per source, then keep the operator's zoom: re-fitting a live
        # view many times a second would throw away the magnification.
        self.canvas.set_source(frame, reset_view=not self._viewed)
        self._viewed = True
        self.canvas.set_boxes([
            (rank, (detection.box[0], detection.box[1],
                    detection.box[2], detection.box[3]))
            for rank, detection in enumerate(self._detections, start=1)])
        self._draw_poses()
        self._render(index)

    def _draw_poses(self):
        """Draw the overlay: one line, two dots and one label per animal.

        What the label *is* depends on the mode (:meth:`_overlay_items`): this frame's
        prediction, or the mean of the animal's repeats. In **average** mode the colour
        and the number are the animal's **identity**, so they stop moving between frames
        - which is what makes "keep track of each one" visible rather than asserted.
        """
        self.canvas.set_annotations(self._overlay_items())
        self._update_average_label()

    def _pose_for(self, detection):
        """The pose of one detection, matched by its **box**, never by list position.

        The two lists are not parallel: the pose pass skips a box whose crop misses the
        frame, so pairing by position would quietly show one animal's keypoints beside
        another animal's row.
        """
        box = tuple(float(value) for value in detection.box)
        for pose in self._poses:
            if pose.box == box:
                return pose
        return None

    def _display_length(self, pose):
        """The length shown for one animal, in the mode in force.

        In **average** mode a track's mean, matched by *object identity* of the pose it
        last saw - so the list and the label on the animal can never disagree, which on
        one screen showing two numbers for one fish would be indefensible. Otherwise the
        frame's own value.
        """
        track = (self._track_for_pose(pose) if self._label_mode == "average" else None)
        return self._scaled(track.mean_px) if track is not None else self._length(pose)

    def _track_for_pose(self, pose):
        """The track whose last measurement is this pose, or ``None``."""
        for track in self._tracks:
            if track.last.payload is pose:
                return track
        return None

    def _render(self, index=0):
        """Write the count, the timing, the measured length and the detection list.

        The panel *is* the readout now (2026-10-18): the median length line that used to
        be a card over the picture goes here, under the count it belongs to - so a fact
        that cannot be seen on the image is still a fact the operator can read, and the
        picture stays the picture.
        """
        count = len(self._detections)
        average = (sum(self._latencies) / len(self._latencies)
                   if self._latencies else 0.0)
        height, width = (self._frame.shape[:2] if self._frame is not None
                         else (0, 0))
        position = f"  frame {index}" if index else ""
        summary = (f"{count} animal(s) found{position}   {width} x {height} px   "
                   f"{average:.0f} ms/frame ({1000.0 / average:.1f} fps)"
                   if average > 0 else f"{count} animal(s) found{position}")
        pose = self._pose_summary()
        if pose:
            summary = f"{summary}\n{pose}"
        self.summary_label.setText(summary)

        # The list reads like the Bench's *This view*: **who** in the first column and
        # **what was found** in the second, so the two surfaces are read the same way and
        # the number this one exists to show is not buried in a sentence.
        self.list_detections.clear()
        for index, detection in enumerate(self._detections, start=1):
            answer = detection.describe()
            pose = self._pose_for(detection)
            if pose is not None and len(pose.points) >= 2:
                # The measured length **leads** the answer - it is the number the
                # surface exists to show.
                value, unit = self._display_length(pose)
                answer = f"{value:.0f} {unit}   " + answer
            self.list_detections.addTopLevelItem(
                QTreeWidgetItem([f"Animal {index}", answer]))
        if not self._detections and self._frame is not None:
            self.list_detections.addTopLevelItem(QTreeWidgetItem([
                "No animal",
                "Nothing above the score threshold.  Lower it to see the marginal "
                "guesses, or check the detector is the right one."]))
        self._refresh_header()

    def _pose_summary(self) -> str:
        """One line about the animals on screen: how many, and the length measured.

        The unit follows the scale in force and the two are never mixed: with no
        calibration the number is in **pixels** and says *unscaled*; with one it is in
        micrometres and names the calibration. The **value** follows the mode - the
        frame's own predictions, or in average mode each tracked animal's mean, with the
        largest ``n`` behind it stated.

        The spread quoted is the **calibration's own**: it is *not* the uncertainty of
        the measurement, which the model contributes to separately
        (`scripts/train_rtmpose.py --evaluate`), and saying so is the point of the
        wording.
        """
        averaging = self._label_mode == "average" and bool(self._tracks)
        if averaging:
            values = [self._scaled(track.mean_px)[0] for track in self._tracks]
            repeats = max(track.count for track in self._tracks)
        else:
            values = [self._length(pose)[0] for pose in self._poses
                      if len(pose.points) >= 2]
            repeats = 0
        widths = sorted(values)
        if not widths:
            return ""
        count = len(widths)
        median = (widths[count // 2] if count % 2 else
                  0.5 * (widths[count // 2 - 1] + widths[count // 2]))
        if not self.um_per_px:
            text = (f"{count} animal(s)   median length {median:.1f} px "
                    "(unscaled - no calibration)")
        else:
            text = f"{count} animal(s)   median length {median:.0f} um"
            if self.scale is not None:
                text += (f"   (scale +/- {self.scale.uncertainty_pct:.2f}% only - the "
                         "model's own error is a separate number)")
        if averaging:
            text += f"   [mean over repeats, up to n={repeats}]"
        return text

    def _refresh_header(self):
        """Publish the surface's facts in the window **header**.

        Which model answered and the last message: the context of what is on screen.
        The **numbers** - how many animals, the length measured, what a frame costs -
        used to be a strip of cards over the picture; that strip is gone (2026-10-18)
        and they are read in the panel, which is where the operator works:
        ``summary_label`` (the count, the timing, the median length),
        ``list_detections`` (each animal with its own number) and ``detector_label``
        (which checkpoint, on which device).
        """
        self.workspace.set_header(self._header_facts())

    def _header_facts(self) -> list:
        """The window facts: which model answered, and the last message.

        The **source** is not one of them any more: *which frames you are looking at* - and
        whether the camera is even connected - is the header's *Source* menu (`X-N13`),
        where the answer is also the control. A menu entry that says "not connected" and
        cannot connect you is a sentence, not a fact.
        """
        return [
            Kpi("model", self._models_summary(), label="Model"),
            Kpi("status", self.status_label.text(), label="Status"),
        ]

    def _models_summary(self) -> str:
        """Which models are loaded, in one line - the header's ``Model`` fact.

        The panel's ``detector_label`` carries the whole story (each checkpoint, the torch
        build, the device, the pose model's own metrics); a header fact is a *bar*, so it
        names the models and stops. The values are elided there, never wrapped.
        """
        if self.detector is None and self.pose_model is None:
            return "none loaded - press Load detector"
        parts = []
        if self.detector is not None:
            parts.append(self.detector.describe())
        else:
            parts.append(f"detector not loaded ({checkpoint_path().name})")
        if self.pose_model is not None:
            parts.append("pose: " + self.pose_model.describe())
        else:
            parts.append(f"pose not loaded ({pose_checkpoint_path().name})")
        return "   ".join(parts)

    # -- thresholds --------------------------------------------------------
    def _on_score_changed(self, value):
        """The score threshold changed: apply it to the next frame at once."""
        if self.detector is not None:
            self.detector.score_threshold = float(value)

    def _on_iou_changed(self, value):
        """The NMS threshold changed: apply it to the next frame at once."""
        if self.detector is not None:
            self.detector.iou_threshold = float(value)

    # -- lifetime ----------------------------------------------------------
    def stop_detection(self):
        """The stop button wired to :meth:`stop` (kept for the window's close)."""
        self.stop()

    def detections(self) -> list:
        """The detections of the frame on screen (for a test or a script)."""
        return list(self._detections)

    def poses(self) -> list:
        """The keypoints of the frame on screen (for a test or a script)."""
        return list(self._poses)

    def latency_ms(self) -> float:
        """The mean inference time over the last few frames, in milliseconds."""
        return (sum(self._latencies) / len(self._latencies)
                if self._latencies else 0.0)
