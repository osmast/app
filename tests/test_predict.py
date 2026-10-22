"""The Predict surface: it works without a model, and says why when it cannot."""

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea, QVBoxLayout, QWidget

from daphnia import surfaces
from daphnia.db import connection, init_db, repository
from daphnia.ui.hub import PredictWindow, window_for
from daphnia.ui.pages.predict import IDLE_SLEEP_MS, InferenceWorker, Predict
from daphnia.ui.shell import CONTROLS_WIDTH
from daphnia.vision.rtmdet import torch_available
from daphnia.vision.rtmdet.detect import (
    CHECKPOINT_ENV,
    Detection,
    ModelUnavailable,
    checkpoint_path,
    load_detector,
)
from daphnia.vision.rtmpose.detect import (
    CHECKPOINT_ENV as POSE_CHECKPOINT_ENV,
    PosePrediction,
)

TORCH = torch_available()


class PredictPageTestCase(unittest.TestCase):
    """One QApplication and a clean environment, shared by the page tests."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        # Point both surfaces at checkpoints that do not exist, so no test can
        # accidentally load a real model (and so the "no model" path is exercised).
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._previous = os.environ.get(CHECKPOINT_ENV)
        self._previous_pose = os.environ.get(POSE_CHECKPOINT_ENV)
        self._previous_db = os.environ.get("DAPHNIA_DB")
        os.environ[CHECKPOINT_ENV] = str(Path(self._tmp.name) / "absent.pt")
        os.environ[POSE_CHECKPOINT_ENV] = str(Path(self._tmp.name) / "absent_pose.pt")
        # A **private, empty** database: the scale this surface measures under must be
        # the one the test put there, not whatever calibration the live zoo.db
        # happens to carry - otherwise a test would pass or fail with the machine.
        self.db_path = Path(self._tmp.name) / "predict.db"
        os.environ["DAPHNIA_DB"] = str(self.db_path)
        init_db(self.db_path)
        self.addCleanup(self._restore)

    def _restore(self):
        for name, value in ((CHECKPOINT_ENV, self._previous),
                            (POSE_CHECKPOINT_ENV, self._previous_pose),
                            ("DAPHNIA_DB", self._previous_db)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def activate_profile(self, profile_id="PROF_P", px_per_um=0.013363,
                         uncertainty_pct=0.5, verified_ago_hours=None):
        """Put a calibration in this test's database and make it the active one.

        ``verified_ago_hours`` back-dates ``last_verified_at``, which is how a **stale**
        calibration is produced - the case the surface must warn about rather than
        quietly measuring under it.
        """
        with connection() as conn:
            repository.upsert_profile(
                conn, profile_id=profile_id, profile_name=profile_id,
                microscope_model="m", objective_magnification="2x",
                camera_device="c", px_per_um=px_per_um,
                uncertainty_pct=uncertainty_pct)
            repository.activate_profile(conn, profile_id)
            if verified_ago_hours:
                conn.execute(
                    "UPDATE calibration_profiles SET last_verified_at ="
                    " datetime('now', ?) WHERE profile_id = ?",
                    (f"-{float(verified_ago_hours):g} hours", profile_id))
        with connection() as conn:
            return repository.profile_by_id(conn, profile_id)

    def _page(self):
        page = Predict()
        self.addCleanup(page.stop)
        return page


class NoModelTests(PredictPageTestCase):
    """With no checkpoint the surface is explicit - never silently empty."""

    def test_it_names_the_checkpoint_it_wanted(self):
        self.assertIn("No detector at", self._page().detector_label.text())

    def test_the_source_choices_are_always_offered(self):
        """They are menu items in the *Source* menu now (`X-N13`).

        A model is not a precondition for *looking*: opening a video or an image before a
        detector is loaded shows the frames, just no boxes. The old panel buttons waited
        for a detector (a dead control, which is what `X-N10` is about).
        """
        page = self._page()
        self.assertFalse(hasattr(page, "btn_live"))
        self.assertFalse(hasattr(page, "btn_open_video"))
        self.assertFalse(hasattr(page, "btn_open_image"))
        for key in ("live", "open_video", "open_image"):
            action = page.workspace.action(key)
            self.assertIsNotNone(action, key)
            self.assertTrue(action.isEnabled(), key)

    def test_loading_a_missing_checkpoint_fails_with_a_reason(self):
        page = self._page()
        self.assertFalse(page.load_detector())
        self.assertIsNone(page.detector)
        self.assertIn("No trained detector", page.detector_label.text())

    def test_the_page_exposes_the_controls_the_requirements_name(self):
        page = self._page()
        for name in ("spin_score", "spin_iou", "btn_play", "btn_stop",
                     "slider_position", "list_detections", "summary_label",
                     "status_label", "detector_label", "source_label",
                     "combo_scale", "combo_labels"):
            self.assertTrue(hasattr(page, name), name)
        # The model loaders and the source choices are header menu items now (`X-N13`).
        for key in ("live", "open_video", "open_image", "previous_frame", "next_frame",
                    "keypoints"):
            self.assertIsNotNone(page.workspace.action(key), key)

    def test_stopping_without_a_worker_is_harmless(self):
        page = self._page()
        page.stop()
        page.stop()
        self.assertIsNone(page.worker)

    def test_it_describes_where_the_frames_come_from(self):
        page = self._page()
        self.assertEqual(page.detections(), [])
        self.assertEqual(page.latency_ms(), 0.0)
        self.assertIn("Live feed", page.source_label.text())
        self.assertIn("video or an image", page.source_label.text())


class DetectionTests(unittest.TestCase):
    """The prediction record: what a caller sees of one animal."""

    def test_it_reports_its_size_and_centre(self):
        detection = Detection(box=(10.0, 20.0, 110.0, 120.0), score=0.87,
                              label=0, name="daphnia")
        self.assertAlmostEqual(detection.width, 100.0)
        self.assertAlmostEqual(detection.height, 100.0)
        self.assertEqual(detection.centre(), (60.0, 70.0))

    def test_it_describes_itself_with_the_name_and_the_score(self):
        text = Detection(box=(10.0, 20.0, 110.0, 120.0), score=0.87, label=0,
                         name="daphnia").describe()
        self.assertIn("daphnia", text)
        self.assertIn("0.87", text)
        self.assertIn("100", text)


class CheckpointPathTests(PredictPageTestCase):
    """Where a checkpoint is looked for, in the order that matters."""

    def test_an_explicit_path_wins(self):
        wanted = Path(self._tmp.name) / "mine.pt"
        self.assertEqual(checkpoint_path(str(wanted)), wanted)

    def test_the_environment_override_is_honoured(self):
        self.assertEqual(checkpoint_path().name, "absent.pt")

    def test_the_default_is_the_models_tree(self):
        os.environ.pop(CHECKPOINT_ENV, None)
        self.assertEqual(checkpoint_path().name, "rtmdet_daphnia.pt")
        self.assertIn("models", str(checkpoint_path()))


class WorkerControlTests(PredictPageTestCase):
    """The inference worker's control surface, without running any inference loop."""

    class StubDetector:
        """A detector that always finds one animal, so the worker needs no model."""

        score_threshold = 0.5
        iou_threshold = 0.6

        def detect(self, frame):
            return [Detection(box=(1.0, 1.0, 9.0, 9.0), score=0.9, label=0,
                              name="daphnia")]

    def _worker(self, source=InferenceWorker.LIVE):
        worker = InferenceWorker(self.StubDetector(), source=source,
                                 service=None, path=None)
        self.addCleanup(worker.stop)
        return worker

    def test_a_new_worker_is_not_running_and_not_paused(self):
        worker = self._worker()
        self.assertFalse(worker.isRunning())
        self.assertFalse(worker._paused)

    def test_pausing_and_stepping_are_remembered(self):
        worker = self._worker(InferenceWorker.VIDEO)
        worker.set_paused(True)
        self.assertTrue(worker._paused)
        worker.request_step()
        self.assertTrue(worker._step)

    def test_a_seek_is_recorded_for_the_loop_to_pick_up(self):
        worker = self._worker(InferenceWorker.VIDEO)
        worker.seek(12)
        self.assertEqual(worker._seek_to, 12)

    def test_one_pass_is_timed_and_returns_the_detections(self):
        worker = self._worker()
        detections, milliseconds = worker._infer(np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertGreaterEqual(milliseconds, 0.0)

    def test_the_live_loop_carries_the_poses_of_its_own_frame(self):
        """Live, not only video: keypoints travel in the signal with their frame."""
        same = np.zeros((8, 8, 3), np.uint8)

        class Service:
            def current_frame(self):
                return same

        class StubPose:
            """Stands in for the pose model: one prediction per detection."""

            def detect_frame(self, frame, detections=None):
                return [pose_prediction(box=d.box) for d in (detections or [])]

        worker = InferenceWorker(self.StubDetector(), source=InferenceWorker.LIVE,
                                 service=Service(), path=None,
                                 pose_model=StubPose())
        self.addCleanup(worker.stop)
        seen = []
        worker.frame_ready.connect(lambda *args: seen.append(args))
        worker.start()
        deadline = 5.0
        started = time.monotonic()
        while (not seen and worker.isRunning()
               and time.monotonic() - started < deadline):
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(seen, "the live loop never emitted a frame")
        frame, detections, _elapsed, _index, poses = seen[-1]
        self.assertEqual(len(detections), 1)
        self.assertEqual(len(poses), 1, "the poses must ride with the frame")
        self.assertEqual(tuple(poses[0].points), ("head", "tail"))

    def test_the_live_loop_skips_a_frame_it_has_already_seen(self):
        """The same buffered frame must not be inferred twice."""
        calls = []

        class Counting(self.StubDetector):
            def detect(self, frame):
                calls.append(frame)
                return []

        same = np.zeros((8, 8, 3), np.uint8)

        class Service:
            def current_frame(self):
                return same

        worker = InferenceWorker(Counting(), source=InferenceWorker.LIVE,
                                 service=Service(), path=None)
        self.addCleanup(worker.stop)
        seen = []
        worker.frame_ready.connect(lambda *args: seen.append(args))
        worker.start()
        # The signal is delivered on the main thread, so the queue must be pumped.
        deadline = 5.0
        started = time.monotonic()
        while (not seen and worker.isRunning()
               and time.monotonic() - started < deadline):
            self.app.processEvents()
            worker.msleep(IDLE_SLEEP_MS)
        worker.stop()
        self.app.processEvents()
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(calls), 1)


class FrameSteppingTests(PredictPageTestCase):
    """Arrow keys step a **video file** - and only a video file.

    A live feed always runs on by itself and a single image has no neighbours, so
    neither gets a *next frame*: the on-canvas tools appear with a video and leave
    with it, and the arrow keys keep panning the canvas everywhere else.
    """

    class VideoWorker:
        """Stands in for the inference worker: records what it is asked to do."""

        source = InferenceWorker.VIDEO

        def __init__(self):
            self.paused = None
            self.steps = 0
            self.seeks = []
            self.stopped = False

        def set_paused(self, paused):
            self.paused = bool(paused)

        def request_step(self):
            self.steps += 1

        def seek(self, index):
            self.seeks.append(int(index))

        def stop(self):
            self.stopped = True

        def deleteLater(self):
            pass

    class LiveWorker(VideoWorker):
        """The same, for the live feed: there is no frame to step to."""

        source = InferenceWorker.LIVE

    def _video_page(self):
        page = self._page()
        page.worker = self.VideoWorker()
        page.refresh_frame_tools()
        return page

    def _visible_tools(self, page, group="Frames"):
        """The page's tools offered in one menu group, as the header would show them.

        The bar carries **menu items** (`X-N13`), so what matters is which actions the page
        makes visible in that group - a group with none behind it leaves the bar.
        """
        return [action.text()
                for title, actions in page.workspace.tool_group.groups()
                if title == group
                for action in actions if action.isVisible()]

    def test_the_tools_and_the_arrow_keys_need_a_video_file(self):
        page = self._page()                       # no source at all
        self.assertFalse(page.frame_stepping())
        for key in ("previous_frame", "next_frame"):
            self.assertFalse(page.workspace.action(key).isVisible(), key)
        self.assertFalse(page.shortcut_previous_frame.isEnabled())
        self.assertFalse(page.shortcut_next_frame.isEnabled())
        # Nothing floats over the picture: the page's own menu has nothing behind it.
        self.assertEqual(self._visible_tools(page), [])

    def test_a_video_file_offers_the_tools_and_the_arrow_shortcuts(self):
        page = self._video_page()
        self.assertTrue(page.frame_stepping())
        for key in ("previous_frame", "next_frame"):
            self.assertTrue(page.workspace.action(key).isVisible(), key)
        self.assertEqual(self._visible_tools(page), ["< Frame", "Frame >"])
        self.assertTrue(page.shortcut_previous_frame.isEnabled())
        self.assertTrue(page.shortcut_next_frame.isEnabled())
        self.assertEqual(page.shortcut_previous_frame.key(), QKeySequence(Qt.Key_Left))
        self.assertEqual(page.shortcut_next_frame.key(), QKeySequence(Qt.Key_Right))

    def test_the_live_feed_takes_the_tools_away_again(self):
        page = self._video_page()
        page.worker = self.LiveWorker()
        page.refresh_frame_tools()
        self.assertFalse(page.frame_stepping())
        self.assertFalse(page.workspace.action("next_frame").isVisible())
        self.assertFalse(page.shortcut_next_frame.isEnabled())

    def test_stopping_the_video_takes_the_tools_away(self):
        """The real path, not the refresh: ``stop()`` itself re-reads the source."""
        page = self._video_page()
        page.stop()
        self.assertIsNone(page.worker)
        self.assertFalse(page.frame_stepping())
        self.assertFalse(page.workspace.action("previous_frame")
                         .isVisible())
        self.assertFalse(page.shortcut_previous_frame.isEnabled())

    def test_the_arrows_step_one_frame_each_way_and_pause_the_video(self):
        page = self._video_page()
        worker = page.worker
        page._index = 5                     # the frame on screen (1-based, as emitted)

        page.next_frame()
        self.assertEqual(worker.steps, 1)
        self.assertEqual(worker.seeks, [])       # forward is just the next read

        page.previous_frame()
        self.assertEqual(worker.seeks, [3])      # back one: the frame before it
        self.assertEqual(worker.steps, 2)        # and a step is what reads it
        self.assertTrue(worker.paused)
        self.assertEqual(page.btn_play.text(), "Resume")

    def test_stepping_back_from_the_first_frame_stays_at_the_first(self):
        page = self._video_page()
        page._index = 1
        page.previous_frame()
        self.assertEqual(page.worker.seeks, [0])

    def test_stepping_without_a_video_says_what_is_missing(self):
        page = self._page()
        page.next_frame()
        self.assertIn("video file", page.status_label.text())

    def test_the_frame_menu_steps_forward_too(self):
        """The *Frames* menu's *Frame >* steps one frame.

        The panel's *Step* row was the same action wired twice (`X-N13`): one action, one
        home - and the arrow key stays the second way to it.
        """
        page = self._video_page()
        page.workspace.action("next_frame").trigger()
        self.assertEqual(page.worker.steps, 1)
        self.assertEqual(page.worker.seeks, [])


class ArrowKeyTests(PredictPageTestCase):
    """The arrow keys change meaning with the source - the shortcut, for real."""

    def _shown(self, worker):
        """A shown window holding the page, so a window shortcut can really fire."""
        page = self._page()
        page.worker = worker
        page.refresh_frame_tools()
        page.canvas.set_source(np.zeros((40, 60, 3), np.uint8))

        window = QWidget()
        layout = QVBoxLayout(window)
        layout.addWidget(page)
        # A top-level widget is owned by Python: without a reference it is collected
        # (and its child, the page, with it) as soon as this method returns.
        self._window = window
        self.addCleanup(window.close)
        window.show()
        window.activateWindow()
        self.app.processEvents()
        page.canvas.setFocus()
        page.canvas.set_zoom(3.0)           # magnified, so panning moves something
        return page

    def test_the_arrows_step_the_frame_instead_of_panning_the_canvas(self):
        page = self._shown(FrameSteppingTests.VideoWorker())
        pan = page.canvas._pan.x()

        QTest.keyClick(page.canvas, Qt.Key_Right)
        self.assertEqual(page.worker.steps, 1, "the Right arrow must step the frame")
        self.assertEqual(page.canvas._pan.x(), pan, "and must not pan the canvas")

        QTest.keyClick(page.canvas, Qt.Key_Left)
        self.assertEqual(page.worker.seeks, [0], "Left steps one frame back")

    def test_the_arrows_still_pan_when_no_video_file_is_open(self):
        page = self._shown(FrameSteppingTests.LiveWorker())
        pan = page.canvas._pan.x()

        QTest.keyClick(page.canvas, Qt.Key_Right)
        self.assertEqual(page.worker.steps, 0, "a live feed has no frame to step to")
        self.assertEqual(page.canvas._pan.x(), pan + page.canvas.PAN_STEP)


def write_tiny_checkpoint(folder) -> Path:
    """Write a real (untrained) RTMDet checkpoint into ``folder``.

    Real weights, one step of nothing: enough to prove that loading, building and a
    forward pass work without needing a training run in the suite.
    """
    from daphnia.vision.rtmdet import CLASSES
    from daphnia.vision.rtmdet.model import build_model
    from daphnia.vision.rtmdet.train import (
        TrainConfig,
        checkpoint_payload,
        save_checkpoint,
    )

    path = Path(folder) / "tiny.pt"
    save_checkpoint(path, checkpoint_payload(
        build_model("tiny", len(CLASSES)), TrainConfig(input_size=128),
        None, {"ap50": 0.25}, {"boxes": 4}))
    return path


class PredictWindowTests(PredictPageTestCase):
    """The launch point: the window exists, and closing it stops the thread."""

    def test_the_predict_surface_is_available_and_launches(self):
        self.assertTrue(surfaces.get("predict").is_available)
        self.assertEqual(surfaces.get("predict").launch_point(),
                         "python run.py predict")

    def test_the_launch_key_opens_the_predict_window(self):
        window = window_for("predict")
        self.assertIsInstance(window, PredictWindow)
        self.assertIsNotNone(window.page)
        window.close()

    def test_closing_the_window_safely_handles_no_thread(self):
        window = PredictWindow()
        window.page.worker = None
        window.close()                       # must not raise
        self.assertIsNone(window.page.worker)

    def test_a_camera_event_after_the_page_is_gone_does_not_reach_it(self):
        """The crash this guards: a lambda connection outliving its widgets.

        `shared_service()` is a process-wide singleton, so it is still emitting
        after the Predict window is closed and its widgets deleted. A connection
        made with a **lambda** has no receiver QObject for PySide6 to disconnect, so
        the callback reached freed C++ objects ("libshiboken: Internal C++ object
        already deleted"). A bound method is disconnected automatically.
        """
        page = Predict()
        service = page.service
        from shiboken6 import Shiboken, delete

        delete(page)                         # destroy the C++ object for real
        self.assertFalse(Shiboken.isValid(page),
                         "the page must really be deleted for this test to mean "
                         "anything")
        service.status.emit("Camera live")   # must be a no-op, not a RuntimeError
        service.opened.emit()
        service.closed.emit()


@unittest.skipUnless(TORCH, "PyTorch is not installed")
class RealModelTests(PredictPageTestCase):
    """With a checkpoint on disk: it loads, runs, and reports its latency."""

    def _checkpoint(self):
        return write_tiny_checkpoint(self._tmp.name)

    def test_a_checkpoint_loads_and_describes_itself(self):
        page = self._page()
        self.assertTrue(page.load_detector(str(self._checkpoint())))
        self.assertIn("rtmdet-tiny", page.detector.describe())
        self.assertEqual(page.status_label.text()[:16], "Detector loaded.")

    def test_detecting_on_one_image_writes_the_summary(self):
        """An untrained model finds nothing - and says *that*, not nothing at all."""
        import cv2

        page = self._page()
        page.load_detector(str(self._checkpoint()))
        image = Path(self._tmp.name) / "frame.png"
        cv2.imwrite(str(image), np.full((64, 128, 3), 200, dtype=np.uint8))
        page.open_image(image)
        self.assertIn("animal(s) found", page.summary_label.text())
        self.assertGreater(page.latency_ms(), 0.0)

    def test_the_detector_runs_at_the_size_the_checkpoint_was_trained_at(self):
        """Native-resolution inference changes the scale statistics it learned."""
        import cv2

        from daphnia.vision.rtmdet.data import letterbox
        from daphnia.vision.rtmdet.detect import load_detector

        detector = load_detector(str(self._checkpoint()))
        self.assertEqual(detector.input_size, 128)
        seen = {}
        real_detect = detector.model.detect

        def spy(images, **kwargs):
            seen["shape"] = tuple(images.shape[-2:])
            return real_detect(images, **kwargs)

        detector.model.detect = spy
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
        detector.detect(frame)
        # 1280x720 letterboxed to a longest side of 128 -> 128x96, padded to 128x96
        # (a multiple of 32), NOT the native 1280x736.
        self.assertEqual(seen["shape"][0] % 32, 0)
        self.assertEqual(max(seen["shape"]), 128)

    def test_the_threshold_controls_reach_the_detector(self):
        page = self._page()
        page.load_detector(str(self._checkpoint()))
        page.spin_score.setValue(0.75)
        page.spin_iou.setValue(0.30)
        self.assertAlmostEqual(page.detector.score_threshold, 0.75)
        self.assertAlmostEqual(page.detector.iou_threshold, 0.30)

    def test_a_foreign_file_is_refused_with_its_reason(self):
        broken = Path(self._tmp.name) / "broken.pt"
        broken.write_bytes(b"not a checkpoint")
        with self.assertRaises(ModelUnavailable):
            load_detector(str(broken))


class LocalizeStepTests(PredictPageTestCase):
    """The pipeline's Localize step is the same detector, called from code."""

    def test_it_names_the_command_that_produces_a_detector(self):
        from daphnia.vision.detection import localize_well

        with self.assertRaises(NotImplementedError) as caught:
            localize_well(np.zeros((32, 32, 3), np.uint8))
        message = str(caught.exception)
        self.assertIn("No trained detector", message)
        self.assertIn("make train", message)

    @unittest.skipUnless(TORCH, "PyTorch is not installed")
    def test_it_returns_the_detections_of_a_loaded_detector(self):
        from daphnia.vision.detection import localize_well

        page = self._page()
        page.load_detector(str(self._checkpoint()))
        regions = localize_well(np.full((64, 128, 3), 200, dtype=np.uint8),
                                detector=page.detector)
        self.assertIsInstance(regions, list)

    def _checkpoint(self):
        return write_tiny_checkpoint(self._tmp.name)


@unittest.skipUnless(TORCH, "PyTorch is not installed")
class PanelWidthTests(PredictPageTestCase):
    """The controls panel is the width every other surface uses - and its content fits.

    The trap this guards is a ``QComboBox`` that sizes itself to its longest **item**: a
    calibration id is a whole setup key, and one of them made this panel's content
    **711 px** wide inside a **420 px** panel, so the combo and everything under it was
    clipped (measured 2026-10-17). The fix is in two places, and both are asserted here:
    the panel is the shared width, and the combo elides its label (the id staying in the
    tooltip and the choice being made by data).
    """

    def _shown(self):
        window = PredictWindow()
        self.addCleanup(window.close)
        window.resize(1240, 820)
        window.show()
        self.app.processEvents()
        return window.page

    def test_the_panel_is_the_shared_width_and_holds_its_content(self):
        self.activate_profile("PROF_ACTIVE_LONG_20261017_164855", uncertainty_pct=0.5)
        page = self._shown()
        scroll = page.controls.findChild(QScrollArea, "controls_scroll")

        self.assertEqual(page.controls.width(), CONTROLS_WIDTH)
        self.assertLessEqual(scroll.widget().minimumSizeHint().width(),
                             scroll.viewport().width(),
                             "the panel clips its content - see PANEL_WIDTH")

    def test_a_long_calibration_id_stays_in_the_tooltip_not_the_panel(self):
        profile_id = "PROF_ACTIVE_LONG_20261017_164855"
        self.activate_profile(profile_id, uncertainty_pct=0.5)
        page = self._shown()
        combo = page.combo_scale

        self.assertLessEqual(combo.minimumSizeHint().width(), 260,
                             "the combo is sizing itself to its longest item again")
        index = next(i for i in range(combo.count())
                     if combo.itemData(i) == ("profile", profile_id))
        self.assertIn(profile_id, combo.itemData(index, Qt.ToolTipRole))
        self.assertIn("um/px", combo.itemText(index))
        # The scale still resolves: the choice is by data, not by the visible label.
        self.assertIsNotNone(page.um_per_px)


class VideoSteppingTests(PredictPageTestCase):



    """The whole path: a real video file, the real worker, real stepping.

    The stub tests above prove what the page *asks for*. This proves the worker does
    it: opening a clip, pausing it, and stepping one frame each way moves the frame
    the page reports by exactly one, in the direction asked for.
    """

    #: Long enough that pausing right after opening it can never race the reader.
    FRAMES = 200

    def _checkpoint(self):
        return write_tiny_checkpoint(self._tmp.name)

    def _clip(self):
        """Write a real (tiny) mp4 whose frames differ, so a step is visible."""
        import cv2

        path = Path(self._tmp.name) / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                                 (64, 48))
        self.assertTrue(writer.isOpened(), "OpenCV could not open a video writer")
        for index in range(self.FRAMES):
            writer.write(np.full((48, 64, 3), index % 250, np.uint8))
        writer.release()
        return path

    def _wait_for(self, condition, timeout=20.0):
        """Pump the event loop until ``condition()`` holds; False on a timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if condition():
                return True
            time.sleep(0.01)
        return False

    def test_stepping_moves_one_frame_each_way_in_a_real_video(self):
        page = self._page()
        page.load_detector(str(self._checkpoint()))
        page.open_video(self._clip())
        self.assertTrue(page.frame_stepping())

        # Pause at once, so the clip cannot run out under the test - and step once to
        # get a first frame: a paused loop reads only what a step asks it to read.
        page.toggle_play()
        page.next_frame()
        self.assertTrue(self._wait_for(lambda: page._index >= 1),
                        "the worker never showed a first frame")

        first = page._index
        page.next_frame()
        self.assertTrue(self._wait_for(lambda: page._index == first + 1),
                        f"the next frame never arrived (index {page._index})")

        page.previous_frame()
        self.assertTrue(self._wait_for(lambda: page._index == first),
                        f"the previous frame never arrived (index {page._index})")

        # And the screen says where we are: the panel's summary, and the *Source* menu's
        # own entries carry the choice (`X-N13` - the camera string left the bar).
        self.assertIn(f"frame {first}", page.summary_label.text())
        self.assertIsNotNone(page.workspace.action("open_video"))


def pose_prediction(box=(10.0, 20.0, 90.0, 100.0), head=(30.0, 40.0),
                    tail=(70.0, 80.0), score=0.9):
    """A keypoint prediction as the pose model would return it, in source pixels."""
    return PosePrediction(box=box, score=score,
                          points={"head": head, "tail": tail},
                          names=("head", "tail"))


class PoseOverlayTests(PredictPageTestCase):
    """The keypoints the surface draws and reports - without loading any model.

    The drawing and the readout are separate from the inference on purpose: what is
    asserted here is that a *prediction already in hand* becomes a line on the canvas,
    a length in the list and a line in the readout - so a wiring bug is caught here
    rather than in a live view nobody is looking at.
    """

    def test_the_keypoints_tool_hides_itself_until_a_model_is_loaded(self):
        page = self._page()
        self.assertIsNotNone(page.workspace.action("keypoints"))
        self.assertFalse(page.workspace.action("keypoints").isVisible())

    def test_without_a_pose_checkpoint_the_label_says_how_to_build_one(self):
        page = self._page()
        self.assertFalse(page.load_pose())
        self.assertIsNone(page.pose_model)
        self.assertIn("make train-pose", page.detector_label.text())

    def test_a_pose_is_drawn_as_the_head_to_tail_line(self):
        page = self._page()
        page._poses = [pose_prediction()]
        page.canvas.set_source(np.zeros((200, 160, 3), np.uint8))
        page._draw_poses()
        self.assertEqual(page.canvas.annotations,
                         [(1, [((30.0, 40.0), (70.0, 80.0))])])

    def test_a_frame_with_no_poses_clears_the_overlay(self):
        page = self._page()
        page.canvas.set_annotations([(1, [((0.0, 0.0), (1.0, 1.0))])])
        page._poses = []
        page._draw_poses()
        self.assertEqual(page.canvas.annotations, [])

    def test_the_median_length_is_reported_in_pixels_and_labelled_unscaled(self):
        page = self._page()
        page._poses = [pose_prediction(head=(0, 0), tail=(30, 40)),    # 50 px
                       pose_prediction(head=(0, 0), tail=(60, 80))]    # 100 px
        self.assertIn("median length 75.0 px", page._pose_summary())
        self.assertIn("unscaled", page._pose_summary())

    def test_a_frame_reports_each_animal_with_its_length(self):
        page = self._page()
        detection = Detection(box=(10.0, 20.0, 90.0, 100.0), score=0.9, label=0,
                              name="daphnia")
        page._on_frame(np.zeros((200, 160, 3), np.uint8), [detection], 5.0, 0,
                       [pose_prediction()])
        self.assertEqual(len(page.poses()), 1)
        # The list reads like the Bench's *This view*: who, then what was found.
        row = page.list_detections.topLevelItem(0)
        self.assertEqual(row.text(0), "Animal 1")
        self.assertTrue(row.text(1).strip().startswith("57 px"), row.text(1))
        self.assertIn("daphnia", row.text(1))
        # The number the surface exists to show is in the **panel** now, under the count it
        # belongs to: the median-length line used to be a card over the picture
        # (2026-10-18).
        self.assertIn("median length", page.summary_label.text())

    def test_a_pose_is_matched_to_its_animal_by_the_box(self):
        page = self._page()
        page._poses = [pose_prediction(box=(1.0, 1.0, 9.0, 9.0))]
        matching = Detection(box=(1.0, 1.0, 9.0, 9.0), score=0.9, label=0, name="d")
        other = Detection(box=(50.0, 50.0, 60.0, 60.0), score=0.9, label=0, name="d")
        self.assertIsNotNone(page._pose_for(matching))
        self.assertIsNone(page._pose_for(other))


def write_tiny_pose_checkpoint(folder) -> Path:
    """Write a real (untrained) RTMPose checkpoint into ``folder``.

    Real weights, no training: enough to prove that the surface loads both models,
    that the label reports the checkpoint's own metrics, and that a frame runs the
    whole cascade - without a training run in the suite.
    """
    from daphnia.vision.rtmpose.model import build_model
    from daphnia.vision.rtmpose.train import (
        TrainConfig,
        checkpoint_payload,
        save_checkpoint,
    )

    path = Path(folder) / "tiny_pose.pt"
    model = build_model("tiny", 2, 128, 2.0, ("head", "tail"))
    save_checkpoint(path, checkpoint_payload(
        model, TrainConfig(input_size=128), None,
        {"median_error_px": 3.0, "length_median_error_um": 420.0},
        {"samples": 384}))
    return path


@unittest.skipUnless(TORCH, "PyTorch is not installed")
class PoseModelLoadTests(PredictPageTestCase):
    """Both models through the one **Load detector** button, on real checkpoints."""

    def _checkpoint(self):
        return write_tiny_checkpoint(self._tmp.name)

    def test_the_pose_model_loads_alongside_the_detector(self):
        os.environ[POSE_CHECKPOINT_ENV] = str(write_tiny_pose_checkpoint(self._tmp.name))
        page = self._page()
        self.assertTrue(page.load_detector(str(self._checkpoint())))
        self.assertIsNotNone(page.pose_model)
        self.assertIn("rtmpose-tiny", page._pose_line())
        self.assertIn("pose model", page.status_label.text())
        self.assertTrue(page.workspace.action("keypoints").isVisible())

    def test_the_detector_still_loads_when_there_is_no_pose_model(self):
        page = self._page()                       # the pose env points at nothing
        self.assertTrue(page.load_detector(str(self._checkpoint())))
        self.assertIsNone(page.pose_model)
        self.assertIn("No trained pose model", page._pose_line())
        self.assertIn("boxes only", page.status_label.text())
        self.assertFalse(page.workspace.action("keypoints").isVisible())

    def test_one_image_runs_both_stages_and_still_reports_a_latency(self):
        import cv2

        os.environ[POSE_CHECKPOINT_ENV] = str(write_tiny_pose_checkpoint(self._tmp.name))
        page = self._page()
        page.load_detector(str(self._checkpoint()))
        image = Path(self._tmp.name) / "frame.png"
        cv2.imwrite(str(image), np.full((64, 128, 3), 200, dtype=np.uint8))
        page.open_image(image)
        self.assertIn("animal(s) found", page.summary_label.text())
        self.assertGreater(page.latency_ms(), 0.0)


class ScaleTests(PredictPageTestCase):
    """Where the micrometres come from: the app's calibration, or a manual value.

    This is the surface's whole unit story. With no calibration it must say *pixels*
    and call them unscaled; with one it must show the calibration it used and warn
    when that calibration is weak or stale - because a scale that is silently wrong is
    worse than no scale at all (`projectbrief.md`, non-negotiable 1).
    """

    def test_without_a_calibration_lengths_are_pixels_and_say_unscaled(self):
        page = self._page()
        self.assertIsNone(page.um_per_px)
        self.assertIn("No scale", page.scale_label.text())
        page._poses = [pose_prediction(head=(0, 0), tail=(30, 40))]      # 50 px
        self.assertIn("50.0 px", page._length_text(page._poses[0]))
        self.assertIn("unscaled", page._pose_summary())

    def test_the_active_calibration_is_used_without_being_asked(self):
        self.activate_profile("PROF_ACTIVE", px_per_um=0.013363)
        page = self._page()
        self.assertIsNotNone(page.scale)
        self.assertEqual(page.scale.profile_id, "PROF_ACTIVE")
        self.assertAlmostEqual(page.um_per_px, 1.0 / 0.013363, places=6)
        self.assertIn("PROF_ACTIVE", page.scale_label.text())

    def test_a_length_is_reported_in_micrometres_once_calibrated(self):
        self.activate_profile("PROF_ACTIVE", px_per_um=0.013363)     # 74.8 um/px
        page = self._page()
        page._poses = [pose_prediction(head=(0, 0), tail=(30, 40))]      # 50 px
        self.assertAlmostEqual(page._length(page._poses[0])[0],
                               50.0 / 0.013363, places=3)
        self.assertIn("um", page._length_text(page._poses[0]))
        self.assertIn("3742 um", page._pose_summary())          # 50 px / 0.013363

    def test_the_label_above_the_animal_carries_the_length(self):
        self.activate_profile("PROF_ACTIVE")
        page = self._page()
        page._poses = [pose_prediction(head=(10, 20), tail=(40, 60))]
        page._draw_poses()
        self.assertEqual(len(page.canvas.annotations_with_text), 1)
        _index, segments, text = page.canvas.annotations_with_text[0]
        self.assertEqual(len(segments), 1)                  # one head-to-tail line
        self.assertTrue(text.endswith(" um"), text)

    def test_a_manual_scale_says_it_is_not_a_verified_calibration(self):
        page = self._page()
        page.combo_scale.setCurrentIndex(page.combo_scale.count() - 1)   # Manual
        page.spin_um_per_px.setValue(21.2)
        self.assertAlmostEqual(page.um_per_px, 21.2)
        self.assertIsNone(page.scale)
        self.assertIn("NOT a verified calibration", page.scale_label.text())

    def test_a_stale_calibration_is_flagged(self):
        self.activate_profile("PROF_OLD", verified_ago_hours=48)
        page = self._page()
        self.assertIn("STALE", page.scale_label.text())

    def test_a_weak_calibration_is_flagged(self):
        self.activate_profile("PROF_WEAK", uncertainty_pct=9.9)
        page = self._page()
        self.assertIn("WEAK", page.scale_label.text())

    def test_switching_the_scale_relabels_without_re_inferring(self):
        """The convert acts on a pixel length already measured, so it is instant."""
        self.activate_profile("PROF_ACTIVE")
        page = self._page()
        page._poses = [pose_prediction()]
        page._frame = np.zeros((200, 160, 3), np.uint8)     # a frame is on screen
        page.canvas.set_source(page._frame)
        calls = []
        page.pose_model = type("Stub", (), {
            # A pose model says what it is (`_pose_line` and the header's Model fact both
            # ask), and does nothing else here: what this test asserts is that switching
            # the scale runs no inference.
            "describe": lambda self: "stub pose model",
            "detect_frame": lambda *args, **kwargs: calls.append(1)})()
        page.combo_scale.setCurrentIndex(0)                 # -> "No scale"
        self.assertIsNone(page.um_per_px)
        self.assertEqual(calls, [], "changing the scale must not run the model")
        self.assertIn("px", page.canvas.annotations_with_text[0][2])

    def test_the_readout_quotes_the_calibrations_own_spread_and_says_what_it_is_not(self):
        self.activate_profile("PROF_ACTIVE", uncertainty_pct=0.5)
        page = self._page()
        page._poses = [pose_prediction(head=(0, 0), tail=(30, 40))]
        summary = page._pose_summary()
        self.assertIn("scale +/- 0.50% only", summary)
        self.assertIn("separate number", summary)   # the model's error is not this


class AverageTests(PredictPageTestCase):
    """Identity across frames and the mean of the repeats - the mode you asked for.

    The point of every one of these is the same: one frame is one *sample*, and a
    container holding several animals needs both an identity that outlives the frame and
    a number that stops bouncing. What is asserted here is the arithmetic and the
    identity; the model's accuracy is a separate, measured thing (`--evaluate`).
    """

    def frame(self, page, poses, index=0):
        """Feed one processed frame to the page, as the worker would."""
        page._on_frame(np.zeros((200, 160, 3), np.uint8), [], 5.0, index, poses)

    def test_the_average_mode_is_the_default(self):
        self.assertEqual(self._page()._label_mode, "average")

    def test_two_frames_give_a_mean_and_its_spread(self):
        """40 px then 48 px -> mean 44.0 px, and the label says n=2 with a +/-."""
        page = self._page()
        self.frame(page, [pose_prediction(box=(0, 0, 80, 80), head=(10, 10),
                                          tail=(50, 10))])                  # 40 px
        self.frame(page, [pose_prediction(box=(1, 1, 81, 81), head=(12, 10),
                                          tail=(60, 10))])                  # 48 px
        track = page.tracker.tracks()[0]
        self.assertEqual(track.count, 2)
        self.assertAlmostEqual(track.mean_px, 44.0)
        label = page.canvas.annotations_with_text[0][2]
        self.assertIn("44.0", label)
        self.assertIn("n=2", label)
        self.assertIn("+/-", label)

    def test_the_average_stays_put_where_the_frame_value_does_not(self):
        page = self._page()
        self.frame(page, [pose_prediction(box=(0, 0, 80, 80), head=(10, 10),
                                          tail=(50, 10))])
        self.frame(page, [pose_prediction(box=(1, 1, 81, 81), head=(12, 10),
                                          tail=(60, 10))])
        self.assertIn("44.0", page.canvas.annotations_with_text[0][2])
        self.frame(page, [pose_prediction(box=(0, 0, 80, 80), head=(10, 10),
                                          tail=(50, 10))])                  # 40 px again
        third = page.canvas.annotations_with_text[0][2]
        self.assertIn("42.7", third)          # (40 + 48 + 40) / 3 - neither 40 nor 48
        self.assertIn("n=3", third)

    def test_per_frame_mode_shows_only_this_frame(self):
        page = self._page()
        page.combo_labels.setCurrentIndex(0)                    # -> Per frame
        self.assertEqual(page._label_mode, "frame")
        self.frame(page, [pose_prediction(box=(0, 0, 80, 80), head=(10, 10),
                                          tail=(50, 10))])
        self.frame(page, [pose_prediction(box=(1, 1, 81, 81), head=(12, 10),
                                          tail=(60, 10))])
        label = page.canvas.annotations_with_text[0][2]
        self.assertIn("48.0", label)                            # this frame, not 44.0
        self.assertNotIn("n=", label)

    def test_hidden_mode_keeps_the_line_and_drops_the_number(self):
        page = self._page()
        page.combo_labels.setCurrentIndex(2)                    # -> Hidden
        self.frame(page, [pose_prediction()])
        _index, segments, text = page.canvas.annotations_with_text[0]
        self.assertTrue(segments, "the line and the dots stay")
        self.assertEqual(text, "")

    def test_an_animal_keeps_its_identity_and_colour_across_frames(self):
        page = self._page()
        self.frame(page, [pose_prediction(box=(0, 0, 80, 80)),
                          pose_prediction(box=(200, 0, 280, 80))])
        first = [index for index, _s, _t in page.canvas.annotations_with_text]
        self.frame(page, [pose_prediction(box=(1, 1, 81, 81)),
                          pose_prediction(box=(201, 1, 281, 81))])
        second = [index for index, _s, _t in page.canvas.annotations_with_text]
        self.assertEqual(first, second, "the number and colour must not move")

    def test_a_new_animal_gets_a_new_identity(self):
        page = self._page()
        self.frame(page, [pose_prediction(box=(0, 0, 80, 80))])
        self.frame(page, [pose_prediction(box=(0, 0, 80, 80)),
                          pose_prediction(box=(400, 0, 480, 80))])
        self.assertEqual([index for index, _s, _t
                          in page.canvas.annotations_with_text], [1, 2])

    def test_the_average_is_converted_to_micrometres_with_the_scale(self):
        self.activate_profile("PROF_ACTIVE", px_per_um=0.013363)     # 74.8 um/px
        page = self._page()
        self.frame(page, [pose_prediction(box=(0, 0, 80, 80), head=(10, 10),
                                          tail=(50, 10))])
        self.frame(page, [pose_prediction(box=(1, 1, 81, 81), head=(12, 10),
                                          tail=(60, 10))])
        label = page.canvas.annotations_with_text[0][2]
        self.assertIn("um", label)
        self.assertIn("n=2", label)          # 44 px / 0.013363 = 3293 um
        self.assertIn("3293", label)

    def test_the_list_and_the_label_agree_because_both_read_the_track(self):
        page = self._page()
        detection = Detection(box=(0.0, 0.0, 80.0, 80.0), score=0.9, label=0,
                              name="daphnia")
        for _ in range(2):
            page._on_frame(np.zeros((200, 160, 3), np.uint8), [detection], 5.0, 0,
                           [pose_prediction(box=(0, 0, 80, 80), head=(10, 10),
                                            tail=(50, 10))])
        label_value = page.canvas.annotations_with_text[0][2].split()[0]
        row_value = page.list_detections.topLevelItem(0).text(1).strip().split()[0]
        self.assertEqual(round(float(label_value)), round(float(row_value)))

    def test_resetting_starts_the_averages_again(self):
        """After a reset the frame on screen falls back to its own value.

        The identities are gone, so there is nothing to average - the label reverts to
        the per-frame number rather than keeping a mean whose history no longer exists.
        """
        page = self._page()
        self.frame(page, [pose_prediction()])
        self.frame(page, [pose_prediction()])         # the same box: one identity
        self.assertIn("n=2", page.canvas.annotations_with_text[0][2])
        page.reset_tracks()
        self.assertEqual(page.tracker.tracks(), [])
        self.assertNotIn("n=", page.canvas.annotations_with_text[0][2])
        self.assertIn("No animal tracked yet", page.average_label.text())

    def test_the_panel_says_how_many_are_tracked_and_the_biggest_repeat_count(self):
        page = self._page()
        self.frame(page, [pose_prediction(box=(0, 0, 80, 80)),
                          pose_prediction(box=(200, 0, 280, 80))])
        self.frame(page, [pose_prediction(box=(1, 1, 81, 81))])
        text = page.average_label.text()
        self.assertIn("2 animal(s) tracked", text)
        self.assertIn("1 on this frame", text)
        self.assertIn("n=2", text)

    def test_the_same_source_keeps_the_averages_and_a_new_one_does_not(self):
        """Toggling Keypoints restarts the worker; that must not wipe the average."""
        page = self._page()
        page.worker = type("Stub", (), {"source": "live", "path": None,
                                        "stop": lambda self: None,
                                        "deleteLater": lambda self: None})()
        self.assertTrue(page._same_source("live", None))
        self.assertFalse(page._same_source("video", None))
        self.assertFalse(page._same_source("live", "/tmp/other.mp4"))

    def test_switching_mode_re_runs_nothing(self):
        """The tracker folds every frame in already; the mode is a display choice."""
        page = self._page()
        self.frame(page, [pose_prediction()])
        calls = []
        page.pose_model = type("Stub", (), {
            "describe": lambda self: "stub pose model",
            "detect_frame": lambda *args, **kwargs: calls.append(1)})()
        page.combo_labels.setCurrentIndex(1)                    # back to Average
        page.combo_labels.setCurrentIndex(0)                    # and to Per frame
        self.assertEqual(calls, [], "changing the label mode must not run the model")


if __name__ == "__main__":
    unittest.main()
