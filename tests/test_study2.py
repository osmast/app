"""The plate page: studies, plates, wells and the burst that lights a well up.

The plate's own arithmetic (the states, the pages, the walk) is
``tests/test_wells.py`` - Qt-free, hand-computed. This module is the **page**: the rows a
burst writes, what the plate does when it is stored, which well the walk goes to next, and
what every refusal says. It runs on a private database **and** a private recordings tree,
because a capture is a file write and the developer's own ``recordings/`` is not a
scratchpad.
"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from daphnia import paths, plates, settings, wells, workflow
from daphnia.db import connection, init_db, queries, repository
from daphnia.streaming import service as streaming_service
from daphnia.ui import theme
from daphnia.ui.pages.study2 import Study2
from daphnia.ui.widgets import well_grid

#: The tables a test may clear, children first (foreign keys are on).
TABLES = ("corpus_labels", "corpus_boxes", "media_assets", "observations",
          "imaging_sessions", "calibration_profiles", "plates", "studies")


class ScriptedReader(QObject):
    """A reader thread that is *attached* and delivers the frames a test hands it.

    The real :class:`~daphnia.streaming.reader.StreamReader` is a ``QThread`` over a
    socket. What the page asks of it is only "is one attached" (``connected``) and "did a
    frame arrive" (``frame_ready`` -> ``StreamService._on_frame`` -> ``frames_changed``),
    so a scripted stand-in drives the whole chain without a camera - the same move
    ``tests/test_streaming.py`` makes with a scripted ``cv2.VideoCapture``.
    """

    connected = Signal()
    codec_known = Signal(str)
    frame_ready = Signal(object)
    degraded = Signal(int)
    connection_failed = Signal(str)
    connection_lost = Signal(str)

    def __init__(self, url=""):
        super().__init__()
        self.url = url
        self.stopped = False

    def start(self):
        self.connected.emit()

    def stop(self):
        self.stopped = True

    def deliver(self, *frames):
        """Hand over frames the way the reader loop does: one signal per frame."""
        for frame in frames:
            self.frame_ready.emit(frame)


class PlatePageTestCase(unittest.TestCase):
    """A page on a throwaway database and a throwaway recordings tree."""

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._previous_recordings = paths.RECORDINGS_DIR
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "plate-page.db")
        paths.RECORDINGS_DIR = Path(cls._tmp.name) / "recordings"
        init_db()
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        paths.RECORDINGS_DIR = cls._previous_recordings
        cls._tmp.cleanup()
        if cls._previous_db is None:
            os.environ.pop("DAPHNIA_DB", None)
        else:
            os.environ["DAPHNIA_DB"] = cls._previous_db

    def setUp(self):
        self._clear_tables()
        self.page = Study2()
        self.addCleanup(self.page.deleteLater)

    def _clear_tables(self):
        with connection() as conn:
            for table in TABLES:
                conn.execute(f"DELETE FROM {table}")

    # -- helpers -----------------------------------------------------------
    def _study(self, name="Plate run", fmt="24", scale=True):
        """A study that runs plates, one plate of ``fmt``, and (by default) a scale."""
        with connection() as conn:
            study_id = repository.create_study(
                conn, name=name, destination=str(Path(self._tmp.name) / "images"),
                target_mode="plates")
            if scale:
                repository.upsert_profile(
                    conn, profile_id="PROF_T", profile_name="t", microscope_model="m",
                    objective_magnification="2x", camera_device="c", px_per_um=5.0)
                repository.activate_profile(conn, "PROF_T")
            plate_id = repository.create_plate(conn, study_id=study_id, format=fmt)
        return study_id, plate_id

    @staticmethod
    def _frame(amplitude):
        """A frame whose *sharpness* follows ``amplitude`` (variance of the Laplacian).

        Deliberate: "the sharpest frame of the burst is the well's image" is then a claim
        the test can check, not a property it silently assumes.
        """
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, ::8] = amplitude
        return frame

    def _buffer(self, *amplitudes):
        """Fill the shared service's buffer - the frames a burst would take."""
        service = self.page.service
        service._recent.clear()
        for amplitude in amplitudes:
            service._recent.append(self._frame(amplitude))
        return service

    def _observation(self, plate_id, well_label):
        """The observation row of a well, or ``None``."""
        with connection() as conn:
            return queries.find_target(conn, study_id=self.page._study_id,
                                       plate_id=plate_id, well_label=well_label)

    def _assets(self, plate_id, well_label):
        """Every frame stored on one well, in capture order."""
        with connection() as conn:
            target = queries.find_target(conn, study_id=self.page._study_id,
                                         plate_id=plate_id, well_label=well_label)
            return [] if target is None else queries.assets_for(
                conn, target["observation_id"])

    def _tile(self, well_label):
        """The drawn well for a label (its state is what the plate shows)."""
        return self.page.grid.tiles[well_label]


class EmptyPageTests(PlatePageTestCase):
    """What the page says before there is anything to capture."""

    def test_no_study_no_plate_and_no_well_are_named(self):
        self.assertEqual(self.page.grid.tiles, {})
        self.assertIn("No plate open", self.page.page_label.text())
        self.assertIn("No plate in this study", self.page.plate_label.text())
        self.assertIn("No well chosen", self.page.well_label.text())

    def test_a_burst_needs_a_plate_and_a_well_first(self):
        self._buffer(200, 40)
        self.assertIsNone(self.page.capture_burst())
        self.assertIn("Choose a plate and a well", self.page.status_label.text())

    def test_the_burst_button_needs_the_camera(self):
        """The page uses the app's one connection, and says so when there is none."""
        self._study()
        self.page.refresh()
        self.assertFalse(self.page.service.connected)
        self.assertFalse(self.page.btn_capture.isEnabled())
        self.assertEqual(self.page.connection_label.text(),
                         workflow.REASONS["connection"])


class StudyAndPlateTests(PlatePageTestCase):
    """Picking a study and adding a plate - the setup, on this page."""

    def test_a_study_created_here_runs_plates_and_becomes_current(self):
        study_id = self.page.new_study("Plate run")
        with connection() as conn:
            study = queries.study_by_id(conn, study_id)
            self.assertEqual(study["target_mode"], "plates")
            self.assertEqual(queries.active_study(conn)["study_id"], study_id)
        self.assertEqual(self.page._study_id, study_id)
        self.assertEqual(self.page.combo_study.currentText(), "Plate run")

    def test_adopting_a_study_makes_it_the_one_being_worked_on(self):
        with connection() as conn:
            first = repository.create_study(
                conn, name="First", destination=str(self._tmp.name))
            repository.create_study(
                conn, name="Second", destination=str(self._tmp.name))
        self.page.refresh()
        self.page.combo_study.setCurrentIndex(self.page.combo_study.findData(first))
        self.page.use_study()
        with connection() as conn:
            self.assertEqual(queries.active_study(conn)["study_id"], first)

    def test_a_plate_is_added_with_its_format_and_label(self):
        self.page.new_study("Plate run")
        plate_id = self.page.add_plate("24", "Run 1")
        with connection() as conn:
            row = queries.list_plates(conn, self.page._study_id)[0]
            self.assertEqual(row["plate_id"], plate_id)
            self.assertEqual((row["format"], row["label"]), ("24", "Run 1"))
        self.assertEqual(self.page._plate["plate_id"], plate_id)

    def test_an_empty_label_gets_the_next_free_name(self):
        self.page.new_study("Plate run")
        self.page.add_plate("6", "")
        self.assertEqual(self.page._plate["label"], "Plate 1")

    def test_a_plate_cannot_be_added_without_a_study(self):
        self.assertIsNone(self.page.add_plate("24", "Nowhere"))
        self.assertIn("Choose a study first", self.page.status_label.text())

    def test_the_plates_of_the_open_study_are_the_ones_offered(self):
        self.page.new_study("One")
        self.page.add_plate("6", "Six")
        self.page.new_study("Two")
        self.page.add_plate("96", "Ninety-six")
        self.assertEqual(self.page.combo_plate.count(), 1)
        self.assertIn("Ninety-six", self.page.combo_plate.currentText())
        # Back to the first study: its own plate is offered again, not the other's.
        self.page.combo_study.setCurrentIndex(self.page.combo_study.findText("One"))
        self.assertEqual(self.page.combo_plate.count(), 1)
        self.assertIn("Six", self.page.combo_plate.currentText())


class PlateDrawingTests(PlatePageTestCase):
    """The plate on screen: one tile per well, in the plate's own shape."""

    def _open(self, fmt, per_page=0):
        """A study with one plate of ``fmt``, shown ``per_page`` wells at a time."""
        self.page.new_study("Plate run")
        self.page.add_plate(fmt, "P")
        self.page.combo_page.setCurrentIndex(wells.PAGE_SIZES.index(per_page))
        return fmt

    def test_every_well_of_the_plate_is_drawn_in_reading_order(self):
        for fmt in plates.FORMATS:
            with self.subTest(fmt=fmt):
                self._open(fmt)
                labels = plates.well_labels(fmt)
                self.assertEqual(list(self.page.grid.tiles), labels)
                self.assertEqual(self.page.grid.columns, plates.rows_columns(fmt)[1])
                self.assertEqual(self.page.grid.labels, tuple(labels))

    def test_a_96_well_plate_can_be_paged_and_the_page_follows_the_walk(self):
        self._open("96", per_page=24)
        self.assertEqual(len(self.page.grid.tiles), 24)
        self.assertIn("Page 1 of 4", self.page.page_label.text())
        self.page.step_page(1)
        self.assertIn("Page 2 of 4", self.page.page_label.text())
        self.assertIn("C01", self.page.page_label.text())
        # The chosen well decides the page, wherever it sits on the plate.
        self.page.choose_well("H12")
        self.assertIn("Page 4 of 4", self.page.page_label.text())
        self.assertTrue(self._tile("H12").is_current)

    def test_the_first_page_cannot_step_backwards(self):
        self._open("96", per_page=24)
        self.assertFalse(self.page.btn_previous_page.isEnabled())
        self.assertTrue(self.page.btn_next_page.isEnabled())
        self.assertIsNone(self.page.step_page(-1))

    def test_the_tiles_are_square_and_the_plate_fits_the_room_it_is_given(self):
        """The one geometric promise: a plate stays inside its own area."""
        self._open("96")
        grid = self.page.grid
        grid.resize(900, 620)
        self.app.processEvents()
        tiles = list(grid.tiles.values())
        self.assertTrue(all(tile.width() == tile.height() for tile in tiles))
        self.assertTrue(all(tile.width() >= well_grid.TILE_MINIMUM for tile in tiles))
        right = max(tile.mapTo(grid, tile.rect().topRight()).x() for tile in tiles)
        bottom = max(tile.mapTo(grid, tile.rect().bottomLeft()).y() for tile in tiles)
        self.assertLessEqual(right, grid.width())
        self.assertLessEqual(bottom, grid.height())

    def test_the_plate_paints(self):
        """A crash inside `paintEvent` is invisible to the rest of the suite."""
        self._open("24")
        self.page.grid.resize(700, 480)
        self.app.processEvents()
        self.assertFalse(self.page.grid.grab().isNull())
        self.assertFalse(self.page.legend.grab().isNull())

    def test_every_state_of_a_well_has_a_colour_in_both_schemes(self):
        """A state with no colour is a well that cannot light up."""
        self.assertEqual(set(well_grid.STATE_TOKENS), set(wells.STATES))
        for state, token in well_grid.STATE_TOKENS.items():
            for scheme in theme.SCHEMES:
                with self.subTest(state=state, scheme=scheme):
                    self.assertTrue(theme.qcolor(token, scheme).isValid())


class CaptureTests(PlatePageTestCase):
    """A burst: the rows it writes, the file it names, and the well it lights up."""

    def setUp(self):
        super().setUp()
        self.study_id, self.plate_id = self._study(fmt="24")
        self.page.refresh()

    def _capture(self, well_label, *amplitudes):
        """Take a burst on a well, with frames of the given sharpness.

        The auto-advance is switched **off** here: these tests are about the burst and the
        well it landed on, and the walk out of a burst has its own class (`WalkTests`).
        """
        self.page.check_advance.setChecked(False)
        self._buffer(*amplitudes)
        self.page.choose_well(well_label)
        return self.page.capture_burst()

    def test_a_burst_is_an_observation_of_the_well_with_one_image(self):
        self._capture("B02", 40, 120, 200)          # the last frame is the sharpest
        assets = self._assets(self.plate_id, "B02")
        self.assertEqual(len(assets), 3)
        self.assertEqual({row["burst_index"] for row in assets}, {1})
        self.assertEqual([row["frame_index"] for row in assets], [0, 1, 2])
        primary = [row for row in assets if row["is_primary"]]
        self.assertEqual(len(primary), 1, "one well, one image")
        self.assertEqual(primary[0]["sharpness"],
                         max(row["sharpness"] for row in assets))
        self.assertTrue(primary[0]["file_path"].endswith("b01_f02.png"))
        self.assertTrue(Path(primary[0]["file_path"]).exists(),
                        "the frame the row points at is on disk")

    def test_a_second_visit_appends_a_burst_and_keeps_the_better_image(self):
        self._capture("A01", 200)
        first = [row for row in self._assets(self.plate_id, "A01") if row["is_primary"]]
        self._capture("A01", 20)                    # a worse burst, later
        assets = self._assets(self.plate_id, "A01")
        self.assertEqual([row["burst_index"] for row in assets], [1, 2])
        primary = [row for row in assets if row["is_primary"]]
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary[0]["asset_id"], first[0]["asset_id"],
                         "a worse burst must not take the well's image away")

    def test_the_well_lights_up_and_the_panel_names_its_image(self):
        self._capture("C03", 120, 200)
        tile = self._tile("C03")
        self.assertEqual(tile.progress.state, wells.CAPTURED)
        self.assertTrue(tile.progress.is_done)
        self.assertIn("Image: b01_f01.png", self.page.well_label.text())
        self.assertEqual(self.page.gallery.count(), 2)
        self.assertIn("1 of 24 wells have an image", self.page.progress_label.text())
        self.assertIn("Captured  1",
                      self.page.legend.state_labels[wells.CAPTURED].text())

    def test_the_strip_selects_the_frame_that_is_the_wells_image(self):
        self._capture("A02", 40, 200)
        primary = [row for row in self._assets(self.plate_id, "A02") if row["is_primary"]]
        ids = self.page.gallery.asset_ids()
        self.assertEqual(ids[self.page.gallery.currentRow()], primary[0]["asset_id"])

    def test_a_burst_without_a_scale_is_refused(self):
        """An image that can never be measured is not captured (`SR-STUDY2-N01`)."""
        with connection() as conn:
            conn.execute("UPDATE calibration_profiles SET is_active = 0")
        self.assertIsNone(self._capture("A01", 200))
        self.assertEqual(self.page.status_label.text(), workflow.REASONS["calibration"])
        self.assertEqual(self._assets(self.plate_id, "A01"), [])

    def test_a_burst_without_buffered_frames_says_so_and_stores_nothing(self):
        self.page.service._recent.clear()
        self.page.choose_well("A01")
        self.assertIsNone(self.page.capture_burst())
        self.assertIn("No frames buffered", self.page.status_label.text())
        self.assertEqual(self._assets(self.plate_id, "A01"), [])

    def test_no_measurement_is_invented_by_a_capture(self):
        """The page captures; it never claims a number."""
        self._capture("A01", 200)
        self.assertEqual(self._tile("A01").progress.measured, 0)
        self.assertIn("0 measured", self.page.progress_label.text())

    def test_the_counts_of_a_well_carry_the_path_of_its_image(self):
        self._capture("B02", 40, 200)
        with connection() as conn:
            row = queries.plate_well_counts(conn, self.plate_id)["B02"]
        self.assertTrue(row["image_path"].endswith("b01_f01.png"))


class RecordingTests(PlatePageTestCase):
    """A recording: a burst whose length the operator decides (`SR-STUDY2-013`).

    Driven through the app's own chain - ``service.start()`` over a scripted reader, one
    ``frame_ready`` per frame - so what is covered is the path a live camera takes and not
    a call to `_record_frame` behind its back.
    """

    def setUp(self):
        super().setUp()
        self.study_id, self.plate_id = self._study(fmt="24")
        self.page.refresh()
        self.page.check_advance.setChecked(False)      # the walk has its own class
        self._connect()

    # -- helpers -----------------------------------------------------------
    def _connect(self):
        """Attach a scripted reader to the app's shared service: a camera, no socket.

        ``StreamReader`` is patched for the duration of the test (the pattern
        ``tests/test_streaming.py`` uses), and the service is disconnected again on
        teardown, because **the shared service outlives this file's every test**.
        """
        original = streaming_service.StreamReader
        streaming_service.StreamReader = ScriptedReader
        self.addCleanup(setattr, streaming_service, "StreamReader", original)
        self.addCleanup(self.page.service.stop)
        self.page.service.start()
        self.reader = self.page.service._reader
        return self.reader

    def _deliver(self, *amplitudes):
        """The camera hands over frames, of the given sharpness, in order."""
        self.reader.deliver(*(self._frame(amplitude) for amplitude in amplitudes))

    def _live(self, well_label, *amplitudes):
        """The whole gesture: choose the well, Start, frames arrive, Stop."""
        self.page.choose_well(well_label)
        burst = self.page.toggle_recording()
        self._deliver(*amplitudes)
        self.page.toggle_recording()
        return burst

    def _burst(self, well_label, *amplitudes):
        """A short burst on one well, for the tests that mix the two captures."""
        self._buffer(*amplitudes)
        self.page.choose_well(well_label)
        return self.page.capture_burst()

    def _recorded_files(self):
        """Every frame on disk under this page's recordings tree - a set, for comparison.

        A *set*, not a count: the tests of this class share one recordings tree, so only a
        before/after comparison says what one refusal did or did not write.
        """
        return set((Path(self._tmp.name) / "recordings").rglob("*.png"))

    # -- what a recording stores -------------------------------------------
    def test_a_recording_stores_every_frame_that_arrived(self):
        """The length follows the stream, never `DAPHNIA_BURST_SIZE`."""
        burst = self._live("B02", *range(1, 31))

        assets = self._assets(self.plate_id, "B02")
        self.assertEqual(burst, 1)
        self.assertEqual(len(assets), 30)
        self.assertGreater(len(assets), settings.BURST_SIZE)
        self.assertEqual({row["burst_index"] for row in assets}, {1})
        self.assertEqual([row["frame_index"] for row in assets], list(range(30)))
        primary = [row for row in assets if row["is_primary"]]
        self.assertEqual(len(primary), 1, "one well, one image")
        self.assertTrue(primary[0]["file_path"].endswith("b01_f29.png"),
                        "the sharpest frame of the recording is the well's image")
        self.assertTrue(Path(primary[0]["file_path"]).exists(),
                        "the frame the row points at is on disk")
        self.assertTrue(all(Path(row["file_path"]).exists() for row in assets))

    def test_the_frames_are_on_disk_while_it_runs_and_the_rows_come_at_the_end(self):
        """A running recording is files; the burst exists only when it stops."""
        self.page.choose_well("B02")
        self.page.start_recording()
        self._deliver(200)
        self._deliver(120, 40)

        recording = self.page._recording
        self.assertIsNotNone(recording)
        self.assertEqual(recording.frames, 3)
        self.assertEqual(len(list(recording.directory.glob("*.png"))), 3)
        self.assertEqual(self._assets(self.plate_id, "B02"), [],
                         "no other surface sees a half-finished recording")

        self.page.stop_recording()
        self.assertEqual(len(self._assets(self.plate_id, "B02")), 3)

    def test_two_recordings_of_different_lengths_are_two_bursts(self):
        self._live("A01", 200, 200, 200)               # three frames
        self._live("A01", *range(1, 41))               # forty frames

        lengths = {}
        for row in self._assets(self.plate_id, "A01"):
            lengths.setdefault(row["burst_index"], []).append(row)
        self.assertEqual(sorted(len(rows) for rows in lengths.values()), [3, 40])

    def test_a_recording_and_a_burst_share_the_well_its_numbering_and_its_one_image(self):
        self.assertEqual(self._burst("A01", 20), 1)
        self.assertEqual(self._live("A01", *[200] * 5), 2)

        assets = self._assets(self.plate_id, "A01")
        self.assertEqual(len(assets), 6)
        self.assertEqual([row["burst_index"] for row in assets], [1] + [2] * 5)
        primary = [row for row in assets if row["is_primary"]]
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary[0]["burst_index"], 2,
                         "the sharper capture takes the well's image")
        target = self._observation(self.plate_id, "A01")
        self.assertEqual({row["observation_id"] for row in assets},
                         {target["observation_id"]})

    def test_a_recorded_well_lights_up_like_a_burst_one(self):
        self._live("C02", *range(1, 8))
        self.assertEqual(self._tile("C02").progress.state, wells.CAPTURED)
        self.assertEqual(self.page.gallery.count(), 7)
        self.assertIn("1 of 24 wells have an image", self.page.progress_label.text())
        self.assertEqual(self.page.well_label.text().count("Image:"), 1)


    # -- what a recording refuses ------------------------------------------
    def test_a_recording_needs_the_camera(self):
        """Nothing is recorded off a stream that is not there (`SR-STUDY2-N01`)."""
        self.page.service.stop()
        self.assertFalse(self.page.btn_record.isEnabled())
        self.assertIsNone(self.page.start_recording())
        self.assertEqual(self.page.status_label.text(), workflow.REASONS["connection"])

    def test_a_recording_needs_a_plate_and_a_well(self):
        self.page.new_study("Nothing here yet")
        self.assertIsNone(self.page.start_recording())
        self.assertIn("Choose a plate and a well first", self.page.status_label.text())

    def test_a_recording_without_a_scale_is_refused_and_writes_nothing(self):
        with connection() as conn:
            conn.execute("UPDATE calibration_profiles SET is_active = 0")
        self.page.choose_well("A01")

        before = self._recorded_files()
        self.assertIsNone(self.page.start_recording())
        self.assertEqual(self.page.status_label.text(), workflow.REASONS["calibration"])
        self.assertIsNone(self.page._recording)
        self._deliver(200)                # frames keep arriving; none of them is written
        self.assertEqual(self._recorded_files(), before)

    def test_a_recording_that_stored_nothing_is_not_stored(self):
        """`SR-STUDY2-N02`: "it said it saved" is how a plate ends up with holes."""
        self.page.choose_well("A01")
        self.page.start_recording()
        self.assertIsNone(self.page.stop_recording())

        self.assertIn("is not stored", self.page.status_label.text())
        self.assertIn("0.0 s", self.page.status_label.text())
        self.assertEqual(self._assets(self.plate_id, "A01"), [])

    # -- the page while a recording runs -----------------------------------
    def test_the_page_is_locked_to_the_well_while_a_recording_runs(self):
        """`SR-STUDY2-014`: the plate cannot move under a recording in progress."""
        self.page.choose_well("A01")
        self.page.start_recording()
        self.assertEqual(self.page.btn_record.text(), "Stop recording and store")
        for widget in (self.page.btn_capture, self.page.btn_next_well,
                       self.page.btn_refresh, self.page.btn_add_plate,
                       self.page.btn_next_page, self.page.check_advance,
                       self.page.combo_plate, self.page.combo_study, self.page.grid):
            with self.subTest(widget=widget.objectName()):
                self.assertFalse(widget.isEnabled())

        self.assertIsNone(self.page.choose_well("B04"))
        self.assertEqual(self.page._current, "A01", "the camera stays over its well")
        self.assertIn("Recording on A01", self.page.status_label.text())
        self.assertIn("press Stop", self.page.status_label.text())
        self.assertIsNone(self.page.capture_burst())
        self.assertIn("Recording on A01", self.page.status_label.text())

        self.page.stop_recording()
        self.assertEqual(self.page.btn_record.text(), "Start recording")
        self.assertTrue(self.page.btn_next_well.isEnabled())
        self.assertTrue(self.page.grid.isEnabled())
        self.assertTrue(self.page.combo_plate.isEnabled())

    # -- how a recording ends (`SR-STUDY2-N06`) ----------------------------
    def test_the_frame_limit_ends_the_recording_and_stores_what_arrived(self):
        """The disk guard names itself - it never truncates a take in silence."""
        original = settings.RECORD_MAX_FRAMES
        settings.RECORD_MAX_FRAMES = 4
        self.addCleanup(setattr, settings, "RECORD_MAX_FRAMES", original)

        self.page.choose_well("A01")
        self.page.start_recording()
        self._deliver(1, 2, 3, 4, 5)

        self.assertIsNone(self.page._recording)
        self.assertIn("stopped at the frame limit", self.page.status_label.text())
        self.assertEqual(len(self._assets(self.plate_id, "A01")), 4)
        self.assertEqual(self.page.btn_record.text(), "Start recording")

    def test_a_dropped_connection_stores_what_arrived_and_names_the_ending(self):
        self.page.choose_well("A01")
        self.page.start_recording()
        self._deliver(200, 200, 200)
        self.reader.connection_lost.emit("Stream ended or disconnected")

        self.assertIsNone(self.page._recording)
        self.assertIn("ended by the dropped connection", self.page.status_label.text())
        self.assertEqual(len(self._assets(self.plate_id, "A01")), 3)

    def test_leaving_the_page_stores_the_recording(self):
        """A hidden page must not go on filling a folder nobody is looking at."""
        self.page.show()
        self.app.processEvents()
        self.page.choose_well("A01")
        self.page.start_recording()
        self._deliver(200, 200)

        self.page.hide()
        self.app.processEvents()
        self.assertIsNone(self.page._recording)
        self.assertIn("ended by the page going off screen",
                      self.page.status_label.text())
        self.assertEqual(len(self._assets(self.plate_id, "A01")), 2)

    def test_closing_the_window_stores_the_recording(self):
        """The window is `WA_DeleteOnClose`: that is the page's last moment."""
        from daphnia.ui.hub import Study2Window

        window = Study2Window()
        self.addCleanup(window.deleteLater)
        window.show()
        self.app.processEvents()
        page = window.page
        page.check_advance.setChecked(False)
        page.choose_well("A01")
        page.start_recording()
        self._deliver(200, 200)

        window.close()
        self.app.processEvents()
        self.assertIsNone(page._recording)
        self.assertIn("ended by the page going off screen", page.status_label.text())
        self.assertEqual(len(self._assets(self.plate_id, "A01")), 2)


class WellPictureTests(PlatePageTestCase):
    """The burst appears **in its well**, so the plate fills up as it is walked."""

    def setUp(self):
        super().setUp()
        self.study_id, self.plate_id = self._study(fmt="24")
        self.page.refresh()
        self.page.check_advance.setChecked(False)

    def _capture(self, well_label, *amplitudes):
        self._buffer(*amplitudes)
        self.page.choose_well(well_label)
        return self.page.capture_burst()

    def test_a_captured_well_draws_its_frame_and_an_empty_one_does_not(self):
        self._capture("B02", 200)
        drawn = self._tile("B02").image
        self.assertIsNotNone(drawn)
        self.assertFalse(drawn.isNull())
        self.assertIsNone(self._tile("A01").image)

    def test_a_sharper_burst_becomes_the_picture_and_a_worse_one_does_not(self):
        self._capture("B02", 20)
        first = self._tile("B02").image
        self._capture("B02", 200)               # sharper: a new image for the well
        sharper = self._tile("B02").image
        self.assertIsNot(sharper, first)
        self._capture("B02", 20)                # worse again: the picture stays
        self.assertIs(self._tile("B02").image, sharper)

    def test_the_picture_in_a_well_can_be_switched_off(self):
        self._capture("B02", 200)
        self.page.check_images.setChecked(False)
        self.assertIsNone(self._tile("B02").image)
        self.assertEqual(self._tile("B02").progress.state, wells.CAPTURED,
                         "the state is not the picture: the well stays captured")
        self.page.check_images.setChecked(True)
        self.assertIsNotNone(self._tile("B02").image)

    def test_a_frame_that_can_no_longer_be_read_leaves_the_state_alone(self):
        """A moved file is a lost picture, not a lost well."""
        self._capture("B02", 200)
        asset = self._assets(self.plate_id, "B02")[0]
        Path(asset["file_path"]).unlink()
        well_grid._PIXMAP_CACHE.clear()          # meet the unreadable file, not a cached one
        self.page.refresh()

        tile = self._tile("B02")
        self.assertIsNone(tile.image)
        self.assertEqual(tile.progress.state, wells.CAPTURED)
        self.assertIn("1 of 24 wells have an image", self.page.progress_label.text())


class WalkTests(PlatePageTestCase):
    """Next well: the loop the operator drives by moving the plate."""

    def setUp(self):
        super().setUp()
        self.study_id, self.plate_id = self._study(fmt="6")
        self.page.refresh()

    def _capture(self, well_label):
        """A burst on one well (the auto-advance is left as the page ships it)."""
        self._buffer(200)
        self.page.choose_well(well_label)
        return self.page.capture_burst()

    def test_the_walk_starts_on_the_first_well_of_the_plate(self):
        self.assertEqual(self.page._current, "A01")
        self.assertTrue(self._tile("A01").is_current)

    def test_the_next_well_is_the_next_one_without_an_image(self):
        self.page.check_advance.setChecked(False)
        self._capture("A01")
        self.assertEqual(self.page.next_well(), "A02")
        self.assertEqual(self.page._current, "A02")
        self.assertIn("Camera over A02", self.page.status_label.text())

    def test_the_walk_wraps_to_the_well_that_was_missed(self):
        self.page.check_advance.setChecked(False)
        self._capture("B03")
        self.assertEqual(self.page.next_well(), "A01")

    def test_a_full_plate_reports_every_well_imaged_and_names_what_is_left(self):
        self.page.check_advance.setChecked(False)
        for label in plates.well_labels("6"):
            self._capture(label)
        self.assertIsNone(self.page.next_well())
        self.assertIn("Every well", self.page.status_label.text())
        self.assertIn("does not measure yet", self.page.status_label.text())
        self.assertEqual(self.page.progress_label.text(),
                         "6 of 6 wells have an image; 0 measured")

    def test_advancing_after_a_burst_is_the_default_loop(self):
        """The loop is burst, move the plate, next well - and it is on by default."""
        self.assertTrue(self.page.check_advance.isChecked())
        self._capture("A01")
        self.assertEqual(self.page._current, "A02")
        self.assertIn("Burst 1 stored for A01", self.page.status_label.text())
        self.assertIn("Next well: A02", self.page.status_label.text())


class MeasuredStateTests(PlatePageTestCase):
    """A well measured somewhere else shows as measured - the end goal's rung."""

    def test_a_measured_well_is_lit_and_counted_but_is_not_the_next_one(self):
        study_id, plate_id = self._study(fmt="6")
        with connection() as conn:
            session_id = repository.ensure_session(
                conn, study_id=study_id, profile_id="PROF_T")
            observation_id = repository.create_observation(
                conn, session_id=session_id, plate_id=plate_id, well_label="B02")
            asset_id = repository.add_media_asset(
                conn, observation_id=observation_id, file_path="b01_f00.png",
                burst_index=1, sharpness=5.0)
            repository.save_manual_measurement(
                conn, observation_id=observation_id, asset_id=asset_id,
                profile=repository.active_profile(conn),
                eye_apex=(0.0, 0.0), spine_base=(50.0, 0.0))
        self.page.refresh()

        tile = self._tile("B02")
        self.assertEqual(tile.progress.state, wells.MEASURED)
        self.assertEqual(tile.progress.measured, 1)
        self.assertIn("Measured  1",
                      self.page.legend.state_labels[wells.MEASURED].text())
        self.assertIn("1 measured", self.page.progress_label.text())
        # The walk passes a measured well by: it is done, image and number alike.
        self.page.choose_well("B01")
        self.assertEqual(self.page.next_well(), "B03")


class PlateWindowTests(PlatePageTestCase):
    """The window the hub opens: the plate, its key and the shared footer."""

    def test_the_window_builds_shows_and_paints(self):
        from daphnia.ui.hub import Study2Window

        window = Study2Window()
        self.addCleanup(window.close)
        window.resize(1200, 800)
        window.show()
        self.app.processEvents()

        page = window.page
        self.assertIsInstance(page, Study2)
        self.assertEqual(page.grid.objectName(), "well_plate")
        self.assertEqual(len(page.legend.swatches), len(wells.STATES))
        self.assertEqual(window.status_footer.objectName(), "status_footer")
        self.assertFalse(page.grid.grab().isNull())
        self.assertFalse(page.legend.grab().isNull())

