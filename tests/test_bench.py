"""Bench: measuring without a study - and adopting the result into one later.

The point of these tests is the *absence* of ceremony: no study, no plate, no
connection, and still a stored, auditable length.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

from daphnia import maintenance, paths
from daphnia.db import connection, init_db, queries, repository
from daphnia.ui.pages.bench import (BENCH_DRAW_TOOLS, NAME_COLUMN, NO_INDIVIDUAL, Bench)
from daphnia.ui.widgets import specimen_color
from tests.test_images import write_image

try:                                     # the video-import test needs a real clip
    import cv2 as _cv2
except ImportError:                      # pragma: no cover - env issue
    _cv2 = None


class BenchStorageTests(unittest.TestCase):
    """Study-less rows: created on demand, promoted later, never duplicated."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "bench.db"
        self.destination = Path(self._tmp.name) / "images"
        init_db(self.db_path)

    def _profile(self, conn):
        repository.upsert_profile(
            conn, profile_id="PROF_B", profile_name="b", microscope_model="m",
            objective_magnification="2x", camera_device="c", px_per_um=2.0)
        repository.activate_profile(conn, "PROF_B")
        return repository.active_profile(conn)

    def _bench_measurement(self, conn, file_path="/tmp/x.png"):
        profile = self._profile(conn)
        observation_id = repository.bench_observation(
            conn, file_path=file_path, profile_id=profile.profile_id)
        asset = repository.bench_asset(conn, observation_id)
        repository.save_measurement(
            conn, observation_id=observation_id, asset_id=asset["asset_id"],
            profile=profile, trait="length", p1=(10.0, 20.0), p2=(110.0, 20.0))
        return observation_id

    def test_a_bench_observation_has_no_study(self):
        with connection(self.db_path) as conn:
            observation_id = self._bench_measurement(conn)
            row = conn.execute(
                "SELECT s.study_id FROM observations o"
                " JOIN imaging_sessions s ON s.session_id = o.session_id"
                " WHERE o.observation_id = ?", (observation_id,)).fetchone()
        self.assertIsNone(row["study_id"])

    def test_the_same_file_reuses_its_observation(self):
        """Re-opening a photo continues it; it must not scatter into new rows."""
        with connection(self.db_path) as conn:
            first = self._bench_measurement(conn)
            second = self._bench_measurement(conn)
            observations = conn.execute(
                "SELECT COUNT(*) FROM observations").fetchone()[0]
        self.assertEqual(first, second)
        self.assertEqual(observations, 1)

    def test_each_image_gets_its_own_session(self):
        """Promotion must not drag every other bench image along with it."""
        with connection(self.db_path) as conn:
            first = self._bench_measurement(conn, "/tmp/a.png")
            second = self._bench_measurement(conn, "/tmp/b.png")
            sessions = conn.execute(
                "SELECT COUNT(*) FROM imaging_sessions WHERE study_id IS NULL"
            ).fetchone()[0]
        self.assertNotEqual(first, second)
        self.assertEqual(sessions, 2)

    def test_a_bench_measurement_keeps_its_landmarks_and_its_scale(self):
        with connection(self.db_path) as conn:
            self._bench_measurement(conn)
            row = queries.bench_measurements(conn)[0]
        self.assertAlmostEqual(row["length_px"], 100.0)
        self.assertAlmostEqual(row["length_um"], 50.0)        # 2 px/um
        self.assertEqual((row["p1_x"], row["p2_x"]), (10.0, 110.0))
        self.assertEqual(row["file_path"], "/tmp/x.png")

    def test_a_study_measurement_is_not_a_bench_measurement(self):
        with connection(self.db_path) as conn:
            profile = self._profile(conn)
            study_id = repository.create_study(
                conn, name="T", destination=self.destination)
            session_id = repository.ensure_session(
                conn, study_id=study_id, profile_id=profile.profile_id)
            observation_id = repository.create_observation(
                conn, session_id=session_id, individual_index=1)
            repository.add_media_asset(
                conn, observation_id=observation_id, file_path="/tmp/study.png",
                frame_index=0, is_primary=True)
            repository.save_measurement(
                conn, observation_id=observation_id, profile=profile,
                trait="length", p1=(0.0, 0.0), p2=(10.0, 0.0))
            self.assertEqual(queries.bench_count(conn), 0)
            self.assertEqual(len(queries.bench_measurements(conn)), 0)

    def test_a_bench_image_can_be_attached_to_a_study_later(self):
        with connection(self.db_path) as conn:
            profile = self._profile(conn)
            study_id = repository.create_study(
                conn, name="Adopted", destination=self.destination)
            observation_id = repository.bench_observation(
                conn, file_path="/tmp/x.png", profile_id=profile.profile_id)
            asset = repository.bench_asset(conn, observation_id)
            repository.save_measurement(
                conn, observation_id=observation_id, asset_id=asset["asset_id"],
                profile=profile, trait="length", p1=(0.0, 0.0), p2=(100.0, 0.0))

            repository.attach_observation_to_study(
                conn, observation_id=observation_id, study_id=study_id)
            self.assertEqual(queries.bench_count(conn), 0)     # it left the bench
            rows = queries.measurements_for_study(conn, study_id)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["length_um"], 50.0)     # scale preserved

    def test_a_measurement_outside_every_individual_records_no_individual(self):
        """`None` is a real answer: the number is stored as belonging to nobody."""
        with connection(self.db_path) as conn:
            self._bench_measurement(conn)
            row = queries.bench_measurements(conn)[0]
        self.assertIsNone(row["individual_index"])

    def test_two_measurements_of_one_individual_do_not_erase_each_other(self):
        """`individual_index` is **not** part of the row key (`SR-BENCH-016`).

        An animal may carry two `length` rows - two axes, two repeats - and the row key
        `(asset, specimen_index, trait)` is what keeps the second from overwriting the first.
        """
        with connection(self.db_path) as conn:
            profile = self._profile(conn)
            observation_id = repository.bench_observation(
                conn, file_path="/tmp/x.png", profile_id=profile.profile_id)
            asset = repository.bench_asset(conn, observation_id)
            for slot, end in ((1, (110.0, 20.0)), (2, (210.0, 20.0))):
                repository.save_measurement(
                    conn, observation_id=observation_id, asset_id=asset["asset_id"],
                    profile=profile, trait="length", specimen_index=slot,
                    p1=(10.0, 20.0), p2=end, individual_index=1)
            rows = queries.bench_measurements(conn)
        self.assertEqual(len(rows), 2, "one length erased the other")
        self.assertEqual({row["individual_index"] for row in rows}, {1})
        self.assertEqual(sorted(row["length_px"] for row in rows), [100.0, 200.0])

    def test_the_bench_export_keeps_the_landmark_coordinates(self):
        """A bench number whose points were thrown away cannot be defended."""
        with connection(self.db_path) as conn:
            self._bench_measurement(conn)
            path = Path(self._tmp.name) / "bench.csv"
            written = maintenance.export_bench_csv(conn, path)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(written, 1)
        for column in ("p1_x", "p1_y", "p2_x", "p2_y", "uncertainty_pct"):
            self.assertIn(column, lines[0])
        self.assertIn("10.000", lines[1])
        self.assertIn("/tmp/x.png", lines[1])

    def test_the_bench_export_carries_what_the_number_is_worth(self):
        """`total_uncertainty_um` travels, so an exported number keeps its budget.

        Hand-computed for this row: 100 px at 2 px/µm is 50 µm, and the budget is
        `sqrt((50*0.005)^2 + (2/2)^2 + (0.408/2)^2) = 1.0508 µm` — the scale term is
        0.25 µm and the *click* is 1.0 µm, i.e. four times larger
        (`docs/accuracy.md` §1, `units.total_uncertainty_um`).
        """
        with connection(self.db_path) as conn:
            self._bench_measurement(conn)
            path = Path(self._tmp.name) / "bench.csv"
            maintenance.export_bench_csv(conn, path)
        columns = path.read_text(encoding="utf-8").strip().splitlines()[0].split(",")
        self.assertIn("total_uncertainty_um", columns)
        values = path.read_text(encoding="utf-8").strip().splitlines()[1].split(",")
        row = dict(zip(columns, values))
        self.assertAlmostEqual(float(row["total_uncertainty_um"]), 1.0508, places=3)
        self.assertGreater(float(row["total_uncertainty_um"]),
                           float(row["uncertainty_pct"]))


class BenchPageTests(unittest.TestCase):
    """The screen itself: open, click, read - and save *this view* or lose it."""

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "bench-ui.db")
        init_db()
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        if cls._previous_db is None:
            os.environ.pop("DAPHNIA_DB", None)
        else:
            os.environ["DAPHNIA_DB"] = cls._previous_db

    def setUp(self):
        with connection() as conn:
            for table in ("corpus_labels", "corpus_boxes", "media_assets", "observations",
                          "imaging_sessions", "calibration_profiles", "studies"):
                conn.execute(f"DELETE FROM {table}")
        self.photo = Path(self._tmp.name) / "specimen.png"
        write_image(self.photo, width=200, height=150)
        self.page = Bench()

    def _calibrate(self, px_per_um=2.0):
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key="PROF_B", profile_name="b", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=px_per_um, uncertainty_pct=1.0)

    def _click(self, x, y):
        """A **real** click on the canvas, in widget pixels - the gesture, not `set_points`."""
        canvas = self.page.canvas
        canvas.resize(600, 400)
        point = QPointF(float(x), float(y))
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, point, point, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, point, point, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def _move(self, x, y):
        canvas = self.page.canvas
        canvas.resize(600, 400)
        point = QPointF(float(x), float(y))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, point, point, Qt.NoButton, Qt.NoButton, Qt.NoModifier))

    def _line(self, x1, y1, x2, y2):
        self._click(x1, y1)
        self._click(x2, y2)

    def test_the_page_is_three_steps(self):
        for name, word in (("image_group", "image"),
                           ("measure_group", "measure"),
                           ("results_group", "view")):
            self.assertTrue(hasattr(self.page, name), name)
            self.assertIn(word, getattr(self.page, name).title().lower())

    def test_there_is_no_mode_and_no_specimen_to_choose(self):
        """The Bench makes measurements without a predefined mode (`X-N14`)."""
        self.assertFalse(hasattr(self.page, "combo_mode"))
        self.assertFalse(hasattr(self.page, "combo_specimen"))

    def test_the_actions_are_menu_items(self):
        for key in ("line", "box", "undo", "clear",
                    "clear_view", "refresh",
                    "open_photo", "use_frame", "export"):
            self.assertIsNotNone(self.page.workspace.action(key), key)
        self.assertFalse(hasattr(self.page, "combo_action"))
        # Saving belongs beside the view it saves, not in the menu bar.
        self.assertTrue(hasattr(self.page, "btn_save"))
        self.assertTrue(hasattr(self.page, "btn_save_study"))

    def test_the_surface_displays_measurements_and_offers_no_model(self):
        """No *Model* switch and no *Keypoints* tool: this screen only shows measurements.

        Running a model is the Predict surface's job (`SR-PREDICT-018`) and one-keypoint-at-a-
        time labelling belongs to the corpus (Dataset), so neither control exists here - and
        the window that hosts this page carries no checkpoint menu either (`OFFERS_MODELS`).
        """
        self.assertIsNone(self.page.workspace.action("model"))
        self.assertIsNone(self.page.workspace.action("keypoints"))
        self.assertFalse(hasattr(self.page, "runner"))
        self.assertFalse(hasattr(self.page, "adopt_models"))
        self.assertNotIn("K", self.page.workspace.tool_shortcuts())
        self.assertEqual(BENCH_DRAW_TOOLS, ("line", "box"))

    def test_the_bench_opens_in_the_line_tool(self):
        self.assertEqual(self.page.canvas.tool, "line")
        self.assertEqual(list(self.page.canvas.point_names), ["A", "B"])
        self.assertTrue(self.page.workspace.action("line").isChecked())
        self.assertIn("click the a", self.page.prompt_label.text().lower())

    def test_an_image_can_be_opened_without_any_study(self):
        self.page.open_image(self.photo)
        self.assertIsNotNone(self.page._frame)
        self.assertIn(self.photo.name, self.page.source_label.text())

    def test_without_a_calibration_a_reading_is_labelled_unscaled(self):
        self.page.open_image(self.photo)
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        readout = self.page.length_label.text()
        self.assertIn("UNSCALED", readout)
        self.assertIn("100.0 px", readout)

    def test_two_clicks_join_the_view_and_store_nothing_yet(self):
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self._line(100, 100, 200, 100)
        self.assertEqual(len(self.page.session), 1, "the line is in this view")
        p1 = self.page.canvas._to_image(QPointF(100, 100))
        p2 = self.page.canvas._to_image(QPointF(200, 100))
        expected = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        self.assertAlmostEqual(self.page.session[0]["length_px"], expected)
        with connection() as conn:
            self.assertEqual(queries.bench_count(conn), 0, "nothing saved yet")
        self.assertIn("not saved yet", self.page.count_label.text().lower())
        self.assertTrue(self.page.unsaved_label.isVisibleTo(self.page))

    def test_a_second_line_becomes_line_two(self):
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self._line(100, 100, 200, 100)
        self._line(100, 160, 240, 160)
        self.assertEqual([item["specimen"] for item in self.page.session], [1, 2])
        # No box was drawn, so both lines are one group: the measurements that belong to
        # **no individual** (`SR-BENCH-016`) - and each of them is still listed.
        tree = self.page.list_results
        self.assertEqual(tree.topLevelItemCount(), 1)
        self.assertEqual(tree.topLevelItem(0).text(0), "No individual")
        self.assertEqual(tree.topLevelItem(0).childCount(), 2)

    def test_saving_without_a_calibration_explains_the_unit_problem(self):
        self.page.open_image(self.photo)
        self._line(100, 100, 200, 100)
        self.page._save_to_dataset()
        self.assertIn("no calibration", self.page.status_label.text().lower())
        with connection() as conn:
            self.assertEqual(queries.bench_count(conn), 0)
        self.assertEqual(len(self.page.session), 1, "the view is kept for after a calibrate")

    def test_save_this_view_to_the_dataset(self):
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self._line(100, 100, 200, 100)
        expected = self.page.session[0]["length_um"]
        self.page._save_to_dataset()
        with connection() as conn:
            rows = queries.bench_measurements(conn)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["length_um"], expected)   # the scale in force
        self.assertEqual(self.page.session, [], "saving clears the view")
        self.assertIn("dataset", self.page.status_label.text().lower())

    def test_several_lines_on_one_image_share_its_observation(self):
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self._line(100, 100, 200, 100)
        self._line(100, 160, 240, 160)
        self.page._save_to_dataset()
        with connection() as conn:
            observations = conn.execute(
                "SELECT COUNT(*) FROM observations").fetchone()[0]
            rows = queries.bench_measurements(conn)
        self.assertEqual(observations, 1)
        self.assertEqual(sorted(row["specimen_index"] for row in rows), [1, 2])

    def test_saving_a_view_to_a_study_attaches_the_photo(self):
        """A bench measurement for one photo can be saved straight onto a study."""
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        with connection() as conn:
            study_id = repository.create_study(
                conn, name="Adopted", destination=str(self._tmp.name))
        self.page.refresh()
        self.page.combo_study.setCurrentIndex(
            self.page.combo_study.findData(study_id))
        self._line(100, 100, 200, 100)
        expected = self.page.session[0]["length_um"]
        self.page._save_to_study()
        with connection() as conn:
            self.assertEqual(queries.bench_count(conn), 0, "it belongs to the study now")
            rows = queries.measurements_for_study(conn, study_id)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["length_um"], expected)

    def test_saving_to_a_study_without_one_says_what_to_do(self):
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self._line(100, 100, 200, 100)
        self.page._save_to_study()
        self.assertIn("no study", self.page.status_label.text().lower())
        self.assertEqual(len(self.page.session), 1)

    def test_each_image_keeps_its_own_view_and_coming_back_restores_it(self):
        """A view belongs to an image: switching frames switches the view (`SR-BENCH-022`)."""
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self._line(100, 100, 200, 100)
        second = Path(self._tmp.name) / "other.png"
        write_image(second, width=200, height=150)
        self.page.open_image(second)
        self.assertEqual(self.page.session, [], "the second image has its own, empty view")
        self.page.open_image(self.photo)
        self.assertEqual(len(self.page.session), 1, "the first image's view came back")
        self.assertIn("already carries 1", self.page.status_label.text())

    def test_clear_view_discards_this_view_only(self):
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self._line(100, 100, 200, 100)
        self.page._save_to_dataset()
        self._line(100, 160, 200, 160)
        self.page.clear_view()
        self.assertEqual(self.page.session, [])
        with connection() as conn:
            self.assertEqual(queries.bench_count(conn), 1, "the saved row stays")

    # -- a line stays editable in this view (`X-N16`) ----------------------
    def _drag(self, from_point, dx=12.0, dy=10.0):
        canvas = self.page.canvas
        canvas.resize(600, 400)
        start = QPointF(*from_point)
        end = QPointF(start.x() + dx, start.y() + dy)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, start, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, start, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def test_hovering_a_view_line_offers_a_grab(self):
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self._line(100, 100, 200, 100)
        p1 = self.page.session[0]["p1"]
        at = self.page.canvas._to_screen(p1)
        self._move(at[0], at[1])
        # No box was drawn, so the line belongs to **no individual** - the canvas numbers that
        # group 0 (`SR-BENCH-016`), and the segment is its own count within it.
        self.assertEqual(self.page.canvas.hover, ("annotation", 0))

    def test_a_view_line_is_editable_before_it_is_saved(self):
        """A line drawn a moment ago is corrected by dragging its end - still unsaved."""
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self._line(100, 100, 200, 100)
        before = self.page.session[0]["p1"]
        self._drag(self.page.canvas._to_screen(before))
        self.assertEqual(len(self.page.session), 1, "updated in place, never duplicated")
        self.assertNotEqual(self.page.session[0]["p1"], before)
        with connection() as conn:
            self.assertEqual(queries.bench_count(conn), 0, "still nothing saved")

    # -- image mode: the frozen frame, not the video (`X-N16`) -------------
    def test_taking_a_live_frame_switches_to_image_mode(self):
        frame = np.zeros((150, 200, 3), dtype="uint8")
        service = self.page.service
        original = service.current_frame
        service.current_frame = lambda: frame
        self.addCleanup(setattr, service, "current_frame", original)
        self.page.use_live_frame()
        self.assertFalse(self.page.stream.preview_enabled)
        self.assertIsNotNone(self.page.canvas.frame)
        self.assertFalse(self.page.workspace.action("live_preview").isChecked())

    def test_opening_a_photo_goes_to_image_mode(self):
        self.page.open_image(self.photo)
        self.assertFalse(self.page.stream.preview_enabled)
        self.assertFalse(self.page.workspace.action("live_preview").isChecked())

    def test_the_live_preview_can_be_turned_back_on(self):
        self.page.open_image(self.photo)
        self.page.workspace.action("live_preview").setChecked(True)
        self.assertTrue(self.page.stream.preview_enabled)

    def test_every_tool_wears_its_key(self):
        """Photoshop's rule: a tool says how to reach it (`X-N18`)."""
        for tool, key in (("line", "L"), ("box", "B")):
            self.assertIn(f"({key})", self.page.workspace.action(tool).text())
            self.assertIn(key, self.page.workspace.tool_shortcuts())

    def test_an_edit_key_is_not_a_tool_because_editing_is_not_a_mode(self):
        """**Touching an object is always available** (`X-N20`): there is no *Select* tool."""
        self.assertIsNone(self.page.workspace.action("select"))
        self.assertNotIn("V", self.page.workspace.tool_shortcuts())
        self.assertEqual(BENCH_DRAW_TOOLS, ("line", "box"),
                         "the tools are the two ways of drawing a measurement")
        self.assertTrue(self.page.canvas.editing,
                        "click-to-edit stays on whatever draw tool is armed")

    def test_the_draw_tools_float_on_the_image(self):
        """The tools are on the canvas as well as in the menu (`X-N20`) - one action, twice."""
        bar = self.page.workspace.draw_bar
        self.assertIsNotNone(bar, "the Bench offers its draw tools on the picture")
        texts = [button.text() for button in bar.buttons]
        self.assertEqual(len(bar.buttons), 2)
        for key in ("(L)", "(B)"):
            self.assertTrue(any(key in text for text in texts), key)
        self.assertTrue(bar.buttons[0].isChecked(), "the bar shows the tool in force")

    def test_the_bar_shows_the_tool_the_key_armed(self):
        """A button on the picture *is* the menu item: the two never disagree."""
        self.page.workspace.tool_shortcuts()["B"].activated.emit()
        texts = {button.text(): button for button in self.page.workspace.draw_bar.buttons}
        box = next(button for text, button in texts.items() if "(B)" in text)
        line = next(button for text, button in texts.items() if "(L)" in text)
        self.assertTrue(box.isChecked())
        self.assertFalse(line.isChecked())
        self.assertTrue(self.page.workspace.action("box").isChecked())

    def test_a_key_switches_the_tool(self):
        """`B` reaches the Box tool, `L` the measuring line - without a trip to the menu."""
        shortcuts = self.page.workspace.tool_shortcuts()
        self.assertEqual(self.page.canvas.tool, "line")
        shortcuts["B"].activated.emit()
        self.assertEqual(self.page.canvas.tool, "box")
        self.assertTrue(self.page.workspace.action("box").isChecked())
        shortcuts["L"].activated.emit()
        self.assertEqual(self.page.canvas.tool, "line")

    # -- a tool can be turned **off**: the normal view (`X-N20`) -----------
    def test_the_armed_tool_can_be_turned_off_and_that_is_the_normal_view(self):
        """Clicking the armed tool again is a state the operator asks for, not a mistake.

        It used to be undone on the spot ("a radio group with nothing selected would leave the
        press undefined") - which left no way to be *in* a view where a click only selects.
        """
        self.assertEqual(self.page.canvas.tool, "line")
        self.page.workspace.action("line").setChecked(False)
        self.assertFalse(self.page.workspace.action("line").isChecked())
        self.assertEqual(self.page.canvas.tool, "select", "no tool armed is the normal view")
        self.assertIn("no tool", self.page.prompt_label.text().lower())
        self.assertIn("normal view", self.page.keypoint_label.text().lower())

    def test_the_key_of_the_armed_tool_turns_it_off(self):
        """*No tool* has to be one key away too (`X-N18`), or nobody ever returns to it."""
        shortcuts = self.page.workspace.tool_shortcuts()
        shortcuts["B"].activated.emit()
        self.assertEqual(self.page.canvas.tool, "box")
        shortcuts["B"].activated.emit()
        self.assertEqual(self.page.canvas.tool, "select")
        self.assertFalse(self.page.workspace.action("box").isChecked())

    def test_the_normal_view_draws_nothing(self):
        """A press in the normal view never makes a measurement - it selects (`X-N20`)."""
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self.page.workspace.action("line").setChecked(False)
        self._click(100, 100)
        self._click(220, 100)
        self.assertEqual(self.page.session, [], "the normal view drew a measurement")

    def test_in_the_normal_view_a_press_selects_and_arms_nothing(self):
        """`X-N19`'s tool follows an object **when a draw tool is armed** (`X-N20`).

        In the normal view there is no tool for it to switch *to* - and switching one on would
        make the normal view last exactly one click.
        """
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)
        self._line(100, 100, 220, 100)
        self.assertEqual(self.page.canvas.tool, "line")
        self.page.workspace.action("line").setChecked(False)
        canvas = self.page.canvas
        canvas.clear_selection()
        p1 = canvas._to_screen(self.page.session[0]["p1"])
        self._click(p1[0], p1[1])                       # a press on the measurement
        self.assertEqual(self.page.canvas.tool, "select", "the normal view armed a tool again")
        self.assertEqual(len(self.page.session), 1, "and made nothing")


if __name__ == "__main__":
    unittest.main()


class BenchBoxTests(unittest.TestCase):
    """A box is a Bench measurement too - drawn, edited, saved, deleted (`SR-BENCH-015`)."""

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "bench-box.db")
        init_db()
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        if cls._previous_db is None:
            os.environ.pop("DAPHNIA_DB", None)
        else:
            os.environ["DAPHNIA_DB"] = cls._previous_db

    def setUp(self):
        with connection() as conn:
            for table in ("corpus_labels", "corpus_boxes", "media_assets", "observations",
                          "imaging_sessions", "calibration_profiles", "studies"):
                conn.execute(f"DELETE FROM {table}")
        self.photo = Path(self._tmp.name) / "specimen.png"
        write_image(self.photo, width=200, height=150)
        self.page = Bench()

    def _calibrate(self, px_per_um=2.0):
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key="PROF_B", profile_name="b", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=px_per_um, uncertainty_pct=1.0)

    def _open(self):
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.open_image(self.photo)

    def _press_release(self, point):
        canvas = self.page.canvas
        canvas.resize(600, 400)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, point, point, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, point, point, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def _drag(self, start, end):
        canvas = self.page.canvas
        canvas.resize(600, 400)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, start, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, start, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def _draw_box(self, x1, y1, x2, y2):
        self.page.workspace.action("box").setChecked(True)
        self._drag(QPointF(float(x1), float(y1)), QPointF(float(x2), float(y2)))

    def test_a_drawn_box_is_a_measurement_with_width_height_and_area(self):
        self._open()
        self._draw_box(100, 100, 300, 250)
        self.assertEqual(len(self.page.session), 1)
        item = self.page.session[0]
        self.assertEqual(item["kind"], "box")
        self.assertEqual(item["trait"], "bbox")
        self.assertGreater(item["width_px"], 0)
        self.assertGreater(item["height_px"], 0)
        self.assertAlmostEqual(item["area_px"], item["width_px"] * item["height_px"])
        self.assertAlmostEqual(item["width_um"], item["width_px"] / 2.0)
        # It is stored as a `bbox` row **and** it is the individual the numbers inside it
        # are grouped under, so the list heads a group with its own size as the group's fact.
        header = self.page.list_results.topLevelItem(0)
        self.assertEqual(header.text(0), "Individual 1")
        self.assertIn("box", header.text(1).lower())

    def test_saving_a_box_writes_its_geometry(self):
        self._open()
        self._draw_box(100, 100, 300, 250)
        item = self.page.session[0]
        self.page._save_to_dataset()
        with connection() as conn:
            row = queries.bench_measurements(conn)[0]
        self.assertEqual(row["trait"], "bbox")
        self.assertAlmostEqual(row["width_px"], item["width_px"])
        self.assertAlmostEqual(row["height_px"], item["height_px"])
        self.assertAlmostEqual(row["area_px"], item["area_px"])
        self.assertAlmostEqual(row["width_um"], item["width_um"])

    def test_a_view_box_can_be_moved_before_it_is_saved(self):
        self._open()
        self._draw_box(100, 100, 300, 250)
        item = self.page.session[0]
        before = item["p1"]
        canvas = self.page.canvas
        canvas.resize(600, 400)
        centre = canvas._screen_box((item["p1"][0], item["p1"][1],
                                     item["p2"][0], item["p2"][1])).center()
        self._drag(QPointF(centre.x(), centre.y()),
                   QPointF(centre.x() + 20, centre.y() + 10))
        self.assertEqual(len(self.page.session), 1, "updated in place")
        self.assertNotEqual(self.page.session[0]["p1"], before)
        with connection() as conn:
            self.assertEqual(queries.bench_count(conn), 0)

    def test_backspace_removes_a_view_measurement(self):
        self._open()
        self.page.canvas.resize(600, 400)
        self.page.canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(100, 100), QPointF(100, 100),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.page.canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, QPointF(100, 100), QPointF(100, 100),
            Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
        self.page.canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(200, 100), QPointF(200, 100),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.page.canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, QPointF(200, 100), QPointF(200, 100),
            Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
        self.assertEqual(len(self.page.session), 1)
        p1 = self.page.canvas._to_screen(self.page.session[0]["p1"])
        self._press_release(QPointF(p1[0], p1[1]))     # click the drawn line
        self.page.canvas.delete_selection()            # Backspace
        self.assertEqual(self.page.session, [], "the measurement is gone from the view")

    def test_backspace_removes_a_view_box(self):
        """A box is deleted like a line: grab it by its outline, `Backspace`, it leaves."""
        self._open()
        self._draw_box(100, 100, 300, 250)
        canvas = self.page.canvas
        canvas.clear_selection()               # start from nothing selected
        canvas.resize(600, 400)
        box = canvas.boxes[0][1]
        self._press_release(QPointF(canvas._screen_box(box).topLeft()))   # the outline
        self.assertEqual(canvas.selected_boxes, [1])
        canvas.delete_selection()
        self.assertEqual(self.page.session, [], "the box is gone from the view")
        self.assertEqual(canvas.boxes, [])

    def test_a_bench_box_is_grabbed_by_its_outline_not_its_inside(self):
        """A box is a measurement with things **inside** it: only its line picks the box."""
        self._open()
        self._draw_box(100, 100, 300, 250)
        canvas = self.page.canvas
        canvas.clear_selection()               # a fresh box is selected; drop that first
        canvas.resize(600, 400)
        screen = canvas._screen_box(canvas.boxes[0][1])
        self._press_release(QPointF(screen.center()))
        self.assertEqual(canvas.selected_boxes, [],
                         "a press deep inside the box does not pick the box")
        self._press_release(QPointF(screen.topLeft()))
        self.assertEqual(canvas.selected_boxes, [1], "coming close to its line does")

    def test_a_freshly_drawn_box_is_the_selection_so_backspace_removes_it(self):
        """The object just made is what `Backspace` takes - no extra click first."""
        self._open()
        self._draw_box(100, 100, 300, 250)
        self.assertEqual(self.page.canvas.selected_boxes, [1],
                         "the box just drawn is selected")
        self.assertTrue(self.page.canvas.delete_selection())
        self.assertEqual(self.page.session, [])

    def test_a_freshly_made_line_is_the_selection_so_backspace_removes_it(self):
        self._open()
        self._press_release(QPointF(100, 100))
        self._press_release(QPointF(220, 100))
        # Drawn outside every box: the canvas numbers it under the *no individual* group, 0.
        self.assertEqual(self.page.canvas.selected_annotations, [(0, 0)],
                         "the line just made is selected")
        self.assertTrue(self.page.canvas.delete_selection())
        self.assertEqual(self.page.session, [])

    def test_clicking_a_box_switches_to_the_box_tool(self):
        """The tool follows the object (`X-N19`): a click on a box arms the Box tool."""
        self._open()
        self._draw_box(100, 100, 300, 250)
        self.page.workspace.action("line").setChecked(True)      # leave Box mode
        self.assertEqual(self.page.canvas.tool, "line")
        canvas = self.page.canvas
        canvas.resize(600, 400)
        box = canvas.boxes[0][1]
        self._press_release(QPointF(canvas._screen_box(box).topLeft()))
        self.assertEqual(self.page.canvas.tool, "box", "the box armed the Box tool")
        self.assertEqual(canvas.selected_boxes, [1], "and kept the press's own selection")

    def test_clicking_a_measurement_switches_to_the_line_tool(self):
        """…and a click on a measurement arms the measuring tool, from Box mode."""
        self._open()
        self._press_release(QPointF(100, 100))
        self._press_release(QPointF(220, 100))
        self.page.workspace.action("box").setChecked(True)
        self.assertEqual(self.page.canvas.tool, "box")
        canvas = self.page.canvas
        canvas.resize(600, 400)
        p1 = canvas._to_screen(self.page.session[0]["p1"])
        self._press_release(QPointF(p1[0], p1[1]))
        self.assertEqual(self.page.canvas.tool, "line", "the measurement armed the line tool")


class BenchIndividualTests(unittest.TestCase):
    """A box is an **individual**, and the measurements inside it are its own (`SR-BENCH-016`).

    The grouping rule itself is proven in `tests/test_individuals.py`; these tests are the
    surface: what the operator draws, what the list shows, and what the store records.
    """

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "bench-individual.db")
        init_db()
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        if cls._previous_db is None:
            os.environ.pop("DAPHNIA_DB", None)
        else:
            os.environ["DAPHNIA_DB"] = cls._previous_db

    def setUp(self):
        with connection() as conn:
            for table in ("corpus_labels", "corpus_boxes", "media_assets", "observations",
                          "imaging_sessions", "calibration_profiles", "studies"):
                conn.execute(f"DELETE FROM {table}")
        self.photo = Path(self._tmp.name) / "specimen.png"
        write_image(self.photo, width=200, height=150)
        self.page = Bench()
        self._calibrate()
        self.page.refresh()
        self.page.open_image(self.photo)

    def _calibrate(self, px_per_um=2.0):
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key="PROF_I", profile_name="i", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=px_per_um, uncertainty_pct=1.0)

    # -- the gestures, in **image** coordinates (what the operator means) ---
    def _press_release(self, point):
        canvas = self.page.canvas
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, point, point, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, point, point, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def _at(self, point):
        """The widget pixel an **image** coordinate is drawn at."""
        self.page.canvas.resize(600, 400)
        at = self.page.canvas._to_screen(point)
        return QPointF(float(at[0]), float(at[1]))

    def _box_at(self, x1, y1, x2, y2):
        """Drag a box at **image** coordinates - the individual it stands for."""
        self.page.workspace.action("box").setChecked(True)
        canvas = self.page.canvas
        canvas.resize(600, 400)
        start, end = self._at((x1, y1)), self._at((x2, y2))
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, start, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, start, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def _line_at(self, p1, p2):
        """Two clicks at **image** coordinates, with the *Line* tool armed."""
        self.page.workspace.action("line").setChecked(True)
        self._press_release(self._at(p1))
        self._press_release(self._at(p2))

    def _group_titles(self):
        tree = self.page.list_results
        return [tree.topLevelItem(row).text(0) for row in range(tree.topLevelItemCount())]

    # -- the grouping -------------------------------------------------------
    def test_a_line_inside_a_box_is_that_individuals_measurement(self):
        self._box_at(10, 10, 100, 90)
        self._line_at((30, 40), (80, 60))
        self.assertEqual(self.page._individual_index(self.page.session[1]), 1)
        tree = self.page.list_results
        self.assertEqual(self._group_titles(), ["Individual 1"])
        header = tree.topLevelItem(0)
        self.assertEqual(header.childCount(), 1, "the line is listed under the individual")
        self.assertIn("Length", header.child(0).text(0))
        self.assertIn("um", header.child(0).text(1))

    def test_a_line_outside_every_box_is_listed_as_belonging_to_nobody(self):
        """Kept and shown - never attached to the nearest animal (`mlContext.md` law 8)."""
        self._box_at(10, 10, 60, 60)
        self._line_at((130, 110), (180, 140))
        self.assertIsNone(self.page._individual_index(self.page.session[1]))
        self.assertEqual(self._group_titles(), ["Individual 1", "No individual"])
        stray = self.page.list_results.topLevelItem(1)
        self.assertEqual(stray.childCount(), 1)
        self.assertIn("measurement(s)", stray.text(1))

    def test_two_boxes_are_two_individuals_with_their_own_numbers(self):
        """The list that was asked for: every individual, and its own measurements."""
        self._box_at(5, 5, 60, 55)
        self._box_at(120, 90, 190, 145)
        self._line_at((15, 20), (50, 20))
        self._line_at((130, 100), (180, 130))
        self.assertEqual(self._group_titles(), ["Individual 1", "Individual 2"])
        self.assertEqual(self.page._individual_index(self.page.session[2]), 1)
        self.assertEqual(self.page._individual_index(self.page.session[3]), 2)
        tree = self.page.list_results
        self.assertEqual(tree.topLevelItem(0).childCount(), 1)
        self.assertEqual(tree.topLevelItem(1).childCount(), 1)

    def test_the_grouping_follows_the_box_when_it_is_moved(self):
        """The grouping is **derived** from the geometry, so it can never go stale.

        The drag itself is covered by `BenchBoxTests`; here the page's own handler for a
        moved box is driven, and the question is whether the *grouping* follows it.
        """
        self._box_at(10, 10, 100, 90)
        self._line_at((30, 40), (80, 60))
        self.assertEqual(self.page._individual_index(self.page.session[1]), 1)
        self.page._on_box_edited(1, (140.0, 100.0, 190.0, 145.0))
        self.assertIsNone(self.page._individual_index(self.page.session[1]),
                          "the line is no longer inside any box")
        self.assertEqual(self._group_titles(), ["Individual 1", "No individual"])

    def test_the_individuals_are_numbered_one_to_n_with_lines_between_them(self):
        """A box and its length each take a row slot; the **individual** number is its own.

        Measured on a real frame: accepting each animal box-then-length gave *Individual 1, 3,
        5 ...*, because every line consumed a number and the individual was read off the row's
        slot (`SR-BENCH-016`).
        """
        self._box_at(5, 5, 60, 55)
        self._line_at((15, 20), (50, 20))
        self._box_at(120, 90, 190, 145)
        self._line_at((130, 100), (180, 130))
        self.assertEqual([item["specimen"] for item in self.page.session], [1, 2, 3, 4],
                         "every row keeps its own slot - that is the row key")
        self.assertEqual(self._group_titles(), ["Individual 1", "Individual 2"])
        self.assertEqual(self.page._individual_index(self.page.session[1]), 1)
        self.assertEqual(self.page._individual_index(self.page.session[3]), 2)

    def test_the_row_slot_and_the_individual_are_recorded_apart(self):
        """`specimen_index` is the row; `individual_index` is the animal (`SR-BENCH-016`)."""
        self._box_at(5, 5, 60, 55)
        self._line_at((15, 20), (50, 20))
        self._box_at(120, 90, 190, 145)
        self._line_at((130, 100), (180, 130))
        self.page._save_to_dataset()
        with connection() as conn:
            rows = {(row["trait"], row["specimen_index"]): row
                    for row in queries.bench_measurements(conn)}
        self.assertEqual(rows[("bbox", 3)]["individual_index"], 2,
                         "the second box is Individual 2, at its own row slot 3")
        self.assertEqual(rows[("length", 4)]["individual_index"], 2)
        self.assertEqual(rows[("length", 2)]["individual_index"], 1)

    def test_the_readout_names_the_individual_the_line_will_belong_to(self):
        """Said **while** it is placed, so a stray line is not a surprise in the list."""
        self._box_at(10, 10, 100, 90)
        self.page.workspace.action("line").setChecked(True)
        self.page.canvas.set_points([(30.0, 40.0), (80.0, 60.0)])
        self.assertIn("Individual 1", self.page.length_label.text())

    # -- one numbering, on the picture and in the panel ---------------------
    def test_the_canvas_numbers_an_individual_and_its_measurements_alike(self):
        """The box and the lines inside it wear the **same** number and the same colour."""
        self._box_at(5, 5, 120, 120)
        self._line_at((20, 40), (60, 40))
        self._line_at((20, 80), (100, 80))
        canvas = self.page.canvas
        self.assertEqual([index for index, _box in canvas.boxes], [1],
                         "the box is drawn as Individual 1")
        drawn = canvas.annotations
        self.assertEqual(len(drawn), 1, "one annotation item per individual")
        index, segments = drawn[0]
        self.assertEqual(index, 1, "its measurements are numbered with it, not by row")
        self.assertEqual(len(segments), 2)

    def test_every_measured_line_carries_its_own_label(self):
        """Not one sentence for the animal: **each line** says what it is, where it is.

        One animal's two numbers used to be collected into a single label at its first point
        (`1: 120.0 um, 250.0 um`), which made the second number impossible to find on the
        picture. Each line now carries its own (`SR-BENCH-024`).
        """
        self._box_at(5, 5, 120, 120)
        self._line_at((20, 40), (60, 40))
        self._line_at((20, 80), (100, 80))
        labels = dict(self.page.canvas.annotation_texts)[1]
        self.assertEqual(len(labels), 2, "one label per line")
        self.assertEqual([label.split()[0] for label in labels], ["1.1", "1.2"],
                         "each in the numbering the panel lists it under")
        for item, label in zip(self.page.session[1:], labels):
            self.assertIn(self.page._compact_value(item), label)
        self.assertEqual(self.page._line_label(self.page.session[1]), labels[0],
                         "the label above a line is the line's own label")

    def test_a_measurement_outside_every_box_is_numbered_zero(self):
        """A stray line is drawn and numbered `0.1` - not captioned *no individual*.

        ``None`` stays a real answer in the store and the panel's group (`SR-BENCH-016`), but
        the words are not written over the photograph: the picture shows a drawn, numbered line
        and the panel is where the grouping is read.
        """
        self._box_at(5, 5, 60, 55)
        self._line_at((130, 110), (180, 140))
        canvas = self.page.canvas
        self.assertEqual(sorted(index for index, _segments in canvas.annotations),
                         [NO_INDIVIDUAL])
        self.assertEqual(canvas.boxes, [(1, canvas.boxes[0][1])])
        self.assertEqual(self.page._row_number(self.page.session[1]), "0.1")
        stray = dict(canvas.annotation_texts)[NO_INDIVIDUAL][0]
        self.assertEqual(stray.split()[0], "0.1")
        self.assertNotIn("no individual", stray.lower())
        self.assertNotIn("no individual", " ".join(
            text for _i, _segments, text in canvas.annotations_with_text).lower())

    def test_the_panel_numbers_each_measurement_inside_its_individual(self):
        """``1.1``, ``1.2``: the same pair the canvas draws for that individual."""
        self._box_at(5, 5, 120, 120)
        self._line_at((20, 40), (60, 40))
        self._line_at((20, 80), (100, 80))
        header = self.page.list_results.topLevelItem(0)
        self.assertEqual(header.text(0), "Individual 1")
        self.assertEqual([header.child(row).text(0).split()[0] for row in range(2)],
                         ["1.1", "1.2"])
        self.assertEqual([header.child(row).text(1) for row in range(2)],
                         [self.page._value_text(item) for item in self.page.session[1:]])

    def test_a_panel_row_wears_the_colour_the_canvas_draws(self):
        """Colour is the identity the picture already uses, so a row is recognisable as it."""
        self._box_at(5, 5, 60, 55)
        self._line_at((15, 20), (50, 20))
        header = self.page.list_results.topLevelItem(0)
        wanted = QColor(specimen_color(1)).name()
        self.assertEqual(header.foreground(0).color().name(), wanted)
        self.assertEqual(header.child(0).foreground(0).color().name(), wanted)

    # -- what is stored -----------------------------------------------------
    def test_saving_records_which_individual_each_number_is_about(self):
        self._box_at(10, 10, 100, 90)
        self._line_at((30, 40), (80, 60))
        self.page._save_to_dataset()
        with connection() as conn:
            rows = {row["trait"]: row for row in queries.bench_measurements(conn)}
        self.assertEqual(rows["bbox"]["individual_index"], 1, "a box is its own individual")
        self.assertEqual(rows["length"]["individual_index"], 1)

    def test_two_lengths_on_one_individual_both_survive_the_save(self):
        """An animal may carry two numbers of the same trait - neither may erase the other."""
        self._box_at(10, 10, 150, 120)
        self._line_at((20, 30), (60, 30))
        self._line_at((20, 80), (110, 80))
        self.assertEqual(len(self.page._grouped()[0][2]), 2)
        self.page._save_to_dataset()
        with connection() as conn:
            rows = queries.bench_measurements(conn)
        lengths = [row for row in rows if row["trait"] == "length"]
        self.assertEqual(len(lengths), 2, "the second length erased the first")
        self.assertEqual({row["individual_index"] for row in lengths}, {1})

    def test_the_export_says_whose_number_each_row_is(self):
        """A bench number without its individual cannot be checked (`SR-BENCH-010`)."""
        self._box_at(10, 10, 100, 90)
        self._line_at((30, 40), (80, 60))
        self.page._save_to_dataset()
        path = Path(self._tmp.name) / "bench.csv"
        with connection() as conn:
            maintenance.export_bench_csv(conn, path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertIn("individual_index", lines[0])
        for row in lines[1:]:
            self.assertIn(",1,", row, "every row of this view belongs to Individual 1")


class BenchNameTests(unittest.TestCase):
    """A measurement can be **named** (`SR-BENCH-024`): the picture shows the name you gave it.

    The name is the lab's own word for a number ('Gut length', 'width'), typed into that row's
    *Name* cell. It is written above the line it belongs to, it is what the panel and the
    picture agree on, and it is **saved with the row** - a name the operator typed and lost by
    saving would be work that never existed.
    """

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "bench-name.db")
        init_db()
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        if cls._previous_db is None:
            os.environ.pop("DAPHNIA_DB", None)
        else:
            os.environ["DAPHNIA_DB"] = cls._previous_db

    def setUp(self):
        with connection() as conn:
            for table in ("corpus_labels", "corpus_boxes", "media_assets", "observations",
                          "imaging_sessions", "calibration_profiles", "studies"):
                conn.execute(f"DELETE FROM {table}")
        self.photo = Path(self._tmp.name) / "specimen.png"
        write_image(self.photo, width=200, height=150)
        self.page = Bench()
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key="PROF_N", profile_name="n", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=2.0, uncertainty_pct=1.0)
        self.page.refresh()
        self.page.open_image(self.photo)

    # -- the gestures and the cell the operator types into ------------------
    def _press_release(self, point):
        canvas = self.page.canvas
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, point, point, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, point, point, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def _at(self, point):
        self.page.canvas.resize(600, 400)
        at = self.page.canvas._to_screen(point)
        return QPointF(float(at[0]), float(at[1]))

    def _line_at(self, p1, p2):
        """Two clicks at **image** coordinates, with the *Line* tool armed."""
        self.page.workspace.action("line").setChecked(True)
        self._press_release(self._at(p1))
        self._press_release(self._at(p2))

    def _box_at(self, x1, y1, x2, y2):
        """Drag a box at **image** coordinates - the individual it stands for."""
        self.page.workspace.action("box").setChecked(True)
        canvas = self.page.canvas
        canvas.resize(600, 400)
        start, end = self._at((x1, y1)), self._at((x2, y2))
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, start, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, start, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def _measurement_row(self, row=0):
        """The tree row of one **measurement** of the view (not the group's header row)."""
        return self.page.list_results.topLevelItem(0).child(row)

    def _type(self, item, text):
        """Type a name into a row the way the operator does: the cell commits, the page reads it."""
        item.setText(NAME_COLUMN, text)       # what a committed editor leaves behind
        self.page._on_row_renamed(item, NAME_COLUMN)
        return item

    # -- what the operator can do -------------------------------------------
    def test_the_panel_offers_a_name_for_every_measurement(self):
        """A third column, per row - and it is the **only** cell an editor opens over."""
        self.assertEqual(self.page.list_results.columnCount(), 3)
        self.assertEqual(self.page.list_results.headerItem().text(NAME_COLUMN), "Name")
        self._line_at((30, 40), (80, 60))
        row = self._measurement_row()
        self.assertEqual(row.text(NAME_COLUMN), "", "a fresh measurement has no name yet")
        self.assertTrue(row.flags() & Qt.ItemIsEditable)
        # ... and only there: the delegate refuses an editor over the number and the value,
        # which is what keeps a double click on a report from putting a box over it.
        view = self.page.list_results
        option, model = QStyleOptionViewItem(), view.model()
        self.assertIsNone(view.itemDelegate().createEditor(
            view, option, model.index(0, 0)), "the number is a report, not a field")
        self.assertIsNotNone(view.itemDelegate().createEditor(
            view, option, model.index(0, NAME_COLUMN)))

    def test_naming_a_row_is_written_above_that_line(self):
        """The name the operator types is what the picture shows for that measurement."""
        self._line_at((30, 40), (80, 60))
        self._type(self._measurement_row(), "Gut length")
        self.assertEqual(self.page.session[0]["name"], "Gut length")
        label = self.page.canvas.annotation_texts[0][1][0]
        self.assertIn("Gut length", label)
        self.assertIn(self.page._compact_value(self.page.session[0]), label,
                      "a name replaces the number, never the value")
        self.assertIn("Gut length", self.page.status_label.text())

    def test_only_the_line_that_was_named_reads_by_its_name(self):
        """One animal's two numbers: the named one by its name, the other by its number."""
        self._box_at(5, 5, 120, 120)
        self._line_at((20, 40), (60, 40))
        self._line_at((20, 80), (100, 80))
        self._type(self._measurement_row(1), "Gut length")
        labels = self.page.canvas.annotation_texts[0][1]
        self.assertTrue(labels[0].startswith("1.1"), labels[0])
        self.assertTrue(labels[1].startswith("Gut length:"), labels[1])

    def test_an_emptied_name_puts_the_number_back(self):
        """Clearing the cell is allowed too - and the measurement reads by its number again."""
        self._line_at((30, 40), (80, 60))
        row = self._measurement_row()
        self._type(row, "Gut length")
        self._type(row, "")
        self.assertEqual(self.page.session[0]["name"], "")
        self.assertTrue(self.page.canvas.annotation_texts[0][1][0].startswith("0.1"))

    # -- and it is stored ---------------------------------------------------
    def test_the_name_is_saved_with_the_row(self):
        """Renaming is a correction of the same measurement - never a second row."""
        self._line_at((30, 40), (80, 60))
        self._type(self._measurement_row(), "Gut length")
        self.page._save_to_dataset()
        with connection() as conn:
            rows = queries.bench_measurements(conn)
        self.assertEqual(len(rows), 1, "the name forked the measurement")
        self.assertEqual(rows[0]["name"], "Gut length")

    def test_the_name_travels_in_the_export(self):
        """A name the operator typed and the CSV dropped would be work that vanished."""
        self._line_at((30, 40), (80, 60))
        self._type(self._measurement_row(), "Gut length")
        self.page._save_to_dataset()
        path = Path(self._tmp.name) / "bench.csv"
        with connection() as conn:
            maintenance.export_bench_csv(conn, path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        columns = lines[0].split(",")
        self.assertIn("name", columns)
        self.assertEqual(dict(zip(columns, lines[1].split(",")))["name"], "Gut length")


class BenchDatasetTests(unittest.TestCase):
    """The Bench stores in a **dataset** - the same store every entry point reads."""

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "bench-dataset.db")
        init_db()
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        if cls._previous_db is None:
            os.environ.pop("DAPHNIA_DB", None)
        else:
            os.environ["DAPHNIA_DB"] = cls._previous_db

    def setUp(self):
        with connection() as conn:
            for table in ("corpus_labels", "corpus_boxes", "media_assets",
                          "observations", "imaging_sessions", "calibration_profiles",
                          "studies", "datasets", "imported_frames", "import_batches"):
                conn.execute(f"DELETE FROM {table}")
        self.photo = Path(self._tmp.name) / "specimen.png"
        write_image(self.photo, width=200, height=150)
        self.page = Bench()

    def test_the_bench_opens_a_dataset_and_offers_the_others(self):
        """No 'bench archive': the surface works in a dataset, and can make another."""
        self.assertIsNotNone(self.page.dataset())
        self.assertGreaterEqual(self.page.combo_dataset.count(), 1)
        created = self.page.create_dataset(name="Pose 4-point", schema_key="pose_full")
        self.assertEqual(self.page.dataset_id(), created)
        self.assertIn("Pose 4-point", self.page.combo_dataset.currentText())

    def test_choosing_another_dataset_makes_it_the_open_one(self):
        first = self.page.dataset_id()
        second = self.page.create_dataset(name="Boxes", schema_key="boxes")
        self.assertNotEqual(first, second)
        self.page.select_dataset(first)
        with connection() as conn:
            current = queries.current_dataset(conn)
        self.assertEqual(current["dataset_id"], first)

    def test_a_saved_view_lands_in_the_open_dataset(self):
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key="PROF_B", profile_name="b", microscope_model="m",
                objective_magnification="2x", camera_device="c", px_per_um=2.0)
        self.page.refresh()
        dataset_id = self.page.dataset_id()
        self.page.open_image(str(self.photo))
        self.page.session = [{"kind": "line", "specimen": 1, "trait": "length",
                              "p1": (10.0, 10.0), "p2": (110.0, 10.0)}]
        self.page._save_to_dataset()
        with connection() as conn:
            rows = queries.dataset_labels(conn, dataset_id)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["length_um"], 50.0)      # 100 px / 2 px per um
        self.assertEqual(rows[0]["dataset_id"], dataset_id)


class BenchImportTests(unittest.TestCase):
    """A folder or a video brings in many frames, and the import is **recorded**."""

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "bench-import.db")
        init_db()
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        if cls._previous_db is None:
            os.environ.pop("DAPHNIA_DB", None)
        else:
            os.environ["DAPHNIA_DB"] = cls._previous_db

    def setUp(self):
        with connection() as conn:
            for table in ("corpus_labels", "corpus_boxes", "imported_frames",
                          "import_batches", "datasets"):
                conn.execute(f"DELETE FROM {table}")
        self.folder = Path(self._tmp.name) / "frames"
        self.folder.mkdir(exist_ok=True)
        self.frames = []
        for index in range(3):
            path = self.folder / f"f{index}.png"
            write_image(path, width=120, height=90)
            self.frames.append(path)
        self.page = Bench()

    def test_a_folder_import_builds_a_queue_and_records_the_batch(self):
        self.page.add_paths(self.frames, source=str(self.folder), kind="folder")
        self.assertEqual(len(self.page._queue), 3)
        with connection() as conn:
            batch = queries.last_import(conn)
            stored = queries.frames_in_batch(conn, batch["batch_id"])
        self.assertEqual(batch["kind"], "folder")
        self.assertEqual(batch["source"], str(self.folder))
        self.assertEqual([row["file_path"] for row in stored],
                         [str(path) for path in sorted(self.frames)])

    def test_stepping_walks_the_queue_and_remembers_where_it_stopped(self):
        self.page.add_paths(self.frames, source=str(self.folder))
        self.page.step(1)
        self.assertEqual(self.page._index, 1)
        with connection() as conn:
            batch = queries.last_import(conn)
        self.assertEqual(batch["last_ordinal"], 1)
        self.page.step(-5)
        self.assertEqual(self.page._index, 0)          # clamped at the start

    def test_importing_the_same_folder_twice_is_idempotent(self):
        self.page.add_paths(self.frames, source=str(self.folder))
        first = self.page._batch_id
        self.page.add_paths(self.frames, source=str(self.folder))
        with connection() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) FROM imported_frames WHERE batch_id = ?",
                (first,)).fetchone()[0]
        self.assertEqual(rows, 3, "the frames of one batch are unique")

    def test_stepping_the_queue_switches_the_view_and_remembers_each_frame(self):
        """Each frame of a queue keeps its own view (`SR-BENCH-022`)."""
        self.page.add_paths(self.frames, source=str(self.folder))
        first = self.page._key
        self.page._add_measurement((0.0, 0.0), (10.0, 0.0))
        self.assertEqual(len(self.page.session), 1)
        self.page.step(1)
        self.assertNotEqual(self.page._key, first, "a new frame is a new view")
        self.assertEqual(self.page.session, [], "and it starts empty")
        self.page.step(-1)
        self.assertEqual(self.page._key, first)
        self.assertEqual(len(self.page.session), 1, "the first frame's view came back")
        self.assertEqual(self.page.list_results.topLevelItemCount(), 1,
                         "and it is listed again")
        self.assertIn("Frame 1 of 3", self.page.count_label.text())

    @unittest.skipUnless(_cv2 is not None, "OpenCV is not installed")
    def test_a_burst_is_written_to_its_own_folder_and_opens_like_an_import(self):
        """A burst lands in a folder of its own, so *Import a folder* gives the same queue."""
        self.page.service._recent.clear()
        for _ in range(3):
            self.page.service._recent.append(np.zeros((48, 64, 3), dtype=np.uint8))
        self.page.burst()

        self.assertEqual(len(self.page._queue), 3)
        folder = self.page._queue[0].parent
        self.assertTrue(folder.name.startswith("burst_"), folder.name)
        self.assertTrue(all(frame.parent == folder for frame in self.page._queue))
        self.assertTrue(all(frame.exists() for frame in self.page._queue),
                        "a burst is a deliberate capture, so its frames are on disk")
        with connection() as conn:
            batch = queries.last_import(conn)
        self.assertEqual(batch["kind"], "folder", "it is imported exactly like a folder")
        self.assertEqual(batch["source"], str(folder))
        self.page.step(1)
        self.assertNotEqual(self.page._key, str(self.page._queue[0]))

    @unittest.skipUnless(_cv2 is not None, "OpenCV is not installed")
    def test_a_video_import_writes_frames_and_records_their_numbers(self):
        """A clip is not a file you can measure - its frames are, and keep their number."""
        path = Path(self._tmp.name) / "clip.mp4"
        writer = _cv2.VideoWriter(str(path), _cv2.VideoWriter_fourcc(*"mp4v"),
                                  5.0, (64, 48))
        for _ in range(4):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()

        self.page.import_video(str(path))
        self.assertEqual(len(self.page._queue), 4)
        self.assertTrue(all(frame.name.startswith("f") for frame in self.page._queue))
        with connection() as conn:
            batch = queries.last_import(conn)
            frames = queries.frames_in_batch(conn, batch["batch_id"])
        self.assertEqual(batch["kind"], "video")
        self.assertEqual([row["frame_index"] for row in frames], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
