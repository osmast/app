"""Dataset: building the training corpus without a study - and exporting it.

The point of these tests is the *absence* of ceremony around the labels and the
*determinism* of the export: many frames, labelled with no study, plate,
connection or calibration, split so a training run can be reproduced.
"""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QLabel

from daphnia import maintenance
from daphnia.db import connection, init_db, queries, repository
from daphnia.ui.pages.dataset import (
    BoxesExportDialog,
    Dataset,
    crop_box,
    suggested_images_dir,
)
from daphnia.vision.crop import crop_for_box
from daphnia.vision.export import ExportError, read_boxes
from tests.test_images import write_image


def write_boxes_export(folder, rows, size=(1280, 720)):
    """An export folder as an operator has one: ``images/`` + ``boxes.csv``.

    Written with the **fixed** column list - ``file`` is the original path and
    ``image`` the exported copy - because that is what the exporter writes now.
    """
    folder = Path(folder)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    photo = folder / "images" / "f00.png"
    write_image(photo, width=size[0], height=size[1])
    with (folder / "boxes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("box_id", "file", "image", "split", "image_width",
                         "image_height", "specimen_index", "x1", "y1", "x2", "y2",
                         "width", "height", "source", "labeled_by", "created_at"))
        for index, (x1, y1, x2, y2, source) in enumerate(rows, start=1):
            writer.writerow((f"BOX_{index}", "/elsewhere/b01_f00.png",
                             "images/f00.png", "train", size[0], size[1], index,
                             x1, y1, x2, y2, x2 - x1, y2 - y1, source, "os",
                             "2026-10-17 00:00:00"))
    return folder / "boxes.csv"


class DatasetStorageTests(unittest.TestCase):
    """Scale-free rows: created without a study, corrected in place, exported."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "dataset.db"
        self.folder = Path(self._tmp.name) / "frames"
        self.folder.mkdir()
        init_db(self.db_path)

    def _label(self, conn, file_path="/tmp/x.png", trait="length",
               specimen_index=1, profile=None, p1=(10.0, 20.0), p2=(110.0, 20.0)):
        return repository.save_corpus_label(
            conn, file_path=file_path, image_size=(200, 150), trait=trait,
            p1=p1, p2=p2, specimen_index=specimen_index, profile=profile)

    def _profile(self, conn):
        repository.upsert_profile(
            conn, profile_id="PROF_D", profile_name="d", microscope_model="m",
            objective_magnification="2x", camera_device="c", px_per_um=2.0)
        repository.activate_profile(conn, "PROF_D")
        return repository.active_profile(conn)

    def test_a_label_is_stored_without_a_study_or_a_scale(self):
        """SR-DATASET-N01: labelling needs neither a study nor a calibration."""
        with connection(self.db_path) as conn:
            self._label(conn)
            row = queries.corpus_labels(conn)[0]
            studies = conn.execute("SELECT COUNT(*) FROM studies").fetchone()[0]
        self.assertEqual(studies, 0)
        self.assertIsNone(row["profile_id"])
        self.assertIsNone(row["length_um"])
        self.assertAlmostEqual(row["length_px"], 100.0)
        self.assertEqual((row["p1_x"], row["p2_y"]), (10.0, 20.0))

    def test_a_scale_is_recorded_when_one_is_in_force(self):
        with connection(self.db_path) as conn:
            profile = self._profile(conn)
            self._label(conn, profile=profile)
            row = queries.corpus_labels(conn)[0]
        self.assertEqual(row["profile_id"], "PROF_D")
        self.assertAlmostEqual(row["px_per_um"], 2.0)
        self.assertAlmostEqual(row["length_um"], 50.0)

    def test_re_labelling_the_same_interval_updates_instead_of_duplicating(self):
        with connection(self.db_path) as conn:
            first = self._label(conn)
            second = self._label(conn, p2=(210.0, 20.0))
            rows = queries.corpus_labels(conn)
        self.assertEqual(first, second)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["length_px"], 200.0)   # the correction won

    def test_several_specimens_and_traits_live_on_one_frame(self):
        with connection(self.db_path) as conn:
            self._label(conn, specimen_index=1, trait="length")
            self._label(conn, specimen_index=2, trait="length")
            self._label(conn, specimen_index=1, trait="width")
            rows = queries.corpus_labels_for_file(conn, "/tmp/x.png")
            stats = queries.corpus_stats(conn)
        self.assertEqual(len(rows), 3)
        self.assertEqual(stats["labels"], 3)
        self.assertEqual(stats["images"], 1)
        self.assertEqual(stats["by_trait"], {"length": 2, "width": 1})

    def test_labelled_paths_report_which_images_carry_a_label(self):
        with connection(self.db_path) as conn:
            self._label(conn, "/tmp/a.png")
            self._label(conn, "/tmp/b.png")
            paths_with_labels = queries.corpus_labelled_paths(conn)
        self.assertEqual(paths_with_labels, {"/tmp/a.png", "/tmp/b.png"})

    def test_delete_removes_one_interval(self):
        with connection(self.db_path) as conn:
            self._label(conn, specimen_index=1)
            self._label(conn, specimen_index=2)
            removed = repository.delete_corpus_label(
                conn, file_path="/tmp/x.png", specimen_index=1, trait="length")
            remaining = queries.corpus_labels(conn)
        self.assertEqual(removed, 1)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["specimen_index"], 2)


    # -- export ------------------------------------------------------------
    def _labelled_image(self, how_many=2):
        """One real image, labelled for the first ``how_many`` traits."""
        photo = self.folder / "specimen.png"
        write_image(photo, width=200, height=150)
        with connection(self.db_path) as conn:
            for trait in ("length", "width")[:how_many]:
                self._label(conn, file_path=str(photo), trait=trait)
        return photo

    def test_the_split_is_deterministic_and_covers_three_buckets(self):
        names = [maintenance.split_of(f"/tmp/frame_{i}.png") for i in range(200)]
        self.assertEqual(set(names), {"train", "val", "test"})
        self.assertEqual(names, [maintenance.split_of(f"/tmp/frame_{i}.png")
                                 for i in range(200)])
        self.assertEqual(names.count("train"), max(names.count(n) for n in names))

    def test_every_label_of_one_image_shares_a_split(self):
        self._labelled_image()
        destination = self.folder / "export"
        with connection(self.db_path) as conn:
            maintenance.export_corpus(conn, destination)
        data = json.loads((destination / "labels.json").read_text(encoding="utf-8"))
        splits = {record["split"] for record in data["labels"]}
        self.assertEqual(len(splits), 1)

    def test_the_export_writes_images_labels_and_a_manifest(self):
        self._labelled_image()
        destination = self.folder / "export"
        with connection(self.db_path) as conn:
            result = maintenance.export_corpus(conn, destination)

        self.assertTrue((destination / "labels.csv").exists())
        self.assertTrue((destination / "labels.json").exists())
        self.assertTrue((destination / "manifest.json").exists())
        self.assertEqual(len(list((destination / "images").glob("*.png"))), 1)

        data = json.loads((destination / "labels.json").read_text(encoding="utf-8"))
        record = data["labels"][0]
        self.assertIn(record["trait"], ("length", "width"))
        self.assertEqual(record["image_width"], 200)
        self.assertIn("keypoints", record)
        self.assertEqual(result["labels"], 2)
        self.assertEqual(result["images"], 1)

    def test_the_manifest_records_the_split_counts(self):
        self._labelled_image()
        destination = self.folder / "export"
        with connection(self.db_path) as conn:
            maintenance.export_corpus(conn, destination)
        manifest = json.loads(
            (destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(manifest["split_counts"].values()), 2)
        self.assertEqual(manifest["stats"]["labels"], 2)


class DatasetPageTests(unittest.TestCase):
    """The screen itself: import a queue, click, save - with no study or scale."""

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "dataset-ui.db")
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
            for table in ("corpus_labels", "corpus_boxes", "datasets",
                          "media_assets",
                          "observations", "imaging_sessions",
                          "calibration_profiles", "studies"):
                conn.execute(f"DELETE FROM {table}")
        self.folder = Path(self._tmp.name) / "frames"
        self.folder.mkdir(exist_ok=True)
        self.photo = self.folder / "specimen.png"
        write_image(self.photo, width=200, height=150)
        self.page = Dataset()

    def _calibrate(self, px_per_um=2.0):
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key="PROF_D", profile_name="d", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=px_per_um, uncertainty_pct=1.0)

    def _select_specimen(self, index):
        """Pick a specimen from the dropdown (as an operator would)."""
        combo = self.page.combo_specimen
        combo.setCurrentIndex(combo.findData(index))

    def test_the_page_is_three_steps(self):
        for name, word in (("corpus_group", "corpus"),
                           ("label_group", "label"),
                           ("export_group", "export")):
            self.assertTrue(hasattr(self.page, name), name)
            self.assertIn(word, getattr(self.page, name).title().lower())

    def test_nothing_is_disabled(self):
        """The lesson from the calibrate button: a dead control looks broken."""
        for key in ("open_image", "import_folder", "use_frame", "export", "model"):
            action = self.page.workspace.action(key)
            self.assertIsNotNone(action, key)
            if action.isVisible():
                self.assertTrue(action.isEnabled(), key)
        # The labelling actions and the queue tools are the header's menu items now
        # (`X-N13`). One that is hidden for the current state - a box tool on a keypoint
        # corpus - is not offered at all, so there is nothing dead to click (`X-N10`); Qt
        # reports a hidden action as disabled as well, so what must hold is that every
        # *shown* one is live.
        for key in ("save", "undo", "clear", "clear_frame", "box", "prev", "next"):
            action = self.page.workspace.action(key)
            self.assertIsNotNone(action, key)
            if action.isVisible():
                self.assertTrue(action.isEnabled(), key)
        for combo in (self.page.combo_mode, self.page.combo_specimen):
            self.assertTrue(combo.isEnabled(), combo.objectName())

    def test_a_repeated_dataset_name_is_numbered_and_reported(self):
        self.page.create_dataset(name="corpus", schema_key="pose_full")
        self.page.create_dataset(name="corpus", schema_key="boxes")
        with connection() as conn:
            names = sorted(row["name"] for row in queries.list_datasets(conn))
        self.assertIn("corpus", names)
        self.assertIn("corpus (2)", names)
        self.assertEqual(self.page.dataset()["name"], "corpus (2)")

    def test_the_combo_shows_the_model_of_each_dataset(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        texts = [self.page.combo_dataset.itemText(i)
                 for i in range(self.page.combo_dataset.count())]
        self.assertTrue(any("Bounding boxes only" in text for text in texts))
        self.assertTrue(any("head, tail & sides" in text for text in texts))

    def test_renaming_keeps_the_model_and_the_labels(self):
        self.page.add_paths([self.photo])
        dataset_id = self.page.dataset_id()
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        self.page.save_label()
        final = self.page.set_dataset_name("renamed corpus")
        with connection() as conn:
            row = queries.dataset_by_id(conn, dataset_id)
            labels = queries.dataset_labels(conn, dataset_id)
        self.assertEqual(final, "renamed corpus")
        self.assertEqual(row["schema_key"], "pose_full")     # model unchanged
        self.assertEqual(len(labels), 1)                     # labels intact
        self.assertIn("renamed corpus", self.page.dataset_label.text())
        self.assertEqual(self.page.combo_dataset.currentData(), dataset_id)

    def test_the_label_panel_has_separate_action_buttons(self):
        """Actions are menu items; dropdowns are only for real choices.

        The labelling actions and the keypoint stepper are the page's own menu items now
        (`X-N13`), so they are looked up in its tool groups; the panel keeps the *choices*
        (mode, specimen, dataset) and nothing that acts.
        """
        for key in ("save", "undo", "clear", "clear_frame", "box", "prev", "next",
                    "next_keypoint", "model"):
            self.assertIsNotNone(self.page.workspace.action(key), key)
        self.assertFalse(hasattr(self.page, "combo_action"))
        for name in ("combo_mode", "combo_specimen"):
            self.assertTrue(hasattr(self.page, name), name)

    def test_importing_frames_builds_a_sorted_deduplicated_queue(self):
        frames = []
        for name in ("c.png", "a.png", "b.png", "a.png"):
            path = self.folder / name
            write_image(path)
            frames.append(path)
        self.page.add_paths(frames)
        self.assertEqual([p.name for p in self.page._queue],
                         ["a.png", "b.png", "c.png"])          # sorted, no repeats
        self.assertIn("3 frame", self.page.queue_label.text())

    def test_navigation_is_clamped_at_the_ends(self):
        frames = [self.folder / f"f{i}.png" for i in range(3)]
        for frame in frames:
            write_image(frame)
        self.page.add_paths(frames)
        self.page.step(-1)
        self.assertEqual(self.page._index, 0)                  # cannot go before 0
        self.page.step(1)
        self.assertEqual(self.page._index, 1)
        for _ in range(5):
            self.page.step(1)
        self.assertEqual(self.page._index, 2)                  # cannot pass the end

    def test_a_frame_can_be_labelled_without_any_calibration(self):
        """SR-DATASET-N01: a label needs no scale - only a resolution is optional."""
        self.page.add_paths([self.photo])
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        self.page.save_label()
        with connection() as conn:
            row = queries.corpus_labels(conn)[0]
        self.assertIsNone(row["profile_id"])
        self.assertIsNone(row["length_um"])
        self.assertAlmostEqual(row["length_px"], 100.0)
        self.assertIn("stored in pixels", self.page.status_label.text())

    def test_a_scale_is_reported_when_one_is_in_force(self):
        self._calibrate(px_per_um=2.0)
        self.page.refresh()
        self.page.add_paths([self.photo])
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        self.assertIn("50.0", self.page.readout_label.text())   # 100 px at 2 px/um
        self.page.save_label()
        with connection() as conn:
            row = queries.corpus_labels(conn)[0]
        self.assertTrue(row["profile_id"].startswith("PROF_D"))
        self.assertAlmostEqual(row["length_um"], 50.0)

    def test_saving_needs_two_landmarks(self):
        self.page.add_paths([self.photo])
        self.page.save_label()
        self.assertIn("click the", self.page.status_label.text().lower())

    def test_several_specimens_on_one_frame(self):
        """A second specimen is simply the next one - save advances the dropdown."""
        self.page.add_paths([self.photo])
        self.page.canvas.set_points([(0.0, 0.0), (10.0, 0.0)])
        self.page.save_label()
        self.assertEqual(self.page.specimen(), 2)
        self.assertEqual(self.page.canvas.points, [])
        self.page.canvas.set_points([(0.0, 0.0), (20.0, 0.0)])
        self.page.save_label()
        with connection() as conn:
            rows = queries.corpus_labels(conn)
        self.assertEqual(sorted(row["specimen_index"] for row in rows), [1, 2])

    def test_re_labelling_the_same_specimen_updates_instead_of_duplicating(self):
        self.page.add_paths([self.photo])
        self.page.canvas.set_points([(0.0, 0.0), (10.0, 0.0)])
        self.page.save_label()
        self._select_specimen(1)                    # pick it from the dropdown
        self.page.canvas.set_points([(0.0, 0.0), (20.0, 0.0)])
        self.page.save_label()
        with connection() as conn:
            rows = queries.corpus_labels(conn)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["length_px"], 20.0)

    def test_switching_trait_changes_the_landmark_prompt(self):
        self.page.add_paths([self.photo])
        self.page.set_trait("length")
        self.assertIn("head", self.page.prompt_label.text())
        self.assertIn("spine base", self.page.canvas.point_names)
        self.page.set_trait("width")
        self.assertIn("left side", self.page.prompt_label.text())
        self.assertIn("right side", self.page.canvas.point_names)

    def test_a_stored_label_is_shown_again_when_the_frame_returns(self):
        other = self.folder / "other.png"
        write_image(other)
        self.page.add_paths([self.photo, other])
        self.page.canvas.set_points([(5.0, 5.0), (55.0, 5.0)])
        self.page.save_label()
        self.page.step(1)                       # move to the other frame
        self.assertEqual(self.page.canvas.annotations, [])
        self.page.step(-1)                      # back to the labelled frame
        overlays = self.page.canvas.annotations
        self.assertEqual([index for index, _ in overlays], [1])
        self.assertEqual(overlays[0][1], [((5.0, 5.0), (55.0, 5.0))])
        self.assertEqual(self.page.canvas.points, [])   # nothing being clicked yet

    def test_the_corpus_stats_report_labels(self):
        self.page.add_paths([self.photo])
        self.page.canvas.set_points([(0.0, 0.0), (10.0, 0.0)])
        self.page.save_label()
        self.assertIn("1 label", self.page.stats_label.text())
        self.assertIn("1 frame", self.page.queue_label.text())
        self.assertIn("1 labelled", self.page.queue_label.text())

    def test_undo_clears_the_clicked_points_first(self):
        self.page.add_paths([self.photo])
        self.page.canvas.set_points([(0.0, 0.0), (10.0, 0.0)])
        self.page.undo_point()
        self.assertEqual(self.page.canvas.points, [])
        self.assertIn("cleared the clicked points",
                      self.page.status_label.text().lower())

    def test_undo_removes_a_stored_pair_when_nothing_is_clicked(self):
        self.page.add_paths([self.photo])
        self.page.canvas.set_points([(0.0, 0.0), (10.0, 0.0)])
        self.page.save_label()
        self._select_specimen(1)
        self.page.undo_point()
        with connection() as conn:
            self.assertEqual(queries.corpus_labels(conn), [])
        self.assertIn("removed the stored", self.page.status_label.text().lower())

    def test_exporting_an_empty_corpus_says_so(self):
        self.page.export()
        self.assertIn("nothing to export", self.page.status_label.text().lower())


class DatasetPassModeTests(unittest.TestCase):
    """One landmark at a time: all the heads, then all the tails, then the sides."""

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "dataset-pass.db")
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
            for table in ("corpus_labels", "corpus_boxes", "datasets",
                          "media_assets",
                          "observations", "imaging_sessions",
                          "calibration_profiles", "studies"):
                conn.execute(f"DELETE FROM {table}")
        self.folder = Path(self._tmp.name) / "frames"
        self.folder.mkdir(exist_ok=True)
        self.photo = self.folder / "specimen.png"
        write_image(self.photo, width=200, height=150)
        self.page = Dataset()
        self.page.add_paths([self.photo])

    def _pass_mode(self):
        combo = self.page.combo_mode
        combo.setCurrentIndex(combo.findData("pass"))

    def _landmark(self, name):
        self.page.set_landmark(name)

    def _click(self, x, y):
        """A canvas click: the canvas reports the full new point list."""
        points = self.page.canvas.points + [(float(x), float(y))]
        self.page.canvas.set_points(points)

    def test_pass_mode_shows_the_keypoint_and_hides_save(self):
        """In two-point mode Save is there; in pass mode it is gone from the image."""
        self.page.show()
        self.addCleanup(self.page.hide)
        self.assertIn("trait", self.page.keypoint_label.text().lower())
        self.assertTrue(self.page.workspace.action("save").isVisible())
        self._pass_mode()
        self.assertIn("keypoint", self.page.keypoint_label.text().lower())
        self.assertFalse(self.page.workspace.action("save").isVisible())  # auto-stores
        self.assertIn("click every specimen", self.page.prompt_label.text().lower())

    def test_all_heads_then_all_tails_makes_two_intervals(self):
        self._pass_mode()
        self._landmark("head")
        self._click(10, 10)
        self._click(30, 10)
        self.assertEqual(len(self.page.canvas.annotations), 2)   # visible per specimen
        with connection() as conn:
            self.assertEqual(queries.corpus_stats(conn)["labels"], 0)  # nothing yet

        self._landmark("spine base")
        self._click(10, 30)
        self._click(30, 30)
        with connection() as conn:
            rows = queries.corpus_labels(conn)
        self.assertEqual(sorted(row["specimen_index"] for row in rows), [1, 2])
        first = [row for row in rows if row["specimen_index"] == 1][0]
        self.assertEqual((first["p1_x"], first["p1_y"]), (10.0, 10.0))
        self.assertEqual((first["p2_x"], first["p2_y"]), (10.0, 30.0))
        self.assertEqual(first["trait"], "length")

    def test_the_width_pass_uses_left_then_right(self):
        self._pass_mode()
        self._landmark("left side")
        self._click(0, 0)
        self._landmark("right side")
        self._click(40, 0)
        with connection() as conn:
            row = queries.corpus_labels(conn)[0]
        self.assertEqual(row["trait"], "width")
        self.assertAlmostEqual(row["length_px"], 40.0)

    def test_undo_removes_the_last_placed_point(self):
        self._pass_mode()
        self._landmark("head")
        self._click(10, 10)
        self._click(30, 10)
        self.page.undo_point()
        self.assertEqual(self.page._draft.specimens(), [1])
        self.page.undo_point()
        self.assertEqual(self.page._draft.specimens(), [])
        self.page.undo_point()               # nothing left to remove
        self.assertIn("nothing to undo", self.page.status_label.text().lower())

    def test_undo_also_removes_a_stored_interval(self):
        """Once the pass stores an interval, Undo must still be able to take it back."""
        self._pass_mode()
        self._landmark("head")
        self._click(10, 10)
        self._landmark("spine base")
        self._click(10, 40)                         # stores one length label
        with connection() as conn:
            self.assertEqual(len(queries.corpus_labels(conn)), 1)
        self.page.undo_point()
        with connection() as conn:
            self.assertEqual(queries.corpus_labels(conn), [])
        self.assertIn("removed the stored", self.page.status_label.text().lower())

    def test_the_pass_readout_counts_intervals_per_trait(self):
        self._pass_mode()
        self._landmark("head")
        self._click(10, 10)
        self._click(30, 10)
        self._landmark("spine base")
        self._click(10, 30)
        self._click(30, 30)
        self.assertIn("Length 2", self.page.readout_label.text())

    def test_the_landmark_advances_itself_after_a_full_pass(self):
        """Heads -> tails -> the sides: the pass moves on by itself, no cycling."""
        self._pass_mode()
        self._landmark("head")
        self._click(10, 10)
        self._click(30, 10)
        self._click(50, 10)
        self._landmark("spine base")
        self._click(10, 40)
        self._click(30, 40)
        self.assertEqual(self.page.landmark(), "spine base")   # not complete yet
        self._click(50, 40)                                     # 3rd -> all N done
        self.assertEqual(self.page.landmark(), "left side")
        self.assertEqual(self.page.specimen(), 1)
        for x in (10, 30, 50):
            self._click(x, 60)
        self.assertEqual(self.page.landmark(), "right side")
        for x in (10, 30, 50):
            self._click(x, 80)
        self.assertIn("fully labelled", self.page.status_label.text().lower())
        with connection() as conn:
            rows = queries.corpus_labels(conn)
        self.assertEqual(len(rows), 6)                          # 3 length + 3 width

    def test_the_head_pass_is_not_cut_short_by_auto_advance(self):
        """The first pass defines N, so it must never advance on its own."""
        self._pass_mode()
        self._landmark("head")
        self._click(10, 10)
        self._click(30, 10)
        self.assertEqual(self.page.landmark(), "head")

    def test_the_next_landmark_button_skips_ahead(self):
        self._pass_mode()
        self._click(10, 10)
        self.page.next_landmark()
        self.assertEqual(self.page.landmark(), "spine base")

    def test_the_pass_readout_reports_the_specimen_count(self):
        self._pass_mode()
        self._landmark("head")
        self._click(10, 10)
        self._click(30, 10)
        self.assertIn("2 specimen", self.page.readout_label.text())

    def test_stored_specimens_are_coloured_on_the_canvas(self):
        """Other specimens show as coloured interval lines - no text needed."""
        self.page.canvas.set_points([(1.0, 1.0), (2.0, 1.0)])
        self.page.save_label()
        overlays = self.page.canvas.annotations
        self.assertEqual([index for index, _ in overlays], [1])
        segments = overlays[0][1]
        self.assertEqual(len(segments), 1)                  # one length interval
        (start, end) = segments[0]
        self.assertNotEqual(start, end)                     # a real line, not a dot

    def test_a_pass_point_without_a_partner_is_a_dot_not_a_line(self):
        """Two heads are not an interval - they must not be drawn joined up."""
        self.page.combo_mode.setCurrentIndex(
            self.page.combo_mode.findData("pass"))
        self.page.set_landmark("head")
        for x in (10, 40):
            self.page.canvas.set_points(self.page.canvas.points + [(float(x), 10.0)])
        segments = self.page.canvas.annotations[0][1]
        self.assertTrue(all(start == end for start, end in segments))
        self.assertFalse(self.page.canvas.connect_points)

    def test_clearing_the_frame_removes_stored_labels_and_open_points(self):
        """A folder spans studies, so a frame must be re-doable from scratch."""
        self._pass_mode()
        self._landmark("head")
        self._click(10, 10)
        self._landmark("spine base")
        self._click(10, 40)                         # stores one length label
        with connection() as conn:
            self.assertEqual(len(queries.corpus_labels(conn)), 1)
        self._click(30, 10)                         # an unpaired point in the draft
        self.page.clear_frame()
        with connection() as conn:
            self.assertEqual(queries.corpus_labels(conn), [])
        self.assertEqual(self.page._draft.specimens(), [])
        self.assertIn("cleared 1 stored", self.page.status_label.text().lower())

class DatasetRegistryTests(unittest.TestCase):
    """Naming and renaming a dataset: never a crash, never a silent model change."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "registry.db"
        init_db(self.db_path)

    def test_a_repeated_name_is_numbered_instead_of_failing(self):
        with connection(self.db_path) as conn:
            first = repository.create_dataset(conn, name="corpus",
                                              schema_key="pose_full")
            second = repository.create_dataset(conn, name="corpus",
                                               schema_key="pose_head_tail")
            rows = queries.list_datasets(conn)
        self.assertNotEqual(first, second)
        names = sorted(row["name"] for row in rows)
        self.assertEqual(names, ["corpus", "corpus (2)"])
        self.assertEqual(rows[0]["dataset_id"], second)   # the new one is open

    def test_an_empty_name_still_gets_one(self):
        with connection(self.db_path) as conn:
            repository.create_dataset(conn, name="   ")
            row = queries.current_dataset(conn)
        self.assertEqual(row["name"], "Untitled")

    def test_renaming_keeps_the_model_the_id_and_the_labels(self):
        with connection(self.db_path) as conn:
            dataset_id = repository.create_dataset(conn, name="one",
                                                   schema_key="pose_head_tail")
            repository.save_corpus_label(
                conn, file_path="/tmp/x.png", image_size=(200, 150), trait="length",
                p1=(1.0, 2.0), p2=(3.0, 4.0), dataset_id=dataset_id)
            final = repository.rename_dataset(conn, dataset_id, "renamed")
            row = queries.dataset_by_id(conn, dataset_id)
            labels = queries.dataset_labels(conn, dataset_id)
        self.assertEqual(final, "renamed")
        self.assertEqual(row["dataset_id"], dataset_id)
        self.assertEqual(row["schema_key"], "pose_head_tail")   # untouched
        self.assertEqual(len(labels), 1)

    def test_renaming_to_a_taken_name_is_numbered(self):
        with connection(self.db_path) as conn:
            first = repository.create_dataset(conn, name="alpha")
            second = repository.create_dataset(conn, name="beta")
            final = repository.rename_dataset(conn, second, "alpha")
            names = sorted(row["name"] for row in queries.list_datasets(conn))
            untouched = queries.dataset_by_id(conn, first)
        self.assertEqual(final, "alpha (2)")
        self.assertEqual(names, ["alpha", "alpha (2)"])
        self.assertEqual(untouched["name"], "alpha")       # the first is unharmed

    def test_renaming_an_unknown_dataset_is_an_error(self):
        with connection(self.db_path) as conn:
            with self.assertRaises(ValueError):
                repository.rename_dataset(conn, "DATA_nope", "x")


class DatasetModelTests(unittest.TestCase):
    """One dataset per model: choosing the model decides what may be labelled."""

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "dataset-models.db")
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
            for table in ("corpus_labels", "corpus_boxes", "datasets",
                          "media_assets", "observations",
                          "imaging_sessions", "calibration_profiles", "studies"):
                conn.execute(f"DELETE FROM {table}")
        self.folder = Path(self._tmp.name) / "frames"
        self.folder.mkdir(exist_ok=True)
        self.photo = self.folder / "specimen.png"
        write_image(self.photo, width=200, height=150)
        self.page = Dataset()
        self.page.add_paths([self.photo])

    def _calibrate(self, px_per_um=2.0, uncertainty=1.0):
        """Register the scale in force, as the Optics surface would."""
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key="PROF_M", profile_name="m", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=px_per_um, uncertainty_pct=uncertainty)

    # -- the dataset itself ------------------------------------------------
    def test_a_default_dataset_exists_so_labelling_can_start(self):
        """Arriving with an empty database must not need a setup detour."""
        with connection() as conn:
            datasets = queries.list_datasets(conn)
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0]["schema_key"], "pose_full")
        self.assertEqual(self.page.dataset_id(), datasets[0]["dataset_id"])
        self.assertIn("Dataset:", self.page.dataset_label.text())

    def test_labels_from_before_datasets_existed_are_adopted(self):
        """A corpus labelled by an earlier build must not vanish from the surface."""
        with connection() as conn:
            conn.execute("DELETE FROM corpus_labels")
            conn.execute("DELETE FROM datasets")        # the state of an older build
            repository.save_corpus_label(
                conn, file_path=str(self.photo), image_size=(200, 150),
                trait="length", p1=(10.0, 10.0), p2=(110.0, 10.0))
        self.page._ensure_dataset()                     # this creates + adopts
        with connection() as conn:
            stats = queries.dataset_stats(conn, self.page.dataset_id())
            rows = queries.dataset_labels(conn, self.page.dataset_id())
        self.assertEqual(stats["labels"], 1)
        self.assertEqual(rows[0]["trait"], "length")

    def test_adoption_also_works_when_a_dataset_already_exists(self):
        """The live database already carries the (empty) default corpus."""
        with connection() as conn:
            conn.execute("DELETE FROM corpus_labels")
            repository.save_corpus_label(
                conn, file_path=str(self.photo), image_size=(200, 150),
                trait="width", p1=(10.0, 10.0), p2=(10.0, 60.0))
        self.page._ensure_dataset()
        with connection() as conn:
            self.assertEqual(
                queries.dataset_stats(conn, self.page.dataset_id())["labels"], 1)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM corpus_labels"
                             " WHERE dataset_id IS NULL").fetchone()[0], 0)

    def test_a_new_dataset_is_created_for_one_model_and_opened(self):
        dataset_id = self.page.create_dataset(name="two-point corpus",
                                              schema_key="pose_head_tail")
        self.assertEqual(self.page.dataset_id(), dataset_id)
        self.assertEqual(self.page.schema().key, "pose_head_tail")
        with connection() as conn:
            self.assertEqual(len(queries.list_datasets(conn)), 2)
            self.assertEqual(queries.current_dataset(conn)["dataset_id"], dataset_id)

    def test_switching_dataset_switches_the_model(self):
        first = self.page.dataset_id()
        second = self.page.create_dataset(name="boxes only", schema_key="boxes")
        self.page.select_dataset(first)
        self.assertEqual(self.page.schema().key, "pose_full")
        self.page.select_dataset(second)
        self.assertEqual(self.page.schema().key, "boxes")
        self.assertEqual(self.page.combo_dataset.currentData(), second)

    def test_the_dataset_dialog_suggests_a_name_and_locks_the_model_on_rename(self):
        from daphnia.ui.pages.dataset import DatasetDialog
        dialog = DatasetDialog()
        default = dialog.suggested_name()
        dialog.model_combo.setCurrentIndex(
            dialog.model_combo.findData("pose_head_tail"))
        self.assertIn("head", dialog.suggested_name().lower())
        dialog.name_edit.setText("")
        self.assertEqual(dialog.dataset_name(), dialog.suggested_name())
        self.assertTrue(default)                       # never an empty name

        rename = DatasetDialog(dataset=self.page.dataset())
        self.assertFalse(rename.model_combo.isEnabled())      # the model is what
        self.assertEqual(rename.dataset_name(),               # the labels *mean*
                         self.page.dataset()["name"])         # ...so it is locked
        rename.name_edit.setText("")
        self.assertEqual(rename.dataset_name(), self.page.dataset()["name"])

    # -- what the model allows to be labelled -----------------------------
    def test_a_two_point_model_offers_no_width(self):
        self.page.create_dataset(name="length only", schema_key="pose_head_tail")
        self.assertEqual(self.page._traits(), ("length",))
        self.page.next_keypoint()
        self.assertEqual(self.page.trait(), "length")   # wraps, never offers width
        self.page.set_trait("width")
        self.assertEqual(self.page.trait(), "length")
        self.page.set_landmark("left side")             # not part of this model
        self.assertEqual(self.page.landmark(), "head")

    def test_a_two_point_model_saves_only_a_length(self):
        self.page.create_dataset(name="length only", schema_key="pose_head_tail")
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        self.page.save_label()
        with connection() as conn:
            rows = queries.corpus_labels(conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trait"], "length")
        self.assertEqual(rows[0]["dataset_id"], self.page.dataset_id())

    def test_a_detection_model_hides_the_keypoint_controls(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.assertTrue(self.page._box_mode)
        self.assertTrue(self.page.canvas.box_mode)
        self.assertTrue(self.page.workspace.action("box").isChecked())
        # This model wants a box, so the Box tool is offered in the header.
        self.assertTrue(self.page.workspace.action("box").isVisible())
        self.assertTrue(self.page.combo_mode.isHidden())
        self.assertIn("box", self.page.keypoint_label.text().lower())
        self.assertIn("drag a rectangle", self.page.prompt_label.text().lower())

    def test_a_pose_model_without_boxes_hides_the_box_control(self):
        self.assertFalse(self.page.workspace.action("box").isVisible())
        self.assertFalse(self.page.combo_mode.isHidden())

    def test_saving_in_box_mode_points_at_the_drag(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.save_label()
        self.assertIn("release the drag", self.page.status_label.text().lower())

    def test_b_is_refused_when_the_model_has_no_boxes(self):
        """A model without boxes must say so, not silently do nothing."""
        self.assertIsNone(self.page.toggle_box_mode())
        self.assertIn("keypoint-only", self.page.status_label.text())
        self.assertFalse(self.page._box_mode)

    def test_b_toggles_box_drawing_when_the_model_has_boxes(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.assertTrue(self.page._box_mode)                 # on by default here
        self.page.toggle_box_mode()
        self.assertFalse(self.page._box_mode)
        self.assertFalse(self.page.canvas.box_mode)
        self.page.toggle_box_mode()
        self.assertTrue(self.page._box_mode)
        self.assertIn("drag", self.page.prompt_label.text().lower())

    def test_selecting_a_specimen_sites_its_stored_box(self):
        """Correcting a box is a redraw of the box you can see - in the **creation** mode.

        In *Edit* mode the same rectangle must not be drawn a second time, so the active box is
        left empty there (see `test_edit_mode_does_not_draw_the_selected_box_twice`).
        """
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.workspace.action("edit").setChecked(False)
        self.page.canvas.set_box((10.0, 20.0, 60.0, 100.0))    # specimen 1
        self.page.canvas.set_box((70.0, 20.0, 120.0, 100.0))   # specimen 2
        self.page._set_specimen(1)
        self.assertEqual(self.page.canvas.box, (10.0, 20.0, 60.0, 100.0))
        self.assertEqual(self.page._stored_boxes()[1]["x2"], 60.0)

    def test_a_new_box_goes_to_the_next_specimen_missing_one(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas.set_box((10.0, 20.0, 60.0, 100.0))    # 1
        self.page.canvas.set_box((70.0, 20.0, 120.0, 100.0))   # 2
        self.page._set_specimen(1)                             # back to correct 1
        self.page.canvas.set_box((5.0, 5.0, 50.0, 50.0))
        self.assertEqual(self.page.specimen(), 3)              # not 2 again

    def test_edit_mode_does_not_draw_the_selected_box_twice(self):
        """The complaint: "we also always see two duplicate boxes".

        In creation mode the current specimen's stored box is *sited* as the active box (so a
        correction is a redraw of what you see). In **edit** mode that drew the same rectangle
        twice - the dashed stored box and the active one - so the active box is left empty and
        the selection outline is the indicator (`SR-DATASET-027`).
        """
        self.page.create_dataset(name="detector", schema_key="boxes")
        canvas = self.page.canvas
        self.page.workspace.action("edit").setChecked(False)
        canvas.set_box((10.0, 20.0, 60.0, 100.0))            # stored for specimen 1
        self.page._set_specimen(1)
        self.assertEqual(canvas.box, (10.0, 20.0, 60.0, 100.0))
        canvas.clear_box()
        self.page.workspace.action("edit").setChecked(True)   # the automatic mode

        self.page._set_specimen(1)
        self.assertIsNone(canvas.box, "edit mode: the stored box is not drawn a second time")

    def _drag_point(self, index, dx=20.0, dy=10.0):
        """Drag point ``index`` by ``(dx, dy)`` screen pixels, on the page's own canvas."""
        canvas = self.page.canvas
        canvas.resize(600, 400)
        start = canvas._to_screen(canvas.points[index])
        begin, end = QPointF(*start), QPointF(start[0] + dx, start[1] + dy)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, begin, begin, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, begin, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, begin, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))

    def test_letting_go_of_a_point_writes_the_correction(self):
        """A stored label is corrected by the drop itself - no second key press, and the surface
        stays on the specimen, because the other point may be next (`SR-DATASET-027`)."""
        canvas = self.page.canvas
        canvas.resize(600, 400)
        canvas.set_points([(10.0, 20.0), (110.0, 20.0)])
        self.page.save_label()
        self.assertEqual(self.page.specimen(), 2, "storing a fresh label moves on")
        self.page._set_specimen(1)
        canvas.set_points([(10.0, 20.0), (110.0, 20.0)])

        self.assertTrue(self.page.workspace.action("edit").isChecked())
        self._drag_point(1)
        with connection() as conn:
            rows = queries.dataset_labels(conn, self.page.dataset_id())
        stored = [row for row in rows if row["specimen_index"] == 1]
        self.assertEqual(len(stored), 1, "the correction updates the row, it does not add one")
        self.assertGreater(stored[0]["p2_x"], 110.0)
        self.assertGreater(stored[0]["p2_y"], 20.0)
        self.assertEqual(self.page.specimen(), 1, "correcting stays on the specimen")
        self.assertIn("corrected", self.page.status_label.text().lower())

    def test_letting_go_of_a_point_that_is_not_stored_stores_nothing(self):
        """A pair still being placed is *Save*'s business: a drag must never store half a
        measurement (`SR-DATASET-027`)."""
        canvas = self.page.canvas
        canvas.resize(600, 400)
        canvas.set_points([(10.0, 20.0), (110.0, 20.0)])
        self._drag_point(1)
        with connection() as conn:
            self.assertEqual(queries.dataset_labels(conn, self.page.dataset_id()), [])

    def test_a_second_click_finishes_the_box_and_the_page_stores_it(self):
        """The requested flow, end to end: click a corner, click the opposite one.

        A box corpus puts the canvas in the **Box** tool, so a click on empty space
        anchors a corner whether or not *Edit* is on - drawing a box and editing the
        boxes already on the frame are one tool now (`SR-DATASET-027`). Clicking a
        **stored** box still selects it (asserted elsewhere); the *Select* tool, not a
        cleared *Edit* check box, is what marquees and never creates.
        """
        self.page.create_dataset(name="detector", schema_key="boxes")
        canvas = self.page.canvas
        canvas.resize(600, 400)
        rect = canvas._display_rect()
        scale = rect.width() / 200.0

        def widget(x, y):
            return QPointF(rect.x() + x * scale, rect.y() + y * scale)

        def click(x, y):
            point = widget(x, y)
            canvas.mousePressEvent(QMouseEvent(
                QEvent.MouseButtonPress, point, point, Qt.LeftButton,
                Qt.LeftButton, Qt.NoModifier))
            canvas.mouseReleaseEvent(QMouseEvent(
                QEvent.MouseButtonRelease, point, point, Qt.LeftButton,
                Qt.NoButton, Qt.NoModifier))

        self.assertTrue(self.page.workspace.action("edit").isChecked())
        click(10.0, 20.0)
        self.assertTrue(canvas.box_pending, "Box tool: empty space anchors a corner")
        self.page.workspace.action("edit").setChecked(False)   # creation mode too
        canvas.clear_box()
        click(10.0, 20.0)
        with connection() as conn:
            self.assertEqual(queries.corpus_boxes(conn), [])   # a corner is not a box
        self.assertTrue(canvas.box_pending)
        self.assertIn("opposite corner", self.page.prompt_label.text().lower())
        self.assertIn("corner at (10, 20)", self.page.readout_label.text().lower())

        click(60.0, 100.0)                                     # the opposite corner
        with connection() as conn:
            rows = queries.corpus_boxes(conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["x1"], rows[0]["y1"], rows[0]["x2"], rows[0]["y2"]),
                         (10.0, 20.0, 60.0, 100.0))
        self.assertFalse(canvas.box_pending)
        self.assertEqual(self.page.specimen(), 2)              # storing moves on

    def test_the_page_says_what_the_second_click_will_do(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas.anchor_box_corner((10.0, 20.0))
        self.assertIn("opposite corner", self.page.prompt_label.text().lower())
        self.page.clear_points()                              # Escape's equivalent
        self.assertFalse(self.page.canvas.box_pending)
        self.assertIn("drag a rectangle", self.page.prompt_label.text().lower())

    def test_a_double_click_abandons_a_half_drawn_box(self):
        """The gesture changes frames; a started box must not survive it."""
        self._two_frames()
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas.anchor_box_corner((10.0, 20.0))
        self.assertTrue(self.page.canvas.box_pending)
        self.page.canvas.double_clicked.emit()
        self.assertFalse(self.page.canvas.box_pending)
        with connection() as conn:
            self.assertEqual(queries.corpus_boxes(conn), [])

    def test_clearing_the_frame_leaves_another_datasets_boxes_alone(self):
        boxes_id = self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas.set_box((10.0, 20.0, 60.0, 100.0))
        pose_id = self.page.create_dataset(name="pose", schema_key="pose_full")
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        self.page.save_label()
        self.page.clear_frame()
        with connection() as conn:
            self.assertEqual(queries.dataset_labels(conn, pose_id), [])
            self.assertEqual(len(queries.dataset_boxes(conn, boxes_id)), 1)

    # -- the double-click gesture (the wiring; the gesture is in test_image_canvas)
# -- the double-click gesture (the wiring; the gesture is in test_image_canvas)
    def _two_frames(self):
        """A queue of two frames, so "next frame" has somewhere to go."""
        second = self.folder / "second.png"
        write_image(second, width=200, height=150)
        self.page.add_paths([self.photo, second])
        return second

    def test_a_double_click_walks_to_the_next_frame(self):
        self._two_frames()
        self.assertEqual(self.page.position_label.text(), "Frame 1 of 2")
        self.page.canvas.double_clicked.emit()
        self.assertEqual(self.page.position_label.text(), "Frame 2 of 2")
        self.assertIn("Frame 2 of 2", self.page.status_label.text())

    def test_a_double_click_stops_at_the_last_frame(self):
        self._two_frames()
        self.page.canvas.double_clicked.emit()
        self.page.canvas.double_clicked.emit()
        self.assertEqual(self.page.position_label.text(), "Frame 2 of 2")
        self.assertIn("last frame", self.page.status_label.text().lower())

    def test_a_double_click_says_so_without_a_queue(self):
        self.page._queue = []
        self.page.canvas.double_clicked.emit()
        self.assertIn("not a queue", self.page.status_label.text().lower())

    def test_a_double_click_leaves_no_label_behind_in_landmark_mode(self):
        """The gesture's own first click must never end up in the corpus."""
        self._two_frames()
        combo = self.page.combo_mode
        combo.setCurrentIndex(combo.findData("pass"))
        self.page.set_landmark("head")
        self.page.canvas.set_points([(10.0, 10.0)])        # the first click of the pair
        self.assertEqual(self.page._draft.specimens(), [1])
        self.page.canvas.double_clicked.emit()
        with connection() as conn:
            self.assertEqual(queries.corpus_labels(conn), [])
        self.assertEqual(self.page._draft.specimens(), [])
        self.assertEqual(self.page.position_label.text(), "Frame 2 of 2")
        self.assertIn("taken back", self.page.status_label.text())

    def test_a_double_click_leaves_no_loose_point_in_two_point_mode(self):
        self._two_frames()
        self.page.canvas.set_points([(10.0, 10.0)])
        self.page.canvas.double_clicked.emit()
        self.assertEqual(self.page.canvas.points, [])
        self.assertEqual(self.page.position_label.text(), "Frame 2 of 2")

    def test_a_double_click_stores_no_box_from_its_own_click(self):
        """The sliver of box the gesture's first press opened must not survive it."""
        self._two_frames()
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas._box = (10.0, 20.0, 10.2, 20.2)   # what a press leaves open
        self.page.canvas.double_clicked.emit()
        with connection() as conn:
            self.assertEqual(queries.corpus_boxes(conn), [])
        self.assertIsNone(self.page.canvas.box)
        self.assertEqual(self.page.position_label.text(), "Frame 2 of 2")
    # -- boxes -------------------------------------------------------------
    def test_a_drawn_box_is_stored_for_the_specimen(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas.set_box((10.0, 20.0, 60.0, 100.0))
        with connection() as conn:
            rows = queries.corpus_boxes(conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["specimen_index"], 1)
        self.assertEqual((rows[0]["x1"], rows[0]["y1"], rows[0]["x2"], rows[0]["y2"]),
                         (10.0, 20.0, 60.0, 100.0))
        self.assertEqual(rows[0]["dataset_id"], self.page.dataset_id())
        self.assertIn("stored the box", self.page.status_label.text().lower())

    def test_a_second_box_moves_to_the_next_specimen(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas.set_box((10.0, 20.0, 60.0, 100.0))
        self.assertEqual(self.page.specimen(), 2)
        self.page.canvas.set_box((70.0, 20.0, 120.0, 100.0))
        with connection() as conn:
            rows = queries.corpus_boxes(conn)
        self.assertEqual(sorted(row["specimen_index"] for row in rows), [1, 2])

    def test_redrawing_a_box_corrects_it_instead_of_duplicating(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas.set_box((10.0, 20.0, 60.0, 100.0))
        self.page._set_specimen(1)
        self.page.canvas.set_box((5.0, 5.0, 50.0, 50.0))
        with connection() as conn:
            rows = queries.corpus_boxes(conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["x1"], 5.0)

    def test_undo_removes_the_last_box(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas.set_box((10.0, 20.0, 60.0, 100.0))
        self.page.undo_point()
        with connection() as conn:
            self.assertEqual(queries.corpus_boxes(conn), [])
        self.assertIn("removed the box", self.page.status_label.text().lower())

# -- two datasets on one frame ----------------------------------------
    def test_two_datasets_label_the_same_frame_without_colliding(self):
        first = self.page.dataset_id()
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        self.page.save_label()
        second = self.page.create_dataset(name="two point",
                                          schema_key="pose_head_tail")
        self.assertEqual(self.page._stored(), {})       # the other corpus is empty
        self.page.canvas.set_points([(20.0, 20.0), (120.0, 20.0)])
        self.page.save_label()
        with connection() as conn:
            self.assertEqual(len(queries.dataset_labels(conn, first)), 1)
            rows = queries.dataset_labels(conn, second)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset_id"], second)

    def test_the_stats_read_the_open_dataset_only(self):
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        self.page.save_label()
        self.page.create_dataset(name="empty", schema_key="pose_head_tail")
        self.assertIn("0 label(s)", self.page.stats_label.text())
        self.assertIn("empty", self.page.stats_label.text())

    # -- export ------------------------------------------------------------
    def _export(self):
        destination = self.folder / "export"
        with connection() as conn:
            result = maintenance.export_dataset(
                conn, self.page.dataset_id(), destination)
        return destination, result

    def test_the_export_carries_the_named_keypoints_of_the_model(self):
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        self.page.save_label()
        destination, result = self._export()
        data = json.loads((destination / "labels.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema"]["key"], "pose_full")
        self.assertEqual(data["schema"]["export_names"],
                         ["head", "tail", "sideL", "sideR"])
        specimen = data["specimens"][0]
        self.assertEqual(specimen["keypoints"]["head"], [10.0, 10.0])
        self.assertEqual(specimen["keypoints"]["tail"], [110.0, 10.0])
        self.assertIsNone(specimen["bbox"])              # this model has no boxes
        self.assertEqual(result["boxes"], 0)

    def test_the_export_carries_the_whole_budget_of_each_label(self):
        """A corpus row says what the number is *worth*, not only how it was scaled.

        `docs/accuracy.md` §3.1, on the eyepiece chain: 54.528 px at 0.027264 px/µm is
        a 2 mm animal whose `±` is 6.8 µm and whose **budget is 75.1 µm** - the click
        and the pixel grid travel in `total_uncertainty_um`, in both the CSV and the
        JSON, and in the per-specimen `traits` block a training config reads.
        """
        self._calibrate(px_per_um=0.027264, uncertainty=0.2934)
        self.page.canvas.set_points([(10.0, 10.0), (64.528, 10.0)])
        self.page.save_label()
        destination, _result = self._export()

        data = json.loads((destination / "labels.json").read_text(encoding="utf-8"))
        trait = next(iter(data["specimens"][0]["traits"].values()))
        self.assertAlmostEqual(trait["length_um"], 2000.0, places=1)
        self.assertAlmostEqual(trait["total_uncertainty_um"], 75.1, places=1)
        self.assertAlmostEqual(trait["uncertainty_pct"], 0.2934)

        with (destination / "labels.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertIn("total_uncertainty_um", rows[0])
        self.assertIn("uncertainty_pct", rows[0])
        self.assertEqual(len(rows[0]), len(rows[1]))      # header and row agree
        self.assertIn("75.099", rows[1])                  # the CSV's own 3 decimals

    def test_the_export_writes_boxes_only_for_a_detection_model(self):
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas.set_box((10.0, 20.0, 60.0, 100.0))
        destination, result = self._export()
        self.assertEqual(result["boxes"], 1)
        self.assertEqual(result["labels"], 0)
        self.assertEqual(len(list((destination / "images").glob("*.png"))), 1)
        data = json.loads((destination / "labels.json").read_text(encoding="utf-8"))
        self.assertEqual(data["specimens"][0]["bbox"], [10.0, 20.0, 60.0, 100.0])
        self.assertEqual(data["specimens"][0]["keypoints"], {})
        self.assertTrue((destination / "boxes.csv").exists())

    def test_the_export_writes_the_skeleton_of_the_model(self):
        """A pose config needs the bones, not only the points (SR-DATASET-023)."""
        destination, _result = self._export()
        data = json.loads((destination / "labels.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema"]["edges"],
                         [["head", "tail"], ["sideL", "sideR"]])

    def test_the_boxes_csv_reads_back_with_its_header_intact(self):
        """Every row has as many values as the header has names.

        The old writer's header had one name more than its rows had values, so any
        spreadsheet or CSV reader shifted every column from ``image`` onwards - a
        corpus defect that looked like data (SR-DATASET-024).
        """
        self.page.create_dataset(name="detector", schema_key="boxes")
        self.page.canvas.set_box((10.0, 20.0, 60.0, 100.0))
        destination, _result = self._export()
        with (destination / "boxes.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(len(rows[0]), len(rows[1]))
        self.assertIn("image", rows[0])
        boxes = read_boxes(destination / "boxes.csv")
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].bbox, (10.0, 20.0, 60.0, 100.0))
        self.assertEqual(boxes[0].image_width, 200)      # the size of the frame
        self.assertEqual(boxes[0].source, "manual")

    def test_the_export_is_scoped_to_one_dataset(self):
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        self.page.save_label()
        self.page.create_dataset(name="detector", schema_key="boxes")
        destination, result = self._export()
        self.assertEqual(result["labels"], 0)         # the other corpus is not in it
        manifest = json.loads(
            (destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset"]["name"], "detector")

    def test_the_export_writes_one_image_list_per_split(self):
        self.page.canvas.set_points([(10.0, 10.0), (110.0, 10.0)])
        self.page.save_label()
        destination, result = self._export()
        for name in ("train", "val", "test"):
            self.assertTrue((destination / "splits" / f"{name}.txt").exists(), name)
        total = sum(entry["images"] for entry in result["split_files"].values())
        self.assertEqual(total, 1)                     # one image, in one split
        data = json.loads((destination / "labels.json").read_text(encoding="utf-8"))
        self.assertEqual(data["splits"], result["split_files"])
        listed = [line for entry in result["split_files"].values()
                  for line in (destination / entry["file"]).read_text().split()
                  if line]
        self.assertEqual(len(listed), 1)
        self.assertTrue(listed[0].startswith("images/"))

    def test_two_datasets_export_to_different_folders(self):
        """The suggested folder carries the dataset name, so exports never mix."""
        first = self.page.dataset()
        self.assertTrue(str(first["name"]).split()[0] in
                        str(self.page._suggested_export_folder().name))
        self.page.create_dataset(name="other corpus", schema_key="boxes")
        self.assertIn("other_corpus",
                      self.page._suggested_export_folder().name)

    def test_the_export_says_so_when_a_split_is_still_empty(self):
        """Five frames cannot fill three buckets - that must be visible, not implied."""
        summary = self.page._export_summary(
            {"dataset": {"name": "corpus"}, "schema": {"label": "model"},
             "labels": 1, "boxes": 0, "images": 1,
             "splits": {"train": 1, "val": 0, "test": 0},
             "split_files": {}},
            "/tmp/export")
        self.assertIn("test/val split is still empty", summary)
        self.assertIn("train 1, val 0, test 0", summary)

    def test_a_full_split_is_not_flagged(self):
        summary = self.page._export_summary(
            {"dataset": {"name": "corpus"}, "schema": {"label": "model"},
             "labels": 3, "boxes": 1, "images": 3,
             "splits": {"train": 2, "val": 1, "test": 1},
             "split_files": {}},
            "/tmp/export")
        self.assertNotIn("still empty", summary)


class ImportBoxesTests(unittest.TestCase):
    """Seeding a corpus from boxes that already exist.

    The first half of the recursive annotator: a detector's boxes (or an earlier
    session's) become rows of a dataset, keeping the source they arrived with -
    because a prediction is not ground truth until a person says so.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "import.db"
        init_db(self.db_path)
        self.export = Path(self._tmp.name) / "bb"
        self.csv = write_boxes_export(
            self.export, [(10.0, 20.0, 110.0, 95.0, "model"),
                          (200.0, 30.0, 300.0, 140.0, "manual")])

    def _dataset(self, conn, schema_key="pose_full_boxes"):
        return repository.create_dataset(conn, name="pose on bb",
                                         schema_key=schema_key)

    def test_boxes_are_imported_with_their_source_and_frame_size(self):
        with connection(self.db_path) as conn:
            dataset_id = self._dataset(conn)
            result = maintenance.import_boxes(conn, dataset_id, self.csv)
            boxes = queries.dataset_boxes(conn, dataset_id)
        self.assertEqual(result["boxes"], 2)
        self.assertEqual(result["images"], 1)
        self.assertEqual([box["source"] for box in boxes], ["model", "manual"])
        self.assertEqual(boxes[0]["x1"], 10.0)
        self.assertEqual((boxes[0]["image_width"], boxes[0]["image_height"]),
                         (1280, 720))

    def test_a_prediction_stays_a_prediction_until_a_person_accepts_it(self):
        with connection(self.db_path) as conn:
            dataset_id = self._dataset(conn)
            maintenance.import_boxes(conn, dataset_id, self.csv)
            changed = repository.set_corpus_box_source(
                conn, file_path=self.export / "images" / "f00.png",
                specimen_index=1, dataset_id=dataset_id, source="manual",
                note="accepted as ground truth by a person")
            boxes = queries.dataset_boxes(conn, dataset_id)
        self.assertEqual(changed, 1)
        self.assertEqual(boxes[0]["source"], "manual")
        self.assertIn("accepted", boxes[0]["note"])
        self.assertEqual(boxes[1]["source"], "manual")     # untouched

    def test_importing_the_same_boxes_twice_updates_instead_of_duplicating(self):
        with connection(self.db_path) as conn:
            dataset_id = self._dataset(conn)
            maintenance.import_boxes(conn, dataset_id, self.csv)
            maintenance.import_boxes(conn, dataset_id, self.csv)
            boxes = queries.dataset_boxes(conn, dataset_id)
        self.assertEqual(len(boxes), 2)

    def test_a_keypoint_only_dataset_refuses_boxes_in_words(self):
        with connection(self.db_path) as conn:
            dataset_id = self._dataset(conn, schema_key="pose_full")
            with self.assertRaises(ValueError) as caught:
                maintenance.import_boxes(conn, dataset_id, self.csv)
        self.assertIn("keypoint-only", str(caught.exception))
        self.assertIn("Bounding boxes only", str(caught.exception))

    def test_a_box_outside_its_image_is_skipped_and_counted(self):
        """A box the labeller would refuse to make must not enter the corpus."""
        with connection(self.db_path) as conn:
            dataset_id = self._dataset(conn)
            result = maintenance.import_boxes(
                conn, dataset_id, write_boxes_export(
                    Path(self._tmp.name) / "bad",
                    [(10.0, 20.0, 110.0, 95.0, "manual"),
                     (1200.0, 30.0, 1400.0, 140.0, "manual")]))
            boxes = queries.dataset_boxes(conn, dataset_id)
        self.assertEqual(result["boxes"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(len(boxes), 1)

    def test_an_unreadable_source_is_named(self):
        with connection(self.db_path) as conn:
            dataset_id = self._dataset(conn)
            with self.assertRaises(ExportError):
                maintenance.import_boxes(conn, dataset_id,
                                         Path(self._tmp.name) / "nowhere")

    def test_an_explicit_image_folder_is_used_and_reported(self):
        """The file in one folder, the pictures in another - said, not guessed."""
        elsewhere = Path(self._tmp.name) / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "f00.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        with connection(self.db_path) as conn:
            dataset_id = self._dataset(conn)
            result = maintenance.import_boxes(conn, dataset_id, self.csv,
                                              images_dir=elsewhere)
            boxes = queries.dataset_boxes(conn, dataset_id)
        self.assertEqual(result["images_dir"], str(elsewhere))
        self.assertEqual(len(boxes), 2)
        self.assertEqual(Path(boxes[0]["file_path"]).parent, elsewhere)

    def test_an_unknown_dataset_is_named(self):
        with connection(self.db_path) as conn:
            with self.assertRaises(ValueError) as caught:
                maintenance.import_boxes(conn, "DATA_NOPE", self.csv)
        self.assertIn("DATA_NOPE", str(caught.exception))


class BoxesExportDialogTests(unittest.TestCase):
    """The two fields an existing box corpus needs: the boxes file, and the images.

    A ``boxes.csv`` names its frames relative to the export folder, so "where are the
    pictures" is a real question - and the folder an operator can *see* in a file
    browser is often ``images/`` itself. Both fields are therefore asked for, and the
    images folder is only ever *suggested* from the file.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.export = Path(self._tmp.name) / "bb"
        self.csv = write_boxes_export(
            self.export, [(10.0, 20.0, 110.0, 95.0, "manual")])

    def test_the_images_folder_is_suggested_from_the_file(self):
        self.assertEqual(suggested_images_dir(self.csv), self.export / "images")

    def test_the_suggestion_falls_back_to_the_files_own_folder(self):
        loose = Path(self._tmp.name) / "loose"
        loose.mkdir()
        self.assertEqual(suggested_images_dir(loose / "boxes.csv"), loose)

    def test_a_folder_can_be_the_source_too(self):
        dialog = BoxesExportDialog(source=self.export)
        self.assertEqual(dialog.boxes_file(), str(self.export))
        self.assertEqual(dialog.images_dir(), str(self.export / "images"))

    def test_the_dialog_pre_fills_both_fields_and_says_what_it_does(self):
        dialog = BoxesExportDialog(source=self.csv)
        self.assertEqual(dialog.boxes_file(), str(self.csv))
        self.assertEqual(dialog.images_dir(), str(self.export / "images"))
        texts = " ".join(label.text() for label in dialog.findChildren(QLabel))
        self.assertIn("magnified crop", texts)
        self.assertIn("original", texts)          # the crop is a view, not the truth

    def test_an_images_folder_the_operator_gave_is_kept(self):
        other = Path(self._tmp.name) / "other"
        other.mkdir()
        dialog = BoxesExportDialog(source=self.csv, images_dir=other)
        self.assertEqual(dialog.images_dir(), str(other))


class DatasetBoxQueueTests(unittest.TestCase):
    """The box queue: one animal at a time, on a magnified crop.

    The whole point of the crop is that a click lands where it looks, so the stored
    coordinates are asserted in **frame** pixels - the frame of reference the corpus
    keeps (`mlContext.md` law 1). Two boxes on one frame are used on purpose: the
    queue's unit is an animal, and the frame's other animals must survive everything
    done to this one.
    """

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "boxqueue.db")
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
            for table in ("corpus_labels", "corpus_boxes", "datasets"):
                conn.execute(f"DELETE FROM {table}")
        self.export = Path(self._tmp.name) / "bb"
        self.csv = write_boxes_export(
            self.export, [(100.0, 100.0, 200.0, 160.0, "model"),
                          (600.0, 300.0, 700.0, 360.0, "manual")])
        self.page = Dataset()
        self.page.show()
        self.page.create_dataset(name="pose on bb", schema_key="pose_full_boxes")

    def _open(self):
        return self.page.open_boxes_export(self.csv)

    def _click(self, first, second):
        """Two clicks, in whichever frame of reference the canvas is showing.

        With a crop in force the click happens in *crop* pixels and the page converts
        it back; in frame view the click already *is* a source pixel. Doing both here
        is the point of the test.
        """
        crop = self.page._crop
        to_view = crop.to_crop if crop is not None else (lambda point: point)
        self.page.canvas.set_points([to_view(first)])
        self.page.canvas.set_points([to_view(first), to_view(second)])

    def _mouse(self, position, end=None):
        """A **real** press (and move/release) at a widget position.

        The other helper places points programmatically, and that is exactly why nobody
        noticed that a real press inside the crop selected the box instead of placing the
        keypoint: the gesture was never the one under test.
        """
        canvas = self.page.canvas
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, position, position, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        if end is None:
            canvas.mouseReleaseEvent(QMouseEvent(
                QEvent.MouseButtonRelease, position, position, Qt.LeftButton,
                Qt.NoButton, Qt.NoModifier))
            return
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, position, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, position, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def _click_view(self, point):
        """A real click at a point in **view** pixels (crop pixels, in the crop)."""
        self._mouse(QPointF(*self.page.canvas._to_screen(point)))

    def test_opening_a_boxes_export_queues_one_item_per_animal(self):
        result = self._open()
        self.assertEqual(result["boxes"], 2)
        self.assertEqual(len(self.page._items), 2)
        self.assertTrue(self.page.item_mode())
        self.assertIn("Animal 1 of 2", self.page.position_label.text())

    def test_the_canvas_shows_the_magnified_crop_with_the_box_drawn(self):
        self._open()
        crop = self.page._crop
        self.assertEqual(crop.scale, 3)
        self.assertEqual(self.page._frame.shape[:2],
                         (crop.output_size[1], crop.output_size[0]))
        index, box = self.page.canvas.boxes[0]
        self.assertEqual(index, 1)
        self.assertGreater(box[0], 0)                     # the box sits *inside*
        self.assertLess(box[2], crop.output_size[0])

    def test_a_click_in_the_crop_is_stored_in_frame_pixels(self):
        self._open()
        self._click((120.0, 130.0), (180.0, 150.0))
        with connection() as conn:
            rows = queries.dataset_labels(conn, self.page.dataset_id())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["specimen_index"], 1)
        self.assertEqual(row["trait"], "length")
        self.assertEqual((row["p1_x"], row["p1_y"]), (120.0, 130.0))
        self.assertEqual((row["p2_x"], row["p2_y"]), (180.0, 150.0))
        # The frame of reference is the frame, never the magnified picture.
        self.assertEqual((row["image_width"], row["image_height"]), (1280, 720))

    def test_a_real_click_in_the_crop_places_the_keypoint(self):
        """*"Not able to correct the keypoint - it first selects the box"* (`SR-DATASET-025`).

        The item's box covers the whole magnified crop, so while *Edit* mode was on (the
        default) every **real** press inside the crop selected that box and no landmark
        could be placed: the prompt asked for two clicks and the clicks went to the box.
        In the crop of a keypoint corpus the press *is* the keypoint gesture now.
        """
        self._open()
        self.assertTrue(self.page.canvas.box_edit, "Edit mode is the default")
        crop = self.page._crop
        for at in ((120.0, 130.0), (180.0, 150.0)):
            self._click_view(crop.to_crop(at))

        with connection() as conn:
            rows = queries.dataset_labels(conn, self.page.dataset_id())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["specimen_index"], 1)
        self.assertAlmostEqual(row["p1_x"], 120.0, delta=1.5)
        self.assertAlmostEqual(row["p1_y"], 130.0, delta=1.5)
        self.assertAlmostEqual(row["p2_x"], 180.0, delta=1.5)
        self.assertAlmostEqual(row["p2_y"], 150.0, delta=1.5)
        self.assertEqual((row["image_width"], row["image_height"]), (1280, 720))
        self.assertIsNone(self.page.canvas.selected_box, "the box was not picked")

    def test_a_box_dragged_in_the_crop_is_stored_in_source_pixels(self):
        """A correction is stored in the image's own frame (`mlContext.md` law 1).

        Measured before this: dragging the item's box in the crop stored the **crop**
        pixels as if they were source pixels, with the crop's own size as the image size -
        specimen 1 of a 1280 x 720 frame became ``(82, 79, 382, 259)`` on a "450 x 330"
        image, so the rectangle the operator moved was not the rectangle the corpus kept.
        A detection-only corpus opens on the frame, so the crop is asked for by hand here.
        """
        self.page.create_dataset(name="detector", schema_key="boxes")
        self._open()
        self.page.set_view(True)
        crop = self.page._crop
        original = self.page._items[0]["box"]
        drawn = crop_box(crop, original)
        centre = QPointF((drawn[0] + drawn[2]) / 2.0, (drawn[1] + drawn[3]) / 2.0)
        start = QPointF(*self.page.canvas._to_screen((centre.x(), centre.y())))
        self._mouse(start, QPointF(start.x() + 12, start.y() + 6))

        with connection() as conn:
            row = queries.dataset_boxes_for_file(conn, self.page._path,
                                                 self.page.dataset_id())[1]
        self.assertEqual((row["image_width"], row["image_height"]), (1280, 720))
        self.assertAlmostEqual(row["x2"] - row["x1"], original[2] - original[0], delta=1.5)
        self.assertAlmostEqual(row["y2"] - row["y1"], original[3] - original[1], delta=1.5)
        self.assertGreater(row["x1"], original[0], "it moved the way it was dragged")
        self.assertGreater(row["y1"], original[1])
        # And the queue holds the correction, so the view and the corpus agree.
        self.assertEqual(self.page._items[0]["box"],
                         (row["x1"], row["y1"], row["x2"], row["y2"]))

    def test_storing_keypoints_of_a_prediction_leaves_the_box_question_open(self):
        """A prediction is never left unanswered: the queue waits for a Y or an N."""
        self._open()
        self._click((120.0, 130.0), (180.0, 150.0))
        self.assertIn("Animal 1 of 2", self.page.position_label.text())
        self.assertIn("Y (yes) or N (no)", self.page.status_label.text())

    def test_yes_makes_a_prediction_ground_truth_and_moves_on(self):
        self._open()
        self.page.accept_box()
        with connection() as conn:
            boxes = queries.dataset_boxes(conn, self.page.dataset_id())
        self.assertEqual(boxes[0]["source"], "manual")
        self.assertIn("accepted", boxes[0]["note"])
        self.assertIn("Animal 2 of 2", self.page.position_label.text())

    def test_no_removes_the_box_and_leaves_the_frame_in_the_corpus(self):
        self._open()
        self.page.step(1)                       # walk to the second animal
        self.page.reject_box()
        with connection() as conn:
            boxes = queries.dataset_boxes(conn, self.page.dataset_id())
        self.assertEqual([box["specimen_index"] for box in boxes], [1])
        self.assertIn("rejected", self.page.status_label.text())

    def test_clearing_this_animal_leaves_the_frames_other_animals_alone(self):
        self._open()
        self._click((120.0, 130.0), (180.0, 150.0))
        self.page.accept_box()                  # animal 1 is ground truth, move on
        self._click((650.0, 320.0), (680.0, 340.0))
        with connection() as conn:
            rows = queries.dataset_labels(conn, self.page.dataset_id())
        self.assertEqual(len(rows), 2)          # both labels on the *same* frame
        self.page.step(-1)                      # back to animal 1
        self.page.clear_frame()
        with connection() as conn:
            rows = queries.dataset_labels(conn, self.page.dataset_id())
        self.assertEqual([row["specimen_index"] for row in rows], [2])

    def test_a_keypoint_only_dataset_refuses_the_boxes_in_words(self):
        self.page.create_dataset(name="pose only", schema_key="pose_full")
        result = self.page.open_boxes_export(self.csv)
        self.assertEqual(result, {})
        self.assertIn("keypoint-only", self.page.status_label.text())

    def test_the_controls_follow_the_mode(self):
        self._open()
        self.assertFalse(self.page.workspace.action("box").isVisible())
        self.assertFalse(self.page.combo_mode.isVisible())
        self.assertFalse(self.page.workspace.action("save").isVisible())
        self.assertTrue(self.page.workspace.action("accept").isVisible())
        self.assertTrue(self.page.workspace.action("reject").isVisible())

    def test_the_queue_walks_and_clamps_at_the_ends(self):
        self._open()
        self.page.step(-1)
        self.assertIn("Animal 1 of 2", self.page.position_label.text())
        self.page.step(1)
        self.assertIn("Animal 2 of 2", self.page.position_label.text())
        self.page.step(1)
        self.assertIn("Animal 2 of 2", self.page.position_label.text())

    def test_the_queue_summary_counts_animals_not_frames(self):
        self._open()
        self.assertIn("2 animal(s) queued", self.page.queue_label.text())
        self.assertIn("1 still a prediction", self.page.queue_label.text())

    def test_the_crop_is_one_view_of_the_animal_and_the_frame_is_the_other(self):
        """The crop is a *view*: switching it never changes what a label stores."""
        self._open()
        self.assertEqual(self.page._frame.shape[:2],
                         self.page._crop.output_size[::-1])      # crop view
        self.page.toggle_view()
        self.assertIsNone(self.page._crop)                        # frame view
        self.assertEqual(self.page._frame.shape, (720, 1280, 3))
        self.assertEqual(self.page.canvas.boxes[0][0], 1)         # boxes still drawn
        self._click((120.0, 130.0), (180.0, 150.0))
        with connection() as conn:
            rows = queries.dataset_labels(conn, self.page.dataset_id())
        self.assertEqual((rows[0]["p1_x"], rows[0]["p1_y"]), (120.0, 130.0))
        self.assertEqual((rows[0]["p2_x"], rows[0]["p2_y"]), (180.0, 150.0))

    def test_the_frame_view_draws_every_box_and_highlights_this_animal(self):
        """The bounding-box display: all boxes visible, the current one solid."""
        self._open()
        self.page.set_view(False)
        self.assertEqual([index for index, _box in self.page.canvas.boxes], [1, 2])
        self.assertEqual(self.page.canvas.box, self.page._items[0]["box"])
        self.page.step(1)
        self.assertEqual(self.page.canvas.box, self.page._items[1]["box"])

    def test_the_view_tool_says_which_view_is_on(self):
        self._open()
        self.assertEqual(self.page.workspace.action("view").text(), "Crop")
        self.page.toggle_view()
        self.assertEqual(self.page.workspace.action("view").text(), "Frame")
        self.assertIn("whole frame", self.page.status_label.text())

    def test_a_detection_only_corpus_opens_on_the_whole_frame(self):
        """A box can only be judged inside its frame, so that is where it starts."""
        self.page.create_dataset(name="detector", schema_key="boxes")
        self._open()
        self.assertIsNone(self.page._crop)
        self.assertEqual(self.page._frame.shape, (720, 1280, 3))
        self.assertEqual(self.page.workspace.action("view").text(), "Frame")

    def test_a_point_outside_the_box_is_reported_and_still_stored(self):
        self._open()
        self.page.set_view(False)
        self._click((120.0, 130.0), (900.0, 600.0))     # the second is a neighbour
        self.assertIn("outside specimen 1's box", self.page.status_label.text())
        with connection() as conn:
            rows = queries.dataset_labels(conn, self.page.dataset_id())
        self.assertEqual(len(rows), 1)

    def test_switching_makes_the_dataset_that_can_hold_a_box(self):
        """The one-click path out of a keypoint-only corpus (the wall in the UI)."""
        self.page.create_dataset(name="pose only", schema_key="pose_full")
        self.assertEqual(self.page.schema().key, "pose_full")
        created = self.page.switch_to_box_dataset()
        self.assertEqual(self.page.dataset_id(), created)
        self.assertEqual(self.page.schema().key, "pose_full_boxes")
        self.assertIn("+ boxes", self.page.dataset()["name"])

    def test_undo_in_item_mode_only_touches_this_animal(self):
        self._open()
        self._click((120.0, 130.0), (180.0, 150.0))     # animal 1 is a prediction
        self.page.accept_box()                          # ground truth, move on
        self._click((650.0, 320.0), (680.0, 340.0))     # animal 2
        self.page.step(-1)                              # back to animal 1
        self.page.undo_point()
        with connection() as conn:
            rows = queries.dataset_labels(conn, self.page.dataset_id())
        self.assertEqual([row["specimen_index"] for row in rows], [2])

    def test_bringing_in_frames_leaves_the_box_queue_behind(self):
        """A frame queue and a box queue are two different questions."""
        self._open()
        self.page.add_paths([self.export / "images" / "f00.png"])
        self.assertFalse(self.page.item_mode())
        self.assertIsNone(self.page._crop)
        self.assertIn("frame(s) in the queue", self.page.queue_label.text())
        self.assertFalse(self.page.workspace.action("accept").isVisible())

    def test_switching_dataset_leaves_the_box_queue_behind(self):
        """Another corpus's boxes are never left on screen."""
        first = self.page.dataset_id()
        self._open()                                     # 2 animals queued
        self.page.create_dataset(name="another", schema_key="pose_full_boxes")
        self.assertFalse(self.page.item_mode())
        self.assertIsNone(self.page._crop)
        # ... and opening them again for that corpus is one explicit act.
        self.page.select_dataset(first)
        self.assertEqual(self.page.open_boxes_export(self.csv)["boxes"], 2)
        self.assertEqual(len(self.page._items), 2)



class DatasetAnnotationEditTests(unittest.TestCase):
    """A stored keypoint pair stays editable in frame view (`X-N16`).

    Driven with **real mouse events** on the drawn label: a press on a stored keypoint reports
    which measurement it is, the page loads that row back, and the same press drags the point -
    so a label set an hour ago is corrected without re-clicking the whole pair.
    """

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "dataset-annot.db")
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
            for table in ("corpus_labels", "corpus_boxes", "datasets",
                          "media_assets",
                          "observations", "imaging_sessions",
                          "calibration_profiles", "studies"):
                conn.execute(f"DELETE FROM {table}")
        self.folder = Path(self._tmp.name) / "frames"
        self.folder.mkdir(exist_ok=True)
        self.photo = self.folder / "specimen.png"
        write_image(self.photo, width=200, height=150)
        self.page = Dataset()
        self.page.add_paths([self.photo])

    def _store_a_pair(self):
        self.page.canvas.set_points([(40.0, 40.0), (120.0, 40.0)])
        self.page.save_label()
        with connection() as conn:
            rows = queries.corpus_labels(conn)
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_a_stored_keypoint_pair_is_editable_in_frame_view(self):
        row = self._store_a_pair()
        canvas = self.page.canvas
        canvas.resize(600, 400)
        before = (row["p1_x"], row["p1_y"])
        start = QPointF(*canvas._to_screen(before))
        end = QPointF(start.x() + 14, start.y() + 10)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        self.assertEqual(len(canvas.points), 2, "the stored pair is back in the draft")
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, start, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, start, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))
        with connection() as conn:
            rows = queries.corpus_labels(conn)
        self.assertEqual(len(rows), 1, "a correction, never a second label")
        self.assertNotAlmostEqual(rows[0]["p1_x"], before[0], msg="the point moved")

    def test_hovering_a_stored_pair_offers_a_grab(self):
        row = self._store_a_pair()
        canvas = self.page.canvas
        canvas.resize(600, 400)
        at = QPointF(*canvas._to_screen((row["p1_x"], row["p1_y"])))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, at, at, Qt.NoButton, Qt.NoButton, Qt.NoModifier))
        self.assertEqual(canvas.hover, ("annotation", 1))



if __name__ == "__main__":
    unittest.main()