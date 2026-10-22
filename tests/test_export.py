"""The dataset export reader: what a training run is allowed to assume."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from daphnia import paths
from daphnia.vision.export import Export, ExportError, read_boxes

#: Two captures, so a group has a sibling and the leak is visible.
CAPTURE_A = "recordings/STUDY_1/SESS_1/OBS_A"
CAPTURE_B = "recordings/STUDY_1/SESS_1/OBS_B"


def write_export(folder, specimens, splits=None):
    """Write a minimal export, the way ``maintenance.export_dataset`` does."""
    folder = Path(folder)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    for record in specimens:
        if record.get("image"):
            (folder / record["image"]).write_bytes(b"\x89PNG\r\n\x1a\n")
    (folder / "labels.json").write_text(json.dumps({
        "dataset": {"name": "unit", "schema_key": "boxes"},
        "schema": {"key": "boxes", "keypoints": [], "boxes": True},
        "specimens": specimens, "splits": {}, "labels": [], "boxes": [],
    }), encoding="utf-8")
    for name, names in (splits or {}).items():
        (folder / "splits").mkdir(exist_ok=True)
        (folder / "splits" / f"{name}.txt").write_text(
            "\n".join(names) + "\n", encoding="utf-8")
    return folder


def specimen(image, x1, y1, x2, y2, split="train", index=1, capture=CAPTURE_A,
             width=1280, height=720):
    """One per-specimen record, as the exporter writes it."""
    return {"file": f"/project/{capture}/b01_f00.png", "image": image,
            "split": split, "image_width": width, "image_height": height,
            "specimen_index": index, "keypoints": {}, "traits": {},
            "bbox": [x1, y1, x2, y2], "source": "manual"}


class ExportReadTests(unittest.TestCase):
    """Reading a folder: what is in it, and what it refuses."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name)

    def test_a_missing_folder_is_named_not_guessed(self):
        with self.assertRaises(ExportError) as caught:
            Export.read(self.folder / "nope")
        self.assertIn("not a folder", str(caught.exception))

    def test_a_folder_without_labels_json_is_refused(self):
        with self.assertRaises(ExportError) as caught:
            Export.read(self.folder)
        self.assertIn("labels.json", str(caught.exception))

    def test_unreadable_labels_are_reported(self):
        (self.folder / "labels.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(ExportError):
            Export.read(self.folder)

    def test_an_older_label_file_is_refused_with_a_reason(self):
        (self.folder / "labels.json").write_text(json.dumps({"labels": []}),
                                                 encoding="utf-8")
        with self.assertRaises(ExportError) as caught:
            Export.read(self.folder)
        self.assertIn("specimens", str(caught.exception))

    def test_it_reads_one_box_per_specimen(self):
        folder = write_export(self.folder, [
            specimen("images/a.png", 10, 20, 110, 95, index=1),
            specimen("images/a.png", 300, 40, 420, 180, index=2)])
        export = Export.read(folder)
        self.assertEqual(len(export.boxes()), 2)
        self.assertEqual([box.specimen_index for box in export.boxes()], [1, 2])
        self.assertEqual(export.boxes()[0].bbox, (10.0, 20.0, 110.0, 95.0))
        self.assertEqual(export.boxes()[0].image.name, "a.png")

    def test_boxes_are_normalised_and_sizes_derived(self):
        folder = write_export(self.folder, [
            specimen("images/a.png", 110, 95, 10, 20)])
        box = Export.read(folder).boxes()[0]
        self.assertEqual(box.bbox, (10.0, 20.0, 110.0, 95.0))
        self.assertAlmostEqual(box.width, 100.0)
        self.assertAlmostEqual(box.height, 75.0)
        self.assertAlmostEqual(box.area, 7500.0)

    def test_a_specimen_without_a_box_is_skipped_but_the_frame_is_kept(self):
        """A keypoint-only record still names a frame; it has nothing to train."""
        folder = write_export(self.folder, [
            {"file": f"/project/{CAPTURE_A}/b01.png", "image": "images/a.png",
             "split": "train", "image_width": 100, "image_height": 100,
             "specimen_index": 1, "keypoints": {"head": [1, 2]}, "traits": {},
             "bbox": None}])
        export = Export.read(folder)
        self.assertEqual(export.boxes(), ())
        self.assertEqual(len(export.frames()), 1)

    def test_a_missing_image_keeps_the_reference_and_drops_the_frame(self):
        folder = write_export(self.folder, [specimen("images/a.png", 1, 1, 9, 9)])
        (folder / "images" / "a.png").unlink()
        export = Export.read(folder)
        self.assertEqual(len(export.boxes()), 1)              # the label survives
        self.assertEqual(export.frames(), ())                 # the frame does not
        self.assertEqual(len(export.frames(existing_only=False)), 1)

    def test_splits_come_from_the_split_files(self):
        folder = write_export(
            self.folder,
            [specimen("images/a.png", 1, 1, 9, 9, split="train"),
             specimen("images/b.png", 1, 1, 9, 9, split="val")],
            splits={"train": ["images/a.png"], "val": ["images/b.png"]})
        export = Export.read(folder)
        self.assertEqual([path.name for path in export.images_in("train")],
                         ["a.png"])
        self.assertEqual([path.name for path in export.images_in("val")],
                         ["b.png"])
        self.assertEqual(len(export.boxes_in("test")), 0)

    def test_splits_are_rebuilt_from_the_records_when_the_files_are_absent(self):
        folder = write_export(self.folder, [
            specimen("images/a.png", 1, 1, 9, 9, split="val")])
        export = Export.read(folder)
        self.assertEqual([path.name for path in export.images_in("val")],
                         ["a.png"])


class ExportGroupTests(unittest.TestCase):
    """The leak-safety view, and the regrouped corpus it produces."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name)
        self.export = Export.read(write_export(self.folder, [
            specimen("images/a1.png", 1, 1, 9, 9, split="train",
                     capture=CAPTURE_A, index=1),
            specimen("images/a2.png", 2, 2, 9, 9, split="val",
                     capture=CAPTURE_A, index=2),
            specimen("images/b1.png", 3, 3, 9, 9, split="test",
                     capture=CAPTURE_B)]))

    def test_the_group_is_the_capture_the_frame_came_from(self):
        self.assertEqual(sorted(self.export.groups()), ["OBS_A", "OBS_B"])

    def test_it_reports_a_capture_that_was_split(self):
        self.assertEqual(self.export.leaking_groups(), ("OBS_A",))

    def test_warnings_name_the_leak_and_the_command_that_avoids_it(self):
        notes = " ".join(self.export.warnings())
        self.assertIn("capture", notes)
        self.assertIn("--split-by group", notes)

    def test_regrouping_holds_a_whole_capture_out(self):
        regrouped = self.export.regroup()
        self.assertEqual(regrouped.leaking_groups(), ())

    def test_regrouping_is_deterministic(self):
        self.assertEqual(self.export.regroup().splits,
                         self.export.regroup().splits)

    def test_regrouping_keeps_every_box(self):
        regrouped = self.export.regroup()
        self.assertEqual(len(regrouped.boxes()), len(self.export.boxes()))
        self.assertEqual({box.relative for box in regrouped.boxes()},
                         {box.relative for box in self.export.boxes()})


class ExportWarningTests(unittest.TestCase):
    """What the corpus tells a training run about itself."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name)

    def test_a_keypoint_only_export_says_a_detector_cannot_train_on_it(self):
        export = Export.read(write_export(self.folder, [
            {"file": "/project/x.png", "image": "images/a.png", "split": "train",
             "image_width": 10, "image_height": 10, "specimen_index": 1,
             "keypoints": {}, "traits": {}, "bbox": None}]))
        self.assertEqual(len(export.warnings()), 1)
        self.assertIn("bounding box", export.warnings()[0])

    def test_a_box_that_leaves_its_image_is_reported_not_silently_kept(self):
        export = Export.read(write_export(self.folder, [
            specimen("images/a.png", 10.0, -8.0, 90.0, 60.0)]))
        self.assertTrue(export.boxes()[0].outside())
        self.assertIn("leave their image", " ".join(export.warnings()))

    def test_a_box_touching_the_border_is_not_outside(self):
        export = Export.read(write_export(self.folder, [
            specimen("images/a.png", 0.0, 0.0, 1280.0, 720.0)]))
        self.assertFalse(export.boxes()[0].outside())

    def test_a_box_too_small_to_be_an_animal_is_outside(self):
        export = Export.read(write_export(self.folder, [
            specimen("images/a.png", 10.0, 10.0, 11.0, 11.0)]))
        self.assertTrue(export.boxes()[0].outside())

    def test_mixed_image_sizes_are_flagged(self):
        export = Export.read(write_export(self.folder, [
            specimen("images/a.png", 1, 1, 9, 9),
            specimen("images/b.png", 1, 1, 9, 9, width=640, height=480)]))
        self.assertIn("different image sizes", " ".join(export.warnings()))

    def test_missing_images_are_counted(self):
        export = Export.read(write_export(self.folder, [
            specimen("images/a.png", 1, 1, 9, 9)]))
        (self.folder / "images" / "a.png").unlink()
        self.assertIn("no image in images/", " ".join(export.warnings()))

    def test_the_summary_counts_everything_a_report_needs(self):
        export = Export.read(write_export(self.folder, [
            specimen("images/a.png", 1, 1, 9, 9, index=1),
            specimen("images/a.png", 20, 20, 40, 40, index=2)],
            splits={"train": ["images/a.png"]}))
        summary = export.stats_summary()
        self.assertEqual(summary["boxes"], 2)
        self.assertEqual(summary["images"], 1)
        self.assertEqual(summary["groups"], 1)
        self.assertEqual(summary["splits"]["train"]["boxes"], 2)
        self.assertTrue(export.describe())


class RealCorpusTests(unittest.TestCase):
    """The corpus on this machine, when it is there - a smoke test on real data."""

    def test_the_shipped_dataset_reads_and_reports_its_shape(self):
        folder = paths.PROJECT_ROOT / "datasets" / "bb"
        if not (folder / "labels.json").exists():
            self.skipTest("no datasets/bb export on this machine")
        export = Export.read(folder)
        self.assertGreater(len(export.boxes()), 0)
        relatives = {box.relative for box in export.boxes()}
        self.assertEqual(len(export.frames()), len(relatives))
        # Several individuals on one frame is this corpus's shape, not an accident.
        self.assertGreaterEqual(
            max(len(export.boxes_of(relative)) for relative in relatives), 2)
        # Regrouping always removes the leak the export's own split has.
        self.assertEqual(export.regroup().leaking_groups(), ())


class ReadBoxesTests(unittest.TestCase):
    """Opening the boxes that already exist: a folder, or the ``boxes.csv`` itself.

    This is the entry point of the recursive annotator ("annotate the boxes we
    already have"), so it must accept what an operator actually has on disk - and
    it must **refuse** a file it cannot read rather than producing plausible,
    shifted numbers.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name) / "bb"
        (self.folder / "images").mkdir(parents=True)
        self.photo = self.folder / "images" / "f00.png"
        self.photo.write_bytes(b"\x89PNG\r\n\x1a\n")

    def _boxes_csv(self, header=None, rows=(), name="boxes.csv"):
        path = self.folder / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header or ("box_id", "file", "image", "split",
                                       "image_width", "image_height",
                                       "specimen_index", "x1", "y1", "x2", "y2",
                                       "width", "height", "source", "labeled_by",
                                       "created_at"))
            for row in rows:
                writer.writerow(row)
        return path

    def _row(self, index=1, x1=10.0, y1=20.0, x2=110.0, y2=95.0,
             source="model", image="images/f00.png"):
        return ("BOX_%d" % index, "/elsewhere/b01_f00.png", image, "train",
                1280, 720, index, x1, y1, x2, y2, x2 - x1, y2 - y1, source, "os",
                "2026-10-17 00:00:00")

    def test_a_csv_opens_without_a_labels_json(self):
        self._boxes_csv(rows=[self._row(), self._row(2, 200.0, 30.0, 300.0, 140.0)])
        boxes = read_boxes(self.folder)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0].specimen_index, 1)
        self.assertEqual(boxes[0].bbox, (10.0, 20.0, 110.0, 95.0))
        self.assertEqual(boxes[0].source, "model")
        self.assertEqual(boxes[0].image_width, 1280)
        self.assertEqual(boxes[0].group, "elsewhere")

    def test_the_csv_file_itself_can_be_the_entry_point(self):
        path = self._boxes_csv(rows=[self._row()])
        self.assertEqual(len(read_boxes(path)), 1)

    def test_the_exported_image_wins_over_the_original_path(self):
        """The corpus travels; the original recording folder may be gone."""
        self._boxes_csv(rows=[self._row()])
        boxes = read_boxes(self.folder)
        self.assertEqual(boxes[0].image, self.photo)

    def test_the_original_path_is_used_when_the_export_has_no_copy(self):
        self._boxes_csv(rows=[self._row(image="images/missing.png")])
        original = self.folder / "original.png"
        original.write_bytes(b"\x89PNG\r\n\x1a\n")
        self._boxes_csv(rows=[("BOX_1", str(original), "images/missing.png", "train",
                               10, 10, 1, 1.0, 2.0, 5.0, 6.0, 4.0, 4.0, "manual",
                               "os", "now")])
        self.assertEqual(read_boxes(self.folder)[0].image, original)

    def test_labels_json_is_preferred_when_both_are_there(self):
        write_export(self.folder, [specimen("images/f00.png", 10, 20, 110, 95)])
        self._boxes_csv(rows=[self._row(x1=999.0, y1=999.0, x2=1000.0, y2=1000.0)])
        boxes = read_boxes(self.folder)
        self.assertEqual(boxes[0].bbox, (10.0, 20.0, 110.0, 95.0))

    def test_a_legacy_csv_is_refused_instead_of_misread(self):
        """The old writer's header had one name more than a row had values.

        Reading it positionally would map every column to the wrong name from
        ``image`` onwards - plausible numbers, silently wrong. So it is refused.
        """
        legacy = self._boxes_csv(
            header=("box_id", "file", "image", "split", "image_width",
                    "image_height", "specimen_index", "x1", "y1", "x2", "y2",
                    "width", "height", "source", "labeled_by", "created_at"),
            rows=[("BOX_1", "images/f00.png", "train", 1280, 720, 1, 10.0, 20.0,
                   110.0, 95.0, 100.0, 75.0, "manual", "os", "now")])
        with self.assertRaises(ExportError) as caught:
            read_boxes(legacy)
        self.assertIn("older exporter", str(caught.exception))

    def test_an_explicit_image_folder_wins_over_the_files_own_folder(self):
        """The case that needs it: the file here, the pictures somewhere else.

        A ``boxes.csv`` names its frames relative to the export folder, so an operator
        whose pictures moved - or who points at the ``images/`` folder they can see -
        has to be able to say so.
        """
        elsewhere = Path(self._tmp.name) / "moved_images"
        elsewhere.mkdir()
        moved = elsewhere / "f00.png"
        moved.write_bytes(b"\x89PNG\r\n\x1a\n")
        self._boxes_csv(rows=[self._row(image="images/f00.png")])
        boxes = read_boxes(self.folder, images_dir=elsewhere)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].image, moved)

    def test_the_images_folder_may_be_the_images_folder_itself(self):
        """What a file browser shows: ``.../bb/images`` is a folder one can pick."""
        self._boxes_csv(rows=[self._row()])
        boxes = read_boxes(self.folder / "boxes.csv",
                           images_dir=self.folder / "images")
        self.assertEqual(boxes[0].image, self.photo)

    def test_an_images_folder_that_does_not_exist_is_named(self):
        self._boxes_csv(rows=[self._row()])
        with self.assertRaises(ExportError) as caught:
            read_boxes(self.folder, images_dir=self.folder / "nope")
        self.assertIn("image folder does not exist", str(caught.exception))

    def test_a_folder_without_boxes_is_named(self):
        with self.assertRaises(ExportError) as caught:
            read_boxes(self.folder)
        self.assertIn("no boxes", str(caught.exception).lower())

    def test_a_path_that_is_neither_file_nor_folder_is_named(self):
        with self.assertRaises(ExportError) as caught:
            read_boxes(self.folder / "nope.csv")
        self.assertIn("not a file or folder", str(caught.exception))

    def test_a_row_without_usable_coordinates_is_named(self):
        self._boxes_csv(rows=[("BOX_1", "f", "images/f00.png", "train", 1, 1, 1,
                               "left", "top", "right", "bottom", 1, 1, "manual",
                               "os", "now")])
        with self.assertRaises(ExportError) as caught:
            read_boxes(self.folder)
        self.assertIn("x1/y1/x2/y2", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
