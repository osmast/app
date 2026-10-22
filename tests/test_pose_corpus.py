"""The pose corpus contract: reading keypoints, grouping by capture, regrouping.

No PyTorch and no Qt: :mod:`daphnia.vision.pose_corpus` is pure stdlib, so the
contract that decides *which animals a validation score is measured on* is testable
on any machine. This is the pose twin of ``tests/test_export.py`` and exists for the
same reason: a split that leaks is invisible until a number looks too good.

The corpus is written to a temporary folder, so the test owns the data it asserts on
and the real ``datasets/kp`` can change underneath it without breaking anything.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from daphnia.vision.pose_corpus import (  # noqa: E402
    PoseCorpus,
    PoseError,
    _capture_of,
    _split_counts,
)

SCHEMA = {
    "key": "pose_head_tail",
    "keypoints": ["head", "spine base"],
    "export_names": ["head", "tail"],
}


def specimen(capture, frame, index, head, tail, bbox=(10, 10, 90, 80),
             split="train", width=1280, height=720, traits=None):
    """One per-specimen record as the exporter writes it."""
    stem = f"{capture}_{frame}_{index:08d}"
    return {
        "file": f"/original/{capture}/{stem}.png",
        "image": f"images/{stem}_deadbeef.png",
        "split": split,
        "image_width": width,
        "image_height": height,
        "specimen_index": index,
        "keypoints": {"head": list(head), "tail": list(tail)},
        "traits": traits or {"length": {"length_px": 70.0, "length_um": 5200.0}},
        "bbox": list(bbox),
    }


def write_corpus(folder, specimens, splits=None):
    """Write ``labels.json`` (and optionally ``splits/``) into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": {"name": "t", "schema_key": "pose_head_tail"},
               "schema": dict(SCHEMA), "specimens": list(specimens)}
    (folder / "labels.json").write_text(json.dumps(payload), encoding="utf-8")
    if splits:
        (folder / "splits").mkdir(exist_ok=True)
        for name, names in splits.items():
            (folder / "splits" / f"{name}.txt").write_text(
                "\n".join(names), encoding="utf-8")
    return folder


class CaptureGroupTests(unittest.TestCase):
    """The group key: the capture, recovered from the frame stem or its path."""

    def test_the_capture_is_the_first_token_of_the_stem(self):
        self.assertEqual(_capture_of({"file": "/x/b01_f00_35755477.png"}), "b01")

    def test_the_exported_hash_suffix_is_ignored(self):
        self.assertEqual(
            _capture_of({"image": "images/b03_f02_abc12345_deadbeef.png"}), "b03")

    def test_a_stem_without_a_separator_falls_back_to_its_folder(self):
        self.assertEqual(_capture_of({"file": "/captures/session7/abc.png"}), "session7")

    def test_no_path_at_all_is_not_a_crash(self):
        self.assertEqual(_capture_of({}), "unknown")


class SplitCountTests(unittest.TestCase):
    """The balanced cuts: every split that can be non-empty gets a capture."""

    def test_ten_captures_split_six_two_two(self):
        self.assertEqual(_split_counts(10), (6, 2, 2))

    def test_four_captures_still_leave_a_validation_set(self):
        self.assertEqual(_split_counts(4), (2, 1, 1))

    def test_three_captures_give_one_each(self):
        self.assertEqual(_split_counts(3), (1, 1, 1))

    def test_two_captures_cannot_hold_one_out(self):
        self.assertEqual(_split_counts(2), (2, 0, 0))


class CorpusReadTests(unittest.TestCase):
    """Reading a folder: what is in it, and what it refuses."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "kp"

    def tearDown(self):
        self.tmp.cleanup()

    def test_it_reads_one_sample_per_keypointed_specimen(self):
        write_corpus(self.folder, [
            specimen("b01", "f00", 1, (10, 20), (80, 60)),
            specimen("b01", "f00", 2, (30, 40), (90, 70)),
        ])
        corpus = PoseCorpus.read(self.folder)
        self.assertEqual(len(corpus.samples), 2)
        self.assertEqual(corpus.names, ("head", "tail"))
        sample = corpus.samples[0]
        self.assertEqual(sample.group, "b01")
        self.assertEqual(sample.point("head"), (10.0, 20.0))
        self.assertEqual(sample.bbox, (10.0, 10.0, 90.0, 80.0))

    def test_a_box_only_specimen_is_not_a_pose_row(self):
        record = specimen("b01", "f00", 1, (10, 20), (80, 60))
        record["keypoints"] = {}
        write_corpus(self.folder, [record, specimen("b01", "f00", 2, (1, 2), (3, 4))])
        self.assertEqual(len(PoseCorpus.read(self.folder).samples), 1)

    def test_a_specimen_without_a_box_is_dropped_and_counted(self):
        record = specimen("b01", "f00", 1, (10, 20), (80, 60))
        del record["bbox"]
        write_corpus(self.folder, [record, specimen("b01", "f01", 1, (1, 2), (30, 40))])
        corpus = PoseCorpus.read(self.folder)
        self.assertEqual(len(corpus.samples), 1)
        self.assertEqual(corpus.dropped.get("no_box"), 1)
        self.assertTrue(any("no bounding box" in note for note in corpus.warnings()))

    def test_coincident_keypoints_are_a_degenerate_label(self):
        write_corpus(self.folder, [
            specimen("b01", "f00", 1, (50, 50), (50, 50)),      # a double click
            specimen("b01", "f01", 1, (10, 20), (80, 60)),
        ])
        corpus = PoseCorpus.read(self.folder)
        self.assertEqual(len(corpus.samples), 1)
        self.assertEqual(corpus.dropped.get("degenerate"), 1)

    def test_a_corpus_without_keypoints_is_refused(self):
        record = specimen("b01", "f00", 1, (0, 0), (0, 0))
        record["keypoints"] = {}
        write_corpus(self.folder, [record])
        with self.assertRaises(PoseError) as caught:
            PoseCorpus.read(self.folder)
        self.assertIn("no labelled keypoint", str(caught.exception))

    def test_a_missing_folder_is_refused_with_the_command(self):
        with self.assertRaises(PoseError) as caught:
            PoseCorpus.read(Path(self.tmp.name) / "nope")
        self.assertIn("Export one first", str(caught.exception))

    def test_the_length_conversions_are_both_available(self):
        write_corpus(self.folder, [specimen("b01", "f00", 1, (10, 20), (80, 60))])
        sample = PoseCorpus.read(self.folder).samples[0]
        self.assertAlmostEqual(sample.um_per_px(), 5200.0 / 70.0)
        self.assertAlmostEqual(sample.px_per_um(), 70.0 / 5200.0)


class RegroupTests(unittest.TestCase):
    """Held out by capture, deterministic, and never an empty validation set."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "kp"

    def tearDown(self):
        self.tmp.cleanup()

    def _corpus(self, captures, split="train"):
        records = []
        for capture in captures:
            for frame in ("f00", "f01"):
                records.append(specimen(capture, frame, 1, (10, 20), (80, 60),
                                        split=split))
        write_corpus(self.folder, records)
        return PoseCorpus.read(self.folder)

    def _captures(self, count):
        return [f"b{n:02d}" for n in range(1, count + 1)]

    def test_captures_are_not_split_across_splits_after_regrouping(self):
        corpus = self._corpus(self._captures(10)).regroup()
        found = {}
        for sample in corpus.samples:
            found.setdefault(sample.group, set()).add(sample.split)
        self.assertTrue(all(len(splits) == 1 for splits in found.values()))

    def test_val_and_test_are_not_empty(self):
        corpus = self._corpus(self._captures(10)).regroup()
        self.assertTrue(corpus.samples_of("val"))
        self.assertTrue(corpus.samples_of("test"))

    def test_regrouping_is_reproducible(self):
        first = self._corpus(self._captures(10)).regroup()
        second = self._corpus(self._captures(10)).regroup()
        self.assertEqual(first.splits, second.splits)

    def test_a_capture_torn_across_splits_is_reported(self):
        """The leak the exporter's own split creates: one capture in two splits."""
        records = [
            specimen("b01", "f00", 1, (10, 20), (80, 60), split="train"),
            specimen("b01", "f01", 1, (10, 20), (80, 60), split="val"),
            specimen("b02", "f00", 1, (10, 20), (80, 60), split="train"),
        ]
        write_corpus(self.folder, records,
                     splits={"train": [records[0]["image"], records[2]["image"]],
                             "val": [records[1]["image"]], "test": []})
        corpus = PoseCorpus.read(self.folder)
        self.assertEqual(corpus.leaking_groups(), ("b01",))
        self.assertTrue(any("split across" in note for note in corpus.warnings()))


if __name__ == "__main__":
    unittest.main()
