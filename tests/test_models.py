"""The model registry: which checkpoints exist, and which one is in force.

Qt-free and torch-free (``daphnia/models.py`` never imports either), so this runs
anywhere: it is the knowledge the header's *Models* menu and any surface that wants to
know "which model is the lab using" both read.
"""

import json
import tempfile
import unittest
from pathlib import Path

from daphnia import models
from daphnia.db import connection, init_db


class RegistryTests(unittest.TestCase):
    """A row per checkpoint, exactly one of them active per kind."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "models.db"
        init_db(self.path)

    def _checkpoint(self, name) -> Path:
        """A stand-in for a checkpoint file - the registry records paths, not weights."""
        target = Path(self._tmp.name) / name
        target.write_bytes(b"not really a checkpoint")
        return target

    def _register(self, kind, name, **fields):
        with connection(self.path) as conn:
            return models.register(conn, kind, self._checkpoint(name), **fields)

    def test_a_registered_checkpoint_is_active_and_readable(self):
        record = self._register("detector", "rtmdet_a.pt", preset="tiny",
                                trained_at="2026-10-17", classes=("daphnia",),
                                metrics={"ap50": 0.745})
        self.assertTrue(record.is_active)
        self.assertEqual(record.name, "rtmdet_a.pt")
        self.assertEqual(record.classes, ("daphnia",))
        self.assertEqual(record.metrics, {"ap50": 0.745})
        self.assertIn("rtmdet-tiny", record.label)
        self.assertIn("predicts: daphnia", record.detail)
        self.assertIn("ap50 0.745", record.detail)

    def test_only_one_model_of_a_kind_is_active(self):
        first = self._register("detector", "rtmdet_a.pt")
        second = self._register("detector", "rtmdet_b.pt")

        with connection(self.path) as conn:
            self.assertEqual(models.active(conn, "detector").path, second.path)
            self.assertEqual([row.is_active for row in models.records(conn, "detector")],
                             [True, False])
            models.activate_model(conn, "detector", first.path)
            self.assertEqual(models.active(conn, "detector").path, first.path)

    def test_the_two_kinds_do_not_compete(self):
        detector = self._register("detector", "rtmdet_a.pt", classes=("daphnia",))
        pose = self._register("pose", "rtmpose_a.pt", keypoints=("head", "tail"))
        with connection(self.path) as conn:
            self.assertEqual(models.active(conn, "detector").path, detector.path)
            self.assertEqual(models.active(conn, "pose").path, pose.path)
            self.assertEqual(len(models.records(conn)), 2)

    def test_registering_the_same_path_again_updates_the_same_row(self):
        """A re-trained checkpoint keeps its identity - the file - and its newer facts."""
        first = self._register("detector", "rtmdet_a.pt", preset="tiny",
                               metrics={"ap50": 0.5})
        again = self._register("detector", "rtmdet_a.pt", preset="small",
                               metrics={"ap50": 0.9})

        self.assertEqual(first.path, again.path)
        self.assertEqual(again.preset, "small")
        self.assertEqual(again.metrics, {"ap50": 0.9})
        with connection(self.path) as conn:
            self.assertEqual(len(models.records(conn, "detector")), 1)

    def test_an_unknown_path_cannot_be_made_active(self):
        self._register("detector", "rtmdet_a.pt")
        with connection(self.path) as conn:
            with self.assertRaises(models.ModelError) as raised:
                models.activate_model(conn, "detector", "/nope/rtmdet_z.pt")
            self.assertIn("not registered", str(raised.exception))
            # ... and the one that *was* registered is still the active one.
            self.assertEqual(models.active(conn, "detector").name, "rtmdet_a.pt")

    def test_an_unknown_kind_is_refused(self):
        with connection(self.path) as conn:
            with self.assertRaises(models.ModelError):
                models.register(conn, "classifier", "/nope/x.pt")
            with self.assertRaises(models.ModelError):
                models.active(conn, "keypoints")

    def test_a_checkpoints_own_payload_becomes_the_facts(self):
        """No number is invented: what is stored is what the checkpoint reported."""
        fields = models.record_from_payload(
            "detector", "/models/rtmdet.pt",
            {"preset": "tiny", "classes": ["daphnia"], "created_at": "2026-10-16",
             "metrics": {"ap50": 0.812, "precision": 0.61}, "epochs": 100})

        self.assertEqual(fields["preset"], "tiny")
        self.assertEqual(fields["classes"], ("daphnia",))
        self.assertEqual(fields["trained_at"], "2026-10-16")
        self.assertEqual(fields["metrics"]["ap50"], 0.812)
        self.assertEqual(fields["keypoints"], ())          # and nothing else

    def test_forget_removes_the_row_and_never_the_file(self):
        record = self._register("detector", "rtmdet_a.pt")
        with connection(self.path) as conn:
            self.assertTrue(models.forget(conn, "detector", record.path))
            self.assertIsNone(models.record(conn, "detector", record.path))
            self.assertFalse(models.forget(conn, "detector", record.path))
        self.assertTrue(Path(record.path).exists())

    def test_no_registry_is_an_empty_registry_not_an_error(self):
        """A fresh database has no models: the menus say so rather than failing."""
        with connection(self.path) as conn:
            self.assertEqual(models.records(conn), [])
            self.assertIsNone(models.active(conn, "detector"))

    def test_an_existing_database_gains_the_table(self):
        """`init_db` is idempotent and additive: an older zoo.db gains the registry."""
        with connection(self.path) as conn:
            conn.execute("DROP TABLE models")
        init_db(self.path)
        with connection(self.path) as conn:
            self.assertEqual(models.records(conn), [])


class ReportSyncTests(unittest.TestCase):
    """The training reports are how the app learns what `make train` just produced.

    `models/*_report.json` is the run's own account of itself (best checkpoint, its metrics,
    the validation size, the epoch), and reading it must stay **torch-free** - the app starts
    without an ML stack (`SR-CORE-011`).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "models.db"
        init_db(self.path)
        self.directory = Path(self._tmp.name) / "models"
        self.directory.mkdir()

    def _checkpoint(self, name) -> Path:
        target = self.directory / name
        target.write_bytes(b"not really a checkpoint")
        return target

    def _report(self, kind, best, final):
        checkpoint = self._checkpoint(best)
        last = self._checkpoint(final)
        payload = {
            "checkpoint": str(checkpoint),
            "last_checkpoint": str(last),
            "device": "mps",
            "corpus": {"schema": "boxes" if kind == "detector" else "pose_full_boxes",
                       "keypoints": ["head", "tail"] if kind == "pose" else 0},
            "best": {"ap50": 0.9579, "precision": 0.9, "recall": 0.98, "images": 15,
                     "median_error_px": 4.82, "oks": 0.60,
                     "length_median_error_um": 825.0},
            "history": {"best_epoch": 110, "best_ap50": 0.9579,
                        "best_median_error_px": 4.82},
        }
        (self.directory / models.REPORT_FILES[kind]).write_text(json.dumps(payload))
        return checkpoint, last

    def test_a_report_registers_both_files_and_activates_the_best(self):
        best, last = self._report("detector", "rtmdet_daphnia.pt", "rtmdet_daphnia_last.pt")
        with connection(self.path) as conn:
            models.sync_from_reports(conn, self.directory)
            records = {Path(row.path).name: row for row in models.records(conn, "detector")}
            active = models.active(conn, "detector")

        self.assertEqual(set(records), {"rtmdet_daphnia.pt", "rtmdet_daphnia_last.pt"})
        self.assertEqual(Path(active.path).name, best.name)
        # The report's numbers belong to the **best** file (law 9): the final epoch has none.
        self.assertEqual(records["rtmdet_daphnia.pt"].metrics["ap50"], 0.9579)
        self.assertEqual(records["rtmdet_daphnia.pt"].metrics["best_epoch"], 110)
        self.assertEqual(records["rtmdet_daphnia.pt"].metrics["val_images"], 15)
        self.assertEqual(records["rtmdet_daphnia_last.pt"].metrics, {})
        self.assertIn("its own report", records["rtmdet_daphnia.pt"].detail)

    def test_the_pose_report_brings_its_keypoints_and_metrics(self):
        self._report("pose", "rtmpose_daphnia.pt", "rtmpose_daphnia_last.pt")
        with connection(self.path) as conn:
            models.sync_from_reports(conn, self.directory)
            record = models.active(conn, "pose")

        self.assertEqual(record.keypoints, ("head", "tail"))
        self.assertEqual(record.metrics["median_error_px"], 4.82)
        self.assertEqual(record.metrics["oks"], 0.60)

    def test_the_best_file_takes_over_from_the_final_epoch(self):
        """Law 9: the final epoch is never left in force when the report names a best."""
        best, last = self._report("detector", "rtmdet_daphnia.pt", "rtmdet_daphnia_last.pt")
        with connection(self.path) as conn:
            models.register(conn, "detector", str(last))       # as an old load would
            self.assertEqual(Path(models.active(conn, "detector").path).name, last.name)
            models.sync_from_reports(conn, self.directory)
            self.assertEqual(Path(models.active(conn, "detector").path).name, best.name)

    def test_a_hand_loaded_checkpoint_is_left_alone(self):
        """A file the reports do not describe was the operator's own choice."""
        self._report("detector", "rtmdet_daphnia.pt", "rtmdet_daphnia_last.pt")
        mine = self._checkpoint("my_own_run.pt")
        with connection(self.path) as conn:
            models.register(conn, "detector", str(mine))
            models.sync_from_reports(conn, self.directory)
            self.assertEqual(Path(models.active(conn, "detector").path).name, mine.name)
            self.assertIn("rtmdet_daphnia.pt",
                          [Path(row.path).name for row in models.records(conn)])

    def test_a_missing_report_is_not_an_error(self):
        with connection(self.path) as conn:
            self.assertEqual(models.sync_from_reports(conn, self.directory), [])
            self.assertIsNone(models.active(conn, "detector"))

    def test_a_missing_checkpoint_is_skipped_not_registered(self):
        """A report whose file was moved away must not register a path that is not there."""
        self._report("detector", "rtmdet_daphnia.pt", "rtmdet_daphnia_last.pt")
        (self.directory / "rtmdet_daphnia.pt").unlink()
        with connection(self.path) as conn:
            models.sync_from_reports(conn, self.directory)
            self.assertEqual([Path(row.path).name for row in models.records(conn)],
                             ["rtmdet_daphnia_last.pt"])
            self.assertIsNone(models.active(conn, "detector"))

    def test_an_unreadable_report_raises_rather_than_being_ignored(self):
        (self.directory / models.REPORT_FILES["detector"]).write_text("{not json")
        with connection(self.path) as conn:
            with self.assertRaises(models.ModelError):
                models.sync_from_reports(conn, self.directory)


class AvailableCheckpointTests(unittest.TestCase):
    """The models **available** in the default directory, offered in the right menu.

    "Show me the models in the models folder": the registry says what is in force, and this
    says what is *there* - including a file a training run has just written, which nobody has
    loaded in the app yet.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)
        for name in ("rtmdet_daphnia.pt", "rtmdet_daphnia_last.pt", "rtmpose_daphnia.pt",
                     "rtmdet_notes.txt", "something_else.pt", "rtmpose_daphnia copy.pth"):
            (self.directory / name).write_bytes(b"not really a checkpoint")
        (self.directory / "subdir").mkdir()

    def test_only_the_kinds_own_checkpoints_are_offered(self):
        detectors = [path.name for path in models.checkpoints_in("detector", self.directory)]
        poses = [path.name for path in models.checkpoints_in("pose", self.directory)]
        self.assertEqual(detectors, ["rtmdet_daphnia.pt", "rtmdet_daphnia_last.pt"])
        self.assertEqual(poses, ["rtmpose_daphnia copy.pth", "rtmpose_daphnia.pt"])

    def test_a_file_that_names_no_kind_is_deliberately_not_offered(self):
        """"something_else.pt" is not a detector or a pose model as far as its name says."""
        listed = (models.checkpoints_in("detector", self.directory)
                  + models.checkpoints_in("pose", self.directory))
        self.assertNotIn("something_else.pt", [path.name for path in listed])
        self.assertNotIn("rtmdet_notes.txt", [path.name for path in listed])

    def test_a_missing_directory_is_an_empty_list_not_an_error(self):
        self.assertEqual(models.checkpoints_in("detector", self.directory / "gone"), [])

    def test_a_pose_checkpoint_is_labelled_as_a_pose_model(self):
        """The menu may not call a pose checkpoint `rtmdet-tiny` - it names the wrong model."""
        record = models.ModelRecord(kind="pose", path="/tmp/rtmpose_x.pt", name="rtmpose_x.pt",
                                    preset="tiny", trained_at="2026-10-17")
        self.assertIn("rtmpose-tiny", record.label)
        self.assertNotIn("rtmdet", record.label)


if __name__ == "__main__":
    unittest.main()
