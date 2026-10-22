"""Study modes: what the spec derives, and that it survives the database."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from daphnia.db import connection, init_db, queries, repository
from daphnia.study import StudySpec, stage_of


def _row(**overrides):
    row = {
        "study_id": "S", "name": "Trial", "organism": "Daphnia magna",
        "organism_kind": "animal", "target_mode": "individuals",
        "timecourse": "longitudinal", "measurement_interval_hours": 24.0,
        "destination": "/tmp/images",
    }
    row.update(overrides)
    return row


class SpecTests(unittest.TestCase):
    def test_defaults_are_a_single_animal_measured_over_time(self):
        spec = StudySpec.from_row(_row())
        self.assertFalse(spec.uses_plates)
        self.assertTrue(spec.is_longitudinal)
        self.assertEqual(spec.keypoints, ("head", "spine base"))
        self.assertEqual(spec.endpoint, "body length")
        self.assertEqual(spec.specimen, "individual")

    def test_a_plant_study_annotates_different_landmarks(self):
        spec = StudySpec.from_row(
            _row(organism="Lemna", organism_kind="plant", target_mode="plates",
                 timecourse="single"))
        self.assertTrue(spec.uses_plates)
        self.assertFalse(spec.is_longitudinal)
        self.assertEqual(spec.keypoints, ("base", "tip"))
        self.assertEqual(spec.endpoint, "shoot length")
        self.assertEqual(spec.specimen, "plant")

    def test_an_unknown_kind_falls_back_to_the_animal_anatomy(self):
        spec = StudySpec.from_row(_row(organism_kind="fungus"))
        self.assertEqual(spec.keypoints, ("head", "spine base"))

    def test_describe_reads_like_the_setup_screen(self):
        spec = StudySpec.from_row(
            _row(organism_kind="plant", target_mode="plates", timecourse="single"))
        self.assertEqual(spec.describe(), "Plant / Multi-well plates / Measured once")

    def test_the_mode_survives_the_database(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "test.db"
        init_db(db_path)

        with connection(db_path) as conn:
            repository.create_study(
                conn, name="Plant trial", destination=Path(tmp.name) / "images",
                organism="Lemna", organism_kind="plant", target_mode="plates",
                timecourse="single", measurement_interval_hours=0.0)
        with connection(db_path) as conn:
            spec = queries.active_study_spec(conn)

        self.assertIsNotNone(spec)
        self.assertEqual((spec.organism_kind, spec.target_mode, spec.timecourse),
                         ("plant", "plates", "single"))
        self.assertTrue(spec.uses_plates)
        self.assertFalse(spec.is_longitudinal)


class ProjectChooserTests(unittest.TestCase):
    """The overview lists projects, and one of them is the current one."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        init_db(self.db_path)

    def _study(self, name, **kwargs):
        with connection(self.db_path) as conn:
            return repository.create_study(
                conn, name=name, destination=Path(self._tmp.name) / name, **kwargs)

    def test_a_new_study_becomes_the_current_project(self):
        self._study("First")
        second = self._study("Second")
        with connection(self.db_path) as conn:
            self.assertEqual(queries.active_study(conn)["study_id"], second)

    def test_switching_projects_marks_exactly_one_as_current(self):
        first = self._study("First")
        self._study("Second")
        with connection(self.db_path) as conn:
            repository.set_current_study(conn, first)
            rows = queries.list_studies_with_progress(conn)
            current = [row for row in rows if row["is_current"]]
        self.assertEqual([row["study_id"] for row in current], [first])
        self.assertEqual(rows[0]["study_id"], first)          # current project first

    def test_the_list_reports_the_progress_of_each_project(self):
        self._study("Empty", target_mode="plates")
        with connection(self.db_path) as conn:
            repository.create_plate(
                conn, study_id=queries.active_study(conn)["study_id"], format="6")
            row = queries.list_studies_with_progress(conn)[0]
        self.assertEqual((row["plates"], row["targets"], row["measured"]), (1, 0, 0))
        self.assertIn("ready to capture", stage_of(row))

    def test_stage_describes_how_far_a_project_got(self):
        base = {"plates": 0, "targets": 0, "frames": 0, "selected": 0, "measured": 0}
        self.assertIn("just created", stage_of(base))
        self.assertIn("ready to capture", stage_of({**base, "plates": 1}))
        self.assertIn("captured", stage_of({**base, "targets": 4}))
        self.assertIn("selected", stage_of({**base, "targets": 4, "selected": 1}))
        self.assertIn("annotated",
                      stage_of({**base, "targets": 4, "selected": 1, "measured": 1}))

    def test_a_legacy_database_adopts_a_current_project(self):
        """Studies created before the flag existed must not look invisible."""
        db_path = Path(self._tmp.name) / "legacy.db"
        legacy = sqlite3.connect(db_path)
        legacy.execute(
            "CREATE TABLE studies (study_id TEXT PRIMARY KEY, name TEXT NOT NULL,"
            " organism TEXT NOT NULL, measurement_interval_hours REAL NOT NULL,"
            " destination TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT)")
        legacy.execute("INSERT INTO studies VALUES"
                       " ('S1', 'Older', 'Daphnia magna', 24.0, '/tmp', 'closed', '2026-01-01')")
        legacy.execute("INSERT INTO studies VALUES"
                       " ('S2', 'Newer', 'Lemna', 24.0, '/tmp', 'active', '2026-02-01')")
        legacy.commit()
        legacy.close()

        init_db(db_path)

        with connection(db_path) as conn:
            current = queries.active_study(conn)
        self.assertIsNotNone(current)
        self.assertEqual(current["study_id"], "S2")          # the unfinished one wins



if __name__ == "__main__":
    unittest.main()
