"""Schema and persistence checks for the SQLite layer."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from daphnia.db import (column_exists, connection, init_db, table_exists,
                        table_names)

PROFILE_SQL = (
    "INSERT INTO calibration_profiles (profile_id, profile_name, microscope_model,"
    " objective_magnification, camera_device, px_per_um) VALUES (?, ?, ?, ?, ?, ?)"
)
PROFILE_ROW = ("PROF_TEST", "Test profile", "Leica DM500", "4x", "iPhone 15 Pro", 0.25)

RECORD_SQL = (
    "INSERT INTO calibration_records (record_id, profile_id, target_type, raw_image_path,"
    " detected_distance_px, actual_distance_um, calibrated_by) VALUES (?, ?, ?, ?, ?, ?, ?)"
)

SESSION_SQL = (
    "INSERT INTO imaging_sessions (session_id, plate_id, profile_id, day_number,"
    " elapsed_seconds, current_well_index, operator_name) VALUES (?, ?, ?, ?, ?, ?, ?)"
)


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        init_db(self.db_path)

    def test_tables_are_created(self):
        with connection(self.db_path) as conn:
            self.assertEqual(
                table_names(conn),
                [
                    "calibration_profiles",
                    "calibration_records",
                    "corpus_boxes",
                    "corpus_labels",
                    "datasets",
                    "imaging_sessions",
                    "import_batches",
                    "imported_frames",
                    "media_assets",
                    "models",
                    "observations",
                    "plates",
                    "studies",
                    "study_templates",
                ],
            )

    def test_measurements_is_a_view_over_the_corpus(self):
        """There is no `measurements` table: a measurement is a corpus label.

        The view exists so every reader written against the old table keeps working;
        the write path is the corpus (``repository.save_corpus_label``).
        """
        with connection(self.db_path) as conn:
            kind = conn.execute(
                "SELECT type FROM sqlite_master WHERE name = 'measurements'"
            ).fetchone()
            self.assertEqual(kind["type"], "view")
            columns = [row["name"] for row in conn.execute("PRAGMA table_info(measurements)")]
            for name in ("measurement_id", "observation_id", "asset_id", "trait",
                         "length_um", "individual_index", "measured_by", "rowid"):
                self.assertIn(name, columns)

    def test_init_is_idempotent(self):
        init_db(self.db_path)
        init_db(self.db_path)
        with connection(self.db_path) as conn:
            self.assertEqual(len(table_names(conn)), 14)

    def test_a_legacy_measurements_table_is_folded_into_the_corpus(self):
        """The old `measurements` table becomes corpus rows + a read-only view.

        The frame is not on disk, so the size the corpus needs is unknown: the row
        stays in the backup and is counted, never silently dropped.
        """
        legacy = Path(self._tmp.name) / "legacy.db"
        with connection(legacy) as conn:
            conn.executescript("""
                CREATE TABLE media_assets (
                    asset_id TEXT PRIMARY KEY, observation_id TEXT NOT NULL,
                    file_path TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'burst',
                    frame_index INTEGER, burst_index INTEGER,
                    is_primary INTEGER NOT NULL DEFAULT 0, sharpness REAL,
                    selection TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')));
                CREATE TABLE measurements (
                    measurement_id TEXT PRIMARY KEY, observation_id TEXT NOT NULL,
                    asset_id TEXT, profile_id TEXT NOT NULL,
                    specimen_index INTEGER NOT NULL DEFAULT 1,
                    trait TEXT NOT NULL DEFAULT 'length',
                    p1_x REAL NOT NULL, p1_y REAL NOT NULL,
                    p2_x REAL NOT NULL, p2_y REAL NOT NULL,
                    length_px REAL NOT NULL, length_um REAL NOT NULL,
                    px_per_um REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual', measured_by TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')));
                INSERT INTO media_assets (asset_id, observation_id, file_path)
                    VALUES ('AST_1', 'OBS_1', '/tmp/not-on-disk.png');
                INSERT INTO measurements (measurement_id, observation_id, asset_id,
                    profile_id, specimen_index, trait, p1_x, p1_y, p2_x, p2_y,
                    length_px, length_um, px_per_um, source, measured_by)
                    VALUES ('MEA_1', 'OBS_1', 'AST_1', 'PROF_1', 1, 'length',
                            1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 'manual', 'os');
            """)

        init_db(legacy)

        with connection(legacy) as conn:
            kind = conn.execute(
                "SELECT type FROM sqlite_master WHERE name = 'measurements'"
            ).fetchone()["type"]
            self.assertEqual(kind, "view")
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM measurements_legacy").fetchone()[0],
                1, "the old table is kept as a backup, never dropped")
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0],
                0, "the frame is not on disk, so nothing was silently invented")

    def test_foreign_keys_are_enabled_and_enforced(self):
        with connection(self.db_path) as conn:
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(RECORD_SQL, ("REC_BAD", "PROF_MISSING", "grid", "x", 1.0, 1.0, "os"))

    def test_generated_scale_column(self):
        with connection(self.db_path) as conn:
            conn.execute(PROFILE_SQL, PROFILE_ROW)
            conn.execute(
                RECORD_SQL,
                ("REC_1", "PROF_TEST", "stage_micrometer_100um", "ref.png", 3000.0, 1000.0, "os"),
            )
            value = conn.execute(
                "SELECT calculated_px_per_um FROM calibration_records WHERE record_id = 'REC_1'"
            ).fetchone()[0]
            self.assertAlmostEqual(value, 3.0, places=9)

    def test_is_active_check_constraint(self):
        with connection(self.db_path) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO calibration_profiles (profile_id, profile_name, microscope_model,"
                    " objective_magnification, camera_device, px_per_um, is_active)"
                    " VALUES ('P','n','m','4x','c',1.0,7)"
                )

    def test_session_defaults_and_resume_pointer(self):
        with connection(self.db_path) as conn:
            conn.execute(PROFILE_SQL, PROFILE_ROW)
            conn.execute(SESSION_SQL, ("SESS_1", "P4", "PROF_TEST", 3, 259200.0, 55, "os"))
            row = conn.execute(
                "SELECT session_status, current_well_index FROM imaging_sessions"
            ).fetchone()
            self.assertEqual(row["session_status"], "in_progress")
            self.assertEqual(row["current_well_index"], 55)

    def test_cascade_deletes_calibration_records(self):
        with connection(self.db_path) as conn:
            conn.execute(PROFILE_SQL, PROFILE_ROW)
            conn.execute(RECORD_SQL, ("REC_1", "PROF_TEST", "grid", "ref.png", 10.0, 1.0, "os"))
            conn.execute("DELETE FROM calibration_profiles WHERE profile_id = 'PROF_TEST'")
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM calibration_records").fetchone()[0], 0)


class LegacyDatabaseTests(unittest.TestCase):
    """A database created before the mode axes existed must be upgradeable."""

    def test_an_older_studies_table_gains_the_mode_columns(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "legacy.db"

        legacy = sqlite3.connect(db_path)
        legacy.execute(
            "CREATE TABLE studies (study_id TEXT PRIMARY KEY, name TEXT NOT NULL,"
            " organism TEXT NOT NULL, measurement_interval_hours REAL NOT NULL,"
            " destination TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT)")
        legacy.execute(
            "INSERT INTO studies VALUES ('STUDY_OLD', 'Old run', 'Daphnia magna',"
            " 24.0, '/tmp/images', 'active', '2026-01-01')")
        legacy.commit()
        legacy.close()

        init_db(db_path)                     # migrate() adds what is missing

        with connection(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM studies WHERE study_id = 'STUDY_OLD'").fetchone()
        self.assertEqual(row["name"], "Old run")          # data preserved
        self.assertEqual(row["target_mode"], "individuals")
        self.assertEqual(row["organism_kind"], "animal")
        self.assertEqual(row["timecourse"], "longitudinal")



def test_an_older_corpus_labels_table_gains_its_dataset_scope(self):
        """The corpus is *rebuilt*, never dropped: old labels survive the upgrade.

        The constraint change is the point. Before datasets existed, one frame could
        carry ground truth for **one** model only (``UNIQUE (file_path, specimen,
        trait)``); after the migration two models may label the same frame.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "legacy_corpus.db"

        legacy = sqlite3.connect(db_path)
        legacy.execute(
            "CREATE TABLE corpus_labels (label_id TEXT PRIMARY KEY,"
            " file_path TEXT NOT NULL, image_width INTEGER NOT NULL,"
            " image_height INTEGER NOT NULL, specimen_index INTEGER NOT NULL DEFAULT 1,"
            " trait TEXT NOT NULL DEFAULT 'length', p1_x REAL NOT NULL,"
            " p1_y REAL NOT NULL, p2_x REAL NOT NULL, p2_y REAL NOT NULL,"
            " length_px REAL NOT NULL, profile_id TEXT, px_per_um REAL,"
            " length_um REAL, source TEXT NOT NULL DEFAULT 'manual',"
            " labeled_by TEXT NOT NULL, note TEXT, created_at TEXT,"
            " UNIQUE (file_path, specimen_index, trait))")
        legacy.execute(
            "INSERT INTO corpus_labels (label_id, file_path, image_width,"
            " image_height, specimen_index, trait, p1_x, p1_y, p2_x, p2_y, length_px,"
            " labeled_by, created_at) VALUES ('LBL_OLD', '/tmp/a.png', 200, 150, 1,"
            " 'length', 10.0, 20.0, 110.0, 20.0, 100.0, 'os', '2026-01-01')")
        legacy.commit()
        legacy.close()

        init_db(db_path)                     # rebuilds corpus_labels + add datasets

        with connection(db_path) as conn:
            self.assertFalse(table_exists(conn, "corpus_labels_legacy"))
            row = conn.execute(
                "SELECT * FROM corpus_labels WHERE label_id = 'LBL_OLD'").fetchone()
            self.assertIsNone(row["dataset_id"])          # kept, unattributed
            self.assertAlmostEqual(row["length_px"], 100.0)
            # Two datasets may now label the same frame, specimen and trait.
            conn.execute(
                "INSERT INTO datasets (dataset_id, name, schema_key)"
                " VALUES ('DATA_A', 'a', 'pose_full'), ('DATA_B', 'b', 'pose_head_tail')")
            for dataset_id in ("DATA_A", "DATA_B"):
                conn.execute(
                    "INSERT INTO corpus_labels (label_id, dataset_id, file_path,"
                    " image_width, image_height, specimen_index, trait, p1_x, p1_y,"
                    " p2_x, p2_y, length_px, labeled_by)"
                    " VALUES (?, ?, '/tmp/a.png', 200, 150, 1, 'length', 1, 2, 3, 4,"
                    " 1, 'os')", (f"LBL_{dataset_id}", dataset_id))
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM corpus_labels").fetchone()[0], 3)


class CorpusRebuildTests(unittest.TestCase):
    """The dataset-scoped corpus is safe to migrate twice and to migrate onto itself."""

    def test_a_fresh_database_is_not_rebuilt(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "fresh.db"
        init_db(db_path)
        init_db(db_path)                     # idempotent: nothing to rebuild
        with connection(db_path) as conn:
            self.assertFalse(table_exists(conn, "corpus_labels_legacy"))
            self.assertTrue(column_exists(conn, "corpus_labels", "dataset_id"))


if __name__ == "__main__":
    unittest.main()
