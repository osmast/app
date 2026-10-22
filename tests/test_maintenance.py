"""Data lifecycle: pruning, CSV export and consolidating a study."""

import tempfile
import unittest
from pathlib import Path

from daphnia import maintenance
from daphnia.db import connection, init_db, queries, repository


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        self.destination = Path(self._tmp.name) / "images"
        init_db(self.db_path)

        with connection(self.db_path) as conn:
            self.study_id = repository.create_study(
                conn, name="Trial", destination=self.destination)
            repository.upsert_profile(
                conn, profile_id="PROF_T", profile_name="t", microscope_model="m",
                objective_magnification="2x", camera_device="c", px_per_um=5.0)
            repository.activate_profile(conn, "PROF_T")
            self.session_id = repository.ensure_session(
                conn, study_id=self.study_id, profile_id="PROF_T")
            self.observation_id = repository.create_observation(
                conn, session_id=self.session_id, individual_index=1)
            self.kept = self._asset(conn, "kept.png", selection="kept", sharpness=9.0)
            self.discarded = self._asset(conn, "discarded.png",
                                         selection="discarded", sharpness=1.0)

    def _asset(self, conn, name, *, selection, sharpness):
        path = self.destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return repository.add_media_asset(
            conn, observation_id=self.observation_id, file_path=path,
            frame_index=0, burst_index=1, sharpness=sharpness, selection=selection)

    def _measure(self, conn, asset_id, spine_x=50.0):
        return repository.save_manual_measurement(
            conn, observation_id=self.observation_id, asset_id=asset_id,
            profile=repository.active_profile(conn),
            eye_apex=(0.0, 0.0), spine_base=(spine_x, 0.0))

    def test_prune_removes_only_the_discarded_frames(self):
        with connection(self.db_path) as conn:
            result = maintenance.prune_discarded(conn, self.study_id)
            remaining = [row["asset_id"] for row in
                         queries.assets_for(conn, self.observation_id)]
        self.assertEqual((result["rows"], result["files"]), (1, 1))
        self.assertFalse((self.destination / "discarded.png").exists())
        self.assertTrue((self.destination / "kept.png").exists())
        self.assertEqual(remaining, [self.kept])

    def test_prune_never_removes_a_measured_frame(self):
        with connection(self.db_path) as conn:
            self._measure(conn, self.discarded)
            result = maintenance.prune_discarded(conn, self.study_id)
        self.assertEqual(result["rows"], 0)
        self.assertTrue((self.destination / "discarded.png").exists())

    def test_export_writes_one_row_per_measurement(self):
        with connection(self.db_path) as conn:
            self._measure(conn, self.kept)
            path = self.destination / "out.csv"
            count = maintenance.export_measurements_csv(conn, self.study_id, path)
        self.assertEqual(count, 1)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)                # header + one measurement
        self.assertIn("length_um", lines[0])
        self.assertIn("10.000", lines[1])              # 50 px / (5 px/um) = 10 um

    def test_consolidate_prunes_exports_and_closes_the_study(self):
        with connection(self.db_path) as conn:
            self._measure(conn, self.kept)
            result = maintenance.consolidate_and_close(conn, self.study_id)
            study = queries.study_by_id(conn, self.study_id)
        self.assertEqual(result["pruned"]["rows"], 1)
        self.assertEqual(result["exported"], 1)
        self.assertTrue(Path(result["csv"]).exists())
        self.assertTrue(Path(result["manifest"]).exists())
        self.assertEqual(study["status"], "closed")
        self.assertIsNotNone(study["closed_at"])


if __name__ == "__main__":
    unittest.main()
