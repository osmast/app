"""End-to-end data flow: study -> profile -> session -> observation -> measurement."""

import tempfile
import unittest
from pathlib import Path

from daphnia.db import connection, init_db, queries, repository
from daphnia.vision.calibration import OpticalProfile
from daphnia.vision.keypoints import Keypoints


class FlowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        self.destination = Path(self._tmp.name) / "images"
        init_db(self.db_path)

    def _study(self, conn):
        return repository.create_study(conn, name="Trial", destination=self.destination)

    def test_calibrate_capture_measure(self):
        with connection(self.db_path) as conn:
            study_id = self._study(conn)

            # 1. calibrate: 1000 px over 1000 um -> 1.0 px/um
            repository.upsert_profile(
                conn, profile_id="PROF_TEST", profile_name="test",
                microscope_model="Leica DM500", objective_magnification="4x",
                camera_device="iPhone", px_per_um=1.0)
            repository.insert_calibration_record(
                conn, profile_id="PROF_TEST", target_type="stage_micrometer",
                raw_image_path="ref.png", detected_distance_px=1000.0,
                actual_distance_um=1000.0)
            repository.activate_profile(conn, "PROF_TEST")

            profile = repository.active_profile(conn)
            self.assertIsNotNone(profile)
            self.assertAlmostEqual(profile.px_per_um, 1.0)

            # 2. capture one specimen in the study's session
            session_id = repository.ensure_session(
                conn, study_id=study_id, profile_id="PROF_TEST")
            observation_id = repository.create_observation(
                conn, session_id=session_id, well_label="E07")
            repository.add_media_asset(
                conn, observation_id=observation_id, file_path="frame_00.png",
                frame_index=0, sharpness=12.5, is_primary=True)

            # 3. measure: 500 px at 1 px/um -> 500 um
            measurement_id = repository.save_manual_measurement(
                conn, observation_id=observation_id, profile=profile,
                eye_apex=(10.0, 10.0), spine_base=(510.0, 10.0))
            self.assertTrue(measurement_id.startswith("LBL"))   # a corpus label now

            row = queries.measurement_for(conn, observation_id)
            self.assertAlmostEqual(row["length_px"], 500.0, places=6)
            self.assertAlmostEqual(row["length_um"], 500.0, places=6)
            self.assertEqual(row["px_per_um"], 1.0)
            self.assertEqual(row["source"], "manual")

            counts = queries.counts(conn)
            self.assertEqual(counts["studies"], 1)
            self.assertEqual(counts["observations"], 1)
            self.assertEqual(counts["assets"], 1)
            self.assertEqual(counts["measurements"], 1)

    def test_session_is_reused_within_the_day(self):
        with connection(self.db_path) as conn:
            study_id = self._study(conn)
            repository.upsert_profile(
                conn, profile_id="PROF_X", profile_name="x", microscope_model="m",
                objective_magnification="4x", camera_device="c", px_per_um=2.0)
            first = repository.ensure_session(
                conn, study_id=study_id, profile_id="PROF_X")
            second = repository.ensure_session(
                conn, study_id=study_id, profile_id="PROF_X")
            self.assertEqual(first, second)
            self.assertEqual(queries.counts(conn)["sessions"], 1)

    def test_recalibration_updates_the_profile_in_place(self):
        with connection(self.db_path) as conn:
            repository.upsert_profile(
                conn, profile_id="PROF_R", profile_name="v1", microscope_model="m",
                objective_magnification="4x", camera_device="c", px_per_um=1.0)
            repository.upsert_profile(
                conn, profile_id="PROF_R", profile_name="v2", microscope_model="m",
                objective_magnification="4x", camera_device="c", px_per_um=1.5)
            self.assertEqual(queries.counts(conn)["profiles"], 1)
            self.assertAlmostEqual(repository.profile_by_id(conn, "PROF_R").px_per_um, 1.5)

    def test_only_one_profile_is_active(self):
        with connection(self.db_path) as conn:
            for name, scale in (("A", 1.0), ("B", 2.0)):
                repository.upsert_profile(
                    conn, profile_id=f"PROF_{name}", profile_name=name,
                    microscope_model="m", objective_magnification="4x",
                    camera_device="c", px_per_um=scale)
                repository.activate_profile(conn, f"PROF_{name}")
            active = conn.execute(
                "SELECT COUNT(*) FROM calibration_profiles WHERE is_active = 1").fetchone()[0]
            self.assertEqual(active, 1)

    def test_scale_conversion_is_the_point_of_decoupled_calibration(self):
        profile = OpticalProfile(profile_id="P", px_per_um=0.5)
        keypoints = Keypoints(eye_apex=(0.0, 0.0), spine_base=(100.0, 0.0))
        self.assertAlmostEqual(profile.to_um(keypoints.length_px), 200.0)


if __name__ == "__main__":
    unittest.main()
