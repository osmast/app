"""The workflow gates: nothing may happen out of order."""

import tempfile
import unittest
from pathlib import Path

from daphnia import workflow
from daphnia.db import connection, init_db, repository

#: The phone connection lives outside the database, so the gates are asked with it.
CONNECTED = {"connection": True}


class GateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        self.destination = Path(self._tmp.name) / "images"
        init_db(self.db_path)

    # -- helpers -----------------------------------------------------------
    def _study(self, conn):
        return repository.create_study(conn, name="Trial", destination=self.destination)

    def _calibrate(self, conn):
        repository.upsert_profile(
            conn, profile_id="PROF_T", profile_name="t", microscope_model="m",
            objective_magnification="2x", camera_device="c", px_per_um=5.0)
        repository.activate_profile(conn, "PROF_T")

    def _capture(self, conn):
        """Capture a target and return (observation_id, first asset_id)."""
        study = conn.execute("SELECT study_id FROM studies").fetchone()["study_id"]
        session_id = repository.ensure_session(
            conn, study_id=study, profile_id="PROF_T")
        observation_id = repository.create_observation(conn, session_id=session_id)
        asset_id = repository.add_media_asset(
            conn, observation_id=observation_id, file_path="b1_f0.png",
            frame_index=0, burst_index=1, sharpness=12.0)
        return observation_id, asset_id

    # -- gates -------------------------------------------------------------
    def test_setup_is_reachable_from_a_fresh_database(self):
        with connection(self.db_path) as conn:
            self.assertTrue(workflow.can_enter(conn, workflow.index_of("experiment")))

    def test_calibrating_requires_a_study_and_the_connection(self):
        index = workflow.index_of("calibration")
        with connection(self.db_path) as conn:
            self.assertFalse(workflow.can_enter(conn, index))
            self.assertIn("study", workflow.blocking_reason(conn, index).lower())

            self._study(conn)
            self.assertFalse(workflow.can_enter(conn, index))          # still offline
            self.assertIn("connect", workflow.blocking_reason(conn, index).lower())
            self.assertTrue(workflow.can_enter(conn, index, CONNECTED))

    def test_acquiring_requires_a_study_a_scale_and_the_connection(self):
        index = workflow.index_of("acquire")
        with connection(self.db_path) as conn:
            self._study(conn)
            self._calibrate(conn)
            self.assertFalse(workflow.can_enter(conn, index))
            self.assertIn("connect", workflow.blocking_reason(conn, index).lower())
            self.assertTrue(workflow.can_enter(conn, index, CONNECTED))

    def test_selecting_requires_a_capture_but_not_the_connection(self):
        index = workflow.index_of("select")
        with connection(self.db_path) as conn:
            self._study(conn)
            self._calibrate(conn)
            self.assertFalse(workflow.can_enter(conn, index))

            self._capture(conn)
            self.assertTrue(workflow.can_enter(conn, index))            # works offline

    def test_annotating_requires_a_selection(self):
        with connection(self.db_path) as conn:
            self._study(conn)
            self._calibrate(conn)
            observation_id, asset_id = self._capture(conn)

            index = workflow.index_of("annotate")
            self.assertFalse(workflow.can_enter(conn, index))
            self.assertIn("select", workflow.blocking_reason(conn, index).lower())

            repository.select_best_frame(conn, observation_id, asset_id)
            self.assertTrue(workflow.can_enter(conn, index))

    def test_analytics_requires_a_measurement(self):
        with connection(self.db_path) as conn:
            self._study(conn)
            self._calibrate(conn)
            observation_id, asset_id = self._capture(conn)
            repository.select_best_frame(conn, observation_id, asset_id)

            index = workflow.index_of("visualize")
            self.assertFalse(workflow.can_enter(conn, index))

            repository.save_manual_measurement(
                conn, observation_id=observation_id, asset_id=asset_id,
                profile=repository.active_profile(conn),
                eye_apex=(0.0, 0.0), spine_base=(100.0, 0.0))
            self.assertTrue(workflow.can_enter(conn, index))

    def test_furthest_step_follows_the_facts(self):
        with connection(self.db_path) as conn:
            self.assertEqual(workflow.furthest_step(conn), workflow.index_of("experiment"))
            self._study(conn)
            # calibrating needs the connection, so the run stops here while offline
            self.assertEqual(workflow.furthest_step(conn), workflow.index_of("experiment"))
            self.assertEqual(workflow.furthest_step(conn, CONNECTED),
                             workflow.index_of("calibration"))
            self._calibrate(conn)
            self.assertEqual(workflow.furthest_step(conn, CONNECTED),
                             workflow.index_of("acquire"))

    def test_facts_are_scoped_to_the_active_study(self):
        """A fresh project must not inherit another project's progress."""
        with connection(self.db_path) as conn:
            self._study(conn)
            self._calibrate(conn)
            self._capture(conn)
        with connection(self.db_path) as conn:
            repository.create_study(conn, name="Second", destination=self.destination)
            state = workflow.facts(conn)
        self.assertTrue(state["study"])
        self.assertFalse(state["observation"])
        self.assertFalse(state["selection"])
        self.assertFalse(state["measurement"])

    def test_steps_are_distinct_and_ordered(self):
        routes = [step.route for step in workflow.STEPS]
        self.assertEqual(routes, ["dashboard", "experiment", "calibration", "acquire",
                                  "select", "annotate", "visualize"])


if __name__ == "__main__":
    unittest.main()
