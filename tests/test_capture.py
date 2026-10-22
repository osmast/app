"""Capture targets: wells or individuals, bursts, selection and annotation."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from daphnia.db import connection, init_db, queries, repository


class TargetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        init_db(self.db_path)
        with connection(self.db_path) as conn:
            self.study_id = repository.create_study(
                conn, name="Trial", destination=Path(self._tmp.name) / "images",
                target_mode="plates")           # this study iterates wells
            self.plate_id = repository.create_plate(
                conn, study_id=self.study_id, format="24")
            repository.upsert_profile(
                conn, profile_id="PROF_T", profile_name="t", microscope_model="m",
                objective_magnification="2x", camera_device="c", px_per_um=5.0)
            repository.activate_profile(conn, "PROF_T")
            self.session_id = repository.ensure_session(
                conn, study_id=self.study_id, profile_id="PROF_T")

    def _burst(self, conn, *, plate_id=None, well_label=None, individual_index=None):
        """Capture a two-frame burst for a target; returns (observation, burst)."""
        observation_id = repository.target_observation(
            conn, session_id=self.session_id, plate_id=plate_id,
            well_label=well_label, individual_index=individual_index)
        burst = repository.next_burst_index(conn, observation_id)
        for index in range(2):
            repository.add_media_asset(
                conn, observation_id=observation_id,
                file_path=f"b{burst}_f{index}.png", frame_index=index,
                burst_index=burst, sharpness=10.0 + index)
        return observation_id, burst

    def test_a_well_accumulates_its_bursts(self):
        with connection(self.db_path) as conn:
            observation_id, first = self._burst(
                conn, plate_id=self.plate_id, well_label="B02")
            again, second = self._burst(
                conn, plate_id=self.plate_id, well_label="B02")
            self.assertEqual(observation_id, again)
            self.assertEqual((first, second), (1, 2))
            self.assertEqual(queries.frame_counts(conn, observation_id),
                             {"frames": 4, "bursts": 2})

    def test_individuals_are_counted_separately(self):
        with connection(self.db_path) as conn:
            first, _ = self._burst(conn, individual_index=1)
            second, _ = self._burst(conn, individual_index=2)
            self.assertNotEqual(first, second)
            self.assertEqual(repository.next_individual_index(conn, self.study_id), 3)

    def test_targets_are_listed_with_their_labels(self):
        with connection(self.db_path) as conn:
            self._burst(conn, plate_id=self.plate_id, well_label="A01")
            self._burst(conn, individual_index=1)
            labels = sorted(queries.target_label(row)
                            for row in queries.list_targets(conn, self.study_id))
            self.assertEqual(labels, ["Individual 1", "Plate 1 A01"])

    def test_selection_drives_the_kept_set(self):
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            assets = queries.assets_for(conn, observation_id)
            repository.set_asset_selection(conn, assets[0]["asset_id"], "kept")
            repository.set_asset_selection(conn, assets[1]["asset_id"], "discarded")
            kept = queries.kept_assets(conn, observation_id)
            self.assertEqual([row["asset_id"] for row in kept],
                             [assets[0]["asset_id"]])

    def test_measurement_is_tied_to_the_annotated_frame(self):
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            asset = queries.assets_for(conn, observation_id)[0]
            repository.save_manual_measurement(
                conn, observation_id=observation_id, asset_id=asset["asset_id"],
                profile=repository.active_profile(conn),
                eye_apex=(0.0, 0.0), spine_base=(100.0, 0.0))
            row = queries.measurement_for_asset(conn, asset["asset_id"])
            self.assertIsNotNone(row)
            self.assertAlmostEqual(row["length_um"], 20.0)   # 100 px / 5 px per um

    def test_primary_frame_is_the_sharpest_of_the_target(self):
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            sharpest = conn.execute(
                "SELECT asset_id FROM media_assets WHERE observation_id = ?"
                " ORDER BY sharpness DESC LIMIT 1", (observation_id,)).fetchone()
            repository.set_primary_asset(conn, observation_id, sharpest["asset_id"])
            self.assertEqual(queries.primary_asset_path(conn, observation_id), "b1_f1.png")

    def test_unknown_selection_is_rejected(self):
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            asset_id = queries.assets_for(conn, observation_id)[0]["asset_id"]
            with self.assertRaises(ValueError):
                repository.set_asset_selection(conn, asset_id, "maybe")

    def test_selecting_the_best_frame_discards_the_others(self):
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            assets = queries.assets_for(conn, observation_id)
            repository.select_best_frame(conn, observation_id, assets[1]["asset_id"])
            self.assertEqual([row["asset_id"] for row in queries.kept_assets(conn, observation_id)],
                             [assets[1]["asset_id"]])
            states = {row["asset_id"]: row["selection"]
                      for row in queries.assets_for(conn, observation_id)}
            self.assertEqual(states[assets[0]["asset_id"]], "discarded")

    def test_clear_selection_resets_the_target(self):
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            assets = queries.assets_for(conn, observation_id)
            repository.select_best_frame(conn, observation_id, assets[0]["asset_id"])
            repository.clear_selection(conn, observation_id)
            states = {row["selection"] for row in queries.assets_for(conn, observation_id)}
            self.assertEqual(states, {"pending"})

    def test_sharpest_frame_is_picked(self):
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            asset_id = repository.select_sharpest_frame(conn, observation_id)
            self.assertEqual([row["asset_id"] for row in queries.kept_assets(conn, observation_id)],
                             [asset_id])
            self.assertEqual(
                asset_id, queries.assets_for(conn, observation_id)[1]["asset_id"])

    def test_annotation_queue_lists_every_kept_frame_of_the_study(self):
        with connection(self.db_path) as conn:
            well, _ = self._burst(conn, plate_id=self.plate_id, well_label="A01")
            individual, _ = self._burst(conn, individual_index=1)
            for observation_id in (well, individual):
                repository.select_sharpest_frame(conn, observation_id)
            queue = queries.kept_assets_all(conn, self.study_id)
            self.assertEqual(len(queue), 2)
            self.assertEqual(sorted(queries.target_label(row) for row in queue),
                             ["Individual 1", "Plate 1 A01"])
    def test_saving_the_same_frame_twice_updates_the_measurement(self):
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            asset = queries.assets_for(conn, observation_id)[0]
            profile = repository.active_profile(conn)
            first = repository.save_manual_measurement(
                conn, observation_id=observation_id, asset_id=asset["asset_id"],
                profile=profile, eye_apex=(0.0, 0.0), spine_base=(100.0, 0.0))
            second = repository.save_manual_measurement(
                conn, observation_id=observation_id, asset_id=asset["asset_id"],
                profile=profile, eye_apex=(0.0, 0.0), spine_base=(200.0, 0.0))
            self.assertEqual(first, second)                 # the same row was updated
            self.assertEqual(queries.counts(conn)["measurements"], 1)
            self.assertAlmostEqual(
                queries.measurement_for_asset(conn, asset["asset_id"])["length_um"], 40.0)

    def test_the_corpus_blocks_a_second_label_for_one_interval(self):
        """One measurement per (dataset, frame, specimen, trait) - the invariant.

        It used to be a unique index on the old ``measurements`` table; now it is the
        corpus's own ``UNIQUE``, which is what the ``measurements`` view reads back.
        """
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            asset = queries.assets_for(conn, observation_id)[0]
            repository.save_manual_measurement(
                conn, observation_id=observation_id, asset_id=asset["asset_id"],
                profile=repository.active_profile(conn),
                eye_apex=(0.0, 0.0), spine_base=(100.0, 0.0))
            self.assertEqual(queries.counts(conn)["measurements"], 1)
            row = conn.execute("SELECT * FROM corpus_labels").fetchone()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO corpus_labels (label_id, dataset_id, file_path,"
                    " specimen_index, trait, p1_x, p1_y, p2_x, p2_y, length_px,"
                    " labeled_by) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    ("LBL_DUPLICATE", row["dataset_id"], row["file_path"],
                     row["specimen_index"], row["trait"], 1.0, 1.0, 2.0, 2.0, 1.0, "os"))

    def test_one_frame_holds_several_specimens_and_traits(self):
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            asset = queries.assets_for(conn, observation_id)[0]
            profile = repository.active_profile(conn)
            for specimen in (1, 2):
                for trait in ("length", "width"):
                    repository.save_measurement(
                        conn, observation_id=observation_id,
                        asset_id=asset["asset_id"], profile=profile, trait=trait,
                        specimen_index=specimen, p1=(0.0, 0.0), p2=(100.0, 0.0))
            rows = queries.measurements_for_asset(conn, asset["asset_id"])
            self.assertEqual(len(rows), 4)                 # 2 specimens x 2 traits
            self.assertEqual({(r["specimen_index"], r["trait"]) for r in rows},
                             {(1, "length"), (1, "width"),
                              (2, "length"), (2, "width")})
            # Re-saving the same interval updates in place (never a 5th row).
            repository.save_measurement(
                conn, observation_id=observation_id, asset_id=asset["asset_id"],
                profile=profile, trait="width", specimen_index=2,
                p1=(0.0, 0.0), p2=(40.0, 0.0))
            stored = {(r["specimen_index"], r["trait"]): r
                      for r in queries.measurements_for_asset(conn, asset["asset_id"])}
            self.assertEqual(len(stored), 4)
            self.assertAlmostEqual(stored[(2, "width")]["length_um"], 8.0)  # 40 px / 5

    def test_annotation_queue_does_not_multiply_with_measurements(self):
        with connection(self.db_path) as conn:
            observation_id, _ = self._burst(
                conn, plate_id=self.plate_id, well_label="A01")
            repository.select_sharpest_frame(conn, observation_id)
            asset = queries.kept_assets(conn, observation_id)[0]
            profile = repository.active_profile(conn)
            for specimen in (1, 2, 3):
                repository.save_measurement(
                    conn, observation_id=observation_id,
                    asset_id=asset["asset_id"], profile=profile, trait="length",
                    specimen_index=specimen, p1=(0.0, 0.0), p2=(10.0, 0.0))
            queue = queries.kept_assets_all(conn, self.study_id)
            self.assertEqual(len(queue), 1)                # one frame, not three rows
            self.assertEqual(queue[0]["measurements"], 3)

if __name__ == "__main__":
    unittest.main()
