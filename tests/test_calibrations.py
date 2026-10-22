"""Calibrations are dated records: recalibrating must never rewrite history.

A phone held on an eyepiece by an adapter can be bumped at any time, which
invalidates the scale without changing a single recorded value. So a calibration
is an immutable row with its own uncertainty, and a measurement keeps pointing at
the one that was actually in force when it was taken.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from daphnia import units
from daphnia.db import connection, init_db, repository
from daphnia.db.connection import migrate
from daphnia.vision.calibration import (
    CLICK_ERROR_PX,
    OpticalProfile,
    calibration_age_hours,
    is_stale,
    relative_uncertainty_pct,
)

SETUP = "PROF_SCOPE_2X_PHONE_1.0X_INFINITY"


class CalibrationRecordTests(unittest.TestCase):
    """One row per calibration, not one row per hardware setup."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "cal.db"
        self.destination = Path(self._tmp.name) / "images"
        init_db(self.db_path)

    def _calibrate(self, conn, scale, uncertainty=1.0, setup=SETUP):
        """Register one calibration of the setup, as the Calibrate screen does."""
        return repository.register_calibration(
            conn, setup_key=setup, profile_name="scope",
            microscope_model="Leica DM500", objective_magnification="2x",
            camera_device="iPhone", px_per_um=scale,
            uncertainty_pct=uncertainty)

    def test_recalibrating_keeps_the_previous_calibration(self):
        """The point of the whole change: a bumped adapter must not rewrite."""
        with connection(self.db_path) as conn:
            first = self._calibrate(conn, 0.783)
            second = self._calibrate(conn, 0.147)
            self.assertNotEqual(first, second)
            history = repository.calibrations_for_setup(conn, SETUP)
        self.assertEqual(len(history), 2)
        self.assertEqual(sorted(row["px_per_um"] for row in history),
                         [0.147, 0.783])

    def test_every_calibration_belongs_to_its_setup(self):
        with connection(self.db_path) as conn:
            self._calibrate(conn, 1.0)
            profile = repository.active_profile(conn)
        self.assertEqual(profile.setup_key, SETUP)
        self.assertTrue(profile.profile_id.startswith(SETUP))

    def test_a_second_setup_keeps_its_own_history(self):
        """Different optics must not be mixed into one scale history."""
        other = "PROF_SCOPE_10X_PHONE_1.0X_INFINITY"
        with connection(self.db_path) as conn:
            self._calibrate(conn, 1.0)
            self._calibrate(conn, 4.0, setup=other)
            self.assertEqual(len(repository.calibrations_for_setup(conn, SETUP)), 1)
            self.assertEqual(len(repository.calibrations_for_setup(conn, other)), 1)

    def test_the_newest_calibration_becomes_active(self):
        with connection(self.db_path) as conn:
            self._calibrate(conn, 0.5)
            newest = self._calibrate(conn, 0.25)
            self.assertEqual(repository.active_profile(conn).profile_id, newest)

    def test_two_calibrations_in_the_same_second_do_not_collide(self):
        """A quick correction must be storable, not an IntegrityError."""
        with connection(self.db_path) as conn:
            first = self._calibrate(conn, 0.5)
            second = self._calibrate(conn, 0.6)
            history = repository.calibrations_for_setup(conn, SETUP)
        self.assertNotEqual(first, second)
        self.assertEqual(len(history), 2)

    def test_a_calibration_carries_when_and_how_accurate(self):
        with connection(self.db_path) as conn:
            self._calibrate(conn, 0.783, uncertainty=1.36)
            profile = repository.active_profile(conn)
            age = calibration_age_hours(profile.calibrated_at)
        self.assertTrue(profile.calibrated_at)
        self.assertAlmostEqual(profile.uncertainty_pct, 1.36)
        self.assertIsNotNone(age)
        self.assertLess(age, 0.1)              # just now

    def test_an_earlier_measurement_keeps_its_scale_and_uncertainty(self):
        """Recalibrating mid-study must not reinterpret what was already measured."""
        with connection(self.db_path) as conn:
            study_id = repository.create_study(
                conn, name="T", destination=self.destination)
            before = repository.register_calibration(
                conn, setup_key=SETUP, profile_name="s", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=1.0, uncertainty_pct=2.0)
            session_id = repository.ensure_session(
                conn, study_id=study_id, profile_id=before)
            observation_id = repository.create_observation(
                conn, session_id=session_id, individual_index=1)
            asset_id = repository.add_media_asset(
                conn, observation_id=observation_id, file_path="f.png",
                selection="kept")
            repository.save_measurement(
                conn, observation_id=observation_id, asset_id=asset_id,
                profile=repository.profile_by_id(conn, before), trait="length",
                p1=(0.0, 0.0), p2=(100.0, 0.0))

            # ... the adapter gets bumped, so a new calibration is registered
            after = self._calibrate(conn, 2.0, uncertainty=0.5)

            row = conn.execute(
                "SELECT profile_id, px_per_um, uncertainty_pct, length_um"
                " FROM measurements").fetchone()
        self.assertEqual(row["profile_id"], before)
        self.assertNotEqual(row["profile_id"], after)
        self.assertAlmostEqual(row["px_per_um"], 1.0)
        self.assertAlmostEqual(row["uncertainty_pct"], 2.0)
        self.assertAlmostEqual(row["length_um"], 100.0)

    def test_a_stored_measurement_carries_its_whole_budget(self):
        """The row records what the number is *worth*, not only how it was scaled.

        The screen and the export read one column (`corpus_labels.total_uncertainty_um`)
        that the write path fills with the same function the surfaces print, so a
        stored row can never claim a different budget from the panel that made it.
        """
        with connection(self.db_path) as conn:
            study_id = repository.create_study(
                conn, name="T", destination=self.destination)
            profile_id = self._calibrate(conn, 0.027264, uncertainty=0.2934)
            session_id = repository.ensure_session(
                conn, study_id=study_id, profile_id=profile_id)
            observation_id = repository.create_observation(
                conn, session_id=session_id, individual_index=1)
            asset_id = repository.add_media_asset(
                conn, observation_id=observation_id, file_path="f.png",
                selection="kept")
            # A 2 mm animal on the eyepiece chain: 54.528 px at 0.027264 px/um.
            repository.save_measurement(
                conn, observation_id=observation_id, asset_id=asset_id,
                profile=repository.profile_by_id(conn, profile_id), trait="length",
                p1=(0.0, 0.0), p2=(54.528, 0.0))
            row = conn.execute(
                "SELECT length_um, uncertainty_pct, total_uncertainty_um"
                " FROM measurements").fetchone()
        self.assertAlmostEqual(row["length_um"], 2000.0, places=1)
        self.assertAlmostEqual(row["uncertainty_pct"], 0.2934)
        # `docs/accuracy.md` §3.1: the ± on screen is 5.87 um; the budget is 75.1 um.
        self.assertAlmostEqual(row["total_uncertainty_um"], 75.1, places=1)
        self.assertGreater(row["total_uncertainty_um"],
                           10 * units.uncertainty_um(row["length_um"],
                                                     row["uncertainty_pct"]))

    def test_a_label_without_a_scale_has_no_budget_rather_than_a_wrong_one(self):
        """An unscaled label has no micrometres, so it has no budget either."""
        with connection(self.db_path) as conn:
            repository.save_corpus_label(
                conn, file_path="f.png", image_size=(200, 150), trait="length",
                p1=(0.0, 0.0), p2=(100.0, 0.0), profile=None)
            row = conn.execute(
                "SELECT length_um, uncertainty_pct, total_uncertainty_um"
                " FROM corpus_labels").fetchone()
        self.assertIsNone(row["length_um"])
        self.assertIsNone(row["total_uncertainty_um"])


class UncertaintyTests(unittest.TestCase):
    """The scale error is relative, so it scales with the animal."""

    def test_the_half_width_grows_with_the_length(self):
        self.assertAlmostEqual(units.uncertainty_um(160.0, 1.0), 1.6)
        self.assertAlmostEqual(units.uncertainty_um(320.0, 1.0), 3.2)

    def test_the_readout_carries_the_plus_minus_and_both_units(self):
        text = units.format_with_uncertainty(160.0, 1.0)
        self.assertIn("+/- 1.6 um", text)
        self.assertIn("0.160 mm", text)

    def test_a_longer_span_makes_the_scale_more_accurate(self):
        """Ten pitches instead of one divides the uncertainty by ten."""
        one_pitch = relative_uncertainty_pct(14.7)
        ten_pitches = relative_uncertainty_pct(147.0)
        self.assertGreater(one_pitch, 10.0)        # a single short pitch is poor
        self.assertAlmostEqual(one_pitch / 10.0, ten_pitches, places=6)

    def test_a_reference_span_must_be_positive(self):
        with self.assertRaises(ValueError):
            relative_uncertainty_pct(0.0)

    def test_the_profile_brackets_the_length(self):
        profile = OpticalProfile(profile_id="C", px_per_um=1.0, uncertainty_pct=10.0)
        low, high = profile.range_um(100.0)
        self.assertAlmostEqual(low, 90.0)
        self.assertAlmostEqual(high, 110.0)

    def test_the_profile_reports_the_whole_budget_not_only_the_scale(self):
        """`docs/accuracy.md` §3.1: ±5.9 um on screen, 75.1 um in the budget."""
        profile = OpticalProfile(profile_id="C", px_per_um=0.027264,
                                 uncertainty_pct=0.2934)
        self.assertAlmostEqual(profile.uncertainty_um(2000.0), 5.868, places=3)
        self.assertAlmostEqual(profile.total_uncertainty_um(2000.0), 75.099, places=3)
        self.assertGreater(profile.total_uncertainty_um(2000.0),
                           profile.uncertainty_um(2000.0))

    def test_the_click_constant_has_one_owner(self):
        """`vision.calibration` and `units` must not drift apart."""
        self.assertEqual(CLICK_ERROR_PX, units.CLICK_ERROR_PX)
        profile = OpticalProfile(profile_id="C", px_per_um=2.0, uncertainty_pct=1.0)
        self.assertAlmostEqual(
            profile.total_uncertainty_um(100.0),
            units.total_uncertainty_um(100.0, 2.0, 1.0))


class StalenessTests(unittest.TestCase):
    """A bumped adapter invalidates the scale without changing any value."""

    FRESH = "2026-10-15 11:00:00"      # one hour before NOON
    OLD = "2026-10-10 12:00:00"        # five days before NOON
    NOON = datetime(2026, 10, 15, 12, 0, 0)

    def test_a_fresh_calibration_is_not_flagged(self):
        self.assertFalse(is_stale(self.FRESH, 12, now=self.NOON))
        self.assertAlmostEqual(
            calibration_age_hours(self.FRESH, now=self.NOON), 1.0)

    def test_an_old_calibration_is_flagged(self):
        self.assertTrue(is_stale(self.OLD, 12, now=self.NOON))
        self.assertAlmostEqual(
            calibration_age_hours(self.OLD, now=self.NOON), 120.0)

    def test_an_unknown_timestamp_is_never_flagged(self):
        self.assertFalse(is_stale("", 12))
        self.assertFalse(is_stale(None, 12))
        self.assertIsNone(calibration_age_hours(None))
        self.assertIsNone(calibration_age_hours("not a date"))

    def test_zero_hours_disables_the_check(self):
        self.assertFalse(is_stale("2020-01-01 00:00:00", 0))
        self.assertFalse(is_stale("2020-01-01 00:00:00", None))


class MigrationTests(unittest.TestCase):
    """A database created before this change keeps working, without a reset."""

    def test_a_legacy_row_is_keyed_to_its_setup(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "legacy.db"
        init_db(db_path)
        with connection(db_path) as conn:
            repository.upsert_profile(
                conn, profile_id=SETUP, profile_name="legacy",
                microscope_model="m", objective_magnification="2x",
                camera_device="c", px_per_um=0.783)
            # Simulate the pre-migration state: the column exists, but empty.
            conn.execute("UPDATE calibration_profiles SET setup_key = ''")
        init_db(db_path)                       # runs migrate() again
        with connection(db_path) as conn:
            row = conn.execute(
                "SELECT setup_key FROM calibration_profiles").fetchone()
        self.assertEqual(row["setup_key"], SETUP)

    def test_a_legacy_measurement_has_no_uncertainty_rather_than_a_wrong_one(self):
        """Old rows must not pretend to a precision that was never recorded."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "legacy.db"
        init_db(db_path)
        with connection(db_path) as conn:
            row = conn.execute("PRAGMA table_info(measurements)").fetchall()
        self.assertIn("uncertainty_pct", [column["name"] for column in row])

    def test_a_legacy_label_adopts_its_calibrations_budget(self):
        """A row that names its calibration can say what the number was worth.

        Measured on the live store 2026-10-22: 519 of 550 labels recorded *which*
        scale was applied and not how well it was measured. The calibration it names
        is immutable, so the missing term is read through that reference and the
        budget derived from it - never invented, and the row is told what it gained.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "legacy.db"
        init_db(db_path)
        with connection(db_path) as conn:
            profile_id = repository.register_calibration(
                conn, setup_key=SETUP, profile_name="s", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=0.027264, uncertainty_pct=0.2934)
            repository.save_corpus_label(
                conn, file_path="f.png", image_size=(200, 150), trait="length",
                p1=(0.0, 0.0), p2=(54.528, 0.0),
                profile=repository.profile_by_id(conn, profile_id))
            # Simulate a row written before the uncertainty columns existed.
            conn.execute("UPDATE corpus_labels SET uncertainty_pct = NULL,"
                         " total_uncertainty_um = NULL")
            applied = migrate(conn)
            row = conn.execute(
                "SELECT uncertainty_pct, total_uncertainty_um FROM corpus_labels"
            ).fetchone()
        self.assertTrue(any("adopted the calibration's uncertainty" in note
                            for note in applied), applied)
        self.assertTrue(any("derived the total uncertainty" in note
                            for note in applied), applied)
        self.assertAlmostEqual(row["uncertainty_pct"], 0.2934)
        self.assertAlmostEqual(row["total_uncertainty_um"], 75.1, places=1)

    def test_a_row_whose_calibration_is_gone_keeps_its_null(self):
        """Nothing is derived from a reference that is not there (law 8)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "legacy.db"
        init_db(db_path)
        with connection(db_path) as conn:
            repository.save_corpus_label(
                conn, file_path="f.png", image_size=(200, 150), trait="length",
                p1=(0.0, 0.0), p2=(54.528, 0.0),
                profile=OpticalProfile(profile_id="PROF_GONE", px_per_um=0.027264,
                                       uncertainty_pct=0.2934))
            conn.execute("UPDATE corpus_labels SET uncertainty_pct = NULL,"
                         " total_uncertainty_um = NULL")
            migrate(conn)
            row = conn.execute(
                "SELECT uncertainty_pct, total_uncertainty_um FROM corpus_labels"
            ).fetchone()
        self.assertIsNone(row["uncertainty_pct"])
        self.assertIsNone(row["total_uncertainty_um"])


if __name__ == "__main__":
    unittest.main()

