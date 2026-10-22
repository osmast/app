"""Calibrate step: the grid, in the order preset -> reference -> scale.

The panel order is the path, and these tests follow it: the preset name is the chain
a scale belongs to, the grid square size is what turns a period in pixels into a
scale, and the result - measured or typed in - is stored back against the preset.
"""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFrame

from daphnia import paths, settings
from daphnia.db import connection, init_db, repository
from daphnia.ids import new_setup_key
from daphnia.ui.pages.calibration import (
    DEFAULT_PRESET_NAME,
    MANUAL_SOURCE,
    PATTERN_CHECKERBOARD,
    PATTERN_DOTS,
    PATTERN_LABELS,
    PATTERN_LINES,
    SOURCE_SPAN,
    TARGET_TYPE,
    Calibration,
    grid_label,
    scale_jump_is_suspicious,
    target_type,
)
from daphnia.vision.grid import GRID_PITCH_UM, PAPER_PITCH_TOLERANCE_PCT
from tests.test_grid import grid_image
from tests.test_lattice import eyepiece_image, pattern


class ScaleGuardTests(unittest.TestCase):
    """A scale that changes by more than 2x is almost always the wrong optics."""

    def test_a_tenfold_error_is_flagged(self):
        self.assertTrue(scale_jump_is_suspicious(0.783, 0.0783))
        self.assertTrue(scale_jump_is_suspicious(0.0783, 0.783))

    def test_a_normal_recalibration_passes(self):
        self.assertFalse(scale_jump_is_suspicious(0.80, 0.783))
        self.assertFalse(scale_jump_is_suspicious(1.5, 0.783))
        self.assertFalse(scale_jump_is_suspicious(0.4, 0.783))

    def test_the_first_calibration_is_never_flagged(self):
        self.assertFalse(scale_jump_is_suspicious(0.0783, None))
        self.assertFalse(scale_jump_is_suspicious(0.0783, 0.0))


class TargetTests(unittest.TestCase):
    """One target - printed grid paper - with the square size the operator sets."""

    def test_the_default_target_is_the_one_millimetre_grid(self):
        self.assertEqual(TARGET_TYPE, "calibrated_grid_1mm")
        self.assertEqual(GRID_PITCH_UM, 1000.0)
        self.assertIn("1 mm grid", grid_label(GRID_PITCH_UM))

    def test_a_finer_square_is_named_in_the_stored_record(self):
        """Half-millimetre paper must not be filed as a 1 mm grid."""
        self.assertEqual(target_type(500.0), "calibrated_grid_500um")
        self.assertEqual(target_type(GRID_PITCH_UM), TARGET_TYPE)
        self.assertIn("0.5 mm grid", grid_label(500.0))


class CalibrationPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._previous_images = paths.CALIBRATION_IMAGES_DIR
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "calibration-ui.db")
        init_db()
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        paths.CALIBRATION_IMAGES_DIR = cls._previous_images
        cls._tmp.cleanup()
        if cls._previous_db is None:
            os.environ.pop("DAPHNIA_DB", None)
        else:
            os.environ["DAPHNIA_DB"] = cls._previous_db

    def setUp(self):
        # Proof images must not land in the project tree, and each test needs a
        # database with nothing in force (the scale-jump guard would ask).
        paths.CALIBRATION_IMAGES_DIR = Path(self._tmp.name) / "proof"
        with connection() as conn:
            conn.execute("DELETE FROM calibration_profiles")
        self.page = Calibration()

    def _photo(self, period=147.3):
        """A real PNG of the grid, at this project's own scale.

        Large enough to hold several squares: a small frame would trip the
        "aim at more" warning and add noise to the assertions.
        """
        path = Path(self._tmp.name) / f"grid_{period}.png"
        frame = grid_image(height=900, width=1200, period=period, noise=0.03)
        cv2.imwrite(str(path), (frame * 255).astype("uint8"))
        return path

    # -- the three panels are the path, in order --------------------------
    def test_the_panels_are_named_in_order(self):
        """Three panels, in the order of the path (`SR-OPTICS-024`)."""
        for name, word in (("preset_group", "Optical chain"),
                           ("reference_group", "Target photo"),
                           ("scale_group", "Scale in force")):
            self.assertTrue(hasattr(self.page, name), name)
            self.assertIn(word, getattr(self.page, name).title())

    # -- panel 1: the preset name the scale belongs to --------------------
    def test_there_are_no_hardware_fields_left(self):
        """The name is the identity (`SR-OPTICS-004`): no hardware to type in twice."""
        for gone in ("microscope_model", "objective", "camera_device"):
            self.assertFalse(hasattr(self.page, gone), gone)

    def test_the_setup_key_follows_the_preset_name(self):
        self.page.profile_name.setText("Leica DM500 2x / iPhone")
        expected = new_setup_key("Leica DM500 2x / iPhone")
        self.assertEqual(expected, "PROF_LEICADM5002XIPHONE")
        self.assertEqual(self.page.setup_key(), expected)
        self.assertIn(expected, self.page.setup_key_label.text())

    def test_renaming_the_preset_moves_the_scale_to_another_chain(self):
        """The interconnection the panel order exists to make visible."""
        self.page.profile_name.setText("scope 2x")
        first = self.page.setup_key()
        self.page.profile_name.setText("scope 10x")
        self.assertNotEqual(first, self.page.setup_key())
        self.page._refresh_optics()
        self.assertIn(self.page.setup_key(), self.page.setup_key_label.text())

    def test_an_unnamed_preset_still_has_a_key(self):
        self.assertEqual(self.page.preset_name(), DEFAULT_PRESET_NAME)
        self.assertEqual(self.page.setup_key(),
                         new_setup_key(DEFAULT_PRESET_NAME))

    def test_it_says_when_nothing_is_in_force(self):
        self.assertIn("none yet", self.page.in_force_label.text().lower())

    def test_the_in_force_line_has_three_cases_and_never_contradicts_itself(self):
        """`SR-OPTICS-025` / D4: the empty name box is *not* "a different chain".

        With nothing typed there is no chain to compare against, so the panel reports the
        scale that is in force and stops claiming a name nobody wrote - which is how the
        header, panel 1 and the footer came to say three different things at once.
        """
        self._register(setup_key="PROF_SOMEONE_ELSE", px_per_um=0.5)

        self.page.profile_name.setText("")
        self.page._refresh_optics()
        empty = self.page.in_force_label.text()
        self.assertIn("any chain", empty)
        self.assertIn("500.00 px/mm", empty)

        self.page.profile_name.setText("scope 2x")
        self.page._refresh_optics()
        other = self.page.in_force_label.text()
        self.assertIn("other chain", other)
        self.assertIn("500.00 px/mm", other)

    def test_the_calibration_of_this_preset_is_reported_with_its_age(self):
        self.page.profile_name.setText("scope 2x")
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key=self.page.setup_key(), profile_name="x",
                microscope_model="m", objective_magnification="2x",
                camera_device="c", px_per_um=0.783, uncertainty_pct=1.36)
        self.page._refresh_optics()
        text = self.page.in_force_label.text()
        self.assertIn("783.00 px/mm", text)
        self.assertIn("± 1.36 %", text)
        self.assertIn("h ago", text)              # the age, not a raw timestamp
        self.assertNotIn("other chain", text)

    def test_a_stale_scale_says_re_verify_where_this_name_would_use_it(self):
        """`SR-OPTICS-013` on the panel: 12 h without re-verifying is a signal."""
        self.page.profile_name.setText("scope 2x")
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key=self.page.setup_key(), profile_name="x",
                microscope_model="m", objective_magnification="2x",
                camera_device="c", px_per_um=0.5, uncertainty_pct=1.0)
            conn.execute("UPDATE calibration_profiles SET last_verified_at = ?",
                         ("2000-01-01 00:00:00",))
        self.page._refresh_optics()
        self.assertIn("RE-VERIFY", self.page.in_force_label.text())

        # A stale row that belongs to *another* chain is not this chain's problem: the
        # footer already flags age for the database as a whole.
        self.page.profile_name.setText("another chain")
        self.page._refresh_optics()
        self.assertNotIn("RE-VERIFY", self.page.in_force_label.text())

    # -- panel 2: the target photo, and the lattice in it -------------------
    def test_the_tools_and_the_panel_buttons_are_the_same_two_actions(self):
        """`X-N13` and `SR-OPTICS-024` resolved as *both*, with one method behind each.

        The step's conclusion has to be on the panel its doctrine defines (panel 3), and
        the window's menus keep every action in the same place. Both entries call one
        method, so neither can drift - and neither is ever disabled (`X-N10`).
        """
        for key in ("detect", "save_profile"):
            action = self.page.workspace.action(key)
            self.assertIsNotNone(action)
            self.assertTrue(action.isEnabled(), key)
        self.assertEqual(self.page.workspace.action("detect").text(), "Measure grid")
        self.assertEqual(self.page.workspace.action("save_profile").text(),
                         "Store this scale")
        for button in (self.page.btn_use_frame, self.page.btn_open_photo,
                       self.page.btn_measure_grid, self.page.btn_store_scale):
            self.assertTrue(button.isEnabled(), button.text())

    def test_the_panel_buttons_do_what_the_menus_do(self):
        """A click on *Store this scale* stores - the operator's own report was D1."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.page.btn_store_scale.click()
        with connection() as conn:
            self.assertIsNotNone(repository.active_profile(conn))
        self.assertIn("Stored for", self.page.status_label.text())

    def test_measuring_without_a_photo_explains_what_is_missing(self):
        """Not a disabled button: the action says what it needs (`X-N10`)."""
        self.page.measure_grid()
        text = self.page.status_label.text().lower()
        self.assertIn("no photo", text)
        self.assertIn("live frame", text)
        self.assertIn("open photo", text)

    def test_the_photo_choices_live_on_the_panel_not_in_the_menus(self):
        """D8/D9: a choice belongs beside its panel - the page declares no File/Source."""
        groups = [title for title, _actions in self.page.workspace.tool_group.groups()]
        self.assertEqual(groups, ["Calibration"])
        self.assertFalse(hasattr(self.page, "btn_open_video"))

    def test_the_span_is_measured_in_squares_not_a_pitch(self):
        self.page.spin_squares.setValue(10)
        self.assertEqual(self.page.span_um(), 10 * GRID_PITCH_UM)
        self.assertIn("10000 um", self.page._manual_prompt())

    def test_the_grid_square_size_is_the_operators_to_set(self):
        """Finer paper is the honest way to calibrate finer optics."""
        self.assertEqual(self.page.grid_pitch_um(), GRID_PITCH_UM)
        self.page.spin_grid_mm.setValue(0.5)
        self.assertEqual(self.page.grid_pitch_um(), 500.0)
        self.page.spin_squares.setValue(10)
        self.assertEqual(self.page.span_um(), 5000.0)

    def _dot_photo(self, period=40.0):
        """A real PNG of a printed **dot** target - the point-based reference."""
        from tests.test_points import dot_grid

        path = Path(self._tmp.name) / f"dots_{period}.png"
        frame = dot_grid(period=period)
        cv2.imwrite(str(path), (frame * 255).astype("uint8"))
        return path

    def test_the_printed_pattern_is_the_operators_to_choose(self):
        """Lines and dots need different detectors, so the choice is explicit, not guessed."""
        values = [self.page.combo_pattern.itemData(index)
                  for index in range(self.page.combo_pattern.count())]
        self.assertIn(PATTERN_LINES, values)
        self.assertIn(PATTERN_DOTS, values)
        self.assertEqual(self.page.target_pattern(), PATTERN_LINES)  # the rig's own sheet

    def test_a_measured_dot_grid_is_named_as_dots(self):
        """The panel says 'dots' - the right unit for the estimator that answered."""
        self.page.combo_pattern.setCurrentIndex(
            self.page.combo_pattern.findData(PATTERN_DOTS))
        self.page.use_photo(self._dot_photo())
        self.page.measure_grid()
        self.assertIsNotNone(self.page._grid)
        self.assertIn("dots", self.page.label_result.text())
        self.assertAlmostEqual(self.page._grid.period_px / 40.0, 1.0, delta=0.03)

    def test_switching_the_pattern_drops_the_measurement_not_just_rescales_it(self):
        """A dot period is not a line period: keeping the old number would mis-file it."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.assertIsNotNone(self.page._grid)
        self.page.combo_pattern.setCurrentIndex(
            self.page.combo_pattern.findData(PATTERN_DOTS))
        self.assertIsNone(self.page._grid)
        self.assertIn("no lattice measured", self.page.grid_label.text().lower())

    # -- panel 2: the binarisation cut, and the results panel ---------------
    def test_the_binarisation_cut_is_the_operators_to_set(self):
        """Only where it acts: the line estimator never binarises, so the control is inert."""
        self.assertEqual(self.page.target_pattern(), PATTERN_LINES)
        self.assertFalse(self.page.check_threshold.isEnabled())
        self.assertIsNone(self.page.binarisation_threshold())

        self.page.combo_pattern.setCurrentIndex(
            self.page.combo_pattern.findData(PATTERN_DOTS))
        self.assertTrue(self.page.check_threshold.isEnabled())
        self.assertIsNone(self.page.binarisation_threshold())      # unticked = Otsu decides
        self.page.check_threshold.setChecked(True)
        self.page.spin_threshold.setValue(120.0)
        self.assertEqual(self.page.binarisation_threshold(), 120.0)

    def test_measuring_cuts_at_the_manual_level_when_the_operator_asks(self):
        """The escape hatch is used, not merely shown - and the estimate carries it."""
        self.page.combo_pattern.setCurrentIndex(
            self.page.combo_pattern.findData(PATTERN_DOTS))
        self.page.check_threshold.setChecked(True)
        self.page.spin_threshold.setValue(160.0)     # dots 64, sheet 255: this splits them
        self.page.use_photo(self._dot_photo())
        self.page.measure_grid()
        self.assertIsNotNone(self.page._grid)
        self.assertEqual(self.page._grid.threshold, 160.0)
        self.assertAlmostEqual(self.page._grid.period_px / 40.0, 1.0, delta=0.03)

    def test_measuring_draws_the_points_it_measured_from(self):
        """Pressing 'Measure grid' must leave a plot on the photo, not only a verdict."""
        self.page.combo_pattern.setCurrentIndex(
            self.page.combo_pattern.findData(PATTERN_DOTS))
        self.page.use_photo(self._dot_photo())
        self.page.measure_grid()
        drawn = self.page.canvas.reference_points
        self.assertIsNotNone(drawn)
        self.assertEqual(len(drawn), self.page._grid.reading_count)
        self.assertGreater(len(drawn), 0)

    def test_a_new_photo_clears_the_points_with_the_lattice(self):
        """The rings are evidence of one photo; they may not outlive it."""
        self.page.combo_pattern.setCurrentIndex(
            self.page.combo_pattern.findData(PATTERN_DOTS))
        self.page.use_photo(self._dot_photo())
        self.page.measure_grid()
        self.assertIsNotNone(self.page.canvas.reference_points)
        self.page.use_photo(self._photo())          # a line grid: a different photo
        self.assertIsNone(self.page.canvas.reference_points)
        self.assertIsNone(self.page.canvas.lattice)

    def test_the_line_grid_has_no_points_to_ring(self):
        """A line lattice is drawn from its estimate; there are no points of its own."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.assertIsNotNone(self.page.canvas.lattice)
        self.assertIsNone(self.page.canvas.reference_points)

    def test_the_results_button_sits_on_the_panel_under_measure(self):
        """Discoverable where the act is, not only in the window menu."""
        self.assertTrue(hasattr(self.page, "btn_results"))
        self.assertIn("Results", self.page.btn_results.text())
        layout = self.page.reference_group.layout()
        self.assertEqual(layout.indexOf(self.page.btn_results),
                         layout.indexOf(self.page.btn_measure_grid) + 1)

    def test_the_results_tool_previews_this_screen_own_estimators(self):
        """The panel must read the photo on screen and this page's own measurement (law 1)."""
        from unittest.mock import patch

        from daphnia.ui.pages import calibration as module

        self.page.combo_pattern.setCurrentIndex(
            self.page.combo_pattern.findData(PATTERN_DOTS))
        self.page.use_photo(self._dot_photo())
        captured = {}

        class StubDialog:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def exec(self):
                return 0

        with patch.object(module, "CalibrationResultsDialog", StubDialog):
            self.page._show_results()
        self.assertIs(captured["frame"], self.page._frame)
        self.assertEqual(captured["pattern"], PATTERN_DOTS)
        self.assertEqual(captured["pitch_um"], self.page.grid_pitch_um())
        estimate = captured["measure"](captured["frame"], PATTERN_DOTS, None)
        self.assertAlmostEqual(estimate.period_px / 40.0, 1.0, delta=0.03)

    def test_the_results_panel_measures_draws_and_reports_the_cut(self):
        from daphnia.ui.calibration_results import CalibrationResultsDialog

        self.page.combo_pattern.setCurrentIndex(
            self.page.combo_pattern.findData(PATTERN_DOTS))
        self.page.use_photo(self._dot_photo())
        dialog = CalibrationResultsDialog(
            frame=self.page._frame, patterns=PATTERN_LABELS, pitch_um=1000.0,
            pattern=PATTERN_DOTS, measure=self.page._measure_target,
            points=self.page._target_points, parent=self.page)
        self.assertIn("per printed cell", dialog.readout.text())
        self.assertIn("40.000 px", dialog.readout.text())
        self.assertIn("binarisation", dialog.readout.text().lower())
        self.assertFalse(dialog.image_label.pixmap().isNull())

    def test_the_results_panel_refuses_the_wrong_target_instead_of_guessing(self):
        """Switching pattern re-runs the other detector - and a dot photo is not a board."""
        from daphnia.ui.calibration_results import CalibrationResultsDialog

        self.page.use_photo(self._dot_photo())
        dialog = CalibrationResultsDialog(
            frame=self.page._frame, patterns=PATTERN_LABELS, pitch_um=1000.0,
            pattern=PATTERN_CHECKERBOARD, measure=self.page._measure_target,
            points=self.page._target_points, parent=self.page)
        self.assertIn("No grid measured", dialog.readout.text())
        self.assertIn("checker", dialog.readout.text().lower())

    def test_the_results_panel_says_so_when_there_is_no_photo(self):
        from daphnia.ui.calibration_results import CalibrationResultsDialog

        dialog = CalibrationResultsDialog(
            frame=None, patterns=PATTERN_LABELS, pitch_um=1000.0,
            pattern=PATTERN_DOTS, measure=self.page._measure_target,
            points=self.page._target_points, parent=self.page)
        self.assertIn("No photo", dialog.readout.text())

    def test_the_correction_button_sits_on_the_panel(self):
        self.assertTrue(hasattr(self.page, "btn_correction"))
        self.assertIn("Correction", self.page.btn_correction.text())

    def test_the_correction_panel_says_so_without_a_photo(self):
        from daphnia.ui.calibration_results import DistortionResultsDialog

        dialog = DistortionResultsDialog(frame=None, parent=self.page)
        self.assertFalse(dialog.run_button.isEnabled())
        self.assertIn("No photo", dialog.status.text())

    def test_the_correction_panel_shows_every_figure_after_the_probe(self):
        """All ten, before and after - and the model's own verdict beside them."""
        from daphnia.ui.calibration_results import DistortionResultsDialog
        from daphnia.vision.distortion import FIGURE_CAPTIONS
        from tests.test_points import dot_grid

        frame = (dot_grid(period=40.0, height=600, width=800) * 255).astype("uint8")
        dialog = DistortionResultsDialog(frame=frame, parent=self.page)
        dialog.run_probe()
        self.assertIn("residual", dialog.readout.text().lower())
        self.assertIn("discorpy's own check", dialog.readout.text())
        self.assertEqual(dialog.grid.count(), len(FIGURE_CAPTIONS))

    def test_the_plot_draws_the_points_it_is_given(self):
        """A drawing that changed nothing would be a lie about what was detected."""
        import numpy as np

        from daphnia.ui.calibration_results import plot_detection

        frame = np.full((80, 120), 200, dtype=np.uint8)
        plain = plot_detection(frame)
        marked = plot_detection(frame, points=np.array([[30.0, 40.0]]))
        self.assertEqual(plain.shape, marked.shape)
        self.assertFalse(np.array_equal(plain, marked))
        self.assertIsNone(plot_detection(None))

    def test_detecting_uses_the_square_size_that_is_set(self):
        """The period is measured in pixels; the square's size is what scales it.

        The same 147.3-px period is *twice as many pixels per micrometre* when the
        square is declared to be half a millimetre: the pixels now span half the
        distance, so the scale goes up, not down.
        """
        self.page.spin_grid_mm.setValue(0.5)
        self.page.use_photo(self._photo())          # a 1 mm grid at 147.3 px/mm
        self.page.measure_grid()
        self.assertAlmostEqual(self.page._grid.mean_period_px / 147.3, 1.0,
                               delta=0.02)
        self.assertAlmostEqual(
            self.page._pitch()[0] / (147.3 / 1000.0 * 2.0), 1.0, delta=0.02)

    def test_changing_the_square_size_re_scales_the_same_detection(self):
        """No second photo: the period in pixels is kept, the pitch is not."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        period = self.page._grid.mean_period_px
        before = self.page._pitch()[0]
        self.page.spin_grid_mm.setValue(2.0)        # a *bigger* square: fewer px/um
        self.assertEqual(self.page._grid.mean_period_px, period)
        self.assertAlmostEqual(self.page._pitch()[0], before * 0.5)
        self.assertIn("2 mm", self.page.grid_label.text())

    def test_the_reference_panel_reports_where_a_photo_can_come_from(self):
        text = self.page.stream_hint_label.text().lower()
        self.assertIn("live feed", text)
        self.assertIn("photo", text)

    def test_a_photo_file_can_be_used_instead_of_the_stream(self):
        path = self._photo()
        self.page.use_photo(path)
        self.assertIsNotNone(self.page._frame)
        self.assertIn(path.name, self.page.status_label.text())

    def test_an_unusable_photo_is_reported_and_changes_nothing(self):
        path = Path(self._tmp.name) / "broken.png"
        path.write_bytes(b"not an image at all")
        self.page.use_photo(self._photo())        # a good reference first
        self.page.use_photo(path)
        self.assertIsNone(self.page._frame)       # ... so the old one must not linger
        self.assertIsNone(self.page._grid)
        self.assertIsNone(self.page.canvas.frame)
        self.assertIn(path.name, self.page.canvas.refusal)
        self.assertIn("could not load", self.page.status_label.text().lower())

    def test_measuring_draws_the_lattice_it_measured(self):
        """`SR-OPTICS-026`: the evidence has to be visible, not only a number to trust."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.assertIsNotNone(self.page._grid)
        period, angle = self.page.canvas.lattice
        self.assertAlmostEqual(period / 147.3, 1.0, delta=0.02)
        self.assertEqual(angle, 0.0)
        self.assertIn("1 mm", self.page.grid_label.text())
        # A photo of the sheet in air has no field stop, and that is a real answer: then
        # the whole frame is paper and there is no circle to draw.
        self.assertIsNone(self.page._grid.field_stop)
        self.assertIsNone(self.page.canvas.field_stop)
        self.assertIn("none in view", self.page.label_field_stop.text())

    def test_the_result_block_carries_each_fact_once(self):
        """D5: the paragraph that used to be in two labels is four values and a tooltip."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        grid = self.page._grid
        self.assertIn(f"{grid.period_px:.3f} px/cell", self.page.label_result.text())
        self.assertIn(f"{grid.bands} bands", self.page.label_result.text())
        self.assertIn("tilt", self.page.label_result.text())
        self.assertIn(f"± {grid.spread_pct:.1f} %", self.page.label_drift.text())
        self.assertIn("across the field", self.page.label_drift.text())
        self.assertIn("Verdict:", self.page.label_verdict.text())
        # The wall of text is gone from both homes it used to fill.
        for label in (self.page.grid_label, self.page.status_label):
            self.assertNotIn("field stop", label.text().lower())
            self.assertNotIn("bands of the field", label.text().lower())

    def test_the_drift_is_reported_separately_from_the_uncertainty(self):
        """`SR-OPTICS-036`: two facts, two lines - one number for both was the bug."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.assertNotIn("Drift", self.page.uncertainty_label.text())
        self.assertNotIn("% across the field", self.page.uncertainty_label.text())
        self.assertIn("Drift:", self.page.label_drift.text())
        # ... and the drift's long form (with the estimator's own caveats) is one hover
        # away rather than printed into the panel.
        self.assertIn("place in the frame", self.page.label_drift.toolTip())

    def test_an_eyepiece_photo_draws_the_paper_circle_it_measured_inside(self):
        """The rig's real frame: a pale lattice in a disc on a black surround."""
        path = Path(self._tmp.name) / "eyepiece.png"
        frame, (centre_x, centre_y, radius) = eyepiece_image(period=27.0, radius=200)
        cv2.imwrite(str(path), (frame * 255).astype("uint8"))
        self.page.use_photo(path)
        self.page.measure_grid()

        self.assertIsNotNone(self.page._grid)
        drawn = self.page.canvas.field_stop
        self.assertIsNotNone(drawn, "the field stop was measured but not drawn")
        drawn_x, drawn_y, drawn_radius = drawn
        self.assertAlmostEqual(drawn_radius / radius, 1.0, delta=0.06)
        self.assertAlmostEqual(drawn_x, centre_x, delta=12.0)
        self.assertAlmostEqual(drawn_y, centre_y, delta=12.0)
        self.assertIn("paper circle", self.page.label_field_stop.text())
        self.assertIn("of the frame", self.page.label_field_stop.text())
        # The circle is evidence of the same photo as the lattice: a new frame clears both.
        self.page.use_photo(self._photo())
        self.assertIsNone(self.page.canvas.field_stop)

    def test_a_new_photo_clears_the_previous_measurement(self):
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.page.use_photo(self._photo(period=80.0))
        self.assertIsNone(self.page._grid)
        self.assertIsNone(self.page.canvas.lattice)
        self.assertIn("no lattice measured", self.page.grid_label.text().lower())

    def test_the_verdict_is_measured_when_the_field_agrees_with_the_print(self):
        """A clean, undistorted sheet: nothing for the operator to act on."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.assertLess(self.page._grid.spread_pct, PAPER_PITCH_TOLERANCE_PCT)
        self.assertIn("MEASURED", self.page.label_verdict.text())

    def test_the_verdict_is_weak_when_the_field_moves_more_than_the_print(self):
        """The rig's own optics: cells that grow towards the edge, against a flat print."""
        height, width = 600, 800
        rows, columns = np.mgrid[0:height, 0:width].astype(float)
        centre_x, centre_y = width / 2.0, height / 2.0
        blob = np.hypot(columns - centre_x, rows - centre_y) / centre_y
        stretch = 1.0 + 0.06 * blob ** 2              # ~6 % larger cells at the edge
        frame = pattern((columns - centre_x) / stretch + centre_x,
                        (rows - centre_y) / stretch + centre_y, 40.0)
        path = Path(self._tmp.name) / "drifted.png"
        cv2.imwrite(str(path), (frame * 255).astype("uint8"))
        self.page.use_photo(path)
        self.page.measure_grid()

        self.assertGreater(self.page._grid.spread_pct, PAPER_PITCH_TOLERANCE_PCT)
        self.assertIn("WEAK - the scale is right where it was measured",
                      self.page.label_verdict.text())
        self.assertIn("keep the specimen centred", self.page.label_verdict.text().lower())
        self.assertIn("panel 1", self.page.label_verdict.text())
        # The verdict names the meaning and the drift line carries the number, so the
        # percentage appears on the panel exactly once.
        self.assertNotIn("%", self.page.label_verdict.text())
        self.assertIn("keep the specimen close to the centre",
                      self.page.label_verdict.toolTip().lower())

    def test_a_frame_with_no_lattice_is_refused_with_its_reason(self):
        """`SR-OPTICS-017`: a wrong scale is worse than no scale, and the why is named."""
        path = Path(self._tmp.name) / "blank_plate.png"
        cv2.imwrite(str(path), np.full((600, 800, 3), 200, dtype="uint8"))
        self.page.use_photo(path)
        self.page.measure_grid()

        self.assertIsNone(self.page._grid)
        self.assertIsNone(self.page.canvas.lattice)
        self.assertIn("REFUSED", self.page.label_verdict.text())
        self.assertIn("no lattice in this frame", self.page.label_verdict.text())
        self.assertIn("panel 2", self.page.status_label.text())
        self.assertTrue(self.page.label_verdict.toolTip(),
                        "the estimator's own reason belongs in the tooltip, not nowhere")
        self.assertIn("periodicity", self.page.label_verdict.toolTip().lower())

    def test_nothing_measured_yet_says_so_before_it_says_anything_else(self):
        """Every state has a value, and the empty one says what to do (`X-N10`)."""
        self.assertIn("NOT MEASURED", self.page.label_verdict.text())
        for label in (self.page.label_result, self.page.label_drift,
                      self.page.label_field_stop):
            self.assertIn(": -", label.text())

    def test_the_screen_never_calls_the_drift_a_measured_distortion(self):
        """`SR-OPTICS-032` / D7: distortion is unmeasured while no model exists."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.assertIn("unmeasured and not corrected",
                      self.page.label_distortion.text())
        # One home for the word, and every other line is about the drift.
        for label in (self.page.grid_label, self.page.status_label,
                      self.page.label_drift, self.page.label_result,
                      self.page.label_verdict):
            self.assertNotIn("distortion", label.text().lower())

    def test_the_distortion_note_belongs_to_the_chain_not_to_a_measurement(self):
        """Both routes set one scale and neither corrects the lens - so it is panel 1's fact.

        It is also **optional** there: it only changes what you measure away from the
        centre of the field, which is the drift line's business in panel 2.
        """
        self.assertTrue(self.page.preset_group.isAncestorOf(self.page.label_distortion))
        self.assertFalse(self.page.reference_group.isAncestorOf(self.page.label_distortion))
        text = self.page.label_distortion.text().lower()
        self.assertIn("one scale covers the whole frame", text)
        self.assertIn("optional", text)
        self.assertIn("off-centre", text)
        # The long form (why no correction can be applied) is the tooltip, not the panel.
        self.assertIn("neither of them corrects the lens",
                      self.page.label_distortion.toolTip().lower())

    def test_the_hand_route_is_behind_a_rule_below_the_measured_one(self):
        """Two routes to one scale, and the gap has to be visible - not implied.

        The grid is the method (the whole field, automatic); the hand-clicked span is the
        fallback, so it sits below a rule, says what it is, and says why it is coarser.
        """
        layout = self.page.reference_group.layout()
        names = [layout.itemAt(index).widget().objectName()
                 for index in range(layout.count())
                 if layout.itemAt(index).widget() is not None]
        self.assertIn("manual_separator", names)
        self.assertLess(names.index("label_verdict"), names.index("manual_separator"))
        self.assertLess(names.index("manual_separator"), names.index("manual_hint_label"))
        # ... and the whole hand route (its Squares row is a nested form) comes after it.
        self.assertLess(names.index("manual_hint_label"), names.index("points_label"))
        self.assertIsInstance(self.page.manual_separator, QFrame)
        hint = self.page.manual_hint_label.text().lower()
        self.assertIn("fallback", hint)
        self.assertIn("coarser than the grid", hint)

    def test_the_header_reports_the_photo_and_the_state_not_the_old_span(self):
        """D4/D5: the header never repeats panel 1, and the span hint is panel 2's."""
        self.assertEqual([kpi.key for kpi in self.page.workspace.header],
                         ["photo", "state"])
        self.assertEqual(self.page.workspace.header[0].value, "none")
        self.assertEqual(self.page.workspace.header[1].value, "ready")

        self.page.use_photo(self._photo())
        self.assertIn("1200x900", self.page.workspace.header[0].value)
        self.assertEqual(self.page.workspace.header[1].value, "no lattice")

        self.page.measure_grid()
        self.assertEqual(self.page.workspace.header[1].value, "ready")
        self.page._save_profile()
        self.assertEqual(self.page.workspace.header[1].value, "stored")

    # -- panel 3: the scale - measured, or typed in ------------------------
    def test_nothing_measured_yet_says_so_instead_of_showing_a_number(self):
        self.assertIn("S = -", self.page.px_per_um_label.text())
        self.assertIn("scale uncertainty: -",
                      self.page.uncertainty_label.text().lower())

    def test_the_detected_scale_carries_its_source_and_its_consequence(self):
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        expected = self.page._grid.px_per_um() * 1000.0
        self.assertIn(f"{expected:.1f} px/mm", self.page.px_per_um_label.text())
        self.assertIn("detected grid", self.page.uncertainty_label.text())
        # The example comes from `daphnia.units`, the one formatter for "a length with its
        # uncertainty" - so it wears that house style, next to the panel's own ±.
        self.assertIn("+/-", self.page.example_label.text())
        self.assertIn("mm", self.page.example_label.text())

    def test_the_grid_route_names_the_print_it_includes(self):
        """`SR-OPTICS-018`: the print's own tolerance is *in* this number."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        text = self.page.uncertainty_label.text()
        self.assertIn(f"{self.page._grid.statistical_uncertainty_pct():.2f} %", text)
        self.assertIn(f"{PAPER_PITCH_TOLERANCE_PCT:.2f} %", text)
        self.assertIn("print", text)

    def test_the_grid_route_never_sends_the_operator_to_the_span(self):
        """D6: 2.54 % on the grid used to recommend a 0.29 % that omits half a percent."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.assertNotIn("span", self.page.uncertainty_label.text().lower())
        # ... and the advice it does give is about the field, not about a route.
        self.assertIn("field", self.page.uncertainty_label.toolTip().lower())

    def test_the_span_route_names_the_tolerance_it_leaves_out(self):
        """The stored 0.29 % row is this bug: the print's half percent is missing."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.page.canvas.set_points([(10.0, 10.0), (500.0, 10.0)])
        scale, _distance_px, uncertainty, source = self.page._pitch()
        self.assertEqual(source, SOURCE_SPAN)
        text = self.page.uncertainty_label.text()
        self.assertIn(f"± {uncertainty:.2f} %", text)
        self.assertIn("clicked span", text)
        self.assertIn("not included", text)

    def test_a_span_that_is_weak_is_told_to_click_a_longer_one(self):
        """`SR-OPTICS-015`: on *this* route the longer span really is the better move."""
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.page.canvas.set_points([(10.0, 10.0), (12.0, 10.0)])   # two pixels: noisy
        uncertainty = self.page._pitch()[2]
        self.assertGreater(uncertainty, settings.CALIBRATION_WEAK_UNCERTAINTY_PCT)
        self.assertIn("longer span", self.page.uncertainty_label.toolTip().lower())

    def test_a_typed_scale_is_never_presented_as_a_measurement(self):
        """`projectbrief.md` 2: honest precision - a claim must say it is a claim."""
        self.page.check_manual_scale.setChecked(True)
        self.page.spin_manual_px_per_um.setValue(0.1473)
        self.page.spin_manual_accuracy.setValue(2.0)

        scale, distance_px, uncertainty, source = self.page._pitch()
        self.assertEqual(source, MANUAL_SOURCE)
        self.assertAlmostEqual(scale, 0.1473)
        self.assertAlmostEqual(distance_px, scale)    # the proof row's own meaning
        self.assertAlmostEqual(uncertainty, 2.0)
        self.assertIn("147.3 px/mm", self.page.px_per_um_label.text())
        self.assertIn("hand-entered", self.page.px_per_um_label.text())
        self.assertIn("not a measurement", self.page.uncertainty_label.text())
        self.assertIn("you stated it", self.page.uncertainty_label.text())

    def test_the_typed_scale_wins_over_a_measured_one(self):
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.assertNotAlmostEqual(self.page._pitch()[0], 0.5, places=3)
        self.page.check_manual_scale.setChecked(True)
        self.page.spin_manual_px_per_um.setValue(0.5)
        self.assertEqual(self.page._pitch()[0], 0.5)

    def test_a_typed_scale_with_no_number_is_not_a_scale(self):
        self.page.check_manual_scale.setChecked(True)
        self.assertIsNone(self.page._pitch())
        self.assertIn("0 is not a scale", self.page.example_label.text())
        self.page._save_profile()
        self.assertIn("nothing to store", self.page.status_label.text().lower())

    def test_the_hand_entry_fields_are_live_only_while_the_box_is_ticked(self):
        self.assertFalse(self.page.spin_manual_px_per_um.isEnabled())
        self.assertFalse(self.page.spin_manual_accuracy.isEnabled())
        self.page.check_manual_scale.setChecked(True)
        self.assertTrue(self.page.spin_manual_px_per_um.isEnabled())
        self.assertTrue(self.page.spin_manual_accuracy.isEnabled())

    def test_a_typed_scale_is_saved_with_the_accuracy_that_was_stated(self):
        self.page.check_manual_scale.setChecked(True)
        self.page.spin_manual_px_per_um.setValue(0.2)
        self.page.spin_manual_accuracy.setValue(3.5)
        self.page._save_profile()

        with connection() as conn:
            profile = repository.active_profile(conn)
            row = conn.execute(
                "SELECT profile_name FROM calibration_profiles").fetchone()
            records = conn.execute(
                "SELECT * FROM calibration_records WHERE profile_id = ?",
                (profile.profile_id,)).fetchall()
        self.assertAlmostEqual(profile.px_per_um, 0.2)
        self.assertAlmostEqual(profile.uncertainty_pct, 3.5)
        self.assertEqual(profile.setup_key, self.page.setup_key())
        self.assertEqual(row["profile_name"], self.page.preset_name())
        self.assertEqual(records, [], "a typed scale has no proof photo to file")
        self.assertIn("your claim", self.page.status_label.text())

    # -- saving: against the preset of panel 1 -----------------------------
    def test_saving_stores_the_calibration_against_the_preset(self):
        self.page.profile_name.setText("scope 2x / iPhone")
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.page._save_profile()

        with connection() as conn:
            profile = repository.active_profile(conn)
            history = repository.calibrations_for_setup(conn, profile.setup_key)
            row = conn.execute(
                "SELECT profile_name FROM calibration_profiles").fetchone()
            record = conn.execute(
                "SELECT * FROM calibration_records WHERE profile_id = ?",
                (profile.profile_id,)).fetchone()
        self.assertEqual(profile.setup_key, self.page.setup_key())
        self.assertEqual(row["profile_name"], "scope 2x / iPhone")
        self.assertAlmostEqual(profile.px_per_um / (147.3 / 1000.0), 1.0, delta=0.02)
        self.assertEqual(len(history), 1)
        self.assertEqual(record["target_type"], TARGET_TYPE)
        self.assertIn("Stored for", self.page.status_label.text())
        self.assertTrue(list(paths.CALIBRATION_IMAGES_DIR.iterdir()))

    def test_a_finer_grid_is_filed_under_its_own_square_size(self):
        self.page.spin_grid_mm.setValue(0.5)
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.page._save_profile()
        with connection() as conn:
            record = conn.execute("SELECT * FROM calibration_records").fetchone()
        self.assertEqual(record["target_type"], "calibrated_grid_500um")
        self.assertAlmostEqual(record["actual_distance_um"], 500.0)
        self.assertIn("0.5 mm grid", record["notes"])

    def test_the_stored_uncertainty_includes_the_paper_tolerance(self):
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.page._save_profile()
        with connection() as conn:
            profile = repository.active_profile(conn)
        self.assertGreater(profile.uncertainty_pct, 0.4)
        self.assertLess(profile.uncertainty_pct, 1.0)

    def test_saving_twice_records_two_calibrations(self):
        """Re-verifying after a refocus adds a record instead of rewriting one."""
        for _ in range(2):
            self.page.use_photo(self._photo())
            self.page.measure_grid()
            self.page._save_profile()
        with connection() as conn:
            profile = repository.active_profile(conn)
            history = repository.calibrations_for_setup(conn, profile.setup_key)
        self.assertEqual(len(history), 2)

    def test_storing_without_a_measurement_says_what_is_missing(self):
        self.page._save_profile()
        self.assertIn("nothing to store", self.page.status_label.text().lower())

    # -- switching the device in force: Load... / New ----------------------
    def test_load_lists_each_device_once_by_name_and_nothing_else(self):
        """`SR-OPTICS-006`: the switcher is names, one per chain, newest first.

        A chain calibrated twice is **one** entry: the name is the identity, so the list
        must not grow a row per calibration (that is the history combo's job, and it
        carries the scale and route columns this one deliberately has not).
        """
        self._register(setup_key=new_setup_key("microscope"), name="microscope",
                       px_per_um=0.5, when="2026-10-01 09:00:00")
        self._register(setup_key=new_setup_key("microscope"), name="microscope",
                       px_per_um=0.51, when="2026-10-02 09:00:00")
        self._register(setup_key=new_setup_key("phone"), name="phone",
                       px_per_um=0.4, when="2026-10-19 09:00:00")

        names = self.page.stored_preset_names()
        self.assertEqual(sorted(names), ["microscope", "phone"])
        self.assertEqual(len(names), 2, "a chain calibrated twice is one entry")
        for name in names:
            self.assertNotIn("px/mm", name)             # a name, not a row of columns

    def test_load_makes_a_stored_device_the_one_in_force(self):
        """The whole point: switch the optics without re-measuring (`SR-OPTICS-006`)."""
        microscope = self._register(setup_key=new_setup_key("microscope"),
                                    name="microscope", px_per_um=0.5,
                                    when="2026-10-01 09:00:00")
        phone = self._register(setup_key=new_setup_key("phone"), name="phone",
                               px_per_um=0.4, when="2026-10-19 09:00:00")
        with connection() as conn:                       # the phone is in force now
            self.assertEqual(repository.active_profile(conn).profile_id, phone)

        self.assertTrue(self.page.load_preset("microscope"))

        with connection() as conn:
            self.assertEqual(repository.active_profile(conn).profile_id, microscope)
        self.assertEqual(self.page.profile_name.text(), "microscope")
        self.assertIn("Switched to 'microscope'", self.page.status_label.text())
        self.assertIn("500.0 px/mm", self.page.status_label.text())

    def test_load_takes_the_newest_calibration_of_that_chain(self):
        """A device's scale is its *latest*, not whichever row came first."""
        self._register(setup_key=new_setup_key("scope"), name="scope",
                       px_per_um=0.5, when="2026-10-01 09:00:00")
        new = self._register(setup_key=new_setup_key("scope"), name="scope",
                             px_per_um=0.52, when="2026-10-19 09:00:00")
        self.page.load_preset("scope")
        with connection() as conn:
            self.assertEqual(repository.active_profile(conn).profile_id, new)

    def test_load_stores_nothing(self):
        """Choosing a device is not an act of record: the row count is unchanged."""
        self._register(setup_key=new_setup_key("scope"), name="scope", px_per_um=0.5)
        with connection() as conn:
            before = len(repository.calibrations(conn))
        self.page.load_preset("scope")
        with connection() as conn:
            self.assertEqual(len(repository.calibrations(conn)), before)

    def test_load_with_a_name_nothing_was_stored_under_says_so(self):
        self._register(setup_key=new_setup_key("scope"), name="scope", px_per_um=0.5)
        self.assertFalse(self.page.load_preset("nothing here"))
        text = self.page.status_label.text()
        self.assertIn("nothing here", text)
        self.assertIn("Store this scale", text)

    def test_load_changes_nothing_when_it_refuses(self):
        """A name with no scale leaves the one in force alone - and names no device."""
        scope = self._register(setup_key=new_setup_key("scope"), name="scope",
                               px_per_um=0.5)
        with connection() as conn:
            before = repository.active_profile(conn).profile_id
        self.assertFalse(self.page.load_preset("nobody"))
        with connection() as conn:
            self.assertEqual(repository.active_profile(conn).profile_id, before)
        self.assertEqual(scope, before)

    def test_new_clears_the_name_and_the_measurement_without_switching(self):
        """`New` starts a chain; it is *Load...* that switches (`SR-OPTICS-031`)."""
        scope = self._register(setup_key=new_setup_key("scope"), name="scope",
                               px_per_um=0.5)
        self.page.profile_name.setText("scope")
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.assertIsNotNone(self.page._grid)

        self.page.new_preset("")

        self.assertEqual(self.page.profile_name.text(), "")
        self.assertIsNone(self.page._grid)
        self.assertIsNone(self.page.canvas.lattice)
        self.assertIsNone(self.page.canvas.field_stop)
        with connection() as conn:                       # nothing switched
            self.assertEqual(repository.active_profile(conn).profile_id, scope)
        self.assertIn("New preset", self.page.status_label.text())

    def test_new_can_name_the_fresh_preset_in_the_popup(self):
        """The *New* popup supplies the name, so the chain exists the moment it is typed."""
        self._register(setup_key=new_setup_key("scope"), name="scope", px_per_um=0.5)
        self.page.new_preset("phone")
        self.assertEqual(self.page.profile_name.text(), "phone")
        self.assertEqual(self.page.setup_key(), new_setup_key("phone"))
        self.assertIn("phone", self.page.status_label.text())

    def test_the_panel_offers_load_and_new_at_its_head(self):
        """Two switches at the top of panel 1, next to the name they act on."""
        self.assertEqual(self.page.btn_load_preset.text(), "Load preset...")
        self.assertEqual(self.page.btn_new_preset.text(), "New preset...")
        for button in (self.page.btn_load_preset, self.page.btn_new_preset):
            self.assertTrue(button.isEnabled(),
                            "a choice that is live, never a greyed-out swallow")


    # -- panel 1 again: the proof photos of earlier calibrations -----------
    def _register(self, *, setup_key="PROF_OLD", px_per_um=0.5, proof=True,
                  when=None, source="detected grid", name="scope") -> str:
        """One stored calibration - with a real proof photo on disk unless told not to.

        ``when`` backdates the record, so "newest first" can be asserted with two
        calibrations that were otherwise saved in the same second. ``source`` is written
        into the proof row's notes exactly as :meth:`_save_profile` writes it, because the
        history reads an entry's route back out of that field. ``name`` is the preset the
        row carries; the switcher tests pass it so the name and the ``setup_key`` agree
        the way :meth:`Calibration._save_profile` keeps them.
        """
        image = Path(self._tmp.name) / f"proof_{px_per_um}.png"
        frame = grid_image(height=60, width=80, period=20.0, noise=0.0)
        cv2.imwrite(str(image), (frame * 255).astype("uint8"))
        with connection() as conn:
            profile_id = repository.register_calibration(
                conn, setup_key=setup_key, profile_name=name,
                microscope_model="m", objective_magnification="2x",
                camera_device="c", px_per_um=px_per_um, uncertainty_pct=1.0)
            record_id = None
            if proof:
                record_id = repository.insert_calibration_record(
                    conn, profile_id=profile_id, target_type=TARGET_TYPE,
                    raw_image_path=image, detected_distance_px=px_per_um * 1000.0,
                    actual_distance_um=GRID_PITCH_UM,
                    notes=f"{grid_label(GRID_PITCH_UM)}, measured from the {source}")
            if when:
                conn.execute("UPDATE calibration_profiles SET last_verified_at = ?"
                             " WHERE profile_id = ?", (when, profile_id))
                if record_id:
                    conn.execute("UPDATE calibration_records SET created_at = ?"
                                 " WHERE record_id = ?", (when, record_id))
        return profile_id

    # -- the calibration pair: Load a stored one, or start a New one --------
    def test_the_calibration_list_is_this_devices_own_newest_first(self):
        """`SR-OPTICS-006`: a preset has many calibrations; the list is *its* history."""
        self.page.profile_name.setText("scope")
        self._register(setup_key=new_setup_key("other"), name="other", px_per_um=0.4)
        older = self._register(setup_key=self.page.setup_key(), name="scope",
                               px_per_um=0.5, when="2026-10-01 09:00:00")
        newest = self._register(setup_key=self.page.setup_key(), name="scope",
                                px_per_um=0.52, when="2026-10-19 09:00:00")

        self.page._refresh_calibrations()
        entries = self.page.stored_calibrations()
        self.assertEqual([entry["profile_id"] for entry in entries], [newest, older])
        self.assertIn("520.0 px/mm", entries[0]["label"])
        self.assertIn("(in force)", entries[0]["label"])   # the last stored is active
        self.assertNotIn("(in force)", entries[1]["label"])
        self.assertIn("2 stored", self.page.history_label.text())

    def test_the_calibration_list_leaves_other_devices_out(self):
        """The list is per device: a renamed preset shows another device's rows, not these."""
        self._register(setup_key=new_setup_key("before"), name="before", px_per_um=0.5)
        self.page.profile_name.setText("after")
        self.page._refresh_all()
        self.assertEqual(self.page.stored_calibrations(), [])
        self.assertIn("none stored", self.page.history_label.text())

    def test_loading_a_calibration_puts_it_in_force_and_shows_its_photo(self):
        """Loading is a switch *and* a look: the chosen row becomes the scale in force."""
        self.page.profile_name.setText("scope")
        older = self._register(setup_key=self.page.setup_key(), name="scope",
                               px_per_um=0.5, when="2026-10-01 09:00:00")
        newest = self._register(setup_key=self.page.setup_key(), name="scope",
                                px_per_um=0.52, when="2026-10-19 09:00:00")
        with connection() as conn:
            self.assertEqual(repository.active_profile(conn).profile_id, newest)

        self.assertTrue(self.page.load_calibration(older))

        with connection() as conn:
            self.assertEqual(repository.active_profile(conn).profile_id, older)
        self.assertIsNotNone(self.page._frame)            # the proof photo
        self.assertIn("Loaded", self.page.status_label.text())
        self.assertIn("500.0 px/mm", self.page.status_label.text())
        self.assertIn("in force", self.page.status_label.text())

    def test_loading_a_calibration_stores_nothing(self):
        self.page.profile_name.setText("scope")
        self._register(setup_key=self.page.setup_key(), name="scope", px_per_um=0.5)
        with connection() as conn:
            before = len(repository.calibrations(conn))
        entry = self.page.stored_calibrations()[0]
        self.page.load_calibration(entry["profile_id"])
        with connection() as conn:
            self.assertEqual(len(repository.calibrations(conn)), before)

    def test_a_hand_entered_calibration_loads_but_shows_no_photo(self):
        """`SR-OPTICS-029`: a typed scale has no evidence - but it can still be loaded."""
        self.page.profile_name.setText("scope")
        hand = self._register(setup_key=self.page.setup_key(), name="scope",
                              px_per_um=0.5, proof=False)
        self.assertTrue(self.page.load_calibration(hand))
        with connection() as conn:
            self.assertEqual(repository.active_profile(conn).profile_id, hand)
        self.assertIsNone(self.page._frame)
        self.assertIn("no proof photo", self.page.status_label.text())
        self.assertIn("entered by hand", self.page.status_label.text())

    def test_a_proof_photo_that_has_gone_is_named_rather_than_skipped(self):
        self.page.profile_name.setText("scope")
        gone_id = self._register(setup_key=self.page.setup_key(), name="scope",
                                 px_per_um=0.5)
        gone = Path(self._tmp.name) / "gone.png"
        with connection() as conn:
            conn.execute("UPDATE calibration_records SET raw_image_path = ?",
                         (str(gone),))
        self.assertTrue(self.page.load_calibration(gone_id))
        self.assertIsNone(self.page._frame)
        self.assertIn("gone.png", self.page.status_label.text())
        with connection() as conn:                       # still loaded, photo missing
            self.assertEqual(repository.active_profile(conn).profile_id, gone_id)

    def test_loading_an_unknown_calibration_says_so(self):
        self.assertFalse(self.page.load_calibration("PROF_NOPE"))
        self.assertIn("no calibration PROF_NOPE", self.page.status_label.text())

    def test_new_calibration_clears_the_measurement_but_keeps_the_device(self):
        """`New` for a scale keeps the preset; it is *New* for a preset that clears it."""
        self.page.profile_name.setText("scope")
        scope = self._register(setup_key=self.page.setup_key(), name="scope",
                               px_per_um=0.5)
        self.page.use_photo(self._photo())
        self.page.measure_grid()
        self.assertIsNotNone(self.page._grid)

        self.page.new_calibration("")

        self.assertEqual(self.page.profile_name.text(), "scope")   # the device stays
        self.assertIsNone(self.page._grid)
        self.assertIsNone(self.page.canvas.lattice)
        with connection() as conn:                                 # nothing switched
            self.assertEqual(repository.active_profile(conn).profile_id, scope)
        self.assertIn("New calibration for 'scope'", self.page.status_label.text())

    def test_the_new_calibration_label_is_carried_to_the_stored_row(self):
        """`New calibration...` asks for a note, and the store writes it on the new row."""
        self.page.profile_name.setText("scope")
        self.page.new_calibration("after refocus")
        self.page.check_manual_scale.setChecked(True)
        self.page.spin_manual_px_per_um.setValue(0.5)        # a scale we know by hand
        self.page._save_profile()
        with connection() as conn:
            row = conn.execute(
                "SELECT label FROM calibration_profiles"
                " ORDER BY rowid DESC").fetchone()
        self.assertEqual(row["label"], "after refocus")
        # The list shows the operator's note, not only a timestamp.
        self.assertIn("after refocus", self.page.history_label.text())

    def test_new_preset_drops_a_pending_calibration_label(self):
        """Load/New must not carry one device's note onto another device's calibration.

        `New calibration...` arms a label for the *next* store; naming a *different* device
        afterwards must forget it, or the note lands on the wrong chain's row.
        """
        self.page.new_calibration("after refocus")     # a note for the next store ...
        self.page.new_preset("phone")                  # ... but this is a different device
        self.assertIsNone(self.page._pending_label)
        self.page.check_manual_scale.setChecked(True)
        self.page.spin_manual_px_per_um.setValue(0.4)
        self.page._save_profile()
        with connection() as conn:
            row = conn.execute("SELECT label FROM calibration_profiles").fetchone()
        self.assertIsNone(row["label"])

    def test_switching_device_drops_a_pending_calibration_label(self):
        self._register(setup_key=new_setup_key("scope"), name="scope", px_per_um=0.5)
        self.page.new_calibration("after refocus")
        self.assertTrue(self.page.load_preset("scope"))
        self.assertIsNone(self.page._pending_label)

    def test_loading_a_calibration_drops_a_pending_calibration_label(self):
        self.page.profile_name.setText("scope")
        stored = self._register(setup_key=self.page.setup_key(), name="scope",
                                px_per_um=0.5)
        self.page.new_calibration("after refocus")
        self.assertTrue(self.page.load_calibration(stored))
        self.assertIsNone(self.page._pending_label)

    def test_the_panel_offers_load_and_new_for_the_calibration_too(self):
        """The same pair of buttons the preset has, for the scale."""
        self.assertEqual(self.page.btn_load_calibration.text(), "Load calibration...")
        self.assertEqual(self.page.btn_new_calibration.text(), "New calibration...")
        for button in (self.page.btn_load_calibration, self.page.btn_new_calibration):
            self.assertTrue(button.isEnabled())
        self.assertFalse(hasattr(self.page, "combo_history"))


if __name__ == "__main__":
    unittest.main()
