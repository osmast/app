"""The scale from a printed **dot** target photographed through an eyepiece.

`vision/lattice.py` measures grid *paper*, whose scale is the spacing of its printed
**lines**; `vision/points.py` measures a printed **dot** target, whose scale is the spacing
of its printed **dots**. A printed dot grid is the target a distortion model wants
(`docs/calibration.md`: 0.149 px rms on a known distortion) and the one `assets/targets/`
renders, so the OPTICS surface needs both estimators.

Two halves matter, and both are here: a dot lattice that **is** there is measured to well
under a percent (the *oracle* test - a perfect lattice must score 1.0, `mlContext.md` law
2), and a photo that is not a dot grid is **refused** - a wrong scale is worse than no
scale, exactly as `vision/grid.py` and `vision/lattice.py` have it. The warped, vignetted
render from `scripts/render_point_target.py` is measured too, so "the detector answers the
artificial ocular photo" is a test rather than a claim.
"""

import unittest

import numpy as np

from daphnia.vision.lattice import measure as measure_lattice
from daphnia.vision.points import PointsNotFound, measure_points, reference_points
from scripts.render_point_target import ocular_point_grid
from tests.test_lattice import grid_image as line_grid_image

SIZE = (600, 800)


def dot_grid(columns=20, rows=15, period=40.0, angle_deg=0.0, dot_ratio=0.4,
             contrast=0.75, noise=0.0, seed=0, height=SIZE[0], width=SIZE[1]):
    """A clean square lattice of round **dots** on a bright sheet, as an image.

    The dots sit at the lattice's nodes - not along its lines - so this is exactly what a
    2D FFT (or a line-band reader) cannot measure and a dot detector must. ``dot_ratio`` is
    a dot's diameter as a fraction of the pitch.
    """
    row_index, column_index = np.mgrid[0:height, 0:width].astype(float)
    theta = np.radians(angle_deg)
    centre_x, centre_y = width / 2.0, height / 2.0
    a = (column_index - centre_x) * np.cos(theta) + (row_index - centre_y) * np.sin(theta)
    b = -(column_index - centre_x) * np.sin(theta) + (row_index - centre_y) * np.cos(theta)
    du = np.abs((a + period / 2.0) % period - period / 2.0)
    dv = np.abs((b + period / 2.0) % period - period / 2.0)
    ink = np.hypot(du, dv) <= 0.5 * dot_ratio * period
    image = 1.0 - contrast * ink.astype(float)
    if noise:
        image = image + np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return np.clip(image, 0.0, 1.0)


class MeasurementTests(unittest.TestCase):
    """A lattice that is there must be measured, and to well under a percent."""

    def test_the_oracle_a_perfect_lattice_scores_exactly(self):
        """Law 2: a perfect target must come back perfect, or the metric is broken."""
        for period in (24.0, 40.0, 60.0):
            estimate = measure_points(dot_grid(period=period))
            self.assertAlmostEqual(estimate.period_px / period, 1.0, delta=0.01)

    def test_the_second_family_is_reported(self):
        estimate = measure_points(dot_grid(period=40.0))
        self.assertAlmostEqual(
            estimate.period_px_vertical / estimate.period_px, 1.0, delta=0.02)

    def test_the_tilt_is_reported_and_does_not_change_the_scale(self):
        straight = measure_points(dot_grid(period=40.0, angle_deg=0.0))
        tilted = measure_points(dot_grid(period=40.0, angle_deg=7.0))
        self.assertAlmostEqual(straight.period_px / tilted.period_px, 1.0, delta=0.02)
        self.assertGreater(abs(tilted.angle_deg), 0.0)

    def test_a_mirror_does_not_change_the_scale(self):
        """A symmetry the estimate must satisfy: no second photo, no fit involved."""
        frame = dot_grid(period=40.0, angle_deg=6.0)
        straight = measure_points(frame)
        mirrored = measure_points(np.ascontiguousarray(frame[:, ::-1]))
        self.assertAlmostEqual(straight.period_px / mirrored.period_px, 1.0, delta=0.02)

    def test_the_scale_and_its_uncertainty_come_from_the_pitch(self):
        estimate = measure_points(dot_grid(period=40.0), pitch_um=1000.0)
        self.assertAlmostEqual(estimate.px_per_um(), 0.04)
        self.assertAlmostEqual(estimate.px_per_um(pitch_um=500.0), 0.08)
        self.assertGreater(estimate.uncertainty_pct(), 0.4)
        self.assertLess(estimate.uncertainty_pct(), 1.0)

    def test_the_panel_readings_are_named_as_dots(self):
        """The Calibrate panel names the right unit whichever estimator answered."""
        estimate = measure_points(dot_grid(period=40.0))
        self.assertEqual(estimate.reading_label, "dots")
        self.assertGreater(estimate.reading_count, 0)

    def test_a_manual_binarisation_level_is_used_and_reported(self):
        """Discorpy's escape hatch: the operator may set the cut, and the result says so.

        A threshold that splits the same two populations must give the same scale, and the
        estimate must **carry** the value - a number nobody chose must not pass as Otsu's.
        """
        frame = dot_grid(period=40.0)
        automatic = measure_points(frame)
        self.assertIsNone(automatic.threshold)
        manual = measure_points(frame, threshold=0.5)
        self.assertAlmostEqual(manual.period_px / automatic.period_px, 1.0, delta=0.02)
        self.assertEqual(manual.threshold, 0.5)
        self.assertTrue(any("set by hand" in note for note in manual.warnings()))

    def test_a_threshold_that_cuts_nothing_is_refused(self):
        """A cut below every printed value finds no ink, and says so - it does not guess."""
        with self.assertRaises(PointsNotFound):
            measure_points(dot_grid(period=40.0), threshold=0.0)

    def test_reference_points_are_the_points_that_were_measured(self):
        """The drawing and the measurement must come from one extraction, not two (law 1)."""
        frame = dot_grid(period=40.0)
        points = reference_points(frame)
        estimate = measure_points(frame)
        self.assertGreater(len(points), 0)
        self.assertEqual(len(points), estimate.reading_count)
        self.assertEqual(points.shape[1], 2)

    def test_reference_points_of_a_checkerboard_are_its_cells(self):
        image, _truth = ocular_point_grid(pattern="checkerboard", seed=0)
        self.assertGreater(len(reference_points(image, pattern="checkerboard")), 0)

    def test_reference_points_refuses_an_unknown_pattern(self):
        with self.assertRaises(ValueError):
            reference_points(dot_grid(period=40.0), pattern="nonsense")


class OcularRenderTests(unittest.TestCase):
    """The synthetic ocular photo - a warped dot grid inside a round vignette."""

    def setUp(self):
        self.image, self.truth = ocular_point_grid(distortion=1.0e-7, seed=0)

    def test_the_photo_has_a_field_stop_a_fitter_can_find(self):
        estimate = measure_points(self.image)
        self.assertIsNotNone(estimate.field_stop)
        radius = estimate.field_stop.radius
        self.assertGreater(radius, 0.9 * self.truth["field_stop_radius_px"])
        self.assertLess(radius, 1.1 * self.truth["field_stop_radius_px"])

    def test_the_warped_lattice_is_measured_inside_the_known_bracket(self):
        estimate = measure_points(self.image)
        centre = self.truth["pitch_px"]
        edge = self.truth["predicted_edge_pitch_px"]
        low, high = min(centre, edge), max(centre, edge)
        tolerance = 0.01 * centre
        self.assertGreaterEqual(estimate.period_px, low - tolerance)
        self.assertLessEqual(estimate.period_px, high + tolerance)

    def test_the_detector_sees_the_drift_the_render_was_given(self):
        """The warp is visible as a positive spread, not rounded away to 0 %."""
        estimate = measure_points(self.image)
        self.assertGreater(estimate.spread_pct, 0.5)
        self.assertLess(estimate.spread_pct, 3.0)

    def test_a_perfect_render_is_measured_exactly(self):
        image, truth = ocular_point_grid(distortion=0.0, seed=0)
        estimate = measure_points(image)
        self.assertAlmostEqual(estimate.period_px / truth["pitch_px"], 1.0, delta=0.01)

    def test_the_ground_truth_sidecar_is_complete(self):
        """A measurement is only reportable with its provenance (law 5)."""
        for key in ("pitch_px", "pitch_um", "distortion_k1", "predicted_edge_pitch_px",
                    "field_stop_radius_px", "tilt_deg"):
            self.assertIn(key, self.truth)


class RefusalTests(unittest.TestCase):
    """Failing loudly is the feature: a wrong scale is worse than none."""

    def _refused(self, frame):
        with self.assertRaises(PointsNotFound):
            measure_points(frame)

    def test_a_flat_field_is_refused(self):
        rng = np.random.default_rng(1)
        self._refused(np.clip(np.full(SIZE, 0.5) + rng.normal(0, 0.02, SIZE), 0, 1))

    def test_a_single_dot_is_refused(self):
        """A lone mark has no lattice: too few dots to measure anything."""
        frame = np.full(SIZE, 0.95)
        rows, columns = np.mgrid[0:SIZE[0], 0:SIZE[1]]
        frame[(columns - 400) ** 2 + (rows - 300) ** 2 < 12 ** 2] = 0.2
        self._refused(frame)

    def test_a_line_grid_is_refused_by_the_dot_detector(self):
        """The two detectors are genuinely different: lines are not dots."""
        self._refused(line_grid_image(period=40.0))

    def test_a_dot_target_is_refused_by_the_checkerboard_detector(self):
        """Round dots are not square cells: refused, rather than read as cells of dot size."""
        with self.assertRaises(PointsNotFound):
            measure_points(dot_grid(period=40.0), pattern="checkerboard")

    def test_the_two_estimators_agree_on_a_dot_lattice(self):
        """A dot lattice's period *is* its dots' spacing, so both detectors must agree.

        The line estimator reads each row of dots as a periodic profile (the dots are its
        "lines"); the dot estimator reads the nodes. On a clean target the two independent
        routes must land on the same number, or one of them is measuring something else.
        """
        frame = dot_grid(period=40.0)
        by_lines = measure_lattice(frame).period_px
        by_dots = measure_points(frame).period_px
        self.assertAlmostEqual(by_lines / by_dots, 1.0, delta=0.02)

    def test_a_one_dimensional_array_is_a_value_error(self):
        with self.assertRaises(ValueError):
            measure_points(np.zeros(64))


if __name__ == "__main__":
    unittest.main()
