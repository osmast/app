"""Auto-calibration from 1 mm grid paper.

The detector is only useful if it is *hard to fool*: a wrong scale is worse than no
scale, so the refusals matter as much as the detections.
"""

import unittest

import numpy as np

from daphnia import units
from daphnia.vision.grid import (
    GRID_PITCH_UM,
    GridNotFound,
    detect_grid,
    grid_scale_px_per_um,
)

SIZE = (600, 800)


def grid_image(height=SIZE[0], width=SIZE[1], period=40.0, period_vertical=None,
               angle_deg=0.0, line_ratio=0.18, noise=0.0, blur=0, seed=0):
    """A synthetic photo of 1 mm grid paper: dark lines on a bright sheet."""
    period_vertical = period if period_vertical is None else period_vertical
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    theta = np.radians(angle_deg)
    u = xx * np.cos(theta) + yy * np.sin(theta)
    v = -xx * np.sin(theta) + yy * np.cos(theta)

    def lines(coord, pitch):
        return (np.mod(coord, pitch) < pitch * line_ratio).astype(float)

    image = 1.0 - 0.6 * np.maximum(lines(u, period), lines(v, period_vertical))
    if blur:
        total = np.zeros_like(image)
        for dy in range(-blur, blur + 1):
            for dx in range(-blur, blur + 1):
                total += np.roll(np.roll(image, dy, axis=0), dx, axis=1)
        image = total / ((2 * blur + 1) ** 2)
    if noise:
        image = image + np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return np.clip(image, 0.0, 1.0)


def specimen_image(count=30, seed=7):
    """A plate-like image: many small dark ellipses on a bright well."""
    rng = np.random.default_rng(seed)
    image = np.full(SIZE, 0.9)
    yy, xx = np.mgrid[0:SIZE[0], 0:SIZE[1]]
    for _ in range(count):
        cy, cx = rng.integers(0, SIZE[0]), rng.integers(0, SIZE[1])
        ry, rx = rng.integers(6, 18), rng.integers(10, 26)
        angle = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        v = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        image[(u / rx) ** 2 + (v / ry) ** 2 < 1.0] = 0.3
    return np.clip(image + rng.normal(0, 0.03, SIZE), 0, 1)


class DetectionTests(unittest.TestCase):
    """A square grid must be measured accurately, whatever it looks like."""

    def _period(self, frame, expected):
        estimate = detect_grid(frame)
        self.assertAlmostEqual(estimate.period_px / expected, 1.0, delta=0.01)
        return estimate

    def test_an_axis_aligned_grid_is_measured(self):
        self._period(grid_image(period=40.0), 40.0)

    def test_a_rotated_grid_is_measured(self):
        """The paper does not have to be aligned with the frame."""
        estimate = self._period(grid_image(period=40.0, angle_deg=8.0), 40.0)
        self.assertAlmostEqual(estimate.angle_deg, 8.0, delta=1.5)

    def test_a_low_magnification_grid_is_measured(self):
        self._period(grid_image(period=147.3), 147.3)

    def test_a_high_magnification_grid_is_measured(self):
        self._period(grid_image(period=14.7), 14.7)

    def test_a_blurred_grid_is_still_measured(self):
        """Out of focus changes the apparent period; it must still be measurable."""
        self._period(grid_image(period=40.0, blur=3), 40.0)

    def test_a_full_size_photo_keeps_the_scale_of_that_photo(self):
        """A multi-megapixel photo is block-averaged before the FFT.

        The period must be reported in the *original* pixels, not in the reduced
        ones - getting that wrong would scale every measurement by the reduction
        factor - and the reduction has to stay isotropic, or a square grid would
        look rectangular and be refused.
        """
        estimate = detect_grid(grid_image(height=1800, width=2400, period=150.0,
                                         noise=0.02))
        self.assertAlmostEqual(estimate.period_px / 150.0, 1.0, delta=0.01)
        self.assertLess(estimate.anisotropy_pct, 2.0)

    def test_a_noisy_grid_is_still_measured(self):
        self._period(grid_image(period=40.0, noise=0.08), 40.0)

    def test_a_colour_frame_is_accepted(self):
        grey = grid_image(period=40.0)
        colour = np.dstack([grey, grey, grey])
        self._period(colour, 40.0)

    def test_the_second_family_is_reported(self):
        estimate = detect_grid(grid_image(period=40.0))
        self.assertAlmostEqual(
            estimate.period_px_vertical / estimate.period_px, 1.0, delta=0.02)


class RefusalTests(unittest.TestCase):
    """Failing loudly is the feature: a wrong scale is worse than none."""

    def _refused(self, frame):
        with self.assertRaises(GridNotFound):
            detect_grid(frame)

    def test_a_flat_field_is_refused(self):
        rng = np.random.default_rng(1)
        self._refused(np.clip(np.full(SIZE, 0.5) + rng.normal(0, 0.02, SIZE), 0, 1))

    def test_a_specimen_plate_is_refused(self):
        """The image this app spends its life looking at must not fake a scale."""
        self._refused(specimen_image())

    def test_random_blobs_are_refused(self):
        rng = np.random.default_rng(3)
        image = np.full(SIZE, 0.85)
        yy, xx = np.mgrid[0:SIZE[0], 0:SIZE[1]]
        for _ in range(40):
            cy, cx = rng.integers(0, SIZE[0]), rng.integers(0, SIZE[1])
            radius = int(rng.integers(6, 12))
            image[(yy - cy) ** 2 + (xx - cx) ** 2 < radius * radius] = 0.25
        self._refused(np.clip(image + rng.normal(0, 0.03, SIZE), 0, 1))

    def test_a_single_stripe_family_is_refused(self):
        """One direction of lines is not a square grid."""
        self._refused(1.0 - 0.5 * (np.mod(np.mgrid[0:SIZE[0], 0:SIZE[1]][1], 55) < 9))

    def test_a_non_square_target_is_refused(self):
        """Rectangular spacing means the paper is tilted, so the scale is ambiguous."""
        self._refused(grid_image(period=40.0, period_vertical=28.0))

    def test_a_tiny_image_is_refused(self):
        self._refused(grid_image(height=16, width=16, period=6.0))

    def test_a_one_dimensional_array_is_a_value_error(self):
        with self.assertRaises(ValueError):
            detect_grid(np.zeros(64))


class ScaleTests(unittest.TestCase):
    """What the detected period means in micrometres, and how sure we are."""

    def _estimate(self, period=40.0, periods=15.0):
        from daphnia.vision.grid import GridEstimate
        return GridEstimate(period_px=period, period_px_vertical=period,
                            angle_deg=0.0, periods=periods, confidence=1.0)

    def test_one_millimetre_square_becomes_the_scale(self):
        self.assertAlmostEqual(self._estimate(period=40.0).px_per_um(), 0.04)

    def test_the_convenience_scale_matches_the_estimate(self):
        frame = grid_image(period=40.0)
        self.assertAlmostEqual(grid_scale_px_per_um(frame),
                               detect_grid(frame).px_per_um())

    def test_the_pitch_defaults_to_one_millimetre(self):
        self.assertEqual(GRID_PITCH_UM, 1000.0)
        self.assertEqual(self._estimate().pitch_um, 1000.0)

    def test_the_fit_error_shrinks_with_more_squares(self):
        """The automated form of 'measure a longer span'."""
        few = self._estimate(periods=4.0).statistical_uncertainty_pct()
        many = self._estimate(periods=40.0).statistical_uncertainty_pct()
        self.assertAlmostEqual(few / 10.0, many, places=6)

    def test_the_paper_tolerance_dominates_the_fit_error(self):
        """Honest engineering: the print, not the detector, is the accuracy floor."""
        estimate = self._estimate()
        self.assertGreater(estimate.uncertainty_pct(),
                           5.0 * estimate.statistical_uncertainty_pct())

    def test_the_uncertainty_combines_fit_and_paper(self):
        estimate = self._estimate()
        self.assertAlmostEqual(estimate.statistical_uncertainty_pct(),
                               0.0707, delta=0.001)
        self.assertAlmostEqual(estimate.uncertainty_pct(), 0.505, delta=0.005)

    def test_a_verified_print_can_be_declared_better(self):
        """A grid checked against a stage micrometer earns a smaller tolerance."""
        estimate = self._estimate()
        self.assertLess(estimate.uncertainty_pct(paper_tolerance_pct=0.1),
                        estimate.uncertainty_pct(paper_tolerance_pct=0.5))

    def test_the_result_reads_as_a_measurable_length(self):
        estimate = self._estimate()
        text = units.format_with_uncertainty(94.07, estimate.uncertainty_pct())
        self.assertIn("+/-", text)
        self.assertIn("mm", text)


class WarningTests(unittest.TestCase):
    """What an operator should be told, in the words the app will use."""

    def _notes(self, frame):
        return detect_grid(frame).warnings()

    def test_a_good_grid_raises_no_alarm(self):
        self.assertEqual(self._notes(grid_image(period=40.0, noise=0.05)), ())

    def test_a_tilted_target_is_reported(self):
        """Square paper photographed obliquely is no longer square in the frame."""
        notes = self._notes(grid_image(period=40.0, period_vertical=36.0))
        self.assertTrue(any("tilted" in note for note in notes), notes)

    def test_too_few_squares_is_reported(self):
        notes = self._notes(grid_image(period=147.3))
        self.assertTrue(any("squares fit" in note for note in notes), notes)

    def test_a_long_period_is_never_called_distorted(self):
        """Centre-vs-frame comparison needs enough squares to mean anything."""
        estimate = detect_grid(grid_image(period=147.3))
        self.assertLess(estimate.periods, 6)
        self.assertEqual(estimate.distortion_pct, 0.0)


if __name__ == "__main__":
    unittest.main()

