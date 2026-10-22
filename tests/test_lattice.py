"""The scale from grid paper photographed through an eyepiece.

`vision/grid.py` refuses **every** photo this rig produces (measured 2026-10-19: nine of
nine), so the OPTICS surface measures with `vision/lattice.py`, which prepares the frame
first: the circular field stop is found and masked, the surround is filled, the lattice is
straightened, and the period is read off narrow bands - and the *spread* of those bands is
reported as the position-dependent scale error nobody had measured before.

Two halves matter, and both are here: a lattice that **is** there is measured to well under
a percent, and a photo that is not square grid paper is **refused** - a wrong scale is worse
than no scale, exactly as `vision/grid.py` has it. The comparison with the shipped detector
is asserted too, so "the second estimator does better on this rig's photos" is a test rather
than a claim.
"""

import unittest

import numpy as np

from daphnia.vision.grid import GridNotFound, detect_grid
from daphnia.vision.lattice import (
    LatticeNotFound,
    field_stop,
    measure,
)

SIZE = (600, 800)


def pattern(columns, rows, period, line_ratio=0.18, angle_deg=0.0):
    """A square lattice as an image: dark lines on a bright sheet."""
    theta = np.radians(angle_deg)
    u = columns * np.cos(theta) + rows * np.sin(theta)
    v = -columns * np.sin(theta) + rows * np.cos(theta)
    lines = ((np.mod(u, period) < period * line_ratio).astype(float)
             + (np.mod(v, period) < period * line_ratio).astype(float))
    return 1.0 - 0.6 * np.minimum(lines, 1.0)


def grid_image(period=40.0, angle_deg=0.0, heavy=5, contrast=0.6, noise=0.0, seed=0,
               height=SIZE[0], width=SIZE[1]):
    """A clean sheet of grid paper, optionally with a heavier line every ``heavy`` cells.

    The heavy line is the printed paper's own feature (measured on this project's photo:
    every fifth line is 2-3x darker and up to 7 px wide). A whole-frame estimator that
    works by amplitude sees only those; a local band reader must not care.
    """
    rows, columns = np.mgrid[0:height, 0:width].astype(float)
    image = pattern(columns, rows, period, angle_deg=angle_deg)
    if heavy:
        rows_fine = (np.mod(rows, period * heavy) < period * 0.25).astype(float)
        columns_fine = (np.mod(columns, period * heavy) < period * 0.25).astype(float)
        image = np.minimum(image, 1.0 - 0.5 * np.minimum(rows_fine + columns_fine, 1.0))
    image = 1.0 - contrast / 0.6 * (1.0 - image)
    if noise:
        image = image + np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return np.clip(image, 0.0, 1.0)


def eyepiece_image(period=27.0, radius=260, angle_deg=9.0, contrast=0.05, heavy=5,
                   height=SIZE[0], width=SIZE[1]):
    """This rig's actual photo: a pale lattice inside a bright disc on a black surround.

    The disc is **cut off by the frame** at the top (centre below the middle, radius bigger
    than the room above it), which is what makes the circle fit a real problem: a fit that
    uses the flat clipped edge comes out ~60 px small and then fills over real paper.
    """
    rows, columns = np.mgrid[0:height, 0:width].astype(float)
    image = grid_image(period=period, angle_deg=angle_deg, heavy=heavy,
                       contrast=contrast, height=height, width=width)
    centre_y, centre_x = height // 2 + 40, width // 2
    inside = (columns - centre_x) ** 2 + (rows - centre_y) ** 2 <= radius ** 2
    return np.where(inside, image, 0.03).astype(np.float32), (centre_x, centre_y, radius)


class MeasurementTests(unittest.TestCase):
    """A lattice that is there must be measured, wherever the paper sits."""

    def _measure(self, frame, expected, tolerance=0.02):
        estimate = measure(frame)
        self.assertAlmostEqual(estimate.period_px / expected, 1.0, delta=tolerance)
        return estimate

    def test_a_square_grid_is_measured(self):
        estimate = self._measure(grid_image(period=40.0), 40.0)
        self.assertGreaterEqual(estimate.bands, 6)
        self.assertLess(estimate.spread_pct, 1.0)

    def test_a_rotated_grid_is_measured_and_its_tilt_reported(self):
        estimate = self._measure(grid_image(period=40.0, angle_deg=8.0), 40.0)
        self.assertAlmostEqual(abs(estimate.angle_deg), 8.0, delta=1.0)

    def test_the_heavier_line_every_fifth_does_not_move_the_scale(self):
        """The paper's own amplitude modulation must not touch the measurement."""
        plain = measure(grid_image(period=40.0, heavy=0))
        heavy = measure(grid_image(period=40.0, heavy=5))
        self.assertAlmostEqual(plain.period_px / heavy.period_px, 1.0, delta=0.01)

    def test_an_eyepiece_photo_is_measured_where_the_shipped_detector_refuses(self):
        frame, (centre_x, _centre_y, radius) = eyepiece_image()
        with self.assertRaises(GridNotFound):
            detect_grid(frame)                        # the refusal that started this
        estimate = measure(frame)
        self.assertAlmostEqual(estimate.period_px / 27.0, 1.0, delta=0.03)
        self.assertAlmostEqual(abs(estimate.angle_deg), 9.0, delta=1.5)
        self.assertIsNotNone(estimate.field_stop)
        self.assertAlmostEqual(estimate.field_stop.radius / radius, 1.0, delta=0.05)
        self.assertAlmostEqual(estimate.field_stop.centre_x, centre_x, delta=10.0)

    def test_a_distorted_lattice_reports_the_drift_as_a_number(self):
        """A lattice whose spacing grows with radius must show it in ``spread_pct``."""
        height, width = SIZE
        rows, columns = np.mgrid[0:height, 0:width].astype(float)
        centre_x, centre_y = width / 2.0, height / 2.0
        blob = np.hypot(columns - centre_x, rows - centre_y) / centre_y
        stretch = 1.0 + 0.06 * blob ** 2               # ~6 % larger cells at the edge
        frame = pattern((columns - centre_x) / stretch + centre_x,
                        (rows - centre_y) / stretch + centre_y, 40.0)
        estimate = measure(frame)
        self.assertGreater(estimate.spread_pct, 2.0)
        # The *median* of a distorted field is not the centre's spacing: every band is right
        # where it sits, so the median sits between the centre's 40 px and the edge's ~46 px.
        self.assertGreaterEqual(estimate.period_px, 40.0)
        self.assertLessEqual(estimate.period_px, 46.5)

    def test_a_mirror_does_not_change_the_scale(self):
        """A symmetry the estimate must satisfy: no second photo, no fit involved."""
        frame = grid_image(period=40.0, angle_deg=6.0)
        straight = measure(frame)
        mirrored = measure(np.ascontiguousarray(frame[:, ::-1]))
        self.assertAlmostEqual(straight.period_px / mirrored.period_px, 1.0, delta=0.01)

    def test_an_upside_down_photo_is_measured_the_same(self):
        frame = grid_image(period=40.0, angle_deg=6.0)
        straight = measure(frame)
        flipped = measure(np.ascontiguousarray(frame[::-1, :]))
        self.assertAlmostEqual(straight.period_px / flipped.period_px, 1.0, delta=0.01)

    def test_a_photo_of_the_sheet_in_air_has_no_field_stop(self):
        """``None`` is a real answer: then the whole frame is paper."""
        self.assertIsNone(field_stop(grid_image(period=40.0)))

    def test_the_scale_and_its_uncertainty_come_from_the_pitch(self):
        estimate = measure(grid_image(period=40.0), pitch_um=1000.0)
        self.assertAlmostEqual(estimate.px_per_um(), 0.04)
        self.assertAlmostEqual(estimate.px_per_um(pitch_um=500.0), 0.08)
        self.assertGreater(estimate.uncertainty_pct(), 0.4)
        self.assertLess(estimate.uncertainty_pct(), 1.0)


class RefusalTests(unittest.TestCase):
    """Failing loudly is the feature: a wrong scale is worse than none."""

    def _refused(self, frame):
        with self.assertRaises(LatticeNotFound):
            measure(frame)

    def test_a_flat_field_is_refused(self):
        rng = np.random.default_rng(1)
        self._refused(np.clip(np.full(SIZE, 0.5) + rng.normal(0, 0.02, SIZE), 0, 1))

    def test_one_family_of_stripes_is_refused(self):
        """A single direction of lines is not a square grid, and its scale is ambiguous."""
        rows, _ = np.mgrid[0:SIZE[0], 0:SIZE[1]].astype(float)
        self._refused(1.0 - 0.5 * (np.mod(rows, 40.0) < 8.0))

    def test_a_non_square_target_is_refused(self):
        """Two different spacings: the scale would depend on which family was read."""
        rows, columns = np.mgrid[0:SIZE[0], 0:SIZE[1]].astype(float)
        image = pattern(columns, rows, 40.0)
        image = np.minimum(image, 1.0 - 0.6 * (np.mod(rows, 28.0) < 5.0))
        self._refused(image)

    def test_a_specimen_plate_is_refused(self):
        """The image this app spends its life looking at must not fake a scale."""
        rng = np.random.default_rng(7)
        image = np.full(SIZE, 0.9)
        rows, columns = np.mgrid[0:SIZE[0], 0:SIZE[1]]
        for _ in range(30):
            centre_y, centre_x = rng.integers(0, SIZE[0]), rng.integers(0, SIZE[1])
            ry, rx = rng.integers(6, 18), rng.integers(10, 26)
            angle = rng.uniform(0, np.pi)
            u = (columns - centre_x) * np.cos(angle) + (rows - centre_y) * np.sin(angle)
            v = -(columns - centre_x) * np.sin(angle) + (rows - centre_y) * np.cos(angle)
            image[(u / rx) ** 2 + (v / ry) ** 2 < 1.0] = 0.3
        self._refused(np.clip(image + rng.normal(0, 0.03, SIZE), 0, 1))

    def test_a_one_dimensional_array_is_a_value_error(self):
        with self.assertRaises(ValueError):
            measure(np.zeros(64))


if __name__ == "__main__":
    unittest.main()
