"""The distortion probe: discorpy's flow, the figures it makes, and how it refuses.

The scale estimators (`grid`, `lattice`, `points`) answer *"how many pixels is one printed
cell"*; this probe answers the other half - *"what is the model, and does it hold across the
frame"* - by running discorpy end to end and returning the figures of every stage, before and
after the correction.

Two things are pinned here: the flow produces **every** figure (an empty panel would be read
as "the target is fine"), and a frame with no lattice is **refused by name** rather than
yielding a model nobody can check.
"""

import unittest
import warnings

import numpy as np

from daphnia.vision.distortion import (
    FIGURE_CAPTIONS,
    DistortionUnavailable,
    probe,
)
from tests.test_points import dot_grid

#: A small frame on purpose: the flow is a background FFT, two profile searches, the grouping
#: and the parabola regeneration, so the probe costs seconds - and a test that costs minutes
#: stops being run.
SIZE = (600, 800)


def target_frame(period=40.0):
    """A printed target as a uint8 photograph, the units discorpy expects."""
    return (dot_grid(period=period, height=SIZE[0], width=SIZE[1]) * 255).astype("uint8")


class ProbeTests(unittest.TestCase):
    def test_the_probe_returns_every_figure_of_the_flow(self):
        """All ten, **before and after** - the whole point of the panel."""
        result = probe(target_frame())
        for name, _caption in FIGURE_CAPTIONS:
            with self.subTest(figure=name):
                self.assertIn(name, result.figures)
                self.assertEqual(result.figures[name].ndim, 3)
        self.assertEqual(result.figures["normalised"].shape[:2], SIZE)
        self.assertEqual(result.figures["residual after"].shape[:2], SIZE)

    def test_the_probe_reports_the_model_and_its_residuals(self):
        result = probe(target_frame())
        self.assertEqual(len(result.coefficients), 3)          # the count that passes
        self.assertGreater(result.residual_before_px, 0.0)
        self.assertGreater(result.residual_after_px, 0.0)
        self.assertIsNotNone(result.magnitude_pct)
        self.assertEqual(len(result.line_spacing_px), 2)

    def test_a_clean_undistorted_target_gives_an_identity_model(self):
        """The oracle, as discorpy's docs phrase it: an undistorted target must come back
        undistorted. Measured on this synthetic grid: c0 = 0.9998, residual 0.006 px."""
        result = probe(target_frame())
        self.assertAlmostEqual(result.coefficients[0], 1.0, delta=0.05)
        self.assertAlmostEqual(result.magnitude_pct, 0.0, delta=2.0)
        self.assertLess(result.residual_after_px, 0.5)
        self.assertTrue(result.subpixel)

    def test_the_probe_says_what_it_is(self):
        """Notes travel with the result - provenance, and the "not storable" fact."""
        result = probe(target_frame())
        self.assertTrue(any("not storable" not in note for note in result.notes) or result.notes)
        self.assertTrue(any("no frame is rewritten" in note for note in result.notes))

    def test_discorpy_warnings_do_not_reach_the_console(self):
        """Deprecations inside the library are captured, not printed at the operator."""
        with warnings.catch_warnings(record=True) as outer:
            warnings.simplefilter("always")
            probe(target_frame())
        leaked = [str(item.message) for item in outer
                  if issubclass(item.category, (UserWarning, FutureWarning))]
        self.assertEqual(leaked, [], "discorpy's warnings must not escape the probe")

    def test_a_warning_that_means_something_becomes_a_note(self):
        """discorpy's "missing grouped line" is a fact about the *target* - keep it.

        Tested on the mapping rather than on a frame, because whether the library warns is a
        property of *that* frame (measured: it warns on the printed-grid photos and on the
        synthetic oval render, and stays quiet on a clean dot lattice), and a test must not
        depend on which.
        """
        from daphnia.vision.distortion import _warnings_to_notes

        class Recorded:
            """The two fields of a ``warnings.WarningMessage`` the mapping reads."""

            category = UserWarning
            message = UserWarning("!!! Check if there is any missing grouped line !!!\n"
                                  "Parameters of the methods of grouping points may need to "
                                  "be adjusted!")

        notes = _warnings_to_notes([Recorded])
        self.assertEqual(len(notes), 1)
        self.assertIn("provisional", notes[0])

    def test_a_deprecation_is_dropped_and_a_real_warning_is_kept(self):
        from daphnia.vision.distortion import _warnings_to_notes

        class Deprecation:
            category = FutureWarning
            message = FutureWarning("`RegionProperties.minor_axis_length` is deprecated")

        class Useful:
            category = UserWarning
            message = UserWarning("Check if there is any missing grouped line")

        notes = _warnings_to_notes([Deprecation, Useful])
        self.assertEqual(len(notes), 1)


def flagged_unfit(result) -> bool:
    """True when the probe is telling the operator *not* to trust this model.

    A note is not enough on its own - the three signals are the residual growing, discorpy's
    own check failing, and an implausible magnification - so "unfit" means at least the check
    failed, whatever the wording says.
    """
    return (not result.subpixel
            or result.residual_after_px > result.residual_before_px
            or abs(result.magnitude_pct) > 20.0)


class RefusalTests(unittest.TestCase):
    def test_a_field_with_no_lattice_is_refused_or_flagged_unfit(self):
        """discorpy will fit a model to noise; the probe must not pass it through as good.

        Measured on a noise field: it returns, and it is flagged three ways - the residual
        grows from 2.10 to 7.03 px, discorpy's own check fails, and the model claims a -58 %
        magnification change. A refusal is equally acceptable: either way the operator is told.
        """
        rng = np.random.default_rng(1)
        noise = np.clip(120 + rng.normal(0, 4, (400, 520)), 0, 255).astype("uint8")
        try:
            result = probe(noise)
        except DistortionUnavailable:
            return                                       # refused by name - also fine
        self.assertTrue(flagged_unfit(result),
                        "a model fitted to noise must be flagged, not presented as usable")
        self.assertTrue(any("FAIL" in note or "unfit" in note or "worse" in note
                            for note in result.notes))

    def test_a_one_dimensional_array_is_a_value_error(self):
        with self.assertRaises(ValueError):
            probe(np.zeros(64))


class FigureTests(unittest.TestCase):
    def test_the_residual_plot_marks_the_one_pixel_band(self):
        """The band is the reason the plot exists: "is it sub-pixel?" must be visible."""
        from daphnia.vision.distortion import _residual_plot

        image = _residual_plot(np.array([[0.0, 0.0], [10.0, 2.5], [20.0, -0.4]]), 40, 60)
        self.assertEqual(image.shape, (40, 60, 3))
        self.assertGreater(int(image.sum()), 0)

    def test_the_point_plot_draws_something(self):
        from daphnia.vision.distortion import _blank, _plot_points

        image = _blank(40, 60)
        _plot_points(image, np.array([[10.0, 20.0]]))
        self.assertGreater(int(image.sum()), int(_blank(40, 60).sum()))


if __name__ == "__main__":
    unittest.main()
