"""Length units: micrometres at the sensor, millimetres on paper."""

import unittest

from daphnia import units


class UnitTests(unittest.TestCase):
    def test_to_mm(self):
        self.assertAlmostEqual(units.to_mm(1000.0), 1.0)
        self.assertAlmostEqual(units.to_mm(160.1), 0.1601)

    def test_format_carries_both_units(self):
        self.assertEqual(units.format_length(160.1), "160.1 um (0.160 mm)")


class TotalUncertaintyTests(unittest.TestCase):
    """The whole budget of one length - `docs/accuracy.md` §1 and §3, hand-computed.

    The two worked cases in the document are the oracle here: if a constant or a
    formula drifts, these two numbers move and the document is wrong with them.
    """

    #: The eyepiece-grid chain the Study path is measured on: 27.264 px/mm, ±0.2934 %.
    EYEPIECE = 0.027264
    #: The wide-lens chain the pose model was scored on: 74.8 µm/px, ±0.50 %.
    WIDE = 0.01336

    def test_the_scale_term_is_the_one_that_was_always_reported(self):
        """§3.1: 2000 µm at ±0.2934 % is the ±5.87 the screen used to show alone."""
        self.assertAlmostEqual(units.uncertainty_um(2000.0, 0.2934), 5.868, places=3)

    def test_the_click_term_is_a_constant_in_micrometres(self):
        """S6: ``2c/S`` - the percentage and the length cancel.

        Two clicks are one line however long the animal is, so a bigger animal does
        not buy a smaller click error. This is the identity the whole section rests on.
        """
        self.assertAlmostEqual(units.click_uncertainty_um(self.EYEPIECE), 73.3568,
                               places=4)
        total_700 = units.total_uncertainty_um(700.0, self.EYEPIECE, 0.2934)
        total_2000 = units.total_uncertainty_um(2000.0, self.EYEPIECE, 0.2934)
        click = units.click_uncertainty_um(self.EYEPIECE)
        self.assertAlmostEqual(total_700 ** 2 - click ** 2,
                               units.quantisation_um(self.EYEPIECE) ** 2
                               + units.uncertainty_um(700.0, 0.2934) ** 2, places=6)
        self.assertLess(abs(total_2000 - total_700), 0.25)

    def test_the_quantisation_term_is_the_pixel_grid(self):
        """S5: ``sqrt(2)/sqrt(12) = 0.408 px`` per length, i.e. 15.0 µm here."""
        self.assertAlmostEqual(units.QUANTISATION_PX, 0.408248, places=6)
        self.assertAlmostEqual(units.quantisation_um(self.EYEPIECE), 14.9739,
                               places=4)

    def test_the_study_path_total_matches_the_document(self):
        """§3.1: scale 5.87 + click 73.4 + quantisation 15.0 -> 75.1 µm."""
        total = units.total_uncertainty_um(2000.0, self.EYEPIECE, 0.2934)
        self.assertAlmostEqual(total, 75.099, places=3)
        # The screen's own +/- is 13x smaller than the budget - the finding of §3.
        self.assertAlmostEqual(total / units.uncertainty_um(2000.0, 0.2934),
                               12.8, places=1)

    def test_the_predict_path_total_matches_the_document(self):
        """§3.2: with the pose model's 224 µm and **no click** (nobody clicked)."""
        total = units.total_uncertainty_um(2000.0, self.WIDE, 0.50,
                                           extra_um=224.0, click_error_px=0.0)
        self.assertAlmostEqual(total, 226.3, places=1)
        # Leaving the click in would be a number about a person who was not there -
        # and it is 150 µm, two thirds of the model's own error.
        with_click = units.total_uncertainty_um(2000.0, self.WIDE, 0.50,
                                               extra_um=224.0)
        self.assertAlmostEqual(with_click, 271.3, places=1)
        self.assertGreater(with_click - total, 40.0)

    def test_a_weak_calibration_dominates_its_own_budget(self):
        """§5(b): a 14.7 px span (±13.61 %) on a 700 µm neonate -> 121 µm."""
        total = units.total_uncertainty_um(700.0, self.EYEPIECE, 13.61)
        self.assertAlmostEqual(total, 121.2, places=1)

    def test_an_extra_term_adds_in_quadrature(self):
        """A term that is 0, then 224 µm, moves the total by 224 µm - not by a ratio."""
        base = units.total_uncertainty_um(2000.0, self.EYEPIECE, 0.50,
                                          click_error_px=0.0)
        with_model = units.total_uncertainty_um(2000.0, self.EYEPIECE, 0.50,
                                                extra_um=224.0, click_error_px=0.0)
        self.assertAlmostEqual(with_model, (base ** 2 + 224.0 ** 2) ** 0.5, places=6)
        self.assertGreater(with_model, base)

    def test_a_scale_that_is_not_a_scale_is_refused(self):
        """A zero or negative scale is a bug, not a small uncertainty (law 8)."""
        for bad in (0.0, -1.0):
            with self.subTest(px_per_um=bad):
                with self.assertRaises(ValueError):
                    units.total_uncertainty_um(100.0, bad, 1.0)

    def test_the_text_carries_the_share_of_the_animal(self):
        text = units.format_total_uncertainty(2000.0, self.EYEPIECE, 0.2934)
        self.assertEqual(text, "total +/- 75.1 um (3.8 %)")
        neonate = units.format_total_uncertainty(700.0, self.EYEPIECE, 13.61)
        self.assertIn("121.2 um", neonate)
        self.assertIn("17.3 %", neonate)

    def test_a_stored_total_is_rendered_instead_of_derived(self):
        """A row's own number wins, so a screen and an export cannot disagree."""
        text = units.format_total_uncertainty(2000.0, self.EYEPIECE, 0.2934,
                                              total_um=80.0)
        self.assertIn("80.0 um", text)
        self.assertIn("4.0 %", text)


if __name__ == "__main__":
    unittest.main()
