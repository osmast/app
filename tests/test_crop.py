"""The crop: the mapping between what the annotator sees and what the corpus stores.

A crop is where a labelling bug would be invisible: a keypoint stored in the
*magnified picture's* coordinates looks perfectly placed on screen and is wrong in
the corpus. So the mapping is pinned with hand-computed numbers, the round trip is
asserted, and the enlargement is checked to be **exact** (each source pixel becomes
a ``scale x scale`` block) - which is what lets an operator trust the pixel they
click. The same class of bug cost this project thirty epochs of a metric reading
0.000 (`mlContext.md` law 1), so it is tested here rather than observed later.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from daphnia.vision.crop import Crop, crop_for_box


class CropMappingTests(unittest.TestCase):
    """Crop pixels and source pixels, in both directions."""

    def test_a_click_is_converted_back_to_source_pixels(self):
        """Hand-computed: origin (100, 50) at x4, so crop (40, 40) is source (110, 60)."""
        crop = Crop(origin=(100, 50), size=(40, 30), scale=4)
        self.assertEqual(crop.to_crop((100, 50)), (0.0, 0.0))
        self.assertEqual(crop.to_crop((110, 60)), (40.0, 40.0))
        self.assertEqual(crop.invert((40, 40)), (110.0, 60.0))

    def test_the_round_trip_is_lossless(self):
        crop = Crop(origin=(496, 539), size=(139, 122), scale=3)
        for point in ((500.5, 540.25), (600.0, 630.0), (634.75, 660.5)):
            self.assertEqual(crop.invert(crop.to_crop(point)), point)

    def test_a_fractional_click_keeps_its_fraction(self):
        """A click is not a pixel: the mapping must not round it away."""
        crop = Crop(origin=(0, 0), size=(100, 100), scale=2)
        self.assertEqual(crop.invert((3, 5)), (1.5, 2.5))

    def test_the_origin_and_size_are_integers(self):
        """``window()`` slices with these, so a fraction would make the two disagree."""
        crop = Crop(origin=(10.4, 20.6), size=(30.5, 40.4), scale=1)
        self.assertEqual(crop.origin, (10, 21))
        self.assertEqual(crop.size, (30, 40))
        self.assertEqual((crop.width, crop.height), (30, 40))

    def test_output_size_is_the_window_times_the_factor(self):
        crop = Crop(origin=(0, 0), size=(40, 30), scale=4)
        self.assertEqual(crop.output_size, (160, 120))

    def test_contains_tells_which_source_points_are_inside(self):
        crop = Crop(origin=(100, 50), size=(40, 30), scale=2)
        self.assertTrue(crop.contains((100, 50)))
        self.assertTrue(crop.contains((139, 79)))
        self.assertFalse(crop.contains((99, 50)))
        self.assertFalse(crop.contains((140, 79)))

    def test_it_describes_itself_for_the_readout(self):
        crop = Crop(origin=(100, 50), size=(40, 30), scale=4)
        self.assertIn("40 x 30 px", crop.describe())
        self.assertIn("x4", crop.describe())
        self.assertIn("160 x 120 px", crop.describe())


class CropWindowTests(unittest.TestCase):
    """The magnified picture: exact, and the same integers the mapping uses."""

    def setUp(self):
        self.frame = np.zeros((60, 100, 3), np.uint8)
        self.frame[20:40, 30:50] = 255        # a white block: the "animal"

    def test_the_window_is_the_size_the_mapping_promises(self):
        crop = Crop(origin=(30, 20), size=(20, 20), scale=4)
        window = crop.window(self.frame)
        self.assertEqual(window.shape[:2],
                         (crop.output_size[1], crop.output_size[0]))

    def test_the_enlargement_is_exact_and_not_interpolated(self):
        """x3 of a 2x2 window is 3x3 blocks of the *same* source pixel."""
        crop = Crop(origin=(0, 0), size=(2, 2), scale=3)
        frame = np.zeros((2, 2, 3), np.uint8)
        frame[0, 0] = (10, 20, 30)
        frame[1, 1] = (40, 50, 60)
        window = crop.window(frame)
        self.assertEqual(window.shape, (6, 6, 3))
        self.assertTrue(np.all(window[0:3, 0:3] == (10, 20, 30)))
        self.assertTrue(np.all(window[3:6, 3:6] == (40, 50, 60)))

    def test_a_scale_of_one_returns_the_patch_untouched(self):
        crop = Crop(origin=(30, 20), size=(20, 20), scale=1)
        window = crop.window(self.frame)
        self.assertEqual(window.shape, (20, 20, 3))
        self.assertTrue(np.all(window == 255))

    def test_nothing_is_returned_for_nothing(self):
        crop = Crop(origin=(0, 0), size=(10, 10), scale=2)
        self.assertIsNone(crop.window(None))


class CropForBoxTests(unittest.TestCase):
    """The crop a box asks for: context, an integer factor, never shrunk."""

    def test_the_window_keeps_context_around_the_box(self):
        crop = crop_for_box((100, 100, 200, 200), (1000, 1000), margin=0.25)
        self.assertEqual((crop.origin, crop.size), ((75, 75), (150, 150)))

    def test_the_window_is_clamped_to_the_image(self):
        crop = crop_for_box((0, 0, 100, 100), (200, 200), margin=0.5)
        self.assertEqual((crop.origin, crop.size), ((0, 0), (150, 150)))

    def test_a_real_corpus_box_is_magnified_three_times(self):
        """The first box of the real corpus: ~92 px of animal in a 1280 x 720 frame."""
        crop = crop_for_box((519.72, 562.38, 611.92, 637.21), (1280, 720))
        self.assertEqual((crop.origin, crop.size), ((496, 539), (139, 122)))
        self.assertEqual(crop.scale, 3)
        self.assertEqual(crop.output_size, (417, 366))

    def test_a_big_box_is_never_shrunk(self):
        crop = crop_for_box((0, 0, 900, 700), (1000, 800))
        self.assertEqual(crop.scale, 1)

    def test_the_magnification_is_capped(self):
        crop = crop_for_box((10, 10, 11, 11), (100, 100), max_scale=8)
        self.assertEqual(crop.scale, 8)

    def test_a_box_at_the_far_corner_stays_inside_the_image(self):
        crop = crop_for_box((1200, 700, 1280, 720), (1280, 720))
        self.assertLessEqual(crop.origin[0] + crop.width, 1280)
        self.assertLessEqual(crop.origin[1] + crop.height, 720)

    def test_a_degenerate_box_is_refused_in_words(self):
        with self.assertRaises(ValueError) as caught:
            crop_for_box((10, 10, 10, 20), (100, 100))
        self.assertIn("degenerate", str(caught.exception))

    def test_a_box_outside_the_image_is_refused_in_words(self):
        with self.assertRaises(ValueError) as caught:
            crop_for_box((200, 200, 300, 300), (100, 100))
        self.assertIn("outside", str(caught.exception))
