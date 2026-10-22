"""Reading any photo: formats, filenames, sizes and failures.

The format an iPhone writes by default is ``.HEIC``, which OpenCV's stock codecs
cannot decode - so "use any photo" lives or dies on this module.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from daphnia.vision.images import ImageLoadError, image_size, load_image

SIPS = "/usr/bin/sips"
HAS_SIPS = sys.platform == "darwin" and Path(SIPS).exists()


def write_image(path, width=60, height=40, seed=0):
    """A small random PNG to read back."""
    image = (np.random.default_rng(seed).random((height, width, 3)) * 255).astype(
        np.uint8)
    cv2.imwrite(str(path), image)
    return image


class LoadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name)

    def test_a_png_is_read_as_a_bgr_frame(self):
        path = self.folder / "a.png"
        write_image(path, width=60, height=40)
        frame = load_image(path)
        self.assertEqual(frame.shape, (40, 60, 3))

    def test_a_jpeg_is_read(self):
        path = self.folder / "a.jpg"
        write_image(path)
        self.assertEqual(load_image(path).shape, (40, 60, 3))

    def test_a_unicode_filename_is_read(self):
        """A German lab folder name must not decide whether calibration works."""
        path = self.folder / "Ordner_\u00fc_\u00f6_\u00df test.png"
        write_image(path)
        self.assertIsNotNone(load_image(path))

    def test_a_string_path_is_accepted(self):
        path = self.folder / "a.png"
        write_image(path)
        self.assertIsNotNone(load_image(str(path)))

    def test_the_size_can_be_read_without_keeping_the_pixels(self):
        path = self.folder / "a.png"
        write_image(path, width=64, height=48)
        self.assertEqual(image_size(path), (64, 48))

    def test_a_missing_file_says_so(self):
        with self.assertRaises(ImageLoadError) as raised:
            load_image(self.folder / "nope.png")
        self.assertIn("no such file", str(raised.exception))

    def test_an_empty_file_says_so(self):
        path = self.folder / "empty.png"
        path.write_bytes(b"")
        with self.assertRaises(ImageLoadError) as raised:
            load_image(path)
        self.assertIn("empty", str(raised.exception))

    def test_something_that_is_not_an_image_says_so(self):
        """A silent None is how "any photo" becomes a mystery."""
        path = self.folder / "junk.png"
        path.write_bytes(b"this is not an image")
        with self.assertRaises(ImageLoadError) as raised:
            load_image(path)
        self.assertIn("could not decode", str(raised.exception))


@unittest.skipUnless(HAS_SIPS, "needs the macOS sips tool to write a .HEIC")
class PhoneFormatTests(unittest.TestCase):
    """The format a phone produces by default, which OpenCV will not read."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name)

    def _make_heic(self, width=60, height=40):
        png = self.folder / "grid.png"
        write_image(png, width=width, height=height)
        heic = self.folder / "grid.heic"
        subprocess.run([SIPS, "-s", "format", "heic", str(png), "--out", str(heic)],
                       check=True, capture_output=True)
        if not heic.exists():
            self.skipTest("sips could not write a HEIC here")
        return heic

    def test_a_heic_is_converted_and_read(self):
        frame = load_image(self._make_heic())
        self.assertEqual(frame.shape, (40, 60, 3))

    def test_conversion_can_be_turned_off(self):
        """Documents the seam: without the system converter a HEIC cannot be read."""
        heic = self._make_heic()
        if cv2.imread(str(heic)) is not None:
            self.skipTest("this OpenCV build reads HEIC natively")
        with self.assertRaises(ImageLoadError):
            load_image(heic, allow_system_conversion=False)

    def test_a_heic_report_is_specific_when_conversion_is_off(self):
        heic = self._make_heic()
        if cv2.imread(str(heic)) is not None:
            self.skipTest("this OpenCV build reads HEIC natively")
        with self.assertRaises(ImageLoadError) as raised:
            load_image(heic, allow_system_conversion=False)
        self.assertIn(".heic", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
