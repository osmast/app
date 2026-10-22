"""Burst selection: the sharpest frame must win."""

import unittest

import numpy as np

from daphnia.vision.quality import score_all, select_best, sharpness


def _flat(size=64):
    return np.full((size, size, 3), 120, dtype=np.uint8)


def _striped(size=64):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[::4, :] = 255
    return image


class SharpnessTests(unittest.TestCase):
    def test_striped_scores_higher_than_flat(self):
        self.assertGreater(sharpness(_striped()), sharpness(_flat()))

    def test_select_best_returns_the_sharpest_index(self):
        index, score = select_best([_flat(), _striped(), _flat()])
        self.assertEqual(index, 1)
        self.assertGreater(score, 0.0)

    def test_select_best_handles_empty_input(self):
        self.assertEqual(select_best([]), (None, 0.0))
        self.assertEqual(select_best(None), (None, 0.0))

    def test_score_all_matches_input_length(self):
        self.assertEqual(len(score_all([_flat(), _striped()])), 2)


if __name__ == "__main__":
    unittest.main()
