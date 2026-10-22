"""Plate geometry: formats, well labels and counts."""

import unittest

from daphnia import plates


class GeometryTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(plates.rows_columns("6"), (2, 3))
        self.assertEqual(plates.rows_columns("24"), (4, 6))
        self.assertEqual(plates.rows_columns("96"), (8, 12))

    def test_well_counts(self):
        self.assertEqual(plates.well_count("6"), 6)
        self.assertEqual(plates.well_count("24"), 24)
        self.assertEqual(plates.well_count("96"), 96)

    def test_labels_are_in_reading_order(self):
        self.assertEqual(plates.well_labels("6"),
                         ["A01", "A02", "A03", "B01", "B02", "B03"])
        labels = plates.well_labels("96")
        self.assertEqual((labels[0], labels[-1], len(labels)), ("A01", "H12", 96))
        self.assertEqual(len(plates.well_labels("24")), 24)

    def test_row_labels(self):
        self.assertEqual(plates.row_labels("24"), ["A", "B", "C", "D"])

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            plates.well_labels("12")


if __name__ == "__main__":
    unittest.main()
