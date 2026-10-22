"""Well states: the ladder, the counts behind it, and the paging of a plate.

This is the Qt-free half of the plate page (`daphnia/wells.py`), so it is checked by
hand-computed cases and nothing else: no database, no window, no camera. Two of these
tests exist because the arithmetic is easy to get subtly wrong - **where a page ends**
(a page never splits a row) and **where the walk wraps** (after the last well the useful
next well is the first one that is still empty).
"""

import unittest

from daphnia import plates, wells


def _counts(**values):
    """A counts row as ``queries.plate_well_counts`` returns one (a mapping)."""
    return values


class StateTests(unittest.TestCase):
    """Counts -> state, including the wells nothing was ever captured on."""

    def test_an_untouched_well_is_pending(self):
        self.assertEqual(wells.state_of(None), wells.PENDING)
        self.assertEqual(wells.state_of(_counts(frames=0, bursts=0)), wells.PENDING)

    def test_one_frame_makes_a_well_captured(self):
        self.assertEqual(wells.state_of(_counts(frames=1, bursts=1)), wells.CAPTURED)

    def test_a_measurement_outranks_the_image(self):
        self.assertEqual(
            wells.state_of(_counts(frames=5, bursts=1, measured=2)), wells.MEASURED)

    def test_a_missing_key_is_zero_not_an_error(self):
        """A query that grows a column must not break an older caller."""
        self.assertEqual(wells.state_of({}), wells.PENDING)

    def test_every_state_is_described(self):
        for state in wells.STATES:
            with self.subTest(state=state):
                self.assertTrue(wells.STATE_LABELS[state])
                self.assertTrue(wells.STATE_HINTS[state])


class ProgressTests(unittest.TestCase):
    """One record per well of the format - the empty wells included."""

    def test_every_well_of_the_plate_appears_in_reading_order(self):
        records = wells.progress("24")
        self.assertEqual([row.well_label for row in records], plates.well_labels("24"))
        self.assertEqual(len(records), 24)
        self.assertEqual(records[0].well_label, "A01")
        self.assertEqual(records[-1].well_label, "D06")

    def test_an_empty_plate_has_no_done_well(self):
        records = wells.progress("6")
        self.assertEqual([row.state for row in records],
                         [wells.PENDING] * 6)
        self.assertEqual(wells.summary(records)["done"], 0)

    def test_the_counts_of_a_captured_well_reach_the_record(self):
        records = wells.progress("6", {"B02": _counts(frames=5, bursts=1, kept=1)})
        row = wells.by_label(records)["B02"]
        self.assertEqual((row.frames, row.bursts, row.kept), (5, 1, 1))
        self.assertEqual(row.state, wells.CAPTURED)
        self.assertTrue(row.is_done)
        self.assertEqual(row.describe(),
                         "B02 - Captured: 5 frame(s) in 1 burst(s), 1 kept")

    def test_a_pending_well_says_so_without_inventing_counts(self):
        self.assertEqual(wells.progress("6")[0].describe(), "A01 - Not captured")

    def test_a_measured_well_reports_its_measurements(self):
        row = wells.by_label(wells.progress(
            "24", {"A01": _counts(frames=5, bursts=1, measured=3)}))["A01"]
        self.assertEqual(row.state, wells.MEASURED)
        self.assertEqual(row.describe(),
                         "A01 - Measured: 5 frame(s) in 1 burst(s), 3 measurement(s)")

    def test_a_label_outside_the_format_does_not_invent_a_well(self):
        """A row from another plate must not add a well to this one."""
        records = wells.progress("6", {"H12": _counts(frames=1, bursts=1)})
        self.assertEqual([row.well_label for row in records], plates.well_labels("6"))
        self.assertEqual(wells.summary(records)["done"], 0)

    def test_the_summary_counts_every_state_once(self):
        records = wells.progress("24", {
            "A01": _counts(frames=5, bursts=1),
            "A02": _counts(frames=5, bursts=1, kept=1),
            "A03": _counts(frames=9, bursts=2, measured=1),
        })
        counted = wells.summary(records)
        self.assertEqual(counted["wells"], 24)
        self.assertEqual(counted["done"], 3)
        self.assertEqual(counted["measured"], 1)
        self.assertEqual(counted["by_state"], {wells.PENDING: 21,
                                               wells.CAPTURED: 2,
                                               wells.MEASURED: 1})

        self.assertEqual(counted["by_state"], {wells.PENDING: 21,
                                               wells.CAPTURED: 2,
                                               wells.MEASURED: 1})


class PageTests(unittest.TestCase):
    """How many individuals a page shows, and which wells they are."""

    def test_the_whole_plate_is_one_page_by_default(self):
        for fmt, count in (("6", 6), ("24", 24), ("96", 96)):
            with self.subTest(fmt=fmt):
                page_list = wells.pages(fmt)
                self.assertEqual(len(page_list), 1)
                self.assertEqual(len(page_list[0]), count)

    def test_a_page_never_splits_a_row(self):
        """12 wells of a 96-well plate are one row; 24 are two."""
        self.assertEqual(wells.rows_per_page("96", 12), 1)
        self.assertEqual(wells.rows_per_page("96", 24), 2)
        self.assertEqual(wells.rows_per_page("96", 18), 1)      # 18 -> one whole row
        self.assertEqual(wells.rows_per_page("96", 0), 8)

    def test_a_request_smaller_than_a_row_is_one_row(self):
        """A page that is not a plate at all would hide where its wells came from."""
        self.assertEqual(wells.rows_per_page("96", 6), 1)
        self.assertEqual(wells.rows_per_page("96", 1), 1)

    def test_a_96_well_plate_paged_at_24_is_four_full_pages(self):
        page_list = wells.pages("96", 24)
        self.assertEqual(len(page_list), 4)
        for page in page_list:
            self.assertEqual(len(page), 24)
        self.assertEqual(page_list[0][:2], ("A01", "A02"))
        self.assertEqual(page_list[0][-1], "B12")
        self.assertEqual(page_list[-1][-1], "H12")

    def test_a_24_well_plate_paged_at_6_is_four_rows(self):
        page_list = wells.pages("24", 6)
        self.assertEqual([page[0] for page in page_list], ["A01", "B01", "C01", "D01"])

    def test_a_page_size_larger_than_the_plate_is_the_plate(self):
        self.assertEqual(wells.pages("6", 48), (tuple(plates.well_labels("6")),))

    def test_every_well_lands_on_exactly_one_page_in_reading_order(self):
        for fmt in plates.FORMATS:
            for per_page in wells.PAGE_SIZES:
                with self.subTest(fmt=fmt, per_page=per_page):
                    page_list = wells.pages(fmt, per_page)
                    flat = [label for page in page_list for label in page]
                    self.assertEqual(flat, plates.well_labels(fmt))
                    self.assertEqual(len(set(flat)), len(flat))

    def test_every_offerable_page_size_is_a_positive_count_or_the_whole_plate(self):
        self.assertIn(0, wells.PAGE_SIZES)
        for size in wells.PAGE_SIZES:
            self.assertGreaterEqual(size, 0)

    def test_the_page_of_a_well_can_be_found(self):
        page_list = wells.pages("96", 24)
        self.assertEqual(wells.page_index(page_list, "A01"), 0)
        self.assertEqual(wells.page_index(page_list, "B12"), 0)
        self.assertEqual(wells.page_index(page_list, "C01"), 1)
        self.assertEqual(wells.page_index(page_list, "H12"), 3)

    def test_a_well_that_is_on_no_page_opens_the_first_one(self):
        """The page reports the mismatch; an empty grid would look like a bug."""
        self.assertEqual(wells.page_index(wells.pages("6"), "H12"), 0)


class NextPendingTests(unittest.TestCase):
    """The walk: which well the camera goes to next."""

    def test_an_untouched_plate_starts_at_the_first_well(self):
        self.assertEqual(wells.next_pending(wells.progress("24")), "A01")

    def test_the_next_well_is_the_next_one_without_an_image(self):
        records = wells.progress("6", {"A01": _counts(frames=5, bursts=1)})
        self.assertEqual(wells.next_pending(records, after="A01"), "A02")

    def test_the_walk_wraps_to_the_well_that_was_missed(self):
        """The operator walks the plate, not a list: the last well is not the end."""
        records = wells.progress("6", {"B03": _counts(frames=5, bursts=1)})
        self.assertEqual(wells.next_pending(records, after="B03"), "A01")

    def test_a_full_plate_has_no_next_well(self):
        counts = {label: _counts(frames=5, bursts=1)
                  for label in plates.well_labels("24")}
        self.assertIsNone(wells.next_pending(wells.progress("24", counts), after="A01"))

    def test_an_unknown_current_well_starts_the_walk_at_the_beginning(self):
        records = wells.progress("6", {"A01": _counts(frames=5, bursts=1)})
        self.assertEqual(wells.next_pending(records, after="H12"), "A02")

    def test_an_empty_page_has_no_next_well(self):
        self.assertIsNone(wells.next_pending(()))


if __name__ == "__main__":
    unittest.main()
