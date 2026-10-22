"""The Bench's grouping rule: which individual a measurement belongs to.

Hand-computed cases only (``memory-bank/mlContext.md`` §4: *hand-computed numerics* and a
*round-trip* are what a rule this load-bearing needs). Pure Python - no Qt, no torch - so
it runs on any machine.
"""

import unittest

from daphnia.vision.individuals import Individual, contains, group, individual_of, midpoint


class ContainmentTests(unittest.TestCase):
    """One box, one point - the smallest question the rule is built from."""

    BOX = (10.0, 20.0, 110.0, 120.0)

    def test_a_point_inside_is_inside(self):
        self.assertTrue(contains(self.BOX, (60.0, 70.0)))

    def test_a_point_outside_is_outside(self):
        self.assertFalse(contains(self.BOX, (5.0, 70.0)))
        self.assertFalse(contains(self.BOX, (60.0, 130.0)))

    def test_the_border_counts_as_inside(self):
        """A landmark placed **on** the rectangle an operator drew is about that animal."""
        for point in ((10.0, 20.0), (110.0, 120.0), (10.0, 70.0), (60.0, 120.0)):
            self.assertTrue(contains(self.BOX, point), point)

    def test_the_corners_do_not_have_to_be_sorted(self):
        """A drag can end anywhere, so a box may arrive bottom-right first."""
        self.assertTrue(contains((110.0, 120.0, 10.0, 20.0), (60.0, 70.0)))

    def test_the_midpoint_is_the_average_of_the_two_ends(self):
        self.assertEqual(midpoint((10.0, 20.0), (30.0, 60.0)), (20.0, 40.0))


class AttributionTests(unittest.TestCase):
    """Which individual a measurement belongs to - including the answer ``None``."""

    def setUp(self):
        self.individuals = [Individual(1, (0.0, 0.0, 100.0, 100.0)),
                            Individual(2, (200.0, 0.0, 300.0, 100.0))]

    def test_a_line_inside_a_box_is_that_individuals(self):
        found = individual_of((20.0, 40.0), (80.0, 60.0), self.individuals)
        self.assertEqual(found.index, 1)

    def test_a_line_in_another_box_is_the_other_individuals(self):
        found = individual_of((220.0, 40.0), (280.0, 60.0), self.individuals)
        self.assertEqual(found.index, 2)

    def test_a_line_outside_every_box_belongs_to_nobody(self):
        """``None`` is a real answer: the number is shown as belonging to no individual."""
        self.assertIsNone(individual_of((150.0, 40.0), (160.0, 60.0), self.individuals))

    def test_a_line_whose_midpoint_is_inside_but_ends_outside_is_attributed(self):
        """The rule that makes it usable: both ends need not be inside.

        A line drawn across an animal from its box edge outwards has a midpoint at 45 px,
        inside box 1 - it is that animal's number, even though neither end is inside.
        """
        found = individual_of((0.0, 50.0), (90.0, 50.0), self.individuals)     # midpoint 45
        self.assertEqual(found.index, 1)

    def test_the_midpoint_decides_not_the_endpoints(self):
        """The converse: one end inside, the midpoint outside, belongs to nobody.

        A line that *leaves* the box and keeps going is a measurement of something else -
        the rule is the midpoint's position, not the fact that a click landed in a box.
        """
        self.assertIsNone(individual_of((95.0, 50.0), (140.0, 50.0), self.individuals))

    def test_the_innermost_box_wins(self):
        """A box inside a box is the normal case; the smaller one is more specific."""
        nested = [Individual(1, (0.0, 0.0, 400.0, 400.0)),
                  Individual(2, (100.0, 100.0, 200.0, 200.0))]
        found = individual_of((120.0, 150.0), (180.0, 150.0), nested)
        self.assertEqual(found.index, 2)

    def test_a_pair_is_accepted_as_an_individual(self):
        """``(index, box)`` works as well as an ``Individual``, for scripts and tests."""
        found = individual_of((20.0, 40.0), (80.0, 60.0), [(7, (0.0, 0.0, 100.0, 100.0))])
        self.assertEqual(found.index, 7)

    def test_the_area_is_the_boxes_own(self):
        self.assertEqual(Individual(1, (10.0, 20.0, 110.0, 70.0)).area, 5000.0)


class GroupTests(unittest.TestCase):
    """The list the operator reads: every individual, and the measurements inside it."""

    def setUp(self):
        self.individuals = [Individual(1, (0.0, 0.0, 100.0, 100.0)),
                            Individual(2, (200.0, 0.0, 300.0, 100.0))]

    def test_measurements_are_listed_under_their_individual(self):
        first = {"p1": (20.0, 40.0), "p2": (80.0, 60.0)}
        second = {"p1": (220.0, 40.0), "p2": (280.0, 60.0)}
        grouped = group([first, second], self.individuals)
        self.assertEqual(list(grouped), [1, 2])
        self.assertEqual(grouped[1], [first])
        self.assertEqual(grouped[2], [second])

    def test_two_measurements_of_one_animal_stay_together(self):
        """The whole point: two lines on one animal are one animal's two numbers."""
        length = {"p1": (20.0, 40.0), "p2": (80.0, 40.0)}
        width = {"p1": (50.0, 20.0), "p2": (50.0, 80.0)}
        grouped = group([length, width], self.individuals)
        self.assertEqual(grouped[1], [length, width])

    def test_a_measurement_outside_every_box_is_kept_under_none(self):
        stray = {"p1": (150.0, 40.0), "p2": (160.0, 60.0)}
        grouped = group([stray], self.individuals)
        self.assertEqual(list(grouped), [None])
        self.assertEqual(grouped[None], [stray])

    def test_the_order_inside_a_group_is_the_order_given(self):
        first = {"p1": (10.0, 10.0), "p2": (20.0, 20.0)}
        second = {"p1": (30.0, 30.0), "p2": (40.0, 40.0)}
        self.assertEqual(group([first, second], self.individuals)[1], [first, second])

    def test_a_plain_pair_works_too(self):
        grouped = group([((20.0, 40.0), (80.0, 60.0))], self.individuals)
        self.assertEqual(len(grouped[1]), 1)

    def test_no_measurements_is_an_empty_mapping(self):
        self.assertEqual(group([], self.individuals), {})


if __name__ == "__main__":
    unittest.main()
