"""The tracker: identity across frames, and the mean of an animal's repeats.

No PyTorch and no Qt - :mod:`daphnia.vision.tracking` is pure stdlib, so the arithmetic
that turns a bouncing prediction into a stated estimate is testable on any machine and
can be checked by hand. That matters more here than anywhere: this is the module that
decides *which animal a number belongs to*, and an identity bug would look exactly like
a measurement.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from daphnia.vision.tracking import Measurement, Tracker, box_iou  # noqa: E402


def measurement(box, length_px):
    """One frame's measurement of one animal."""
    return Measurement(box=box, length_px=length_px)


class BoxIouTests(unittest.TestCase):
    """Overlap is what decides identity, so it is pinned with hand-computed numbers."""

    def test_a_box_overlaps_itself_completely(self):
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_half_an_overlap_is_one_third(self):
        """Two 10x10 boxes offset by 5: intersection 50, union 150 -> 1/3."""
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=6)

    def test_corners_in_any_order_and_no_overlap(self):
        self.assertEqual(box_iou((10, 10, 0, 0), (0, 0, 10, 10)), 1.0)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertEqual(box_iou(None, (0, 0, 10, 10)), 0.0)


class MeanTests(unittest.TestCase):
    """The estimate and its spread - hand-computed on two repeats."""

    def test_two_repeats_give_the_mean_and_the_standard_error(self):
        """lengths 10 and 20: mean 15, sample sd sqrt(50)=7.0711, SE 5.0, n 2."""
        tracker = Tracker()
        tracker.update([measurement((0, 0, 10, 10), 10.0)])
        tracker.update([measurement((1, 1, 11, 11), 20.0)])
        track = tracker.tracks()[0]
        self.assertEqual(track.count, 2)
        self.assertAlmostEqual(track.mean_px, 15.0)
        self.assertAlmostEqual(track.sd_px, 50 ** 0.5, places=6)
        self.assertAlmostEqual(track.stderr_px, 5.0, places=6)

    def test_one_repeat_has_no_invented_spread(self):
        """A single sample has no measured spread, so none is reported."""
        track = Tracker().update([measurement((0, 0, 10, 10), 42.0)])[0]
        self.assertEqual(track.count, 1)
        self.assertEqual(track.mean_px, 42.0)
        self.assertEqual(track.sd_px, 0.0)
        self.assertEqual(track.stderr_px, 0.0)

    def test_identical_repeats_have_no_spread(self):
        tracker = Tracker()
        for _ in range(4):
            tracker.update([measurement((0, 0, 10, 10), 33.0)])
        track = tracker.tracks()[0]
        self.assertEqual(track.count, 4)
        self.assertAlmostEqual(track.mean_px, 33.0)
        self.assertAlmostEqual(track.sd_px, 0.0)

    def test_the_spread_shrinks_with_more_repeats(self):
        """That is the whole reason to average: more repeats, a tighter estimate."""
        tracker = Tracker()
        lengths = [40.0, 44.0, 42.0, 46.0, 41.0, 43.0]
        spreads = []
        for length in lengths:
            tracker.update([measurement((0, 0, 10, 10), length)])
            spreads.append(tracker.tracks()[0].stderr_px)
        self.assertGreater(spreads[1], spreads[-1])
        self.assertLess(spreads[-1], 2.0)


class AssociationTests(unittest.TestCase):
    """Which measurement continues which animal."""

    def test_an_overlapping_box_continues_the_same_animal(self):
        tracker = Tracker()
        tracker.update([measurement((0, 0, 20, 20), 10.0)])
        seen = tracker.update([measurement((2, 2, 22, 22), 12.0)])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].track_id, 1)
        self.assertEqual(seen[0].count, 2)

    def test_a_far_box_is_a_new_animal(self):
        tracker = Tracker()
        tracker.update([measurement((0, 0, 20, 20), 10.0)])
        seen = tracker.update([measurement((500, 500, 520, 520), 10.0)])
        self.assertEqual([track.track_id for track in seen], [2])
        self.assertEqual(len(tracker), 2)          # the old one is still alive

    def test_two_animals_keep_separate_identities(self):
        tracker = Tracker()
        tracker.update([measurement((0, 0, 20, 20), 10.0),
                        measurement((100, 100, 120, 120), 30.0)])
        seen = tracker.update([measurement((1, 1, 21, 21), 12.0),
                               measurement((101, 101, 121, 121), 34.0)])
        self.assertEqual([(track.track_id, track.count) for track in seen],
                         [(1, 2), (2, 2)])
        self.assertEqual([round(track.mean_px, 1) for track in tracker.tracks()],
                         [11.0, 32.0])

    def test_the_best_overlap_wins_not_the_first_seen(self):
        """A measurement between two tracks goes to the one it overlaps most."""
        tracker = Tracker()
        tracker.update([measurement((0, 0, 40, 40), 10.0)])       # track 1
        tracker.update([measurement((100, 0, 140, 40), 10.0)])    # track 2
        seen = tracker.update([measurement((5, 5, 45, 45), 11.0)])
        self.assertEqual([track.track_id for track in seen], [1])

    def test_an_animal_missed_for_a_while_keeps_its_identity(self):
        """Within max_age a dropped detection does not start a second animal."""
        tracker = Tracker(max_age=3)
        tracker.update([measurement((0, 0, 20, 20), 10.0)])
        for _ in range(2):
            tracker.update([])
        seen = tracker.update([measurement((1, 1, 21, 21), 12.0)])
        self.assertEqual([track.track_id for track in seen], [1])
        self.assertEqual(seen[0].count, 2)

    def test_a_track_older_than_max_age_is_dropped(self):
        tracker = Tracker(max_age=2)
        tracker.update([measurement((0, 0, 20, 20), 10.0)])
        for _ in range(3):
            tracker.update([])
        self.assertEqual(len(tracker), 0)

    def test_matching_is_deterministic(self):
        """Same input, same identities - what makes an average checkable."""
        def run():
            tracker = Tracker()
            for _ in range(3):
                seen = tracker.update([measurement((0, 0, 30, 30), 10.0),
                                       measurement((40, 40, 70, 70), 20.0)])
            return [(track.track_id, track.mean_px) for track in seen]
        self.assertEqual(run(), run())

    def test_resetting_forgets_every_identity_and_starts_numbering_again(self):
        tracker = Tracker()
        tracker.update([measurement((0, 0, 20, 20), 10.0)])
        tracker.reset()
        self.assertEqual(len(tracker), 0)
        self.assertEqual(tracker.frame, 0)
        seen = tracker.update([measurement((0, 0, 20, 20), 10.0)])
        self.assertEqual(seen[0].track_id, 1)

    def test_the_payload_rides_along_so_a_caller_keeps_its_own_object(self):
        """The tracker hands back whatever it was given - it knows nothing of poses."""
        tracker = Tracker()
        tracker.update([Measurement(box=(0, 0, 10, 10), length_px=5.0,
                                    payload={"head": (1, 2)})])
        self.assertEqual(tracker.tracks()[0].last.payload, {"head": (1, 2)})

    def test_a_crossing_pair_is_mis_assigned_and_the_spread_exposes_it(self):
        """The acknowledged limit, pinned so it is written down rather than discovered.

        Matching uses **position alone** (no appearance, no motion model), so two animals
        that exchange places have their histories exchanged: each track's mean is dragged
        towards the other animal's length. The guard in the surface is that a track
        reports the **spread** of its repeats - which jumps from ~1 px to ~55 px here -
        instead of the average silently absorbing a wrong history.
        """
        tracker = Tracker()
        tracker.update([measurement((0, 0, 40, 40), 10.0),      # track 1 (left)
                        measurement((60, 0, 100, 40), 90.0)])   # track 2 (right)
        # They have exchanged places, so each box now holds the other's length.
        seen = tracker.update([measurement((60, 0, 100, 40), 12.0),
                               measurement((0, 0, 40, 40), 88.0)])
        by_id = {track.track_id: round(track.mean_px, 1) for track in seen}
        self.assertEqual(by_id, {1: 49.0, 2: 51.0})     # histories exchanged
        for track in seen:
            self.assertGreater(track.sd_px, 30.0, "the swap must be visible, not hidden")


if __name__ == "__main__":
    unittest.main()
