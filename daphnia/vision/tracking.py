"""Keeping the same animal the same animal, and averaging its length over repeats.

## Why this exists

A continuing measurement of a container holding several specimens has two problems a
single frame cannot answer, and both are solved by the same bookkeeping:

1. **Which animal is this?** A per-frame list of boxes is anonymous: box 3 on frame 40
   has no relation to box 3 on frame 41, and a colour assigned by list position moves
   between animals as they swim past each other. Measuring "each one" needs an identity
   that outlives the frame.
2. **One frame is one sample.** The pose model jitters, and the animal rotates - the
   lateral-view length changes with the angle it happens to be at. The **mean of N
   repeats** has a standard error of ``sd / sqrt(N)``: that is the whole reason to
   average, and it is the difference between a bouncing number and a stated estimate
   (``memory-bank/mlContext.md`` §3, §8).

## What this is, and is not

It is **IoU association**: each measurement is matched to the track it overlaps most,
the classic SORT idea (arXiv:1602.00763) reduced to what this container needs - no
model, no Kalman filter, no appearance feature, **no dependency**. ``mlContext.md`` §5
notes that the animal-pose projects (SLEAP, openpifpaf, AlphaPose) are worth reading
for their *tracking-free* inference, and that we did not need tracker machinery yet;
several species in one container is the case that changes that, so this is the
smallest thing that answers "which one was that".

What it is **not** is a solution to identity when animals overlap and cross: a greedy
IoU match can hand one animal's history to another, and the only honest guard is that
the **spread across repeats is reported with the average** - a swap then shows up as a
jump in a track's spread instead of hiding inside it.

## The unit law

A track averages **pixel lengths**, and the conversion to micrometres happens at
*display* time. So changing the calibration re-labels every track correctly, and a
track can never mix two scales inside one mean - the trap that makes two numbers
quietly incomparable (``projectbrief.md``, non-negotiable 1).
"""

import math
from dataclasses import dataclass

#: How much a measurement and a track must overlap to be called the same animal.
DEFAULT_IOU = 0.30

#: How many processed frames a track survives without being seen again. Long enough to
#: bridge a detection that dropped out, short enough that the average does not silently
#: span a minute of swimming.
DEFAULT_MAX_AGE = 15


def _normalised(box) -> tuple:
    """``(x1, y1, x2, y2)`` with the corners sorted."""
    x1, y1, x2, y2 = (float(value) for value in box)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def box_iou(first, second) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes, in any frame."""
    if first is None or second is None:
        return 0.0
    ax1, ay1, ax2, ay2 = _normalised(first)
    bx1, by1, bx2, by2 = _normalised(second)
    overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = overlap_w * overlap_h
    if intersection <= 0.0:
        return 0.0
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection)
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class Measurement:
    """One animal measured on one frame: where it was, how long it was, and what it was.

    ``payload`` carries the caller's own object (a
    :class:`~daphnia.vision.rtmpose.detect.PosePrediction`), so a track can hand back
    the keypoints it last saw without this module knowing what a keypoint is.
    """

    box: tuple                 # (x1, y1, x2, y2) in source pixels
    length_px: float           # the head-to-tail distance, in source pixels
    payload: object = None


@dataclass
class Track:
    """One individual's history: its latest measurement, and the mean over its repeats.

    ``sum_px`` and ``sum_sq_px`` are accumulated so the mean and the spread come out of
    one pass - O(1) per frame however long the animal is followed, and the same
    arithmetic a summing scale uses.
    """

    track_id: int
    last: Measurement
    hits: int = 1
    first_frame: int = 0
    last_frame: int = 0
    sum_px: float = 0.0
    sum_sq_px: float = 0.0

    def add(self, measurement, frame):
        """Fold one more repeat into the track."""
        self.last = measurement
        self.hits += 1
        self.last_frame = int(frame)
        self.sum_px += float(measurement.length_px)
        self.sum_sq_px += float(measurement.length_px) ** 2

    @property
    def count(self) -> int:
        """How many repeats this track's mean is over - its ``n``."""
        return self.hits

    @property
    def mean_px(self) -> float:
        """Mean head-to-tail length over the repeats, in source pixels."""
        return self.sum_px / self.hits if self.hits else 0.0

    @property
    def sd_px(self) -> float:
        """Sample standard deviation over the repeats (``n - 1``); 0.0 for one.

        The **sample** deviation, not the population one: these repeats are a sample of
        the poses this animal will take, and the mean of that sample is what gets
        reported - so its own uncertainty is what ``sd / sqrt(n)`` describes.
        """
        if self.hits < 2:
            return 0.0
        mean = self.mean_px
        variance = (self.sum_sq_px - self.hits * mean * mean) / (self.hits - 1)
        return math.sqrt(max(0.0, variance))

    @property
    def stderr_px(self) -> float:
        """Standard error of the mean, in pixels - the spread of the *estimate*.

        Zero for a single repeat, deliberately: one sample has no measured spread, and
        inventing one would be exactly the confident-looking claim this project refuses.
        """
        return self.sd_px / math.sqrt(self.hits) if self.hits else 0.0

    def describe(self) -> str:
        """One line for a status label."""
        return (f"#{self.track_id}  n={self.hits}  {self.mean_px:.1f} +/- "
                f"{self.stderr_px:.1f} px  sd {self.sd_px:.1f}")


class Tracker:
    """Matches each frame's measurements to the tracks they continue.

    Greedy and deterministic: every (measurement, track) pair above the overlap
    threshold is taken **best-first**, so the same input always yields the same
    identities - which is what makes a reported average checkable by re-running it.
    """

    def __init__(self, iou=DEFAULT_IOU, max_age=DEFAULT_MAX_AGE):
        self.iou = float(iou)
        self.max_age = int(max_age)
        self._tracks = {}
        self._next_id = 1
        self._frame = 0

    # -- the API -----------------------------------------------------------
    def update(self, measurements) -> list:
        """Fold one frame's measurements in; returns the tracks **seen this frame**.

        A measurement that overlaps no known track starts a new one. A track not seen
        for more than :attr:`max_age` processed frames is dropped, so an animal that
        left the field of view, or was missed for a while, is not kept alive by stale
        arithmetic.
        """
        self._frame += 1
        self._tracks = {track.track_id: track for track in self._tracks.values()
                        if self._frame - track.last_frame <= self.max_age}

        pairs = []
        for index, measurement in enumerate(measurements):
            for track in self._tracks.values():
                overlap = box_iou(measurement.box, track.last.box)
                if overlap >= self.iou:
                    pairs.append((overlap, index, track.track_id))
        # Best overlap first, ties broken by (measurement, track) so the outcome never
        # depends on dictionary order.
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

        claimed_measurements, claimed_tracks, seen = set(), set(), []
        for _overlap, index, track_id in pairs:
            if index in claimed_measurements or track_id in claimed_tracks:
                continue
            track = self._tracks[track_id]
            track.add(measurements[index], self._frame)
            claimed_measurements.add(index)
            claimed_tracks.add(track_id)
            seen.append(track)

        for index, measurement in enumerate(measurements):
            if index not in claimed_measurements:
                seen.append(self._start(measurement))

        seen.sort(key=lambda track: track.track_id)
        return seen

    def _start(self, measurement) -> Track:
        """A new identity for a measurement nothing matched."""
        track = Track(track_id=self._next_id, last=measurement,
                      first_frame=self._frame, last_frame=self._frame,
                      sum_px=float(measurement.length_px),
                      sum_sq_px=float(measurement.length_px) ** 2)
        self._tracks[track.track_id] = track
        self._next_id += 1
        return track

    def reset(self):
        """Forget every track - a new source is new animals.

        Called when the source changes, because carrying an identity across a video
        change would attribute one animal's average to another.
        """
        self._tracks = {}
        self._next_id = 1
        self._frame = 0

    # -- views -------------------------------------------------------------
    @property
    def frame(self) -> int:
        """How many frames this tracker has folded in."""
        return self._frame

    def tracks(self) -> list:
        """Every live track, oldest identity first."""
        return sorted(self._tracks.values(), key=lambda track: track.track_id)

    def __len__(self) -> int:
        return len(self._tracks)
