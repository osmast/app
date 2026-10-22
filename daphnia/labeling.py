"""Landmark annotation: two points per specimen, or one landmark at a time.

Two models are supported, and both the Bench and the Dataset surface use them:

* **paired** - click both landmarks of the current specimen; the distance between
  them is one interval. Best for a single animal.
* **pass** - place *one* landmark across many specimens ("all the heads"), then the
  next landmark ("all the tails"), then the width sides. The eye and the hand find
  the same landmark once per pass instead of once per animal, which is what makes a
  frame full of specimens bearable to label.

A :class:`Draft` holds the points of a pass until both ends of an interval exist, at
which point ``complete_intervals`` yields the rows that may be stored. The module is
deliberately **Qt-free**, so the workflow is unit-testable and usable from a script.
"""

from daphnia import annotation
from daphnia.study import LANDMARKS, LANDMARK_TRAIT, TRAITS, TRAIT_LANDMARKS

#: Two clicks describe one interval for the current specimen.
PAIR_MODE = "pair"

#: One landmark is placed for many specimens, then the next landmark.
PASS_MODE = "pass"

#: How a mode is phrased for an operator.
MODE_LABELS = {
    PAIR_MODE: "Two points per specimen",
    PASS_MODE: "One landmark at a time",
}


def intervals_for(schema) -> dict:
    """``{trait: (first, second)}`` of an annotation schema.

    A dataset picks the keypoint system (two points, four points), so the intervals
    an operator measures follow from the *model*, not from a module constant. The
    default keeps the study's own vocabulary (length + width).
    """
    if schema is None:
        return dict(TRAIT_LANDMARKS)
    return {trait: (first, second) for trait, first, second in schema.intervals}


def landmarks_for(schema) -> tuple:
    """The landmarks of a schema, in click order (the study's, when there is none)."""
    if schema is None:
        return tuple(LANDMARKS)
    return tuple(schema.keypoints)


class Draft:
    """Partial landmarks for one frame, merged into intervals as pairs complete.

    In the pass workflow nothing can be stored until a specimen's two landmarks both
    exist. The draft keeps the single points until then; ``complete_intervals`` says
    which specimens are now finished - exactly when a row may be written.

    The intervals are passed in (see :func:`intervals_for`), so the same draft works
    for a two-keypoint model, a four-keypoint one, or the study's own vocabulary.
    """

    def __init__(self, intervals=None, landmarks=None):
        self._points = {}   # (specimen_index, landmark) -> (x, y)
        self._intervals = dict(intervals) if intervals else dict(TRAIT_LANDMARKS)
        self._order = tuple(landmarks) if landmarks else tuple(LANDMARKS)

    @property
    def intervals(self) -> dict:
        """``{trait: (first, second)}`` this draft merges into."""
        return dict(self._intervals)

    @property
    def landmark_order(self) -> tuple:
        """Every landmark this draft knows, in click order."""
        points = [name for first, second in self._intervals.values()
                  for name in (first, second)]
        for name in self._order:
            if name not in points:
                points.append(name)
        return tuple(points)

    def place(self, specimen_index, landmark, point):
        """Record one landmark of one specimen, in image pixels."""
        self._points[(int(specimen_index), landmark)] = (float(point[0]),
                                                         float(point[1]))

    def get(self, specimen_index, landmark):
        """The point placed for that landmark of that specimen, or ``None``."""
        return self._points.get((int(specimen_index), landmark))

    def unplace(self, specimen_index, landmark) -> bool:
        """Forget one placed point; returns True when there was one to forget."""
        return self._points.pop((int(specimen_index), landmark), None) is not None

    def last(self):
        """``(specimen_index, landmark)`` of the most recently placed point, or ``None``.

        Insertion order is preserved, so this is the point Undo should remove first -
        whatever landmark it belongs to.
        """
        if not self._points:
            return None
        return next(reversed(self._points))

    def points_by_specimen(self) -> dict:
        """``{specimen_index: [point, ...]}`` - every placed point, grouped."""
        grouped = {}
        for (specimen, _landmark), point in self._points.items():
            grouped.setdefault(specimen, []).append(point)
        return grouped

    def landmarks(self) -> set:
        """The landmark names with at least one point placed (the open passes)."""
        return {landmark for (_specimen, landmark) in self._points}

    def specimens(self) -> list:
        """The specimen indices with at least one landmark placed, sorted."""
        return sorted({specimen for specimen, _ in self._points})

    def placed_landmarks(self, landmark) -> dict:
        """``{specimen_index: point}`` for one landmark - the current pass."""
        return {specimen: point
                for (specimen, name), point in self._points.items()
                if name == landmark}

    def complete_intervals(self) -> list:
        """``[(specimen, trait, p1, p2)]`` for every specimen with a full interval."""
        done = []
        for specimen in self.specimens():
            for trait, (first, second) in self._intervals.items():
                p1 = self.get(specimen, first)
                p2 = self.get(specimen, second)
                if p1 is not None and p2 is not None:
                    done.append((specimen, trait, p1, p2))
        return done

    def count(self, trait) -> int:
        """How many specimens have both landmarks of ``trait``."""
        return sum(1 for _, name, _, _ in self.complete_intervals() if name == trait)

    def take_complete(self) -> list:
        """Return the completed intervals **and** drop their points from the draft.

        Called after every placed point: once an interval is stored, its two points
        no longer belong in the draft, so the in-progress pass only ever holds the
        landmarks that are still missing a partner.
        """
        done = self.complete_intervals()
        for specimen, trait, _, _ in done:
            for landmark in self._intervals.get(trait, ()):
                self._points.pop((specimen, landmark), None)
        return done

    def clear(self):
        """Forget every partial point (a new frame, or a fresh pass)."""
        self._points.clear()


def landmark_options(schema=None) -> list:
    """``[(value, label)]`` for a landmark selector, in click order.

    With a schema (a dataset's model) the choices are that model's keypoints; without
    one they are the study's: the length's pair, then the width's.
    """
    if schema is not None:
        return list(annotation.landmark_choices(schema))
    return [(name, f"{name}  ({LANDMARK_TRAIT[name]})") for name in LANDMARKS]


def next_landmark(landmark, order=None):
    """The landmark after ``landmark`` in click order, or ``None`` at the end.

    This is what a completed pass advances to: heads -> spine base -> left side ->
    right side, so an operator never has to cycle specimens (or landmarks) by hand.
    A dataset's keypoint system may be shorter: two points simply ends after the
    second one.
    """
    order = tuple(order) if order else tuple(LANDMARKS)
    if landmark not in order:
        return None
    position = order.index(landmark) + 1
    return order[position] if position < len(order) else None


def next_trait(trait, traits=None):
    """The next trait in measure order, wrapping (length -> width -> length)."""
    order = list(traits) if traits else list(TRAITS)
    if not order:
        return trait
    if trait not in order:
        return order[0]
    return order[(order.index(trait) + 1) % len(order)]


def next_missing(draft: Draft, landmark) -> int:
    """The first specimen index (1-based) with no point for ``landmark`` yet.

    This is what a click in pass mode advances to, so an operator can place "all the
    heads" without touching a specimen counter.
    """
    placed = draft.placed_landmarks(landmark)
    index = 1
    while index in placed:
        index += 1
    return index
