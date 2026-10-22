"""Which **individual** a measurement belongs to - the Bench's grouping.

## Why this exists

A Bench frame can hold several animals, and the Bench measures *distances*: two clicks
make a line. That left every number anonymous - two lines on one animal were two rows
with nothing saying they were the same animal, and the only "specimen" the surface knew
was the position of the row in its own list. The question *"how long **is this one**, and
what else did I measure on it?"* could not be asked of the result at all.

The answer is a concept the rest of the application already has: an **individual**. On the
Study side it is `observations.individual_index` ("Individual 3"); in the corpus it is one
box per specimen (`corpus_boxes`, ``mlContext.md`` §10: *one box = one individual*); on the
Predict side it is `vision.tracking.Track` - a box that stands for one animal and keeps its
number and colour. On the Bench the box the operator draws (or accepts from RTMDet) **is**
that individual, and every measurement drawn inside it belongs to it.

## The rule, stated once (``memory-bank/mlContext.md`` law 1)

    A measurement belongs to the individual whose box contains the measurement's
    **midpoint**. When several boxes contain it, the **smallest** (innermost) one wins.
    When no box contains it, the measurement belongs to **no** individual - and it is
    listed as such rather than attached to one (law 8: a failure is named, never silent).

* the **midpoint**, not the endpoints: an animal's length line very often starts on the
  box edge or a few pixels outside it, and a rule demanding *both* ends inside would
  silently drop measurements the operator clearly meant for that animal;
* the **innermost** box, not the first or the largest: a box drawn inside a box is the
  normal case when one animal sits inside a region, and the smaller box is the more
  specific statement about where the measurement was taken;
* everything in **source-image pixels** of the frame on screen - the one frame of
  reference the canvas, the measurements and the boxes all share. No mapping is re-derived
  at a call site.

This module is pure Python - no Qt, no torch (``memory-bank/systemPatterns.md``), so the
rule can be tested with one hand-computed box and used from a script.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Individual:
    """One animal on the frame: the box that stands for it, and the index it answers to.

    ``index`` is the measurement row's own ``specimen_index``, so the number the operator
    reads in the list ("Individual 2") is the number the store records - one vocabulary,
    no translation table.
    """

    index: int
    box: tuple                  # (x1, y1, x2, y2), source pixels

    @property
    def area(self) -> float:
        """The box's area in square pixels - what *innermost* is decided by."""
        x1, y1, x2, y2 = _normalised(self.box)
        return (x2 - x1) * (y2 - y1)


def _normalised(box) -> tuple:
    """``(x1, y1, x2, y2)`` with the top-left corner first, whatever was passed."""
    x1, y1, x2, y2 = (float(value) for value in box)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _as_individual(item) -> Individual:
    """``(index, box)`` or an :class:`Individual` - whichever the caller has to hand."""
    if isinstance(item, Individual):
        return item
    index, box = item
    return Individual(int(index), tuple(float(value) for value in box))


def _ends(measurement) -> tuple:
    """``(p1, p2)`` of a measurement: the Bench's view item, or a plain pair.

    A view item is a dict with ``p1``/``p2``; a script or a test may pass the pair
    directly. Accepting both keeps the rule usable without the page's own shape (the same
    tolerance ``ui.model_pass`` shows for a pose it did not build).
    """
    if isinstance(measurement, dict):
        return tuple(measurement["p1"]), tuple(measurement["p2"])
    first, second = measurement
    return tuple(first), tuple(second)


def midpoint(p1, p2) -> tuple:
    """The point halfway between two landmarks, in source pixels."""
    return ((float(p1[0]) + float(p2[0])) / 2.0,
            (float(p1[1]) + float(p2[1])) / 2.0)


def contains(box, point) -> bool:
    """True when a source-pixel point lies inside ``box`` - the border counts as inside.

    The border is included because a landmark placed *on* the rectangle an operator drew
    around an animal is unambiguously about that animal.
    """
    x1, y1, x2, y2 = _normalised(box)
    x, y = float(point[0]), float(point[1])
    return x1 <= x <= x2 and y1 <= y <= y2


def individual_of(p1, p2, individuals) -> "Individual | None":
    """The individual a measurement between ``p1`` and ``p2`` belongs to, or ``None``.

    ``None`` is a real answer, not a failure: it says *this number belongs to no animal
    you have boxed*, and the page shows it that way instead of guessing.
    """
    where = midpoint(p1, p2)
    containing = [item for item in (_as_individual(entry) for entry in individuals)
                  if contains(item.box, where)]
    if not containing:
        return None
    return min(containing, key=lambda item: item.area)


def group(measurements, individuals) -> dict:
    """``{individual index or None: [measurement, ...]}``, each list in the order given.

    The ``None`` key holds the measurements that fall inside no box - they are kept and
    listed, never dropped and never attached to the nearest animal (law 8).
    """
    individuals = list(individuals)
    grouped = {}
    for measurement in measurements:
        p1, p2 = _ends(measurement)
        found = individual_of(p1, p2, individuals)
        grouped.setdefault(found.index if found is not None else None, []).append(measurement)
    return grouped
