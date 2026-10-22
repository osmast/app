"""Well states: what one well of a plate holds, and what that means on screen.

The plate page is a loop the operator drives **by hand**: move the plate under the camera,
take a burst, watch the well light up, step to the next well. That loop needs one
vocabulary which the screen, the tests and the (future) measuring half all read from the
same place - so the states and the arithmetic behind the plate live here, Qt-free, exactly
like the workflow gates of :mod:`daphnia.workflow` and the geometry of
:mod:`daphnia.plates`.

Three states, in the order a well climbs them::

    pending     nothing captured yet - the well is dark
    captured    at least one frame is on disk; the well is **done for now**. The sharpest
                frame of the well is its image (``media_assets.is_primary``), so one well
                has one picture without anybody having to pick it.
    measured    at least one measurement exists for the well. Today that means another
                surface annotated it; this is the rung the end goal lands on - a size per
                individual in the well, from the pose model or a manual line.

``kept`` is deliberately **not** a state. How many frames a human kept is a *count* the
plate page shows about the well (the old Select step's decision); making it a rung of the
ladder would claim a reviewed well is a different kind of captured, which it is not.

The counts come from ``queries.plate_well_counts``; nothing here touches the database, a
widget or the file system.
"""

from dataclasses import dataclass

from daphnia import plates

#: The well has no image yet.
PENDING = "pending"

#: The well has at least one frame on disk (its sharpest frame is its image).
CAPTURED = "captured"

#: The well carries at least one measurement - the end goal's state.
MEASURED = "measured"

#: Every state, weakest first. The plate page's legend is drawn in this order.
STATES = (PENDING, CAPTURED, MEASURED)

#: How a state is phrased for an operator.
STATE_LABELS = {
    PENDING: "Not captured",
    CAPTURED: "Captured",
    MEASURED: "Measured",
}

#: What one state promises about the well, in the operator's own terms.
STATE_HINTS = {
    PENDING: "No image of this well yet - it is the next one to take.",
    CAPTURED: "The well has an image; the sharpest frame of the burst is its picture.",
    MEASURED: "The well has an image and at least one measurement.",
}

#: Wells per page a plate may be split into: 0 means the whole plate.
#: A page never splits a row (12 wells of a 96-well plate are one row of it), so a
#: request is rounded to whole rows - see :func:`rows_per_page`.
PAGE_SIZES = (0, 6, 12, 24, 48)

#: How a page size is phrased in a combo box (a page size is a count of wells).
PAGE_SIZE_LABELS = {0: "Whole plate"}


def _count(row, name) -> int:
    """One count of a ``plate_well_counts`` row (or of a plain mapping), default 0.

    A missing column is 0 and not an error: a well with no row at all *is* an empty well,
    which is the honest reading of "nothing was captured here".
    """
    if row is None:
        return 0
    try:
        value = row[name]
    except (KeyError, IndexError):
        return 0
    return int(value or 0)


def state_of(counts=None) -> str:
    """The state a well's counts mean: the strongest rung they reach.

    ``counts`` is one row of :func:`daphnia.db.queries.plate_well_counts` (or a mapping
    with the same keys); ``None`` means the well is untouched.
    """
    if _count(counts, "measured"):
        return MEASURED
    if _count(counts, "frames"):
        return CAPTURED
    return PENDING


@dataclass(frozen=True)
class WellProgress:
    """What one well holds, as the plate page reads it.

    Counts, never a verdict: the state is *derived* (:attr:`state`), so a well that gains
    a measurement later climbs without anything in the database being rewritten.
    """

    well_label: str
    frames: int = 0
    bursts: int = 0
    kept: int = 0
    measured: int = 0

    @property
    def state(self) -> str:
        """One of :data:`PENDING`, :data:`CAPTURED`, :data:`MEASURED`."""
        if self.measured:
            return MEASURED
        return CAPTURED if self.frames else PENDING

    @property
    def is_done(self) -> bool:
        """True when the well has an image - "this well is done" for the capture loop.

        The **end goal raises this bar**: a run is finished when every well carries a
        *size* per individual, which is ``state == MEASURED``. Until the measuring half
        exists, "an image of every well" is the reachable definition, and the plate page
        says so rather than pretending a captured well is a measured one.
        """
        return self.frames > 0

    def describe(self) -> str:
        """One line for the panel and the tile's tooltip, counts and all."""
        if not self.frames:
            return f"{self.well_label} - {STATE_LABELS[PENDING]}"
        parts = [f"{self.frames} frame(s) in {self.bursts} burst(s)"]
        if self.kept:
            parts.append(f"{self.kept} kept")
        if self.measured:
            parts.append(f"{self.measured} measurement(s)")
        return f"{self.well_label} - {STATE_LABELS[self.state]}: " + ", ".join(parts)


def well_progress(well_label, counts=None) -> WellProgress:
    """One well's record from a counts row; ``None`` (or a missing key) is an empty well."""
    return WellProgress(
        well_label=well_label,
        frames=_count(counts, "frames"),
        bursts=_count(counts, "bursts"),
        kept=_count(counts, "kept"),
        measured=_count(counts, "measured"),
    )


def progress(fmt, counts=None) -> tuple:
    """One :class:`WellProgress` per well of a plate format, in reading order.

    Every well of the format appears, whether or not anything was captured on it: the
    plate page's whole point is showing the wells that are still **empty**, and a database
    query alone can never do that (it only knows the wells that have rows).
    """
    rows = counts or {}
    return tuple(well_progress(label, rows.get(label))
                 for label in plates.well_labels(fmt))


def by_label(progress_rows) -> dict:
    """``{well_label: WellProgress}`` for a plate (or for the page of one on screen)."""
    return {row.well_label: row for row in progress_rows}


def summary(progress_rows) -> dict:
    """The plate's totals: wells, done, measured, and how many wells are in each state."""
    states = {state: 0 for state in STATES}
    done = measured = 0
    for row in progress_rows:
        states[row.state] += 1
        done += 1 if row.is_done else 0
        measured += 1 if row.measured else 0
    return {"wells": len(progress_rows), "done": done, "measured": measured,
            "by_state": states}


def rows_per_page(fmt, per_page=0) -> int:
    """How many plate rows one page shows.

    A page never splits a row, because the plate is read row by row and half a row on
    screen would hide which wells it came from. So a request is rounded to whole rows:
    12 wells of a 96-well plate are one row, 24 are two - and a request smaller than one
    row is one row, because the column count is the smallest page that is still a plate.
    """
    rows, columns = plates.rows_columns(fmt)
    if not per_page or per_page <= 0:
        return rows
    if per_page >= rows * columns:
        return rows
    return max(1, min(rows, int(per_page) // columns))


def pages(fmt, per_page=0) -> tuple:
    """The plate's wells split into pages of whole rows, in reading order.

    ``per_page`` is how many wells one page shows; ``0`` (the default) is the whole plate
    on a single page. The result is a tuple of tuples of well labels - what the plate
    widget is handed, one page at a time.
    """
    labels = plates.well_labels(fmt)
    columns = plates.rows_columns(fmt)[1]
    step = rows_per_page(fmt, per_page) * columns
    return tuple(tuple(labels[at:at + step]) for at in range(0, len(labels), step))


def page_index(plate_pages, well_label) -> int:
    """The index of the page holding ``well_label``; ``0`` when it is on none of them.

    A well that belongs to no page is a data error (a label from another plate format),
    and the first page is the one answer that still shows a plate instead of an empty
    grid - the page reports the mismatch itself.
    """
    for index, page in enumerate(plate_pages):
        if well_label in page:
            return index
    return 0


def next_pending(progress_rows, after=None) -> str:
    """The first well after ``after`` that has **no image yet**, or ``None``.

    Reading order, wrapping at the end of the plate: the loop is "walk the plate, take a
    burst on each well, move the plate on", so after the last well the useful question is
    "which one did I miss", never "the run is over". ``None`` is therefore the honest
    *finished* answer - every well has an image - and the page says exactly that.
    """
    rows = by_label(progress_rows)
    labels = list(rows)
    if not labels:
        return None
    start = labels.index(after) + 1 if after in labels else 0
    for label in labels[start:] + labels[:start]:
        if not rows[label].is_done:
            return label
    return None
