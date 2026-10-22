"""Measuring the scale from a printed **dot** target photographed through an eyepiece.

`vision/lattice.py` measures grid *paper*, whose scale is the spacing of its printed
**lines**. A printed target can be built the other way round - a regular field of round
**dots**, which is what `assets/targets/` renders and what a distortion model wants
(`docs/calibration.md`: the dot grid fits a known distortion at **0.149 px rms**, better
than a single-weight line grid). This module is the estimator that reads such a target and
turns it into the same ``px_per_um`` the Calibrate screen already stores.

It is deliberately the *same shape* as `vision/lattice.py`, so the OPTICS surface can use
either without noticing which one answered:

1. **the field stop is found and masked as a circle** - `vision/lattice.field_stop`, called
   here rather than copied, because the phone-through-an-eyepiece frame is the same frame
   whether the target is lines or dots. The dark surround would otherwise be one huge blob
   of ink and the dots inside the disc would be the only ones left; eroding the disc by a
   margin keeps the stop's own hard ring out.
2. **the dots are found as round components.** Inside the paper, Otsu separates the printed
   dots from the sheet; connected components then keep the ones that *look like a dot* (a
   blob that fills ~pi/4 of its bounding box, not a long line, not a speck) and are of a
   consistent size. This is the step that makes the detector a *dot* detector: a
   photograph of ruled paper has no round blobs, and it is refused.
3. **the dots are grouped into a lattice, and the pitch is read off it.** The nearest
   neighbour of each dot gives the coarse pitch and the lattice's orientation (the two
   families of a square lattice reinforce under a doubled-angle mean); projecting the dots
   onto each axis and taking the robust median spacing of the resulting lines gives the
   scale in each direction. Rotating the frame does not change the scale, so no
   straightening pass is needed - the *measurement itself* carries the angle.
4. **the spread of the local spacing is reported as the drift.** The nearest-neighbour
   distance of the dots at the centre of the field against the dots at its edge is the
   position-dependent scale variation - the same quantity `vision/lattice.py` reads from
   its bands, measured without any model and without any fit.

Only numpy and OpenCV are used (OpenCV lazily): no new dependency, and nothing here needs
discorpy. The distortion *model* - coefficients and correction - is `docs/calibration.md`
and `scripts/calibration_probe.py`'s business for now.
"""

from dataclasses import dataclass, field as dataclass_field

import numpy as np

from daphnia.vision.grid import (
    GRID_PITCH_UM,
    PAPER_PITCH_TOLERANCE_PCT,
)
#: The circle fit, the grey view and Otsu are `vision/lattice`'s - the *same*
#: preprocessing this surface shares, imported rather than copied so the two estimators
#: cannot drift apart. `field_stop` is public; the other two are its documented helpers.
from daphnia.vision.lattice import (
    FIELD_MARGIN_PX,
    FieldStop,
    _grey,
    _otsu,
    field_stop,
)

#: Fewer printed dots than this in the whole field and there is no lattice to measure.
#: A 3x3 patch is the smallest a two-family grid can be, and 12 leaves room for the odd
#: dot the paper or the print lost.
MIN_POINTS = 12

#: A component smaller than this many pixels is a speck of print or sensor noise, not a dot.
MIN_DOT_AREA = 4.0

#: A round dot fills ~0.785 of its bounding box (pi/4). Anything from a half to most of it
#: is round enough; a line or a cross is far below, an L-shaped blob far above.
MIN_DOT_FILL = 0.45
MAX_DOT_FILL = 0.98

#: A dot's bounding box is square; this is how far from square it may still be (a long
#: thin component is a line, a printed margin, or two dots merged by a smear).
MAX_DOT_ASPECT = 1.8

#: A dot's area may differ from the field's median dot by this factor before it is dropped
#: as merged or split, not measured.
MIN_DOT_SIZE_RATIO = 0.25
MAX_DOT_SIZE_RATIO = 4.0

#: Plausible dot spacings in pixels - the same bounds `vision/grid.py` and
#: `vision/lattice.py` use, for the same reason: below them the print is finer than the
#: sensor resolves, above them too few dots fit in the field to measure anything.
MIN_PERIOD_PX = 6.0
MAX_PERIOD_PX = 500.0

#: The two families of a square dot lattice must agree this closely, or this is not square
#: dots and the scale would be ambiguous. Tighter than the line estimator's 3 %: a dot is
#: symmetric, so a real anisotropy is the target being oblique, not the print.
MAX_ANISOTROPY_PCT = 5.0

#: How far a dot's spacing may scatter (relative to the period) before the axis is not a
#: lattice. Ink spread and a hand-held target move a dot's centre by a fraction of the
#: pitch; a printed line every row does not.
MAX_SPACING_SCATTER = 0.15

#: Fewer than this many dots on the inner half (or the outer quarter) of the field and the
#: drift is not measured - it is left at zero, which the panel reports as "0.0 %" ... and
#: that is honest, because nothing was compared.
MIN_POINTS_FOR_DRIFT = 6


class PointsNotFound(ValueError):
    """The photo does not hold a measurable lattice of printed dots."""


@dataclass(frozen=True)
class PointGridEstimate:
    """A measured dot lattice: the scale, where it was measured, and how much it drifts."""

    period_px: float                    # the scale: pixels between printed dots
    angle_deg: float                    # the lattice's tilt in the *original* frame
    period_px_vertical: float = 0.0     # the second family's spacing, when it was read
    pitch_um: float = GRID_PITCH_UM
    field_stop: FieldStop = None
    coverage: float = 1.0               # fraction of the frame that is paper
    points: int = 0                     # how many printed points it rests on
    pattern: str = "dots"               # "dots" or "checkerboard" - what a point *is* here
    spread_pct: float = 0.0             # how far the local spacing moves across the field
    statistical_pct: float = 0.0        # error of the period itself
    threshold: float = None             # the binarisation level used; None = Otsu decided
    caveats: tuple = dataclass_field(default_factory=tuple)

    def px_per_um(self, pitch_um=None) -> float:
        """Pixels per micrometre, from this period and the printed dot spacing."""
        if self.period_px <= 0:
            raise ValueError("period_px must be > 0")
        return self.period_px / (self.pitch_um if pitch_um is None else pitch_um)

    def statistical_uncertainty_pct(self) -> float:
        """Error of the *centre* scale, from the dot-to-dot scatter of the period."""
        return float(self.statistical_pct)

    def uncertainty_pct(self, paper_tolerance_pct=PAPER_PITCH_TOLERANCE_PCT) -> float:
        """The scale's uncertainty: the measurement's own error plus the printed target's.

        Deliberately *not* including ``spread_pct``: that is the drift - a property of the
        optics at a place in the frame - not an uncertainty of the scale at the point it was
        measured. Same rule as `vision/lattice.LatticeEstimate.uncertainty_pct`.
        """
        return float(np.hypot(self.statistical_pct, paper_tolerance_pct))

    @property
    def mean_period_px(self) -> float:
        """``LatticeEstimate``'s name for the same number, so the panel need not care."""
        return self.period_px

    @property
    def reading_count(self) -> int:
        """How many *readings* the scale rests on - here, printed dots."""
        return int(self.points)

    @property
    def reading_label(self) -> str:
        """What one reading is called on the panel: a dot, or a checkerboard **cell**."""
        return "cells" if self.pattern == "checkerboard" else "dots"

    def warnings(self) -> list:
        """Sentences for the operator: what was measured, and what it means.

        The Calibrate screen reads the **fields** below, not this list (one fact, one home,
        `ui-calibrate-redesign.md` D5); it stays for a caller that wants the whole sentence.
        """
        notes = [f"{self.period_px:.3f} px between printed reference points, from "
                 f"{self.points} of them"]
        if self.threshold is not None:
            notes.append(f"binarisation at grey level {self.threshold:g} - set by hand, "
                         "not by Otsu")
        if self.spread_pct > 1.0:
            notes.append(
                f"the spacing varies {self.spread_pct:.1f}% across the field (measured from "
                "the same dots) - position-dependent, not an uncertainty of this number")
        if self.field_stop is not None:
            notes.append(f"field stop: centre ({self.field_stop.centre_x:.0f}, "
                         f"{self.field_stop.centre_y:.0f}), radius "
                         f"{self.field_stop.radius:.0f} px, {self.coverage * 100:.0f}% of "
                         "the frame")
        notes.extend(self.caveats)
        return notes


def _paper_mask(grey, stop, margin=FIELD_MARGIN_PX):
    """A boolean mask of the paper inside the field stop (or the whole frame).

    ``field_stop`` returns ``None`` for a photo of the sheet in air, and then the whole
    frame is paper - the same "``None`` is a real answer" contract `vision/lattice` follows.
    """
    if stop is None:
        return np.ones(grey.shape, dtype=bool)
    height, width = grey.shape
    yy, xx = np.mgrid[0:height, 0:width]
    disc = ((xx - stop.centre_x) ** 2 + (yy - stop.centre_y) ** 2
            <= max(1.0, stop.radius - margin) ** 2)
    return disc if disc.any() else np.ones(grey.shape, dtype=bool)


def _ink_mask(grey, mask, pattern, threshold=None):
    """The binary **ink** mask one detector cuts - the single place the cut is decided.

    Kept as one function so a *view* that draws the detection and the *detector* that measures
    from it can never disagree about which pixels are ink - the same single-owner rule the
    coordinate mappings follow (``mlContext.md`` law 1).

    ``"dots"`` cuts the raw frame at Otsu, or at ``threshold`` when the operator set one.
    ``"checkerboard"`` first corrects the non-uniform background (see :func:`_flatten`) and
    then cuts; a manual ``threshold`` is applied to the **raw** frame instead, because the
    operator sets a level they can *see* on the photo, and comparing it against a corrected
    image would silently mean something they never looked at.
    """
    if pattern == "checkerboard" and threshold is None:
        # The surround is filled with the *paper's* brightness (a high percentile), not the
        # field's median: a checkerboard's median is dragged down by the black cells, and
        # filling the surround with it would leave a dark ring for the low-pass to smear over
        # the outer cells.
        inside = grey[mask] if mask.any() else grey.reshape(-1)
        filled = np.where(mask, grey, float(np.percentile(inside, 90)))
        flat = _flatten(filled)
        values = flat[mask] if mask.any() else flat.reshape(-1)
        return (flat <= _otsu(values)) & mask
    values = grey[mask] if mask.any() else grey.reshape(-1)
    cut = _otsu(values) if threshold is None else float(threshold)
    return (grey <= cut) & mask


def _detect_dots(grey, mask, threshold=None):
    """The centroids of the printed dots inside ``mask``, as an ``(N, 2)`` array of pixels.

    Inside the paper, Otsu separates the dots from the sheet; the components that survive
    are the ones a *dot* makes - a round blob filling most of its (square) bounding box, of
    a size consistent with the field's other dots. Nothing about the detector is tuned to a
    particular print: the size test is *relative to this field's own median dot*, so a fine
    print and a coarse one both work.

    ``threshold`` is an optional **manual override**: an absolute grey level the operator
    sets when Otsu lands badly on a particular photo (discorpy's own escape hatch,
    ``prep.binarization(mat, ratio, thres=)``). ``None`` means "decide it here" - the default,
    and the one the caller must state if it overrides it, because a silently changed threshold
    is a silently changed measurement.
    """
    import cv2

    dark = _ink_mask(grey, mask, "dots", threshold)

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark.astype(np.uint8), connectivity=8)
    if count < 2:
        return np.zeros((0, 2), dtype=np.float64)

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    centres = centroids[1:].astype(np.float64)
    boxes = np.maximum(widths * heights, 1.0)
    fill = areas / boxes
    aspect = np.maximum(widths / np.maximum(heights, 1.0),
                        heights / np.maximum(widths, 1.0))

    round_enough = ((areas >= MIN_DOT_AREA) & (fill >= MIN_DOT_FILL)
                    & (fill <= MAX_DOT_FILL) & (aspect <= MAX_DOT_ASPECT))
    if round_enough.sum() < MIN_POINTS:
        return centres[round_enough]

    # A *consistent* size: an artifact - two dots merged by a smear, one dot split - is not
    # a dot, and keeping it would corrupt both the median spacing and the size statistics.
    median_area = float(np.median(areas[round_enough]))
    sized = (round_enough
             & (areas >= MIN_DOT_SIZE_RATIO * median_area)
             & (areas <= MAX_DOT_SIZE_RATIO * median_area))
    return centres[sized]


def _coarse_pitch(points):
    """Every dot's nearest-neighbour spacing *and vector*, and the field's median spacing.

    Returns ``(spacing, coarse, steps)``. ``steps`` is one vector per dot - the step to its
    **nearest** neighbour, which is always a lattice step (the diagonal of a square lattice
    is ``sqrt(2)`` times the pitch, so it is never the nearest). Only the nearest is kept:
    measured on the synthetic target, taking *every* neighbour within a factor of the pitch
    lets the diagonals in and the doubled-angle mean then answers -31 deg on a lattice that
    is tilted 8 deg.
    """
    deltas = points[:, None, :] - points[None, :, :]        # (N, N, 2)
    distance = np.hypot(deltas[..., 0], deltas[..., 1])
    np.fill_diagonal(distance, np.inf)
    nearest = distance.argmin(axis=1)
    spacing = distance[np.arange(len(points)), nearest]
    steps = deltas[np.arange(len(points)), nearest]
    return spacing, float(np.median(spacing)), steps


def _dominant_axis(steps):
    """The lattice's orientation in radians, from the directions of the nearest steps.

    A square lattice has two families of steps, at ``theta`` and ``theta + 90``. Doubling
    the angle maps both onto ``2 * theta`` (``2(theta + 90) = 2theta + 180`` is the same
    direction), so the doubled-angle mean reinforces instead of cancelling - no histogram,
    no peak-picking, and it is invariant to which family a given dot's neighbour lies on.
    """
    length = np.hypot(steps[:, 0], steps[:, 1])
    if not (length > 0).any():
        raise PointsNotFound("no two dots are close enough to be a lattice step")
    unit = steps / np.maximum(length, 1e-9)[:, None]
    doubled = 2.0 * np.arctan2(unit[:, 1], unit[:, 0])
    return 0.5 * float(np.arctan2(float(np.sin(doubled).mean()),
                                  float(np.cos(doubled).mean())))


def _merge(coordinates, coarse):
    """The centres of the lines the dots fall on, along one axis, as a sorted array.

    Dots sharing a coordinate (a whole column of the lattice) must not each count as a
    line: they are merged first, then the spacing is the distance between the merged
    centres - which is why the scale does not depend on how many dots sit on a line.
    """
    ordered = np.sort(np.asarray(coordinates, dtype=np.float64))
    tolerance = 0.5 * coarse
    centres = []
    start = 0
    for index in range(len(ordered) - 1):
        if ordered[index + 1] - ordered[index] > tolerance:
            centres.append(float(np.median(ordered[start:index + 1])))
            start = index + 1
    centres.append(float(np.median(ordered[start:])))
    return np.array(centres, dtype=np.float64)


def _axis_period(coordinates, coarse):
    """``(period, error)`` of one axis, by the robust median spacing of its merged lines.

    The spacings are filtered to those within 25 % of the running median - a missing or
    merged line doubles a spacing - then the median is recomputed on the inliers, exactly
    as `vision/lattice._band_reading` reads one band.
    """
    centres = _merge(coordinates, coarse)
    if len(centres) < 2:
        return None
    spacings = np.diff(centres)
    period = float(np.median(spacings))
    inliers = spacings
    for _ in range(3):
        inliers = spacings[np.abs(spacings - period) <= 0.25 * period]
        if len(inliers) < 2:
            return None
        period = float(np.median(inliers))
    if not (MIN_PERIOD_PX <= period <= MAX_PERIOD_PX):
        return None
    scatter = float(np.median(np.abs(inliers - period))) / period
    if scatter > MAX_SPACING_SCATTER:
        return None
    error = 1.4826 * float(np.median(np.abs(inliers - period))) / np.sqrt(len(inliers))
    return period, error


def _drift_pct(points, spacing):
    """How far the local spacing moves from the field's centre to its edge, in percent.

    The *nearest-neighbour* distance of the dots on the inner half against the dots on the
    outer quarter of the field - a place in the frame changing the scale, measured without
    a model. Returns 0.0 (never a fit) when either ring holds too few dots to compare.
    """
    centre = points.mean(axis=0)
    radius = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1])
    if radius.max() <= 0:
        return 0.0
    inner = spacing[radius <= 0.5 * radius.max()]
    outer = spacing[radius >= 0.75 * radius.max()]
    if len(inner) < MIN_POINTS_FOR_DRIFT or len(outer) < MIN_POINTS_FOR_DRIFT:
        return 0.0
    core = float(np.median(inner))
    if core <= 0:
        return 0.0
    return abs(float(np.median(outer)) - core) / core * 100.0


#: A black checkerboard cell is a *square*, but a **rotated** one fills less of its
#: axis-aligned bounding box than an upright one (measured here: a cell tilted 8 deg fills
#: ~0.8, and a 45 deg one would fill 0.5). So the bar is 0.5 - enough to reject a line, a
#: smear or a merged blob, which is all that matters in the checkerboard branch.
MIN_CELL_FILL = 0.5
MAX_CELL_ASPECT = 1.3

#: Largest ``area / (pi * r_enclosing^2)`` a black *cell* may have. This is what separates a
#: square from a round dot, and it does so whatever the cell's rotation: a disc fills its
#: enclosing circle (ratio 1.0), a square fills only ``2/pi`` of it (0.637) - measured here,
#: 0.64 for the synthetic cells against 1.0 for the dots. Without it a photo of the **dot**
#: target handed to the checkerboard detector is read as cells whose "side" is the dot's
#: diameter, and the scale comes out wrong rather than refused.
MAX_CELL_ROUNDNESS = 0.85

#: The black cells of a checkerboard sit **two** printed cells apart (a black cell neighbours
#: only white ones). So the lattice of their centres is measured at ``2 * cell``, and the
#: printed cell - the number the Calibrate screen's *Printed square* expects - is half of it.
#: The cell's own area gives the same square independently; the two must agree, which is what
#: stops a factor of two (the one error `vision/lattice.py` refuses to make) from creeping in.
CELLS_APART = 2.0


def _flatten(grey):
    """Correct the **non-uniform background** before thresholding, as discorpy's own flow does.

    A phone photo through an eyepiece is **vignetted** - the paper near the field stop is
    darker than the paper at the centre - so one global threshold cannot separate ink from
    paper across the whole field (measured on the synthetic checkerboard: Otsu lands *above*
    the paper's own median and the outer cells are lost, merging into one blob).

    This is discorpy's ``prep.normalization_fft(mat, sigma=20)`` (its documentation, *3.3.3
    Process a challenging X-ray target image*, is the reference for exactly this step): take
    a low-pass of the image and subtract it, so the only thing left is the *pattern* on a flat
    zero. The low-pass is a wide Gaussian rather than a true FFT - same effect, OpenCV is
    already a dependency and discorpy is not - and its width is several printed periods, so
    it carries the slow vignette and not the pattern.
    """
    import cv2

    sigma = max(4.0, 0.12 * min(grey.shape))
    background = cv2.GaussianBlur(grey, (0, 0), sigma)
    return (grey - background) + float(np.median(grey))


def _detect_cells(grey, mask, threshold=None):
    """The **black cells** of a printed checkerboard: their centres and their side, in pixels.

    A checkerboard has no round dot, and its corners are where a corner detector over-fires
    (measured on the synthetic render: 1882 responses for ~430 real corners, because the
    vignette and the blurred cell edges are corner-like). Its *cells* are unambiguous:
    each is a solid square. ``connectivity=4`` keeps the black cells apart - they touch only
    diagonally, so an 8-connected labelling would fuse the whole board into one blob. The
    illumination is flattened first (:func:`_flatten`), or the vignette eats the outer cells.

    Returns ``(centres, sides, angles)``: the centroids of the cells, their side lengths
    (``sqrt(area)``, exact for a square and independent of the cell's rotation or a missing
    neighbour) and their edge angle in degrees from a min-area rectangle, all filtered to
    cells of a consistent square size.

    ``threshold`` is an optional **manual override**, as in :func:`_detect_dots` - but with a
    deliberate difference, documented on :func:`_ink_mask`: it cuts the **raw** frame rather
    than the illumination-corrected one.
    """
    import cv2

    dark = _ink_mask(grey, mask, "checkerboard", threshold).astype(np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark, connectivity=4)
    if count < 2:
        empty = (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64),
                 np.zeros(0, dtype=np.float64))
        return empty

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    centres = centroids[1:].astype(np.float64)
    boxes = np.maximum(widths * heights, 1.0)
    fill = areas / boxes
    aspect = np.maximum(widths / np.maximum(heights, 1.0),
                        heights / np.maximum(widths, 1.0))

    kept = ((areas >= MIN_DOT_AREA) & (fill >= MIN_CELL_FILL)
            & (aspect <= MAX_CELL_ASPECT))
    if kept.sum() >= MIN_POINTS:
        median_area = float(np.median(areas[kept]))
        kept = (kept
                & (areas >= MIN_DOT_SIZE_RATIO * median_area)
                & (areas <= MAX_DOT_SIZE_RATIO * median_area))

    angles = np.full(len(areas), np.nan, dtype=np.float64)
    roundness = np.full(len(areas), np.nan, dtype=np.float64)
    for index in np.nonzero(kept)[0]:
        ys, xs = np.nonzero(labels == index + 1)
        if len(xs) < 3:
            continue
        points = np.column_stack([xs, ys]).astype(np.float32)
        rectangle = cv2.minAreaRect(points)
        angles[index] = float(rectangle[2]) % 90.0
        _centre, radius = cv2.minEnclosingCircle(points)
        if radius > 0:
            roundness[index] = float(areas[index]) / (np.pi * radius * radius)
    # A **square** cell, not a round dot (see ``MAX_CELL_ROUNDNESS``); NaN fails the test, so
    # a component too small to describe itself is dropped with the rest.
    square = kept & (roundness <= MAX_CELL_ROUNDNESS)
    return centres[square], np.sqrt(areas[square]), angles[square]


def _drift_from_values(points, values):
    """Centre-to-edge variation of a per-point quantity (a cell's side), in percent.

    The checkerboard analogue of :func:`_drift_pct`: the *cell side* is measured on every cell
    anyway, so the drift is simply how much it changes from the field's centre to its edge -
    no complete lattice is needed for it to work.
    """
    centre = points.mean(axis=0)
    radius = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1])
    if radius.max() <= 0:
        return 0.0
    inner = values[radius <= 0.5 * radius.max()]
    outer = values[radius >= 0.75 * radius.max()]
    if len(inner) < MIN_POINTS_FOR_DRIFT or len(outer) < MIN_POINTS_FOR_DRIFT:
        return 0.0
    core = float(np.median(inner))
    if core <= 0:
        return 0.0
    return abs(float(np.median(outer)) - core) / core * 100.0


def _estimate_checkerboard(centres, sides, angles, stop, mask, pitch_um, threshold=None):
    """The scale of a checkerboard from its cells, whose **side** *is* the printed cell.

    A dot lattice's scale is a distance between dots in a *complete* array; a checkerboard's
    is the side of one cell - measured on every individual cell (``sqrt(area)``), so a missing
    or clipped neighbour cannot bias it. The cell *centres* confirm the board is a square
    lattice: projected onto the cells' own edges they must fall on the same pitch,
    ``CELLS_APART`` cells apart, or the target is refused.
    """
    side = float(np.median(sides))
    finite = angles[np.isfinite(angles)]
    tilt = float(np.median(finite)) if finite.size else 0.0
    theta = np.radians(tilt)
    cos_a, sin_a = float(np.cos(theta)), float(np.sin(theta))
    along = centres[:, 0] * cos_a + centres[:, 1] * sin_a
    across = -centres[:, 0] * sin_a + centres[:, 1] * cos_a
    coarse = CELLS_APART * side

    # The cell *centres* are only *evidence* here: when the whole board is legible they lie on
    # a lattice `CELLS_APART` cells apart, which confirms the target is square. When the
    # vignette leaves only part of the board as clean cells (measured: 63 of ~200 on this
    # render), the surviving centres fall on a *sparser* sublattice and their spacing is a
    # multiple of the pitch - so a missing centre must **not** be read as a wrong scale. The
    # scale is the cell's own side, which every cell gives independently; a lattice that is
    # readable must merely agree with it.
    vertical_reading = _axis_period(across, coarse)
    caveats = []
    for name, reading in (("one", _axis_period(along, coarse)),
                          ("two", vertical_reading)):
        if reading is None:
            continue
        cell = reading[0] / CELLS_APART
        if abs(cell - side) > 0.15 * side:
            caveats.append(
                f"the cell centres of family {name} sit {cell:.1f} px apart where the cell "
                f"side is {side:.1f} px - part of the board may be unreadable (out of focus, "
                "or beyond the field stop)")

    period = side
    error = 1.4826 * float(np.median(np.abs(sides - side))) / np.sqrt(len(sides))
    spread_pct = _drift_from_values(centres, sides)
    if spread_pct > 1.0:
        caveats.append(
            f"the cells change size {spread_pct:.1f}% from the centre of the field to its "
            "edge - that is the drift, measured from the cells themselves, and it is larger "
            "than this scale's own uncertainty")
    coverage = float(mask.mean()) if stop is not None else 1.0
    return PointGridEstimate(
        period_px=period,
        angle_deg=tilt % 180.0,
        period_px_vertical=(vertical_reading[0] / CELLS_APART
                            if vertical_reading is not None else side),
        pitch_um=pitch_um,
        field_stop=stop,
        coverage=coverage,
        points=int(len(centres)),
        pattern="checkerboard",
        spread_pct=spread_pct,
        statistical_pct=error / period * 100.0,
        threshold=(None if threshold is None else float(threshold)),
        caveats=tuple(caveats),
    )


#: Which detector reads which printed pattern. Both return a point lattice; the caller
#: (the Calibrate screen) picks the one that matches the sheet in front of the lens.
PATTERNS = ("dots", "checkerboard")


def reference_points(frame, pattern="dots", threshold=None):
    """The reference points the detector found, as ``(N, 2)`` **source** pixels.

    :func:`measure_points` summarises these; a *view* that draws the detection (the Calibrate
    screen's results panel) must draw the very points the measurement used, so they are
    returned from here rather than recomputed at the call site - single owner, again
    (``mlContext.md`` law 1). An empty array is a real answer: nothing was found, which is
    exactly what a view has to show as "nothing".

    Raises:
        ValueError: ``pattern`` is not a known one, or ``frame`` is not an image.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}, got {pattern!r}")
    grey = _grey(frame)
    mask = _paper_mask(grey, field_stop(grey))
    if pattern == "checkerboard":
        centres, _sides, _angles = _detect_cells(grey, mask, threshold)
        return centres
    return _detect_dots(grey, mask, threshold)


def measure_points(frame, pitch_um=GRID_PITCH_UM, pattern="dots", threshold=None):
    """Measure the scale in one photo of a printed point target, wherever it sits.

    ``pattern`` selects what a *point* is on the sheet: ``"dots"`` (a field of round dots,
    the default), measured as a lattice of centres, or ``"checkerboard"``, measured from the
    **cells** whose side *is* the printed cell. Either way the result is the same
    ``px_per_um`` the Calibrate screen stores.

    ``threshold`` is an optional **manual binarisation level** for the ink/paper split, an
    absolute grey value in the frame's own units (``None`` = Otsu decides, the default). It is
    the discorpy-style escape hatch for a photo where the automatic split lands badly - *not* a
    cure for the eyepiece's **vignette**: the paper's brightness falls across the field, so the
    ink/paper boundary is a slope and no single number is right everywhere. That is what the
    illumination correction inside the detectors is for. The value used is reported back on the
    estimate (:attr:`PointGridEstimate.threshold`) so a number nobody chose cannot pass as the
    default.

    Refuses - with the reason - when the frame is not a square lattice of reference points:
    too few found, one direction only, or the two families disagreeing. A wrong scale is
    worse than no scale, which is the rule `vision/grid.py` and `vision/lattice.py` already
    follow.

    Raises:
        PointsNotFound: the photo cannot give a scale.
        ValueError: ``frame`` is not an image, or ``pattern`` is not a known one.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}, got {pattern!r}")
    grey = _grey(frame)
    stop = field_stop(grey)
    mask = _paper_mask(grey, stop)

    if pattern == "checkerboard":
        # A checkerboard is measured from its *cells*, not from a complete lattice of dots:
        # the cell's own side is the printed cell (`_estimate_checkerboard`).
        centres, sides, angles = _detect_cells(grey, mask, threshold)
        if len(centres) < MIN_POINTS:
            raise PointsNotFound(
                f"only {len(centres)} black checker cell(s) found in the field "
                f"({MIN_POINTS} needed) - is the checkerboard in view, in focus, and filling "
                "the field?")
        return _estimate_checkerboard(centres, sides, angles, stop, mask, pitch_um,
                                      threshold=threshold)

    points = _detect_dots(grey, mask, threshold)
    if len(points) < MIN_POINTS:
        raise PointsNotFound(
            f"only {len(points)} printed reference point(s) found in the field "
            f"({MIN_POINTS} needed) - is the point target in view, in focus, and filling "
            "the field?")

    spacing, coarse, steps = _coarse_pitch(points)
    if not (MIN_PERIOD_PX <= coarse <= MAX_PERIOD_PX):
        raise PointsNotFound(
            f"the reference points are {coarse:.1f} px apart - finer or coarser than this "
            "detector can measure (is this the point target, and in focus?)")

    axis = _dominant_axis(steps)
    cos_a, sin_a = float(np.cos(axis)), float(np.sin(axis))
    along = points[:, 0] * cos_a + points[:, 1] * sin_a
    across = -points[:, 0] * sin_a + points[:, 1] * cos_a

    horizontal = _axis_period(along, coarse)
    vertical = _axis_period(across, coarse)
    if horizontal is None or vertical is None:
        raise PointsNotFound(
            "periodicity found in only one direction - a point *grid* has two families of "
            "spacing; aim at the printed target, not a single row of marks")

    mean_period = (horizontal[0] + vertical[0]) / 2.0
    anisotropy = abs(horizontal[0] - vertical[0]) / mean_period * 100.0
    if anisotropy > MAX_ANISOTROPY_PCT:
        raise PointsNotFound(
            f"the two families of reference points are {anisotropy:.1f}% apart in spacing "
            f"({horizontal[0]:.2f} px against {vertical[0]:.2f} px) - that is not a "
            "square point grid (lay the target flat and perpendicular to the lens)")

    period = float(np.median([horizontal[0], vertical[0]]))
    error = float(np.max([horizontal[1], vertical[1]]))
    caveats = []
    spread_pct = _drift_pct(points, spacing)
    if spread_pct > 1.0:
        caveats.append(
            f"the local spacing moves {spread_pct:.1f}% across the field - that is the "
            "drift, measured from the same dots, and it is larger than this scale's own "
            "uncertainty")
    coverage = float(mask.mean()) if stop is not None else 1.0
    return PointGridEstimate(
        period_px=period,
        angle_deg=float(np.degrees(axis) % 180.0),
        period_px_vertical=vertical[0],
        pitch_um=pitch_um,
        field_stop=stop,
        coverage=coverage,
        points=int(len(points)),
        pattern=pattern,
        spread_pct=spread_pct,
        statistical_pct=error / period * 100.0,
        threshold=(None if threshold is None else float(threshold)),
        caveats=tuple(caveats),
    )
