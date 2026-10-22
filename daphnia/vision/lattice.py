"""Measuring the scale from grid paper photographed through an eyepiece.

`vision/grid.py` measures a clean photo of grid paper: a 2D FFT finds the lattice, and a
projection plus a zero-padded DFT refines the period. Measured on this project's own
photos (2026-10-19) that detector **refuses all nine** of them - a phone photo through an
eyepiece is pale, tilted, shaded, and framed by a hard black field stop whose edge is
broadband energy. This module is the estimator the OPTICS surface uses (`SR-OPTICS-035`,
`SR-OPTICS-036`), and it measures those photos in four steps:

1. **the field stop is found and masked as a circle.** Otsu separates the paper from the
   surround (measured on the project's photo: threshold 100.4 against paper at 143-210 and
   surround at <= 60); the largest bright region is filled and a circle is fitted to its
   boundary, *ignoring the frame border* because the disc is cut off by the frame at the
   top. The disc is eroded by a margin so the stop's own ring - a hard edge that any
   profile crossing it turns into a spurious line - stays out; the surround is then filled
   with the paper's median so no edge remains in the picture at all.
2. **the lattice is straightened.** The tilt is found by projecting the frame at candidate
   angles and keeping the one whose profile is the peakiest, then the frame is rotated by
   it. Rotating does not change the scale, and it removes a source of error: a tilted grid
   makes a *narrow band* useless as a profile (a 24-px band smears a 9 deg tilt over ~4 px).
3. **the period is read off narrow bands, and the spread across them is reported.** Several
   bands give several *local* periods; their median is the scale, and their spread is the
   **position-dependent scale variation** - the distortion - measured without any model and
   without any fit. Measured on the project's photo: 26.0-28.0 px per cell, i.e. +-3.8 %, up
   to ~5.7 % from the centre to the edge of the field, which is 76-114 um on a 2 mm animal -
   seven times the 0.5 % tolerance of the print itself.

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

#: Side of a band the period is measured in, in pixels. Narrow on purpose: a wide band
#: smears a tilted grid, and a narrow one is a *local* measurement, which is what makes the
#: spread across bands mean something.
BAND_PX = 24

#: Bands are placed on a grid this far apart, in pixels.
BAND_STEP_PX = 60

#: Fewer bands than this in *each* direction and there is no scale to report. Three, because
#: a circular field narrows the bands that fit: on this rig's photo the paper spans x 156-1044
#: of a 1280-px frame, so a row band every 60 px leaves ~15 candidates of which a handful sit
#: wholly on paper.
MIN_BANDS = 3

#: Fewer line spacings than this on a band and the profile says nothing.
MIN_LINES = 5

#: How far a band's line spacings may scatter (relative to their median) before the band is
#: not a lattice. Measured on this project's photo the real spacings agree to ~0.03 of the
#: period, so 0.15 leaves room for a print's own irregularity and still rejects lines that
#: merely exist (a ruler's graduations).
MAX_SPACING_SCATTER = 0.15

#: How much of the band's profile the period must *explain* (its own correlation at that
#: lag). This is the test that separates a lattice from dips that merely exist: measured on
#: this project's photo the lattice correlates 0.67-0.83, a field of pure noise ~0.1-0.2.
MIN_PERIODICITY = 0.5

#: A line must be this many profile standard deviations below the band's median. Measured
#: on this project's photo: the printed lines are 2.4 sd, a contrast-free field is not.
MIN_DEPTH_SIGMA = 1.5

#: A dip must be at least this fraction of the band's *deepest* line to count as a line.
#: Relative to the deepest, not to the spread: the paper's heavy every-fifth line is much
#: darker than its fine ones, and a spread-based threshold either catches only the heavy
#: lines (too high) or the noise (too low). Measured on this project's photo the fine lines
#: are ~40 % of the deepest, and the noise is a few percent.
DIP_DEPTH_FRACTION = 0.25

#: Blur applied to a band's profile before its dips are found, in pixels. A pale print's
#: lines split into two dips 4-12 px apart (measured), which halves the apparent spacing; a
#: 3-px blur merges the shoulders and cannot touch a lattice pitch of tens of pixels.
SMOOTH_SIGMA_PX = 2.0

#: Plausible cell spacings in pixels - the same bounds `vision/grid.py` uses, for the same
#: reason: below them the print is finer than the sensor resolves, above them too few cells
#: fit in the field to measure anything.
MIN_PERIOD_PX = 6.0
MAX_PERIOD_PX = 500.0

#: The two families of a square grid must agree this closely, or this is not square grid
#: paper and the scale would be ambiguous.
MAX_ANISOTROPY_PCT = 3.0

#: How far a band may disagree with the field's consensus period before it is rejected
#: rather than averaged in. 25 % keeps a real distortion (measured here: ~4-6 % across the
#: field) and rejects a band that locked onto half or twice the pitch.
CONSENSUS_TOLERANCE = 0.25

#: A field stop smaller than this many pixels cannot hold a lattice worth measuring.
MIN_FIELD_PX = 200

#: The disc is eroded by this much before it is used: the stop's ring is a hard edge.
FIELD_MARGIN_PX = 16

#: Angles tried when straightening, in degrees, and the refinement step.
SEARCH_DEG = 15.0
COARSE_STEP_DEG = 0.5
FINE_STEP_DEG = 0.1


class LatticeNotFound(ValueError):
    """The photo does not hold a measurable grid, even after preparation."""


@dataclass(frozen=True)
class FieldStop:
    """The circular paper area inside an eyepiece's field stop, in image pixels."""

    centre_x: float
    centre_y: float
    radius: float


@dataclass(frozen=True)
class LatticeEstimate:
    """A measured lattice: the scale, where it was measured, and how much it drifts."""

    period_px: float                    # the scale: pixels per printed cell
    angle_deg: float                    # the lattice's tilt in the *original* frame
    pitch_um: float = GRID_PITCH_UM
    field_stop: FieldStop = None
    coverage: float = 1.0               # fraction of the frame that is paper
    bands: int = 0                      # how many local measurements it rests on
    spread_pct: float = 0.0             # how far the local scale moves across the field
    statistical_pct: float = 0.0        # error of the period itself
    caveats: tuple = dataclass_field(default_factory=tuple)

    def px_per_um(self, pitch_um=None) -> float:
        """Pixels per micrometre, from this period and the printed cell size."""
        if self.period_px <= 0:
            raise ValueError("period_px must be > 0")
        return self.period_px / (self.pitch_um if pitch_um is None else pitch_um)

    def statistical_uncertainty_pct(self) -> float:
        """Error of the *centre* scale, from the band-to-band scatter of the period."""
        return float(self.statistical_pct)

    def uncertainty_pct(self, paper_tolerance_pct=PAPER_PITCH_TOLERANCE_PCT) -> float:
        """The scale's uncertainty: the fit's own error plus the printed paper's.

        Deliberately *not* including ``spread_pct``: that is the distortion - a property of
        the optics at a place in the frame - not an uncertainty of the scale at the point it
        was measured. One number for both is how "keep specimens central" became advice
        nobody could check.
        """
        return float(np.hypot(self.statistical_pct, paper_tolerance_pct))

    @property
    def mean_period_px(self) -> float:
        """``GridEstimate``'s name for the same number, so the panel need not care."""
        return self.period_px

    @property
    def reading_count(self) -> int:
        """How many *readings* the scale rests on - here, the bands of the field."""
        return int(self.bands)

    @property
    def reading_label(self) -> str:
        """What one reading is called on the panel - a band of the printed lines.

        ``PointGridEstimate`` answers ``"dots"`` for the same property, which is how the
        Calibrate screen names the right thing whichever estimator answered (`SR-OPTICS-016`
        covers both targets).
        """
        return "bands"

    def warnings(self) -> list:
        """Sentences for the operator: what was measured, and what it means.

        Named as ``GridEstimate.warnings`` is, for the same reason as
        :attr:`mean_period_px` - one panel, two estimators. The Calibrate screen no longer
        prints this list: it reads the **fields** above, one fact per home, because a
        paragraph in a status line is where a fact goes to be lost (`ui-calibrate-redesign.md`
        D5). The list stays for a caller that wants the whole sentence at once.
        """
        notes = [f"{self.period_px:.3f} px per printed cell, from {self.bands} bands of "
                 "the field"]
        if self.spread_pct > 1.0:
            notes.append(
                f"the scale varies {self.spread_pct:.1f}% across the field (measured from "
                "the same bands) - position-dependent, not an uncertainty of this number")
        if self.field_stop is not None:
            notes.append(f"field stop: centre ({self.field_stop.centre_x:.0f}, "
                         f"{self.field_stop.centre_y:.0f}), radius "
                         f"{self.field_stop.radius:.0f} px, {self.coverage * 100:.0f}% of "
                         "the frame")
        notes.extend(self.caveats)
        return notes


def _grey(frame):
    """A 2D float view of a frame, whatever it is (colour, grey, or already float)."""
    image = np.asarray(frame)
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D or 3D image, got shape {image.shape}")
    return image.astype(np.float32)


def _otsu(grey):
    """Otsu's threshold, over the observed range only (no cv2, no scikit-image).

    The images here are not 0..255 (the app's frames are, a test's may not be), and a
    histogram over the *observed* range keeps the split where the two populations are.
    """
    values = grey.reshape(-1)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return low
    counts, edges = np.histogram(values, bins=256, range=(low, high))
    centres = (edges[:-1] + edges[1:]) / 2.0
    weights = counts.astype(np.float64)
    total = weights.sum()
    cumulative = np.cumsum(weights)
    cumulative_mean = np.cumsum(weights * centres)
    mean_total = cumulative_mean[-1] / total
    denominator = cumulative * (total - cumulative)
    between = np.where(denominator > 0,
                       (mean_total * cumulative - cumulative_mean) ** 2
                       / np.where(denominator > 0, denominator, 1.0), 0.0)
    return float(centres[int(np.argmax(between))])


def _largest_component(binary):
    """The largest 8-connected region of a binary mask (OpenCV, lazily)."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    if count < 2:
        return np.zeros(binary.shape, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def _fill_holes(mask):
    """Fill the holes of a mask by flood-filling the background from the border."""
    import cv2

    height, width = mask.shape
    background = (~mask).astype(np.uint8)
    flooded = background.copy()
    cv2.floodFill(flooded, np.zeros((height + 2, width + 2), np.uint8), (0, 0), 2)
    holes = (background == 1) & (flooded != 2)
    return mask | holes


def _fit_circle(mask):
    """Least-squares circle through a mask's boundary, ignoring the frame border.

    The border points must go: a field stop is often cut off by the frame, and a flat
    clipped edge drags a circle fit inwards - measured on this project's photo, that is a
    **60 px** error on the radius, which then fills over real paper along the top.
    """
    import cv2

    height, width = mask.shape
    boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    rows, columns = np.nonzero(boundary)
    keep = ((columns > 3) & (columns < width - 4) & (rows > 3) & (rows < height - 4))
    if int(keep.sum()) < 16:
        return None
    points = np.stack([columns[keep], rows[keep]], axis=1).astype(np.float64)
    matrix = np.stack([points[:, 0], points[:, 1], np.ones(len(points))], axis=1)
    solution, *_ = np.linalg.lstsq(matrix, (points ** 2).sum(axis=1), rcond=None)
    centre_x, centre_y = solution[0] / 2.0, solution[1] / 2.0
    radius = float(np.sqrt(solution[2] + centre_x ** 2 + centre_y ** 2))
    if not np.isfinite(radius) or radius <= MIN_FIELD_PX / 2.0:
        return None
    return FieldStop(float(centre_x), float(centre_y), radius)


def field_stop(grey):
    """The paper's circular field stop in a photo, or ``None`` when there is not one.

    ``None`` is a real answer: a photo of the sheet in air has no stop, and the caller then
    measures the whole frame rather than an invented circle.
    """
    threshold = _otsu(grey)
    bright = grey > threshold
    if bright.mean() > 0.98 or bright.mean() < 0.02:
        return None
    biggest = _largest_component(bright)
    if not biggest.any():
        return None
    return _fit_circle(_fill_holes(biggest))


def _prepare(frame, margin=FIELD_MARGIN_PX):
    """Mask the field stop, fill the surround, and straighten the lattice.

    Returns ``(grey, mask, field_stop, angle_deg)``. The angle is the rotation that was
    applied, i.e. the lattice's tilt *in the original frame* - which is what an overlay has
    to draw.
    """
    import cv2

    grey = _grey(frame)
    stop = field_stop(grey)
    mask = np.ones(grey.shape, dtype=bool)
    if stop is not None:
        height, width = grey.shape
        yy, xx = np.mgrid[0:height, 0:width]
        disc = ((xx - stop.centre_x) ** 2 + (yy - stop.centre_y) ** 2
                <= (stop.radius - margin) ** 2)
        if disc.any():
            mask = disc
            grey = np.where(mask, grey, float(np.median(grey[mask]))).astype(np.float32)
    angle = _straighten_angle(grey)
    if abs(angle) > 0.05:
        grey, mask = _rotated(grey, mask, angle)
    else:
        # A lattice that needed no straightening is reported as exactly square: the page
        # draws this angle over the photo, and 1e-16 deg of floating point is not a tilt
        # (a test that asserts "the overlay is straight" is right to insist on 0.0).
        angle = 0.0
    return grey.astype(np.float32), mask, stop, float(angle)


def _rotated(grey, mask, angle_deg):
    """Rotate a frame and its mask by ``angle_deg`` about the frame's centre."""
    import cv2

    height, width = grey.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(grey, matrix, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    rotated_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (width, height),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return rotated, rotated_mask


def _peakiness(profile):
    """How strongly a projection profile oscillates (a lattice makes it oscillate)."""
    return float(np.var(np.diff(profile, 2)))


def _straighten_angle(grey, limit=SEARCH_DEG, coarse=COARSE_STEP_DEG,
                      fine=FINE_STEP_DEG):
    """The rotation (degrees) that makes the lattice's lines horizontal.

    Found by *doing it*: rotate the frame by each candidate angle and keep the one whose
    row profile is the peakiest. No sign convention to get wrong - the search is its own
    check, which is why a flipped or rotated photo needs no special case. Run on a
    downscaled copy: the angle does not need 12 megapixels.
    """
    import cv2

    height, width = grey.shape
    scale = min(1.0, 640.0 / float(max(height, width)))
    small = grey if scale >= 1.0 else cv2.resize(
        grey, (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA)
    small_height, small_width = small.shape
    centre = (small_width / 2.0, small_height / 2.0)

    def score(angle):
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(small, matrix, (small_width, small_height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return _peakiness(rotated.mean(axis=1))

    candidates = np.arange(-limit, limit + 1e-9, coarse)
    best = float(candidates[int(np.argmax([score(a) for a in candidates]))])
    refined = np.arange(best - coarse, best + coarse + 1e-9, fine)
    best = float(refined[int(np.argmax([score(a) for a in refined]))])
    return best


def _band_reading(profile):
    """``(period, error, consistency)`` from one band, by the spacing of its dark lines.

    The period is the **robust median spacing of the dips** in the band's profile: the
    spacings are first filtered to those within 25 % of the running median (a missed line
    doubles a spacing, and a line whose profile splits into two dips halves one), then the
    median is recomputed on the inliers. ``consistency`` is how tightly those inliers agree,
    and a band whose spacings scatter by more than ``MAX_SPACING_SCATTER`` is **not used**:
    that is what separates "a lattice" from "some lines" (a ruler's graduations have spacing,
    but no agreement under this test and no second family at all).

    Tried first and rejected on a measurement: the period as the *first peak of the
    autocorrelation*. On this project's photo that answers **13.4 px** where the dips are
    **26-27 px** apart, because each printed line's own micro-structure repeats at half the
    lattice pitch - an autocorrelation cannot tell "a repetition" from "the shape of one
    line", and a factor of two on the scale is the one error this module must never make.
    """
    smoothed = _smooth(np.asarray(profile, dtype=np.float64))
    centres = _line_centres(smoothed)
    if len(centres) < MIN_LINES + 1:
        return None
    spacings = np.diff(centres)
    period = float(np.median(spacings))
    if period <= 0:
        return None
    inliers = spacings
    for _ in range(3):
        inliers = spacings[np.abs(spacings - period) <= 0.25 * period]
        if len(inliers) < MIN_LINES:
            return None
        period = float(np.median(inliers))
    if not (MIN_PERIOD_PX <= period <= MAX_PERIOD_PX):
        # A plausible cell spacing, or nothing. Without this a field of *noise* gives dips
        # every few pixels with a tight scatter and passes every other test (measured: the
        # flat-field refusal below failed until this bound was applied).
        return None
    scatter = float(np.median(np.abs(inliers - period))) / period
    if scatter > MAX_SPACING_SCATTER:
        return None
    if _periodicity(smoothed, period) < MIN_PERIODICITY:
        # The spacing is consistent, but the profile is not *periodic* at that spacing:
        # dips that merely exist (noise, a speckled sheet) scatter evenly enough to pass
        # every other test here - measured: a flat field with 2 % noise does exactly that.
        return None
    error = 1.4826 * float(np.median(np.abs(inliers - period))) / np.sqrt(len(inliers))
    return period, error, float(np.clip(1.0 - scatter / MAX_SPACING_SCATTER, 0.0, 1.0))


def _smooth(profile, sigma=SMOOTH_SIGMA_PX):
    """A Gaussian blur of one profile, written out rather than imported (numpy only)."""
    if sigma <= 0:
        return profile
    radius = int(max(1, round(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(profile, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _periodicity(profile, period):
    """The profile's own correlation at ``period`` - how much of it that spacing explains."""
    values = np.asarray(profile, dtype=np.float64) - float(np.mean(profile))
    correlation = np.correlate(values, values, mode="full")[len(values) - 1:]
    if correlation[0] <= 0:
        return 0.0
    correlation = correlation / correlation[0]
    low = int(period)
    if low < 1 or low + 1 >= len(correlation):
        return 0.0
    weight = period - low
    return float((1.0 - weight) * correlation[low] + weight * correlation[low + 1])


def _line_centres(profile, depth_fraction=DIP_DEPTH_FRACTION):
    """The centre of every dark dip in a profile, as the centroid of the dip.

    Expects an already smoothed profile (``_band_reading`` smooths once, for both this and
    :func:`_periodicity`), and thresholds relative to the band's own median and a robust
    "deepest line". Both are measured requirements, not taste:

    * smoothing - measured on this project's photo, one band's printed lines each register as
      **two** dips 4-12 px apart (the shoulders of a pale print), which halves the apparent
      spacing and made that band answer 18.2 px where the same paper answers 26.8 px two bands
      away. A 3-px blur merges the shoulders and cannot touch a lattice whose pitch is 27 px.
    * a local, conservative threshold (0.7 sd) - so the printed paper's heavier every-fifth
      line is harmless, and so the *outside* of the paper, which the caller filled with a
      constant, never becomes a line.
    """
    profile = np.asarray(profile, dtype=np.float64)
    median = float(np.median(profile))
    # A robust "deepest line": the 1st percentile, not the minimum. One dark blob on the
    # paper would otherwise set the reference and hide every real line.
    deepest = median - float(np.percentile(profile, 1.0))
    if deepest < MIN_DEPTH_SIGMA * float(profile.std()):
        # Nothing here is a *line*: a contrast-free field, or one dark blob. A dip must be
        # deeper than the profile's own scatter to be a printed line (measured on this
        # project's photo: the lines are 2.4 of the profile's spread, the noise is not).
        return np.array([])
    dark = profile < median - depth_fraction * deepest
    centres = []
    start = None
    for index, flag in enumerate(dark):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            weights = median - profile[start:index]
            if weights.sum() > 0:
                centres.append(float((weights * np.arange(start, index)).sum()
                                     / weights.sum()))
            start = None
    return np.array(centres)


def _band_profile(grey, mask, start, band, vertical):
    """One band's profile, cropped to the paper, or ``None`` when it is mostly not paper.

    The crop matters on a *circular* field: a band of columns is ~99 % paper on this rig's
    photo, but a band of *rows* is only ~70 %, so a coverage test on the whole frame width
    throws away every row band and with it one of the two line families (measured: the
    "periodicity found in 1 of the two line families" refusal). Cropping to the paper's own
    extent makes both directions equally measurable, and the pixels outside the paper - which
    the caller has filled with a constant - are replaced by the band's median so the
    autocorrelation is not run on a step.
    """
    inside = mask[:, start:start + band] if vertical else mask[start:start + band, :]
    values = grey[:, start:start + band] if vertical else grey[start:start + band, :]
    along = np.nonzero(inside.any(axis=1 if vertical else 0))[0]
    if len(along) < 2:
        return None
    low, high = int(along.min()), int(along.max()) + 1
    if vertical:
        inside, values = inside[low:high, :], values[low:high, :]
        if inside.mean() < 0.85:
            return None
        fill = float(np.median(values[inside]))
        return np.where(inside, values, fill).mean(axis=1)
    inside, values = inside[:, low:high], values[:, low:high]
    if inside.mean() < 0.85:
        return None
    fill = float(np.median(values[inside]))
    return np.where(inside, values, fill).mean(axis=0)


def _band_readings(grey, mask, band=BAND_PX, step=BAND_STEP_PX):
    """Local readings, by direction: ``{"columns": [...], "rows": [...]}``.

    A band of *columns* averages along the rows, so it measures the spacing of the
    **horizontal** lines; a band of *rows* measures the vertical ones. Both directions are
    asked for separately because a square grid has both, and one family of stripes does not.
    """
    height, width = grey.shape
    readings = {"columns": [], "rows": []}
    for x in range(0, max(1, width - band + 1), step):
        profile = _band_profile(grey, mask, x, band, vertical=True)
        if profile is None:
            continue
        reading = _band_reading(profile)
        if reading is not None:
            readings["columns"].append(reading)
    for y in range(0, max(1, height - band + 1), step):
        profile = _band_profile(grey, mask, y, band, vertical=False)
        if profile is None:
            continue
        reading = _band_reading(profile)
        if reading is not None:
            readings["rows"].append(reading)
    return readings


def measure(frame, pitch_um=GRID_PITCH_UM, band=BAND_PX, step=BAND_STEP_PX,
            warn_spread_pct=1.0):
    """Measure the scale in one photo of grid paper, wherever the paper sits.

    Refuses - with the reason - when the frame is not square grid paper: too few bands hold
    a periodicity, only one family is present, or the two families disagree. A wrong scale is
    worse than no scale, which is the rule `vision/grid.py` already follows.

    Raises:
        LatticeNotFound: the photo cannot give a scale.
    """
    grey, mask, stop, angle = _prepare(frame)
    readings = _band_readings(grey, mask, band, step)
    every = [reading for values in readings.values() for reading in values]
    if len(every) < 2 * MIN_BANDS:
        raise LatticeNotFound(
            f"periodicity found in {len(every)} band(s) of the frame "
            f"({len(readings['columns'])} column bands, {len(readings['rows'])} row bands, "
            f"{2 * MIN_BANDS} needed) - is the grid paper in view, in focus and filling "
            "the field?")

    # The scale is the *consensus* of the field's bands. A band can lock onto a structure
    # that is internally consistent and yet not the lattice - measured here: one band in
    # seven answered 13.4 px, half the pitch, from the shoulders of the printed lines'
    # profile. Such a band is not averaged in (that would double the spread and drag the
    # median); it is rejected, and the surviving bands are what the scale rests on.
    consensus = float(np.median([period for period, _, _ in every]))
    tolerance = CONSENSUS_TOLERANCE * consensus
    by_direction = {direction: [reading for reading in values
                                if abs(reading[0] - consensus) <= tolerance]
                    for direction, values in readings.items()}
    thin = [direction for direction, values in by_direction.items()
            if len(values) < MIN_BANDS]
    if len(by_direction) < 2 or thin:
        raise LatticeNotFound(
            "periodicity found in only one line family - "
            f"{len(by_direction.get('columns', []))} column bands and "
            f"{len(by_direction.get('rows', []))} row bands agree with the field's consensus, "
            f"and {MIN_BANDS} of each are needed (is the grid paper in view, in focus and "
            "filling the field?)")

    medians = {direction: float(np.median([period for period, _, _ in values]))
               for direction, values in by_direction.items()}
    mean_period = float(np.mean(list(medians.values())))
    anisotropy = abs(medians["columns"] - medians["rows"]) / mean_period * 100.0
    if anisotropy > MAX_ANISOTROPY_PCT:
        raise LatticeNotFound(
            f"the two line families are {anisotropy:.1f}% apart in spacing "
            f"({medians['columns']:.2f} px against {medians['rows']:.2f} px) - that is not "
            "a square grid (lay the paper flat and perpendicular to the lens)")

    periods = np.array([period for values in by_direction.values()
                        for period, _, _ in values])
    period = float(np.median(periods))
    statistical_pct = float(np.median([error for values in by_direction.values()
                                       for _, error, _ in values])) / period * 100.0
    spread_pct = float(np.max(np.abs(periods - period)) / period * 100.0)
    coverage = float(mask.mean()) if stop is not None else 1.0
    caveats = []
    if spread_pct > warn_spread_pct:
        # "drift", never "lens distortion, measured": the drift is what these bands
        # measure (`SR-OPTICS-036`); a distortion *model* is a different thing and stays
        # unmeasured (`SR-OPTICS-032`). The wording is load-bearing - it is quoted on the
        # Calibrate screen.
        caveats.append(
            f"the local scale moves {spread_pct:.1f}% across the field - that is the "
            "drift, measured from the same bands, and it is larger than this scale's own "
            "uncertainty")
    return LatticeEstimate(period_px=period, angle_deg=angle, pitch_um=pitch_um,
                           field_stop=stop, coverage=coverage,
                           bands=sum(len(values) for values in by_direction.values()),
                           spread_pct=spread_pct, statistical_pct=statistical_pct,
                           caveats=tuple(caveats))
