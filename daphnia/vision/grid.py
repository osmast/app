"""Auto-calibration from 1 mm grid paper.

The project's reference target is a **printed 1 mm grid**. It needs no clicking: a
square grid is a periodic pattern, so the scale *is* its spatial period, and a
period can be measured across the whole frame instead of between two hand-placed
points. That is the same reason "click a longer span" helps - automated, and at
sub-pixel precision.

> **No longer the OPTICS surface's estimator (2026-10-19).** This detector assumes a
> *clean* photo - a bright, evenly lit, unshaded square grid that fills the frame - and
> measured against this rig's actual photos it **refuses all nine** of them: a phone held
> over an eyepiece gives a pale print inside a hard black field stop whose edge is
> broadband energy (`MIN_PROMINENCE`), tilted by ~9 deg with a harmonic where the period
> search looks (`MIN_PERIOD_PX`). `daphnia/vision/lattice.py` is the estimator the
> Calibrate screen uses instead (`SR-OPTICS-035`), and this module now supplies what it
> shares with it: the pitch defaults, the paper-tolerance and line-sigma constants, and
> `detect_grid` itself, which stays for clean photos (its tests still pin every refusal
> that should stay a refusal) and for the probe in `scripts/calibration_probe.py`, where
> "does the shipped detector answer this photo?" is a measurement worth making.

Two stages:

1. a 2D FFT finds the grid's two lattice directions and a coarse period. It is
   rotation-invariant, so the paper does not have to be aligned;
2. a projection onto each direction, followed by a zero-padded DFT peak search,
   refines the period to a fraction of a pixel and reports how strong it is.

Why this suits a phone held over an eyepiece: the apparent period changes with
zoom, focus and any knock to the adapter, so the only honest definition of the
scale is *how many pixels a 1 mm square spans in this image* - which is what this
measures. Re-run it whenever the image could have changed.

The accuracy floor is not the detector, it is the paper. Printed 1 mm spacing is
not metrology-grade (sheet stretch, print scaling, and a photocopy at "fit to
page" is far worse), so :data:`PAPER_PITCH_TOLERANCE_PCT` is folded into every
reported uncertainty instead of quoting a falsely precise number. Verify the print
once against a stage micrometer and the grid becomes trustworthy.
"""

from dataclasses import dataclass

import numpy as np

#: The reference target: printed 1 mm grid paper. One square = 1 mm = 1000 um.
GRID_PITCH_UM = 1000.0

#: Plausible grid periods, in pixels. Below the lower bound the squares are finer
#: than the sensor resolves; above the upper bound too few fit in the frame.
MIN_PERIOD_PX = 6.0
MAX_PERIOD_PX = 500.0

#: Fewer visible periods than this and the estimate is not worth trusting.
MIN_PERIODS = 3

#: Below this many periods, comparing the frame's centre with the whole frame says
#: more about the estimator than about the lens, so no distortion verdict is given.
MIN_PERIODS_FOR_DISTORTION = 6

#: Longest side the detector works on, in pixels. A phone photo is 12 MP or more, and
#: a full-resolution FFT costs hundreds of megabytes to answer a question that
#: survives downscaling perfectly well: a block mean is a low-pass filter, and the
#: period is located to a fraction of a pixel either way. Every period is scaled back
#: to original pixels before it is reported, so this costs accuracy nothing and
#: memory a great deal.
MAX_DETECTION_SIDE_PX = 2000

#: How closely a *printed* 1 mm spacing matches 1 mm, in percent. This is the floor
#: under any grid-based calibration, and usually dominates the fit's own error.
PAPER_PITCH_TOLERANCE_PCT = 0.5

#: How well a grid line's centre can be located, in pixels - the statistical part
#: of the uncertainty.
LINE_SIGMA_PX = 0.3

#: A centre-vs-periphery scale difference above this is optical distortion rather
#: than a bigger grid, in percent.
DISTORTION_WARN_PCT = 2.0

#: How far the grid peak must stand above the spectrum's own noise floor. Noise has
#: a broad, irregular maximum; a periodic pattern has a sharp, isolated peak. This
#: is what stops the detector from inventing a scale out of a specimen photo, and it
#: has to hold for *both* line families.
MIN_PROMINENCE = 100.0

#: The prominence denominator never drops below this fraction of the spectrum's
#: strongest value. Without it, a noiseless image has a background of ~0 and any
#: numerically insignificant peak looks infinitely prominent.
_BACKGROUND_FLOOR_FRACTION = 1e-4

#: 1 mm grid paper is square. Two families whose spacings differ by more than this
#: are not a square grid (or the target is too oblique for the scale to mean
#: anything), so the detection is refused rather than averaged into a wrong number.
MAX_ANISOTROPY_PCT = 25.0

#: How far the two line families may deviate from perpendicular, in degrees. A
#: square grid's families cross at 90 degrees; a single striped pattern's "second
#: family" is only its own harmonic leakage, and this is what rejects it.
MAX_FAMILY_SKEW_DEG = 20.0

#: Zero-padding factor for the period refinement: a larger factor resolves a finer
#: frequency step, which is what makes the period sub-pixel accurate.
_ZERO_PADDING = 32


@dataclass(frozen=True)
class GridEstimate:
    """A detected 1 mm grid: the scale, measured from the image itself."""

    period_px: float            # spacing of the horizontal-ish line family
    period_px_vertical: float   # spacing of the other family
    angle_deg: float            # rotation of the grid in the image
    periods: float              # how many periods span the frame
    confidence: float           # 0..1, how much this stands out from the background
    distortion_pct: float = 0.0  # how much the scale drifts from centre to edge
    prominence: float = 0.0      # how far the grid peak stands above the noise floor
    pitch_um: float = GRID_PITCH_UM

    @property
    def mean_period_px(self) -> float:
        """Mean of the two line families, in pixels."""
        return (self.period_px + self.period_px_vertical) / 2.0

    @property
    def anisotropy_pct(self) -> float:
        """How different the two families are.

        Real grid paper is square, so a large value means the target is tilted
        relative to the optical axis and the squares only *look* rectangular.
        """
        mean = self.mean_period_px
        if mean <= 0:
            return 0.0
        return abs(self.period_px - self.period_px_vertical) / mean * 100.0

    def px_per_um(self, pitch_um=None) -> float:
        """Scale factor: pixels per micrometre of the specimen plane."""
        pitch = self.pitch_um if pitch_um is None else pitch_um
        if self.mean_period_px <= 0:
            raise ValueError("period_px must be > 0")
        return self.mean_period_px / pitch

    def statistical_uncertainty_pct(self) -> float:
        """Fit error only: how well the period was located, in percent."""
        if self.periods <= 0:
            return 100.0
        return (100.0 * LINE_SIGMA_PX * float(np.sqrt(2.0))
                / (self.periods * self.mean_period_px))

    def uncertainty_pct(self, paper_tolerance_pct=PAPER_PITCH_TOLERANCE_PCT) -> float:
        """Total uncertainty of the scale: the fit combined with the paper's own.

        The two are independent, so they combine in quadrature. In practice the
        paper term dominates, and saying so is the honest answer.
        """
        fit = self.statistical_uncertainty_pct()
        return float(np.hypot(fit, paper_tolerance_pct))

    def warnings(self) -> tuple:
        """What an operator should be told about this detection."""
        notes = []
        if self.periods < MIN_PERIODS * 1.5:
            notes.append(
                f"Only {self.periods:.1f} squares fit in the frame - aim at more.")
        if self.anisotropy_pct > 5.0:
            notes.append(
                f"The two axes differ by {self.anisotropy_pct:.1f}% - the grid is "
                "tilted, not perpendicular to the lens.")
        if self.confidence < 0.25:
            notes.append("The grid is faint in this image - refocus or relight.")
        if self.prominence < MIN_PROMINENCE * 2.0:
            notes.append(
                "The grid is not very distinct in this image - check focus and "
                "that the paper fills the field.")
        if self.distortion_pct > DISTORTION_WARN_PCT:
            notes.append(
                f"The scale drifts {self.distortion_pct:.1f}% between the centre and "
                "the edge of the frame (lens distortion) - keep specimens within "
                "the central 60-70% of the field.")
        return tuple(notes)


class GridNotFound(ValueError):
    """The image does not hold a usable 1 mm grid."""


def _to_grayscale(frame) -> np.ndarray:
    """A float grayscale view of a BGR, RGB or greyscale frame."""
    array = np.asarray(frame, dtype=float)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=2)
    if array.ndim != 2:
        raise ValueError("expected a 2D grayscale or 3D colour frame")
    return array


def _spectrum(image: np.ndarray) -> np.ndarray:
    """Centred magnitude spectrum of the windowed image."""
    height, width = image.shape
    centred = image - float(image.mean())
    window = np.outer(np.hanning(height), np.hanning(width))
    return np.fft.fftshift(np.abs(np.fft.fft2(centred * window)))


def _lattice_peaks(spectrum, min_period_px, max_period_px, count=2):
    """The strongest non-collinear lattice peaks, as (fy, fx) cycles per pixel.

    A square grid produces two orthogonal line families, so its spectrum holds two
    pairs of symmetric peaks. Taking the strongest peak and then masking its whole
    direction leaves the other family as the next maximum - no alignment needed.
    """
    height, width = spectrum.shape
    centre_y, centre_x = height // 2, width // 2
    fy = (np.arange(height) - centre_y) / height
    fx = (np.arange(width) - centre_x) / width
    grid_fy, grid_fx = np.meshgrid(fy, fx, indexing="ij")

    radius = np.hypot(grid_fy, grid_fx)
    allowed = (radius >= 1.0 / max_period_px) & (radius <= 1.0 / min_period_px)
    if not allowed.any():
        return []

    background = float(np.median(spectrum[allowed]))
    work = np.where(allowed, spectrum, 0.0)

    peaks = []
    for _ in range(count):
        index = np.unravel_index(int(np.argmax(work)), work.shape)
        value = float(work[index])
        if value <= 0 or value <= background:
            break
        peak_fy, peak_fx = float(grid_fy[index]), float(grid_fx[index])
        peaks.append((peak_fy, peak_fx, value, background))
        # Suppress this peak's whole direction (the line through the origin, which
        # also carries the symmetric partner and every harmonic), so the next pick
        # is the *other* line family. The test is an angular one: a distance test
        # would depend on the period and would mask the other family outright.
        norm = max(float(np.hypot(peak_fx, peak_fy)), 1e-12)
        distance = np.abs(peak_fx * grid_fy - peak_fy * grid_fx) / norm
        with np.errstate(invalid="ignore", divide="ignore"):
            deviation = distance / np.where(radius > 0, radius, np.nan)
        work = np.where(deviation < 0.1, 0.0, work)
    return peaks


def _refine_period(image, peak_fy, peak_fx, min_period_px, max_period_px):
    """Refine one lattice period by projecting the image along its direction.

    The projection of a line family onto its own normal is a 1D periodic signal
    spanning the frame. A heavily zero-padded DFT of it, with parabolic
    interpolation around the peak, locates the period to a fraction of a pixel -
    the automated version of "measure ten squares instead of one".

    Because it measures the *pitch* between repeats rather than the distance
    between two chosen edges, uneven line width and ink bleed cancel out instead of
    biasing the scale.
    """
    height, width = image.shape
    norm = float(np.hypot(peak_fx, peak_fy))
    if norm <= 0:
        return None
    unit_x, unit_y = peak_fx / norm, peak_fy / norm

    yy, xx = np.mgrid[0:height, 0:width]
    projected = xx * unit_x + yy * unit_y
    extent = float(projected.max() - projected.min())
    if extent <= 0:
        return None

    bins = int(np.ceil(extent)) + 1
    index = np.clip(((projected - projected.min()) * bins / extent).astype(int),
                    0, bins - 1)
    profile = np.bincount(index.ravel(),
                          weights=(image - float(image.mean())).ravel(),
                          minlength=bins)

    # A Hann window on the profile suppresses the spectral leakage that a
    # non-integer number of periods would otherwise bias the peak with; the peak
    # widens but stays symmetric, which parabolic interpolation locates precisely.
    magnitude = np.abs(np.fft.rfft(profile * np.hanning(bins),
                                   n=bins * _ZERO_PADDING))

    guess = 1.0 / norm
    if not min_period_px <= guess <= max_period_px:
        return None
    # A DFT bin j corresponds to a period of ``_ZERO_PADDING * extent / j`` pixels.
    expected = _ZERO_PADDING * extent / guess
    low = max(2, int(expected * 0.7))
    high = min(magnitude.size - 2, int(expected * 1.4) + 1)
    if high <= low:
        return None

    j = low + int(np.argmax(magnitude[low:high + 1]))
    y0 = float(magnitude[j - 1])
    y1 = float(magnitude[j])
    y2 = float(magnitude[j + 1])
    denominator = y0 - 2.0 * y1 + y2
    shift = 0.5 * (y0 - y2) / denominator if denominator else 0.0
    refined = j + float(np.clip(shift, -1.0, 1.0))
    if refined <= 0:
        return None

    period = _ZERO_PADDING * extent / refined
    return period if min_period_px <= period <= max_period_px else None


def _reduce(image, max_side):
    """Block-average the image so its longest side is at most ``max_side``.

    Returns ``(reduced, factor)``, where ``factor`` is what a period measured in the
    reduced image must be multiplied by to come back to original pixels. The factor
    is isotropic, so a square grid stays square - which the anisotropy gate depends
    on. A factor of 1 means nothing was changed.
    """
    height, width = image.shape
    factor = int(np.ceil(max(height, width) / float(max_side)))
    if factor <= 1:
        return image, 1
    rows = (height // factor) * factor
    columns = (width // factor) * factor
    block = image[:rows, :columns]
    reduced = block.reshape(rows // factor, factor,
                            columns // factor, factor).mean(axis=(1, 3))
    return reduced, factor


def detect_grid(frame, pitch_um=GRID_PITCH_UM, min_period_px=MIN_PERIOD_PX,
                max_period_px=MAX_PERIOD_PX,
                max_side_px=MAX_DETECTION_SIDE_PX) -> GridEstimate:
    """Measure the 1 mm grid in ``frame`` and return the scale it implies.

    ``frame`` may be any size: a phone photo is block-averaged down to ``max_side_px``
    before the FFT, and every period is scaled back to original pixels, so the
    returned scale is the scale of the photo that was supplied.

    Raises:
        GridNotFound: when no two line families form a usable square grid - too
            blurry, too few squares, or simply no grid in view. Failing loudly is
            the point: a wrong scale is worse than no scale.
    """
    image = _to_grayscale(frame)
    image, factor = _reduce(image, max_side_px)
    height, width = image.shape
    if min(height, width) < 32:
        raise GridNotFound("image is too small to hold a grid")

    # In the reduced image every period is ``factor`` times shorter, so the bounds
    # travel with it and the gates then mean the same thing at any resolution.
    low = min_period_px / factor
    high = max_period_px / factor

    spectrum = _spectrum(image)
    peaks = _lattice_peaks(spectrum, low, high, count=2)
    if len(peaks) < 2:
        raise GridNotFound("no two line families found - is the 1 mm grid in view?")

    (fy_a, fx_a, value_a, background), (fy_b, fx_b, value_b, _bg) = peaks
    # Both families have to stand out, or this is not a square grid: a single
    # family of stripes plus noise must never be enough to define a scale.
    floor = max(background, float(spectrum.max()) * _BACKGROUND_FLOOR_FRACTION)
    prominence = value_a / floor
    second_prominence = value_b / floor
    # The gate uses the weaker family, and so does the reported number: it is the
    # weakest line family that decides how convincing this grid is.
    prominence = min(prominence, second_prominence)
    if prominence < MIN_PROMINENCE:
        raise GridNotFound(
            "no square grid stands out in this image - is the 1 mm paper in "
            "view, and in focus?")

    period_a = _refine_period(image, fy_a, fx_a, low, high)
    period_b = _refine_period(image, fy_b, fx_b, low, high)
    if period_a is None or period_b is None:
        raise GridNotFound("the grid is too faint or too fine to measure")

    periods = float(min(width, height) / min(period_a, period_b))
    if periods < MIN_PERIODS:
        raise GridNotFound(
            f"only {periods:.1f} squares fit in the frame - aim at more of the grid")

    mean_period = (period_a + period_b) / 2.0
    anisotropy = (abs(period_a - period_b) / mean_period * 100.0
                  if mean_period > 0 else 0.0)
    if anisotropy > MAX_ANISOTROPY_PCT:
        raise GridNotFound(
            "the two line families are not square to each other - lay the 1 mm "
            "paper flat and perpendicular to the lens")

    # The families of a square grid cross at right angles. This is the gate that
    # rejects a single striped pattern, whose alleged "second family" is nothing
    # but the leakage of its own harmonics.
    magnitudes = np.hypot(fx_a, fy_a) * np.hypot(fx_b, fy_b)
    if magnitudes > 0:
        cosine = (fx_a * fx_b + fy_a * fy_b) / magnitudes
        if abs(float(cosine)) > float(np.sin(np.radians(MAX_FAMILY_SKEW_DEG))):
            raise GridNotFound(
                "the lines in this image are not a square grid - only one line "
                "direction is present")

    # Report the horizontal-ish family first, so the two axes stay comparable.
    if abs(fx_a) < abs(fx_b):
        (fy_a, fx_a), (fy_b, fx_b) = (fy_b, fx_b), (fy_a, fx_a)
        period_a, period_b = period_b, period_a
        value_a, value_b = value_b, value_a

    # A scale that differs in the middle of the frame is lens distortion, so it is
    # measured rather than assumed. The centre crop must itself hold enough periods
    # for its estimate to mean anything, otherwise the comparison is just noise.
    distortion_pct = 0.0
    if periods >= MIN_PERIODS_FOR_DISTORTION:
        centre = image[height // 4: max(height // 4 + 1, height * 3 // 4),
                       width // 4: max(width // 4 + 1, width * 3 // 4)]
        drift = []
        for peak_y, peak_x, whole in ((fy_a, fx_a, period_a),
                                      (fy_b, fx_b, period_b)):
            middle = _refine_period(centre, peak_y, peak_x, low, high)
            if middle is not None:
                drift.append(abs(middle - whole) / whole * 100.0)
        distortion_pct = max(drift) if drift else 0.0

    angle = float(np.degrees(np.arctan2(fy_a, fx_a)) % 180.0)
    strongest = max(1.0, float(spectrum.max()))
    confidence = float(np.clip((value_a - background) / strongest, 0.0, 1.0))

    return GridEstimate(
        # Back to the pixels of the frame that was supplied, not of the reduced one.
        period_px=period_a * factor,
        period_px_vertical=period_b * factor,
        angle_deg=angle,
        periods=periods,
        confidence=confidence,
        distortion_pct=distortion_pct,
        prominence=prominence,
        pitch_um=pitch_um,
    )


def grid_scale_px_per_um(frame, pitch_um=GRID_PITCH_UM) -> float:
    """Convenience: the pixels-per-micrometre scale of a 1 mm grid photo."""
    return detect_grid(frame, pitch_um=pitch_um).px_per_um()
