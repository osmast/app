"""The distortion **probe**: discorpy's calibration flow, and the figures it produces.

`vision/grid.py`, `vision/lattice.py` and `vision/points.py` measure the **scale** - one
number for the whole frame. This module goes after the other half: the *distortion model*,
and it does it by running **discorpy** (Apache-2.0, Diamond Light Source) end to end, the way
its own documentation shows it -

* ``discorpy.losa/prep/proc/post`` in the order of *3.3.3 Process a challenging X-ray target
  image* (dot pattern: background correction, binarisation, dot selection, grouping) and
* *3.3.4 Process a line-pattern image* (the flow this project's printed grid paper actually
  needs: slope+spacing, cross points, grouping, parabola regeneration, coefficients, unwarp);

- and returns, rather than a single number, **the figures that flow makes** at every stage,
  *before* and *after* the correction: the normalised image, the binarised and cleaned
  segmentations, the extracted points, the grouped lines, the residual **before**, the
  corrected lines, the residual **after**, the unwarped image and the difference image.

## What this is, and what it is not

It is a **probe**, exactly as `scripts/calibration_probe.py` is: nothing here is stored, and
no frame is rewritten by the app (`SR-OPTICS-032`/`SR-OPTICS-033` - the model is not storable
yet, and its magnitude is still ~1.5-2x the drift measured directly). The defaults are the
ones that script *measured* on this rig's photos, not discorpy's demo defaults (noted where
they differ); a design with no reference is a guess, and the reference here is that script
plus the library's own documentation.

discorpy is imported **inside** the functions, so the domain layer still imports without it,
and a missing discorpy is a named error, never a silent empty result.
"""

import warnings
from dataclasses import dataclass, field as dataclass_field, replace
import numpy as np


#: Polynomial coefficients for the backward model. Measured on this rig's photos: 3 is the
#: count that passes discorpy's own sub-pixel check (4 gives 2.22 px, 5 gives 1.08 px - both
#: failing), so 3 is the default here rather than discorpy's demo's 5.
DEFAULT_COEF = 3

#: The peak-search radius, in pixels. Measured: it must match the ~10 px interval the cross
#: points are sampled at - smaller is *worse*, not better (radius 4 -> 6.25 px residual,
#: 9 -> 0.51 px, on the same photo).
DEFAULT_RADIUS = 9

#: How far each printed line's amplitude is clipped down to, in standard deviations of the
#: normalised image. Printed graph paper's every-fifth heavy line otherwise dominates the
#: peak detector and the "lines" it groups are a patchwork of the lattice.
DEFAULT_EQUALISE_SD = 1.0

#: Zero-padding sigma of discorpy's own background correction.
DEFAULT_SIGMA = 10

#: The dot-pattern segmentation knobs (discorpy 3.3.3): the SNR the threshold is taken at,
#: the binarisation ratio, and how far a blob's size may differ from the median before it is
#: not a dot. The ratio is the demo's raised 0.8, for the same reason: small binary objects
#: cluster along the edges of the image.
DEFAULT_SNR = 3.0
DEFAULT_BINARY_RATIO = 0.5
DEFAULT_SIZE_RATIO = 0.8

#: The line-pattern flow knobs (this project's measured values, `scripts/calibration_probe.py`).
DEFAULT_ROI = 0.3
DEFAULT_SENSITIVE = 0.1
DEFAULT_GROUP_RATIO = 0.3
DEFAULT_DOT_MISS = 6
DEFAULT_ACCEPTED = 0.65
DEFAULT_RESIDUAL_PX = 2.0

#: The figures, in the order the flow makes them, with the caption each is shown under.
FIGURE_CAPTIONS = (
    ("normalised", "background corrected (normalization_fft)"),
    ("binarised", "binarised (Otsu, or the manual cut)"),
    ("cleaned", "non-dot objects removed"),
    ("cross points", "cross points extracted, before grouping"),
    ("grouped lines", "points grouped into lines"),
    ("residual before", "residual of the distorted lines (px)"),
    ("corrected lines", "lines after the model was applied"),
    ("residual after", "residual of the corrected lines (px)"),
    ("image unwarped", "the frame unwarped by the fitted model"),
    ("difference", "unwarped minus original"),
)


class DistortionUnavailable(RuntimeError):
    """The probe cannot run - the message names the missing piece and what fixes it."""


@dataclass(frozen=True)
class DistortionProbe:
    """What discorpy's flow made of one frame: the model, its residuals, and every figure."""

    x_center: float
    y_center: float
    coefficients: tuple
    residual_before_px: float
    residual_after_px: float
    worst_line_px: float
    subpixel: bool
    line_spacing_px: tuple
    lines: tuple                      # (horizontal, vertical) lines kept
    points: tuple                     # (horizontal, vertical) cross points kept
    frame_size: tuple = (0, 0)        # (width, height) of the probed frame
    figures: dict = dataclass_field(default_factory=dict)
    notes: tuple = ()

    @property
    def magnitude_pct(self):
        """The model's magnification change at the frame's edge, in percent, or ``None``.

        For the backward model ``xd = xc + fact(ru) * xu`` the local magnification is
        ``fact(r) + r * fact'(r)``, so this is the position-dependent error a single-number
        calibration carries - the number that says whether the correction is worth having.
        """
        if not self.coefficients or not self.frame_size[0]:
            return None
        radius = float(self.frame_size[0]) / 2.0
        fact = sum(float(c) * radius ** i for i, c in enumerate(self.coefficients))
        derivative = sum(i * float(c) * radius ** (i - 1)
                         for i, c in enumerate(self.coefficients) if i)
        return float((fact + radius * derivative - 1.0) * 100.0)


def _require_discorpy():
    """The discorpy modules the flow needs, or a named error saying how to get them."""
    try:
        import discorpy.losa.loadersaver as losa
        import discorpy.post.postprocessing as post
        import discorpy.prep.linepattern as lprep
        import discorpy.prep.preprocessing as prep
        import discorpy.proc.processing as proc
    except ImportError as error:                     # pragma: no cover - env issue
        raise DistortionUnavailable(
            "discorpy is not installed in this environment, so the distortion probe "
            "cannot run. Install it into the environment that runs the app:\n"
            "    conda env update -f environment.yml") from error
    return losa, prep, lprep, proc, post


def _grey(frame):
    """A 2D float view of a frame, whatever it is (colour, grey, or already float)."""
    image = np.asarray(frame)
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D or 3D image, got shape {image.shape}")
    return image.astype(np.float32)


def _to_bgr(image):
    """An 8-bit BGR view of anything the flow produced, so every figure is drawable."""
    import cv2

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(array[:, :, :3])


def _paper_mask(grey, margin=16):
    """A boolean mask of the paper inside the eyepiece's field stop, and the fit circle.

    The stop's own ring is a hard edge that discorpy reads as one enormous line - measured on
    this project's photo, enough to drop the grouping from 40 lines to 1 - so the disc is
    eroded by a margin and the surround is filled with the paper's median before the flow
    sees the frame.
    """
    from daphnia.vision.lattice import field_stop

    stop = field_stop(grey)
    height, width = grey.shape
    if stop is None:
        return np.ones((height, width), dtype=bool), None
    yy, xx = np.mgrid[0:height, 0:width]
    disc = ((xx - stop.centre_x) ** 2 + (yy - stop.centre_y) ** 2
            <= max(1.0, stop.radius - margin) ** 2)
    return (disc if disc.any() else np.ones((height, width), dtype=bool)), stop


def _fill_surround(grey, mask):
    """The paper's median outside the mask, so no hard stop edge is left in the picture."""
    inside = grey[mask] if mask.any() else grey.reshape(-1)
    return np.where(mask, grey, float(np.median(inside))).astype(np.float32)


def _shrink(mask, pixels):
    """Inset a mask by ``pixels`` - the keep-out band along the paper's own edge."""
    import cv2

    if pixels <= 0:
        return mask
    kernel = np.ones((2 * int(round(pixels)) + 1, 2 * int(round(pixels)) + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def _inside(points, mask):
    """Keep only the ``(y, x)`` points that fall inside a mask.

    discorpy has no mask parameter, and it needs none for a flat target; this project's photo
    has a circular field stop, so its edge is a strong line-shaped structure the cross-point
    detector happily finds. Dropping those points *before* the grouping keeps that arc out of
    the parabola fits.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if points.size == 0:
        return points
    height, width = mask.shape[:2]
    rows = np.clip(np.round(points[:, 0]).astype(int), 0, height - 1)
    columns = np.clip(np.round(points[:, 1]).astype(int), 0, width - 1)
    return points[mask[rows, columns]]


def _equalise(frame, depth_sd=DEFAULT_EQUALISE_SD):
    """Clip the dark side of the background-corrected image to one common depth.

    Printed grid paper is one lattice with a **heavier line every fifth cell**: the *positions*
    are uniformly spaced, only the darkness is not. A peak detector that works by amplitude
    finds whichever weight dominates and leaves the rest, so the "lines" it groups are a
    patchwork of the lattice instead of the lattice. Measured effect of this step on this
    project's photo: the grouped lattice went from **8 lines to up to 40** per family, and
    discorpy's spacing estimate from 19.1 px (wrong) to 28.2 px (right). The order matters -
    clipping the *raw* image answers 45.1/21.9 px on the same paper.
    """
    _, prep, _, _, _ = _require_discorpy()

    normalised = prep.normalization_fft(frame, DEFAULT_SIGMA)
    median = float(np.median(normalised))
    return np.maximum(normalised,
                      median - depth_sd * float(normalised.std())).astype("float32")


def _align(frame, mask, angle_deg):
    """Straighten the lattice, returning ``(frame, mask)``.

    Measured why: discorpy's cross-point extraction samples *tilted* profiles and truncates
    that sampling to stay inside the image - at a 9 deg tilt that is ~114 px along one side
    where **no points are extracted at all**. Aligned, both families are sampled over the whole
    frame and the spacing estimator no longer has to search +-30 deg.
    """
    import cv2

    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    border = float(np.median(frame))
    rotated = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    rotated_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (width, height),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0) > 0
    return rotated, rotated_mask


def _prepare(frame, radius=DEFAULT_RADIUS, margin=16):
    """Everything this rig's photos need before discorpy sees them, in order.

    Returns ``(grey, lattice, mask, stop, angle)``: the raw grey frame; the
    amplitude-equalised, straightened frame the flow runs on; the paper mask; the fitted field
    stop; and the tilt that was removed.
    """
    grey = _grey(frame)
    mask, stop = _paper_mask(grey, margin)
    lattice = _equalise(_fill_surround(grey, mask))
    _, _, lprep, _, _ = _require_discorpy()

    def slope_of(image):
        value, _ = lprep.calc_slope_distance_hor_lines(
            image, ratio=DEFAULT_ROI, radius=radius, sensitive=DEFAULT_SENSITIVE,
            select_peaks=False)
        return float(value)

    slope = slope_of(lattice)
    angle = float(np.degrees(np.arctan(slope)))
    if abs(angle) > 0.05:
        straight, straight_mask = _align(lattice, mask, angle)
        # A flipped or rotated photo needs the other sign; rather than trust the convention
        # the alignment is *checked* - getting it wrong *doubles* the tilt (measured: -8.95
        # deg became -17.70 deg).
        if abs(slope_of(straight)) > abs(slope):
            angle = -angle
            straight, straight_mask = _align(lattice, mask, angle)
        lattice, mask = straight, straight_mask
    else:
        angle = 0.0
    return grey, lattice, mask, stop, angle


# --- the figures ------------------------------------------------------------
#
# discorpy's own `save_plot_image` / `save_residual_plot` need matplotlib and write files; the
# app draws its figures with OpenCV into arrays instead - no new dependency, and they can go
# straight into a Qt label. Same data, same look: points, the lines they were grouped into,
# and the residual the correction left behind.

POINT_COLOR = (60, 60, 220)
LINE_COLOR = (0, 191, 255)
CANVAS_COLOR = 18


def _blank(height, width):
    """A dark canvas the plots are drawn on, so they read against any theme."""
    return np.full((int(height), int(width), 3), CANVAS_COLOR, np.uint8)


def _plot_points(image, points, colour=POINT_COLOR):
    """One small dot per extracted point (discorpy's points are ``(y, x)``)."""
    import cv2

    for y, x in np.asarray(points, dtype=float).reshape(-1, 2):
        if np.isfinite(y) and np.isfinite(x):
            cv2.circle(image, (int(round(x)), int(round(y))), 2, colour, -1, cv2.LINE_AA)


def _plot_lines(image, lines, colour=LINE_COLOR):
    """Each grouped line as the polyline through its points.

    ``lines`` is a numpy array (discorpy's grouped lines are), so the empty test must come
    first and must not be ``lines or []`` - that raises on more than one element.
    """
    import cv2

    if lines is None:
        return
    for line in lines:
        points = np.asarray(line, dtype=float).reshape(-1, 2)
        finite = points[np.isfinite(points).all(axis=1)]
        if len(finite) < 2:
            continue
        xy = np.stack([finite[:, 1], finite[:, 0]], axis=1)      # (y, x) -> (x, y)
        cv2.polylines(image, [np.round(xy).astype(np.int32)], False, colour, 1, cv2.LINE_AA)


def _residual_plot(residuals, height, width, limit=3.0):
    """Deviation against position, with the **±1 px** band discorpy's own check uses.

    ``residuals`` is discorpy's ``(N, 2)`` array: a position and that point's deviation. The
    band is drawn because "is this sub-pixel?" is the question the plot exists to answer, and
    a plot without the threshold only invites a guess.
    """
    import cv2

    canvas = _blank(height, width)
    centre = height / 2.0
    scale = (height / 2.0) / max(float(limit), 1e-6)
    band = max(1, int(round(scale)))
    for sign in (-1, 1):
        cv2.line(canvas, (0, int(centre + sign * band)), (width, int(centre + sign * band)),
                 (90, 90, 90), 1)
    cv2.line(canvas, (0, int(centre)), (width, int(centre)), (150, 150, 150), 1)

    values = np.asarray(residuals, dtype=float).reshape(-1, 2)
    if values.size:
        deviations = values[:, 1]
        count = len(deviations)
        xs = np.linspace(0, width - 1, count)
        ys = np.clip(centre + deviations * scale, 0, height - 1)
        for x, y in zip(xs, ys):
            cv2.circle(canvas, (int(round(x)), int(round(y))), 1, POINT_COLOR, -1)
    return canvas


def _figures(prep, lattice, points, lines, corr, residuals, unwarped_image, difference):
    """Assemble every figure of the flow, all in the **one** frame the flow worked in.

    Every image here comes from the same `_prepare`d frame, so nothing is superimposed across
    frames of reference - the rule that cost this project thirty epochs when a metric mixed two
    of them (`mlContext.md` law 1). That frame is the *straightened* one; the tilt is removed by
    rotation, which does not change the scale.
    """
    height, width = lattice.shape[:2]
    figures = {"normalised": _to_bgr(lattice)}

    # The dot-pattern segmentation (discorpy 3.3.3), so its two figures can be seen too. A
    # failure here is not a failure of the flow - it only means the preview is unavailable, so
    # it falls back to the normalised image rather than take the panel down.
    try:
        threshold = prep.calculate_threshold(lattice, bgr="bright", snr=DEFAULT_SNR)
        binary = prep.binarization(lattice, ratio=DEFAULT_BINARY_RATIO, thres=threshold)
        figures["binarised"] = _to_bgr(binary * 255.0)
        size, _distance = prep.calc_size_distance(binary)
        cleaned = prep.select_dots_based_size(binary, size, ratio=DEFAULT_SIZE_RATIO)
        cleaned = prep.select_dots_based_ratio(cleaned, ratio=DEFAULT_SIZE_RATIO)
        figures["cleaned"] = _to_bgr(cleaned * 255.0)
    except Exception:                                # noqa: BLE001 - a preview, not the flow
        figures["binarised"] = _to_bgr(lattice)
        figures["cleaned"] = _to_bgr(lattice)

    points_image = _blank(height, width)
    _plot_points(points_image, points[0], POINT_COLOR)
    _plot_points(points_image, points[1], LINE_COLOR)
    figures["cross points"] = points_image

    lines_image = _blank(height, width)
    _plot_lines(lines_image, lines[0], LINE_COLOR)
    _plot_lines(lines_image, lines[1], POINT_COLOR)
    figures["grouped lines"] = lines_image

    figures["residual before"] = _residual_plot(residuals[0], height, width)
    corrected = _blank(height, width)
    _plot_lines(corrected, corr[0], LINE_COLOR)
    _plot_lines(corrected, corr[1], POINT_COLOR)
    figures["corrected lines"] = corrected
    figures["residual after"] = _residual_plot(residuals[1], height, width)

    figures["image unwarped"] = _to_bgr(unwarped_image)
    figures["difference"] = _to_bgr(np.asarray(difference, np.float32) + 128.0)
    return figures


def _flow(frame, coef=DEFAULT_COEF, radius=DEFAULT_RADIUS):
    """The flow itself, called by :func:`probe` inside its warning capture.

    Raises:
        DistortionUnavailable: discorpy is missing, or no line family was found.
        ValueError: ``frame`` is not an image.
    """
    _, prep, lprep, proc, post = _require_discorpy()

    grey, lattice, mask, stop, _angle = _prepare(frame, radius=radius)
    height, width = lattice.shape[:2]

    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(
        lattice, ratio=DEFAULT_ROI, radius=radius, sensitive=DEFAULT_SENSITIVE,
        chessboard=False, select_peaks=False)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(
        lattice, ratio=DEFAULT_ROI, radius=radius, sensitive=DEFAULT_SENSITIVE,
        chessboard=False, select_peaks=False)
    if not (np.isfinite(dist_hor) and np.isfinite(dist_ver)
            and dist_hor > 0 and dist_ver > 0):
        raise DistortionUnavailable(
            "discorpy's flow found no line family in this frame - is the printed target in "
            "view, in focus and filling the field?")

    points_hor = lprep.get_cross_points_hor_lines(
        lattice, slope_ver, dist_ver, ratio=DEFAULT_BINARY_RATIO, bgr="bright",
        chessboard=False, radius=radius, sensitive=DEFAULT_SENSITIVE, select_peaks=False)
    points_ver = lprep.get_cross_points_ver_lines(
        lattice, slope_hor, dist_hor, ratio=DEFAULT_BINARY_RATIO, bgr="bright",
        chessboard=False, radius=radius, sensitive=DEFAULT_SENSITIVE, select_peaks=False)
    # Keep only points well inside the paper: the step between paper and the filled surround is
    # a local extremum in any profile crossing it, so points pile up along that edge.
    keep = _shrink(mask, radius + 2)
    points_hor = _inside(points_hor, keep)
    points_ver = _inside(points_ver, keep)

    # The grouping and outlier removal live in `prep` (discorpy.prep.preprocessing), not in
    # `lprep` - `lprep` has the slope/spacing and cross-point steps only.
    hor_lines = prep.group_dots_hor_lines(
        points_hor, slope_hor, dist_hor, ratio=DEFAULT_GROUP_RATIO,
        num_dot_miss=DEFAULT_DOT_MISS, accepted_ratio=DEFAULT_ACCEPTED)
    ver_lines = prep.group_dots_ver_lines(
        points_ver, slope_ver, dist_ver, ratio=DEFAULT_GROUP_RATIO,
        num_dot_miss=DEFAULT_DOT_MISS, accepted_ratio=DEFAULT_ACCEPTED)
    hor_lines = prep.remove_residual_dots_hor(hor_lines, slope_hor, DEFAULT_RESIDUAL_PX)
    ver_lines = prep.remove_residual_dots_ver(ver_lines, slope_ver, DEFAULT_RESIDUAL_PX)

    # `regenerate_grid_points_parabola` is the documented step that took the residual from
    # 3.18 px to 0.51 px: the model fits *lines*, and this rebuilds the noisy point cloud from
    # those lines instead of fitting the cloud directly.
    hor_lines, ver_lines = proc.regenerate_grid_points_parabola(
        hor_lines, ver_lines, perspective=False, find_center=True)
    x_center, y_center = proc.find_cod_coarse(hor_lines, ver_lines)
    corr_hor, corr_ver = proc.correct_perspective_effect(
        hor_lines, ver_lines, x_center, y_center)
    coefficients = proc.calc_coef_backward(corr_hor, corr_ver, x_center, y_center, coef)

    residual_before = _residual_rms(post, corr_hor, corr_ver, 0.0, 0.0)
    unwarped_hor = post.unwarp_line_backward(corr_hor, x_center, y_center, coefficients)
    unwarped_ver = post.unwarp_line_backward(corr_ver, x_center, y_center, coefficients)
    residual_after = _residual_rms(post, unwarped_hor, unwarped_ver, x_center, y_center)
    subpixel = not (
        post.check_distortion(post.calc_residual_hor(unwarped_hor, x_center, y_center))
        or post.check_distortion(post.calc_residual_ver(unwarped_ver, x_center, y_center)))
    worst = max(_worst_line(unwarped_hor, horizontal=True),
                _worst_line(unwarped_ver, horizontal=False))

    unwarped_image = post.unwarp_image_backward(lattice, x_center, y_center, coefficients)
    difference = np.asarray(unwarped_image, np.float32)[:height, :width] - lattice

    figures = _figures(
        prep, lattice, points=(points_hor, points_ver), lines=(hor_lines, ver_lines),
        corr=(corr_hor, corr_ver),
        residuals=(_residuals(post, corr_hor, corr_ver, 0.0, 0.0),
                   _residuals(post, unwarped_hor, unwarped_ver, x_center, y_center)),
        unwarped_image=unwarped_image, difference=difference)

    notes = []
    if stop is not None:
        notes.append(f"field stop masked: r {stop.radius:.0f} px, surround filled before the "
                     "flow saw the frame")
    if residual_after >= residual_before > 0:
        notes.append(f"the model left a LARGER residual than it started with "
                     f"({residual_after:.2f} px against {residual_before:.2f} px) - the "
                     "correction made this frame worse, so it is not usable")
    if not subpixel:
        notes.append("discorpy's own check FAILS this model: a residual over 1 px on more than "
                     "15 % of the points is not good enough - do not store it")
    magnitude = None
    try:
        radius_edge = float(width) / 2.0
        fact = sum(float(c) * radius_edge ** i for i, c in enumerate(coefficients))
        slope = sum(i * float(c) * radius_edge ** (i - 1)
                    for i, c in enumerate(coefficients) if i)
        magnitude = (fact + radius_edge * slope - 1.0) * 100.0
    except (TypeError, ValueError):
        magnitude = None
    if magnitude is not None and abs(magnitude) > 20.0:
        notes.append(f"the model would change magnification by {magnitude:+.0f} % at the edge "
                     "of the frame - no eyepiece does that, so these coefficients are unfit "
                     "(a mixed-weight target, or too few grouped lines)")
    notes.append("a probe: nothing is stored and no frame is rewritten (SR-OPTICS-032/033)")
    return DistortionProbe(
        x_center=float(x_center), y_center=float(y_center),
        coefficients=tuple(float(value) for value in coefficients),
        residual_before_px=float(residual_before), residual_after_px=float(residual_after),
        worst_line_px=float(worst), subpixel=bool(subpixel),
        line_spacing_px=(float(dist_hor), float(dist_ver)),
        lines=(len(hor_lines), len(ver_lines)), points=(len(points_hor), len(points_ver)),
        frame_size=(int(width), int(height)), figures=figures, notes=tuple(notes))


#: discorpy's own warnings that mean something to the operator, and the sentence each is turned
#: into. The library's deprecations (a scikit-image property rename *inside* discorpy) are
#: captured and dropped: nothing here can act on them, and printing them into the operator's
#: console is noise, not information.
WARNING_NOTES = (
    ("missing grouped line",
     "discorpy warns that the grouping may have MISSED a line - a printed target with more "
     "than one line weight does exactly this (the documented reason for this project's "
     "residual), so treat the model as provisional"),
)


def _warnings_to_notes(caught):
    """Turn discorpy's own warnings into notes; drop the library's deprecation noise.

    Measured on this probe: the flow emits ``UserWarning: Check if there is any missing
    grouped line`` (a real signal about the *target*) and two ``FutureWarning``\\ s from
    discorpy's own use of a renamed scikit-image property (not actionable here). The first
    belongs in the panel, the second belongs nowhere the operator can see.
    """
    notes = []
    for item in caught:
        if not issubclass(item.category, UserWarning):
            continue
        text = " ".join(str(item.message).split()).lower()
        for marker, sentence in WARNING_NOTES:
            if marker in text and sentence not in notes:
                notes.append(sentence)
    return notes


def probe(frame, coef=DEFAULT_COEF, radius=DEFAULT_RADIUS):
    """Run discorpy's calibration flow on one frame and return the model **and the figures**.

    discorpy's warnings are **captured**, not printed: the ones that say something about the
    frame become notes on the result (see `_warnings_to_notes`), and the library's own
    deprecations are dropped. A console full of ``RegionProperties.minor_axis_length`` tells the
    operator nothing; "the grouping may have missed a line" tells them the model is provisional
    - and that is exactly the case `docs/calibration.md` documents for printed graph paper's
    three line weights.

    Raises:
        DistortionUnavailable: discorpy is missing, or the flow found no line family - the
            message says which, and what to do about it.
        ValueError: ``frame`` is not an image.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _flow(frame, coef=coef, radius=radius)
    extra = tuple(_warnings_to_notes(caught))
    return replace(result, notes=result.notes + extra) if extra else result


def _residuals(post, horizontal, vertical, x_center, y_center):
    """discorpy's raw residual arrays of both families, concatenated for one plot."""
    hor = np.asarray(post.calc_residual_hor(horizontal, x_center, y_center), dtype=float)
    ver = np.asarray(post.calc_residual_ver(vertical, x_center, y_center), dtype=float)
    return np.concatenate([hor.reshape(-1, 2), ver.reshape(-1, 2)], axis=0)


def _residual_rms(post, horizontal, vertical, x_center, y_center):
    """RMS deviation of both families, in pixels - the residual the model left behind."""
    values = _residuals(post, horizontal, vertical, x_center, y_center)
    if not values.size:
        return 0.0
    return float(np.sqrt(np.mean(values[:, 1] ** 2)))


def _worst_line(lines, horizontal):
    """RMS deviation of the worst-fitted line, in pixels.

    One badly grouped line is invisible in an overall RMS, and a badly grouped line is exactly
    what a mixed-weight target produces - so it is reported separately.
    """
    worst = 0.0
    if lines is None:
        return worst
    for line in lines:
        points = np.asarray(line, dtype=float).reshape(-1, 2)
        finite = points[np.isfinite(points).all(axis=1)]
        if len(finite) < 2:
            continue
        values = finite[:, 0] if horizontal else finite[:, 1]
        worst = max(worst, float(np.sqrt(np.mean((values - values.mean()) ** 2))))
    return worst
