#!/usr/bin/env python3
"""The flow discorpy's *documentation* prescribes, as written, on one photo.

    python scripts/discorpy_doc_probe.py recordings/calibration/PROF_*.png --coef 3
    python scripts/discorpy_doc_probe.py PHOTO --no-mask      # what the mask buys, isolated

Where this sits among the three probes
--------------------------------------
* `scripts/calibration_probe.py` - the rig-specific one: the frame is *prepared* first
  (field stop fitted as a circle, line weights equalised, lattice straightened) with our
  own scipy/scikit-image/OpenCV code, then a discorpy flow runs on it.
* `scripts/discorpy_probe.py` - discorpy's *example* flow, frame exactly as loaded, no
  preprocessing at all.
* **this probe** - the flow the **documentation** prescribes for strong distortion plus
  perspective (discorpy docs §4.1, "Distortion calibration and correction for fisheye
  cameras" - the 1.7 / [C2] pipeline). It is the only one of the three that calls the four
  steps the others never do:

      prep.normalization_fft(img0, 10)                       # the docs normalise UP FRONT
      prep.remove_points_using_parabola_mask(...)            # the docs' "unwanted points"
      prep.group_dots_{hor,ver}_lines_based_polyfit(...)     # the docs' grouping
      proc.find_center_based_vanishing_points_iteration(...) # the 1.7 centre of distortion
      util.unwarp_color_image_backward(..., pad=...)         # colour, view preserved
      losa.save_metadata_json(...)                           # the library's own record

Every processing step is a discorpy call, with the documentation's own arguments; the
additions are a CLI, the reporting and the exit code - the same contract as
`scripts/discorpy_probe.py`. Nothing here re-implements anything the library ships.

**The control that makes this a comparison rather than a complaint.** On discorpy's own
synthetic single-weight grid the same flow, with the documentation's own arguments
(`--coef 5`, radius 9), groups **24 + 41 lines**, recovers the centre of distortion to
**(639.16, 359.58)** against a true (640, 360), and unwraps to **0.018 px rms** with 0 % of
the points over 1 px - a pass, and the best any probe in this repo has done on that target.
What it fails at on this rig's graph paper (below) is therefore a property of the *target
and the frame*, not of the flow.

Why those four steps are expected to matter (measured, elsewhere in this repo)
-----------------------------------------------------------------------------
* **normalise first.** The example flow hands the raw frame to the slope/distance
  estimator and gets **+30.20 deg / 61.0 px** on this photo, where the tilt is -9.0 deg
  and the cell is 27 px: the estimator only sees a 216x216 central ROI
  (`prep._select_roi(ratio=0.3, square=True)`), and on the unnormalised frame the
  lattice is not its strongest ridge. The docs put `normalization_fft(img0, 10)` in front
  of *everything*, and this repo measured that normalising (then flattening the print)
  is what fixed the spacing estimate (19.1 px -> 28.2 px).
* **the mask** is the docs' own answer to "some unwanted points may be picked up". On
  this photo the unwanted points are the eyepiece's ocular ring: a hard black circle with
  its left, right and bottom arcs *inside* the frame (x 152-1049, y up to 714), which the
  cross-point detector samples happily.
* **the centre of distortion** comes from the vanishing-point iteration, whose own
  docstring calls it "more robust than the others when there's significant perspective
  distortion" - and this photo carries a -9 deg tilt.
* **the unwarp** is colour and padded, so the corrected frame keeps the original view.

What it reports, refuses and returns
------------------------------------
Every number printed is an argument or a return of a discorpy call, except two clearly
labelled reductions: the slope spread (§4.1.4's own diagnostic, as a number) and the
local magnification the app cares about (docs/calibration.md §2), both computed from
library output. The verdict is discorpy's own `post.check_distortion` - a residual over
1 px on more than 15 % of the points means the correction is not good enough - plus two
guards the library's check does not make:

* the centre of distortion must land **inside the frame** (a grid with no distortion left
  to fit gives a centre outside it, and `check_distortion` is happy with that);
* the line unwarp must actually **move the points**. On the example flow's float32 lines
  `post.unwarp_line_backward` is a silent no-op (measured: 0.00 px of movement against
  6.25 px in float64), which is why that probe casts to float64. Here the lines come from
  the grouping, so they are float64 already - and that is *measured*, not assumed, from
  the displacement printed in step 8.

Exit code 0 when the check passes, the centre is inside the frame and the unwarp moved
the points; 1 otherwise. No micrometre value is derived and nothing is wired into the app.

Comparing with the rig-specific probe
-------------------------------------
    conda run -n z-app python scripts/calibration_probe.py <photo> --coef 3
    conda run -n z-app python scripts/discorpy_doc_probe.py <photo> --coef 3

The sibling prints its numbers from the *straightened* frame and maps the centre back, so
compare this probe's centre with the one it calls "in the original frame". Its residual is
measured on the lines the coefficients were *fitted from*; this probe prints that number
too ("after, corrected lines") beside the docs' own figure ("after, grouped lines"),
because the two are not the same.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Everything heavy is imported lazily, so ``--help`` works without the CV stack and the
#: failure names the command that fixes it.
INSTALL = ("discorpy is not installed in this environment. Install it and re-run:\n"
           "  conda run -n z-app pip install discorpy\n"
           "  # it needs scipy + scikit-image, which this environment already has")


def _require_discorpy():
    """Import discorpy, or stop with the command that installs it."""
    try:
        import discorpy.losa.loadersaver as losa
        import discorpy.post.postprocessing as post
        import discorpy.prep.linepattern as lprep
        import discorpy.prep.preprocessing as prep
        import discorpy.proc.processing as proc
        import discorpy.util.utility as util
    except ImportError as error:
        raise SystemExit(f"{INSTALL}\n  ({error})")
    return losa, prep, lprep, proc, post, util


#: The documentation's own arguments, per call, from docs §4.1.3/§4.1.4 - a named bundle
#: because "the documented flow" is only a claim if the *arguments* are the documented
#: ones too. Anything not listed keeps the library default, which is what the docs use
#: when they pass nothing (e.g. the cross-point ratio, 0.3 by default). The two
#: ``*_fraction`` entries are the docs' mask margins - (400, 300) of a 4000-wide and
#: (150, 200) of a 3000-high GoPro frame - expressed as fractions, so the same mask
#: geometry applies to this rig's 1280x720 frames.
DOC = {"roi": 0.3, "radius": 9, "sensitive": 0.1, "ratio": 0.3,
       "group_ratio": 0.1, "dot_miss": 3, "accepted": 0.65, "order": 2,
       "residual": 3.0, "iteration": 2, "coef": 5, "dist_px": None,
       "grouping": "polyfit",
       "hor_curviness": 0.4, "ver_curviness": 0.3, "rotate": 0.0,
       "hor_fraction": (0.100, 0.075), "ver_fraction": (0.050, 0.067)}

def _magnification(coefficients, radius):
    """The local magnification of discorpy's backward model at ``radius``, and its ratio.

    docs/calibration.md §2: for the backward model ``xd = xc + fact(ru) * xu`` with
    ``fact(r) = sum c_i r^i``, a point at distance ``ru`` lands at ``ru * fact(ru)``, so
    the local magnification is the derivative ``fact(r) + r * fact'(r)``. It is the one
    number the app cares about, because a scale is positional; it is *evaluated* from
    discorpy's coefficients, not fitted here, and the rig-specific probe prints the same
    quantity over a radius sweep so a run of either can be compared with the other.
    """
    import numpy as np

    coefficients = np.asarray(coefficients, dtype=float)
    value = float(sum(c * radius ** i for i, c in enumerate(coefficients))
                  + radius * sum(i * c * radius ** (i - 1)
                                 for i, c in enumerate(coefficients) if i > 0))
    return value, value / float(coefficients[0])


def _residual(post, residual_hor, residual_ver):
    """discorpy's own sub-pixel verdict on a pair of residual arrays, plus its numbers.

    ``post.check_distortion`` answers "is there still distortion": True when more than
    15 % of the points sit further than 1 px from their fitted line. Quoted, not
    re-derived; ``over_1px_pct`` is the same fraction printed so a reader can see how far
    from the bar a run landed without reading it off the plots.
    """
    import numpy as np

    values = np.concatenate([np.asarray(residual_hor)[:, 1],
                             np.asarray(residual_ver)[:, 1]])
    return {"rms_px": float(np.sqrt(np.mean(values ** 2))),
            "max_px": float(values.max()), "mean_px": float(values.mean()),
            "over_1px_pct": float(100.0 * float((values > 1.0).mean())),
            "passed": not (post.check_distortion(residual_hor)
                           or post.check_distortion(residual_ver))}


def _slope_spread(proc, hor_lines, ver_lines, xcenter, ycenter):
    """§4.1.4's own diagnostic as a number: how much the line slopes vary across the grid.

    The docs plot the parabolas' b-coefficient against their c-coefficient before and
    after `correct_perspective_effect`: b varying with c *is* the perspective effect, and
    after the correction "the variation of the slope of lines is under 0.0035". This is
    that variation (peak-to-peak, radians) read out of discorpy's own
    ``_para_fit_hor/ver`` - a private function the docs call directly. Returns
    ``(horizontal, vertical)``.
    """
    import numpy as np

    coef_hor = proc._para_fit_hor(hor_lines, xcenter, ycenter)[0]
    coef_ver = proc._para_fit_ver(ver_lines, xcenter, ycenter)[0]
    return float(np.ptp(coef_hor[:, 1])), float(np.ptp(coef_ver[:, 1]))


def _displacement(before, after):
    """Largest movement the unwarp applied to any point, in pixels.

    The guard for the silent no-op: ``post.unwarp_line_backward`` solves ``rd = ru *
    fact(ru)`` with ``scipy.optimize.minimize``, and on float32 input the numerical
    gradient fails and the function hands back its input unchanged (measured on the
    example flow: 0.00 px there against 6.25 px in float64). If this reads 0.00 px, the
    residual "after" is not a measurement of anything.
    """
    import numpy as np

    if len(before) == 0:
        return 0.0
    return float(max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
                            ).max() for a, b in zip(after, before)))


def _run_flow(mat, path, options, run_dir, say=print):
    """The documented flow, step for step, on ``mat`` (the frame as loaded).

    Returns the numbers discorpy produced, for the JSON beside them. Raises ``SystemExit``
    with a named reason when the frame cannot yield a model: a silent empty result would
    be blamed on the photo (mlContext.md §2, law 8).
    """
    import numpy as np

    losa, prep, lprep, proc, post, util = _require_discorpy()
    base = str(run_dir)
    height, width = mat.shape
    radius = int(options["radius"])

    say("1-> Normalise the frame: the docs' step 1 (prep.normalization_fft(img0, 10))")
    normalised = prep.normalization_fft(mat, 10)
    losa.save_image(f"{base}/normalised.png", normalised)

    say("2-> Slope and distance between the lines: the docs run this on the normalised "
        "frame (lprep.calc_slope_distance_hor_lines / _ver_lines)")
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(
        normalised, ratio=options["roi"], radius=radius, chessboard=False)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(
        normalised, ratio=options["roi"], radius=radius, chessboard=False)
    say(f"    horizontal  slope {slope_hor:+.5f} rad "
        f"({np.degrees(np.arctan(slope_hor)):+.2f} deg), distance {dist_hor:.3f} px")
    say(f"    vertical    slope {slope_ver:+.5f} rad "
        f"({np.degrees(np.arctan(slope_ver)):+.2f} deg), distance {dist_ver:.3f} px")
    if not (np.isfinite(dist_hor) and np.isfinite(dist_ver)
            and dist_hor > 0 and dist_ver > 0):
        raise SystemExit(
            "no line family at all: a distance that is not a positive finite number "
            f"({dist_hor!r} / {dist_ver!r}). Try --roi or --radius, or the prepared frame "
            "in scripts/calibration_probe.py")
    if options["dist_px"]:
        # Every later step takes this number as its scale - the generated profiles are
        # spaced `ratio * dist` apart and the grouping measures against `line_dist` - and
        # on a mixed-weight target it is the *unreliable* number (measured: 19.06 px for
        # one family and 28.66 px for the other on the *same* prepared frame, against a
        # printed cell of 27 px). Pinning it is the library's own parameter, not new code.
        say(f"    spacing pinned to {options['dist_px']:.3f} px for both families "
            "(--dist-px)")
        dist_hor = dist_ver = float(options["dist_px"])

    say("3-> Cross points (lprep.get_cross_points_hor_lines / _ver_lines), then the "
        "docs' unwanted-point mask (prep.remove_points_using_parabola_mask)")
    points_hor = lprep.get_cross_points_hor_lines(
        normalised, slope_ver, dist_ver, ratio=options["ratio"], bgr="bright",
        chessboard=False, radius=radius, sensitive=options["sensitive"],
        select_peaks=False)
    points_ver = lprep.get_cross_points_ver_lines(
        normalised, slope_hor, dist_hor, ratio=options["ratio"], bgr="bright",
        chessboard=False, radius=radius, sensitive=options["sensitive"],
        select_peaks=False)
    say(f"    {len(points_hor)} + {len(points_ver)} points before the mask")
    if len(points_hor) == 0 or len(points_ver) == 0:
        raise SystemExit(
            "no cross point was found at all, so the grouping would raise on an empty "
            "input. Try --radius 11 (measured right for this paper) or the prepared frame "
            "in scripts/calibration_probe.py")
    if options["mask"]:
        hor_margin = tuple(int(round(width * f)) for f in options["hor_fraction"])
        ver_margin = tuple(int(round(height * f)) for f in options["ver_fraction"])
        points_hor = prep.remove_points_using_parabola_mask(
            points_hor, height, width, hor_curviness=options["hor_curviness"],
            ver_curviness=options["ver_curviness"], hor_margin=hor_margin,
            ver_margin=ver_margin, rotate=options["rotate"])
        points_ver = prep.remove_points_using_parabola_mask(
            points_ver, height, width, hor_curviness=options["hor_curviness"],
            ver_curviness=options["ver_curviness"], hor_margin=hor_margin,
            ver_margin=ver_margin, rotate=options["rotate"])
        say(f"    {len(points_hor)} + {len(points_ver)} points kept inside the parabola "
            f"mask (hor_margin {hor_margin} px, ver_margin {ver_margin} px, curviness "
            f"{options['hor_curviness']}/{options['ver_curviness']}, "
            f"rotate {options['rotate']} deg)")
        if len(points_hor) == 0 or len(points_ver) == 0:
            raise SystemExit(
                "the mask removed every point: its margins/curviness are too aggressive "
                "for this frame (the docs' values are 10 %/7.5 % of the width and "
                "5 %/6.7 % of the height, from a 4000x3000 fisheye). Pass --hor-margin / "
                "--ver-margin in pixels, or --no-mask")

    say("4-> Group the points on curved lines "
        "(prep.group_dots_hor_lines_based_polyfit / _ver_lines_based_polyfit) and drop "
        "the outliers (prep.remove_residual_dots_hor / _ver)")
    if options["grouping"] == "neighbour":
        # The library's *other* grouping: the neighbour search the 3.3.4 tutorial uses.
        # Exposed here so a run can tell "the docs' grouping is the limit" from "the
        # points are the limit" - it is the same library, the same points, one call apart.
        lines_hor = prep.group_dots_hor_lines(
            points_hor, slope_hor, dist_hor, ratio=options["group_ratio"],
            num_dot_miss=options["dot_miss"], accepted_ratio=options["accepted"])
        lines_ver = prep.group_dots_ver_lines(
            points_ver, slope_ver, dist_ver, ratio=options["group_ratio"],
            num_dot_miss=options["dot_miss"], accepted_ratio=options["accepted"])
    else:
        lines_hor = prep.group_dots_hor_lines_based_polyfit(
            points_hor, slope_hor, dist_hor, ratio=options["group_ratio"],
            num_dot_miss=options["dot_miss"], accepted_ratio=options["accepted"],
            order=options["order"])
        lines_ver = prep.group_dots_ver_lines_based_polyfit(
            points_ver, slope_ver, dist_ver, ratio=options["group_ratio"],
            num_dot_miss=options["dot_miss"], accepted_ratio=options["accepted"],
            order=options["order"])
    say(f"    {len(lines_hor)} horizontal and {len(lines_ver)} vertical lines grouped")
    if len(lines_hor) < 4 or len(lines_ver) < 4:
        raise SystemExit(
            f"only {len(lines_hor)} + {len(lines_ver)} lines were grouped - the centre of "
            "distortion needs curved lines in *both* directions, so this is too few. "
            "Loosen --group-ratio / --dot-miss / --accepted, widen the mask, or print a "
            "single-weight target")
    lines_hor = prep.remove_residual_dots_hor(lines_hor, slope_hor, options["residual"])
    lines_ver = prep.remove_residual_dots_ver(lines_ver, slope_ver, options["residual"])
    say(f"    {len(lines_hor)} + {len(lines_ver)} lines left after a "
        f"{options['residual']:.1f} px residual pass")


    say("5-> Centre of distortion "
        "(proc.find_center_based_vanishing_points_iteration)")
    xcenter, ycenter = proc.find_center_based_vanishing_points_iteration(
        lines_hor, lines_ver, iteration=options["iteration"])
    inside = bool(0.0 <= xcenter <= width and 0.0 <= ycenter <= height)
    say(f"    x {xcenter:.2f} px, y {ycenter:.2f} px, in a {width} x {height} frame"
        + ("" if inside else "   <-- OUTSIDE the frame: a grid with no distortion left "
                             "to fit gives exactly this"))

    say("6-> Correct the perspective effect (proc.correct_perspective_effect); the docs' "
        "diagnostic is the spread of the line slopes, before and after")
    spread_before = _slope_spread(proc, lines_hor, lines_ver, xcenter, ycenter)
    corrected_hor, corrected_ver = proc.correct_perspective_effect(
        lines_hor, lines_ver, xcenter, ycenter)
    spread_after = _slope_spread(proc, corrected_hor, corrected_ver, xcenter, ycenter)
    say(f"    slope spread, peak to peak: horizontal {spread_before[0]:.5f} -> "
        f"{spread_after[0]:.5f} rad, vertical {spread_before[1]:.5f} -> "
        f"{spread_after[1]:.5f} rad")

    say("7-> Radial coefficients (proc.calc_coef_backward), fitted on the "
        "perspective-corrected lines")
    coefficients = proc.calc_coef_backward(corrected_hor, corrected_ver, xcenter, ycenter,
                                           options["coef"])
    say("    " + ", ".join(f"c{i}={value:+.6e}" for i, value in enumerate(coefficients)))

    say("8-> Unwarp the lines (post.unwarp_line_backward) and measure both residuals - "
        "the docs' own figure is the grouped lines, the coefficients come from the "
        "corrected ones")
    unwarped_hor = post.unwarp_line_backward(lines_hor, xcenter, ycenter, coefficients)
    unwarped_ver = post.unwarp_line_backward(lines_ver, xcenter, ycenter, coefficients)
    moved = _displacement(lines_hor, unwarped_hor)
    before = _residual(post, post.calc_residual_hor(lines_hor, xcenter, ycenter),
                       post.calc_residual_ver(lines_ver, xcenter, ycenter))
    grouped = _residual(post, post.calc_residual_hor(unwarped_hor, xcenter, ycenter),
                        post.calc_residual_ver(unwarped_ver, xcenter, ycenter))
    fitted_hor = post.unwarp_line_backward(corrected_hor, xcenter, ycenter, coefficients)
    fitted_ver = post.unwarp_line_backward(corrected_ver, xcenter, ycenter, coefficients)
    fitted = _residual(post, post.calc_residual_hor(fitted_hor, xcenter, ycenter),
                       post.calc_residual_ver(fitted_ver, xcenter, ycenter))
    say(f"    unwarp displacement           {moved:.3f} px largest point movement - "
        + ("the unwarp is live" if moved > 0.05 else
           "THE UNWARP DID NOTHING, see _displacement"))
    say(f"    residual before unwarping     {before['rms_px']:.3f} px rms, "
        f"{before['max_px']:.3f} px worst, {before['over_1px_pct']:.1f} % over 1 px")
    say(f"    after, grouped lines          {grouped['rms_px']:.3f} px rms, "
        f"{grouped['max_px']:.3f} px worst, {grouped['over_1px_pct']:.1f} % over 1 px "
        "(the docs' §4.1.5 figure)")
    say(f"    after, corrected lines        {fitted['rms_px']:.3f} px rms, "
        f"{fitted['max_px']:.3f} px worst, {fitted['over_1px_pct']:.1f} % over 1 px "
        "(what the coefficients were fitted from)")

    say("9-> Unwarp the image in colour, keeping the view "
        "(util.unwarp_color_image_backward) and save the library's own record "
        "(losa.save_metadata_json)")
    colour = losa.load_image(str(path), average=False)
    corrected = util.unwarp_color_image_backward(colour, xcenter, ycenter, coefficients,
                                                 pad=options["pad"])
    losa.save_image(f"{base}/corrected_colour.png", corrected)
    losa.save_metadata_json(f"{base}/distortion_parameters.json", xcenter, ycenter,
                            coefficients)
    say(f"    corrected image               {corrected.shape[1]} x {corrected.shape[0]} "
        f"px (pad={options['pad']}), saved with the metadata the library can read back")

    outer = 0.42 * min(height, width)
    _, ratio = _magnification(coefficients, outer)
    say(f"    local magnification at r={outer:.0f} px   {ratio * 100:8.3f} % of the "
        f"centre ({100 * (ratio - 1):+7.3f} %)   (docs/calibration.md §2)")

    return {"xcenter": float(xcenter), "ycenter": float(ycenter),
            "centre_inside_frame": inside,
            "coefficients": [float(value) for value in coefficients],
            "line_distance_px": [float(dist_hor), float(dist_ver)],
            "line_slope_rad": [float(slope_hor), float(slope_ver)],
            "lines": [len(lines_hor), len(lines_ver)],
            "points": [len(points_hor), len(points_ver)],
            "slope_spread_rad": {"before": list(spread_before),
                                 "after": list(spread_after)},
            "unwarp_displacement_px": moved,
            "residual_before": before,
            "residual_after_grouped_lines": grouped,
            "residual_after_corrected_lines": fitted}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the discorpy documentation's line-pattern flow (docs §4.1), as "
                    "written, on one photo.")
    parser.add_argument("image", help="a photo of the printed grid target")
    parser.add_argument("--out", default=None,
                        help="output folder (default: recordings/discorpy_doc_probe); one "
                             "sub-folder per photo is created inside it")
    for name, kind, text in (
            ("coef", int, "number of radial coefficients"),
            ("roi", float, "ROI ratio around the middle, for the slope/distance estimate"),
            ("radius", int, "search radius in px"),
            ("sensitive", float, "peak sensitivity, smaller = more sensitive"),
            ("ratio", float, "profile spacing for the cross points, as a fraction of the "
                             "line distance"),
            ("group_ratio", float, "grouping tolerance ratio"),
            ("dot_miss", int, "dots allowed to be missed between two dots"),
            ("accepted", float, "fraction of the longest line a grouped line must have"),
            ("order", int, "polynomial order of the grouping fit"),
            ("residual", float, "drop points further than this many px from their line"),
            ("iteration", int, "iterations of the vanishing-point centre estimate"),
            ("dist_px", float, "pin the line spacing for both families instead of "
                               "trusting the estimate (px; the estimator is unstable on a "
                               "mixed-weight target)")):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=kind,
                            default=DOC[name],
                            help=f"discorpy: {text} ({DOC[name]} in the docs)")
    parser.add_argument("--hor-curviness", dest="hor_curviness", type=float,
                        default=DOC["hor_curviness"],
                        help="discorpy: curvature of the mask's left/right boundaries "
                             f"({DOC['hor_curviness']} in the docs)")
    parser.add_argument("--ver-curviness", dest="ver_curviness", type=float,
                        default=DOC["ver_curviness"],
                        help="discorpy: curvature of the mask's top/bottom boundaries "
                             f"({DOC['ver_curviness']} in the docs)")
    parser.add_argument("--rotate", type=float, default=DOC["rotate"],
                        help="discorpy: mask rotation in degrees (0 in the docs)")
    parser.add_argument("--hor-margin", dest="hor_margin", default=None,
                        help="mask margins in px as 'left,right'; default is the docs' "
                             f"{DOC['hor_fraction'][0]:.3f} / {DOC['hor_fraction'][1]:.3f} "
                             "of the width")
    parser.add_argument("--ver-margin", dest="ver_margin", default=None,
                        help="mask margins in px as 'top,bottom'; default is the docs' "
                             f"{DOC['ver_fraction'][0]:.3f} / {DOC['ver_fraction'][1]:.3f} "
                             "of the height")
    parser.add_argument("--mask", action=argparse.BooleanOptionalAction, default=True,
                        help="apply remove_points_using_parabola_mask (default True, as "
                             "in the docs)")
    parser.add_argument("--grouping", choices=("polyfit", "neighbour"),
                        default=DOC["grouping"],
                        help="discorpy: which of the library's two line groupings to use. "
                             "'polyfit' is the docs' §4.1 call "
                             "(group_dots_*_lines_based_polyfit); 'neighbour' is the "
                             "3.3.4 tutorial's (group_dots_*_lines). Same points, one "
                             "call apart")
    parser.add_argument("--pad", type=int, default=None,
                        help="padding in px for the colour unwarp; omit it for the "
                             "library's automatic pad, which keeps the original view")
    args = parser.parse_args(argv)
    options = {key: getattr(args, key) for key in DOC
               if key not in ("hor_fraction", "ver_fraction")}
    options["hor_fraction"] = DOC["hor_fraction"]      # --hor-margin overrides it later
    options["ver_fraction"] = DOC["ver_fraction"]      # --ver-margin overrides it later
    options["mask"] = args.mask
    options["pad"] = True if args.pad is None else args.pad

    from daphnia import paths
    from daphnia.vision.calibration import utc_now

    source = Path(args.image).expanduser()
    if not source.exists():
        raise SystemExit(f"no such file: {source}")
    out_root = (Path(args.out) if args.out
                else paths.CALIBRATION_IMAGES_DIR.parent / "discorpy_doc_probe")
    run_dir = out_root / source.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    print("DaphniaScope - discorpy probe: the documented §4.1 flow, as written")
    print(f"  photo                        {source}")
    print(f"  output                       {run_dir}")
    print("  arguments                    "
          + ", ".join(f"{key}={value}" for key, value in options.items()))

    losa, *_ = _require_discorpy()
    try:
        mat = losa.load_image(str(source))
    except (ValueError, OSError) as error:
        raise SystemExit(
            f"discorpy's loader could not read {source}: {error}\n"
            "  it goes through PIL, which does not decode .HEIC/.HEIF. Convert first:\n"
            f"  sips -s format png {source} --out {source.with_suffix('.png')}")
    if mat.ndim != 2:
        raise SystemExit(f"{source} loaded as {mat.shape} - the line-pattern flow needs a "
                         "2D (grey) array")
    print(f"  frame                        {mat.shape[1]} x {mat.shape[0]} px, grey levels "
          f"{mat.min():.0f} - {mat.max():.0f}")
    if args.hor_margin:
        left, right = (float(value) for value in args.hor_margin.split(","))
        options["hor_fraction"] = (left / mat.shape[1], right / mat.shape[1])
    if args.ver_margin:
        top, bottom = (float(value) for value in args.ver_margin.split(","))
        options["ver_fraction"] = (top / mat.shape[0], bottom / mat.shape[0])
    print()

    model, reason, hint = None, None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = _run_flow(mat, source, options, run_dir)
        except SystemExit as error:              # a named refusal from the flow
            reason = str(error)
        except Exception as error:               # noqa: BLE001 - discorpy raises wildly
            reason = f"{type(error).__name__}: {error}"
            hint = ("the documented flow needs curved lines in both directions and a "
                    "parabola fit opening the right way, which is the step that usually "
                    "needs help on a photo like ours: try --no-mask, wider --hor-margin / "
                    "--ver-margin, a looser --group-ratio / --accepted, or the prepared "
                    "frame in scripts/calibration_probe.py")
        # discorpy warns "Check if there is any missing grouped line" once per fit that
        # suspects the family is not a uniform lattice: evidence about the *target*, so it
        # is a reported number here, not screen noise.
        target_warnings = sum(1 for item in caught
                              if "missing grouped line" in str(item.message))
    if reason is not None:
        print(f"\n  no model                     {reason}")
    if hint is not None:
        print(f"  what to try                  {hint}")

    print()
    print("10-> discorpy's own verdict (post.check_distortion, quoted not re-derived)")
    if model is None:
        verdict = ("no model was fitted from this photo, so the distortion here is "
                   "unmeasured - it is not 0 %. Nothing was corrected and nothing was "
                   "stored.")
    else:
        before = model["residual_before"]
        fitted = model["residual_after_corrected_lines"]
        grouped = model["residual_after_grouped_lines"]
        print(f"  residual before unwarping    {before['rms_px']:.3f} px rms, "
              f"{before['max_px']:.3f} px worst, {before['over_1px_pct']:.1f} % over 1 px")
        print(f"  after, grouped lines         {grouped['rms_px']:.3f} px rms, "
              f"{grouped['max_px']:.3f} px worst, {grouped['over_1px_pct']:.1f} % "
              "over 1 px")
        print(f"  after, corrected lines       {fitted['rms_px']:.3f} px rms, "
              f"{fitted['max_px']:.3f} px worst, {fitted['over_1px_pct']:.1f} % "
              "over 1 px")
        if not model["centre_inside_frame"]:
            verdict = (f"the centre of distortion landed outside the frame "
                       f"({model['xcenter']:.1f}, {model['ycenter']:.1f} px in a "
                       f"{mat.shape[1]} x {mat.shape[0]} frame): the fit is degenerate and "
                       "the residuals above describe nothing. discorpy's own sub-pixel "
                       "check does not catch that, which is why it is named here.")
        elif model["unwarp_displacement_px"] <= 0.05:
            verdict = ("the line unwarp moved nothing (see step 8), so the 'after' "
                       "residuals are not a measurement of the model.")
        elif fitted["passed"]:
            verdict = ("the correction passes discorpy's own sub-pixel check on this "
                       "photo, so the coefficients describe a distortion. Not measured, "
                       "and not claimed: whether a corrected frame measures a specimen any "
                       "better, what the corrected frame's scale is (c0 is a centre "
                       "magnification, not a scale), and whether the coefficients survive "
                       "the mirror test.")
        elif fitted["rms_px"] > before["rms_px"]:
            verdict = (f"the residual is *worse* after unwarping "
                       f"({before['rms_px']:.3f} -> {fitted['rms_px']:.3f} px rms, worst "
                       f"{fitted['max_px']:.3f} px): the fitted coefficients do not "
                       "describe this grid, so they are not a distortion measurement. "
                       "That is what mis-grouped lines produce - read the line counts and "
                       "the centre above.")
        else:
            verdict = (f"{fitted['over_1px_pct']:.1f} % of the points stay more than 1 px "
                       f"from their line (worst {fitted['max_px']:.3f} px), which is "
                       "discorpy's own bar for \"the correction is not good enough\"; the "
                       "coefficients are then not a distortion measurement.")
    print(f"  verdict                      {verdict}")
    print(f"  discorpy warnings            {target_warnings} x \"check if there is any "
          "missing grouped line\" (0 = the lattice looks uniform to the library; more = "
          "the target mixes line weights)")
    print("  not measured here            any micrometre value: this probe reports pixels, "
          "derives no scale, and is wired into nothing")

    report = {"tool": "scripts/discorpy_doc_probe.py",
              "measured_at": utc_now().isoformat(), "image": str(source),
              "image_size": [int(mat.shape[1]), int(mat.shape[0])],
              "documented_arguments": {key: (list(value) if isinstance(value, tuple)
                                             else value)
                                       for key, value in options.items()},
              "model": model, "missing_line_warnings": target_warnings,
              "verdict": verdict}
    if reason is not None:
        report["no_model_reason"] = reason
    report_path = run_dir / f"{source.stem}_probe.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print()
    print(f"written  {report_path}")
    for path in sorted(run_dir.iterdir()):
        if path != report_path:
            print(f"         {path}")
    return 0 if (model is not None and model["centre_inside_frame"]
                 and model["unwarp_displacement_px"] > 0.05
                 and model["residual_after_corrected_lines"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())




