#!/usr/bin/env python3
"""discorpy's own line-pattern flow, unmodified, on one calibration photo.

    python scripts/discorpy_probe.py recordings/calibration/PROF_*.png
    python scripts/discorpy_probe.py PHOTO --out /tmp/probe --coef 3

Why this exists
---------------
`scripts/calibration_probe.py` answers *can discorpy calibrate our photo* by preparing
the frame first: the eyepiece's field stop is masked as a fitted circle, the three
printed line weights are equalised, and the tilted lattice is straightened - all with
scipy / scikit-image / OpenCV, none of it discorpy. That preparation is measured to
matter, and because it is ours it also hides what the library does *on its own*.

This script is the opposite by design: **the reference flow, verbatim, on the photo as
it came off the phone**, with no pre-processing of any kind. Its only additions to
discorpy's own line-pattern example (`examples/fisheye_calibration_line_pattern.py`,
discorpy 1.7.0, Apache-2.0) are a CLI, one output folder per photo, and discorpy's own
verdict. The two scripts therefore differ in exactly one axis - the prepared frame -
which is the comparison an operator wants: *is it the library or the photo?*

The flow, in the reference's order and with the reference's arguments:

    losa.load_image
      -> lprep.calc_slope_distance_{hor,ver}_lines
      -> lprep.get_cross_points_{hor,ver}_lines         (ratio 0.5, sensitive 0.1)
      -> prep.group_dots_{hor,ver}_lines                (the library's own defaults)
      -> prep.remove_residual_dots_{hor,ver}            (residual 2.0)
      -> proc.regenerate_grid_points_parabola           (perspective=True)
      -> proc.find_cod_coarse -> proc.calc_coef_backward
      -> post.unwarp_line_backward + post.calc_residual_{hor,ver}
      -> post.unwarp_image_backward, losa.save_image

Every number printed and written is an argument or a return value of one of those
functions or of `post.check_distortion`. Nothing is re-derived here, and no image
processing happens outside the library (mlContext.md §2, law 1: one frame of reference
- the frame the library was handed).

What it reports, and what it refuses to say
-------------------------------------------
The verdict is discorpy's own `post.check_distortion`: a residual over 1 px on more
than 15 % of the points means the correction is not good enough (discorpy 1.7.0,
`post/postprocessing.py:391`; docs/calibration.md §5, rule 2). A run that fails it is
**not** "a small distortion": it is not a distortion measurement, and nothing here
rounds it to 0 % (docs/calibration.md §5, rule 1). The exit code is 0 when that check
passes *and* the centre of distortion lands inside the frame, and 1 otherwise - so a
shell can gate on it.

Not measured, and therefore not claimed: whether a corrected frame measures a specimen
any better, what the corrected frame's *scale* is (`c0` is a centre magnification, not a
scale), whether the coefficients survive the mirror test (that lives in the sibling
script), and any micrometre value - no scale is derived here and nothing is wired into
the app.

Known, measured, and deliberately **not** fixed here
----------------------------------------------------
On this project's photos (docs/calibration.md §0) two things get in the flow's way, and
both are the *target*, not the library:

* printed graph paper is one lattice with a heavier line every fifth cell, while the
  flow assumes one uniformly weighted family - discorpy says so itself, once per fit,
  with "Check if there is any missing grouped line";
* the phone sees the eyepiece's ocular ring around the paper, and the cross-point
  detector reads that hard edge as one enormous line. Measured on this project's photo:
  the slope/distance estimator is thrown to its **+30 deg search limit** (against a true
  -9.00 deg tilt), and the cross-point call then raises discorpy's own "Row index is out
  of range" - so on that photo the reference flow does not get past step 3 at all.

Both are exactly what the sibling script's preprocessing removes, and both are why its
numbers and these differ.

Two things the reference flow turns out not to do as written
------------------------------------------------------------
`proc.regenerate_grid_points_parabola` returns **float32** arrays, and discorpy's own
`post.unwarp_line_backward` - which unwraps *lines* by solving ``rd = ru * fact(ru)`` with
``scipy.optimize.minimize`` - is then a **silent no-op**: the numerical gradient comes
back non-finite at float32 precision, the optimiser stops at its starting point, and the
function returns its input (scipy says so only as a RuntimeWarning). Measured on this
project's synthetic target with discorpy 1.7.0 / scipy 1.18.1: a point 400 px from the
centre moves **0.00 px** in float32 and **6.25 px** in float64, and the "residual after
unwarping" reads **9.01 px** against **0.20 px**. So the lines are cast to ``float64``
before that one call (`_float64`) - discorpy's model and discorpy's optimiser, only the
dtype is ours - because without the cast the verdict below is wrong on *every* input,
including a target the library can handle.

And `post.check_distortion`, the library's own verdict and the one this script reports,
cannot tell a *degenerate* fit from a bad one: hand the flow a grid with no distortion
left to fit - this probe's own corrected output, say - and it still fits coefficients, the
centre of distortion lands outside the frame, and the check just reports "distortion".
Measured on that round trip: centre **(192.6, -101.3)**, ``c0`` = **1.884**, residual
after unwarping **116.6 px**. So that case is named separately here rather than quoted as
a verdict about a model that does not exist.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Everything heavy is imported lazily, so ``--help`` works without the CV stack and
#: the failure names the command that fixes it.
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
    except ImportError as error:
        raise SystemExit(f"{INSTALL}\n  ({error})")
    return losa, prep, lprep, proc, post


#: The arguments discorpy's line-pattern example sets explicitly, with the example's
#: values. Anything not listed stays the library's own default, and ``radius`` stays
#: ``None`` on purpose: the example passes no radius, and the library's defaults differ
#: between the two jobs it feeds (9 px for the slope/distance search, 11 px for the
#: cross-point detector) - a single number here would collapse two defaults into one.
REFERENCE = {"roi": 0.3, "coef": 5, "ratio": 0.5, "sensitive": 0.1, "residual": 2.0,
             "radius": None, "perspective": True}


def _load(path):
    """Step 1 - the reference's own loader, with a named failure when it cannot read it.

    ``losa.load_image`` reads through PIL and averages the channels, which covers what a
    phone and this app write (PNG, JPEG, TIFF). It does **not** decode ``.HEIC``/``.HEIF``
    - the default format of the phone this project photographs with - so the failure
    names the fix instead of returning a mystery.
    """
    losa, *_ = _require_discorpy()
    try:
        mat = losa.load_image(str(path))
    except (ValueError, OSError) as error:
        raise SystemExit(
            f"discorpy's loader could not read {path}: {error}\n"
            "  it goes through PIL, which does not decode .HEIC/.HEIF. Convert first:\n"
            f"  sips -s format png {path} --out {path.with_suffix('.png')}\n"
            "  (daphnia.vision.images.load_image does exactly that, via sips on macOS)")
    if mat.ndim != 2:
        raise SystemExit(f"{path} loaded as {mat.shape} - the line-pattern flow needs a "
                         "2D (grey) array; losa.load_image averages colour for you only "
                         "when called with its default average=True")
    return mat


def _residual(post, residual_hor, residual_ver):
    """discorpy's own sub-pixel verdict on a pair of residual arrays, plus its numbers.

    ``post.check_distortion`` answers "is there still distortion": True when more than
    15 % of the points sit further than 1 px from their fitted line. That is the
    library's definition of "the correction is not good enough", quoted rather than
    re-derived; ``over_1px_pct`` is the same fraction printed so a reader can see how far
    from the bar a run landed without reading it off the plots.
    """
    import numpy as np

    values = np.concatenate([np.asarray(residual_hor)[:, 1],
                             np.asarray(residual_ver)[:, 1]])
    return {"max_px": float(values.max()), "mean_px": float(values.mean()),
            "over_1px_pct": float(100.0 * float((values > 1.0).mean())),
            "passed": not (post.check_distortion(residual_hor)
                           or post.check_distortion(residual_ver))}


def _float64(lines):
    """The regenerated lattice in ``float64`` - the dtype discorpy's own unwarp needs.

    ``proc.regenerate_grid_points_parabola`` returns ``float32`` arrays, and
    ``post.unwarp_line_backward`` - which unwraps *lines* by solving ``rd = ru *
    fact(ru)`` with ``scipy.optimize.minimize`` - is then a silent no-op: the numerical
    gradient is non-finite at float32 precision, the optimiser stops at its starting
    point and the function hands back its input. Measured (discorpy 1.7.0, scipy
    1.18.1), one point 400 px from the centre: 0.00 px of movement in float32 against
    6.25 px in float64, and the "residual after unwarping" reads 9.01 px where float64
    gives 0.20 px. The model and the optimiser are still discorpy's - only the dtype is
    ours - and without this cast the residual after unwarping measures nothing at all.
    """
    import numpy as np

    return np.asarray(lines, dtype=np.float64)


def _run_flow(mat, options, run_dir, say=print):
    """Steps 2-8 of the reference, in its order, with its arguments, on ``mat``.

    ``mat`` is the frame exactly as ``losa.load_image`` returned it - nothing is masked,
    equalised, rotated or resampled before the library sees it, which is the whole point
    of this script. Every plot and image the reference example writes is written here
    too, into ``run_dir``.

    Returns the numbers discorpy produced, for the JSON beside them. Raises
    ``SystemExit`` with a named reason when the frame cannot yield a model: a silent
    empty result would be blamed on the photo, which is the mistake this pair of scripts
    exists to prevent (mlContext.md §2, law 8).
    """
    import numpy as np

    losa, prep, lprep, proc, post = _require_discorpy()
    base = str(run_dir)
    height, width = mat.shape
    radius = {} if options["radius"] is None else {"radius": int(options["radius"])}

    say("2-> Slope and distance between the printed lines "
        "(lprep.calc_slope_distance_{hor,ver}_lines)")
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(
        mat, ratio=options["roi"], sensitive=options["sensitive"], **radius)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(
        mat, ratio=options["roi"], sensitive=options["sensitive"], **radius)
    say(f"    horizontal  slope {slope_hor:+.5f} rad "
        f"({np.degrees(np.arctan(slope_hor)):+.2f} deg), "
        f"distance {dist_hor:.3f} px")
    say(f"    vertical    slope {slope_ver:+.5f} rad "
        f"({np.degrees(np.arctan(slope_ver)):+.2f} deg), "
        f"distance {dist_ver:.3f} px")
    if not (np.isfinite(dist_hor) and np.isfinite(dist_ver)
            and dist_hor > 0 and dist_ver > 0):
        raise SystemExit(
            "no line family at all: a distance that is not a positive finite number "
            f"({dist_hor!r} / {dist_ver!r}). Try --roi or --radius, or the prepared "
            "frame in scripts/calibration_probe.py")

    say("3-> Cross points: where the generated lines meet the printed ones "
        "(lprep.get_cross_points_{hor,ver}_lines)")
    points_hor = lprep.get_cross_points_hor_lines(
        mat, slope_ver, dist_ver, ratio=options["ratio"],
        sensitive=options["sensitive"], **radius)
    points_ver = lprep.get_cross_points_ver_lines(
        mat, slope_hor, dist_hor, ratio=options["ratio"],
        sensitive=options["sensitive"], **radius)
    say(f"    {len(points_hor)} points on the horizontal family, "
        f"{len(points_ver)} on the vertical")
    if len(points_hor) == 0 or len(points_ver) == 0:
        raise SystemExit(
            "no cross point was found at all, so discorpy's grouping would raise on an "
            "empty input. The usual cause on this project's photos is the search radius "
            "against the printed lattice: try --radius 11 (the measured value for graph "
            "paper), or the prepared frame in scripts/calibration_probe.py")
    losa.save_plot_points(f"{base}/extracted_hor_points.png", points_hor, height, width)
    losa.save_plot_points(f"{base}/extracted_ver_points.png", points_ver, height, width)

    say("4-> Group the points into lines, then drop the residual ones "
        "(prep.group_dots_{hor,ver}_lines, prep.remove_residual_dots_{hor,ver})")
    lines_hor = prep.group_dots_hor_lines(points_hor, slope_hor, dist_hor)
    lines_ver = prep.group_dots_ver_lines(points_ver, slope_ver, dist_ver)
    lines_hor = prep.remove_residual_dots_hor(lines_hor, slope_hor, options["residual"])
    lines_ver = prep.remove_residual_dots_ver(lines_ver, slope_ver, options["residual"])
    say(f"    {len(lines_hor)} horizontal and {len(lines_ver)} vertical lines kept")
    if len(lines_hor) < 6 or len(lines_ver) < 6:
        raise SystemExit(
            f"only {len(lines_hor)} horizontal and {len(lines_ver)} vertical lines were "
            "grouped - too few for a model. The flow needs a target that fills the frame "
            "with evenly spaced lines of one weight; this project's graph paper carries "
            "three weights (1/5/10 mm) and its ocular ring reads as one more line")
    losa.save_plot_image(f"{base}/grouped_hor_lines.png", lines_hor, height, width)
    losa.save_plot_image(f"{base}/grouped_ver_lines.png", lines_ver, height, width)

    say("5-> Regenerate the lattice from the parabola fits "
        f"(proc.regenerate_grid_points_parabola, perspective={options['perspective']})")
    lines_hor, lines_ver = proc.regenerate_grid_points_parabola(
        lines_hor, lines_ver, perspective=options["perspective"])
    before_hor = post.calc_residual_hor(lines_hor, 0.0, 0.0)
    before_ver = post.calc_residual_ver(lines_ver, 0.0, 0.0)
    losa.save_residual_plot(f"{base}/residual_before_hor.png", before_hor, height, width)
    losa.save_residual_plot(f"{base}/residual_before_ver.png", before_ver, height, width)

    say("6-> Centre of distortion (proc.find_cod_coarse)")
    xcenter, ycenter = proc.find_cod_coarse(lines_hor, lines_ver)
    inside = bool(0.0 <= xcenter <= width and 0.0 <= ycenter <= height)
    say(f"    x {xcenter:.2f} px, y {ycenter:.2f} px, in a {width} x {height} frame"
        + ("" if inside else "   <-- OUTSIDE the frame: a grid with no distortion left "
                             "to fit gives exactly this"))

    say("7-> Radial coefficients (proc.calc_coef_backward), then the residual after "
        "unwarping")
    coefficients = proc.calc_coef_backward(lines_hor, lines_ver, xcenter, ycenter,
                                          options["coef"])
    say("    " + ", ".join(f"c{i}={value:+.6e}" for i, value in enumerate(coefficients)))
    say("    note: the regenerated lattice comes back from discorpy as float32, and on "
        "that dtype its own unwarp_line_backward does *nothing* (measured: 0.00 px of "
        "movement, against 6.25 px in float64), so the lines are cast to float64 first "
        "- see _float64")
    unwarped_hor = post.unwarp_line_backward(_float64(lines_hor), xcenter, ycenter,
                                             coefficients)
    unwarped_ver = post.unwarp_line_backward(_float64(lines_ver), xcenter, ycenter,
                                             coefficients)
    after_hor = post.calc_residual_hor(unwarped_hor, xcenter, ycenter)
    after_ver = post.calc_residual_ver(unwarped_ver, xcenter, ycenter)
    losa.save_residual_plot(f"{base}/residual_after_hor.png", after_hor, height, width)
    losa.save_residual_plot(f"{base}/residual_after_ver.png", after_ver, height, width)

    say("8-> Apply the correction to the image (post.unwarp_image_backward) and save "
        "the coefficients (losa.save_metadata_txt)")
    corrected = post.unwarp_image_backward(mat, xcenter, ycenter, coefficients)
    losa.save_image(f"{base}/corrected_image.tif", corrected)
    losa.save_image(f"{base}/difference.tif", mat - corrected)
    losa.save_metadata_txt(f"{base}/coefficients.txt", xcenter, ycenter, coefficients)

    return {"xcenter": float(xcenter), "ycenter": float(ycenter),
            "centre_inside_frame": inside,
            "coefficients": [float(value) for value in coefficients],
            "line_distance_px": [float(dist_hor), float(dist_ver)],
            "line_slope_rad": [float(slope_hor), float(slope_ver)],
            "lines": [len(lines_hor), len(lines_ver)],
            "points": [len(points_hor), len(points_ver)],
            "residual_before": _residual(post, before_hor, before_ver),
            "residual_after": _residual(post, after_hor, after_ver),
            "notes": ["the residual after unwarping was measured on a float64 copy of "
                      "the regenerated lattice: on discorpy's own float32, its "
                      "unwarp_line_backward is a silent no-op (see _float64)"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run discorpy's reference line-pattern flow, unmodified, on one "
                    "photo.")
    parser.add_argument("image", help="a photo of the printed line/grid target")
    parser.add_argument("--out", default=None,
                        help="output folder (default: recordings/discorpy_probe); one "
                             "sub-folder per photo is created inside it")
    parser.add_argument("--coef", type=int, default=REFERENCE["coef"],
                        help="discorpy: number of radial coefficients "
                             f"({REFERENCE['coef']}, the example's value; measured on "
                             "this project's target, 3 is the value that passes "
                             "discorpy's own sub-pixel check)")
    parser.add_argument("--roi", type=float, default=REFERENCE["roi"],
                        help="discorpy: ROI ratio around the middle of the frame, used "
                             f"for the slope/distance estimate ({REFERENCE['roi']}, the "
                             "library's default; the example does not set it)")
    parser.add_argument("--ratio", type=float, default=REFERENCE["ratio"],
                        help="discorpy: how densely the profiles that look for cross "
                             "points are placed, as a fraction of the line distance "
                             f"({REFERENCE['ratio']}, the example's value)")
    parser.add_argument("--sensitive", type=float, default=REFERENCE["sensitive"],
                        help="discorpy: peak sensitivity, smaller = more sensitive "
                             f"({REFERENCE['sensitive']}, the example's value)")
    parser.add_argument("--residual", type=float, default=REFERENCE["residual"],
                        help="discorpy: drop points further than this many px from a "
                             f"line's parabola ({REFERENCE['residual']}, the example's "
                             "value; the library's default is 2.5)")
    parser.add_argument("--radius", type=int, default=REFERENCE["radius"],
                        help="discorpy: line search radius in px. The example leaves it "
                             "unset, so the library's two own defaults stand (9 for the "
                             "slope/distance search, 11 for the cross points); --radius "
                             "11 was the measured value for this project's paper")
    parser.add_argument("--perspective", action=argparse.BooleanOptionalAction,
                        default=REFERENCE["perspective"],
                        help="discorpy: correct the target's tilt from the parabola "
                             f"coefficients ({REFERENCE['perspective']}, the example's "
                             "value)")
    args = parser.parse_args(argv)
    options = {key: getattr(args, key) for key in REFERENCE}

    from daphnia import paths
    from daphnia.vision.calibration import utc_now

    source = Path(args.image).expanduser()
    if not source.exists():
        raise SystemExit(f"no such file: {source}")
    out_root = (Path(args.out) if args.out
                else paths.CALIBRATION_IMAGES_DIR.parent / "discorpy_probe")
    run_dir = out_root / source.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    print("DaphniaScope - discorpy probe: its reference line-pattern flow, unmodified")
    print(f"  photo                        {source}")
    print(f"  output                       {run_dir}")
    print("  arguments                    "
          + ", ".join(f"{key}={value}" for key, value in options.items()))
    print()
    print(f"1-> Load image: {source}")
    mat = _load(source)
    print(f"    {mat.shape[1]} x {mat.shape[0]} px, grey levels "
          f"{mat.min():.0f} - {mat.max():.0f}")

    model, reason, hint = None, None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = _run_flow(mat, options, run_dir)
        except SystemExit as error:              # a named refusal from the flow
            reason = str(error)
        except Exception as error:               # noqa: BLE001 - discorpy raises wildly
            reason = f"{type(error).__name__}: {error}"
            hint = ("discorpy cannot read this frame as it is - on this project's photos "
                    "the ocular ring around the paper is what breaks it. Try --radius 11 "
                    "(the value measured for this paper), or run "
                    "scripts/calibration_probe.py, which runs this same flow on a frame "
                    "whose field stop is masked, whose line weights are equalised and "
                    "whose lattice is straightened")
        # discorpy warns "Check if there is any missing grouped line" once per fit that
        # suspects the family is not a uniform lattice. That is evidence about the
        # *target*, so it is a reported number, not screen noise.
        target_warnings = sum(1 for item in caught
                              if "missing grouped line" in str(item.message))
    if reason is not None:
        print(f"\n  no model                     {reason}")
    if hint is not None:
        print(f"  what to try                  {hint}")

    print()
    print("9-> discorpy's own verdict (post.check_distortion, quoted not re-derived)")
    if model is None:
        verdict = ("no model was fitted from this photo, so the distortion here is "
                   "unmeasured - it is not 0 %. Nothing was corrected and nothing was "
                   "stored. scripts/calibration_probe.py runs the *same* flow on a frame "
                   "whose field stop, line weights and tilt have been prepared, which is "
                   "the comparison that says whether the library or the photo is the "
                   "limit.")
    else:
        before, after = model["residual_before"], model["residual_after"]
        print(f"  residual before unwarping    {before['mean_px']:.3f} px mean, "
              f"{before['max_px']:.3f} px worst, {before['over_1px_pct']:.1f} % of the "
              "points over 1 px")
        print(f"  residual after unwarping     {after['mean_px']:.3f} px mean, "
              f"{after['max_px']:.3f} px worst, {after['over_1px_pct']:.1f} % of the "
              "points over 1 px")
        if not model["centre_inside_frame"]:
            verdict = (f"the centre of distortion landed outside the frame "
                       f"({model['xcenter']:.1f}, {model['ycenter']:.1f} px in a "
                       f"{mat.shape[1]} x {mat.shape[0]} frame), which is what a grid "
                       "with no distortion left to fit produces: the fit is degenerate "
                       "and the residual above describes nothing. discorpy's own "
                       "sub-pixel check does not catch that, which is why it is named "
                       "here - read the centre before believing the verdict.")
        elif after["passed"]:
            verdict = ("the correction passes discorpy's own sub-pixel check on this "
                       "photo, so the coefficients describe a distortion. Not measured, "
                       "and not claimed: whether a corrected frame measures a specimen "
                       "any better, what the corrected frame's scale is (c0 is a centre "
                       "magnification, not a scale), and whether the coefficients "
                       "survive the mirror test.")
        else:
            verdict = (f"{after['over_1px_pct']:.1f} % of the points stay more than 1 px "
                       f"from their line (worst {after['max_px']:.3f} px), which is "
                       "discorpy's own bar for \"the correction is not good enough\". "
                       "The coefficients are then not a distortion measurement and are "
                       "not stored-ready. The frame was read as it is - no masking, no "
                       "equalisation, no alignment - so the next step is the comparison: "
                       "scripts/calibration_probe.py runs this same flow on a prepared "
                       "frame.")
    print(f"  verdict                      {verdict}")
    print(f"  discorpy warnings            {target_warnings} x \"check if there is any "
          "missing grouped line\" (0 = the lattice looks uniform to the library; more = "
          "the target mixes line weights)")
    print("  not measured here            any micrometre value: this probe reports "
          "pixels, derives no scale, and is wired into nothing")

    report = {"tool": "scripts/discorpy_probe.py", "measured_at": utc_now().isoformat(),
              "image": str(source), "image_size": [int(mat.shape[1]), int(mat.shape[0])],
              "reference_arguments": options, "model": model,
              "missing_line_warnings": target_warnings, "verdict": verdict}
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
                 and model["residual_after"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

