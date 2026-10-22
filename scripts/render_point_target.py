#!/usr/bin/env python3
"""Render an artificial **dot target as a phone sees it through an eyepiece**, and measure it.

    conda run -n z-app python scripts/render_point_target.py
    conda run -n z-app python scripts/render_point_target.py --out /tmp/point.png --radius 260

Why this exists
---------------
`assets/targets/` holds printable dot grids, and `daphnia/vision/points.py` is the
estimator that reads one into a scale. But a *printed* sheet is only measurable once it is
printed, taped to the stage and photographed - which is a slow way to find out whether the
detector answers. This script renders the *same* lattice the way the optics will show it -
round **barrel-warped dots inside the eyepiece's circular field stop**, with the dark
surround and the brightness falling off towards the edge - and writes it to disk with its
ground truth beside it. That gives the Calibrate screen a photo to open and the point
detector a case with a *known* answer.

What it is, and what it is not
------------------------------
The distortion is a **known** radial model ``r_u = r * (1 + k1 * r^2)``: the ground truth is
the clean lattice's pitch, and the local spacing at the edge is ``~ pitch * (1 - k1 * R^2)``.
This is what `scripts/calibration_probe.py` does for a line grid, and it is the ML law "a
test set that defines done" (memory-bank/mlContext.md §2) applied to a target. It is
**not** a substitute for a printed sheet: paper stretch, ink spread and the printer's own
scaling are absent here, so a number measured on this PNG is an *upper* bound on what the
real target can do, and is reported as such.

The output is written next to a ``.json`` carrying every knob that produced it - corpus,
frame, pitch, distortion, vignette - so a measurement made on the PNG can be reported with
its provenance (mlContext §8) instead of as a bare figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# The app package lives at the repository root; a script run by path only has its own
# directory on `sys.path`, so the root is added explicitly (as the other scripts do).
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DIR = REPO_ROOT / "assets/targets"


def ocular_point_grid(height=720, width=1280, centre=(640.0, 360.0), radius=330.0,
                      pitch=27.0, pattern="dots", dot_ratio=0.42, distortion=1.0e-7,
                      angle_deg=8.0, vignette=0.25, paper=0.92, ink=0.12, surround=0.03,
                      noise=0.015, blur_px=1.0, seed=0):
    """A printed **point target** through a round eyepiece, as ``(uint8, truth)``.

    ``pattern`` is what a reference point *is*: ``"dots"`` renders round dots at the
    lattice's nodes, ``"checkerboard"`` renders the cells whose **corners** are those nodes.
    ``pitch`` is the reference-point spacing **at the centre of the field**, in clean pixels
    - the number the detector must recover. ``dot_ratio`` is a dot's diameter as a fraction
    of the pitch (dots only). ``distortion`` is the barrel coefficient ``k1`` (``0`` renders
    a perfect lattice, which is the clean case and the detector's oracle). ``vignette`` is
    how much of the paper's brightness is lost from the centre to the rim; beyond ``radius``
    the frame is the black field stop, which is what the detector masks.
    """
    import cv2

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    dx = xx - centre[0]
    dy = yy - centre[1]
    r = np.hypot(dx, dy)

    # The sampling map: the output pixel at radius r shows the lattice at radius ru, so the
    # pattern is compressed radially (barrel). Near r=0 the map is the identity, which is
    # why the *centre* pitch is the ground truth and the edge spacing is the distorted one.
    ru = r * (1.0 + distortion * r * r)
    safe = np.maximum(r, 1e-9)
    sx = centre[0] + dx / safe * ru
    sy = centre[1] + dy / safe * ru

    # Rotate into the lattice's own frame (the target is not square to the frame: measured
    # on this rig at ~9 deg) and measure the distance to the nearest printed dot.
    theta = np.radians(angle_deg)
    a = (sx - centre[0]) * np.cos(theta) + (sy - centre[1]) * np.sin(theta)
    b = -(sx - centre[0]) * np.sin(theta) + (sy - centre[1]) * np.cos(theta)

    if pattern == "checkerboard":
        # The nodes are the cell *corners*; the phase puts a cell centre on the field centre.
        cell_a = np.floor(a / pitch + 0.5)
        cell_b = np.floor(b / pitch + 0.5)
        ink_mask = ((cell_a + cell_b) % 2.0) != 0.0
    else:
        # Round dots at the nodes.
        du = np.abs((a + pitch / 2.0) % pitch - pitch / 2.0)
        dv = np.abs((b + pitch / 2.0) % pitch - pitch / 2.0)
        ink_mask = np.hypot(du, dv) <= 0.5 * dot_ratio * pitch

    # Brightness: paper inside the field stop, with a radial falloff; black outside it.
    gain = 1.0 - vignette * np.clip(r / radius, 0.0, 1.0) ** 2
    frame = np.where(ink_mask, ink, paper) * gain
    outside = r > radius
    frame = np.where(outside, surround, frame)

    frame = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), blur_px)
    if noise:
        frame = frame + np.random.default_rng(seed).normal(0.0, noise, frame.shape)
    frame = np.clip(frame, 0.0, 1.0)
    image = np.round(frame * 255.0).astype(np.uint8)

    edge_pitch = pitch * (1.0 - distortion * radius * radius)
    truth = {
        "name": "ocular_point_grid" if pattern != "checkerboard" else "ocular_checkerboard",
        "kind": "rendered",
        "pattern": pattern,
        "height": int(height),
        "width": int(width),
        "centre_px": [float(centre[0]), float(centre[1])],
        "field_stop_radius_px": float(radius),
        "pitch_px": float(pitch),
        "pitch_um": 1000.0,
        "px_per_um": float(pitch) / 1000.0,
        "dot_ratio": float(dot_ratio),
        "distortion_k1": float(distortion),
        "predicted_edge_pitch_px": float(edge_pitch),
        "predicted_drift_pct": float(abs(edge_pitch - pitch) / pitch * 100.0),
        "tilt_deg": float(angle_deg),
        "vignette": float(vignette),
        "blur_px": float(blur_px),
        "noise_sigma": float(noise),
        "note": "rendered by scripts/render_point_target.py - the ground truth of the tests",
    }
    return image, truth


def measure_and_report(image, truth):
    """Read the rendered frame with the shipped dot detector and print the verdict.

    The number this script exists to state is the **error**: the detector's period against
    the lattice's own pitch. It is reported with the frame of reference (which PNG, which
    detector, which pitch) so it can be compared with another run - and it says what it does
    *not* cover (the paper and the printer are absent here).
    """
    from daphnia.vision.points import PointsNotFound, measure_points

    pitch = truth["pitch_px"]
    edge = truth["predicted_edge_pitch_px"]
    try:
        estimate = measure_points(image, pitch_um=truth["pitch_um"],
                                  pattern=truth.get("pattern", "dots"))
    except PointsNotFound as error:
        print(f"  detector                    REFUSED: {error}")
        return False

    # The detector reports the field's **consensus** period (the median where it could see),
    # not the centre's - exactly as `vision/lattice.py` does. On a warped frame that period
    # therefore sits between the centre's pitch and the edge's, and the honest check is that
    # it lands in that bracket, not that it equals the centre. Claiming otherwise would be a
    # number whose frame of reference is false (`memory-bank/mlContext.md` law 5).
    low, high = sorted((pitch, edge))
    tolerance = 0.01 * pitch
    inside = low - tolerance <= estimate.period_px <= high + tolerance
    stop = estimate.field_stop
    print(f"  known pitch                 centre {pitch:.3f} px, edge {edge:.3f} px")
    print(f"  measured pitch              {estimate.period_px:.3f} px "
          f"({estimate.points} dots)")
    print(f"  measured drift              {estimate.spread_pct:.2f} % "
          f"(known centre-to-edge {truth['predicted_drift_pct']:.2f} %)")
    print(f"  field stop                  " + (
        f"({stop.centre_x:.0f}, {stop.centre_y:.0f}) r {stop.radius:.0f} px"
        if stop else "none"))
    print(f"  verdict                     {'PASS' if inside else 'FAIL'} "
          f"(the period must fall inside [centre, edge] +-1 %)")
    print("  not covered                 paper stretch, ink spread and the printer's own "
          "scaling - a real sheet measures worse than this PNG")
    return inside


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the PNG (a .json lands beside it; defaults to "
                             "assets/targets/ocular_<pattern>_1000um.png)")
    parser.add_argument("--pattern", choices=("dots", "checkerboard"), default="dots",
                        help="a reference point is a round dot, or a checkerboard corner")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--radius", type=float, default=330.0,
                        help="field-stop radius in px")
    parser.add_argument("--pitch", type=float, default=27.0,
                        help="spacing at the field's centre, in px")
    parser.add_argument("--distortion", type=float, default=1.0e-7,
                        help="barrel coefficient k1 (0 = a perfect lattice)")
    parser.add_argument("--tilt", type=float, default=8.0, help="the target's tilt, in deg")
    parser.add_argument("--vignette", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import cv2

    stem = "ocular_point_grid" if args.pattern != "checkerboard" else "ocular_checkerboard"
    out = args.out or (DEFAULT_DIR / f"{stem}_1000um.png")

    image, truth = ocular_point_grid(
        height=args.height, width=args.width, radius=args.radius, pitch=args.pitch,
        pattern=args.pattern, distortion=args.distortion, angle_deg=args.tilt,
        vignette=args.vignette, seed=args.seed)
    truth["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out} ({args.width} x {args.height}) pattern={args.pattern}")
    print(f"wrote {sidecar} (the ground truth of this render)")
    okay = measure_and_report(image, truth)
    return 0 if okay else 1


if __name__ == "__main__":
    sys.exit(main())
