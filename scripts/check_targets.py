#!/usr/bin/env python3
"""Does the printed target measure what it claims? - measured, not assumed.

    make targets          # build assets/targets/discorpy_targets.pdf
    python scripts/check_targets.py [pdf] [--dpi 254]

Why this exists
---------------
A calibration target is a *length standard made of ink*: every millimetre of it
becomes `px_per_um` in the app (`docs/calibration.md`), so "the LaTeX looks
right" is not a statement anyone can defend. This script renders the built PDF
at a known resolution, measures the pattern in millimetres, and prints a verdict
per claim:

    claim                        expected     measured     verdict
    page size (PDF MediaBox)     210 x 297 mm 210.000 ...  PASS
    p1 lattice pitch (dots)      2.000 mm     ...
    p1 first marker, absolute    6.000 mm     ...
    verification rule length     100.000 mm   ...

It is the same discipline the rest of the repo uses for a metric: the number
carries its frame of reference (which PDF, which page, which dpi, which
renderer), and it says what it does *not* cover. What it cannot measure: the
paper's own shrinkage and the printer's ink spread - those live on the physical
sheet, which is why the sheet carries its own rule and a date line.

Dependencies: `pdftoppm` (poppler, GPL-2, invoked as a program - nothing is
linked) and OpenCV, which the app already requires. No new Python dependency.

The expected geometry mirrors section 2 of assets/targets/discorpy_targets.tex;
if that file's knobs change, update the constants below too - a verifier with
stale expectations is worse than none.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PDF_DEFAULT = Path(__file__).resolve().parents[1] / "assets/targets/discorpy_targets.pdf"

#: 254 dpi renders at exactly 10 pixels per millimetre, which keeps every
#: tolerance below readable as a fraction of a pixel.
DPI_DEFAULT = 254

#: The document's own knobs (assets/targets/discorpy_targets.tex section 2).
TILE_MM = 100.0
PITCH_MM = 2.0
MARKER_MM = 0.8
MARGIN_MM = 5.0
TILES_X, TILES_Y = 2, 2
A4_MM = (210.0, 297.0)

#: Tolerances, in millimetres. A rendered centroid is good to ~0.05 px at this
#: dpi (0.005 mm), so these are generous by an order of magnitude and still catch
#: every failure that matters: a scaled page (1 % = 1 mm of a 100 mm tile), a
#: wrong pitch, or a marker written on the cut line (half a pitch = 1 mm).
TOL_PITCH = 0.010
TOL_ABS = 0.050
TOL_SIZE = 0.030


def _need(tool: str) -> None:
    if shutil.which(tool) is None:
        raise SystemExit(
            f"{tool} is not installed, so the target cannot be rendered and "
            f"measured. Install it (macOS: `brew install poppler`)."
        )


def render_page(pdf: Path, page: int, dpi: int, tmp: Path) -> np.ndarray:
    """Render one page to a 2D greyscale array (0..255).

    `pdftoppm -gray` writes a binary PGM (P5): a short header, then the pixels.
    That is parseable with the standard library, so this script needs no PDF or
    imaging dependency beyond the OpenCV the app already has.
    """
    _need("pdftoppm")
    stem = tmp / f"page{page}"
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-gray", "-f", str(page), "-l", str(page),
         str(pdf), str(stem)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    written = sorted(tmp.glob(f"page{page}*.pgm"))
    if not written:
        raise SystemExit(f"pdftoppm rendered nothing for page {page} of {pdf}")
    data = written[0].read_bytes()
    if not data.startswith(b"P5"):
        raise SystemExit(f"{written[0]} is not a binary PGM; unexpected renderer output")
    parts = data.split(maxsplit=4)
    width, height = int(parts[1]), int(parts[2])
    pixels = np.frombuffer(parts[4][: width * height], dtype=np.uint8)
    return pixels.reshape(height, width)


def page_size_mm(pdf: Path) -> tuple[float, float]:
    """The PDF's own MediaBox in millimetres - the sheet the printer is *told*
    about, which is a different thing from what the printer then does with it."""
    match = re.search(
        rb"/MediaBox\s*\[\s*([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)",
        pdf.read_bytes(),
    )
    if match is None:
        raise SystemExit(f"{pdf} has no /MediaBox - is it a PDF?")
    _, _, width, height = (float(g) for g in match.groups())
    return width * 25.4 / 72.0, height * 25.4 / 72.0


def _lattice(values: np.ndarray, gap: float) -> np.ndarray:
    """Cluster 1-D positions into lattice lines: sort, split at gaps > `gap`,
    average each cluster. Robust to a few missing or noisy points, which is what
    a real print has - a plain `unique()` is not."""
    if values.size == 0:
        return values
    ordered = np.sort(values)
    breaks = np.nonzero(np.diff(ordered) > gap)[0]
    return np.array([group.mean() for group in np.split(ordered, breaks + 1)])


def _blobs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroids (px) and areas (px^2) of the connected dark regions."""
    import cv2  # lazy, as elsewhere in the repo: the domain layer imports without it

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    keep = np.arange(1, count)  # label 0 is the background
    return centroids[keep], stats[keep, cv2.CC_STAT_AREA].astype(float)


def _tile_window(page: np.ndarray, col: int, row: int, mm: float) -> np.ndarray:
    """One tile's pixels, from the layout the target promises. Tiles are
    adjacent and every marker is inset by half a pitch, so a window on the tile
    boundary holds exactly this tile's pattern and nothing of its neighbour."""
    x0 = int(round((MARGIN_MM + col * TILE_MM) * mm))
    y0 = int(round((MARGIN_MM + row * TILE_MM) * mm))
    size = int(round(TILE_MM * mm))
    return page[y0:y0 + size, x0:x0 + size]


def measure_marker_page(page: np.ndarray, dpi: int) -> dict:
    """Dots and squares: the same lattice, measured the same way.

    The number that matters most is the *absolute* position of the first marker:
    the pitch is a ratio and survives a uniformly scaled page, but
    `MARGIN + PITCH/2` can only be right if the sheet is placed at page
    coordinates, in millimetres, exactly as the file says.
    """
    mm = 25.4 / dpi
    counts, pitches, sizes, firsts = [], [], [], []
    for row in range(TILES_Y):
        for col in range(TILES_X):
            window = _tile_window(page, col, row, mm)
            centroids, areas = _blobs(window < 128)
            counts.append(int(areas.size))
            if areas.size == 0:
                continue
            xs_mm, ys_mm = centroids[:, 0] * mm, centroids[:, 1] * mm
            pitches += list(np.diff(_lattice(xs_mm, PITCH_MM / 2)))
            pitches += list(np.diff(_lattice(ys_mm, PITCH_MM / 2)))
            # Equivalent-circle diameter from the blob's area: exact for a dot,
            # and the honest single number for a rounded printed square.
            sizes += list(2 * np.sqrt(areas / np.pi) * mm)
            # The first marker of this tile, in the sheet's coordinates.
            nearest = np.argmin(xs_mm + ys_mm)
            firsts.append((MARGIN_MM + col * TILE_MM + xs_mm[nearest],
                           MARGIN_MM + row * TILE_MM + ys_mm[nearest]))
    return {
        "markers_per_tile": counts,
        "pitch_mm": float(np.mean(pitches)) if pitches else float("nan"),
        "pitch_spread_mm": (float(np.max(pitches) - np.min(pitches))
                            if pitches else float("nan")),
        "size_mm": float(np.mean(sizes)) if sizes else float("nan"),
        "first_marker_mm": np.array(firsts).min(axis=0) if firsts else np.zeros(2),
    }


def measure_cells(page: np.ndarray, dpi: int) -> dict:
    """The checkerboard: its black cells merge into one blob, so it is measured
    by run lengths along a row through the middle of a tile. The cell size is the
    lattice pitch of the crossing pattern discorpy fits from it, so it is the
    number to type into the Calibrate screen."""
    mm = 25.4 / dpi
    runs, black = [], []
    for row in range(TILES_Y):
        for col in range(TILES_X):
            window = _tile_window(page, col, row, mm)
            line = window[window.shape[0] // 2] < 128
            edges = np.nonzero(np.diff(line.astype(np.int8)))[0] + 1
            bounds = np.concatenate([[0], edges, [line.size]])
            lengths = np.diff(bounds) * mm
            # The first and last run can be clipped by the window's rounding.
            runs += list(lengths[1:-1]) if lengths.size > 2 else list(lengths)
            black.append(float(line.mean()))
    per_tile = len(runs) + 2 if runs else 0
    return {
        "cell_mm": float(np.mean(runs)) if runs else float("nan"),
        "cell_spread_mm": float(np.max(runs) - np.min(runs)) if runs else float("nan"),
        "runs_per_row": round(per_tile / (TILES_X * TILES_Y)),
        "black_fraction": float(np.mean(black)) if black else float("nan"),
    }


def measure_rule(page: np.ndarray, dpi: int) -> tuple[float, float]:
    """The verification rule's length and its left end, in mm from the sheet's
    top-left corner. Found as the longest horizontal dark run in the band where
    the footer draws it (below the tile block, 20 mm down)."""
    mm = 25.4 / dpi
    top = int(round((MARGIN_MM + TILE_MM * TILES_Y + 20) * mm))
    bottom = int(round((MARGIN_MM + TILE_MM * TILES_Y + 32) * mm))
    best = (0.0, 0.0)
    for y in range(top, min(bottom, page.shape[0])):
        dark = np.nonzero(page[y] < 128)[0]
        if dark.size < 2:
            continue
        length = (dark[-1] - dark[0]) * mm
        if length > best[0]:
            best = (length, float(dark[0]) * mm)
    return best


def _row(claim: str, expected: str, measured: str, ok: bool) -> str:
    return f"{claim:<34} {expected:<14} {measured:<16} {'PASS' if ok else 'FAIL'}"

