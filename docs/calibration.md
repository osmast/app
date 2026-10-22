# Calibration — the scale, and the distortion under it

The guide for the optics side of DaphniaScope: how a grid photo becomes `px_per_um`,
what is *not* measured by that number, and how to measure the rest with
**discorpy**. The code lives in `daphnia/vision/grid.py` (the shipped detector),
`daphnia/vision/calibration.py` (the record) and `daphnia/ui/pages/calibration.py`
(the OPTICS surface). The probe that produced every number below is
`scripts/calibration_probe.py`.

> **Read `memory-bank/mlContext.md` before changing anything that feeds a metric.**
> The scale *is* the metric: it is what turns pixels into micrometres, so the laws
> about provenance, frames of reference and "never report a number whose frame is
> unstated" apply here exactly as they do to a loss.
>
> **`docs/accuracy.md` is the other half of this file.** This document is how the scale is
> *obtained*; that one is the **error budget** — what the scale's uncertainty, the drift, the
> print's tolerance, the pixel quantisation, the model's own error and the repeat spread are
> each worth in µm, how they combine, and which of them the app reports. Read it before
> claiming an accuracy figure for a length.

---

## 0. Verdict — can discorpy calibrate a photo like ours? (read this first)

**Yes — and it now does, on this exact photo, once the *documented* line-pattern flow is
followed.** (The first three sections of this document said "not yet"; they were written
before the third bullet below was understood. The corrections are labelled where they sit.)

```bash
conda run -n z-app python scripts/calibration_probe.py recordings/calibration/PROF_…_133641.png --coef 3
```

| Question | Answer, measured |
| --- | --- |
| **the scale, in the app** | **implemented** — `daphnia/vision/lattice.py`, called by the Calibrate screen (`SR-OPTICS-035/036`). On this rig's own photos: **27.239 px** per cell (10 bands, tilt −9.00°, spread **5.10 %**, statistical 2.49 %), **27.03 / 27.24** on three more, and a named refusal for the two ruler photos and three unusable grid photos. `vision/grid.py` is superseded on this path and `detect_grid` still refuses every one of them |
| **the drift, in the app** | **implemented** as a number separate from the scale's uncertainty: **5.1–6.3 %** across the field on those photos — reported, never rounded into "0 %" |
| **the distortion model** | **not storable yet**: the documented discorpy flow reaches **0.511 px rms** and passes discorpy's own sub-pixel check with 3 coefficients, but it **fails the mirror test** (an exact pixel permutation: coefficients differ by **0.82**, the centre lands **184 px** off, mirrored residual 8.68 px), and its magnitude (+8.1–9.8 % at r = 302) is ~1.5–2x the **5.7 %** measured directly from the cell spacing. `scripts/calibration_probe.py` is where that check lives |
| **a proper target** | single-weight line grid → **0.264 px** rms (0.056 px with Gaussian peaks); dot grid → **0.149 px** rms. Both proven on a known distortion; neither is needed for the scale |

**So the answer to "can we or can we not": we can, on this photo, today.** Three things
made the difference, and the third is the one that mattered most:

1. **the field stop masked as a fitted circle** (Otsu → circle fit → eroded margin), with
   every cross point outside it dropped - see the correction note below;
2. **the lattice straightened** to 0.00 deg, so discorpy's tilted-profile sampling is not
   truncated; and the printed line weights equalised, so all the fine lines are detected;
3. **the documented flow, including `proc.regenerate_grid_points_parabola`** (docs 3.3.4
   step 5) - the step that rebuilds the lattice from the parabola fits. It took the
   residual from **3.18 px to 0.51 px**, and it is the direct answer to *"why points
   instead of lines? the points seem inaccurate"*: the model fits **lines**, and this step
   replaces the noisy point cloud with those lines.

**Two caveats, both measured, both about the *magnitude* rather than the effect:**

* the model's **+8.1–9.8 %** disagrees with the direct spacing measurement's **~5.7 %**
  centre-to-edge - the *sign and shape* agree, the *size* is ~1.5–2x too large, so the
  number to *store* is not settled yet;
* the passing model is **fragile at its edges but repeatable in its middle**: exactly 3
  coefficients (4 gives 2.22 px, 5 gives 1.08 px, both failing), `--perspective` changes
  the curvature 2x without moving the COD, pinning the spacing with `--dist-px 27` is
  **wrong here** (it moves the COD by 78 px and the residual to 2.57 px), and **a smaller
  `--radius` is worse, not better** (measured: radius 4 → 6.25 px, 6 → 1.61 px, 8 →
  7.31 px, 9 → 0.51 px) - the search radius must match the ~10 px sampling interval the
  cross points are taken at, not be reduced for tidiness.

**The points on the photo are clean now** (`*_points.png`): a regular lattice across the
paper, with nothing on the ocular ring and no pile-up along the paper's edge. The last
defect was that boundary: the paper/filled-surround junction is a *step*, any profile
crossing it has a local extremum *there*, and the points landed on that edge. Points are
now kept only `radius + 2` px inside the paper (`--edge-margin` overrides it; 30 px also
passes and gives the same COD).

| question | answer, measured |
| --- | --- |
| does the **preparation** survive a flipped / rotated / re-positioned disc? | **Yes.** On a mirrored copy, an upside-down copy, a 90° rotation and a copy where the disc sits clear of every edge: the circle is found identically (**r 448.7 px** under mirror/vertical-flip/90°; **418.5 px** when the disc is fully inside), and the **alignment self-check** picks the right sign every time (mirrored → **+8.95°** where the original needs −8.95°; a wrong sign is detected because it doubles the tilt) |
| does the **model** come back? | **No.** Residuals: original **0.51–0.68 px**, mirrored **1.18**, upside-down **8.58**, 90°-rotated **4.07**, re-positioned **1.48** |
| the sharpest test — a **mirror**, which is an exact pixel permutation | **FAILS on this photo**: coefficients differ by **0.82**, the centre lands **184 px** from the mirrored expectation (665.0, 316.8), and the mirrored residual is 8.68 px. A model that cannot reproduce itself under a symmetry it must satisfy is not a model to store. `--mirror-check` (on by default) runs it in the same command |

**So: the pipeline is orientation-proof, the *model* is not yet.** The mirror test is the
acceptance test to keep, and it needed no second photo.

> **Corrected 2026-10-19, after looking at `*_points.png`** (the operator's own check -
> it caught two defects in my method, and the documentation you pasted caught a third):
>
> 1. **The field mask was ~60 px too small on the radius** (390 and 381 px against the
>    true **448.7 px**), so real paper along the top - where the disc is **cut off by the
>    frame** - and along the dimmer right side was filled over. That is the "it fails at
>    the top and right edges" an operator sees. Fixed by **Otsu** + a circle fit that
>    ignores the frame border, verified against the raw edges (left x 156-163, right
>    x 1044, bottom y 713 at x=640; all within a pixel).
> 2. **The lattice was left tilted 9 deg**, and discorpy samples *tilted* profiles whose
>    range is truncated to stay inside the image - ~114 px along one side with **no points
>    extracted at all**. Fixed by straightening; measured slope after: **0.00 deg**.
> 3. **My flow was an approximation, not the library's.** I had used the *fisheye*
>    example's grouping plus my own centre-finding, and had skipped
>    `regenerate_grid_points_parabola`. Following docs 3.3.4 instead (neighbour-search
>    grouping, `remove_residual_dots(…, 2.0)`, regenerate, `find_cod_coarse`) is what
>    produced the 0.511 px model - with the *same* preparation as before.

---

## 1. The two questions a calibration photo answers

| Question | Answered by | State today |
| --- | --- | --- |
| **"How many pixels is one printed millimetre?"** | `vision/grid.py`: 2D FFT for the lattice, then a projection + zero-padded DFT for a sub-pixel period | **the detector refuses every photo in this repository** (§3) |
| **"Is that number the same everywhere in the field?"** | nothing today: `distortion_pct` *reports* the centre-vs-edge drift it can detect and **corrects nothing** | **unmeasured** — which is not the same as zero |

The second question is not academic. At 1 mm ~ 18.7 px, a specimen 2 mm long sits on
~37 px of measurement; a 3 % change of scale between the middle and the edge of the
field is 60 µm on that animal — an order of magnitude more than the keypoint model's
error and six times the paper tolerance the record already carries.

## 2. discorpy in one paragraph

**discorpy** (Diamond Light Source, `discorpy` 1.7.0, **Apache-2.0**) calibrates and
corrects radial and perspective distortion of a lens system from a **single**
calibration image, by direct computation rather than iterative optimisation: it
extracts reference points from a line-pattern, dot-pattern or chessboard image, finds
the centre of distortion, and fits a polynomial radial model. It is the same family of
method as the tomography detectors it was written for, so it is used to sub-pixel
accuracy and it reports its own residual. `environment.yml` carries it as an optional
pip dependency; nothing in the application imports it, exactly like PyTorch.

### The flow (discorpy's own line-pattern example, function by function)

Printed grid paper *is* a line pattern, so the flow is the one in
`discorpy/examples/fisheye_calibration_line_pattern.py`, in that order:

```python
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post
import discorpy.util.utility as util

mat = prep.normalization_fft(grey, 10)                       # flat background
slope_h, dist_h = lprep.calc_slope_distance_hor_lines(mat, ratio=0.3, radius=9)
slope_v, dist_v = lprep.calc_slope_distance_ver_lines(mat, ratio=0.3, radius=9)
# note the crossing: horizontal-line points need the *vertical* family's geometry
ph = lprep.get_cross_points_hor_lines(mat, slope_v, dist_v, bgr="bright",
                                      chessboard=False, radius=9, sensitive=0.1)
pv = lprep.get_cross_points_ver_lines(mat, slope_h, dist_h, bgr="bright",
                                      chessboard=False, radius=9, sensitive=0.1)
hl = prep.group_dots_hor_lines_based_polyfit(ph, slope_h, dist_h, ratio=0.1,
                                             num_dot_miss=3, accepted_ratio=0.65,
                                             order=2)
vl = prep.group_dots_ver_lines_based_polyfit(pv, slope_v, dist_v, ratio=0.1,
                                             num_dot_miss=3, accepted_ratio=0.65,
                                             order=2)
hl = prep.remove_residual_dots_hor(hl, slope_h, 3.0)
vl = prep.remove_residual_dots_ver(vl, slope_v, 3.0)
xc, yc = proc.find_center_based_vanishing_points_iteration(hl, vl, iteration=2)
ch, cv = proc.correct_perspective_effect(hl, vl, xc, yc)      # tilt, before radial
coef = proc.calc_coef_backward(ch, cv, xc, yc, num_fact=3)    # the model
corrected = post.unwarp_image_backward(grey, xc, yc, coef, order=1, mode="reflect")
```

`post.calc_residual_hor/ver` + `post.check_distortion` are the **acceptance test**:
a residual over 1 px on more than 15 % of the points means the correction is not good
enough. That is discorpy's own bar, and it is the one to hold it to.

### What the coefficients mean

The fitted model is a **backward** model: for an output (undistorted) point at radius
``ru`` from the centre, the source (distorted) point is at

```
        xd = xc + fact(ru) * xu,        fact(r) = c0 + c1*r + c2*r^2 + ...
```

so the **local magnification** at radius ``r`` — the thing the app actually cares
about, scale-comparisons being local — is the derivative of ``r * fact(r)``:

```
        magnification(r) = fact(r) + r * fact'(r),        magnification(0) = c0
```

The ratio ``magnification(r) / c0`` is the position-dependent error a single-number
calibration carries, and it is what `scripts/calibration_probe.py` prints. ``c0``
itself is only the scale *at the centre*: it is not identifiable against an overall
zoom (as `SR-OPTICS-004` already says, the scale belongs to the optical chain), which
is why a corrected frame needs its own `px_per_um` measured from the corrected grid -
not a coefficient.

---

## 3. What was measured on this project's own photo (2026-10-19, corrected)

`recordings/calibration/` holds **nine** photos (seven 1280x720, two 1920x1080). All
nine were run through `daphnia.vision.grid.detect_grid`, and **all nine were refused**
("the grid is too faint or too fine to measure"), including on every one of five
200-px windows of the eyepiece photo. One 1920x1080 file is a **ruler on a bench**, not
a grid at all.

### What the paper actually is (a profile, not a guess)

A profile down a 600-px *narrow* strip (20 columns only - averaging a wide band smears
a grid rotated by 9°, which is how the first, wrong reading was produced):

| Measured | Value |
| --- | --- |
| spacing between consecutive printed lines | **27 px**, uniform (diffs 24–31 px, none missing) |
| the heaviest lines | exactly **5 cells apart = 135 px** |
| line widths / depths in that strip | 1–7 px wide, 7–16 out of a ~174 background |
| amplitude levels | **two** (one heavy family every fifth cell) in that strip - not three |

So one printed cell is **27 px** at this zoom. Which *printed size* that is (1 mm with
a heavy line every 5 mm, or 2 mm with a heavy line every 1 cm) is a fact about the
sheet in your hand and settles in one glance with a ruler - it only sets `--pitch-um`
(and `SR-OPTICS-030` already carries that field). It also means the sheet is far larger
than the frame: at 27 px/cell a 10x10 cm sheet is thousands of pixels across, which is
exactly what *"the grid overflows the image"* looks like.

### Why this photo is hard, and what still limits it

Sharpness across the paper (mean |high-pass| in 120-px tiles, paper tiles only):

```
            x=180  300  420  540  660  780  900  1020
   y= 60      6.7  5.5  5.2  4.9  4.6  4.2  4.8     -
   y=180      6.1  6.2  6.3  6.5  5.8  4.9  4.4     -
   y=300      6.4  6.2  6.3  6.3  6.2  5.1  4.3     -
   y=420      7.8  6.3  5.6  5.4  5.4  4.9  4.9     -
   y=540        -  7.4  5.6  5.2  5.4  6.0    -     -
   y=660        -    -  8.6  8.1  8.1    -     -     -
```

The **top rows (4.2-5.2)** and the **right column (4.3-4.9)** carry about half the
high-frequency detail of the bottom (8.6, 8.1) - the grid there is defocused and dimmer,
which is visible in the original photo and in `*_points.png` (the points drift off the
lines along the top). The phone is clamped to the eyepiece at a slight angle to the
image plane, so one side of the field is out of focus. **That is what limits the
*magnitude* of the passing model** (§0), not the effect: the model passes its own
sub-pixel check at 0.511 px, but its +9.75 % disagrees with the ~5.7 % measured from the
cell spacing, and the defocused band is where the line positions are least trustworthy.

So the cheapest way to firm the magnitude up is **another photo**: paper flat and filling
the field, both the ring and the grid in focus, even illumination. Re-run the same command
and compare the two profiles.

### The distortion itself, measured without the model

The cell spacing in 24-px bands of the *straightened* frame (a narrow band is a clean
profile once the lattice is aligned - this is the measurement to trust, and it needs no
fit):

```
   across the field                     up and down the field
   x=200  28.00 px/cell                 y= 60  28.00 px/cell
   x=400  26.00                         y=160  27.00
   x=500  26.25                         y=260  26.50
   x=600  26.50                         y=360  26.50
   x=700  26.25                         y=460  26.00
   x=800  27.00                         y=560  26.50
   x=900  27.25
```

**26.0-28.0 px per cell across a 40 mm field: ±3.8 %, and up to ~5.7 % between the centre
and the left/top edge.** At a nominal 27 px/mm that is a **3.8-5.7 % scale error**, i.e.
**76-114 µm on a 2 mm animal** - seven times the paper tolerance (0.5 %) and comparable to
the keypoint model's own error. The effect is real, it is *positive* (edge cells are
*larger*, a pincushion-like term), and it is monotone with radius, all of which the fitted
model agrees with. What the fitted model gets wrong is only the **size**.

### The scale: measurable from this paper, but by a profile, not by discorpy's estimator

| Run | Reported spacing per cell |
| --- | --- |
| **a narrow-strip profile** (20 columns of the equalised frame, run near the discarded 160-column smearing) | **27.0 px, uniform** (diffs 24–31, none missing) - the stable measurement |
| discorpy's estimator, demo default `radius=9` | 19.144 / 18.376 px (old pipeline) and **18.021 / 21.530 px** (after the mask and straightening) - **unstable, and wrong** |
| discorpy's estimator, `radius=11` | 28.193 / 27.905 px (old pipeline) and **18.497 / 25.049 px** (after the mask and straightening) |
| the *grouped lattice* (cross points) | 25.936 / 25.518 px → and with `--dist-px 27` it assembles **24 + 30 lines**, so the grouped lattice agrees with the profile |
| `daphnia.vision.grid.detect_grid` | **refused** - on the raw photo, with the field stop filled, and even with the line weights equalised |

The estimator's answer **moves by ~1.5x on the same photo** as preprocessing and radius
change (16.9–28.2 px across all runs), so it is not the number to gate on. What is
stable is the profile (27.0 px) and the *assembled lattice* (24 + 30 lines at 27 px).
That distinction is the honest form of "the scale is measurable from this paper" -
and it is why `calibration_probe.py` prints the estimator's two radii **and** the
pinned-lattice result side by side instead of one number.

**The shipped detector's failures, diagnosed.** Two different gates refuse these
photos, and neither is the paper's period:

* *prominence* (`MIN_PROMINENCE`): the grid is a pale mint-green print on white, and
  the surrounding field stop is a hard black circle whose edge is broadband energy.
  The gate needs the lattice peak to stand 100x above the noise floor; that surround
  poisons the floor. Filling the surround with the paper's median does **not** rescue
  it, and neither does equalising the line weights - measured on all four variants.
* *refinement range* (`MIN_PERIOD_PX = 6 … MAX_PERIOD_PX = 500`): when the peak search
  lands on a harmonic, `_refine_period` returns `None` and the refusal message is the
  "too faint or too fine" one. On a hard-edged *synthetic* grid the fundamental and the
  second harmonic have **equal** amplitude, and the detector reported **9.45 px** for a
  synthetic grid whose true period is **19 px** — a factor-2 error that a
  real-paper photo with a dominant fundamental hides.

**The distortion fit on the real photo, with the lattice made detectable.** Equalising
the line weights (clip the dark side to one common depth, *after* the background
normalisation) takes the grouped lattice from **8 lines to 12–40 per family** - so the
detection problem is solvable - and `--radius 11` with the demo grouping gives 2798 +
1693 points on **12 + 15 lines**, a centre of distortion at **(666.2, 285.2)**,
coefficients `c0=0.8987, c1=+8.91e-4, c2=-2.02e-6`, and a residual of **8.83 px rms,
worst line 23.15 px** → **not sub-pixel**.

**Why, and it is not a tuning problem.** Thirty combinations were swept over radius,
sensitivity, grouping tolerance, missed-dot allowance and acceptance ratio: the cases
with a *plausible* centre of distortion (inside or near the frame) leave **3.7–16 px**
rms, and the only sub-pixel ones put the centre ~10⁴–10⁸ px outside the frame - fits
that **collapsed** to a nearly flat map, which is trivially "straight". Nothing in that
sweep is a model that may be stored. The reasons are all about *this photo*: the field
stop clips every line into an arc, one line weight in five is 4–7 px wide while the
rest are 1–2 px (their sub-pixel centres are not equally reliable), the print is ~1.5 %
contrast, and the whole field holds only ~26 cells across - little signal for a radial
model that has to be extrapolated to the centre of distortion.

**The oracle says the method is fine.** The same code, same parameters, on
discorpy's own **single-weight** line pattern (19 px pitch) distorted with a *known*
model (`[1, 0, -1e-7]` = 4.8 % of local scale at r = 400 px):

| Run | Centre recovered | Error | Residual | discorpy verdict |
| --- | --- | --- | --- | --- |
| demo params (`radius=9, sensitive=0.1`) | (638.97, 359.52) vs truth (640, 360) | **1.13 px** | 0.264 px rms, worst line 0.771 px | **sub-pixel** |
| `radius=7, sensitive=0.02` | (639.26, 359.48) | **0.90 px** | 0.112 px rms, worst line 0.474 px | **sub-pixel** |
| **dot grid** (27 px pitch, dot-pattern flow) | (627.04, 357.26) | 13.0 px | **0.149 px** rms | **sub-pixel** |

The dot-grid run is a scratch measurement (2026-10-19, not yet a `--oracle` mode in the
probe): `make_dot_pattern(720, 1280, dot_distance=27, dot_size=11)` distorted by the
same known model, then `binarization` → `calc_size_distance` → `select_dots_based_size`
→ `select_dots_based_ratio` → `calc_hor_slope`/`calc_ver_slope` → `group_dots_*` →
`find_cod_coarse` → `calc_coef_backward`. 24 + 44 lines, 1045 + 1045 dots.

So: the tool recovers a known distortion from one image to about a pixel on the centre
and a fifth of a pixel on the residual, for **both** reference patterns. **The photo is
the limit here, not the method** — a dot grid and a single-weight line grid both work.

---

## 4. Using the probe

```bash
python scripts/calibration_probe.py recordings/calibration/<photo>.png            # the verdict
python scripts/calibration_probe.py <photo> --radius 11                           # right for graph paper
python scripts/calibration_probe.py <photo> --dist-px 27 --dot-miss 10 --accepted 0.25   # try harder
python scripts/calibration_probe.py <photo> --coef 3 --field 0.42                 # model shape
```

Four sections: the shipped detector (whole frame + five windows, refusals quoted),
**§1b the lattice as discorpy measures it at two radii** (the scale answer - the two
numbers are 19 px and 28 px on the same photo, and only one of them is the paper),
**§2 the oracle** (a synthetic single-weight grid with a known distortion), **§3 the
model on this photo**. Exit code 0 means a model was fitted **and** passed discorpy's
sub-pixel check; 1 means it did not, and the script says which of the two failed.

Outputs (a JSON with every number plus the settings, an overlay of the detected cross
points, and the unwarped image) go to `recordings/calibration_probe/` unless `--out`
says otherwise.

The *count* of coefficients is a measurement too: on the oracle, `--coef 5` gave **2.006 px**
rms where `--coef 3` gave **0.264 px** — so the residual decides the model order, and a
convention (three coefficients, five coefficients) is never the reason to trust one.

### 4.1 The reference flow, *unmodified* — `scripts/discorpy_probe.py`

```bash
python scripts/discorpy_probe.py recordings/calibration/<photo>.png   # the library alone
python scripts/discorpy_probe.py <photo> --radius 11 --coef 3        # one flag at a time
```

The probe above prepares the frame - field stop, line weights, tilt - before discorpy
sees it, so it cannot say what the *library* does by itself. `scripts/discorpy_probe.py`
is that answer: discorpy's own line-pattern example, in its order and with its arguments,
on the frame exactly as it was loaded, plus a CLI, one output folder per photo and
discorpy's own verdict. The two scripts differ in one axis - the prepared frame - and
every number either prints comes from the same library calls.

Measured 2026-10-19 (discorpy 1.7.0, scipy 1.18.1):

| Run | What came out |
| --- | --- |
| this rig's photo, as it is | **no model**. The slope/distance estimator is thrown to its **+30 deg search limit** (the ocular ring), and the cross-point call then raises discorpy's own "Row index is out of range": the reference flow does not get past step 3 |
| the same photo, `--radius 11` | the same refusal - the search radius is not what stops it |
| a synthetic single-weight line grid with 1.6 % barrel | **passes**: centre (641.97, 354.23) against a true (640, 360), i.e. 6.1 px off, and a residual after unwarping of **0.204 px** worst (9.010 px before) |
| the probe's own corrected output | a **degenerate** fit: centre (192.6, -101.3) *outside* the frame, `c0` = 1.884, residual 116.6 px - a grid with no distortion left to fit |

Two things the flow does not do as written, both named by the script rather than
silently absorbed:

* **`post.unwarp_line_backward` is a silent no-op on the float32 arrays that
  `proc.regenerate_grid_points_parabola` returns** - the numerical gradient is non-finite
  at float32, so the optimiser stops at its starting point (a point 400 px from the centre
  moves **0.00 px**, against **6.25 px** in float64). The script casts to `float64` for
  that one call: without it the "residual after unwarping" reads 9.01 px on a target the
  library takes to 0.20 px, so the verdict would be wrong on *every* input;
* **`post.check_distortion` cannot tell a degenerate fit from a bad one**, so the script
  checks that the centre of distortion is inside the frame and names the case when it is
  not.

Exit code 0 means a model was fitted, its centre is inside the frame **and** it passed
discorpy's sub-pixel check; 1 otherwise.

### 4.2 The documented §4.1 flow — `scripts/discorpy_doc_probe.py`

The third probe runs what the **documentation** prescribes for strong distortion plus
perspective (docs §4.1, the 1.7 / [C2] pipeline): normalise up front
(`prep.normalization_fft(img0, 10)`), the docs' own unwanted-point mask
(`prep.remove_points_using_parabola_mask`), the docs' grouping
(`prep.group_dots_*_lines_based_polyfit`, ratio 0.1 / 3 dots missed / order 2), the 1.7
centre of distortion (`proc.find_center_based_vanishing_points_iteration`), a residual
pass at 3.0 px, `util.unwarp_color_image_backward(..., pad=...)` and
`losa.save_metadata_json`.

```bash
python scripts/discorpy_doc_probe.py <photo> --coef 3              # the docs' own arguments
python scripts/discorpy_doc_probe.py <photo> --radius 11 --no-mask
python scripts/discorpy_doc_probe.py <prepared frame> --grouping neighbour --ratio 0.5
```

**Answer: no — it does not get close on this photo.** Same photo, same session
(2026-10-19); every residual is rms, so it is comparable with §4.1's numbers:

| Run | lines | centre of distortion | residual after unwarping | pass? |
| --- | --- | --- | --- | --- |
| **doc flow on discorpy's synthetic single-weight grid** | 24 + 41 | **(639.16, 359.58)** vs a true (640, 360) | **0.018 px rms**, worst 0.146 px, 0 % over 1 px | **✓** |
| `calibration_probe.py --coef 3` | 16 + 21 | (621.09, 313.28) | **0.679 px rms**, worst 1.283 px | ✗ (marginal) |
| doc flow, docs' arguments, raw frame | 1 + 5 | — | refused: too few lines | ✗ |
| doc flow, `--radius 11`, raw frame | 1 + 4 | — | refused | ✗ |
| doc flow, docs' mask, loose grouping | 7 + 10 | (623.65, **-2139.68**) | 2.2 px, c0 = 1.81 | ✗ degenerate |
| doc flow on the app's prepared frame | 4 + 17 | (681, 291) | 10.9 px, c0 = 1.16 | ✗ |
| … plus equalised, spacing pinned at 27 | 7 + 15 | (662, 275) | 17.4 px, c0 = 1.56 | ✗ |
| … `--grouping neighbour --ratio 0.5` | **16 + 23** | (707.54, 305.00) | **2.631 px rms**, worst 16.13 px | ✗ |

What the documented flow **did** give, all measured:

* **`prep.normalization_fft(img0, 10)` up front fixes the slope estimate.** On the raw
  frame the example flow reads **+30.20 deg** (its own search limit) where this one reads
  **-8.95 deg**, against a true tilt of **-9.00 deg**: part of what
  `scripts/calibration_probe.py` calls its own preprocessing is a *documented library
  step*. With `--radius 11` the same call gives **28.252 / 27.977 px** per cell - the
  paper's own 27.0 px profile - where the default radius 9 answers 19.168 px (**-32 %**).
* **the docs' perspective diagnostic works**: the peak-to-peak spread of the line slopes
  falls from 0.060 to 0.0077 rad across `correct_perspective_effect` on a prepared frame.
* **the library's own artifact and record**: `util.unwarp_color_image_backward(pad=True)`
  produced a **colour, view-preserving** corrected frame (2071x1222 / 2610x1477) where the
  sibling writes a greyscale, unpadded one; `losa.save_metadata_json` writes a file
  `load_metadata_json` reads back.

What it **failed** at, each named by the probe rather than hidden:

* **the parabola mask cannot isolate a circular field stop.** It is a *border* trim: with
  the docs' margins as fractions the ocular ring's arcs sit *inside* it, and a rectangle
  that does cut them removes paper instead (3 + 5 lines against 4 + 17).
* **the docs' polyfit grouping collapses on this grid**: 1 + 5 lines with the docs' own
  parameters; 7 + 15 at the loosest settings, where the centre of distortion leaves the
  frame - a degenerate fit that `post.check_distortion` alone would not have caught.
* **so the coefficients do not describe the grid**: c0 wandered from 0.96 to 1.91 and the
  residual *rose* after unwarping (1.708 -> 2.741 px rms). The same points, grouped by the
  library's *other* grouping (`--grouping neighbour`), give 16 + 23 lines and the best
  result in the table.

**Conclusion.** The control run makes it unambiguous. On discorpy's own synthetic
single-weight grid the same flow, with the documentation's own arguments (`--coef 5`,
radius 9), groups **24 + 41 lines**, finds the centre of distortion within **~1 px** of the
truth, and unwraps **0.973 -> 0.018 px rms** (0 % of points over 1 px, exit 0) - the best
any of the three probes has done on that target. **So the documented flow is not the
problem; our target and frame are**, exactly as §4.1.2 says. On this photo cross-point
extraction gives 687 + 461 points on the raw frame against 3240 + 3167 on a prepared one,
and only the prepared frame lets any grouping assemble enough lines: the flow is not a
shortcut past the sibling's preprocessing - but its normalisation step, its colour+padded
unwarp and its metadata format are three things worth borrowing, and a flat, parallel,
uniform **single-weight** target is still the real fix.

Two knobs exist *because* of what was measured here, not for convenience:

* **`--coef` decides whether the model passes at all**, and it is a measurement, not a
  convention: on this photo **3 coefficients pass** (0.511 px rms), **4 give 2.22 px** and
  **5 give 1.08 px** (both fail), while on the *synthetic* target 5 was fine. Fewer
  parameters fit the optics instead of the residual scatter.
* **`--dist-px` must NOT be used with the documented flow.** The documented grouping
  expects the *sampled* spacing the estimator returns (16.9/21.0 px here), not the paper's
  line pitch: pinning it to 27 px moved the centre of distortion by 78 px and the residual
  from 0.51 to 2.57 px. (It was the right knob for the *earlier*, fisheye-style grouping -
  which is why the measurement is written here rather than the anecdote.)
* `--regenerate/--perspective/--cross-ratio/--residual-px` are the documented flow's own
  steps (docs 3.3.4): regenerate the lattice from the parabola fits (the step that took
  the residual from 3.18 to 0.51 px), correct the target's tilt from the parabola
  coefficients (measured here: changes the curvature ~2x, so it must be *decided*, not
  assumed), sample the cross points every `0.5 x` the line distance, and drop points more
  than 2 px from their parabola.
* `--radius` still matters for the *spacing* estimate on this paper (9 answers 19.1 px,
  11 answers 28.2 px on the older pipeline) - but it no longer decides whether the model
  fits.
* everything else (`--roi/--sensitive/--select-peaks/--field`) is discorpy's own.
  `--select-peaks` (Gaussian fitting) takes the **synthetic oracle** from 0.264 px to
  **0.056 px** and on this photo fits *nothing* - the print is too noisy for a Gaussian.

The probe prepares every frame the same way, and every step is a measured requirement of
*this* rig rather than hygiene:

1. **the field stop is masked as a fitted circle** (Otsu → largest region → circle fit that
   ignores the frame border → eroded margin): without it the stop's ring is read as one
   enormous line (the grouping kept 1 line instead of 34), and a mask smaller than the true
   circle fills over real paper - the first version was 60 px small on the radius and that
   is exactly the "fails at the top and right edges" it produced;
2. **every cross point outside that circle is dropped** (the ring's arc was hundreds of
   points on top of ~2 200 real ones);
3. **the line weights are equalised** (clip the dark side to one common depth *after*
   background normalisation) - the grouped lattice goes from 8 lines to 24-40 per family;
4. **the lattice is straightened** to 0.00 deg, so discorpy's tilted-profile sampling is
   not truncated (~114 px along one side at a 9 deg tilt had no points at all).

Field coverage is reported (55.2 % of the frame kept on the eyepiece photo) and so is the
fitted circle (centre (600.7, 265.6), radius 448.7 px) - the second number is what makes
the first checkable.

---

## 5. What to do about it (the decision, with its cost)

Three separate jobs, and only the middle one needs a purchase:

1. **Get the scale back on this rig - do this first, it is two preprocessing steps and
   one parameter.** The prepared frame (fill the field stop, equalise the line weights)
   plus `radius=11` measures **28.19 / 27.91 px** per cell where the shipped detector
   refuses and `radius=9` lies by 1.5x. That belongs in `vision/grid.py` (or a
   `vision/lattice.py` beside it) with a test that pins the 19-vs-28 distinction.
2. **Get a distortion model - print one target.** A **single-weight line grid** or a
   **dot grid** at the specimen plane; both were measured to reach sub-pixel (0.240 px
   and 0.149 px rms) on a known distortion.
3. **Then decide what to do with the model**, in increasing cost:

| Option | What it buys | What it costs |
| --- | --- | --- |
| **Report, do not correct** | the app says *"distortion unmeasured"* instead of implying 0 %, and the "keep specimens central" advice stays grounded | nothing; the scale stays position-dependent |
| **A local scale** (measure the period where the specimen is, store it per measurement) | every number is right where it was taken | a second measurement per frame, and a scale *function* in the record instead of one value — `measurements.px_per_um` is one value today |
| **Unwarp the frames** (a discorpy model stored with the preset, applied before measuring) | one scale is true everywhere, and a corrected frame looks like the reference books | the model must be stored per optical chain, every frame must be resampled (cost + resampling blur), and the *corrected* scale must be re-measured and recorded |

**What must hold in any of the three** — the rules the probe already follows, and the
ones a future `daphnia/vision/distortion.py` should too:

1. **A drift that was not measured is reported as unmeasured, never as 0 %.** The
   shipped `GridEstimate.distortion_pct = 0.0` default means "not computed"; a surface
   must not read it as "none".
2. **A model is only usable if it passes the sub-pixel check.** A residual over 1 px
   on more than 15 % of the points (discorpy's own rule) means the coefficients are not
   a distortion measurement and are not stored. The oracle and the photo result above
   are exactly that distinction in one script.
3. **A model is hardware**, like the calibration record itself: dated, immutable,
   grouped by `setup_key`, never edited.
4. **The scale of a corrected frame is re-measured from the corrected grid.** `c0` is
   a centre magnification, not a scale — see §2.

## 6. The tools around this job

| Tool | Type | Best for | How it addresses the problem |
| --- | --- | --- | --- |
| **OpenCV / OpenCV.js** | library (C++, Python, WebAssembly) | direct integration, including a web app | `calibrateCamera`, `findCirclesGrid` and `undistort`: **several views** of a known target give the full intrinsics plus the radial terms (`k1, k2`), and the same maths runs in JavaScript, so a live feed can be rectified in the browser. Multiple views and a whole-target geometry are the price, which is why it is the wrong tool for "one photo of graph paper through an eyepiece" |
| **Discorpy** | Python library | high-precision lens distortion, microscopy and detector systems | corrects radial and perspective distortion **independently**, from a **single** image, by direct computation rather than iterative optimisation; it locates the centre of distortion and reports its own sub-pixel residual. Used here to validate the calibration maths **offline** (§3) |
| **ImageJ / FIJI** | desktop application (Java) | the gold-standard bio-metrology baseline | *Lens Distortion Correction* and *Set Scale* let a lab check the app's pixel-to-micron readings against proven desktop software — an independent referee, not a component |
| **Micro-Manager** | open-source microscopy UI | automated acquisition (stages, cameras, pixel calibrations) | the standard framework for driving hardware and holding spatial calibrations; built for a desktop microscope rather than a phone clamped to an eyepiece, so it is a *reference for what a calibration record should contain*, not a dependency |
