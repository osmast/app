# RTMPose — the top-down keypoint stage (head → tail)

The guide for the **pose** half of the pipeline: how a detector box becomes two
keypoints, how those keypoints become a body length, how to train the model, and
how to read the number it reports. The code lives in `daphnia/vision/rtmpose/`.

> **Before changing anything here, read `memory-bank/mlContext.md`** — the expert
> stance, the ML laws, the code shape, the reference implementations and the trap
> catalogue. Invocation rule: `.clinerules/ml-engineering.md`. This document is the
> *task* guide; that one is the *craft*.

---

## 1. The pipeline (top-down = detect, then pose inside the box)

```
frame ──RTMDet──▶ boxes (models/rtmdet_daphnia.pt, already trained)
                     │  for each box:
                     ▼
          square window around the box  (rtmpose.data.pose_input_for_box)
                     ▼
          resize to 192, edge-pad        (rtmpose.data, owns the invert)
                     ▼
          RTMPose (CSPNeXt + SimCC head) ──▶ head, tail in *input* pixels
                     ▼
          invert → source pixels → Keypoints(eye_apex, spine_base)
                     ▼
          pipeline/derive.py → length_µm (Optics) → persist
```

This is the arrangement `mlContext.md` §5 names as mmpose's **top-down two-stage**
(RTMPose, arXiv:2303.07399; SimCC, arXiv:2107.03332): our `localize_well` (RTMDet)
is the detector half, and this package is the pose half. We re-implement from the
papers in plain PyTorch: **no mmpose dependency**, and no AGPL code.

---

## 2. What the corpus actually contains (measured 2026-10-17)

Read from `datasets/kp/labels.json`:

| Fact | Value | Consequence |
| --- | --- | --- |
| Schema declared | `pose_full_boxes` — **4** keypoints + box | but only 2 were placed |
| Specimens | 526 boxes on 223 images | the detector corpus, intact |
| Specimens **with keypoints** | **384** usable, all `{head, tail}` (1 degenerate dropped) | this is the pose corpus |
| With sides (`sideL`/`sideR`) | **0** | no width branch: the model is **2-point** |
| Box only (no keypoints) | 141 | excluded from pose training |
| Degenerate row | **1** with `head == tail` | dropped by the reader |
| Keypoints outside their box | **7** | covered by the crop margin, else clamped |
| Captures in the folder | 10 (`b01..b10`) | but only **4** (`b01..b04`) carry keypoints |
| Keypointed frames | 155 of 223 | the detector sees more frames than the pose model |

**The 2-point decision.** The endpoint (`projectbrief.md`) is *eye apex → spine
base*, which is exactly `head → tail`. Sides are for a width nobody measures yet,
and there is **no ground truth** for them, so the pose model predicts **two**
keypoints. A four-point model is another `labels.json` and another checkpoint, not
another code path — `rtmpose.data` reads whatever names the corpus names.

**The group decision (law 4).** The Dataset export flattens every frame into
`images/` and hashes the *exported copy* path for its splits, so neither the parent
directory nor `splits/*.txt` is leak-safe: frames of one capture land on both sides.
`pose_corpus.py` recovers the group from the **frame stem** (`b01_f00_35755477` →
`b01`), and the trainer regroups by it, so validation is animals the model has not
seen. With 10 captures the val split is **small** — stated in the report, never
hidden.

---

## 3. The metric, stated before the model (law 1, §9.2)

Frame of reference: **original source-image pixels**, via the one `PoseInput`
object that owns the mapping and its `invert` (the same discipline as
`rtmdet.data.Letterbox` and `vision.crop.Crop`).

| Metric | Definition | Reported with |
| --- | --- | --- |
| **per-keypoint error** | Euclidean px, median and p90 | corpus, split, epoch, device |
| **PCK@k** | fraction of keypoints within `k` px | `k` in **px and µm** |
| **length error** | \|predicted − true\| of the head→tail distance, in **µm** | via `traits.length` |
| **OKS** | COCO keypoint overlap, sigmas stated | corpus + sigmas |

It is **not** COCO mAP, and an OKS computed with our own sigmas is comparable with
*another run of this script on this corpus* and nothing else — the report says so
itself. `pose_metrics` is a pure function, so the **oracle** (ground truth in →
error 0, PCK 1.0) and the **blind stub** (→ 0.0) are unit tests, not hopes.

---

## 4. Train

```bash
python scripts/train_rtmpose.py --inspect      # the corpus, and its warnings
make train-pose                                # the default run on datasets/kp
python scripts/train_rtmpose.py --epochs 40 --max-steps 20 --workers 0   # smoke
python scripts/train_rtmpose.py --evaluate     # score the checkpoint
```

Default `TrainConfig`: `rtmpose-tiny`, input **192**, `simcc_split_ratio` **2.0**
(mmpose's default), split by **group**, MPS when available. Weights land in
`models/rtmpose_daphnia.pt` (best) and `…_last.pt` (final epoch) — two files, each
meaning what it says (law 9).

---

## 5. What is, and is not, RTMPose here

**Faithful to RTMPose / SimCC:** a CSPNeXt backbone (shared with `rtmdet.model`);
a SimCC head that predicts the x and y coordinates as two 1-D distributions
(Gaussian soft targets, `KLDivLoss`), decoded by `argmax`, at
`simcc_split_ratio` × the input resolution; horizontal flip and colour jitter
augmentation; MPS/CPU on the machine that takes the pictures.

**Deliberate differences** — recorded so no number is compared by mistake:

1. **Not an `mmpose` port.** The head collapses each axis with mean pooling and
   classifies the bins with a linear layer; the exact RTMPose head width/stack is a
   documented choice here, not a copied constant.
2. **Two keypoints, not 17.** One organism, `head`/`tail`; no left/right, so a
   horizontal flip is anatomically safe.
3. **A square box window, not the box itself.** The window is `max(w,h) × 1.5`
   centred on the box (margin 0.25), which keeps a landmark on the box edge inside
   the crop and makes the mapping invertible in closed form.
4. **Edge replication, not black padding**, for a window that reaches past the frame
   — a black band is an artificial edge the backbone would learn as a feature.
5. **No soft-argmax refinement.** Decoding is `argmax` of the trained target — what
   the loss is trained against — and soft-argmax is noted as the next lever.
6. **Metrics are ours** (§3), not COCO keypoint AP.

Anything not listed is meant to behave like the reference. If it does not, it is a
bug, not a choice.

---

## 6. See it on the Predict surface

`python run.py predict` is the whole cascade, visible - and it is the fastest way to
tell "the annotations are wrong" from "the model is wrong", which is the reason the
surface exists at all.

1. **Detector (RTMDet) ▸ Load the active detector** loads *both* models in practice:
   `models/rtmdet_daphnia.pt` and, when it exists, `models/rtmpose_daphnia.pt` (overridable
   with `DAPHNIA_RTMDET` / `DAPHNIA_RTMPOSE`); **Pose (RTMPose) ▸ Load the active pose model**
   loads the pose half on its own. Each menu also lists **every checkpoint in the models
   folder** (loading one registers and activates it), and loading a model switches the page's
   ***Model*** tool on. The panel's second line names the pose model **and the metrics that
   checkpoint itself earned** - not a claim about pose models in general.
2. **Live feed**, **Open video file…**, **Open a single image…** - each detected
   animal gets its box, a dot on `head` and on `tail`, and the head-to-tail line, in
   the animal's own colour.
3. **Keypoints**, the toggle in the *Frames* menu of the window header, turns the pose
   pass off and on. It costs real time, so watching the detector *alone* has to stay
   possible; the tool appears with the pose model and is absent without one.
4. The list leads each animal with its measured length in **pixels**, and the
   on-image readout adds the median length. Both say *unscaled*: this surface holds
   no optical profile, and a micrometre nobody calibrated would be exactly the
   unscaled number ``projectbrief.md`` forbids.

`docs/screenshot-predict-pose.png` and `docs/screenshot-predict-length.png` are that view
on a real frame from `datasets/kp` - the first with the keypoints, the second with the
**length written above each animal**. Nothing is stored by it - not the box, not the
keypoint (`SR-PREDICT-N01`).

## 7. The scale, and why the number above the animal can be trusted (or not)

The surface converts pixels to micrometres with the calibration the **rest of the
application** is using (`repository.active_profile`), and it says so on every length:

* the **Scale** group names the profile, its `px/mm`, its `µm/px` and the
  calibration's **own** uncertainty, and flags a **WEAK** (`uncertainty_pct` over
  `CALIBRATION_WEAK_UNCERTAINTY_PCT`) or **STALE** (older than
  `CALIBRATION_MAX_AGE_HOURS`) one - a bumped adapter invalidates the scale without
  changing a recorded value, so age is a signal, not a formality;
* the combo lists every calibration in the database, so switching between two optics
  is a click - **without re-running the models**, because the scale converts a pixel
  length that is already measured;
* with no calibration at all the number is in **pixels** and says *unscaled*, and the
  manual entry says it is **not a verified calibration**. There is no path that shows
  a micrometre without a scale.

**What the spread quoted in the readout is *not*:** it is the calibration's own
`± %`, and nothing else. The model's error is a separate, larger number
(`--evaluate`), and the label says so rather than letting one small percentage stand
in for the measurement's uncertainty. Combining the two into a single figure - with
repeats and the reference's own uncertainty (`docs/rtmpose.md` §3, §8) - is the next
step, and it belongs in `derive.py`, not in a status label.

### Before you trust a number in the lab

Two measured facts about what you see (2026-10-17, `models/rtmpose_daphnia.pt`, MPS):

* **Inference is deterministic.** Six identical passes on one frame each gave
  **5360.0 µm** (spread **0.0 µm**). So the wobble you see between *frames* is the
  animal, the focus and the pose - not the code, and not a thin-air excuse to hide
  behind.
* On the corpus' own box for one specimen the model measured **5360 µm against the
  expert's 5362 µm**. That frame is in **train**, so it is evidence the plumbing is
  exact - **not** evidence of accuracy. The honest accuracy figure is the held-out one
  in §3: **median 227 µm, p90 2.8 mm**.
* The on-screen label belongs to a **detected animal**, not to a corpus row. Comparing
  it with a stored `length_um` requires matching the boxes; the two are different
  animals far more often than they look.

1. **Detector (RTMDet) ▸ Load the active detector** - the panel must show *both* lines:
   `Detector: rtmdet-tiny …` **and** `Pose: rtmpose-tiny …`. If the pose line is missing, the
   keypoints will not appear, and it says why.
2. **Pick the checkpoint you mean.** The default is the *best* epoch
   (`models/rtmpose_daphnia.pt`), marked with a **check mark** in the menu; the list under
   *checkpoints in models/* offers every other file in the folder in one click (loading one
   registers and activates it), and *Open a pose model checkpoint…* takes any file anywhere.
3. **Scale** - verify (or re-verify) the calibration in Optics, come back, press
   *Re-read calibrations*. A **STALE** or **WEAK** flag means the µm figure is not to be
   trusted: fix the calibration, or use *Manual um/px* knowing it is unverified. The
   live `zoo.db` is currently **stale** (29 h against a 12 h limit).
4. **Sight the scale on something known** before reading animals: with this profile
   74.84 µm/px means a 5.4 mm adult is ~72 px long.
5. **Changing the optics changes two things.** The µm/px follows whatever calibration
   you apply (correct by construction), but the **detector** then sees box sizes it did
   not train on - the corpus is 1280×720 at ~75 µm/px and the detector letterboxes to
   640. Watch whether the boxes stay sane before believing the keypoints. The pose crop
   is **box-normalised**, so it tolerates a magnification change far better than the
   detector does.

**A caveat about what you see.** A freshly-trained model's keypoints are only as good
as the report says: the first real run (250 epochs planned, val = **one held-out
capture**, 101 samples) reached a median keypoint error of ~5 px but a **p90 of ~54 px
and PCK@5px ~0.49** - good on most animals, badly wrong on a tail of them. Train,
read `--evaluate`, and only then trust the line you see. With one validation capture,
the number cannot resolve small differences (``mlContext.md`` §6, trap 8).
