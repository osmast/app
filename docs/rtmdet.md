# RTMDet — training, evaluating and running the detector

The guide for the model side of DaphniaScope: how to train a detector on an exported
dataset, how to read the number it reports, and how to run it live in
`python run.py predict`. The code lives in `daphnia/vision/rtmdet/`.

> **Before changing anything here, read `memory-bank/mlContext.md`** — the expert
> stance, the ML laws (one frame of reference, oracle metrics, inference at the trained
> size, split by group), the code shape, the reference implementations to consult and
> the catalogue of traps already paid for. This document is the *task* guide; that one
> is the *craft*. Invocation rule: `.clinerules/ml-engineering.md`.

---

## 1. The loop

```
Dataset surface            scripts/train_rtmdet.py         Predict surface
(label boxes)       ->     (train + evaluate)        ->    (run live / on video)
   |                              |                              |
datasets/<name>/              models/rtmdet_daphnia.pt      python run.py predict
  images/  labels.json  boxes.csv  splits/  manifest.json
```

Nothing else is needed: the trainer reads **only the export folder**, so a model can
be trained on a machine that has never opened the application.

---

## 2. Install

PyTorch belongs in the training environment, and `environment.yml` now carries it:

```bash
conda env update -f environment.yml      # adds pytorch to z-app
conda activate z-app
python scripts/check_torch.py            # proves MPS, OpenCV and a forward pass
```

`make train-check` does the same thing. It prints the torch build, whether **MPS**
(the Apple-silicon GPU) is available, and then runs a real forward *and backward*
pass on the chosen device — because a Metal build can report availability and still
fail on an operation, and that is much better learned now than ninety minutes into a
long run.

### macOS / M4 notes

* **MPS is used automatically** (`--device auto`). `--device cpu` is the escape
  hatch; `--device mps` on a machine without Metal is an **error**, never a silent
  fall back to a 20× slower run.
* `conda-forge`'s `pytorch` and Anaconda's `gpu_mps` build both work; check with
  `python scripts/check_torch.py` rather than assuming.
* **Never `pip install` torch into `z-app`.** conda-forge's `py-opencv` ships no
  `RECORD`, so pip cannot uninstall or replace OpenCV, and a pip transaction that
  dies half-way leaves a mixed-channel environment — which has already happened once
  (see `memory-bank/techContext.md`, constraint 7).
* The data loader defaults to `--workers 2`. With a handful of frames `--workers 0`
  is often faster (no process start-up); with hundreds, workers help.
* Unified memory is the limit, not the GPU: `--batch-size 4 --input-size 640` with
  `rtmdet-tiny` is the comfortable default on 16 GB. Reduce `--batch-size` first.

---

## 3. Train

```bash
python scripts/train_rtmdet.py --inspect      # what is in the corpus, and what is wrong with it
make train                                    # the default run on datasets/bb
python scripts/train_rtmdet.py --epochs 40 --max-steps 20 --workers 0   # a smoke run
```

`--inspect` prints the corpus's own report **before** anything is trained, including
the two things that decide whether a score is worth reading:

```
bb: 526 box(es) on 223 image(s) from 14 capture(s)
  train  160 image(s)   384 box(es)  14 capture(s)
  ...
read this before trusting a number:
  - 11 of 14 capture(s) are split across train/val/test ...
  - 11 box(es) leave their image or are smaller than 3 px and are dropped ...
```

### Defaults

| Setting | Default | Why |
| --- | --- | --- |
| `--preset` | `tiny` (316 k parameters) | fits a laptop; `small` (≈1.2 M) when the corpus justifies it |
| `--input-size` | `640` (longest side, padded to a multiple of 32) | RTMDet's own training size |
| `--epochs` / `--batch-size` | `100` / `4` | |
| `--lr` → final | `0.004` → `0.0002`, AdamW, `--weight-decay 0.05` | RTMDet's schedule, with a linear warm-up |
| `--split-by` | `group` | held-out **captures** (see §4) |
| `--mosaic` | `0.5` | four frames in one: the augmentation a small corpus needs |
| `--grad-clip` | `35` | a stability guard for MPS |

### Can I stop and resume?

Not yet. A run trains from scratch; the checkpoint it writes is the **best**
validation epoch plus the final state. `--epochs` is the only "how long" knob, and a
resumed run is a documented next step rather than a half-working feature.

---

## 4. The split — and why the default is not the export's

`splits/*.txt` are built by hashing the **image path**, so two frames of one capture
(the same plate, seconds apart) can land in train *and* validation. The validation
score then measures memorisation. On the corpus measured here, **11 of 14 captures**
were split across the three sets.

`--split-by group` (the default) re-partitions by **capture**, using the observation
directory recorded in each specimen's original file path. It is deterministic — a
SHA-1 of the group key, exactly as the exporter does for one image — so a score can
be reproduced. With only 14 captures the val split is small (for this corpus: 1
capture, 15 frames), and the run prints that.

`--split-by export` uses the export's own split. It exists because that split is what
the *deliverable* defines, and being able to reproduce it is honest — but the report
says which one was used, and a score from `export` must never be compared with one
from `group`.

**The durable fix is upstream:** annotate more distinct captures (each session is one
group), and make the export split by specimen/study (`SR-DATASET-010`).

---

## 5. What the number means

Measured on `datasets/bb` (158 training frames / 332 boxes from 10 captures, 30
epochs of `rtmdet-tiny` on an M4, ~4 min), validated on the **held-out capture**
(15 frames, 54 boxes):

```
epoch 28/30  P 0.012  R 0.963  F1 0.023  AP50 0.813
```

Those precision/F1 numbers are at the **evaluation threshold of 0.05**, which is
deliberately low so AP can see the marginal detections — and it is why they look
terrible. What an operator actually gets depends on the threshold, and that is the
number to read:

| score ≥ | precision | recall | F1 | detections/frame | animals/frame |
| --- | --- | --- | --- | --- | --- |
| 0.05 (AP's threshold) | 0.011 | 0.926 | 0.022 | 300 (capped) | 3.6 |
| 0.20 | 0.151 | 0.870 | 0.257 | 20.8 | 3.6 |
| **0.35 (the surface's default)** | **0.608** | **0.833** | **0.703** | **4.9** | 3.6 |
| 0.50 | 0.727 | 0.148 | 0.246 | 0.7 | 3.6 |
| 0.70 | 0.000 | 0.000 | 0.000 | 0.0 | 3.6 |

So the detector **finds the animals** (recall 0.83, AP50 0.73) and the default
threshold of 0.35 is the operating point — P 0.61, R 0.83, F1 0.70, about 5 boxes
for ~4 animals. Two honest caveats:

* **The score is not well calibrated, and the usable range is narrow.** Between 0.35
  and 0.50 recall collapses from 0.83 to 0.15: most animals score just above 0.35 and
  almost nothing scores above 0.5. Expect to re-tune the threshold for a different
  illumination, and read a 0.4 as "probably", not "no".
* **The validation set cannot resolve the differences.** It is **one capture, 15
  frames** — epoch-to-epoch AP50 ranged from 0.16 to 0.81 with no trend, so "best
  AP50 0.813" is one lucky draw from a noisy series, not evidence that epoch 28 is
  the best model. More captures are what turns this into a measurement
  (`SR-DATASET-010`).

Also note the two splits are **not comparable**: the export's own val split contains
*different frames* (33 frames of 9 captures), so its lower numbers (P 0.41, F1 0.55,
AP50 0.63 at 0.35) do not show the leak inflating anything on this corpus. The leak
is a structural argument — 11 of 14 captures are spread over the splits — not a
demonstrated inflation here.

Latency, measured: **15 ms/frame** for inference at 640 (≈68 fps) and **80 ms per
training step** (batch 4 at 640) on MPS, after an ~8 s first-step warm-up.

* **Precision** — of the boxes it drew, how many were a real animal.
* **Recall** — of the animals that were there, how many it found.
* **AP50** — the all-point interpolated area under the precision/recall curve, at a
  **greedy IoU ≥ 0.5** match (`--match-iou`).

**This is not COCO mAP.** There is no pycocotools here, no 101-point interpolation and
no per-class average, so an AP50 from this script is comparable with *another run of
this script on the same corpus* — and with nothing else. The report prints that
warning itself, so a number cannot travel out of context by accident.

Matching happens in the **original image pixels**, with **both** the predictions and
the labels mapped back through the letterbox first. Getting half of that wrong (the
prediction mapped, the label left in the padded frame) gives an IoU of 0 for every
pair — a metric that reported AP50 0.000 for thirty epochs of a model that was in
fact learning. `tests/test_rtmdet.py::EvaluationTests` now pins it with an "oracle"
detector: a model that returns the labels must score exactly 1.0.

---

## 6. Run it

```bash
python run.py predict
```

Then: **Detector (RTMDet) ▸ Load the active detector** (or a checkpoint from the list under
*checkpoints in models/* - loading one registers and activates it), and
**Pose (RTMPose) ▸ Load the active pose model** for the keypoints. Then a source from the
**Source** menu: **Use the live feed**, **Open a video file…** or **Open a single image…**.
Loading a model switches the page's **Model** tool on (it is off by default), so the answer
appears on the frame as soon as the model is in force. The score and NMS thresholds are live —
drag them while watching a frame and the next frame obeys. The panel reports animals found, the
frame size and the measured **ms/frame**.

**Loading a model applies it at once** (`SR-PREDICT-017`): a pass that is already running
is restarted with the checkpoint just loaded (the same source is kept, so the identities
it has averaged survive), a still frame on screen is detected again, and the window says
so — the header names the model that answered and the readout reports what *that* model
found, `0 found  NONE ABOVE THE THRESHOLD` included. Which checkpoint is in force is the
**registry's** decision (`daphnia/models` → the `models` table, one active row per kind),
which is why a source chosen before any model is loaded loads *the active one* rather
than a default file.

The frame is letterboxed **to the size the checkpoint was trained at** (640 here),
not to its native resolution: feeding the network a 1280-pixel frame when it learned
on 640 changes the scale statistics *and* costs 2.4× the compute. The checkpoint
records its input size so the surface cannot get this wrong.

Predictions are **never stored** (`SR-PREDICT-N01`): a prediction is not ground truth.
Offering them to the Dataset surface as pre-labels (`source = 'model'`,
`SR-DATASET-007`) is the surface's next step.

From a script or a notebook:

```python
from daphnia.vision.rtmdet.detect import load_detector
from daphnia.vision.images import load_image

detector = load_detector("models/rtmdet_daphnia.pt")     # MPS if available
for detection in detector.detect(load_image("frame.png")):
    print(detection.describe())        # daphnia 0.87  92 x 75 px  @ 519,562
```

---

## 7. What is, and is not, RTMDet here

**Faithful to RTMDet:** the CSPNeXt backbone family with a deep stem and
depthwise-separable CSP stages; the CSPNeXtPAFPN neck with 5×5 depthwise convolutions;
a head whose convolutions are **shared across levels with a separate BatchNorm per
level**; **one anchor per location**, predicting the four distances to the box sides;
**SimOTA** dynamic label assignment with a centre prior and
`k = max(1, ⌊Σ top-10 IoU⌋)`; **QualityFocalLoss** (soft, IoU-valued classification
targets) and **GIoU**; the 0.01 class prior; mosaic, random flip and colour jitter.

**Deliberate differences** — recorded here so nobody compares this with the reference
implementation and wonders:

1. **Not an `mmdet` port.** Block internals, exact channel multipliers and the depth
   of the stem are documented choices (`rtmdet_tiny`: stem 16, stages
   `(32, 64, 128)` at strides 8/16/32, neck 64), not copied numbers.
2. **Three stages, not four or five.** The head consumes P3/P4/P5; there is no P6 lift.
3. **The input is letterboxed to a multiple of 32, not padded to a square.** Cheaper,
   and the pad is *recorded*, so boxes map back exactly.
4. **NMS is NumPy**, not `torchvision.ops.nms` — sixteen lines instead of a dependency.
5. **No EMA, no mixed precision, no `train1x` schedule.** The default run is short
   enough to finish on a laptop in one sitting.
6. **Metrics are not COCO.** See §5.

Anything not listed as a difference is meant to behave like RTMDet. If you find a
behaviour that is neither, it is a bug, not a choice.

