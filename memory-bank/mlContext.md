# ML Context — how machine learning is done in this project

> **Read this in full before writing, changing, measuring or *reporting* anything that
> involves a model.** It is the stance, the invariants and the references that make ML
> work here reproducible instead of merely plausible.
>
> `docs/rtmdet.md` is the task-specific guide (install, train, metrics). This file is
> the discipline behind it, and it applies to the *next* model too — pose, tracking,
> segmentation — not only to RTMDet.

---

## 1. How to invoke the expert (the skill)

For any ML/DL task, adopt the stance of **an ML engineer who has shipped and debugged
computer-vision systems, and who has to defend every number to a reviewer**. Then name
— in the response, out loud — which sub-role you are invoking for the part of the task
at hand, because each one refuses a different kind of sloppiness:

| Sub-role | Invoke when | What it refuses to accept |
| --- | --- | --- |
| **detection / pose expert** | architecture, head, loss, label assignment, decoding | "the loss went down" as evidence; a head with no positives; an assigner nobody checked on a hand-made case |
| **evaluation expert** | any metric, any reported number | a metric whose frame of reference is unstated; a protocol named after something it is not (`AP50` called `mAP`); a single number from a noisy val set |
| **data-pipeline expert** | datasets, transforms, letterbox, splits, augmentation | a split that leaks; a coordinate mapping that is not owned by one object; augmentation nobody tested on a corner case |
| **performance expert** | speed, memory, device, batching | estimated latency; "it should be real-time"; a measurement whose conditions are unstated (warm-up, resolution, batch) |
| **reproducibility expert** | configs, seeds, checkpointing, reporting | a run that cannot be repeated; an artifact that disagrees with its own metadata; a default with no reason |

The stance in one line: **a design with no reference implementation is a guess, and a
number with no provenance is a claim.**

---

## 2. The laws (each one is a bug this project already paid for)

1. **One frame of reference.** Coordinates carry their frame — original image pixels,
   the padded network input, or normalised. Every mapping is owned by a single object
   with an `invert` (`Letterbox`), never re-derived at a call site. *Paid for:* the
   evaluation matched mapped predictions against unmapped labels, giving IoU 0 for
   every pair — **AP50 reported 0.000 for thirty epochs of a model that was learning**.
2. **A metric must be able to score a perfect model 1.0.** Write the *oracle* test: a
   stub that returns the ground truth must score exactly 1.0, and a stub that returns
   nothing must score 0.0. *Paid for:* law 1.
3. **Inference runs at the size the checkpoint trained at**, read from the checkpoint.
   *Paid for:* the Predict surface fed the network a native 1280-pixel frame while it
   learned on 640 — wrong scale statistics **and** 2.4× the compute.
4. **Splits are held out by group** (capture, specimen, session) — never by frame.
   `splits/*.txt` hashed by image path leaked 11 of 14 captures across train/val/test.
5. **A number is reported with its provenance** — corpus, split, threshold, epoch,
   device — **or it is not reported.** Say what it is *not* comparable with.
6. **Latency and memory are measured, and the conditions are stated** (device,
   resolution, batch, warm-up). A first call on MPS is ~8 s of compilation; steady
   state is the number.
7. **A prediction is never persisted as ground truth.** Only a human accept may make
   one (`SR-PREDICT-N01`, `SR-DATASET-007`).
8. **Failures are named, never silent.** No fallback device, no fallback checkpoint, no
   empty view: `ModelUnavailable` says *which* file and *which command* fixes it.
9. **The artifact must be what it says it is.** A checkpoint's metrics must describe
   the tensors beside them; best and final are separate files. *Paid for:* the final
   epoch overwrote the best checkpoint while keeping the best metrics.
10. **Config is data.** A dataclass (`TrainConfig`) plus argparse; every default
    carries a reason in a comment, because a default is a design decision.

---

## 3. Code shape (this repo's ML style — mirror it for the next model)

The convention is set by `daphnia/vision/rtmdet/`, and a new model should be a sibling
of it with the same five files:

```
model.py    architecture only (backbone / neck / head) — names from the reference
losses.py   the maths: coding, overlap, assigner, losses (pure tensor in → tensor out)
data.py     frame → tensor: letterbox, mapping object, augmentation, Dataset, collate
train.py    config, loop, evaluation, checkpoint IO, CLI
detect.py   the inference object a surface or a script calls (one call, one frame)
```

Rules that make that shape work here:

* **Docstrings argue, they do not narrate.** Every non-obvious choice says *why* and
  what the alternative would cost. The existing files are the model to copy.
* **Shape contracts in every docstring** — what goes in, what comes out, in which
  frame, in which unit. `(N, 4)`, `(N, C)`, `(B, 3, H, W)`, "image pixels",
  "loss per image".
* **Pure maths first, model second.** Losses and coders are functions of tensors with
  no model in sight, so they can be tested with one hand-computed box.
* **Heavy imports are lazy** (`torch`, `cv2` inside the function). The domain layer is
  Qt-free and importable without the ML stack — `SR-PREDICT-N02` depends on it.
* **Errors name the fix.** `ModelUnavailable` lists the path and the command.
* **Vocabulary follows mmdetection** (`backbone`, `neck`, `head`, `assigner`, `prior`,
  `num_classes`, `loss_cls`, `loss_bbox`, `bbox_coder`) so a reader can cross-reference
  the reference implementation line by line. Deviations are listed in `docs/rtmdet.md`
  §7 rather than left to be discovered.
* **Dataclasses for records** (`Box`, `Detection`, `Sample`, `Letterbox`,
  `TrainConfig`) — frozen where they are values.
* **The Device question is answered once** (`pick_device`) and asking for a device the
  machine does not have is an **error**, not a fallback.

---

## 4. What "done" means for an ML change (the test set)

No ML change is done without the tests that would have caught its failure mode:

| Test | It catches |
| --- | --- |
| **contract test, no heavy stack** (`skipUnless(torch)`) | a change that makes the app or the suite depend on PyTorch |
| **hand-computed numerics** (one box, a known IoU/GIoU, `distance2bbox` round-trip) | a formula quietly inverted |
| **oracle test** (perfect stub → 1.0, blind stub → 0.0) | a metric with no frame of reference (law 1) |
| **round-trip test** (encode/decode, save/load, letterbox `invert`) | a mapping that is not invertible |
| **shape/contract test** (what the network is actually fed) | inference at the wrong resolution (law 3) |
| **determinism test** (same seed → same sample; `set_epoch` changes the stream) | irreproducible augmentation |
| **shape test on the model** (one level per stride, one map per level) | a broken backbone/neck wiring |
| **a failing-then-passing test for every bug fixed** | the same bug returning |

And the *empirical* checks that no unit test replaces, run at least once per milestone:
`scripts/check_torch.py` (forward **and** backward on the real device), a short real
corpus run, and one end-to-end pass through the surface.

---

## 5. Reference implementations — read them, take the pattern, cite the source

**A design with no reference implementation is a guess.** Before writing a model, loss,
assigner, metric or training loop, pick the reference below and *say which one you read
and what you took from it*. Where a formula is non-obvious, a code comment names its
source (the reference file or the paper).

> **Verify before you rely.** The names below are from working knowledge of these
> projects; confirm the exact path/function at the version you are reading before
> quoting it in a comment or a doc.

### Detection (our current stack)

| Project | Licence | What to take | What *not* to copy |
| --- | --- | --- | --- |
| **open-mmlab/mmdetection** | Apache-2.0 | the `backbone`/`neck`/`head` split; `RTMDetHead.loss_by_feat` as the shape of a loss function; `BaseDenseHead`; the `prior_generator` idea; `SimOTAAssigner` | the `Config`/registry machinery — it is a framework for 100 models, not for one |
| **open-mmlab/mmyolo** | Apache-2.0 | RTMDet's *config* as the source of truth for hyper-parameters (CSPNeXt widths, neck, SimOTA's `center_radius`/`topk`); the mosaic/mixup pipeline | its inheritance-heavy config composition |
| **Megvii-BaseDetection/YOLOX** | Apache-2.0 | **the regression-bias init** (see §6 trap 4) and the head structure it comes from; SimOTA's ancestor (OTA) | the DataLoader/`Exp` singleton pattern |
| **facebookresearch/detr** | Apache-2.0 | the cleanest separation of *assigner* (`matcher`), *loss* (`SetCriterion`) and model in the field — a good model for `losses.py` | its transformer training recipe; it needs far more data than we have |

### Pose (our next stack)

| Project | Licence | What to take | What *not* to copy |
| --- | --- | --- | --- |
| **open-mmlab/mmpose** | Apache-2.0 | the **top-down two-stage** arrangement (detector → pose) that our `localize_well` → pose split already mirrors; RTMPose's structure; the heatmap **and** SimCC decoding options; **OKS** as the keypoint metric | its config/registry machinery, as above |
| **microsoft/… HRNet** (high-resolution nets for pose) | MIT/Apache-2.0 (check) | the argument for *why* pose needs resolution kept high — relevant to our eye-apex/spine-base precision | the multi-branch complexity unless the corpus justifies it |
| **openpifpaf / AlphaPose / SLEAP** | various (check each) | practical single-animal pipelines, crop-and-upscale before pose, tracking-free inference | tracker machinery we do not need yet |
| **DeepLabCut / SLEAP** (animal pose, in-lab tools) | check each | how *animal* pose tools structure labelling output and evaluation; PCK/OKS reporting conventions | their GUI-centric architecture |

### Training infrastructure and style

| Project | Licence | What to take | What *not* to copy |
| --- | --- | --- | --- |
| **pytorch/vision** (`references/detection`) | BSD-3 | the shape of a train/eval loop and of a checkpoint that carries its own config; `torch.testing.assert_close` for numerics | the older API style |
| **huggingface/pytorch-image-models (timm)** | Apache-2.0 | `create_model`-style **model registry**, EMA, layer-wise LR decay, augmentation choices documented with reasons | the sheer breadth — we add one model at a time |
| **open-mmlab/mmengine / facebookresearch/detectron2** | Apache-2.0 | the *separation* of runner / hooks / checkpointing / evaluation, i.e. what to factor out *when* a second model arrives | adopting a whole framework for one detector |
| **Lightning-AI/pytorch-lightning** | Apache-2.0 | the `LightningModule`/`DataModule` split as a **design idea** (model ≠ loop ≠ data), and its reproducibility utilities; we can keep the idea without the dependency | the dependency itself |
| **Lightning-AI/torchmetrics** | Apache-2.0 | how to document a metric's **protocol** so two numbers cannot be silently compared | its abstractions for our two metrics |
| **karpathy/nanoGPT / minGPT** | MIT | one *config dataclass* + a `train.py` where each concept fits on a screen — the closest style to our `train.py` + `TrainConfig` | — |
| **lucidrains/\*** (e.g. `vit-pytorch`) | MIT | single-file readability; a module that can be read top to bottom | minimal testing habits |
| **huggingface/transformers** | Apache-2.0 | docstrings that state the **tensor shapes and the semantics** of each argument; config as the single source of truth | the framework size |
| **albumentations** | MIT | transform-pipeline design (a list of named transforms, each with a probability) | its dependency for a five-augmentation pipeline |

### Papers to cite where a formula is non-obvious

| Formula / idea | Source |
| --- | --- |
| CSPNeXt backbone, CSPNeXtPAFPN neck, shared SepBN head, the whole detector | **RTMDet** (arXiv 2212.07784) |
| dynamic label assignment with `k = ⌊Σ top-10 IoU⌋` | **OTA / SimOTA** (arXiv 2103.14258) |
| the IoU-valued soft classification target | **GFL / Quality Focal Loss** (arXiv 2006.04388) |
| point-based pose with heatmap *and* SimCC decoding (our next model) | **RTMPose** (arXiv 2303.07399), **SimCC** (arXiv 2107.03332) |
| the conceptual ancestor of the localise-then-pose cascade | **DeepPose** (arXiv 1312.4659) |
| the keypoint metric (the analogue of IoU for points) | **OKS**, in the COCO keypoint evaluation |

---

## 6. The traps, catalogued from this repo (do not re-learn them)

1. **A metric in an unstated frame.** Predictions mapped back, labels left in the padded
   frame → IoU 0 for every pair. *Rule:* law 1 + the oracle test.
2. **A long-lived Qt object connected to a lambda.** `shared_service()` outlives the
   window; a lambda has no receiver QObject for PySide6 to auto-disconnect, so the
   callback reached freed C++ widgets (`libshiboken: Internal C++ object already
   deleted`). *Rule:* connect a **bound method**; test by deleting the page
   (`shiboken6.delete`) and then emitting.
3. **Inference at a different size from training.** Native 1280 into a 640 model: wrong
   scale statistics *and* 2.4× the compute. *Rule:* law 3.
4. **An untrained head that starts by predicting degenerate boxes.** With all-zero
   regression bias the boxes have zero area, so IoUs (and therefore the quality labels
   and the assignment) are meaningless from step 0. The YOLOX/RTMDet lineage
   **initialises the regression bias to 1.0** — i.e. the initial box is the size of the
   stride — so the very first assignment is meaningful. *Check this before blaming the
   learning rate.*
5. **A chain of stages fed the same input.** `CSPNeXt.forward` used a list
   comprehension over its stages: harmless-looking, and every stage saw the *stem's*
   output. *Rule:* the model shape test.
6. **An augmentation allowed to overrun its geometry.** A mosaic tile scaled past its
   quadrant → broadcast error (and silently clipped boxes if it had "worked").
   *Rule:* assert every augmented sample's boxes are inside the frame.
7. **A "best" checkpoint overwritten by the last epoch, keeping the best metrics.**
   *Rule:* law 9.
8. **A promising first result that is really a lucky draw.** One held-out capture, 15
   frames: AP50 swung 0.16↔0.81 epoch to epoch with no trend. *Rule:* report the spread,
   not the maximum, and say the set cannot resolve differences.
9. **A mixed-channel conda environment.** `opencv`/`pytorch` from `defaults` with
   `libprotobuf`/`libsqlite` from `conda-forge` produce *unrelated-looking* missing-symbol
   failures in `cv2`, `_sqlite3` **and** `import torch`. *Rule:* `conda list
   --show-channel-urls` first; one channel per environment (`techContext.md` 9-10).

---

## 7. Licence hygiene

* **Apache-2.0 / MIT / BSD** (mmdetection, mmpose, mmyolo, YOLOX, timm, torchvision,
  DETR, Lightning, nanoGPT): studying and adapting is fine; **attribute** the source in
  the code comment or the doc.
* **AGPL-3.0 (ultralytics YOLO)**: read it for *ideas*, **never copy code into this
  repository.** One copied function would relicense the whole deliverable.
* We **re-implement from the paper**, and every deviation from a reference is written
  down (`docs/rtmdet.md` §7) so nobody compares our numbers with the reference's by
  mistake. That section is not a disclaimer — it is part of the result.

---

## 8. How ML results are reported here (the honest default)

1. **One number plus its provenance**: corpus, split, threshold, epoch, device.
2. **The threshold table, not a single score** — the operating point is what an operator
   experiences; AP's threshold is not it.
3. **The validation set's size and the epoch-to-epoch spread** when it is small.
4. **What was *not* measured** — said out loud rather than left implied.
5. **The comparison you are allowed to make, and the one you are not.** Here: another
   run of *this* script on *this* corpus. Never a paper's number.
6. **Latency with its conditions** (device, resolution, batch, warm-up).

Anything less is a claim, not a measurement — and this project's whole premise is that
its numbers can be defended.

---

## 9. Invocation summary (run this for any ML change)

1. **Name the task family** (detection / pose / segmentation / …) and **pick the
   reference implementation** to read first; say which, in the response.
2. **State the metric, the split and the frame of reference** *before* touching the
   model. If you cannot state them, you are not ready to write the model.
3. **Write the data contract** (shapes, frame, units) as a docstring.
4. **Pure maths first**, with a hand-computed test.
5. **Model, then loop, then report** — each with the test that would fail if it were
   wrong (see §4).
6. **Add a failing-then-passing test for every bug fixed.**
7. **Update the task doc** (`docs/rtmdet.md`, or its successor) with the deviation list
   and the measured numbers.
8. **Then** update `activeContext.md` / `progress.md` — the memory bank records the
   outcome, it does not replace the work.

---

## 10. Concepts → where they live here (so explanations stay grounded)

The operator is **new to machine learning**. When a concept comes up, do not answer in
the abstract: point at the file in this project that *is* that concept, and use this
vocabulary. This is also the map to check when a task is described in textbook terms.

| Concept | Where it actually is here |
| --- | --- |
| **Object detection** — "where are the animals" | `daphnia/vision/rtmdet/`; the Localize step (`vision/detection.localize_well`); the Predict surface |
| **A bounding box / a label** | `annotation.BOX_TRAIT`, `corpus_boxes`, `vision/export.Box` (one box = one **individual**) |
| **An individual** — which animal a number is *about* | the same concept in four places: `observations.individual_index` (Study: "Individual 3"), `corpus_boxes` (one box per specimen), `vision.tracking.Track` (Predict: number + colour + mean over repeats), and `vision.individuals` (the Bench's **grouping rule**: a measurement belongs to the box containing its **midpoint**, innermost wins, `None` = nobody). Stored on the row as `measurements.individual_index` — a different question from `specimen_index`, which is only *which row on the frame this is* |
| **Anchor / prior** — the candidate locations | `model.RTMDet.level_priors` — **one point per location**, sized by the stride |
| **The predicted quantity** — four distances to the box sides | `losses.distance2bbox` / `bbox2distance`; *not* a distribution (no DFL) |
| **Label assignment** — which locations are "right" | `losses.SimOTAAssigner` (dynamic k from the top-10 IoUs) |
| **Loss** — how wrong the network is | `losses.QualityFocalLoss` (classification with an **IoU-valued soft target**) + `losses.GIoULoss` (boxes) |
| **IoU** — how much two boxes overlap | `losses.bbox_overlaps` (also `mode="giou"`) — the same idea `ui.widgets.box_iou` uses in the labeller |
| **Confidence / score** | `head.cls` → `sigmoid`; the surface's `spin_score`, and why its calibration matters (`docs/rtmdet.md` §5) |
| **NMS** — removing duplicates | `losses.batched_nms` (NumPy, class-aware) |
| **Letterbox / resizing / normalisation** | `data.letterbox` + `data.Letterbox` (the mapping *object*) + `data.to_tensor` |
| **Augmentation** | `data` mosaic, `horizontal_flip`, `colour_jitter` |
| **Hyper-parameter / config** | `train.TrainConfig` (a dataclass) + the CLI in `train.build_parser` |
| **Learning rate schedule / warm-up** | `train.learning_rate_at` |
| **Epoch / step / batch** | the loop in `train.train`; `History` records the curve |
| **Checkpoint** — the saved weights | `train.checkpoint_payload` / `save_checkpoint` / `load_checkpoint`; `models/rtmdet_daphnia.pt` is *best*, `…_last.pt` is the final epoch |
| **Validation vs test split** | `vision/export.py` — `Export.splits`, `Export.groups()`, `Export.regroup()`, `Export.warnings()` — and why by **group** (`memory-bank/activeContext.md`) |
| **Precision / recall / F1 / AP** | `train.evaluate` + `train._interpolated_ap`; `docs/rtmdet.md` §5 reads them out |
| **Overfitting / underfitting** | the loss curve in the report vs the val curve — and the small-val-set caveat |
| **Keypoints / pose vector** | **built** — `daphnia/vision/pose_corpus.py` (the keypoint reader), `daphnia/vision/rtmpose/` (the model), `vision/keypoints.Keypoints`, `labels.json`'s `keypoints` block |
| **Top-down vs bottom-up pose** | top-down, **built**: **Localize first** (RTMDet) → pose inside the box (`rtmpose/detect.py`: `PoseModel.detect_frame`) |
| **Heatmap vs regression decoding** | decided: **SimCC** 1-D coordinate classification (`rtmpose/losses.py` + `rtmpose/model.SimCCHead`, arXiv:2107.03332), not a heatmap — `docs/rtmpose.md` §5 |
| **OKS / PCK** — keypoint accuracy metrics | **built** — `rtmpose/losses.oks` / `pck`, reported by `rtmpose/train.pose_metrics`; sigmas are ours (`losses.DEFAULT_SIGMAS`) and stated, so the number is comparable only with another run of this script |
| **Inference latency / throughput** | `Detector` times each pass (`ms/frame`); `scripts/check_torch.py` proves a forward **and backward** pass works on the real device — it does *not* measure fps |
| **Device (CPU vs MPS)** | `train.pick_device` — MPS when there is one, and an **error** when asked for one there isn't |




