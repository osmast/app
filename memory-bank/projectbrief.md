# Project Brief — DaphniaScope

> Foundation document. Shapes every other memory-bank file. Source of truth for scope.

## What this is

**DaphniaScope** — automated, non-invasive **body-length measurement for living
*Daphnia magna*** in multi-well plates. An imaging + computer-vision pipeline
wrapped in a Qt desktop application. Built for the SNSF "Aquaculture Systems"
project at **ZHAW Wädenswil**.

It replaces manual stereomicroscope sizing: high-throughput longitudinal growth
assays with a reproducible, objective endpoint.

## The endpoint (non-negotiable definition)

> **Body length = distance from the anterior margin of the compound eye to the
> base of the apical spine.** The spine itself is excluded. Lateral view assumed.

This definition is fixed. Every landmark vocabulary, every model, and every
requirement in `docs/requirements/` serves it.

## The end goal (shared by all stakeholders)

Enable **studies that last days to weeks**: the same specimens followed over
time, measured reproducibly, at a scale manual sizing cannot reach — and
eventually for organisms far smaller than *Daphnia*.

## Thesis anchor — `latex/research-outline/Chapters/Chapter1.tex`

**Primary goal:** design, evaluate and package an automated CV pipeline and
software prototype for non-invasive body-length measurement of living
*Daphnia magna* in multi-well plates, replacing manual stereomicroscopy.

### Specific objectives (traceability keys `O1..O4`)

| Key | Objective |
| --- | --- |
| **O1** | Standardized image **acquisition** workflow for individual daphnids in multi-well plates |
| **O2** | CV pipeline (classical + deep learning) to identify keypoints and compute body length |
| **O3** | **Evaluate** accuracy, robustness and inference latency against an expert-annotated benchmark dataset |
| **O4** | Package the validated algorithms into an accessible, **standalone desktop application** for lab end users |

### Deliverables

| Key | Deliverable |
| --- | --- |
| **D1** | Curated, annotated **benchmark dataset** of *D. magna* well images (boxes + keypoints) |
| **D2** | Open-source Python codebase + **trained model weights** |
| **D3** | Packaged **desktop GUI** for routine high-throughput growth profiling |

### Fixed constraints

- Submission deadline **10.12.2026**; presentation **18.12.2026**.
- Timeline WPs: optical setup & acquisition · dataset annotation & ground truth ·
  model development & training · validation vs. manual reference · software & UI.
- Keypoint accuracy target: resolution >= 1200 dpi equivalent (~21.2 um/px) for
  neonate-grade landmark accuracy (Agatz 2015); adults are more tolerant.
- Documented hard cases: neonate transparency, group crowding, meniscus
  curvature, motion blur, detritus/exuviae occlusion.

## Surfaces — one backend, several launch points

The repository is **one repo, one Python package (`daphnia`), one SQLite store,
one `recordings/` tree**. It exposes several **launch points** (surfaces), each
with its own requirement document:

| Surface | Launch point | Requirement doc | Primary audience |
| --- | --- | --- | --- |
| **CORE** | (library, not a window) | `SR-CORE.md` | — shared domain + persistence |
| **OPTICS** | `python run.py optics` | `SR-OPTICS.md` | whoever sets up the microscope |
| **BENCH** | `python run.py bench` | `SR-BENCH.md` | colleagues: quick number, no study |
| **DATASET** | `python run.py dataset` | `SR-DATASET.md` | the thesis: build the training corpus |
| **PREDICT** | `python run.py predict` | `SR-PREDICT.md` | the thesis: watch the detector work (the Localize step) |
| **STUDY** | `python run.py study` (today's app) | `SR-STUDY.md` | the end goal: days/weeks studies |

> Splitting into separate *repositories* is explicitly **out of scope**: the
> shared domain layer is the value. Only entry points and gate policies differ.

## Non-negotiables

1. **No unscaled number.** A measurement is meaningless without the scale that
   produced it. `measurements.px_per_um` must always record the applied scale.
2. **Provenance.** Every measurement traces to an image, a scale and an operator
   (`asset_id`, `profile_id`, `measured_by`, `source`).
3. **The corpus is not disposable.** Frames that carry labels are never deleted
   as a side effect of curation (see `SR-DATASET.md`).
4. **Order cannot be bypassed *in a governed study*** — but that strictness is
   the *Study* surface's policy, not a property of the whole application.

## Scope

**In scope:** acquisition from a phone via SRT/MediaMTX; optical calibration;
keypoint annotation; **dataset management (one corpus per model: 2/4 keypoints and
bounding boxes)**; length derivation; persistence; longitudinal aggregation;
CSV/manifest export; the dataset export needed to train the model; **training the
detector (RTMDet) and running it live** (the Predict surface — the Localize step).

**Out of scope (this thesis):** automated stage tracking; in-situ analysis;
treatment-assignment builder on top of plates (noted as "still to design" in the
codebase); cloud/multi-user deployment. The keypoint model (**RTMPose**) is *not* out
of scope but is not built yet: it needs its own four-keypoint corpus, and RTMDet
comes first because its boxes are what a pose model is given.

> **Changed 2026-10-16.** Real-time bounding-box prediction used to be listed as out of
> scope here. It is now built (`daphnia/vision/rtmdet/`, `SR-PREDICT.md`) — the
> labelling side already existed, and "watch the detector" turned out to be the only
> way to answer *whether* the corpus is good enough.
