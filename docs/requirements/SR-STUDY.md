# SR-STUDY — governed longitudinal study requirements

Serves: **O1** (standardized acquisition), **O4** (desktop app), and the shared
end goal: **studies that last days to weeks**. Launched with
`python run.py study` — **today's application, unchanged in behaviour**.

This is the surface where the strict gates are a **feature**: order cannot be
bypassed, so the resulting numbers are reproducible and reviewer-defensible.

## Gate policy — `StudyPolicy` (as implemented today)

A step may be entered only when every earlier step's fact holds **for the active
project**:

| Fact | Produced by | Why it gates |
| --- | --- | --- |
| `study` | Setup | decides **where the images are saved** |
| `connection` | Setup (runtime) | nothing captured/calibrated off a stream that isn't there |
| `calibration` | Calibrate | an image without a scale can never be measured (global: hardware) |
| `observation` | Acquire | Select needs bursts to compare |
| `selection` | Select | exactly one best frame per well / individual |
| `measurement` | Annotate | Analytics needs something to plot |

Select deliberately does **not** require the connection: once frames are on disk,
the phone can be put away and the work continues offline.

## Functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-STUDY-001` | Entering a step **shall** be refused while its required facts are unmet, and the reason **shall** be shown on screen. | The reproducibility guarantee of the surface. | `T` `tests/test_workflow.py` | implemented | O1 |
| `SR-STUDY-002` | Facts **shall** be counted **per active study**, so a new project never inherits another project's captures, selections or measurements. | Cross-contamination would silently corrupt results. | `T` `tests/test_workflow.py` | implemented | O1 |
| `SR-STUDY-003` | A study **shall** declare organism, organism kind, target mode, timecourse, interval and acquisition protocol. | These configure every downstream screen via `StudySpec`. | `T` `tests/test_study.py` | implemented | O1 |
| `SR-STUDY-004` | The surface **shall** resolve the destination folder and the active scale before every capture and display both. | "Where does this image go" and "is this calibrated" must be answered on screen. | `D` | implemented | O1 |
| `SR-STUDY-005` | Frames **shall** be stored as bursts under the study's destination and re-visiting a target **shall** append a burst, never overwrite. | A specimen is improved, not lost. | `T` `tests/test_capture.py` | implemented | O1 |
| `SR-STUDY-006` | Exactly one frame per target **shall** be kept, and the sharpest-frame rule **shall** be available as an automatic choice. | One well -> one measurement, the study's central invariant. | `T` `tests/test_repository.py` | implemented | O1 |
| `SR-STUDY-007` | Every kept frame **shall** be annotated for the study's traits (`length`, `width`) at the study's landmarks. | The endpoint definition of the thesis. | `T` `tests/test_ui_smoke.py` | implemented | O2 |
| `SR-STUDY-008` | The surface **shall** declare a plan of **timepoints** and compute which session is due now, which is overdue, and which is next. | Long studies fail without "what do I do today". Currently only `interval_hours` exists. | `T` | proposed | O1 |
| `SR-STUDY-009` | The surface **shall** show a **daily/weekly queue** derived from that plan, per study. | The operational face of a multi-week study. | `D` | proposed | O1 |
| `SR-STUDY-010` | A run **shall** be resumable at the well / individual it stopped on. | Long sessions are interrupted; `current_well_index` already supports this. | `T` `tests/test_capture.py` | implemented | O1 |
| `SR-STUDY-011` | Select and Annotate **shall** be separate steps. | Deciding the best frame and measuring are different judgements. | `I` | implemented | O1 |
| `SR-STUDY-012` | The surface **shall** support multi-well plates (6/24/96) and a single-specimen counter as target modes. | Two acquisition regimes the study must cover. | `T` `tests/test_plates.py` | implemented | O1 |
| `SR-STUDY-013` | A study **shall** be closable: prune discarded, export measurements, write a manifest, mark closed. | The graceful end of a run. | `T` `tests/test_maintenance.py` | implemented | O4 |
| `SR-STUDY-014` | Analytics **shall** expose the study's measurements and export them as CSV. | The reporting end of the pipeline. | `T` `tests/test_ui_smoke.py` | implemented | O4 |
| `SR-STUDY-015` | Study setup **shall** be restartable from a saved, versioned template. | Repeating a longitudinal protocol must take seconds. | `T` `tests/test_templates.py` | implemented | O1 |
| `SR-STUDY-016` | The surface **shall** support treatment assignment on top of plates. | Required for dose-response designs. **Currently "still to design"** in the codebase. | `T` | proposed | O1 |

## Non-functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-STUDY-N01` | The surface **shall** open on the Overview and **shall** never resume behind the operator's back. | No invisible state. | `D` | implemented | O4 |
| `SR-STUDY-N02` | A study-less BENCH/DATASET session **shall** be promotable into this surface's data model without re-measurement. | Keeps ad-hoc work recoverable into a governed study. | `T` | proposed | O4 |
| `SR-STUDY-N03` | The session **shall** lock the optical preset for its duration. | A study must not drift between scales mid-run. | `T` `tests/test_repository.py` | implemented | O1, O3 |
| `SR-STUDY-N04` | Creating a study **shall** be possible with a single default action, with the mode axes available as an advanced expansion. | All 12 throwaway studies on 2026-10-15 were identical on every axis: the interview, not the axes, was the cost. | `D` | proposed | O4 |

## Open issues

- `SR-STUDY-008` / `SR-STUDY-009` are the missing pieces for the "days or weeks"
  goal and should be scoped as one feature (plan + queue).
- `SR-STUDY-016` (treatments) is unmodelled; it needs a table and a plate-builder
  UI before it can be accepted.
- `SR-STUDY-N04`: decide whether the advanced axes move behind an "Advanced"
  drawer or into a template-first flow.
- `SetupPlates` exists as a stub and is **not** wired into `PAGE_CLASSES` or
  `workflow.STEPS` — resolve by either implementing it inside Setup or deleting it.
- The Study surface is the **only** surface that may produce publication
  numbers; keep it that way.

