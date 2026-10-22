# Product Context — why DaphniaScope exists

## The problem

Manual body-size measurement under a stereomicroscope is the bottleneck:

- **Throughput:** manual alignment + measurement caps sample sizes across
  conditions; hundreds of moving specimens across longitudinal timepoints is
  impractical by hand.
- **Observer bias:** inconsistent measurement lines between analysts degrade
  inter-laboratory reproducibility.
- **Specimen variability:** live animals swim, tilt, and are occluded by exuviae
  (molts) and detritus.

## Three audiences — and why one pipeline could not serve them

Discovered 2026-10-15 from the project's own artefacts (12 studies created in one
afternoon, 293 frames captured, 5 measurements persisted):

| Audience | Job to be done | What the single-pipeline design did to them |
| --- | --- | --- |
| **Colleagues** | "Measure this, now, without a study" | Forced through study → connection → calibration → acquire → select → annotate. And **no way to open an image from disk** (only the live stream could be measured). |
| **Thesis / model** | "Give me a big annotated corpus to train keypoints" | `select_best_frame()` keeps exactly **one** frame per target and trays the rest; `prune_discarded()` then **deletes those frames from disk**. The CSV export **omits `p1_*`/`p2_*`** — the ground-truth coordinates. |
| **All** | "Studies that last days/weeks" | `interval_hours`, `day_number`, `current_well_index` exist, but there is no plan of **timepoints** and no "what is due today". |
| **Smaller organisms** | "Much smaller than *Daphnia*" | One global active `px_per_um` profile; `ANATOMY` hard-codes animal/plant landmarks; profile identity does not pin the optics (observed: same `profile_id` carried `px_per_um` 0.783 then 0.147 — 5.3x). |

## The root cause

The app modelled **one** modality — the governed longitudinal study — and encoded
it as hard gates in `daphnia/workflow.py`. Every intent was forced through it.
Twelve throwaway studies named `test`, `max`, `111111` in five hours is the
*design's* output, not sloppiness: "let's test the system" cost a full project
setup, and the only way to reset was `make db-reset`.

The three existing mode axes (`organism_kind`, `target_mode`, `timecourse`) are
axes of *how a study varies*. They were **never exercised** (12/12 studies were
`individuals`/`longitudinal`/`standard`). The axis that actually varied was
**intent / rigor** — which does not exist in the model.

## The product answer

Make **intent** the top-level axis. Five surfaces over one backend:

```
                      HOME (hub / launcher)
     +----------+----------+----------+-----------+----------+
     v          v          v          v           v          v
  OPTICS     BENCH     DATASET    PREDICT      STUDY      (the detector,
 presets +   measure   corpus +   watch the   guarded     before a
 objectives  now       export     detector    7-step      study uses
 (shared HW) (no study)(thesis)   (Localize)  run         it)
```

- **STUDY keeps today's strict gates** — there, they are a feature: that is how
  reviewer-defensible numbers are produced.
- **OPTICS is pulled out of the linear path.** A preset is hardware; the code
  already treats the `calibration` fact as global, not per-study.
- **BENCH, DATASET and PREDICT are gate-light** and write to the *same* store with
  `study_id = NULL` (PREDICT writes nothing at all).
- **PREDICT exists because the corpus needed a mirror.** Labelling in the dark makes
  it impossible to tell "the annotations are bad" from "the model is bad"; watching
  the detector on a live frame turns that into a five-second observation.
- **Promotion path:** a Bench session or Dataset can be **attached to a study
  later** (`session.study_id := ...`). Provenance without the cascade.

## UX goals

1. **Time-to-first-number** measured in seconds, not screens. Under Bench: one
   button, one image, two clicks.
2. **Honest precision.** Show uncertainty, not false confidence. At 0.147 px/um,
   1 px ~ 6.8 um — the readout must not imply more than it knows.
3. **Never destroy the corpus.** Curation is not deletion.
4. **Progressive disclosure.** Metadata needed only for *aggregation* must not
   gate the *measurement*.
5. **Reproducibility where it matters.** In Study, gate order is enforced and
   visible; elsewhere, stay out of the way.

## Anti-goals

- Four apps drifting into four UXs. Mitigation: one shell, one store, one
  `ImageCanvas`, one `StreamService`.
- A "quick" mode that produces numbers nobody can defend. Mitigation: Study
  remains the **only** surface that yields publication figures; Bench marks
  unscaled results explicitly.
