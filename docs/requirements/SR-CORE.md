# SR-CORE — shared backend requirements

Serves: **O2** (CV pipeline), **D2** (codebase + weights). Consumed by every
surface. No window of its own.

The shared backend is what makes "different apps, same repo" viable: it owns the
data model, the pipeline stages, the optical scale and the export formats.

## Functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-CORE-001` | The backend **shall** expose the pipeline as the ordered stages `Ingest -> Localize -> Calibrate -> Predict -> Derive -> Persist -> Aggregate`, each as an independently callable unit. | One vocabulary for the thesis and the code. | `I` `daphnia/pipeline/steps.py` | implemented | O2 |
| `SR-CORE-002` | The backend **shall** persist a measurement as one row per `(frame, specimen, trait)` and update in place on re-save. | Prevents inflated tables; matches `idx_measurement_asset_trait`. | `T` `tests/test_repository.py` | implemented | O2 |
| `SR-CORE-003` | Every measurement **shall** store both landmark points (`p1_*`, `p2_*`) in source-image pixels, not only the derived distance. | The points *are* the training ground truth. | `T` `tests/test_repository.py` | implemented | O2, D1 |
| `SR-CORE-004` | The backend **shall** distinguish `source = 'manual'` from `source = 'model'` and allow both on the same frame. | Manual labels and predictions must coexist to measure agreement. | `T` `tests/test_repository.py` | implemented | O2, O3 |
| `SR-CORE-005` | The backend **shall** allow a session and an observation to exist with **no study** (`study_id`, `plate_id` NULL). | This is the mechanism that lets BENCH/DATASET exist without a project. | `T` `tests/test_repository.py` | accepted | O4 |
| `SR-CORE-006` | The backend **shall** allow a study-less session to be **promoted** onto a study later, without rewriting its observations or measurements. | Provenance without the setup cascade. | `T` proposed | proposed | O1, O4 |
| `SR-CORE-007` | `init_db()` **shall** be idempotent and **shall** apply all pending column migrations and landmark backfills on every launch. | Databases from older builds must never break queries. | `T` `tests/test_db.py` | implemented | D2 |
| `SR-CORE-008` | The backend **shall** provide a dataset export containing, per annotated specimen: image path, image size, specimen index, trait, the landmark coordinates, and the applied scale. | D1 cannot be produced from a CSV that drops the points. | `T` `tests/test_maintenance.py` | proposed | D1, O3 |
| `SR-CORE-009` | The dataset export **shall** partition data by **specimen and study**, never by random frame, and **shall** record the partition in the manifest. | Burst frames are near-duplicates; a frame-level split leaks and inflates validation scores. | `A` + `T` | proposed | O3 |
| `SR-CORE-010` | The backend **shall** report the scale actually applied next to any derived length, including for rows measured under a since-changed preset. | Recalibration must not silently reinterpret old data. | `T` `tests/test_repository.py` | implemented | O1, O3 |
| `SR-CORE-011` | The backend **shall** be importable and usable headless (no Qt, no display) for scripts and notebooks. | Training and batch analysis run outside the GUI. | `I` import guard | implemented | D2 |
| `SR-CORE-012` | The backend **shall** expose read models via `from_row()`-style constructors that tolerate older schema revisions. | Rows from older databases must remain readable. | `T` `tests/test_study.py` | implemented | D2 |
| `SR-CORE-013` | The application **shall** offer exactly one launch point per surface, and that launch point **shall** open exactly one window. | One backend, several entries - the choice must be explicit and in one place. | `T` `tests/test_surfaces.py` | implemented | O4, D3 |
| `SR-CORE-014` | A surface that is not built yet **shall** open its own specification, never another surface's window. | A launch point must never silently do the wrong thing. | `T` `tests/test_surfaces.py` | implemented | O4 |
| `SR-CORE-015` | The surface registry **shall** be usable without a display (Qt-free). | The hub renders it and the tests check it headless. | `T` `tests/test_surfaces.py` | implemented | D2 |
| `SR-CORE-016` | Every measurement **shall** record the relative scale uncertainty that applied, so the `+/-` travels with the result and cannot be re-derived wrongly later. | A length without its method uncertainty cannot be compared across timepoints or calibrations. | `T` `tests/test_calibrations.py` | implemented | O3 |
| `SR-CORE-017` | The camera connection **shall** be reported as one of four distinct states - disconnected, connecting, connected-but-silent, live - and an open socket **shall never** be equated with a picture. | "Connected" while nothing arrives is the failure an operator cannot otherwise see. | `T` `tests/test_streaming.py` | implemented | O1 |
| `SR-CORE-018` | The camera **shall** be connectable and disconnectable from **any** surface, not only from one step of the study workflow. | An Optics or Bench window truthfully reported "no connected camera" because it had never opened a reader. | `T` `tests/test_ui_smoke.py` | implemented | O1, O4 |
| `SR-CORE-019` | A failed or dropped connection **shall** state its reason, and a dropped connection **shall** clear the stale preview. | A stale picture is worse than a blank one, and "not connected" without a reason is not actionable. | `T` `tests/test_streaming.py` | implemented | O1 |
| `SR-CORE-020` | A **failed frame read shall not end the capture** unless it persists (`MAX_FAILED_READS` in a row), and the number of frames that could not be decoded **shall** be stated while the stream is still up. | Measured 2026-10-19: a decode failure and a momentary starvation *both* return `False` and the frames after them are fine - so a live stream was being reported as "Stream ended or disconnected". | `T` `tests/test_streaming.py` | implemented | O1 |
| `SR-CORE-021` | The app **shall** report the publisher's video codec (asked from the MediaMTX control API) and, when it is one this decoder pays frames for (H.265/HEVC), **shall** name the setting that fixes it - not only FFmpeg's own log. | Measured 2026-10-19: an H.265 publisher loses 7-21 of the first 120 frames, at 24.1 fps against 27.5 fps for H.264 - and the fix is a setting on the phone, which a stderr flood cannot tell an operator. | `T` `tests/test_streaming.py` | implemented | O1 |

## Non-functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-CORE-N01` | SQLite access **shall** enable `journal_mode=WAL` and a non-zero `busy_timeout`. | Two surfaces as separate processes would otherwise deadlock on "database is locked". | `T` `tests/test_db.py` | proposed | O4 |
| `SR-CORE-N02` | Data file paths **shall** derive from `daphnia.paths` only — no `__file__` or `cwd` arithmetic in modules. | Single source of truth for the storage layout. | `I` | implemented | D2 |
| `SR-CORE-N03` | The backend **shall** keep the landmark vocabulary data-driven (organism -> keypoint names + endpoint), not hard-coded per species. | Adding a smaller organism must not require editing logic. | `T` `tests/test_study.py` | proposed | O2 |
| `SR-CORE-N04` | Length derivation **shall** report values in micrometres as the canonical unit, with millimetres only for display. | One unit in the store; formatting is a view concern. | `T` `tests/test_units.py` | implemented | O2 |

## Open issues

- `SR-CORE-009`: the partition policy needs a concrete decision (specimen-level
  holdout vs study-level holdout) before the first model run.
- `SR-CORE-N01`: must land **before** any second surface runs as its own process.
