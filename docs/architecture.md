# Architecture

## Pipeline

The system is organised around seven steps (`daphnia/pipeline/steps.py`):

```
Ingest -> Localize -> Calibrate -> Predict -> Derive -> Persist -> Aggregate
```

| Step | Module | Responsibility |
| ---- | ------ | -------------- |
| Ingest | `daphnia.streaming` | Frames from the MediaMTX stream (`StreamReader`). |
| Localize | `daphnia.vision.detection` | Isolate the specimen in the well (Mode A/B). |
| Calibrate | `daphnia.vision.calibration` | Pixels -> micrometres via the optical profile. |
| Predict | `daphnia.vision.keypoints` | Eye apex and spine base. |
| Derive | `daphnia.pipeline.derive` | Body length in micrometres, reported in mm too (`daphnia.units`). |
| Persist | `daphnia.pipeline.persist` + `daphnia.db` | SQLite writes. |
| Aggregate | `daphnia.pipeline.aggregate` | Longitudinal series, dose-response, survival. |

## Directory map

```
app/
├── run.py                  entry point (python run.py)
├── Makefile                convenience targets (make run / test / stream)
├── assets/
│   ├── ui/                 Qt Designer .ui files (one per screen)
│   └── theme/              daphniascope.qss  (the QSS template; colours live in
│                           daphnia/ui/theme.py as light/dark tokens)
├── config/mediamtx.yml     version-controlled MediaMTX configuration
├── daphnia/                Python package
│   ├── paths.py            every project directory, in one place
│   ├── settings.py         runtime settings (env-overridable)
│   ├── protocols.py        named acquisition presets (burst + QA trade-off)
│   ├── templates.py        versioned study templates
│   ├── maintenance.py      prune / export / consolidate a study
│   ├── units.py            um <-> mm formatting for reports
│   ├── db/                 connection + schema.sql  (Persist)
│   ├── streaming/          mediamtx, reader, sources (Ingest)
│   ├── vision/             calibration, detection, grid, images, keypoints
│   ├── pipeline/           steps, derive, persist, aggregate
│   └── ui/                 app, main_window, loader, theme, pages/
├── datasets/               raw/ processed/ annotations/   (git-ignored)
├── models/                 trained weights                (git-ignored)
├── recordings/             burst captures / segments      (git-ignored)
├── scripts/                setup_mediamtx.sh, dev_publish_stream.sh,
│                           check_stream.py, init_db.py
├── docs/
└── tests/                  stdlib unittest
```

## Layers

* **UI** (`daphnia.ui`) — Qt only. Screens are separate `QWidget`s loaded from
  `.ui` files; `MainWindow` is the parent controller that owns the stacked widget
  and the navigation. No pipeline logic lives in the pages.
* **Domain** (`daphnia.vision`, `daphnia.pipeline`, `daphnia.protocols`,
  `daphnia.templates`, `daphnia.study`) — pure Python; no Qt imports, so it is
  unit-testable and reusable from scripts/notebooks.
* **Infrastructure** (`daphnia.db`, `daphnia.streaming`, `daphnia.paths`,
  `daphnia.settings`, `daphnia.maintenance`) — persistence, transport,
  configuration and data lifecycle (prune / export / consolidate).

## Screen routes

`dashboard` (home) · `experiment` · `calibration` · `acquire` · `select` ·
`annotate` · `visualize`

Buttons are wired by convention: `objectName = "nav_<route>"`. See
`daphnia/ui/main_window.py`.

## Surfaces, requirements and the memory bank

The app is being organised into **intent-based surfaces over one shared backend**
(one repo, one `daphnia` package, one SQLite store, several launch points):

| Surface | Launch point | Requirement doc |
| ------- | ------------ | --------------- |
| CORE | (library) | `docs/requirements/SR-CORE.md` |
| OPTICS | `python run.py optics` | `docs/requirements/SR-OPTICS.md` |
| BENCH | `python run.py bench` | `docs/requirements/SR-BENCH.md` |
| DATASET | `python run.py dataset` | `docs/requirements/SR-DATASET.md` |
| PREDICT | `python run.py predict` | `docs/requirements/SR-PREDICT.md` |
| STUDY | `python run.py study` (today's app) | `docs/requirements/SR-STUDY.md` |

Requirements are numbered (`SR-<SURFACE>-<NNN>`), `shall`-phrased, verifiable and
traced to the thesis objectives `O1..O4` / deliverables `D1..D3`. Read
**[`requirements/00-framework.md`](requirements/00-framework.md)** for the format
and the traceability matrix.

The agent's persistent context lives in
**[`../memory-bank/`](../memory-bank/)**: `projectbrief.md`, `productContext.md`,
`activeContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md` — and
**`mlContext.md`**, which governs every model, loss, metric, training run and
performance claim (`.clinerules/ml-engineering.md` invokes it). A task about a model
is not started without `mlContext.md`.

