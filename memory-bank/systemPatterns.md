# System Patterns — architecture and conventions

## Layers (strict)

| Layer | Modules | Rule |
| --- | --- | --- |
| **UI** | `daphnia.ui`, `assets/ui/*.ui`, `assets/theme/*.qss` | Qt only. **No pipeline logic in pages.** |
| **Domain** | `daphnia.vision`, `daphnia.pipeline`, `daphnia.protocols`, `daphnia.templates`, `daphnia.study`, `daphnia.annotation`, `daphnia.labeling`, `daphnia.units`, `daphnia.plates` | **Pure Python, no Qt imports** — unit-testable and usable from scripts/notebooks. |
| **Infrastructure** | `daphnia.db`, `daphnia.streaming`, `daphnia.paths`, `daphnia.settings`, `daphnia.maintenance` | Persistence, transport, config, data lifecycle. |

**Heavy third-party imports are lazy, always.** OpenCV *and* PyTorch are imported
inside the functions that need them, never at module import: the Qt app must start
without the CV or ML stack (`SR-CORE-011`), and `make test` must run the tests that
do not need them. `daphnia/vision/rtmdet/` is the largest example — ~1 900 lines of
PyTorch that this rule keeps out of the app's start-up path.

**Where Qt is allowed** (enforced by `tests/test_surfaces.py::PurityTests`): only
`daphnia/ui/**` and `daphnia/streaming/**`. `streaming` is the documented
exception - `reader.py`/`service.py` wrap OpenCV in a `QThread`. Everything else,
including `daphnia/surfaces.py`, must import headless so it stays usable from
scripts and notebooks.

## Pipeline — 7 steps (`daphnia/pipeline/steps.py`)

```
Ingest -> Localize -> Calibrate -> Predict -> Derive -> Persist -> Aggregate
streaming  vision.detection  vision.calibration  vision.keypoints
           pipeline.derive  pipeline.persist+db  pipeline.aggregate
```

`vision.detection.localize_well()` is **implemented**: it returns one
`Detection` per animal from the trained RTMDet
(`daphnia/vision/rtmdet/`), and raises `NotImplementedError` naming `make train`
when there is no checkpoint. The **Predict surface** is this step made visible.

`vision.keypoints.predict_keypoints()` still raises `NotImplementedError` — the
*keypoint* model (RTMPose) is not trained; there is no four-keypoint corpus yet,
while the detector's boxes now exist. `Keypoints.length_px` (Euclidean) is the only
live part of that module today.

**The model side is a subpackage, not a step:** `daphnia/vision/rtmdet/` holds the
architecture (`model.py`), the losses (`losses.py`), the data pipeline (`data.py`),
the trainer (`train.py`) and inference (`detect.py`). PyTorch is imported *inside*
its functions, so the Qt app starts and the test suite runs without it
(`SR-PREDICT-N02`).

> **The ML discipline lives in `mlContext.md`.** Any work on a model, a loss, a metric,
> a training run, a checkpoint or a performance claim is governed by
> `memory-bank/mlContext.md` (invoked by `.clinerules/ml-engineering.md`): the expert
> stance, the ten laws, the code shape, the reference implementations to read, and the
> catalogue of traps this project already paid for. `systemPatterns.md` keeps the
> *architecture*; `mlContext.md` keeps the *craft*.

## Workflow gates (`daphnia/workflow.py`) — the thing to generalise

Linear run, entered only when every earlier step's *fact* holds **for the active
project**:

```
1 Overview -> 2 Setup -> 3 Calibrate -> 4 Acquire -> 5 Select -> 6 Annotate -> 7 Analytics
```

| Fact | Produced by | Gate rationale |
| --- | --- | --- |
| `study` | Setup | it decides **where images are saved** |
| `connection` | Setup (runtime) | nothing captured/calibrated off a stream that isn't there |
| `calibration` | Calibrate | *"an image without a scale can never be measured"* — **global, not per study** (hardware) |
| `observation` | Acquire | Select needs bursts to compare |
| `selection` | Select | exactly one best frame per well/individual |
| `measurement` | Annotate | Analytics needs something to plot |

`facts()` counts run-progress **per study** (never globally); `calibration` is
**global**. `main_window.go_to_step()` refuses entry when `can_enter()` is false
and shows `blocking_reason()`. Select deliberately does **not** require the
connection — once frames are on disk, work continues offline.

> **Pattern to introduce:** `workflow.py` becomes a *policy* consulted by the
> shell — `StudyPolicy` (above, unchanged), `BenchPolicy` (frame + scale),
> `DatasetPolicy` (frames; labels optional). Same module, three behaviours.

## Mode axes (`daphnia/study.py`)

`StudySpec.from_row(row)` is the read model every screen asks, so the Setup
config drives the rest of the app:

| Axis | Values | UI effect |
| --- | --- | --- |
| `target_mode` | `plates` / `individuals` | plate + well pickers, or only a specimen counter |
| `organism_kind` | `animal` / `plant` | anatomy annotated: head->spine base *body length*, base->tip *shoot length* |
| `timecourse` | `longitudinal` / `single` | repeated timepoints vs measured once |

Exposes `uses_plates`, `is_longitudinal`, `keypoints`, `endpoint`, `specimen`,
`describe()`, `protocol`, `points_for(trait)`, `endpoint_for(trait)`.
`ANATOMY` is currently a dict in code (animal/plant only) — extending an organism
should be a **data** entry, not a code change.

`TRAITS = ("length", "width")`; the landmarks come from `TRAIT_LANDMARKS`
(`length`: `head` -> `spine base`; `width`: `left side` -> `right side`), with
`LANDMARKS` (click order) and `LANDMARK_TRAIT` derived from them. `daphnia/labeling.py`
adds the two **labelling modes** (`PAIR_MODE` / `PASS_MODE`), a `Draft` that holds
a landmark pass's partial points until they pair into intervals, and
`next_landmark()` (the `head -> spine base -> left side -> right side` order) - all
Qt-free, so the Bench and Dataset pages share it. Both pages auto-advance the pass
once a landmark covers every specimen (`_advance_landmark_if_complete`); the first
pass sets the specimen count and is never cut short, and `next_trait()` cycles
`length -> width`.

**The label panels use each control for what it is** (Bench + Dataset):
`combo_mode` and `combo_specimen` are **dropdowns** (real choices; the specimen list
grows as specimens are added), the current **keypoint/trait is *displayed*** in
`keypoint_label`, and the actions are **menu items** now (`X-N13`, in the page's own
menu - *Measure* on Bench, *Label* on Dataset): *Save* (hidden in landmark mode, which
stores automatically), *Undo*, *Clear points*, *Clear frame*, and *Next keypoint* (the
first switch is a judgement call the auto-advance cannot make). The old panel buttons
(`btn_save`, `btn_undo`, `btn_clear_points`, `btn_clear_frame`, `btn_next_keypoint`) and
the `*_tool` keys are the same actions under `workspace.set_group(...)`.
`_set_specimen`/`set_landmark`/`set_trait` are the programmatic setters; the trait and
landmark live in `_trait`/`_landmark`, not in a widget. `Clear frame` deletes the
frame's stored rows as well as the draft.

## UI conventions

- **Screens are separate `QWidget`s** loaded from `assets/ui/*.ui` at runtime via
  `daphnia.ui.loader.load_ui()` (a PySide6 `uic.loadUi` replacement). Every child
  with an `objectName` becomes an **attribute of the page** - which is why a widget
  name is an API, and why the `.ui` files are editable in **Qt Designer** but bound by
  three rules, enforced by `tests/test_ui_files.py` and reported by `make ui-list`:
  **one name per widget** (a duplicate means the attribute is a coin toss: two
  `btn_new_specimen` in `annotate.ui` left one button unwired), **no promoted
  widgets** (QUiLoader cannot create a Python class - it renders a plain `QWidget`,
  a silent hole), and **no `.qrc`** (the app starts without a compiled resource file).
  `make ui-designer FILE=bench.ui` opens the env's own editor (`Designer6`, the Qt
  this app runs on) after checking the file; the run-time-only parts (visual area,
  tool clusters, the shell's re-parenting) are not in the file and are not in the
  editor either.
- **`MainWindow`** owns `QStackedWidget#stackedWidget` + `StepBar`; `PAGE_CLASSES`
  maps `route -> screen class`; order/labels come from `workflow.STEPS`.
- **Button wiring by convention:** give a `QPushButton` the `objectName`
  `nav_<route>` and it navigates automatically.
- **`ShellPage`** gives every screen a left **visual area with zero margins**
  (live stream flush to the edges, large minimum size so it never collapses) and
  a right **controls panel** whose content scrolls *inside* the panel
  (`QScrollArea#controls_scroll`). Its width is **measured, not chosen**:
  `CONTROLS_WIDTH = 420` fits every shared screen (the densest asks for 345 px of
  content; the panel scrolls vertically only, so a wider child would be clipped and
  unreachable), and a screen whose content cannot shrink further carries its own
  `PANEL_WIDTH` (`Dataset.PANEL_WIDTH = 500`, sized from its dataset combo + two
  buttons). `tests/test_ui_smoke.py::test_every_panel_is_wide_enough_for_what_is_in_it`
  measures this per surface and `scripts/check_ui.py` per window, in both schemes.
  Calibrate used to carry a 400 px override; the shared width made it redundant.
  Visual widgets are code-built and added with `add_visual()`: `StreamView`,
  `ImageCanvas`, `QTableWidget`, `ThumbnailGallery`.
- **The panel holds choices; the image holds actions.** Anything the operator presses
  *while looking at the specimen* - zoom, fit, pan, full screen, save, undo, clear,
  box, the queue steps - is an **on-canvas tool** (`X-N11`), added in one line per page
  with `self.canvas_workspace(self.canvas)`. The panel keeps the dropdowns, the
  displays, the forms and the numbers. `X-N12` adds full screen on top of the same
  workspace.
- **`workflow_changed` signal** on `ShellPage`: emitted via `notify_workflow()`
  when a screen produced a fact the gates depend on.
- **Hooks:** the shell calls `on_shown()` / `on_hidden()` on the incoming/outgoing
  page.
- **The status footer is in *every* window** (`StatusFooter`): camera state + Connect,
  and the scale in force (µm/px, uncertainty, age, RE-VERIFY). The hub owned it first;
  now `hub.SurfaceWindow` (the base class of Calibration/Bench/Dataset/Predict/plan
  windows) mounts it once, under the screen, and `MainWindow` mounts it from
  `pages.ui`. The attribute is named **`status_footer`** on every window because that
  is the name `page_chrome`/`WINDOW_CHROME` hides for full screen. `Calibrate...`
  (offered by every window except Optics, which *is* the calibration) triggers
  `SurfaceWindow.open_calibration`: the window hides, a `CalibrationWindow` opens
  (`WA_DeleteOnClose`), and closing it re-reads the footer and shows the window again -
  one window on screen, the same rule the hub follows. Two screens' text ("press
  Calibrate/Connect in the bar below") is only true because of this; it was a lie
  before.
- **The header is in *every* window too** (`HeaderBar`, `X-N13`): the menus (`File`,
  `Source`, `Models`, `View`, `Go`, `Help` plus the page's own, in one fixed order) and the
  facts that are not a measurement, mounted as the window's menu-bar area
  by `SurfaceWindow` (and by `MainWindow`/`HubWindow`). See *The window header* below for
  the rule that decides which facts go there rather than onto the image.
- **Styling — one system, two schemes** (`daphnia/ui/theme.py` +
  `assets/theme/daphniascope.qss`; the GTRONICK `MacOS.qss` is deleted). Three parts:
  **tokens** (`LIGHT`/`DARK`, same key set - tested) are the only place a colour is
  written down; the **QPalette** built from them colours everything the stylesheet
  does not mention (group boxes, tables, lists, text browsers, `QMessageBox`, menus,
  tool tips); the **template** carries structure only (padding, radii, hairlines,
  `objectName` blocks) plus the decisions - primary action, step states, and the
  `PHOTO_OVERLAY` tokens for the translucent tools *over* a photograph, which are
  identical in both schemes on purpose. Style is set to **Fusion** (drawn check
  marks, combo arrows, spin buttons - the old theme's were `url()`s to Adwaita files
  that do not exist on macOS), and no `::indicator` is ever styled. The scheme
  **follows the desktop** (`QStyleHints.colorSchemeChanged` -> re-apply) and
  `DAPHNIA_THEME=light|dark` pins it; an unknown value raises rather than quietly
  giving the other theme. Painter code asks `theme.qcolor("canvas")` instead of
  holding a hex value, and no screen sets a colour inline (`setStyleSheet` outside
  `theme.py` fails `tests/test_theme.py`). QSS is still a CSS2 *subset* - **no
  layout properties, no classes, no variables**; restyle by changing a token, not a
  rule. The app-wide setStyleSheet also means `app.style()` becomes a stylesheet
  proxy, which is why the "which base style is in force" test drops the sheet first.
- **Qt gotcha:** `&` is a mnemonic marker in widget text; a literal `&` in a
  `QPushButton` label must be `&&` (`&amp;&amp;` in `.ui`). `QLabel` renders a
  single `&` as-is.
- **One image, and one way in** (`ImageCanvas.set_source`, `frame_from_source`).
  A canvas accepts a numpy frame, a **file path**, a `QImage`/`QPixmap` or `None`;
  a path is read by `daphnia.vision.images.load_image` (the only loader: `.HEIC` via
  the system converter, a *named* refusal otherwise) so no screen repeats the file
  reading and none has a second, weaker loader (`set_frame`/`set_image_path` and
  `annotate.py`'s silent `QPixmap(path)` are gone). `set_source` returns the frame it
  displayed, so a page that needs the pixels (crop, detection, grid) does not load the
  file twice; `canvas.frame`/`canvas.refusal` are the same state read back. Every call
  replaces the previous image **and** its points, box, overlays and grid - and a
  *refused* source (drawn on the canvas + `source_failed` signal, which each screen
  wires to its status line) also leaves the page's own frame state empty, so no number
  is left standing over a picture that is not there. A wrong *kind* of source raises
  `TypeError` (a programming error, not something to draw).
  **Trap paid for here:** `np.ascontiguousarray` on an already-contiguous slice is
  *not* a copy, so `qimage_to_frame` once returned a view of Qt's buffer - a
  use-after-free whose symptom was a first row shifted by one byte, only sometimes.
  It now `.copy()`s, and `tests/test_image_canvas.py` asserts `frame.flags["OWNDATA"]`.
- Canvas interaction (`ImageCanvas`, shared by Calibrate, Annotate, Bench, Dataset):
  pinch or **vertical** scroll zooms toward the cursor **proportionally to the delta**
  (`ZOOM_STEP ** (dy / SCROLL_REFERENCE)`, so a trackpad's many small deltas move
  gently); a **horizontal** scroll (`angleDelta().x()` dominating `y`) **pans** -
  side to side without a modifier; **Command/Control + left-drag** - or `pan_mode`
  (the **Pan** hand tool) - pans **without placing a landmark**; right/middle-drag and
  arrows also pan; `+`/`-`/`F` zoom in/out/fit. Clicks stay mapped to original image
  pixels at any magnification. Frames are **rotated to landscape once on arrival** with
  the aspect ratio preserved, so preview, on-disk burst and clicked keypoints share one
  coordinate space.
- **Multi-specimen visibility:** `ImageCanvas.set_annotations([(index, segments)])`
  draws each specimen in `specimen_color(index)` **with its number**
  (`show_annotation_index`) at the first point. In **landmark mode the editable set is
  empty** and everything is drawn through the overlays, so a specimen gets its colour
  from the first click instead of sharing `MARKER_COLOR` until the pass is switched.
  `show_point_labels` / `connect_points` control the active set's text and its
  connecting line (off for a multi-point landmark pass).

## Data model invariants (`daphnia/db/schema.sql`)

- **`corpus_labels` — the ONE store of a measurement** (one per
  `dataset, file, specimen, trait`, enforced by UNIQUE). A measurement is a
  `corpus_labels` row **whichever surface wrote it**: the Bench, the Dataset
  labeller, or a governed study capture. It is **scale-free** by design -
  `p1_*`/`p2_*` in source pixels, `image_width`/`image_height` (nullable: a frame
  not on disk has no size, and NULL says so), and `profile_id`/`px_per_um`/
  `length_um`/`uncertainty_pct` filled **only** when a calibration was in force -
  so a corpus can be labelled with no study/plate/connection/calibration
  (`SR-DATASET-N01`). It also carries the governed provenance
  (`observation_id`/`asset_id`, NULL for a Bench/Dataset label), the
  `individual_index` the number is about (`vision.individuals`), and a box's
  geometry (`width_px/height_px/area_px` + `_um`, `trait = 'bbox'`,
  `SR-BENCH-015`). `repository.save_corpus_label` is the **one write path**.
  Re-saving the same key *updates*.
- **`measurements` is a VIEW, not a table** (`connection.MEASUREMENTS_VIEW`):
  `SELECT label_id AS measurement_id, …, corpus_labels.rowid AS rowid FROM
  corpus_labels WHERE observation_id IS NOT NULL`. It exists so every reader
  written against the old table - `queries._PROGRESS`,
  `list_studies_with_progress`, `maintenance.prune_discarded`, the annotate and
  analytics screens - keeps working **unchanged** while the write path lives in
  the corpus. `save_measurement` (the study path) is a thin adapter: it resolves
  the frame (path + size) and the dataset, records the observation, and delegates
  to `save_corpus_label`. **The landmarks are the training ground truth.**
- **`corpus_boxes` — one box per `(dataset, file, specimen)`** — one box is one
  **individual**. It carries the scale it was drawn under (`profile_id`/
  `px_per_um`) so its size in µm is derivable exactly as a label's `length_um` is,
  and the governed provenance (`observation_id`/`asset_id`).
- **`datasets.study_id`** links a corpus to a governed study
  (`dataset_for_study` creates it with `make_current=False`); `is_current` is the
  **one open dataset** every entry point shares (`ensure_default_dataset` creates
  the default on a fresh database). "Another model = another dataset" still holds:
  the `schema_key` is what a label *means*, and a rename locks it.
- **`import_batches` / `imported_frames`** record where a queue came from:
  `kind` (`folder`/`video`/`single`/`live`/`boxes`), `source`, per-frame
  `frame_index` (a video's frame number, read back by
  `repository.frame_index_of`), `ordinal` (queue order) and the resume pointer
  `last_ordinal`. `UNIQUE (batch_id, file_path)` makes a repeated import the same
  work. A video's frames are **materialised** to `datasets/raw/<stem>/f000123.png`
  by `daphnia/vision/video.py`.
- **`media_assets.selection IN ('pending','kept','discarded')`**;
  `select_best_frame()` keeps the chosen one and sends **the siblings to
  `discarded`**. `prune_discarded()` deletes `discarded`+un-measured frames from
  disk — **destructive by design**; it now checks the `measurements` view, so a
  frame carrying a corpus label is still never pruned.
- **`is_current`** on `studies`: exactly one current project;
  `ensure_current_study()` adopts one for legacy databases.
- **`calibration_profiles.is_active`**: exactly one active row (single global
  scale) — `activate_profile()` clears the others.
- Nullable-by-design: `imaging_sessions.study_id`, `observations.plate_id`,
  `observations.well_label` / `individual_index`. **This is what makes
  study-less Bench/Dataset rows possible without a schema change.**
- **Two questions, two columns** (`SR-BENCH-016`). `measurements.specimen_index` answers
  *which row on the frame is this* (the row key is `(asset_id, specimen_index, trait)`,
  which is what makes re-measuring the same thing update instead of duplicate), and
  `measurements.individual_index` answers *whose number is it* (`NULL` = measured outside
  every individual, `daphnia/vision/individuals.py`). They are kept apart on purpose: if a
  measurement's `specimen_index` were its individual's index, two `length` lines on one
  animal would collide on the row key and the second would **silently overwrite** the
  first. A Bench **box** is the individual, so its own `bbox` row carries its own index in
  both columns; a line inside it carries the box's index in `individual_index` only.
- `init_db()` is **idempotent**; `migrate()` patches columns added later via
  `COLUMN_MIGRATIONS` (`CREATE TABLE IF NOT EXISTS` never alters), backfills
  `p1/p2`, de-dupes measurements, then (re)creates indexes.

## Calibration model (calibration = an immutable record, setup = hardware)

The two concepts were fused in one mutable row; they are now separate in behaviour
even though they share a table:

| Concept | Where | Mutable? |
| --- | --- | --- |
| **Optical setup** — the hardware chain | `calibration_profiles.setup_key` (from `ids.new_setup_key`, i.e. the preset **name**) | yes, it is just a label |
| **Calibration** — the scale measured on that chain at a moment | one `calibration_profiles` row (`ids.new_calibration_id`) | **no** — ever |
| **Uncertainty** — how good that reading was | `calibration_profiles.uncertainty_pct` | no |
| **What a measurement used** | `measurements.profile_id` + `px_per_um` + `uncertainty_pct` | no |

- `repository.register_calibration()` **always inserts**; `upsert_profile()` is the
  legacy overwrite path and is kept only for older callers/tests.
- **Uncertainty is derived, not declared:** two clicked landmarks carry about
  `CLICK_ERROR_PX` each, so `relative_uncertainty_pct(distance_px)` is the scale's
  relative error. A single 0.1 mm pitch at 2x is 14.7 px -> **+/- 13.6 %**; ten
  pitches is 147 px -> **+/- 1.36 %**. This is why "click a longer span" is advice
  with a number behind it.
- **Staleness is a first-class signal:** `is_stale(calibrated_at,
  settings.CALIBRATION_MAX_AGE_HOURS)`, because a bumped adapter invalidates the
  scale without changing a single stored value. SQLite timestamps are **naive
  UTC**, and `calibration_age_hours` documents that.
- **Recalibrating mid-study** creates a new record; sessions keep their
  `profile_id`, and earlier measurements keep the scale *and* the `+/-` they were
  taken under (`SR-OPTICS-011`, `SR-OPTICS-014`).

## Auto-calibration from grid paper (`daphnia/vision/lattice.py`)

The OPTICS surface measures with **`vision/lattice.py`** (`SR-OPTICS-035/036`). It is the
estimator for photos *as this rig produces them* - grid paper seen through an eyepiece -
and it exists because `vision/grid.py` (below) refuses every one of them. Four steps, each
one a measured requirement rather than hygiene:

1. **the field stop is masked as a fitted circle.** Otsu separates paper from surround
   (measured on the project's photo: threshold 100.4, paper 143-210, surround <= 60), the
   largest bright region is filled, and a circle is fitted to its boundary **ignoring the
   frame border** - the disc is cut off by the frame at the top, and a fit that includes the
   flat clipped edge answers radius 390/381 px where the truth is **448.7 px**, i.e. 60 px
   small, so real paper along the top and the dimmer right is filled over (an operator sees
   that as "it fails at the top and right edges"). The disc is eroded by a margin so the
   stop's ring - a hard edge every profile turns into a line - stays out, and the surround is
   filled with the paper's median.
2. **the lattice is straightened**, by a projection search: rotate the frame by each
   candidate angle and keep the one whose row profile is the peakiest. The search *is* its
   own check, which is why a flipped or rotated photo needs no special case - and why the
   sign that `vision/grid.py`-style fixes get wrong (a wrong sign *doubles* the tilt,
   measured -8.95 -> -17.70 deg) cannot happen here.
3. **the period is read off narrow bands** (24 px wide, every 60 px), from the robust median
   spacing of the dips in each band: spacings are filtered to those within 25 % of the
   running median, and a band whose spacings scatter by more than `MAX_SPACING_SCATTER` is
   not a lattice. A band must also *explain* its own profile (`MIN_PERIODICITY`: correlation
   at the measured lag; the lattice measures 0.67-0.83, noise 0.1-0.2) - which is the test
   that refuses a flat field, and the one that made a *ruler's* graduations (regular spacing,
   no second family) fail the anisotropy test instead.
4. **the spread across those bands is the distortion**, reported separately from the scale's
   uncertainty (`SR-OPTICS-036`). Measured on the rig's own photo: **27.239 px** per cell
   from 10 bands, tilt -9.00 deg, spread **5.10 %**, statistical 2.49 %; three more photos
   of the same setup give **27.03 / 27.24 px** and 6.33 / 5.10 %.

Two measured traps are recorded in the module because they cost real time: an
**autocorrelation** period answers **13.4 px** where the dips are **26-27 px** apart (each
printed line's own micro-structure repeats at half the pitch - an autocorrelation cannot
tell "a repetition" from "the shape of one line"), and a printed line's profile **splits
into two dips 4-12 px apart** in some bands (a 2-px blur merges the shoulders; a
threshold relative to the band's *deepest* line keeps the faint ones).

## Auto-calibration from a clean grid photo (`daphnia/vision/grid.py`)

*Kept, and no longer the OPTICS path* (superseded by the above, 2026-10-19; its refusals
and its tests still stand). It measures a **clean** grid photo: a 2D FFT finds the lattice,
a projection plus a zero-padded DFT refines the period, and `GridNotFound` means nothing is
stored.

The reference target is **only** printed grid paper, and it is *detected*, not
clicked (`SR-OPTICS-016`, `SR-OPTICS-019`). How big one printed square is (1 mm by
default) is stated once in panel 2 and travels with the calculation
(`detect_grid(frame, pitch_um=...)`, `SR-OPTICS-030`) - the detector always measures a
*period in pixels*, and the square's true size is what turns that into a scale:

1. a 2D FFT finds the grid's two lattice directions - rotation-invariant, so the
   paper need not be aligned;
2. the projection onto each direction, then a zero-padded DFT peak search with
   parabolic interpolation (on a Hann-windowed profile, which suppresses leakage),
   refines the period to a fraction of a pixel.

Two things that matter more than the maths:

- **It refuses.** `MIN_PROMINENCE`, `MIN_PERIODS`, `MAX_ANISOTROPY_PCT` and
  `MAX_FAMILY_SKEW_DEG` gate it, and `GridNotFound` means *nothing is stored*.
  Validated failures: a flat field, a **specimen-plate-like image**, blobs, one
  stripe family, and a non-square target.
- **The paper is the accuracy floor.** `PAPER_PITCH_TOLERANCE_PCT` (0.5 %) is
  combined in quadrature with the fit error, because printed 1 mm spacing is not
  metrology-grade. A grid verified once against a stage micrometer earns a smaller
  tolerance (`uncertainty_pct(paper_tolerance_pct=...)`).
- `distortion_pct` measures the centre-vs-edge scale drift, so lens distortion is
  *measured* rather than assumed, and the recommendation to keep specimens central
  is grounded.
- **The reference photo may be anything.** `daphnia/vision/images.py` reads any file:
  bytes + `cv2.imdecode` (so filenames are never a platform question), then macOS
  `sips` when OpenCV has no codec - which is exactly the case for the `.HEIC` an
  iPhone writes by default - and otherwise a **named refusal** rather than a silent
  `None`. A 12 MP photo is block-averaged to `MAX_DETECTION_SIDE_PX` before the FFT
  and the period is scaled back, so the reported scale is the scale of the photo
  supplied (`SR-OPTICS-022`, `SR-OPTICS-023`, `SR-OPTICS-003`).

### It refuses the photos this rig actually produces (measured 2026-10-19)

The detector above was run on **all nine** photos in `recordings/calibration/`
(seven 1280x720, two 1920x1080) and **refused every one**: *"the grid is too faint or
too fine to measure"*. Two independent gates are responsible and neither is the
paper's period:

* **prominence** (`MIN_PROMINENCE = 100`): a pale mint-green print on white, inside a
  hard black field stop whose edge is broadband energy, cannot put its lattice peak
  100x above that floor. Filling the surround with the paper's median does **not**
  rescue it.
* **refinement range** (`MIN_PERIOD_PX`): the peak search can land on a harmonic and
  `_refine_period` then returns `None`, which reports as "too faint or too fine". On a
  hard-edged *synthetic* grid the fundamental and the second harmonic have equal
  amplitude, and the detector answered **9.45 px** for a synthetic grid whose true
  period is **19 px** - a factor-2 error that a real photo with a dominant fundamental
  never exposes.

So: `distortion_pct` is a *computed* number on paper only - on this rig it is never
reached, and its `0.0` default means **"not computed"**, never "none". A surface must
not read it as "no distortion" (`SR-OPTICS-032`).

### What the scale does not cover: distortion (`scripts/calibration_probe.py`)

The scale is one number, so it is true where it was measured; **discorpy** (Apache-2.0,
optional pip in `z-app`, `docs/calibration.md`) measures the rest from *one* photo -
the line-pattern flow, the centre of distortion, and a polynomial radial model whose
sub-pixel residual is its own acceptance test:

* on a **synthetic single-weight grid** with a known distortion the flow recovers the
  centre to **1.00 px** and unwraps to **0.264 px** rms (**0.056 px** with Gaussian peak
  fitting) - sub-pixel, so the method is proven before any real photo is judged;
* on the real graph-paper photo, **after the three preparations that photo needs**, it
  assembles **24 + 30 lines** (the real lattice) and unwraps to **3.18 px rms, worst
  line 16.6 px** - still short of the ~1 px bar, and the worst line being 5x the rms
  says *a few mis-grouped lines* are the limit, not a smooth optical term.
* the paper itself: one uniform lattice at **27.0 px** per cell with a heavier line
  every fifth cell (135 px), two amplitude levels, widths 1-7 px. discorpy's spacing
  *estimator* is unstable on it (16.9-28.2 px depending on preprocessing and radius) -
  a direct narrow-strip profile is the measurement to trust, and it puts the **cell
  spacing at 26.0-28.0 px across the field (±3.8 %, ~5.7 % centre-to-edge)**.
* **the model now fits and is repeatable**: the documented flow (docs 3.3.4, including
  `regenerate_grid_points_parabola`) with **3 coefficients** gives **0.511 px rms,
  worst line 1.223 px → sub-pixel ✓**, COD **(616.3, 316.7)** (straightened) and a local
  scale of **+9.75 % at r = 302 px**; `--edge-margin 30` also passes (0.667 px) with COD
  (613.1, 318.2) and **+8.09 %**. The model's *shape* is repeatable; its *magnitude* is
  ~1.5-2x larger than the spacing measurement says, so it is not yet storable.
* **`--radius` must not be reduced for tidiness** (measured: 4 → 6.25 px, 6 → 1.61 px,
  8 → 7.31 px, 9 → **0.51 px**), `--coef` decides whether it passes at all (3 ✓, 4 → 2.22,
  5 → 1.08), `--perspective` moves the curvature ~2x, and `--dist-px` (pinning the pitch)
  is wrong for the documented grouping (COD +78 px, residual 2.57 px).
* **the mirror check is the acceptance test** (`--mirror-check`, on by default): a mirrored
  image is an exact pixel permutation, so a real model must come back mirrored -
  coefficients identical, COD at ``width - 1 - x_c``. Measured on this photo: coefficients
  differ by **0.82**, the centre lands **184 px** off the expectation, and the mirrored
  residual is **8.68 px** → *the model is not storable yet*, and no second photo was
  needed to learn that. The **preparation** does survive the same transformations: on a
  mirrored / upside-down / 90°-rotated / re-positioned copy the circle is found identically
  (r **448.7 px**; **418.5** when the disc sits clear of every edge) and the alignment
  self-check picks the right rotation sign every time (mirrored → **+8.95°** where the
  original needs −8.95°; a wrong sign *doubles* the tilt and is therefore detected).

**The three preparations, each a defect found by putting the points on the photo**
(`_points.png`) - which is how an operator checks this, and how the mask was caught:
the field stop must be masked as a **fitted circle** (measured radius **448.7 px** at
(600.7, 265.6); the earlier brightness-percentile and lattice-energy masks answered
**381-390 px**, i.e. 60 px small, so real paper along the top - where the disc is **cut
off by the frame** - and along the dimmer right side was filled over: *"it fails at the
top and right edges"*); the lattice must be **straightened** (discorpy samples tilted
profiles and truncates them to stay inside the image: ~114 px along one side with **no
points at all** at a 9 deg tilt; measured 0.00 deg after); and points must be **kept
`radius + 2` px inside the paper** because the paper/surround junction is a *step* whose
local extremum collects points along that edge (the last pile-up the operator could see,
now gone).

Rules a future `daphnia/vision/distortion.py` inherits from that run: a drift that was
not measured is reported as unmeasured (**not 0 %**); a model is stored only if it
passes the sub-pixel check; a model is hardware (dated, immutable, per `setup_key`);
and a corrected frame's scale is **re-measured from the corrected grid**, since `c0` is
a centre magnification, not a scale. The probe itself is Qt-free and lazily imports
discorpy, so `make test` and the app are untouched by its existence.

## The bench: a measurement without a study

`daphnia/ui/pages/bench.py` is the fast path, and it exists to prove the surfaces
are really separate. What it needs from CORE:

- `repository.bench_observation(conn, file_path=..., profile_id=...)` — creates a
  **study-less session per image** (`study_id = NULL`). One session per image on
  purpose: a shared one would drag every other bench image into a study on promotion.
  The image is keyed by its **file path**, so re-opening a photo continues it and a
  re-measurement updates instead of duplicating (`SR-BENCH-011`).
- `repository.attach_observation_to_study(...)` — the whole reason study-less rows
  exist: work done before a project can be adopted into one, keeping its scale.
- `queries.bench_measurements` / `bench_count` / `bench_observations` — the flat
  "unsorted" list, selected on `s.study_id IS NULL`.
- `maintenance.export_bench_csv` — like the study export, but **keeps `p1`/`p2`**,
  because an ad-hoc number without its points cannot be defended or reused.

Two rules the surface keeps:

- **The scale is non-negotiable, the study is not.** With no calibration the readout
  is `UNSCALED` and *nothing is stored*: `measurements.length_um` is `NOT NULL`, and a
  pixel count in a micrometre column would be fabricated data.
- **A live frame is written to disk only when a measurement is saved**, so a quick
  look leaves nothing behind.

## The calibrate surface: one photo, one press, one result, one stored record

Three panels, in the order of the path (`SR-OPTICS-024`; S1 of
`docs/ui-calibrate-redesign.md`, shipped 2026-10-20):

| Panel | Carries | Why it is there |
| --- | --- | --- |
| **1 Optical chain** | a **read-only list of facts** (`Preset`, `Setup key`, `In force`, `Calibrations`, `Mount`, `Distortion`) and two `[Load...] [New]` pairs, one for the **preset** and one for the **calibration** | a device is *loaded or started*, and so is a scale - the same gesture for both, because a preset has many calibrations; the facts are values in a labelled list, not prose, and the name is *entered in a popup and displayed here read-only*, so there is exactly one input per fact |
| **2 Target photo** | `[Use live frame] [Open photo...]`, the live feed's state, the **printed square size** (mm), `[Measure grid]`, then the **result block**: `period · bands · tilt`, the **drift**, the field stop, the **verdict** - and one line that always says lens distortion is **unmeasured** | the period in pixels only becomes a scale once the square's true size is known, the numbers have to be readable as *numbers*, and the judgement has to be one line the operator can act on |
| **3 Scale in force** | S, the uncertainty **with the route it came from**, a concrete example length, `[Store this scale]`, and the hand-entry row (`I know the scale already`) | the result, measured **or typed in**, stored back onto panel 1 |

- **The one deliberate exception to `X-N13`.** On every other screen an action is a menu
  item; **here *Measure grid* and *Store this scale* are a panel button *and* a menu item**,
  because `SR-OPTICS-024` makes this step's conclusion panel 3's own content and the daily act
  has to end on the panel that defines it. Both entries call **one** method
  (`Calibration.measure_grid` / `Calibration._save_profile`), so they cannot drift; the menu
  group is `Calibration`. This is the only screen where an action is deliberately in two
  places, and it is recorded here so the next reader does not "fix" it back.
- **The three cases of the in-force line, not three vocabularies.** An empty name box says
  *any* chain is in force (nothing was typed, so no chain was named), the matching name says
  *this name* with the age, and a different chain says *nothing for this name yet*.
  `RE-VERIFY` is appended where the quoted scale is the one this name would use, and the
  chain's key is that line's tooltip. Before this, the header said "No calibration yet",
  panel 1 said "a different preset" and the footer had a third answer, all on one screen.
- **One fact, one home.** The detector's sentence list (`LatticeEstimate.warnings()`) is
  *read for its fields*, not printed: period/bands/tilt → one line, drift → its own, the field
  stop → its own, the judgement → `label_verdict`, and the long caveats → tooltips. A
  refusal's *reason* (which line family, how many bands) is the verdict's tooltip, not a
  paragraph in the status line.
- **The drift is not the uncertainty, and not "distortion".** `spread_pct` is the place in the
  frame changing the scale; the uncertainty is how well the centre was measured
  (`SR-OPTICS-036`); the verdict calls the drift **weak** when it exceeds the print's own
  tolerance, and *lens distortion* appears only as *unmeasured* (`SR-OPTICS-032`). The
  estimator's own caveat wording was fixed for the same reason.
- **The route of a stored scale is named where it matters.** The grid route's uncertainty
  **includes** the print's ~0.5 %, the clicked-span route's does not (the panel says so), and
  a hand-entered scale is a claim. The "weak" hint therefore never sends the operator from the
  grid to the span - that advice traded an honest 2.54 % for a 0.29 % that left half a percent
  out.
- **The hardware description stays removed on purpose (2026-10-19).** Microscope, objective,
  camera and locked zoom/focus were four fields nobody filled in consistently, and the identity
  they were meant to pin is the preset name; the code still records `zoom 1.0x` /
  `focus at infinity` as the mount's constants (`SR-OPTICS-004`), and the old `Fixed` form row
  is now one footnote line under panel 1 (its value widget survives; the "Fixed" caption does
  not).
- **One number is the operator's, and the panel says which.** A typed scale
  (`px/um` + the accuracy stated for it) overrides the measurement, is flagged
  `hand-entered` in panel 3, gets **no proof image and no `calibration_records`
  row**, and the status line calls it a claim (`SR-OPTICS-029`).
- **A device and a scale are loaded or started the same way, and the panel is a list.**
  Panel 1 is a read-only `QFormLayout` (`facts_form`: `Preset · Setup key · In force ·
  Calibrations · Mount · Distortion`) with two `[Load...] [New]` pairs. The **preset** pair:
  `[Load preset...]` lists **names only** (`Calibration.stored_preset_names`: distinct
  `setup_key`s, one name each, newest first) and `load_preset(name)` makes that chain's
  **newest** calibration the active one (`repository.activate_profile`, `SR-OPTICS-006`);
  `[New preset...]` asks for a name in a popup and clears the measurement. The **calibration**
  pair mirrors it: `[Load calibration...]` lists **this device's own** calibrations
  (`Calibration.stored_calibrations` - `repository.calibrations_for_setup`, each
  `label|date · px/mm · (in force)`) and `load_calibration(profile_id)` both activates the
  picked row and puts its **proof photo** on the canvas; `[New calibration...]` asks for a
  **label** (a note like "after refocus", stored in `calibration_profiles.label` and written
  by `_save_profile`) and clears the measurement but keeps the preset. **The name is entered
  only in a popup and shown read-only** (`profile_name` is a frameless read-only `QLineEdit`,
  2026-10-20) - the earlier inline field duplicated the popup input. Both `[Load...]` paths
  measure and store nothing, and both are guarded by the 2x check (`_confirm_scale`,
  `SR-OPTICS-008`). Each chooser is a `QInputDialog` **inside the button handler**
  (`_load_preset` / `_new_calibration` / `_load_calibration`), never inside the `load_*` or
  `new_*` core, because a modal dialog in the method a test drives is how a suite hangs.
  **`SR-OPTICS-031` is superseded here:** the list used to be a *view* only
  (`combo_history`, `load_calibration_image`), and the operator asked for one gesture -
  load or start - so loading a calibration now *switches* the scale in force.
- **The interconnections are visible, not implied.** Renaming panel 1 recomputes the
  setup key and restates what is in force. Changing the square size re-scales the detection
  already on screen (the estimate is `dataclasses.replace`d, never re-detected). Panel 3
  names the route of the number and shows what the uncertainty means for a real length
  ("160.0 +/- 8 um"), because a percentage is not actionable (`SR-OPTICS-028`). The example
  line keeps `daphnia.units`' own `+/-` house style, next to the panel's `±`.
- **Detection is drawn - and so is the field stop.** `ImageCanvas.set_lattice(period_px,
  angle_deg)` overlays the measured grid in amber (`SR-OPTICS-026`), and
  `set_field_stop(centre_x, centre_y, radius)` the paper's circle beside it: the lattice says
  *what* was measured, the circle *where it was possible to measure it*. Both are cleared
  with the frame (and by `clear_points`), so neither can describe a photo it did not come
  from.
- **The hand method counts squares, not a pitch** (`SR-OPTICS-027`), which deletes the
  "wrong pitch for the wrong slide" error class entirely.
- **No dead controls.** Every action is live and explains what it needs (`X-N10`).
  `btn_detect` used to be disabled until a frame existed, and a disabled button
  swallows the click - which is precisely why it "did nothing".

## The shared status footer (`daphnia/ui/widgets/status_footer.py`)

One `StatusFooter` in **every** window - the hub and the study shell, and therefore
every future surface (`X-N09`). It carries two **machine facts**, because both change
while you work and both change what a number means:

| Fact | Shows | Action |
| --- | --- | --- |
| **Camera** | `StreamService.state_label()` | Connect / Disconnect |
| **Scale** | scale, `+/-`, age, `RE-VERIFY` | `calibrate_requested` |

- The connection has **four** states, not two: `DISCONNECTED`, `CONNECTING`,
  `WAITING` ("connected, but no frames arriving") and `LIVE`. `service.connected`
  only means a socket is attached; `service.has_signal` means frames are actually
  arriving. Conflating them is what let a window claim "no connected camera" while
  the phone was streaming (`SR-CORE-017`, `SR-CORE-018`).
- `_on_failure` stores the reason in `_last_error`, so the bar says *why* it is not
  connected instead of just that it is not (`SR-CORE-019`). `stop()` clears the
  frame buffer, and `StreamView` clears its pixmap on `closed`, so a dropped stream
  never leaves a stale picture on screen.
- **A failed read is retried, not a disconnection** (`SR-CORE-020`). Measured
  2026-10-19: both a decoder failure and a moment with nothing buffered return `False`
  from `capture.read()` while the frames after them are fine, so `StreamReader` counts
  failures, emits `degraded(count)` once per run, resets the count on the next good
  frame, and calls the stream lost only after `MAX_FAILED_READS = 5` in a row.
- **The publisher's codec is asked and named** (`SR-CORE-021`): the reader asks
  MediaMTX's control API (`mediamtx.stream_codec()`), and `StreamService.stream_note()`
  turns "H265" into `the publisher sends H265 - set the phone's video codec to H.264`
  plus any `N frame(s) could not be decoded`. Measured: H.265 loses 7-21 of the first
  120 frames at 24.1 fps against 27.5 fps for H.264 - so FFmpeg's own two-lines-per-frame
  log is quietened (`settings.FFMPEG_LOGLEVEL`) and the actionable sentence goes to the
  status bar (footer, panel hints, `StreamView` placeholder), which is where an operator
  reads it.
- Host actions stay in the host: the hub opens `CalibrationWindow`
  (`python run.py optics`); the study shell navigates *back* to the Calibrate step,
  which the guarded workflow always permits while the optics are in use.

## The menu bar (`daphnia/ui/widgets/header_bar.py`) — `X-N13`

**One order, the same in every window** (`MENU_ORDER`):

```
File | <the page's own menus> | Source | Models | View | Go | Help
```

An operator learns the bar once. The app's menus are created **empty, in that order**, in
`HeaderBar.__init__`; the window fills `View`/`Go`/`Models`, the page fills `File` and
`Source`, and `hide_empty_menus()` (called by the window after its setup, and by
`MainWindow` after every step switch) takes out a menu that stayed empty - a menu that
opens onto nothing teaches the operator that the bar lies.

| Menu | Owner | Content |
| --- | --- | --- |
| `FILE_MENU` | app + page | the page's file entries (*Open an image...*, *Import a folder...*, *Export the dataset*), then *Exit* (`QKeySequence.Quit`, `MenuRole.QuitRole`, closes every window) |
| *(the page's own)* | page | one menu per `ToolGroup` group the page named - *Measure* (Bench, Calibrate, Annotate), *Label* (Dataset), *Frames* (Predict) - inserted **directly after File** (`_insert_menu`) |
| `SOURCE_MENU` | app + page | *Connect the live feed* (checkable: the check mark **is** the state, driving the same `StreamService` as the footer's Connect), then the page's sources under a separator (*Use the live frame*, *Use the current frame*, *Open a video file...*) |
| `MODELS_MENU` | window | `ModelMenu` (`daphnia/models.py` registry) |

### Editing the frame: `ImageCanvas.box_edit` — the automatic *Edit* mode (`SR-DATASET-026/027`)

The canvas keeps three overlay layers, and they are deliberately different things:

| Layer | Drawn as | Owner |
| --- | --- | --- |
| **stored labels** (`set_boxes`, `set_annotations`) | dashed, per-specimen colour | the database - what a human stored |
| **the model's answer** (`set_predictions`) | dotted, one colour (`theme.qcolor("prediction")`) | the *Model* switch; never stored (law 7) |
| **the selection** (`box_edit`) | solid outline in `accent` + 8 handles | the operator, right now |

`box_edit` is switched on by the page at start-up (`Edit`, a checkable menu item **on by
default**): the mode decides what a click *means*, so it is always offered, and turning it off is
the creation mode. With it on a press is an **edit**, never a label, and the order of the
questions in `_begin_edit` is the order of what is on screen, front to back:

| Press lands on | What happens |
| --- | --- |
| a **handle** of the primary box | resize it (`resize_handle` - corners beat edges) |
| an editable **point** (`_points`) | drag that point alone; `Backspace` removes it |
| a **stored box** | click selects it, Shift/Cmd-click adds to the selection, dragging moves **the whole selection** |
| a **model box** | promoted to the draft *silently* and dragged; the drop emits `box_changed`, so the page **stores it as an annotation** - a *click* stores nothing |
| **empty space** | a **marquee**: every stored box it covers is selected |

An editable **point** has two signals and they mean different things: `points_changed(list)` is a
**placement** ("a click put a landmark here" - the pages draft and re-render on it), while
`points_moved()` is a **move** ("look again" - the canvas keeps the position). Emitting
`points_changed` for a drag is the point snapping back under the operator's hand: a pass-mode page
takes the move for a placement, walks to the next specimen and re-sets the canvas' points
(measured; `SR-DATASET-027`). On the **drop** the canvas emits `points_edited()`: each labelling
page writes the correction for the (specimen, trait) it already has stored - and writes nothing
when the pair is still being placed, because that one belongs to *Save*.

The **prediction layer** is the model's, and it is drawn to be judged, not to be a second copy of
the labels. Two of its boxes are therefore **not painted** (`SR-PREDICT-018`): the one the operator
is **editing** (a dotted original left at the old position is the "ghost lagging behind" a drag),
and the one an existing label already **covers** (IoU >= 0.5 - "one box per animal"). A drag that
turns a prediction into an annotation takes it out of the layer (`_drop_prediction`, the same path
`Backspace` uses), so the frame never shows a suggestion for an animal it has just labelled.

`Backspace`/`Delete` removes the selection, and the canvas distinguishes three kinds of removal:
`box_delete_requested` per selected corpus box (the page deletes the label), `prediction_removed`
for a model's box (nothing was stored - the page drops it from the layer), and for an editable
**point** the canvas removes it itself and reports `points_changed`. The rule lives in
**`delete_selection()`** (public): the pages bind `Backspace`/`Delete` as shortcuts to it, because
a destructive key that only works while the canvas holds focus is a key the operator presses again
and again. And in this mode a **double click is two selections, not "next frame"** - the
queue-walking binding (`double_clicked` → `next_frame`) fired mid-correction, jumping a frame and
throwing the selection away; it stays bound for the creation mode, where it belongs.

A page that has an *active* box to site (Dataset's current specimen) **must leave it empty while
`box_edit` is on**: siting it drew the selected animal twice - the dashed stored box and the
active one - which is the "two duplicate boxes" an operator reports as a bug.

`box_at` picks the **innermost** box (a box drawn around another one is the common mistake), and
the `box_edited` events are **captured before they are emitted**: a page stores each one and then
re-reads the frame's boxes, so emitting as we go made the second of two selected boxes snap back
to where the first store had left it (measured, and now a test).

The maths is pure and unit-tested without Qt: `box_at`, `resize_handle`, `moved_box`,
`resized_box` (size kept on a move, frame bounds respected, a minimum size so a box can never
be dragged through itself).

### The model on a frame: one runner, every surface (`daphnia/ui/model_pass.py`)

`ModelRunner` loads the checkpoints the **registry** says are in force (lazily, once) and runs
the top-down pipeline on one frame: `run(frame, um_per_px) -> Result(predictions, ms, note)`.
Dataset, Bench and Annotate call it from their own *Model* switch (**off by default**) whenever
a frame arrives, draw `Result.predictions`, and hand `primary(...)` - the **largest** animal -
to their editable draft. Predict keeps its streaming worker (`SR-PREDICT-004`). A failure is a
sentence (`ModelRunner.problems`), never a traceback or an empty view (law 8).

### Loading a model: one registry, one hand-over, applied at once (`SR-PREDICT-017`)

* The **registry decides which checkpoint is in force** (`models` table, one `is_active`
  row per kind). `ModelMenu.load(kind, path)` loads it *and* registers it as active, so
  "activate it in the database" and "load it" are one act; `ModelMenu.use_active(kind)`
  loads what the registry says, with no dialog.
* **One hand-over, three windows**: `model_menu.hand_over(page, kind, model)` gives the
  model to a page that declares `adopt_models` (the window tab in `SurfaceWindow`,
  `HubWindow`'s `_model_loaded`, and `MainWindow._model_loaded` all call it) and returns
  the sentence to show in the status bar when the page cannot use it. The study shell's
  connection to `loaded` was **missing**, which made a load there completely silent.
  A page that takes a model **switches its *Model* tool on** (`set_checked("model", True)`):
  the menu click is the decision to use it, and leaving the switch off made the menus look
  broken while the model was in fact loaded.
* **The menus list the models directory too** (`models.checkpoints_in`, by name
  `rtmdet*`/`rtmpose*` - `KIND_PREFIXES`; reading a file to classify it would load PyTorch to
  draw a menu, `SR-CORE-011`): a listed file is loaded, registered and activated in one click
  (`ModelMenu.activate` loads an unknown path first), it carries a check mark when it is in
  force, and the top entry carries one when this window has that kind loaded
  (`ModelMenu.loaded_paths` - *not* `loaded`, which is the signal).
* **A load is applied to what is on screen** (`Predict._apply_loaded_models`): a running
  worker is restarted on the same source (a model change is not new animals - the tracks
  survive; a video restarts at frame 1, the same cost the Keypoints toggle pays), a still
  frame is detected again, and with no source the readout and the header publish the model
  at once. Before this, a running pass kept the model it was handed: the bar named the new
  checkpoint while the boxes came from the old one.
* The page never picks a checkpoint of its own: `_require_detector` loads the **registry's**
  active model, and with none registered it says which menu loads one (law 8).
| `VIEW_MENU` | window | *Zoom in*, *Zoom out*, *Fit the image*, *Pan (hand tool)*, *Full screen* |
| `GO_MENU` | window | Home + every other available surface, never itself |
| `HELP_MENU` | app | *How to calibrate (grid paper)* (`CalibrationHelpDialog`) and *Where the data lives* (`daphnia.paths`) |

Two rules decide what goes where:

* **an action is a menu item; a choice stays beside its panel.** A button that *does*
  something moves to the bar; a combo together with the button that applies it is a
  *choice* and stays (Dataset's dataset + *New*/*Rename*, Bench's study + *Attach*,
  Setup's host/URL + *Connect*). The **workflow step's** own controls stay too (Acquire's
  *Capture*, Select's *Pick sharpest*): there the step bar gates them.
* **a page's `File`/`Source` group is not a menu of its own** - `render_tools` sorts it
  into the shared menu (`_fill_file_menu`, `_fill_source_menu`), and empties both first,
  so a step inherits nothing from the page before it.

A **canvas-less** step (Dashboard, Visualize) declares its menu items through
`ShellPage.menu_tools()` - the same `ToolGroup` a canvas page's `workspace` exposes -
and `MainWindow._follow_header` reads it when the page publishes no workspace.

`MainWindow._mount_view_menu` **empties** *View* when the step on screen has no canvas
(`view_menu(None, None, rebuild=True)`), because one canvas per step means an action bound
to the last step's canvas zooms a widget nobody can see - Acquire and Select (a live
`StreamView` and a gallery, no zoomable canvas) then offer no *View* at all, and
`hide_empty_menus` takes the menu out of the bar.

One `HeaderBar` at the top of **every** window, mounted with
`QMainWindow.setMenuWidget`, so it *is* the window's menu-bar area (one bar, not a menu
bar above a strip). It holds two things:

| Part | Built by | Content |
| --- | --- | --- |
| **Menus** | app + window + page | the seven above, in the fixed order |
| **Facts** | the page | `Kpi` records published with `workspace.set_header(...)` - the facts that are *not* a measurement |

The rule that decides **where a fact is read** (and the reason this exists):

* **on the image** (`workspace.set_kpis`) - what moves while your eyes are on the
  specimen: the headline number, its scale, the count, the speed;
* **in the header** (`workspace.set_header`) - which file/frame you are on, which model
  answered, the last message: the same whether you look at the picture or the panel.

- The values **elide** (`ElidedLabel`, `QFontMetrics.elidedText`) instead of wrapping, so
  the bar is always one line, and the whole text stays in the value's tooltip. A header
  fact **must** carry a heading (`Kpi.label`) - a bare value in a bar is a number nobody
  can name, so `HeaderFact` refuses it.
- The menubar is **not native** (`setNativeMenuBar(False)`): macOS otherwise moves it into
  the system menu, where a header is invisible.
- It is **chrome**: `WINDOW_CHROME` names `header_bar`, so full screen hides it, exactly
  like the footer.
- **One window on screen, always.** `SurfaceWindow.open_surface_window(key)` is the
  generalisation of the footer's `Calibrate...`: the window hides, the target opens
  (`WA_DeleteOnClose`), and closing it re-reads the footer and shows the first window
  again. It is driven by the `Go` menu - so `_calibration_window` is a *property* of the
  same dict, not a second field.
- The **study shell has no `Go` menu**: `MainWindow` keeps the guarded workflow (its own
  navigation is the step bar), and a page that is not on screen may not write the bar
  (`_on_header` checks `self.sender()`).



## The annotation schema (`daphnia/annotation.py`) — new pattern

A **dataset** is the corpus of exactly one **model**, and the model decides what a
label *is*:

```
dataset (datasets row: name + schema_key)
   |
   +-- AnnotationSchema  'pose_full' (4 kp, default) | 'pose_head_tail' (2 kp)
   |                     'pose_full_boxes' (4 kp + box) | 'boxes' (detection)
         |
         +-- KeypointSystem   keypoints (click order) + intervals (trait, first, second)
         |                    + EXPORT_NAMES: head, tail, sideL, sideR
         |
         +-- labels: corpus_labels (one interval per trait)   uniqueness
                     corpus_boxes  (one rectangle per specimen) (dataset, file, specimen, trait)
```

Rules that follow from it:

- **The vocabulary is the study's** (`daphnia.study` landmark names) so a corpus
  label and a study measurement name the same anatomy (`X-N08`); the short
  training-config names exist only in the export.
- **Only the model's traits are offered.** `labeling.intervals_for(schema)` is the
  single place that turns a keypoint system into traits; the canvas, the draft and
  the export all read from it (`Draft(intervals=…, landmarks=…)`).
- **Another model = another dataset.** Labels are scoped by `dataset_id`, so
  re-purposing a corpus is impossible by construction, and the same frame can be
  ground truth for two models.
- **A box is not an interval.** It lives in `corpus_boxes` (its own table, its own
  uniqueness), is drawn by dragging in `ImageCanvas` (`box_mode`, `box_changed`,
  `set_boxes` overlays), and is exported to `boxes.csv` /
  `labels.json:specimens[].bbox`. Selecting a specimen *sites* its stored box as the
  editable one, so a correction is a redraw, not a blind new rectangle.
- **One export = one dataset**: `images/` + `labels.csv` (+`p1_name`/`p2_name`) +
  `boxes.csv` + `labels.json` (schema, per-specimen records, `splits`) +
  `splits/{train,val,test}.txt` + `manifest.json`, into
  `datasets/annotations/<dataset name>_<stamp>/`, so two models never share a folder.
- **Naming is never a failure mode.** A dataset name collision is numbered
  (`repository.next_dataset_name`), and the model can never be switched on an
  existing dataset (a rename locks it) - the schema is what its labels mean.
- **Scope changes rebuild, they do not truncate.**
  `connection.rescope_corpus_labels()` copies the legacy table into the new shape
  before dropping anything, and `repository.adopt_legacy_corpus()` gives the
  pre-dataset rows to the first dataset - the corpus is never disposable
  (`SR-DATASET-N03`).

## The annotation canvas (`ImageCanvas`) — interaction contract

The canvas is the one place where clicks become ground truth, so its gestures are a
deliberate contract, not a convenience:

| Gesture | Meaning |
| --- | --- |
| **hover** a keypoint, box or prediction | it is highlighted (accent ring/outline) and the cursor becomes a grab hand, reporting `ImageCanvas.hover`; the press will edit it in **any** tool (`X-N15`) |
| left click | inside the **Tool** in force (`X-N14`): **`line`/`keypoints`** place the next point (in box mode a corner is anchored instead, a second click or a drag finishes the box); **`select`** picks what is under it; **`box`** anchors a corner on empty space and selects a stored box - but a press **on** a grabbable target edits it in any tool (`X-N15`), and only empty space runs the tool's own gesture (`SR-DATASET-026/027`) |
| left click, `point_mode` on | **the press is the keypoint gesture**: within reach of a point it picks that point up, anywhere else it places the next landmark - no box is selected and empty space starts no marquee (`SR-DATASET-028`) |
| **two-finger scroll** | **pan, in both axes - never zoom** |
| two-finger **pinch** | zoom towards the cursor |
| `+` / `-` / `F`, the *Zoom +/-*/*Fit* **tools on the image** | zoom in / out / fit |
| ⌘/Ctrl + drag, right/middle drag, arrow keys, the **Pan** tool | pan, placing nothing |
| click on a tool cluster | a tool (or the header, which folds the cluster away) |
| **double click** | the page's call - the Dataset queue reads it as *next frame* |

- **Zooming is deliberate.** Scroll used to zoom (and `⌘+scroll` too); it now only
  pans, because an accidental zoom while clicking is an *error in the label*, not an
  inconvenience. The pan is proportional to the wheel/pixel delta, so a trackpad's
  many small deltas move gently.
- **A gesture must not label.** `mouseDoubleClickEvent` places nothing (Qt sends it
  *instead of* a second press, and it aborts a box preview the first press opened),
  and `Dataset.next_frame()` takes back whatever the gesture's own first press did
  (`_take_back_click`) using the same undo a mistaken click gets.
- **A corner is not a box.** The box state machine is one anchored corner
  (`_box_origin`) plus the cursor, so *drag* and *click-click* share it:
  `anchor_box_corner()` on press, preview on move, and `mouseReleaseEvent` stores only
  a box the gesture actually made (>= `MIN_BOX_PIXELS`). A click therefore anchors and
  waits, `box_pending`/`box_origin` expose it, `box_anchor_changed` tells the page to
  re-word the prompt without ever being mistaken for "a box was stored", and `Esc` /
  right click / `clear_box()` abandon it.
- **Painting is tested through a painter.** A raise *inside* `paintEvent` leaves the
  painter active and **segfaults the process** while the test suite stays green - one
  `QPoint + QPointF` in the box overlay did exactly that. `tests/test_image_canvas.py
  ::PaintTests` therefore opens a real `QPainter` and calls the overlay painters from
  the test's own frame, so a broken draw is an ordinary test failure.

## Cross-cutting pattern: the box queue and the crop (`vision/crop.py`)

The Dataset surface can work **one animal at a time** instead of one frame at a time
(`SR-DATASET-022`, `-025`), and the pattern is worth copying for anything else that
labels per object:

1. **The queue item is the object, not the picture.** `Dataset._items` holds one entry
   per stored box (`path`, `specimen`, `box`, `source`, `size`) and is built **from the
   corpus**, so an object added by hand is queued too. `step()`/`next_frame()`/`A`/`D`
   walk items when there are any, frames otherwise, and the position readout says
   which.
2. **One object owns the mapping, with an `invert`.** `daphnia/vision/crop.py`:
   `crop_for_box(box, image_size)` → `Crop(origin, size, scale)`, whose
   `to_crop`/`invert` are the *only* conversion between the magnified picture and the
   frame. `window()` slices with the same integers `invert()` uses, so the two cannot
   disagree, and the resampling is nearest-neighbour at an integer factor so the
   enlargement is exact. This is `Letterbox`'s arrangement applied to the labeller, and
   it exists because the same bug class (a coordinate mapping applied on one side only)
   once made a metric read 0.000 for thirty epochs (`mlContext.md` law 1).
3. **A prediction is not ground truth until a person says so.** A `source='model'` box
   is answered by **Yes** (rewrite to `source='manual'`, name the accept) or **No**
   (delete the row) - `repository.set_corpus_box_source`. The queue **waits** on an
   unanswered prediction instead of advancing, so it cannot be skipped silently, and
   the session counts both answers in the queue summary.
4. **A view is only a view.** One item has two pictures - the whole frame with every
   box, or the magnified crop - and switching between them must not change what a label
   stores: `_crop` is *the mapping in force* (``None`` in frame view, so a click already
   is a source pixel), and `_crop_point` is the single conversion. The same discipline
   as `Letterbox`: one object owns the mapping, callers never re-derive it.
   **A box is no different from a click**: `crop_box` draws a stored box on the
   enlargement and `source_box` is the way back for a *dragged* one, with
   `_source_size()` naming the frame the row belongs to. Both were learned the hard way -
   the drag shipped without them, so a correction made in the crop was stored as the
   crop's own pixels against the crop's own size (specimen 1 of a 1280 x 720 frame became
   ``(82, 79, 382, 259)`` on a "450 x 330" image).
   **And the press has to mean one thing per view**: the item's box covers the whole crop,
   so the ordinary order (a press inside a box selects it) ate every click and no landmark
   could be placed - `ImageCanvas.point_mode` is on while a keypoint corpus is shown as a
   crop, a point under the cursor is still picked up first, and a box is corrected in
   *Frame* view where every box is visible (`SR-DATASET-028`).
5. **Two Qt traps this paid for, worth not repeating:**
   * `isHidden()` is true for **every** child of a window that has not been shown yet -
     so a widget that asks "should I lay this out?" with `isHidden()` places nothing at
     all before the window is up. Track the flag (or use `isVisibleTo`) instead.
   * a `QMessageBox` in a method a **test or a script** drives blocks for ever (it hung
     the whole suite). Dialogs belong in the button handler; the core method refuses in
     words (`X-N10`), and `ToolCluster.relayout()` re-flows the visible tools so a
     hidden one never leaves an empty cell - or a `wide` one on top of its neighbour.

## Cross-cutting pattern: the on-canvas tools (`CanvasWorkspace`)

Five surfaces share one canvas, so they share one set of tools (framework `X-N11`,
`X-N12`). Before this, each screen wired its own zoom row inside its own ``.ui`` file -
three copies of the same three buttons, none of them reachable in full screen.

```
ImageCanvas  <-- children of the canvas -->  CanvasTools            (canvas_tools.py)
                                              View  (top right)  : Zoom + / Zoom - / Fit / Pan / Full screen
                                              Tools (lower left) : whatever the page adds
                                              readout (top left) : a column of KPI cards, one fact each
                                                                   (KpiStack: heading / value / caveat flag)
                                                    ^
CanvasWorkspace (canvas_workspace.py) : owns the clusters, the page's tools, the
                                        full-screen view, the chrome to hide, the readout
Page (one line)  : self.workspace = self.canvas_workspace(self.canvas)
                   self.workspace.add_tool(ToolSpec("save", "Save", ...), self._save)
```

1. **The tools are children of the canvas** and pinned to their corner by an event
   filter, so they sit *on the picture* and survive full screen. A cluster collapses
   to its header; a cluster with no tools hides itself - and one whose tools have all
   been hidden for the current state hides too (`ToolCluster.sync_visibility`), so a
   live feed shows no *frame* controls and a canvas without measuring tools shows no
   cluster at all. That is the **conditional overlay**: `workspace.set_visible(key,
   applies)` is how a page says "not now" (Predict: only a video file has frames to
   step to, `SR-PREDICT-011`).
2. **No button takes focus** (`Qt.NoFocus`) and a cluster sets
   `WA_NoMousePropagation`: clicking a tool never steals `+`/`-`/`F`/arrows from the
   canvas, and a click that misses a button can never drop a landmark behind the
   panel. Both are tested (`tests/test_canvas_workspace.py`).
3. **Full screen hides chrome; it never reparents the canvas** (`CanvasFullScreen`).
   The panel, the sibling visuals (the live preview) and the window's
   `header_bar`/`step_bar`/`status_footer` (`WINDOW_CHROME`, collected by `page_chrome()`)
   are hidden and the **window** goes full screen, so the canvas keeps its points, its
   overlays, its tools and its zoom. Only what was visible comes back (a control the
   page had hidden stays hidden), `Esc` leaves, and the shortcut is enabled **only**
   while full screen - outside it, `Esc` keeps its meaning (abandon the box).
4. **The readout is what makes full screen honest.** The page says which facts are KPIs -
   `workspace.set_kpis([Kpi("length", text, label="Body length", primary=True), ...])` - and
   **one card per fact** appears on the image, the headline first and in larger type.
   `set_readout(text)` is the plain route: one card per line, no heading, for a sentence
   with no KPI in it (the prompt, the last message). A Bench action tool calls through
   `_tool_action()` so its answer lands on the image too. Without it, full screen would hide
   the very number being measured.
   * One card is **one fact**: a micro-caps heading, the value, and - when the value needs
     it - a **flag** (`WEAK`, `STALE`, `UNSCALED`) with a tone. `Kpi` refuses a tone the
     stylesheet does not select on, and the card's `objectName` follows its shape
     (`canvas_kpi_card` with a heading, `canvas_readout` without one), so the styling can
     never disagree with the widget.
   * The **colour is the theme's** (`PHOTO_OVERLAY` + `[tone="warning"]`), the hierarchy is
     the **type** (9 px caps / 12 px value / 17 px headline) - a KPI is a fact, not a
     decoration. Letter-spacing is the one type decision QSS cannot express; it is set on
     the font in `canvas_tools.py`.
   * **Keys, not rebuilds**: `set_kpis` re-uses each card while its key *and shape* are
     unchanged, because Predict rewrites its readout on every frame - a rebuild per frame
     is how a readout becomes a leak.
   * **A card is as wide as its words** (`ElidedLabel.sizeHint` overrides Qt's, which came
     from the empty text ``QLabel`` was given): `KpiStack.natural_width()` is that sum, and
     the strip takes it up to `canvas - 2 * MARGIN`. It shipped without the override, so
     every card asked for a sliver and a sentence was elided on a canvas with 700 px to
     spare - *"the info box overlay is very small and needs to be as wide as image
     canvas"*. `minimumSizeHint()` stays `VALUE_MIN_WIDTH`, which is what lets a narrow
     canvas elide instead of overflowing.
   * **A fact that would push the column past the canvas is not a card.** The column is
     measured against the room above the lower-left cluster (Predict, pose loaded, minimum
     window: **239 px of column against 166 px of room** - a sixth, checkpoint card reached
     367 px, which is why it is in the panel and not on the picture). The card set per
     surface is therefore a *decision with a number behind it*, not a dump of the panel.
5. **Tested once, in the abstract** (`tests/test_canvas_workspace.py`, 38 cases). The
   `KpiCardTests` hold the readout: one card per fact and in the page's order, the
   headline's `emphasis`, a headingless card being the plain readout line, a blank value
   never drawn, two cards never sharing a key, a card re-used while its words change and
   replaced when its shape does, a flag taking its tone from the theme, one column width in
   the corner the View cluster leaves, and no card eating a click. The rest hold the tool
   clusters: corners, folding, callbacks, "the panel does not mark the image", chrome going
   and returning, and a **pixel** test (`grab()` + `pixelColor`) proving the translucent
   panels are really painted over the image - the one thing a widget-tree test cannot see.
   `tests/test_ui_smoke.py` keeps the per-screen contract (the tools are children of *this*
   canvas, nothing duplicates them in the panel).

## Cross-cutting pattern: `from_row` classmethods

`StudySpec`, `StudyTemplate`, `OpticalProfile` all build from a DB row via
`from_row(cls, row)`, tolerating newer/older columns (e.g. `protocol_key` may be
absent from an old row). Follow this for every new read model.

## Storage layout (`daphnia/paths.py`)

`datasets/{raw,processed,annotations}` · `models/` ·
`recordings/`
(`<study_id>/<session_id>/<observation_id>/b<burst>_f<frame>.png`) ·
`recordings/calibration/` · `zoo.db`. All git-ignored except `.gitkeep`.

## Surface / launch-point pattern (implemented for the hub)

```
run.py -> daphnia.ui.app.main(argv)
             |
             +-- parse_launch(args) ------> key ('hub' when no argument given)
             |
             +-- daphnia.surfaces --------> the registry (Qt-free: the hub's data)
             |
             +-- hub.window_for(key) -----> +---------+--------+---------+---------+----------+------------+
                                            |         |        |         |         |          |            |
                                       HubWindow  MainWindow BenchWindow DatasetWindow PredictWindow CalibrationWindow
                                        (Home)     (STUDY)   (BENCH)   (DATASET)  (PREDICT)   (OPTICS)
```

All five surfaces are **built** today: `surface.status == available` for study,
bench, dataset, predict and optics (CORE is the library). `SurfacePlanWindow` still
renders a `planned` surface's specification; the next surface added as `planned`
opens there instead of any other surface's window (`SR-CORE-014`).

- A surface = (a launch point) + (a gate policy) + (save bookkeeping). **Nothing
  else.** Every surface shares `daphnia.surfaces` for its description,
  `daphnia.ui.shell.py` for its look, and `zoo.db` for its data.
- **`daphnia/surfaces.py` is the single source of truth**: the hub renders its
  tiles, `run.py --help` prints its lines, and `daphnia/ui/hub.py::WINDOW_FACTORIES`
  maps it to a window. It is Qt-free so all of that works headless.
- **Adding a surface** = one `Surface(...)` entry + one
  `docs/requirements/SR-<SURFACE>.md`. Until its window is registered in
  `WINDOW_FACTORIES`, its launch point opens its *specification* - so a launch
  point can never silently open the wrong thing (`SR-CORE-014`).
- Bench/Dataset will persist with `study_id = NULL` and be **promoted** later by
  setting `session.study_id`: provenance without the setup cascade.
- **Exception:** only the hub and `daphnia/ui/app.py` may reference a surface's
  window. A surface must not import another surface (`X-N04`).

## Cross-cutting pattern: the off-thread worker (`InferenceWorker`)

The Predict surface runs a model that may be slower than the frame rate, so it uses
a `QThread` (`daphnia/ui/pages/predict.py::InferenceWorker`) with three rules worth
copying for anything else that keeps up with a camera:

1. **Always the newest frame.** The loop compares the *identity* of the frame the
   `StreamService` last buffered and sleeps when it has already seen it, so a 30 fps
   stream with a 12 fps detector shows twelve boxes a second instead of working
   through a backlog it can never clear.
2. **The page owns the thread's life.** `PredictWindow.closeEvent` calls
   `page.stop()`, and a `QThread` destroyed while running **aborts the process** -
   the same trap `streaming.reader` documents.
3. **A failure is a message, not a traceback.** The worker's `run()` catches,
   emits `message(str)` and stops; the page shows it. A silent empty view would be
   blamed on the model.

The thread also never touches the canvas: it emits `(frame, detections, ms)` and the
page paints, which is why the panel can be tested with a **stub detector** and no
model at all.
