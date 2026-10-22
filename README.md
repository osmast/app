# DaphniaScope

Automated, non-invasive **body-length measurement for living *Daphnia magna*** in
multi-well plates — an imaging + computer-vision pipeline wrapped in a Qt desktop
application. Developed for the SNSF "Aquaculture Systems" project at ZHAW.

It replaces manual stereomicroscope sizing, enabling high-throughput longitudinal
growth assays with a reproducible, objective endpoint: body length, defined as the
distance from the anterior margin of the compound eye to the base of the apical
spine.

> **Status:** navigation skeleton + persistence + streaming scaffolding. The
> screens and the algorithm modules are placeholders — nothing is measured yet.

## Quickstart

```bash
conda env create -f environment.yml     # once
conda activate z-app
python run.py                           # Home hub; or: python run.py study
```

Everything is also wrapped in `make`:

```bash
make run              # launch the desktop app
make test             # run the test suite
make ui-check         # paint every surface offscreen, full screen and back
make db               # (re)initialise the SQLite database
make stream           # probe the configured MediaMTX stream
make stream-publish   # publish an ffmpeg test pattern to MediaMTX
make mediamtx-setup   # install config/mediamtx.yml system-wide
make train-check      # prove PyTorch + MPS + OpenCV work before a long run
make train            # train RTMDet on datasets/bb
make train-inspect    # describe the corpus the trainer would read
```

## Run it on another machine (macOS or Windows)

Nothing is compiled and nothing is installed system-wide: a clone, one conda
environment, one SQLite file. **MediaMTX runs on Windows as well** — it is a single
executable with no dependencies (*"Compatible with Linux, Windows and macOS"*, its
README) — so the only genuinely platform-specific parts of this repository are the two
`scripts/*.sh` helpers, the `make` targets and one hard-coded Homebrew path.

| Step | macOS | Windows |
| ---- | ----- | ------- |
| 1. Code | `git clone … && cd app` | the same |
| 2. Environment | `conda env create -f environment.yml` then `conda activate z-app` | the same — `environment.yml` asks for `conda-forge` only, which builds for `win-64` |
| 3. Run | `python run.py` | `python run.py` |
| 4. Stream server | `brew install mediamtx` · `./scripts/setup_mediamtx.sh` · `brew services start mediamtx` | download the Windows archive from the [releases page](https://github.com/bluenviron/mediamtx/releases) (`mediamtx_v1.21.2_windows_amd64.zip` as of writing — take the newest for your architecture), unzip it, then `mediamtx.exe config\mediamtx.yml` |
| 5. Phone | Moblin → SRT → `srt://<ip>:8890?streamid=publish:live/cam` | the same |
| 6. That `<ip>` | `ipconfig getifaddr en0` | `ipconfig` |

`make` is a convenience layer, not a requirement. On Windows run what the target runs:

| `make …` | the command it wraps |
| -------- | -------------------- |
| `make run` | `python run.py` |
| `make test` | `python -m unittest discover -s tests -t . -v` |
| `make ui-check` | `python scripts/check_ui.py` |
| `make db` | `python scripts/init_db.py` |
| `make stream` | `python scripts/check_stream.py` |
| `make mediamtx-run` | `mediamtx config/mediamtx.yml` — on Windows `mediamtx.exe config\mediamtx.yml` |
| `make stream-publish` | **macOS/Linux only** (bash + `ffmpeg` + `srt-live-transmit`): a synthetic test pattern, not part of the app |

Three things are macOS-shaped and none of them is required:

* `scripts/setup_mediamtx.sh` copies the repository's config into Homebrew's
  `/opt/homebrew/etc/mediamtx/`. On Windows there is nothing to copy — run the
  version-controlled `config/mediamtx.yml` in place, as step 4 does.
* `daphnia/streaming/mediamtx.py::SYSTEM_CONFIG` names that same Homebrew path, and it
  is only ever *reported* by `mediamtx.status()`.
* `daphnia/vision/images.py` uses `sips` to convert a format OpenCV cannot read. It is
  guarded by `sys.platform != "darwin"`, so elsewhere it is a no-op and the caller
  reports the unsupported format instead of pretending to load it.

**The binary is found, not assumed installed.** `mediamtx.binary_path()` is
`shutil.which(settings.MEDIAMTX_BINARY)`, so either put `mediamtx.exe` on `PATH` or set
`DAPHNIA_MEDIAMTX_BINARY=C:\tools\mediamtx.exe`. The app reads the same RTSP URL either
way, because the phone, not the app, is the SRT publisher.

**Firewall.** The machine running MediaMTX must accept inbound **UDP 8890** (the phone's
SRT publish) and **TCP 8554** (OpenCV's RTSP read). macOS asks once when `mediamtx`
first listens; Windows Defender asks on first run — allow *Private networks*. A phone
that "cannot connect" is almost always this, or a different network (`docs/streaming.md`
§Troubleshooting).

### Where the data lives (and how to move it)

`DAPHNIA_DATA_DIR` is the root of everything the app **writes** — `zoo.db`, `recordings/`,
`datasets/`, `models/` — and it defaults to the checkout, which is what a development run
wants. It exists so a *packaged* app can point its state somewhere writable instead of
inside its own bundle:

```bash
export DAPHNIA_DATA_DIR=~/DaphniaScope          # macOS / Linux
set DAPHNIA_DATA_DIR=%LOCALAPPDATA%\DaphniaScope  # Windows (cmd); $env:… in PowerShell
```

`DAPHNIA_DB` still points at one database file and wins over the root. Copying `zoo.db`
to another machine moves the studies, the labels **and the calibrations** — but a
calibration is only true on the rig it was measured on (another phone clamp, another
objective, another zoom is another `setup_key`), so calibrate there before trusting a
number: the footer will say `RE-VERIFY` for a stale scale, and it cannot tell you that
the *hardware* changed (`docs/accuracy.md` S10, S13).

PyTorch's **MPS** backend is Apple-only. `train.pick_device` takes MPS where there is one
and CPU otherwise, and asking for `mps` on a machine without it is an *error* rather than a
quiet 20× slowdown. So on Windows the detector and the pose model run on CPU, and Predict's
latency there is **unmeasured**: run `python scripts/check_torch.py` and
`python scripts/train_rtmdet.py --evaluate` on that machine before promising a frame rate.

## Pipeline

Seven steps (`daphnia/pipeline/steps.py`):

```
Ingest -> Localize -> Calibrate -> Predict -> Derive -> Persist -> Aggregate
```

| Step | Module | Responsibility |
| ---- | ------ | -------------- |
| Ingest | `daphnia.streaming` | Frames from the MediaMTX stream (`StreamReader`). |
| Localize | `daphnia.vision.detection` | Isolate the specimen in the well (Mode A/B). |
| Calibrate | `daphnia.vision.calibration` | Pixels → micrometres via the optical profile. |
| Predict | `daphnia.vision.keypoints` | Eye apex and spine base. |
| Derive | `daphnia.pipeline.derive` | Body length in micrometres, reported in mm too (`daphnia.units`). |
| Persist | `daphnia.pipeline.persist` + `daphnia.db` | SQLite writes. |
| Aggregate | `daphnia.pipeline.aggregate` | Longitudinal series, dose-response, survival. |

## Screens

The application opens on **Home**, the hub that offers the surfaces; `python
run.py study` opens the study shell directly.

### Surfaces (launch points)

| Command | Surface | What it is for |
| ------- | ------- | -------------- |
| `python run.py` | Home | the hub: choose a surface |
| `python run.py study` | Study | the governed pipeline below (built) |
| `python run.py study2` | Study 2 | capture a whole 6/24/96-well plate, well by well (built; will replace Study) |
| `python run.py bench` | Bench | measure a specimen now, no study needed (built) |
| `python run.py optics` | Optics | calibrate the optics from the 1 mm grid (built) |
| `python run.py dataset` | Dataset | build and label the training corpus (built) |
| `python run.py predict` | Predict | watch RTMDet find the animals, live or on a video (built) |
| `python run.py --help` | — | list the launch points |

One repository, one `daphnia` package, one database - only the entry point and
the gate policy differ. `daphnia/surfaces.py` is the registry that the hub renders
and `run.py` dispatches on; a surface that is not built yet opens its
specification rather than the wrong window. Each surface has its own requirement
document in `docs/requirements/`.

### Study 2 — the plate page

`python run.py study2` is the **acquisition half** of the plate routine the project is
about, built beside the governed pipeline and meant to replace it: one plate of 6, 24 or 96
wells drawn in its own shape, each well lit by its state (*not captured* / *captured* /
*measured*) and - once it has one - showing its own frame inside the well, so the plate
fills up as it is walked.

```
2. Plate: 6 / 24 / 96, one per study   [Capture burst]  [Start recording]  [Next well >]
3. Wells: n per page (a page is whole rows)      <- the plate lights up here
4. Capture: the live preview + the chosen well's frames, its image selected
```

The loop is: put the well under the camera, take a **capture**, move the plate, *Next
well* - which wraps around the plate, because the well you missed is the interesting one.
A capture is either a **burst** (*Capture burst*: the `DAPHNIA_BURST_SIZE` frames the
stream already buffered, the sharpest of them becoming the well's image) or a
**recording** (*Start recording* / *Stop recording and store*, `SR-STUDY2-013`): every
frame that arrives in between is written to the well's folder as it comes and the stop
stores the whole take as **one burst** of that well - so a well that drifts in and out of
focus gets as many frames as it needs, not five. While a recording runs, the plate is
locked to its well and the line under the button counts frames and seconds
(`SR-STUDY2-014`); `DAPHNIA_RECORD_MAX_FRAMES` (300 ≈ 10 s ≈ 360 MB) is only the disk
guard, and reaching it is said out loud (`SR-STUDY2-N06`).

A capture needs a study (it owns the folder), a plate, a well, the camera and an active
scale; it does **not** need the seven-step order, a selection or an annotation. Measuring
a well's individuals is the next step (`docs/requirements/SR-STUDY2.md`,
`SR-STUDY2-006`).

### Study routes

| Route | Screen |
| ----- | ------ |
| `dashboard` | Overview — project monitor, where each project stands |
| `experiment` | Project & Experiment Setup (study, plates, phone connection) |
| `calibration` | Optical Calibration Wizard |
| `acquire` | Live Acquisition & Well Scanner |
| `select` | Frame Selection — one best frame per well / individual |
| `annotate` | Measurement & Annotation Editor |
| `visualize` | Longitudinal Visualization & Analytics |

Screens are separate `QWidget`s loaded from `assets/ui/*.ui` at runtime;
`daphnia/ui/main_window.py` is the parent controller that owns the
`QStackedWidget` and the navigation. Buttons are wired by convention — give a
`QPushButton` the `objectName` `nav_<route>` and it navigates automatically.
`daphnia/ui/hub.py` is the launcher window that owns the hub and the surface
windows.

## Layout

```
app/
├── run.py                  entry point (python run.py)
├── Makefile                convenience targets
├── assets/                 ui/*.ui and theme/*.qss
├── config/mediamtx.yml     version-controlled MediaMTX configuration
├── daphnia/                the Python package
│   ├── paths.py            every project directory, in one place
│   ├── settings.py         runtime settings (env-overridable)
│   ├── plates.py           multi-well plate geometry (6 / 24 / 96)
│   ├── wells.py            well states, pages and the next-empty walk (Study 2)
│   ├── protocols.py        named acquisition presets (burst + QA trade-off)
│   ├── templates.py        versioned study templates
│   ├── maintenance.py      prune / export / consolidate a study
│   ├── units.py            um <-> mm formatting for reports
│   ├── db/                 connection + schema.sql          (Persist)
│   ├── streaming/          mediamtx, reader, sources        (Ingest)
│   ├── vision/             calibration, detection, grid, lattice, images, keypoints
│   ├── pipeline/           steps, derive, persist, aggregate
│   └── ui/                 app, main_window, loader, theme, pages/
├── datasets/               raw/ processed/ annotations/     (git-ignored)
├── models/                 trained weights                  (git-ignored)
├── recordings/             burst captures / segments        (git-ignored)
├── scripts/                setup_mediamtx.sh, dev_publish_stream.sh,
│                           check_stream.py, init_db.py
├── docs/                   architecture.md, streaming.md
└── tests/                  stdlib unittest
```

## Step-based workflow

The app is not a menu of tasks: it enforces the order of an experiment
(`daphnia/workflow.py`).

```
1 Overview -> 2 Setup -> 3 Calibrate -> 4 Acquire -> 5 Select -> 6 Annotate -> 7 Analytics
```

Every step declares a *fact* it produces, and a step can only be **entered** once
every fact from the earlier steps exists in the database **for the active project**
- so the order cannot be bypassed from the UI, even by accident, and a new project
never inherits another project's captures, selections or measurements:

| Fact | Produced by | Why it gates |
| ---- | ----------- | ------------ |
| `study` | Setup | it decides **where the images are saved**; nothing may be captured before it exists |
| `connection` | Setup (runtime) | the phone is connected **once**, on the Setup step, and both Calibrate and Acquire require it - nothing is calibrated or captured off a stream that is not there |
| `calibration` | Calibrate | an image without a scale can never be measured, so Acquire stays locked until an optical preset is active |
| `observation` | Acquire | Select needs bursts to compare |
| `selection` | Select | exactly one best frame per well / individual; Annotate stays locked until it exists |
| `measurement` | Annotate | Analytics needs something to plot |

`MainWindow` offers only **Back/Continue**: Continue is disabled while the next
step's requirement is unmet and the step bar says why. The app always opens on
**Overview**, so nothing is resumed behind your back. `Acquire` shows the resolved
destination folder and the active scale before every capture, so "where does this
image go" and "is this calibrated" are answered on screen rather than assumed.

### Reviewing captures (Select)

Select works one **target** at a time - a well or an individual - and shows **every
frame** of that target in a grid. Use **Previous / Next** (with the `Individual n of
N` label) or the dropdown to move back and forth between captures; the individual
you were on is kept when the list reloads. Tick the frame that shows the specimen
best (or *Pick sharpest frame*); the others go to the tray, so each target yields
exactly one measurement. Double-click opens a frame full screen.

### Phone connection

The connection is made **once, on the Setup step**: the host field defaults to this
Mac's LAN address (`daphnia.streaming.lan_ip`) and the page prints the exact SRT URL
to type into the phone's broadcaster (e.g. the **Moblin** iOS app), for example
`srt://192.168.0.24:8890?streamid=publish:live/cam`. Every other screen binds a
preview to that one connection
(`daphnia.streaming.StreamService`), so nothing reconnects per screen. Calibrate and
Acquire require it; Select, Annotate and Analytics deliberately do not - once the
frames are on disk the phone can be put away and the work continues offline.

A **`?  How to set up calibration`** button opens a step-by-step dialog (mount the
phone with zoom at 1.0x and focus at infinity, name the chain, set the printed square
size, place the grid paper, take one photo, press *Measure grid*, read the verdict,
*Store this scale*) and is reachable from both Setup and Calibrate.

### Study modes

A study is not one thing, and the Setup step is where that gets decided. Three axes
are stored on the study itself (`daphnia/study.py`); every screen reads the
resulting `StudySpec` instead of assuming, so the **Setup config drives the rest of
the app**:

| Axis | Values | Effect on the UI |
| ---- | ------ | ---------------- |
| `target_mode` | `plates` / `individuals` | Acquire shows the plate + well pickers, or only the specimen counter |
| `organism_kind` | `animal` / `plant` | the anatomy that gets annotated: *head -> spine base* **body length**, or *base -> tip* **shoot length** |
| `timecourse` | `longitudinal` / `single` | repeated timepoints vs measured once |

`StudySpec` exposes exactly what screens ask - `uses_plates`, `is_longitudinal`,
`keypoints`, `endpoint`, `specimen`, `describe()` - so supporting another mode is a
vocabulary entry plus whatever that screen does with it. Verified end to end:

```
Plant  / Multi-well plates / Measured once :
    keypoints ('base', 'tip'), endpoint 'shoot length', specimen 'plant'
    plate pickers shown -> "Plate 1 well B02 (5/6)", annotate canvas labelled base/tip

Animal / Single specimens  / Repeated over time :
    plate pickers hidden -> "individual 1", keypoints ('head', 'spine base')
```

Calibration is deliberately **mode-independent** - the optics do not care what is
being measured - and Select, Annotate and Analytics all read the same `StudySpec`,
so a plant study measures shoot length without a single change outside the spec.

## Application shell

Every screen sits inside one shell (`daphnia/ui/shell.py`):

```
+--------------------------------------------+--------------------+
| visual area, edge to edge, no margins      | controls panel     |
| (live stream top-left, captured frame,     | fixed width,       |
| table, ...) expands to fill                | padded             |
+--------------------------------------------+--------------------+
| step bar: 1 Overview 2 Setup ...  hint       [Back] [Continue]  |
+-----------------------------------------------------------------+
```

* There is **no free navigation between steps** - the step bar is the only way to move
  through a study and it is guarded by `daphnia.workflow`. The header's menus act on the
  *window* and on the page on screen (`File`, `Source`, `Models`, `View`, `Help`), never
  on the order.
* `ShellPage` gives a screen its body: a left **visual area with no margins at all**
  (a live stream runs flush to the window edges and keeps a large minimum size, so
  it never collapses) and a right **controls panel**. The controls come from the
  screen's `.ui` file (still editable in Qt Designer - see *Editing the screens in Qt
  Designer*); the visual widgets are
  code-built (`StreamView`, `ImageCanvas`, `QTableWidget`) and added with
  `add_visual()`.
* The **controls scroll inside their own panel** (`QScrollArea#controls_scroll`), so a
  tall form never stretches the page: the visual area always fits the viewport and
  the **live stream stays at the top-left** without scrolling. The whole body
  (`QScrollArea#body_scroll`) still scrolls as one region as a safety net for very
  small windows, and every step opens scrolled to the top.
* **The panel is wide enough for what is in it** — measured, not guessed:
  `CONTROLS_WIDTH = 420` fits every shared screen and wraps the explanatory sentences in
  two or three lines instead of six.
  Because the panel scrolls **vertically only**, a screen whose content needs a
  different width carries its own `PANEL_WIDTH`: the **Dataset** panel is 500 px, sized
  from its dataset row (a combo showing `<name> [<what it labels>]` beside two buttons,
  which needs 458 px plus padding and the scroll bar), and the **Predict** panel is
  **340 px** — its content needs 262 px, and the 80 px it gives back go to the picture
  the detector is judged on. The **Calibrate** panel is **500 px** for the same measured
  reason: its result block, verdict sentence and distortion note wrap at 420 px into five
  more lines, which is the difference between *Store this scale* being on screen and being
  one scroll below the fold. What no panel may be is narrower than its own content.
  `tests/test_ui_smoke.py` measures this for every built surface and `make ui-check` for
  every window, in both theme schemes, so a new control that does not fit fails a test
  instead of disappearing off the edge.
* **The page's own actions are menu items in the header** (`X-N13`, superseding
  `X-N11`; `daphnia/ui/widgets/canvas_tools.py`, `canvas_workspace.py`). Every canvas
  used to carry two translucent, expandable tool clusters floating over the image -
  *View* at the top right (*Zoom +*, *Zoom -*, *Fit*, *Pan*, *Full screen*, with the
  magnification in its header) and a measuring cluster at the lower left. The view
  actions are now the *View* menu, and each screen's own actions form **its own menu**:
  *Measure* on Bench (*Undo*, *Clear points*, *Clear this view*, *Refresh*), *Label* on Dataset
  (*Box*, *Yes*/*No*, *Crop*/*Frame*, *< Prev*/*Next >*), *Frames* on Predict (*< Frame*,
  *Frame >*, *Keypoints*), *Calibration* on Calibrate (*Measure grid*, *Store this scale* -
  the one screen whose two actions are also panel buttons, see `systemPatterns.md`).
  Nothing is drawn over
  the photograph but a page's own **draw tools** (`X-N20`, the bar on the Bench and the
  Dataset canvases). A tool a state cannot use is **hidden** rather
  than disabled, and a group with nothing left to offer leaves the bar with it. The page
  asks for all of it in one line - `self.canvas_workspace(self.canvas)` - and adds its
  own tools with `workspace.set_group("Measure")` + `workspace.add_tool(...)`; the entry
  points know nothing about it.
* **A model is a tool the operator reaches for: *Model* is off by default** (`SR-PREDICT-018`,
  `daphnia/ui/model_pass.py`). Loading one from the **Detector (RTMDet)** / **Pose (RTMPose)**
  menus switches it **on** - the menu click is the decision to use it. With the switch on, **Dataset** and **Annotate** run
  the selected models once on every frame they show - Predict streams them - and the answer is
  drawn in the canvas' own *prediction* layer (dotted, one colour, never a stored label's
  colour). The **largest** animal's answer also lands in the **editable draft**: on Dataset
  that is the box you drag onto the animal you mean (letting go stores it) or the two keypoints
  you move before *Save*; on Annotate it is the measured pair. Correcting a prediction
  is therefore the same gesture as labelling by hand, and nothing is ever stored by a prediction
  (`SR-PREDICT-N01`). Turning the switch off clears both the layer and what it put in the draft.
  **The Bench is not one of these screens** (`SR-BENCH-017`, removed 2026-10-22): it displays
  measurements, so it carries no *Model* switch and its window offers no checkpoint menus
  (`hub.BenchWindow.OFFERS_MODELS = False`) - a model is loaded and watched on Predict, which is
  the screen built for judging it.
* **What a press means is a *tool*, not an "edit mode"** (`X-N14`, `daphnia/ui/widgets/image_canvas.py`).
  Editing used to be one boolean (*Edit*, on by default) doing four jobs at once, so a page could
  not offer *select* without turning *create* off - and a measuring surface opened in a mode where
  the first click selected instead of measuring. The gesture is now an explicit, mutually-exclusive
  **Tool** group on every labelling/measuring canvas - `ImageCanvas.tool`:
  * **Line** - place two points and they are one measured line (never selects). On **Bench** the
    ends are **A** and **B** and the line **joins this view the moment B lands**, so A_1/B_1 then
    A_2/B_2 follow in one rhythm with no *Save* press between them (`SR-BENCH-013`); the *saved*
    trait is still `length` with the real scale, so a bench line and a study length mean the same
    thing. *This view* is a scratchpad: leaving it unsaved discards the lines (the panel says so),
    and the buttons under the list save it to the **bench archive** or onto a **study**
    (`SR-BENCH-014`).
  * **Keypoints** - place the schema's anatomical keypoints (head, tail, sides); the canvas draws
    the schema's **bones** (`KeypointSystem.edges`) rather than a polyline through the click order.
  * **Box** - draw a new box on empty space **and** edit the boxes already on the frame (a press on
    a stored box selects it). Making a box and editing a box are one tool now.
  * **Select** - pick a box, a model's box or a point; drag moves/resizes, `Backspace`/`Delete`
    removes, **Shift/Cmd-click** or a **marquee** selects several and they then move and disappear
    together. A click never creates a label, and a double click does not walk the frame.
  The older `box_edit` and `box_mode` flags still work (the getter folds them into `tool`), so the
  surfaces that set them are unchanged; `point_mode` remains the crop exception (`SR-DATASET-028`).
  A **point** is still draggable on its own and removable, and a **model box** dragged is promoted
  to the draft and stored as an annotation on the drop (`SR-DATASET-027`).
* **Hover a keypoint, box or prediction and click it to edit it right away** (`X-N15`). The canvas
  answers *"what would the next press grab?"* in one place (`ImageCanvas._hit_editable`), reports it
  as `ImageCanvas.hover`, highlights it in the accent colour and shows a grab cursor. A press **on**
  that target edits it - move/resize/delete a box, drag a point, or pick a model's box into the
  draft - **in any tool**, so a small correction no longer costs a switch to *Select* and back. A
  press on **empty space** still runs the tool's own gesture, so measuring is unchanged.
* **A measurement stays editable after it is made** (`X-N16`). With `annotation_edit` on (Bench and
  Dataset turn it on) the *drawn* measurement - a Bench line, a Dataset keypoint pair - is grabbable
  by its **ends**: hovering shows the ring, and a press reports the exact measurement
  (`annotation_point_selected`: specimen, segment, endpoint), the page loads that row back (from the
  saved store on Dataset, from *this view* on Bench) into the draft, and the **same press drags the
  point**; letting go writes the correction, never a second measurement. Which measurement a segment
  is rides on an **opaque per-segment tag** passed to `set_annotations` - never guessed from the
  order lines happen to be drawn in.
* **A box is a measurement too** (`SR-BENCH-015`). With the *Box* tool a drag records a box with its
  **width, height and area** (px and µm/µm²) alongside the corners, and it is listed, moved, resized
  and removed exactly like a line - the same object the canvas already knows how to edit (`X-N15`).
  It is stored through the same `measurements` row (trait `bbox`, geometry in the
  `width_px/height_px/area_px` + `_um` columns, added by a `COLUMN_MIGRATIONS` entry, `X-N06`).
* **A box is an *individual*, and the numbers inside it are its own** (`SR-BENCH-016`, `X-N21`). The
  frame may hold several animals, so the results list is **grouped**: each box heads a group carrying
  its own size, and every measurement whose **midpoint** falls inside it is listed under it - so one
  animal's length *and* width *and* whatever else you measured on it read as one animal's numbers, and
  several boxes give several such groups. The rule lives in `daphnia/vision/individuals.py` (midpoint,
  **innermost** box wins, source pixels - one frame of reference, `mlContext.md` law 1) and is proved
  by `tests/test_individuals.py`. A measurement that falls inside **no** box is listed under *No
  individual* - kept and shown, never attached to the nearest animal (law 8). It is the concept the
  rest of the app already had (`observations.individual_index` on Study, one box per specimen in the
  corpus, `vision.tracking.Track` on Predict); the Bench was the one place it was missing. Which
  individual a row is about is **recorded** (`measurements.individual_index`) and travels in the CSV
  export, and it is deliberately **not** part of the row key - so one animal can hold two lengths
  without the second erasing the first. Accepting a detected animal's **box** makes that individual;
  accepting its **keypoints** makes a measurement inside it (the mmpose top-down split, on the Bench).
* **A predicted box or keypoint becomes a measurement when the interaction ends** (`SR-BENCH-015`).
  Pressing a model's box picks it and lets you **drag it where it belongs**; the measurement is made
  on **release**, from the geometry you actually dropped - so the dragged box is the one added and the
  untouched suggestion never lingers as a ghost (`prediction_accepted`). **What you click decides what
  you get**: a press **on the box** makes a **box** measurement, a press **on one of the model's
  keypoints** makes a **line**. A model's answer is **two independent objects** - the box RTMDet found
  and the keypoints RTMPose placed in it - so accepting one **keeps the other** (a box click leaves the
  keypoints to be measured as a line, and vice versa), and nothing else in the layer is touched
  (`drop_prediction(index, part=...)`, `mlContext.md` law 7).
* **Every object on the canvas is selectable on the same terms** (`X-N17`). A **box**, a
  **measurement** and a **prediction** are all objects: a press selects one, **Shift/Cmd** adds or
  removes, a **marquee** takes every object it covers, a selected object is drawn **selected** (a
  prediction used to disappear when picked - it now highlights), and **one `Backspace` removes
  everything selected**. The object you just made is already the selection, so `Backspace` deletes it
  without an extra click. In the **Box tool** you can draw a box *and* edit the boxes on the frame;
  the *Select* tool only selects - so it can touch a prediction without accepting it.
  **A Bench box is grabbed by its outline, not by its inside** (`box_edge_grab`): a box around an
  animal is a *measurement* with the keypoints/line drawn **inside** it, so a press deep in the box
  reaches those objects and only coming close to the box's line picks the box itself. (The corpus
  keeps interior picking - there a box *is* the animal.)
* **Every tool is one key away** (`X-N18`). The tool's menu text carries its key - `Line (L)`,
  `Keypoints (K)`, `Box (B)`, `Select (V)` - and the shortcut is bound at window scope, so the tool
  switches from anywhere without a trip to the menu bar (the rule an image editor uses). Pressing a
  key is exactly the menu click: it ticks the tool's action, and the radio group does the rest.
* **And a tool can be turned *off*: that is the normal view** (`X-N20`). Clicking the armed tool
  again - or pressing its key again - leaves **no tool armed**, and a press then simply **selects**
  what is under it (and drags it, resizes it, `Backspace`s it): nothing is drawn and no measurement
  is made, so you can look, pick and move without arming anything. It used to be impossible - the
  page put the click straight back on the tool it tried to leave, "a radio group with nothing
  selected would leave the press undefined" - and the *normal view* is not a *Select* entry in the
  menu: nothing-to-draw is the state you are in before you reach for a tool. While no tool is armed
  the tool-follows-the-object switch (`X-N19`) does not fire either, or the normal view would last
  exactly one click.
* **The model's answers are listed in the panel, and a click adds one** (`SR-BENCH-017`). After a
  pass, the results tree opens with *From the model* - one row per prediction, saying what the click
  is worth (`box 107 x 89 px, length 46 um   click to add it`) - and clicking a row adds that animal
  to *this view*: its **box** as the individual and its **keypoints** as a measurement inside it, the
  top-down pair RTMDet and RTMPose actually produced. The row leaves the list and the prediction the
  canvas layer in the same breath (no row, no dotted ghost), the rows renumber as the list drains,
  `Backspace` on the canvas takes its row too, and switching the model off empties the whole group.
  This is the fast route - no hunting for one animal among several overlapping boxes - and it is
  still a **human accept**, which is the only thing that may turn a prediction into data
  (`SR-PREDICT-N01`, `mlContext.md` law 7); nothing is stored until *Save*.
* **And the tool follows the object you click** (`X-N19`). A press on an object switches the tool to
  the one that owns it - a **box** (stored or predicted) arms the *Box* tool, a **measurement** arms
  the measuring tool, a model's **keypoints** arm the *Keypoints* tool - **in any mode**, and without
  disturbing the press: the selection, the drag and the points already placed stay, so one click both
  reaches the object and leaves the right tool in force for the next one. A promoted prediction is
  reported **once** (`prediction_accepted`), so acting on both that and a drawn box cannot make two.
* **Taking a live frame switches to image mode** (`X-N16`). *Use the live frame* (and opening a
  photo) hides the live preview so the frozen image fills the visual area - a moving video beside
  the picture you are measuring is noise and a source of mis-clicks. The camera keeps running, so
  *Use the live frame* still grabs the newest picture; the *Live preview* item in the *Source* menu
  brings the video back.
* **And every point, box and model box is editable** (`SR-DATASET-027`). A **point** can be
  dragged on its own and removed; a **model box** can be dragged - it is promoted to the draft
  and, on the drop, **stored as an annotation**, so a prediction you could not correct is now a
  prediction you can use; **several boxes** can be selected (Shift/Cmd-click, or a marquee
  dragged over them) and then **move and disappear together**, which is what a frame whose
  boxes all drifted by the same amount asks for. On Bench and Annotate a click picks *which
  animal* the model was asked about, putting that one's keypoints in the draft to correct.
* **Every window carries the same status bar** (`StatusFooter`): the camera state
  (*not connected / connecting / connected but silent / live*, with Connect) and the
  optics **scale in force** (its `µm/px`, uncertainty, age, and a RE-VERIFY flag when
  it may have moved). Both are facts about the machine that change what a number
  means, and two screens already said *"press Calibrate in the bar below"* - the bar
  is this one. In every window but Optics it offers **Calibrate...**, which hands over
  to the Optics surface: that window steps aside and comes back when calibration closes
  (`SurfaceWindow.open_calibration`), so there is still only one window on screen. The
  Optics window hides that one button - it *is* the calibration.
  (`tests/test_ui_smoke.py::WindowFooterTests` measures it per window.)
* **Full screen** (`workspace.button("full_screen")`, `Esc` leaves) hides the controls
  panel, the live preview and the window chrome - the header, the step bar and the footer
  in the Study shell - and goes full screen, so the image is the only thing on screen. The
  canvas keeps its points, its tools, its zoom and - where the page put them - its draw
  tools; nothing of the window's own is added to the picture.
* **Every window has a header: the menu bar and the facts that are not measurements**
  (`daphnia/ui/widgets/header_bar.py`, mounted with `QMainWindow.setMenuWidget`, so it
  *is* the menu-bar area - one top bar, not a menu bar above a strip). It carries a
  non-native `QMenuBar` (in the window on macOS too - a header in the system menu is a
  header nobody sees) and, beside the menus, the window facts - which file/frame, which
  model, the last message - as `LABEL + value` pairs whose long values **elide** (never
  wrap, so the bar stays one line) with the whole text in their tooltip. It is chrome, so
  full screen hides it. (`tests/test_ui_smoke.py::HeaderBarTests`.)
* **The menu bar is the application's own, in one fixed order** (`X-N13`;
  `daphnia/ui/widgets/header_bar.py`, `MENU_ORDER`). Every window reads:

  ```
  File | <the page's own menus> | Source | Models | View | Go | Help
  ```

  * **File** - what this window opens and what it exports (*Open an image...*,
    *Import a folder...*, *Open a video file...*, *Export the dataset*), then *Exit*
    (`Cmd/Ctrl+Q`). The page's file entries land here, above *Exit*.
  * **the page's own menus** - the work: *Measure* (Bench, Calibrate, Annotate),
    *Label* (Dataset), *Frames* (Predict). They sit directly after *File*, because a
    lab tool is read for its work first.
  * **Source** - the one shared camera (*Connect the live feed*, checkable: its check
    mark **is** the state, and it drives the same `StreamService` as the footer's
    Connect), then the page's own sources under a separator (*Use the live frame*,
    *Use the current frame*, *Open a video file...*).
  * **Detector (RTMDet)** and **Pose (RTMPose)** - one menu per model kind, titled with the
    network's own name (`daphnia/models.py`: RTMDet is the mmdetection-lineage detector,
    RTMPose the mmpose-lineage pose model). Each carries *Load the active …*, *Open a …
    checkpoint…* and, under a section header, **every checkpoint in the models folder**
    (`models.checkpoints_in`): a file the lab has just trained is one click away - loading it
    **registers and activates it** - without a file dialog. A **check mark** sits on the
    checkpoint **in force** (and on the top entry once this window has it loaded), so "which
    model answers" is visible where the loading happens, and **loading one switches the page's
    *Model* tool on** - asking for a model from the menu *is* the decision to use it. The
    registry is filled from the **training reports** (`make train` writes
    `models/rtmdet_report.json` / `rtmpose_report.json`), and the report's **best** file is the
    one that comes into force - never the final epoch (`mlContext.md` law 9).
  * **View** - *Zoom in*, *Zoom out*, *Fit the image*, *Pan (hand tool)*, *Full screen*,
    acting on the canvas of the page on screen, and **empty - so absent - on a step with
    no canvas** (Acquire, Select: a live preview and a gallery have nothing to zoom).
  * **Go** - Home and the other surfaces (deliberately absent from the study shell: the
    order there is `daphnia.workflow`'s job).
  * **Help** - *How to calibrate (grid paper)* (the walkthrough dialog every window can
    show) and *Where the data lives* (the database, recordings, datasets, models paths).

  The app's menus are built **once per window, empty, in that order**, so no window can
  invent a different order; the window fills *View*/*Go*/*Models*, the page fills *File*
  and *Source*, and `hide_empty_menus()` takes out any menu that stayed empty - a menu
  that opens onto nothing teaches the operator that the bar lies.
* **An action is a menu item; a choice stays beside its panel** (`X-N13`). The rule that
  moved the measuring tools off the image applies to the panel too: a *button that does
  something* is a menu item, a *combo with the button that applies it* is a choice and
  stays (Dataset's dataset + *New*/*Rename*, Bench's study + *Save this view to the study*,
  Setup's host/URL +
  *Connect*). So Calibrate lost *Use current frame*, *Open a photo...*, *Save
  calibration* and *How to set up calibration* (all four are in the bar), Bench lost
  *Open a photo...*, *Use live frame*, *Refresh list* and *Export bench
  results (CSV)*, Dataset lost seven (*Open an image...*, *Import a folder...*, *Open a
  boxes export...*, *Export dataset*, *Use the live frame*, *Next keypoint*, *Reload
  queue*), Annotate lost *New specimen* and *Reload queue*, Dashboard lost the project
  lifecycle (*New project...*, *Open this project*, *Close & consolidate...*, *Prune
  discarded frames*) and Visualize lost *Export measurements (CSV)*. A canvas-less step
  declares its menu items through `ShellPage.menu_tools()` - the same `ToolGroup` a
  canvas page's `workspace` exposes - so a step with no image to measure is in the bar
  too. What stays in a panel is the *workflow* step's own controls (Acquire's *Capture*,
  Select's *Pick sharpest*), because there the step bar gates them.
* **Nothing is drawn over the picture: the numbers are read in the panel and the facts in
  the header** (`workspace.set_header`, `daphnia/ui/widgets/canvas_tools.py`). There used
  to be a strip of KPI cards pinned to the canvas' top-left corner - one card per fact,
  the headline set larger, a caveat flag such as `WEAK`, `STALE`, `UNSCALED`, drawn from
  the theme's `PHOTO_OVERLAY` tokens. It is **gone** (2026-10-18): every fact it carried
  was already in the panel (the length being measured, the grid pitch, the pose median,
  the queue position, the model's own description of what it found) and the context was
  already in the header, so the picture keeps only what the picture *is*: the specimen,
  the boxes, the points, the lattice - plus the page's own draw tools in the corner
  (`X-N20`). A page states its facts as :class:`~daphnia.ui.widgets.Kpi` records and the
  window **header** reads them
  (`workspace.set_header`); a page with nothing to publish besides a message leaves the
  bar to the pages that do. (`tests/test_canvas_workspace.py::PaintTests` measures that a
  workspace adds no pixel to the canvas.)
* The stream preview and the image canvas share one look - both ask the **theme** for
  the canvas tokens and share the same minimum size (`PANEL_MINIMUM`), so the panel
  looks identical wherever it appears and follows the light/dark scheme with the rest
  of the window (see *Styling*).
* **An image canvas takes any source - one way in** (`ImageCanvas.set_source`): a
  numpy frame, a **file path**, a `QImage`/`QPixmap`, or `None` to clear. Five screens
  used to repeat the file reading, and the Annotate screen had a second, weaker loader
  (`QPixmap(path)`) that showed *nothing* when the file was unreadable. Now the canvas
  reads the file through `daphnia.vision.images.load_image` (the one loader that also
  reads what a phone writes: `.HEIC`), returns the frame it displayed - so a page that
  needs the pixels (a crop, a detection, a grid) does not load the file twice - and
  **refuses a source it cannot use by drawing the reason and emitting
  `source_failed`**, which each screen wires to its own status line with one line
  (`canvas.source_failed.connect(self.status_label.setText)`). Whatever arrives
  replaces what was there: the previous image, its points, box, overlays and grid go
  with it, and a refused source leaves the page's own frame state empty too, so **no
  number is ever left standing over a picture that is not on screen**. A source of the
  wrong *kind* raises `TypeError` - that is a programming error, not something to draw.
* The live preview is **ratio-driven**: it reads the stream's width/height from the
  incoming frames and derives its own height from its width (`heightForWidth`), so
  the image fills the preview exactly instead of being letterboxed.
* The Calibrate, Annotate, Bench, Dataset and Predict canvases are **zoomable** so
  landmarks can be placed precisely - but **zooming is deliberate and panning is
  cheap**. A **two-finger pinch** (macOS trackpad) zooms towards the cursor, `+` / `-`
  / `F` (or the *Zoom +/-* and *Fit* tools on the image) zoom in/out/fit, and a
  **two-finger scroll pans, in both axes, without ever changing the magnification** -
  sideways moves sideways, up/down moves up/down. (An accidental zoom while clicking
  is an error in the ground truth, not an inconvenience, so scroll no longer zooms at
  all - `⌘ + scroll` does not either.) **⌘/Ctrl + left drag** or the **Pan** tool pan
  without placing a landmark; right/middle-drag and the arrow keys pan too. Clicks stay
  mapped to the original image pixels at any magnification.
* In the Dataset queue a **double click on the image walks to the next frame** - and
  stores nothing: the gesture's own first press is *taken back* with the same undo a
  mistaken click gets, so a double click can never leave a bogus landmark behind.
* Every specimen is **drawn in its own colour with its number** (`specimen_color`,
  `ImageCanvas.set_annotations`) **from the moment a point is placed** - including
  during a landmark pass, so two heads are already distinguishable. Each measured
  interval is drawn as the line it is (width left→right, length head→spine base), and
  a point without a partner is a dot, never a line to another specimen. **Bench** has no
  mode to choose: the default **Line** tool names the ends **A** and **B** and the line
  joins *this view* the moment B lands (A_1/B_1, A_2/B_2 ... - no *Save* between them,
  `SR-BENCH-013`/`SR-BENCH-014`). **Dataset** labels in two **modes**: *two points per
  specimen* (an interval at a time) or
  *one landmark at a time* - click every head, then every tail, then the width sides -
  which stores each interval the moment both of its ends exist. In the landmark mode
  the **specimen counter and the landmark both advance themselves** (the first pass
  sets the specimen count; a completed pass moves on to `head -> spine base -> left
  side -> right side`).
* **A dataset is one corpus per model.** The Dataset surface starts by choosing (or
  creating) a **dataset**, and the dataset's **model** fixes what a label is
  (`daphnia/annotation.py`): *pose, head → tail* (2 keypoints), *pose, head, tail &
  sides* (4 keypoints, the default), either of those **plus a bounding box**, or
  *bounding boxes only* (a detector's target). So training another model means
  creating another dataset - labels are never re-purposed - and the same frame can
  carry ground truth for two models at once. The **dataset list shows each model**,
  `New dataset...` creates one, `Rename...` renames the open one (the model is locked
  there: the schema is what the existing labels *mean*), and a name that is taken is
  **numbered** (`corpus (2)`) instead of refused. A **detection-only** dataset hides
  the keypoint controls and labels **one box per specimen** - either by **dragging** a
  rectangle or by **clicking one corner and then the opposite one** (the anchored
  corner stays visible, `Esc` abandons it; a box under 3 px is treated as a mis-click
  and is never stored, and `B` toggles box drawing); a dataset without boxes hides the
  box control, and `B` says *why*
  instead of doing nothing. Whatever the model, only *its* traits are offered:
  a two-keypoint corpus cannot label a width. Selecting a specimen shows **the box it
  already has**, so a correction is a redraw of the rectangle on screen.
* **Boxes that already exist can be worked through one animal at a time.** `Open
  boxes export...` asks for the two halves of a box corpus - the **boxes file**
  (`boxes.csv` or `labels.json`, or the export folder itself) and the **folder the
  images are in** (suggested as the `images/` folder beside it, never guessed) - then
  queues every box as **one animal per item**. If the open dataset has no box in its
  model, you are offered the dataset that does, in one click.
* **In the crop of a keypoint corpus the press *is* the keypoint gesture**
  (`ImageCanvas.point_mode`, `SR-DATASET-028`): the item's box covers the whole
  magnification, so a click used to select **the box** - and the prompt's *"click the
  head, then the spine base"* did nothing. Now a press within reach of a point picks it
  up, and any other press places the next landmark - empty space is not a marquee there,
  because in that
  work the box is the animal. The box itself is moved in **Frame** view, where every box
  is visible, and a box dragged while a crop is on screen is stored **in the frame's own
  pixels and size** through the same `Crop` the clicks use (`source_box`): storing the
  crop's numbers wrote a box three times too small at the crop's offset.
* **Two views of one animal, switched with `C`** (the `Crop`/`Frame` tool): **Frame**
  shows the whole frame with **every** box drawn in its specimen's colour, the animal
  being labelled solid on top - which is the only way to see whether a box is really
  around an animal, or whether a neighbour was boxed twice; **Crop** shows that animal
  alone, magnified (`daphnia/vision/crop.py`: box plus a context margin, an integer
  factor, nearest-neighbour so the enlargement is exact). **Yes (`Y`)** keeps a box as
  ground truth (a `source='model'` prediction becomes `source='manual'`, and the note
  names the accept - a prediction is not ground truth until a person says so); **No
  (`N`)** removes it, so no model learns a rectangle a person rejected. A prediction is
  never left unanswered: the queue waits for the answer instead of moving on, and the
  queue is built from the **corpus**, so an animal the detector missed can be added by
  hand. Either way the keypoints are stored on the **original** frame - the view is only
  a view.
* **Each control is used for what it is, and each lives where it is used**: `Dataset`,
  `Mode` and `Specimen` are
  dropdowns (real choices), the current **keypoint/trait is displayed** with a **Next
  keypoint** button to step it (the first switch, e.g. all heads → the tails, is
  manual; the rest auto-advance), and the **actions are the tools on the image** -
  *Save* (hidden in landmark and box mode, which store automatically), *Undo*, *Clear
  points* and
  *Clear frame* sit in the lower-left cluster, next to the frame they act on. *Undo*
  removes the **last thing placed** - the loose point first, then
  the stored interval or box - so nothing you just did is unrecoverable. *Save*
  advances to
  the next specimen; *Clear frame* deletes every stored label/measurement/box of the
  frame
  **and** the open points, so a frame imported from a different study with a different
  specimen count can be re-done from scratch. Each dataset's labels stay its own:
  clearing one never touches another model's ground truth.
* Incoming frames are **rotated to landscape once, on arrival** and scaled with the
  aspect ratio preserved (no distortion, no cropping), so the live preview, the
  burst written to disk and the clicked keypoints share one coordinate space. Set
  `DAPHNIA_STREAM_ORIENTATION=native` to keep the phone's own portrait framing.

## Database migrations

`init_db()` runs `daphnia/db/schema.sql` and then patches columns that were added
after a database was first created (`COLUMN_MIGRATIONS` in
`daphnia/db/connection.py`) - `CREATE TABLE IF NOT EXISTS` never alters an
existing table. Migrated databases keep their data; `make db-reset` is only needed
if you want the pristine schema (e.g. to get the foreign key that ALTER TABLE
cannot add).

It also **backfills the generic landmark columns** (`p1_*`/`p2_*`) from the old
`eye_apex`/`spine_base` columns, **collapses duplicate measurements of the same
interval** left behind by earlier builds (keeping the newest) and then creates the
unique index that keeps the one-measurement-per-`(frame, specimen, trait)` rule true
from then on.

## Capturing and measuring

A **target** is either a **well of a plate** or **one individual** of the simple
counter, and the acquisition step picks between them:

* Setup defines the plates - **6 (2x3) / 24 (4x6) / 96 (8x12)**. `daphnia/plates.py`
  turns a format into well labels (`A01 ... H12`), so the geometry is one line of
  configuration and the treatment builder can sit on top of it later.
* Acquire iterates the wells of a plate, or counts individuals 1, 2, 3 ... . Every
  **burst** is appended to that target's observation, so a specimen is re-visited
  and improved instead of being overwritten.
* Frames are written to ``<study destination>/<session>/<observation>/b<burst>_f<frame>.png``
  and shown in the **strip below the stream**: double-click any frame for a
  full-screen look, then decide whether another burst is needed. The sharpest
  frame of the whole target is kept as ``is_primary``.

**Select** and **Annotate** are two separate steps on purpose:

1. **Select** - every burst of a target is on the table at once and exactly **one
   frame per well / individual is kept**: ticking a frame sends the others of that
   target to the tray, so each well yields exactly one measurement. *Pick sharpest
   frame* does it automatically from the stored Laplacian score, *Clear selection*
   resets the target, and double-click opens any frame full screen.
2. **Annotate** - the carousel walks every kept frame of the study. A photo can
   hold **several specimens**, and each specimen is measured for **more than one
   trait**, so the panel has a *Specimen* spinner, a *Trait* selector (*Length* =
   head -> spine base, *Width* = the two side points) and a live list of the
   measurements already on the frame. Click the two landmarks, *Save measurement*,
   *New specimen* for the next animal. Each measurement is tied to the exact
   `asset_id` it was drawn on; those landmarks are the ground truth the prediction
   model will be trained on. Re-saving the **same `(frame, specimen, trait)`**
   *updates* it - `idx_measurement_asset_trait` keeps exactly one such row, so the
   table cannot be inflated.

## Calibration rule (grid paper, auto-detected or typed in)

The phone is fixed to the eyepiece by the mount, so the scale changes only when
the **zoom**, the **focus** or the **objective** changes - or when the adapter is
knocked. It is therefore re-measured whenever that could have happened, and the
target is a single, well-known *kind* of target: **printed grid paper**.

The surface is three panels, and the name comes first. **1 Optical chain** is a
read-only **list of facts** (`Preset`, `Setup key`, `In force`, `Calibrations`, `Mount`,
`Distortion`) with two `[Load...] [New]` button pairs above and below it - one pair for
the device, one for its scale: one name is one optical chain (`PROF_<name>`), so nothing
about the microscope, the camera, the objective or the zoom is typed in twice, and the
name is only ever entered in a popup (`Load preset...` picks an existing device by name,
`New preset...` names a fresh one) - the panel shows it, read-only. `In force` carries the
scale with `RE-VERIFY` past 12 h (the chain's own key is its tooltip), `Mount` is the
locked zoom and focus, and `Distortion` is the one fact this chain does **not** correct:
`unmeasured and not corrected - one scale covers the whole frame. Optional: it only
matters off-centre.` Neither `Load preset...` nor `New preset...` measures or stores: they
only choose, and `Load preset...` asks first when the scale would change by more than 2x.
**2 Target photo** -
the frame the scale is measured from (`Use live frame`, or `Open photo...`), how big one
printed square is (1 mm by default, 0.5 mm or 0.1 mm for finer optics), and the
**result**: the period with the bands it rests on, the **drift** across the field, the
paper's circular field stop, and a one-line **verdict**. Below a rule, as a clearly
separated **fallback**, the hand-clicked span: a second route to the same *one* scale, and
coarser. **3 Scale in force** - the
number, its uncertainty *together with the route it came from*, and what that implies
for a real animal. `Store this scale` - on the panel and in the *Calibration* menu - is
the conclusion of the step.

1. Put the grid paper under the objective, at the height the specimen will be.
2. Set **Printed square** to the paper you actually have: the period is measured in
   pixels, and this number is what turns it into a scale.
3. Press **Use live frame** if the phone is streaming, or **Open photo...** and point it
   at any grid photo you already have - including an iPhone `.HEIC`, which is converted
   automatically, and a 12 MP original, which is fine.
4. Press **Measure grid**. The grid is a periodic pattern, so
   `daphnia/vision/lattice.py` finds the paper's circular field stop, fills the surround,
   straightens the tilted lattice and reads the period off narrow bands - giving
   `px_per_um` (shown as `S` in px/mm) with no clicking at all. What it measured is
   **drawn over the photo**: the lattice in amber *and* the circle it was measured
   inside, so the evidence can be seen rather than trusted. The verdict line then says
   whether the result is `MEASURED - the scale holds across the whole field: the bands agree
   to within the print's own tolerance` or `WEAK - the scale is right where it was measured,
   and less right towards the edge. Keep the specimen centred; see panel 1 for the distortion
   that is not corrected`, and a frame with no lattice
   in it is `REFUSED` with the reason in the line's tooltip. The detector **refuses**
   anything that is not a square grid, so it can never invent a scale. Changing the square
   size afterwards re-scales the same detection: no second photo.
5. Or tick **I know the scale already** and type the scale in px/um, together with the
   accuracy you state for it. That is stored as *your claim* - no proof image and no
   measured uncertainty - and the panel says so rather than dressing it up.
6. Press **Store this scale**: a **dated calibration** - the scale, when it was verified,
   and its accuracy - which becomes the active one.

Panel 1 is arranged the way a **device** and its **calibrations** both work: you *load*
one, or start a *new* one, each with its own `[Load...] [New]` pair - the preset pair at
the head of the panel, the calibration pair below the fact list. `Load preset...` lists
one name per device (no scale or date columns: the name is the whole label) and makes that
chain's newest scale the one in force; `New preset...` names a fresh chain in a popup.
`Load calibration...` lists **this device's** stored calibrations - the operator's own
`New calibration...` note when they gave one, the date otherwise, plus the scale - and
loading one both **puts it in force** and shows **the photo it was measured from** (the
only way to judge a stored scale from the outside: was the paper flat, was one square
really the size the record claims); `New calibration...` asks for a note, clears the
measurement, and the next *Measure grid* + *Store this scale* writes a fresh calibration
of the same device carrying that note. A calibration you typed in has no proof photo, so
the panel says that instead of showing an unrelated image, and a proof photo that has since
been deleted is named with the file that is missing. Both `Load...` buttons ask first when
the scale would change by more than 2x.

Three numbers come out of it, and each answers a different question:

* **± combines the fit with the paper.** The period is located to well under a
  percent, but printed 1 mm spacing is not a length standard (~0.5 %), so that
  tolerance is included rather than quoting a falsely precise figure. The panel says
  which route produced the number, because only the grid route includes the print's
  half percent - the clicked span is a measurement of the *photograph*, and it says
  so. Verify the sheet once against a stage micrometer and the grid becomes
  trustworthy. A hand-entered scale has no fit to combine: it carries only the accuracy
  you stated for it.
* **`total +/-` is the whole budget of the length**, and it is the honest one: the
  `±` above is the scale's own relative error and nothing else, while a measured line
  also carries *where the two landmarks were clicked* and *the pixel grid they had to
  land on*. Every measuring screen shows both, side by side - on the eyepiece chain
  (0.027264 px/µm, `±0.2934 %`) a 2.314 mm animal reads
  `2314.0 +/- 6.8 um (2.314 mm)   total +/- 75.2 um (3.2 %)` - because there the click
  term alone is 73.4 µm, i.e. the scale's `±` is **11× smaller than the budget** it sits
  beside. The same figure is stored with the result (`corpus_labels.total_uncertainty_um`)
  and travels in every export, so a number outlives the screen that made it. What it
  deliberately *excludes* is in `docs/accuracy.md` §6 - the pose model's own error (a
  separate, offline number), the repeat spread (shown as `mean ±SE (n=…)` on Predict) and
  the field's drift.
* **Drift, as its own number.** The same bands that produced the period also say how
  much the scale moves from the centre to the edge of the field, and it is reported
  **separately** from both uncertainties: the uncertainty is how well the centre was
  measured, the drift is how much the optics change it elsewhere. Measured on this rig:
  **5.1 %**, i.e. 114 µm on a 2 mm animal - seven times the print's own tolerance. That
  is what the `WEAK` verdict is about, and it is why specimens stay central.
* **Age.** A knock changes nothing that is recorded, so the app states how old the
  calibration is and flags `RE-VERIFY` past `DAPHNIA_CALIBRATION_MAX_AGE_HOURS`.

**Every one of these terms has a formula and a worth on this rig, and they are collected in
`docs/accuracy.md` (the error budget):** the scale's uncertainty, the drift, the print's own
tolerance, the pixel quantisation, the model's own error and the repeat spread — what each is
worth in micrometres, how they combine, and which of them the app reports and which it still
does not.

Every later photo taken under that calibration is converted with the stored scale
directly. No container edges are used and the animal only has to be roughly
centred in the field of view. `measurements.px_per_um` and
`measurements.uncertainty_pct` record what was actually applied, so recalibrating
later can neither reinterpret old data nor silently change how precise it claims
to be. Every calibration is its own immutable record, grouped by `setup_key`.

### Where the scale is *not* constant (and what "unmeasured" means)

One scale is one number, so it is true at the place it was measured - and **both routes to
it (the detected grid and the hand-clicked span) only ever set that one scale; neither of
them corrects the lens.** That is why the distortion is a fact of **panel 1** (the optical
chain) and not a result of the measurement in panel 2, and why it is stated as *optional*:
it only matters away from the centre of the field. The panel says
`Distortion: unmeasured and not corrected - one scale covers the whole frame. Optional: it
only matters off-centre.`, and panel 2's **drift** line says how much the scale actually
moves towards the edge on the photo at hand.

Measured on this project's own `recordings/calibration`
photos (2026-10-19), **the detector refuses all nine of them** - a phone photo taken
through the eyepiece is low-contrast, edge-darkened and tilted, and the detector's
prominence gate is what fails. So today the honest statement is *"the drift is
unmeasured"*, **not** *"the drift is 0 %"*.

`scripts/calibration_probe.py` measures that question instead of assuming an answer:

```bash
python scripts/calibration_probe.py recordings/calibration/<photo>.png
```

It runs the shipped detector on the photo and on five 200-px windows, then runs
**discorpy** (Diamond Light Source, Apache-2.0, an optional pip dependency) through
its own line-pattern flow - and it runs an **oracle** first: a synthetic
single-weight line grid with a known distortion, so "the method works" is proven
before a real photo is blamed. On the oracle the flow recovers the centre of
distortion to ~1 px and unwraps to **0.26 px** residual; on the graph-paper photo it
finds both line families (19.17 / 18.16 px spacing, 8.95 deg) but the model does
**not** reach discorpy's own sub-pixel bar, because printed graph paper carries
three line weights (1/5/10 mm) while the flow needs one evenly spaced family. The
script prints that verdict instead of a number, and `docs/calibration.md` records
what to print to fix it.

`scripts/discorpy_probe.py` is the companion that runs **discorpy's own line-pattern
example, unmodified**, on the same photo — no masking, no equalisation, no alignment —
so the two scripts differ in one axis and the comparison says whether the photo or the
library is the limit (`docs/calibration.md` §4.1). `scripts/discorpy_doc_probe.py` runs
the flow the discorpy **documentation** prescribes for strong distortion plus perspective
(§4.1's fisheye pipeline: normalise first, the docs' mask, the polyfit grouping, the 1.7
vanishing-point centre of distortion, a colour padded unwarp) — and the measured answer,
next to the two other probes, is in `docs/calibration.md` §4.2.

## Acquisition protocols, templates and data lifecycle

Three small domain modules configure a run and tidy it up afterwards:

* **`daphnia.protocols`** — named acquisition presets (`Quick scan`, `Standard`,
  `Careful`) that bundle the burst size and a QA sharpness hint with the trade-off
  each one implies. A study stores the protocol it was run with
  (`studies.protocol_key`) and Acquire defaults its burst size from it, so *how the
  frames were taken* is recorded right next to the scale.
* **`daphnia.templates`** — reusable starting points for a study, **versioned by
  name**: saving under an existing name appends a version instead of overwriting, so
  the earlier ones stay as history. Load one on Setup to fill the whole form.
* **`daphnia.maintenance`** — the graceful end of a run: prune the discarded,
  un-measured frames, export the measurements as CSV and write a `manifest.json`
  next to the images, then close the study (Overview → *Close & consolidate*). A
  standalone *Export measurements (CSV)* button lives on the Analytics screen.

## Database (SQLite)

Schema in `daphnia/db/schema.sql`, bootstrapped idempotently by
`daphnia.db.init_db()` on every launch. Decoupled profile calibration:

```python
from daphnia.db import connection, init_db

init_db()                    # creates zoo.db + tables; called at startup
with connection() as conn:   # foreign keys ON, dict-like rows, auto-closed
    rows = conn.execute("SELECT * FROM calibration_profiles").fetchall()
```

| Table | Role |
| ----- | ---- |
| `calibration_profiles` | One row per **calibration**: the scale measured on an optical chain at a point in time, grouped by `setup_key`, with its `uncertainty_pct`. |
| `calibration_records` | Proof images / reference measurements — stage micrometer or calibration paper (`calculated_px_per_um` is a generated column). |
| `imaging_sessions` | One run per plate; locks the `profile_id`, holds `current_well_index` for *Resume Run*. |
| `study_templates` | Versioned starting points for a study (`name` → v1, v2…); only the latest version of each name is offered. |

Still to design: treatment assignment on top of `plates` (the plate builder).

## Live streaming (MediaMTX)

```bash
brew install mediamtx              # macOS; on Windows see "Run it on another machine"
./scripts/setup_mediamtx.sh        # installs config/mediamtx.yml (macOS/Homebrew)
brew services start mediamtx       # or: make mediamtx-run
```

The phone publishes SRT to `srt://<mac-lan-ip>:8890?streamid=publish:live/cam` (use
the free **Moblin** iOS app, SRT mode); MediaMTX republishes the same stream over
RTSP, which OpenCV reads as `rtsp://127.0.0.1:8554/live/cam`. MediaMTX is a single
dependency-free executable on **Windows as well as macOS** — the Windows route (download
the release, run `mediamtx.exe config\mediamtx.yml`, open UDP 8890 / TCP 8554) is in
[Run it on another machine](#run-it-on-another-machine-macos-or-windows). Full details,
URLs, environment variables and troubleshooting: **[docs/streaming.md](docs/streaming.md)**.

## Styling

The app has **one theme system**: `daphnia/ui/theme.py` (the tokens, the palette,
which scheme is in force) and `assets/theme/daphniascope.qss` (the template). Nothing
else decides a colour — no screen sets a colour inline, and a widget-level
`setStyleSheet` is a theme bug (`tests/test_theme.py` fails on one).

Three parts, and each answers a failure the previous setup had:

| Part | What it is | The failure it answers |
| --- | --- | --- |
| **Tokens** (`LIGHT` / `DARK` in `theme.py`) | the only place a colour is written down; both schemes define the same keys | the old stylesheet painted light colours onto whatever palette the OS supplied |
| **Palette** (`build_palette`) | the colours of everything the stylesheet does *not* mention: group boxes, tables, lists, text browsers, message boxes, menus, tool tips | those widgets followed the *system* while the styled ones did not — dark boxes, invisible text |
| **Template** (`daphniascope.qss`) | structure (padding, radii, hairlines, `objectName` blocks) and the few colours that are a decision: the primary action, the step states, the translucent tools over a photograph | no `url(...)` icons and no indicator rules, so the check mark and the combo arrow are drawn by the `Fusion` style instead of pointing at Adwaita files that only exist on Linux |

**The scheme follows the desktop** (macOS light/dark) and switches when the desktop
does. Pin it when you need to:

```bash
DAPHNIA_THEME=dark python run.py       # light | dark | system (the default)
```

An unrecognised value is an *error* that names itself, never a silent fall back to the
other theme. From Python: `from daphnia.ui import theme; theme.set_scheme("dark")`.

> Qt implements only a CSS2 *subset* (QSS) — no layout properties, no CSS variables,
> no classes — and `QColor` cannot read the `rgba(...)` values, so colour is split
> deliberately: **the palette carries it, the template shapes it.** To restyle, change
> a token (both schemes) and not a rule; to restructure, edit the template. The
> canvas' *overlay* colours (`PHOTO_OVERLAY`) are identical in both schemes on purpose:
> those panels lie on a photograph, not on a window.

The old single-file theme (`assets/theme/MacOS.qss`, GTRONICK) is **gone** — see
`git log -- assets/theme/` if you ever need to read it; keeping it would leave two
files that both look authoritative.

> Qt gotcha: in widget text `&` is a mnemonic marker. A literal ampersand in a
> `QPushButton` label must be written `&&` (in a `.ui` file: `&amp;&amp;`).
> `QLabel` renders a single `&` as-is.

## Packaging: an `.exe`, a `.dmg`, or conda-pack?

**Status: none of these is built today.** There is no `.spec` file, no `make package`
and no CI job; neither route below has been run for this project. What follows is what
the code and the tooling *actually* support, and what each route would cost — so the
decision can be made without a surprise.

### The rule that decides everything: PyInstaller cannot cross-build

> *"The output of PyInstaller is specific to the active operating system and the active
> version of Python. This means that to prepare a distribution for: a different OS, a
> different version of Python, a 32-bit or 64-bit OS, you run PyInstaller on that OS,
> under that version of Python."* — [PyInstaller, *What PyInstaller Does*](https://pyinstaller.org/en/stable/operating-mode.html)

So a `.exe` is produced **on a Windows machine**, and a `.app`/`.dmg` **on a Mac**;
neither can be produced from the other. That is also true of `conda-pack`. There are
therefore two supported ways to get this app onto a colleague's machine, and they answer
different questions:

| Route | What it produces | Keeps | Costs |
| ----- | ---------------- | ----- | ----- |
| **`conda-pack`** (recommended for the lab) | a tarball of the *exact* `z-app` environment, unpacked and activated on the target | the environment the test suite was measured in — same Python, PySide6, OpenCV and PyTorch builds | a clone of the repository still has to travel with it; it is not a double-clickable icon |
| **PyInstaller** | a self-contained `.app` (then a `.dmg`) or a folder/`.exe` | nothing but the app: no conda, no Python on the target | one build per OS *and* per CPU architecture; the bundle is large (see torch below); a frozen app is harder to debug than a conda environment |

### conda-pack, in full

```bash
conda install -c conda-forge conda-pack     # or: pip install conda-pack
conda-pack -n z-app -o z-app.tar.gz         # run this on the *target* OS/architecture
```

On the target machine: copy the repository **and** the tarball, then

```bash
mkdir -p ~/z-app && tar -xzf z-app.tar.gz -C ~/z-app
source ~/z-app/bin/activate                 # macOS/Linux
# Windows:  .\z-app\Scripts\activate.bat
cd app && python run.py
```

`conda-pack` needs the target's OS and architecture to match the machine that packed it
(it is not a cross-packer either), and `conda-unpack` (run by the activate script) fixes
the hard-coded prefix. Nothing else is required: `environment.yml` has the whole
dependency list.

### PyInstaller: what this app needs from a bundle

PyInstaller knows PySide6 out of the box — *"PyInstaller also knows about many major
Python packages, including the GUI packages Qt (imported via PyQt or PySide)"* — so the
Qt plugins and the interpreter are collected for you. **Three things are read from disk
at run time and must be added by hand**, because they are not importable modules:

| What | Read by | Why it must be bundled |
| ---- | ------- | ---------------------- |
| `assets/ui/*.ui` (14 files) | `daphnia/ui/loader.load_ui` → `QUiLoader` + `QFile` | every screen is a `.ui` file loaded *after* start-up; without it, `FileNotFoundError: Could not open UI file` |
| `assets/theme/daphniascope.qss` | `daphnia/ui/theme.build_stylesheet` | the theme template; without it the app paints unnatively or not at all |
| `daphnia/db/schema.sql` | `daphnia/db/connection.init_db` → `paths.SCHEMA_PATH` | the database is created from this file on first launch |

```bash
# on Windows (; is the --add-data separator)
pyinstaller --noconfirm --windowed --name DaphniaScope ^
  --add-data "assets;assets" --add-data "daphnia/db/schema.sql;daphnia/db" run.py

# on macOS (: is the separator)
pyinstaller --noconfirm --windowed --name DaphniaScope \
  --osx-bundle-identifier ch.zhaw.daphniascope \
  --add-data "assets:assets" --add-data "daphnia/db/schema.sql:daphnia/db" run.py
```

Then point the app's state outside the bundle, and hand over the database:

```bash
export DAPHNIA_DATA_DIR=~/DaphniaScope          # %LOCALAPPDATA%\DaphniaScope on Windows
```

Without it the app writes `zoo.db`, `recordings/`, `datasets/` and `models/` *inside* its
own folder — which on macOS breaks the code signature (so Gatekeeper can refuse to launch
it afterwards) and on Windows means Program Files. With it, put the checkpoints where the
app now looks for them (`<DAPHNIA_DATA_DIR>/models/*.pt`); a missing one is named by the
Predict surface rather than silently ignored.

### Three costs to decide about, not to discover

1. **PyTorch decides the bundle size.** `pytorch` is in `environment.yml` and is imported
   *lazily* by `daphnia/vision/rtmdet` and `rtmpose`: the app starts, measures and labels
   without it, and only Predict reports `ModelUnavailable` (`SR-PREDICT-N02`). So a
   **Bench + Dataset + Study build can leave torch out**, while a **Predict-capable
   build** needs `--collect-all torch` plus its native libraries — and `--onefile` then
   unpacks that to a temporary folder on every start. Neither size is measured here;
   measure the bundle before quoting one.
2. **MPS is Apple-only.** On Windows the detector and the pose model run on CPU, so
   inference is slower there. The *accuracy* is unaffected (same checkpoint, same
   corpus), but **latency is unmeasured on Windows** and must be measured on the target
   machine before anyone promises a live frame rate (`python scripts/check_torch.py`,
   `python scripts/train_rtmdet.py --evaluate`). This is also why `train.pick_device`
   refuses a device that is not present instead of falling back to one.
3. **Signing.** An unsigned `.app`/`.dmg` is blocked by Gatekeeper ("unidentified
   developer"; the workaround is right-click → Open, or
   `xattr -dr com.apple.quarantine /Applications/DaphniaScope.app`) and an unsigned
   `.exe` raises SmartScreen. Launching on a colleague's laptop with no warning at all
   needs an Apple Developer ID + notarisation (paid) and a Windows code-signing
   certificate. Related: conda-forge builds PySide6 and PyTorch **per architecture**, so
   an arm64 `.app` does not run on an Intel Mac (and vice versa) — there is no universal
   binary here.

### Making a `.dmg` from the `.app`

A `.dmg` is only a container around the `.app` PyInstaller produced, so it is the last
step and the cheapest one:

```bash
pyinstaller …                       # → dist/DaphniaScope.app   (built on macOS)
hdiutil create -volname DaphniaScope -srcfolder dist/DaphniaScope.app \
        -ov -format UDZO dist/DaphniaScope.dmg
```

The app must still be *given* its `DAPHNIA_DATA_DIR`: a mounted disk image is read-only,
so a bundle that writes inside itself cannot work from a `.dmg` at all.

## The detector (RTMDet)

The **Dataset** surface produces a labelled corpus; **`python run.py predict`** shows
what a trained detector does with it. Between them sits a trainer that reads *only*
the exported folder:

```bash
python scripts/train_rtmdet.py --inspect        # what is in datasets/bb, and what is wrong with it
make train                                      # train (MPS if this Mac has it)
python run.py predict                           # live feed, a video file, or one image
```

* RTMDet's architecture — CSPNeXt backbone, CSPNeXtPAFPN neck, a head sharing its
  convolutions across levels with a separate BatchNorm per level, one anchor per
  location, **SimOTA** assignment, quality focal + GIoU losses — in plain PyTorch, so
  it trains on the machine that takes the pictures. No `mmcv`/`mmdet`.
* Trains and predicts on **MPS** (Apple-silicon GPU) or CPU; `--device` chooses, and
  asking for MPS where there is none is an error rather than a silent slow run.
* Validity first: the default split holds out whole **capture sessions**
  (`--split-by group`), because the export's own split-by-image lets sibling frames
  leak into validation. The run prints the corpus's warnings and says plainly that its
  AP50 is *its own* IoU-0.5 score, **not COCO mAP**.
* A **video file** is read frame by frame: *Pause*, then `←` / `→` (or the *< Frame*
  and *Frame >* tools on the image) move exactly **one frame** - forwards or backwards,
  pausing first so nothing races the reader - and the panel says which frame is on screen.
  Those controls exist **only** while a video file is
  open (`SR-PREDICT-011`): with the live feed or a single image they are gone, and the
  arrow keys pan the canvas as they do on every other screen.
* The Predict surface never stores a prediction: a prediction is not ground truth
  (`SR-PREDICT-N01`). Pre-labels for the Dataset surface are the next step.

Full guide — install, the split, what the metrics mean, and exactly where this differs
from the reference implementation: **[docs/rtmdet.md](docs/rtmdet.md)**.

## Development

* Architecture, layers and conventions: **[docs/architecture.md](docs/architecture.md)**
* Per-surface system requirements: **[docs/requirements/](docs/requirements/)** —
  numbered, traced to the thesis objectives `O1..O4` / deliverables `D1..D3`.
* Agent context (project brief, patterns, status): **[memory-bank/](memory-bank/)**
* Settings are environment-driven (`DAPHNIA_*`), see `daphnia/settings.py`.
* Python 3.13 / PySide6 6.11 / OpenCV (headless) come from `environment.yml`.
* Tests use the standard library: `make test`.

### Editing the screens in Qt Designer

**Yes** — the screens are plain Qt Designer files, and the editor that matches this
runtime ships with the environment (`z-app` → `Designer6`, PySide6 6.11):

```bash
make ui-designer FILE=bench.ui      # or: python scripts/edit_ui.py bench.ui
make ui-list                        # what every screen declares, and what would break
```

What Designer shows, and what it cannot:

* it shows the **controls panel** — everything a screen declares in its `.ui` file.
  That is the whole right-hand side of a screen, and nothing else;
* it does **not** show the **visual area**: the live preview, the image canvas and the
  on-canvas tool clusters are built in code (`ShellPage.add_visual`,
  `canvas_workspace`) and added at runtime, so a screen opened here looks like a form
  with empty space beside it. That space is where the picture goes;
* it does not show the shell's post-load work either: `MainWindow` wraps
  `stackedWidget` in a scroll area and mounts the status footer + step bar into
  `pages.ui`'s `footer_container`.

Three rules, all enforced by `tests/test_ui_files.py` (and checked by
`make ui-list` before the editor opens, so a mistake is caught before you edit):

1. **a widget's `objectName` is the API.** `load_ui` exposes every named widget as an
   attribute of the page (`self.combo_dataset`), and the pages *and the tests* use
   those names. Renaming a control in Designer is a code change - and so is *deleting*
   one: the controls that moved into the header's menus (`X-N13`) had to leave the pages'
   bindings with them (`self.btn_open_photo` -> the *File* menu's *Open a photo...*);
2. **one name, one widget.** Two widgets sharing a name means `load_ui` picks one —
   silently. This was a real bug: `annotate.ui` had two `btn_new_specimen` buttons, so
   the one inside the Subject group was never wired and never disabled - the second one
   is gone now, and the action is the *Measure* menu's *Next specimen*;
3. **plain Qt widgets, and no `.qrc`.** A *promoted* widget arrives as a bare
   `QWidget` (QUiLoader cannot create a Python class) and a resource URL needs a
   compiled resource file the app deliberately does not have.

After editing: `make ui-check` (every window, both theme schemes, full screen and back)
and `make test`.
