# SR-BENCH — quick-measurement requirements

Serves: **O4** (standalone desktop app), **D3**. Launched with
`python run.py bench`, or the *Bench* tile in Home.

Audience: colleagues who want *a number now*, without a study. Before this surface
existed that was impossible: measuring required study -> connection -> calibration ->
acquire -> select -> annotate, and there was no way to open an image from disk.

## Gate policy

Two things must hold to measure: **an image** and **a scale** — nothing else. No
study, no plate, no phone, no selection step. The scale is the one requirement that
cannot be waived, because a number without a unit is not a length: with no
calibration the readout is in pixels and labelled **unscaled**, and only a scaled
measurement can be stored.

## Functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-BENCH-001` | The surface **shall** open an image **from disk**, in any format the machine can decode and at any size. | The single biggest missing capability; without it colleagues cannot use the app at all. | `T` `tests/test_bench.py`, `tests/test_images.py` | implemented | O4 |
| `SR-BENCH-002` | The surface **shall** take the **live frame** as an alternative input. | Same measurement, no file needed. | `T` `tests/test_bench.py` | implemented | O4 |
| `SR-BENCH-003` | The surface **shall** accept exactly two landmarks per specimen, labelled with the vocabulary the study pipeline uses (default head -> spine base). | A bench number and a study number must mean the same thing. | `T` `tests/test_bench.py` | implemented | O2, O4 |
| `SR-BENCH-004` | The surface **shall** show the measured distance live, in micrometres, as soon as two points exist. | Immediate feedback is the point of the surface. | `T` `tests/test_bench.py` | implemented | O4 |
| `SR-BENCH-005` | The surface **shall** allow several specimens on one image, each with its own index and trait, and **shall** let the operator say **which** specimen a number belongs to by drawing the **box** that stands for it (`SR-BENCH-016`). | A well or a field may hold several animals, and a number that cannot say which animal it is about cannot be compared with anything. | `T` `tests/test_bench.py::BenchIndividualTests` | implemented | O2, O4 |
| `SR-BENCH-006` | The surface **shall** use the **calibration in force**; with none it **shall** report pixels only and label the result **unscaled**. | Honest fallback instead of a fabricated micron value. Picking among several presets belongs to `SR-OPTICS-006`. | `T` `tests/test_bench.py` | implemented (single scale) | O3, O4 |
| `SR-BENCH-007` | The surface **shall** persist results with `study_id = NULL` into the shared store, with image path, scale, uncertainty, operator and timestamp. | Results must be findable and defendable, not screenshots. | `T` `tests/test_bench.py` | implemented | O4 |
| `SR-BENCH-008` | The surface **shall** allow a study-less measurement to be **promoted** onto a study without re-measuring it. | Turns ad-hoc work into study data instead of discarding it. | `T` `tests/test_bench.py` | implemented | O1, O4 |
| `SR-BENCH-009` | The surface **shall** show, in its results area, **only what is in the current view** - the measurements made since the Bench was opened - and **shall not** list the measurements of previous bench sessions there; what was saved to the dataset is read through the CSV export and the Dataset surface (`SR-BENCH-014`). | Listing every previous session buried the thing being measured now, and a list of rows the operator did not just make is not feedback. | `T` `tests/test_bench.py::BenchPageTests` | implemented | O4 |
| `SR-BENCH-010` | The surface **shall** allow the results to be exported as CSV **including the landmark coordinates**. | Ad-hoc results must leave the app intact, and a number without its points cannot be checked. | `T` `tests/test_bench.py` | implemented | D1 |
| `SR-BENCH-011` | Re-measuring the same `(image, specimen, trait)` **shall** update that row and **shall never** duplicate it. | Reuses the CORE invariant; keeps the bench list honest. | `T` `tests/test_bench.py` | implemented | O2 |
| `SR-BENCH-012` | A live frame **shall** be written to disk only when a measurement is **stored**. | A quick look must leave nothing behind. | `T` `tests/test_bench.py` | implemented | O4 |
| `SR-BENCH-013` | The surface **shall** open in a **Line** tool: the two endpoints are **A** and **B** (not an anatomy), and the line is **stored and numbered the moment B lands** - so a measurement costs its **two clicks** and several lines on one frame follow each other (A_1/B_1, A_2/B_2 ...) with **no *Save* press between them**. The surface's draw tools **shall** be exactly two - **Line** and **Box** - in one mutually-exclusive group (`X-N14`), and it **shall** offer **no keypoint tool**: placing an anatomy's landmarks one at a time is a **corpus** workflow and belongs to the Dataset surface, not to a screen that displays measurements (`SR-BENCH-017`). | The Bench measures *a distance*, so forcing the ends to be called "head" and "spine base" named an anatomy the operator did not choose; and a single pair with a *Save* between each measurement added a press to every number. | `T` `tests/test_bench.py::BenchPageTests` | implemented | O4, X-N14 |
| `SR-BENCH-014` | A measurement made on the Bench **shall** live in **the view of the image it was drawn on** until it is saved (`SR-BENCH-022`), and **shall** be lost only when the surface is left unsaved, with the panel **saying so** ("if you leave the Bench without saving, these measurements are gone"). The surface **shall** offer two saving actions - **save this view to the dataset** (the corpus open at the top of the panel) and **save this view to a study** (the same rows, in the study’s own dataset, attached to the chosen study in one action) - **shall** offer no predefined *mode* or *specimen* control (the tools decide the gesture, and lines are numbered automatically), **shall** allow a view line to be **corrected before saving** (`X-N16`), and **shall** refuse to save without a scale. | A scratchpad that quietly keeps everything is a scratchpad whose history nobody can trust, and one that silently drops work is worse; stating the rule and offering *where to save it* is what makes "make a measurement now" safe. Saving straight onto a study is the point of a quick measurement that belongs to a project. | `T` `tests/test_bench.py::BenchPageTests` | implemented | O1, O4, X-N16 |
| `SR-BENCH-015` | A **box** **shall** be a Bench measurement like a line: drawn with the *Box* tool, it **shall** record **width, height and area** (in px and µm/µm²) and the two opposite corners, **shall** be listed, edited (moved/resized) and removed (`Backspace`) exactly like a line, and **shall** be saved to the same dataset/store (trait `bbox`, its geometry in the `measurements` width/height/area columns - `X-N06`). A model's prediction **shall** become a measurement when the **interaction ends**, built from the geometry the operator actually dropped (never the untouched suggestion, so no ghost remains), and **what was aimed at `shall` decide the kind**: a press **on the predicted box** makes a **box** (even when the prediction also carries keypoints), a press **on one of its keypoints** makes a **line**. The prediction **shall** then leave the layer, and `Backspace` **shall** remove any selected object (box, line, prediction or point). A box **shall** *also* be the **individual** it stands for (`SR-BENCH-016`): its own row records its index as the individual, and its size is the fact the group it heads carries. | The models answer with boxes as often as with keypoints, and "how big is this animal" is a real bench question (area, not only length); a predicted box the operator accepts should not need re-drawing, and it must be **the box that was dragged** that is added. One rule for every object - click to select, drag to edit, Backspace to remove - is what makes the canvas predictable (`X-N15`, `X-N17`). | `T` `tests/test_bench.py::BenchBoxTests`, `tests/test_individuals.py` | implemented | O2, O4, X-N06 |
| `SR-BENCH-016` | A Bench **box** **shall** stand for one **individual**, and every measurement whose **midpoint** falls inside it **shall** belong to it - the **innermost** box winning when several contain it, in source-image pixels of the frame on screen. The results list **shall** be **grouped by individual**, each group headed by the individual and carrying the box's own size, with the measurements inside it listed under it; a measurement that falls inside **no** box **shall** be listed under *No individual*, kept and shown rather than attached to the nearest animal. Which individual a number belongs to **shall** be recorded with the row (`measurements.individual_index`, `NULL` for none) and **shall** travel in the CSV export, and it **shall not** be part of the row key, so one individual may hold two measurements of the same trait without the second erasing the first. | "A box should be an animal, not a box measurement: I want to see when a line or keypoint falls inside a drawn box and list every measurement inside it, so I can read more than a length - and do it for several boxes and see a list per individual." Two lines on one animal were two unrelated rows, and the only specimen identity the surface had was the row's position in its own list. The concept already existed three times in this repository (`observations.individual_index`, `corpus_boxes`' one box per specimen, `vision.tracking.Track`); the Bench was the one place it was missing. | `T` `tests/test_individuals.py`, `tests/test_bench.py::BenchIndividualTests` | implemented | O2, O4, X-N06 |

| ~~`SR-BENCH-017`~~ | **Removed 2026-10-22.** The Bench used to carry the *Model* switch and list the models' answers as clickable rows. **The surface displays measurements and runs no model:** a checkpoint is loaded and watched on the **Predict** surface (`SR-PREDICT-018`), which is what that surface is *for*, and a second, quieter way to run one - with no readout of what it was doing - is not a convenience but an ambiguity. The switch, the prediction rows, the dotted layer and the page's `adopt_models` are gone; the window carries no *Detector (RTMDet)* / *Pose (RTMPose)* menus either (`hub.BenchWindow.OFFERS_MODELS = False`). The canvas' own prediction layer, and the gesture that turns a picked prediction into a measurement, remain (`X-N15`, `test_image_canvas.py::AnnotationEditingTests`) - they are simply not reachable from this surface. | A model is a tool the operator reaches for, and reaching for it belongs on the screen built for judging it (Precision/recall on the frames you can see). Two ways to start a model, one of them invisible in the readout, is how "the numbers changed and nobody knows why" happens. | `T` `tests/test_bench.py::BenchPageTests::test_the_surface_displays_measurements_and_offers_no_model`, `tests/test_ui_smoke.py::HeaderBarTests::test_the_bench_window_offers_no_model_menu` | **removed** | O3, O4 |

| `SR-BENCH-018` | The surface **shall** work **inside a dataset** - the same store every other entry point reads - and **shall** let the operator **create a new dataset** (name + model) and **choose among the existing ones** from the top of the panel. A fresh database **shall** get a default dataset, and a saved view **shall** be stored in the open dataset (so it is visible in the Dataset labeller and the export). | There is no "bench archive": one set of storing conventions is what makes work done here reachable everywhere else. The selector is the Dataset surface's own control, so the two entry points cannot drift. | `T` `tests/test_bench.py::BenchDatasetTests` | implemented | O4, D1 |
| `SR-BENCH-019` | The surface **shall** import **every image under a folder** (recursively) as a walkable queue, and **shall** record the import (its source, and one row per frame) so re-importing the same folder is the same work. | Measuring a folder of frames one file dialog at a time is the bottleneck the surface exists to remove. | `T` `tests/test_bench.py::BenchImportTests` | implemented | O4, D1 |
| `SR-BENCH-020` | The surface **shall** import a **video file** by writing its frames to disk and queueing them, each frame keeping its number in the clip, and **shall** record that number so a frame traces back to the video it came from. | A video is not a file you can measure - its frames are - and a frame whose clip and number are unknown cannot be checked. | `T` `tests/test_bench.py::BenchImportTests` | implemented | O4, D1 |
| `SR-BENCH-021` | An imported queue **shall** be walked with **A/D**, the arrow keys and a **double click**, and the position **shall** be remembered on the import, so a session resumes where it stopped. | Labelling at volume is keyboard work; a queue that restarts at frame 1 is not resumable. | `T` `tests/test_bench.py::BenchImportTests` | implemented | O4 |
| `SR-BENCH-022` | A **view shall belong to an image**: switching to another frame **shall** switch to that frame's view, and returning to a frame **shall** show its measurements again - on the picture (`_refresh_overlays`) as well as in the list. The image on screen **shall** be named in the panel, and the model's answers **shall** be dropped when the frame changes (a prediction belongs to the frame it was computed on). | "When switching back to an image the measurements disappear again" - the view was one global scratchpad, so walking a queue silently destroyed work and the picture and the list disagreed. A measurement is *of* an image; the view must say so. | `T` `tests/test_bench.py::BenchPageTests`, `BenchImportTests` | implemented | O4, X-N17 |
| `SR-BENCH-023` | The surface **shall** take a **burst** of live frames and write them into a **folder of its own** under `datasets/raw/`, then open them through the **same path a folder import takes** - so the same folder can be brought back later with *Import a folder...* and gives the same queue, the same per-image views and the same store. Each frame of a burst **shall** keep its own view. | "Take a burst ... saves it to dataset folder so we can directly *import folder* and have same abilities." A burst is several images, not one; giving it a folder means the capture is not a special case, it is just frames. | `T` `tests/test_bench.py::BenchImportTests` | implemented | O4, D1 |
| `SR-BENCH-024` | Every measurement **shall** carry **its own label on the picture**, drawn **above the line it belongs to** (`ImageCanvas.set_annotations`' per-segment labels): the operator's **name** for it when they gave one - typed into that row's *Name* cell, and stored in `corpus_labels.label_name` - else the number the panel lists it under (`1.1`, `0.1`) together with its value. A name **shall** be saved with the row and carried by the Bench CSV, and emptying the cell **shall** put the number back. The panel's number and value columns **shall not** be editable. A measurement that belongs to no individual **shall** be drawn and numbered (`0.1` ...) but **shall not** be captioned *no individual* over the image: the grouping is read in the panel (`SR-BENCH-016`), the picture shows the line. | One animal's two numbers used to be collected into one sentence at its first point, so the second number could not be found on the picture at all; and a lab says "gut length", not "1.1". The words *no individual* written over a photograph answered a question nobody had asked, in the one place the measurement itself had to be read. | `T` `tests/test_bench.py::BenchIndividualTests`, `BenchNameTests`, `tests/test_image_canvas.py::AnnotationTests` | implemented | O4 |

## Non-functional


| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-BENCH-N01` | The surface **shall** reach a reading in at most **two** operator inputs after the image is loaded. | Directly counters the 14-input cascade that produced 12 throwaway studies. | `D` | implemented | O4 |
| `SR-BENCH-N02` | The surface **shall** start without any study, connection or plate existing. | Removes the entry barrier entirely. | `T` `tests/test_bench.py` | implemented | O4 |
| `SR-BENCH-N03` | The surface **shall** mark study-less and unscaled results as such wherever they appear. | Prevents ad-hoc numbers being mistaken for study results. | `T` `tests/test_bench.py` | implemented | O3 |
| `SR-BENCH-N04` | The surface **shall** reuse `ImageCanvas` and `ShellPage` rather than introducing new visual widgets. | Keeps one look and one click-mapping implementation. | `I` | implemented | D3 |
| `SR-BENCH-N05` | No control on the surface **shall** be presented disabled without saying why (see `X-N10`). | The calibrate button looked broken for exactly this reason. | `T` `tests/test_bench.py` | implemented | O4 |

## Still to build

- **`SR-BENCH-006`'s preset picker** — the surface uses the calibration *in force*
  rather than offering a choice among presets. That choice is meaningless until the
  active scale is resolvable per optical chain (`SR-OPTICS-006`, open).
- **Attaching a whole bench session** to a study in one action, rather than one image
  at a time.
- **Reading a saved view back** for a further pass over the same image. *Pre-labelling
  on the Bench is not on this list any more* (`SR-BENCH-017`, removed 2026-10-22):
  running a model belongs to Predict, and a bench click that starts from a prediction
  is a bench number nobody watched being made.
- **Resuming an import across launches** — the pointer is recorded
  (`import_batches.last_ordinal`), but the surface still opens the queue at the first
  frame; a "resume the last import" entry is the next step.

## Notes on the design

- **No predefined mode, and only two tools**: the canvas' own **tools**
  (`ImageCanvas.tool`, `X-N14`) decide what a press means, and the Bench opens in **Line** -
  click A, click B, and the line is a measurement; *Box* draws the individual a number
  belongs to. There is **no keypoint tool** and **no model switch** (`SR-BENCH-017`,
  removed): the surface displays measurements, one-keypoint-at-a-time labelling is a
  **corpus** workflow (Dataset), and running a checkpoint is Predict's
  (`SR-PREDICT-018`). *Select* is not a tool either - touching an object is always
  available (`X-N20`) - and either tool is one key (`L`/`B`) away, or off, which is the
  **normal view**.
- **Every measurement is labelled where it is measured** (`SR-BENCH-024`). Each line wears
  its own label *above its middle*: the **name** the operator typed for it (the panel's
  *Name* cell, stored in `corpus_labels.label_name` and exported with the row), else the
  number the panel lists it under - `1.1`, `0.1` - and its value. One sentence per animal at
  its first point is gone: on a frame holding an animal with two numbers, the second number
  had no place to be read. Nothing is captioned over the picture about *no individual*: a
  stray line is drawn and numbered, and the grouping is read in the panel
  (`SR-BENCH-016`). **Scrolling pans** (both axes) and never zooms - the pinch, the
  *Zoom +/-* tools or `+`/`-`/`F` zoom - and `Command`/`Control`-drag pans without dropping
  a point.
- **This view is a scratchpad** (`SR-BENCH-014`): the measurements made since the Bench was
  opened, listed in the panel with the warning that they are **not saved**. Loading another
  image (or taking a new live frame) **discards** them and says so; the two buttons under the
  list write them to the **dataset open at the top of the panel** or straight onto a **study**
  (the same rows, in the study's own dataset, with the photo attached in one action). A line
  already made is still editable before it is saved - hover it and drag its end (`X-N16`).
- **There is no separate "bench archive"** (`SR-BENCH-018`): the surface works in a
  **dataset**, the same store the Dataset labeller and a study write, so a bench number is
  visible from every entry point and the CSV export reads one store. *New dataset...* and
  *Rename...* are the Dataset surface's own controls, so the two entry points cannot drift.
- **A folder, a video or a burst becomes a queue** (`SR-BENCH-019`, `-020`, `-021`,
  `-023`): importing a folder (recursively), a clip (whose frames are written to
  `datasets/raw/<stem>/` and keep their number in the file name) or a **burst** of live
  frames (written to `datasets/raw/burst_<stamp>/`) fills the queue, which is walked with
  `A`/`D`, the arrows or a double click. A burst takes the **same path a folder import
  takes**, so its folder can be reopened with *Import a folder...* and gives the same
  queue. The import is recorded (`import_batches`/`imported_frames`), so re-importing the
  same source is the same work and a session is resumable.
- **A view belongs to an image** (`SR-BENCH-022`): stepping to another frame switches to
  *that* frame's view, and stepping back shows its measurements again - on the picture as
  well as in the list. The frame on screen is named in the panel, and the model's answers
  are dropped when the frame changes, because a prediction belongs to the frame it was
  computed on.
- **The page's own actions are menus in the header** (`X-N13`): the *Tool* group (what a
  press means), a *Measure* menu (*Undo*, *Clear points*, *Clear this view*, *Refresh*) and
  the *File* / *Source* entries, with nothing of it drawn over the photograph - the one
  thing on the image is the **draw-tool bar** in its corner (`X-N20`). The numbers are read
  in the panel (the length, and this view's list) and the picture's name and the last
  message are header facts, so **Full screen** (`X-N12`, `Esc` leaves) is a complete
  working state rather than a view that hides half the controls.
- **The panel keeps the choices and the view**: the *study* to save into, the list of what is
  in this view, and the message. *Saving* is a button beside that list, because it acts on
  the list.
- **The camera keeps running in image mode**: taking a live frame (or opening a photo) hides
  the preview so the frozen picture fills the area (`X-N16`); *Use the live frame* still
  grabs the newest picture, and the *Live preview* item brings the video back.
- **Study-less sessions are per image**, deliberately: a shared session would drag every
  other bench image into a study when one is promoted (`repository.bench_observation`).
- **The image is identified by its file path**, so re-opening a photo continues the
  same observation — which is what makes `SR-BENCH-011` true rather than aspirational.
- **Unscaled results are never stored.** `measurements.length_um` is `NOT NULL` for a
  reason: storing a pixel count in a micrometre column would be a fabrication. The
  readout says `UNSCALED`, and saving explains that a calibration is one press away.
