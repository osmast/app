# Progress — status, known issues, decision evolution

_Last updated: 2026-10-22_

## What works

- **The total uncertainty of a length is a number, not a formula in a document (2026-10-22,
  twenty-eighth round).** `units.total_uncertainty_um(L, S, δ, *, extra_um, click_error_px)`
  combines the four terms `docs/accuracy.md` §1 had only *derived*: the scale's own `±`, where
  the two landmarks were **clicked** (`2c/S`, a constant in µm), the **pixel grid** they landed
  on (`0.408/S`) and anything a caller measured elsewhere (the pose model's error, a repeat's
  spread). `format_total_uncertainty` renders it as `total +/- 75.1 um (3.8 %)`, and **every
  measuring readout prints it beside the `±`** — Bench, Dataset, Annotate/Study, and the
  Calibrate panel's example line — because on the eyepiece chain the click term alone is 73.4 µm
  against a 6.8 µm `±`, i.e. the number on screen was **11× smaller than the budget**. It is
  **stored** (`corpus_labels.total_uncertainty_um`, written by the one write path from the same
  function the screens print) and **exported** (`labels.csv`, `labels.json`'s rows *and* its
  per-specimen `traits` block, the Bench CSV, the corpus export; `CORPUS_COLUMNS`/`BENCH_COLUMNS`
  gained `uncertainty_pct` too). The live store was migrated in place: **519 of 550 labels had
  recorded *which* scale applied and not how well it was measured**, so they adopt their
  calibration's δ by reference (immutable record) and all **550** got a budget — per chain
  152.8–178.8 / 203.6–238.3 / 74.9–77.3 µm, both steps reported by name in `migrate()`.
  `docs/accuracy.md` §1/§2/§3/§4/§6/§7/§8 were corrected with it (a document that says "nothing
  combines the terms" must not survive the change that combines them), and the document's two
  worked cases are now the tests: **75.1 µm** (Study path) and **226.3 µm** (Predict path with
  `click_error_px=0.0` — nobody clicked). Tests: `tests/test_units.py::TotalUncertaintyTests`
  (12), 3 in `tests/test_calibrations.py` (profile, stored, migrated), 1 in `tests/test_dataset.py`
  (exported). **Deliberately not added:** the pose model's term on any surface (Predict still
  names it separately), the drift (S4), the age (S10), the annotator's variance.
- **The app can be run on another machine, and the packaging question has an answer (2026-10-22).**
  The README now carries `## Run it on another machine (macOS or Windows)` and
  `## Packaging: an .exe, a .dmg, or conda-pack?`. Two premises were checked rather than assumed:
  **MediaMTX runs on Windows** (a single dependency-free executable; the release ships
  `mediamtx_v1.21.2_windows_amd64.zip`) — what is macOS-shaped here is only `setup_mediamtx.sh`,
  the informational `SYSTEM_CONFIG` path, the two shell scripts, `make`, and `sips` (already
  guarded); and **PyInstaller cannot cross-build** (*"you run PyInstaller on that OS"*), so a
  `.exe` needs a Windows machine and a `.dmg` a Mac, one build per architecture. To make a bundle
  possible at all, `paths.py` gained **`DAPHNIA_DATA_DIR`**: everything the app writes (`zoo.db`,
  `recordings/`, `datasets/`, `models/`) now hangs off one root, where before only the database
  could be moved and a frozen app would have written inside its own read-only bundle (on macOS,
  breaking the signature). A bundle must also carry `assets/ui/*.ui` (14), `assets/theme/daphniascope.qss`
  and `daphnia/db/schema.sql`, which are read from disk at run time — the `--add-data` lines for
  both OSes are in the README. Tests: `tests/test_paths.py` (4, new). **Nothing is built** — no
  `.spec`, no measured bundle size — and the README says so.

- **The measurement error budget is one document (2026-10-22).** `docs/accuracy.md` —
  *"Accuracy: the error budget of one length"* — carries **fourteen numbered sources (S1–S14)**,
  each with its formula, its worth on this rig in µm, and how the app handles it, plus the
  combination (§3), the handling policy (§4), worked examples (§5) and the commands that
  reproduce every number (§7). What it establishes: the `±` on screen is the **scale term alone**
  and the true budget is **13× larger on the Study path** (the human click dominates) and
  **23× on the Predict path** (the pose model dominates); the pixel quantisation (S5) and the
  measurement's own click (S6) were **derivable and unshown** — *the round below built them* —
  and **nothing checks plausibility (S14** — see known issue 27). Linked from `docs/calibration.md`
  and the README. No code changed in that round.

- **Study 2 captures a burst *or* a recording whose length is the operator's (2026-10-22).**
  *Start recording* opens a take on the well the camera is over and *Stop recording and
  store* writes it as **one burst** of that well (`SR-STUDY2-013`): every frame that arrives
  is written to the well's own folder as it comes (`b<burst>_f<index>.png`), the sharpest of
  them becomes the well's image, and a stopped recording is read by Study, Dataset and the
  Bench exactly as a burst is - no schema change, no second storage path
  (`SR-STUDY2-N03`). While it runs the page is **locked to its well** (plate, study, page
  size, the walk, *Re-read the plate* and the grid: `SR-STUDY2-014`) and the line under the
  button counts frames and seconds. It ends at *Stop*, at `DAPHNIA_RECORD_MAX_FRAMES`
  (300 ≈ 10 s ≈ 360 MB), at a dropped stream, or when the page goes off screen - and it says
  **which** of those it was (`SR-STUDY2-N06`); a take that stored nothing is reported as not
  stored (`SR-STUDY2-N02`). Measured, because the design rests on it: **17.2 ms to write +
  2.5 ms to score** one 1280x720 frame, **1241 kB** each → ~**51 frames/s** of headroom
  against the 30 fps this stream delivers (M4, offscreen, 2026-10-22). Tests:
  `tests/test_study2.py::RecordingTests` (14), driven through `service.start()` over a
  scripted reader. Suite green: **1045 tests, 1 skipped**.
- **Panel 1 is a list, and a device *and* a scale are each loaded or started (2026-10-20, the
  operator's follow-up).** The Calibrate screen's **1 Optical chain** is now a read-only
  `label - value` list (`Preset · Setup key · In force · Calibrations · Mount · Distortion`) with
  **two `[Load...] [New]` pairs**: `Load preset...` makes a stored device's **newest** scale the
  one in force everywhere (`Calibration.load_preset`, `repository.activate_profile`,
  `SR-OPTICS-006`); `Load calibration...` picks one of *this device's* calibrations and both puts
  it in force **and** shows its proof photo; `New calibration...` asks for a **label** (stored in
  the new `calibration_profiles.label`) and starts a fresh measurement of the same device. The
  operator's name is entered **only in a popup** and shown read-only, so the popup/panel
  duplication is gone. Before this, the *only* way the active row changed was **storing** a new
  scale, so a microscope and a phone rig each had a scale but could never swap without measuring
  again. Nothing is measured or stored by a switch, and the `>2×` guard (`SR-OPTICS-008`) now
  covers storing **and** both loads. **The two pairs share no state:** `_pending_label` is cleared
  by `new_preset`, `load_preset` and `load_calibration`, so a note armed by `New calibration...`
  cannot be written onto another device's row (the bug the operator reported as *"load and new
  does not work together"*). `make ui-check` green; run green below.
- **The Calibrate surface is the daily act now (2026-10-20, S1 of
  `docs/ui-calibrate-redesign.md`).** One photo → one press → one result → one stored record,
  arranged as the doctrine says (`SR-OPTICS-024`): **1 Optical chain** (name, setup key, the
  scale in force as one of three cases with `RE-VERIFY`, the mount footnote, every earlier
  calibration with its route and proof photo) · **2 Target photo** (`Use live frame` /
  `Open photo...`, printed square size, `Measure grid`, then the result block: `Period: 27.239
  px/cell · 10 bands · tilt -9.0°`, `Drift: ± 5.1 % across the field (centre → edge)`,
  `Field stop: paper circle (601, 266) r 448 px · 54 % of the frame`, a **verdict**, and
  `lens distortion: unmeasured`) · **3 Scale in force** (S, the uncertainty **with its route**,
  the example length, `Store this scale`, the hand-entry row). Measured by rendering the real
  window: the verdict on this rig's photo is **`WEAK - the field moves more than the print`**
  (5.1 % of drift against the print's 0.5 %), the drawn evidence is the amber lattice **and**
  the paper's circle, and the header carries `PHOTO … / STATE …` instead of repeating panel 1.
  The `X-N13`-vs-`SR-OPTICS-024` conflict is resolved as "both" (panel button **and** menu item,
  one method behind each) and recorded in `systemPatterns.md §The calibrate surface`. Nothing in
  the pixel path, the schema or the store changed. Run green: `tests.test_calibration` (60),
  `test_image_canvas` (152), `test_ui_smoke` (60), `test_theme` (33), `test_surfaces` (23),
  `test_ui_files` (9) — 337 tests, plus `make ui-check` (12 canvases, both schemes) and
  `make ui-render` (looked at). **The whole suite was not run this round.**
- **The Calibrate screen measures the scale again - on the photos this rig actually
  produces (2026-10-19).** `daphnia/vision/lattice.py` replaced `vision/grid.py` as the
  OPTICS estimator (`SR-OPTICS-035/036`): it masks the eyepiece's circular field stop as a
  **fitted circle** (radius 448.7 px on the project's photo - a fit that includes the
  frame-clipped top answers 60 px small and eats real paper there), fills the surround,
  straightens the lattice by a projection search that is its own sign check, and reads the
  period off **narrow bands** whose spread is the distortion. Measured: **27.239 px** per
  cell on the eyepiece photo (10 bands, tilt -9.00°, spread **5.10 %**, statistical 2.49 %),
  **27.03/27.24 px** on three more photos of the same setup, and a **named refusal** for the
  two ruler photos and three unusable grid photos. The panel now states the drift as a
  number separate from the uncertainty, which is what `SR-OPTICS-036` asks for and what the
  old single number hid. **`make test` 1031 tests, 1 skipped, green** (14 new in
  `tests/test_lattice.py`, including "the shipped detector refuses this synthetic eyepiece
  photo and this one measures it" and "a mirror does not change the scale").
- **The distortion *model* is still not storable** - see `docs/calibration.md` and issues
  22-25: the documented discorpy flow reaches 0.511 px rms and passes its own sub-pixel
  check with 3 coefficients, but the model **fails the mirror test** (an exact pixel
  permutation; coefficients differ by 0.82, the centre lands 184 px off), and its magnitude
  is ~1.5-2x the drift measured directly from the cell spacing. The *effect* (4-6 % across
  the field) may be reported; the coefficients may not be stored.
- **The distortion *under* the scale is measured-or-stated, never assumed (2026-10-19).**
  `scripts/calibration_probe.py` runs the shipped detector and **discorpy** (Apache-2.0,
  optional pip dependency, lazily imported) on one calibration photo - and runs an
  **oracle** first: a synthetic single-weight line grid with a *known* distortion, so the
  method is proven before any photo is judged. Measured: the shipped detector **refuses
  all nine** photos in `recordings/calibration`; discorpy finds both line families on the
  eyepiece photo (**19.168 / 18.160 px**, **-8.95 deg**) and recovers a known distortion
  from one image to **1.13 px** centre error / **0.264 px rms** residual (sub-pixel) on
  the synthetic grid, but only **7.87 px rms** on printed graph paper - whose three line
  weights (1/5/10 mm) break the flow's one-uniform-family assumption. So the tool is
  sound and the *target* is the limit: `docs/calibration.md` records the evidence, the
  cost of each of the three ways forward, and what must hold in any of them
  (`SR-OPTICS-032/033/034`). Nothing in the app changed; the probe prints a verdict, and
  it does not print coefficients from a model that failed its own sub-pixel check.

- **One store, one dataset convention, no "archive"** (2026-10-18; asked for as *"i would like to
  merge dataset entry point with the bench entry point ... in bench we are missing: folder and video
  import, ability to create new dataset and choose from existing"*, then *"remove term 'archive'; all
  pages must use same dataset storing conventions"* and *"unify everything, Study included"*;
  `SR-BENCH-018`…`-021`). `measurements` is no longer a table - it is a **read-only view** over
  `corpus_labels`, which is now the **only** place a measurement lives, whichever surface wrote it.
  `repository.save_corpus_label` is the one write path; `save_measurement` (the study path) is a thin
  adapter that resolves the frame, the dataset and the governed provenance and delegates. `datasets`
  gained `study_id` (a study owns its measurement corpus); `corpus_boxes` gained `profile_id`/
  `px_per_um`/`observation_id`/`asset_id`; `corpus_labels` gained the observation/asset provenance,
  `individual_index`, the scale's uncertainty and the box geometry, and its image size became
  **nullable** (a frame not on disk has no size, and NULL says so). The migration is non-destructive
  and **measured on a copy of the live `zoo.db`: 28 measurements folded, 519 → 547 corpus labels, 0
  lost**, with `measurements_legacy` kept as a backup and a second `migrate()` applying nothing. The
  Bench gained the **dataset selector** (+ *New dataset...*/*Rename...*, the Dataset surface's own
  dialog), **Import a folder...**, **Import a video...** (frames materialised by the new
  `daphnia/vision/video.py`) and a walkable queue, with every import **recorded** in
  `import_batches`/`imported_frames`. `vision/images.header_size()` reads a PNG/JPEG size with **no
  OpenCV**, so the migration and the queue keep the app startable without the CV stack. Tests: 2 in
  `test_db`, `BenchDatasetTests` (3), `BenchImportTests` (4). Full suite **981 tests, 1 skipped, green**.

- **A tool can be turned *off*, and the model's answers are listed in the panel** (2026-10-18; asked for as
  *"i want also to be able to disable a tool not just switch between ... we are in normal view where we can
  select objects ... list predictions in measurments panel and be able to click on them to make
  measurements ... faster make predictions and add as measurments"*; `X-N20`, `SR-BENCH-017`). Unticking the
  armed draw tool - by clicking it again or pressing its key again - leaves **no tool armed**: the **normal
  view**, where a press selects, drags, resizes or removes and **draws nothing** (the canvas is given the
  *Select* tool's own behaviour, `BENCH_NORMAL_VIEW`), and where `X-N19`'s tool-follows-the-object switch does
  not fire. The results tree opens with *From the model* - one row per prediction saying what the click is
  worth (`box 107 x 89 px, length 46 um   click to add it`) - and a click adds that animal to *this view*: its
  box as the individual and its keypoints as a measurement inside it. The row and the layer entry go together,
  the rows renumber, `Backspace` takes a row away, and switching the model off empties the group. Fixed on the
  way: `ImageCanvas.drop_prediction(part="all")` could not remove a **box-only** prediction (early return on
  the absent keypoints - no page had ever asked for `part="all"` before), and the individual's number is now
  **derived** from the box order (1..N) instead of the row slot, which had made the individuals read 1, 3, 5.
  Tests: `tests/test_bench.py::BenchPredictionListTests` (8), `BenchPageTests` (+4), `BenchIndividualTests`
  (+2), `tests/test_image_canvas.py::ToolTests` (+2). Measured end to end with the real models on a real frame.

- **A box is an individual, and the measurements inside it are its own** (2026-10-18; asked for as
  *"the box itself is an 'animal' or individual ... detect when line or keypoint fall inside a drawn box and
  list all measurements inside that box ... a list of all measurements per individual"*; `SR-BENCH-016`,
  `X-N21`). The Bench's results list is now **grouped**: each box heads a group carrying its own
  width/height/area, and every measurement whose **midpoint** falls inside it is listed under it
  (`daphnia/vision/individuals.py`; innermost box wins; source pixels; `None` when no box contains it, shown
  as *No individual* rather than attached to the nearest animal). A box is **still stored** as its `bbox`
  row, so *how big is this animal* is answered too. Which individual a row is about is **recorded** in a new
  `measurements.individual_index` column and travels in the CSV export - and it is deliberately **not** part
  of the row key, so one animal may hold two lengths without the second erasing the first (the silent
  overwrite this design avoided). Tests: `tests/test_individuals.py` (19, hand-computed),
  `tests/test_bench.py::BenchIndividualTests` (9) and two storage tests. Measured end to end on a real frame
  with the real models: 5 detections → 2 individuals, and a pose answer stored as
  `length specimen=3 individual=1`.

- **The tool follows the object you click** (2026-10-18; asked for as *"when clicking on an object to
  switch to that object mode ... in selection mode and selecting an object or in another mode and one
  clicks on another object"*; `X-N19`). A press on an object switches the page's tool to the one that
  owns it - box (stored or predicted) → *Box*, measurement → the measuring tool, model keypoints →
  *Keypoints* - in any mode, **without disturbing the press** (`set_tool(..., keep=True)` keeps the
  selection/drag/points; `object_pressed` tells the page). A promoted prediction is reported **once**
  (`prediction_accepted`), so a page acting on both that and `box_changed` cannot make two - `Dataset`
  now stores its promoted label in `_on_prediction_accepted`. Tests:
  `tests/test_image_canvas.py::ToolTests` (+5), `tests/test_bench.py::BenchBoxTests` (+2). **921 tests,
  1 skipped**; `make ui-check` green.

- **Predictions are two objects, boxes are grabbed by their line, tools are one key away** (2026-10-18;
  asked for as *"clicking on one prediction box ... the keypoints disappear ... if we click on keypoint,
  the box disappears but also all other keypoint predictions ... treated all individually and not
  connected ... box should not have click-with feature but one has to come close to the line ... when
  in box mode we can't switch quickly to line or point mode"*; `X-N17`, `X-N18`). A model's answer
  (RTMDet's box + RTMPose's keypoints) is now **two independent objects**: accepting the box keeps the
  keypoints and vice versa (`drop_prediction(index, part=...)`), and removing one object never removes
  another (the canvas no longer double-drops - the page drops exactly what it used). A **Bench box is
  grabbed by its outline** (`box_edge_grab`), so a press inside it reaches the objects drawn there; the
  corpus keeps interior picking. Every tool is **one key** away (`Line (L)`, `Keypoints (K)`, `Box (B)`,
  `Select (V)`), the key shown in the menu text and bound at window scope. Tests:
  `tests/test_bench.py::BenchPageTests` (keys), `BenchBoxTests` (independent halves, outline grab);
  `tests/test_image_canvas.py::ToolTests`/`EditAllTests`. **914 tests, 1 skipped**; `make ui-check`
  green.

- **One selection, and a prediction is made where you drop it** (2026-10-18; asked for as *"not able
  to select multiple objects ... select predictions ... not able to delete boxes on backspace ...
  clicking on prediction box ... leaves behind a ghost ... the dragged-around gets added ... if click
  box only box to be added"*; `X-N15`, `X-N17`, `SR-BENCH-015`). The canvas now treats **boxes,
  measurements and predictions** as objects on the same terms: a press selects one, Shift/Cmd adds or
  removes, a **marquee** covers all three, a selected prediction is **highlighted (never hidden)**, and
  **one `Backspace` removes everything selected**. A model's prediction becomes a measurement when the
  interaction **ends**, from the geometry that was dropped (`prediction_accepted`) - so the dragged box
  is the one added and no ghost survives - and **what was aimed at decides the kind**: a press on the
  box makes a box (even with pose keypoints present), a press on a keypoint makes a line. Bench handles
  `box_delete_requested` now, and a freshly made object is left selected so `Backspace` removes it.
  Tests: `tests/test_bench.py::BenchBoxTests` (prediction box/keypoint/drag/delete), 
  `tests/test_image_canvas.py::ToolTests` (multi-select, marquee, prediction highlight). **910 tests, 1
  skipped**; `make ui-check` green.

- **A box is a measurement, and a clicked prediction becomes one** (2026-10-18; asked for as *"we
  would like to be able to measure boxes (area and height width) ... edit boxes similar to line ...
  as soon as a predicted box or keypoint gets clicked on, the object turns into measurement ... delete
  any object or measurement or predicted by using backspace"*; `SR-BENCH-015`). A Box-tool drag is a
  measurement with **width/height/area** (px and µm/µm²), listed, moved, resized and deleted like a
  line; it is stored through the same `measurements` row (trait `bbox`, geometry in the new
  `width_px/height_px/area_px` + `_um` columns - `COLUMN_MIGRATIONS`, `X-N06`). Clicking a model's box
  or keypoint on the Bench **is** the human accept that turns it into a measurement at once (a pose
  pair a line, a detection-only box a box; `mlContext.md` law 7), and `Backspace` removes whatever
  object is selected - box, line/measurement, prediction or point (`X-N15`, canvas
  `annotation_delete_requested`). Tests: `tests/test_bench.py::BenchBoxTests` (+6),
  `tests/test_image_canvas.py::ToolTests` (+1). **899 tests, 1 skipped**; `make ui-check` green.

- **Bench loses its modes: a staged "this view", saved or lost** (2026-10-18; asked for as
  *"remove measure mode and specimen in bench ... a new mode where we only see what's been added in
  current view ... save to bench archive ... if bench results are not saved they should be discarded
  and the user must be notified"*; `SR-BENCH-013`/`SR-BENCH-014`). The Bench no longer carries the
  *Mode*/*Specimen* dropdowns or the pass machinery - the canvas' *Tool* group decides what a press
  means and it opens in **Line**. Measurements made now live in `Bench.session` (*this view*), listed
  under an explicit "not saved ... these measurements are gone" warning; loading another image
  discards them and says so; two buttons save the view to the **bench archive** or onto a **study**.
  A view line stays editable before saving (`X-N16`). Tests: `tests/test_bench.py` rewritten (the
  pass-mode class replaced by the view/save tests). **892 tests, 1 skipped**; `make ui-check` green.

- **Hover a target, click it, edit it - in any tool** (2026-10-18; asked for as *"when near to
  another keypoint or spine or box or prediction (eg on hover) i want to be able to click on it and
  edit right away, without having to [switch to Select]"*; `X-N15`). The canvas now answers *"what
  would the next press grab?"* in one place (`ImageCanvas._hit_editable`) and, on hover, reports it
  (`ImageCanvas.hover`), highlights it in the accent colour (`_paint_hover`) and shows a grab cursor.
  A press **on** a keypoint, box or prediction edits it - move/resize/delete, drag a point, or pick a
  model's box into the draft - **even while a create tool is on**, so a correction no longer costs a
  tool switch. Only **empty space** runs the tool's own gesture, so measuring is unchanged. The
  point-drag branch now keys off the drag's own `grab` flag, so a grabbed point is a **move**, not a
  placement, in any tool. Measured with real mouse events (Bench): clicking the just-placed A point
  moves it and stores no line; empty space still places a point. Tests:
  `tests/test_image_canvas.py::ToolTests` +6, `tests/test_bench.py::BenchToolTests` +2; one test that
  asserted the old rule now states the new one. **895 tests, 1 skipped**; `make ui-check` green.

- **One press, one meaning: the canvas is a tool, and Bench measures lines** (2026-10-18;
  asked for as *"currently by default the edit mode is on ... i would like a toolbar like
  'line' with A and B points ... add multiple measurements and call it A_1 A_2 ... split the
  ability of edit into multiple. like select can be its own and in box mode we can edit boxes
  ... keypoint mode ... should be its own mode or tool"*; `X-N14`, `SR-BENCH-013`). Editing
  was one boolean (`box_edit`, on by default) doing five jobs, so the Bench opened in a mode
  where the first click **selected** rather than measured - a mode switch before *any*
  landmark could be placed. The gesture is now an explicit, mutually-exclusive **Tool**:
  `line` (place two points = one measured line), `keypoints` (place the anatomy's keypoints,
  drawn as the schema's **bones** from `KeypointSystem.edges`), `box` (empty space anchors a
  corner so a box can be drawn *and* edited; a stored box's press selects it) and `select`
  (pick/move/resize/delete; only this one marquees). On the Bench the Line tool's ends are
  **A**/**B** and the line is stored the moment B lands - A_1/B_1 then A_2/B_2 with **no
  *Save* between them**; the stored trait stays `length` with the real scale. `box_edit` and
  `box_mode` still work (folded into `tool`), so Dataset and Annotate are unchanged.
  Measured with real mouse events: two clicks store one row and the next line is specimen 2;
  a *Box*-tool press on empty space anchors a corner while a **stored** box's press selects.
  Tests: `tests/test_image_canvas.py::ToolTests` +8, `tests/test_bench.py::BenchToolTests`
  +4, and `tests/test_dataset.py`'s box-corpus flow now states the new contract.
  **887 tests, 1 skipped**.

- **The crop loop's press is the keypoint gesture, a dragged box is stored in the frame's
  pixels, and the readout is as wide as its words** (2026-10-17, reported as *"not able to
  correct keypoint as it first selects box ... the info box overlay is very small and needs
  to be as wide as image canvas"*; `SR-DATASET-028`). The item's box covers the whole
  magnified crop, so with *Edit* mode on (the default) the ordinary press order selected
  that box and **no landmark could be placed** - measured: a real click at the centre of the
  crop left `points == []`, `selected_box == 1`. `ImageCanvas.point_mode` now makes the press
  the keypoint gesture there (a point within `BOX_GRAB_PIXELS` is still picked up first, so
  correcting a keypoint is one press; anything else places the next landmark; empty space is
  not a marquee), and a box is corrected in *Frame* view. A box **dragged** while a crop was
  on screen was stored as the crop's own pixels against the crop's own size (specimen 1 of a
  1280 x 720 frame became `(82, 79, 382, 259)` on a "450 x 330" image): `source_box()` is
  now `crop_box()`'s way back, `_source_size()` names the frame, and the queue item takes the
  correction. And the readout: `ElidedLabel` never handed its text to `QLabel`, so
  `sizeHint()` was the width of *nothing* and the strip asked for a sliver -
  **120 px of card on a 740 px canvas**, its sentence elided; the honest hint makes it
  **701 px** (cap 716), while a short fact still gets a small card.
  Tests: `tests/test_image_canvas.py::EditAllTests` +2,
  `tests/test_dataset.py::DatasetBoxQueueTests` +2 (driven with **real** mouse events),
  `tests/test_canvas_workspace.py::StripTests` +1. **875 tests, 1 skipped**; `make ui-check`
  passes.

- **The canvas readout is a strip of KPI cards** (2026-10-17; asked for as "separate the
  info card into multiple cards ... subtle swiss design ... identify primary kpis that make
  sense to display individually"). One translucent plate holding a `"\n"`-joined paragraph
  became `Kpi` → `KpiCard` → `KpiStack` (`daphnia/ui/widgets/canvas_tools.py`): a
  **micro-caps heading**, the value, and an optional **caveat flag** (`WEAK` / `STALE` /
  `UNSCALED` / `NONE ABOVE THE THRESHOLD`) with a tone that is *refused* when the stylesheet
  does not select on it. `CanvasWorkspace.set_kpis(cards)` is how a page publishes them -
  **the headline first and in larger type**, so the number is read *before* the sentence
  that qualifies it - while `set_readout(text)` still means "one card per line, no heading"
  (the prompt, the last message). Cards are **re-used in place** while their keys and shape
  hold (Predict rewrites its readout on every frame), each one **as wide as its own words**,
  pinned to the canvas' top-left corner, and no card ever eats a click
  meant for the image. The colours are `PHOTO_OVERLAY` tokens (identical in both schemes -
  they lie on a photograph); the hierarchy is the **type** (9 px caps / 12 px value / 17 px
  headline). The KPIs per surface: the **length** (Bench, Annotate), the **value being
  labelled** (Dataset), the **measured 1 mm grid** (Optics), and the **animal count** +
  pose median + `ms/frame`/`fps` + source (Predict — the *checkpoint* is metadata and stays
  in the panel, because six cards measured 367 px against 286 px of canvas). Contract:
  `tests/test_canvas_workspace.py::KpiCardTests`.

- **One UI system for every window** (2026-10-17). Every window — hub, Study shell,
  Optics, Bench, Dataset, Predict and a *not-built* surface's plan — carries the same
  **status footer** (camera state + Connect, the scale in force with its uncertainty,
  age and RE-VERIFY flag, and `Calibrate...` that hands over to Optics and comes back),
  the same **theme** (tokens + palette + template, following the desktop's light/dark
  mode, pinned by `DAPHNIA_THEME`), the same **right-hand panel width** (measured, so
  nothing is clipped) and the same **image canvas entry point**
  (`ImageCanvas.set_source`: frame, file, `QImage`/`QPixmap` or `None`; one loader, a
  named refusal drawn on the canvas, and never two images or a stale number).
  `make ui-check` builds **every window in both schemes** and asserts the panel fits;
  `tests/test_theme.py`, `tests/test_image_canvas.py::SourceTests` and
  `tests/test_ui_smoke.py::WindowFooterTests` are the contracts. **779 tests.**

- **The top-down pose stage, visible end to end** (`python run.py predict` is the
  witness; `make train-pose` / `scripts/train_rtmpose.py` is the entry point).
  `daphnia/vision/pose_corpus.py` reads a Dataset export's **keypoints**;
  `daphnia/vision/rtmpose/` is a CSPNeXt + **SimCC** model, its data pipeline (a
  square box crop with one object owning the source↔input mapping), its trainer
  (group-held-out splits, honest `pose_metrics`), its checkpoint and its inference
  object. On the Predict surface **Load detector** loads both models, each animal
  gets its box, a `head`/`tail` dot, the head→tail line **and its length written above
  it**, the **Keypoints** toggle turns the pose pass off and on, and the **Scale**
  group converts to micrometres with the app's own calibration - naming it, warning
  when it is weak or stale, and re-labelling in place when it changes
  (`SR-PREDICT-012..016`). `docs/rtmpose.md` is the task doc,
  `docs/screenshot-predict-length.png` the view, and the first real run
  (`models/rtmpose_daphnia.pt`, best epoch 135) reports **median 3.2 px / 227 µm,
  p90 37.5 px** on **one held-out capture** - so the report says the validation set
  cannot resolve small differences (`mlContext.md` §6 trap 8). Corpus reality
  (2026-10-17): **384** 2-point samples on **155** frames from **4** captures.

- **The recursive annotator's first half** (`python run.py dataset`, `SR-DATASET-022`,
  `-023`, `-024`, `-025`). `Open boxes export...` asks for the **boxes file and the
  folder its images are in** (suggested, never guessed), imports every box into the open
  dataset keeping its source (`model` / `manual`), and queues them **one animal per
  item**. Each item has **two views** (`Crop` tool / `C`): the **whole frame with every
  box** drawn, the current animal solid on top - the bounding-box display - or the animal
  alone, magnified (`daphnia/vision/crop.py`: context margin, integer factor,
  nearest-neighbour). Both views store on the **original frame** (the view is only a
  view). **Yes (Y)** turns a `source='model'` box into ground truth (rewriting the row
  and naming the accept); **No (N)** removes the box and leaves the frame in the corpus.
  A prediction is never left unanswered. The queue is built from the **corpus**, so an
  animal the detector missed can be added by hand, and a keypoint-only dataset is offered
  the dataset that *can* hold a box in one click. The **skeleton** (`edges`) is part of
  the schema and of the export.
  Verified by `tests/test_crop.py` (19: the mapping with hand-computed numbers, exact
  enlargement, clamping, refusals), `tests/test_dataset.py::ImportBoxesTests` (7),
  `::BoxesExportDialogTests` (5) and `::DatasetBoxQueueTests` (25: a click in crop
  coordinates stored **exactly** as frame pixels, both views, Yes/No, per-animal
  clearing), `tests/test_export.py::ReadBoxesTests` (12) and
  `tests/test_annotation.py::SkeletonTests` (6) — **626 tests green**, `make ui-check`
  paints all six canvases, and both views were **rendered offscreen on the real 526-box
  corpus**.
- **Real-time RTMDet detection, trained from the corpus** (`python run.py predict`,
  `scripts/train_rtmdet.py`). The **Localize step is no longer a stub**: a trained
  detector finds each animal in a live frame, a video file or a single image, and the
  boxes are mapped back to original pixels before anything is drawn
  (`docs/requirements/SR-PREDICT.md`). RTMDet's architecture — CSPNeXt backbone,
  CSPNeXtPAFPN neck, a head sharing its convolutions across levels with a separate
  BatchNorm per level, one prior per location, **SimOTA** assignment with a quality
  focal classification loss and GIoU regression — is re-implemented in plain PyTorch
  so it trains on this M4 (`rtmdet-tiny`, 316 k parameters). `docs/rtmdet.md` lists
  exactly where it differs from the reference implementation.
  - **Verified**: `make train-check` (a real forward + backward on **MPS**; 139
    tensors graded), a 12-step training run over `datasets/bb` (group split: 158
    train / 15 val frames), evaluation, and checkpoint write/read. What is *not*
    done: a model trained to convergence — see Known issues.
  - Input is letterboxed to a multiple of 32 and the pad is **recorded** in a
    `Letterbox` whose `invert()` is the only way back, so a detection cannot be
    drawn or scored off by a constant factor.
  - The split defaults to **`--split-by group`**: whole capture sessions are held
    out, so a validation score cannot come from sibling frames. The export's own
    leaky split stays available and the report says which was used.
  - The reported **AP50 is this script's own greedy IoU-0.5 score, not COCO mAP**,
    and it says so in the report. Predictions are never stored
    (`SR-PREDICT-N01`).
  - A **video file** is readable frame by frame: `←`/`→` (or the *< Frame* / *Frame >*
    tools **on the image**) step exactly one frame, forwards or backwards, pausing
    first; the readout says which frame is on screen. The tools and the arrow keys
    exist **only** while a video is open (`SR-PREDICT-010`, `SR-PREDICT-011`) - with a
    live feed a frame cannot be "next", and the arrows keep panning. Verified against a
    **real mp4 through the real worker** (`tests/test_predict.py::VideoSteppingTests`)
    and by the shortcut-vs-pan test that presses the actual key.
- **Streaming + capture.** Phone -> SRT -> MediaMTX -> RTSP -> OpenCV. Bursts
  land on disk reliably (293 frames in one afternoon, 319 MB in `recordings/`).
  Landscape rotation once on arrival keeps one coordinate space.
- **Optical calibration (decoupled), as three panels: a name, a grid size, a scale.**
  **1 Preset** - the name the scale belongs to (`PROF_<name>`, one name is one optical
  chain) plus what is in force for it; **2 Reference** - the printed square size
  (1 mm by default, finer paper for finer optics) and a live frame or any photo, with
  the measured grid **drawn over the photo**; **3 Scale** - S, its `+/-` **with its
  source**, a concrete example length, and a **hand-entry** row for a scale the
  operator already knows (stored as their stated value, with no proof image). Panel 1
  also lists **every calibration in the store** (newest first, the one in force marked)
  and loads the **proof photo** of any chosen one - a view that changes nothing in
  force, says so when a typed scale has no photo, and names a photo that has gone
  (`SR-OPTICS-031`). The target is only printed **grid paper**, detected automatically
  (`daphnia/vision/grid.py`: 2D FFT for the lattice, projection + zero-padded DFT for
  a sub-pixel period); it **refuses** anything that is not a square grid. The photo
  may be **any file**, including the `.HEIC` an iPhone writes by default
  (`daphnia/vision/images.py`, via macOS `sips`). A calibration is an **immutable,
  dated record** grouped by `setup_key`, carrying its own uncertainty (fit error
  combined with the printed paper's ~0.5 % tolerance); every measurement stores the
  scale, the uncertainty and the calibration it used.
- **One shared status footer in every window** (`daphnia/ui/widgets/status_footer.py`):
  the **camera state** (not connected / connecting / connected-but-silent / live)
  with a Connect-Disconnect action, plus the scale, its `+/-`, its age, `RE-VERIFY`
  and a Calibrate action - so both facts are visible and actionable anywhere.
- **Bench: measuring without a study** (`daphnia/ui/pages/bench.py`,
  `python run.py bench`). Open any image or use the live frame, click two landmarks,
  read the length. Results persist with `study_id = NULL` (a session per image, so
  promotion is precise), appear in a flat list, export as CSV **with their landmark
  coordinates**, and can be **attached to a study later** without re-measuring.
  With no calibration the readout is labelled `UNSCALED` and nothing is stored,
  because a pixel count in a micrometre column would be a fabrication.
- **Dataset: building the training corpus, one dataset per model**
  (`daphnia/ui/pages/dataset.py`, `daphnia/ui/hub.py::DatasetWindow`,
  `python run.py dataset`). The surface starts by choosing or creating a **dataset**,
  and the dataset's **model** (`daphnia/annotation.py`) fixes what a label is: *pose
  2 keypoints* (head → tail), *pose 4 keypoints* (head, tail, sideL, sideR - the
  default), either **plus a bounding box**, or *bounding boxes only*. Only the chosen
  model's traits are offered (a two-keypoint corpus cannot label a width), a
  detection-only dataset hides the keypoint controls and labels **one box per
  specimen by dragging a rectangle** (≥ 3 px; corners normalised; stored on release),
  and a box is corrected by redrawing it. Frames come from a folder (import
  recursively), a single file, or the live frame, navigated with Previous/Next and the
  keyboard (A/D, arrows, S to save). Labels are **scale-free** (source-image pixels,
  no study/plate/connection/calibration required - `SR-DATASET-N01`); when a
  calibration is in force its scale and the µm length are recorded too. Labels are
  unique per `(dataset, file, specimen, trait)`, boxes per `(dataset, file, specimen)`,
  so two models can share a frame and no dataset can overwrite another's ground
  truth. Live per-dataset statistics, and `maintenance.export_dataset()` writes
  `images/` + `labels.csv` (+ `p1_name`/`p2_name`) + `boxes.csv` + `labels.json`
  (schema and per-specimen records: named keypoints and `bbox`) + `manifest.json`,
  with a deterministic 70/15/15 split by image. Nothing here prunes, trays or
  auto-selects, so the corpus is never destroyed.
- **Annotation that does not fight the operator** (Bench + Dataset). Zoom is
  **proportional to the wheel/pinch delta** (a trackpad's small deltas move gently,
  no runaway); **`Command`/`Control`-drag or the Pan tool** moves the view without
  dropping a landmark; every already-annotated specimen is **drawn in its own colour**
  so the frame's progress is visible at a glance with no label text; and both
  surfaces offer **two labelling modes** - two points per specimen, or **one landmark
  at a time** (all heads, then all tails, then the width sides), storing each interval
  the moment both ends exist (`daphnia/labeling.py`). The **pass guides itself**: the
  specimen counter advances per click, the first pass sets the specimen count, and a
  completed pass auto-selects `head -> spine base -> left side -> right side`, so
  nothing is cycled by hand.
- **Label controls used for what they are + honest frame clearing** (Bench +
  Dataset). `Mode` and `Specimen` are **dropdowns** (the specimen list grows as
  specimens are added) - they are real choices. The current **keypoint/trait is
  displayed** with a **Next keypoint** button to step it (the first switch is manual;
  the rest auto-advance). The actions are **separate buttons**: *Save* (hidden in
  landmark mode, which stores automatically), *Undo*, *Clear points*, *Clear frame*.
  *Save* advances to the next specimen. **Clear frame** deletes the frame's stored
  labels/measurements *and* the open points, so a frame imported from another study is
  re-doable from scratch. Overlays draw each measured **interval** as its own line
  (width connects left→right; two unpaired pass points are dots, never a line).
- **Interaction fixes from real use** (Bench + Dataset). **Undo removes the last thing
  placed** - the loose point first, then the stored interval - and stored points are no
  longer preloaded into the editable set, so a stored row stays reachable. Specimens
  get their **colour from the first click** (in landmark mode the pass points are not
  the editable set; everything is drawn by the overlays) and each is **numbered** at
  its first point. A **horizontal two-finger scroll pans** (previously only ⌘-drag or
  the hand tool moved the view). The **specimen dropdown is wide**.
- **Persistence.** Schema + idempotent `init_db()` + `migrate()`; foreign keys on;
  dict-like rows; studies, sessions, observations, media assets, measurements,
  plates, calibration records, templates.
- **Step-based shell.** Guarded Back/Continue, `StepBar` states, `ShellPage`
  visual/controls split, controls scroll inside the panel, zoomable canvases with
  pixel-accurate click mapping, full-screen frame viewer.
- **The measuring controls live on the image** (`X-N11`): every canvas carries an
  expandable **View** cluster (top right: Zoom +, Zoom -, Fit, Pan, Full screen - the
  header shows the magnification) and a page-filled **Tools/Label/Measure** cluster
  (lower left), plus a readout line on the picture (`X-N12`). **Full screen** hides
  the panel, the live preview and the window chrome (step bar, footer) and works on
  the image alone; `Esc` leaves. Built once in `daphnia/ui/widgets/canvas_tools.py` +
  `canvas_workspace.py`, used by Calibrate, Annotate, Bench, Dataset and Predict.
- **Select** (one best frame per target, sharpest auto-pick) and **Annotate**
  (carousel, several specimens and traits per frame, live readout,
  `(frame, specimen, trait)` uniqueness).
- **Data lifecycle.** `prune_discarded`, `export_measurements_csv`,
  `write_manifest`, `consolidate_and_close`.
- **Study templates** (versioned by name) and **acquisition protocols**
  (`quick`/`standard`/`careful`).
- **A real, accurate reading.** Five body lengths 94–279 um through two clicks;
  ~**2.5 %** repeatability on the same specimen.
- **Multiple launch points over one backend.** The Home hub plus the study shell;
  a surface key on the command line (`python run.py bench`) and a specification
  window for surfaces that are not built yet. Adding a surface is one registry
  entry plus one requirement document - no new application.
- **No ghost, one box per animal, and letting go is the update** (`SR-PREDICT-018`,
  `SR-DATASET-027`, 2026-10-17 seventh round). "I see a ghost lagging behind ... I only ever want
  one box and no model box and real box ... when letting go box or point the new position is not
  updated": the prediction being edited stayed in the layer (a dotted box at the **old** position
  under the moving draft), the model's suggestion was drawn on top of an animal the human had
  already labelled, and a point drag was reported as a **placement** - which a pass-mode page
  answers by walking to the next specimen and re-setting the canvas' points (the point snapping
  back mid-drag). Now: a prediction is not painted while it is being edited and leaves the layer
  when a drag turns it into an annotation; a prediction a stored label covers (IoU >= 0.5) is not
  painted at all; a point move emits `points_moved` and the **drop** (`points_edited`) writes the
  correction for a (specimen, trait) that is already stored - and nothing for a pair still being
  placed. The two paint rules are asserted by **counting pixels** of the grabbed canvas, because
  that is what the report was about. Full suite: 870 tests, `make ui-check` 12 canvases.
- **One click selects, one key deletes, and nothing is drawn twice** (`SR-DATASET-027`,
  2026-10-17 sixth round). Three complaints, one chain: a fast second click still reached
  `double_clicked` → Dataset's `next_frame()` (so the frame jumped and the selection was thrown
  away, which is why a click seemed to need two presses); the page also *sited* the selected
  specimen's stored box as the active box, so the same animal was drawn twice. Now: in *Edit*
  mode a double click is two selections (the queue-walk gesture stays for creation mode), the
  active box stays empty while editing, and `Backspace`/`Delete` are bound as **page shortcuts**
  on Dataset, Bench and Annotate calling `canvas.delete_selection()` - one implementation, no
  dependence on which widget has focus.
- **`Edit` mode is automatic: a click selects, `Backspace` deletes** (`SR-DATASET-027`,
  2026-10-17 fifth round). *Edit* is a checkable menu item **on by default** on Dataset, Bench and
  Annotate: a click inside a box or on a point selects it and **never** creates a label, clicking
  a model's box puts it in the editable draft (a click stores nothing, a drag stores it as an
  annotation), and `Backspace` removes the selection - a corpus box (the page deletes the label),
  a model's box (`prediction_removed`: nothing was stored, the layer drops it), or a point.
  Several boxes are selected with Shift/Cmd-click or a marquee and then move and disappear
  together. `Edit` off is the creation mode, and `tests/test_dataset.py` asserts both halves.
- **Every point, box and model box on a frame is editable** (`SR-DATASET-026`, `SR-DATASET-027`,
  2026-10-17 fourth round). The canvas' *Edit* mode: click a box to select it (Shift/Cmd-click
  or a marquee for several), drag its middle to move it - the whole selection moves together -
  an edge or a corner to resize it, `Delete` to remove it; drag one **point** on its own;
  drag a **model box** and the drop stores it as an annotation. Four bugs were measured out on
  the way: the group move reverted all but the first box, a dragged model box stored two labels,
  the hit test took the first box instead of the innermost, and it crashed on `QPointF`/`QRectF`.
  23 tests in `tests/test_image_canvas.py` (`BoxEditTests` + `EditAllTests`).
- **The model menus list the models folder, and a load switches the tool on** (`SR-PREDICT-019`,
  2026-10-17 third round). Each of *Detector (RTMDet)* / *Pose (RTMPose)* lists every checkpoint
  in the default models directory by name (`models.checkpoints_in`), loads **and** registers one
  in a single click, and carries a check mark on the checkpoint in force and on the entry this
  window has loaded. Taking a model from the menu switches the page's *Model* tool on, because
  the menu click *is* the decision to use it - the previous behaviour (model loaded, switch off,
  no prediction) made the menus look broken.
- **The model is off by default, and any box on a frame can be edited** (2026-10-17, second
  round). `ModelRunner` (`daphnia/ui/model_pass.py`) runs the top-down pipeline on one frame;
  Dataset, Bench and Annotate call it from a *Model* switch that is **off by default**,
  draw its answer in the canvas' *prediction* layer (dotted, one colour, never stored) and
  hand the **largest** animal to their editable draft. The canvas' *Edit* mode
  (`SR-DATASET-026`) picks the innermost box under a click, moves it by its middle, resizes
  it on an edge or a corner, removes it with `Delete`, and reports the corrected geometry so
  the page stores it with the same write a drawn box uses; a click that the tool uses never
  creates a second label. 13 new tests (`tests/test_image_canvas.py::BoxEditTests`) and 8
  (`tests/test_model_pass.py`).
- **One menu bar for the application, in one fixed order** (`X-N13`, 2026-10-17):
  `File | <the page's own menus> | Source | Models | View | Go | Help`, built empty in that
  order by `HeaderBar` so every window reads the same, filled by the window (*View*, *Go*,
  *Models*), by the page (*File*, *Source*, its own *Measure*/*Label*/*Frames*) and by the
  app (*File*'s *Exit*, *Help*'s walkthrough + *Where the data lives*);
  `hide_empty_menus()` takes out a menu with nothing behind it. A **canvas-less** step
  declares its menu items through `ShellPage.menu_tools()`. **An action is a menu item, a
  choice stays beside its panel**: ~20 panel buttons moved into the bar (Calibrate 4,
  Bench 5, Dataset 7, Annotate 2, Dashboard 4, Visualize 1), while combos with their
  applying button, Predict's control panel and each workflow step's own controls stayed.
- **Loading a model is applied and visible** (`SR-PREDICT-017`, 2026-10-17): the *Models*
  menu entry loads the checkpoint **and** registers it as the active one; the window hands
  it to the page (`model_menu.hand_over` - one implementation for the hub, a surface window
  and the study shell, whose connection to `loaded` had been missing so a load there was
  silent) or names it in the status bar; and on the Predict surface the load **runs at
  once** - a running pass is restarted on the same source, a still frame is detected again,
  and the header/readout report the model and what *it* found (`0 found  NONE ABOVE THE
  THRESHOLD` included). Which checkpoint is in force is the registry's decision: a source
  chosen with nothing loaded loads `models.active(...)`, never a default file.

## What is not built yet

- **The Calibrate surface's S2 and S3 are designed, not built.**
  `docs/ui-calibrate-redesign.md` §8: **S2** makes the record carry what the surface measures
  (`calibration_profiles` gains `drift_pct`, the field-stop circle and the pitch's provenance,
  so a measurement's `uncertainty_pct` can state what is true *where it was taken*: `hypot(2.49,
  0.5) = 2.54 %` at the calibrated spot, `hypot(2.49, 5.1) = 5.68 %` at the edge) and **S3**
  makes the gate session-scoped (`workflow.facts()` requiring a calibration for *this* session,
  i.e. inside `CALIBRATION_MAX_AGE_HOURS` **and** matching today's mount). S1 (the front end)
  is done - see "What works". Neither S2 nor S3 stores a distortion model: the mirror test stays
  the gate (`SR-OPTICS-033`).

- **No distortion step in the app.** The probe measures the question offline; nothing in
  `daphnia/` reads, stores or applies a distortion model, and no surface *applies* one -
  the Calibrate screen now says `lens distortion: unmeasured - no model for this preset`
  (`SR-OPTICS-032`'s wording is on screen; the model is still proposed, not implemented).
  The three
  options and their costs are in `docs/calibration.md` §5; the shape if it lands is
  `daphnia/vision/distortion.py` (Qt-free, discorpy imported lazily, coefficients +
  residual stored with the preset by `setup_key`).

- **No trained model.** `vision.keypoints.predict_keypoints()` raises
  `NotImplementedError`. `models/` is empty. Localize (`vision.detection`) and
  the `pipeline/*` steps are likewise placeholders.
- **No annotated dataset *content* yet.** The DATASET surface and its store
  (`corpus_labels`) and export exist; `datasets/annotations/` still holds only
  `.gitkeep` because no frames have been labelled yet.
- **The study CSV export still drops `p1/p2`.** `EXPORT_COLUMNS` in
  `maintenance.py` writes `study_id, target, specimen_index, trait, length_um,
  length_mm, length_px, px_per_um, source, measured_by, created_at` - the landmark
  ground truth is dropped. The DATASET export (`export_corpus`) and the bench export
  keep them; the study export was left untouched (its surface's policy).
- **No specimen/study split yet.** `export_corpus` groups by *image* file; the
  per-specimen/per-study holdout (`SR-DATASET-010`) is the next step.
- **No way to measure an image from disk.** `QFileDialog` appears only for the
  destination folder and the CSV save. Calibrate and Acquire read the live stream
  only.
- **No timepoint schedule.** `interval_hours`/`day_number`/`current_well_index`
  exist, but there is no "what is due today" and no overdue view.
- **Treatment assignment on plates** — explicitly "still to design".
- **`setup_plates.py` is a 12-line stub**; `SetupPlates` is imported by
  `daphnia/ui/pages/__init__.py` but is **not** in `PAGE_CLASSES` or
  `workflow.STEPS`, so it is dead code today.
- **A package for another machine** (2026-10-22). No `.exe`, no `.dmg`, no `.spec`, no CI
  job, no measured bundle size. Today a second machine gets the app by cloning the
  repository and running `conda env create -f environment.yml` — or by `conda-pack`ing
  `z-app` for an offline target. The code side is ready for a bundle
  (`DAPHNIA_DATA_DIR`, and `pytorch` imported lazily so a Bench/Dataset/Study build needs
  no torch), and the route with its costs is in the README's *Packaging* section; nothing
  is built. The rule that shapes the work is not ours to choose: PyInstaller cannot
  cross-build, so a `.exe` needs a Windows machine and a `.dmg` a Mac, one build per
  architecture.
- **Predict's latency on Windows.** PyTorch's MPS backend is Apple-only, so the detector
  and the pose model run on CPU there — and no run has been timed on a Windows machine.
  The numbers in `docs/rtmdet.md`/`docs/rtmpose.md` are MPS on the M4 and must not be
  quoted for another platform.

## Known issues

| # | Issue | Impact |
| --- | --- | --- |
| 1 | **One pipeline for all intents** — 14 inputs / 6 screens before a number | 12 throwaway studies in one afternoon |
| 2 | **No file-based measurement path** | colleagues cannot use it at all without a phone + study |
| 3 | **`prune_discarded` deletes frames** that DATASET would want | still true for *study* frames; DATASET now stores its own labels in `corpus_labels`, which prune never touches |
| 4 | **Export drops `p1/p2`** | DATASET and bench export keep them; the study export still drops the landmarks |
| 5 | ~~Calibration identity did not pin the optics, and recalibrating overwrote the scale (observed 0.783 -> 0.147 under one `profile_id`)~~ | **fixed** - each calibration is its own immutable record, so that swap can no longer happen silently. *Remaining:* the active scale is still one global row rather than per setup |
| 6 | ~~`ANATOMY` hard-codes animal/plant~~ | open - still a dict in code; a new organism is still a code change |
| 7 | **Select enforces exactly one frame per target** and `workflow` counts `kept` | cannot accumulate many frames per specimen — the opposite of a corpus |
| 8 | **SQLite has no WAL/`busy_timeout`** | two surfaces as separate processes would lock each other |
| 9 | ~~README/docs route table listed routes that do not exist (`plates`, `review`)~~ | **fixed** - both now list the real study routes and the launch points |
| 10 | **Reset means data loss** — `make db-reset` was the escape hatch; today's numbers survive only in `zoo.db.bak.20261015-164012` | no "clean bench" without losing the corpus |
| 11 | ~~No detector had been trained~~ **trained once (2026-10-16, 30 epochs)**, and the machinery works — but the reported "best AP50 0.813" sits on **one held-out capture of 15 frames** | the number cannot distinguish epochs (0.16→0.81 across epochs); more captures are needed before a score means anything |
| 12 | **14 capture sessions is very few, and the export's split leaks** (11 of 14 captures appear in more than one split) | the grouped split fixes the *score's* validity but leaves one capture in validation; the corpus needs more distinct sessions (`SR-DATASET-010` is the durable fix) |
| 13 | **RTMPose is not built** — `vision.keypoints.predict_keypoints()` still raises | the body-length endpoint needs it; it needs its own four-keypoint corpus, and RTMDet's boxes come first |
| 14 | **The trainer cannot resume** — a run trains from scratch and writes the best + final state | a 30-epoch run is minutes, a 100-epoch one is a sitting; resuming is a documented next step, not a half-working feature |
| 15 | **The detection score is not well calibrated** — recall collapses from 0.83 to 0.15 between score thresholds 0.35 and 0.50 | 0.35 is the operating point, but the usable range is narrow: the threshold must be re-tuned per illumination, and a value near 0.4 means "probably", not "no" |
| 16 | **Wall-clock time on this machine is not stable** — the identical 30-epoch run took 4 min and then 65 min (245 s of CPU both times) | plan long runs with a margin, and re-measure on a quiet machine |
| 17 | **The detector numbers in this memory bank and in `docs/rtmdet.md` §5 are from an older run than the checkpoint on disk.** Measured 2026-10-17: `models/rtmdet_daphnia.pt` embeds a 250-epoch run (best epoch 69, best AP50 0.944), `models/rtmdet_report.json` is a *different* 150-epoch run (best epoch 110, best AP50 0.958; series min 0.000, p25 0.787, median 0.860, max 0.958), and the documented P 0.61 / R 0.83 / F1 0.70 at ≥0.35 belongs to the earlier 30-epoch run | re-measure before quoting: `python scripts/train_rtmdet.py --evaluate --checkpoint models/rtmdet_daphnia.pt --score-threshold 0.35` (one threshold per run — there is **no sweep command**, so the threshold table in the docs is not reproducible by one command). On 2026-10-17 that gave **P 0.958 / R 0.852 / F1 0.902 / AP50 0.848** at ≥0.35 and **P 1.000 / R 0.296** at ≥0.50, on the same one-capture validation set |
| 18 | **The keypoint vocabulary assumes a lateral view** (`projectbrief.md`: "lateral view assumed"), but the corpus is photographed from above, from the side and on the belly — different measurements of the same animal | the four-point system (`head, tail, sideL, sideR`) has no view/orientation axis, and the boxes carry no body axis at all (median box aspect ratio **1.06**, 42.6 % taller than wide — measured on the 526 boxes). The decision belongs in the pose task doc **before** keypoints are labelled at volume (`SR-DATASET.md`, "Still to build") |
| 19 | **`assets/ui/dataset.ui` is one group box wrapping two others** — `export_group` ("DataSet UI - corpus, label and export") contains `corpus_group` and `label_group`, where every other surface has three **sibling** sections (`bench.ui`: 1 Image / 2 Measure / 3 Results; `calibration.ui`: 1 Preset / 2 Reference / 3 Scale). Found 2026-10-17 while validating the readout work: the same untracked, mid-edit file was also missing the `stream_hint_label` the page already set, which crashed the Dataset and Bench windows on open | fixed the crash (labels added, titles restored) so the app and the suite run; the **nesting and the stale absolute geometry** (children sitting at y=590 inside a 111 px box) are still the file's, because redesigning it in code would guess at a layout the operator has not finished drawing. Restore the three siblings in Designer: `make ui-designer FILE=dataset.ui`, then `make ui-list`. **Re-checked 2026-10-17 16:19:** the operator's own Designer save dropped `dataset_label` again (the page sets it, so the Dataset window crashed with 93 suite errors); it is back at `corpus_group` row 3 |
| 20 | **At the *absolute minimum* window size the Predict readout column is taller than its canvas** — with a pose model loaded, five cards measure **239 px** against **166 px** of room (canvas 480x260, lower-left cluster 41 px; measured 2026-10-17) | the top cards stay readable and nothing is dropped silently — but the old single plate measured **150 px** in the same state, so this is a *small* regression in a state only reached by dragging the window to its floor (the default window measures 239 px against 286 px, full screen has room to spare, and the panel carries every fact). Closing it means deciding which facts leave the canvas below a given canvas height — a design decision, not a bug fix |
| 21 | ~~The grid detector refuses every calibration photo this rig produces~~ | **the OPTICS surface no longer uses it** (2026-10-19): `daphnia/vision/lattice.py` measures the same photos - **27.03–27.24 px per cell** on four of them, tilt −9.0°, spread **5.1–6.3 %**, and it *refuses* the two ruler photos and three unusable grid photos with a named reason. `vision/grid.py` stays for clean photos, and the probe still reports its refusal as evidence (`SR-OPTICS-035/036`, `tests/test_lattice.py`). The refusal itself is **not** fixed - it is bypassed, deliberately, and the reasons are recorded in both modules |
| 22 | **The documented line-pattern flow is what made the model fit** (docs 3.3.4, `regenerate_grid_points_parabola`) — measured 2026-10-19: with the field masked as a fitted circle (**448.7 px**, was 60 px small), the lattice straightened (9 deg → **0.00 deg**), the line weights equalised, the paper/surround junction excluded from the points, and **3 coefficients**: **0.511 px rms, worst line 1.223 px → passes discorpy's own sub-pixel check ✓**, COD **(616.3, 316.7)**, local scale **+9.75 % at r = 302 px**. `--edge-margin 30` also passes (0.667 px, COD (613.1, 318.2), +8.09 %) | the distortion is now *measured* on this rig's own paper — the earlier "not yet" was my approximation, not the library's limit. Coefficients are **not** storable yet: the *magnitude* disagrees with the independent measurement (§24) |
| 23 | **The distortions are ~4–6 %, independent of any model** — cell spacing in 24-px bands of the straightened frame: **26.0–28.0 px** per cell across a 40 mm field (**±3.8 %**, up to **~5.7 %** centre-to-edge) = **76–114 µm on a 2 mm animal**, seven times the 0.5 % paper tolerance | the effect is real, positive (edge cells *larger*, pincushion-like) and monotone with radius — so it *must* be corrected or carried per measurement (`docs/calibration.md` §5). This measurement needs no fit and no library, so it is the one to put next to any model |
| 25 | **The passing model fails the mirror test → not storable** — a mirrored image is an *exact pixel permutation*, so a correct model must come back mirrored (coefficients identical, COD at `width - 1 - x_c`). Measured 2026-10-19 with `--mirror-check`: coefficients differ by **0.82**, the centre lands **184 px** from the expected (665.0, 316.8), mirrored residual **8.68 px** vs 0.68 px unmirrored. Transformed-copy residuals for reference: mirrored 1.18, upside-down 8.58, 90°-rotated 4.07, disc re-positioned 1.48 px | the model's *shape* is repeatable across settings (COD within 3 px) but it does not reproduce itself under a symmetry it must satisfy — so the coefficients stay out of the store, and the *effect* (4–6 %, §23) is what gets reported. **The preparation** survives all four transformations (circle r 448.7 identically; alignment sign self-checked), so the pipeline is orientation-proof — the fit is the remaining problem |
| 26 | **The points overlay was drawn on the un-rotated photo** — the points are measured in the *straightened* frame's coordinates, so on the untouched photo they looked rotated relative to the grid (spotted by the operator) | fixed: `*_points.png` is drawn on the straightened frame and a second `*_points_original.png` maps the points back through the rotation matrix. A wrong-looking overlay is a wrong overlay — the evidence image must be in the frame the measurement lives in |
| 23 | **`scripts/calibration_probe.py` had two method bugs of its own** (found by looking at `*_points.png`, not by a test): the field mask was too small (§22) and the tilted-profile sampling skipped ~114 px along one side of a 9 deg-tilted lattice | both fixed and both *measured* now (circle verified against the raw edges within a pixel; slope 0.00 deg after straightening). It is a script, so no test covers it — if any of this moves into `daphnia/vision/`, the circle/straightening/mask-filter steps need tests of their own |
| 27 | **Nothing checks a measured length for plausibility** (`docs/accuracy.md` S14) — audit of the live store 2026-10-22: **550 labelled rows, all `manual`; 440 read over 4 mm and 18 read 6.6–56 mm** ($566.5\ \text{px}$ at $0.01005\ \text{px}/\text{µm} = 56363\ \text{µm}$); one row is degenerate ($s = 0 \to L = 0$). The arithmetic says the **line** is wrong, not the scale: the same 566.5 px line reads 20.8 mm at 27.3 px/mm and 42.4 mm at 13.4 px/mm, so it is not a mis-scaled animal under any stored scale — and **no field records what was on the frame**, so "a line across a plate" and "a label on a non-animal image" are indistinguishable from the row | an impossible number is stored and reported exactly like a real one; the app has no notion of "this is not a plausible organism length". Proposed (named warnings, never silent clamps): a per-study plausibility band from `studies.organism`, a "line spans x % of the frame" note, and a zero-length guard. It is the only source in the budget with **no** handling at all — the other thirteen are derived, displayed, flagged or refused |

## Evolution of decisions

- **v0** — navigation skeleton + persistence + streaming scaffolding.
- **v1** — step-based workflow with hard gates; one governed study modality;
  mode axes (`organism_kind`/`target_mode`/`timecourse`); templates; protocols;
  data lifecycle.
- **v2** — recognise the three audiences, make **intent** a first-class axis, split
  into **surfaces over one backend**, freeze intent in per-surface **system
  requirements** traced to `O1..O4`/`D1..D3`. *Landed:* the requirement set, the
  surface registry, the Home hub, the launch points, and **all five surfaces built**
  (study, bench, dataset, predict, optics).
- **v3 (this step)** — **the model side, as a surface.** The corpus gets a trainer and
  a mirror: RTMDet in plain PyTorch so it trains on the machine that takes the
  pictures, and a Predict surface that runs it live. Two consequences recorded
  deliberately: a split **must hold out whole captures** (so the export's own split is
  labelled leaky rather than trusted), and a reported metric **must state what it is**
  (so AP50 here is not called mAP). *Still open:* a model trained to convergence, the
  RTMPose keypoint model, and pre-labels back into the Dataset surface.

## Verification status

`make test` passes on the current tree: **1220 stdlib `unittest` tests** across the
`tests/` modules (1 skipped — the "MPS is missing" guard — on this machine; measured
2026-10-22, after the total-uncertainty and `DAPHNIA_DATA_DIR` rounds added 20). Any new
surface must: extend `test_ui_smoke` with its page/attribute surface, update
`test_surfaces`' "which surfaces are available" assertion, and add requirement IDs to
the traceability matrix in `docs/requirements/00-framework.md`.

**Added by the readout-cards work (2026-10-17):** `tests/test_canvas_workspace.py
::KpiCardTests` (11 — order and keys, the headline's emphasis, a headingless card keeping
the `canvas_readout` name, one card per line of plain text, a blank value never drawn, a
duplicate key refused, a card re-used in place, a card re-plated when its shape changes, a
flag's tone reaching it from the theme, one column width beside the View cluster, and no
card eating a click) and the five new object names in `tests/test_theme.py`. `make ui-check`
reports **12 canvases painted in light and dark, full screen and back**; the extent of the
change was measured, not assumed: the whole readout route (`set_kpis` / `set_readout`) is
covered by the tests above and by the per-surface readout assertions that already existed
(`test_predict`'s `frame n` and `Pose:`, the pixel test on the tools). The card set was
**measured, not assumed** (offscreen, both window sizes, no display):

| surface | column | room at minimum | verdict |
| --- | --- | --- | --- |
| Calibration | 134 px | 426 px | fits |
| Bench | 127 px | 360 px | fits |
| Dataset | 217 px | 327 px | fits |
| Study · Calibrate | 134 px | 296 px | fits |
| Study · Annotate | 167 px | 263 px | fits |
| Predict (pose loaded) | 239 px | 166 px | **taller than its floor** — known issue 20 |

Dropping the checkpoint card and tightening the padding is what took Predict from 367 px
(the first version) to 239 px; the remaining 73 px belong to the minimum-window state, and
the numbers are recorded as an issue rather than smoothed over.

**Repaired in passing (not caused by it):** `assets/ui/bench.ui` and
`assets/ui/dataset.ui` were missing the `stream_hint_label` their pages already set, so
`python run.py` raised `AttributeError: 'Dataset' object has no attribute
'stream_hint_label'` on the Dataset and Bench tiles and **124 tests errored**; the labels
and the panel's section titles were restored, and the suite went back to green. The
remaining structural oddity of `dataset.ui` is known issue 19.

**Added by the box-queue annotator work:** `tests/test_crop.py` (19 — the crop↔frame
mapping with hand-computed numbers, exact nearest-neighbour enlargement, clamping and
refusals; **no Qt and no model**, so it runs anywhere OpenCV does),
`tests/test_dataset.py::ImportBoxesTests` (7), `::BoxesExportDialogTests` (5) and
`::DatasetBoxQueueTests` (25),
`tests/test_export.py::ReadBoxesTests` (9), `tests/test_annotation.py::SkeletonTests`
(6), plus two export regressions (the skeleton in `labels.json`, the `boxes.csv`
header/row widths) and two `tests/test_canvas_workspace.py` cluster-layout tests (a wide
tool never lands on a half-filled row; no two tools share a cell). `make ui-check`
reports **6 canvases painted, full screen and back**, and both views of the box queue
were rendered offscreen on the real corpus.

**Added by the detector work:** `tests/test_export.py` (the corpus contract — pure
stdlib, so it runs anywhere), `tests/test_rtmdet.py` (losses, SimOTA, letterbox,
the model, checkpoints, the data pipeline — `skipUnless(torch)`), and
`tests/test_predict.py` (the surface with **no** model, a stub detector for the
worker, and a real checkpoint when torch is present).

**Verified beyond the suite, by hand:**
* `scripts/check_torch.py` — `torch 2.14.0`, **MPS available**, a forward *and
  backward* pass on MPS with 139 tensors receiving a gradient.
* A 12-step training run over `datasets/bb` on MPS: loss finite and falling
  (2.0039 → 1.9840), 32→28 priors matched, evaluation and checkpoint write/read.
* `scripts/repair_boxes.py --apply` on the live `zoo.db`: 11 of 526 boxes clamped,
  0 degenerate, `datasets/bb` re-exported and re-read.
* `python run.py --help` lists all five surfaces, and `window_for('predict')`
  builds a `PredictWindow` headless that reports
  `No detector at …/models/rtmdet_daphnia.pt   PyTorch 2.14.0   device: MPS (Apple GPU)`.
