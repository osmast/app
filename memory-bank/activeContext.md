# Active Context — current work focus

_Last updated: 2026-10-22_

## The total uncertainty exists, and the app can now be moved to another machine (2026-10-22, twenty-eighth round)

**Asked, in two messages:** *"we want to display and store the total uncertainty of a length"* —
and then, narrowing it — *"colleagues would like to run the app on a Windows computer with their
own phone streaming … focus on the transferability of the Qt app and the possibility of an exe or
dmg"*. Two deliverables, one round: **the total uncertainty, implemented** (`docs/accuracy.md` §1
had described it as derivable-but-unbuilt), and **the transferability/packaging question answered
with verified facts rather than assumptions** (README, two new sections).

### 1. The total uncertainty — from a formula in a document to a number on screen and in a row

| Layer | What was added |
| --- | --- |
| **The maths** (`daphnia/units.py`) | `CLICK_ERROR_PX = 1.0`, `QUANTISATION_PX = sqrt(2/12) = 0.408`, `quantisation_um(S)`, `click_uncertainty_um(S)`, `total_uncertainty_um(L, S, δ, *, extra_um, click_error_px)` = `sqrt(u_scale² + u_click² + σ_quant² + extra²)`, `format_total_uncertainty(...)` → `total +/- 75.1 um (3.8 %)`. Pure stdlib, no Qt, no torch |
| **The object door** (`vision/calibration.py`) | `OpticalProfile.total_uncertainty_um(um, extra_um=…)`; `CLICK_ERROR_PX` now *is* `units.CLICK_ERROR_PX` — one constant, one owner |
| **Stored** (`db/schema.sql`, `connection.py`, `repository.py`) | `corpus_labels.total_uncertainty_um`, filled by the one write path (`save_corpus_label`) from the same function the screens print; the `measurements` view exposes it; `MEASUREMENTS_VIEW_COLUMNS` is the new sentinel that makes `refresh_measurements_view` rebuild an old view (without it, the live store's view would silently lack the column) |
| **Migrated** (`connection.backfill_total_uncertainty`) | 2 steps, both filling only NULLs: a label that names its calibration adopts that calibration's δ (an immutable record), then the budget is derived. Reported by `migrate()` as two named lines |
| **Shown** (`bench.py`, `dataset.py`, `annotate.py`, `calibration.py`) | every measuring readout now prints both: `2314.0 +/- 6.8 um (2.314 mm)   total +/- 75.2 um (3.2 %)`. Annotate's stored-row readout **reads** the row's own column rather than recomputing it |
| **Exported** (`maintenance.py`) | `CORPUS_COLUMNS` and `BENCH_COLUMNS` gained `uncertainty_pct` + `total_uncertainty_um`; `labels.csv`, `labels.json` (both the flat rows and the per-specimen `traits` block) and the corpus export carry it |

**Measured on the live store (2026-10-22, 550 labelled rows).** The backfill reported
**519 rows adopted the calibration's uncertainty** and **550 rows got a budget** — the second
number is the finding: **519 of 550 labels recorded *which* scale was applied and not how well it
was measured**, so before this round the corpus could not say what any of its numbers was worth.
Per chain (via `zoo.db`): 520 rows at S=0.01336 → 152.8–178.8 µm; 18 at S=0.01005 → 203.6–238.3 µm;
12 at S=0.02726 → 74.9–77.3 µm. The eyepiece chain's 74.9–77.3 µm is the *click* term (73.4 µm)
plus the grid, exactly as §3.1 predicts — i.e. **the budget is 11× the `±`** on that chain, and
the app now says so instead of leaving it to be derived by hand.

**The arithmetic was re-derived against the document, not the other way round**: the two worked
cases in `docs/accuracy.md` §3 are now *oracle tests* — `tests/test_units.py::TotalUncertaintyTests`
pins 75.1 µm (Study path) and 226.3 µm (Predict path, `click_error_px=0.0` because nobody
clicked), and §5(b)'s neonate at 121.2 µm. `docs/accuracy.md` was then corrected wherever it had
become false (§1, §2 S5/S6, §3.1/§3.2/§3.3, §4, §6 item 1, §7, §8) — the document's own rule is
that a changed metric makes it *stale* and the honest response is to re-measure and update, and
that applies to its claims about the app as much as to its numbers.

**Deliberately not done, and stated in the doc:** the pose model's term is *not* added by any
surface (`Predict` still names it as a separate number — folding it in would make a Predict
number look like a Study number), and the drift (S4), the age (S10) and the annotator's own
variance stay outside the sum. `extra_um` and `click_error_px=0.0` exist so a caller that *has*
measured the model term can ask for the combination explicitly.

### 2. Transferability and packaging — two premises checked, one of them wrong

* **"MediaMTX is not available on Windows" — it is.** Its own README: *"Compatible with Linux,
  Windows and macOS, does not require any dependency or interpreter, it's a single executable"*,
  and the v1.21.2 release carries `mediamtx_v1.21.2_windows_amd64.zip` alongside
  `mediamtx_v1.21.2_darwin_arm64.tar.gz`. What is macOS-shaped in *this* repository is only:
  `scripts/setup_mediamtx.sh`, the `SYSTEM_CONFIG` path in `daphnia/streaming/mediamtx.py`
  (reported, never required), `scripts/dev_publish_stream.sh`, the `make` targets, and `sips` in
  `vision/images.py` (guarded by `sys.platform != "darwin"`). `binary_path()` is
  `shutil.which(...)`, so `mediamtx.exe` on `PATH` (or `DAPHNIA_MEDIAMTX_BINARY`) is enough.
* **"Can we create an exe?" — not from this Mac.** PyInstaller's own words: *"The output of
  PyInstaller is specific to the active operating system and the active version of Python … you
  run PyInstaller on that OS, under that version of Python."* Same for `conda-pack`. So: `.exe` on
  Windows, `.app`/`.dmg` on macOS, one build per OS **and** per architecture.

**One code change made packaging possible at all** (`daphnia/paths.py`): everything the app
*writes* now hangs off `DATA_ROOT = DAPHNIA_DATA_DIR or PROJECT_ROOT`. Before it, only the
database could be moved (`DAPHNIA_DB`), so a frozen app would have written `recordings/`,
`datasets/` and `models/` **inside its own bundle** — read-only in effect on macOS (it breaks the
code signature) and Program Files on Windows. `tests/test_paths.py` (new, 4 tests) pins the
default, the move, `DAPHNIA_DB` still winning for one file, and `ensure_runtime_dirs()` creating
the tree wherever it is told.

**Documented in the README**, because that is where a new operator looks: a
`## Run it on another machine (macOS or Windows)` section (a step table for both OSes, the
`make`-target equivalence table, the three macOS-shaped things, the firewall ports, the phone URL,
`DAPHNIA_DATA_DIR`, and "MPS is Apple-only, so Predict latency on Windows is unmeasured") and a
`## Packaging: an .exe, a .dmg, or conda-pack?` section (the cross-build rule quoted from the
PyInstaller docs, conda-pack as the recommended lab route, the three files a bundle must carry —
`assets/ui/*.ui` (14 of them), `assets/theme/daphniascope.qss`, `daphnia/db/schema.sql` — the
`--add-data` invocations for both separator conventions, the torch/onefile/signing/architecture
costs, and `hdiutil` for the `.dmg`). **Status stated plainly: none of it is built**, no `.spec`
file exists, and no bundle size is quoted because none was measured.

## The error budget exists as one document now (`docs/accuracy.md`, 2026-10-22, twenty-seventh round)

**Asked:** *"access the systematic and other error sources and list them… how accurate is a
measurement, and how do we handle each one?"*

**Answer:** a new document, **`docs/accuracy.md` — "Accuracy: the error budget of one length"**,
holding **fourteen numbered sources (S1–S14)** with the **formula for each**, what each is worth
on this rig in µm, and how the app handles it. It is the one place the terms are put side by side
and combined; `docs/calibration.md` is how a scale is *obtained*, `docs/rtmdet.md` §5 /
`docs/rtmpose.md` §3 are the model metrics, and `memory-bank/mlContext.md` is the discipline.
Linked from `docs/calibration.md` (top note) and README (the "Two numbers come out of it"
section).

**What the ledger says, and it is the finding the round produced:**

1. **The `±` on screen is the scale's relative error alone**, and the true budget is
   **13–23× larger** — 13× on the Study path, where the *human click* dominates (S6:
   $2c/S = 73.4$ µm on the eyepiece chain, against a $5.9$ µm `±`), and 23× on the Predict path,
   where the *pose model* dominates (S8: 224 µm median against a 10 µm `±`). Nothing in the
   repository combines the terms; `pipeline/derive.py` still does not consume the pose model.
2. **Two terms are derivable and unshown** — the pixel quantisation (S5,
   $\sigma = 0.408/S$: 15.0 µm on the eyepiece chain, *larger* than the `±` it sits beside) and
   the measurement's own click (S6, a *constant* $2c/S$ µm because the percentage and the length
   cancel). Both need no new measurement: `S` and `s` are already in the row.
3. **S14 is the open item: nothing checks plausibility.** An audit of the live store (SQL in the
   doc) finds **550 labelled rows, all `manual`; 440 read over 4 mm and 18 read 6.6–56 mm**
   ($566.5\ \text{px}/0.01005 = 56363\ \text{µm}$). The arithmetic says the *line* is wrong rather
   than the scale — the same line reads 20.8 mm at 27.3 px/mm and 42.4 mm at 13.4 px/mm — so it
   is not a mis-scaled animal under any stored scale, and **no field says what was on the frame**.
   One row is degenerate ($s = 0 \to L = 0$). A per-study plausibility band, a
   "line spans x % of the frame" note and a zero-length guard are proposed — named warnings, never
   silent clamps.
4. **Two provenance errors were caught while writing it** and are recorded rather than smoothed:
   the detector's numbers come from **two different runs** (the docs' 30-epoch table vs the
   on-disk checkpoint re-measured 2026-10-17), so the doc keeps them in **separate, labelled**
   blocks (`progress.md` known issue 17); and the pose report's recorded pair
   ($3.0\ \text{px} = 227\ \text{µm}$) implies $75.7\ \text{µm}/\text{px}$ while the corpus is
   documented at $74.8$ (where the same error is 224 µm), so the doc states both rather than
   silently averaging them.

**Validated, not asserted.** The document's arithmetic was re-derived against the project's own
code before writing it: `relative_uncertainty_pct` → $14.7 \to 13.61\,\%$, $147 \to 1.36\,\%$;
`LatticeEstimate` → `statistical 2.49 %`, `spread 5.10 %`, `uncertainty_pct() 2.54 %` (= hypot
with the print's 0.5 %); `0.408/S` and `2c/S` on the eyepiece chain → 15.0 µm and 73.4 µm; and
the three stored scale groups (520 / 18 / 12 rows). The file's structure was checked
mechanically: 27 display-math blocks all closed, 221 inline math spans well formed, and every
table row the same width as its header.

**No code changed.** `docs/accuracy.md` is new; `docs/calibration.md` and `README.md` gained a
pointer each. Every value in it is read from a stored row, a constant in the code, or a
documented measurement — §7 of the doc is the command that reproduces each one.

## A recording: a burst whose length the operator decides (2026-10-22, twenty-sixth round)

**Asked:** *"in study2 allow for capturing video stream of variable length, just start stop"*
- Study 2's *Capture burst* stores the `DAPHNIA_BURST_SIZE` frames the shared service happens
to have buffered, and five frames are a lottery on a well that drifts in and out of focus.
The ask is a **Start/Stop recording** on the well the camera is over.

**Answer:** a recording **is** a burst whose length the operator decides (`SR-STUDY2-013`).
*Start recording* opens it, every frame that arrives is written to the well's own folder **as
it comes** (`b<burst>_f<index>.png` - the same names a burst uses), *Stop* stores the whole
take as **one burst** of that well with its sharpest frame as the well's image. Nothing was
added to the schema, the domain layer or the corpus: one writer, one storage path, one rule
for the image (`SR-STUDY2-N03`, `SR-STUDY2-003`).

| File | What changed |
| --- | --- |
| `daphnia/ui/pages/study2.py` | new `Recording` dataclass (burst, folder, observation, well, clock, entries, failed count) and `start_recording` / `stop_recording` / `toggle_recording` / `_record_frame` / `_update_record_label`; `capture_burst` **split** into `_prepare_target()` (study · session · observation · burst · folder, with all the refusals) and `_write_rows()` (the rows + which frame is the well's image), so a burst and a recording cannot store differently; `_sync_state` locks the page to its well while a recording runs; `choose_well` refuses a move with the sentence that ends it; `on_shown` and `hideEvent` |
| `daphnia/settings.py` | `RECORD_MAX_FRAMES` (`DAPHNIA_RECORD_MAX_FRAMES`, default **300** ≈ 10 s ≈ 360 MB) |
| `assets/ui/study2.ui` | `btn_record` + `record_label` in group 4; the intro line names both captures |
| `daphnia/ui/hub.py` | `Study2Window.closeEvent` stores a running recording - the window is `WA_DeleteOnClose` |
| `tests/test_study2.py` | `ScriptedReader` (a reader that is attached, without a socket) + `RecordingTests` (14 tests) |
| `docs/requirements/SR-STUDY2.md`, `README.md`, `docs/streaming.md` | `SR-STUDY2-013`, `-014`, `-N06`; the loop paragraph and the diagram; the env var |

**Measured, because the design rests on it** (M4, offscreen, the rig's own 1280x720 frame):
`cv2.imwrite` **17.2 ms**, `sharpness` **2.5 ms**, **1241 kB/frame** → about **51 frames/s**
of headroom in the main thread against the 30 fps this stream delivers. That is why the
frames go to disk **as they arrive** and only `(index, path, sharpness)` stays in memory:
ten seconds is 360 MB of files, and holding the arrays as well would be that memory a second
time, unbounded from the operator's side. **Not measured:** the same cost on the lab's Mac,
on a nearly full disk, or while a page of wells is drawing its frames.

**Deliberate choices.** The **rows are written when the recording stops**, so a take in
progress is invisible to every other surface instead of half-stored; what a stopped
recording leaves behind is a normal burst that Study, Dataset and the Bench read unchanged.
A recording has three endings besides *Stop* - the frame limit, a dropped stream, the page
going off screen - and each **names itself** on the status line (`SR-STUDY2-N06`): a guard
that truncates a take in silence is exactly what the ML laws forbid (`mlContext.md` law 8).
The page is **locked to its well** while a recording runs (plate, study, page size, the
walk, *Re-read the plate*, and the grid), because the rest of the take would otherwise land
in another well's folder - and the plate keeps its colours, so "where am I" survives the
lock. `_record_frame` is connected as a **bound method**, never a lambda: the shared stream
service outlives the page (`mlContext.md` §6, trap 2), so PySide6 has to be able to take the
slot off it. A recording is refused by the **same** gates as a burst - a plate and a well, a
study, an active scale, the connection - because it is stored by the same writer
(`SR-STUDY2-N01`).

**Tests:** `RecordingTests` drives the app's own chain (`service.start()` over a scripted
`StreamReader`, one `frame_ready` per frame) and covers: every frame stored as one burst,
files on disk while it runs with no rows until it stops, two takes of different lengths,
a burst and a recording sharing one well and one image, the live counter and the button, the
four refusals, the lock and the refusal to move, the frame limit, a dropped connection, the
page going off screen, and the window closing. Suite green: **1045 tests, 1 skipped**.

## Panel 1 rebuilt: load-or-start for a device *and* a scale (2026-10-20, twenty-fifth round)

**Asked, over four messages:** *"in calibration I'm not able to switch between devices … just
one string with a name … no extra columns … put Save at the end and add at the start two
buttons, Load and New"* → *"why is earlier calibrations not same as preset"* → *"make new also
a popup with input field"* → *"a calibration also needs to be loaded or created, so we need
same buttons etc from preset"* → *"setup input is duplicated now in popup and panel … display
text in a more formatted list way in panel not like now apart of sentences."* Confirmed
choices: **Load switches the active device**; panel 1 is a **read-only list of facts**; a
calibration gets its own **label**; every input is in a popup.

**The gap it names (verified in the code):** the only way the global `is_active` row ever
changed was `register_calibration` inside `_save_profile` - i.e. **storing**. Two
already-calibrated rigs could not take turns, and the one list on the screen (`combo_history`)
only *viewed* a proof photo.

| File | What changed |
| --- | --- |
| `assets/ui/calibration.ui` | panel 1 is now a read-only `QFormLayout` `facts_form` (`Preset`, `Setup key`, `In force`, `Calibrations`, `Mount`, `Distortion`) with **four buttons**: `btn_load_preset`/`btn_new_preset` (`Load preset...`/`New preset...`) and `btn_load_calibration`/`btn_new_calibration` (`Load calibration...`/`New calibration...`). `profile_name` is a **read-only, frameless** `QLineEdit` (name entered only in a popup); the `Earlier` combo, `label_footnote` and the loose sentence labels are gone |
| `daphnia/db/schema.sql`, `daphnia/db/connection.py` | new `calibration_profiles.label TEXT` (+ `COLUMN_MIGRATIONS` entry) - a calibration's own note |
| `daphnia/db/repository.py`, `daphnia/vision/calibration.py` | `register_calibration(..., label=None)` writes it; `OpticalProfile.label` reads it |
| `daphnia/ui/pages/calibration.py` | `stored_preset_names()` (names only) + `load_preset(name)`; `stored_calibrations()` (this device's rows, `label\|date · px/mm · (in force)`) + `load_calibration(profile_id)` (activates **and** shows the proof photo); `new_preset(name)` / `new_calibration(label)` cores with `_new_preset` / `_new_calibration` popup handlers; `_refresh_calibrations()`; `_in_force_text` returns **list values**, not sentences; `_save_profile` writes the pending label; `_confirm_scale` covers storing and both loads |
| `daphnia/ui/hub.py` | `SurfaceWindow` connects the page's `workflow_changed` to `status_footer.refresh`, so the Optics window's own bar re-reads the scale the page just switched |
| `tests/test_calibration.py` | the preset pair, the calibration pair, `load_preset`/`load_calibration` (activate, newest, store-nothing, refuse, hand-entered, missing photo), `new_preset`/`new_calibration`, the label reaching the stored row, the compact values, the read-only `Preset` |
| `README.md`, `docs/requirements/SR-OPTICS.md`, `docs/ui-calibrate-redesign.md`, `memory-bank/systemPatterns.md` | the surface description and the panel bullet; `SR-OPTICS-006` → **partial (selection shipped)**, `SR-OPTICS-008` → **implemented**, `SR-OPTICS-031` → **revised** (loading a calibration now switches) |

**Deliberate choices.** A **device** and a **scale** get the same gesture - load or start.
`New` leaves what is in force **alone**: naming a device (or a calibration) is not switching to
one. **The two pairs share no state.** `_pending_label` (the note `New calibration...` arms for
the next store) is cleared by `new_preset`, `load_preset` **and** `load_calibration`, because the
operator hit exactly that bug - "load and new does not work together" was a note typed for one
device being written onto another device's row. The `>2×` guard fires on switching too
(`SR-OPTICS-008`) - the same 5.3× move that burned this project, made deliberate with both values
named. The choosers are `QInputDialog`s **inside the button handlers**; the `load_*`/`new_*`
cores take a value, so no test meets a modal. No data-model change beyond the `label` column:
the store still keeps one active row, so `SR-OPTICS-006` is *partial*, not done.

**Measured:** `make ui-check` green (12 canvases, both schemes); full `discover` re-run below.

## S1 of the Calibrate redesign, implemented (2026-10-20, twenty-fourth round)

**Asked:** *"start working directly on implementing new changes to calibration as described in
handoff and memorybank"* — i.e. `docs/ui-calibrate-redesign.md` §5's file-by-file S1 plan, with no
schema change, no distortion model and no new target.

**Built** (the handover's own file list, plus the two docs that describe the screen):

| File | What changed |
| --- | --- |
| `assets/ui/calibration.ui` | three group titles/order per §3.2 (`1. Optical chain`, `2. Target photo`, `3. Scale in force`); **added** `btn_use_frame`, `btn_open_photo`, `btn_measure_grid`, `btn_store_scale`, `label_result`, `label_drift`, `label_field_stop`, `label_verdict`, `label_distortion`, `label_footnote`; the `Fixed` form row became the `Mount` footnote; `status_label` now starts at `Ready.`; every other objectName kept (renames were explicitly ruled out). `label_locked` was **deleted** — see the deviations below |
| `daphnia/ui/pages/calibration.py` | menu group `Calibration` (*Measure grid*, *Store this scale*) with the panel buttons wired to the same two methods; the page's own *File*/*Source* tools removed (a choice stays beside its panel, D8/D9); `_refresh_reference` split into `_refresh_grid_state` / `_refresh_result` / `_refresh_scale`; the result block reads `LatticeEstimate`'s **fields** (never `warnings()` as a paragraph); `_in_force_text` three cases + `RE-VERIFY`; header facts `PHOTO`/`STATE`; the route vocabulary (`SOURCE_SPAN`/`SOURCE_GRID`) named once; `_uncertainty_text`/`_uncertainty_advice` so the weak hint never sends the operator from the grid to the span; the refusal's reason → the verdict's tooltip |
| `daphnia/ui/widgets/image_canvas.py` | `set_field_stop(centre_x, centre_y, radius)` + `field_stop` + a dashed amber circle painted after the lattice, with the lattice's exact lifecycle (cleared by a new frame and by `clear_points`) |
| `daphnia/ui/theme.py`, `assets/theme/daphniascope.qss` | the `verdict_ok`/`verdict_warn` token pair in both schemes + one `QLabel#label_verdict[tone=…]` block; the tone is re-polished by the page, so a second verdict changes colour |
| `daphnia/vision/lattice.py` | the estimator's own caveat wording: "**the drift**, measured from the same bands" instead of "lens distortion, measured" (`SR-OPTICS-032`/D7), and `warnings()` now says the panel reads its fields |
| `tests/test_calibration.py` (43 → 60), `tests/test_image_canvas.py` (146 → 152) | +17 and +6 cases: the three in-force cases, the stale `RE-VERIFY`, the span route naming the tolerance it omits, the grid route *not* advising the span, the drift as its own value, the four verdict states (incl. the refusal reason in the tooltip), `lens distortion: unmeasured` on screen and nowhere else, the header facts, and the field-stop circle drawn (pixel-counted, and proven to fail with the paint call removed) |
| `README.md`, `daphnia/ui/help.py`, `docs/requirements/SR-OPTICS.md`, `scripts/render_ui.py` | the surface's description, the in-app walkthrough and the one requirement rationale now use the shipped words; `make ui-render` renders the daily act in **both** schemes and prints the result block, verdict and distortion line |

**Measured by rendering the real window** (`make ui-render`, then *looking* at
`recordings/ui-render/{light,dark}_CalibrationWindow_measured.png`): `Measured on this photo: one
printed cell = 1 mm.` · `Period: 27.239 px/cell · 10 bands · tilt -9.0°` · `Drift: ± 5.1 % across
the field (centre → edge)` · `Field stop: paper circle (601, 266) r 448 px · 54 % of the frame` ·
`Verdict: WEAK - the field moves more than the print` (amber, bold, in both schemes) · `lens
distortion: unmeasured - no model for this preset` · `Scale uncertainty: ± 2.54 % from the detected
grid - the bands' own fit (2.49 %) plus the print's own 0.50 %` · `In force (any chain): 27.26
px/mm ± 0.29 % · … · RE-VERIFY`. The lattice **and** the paper's circle are drawn on the photo, and
the two long prose blocks were tightened because the panel's content (1077 px) exceeded the
viewport (823 px at a 900 px window): `Store this scale` went from 59 px below the fold to 29 px.

**Two deliberate deviations from the handover, both recorded in it:**

1. **§7's expected verdict was wrong.** It asks for `MEASURED` on the reference photo, but that
   photo's drift is 5.1 % against the print's 0.5 %, and §3.3's own copy is `WEAK - the field moves
   more than the print`. The copy (and `SR-OPTICS-036`) won; `MEASURED` is asserted on an
   undistorted sheet instead. A verdict that said `MEASURED` beside `drift ±5.1 %` would contradict
   the line above it.
2. **`label_locked` was deleted, `label_footnote` added as the footnote's caption.** D10 takes the
   `Fixed` row out of the form column; the "Fixed" caption *was* that form-row styling, nothing in
   code or tests read it, and the widget that carries the fact (`locked_label`) survives.

**Not run this round, and therefore not claimed:** the whole `make test` suite. Run green:
`tests.test_calibration` (60), `test_image_canvas` (152), `test_ui_smoke` (60), `test_theme` (33),
`test_surfaces` (23), `test_ui_files` (9) — 337 tests in one combined run — plus `make ui-check`
(12 canvases, both schemes, full screen and back) and `make ui-render`. `tests.test_theme` gained
`label_verdict` in `OBJECT_NAMES`, so the new screen's colour has to keep coming from the theme.

## The Calibrate surface, rendered and read (2026-10-20, twenty-third round)

**Asked:** *"from my perspective the calibration tool today is not usable. we are not able to add a
new calibration … Detect grid is a single button in menu header … calibration UI should [be] more as
you described … we now have multiple implementations of the word 'calibration' … find a new frontend
design flow that is more efficient and aware of how frontend works."*

**Built for this round:**

| Piece | What |
| --- | --- |
| `scripts/render_ui.py` + `make ui-render` | builds **every surface offscreen** (both schemes) and grabs PNGs to `recordings/ui-render/`, including the Calibrate screen **with a real photo measured**. Reads a *copy* of `zoo.db`. This is how a front-end change gets reviewed when there is no display - `make ui-check` only proves a screen *paints*. It reproduces the evidence below verbatim |
| `docs/ui-calibrate-redesign.md` | the **handover document** for the next agent: goal, the D1-D13 discrepancy table with the rule each violates, the target design with exact copy strings, requirements traceability (`SR-OPTICS-024…036`, `X-N09…X-N13`), a file-by-file S1 plan, the **code bar** (efficient, logical, no comment bloat, correct on the first run), definition of done, and S2/S3 designed but out of scope |

**What the renders proved (not inferred):** the store has **5 calibrations**, newest active
`27.2645 px/mm ±0.29 %` from the *clicked span*, agreeing with the grid's `27.239 px/cell` to
0.09 %; at rest the header says `No calibration yet.` **while** panel 1 says `In force for a
*different* preset … 27.3 px/mm ±0.29 %` (the empty name box yields `PROF_PRESET`); after
`Detect grid` the whole `LatticeEstimate.warnings()` paragraph is printed **twice** and elided in
the header; the panel says `Accuracy: +/- 2.54% from the detected grid - weak: click a longer
span, or use the detected grid.` - i.e. it recommends the route that produced the number **and**
the route that omits the print's own 0.5 % (the 0.29 % row); and the status text calls the drift
"**lens distortion, measured**" where `SR-OPTICS-032` says distortion is unmeasured and
`SR-OPTICS-036` says the *drift* is what is measured. Panel 3 sits at the scroll fold with the
*fallback* (hand-entry) as its bottom row.

**The conflict to resolve while implementing S1:** `X-N13` ("an action is a menu item") and
`SR-OPTICS-024` ("Save is panel 3's content") genuinely disagree on this screen; the handover
resolves it as **both** (panel button + menu item) and records the exception in
`systemPatterns.md §The calibrate surface`. Nothing was implemented *in this round*: no `.ui`, no
page, no schema was touched. → **Implemented in the next round** (see the section above), including
the one recorded deviation from this round's `§7` (`WEAK`, not `MEASURED`, on the reference photo).

## The documented §4.1 flow, measured against the others (2026-10-19, twenty-second round)

**Asked:** *"we would really like to use the library's full abilities and not self-code
functions … implement in new probe to see if it gets us close to the main probe."*

**Built: `scripts/discorpy_doc_probe.py`** (596 lines) - the flow the discorpy docs §4.1
*prescribe* for strong distortion + perspective (the 1.7 / [C2] pipeline), as written, on
our photo: `normalization_fft(img0, 10)` up front, `remove_points_using_parabola_mask`,
`group_dots_*_lines_based_polyfit`, `find_center_based_vanishing_points_iteration`,
residual 3.0, `util.unwarp_color_image_backward(..., pad=...)`, `losa.save_metadata_json`.
CLI: `--mask/--no-mask`, `--hor/ver-margin`, `--hor/ver-curviness`, `--rotate`,
`--grouping {polyfit,neighbour}`, `--dist-px`. Guards, two of which caught real runs: the
centre must be inside the frame, the unwarp must move the points, and a residual that
*rises* after unwarping is named.

**Answer: no, it does not get close on this photo** - but two of its steps are real wins,
and both are library calls the older probes were missing. Same photo, same session, rms:

| Run | lines | centre | residual after unwarp |
| --- | --- | --- | --- |
| `calibration_probe.py --coef 3` | 16+21 | (621.09, 313.28) | **0.679 px**, worst 1.283 |
| doc flow, docs' args, raw frame | 1+5 | - | refused |
| doc flow, docs' mask, loose grouping | 7+10 | (623.65, **-2139.68**) | 2.2 px, c0=1.81 (degenerate) |
| doc flow on the app's `lattice._prepare` output | 4+17 | (681, 291) | 10.9 px, c0=1.16 |
| doc flow, prepared + equalised, `--grouping neighbour --ratio 0.5` | 16+23 | (707.54, 305.00) | **2.631 px**, worst 16.13 |

1. **`prep.normalization_fft(img0, 10)` in front of everything fixes the slope estimate:**
   raw frame **+30.20 deg** (its search limit) → normalised **-8.95 deg**, true tilt
   **-9.00 deg**. So part of what the sibling calls its own preprocessing is a *documented
   library step*, and the example probe was missing it. With `--radius 11` the same call
   gives **28.252 / 27.977 px** per cell (the paper's 27.0) against 19.168 px at radius 9
   (**-32 %**).
2. The library's own **colour + padded unwarp** and **`save_metadata_json`** are the
   artifact and the record format the app would want (the sibling writes greyscale,
   unpadded, and its own JSON).

**What failed, and why it matters:** the parabola mask is a *border* trim, not a field-stop
fit - the ocular ring's arcs sit inside it, and a rectangle that cuts them removes paper
instead (3+5 lines against 4+17); and the docs' polyfit grouping **collapses on this grid**
(1+5 with the docs' parameters, 7+15 at the loosest, at which point the centre leaves the
frame and the residual *rises* after unwarping, 1.708 → 2.741 px). The same points with the
library's *other* grouping (`--grouping neighbour`) give 16+23 and the best doc-flow result.
Extraction is 687+461 points on the raw frame against 3240+3167 on a prepared one: **the
limit is the target and the frame, not the choice of flow.** Recorded in
`docs/calibration.md` §4.2 and pointed at from `README.md`; nothing in the app changed.

## The reference flow, on purpose (2026-10-19, twenty-first round)

**Asked:** *"let's deliberately simplify `scripts/calibration_probe.py` … my goal is that we
only use discorpy prebuilt functions. create a new probe that tries to stay as close as
possible to reference code."*

**Built: `scripts/discorpy_probe.py`** (477 lines against the sibling's 924). It is
discorpy's own line-pattern example - `losa.load_image` → `calc_slope_distance_*` →
`get_cross_points_*` (ratio 0.5, sensitive 0.1) → `group_dots_*` (library defaults) →
`remove_residual_dots_*` (2.0) → `regenerate_grid_points_parabola(perspective=True)` →
`find_cod_coarse` → `calc_coef_backward` → `unwarp_line_backward` + `calc_residual_*` →
`unwarp_image_backward` + `losa.save_*` - on the frame **as loaded**: no field-stop mask,
no line-weight equalisation, no lattice straightening, no scipy/skimage/cv2 arithmetic. The
only additions are a CLI, one output folder per photo (`recordings/discorpy_probe/<stem>/`)
and discorpy's own verdict. The two probes now differ in one axis - the prepared frame -
which is the comparison an operator wants. `scripts/calibration_probe.py` is untouched.

**Measured** (discorpy 1.7.0, scipy 1.18.1, this rig's `PROF_…_133641.png`):

| Run | Result |
| --- | --- |
| the photo as it is | **no model**: the slope/distance estimator is thrown to its **+30 deg search limit** by the ocular ring (true tilt −9.00 deg) and the cross-point call raises discorpy's "Row index is out of range" - the reference flow does not get past step 3. `--radius 11` changes nothing: **the library alone cannot read our photo, the preparation is what makes it readable** |
| a synthetic single-weight line grid, 1.6 % barrel | **passes**: centre (641.97, 354.23) vs truth (640, 360) = 6.1 px, residual after unwarping **0.204 px** worst / 0.034 mean (before: 9.010 / 1.386 px, 48.7 % over 1 px) |
| the probe's own corrected output | **degenerate**: centre (192.6, −101.3) *outside* the frame, `c0` = 1.884, residual 116.6 px |

Two library behaviours the flow does not survive as written, both now named by the script
instead of silently absorbed:

1. **`post.unwarp_line_backward` is a silent no-op on `float32` lines** - and
   `proc.regenerate_grid_points_parabola` *returns* `float32`. The line unwarp solves
   `rd = ru·fact(ru)` with `scipy.optimize.minimize`; at float32 the numerical gradient
   comes back non-finite, the optimiser stops at its start point and the function returns
   its input (scipy says so only as a `RuntimeWarning`). Measured: a point 400 px from the
   centre moves **0.00 px** in float32, **6.25 px** in float64 - so the "residual after
   unwarping" reads 9.01 px where float64 gives 0.20 px, i.e. the verdict is wrong on
   *every* input. The probe casts to `float64` for that one call (`_float64`, documented
   with the measurement). **The sibling is not affected**: its model goes through
   `correct_perspective_effect` (float64), measured 0.054 px on the same synthetic image -
   so the documented 0.511 px / passes on our photo stands.
2. **`post.check_distortion` cannot tell a degenerate fit from a bad one** - the probe
   therefore also requires the centre of distortion to be **inside the frame**, and says so
   instead of quoting a verdict about a model that does not exist.

Also documented in `docs/calibration.md` §4.1 and pointed at from `README.md`. Nothing in
the app changed; no test covers `scripts/` (as for its sibling).

## Is one scale enough? Measured, with discorpy (2026-10-19, twentieth round)

> **Follow-up the same day, from looking at `*_points.png`** (the operator's own
> check, and it caught two of my mistakes): *"we can see points falling onto the ocular
> ring surrounding the image … we need to crop the image"* and *"fails on top and on
> right edges … why are you using points instead of lines? the points seem very
> inaccurate"*.
>
> 1. **The mask was ~60 px too small.** A **fitted circle** measured against the raw
>    edges gives centre **(600.7, 265.6), radius 448.7 px** - verified within a pixel at
>    the left (x 156-163), the right (x 1044) and the bottom (y 713 at x=640). The
>    brightness-percentile mask answered radius 390 and the lattice-energy mask 381, so
>    real paper along the **top** (where the disc is *cut off by the frame*) and along the
>    dimmer **right** was being filled over: exactly the "fails at the top and right
>    edges". Now: Otsu threshold → largest region → circle fit that **ignores the frame
>    border** → disc eroded by a margin; every cross point outside it is dropped (the
>    ocular ring's arc of hundreds of points was being grouped as one enormous line).
> 2. **The lattice was left tilted 9 deg**, and discorpy's cross-point extraction samples
>    *tilted* profiles whose range is truncated to stay inside the image - **~114 px along
>    one side with no points at all**. The frame is now straightened before the fit;
>    measured slope afterwards: **0.00 deg** (the sign matters - the wrong one *doubles*
>    the tilt, measured -8.95 → -17.70 deg).
> 3. **On points vs lines:** the intersections are the weak part, and the *lines* are what
>    `calc_coef_backward` actually fits (a parabola per line, then its intercept). Two
>    measurements: `--select-peaks` (Gaussian fitting) takes the **oracle** from 0.264 px
>    to **0.056 px** rms - and on this photo fits **nothing** (the print is too noisy), so
>    it is a flag with its number, not a default; and the residual on this photo is
>    **3.18 px rms but 16.6 px on the worst line**, i.e. a few mis-grouped lines dominate -
>    extraction noise, not a smooth optical term.
>
> Net effect on the verdict: **it flipped to yes.** Following the *documented* flow
> (docs 3.3.4 - neighbour-search grouping, `remove_residual_dots(…, 2.0)`,
> **`regenerate_grid_points_parabola`**, `find_cod_coarse`) with **3 coefficients** gives,
> on this very photo: **0.511 px rms, worst line 1.223 px → passes discorpy's own
> sub-pixel check**, COD **(616.3, 316.7)** in the straightened frame → **(623.3, 313.5)**
> in the original, c0 = 1.014, and a local scale of **+9.75 % at r = 302 px**. My earlier
> "not yet" was my approximation (fisheye-style grouping, my own centre-finding, the
> regeneration step skipped), not the library's limit. `--edge-margin 30` also passes
> (0.667 px) with COD (613.1, 318.2) and **+8.09 %** → the model's *shape* is repeatable.
>
> 4. **The last pile-up was the paper's own edge.** The paper/filled-surround junction is a
>    *step* and any profile crossing it has a local extremum there, so points collected
>    along that boundary; they are now kept `radius + 2` px inside the paper
>    (`--edge-margin`). `*_points.png` is a clean regular lattice across the paper.
> 5. **A smaller `--radius` is worse, not better** (measured: 4 → 6.25 px, 6 → 1.61 px,
>    8 → 7.31 px, 9 → **0.51 px**), `--coef` decides whether the model passes at all
>    (3 ✓, 4 → 2.22, 5 → 1.08), `--perspective` moves the curvature ~2x, and `--dist-px`
>    (pinning the pitch to 27) is wrong for the documented grouping (COD +78 px,
>    residual 2.57 px).
> 6. **The distortion is real, and measured without any model:** cell spacing in 24-px
>    bands of the straightened frame is **26.0-28.0 px** per cell across a 40 mm field
>    (**±3.8 %**, up to **~5.7 %** centre-to-edge) = **76-114 µm on a 2 mm animal** - seven
>    times the paper tolerance. The model's *shape* agrees (positive, monotone with radius);
>    its *magnitude* is ~1.5-2x larger.
> 7. **The model fails a mirror — the acceptance test that needs no second photo.** *"ok then
>    what if we use a different photo which does have the circle at a different place in
>    image? will it still work? flip image?"* Measured on transformed copies: the
>    **preparation survives all of them** (circle r **448.7 px** identical under mirror,
>    vertical flip and 90° rotation; **418.5 px** with the disc clear of every edge; and the
>    alignment self-check picks the right sign — mirrored needs **+8.95°** where the original
>    needs −8.95°, because a wrong sign *doubles* the tilt), while the **residual balloons**:
>    mirrored **1.18 px**, upside-down **8.58**, 90°-rotated **4.07**, re-positioned **1.48**
>    (original 0.51-0.68). A mirror is an exact pixel permutation, so it is now a check in the
>    probe (`--mirror-check`, on by default) and the model **fails it**: coefficients differ by
>    **0.82**, COD **184 px** off the mirrored expectation, mirrored residual **8.68 px**.
>    Verdict: the *effect* (4-6 %) may be reported, the *coefficients* may not be stored. Also
>    fixed: the points overlay was drawn on the un-rotated photo although the points live in
>    the straightened frame (`*_points.png` now straightened + `*_points_original.png` mapped
>    back).
> 8. **"Why implement it again when the probe works? I want to completely replace our
>    calibration and detect grid."** — done, and the question's premise deserves an answer:
>    *the device detector is replaced; the probe is not a detector, it is the evidence tool* -
>    and the app needs one number the probe cannot give. `daphnia/vision/lattice.py` is now
>    the OPTICS estimator (`SR-OPTICS-035/036`, 14 tests in `tests/test_lattice.py`) and
>    `daphnia/ui/pages/calibration.py` calls it; `vision/grid.py` is superseded on that path
>    (kept, with its refusals and tests intact, and the probe still reports its refusal as
>    evidence). **What the app needed and the probe lacked is a *scale*:** the probe's §1b
>    prints **19.1 px and 28.2 px** for the same paper at two radii, and its grouping needs
>    `--dist-px` to be *told* the pitch - neither is storable. The band reader answers
>    **27.239 px from 10 bands (spread 5.10 %, statistical 2.49 %)** on the eyepiece photo and
>    **27.03 / 27.24** on three more, refusing the two ruler photos and three unusable ones by
>    name. Everything the probe proved is *in* the module, cited there: the Otsu + circle mask
>    (60 px too small without the frame-border exclusion), the surround fill, the alignment
>    (a projection search that cannot get the sign wrong - the probe's fixed-sign version
>    needed a self-check for exactly that), the amplitude modulation, and the two traps that
>    cost the most time: autocorrelation answering **13.4 px** for a **26-27 px** lattice, and
>    a printed line's profile splitting into two dips 4-12 px apart. The *distortion model*
>    deliberately did **not** move into the app: it fails the mirror test (§7), so the surface
>    reports the measured drift (5.1-6.3 %) and says the model is not storable.

**Reported:** *"use [discorpy] for the calibration … assess if it's possible to improve the calibration
step and use images of the 1 mm paper … i want to get rid of camera distortion in this step"*, and a
tool table (OpenCV/OpenCV.js, Discorpy, ImageJ/FIJI, Micro-Manager) to be **added to the memory bank**.

**The answer in one line:** discorpy is the right tool and it beats our detector at exactly the thing
our detector cannot do - but **printed graph paper is the wrong target for it**. That is the decision
to take now; the code is a day's work after it.

| Measured on this machine (full evidence: `docs/calibration.md` §3) | Number |
| --- | --- |
| `detect_grid` on the **nine** photos in `recordings/calibration` | **refused — all nine** ("too faint or too fine to measure") |
| discorpy's line families on the eyepiece photo | **19.168 / 18.160 px**, slopes **-8.95 deg / +8.61 deg** |
| **oracle**: discorpy's own single-weight 19-px line grid, known barrel (`[1,0,-1e-7]` = 4.8 % at r=400) | centre error **1.13 px**, residual **0.264 px rms** (worst line 0.771 px) → **sub-pixel** |
| discorpy on the *real* graph-paper photo (detection opened up, distance pinned) | centre (630.7, 288.5), 16 + 26 lines, residual **7.87 px rms**, worst line **20.03 px** → **not sub-pixel** |

**Why the photo fails and the synthetic does not.** Printed graph paper carries **three line weights**
(1 mm, 5 mm and 1 cm), and discorpy's line-pattern flow assumes **one evenly spaced family**: the
cross-point detector picks up whichever weight dominates locally, so the grouped "lines" are a mixture
of three lattices. Sixteen parameter combinations were swept; residuals stayed 2.6–16 px, and the two
2.6 px cases had a centre of distortion ~10^4 px outside the frame (fits that gave up, not fits that
found it). discorpy says this itself - *"check if there is any missing grouped line"* - which is why
that warning count is a printed number in the probe, not screen noise.

Four things worth keeping, each paid for by a measurement:

1. **Our detector refuses this rig's photos, and that is a finding, not a nuisance.** Two gates do it
   and neither is the paper's period: the **prominence** gate (a pale mint print inside a hard black
   field stop poisons the noise floor) and the **period refinement range** (a harmonic peak sends
   `_refine_period` to `None`). Filling the surround with the paper's median does not rescue it.
   *Consequence:* the distortion is **unmeasured**, and `distortion_pct`'s `0.0` default means "not
   computed" - a surface must never read it as "no distortion" (`SR-OPTICS-032`).
2. **The method is proven before the photo is blamed.** The script runs an **oracle** - a synthetic
   single-weight grid with a *known* distortion - before touching a real photo (`mlContext.md` law 2:
   a metric must be able to score a perfect model 1.0). Without it, "we could not fit the photo"
   would have been read as "discorpy is not good enough"; with it, the same code and parameters
   recover a known distortion to **0.264 px**.
3. **An eyepiece photo has a surround, and discorpy reads its edge as a line.** Measured: the grouping
   kept **1 line instead of 34** until the field stop was filled with the paper's own median. Reported
   field coverage (60.4 %) is marked a *caveat*, because the fill threshold also catches bright
   background on that frame.
4. **A model that fails its own check is not reported.** `post.check_distortion` is the bar (over 1 px
   on more than 15 % of the points = not good enough); the probe prints the residual and the verdict
   and withholds coefficients that missed it. The stored model would otherwise mis-scale every later
   measurement (`SR-OPTICS-033`), and a corrected frame's scale is a *different* scale that must be
   re-measured (`SR-OPTICS-034`).

**Added by this round:** `scripts/calibration_probe.py` (Qt-free, discorpy lazily imported, exit 0/1;
JSON + overlay + unwarped PNG into `recordings/calibration_probe/`); **`docs/calibration.md`** (the
task guide: the flow function by function, what the coefficients mean, the measurements above, the
three ways forward with their costs, and the tool table); `SR-OPTICS-032/033/034` + two open issues;
`discorpy` in `environment.yml` as an **optional pip dependency** with the licence and the
`pip install --dry-run` verification recorded in `techContext.md`; a README subsection; and the memory
bank (`techContext` tool table, `systemPatterns` correction of the "distortion is measured" claim,
`progress` issues 21-22). **Nothing in the app changed** - no surface imports discorpy;
the suite was run to prove it: **`make test` 1017 tests, 1 skipped, green** (the count in
`techContext.md` still reads 1002 - it was already stale before this round).

**Next (in order of what it costs):** (a) print a **single-weight line grid or dot grid** and re-run the
probe - that is the whole of `SR-OPTICS-033`'s blocker; (b) fix `vision/grid.py`'s gates so the
*scale* can be measured again from the paper the lab actually has (keeping the refusals that should
stay refusals); (c) check one pixel→µm reading against **ImageJ/FIJI**, the independent referee nobody
has used yet; (d) only then decide between reporting the drift, a local per-measurement scale, or
unwarping frames.

## A past calibration's proof photo is one choice away (2026-10-19, follow-up)

**Reported:** *"add abilty to chosse previous calibartaion and load the img of that calibration on
switch"*

Panel 1 gained an **Earlier** combo (`combo_history`) plus a `history_label`. Two new read-side
helpers carry it: `repository.calibrations(conn)` (every calibration, newest first, across all
setups) and `repository.proof_record(conn, profile_id)` (the `calibration_records` row, or `None`).
Choosing an entry calls `load_calibration_image(profile_id)`, which loads
`calibration_records.raw_image_path` through the existing `_set_reference` path.

Four decisions worth keeping:

1. **The list is not filtered by the name in panel 1.** The question it answers is "which photo
   produced the number on this animal?", and that calibration may belong to a chain the operator has
   since renamed; a list filtered to the current name also empties itself exactly when a preset is
   renamed, which reads as a broken control. An entry shows date + px/mm + `(in force)`, and the
   tooltip carries `profile_id`, preset name, setup key and accuracy - what a 340 px panel has no
   room for (the combo also wears Predict's `AdjustToMinimumContentsLengthWithIcon`, the documented
   fix for a combo deciding a panel's width).
2. **Choosing is a view, not a switch.** Nothing in the database changes: the calibration in force
   stays in force (`test_looking_at_a_past_calibration_changes_nothing_in_force` asserts the active
   profile id before and after). Adopting a scale is *Save this calibration*, and the status line
   says so rather than leaving the operator to guess.
3. **No photo is a real answer.** A hand-entered calibration (`SR-OPTICS-029`) has no
   `calibration_records` row, so the panel says "no proof photo … entered by hand" instead of
   leaving whatever image was on screen - which would have been the silent substitution the schema
   exists to prevent. A proof photo that has since been deleted is named by the canvas refusal,
   which carries both the file and the reason ("… gone.png … the calibration in force is
   unchanged").
4. **The combo holds no mode.** It *does* one thing, so the selection returns to its prompt after
   every rebuild (i.e. after a save, `_refresh_all` → `_refresh_history`), which also means the
   label can never end up describing a photo that is no longer on screen.

**Measured:** `tests/test_calibration.py` **36 → 43 tests** (list order + the in-force mark, the
load itself, the view-not-a-switch rule, hand-entered, a missing file, a rename not hiding the
history, and the return to the prompt after a save); `systemPatterns.md`, README and
`SR-OPTICS-031` updated with it. Full suite **1002 tests, 1 skipped, green**; `make ui-check` paints
all 12 canvases in both themes.

## The calibration page is a name, a grid size and a scale (2026-10-19, nineteenth round)

**Reported:** *"update calibartion page to be simpler. no need for adding microscop or zoom. just a
input for a name of preset. i also need to be able to set scale manually and be able to set grid
size"*

Three changes, and the first one removes more than it adds:

| Was | Is |
| --- | --- |
| **1 Optics**: microscope, objective (combo `2x`/`4x`/`10x`), camera, and a read-only "locked: zoom 1.0x, focus at infinity"; `setup_key = new_profile_id(scope, objective, camera, 1.0, "infinity")` | **1 Preset**: one `QLineEdit` - the name. `setup_key = ids.new_setup_key(name)` = `PROF_<NAME>` (`SR-OPTICS-004`); the hardware columns stay `"unknown"`, and zoom/focus stay the code's constants |
| The grid was always **1 mm** (`detect_grid(frame)`, `GRID_PITCH_UM` in `span_um()`) | **`spin_grid_mm`** (0.001-100 mm, default 1.0): `detect_grid(frame, pitch_um=...)`, the hand-clicked span and the stored `target_type`/`actual_distance_um` all read it (`SR-OPTICS-030`) |
| The scale could only be **measured** (detected grid, or a clicked span) | **`check_manual_scale` + `spin_manual_px_per_um` + `spin_manual_accuracy`**: a typed scale wins over the measurement, is labelled `hand-entered`, and is stored as a **claim** (`SR-OPTICS-029`) |

**What the honesty rules cost, and where they were paid.** A typed scale gets **no
`calibration_records` row**: `raw_image_path` is NOT NULL because a *measured* calibration has a
proof photo, and inventing a path for a number nobody photographed would be the fabrication the
schema exists to prevent - so the record is not written at all and the panel says there is no proof
image. Its accuracy is **stated by the operator** (the spin starts at
`settings.CALIBRATION_WEAK_UNCERTAINTY_PCT` and is visible), never derived: the alternative was to
store `0.0` and let the footer print `+/- 0.00%`, which is the false precision `SR-OPTICS-005`
exists to prevent. A typed scale of `0` is *not* a scale, and the panel asks for the number instead
of saving one.

**The grid size is one field, used in four places, and the direction is easy to get backwards.**
`grid_pitch_um()` feeds detection, the clicked-span arithmetic (`squares x pitch`), the stored
`target_type` (`calibrated_grid_500um`) and `actual_distance_um`; changing it re-scales the
detection **already on screen** (`dataclasses.replace(estimate, pitch_um=...)`) instead of asking
for a second photo. The first version of the test asserted the wrong direction and the run caught
it: a square declared **0.5 mm** makes the scale *larger* (147.3 px / 500 um = **0.2946 px/um**, not
0.0736), because the same pixels now span half the distance - the failing assertion was the test's,
not the code's, and the comment now states the physics.

**Measured, not asserted.** `tests/test_calibration.py` went **24 -> 36 tests**. The old
`assets/ui/calibration.ui` rows are **gone, not hidden**: `test_there_are_no_hardware_fields_left`
asserts `microscope_model`, `objective` and `camera_device` are not attributes of the page at all.
The panel still fits the shared `CONTROLS_WIDTH` (`test_ui_smoke` measures every page's content
against its panel), `make ui-check` paints all 12 canvases in both themes, and the full suite is
**995 tests, 1 skipped, green**.

Also updated with it: `daphnia/ui/help.py` (the walkthrough: name the preset, set the grid square,
*already know the scale?*), the Help menu item (*How to calibrate (grid paper)*), README's
calibration-rule section, `systemPatterns.md` (the calibrate-surface table), and `SR-OPTICS.md` -
`-004`, `-012`, `-016`, `-019`, `-024`, `-025`, `-027` restated, **`-029` and `-030` added**.

## One store, one dataset convention, no "archive" (2026-10-18, eighteenth round)

**Reported:** *"i would like to merge dataset entry point with the bench entry point ... in bench we
are missing: folder and video import, ability to create new dataset and choose from existing"* — then,
asked which surface's store should win: *"remove term 'archive'; all pages must use same dataset
storing conventions so it can be accessed across entry points"*, and *"unify everything, Study
included"*.

The whole application now writes **one** kind of row. `measurements` is no longer a table: a
measurement is a **`corpus_labels` row**, whichever surface made it - the Bench, the Dataset
labeller, or a governed study capture (which additionally carries `observation_id`/`asset_id`).

| Piece | What it is |
| --- | --- |
| **The one store** | `corpus_labels` (a measurement) + `corpus_boxes` (an individual). `save_corpus_label` is the single write path; `repository.save_measurement` is a thin adapter that resolves the frame, the dataset and the governed provenance, then delegates. |
| **`measurements` is a VIEW** (`MEASUREMENTS_VIEW`) | `SELECT label_id AS measurement_id, …, corpus_labels.rowid AS rowid FROM corpus_labels WHERE observation_id IS NOT NULL`. Every reader written against the old table - `queries._PROGRESS`, `list_studies_with_progress`, `maintenance.prune_discarded`, the annotate/analytics screens - keeps working **unchanged**; only the write path moved. |
| **The container** | `datasets` gains `study_id`: a dataset may belong to a governed study (`dataset_for_study` creates it, `make_current=False`), and `is_current` is the one open dataset every entry point shares. |
| **A box carries its scale** | `corpus_boxes` gains `profile_id`/`px_per_um`/`observation_id`/`asset_id`, so its size in µm is derivable exactly as a label's `length_um` is (`SR-BENCH-015`). |
| **The corpus can say "I do not know"** | `corpus_labels.image_width`/`image_height` are nullable: a labelled frame records its size, a row whose file is not on disk (an archived study, a test's placeholder) has none, and NULL says so. |

**The migration is in place, non-destructive, and measured** (`db/connection.py`): `migrate()` runs
the legacy measurement migrations while `measurements` is still a table, then
`unify_measurements_into_corpus` copies every row into the default dataset (reading each frame's size
from its **header**, so no OpenCV is needed - `SR-CORE-011`), renames the table to
`measurements_legacy` (a backup, never dropped) and creates the view. Measured on a copy of the live
`zoo.db`: **28 measurements folded, 519 → 547 corpus labels, 0 lost**, `measurements_legacy` keeps
all 28, and a second `migrate()` applies nothing. A row whose frame is missing is **counted and left
in the backup**, never silently dropped.

**The Bench gained what only the Dataset had** (`SR-BENCH-018`…`-021`): a **dataset selector** at the
top of the panel (`combo_dataset` + *New dataset...* + *Rename...*, the Dataset surface's own
`DatasetDialog`), **Import a folder...** (recursive) and **Import a video...**, and a walkable queue
(`A`/`D`, the arrows, a double click). A video is not a file you can measure, so its frames are
**materialised** to `datasets/raw/<stem>/f000123.png` by the new Qt-free, OpenCV-lazy
`daphnia/vision/video.py`; every import is **recorded** in the new `import_batches`/`imported_frames`
tables (source, per-frame `frame_index`, `ordinal`, and the resume pointer `last_ordinal`), so
re-importing the same folder is the same work.

`daphnia/vision/images.py` gained `header_size()` (stdlib PNG/JPEG dimension reader) so a frame's size
is readable **without OpenCV** - which is what the migration and the import queue need, and what keeps
the app startable without the CV stack.

Tests: `tests/test_db.py::SchemaTests::test_a_legacy_measurements_table_is_folded_into_the_corpus`,
`test_measurements_is_a_view_over_the_corpus`, `tests/test_bench.py::BenchDatasetTests` (3),
`BenchImportTests` (4, one `skipUnless(cv2)`). The full suite is green (981 tests, 1 skipped).

## A tool can be turned off, and the model's answers are listed in the panel (2026-10-18, seventeenth round)

**Reported:** *"i want also to be able to disable a tool not just switch between. when no tools are
disabled we are in normal view where we can select objects i want to list predictions in measurments panel
and be able to click on them to make measurements in the panel. same as when i select a prediction we make
it a measurment the notice goes away. this way will be able to faster make predictions and add as
measurments"*

Two changes, both in `daphnia/ui/pages/bench.py` + one in the canvas.

| Piece | What it is |
| --- | --- |
| **The normal view** (`X-N20`) | Unticking the armed draw tool - by clicking it again **or** pressing its key again (`CanvasWorkspace.set_tool_shortcuts(..., toggle=True)`, `_activate_tool(action, toggle)`) leaves **no tool armed**, and the page gives the canvas the *Select* tool's own behaviour (`BENCH_NORMAL_VIEW = TOOL_SELECT`): a press selects, drags, resizes or `Backspace`s - and **draws nothing**. The old handler put the click *back* on the tool it tried to leave ("a radio group with nothing selected would leave the press undefined"), so there was no way to be in a view where a click only selects |
| **`X-N19` is now conditional** | The tool-follows-the-object switch fires only when a draw tool is armed; in the normal view a press selects and **arms nothing**, or the normal view would last exactly one click |
| **The inbox** (`SR-BENCH-017`) | The results tree opens with *From the model* - one row per prediction, `Prediction N` + `box 107 x 89 px, length 46 um   click to add it` - and a click adds that animal: its **box** as the individual and its **keypoints** as a measurement inside it (`accept_prediction`, `_on_result_clicked`). The row and the layer entry go together, the rows renumber, `Backspace` on the canvas takes its row too, and switching the model off empties the group. The page's copy of the answer follows the canvas through `prediction_removed` alone - never dropped twice |

**A real bug the inbox found, and fixed.** `ImageCanvas.drop_prediction(index, part="all")` returned early
whenever one of the parts was already absent - so a **box-only** prediction (the detector answered, the pose
model did not) could **never** be removed from the layer, by any caller. It survived because no page had ever
asked for `part="all"`; the inbox is the first caller that does. Now each part is cleared on its own and the
removal is decided afterwards. Its regression test is `tests/test_image_canvas.py::ToolTests
::test_part_all_drops_a_box_only_prediction`. While there, a **dead duplicate** of the same method (a 2-arg
`drop_prediction` shadowed by the 3-arg one) was removed - unreachable code that reads like an overload.

**A real defect in my own first cut, found by measuring.** Accepting each animal from the panel gave
*Individual 1, 3, 5, 7, 9*: the box and its length each consumed a row slot, and the individual's number was
read off that slot. The **individual's number is now derived** - its position among the boxes, 1..N, in the
order they were drawn, like the grouping itself - while `specimen_index` stays the row key's business
(`(asset, specimen_index, trait)`). So a box row can read `specimen=3, individual=2`, and that is the two
questions the split exists for.

**Measured, on a real frame with the real models** (`models/rtmdet_daphnia.pt` + `models/rtmpose_daphnia.pt`,
`datasets/kp/images/b01_f00_35755477_ee94ef9c.png`, MPS, 2.0 px/µm in a throwaway DB):

```
RTMDet rtmdet-tiny + RTMPose: 5 animal(s), 5 with keypoints, 416 ms
From the model      5 to add
    Prediction 1   box 107 x 89 px, length 46 um   click to add it      ... 5 rows
[0 individual(s), 0 measurement(s) in this view - not saved yet.  5 from the model still to add.]
5 clicks -> 10 objects in this view:
    Individual 1   box 53.7 x 44.6 um = 2395 um^2   0 measurement(s)
    Individual 2   box 55.4 x 40.1 um = 2219 um^2   1 measurement(s)
        Length  36.8 um
    Individual 3   ... 1 measurement(s)      Length 35.9 um
    Individual 4   box 22.3 x 24.1 um =  537 um^2   2 measurement(s)
        Length  18.6 um | Length  35.2 um
    Individual 5   ... 1 measurement(s)      Length 46.2 um
normal view: canvas.tool='select'   two clicks on empty space -> 10 objects (drew nothing)
stored: bbox specimen=1 individual=1 | length specimen=2 individual=5 | bbox specimen=3 individual=2 | ...
```

**What that run also shows, and it is not the rule's fault.** Two of the five lengths are listed under an
individual **other than the box they came with**: this frame has **three overlapping detections of one
animal** (107x89, 83x69 and 45x48 px boxes), and the rule is *midpoint, innermost box wins* - so a length
whose midpoint falls in the smallest of two boxes belongs to it. That is `SR-BENCH-016` behaving exactly as
stated, and it is the detector/corpus that is being reported: the fix is to drop or move the duplicate box
(one `Backspace` on the canvas), not to add a second attribution rule. Recorded as a known behaviour rather
than smoothed over.

**Not measured:** nothing about the models - no new accuracy claim, no re-training, and the pass cost is the
same (416 ms for 5 animals on MPS). `make test` **976 tests, 1 skipped** (this round adds 16: 4 for the
normal view, 8 for the inbox, 2 for the individual numbering, 2 for the canvas bug); `make ui-check` green
(12 canvases, light and dark, full screen and back).

## A box is an individual, and the measurements inside it are its own (2026-10-18, sixteenth round)

**Reported:** *"on predict entry point we display as animals found with their size measurements, not the
box size measurements. remove ability to measure a box and replace with a similar mechanism where the box
itself is an 'animal' or individual. we would be able in bench for example to detect when line or keypoint
fall inside a drawn box and list all measurements inside that box so we could not only see length but also
measure different attributes. and do that for multiple boxes and see a list of all measurements per
individual"* - then, choosing what a box becomes: **"still stored with its width/height/area as a bbox
row (today's behaviour, plus the grouping of measurements under it)."**

**The finding that made it small: the concept already existed three times in this repo.** Study has
`observations.individual_index` ("Individual 3", `schema.sql:70`); the corpus is *one box = one specimen*
(`corpus_boxes`, `mlContext.md` §10); Predict has `vision.tracking.Track` - a box that stands for one
animal and keeps its number and colour. **The Bench was the one place it was missing**: a box was a
measurement, and a line's only identity was `len(self.session) + 1`, so two lines on one animal were two
unrelated "specimens" and nothing could group them. Predict, notably, *already* does what was asked
(`predict.py:1186`: `"{count} animal(s) found"` and the animal's measured length per row, never the box's).

| Piece | Where | What it is |
| --- | --- | --- |
| the rule, pure Python | `daphnia/vision/individuals.py` (new) | `Individual`, `midpoint`, `contains`, `individual_of`, `group`. **Midpoint** of the measurement (an animal's length line often starts on the box edge), **innermost** box wins, **source pixels**, `None` when no box contains it |
| the rule's proof | `tests/test_individuals.py` (new, 19 tests) | hand-computed: containment incl. borders, nesting, the midpoint's two directions, a stray line under `None`, the order inside a group |
| the surface | `daphnia/ui/pages/bench.py` | `_individuals()`, `_individual_index(item)`, `_grouped()`, `_group_title/_group_value`; `_refresh_results` builds a **`QTreeWidget`** (two columns) - one group per box, *No individual* last |
| the widget | `assets/ui/bench.ui` | `QListWidget list_results` → **`QTreeWidget list_results`** (Individual / Measurement) |
| the record | `measurements.individual_index INTEGER` (`schema.sql`, `COLUMN_MIGRATIONS`, `X-N06`) | **whose** number this is; `NULL` = outside every box. Saved by `repository.save_measurement(..., individual_index=...)`, read by `queries._BENCH_SELECT`/`samples_for_observation`, exported in `BENCH_COLUMNS` |
| the contract | `docs/requirements/SR-BENCH.md` (`SR-BENCH-005`, `SR-BENCH-015`, new **`SR-BENCH-016`**), `00-framework.md` (**`X-N21`**, and `X-N20` written down - it was cited in code but missing from the matrix) | |

**The one thing that would have been a silent data loss, and is not.** `save_measurement` dedupes on
`(asset_id, specimen_index, trait)` - so if a line's `specimen_index` had *become* its individual's index,
two `length` lines on one animal would have overwritten each other with nothing said (law 8). The fix is
the **two-question split**: `specimen_index` stays *which row on the frame this is*, `individual_index` is
*whose number it is*, and the row key is untouched - so an individual holds two lengths and both survive.
Proved at the store (`BenchStorageTests`) and through the page (`BenchIndividualTests`).

**Measured, on a real frame with the real models** (`models/rtmdet_daphnia.pt` + `models/rtmpose_daphnia.pt`,
`datasets/kp/images/b01_f00_35755477_ee94ef9c.png`, MPS, 2.0 px/µm registered in a throwaway DB):

```
RTMDet rtmdet-tiny + RTMPose: 5 animal(s), 5 with keypoints, 458 ms
accepting two distinct animals' boxes -> Individual 1, Individual 2 (each a bbox row, individual_index = its own)
accepting prediction 1's keypoints    -> a length inside Individual 1
this view, grouped:
   Individual 1   box 55.4 um x 40.1 um = 2219 um^2   1 measurement(s)
       length  36.8 um   -> individual 1
   Individual 2   box 42.6 um x 39.3 um = 1678 um^2   0 measurement(s)
stored: bbox specimen=1 individual=1 | bbox specimen=2 individual=2 | length specimen=3 individual=1
```

The last row is the design in one line: the measurement's own slot is **3**, the individual it belongs to
is **1**.

**Two interaction facts found while driving it, both pre-existing and both reported rather than smoothed
over** (they are `X-N15`/`X-N17`'s documented rules, not defects):

1. **The aim decides what the release makes.** A press inside a box that **two** detections cover promotes
   the *other* one (`_hit_prediction`), and a press where a line or keypoint already lies edits **that**
   (`X-N15`) - so a driver that presses the same spot twice does not accept two animals. Three of the five
   detections on this frame are one animal detected three times, which is what made it visible.
2. **A press inside the box that is currently selected moves it** (`X-N17`), so placing a measurement
   inside a box you have just drawn needs the selection gone first - one key (`L`/`K`, which drops the
   selection, `X-N18`) or a click elsewhere. Not changed: the rule is deliberate.

**Not measured / not done:** no new accuracy claim - the model's numbers are unchanged (`docs/rtmpose.md`
§3, §8: held-out median 227 µm, p90 2.8 mm; val = one capture). "Measure **different attributes**" is by
hand today: `study.TRAITS` is `length`/`width` and the corpus holds **0** side keypoints, so a *model*
cannot produce width until a `sideL`/`sideR` labelling pass exists - a corpus task, not a code path. A
*points-only* leftover in the prediction layer (its box accepted, its keypoints not) is a small dead spot
for a press that aims at neither - noted, not fixed. `make test` **960 tests, 1 skipped** (this change adds
30: 19 in `tests/test_individuals.py`, 11 in `tests/test_bench.py`); `make ui-check` green - the tree does
not clip the Bench panel (12 canvases, light and dark, full screen and back).

## The tool follows the object you click (2026-10-18, fifteenth round)

**Reported:** *"no actually i want when clicking on a object to switch to that object mode. when in
selection mode and selecting a object or when in another mode and one clicks on another object for that
to switch tools"*.

**Built (`X-N19`).** A press on an object now switches the page's tool to the one that **owns** it - a
stored box or a predicted box → *Box*, a measurement → the measuring tool, a model's keypoints →
*Keypoints* - **in any mode**. The switch is deliberately non-destructive:
`ImageCanvas.set_tool(tool, keep=True)` leaves the press's own work alone (selection, drag, open
points), and `ImageCanvas.object_pressed(tool)` tells the page which tool the object wants; the page
(`Bench._on_object_pressed` → `_switch_tool_quietly`) ticks the action **silently** and refreshes the
prompt, so one click both selects the object and arms the right tool for the next one.

Two follow-on fixes this forced into the open, both measured:

| Symptom found while testing | Cause | Fix |
| --- | --- | --- |
| a Select-mode press on a prediction **accepted it** | the tool switched to *Box* mid-press, and the "is this the Select tool?" check was read *after* the switch | the tool is captured **before** `object_pressed` is emitted |
| a dragged prediction made **two** measurements | the switch to *Box* armed `box_changed`, which fired *as well as* `prediction_accepted` | a promoted prediction is reported **once**; `Dataset` now stores its label in `_on_prediction_accepted` (reusing `_on_box_changed`) |

Measured: clicking a box's outline from Line mode leaves tool `box` **and** the box selected; clicking a
measurement from Box mode leaves tool `line`; a dragged prediction makes exactly one measurement.
**921 tests OK (1 skipped)**; `make ui-check` green.

## Predictions are two objects, boxes are grabbed by their line, tools are one key away (2026-10-18, fourteenth round)

**Reported:** *"when we click on one prediction box, the keypoints inside disappear. if we click on
keypoint, the box disappears but also all other keypoint predictions. i want them to be treated all
individually and not connected"*, then *"box should not have click-with feature but one has to come
close to the line"*, then *"when in box mode we can't switch quickly to line or point mode ... think
about how photoshop or other ui based image editors use tools"*.

**Causes (measured, real mouse events, three predictions in the layer):**

| Symptom | Cause | Fix |
| --- | --- | --- |
| "click the keypoint, the box goes too" | a prediction was **one** object `(box, points, label)`, and accepting either removed the whole item | `drop_prediction(index, part="box"｜"points"｜"all")` (`X-N17`): a model's answer is **two independent objects** (RTMDet's box, RTMPose's keypoints); accepting one keeps the other |
| "all other keypoint predictions disappear" | **double drop**: the canvas dropped the moved draft *and* `Bench._on_prediction_accepted` dropped again on a shifted index | the canvas no longer drops; the **page** drops exactly the part it used (`Bench` emits nothing itself; `Dataset` drops the box only when `moved`, keeping `SR-DATASET-027`'s "a click stores nothing") |
| "clicking a box swallows the objects inside it" | `box_at` hit the whole **interior**, and a measurement box is drawn *around* its keypoints | `box_edge_grab` (`X-N17`): a page whose boxes are measurements (Bench) grabs a box by its **outline** (`box_edge_at`); a press inside an **already-selected** box still moves it. The corpus keeps interior picking - there a box *is* the animal |
| "in box mode we can't switch quickly to line/point" | the tools lived only in the header menu | `CanvasWorkspace.set_tool_shortcuts` + `X-N18`: every tool is **one key** away and wears it in its text (`Line (L)`, `Keypoints (K)`, `Box (B)`, `Select (V)`), bound at window scope |

Measured after: with three predictions and a detection+pose answer, a **keypoint** click makes one line
and leaves that prediction's **box** (and both other predictions) intact; a **box** click/drag makes one
box and leaves its **keypoints**; the drawn box equals the measurement at the dropped position. **914
tests OK (1 skipped)**; `make ui-check` green.

## One selection, and a prediction that is made where you drop it (2026-10-18, thirteenth round)

**Reported:** *"in select i'm not able to select multiple objects. i would like also to be able to
select predictions. atm i'm not able to edit boxes properly"*, then *"since clicking on prediction box
creates it, this leaves behind a ghost and the box we drag around gets deleted as the ghost is the real
one ... i would like it just to be created on interaction ... so when we drag around the dragged-around
gets added"*, then *"also we are not able to delete boxes on backspace"* and *"when selecting a box from
predicted it disappears right away and the keypoints inside get added as measurements. i want if click
box only box to be added"*.

**Cause (measured, real mouse events).** Three separate faults:

| Symptom | Cause | Fix |
| --- | --- | --- |
| "cannot select multiple objects"; "cannot select predictions" | selection was one kind at a time (`select_box` cleared predictions; `select_prediction` cleared boxes), predictions had no additive path, and `_paint_predictions` **skipped** the selected one - so picking a prediction made it vanish | `X-N17`: one selection (`_selection` boxes + `_selected_predictions` + `_selected_annotations`), Shift/Cmd adds/removes for all three, the marquee covers all three, and a selected prediction is **highlighted, never hidden** |
| "ghost behind, the dragged box gets deleted" | the measurement was created on **press** at the suggestion's position, while `_finish_edit` then dropped the prediction and discarded the dragged box | `prediction_accepted(index, box, points, moved)` is emitted on **release**, and `Bench` builds the measurement from that **final** geometry, then drops the prediction |
| "cannot delete boxes on Backspace" | `box_delete_requested` was never handled on Bench (only Dataset did) | `Bench._on_box_delete_requested` → the same `_remove_from_view` as a line; and a **just-made object is left selected** so `Backspace` removes it at once |
| "keypoints inside get added as measurements" | a detection+pose prediction carries both, and the release always preferred `points` | `_prediction_points_at` records what the press **aimed at**: a press on a **keypoint** makes a line, a press on the **box** makes a box |

Measured: two shift-clicks select two predictions; the marquee takes a box + a measurement + a
prediction at once; one `Backspace` removes every selected prediction; clicking a prediction box adds
**one box** at the dropped position with **no** prediction left and the drawn box equal to the
measurement; clicking one of its keypoints adds a line; a freshly drawn box/line is already the
selection, so `Backspace` removes it. **910 tests OK (1 skipped)**; `make ui-check` green.

## A box is a measurement, and a clicked prediction becomes one (2026-10-18, twelfth round)

**Reported:** *"box is n ... since bench mode uses or is able to use rtmdet and rtmpose we would like
to be able to measure boxes (area and height width and also include as measurement as well as to be
able to edit boxes similar to line. we want also to be able to edit keypoints and adjust measurements.
as soon as a predicted box or keypoint gets clicked on, the object turns into measurement, we are able
to when clicked on, delete any object or measurement or predicted by using backspace. i want a clean
flow for each object we add on image canvas."*

**Decided with the operator:** a box measurement is a `measurements` row with `trait='bbox'`, `p1`/`p2`
the two **opposite corners** (so `length_*` is the diagonal and the `NOT NULL` distance columns stay
honest), plus **new** `width_px/height_px/area_px` + `_um` columns and a `COLUMN_MIGRATIONS` entry
(`X-N06`). The unrelated opening fragment "box is n" was **not** decoded - it is read as *"the Box
[tool] is not [yet able to measure]"*, which the rest of the message supports.

**Built:**

| Piece | Where |
| --- | --- |
| `measurements` gains `width_px/height_px/area_px` + `_um`; `COLUMN_MIGRATIONS` declares them; `save_measurement(..., box=...)` writes them; `_BENCH_SELECT` reads them | `daphnia/db/schema.sql`, `connection.py`, `repository.py`, `queries.py` |
| The canvas exposes **an annotation as a selectable object**: a press on a measured line selects it and `Backspace` asks the page to remove it (`annotation_delete_requested`) | `daphnia/ui/widgets/image_canvas.py` |
| Bench: the *Box* tool's drag **is** a measurement (`_on_box_changed`→`_add_box`), a view box is moved/resized in place (`_on_box_edited`), a measurement is deleted (`_on_annotation_delete_requested`, the view is renumbered), and a **clicked prediction becomes a measurement** - a pose pair a line, a detection-only box a box (`_on_prediction_selected`, `mlContext.md` law 7) | `daphnia/ui/pages/bench.py` |
| The view lists `Box n  W x H = Area  Bounding box` beside `Line n  L  Length`, and saving writes the box geometry | `bench.py::_value_text`, `_persist_view` |

Measured (real mouse events): a Box-tool drag puts **one** box in the view with `area == width*height`
and `width_um == width_px/px_per_um`; saving writes `trait='bbox'` with the geometry columns; a view box
dragged on the canvas updates in place and stores nothing; clicking a detection-only prediction adds a
box item, a pose prediction a line; clicking a drawn line and pressing `Backspace` removes it from the
view. **899 tests OK (1 skipped)**; `make ui-check` green.

## Bench loses its modes: a staged "this view", saved or lost (2026-10-18, eleventh round)

**Reported:** *"remove measure mode and specimen in bench ... currently bench results show all
results from all previous bench sessions but i want a new mode where we only see what's been added in
current view, line etc tools and what their measurement are. we can then save to bench archive"* -
and then *"one landmark at a time just does not make sense here in bench, it's needed in dataset ...
i want the bench to be the place where one can create measurements without predefined modes. it
should be possible to save a bench measurement for one photo to a study. if bench results are not
saved they should be discarded and the user must be notified that next time he opens bench the
results will be gone."*

**Cause.** Bench carried the same *Mode* (two points / one landmark) and *Specimen* dropdowns as the
corpus labeller, and every measurement was written to the database the moment its second point
landed. So the results list showed **every previous session's rows**, the thing being measured now
was buried, and a quick measurement was indistinguishable from archiving it.

**Fix (one change):** Bench is now **mode-free** - the canvas' *Tool* group (`X-N14`) is the only
"what does a press mean", and it opens in **Line** (`SR-BENCH-013`). Measurements made now live in
**`Bench.session`**, *this view*, listed in the panel with an explicit "not saved: if you leave the
Bench without saving, these measurements are gone" warning (`SR-BENCH-014`). Loading another image
(or taking a new live frame) **discards** them and says so. Two buttons save the view: **to the
bench archive** (`study_id = NULL`) or **to a study** (the same rows, photo attached in one action).
A view line is still editable before saving (`X-N16`, now sourcing from the view). One-landmark mode
stays on Dataset. `combo_mode`/`combo_specimen` and the whole pass/draft machinery are gone from
`bench.py`; the panel is *Image / Measure / This view*.

Measured (real mouse events): two clicks put **one** line in the view and **nothing** in the
database; *Save this view to the bench archive* writes the row (scale preserved) and clears the
view; saving to a study writes it and leaves the bench list; a new image discards the unsaved view
with a stated reason. **892 tests OK (1 skipped)** (the pass-mode tests were replaced by the view
tests); `make ui-check` green.

## Hover a target, click it, edit it - in any tool (2026-10-18, tenth round)

**Reported:** *"when near to another keypoint or spine or box or prediction (eg on hover) i want to be
able to click on it and edit right away. without having to [switch to Select]"*.

**Fix (`X-N15`).** The press logic was split so both the *Select* tool and a create tool ask the same
question first: `ImageCanvas._hit_editable` returns `("point", i)` / `("box", specimen)` /
`("prediction", i)` for whatever is under the cursor, front to back. `_start_edit` turns that into a
drag (resize handle, point move, whole-selection move, prediction → draft), and `_begin_edit` adds
the *Select* tool's marquee for empty space. `mousePressEvent` now runs `_start_edit` for a **create**
tool too, so a click **on** a keypoint/box/prediction edits it and only **empty space** runs the
tool's own gesture. On hover the canvas reports `ImageCanvas.hover`, draws an accent ring/outline
(`_paint_hover`) and sets a grab cursor; `leaveEvent` clears it. The point-drag branch keys off the
drag's own `grab` flag, not `box_edit`, so a grabbed point is a **move** (`points_moved`) in any tool
and never a placement.

Measured (real mouse events): on the Bench, clicking the just-placed A point **moves** it and stores
**no** line (one point stays on screen); hovering it reports `("point", 0)`; hovering a stored box
reports `("box", 1)` and sets `Qt.OpenHandCursor`; a *Line*-tool press on a stored box / a prediction
selects it and places no point; empty space still places a point. Full suite **895 tests OK (1
skipped)**; `make ui-check` green. One test that asserted the old rule ("without `box_edit` a click on
a box is still the page's own gesture") now states the new one - a box is grabbable without the
*Select* tool.

## One press, one meaning: the canvas is a tool, and Bench measures lines (2026-10-18, ninth round)

**Reported:** *"currently by default the edit mode is on. think of user flow and the amount of
buttons he has to press to get measurement. i dont like that we call measurement in bench head and
tail and that we only can measure once. i would like a tool bar like 'line' with A and B points ...
we want to be able to add multiple measurements and call it A_1 A_2 ... try to split ability of
edit into multiple. like select can be its own and in box mode we can edit boxes etc"* - and then
*"you forgot keypoint mode where we use those skeletons with spines but that should be its own mode
or tool. Do both together in one change"*.

**Cause.** Editing was a single boolean (`ImageCanvas.box_edit`), on by default on Bench, that did
five jobs at once (`_begin_edit`: resize a handle, drag a point, select/move boxes, promote a model
box, marquee). Because it was on, the first click on the Bench *selected* rather than measured, so
the operator had to open the *Measure* menu and switch *Edit* **off** before a single landmark
could be placed; and `max_points=2` meant one pair with a *Save* press between specimens, named
"head"/"spine base" on a surface that measures a distance.

**Fix, in one change:**

| Change | Where |
| --- | --- |
| An explicit **tool** - `line`/`keypoints`/`box`/`select` - with a `tool` getter that folds in the old `box_edit`/`box_mode` flags, a `set_tool`, and a `tool_changed` signal (`X-N14`) | `daphnia/ui/widgets/image_canvas.py` |
| In the **box** tool, empty space **anchors a corner** (so a box can be drawn *and* edited); only **select** marquees | `ImageCanvas._begin_edit` |
| A **skeleton** drawn from `KeypointSystem.edges` (bones, not a polyline through the click order) | `ImageCanvas.skeleton` + `_paint_points` |
| A `point_placed` signal: a **real** press, distinct from a programmatic `set_points` | `ImageCanvas.mousePressEvent` |
| **Bench** opens in **Line**: ends **A**/**B**, the line stored the moment B lands, A_1/B_1 → A_2/B_2 with no *Save* between them; a mutually-exclusive **Tool** group (`SR-BENCH-013`) | `daphnia/ui/pages/bench.py` |

Measured (real mouse events, a throwaway database): two clicks store **one** row and the next line
becomes **specimen 2**; `tool` is `line` at start and clicking *Select* flips `box_edit` and places
no point; a *box* corpus' empty-space press anchors a corner, a **stored** box's press still selects.
Full suite **887 tests OK (1 skipped)**. The stored trait stays `length` with the real scale, so a
bench line and a study number still mean the same thing (`projectbrief.md`, non-negotiable 1); the
corpus vocabulary is unchanged, so nothing under `mlContext.md` §3-4 moves.

## The press means one thing per view, a box is stored in the frame's pixels, and the readout is as wide as its words (2026-10-17, eighth round)

**Reported:** *"when moving boxes from prediction around, after clicking on 2 and 3 box the first
disappears again. also im not able to correct keypoint as it first selects box. also the info box
overlay is very small and needs to be as wide as image canvas"*

Three complaints, two causes found by measurement and one by arithmetic - and one symptom that was
**not** reproduced, which is stated rather than invented:

| Symptom | Cause (measured on a copy of the live corpus, real mouse events) | Fix |
| --- | --- | --- |
| "not able to correct keypoint as it first selects box" | in the **crop loop** the item's box covers the whole magnification, and with *Edit* mode on (the default) `_begin_edit`'s step 3 - *a stored box* - took the press: a real click at the centre of the crop left `canvas.points == []` and `selected_box == 1`, so the prompt's *"click the head, then the spine base"* did nothing at all | **`ImageCanvas.point_mode`**: while a keypoint corpus is shown as a crop, the press *is* the keypoint gesture - a point within `BOX_GRAB_PIXELS` is still picked up first (one press corrects a keypoint), any other press places the next landmark, and empty space is not a marquee. Set by `Dataset._apply_item_view`, cleared with the crop; a *box* is corrected in *Frame* view (`SR-DATASET-028`) |
| boxes changed under the operator while being moved | `Dataset._on_box_edited` stored the geometry the canvas reported **in the frame it was showing** as if it were source pixels, with the crop's own size as the image size: dragging specimen 1's box (100, 100, 200, 160) in the crop wrote `(82.3, 78.6) - (382.3, 258.6)` against a "450 x 330" image. The drawing, the queue item and the corpus then disagreed - the rectangle you moved is not the rectangle the corpus keeps | **`source_box(crop, box)`** next to `crop_box(crop, box)` (both corners through the one `Crop`, corners re-sorted) and **`Dataset._source_size()`** (the frame the row belongs to); the queue item takes the correction, so crop view, frame view and the corpus agree (`mlContext.md` law 1) |
| "the info box overlay is very small" | `ElidedLabel` keeps its words itself and paints them, so it never gave them to `QLabel` - and Qt's `sizeHint()` answered the width of *nothing*. `KpiStack.natural_width()` is that sum, and the strip is `min(natural_width, canvas - 2 * MARGIN)`: **120 px of card on a 740 px canvas**, a sentence elided to `"Click the head, then the spine..."` | **`ElidedLabel.sizeHint()`** returns the width of the full text (`minimumSizeHint()` stays `VALUE_MIN_WIDTH` so a narrow canvas still elides). Measured on the same page: **120 px → 701 px** on a 740 px canvas - the box is as wide as the picture, and a *short* fact still gets a small card |
| "after clicking on 2 and 3 box the first disappears again" | **not reproduced.** In the item queue, real clicks on boxes 2 and 3 left every overlay and every row in place (`canvas.boxes` unchanged, `selected_box` following the click); in the model pass, each dragged prediction is stored once, leaves the layer (by design) and the other predictions stay drawn | the crop-space rewrite above is the closest measured fault - *every* box touched in the crop was silently replaced by a wrong rectangle - and it is fixed. If the symptom survives, the next step is the exact click sequence: **which surface, which view (Crop/Frame), and what "the first" is** |

Measured before → after (headless, a throwaway database, the real page and real `QMouseEvent`s):

```
crop loop, Edit mode on, a click at the centre of the crop:
   before: points []                        selected_box 1   status "Specimen 1 selected ..."
   after:  points [(225.0, 165.0)]           selected_box None
a box dragged in the crop (boxes-only corpus, crop view):
   before: specimen 1 (82.3, 78.6)-(382.3, 258.6)   image 450 x 330
   after:  specimen 1 (102.4, 101.2)-(202.4, 161.2)  image 1280 x 720   (size kept, moved 2 px)
the Dataset readout on a 740 px canvas:
   before: natural_width 120   strip 120 px   (value elided)
   after:  natural_width 701   strip 701 px   (cap 716 = canvas - 2 x MARGIN)
```

Tests: `tests/test_image_canvas.py::EditAllTests` **+2** (`point_mode` places the keypoint where a
box would be selected; a point under the cursor is still picked up and dragged),
`tests/test_dataset.py::DatasetBoxQueueTests` **+2** (a **real** click in the crop stores the pair
in source pixels - the helper that placed points programmatically is *why* this was never caught -
and a box dragged in the crop is stored in source pixels at the frame's size, with the queue
following), `tests/test_canvas_workspace.py::StripTests` **+1** (a card is as wide as its words).
Full suite **875 tests, 1 skipped, OK**; `make ui-check` passes (12 canvases, light and dark, full
screen and back); `docs/requirements/SR-DATASET.md` gained **`SR-DATASET-028`**.

What this round deliberately does **not** do: the crop loop still *consumes the pair on the second
click*, so a **stored** pair cannot be dragged - the pair on screen is an overlay, and loading it
into the draft would draw the same two keypoints twice (the *"two duplicate boxes"* complaint), so
correcting a keypoint in the crop means clicking it again (the upsert `save_corpus_label` already
performs). And the item queue's *Frame* view still sites the current item's box as the active box
next to its stored one, because that view is where a box is judged and
`test_the_frame_view_draws_every_box_and_highlights_this_animal` pins it.

## No ghost, one box per animal, and letting go is the update (2026-10-17, seventh round)

**Reported:** *"i wnat to be able to edit model output. atm i see a ghost lacking behind and when
letting go box or point the new position is not updated. i only ever want one box and no model box
and readl box"*

Three complaints, three causes - two of them paint, one of them a signal contract:

| Symptom | Cause | Fix |
| --- | --- | --- |
| "a ghost lagging behind" | the prediction being edited stayed in the layer, so the dotted box at the **old** position was painted under the moving draft | a prediction is **not painted while it is the one being edited**, and a drag that turns it into an annotation **removes it from the layer** (`SR-PREDICT-018`) |
| "one box, no model box and real box" | the model's dotted suggestion for an animal the human had **already labelled** was drawn on top of the label: two rectangles for one animal | a prediction that a stored label **covers** (IoU >= 0.5) is **not drawn** - "one box per animal" |
| "letting go of box or point, the new position is not updated" | a *point* drag emitted **`points_changed`**, which means "a click **placed** a landmark": Bench's pass-mode handler took the move for a placement, walked to the next specimen and re-set the canvas' points - the point snapped back **mid-drag** | in *Edit* mode a point drag emits the new **`points_moved`** ("look again"), `points_changed` stays for placements; on the **drop** (`points_edited`) the page writes the correction **when that (specimen, trait) is already stored** - and nothing when it is not, because a half-placed pair is *Save*'s business |

Measured with real mouse events, on a copy of the live database:

```
drag the model's box, during the drag: editing prediction 3 -> not painted (no ghost)
after the drop:  stored {1: (746, 31)} | predictions left: 4 (the label is not a suggestion)
                 "Stored the box of specimen 1 (45 x 48 px) on b01_f00_...png."
Bench, drag a stored point (700, 400) -> (733, 416):
   during the drag canvas points: [(600, 300), (732.8, 416.4)]
   row after letting go:          (600, 300, 733, 416, 1)      # written by the drop itself
   status: "Specimen 1 Length corrected on b01_f00_....png - stored.  The specimen stays selected."
```

The two paint rules are asserted by **counting pixels** (`canvas.grab()`), because "I see a ghost"
and "I see two boxes" are complaints about paint, not about data: `EditAllTests._painted()`.

Tests: `EditAllTests` +4 (a prediction over a label is not drawn / over another animal still is;
no ghost behind the edited box; a point move is a move, never a placement), `test_dataset.py` +2
(letting go of a stored point writes the correction and stays on the specimen; letting go of an
unstored pair stores nothing). Full suite and `make ui-check` green.

Two things this round deliberately does **not** do, both measured and stated rather than assumed:
the **crop loop** (a keypoint dataset opened on an imported boxes export) has no correction
gesture - it consumes the pair the moment both ends exist and leaves no point on screen to drag,
so there is nothing to let go of; and `points_changed` is still the *placement* signal, so
Calibration and Predict, which never turn `box_edit` on, are untouched by the new contract.

## One click, one key, no frame jump, no duplicate box (2026-10-17, sixth round)

**Reported:** *"i have to click it like twice and press backspace multiple times till it disappears.
it also seems to randomly go to next frame. we also always see two duplicate boxes. is this
wanted?"*

**No, none of it was wanted** - one causal chain, four symptoms, all measured and fixed:

| Symptom | Cause | Fix |
| --- | --- | --- |
| "two duplicate boxes" | the page **sited the stored box a second time** as the *active* box (`_show_box`) while the selection was drawn on top of it - up to three rectangles on one animal | in *Edit* mode `_show_box` leaves the active box **empty**: the selection outline is the indicator (`SR-DATASET-027`) |
| "randomly goes to next frame" | a fast second click reached `mouseDoubleClickEvent`, which still emitted `double_clicked` → Dataset's **`next_frame()`** → `_take_back_click()` cleared the sited box and walked the queue | in *Edit* mode a double click is **two selections**: no `double_clicked`, so the queue-walk binding is unused (it stays for creation mode) |
| "have to click it twice" | the frame jump above threw the selection away, so it looked as if the first click had not registered | falls out of the fix above |
| "backspace several times" | the key only worked while the canvas held focus, and after a jump the selection was gone anyway | `canvas.delete_selection()` (public, one implementation) + **`Backspace`/`Delete` bound as page shortcuts** on Dataset, Bench and Annotate |

Measured with real mouse and key events on a copy of the live database:

```
labels: [1, 2, 3, 4] | frame index: 0
ONE click            -> selected: [1] | active box drawn: None | frame index: 0
ONE Backspace        -> labels: [2, 3, 4] | selection: []
a double click       -> selected: [2] | frame index before/after: 0 0
                        "Specimen 2 selected (100 x 104 px) - drag it, resize an edge, or press Delete"
```

Tests: three in `tests/test_image_canvas.py` (a double click in edit mode is not "next frame" and
*your* creation-mode gesture still is; `delete_selection` in one call) and one in
`tests/test_dataset.py` (edit mode does not draw the selected box twice, creation mode still sites
it). `tests/test_dataset.py`'s corner-click flow now states which mode it belongs to.

## Edit mode is automatic: click selects, Backspace deletes (2026-10-17, fifth round)

**Reported:** *"no what i want is a automatic edit mode. where one does not create a new box or
whateve is selected but one is able to edit the predictions and by clicking whin boundingbox one
slects it and is able to use back button to delete"*

The *Select* tool became ***Edit*** **mode, on by default** on Dataset, Bench and Annotate
(`ToolSpec("edit", "Edit", checkable=True, checked=True)` + `canvas.box_edit = True`), and the
pages no longer hide it: the mode decides what a click *means*, so it is always offered.
`Backspace` removes the selection, and the canvas now distinguishes three kinds of removal:

| Selected | `Backspace` |
| --- | --- |
| a corpus box (one or several) | `box_delete_requested` per box - the page deletes the label |
| a **model's** box | `prediction_removed` - nothing was stored, so the page drops it from the layer |
| a point | the canvas removes it and reports `points_changed` |

Three behaviour changes fell out of "automatic", all measured with real mouse events on a copy of
the live database:

```
Edit mode on by default: True | canvas.box_edit: True
click inside a box        -> selected: 1 | boxes unchanged: True       # nothing created
Backspace                 -> the label is gone from corpus_boxes
click a prediction        -> picked: 3, in the draft, stored rows: {}   # a click stores nothing
drag the same prediction  -> stored {1: (742, 29, 786, 77)}             # the drag does
Backspace on a prediction -> "Predicted animal 3 removed from this frame - 4 left."
Edit off, drag empty space-> a new box is drawn and stored               # creation mode
```

* a **click** on a model box no longer stores it (it used to report a completed box): only a
  **drag** does - "selecting is not accepting";
* a click on **empty space** in this mode is a **marquee** (or a deselect), never a new label -
  which is why the creation flow ("click a corner, click the opposite one") now lives behind
  `Edit` **off**: `tests/test_dataset.py` asserts both halves;
* `make ui-check` still paints 12 canvases (the modes are menu items, not panel widgets).

Tests: `EditAllTests` grew to 13 (a click stores nothing, `Backspace` removes a prediction, the
mode is automatic). The full suite is green (see `progress.md`).

## Every point, box and model box is editable (2026-10-17, fourth round)

**Reported:** *"i am not able to edit boxes of predictions and save as annotations. i want really
to be able to drag around, select or delete any point, box or seelction of multible"*

The *Edit* mode (`SR-DATASET-027`) now covers everything on the frame, and the order of the
questions in `ImageCanvas._begin_edit` is the order of what is on screen, front to back:

| Press lands on | What happens |
| --- | --- |
| a **handle** of the primary box | resize it (edge or corner) |
| an editable **point** | drag that point alone; `Delete` removes it |
| a **stored box** | click selects it, **Shift/Cmd-click adds to the selection**, dragging moves **the whole selection** |
| a **model box** | it is promoted to the draft and dragged: on the drop the page **stores it as an annotation** |
| **empty space** | a **marquee**: every stored box it covers is selected |

Four bugs were found and fixed while measuring, all with real mouse events:

1. **the whole selection moved only one box** - `_finish_edit` emitted `box_edited` per box
   while the page re-read the frame's boxes after each store, so the *second* box was compared
   against the geometry the first store had already replaced (specimen 1 moved, specimen 2
   snapped back). The moved geometry is now **captured before anything is emitted**.
2. **dragging a model box stored two labels** - the promotion used `set_box`, which *reports a
   completed box*, so the untouched prediction was stored as specimen N and the corrected one as
   N+1. The promotion is silent; only the drop reports.
3. the hit test returned the **first** box rather than the innermost one (the rule `box_at`
   documents).
4. the hit test crashed on a `QPointF`/`QRectF` argument - the helpers now read tuples, lists,
   `QPointF` and `QRectF` alike.

Measured on a throwaway copy of the live database:

```
boxes on the frame:      {1: (961,183,1101,281), 2: (635,392,735,496), 3: (224,228,329,312), 4: (222,388,284,461)}
click + Shift-click   -> selection [1, 2]
drag one of them      -> specimen 1 (961,183,1101,281) -> (1034,228,1174,327)
                         specimen 2 (635,392,735,496) -> (708,438,808,542)     # both stored
marquee over the frame-> selection [1, 2, 3, 4]
Delete                -> boxes left: {}                                        # all four removed
drag a prediction     -> stored {1: (788,278,872,375)}                          # one drag, one label
drag one keypoint     -> point 1 (761,27) -> (802,46); point 0 untouched
```

Tests: `tests/test_image_canvas.py::EditAllTests` (10) + `::BoxEditTests` (13).

## The model menus list the models folder, and loading one switches the tool on (2026-10-17)

**Reported (third round):** *"i am now not longer able to enable the prediction models form
menubar. now its alwas off. i want you to add a checkbox or a hacken to the one thats actiave
and show menu of availbe models in menubar by setting a deefault dir for models"*

Reproduced first, then fixed:

```
Detector (RTMDet)
   [ ] Load the active detector (RTMDet)      <- the load worked, nothing said so
       Open a detector checkpoint...
   [x] rtmdet_daphnia.pt   (in force)
   [ ] rtmdet_daphnia_last.pt

Model switch before: False     -> after the load: still False, predictions = 0
```

Three changes (`SR-PREDICT-019`):

* **loading a model from the menu switches the page's *Model* tool on**
  (`adopt_models` → `workspace.set_checked("model", True)` + `prelabel(announce=True)`). Asking
  for a model from the menu *is* the decision to use it; leaving the switch off made the menu
  look broken while the model was in fact loaded. Measured after: `Model switch: True`,
  5 predictions, the largest animal's points in the draft, the status line naming both networks.
* **the menus list what is in the models folder** (`models.checkpoints_in`, by name:
  `rtmdet*`/`rtmpose*`, `models.KIND_PREFIXES`), under a section header
  *checkpoints in models/* - registered or not. Clicking one loads **and** registers **and**
  activates it (`ModelMenu.activate` loads an unknown file first) - verified with
  `rtmpose_daphnia copy.pt`, which was never in the registry: registered, active, loaded, switch
  on, check mark moved. A file named after neither network is deliberately **not** offered
  (reading its bytes to decide would load PyTorch to draw a menu, `SR-CORE-011`), and a file
  that will not load says so by name (law 8).
* **check marks carry two facts**: the checkpoint **in force** (the registry's active row) and
  the top entry when **this window** has that kind loaded (`ModelMenu.loaded_paths`).

Also fixed on the way: `ModelRecord.label` called a pose checkpoint `rtmdet-tiny` (the preset
prefix was hard-coded); it now uses the kind's network name. And `ModelMenu.loaded` (the signal)
must not be shadowed by a dict - that naming collision broke every window for one compile step
(`loaded_paths` is the dict).

Tests: `tests/test_models.py::AvailableCheckpointTests` (4) and two in
`tests/test_ui_smoke.py::HeaderBarTests` (the folder listing + the switch following the load).


## The model is off by default; a box can be clicked, moved, resized and removed (2026-10-17)

**Reported (second round):** *"by default the prediction is off and is something we can turn on
or off. also i want it to be possible to always be able to click on a bounding box or inside of
it to select it and edit it. atm in most modes we create a new measurement or annotation but i
need to be able to have those tools available and usable."*

Two changes, both in the canvas and in the pages that own it:

* **the model is off by default** (`ToolSpec("model", …, checked=False)` on Dataset, Bench and
  Annotate): a model is an offer, not an uninvited hand in the draft. Verified: with a registered
  detector and a frame open, `predictions = 0` and the page's own status line stands; flipping
  the switch on runs the pipeline (5 animals, 5 keypoints, 31-45 ms warm on MPS) and puts the
  largest animal's answer in the editable draft.
* **boxes can be selected and edited** (`ImageCanvas.box_edit`, the *Edit* mode;
  `SR-DATASET-026`): `box_at` (innermost wins, tolerance), `resize_handle` (corners beat edges),
  `moved_box` / `resized_box` (size kept, frame bounds, minimum size), signals
  `box_selected` / `box_edited` / `box_delete_requested` / `prediction_selected`, and a painted
  selection (solid accent outline + 8 handles). A press that the editing layer uses **never**
  also places a point or opens a box.

Measured with real mouse events (`QTest`) on a throwaway copy of the live database, on the
`bb` corpus frame `b01_f00.png` (stored box (520, 562, 612, 637)):

| Gesture | Result |
| --- | --- |
| click inside the box | `selected_box = 1`, the panel's specimen follows, status names the size |
| drag its middle | stored row `(575, 599, 667, 674)` - moved 55/37 px, size 92×75 kept |
| drag its bottom-right corner | stored row `(575, 599, 711, 700)` - resized, top-left fixed |
| `Delete` | row gone, status says so |
| Bench, model on, *Select*, click the smallest predicted box | `selected_prediction = 3`, its points (733, 49)/(760, 23) in the draft, score 0.40 named |

Also fixed while testing: the hit test returned the *first* box rather than the innermost one
(the rule `box_at` documents), and the hit test crashed on a `QPointF`/`QRectF` argument - the
helpers now read tuples, lists, `QPointF` and `QRectF` alike.

Tests: `tests/test_image_canvas.py::BoxEditTests` (13: the maths, the gesture, the tool being
off, nothing-to-pick) and `tests/test_model_pass.py` (8, from the round before). `make ui-check`
passes; the full suite is running in the background (`/tmp/fulsuite2.log`).

## The model menus, the reports, and a load that is applied (2026-10-17)

**Reported:** *"loading the active detector I expect to see a prediction appear or at least
info overlay to show 0 above confidence, but atm all the model menubar options do
nothing."* Four separate defects, all in the load path, none of them a crash:

| Where | The defect (measured, not guessed) | Now |
| --- | --- | --- |
| `MainWindow` | `model_menu.loaded` was **never connected** (only `failed` was), so a load in the study shell registered the model and did nothing else at all | `_model_loaded` → `hand_over(page)` or a footer note |
| `Predict.adopt_models` | with no frame yet the load left **no trace on screen**: the header still had no facts, no readout card existed, the only change was a panel line | `_refresh_readout()` publishes the `Model` + `Status` facts at once |
| `Predict.adopt_models` | a **running pass kept the model it was handed**: after loading `rtmdet_daphnia.pt` over `rtmdet_daphnia_last.pt`, `page.detector` was the new one and `worker.detector` the old one while the bar named the new one | `_apply_loaded_models()` restarts the worker on the same source, re-detects a still frame |
| `Predict._require_detector` | a source chosen with nothing loaded silently loaded the page's **own default file** (`models/rtmdet_daphnia.pt`), not the registry's active checkpoint | loads `models.active(conn, "detector")`; with none registered it points at the *Detector (RTMDet)* menu |

Measured before/after (headless, real checkpoints, `z-app`, macOS/MPS):

* still frame, load the other checkpoint → scores move `[0.74, 0.64, 0.63, 0.56, 0.38]` →
  `[0.72, 0.66, 0.60, 0.40, 0.38]` (the picture is re-detected, not re-scaled);
* running video, load another checkpoint → `worker.detector is page.detector` **True**
  (was `False`, i.e. the boxes came from the previous file);
* score threshold 0.99 → the overlay card reads `0 found  NONE ABOVE THE THRESHOLD`;
* source first, nothing loaded → loads the registry's `rtmdet_daphnia_last.pt` (5 animals
  found), and with no active row it says *"No detector is registered yet. Load one in the
  Models menu…"* rather than inventing one;
* Calibration window, study shell and hub → the footer note names the checkpoint and its
  trained date.

Not measured: nothing about accuracy or latency changed here - this is the load and
publish path. `tests/test_predict.py`, `tests/test_models.py` gain `SR-PREDICT-017`; the
suite has **not** been run (the operator asked for no test runs this session), so the
existing Predict tests are the first thing to run next.

## The menu bar is the application's own — one order, and the panels lost their duplicates

**What was asked:** move the Predict window's model-loading and source-select actions out of
the sidebar into *Models* / *Source* menus, fix the Predict panel width - then: *"update as
many side bars and menu bars as possible, create a logical menubar for the app."*

**The order, fixed once in `header_bar.MENU_ORDER`** (the app's menus are built empty, in
this order, in `HeaderBar.__init__`, so no window can invent another reading):

```
File | <the page's own menus> | Source | Models | View | Go | Help
```

* **File** - what the window opens/exports (the page's `FILE_MENU` group sorts in here) and
  *Exit* (`QKeySequence.Quit`, `QuitRole`, `closeAllWindows()`).
* **the page's own menu** - *Measure* (Bench, Calibrate, Annotate), *Label* (Dataset),
  *Frames* (Predict) - inserted **directly after File** (`_insert_menu`).
* **Source** - the app's checkable *Connect the live feed* (the check mark *is* the
  `StreamService` state, same socket as the footer) + the page's sources under a separator.
* **Models** - `ModelMenu`/`daphnia/models.py` (unchanged this turn).
* **View / Go** - the window's, per page on screen.
* **Help** - *How to calibrate (grid paper)* (`CalibrationHelpDialog`) + *Where the data
  lives* (`daphnia.paths` paths, new).
* `hide_empty_menus()` hides a menu with nothing behind it; `render_tools` **empties**
  *File*/*Source* before refilling, so a step inherits nothing from the last one (this was
  a real bug found by sweeping the seven study steps: every step showed Dashboard's
  project entries). The same sweep found the second one: `View` was not emptied for a step
  with no canvas, so Acquire/Select showed *Calibrate's* canvas actions - now
  `_mount_view_menu` clears it (`view_menu(None, None, rebuild=True)`) and those two steps
  offer no *View* at all.

**The rule that decided every move: an action is a menu item, a choice stays beside its
panel.** Removed from panels (→ where they went):

| Page | Buttons that left the panel | Now |
| --- | --- | --- |
| Calibrate (Optics) | *Use current frame*, *Open a photo...*, *Save calibration*, *? How to set up calibration* | Source, File, Measure, Help |
| Bench | *Open a photo...*, *Use live frame*, *Next keypoint*, *Refresh list*, *Export bench results (CSV)* | File, Source, Measure, Measure, File |
| Dataset | *Open an image...*, *Import a folder...*, *Open a boxes export...*, *Export dataset*, *Use the live frame*, *Next keypoint*, *Reload queue* | File ×4, Source, Label ×2 |
| Annotate | *New specimen*, *Reload queue* | Measure ×2 |
| Dashboard | *New project...*, *Open this project*, *Close & consolidate...*, *Prune discarded frames* | File ×4 |
| Visualize | *Export measurements (CSV)* | File |

Kept in their panels, deliberately: a **combo with the button that applies it** (Dataset's
dataset + *New*/*Rename*, Bench's study + *Attach*, Setup's host/URL + *Connect*),
Predict's transport and settings (play/stop/scale/tracks - a control panel, not a list of
commands) and each **workflow step's** own controls (Acquire's *Capture*, Select's
*Pick sharpest*), which the step bar gates.

**New mechanism: `ShellPage.menu_tools()`** - a canvas-less step (Dashboard, Visualize)
declares menu items with the same `ToolGroup` a canvas page's `workspace` exposes;
`MainWindow._follow_header` reads it when the page publishes no workspace. That is what let
the project lifecycle and the CSV export leave their panels.

**Verified without the suite (asked not to run or write tests this turn):** a script that
builds every window and sweeps the seven study steps prints the bar per step (Dashboard:
*File* = the four project entries; Calibrate: *File* = *Open a photo...*, *Measure* =
*Detect grid*/*Save this calibration*, *Source* = both sources, *View* present; Visualize:
*File* = *Export measurements...*), and `make ui-check` reports **ok: 12 canvases painted in
light and dark, full screen and back** - both `assets/ui/*.ui` and the pages are in step.

**Left undone, and known:** `tests/` still names the removed widgets
(`test_bench.py:185-202`, `test_dataset.py:250-307`, `test_calibration.py:154`,
`test_ui_smoke.py:229-231`, `test_predict.py:117`) and
`test_ui_files.py::test_a_screen_with_a_duplicate_name_would_lose_a_control` loads
`annotate.ui` expecting the two `btn_new_specimen` widgets that are now gone. The suite has
**not** been run since these moves.

## Current focus

**The pose half of the top-down pipeline exists: detector box → two keypoints.**
`daphnia/vision/rtmpose/` is a sibling of `rtmdet/` with the same five-file shape
(`model`, `losses`, `data`, `train`, `detect`), and `daphnia/vision/pose_corpus.py`
reads the keypoints `datasets/kp` already carries. Train with
`make train-pose`; the task doc is `docs/rtmpose.md`.

- **The corpus dictated the shape, not the plan.** `datasets/kp` declares
  `pose_full_boxes` (4 keypoints + box) but the manual pass placed **only head and
  tail** on **384** usable specimens across **4** captures (`b01..b04`), with 141
  box-only rows and 1 degenerate double-click. So the model is **2-point** —
  head → tail *is* the endpoint (`projectbrief.md`) — and the sides wait for their
  own corpus. `PoseCorpus.warnings()` says all of this out loud.
- **The group is the capture, recovered from the frame stem** (`b01_f00_...` → `b01`),
  because the export flattens every frame into `images/` and its own split hashes the
  copy path — `Export`'s directory group would collapse to one value and
  `leak-safe` would be false. `PoseCorpus.regroup()` cuts captures deterministically
  and, with only four, **cannot** starve validation the way the hash-bucket regroup
  did (every capture landed in `train`, val came out empty — the trainer now refuses
  that, which is how it was caught).
- **SimCC head** (arXiv:2107.03332, RTMPose arXiv:2303.07399): each coordinate is a
  1-D distribution over `input_size * 2.0` bins, Gaussian soft target, `KLDivLoss`,
  `argmax` decode. `simcc_split_ratio = 2.0` is mmpose's own default.
- **The metric is stated before the model** (`mlContext.md` §9.2): median/p90
  keypoint error in **px and µm**, PCK@k, an OKS with *our* sigmas, and the task's
  own number — the head→tail length error in **µm**. `pose_metrics` is pure, so the
  **oracle** (truth in → 1.0) and **blind** (→ 0.0) cases are unit tests.
- **Verified**: `make test` **677 tests** green (`z-app`); a 3-step smoke run on MPS
  trained and evaluated end to end (median error fell 51 → 43 px, length error
  1473 → 586 µm over 3 steps — the plumbing is exact, the weights are not yet a
  result); `--inspect` prints the corpus and its warnings; `--evaluate` scores a
  checkpoint; the inference object returned source-pixel keypoints for a real frame.
- **Visible on the Predict surface** (`python run.py predict`): **Load detector** now
  loads *both* models (`models/rtmdet_daphnia.pt` + `models/rtmpose_daphnia.pt`,
  overridable with `DAPHNIA_RTMDET` / `DAPHNIA_RTMPOSE`, and `Open pose model...`);
  each detected animal gets its box, a dot on `head`/`tail`, the head→tail line, and
  **its measured length written above it** (live feed, video file and single image
  alike); the **Keypoints** toggle in the bottom-left *Frames* cluster switches the pose
  pass off and on; the **2. Scale** group uses the app's own `active_profile`, names it
  with its `µm/px` and its uncertainty, and flags a **WEAK** or **STALE** calibration
  (with a manual µm/px fallback that says it is *not* verified) - and switching the
  scale **re-labels in place, without re-running the models** (`SR-PREDICT-012..016`).
  `docs/screenshot-predict-length.png` is that view on a real frame.
- **A real 2-point run exists** (`models/rtmpose_daphnia.pt`, 250 epochs, best epoch
  **135**, `--split-by group`, `models/rtmpose_report.json`): on **one held-out capture**
  (101 samples, `b03`) - median keypoint error **3.2 px** (head 2.2, tail 3.9), p90
  **37.5 px**, PCK@5px **0.67**, OKS **0.56**, head→tail length error median
  **3.0 px = 227 µm**. Good on most animals, badly wrong on a tail — and one validation
  capture cannot resolve small differences (`mlContext.md` §6 trap 8).
- **The live calibration is STALE, and the surface says so.** `zoo.db`'s active profile
  (`…IPHONE_1.0X_INFINITY_20261015_211938`, 74.84 µm/px, ±0.50 %) was verified 29 h ago
  against a 12 h limit, and a **second** profile in the same database differs by **11×**
  (0.147 vs 0.0134 px/µm) — the scale drift `productContext.md` recorded. Both are one
  click apart in the Scale combo, which is why that combo exists.
- **Two facts measured on the real checkpoint** (2026-10-17, MPS): inference is
  **deterministic** — six identical passes on one frame gave 5360.0 µm every time
  (spread 0.0 µm), so the frame-to-frame wobble is the animal and the photograph, not
  the code; and on the corpus' own box for one specimen the model read **5360 µm vs the
  expert's 5362 µm** — on a **train** frame, so it shows the plumbing is exact and is
  *not* an accuracy claim (that is the held-out 227 µm above).
- **Still not wired** (deliberate): `vision/keypoints.py` keeps raising
  `NotImplementedError`, and `pipeline/derive.py` does not consume the pose model — the
  surface shows it live; a *study* number, and any combined uncertainty, still needs
  that step.

**The Dataset surface can work through the boxes that already exist - one animal at a
time, in either of two views, with Yes/No per box.** (`SR-DATASET-022`, `-023`, `-024`,
`-025`.)

- **`Open boxes export...`** opens `BoxesExportDialog`: the **boxes file**
  (`boxes.csv` / `labels.json` / the export folder) **and the folder its images are in**
  (suggested from the file, never guessed). This was the missing step - the first
  dialog opened in `datasets/annotations` and could not take a folder at all, and there
  was no way to say "the CSV is here, the pictures are there". `read_boxes(source,
  images_dir=...)` resolves an explicit image folder first (an explicit statement beats
  a convention), so `datasets/bb/images` can be picked as-is. Last file remembered.
- **One item = one animal**, and **two views of it**, switched with the `Crop`/`Frame`
  tool or `C` (sticky across animals; **Frame** is where a detection-only corpus opens,
  **Crop** where a keypoint corpus does):
  - **Frame** = the whole frame with **every** stored box drawn, each in its specimen's
    colour and numbered, the current animal drawn **solid** on top. That is the
    bounding-box display: it is how a box gets judged at all.
  - **Crop** = the animal alone, magnified (`vision/crop.py`: context margin, integer
    factor, nearest-neighbour so the enlargement is exact).
  Both are *views*: the frame stays the coordinate space. `_crop_point` inverts through
  the `Crop` when one is in force and is the identity when not, so the stored pixels are
  the same either way (verified: clicks in crop coordinates stored **exactly** as frame
  pixels).
- **Yes (Y) / No (N)**: Yes rewrites `source='model'` to `source='manual'` and names the
  accept (law 7); No deletes the box. A prediction is never left unanswered - storing
  keypoints on one does not advance the queue. A point outside the animal's own box is
  reported and stored ("may be a click on a neighbour").
- **No wall for a keypoint-only corpus**: the *button* asks whether to create the
  dataset whose model has a box and opens the boxes there in one click
  (`switch_to_box_dataset`). The **dialog-free core never asks** - a modal window in the
  code a test or a script drives hangs the caller (paid for: it hung the whole suite).
- **The skeleton is first class** (`SR-DATASET-023`): `KeypointSystem.edges`,
  `AnnotationSchema.export_edges()`, an `edges` block in `labels.json`.
- **Two corpus defects fixed** (`SR-DATASET-024`): the exported `boxes.csv` header had
  16 names for 15 values (every column from `image` onwards was shifted for any CSV
  reader; the trainer never saw it because `Export` reads `labels.json`); it now writes
  `file` = original path and `image` = exported copy, and `read_boxes` **refuses** a file
  whose header and rows disagree. The live-frame filename collision is fixed too.
- **Two UI defects fixed, both found by looking at a screenshot of the real window**:
  `Clear points` and `Clear frame` were drawn **on top of each other** (a `wide` tool
  landing in the half-filled row), and hiding a tool left its **empty cell** behind in
  the cluster. `ToolCluster.relayout()` now places the visible tools in order - and
  cannot ask Qt, because `isHidden()` is true for every child of a window that has not
  been shown yet.
- **Verified**: `make test` **626 tests** green (was 552), `make ui-check` paints all
  six canvases, and the surface was **rendered offscreen on the real 526-box corpus in
  both views** (`/tmp/shot_frame.png`, `/tmp/shot_crop.png`): 526 animals queued,
  "Animal 1 of 526", frame view drawing boxes 1/2/3 with 3 active, no overlapping tools.

**The measuring controls now live on the image, and every canvas works full screen.**
The *Zoom -/Fit/Zoom +* row left the four ``.ui`` files and became two expandable
clusters **on the canvas itself** (`X-N11`): a *View* cluster at the top right
(*Zoom +*, *Zoom -*, *Fit*, *Pan*, *Full screen*, with the live magnification in its
header) and a page-filled measuring cluster at the lower left (*Measure* on Bench,
*Label* on Dataset, *Save/Clear points/Prev/Next* on Annotate, *Detect grid* on
Calibrate). **Full screen** (`X-N12`) hides the controls panel, the live preview and
the window chrome - the step bar and the footer in the Study shell - keeps the readout
on the picture, and leaves with `Esc`. Built **once**, in
`daphnia/ui/widgets/canvas_tools.py` and `canvas_workspace.py`, and shared by all five
canvas surfaces, which is what keeps five entry points from growing five toolbars.
Verified by `tests/test_canvas_workspace.py` (25 cases, including a **pixel** test that
the translucent panels really paint over the photograph) and by an offscreen
doing-what-the-user-does run of all five windows: paint, full screen, back -
`make test` 552 green.

**A video file is now readable frame by frame on the Predict surface**
(`SR-PREDICT-010`, `SR-PREDICT-011`): `←`/`→` and the *< Frame* / *Frame >* tools on
the image step exactly one frame - forwards or backwards - with the frame number in
the readout. The tools and the keys exist **only** while a video file is open, which
is what `workspace.set_visible(key, applies)` is for: with the live feed the arrows
keep panning the canvas. Proven against a real mp4 through the real inference worker.

**The detector trains, runs live, and has been measured once.** `python run.py
predict` is a fifth surface — the **Localize** step made visible — fed by a trainer
that reads *only* the exported corpus folder: `scripts/train_rtmdet.py` (RTMDet in
plain PyTorch, MPS/CPU, `make train`).

**First real run (2026-10-16)**: `rtmdet-tiny`, 158 training frames from 10 captures,
30 epochs, ~4 min on MPS. At the operating threshold (score ≥ 0.35) it reaches
**precision 0.61 / recall 0.83 / F1 0.70** for ~5 boxes on ~3.6 animals, AP50 0.73,
**15 ms/frame** inference (≈68 fps). The run also found two bugs in my own code (a
metric that compared two different coordinate frames, and a Qt connection that
outlived its widgets) — both fixed and pinned by tests. See *Recent changes*.

That closes the loop the Dataset surface was groundwork for
(`SR-DATASET-015..018`): label boxes → export → train → watch. The keypoint model
(RTMPose) is deliberately **not** next-in-line for this surface; boxes are its
prerequisite, and pose has its own corpus.

**Datasets are one corpus per model, and the annotation system is schema-driven.** A
*dataset* (`daphnia/annotation.py` + the `datasets` table) fixes what a label *is*:
**two keypoints** (head → tail), **four** (head, tail, sideL, sideR — the default),
either **plus a bounding box**, or **boxes only**. So "train another model" means
*create another dataset*; a corpus's labels are never re-purposed, and the same frame
can carry ground truth for two models at once.

**Two corpus defects were found by building the detector, and both are fixed:**

1. **11 of 526 boxes left their image** (`y1 = -9.8`, `y2 = 724 > 720`) — a drag that
   crossed the letterbox. The canvas now clamps a box corner to the image
   (`ImageCanvas._to_image(clamp=True)`), and `scripts/repair_boxes.py` +
   `maintenance.clamp_corpus_boxes()` repaired the existing corpus in place,
   non-destructively and dry-run first. The live `zoo.db` and `datasets/bb` were
   updated: 11 rows clamped, 0 degenerated.
2. **The export's split leaks.** It hashes the *image path*, so **11 of 14 capture
   sessions** appear in train *and* validation; sibling frames are near-duplicates and
   the score measures memorisation. The trainer therefore defaults to
   `--split-by group`, which re-partitions by **capture** (deterministically), and
   `Export.warnings()` says so out loud. The durable fix is still upstream
   (`SR-DATASET-010`).

**The annotation tool's multi-individual behaviour was verified, not changed:** one box
per individual with `specimen_index` 1..4 per frame (223/145/117/41 boxes), auto-advance
on store, the index drawn on every box. Two additions were made anyway — the clamp
above, and a **duplicate-box warning** (a box overlapping a stored one above IoU 0.5
names the specimen it may be duplicating; it is still stored, never dropped).

**Splitting the one-pipeline app into intent-based surfaces over a shared backend, and
freezing that intent in per-surface system requirements** so the project cannot drift
from `O1..O4` / `D1..D3` (see `projectbrief.md`) remains the larger arc. Five surfaces
are built now.

Trigger: on 2026-10-15 the system was tested end to end. Capture and calibration
worked and produced usable readings, but a single "new project" triggered a
cascade of decisions, and the three stakeholder groups want three different
things right now (see `productContext.md`).

## Recent changes

- **2026-10-17 (latest) — the canvas readout is a column of KPI cards** (asked: "in the
  image canvas i would like to separate the info card into multiple cards. i want it to be
  subtle swiss design. identify primary kpis that make sense to display individually").
  The readout was *one* translucent plate holding a `"\n"`-joined paragraph
  (`set_readout`), which is why the number being measured had to be found *inside* the
  sentence - a joined sentence is not a readout. Now:
  - **`Kpi` → `KpiCard` → `KpiStack`** (`daphnia/ui/widgets/canvas_tools.py`). One card per
    fact: a **micro-caps heading** (written in sentence case, upper-cased by the card, so
    the page's own wording stays in the data a test reads back), the value, an optional
    **caveat flag** (`WEAK`, `STALE`, `UNSCALED`, `NONE ABOVE THE THRESHOLD`) with a tone
    (`""` / `warning` / `danger`, refused at construction when unknown), and `primary` for
    the **headline**. The hierarchy is typographic - 9 px bold caps / 12 px value / 17 px
    bold headline - never chromatic: a KPI is a fact, not a decoration.
  - **`CanvasWorkspace.set_kpis(cards)`** is the page-side route; **`set_readout(text)`
    survives** and means "one card per line, no heading" (the last message, the prompt).
    The stack is transparent - the cards carry the plates - as wide as its widest card
    (all cards one width, pinned `MARGIN` from the top-left, capped at the room the View
    cluster leaves), and it **reuses its cards in place** while the same keys stay on
    screen, because Predict refreshes its readout on **every frame** (rebuilding a widget
    per frame is how a readout becomes a leak).
  - **The primary KPIs, per surface** (the "which KPIs" answer): *Bench / Annotate* - the
    length, headline; *Dataset* - the value being labelled, headline; *Optics* - the
    measured 1 mm grid, headline (it is the thing drawn over the photo as evidence);
    *Predict* - the animal count, headline (`NONE ABOVE THE THRESHOLD` as a warning flag at
    zero), then the `Pose:` median length, then `ms/frame` + `fps`, then the source **with
    the video's frame number**, then the last message. The scale in force stays the footer's
    job; a card carries it where the surface measures with it.
  - **What is *not* a KPI, measured rather than assumed:** the Predict surface briefly had a
    sixth card for the **checkpoint** - and that column reached **367 px** where the canvas
    left it **286 px** (default window) and **166 px** (minimum window). The checkpoint is
    metadata, not a value that changes while you watch, so it stays in the panel verbatim
    (device + pose epoch) and the canvas keeps the five facts that move; padding and the gap
    were tightened (8/5, gap 4) until the same column measured **239 px**. Re-measured with
    a pose model loaded, at both window sizes; the residual is recorded in `progress.md`.
  - **Theme**: four new `PHOTO_OVERLAY` tokens (`overlay_text_muted`, `overlay_card_bg`,
    `overlay_flag_warning`, `overlay_flag_danger` - identical in both schemes, because the
    cards lie on a photograph) and QSS for `canvas_kpi_stack` / `_card` / `_label` /
    `_value` (`[emphasis="primary"]`) / `_flag` (`[tone=...]`). A flag's colour reaches the
    card through a **dynamic property plus a re-polish**, never from a page, so
    `test_theme.py::RepositoryTests::test_no_screen_colours_itself` still holds.
    **Letter-spacing is the one type decision QSS cannot express** (`LABEL_LETTER_SPACING`,
    set on the font) - said in both places rather than discovered later.
  - **Tests**: `tests/test_canvas_workspace.py::KpiCardTests` (11) plus every readout
    contract that already existed (`ReadoutTests`, the pixel test, `test_predict`'s
    `frame n` / `Pose:` assertions) - they pass unchanged, because `readout` reports the
    stack back as text. `tests/test_theme.py` gained the five new object names.
  - **Found while validating - not caused by this change:** `assets/ui/bench.ui` and
    `assets/ui/dataset.ui` (both untracked, last written 15:45) did not contain the
    `stream_hint_label` that `bench.py:234` / `dataset.py:791` already set, so **`python
    run.py` crashed on the Bench and Dataset tiles and 124 suite tests errored**. Both
    labels added (bench: `image_group` row 3; dataset: `corpus_group` row 6) and
    `dataset.ui`'s lost section titles restored (`corpus_group` had an empty title, and the
    panel's own frame carried the window title). `dataset.ui` is still **one** group box
    wrapping the corpus and label boxes, where every other surface has three **sibling**
    sections (`bench.ui`: 1 Image / 2 Measure / 3 Results; `calibration.ui`: 1 Optics /
    2 Reference / 3 Scale) - recorded in `progress.md` as a known issue, not silently
    redesigned.
  - **Verified**: `make ui-check` paints **12 canvases** in light and dark, full screen and
    back; the full suite is green again (see `progress.md` for the count).

- **2026-10-17 (later) — the screens are editable in Qt's own UI editor (asked: "is it
  possible to use the qt built in editor with this app?").** Yes, and now it is a
  supported workflow instead of a README aside:
  - **The editor is already installed.** The `z-app` env ships **`Designer6`**
    (`$CONDA_PREFIX/bin/Designer6` → `lib/qt6/bin/Designer.app`), i.e. the Qt Designer
    that matches the PySide6 6.11 runtime the app loads the files with. All 13
    `assets/ui/*.ui` files are plain Designer XML with **no `<customwidget>` and no
    `<resources>`**, and every one of them loads through the app's `QUiLoader`
    (`scripts/edit_ui.py --list` prints each screen's class, name and widget count).
  - **`scripts/edit_ui.py` + `make ui-designer FILE=bench.ui`** open one screen in that
    editor *after* loading it the way the app loads it, so "would this even open?" is
    answered before the edit; `make ui-list` checks every screen and reports what a
    Designer edit would break (duplicate `objectName`, a promoted widget, a `.qrc`).
  - **`tests/test_ui_files.py` (9 tests)** is the guard: XML well-formed, **no widget
    name used twice**, no promoted widgets, no resource files, every widget class is a
    Qt class, every screen loads, and every named control reaches the page as an
    attribute.
  - **That check found a real bug:** `annotate.ui` declared **two** `btn_new_specimen`
    buttons, and `load_ui` exposes one attribute per name (`findChildren` order decides
    which) - so the "New specimen" button inside the Subject group was never wired,
    never enabled/disabled, and the operator saw two identical buttons of which one did
    nothing. The stray duplicate is removed (the group-box button is now the wired one).
  - **What Designer cannot show, documented in the README** rather than discovered: the
    **visual area** (live preview, image canvas, on-canvas tool clusters) is built in
    code and added at runtime, and `MainWindow` re-parents `stackedWidget` into a scroll
    area and mounts the footer + step bar after loading `pages.ui`.
  - **Tests: 788** (was 779).

- **2026-10-17 (later) — the bar is in every window, and an image canvas takes any
  source** (asked for as "add the footer that currently on homepage to all other
  windows" and "unify the image canvas to accept different source so we can get rid of
  duplication when loading from file and see only ever one image").
  - **The footer now exists everywhere, so two screens stopped lying.** Bench and
    Calibrate both told the operator to *"press Calibrate/Connect in the bar below"* -
    a bar those windows did not have (only the hub and the Study shell did). New base
    class **`hub.SurfaceWindow`** mounts the screen above one `StatusFooter` for
    Calibration/Bench/Dataset/Predict and the **surface plan** window; the attribute is
    **`status_footer`** on *every* window (`page_chrome`/`WINDOW_CHROME` already hid
    that name for full screen - the hub's `footer` was renamed to match). `Calibrate...`
    is offered by every window but Optics (`StatusFooter(show_calibrate=False)`: it *is*
    the calibration) and **hands over to Optics** - the window hides, a
    `CalibrationWindow` opens with `WA_DeleteOnClose`, and closing it re-reads the
    footer (the scale may have changed) and shows the window again, so there is still
    one window on screen. A second press re-raises the window it already opened.
    `tests/test_ui_smoke.py::WindowFooterTests` (4) covers all of it.
  - **`ImageCanvas.set_source` is the canvas' only way in** - a numpy frame, a **file
    path** (read by `daphnia.vision.images.load_image`, the loader that also reads
    `.HEIC`), a `QImage`/`QPixmap`, or `None`. `set_frame`/`set_image_path` are deleted
    and all five screens (Calibrate, Annotate, Bench, Dataset, Predict) lost their
    `try: load_image(...) except ImageLoadError` dance **and** their own wording: the
    canvas now refuses a source by drawing the reason where the image would be and
    emitting `source_failed`, and each screen wires that to its status line in one line
    (`canvas.source_failed.connect(self.status_label.setText)`) - the refusal reads
    *"Could not load 'x.heic' - could not decode /… (.heic) …"*.
    **Annotate's hidden bug is gone**: it used `QPixmap(path)`, a second loader that
    showed *nothing at all* when a frame's file was missing.
  - **"Only ever one image" is now enforced, not hoped for.** `set_source` returns the
    frame it displayed (so a page that needs the pixels does not read the file twice -
    `canvas.frame` is that same array) and every call clears the previous image's
    points, box, overlays and grid (`set_annotations([])` included - it did *not* before).
    A **refused** source leaves the page's own frame state empty too: the probe that
    found this showed `page._frame` still holding the previous specimen while the canvas
    showed the refusal, i.e. numbers standing over a picture nobody could see. Bench,
    Calibrate, Dataset (frame queue *and* box queue) and Predict now clear their image
    state on refusal.
  - **A real memory bug found while doing it:** `np.ascontiguousarray` on an
    already-contiguous slice returns the *same* array, so `qimage_to_frame`'s grayscale
    branch returned a **view of Qt's buffer** - a use-after-free once the `QImage` went
    out of scope, whose only symptom was a first row shifted by one byte, and only
    sometimes (it reproduced in the test suite and not in a script, which is what made
    it worth chasing). It `.copy()`s now, and the round trip
    (`frame -> QImage -> frame`, both grey and BGR) is asserted together with
    `frame.flags["OWNDATA"]`.
  - **Tests: 779** (was 761): `SourceTests` in `tests/test_image_canvas.py` (12: every
    accepted source, the clear, the named refusal + signal, the wrong kind raising,
    "the new image inherits no markings", the ownership trap, the painted refusal) and
    the strengthened page tests (Bench, Calibrate) that load a good image *first* so a
    survivor would be visible. Probe scripts: `/tmp/ds_source_probe.py`,
    `/tmp/ds_footer_probe.py`.

- **2026-10-17 — the theme is one system, and the right-hand panel finally fits what
  is in it** (asked for as "all pages and window ui seem to have a dark mode that is
  currently awful, some follow system setting some don't - update to a unified system"
  and then "make in all windows that have right panel the panel bit wider so it fits
  content").
  - **The diagnosis, measured on this Mac (macOS in Dark Mode).** The app applied a
    single *light* stylesheet (`assets/theme/MacOS.qss`, GTRONICK) on top of the
    platform palette, which here is dark (`QStyleHints.colorScheme()` ->
    `ColorScheme.Dark`, `QPalette.Window` = `#323232`). The stylesheet paints
    `QLabel { color: #272727 }`, so body text sat at **1.1:1 contrast** on the window;
    every widget the stylesheet never mentioned (`QGroupBox` - 24 of them in the .ui
    files -, `QTableWidget`, `QListWidget`, `QTextBrowser`, `QMessageBox`, menus, tool
    tips) kept the *system's* dark look while the buttons and fields stayed light; the
    file's `QScrollArea { background-color:#222b2e }` and `QTabWidget
    { background-color:#000000 }` blocks are dark even in a light theme; and its check
    mark / combo arrow / spin arrow icons were `url()`s to **Adwaita PNGs that only
    exist on Linux**, so check boxes had no tick at all. Screenshots of all five
    windows before and after: `/tmp/ds_shots/before|panel` (the dataset panel's
    buttons were cut off at the panel edge; the hub's tiles were dark with dark text).
  - **One system, three parts** (`daphnia/ui/theme.py`):
    **tokens** (`LIGHT`/`DARK`, the *same* key set - a missing key is a test failure,
    not a fallback) are the only place a colour is written down; the **QPalette**
    built from them now colours everything the stylesheet does not mention (that was
    the half following the system); the **template**
    (`assets/theme/daphniascope.qss`, `%token%` placeholders) carries structure and
    the few colours that are decisions - the primary action, the step states, and
    `PHOTO_OVERLAY`, the translucent tools *over* a photograph, identical in both
    schemes on purpose. Style is **Fusion**, which draws the check mark, the combo
    arrow and the spin buttons from the palette on every platform; no `::indicator`
    is styled anywhere and no `url(` remains.
  - **The scheme follows the desktop** (`QStyleHints.colorSchemeChanged` ->
    re-apply) so the whole application is dark on a dark Mac and light on a light one;
    `DAPHNIA_THEME=light|dark` (or `theme.set_scheme`) pins it, and a typo **raises**
    instead of silently giving the other theme. `ImageCanvas` repaints on
    `QEvent.StyleChange` because its panel colour comes from a token, not from a
    stylesheet rule. `ImageCanvas.PANEL_BACKGROUND/PANEL_TEXT` and the inline
    stylesheets in `ImageViewer`/`StreamView` are gone: painter code asks
    `theme.qcolor("canvas")`, and the two panels are themed by `objectName`
    (`#stream_view`, `#image_viewer`) so a scheme change reaches them.
  - **The panel width is measured.** The controls panel scrolls vertically only, so a
    wider child is clipped *and unreachable*: `CONTROLS_WIDTH` 360 -> **420** (the
    densest shared screen asks for 345 px of content; the long explanatory sentences
    now wrap in 2-3 lines instead of 6) and `Dataset.PANEL_WIDTH = **500**`, sized from
    its dataset row (`<name> [<what it labels>]` combo + two buttons = 458 px + panel
    padding + scroll bar) - that panel was visibly cut off before. Calibrate's 400 px
    override was removed because the shared width made it redundant.
  - **Tests: 761** (was 727): new `tests/test_theme.py` (33 - token parity between
    schemes, WCAG contrast >= 4:1 for body text in both, no `url(`, no `::indicator`,
    **every widget class and every `objectName` the app uses is themed**, every token
    substituted, a broken template and a bad `DAPHNIA_THEME` raise with the name and
    the fix, palette = tokens, pinned vs followed scheme, and "no module outside
    `theme.py` calls `setStyleSheet`"), the canvas-overlay pixel test now runs in
    **both** schemes, and
    `test_ui_smoke.py::test_every_panel_is_wide_enough_for_what_is_in_it` builds all
    five windows and asserts, per page, that the panel is at least the shared width and
    that its content's minimum width fits the viewport. `scripts/check_ui.py` (=
    `make ui-check`) now builds every window **twice** (once per scheme) with the same
    fit assertion: 12 canvases, both schemes, green.
  - **`assets/theme/MacOS.qss` is deleted** (`git rm`): two theme files that both look
    authoritative is the trap; `README.md` §Styling, `docs/architecture.md` and
    `memory-bank/systemPatterns.md` now describe the token/template system.

- **2026-10-16 — a video file is stepped frame by frame, from the canvas** (asked for
  as "if input is video, arrow keys go to the next frame or back; conditional overlay
  on canvas"; `SR-PREDICT-010`, `SR-PREDICT-011`).
  - **The tools are conditional, and so is the cluster.** Predict adds *< Frame* and
    *Frame >* to the lower-left cluster and hides both while the source is not a video
    file; `ToolCluster.sync_visibility()` now hides a cluster whose tools are *all*
    hidden, so a live feed shows **no** frame controls and no empty panel over the
    picture. `workspace.set_visible(key, applies)` is the page's whole vocabulary for
    this.
  - **Arrow keys change meaning with the source.** `Left`/`Right` are window shortcuts
    **enabled only for a video file**; elsewhere they are off and keep panning the
    canvas exactly as before. The two halves are tested by *pressing the key*
    (`QTest.keyClick`): with a video the page steps and the canvas does not pan; with
    the live feed it pans by `PAN_STEP` and nothing steps.
  - **Backwards needs a seek.** The reader sits on the frame *after* the one on screen,
    so one frame back is `seek(index - 2)` plus a step (a paused loop only reads what a
    step asks it to read). Forward stays a plain step. `SR-PREDICT-010` says this in
    words, so the next reader does not "simplify" it into an off-by-one.
  - **Stepping pauses first** (as `Step` already did), so it never races the reader, and
    `_index` is now tracked from the frames the worker reports, so the panel, the
    slider and the on-canvas readout all agree on *which* frame is on screen.
  - **Found while wiring it**: `_start_worker` never refreshed the source, so the name
    of an opened video only appeared once a camera event happened to fire; it now
    re-reads the source (and so the tools) when it changes.
  - **Tests: 552 passing** (was 541). `FrameSteppingTests` drives the page through a
    recording stub worker (offered/not offered, both directions, the index, the pause,
    the message when there is no video); `ArrowKeyTests` presses the real keys in a
    shown window; `VideoSteppingTests` (torch-gated) writes a **real mp4**, loads a
    tiny checkpoint, pauses it and steps one frame each way through the **real
    inference worker**, asserting the frame index in the panel *and* the readout.
- **2026-10-16 — the measuring controls moved onto the canvas, and full screen**
  (`X-N11`, `X-N12`; asked for as "expandable tools in top right and lower left, full
  screen so one can work only in the image canvas").
  - **Two reusable modules, not five toolbars.** `daphnia/ui/widgets/canvas_tools.py`
    (the overlay: `ToolSpec`, `ToolCluster`, `CanvasTools` - the clusters, the readout,
    the corner geometry) and `canvas_workspace.py` (`CanvasWorkspace` = the five view
    tools every canvas gets + the page's own tools + the full-screen view +
    `page_chrome()`). A page asks in **one line** - `self.canvas_workspace(self.canvas)`
    - and then `workspace.add_tool(ToolSpec(...), callback)`; `ShellPage` owns the
    helper, so a new surface gets the tools without touching the shell.
  - **The tools are children of the canvas**, pinned by an event filter
    (`QEvent.Resize` / `QEvent.Show`), so they are on the picture and survive full
    screen. A cluster folds to its header (a native arrow, no glyph to be missing from
    a font); a cluster with no tools hides itself. `set_visible`/`set_checked` let a
    page keep the cluster honest (Save vanishes in landmark mode, Box follows the
    model) without firing the tool's own callback.
  - **Three traps closed, each pinned by a test.** (1) Every tool takes
    `Qt.NoFocus`, so pressing one never steals `+`/`-`/`F`/arrows from the canvas - an
    operator labelling at volume lives on those keys. (2) A cluster sets
    `WA_NoMousePropagation`, so a click that misses a button cannot fall through and
    drop a landmark under the panel. (3) Placement reads `isVisibleTo(canvas)`, **not**
    `isVisible()`: a page is laid out before its window is on screen, and Qt defers
    resize events for hidden widgets - my first version placed nothing until the window
    appeared, and the test caught it.
  - **Full screen hides chrome; it never reparents the canvas** (`CanvasFullScreen`).
    The panel, the sibling visuals and the window's `step_bar`/`status_footer` are
    hidden and the *window* goes full screen, so the canvas keeps its points, overlays,
    tools and zoom - no layout surgery, nothing to put back in the wrong order. Only
    what was visible returns; `Esc` is a **window shortcut enabled only while full
    screen**, so outside it `Esc` still abandons a half-drawn box.
  - **The readout is what makes full screen honest.** The panel that carried the number
    is hidden there, so every page repeats its own lines on the canvas
    (`workspace.set_readout`): Bench and Dataset route their action tools through
    `_tool_action()` so a tool's answer lands on the image too, Annotate shows frame
    and length, Calibrate shows the grid/span/scale evidence, Predict shows the
    detector, the source, the count and any failure.
  - **The panels gave the controls up rather than duplicating them**: the four
    `zoom_row`s and Bench/Dataset's `action_row`s, Dataset's `btn_box` and `nav_row`,
    Annotate's `nav_row`/save/clear came out of the `.ui` files, together with the page
    wiring and the now-dead `_on_pan_toggled` (the workspace owns the hand tool). The
    `.ui` files stay the place for *choices* (mode, specimen, dataset, optics, export).
  - **Tests: 541 passing** (was 515). New `tests/test_canvas_workspace.py` (25: corners,
    folding, callbacks, "the panel does not mark the image", chrome going and
    returning, `Esc`, and a **pixel** test via `grab()`/`pixelColor` that the
    translucent panels are really painted over the image). `test_ui_smoke` now asserts
    the tools are children of *the* canvas and that nothing duplicates them in the
    panel; `test_bench`/`test_dataset` look the actions up in the clusters (and their
    "Save is hidden in landmark mode" tests now **show** the page, so visibility means
    something). `test_image_canvas` covers the new `zoom_changed` signal that feeds the
    View header.
  - **Verified the way a user would**: offscreen run of all five windows
    (`CalibrationWindow`, `BenchWindow`, `DatasetWindow`, `PredictWindow` and the study
    shell's Calibrate **and** Annotate pages) - each canvas painted, entered full
    screen (panel gone), painted again, came back with the panel - no crash in a paint
    path.
- **2026-10-16 — the first real training run, and the two bugs it exposed.**
  Trained `rtmdet-tiny` on `datasets/bb` (158 train frames / 332 boxes from 10
  captures, 30 epochs, ~4 min on MPS) and validated on the **held-out capture**
  (15 frames, 54 boxes). Findings, in the order they hurt:
  - **The evaluation compared two different coordinate frames.** Predictions were
    mapped back to the original image, the labels were left in the *padded* frame, so
    every pair had IoU 0 - **AP50 reported 0.000 for thirty epochs** while the model
    was learning fine (the same weights score **0.745** once the metric is fixed).
    Fixed: both sides go through `Letterbox.invert`. Pinned by
    `tests/test_rtmdet.py::EvaluationTests`, whose "oracle" detector returns the
    labels and must score exactly 1.0.
  - **A lambda connected to the process-wide stream service outlived its widgets.**
    `shared_service()` keeps emitting after the Predict window closes, and a PySide6
    connection made with a lambda has no receiver QObject to auto-disconnect, so the
    callback reached freed C++ objects
    (`libshiboken: Internal C++ object already deleted`, hit by the user). Fixed by
    connecting the **bound method**; pinned by
    `tests/test_predict.py::PredictWindowTests`, which deletes the page with
    `shiboken6.delete` and then emits on every signal.
  - **What the model actually does** (the number to read is at the *operating*
    threshold, not AP's 0.05): at **score ≥ 0.35** — the surface's default —
    precision **0.61**, recall **0.83**, F1 **0.70**, ~5 boxes for ~3.6 animals;
    AP50 0.73. At 0.05 (AP's threshold) precision is 0.011 with 300 capped
    detections/frame, which is what made the run look broken.
  - **The score is not well calibrated and the usable range is narrow:** recall
    collapses from 0.83 to 0.15 between thresholds 0.35 and 0.50.
  - **The validation set cannot resolve differences:** one capture, 15 frames;
    epoch-to-epoch AP50 swung 0.16→0.81 with no trend, so "best 0.813" is a lucky
    draw, not a best model. **This is the strongest argument so far for annotating
    more distinct captures** (`SR-DATASET-010`).
  - **Latency:** 15 ms/frame inference at 640 (≈68 fps) and 80 ms per training step
    (batch 4 at 640) on MPS, after an ~8 s first-step warm-up. The same 30-epoch run
    took 4 min once and 65 min another time (CPU time 245 s both ways), so wall time
    on this machine is not stable - re-measure on a quiet machine before planning a
    long run.
  - **The two splits are not comparable:** the export's val split holds *different*
    frames (33 frames, 9 captures), so its lower result does not demonstrate the leak
    inflating a score here. The leak remains a structural argument (11 of 14
    captures), not a measured inflation.
- **2026-10-16 — RTMDet landed: training, evaluation and a fifth surface (`predict`).**
  - **`daphnia/vision/export.py`** reads a Dataset export into frames + boxes, grouped
    by **capture** (the observation directory of each specimen's original path), and
    reports what is wrong with the corpus (`warnings()`, `stats_summary()`,
    `describe()`). Pure stdlib: no Qt, no PyTorch.
  - **`daphnia/vision/rtmdet/`** — `model.py` (CSPNeXt, CSPNeXtPAFPN, a SepBN head
    sharing convolutions across levels), `losses.py` (SimOTA, QualityFocalLoss, GIoU,
    NumPy NMS, `distance2bbox`), `data.py` (letterbox to a multiple of 32 with a
    recorded mapping, mosaic, flip, colour jitter, BGR normalisation with RTMDet's
    numbers), `train.py` (warm-up + cosine AdamW, MPS/CPU, checkpoints, an honest
    AP50), `detect.py` (`Detector` — thresholds, the map-back, `Detection` records).
  - **`scripts/train_rtmdet.py`**, **`scripts/check_torch.py`**, `make train`,
    `make train-check`, `make train-inspect`. `pytorch` uncommented in
    `environment.yml`. Guide: **`docs/rtmdet.md`**.
  - **Surface `predict`** (`surfaces.py`, `pages/predict.py`, `assets/ui/predict.ui`,
    `hub.PredictWindow`, `docs/requirements/SR-PREDICT.md`): RTMDet boxes on the live
    feed, a video file or one image, in an `InferenceWorker` thread that always takes
    the **newest** frame. Never stores a prediction (`SR-PREDICT-N01`).
  - **`daphnia/vision/detection.py`** — `localize_well()` now returns one region per
    animal from the trained detector, and raises `NotImplementedError` **naming
    `make train`** when there is none. The Localize step is no longer a stub.
  - **Annotation-tool fixes**: box corners are clamped to the image
    (`_to_image(clamp=True)`; 11 of 526 real boxes were out of bounds), a
    duplicate-box warning on store, `annotation.MIN_BOX_PIXELS` as the single source
    of truth for "too small to be a specimen", `ImageCanvas.set_source(reset_view=)`
    (then `set_frame`; renamed 2026-10-17) so a live view keeps the operator's zoom, and
    `maintenance.find_boxes_outside_image()` / `clamp_corpus_boxes()` +
    `scripts/repair_boxes.py` (dry run by default).
  - **Verified end to end on MPS**: `check_torch.py` (forward + backward, 139 tensors
    graded), a 12-step training run over `datasets/bb` (group split: 158 train / 15 val
    frames), evaluation, checkpoint write/read. **508 tests pass** (was 410).
- **2026-10-16 — `z-app` repaired (twice) out of a mixed-channel state, and PyTorch
  added.** The environment had drifted: `opencv`/`libopencv`/`py-opencv` and
  `pytorch`/`libtorch` (the `gpu_mps` build) from **pkgs/main**, `libprotobuf`,
  `libsqlite`, `freetype`, `harfbuzz` and others from **conda-forge**, plus a
  pip-installed `protobuf`. Symptoms in order: `cv2` failed on
  `ThreadSafeArena::thread_cache` (protobuf ABI), then on a missing
  `libfreetype.6.dylib`; `_sqlite3` failed on a missing
  `_sqlite3_enable_load_extension`; `import torch` failed on the protobuf symbol
  again. Repaired with `conda install --override-channels -c conda-forge` for the
  offending packages, plus `--force` remove+install where conda's record claimed a
  file that was not on disk. **Result:** `cv2 5.0.0`, `sqlite3 3.54.0`,
  `torch 2.14.0` with **MPS available** on this M4. Constraint 7 stands: `py-opencv`
  still ships no `RECORD`, so never `pip install` into `z-app`.
- **2026-10-16 — `z-app` repaired after an aborted `pip install`; Label Studio
  moved out of the app env.** No app code changed. A `pip install label-studio`
  run (aimed at the Dataset surface's labelling work) died mid-transaction on
  `error: uninstall-no-record-file` (`opencv-python-headless`, installed by conda)
  and left `z-app` half-migrated: conda-forge `numpy` was replaced by pip's 2.2.6
  (conda stopped tracking it) plus two orphaned Django packages
  (`django-environ`, `django-ranged-fileresponse`). Restored with
  `pip uninstall` + `conda install -n z-app -c conda-forge numpy`; **410 tests
  pass** on the repaired env. Label Studio 1.23.2 now lives in an isolated `uv`
  tool instead (see `techContext.md`, constraints 7-8). *Not* wired into any
  surface — the app keeps labelling into `corpus_labels` via `DatasetWindow`.
- **One corpus per model: datasets + a schema-driven annotation system.**
  - **New `daphnia/annotation.py`** (Qt-free): `KeypointSystem` (`head_tail`,
    `head_tail_sides`, `none`) and `AnnotationSchema` - the four model targets the
    Dataset surface offers (`pose_full` = 4 points, the **default**, so nothing
    changes for an existing corpus; `pose_head_tail` = 2 points; `pose_full_boxes`;
    `boxes`). It also owns the **export names** (`head`, `tail`, `sideL`, `sideR`)
    that a training config expects, and reuses `daphnia.study`'s landmark names so a
    corpus label and a study measurement still name the same anatomy (`X-N08`).
  - **`datasets` table** (+ `corpus_boxes`) and `corpus_labels.dataset_id`; labels
    are unique per `(dataset, file, specimen, trait)` and boxes per
    `(dataset, file, specimen)`, so two models may label the same frame. So
    training another model means *creating another dataset*.
  - **A non-destructive rebuild migration**: SQLite cannot drop the old implicit
    `UNIQUE (file_path, specimen_index, trait)`, so `rescope_corpus_labels()` renames
    the table, re-runs `schema.sql`, **copies every row across** and only then drops
    the legacy table. Verified on a copy of the live `zoo.db`: 19/19 labels kept.
    The first dataset **adopts** labels made before datasets existed
    (`repository.adopt_legacy_corpus`), so nothing silently disappears.
  - **Bounding-box labelling**: `ImageCanvas.set_box`/`set_boxes`/`box_changed`,
    drag-to-draw with corner normalisation, dashed per-specimen overlays, and a
    mis-click guard (`MIN_BOX_PIXELS`). A detection-only dataset hides the keypoint
    controls and shows the box control; a dataset without boxes hides it.
  - **`labeling.Draft` is schema-aware** (`intervals`/`landmarks` injected) with the
    old behaviour as the default; `labeling.intervals_for(schema)` is the one place
    that turns a model's keypoint system into the traits an operator measures.
  - **`maintenance.export_dataset()`** exports **one dataset**: `images/`,
    `labels.csv` (with `p1_name`/`p2_name`), `boxes.csv`, `labels.json` (schema +
    per-specimen records joining named keypoints and `bbox`) and `manifest.json`.
    The legacy, unscoped `export_corpus()` is untouched.
  - **Tests: 364 passing** (was 321). New `tests/test_annotation.py` (15), a
    `DatasetModelTests` class (17: datasets, model-scoped traits, boxes, per-dataset
    export, adoption), a canvas `BoxTests` class (8) and two DB migration tests.
  - Requirements `SR-DATASET-015..018` added; `SR-DATASET.md` documents the dataset
    model table, the box rules and the new export format.
- **The Dataset entry point finished around the dataset/model model.**
  - **The dataset list is a real control**: every entry shows its model
    (`name [Pose - head, tail & sides (4 points)]` + a tooltip with the schema),
    `New dataset...` creates, **`Rename...`** renames the open one - and *locks the
    model*, because the schema is what the existing labels mean
    (`repository.rename_dataset`, page `set_dataset_name`).
  - **A taken name is numbered, never refused** (`repository.next_dataset_name` →
    `corpus (2)`), used by both create and rename, so a collision can never fail the
    insert (the `UNIQUE(name)` index used to be able to crash the dialog).
  - **Box workflow**: selecting a specimen **sites its stored box** as the editable
    one (a correction is a redraw of the rectangle on screen, `_show_box`), a new box
    advances to the **next specimen missing one**, and `B` / the button toggles box
    labelling - *refused in words* by a keypoint-only model (`toggle_box_mode`).
    Save is hidden whenever a label stores itself (landmark **and** box mode).
  - **Export**: opens on `datasets/annotations/<dataset name>_<timestamp>`, and now
    also writes **`splits/{train,val,test}.txt`** (the exported image paths per split,
    recorded in `labels.json.splits` and `manifest.json`) so a training run needs no
    CSV parsing.
  - **Tests: 378 passing** (was 364). `DatasetRegistryTests` (5: naming/renaming
    rules), plus page tests for rename, the model-labelled combo, box siting, the
    `B` toggle, the missing-box advance and the split lists.
- **Two interaction fixes in the annotation canvas, and a crash.**
  - **Scroll pans, it no longer zooms** (`wheelEvent` + `_wheel_delta`): a two-finger
    scroll moves the view in **both** axes, proportionally to the pixel/angle delta,
    and never changes the magnification - `⌘+scroll` does not either. Zooming is the
    **pinch**, the *Zoom -/Fit/Zoom +* buttons and `+`/`-`/`F`. Rationale: an
    accidental zoom while clicking is an *error in the ground truth*, not an
    inconvenience. Tooltips (annotation/calibration `.ui`), README, `SR-BENCH` and
    `SR-DATASET` updated to match.
  - **A box can be drawn two ways** (`ImageCanvas.anchor_box_corner`): **drag** a
    rectangle, or **click one corner and then the opposite one** - the anchored corner
    stays visible (painted marker), the prompt/readout say what the second click will
    do (`box_anchor_changed`, `box_pending`, `box_origin`), and `Esc` / a right click /
    *Clear points* abandon a half-drawn box. A corner is never mistaken for a box:
    `mouseReleaseEvent` stores only a box the gesture actually made, so a click emits
    nothing and writes nothing.
  - **Double click walks to the next frame** in the Dataset queue
    (`ImageCanvas.double_clicked` → `Dataset.next_frame`, `SR-DATASET-019`). The
    gesture **stores nothing**: `mouseDoubleClickEvent` places no point (Qt sends it
    instead of a second press) and aborts a box preview, and `_take_back_click()`
    takes back what the gesture's own first press did - using the same undo a
    mistaken click gets. Ends of the queue are reported, so a stuck frame is visible.
  - **A crash I introduced, found by the user, and the testing gap behind it:**
    `_paint_boxes` added a `QPointF` to `QRect.topLeft()` (a `QPoint`). The exception
    escaped *inside* `paintEvent`, leaving the painter active → `QBackingStore::
    endPaint() called with active painter` → **segfault** in `python run.py`, while
    all tests stayed green (the error happened in Qt's event loop, not the test body).
    Fixed with explicit float arithmetic, and closed with a new
    `tests/test_image_canvas.py::PaintTests` that opens a real `QPainter` and calls
    the overlay painters from the test's own frame - verified to fail on the old
    expression. The crash path was then replayed end to end offscreen (boxes dataset,
    two stored boxes, window shown, canvas repainted): renders, no crash.
  - **Tests: 410 passing** (was 378).
- **Memory bank initialized** in `app/memory-bank/` (this file set).
- **Requirement documents created** under `docs/requirements/`:
  `00-framework.md` plus `SR-CORE`, `SR-OPTICS`, `SR-BENCH`, `SR-DATASET`,
  `SR-STUDY` (82 numbered, traceable requirements).
- **Surface launch points implemented.** New: `daphnia/surfaces.py` (Qt-free
  registry), `daphnia/ui/hub.py` (`HubWindow`, `SurfacePlanWindow`, `window_for`),
  `assets/ui/home.ui` and `daphnia/ui/pages/home.py` (the hub screen). `run.py`
  and `daphnia/ui/app.py` now dispatch on a surface key
  (`hub` | `study` | `bench` | `dataset` | `optics`), `python -m daphnia` accepts
  a key too, and `--help` lists the launch points.
- **A not-yet-built surface opens its specification**, never another surface's
  window - a launch point cannot silently do the wrong thing (`SR-CORE-014`).
- **Behaviour-preserving refactor:** the study window's screen fitting moved to
  `daphnia/ui/shell.py::fit_to_screen`, now shared with the hub.
- **Tests: 141 passing** (was 114). New `tests/test_surfaces.py` (registry, launch
  dispatch, Qt-purity guard, window mapping) plus `HubTests` in
  `tests/test_ui_smoke.py`. The study shell's original 114 tests are untouched and
  still green.
- **Docs corrected:** `docs/architecture.md` and the README listed two routes that
  do not exist (`plates`, `review`); both now list the real study routes plus the
  launch points.
- **Calibration became an immutable record** (the user's core realisation: a good
  adapter can still be bumped, so recalibration must be possible at any point and
  must never rewrite the scale earlier measurements used):
  - `calibration_profiles` rows are now *one calibration each*, grouped by a
    `setup_key` (the optical chain); `repository.register_calibration()` always
    inserts, and `upsert_profile()` is documented as the legacy overwrite path.
  - `measurements.uncertainty_pct` records the `+/-` that applied, so the error
    travels with the result.
  - The uncertainty is **derived from the clicked reference span**
    (`relative_uncertainty_pct`), not a magic constant. At the project's own 2x
    setup a single 0.1 mm pitch is 14.7 px -> **+/- 13.6 %**, where the old
    hard-coded default claimed 0.5 % - a **27x overstatement of precision**.
  - `is_stale()` + `CALIBRATION_MAX_AGE_HOURS` flag a calibration that may have
    been invalidated by a knock; the hub footer now reports scale, `+/-`, age and
    `RE-VERIFY`, and the Calibrate and Annotate readouts show `+/-`.
  - **Tests: 161 passing** (was 141). New `tests/test_calibrations.py` (18 tests)
    plus two hub-footer tests.
  - **Migration verified on the live database**: `setup_key` backfilled,
    `uncertainty_pct` added, no reset needed.
- **The reference target is now only the printed 1 mm grid** (the other pitches
  are removed, in the Python constants *and* in `calibration.ui`), and it is
  **detected automatically**:
  - new `daphnia/vision/grid.py`: 2D FFT for the lattice directions, then a
    projection + zero-padded DFT for a sub-pixel period. 0.09-0.14 % period
    accuracy on synthetic grids, rotated, blurred or noisy.
  - **It refuses what it should**: prominence, period-count, squareness and
    orthogonality gates reject a flat field, blobs, a single stripe family, a
    non-square target - and, critically, a **specimen-plate-like image**.
  - The stored uncertainty combines the fit error with the printed paper's own
    ~0.5 % tolerance, because the paper is the accuracy floor.
  - The page gained "Detect 1 mm grid" and "Open a photo..." (a calibration
    no longer needs the phone attached; any file, any format).
- **One shared status footer in every window** (`widgets/status_footer.py`): the
  camera state with Connect/Disconnect, and the scale with `+/-`, age,
  `RE-VERIFY` and Calibrate. The hub opens a standalone `CalibrationWindow`; the
  study shell navigates back to the Calibrate step.
- **OPTICS is now an available surface** (`python run.py optics` opens the
  calibration window), so the hub is 2 ready / 2 specified.
- **Tests: 212 passing** (was 161). New `tests/test_grid.py` (27) plus the
  end-to-end grid-to-calibration flow in `tests/test_calibration.py`.
- **The camera is now a first-class, always-visible state**, fixing a real bug: the
  standalone Optics window said "no connected camera" while the phone was streaming,
  because `StreamService.start()` was only ever called by the Setup step's Connect
  button. `StreamService` now reports four states
  (`DISCONNECTED`/`CONNECTING`/`WAITING`/`LIVE`), distinguishes `connected` (socket)
  from `has_signal` (frames), keeps the failure reason, and is started/stopped from
  the shared footer in **every** window (`SR-CORE-017/018/019`). `StreamView` clears
  a stale preview on drop; the Calibrate screen's message is state-aware instead of
  pointing at a Setup step that a standalone window does not have.
- `ScaleFooter` was renamed **`StatusFooter`** because it now carries two facts.
- **The calibrate surface was redesigned as a chain, and "Detect 1 mm grid" fixed.**
  The bug was concrete: `btn_detect` was **disabled until a frame existed**, and a
  disabled `QPushButton` swallows the click - so it "did nothing", silently. Reproduced
  before fixing. Now no action is ever disabled (`X-N10`); an action that cannot
  proceed says what is missing.
  `assets/ui/calibration.ui` was rebuilt as **1 Optics / 2 Reference / 3 Scale** in
  the order of the physics, with the live **chain key**, the calibration **in force for
  that chain**, the detected grid **drawn over the photo** (`ImageCanvas.set_lattice`),
  the hand span counted in **squares** (no pitch to mistype), and panel 3 showing the
  `+/-`, its **source** and a concrete example length. Old widgets removed:
  `target` (single target now), `actual_distance_um` (replaced by `spin_squares` in
  the UI; the database column keeps its name), `stream_group`. **Tests: 238 passing**
  (was 227) after rewriting `tests/test_calibration.py` around the panels and adding
  the overlay tests.
- **The controls panel got more room, and an empty stream preview steps aside.**
  `CONTROLS_WIDTH` 320 -> 360, and Calibration asks for 400 (`PANEL_WIDTH`).
  `StreamView(hide_when_empty=True)` hides itself while there is no frame, so the
  image keeps the whole visual area; it reappears with the first frame.
- **BENCH is built** (`python run.py bench`, `daphnia/ui/pages/bench.py`): open any
  image or the live frame -> two landmarks -> a length, with **no study, plate or
  phone**. New domain support: `repository.bench_observation` (one study-less session
  **per image**, so promotion is precise), `bench_asset`,
  `attach_observation_to_study`, `queries.bench_measurements`/`bench_count`/
  `bench_observations`, `maintenance.export_bench_csv` (**landmark coordinates
  included**). Unscaled readings are shown but never stored. **Tests: 260 passing**
  (was 238), with `tests/test_bench.py` covering the storage invariants and the
  screen. Surfaces are now **3 ready** (study, bench, optics) / 1 specified (dataset).
- **Calibration accepts any photo, not only the stream.** New
  `daphnia/vision/images.py` (`load_image`): bytes + `cv2.imdecode`, then macOS
  `sips` when OpenCV has no codec - the case for the **`.HEIC` an iPhone writes by
  default**, verified end to end on a real sips-produced HEIC - and otherwise a
  named `ImageLoadError`. `detect_grid` block-averages a 12 MP photo to
  `MAX_DETECTION_SIDE_PX` and scales the period back, so the reported scale stays in
  original pixels (0.05 % error at 4000x3000, *better* than at 600x800 because
  averaging suppresses noise). `Calibration.use_photo()` is the dialog-free half, so
  it is directly testable. **Tests: 227 passing** (was 212).
- **DATASET is built** (`python run.py dataset`, `daphnia/ui/pages/dataset.py`,
  `hub.DatasetWindow`). The corpus surface from `SR-DATASET.md`, using the Bench
  pattern: import a folder (recursively) / open a file / use the live frame into a
  **deterministic queue**, walk it with Previous/Next and the keyboard (A/D, arrows,
  S to save), click two landmarks per trait and specimen, and export.
  - New table **`corpus_labels`** (one row per file/specimen/trait): the label is
    **scale-free** - keypoints in source pixels + image size - so labelling needs no
    study, plate, connection or calibration (`SR-DATASET-N01`). When a calibration is
    in force its `profile_id`/`px_per_um`/`length_um` are recorded too. Re-labelling
    the same interval updates the row (`repository.save_corpus_label`).
  - Traits: `length` (eye apex -> spine base) and `width` (dorsal -> ventral margin),
    several specimens per frame, re-usable landmark names documented in the doc.
  - `maintenance.export_corpus` writes `images/` + `labels.csv` + `labels.json` +
    `manifest.json` with a deterministic 70/15/15 split by **image** (the
    specimen/study split is the documented next step, `SR-DATASET-010`).
  - **Zero destructive automation**: no prune, no tray, no auto-select.
  - `surfaces.dataset` is now **available**; the hub is **4 ready / 0 specified**.
  - **Tests: 284 passing** (was 260). New `tests/test_dataset.py` (23) plus the
    surface/registry smoke updates.
- **Annotation UX reworked (Bench + Dataset).** The productivity blockers reported
  from real labelling:
  - **Zoom is proportional** to the wheel/pinch delta now (`ZOOM_STEP **
    (delta / SCROLL_REFERENCE)`), so a trackpad's many small deltas move gently
    instead of running away; pinch still works.
  - **Pan without dropping a point**: `Command`/`Control` + left-drag, or the new
    **Pan** hand tool (`ImageCanvas.pan_mode`).
  - **Two labelling modes** in both surfaces (`daphnia/labeling.py`): *two points per
    specimen* (default) and **one landmark at a time** - click every head, then every
    tail, then *left side*, then *right side*; each interval is stored the moment both
    ends exist. Partial points live in a `Draft`.
  - **The pass guides itself**: the specimen counter advances per click, the first
    pass sets the specimen count (N), and once a landmark sits on all `1..N` the next
    landmark in `head -> spine base -> left side -> right side` is selected
    automatically (`next_landmark`, `_advance_landmark_if_complete`) - so neither
    specimens nor landmarks are cycled by hand. The first pass is never cut short
    (auto-advance needs a *second* populated landmark). A **Next landmark** action
    remains available programmatically; the readout shows the specimen count. Bench does not advance while it
    cannot store (no scale) - `_commit_completed` returns `-1` then.
  - **See what is done without text**: `ImageCanvas.set_annotations` colours every
    already-annotated specimen (`specimen_color`), on both surfaces.
  - Landmark vocabulary centralised in `study.TRAIT_LANDMARKS` (`head`->`spine base`,
    `left side`->`right side`), shared by Study/Bench/Dataset. Bench's width trait now
    shows the side landmarks instead of the length ones (a latent bug).
  - **Tests: 314 passing** (was 308). New `InteractionTests` (canvas) + pass-mode
    suites for Dataset and Bench (including the guided landmark sequence), and
    `queries.bench_observation_for_file` to read back what an image already has.
- **Each control used for what it is, and frame clearing got honest** (Bench +
  Dataset), from the "importing a folder spans studies" feedback:
  - `combo_mode` / `combo_specimen` (the **specimen spin box became a dropdown** whose
    list grows as specimens are added) are the only dropdowns - they are real choices.
    The **keypoint/trait is *displayed*** (`keypoint_label`) with a
    **`btn_next_keypoint`** button to step it, because the *first* switch (all heads ->
    the tails) is a judgement call the auto-advance cannot make; after that each pass
    advances on its own. `labeling.next_trait()` cycles `length -> width`.
  - Actions are **separate buttons**: `btn_save` (**hidden in landmark mode**, which
    stores automatically), `btn_undo`, `btn_clear_points`, `btn_clear_frame`. *Save*
    advances to the next specimen. (An "Action" dropdown briefly replaced them, then
    was reverted - buttons for actions, dropdowns for choices.)
  - **Clear frame** deletes the frame's stored labels/measurements *and* the
    open points (`repository.delete_corpus_labels_for_file` /
    `delete_measurements_for_observation`), so a frame from another study with another
    specimen count is re-doable from scratch.
  - **Overlays are intervals, not point lists**: `set_annotations([(index, [(a,b)…])])`
    draws each measured interval as its own line (width = left→right, length =
    head→spine base) and unpaired pass points as dots; `ImageCanvas.connect_points` is
    off in landmark mode, so two heads are never joined.
  - **Tests: 314 passing** (was 308).
- **Interaction fixes from real use** (Bench + Dataset):
  - **Undo removes the last thing placed**: `_undo_pair`/`_undo_pass` try the loose
    point first (via `Draft.last()`, whatever landmark it belongs to), then the stored
    interval (`repository.delete_corpus_label` / `delete_measurement`). Stored points
    are **no longer preloaded into the editable canvas set** in pair mode, so a stored
    row stays reachable by Undo (and re-measuring never starts from stale points).
  - **Specimens get their colour from the first click**: in landmark mode the pass
    points are *not* the editable set - the canvas is cleared and everything is drawn
    through the overlays, so each specimen keeps `specimen_color(index)` immediately
    instead of all sharing the marker colour until the pass is switched. The
    **specimen number is drawn at its first point**
    (`ImageCanvas.show_annotation_index`).
  - **A horizontal two-finger scroll pans** (`wheelEvent` uses `angleDelta().x()` when
    it dominates `y`) - previously only ⌘-drag or the hand tool moved the view.
    *(Superseded: scroll now pans in **both** axes and never zooms - see the
    annotation-canvas entry below.)*
  - The **specimen dropdown is wide** (`.ui` `minimumWidth` 170).
  - **Tests: 320 passing** (was 314).

## Evidence snapshot (2026-10-15)

From `zoo.db.bak.20261015-164012` (16:40) and live `zoo.db`:

| Metric | Value |
| --- | --- |
| Studies created that afternoon | **12** (`test1, test2, test2, test1, test2, "Real Project Test", P1_1, max, max1, testtest, testtesttest, 111111`) |
| Frames captured | **293** |
| Measurements persisted | **5** (4 distinct frames → ~1.4 % of frames labelled) |
| Mode axes used | **12/12** `individuals` / `longitudinal` / `standard` (plates, plant, single-timecourse: never) |
| Live DB now | 1 study "test", 1 session, 2 observations, 10 assets, **0 measurements** |

### The readings that did work

```
160.10 um  (125.38 px @ 0.783118 px/um)
279.33 um  (218.75 px)
184.44 um  (144.44 px)
 94.07 um  ( 73.667 px)  } same observation, re-clicked
 96.46 um  ( 75.542 px)  } -> 2.5 % click-to-click difference
```

**Learnings from this:**
- Two-click annotation repeatability is ~**2.5 %** bare-handed — the method is
  sound; the tooling is the bottleneck, not the anatomy.
- The same `profile_id` (`PROF_LEICADM500_2X_IPHONE_1.0X_INFINITY`) carried
  `px_per_um` **0.783** at 11:10 and **0.147** now — a **5.3x** change under a
  stable identity. `measurements.px_per_um` records the applied scale (good), but
  nothing forces anyone to *notice* the scale moved. This is a prerequisite fix
  for "much smaller organisms".
- Today's good numbers survive only in a **`.bak` file** — because resetting to a
  clean bench meant `make db-reset`.

## Active decisions and considerations

| # | Decision | Status |
| --- | --- | --- |
| 1 | One repo, one package, one DB, **N launch points** (not separate repos) | **implemented** - `daphnia/surfaces.py` + `hub.window_for` |
| 2 | `daphnia/workflow.py` becomes a **per-surface policy** (`StudyPolicy` strict, `BenchPolicy`/`DatasetPolicy` light) instead of a global gate | proposed - `workflow.py` is still the only policy |
| 3 | OPTICS extracted from the linear Study path | proposed |
| 4 | Bench/Dataset rows use `study_id = NULL`, promotable to a study later | proposed |
| 5 | `discarded` must stop meaning "safe to delete" in DATASET | proposed |
| 6 | Dataset export must include `p1_*`/`p2_*`; split **by specimen/study**, not by frame | proposed |
| 7 | If surfaces are separate **processes**, add `PRAGMA journal_mode=WAL` + `busy_timeout` | open — needed before multi-process |
| 8 | Exact format of the requirement docs — match the "marble machine" project if it can be supplied | **blocked**: that project was not found on this machine |
| 9 | A calibration is an **immutable dated record** grouped by `setup_key`; `register_calibration` always inserts | **implemented** |
| 10 | The scale uncertainty is **derived from the clicked reference span**, and travels with each measurement | **implemented** - see `SR-OPTICS-005`, `SR-CORE-016` |
| 11 | Calibration is a separate surface (`optics`), reachable at any point incl. mid-study | **implemented** - `CalibrationWindow` + the shared footer in every window |
| 12 | The active scale should be resolvable **per optical setup** (for smaller organisms) | open - grouping landed, per-setup selection is next (`SR-OPTICS-006`) |
| 13 | A **dataset** is the corpus of exactly one **model**; the model's schema fixes what a label is (2/4 keypoints, boxes) | **implemented** - `daphnia.annotation`, `datasets`, `SR-DATASET-015..018` |
| 14 | The corpus table is **rebuilt** (copy, not truncate) to scope its uniqueness by dataset | **implemented** - `connection.rescope_corpus_labels()`, verified on the live DB |
| 15 | Canvas gestures: **scroll pans only** (never zooms), pinch/buttons/keys zoom, **double click = next frame** and stores nothing | **implemented** - `ImageCanvas.wheelEvent`/`mouseDoubleClickEvent`, `Dataset.next_frame`, `SR-DATASET-019` |
| 16 | A paint handler that raises must be caught by a **test that opens its own QPainter** (the only way; the event loop swallows it and segfaults the app) | **implemented** - `tests/test_image_canvas.py::PaintTests` |
| 17 | **Real-time detection is a surface of its own** (`predict`), and it is the **Localize** step: the same `Detector` serves the live view, a video file and one image, and returns boxes in **original image pixels** | **implemented** - `daphnia/vision/rtmdet/detect.py`, `daphnia/vision/detection.localize_well`, `SR-PREDICT-001..008` |
| 18 | A **prediction is never stored**; only a human accept may make it ground truth | **implemented** - the Predict surface has no write path (`SR-PREDICT-N01`); pre-labels (`source='model'`) are the accepted next step |
| 19 | A corpus split **shall hold out whole capture sessions**, because the export's split-by-image leaks sibling frames | **implemented** - the trainer defaults to `--split-by group`; the export's own split stays available and is labelled as leaky (`SR-DATASET-010` is the durable fix) |
| 20 | The detector is **RTMDet re-implemented in plain PyTorch** (`mmcv`/`mmdet` do not install on Apple-silicon macOS), with the differences from the reference implementation **listed in `docs/rtmdet.md`** | **implemented** - `daphnia/vision/rtmdet/`; PyTorch imported lazily so the app runs without it |
| 21 | A **box corner dragged past the image** is clamped onto the border rather than stored outside it | **implemented** - `ImageCanvas._to_image(clamp=True)`, `SR-DATASET-020`; 11 of 526 boxes in the real corpus were repaired by `scripts/repair_boxes.py` |
| 22 | The annotator's unit of work is **one box = one animal**, not one frame: the queue is built from the **corpus** (so a hand-added box is queued) and each item is shown as a magnified crop of its box | **implemented** - `Dataset._items`/`_show_item`, `SR-DATASET-022`, `SR-DATASET-025` |
| 23 | A **`Crop` owns the crop↔frame mapping and its `invert`**; a click is converted to source pixels before anything is stored, and the magnification is an integer factor with nearest-neighbour resampling (exact, no invented detail) | **implemented** - `daphnia/vision/crop.py`, `tests/test_crop.py`; the same pattern as `Letterbox` (`mlContext.md` law 1) |
| 24 | A **`source='model'` box becomes ground truth only by a person's accept**; a rejected prediction is deleted rather than kept, and the hard-negative store that would keep it is `SR-DATASET-014` | **implemented** - `repository.set_corpus_box_source`, `Dataset.accept_box`/`reject_box`, `SR-DATASET-007` |
| 25 | The **skeleton** (which keypoints are connected) is part of the schema and of the export, not something a training config guesses | **implemented** - `KeypointSystem.edges`, `AnnotationSchema.export_edges()`, `labels.json.schema.edges`, `SR-DATASET-023` |
| 26 | An exported file **shall** be readable by the tools that read it: every `boxes.csv` row has as many values as the header has names, and a file that disagrees is **refused** rather than read shifted | **implemented** - `maintenance._write_dataset_boxes`, `export.read_boxes`, `SR-DATASET-024` |
| 27 | A box corpus's two halves are **asked for, in plain fields**: the boxes file and the folder its images are in (suggested, never guessed) - a `boxes.csv` names its frames relative to the export folder | **implemented** - `BoxesExportDialog`, `suggested_images_dir`, `read_boxes(images_dir=...)` |
| 28 | One queued animal has **two views** - the whole frame with every box, or the magnified crop - and the view is **only** a view: the frame stays the coordinate space, so switching cannot change what a label stores | **implemented** - `Dataset._crop_view`/`set_view`, `_crop_point`, `_refresh_item_overlays`, `SR-DATASET-025` |
| 29 | A **modal question never lives in the dialog-free core**: the button may ask (and here it offers the dataset that *can* hold a box), `open_boxes_export` only refuses in words | **implemented** - `choose_boxes_export` vs `open_boxes_export`, `switch_to_box_dataset` (a `QMessageBox` in the core hung the suite) |
| 30 | A tool cluster **re-flows** when a mode hides a tool: no empty cell, and no `wide` tool on top of the button beside it | **implemented** - `ToolCluster.relayout()`/`set_tool_visible`, `tests/test_canvas_workspace.py` |

## Where the detector's output goes

| Sink | Notes |
| --- | --- |
| `localize_well()` (`vision/detection.py`) | The pipeline's **Localize** step now returns one `Detection` per animal from the trained detector; raises `NotImplementedError` naming `make train` when there is none. |
| **Predict surface** (`pages/predict.py`) | The same `Detector` in an `InferenceWorker` thread: live feed / video file / one image, boxes + scores + measured ms/frame, nothing stored. |
| `Detector` (`vision/rtmdet/detect.py`) | Usable from a script or a notebook: `load_detector(path).detect(frame)`. |

## The detector (`daphnia/vision/rtmdet/`) — what a reader must know

| Piece | Module | The thing to remember |
| --- | --- | --- |
| Architecture | `model.py` | CSPNeXt backbone (deep stem, depthwise-separable CSP stages), CSPNeXtPAFPN neck, **one** head whose convs are shared across levels with a **separate BatchNorm per level**, one prior per location, four predicted distances. `tiny` = 316 k parameters. |
| Losses | `losses.py` | **SimOTA** assignment (centre prior, `k = max(1, ⌊Σ top-10 IoU⌋)`, contested locations go to the cheapest animal), **QualityFocalLoss** with IoU-valued soft targets, **GIoU** on the decoded box, NumPy NMS. |
| Data | `data.py` | Letterbox to the longest side, pad to a multiple of **32**, and the pad is **recorded** in a `Letterbox` whose `invert()` is the only way back to original pixels. Mosaic / flip / colour jitter. BGR normalisation with RTMDet's numbers. |
| Training | `train.py` | Warm-up + cosine AdamW, `--device auto` (MPS or CPU; asking for MPS where there is none is an **error**), atomic checkpoints that carry preset + input size + corpus, and an AP50 that is **this script's own IoU-0.5 score, not COCO mAP**. |
| Inference | `detect.py` | `Detector` owns the thresholds, the device and the map-back. `ModelUnavailable` is a *named* failure — a surface says "train one first", never shows an empty view. |

**Read `docs/rtmdet.md` before comparing a number with anything.** It lists exactly
where this differs from the reference implementation (block internals, three stages
instead of four, letterbox instead of square pad, NumPy NMS, no EMA/AMP) — anything
not on that list that behaves differently is a bug, not a choice.

## Corpus validity: the two things that were wrong

1. **A box could be stored outside its image.** `ImageCanvas._to_image(clamp=True)`
   pulls a box corner back onto the border (a *landmark* click is not clamped — a
   click outside the image is not a landmark). `annotation.MIN_BOX_PIXELS` is the one
   definition of "too small to be a specimen", used by the canvas, `vision/export.py`
   and `maintenance`.
2. **The export's split leaks.** It hashes the image path, so sibling frames of one
   capture land in train *and* validation. `Export.groups()` / `regroup()` /
   `warnings()` exist for this, and the trainer defaults to `--split-by group`.

Neither is repaired silently: `scripts/repair_boxes.py` is a **dry run** unless
`--apply`, and the trainer prints the warnings before it trains.


## Next steps

0. **Predict surface follow-ups**: pre-labels for the Dataset surface
   (`source='model'`, `SR-PREDICT-009`), the RTMPose keypoint model on top of the
   boxes, and "measure these boxes" (the Localize → Predict hand-off). Then the
   **Dataset surface follow-ups** the corpus still needs: the specimen/study split
   (`SR-DATASET-010` — the split leak is real, see the focus above), resumable
   sessions, hard negatives, and *more distinct captures* (14 is very few: each
   session is a group, and the grouped val split is one capture). **Also open:** the
   keypoint system is now a *dataset* choice; offering the same 2-vs-4-point choice
   on the Bench/Study annotate screens means making it a `studies` mode axis
   (`daphnia/study.py`).
0a. **Train the first real detector** — the machinery is proven (12 smoke steps on
   MPS, forward + backward, checkpoint, evaluation), but no model has been trained to
   convergence yet, so `models/rtmdet_daphnia.pt` does not exist and the Predict
   surface correctly says so. `make train` is the run; `--epochs 100` is the default.
1. Confirm the requirement format (decision 8) - the one open question.
2. Fix the small high-leverage defects, in this order:
   export `p1/p2` -> open-image-file + Bench -> prune must not eat the corpus.
3. Turn `workflow.py` into gate policies, so BENCH/DATASET can skip the study
   gates instead of inheriting them.
4. Build BENCH (the colleagues' blocker): `Open image...`, snapshot, two clicks.
5. ~~Then DATASET (the thesis blocker): labelling at volume + the dataset export.~~
   **Done** - `python run.py dataset` labels a frame queue into `corpus_labels` and
   exports images + labels + manifest. *Next:* pre-labels (`source='model'`), the
   specimen/study split (`SR-DATASET-010`), resumable sessions, hard negatives.
6. Register each new surface's window in `hub.WINDOW_FACTORIES` and flip its
   registry status from `planned` to `available` - the hub then offers it as
   "Open" automatically, with no other code change.
7. **Verify the printed grid once against a stage micrometer** and record the
   offset: that converts the paper's 0.5 % *unknown* error into a known
   correction, and is the single cheapest accuracy win available.
8. Validate `CLICK_ERROR_PX` / `LINE_SIGMA_PX` by calibrating the same slide
   repeatedly and comparing the spread.

## Patterns and preferences

- **For any ML/DL work, invoke the expert stance first: `memory-bank/mlContext.md`**
  (trigger: `.clinerules/ml-engineering.md`). Name the sub-role (detection/pose,
  evaluation, data-pipeline, performance, reproducibility), cite the reference
  implementation being followed, and report numbers with their provenance. The file
  also holds the ten ML laws, the code shape, and the catalogue of traps already paid
  for — read it before proposing an architecture or a metric.
- Requirements are **numbered, `shall`-phrased, verifiable and traced** to
  `O1..O4` / `D1..D3`. IDs are stable; never renumber, only deprecate.
- Documentation is dense Markdown in `docs/`; the memory bank references it
  rather than duplicating it.
- Domain modules stay **Qt-free** so they remain unit-testable and reusable from
  scripts/notebooks.
- **A number is never reported alone**: corpus, split, threshold, epoch, device — or it
  is not reported (see `mlContext.md` §8).
