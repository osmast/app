# Calibrate surface — redesign handover

**For the next agent. Status: S1 is implemented (2026-10-20).** Scope of this document:
**S1** (front-end alignment) — no schema change, no distortion model, no new target.
S2/S3 (§8) are still designed and not built.

What S1 changed, and where it deviated from this document on purpose:

* **§7's expected verdict is wrong, and the copy won.** A 5.1 % drift is *ten times* the
  print's own 0.5 % tolerance, so the reference photo renders **`WEAK`**, not `MEASURED`.
  §3.3's three-state copy is what the screen implements (the yardstick is the print's
  tolerance, `SR-OPTICS-036`), because `MEASURED` next to `drift ±5.1 %` would contradict
  the line above it. `MEASURED` is the state a rig whose field agrees with the print reaches.
* **The operator's second pass (2026-10-20, same day).** Three changes on top of the plan:
  1. **the distortion is panel 1's fact, and it is optional**: both routes to a scale (grid
     and hand span) only set *one* scale and neither corrects the lens, so
     `label_distortion` moved out of panel 2's result block into panel 1 and now reads
     `Distortion: unmeasured and not corrected - one scale covers the whole frame. Optional:
     it only matters off-centre.` with the long explanation in its tooltip;
  2. **the two routes are separated by a rule**: a `QFrame` (`manual_separator`) sits between
     the result block and the hand route, and the hand copy now says what it *is* -
     `Or by hand - the fallback: ... Coarser than the grid, but it works when the paper is
     not fully in view.`;
  3. **the verdict explains instead of labelling**: `MEASURED`/`WEAK` each say what they mean
     and what to do, and `WEAK` points at panel 1 for the correction that does not exist -
     a verdict the operator has to decode is a verdict nobody acts on.
* **The panel is 500 px wide (`Calibration.PANEL_WIDTH`), not the shared 420.** Measured:
  the content asks ~1131 px of height at 420 px against an ~823 px viewport, which put
  *Store this scale* 83 px below the fold; at 500 px the same content is 1041 px and the
  button is at the fold (8 px) — the `Dataset`/`Bench` precedent, and the knob §5.1 names
  for exactly this ("set `PANEL_WIDTH` rather than shrinking the copy").
* **`label_locked` is gone, `locked_label` stays.** D10 takes the `Fixed` row out of the
  form column; the row is now `label_footnote` (caption `Mount`) + `locked_label` (the
  value). The caption "Fixed" was the form-row styling D10 objects to, and nothing read it.
* **The `STATE` fact has three values, not four.** `measuring` cannot be painted: the
  measurement is synchronous on the GUI thread, so a state that says "measuring" would be a
  lie in a bar read at a glance. `ready | no lattice | stored` are all reachable, and the
  `STATE_*` constants say so.
* **The example length keeps `daphnia.units`' `+/-`** next to the panel's `±`, because that
  line is the shared "a length with its uncertainty" formatter (Bench, Predict and the study
  export use it too) - one formatter, one house style, rather than a fourth spelling of
  plus/minus.
* **Two prose blocks were tightened** (`reminder_label`, `zoom_hint_label`): at a 900 px
  window the panel's content is ~1077 px against an ~823 px viewport, so *Store this scale*
  sat 59 px below the fold; after the tightening it is 29 px, and a maximised window shows
  panel 3 whole. (The old `zoom_hint_label` also still described a *View* cluster that
  `X-N13` had removed.)
* **`make ui-render` now renders the daily act in both schemes** and prints the result
  block, the verdict and the distortion line - the facts a picture cannot be read for.
* **Device switching, then the calibration pair - the operator's follow-up (2026-10-20, same
  day).** Panel 1 is now a **read-only `label - value` list** (`Preset · Setup key · In force ·
  Calibrations · Mount · Distortion`) with two `[Load...] [New]` pairs: the **preset** pair at
  its head (`Load preset...` makes an already-stored device's newest scale the one in force,
  `Calibration.load_preset`, `SR-OPTICS-006`; `New preset...` asks for a name in a popup) and the
  **calibration** pair below the list (`Load calibration...` picks one of *this device's* stored
  calibrations and puts it in force **with its proof photo**, `load_calibration`;
  `New calibration...` asks for a **label**, stored in the new `calibration_profiles.label`, and
  clears the measurement to take a fresh one). The name is entered **only in a popup** and shown
  read-only, so the popup/panel duplication the operator reported is gone, and the panel's loose
  sentences became short values. The `Earlier calibrations…` `combo_history` dropdown was
  **removed** - the operator asked for one gesture (load or start) for a device and for a scale
  alike, so `SR-OPTICS-031` is **revised**: loading a calibration now *switches* the scale in
  force instead of merely viewing its photo. The only schema change is the `label` column.
  `_confirm_scale` (`SR-OPTICS-008`) now guards storing **and** both loads.

Files this document owns: `assets/ui/calibration.ui`, `daphnia/ui/pages/calibration.py`,
`daphnia/ui/widgets/image_canvas.py` (the field-stop overlay), `daphnia/ui/theme.py` +
`assets/theme/daphniascope.qss` (one token pair), `tests/test_calibration.py`,
`tests/test_image_canvas.py`, and one line in `memory-bank/systemPatterns.md` (§3.1's exception).

## 0. The goal (read this before touching anything)

This rig calibrates **daily**: a phone clamped to an eyepiece, 1 mm grid paper at the specimen
plane, a circular field stop (the ocular ring) inside the frame, the lattice tilted a few
degrees, and the printed sheet carrying a heavier line every fifth cell. The Calibrate surface
must make **the daily act** fit that reality:

> **one photo → one press → one result → one stored record**

No distortion model may be introduced (`SR-OPTICS-032`), no target picker may come back
(`SR-OPTICS-019`), and the scale stays **one number plus a separately reported drift**
(`SR-OPTICS-036`). Everything else is arrangement, copy and hierarchy.

## 1. Evidence (verified 2026-10-20 — do not re-derive it)

**The screen can be rendered and looked at** (there is no display in the agent environment):

```bash
make ui-render    # scripts/render_ui.py: every surface in both schemes, plus the Calibrate
                  # screen with a real photo measured -> recordings/ui-render/*.png
                  # (reads a *copy* of zoo.db; the original is never written to)
make ui-check     # existing: every surface paints, goes full screen and comes back
```

Both renders quoted below were made with `make ui-render` (light **and** dark show the same
defects); S1 re-runs it and *looks* at the pictures.

What the renders show, and the store behind them:

| Fact | Value |
| --- | --- |
| panel 1 at rest | `Setup key: PROF_PRESET`, `In force for a *different* preset (PROF_LEICADM500_2X_IPHONE_1.0X_INFINITY): 27.3 px/mm +/- 0.29 %, verified 34.6 h ago. Nothing is stored for the name above yet.` |
| header STATUS at rest | `No calibration yet.` — **contradicts panel 1** |
| menu bar | `File │ Measure │ Source │ Detector │ Pose │ View │ Go │ Help` — the two conclusion actions live only in **Measure** |
| after `Detect grid` | `status_label` **and** `grid_label` both carry the whole `LatticeEstimate.warnings()` paragraph; the header's STATUS KPI elides it |
| measured (this photo) | `27.239 px/cell`, **10 bands**, tilt **−9.0°**, spread **5.1 %**, field stop `(601, 266) r 448 px`, 54 % of frame, uncertainty **±2.54 %** |
| store (live `zoo.db`) | 5 rows, newest active **27.2645 px/mm ±0.29 %** — the *clicked-span* route; it agrees with the grid's 27.239 to **0.09 %** |
| thresholds | `settings.CALIBRATION_MAX_AGE_HOURS = 12`, `settings.CALIBRATION_WEAK_UNCERTAINTY_PCT = 2.0` |
| constants in the page | `EXAMPLE_LENGTH_UM = 160.0`, `CAMERA_ZOOM_RATIO = 1.0`, `FOCUS_MODE = "infinity_locked"`, `MANUAL_SOURCE = "hand-entered"` |

**Keep** (these are right and cost nothing): three panels in path order (`SR-OPTICS-024`), one
name = one optical chain with the *different-preset* warning (`SR-OPTICS-025`), history +
proof photo as a **view, not a switch** (`SR-OPTICS-031`), the example length (`SR-OPTICS-028`),
the lattice drawn on the photo (`SR-OPTICS-026`), the footer (`SR-OPTICS-020`), squares-not-a-pitch
(`SR-OPTICS-027`), every action live with a reason (`X-N10`).

## 2. Discrepancies to fix in S1 (each is front-end only)

| # | Observed | Rule it violates |
| --- | --- | --- |
| **D1** | The step's conclusion is only `Measure ▸ Save this calibration`; panel 3 has no store control | `systemPatterns.md §The calibrate surface` lists **Save as panel 3's content**; `X-N13` moved it to a menu. *This conflict is real and is resolved in §3.* Symptom the operator reports: "we are not able to add a new calibration" |
| **D2** | Panel 3 sits at/below the scroll fold; its bottom row is the *fallback* (hand-entry) | panel order must follow the path (`SR-OPTICS-024`) |
| **D3** | menu says **Measure**, panels say **Preset / Reference / Scale** | two vocabularies for one job |
| **D4** | header `No calibration yet.` **and** panel 1 `In force for a *different* preset …` **and** footer `27.3 px/mm … RE-VERIFY` | three states, one screen; the empty name box yields key `PROF_PRESET`, so the real chain never matches |
| **D5** | the `warnings()` paragraph is in `status_label` **and** `grid_label`, elided in the header | no information hierarchy; one fact, three homes, the visible one truncated |
| **D6** | `Accuracy: +/- 2.54% from the detected grid - weak: click a longer span, or use the detected grid.` | self-contradictory, and it recommends the route that **omits the paper tolerance** — the DB's 0.29 % row vs the 0.50 % row is that bug in the data. With `CALIBRATION_WEAK_UNCERTAINTY_PCT = 2.0` and a grid route that floors at `hypot(0, 0.5)`, the grid is *permanently* labelled weak |
| **D7** | "the local scale moves 5.1 % … **that is lens distortion, measured**" | `SR-OPTICS-032` (distortion unmeasured without a model) + `SR-OPTICS-036` (the **drift** is what is measured) |
| **D8** | panel 2 explains *Source ▸ Use the current frame* / *File ▸ Open a photo…* in prose | `X-N11`'s split is inverted: a **choice** belongs beside its panel |
| **D9** | gaining a target photo has three homes (`Source`, `File`, panel prose) | `systemPatterns.md:723`: an action is a menu item, **a choice stays beside its panel** — one home each |
| **D10** | `Fixed: every calibration records zoom 1.0x and focus at infinity` is a **form row** | not a choice; hardware description was removed on purpose (2026-10-19) |
| **D11** | the amber lattice covers the whole frame; the measured **field stop** is only a sentence | `SR-OPTICS-026` — the drawing is the evidence of what was measured |
| **D12** | `combo_history` item 0 is the instruction *"Show the proof photo of an earlier calibration"*; entries show `147.3 px/mm`, `10.1 px/mm` with no source | an item that is an instruction is a control that lies; `SR-OPTICS-029/031` want the source |
| **D13** | nothing on the screen says **today**; the only time signal is the footer's age | the protocol is daily; the screen should own the daily act |

## 3. The target design (S1)

**Where each kind of statement lives** (this is `systemPatterns.md:749` applied): the **header**
carries facts that are *not* a measurement (which photo, which state); the **panel** carries the
measurement and the record; the **canvas** carries the evidence.

### 3.1 The one real conflict, resolved

`X-N13` says *an action is a menu item*; `SR-OPTICS-024`/`§The calibrate surface` say *Save is
panel 3's content*. Resolution for this screen: **both, deliberately** — a primary
`Store this scale` button in panel 3 **and** the same action in the page's menu. Reason: the
conclusion of a step the operator performs daily must be visible on the panel the step's
doctrine defines, while the menu keeps cross-screen consistency. Write the exception down in
`memory-bank/systemPatterns.md` (one line, in §The calibrate surface) as part of S1.

### 3.2 Group titles and order (`assets/ui/calibration.ui`)

| Group (`QGroupBox` objectName) | Title | Order inside |
| --- | --- | --- |
| `preset_group` | `1 Optical chain` | name → setup key (subtitle) → in-force line → **footnote** (the old `Fixed` row) → `Earlier ▸ combo_history` |
| `reference_group` | `2 Target photo` | `[ Use live frame ] [ Open photo… ]` → `Printed square [1.000] mm` → `[ Measure grid ]` → **result block** → squares row (the hand route, below the measured one) |
| `scale_group` | `3 Scale in force` | S → scale uncertainty + source → example length → `[ Store this scale ]` → `I know the scale already` (reveals the hand-entry row) |

### 3.3 Exact copy (do not invent synonyms)

* Panel 1, in force (three cases, `_in_force_text`):
  * name box **empty**: `In force (any chain): 27.26 px/mm ± 0.29 % · 2026-10-18 13:36`
  * name **matches**: `In force for this name: 27.26 px/mm ± 0.29 % · measured 2.4 h ago`
  * name **differs**: `Nothing for this name yet. In force (other chain): 27.26 px/mm ± 0.29 %`
  * stale (`is_stale(..., CALIBRATION_MAX_AGE_HOURS)`, 12 h) → append `· RE-VERIFY`
* Panel 2, result block — label left, value right, one line each (four `QLabel`s):
  * `period` → `27.239 px/cell · 10 bands · tilt −9.0°`
  * `drift` → `± 5.1 % across the field (centre → edge)`
  * `field` → `paper circle (601, 266) r 448 px · 54 % of the frame`
  * `verdict` → `MEASURED` | `WEAK — the field moves more than the print` | `REFUSED — no lattice in this frame. Fill the field with the paper, or use a frame with the ring out of view.`
  * one line that is always present: `lens distortion: unmeasured — no model for this preset` (`SR-OPTICS-032`)
* Panel 3: `scale uncertainty ± 2.54 %` (grid route) or `± 0.29 % from the clicked span — the print's own 0.5 % is not included` (span route) or `± 0.50 % — you stated it` (hand route, `SR-OPTICS-029`); example stays `SR-OPTICS-028` (`160.0 ± 4.1 µm`).
* Header facts, short and elide-safe: `PHOTO <name> 1280×720`, `STATE ready | measuring | no lattice | stored`.
* Menu group renamed `Calibration`, two items: `Measure grid`, `Store this scale`.

### 3.4 The fixes this copy encodes

| Fix | How |
| --- | --- |
| **D4** (contradiction) | `.ui` default text of `status_label` becomes empty/`Ready.`; the header never repeats panel 1; the *empty name box* case stops claiming a different preset |
| **D5** (wall of text) | `LatticeEstimate.warnings()` is **not** printed into `status_label`; each fact goes to exactly one home: period/drift/field → the result block, refusal reason → `label_verdict`, long caveats → the value's tooltip |
| **D6** (misleading advice) | the "weak" judgement stops recommending the span when the grid produced the number; the span route names the tolerance it omits. Note in code: a threshold on a route whose floor is `hypot(band scatter, 0.5 %)` is not a quality measure |
| **D7** (wording) | "lens distortion" appears only as *unmeasured — no model for this preset*; the measured thing is called **drift** |
| **D8/D9** (choices) | `Use live frame` / `Open photo…` become panel buttons; the page's `use_frame` (Source) and `open_photo` (File) entries are **removed**; the app's `Connect the live feed` entry stays |
| **D10** | the `Fixed` row leaves the form column and becomes a footnote line under panel 1 |
| **D11** | `ImageCanvas.set_lattice(period_px, angle_deg)` **plus** the field-stop circle drawn from `LatticeEstimate.field_stop` (a new overlay call, canvas-side) |
| **D12** | `combo_history` item 0 becomes a real placeholder (`Earlier calibrations…`, not selectable-as-a-command) and each entry carries `date · S · source · (in force)` |

## 4. Traceability

| Change | Requirement / rule | Guarded by |
| --- | --- | --- |
| three panels, path order, group titles | `SR-OPTICS-024` | `tests/test_calibration.py` |
| in-force as three cases (empty / same / other) | `SR-OPTICS-025` | `tests/test_calibration.py` |
| lattice **and** field stop drawn | `SR-OPTICS-026` | `tests/test_image_canvas.py` (+ new case) |
| hand route counts squares, never a pitch | `SR-OPTICS-027` | `tests/test_calibration.py` |
| example length beside the percentage | `SR-OPTICS-028` | `tests/test_calibration.py` |
| hand route is a *claim*, no proof image | `SR-OPTICS-029` | `tests/test_calibration.py`, `test_calibrations.py` |
| printed square stored with the record | `SR-OPTICS-030` | `tests/test_calibrations.py` |
| history is a view, never a switch | `SR-OPTICS-031` | `tests/test_calibration.py` |
| "lens distortion: unmeasured", never measured-without-a-model | `SR-OPTICS-032` | `tests/test_calibration.py` (+ new assertion) |
| drift reported **separate** from the uncertainty | `SR-OPTICS-036` | `tests/test_lattice.py`, `tests/test_calibration.py` |
| footer: scale, uncertainty, age, RE-VERIFY, Calibrate | `SR-OPTICS-020` | `tests/test_ui_smoke.py` |
| **an action is a menu item; a choice stays beside its panel** | `systemPatterns.md:723` | review — `make ui-list` shows the declared tools |
| one name per widget, no promoted widget, no `.qrc` | `§UI conventions` | `tests/test_ui_files.py` |
| panel fits `CONTROLS_WIDTH = 420` (or `PANEL_WIDTH`) | `§UI conventions` | `tests/test_ui_smoke.py::test_every_panel_is_wide_enough_for_what_is_in_it` |
| every action live and explains what it needs | `X-N10` | `tests/test_calibration.py` |

## 5. Implementation plan (S1), file by file

### 5.1 `assets/ui/calibration.ui` — Designer work

```bash
make ui-designer FILE=calibration.ui     # Designer6, from the env
make ui-list                             # after every save: names, promoted widgets, .qrc
```

* Rename the three group titles and reorder the rows exactly as §3.2 says.
* **Add** these widgets (these objectNames, nothing more): `btn_use_frame`, `btn_open_photo`,
  `btn_measure_grid`, `btn_store_scale`, `label_result`, `label_drift`, `label_field_stop`,
  `label_verdict`, `label_distortion`, `label_footnote`.
* **Keep every existing objectName** — page code reaches them by name and
  `tests/test_ui_files.py::test_every_named_control_becomes_an_attribute_of_the_page` enforces
  it. In particular **do not rename** `check_manual_scale`, `spin_manual_px_per_um`,
  `spin_manual_accuracy`, `manual_hint_label`: change their *text* only. Renaming is churn with
  no user-visible gain.
* The `Fixed` row: get its objectName from `make ui-list`, move it to the last row of
  `preset_group`, retitle as a footnote (`mount: zoom 1.0×, focus locked at infinity`).
* Budget: content asks **345 px** of the 420 px panel today. The result block adds four rows in
  the same column as the existing `grid_label`. Measure after the edit
  (`python -m unittest tests.test_ui_smoke -v`); if it does not fit, set `PANEL_WIDTH`
  (the `Dataset.PANEL_WIDTH = 500` precedent) rather than shrinking the copy.

### 5.2 `daphnia/ui/pages/calibration.py`

* `self.workspace.set_group("Calibration")`; tool labels `Measure grid`, `Store this scale`.
* **Wire the new buttons to methods that already exist** — no new business logic:
  `btn_use_frame → self._capture_frame`, `btn_open_photo → self._load_reference_image`,
  `btn_measure_grid → self.measure_grid`, `btn_store_scale → self._save_profile`.
* Remove the page's `use_frame` (Source) and `open_photo` (File) tools — those choices now live
  in panel 2 (§3.4, D9). The app's `Connect the live feed` entry stays untouched.
* Split the refresh: `_refresh_result()` fills the four result labels, the verdict and
  `label_distortion` from the current `LatticeEstimate`; `_refresh_scale()` keeps panel 3.
  `LatticeEstimate.warnings()` becomes a **source of fields, not of sentences**.
* `_in_force_text()`: the three cases of §3.3 + `· RE-VERIFY` when
  `is_stale(active.calibrated_at, settings.CALIBRATION_MAX_AGE_HOURS)`.
* `_refresh_scale()`: the uncertainty sentence carries its route (`from the detected grid` /
  `from the clicked span — the print's own 0.5 % is not included` / `you stated it`), and the
  weak hint never advises the route that produced the number.
* `measure_grid()`: on success set `label_verdict` + a short `status_label`; on
  `LatticeNotFound` put the refusal + next action in `label_verdict` and the long `str(error)`
  in the tooltip. Nothing is written into `status_label` twice.
* Header facts: `Kpi("photo", …)`, `Kpi("state", …)` — drop `Kpi("span", points_label.text())`;
  that instruction belongs to panel 2's squares row.
* Panel 1's combo: item 0 a placeholder, entries as `date · S · source · (in force)`.

### 5.3 `daphnia/ui/widgets/image_canvas.py`

`set_field_stop(centre_x, centre_y, radius)` beside `set_lattice`, with the **same lifecycle**:
cleared whenever the frame changes, never describing a photo it did not come from. Thin ring in
a theme token. One new case in `tests/test_image_canvas.py`.

### 5.4 Theme (`daphnia/ui/theme.py` + `assets/theme/daphniascope.qss`)

One token pair for the verdict (`verdict_ok`, `verdict_warn`) in **both** schemes, plus one
`objectName` block in the QSS. **No layout properties in QSS, no inline `setStyleSheet`** —
`tests/test_theme.py` fails otherwise.

### 5.5 Tests and the visual check

Extend `tests/test_calibration.py`: the three in-force cases; the span route naming the tolerance
it omits; the grid route not advising the span; `lens distortion: unmeasured` present and never
qualified as measured; the drift shown as its own value. Extend `tests/test_image_canvas.py` with
the field-stop overlay. Then the whole gate:

```bash
make ui-list
conda run -n z-app python -m unittest tests.test_calibration tests.test_image_canvas \
      tests.test_ui_files tests.test_ui_smoke -v
make ui-check
make ui-render                                 # then LOOK at recordings/ui-render/*.png
```

## 6. Code bar (backend — read before writing a line)

Requested by the operator, and it matches this repo's own style: **efficient, logical, no comment
bloat, correct on the first run.**

* **Efficient.** Add no work to the paint path. Reuse what already exists: `LatticeEstimate` *is*
  the measurement (a changed printed-square size re-scales it with `dataclasses.replace`, never
  re-detects). One refresh method per panel, called once per change; no loops over widgets, no
  `findChild` in a hot path. No new dependency, no new config flag, no new CLI — if a knob is
  needed, justify it in the docstring or leave it out.
* **Logical.** One home per fact and one source of truth per number: panel 2 reads the
  *measurement* (`LatticeEstimate`), panel 3 reads the *record* (`OpticalProfile` + the row),
  the canvas draws the *evidence*. The page holds no shadow copy of either.
* **Comments.** A docstring states the contract — what goes in, what comes out, **why it is
  shaped that way**. An inline comment exists only for a trap or a measurement ("measured on this
  photo: …"), never to narrate code. No restating the next line, no commented-out code, no
  `TODO` without a name. If a line needs a comment to be understood, change the line.
* **First-run correctness.** Every new widget is wired to an existing method; every label has a
  value for **every** state (`none` / `measured` / `weak` / `refused`); the empty state says what
  to do (`X-N10`); a failure prints the reason and the next action, never a traceback and never
  an empty field. Wire first, then run the gate in §5.5, then look at the render.

## 7. Definition of done (S1)

* [x] `make ui-list` clean: no duplicate names, no promoted widget, no `.qrc`.
* [x] `tests.test_calibration`, `test_image_canvas`, `test_ui_files`, `test_ui_smoke` pass.
* [x] `make ui-check` green in **both** schemes (12 canvases painted, full screen and back).
* [x] The render, looked at: `period 27.239 px/cell · 10 bands · tilt −9.0°`,
      `drift ±5.1 %`, `field r 448 px · 54 %`, `uncertainty ±2.54 %`, verdict
      **`WEAK - the field moves more than the print`** (see the note at the top: the 5.1 %
      drift is ten times the print's 0.5 %, so §3.3's `MEASURED` could not have been right
      here) — reproduced on `PROF_…_133641.png` in both schemes.
* [x] Store is reachable **on the panel** and values land in the store (5 rows still, one
      active): `btn_store_scale` is wired to `_save_profile`, and a test clicks it.
* [x] No contradiction on the rest screen: the header says `PHOTO … / STATE …`, panel 1 says
      *In force (any chain): …*, the footer says the same scale, and `status_label` starts at
      `Ready.`
* [x] Nothing in the pixel path changed (the lattice overlay and the field-stop circle are
      the same paint pass, no new work per frame); no distortion model; no target picker; no
      schema change.
* [x] `memory-bank/systemPatterns.md §The calibrate surface` carries the one-line exception
      from §3.1, and `docs/calibration.md` §4.1/4.2 keep their numbers.

## 8. S2 and S3 (designed, **not** part of S1)

**S2 — make the record carry what the surface measures.** `calibration_profiles` gains
`drift_pct`, the field-stop circle (`field_stop_centre_x/y/radius`) and the pitch's provenance
(`pitch_pct`, `pitch_verified_at`) — declared in `COLUMN_MIGRATIONS` (`daphnia/db/connection.py`),
written by `repository.register_calibration`, surfaced on `OpticalProfile`. Then a measurement's
`uncertainty_pct` becomes the combination that is honest *where it was taken*:
`hypot(2.49, 0.5) = 2.54 %` at the calibrated spot, `hypot(2.49, 5.1) = 5.68 %` at the edge.
The printed paper's 0.5 % applies when the pitch is *stated*; a pitch verified against a stage
micrometer replaces it. **No target picker** (`SR-OPTICS-019`).

**S3 — make it daily.** `workflow.facts()` requires a calibration *for this session*, i.e. the
active row's `last_verified_at` inside `CALIBRATION_MAX_AGE_HOURS` (12 h) **and** its `setup_key`
matching today's mount. The footer's `RE-VERIFY` already uses that threshold, so the screen and the
gate get one meaning of "today".

Neither slice stores a distortion model: the mirror test stays the gate (`SR-OPTICS-033`), and
`SR-OPTICS-032`'s "unmeasured" wording is what the surface says until one passes.

## 9. Traps already paid for (do not re-pay)

* A `QMessageBox` inside a method a test or script drives **hangs** the suite — dialogs live in
  handlers; the core method refuses in words.
* `isHidden()` is true for every child of a window that has not been shown yet.
* No button takes focus (`Qt.NoFocus`); tool clusters set `WA_NoMousePropagation`.
* `&` in widget text is a mnemonic — `&&` for a literal.
* Full screen hides chrome through `WINDOW_CHROME`/`page_chrome()`; the canvas workspace is
  **code**, not `.ui`.
* QSS is a CSS2 subset (**no layout properties**) and no screen may call `setStyleSheet`
  (`tests/test_theme.py`).
* `GridEstimate.distortion_pct = 0.0` means *not computed* — never render it as zero distortion.
* The span route's missing 0.5 % is **in the store today** (the 0.29 % row): S1 must display it,
  not hide it; S2 fixes the arithmetic.
* `post.unwarp_line_backward` is a silent no-op on float32 (`docs/calibration.md` §4.1) — reason
  enough never to wire unwarping into this screen.

## 10. Open questions (operator's, not the agent's)

1. What is **a session** for the gate — per day, per mount, per study visit? (S3 needs the answer.)
2. Will the target be replaced (single weight, or a multi-pitch sheet), and who verifies the pitch
   against a stage micrometer? (S2's pitch provenance is only useful if someone does it.)
3. Is the specimen's position in the frame **controlled** (same well, same zoom, same clamp)?
   If yes, S2's drift-by-position matters far less and the daily act stays a scale measurement.





