# System Requirements — Framework

Purpose: freeze **what each surface must do** so the project cannot drift from
its end goal (`O1..O4` / `D1..D3` in `memory-bank/projectbrief.md`).

> **Format note.** This format is IEEE-830-derived (numbered, `shall`-phrased,
> verifiable, traced). It was chosen because the "marble machine" project's
> requirement documents were **not found on this machine**; if that project's
> files can be supplied, these documents will be re-cut to match it exactly.

## Surfaces

The repository is one repo, one Python package (`daphnia`), one SQLite store and
one `recordings/` tree. It exposes **several launch points (surfaces)**:

| Surface | Launch point | Document | Serves |
| --- | --- | --- | --- |
| **CORE** | library, no window | `SR-CORE.md` | shared domain + persistence |
| **OPTICS** | `python run.py optics` | `SR-OPTICS.md` | optical presets + objectives |
| **BENCH** | `python run.py bench` | `SR-BENCH.md` | a number now, no study |
| **DATASET** | `python run.py dataset` | `SR-DATASET.md` | the training corpus |
| **PREDICT** | `python run.py predict` | `SR-PREDICT.md` | real-time RTMDet detection (Localize) |
| **STUDY** | `python run.py study` | `SR-STUDY.md` | days/weeks studies |

## Requirement ID scheme

```
SR-<SURFACE>-<NNN>[<suffix>]
```

- `SURFACE` in `CORE · OPTICS · BENCH · DATASET · PREDICT · STUDY`
- `NNN` = three-digit number, unique within the surface, **never reused**
- optional lowercase suffix letter for sub-items of an existing requirement
  (the suffix `a`, `b`, ... appended to the parent ID)
- non-functional / cross-cutting requirements carry a `-N` group:
  `SR-<SURFACE>-N01`

**Change control:** IDs are immutable. A superseded requirement is marked
`deprecated by SR-X-NNN`, never renumbered and never silently deleted.

## Requirement record format

| Column | Meaning |
| --- | --- |
| **ID** | stable identifier |
| **Requirement** | one sentence, subject + **shall** + observable behaviour |
| **Rationale** | why it exists (or "trap avoided") |
| **Verification** | `T` Test · `D` Demo · `I` Inspection · `A` Analysis, plus the artefact (`tests/test_x.py::case`) |
| **Status** | `proposed` · `accepted` · `implemented` · `verified` · `deprecated` |
| **Traces** | `O1..O4` / `D1..D3` |

A requirement with no `Verification` is not a requirement — it is a wish.

## Traceability matrix (rolling)

| Chapter-1 objective / deliverable | Surfaces that must serve it |
| --- | --- |
| **O1** standardized acquisition workflow | STUDY, OPTICS, CORE |
| **O2** CV pipeline (keypoints -> length) | CORE, PREDICT |
| **O3** evaluate accuracy vs expert-annotated benchmark | DATASET, PREDICT, OPTICS, CORE |
| **O4** packaged standalone desktop application | BENCH, STUDY, OPTICS, PREDICT |
| **D1** curated annotated benchmark dataset | DATASET |
| **D2** open-source codebase + trained weights | CORE, DATASET, PREDICT |
| **D3** packaged desktop GUI | BENCH, STUDY, OPTICS, PREDICT |

## Cross-cutting (non-functional) requirements

These apply to **every** surface; ids `X-N01..`:

| ID | Requirement | Verification | Status |
| --- | --- | --- | --- |
| `X-N01` | Every persisted measurement **shall** record the scale actually applied (`px_per_um`), its uncertainty, and the frame it was drawn on. | `T` `tests/test_calibrations.py` | implemented |
| `X-N02` | No surface **shall** delete a file that carries a label as a side effect of curation. | `T` `tests/test_maintenance.py` | accepted |
| `X-N03` | Domain modules **shall** import no Qt; surfaces **shall** hold no pipeline logic. | `T` `tests/test_surfaces.py::PurityTests` | implemented |
| `X-N04` | A surface **shall** not import another surface's pages; only the hub and the launch dispatch may reference surface windows. | `I` inspection (automated guard still to add) | accepted |
| `X-N05` | Any number shown without a calibration **shall** be labelled as unscaled. | `D` | proposed |
| `X-N06` | Adding a column **shall** ship with a `COLUMN_MIGRATIONS` entry (never a silent schema change). | `T` `tests/test_db.py` | accepted |
| `X-N07` | Every new surface **shall** extend `tests/test_ui_smoke.py` with its page/attribute surface. | `T` `tests/test_ui_smoke.py::HubTests` | implemented |
| `X-N08` | Terminology **shall** come from `daphnia.study` (`specimen`, `endpoint`, `keypoints`) — no surface invents its own wording. | `I` | proposed |
| `X-N09` | Every surface **shall** show the shared status footer - the camera state and the scale in force - so both read identically everywhere and are actionable from anywhere. | `T` `tests/test_ui_smoke.py` | implemented |
| `X-N10` | No control **shall** be presented disabled without saying why on screen; an action that cannot proceed **shall** state what is missing when it is used. | A greyed-out button swallows the click: that is how "Detect 1 mm grid" looked broken while a photo was simply absent. | `T` `tests/test_calibration.py` | implemented |
| `X-N11` | ~~Every surface **shall** present the controls that act on the image **on the image**~~ - **superseded by `X-N13`** (2026-10-17): the controls are **menu items in the window header** now, and only a page's own **draw tools** (`X-N20`) stay on the picture - the numbers are read in the panel and the facts in the header. The reason it was written (the gesture happened in one place and the button answering it in another) still holds; the *place* moved from the image to the bar. | One screen's controls no longer sit on the animal being measured, and the header is the one bar every window already carries. | `T` `tests/test_canvas_workspace.py`, `tests/test_ui_smoke.py` | superseded |
| `X-N12` | Every canvas surface **shall** offer a **full-screen** view in which the canvas is the only thing on screen - the controls panel, the live preview and the window chrome hidden - with `Esc` leaving it, and **shall** add nothing of its own to the picture. ~~**shall** keep the measurement readout visible on the image~~ - **superseded (2026-10-18)**: every fact the readout carried is read in the panel and the window header, and nothing but a page's own **draw tools** (`X-N20`) is drawn over the photograph. | "Work on the image alone" is only true if the things you need while measuring are still reachable *and* the state you entered is kept: the specimen fills the screen, the canvas keeps its points, its zoom and its tools, and `Esc` (or the tool button, on a device without a keyboard) brings everything back. The first version of this rule also repeated the measurement as a strip of cards over the image - the operator asked for the picture to be the picture, and a number that is already in the panel on the same screen does not need to be written over the animal as well. | `T` `tests/test_canvas_workspace.py` | implemented (readout clause superseded) |
| `X-N13` | Every window **shall** carry a **header** at the top that holds the application's menus **and** the page's own actions as menus, together with the facts that are **not** a measurement - which file or frame is on screen, which model answered, the last message - and the surface's numbers are read **in its own panel** (~~the numbers stay **on the image as a one-line strip of cards**~~; **superseded 2026-10-18**): **nothing** is drawn over the picture but a page's own **draw tools** (`X-N20`). The menus **shall** be read in one fixed order - **File** (what the window opens and exports, then *Exit*), the **page's own menus** (*Measure*, *Label*, *Frames*...) directly after it, then **Source** (the one shared camera and what the surface looks at), **Models** (which checkpoint answers), **View** (the canvas' view actions), **Go** (Home and the other surfaces) and **Help** (the calibration walkthrough, where the data lives) - and a menu with nothing behind it **shall** be hidden rather than shown empty. A page's *File* and *Source* entries **shall** sort into those shared menus, never into a menu of their own. An **action** **shall** be a menu item; a **choice** (a combo together with the button that applies it) stays beside its panel. The header **shall** be window chrome (hidden in full screen) and **shall** elide a long value rather than wrap, never widening the window. | The readout over the image had grown into a paragraph in which the number had to be found, and the controls floated over the animal they acted on. Splitting by *kind* puts each thing where it is read: the number in the panel the operator is already working in, the context and the actions in the window - and the picture free of everything that is not the specimen. A **fixed order** is what makes the bar learnable - *File* first, *Help* last, the work in between - instead of an accumulation in the order the code happened to build it; and "action in the bar, choice in the panel" is the one line that says where every new control goes. | `T` `tests/test_ui_smoke.py`, `tests/test_canvas_workspace.py` | implemented (the on-image strip superseded) |
| `X-N14` | What a left press **means** on a canvas **shall** be an explicit **tool** - `line` (place two points that are one measured line), `keypoints` (place the schema's anatomical keypoints, drawn as the schema's bones), `box` (draw a new box and edit the boxes on the frame) or `select` (pick what is under the press and edit it) - offered as **one mutually-exclusive group of menu items**, and a create tool **shall** be what a measuring surface opens in. A press **shall not** depend on an "edit mode" the operator has to switch **off** before a measurement can be placed. In the `box` tool a press on **empty space** **shall** anchor a box corner (so a box can be *drawn* as well as edited) and a press on a stored box **shall** select it; only the `select` tool **shall** marquee. The canvas **shall** expose the chosen tool as :attr:`ImageCanvas.tool` and **shall** keep the older `box_edit`/`box_mode` flags working. | Editing was one boolean (*Edit*, on by default) doing four jobs at once, so a page could not offer *select* without turning *create* off - and the Bench opened in an edit mode where the first click did not measure, forcing a mode switch before **any** point could be placed ("the amount of buttons to get a measurement"). And a keypoint corpus' bones (`KeypointSystem.edges`) were never drawn, because a *mode* could not be named for them. | `T` `tests/test_image_canvas.py::ToolTests`, `tests/test_bench.py::BenchToolTests` | implemented |
| `X-N15` | A keypoint, a stored box or a prediction **under the cursor** **shall** be **grabbable in any tool**: the canvas **shall** report it (`ImageCanvas.hover`), show it (a hover outline/ring in the accent colour) and offer a grab cursor on hover, and a **press on it `shall` edit it right away** - select/move/resize/delete a box, drag a point, or pick a model's box into the draft - **without** the operator switching to the *Select* tool first. A press on **empty space `shall` still run the tool's own gesture** (place a point, anchor a box), so measuring is unchanged; the hit test **shall** be the one object (`_hit_editable`) that both the press and the hover consult, so they can never disagree. | A nearby target could only be edited after switching to *Select*, so a small correction to a point, spine, box or prediction cost a tool change and back. Showing what is under the cursor before the press is what makes "click it and edit right away" safe rather than a guess. | `T` `tests/test_image_canvas.py::ToolTests`, `tests/test_bench.py::BenchToolTests` | implemented |
| `X-N16` | A **stored measurement** drawn as an annotation (a Bench line, a Dataset keypoint pair) **shall** stay editable after it is stored: when the page turns `annotation_edit` on, hovering one of its **ends** **shall** show the grab affordance, and a press on an end **shall** report the exact measurement (`annotation_point_selected`: specimen, segment, endpoint) so the page loads that stored row back into the editable draft and the **same press drags the point** - a correction, stored on the drop, never a second measurement. Which measurement a segment is **shall** be carried by an **opaque per-segment tag** (`set_annotations`), never guessed from draw order; and the grab **shall not** run in the crop's `point_mode` (there the press places a landmark, `SR-DATASET-028`). A measuring surface that has taken a frame **shall** switch to **image mode** - the live preview hidden so the frozen image fills the visual area (`StreamView.set_preview_enabled`), the camera still running - and **shall** offer a *Live preview* switch to bring the video back. | A measurement drawn an hour ago could only be corrected by re-measuring it or switching tools, so a one-pixel fix cost the whole gesture; and on the Bench the live video kept playing beside the frozen frame that was actually being measured, which is both noise and a source of mis-clicks. | `T` `tests/test_image_canvas.py::ToolTests`, `tests/test_bench.py::BenchToolTests`, `tests/test_dataset.py::DatasetAnnotationEditTests` | implemented |
| `X-N17` | The canvas **shall** treat every object it shows - a **box**, a **measurement** and a **prediction** - as **selectable on the same terms**: a press selects it, **Shift/Cmd adds to or removes from** the selection, a **marquee** selects every object it covers (boxes, measurements *and* predictions), every selected object **shall be drawn as selected** (never hidden), and **one `Backspace` shall remove everything selected** - boxes via `box_delete_requested`, measurements via `annotation_delete_requested`, predictions via `prediction_removed`, each decided by the page. A **selected prediction shall stay visible** (highlighted), and the *Select* tool **shall only select** a prediction - promoting a model's answer into a measurement/label is the **measuring** tools' gesture, so *Select* can touch a prediction without accepting it. A prediction that **is** promoted **shall** be built from the **final** geometry the operator dropped (never the untouched suggestion, so no ghost survives), and **what was aimed at shall decide the kind**: a press on the predicted **box** makes a box, a press on one of its **keypoints** makes a line. A page whose boxes are **measurements** (objects inside a rectangle) **shall** be able to require a box to be grabbed by its **outline** (`ImageCanvas.box_edge_grab`), so a press inside the box reaches the keypoints and measurements drawn in it; a press inside a box that is **already selected** **shall** still move it. A model's answer is **two independent objects** - the **box** RTMDet found and the **keypoints** RTMPose placed in it - so accepting one **shall not** remove the other (`drop_prediction(index, part=...)`), and removing one object **shall never** remove another one. | "I'm not able to select multiple objects", "I'd like to select predictions too" - a selection that held one kind at a time, hid the prediction it picked and could not reach a raw model box was an object model nobody could predict. One selection, one delete key, one visible highlight per object is the "clean flow" the operator asked for; and a detection+pose answer must not turn a box click into a keypoint line behind the operator's back - nor take the keypoints away with it. | `T` `tests/test_image_canvas.py::ToolTests`, `tests/test_bench.py::BenchBoxTests` | implemented |
| `X-N18` | Every tool a canvas offers **shall** be reachable by **one key**, the key **shall** be shown in the tool's menu text (`Line (L)`, `Select (V)`), and the shortcut **shall** be bound at **window** scope - so it works wherever the keyboard focus is and switching tools never costs a trip to the menu bar (`CanvasWorkspace.set_tool_shortcuts`). | The image editors this operator knows put the tool one key away (Photoshop's `V`, `B`, `L`); a tool that is only in a menu is a tool one stays in, which is exactly the reported trouble - "when in box mode we can't switch quickly to line or point mode". | `T` `tests/test_bench.py::BenchPageTests` | implemented |
| `X-N19` | A press on an **object** **shall** switch the page's tool to the one that **owns** that object - a stored **box** or a predicted box to the Box tool, a **measurement** to the measuring tool, a model's **keypoints** to the keypoint tool - **in any mode**, and the switch **shall not disturb the press it followed** (the selection, the drag and the points already placed are kept; `ImageCanvas.set_tool(..., keep=True)`, `object_pressed`). A promoted prediction **shall** be reported **once** (through `prediction_accepted`) so a page acting on both signals cannot make two of it. | "When clicking on an object, switch to that object mode - when in selection mode and selecting an object, or in another mode and clicking another object." Following the cursor is how a single click both reaches an object and leaves the right tool for it, so the *next* press means what the object needs. | `T` `tests/test_image_canvas.py::ToolTests`, `tests/test_bench.py::BenchBoxTests` | implemented |


| `X-N20` | **Touching an object is not a mode**: on a page whose tools are for **drawing**, a press on a box, a measurement or a prediction **shall** select, drag, resize or remove it **whatever** draw tool is in force - the tool decides only what a press on **empty space** draws - so no *Select* tool is needed beside them; and every draw tool **shall** be offered **on the canvas** as a floating bar **as well as** in the menu, both bound to the one action so the two can never disagree; and a draw tool **shall** be able to be turned **off** - *no tool armed* being a state of its own, the **normal view**, where a press selects what is under it and **arms nothing** (`X-N19`'s tool-follows-the-object switch does not fire there, or the normal view would last one click) - reachable both by clicking the armed tool again and by its key (`X-N18`). | A mode you must leave before you may touch something is a mode that eats clicks; a tool you have to travel to the menu bar for gets used wrongly; and a set of tools you can only *switch between* leaves no way to simply look, select and drag without arming something. | `T` `tests/test_bench.py::BenchPageTests` | implemented |
| `X-N21` | A **measurement** **shall** be attributable to the **individual** it was taken on: an individual is the **box** that stands for one animal (`observations.individual_index`, `corpus_boxes`' one box per specimen, `vision.tracking.Track`), a measurement belongs to the individual whose box contains its **midpoint** - the **innermost** box winning when several contain it - in **source-image pixels** of the frame on screen, and a measurement that falls inside **no** box belongs to **none**, which the surface **shall** show as such rather than attach to the nearest animal. The attribution **shall** be recorded with the row (`measurements.individual_index`, `NULL` for none) and **shall not** be part of the row key, so one individual may hold two measurements of one trait. | "The box itself should be an animal or individual", and every number should say which animal it is about - a length that cannot be attributed cannot be compared, and a rule with no stated frame of reference is how two numbers get quietly mixed. Recording it makes the grouping auditable after the fact; keeping it out of the key is what lets one animal hold two lengths. | `T` `tests/test_individuals.py` | implemented |

## How to add a requirement

1. Pick the surface; take the next free `NNN` in that file.
2. Write it as `shall` + observable behaviour. No implementation detail.
3. Name the verification artefact **before** implementation.
4. Add its row to the surface's table and, if it serves a new objective, to the
   matrix above.
5. If it changes an existing behaviour, mark the old row `deprecated by <new id>`.

## How to add a surface

1. Write `SR-<SURFACE>.md` first, tracing to `O1..O4`/`D1..D3`.
2. Define its **gate policy** (what must hold before it acts) — see
   `memory-bank/systemPatterns.md`.
3. Define its **launch point** in `run.py` / `daphnia.ui.app.main`.
4. Add its `.ui` files under `assets/ui/`, its pages under
   `daphnia/ui/pages/<surface>/`, and its smoke tests.
5. Update the surface table above and `memory-bank/progress.md`.
