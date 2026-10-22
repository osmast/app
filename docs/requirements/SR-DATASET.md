# SR-DATASET — corpus & annotation requirements

Serves: **D1** (curated annotated benchmark dataset), **O3** (evaluation), **O2**
(training the keypoint model). Launched with `python run.py dataset`.

Audience: the thesis itself. This surface produces the data the model is trained
and validated on, so it must **maximise labelled volume** — the opposite of the
Study surface's "exactly one frame per target".

## Gate policy

Only one thing must hold: **frames exist** (`DatasetPolicy`). Labels are optional
at any moment; an unlabelled frame is a queue item, never a failure.

## Functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-DATASET-001` | The surface **shall** allow **many frames per specimen** to be labelled and kept. | Volume comes from bursts (`b01_f00..f04`), which today are reduced to one frame per target. | `T` | implemented | D1 |
| `SR-DATASET-002` | The surface **shall** label a frame without requiring it to be the "best" frame of its target. | Blur, tilt and occlusion are exactly the hard cases the model must learn. | `T` | implemented | D1, O3 |
| `SR-DATASET-003` | Curation **shall not** delete any frame that carries a measurement, in this surface or any other. | `prune_discarded()` currently removes them; this is the corpus-destroying defect. | `T` `tests/test_maintenance.py` | implemented (this surface never prunes; `corpus_labels` is outside `prune_discarded`) | X-N02, D1 |
| `SR-DATASET-004` | The surface **shall** support several specimens per frame, each annotated with its own landmarks and index. | Uses the existing `measurements` invariant directly. | `T` `tests/test_dataset.py` | implemented | D1 |
| `SR-DATASET-005` | The surface **shall** record both landmark points per specimen and trait, in source-image pixels. | `p1/p2` are the ground truth; the CSV export currently drops them. | `T` `tests/test_dataset.py` | implemented | D1, X-N01 |
| `SR-DATASET-006` | The surface **shall** offer a deterministic queue of unlabelled frames with next/previous navigation and keyboard shortcuts. | Labelling speed is the binding constraint (~4 labels/afternoon today). | `D` + `T` | implemented (sorted queue; A/D and arrows) | D1 |
| `SR-DATASET-007` | The surface **shall** accept model predictions as **pre-labels** (`source = 'model'`) that a human can accept or correct. | Cheap labelling once a first model exists; enables human-in-the-loop iteration. | `T` `tests/test_dataset.py` | implemented for **boxes** (import an export, review each box Yes/No, then label keypoints in its crop); the *Predict* surface still has no "send to dataset" button (`SR-PREDICT-009`) | O2, D1 |
| `SR-DATASET-008` | The surface **shall** keep manual and model labels distinguishable and **shall** be able to report their agreement. | Accuracy claims must separate ground truth from prediction. | `A` + `T` | accepted (store) / proposed (agreement report) | O3 |
| `SR-DATASET-009` | The surface **shall** export the corpus as images + a machine-readable label file containing landmark coordinates, image size, specimen index, trait and scale. | The deliverable D1 must be consumable by a training script without the app. | `T` `tests/test_dataset.py` | implemented (`labels.csv` + `labels.json` + `manifest.json` + `images/`) | D1, D2 |
| `SR-DATASET-010` | The export **shall** include a train/val/test partition by **specimen and study**, recorded in the manifest. | Burst siblings are near-duplicates; a frame-level split leaks and inflates scores. | `A` + `T` | implemented as a deterministic split by **image** (a first cut; per specimen/study is the next step) | O3 |
| `SR-DATASET-011` | The surface **shall** report live corpus statistics: frames, labelled specimens, labels per trait, distribution of lengths, and unlabelled backlog. | You cannot manage a corpus you cannot see. | `D` | implemented (frames queued/labelled, labels and per-trait counts; length histogram is next) | D1 |
| `SR-DATASET-012` | The surface **shall** record the operator and timestamp on every label. | Provenance and inter-annotator reliability. | `T` `tests/test_dataset.py` | implemented | D1, O3 |
| `SR-DATASET-013` | The surface **shall** be able to import frames from directories outside a study into the corpus. | Archived and third-party images must be usable. | `T` | implemented (import a folder, recursively; open a single file) | D1 |
| `SR-DATASET-014` | The surface **shall** support deliberate inclusion of **hard negatives** (empty wells, crowded frames, detritus) without them counting as failed labels. | Robustness claims require them. | `D` | proposed | O3 |
| `SR-DATASET-015` | The surface **shall** organise the corpus as **one dataset per model**: creating a dataset fixes the annotation schema (keypoint system and/or bounding boxes) for every label it will hold. | A corpus whose labels mix two keypoint systems cannot train either model; "train another model" must mean "create another dataset". | `T` `tests/test_dataset.py::DatasetModelTests` | implemented | D1, O2 |
| `SR-DATASET-016` | The surface **shall** let the operator choose the keypoint system: **head → tail** (2 points) or **head, tail, sideL, sideR** (4 points), and **shall** offer only the traits of the chosen system. | Two points is enough for a length model; the four-point system also gives the width. Labelling a point the model does not predict is wasted work. | `T` `tests/test_annotation.py` | implemented | D1, O2 |
| `SR-DATASET-017` | The surface **shall** support **bounding-box labelling** (one box per specimen), alone or alongside the keypoints, and **shall** store it in source-image pixels with its dataset and specimen. | It is the ground truth a specimen **detector** needs; real-time box *prediction* is explicitly out of scope for now. | `T` `tests/test_dataset.py`, `tests/test_image_canvas.py` | implemented | D1, O2 |
| `SR-DATASET-018` | The export **shall** be scoped to one dataset and **shall** carry that model's schema (keypoint names, intervals, boxes) and a per-specimen record joining keypoints and box. | A training script must be able to read one folder and know exactly what the model predicts. | `T` `tests/test_dataset.py::DatasetModelTests` | implemented | D1, D2, O3 |
| `SR-DATASET-019` | A **double click** on the image **shall** walk to the next frame and **shall not** store what the gesture itself placed (its first press is taken back, using the same undo a mistaken click gets). | The fastest path through a queue is the mouse alone, and a gesture that silently writes a bogus landmark would poison the ground truth. | `T` `tests/test_image_canvas.py`, `tests/test_dataset.py` | implemented | D1 |
| `SR-DATASET-020` | A **box corner dragged past the image** **shall** be clamped onto the image border, and a box that would collapse below `MIN_BOX_PIXELS` **shall not** be stored. | 11 of 526 boxes in the first real corpus were stored outside the frame (`y1 = -9.8`, `y2 = 724 > 720`), because the drag crossed the letterbox. A detector trained on them learns a rectangle that cannot exist. | `T` `tests/test_image_canvas.py::BoxTests`, `tests/test_export.py` | implemented | D1, O2 |
| `SR-DATASET-021` | The surface **shall warn** when a stored box overlaps an existing box of the same frame above IoU 0.5, **and shall still store it**. | With several individuals per frame, boxing one animal twice is the easy mistake - but a legitimate overlap (an animal in front of another) must remain possible, and a label is never dropped silently (`SR-DATASET-N03`). | `T` `tests/test_image_canvas.py` | implemented | D1 |

| `SR-DATASET-022` | The surface **shall** open the boxes of an existing export - its `boxes.csv` or `labels.json`, with `images/` beside it - into the open dataset and queue them **one animal per item**, and **shall** build that queue from the corpus rather than from the file, so a box added by hand is queued too. | This is the recursive annotator's front door: the boxes that already exist (a detector's predictions, or an earlier session's) are exactly the ones worth reviewing, and a queue built from the file could never gain an animal the detector missed. | `T` `tests/test_dataset.py::ImportBoxesTests`, `::DatasetBoxQueueTests` | implemented | D1, O2 |
| `SR-DATASET-023` | The annotation schema **shall** carry the model's **skeleton** - which keypoints are anatomically connected - and the export **shall** write it in the schema block. | A pose model is points *and* the bones between them; without the bones written down, a training config guesses the connectivity, and two projects that guess differently cannot compare results. | `T` `tests/test_annotation.py::SkeletonTests`, `tests/test_dataset.py` | implemented | D1, O2 |
| `SR-DATASET-024` | Every row of the exported `boxes.csv` **shall** have as many values as the header has names, and the reader **shall refuse** a file where they disagree, naming the fix. | The first writer's header had one name more than its rows had values, so any spreadsheet or CSV reader shifted every column from `image` onwards - plausible numbers, silently wrong, and no test noticed. | `T` `tests/test_export.py::ReadBoxesTests`, `tests/test_dataset.py` | implemented | D1, D2 |
| `SR-DATASET-025` | The surface **shall** label the keypoints of **one specimen at a time on a magnified crop of its box** (an integer factor, nearest-neighbour so the enlargement is exact), and **shall** store every coordinate in **source-image pixels** through one mapping object that owns the way back. | The frame holds a median of three animals at ~84 px each; labelling on the whole frame makes the operator find the same animal twice, and a click stored in the *picture's* coordinates is a wrong label that looks right. | `T` `tests/test_crop.py`, `tests/test_dataset.py::DatasetBoxQueueTests` | implemented | D1, O2 |
| `SR-DATASET-026` | A box already labelled on a frame **shall** be selectable by clicking it, **movable** by dragging its middle and **resizable** by dragging an edge or a corner, **removable** with `Delete`, and while the *Select* tool is on a click **shall not** create a second label. The corrected geometry **shall** be stored for that specimen, in the image's own pixel frame. | "I need to be able to click a box and edit it": before this, every click in box mode started a *new* box, so a wrong or drifted label could only be deleted and redrawn - and a frame with five animals was retyped from scratch. Correcting a label is labelling: the same write (`save_corpus_box`) serves both, so an edited box is a label like any other. | `T` `tests/test_image_canvas.py::BoxEditTests` | implemented | O2, X-N10 |
| `SR-DATASET-027` | **Edit mode `shall` be the default** on every labelling and measuring surface: **one click** inside a box or on a point `shall` select it and `shall` never create a new label; a click on a **model box** `shall` put it in the editable draft (a click stores nothing, a drag **shall** store it as an annotation); a **point**, a **box** and a **model box** `shall` all be draggable, **several boxes `shall` be selectable** (Shift/Cmd-click or a marquee) and `shall` then move and be removed together; **one** press of `Backspace`/`Delete` `shall` remove the selection wherever the keyboard focus is - a labelled box from the corpus, a model box from the frame's prediction layer, or a single point; a **double click `shall not` walk to the next frame** in this mode; and the selected box `shall be drawn once` (the selection outline), not twice. Turning the mode **off** `shall` restore creation (clicks place points, drags draw boxes) and the double-click "next frame" gesture. **Letting go `shall` be the update:** a dropped **box** `shall` be stored, a dropped **point** `shall` write the correction when that (specimen, trait) is already stored - and `shall` do nothing when it is not (a pair that is still being placed waits for *Save* - the crop loop, which consumes the pair on the click and leaves no point on screen, has no drop to answer) - and a point being **moved `shall`** be reported as a move, never as a placement (a page that re-renders on a move is the point snapping back). | "An automatic edit mode, where one does not create a new box ... but one is able to edit the predictions, and by clicking within a bounding box one selects it, and the back button deletes" - and the three failures that came with it: a selection that took two clicks, a `Backspace` that needed several presses, a frame that jumped mid-correction (the double-click binding was still walking the queue), and "two duplicate boxes" (the page sited the stored box a second time as the active box). Then: "I see a ghost lagging behind and when letting go box or point the new position is not updated". | `T` `tests/test_image_canvas.py::EditAllTests`, `tests/test_dataset.py::DatasetModelTests` | implemented | O2, O3, X-N10 |
| `SR-DATASET-028` | In the **magnified crop of a keypoint corpus** the press **shall** be the keypoint gesture (`ImageCanvas.point_mode`): a press within reach of a point `shall` pick that point up, and any other press `shall` place the next landmark - it **shall not** select a box, and empty space **shall not** start a marquee. A box **dragged** while a crop is in force `shall` be stored in **source-image pixels** through the same mapping object the clicks use, with the source frame's own size. | The item's box covers the whole crop, so with *Edit* mode on (the default) every real press selected that box: the prompt asked for *"click the head, then the spine base"* and the clicks went to the box - *"not able to correct the keypoint: it first selects the box"*. Measured on a boxes export: a click at the centre of the crop placed **no** point and selected the box. And a box dragged in the crop was stored **in crop pixels with the crop's size as the image size** - specimen 1 of a 1280 x 720 frame became ``(82, 79, 382, 259)`` on a "450 x 330" image, so the rectangle the operator moved was not the rectangle the corpus kept. | `T` `tests/test_dataset.py::DatasetBoxQueueTests`, `tests/test_image_canvas.py::EditAllTests` | implemented | D1, O2, law 1 |

## Non-functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-DATASET-N01` | The surface **shall** never require a study, plate, connection or calibration to *label*, though a scale is required to *report micrometres*. | Labelling topology is scale-independent; only the report needs it. | `T` `tests/test_dataset.py` | implemented | D1 |
| `SR-DATASET-N02` | A labelling session **shall** be resumable, returning to the last frame. | Sessions are interrupted constantly. | `T` | proposed (the queue is rebuilt by re-import; a stored resume pointer is next) | D1 |
| `SR-DATASET-N03` | The surface **shall not** auto-delete, auto-tray or auto-select anything. | Zero destructive automation. | `I` + `T` | implemented | X-N02 |
| `SR-DATASET-N04` | Annotation latency per specimen **shall** stay within reach of the current ~2.5 % repeatability, i.e. the tool must not be the dominant error term. | A tool slower/more jittery than bare-handed clicking is not worth using. | `A` | proposed | O3 |

## Datasets and models

A **dataset** is the corpus of exactly one model (`datasets` table). Creating one
asks for two things - a name and the **model** - and the model is what fixes the
annotation schema for every label the dataset will ever hold (`daphnia/annotation.py`):

| Schema key | Model target | Keypoints | Traits measured | Box |
| --- | --- | --- | --- | --- |
| `pose_full` | Pose, 4 points (**default**) | `head`, `spine base`, `left side`, `right side` | `length`, `width` | — |
| `pose_head_tail` | Pose, 2 points | `head`, `spine base` | `length` | — |
| `pose_full_boxes` | Pose 4 + detection | the four above | `length`, `width` | yes |
| `boxes` | Detection only | — | — | yes |

The names a *training config* expects (`head`, `tail`, `sideL`, `sideR`) are the
schema's **export names**; the corpus keeps the study's vocabulary internally, so a
corpus label and a study measurement still name the same anatomy (`X-N08`).

Belonging to a dataset is what lets the same frame carry ground truth for two
models: labels are unique per `(dataset, file, specimen, trait)` and boxes per
`(dataset, file, specimen)`, so labelling the same specimen for the length model
and then for the detector never overwrites either.

**Bounding boxes** are labelled either way round, and the two mix freely: **drag** a
rectangle around the specimen, or **click one corner and then the opposite one**
(the first click anchors the corner and shows it, the second finishes the box, `Esc`
or a right click abandons a half-drawn one). A box smaller than 3 px is treated as a
mis-click and never stored, and the box is kept in its own table - a box is not a
distance between two landmarks. A **detection-only** dataset hides the keypoint
controls entirely instead of showing dead ones.

The dataset list is a real control, not a label: every entry shows the model it
belongs to, `New dataset...` creates one, `Rename...` renames the open one (and
locks the model - the schema is what the existing labels *mean*), and a name that is
already taken is **numbered, never refused** (`corpus (2)`), so a collision cannot
cost a labelling session. `B` (or the button) toggles box labelling, and is *refused
in words* by a model that has no box instead of doing nothing.

## Annotation schema

Inside a keypoint dataset, each trait is exactly two landmarks in **source-image
pixels**. The keys match the rest of the app (`length` | `width`) so a corpus label
and a study measurement mean the same thing.

| Trait | Meaning | Landmarks | Model output |
| --- | --- | --- | --- |
| `length` | Body length — the project's fixed endpoint | `head` -> `spine base` (anterior margin of the compound eye -> base of the apical spine) | distance between the pair |
| `width` | Body width / height — across the carapace | `left side` -> `right side` | distance between the pair |

The vocabulary comes from `daphnia/study.py` (`TRAIT_LANDMARKS`, `LANDMARKS`,
`LANDMARK_TRAIT`), shared by the Study, Bench and Dataset surfaces, so every surface
names the same points.

**Two labelling modes** (`daphnia/labeling.py`):

* **two points per specimen** - click both landmarks of the current specimen; the
  interval is saved on *Save*. Best for a single animal.
* **one landmark at a time** - pick a landmark, then click it for every specimen in
  turn (the specimen counter advances itself); the interval is stored the moment
  both of its ends exist, so clicking all the heads, then all the tails, then
  `left side`, then `right side` fills a crowded frame without moving a counter.
  Partial points live in a :class:`~daphnia.labeling.Draft` until they pair up.
  The **pass advances itself**: the first pass sets the specimen count, and once a
  landmark sits on every specimen the next landmark in
  `head -> spine base -> left side -> right side` is selected automatically, so
  neither specimens nor landmarks are ever cycled by hand. *Next landmark* skips
  ahead on demand.

Every specimen is drawn on the canvas in its **own colour with its number**
(`ImageCanvas.set_annotations`, `specimen_color`) **from the moment a point is
placed** - during a landmark pass too, so two heads are already distinguishable. The
canvas **pans with a two-finger scroll (both axes) and never zooms with it** - the
pinch, the *Zoom +/-* tools or `+`/`-`/`F` zoom - and `Command`/`Control`-drag (or the
*Pan* hand tool) moves the view without placing a point.

**The page's own actions are menus in the header** (`X-N13`): a *Label* menu (*Save*,
*Undo*, *Clear points*, *Clear frame*, *Box*, *Yes*/*No*, *Crop*/*Frame*, and
*< Prev*/*Next >* for the queue) beside the window's *Go* and *View* menus, plus the
**draw-tool bar** on the image itself (`X-N20`). The numbers are read in the panel - the
value being labelled, the queue and the corpus counts - and the queue position and the last
message are header facts, so **Full screen** (`X-N12`, `Esc`
leaves; `A`/`D`, the arrows and a double click still walk the queue) is a complete
labelling state - which is what "label at volume" needs, because the panel is where the
corpus choices are, the image is where the work is, and the header is where the actions
are. The panel therefore holds **the corpus and the choices**; the actions are in the bar.

**The label panel uses each control for what it is.** `Dataset`, `Mode` and
`Specimen` are
**dropdowns** (real choices; the specimen list grows as specimens are added). The
current **keypoint/trait is *displayed*, not chosen** - and a **Next keypoint**
button advances it, because the first switch (all heads -> the tails) is a judgement
call the auto-advance cannot make; after that each pass advances on its own. The
actions are **separate buttons**: *Save* (hidden in landmark mode, which stores
automatically), *Undo*, *Clear points* and *Clear frame*. **Undo removes the last
thing placed** - the loose point first, then the stored interval - so a mis-click or
a wrong interval is always recoverable. **Clear frame** removes every stored label of
the frame *and* the open points, so a frame from a different study can be re-done
from scratch.

A label stores `image_width`/`image_height`, because the click coordinates are only
meaningful relative to the frame they were taken on. The calibration in force, when
there is one, is stored as `profile_id` + `px_per_um` + `length_um`, so a corpus
label can also carry a physical length without the corpus *requiring* a scale.

## Export format (`datasets/annotations/<folder>/`)

An export is **one dataset** (`maintenance.export_dataset`), so its labels describe
exactly one model:

- `images/` — one copy per referenced frame, named `<stem>_<hash8>.<ext>` so names
  never collide across source folders.
- `labels.csv` — one row per keypoint label: `label_id, file, split, image_width,
  image_height, specimen_index, trait, p1_x, p1_y, p2_x, p2_y, length_px, px_per_um,
  length_um, source, labeled_by, created_at, p1_name, p2_name` (the last two name the
  landmarks with their **export names**).
- `boxes.csv` — one row per bounding box: `box_id, file, image, split, image_width,
  image_height, specimen_index, x1, y1, x2, y2, width, height, source, labeled_by,
  created_at` - `file` is the **original** frame path and `image` the exported copy,
  exactly as in `labels.csv`, and every row has as many values as the header has
  names (`SR-DATASET-024`). A box is not a pair of landmarks, so it is not folded
  into the keypoint table.
- `labels.json` — the dataset and its **schema** (key, keypoint names, export names,
  intervals, **edges** - the skeleton, `SR-DATASET-023` - and boxes), the raw
  `labels` and `boxes` arrays, and a `specimens` array:
  one record per `(image, specimen)` carrying `keypoints` **by export name** and the
  `bbox` - the shape a pose/detection training config consumes.
- `manifest.json` — dataset, schema, `stats`, `split_counts`, `splits`,
  `images_copied`, `images_missing` and the export timestamp.
- `splits/{train,val,test}.txt` — the **exported image paths** of each split, one per
  line. Every trainer wants a per-split image list and almost none agree on how to
  spell it, so the neutral form is written out and `labels.json.splits` records the
  same lists.

The split is deterministic (SHA-1 of the *image path* bucketed 70/15/15), so every
label of one frame - keypoints *and* box - lands in the same split and re-running the
export never reshuffles the corpus. The export dialog opens on
`datasets/annotations/<dataset name>_<timestamp>`, so exporting several models never
mixes them into one folder.

## The box queue: labelling the boxes that already exist

A frame holds several animals (median three in the first corpus) and each of them is
~84 px in a 1280 px frame, so "label the frame" means finding the same animal twice
and clicking at whatever zoom happens to be set. So the surface can work **one
animal at a time** instead:

1. **Open boxes export...** (*File* menu, `open_boxes` - it used to be the panel button
   `btn_open_boxes`) asks for the **two halves of a box
   corpus**: the boxes file (`boxes.csv` or `labels.json`, or the export folder
   itself) and the folder its images are in - usually the `images/` folder beside the
   file, and only ever *suggested* from the file, never guessed (`BoxesExportDialog`,
   `suggested_images_dir`). It imports every box into the open dataset with the source
   it arrived with and queues them - one item per **animal**, not per frame
   (`SR-DATASET-022`). If the open dataset's model has **no box**, the operator is
   *offered* the dataset that does and the boxes open there in one click
   (`switch_to_box_dataset`) - the dialog-free core (`open_boxes_export`) only refuses
   in words, because a modal window inside the code a test or a script drives is a
   hang.
2. Each item is shown in **one of two views**, switched with the `Crop`/`Frame` tool
   or the `C` key - and the toggle sticks across animals:
   - **Frame** (where a detection-only corpus opens): the **whole frame with every
     stored box** drawn, each in its specimen's colour and numbered, with the animal
     being labelled drawn **solid** on top (`_refresh_item_overlays`,
     `_set_active_box`). This is the *bounding-box display*: it is the only way to see
     whether a box is actually around an animal - and whether a neighbour was boxed
     twice.
   - **Crop** (where a keypoint corpus opens): the animal alone, magnified
     (`vision/crop.py`; box + context margin, integer factor, nearest-neighbour so the
     enlargement is exact).
   Both are *views of the same item*: the frame stays the coordinate space
   (`SR-DATASET-025`), so switching can never change what a label stores. A click is
   converted back to source pixels by the one `Crop` object that owns the mapping and
   its `invert` - and in frame view there is simply no crop in force, so the click
   already *is* a source pixel (`_crop_point`); a **dragged box** goes the same way back
   (`source_box`), and is stored against the **frame's** own size (`_source_size`). A
   point that lands outside the specimen's own box is **reported** and still stored: it
   is usually a click on the neighbour, but the box may just be tight.
   In the **crop of a keypoint corpus** the press *is* the keypoint gesture
   (`ImageCanvas.point_mode`, `SR-DATASET-028`): the box covers the whole crop, so
   without it every click selected the box and no landmark could be placed. A box is
   moved in *Frame* view, where every box is visible and can be judged.
3. **Yes (Y)** keeps the box as ground truth - a `source='model'` box becomes
   `source='manual'` and the note names the accept, because a prediction is not
   ground truth until a person says so (`SR-PREDICT-N01`, law 7). **No (N)** removes
   the box: a detector trained on a rectangle a person rejected would learn the wrong
   animal. Keeping the *rejection* as a hard negative needs somewhere to put it
   (`SR-DATASET-014`).
4. The **keypoints are clicked in either view** and the pair is stored by the second
   click - *click, click, next animal*. In the **crop** of a keypoint corpus that is the
   whole of the gesture: the press places the landmark (`SR-DATASET-028`), so the second
   click stores the pair (and re-stores it if there was one - correcting a keypoint is
   clicking it again). Storing keypoints on a box that is still a
   prediction does **not** move on: the box question stays open until it is answered,
   so a `source='model'` row can never be skipped silently.
5. The queue is built from the **corpus**, so a box added by hand is in it too - the
   animals the detector missed (recall 0.852 at the operating threshold, measured)
   have to be addable, or the pose corpus inherits the detector's blind spots.

The crop's magnification, margin and window are in `daphnia/vision/crop.py`
(`crop_for_box`, `DEFAULT_MARGIN = 0.25`, `DEFAULT_TARGET_SIZE = 384`,
`MAX_SCALE = 8`); `tests/test_crop.py` pins the mapping with hand-computed numbers
and checks the enlargement is exact. A tool a state cannot use is hidden from its menu
rather than shown dead, so a mode never offers a choice that cannot be made.

Both views, rendered on the real 526-box corpus: `docs/screenshot-dataset-frame-view.png`
(the whole frame, boxes 1-3, the current animal solid) and
`docs/screenshot-dataset-crop-view.png` (the same animal magnified).

## Still to build

- **Pre-labels from the *Predict* surface** (`SR-PREDICT-009`): the Dataset side is
  built - an export can be opened, reviewed Yes/No and labelled - but *Predict* has
  no "send these boxes to a dataset" button yet, so the round trip still goes through
  a file.
- **Pre-predicted keypoints.** The keypoint model is not trained; when it is, its
  predictions enter the same queue as `source='model'` **keypoint** rows and the same
  Yes/No tools answer them. The store already distinguishes `manual` from `model`, and
  `SR-DATASET-008`'s agreement report needs exactly that split - so this is a model to
  train, not a schema to invent.
- **A view/orientation axis.** The four-point vocabulary assumes a lateral view
  (`projectbrief.md`); the corpus is photographed from above, from the side and on the
  belly, and those are different measurements. The decision belongs in the pose task
  doc *before* the keypoints are labelled at volume.
- **A specimen/study split** (`SR-DATASET-010`): today the grouping key is the image
  file. Once frames are linked to specimens and studies, group by those instead.
- **Resumable sessions** (`SR-DATASET-N02`) and a **length histogram** (`SR-DATASET-011`).
- **Hard negatives** (`SR-DATASET-014`) as a first-class, non-counting queue item -
  which is also where a **rejected** box should go instead of being deleted.

## Open issues

- Target corpus size: ~500–2000 labelled specimens with morphological variety.
- The split unit (`SR-DATASET-010`) must be settled before the first training run;
  it determines whether reported accuracy is defensible.
