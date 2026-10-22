# SR-PREDICT — real-time detection requirements

Serves: **O2** (a CV pipeline that identifies the animal), **O3** (evaluating
robustness and inference latency), **D2** (the trained weights, actually used).
Launched with `python run.py predict`.

Audience: the thesis. This surface is where "the detector exists" becomes "the
detector works" — and it is the Localize step that every measured number in this
project will eventually rest on.

## Gate policy

**A trained detector** (`models/rtmdet_daphnia.pt`) is the one thing that must
hold. It is not a study gate: no project, no plate, no connection and no
calibration is required, exactly as in Bench and Dataset. Without a checkpoint the
surface says so and names the command that produces one — it never opens onto an
empty view that looks broken.

## Functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-PREDICT-001` | The surface **shall** run the trained detector on a **live stream** and on a **video file**, chosen at any moment. | Both cases exist in the lab: a phone on the eyepiece for a look, and a recorded clip for a figure or a repeat. | `T` | implemented | O1, O2 |
| `SR-PREDICT-002` | The surface **shall** draw every detection as a box **with its score**, in the original image's pixels. | A box without its confidence invites the operator to treat a 0.4 guess as a fact. | `T` `tests/test_predict.py` | implemented | O3 |
| `SR-PREDICT-003` | The surface **shall** map predictions back through the letterbox before drawing, and **shall not** draw in network coordinates. | A detection drawn 280 px off is worse than no detection: it looks like a wrong animal. | `T` `tests/test_rtmdet.py` | implemented | O2, O3 |
| `SR-PREDICT-004` | The score and NMS thresholds **shall** be adjustable **while looking at a frame**. | "How sure must it be" depends on the illumination of the day, and can only be set by watching. | `D` | implemented | O3 |
| `SR-PREDICT-005` | Inference **shall** run off the GUI thread, and frame handling **shall not** queue up without bound. | A detector slower than the frame rate must drop frames, not freeze the window or grow a backlog. | `T` | implemented (worker thread, latest-frame-only) | O4 |
| `SR-PREDICT-006` | The surface **shall** report per-frame **latency and detection count**. | O3 asks for inference latency; a view that only shows boxes cannot answer it. | `T` | implemented | O3 |
| `SR-PREDICT-007` | The surface **shall** offer a **single frame** from disk, so one photo can be checked without a video. | The fastest way to answer "does it see this one?" is one file, not a clip. | `T` | implemented | O2 |
| `SR-PREDICT-008` | The surface **shall** state which checkpoint is in force, and what it scored. | A result is only interpretable next to the model that produced it. | `T` | implemented | O3, X-N09 |
| `SR-PREDICT-009` | The surface **shall** offer predictions as **pre-labels to the Dataset surface** (`source = 'model'`). | Human-in-the-loop labelling is how a first model stops being a bottleneck. | `T` | proposed - the *receiving* half is built: the Dataset surface opens a boxes export, reviews each box Yes/No and labels keypoints in its crop (`SR-DATASET-022`); a direct "send to dataset" button here is the remaining step, so today the round trip goes through a file | O2, D1 |
| `SR-PREDICT-010` | The surface **shall** step a **video file** one frame **forwards** and one frame **backwards**, from the *Frames* menu and from the `Left`/`Right` arrow keys, and **shall** show which frame is on screen. | The frame that matters is usually the one *before* the one you are looking at - a clip must be readable frame by frame, not only played. Reading it on the image is also what makes the full-screen view work for inspection. | `T` `tests/test_predict.py` (a real mp4 through the real worker, plus the page's own contract) | implemented | O2, O3, X-N13 |
| `SR-PREDICT-011` | Frame stepping **shall** be offered **only** while a video file is the source: the tools appear with the file and leave with it, and the arrow keys keep panning the canvas otherwise. | A live feed has no next frame and a single image has no neighbour, so the controls would be dead ends - and `Left`/`Right` already mean *pan* on every other canvas (`X-N10`). | `T` `tests/test_predict.py` | implemented | O4, X-N13, X-N12 |
| `SR-PREDICT-012` | The surface **shall** run the **pose model** on each detected box and draw `head` and `tail` with the head-to-tail line, and **shall** offer a toggle that turns that pass off and on. | The endpoint is a head-to-tail distance, so the surface must show the *whole* cascade - and the pose pass costs real time, so watching the detector alone has to stay possible. | `T` `tests/test_predict.py` | implemented | O2, O3 |
| `SR-PREDICT-013` | The surface **shall** report each animal's length in **micrometres** when a calibration is in force - naming the calibration, its µm/px and its own uncertainty, and flagging a **weak** or **stale** one - and in **pixels labelled unscaled** when none is, and **shall not** present µm without a calibration. | A micrometre without a scale is a number without its provenance (`projectbrief.md`, non-negotiable 1); and a scale that is stale or weak still yields a number, so it has to say so. | `T` `tests/test_predict.py` | implemented | O3, X-N08 |
| `SR-PREDICT-014` | The pose model **shall** be loaded together with the detector from the one button, its **absence** **shall** leave the detector fully usable, and the panel **shall** name how to build one. | Boxes are the prerequisite and the surface must always answer "does it see the animal?"; a half-loaded pipeline that only looks empty is worse than one that says why. | `T` `tests/test_predict.py` | implemented | O2, D2 |
| `SR-PREDICT-015` | The measured length **shall** be drawn **above the animal it belongs to**, on the frame, in the live feed and in a video file alike. | On a frame holding several animals, a number in a list does not say which animal it measures - the association has to be on the picture. | `T` `tests/test_predict.py` | implemented | O2, O4 |
| `SR-PREDICT-016` | Changing the scale **shall** re-label the frame on screen **without re-running** the models. | The scale converts a pixel length already known, and "try the other calibration and look again" has to cost a second in the lab, not a re-run. | `T` `tests/test_predict.py` | implemented | O3, O4 |
| `SR-PREDICT-017` | Loading a model from the *Detector (RTMDet)* / *Pose (RTMPose)* menus **shall** be **applied to what is on screen** - a pass that is running is restarted with it on the same source, a still frame is detected again - and **shall** be **visible at once**: the header names the model that answered, the panel's *What it found* says what *that* model found (including *none above the threshold*), and a window whose page cannot use the model says so in the status bar. | The entries used to leave the window unchanged: with nothing playing, nothing on screen changed at all, and with a pass running the boxes went on coming from the **previous** checkpoint while the bar named the new one - a claim the picture did not support. The study shell was worse: `loaded` was not connected there, so the load *was* silent. | `T` `tests/test_predict.py`, `tests/test_models.py` | implemented | O3, X-N09, X-N13 |
| `SR-PREDICT-018` | A model that is **selected** (the registry, written by the two model menus) **shall** answer **on the frame each surface is showing** when the page's *Model* switch is on - **off by default**, because a model is a tool the operator reaches for - with the detector's boxes and the pose model's head/tail points drawn in the canvas' own *prediction* layer (dotted, one colour, never a stored label's colour). A predicted box **shall** be selectable (clicking it makes that animal the one in the draft), and a prediction **shall never** be stored by itself. **One box per animal:** a prediction that a **stored label already covers** (IoU >= 0.5 with it) **shall not** be drawn, and a prediction **being edited** - and one that a drag turned into an annotation - **shall not** be drawn at its old position either (the "ghost that lags behind"). | "Wherever we are we get the box or the keypoints": before this, RTMDet and RTMPose could only be seen on Predict, so every other surface showed a frame and no model answer - and a model that is selected but invisible is a model nobody can check. Off by default keeps the operator the author: the model is an offer, not an uninvited hand in the draft. The two paint rules come from the report that followed: "I only ever want one box and no model box and real box", and a dotted copy left behind by a drag reads as "the new position was not updated". | `T` `tests/test_model_pass.py`, `tests/test_image_canvas.py::BoxEditTests`, `tests/test_image_canvas.py::EditAllTests` | implemented | O2, O3, X-N13 |
| `SR-PREDICT-019` | The two model menus **shall** list the checkpoints **available in the default models directory** (a file belongs to a kind by its name, `rtmdet*`/`rtmpose*`), **shall** carry a **check mark on the one in force** and on the entry this window has loaded, and **shall** load *and* register a listed file in one click. Loading a model from the menu **shall** switch the page's *Model* tool on. | "Show me the models in the models folder, and mark the active one": a checkpoint the lab has just trained had to be looked up in a file dialog, and because the load left the page's switch where it was, the menu looked broken - the model was in fact loaded and no prediction appeared ("it is always off"). A file named after neither network is deliberately *not* offered: guessing from its bytes would load PyTorch to draw a menu (`SR-CORE-011`). | `T` `tests/test_ui_smoke.py::HeaderBarTests`, `tests/test_models.py::AvailableCheckpointTests` | implemented | O3, X-N09, X-N13 |

## Non-functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-PREDICT-N01` | The surface **shall** never store a prediction as a measurement or a label. | A prediction is not ground truth; only a human accept may make it one (`SR-DATASET-007`). | `I` + `T` | implemented | D1, O3 |
| `SR-PREDICT-N02` | The surface **shall** start and show "no detector yet" on a machine where **PyTorch is not installed**. | The app must not depend on the ML stack to open a window (`SR-CORE-011`). | `T` `tests/test_predict.py` | implemented | D3 |
| `SR-PREDICT-N03` | A missing, unreadable or foreign checkpoint **shall** produce a named error, never a silent empty view. | "It shows nothing" is unactionable; "not a DaphniaScope checkpoint" is. | `T` `tests/test_rtmdet.py` | implemented | O4 |
| `SR-PREDICT-N04` | The reported latency **shall** be measured, not estimated. | An estimated latency is a claim about hardware that cannot be checked. | `T` | implemented (per-frame wall clock) | O3 |

## What "RTMDet" means here

The detector is RTMDet's architecture — CSPNeXt backbone, CSPNeXtPAFPN neck, a
head whose convolutions are **shared across levels with a separate BatchNorm per
level**, one anchor per location, four predicted distances, and **SimOTA** label
assignment with a **quality focal** classification loss — re-implemented in plain
PyTorch so it trains on the machine that takes the pictures (see
`daphnia/vision/rtmdet/`). It is not an `mmdet` port, and
`docs/rtmdet.md` lists where it differs.

The keypoint model (**RTMPose**) **is** here now (`SR-PREDICT-012..014`): a CSPNeXt +
**SimCC** model (`daphnia/vision/rtmpose/`, task doc `docs/rtmpose.md`) runs on each
detected box and draws `head`/`tail` with the head-to-tail line. It is the **second
stage of the top-down pipeline** - RTMDet finds the animal, the pose model places the
points inside its box - and it is trained from the corpus's own keypoints
(`datasets/kp`), which today carries **head and tail only**, so the model is
**2-point**. A four-point (width) model is another corpus and another checkpoint, not
another code path.
