# SR-STUDY2 — plate capture requirements

Serves: **O1** (a standardized acquisition workflow for individuals in multi-well plates),
**O4** (the desktop application) and the shared end goal: **studies that last days to
weeks**. Launched with `python run.py study2`, and offered by the hub as *Study 2*.

This surface is the **acquisition half** of the plate routine the project is about. It is
built next to the governed Study pipeline (`SR-STUDY.md`) rather than out of it, and it is
meant to **replace** it once it covers those gates: the loop here is a *plate*, not a
seven-step pipeline. It walks the wells of one plate, captures the well under the camera -
a short **burst**, or a **recording** the operator starts and stops - and shows the plate
lighting up as the wells are imaged.

Deliberately **not** a governed run: there is no step order to walk, no selection step and
no mandatory annotation. What it keeps is the part a number cannot survive without - the
images land in the study's own folder, under a session that froze the optical scale.

## Gate policy

| Fact | Where it comes from | Why it gates |
| --- | --- | --- |
| `study` | chosen or created on this page | it decides **where the images are saved** |
| `plate` + `well` | chosen on this page | a burst is captured *of* something, and its well is how it is found again |
| `connection` | the app's shared camera service | nothing is captured off a stream that is not there |
| `calibration` | an active optical profile | an image without a scale can never be measured (`SR-CORE`) |

Not gating, on purpose: frame **selection** (the well's image is the sharpest frame,
chosen by the machine), **annotation** and **measurement** - this page captures, and says
so where it would otherwise look finished.

## Functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-STUDY2-001` | The surface **shall** offer the plate formats 6, 24 and 96 wells, and **shall** draw a plate in its own rows x columns. | The three formats are the ones the lab owns; a plate drawn in another shape is unrecognisable while the operator is looking at the real one. | `T` `tests/test_study2.py` | implemented | O1 |
| `SR-STUDY2-002` | A burst **shall** be stored as an observation **of the well** in a session of the study, and re-visiting a well **shall** append a burst rather than overwrite one. | Same rows the Acquire step writes, so nothing this page captures is second-class. | `T` `tests/test_study2.py` | implemented | O1, D1 |
| `SR-STUDY2-003` | A well's **image** **shall** be the sharpest frame ever captured on that well, and the surface **shall** name that file. | One well -> one picture, chosen by the machine, unaffected by a later, worse burst. | `T` `tests/test_study2.py` | implemented | O1 |
| `SR-STUDY2-004` | Each well **shall** be shown **lit by its state** (not captured / captured / measured), with a key that names the states and counts them. | "Which wells are still empty" is the question the page exists for, and it is read at a glance, not by reading labels. | `T` `tests/test_wells.py`, `tests/test_study2.py` | implemented | O1, O4 |
| `SR-STUDY2-005` | The plate **shall** be showable in pages of *n* wells (whole plate, 6, 12, 24, 48), a page **shall never** split a row, and the chosen well **shall** keep its selection across pages. | A 96-well plate with *n* wells per page is the working scale; half a row on screen hides where the wells came from. | `T` `tests/test_wells.py` | implemented | O1 |
| `SR-STUDY2-006` | For every well the surface **shall** display the **size of each individual in it** - from the pose model on the well's image, or from a manual line. | The stated end goal of the project, and the reason the plate page exists. | `T` | **proposed** - needs the pose model (`docs/rtmpose.md`) *and* a target per individual inside the well (see the open issues) | O2, O3 |
| `SR-STUDY2-007` | The surface **shall** offer the **next well without an image**, wrapping at the end of the plate, and **shall** say when every well has one. | The operator moves the plate by hand; after the last well the useful answer is "which one did I miss", not "the run is over". | `T` `tests/test_wells.py`, `tests/test_study2.py` | implemented | O1 |
| `SR-STUDY2-008` | The surface **shall** let a study be chosen, adopted as the current one, or created for plates, and a plate of a chosen format added to it - without leaving the page. | The setup of a plate run is not a reward for walking a seven-step wizard. | `T` `tests/test_study2.py` | implemented | O4 |
| `SR-STUDY2-009` | The surface **shall** report, per plate, how many wells have an image and how many are measured. | The plate's own progress, and the honest difference between "imaged" and "measured". | `T` `tests/test_study2.py` | implemented | O1 |
| `SR-STUDY2-010` | A captured frame **shall** open full size from the well's strip. | Judging a capture is a full-size job. | `D` | implemented | O1 |
| `SR-STUDY2-011` | The strip **shall** show which frame is the well's image. | A well accumulates bursts; the one the number will come from has to be identifiable. | `T` `tests/test_study2.py` | implemented | O1 |
| `SR-STUDY2-012` | A captured well **shall** draw **its own frame inside the well**, with the well's state kept visible as the band around it, and the drawing **shall** be switchable off. A frame that cannot be read **shall** fall back to the state colour. | "The burst appears along the way": while the plate is walked, the image has to appear **where it was taken**, so one look at the plate answers "what did I get, and where is the plate". The state may not be lost to the picture - the band keeps *what a well holds* apart from *what is in it* - and a moved file must not turn a captured well into an untaken one. | `T` `tests/test_study2.py` | implemented | O1, O4 |
| `SR-STUDY2-013` | The surface **shall** let the live stream be **recorded** on the chosen well - started and stopped by the operator, of any length the operator chooses - and **shall** store the recording as **one burst** of that well, with the sharpest frame of it as the well's image. | A well that drifts in and out of focus is not served by a fixed `DAPHNIA_BURST_SIZE` buffer: the operator can see the picture and decides when it is right. The length is theirs; the storage is the burst's - same folder, same rows, same rule for the image - so a recording is not a second kind of data. | `T` `tests/test_study2.py` | implemented | O1, D1 |
| `SR-STUDY2-014` | While a recording runs the page **shall** show its well, its frames and its seconds, and **shall** keep the plate, the study, the page size, the walk and the re-read locked: a move refused with the sentence that ends it. | A recording belongs to the well under the camera, and moving the plate mid-recording would write the rest of it into another well's folder. The counter is what makes a take judgeable while it is still running. | `T` `tests/test_study2.py` | implemented | O1, O4 |

## Non-functional

| ID | Requirement | Rationale | Verification | Status | Traces |
| --- | --- | --- | --- | --- | --- |
| `SR-STUDY2-N01` | A burst - and a recording - **shall** be refused while no optical calibration is active, and the refusal **shall** name the fix. | An image without a scale can never be measured, and this surface's frames are captured to be measured. | `T` `tests/test_study2.py` | implemented | O3 |
| `SR-STUDY2-N02` | A capture that stored nothing (no buffered frames, a folder that cannot be written) **shall** be reported as **not stored**, and a partial burst **shall** say how many frames were written. | "It said it saved" is how a plate ends up with silent holes. | `T` `tests/test_study2.py` | implemented | O4 |
| `SR-STUDY2-N03` | The page **shall** use the existing study / session / observation / media-asset model and **shall** add no second storage path. | Everything captured here has to stay readable by Study, Dataset, the Bench and the corpus. | `I` | implemented | O1, D1 |
| `SR-STUDY2-N04` | The surface **shall** open on a machine where PyTorch is not installed. | The app must not depend on the ML stack to open a window (`SR-CORE-011`, `SR-PREDICT-N02`). | `T` `tests/test_surfaces.py` (`PurityTests`) | implemented | D3 |
| `SR-STUDY2-N05` | **Every** well of the plate **shall** appear on the plate, whether or not anything was captured on it. | The empty wells are the information; a query alone can only show the wells that have rows. | `T` `tests/test_wells.py` | implemented | O1 |
| `SR-STUDY2-N06` | A recording **shall** end at the frame limit (`DAPHNIA_RECORD_MAX_FRAMES`), when the stream drops, or when its page goes off screen; it **shall** store what arrived, **shall** name **which** of those endings it was, and **shall** say how many frames reached the disk. | A recording nobody stops must not fill the drive in silence, and an ending the operator did not choose (a dropped stream, a closed window) has to be distinguishable from the one they did. "It said it saved" is how a plate ends up with holes (`SR-STUDY2-N02`). | `T` `tests/test_study2.py` | implemented | O4 |

## Open issues

- **`SR-STUDY2-006` (the per-individual size) is the reason this page exists.** It needs
  two things that are not here yet: the pose model on a well's image
  (`daphnia/vision/rtmpose/`, `docs/rtmpose.md`), and a target *per individual inside the
  well*.
- **"Individuals per well" is not modelled.** `observations.individual_index` counts
  *specimens of the counter*, not animals inside a well: a well with five daphnids holds
  one observation today. Two candidate readings, to be decided before the measuring half:
  (a) one observation per individual of a well (`individual_index` 1..N on the same well
  label), or (b) one observation per well and N measurement rows, distinguished by
  `corpus_labels.individual_index` - which is the vocabulary `vision.individuals` already
  uses for "which animal is this number about". (b) matches the Bench and needs no schema
  change.
- **Timepoints** (`SR-STUDY-008`, `SR-STUDY-009`) are not in this page: a plate walked
  again tomorrow is a new session by accident of the calendar rather than by plan. Until a
  plan exists, this page cannot say "this plate is due today" - which is the missing half
  of "studies that last days to weeks".
- **A recording's throughput is measured on one machine, not guaranteed.** One 1280x720
  frame of this rig costs **17.2 ms to write plus 2.5 ms to score**, at **1241 kB** each
  (measured 2026-10-22, M4, offscreen) - about **51 frames/s** of headroom in the main
  thread against the 30 fps the stream delivers, so the queue keeps up *here*. **Not
  measured:** the same cost on the lab's Mac, on a nearly full disk, or while a page of
  wells is drawing its frames at the same time - and what the recorder does when it does
  fall behind (frames would queue rather than be dropped, which is the right failure, but
  it is untested). The frame limit is the guard: `DAPHNIA_RECORD_MAX_FRAMES` defaults to
  300 frames ≈ 10 s ≈ 360 MB, and reaching it is reported, never silent.
- **The contact sheet is a screen, not an artefact.** Each well draws its own frame
  (`SR-STUDY2-012`), but nothing exports the plate as one image - the shape a thesis
  figure or a lab-book page wants.
- **Drawing a full plate of frames costs ~1 s, measured.** 96 real 1280x720 PNGs (the
  capture size of this rig, `recordings/STUDY_20261015_153059_1114a8`) decode to 160x90
  thumbnails in **1083 ms cold - 11.3 ms each - and 0.01 ms cached** (measured
  2026-10-21, M4, offscreen; `well_grid.well_pixmap`). So the *first* draw of a 96-well
  plate that is already full is a visible pause, and every later one is free. The cache is
  capped at 160 entries; the switch *Draw each well's frame in its well* is the escape
  hatch, and drawing more than one page at a time is what would need the number to move.
  What is **not** measured: the paint cost on screen (this number is decode only), and the
  cost for a plate of 12 MP stills rather than the stream's frames.
- **Nothing here is a publication number.** The frames this page captures are as good as
  the governed ones (same rows, same frozen scale), but it offers no selection and no
  annotation, so `SR-STUDY`'s rule stands: the Study surface is still the only one that
  produces figures - and closing that gap is what "replacing Study" means.

