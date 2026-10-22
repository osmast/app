"""The models in force, run on the frame a surface is showing (`SR-PREDICT-018`).

**One rule.** Which checkpoint is selected is the *Detector (RTMDet)* / *Pose (RTMPose)*
menus' business (the registry, `daphnia/models`). When one is in force and a surface shows a
frame, that model's prediction is shown **on the current frame** - and on a surface where the
operator labels or measures, the prediction also lands in the *editable* draft, so correcting
it is the same gesture as placing it: drag the box, move the two points, Save.

The pipeline is the **top-down** one (`mlContext.md` §10, `docs/rtmpose.md`): RTMDet
localises every animal, RTMPose places head and tail *inside each box*, and the length
follows from the two points.

Drawn in the canvas' own **prediction layer** - one colour, dotted - and **never stored**: a
prediction is not ground truth (`SR-PREDICT-N01`, `mlContext.md` law 7). Only the operator's
*Save* writes a label.

Two deliberate limits:

* **one pass per frame shown**, in the GUI thread: these surfaces show a *still* frame (a
  file, a queued item, a frozen live frame) and one inference is ~60-100 ms on MPS. The
  streaming case is the Predict surface's own worker (`SR-PREDICT-004`);
* **no silent fallback** (law 8): with no checkpoint in force the runner names the menu that
  loads one, and a checkpoint that refuses to load names the file.
"""

import time
from dataclasses import dataclass

from PySide6.QtCore import QObject

from daphnia import models
from daphnia.db import connection
from daphnia.ui.model_menu import loader_for


@dataclass(frozen=True)
class Prediction:
    """One animal as the models see it: a box, its keypoints, the score that found it.

    ``points`` are in **source pixels** of the frame that was passed in, in the model's own
    order (``head``, ``tail`` for the pose corpus). ``label`` is what a surface may write
    above the animal - the head-to-tail length, in the unit the scale in force allows.
    """

    box: tuple
    points: tuple = ()
    score: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class Result:
    """What one pass produced: the animals, what it cost, and the line to show."""

    predictions: tuple = ()
    milliseconds: float = 0.0
    note: str = ""

    def __len__(self):
        return len(self.predictions)


def primary(predictions):
    """The prediction the operator is most likely working on: the **largest** animal.

    A frame holds several animals (median three in the first corpus) and the top-down
    pipeline returns one box each. The largest is the one filling the frame, which is what a
    bench photo and a labelling click are usually about - and the *others* stay drawn in the
    prediction layer, so the choice hides nothing: it only decides which one is editable.
    """
    predictions = list(predictions)
    if not predictions:
        return None
    return max(predictions, key=lambda item: _area(item.box))


def _area(box) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class ModelRunner(QObject):
    """The top-down pipeline, applied to one frame, with the models the registry names."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.detector = None
        self.pose_model = None
        self._problems = []          # named reasons a model is not running

    # -- which models ------------------------------------------------------
    def adopt_models(self, detector=None, pose=None) -> bool:
        """Take a model a window loaded from its menus (`SurfaceWindow.adopt_model`)."""
        if detector is None and pose is None:
            return False
        if detector is not None:
            self.detector = detector
        if pose is not None:
            self.pose_model = pose
        self._problems = []
        return True

    def load(self) -> bool:
        """Load the checkpoints the registry says are in force - once, lazily.

        The same rule as every other consumer: the registry decides *which* file
        (`daphnia.models.active`), and a page never picks one of its own.
        """
        self._problems = []              # a fresh account, not a growing one
        for kind, name in (("detector", "detector"), ("pose", "pose_model")):
            if getattr(self, name) is not None:
                continue
            with connection() as conn:
                active = models.active(conn, kind)
            if active is None:
                self._problems.append(
                    f"no active {models.KIND_LABELS[kind].lower()} - load one in the "
                    f"{models.KIND_MODELS[kind]} menu")
                continue
            loader, unavailable = loader_for(kind)
            try:
                model = loader(active.path)
            except unavailable as error:            # named, never silent (law 8)
                self._problems.append(str(error))
                continue
            except Exception as error:              # noqa: BLE001 - reported, not swallowed
                self._problems.append(f"{type(error).__name__}: {error}")
                continue
            setattr(self, name, model)
        return self.detector is not None

    @property
    def problems(self) -> tuple:
        """Why nothing ran, in the model's own words (empty when everything loaded)."""
        return tuple(self._problems)

    def describe(self) -> str:
        """Which networks are in force, with the name of the checkpoint each came from."""
        parts = []
        if self.detector is not None:
            parts.append("RTMDet " + str(self.detector.describe()).split("  ")[0])
        if self.pose_model is not None:
            parts.append("RTMPose")
        return " + ".join(parts) if parts else "no model in force"

    # -- the pass ----------------------------------------------------------
    def run(self, frame, um_per_px=None) -> Result:
        """Detect every animal, place head and tail inside each box, and time it.

        ``um_per_px`` is the scale in force, when there is one: with it each animal's label
        is its head-to-tail length in micrometres, without it the number stays in pixels and
        says *unscaled* - the two units are never mixed (`SR-PREDICT-013`).
        """
        if frame is None:
            return Result(note="No frame on screen to run the model on.")
        if self.detector is None and not self.load():
            reason = "; ".join(self._problems) or "not loaded"
            return Result(note=f"No model in force ({reason}).")
        started = time.perf_counter()
        try:
            detections = self.detector.detect(frame)
            poses = (self.pose_model.detect_frame(frame, detections=detections)
                     if self.pose_model is not None else [])
        except Exception as error:                  # noqa: BLE001 - a failure is a message
            return Result(note=f"The model failed: {type(error).__name__}: {error}")
        elapsed = (time.perf_counter() - started) * 1000.0

        predictions = tuple(
            self._prediction(detection, self._pose_for(detection, poses), um_per_px)
            for detection in detections)
        with_points = sum(1 for item in predictions if len(item.points) >= 2)
        note = (f"{self.describe()}: {len(predictions)} animal(s)"
                + (f", {with_points} with keypoints" if with_points else "")
                + f", {elapsed:.0f} ms")
        return Result(predictions, elapsed, note)

    def _pose_for(self, detection, poses):
        """The pose of one detection, matched by its **box** - never by list position.

        The two lists are not parallel: the pose pass skips a box whose crop misses the
        frame, so pairing by position would put one animal's keypoints on another's box.
        """
        box = tuple(float(value) for value in detection.box)
        for pose in poses:
            if tuple(float(value) for value in pose.box) == box:
                return pose
        return None

    @staticmethod
    def _prediction(detection, pose, um_per_px) -> Prediction:
        """One detection, with its keypoints and the length its first two points imply."""
        box = tuple(float(value) for value in detection.box)
        points = _pose_points(pose)
        label = ""
        if len(points) >= 2 and um_per_px:
            length = ((points[1][0] - points[0][0]) ** 2
                      + (points[1][1] - points[0][1]) ** 2) ** 0.5
            label = f"{length * float(um_per_px):.0f} um"
        return Prediction(box=box, points=points,
                          score=float(detection.score), label=label)


def _pose_points(pose) -> tuple:
    """A pose's keypoints as ``((x, y), ...)``, in the model's own order.

    ``rtmpose`` reports them as a dict keyed by name (``{"head": (x, y), "tail": (x, y)}``)
    with the order in ``pose.names`` - head first, which is the pair the length is measured
    between (`docs/rtmpose.md`). A plain sequence of pairs is accepted too, so this does not
    depend on which generation of the reader produced the pose.
    """
    if pose is None:
        return ()
    table = getattr(pose, "points", ()) or ()
    if isinstance(table, dict):
        names = getattr(pose, "names", ()) or tuple(table)
        return tuple((float(table[name][0]), float(table[name][1]))
                     for name in names if name in table)
    return tuple((float(x), float(y)) for x, y in table)
