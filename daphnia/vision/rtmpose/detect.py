"""Running the trained pose model on a frame - the top-down cascade in one object.

The whole inference path so a script, a notebook and (later) a surface share it:

    pose = load_pose_model("models/rtmpose_daphnia.pt")     # with the detector too
    for animal in pose.detect_frame(frame):
        print(animal.points)                  # {head: (x, y), tail: (x, y)} in px

Each detector box becomes a crop, the crop becomes two keypoints, and the keypoints
are mapped **back to source pixels** through the same
:class:`~daphnia.vision.rtmpose.data.PoseInput` the training pipeline used - so a
prediction is never in an ambiguous frame (``mlContext.md`` law 1). PyTorch and the
detector are imported lazily, so importing this module succeeds on a machine without
the ML stack and only *using* it fails - which is what lets the app say "train a
model first" instead of refusing to start.
"""

from dataclasses import dataclass
from pathlib import Path

from daphnia import paths
from daphnia.vision.crop import DEFAULT_MARGIN
from daphnia.vision.rtmpose import DEFAULT_INPUT_SIZE

#: The name a caller expects inside the git-ignored ``models/`` tree.
DEFAULT_CHECKPOINT = "rtmpose_daphnia.pt"

#: Environment override, so a run can be pointed at another checkpoint.
CHECKPOINT_ENV = "DAPHNIA_RTMPOSE"

#: How sure a detector box must be before a crop is worth running the pose model on.
DEFAULT_SCORE_THRESHOLD = 0.35


class ModelUnavailable(RuntimeError):
    """No usable pose checkpoint (or no PyTorch) - the caller must say so, not guess."""


@dataclass(frozen=True)
class PosePrediction:
    """One animal's pose: the box it was found in, and the keypoints inside it."""

    box: tuple                 # (x1, y1, x2, y2) in source pixels
    score: float               # the detector's score for this box
    points: dict               # {name: (x, y)} in source pixels
    names: tuple               # the keypoint order

    def length_px(self, first=None, second=None) -> float:
        """Euclidean distance between two named keypoints, in source pixels.

        Defaults to the first two names (``head``, ``tail``), which is the body
        length this project measures.
        """
        first = first or (self.names[0] if self.names else None)
        second = second or (self.names[1] if len(self.names) > 1 else None)
        if first not in self.points or second not in self.points:
            return 0.0
        (x1, y1), (x2, y2) = self.points[first], self.points[second]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def describe(self) -> str:
        """One line for a list widget."""
        pairs = ", ".join(f"{name} {x:.0f},{y:.0f}" for name, (x, y) in self.points.items())
        return f"{self.score:.2f}  length {self.length_px():.1f} px   {pairs}"


def default_checkpoint() -> Path:
    """Where a caller expects the trained pose model to be."""
    return paths.MODELS_DIR / DEFAULT_CHECKPOINT


def checkpoint_path(path=None) -> Path:
    """The checkpoint to use: the given one, the environment override, or default."""
    import os

    if path:
        candidate = Path(path).expanduser()
    else:
        candidate = Path(os.environ.get(CHECKPOINT_ENV) or default_checkpoint())
    return candidate if candidate.is_absolute() else paths.PROJECT_ROOT / candidate


class PoseModel:
    """A loaded RTMPose, with the box→input mapping and the detector thresholds.

    The thresholds live here rather than in the model so they can be changed while
    looking at a frame. The input size and split ratio come from the checkpoint, so a
    frame is never fed at a size the model did not train at (``mlContext.md`` law 3).
    """

    def __init__(self, model, payload, device, score_threshold=None):
        self.model = model
        self.payload = payload
        self.device = device
        self.input_size = int(payload.get("input_size", DEFAULT_INPUT_SIZE))
        self.split_ratio = float(payload.get("split_ratio", 2.0))
        self.names = tuple(payload.get("keypoints") or model.keypoint_names)
        # The **crop margin** is a hyper-parameter of the *mapping*, not a taste: it decides
        # how far the window reaches beyond the box, and therefore where a keypoint lands
        # once it is mapped back to source pixels (`rtmpose.data.pose_input_for_box`). It is
        # read from the checkpoint the way the input size is (`mlContext.md` law 3) - a model
        # trained with ``--margin 0.5`` served with the 0.25 default would place every
        # keypoint systematically wrong, and nothing would raise. Old checkpoints carry it
        # under ``config``; one that does not is served the module default.
        self.margin = float((payload.get("config") or {}).get("margin", DEFAULT_MARGIN))
        self.score_threshold = (DEFAULT_SCORE_THRESHOLD
                                if score_threshold is None else score_threshold)

    def describe(self) -> str:
        """One line for a status label: the model, what it predicts, how it scored.

        The metrics come from the checkpoint's own report, so the surface shows the
        number *that* checkpoint earned rather than a claim about pose models in
        general (``mlContext.md`` §8 - provenance travels with the number).
        """
        metrics = self.payload.get("metrics") or {}
        text = (f"rtmpose-{self.payload.get('preset', 'tiny')}  {self.input_size} px  "
                f"{', '.join(self.names) or 'keypoints'}")
        if metrics.get("median_error_px") is not None:
            text += f"   median {metrics['median_error_px']:.1f} px"
        if metrics.get("length_median_error_um") is not None:
            text += f", length {metrics['length_median_error_um']:.0f} um"
        return text

    def keypoints(self, frame, box):
        """``(K, 2)`` keypoints in **source pixels** for one box, or ``None``.

        ``None`` means the box does not overlap its frame, so there is nothing to
        crop - a named outcome the caller can skip, never a silent zero.
        """
        from daphnia.vision.rtmpose.data import (
            pose_input_for_box,
            to_tensor,
            window_image,
        )

        if frame is None or getattr(frame, "ndim", 0) < 2:
            return None
        height, width = frame.shape[:2]
        pose = pose_input_for_box(box, (width, height), self.input_size,
                                  margin=self.margin)
        window = window_image(frame, pose)
        if window is None:
            return None
        import torch

        tensor = to_tensor(window).unsqueeze(0).to(self.device)
        predicted = self.model.predict(tensor).detach().cpu().numpy()[0]
        return pose.invert_points(predicted)

    def detect_box(self, frame, box, score=1.0) -> PosePrediction:
        """The pose of one known box (a labelled or detected animal)."""
        points = self.keypoints(frame, box)
        mapping = ({} if points is None else
                   {name: (float(x), float(y))
                    for name, (x, y) in zip(self.names, points)})
        return PosePrediction(box=tuple(float(v) for v in box), score=float(score),
                              points=mapping, names=self.names)

    def detect_frame(self, frame, detections=None, detector=None,
                     score_threshold=None) -> list:
        """Every animal in a frame: the detector's boxes, each with its keypoints.

        ``detections`` lets a caller supply its own boxes (a labelled corpus, a
        tracking result); otherwise the trained RTMDet runs first - which *is* the
        top-down cascade, and the only reason this is two models and not one.
        """
        if detections is None:
            from daphnia.vision.rtmdet.detect import cached_detector

            detections = (detector or cached_detector()).detect(frame)
        threshold = (self.score_threshold if score_threshold is None
                     else float(score_threshold))
        results = []
        for detection in detections:
            if float(getattr(detection, "score", 1.0)) < threshold:
                continue
            prediction = self.detect_box(frame, getattr(detection, "box", detection),
                                         score=getattr(detection, "score", 1.0))
            if prediction.points:
                results.append(prediction)
        return results


def load_pose_model(path=None, device="auto", score_threshold=None) -> PoseModel:
    """Load a checkpoint into a :class:`PoseModel`.

    Raises:
        ModelUnavailable: PyTorch is missing, or there is no checkpoint at ``path``.
            The message says which, and what to run.
    """
    target = checkpoint_path(path)
    if not target.exists():
        raise ModelUnavailable(
            f"No trained pose model at {target}.\n"
            "Export a keypoint corpus, then run `make train-pose` "
            "(or `python scripts/train_rtmpose.py`).")
    try:
        from daphnia.vision.rtmpose.train import load_checkpoint, pick_device
    except ImportError as error:                     # pragma: no cover - env issue
        raise ModelUnavailable(
            "PyTorch is not installed in this environment, so the pose model cannot "
            "run. Install the training environment - see docs/rtmpose.md:\n"
            "    conda env update -f environment.yml") from error
    try:
        model, payload = load_checkpoint(target, device=device)
    except (OSError, ValueError, RuntimeError) as error:
        raise ModelUnavailable(
            f"{target} is not a usable DaphniaScope RTMPose checkpoint.\n"
            f"{error}") from error
    return PoseModel(model, payload, pick_device(device),
                     score_threshold=score_threshold)


#: The lazily created model of :func:`pose_keypoints`, so a loop over frames does not
#: reload the weights every frame.
_CACHED = None


def cached_pose_model(path=None) -> PoseModel:
    """The process-wide pose model, loaded once (or reloaded when ``path`` is given)."""
    global _CACHED
    if _CACHED is None or path is not None:
        _CACHED = load_pose_model(path)
    return _CACHED


def pose_keypoints(frame, detector=None, pose_model=None, path=None) -> list:
    """One call for "put keypoints on every animal in this frame"."""
    model = pose_model or cached_pose_model(path)
    return model.detect_frame(frame, detector=detector)
