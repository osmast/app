"""Running a trained RTMDet on a frame - the one thing the Predict surface calls.

The whole inference path in one object, so the Qt page holds no model logic and the
same code serves a script, a notebook and the tests:

    detector = load_detector("models/rtmdet_daphnia.pt")
    detections = detector.detect(frame)        # boxes in the *original* pixels

:class:`Detector` owns the three things that must not be re-implemented per caller:
the letterbox mapping back to original coordinates, the device, and the score/NMS
thresholds. PyTorch is imported lazily, so importing this module on a machine
without it succeeds and only *using* it fails - which is what lets the app say
"train a model first" instead of refusing to start.
"""

from dataclasses import dataclass
from pathlib import Path

from daphnia import paths

#: The name the Predict surface looks for, inside the git-ignored ``models/`` tree.
DEFAULT_CHECKPOINT = "rtmdet_daphnia.pt"

#: Sensible defaults for a live view: a detector that shows a marginal guess is
#: worse than one that shows nothing, because the operator cannot tell them apart.
DEFAULT_SCORE_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.60

#: Environment override, so a run can be pointed at another checkpoint without
#: editing anything.
CHECKPOINT_ENV = "DAPHNIA_RTMDET"


class ModelUnavailable(RuntimeError):
    """No usable checkpoint (or no PyTorch) - the caller must say so, not guess."""


@dataclass(frozen=True)
class Detection:
    """One predicted animal: where, how sure, and what it is."""

    box: tuple            # (x1, y1, x2, y2) in **original image** pixels
    score: float
    label: int
    name: str = ""

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.box[3] - self.box[1]

    def centre(self) -> tuple:
        """``(x, y)`` of the box centre, in original image pixels."""
        return ((self.box[0] + self.box[2]) / 2.0,
                (self.box[1] + self.box[3]) / 2.0)

    def describe(self) -> str:
        """One line for a list widget."""
        return (f"{self.name or self.label}  {self.score:.2f}   "
                f"{self.width:.0f} x {self.height:.0f} px  "
                f"@ {self.box[0]:.0f},{self.box[1]:.0f}")


def default_checkpoint() -> Path:
    """Where the Predict surface expects the trained detector to be."""
    return paths.MODELS_DIR / DEFAULT_CHECKPOINT


def checkpoint_path(path=None) -> Path:
    """The checkpoint to use: the given one, the environment override, or default."""
    import os

    if path:
        candidate = Path(path).expanduser()
    else:
        candidate = Path(os.environ.get(CHECKPOINT_ENV) or default_checkpoint())
    return candidate if candidate.is_absolute() else paths.PROJECT_ROOT / candidate


class Detector:
    """A loaded RTMDet, with the letterbox mapping and the thresholds applied.

    The thresholds live here rather than in the model so they can be changed *while
    looking at a frame*: tuning "how sure must it be" against a live picture is how
    an operator finds the value that is right for their illumination.
    """

    def __init__(self, model, payload, device, score_threshold=None,
                 iou_threshold=None):
        self.model = model
        self.payload = dict(payload or {})
        self.device = device
        self.input_size = int(self.payload.get("input_size") or 640)
        self.classes = tuple(self.payload.get("classes") or ("daphnia",))
        self.score_threshold = float(DEFAULT_SCORE_THRESHOLD
                                     if score_threshold is None
                                     else score_threshold)
        self.iou_threshold = float(DEFAULT_IOU_THRESHOLD
                                   if iou_threshold is None else iou_threshold)

    # -- facts for the UI --------------------------------------------------
    @property
    def preset(self) -> str:
        """The model size this checkpoint was trained at."""
        return str(self.payload.get("preset") or self.model.preset)

    @property
    def trained_at(self) -> str:
        """When the checkpoint was written."""
        return str(self.payload.get("created_at") or "unknown")

    @property
    def metrics(self) -> dict:
        """The validation numbers recorded at the best epoch."""
        return dict(self.payload.get("metrics") or {})

    def describe(self) -> str:
        """A short line for a status label."""
        metrics = self.metrics
        score = (f"  val AP50 {metrics['ap50']:.3f}"
                 if metrics.get("ap50") is not None else "")
        return (f"rtmdet-{self.preset}   classes: {', '.join(self.classes)}   "
                f"trained {self.trained_at}{score}")

    def name_of(self, label) -> str:
        """The class name of a label index, falling back to the index itself."""
        index = int(label)
        return (self.classes[index] if 0 <= index < len(self.classes)
                else str(index))

    # -- inference ---------------------------------------------------------
    def detect(self, frame) -> list:
        """Predict the animals in one BGR frame; returns :class:`Detection` values.

        The frame is letterboxed **to the size the checkpoint was trained at**
        (``input_size``), inferred, and the boxes are **mapped back** to the original
        pixels before anything is drawn or measured
        (:class:`~daphnia.vision.rtmdet.data.Letterbox` owns that mapping, so a caller
        cannot forget a step of it).

        Feeding the network a *native* frame instead would change the scale
        statistics it learned - an animal 640 pixels tall at training time and 1280
        at inference time - and it would cost about 2.4x the compute for the
        privilege. The checkpoint records its input size for exactly this reason.
        """
        from daphnia.vision.rtmdet.data import letterbox, to_tensor

        if frame is None or getattr(frame, "ndim", 0) < 2:
            return []
        padded, mapping = letterbox(frame, self.input_size)
        tensor = to_tensor(padded).unsqueeze(0).to(self.device)
        boxes, scores, labels = self.model.detect(
            tensor, score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold)[0]
        mapped = mapping.invert(boxes)
        return [Detection(box=(float(box[0]), float(box[1]),
                               float(box[2]), float(box[3])),
                          score=float(score), label=int(label),
                          name=self.name_of(int(label)))
                for box, score, label in zip(mapped, scores, labels)]


def load_detector(path=None, device="auto", score_threshold=None,
                  iou_threshold=None) -> Detector:
    """Load a checkpoint into a :class:`Detector`.

    Raises:
        ModelUnavailable: PyTorch is missing, or there is no checkpoint at ``path``.
            The message says which, and what to run - a live view that silently
            showed nothing would be blamed on the model.
    """
    target = checkpoint_path(path)
    if not target.exists():
        raise ModelUnavailable(
            f"No trained detector at {target}.\n"
            "Build a 'Bounding boxes only' dataset, export it, then run "
            "`make train` (or `python scripts/train_rtmdet.py`).")
    try:
        from daphnia.vision.rtmdet.train import load_checkpoint, pick_device
    except ImportError as error:                     # pragma: no cover - env issue
        raise ModelUnavailable(
            "PyTorch is not installed in this environment, so the detector cannot "
            "run. Install the training environment - see docs/rtmdet.md:\n"
            "    conda env update -f environment.yml") from error
    try:
        model, payload = load_checkpoint(target, device=device)
    except (OSError, ValueError, RuntimeError) as error:
        # A named failure, never a silent empty view (SR-PREDICT-N03).
        raise ModelUnavailable(
            f"{target} is not a usable DaphniaScope RTMDet checkpoint.\n"
            f"{error}") from error
    return Detector(model, payload, pick_device(device),
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold)


#: The lazily created detector of :func:`detect_frame`, so a loop over frames does
#: not reload several hundred megabytes per frame.
_CACHED = None


def cached_detector(path=None):
    """The process-wide detector, loaded once (or reloaded when ``path`` is given)."""
    global _CACHED
    if _CACHED is None or path is not None:
        _CACHED = load_detector(path)
    return _CACHED


def detect_frame(frame, detector=None, path=None) -> list:
    """One call for "what is in this frame", loading the detector on first use."""
    return (detector or cached_detector(path)).detect(frame)
