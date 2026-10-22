"""A pose sample in, a normalised tensor out - the top-down crop pipeline.

The unit of work is **one detector box**: the frame is cropped to a square window
around the box (the box grown by a margin, made square on its longest side),
resized to the network's input size, and the keypoints are carried along. The
mapping between source-image pixels and the network's input pixels is owned by one
object, :class:`PoseInput`, with its own inverse - exactly as
:class:`~daphnia.vision.rtmdet.data.Letterbox` and :class:`~daphnia.vision.crop.Crop`
own theirs. That single-owner rule is ``mlContext.md`` **law 1**, and it is the bug
that once made a metric read 0.000 for a model that was learning.

Augmentation is the two that matter for a 384-sample corpus with one organism:
a **horizontal flip** (a daphnid carries no left/right semantics in a head-to-tail
model, so a mirror is free) and a cheap **colour jitter** (the illumination drifts
between sessions). Both are seeded per ``(seed, epoch, index)``, so a run is
reproducible and still varied.
"""

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset

from daphnia.vision.crop import DEFAULT_MARGIN
from daphnia.vision.rtmdet.data import colour_jitter, read_frame, to_tensor
from daphnia.vision.rtmpose import DEFAULT_INPUT_SIZE


class PoseDataError(ValueError):
    """A box cannot become a pose crop (degenerate, or outside its image)."""


def _normalised(box) -> tuple:
    """``(x1, y1, x2, y2)`` with the corners sorted, so a drag direction never matters."""
    x1, y1, x2, y2 = (float(value) for value in box)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


@dataclass(frozen=True)
class PoseInput:
    """How a square window of a frame became the network's input - and back.

    ``origin`` and ``side`` are **integers in source-image pixels** (the frame every
    label is stored in), and ``size`` is the square the window is resized to. The
    scale is ``size / side``. Both directions of the mapping live here and nowhere
    else::

        >>> pose = PoseInput(origin=(100, 50), side=200, size=100)
        >>> pose.to_input((150, 100))
        (25.0, 25.0)
        >>> pose.invert((25.0, 25.0))
        (150.0, 100.0)

    The integers are enforced in ``__post_init__`` because :func:`window_image`
    slices with them; a fraction would make the picture and the mapping disagree.
    """

    origin: tuple          # (x, y) of the window's top-left, source pixels
    side: int              # the window's side length, source pixels (square)
    size: int              # the network input the window is resized to
    source_width: int = 0
    source_height: int = 0

    def __post_init__(self):
        object.__setattr__(self, "origin",
                           (int(round(self.origin[0])), int(round(self.origin[1]))))
        object.__setattr__(self, "side", max(1, int(round(self.side))))
        object.__setattr__(self, "size", max(1, int(self.size)))

    @property
    def scale(self) -> float:
        """Source pixels -> input pixels."""
        return float(self.size) / float(self.side)

    def to_input(self, point) -> tuple:
        """A **source**-pixel point as an **input**-pixel point."""
        return ((float(point[0]) - self.origin[0]) * self.scale,
                (float(point[1]) - self.origin[1]) * self.scale)

    def invert(self, point) -> tuple:
        """An **input**-pixel point as a **source**-pixel point."""
        return (float(point[0]) / self.scale + self.origin[0],
                float(point[1]) / self.scale + self.origin[1])

    def to_input_points(self, points) -> np.ndarray:
        """``(K, 2)`` source-pixel points as ``(K, 2)`` input-pixel points."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        return (points - np.array(self.origin, np.float32)) * self.scale

    def invert_points(self, points) -> np.ndarray:
        """``(K, 2)`` input-pixel points as ``(K, 2)`` source-pixel points."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        return points / self.scale + np.array(self.origin, np.float32)

    def contains(self, point) -> bool:
        """True when a **source**-pixel point falls inside the window."""
        x, y = float(point[0]), float(point[1])
        return (self.origin[0] <= x < self.origin[0] + self.side
                and self.origin[1] <= y < self.origin[1] + self.side)

    def describe(self) -> str:
        """One line for a readout: the window, the factor and the input size."""
        return (f"{self.side} x {self.side} px at ({self.origin[0]}, "
                f"{self.origin[1]}), x{self.scale:.3f} -> {self.size} x {self.size} px")


def pose_input_for_box(box, image_size, size=DEFAULT_INPUT_SIZE,
                       margin=DEFAULT_MARGIN) -> PoseInput:
    """The square window that shows one box, and the mapping to the network input.

    The window is centred on the box and its side is ``max(w, h) * (1 + 2 * margin)``
    - square, so the mapping is invertible in closed form, and grown by ``margin`` so
    a landmark sitting on the box edge is still inside. It is **not** clamped to the
    frame: a window that reaches past the image is edge-padded by
    :func:`window_image`, so the geometry (and therefore the mapping) never changes.

    Raises:
        PoseDataError: the box is degenerate, or the image size is not positive.
    """
    x1, y1, x2, y2 = _normalised(box)
    width, height = int(image_size[0]), int(image_size[1])
    if x2 <= x1 or y2 <= y1:
        raise PoseDataError(f"degenerate box {box} - a crop needs a rectangle")
    if width <= 0 or height <= 0:
        raise PoseDataError(f"degenerate image size {image_size}")
    side = max(1, int(round(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * float(margin)))))
    centre_x, centre_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    origin = (int(round(centre_x - side / 2.0)), int(round(centre_y - side / 2.0)))
    return PoseInput(origin=origin, side=side, size=int(size),
                     source_width=width, source_height=height)


def window_image(image, pose_input: PoseInput):
    """The ``(size, size, 3)`` input picture of a window, edge-padded where needed.

    A window that reaches past the frame is filled by **replicating the edge**, not
    with black: a black band is a hard artificial edge the backbone would happily
    learn as a feature. Returns ``None`` when the window misses the image entirely -
    a caller that crops nothing has a bug worth surfacing, not an empty tensor.
    """
    if image is None or getattr(image, "ndim", 0) < 2:
        return None
    height, width = image.shape[:2]
    x1, y1 = pose_input.origin
    x2, y2 = x1 + pose_input.side, y1 + pose_input.side
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(width, x2), min(height, y2)
    patch = image[top:bottom, left:right]
    if not patch.size:
        return None
    padded = np.pad(patch, ((top - y1, y2 - bottom), (left - x1, x2 - right), (0, 0)),
                    mode="edge")
    if (padded.shape[1], padded.shape[0]) == (pose_input.size, pose_input.size):
        return padded
    import cv2                         # heavy import, inside the function (SR-CORE-011)

    return cv2.resize(padded, (pose_input.size, pose_input.size),
                      interpolation=cv2.INTER_LINEAR)


@dataclass(frozen=True)
class PoseSample:
    """One training/validation item: an input picture, its keypoints, and the mapping.

    ``keypoints`` are ``(K, 2)`` in **input pixels**, aligned to the corpus's keypoint
    order; ``source_points`` keep the original ``((name, x, y), ...)`` in source
    pixels for reporting; ``visible`` is ``0`` for a keypoint the window does not
    contain, so the loss ignores it instead of learning a clamped lie.
    """

    image: np.ndarray          # (size, size, 3) BGR uint8
    keypoints: np.ndarray      # (K, 2) float32, input pixels
    visible: np.ndarray        # (K,) float32, 1.0 where the point is inside
    source_points: tuple       # ((name, x, y), ...) source pixels
    pose_input: PoseInput      # the mapping this sample was built with
    path: object               # the file it came from
    specimen_index: int = 1
    bbox: tuple = ()           # the source-pixel box the crop was built from
    length_um: float = 0.0     # the stored body length, for reporting only
    augmented: bool = False

    @property
    def size(self) -> tuple:
        """``(width, height)`` of the input picture (square)."""
        height, width = self.image.shape[:2]
        return width, height

    def source_keypoints(self) -> np.ndarray:
        """``(K, 2)`` keypoints mapped back to **source** pixels."""
        return self.pose_input.invert_points(self.keypoints)


class PoseDataset(Dataset):
    """The samples of one split of a :class:`~daphnia.vision.pose_corpus.PoseCorpus`.

    Built from the corpus rather than the database, so a training run needs only the
    exported folder. A sample whose frame cannot be read, or whose box cannot become
    a crop, is skipped by name rather than silently turned into a negative example -
    training on it would teach the model about a picture it never saw.
    """

    def __init__(self, corpus, split="train", input_size=DEFAULT_INPUT_SIZE,
                 augment=False, seed=0, margin=DEFAULT_MARGIN,
                 flip_probability=0.5):
        self.corpus = corpus
        self.split = split
        self.input_size = int(input_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.margin = float(margin)
        self.flip_probability = float(flip_probability) if augment else 0.0
        self.epoch = 0
        self.names = tuple(corpus.names)
        self.items = list(corpus.samples_of(split))

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch):
        """Change the augmentation stream (called once per epoch by the trainer)."""
        self.epoch = int(epoch)

    def keypoint_names(self) -> tuple:
        """The keypoint names every sample of this dataset carries, in order."""
        return self.names

    def _build(self, item, rng=None) -> PoseSample:
        """One sample: read the frame, crop the box, carry the keypoints along."""
        frame = read_frame(item.image)
        if frame is None:
            raise PoseDataError(f"could not read {item.image}")
        height, width = frame.shape[:2]
        image_size = (item.image_width or width, item.image_height or height)
        pose = pose_input_for_box(item.bbox, image_size, self.input_size, self.margin)
        window = window_image(frame, pose)
        if window is None:
            raise PoseDataError(
                f"box {tuple(round(v, 1) for v in item.bbox)} of {item.relative} does "
                "not overlap its image - nothing to crop")
        points = np.asarray([[x, y] for _name, x, y in item.points], np.float32)
        visible = np.asarray([1.0 if pose.contains(point) else 0.0
                              for point in points], np.float32)
        input_points = pose.to_input_points(points)
        if rng is not None and rng.random() < self.flip_probability:
            window = np.ascontiguousarray(window[:, ::-1])
            # A continuous coordinate mirrors as S - x, which is what an array flip
            # does to pixel centres; no left/right semantics are involved.
            input_points[:, 0] = self.input_size - input_points[:, 0]
        if rng is not None:
            window = colour_jitter(window, rng)
        return PoseSample(image=window, keypoints=input_points, visible=visible,
                          source_points=item.points, pose_input=pose,
                          path=item.image, specimen_index=item.specimen_index,
                          bbox=tuple(item.bbox),
                          length_um=float(item.length_um() or 0.0),
                          augmented=rng is not None)

    def __getitem__(self, index) -> PoseSample:
        if not self.augment:
            return self._build(self.items[index])
        rng = np.random.default_rng((self.seed, self.epoch, int(index)))
        return self._build(self.items[index], rng)


def collate(samples) -> dict:
    """Stack samples into a batch the model can take, targets in input pixels.

    The pictures are already the same square, so no padding is needed - a batch is a
    plain stack, and every coordinate stays where it was.
    """
    import torch

    images = torch.stack([to_tensor(sample.image) for sample in samples])
    targets = [{"keypoints": torch.from_numpy(sample.keypoints),
                "visible": torch.from_numpy(sample.visible)}
               for sample in samples]
    return {"images": images, "targets": targets, "samples": list(samples)}
