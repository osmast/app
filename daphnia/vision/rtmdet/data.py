"""Frames and boxes in, normalised tensors out - with the augmentation RTMDet wants.

Three things live here, in the order they happen to a frame:

1. **letterbox** - resize so the longest side is ``input_size``, then pad so both
   sides are a multiple of 32 (the deepest stride). The pad is *recorded*
   (:class:`Letterbox`), never guessed, because the boxes come back out of the
   network in this frame and must be mapped to the original image before anything
   draws or measures them.
2. **augmentation** - the ones that matter for a small corpus: a **mosaic** (four
   frames in one, which is what turns 223 images into enough context for a
   detector to learn "animal vs. detritus") and a **horizontal flip**, plus a cheap
   colour jitter, because the illumination of a phone clamped to an eyepiece drifts
   between sessions.
3. **normalisation** - RTMDet's own numbers, in **BGR** because that is what
   OpenCV reads.

The dataset reads a :class:`~daphnia.vision.export.Export`, so it can only train
on what the Dataset surface actually produced - and a sample whose boxes all fall
outside the image is dropped instead of becoming a confusing negative.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from daphnia.vision.export import Export

#: The input normalisation, from RTMDet's configuration. BGR, because the frames
#: arrive from OpenCV in BGR and converting them would only add a copy.
MEAN = (103.53, 116.28, 123.675)
STD = (57.375, 57.12, 58.395)

#: Every side is padded to a multiple of this - the deepest stride in the head.
SIZE_DIVISOR = 32

#: The single class this corpus has. A detector trained on one organism needs one
#: name, and keeping it here (rather than as a bare 0) is what lets the Predict
#: surface draw "daphnia 0.87" instead of "class 0".
CLASSES = ("daphnia",)

#: Probability that a training sample is built from four frames instead of one.
MOSAIC_PROBABILITY = 0.5


class MissingOpenCV(RuntimeError):
    """OpenCV is needed to read and resize frames but could not be imported."""


def _cv2():
    """Import OpenCV lazily, with a message that says what to do about it."""
    try:
        import cv2
    except ImportError as error:                     # pragma: no cover - env issue
        raise MissingOpenCV(
            "OpenCV is required to train or run the detector. Install it into the "
            "environment that runs the training (see environment.yml).") from error
    return cv2


@dataclass(frozen=True)
class Letterbox:
    """How a frame was mapped into the padded input: nothing here is implicit.

    :meth:`invert` is the only way back, so no caller can forget a step of the
    mapping - forgetting one is how a detection ends up drawn 280 pixels off.
    """

    scale: float
    pad_left: int
    pad_top: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int

    def invert(self, boxes):
        """Map ``(N, 4)`` boxes from the padded frame back to the original image."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if boxes.size == 0:
            return boxes
        scale = self.scale if self.scale > 0 else 1.0
        mapped = np.stack([
            (boxes[:, 0] - self.pad_left) / scale,
            (boxes[:, 1] - self.pad_top) / scale,
            (boxes[:, 2] - self.pad_left) / scale,
            (boxes[:, 3] - self.pad_top) / scale], axis=1)
        mapped[:, 0] = np.clip(mapped[:, 0], 0, self.source_width)
        mapped[:, 2] = np.clip(mapped[:, 2], 0, self.source_width)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, self.source_height)
        mapped[:, 3] = np.clip(mapped[:, 3], 0, self.source_height)
        return mapped

    def describe(self) -> str:
        """One line for a log: how the frame became the input."""
        return (f"{self.source_width}x{self.source_height} -> "
                f"{self.target_width}x{self.target_height} "
                f"(x{self.scale:.3f}, pad {self.pad_left},{self.pad_top})")


def letterbox(frame, input_size=None, divisor=SIZE_DIVISOR):
    """Resize ``frame`` so its longest side is ``input_size``, then pad.

    Returns ``(padded_frame, Letterbox)``. ``input_size=None`` keeps the frame's
    own size and only pads it to the divisor - what inference does, so a live frame
    is never resized for no reason.
    """
    height, width = frame.shape[:2]
    scale = (min(float(input_size) / width, float(input_size) / height)
             if input_size else 1.0)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if (new_width, new_height) != (width, height):
        cv2 = _cv2()
        frame = cv2.resize(frame, (new_width, new_height),
                           interpolation=cv2.INTER_LINEAR)

    target_width = int(np.ceil(new_width / divisor) * divisor)
    target_height = int(np.ceil(new_height / divisor) * divisor)
    pad_right = target_width - new_width
    pad_bottom = target_height - new_height
    if pad_right or pad_bottom:
        # Replicate the edge rather than adding black: a black band is a hard
        # artificial edge the backbone would happily learn as a feature.
        frame = np.pad(frame, ((0, pad_bottom), (0, pad_right), (0, 0)),
                       mode="edge")
    return frame, Letterbox(scale=scale, pad_left=0, pad_top=0,
                            source_width=width, source_height=height,
                            target_width=target_width, target_height=target_height)


def to_tensor(frame) -> torch.Tensor:
    """``(H, W, 3)`` uint8 **BGR** -> normalised ``(3, H, W)`` float32 tensor."""
    array = np.asarray(frame, dtype=np.float32)
    mean = np.asarray(MEAN, dtype=np.float32)
    std = np.asarray(STD, dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def from_tensor(tensor) -> np.ndarray:
    """The inverse of :func:`to_tensor` - for showing what the network was fed."""
    array = tensor.detach().float().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray(MEAN, dtype=np.float32)
    std = np.asarray(STD, dtype=np.float32)
    return np.clip(array * std + mean, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class Sample:
    """One training/validation item: a frame, its boxes, and how they got there."""

    image: np.ndarray          # letterboxed (and maybe augmented) BGR uint8 frame
    boxes: np.ndarray          # (M, 4) float32, in ``image``'s coordinates
    labels: np.ndarray         # (M,) int64, class indices
    path: object               # the file it came from (None for a mosaic)
    letterbox: Letterbox       # how to get back to the original image
    augmented: bool = False

    @property
    def size(self):
        """``(width, height)`` of the sample's frame."""
        height, width = self.image.shape[:2]
        return width, height


def read_frame(path):
    """Read one frame from disk with the project's own loader.

    :func:`daphnia.vision.images.load_image` is used rather than ``cv2.imread`` so
    a corpus annotated from ``.HEIC`` or ``.TIFF`` photos trains exactly like one
    from PNG bursts - the loader knows what macOS can convert and ``cv2.imread``
    would silently return ``None``.
    """
    from daphnia.vision.images import load_image

    return load_image(path)


def _clip_boxes(boxes, width, height):
    """Clip ``(M, 4)`` boxes into ``width`` x ``height``, dropping what is left."""
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    clipped = np.stack([
        np.clip(boxes[:, 0], 0, width), np.clip(boxes[:, 1], 0, height),
        np.clip(boxes[:, 2], 0, width), np.clip(boxes[:, 3], 0, height)], axis=1)
    keep = ((clipped[:, 2] - clipped[:, 0]) >= 1.0) & \
           ((clipped[:, 3] - clipped[:, 1]) >= 1.0)
    return clipped[keep].astype(np.float32)


def horizontal_flip(frame, boxes):
    """Mirror a frame and its boxes.

    A daphnid is not chiral as far as a detector is concerned, so a flip is free
    augmentation - and an important one, because a plate is photographed from one
    side and every animal would otherwise be seen in the same orientation.
    """
    flipped = np.ascontiguousarray(frame[:, ::-1])
    if boxes.size == 0:
        return flipped, boxes
    width = frame.shape[1]
    mirrored = boxes.copy()
    mirrored[:, 0] = width - boxes[:, 2]
    mirrored[:, 2] = width - boxes[:, 0]
    return flipped, mirrored


def colour_jitter(frame, rng, brightness=0.25, contrast=0.25):
    """Randomly nudge brightness and contrast, in uint8.

    Cheap, and it matters more here than in a benchmark dataset: the light over the
    eyepiece drifts between sessions, so a detector that only ever saw one exposure
    is a detector that only works on the day it was trained.
    """
    array = frame.astype(np.float32)
    array = array * (1.0 + float(rng.uniform(-brightness, brightness)))
    mean = float(array.mean())
    array = (array - mean) * (1.0 + float(rng.uniform(-contrast, contrast))) + mean
    return np.clip(array, 0, 255).astype(np.uint8)


class DetectionDataset(Dataset):
    """The frames of one split of an :class:`~daphnia.vision.export.Export`.

    Deliberately built from the export rather than from the database: the export is
    the deliverable (``D1``) and the trainer must be able to run on a machine that
    has only that folder. A frame whose boxes all fall outside the image is dropped
    - training on it silently would teach "this frame has no animal", which is
    false, and would be the first thing that made a metric look odd.

    Augmentation is seeded per ``(seed, epoch, index)``: two runs with the same seed
    see the same sequence, and a different epoch is a different sequence. That is
    what makes a training run reproducible *and* still varied.
    """

    def __init__(self, export: Export, split="train", input_size=640,
                 augment=False, seed=0, mosaic_probability=MOSAIC_PROBABILITY):
        self.export = export
        self.split = split
        self.input_size = int(input_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.mosaic_probability = float(mosaic_probability) if augment else 0.0
        self.epoch = 0
        self.items = self._collect()

    # -- collection --------------------------------------------------------
    def _collect(self) -> list:
        """``[(path, boxes)]`` for every frame of the split that has a usable box."""
        grouped = {}
        for box in self.export.boxes_in(self.split):
            if box.outside():
                continue                     # reported by Export.warnings(), not used
            grouped.setdefault(box.relative, []).append(box.bbox)
        items = []
        for relative in sorted(grouped):
            path = self.export.image_path(relative)
            if path.exists():
                items.append((path, np.asarray(grouped[relative], dtype=np.float32)))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch):
        """Change the augmentation stream (called once per epoch by the trainer)."""
        self.epoch = int(epoch)

    # -- items -------------------------------------------------------------
    def _plain(self, index) -> Sample:
        """One frame, letterboxed, no augmentation - the honest validation path."""
        path, boxes = self.items[index]
        frame = read_frame(path)
        padded, mapping = letterbox(frame, self.input_size)
        scaled = boxes * mapping.scale if mapping.scale != 1.0 else boxes.copy()
        usable = _clip_boxes(scaled, padded.shape[1], padded.shape[0])
        return Sample(image=padded, boxes=usable,
                      labels=np.zeros(len(usable), dtype=np.int64),
                      path=path, letterbox=mapping)

    def _mosaic(self, indices, rng) -> Sample:
        """Four frames in one, each in a quadrant - the augmentation for a small corpus.

        This is the heaviest augmentation RTMDet uses, and the reason a few hundred
        images can teach "which blob is an animal": the composite places a different
        animal, from a different plate, in each corner, so the detector cannot solve
        the task by learning one well's fixed appearance.
        """
        cv2 = _cv2()
        size = self.input_size
        half = size // 2
        canvas = np.full((size, size, 3), 127, dtype=np.uint8)
        corners = ((0, 0), (half, 0), (0, half), (half, half))
        box_parts, label_parts = [], []
        for index, (offset_x, offset_y) in zip(indices, corners):
            sample = self._plain(index)
            height, width = sample.image.shape[:2]
            # A tile is roughly a quadrant, jittered: the animals must not always
            # appear at the same relative scale, or the mosaic teaches nothing.
            # Capped at the quadrant, so a tile can never overrun its own corner.
            target = max(32, min(half, int(half * float(rng.uniform(0.75, 1.25)))))
            factor = min(target / width, target / height)
            new_width = max(1, int(round(width * factor)))
            new_height = max(1, int(round(height * factor)))
            resized = cv2.resize(sample.image, (new_width, new_height),
                                 interpolation=cv2.INTER_LINEAR)
            dx = offset_x + int(rng.integers(0, max(0, half - new_width) + 1))
            dy = offset_y + int(rng.integers(0, max(0, half - new_height) + 1))
            # Several frames in one corpus can letterbox to different sizes, so the
            # tile is clipped to what the canvas can hold rather than assumed to fit.
            visible_width = min(new_width, size - dx)
            visible_height = min(new_height, size - dy)
            if visible_width <= 0 or visible_height <= 0:
                continue
            canvas[dy:dy + visible_height, dx:dx + visible_width] = \
                resized[:visible_height, :visible_width]
            if len(sample.boxes):
                moved = sample.boxes * factor
                moved += np.array([dx, dy, dx, dy], dtype=np.float32)
                kept = _clip_boxes(moved, size, size)
                box_parts.append(kept)
                label_parts.append(np.zeros(len(kept), dtype=np.int64))

        boxes = (np.concatenate(box_parts) if box_parts
                 else np.zeros((0, 4), dtype=np.float32))
        labels = (np.concatenate(label_parts) if label_parts
                  else np.zeros((0,), dtype=np.int64))
        return Sample(image=canvas, boxes=boxes, labels=labels, path=None,
                      letterbox=Letterbox(1.0, 0, 0, size, size, size, size),
                      augmented=True)

    def __getitem__(self, index) -> Sample:
        if not self.augment:
            return self._plain(index)

        rng = np.random.default_rng((self.seed, self.epoch, int(index)))
        if rng.random() < self.mosaic_probability and len(self) >= 4:
            others = [int(value) for value in rng.integers(0, len(self), size=3)]
            sample = self._mosaic([int(index)] + others, rng)
        else:
            sample = self._plain(index)

        frame, boxes = sample.image, sample.boxes
        if rng.random() < 0.5:
            frame, boxes = horizontal_flip(frame, boxes)
        frame = colour_jitter(frame, rng)
        return Sample(image=frame, boxes=boxes, labels=sample.labels,
                      path=sample.path, letterbox=sample.letterbox,
                      augmented=True)

    def class_names(self) -> tuple:
        """The names of the classes this corpus carries."""
        return CLASSES


def collate(samples) -> dict:
    """Stack samples into a batch, padding any odd size at the bottom right.

    Different frames can letterbox to different sizes (a portrait photo and a
    landscape one), so a batch pads to its own maximum. Padding only ever grows the
    bottom right, which leaves every coordinate - and every prior - where it was, so
    nothing has to be re-mapped because of the batch.
    """
    height = max(sample.image.shape[0] for sample in samples)
    width = max(sample.image.shape[1] for sample in samples)
    images, targets = [], []
    for sample in samples:
        frame = sample.image
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = np.pad(frame, ((0, height - frame.shape[0]),
                                   (0, width - frame.shape[1]), (0, 0)),
                           mode="edge")
        images.append(to_tensor(frame))
        targets.append({"boxes": torch.from_numpy(sample.boxes),
                        "labels": torch.from_numpy(sample.labels)})
    return {"images": torch.stack(images), "targets": targets,
            "samples": list(samples)}



