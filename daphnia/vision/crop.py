"""Cropping one specimen out of a frame - the annotator's unit of work.

A frame in this corpus holds a median of **three** animals (measured on the first
526 boxes: 145 of 223 frames hold two or more), and the boxes are near-square
(median aspect ratio 1.06) because the animals swim at every angle. Labelling
keypoints on the whole frame therefore means finding the same animal twice with the
eye - once to know which box you are on, once to place the point - and at ~84 px of
animal in a 1280 px frame the point lands wherever the zoom happened to be.

So the unit of work is one **box**: the frame is cropped to that box (plus a
context margin) and magnified by an integer factor, and the operator labels one
animal at a time on a picture where the animal is big.

## The one risk, and how it is contained

Crop pixels are not source pixels. A keypoint stored in the wrong frame of
reference looks right on screen and is wrong in the corpus - the same class of bug
as ``mlContext.md`` **law 1** ("one frame of reference, owned by one object with an
``invert``"), which in this repo once reported AP50 0.000 for thirty epochs of a
model that was learning. So a :class:`Crop` owns the mapping *and* its ``invert``,
exactly as :class:`~daphnia.vision.rtmdet.data.Letterbox` owns the letterbox's; no
caller re-derives it, and the round trip is unit-tested with hand-computed numbers.

The magnification is an **integer** factor and the resampling is **nearest
neighbour**, so the magnified picture is an exact enlargement: each source pixel
becomes a ``scale x scale`` block. Interpolation would invent detail that is not in
the photograph, and an annotator placing ground truth on invented detail is
guessing. Qt-free and NumPy-only, so the mapping is testable and scriptable.
"""

import math
from dataclasses import dataclass

#: How much context is kept around the box, as a fraction of its longest side.
DEFAULT_MARGIN = 0.25

#: The longest side a magnified crop aims for, in pixels.
DEFAULT_TARGET_SIZE = 384

#: Never magnify more than this: past ~8x a nearest-neighbour enlargement is a wall
#: of blocks and the operator is guessing *between* pixels.
MAX_SCALE = 8


def _normalised(box) -> tuple:
    """``(x1, y1, x2, y2)`` with the corners sorted, so a drag direction never matters."""
    x1, y1, x2, y2 = (float(value) for value in box)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


@dataclass(frozen=True)
class Crop:
    """A rectangular window of a source image, magnified by an integer factor.

    ``origin`` and ``size`` are **integers in source-image pixels** - the frame of
    reference every label is stored in - and ``scale`` maps them to the picture that
    is shown. Both directions of the mapping live here and nowhere else::

        >>> crop = Crop(origin=(100, 50), size=(40, 30), scale=4)
        >>> crop.to_crop((110, 60))
        (40.0, 40.0)
        >>> crop.invert((40, 40))
        (110.0, 60.0)

    The integers are enforced in ``__post_init__`` so that ``window()`` - which
    slices the array with them - can never disagree with ``invert()``.
    """

    origin: tuple                 # (x, y) of the window's top-left, source pixels
    size: tuple                   # (width, height) of the window, source pixels
    scale: float = 1.0            # the output is the window magnified by this

    def __post_init__(self):
        object.__setattr__(self, "origin",
                           (int(round(self.origin[0])), int(round(self.origin[1]))))
        object.__setattr__(self, "size",
                           (int(round(self.size[0])), int(round(self.size[1]))))
        object.__setattr__(self, "scale", float(self.scale))

    @property
    def width(self) -> int:
        """Window width, in source pixels."""
        return int(self.size[0])

    @property
    def height(self) -> int:
        """Window height, in source pixels."""
        return int(self.size[1])

    @property
    def output_size(self) -> tuple:
        """``(width, height)`` of the magnified picture, in crop pixels."""
        return (int(round(self.width * self.scale)),
                int(round(self.height * self.scale)))

    def to_crop(self, point) -> tuple:
        """A **source**-pixel point as crop pixels (what the canvas shows)."""
        return ((float(point[0]) - self.origin[0]) * self.scale,
                (float(point[1]) - self.origin[1]) * self.scale)

    def invert(self, point) -> tuple:
        """A **crop**-pixel point as source pixels (what a label stores).

        Named ``invert`` on purpose: it is the one way back, so a click cannot be
        stored in the picture's frame of reference by accident (``mlContext.md``
        law 1).
        """
        return (float(point[0]) / self.scale + self.origin[0],
                float(point[1]) / self.scale + self.origin[1])

    def contains(self, point) -> bool:
        """True when a **source**-pixel point falls inside the window."""
        x, y = float(point[0]), float(point[1])
        return (self.origin[0] <= x < self.origin[0] + self.width
                and self.origin[1] <= y < self.origin[1] + self.height)

    def window(self, image):
        """The magnified window of ``image`` (a BGR array), or ``None``.

        The slice uses exactly the integers :meth:`invert` uses, and an integer
        factor with **nearest-neighbour** resampling, so the enlargement is exact.
        """
        if image is None or getattr(image, "ndim", 0) < 2:
            return None
        height, width = image.shape[:2]
        x1, y1 = self.origin
        x2, y2 = x1 + self.width, y1 + self.height
        if x2 <= x1 or y2 <= y1 or x1 >= width or y1 >= height:
            return None
        patch = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if not patch.size:
            return None
        factor = int(round(self.scale))
        if factor <= 1:
            return patch
        import cv2                    # heavy import, inside the function (SR-CORE-011)

        return cv2.resize(patch, (patch.shape[1] * factor, patch.shape[0] * factor),
                          interpolation=cv2.INTER_NEAREST)

    def describe(self) -> str:
        """One line for a readout: the window, the factor and the picture size."""
        return (f"{self.width} x {self.height} px at ({self.origin[0]}, "
                f"{self.origin[1]}), x{self.scale:g} -> {self.output_size[0]} x "
                f"{self.output_size[1]} px")


def crop_for_box(box, image_size, margin=DEFAULT_MARGIN,
                 target_size=DEFAULT_TARGET_SIZE, max_scale=MAX_SCALE) -> Crop:
    """The crop that shows one box, with context, magnified to be workable.

    ``box`` is ``(x1, y1, x2, y2)`` in source pixels (any corner order) and
    ``image_size`` is ``(width, height)`` of the frame. The window is the box grown
    by ``margin`` of its **longest** side on every side - context, so a landmark
    that sits on the box edge is still placeable - and clamped to the image.

    ``scale`` is the integer factor that brings the longest side to about
    ``target_size``, never above ``max_scale`` and **never below 1**: a crop is
    never shrunk, because throwing pixels away would make the operator guess.

    Raises:
        ValueError: the box is degenerate, or lies entirely outside the image. A
            caller that asks for a crop of nothing has a bug, and a silently empty
            picture would be blamed on the photograph.
    """
    x1, y1, x2, y2 = _normalised(box)
    width, height = int(image_size[0]), int(image_size[1])
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate box {box} - a crop needs a rectangle")
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate image size {image_size}")
    pad = float(margin) * max(x2 - x1, y2 - y1)
    left = max(0, int(math.floor(x1 - pad)))
    top = max(0, int(math.floor(y1 - pad)))
    right = min(width, int(math.ceil(x2 + pad)))
    bottom = min(height, int(math.ceil(y2 + pad)))
    if right <= left or bottom <= top:
        raise ValueError(
            f"box {box} lies outside a {width} x {height} image - nothing to crop")
    span = max(right - left, bottom - top)
    scale = max(1, min(int(max_scale), int(round(float(target_size) / span)) or 1))
    return Crop(origin=(left, top), size=(right - left, bottom - top), scale=scale)
