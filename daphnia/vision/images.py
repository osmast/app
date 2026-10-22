"""Reading an image from anywhere: any format, any filename, any size.

The calibration target is a *photo*, and the most likely photo is the one already on
the phone - which is a ``.HEIC`` by default. OpenCV's codecs do not include HEIF, so a
naive ``cv2.imread`` fails on exactly the file an operator would try first. This module
is the one place that reads an image, and it goes through three stages:

1. **bytes, not paths** - ``np.frombuffer`` + ``cv2.imdecode``, which sidesteps
   platform-specific path handling entirely;
2. **the platform's own converter** when OpenCV cannot decode the format - ``sips``
   on macOS, a system tool, so no new dependency and no bundled codec;
3. **a clear refusal** if neither works, naming the file and the reason. A silent
   ``None`` from ``imread`` is how "any photo" turns into a mystery.

Qt is deliberately absent: this is usable from a script or a notebook, like the rest
of :mod:`daphnia.vision`.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

#: macOS ships ``sips`` at a fixed location; it reads HEIC and much more.
SIPS = "/usr/bin/sips"

#: How long the system converter may take before it is abandoned, in seconds.
CONVERT_TIMEOUT_S = 30

#: Formats OpenCV cannot decode in a stock build but that a phone produces.
SYSTEM_CONVERTED_SUFFIXES = (".heic", ".heif", ".dng", ".avif")


class ImageLoadError(ValueError):
    """An image file could not be read by OpenCV or by the system converter."""


def _decode(data: bytes):
    """Decode image bytes with OpenCV, or return ``None`` when unsupported."""
    try:
        import cv2
    except ImportError as error:                     # pragma: no cover - env issue
        raise ImageLoadError(
            "OpenCV is not available - install it (conda env install / opencv)") \
            from error
    if not data:
        return None
    try:
        buffer = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:                                # noqa: BLE001 - cv2 raises wildly
        return None


def _via_system_tool(path: Path):
    """Convert with ``sips`` and decode the result, or return ``None``.

    Only macOS has ``sips``; elsewhere this is a no-op and the caller reports the
    unsupported format.
    """
    if sys.platform != "darwin" or not os.path.exists(SIPS):
        return None
    with tempfile.TemporaryDirectory(prefix="daphnia-image-") as folder:
        target = Path(folder) / f"{path.stem or 'image'}.png"
        try:
            subprocess.run(
                [SIPS, "-s", "format", "png", str(path), "--out", str(target)],
                check=True, capture_output=True, timeout=CONVERT_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return None
        try:
            return _decode(target.read_bytes())
        except OSError:
            return None


def load_image(path, allow_system_conversion=True):
    """Read any image file into a BGR ``numpy.ndarray``.

    Args:
        path: the image file.
        allow_system_conversion: when OpenCV cannot decode the format, try the
            platform's own converter (macOS ``sips``). Off by default in tests.

    Raises:
        ImageLoadError: the file is missing, unreadable, empty, or in a format that
            neither OpenCV nor the system converter can decode.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"no such file: {path}")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ImageLoadError(f"could not read {path}: {error}") from error
    if not data:
        raise ImageLoadError(f"{path} is empty")

    frame = _decode(data)
    if frame is not None:
        return frame

    if allow_system_conversion:
        frame = _via_system_tool(path)
        if frame is not None:
            return frame

    suffix = path.suffix.lower() or "(no extension)"
    hint = ""
    if suffix in SYSTEM_CONVERTED_SUFFIXES:
        hint = (f" - {suffix} needs a converter and none is available here "
                "(on macOS it is converted automatically with sips)")
    raise ImageLoadError(f"could not decode {path} ({suffix}){hint}")


def _png_size(data: bytes):
    """``(width, height)`` from a PNG header, or ``None``."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return (int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"))
    return None


#: JPEG start-of-frame markers that carry the dimensions.
_JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
             0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg_size(data: bytes):
    """``(width, height)`` from a JPEG's SOF marker, or ``None``."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    index, length = 2, len(data)
    while index + 9 < length:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        segment = int.from_bytes(data[index + 2:index + 4], "big")
        if marker in _JPEG_SOF:
            height = int.from_bytes(data[index + 5:index + 7], "big")
            width = int.from_bytes(data[index + 7:index + 9], "big")
            return width, height
        index += 2 + segment
    return None


def header_size(path):
    """``(width, height)`` from an image file's **header**, or ``None``.

    Reads only the first kilobytes of the file, so it needs **no OpenCV** - which
    matters because the app must start without the CV stack (``SR-CORE-011``) and
    the database migration needs a frame's size to record it with a label. PNG and
    JPEG (the formats a phone and this app produce) are covered; anything else
    returns ``None`` and the caller falls back to a full decode.
    """
    path = Path(path)
    try:
        with path.open("rb") as handle:
            head = handle.read(64 * 1024)
    except OSError:
        return None
    return _png_size(head) or _jpeg_size(head)


def image_size(path):
    """``(width, height)`` of an image file, without keeping the pixels.

    The header is tried first (no decoder needed); a format it does not understand
    falls back to a full :func:`load_image`, so ``.HEIC`` and friends still work.
    """
    size = header_size(path)
    if size is not None:
        return size
    frame = load_image(path)
    height, width = frame.shape[:2]
    return width, height
