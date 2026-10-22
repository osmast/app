"""Turning a video file into frames - the one place a clip becomes measurable.

A video is **not** a file you can label: a corpus label, a measurement and a media
asset all key on a *frame file*, so importing a clip means writing its frames to
disk and queueing those paths. That is what this module does, and why it exists
instead of reading the clip lazily the way the Predict surface does (it only
*watches*, it stores nothing).

Frames are named ``f000123.png`` - zero-padded, so the **path order is the frame
order**, which is what lets an import be resumed from its file names alone.

OpenCV is imported **inside** the functions, so the Qt app starts without the CV
stack (``SR-CORE-011``) and the tests that do not need it still run.
"""

from pathlib import Path

#: Video containers the stack reads (H.264/H.265 in these wrappers; see
#: ``docs/streaming.md`` for why the codec matters).
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".ts")


def is_video(path) -> bool:
    """True when a path names a video container this stack can open."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def video_length(path):
    """How many frames a video holds, or ``None`` when it cannot be opened."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return total or None


def extract_frames(path, target_dir, *, step=1, limit=None, on_frame=None):
    """Write a video's frames to ``target_dir`` and return their paths, in order.

    Args:
        path: the video file.
        target_dir: where the frames are written (created if missing).
        step: keep every n-th frame (``step=1`` is every frame).
        limit: stop after this many frames (``None`` = all).
        on_frame: called with the running count, for a progress report.

    Raises:
        ValueError: the clip cannot be opened - named with the file, so the failure
            is never a silent empty queue.
    """
    import cv2
    path = Path(path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(
            f"could not open the video {path} - is it a format OpenCV can read? "
            "(H.264 in .mp4 is the safe choice)")

    written = []
    index = -1
    stride = max(1, int(step))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if index % stride != 0:
                continue
            target = target_dir / f"f{index:06d}.png"
            if not cv2.imwrite(str(target), frame):
                continue                      # an unwritable frame is skipped, not counted
            written.append(target)
            if on_frame is not None:
                on_frame(len(written))
            if limit is not None and len(written) >= int(limit):
                break
    finally:
        capture.release()
    return written
