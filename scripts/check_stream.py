#!/usr/bin/env python3
"""Probe a stream and report frame geometry; optionally save one frame.

    python scripts/check_stream.py                          # MediaMTX RTSP default
    python scripts/check_stream.py rtsp://host:8554/live/cam --save frame.png
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daphnia import settings
from daphnia.streaming import mediamtx, read_rtsp_url
from daphnia.streaming.reader import FFMPEG_CAPTURE_OPTIONS


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe a live stream.")
    parser.add_argument("url", nargs="?", default=None, help="stream URL (default: MediaMTX RTSP)")
    parser.add_argument("--frames", type=int, default=30, help="frames to read (default: 30)")
    parser.add_argument("--save", type=Path, default=None, help="write the last frame to this file")
    args = parser.parse_args()

    url = args.url or read_rtsp_url()
    codec = mediamtx.stream_codec()
    print(f"mediamtx binary : {mediamtx.binary_path()}")
    print(f"mediamtx running: {mediamtx.is_running()}")
    print(f"stream url      : {url}")
    print(f"publisher codec : {codec or 'unknown (nothing published, or no API answer)'}")

    try:
        import cv2
    except ImportError:
        print("error: OpenCV is not installed — cannot read frames", file=sys.stderr)
        return 2

    # Same quiet decoder log the app uses (`settings.FFMPEG_LOGLEVEL`): what is being
    # published is reported above, which is the part an operator can act on.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", settings.FFMPEG_LOGLEVEL)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", FFMPEG_CAPTURE_OPTIONS)
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        print("error: could not open the stream", file=sys.stderr)
        return 1

    # A failed read is normal (a decode failure, or a moment with nothing buffered):
    # count it and keep going, exactly as `daphnia.streaming.reader` does.
    frames, failures, frame, started = 0, 0, None, time.monotonic()
    for _ in range(args.frames):
        ok, grabbed = capture.read()
        if not ok:
            failures += 1
            continue
        frames += 1
        frame = grabbed
    elapsed = time.monotonic() - started

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    print(f"frames read     : {frames}")
    print(f"frames failed   : {failures}   (a decode failure or a moment with no frame)")
    print(f"resolution      : {width}x{height}")
    print(f"reported fps    : {fps:.2f}")
    if elapsed > 0:
        print(f"read rate       : {frames / elapsed:.1f} fps")

    if mediamtx.is_hevc(codec):
        print("advice          : H.265/HEVC costs this decoder the first ~120 frames "
              "about 21 of them (the reader joins mid-GOP; measured 2026-10-19).\n"
              "                  Set the phone's video codec to H.264.")

    if frame is not None and args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), frame)
        print(f"saved frame     : {args.save}")

    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
