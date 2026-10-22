"""Runtime settings, overridable through environment variables.

Every value has a safe local-development default, so the app runs out of the box.
"""

import getpass
import os

# --- Streaming (MediaMTX) -------------------------------------------------
# Host as seen from this machine: the Mac's LAN IP (e.g. 192.168.x.x), or a
# tether/Ethernet address when the phone is wired.
STREAM_HOST = os.environ.get("DAPHNIA_STREAM_HOST", "127.0.0.1")

# Ingest path the phone publishes to, and the pipeline reads from.
STREAM_PATH = os.environ.get("DAPHNIA_STREAM_PATH", "live/cam")

# The phone publishes SRT here; OpenCV reads the same path over RTSP and MediaMTX
# converts between the two.
SRT_PORT = int(os.environ.get("DAPHNIA_SRT_PORT", "8890"))
RTSP_PORT = int(os.environ.get("DAPHNIA_RTSP_PORT", "8554"))

MEDIAMTX_BINARY = os.environ.get("DAPHNIA_MEDIAMTX_BINARY", "mediamtx")

# MediaMTX's control API (``apiAddress`` in ``config/mediamtx.yml``), localhost-only.
# The app asks it what codec the publisher is sending, because that is the difference
# between "no frames arriving" and "your phone is sending H.265" - a setting to change.
MEDIAMTX_API_HOST = os.environ.get("DAPHNIA_MEDIAMTX_API_HOST", "127.0.0.1")
MEDIAMTX_API_PORT = int(os.environ.get("DAPHNIA_MEDIAMTX_API_PORT", "9997"))

# How loud FFmpeg's *own* decoder log is, which OpenCV writes to stderr. Quiet by
# default: an H.265 publisher makes libavcodec print two lines per undecodable frame
# (measured 2026-10-19: 42 lines for one 120-frame read, 0 on H.264, 0 at this level),
# while the app states the same condition once, in the status bar, with the setting to
# change it. Set a positive level (e.g. 32 = AV_LOG_WARNING) to debug the decoder.
FFMPEG_LOGLEVEL = os.environ.get("DAPHNIA_FFMPEG_LOGLEVEL", "-8")

# Frame orientation used by the live preview. "landscape" rotates portrait frames
# so the display is always landscape; "native" keeps exactly what the phone sends
# (then a portrait stream makes the preview tall and the body scrolls).
STREAM_ORIENTATION = os.environ.get("DAPHNIA_STREAM_ORIENTATION", "landscape")

# --- Acquisition ----------------------------------------------------------
# Frames captured per well before the sharpest one is auto-selected.
BURST_SIZE = int(os.environ.get("DAPHNIA_BURST_SIZE", "5"))

# Longest a **recording** (Study 2's Start/Stop capture) may run before it stops itself.
#
# The operator decides the length - that is what a recording is for - so this is not a
# capture size but a disk guard: a forgotten recording must not fill the drive, and a
# recording that reaches it **says so** and stores what arrived (`SR-STUDY2-N06`).
# 300 frames is ~10 s at the 30 fps this rig streams and ~360 MB on disk, measured
# 2026-10-22 on the M4 with the rig's own 1280x720 frames: 1241 kB each, 17.2 ms to
# write plus 2.5 ms to score. Raise it for a longer take; the page reads it per frame.
RECORD_MAX_FRAMES = int(os.environ.get("DAPHNIA_RECORD_MAX_FRAMES", "300"))

# --- Calibration ----------------------------------------------------------
# How long a calibration is trusted before the app suggests re-verifying it. A
# phone clamped to an eyepiece can be bumped at any time, which invalidates the
# scale silently, so age is a warning signal rather than a formality. Set to 0 to
# disable the warning.
CALIBRATION_MAX_AGE_HOURS = float(
    os.environ.get("DAPHNIA_CALIBRATION_MAX_AGE_HOURS", "12"))

# Relative scale uncertainty above which a calibration is called weak, in percent.
# Derived from the clicked reference span, so a single short pitch can exceed it.
CALIBRATION_WEAK_UNCERTAINTY_PCT = float(
    os.environ.get("DAPHNIA_CALIBRATION_WEAK_UNCERTAINTY_PCT", "2.0"))

# --- Operator -------------------------------------------------------------
# Recorded on sessions, calibration records and measurements.
OPERATOR_NAME = os.environ.get("DAPHNIA_OPERATOR", getpass.getuser())
