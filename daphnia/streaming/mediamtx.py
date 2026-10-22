"""MediaMTX discovery and health check.

The phone publishes SRT to MediaMTX and OpenCV reads the same path over RTSP. The
server configuration is version-controlled in ``config/mediamtx.yml``;
``scripts/setup_mediamtx.sh`` installs it system-wide.
"""

import json
import shutil
import socket
from pathlib import Path
from urllib.request import urlopen

from daphnia import paths, settings

# Homebrew places the system-wide configuration here (macOS / Apple Silicon).
SYSTEM_CONFIG = Path("/opt/homebrew/etc/mediamtx/mediamtx.yml")


def binary_path():
    """Absolute path of the MediaMTX binary, or ``None`` when not installed."""
    return shutil.which(settings.MEDIAMTX_BINARY)


def config_path() -> Path:
    """The version-controlled MediaMTX configuration bundled with the repo."""
    return paths.MEDIAMTX_CONFIG


def is_running(host=None, port=None, timeout=0.5) -> bool:
    """True when the MediaMTX RTSP port accepts TCP connections."""
    host = host or settings.STREAM_HOST
    port = port or settings.RTSP_PORT
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def status() -> dict:
    """Snapshot of the local MediaMTX installation and runtime state."""
    binary = binary_path()
    config = config_path()
    return {
        "installed": binary is not None,
        "binary": binary,
        "config": str(config),
        "config_exists": config.exists(),
        "system_config": str(SYSTEM_CONFIG),
        "running": is_running(),
    }


# --- the control API: what is being published ------------------------------
def api_url(path="/v3/paths/list", host=None, port=None) -> str:
    """One MediaMTX control-API URL (localhost by default)."""
    host = host or settings.MEDIAMTX_API_HOST
    port = port or settings.MEDIAMTX_API_PORT
    return f"http://{host}:{port}{path}"


def fetch_json(url, timeout=1.0):
    """GET one JSON document, or ``None`` when the API cannot be reached.

    ``None`` means "I could not ask", never "the answer is no": a caller must not turn
    an unreachable API into a verdict about the stream.
    """
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:                      # noqa: BLE001 - offline, refused, not JSON
        return None


def path_items(payload) -> list:
    """The ``items`` of a ``/v3/paths/list`` answer (``[]`` when it is not one).

    Named ``path_items`` and not ``paths``: this module already imports
    :mod:`daphnia.paths` for the bundled configuration, and a helper that shadowed it
    broke every caller of that module (found by the streaming tests).
    """
    if not isinstance(payload, dict):
        return []
    items = payload.get("items")
    return items if isinstance(items, list) else []


def path_status(payload, name=None):
    """One path's status from a paths list, or ``None`` when it is not listed.

    ``None`` ("this answer has no such path") is not the same as a path that exists and
    is idle - that one comes back with ``online: False`` and no tracks.
    """
    wanted = settings.STREAM_PATH if name is None else name
    for item in path_items(payload):
        if isinstance(item, dict) and item.get("name") == wanted:
            return item
    return None


def codec_of(item) -> str:
    """The video codec of a path status, e.g. ``H264`` / ``H265`` (``""`` when idle)."""
    if not isinstance(item, dict):
        return ""
    tracks = item.get("tracks")
    return str(tracks[0]) if isinstance(tracks, list) and tracks else ""


def stream_codec(host=None, port=None, name=None, timeout=1.0) -> str:
    """Ask the local MediaMTX API what codec is being published.

    ``""`` means "not published, or the API could not be asked" - the app must say that
    rather than claim a codec.
    """
    payload = fetch_json(api_url("/v3/paths/list", host, port), timeout=timeout)
    return codec_of(path_status(payload, name))


def is_hevc(codec) -> bool:
    """True for the one codec this pipeline pays frames for: H.265/HEVC.

    Measured 2026-10-19 through SRT -> MediaMTX -> RTSP on this machine: an H.265
    publisher gives **21 undecodable frames of the first 120** (`Could not find ref
    with POC ...` / `Error constructing the frame RPS` - the reader joins mid-GOP),
    at 24.1 fps against 27.5 fps for H.264, which gives none. It is a *setting on the
    phone* (`docs/streaming.md`), so the app has to name it.
    """
    return str(codec or "").upper().startswith(("H265", "HEVC"))
