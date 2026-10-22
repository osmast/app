"""Filesystem layout — the single source of truth for project directories.

Import these instead of building paths with ``__file__`` or ``os.getcwd()``.
"""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Tracked resources -------------------------------------------------------
ASSETS_DIR = PROJECT_ROOT / "assets"
UI_DIR = ASSETS_DIR / "ui"          # Qt Designer .ui files
THEME_DIR = ASSETS_DIR / "theme"    # Qt Style Sheets (QSS)

CONFIG_DIR = PROJECT_ROOT / "config"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

MEDIAMTX_CONFIG = CONFIG_DIR / "mediamtx.yml"

# Generated / local state -------------------------------------------------
#: Where the app **writes**: ``datasets/``, ``models/``, ``recordings/`` and ``zoo.db``.
#:
#: The default is the checkout itself, which is what a development run wants - the
#: pictures and the database sit beside the code that made them. A **packaged** app
#: (``.exe``, ``.app``/``.dmg``) must point this somewhere writable with
#: ``DAPHNIA_DATA_DIR``, for two reasons: its own folder may be read-only (Program
#: Files, or a mounted disk image) and - on macOS - writing *inside* a signed bundle
#: invalidates its signature. One variable moves the whole state, so a packaged app
#: never mixes the operator's data with its own code.
#:
#: The weights under ``models/`` follow this root too, because a training run writes
#: them; a packaged app therefore has to be given its checkpoints (copy ``*.pt`` into
#: ``<DAPHNIA_DATA_DIR>/models/``), and the Predict surface says so by name when one
#: is missing rather than starting with no model (`mlContext.md` law 8).
DATA_ROOT = Path(os.environ.get("DAPHNIA_DATA_DIR", PROJECT_ROOT)).expanduser()

DATASETS_DIR = DATA_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
ANNOTATIONS_DIR = DATASETS_DIR / "annotations"
MODELS_DIR = DATA_ROOT / "models"        # trained model weights
RECORDINGS_DIR = DATA_ROOT / "recordings"  # burst captures / segments
CALIBRATION_IMAGES_DIR = RECORDINGS_DIR / "calibration"  # micrometer reference shots

DB_PATH = DATA_ROOT / "zoo.db"
SCHEMA_PATH = PACKAGE_DIR / "db" / "schema.sql"

# Directories the application writes to at runtime.
RUNTIME_DIRS = (
    RAW_DIR,
    PROCESSED_DIR,
    ANNOTATIONS_DIR,
    MODELS_DIR,
    RECORDINGS_DIR,
    CALIBRATION_IMAGES_DIR,
)


def ensure_runtime_dirs() -> None:
    """Create the directories the app writes to (idempotent)."""
    for directory in RUNTIME_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def study_dir(study_id: str) -> Path:
    """Root folder that holds every image of a study."""
    return RECORDINGS_DIR / study_id


def observation_dir(study_id: str, session_id: str, observation_id: str) -> Path:
    """Directory that holds the burst frames of one observation."""
    return study_dir(study_id) / session_id / observation_id


def burst_dir(name: str) -> Path:
    """Folder that holds the frames of one **burst** captured on the Bench.

    A burst is written into a folder of its own under ``datasets/raw`` - a real folder in
    the dataset tree - so the very same folder can be brought back later with *Import a
    folder...* and gives the same queue and the same per-image views (`SR-BENCH-023`).
    """
    return RAW_DIR / str(name)
