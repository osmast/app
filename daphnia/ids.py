"""Identifier factory for database rows.

IDs follow the convention already used in the schema — a type prefix plus a
timestamp and a short random suffix, e.g. ``OBS_20261015_143012_a1b2c3``.
Human-chosen identifiers (like ``PROF_LEICA_4X_IPHONE15_1X``) are passed in
verbatim instead of being generated.
"""

from datetime import datetime
from uuid import uuid4


def new_id(prefix: str) -> str:
    """Return ``<PREFIX>_<YYYYmmdd_HHMMSS>_<6 hex chars>``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid4().hex[:6]}"


def slug_key(value: str) -> str:
    """Upper-case alphanumerics only, so a key never holds spaces or punctuation."""
    cleaned = "".join(ch if ch.isalnum() else "" for ch in (value or "")).upper()
    return cleaned or "UNKNOWN"


def new_profile_id(microscope: str, objective: str, camera: str,
                   zoom_ratio: float = 1.0, focus: str = "infinity") -> str:
    """Readable preset id, e.g. ``PROF_SCOPEA_2X_IPHONE15_1.0X_INFINITY``.

    The locked zoom ratio and focus mode are part of the identity: decoupled
    profile calibration is only valid while both stay unchanged.

    This is an **optical setup** key: it names the hardware chain, so every
    calibration taken on that chain shares it (see :func:`new_calibration_id`).
    """
    return (f"PROF_{slug_key(microscope)}_{slug_key(objective)}_{slug_key(camera)}"
            f"_{zoom_ratio}X_{slug_key(focus)}")


def new_setup_key(preset_name: str) -> str:
    """Optical-chain key from the name the operator gave a preset: ``PROF_<NAME>``.

    The **name is the identity**: one preset name is one optical chain, so every
    calibration saved under that name shares one scale history and the history is
    readable (``PROF_LEICA2X_IPHONE``) without the operator describing the hardware
    twice. A scale is only valid for the chain it was measured on, so renaming a
    preset starts a new chain - which is the honest reading of a rename: nothing
    recorded can say the two names were the same optics.
    """
    return f"PROF_{slug_key(preset_name)}"


def new_calibration_id(setup_key: str) -> str:
    """One calibration of an optical setup, e.g. ``PROF_SCOPE_2X_..._20261015_201500``.

    A calibration is immutable: every re-calibration gets its own id, so the
    measurements taken under an earlier scale keep pointing at the record that
    actually produced them instead of at a since-changed value.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{setup_key}_{stamp}"
