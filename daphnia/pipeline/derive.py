"""Derive step: turn keypoints into biological measurements."""

from daphnia.vision.calibration import OpticalProfile
from daphnia.vision.keypoints import Keypoints


def body_length_um(keypoints: Keypoints, profile: OpticalProfile) -> float:
    """Body length in micrometres: pixel distance divided by the profile scale."""
    return profile.to_um(keypoints.length_px)
