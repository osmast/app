"""Predict step: anatomical keypoints and raw pixel length.

Body length is defined as the distance from the anterior margin of the compound
eye to the base of the apical spine. The model is not trained yet, so prediction
is a placeholder while the annotation dataset is being built.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Keypoints:
    """Eye apex and apical-spine base, in pixel coordinates of the source image."""

    eye_apex: tuple
    spine_base: tuple

    @property
    def length_px(self) -> float:
        """Euclidean distance between the two keypoints, in pixels."""
        (x1, y1), (x2, y2) = self.eye_apex, self.spine_base
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def predict_keypoints(image, profile=None, **kwargs) -> Keypoints:
    """Estimate keypoints for a localized specimen.

    TODO: load the trained weights from ``models/`` and run inference (PyTorch).

    Raises:
        NotImplementedError: always, until a model is available.
    """
    raise NotImplementedError("Keypoint prediction is not implemented yet (no model trained).")
