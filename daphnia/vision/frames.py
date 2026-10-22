"""Frame normalisation: the one place that decides a frame's orientation.

Applied where frames enter the app, so the live preview, the burst written to
disk and the clicked keypoints all use the same coordinate space.
"""

import numpy as np


def to_landscape(frame):
    """Rotate a portrait frame by 90 degrees so the display is always landscape."""
    if frame is not None and frame.ndim >= 2 and frame.shape[0] > frame.shape[1]:
        return np.ascontiguousarray(np.rot90(frame, k=-1))
    return frame
