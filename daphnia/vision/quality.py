"""Frame quality metrics used for burst selection (Well QA & Selection step)."""

import cv2


def sharpness(image) -> float:
    """Variance of the Laplacian — higher means sharper (more in focus).

    Cheap enough to score every frame of a burst inline.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def select_best(frames):
    """Return ``(index, score)`` of the sharpest frame, or ``(None, 0.0)``.

    This is the automated "auto-chosen best frame" of the burst review strip.
    """
    best_index, best_score = None, 0.0
    for index, frame in enumerate(frames or []):
        score = sharpness(frame)
        if best_index is None or score > best_score:
            best_index, best_score = index, score
    return best_index, best_score


def score_all(frames) -> list:
    """Return the sharpness of every frame, in order."""
    return [sharpness(frame) for frame in (frames or [])]
