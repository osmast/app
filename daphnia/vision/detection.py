"""Localize step: isolate the specimen in a well.

The **Predict** surface (`python run.py predict`) is this step, made visible: it runs
the trained RTMDet on a live stream or a video file and draws the boxes. This module
is the programmatic half of the same thing, for the pipeline and for scripts.

Two targeting modes remain the acquisition-time plan:

* Mode A (96-well): in-frame well circle detector producing a bounding region and
  an automated scale read-out.
* Mode B (6-/24-well): centre-target crosshair crop driven by the active
  calibrated optical profile.

Both are now answerable by the **same** trained detector rather than by a separate
circle fitter: :func:`localize_well` returns one region per animal, which is what the
Predict step needs and what the measurement pipeline will consume.
"""


def localize_well(image, profile=None, detector=None, path=None, **kwargs):
    """Return the specimen regions of a well image.

    Args:
        image: BGR ``numpy.ndarray`` of one well.
        profile: active :class:`~daphnia.vision.calibration.OpticalProfile`
            (accepted for the pipeline's signature; the detector is scale-free).
        detector: an already loaded
            :class:`~daphnia.vision.rtmdet.detect.Detector`; loaded on first use
            otherwise, so a loop over frames does not reload the weights.
        path: an explicit checkpoint, overriding ``models/rtmdet_daphnia.pt``.

    Returns:
        A list of :class:`~daphnia.vision.rtmdet.detect.Detection` - one per animal,
        in **original image pixels**.

    Raises:
        NotImplementedError: no trained detector is available yet. The message names
            the command that produces one, because "Localize is not implemented"
            stops being true the moment a model exists.
        ModelUnavailable: a checkpoint exists but cannot be used (or PyTorch is
            missing), with the reason.
    """
    from daphnia.vision.rtmdet.detect import (
        ModelUnavailable,
        detect_frame,
    )

    try:
        return detect_frame(image, detector=detector, path=path)
    except ModelUnavailable as error:
        raise NotImplementedError(
            "Localization needs the trained RTMDet detector, and there is none "
            "usable here:\n"
            f"  {error}") from error

