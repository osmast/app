"""RTMDet for DaphniaScope: the Localize step, trained from a Dataset export.

This package is the **model side** of the application: RTMDet's architecture
(:mod:`~daphnia.vision.rtmdet.model`), its SimOTA assignment and losses
(:mod:`~daphnia.vision.rtmdet.losses`), the data pipeline
(:mod:`~daphnia.vision.rtmdet.data`), the training run
(:mod:`~daphnia.vision.rtmdet.train`) and inference
(:mod:`~daphnia.vision.rtmdet.detect`).

It sits in the **domain layer** and imports no Qt, so a training run, a notebook
and the Predict surface all use the same code. **PyTorch is imported inside the
functions that need it**, not at module import, so:

* the Qt application starts and every other surface works on a machine without
  PyTorch;
* ``make test`` runs the parts of the suite that need no model;
* the Predict surface can say "train a model first" instead of failing to launch.

Typical use::

    from daphnia.vision.export import Export
    from daphnia.vision.rtmdet.train import TrainConfig, train

    train(Export.read("datasets/bb"), TrainConfig(epochs=100))
"""

#: Presets that :func:`~daphnia.vision.rtmdet.model.build_model` accepts.
PRESETS = ("tiny", "small")

#: The class names of this project's detector (one organism, one box).
CLASSES = ("daphnia",)

#: Checkpoint format this package writes and reads.
CHECKPOINT_FORMAT = "daphnia-rtmdet-1"

__all__ = ["CHECKPOINT_FORMAT", "CLASSES", "PRESETS", "torch_available",
           "torch_status"]


def torch_available() -> bool:
    """True when PyTorch can be imported - what a surface needs to know first."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def torch_status() -> str:
    """One line describing PyTorch and the accelerator, for a status label.

    Says which device a run *would* use, because "MPS available" is the difference
    between a training run taking minutes and taking an afternoon.
    """
    try:
        import torch
    except ImportError:
        return ("PyTorch is not installed - training and prediction are "
                "unavailable. See docs/rtmdet.md (make train-env).")
    device = "MPS (Apple GPU)" if torch.backends.mps.is_available() else "CPU"
    return f"PyTorch {torch.__version__}   device: {device}"
