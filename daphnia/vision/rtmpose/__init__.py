"""RTMPose for DaphniaScope: the pose half of the top-down pipeline.

The **Localize → pose** cascade this package completes is the top-down arrangement
``mlContext.md`` §5 names from mmpose: RTMDet finds an animal, and this model places
two keypoints inside that box. It is a sibling of :mod:`daphnia.vision.rtmdet` and
follows the same five-file shape:

* :mod:`~daphnia.vision.rtmpose.model` - the architecture (a CSPNeXt backbone and a
  SimCC head);
* :mod:`~daphnia.vision.rtmpose.losses` - the maths: SimCC coding, KL loss, and the
  point metrics (error, PCK, OKS), as pure tensors;
* :mod:`~daphnia.vision.rtmpose.data` - a keypoint sample to a normalised tensor,
  with the box→input mapping owned by one object;
* :mod:`~daphnia.vision.rtmpose.train` - the run, the evaluation and the checkpoint;
* :mod:`~daphnia.vision.rtmpose.detect` - the inference object a call site uses.

The corpus is read by :mod:`daphnia.vision.pose_corpus`, which is Qt- and
torch-free, so the data contract is testable without the ML stack. PyTorch is
imported inside the functions that need it, exactly as in ``rtmdet``.

Typical use::

    from daphnia.vision.pose_corpus import PoseCorpus
    from daphnia.vision.rtmpose.train import TrainConfig, train

    train(PoseCorpus.read("datasets/kp"), TrainConfig(epochs=200))
"""

#: Presets :func:`~daphnia.vision.rtmpose.model.build_model` accepts.
PRESETS = ("tiny", "small")

#: Checkpoint format this package writes and reads. A reader refuses any other.
CHECKPOINT_FORMAT = "daphnia-rtmpose-1"

#: The default 1-D resolution each coordinate is classified at, as a multiple of the
#: input: SimCC's ``split_ratio``. 2.0 is mmpose's own default for RTMPose.
SIMCC_SPLIT_RATIO = 2.0

#: The default square the box is resized to before the network sees it.
DEFAULT_INPUT_SIZE = 192

__all__ = ["CHECKPOINT_FORMAT", "DEFAULT_INPUT_SIZE", "PRESETS",
           "SIMCC_SPLIT_RATIO", "torch_available", "torch_status"]


def torch_available() -> bool:
    """True when PyTorch can be imported - what a surface needs to know first."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def torch_status() -> str:
    """One line describing PyTorch and the accelerator, for a status label."""
    try:
        import torch
    except ImportError:
        return ("PyTorch is not installed - training and prediction are "
                "unavailable. See docs/rtmpose.md (make train-env).")
    device = "MPS (Apple GPU)" if torch.backends.mps.is_available() else "CPU"
    return f"PyTorch {torch.__version__}   device: {device}"
