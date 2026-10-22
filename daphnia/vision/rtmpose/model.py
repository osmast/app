"""RTMPose in plain PyTorch: a CSPNeXt backbone and a SimCC head.

## What this is, honestly

The top-down pose model of RTMPose (arXiv:2303.07399), with its defining ideas intact,
written against bare PyTorch so it trains on the laptop that takes the pictures:

* a **CSPNeXt backbone** - the *same* module RTMDet uses
  (:class:`daphnia.vision.rtmdet.model.CSPNeXt`), because a second copy of a backbone
  is a second place for the stride contract to drift;
* a **SimCC head** (:class:`SimCCHead`) that predicts each coordinate as a 1-D
  distribution over ``input_size * split_ratio`` bins (arXiv:2107.03332).

It is *not* a bit-exact port of ``mmpose``'s head. The head collapses each axis with
mean pooling and reads the bins off a linear layer; the exact RTMPose head width and
stack depth are documented choices here, not copied constants, and
``docs/rtmpose.md`` §5 lists every difference so a score from this trainer is never
confused with a score from the reference implementation.

## Why the input size lives in the model

The SimCC head's classifiers have ``feat_channels * feature_side`` inputs, so the
network's input size is part of its *shape*, not a runtime option. It is stored on the
model and written into the checkpoint, and inference reads it back (``mlContext.md``
law 3): feeding the network a size it did not train at would be a shape error, not a
silently worse prediction.
"""

import torch
import torch.nn as nn

from daphnia.vision.rtmdet.model import CSPNeXt, ConvBNAct
from daphnia.vision.rtmpose import DEFAULT_INPUT_SIZE, SIMCC_SPLIT_RATIO
from daphnia.vision.rtmpose.losses import simcc_decode, simcc_encode, simcc_loss

#: The stride of the backbone feature the SimCC head reads. CSPNeXt's stem is stride
#: 4 and its first stage another 2, so ``features[0]`` (C3) is stride 8.
FEATURE_STRIDE = 8

#: The presets this trainer ships. ``tiny`` fits an M4 laptop and 384 samples;
#: ``small`` is for a corpus that justifies it.
CONFIGS = {
    "tiny": {
        "stem_channels": 16,
        "stage_channels": (32, 64, 128),
        "stage_blocks": (1, 1, 1),
        "stage_expand": 0.5,
        "channel_attention": True,
        "head_channels": 64,
        "head_convs": 2,
    },
    "small": {
        "stem_channels": 32,
        "stage_channels": (64, 128, 256),
        "stage_blocks": (2, 2, 2),
        "stage_expand": 0.5,
        "channel_attention": True,
        "head_channels": 128,
        "head_convs": 2,
    },
}


class SimCCHead(nn.Module):
    """Predicts the x and y of each keypoint as a 1-D distribution (SimCC).

    ``forward`` returns ``(x_logits, y_logits)`` of shapes ``(B, K, nx)`` and
    ``(B, K, ny)`` - raw scores, not probabilities, so the loss can log-softmax them
    itself. The two axes are read off the **same** feature: the x classifier sees the
    feature collapsed over height, the y classifier the feature collapsed over width.
    That is SimCC's economy - one feature map, two coordinate problems - and it is why
    the bins are as fine as they are without a heavier backbone.
    """

    def __init__(self, in_channels, num_keypoints, input_size=DEFAULT_INPUT_SIZE,
                 split_ratio=SIMCC_SPLIT_RATIO, feat_channels=64, convs=2):
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.input_size = (int(input_size[0]), int(input_size[1])) \
            if isinstance(input_size, (tuple, list)) else (int(input_size),
                                                           int(input_size))
        self.split_ratio = float(split_ratio)
        self.feature_size = (self.input_size[0] // FEATURE_STRIDE,
                             self.input_size[1] // FEATURE_STRIDE)
        self.x_bins = int(round(self.input_size[0] * self.split_ratio)) + 1
        self.y_bins = int(round(self.input_size[1] * self.split_ratio)) + 1

        self.stem = nn.Sequential(*[
            ConvBNAct(in_channels if index == 0 else feat_channels, feat_channels,
                      3, padding=1) for index in range(max(1, int(convs)))])
        # The x classifier: over the width, with the height pooled away.
        self.linear_x = nn.Linear(feat_channels * self.feature_size[0],
                                  self.num_keypoints * self.x_bins)
        self.linear_y = nn.Linear(feat_channels * self.feature_size[1],
                                  self.num_keypoints * self.y_bins)

    def forward(self, feature):
        """``(B, C, H', W')`` -> ``((B, K, nx), (B, K, ny))``."""
        batch = feature.shape[0]
        if tuple(feature.shape[-2:]) != (self.feature_size[1], self.feature_size[0]):
            raise ValueError(
                f"the SimCC head was built for a "
                f"{self.input_size[0]}x{self.input_size[1]} input "
                f"(feature {self.feature_size[1]}x{self.feature_size[0]}) but saw a "
                f"feature of {feature.shape[-2]}x{feature.shape[-1]}. Feed the size "
                "the checkpoint trained at (mlContext.md law 3).")
        x = self.stem(feature)
        collapsed_x = x.mean(dim=2).reshape(batch, -1)      # pool height -> width
        collapsed_y = x.mean(dim=3).reshape(batch, -1)      # pool width -> height
        x_logits = self.linear_x(collapsed_x).reshape(
            batch, self.num_keypoints, self.x_bins)
        y_logits = self.linear_y(collapsed_y).reshape(
            batch, self.num_keypoints, self.y_bins)
        return x_logits, y_logits


class RTMPose(nn.Module):
    """The pose model: a CSPNeXt backbone and a SimCC head, with loss and inference.

    One object carries everything a checkpoint must describe and a call site needs:
    :meth:`forward` returns the two coordinate distributions, :meth:`predict` decodes
    them to points in **input pixels**, and :meth:`loss` trains from a batch of
    ``{"keypoints": (K, 2)}`` dictionaries in the same frame.
    """

    def __init__(self, num_keypoints, preset="small", input_size=DEFAULT_INPUT_SIZE,
                 split_ratio=SIMCC_SPLIT_RATIO, keypoint_names=()):
        super().__init__()
        if preset not in CONFIGS:
            raise ValueError(f"Unknown preset '{preset}'. Known: "
                             + ", ".join(sorted(CONFIGS)))
        if int(input_size) % FEATURE_STRIDE:
            raise ValueError(
                f"input_size must be a multiple of {FEATURE_STRIDE} (the head's "
                f"stride); got {input_size}.")
        config = CONFIGS[preset]
        self.preset = preset
        self.num_keypoints = int(num_keypoints)
        self.input_size = int(input_size)
        self.split_ratio = float(split_ratio)
        self.keypoint_names = tuple(keypoint_names) or tuple(
            f"k{index}" for index in range(self.num_keypoints))
        self.backbone = CSPNeXt(
            stem_channels=config["stem_channels"],
            stage_channels=config["stage_channels"],
            stage_blocks=config["stage_blocks"],
            stage_expand=config["stage_expand"],
            channel_attention=config["channel_attention"])
        self.head = SimCCHead(
            self.backbone.out_channels[0], self.num_keypoints, self.input_size,
            self.split_ratio, feat_channels=config["head_channels"],
            convs=config["head_convs"])
        self.init_weights()

    # -- weights -----------------------------------------------------------
    def init_weights(self):
        """Kaiming-normal convolutions, zero biases, unit BatchNorms - the default."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def parameter_count(self) -> int:
        """How many trainable parameters this preset has."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- forward -----------------------------------------------------------
    def forward(self, images):
        """Return ``(x_logits, y_logits)`` for a ``(B, 3, H, W)`` batch.

        The input size is checked *here*, before the backbone, so feeding a size the
        model did not train at fails with a message that names the reason rather than
        a BatchNorm shape error deep inside a stage (``mlContext.md`` law 3).
        """
        if images.shape[-1] != self.input_size or images.shape[-2] != self.input_size:
            raise ValueError(
                f"this model was built for a {self.input_size} x {self.input_size} "
                f"input but received {images.shape[-2]} x {images.shape[-1]}. Feed the "
                "size the checkpoint recorded - inference at a different size changes "
                "the scale statistics and the compute (mlContext.md law 3).")
        features = self.backbone(images)
        return self.head(features[0])

    @torch.no_grad()
    def predict(self, images):
        """``(B, K, 2)`` keypoints in **input pixels** for a normalised batch.

        The caller owns the mapping back to source pixels
        (:class:`daphnia.vision.rtmpose.data.PoseInput`), so no coordinate ever
        leaves this method in an ambiguous frame.
        """
        x_logits, y_logits = self(images)
        return simcc_decode(x_logits, y_logits, self.split_ratio)

    # -- loss --------------------------------------------------------------
    def loss(self, images, targets):
        """SimCC loss over one batch.

        Args:
            images: ``(B, 3, H, W)`` already normalised by the data pipeline.
            targets: one dict per image with ``"keypoints"`` ``(K, 2)`` in the
                **input** frame, and an optional ``"visible"`` mask of length ``K``.
                A keypoint a sample does not carry keeps an all-zero target
                distribution, which the KL divergence ignores - so a partly-labelled
                animal trains what it has.

        Returns:
            ``{"loss", "loss_x", "loss_y"}`` - the first differentiably, the axes
            detached so a run can see which coordinate lags.
        """
        import numpy as np

        x_logits, y_logits = self(images)
        batch = x_logits.shape[0]
        x_target = np.zeros((batch, self.num_keypoints, self.head.x_bins), np.float32)
        y_target = np.zeros((batch, self.num_keypoints, self.head.y_bins), np.float32)
        for index, target in enumerate(targets):
            points = target["keypoints"]
            points = points.detach().cpu().numpy() if hasattr(points, "detach") \
                else np.asarray(points, np.float32)
            visible = target.get("visible")
            for keypoint in range(self.num_keypoints):
                if visible is not None and not float(visible[keypoint]):
                    continue                    # a missing point keeps a zero target
                encoded_x, encoded_y = simcc_encode(
                    points[keypoint:keypoint + 1], self.head.input_size,
                    self.split_ratio)
                x_target[index, keypoint] = encoded_x[0]
                y_target[index, keypoint] = encoded_y[0]
        total, loss_x, loss_y = simcc_loss(x_logits, y_logits, x_target, y_target)
        return {"loss": total, "loss_x": loss_x, "loss_y": loss_y}


def build_model(preset="tiny", num_keypoints=2, input_size=DEFAULT_INPUT_SIZE,
                split_ratio=SIMCC_SPLIT_RATIO, keypoint_names=()):
    """Build a pose model for one preset name (``tiny`` or ``small``)."""
    return RTMPose(num_keypoints=num_keypoints, preset=preset,
                   input_size=input_size, split_ratio=split_ratio,
                   keypoint_names=keypoint_names)
