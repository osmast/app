"""RTMDet in plain PyTorch: CSPNeXt backbone, CSPNeXtPAFPN neck, SepBN head.

## What this is, honestly

RTMDet's architecture with its defining ideas intact, written against bare
PyTorch so it can be trained on a laptop:

* **CSPNeXt** backbone - a two-conv *deep stem* (stride 4), then three CSP stages
  that produce C3/C4/C5 at strides 8/16/32. Every block inside a stage is a
  **depthwise-separable** 3x3 pair with a residual, and each stage can add a
  :class:`ChannelAttention`.
* **CSPNeXtPAFPN** neck - the CSP fusion of a top-down and a bottom-up path, using
  5x5 depthwise convolutions, which is what makes RTMDet's neck cheap per level.
* **SepBN head** - one set of convolutions **shared across all three levels**, with
  a *separate* BatchNorm per level. Sharing is why RTMDet needs no per-level head
  tuning; the separate BNs are what let the levels stay statistically different.
* **One anchor per location** with side ``stride``, and a prediction of the four
  distances to the box sides (:func:`~daphnia.vision.rtmdet.losses.distance2bbox`).
* **SimOTA** assignment and **QualityFocalLoss / GIoULoss**, in
  :mod:`daphnia.vision.rtmdet.losses`.

It is *not* a bit-exact port of ``mmdet``'s module tree: block internals, exact
channel multipliers and the stem depth are documented choices here rather than
copied numbers. What is faithful is everything that makes RTMDet RTMDet - the
backbone family, the shared head, one point per location, the four-distance
coding, SimOTA and the quality focal loss - and
``docs/rtmdet.md`` records the differences so a score from this trainer is never
confused with a score from the reference implementation.

## Why bare PyTorch

``mmcv``/``mmdet`` do not install cleanly on Apple-silicon macOS, and the thesis
must be reproducible on the machine that takes the pictures. The whole detector is
therefore ~500 lines of readable PyTorch with three dependencies: ``torch``,
``numpy`` and OpenCV (for reading frames).
"""

import math

import torch
import torch.nn as nn

from daphnia.vision.rtmdet.losses import (
    GIoULoss,
    QualityFocalLoss,
    SimOTAAssigner,
    batched_nms,
    distance2bbox,
)

#: Image size (a multiple of 32) each preset trains at.
DEFAULT_INPUT_SIZE = 640

#: The presets this trainer ships. ``tiny`` fits an M4 laptop with 8 GB of unified
#: memory and 500-odd boxes; ``small`` is for a corpus that justifies it.
CONFIGS = {
    "tiny": {
        "stem_channels": 16,
        "stage_channels": (32, 64, 128),      # C3 (stride 8), C4 (16), C5 (32)
        "stage_blocks": (1, 1, 1),
        "stage_expand": 0.5,
        "channel_attention": True,
        "neck_channels": 64,
        "neck_blocks": 2,
        "feat_channels": 64,
        "stacked_convs": 2,
    },
    "small": {
        "stem_channels": 32,
        "stage_channels": (64, 128, 256),
        "stage_blocks": (2, 2, 2),
        "stage_expand": 0.5,
        "channel_attention": True,
        "neck_channels": 128,
        "neck_blocks": 2,
        "feat_channels": 128,
        "stacked_convs": 2,
    },
}

#: The strides the head predicts at, in the order the neck returns its levels.
STRIDES = (8, 16, 32)


class ConvBNAct(nn.Sequential):
    """``conv -> BatchNorm -> SiLU``, the one building block of RTMDet.

    A ``Sequential`` subclass rather than a hand-written ``forward`` so the module
    prints and traces like the stack of layers it is.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True))


class ChannelAttention(nn.Module):
    """RTMDet's cheap channel attention: shared 1x1 over average *and* max pooling.

    Average pooling alone smooths a small, low-contrast animal away; the max path
    keeps the one location that actually looks like a body. One shared projection
    for both paths is what keeps this at a few thousand parameters.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True))
        self.average = nn.Conv2d(hidden, channels, 1)
        self.maximum = nn.Conv2d(hidden, channels, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        pooled = self.pool(x)
        hidden = self.shared(pooled)
        weights = self.activation(self.average(hidden) + self.maximum(hidden))
        return x * weights


class CSPNeXtBlock(nn.Module):
    """One **depthwise-separable** residual block, the unit of a CSPNeXt stage.

    A 3x3 depthwise convolution mixes space without mixing channels, a 1x1 point
    convolution then mixes channels - so a block costs a fraction of a dense 3x3
    pair at the same width. That is what lets RTMDet keep a real backbone on a
    small compute budget, and it is the single most important detail of the family.
    """

    def __init__(self, channels, expand=1.0):
        super().__init__()
        hidden = max(1, int(round(channels * expand)))
        self.depthwise = ConvBNAct(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = ConvBNAct(channels, hidden, 1)
        self.project = ConvBNAct(hidden, channels, 1) if hidden != channels \
            else nn.Identity()

    def forward(self, x):
        return x + self.project(self.pointwise(self.depthwise(x)))


class CSPNeXtStage(nn.Module):
    """A CSP stage: split into two 1x1 paths, transform one, concatenate, merge.

    ``stride=2`` downsampling happens in the *main* path, so the shortcut carries
    the full-resolution signal and the stage can still be residual.
    """

    def __init__(self, in_channels, out_channels, blocks=1, expand=0.5,
                 stride=1, channel_attention=False):
        super().__init__()
        mid = max(1, int(round(out_channels * expand)))
        self.main_conv = ConvBNAct(in_channels, mid, 1)
        self.short_conv = ConvBNAct(in_channels, mid, 1)
        self.blocks = nn.Sequential(
            *[CSPNeXtBlock(mid) for _ in range(max(1, int(blocks)))])
        self.final_conv = ConvBNAct(2 * mid, out_channels, 1)
        self.downsample = (ConvBNAct(mid, mid, 3, stride=2, padding=1)
                           if stride > 1 else nn.Identity())
        self.attention = ChannelAttention(out_channels) if channel_attention \
            else nn.Identity()

    def forward(self, x):
        main = self.downsample(self.blocks(self.main_conv(x)))
        short = self.short_conv(x)
        if short.shape[-2:] != main.shape[-2:]:
            # The shortcut follows the main path down; average pooling (not a conv)
            # keeps this from adding parameters that do nothing but resize.
            short = nn.functional.adaptive_avg_pool2d(short, main.shape[-2:])
        return self.attention(self.final_conv(torch.cat([main, short], dim=1)))


class CSPNeXt(nn.Module):
    """The backbone: a deep stride-4 stem, then C3/C4/C5 at strides 8/16/32.

    The stem is two stride-2 convolutions, so the *first* stage already sees a
    field wide enough for a whole animal instead of a blob of pixels - the reason
    RTMDet can put a heavy stem in front of a comparatively light P3.
    """

    def __init__(self, in_channels=3, stem_channels=16, stage_channels=(32, 64, 128),
                 stage_blocks=(1, 1, 1), stage_expand=0.5,
                 channel_attention=True):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, stem_channels, 3, stride=2, padding=1),
            ConvBNAct(stem_channels, stem_channels, 3, stride=2, padding=1))
        stages, previous = [], stem_channels
        for index, channels in enumerate(stage_channels):
            stages.append(CSPNeXtStage(
                previous, channels,
                blocks=stage_blocks[index] if index < len(stage_blocks) else 1,
                expand=stage_expand,
                stride=2,                       # 4 -> 8 -> 16 -> 32
                channel_attention=channel_attention and index > 0))
            previous = channels
        self.stages = nn.ModuleList(stages)
        self.out_channels = tuple(stage_channels)

    def forward(self, x):
        """Return ``[C3, C4, C5]``, the feature maps the neck fuses.

        The stages are chained, each one taking the previous stage's output: a list
        comprehension over ``self.stages`` would feed every stage the *stem's*
        output, which silently trains a backbone that never sees its own pyramid.
        """
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class NeckBlock(nn.Module):
    """One CSPNeXt neck block: 1x1 squeeze, **5x5 depthwise**, 1x1 expand.

    The 5x5 depthwise convolution is the neck's signature: a large enough kernel
    to fuse a whole animal across one level's receptive field, at the cost of nine
    parameters per channel instead of twenty-five.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.squeeze = ConvBNAct(in_channels, out_channels, 1)
        self.spatial = ConvBNAct(out_channels, out_channels, 5, padding=2,
                                 groups=out_channels)
        self.expand = ConvBNAct(out_channels, out_channels, 1)

    def forward(self, x):
        return self.expand(self.spatial(self.squeeze(x)))


class CSPNeXtPAFPN(nn.Module):
    """The neck: a top-down path and a bottom-up path, each fuse step a CSP block.

    Down-sampling in the bottom-up path is a stride-2 3x3 convolution rather than
    a pool, so the path *learns* what to keep when it goes back down - which is
    what makes the P3-with-heavy-context / P5-with-location pair that RTMDet's
    shared head depends on.
    """

    def __init__(self, in_channels, out_channels, blocks=2):
        super().__init__()
        self.reduce = nn.ModuleList(
            [ConvBNAct(channels, out_channels, 1) for channels in in_channels])
        self.top_down = nn.ModuleList(
            [NeckBlock(2 * out_channels, out_channels) for _ in range(2)])
        self.downsample = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, 3, stride=2, padding=1)
             for _ in range(2)])
        self.bottom_up = nn.ModuleList(
            [NeckBlock(2 * out_channels, out_channels) for _ in range(2)])
        self._blocks = max(1, int(blocks))

    def forward(self, features):
        """Fuse ``[C3, C4, C5]`` into ``[P3, P4, P5]`` at the neck's width."""
        c3, c4, c5 = [reduce(level) for reduce, level in zip(self.reduce, features)]

        p5 = c5
        p4 = self.top_down[0](torch.cat(
            [c4, nn.functional.interpolate(p5, size=c4.shape[-2:],
                                           mode="nearest")], dim=1))
        p3 = self.top_down[1](torch.cat(
            [c3, nn.functional.interpolate(p4, size=c3.shape[-2:],
                                           mode="nearest")], dim=1))

        n4 = self.bottom_up[0](torch.cat([p4, self.downsample[0](p3)], dim=1))
        n5 = self.bottom_up[1](torch.cat([p5, self.downsample[1](n4)], dim=1))
        return [p3, n4, n5]


class RTMDetSepBNHead(nn.Module):
    """The shared head: **one** set of convolutions, **one BatchNorm per level**.

    Sharing the convolutions across P3/P4/P5 is what removes RTMDet's per-level
    head tuning; the separate BatchNorms are what stop the levels - a 20-pixel
    animal and a 300-pixel one - from being forced into one distribution. Two 1x1
    convolutions then read off the class scores and the four distances.
    """

    def __init__(self, in_channels, num_classes, feat_channels=64,
                 stacked_convs=2, num_levels=3):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_levels = int(num_levels)
        self.shared = nn.ModuleList(
            [nn.Conv2d(in_channels, feat_channels, 3, padding=1, bias=False)
             for _ in range(max(1, int(stacked_convs)))])
        self.bns = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(feat_channels)
                            for _ in range(self.num_levels)])
             for _ in self.shared])
        self.cls = nn.Conv2d(feat_channels, self.num_classes, 1)
        self.reg = nn.Conv2d(feat_channels, 4, 1)
        self._init_weights()

    def _init_weights(self):
        """Start the class scores near 0.01, as RTMDet does.

        An untrained detector must begin by saying "background" everywhere, or the
        first few hundred steps are spent undoing a confident nonsense.
        """
        prior_probability = 0.01
        bias = -math.log((1 - prior_probability) / prior_probability)
        nn.init.constant_(self.cls.bias, bias)

    def forward(self, features):
        """Return ``(cls_scores, bbox_preds)``, each one tensor per level."""
        cls_scores, bbox_preds = [], []
        for level, feature in enumerate(features):
            x = feature
            for index, conv in enumerate(self.shared):
                x = nn.functional.silu(self.bns[index][level](conv(x)))
            cls_scores.append(self.cls(x))
            bbox_preds.append(self.reg(x))
        return cls_scores, bbox_preds


class RTMDet(nn.Module):
    """The detector: backbone, neck and shared head, plus its loss and inference.

    One object carries all three concerns so a checkpoint is self-describing and
    the Predict surface needs nothing else: :meth:`forward` returns features,
    :meth:`loss` trains from a batch of ``{image, boxes, labels}`` dictionaries, and
    :meth:`detect` turns a batch of frames into boxes in **image pixels**.
    """

    def __init__(self, num_classes=1, preset="tiny", strides=STRIDES):
        super().__init__()
        if preset not in CONFIGS:
            raise ValueError(f"Unknown preset '{preset}'. Known: "
                             + ", ".join(sorted(CONFIGS)))
        config = CONFIGS[preset]
        self.preset = preset
        self.num_classes = int(num_classes)
        self.strides = tuple(int(stride) for stride in strides)
        self.backbone = CSPNeXt(
            stem_channels=config["stem_channels"],
            stage_channels=config["stage_channels"],
            stage_blocks=config["stage_blocks"],
            stage_expand=config["stage_expand"],
            channel_attention=config["channel_attention"])
        self.neck = CSPNeXtPAFPN(
            in_channels=self.backbone.out_channels,
            out_channels=config["neck_channels"],
            blocks=config["neck_blocks"])
        self.head = RTMDetSepBNHead(
            config["neck_channels"], self.num_classes,
            feat_channels=config["feat_channels"],
            stacked_convs=config["stacked_convs"],
            num_levels=len(self.strides))
        self.assigner = SimOTAAssigner(num_classes=self.num_classes)
        self.loss_cls = QualityFocalLoss()
        self.loss_box = GIoULoss()
        self.init_weights()

    # -- weights -----------------------------------------------------------
    def init_weights(self):
        """Kaiming-normal convolutions and zero biases, the training-stable default."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.head._init_weights()       # keep the 0.01 prior on the class scores

    def parameter_count(self) -> int:
        """How many trainable parameters this preset has."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- priors ------------------------------------------------------------
    def level_priors(self, height, width, stride, device):
        """``(height * width, 4)`` priors for one level: one **centre** per location.

        A prior is ``(x, y, stride, stride)`` in image pixels - the point the four
        distances are measured from, plus the level's stride as the box's nominal
        size. There is exactly one, which is why RTMDet has no anchor set to tune.
        """
        ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        side = torch.full_like(grid_x.reshape(-1), float(stride))
        return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), side, side],
                           dim=-1)

    def priors_for(self, shapes, device):
        """The prior set of one image, levels concatenated in head order."""
        return torch.cat([self.level_priors(height, width, stride, device)
                          for (height, width), stride
                          in zip(shapes, self.strides)])

    # -- forward -----------------------------------------------------------
    def forward(self, images):
        """Return ``(features, (cls_scores, bbox_preds))``."""
        features = self.neck(self.backbone(images))
        return features, self.head(features)

    def flatten_predictions(self, cls_scores, bbox_preds):
        """Flatten the three levels into per-image ``(B, N, ...)`` tensors.

        The head predicts per level; the assignment is a *global* competition for
        ground truths, so the levels must be one flat set of locations first - which
        is exactly what ``batch_priors`` is in the reference implementation.
        """
        batch = cls_scores[0].size(0)
        shapes = [tuple(tensor.shape[-2:]) for tensor in cls_scores]
        priors = self.priors_for(shapes, cls_scores[0].device)
        strides = torch.cat([
            torch.full((height * width,), float(stride),
                       device=priors.device, dtype=priors.dtype)
            for (height, width), stride in zip(shapes, self.strides)])
        cls = torch.cat([tensor.permute(0, 2, 3, 1).reshape(
            batch, -1, self.num_classes) for tensor in cls_scores], dim=1)
        reg = torch.cat([tensor.permute(0, 2, 3, 1).reshape(batch, -1, 4)
                         for tensor in bbox_preds], dim=1)
        return cls, reg, priors, strides

    @staticmethod
    def decode(reg, priors, strides):
        """Decode ``(B, N, 4)`` raw distances into ``(B, N, 4)`` boxes in pixels."""
        return distance2bbox(priors[:, :2][None], reg * strides[None, :, None])

    # -- loss --------------------------------------------------------------
    def loss(self, images, targets):
        """RTMDet's total loss over one batch.

        Args:
            images: ``(B, 3, H, W)`` already normalised by the data pipeline.
            targets: one dict per image with ``boxes`` ``(M, 4)`` in the **padded
                input** frame and ``labels`` ``(M,)``; an image with no animal
                passes an empty ``boxes`` and contributes classification only -
                which is what teaches the detector about empty wells.

        Returns:
            ``{"loss", "loss_cls", "loss_box", "matched"}`` - the first three are
            tensors (the first differentiably), ``matched`` counts the positives
            the assignment chose, the number that tells a silent failure ("nothing
            was assigned") from slow learning.
        """
        _, (cls_scores, bbox_preds) = self(images)
        cls, reg, priors, strides = self.flatten_predictions(cls_scores, bbox_preds)
        boxes = self.decode(reg, priors, strides)

        zero = cls.sum() * 0.0
        cls_losses, box_losses, matched_total = [], [], 0
        for index, target in enumerate(targets):
            gt_boxes = target["boxes"].to(cls.device, cls.dtype).reshape(-1, 4)
            gt_labels = target["labels"].to(cls.device).reshape(-1).long()
            cls_target, cls_weight, box_target, box_weight, matched = \
                self.assigner.assign(priors, cls[index], boxes[index],
                                     gt_boxes, gt_labels)
            matched_total += matched
            cls_losses.append(self.loss_cls(cls[index], cls_target, cls_weight))
            box_losses.append(self.loss_box(boxes[index], box_target, box_weight))

        cls_loss = (torch.stack(cls_losses).mean() if cls_losses else zero)
        box_loss = (torch.stack(box_losses).mean() if box_losses else zero)
        return {"loss": cls_loss + box_loss,
                "loss_cls": cls_loss.detach(),
                "loss_box": box_loss.detach(),
                "matched": matched_total}

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def detect(self, images, score_threshold=0.30, iou_threshold=0.60,
               max_detections=300):
        """Boxes for a batch of frames: ``[(boxes, scores, labels), ...]``.

        Coordinates are in the **padded input** frame the caller built, so the
        ``Detector`` maps them back to the original image before drawing anything.
        Boxes are clipped to the pad rectangle first, so NMS never spends a slot on
        a prediction that reaches into the letterbox.
        """
        _, (cls_scores, bbox_preds) = self(images)
        cls, reg, priors, strides = self.flatten_predictions(cls_scores, bbox_preds)
        decoded = self.decode(reg, priors, strides)
        height, width = images.shape[-2:]
        scores_all = cls.sigmoid()

        results = []
        for index in range(images.size(0)):
            scores, labels = scores_all[index].max(dim=-1)
            candidates = decoded[index].clamp(min=0.0).float().cpu().numpy()
            candidates[:, 2] = candidates[:, 2].clip(0.0, float(width))
            candidates[:, 3] = candidates[:, 3].clip(0.0, float(height))
            keep = batched_nms(
                candidates, scores.cpu().numpy(), labels.cpu().numpy(),
                iou_threshold=iou_threshold, score_threshold=score_threshold)
            keep = keep[:max_detections]
            results.append((candidates[keep],
                            scores.cpu().numpy()[keep],
                            labels.cpu().numpy()[keep]))
        return results


def build_model(preset="tiny", num_classes=1, strides=STRIDES):
    """Build a detector for one preset name (``tiny`` or ``small``)."""
    return RTMDet(num_classes=num_classes, preset=preset, strides=strides)



