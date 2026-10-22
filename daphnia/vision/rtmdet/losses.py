"""RTMDet's losses and the label assignment that makes them work.

No mmdetection, no mmcv: this is the part of RTMDet that is *algorithm*, written
against plain PyTorch so it runs on an M4 Mac (MPS) or on CPU.

* :func:`distance2bbox` / :func:`bbox2distance` - the coding RTMDet uses. A
  prediction is **four distances to the four sides** of the box from one point in
  a feature map, not an anchor and not a distribution (no DFL). One anchor per
  location with side ``stride`` is the whole prior set, which is why RTMDet has no
  ``num_anchors`` to configure.
* :class:`QualityFocalLoss` - the classification loss whose target is a **soft
  label**: the IoU of the assigned box with the ground truth, so a location that
  nearly covers an animal is asked to be slightly less confident than one that
  covers it exactly.
* :class:`GIoULoss` - the regression loss, on the decoded box (RTMDet regresses
  boxes, it does not select sides from a distribution).
* :class:`SimOTAAssigner` - RTMDet's dynamic label assignment. For every ground
  truth the ``dynamic_k`` cheapest locations become its positives, where the cost
  mixes classification and IoU and ``k`` is the sum of the ten best IoUs, floored
  at 1. That is what gives RTMDet its accuracy without any anchor tuning - and it
  is also the slowest part of a training step, so the tensors stay on the device
  and nothing is gathered per prior.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Guard against a log of zero and against dividing by an empty overlap.
EPS = 1e-7


def distance2bbox(points, distance):
    """``(x1, y1, x2, y2)`` from prior points and ``(l, t, r, b)`` distances.

    ``points`` is ``(..., 2)`` and ``distance`` is ``(..., 4)``; both are in image
    pixels, so the caller must have scaled the raw prediction by the level's
    stride first (RTMDet does that in :meth:`RTMDet.decode`).
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox2distance(points, bbox):
    """The inverse of :func:`distance2bbox`, clamped at 0 (a negative distance
    would be a box inside out)."""
    left = (points[..., 0] - bbox[..., 0]).clamp(min=0)
    top = (points[..., 1] - bbox[..., 1]).clamp(min=0)
    right = (bbox[..., 2] - points[..., 0]).clamp(min=0)
    bottom = (bbox[..., 3] - points[..., 1]).clamp(min=0)
    return torch.stack([left, top, right, bottom], dim=-1)


def bbox_overlaps(first, second, mode="iou"):
    """Pairwise overlap of ``(N, 4)`` and ``(M, 4)`` boxes - ``(N, M)``.

    ``mode='giou'`` also returns the generalised IoU, which adds the smallest
    enclosing box so a *non-overlapping* prediction still gets a gradient.
    """
    n, m = first.size(0), second.size(0)
    if n == 0 or m == 0:
        shape = (n, m, 1) if mode == "giou" else (n, m)
        return first.new_zeros(shape)

    lt = torch.max(first[:, None, :2], second[None, :, :2])       # (N, M, 2)
    rb = torch.min(first[:, None, 2:], second[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area_a = (first[:, 2] - first[:, 0]).clamp(min=0) * \
             (first[:, 3] - first[:, 1]).clamp(min=0)
    area_b = (second[:, 2] - second[:, 0]).clamp(min=0) * \
             (second[:, 3] - second[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union.clamp(min=EPS)
    if mode != "giou":
        return iou

    enclosing_lt = torch.min(first[:, None, :2], second[None, :, :2])
    enclosing_rb = torch.max(first[:, None, 2:], second[None, :, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing = enclosing_wh[..., 0] * enclosing_wh[..., 1]
    return iou - (enclosing - union) / enclosing.clamp(min=EPS)


class QualityFocalLoss(nn.Module):
    """Classification with a **continuous** target (RTMDet's ``QualityFocalLoss``).

    A location's target is its assigned IoU rather than 1 - "this is an animal, and
    I am 0.83 sure" - which is exactly what makes the score usable as a quality
    estimate at inference time. ``beta=2`` is the focal exponent, so easy negatives
    (the overwhelming majority of 640x640 locations) contribute almost nothing.

    ``pred`` arrives as raw logits ``(N, C)``; the target is ``(N, C)`` with values
    in ``[0, 1]`` and 0 everywhere the location is background.
    """

    def __init__(self, beta=2.0, loss_weight=1.0):
        super().__init__()
        self.beta = float(beta)
        self.loss_weight = float(loss_weight)

    def forward(self, pred, target, weight=None):
        """Mean quality focal loss over the locations of one image."""
        if pred.numel() == 0:
            return pred.sum() * 0.0
        score = pred.sigmoid()
        # |y - sigma(x)|^beta turns the focal factor into a *soft* one.
        focal = (target - score).abs().pow(self.beta)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        per_location = (bce * focal).sum(dim=-1)        # over the class axis
        if weight is not None:
            per_location = per_location * weight
        return per_location.sum() / max(1, pred.size(0)) * self.loss_weight


class GIoULoss(nn.Module):
    """Regression loss on the **decoded** box: mean ``1 - GIoU`` over positives.

    ``target`` is the matched **ground-truth box** in image pixels, not a coded
    distance: RTMDet regresses a box, so the loss compares boxes. ``weight`` is 1
    at the locations the assignment chose and 0 everywhere else, which is why the
    positives are selected *before* any arithmetic - a background prediction must
    not enter the mean at all, and must never be able to produce a NaN through a
    degenerate box multiplied by zero.
    """

    def __init__(self, loss_weight=2.0):
        super().__init__()
        self.loss_weight = float(loss_weight)

    def forward(self, pred, target, weight=None):
        """Loss of ``(N, 4)`` decoded predictions against ``(N, 4)`` boxes."""
        if pred.numel() == 0:
            return pred.sum() * 0.0
        if weight is not None:
            positive = weight > 0
            if not bool(positive.any()):
                return pred.sum() * 0.0
            pred, target = pred[positive], target[positive]
        giou = bbox_overlaps(pred, target, mode="giou").diagonal()
        return (1.0 - giou).mean() * self.loss_weight


def batched_nms(boxes, scores, labels, iou_threshold=0.6, score_threshold=0.0):
    """Class-aware greedy non-maximum suppression, on **NumPy**.

    Deliberately not ``torchvision.ops.nms``: the app must not gain a dependency
    for sixteen lines of arithmetic, and inference happens on a frame at a time so
    plain NumPy is fast enough. Returns the kept indices, best score first.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    keep = []
    if scores.size == 0:
        return keep

    order = np.argsort(-scores)
    candidate = order[scores[order] >= score_threshold]
    suppressed = np.zeros(scores.shape[0], dtype=bool)
    for index in candidate:
        if suppressed[index]:
            continue
        keep.append(int(index))
        rest = candidate[candidate != index]
        rest = rest[(labels[rest] == labels[index]) & (~suppressed[rest])]
        if rest.size == 0:
            continue
        overlap = _iou_1_to_many(boxes[index], boxes[rest])
        suppressed[rest[overlap > iou_threshold]] = True
    return keep


def _iou_1_to_many(box, others):
    """IoU of one box against many ``(M, 4)`` boxes - NumPy, one allocation."""
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = (np.clip(others[:, 2] - others[:, 0], 0, None)
              * np.clip(others[:, 3] - others[:, 1], 0, None))
    union = area_a + area_b - inter
    return inter / np.maximum(union, EPS)


class SimOTAAssigner:
    """RTMDet's dynamic label assignment (simplified optimal transport).

    For one image:

    1. a prior is a candidate only if its **centre** falls inside a ground-truth
       box *and* within ``center_radius * stride`` of that box's centre - the
       "centre prior" that keeps the assignment local and out of the background;
    2. a cost matrix pairs every candidate with every ground truth, mixing the
       classification cost (RTMDet's focal-style cost) and ``-log(IoU)``;
    3. each ground truth takes its ``k`` cheapest candidates, with
       ``k = max(1, floor(sum of its ten best IoUs))`` - the *dynamic* part: a big
       animal claims many locations, a neonate claims a few, and nothing is tuned;
    4. a location wanted by two animals goes to the cheaper of them, each positive's
       classification target becomes **its own IoU** (a soft label), and its
       regression target becomes the ground-truth **box**.
    """

    #: Cost weights of the assignment, from the RTMDet configuration.
    IOU_WEIGHT = 3.0
    CLS_WEIGHT = 1.0
    #: The focal parameters the assignment's classification cost uses.
    ALPHA = 0.25
    BETA = 2.0
    #: A pair that is not a candidate may never be chosen.
    INVALID_COST = 1e8

    def __init__(self, num_classes=1, center_radius=2.5, candidate_topk=13,
                 topk_iou=10):
        self.num_classes = int(num_classes)
        self.center_radius = float(center_radius)
        self.candidate_topk = int(candidate_topk)
        self.topk_iou = int(topk_iou)

    def assign(self, priors, cls_pred, decoded_boxes, gt_bboxes, gt_labels):
        """Assign every prior to a ground truth, or to the background.

        Args:
            priors: ``(N, 4)`` as ``(x_centre, y_centre, stride_w, stride_h)`` in
                image pixels - one point per feature-map location.
            cls_pred: ``(N, C)`` raw classification logits.
            decoded_boxes: ``(N, 4)`` boxes in image pixels (stride-scaled already).
            gt_bboxes: ``(M, 4)`` ground-truth boxes in image pixels.
            gt_labels: ``(M,)`` zero-based class indices.

        Returns:
            ``(cls_target, cls_weight, box_target, box_weight, matched)`` - the
            first four ``(N, ...)`` tensors on the prediction's device, and
            ``matched`` the number of positive locations (0..N).
        """
        n = priors.size(0)
        cls_target = cls_pred.new_zeros((n, self.num_classes))
        cls_weight = cls_pred.new_ones((n,))
        box_target = priors.new_zeros((n, 4))
        box_weight = priors.new_zeros((n,))
        if gt_bboxes.numel() == 0 or n == 0:
            return cls_target, cls_weight, box_target, box_weight, 0

        centre = priors[:, :2]
        radius = priors[:, 2:4] * self.center_radius
        gt_centre = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5

        # 1. the centre prior: inside the box, and near enough to its centre.
        inside = ((centre[:, None, 0] > gt_bboxes[None, :, 0])
                  & (centre[:, None, 0] < gt_bboxes[None, :, 2])
                  & (centre[:, None, 1] > gt_bboxes[None, :, 1])
                  & (centre[:, None, 1] < gt_bboxes[None, :, 3]))
        near = ((centre[:, None, :] - gt_centre[None, :, :]).abs()
                < radius[:, None, :]).all(dim=-1)
        candidate = inside & near

        # 2. the cost matrix.
        iou = bbox_overlaps(decoded_boxes, gt_bboxes)                 # (N, M)
        cost = (self._cls_cost(cls_pred, gt_labels) * self.CLS_WEIGHT
                - torch.log(iou.clamp(min=EPS)) * self.IOU_WEIGHT)
        invalid = cost.new_full((), self.INVALID_COST)
        cost = torch.where(candidate, cost, invalid)

        with torch.no_grad():
            # 3. the dynamic k, and the k cheapest candidates of each ground truth.
            best_iou = iou.topk(min(self.topk_iou, n), dim=0).values  # (k, M)
            dynamic_k = best_iou.sum(dim=0).floor().clamp(min=1).long()
            matching = torch.zeros_like(cost)
            for index in range(gt_bboxes.size(0)):
                k = min(int(dynamic_k[index]), self.candidate_topk, n)
                chosen = cost[:, index].topk(k, largest=False).indices
                matching[chosen, index] = 1.0

            # A location claimed by several animals goes to the cheapest of them.
            contested = matching.sum(dim=1) > 1
            if bool(contested.any()):
                rows = contested.nonzero(as_tuple=False).flatten()
                matching[rows] = 0.0
                matching[rows, cost[rows].argmin(dim=1)] = 1.0

        # 4. the targets, from the assigned pairs.
        positive = matching.sum(dim=1) > 0
        matched = int(positive.sum())
        if matched == 0:
            return cls_target, cls_weight, box_target, box_weight, 0
        chosen_gt = matching[positive].argmax(dim=1)

        cls_target[positive, gt_labels[chosen_gt]] = iou[positive, chosen_gt]
        box_target[positive] = gt_bboxes[chosen_gt]
        box_weight[positive] = 1.0
        return cls_target, cls_weight, box_target, box_weight, matched

    def _cls_cost(self, cls_pred, gt_labels):
        """RTMDet's focal-style classification cost, ``(N, M)``.

        ``pos_cost - neg_cost`` for the ground truth's class: what a location loses
        by being confident about that class, minus what it gains by being confident
        about *not* it. It is the classification term the assignment sorts by.
        """
        score = cls_pred.sigmoid()
        neg = (-(1 - score + EPS).log() * (1 - self.ALPHA) * score.pow(self.BETA))
        pos = (-(score + EPS).log() * self.ALPHA * (1 - score).pow(self.BETA))
        return pos[:, gt_labels] - neg[:, gt_labels]

