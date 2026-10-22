"""The maths of RTMPose: SimCC coding, the KL loss, and the point metrics.

Everything here is a function of tensors or arrays with **no model in sight**, so it
can be tested on one hand-computed point - the discipline ``mlContext.md`` §3 asks
for, and the reason a coordinate bug shows up in a unit test instead of in a training
run.

## The SimCC idea (arXiv:2107.03332, RTMPose arXiv:2303.07399)

Instead of a 2-D heatmap per keypoint, each coordinate is predicted as a **1-D
distribution**: the x of a keypoint is classified over ``input_width * split_ratio``
bins, and the y over ``input_height * split_ratio`` bins, independently. The target
is a Gaussian centred on the true bin, and the loss is a KL divergence between the
predicted log-distribution and that target. Decoding is the ``argmax`` of the trained
distribution - what the loss was trained against - divided by the split ratio.
"""

import numpy as np

#: The width of the Gaussian soft target, in *bins*. One bin is the training
#: resolution, so a sigma of 2 spreads the target over ~5 bins: enough that the
#: classifier is not asked to nail one integer, small enough to keep precision.
DEFAULT_SIGMA = 2.0

#: OKS keypoint sigmas, as a fraction of the box side. A documented choice of *ours*
#: (one organism, two near-symmetric landmarks) - so an OKS printed by this project
#: is comparable with another run of this project and with nothing else.
DEFAULT_SIGMAS = {"head": 0.05, "tail": 0.05}


def gaussian_1d(centre, length, sigma=DEFAULT_SIGMA) -> np.ndarray:
    """A normalised 1-D Gaussian over ``length`` bins, centred at ``centre``.

    Returns a ``float32`` array summing to 1 - a valid target for a KL divergence.
    """
    bins = np.arange(int(length), dtype=np.float32)
    spread = max(float(sigma), 1e-6)
    weights = np.exp(-0.5 * ((bins - float(centre)) / spread) ** 2)
    total = float(weights.sum())
    return (weights / total if total > 0 else weights).astype(np.float32)


def simcc_encode(points, input_size, split_ratio, sigma=DEFAULT_SIGMA):
    """``((K, 2)`` points in input pixels``)`` -> ``(x_target, y_target)`` distributions.

    ``input_size`` is ``(width, height)`` of the network input; ``split_ratio`` is the
    SimCC magnification (2.0 by default, mmpose's own). The bin of a coordinate ``p``
    in a side of length ``L`` is ``round(p * split_ratio)`` and the output has
    ``round(L * split_ratio) + 1`` bins, so a point on the far edge still has a bin.
    """
    width, height = int(input_size[0]), int(input_size[1])
    x_bins = int(round(width * split_ratio)) + 1
    y_bins = int(round(height * split_ratio)) + 1
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        return (np.zeros((0, x_bins), np.float32),
                np.zeros((0, y_bins), np.float32))
    x_target = np.stack([gaussian_1d(min(max(p[0] * split_ratio, 0.0), x_bins - 1),
                                     x_bins, sigma) for p in points])
    y_target = np.stack([gaussian_1d(min(max(p[1] * split_ratio, 0.0), y_bins - 1),
                                     y_bins, sigma) for p in points])
    return x_target.astype(np.float32), y_target.astype(np.float32)


def simcc_decode(x_logits, y_logits, split_ratio):
    """``(B, K, nx)`` and ``(B, K, ny)`` logits -> ``(B, K, 2)`` points in input pixels.

    ``argmax`` per coordinate, divided by ``split_ratio`` - the inverse of
    :func:`simcc_encode`'s bin rule, so an encoded integer or half-pixel decodes back
    to itself (the round-trip test pins it).
    """
    import torch

    x_index = torch.argmax(x_logits, dim=-1).to(torch.float32) / float(split_ratio)
    y_index = torch.argmax(y_logits, dim=-1).to(torch.float32) / float(split_ratio)
    return torch.stack([x_index, y_index], dim=-1)


def simcc_loss(x_logits, y_logits, x_target, y_target):
    """KL divergence of both coordinate distributions against their soft targets.

    Returns ``(loss, loss_x, loss_y)`` where ``loss_x`` and ``loss_y`` are detached,
    so a run can see *which* coordinate is lagging - a symmetric animal can hide a
    one-axis failure inside a falling total (``mlContext.md`` §8).

    The two KL sums are divided by ``batch * num_keypoints``, so the number is a
    **mean per-keypoint divergence over the bins** - readable, and small enough that
    the learning rate means the same thing for a 4-keypoint corpus as for a 2-point
    one.
    """
    import torch
    from torch.nn import functional

    batch, keypoints = x_logits.shape[0], x_logits.shape[1]
    divisor = max(1, batch * keypoints)
    target_x = torch.as_tensor(x_target, dtype=torch.float32, device=x_logits.device)
    target_y = torch.as_tensor(y_target, dtype=torch.float32, device=y_logits.device)
    # log_softmax for the prediction, the target's own mass for the reference.
    loss_x = functional.kl_div(x_logits.log_softmax(dim=-1), target_x,
                               reduction="sum") / divisor
    loss_y = functional.kl_div(y_logits.log_softmax(dim=-1), target_y,
                               reduction="sum") / divisor
    return loss_x + loss_y, loss_x.detach(), loss_y.detach()


# --- the metrics (pure NumPy, so the oracle test needs no model) -------------


def keypoint_errors(predicted, truth) -> np.ndarray:
    """``(N, K, 2)`` predicted vs ``(N, K, 2)`` truth -> ``(N, K)`` pixel errors.

    Both sides are in the **same** frame - input pixels or source pixels, whichever
    the caller decided - and comparing a prediction in one frame with a truth in
    another is exactly the bug that cost this project thirty epochs
    (``mlContext.md`` law 1). The unit test hands both sides the same array.
    """
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 2)
    count = min(len(predicted), len(truth))
    if count == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(predicted[:count] - truth[:count], axis=-1)


def pck(errors, threshold) -> float:
    """Fraction of keypoints within ``threshold`` pixels - ``errors`` is flat.

    ``threshold`` is in the units of ``errors``; a caller reporting in µm converts
    first, so PCK is always quoted with its unit.
    """
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float((values <= float(threshold)).mean())


def oks(predicted, truth, area, sigmas=None) -> float:
    """COCO keypoint OKS for one instance, meaned over keypoints.

    ``area`` is the object's box area in the same squared unit as the errors
    (pixels²), and ``sigmas`` is the per-keypoint sigma vector for that instance (see
    :data:`DEFAULT_SIGMAS`). Returns 0.0 for an empty instance rather than NaN, so a
    corpus with no truth scores 0.0 - the blind-stub half of the oracle test.
    """
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 2)
    count = min(len(predicted), len(truth))
    if count == 0:
        return 0.0
    sigmas = np.asarray(sigmas if sigmas is not None else [0.05] * count,
                        dtype=np.float64)[:count]
    distances = ((predicted[:count] - truth[:count]) ** 2).sum(axis=1)
    variance = 2.0 * float(area) * sigmas ** 2
    return float(np.mean(np.exp(-distances / np.maximum(variance, 1e-9))))
