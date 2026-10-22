"""Training RTMPose on a Dataset export's keypoints - and evaluating it honestly.

Three things this module refuses to do:

* **train on a leaky split by default.** The export's own split hashes the exported
  copy path, so frames of one capture land on both sides. ``split_by="group"``
  re-partitions by **capture** with :meth:`daphnia.vision.pose_corpus.PoseCorpus.regroup`,
  so validation animals are ones the model has never seen.
* **report a number it cannot stand behind.** :func:`evaluate` reports per-keypoint
  pixel error, PCK at stated thresholds, an OKS with *our* sigmas, and the task's own
  metric - the head-to-tail length error in µm. None of it is COCO keypoint AP, and
  the report says so; what it is comparable to is the next run of this script on the
  same corpus.
* **fall back silently.** A missing checkpoint, a device the machine does not have, an
  empty validation split: each raises a named error, never a quietly worse number.

The device is chosen once (:func:`~daphnia.vision.rtmdet.train.pick_device`): MPS when
available, otherwise CPU. ``torch`` is imported here and nowhere the app needs to
import, so the Qt application keeps starting without it.
"""

import argparse
import json
import math
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from daphnia import paths
from daphnia.vision.pose_corpus import PoseCorpus, PoseError
from daphnia.vision.rtmpose import DEFAULT_INPUT_SIZE, SIMCC_SPLIT_RATIO
from daphnia.vision.rtmpose.data import PoseDataError, PoseDataset, collate
from daphnia.vision.rtmpose.losses import (
    DEFAULT_SIGMAS,
    pck,
    keypoint_errors,
    oks,
)
from daphnia.vision.rtmpose.model import RTMPose, build_model
from daphnia.vision.rtmdet.train import pick_device, seed_everything

#: Bumped when a checkpoint's contents change shape; a reader refuses an older one.
CHECKPOINT_FORMAT = "daphnia-rtmpose-1"

#: The default checkpoint, in the git-ignored ``models/`` tree.
DEFAULT_CHECKPOINT = "models/rtmpose_daphnia.pt"

#: PCK thresholds reported by default, in **input** pixels. They are tiny on purpose:
#: at ~75 µm/px (this corpus) 1 px is ~75 µm, so 5 px is a body-length error of
#: ~0.4 mm - the resolution the thesis actually cares about.
DEFAULT_THRESHOLDS_PX = (1.0, 2.0, 5.0)


def default_output() -> Path:
    """Where a training run writes the best checkpoint by default."""
    return paths.PROJECT_ROOT / DEFAULT_CHECKPOINT


@dataclass
class TrainConfig:
    """Everything one pose-training run needs - the whole config, in one place."""

    data: str = "datasets/kp"          # the exported corpus folder
    output: str = DEFAULT_CHECKPOINT
    preset: str = "tiny"
    input_size: int = DEFAULT_INPUT_SIZE
    split_ratio: float = SIMCC_SPLIT_RATIO
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 0.002
    final_learning_rate: float = 0.0001
    weight_decay: float = 0.05
    warmup_steps: int = 200
    grad_clip: float = 10.0
    num_workers: int = 2
    seed: int = 0
    device: str = "auto"
    split_by: str = "group"            # 'group' (held-out captures) | 'export'
    val_interval: int = 5
    margin: float = 0.25               # the crop context, as a fraction of the box
    flip_probability: float = 0.5
    soft_sigma: float = 2.0            # the SimCC Gaussian width, in bins
    thresholds_px: tuple = DEFAULT_THRESHOLDS_PX
    max_steps: int = 0                 # >0 stops each epoch early (a smoke test)
    log_every: int = 10

    def resolved_output(self) -> Path:
        """The checkpoint path, absolute, with its folder created."""
        path = Path(self.output)
        if not path.is_absolute():
            path = paths.PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class History:
    """What a run saw, so a report can be written without re-reading the log."""

    steps: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    epochs: list = field(default_factory=list)
    metrics: list = field(default_factory=list)      # the validation reports
    best_epoch: int = 0
    best_median_error_px: float = float("inf")
    seconds: float = 0.0

    def note_step(self, step, loss):
        self.steps.append(int(step))
        self.losses.append(float(loss))

    def note_epoch(self, epoch, metrics):
        self.epochs.append(int(epoch))
        self.metrics.append(dict(metrics))
        score = float(metrics.get("median_error_px", float("inf")))
        if score <= self.best_median_error_px:           # lower error is better
            self.best_median_error_px = score
            self.best_epoch = int(epoch)

    def smoothed(self, window=20) -> float:
        """Mean of the last ``window`` step losses - the number worth reading."""
        if not self.losses:
            return 0.0
        return float(np.mean(self.losses[-window:]))

    def as_dict(self) -> dict:
        """A JSON-safe summary (the raw step losses are dropped: too long)."""
        return {"best_epoch": self.best_epoch,
                "best_median_error_px": self.best_median_error_px,
                "seconds": self.seconds, "epochs": self.epochs,
                "metrics": self.metrics, "final_loss": self.smoothed()}


def learning_rate_at(step, total_steps, config: TrainConfig) -> float:
    """Linear warm-up then a cosine decay, as the detector's schedule does.

    The warm-up matters even more here than for the detector: a SimCC head starts
    with a near-uniform distribution over ~385 bins, so the first gradients are large
    and, at full learning rate, would wreck the BatchNorms in the shared backbone.
    """
    warmup = max(1, int(config.warmup_steps))
    if step < warmup:
        return config.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (config.final_learning_rate
            + (config.learning_rate - config.final_learning_rate) * cosine)


def build_datasets(corpus: PoseCorpus, config: TrainConfig):
    """``(train, val, corpus)`` - the two datasets and the corpus they came from.

    With ``split_by='group'`` the corpus is regrouped by capture first, and the
    *regrouped* corpus is what both datasets read, so train and validation can never
    disagree about which capture each frame belongs to.
    """
    source = corpus
    if config.split_by == "group":
        source = corpus.regroup()
    elif config.split_by != "export":
        raise ValueError("split_by must be 'group' or 'export'")
    train = PoseDataset(source, "train", input_size=config.input_size, augment=True,
                        seed=config.seed, margin=config.margin,
                        flip_probability=config.flip_probability)
    val = PoseDataset(source, "val", input_size=config.input_size, augment=False,
                      seed=config.seed, margin=config.margin)
    return train, val, source


# --- the metric -------------------------------------------------------------


def _instances(array) -> np.ndarray:
    """Coerce to ``(N, K, 2)`` float64, so a caller cannot pass the wrong frame twice."""
    values = np.asarray(array, dtype=np.float64)
    if values.ndim < 2:
        return values.reshape(-1, 0, 2)
    keypoints = values.shape[-2] if values.ndim >= 3 else 1
    if values.ndim == 2:
        values = values.reshape(-1, 1, 2)
    return values.reshape(-1, keypoints, 2)


def _sigma_vector(names, keypoints):
    """Per-keypoint OKS sigmas for this corpus's keypoint order."""
    if names:
        return np.asarray([DEFAULT_SIGMAS.get(name, 0.05) for name in names[:keypoints]],
                          dtype=np.float64)
    return np.full(keypoints, 0.05, dtype=np.float64)


def pose_metrics(predicted, truth, areas=None, names=(),
                 thresholds_px=DEFAULT_THRESHOLDS_PX, um_per_px=None,
                 length_pair=(0, 1)) -> dict:
    """Score predictions against truth, in **source-image pixels**.

    Args:
        predicted, truth: ``(N, K, 2)`` in source pixels, keypoint order aligned.
        areas: ``(N,)`` box areas in pixels², for the OKS scale. Missing is allowed
            and only makes OKS uninformative, never the pixel error.
        thresholds_px: the PCK thresholds to report, in pixels.
        um_per_px: micrometres one source pixel spans (this corpus ~74.8). Given, the
            medians and PCK thresholds are restated in **µm** - which is the unit the
            endpoint is judged in.
        length_pair: the two keypoint indices whose distance is the body length.

    Returns:
        A dict of the median/p90/mean keypoint error, per-keypoint medians, PCK at
        each threshold (px and, when ``um_per_px`` is given, µm), the mean OKS, and
        the head-to-tail length error in px and µm.

    This is a **pure function** on purpose, so the oracle test can hand it the truth
    as its own prediction and assert a perfect 1.0 - and a blind stub and assert 0.0.
    """
    predicted = _instances(predicted)
    truth = _instances(truth)
    if predicted.shape != truth.shape:
        raise ValueError(
            f"predicted and truth must have the same shape; got {predicted.shape} and "
            f"{truth.shape}. A shape mismatch here is a frame-of-reference bug "
            "(mlContext.md law 1), not a metric one.")
    instances, keypoints = truth.shape[0], truth.shape[1]
    if instances == 0 or keypoints == 0:
        return {"images": int(instances), "keypoints": int(keypoints),
                "median_error_px": 0.0, "p90_error_px": 0.0, "mean_error_px": 0.0,
                "per_keypoint_median_px": {}, "pck": {}, "oks": 0.0,
                "length_median_error_px": 0.0, "length_median_error_um": None}

    per_instance = np.linalg.norm(predicted - truth, axis=-1)     # (N, K)
    errors = per_instance.reshape(-1)
    median = float(np.median(errors))
    report = {
        "images": int(instances),
        "keypoints": int(keypoints),
        "median_error_px": median,
        "p90_error_px": float(np.percentile(errors, 90)),
        "mean_error_px": float(errors.mean()),
        "per_keypoint_median_px": {
            (names[index] if index < len(names) else f"k{index}"):
                float(np.median(per_instance[:, index]))
            for index in range(keypoints)},
        "pck": {f"{threshold:g}px": pck(errors, threshold)
                for threshold in thresholds_px},
    }
    if um_per_px:
        report["median_error_um"] = median * float(um_per_px)
        report["p90_error_um"] = report["p90_error_px"] * float(um_per_px)
        report["pck_um"] = {
            f"{threshold * float(um_per_px):.0f}um": pck(errors, threshold)
            for threshold in thresholds_px}

    sigmas = _sigma_vector(names, keypoints)
    if areas is None:
        areas = np.ones(instances, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64).reshape(-1)[:instances]
    report["oks"] = float(np.mean([
        oks(predicted[index], truth[index], areas[index], sigmas)
        for index in range(instances)]))

    first, second = int(length_pair[0]), int(length_pair[1])
    if keypoints > max(first, second):
        predicted_length = np.linalg.norm(
            predicted[:, first] - predicted[:, second], axis=-1)
        true_length = np.linalg.norm(truth[:, first] - truth[:, second], axis=-1)
        length_error = np.abs(predicted_length - true_length)
        report["length_median_error_px"] = float(np.median(length_error))
        report["length_median_error_um"] = (float(np.median(length_error))
                                            * float(um_per_px) if um_per_px else None)
    return report


def format_metrics(metrics) -> str:
    """One line a human can read at the end of an epoch."""
    parts = [f"median {metrics['median_error_px']:.2f} px",
             f"p90 {metrics['p90_error_px']:.2f} px"]
    if metrics.get("median_error_um") is not None:
        parts.append(f"median {metrics['median_error_um']:.1f} um")
    pck_values = ", ".join(f"{name} {value:.3f}"
                           for name, value in metrics.get("pck", {}).items())
    parts.append(f"PCK {pck_values}")
    parts.append(f"OKS {metrics['oks']:.3f}")
    if metrics.get("length_median_error_um") is not None:
        parts.append(f"length err {metrics['length_median_error_um']:.1f} um")
    parts.append(f"({metrics['images']} sample(s), {metrics['keypoints']} point(s))")
    return "  ".join(parts)


def _corpus_um_per_px(dataset):
    """The corpus's µm-per-pixel, from the first sample that records a length."""
    for item in getattr(dataset, "items", ())[:20]:
        value = item.um_per_px()
        if value:
            return float(value)
    return None


def evaluate(model, dataset, config: TrainConfig, device, log=None) -> dict:
    """Score ``model`` on ``dataset``, in source-image pixels, with µm alongside.

    Each sample's prediction is mapped back to **source** pixels through its own
    :class:`~daphnia.vision.rtmpose.data.PoseInput` before it is compared with the
    label - both sides in the same frame, always (``mlContext.md`` law 1).
    ``device`` may be a ``torch.device`` or one of the strings :func:`pick_device`
    accepts.
    """
    import torch
    from torch.utils.data import DataLoader

    if isinstance(device, str):
        device = pick_device(device)
    loader = DataLoader(dataset, batch_size=max(1, int(config.batch_size)),
                        shuffle=False, num_workers=0, collate_fn=collate)
    names = tuple(dataset.keypoint_names())
    was_training = model.training
    model.eval()

    predicted, truth, areas = [], [], []
    try:
        for batch in loader:
            images = batch["images"].to(device)
            outputs = model.predict(images).detach().cpu().numpy()
            for index, sample in enumerate(batch["samples"]):
                predicted.append(sample.pose_input.invert_points(outputs[index]))
                truth.append(np.asarray([[x, y] for _name, x, y in sample.source_points],
                                        dtype=np.float64))
                box = sample.bbox
                areas.append((box[2] - box[0]) * (box[3] - box[1])
                             if len(box) == 4 else 1.0)
    finally:
        model.train(was_training)

    keypoints = len(names) or 2
    predicted = (np.asarray(predicted, dtype=np.float64)
                 if predicted else np.zeros((0, keypoints, 2)))
    truth = (np.asarray(truth, dtype=np.float64)
             if truth else np.zeros((0, keypoints, 2)))
    metrics = pose_metrics(predicted, truth, areas=np.asarray(areas, np.float64),
                           names=names, thresholds_px=config.thresholds_px,
                           um_per_px=_corpus_um_per_px(dataset),
                           length_pair=(0, 1))
    if log is not None:
        log(f"  evaluated {metrics['images']} sample(s) in source pixels")
    return metrics


# --- checkpoints ------------------------------------------------------------


def checkpoint_payload(model, config: TrainConfig, history=None, metrics=None,
                       corpus=None, weights=None) -> dict:
    """Everything a checkpoint must carry to be usable *and* explainable.

    Not just the weights: the **input size**, the **split ratio** and the **keypoint
    names** decide how the model is rebuilt and what its outputs mean, and the corpus
    summary records what it was trained on. A checkpoint that holds only tensors
    forces the reader to guess - and a wrong guess loads without an error and then
    predicts nonsense (``mlContext.md`` law 9).

    ``weights`` lets a caller save a **kept** state dict (the best epoch) instead of
    the model's current one, so the metrics in the payload and the tensors beside them
    always come from the same epoch.
    """
    from datetime import datetime

    state = (weights if weights is not None else
             {key: value.detach().cpu()
              for key, value in model.state_dict().items()})
    return {
        "format": CHECKPOINT_FORMAT,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": model.preset,
        "input_size": int(model.input_size),
        "split_ratio": float(model.split_ratio),
        "num_keypoints": int(model.num_keypoints),
        "keypoints": list(model.keypoint_names),
        "model": dict(state),
        "config": asdict(config),
        "history": history.as_dict() if history is not None else {},
        "metrics": dict(metrics or {}),
        "corpus": dict(corpus or {}),
        "split_by": config.split_by,
    }


def save_checkpoint(path, payload) -> Path:
    """Write a checkpoint atomically, so an interrupted run leaves no stub."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path, device="auto"):
    """``(model, payload)`` from a checkpoint, ready for inference.

    The model is rebuilt from the checkpoint's own preset, input size, split ratio and
    keypoint names, so a checkpoint from a different preset loads correctly instead of
    raising a wall of mismatched-shape errors.
    """
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError,
            pickle.UnpicklingError) as error:
        raise ValueError(f"{path} could not be read as a checkpoint: {error}") \
            from error
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a DaphniaScope RTMPose checkpoint "
                         f"(expected format '{CHECKPOINT_FORMAT}').")
    model = build_model(
        payload.get("preset", "tiny"),
        int(payload.get("num_keypoints", 2)),
        int(payload.get("input_size", DEFAULT_INPUT_SIZE)),
        float(payload.get("split_ratio", SIMCC_SPLIT_RATIO)),
        payload.get("keypoints") or ())
    model.load_state_dict(payload["model"])
    target = pick_device(device)
    model.to(target).eval()
    return model, payload


# --- the run ----------------------------------------------------------------


def train(corpus: PoseCorpus, config: TrainConfig, log=print) -> dict:
    """Train a pose model on ``corpus``; returns a report of the run.

    Returns ``{"checkpoint", "last_checkpoint", "best", "history", "device",
    "parameters", "train_samples", "val_samples", "corpus"}`` - what the CLI prints,
    and what a later evaluation needs in order to say *what* was measured.
    """
    import torch
    from torch.utils.data import DataLoader

    seed_everything(config.seed)
    device = pick_device(config.device)
    train_set, val_set, source = build_datasets(corpus, config)
    if not len(train_set):
        raise ValueError(
            "No training samples with a usable box and keypoints. Inspect the corpus "
            "first: python scripts/train_rtmpose.py --inspect --data <folder>")
    if not len(val_set):
        raise ValueError(
            "No validation samples: this corpus has too few captures to hold one out. "
            "Annotate another session, or pass --split-by export (and read the caveat "
            "in the report).")

    names = tuple(corpus.names)
    model = build_model(config.preset, len(names), config.input_size,
                        config.split_ratio, names).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.num_workers, collate_fn=collate,
                        drop_last=False)
    steps_per_epoch = len(loader)
    if config.max_steps:
        steps_per_epoch = min(steps_per_epoch, config.max_steps)
    total_steps = max(1, steps_per_epoch * config.epochs)

    log(f"device        {device}")
    log(f"model         rtmpose-{config.preset}  "
        f"({model.parameter_count():,} parameters)")
    log(f"input         {config.input_size} x {config.input_size}, SimCC "
        f"split ratio {config.split_ratio}  ->  "
        f"{model.head.x_bins} x {model.head.y_bins} bins per keypoint")
    log(f"keypoints     {', '.join(names)}")
    log(f"data          {len(train_set)} train sample(s), {len(val_set)} val "
        f"sample(s), split by {config.split_by}")
    log(f"schedule      {config.epochs} epoch(s) x {steps_per_epoch} step(s) = "
        f"{total_steps} step(s), lr {config.learning_rate} -> "
        f"{config.final_learning_rate}")

    history = History()
    started = time.time()
    output = config.resolved_output()
    best_weights = None
    best_result = {}
    step = 0
    for epoch in range(1, config.epochs + 1):
        train_set.set_epoch(epoch)
        model.train()
        for batch_index, batch in enumerate(loader):
            if config.max_steps and batch_index >= config.max_steps:
                break
            rate = learning_rate_at(step, total_steps, config)
            for group in optimizer.param_groups:
                group["lr"] = rate
            images = batch["images"].to(device)
            outputs = model.loss(images, batch["targets"])
            optimizer.zero_grad(set_to_none=True)
            outputs["loss"].backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            step += 1
            history.note_step(step, float(outputs["loss"].detach()))
            if config.log_every and step % config.log_every == 0:
                log(f"  step {step}/{total_steps}  lr {rate:.5f}  "
                    f"loss {history.smoothed():.4f}  "
                    f"(x {float(outputs['loss_x']):.4f}, "
                    f"y {float(outputs['loss_y']):.4f})")

        if config.val_interval and (epoch % config.val_interval == 0
                                    or epoch == config.epochs):
            metrics = evaluate(model, val_set, config, device)
            improved = metrics["median_error_px"] <= history.best_median_error_px
            history.note_epoch(epoch, metrics)
            log(f"epoch {epoch}/{config.epochs}  {format_metrics(metrics)}")
            if improved:
                # The weights are kept, not just the number: the checkpoint on disk
                # must *be* the model its metrics describe.
                best_weights = {key: value.detach().cpu()
                                for key, value in model.state_dict().items()}
                best_result = dict(metrics)
                saved = save_checkpoint(output, checkpoint_payload(
                    model, config, history, metrics, source.stats_summary(),
                    weights=best_weights))
                log(f"  best so far -> {saved}")

    history.seconds = time.time() - started
    best = (min(history.metrics, key=lambda item: item["median_error_px"])
            if history.metrics else {})

    # Two artifacts, and each one means what it says: `output` holds the **best**
    # weights with the metrics of that same epoch, and `<name>_last.pt` holds the
    # final epoch - so no checkpoint ever reports a score its weights did not earn.
    if best_weights is not None:
        save_checkpoint(output, checkpoint_payload(
            model, config, history, best_result, source.stats_summary(),
            weights=best_weights))
    final = save_checkpoint(
        output.with_name(output.stem + "_last" + output.suffix),
        checkpoint_payload(model, config, history, best, source.stats_summary()))
    log(f"done in {history.seconds / 60.0:.1f} min")
    log(f"  best   {output}   {format_metrics(best) if best else 'n/a'}")
    log(f"  last   {final}")
    return {"checkpoint": str(output), "last_checkpoint": str(final),
            "best": best, "history": history.as_dict(), "device": str(device),
            "parameters": model.parameter_count(),
            "train_samples": len(train_set), "val_samples": len(val_set),
            "corpus": source.stats_summary()}


# --- the command line -------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The pose-training command, with every knob named and defaulted."""
    parser = argparse.ArgumentParser(
        prog="train_rtmpose",
        description="Train RTMPose on a DaphniaScope dataset export's keypoints.")
    parser.add_argument("--data", default=TrainConfig.data,
                        help="the exported corpus folder (labels.json + images/)")
    parser.add_argument("--output", default=TrainConfig.output,
                        help="where the best checkpoint is written")
    parser.add_argument("--preset", default=TrainConfig.preset,
                        choices=("tiny", "small"))
    parser.add_argument("--input-size", type=int, default=TrainConfig.input_size,
                        help="the square the box is resized to (a multiple of 8)")
    parser.add_argument("--split-ratio", type=float, default=TrainConfig.split_ratio,
                        help="SimCC bins per input pixel (2.0 = mmpose default)")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float,
                        default=TrainConfig.weight_decay)
    parser.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device,
                        help="auto | mps | cpu (auto = MPS when available)")
    parser.add_argument("--split-by", default=TrainConfig.split_by,
                        choices=("group", "export"),
                        help="group = hold out whole captures (recommended)")
    parser.add_argument("--val-interval", type=int,
                        default=TrainConfig.val_interval)
    parser.add_argument("--margin", type=float, default=TrainConfig.margin,
                        help="crop context around the box, as a fraction of its side")
    parser.add_argument("--flip", type=float, default=TrainConfig.flip_probability,
                        help="probability of a horizontal-flip augmentation")
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps,
                        help="stop each epoch after this many steps (a smoke test)")
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--inspect", action="store_true",
                        help="describe the corpus and its warnings, then exit")
    parser.add_argument("--evaluate", action="store_true",
                        help="score a checkpoint on the validation split, then exit")
    parser.add_argument("--checkpoint", default=None,
                        help="the checkpoint to evaluate (default: --output)")
    parser.add_argument("--report", default=None,
                        help="write the run's report as JSON to this file")
    return parser


def config_from_args(args) -> TrainConfig:
    """A :class:`TrainConfig` from parsed arguments."""
    return TrainConfig(
        data=args.data, output=args.output, preset=args.preset,
        input_size=args.input_size, split_ratio=args.split_ratio,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, num_workers=args.workers, seed=args.seed,
        device=args.device, split_by=args.split_by, val_interval=args.val_interval,
        margin=args.margin, flip_probability=args.flip, max_steps=args.max_steps,
        log_every=args.log_every)


def describe_corpus(corpus: PoseCorpus, config: TrainConfig, log=print) -> None:
    """Print what the corpus is, and everything about it worth knowing first."""
    log(corpus.describe())
    if config.split_by == "group":
        log("")
        log("held out by capture (--split-by group):")
        log(corpus.regroup().describe())
    notes = corpus.warnings()
    if notes:
        log("")
        log("read this before trusting a number:")
        for note in notes:
            log(f"  - {note}")


def evaluate_checkpoint(corpus: PoseCorpus, config: TrainConfig, path,
                        log=print) -> dict:
    """Load ``path`` and evaluate it on the validation split of ``config``."""
    device = pick_device(config.device)
    model, payload = load_checkpoint(path, device=device)
    _train, val_set, _source = build_datasets(corpus, config)
    if not len(val_set):
        raise ValueError("No validation samples to evaluate on.")
    metrics = evaluate(model, val_set, config, device, log=log)
    log("")
    log(f"{Path(path).name}: {format_metrics(metrics)}")
    log(f"  trained on {payload.get('corpus', {}).get('samples', '?')} sample(s), "
        f"best epoch {payload.get('history', {}).get('best_epoch', '?')}")
    return metrics


def main(argv=None) -> int:
    """The ``train_rtmpose`` entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    try:
        corpus = PoseCorpus.read(config.data)
    except PoseError as error:
        print(f"error: {error}")
        return 2

    if args.inspect:
        describe_corpus(corpus, config)
        return 0

    if args.evaluate:
        path = args.checkpoint or config.output
        try:
            metrics = evaluate_checkpoint(corpus, config, path)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"error: {error}")
            return 2
        if args.report:
            Path(args.report).write_text(json.dumps(metrics, indent=2),
                                         encoding="utf-8")
        return 0

    describe_corpus(corpus, config)
    print("")
    try:
        report = train(corpus, config)
    except (OSError, ValueError, RuntimeError, PoseDataError) as error:
        print(f"error: {error}")
        return 2

    print("")
    print("RESULT")
    print(f"  checkpoint  {report['checkpoint']}")
    print(f"  device      {report['device']}")
    print(f"  parameters  {report['parameters']:,}")
    if report["best"]:
        print(f"  best        {format_metrics(report['best'])}")
    print(f"  split by    {config.split_by}")
    print("  These are keypoint pixel errors, PCK and an OKS with OUR sigmas - not "
          "COCO keypoint AP. Compare with another run of this script on the same "
          "corpus, not with a paper.")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str),
                                     encoding="utf-8")
        print(f"  report      {args.report}")
    return 0
