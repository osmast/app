"""Training RTMDet on a Dataset export - and evaluating it honestly.

Two things this module refuses to do:

* **train on a leaky split by default.** A corpus split by image puts frames of one
  capture session in train *and* validation, and then reports a score that measures
  memorisation. ``split_by="group"`` (the default) re-partitions by capture with
  :meth:`daphnia.vision.export.Export.regroup`, so validation frames are of animals
  the model has never seen.
* **report a number it cannot stand behind.** :func:`evaluate` computes precision,
  recall, F1 and an interpolated AP50 from its own greedy matching at IoU 0.5, and
  the report says so: it is *not* COCO mAP, and it is not comparable to a number
  from a detector benchmark. What it is comparable to is the next run of this same
  script on the same corpus - which is the comparison that matters here.

The device is chosen for the machine it runs on: **MPS** (Apple-silicon GPU) when
available, otherwise CPU. ``torch`` is imported here and nowhere else in ``daphnia``,
so the Qt application keeps starting without it.
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
from daphnia.vision.export import Export, ExportError
from daphnia.vision.rtmdet import data as data_module
from daphnia.vision.rtmdet.data import (CLASSES, DetectionDataset, collate)
from daphnia.vision.rtmdet.model import CONFIGS, RTMDet, build_model
from daphnia.vision.rtmdet.losses import bbox_overlaps

#: Bumped when a checkpoint's contents change shape; a reader refuses an older one.
CHECKPOINT_FORMAT = "daphnia-rtmdet-1"

#: The default checkpoint, in the git-ignored ``models/`` tree.
DEFAULT_CHECKPOINT = "models/rtmdet_daphnia.pt"


def pick_device(requested=None):
    """``torch.device`` to train on: MPS when it exists, else CPU.

    ``requested`` may be ``"auto"``, ``"mps"``, ``"cpu"``. Asking for MPS on a
    machine that cannot provide it is an error rather than a silent fallback: a
    training run that quietly became 20x slower would be blamed on the model.
    """
    import torch

    if requested in (None, "auto"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is not available. Check that this is an "
            "Apple-silicon Mac with a PyTorch build that includes Metal "
            "(torch.backends.mps.is_available()).")
    return torch.device(requested)


def seed_everything(seed):
    """Seed every generator a run touches, so a result can be reproduced."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    """Everything one training run needs - the whole config, in one place."""

    data: str = "datasets/bb"          # the exported corpus folder
    output: str = DEFAULT_CHECKPOINT
    preset: str = "tiny"
    input_size: int = 640
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 0.004
    final_learning_rate: float = 0.0002
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 35.0
    num_workers: int = 2
    seed: int = 0
    device: str = "auto"
    split_by: str = "group"            # 'group' (held-out captures) | 'export'
    val_interval: int = 1
    score_threshold: float = 0.05      # low: AP must see the marginal detections
    iou_threshold: float = 0.60
    match_iou: float = 0.50            # the IoU a detection must reach to count
    mosaic_probability: float = data_module.MOSAIC_PROBABILITY
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
    best_ap50: float = 0.0
    seconds: float = 0.0

    def note_step(self, step, loss):
        self.steps.append(int(step))
        self.losses.append(float(loss))

    def note_epoch(self, epoch, metrics):
        self.epochs.append(int(epoch))
        self.metrics.append(dict(metrics))
        score = float(metrics.get("ap50", 0.0))
        if score >= self.best_ap50:
            self.best_ap50 = score
            self.best_epoch = int(epoch)

    def smoothed(self, window=20) -> float:
        """Mean of the last ``window`` step losses - the number worth reading."""
        if not self.losses:
            return 0.0
        return float(np.mean(self.losses[-window:]))

    def as_dict(self) -> dict:
        """A JSON-safe summary (the raw step losses are dropped: too long)."""
        return {"best_epoch": self.best_epoch, "best_ap50": self.best_ap50,
                "seconds": self.seconds, "epochs": self.epochs,
                "metrics": self.metrics, "final_loss": self.smoothed()}


def learning_rate_at(step, total_steps, config: TrainConfig) -> float:
    """Linear warm-up then a cosine decay, as RTMDet's schedule does.

    The warm-up matters on a laptop: at full learning rate, the first steps of an
    untrained shared head produce a loss large enough to wreck the BatchNorms, and
    the run then never recovers.
    """
    warmup = max(1, int(config.warmup_steps))
    if step < warmup:
        return config.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (config.final_learning_rate
            + (config.learning_rate - config.final_learning_rate) * cosine)


def build_datasets(export: Export, config: TrainConfig):
    """``(train, val, corpus)`` - the two datasets and the export they came from.

    With ``split_by='group'`` the corpus is regrouped first, and the *regrouped*
    export is what both datasets read, so train and validation can never disagree
    about which capture each frame belongs to.
    """
    corpus = export
    if config.split_by == "group":
        corpus = export.regroup()
    elif config.split_by != "export":
        raise ValueError("split_by must be 'group' or 'export'")
    train = DetectionDataset(
        corpus, "train", input_size=config.input_size, augment=True,
        seed=config.seed, mosaic_probability=config.mosaic_probability)
    val = DetectionDataset(
        corpus, "val", input_size=config.input_size, augment=False,
        seed=config.seed)
    return train, val, corpus


def _match_image(predicted, truth, match_iou):
    """Greedy per-image matching at one IoU; returns the TP/FP flags and counts.

    Predictions are taken in descending score order and each claims the *best*
    still-unclaimed ground truth it overlaps enough. Greedy-by-score is the
    definition every detector benchmark uses, so a number from here can be
    explained - unlike one from an average over classes nobody inspected.
    """
    flags = np.zeros(len(predicted), dtype=np.int64)
    claimed = np.zeros(len(truth), dtype=bool)
    if len(predicted) and len(truth):
        overlap = np.asarray(bbox_overlaps(
            _as_tensor(predicted), _as_tensor(truth)), dtype=np.float32)
        for index in range(len(predicted)):            # already sorted by score
            candidates = np.where((overlap[index] >= match_iou) & (~claimed))[0]
            if candidates.size == 0:
                continue
            best = candidates[np.argmax(overlap[index][candidates])]
            claimed[best] = True
            flags[index] = 1
    return flags, int(claimed.sum())


def _as_tensor(boxes):
    """A ``(N, 4)`` float tensor from anything array-like (torch is imported here)."""
    import torch

    return torch.as_tensor(np.asarray(boxes, dtype=np.float32))


def _interpolated_ap(flags, scores, total_truth):
    """All-point interpolated average precision from ranked TP/FP flags.

    The precision envelope is made monotone first (a later, higher-recall point can
    never have *lower* precision than an earlier one, by definition), which is the
    standard all-point interpolation. No COCO machinery, no 101-point grid - and
    the report names it ``ap50`` rather than ``mAP`` for exactly that reason.
    """
    if total_truth == 0 or len(flags) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    flags = np.asarray(flags, dtype=np.float32)[order]
    true_positive = np.cumsum(flags)
    false_positive = np.cumsum(1.0 - flags)
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-9)
    recall = true_positive / float(total_truth)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    area = 0.0
    previous = 0.0
    for index in range(len(recall)):
        area += max(0.0, recall[index] - previous) * envelope[index]
        previous = max(previous, recall[index])
    return float(area)


def evaluate(model, dataset, config: TrainConfig, device, log=None) -> dict:
    """Precision, recall, F1 and AP50 of ``model`` on ``dataset``.

    Runs each frame at its own letterbox and **maps both sides back** before
    matching, so the IoU is measured in the coordinates the labels were annotated
    in - matching in the padded input would quietly reward a model that is wrong by
    a constant factor. ``device`` may be a ``torch.device`` or one of the strings
    :func:`pick_device` accepts.
    """
    import torch
    from torch.utils.data import DataLoader

    if isinstance(device, str):
        device = pick_device(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate)
    was_training = model.training
    model.eval()

    all_flags, all_scores = [], []
    true_positives = false_positives = false_negatives = detections = 0
    total_truth = 0
    try:
        for batch in loader:
            images = batch["images"].to(device)
            samples = batch["samples"]
            results = model.detect(images, score_threshold=config.score_threshold,
                                   iou_threshold=config.iou_threshold)
            boxes, scores, _labels = results[0]
            sample = samples[0]
            # **Both** sides are mapped back to the original image before matching.
            # The predictions arrive in the padded input frame and the labels are
            # stored there too, so comparing one mapped side with one unmapped side
            # gives an IoU of 0 for *every* pair - a metric that reports 0.000 for a
            # perfect detector, and did so for 30 epochs before this line changed.
            mapped = sample.letterbox.invert(boxes)
            truth = sample.letterbox.invert(sample.boxes)
            total_truth += len(truth)
            detections += len(mapped)
            if log is not None:
                log(f"    {getattr(sample.path, 'name', 'frame')}: "
                    f"{len(mapped)} detection(s), {len(truth)} labelled")
            flags, matched = _match_image(mapped, truth, config.match_iou)
            true_positives += int(flags.sum())
            false_positives += int(len(flags) - flags.sum())
            false_negatives += int(len(truth) - matched)
            all_flags.extend(flags.tolist())
            all_scores.extend(np.asarray(scores, dtype=np.float32).tolist())
    finally:
        model.train(was_training)

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return {"images": len(dataset), "labelled_boxes": total_truth,
            "detections": detections,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1),
            "ap50": _interpolated_ap(all_flags, all_scores, total_truth),
            "match_iou": float(config.match_iou),
            "score_threshold": float(config.score_threshold)}


def format_metrics(metrics) -> str:
    """One line a human can read at the end of an epoch."""
    return (f"P {metrics['precision']:.3f}  R {metrics['recall']:.3f}  "
            f"F1 {metrics['f1']:.3f}  AP50 {metrics['ap50']:.3f}   "
            f"({metrics['detections']} detection(s) for "
            f"{metrics['labelled_boxes']} labelled box(es) on "
            f"{metrics['images']} frame(s))")


# --- checkpoints ------------------------------------------------------------


def checkpoint_payload(model, config: TrainConfig, history=None, metrics=None,
                       corpus=None, weights=None) -> dict:
    """Everything a checkpoint must carry to be usable *and* explainable.

    Not just the weights: the preset and the **input size** decide how the model is
    rebuilt, and the corpus summary records what it was trained on. A checkpoint
    that holds only tensors forces the reader to guess - and a wrong guess loads
    without an error and then predicts nonsense.

    ``weights`` lets a caller save a **kept** state dict (the best epoch) instead of
    the model's current one, so the metrics in the payload and the tensors beside
    them always come from the same epoch.
    """
    from datetime import datetime

    state = (weights if weights is not None else
             {key: value.detach().cpu()
              for key, value in model.state_dict().items()})
    return {
        "format": CHECKPOINT_FORMAT,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": model.preset,
        "input_size": int(config.input_size),
        "num_classes": int(model.num_classes),
        "classes": list(CLASSES)[:int(model.num_classes)],
        "strides": list(model.strides),
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

    The model is rebuilt from the checkpoint's own preset and stride list, so a
    checkpoint from a different preset loads correctly instead of raising a wall of
    mismatched-shape errors.
    """
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError,
            pickle.UnpicklingError) as error:
        # A truncated download, a YAML file renamed .pt, or an HTML error page: all
        # of them arrive here, and all of them deserve the path in the message.
        raise ValueError(f"{path} could not be read as a checkpoint: {error}") \
            from error
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a DaphniaScope RTMDet checkpoint "
                         f"(expected format '{CHECKPOINT_FORMAT}').")
    model = build_model(payload.get("preset", "tiny"),
                        int(payload.get("num_classes", len(CLASSES))),
                        payload.get("strides") or None)
    model.load_state_dict(payload["model"])
    target = pick_device(device)
    model.to(target).eval()
    return model, payload


# --- the run ----------------------------------------------------------------


def train(export: Export, config: TrainConfig, log=print) -> dict:
    """Train a detector on ``export``; returns a report of the run.

    Returns ``{"checkpoint", "best", "history", "device", "parameters",
    "train_images", "val_images", "corpus"}`` - what the CLI prints, and what a
    later evaluation needs in order to say *what* was measured.
    """
    import torch
    from torch.utils.data import DataLoader

    seed_everything(config.seed)
    device = pick_device(config.device)
    train_set, val_set, corpus = build_datasets(export, config)
    if not len(train_set):
        raise ValueError(
            "No training frames with a usable box. Inspect the export first: "
            "python scripts/train_rtmdet.py --inspect --data <folder>")
    if not len(val_set):
        raise ValueError(
            "No validation frames: this corpus has too few captures to hold one "
            "out. Annotate another session, or pass --split-by export (and read "
            "the caveat in the report).")

    model = build_model(config.preset, len(CLASSES)).to(device)
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
    log(f"model         rtmdet-{config.preset}  "
        f"({model.parameter_count():,} parameters)")
    log(f"input         longest side {config.input_size}, padded to a multiple of "
        f"{data_module.SIZE_DIVISOR}")
    log(f"data          {len(train_set)} train frame(s), {len(val_set)} val "
        f"frame(s), split by {config.split_by}")
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
            targets = [{"boxes": target["boxes"].to(device),
                        "labels": target["labels"].to(device)}
                       for target in batch["targets"]]
            outputs = model.loss(images, targets)
            optimizer.zero_grad(set_to_none=True)
            outputs["loss"].backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config.grad_clip)
            optimizer.step()
            step += 1
            history.note_step(step, float(outputs["loss"].detach()))
            if config.log_every and step % config.log_every == 0:
                log(f"  step {step}/{total_steps}  lr {rate:.5f}  "
                    f"loss {history.smoothed():.4f}  "
                    f"(cls {float(outputs['loss_cls']):.4f}, "
                    f"box {float(outputs['loss_box']):.4f}, "
                    f"{outputs['matched']} matched)")

        if config.val_interval and (epoch % config.val_interval == 0
                                    or epoch == config.epochs):
            metrics = evaluate(model, val_set, config, device)
            improved = metrics["ap50"] >= history.best_ap50
            history.note_epoch(epoch, metrics)
            log(f"epoch {epoch}/{config.epochs}  {format_metrics(metrics)}")
            if improved:
                # The weights are kept, not just the number: the checkpoint on disk
                # must *be* the model its metrics describe, and the final epoch is
                # written afterwards with its own numbers.
                best_weights = {key: value.detach().cpu()
                                for key, value in model.state_dict().items()}
                best_result = dict(metrics)
                saved = save_checkpoint(output, checkpoint_payload(
                    model, config, history, metrics, corpus.stats_summary(),
                    weights=best_weights))
                log(f"  best so far -> {saved}")

    history.seconds = time.time() - started
    best = (max(history.metrics, key=lambda item: item["ap50"])
            if history.metrics else {})

    # Two artifacts, and each one means what it says: `output` holds the **best**
    # weights with the metrics of that same epoch, and `<name>_last.pt` holds the
    # final epoch. Overwriting one file with the other's numbers is how a checkpoint
    # ends up reporting a score its weights did not earn.
    if best_weights is not None:
        save_checkpoint(output, checkpoint_payload(
            model, config, history, best_result, corpus.stats_summary(),
            weights=best_weights))
    final = save_checkpoint(
        output.with_name(output.stem + "_last" + output.suffix),
        checkpoint_payload(model, config, history, best, corpus.stats_summary()))
    log(f"done in {history.seconds / 60.0:.1f} min")
    log(f"  best   {output}   {format_metrics(best) if best else 'n/a'}")
    log(f"  last   {final}")
    return {"checkpoint": str(output), "last_checkpoint": str(final),
            "best": best, "history": history.as_dict(), "device": str(device),
            "parameters": model.parameter_count(),
            "train_images": len(train_set), "val_images": len(val_set),
            "corpus": corpus.stats_summary()}


# --- the command line -------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The training command, with every knob named and defaulted."""
    parser = argparse.ArgumentParser(
        prog="train_rtmdet",
        description="Train RTMDet on an exported DaphniaScope dataset.")
    parser.add_argument("--data", default=TrainConfig.data,
                        help="the exported corpus folder (labels.json + images/)")
    parser.add_argument("--output", default=TrainConfig.output,
                        help="where to write the checkpoint")
    parser.add_argument("--preset", default=TrainConfig.preset,
                        choices=sorted(CONFIGS),
                        help="model size (tiny fits a laptop)")
    parser.add_argument("--input-size", type=int, default=TrainConfig.input_size,
                        help="longest side of the network input")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float,
                        default=TrainConfig.weight_decay)
    parser.add_argument("--warmup-steps", type=int,
                        default=TrainConfig.warmup_steps)
    parser.add_argument("--workers", type=int, default=TrainConfig.num_workers,
                        help="data-loader processes (use 0 if macOS misbehaves)")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device,
                        choices=("auto", "mps", "cpu"))
    parser.add_argument("--split-by", default=TrainConfig.split_by,
                        choices=("group", "export"),
                        help="'group' holds out whole captures (honest); "
                             "'export' uses the export's own split (leaky)")
    parser.add_argument("--val-interval", type=int,
                        default=TrainConfig.val_interval)
    parser.add_argument("--score-threshold", type=float,
                        default=TrainConfig.score_threshold)
    parser.add_argument("--match-iou", type=float, default=TrainConfig.match_iou)
    parser.add_argument("--mosaic", type=float,
                        default=TrainConfig.mosaic_probability,
                        help="probability a training sample is a 4-frame mosaic")
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps,
                        help="stop each epoch after N steps (a smoke test)")
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every,
                        help="print a step line every N steps (0 = never)")
    parser.add_argument("--report", default="",
                        help="also write the run report as JSON to this path")
    parser.add_argument("--inspect", action="store_true",
                        help="describe the corpus and stop (no training)")
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate an existing checkpoint on val and stop")
    parser.add_argument("--checkpoint", default="",
                        help="the checkpoint --evaluate should load")
    return parser


def config_from_args(args) -> TrainConfig:
    """A :class:`TrainConfig` from parsed arguments."""
    return TrainConfig(
        data=args.data, output=args.output, preset=args.preset,
        input_size=args.input_size, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        num_workers=args.workers, seed=args.seed, device=args.device,
        split_by=args.split_by, val_interval=args.val_interval,
        score_threshold=args.score_threshold, match_iou=args.match_iou,
        mosaic_probability=args.mosaic, max_steps=args.max_steps,
        log_every=args.log_every)


def describe_export(export: Export, config: TrainConfig, log=print) -> None:
    """Print what the corpus is, and everything about it worth knowing first."""
    log(export.describe())
    if config.split_by == "group":
        log("")
        log("held out by capture (--split-by group):")
        log(export.regroup().describe())
    notes = export.warnings()
    if notes:
        log("")
        log("read this before trusting a number:")
        for note in notes:
            log(f"  - {note}")


def evaluate_checkpoint(export: Export, config: TrainConfig, path, log=print):
    """Load ``path`` and evaluate it on the validation split of ``config``."""
    device = pick_device(config.device)
    model, payload = load_checkpoint(path, device=device)
    _train, val_set, _corpus = build_datasets(export, config)
    if not len(val_set):
        raise ValueError("No validation frames to evaluate on.")
    metrics = evaluate(model, val_set, config, device, log=log)
    log("")
    log(f"{Path(path).name}: {format_metrics(metrics)}")
    log(f"  trained on {payload.get('corpus', {}).get('images', '?')} image(s), "
        f"best epoch {payload.get('history', {}).get('best_epoch', '?')}")
    return metrics


def main(argv=None) -> int:
    """The ``train_rtmdet`` entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    try:
        export = Export.read(config.data)
    except ExportError as error:
        print(f"error: {error}")
        return 2

    if args.inspect:
        describe_export(export, config)
        return 0

    if args.evaluate:
        path = args.checkpoint or config.output
        try:
            metrics = evaluate_checkpoint(export, config, path)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"error: {error}")
            return 2
        if args.report:
            Path(args.report).write_text(json.dumps(metrics, indent=2),
                                         encoding="utf-8")
        return 0

    describe_export(export, config)
    print("")
    try:
        report = train(export, config)
    except (OSError, ValueError, RuntimeError) as error:
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
    print("  AP50 here is this script's own greedy IoU-0.5 score, not COCO mAP: "
          "compare it with another run of this script, not with a paper.")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str),
                                     encoding="utf-8")
        print(f"  report      {args.report}")
    return 0

