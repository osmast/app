"""The model registry: which checkpoints exist, and which one is in force.

The Predict surface used to be the only place that knew which checkpoint was loaded, and
it kept that knowledge in its own panel: a detector loaded there was invisible to every
other surface - a Dataset queue wanting pre-labels had no way to ask "which model is the
lab using?". The registry is that knowledge in **one place**: a ``models`` row per
checkpoint (its path, preset, training date, the classes or keypoints it predicts, and the
metrics it earned) with exactly one **active** row per kind.

Two properties matter:

* **Qt-free and torch-free.** It records what a checkpoint *is*; loading one is
  :mod:`daphnia.vision.rtmdet`'s or :mod:`daphnia.vision.rtmpose`'s job, behind a lazy
  import. That keeps this module testable anywhere and keeps PyTorch out of the app's
  start-up path (``SR-CORE-011``).
* **A metric travels with its checkpoint** (``mlContext.md`` §8): the numbers are stored
  as they were written in the checkpoint's own report, and reading them back never
  invites a claim about a model in general.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

#: The two models the pipeline has - and so the two kinds of registry row.
KINDS = ("detector", "pose")

#: How a kind is phrased for an operator.
KIND_LABELS = {"detector": "Detector", "pose": "Pose model"}

#: The **network** each kind is, with the name its paper and its reference implementation use
#: (`mlContext.md` §5). The menu is titled with these (`X-N13`): an operator who reads the
#: stack elsewhere finds *RTMDet* and *RTMPose* here, not an invented word.
KIND_MODELS = {"detector": "RTMDet", "pose": "RTMPose"}

#: The name a checkpoint of each kind starts with - what a training run writes
#: (`models/rtmdet_daphnia.pt`, `models/rtmpose_daphnia.pt`; `Makefile`: `make train`). It is
#: how the *available* files in the models directory are offered in the right menu without
#: opening them (opening one to decide whether to offer it would drag PyTorch into the menu,
#: `SR-CORE-011`).
KIND_PREFIXES = {"detector": "rtmdet", "pose": "rtmpose"}

#: The suffixes a checkpoint file has.
CHECKPOINT_SUFFIXES = (".pt", ".pth")

#: The report a training run writes for each kind (`Makefile`: `make train`, `make train-pose`).
REPORT_FILES = {"detector": "rtmdet_report.json", "pose": "rtmpose_report.json"}

#: Which numbers of a report go into the registry, and under which name. Explicit, so a new
#: model adds a row here instead of a heuristic; every one of them is the report's own value.
REPORT_METRICS = {
    "detector": ("ap50", "precision", "recall", "best_epoch", "val_images"),
    "pose": ("median_error_px", "p90_error_px", "oks", "length_median_error_um",
             "best_epoch", "val_samples"),
}


class ModelError(ValueError):
    """A registry problem to fix rather than to guess around (no silent fallback)."""


@dataclass(frozen=True)
class ModelRecord:
    """One checkpoint, as the registry knows it."""

    kind: str
    path: str
    name: str
    preset: str = ""
    trained_at: str = ""
    classes: tuple = ()
    keypoints: tuple = ()
    metrics: dict = field(default_factory=dict)
    added_at: str = ""
    is_active: bool = False

    @property
    def label(self) -> str:
        """What a menu entry says: the file, and what kind of model it is.

        ``rtmdet_daphnia.pt  (rtmdet-tiny, trained 2026-10-17)`` - the network's own name comes
        from the **kind** (`KIND_PREFIXES`), because a pose checkpoint labelled ``rtmdet-tiny``
        is a menu that lies about which model would answer.
        """
        network = KIND_PREFIXES.get(self.kind, "")
        preset = f"{network}-{self.preset}" if (self.preset and network) else self.preset
        parts = [part for part in (preset,
                                   f"trained {self.trained_at}" if self.trained_at else "")
                 if part]
        return f"{self.name}  ({', '.join(parts)})" if parts else self.name

    @property
    def what_it_predicts(self) -> str:
        """The classes (detector) or keypoints (pose) this checkpoint was trained for."""
        return ", ".join(self.classes or self.keypoints)

    @property
    def detail(self) -> str:
        """Everything a tooltip can hold: the path, what it predicts, its own metrics."""
        lines = [self.path]
        if self.what_it_predicts:
            lines.append(f"predicts: {self.what_it_predicts}")
        if self.metrics:
            measured = "   ".join(f"{key} {value:.3f}" if isinstance(value, float)
                                  else f"{key} {value}"
                                  for key, value in sorted(self.metrics.items()))
            lines.append(f"its own report: {measured}")
        return "\n".join(lines)


def record_from_payload(kind, path, payload) -> dict:
    """The registry fields a checkpoint's own payload carries.

    The two vision modules write similar payloads (``preset``, ``classes`` /
    ``keypoints``, ``created_at``, ``metrics``); this reads them without importing either
    of them, so the caller keeps PyTorch at arm's length.
    """
    payload = dict(payload or {})
    return {
        "preset": str(payload.get("preset") or ""),
        "trained_at": str(payload.get("created_at") or payload.get("trained_at") or ""),
        "classes": tuple(payload.get("classes") or ()),
        "keypoints": tuple(payload.get("keypoints") or ()),
        "metrics": dict(payload.get("metrics") or {}),
    }


def _checked_kind(kind) -> str:
    key = str(kind or "").strip().lower()
    if key not in KINDS:
        raise ModelError(f"unknown model kind {kind!r}; the kinds are "
                         f"{', '.join(KINDS)}")
    return key


def _split(value) -> tuple:
    return tuple(part for part in str(value or "").split(",") if part)


def _row_to_record(row) -> ModelRecord:
    """One ``models`` row as a :class:`ModelRecord`."""
    try:
        metrics = json.loads(row["metrics_json"] or "{}")
    except (TypeError, ValueError):
        metrics = {}
    return ModelRecord(
        kind=row["kind"], path=row["path"], name=row["name"],
        preset=row["preset"] or "", trained_at=row["trained_at"] or "",
        classes=_split(row["classes"]), keypoints=_split(row["keypoints"]),
        metrics=metrics if isinstance(metrics, dict) else {},
        added_at=row["added_at"] or "", is_active=bool(row["is_active"]))


def record_from_report(kind, report) -> dict:
    """The registry fields a **training report** carries (`models/<kind>_report.json`).

    Every number here is the *report's own*: the metrics come from its ``best`` block, which
    is the run's account of the epoch it chose, and ``best_epoch`` / the validation size are
    kept beside them so the tooltip can say where the number came from (`mlContext.md` law 5).
    What the report cannot know stays empty rather than guessed - the *preset* and the
    ``created_at`` live in the checkpoint payload and are filled in by
    :meth:`daphnia.ui.model_menu.ModelMenu.load` the first time the file is loaded (law 9).
    """
    report = dict(report or {})
    corpus = dict(report.get("corpus") or {})
    history = dict(report.get("history") or {})
    best = dict(report.get("best") or {})
    key = _checked_kind(kind)

    metrics = {}
    for name in REPORT_METRICS[key]:
        if name in best:
            metrics[name] = best[name]
        elif name in history:
            metrics[name] = history[name]
    if "val_samples" not in metrics and "images" in best:
        metrics["val_images"] = best["images"]

    keypoints = tuple(corpus.get("keypoints") or ()) if key == "pose" else ()
    classes = tuple(corpus.get("classes") or ()) if key == "detector" else ()
    return {
        "preset": str(report.get("preset") or ""),
        "trained_at": "",               # the payload's own claim; the report has none
        "classes": classes,
        "keypoints": keypoints,
        "metrics": metrics,
    }


def _normalised(path) -> str:
    """One checkpoint, one string: every path is **resolved** before it is stored or looked
    up, so a symlinked `models/` directory cannot put the same file in the registry twice
    (and `_same_file` can compare two paths by name)."""
    return str(Path(path).expanduser().resolve())


def _same_file(path, other) -> bool:
    """Whether two paths name the same checkpoint (``other`` may be ``None``)."""
    if not path or not other:
        return False
    try:
        return _normalised(path) == _normalised(other)
    except OSError:                                  # pragma: no cover - unreadable path
        return str(path) == str(other)


def _report_takes_over(conn, kind, report) -> bool:
    """Whether the report's **best** checkpoint should be the active one of its kind.

    Decided on what the files *are*, never on their timestamps (a run writes its two
    checkpoints within the same second):

    * nothing is active yet - yes: this is how the registry learns about a model nobody has
      opened in the app (`SR-PREDICT-018`);
    * the active checkpoint is the report's ``checkpoint`` already - no, nothing to do;
    * the active checkpoint is the report's ``last_checkpoint`` (the **final** epoch of the
      same run) - yes: the report says which file earned the numbers, and leaving the final
      epoch in force is the very confusion ``mlContext.md`` law 9 exists to prevent;
    * the active checkpoint is one the report does not describe - **no**: that file was
      loaded by hand from somewhere else, and that choice belongs to the operator. A run
      that writes new file names is *offered* in the menu, not imposed.
    """
    current = active(conn, kind)
    if current is None:
        return True
    if _same_file(current.path, report.get("checkpoint")):
        return False
    described = (_same_file(current.path, report.get("last_checkpoint")),
                 _same_file(current.path, report.get("checkpoint")))
    return any(described)


def sync_from_reports(conn, directory=None) -> list:
    """Register what the **training reports** name, best file first (`make train`).

    A training run writes a report next to its checkpoints - the run's own account of which
    checkpoint is *best*, what it scored, on how many validation images and at which epoch.
    Reading it is how the registry learns about a model nobody has loaded in the app yet,
    and it keeps PyTorch out of the app's start-up path (this module stays torch-free,
    ``SR-CORE-011``).

    * the report's ``checkpoint`` (**best**) and ``last_checkpoint`` (**final**) are both
      registered, so the menu offers both - they are separate files with separate numbers
      (``mlContext.md`` law 9), and activating the best is the default;
    * a missing report is normal (nothing trained yet) and a missing *checkpoint* is
      skipped rather than registered as a path that is not there;
    * a report that cannot be read raises: it is a file the lab believes in.
    """
    directory = Path(directory) if directory is not None else _models_dir()
    registered = []
    for kind, filename in REPORT_FILES.items():
        report_path = directory / filename
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError) as error:
            raise ModelError(f"{report_path} cannot be read: {error}") from error

        for path, is_best in ((report.get("checkpoint"), True),
                              (report.get("last_checkpoint"), False)):
            if not path or not Path(path).exists():
                continue
            fields = record_from_report(kind, report)
            if not is_best:
                # The report's numbers describe the **best** file; the final epoch's own
                # weights are registered without them (law 9).
                fields["metrics"] = {}
            registered.append(register(conn, kind, _normalised(path),
                                       activate=False, **fields))
        best = report.get("checkpoint")
        if best and Path(best).exists() and _report_takes_over(conn, kind, report):
            activate_model(conn, kind, str(Path(best).resolve()))
    return registered


def models_dir() -> Path:
    """Where the lab's checkpoints live unless a caller says otherwise (`daphnia.paths`)."""
    from daphnia import paths

    return Path(paths.MODELS_DIR)


def checkpoints_in(kind, directory=None) -> list:
    """Every checkpoint of ``kind`` **available** in the models directory, by name.

    The registry says what is *in force*; this says what is *there* - a file a training run
    has just written, or one copied in by hand. A file belongs to a kind by its **name**
    (``rtmdet*`` / ``rtmpose*``, `models.KIND_PREFIXES`): the surface that lists them is a
    menu, and reading a checkpoint's bytes to decide where to list it would load PyTorch to
    draw a menu (`SR-CORE-011`).
    """
    directory = Path(directory) if directory is not None else models_dir()
    prefix = KIND_PREFIXES[_checked_kind(kind)]
    if not directory.is_dir():
        return []
    return sorted((path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
                   and path.name.lower().startswith(prefix)),
                  key=lambda path: path.name)


def _models_dir() -> Path:
    """Where checkpoints and their reports live (`daphnia.paths`, imported lazily)."""
    return models_dir()


def register(conn, kind, path, *, preset="", trained_at="", classes=(),
             keypoints=(), metrics=None, activate=True) -> ModelRecord:
    """Record a checkpoint, optionally making it the active one of its kind.

    Registering the *same path* again updates the row: a re-trained checkpoint keeps its
    identity (the file) and gets the newer facts (the preset, the metrics). Nothing is
    deleted here - the registry is the list of what the lab has, not a cache.
    """
    key = _checked_kind(kind)
    target = _normalised(path)
    if not target or target == ".":
        raise ModelError(f"a {key} needs a checkpoint path")
    conn.execute(
        "INSERT INTO models (kind, path, name, preset, trained_at, classes, keypoints,"
        " metrics_json, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)"
        " ON CONFLICT (kind, path) DO UPDATE SET"
        "   name = excluded.name, preset = excluded.preset,"
        "   trained_at = excluded.trained_at, classes = excluded.classes,"
        "   keypoints = excluded.keypoints, metrics_json = excluded.metrics_json",
        (key, target, Path(target).name, str(preset or ""), str(trained_at or ""),
         ",".join(classes or ()), ",".join(keypoints or ()),
         json.dumps(dict(metrics or {}), sort_keys=True)))
    if activate:
        activate_model(conn, key, target)
    return record(conn, key, target)


def record(conn, kind, path) -> ModelRecord:
    """One checkpoint as the registry knows it, or ``None``."""
    row = conn.execute("SELECT * FROM models WHERE kind = ? AND path = ?",
                       (_checked_kind(kind), _normalised(path))).fetchone()
    return _row_to_record(row) if row is not None else None


def records(conn, kind=None) -> list:
    """Every registered checkpoint, active first, then newest - optionally one kind."""
    if kind is None:
        rows = conn.execute("SELECT * FROM models ORDER BY kind, is_active DESC,"
                            " added_at DESC, name").fetchall()
    else:
        rows = conn.execute("SELECT * FROM models WHERE kind = ?"
                            " ORDER BY is_active DESC, added_at DESC, name",
                            (_checked_kind(kind),)).fetchall()
    return [_row_to_record(row) for row in rows]


def active(conn, kind) -> ModelRecord:
    """The checkpoint in force for ``kind``, or ``None`` when none is registered."""
    rows = conn.execute("SELECT * FROM models WHERE kind = ? AND is_active = 1"
                        " LIMIT 1", (_checked_kind(kind),)).fetchall()
    return _row_to_record(rows[0]) if rows else None


def activate_model(conn, kind, path) -> ModelRecord:
    """Make one registered checkpoint the active one of its kind.

    Refuses an unknown path rather than inventing a row: *"the active model"* must be
    something the registry was actually told about (no silent fallback).
    """
    key = _checked_kind(kind)
    target = _normalised(path)
    if record(conn, key, target) is None:
        raise ModelError(f"{target} is not registered as a {key}. Load it first.")
    conn.execute("UPDATE models SET is_active = 0 WHERE kind = ?", (key,))
    conn.execute("UPDATE models SET is_active = 1 WHERE kind = ? AND path = ?",
                 (key, target))
    return record(conn, key, target)


def forget(conn, kind, path) -> bool:
    """Drop a checkpoint from the registry; returns whether a row was removed.

    Housekeeping only - it is *not* how a model stops being used (that is
    :func:`activate_model`), and it never touches the file on disk.
    """
    cursor = conn.execute("DELETE FROM models WHERE kind = ? AND path = ?",
                          (_checked_kind(kind), _normalised(path)))
    return bool(cursor.rowcount)
