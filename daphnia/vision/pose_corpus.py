"""Reading a Dataset export's **keypoints** - the one input a pose run needs.

:mod:`daphnia.vision.export` reads a Dataset export for its *boxes*; this module
reads the same folder for the *keypoints* beside them. Both read the one
``labels.json`` the Dataset surface writes, and both are Qt-free **and PyTorch-free**,
so the corpus contract is testable on any machine - the property that keeps a
coordinate bug from being discovered only by training.

## The group, and why it is *not* the directory here (law 4)

The Dataset export copies every frame into a flat ``images/`` folder, so the parent
directory of a specimen's original path is ``images/`` for **every** frame and
``Export``'s directory-based group collapses to a single value. This corpus instead
encodes the capture in the **frame stem**: ``b01_f00_35755477`` is frame ``f00`` of
capture ``b01``. :func:`_capture_of` recovers the first ``_``-token as the group, so
a split can still be held out by capture - the whole point of law 4.

## What a sample is

One **specimen** with at least one keypoint and a bounding box: the box is what the
detector predicts and what the pose crop is built from, and the keypoints are the
answer. A specimen with a box but no keypoints is a detection row, not a pose row,
and is skipped; an all-coincident keypoint set is a degenerate label and is dropped.
"""

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from daphnia.vision.export import (
    DEFAULT_GROUP_WEIGHTS,
    SPLITS,
    _resolve_image,
)

#: The file that carries the machine-readable truth.
LABELS_FILE = "labels.json"

#: The manifest beside it, when the exporter wrote one.
MANIFEST_FILE = "manifest.json"

#: The trailing token an exported copy carries (``<stem>_<8 hex>.png``). Stripped
#: before the capture is read, because the exporter renames the copies.
_COPY_SUFFIX = re.compile(r"_[0-9a-f]{8}$")

#: A keypoint pair closer than this (source pixels) is a double click, not a body.
MIN_KEYPOINT_SPAN = 1.0


class PoseError(ValueError):
    """A folder is not a Dataset export with keypoints a pose model can learn from."""


@dataclass(frozen=True)
class PoseSample:
    """One labelled specimen: where it is, and the points inside it.

    ``points`` is ``((name, x, y), ...)`` in **source-image pixels**, in the schema's
    order, and ``bbox`` is ``(x1, y1, x2, y2)`` in the same frame - the frame every
    other module agrees is *the* frame of reference (``mlContext.md`` law 1).
    """

    image: Path
    relative: str                  # 'images/xxx.png', how the export spells it
    split: str
    group: str                     # the capture the frame came from (leak-safe)
    points: tuple                  # ((name, x, y), ...) in source pixels
    bbox: tuple                    # (x1, y1, x2, y2) in source pixels
    specimen_index: int
    image_width: int = 0
    image_height: int = 0
    source: str = "manual"
    traits: dict = field(default_factory=dict)

    @property
    def names(self) -> tuple:
        """The keypoint names of this sample, in order."""
        return tuple(name for name, _x, _y in self.points)

    def point(self, name) -> tuple:
        """``(x, y)`` of one named keypoint, in source pixels."""
        for key, x, y in self.points:
            if key == name:
                return float(x), float(y)
        raise KeyError(f"{name!r} is not a keypoint of this specimen "
                       f"(has {', '.join(self.names)})")

    def as_dict(self) -> dict:
        """``{name: (x, y)}`` - the keypoints keyed by name."""
        return {name: (float(x), float(y)) for name, x, y in self.points}

    def length_um(self):
        """The stored body length in micrometres, when the corpus carries it."""
        for trait in (self.traits or {}).values():
            if isinstance(trait, dict) and trait.get("length_um") is not None:
                return float(trait["length_um"])
        return None

    def px_per_um(self):
        """``length_px / length_um`` of this specimen - pixels per micrometre.

        This is the meaning the database's ``measurements.px_per_um`` column carries
        (a small number, e.g. 0.0134). For the inverse - the micrometres one pixel
        spans, e.g. 74.8 - use :meth:`um_per_px`.
        """
        for trait in (self.traits or {}).values():
            if not isinstance(trait, dict):
                continue
            length_px = trait.get("length_px")
            length_um = trait.get("length_um")
            if length_px and length_um:
                return float(length_px) / float(length_um)
        return None

    def um_per_px(self):
        """``length_um / length_px`` of this specimen - micrometres per pixel."""
        for trait in (self.traits or {}).values():
            if not isinstance(trait, dict):
                continue
            length_px = trait.get("length_px")
            length_um = trait.get("length_um")
            if length_px and length_um:
                return float(length_um) / float(length_px)
        return None


@dataclass(frozen=True)
class PoseCorpus:
    """One Dataset export, read from disk into keypoint samples."""

    folder: Path
    dataset: dict                  # the 'dataset' block of labels.json
    schema: dict                   # the 'schema' block of labels.json
    specimens: tuple               # the raw per-specimen records
    splits: dict                   # name -> tuple of export-relative image names
    stats: dict                    # the manifest's 'stats', when present
    samples: tuple = ()
    names: tuple = ()              # the keypoint names this corpus actually labels
    dropped: dict = field(default_factory=dict)     # why rows were skipped

    # -- reading -----------------------------------------------------------
    @classmethod
    def read(cls, folder) -> "PoseCorpus":
        """Read ``folder`` as a Dataset export with keypoints.

        Raises:
            PoseError: the folder does not exist, has no ``labels.json``, was
                written by an older exporter, or labels **no keypoints** at all.
        """
        folder = Path(folder).expanduser()
        if not folder.is_dir():
            raise PoseError(
                f"not a folder: {folder}\n"
                "Export one first: python run.py dataset  ->  Export dataset...")
        label_path = folder / LABELS_FILE
        if not label_path.exists():
            raise PoseError(
                f"{label_path.name} is missing in {folder} - this is not a dataset "
                "export. Run the Dataset surface and export the corpus.")
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PoseError(f"could not read {label_path}: {error}") from error
        if not isinstance(payload, dict) or "specimens" not in payload:
            raise PoseError(
                f"{label_path} has no 'specimens' block - it was written by an "
                "older export format. Re-export the dataset.")

        manifest = {}
        manifest_path = folder / MANIFEST_FILE
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}                       # a manifest is a convenience

        corpus = cls(
            folder=folder,
            dataset=dict(payload.get("dataset") or {}),
            schema=dict(payload.get("schema") or {}),
            specimens=tuple(payload.get("specimens") or ()),
            splits=_read_splits(folder, payload),
            stats=dict(manifest.get("stats") or {}))
        samples, dropped = corpus._samples_from_specimens()
        names = _names_of(samples, corpus.schema)
        if not samples or not names:
            raise PoseError(
                f"{label_path} carries no labelled keypoint. This corpus can train a "
                "detector; a pose model needs a dataset whose model has keypoints "
                "(e.g. 'Pose - head & tail (2 points)').")
        return replace(corpus, samples=tuple(samples), names=tuple(names),
                       dropped=dropped)

    # -- views -------------------------------------------------------------
    def samples_of(self, split=None) -> tuple:
        """Every sample, or the samples of one split."""
        if split is None:
            return self.samples
        return tuple(sample for sample in self.samples if sample.split == split)

    def frames(self, split=None) -> tuple:
        """Distinct frames (export-relative names) that carry a labelled sample."""
        wanted = self.samples_of(split)
        return tuple(sorted({sample.relative for sample in wanted}))

    def groups(self, split=None) -> dict:
        """``{group: {splits it appears in}}`` - the leak-safety view."""
        found = {}
        for sample in self.samples_of(split):
            found.setdefault(sample.group, set()).add(sample.split)
        return found

    def leaking_groups(self) -> tuple:
        """Captures whose frames the export's own split spread over more than one."""
        return tuple(sorted(group for group, splits in self.groups().items()
                            if len(splits) > 1))

    def regroup(self, weights=DEFAULT_GROUP_WEIGHTS) -> "PoseCorpus":
        """A copy whose splits are rebuilt **by capture**, deterministically.

        This corpus has too few captures for the export's hash-bucket regroup -
        :meth:`daphnia.vision.export.Export.regroup` - to be usable: with four
        captures (``b01..b04``) every hash landed in ``train`` and the validation set
        came out empty, which the trainer rightly refuses. So groups are sorted by
        key and cut proportionally instead, with **every split that can be non-empty
        given at least one capture**. Sorted-and-cut is as reproducible as a hash -
        two runs produce the same partition, which is what makes a score checkable -
        and it cannot starve the validation set. Fewer than three captures is
        reported by :meth:`warnings` rather than papered over.
        """
        names = sorted(self.groups())
        train_count, val_count, test_count = _split_counts(len(names), weights)
        assignment = {}
        for index, name in enumerate(names):
            if index < train_count:
                assignment[name] = "train"
            elif index < train_count + val_count:
                assignment[name] = "val"
            else:
                assignment[name] = "test"
        members = {name: [] for name in SPLITS}
        for sample in self.samples:
            members[assignment[sample.group]].append(sample.relative)
        splits = {name: tuple(sorted(set(names))) for name, names in members.items()}
        return replace(
            self, splits=splits,
            samples=tuple(replace(sample, split=assignment[sample.group])
                          for sample in self.samples))

    # -- honesty -----------------------------------------------------------
    def warnings(self) -> tuple:
        """Everything about this corpus a pose run should be told."""
        notes = []
        leaking = self.leaking_groups()
        if leaking:
            shown = ", ".join(leaking[:4]) + (", ..." if len(leaking) > 4 else "")
            notes.append(
                f"{len(leaking)} of {len(self.groups())} capture(s) are split across "
                f"train/val/test ({shown}): frames of one capture show the same "
                "animals seconds apart, so the export's own split reports an "
                "optimistic score. Train with --split-by group for a held-out one.")

        degenerate = self.dropped.get("degenerate", 0)
        if degenerate:
            notes.append(
                f"{degenerate} specimen(s) have coincident keypoints (a double "
                "click, or a length of 0) and are dropped. Re-open the frame in the "
                "Dataset surface and re-place them.")

        no_box = self.dropped.get("no_box", 0)
        if no_box:
            notes.append(
                f"{no_box} specimen(s) have keypoints but no bounding box and are "
                "dropped: the pose crop is built from the box. Label the box too.")

        empty = [name for name in SPLITS if not self.samples_of(name)]
        if empty:
            notes.append(f"split(s) {', '.join(empty)} are empty for this corpus - "
                         "val/test numbers from them would mean nothing.")

        if len(self.names) < 2:
            notes.append(
                f"only {len(self.names)} keypoint name(s) are labelled "
                f"({', '.join(self.names)}): a length needs two.")
        return tuple(notes)

    def stats_summary(self) -> dict:
        """Counts that describe the corpus, for a report or a status label."""
        per_split = {}
        for name in SPLITS:
            wanted = self.samples_of(name)
            per_split[name] = {"images": len(self.frames(name)),
                               "samples": len(wanted),
                               "groups": len({sample.group for sample in wanted})}
        return {"dataset": self.dataset.get("name"),
                "schema": self.schema.get("key"),
                "keypoints": list(self.names),
                "samples": len(self.samples),
                "images": len(self.frames()),
                "groups": len(self.groups()),
                "dropped": dict(self.dropped),
                "splits": per_split}

    def describe(self) -> str:
        """A short human summary, for a log line or a status label."""
        summary = self.stats_summary()
        lines = [f"{summary['dataset'] or self.folder.name}: {summary['samples']} "
                 f"keypoint sample(s) on {summary['images']} image(s) from "
                 f"{summary['groups']} capture(s)"]
        lines.append(f"  keypoints   {', '.join(summary['keypoints'])}")
        for name in SPLITS:
            counts = summary["splits"][name]
            lines.append(f"  {name:<5} {counts['images']:>4} image(s)  "
                         f"{counts['samples']:>4} sample(s)  "
                         f"{counts['groups']:>2} capture(s)")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------
    def image_path(self, relative) -> Path:
        """Where one export-relative frame actually is on this machine."""
        found = self.folder / relative
        if found.exists():
            return found
        original = next((str(record.get("file")) for record in self.specimens
                         if record.get("image") == relative and record.get("file")),
                        None)
        return _resolve_image(self.folder, relative, original or "")

    def _samples_from_specimens(self):
        """Turn the per-specimen records into :class:`PoseSample` values.

        Counts *why* each row was skipped, so the warning can name the reason rather
        than leaving the 141 box-only rows of this corpus unexplained.
        """
        samples, dropped = [], {}
        for record in self.specimens:
            points = _points_of(record)
            if not points:
                continue                        # a detection row, not a pose row
            relative = record.get("image")
            bbox = record.get("bbox")
            if relative is None or bbox is None or len(bbox) != 4:
                dropped["no_box"] = dropped.get("no_box", 0) + 1
                continue
            if _is_degenerate(points):
                dropped["degenerate"] = dropped.get("degenerate", 0) + 1
                continue
            values = [float(value) for value in bbox]
            x1, y1, x2, y2 = (min(values[0], values[2]), min(values[1], values[3]),
                              max(values[0], values[2]), max(values[1], values[3]))
            samples.append(PoseSample(
                image=self.image_path(relative),
                relative=str(relative),
                split=record.get("split") or SPLITS[0],
                group=_capture_of(record),
                points=points,
                bbox=(x1, y1, x2, y2),
                specimen_index=int(record.get("specimen_index") or 1),
                image_width=int(record.get("image_width") or 0),
                image_height=int(record.get("image_height") or 0),
                source=str(record.get("source") or "manual"),
                traits=dict(record.get("traits") or {})))
        samples.sort(key=lambda sample: (sample.relative, sample.specimen_index))
        return samples, dropped


def _points_of(record) -> tuple:
    """``((name, x, y), ...)`` of one specimen record, decimals preserved."""
    raw = record.get("keypoints") or {}
    points = []
    for name, value in raw.items():
        try:
            x, y = float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError):
            continue
        points.append((name, x, y))
    return tuple(points)


def _is_degenerate(points) -> bool:
    """True when every keypoint coincides - a double click, not a body."""
    xs = [x for _name, x, _y in points]
    ys = [y for _name, _x, y in points]
    return ((max(xs) - min(xs)) < MIN_KEYPOINT_SPAN
            and (max(ys) - min(ys)) < MIN_KEYPOINT_SPAN)


def _names_of(samples, schema) -> tuple:
    """The keypoint names in schema order, then any extras the corpus carries."""
    present = {name for sample in samples for name in sample.names}
    ordered = [name for name in (schema.get("export_names") or ()) if name in present]
    ordered += sorted(present - set(ordered))
    return tuple(ordered)


def _capture_of(record) -> str:
    """The leak-safe group: the capture prefix of the frame stem.

    ``b01_f00_35755477_ee94ef9c.png`` -> ``b01`` - the exported copy's hash suffix
    is stripped first, then the token before the first ``_`` is the capture. When a
    name carries no separator the whole stem (or, failing that, the parent
    directory) is the best available key, which never *invents* a leak.
    """
    original = record.get("file")
    stem = Path(str(original)).stem if original \
        else Path(str(record.get("image") or "")).stem
    stem = _COPY_SUFFIX.sub("", stem)
    token = stem.split("_")[0]
    if token and token != stem:
        return token
    if original:
        parent = Path(str(original)).parent.name
        if parent:
            return parent
    return token or "unknown"


def _split_counts(count, weights=DEFAULT_GROUP_WEIGHTS):
    """``(train, val, test)`` capture counts for ``count`` groups.

    Proportional to ``weights`` but every split that can be non-empty gets one
    capture, so a four-capture corpus yields a real validation set instead of an
    empty one. With fewer than three captures nothing can be held out and all of
    them go to ``train`` - :meth:`PoseCorpus.warnings` says so.
    """
    if count <= 0:
        return 0, 0, 0
    if count < 3:
        return count, 0, 0
    total = float(sum(weights)) or 1.0
    val = max(1, int(round(count * weights[1] / total)))
    test = max(1, int(round(count * weights[2] / total)))
    if val + test >= count:
        remaining = count - 1
        val = max(1, remaining // 2)
        test = max(1, remaining - val) if remaining - val >= 1 else 0
    return count - val - test, val, test


def _read_splits(folder, payload) -> dict:
    """Read ``splits/*.txt``; fall back to the split stored per specimen."""
    splits = {}
    for name in SPLITS:
        path = folder / "splits" / f"{name}.txt"
        if not path.exists():
            continue
        try:
            names = [line.strip() for line
                     in path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        except OSError:
            names = []
        splits[name] = tuple(names)
    if any(splits.values()):
        for name in SPLITS:
            splits.setdefault(name, ())
        return splits

    rebuilt = {name: set() for name in SPLITS}
    for record in payload.get("specimens") or ():
        if record.get("image"):
            rebuilt.setdefault(record.get("split") or SPLITS[0], set()).add(
                record["image"])
    return {name: tuple(sorted(names)) for name, names in rebuilt.items()}
