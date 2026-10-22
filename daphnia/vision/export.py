"""Reading a Dataset export - the one input a training run needs.

The Dataset surface's ``export_dataset()`` writes one folder per model's corpus
(``SR-DATASET-009`` / ``SR-DATASET-018``)::

    <export>/
      images/                   the frames, copied once, named <stem>_<hash>.png
      labels.json               the machine-readable truth: schema + per specimen
      boxes.csv                 the same boxes as CSV, for a spreadsheet
      labels.csv                the keypoint rows (empty for a boxes-only model)
      splits/{train,val,test}.txt
      manifest.json             what was written, when, and how much

This module reads exactly that folder and turns it into what a detector needs -
**frames with their boxes** - and refuses loudly when the folder is not what it
claims to be. It imports no Qt and no PyTorch, so a training run, the Predict
surface and the tests all read a corpus the same way.

## Why a *group* matters

``splits/*.txt`` are built by hashing the **image path** (``SR-DATASET-010``), so
two frames of one capture - the same plate, the same animals, seconds apart - can
land in different splits. A validation score then measures memorisation, not
detection. The export records every specimen's **original file path**, and the
parent directory of that path *is* the observation (one capture session), so the
leak-safe group is recovered from the data instead of guessed from a file name:
:meth:`Export.groups` and :meth:`Export.regroup` group by capture, and
:meth:`Export.warnings` reports the leak instead of leaving it to be discovered as
a suspiciously good score.

## What a box is

One **box is one individual**, not one frame and not one capture: a frame holds as
many boxes as it holds animals (``SR-DATASET-004``), each with its own
``specimen_index``. ``Box.group`` is therefore *not* the box's identity - it is the
capture the frame came from, which is the unit a split must respect.
"""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from daphnia import annotation

#: The split names the Dataset surface writes.
SPLITS = ("train", "val", "test")

#: Files and folders inside an export.
LABELS_FILE = "labels.json"
MANIFEST_FILE = "manifest.json"
IMAGES_DIR = "images"
SPLITS_DIR = "splits"

#: The relative share of a corpus each split gets when it is regrouped.
DEFAULT_GROUP_WEIGHTS = (70, 15, 15)

#: A box smaller than this (in pixels) cannot be a specimen: the labeller refuses
#: anything under ``annotation.MIN_BOX_PIXELS``, so anything smaller here is
#: corruption rather than a small animal.
MIN_BOX_PIXELS = annotation.MIN_BOX_PIXELS


class ExportError(ValueError):
    """A folder is not a Dataset export, or not one a detector can learn from."""


@dataclass(frozen=True)
class Box:
    """One labelled specimen in one frame."""

    image: Path                # absolute path of the image on disk
    relative: str              # 'images/xxx.png' - how the export spells it
    split: str                 # the export's own split for this frame
    group: str                 # the observation the frame came from (leak-safe)
    bbox: tuple                # (x1, y1, x2, y2) in source-image pixels
    specimen_index: int
    source: str = "manual"     # 'manual' | 'model'
    image_width: int = 0
    image_height: int = 0

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    def outside(self) -> bool:
        """True when the box leaves its image, or is too small to be a specimen."""
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 > self.image_width or y2 > self.image_height:
            return True
        return self.width < MIN_BOX_PIXELS or self.height < MIN_BOX_PIXELS


@dataclass(frozen=True)
class Export:
    """One Dataset export, read from disk into frames and boxes.

    A read model in the spirit of :meth:`daphnia.study.StudySpec.from_row`: every
    consumer asks this object rather than re-parsing ``labels.json``.
    """

    folder: Path
    dataset: dict                  # the 'dataset' block of labels.json
    schema: dict                   # the 'schema' block of labels.json
    specimens: tuple               # the raw per-specimen records
    splits: dict                   # name -> tuple of export-relative image names
    stats: dict                    # the manifest's 'stats', when present
    _boxes: tuple = ()

    # -- reading -----------------------------------------------------------
    @classmethod
    def read(cls, folder) -> "Export":
        """Read ``folder`` as a Dataset export.

        Raises:
            ExportError: the folder does not exist, has no ``labels.json``, or
                carries a label file that cannot be interpreted.
        """
        folder = Path(folder).expanduser()
        if not folder.is_dir():
            raise ExportError(
                f"not a folder: {folder}\n"
                "Export one first: python run.py dataset  ->  Export dataset...")
        label_path = folder / LABELS_FILE
        if not label_path.exists():
            raise ExportError(
                f"{label_path.name} is missing in {folder} - this is not a dataset "
                "export. Run the Dataset surface and export the corpus (it writes "
                "images/, labels.json, boxes.csv and splits/).")
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ExportError(f"could not read {label_path}: {error}") from error
        if not isinstance(payload, dict) or "specimens" not in payload:
            raise ExportError(
                f"{label_path} has no 'specimens' block - it was written by an "
                "older export format. Re-export the dataset.")

        manifest = {}
        manifest_path = folder / MANIFEST_FILE
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}                       # a manifest is a convenience

        export = cls(
            folder=folder,
            dataset=dict(payload.get("dataset") or {}),
            schema=dict(payload.get("schema") or {}),
            specimens=tuple(payload.get("specimens") or ()),
            splits=_read_splits(folder, payload),
            stats=dict(manifest.get("stats") or {}))
        return replace(export, _boxes=export._boxes_from_specimens())

    @classmethod
    def from_database(cls, conn, dataset_id, folder) -> "Export":
        """Export ``dataset_id`` into ``folder``, then read it back.

        The Predict surface and the trainer both work from a *folder*, so this is
        the one place that couples the store to the reader - and it keeps
        ``maintenance.export_dataset`` the single writer of an export.
        """
        from daphnia import maintenance

        maintenance.export_dataset(conn, dataset_id, folder)
        return cls.read(folder)


    # -- frames ------------------------------------------------------------
    def image_path(self, relative) -> Path:
        """Absolute path of one export-relative image name."""
        return self.folder / relative

    def _frames(self) -> tuple:
        """Every export-relative image name, in a stable order."""
        names = set()
        for paths in self.splits.values():
            names.update(paths)
        for record in self.specimens:
            if record.get("image"):
                names.add(record["image"])
        return tuple(sorted(names))

    def frames(self, existing_only=True) -> tuple:
        """Absolute paths of every frame the corpus refers to.

        ``existing_only`` skips frames whose image was missing when the export
        ran (the exporter keeps the reference and writes ``image: null``), so a
        loader never trips over a path that is not there.
        """
        paths = [self.image_path(name) for name in self._frames()]
        if existing_only:
            paths = [path for path in paths if path.exists()]
        return tuple(paths)

    def boxes(self) -> tuple:
        """Every labelled box, in a stable order."""
        return tuple(sorted(self._boxes,
                            key=lambda box: (box.relative, box.specimen_index)))

    def boxes_of(self, relative) -> tuple:
        """The boxes of one export-relative frame name."""
        return tuple(box for box in self.boxes() if box.relative == relative)

    def images_in(self, split) -> tuple:
        """Absolute paths of the frames the export put in ``split``."""
        paths = [self.image_path(name) for name in self.splits.get(split, ())]
        return tuple(path for path in paths if path.exists())

    def boxes_in(self, split) -> tuple:
        """Every box of ``split``."""
        return tuple(box for box in self.boxes() if box.split == split)

    # -- grouping ----------------------------------------------------------
    def groups(self, split=None) -> dict:
        """``{group: {splits it appears in}}`` - the leak-safety view.

        The group is the **capture** a frame came from - the observation directory
        of the specimen's original file path - which is what makes the frames of
        one session recognise each other even after the exporter renamed the
        copies. It is deliberately *not* the individual: identity across frames
        ("this is animal 3 again tomorrow") is not recorded yet
        (``SR-DATASET-010``), so the capture is the tightest unit a split can be
        held out at.
        """
        found = {}
        for box in self.boxes():
            if split is not None and box.split != split:
                continue
            found.setdefault(box.group, set()).add(box.split)
        return found

    def leaking_groups(self) -> tuple:
        """Captures whose frames are spread over more than one split.

        Those are exactly the sessions a hash-by-image split tears apart - and the
        reason a regrouped corpus is the honest way to report a score.
        """
        return tuple(sorted(group for group, splits in self.groups().items()
                            if len(splits) > 1))


    def regroup(self, weights=DEFAULT_GROUP_WEIGHTS) -> "Export":
        """Return a copy whose splits are rebuilt **by group**, not by image.

        Deterministic - a SHA-1 of the group key picks the bucket, exactly as the
        exporter does for one image - so two runs on the same corpus produce the
        same partition and a reported score can be reproduced. With few
        observations a split can end up empty; :meth:`warnings` says so instead of
        the run quietly reporting a number from nowhere.
        """
        mapping = {group: _bucket_split(group, weights)
                   for group in sorted(self.groups())}
        members = {name: [] for name in SPLITS}
        for box in self.boxes():
            members[mapping[box.group]].append(box.relative)
        splits = {name: tuple(sorted(set(names))) for name, names in members.items()}

        # Frames with no box belong to no group: keep them in train, so a frame
        # that merely has no box yet is not silently dropped from the corpus.
        boxed = set()
        for names in splits.values():
            boxed.update(names)
        unboxed = [name for name in self._frames() if name not in boxed]
        if unboxed:
            splits["train"] = tuple(sorted(set(splits["train"]) | set(unboxed)))

        return replace(self, splits=splits,
                       _boxes=tuple(replace(box, split=mapping[box.group])
                                    for box in self._boxes))

    # -- honesty -----------------------------------------------------------
    def warnings(self) -> tuple:
        """Everything about this corpus a training run should be told.

        Not an error list: a warning is a *reason to be careful*, and this corpus
        can still train a useful model. It exists so a suspiciously good score is
        explained by the data rather than discovered later.
        """
        notes = []
        if not self._boxes:
            notes.append(
                "The export carries no bounding box. Build a dataset whose model is "
                "'Bounding boxes only' (or 'Pose + bounding box') before training a "
                "detector.")
            return tuple(notes)

        leaking = self.leaking_groups()
        if leaking:
            shown = ", ".join(leaking[:4]) + (", ..." if len(leaking) > 4 else "")
            notes.append(
                f"{len(leaking)} of {len(self.groups())} capture(s) are split across "
                f"train/val/test ({shown}): frames of one capture show the same "
                "plate seconds apart, so the export's own split reports an "
                "optimistic score. Train with --split-by group for a held-out one.")

        flat = [box for box in self._boxes if box.outside()]
        if flat:
            example = flat[0]
            notes.append(
                f"{len(flat)} box(es) leave their image or are smaller than "
                f"{MIN_BOX_PIXELS:.0f} px and are dropped from training "
                f"(e.g. {example.relative} specimen {example.specimen_index}: "
                f"{tuple(round(value, 1) for value in example.bbox)}). Re-open the "
                "frame in the Dataset surface and redraw it.")

        missing = len(self._frames()) - len(self.frames())
        if missing:
            notes.append(f"{missing} frame(s) named by the labels have no image in "
                         "images/ and are skipped.")

        empty = [name for name in SPLITS if not self.images_in(name)]
        if empty:
            notes.append(f"split(s) {', '.join(empty)} are empty for this corpus - "
                         "val/test numbers from it would mean nothing.")

        sizes = {(box.image_width, box.image_height) for box in self._boxes}
        if len(sizes) > 1:
            notes.append(
                f"{len(sizes)} different image sizes in one corpus "
                f"({sorted(sizes)}). Training letterboxes them all to one size, so "
                "keep the optics identical between sessions or the scale becomes "
                "one more thing the model must learn.")
        return tuple(notes)

    def stats_summary(self) -> dict:
        """Counts that describe the corpus, for a report or a status label."""
        per_split = {}
        for name in SPLITS:
            boxes = self.boxes_in(name)
            per_split[name] = {"images": len(self.images_in(name)),
                               "boxes": len(boxes),
                               "groups": len({box.group for box in boxes})}
        areas = [box.area for box in self._boxes]
        return {"dataset": self.dataset.get("name"),
                "schema": self.schema.get("key"),
                "keypoints": len(self.schema.get("keypoints") or ()),
                "boxes": len(self._boxes),
                "images": len(self.frames()),
                "groups": len(self.groups()),
                "mean_box_area": (sum(areas) / len(areas)) if areas else 0.0,
                "splits": per_split}

    def describe(self) -> str:
        """A short human summary, for a log line or a status label."""
        summary = self.stats_summary()
        lines = [f"{summary['dataset'] or self.folder.name}: {summary['boxes']} "
                 f"box(es) on {summary['images']} image(s) from "
                 f"{summary['groups']} capture(s)"]
        for name in SPLITS:
            counts = summary["splits"][name]
            lines.append(f"  {name:<5} {counts['images']:>4} image(s)  "
                         f"{counts['boxes']:>4} box(es)  "
                         f"{counts['groups']:>2} capture(s)")
        return "\n".join(lines)


    # -- internals ---------------------------------------------------------
    def _boxes_from_specimens(self) -> tuple:
        """Turn the per-specimen records into :class:`Box` values."""
        boxes = []
        for record in self.specimens:
            bbox = record.get("bbox")
            relative = record.get("image")
            if bbox is None or relative is None:
                continue                        # keypoint-only, or a lost image
            values = [float(value) for value in bbox]
            if len(values) != 4:
                continue
            x1, y1, x2, y2 = (min(values[0], values[2]), min(values[1], values[3]),
                              max(values[0], values[2]), max(values[1], values[3]))
            boxes.append(Box(
                image=self.image_path(relative),
                relative=relative,
                split=record.get("split") or SPLITS[0],
                group=_group_of(record),
                bbox=(x1, y1, x2, y2),
                specimen_index=int(record.get("specimen_index") or 1),
                source=str(record.get("source") or "manual"),
                image_width=int(record.get("image_width") or 0),
                image_height=int(record.get("image_height") or 0)))
        return tuple(boxes)


def _read_splits(folder, payload) -> dict:
    """Read ``splits/*.txt``; fall back to the split stored per specimen."""
    splits = {}
    for name in SPLITS:
        path = folder / SPLITS_DIR / f"{name}.txt"
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

    # No split files at all (an older or hand-made export): rebuild them from the
    # per-specimen records, so the reader never depends on the extra files.
    rebuilt = {name: set() for name in SPLITS}
    for record in payload.get("specimens") or ():
        if record.get("image"):
            rebuilt.setdefault(record.get("split") or SPLITS[0], set()).add(
                record["image"])
    return {name: tuple(sorted(names)) for name, names in rebuilt.items()}


def _group_of(record) -> str:
    """The leak-safe group of one specimen record.

    The observation directory of the **original** file path is the group. When an
    export predates that field, the exported file's own name is the best available
    key (one group per frame) - which never *invents* a leak.
    """
    original = record.get("file")
    if original:
        parent = Path(str(original)).parent.name
        if parent:
            return parent
    relative = str(record.get("image") or "")
    return Path(relative).stem or "unknown"


def read_boxes(source, images_dir=None) -> tuple:
    """The boxes of an export **folder**, or of a ``boxes.csv``/``labels.json`` file.

    The entry point for "annotate the boxes we already have": an operator has
    either the export folder the Dataset surface wrote, or the ``boxes.csv`` from
    it with ``images/`` beside it, and both must open the same way
    (``SR-DATASET-022``). ``labels.json`` is preferred when it is present, because
    it is the machine-readable truth; ``boxes.csv`` is read when it is not - which
    is what makes a hand-kept or spreadsheet-edited box list usable.

    ``images_dir`` is where the pictures are, and it is used when the operator says
    so: a file may live in one folder while its images were moved to another (or the
    operator simply points at the ``images/`` folder they can see). Given, it wins
    over the file's own folder - an explicit statement beats a convention - and the
    names in the file are resolved inside it by their **file name**. Not given, each
    box's image is resolved in this order, first existing path wins: the exported copy
    (``images/...`` relative to the boxes file), the **original** path recorded in the
    file, then a file of the same name inside the folder's ``images/``. When none
    exists the reference is kept, so the caller can report a missing frame instead of
    quietly annotating nothing.

    Raises:
        ExportError: ``source`` is neither a folder nor a box file, holds no boxes,
            or was written by the older exporter whose ``boxes.csv`` header and
            rows disagree by one column - which would otherwise be read as shifted,
            plausible-looking numbers (``SR-DATASET-024``).
    """
    path = Path(source).expanduser()
    if path.is_dir():
        folder, csv_path, label_path = path, path / "boxes.csv", path / LABELS_FILE
    elif path.is_file():
        folder = path.parent
        csv_path = path if path.suffix.lower() == ".csv" else path.parent / "boxes.csv"
        label_path = path if path.name == LABELS_FILE else path.parent / LABELS_FILE
    else:
        raise ExportError(
            f"not a file or folder: {path}\n"
            "Point at an export folder, or at its boxes.csv / labels.json.")

    images = Path(images_dir).expanduser() if images_dir else None
    if images is not None and not images.is_dir():
        raise ExportError(
            f"the image folder does not exist: {images}\n"
            "Point at the folder that holds the frames the boxes were drawn on.")

    if label_path.exists():
        export = Export.read(folder)
        if export.boxes():
            return _with_images(export.boxes(), folder, images)
    if not csv_path.exists():
        raise ExportError(
            f"{folder} has no {LABELS_FILE} and no boxes.csv, so there are no boxes "
            "to annotate. Export a dataset from the Dataset surface first.")
    return _boxes_from_csv(folder, csv_path, images)


def _with_images(boxes, folder, images_dir) -> tuple:
    """Point every box at ``images_dir`` when one was named explicitly."""
    if images_dir is None:
        return tuple(boxes)
    return tuple(replace(box, image=_resolve_image(folder, box.relative,
                                                   str(box.image), images_dir))
                 for box in boxes)


def _boxes_from_csv(folder, csv_path, images_dir=None) -> tuple:
    """Read ``boxes.csv`` into :class:`Box` values, refusing a shifted header."""
    import csv

    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError as error:
        raise ExportError(f"could not read {csv_path}: {error}") from error
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ExportError(f"{csv_path} has no box rows.")
    header, body = rows[0], rows[1:]
    widths = {len(row) for row in body}
    if widths != {len(header)}:
        raise ExportError(
            f"{csv_path} is not readable: its header has {len(header)} column(s) "
            f"but its rows have {sorted(widths)}. This file was written by an "
            "older exporter whose header and rows disagreed - re-export the "
            "dataset and open the new folder.")

    boxes = []
    for row in body:
        record = dict(zip(header, row))
        try:
            bbox = (float(record["x1"]), float(record["y1"]),
                    float(record["x2"]), float(record["y2"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ExportError(
                f"{csv_path} has a row without usable x1/y1/x2/y2: {row}") from error
        x1, y1, x2, y2 = (min(bbox[0], bbox[2]), min(bbox[1], bbox[3]),
                          max(bbox[0], bbox[2]), max(bbox[1], bbox[3]))
        relative = (record.get("image") or "").strip()
        original = (record.get("file") or "").strip()
        boxes.append(Box(
            image=_resolve_image(folder, relative, original, images_dir),
            relative=relative or f"{IMAGES_DIR}/{Path(original).name}",
            split=(record.get("split") or SPLITS[0]).strip() or SPLITS[0],
            group=_group_of({"file": original, "image": relative}),
            bbox=(x1, y1, x2, y2),
            specimen_index=_as_int(record.get("specimen_index"), 1),
            source=(record.get("source") or "manual").strip() or "manual",
            image_width=_as_int(record.get("image_width"), 0),
            image_height=_as_int(record.get("image_height"), 0)))
    return tuple(sorted(boxes, key=lambda box: (box.relative, box.specimen_index)))


def _resolve_image(folder, relative, original, images_dir=None) -> Path:
    """Where one box's frame actually is.

    An explicit ``images_dir`` wins - it is the operator's statement, and the case
    that needs it is exactly the one where the convention does not hold (the file and
    the pictures in different folders, or the operator pointing at ``images/`` itself,
    which is the folder they can see in a file browser).
    """
    name = Path(relative).name if relative else Path(original).name
    candidates = []
    if images_dir is not None:
        candidates.append(images_dir / name if name else None)
    candidates += [folder / relative if relative else None,
                   Path(original) if original else None,
                   folder / IMAGES_DIR / name if name else None]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    if images_dir is not None and name:
        return images_dir / name
    if relative:
        return folder / relative
    return Path(original) if original else folder


def _as_int(value, default=0) -> int:
    """``int(value)`` when it parses, else ``default`` - a CSV is not typed."""
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return int(default)


def _bucket_split(key, weights=DEFAULT_GROUP_WEIGHTS) -> str:
    """Deterministically assign ``key`` to one of :data:`SPLITS`.

    Same idea as ``maintenance.split_of`` (a SHA-1 bucket rather than a random
    draw), so a split is reproducible across runs and platforms - the property
    that makes a reported score checkable.
    """
    digest = int(hashlib.sha1(str(key).encode("utf-8")).hexdigest(), 16)
    bucket = digest % 100
    edge = 0
    for name, weight in zip(SPLITS, weights):
        edge += weight
        if bucket < edge:
            return name
    return SPLITS[-1]

