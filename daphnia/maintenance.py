"""Data lifecycle: prune discarded frames, export measurements, consolidate a study.

The graceful end of a run. Long microscopy studies accumulate a lot of frames, so
this module decides what is safe to throw away (discarded frames that were never
measured), publishes the results next to the images (a CSV of the measurements and
a manifest of the final state) and then closes the study - leaving the machine and
the database in a known, tidy state instead of an ever-growing folder.
"""

import csv
import hashlib
import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from daphnia import annotation, units
from daphnia.annotation import MIN_BOX_PIXELS
from daphnia.db import queries, repository

#: Columns written by :func:`export_measurements_csv`, in order.
EXPORT_COLUMNS = ("study_id", "target", "specimen_index", "trait", "length_um",
                  "length_mm", "length_px", "px_per_um", "source", "measured_by",
                  "created_at")


def prune_discarded(conn, study_id) -> dict:
    """Delete the discarded, un-annotated frames of a study; returns counts.

    A frame the selection step sent to the tray is never measured, so both its
    file and its row can go. A frame that already carries a measurement is left
    alone, so pruning can never orphan a result.
    """
    rows = conn.execute(
        "SELECT a.asset_id, a.file_path FROM media_assets a"
        " JOIN observations o ON o.observation_id = a.observation_id"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " WHERE s.study_id = ? AND a.selection = 'discarded'"
        "   AND NOT EXISTS (SELECT 1 FROM measurements m WHERE m.asset_id = a.asset_id)",
        (study_id,)).fetchall()

    removed_files = 0
    for row in rows:
        path = Path(row["file_path"])
        try:
            if path.exists():
                path.unlink()
                removed_files += 1
        except OSError:
            pass                       # a missing/locked file must not stop the prune
        conn.execute("DELETE FROM media_assets WHERE asset_id = ?", (row["asset_id"],))
    return {"rows": len(rows), "files": removed_files}


def export_measurements_csv(conn, study_id, path) -> int:
    """Write a study's measurements as CSV; returns the number of rows written."""
    rows = queries.measurements_for_study(conn, study_id)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPORT_COLUMNS)
        for row in rows:
            writer.writerow((
                row["study_id"], queries.target_label(row),
                row["specimen_index"], row["trait"],
                f"{row['length_um']:.3f}", f"{units.to_mm(row['length_um']):.3f}",
                f"{row['length_px']:.3f}",
                f"{row['px_per_um']:.6f}", row["source"], row["measured_by"],
                row["created_at"]))
    return len(rows)


def write_manifest(conn, study_id, destination) -> Path:
    """Write a small JSON manifest of a study's final state; returns its path."""
    study = queries.study_by_id(conn, study_id)
    manifest = {
        "study_id": study_id,
        "name": study["name"] if study else None,
        "destination": str(destination),
        "consolidated_at": datetime.now().isoformat(timespec="seconds"),
        "progress": queries.study_progress(conn, study_id),
    }
    path = Path(destination) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def consolidate_and_close(conn, study_id, *, prune=True, export=True) -> dict:
    """Tidy a study's data and close it (the graceful end of a run).

    Prunes discarded frames, exports the measurements as CSV next to the images and
    writes a manifest, then marks the study closed. Returns what was done so the
    caller can report it.
    """
    study = queries.study_by_id(conn, study_id)
    destination = Path(study["destination"]) if study else None

    result = {"pruned": {"rows": 0, "files": 0}, "exported": 0,
              "csv": None, "manifest": None}
    if prune:
        result["pruned"] = prune_discarded(conn, study_id)
    if destination is not None:
        if export:
            csv_path = destination / "measurements.csv"
            result["exported"] = export_measurements_csv(conn, study_id, csv_path)
            result["csv"] = str(csv_path)
        result["manifest"] = str(write_manifest(conn, study_id, destination))

    repository.close_study(conn, study_id)
    return result


#: Columns written by :func:`export_bench_csv`, in order. The landmark coordinates
#: are included deliberately: a bench result without its points cannot be checked,
#: and cannot be reused as training data. ``individual_index`` says **whose** number
#: each row is (the box that stands for that animal; empty = outside every box), and
#: ``specimen_index`` stays the row's own slot on the frame (`vision.individuals`).
#: ``total_uncertainty_um`` is the whole budget of the length - the scale term in
#: ``uncertainty_pct`` plus the click and the pixel grid - so an exported bench
#: number carries how accurate it is, not only how it was scaled.
BENCH_COLUMNS = ("measurement_id", "file", "specimen_index", "individual_index", "trait",
                 "name", "length_um", "length_mm", "length_px", "p1_x", "p1_y", "p2_x", "p2_y",
                 "px_per_um", "uncertainty_pct", "total_uncertainty_um", "source",
                 "measured_by", "created_at")

#: ``name`` is the operator's **own name** for the measurement (`SR-BENCH-024`,
#: ``corpus_labels.label_name``), blank when they gave it none. It is not part of
#: ``CORPUS_COLUMNS``: the training corpus is read for pixels and traits, and a name the lab
#: typed is a human handle on a row, not something a model is trained on.


def export_bench_csv(conn, path, dataset_id=None) -> int:
    """Write the Bench's measurements as CSV; returns the number of rows.

    With ``dataset_id`` it writes exactly that dataset (what the surface's *Export
    this dataset* means); without one, the study-less rows. Unlike the study export
    this keeps the landmarks: bench work is the ad-hoc end of the app, and a number
    whose points were thrown away cannot be defended. The name the operator gave each
    measurement travels with it too (`SR-BENCH-024`): a name that was dropped by the export
    would be work the operator did that no longer exists.
    """
    rows = queries.bench_measurements(conn, limit=100000, dataset_id=dataset_id)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(BENCH_COLUMNS)
        for row in rows:
            uncertainty = row["uncertainty_pct"]
            total = row["total_uncertainty_um"]
            writer.writerow((
                row["measurement_id"], row["file_path"],
                row["specimen_index"],
                "" if row["individual_index"] is None else row["individual_index"],
                row["trait"], row["name"] or "",
                f"{row['length_um']:.3f}", f"{units.to_mm(row['length_um']):.3f}",
                f"{row['length_px']:.3f}",
                f"{row['p1_x']:.3f}", f"{row['p1_y']:.3f}",
                f"{row['p2_x']:.3f}", f"{row['p2_y']:.3f}",
                f"{row['px_per_um']:.6f}",
                "" if uncertainty is None else f"{uncertainty:.3f}",
                "" if total is None else f"{total:.3f}",
                row["source"], row["measured_by"], row["created_at"]))
    return len(rows)


# --- the training corpus (DATASET surface) ----------------------------------


#: Columns written by :func:`export_corpus`'s CSV, in order. The landmarks *and*
#: the split travel with every row: a training script must be able to build the
#: dataset from this file alone, and a number without its points cannot be checked.
#: ``total_uncertainty_um`` travels too - the whole budget of the length (scale,
#: click and pixel grid), so a corpus row says what it is worth and not only how
#: well the scale was measured (`docs/accuracy.md` §1).
CORPUS_COLUMNS = ("label_id", "file", "split", "image_width", "image_height",
                  "specimen_index", "trait", "p1_x", "p1_y", "p2_x", "p2_y",
                  "length_px", "px_per_um", "length_um", "uncertainty_pct",
                  "total_uncertainty_um", "source", "labeled_by",
                  "created_at")

#: Fractions of the corpus the deterministic split aims at (train / val / test).
SPLIT_BUCKETS = (("train", 70), ("val", 15), ("test", 15))


def split_of(group_key: str) -> str:
    """The split a group of labels belongs to, deterministically.

    The *group key is the image file*, so every label of one frame lands in the
    same split and re-running the export never reshuffles the corpus. A hash is
    used because it is stable across runs and platforms, unlike a random choice.
    """
    digest = int(hashlib.sha1(str(group_key).encode("utf-8")).hexdigest(), 16)
    bucket = digest % 100
    edge = 0
    for name, weight in SPLIT_BUCKETS:
        edge += weight
        if bucket < edge:
            return name
    return SPLIT_BUCKETS[-1][0]


def _exported_name(file_path: str) -> str:
    """A collision-free, deterministic file name for one corpus image."""
    path = Path(file_path)
    digest = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:8]
    suffix = (path.suffix or ".png").lower()
    return f"{path.stem}_{digest}{suffix}"


def export_corpus(conn, destination) -> dict:
    """Write the labelled corpus: images + a label file + a manifest.

    The deliverable (``D1``) must be consumable by a training script without the
    app, so the export is self-contained: every referenced image is copied into
    ``<destination>/images/``, all labels go to ``labels.csv`` and ``labels.json``
    (landmark coordinates, image size, specimen, trait, scale and split), and a
    ``manifest.json`` records the counts. Returns what was written.
    """
    destination = Path(destination)
    images_dir = destination / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = queries.corpus_labels(conn)

    # One copy per distinct source image; the export references the copy by name.
    exported = {}
    copied = 0
    for row in rows:
        if row["file_path"] in exported:
            continue
        source = Path(row["file_path"])
        if not source.exists():
            exported[row["file_path"]] = None      # reference kept, image missing
            continue
        target = images_dir / _exported_name(row["file_path"])
        try:
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1
        except OSError:
            exported[row["file_path"]] = None
            continue
        exported[row["file_path"]] = f"images/{target.name}"

    split_counts = {name: 0 for name, _ in SPLIT_BUCKETS}
    records = []
    csv_path = destination / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CORPUS_COLUMNS)
        for row in rows:
            split = split_of(row["file_path"])
            split_counts[split] += 1
            image = exported.get(row["file_path"])
            records.append({
                "label_id": row["label_id"],
                "file": row["file_path"],
                "image": image,
                "split": split,
                "image_width": row["image_width"],
                "image_height": row["image_height"],
                "specimen_index": row["specimen_index"],
                "trait": row["trait"],
                "keypoints": {"p1": [row["p1_x"], row["p1_y"]],
                              "p2": [row["p2_x"], row["p2_y"]]},
                "length_px": row["length_px"],
                "px_per_um": row["px_per_um"],
                "length_um": row["length_um"],
                "uncertainty_pct": row["uncertainty_pct"],
                "total_uncertainty_um": row["total_uncertainty_um"],
                "source": row["source"],
                "labeled_by": row["labeled_by"],
                "created_at": row["created_at"],
            })
            writer.writerow((
                row["label_id"], image or row["file_path"], split,
                row["image_width"], row["image_height"], row["specimen_index"],
                row["trait"],
                f"{row['p1_x']:.3f}", f"{row['p1_y']:.3f}",
                f"{row['p2_x']:.3f}", f"{row['p2_y']:.3f}",
                f"{row['length_px']:.3f}",
                "" if row["px_per_um"] is None else f"{row['px_per_um']:.6f}",
                "" if row["length_um"] is None else f"{row['length_um']:.3f}",
                "" if row["uncertainty_pct"] is None else f"{row['uncertainty_pct']:.3f}",
                "" if row["total_uncertainty_um"] is None
                else f"{row['total_uncertainty_um']:.3f}",
                row["source"], row["labeled_by"], row["created_at"]))

    json_path = destination / "labels.json"
    json_path.write_text(json.dumps({"labels": records}, indent=2), encoding="utf-8")

    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "destination": str(destination),
        "stats": queries.corpus_stats(conn),
        "split_counts": split_counts,
        "images_copied": copied,
        "images_missing": sum(1 for value in exported.values() if value is None),
        "labels_csv": csv_path.name,
        "labels_json": json_path.name,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"labels": len(rows), "images": copied, "splits": split_counts,
            "csv": str(csv_path), "json": str(json_path),
            "manifest": str(manifest_path), "destination": str(destination)}
# --- one dataset (one model's corpus) ---------------------------------------


def export_dataset(conn, dataset_id, destination) -> dict:
    """Write one dataset - one model's corpus - as images + labels + a manifest.

    Unlike :func:`export_corpus` (which dumps the whole, unscoped corpus), this
    exports exactly what the dataset's *model* needs, and nothing it does not: the
    keypoints of its keypoint system under their export names, and the bounding
    boxes when the model predicts a box. A training script therefore needs no other
    source of truth than the folder - and re-exporting one dataset can never change
    another model's data.
    """
    dataset = queries.dataset_by_id(conn, dataset_id)
    if dataset is None:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    schema = annotation.get(dataset["schema_key"])
    destination = Path(destination)
    images_dir = destination / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels = queries.dataset_labels(conn, dataset_id)
    boxes = queries.dataset_boxes(conn, dataset_id)
    exported, copied = _copy_corpus_images(images_dir, list(labels) + list(boxes))

    csv_path, keypoint_records, split_counts = _write_dataset_labels(
        destination, labels, exported, schema)
    boxes_path, box_records = _write_dataset_boxes(
        destination, boxes, exported)
    split_files = _write_dataset_splits(destination, labels, boxes, exported)
    json_path = _write_dataset_json(destination, dataset, schema, keypoint_records,
                                    box_records, split_files)

    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dict(dataset),
        "schema": {"key": schema.key, "label": schema.label,
                   "keypoints": list(schema.keypoints), "boxes": bool(schema.boxes)},
        "stats": queries.dataset_stats(conn, dataset_id),
        "split_counts": split_counts,
        "splits": split_files,
        "images_copied": copied,
        "images_missing": sum(1 for value in exported.values() if value is None),
        "labels_csv": csv_path.name,
        "boxes_csv": boxes_path.name,
        "labels_json": json_path.name,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"dataset": dict(dataset),
            "schema": {"key": schema.key, "label": schema.label},
            "labels": len(labels), "boxes": len(boxes), "images": copied,
            "splits": split_counts, "split_files": split_files,
            "csv": str(csv_path), "boxes_csv": str(boxes_path),
            "json": str(json_path), "manifest": str(manifest_path),
            "destination": str(destination)}


def import_boxes(conn, dataset_id, source, *, images_dir=None, labeled_by=None,
                 note=None) -> dict:
    """Seed a dataset's boxes from an existing export folder or ``boxes.csv``.

    The first half of the recursive annotator: boxes that already exist - a
    detector's predictions, or boxes drawn in an earlier session - become rows of
    the open dataset, so they can be reviewed yes/no and then carry keypoints
    (``SR-DATASET-022``). Every row keeps the source it arrived with (``'model'``
    for a prediction, ``'manual'`` for a human's box): ``SR-PREDICT-N01`` and
    ``mlContext.md`` law 7 say a prediction is not ground truth until a human
    accepts it, and the import is not that acceptance.

    A box that leaves its image, or is smaller than ``MIN_BOX_PIXELS``, is **not**
    imported and is counted in ``skipped``: the labeller refuses to *make* such a
    box, so importing one would put a rectangle in the corpus that cannot exist
    (``SR-DATASET-020``). The frame's size is read from the file when the source
    does not carry it, so a hand-kept ``boxes.csv`` without ``image_width`` still
    imports.

    Args:
        conn: open database connection.
        dataset_id: the dataset whose corpus receives the boxes.
        source: an export folder, a ``boxes.csv``, or a ``labels.json``.
        images_dir: where the frames are, when that is not beside the file - the
            operator may have the boxes file in one folder and the pictures in
            another, or point at ``images/`` itself.
        labeled_by: the operator to record (defaults to the configured one).
        note: written into each row's ``note``; the source is named by default.

    Returns:
        ``{"boxes": imported, "images": frames, "skipped": unusable,
        "missing": frames whose file is not there, "folder": str,
        "images_dir": str or None}``.

    Raises:
        ValueError: the dataset does not exist, or its model has no bounding box -
            the fix is another dataset, so it is said in words (``X-N10``).
        ExportError: ``source`` holds no readable boxes, or ``images_dir`` is not
            a folder.
    """
    from daphnia.vision.export import ExportError, read_boxes

    dataset = queries.dataset_by_id(conn, dataset_id)
    if dataset is None:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    schema = annotation.get(dataset["schema_key"])
    if not schema.exports_boxes:
        raise ValueError(
            f"'{dataset['name']}' is a keypoint-only dataset ({schema.label}), so it "
            "has nowhere to put a bounding box. Create a dataset whose model has a "
            "box - 'Pose (4 points) + bounding box' or 'Bounding boxes only' - and "
            "import into that.")

    boxes = read_boxes(source, images_dir=images_dir)
    if not boxes:
        raise ExportError(f"no boxes found in {source}")

    label = note or f"imported from {Path(source).name}"
    imported, skipped, images, missing = 0, 0, set(), set()
    for box in boxes:
        image = Path(box.image)
        width, height = int(box.image_width), int(box.image_height)
        if not (width and height):
            width, height = _image_size(image)
        if not image.exists():
            missing.add(str(image))
        checked = replace(box, image_width=width, image_height=height)
        if checked.outside():
            skipped += 1
            continue
        repository.save_corpus_box(
            conn, file_path=str(image), image_size=(width, height), box=checked.bbox,
            dataset_id=dataset_id, specimen_index=checked.specimen_index,
            source=checked.source, labeled_by=labeled_by, note=label)
        imported += 1
        images.add(str(image))
    return {"boxes": imported, "images": len(images), "skipped": skipped,
            "missing": len(missing), "folder": str(Path(source)),
            "images_dir": str(Path(images_dir)) if images_dir else None}


def _image_size(path) -> tuple:
    """``(width, height)`` of an image file, or ``(0, 0)`` when it cannot be read.

    Only needed when the source of an import does not record the frame's size; the
    loader is used rather than a raw OpenCV call so a ``.HEIC`` imports too.
    """
    from daphnia.vision.images import ImageLoadError, load_image

    try:
        frame = load_image(path)
    except (ImageLoadError, OSError, ValueError):
        return 0, 0
    if frame is None or getattr(frame, "ndim", 0) < 2:
        return 0, 0
    height, width = frame.shape[:2]
    return int(width), int(height)


def _copy_corpus_images(images_dir, rows) -> tuple:
    """Copy every referenced image once; returns ``({file: exported or None}, n)``."""
    exported = {}
    copied = 0
    for row in rows:
        file_path = row["file_path"]
        if file_path in exported:
            continue
        source = Path(file_path)
        if not source.exists():
            exported[file_path] = None      # reference kept, image missing
            continue
        target = images_dir / _exported_name(file_path)
        try:
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1
        except OSError:
            exported[file_path] = None
            continue
        exported[file_path] = f"images/{target.name}"
    return exported, copied
def _write_dataset_labels(destination, labels, exported, schema) -> tuple:
    """Write ``labels.csv`` (keypoint rows); returns ``(path, records, splits)``."""
    path = destination / "labels.csv"
    records = []
    split_counts = {name: 0 for name, _ in SPLIT_BUCKETS}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CORPUS_COLUMNS + ("p1_name", "p2_name"))
        for row in labels:
            split = split_of(row["file_path"])
            split_counts[split] += 1
            first, second = schema.keypoint_system.points_for(row["trait"])
            first_name = annotation.export_name(first)
            second_name = annotation.export_name(second)
            records.append({
                "label_id": row["label_id"],
                "file": row["file_path"],
                "image": exported.get(row["file_path"]),
                "split": split,
                "image_width": row["image_width"],
                "image_height": row["image_height"],
                "specimen_index": row["specimen_index"],
                "trait": row["trait"],
                "keypoints": {first_name: [row["p1_x"], row["p1_y"]],
                              second_name: [row["p2_x"], row["p2_y"]]},
                "p1": [row["p1_x"], row["p1_y"]],
                "p2": [row["p2_x"], row["p2_y"]],
                "length_px": row["length_px"],
                "px_per_um": row["px_per_um"],
                "length_um": row["length_um"],
                "uncertainty_pct": row["uncertainty_pct"],
                "total_uncertainty_um": row["total_uncertainty_um"],
                "source": row["source"],
                "labeled_by": row["labeled_by"],
                "created_at": row["created_at"],
            })
            writer.writerow((
                row["label_id"], exported.get(row["file_path"]) or row["file_path"],
                split, row["image_width"], row["image_height"],
                row["specimen_index"], row["trait"],
                f"{row['p1_x']:.3f}", f"{row['p1_y']:.3f}",
                f"{row['p2_x']:.3f}", f"{row['p2_y']:.3f}",
                f"{row['length_px']:.3f}",
                "" if row["px_per_um"] is None else f"{row['px_per_um']:.6f}",
                "" if row["length_um"] is None else f"{row['length_um']:.3f}",
                "" if row["uncertainty_pct"] is None else f"{row['uncertainty_pct']:.3f}",
                "" if row["total_uncertainty_um"] is None
                else f"{row['total_uncertainty_um']:.3f}",
                row["source"], row["labeled_by"], row["created_at"],
                first_name, second_name))
    return path, records, split_counts


def _write_dataset_boxes(destination, boxes, exported) -> tuple:
    """Write ``boxes.csv``; returns ``(path, records)``.

    A box gets its own file because it is not a pair of landmarks: folding it into
    the keypoint table would leave two of its columns meaningless on every row.
    """
    path = destination / "boxes.csv"
    records = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("box_id", "file", "image", "split", "image_width",
                         "image_height", "specimen_index", "x1", "y1", "x2", "y2",
                         "width", "height", "source", "labeled_by", "created_at"))
        for row in boxes:
            split = split_of(row["file_path"])
            width = row["x2"] - row["x1"]
            height = row["y2"] - row["y1"]
            records.append({
                "box_id": row["box_id"],
                "file": row["file_path"],
                "image": exported.get(row["file_path"]),
                "split": split,
                "image_width": row["image_width"],
                "image_height": row["image_height"],
                "specimen_index": row["specimen_index"],
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
                "width": width,
                "height": height,
                "source": row["source"],
                "labeled_by": row["labeled_by"],
                "created_at": row["created_at"],
            })
            writer.writerow((
                row["box_id"], row["file_path"],
                exported.get(row["file_path"]) or "",
                split, row["image_width"], row["image_height"],
                row["specimen_index"],
                f"{row['x1']:.3f}", f"{row['y1']:.3f}",
                f"{row['x2']:.3f}", f"{row['y2']:.3f}",
                f"{width:.3f}", f"{height:.3f}",
                row["source"], row["labeled_by"], row["created_at"]))
    return path, records

def _write_dataset_splits(destination, labels, boxes, exported) -> dict:
    """Write one image list per split so a training run needs no CSV parsing.

    ``splits/train.txt`` etc. hold the **exported image paths** that belong to that
    split, in a stable order - the one thing every trainer wants and almost none
    agree on how to spell. The lists are derived from the same deterministic
    ``split_of`` the label files use, so they can never disagree with them.
    """
    folder = destination / "splits"
    folder.mkdir(parents=True, exist_ok=True)
    members = {}
    for row in list(labels) + list(boxes):
        image = exported.get(row["file_path"])
        if image is None:
            continue                                  # a missing image is not in a split
        members.setdefault(split_of(row["file_path"]), set()).add(image)

    written = {}
    for name, _weight in SPLIT_BUCKETS:
        images = sorted(members.get(name, ()))
        path = folder / f"{name}.txt"
        path.write_text("\n".join(images) + ("\n" if images else ""),
                        encoding="utf-8")
        written[name] = {"file": f"splits/{path.name}", "images": len(images)}
    return written


def _write_dataset_json(destination, dataset, schema, keypoint_records,
                        box_records, split_files) -> Path:
    """Write the machine-readable label file; returns its path."""
    path = destination / "labels.json"
    path.write_text(json.dumps({
        "dataset": {key: dataset[key] for key in dataset.keys()
                    if key != "is_current"},
        "schema": {
            "key": schema.key,
            "label": schema.label,
            "keypoints": list(schema.keypoints),
            "export_names": [annotation.export_name(name)
                             for name in schema.keypoints],
            "intervals": [{"trait": trait,
                           "from": annotation.export_name(first),
                           "to": annotation.export_name(second)}
                          for trait, first, second in schema.intervals],
            # The skeleton: which keypoints are anatomically connected. A pose
            # config draws bones from this, and it is the one place the *model's*
            # connectivity is written down (SR-DATASET-023).
            "edges": [[first, second] for first, second in schema.export_edges()],
            "boxes": bool(schema.boxes),
        },
        # The per-specimen view is what a pose/detection training config consumes:
        # the keypoints by name, plus the box when the model predicts one.
        "specimens": _specimen_records(keypoint_records, box_records, schema),
        "splits": split_files,
        "labels": keypoint_records,
        "boxes": box_records,
    }, indent=2), encoding="utf-8")
    return path


def _specimen_records(keypoint_records, box_records, schema) -> list:
    """Merge the keypoint and box rows into one record per ``(image, specimen)``.

    That is the shape a pose/detection training config wants - everything about one
    animal in one place - and building it here keeps every consumer from
    re-implementing the join. Only the keypoints the schema actually labels are
    reported, so a two-keypoint corpus never claims to know the sides.
    """
    def blank(row):
        return {"file": row["file"], "image": row["image"], "split": row["split"],
                "image_width": row["image_width"],
                "image_height": row["image_height"],
                "specimen_index": row["specimen_index"],
                "keypoints": {}, "traits": {}, "bbox": None}

    merged = {}
    for row in keypoint_records:
        record = merged.setdefault((row["file"], row["specimen_index"]), blank(row))
        record["keypoints"].update(row["keypoints"])
        # The trait block carries the whole budget of the length, so a training
        # config (or a reader like `vision.pose_corpus`) can say what a number is
        # worth and not only what it is (`docs/accuracy.md` §1).
        record["traits"][row["trait"]] = {"length_px": row["length_px"],
                                          "length_um": row["length_um"],
                                          "uncertainty_pct": row["uncertainty_pct"],
                                          "total_uncertainty_um":
                                              row["total_uncertainty_um"]}
    for row in box_records:
        record = merged.setdefault((row["file"], row["specimen_index"]), blank(row))
        record["bbox"] = row["bbox"]

    records = []
    for key in sorted(merged):
        record = merged[key]
        wanted = [annotation.export_name(name) for name in schema.keypoints]
        record["keypoints"] = {name: record["keypoints"][name]
                               for name in wanted if name in record["keypoints"]}
        records.append(record)
    return records


# --- repairing a corpus in place --------------------------------------------


#: Columns one out-of-bounds finding reports.
BOX_REPAIR_COLUMNS = ("box_id", "dataset_id", "file_path", "specimen_index",
                      "before", "after")


def find_boxes_outside_image(conn, dataset_id=None) -> list:
    """Bounding boxes that leave their image, or that are too small to be one.

    A box is *stored* in source-image pixels, so a corner outside ``0..width`` /
    ``0..height`` can only come from a drag that left the frame - which is what the
    canvas used to allow (fixed: it now pulls the corner back onto the border). The
    rows are already in the corpus, so they are reported rather than deleted: a
    label is never thrown away as a side effect of a check (``SR-DATASET-N03``).
    """
    rows = conn.execute(
        "SELECT box_id, dataset_id, file_path, specimen_index, image_width,"
        " image_height, x1, y1, x2, y2 FROM corpus_boxes"
        " WHERE dataset_id IS ? AND (x1 < 0 OR y1 < 0 OR x2 > image_width"
        "   OR y2 > image_height OR (x2 - x1) < ? OR (y2 - y1) < ?)"
        " ORDER BY file_path, specimen_index",
        (dataset_id, MIN_BOX_PIXELS, MIN_BOX_PIXELS)).fetchall()
    return [{"box_id": row["box_id"], "dataset_id": row["dataset_id"],
             "file_path": row["file_path"],
             "specimen_index": row["specimen_index"],
             "before": (row["x1"], row["y1"], row["x2"], row["y2"]),
             "after": _clamp_box((row["x1"], row["y1"], row["x2"], row["y2"]),
                                 row["image_width"], row["image_height"])}
            for row in rows]


def _clamp_box(box, width, height) -> tuple:
    """Pull a box inside a ``width`` x ``height`` image."""
    x1, x2 = sorted((max(0.0, min(float(box[0]), float(width))),
                     max(0.0, min(float(box[2]), float(width)))))
    y1, y2 = sorted((max(0.0, min(float(box[1]), float(height))),
                     max(0.0, min(float(box[3]), float(height)))))
    return (x1, y1, x2, y2)


def clamp_corpus_boxes(conn, dataset_id=None, apply=False) -> dict:
    """Clamp every out-of-bounds box of a dataset back inside its image.

    ``apply=False`` only reports. That is the default on purpose: this touches
    ground truth, so the first run of anything that rewrites a corpus must be a
    dry run. Nothing is deleted - a box that would collapse to nothing is reported
    so the frame can be labelled again by hand instead of being silently dropped.

    Returns ``{"found": [...], "clamped": n, "degenerate": [...], "applied": bool}``.
    """
    found = find_boxes_outside_image(conn, dataset_id)
    degenerate, clamped = [], 0
    for item in found:
        after = item["after"]
        if (after[2] - after[0]) < MIN_BOX_PIXELS \
                or (after[3] - after[1]) < MIN_BOX_PIXELS:
            degenerate.append(item)
            continue                              # no clamp can rescue it
        if apply:
            conn.execute(
                "UPDATE corpus_boxes SET x1 = ?, y1 = ?, x2 = ?, y2 = ?"
                " WHERE box_id = ?",
                (after[0], after[1], after[2], after[3], item["box_id"]))
            clamped += 1
    return {"found": found, "clamped": clamped, "degenerate": degenerate,
            "applied": bool(apply)}