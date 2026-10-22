"""Reporting read models: the query side of the data-access layer.

Writes live in :mod:`daphnia.db.repository`; these are the lists the UI needs
(targets to capture, frames to review, measurements to plot, counters).
"""

from daphnia.study import StudySpec


def list_profiles(conn):
    """All profiles, most recently verified first."""
    return conn.execute(
        "SELECT profile_id, profile_name, px_per_um, is_active, last_verified_at"
        " FROM calibration_profiles ORDER BY last_verified_at DESC"
    ).fetchall()


def active_study(conn):
    """The project currently being worked on, or ``None``."""
    return conn.execute(
        "SELECT * FROM studies WHERE is_current = 1"
        " ORDER BY created_at DESC, rowid DESC LIMIT 1"
    ).fetchone()


def study_by_id(conn, study_id):
    """One study row, or ``None``."""
    return conn.execute(
        "SELECT * FROM studies WHERE study_id = ?", (study_id,)).fetchone()


#: Sub-selects that count a study's progress, used by the project list.
_PROGRESS = {
    "plates": "(SELECT COUNT(*) FROM plates p WHERE p.study_id = s.study_id)",
    "targets": ("(SELECT COUNT(*) FROM observations o"
                " JOIN imaging_sessions i ON i.session_id = o.session_id"
                " WHERE i.study_id = s.study_id)"),
    "frames": ("(SELECT COUNT(*) FROM media_assets a"
               " JOIN observations o ON o.observation_id = a.observation_id"
               " JOIN imaging_sessions i ON i.session_id = o.session_id"
               " WHERE i.study_id = s.study_id)"),
    "selected": ("(SELECT COUNT(*) FROM media_assets a"
                 " JOIN observations o ON o.observation_id = a.observation_id"
                 " JOIN imaging_sessions i ON i.session_id = o.session_id"
                 " WHERE i.study_id = s.study_id AND a.selection = 'kept')"),
    "measured": ("(SELECT COUNT(*) FROM measurements m"
                 " JOIN observations o ON o.observation_id = m.observation_id"
                 " JOIN imaging_sessions i ON i.session_id = o.session_id"
                 " WHERE i.study_id = s.study_id)"),
}


def list_studies_with_progress(conn):
    """Every study with the counts the overview shows, current project first."""
    columns = ", ".join(f"{sql} AS {name}" for name, sql in _PROGRESS.items())
    return conn.execute(
        f"SELECT s.*, {columns} FROM studies s"
        " ORDER BY s.is_current DESC, s.created_at DESC, s.rowid DESC"
    ).fetchall()


def study_progress(conn, study_id) -> dict:
    """Progress counts of one study (plates, targets, frames, selected, measured)."""
    columns = ", ".join(f"{sql} AS {name}" for name, sql in _PROGRESS.items())
    row = conn.execute(
        f"SELECT {columns} FROM studies s WHERE s.study_id = ?", (study_id,)
    ).fetchone()
    return dict(row) if row is not None else {}


def active_study_spec(conn):
    """The active study as a :class:`~daphnia.study.StudySpec`, or ``None``.

    This is what the screens read to decide *what to show*: plates or a counter,
    which anatomy to annotate, repeated timepoints or a single measurement.
    """
    row = active_study(conn)
    return StudySpec.from_row(row) if row is not None else None


def list_studies(conn, limit=50):
    """All studies, newest first."""
    return conn.execute(
        "SELECT study_id, name, organism, measurement_interval_hours, destination,"
        "       status, created_at FROM studies"
        " ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
    ).fetchall()


def list_study_templates(conn):
    """Templates, newest name first, only the latest version of each name."""
    return conn.execute(
        "SELECT t.* FROM study_templates t"
        " JOIN (SELECT name, MAX(version) AS version FROM study_templates"
        "       GROUP BY name) latest"
        "   ON latest.name = t.name AND latest.version = t.version"
        " ORDER BY t.created_at DESC, t.rowid DESC"
    ).fetchall()


def list_plates(conn, study_id):
    """The plates of a study, in creation order."""
    return conn.execute(
        "SELECT plate_id, format, label, created_at FROM plates"
        " WHERE study_id = ? ORDER BY created_at, rowid", (study_id,)
    ).fetchall()


def plate_well_counts(conn, plate_id) -> dict:
    """Per **well** of a plate: frames, bursts, kept frames, measurements, and its image.

    Keyed by well label, and only the wells that have rows: the wells of the format that
    are still empty are added by ``daphnia.wells.progress``, which is the half that knows
    the plate's geometry. The counts are aggregated over every session and every visit -
    a longitudinal study captures the same well more than once, and the plate page asks
    "has this well been imaged", not "which burst was it".

    ``kept`` is the number of frames a human kept (the Select step's decision) and
    ``measured`` the number of measurement rows on the well's observation(s); both are
    shown for a well, none of them is a state (see ``daphnia.wells``).

    ``image_path`` is the well's **image**: the primary frame of the newest observation
    that has one, so the plate can draw what was captured where the capture belongs. The
    path is a *view* of the row, never part of the state - a well without a file it can
    read is still a captured well, and the grid draws it as one.
    """
    rows = conn.execute(
        "SELECT o.well_label AS well_label,"
        "       COUNT(a.asset_id) AS frames,"
        "       COUNT(DISTINCT a.burst_index) AS bursts,"
        "       COALESCE(SUM(a.selection = 'kept'), 0) AS kept,"
        "       COALESCE(MAX(m.measured), 0) AS measured,"
        "       (SELECT p.file_path FROM media_assets p"
        "         WHERE p.observation_id = o.observation_id AND p.is_primary = 1"
        "         ORDER BY p.created_at DESC, p.rowid DESC LIMIT 1) AS image_path"
        " FROM observations o"
        " LEFT JOIN media_assets a ON a.observation_id = o.observation_id"
        " LEFT JOIN (SELECT observation_id, COUNT(*) AS measured FROM measurements"
        "            GROUP BY observation_id) m"
        "   ON m.observation_id = o.observation_id"
        " WHERE o.plate_id = ? AND o.well_label IS NOT NULL"
        " GROUP BY o.well_label", (plate_id,)).fetchall()
    return {row["well_label"]: row for row in rows}


def list_targets(conn, study_id, limit=500):
    """Every captured target of a study with its frame and measurement counts.

    A target is a well of a plate or one individual of the counter.
    """
    return conn.execute(
        "SELECT o.observation_id, o.plate_id, o.well_label, o.individual_index,"
        "       o.created_at, p.label AS plate_label, p.format AS plate_format,"
        "       (SELECT COUNT(*) FROM media_assets a"
        "         WHERE a.observation_id = o.observation_id) AS frames,"
        "       (SELECT COUNT(*) FROM media_assets a"
        "         WHERE a.observation_id = o.observation_id AND a.selection = 'kept') AS kept,"
        "       (SELECT COUNT(*) FROM measurements m"
        "         WHERE m.observation_id = o.observation_id) AS measured"
        " FROM observations o"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " LEFT JOIN plates p ON p.plate_id = o.plate_id"
        " WHERE s.study_id = ?"
        " ORDER BY o.created_at, o.rowid LIMIT ?", (study_id, limit)
    ).fetchall()


def target_label(row) -> str:
    """Human label of a target row: 'Plate 1 A01', 'Individual 3' or 'ad-hoc'."""
    if row["well_label"]:
        plate = row["plate_label"] if "plate_label" in row.keys() else None
        return f"{plate or 'plate'} {row['well_label']}"
    if row["individual_index"] is not None:
        return f"Individual {row['individual_index']}"
    return "ad-hoc"


def assets_for(conn, observation_id):
    """Every frame of a target, burst by burst, in capture order."""
    return conn.execute(
        "SELECT asset_id, observation_id, file_path, kind, frame_index, burst_index,"
        "       is_primary, sharpness, selection"
        " FROM media_assets WHERE observation_id = ?"
        " ORDER BY burst_index, frame_index, rowid", (observation_id,)
    ).fetchall()


def asset_by_id(conn, asset_id):
    """One media asset row, or ``None``."""
    return conn.execute(
        "SELECT * FROM media_assets WHERE asset_id = ?", (asset_id,)).fetchone()


def kept_assets(conn, observation_id):
    """Good candidates of a target, with how many measurements they carry."""
    return conn.execute(
        "SELECT a.asset_id, a.file_path, a.burst_index, a.frame_index,"
        " (SELECT COUNT(*) FROM measurements m WHERE m.asset_id = a.asset_id)"
        " AS measurements"
        " FROM media_assets a"
        " WHERE a.observation_id = ? AND a.selection = 'kept'"
        " ORDER BY a.burst_index, a.frame_index, a.rowid", (observation_id,)
    ).fetchall()


def observation_by_id(conn, observation_id):
    """One observation row, or ``None``."""
    return conn.execute(
        "SELECT * FROM observations WHERE observation_id = ?", (observation_id,)
    ).fetchone()


def primary_asset_path(conn, observation_id):
    """File path of the burst's auto-selected best frame, or ``None``."""
    row = conn.execute(
        "SELECT file_path FROM media_assets WHERE observation_id = ? AND is_primary = 1"
        " ORDER BY created_at DESC, rowid DESC LIMIT 1", (observation_id,)
    ).fetchone()
    return row["file_path"] if row else None


def measurement_for(conn, observation_id):
    """The most recent measurement of an observation, or ``None``."""
    return conn.execute(
        "SELECT * FROM measurements WHERE observation_id = ?"
        " ORDER BY created_at DESC, rowid DESC LIMIT 1", (observation_id,)
    ).fetchone()


def measurement_for_asset(conn, asset_id):
    """The first measurement annotated on a specific frame, or ``None``."""
    return conn.execute(
        "SELECT * FROM measurements WHERE asset_id = ?"
        " ORDER BY specimen_index, rowid LIMIT 1", (asset_id,)
    ).fetchone()


def measurements_for_asset(conn, asset_id):
    """Every measurement annotated on a frame, by specimen then trait."""
    return conn.execute(
        "SELECT * FROM measurements WHERE asset_id = ?"
        " ORDER BY specimen_index, trait, rowid", (asset_id,)
    ).fetchall()


def list_measurements(conn, limit=500):
    """Recent measurements with their target and session context."""
    return conn.execute(
        "SELECT m.measurement_id, m.length_um, m.length_px, m.px_per_um, m.source,"
        "       m.measured_by, m.created_at, m.asset_id,"
        "       o.well_label, o.individual_index, p.label AS plate_label,"
        "       s.plate_id, s.session_id"
        " FROM measurements m"
        " JOIN observations o ON o.observation_id = m.observation_id"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " LEFT JOIN plates p ON p.plate_id = o.plate_id"
        " ORDER BY m.created_at DESC, m.rowid DESC LIMIT ?", (limit,)
    ).fetchall()


def measurements_for_study(conn, study_id):
    """Every measurement of a study, with its target context, in capture order."""
    return conn.execute(
        "SELECT m.*, o.well_label, o.individual_index, p.label AS plate_label,"
        "       s.study_id"
        " FROM measurements m"
        " JOIN observations o ON o.observation_id = m.observation_id"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " LEFT JOIN plates p ON p.plate_id = o.plate_id"
        " WHERE s.study_id = ?"
        " ORDER BY m.created_at, m.rowid", (study_id,)
    ).fetchall()


def find_target(conn, *, study_id, plate_id=None, well_label=None,
                individual_index=None):
    """Observation of a target if it was captured already, else ``None``."""
    if plate_id and well_label:
        return conn.execute(
            "SELECT o.* FROM observations o"
            " JOIN imaging_sessions s ON s.session_id = o.session_id"
            " WHERE s.study_id = ? AND o.plate_id = ? AND o.well_label = ?"
            " ORDER BY o.rowid LIMIT 1", (study_id, plate_id, well_label)
        ).fetchone()
    return conn.execute(
        "SELECT o.* FROM observations o"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " WHERE s.study_id = ? AND o.individual_index = ?"
        " ORDER BY o.rowid LIMIT 1", (study_id, individual_index)
    ).fetchone()


def frame_counts(conn, observation_id) -> dict:
    """How many frames and how many bursts a target holds."""
    row = conn.execute(
        "SELECT COUNT(*) AS frames, COUNT(DISTINCT burst_index) AS bursts"
        " FROM media_assets WHERE observation_id = ?", (observation_id,)
    ).fetchone()
    return dict(row)


def kept_assets_all(conn, study_id, limit=1000):
    """Every kept frame of a study, with its target label and measurement count.

    This is the annotation queue: one entry per well / individual that the
    selection step marked as the best frame. A frame may hold several measurements
    (specimens x traits), so the column is a count, never a joined row.
    """
    return conn.execute(
        "SELECT a.asset_id, a.file_path, a.burst_index, a.frame_index,"
        "       o.observation_id, o.well_label, o.individual_index,"
        "       p.label AS plate_label,"
        "       (SELECT COUNT(*) FROM measurements m WHERE m.asset_id = a.asset_id)"
        "       AS measurements"
        " FROM media_assets a"
        " JOIN observations o ON o.observation_id = a.observation_id"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " LEFT JOIN plates p ON p.plate_id = o.plate_id"
        " WHERE s.study_id = ? AND a.selection = 'kept'"
        " ORDER BY o.created_at, o.rowid, a.burst_index, a.frame_index LIMIT ?",
        (study_id, limit)
    ).fetchall()


def counts(conn) -> dict:
    """Row counts for the dashboard and the workflow gates."""
    row = conn.execute(
        "SELECT"
        " (SELECT COUNT(*) FROM studies) AS studies,"
        " (SELECT COUNT(*) FROM plates) AS plates,"
        " (SELECT COUNT(*) FROM calibration_profiles) AS profiles,"
        " (SELECT COUNT(*) FROM imaging_sessions) AS sessions,"
        " (SELECT COUNT(*) FROM observations) AS observations,"
        " (SELECT COUNT(*) FROM media_assets) AS assets,"
        " (SELECT COUNT(*) FROM media_assets WHERE selection = 'kept') AS selected,"
        " (SELECT COUNT(*) FROM measurements) AS measurements"
    ).fetchone()
    return dict(row)


# --- the bench: measurements that belong to no study -----------------------


#: Every column a bench result needs, including the landmarks - a bench number
#: without its points cannot be checked or reused.
_BENCH_SELECT = (
    "SELECT m.measurement_id, m.observation_id, m.asset_id, m.specimen_index,"
    "       m.individual_index, m.name,"
    "       m.trait, m.length_um, m.length_px, m.px_per_um, m.uncertainty_pct,"
    "       m.total_uncertainty_um,"
    "       m.p1_x, m.p1_y, m.p2_x, m.p2_y, m.source, m.measured_by, m.created_at,"
    "       m.width_px, m.height_px, m.area_px, m.width_um, m.height_um, m.area_um,"
    "       a.file_path, o.notes"
    " FROM measurements m"
    " JOIN observations o ON o.observation_id = m.observation_id"
    " JOIN imaging_sessions s ON s.session_id = o.session_id"
    " LEFT JOIN media_assets a ON a.asset_id = m.asset_id"
)


def bench_measurements(conn, limit=500, dataset_id=None):
    """The Bench's measurements, newest first.

    With ``dataset_id``: exactly that dataset's rows - which is what "export this
    dataset" means. Without one: the **study-less** rows (the Bench's own work,
    whatever dataset it went into), which are exactly the rows
    :func:`daphnia.db.repository.attach_observation_to_study` can adopt into a study.
    """
    if dataset_id is not None:
        return conn.execute(
            _BENCH_SELECT + " WHERE m.dataset_id IS ?"
            " ORDER BY m.created_at DESC, m.rowid DESC LIMIT ?",
            (dataset_id, limit)).fetchall()
    return conn.execute(
        _BENCH_SELECT + " WHERE s.study_id IS NULL"
        " ORDER BY m.created_at DESC, m.rowid DESC LIMIT ?", (limit,)).fetchall()


def bench_count(conn, dataset_id=None) -> int:
    """How many measurements the Bench holds - of one dataset, or study-less ones."""
    if dataset_id is not None:
        return conn.execute(
            "SELECT COUNT(*) FROM measurements WHERE dataset_id IS ?",
            (dataset_id,)).fetchone()[0]
    return conn.execute(
        "SELECT COUNT(*) FROM measurements m"
        " JOIN observations o ON o.observation_id = m.observation_id"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " WHERE s.study_id IS NULL").fetchone()[0]


def bench_observations(conn, limit=200):
    """The distinct images the bench holds measurements for, newest first."""
    return conn.execute(
        _BENCH_SELECT + " WHERE s.study_id IS NULL"
        " GROUP BY m.observation_id"
        " ORDER BY m.created_at DESC, m.rowid DESC LIMIT ?", (limit,)).fetchall()


def bench_observation_for_file(conn, file_path):
    """The bench observation standing for ``file_path``, or ``None``.

    Used to *read back* what is already measured on an image - without creating an
    observation for a file nobody has measured yet.
    """
    row = conn.execute(
        "SELECT o.observation_id FROM observations o"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " JOIN media_assets a ON a.observation_id = o.observation_id"
        " WHERE s.study_id IS NULL AND a.file_path = ?"
        " ORDER BY o.rowid LIMIT 1", (str(file_path),)).fetchone()
    return row["observation_id"] if row is not None else None


def samples_for_observation(conn, observation_id):
    """Every measurement taken on one bench image, by **individual** then specimen, trait.

    The individual comes first so the rows arrive in the order the surface lists them -
    one animal's numbers together, with the measurements that belong to no individual
    (``individual_index IS NULL``) last.
    """
    return conn.execute(
        "SELECT * FROM measurements WHERE observation_id = ?"
        " ORDER BY individual_index IS NULL, individual_index, specimen_index, trait,"
        " rowid", (observation_id,)).fetchall()


# --- the corpus: ground-truth labels for the training set -------------------


def corpus_labels_for_file(conn, file_path):
    """Every label on one image, by specimen then trait."""
    return conn.execute(
        "SELECT * FROM corpus_labels WHERE file_path = ?"
        " ORDER BY specimen_index, trait", (str(file_path),)).fetchall()


def corpus_labels(conn, limit=100000):
    """Every corpus label, in a deterministic order (the export order)."""
    return conn.execute(
        "SELECT * FROM corpus_labels ORDER BY file_path, specimen_index, trait"
        " LIMIT ?", (limit,)).fetchall()


def corpus_labelled_paths(conn):
    """The set of image paths that carry at least one label."""
    rows = conn.execute("SELECT DISTINCT file_path FROM corpus_labels").fetchall()
    return {row["file_path"] for row in rows}


def corpus_stats(conn) -> dict:
    """What the corpus holds: labels, distinct images, and the per-trait counts."""
    total = conn.execute("SELECT COUNT(*) FROM corpus_labels").fetchone()[0]
    images = conn.execute(
        "SELECT COUNT(DISTINCT file_path) FROM corpus_labels").fetchone()[0]
    by_trait = {row["trait"]: row["n"] for row in conn.execute(
        "SELECT trait, COUNT(*) AS n FROM corpus_labels GROUP BY trait")}
    return {"labels": total, "images": images, "by_trait": by_trait}


def corpus_boxes(conn, limit=100000):
    """Every bounding box of the corpus, in a deterministic order (export order)."""
    return conn.execute(
        "SELECT * FROM corpus_boxes ORDER BY file_path, specimen_index LIMIT ?",
        (limit,)).fetchall()


# --- datasets: one corpus per model ----------------------------------------


def list_datasets(conn):
    """Every dataset, the open one first, then newest first."""
    return conn.execute(
        "SELECT * FROM datasets ORDER BY is_current DESC, created_at DESC, rowid DESC"
    ).fetchall()


def current_dataset(conn):
    """The dataset being labelled, or ``None`` when none exists yet."""
    return conn.execute(
        "SELECT * FROM datasets WHERE is_current = 1 ORDER BY rowid LIMIT 1").fetchone()


def dataset_by_id(conn, dataset_id):
    """One dataset row, or ``None``."""
    return conn.execute("SELECT * FROM datasets WHERE dataset_id = ?",
                        (str(dataset_id),)).fetchone()


def dataset_labels(conn, dataset_id, limit=100000):
    """Every label of one dataset, in a deterministic order (the export order)."""
    return conn.execute(
        "SELECT * FROM corpus_labels WHERE dataset_id IS ?"
        " ORDER BY file_path, specimen_index, trait LIMIT ?",
        (dataset_id, limit)).fetchall()


def dataset_labels_for_file(conn, file_path, dataset_id):
    """Every label of one frame **in one dataset**, by specimen then trait.

    Labels belong to the dataset (model) they were made for, so this never mixes
    one corpus into another - which is what lets two models share a frame.
    """
    return conn.execute(
        "SELECT * FROM corpus_labels WHERE dataset_id IS ? AND file_path = ?"
        " ORDER BY specimen_index, trait",
        (dataset_id, str(file_path))).fetchall()


def dataset_boxes(conn, dataset_id, limit=100000):
    """Every bounding box of one dataset, in a deterministic order."""
    return conn.execute(
        "SELECT * FROM corpus_boxes WHERE dataset_id IS ?"
        " ORDER BY file_path, specimen_index LIMIT ?",
        (dataset_id, limit)).fetchall()


def dataset_boxes_for_file(conn, file_path, dataset_id):
    """``{specimen_index: row}`` - the boxes stored on one frame of one dataset."""
    rows = conn.execute(
        "SELECT * FROM corpus_boxes WHERE dataset_id IS ? AND file_path = ?"
        " ORDER BY specimen_index",
        (dataset_id, str(file_path))).fetchall()
    return {row["specimen_index"]: row for row in rows}


def dataset_stats(conn, dataset_id) -> dict:
    """What one dataset holds: labels, boxes, images, specimens, per trait."""
    labels = conn.execute(
        "SELECT COUNT(*) FROM corpus_labels WHERE dataset_id IS ?",
        (dataset_id,)).fetchone()[0]
    boxes = conn.execute(
        "SELECT COUNT(*) FROM corpus_boxes WHERE dataset_id IS ?",
        (dataset_id,)).fetchone()[0]
    images = conn.execute(
        "SELECT COUNT(DISTINCT file_path) FROM corpus_labels WHERE dataset_id IS ?",
        (dataset_id,)).fetchone()[0]
    box_images = conn.execute(
        "SELECT COUNT(DISTINCT file_path) FROM corpus_boxes WHERE dataset_id IS ?",
        (dataset_id,)).fetchone()[0]
    by_trait = {row["trait"]: row["n"] for row in conn.execute(
        "SELECT trait, COUNT(*) AS n FROM corpus_labels WHERE dataset_id IS ?"
        " GROUP BY trait", (dataset_id,))}
    return {"labels": labels, "boxes": boxes, "images": max(images, box_images),
            "by_trait": by_trait}


# --- imports: the frames a folder or a video brought in ---------------------


def frames_in_batch(conn, batch_id):
    """The frames of one import, in **queue order** (``ordinal``)."""
    return conn.execute(
        "SELECT * FROM imported_frames WHERE batch_id = ? ORDER BY ordinal",
        (str(batch_id),)).fetchall()


def last_import(conn, dataset_id=None):
    """The most recent import (optionally of one dataset), or ``None``.

    The resume pointer is on the batch (``last_ordinal``), so reopening a session
    picks up where the operator stopped instead of at the first frame.
    """
    if dataset_id is None:
        return conn.execute(
            "SELECT * FROM import_batches"
            " ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    return conn.execute(
        "SELECT * FROM import_batches WHERE dataset_id IS ?"
        " ORDER BY created_at DESC, rowid DESC LIMIT 1", (dataset_id,)).fetchone()
