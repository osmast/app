"""Data-access layer: commands and point lookups - all writing SQL lives here.

Reporting read models (lists for the UI, dashboard counters) are in
:mod:`daphnia.db.queries`. Pages never build SQL themselves.
"""

from pathlib import Path

from daphnia import annotation, protocols, settings
from daphnia.ids import new_calibration_id, new_id
from daphnia.vision.calibration import OpticalProfile
from daphnia.vision.keypoints import Keypoints

SELECTION_STATES = ("pending", "kept", "discarded")


# --- studies ---------------------------------------------------------------


def create_study(conn, *, name, destination, study_id=None, organism="Daphnia magna",
                 organism_kind="animal", target_mode="individuals",
                 timecourse="longitudinal", measurement_interval_hours=24.0,
                 protocol_key=protocols.DEFAULT_PROTOCOL) -> str:
    """Create the study that owns this run's images; returns its id.

    ``study_id`` may be supplied so the caller can pre-compute the destination
    folder (``paths.study_dir(study_id)``). The folder is created immediately, so
    every later capture has a known home instead of writing to a default location.
    """
    study_id = study_id or new_id("STUDY")
    Path(destination).mkdir(parents=True, exist_ok=True)
    # A new study becomes the project being worked on.
    conn.execute("UPDATE studies SET is_current = 0")
    conn.execute(
        "INSERT INTO studies (study_id, name, organism, organism_kind, target_mode,"
        " timecourse, measurement_interval_hours, destination, protocol_key, is_current)"
        " VALUES (?,?,?,?,?,?,?,?,?,1)",
        (study_id, name, organism, organism_kind, target_mode, timecourse,
         measurement_interval_hours, str(destination), protocol_key))
    return study_id


def set_current_study(conn, study_id) -> str:
    """Make ``study_id`` the project being worked on.

    Exactly one study carries the flag at a time; the others stay in the list so they
    can be reopened later.
    """
    conn.execute("UPDATE studies SET is_current = 0")
    conn.execute("UPDATE studies SET is_current = 1 WHERE study_id = ?", (study_id,))
    return study_id


def close_study(conn, study_id) -> str:
    """Close a study and stamp the moment it was consolidated.

    The current-project flag is left as it is, so the existing contract of the
    overview screen (a just-finished run stays selected) is preserved.
    """
    conn.execute(
        "UPDATE studies SET status = 'closed', closed_at = datetime('now')"
        " WHERE study_id = ?", (study_id,))
    return study_id


# --- study templates --------------------------------------------------------


def save_study_template(conn, *, name, organism="Daphnia magna",
                        organism_kind="animal", target_mode="individuals",
                        timecourse="longitudinal", measurement_interval_hours=24.0,
                        protocol_key=protocols.DEFAULT_PROTOCOL) -> str:
    """Append a new version of a named template; returns its id.

    A template is never overwritten: saving under an existing name stores the next
    version, so the earlier ones stay available as history.
    """
    row = conn.execute(
        "SELECT MAX(version) AS version FROM study_templates WHERE name = ?",
        (name,)).fetchone()
    version = (row["version"] or 0) + 1
    template_id = new_id("TPL")
    conn.execute(
        "INSERT INTO study_templates (template_id, name, version, organism,"
        " organism_kind, target_mode, timecourse, measurement_interval_hours,"
        " protocol_key) VALUES (?,?,?,?,?,?,?,?,?)",
        (template_id, name, version, organism, organism_kind, target_mode,
         timecourse, measurement_interval_hours, protocol_key))
    return template_id


def template_by_id(conn, template_id):
    """One template row, or ``None``."""
    return conn.execute(
        "SELECT * FROM study_templates WHERE template_id = ?", (template_id,)
    ).fetchone()


# --- plates ----------------------------------------------------------------


def create_plate(conn, *, study_id, format, label=None) -> str:
    """Add a multi-well plate to the study; returns its id.

    The format alone defines the wells to iterate (``daphnia.plates``); treatment
    assignment on top of the geometry comes with the plate builder.
    """
    plate_id = new_id("PLATE")
    existing = conn.execute(
        "SELECT COUNT(*) FROM plates WHERE study_id = ?", (study_id,)).fetchone()[0]
    conn.execute(
        "INSERT INTO plates (plate_id, study_id, format, label) VALUES (?,?,?,?)",
        (plate_id, study_id, str(format), label or f"Plate {existing + 1}"))
    return plate_id


def delete_plate(conn, plate_id) -> str:
    """Remove a plate (only valid while nothing was captured for it)."""
    conn.execute("DELETE FROM plates WHERE plate_id = ?", (plate_id,))
    return plate_id


# --- calibration profiles --------------------------------------------------


def upsert_profile(conn, *, profile_id, profile_name, microscope_model,
                   objective_magnification, camera_device, px_per_um,
                   adapter_model=None, camera_zoom_ratio=1.0,
                   focus_mode="infinity_locked", uncertainty_pct=0.5) -> str:
    """Insert a profile, or update it when the id already exists.

    **Legacy behaviour:** recalibrating the same optics reuses the profile id, so
    the new scale *replaces* the previous one and ``last_verified_at`` is
    refreshed. That loses the scale earlier measurements were taken under, which is
    why new code records a calibration with :func:`register_calibration` instead.
    """
    conn.execute(
        "INSERT INTO calibration_profiles (profile_id, profile_name, microscope_model,"
        " objective_magnification, adapter_model, camera_device, camera_zoom_ratio,"
        " focus_mode, px_per_um, uncertainty_pct) VALUES (?,?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT(profile_id) DO UPDATE SET"
        "   profile_name = excluded.profile_name,"
        "   microscope_model = excluded.microscope_model,"
        "   objective_magnification = excluded.objective_magnification,"
        "   adapter_model = excluded.adapter_model,"
        "   camera_device = excluded.camera_device,"
        "   camera_zoom_ratio = excluded.camera_zoom_ratio,"
        "   focus_mode = excluded.focus_mode,"
        "   px_per_um = excluded.px_per_um,"
        "   uncertainty_pct = excluded.uncertainty_pct,"
        "   last_verified_at = datetime('now')",
        (profile_id, profile_name, microscope_model, objective_magnification,
         adapter_model, camera_device, camera_zoom_ratio, focus_mode, px_per_um,
         uncertainty_pct))
    return profile_id


def register_calibration(conn, *, setup_key, profile_name, microscope_model,
                         objective_magnification, camera_device, px_per_um,
                         uncertainty_pct=0.5, adapter_model=None,
                         camera_zoom_ratio=1.0, focus_mode="infinity_locked",
                         label=None, activate=True) -> str:
    """Record a new calibration of one optical setup; returns its calibration id.

    A calibration is a *dated measurement of the scale*, so it is immutable: this
    always inserts a new row instead of rewriting an existing one. Recalibrating
    mid-study - a bumped adapter, a changed objective, a new session - therefore
    adds a calibration, and the measurements already taken keep pointing at the
    scale (and the uncertainty) they were actually taken under.

    ``setup_key`` names the optical chain (see :func:`daphnia.ids.new_profile_id`)
    and groups every calibration taken on that hardware.
    """
    # A quick correction can land in the same second as the calibration it fixes;
    # the timestamp-based id must still be unique.
    profile_id = new_calibration_id(setup_key)
    suffix = 0
    while conn.execute(
            "SELECT 1 FROM calibration_profiles WHERE profile_id = ?",
            (profile_id,)).fetchone() is not None:
        suffix += 1
        profile_id = f"{new_calibration_id(setup_key)}_{suffix}"

    conn.execute(
        "INSERT INTO calibration_profiles (profile_id, profile_name, setup_key,"
        " microscope_model, objective_magnification, adapter_model, camera_device,"
        " camera_zoom_ratio, focus_mode, px_per_um, uncertainty_pct, label, is_active)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)",
        (profile_id, profile_name, setup_key, microscope_model,
         objective_magnification, adapter_model, camera_device, camera_zoom_ratio,
         focus_mode, px_per_um, uncertainty_pct, label))
    if activate:
        activate_profile(conn, profile_id)
    return profile_id


def insert_calibration_record(conn, *, profile_id, target_type, raw_image_path,
                              detected_distance_px, actual_distance_um,
                              calibrated_by=None, notes=None) -> str:
    """Insert the proof-image row of a calibration; returns its id."""
    record_id = new_id("CREC")
    conn.execute(
        "INSERT INTO calibration_records (record_id, profile_id, target_type,"
        " raw_image_path, detected_distance_px, actual_distance_um, calibrated_by, notes)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (record_id, profile_id, target_type, str(raw_image_path),
         detected_distance_px, actual_distance_um,
         calibrated_by or settings.OPERATOR_NAME, notes))
    return record_id


def activate_profile(conn, profile_id) -> str:
    """Make ``profile_id`` the single active profile."""
    conn.execute("UPDATE calibration_profiles SET is_active = 0")
    conn.execute(
        "UPDATE calibration_profiles SET is_active = 1,"
        " last_verified_at = datetime('now') WHERE profile_id = ?", (profile_id,))
    return profile_id


def active_profile(conn):
    """The active :class:`~daphnia.vision.calibration.OpticalProfile`, or ``None``."""
    row = conn.execute(
        "SELECT * FROM calibration_profiles WHERE is_active = 1"
        " ORDER BY last_verified_at DESC LIMIT 1").fetchone()
    return OpticalProfile.from_row(row) if row else None


def profile_by_id(conn, profile_id):
    """Look up a single profile by id."""
    row = conn.execute(
        "SELECT * FROM calibration_profiles WHERE profile_id = ?", (profile_id,)).fetchone()
    return OpticalProfile.from_row(row) if row else None


def calibrations_for_setup(conn, setup_key, limit=50) -> list:
    """Every calibration of one optical setup, newest first.

    This is the audit trail of the scale: what it was, when it was verified, how
    accurate the reference span made it, and who did it.
    """
    return conn.execute(
        "SELECT * FROM calibration_profiles WHERE setup_key = ?"
        " ORDER BY last_verified_at DESC, rowid DESC LIMIT ?",
        (setup_key, limit)).fetchall()


def calibrations(conn, limit=None) -> list:
    """Every calibration in the store, newest first - across all optical setups.

    The Calibrate screen lists these so the **proof photo** of a past calibration is
    one click away, which is the only way to judge a stored scale from the outside.
    It is deliberately **not** filtered by ``setup_key``: the calibration a measurement
    was taken under may belong to a chain the operator has since renamed, and a list
    that empties itself whenever the preset name changes reads as a broken control.
    """
    query = ("SELECT * FROM calibration_profiles"
             " ORDER BY last_verified_at DESC, rowid DESC")
    if limit is None:
        return conn.execute(query).fetchall()
    return conn.execute(f"{query} LIMIT ?", (limit,)).fetchall()


def proof_record(conn, profile_id):
    """The proof-image row of one calibration, or ``None``.

    ``None`` is a real answer rather than a failure: a scale the operator typed in has
    no photo behind it (``SR-OPTICS-029``), so a caller must say so instead of showing
    whatever image happens to be on screen.
    """
    return conn.execute(
        "SELECT * FROM calibration_records WHERE profile_id = ?"
        " ORDER BY created_at DESC, rowid DESC LIMIT 1", (profile_id,)).fetchone()


def latest_calibration(conn, setup_key=None):
    """The newest calibration - of one setup, or of any - or ``None``.

    The active calibration is the one in force; the latest one is what a re-verify
    prompt looks at, even when nothing has been activated yet.
    """
    if setup_key:
        row = conn.execute(
            "SELECT * FROM calibration_profiles WHERE setup_key = ?"
            " ORDER BY last_verified_at DESC, rowid DESC LIMIT 1",
            (setup_key,)).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM calibration_profiles"
            " ORDER BY last_verified_at DESC, rowid DESC LIMIT 1").fetchone()
    return OpticalProfile.from_row(row) if row else None


# --- sessions --------------------------------------------------------------


def ensure_session(conn, *, study_id, profile_id, operator=None) -> str:
    """Return today's in-progress session of the study, creating it if needed.

    A session locks the optical profile used at runtime and anchors every
    observation, media asset and measurement of the day.
    """
    operator = operator or settings.OPERATOR_NAME
    row = conn.execute(
        "SELECT session_id FROM imaging_sessions"
        " WHERE study_id = ? AND session_status = 'in_progress'"
        "   AND profile_id = ? AND date(started_at) = date('now')"
        " ORDER BY started_at DESC LIMIT 1", (study_id, profile_id)).fetchone()
    if row is not None:
        return row["session_id"]

    session_id = new_id("SESS")
    conn.execute(
        "INSERT INTO imaging_sessions (session_id, study_id, plate_id, profile_id,"
        " day_number, elapsed_seconds, operator_name) VALUES (?, ?, NULL, ?, 0, 0, ?)",
        (session_id, study_id, profile_id, operator))
    return session_id


# --- capture targets -------------------------------------------------------


def next_individual_index(conn, study_id) -> int:
    """Next number of the simple individual counter for a study."""
    row = conn.execute(
        "SELECT MAX(o.individual_index) AS last FROM observations o"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " WHERE s.study_id = ?", (study_id,)).fetchone()
    return (row["last"] or 0) + 1


def create_observation(conn, *, session_id, plate_id=None, well_label=None,
                       individual_index=None, notes=None) -> str:
    """Register one capture target (a well or one individual); returns its id."""
    observation_id = new_id("OBS")
    conn.execute(
        "INSERT INTO observations (observation_id, session_id, plate_id, well_label,"
        " individual_index, notes) VALUES (?,?,?,?,?,?)",
        (observation_id, session_id, plate_id, well_label, individual_index, notes))
    return observation_id


def target_observation(conn, *, session_id, plate_id=None, well_label=None,
                       individual_index=None) -> str:
    """Observation of a target, created on first use.

    One target accumulates every burst taken of that specimen, so re-visiting a
    well or an individual appends to its existing observation instead of
    scattering the frames over several rows.
    """
    if plate_id and well_label:
        row = conn.execute(
            "SELECT observation_id FROM observations"
            " WHERE session_id = ? AND plate_id = ? AND well_label = ?"
            " ORDER BY rowid LIMIT 1",
            (session_id, plate_id, well_label)).fetchone()
    elif individual_index is not None:
        row = conn.execute(
            "SELECT observation_id FROM observations"
            " WHERE session_id = ? AND individual_index = ?"
            " ORDER BY rowid LIMIT 1",
            (session_id, individual_index)).fetchone()
    else:
        raise ValueError("a target needs a well (plate_id + well_label) or an individual index")

    if row is not None:
        return row["observation_id"]
    return create_observation(conn, session_id=session_id, plate_id=plate_id,
                              well_label=well_label, individual_index=individual_index)


def next_burst_index(conn, observation_id) -> int:
    """Next burst number for a target."""
    row = conn.execute(
        "SELECT MAX(burst_index) AS last FROM media_assets WHERE observation_id = ?",
        (observation_id,)).fetchone()
    return (row["last"] or 0) + 1


# --- media assets ----------------------------------------------------------


def add_media_asset(conn, *, observation_id, file_path, kind="burst",
                    frame_index=None, burst_index=None, sharpness=None,
                    is_primary=False, selection="pending") -> str:
    """Register a frame that was written to disk; returns its id."""
    asset_id = new_id("AST")
    conn.execute(
        "INSERT INTO media_assets (asset_id, observation_id, file_path, kind,"
        " frame_index, burst_index, is_primary, sharpness, selection)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (asset_id, observation_id, str(file_path), kind, frame_index, burst_index,
         int(bool(is_primary)), sharpness, selection))
    return asset_id


def set_asset_selection(conn, asset_id, selection) -> str:
    """Mark a frame as a good candidate, discarded, or undecided."""
    if selection not in SELECTION_STATES:
        raise ValueError(f"selection must be one of {SELECTION_STATES}")
    conn.execute("UPDATE media_assets SET selection = ? WHERE asset_id = ?",
                 (selection, asset_id))
    return asset_id


def set_primary_asset(conn, observation_id, asset_id) -> str:
    """Make ``asset_id`` the single primary frame of its observation."""
    conn.execute(
        "UPDATE media_assets SET is_primary = 0 WHERE observation_id = ?",
        (observation_id,))
    conn.execute("UPDATE media_assets SET is_primary = 1 WHERE asset_id = ?", (asset_id,))
    return asset_id


def select_best_frame(conn, observation_id, asset_id) -> str:
    """Mark one frame of a target as THE good one and send the rest to the tray.

    Selection is one-per-well on purpose: a well or individual yields exactly one
    measurement, so exactly one frame per target is kept.
    """
    conn.execute(
        "UPDATE media_assets SET selection = 'discarded' WHERE observation_id = ?",
        (observation_id,))
    conn.execute(
        "UPDATE media_assets SET selection = 'kept' WHERE asset_id = ?", (asset_id,))
    return asset_id


def clear_selection(conn, observation_id) -> str:
    """Reset every frame of a target back to 'pending'."""
    conn.execute(
        "UPDATE media_assets SET selection = 'pending' WHERE observation_id = ?",
        (observation_id,))
    return observation_id


def select_sharpest_frame(conn, observation_id):
    """Keep the sharpest frame of a target; returns its id or ``None``."""
    row = conn.execute(
        "SELECT asset_id FROM media_assets WHERE observation_id = ?"
        " ORDER BY sharpness DESC, rowid LIMIT 1", (observation_id,)).fetchone()
    if row is None:
        return None
    return select_best_frame(conn, observation_id, row["asset_id"])


# --- measurements ----------------------------------------------------------


def frame_size(file_path):
    """``(width, height)`` of a frame, or ``(None, None)`` when it cannot be read.

    The corpus records a frame's size because it *is* the coordinate space of the
    landmarks. A frame that is not on disk (an archived study, a test's placeholder
    path) has no size to record, and ``None`` says exactly that.
    """
    if not file_path:
        return (None, None)
    from daphnia.vision.images import ImageLoadError, header_size, image_size
    size = header_size(file_path)                    # stdlib: no OpenCV needed
    if size is None:
        try:
            size = image_size(file_path)
        except (ImageLoadError, OSError, ValueError):
            size = None
    return (None, None) if size is None else (int(size[0]), int(size[1]))


def dataset_for_study(conn, study_id) -> str:
    """The dataset a study's measurements belong to, created on first use.

    Every measurement lives in a dataset now, so a study owns one too - the corpus
    its captures are ground truth for. ``datasets.study_id`` records the link; a
    study may own several datasets ("another model = another dataset"), and this is
    the first one, the one its measurements go to. A study-less measurement (the
    Bench) uses the default dataset.
    """
    if not study_id:
        return ensure_default_dataset(conn)
    row = conn.execute(
        "SELECT dataset_id FROM datasets WHERE study_id = ? ORDER BY rowid LIMIT 1",
        (str(study_id),)).fetchone()
    if row is not None:
        return row["dataset_id"]
    study = conn.execute("SELECT name FROM studies WHERE study_id = ?",
                         (str(study_id),)).fetchone()
    name = f"{study['name'] if study is not None else 'Study'} - measurements"
    return create_dataset(conn, name=name, schema_key=annotation.DEFAULT_SCHEMA,
                          study_id=study_id, make_current=False)


def save_measurement(conn, *, observation_id, profile, trait, p1, p2,
                     specimen_index=1, asset_id=None, measured_by=None,
                     source="manual", box=None, individual_index=None,
                     name=None, dataset_id=None) -> str:
    """Store one interval (or one **box**) as a **corpus label**; returns its id.

    There is one place a measurement lives - the corpus - so this is a thin adapter
    over :func:`save_corpus_label`: it resolves the frame (path and size) and the
    dataset (the study's, or the default for a study-less capture), records the
    governed provenance (``observation_id``/``asset_id``) that the ``measurements``
    view reads back, and delegates. ``profile`` is **required**: a governed
    measurement is never stored without a scale (``projectbrief.md`` non-negotiable 1).

    A measurement is the distance between two clicked landmarks (``p1``, ``p2``) for a
    ``trait`` of a ``specimen_index`` on a frame. ``box`` is
    ``(width_px, height_px, area_px, width_um, height_um, area_um)`` for a box
    measurement (``trait = annotation.BOX_TRAIT``), whose ``p1``/``p2`` are two opposite
    corners. Saving the **same (frame, specimen, trait)** again updates that row.

    ``individual_index`` is the **individual this number is about** - the box that stands
    for one animal (`vision.individuals`), ``None`` when the measurement falls inside no
    box. It is deliberately *not* part of the row key: two ``length`` lines on one animal
    are two measurements of it.

    ``name`` is the operator's **own name** for the measurement (`SR-BENCH-024`,
    ``corpus_labels.label_name``): free text, ``None`` when they gave it none. Like
    ``individual_index`` it is deliberately *not* part of the row key - renaming a
    measurement is a correction of that row, not a second measurement.

    ``dataset_id`` chooses the corpus the label belongs to - the dataset the operator
    selected on the Bench, or the study's own when it is ``None`` (the default).
    """
    if profile is None:
        raise ValueError(
            "a study measurement needs a scale - calibrate first "
            "(python run.py optics), then measure again")

    asset = None
    if asset_id is not None:
        asset = conn.execute("SELECT * FROM media_assets WHERE asset_id = ?",
                             (asset_id,)).fetchone()
    if asset is None:
        asset = conn.execute(
            "SELECT * FROM media_assets WHERE observation_id = ? ORDER BY rowid LIMIT 1",
            (observation_id,)).fetchone()
    if asset is None:
        raise ValueError(
            f"observation {observation_id} has no frame - add a media asset before "
            "measuring it")

    session = conn.execute(
        "SELECT s.study_id FROM observations o"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " WHERE o.observation_id = ?", (observation_id,)).fetchone()
    if dataset_id is None:
        dataset_id = dataset_for_study(conn, session["study_id"] if session else None)
    return save_corpus_label(
        conn, file_path=asset["file_path"], image_size=frame_size(asset["file_path"]),
        trait=trait, p1=p1, p2=p2, specimen_index=specimen_index, profile=profile,
        source=source, labeled_by=measured_by, dataset_id=dataset_id,
        observation_id=observation_id, asset_id=asset["asset_id"],
        individual_index=individual_index, name=name, box=box)


def save_manual_measurement(conn, *, observation_id, profile, eye_apex, spine_base,
                            asset_id=None, measured_by=None, source="manual") -> str:
    """Store the *length* (head -> spine base) of the first specimen on a frame.

    Thin, backwards-compatible wrapper over :func:`save_measurement`.
    """
    return save_measurement(
        conn, observation_id=observation_id, profile=profile, trait="length",
        specimen_index=1, p1=eye_apex, p2=spine_base, asset_id=asset_id,
        measured_by=measured_by, source=source)


# --- the bench: measuring without a study ----------------------------------


def bench_observation(conn, *, file_path, profile_id, operator=None) -> str:
    """The observation standing for one imported image; created on first use.

    Bench work belongs to **no study**: the session it creates carries
    ``study_id = NULL``, which is what makes measuring possible before a project
    exists. Each image gets its *own* session on purpose - attaching an image to a
    study later must not drag every other bench image along with it.

    The image is identified by its file, so re-opening the same photo continues the
    same observation and a re-measurement updates instead of duplicating.
    """
    operator = operator or settings.OPERATOR_NAME
    row = conn.execute(
        "SELECT o.observation_id FROM observations o"
        " JOIN imaging_sessions s ON s.session_id = o.session_id"
        " JOIN media_assets a ON a.observation_id = o.observation_id"
        " WHERE s.study_id IS NULL AND a.file_path = ?"
        " ORDER BY o.rowid LIMIT 1", (str(file_path),)).fetchone()
    if row is not None:
        return row["observation_id"]

    session_id = new_id("SESS")
    conn.execute(
        "INSERT INTO imaging_sessions (session_id, study_id, plate_id, profile_id,"
        " day_number, elapsed_seconds, operator_name)"
        " VALUES (?, NULL, NULL, ?, 0, 0, ?)", (session_id, profile_id, operator))
    observation_id = create_observation(
        conn, session_id=session_id, notes=f"bench: {Path(file_path).name}")
    add_media_asset(conn, observation_id=observation_id, file_path=file_path,
                    kind="imported", is_primary=True, selection="kept")
    return observation_id


def bench_asset(conn, observation_id):
    """The frame row of a bench observation, or ``None``."""
    return conn.execute(
        "SELECT * FROM media_assets WHERE observation_id = ?"
        " ORDER BY rowid LIMIT 1", (observation_id,)).fetchone()


def attach_observation_to_study(conn, *, observation_id, study_id) -> str:
    """Promote a bench observation onto a study, without re-measuring anything.

    This is what the study-less session exists for: work done quickly, before a
    project was set up, can be adopted once one is - and the measurements keep the
    scale they were taken under, because that is recorded on the row.
    """
    conn.execute(
        "UPDATE imaging_sessions SET study_id = ? WHERE session_id ="
        " (SELECT session_id FROM observations WHERE observation_id = ?)",
        (study_id, observation_id))
    return observation_id


def delete_measurements_for_observation(conn, observation_id) -> int:
    """Remove every measurement stored on one observation; returns the count.

    A measurement is a corpus row that carries the observation, so this deletes
    those rows - not just the points drawn on the canvas.
    """
    cursor = conn.execute(
        "DELETE FROM corpus_labels WHERE observation_id = ?", (str(observation_id),))
    return cursor.rowcount or 0


def delete_measurement(conn, *, observation_id, specimen_index=1, trait="length") -> int:
    """Remove one stored measurement (specimen + trait of one observation)."""
    cursor = conn.execute(
        "DELETE FROM corpus_labels WHERE observation_id = ? AND specimen_index = ?"
        " AND trait = ?",
        (str(observation_id), int(specimen_index), trait))
    return cursor.rowcount or 0


# --- the corpus: scale-free ground-truth landmarks -------------------------


def save_corpus_label(conn, *, file_path, image_size, trait, p1, p2,
                      specimen_index=1, profile=None, source="manual",
                      labeled_by=None, note=None, dataset_id=None,
                      observation_id=None, asset_id=None, individual_index=None,
                      name=None, box=None) -> str:
    """Store one measurement - a ground-truth landmark interval - in the corpus.

    This is **the one write path for a measurement**, whichever surface made it.
    A corpus label is **scale-free** on purpose: it records the clicked keypoints in
    *source-image pixels* (plus the image size, which is the coordinate space), so
    any frame can be labelled with no study, plate, connection or calibration - the
    model learns pixels, and archived or third-party images must be usable too. When
    ``profile`` is given the pixel distance is also converted, and the scale that
    produced it is stored alongside.

    ``dataset_id`` names the model this label is ground truth *for* (see
    :mod:`daphnia.annotation`). ``observation_id``/``asset_id`` record the governed
    capture a study measurement came from, which is what the ``measurements`` view
    filters on; a Bench/Dataset label leaves them ``None``. ``individual_index`` is
    the individual the number is about (`vision.individuals`); ``name`` is the
    operator's **own name** for the measurement (``label_name``, `SR-BENCH-024`), free
    text and ``None`` when they gave it none; ``box`` is a box
    measurement's ``(w_px, h_px, a_px, w_um, h_um, a_um)``. Saving the same
    ``(dataset, file, specimen, trait)`` again **updates** that row instead of adding
    a duplicate, so re-labelling - or renaming - a specimen is a correction, not a fork.
    """
    keypoints = Keypoints(eye_apex=tuple(p1), spine_base=tuple(p2))
    operator = labeled_by or settings.OPERATOR_NAME
    width = None if image_size is None or image_size[0] is None else int(image_size[0])
    height = None if image_size is None or image_size[1] is None else int(image_size[1])
    profile_id = profile.profile_id if profile is not None else None
    px_per_um = profile.px_per_um if profile is not None else None
    length_um = profile.to_um(keypoints.length_px) if profile is not None else None
    uncertainty = profile.uncertainty_pct if profile is not None else None
    # The **whole** budget of this number, not only the scale term: the same
    # `units.total_uncertainty_um` the surfaces print, so a stored row and the
    # screen that produced it cannot disagree (`docs/accuracy.md` §1).
    total_uncertainty = (profile.total_uncertainty_um(length_um)
                         if profile is not None else None)
    individual = None if individual_index is None else int(individual_index)
    label_name = None if name is None else (str(name).strip() or None)
    box_geometry = tuple(box) if box is not None else (None,) * 6

    existing = conn.execute(
        "SELECT label_id FROM corpus_labels WHERE dataset_id IS ?"
        " AND file_path = ? AND specimen_index = ? AND trait = ?",
        (dataset_id, str(file_path), int(specimen_index), trait)).fetchone()

    if existing is not None:
        conn.execute(
            "UPDATE corpus_labels SET image_width = ?, image_height = ?,"
            " p1_x = ?, p1_y = ?, p2_x = ?, p2_y = ?, length_px = ?,"
            " profile_id = ?, px_per_um = ?, length_um = ?, individual_index = ?,"
            " label_name = ?,"
            " width_px = ?, height_px = ?, area_px = ?, width_um = ?, height_um = ?,"
            " area_um = ?, uncertainty_pct = ?, total_uncertainty_um = ?,"
            " observation_id = ?, asset_id = ?,"
            " source = ?, labeled_by = ?, note = ?, created_at = datetime('now')"
            " WHERE label_id = ?",
            (width, height, keypoints.eye_apex[0], keypoints.eye_apex[1],
             keypoints.spine_base[0], keypoints.spine_base[1],
             keypoints.length_px, profile_id, px_per_um, length_um, individual, label_name)
            + box_geometry + (uncertainty, total_uncertainty, observation_id,
                              asset_id, source, operator, note,
                              existing["label_id"]))
        return existing["label_id"]

    label_id = new_id("LBL")
    conn.execute(
        "INSERT INTO corpus_labels (label_id, dataset_id, file_path, image_width,"
        " image_height, specimen_index, trait, p1_x, p1_y, p2_x, p2_y, length_px,"
        " profile_id, px_per_um, length_um, individual_index, label_name,"
        " width_px, height_px,"
        " area_px, width_um, height_um, area_um, uncertainty_pct,"
        " total_uncertainty_um, observation_id,"
        " asset_id, source, labeled_by, note)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (label_id, dataset_id, str(file_path), width, height, int(specimen_index),
         trait, keypoints.eye_apex[0], keypoints.eye_apex[1],
         keypoints.spine_base[0], keypoints.spine_base[1], keypoints.length_px,
         profile_id, px_per_um, length_um, individual, label_name) + box_geometry
        + (uncertainty, total_uncertainty, observation_id, asset_id, source,
           operator, note))
    return label_id


# --- datasets: one corpus per model ----------------------------------------


def dataset_named(conn, name):
    """The dataset with this exact name, or ``None``."""
    return conn.execute("SELECT * FROM datasets WHERE name = ?",
                        (str(name),)).fetchone()


def next_dataset_name(conn, name) -> str:
    """``name`` when it is free, else ``name (2)``, ``name (3)``, ...

    A dataset's identity is its ``dataset_id``; the name is what an operator reads
    in the list. So a collision is resolved by numbering rather than by failing -
    a name is not worth losing someone's labelling session over - and the caller
    can report the name that was actually used.
    """
    wanted = str(name).strip() or "Untitled"
    if dataset_named(conn, wanted) is None:
        return wanted
    index = 2
    while dataset_named(conn, f"{wanted} ({index})") is not None:
        index += 1
    return f"{wanted} ({index})"


def rename_dataset(conn, dataset_id, name) -> str:
    """Rename a dataset; returns the name it ended up with.

    **The model cannot change here.** The schema is what its labels mean, so a
    different model would invalidate every one of them - that is why "another
    model" is another dataset. Renaming keeps the id, so labels, boxes and the
    export history all stay attached to the same corpus.
    """
    row = conn.execute("SELECT name FROM datasets WHERE dataset_id = ?",
                       (str(dataset_id),)).fetchone()
    if row is None:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    wanted = str(name).strip() or row["name"]
    if wanted == row["name"]:
        return wanted
    final = next_dataset_name(conn, wanted)
    conn.execute("UPDATE datasets SET name = ? WHERE dataset_id = ?",
                 (final, str(dataset_id)))
    return final


def create_dataset(conn, *, name, schema_key=annotation.DEFAULT_SCHEMA,
                   organism="Daphnia magna", note=None, dataset_id=None,
                   study_id=None, make_current=True) -> str:
    """Create a dataset (one model's corpus) and make it the open one.

    The schema key is what fixes *what gets labelled* - two keypoints, four, a box
    or a combination - so "changing the model" means creating a dataset, never
    re-purposing an existing corpus. The name is made unique (``next_dataset_name``)
    so a repeated name can never fail the insert. Returns the new id.

    ``study_id`` links the corpus to a governed study (a study's measurements land
    in its dataset); ``make_current=False`` creates it without stealing the open
    dataset - which is what a study's auto-created corpus does.
    """
    dataset_id = dataset_id or new_id("DATA")
    unique = next_dataset_name(conn, name)
    if make_current:
        conn.execute("UPDATE datasets SET is_current = 0")
    conn.execute(
        "INSERT INTO datasets (dataset_id, name, schema_key, organism, note,"
        " study_id, is_current) VALUES (?,?,?,?,?,?,?)",
        (dataset_id, unique, annotation.get(schema_key).key, organism, note,
         study_id, 1 if make_current else 0))
    return dataset_id


def activate_dataset(conn, dataset_id) -> int:
    """Make one dataset the open one; returns the number of rows changed."""
    conn.execute("UPDATE datasets SET is_current = 0")
    cursor = conn.execute("UPDATE datasets SET is_current = 1 WHERE dataset_id = ?",
                          (str(dataset_id),))
    return cursor.rowcount or 0


def adopt_legacy_corpus(conn, dataset_id) -> int:
    """Attribute labels made *before* datasets existed to their owning dataset.

    An earlier build stored corpus labels with no dataset (``dataset_id IS NULL``).
    They were labelled against exactly the default vocabulary, so the default dataset
    adopts them - otherwise perfectly good labels would silently disappear from the
    surface that exists to show them. Nothing else about the rows changes; only the
    attribution is filled in. Returns how many rows were adopted.
    """
    adopted = 0
    for table in ("corpus_labels", "corpus_boxes"):
        cursor = conn.execute(
            f"UPDATE {table} SET dataset_id = ? WHERE dataset_id IS NULL",
            (str(dataset_id),))
        adopted += cursor.rowcount or 0
    return adopted


def ensure_default_dataset(conn) -> str:
    """The current dataset's id, creating the default corpus when there is none.

    A corpus has to exist before a label can belong to one, and the default is the
    vocabulary the app measured with before datasets existed (head, tail and both
    sides) - so arriving here with an empty database changes nothing an operator
    sees, and an existing corpus is adopted rather than stranded. Returns the id.
    """
    row = conn.execute(
        "SELECT dataset_id FROM datasets WHERE is_current = 1 LIMIT 1").fetchone()
    if row is not None:
        dataset_id = row["dataset_id"]
    else:
        row = conn.execute(
            "SELECT dataset_id FROM datasets ORDER BY rowid LIMIT 1").fetchone()
        if row is not None:
            dataset_id = row["dataset_id"]
            activate_dataset(conn, dataset_id)
        else:
            dataset_id = create_dataset(conn, name=DEFAULT_DATASET_NAME,
                                        schema_key=annotation.DEFAULT_SCHEMA)
    # Whatever the path taken, labels that belong to no dataset (they were made
    # before datasets existed) are this corpus's - a no-op once they are attributed.
    adopt_legacy_corpus(conn, dataset_id)
    return dataset_id


#: The name the auto-created corpus gets - named for what it is, not "default".
DEFAULT_DATASET_NAME = "Daphnia - head, tail & sides"


def delete_corpus_label(conn, *, file_path, specimen_index=1, trait="length",
                        dataset_id=None) -> int:
    """Remove one corpus label; returns the number of rows removed."""
    cursor = conn.execute(
        "DELETE FROM corpus_labels WHERE dataset_id IS ? AND file_path = ?"
        " AND specimen_index = ? AND trait = ?",
        (dataset_id, str(file_path), int(specimen_index), trait))
    return cursor.rowcount or 0


def delete_corpus_labels_for_file(conn, file_path, dataset_id=None) -> int:
    """Remove the labels of one frame **of one dataset**; returns the rows removed.

    The corpus spans many studies and many frames, and a frame can be re-done from
    scratch - so clearing a frame has to remove its stored labels, not just the points
    currently drawn on the canvas. ``dataset_id=None`` means the legacy, unscoped
    rows only, so clearing one dataset never touches another model's ground truth.
    """
    cursor = conn.execute(
        "DELETE FROM corpus_labels WHERE dataset_id IS ? AND file_path = ?",
        (dataset_id, str(file_path)))
    return cursor.rowcount or 0


def delete_corpus_box(conn, *, file_path, specimen_index=1,
                      dataset_id=None) -> int:
    """Remove one specimen's bounding box; returns the number of rows removed."""
    cursor = conn.execute(
        "DELETE FROM corpus_boxes WHERE dataset_id IS ? AND file_path = ?"
        " AND specimen_index = ?",
        (dataset_id, str(file_path), int(specimen_index)))
    return cursor.rowcount or 0


def delete_corpus_boxes_for_file(conn, file_path, dataset_id=None) -> int:
    """Remove every bounding box of one frame (of one dataset)."""
    cursor = conn.execute(
        "DELETE FROM corpus_boxes WHERE dataset_id IS ? AND file_path = ?",
        (dataset_id, str(file_path)))
    return cursor.rowcount or 0


def set_corpus_box_source(conn, *, file_path, specimen_index=1, dataset_id=None,
                          source="manual", note=None) -> int:
    """Say where one stored box came from - the human's **accept** of a prediction.

    A prediction is not ground truth until a person accepts it (``SR-PREDICT-N01``,
    ``mlContext.md`` law 7), so the accept is recorded by moving the row from
    ``source='model'`` to ``source='manual'`` and naming the model that proposed it
    in ``note``. ``SR-DATASET-008`` needs exactly this split to report how well the
    two agree - which is only possible while both are distinguishable.

    Returns the number of rows changed (``0`` when there is no such box).
    """
    cursor = conn.execute(
        "UPDATE corpus_boxes SET source = ?, note = COALESCE(?, note),"
        " labeled_by = ? WHERE dataset_id IS ? AND file_path = ?"
        " AND specimen_index = ?",
        (source, note, settings.OPERATOR_NAME, dataset_id, str(file_path),
         int(specimen_index)))
    return cursor.rowcount or 0


def save_corpus_box(conn, *, file_path, image_size, box, dataset_id=None,
                    specimen_index=1, source="manual", labeled_by=None,
                    note=None, profile=None, observation_id=None,
                    asset_id=None) -> str:
    """Store one specimen's bounding box for the corpus - one box, one individual.

    ``box`` is ``(x1, y1, x2, y2)`` in *source-image pixels* - the corners are
    normalised so a drag in any direction stores the same rectangle. ``profile``
    records the scale the box was drawn under, so its size in micrometres is
    derivable exactly as a label's ``length_um`` is (`SR-BENCH-015`);
    ``observation_id``/``asset_id`` record a governed capture. Saving the same
    ``(dataset, file, specimen)`` again updates the row instead of adding a second
    box, so correcting a box is an edit, not a fork.
    """
    operator = labeled_by or settings.OPERATOR_NAME
    width = None if image_size is None or image_size[0] is None else int(image_size[0])
    height = None if image_size is None or image_size[1] is None else int(image_size[1])
    profile_id = profile.profile_id if profile is not None else None
    px_per_um = profile.px_per_um if profile is not None else None
    values = list(box)
    if len(values) == 4:                    # (x1, y1, x2, y2)
        ax, ay, bx, by = values
    else:                                   # ((x1, y1), (x2, y2))
        (ax, ay), (bx, by) = values
    x1, x2 = sorted((float(ax), float(bx)))
    y1, y2 = sorted((float(ay), float(by)))

    existing = conn.execute(
        "SELECT box_id FROM corpus_boxes WHERE dataset_id IS ? AND file_path = ?"
        " AND specimen_index = ?",
        (dataset_id, str(file_path), int(specimen_index))).fetchone()
    if existing is not None:
        conn.execute(
            "UPDATE corpus_boxes SET image_width = ?, image_height = ?,"
            " x1 = ?, y1 = ?, x2 = ?, y2 = ?, profile_id = ?, px_per_um = ?,"
            " observation_id = ?, asset_id = ?, source = ?, labeled_by = ?,"
            " note = ?, created_at = datetime('now') WHERE box_id = ?",
            (width, height, x1, y1, x2, y2, profile_id, px_per_um,
             observation_id, asset_id, source, operator, note,
             existing["box_id"]))
        return existing["box_id"]

    box_id = new_id("BOX")
    conn.execute(
        "INSERT INTO corpus_boxes (box_id, dataset_id, file_path, image_width,"
        " image_height, specimen_index, x1, y1, x2, y2, profile_id, px_per_um,"
        " observation_id, asset_id, source, labeled_by, note)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (box_id, dataset_id, str(file_path), width, height, int(specimen_index),
         x1, y1, x2, y2, profile_id, px_per_um, observation_id, asset_id,
         source, operator, note))
    return box_id


# --- imports: a folder, a video or a single file becomes a queue -------------


def frame_index_of(file_path):
    """The frame number inside a video, read from an ``f000123.png`` name, or ``None``.

    The name is the provenance: :func:`daphnia.vision.video.extract_frames` writes
    frames zero-padded, so the file name *is* the frame's place in the clip and the
    path order is the frame order.
    """
    name = Path(file_path).stem
    if len(name) > 1 and name[0] == "f" and name[1:].isdigit():
        return int(name[1:])
    return None


def open_import(conn, *, kind, paths, source=None, dataset_id=None,
                operator=None) -> str:
    """Record one import and the frames it brought in; returns the batch id.

    A queue used to be an in-memory list, rebuilt on every import: nothing recorded
    where the frames came from, re-importing a folder duplicated the work, and a
    video's frame 123 was only a file name. A batch fixes all three - ``source``
    names the folder or clip, ``frame_index`` is the frame's place in the video, and
    ``UNIQUE (batch_id, file_path)`` makes a repeated import the same work.
    """
    operator = operator or settings.OPERATOR_NAME
    batch_id = new_id("IMP")
    conn.execute(
        "INSERT INTO import_batches (batch_id, kind, source, dataset_id, operator)"
        " VALUES (?,?,?,?,?)",
        (batch_id, kind, str(source) if source else None, dataset_id, operator))
    for ordinal, path in enumerate(paths):
        conn.execute(
            "INSERT OR IGNORE INTO imported_frames (frame_id, batch_id, file_path,"
            " frame_index, ordinal) VALUES (?,?,?,?,?)",
            (new_id("FRM"), batch_id, str(path), frame_index_of(path), ordinal))
    return batch_id


def resume_import(conn, batch_id, ordinal) -> int:
    """Remember where the operator stopped in an import; returns the new pointer."""
    cursor = conn.execute(
        "UPDATE import_batches SET last_ordinal = ? WHERE batch_id = ?",
        (int(ordinal), str(batch_id)))
    return cursor.rowcount or 0
