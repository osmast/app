-- Zoo study app — SQLite schema (decoupled profile calibration).
-- Loaded by db.init_db(); safe to run repeatedly (CREATE TABLE IF NOT EXISTS).

PRAGMA foreign_keys = ON;

-- 1. Hardware & optical configuration presets.
--    One row per *calibration*: the scale measured on an optical chain at a point
--    in time. Recalibrating (a bumped adapter, a changed objective, a new session)
--    inserts a NEW row rather than editing this one, so a measurement keeps
--    pointing at the scale and the uncertainty it was actually taken under - see
--    repository.register_calibration.
CREATE TABLE IF NOT EXISTS calibration_profiles (
    profile_id TEXT PRIMARY KEY,               -- one calibration: '<setup_key>_<stamp>'
    profile_name TEXT NOT NULL,                -- 'Leica DM500 4x / iPhone 15 wide 1.0x'
    setup_key TEXT NOT NULL DEFAULT '',        -- the optical chain: scope+objective+camera+zoom+focus
    label TEXT,                                -- the operator's own note for THIS calibration
    microscope_model TEXT NOT NULL,
    objective_magnification TEXT NOT NULL,     -- '2x', '4x', '10x'
    adapter_model TEXT,                        -- 'LabCam 3D Mount v2'
    camera_device TEXT NOT NULL,               -- 'iPhone 15 Pro'
    camera_zoom_ratio REAL NOT NULL DEFAULT 1.0, -- must be locked to 1.0x
    focus_mode TEXT NOT NULL DEFAULT 'infinity_locked',

    -- Calibrated scale target
    px_per_um REAL NOT NULL,                   -- e.g., 0.2145 (pixels per micrometer)
    uncertainty_pct REAL DEFAULT 0.5,          -- relative scale error, derived from the clicked span
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_verified_at TEXT NOT NULL DEFAULT (datetime('now'))  -- when THIS calibration was made
);

-- 2. Audit trail for optical calibrations (proof images)
CREATE TABLE IF NOT EXISTS calibration_records (
    record_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    target_type TEXT NOT NULL,                 -- 'stage_micrometer_100um', 'calibration_paper_10um', 'calibrated_grid_1mm'
    raw_image_path TEXT NOT NULL,              -- file path to reference photo
    detected_distance_px REAL NOT NULL,        -- measured pixels between pitch marks
    actual_distance_um REAL NOT NULL,          -- known distance (e.g. 1000 um)
    calculated_px_per_um REAL GENERATED ALWAYS AS (detected_distance_px / actual_distance_um) STORED,
    calibrated_by TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (profile_id) REFERENCES calibration_profiles(profile_id) ON DELETE CASCADE
);

-- 3. Sessions lock the optical profile used at runtime
CREATE TABLE IF NOT EXISTS imaging_sessions (
    session_id TEXT PRIMARY KEY,               -- 'SESS_20261014_P1_D03'
    study_id TEXT,                             -- the study this run belongs to (owns the image folder)
    plate_id TEXT,                             -- NULL = ad-hoc / single specimen (until the plate builder exists)
    profile_id TEXT NOT NULL,                  -- profile providing px_per_um scale
    day_number INTEGER NOT NULL,               -- Day 0, Day 1, Day 2...
    elapsed_seconds REAL NOT NULL,             -- allows arbitrary minute/hour/day intervals
    session_status TEXT NOT NULL DEFAULT 'in_progress', -- 'in_progress', 'completed', 'flagged'
    current_well_index INTEGER DEFAULT 0,      -- resume pointer (e.g., well 14 of 96)
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    operator_name TEXT NOT NULL,
    FOREIGN KEY (profile_id) REFERENCES calibration_profiles(profile_id),
    FOREIGN KEY (study_id) REFERENCES studies(study_id)
);

-- 4. One specimen capture event: a well inside a session, or an ad-hoc single
--    specimen measured outside any plate layout.
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    plate_id TEXT,                             -- NULL when counting individuals
    well_label TEXT,                           -- 'E07' inside a plate
    individual_index INTEGER,                  -- 1..n of the simple individual counter
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES imaging_sessions(session_id) ON DELETE CASCADE
);

-- 5. Every frame written to disk for an observation (a burst holds several).
CREATE TABLE IF NOT EXISTS media_assets (
    asset_id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'burst',        -- 'burst', 'calibration_reference'
    frame_index INTEGER,
    burst_index INTEGER,                       -- which burst of the target this frame belongs to
    is_primary INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0, 1)),
    sharpness REAL,                            -- variance of Laplacian at capture time
    selection TEXT NOT NULL DEFAULT 'pending' CHECK (selection IN ('pending', 'kept', 'discarded')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (observation_id) REFERENCES observations(observation_id) ON DELETE CASCADE
);

-- 6. Measurements are **corpus labels** now (table 10 below). There is no separate
--    `measurements` table: one measurement of one specimen on one frame is a
--    `corpus_labels` row, whichever surface made it - the Bench, the Dataset
--    labeller, or a governed study capture (which additionally carries
--    `observation_id`/`asset_id`). `measurements` survives as a read-only VIEW
--    over the rows that carry an observation, so every existing reader keeps
--    working while the write path lives in the corpus. See `db/connection.py`:
--    `unify_measurements_into_corpus` builds it, and `MEASUREMENTS_VIEW` defines it.

-- 8. A multi-well plate of a study. Its format defines the wells to iterate;
--    treatment assignment on top of it comes with the plate builder.
CREATE TABLE IF NOT EXISTS plates (
    plate_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL,
    format TEXT NOT NULL CHECK (format IN ('6', '24', '96')),
    label TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (study_id) REFERENCES studies(study_id)
);

-- 7. A study (experiment). Creating one is the first workflow step, because it
--    decides where the images of this run are stored and how often they are taken.
CREATE TABLE IF NOT EXISTS studies (
    study_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    organism TEXT NOT NULL DEFAULT 'Daphnia magna',
    -- Mode axes: they configure what the rest of the app shows.
    organism_kind TEXT NOT NULL DEFAULT 'animal',        -- 'animal' | 'plant'
    target_mode TEXT NOT NULL DEFAULT 'individuals',     -- 'plates' | 'individuals'
    timecourse TEXT NOT NULL DEFAULT 'longitudinal',     -- 'longitudinal' | 'single'
    measurement_interval_hours REAL NOT NULL DEFAULT 24.0,  -- daily by default
    destination TEXT NOT NULL,               -- root folder for this study's images
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closed')),
    is_current INTEGER NOT NULL DEFAULT 0 CHECK (is_current IN (0, 1)),  -- the project being worked on
    protocol_key TEXT NOT NULL DEFAULT 'standard',   -- acquisition protocol (see daphnia.protocols)
    closed_at TEXT,                                  -- set when the run is consolidated and closed
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 11. Datasets: one corpus per model (the DATASET surface's subject).
--     A dataset is *what a model is trained to predict*: two keypoints, four
--     keypoints, a bounding box, or several of those at once. The schema key
--     points into daphnia.annotation, so which points get clicked, what is
--     exported and what the model consumes all follow from this one choice -
--     changing the model means adding a dataset, never re-purposing one.
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,                        -- 'Daphnia 4-point pose, batch 1'
    schema_key TEXT NOT NULL DEFAULT 'pose_full',  -- see daphnia.annotation.SCHEMAS
    organism TEXT NOT NULL DEFAULT 'Daphnia magna',
    note TEXT,
    -- The governed study this corpus belongs to, when there is one. A dataset is
    -- the container of *every* measurement now - bench, dataset and study alike -
    -- so a study's measurements land in the study's dataset; NULL means a
    -- standalone corpus (the Bench and the Dataset labeller). One study may own
    -- several datasets, because "another model = another dataset".
    study_id TEXT,
    is_current INTEGER NOT NULL DEFAULT 0 CHECK (is_current IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 10. Ground-truth landmarks for the training corpus (the DATASET surface).
--     A label records the clicked keypoints in *source-image pixels*, so a frame
--     can be labelled with no study, plate, connection or calibration at all.
--     When a scale is in force the pixel distance is also converted and the scale
--     that produced it is stored alongside - but the corpus itself is scale-free,
--     because the model learns pixels and the corpus must accept archived and
--     third-party images too. One row per (file, specimen, trait): re-labelling
--     the same interval updates it instead of adding a duplicate.
CREATE TABLE IF NOT EXISTS corpus_labels (
    label_id TEXT PRIMARY KEY,
    dataset_id TEXT,                             -- which model's corpus (NULL = legacy)
    file_path TEXT NOT NULL,                     -- the labelled frame
    -- Source pixels are the coordinate space. The size is known when the frame is
    -- on disk (a labelled corpus frame always records it); a measurement whose file
    -- is not readable - an archived study, a test - has none, and NULL says so
    -- rather than pretending the frame was zero pixels wide.
    image_width INTEGER,
    image_height INTEGER,
    specimen_index INTEGER NOT NULL DEFAULT 1,   -- which animal on the frame
    trait TEXT NOT NULL DEFAULT 'length',        -- 'length' | 'width' | 'bbox'
    p1_x REAL NOT NULL,                          -- first landmark (image px)
    p1_y REAL NOT NULL,
    p2_x REAL NOT NULL,                          -- second landmark (image px)
    p2_y REAL NOT NULL,
    length_px REAL NOT NULL,                     -- distance between the two landmarks
    profile_id TEXT,                             -- the scale in force, when there was one
    px_per_um REAL,
    length_um REAL,
    -- Which **individual** this number is about (`vision.individuals`): 1..N in the
    -- order the frame's individuals were boxed, or NULL when it falls inside no box.
    -- It answers a different question from `specimen_index`, which is only *which row
    -- on the frame this is* - keeping the two apart is what lets one individual hold
    -- two `length` rows without the second erasing the first (`SR-BENCH-016`).
    individual_index INTEGER,
    -- The operator's **own name** for this measurement (`SR-BENCH-024`): the Bench's *Name*
    -- column - 'Gut length', 'width', 'repeat 2'. Free text on purpose: it is the lab's word
    -- for what was measured, not a vocabulary the app fixes (that is `trait`), and NULL says
    -- nobody gave one rather than that the name is empty.
    label_name TEXT,
    -- A **box** measurement (trait 'bbox') stores its geometry here, in px and um;
    -- NULL for a distance - a line has no width or area (`SR-BENCH-015`).
    width_px REAL,
    height_px REAL,
    area_px REAL,
    width_um REAL,
    height_um REAL,
    area_um REAL,
    uncertainty_pct REAL,                        -- relative scale error that applied
    -- The **whole** budget of this length in micrometres: the scale above combined in
    -- quadrature with the two terms that come from measuring at all - where the two
    -- landmarks were clicked (`units.CLICK_ERROR_PX`) and the pixel grid they had to
    -- land on (`units.QUANTISATION_PX`). Written by `repository.save_corpus_label`
    -- from the same `units.total_uncertainty_um` the surfaces print, so the stored row
    -- and the screen cannot disagree, and derived once for older rows by
    -- `connection.backfill_total_uncertainty`. NULL = no scale was applied, so the
    -- label has no micrometres and no budget (`docs/accuracy.md` §1).
    total_uncertainty_um REAL,
    -- The governed-capture provenance. A study measurement carries the observation
    -- and the frame it was taken on; a Bench/Dataset label leaves them NULL. This
    -- pair is what the `measurements` view filters on.
    observation_id TEXT,
    asset_id TEXT,
    source TEXT NOT NULL DEFAULT 'manual' CHECK (source IN ('manual', 'model')),
    labeled_by TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- A label belongs to the dataset (model) it was made for, so the same frame
    -- can carry grounds truth for two different models side by side.
    UNIQUE (dataset_id, file_path, specimen_index, trait)
);

-- 10b. Bounding boxes: where each specimen is, for a detector.
--      A box is not a distance between two landmarks, so it gets its own table
--      instead of overloading the keypoint rows - but it is scoped to a dataset
--      and a specimen exactly like they are, and export merges the two.
CREATE TABLE IF NOT EXISTS corpus_boxes (
    box_id TEXT PRIMARY KEY,
    dataset_id TEXT,                             -- which model's corpus (NULL = legacy)
    file_path TEXT NOT NULL,
    image_width INTEGER,
    image_height INTEGER,
    specimen_index INTEGER NOT NULL DEFAULT 1,
    x1 REAL NOT NULL,                            -- top-left corner (image px)
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,                            -- bottom-right corner (image px)
    y2 REAL NOT NULL,
    -- The scale in force when the box was drawn, so its size in micrometres is
    -- derivable exactly as a label's `length_um` is (`SR-BENCH-015`). NULL = none.
    profile_id TEXT,
    px_per_um REAL,
    observation_id TEXT,                         -- governed-capture provenance, else NULL
    asset_id TEXT,
    source TEXT NOT NULL DEFAULT 'manual' CHECK (source IN ('manual', 'model')),
    labeled_by TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (dataset_id, file_path, specimen_index)
);

-- 12. One import: a folder, a video, a single file, a live capture, or a boxes
--     export. It is the unit a queue is rebuilt from and a session resumed by, so
--     importing the same source twice is the same work, not a second copy.
CREATE TABLE IF NOT EXISTS import_batches (
    batch_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('folder', 'video', 'single', 'live', 'boxes')),
    source TEXT,                                 -- the folder/video/file it came from (NULL = live)
    dataset_id TEXT,                             -- the corpus it was imported into (NULL = none yet)
    last_ordinal INTEGER NOT NULL DEFAULT 0,     -- the resume pointer (cf. imaging_sessions.current_well_index)
    operator TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 13. One frame of a batch: what a queue item points at. The *file* is the
--     identity - a measurement and a label both key on `file_path` - and the batch
--     adds where it came from and where in the source it sat (a video's frame
--     number). UNIQUE(batch_id, file_path) is what makes re-importing idempotent.
CREATE TABLE IF NOT EXISTS imported_frames (
    frame_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    frame_index INTEGER,                         -- the frame number inside a video (NULL otherwise)
    ordinal INTEGER NOT NULL,                    -- the queue order (for a video: the frame order)
    image_width INTEGER,
    image_height INTEGER,
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (batch_id, file_path),
    FOREIGN KEY (batch_id) REFERENCES import_batches(batch_id) ON DELETE CASCADE
);

-- 9. Reusable, versioned starting points for a study (see daphnia.templates).
--    Saving under an existing name appends a new version; older versions stay as
--    history, so a template is never overwritten in place.
-- 9. Reusable, versioned starting points for a study (see daphnia.templates).
--    Saving under an existing name appends a new version; older versions stay as
--    history, so a template is never overwritten in place.
CREATE TABLE IF NOT EXISTS study_templates (
    template_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    organism TEXT NOT NULL DEFAULT 'Daphnia magna',
    organism_kind TEXT NOT NULL DEFAULT 'animal',
    target_mode TEXT NOT NULL DEFAULT 'individuals',
    timecourse TEXT NOT NULL DEFAULT 'longitudinal',
    measurement_interval_hours REAL NOT NULL DEFAULT 24.0,
    protocol_key TEXT NOT NULL DEFAULT 'standard',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 10. The model registry: which checkpoints exist and which one is in force.
--     One row per checkpoint *file*, and exactly one `is_active` row per kind (enforced
--     by daphnia.models.activate, which clears the others). It records what a checkpoint
--     *is* - its path, preset, training date and the metrics it earned - so a window can
--     offer to load one and any surface can ask which model the lab is using without
--     importing PyTorch (see daphnia/models.py).
CREATE TABLE IF NOT EXISTS models (
    kind TEXT NOT NULL CHECK (kind IN ('detector', 'pose')),
    path TEXT NOT NULL,                          -- the checkpoint file
    name TEXT NOT NULL,                          -- its file name, for a menu entry
    preset TEXT,                                 -- 'tiny' / 'small' / ...
    trained_at TEXT,                             -- when the checkpoint was written
    classes TEXT,                                -- comma-separated, detector only
    keypoints TEXT,                              -- comma-separated, pose only
    metrics_json TEXT NOT NULL DEFAULT '{}',     -- the numbers it earned (never a claim)
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
    PRIMARY KEY (kind, path)
);

