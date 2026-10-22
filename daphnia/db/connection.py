"""SQLite persistence (Persist step): connection handling and schema migration.

The schema lives in ``daphnia/db/schema.sql``. ``init_db()`` is idempotent: it
creates the database and its tables, then applies the small column migrations
needed by databases created by an earlier version of the app.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from daphnia import paths


#: Columns added after the first version. ``CREATE TABLE IF NOT EXISTS`` never
#: alters an existing table, so a database created by an older build has to be
#: patched explicitly - otherwise queries fail with "no such column".
COLUMN_MIGRATIONS = (
    ("imaging_sessions", "study_id", "TEXT REFERENCES studies(study_id)"),
    ("observations", "plate_id", "TEXT"),
    ("observations", "individual_index", "INTEGER"),
    ("media_assets", "burst_index", "INTEGER"),
    ("media_assets", "selection", "TEXT NOT NULL DEFAULT 'pending'"),
    ("measurements", "asset_id", "TEXT"),
    ("studies", "organism_kind", "TEXT NOT NULL DEFAULT 'animal'"),
    ("studies", "is_current", "INTEGER NOT NULL DEFAULT 0"),
    ("studies", "target_mode", "TEXT NOT NULL DEFAULT 'individuals'"),
    ("studies", "timecourse", "TEXT NOT NULL DEFAULT 'longitudinal'"),
    ("studies", "protocol_key", "TEXT NOT NULL DEFAULT 'standard'"),
    ("studies", "closed_at", "TEXT"),
    # Measurements went from "one per frame" to "one per (frame, specimen, trait)".
    ("measurements", "specimen_index", "INTEGER NOT NULL DEFAULT 1"),
    ("measurements", "trait", "TEXT NOT NULL DEFAULT 'length'"),
    ("measurements", "p1_x", "REAL"),
    ("measurements", "p1_y", "REAL"),
    ("measurements", "p2_x", "REAL"),
    ("measurements", "p2_y", "REAL"),
    # A calibration became a dated record of one optical setup, and every
    # measurement now carries the scale uncertainty that applied when it was taken.
    ("calibration_profiles", "setup_key", "TEXT NOT NULL DEFAULT ''"),
    # A calibration can carry the operator's own note ("after refocus"), so the
    # Calibrate screen's per-device list has something human to show and pick by.
    ("calibration_profiles", "label", "TEXT"),
    ("measurements", "uncertainty_pct", "REAL"),
    # A corpus label now belongs to the dataset (model) it was made for. The column
    # is normally created by `rescope_corpus_labels` (an ALTER TABLE cannot replace
    # the old UNIQUE index); this entry is the explicit declaration of the column.
    ("corpus_labels", "dataset_id", "TEXT"),
    # A measurement can now be a **box** (trait 'bbox'), not only a distance: its geometry
    # travels with the same row (width/height/area, px and um). NULL for a line. `X-N06`,
    # `SR-BENCH-015`.
    ("measurements", "width_px", "REAL"),
    ("measurements", "height_px", "REAL"),
    ("measurements", "area_px", "REAL"),
    ("measurements", "width_um", "REAL"),
    ("measurements", "height_um", "REAL"),
    ("measurements", "area_um", "REAL"),
    # A Bench measurement now says **which individual** it is about, so several
    # measurements of one animal can be listed together - and a box can be that animal
    # rather than only one more number beside it. NULL = measured outside every
    # individual (`vision.individuals`, `SR-BENCH-016`). `X-N06`.
    ("measurements", "individual_index", "INTEGER"),
    # The corpus is now the **only** place a measurement lives: a `corpus_labels` row
    # can carry a governed study's provenance (observation + frame), the individual it
    # is about, the scale's uncertainty and a box's geometry - so `measurements` can
    # become a read-only view over it (`unify_measurements_into_corpus`).
    ("corpus_labels", "observation_id", "TEXT"),
    ("corpus_labels", "asset_id", "TEXT"),
    ("corpus_labels", "individual_index", "INTEGER"),
    # The operator's own name for a measurement - the Bench's *Name* column, free text
    # (`SR-BENCH-024`). NULL = nobody gave one, which is not the same as an empty name.
    ("corpus_labels", "label_name", "TEXT"),
    ("corpus_labels", "uncertainty_pct", "REAL"),
    # The whole error budget of a label in micrometres (scale + click + quantisation),
    # so the stored row records what the app claimed the number was worth and not only
    # the scale term (`units.total_uncertainty_um`, `docs/accuracy.md` §1).
    ("corpus_labels", "total_uncertainty_um", "REAL"),
    ("corpus_labels", "width_px", "REAL"),
    ("corpus_labels", "height_px", "REAL"),
    ("corpus_labels", "area_px", "REAL"),
    ("corpus_labels", "width_um", "REAL"),
    ("corpus_labels", "height_um", "REAL"),
    ("corpus_labels", "area_um", "REAL"),
    # A box carries the scale it was drawn under, so its size in micrometres is
    # derivable exactly as a label's `length_um` is (`SR-BENCH-015`), and the governed
    # provenance when a study drew it.
    ("corpus_boxes", "profile_id", "TEXT"),
    ("corpus_boxes", "px_per_um", "REAL"),
    ("corpus_boxes", "observation_id", "TEXT"),
    ("corpus_boxes", "asset_id", "TEXT"),
    # A dataset may belong to a study: the study's measurements land in its dataset.
    ("datasets", "study_id", "TEXT"),
)


def default_db_path() -> Path:
    """Database location (override with the ``DAPHNIA_DB`` environment variable)."""
    return Path(os.environ.get("DAPHNIA_DB", paths.DB_PATH))


def connect(db_path=None) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced and dict-like rows."""
    conn = sqlite3.connect(db_path or default_db_path())
    conn.row_factory = sqlite3.Row
    # Foreign keys are OFF by default in SQLite and must be enabled per connection.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection(db_path=None):
    """Yield a connection that is committed and closed on exit.

    Preferred over ``connect()`` for everyday use::

        with connection() as conn:
            conn.execute(...)
    """
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def table_exists(conn, table) -> bool:
    """True when ``table`` is present in the database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def view_exists(conn, view) -> bool:
    """True when ``view`` is present in the database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'view' AND name = ?", (view,)
    ).fetchone()
    return row is not None


def column_exists(conn, table, column) -> bool:
    """True when ``table`` already has ``column``."""
    if not table_exists(conn, table):
        return False
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


#: Indexes created after the tables. They also enforce the invariant that keeps
#: exactly one measurement per (frame, specimen, trait).
INDEXES = (
    # Templates are versioned by name: appending a version must never collide.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_template_name_version"
    " ON study_templates(name, version)",
    # Calibrations are listed per optical setup, newest first.
    "CREATE INDEX IF NOT EXISTS idx_profile_setup"
    " ON calibration_profiles(setup_key, last_verified_at)",
    # A dataset's name identifies it in a dropdown, and only one is the open one.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_name ON datasets(name)",
    "CREATE INDEX IF NOT EXISTS idx_dataset_current ON datasets(is_current)",
    # The corpus is read per dataset (one model's labels at a time).
    "CREATE INDEX IF NOT EXISTS idx_corpus_label_dataset"
    " ON corpus_labels(dataset_id, file_path)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_box_dataset"
    " ON corpus_boxes(dataset_id, file_path)",
    # The `measurements` view is read per observation (the study side).
    "CREATE INDEX IF NOT EXISTS idx_corpus_label_observation"
    " ON corpus_labels(observation_id)",
)

#: Indexes from earlier versions that no longer fit the model; drop them so a
#: legacy database can store more than one measurement per frame. The unique
#: `measurements` index goes with the table: `measurements` is a view now, and the
#: corpus's own `UNIQUE (dataset_id, file_path, specimen_index, trait)` is the
#: invariant that keeps one row per interval.
OBSOLETE_INDEXES = ("idx_measurement_asset", "idx_measurement_asset_trait")

#: `measurements` is a read-only VIEW over the corpus rows that carry an
#: observation. It exists so every reader written against the old table - the
#: workflow gates, the study export, the analytics screen - keeps working while the
#: one place a measurement is *written* is `corpus_labels` (the dataset store).
MEASUREMENTS_VIEW = """
CREATE VIEW IF NOT EXISTS measurements AS
SELECT label_id AS measurement_id, dataset_id, observation_id, asset_id, profile_id,
       specimen_index, trait, individual_index,
       label_name AS name,
       p1_x, p1_y, p2_x, p2_y, length_px, length_um,
       width_px, height_px, area_px, width_um, height_um, area_um,
       px_per_um, uncertainty_pct, total_uncertainty_um, source,
       labeled_by AS measured_by, created_at,
       corpus_labels.rowid AS rowid
FROM corpus_labels WHERE observation_id IS NOT NULL
"""

#: Columns the view must expose for a database's existing view to be current. Adding a
#: column to :data:`MEASUREMENTS_VIEW` means adding it here, or an old database keeps a
#: view that is missing it and every reader of that column fails (`refresh_measurements_view`).
MEASUREMENTS_VIEW_COLUMNS = ("measurement_id", "dataset_id", "rowid",
                             "total_uncertainty_um", "name")


def dedupe_measurements(conn) -> int:
    """Collapse duplicate measurements of the same (frame, specimen, trait).

    Before the one-measurement-per-interval rule existed, saving the same interval
    twice inserted a second row; the newest row per interval is kept.
    """
    if not table_exists(conn, "measurements"):
        return 0
    if not column_exists(conn, "measurements", "asset_id"):
        return 0
    key = "asset_id"
    for column in ("specimen_index", "trait"):
        if column_exists(conn, "measurements", column):
            key += f", {column}"
    before = conn.execute(
        "SELECT COUNT(*) FROM measurements WHERE asset_id IS NOT NULL").fetchone()[0]
    conn.execute(
        "DELETE FROM measurements WHERE asset_id IS NOT NULL AND rowid NOT IN ("
        f" SELECT MAX(rowid) FROM measurements WHERE asset_id IS NOT NULL"
        f" GROUP BY {key})")
    after = conn.execute(
        "SELECT COUNT(*) FROM measurements WHERE asset_id IS NOT NULL").fetchone()[0]
    return before - after


def backfill_measurement_points(conn) -> int:
    """Copy the legacy eye_apex/spine_base columns into the generic p1/p2 columns."""
    if not table_exists(conn, "measurements"):
        return 0
    if not column_exists(conn, "measurements", "eye_apex_x"):
        return 0
    if not column_exists(conn, "measurements", "p1_x"):
        return 0
    pending = conn.execute(
        "SELECT COUNT(*) FROM measurements"
        " WHERE p1_x IS NULL AND eye_apex_x IS NOT NULL").fetchone()[0]
    if not pending:
        return 0
    conn.execute(
        "UPDATE measurements SET p1_x = eye_apex_x, p1_y = eye_apex_y,"
        " p2_x = spine_base_x, p2_y = spine_base_y WHERE p1_x IS NULL")
    return pending


def backfill_total_uncertainty(conn) -> dict:
    """Give the labels written before this column their whole error budget.

    Two steps, both only ever filling a NULL, both derivable from what the row
    *already* records:

    1. A label that names its calibration (``profile_id``) but carries no
       ``uncertainty_pct`` of its own adopts the calibration's - which is exactly
       what :func:`daphnia.db.repository.save_corpus_label` would have stored, since
       a calibration is an **immutable** record (``SR-OPTICS-016``). Measured on the
       live store 2026-10-22: 519 of 550 rows were in this state, i.e. they recorded
       *which* scale was applied and not how well it was measured.
    2. The budget itself is then the same :func:`daphnia.units.total_uncertainty_um`
       the surfaces print and the write path stores - so an export carries it
       instead of an empty field.

    A label with no scale keeps NULL: it has no micrometres, so it has no budget.
    Returns ``{"scales": n, "budgets": n}``, and both are reported by :func:`migrate`
    rather than changed in silence (``mlContext.md`` law 8).
    """
    result = {"scales": 0, "budgets": 0}
    if not table_exists(conn, "corpus_labels"):
        return result
    if not column_exists(conn, "corpus_labels", "total_uncertainty_um"):
        return result
    from daphnia import units                    # lazy: units is leaf-level, no cycle

    if column_exists(conn, "calibration_profiles", "uncertainty_pct"):
        cursor = conn.execute(
            "UPDATE corpus_labels SET uncertainty_pct = ("
            "  SELECT p.uncertainty_pct FROM calibration_profiles p"
            "  WHERE p.profile_id = corpus_labels.profile_id)"
            " WHERE uncertainty_pct IS NULL AND profile_id IS NOT NULL"
            "   AND EXISTS (SELECT 1 FROM calibration_profiles p"
            "               WHERE p.profile_id = corpus_labels.profile_id"
            "                 AND p.uncertainty_pct IS NOT NULL)")
        result["scales"] = cursor.rowcount or 0

    rows = conn.execute(
        "SELECT label_id, length_um, px_per_um, uncertainty_pct FROM corpus_labels"
        " WHERE total_uncertainty_um IS NULL AND length_um IS NOT NULL"
        "   AND px_per_um > 0 AND uncertainty_pct IS NOT NULL").fetchall()
    for row in rows:
        conn.execute(
            "UPDATE corpus_labels SET total_uncertainty_um = ? WHERE label_id = ?",
            (units.total_uncertainty_um(row["length_um"], row["px_per_um"],
                                        row["uncertainty_pct"]), row["label_id"]))
    result["budgets"] = len(rows)
    return result


def backfill_setup_key(conn) -> int:
    """Give legacy calibration rows their optical-chain key.

    Before calibrations were separated from setups, the readable preset id *was*
    the primary key - and it names exactly the optical chain (scope, objective,
    camera, locked zoom, focus). So it becomes the setup key, and later
    calibrations of that chain join it instead of replacing it.
    """
    if not table_exists(conn, "calibration_profiles"):
        return 0
    if not column_exists(conn, "calibration_profiles", "setup_key"):
        return 0
    cursor = conn.execute(
        "UPDATE calibration_profiles SET setup_key = profile_id"
        " WHERE setup_key = '' OR setup_key IS NULL")
    return cursor.rowcount or 0


def ensure_current_study(conn) -> int:
    """Make sure exactly one study is flagged as the current project.

    Databases created before the flag existed would otherwise look empty, so the
    newest unfinished study is adopted. Returns how many studies were marked.
    """
    if not table_exists(conn, "studies") or not column_exists(conn, "studies", "is_current"):
        return 0
    already = conn.execute(
        "SELECT COUNT(*) FROM studies WHERE is_current = 1").fetchone()[0]
    if already:
        return 0
    cursor = conn.execute(
        "UPDATE studies SET is_current = 1 WHERE rowid = ("
        " SELECT rowid FROM studies ORDER BY (status = 'active') DESC,"
        " created_at DESC, rowid DESC LIMIT 1)")
    return cursor.rowcount or 0


def rebuild_from_schema(conn, table) -> int:
    """Rebuild ``table`` from ``schema.sql``, copying every shared column across.

    SQLite cannot change a column's NOT NULL-ness or drop an implicit UNIQUE index
    in place, so a table whose *constraints* changed is rebuilt: the old one is
    renamed, ``schema.sql`` recreates the new shape, the shared columns are copied,
    and the legacy table is dropped **only after the copy** - so no row is ever
    destroyed. Returns the number of rows carried over.
    """
    legacy = f"{table}_legacy"
    conn.execute(f"ALTER TABLE {table} RENAME TO {legacy}")
    # Re-running the schema creates the new shape; everything else is CREATE IF NOT
    # EXISTS, so nothing else is disturbed.
    conn.executescript(paths.SCHEMA_PATH.read_text(encoding="utf-8"))
    current = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]
    old = [row["name"] for row in conn.execute(f"PRAGMA table_info({legacy})")]
    shared = [name for name in current if name in old]
    columns = ", ".join(shared)
    conn.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM {legacy}")
    moved = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.execute(f"DROP TABLE {legacy}")
    return moved


def rescope_corpus_labels(conn) -> int:
    """Give the corpus labels a dataset scope, keeping every existing row.

    Older databases declared ``UNIQUE (file_path, specimen_index, trait)``, which
    means one frame could carry ground truth for **one** model only - labelling the
    same frame for a second dataset would collide. SQLite cannot drop the implicit
    unique index, so the table is rebuilt (:func:`rebuild_from_schema`). Returns the
    number of rows carried over.
    """
    if not table_exists(conn, "corpus_labels"):
        return 0
    if column_exists(conn, "corpus_labels", "dataset_id"):
        return 0
    return rebuild_from_schema(conn, "corpus_labels")


def relax_corpus_image_size(conn) -> int:
    """Make a label's ``image_width``/``image_height`` nullable, keeping every row.

    A measurement's frame size is known when the frame is on disk - a labelled corpus
    frame always records it - but a row written from a capture whose file is not
    readable (a test, an archived study) has none, and NULL says so rather than
    pretending the frame was zero pixels wide. SQLite cannot drop NOT NULL in place,
    so the table is rebuilt. Returns the number of rows carried over.
    """
    if not table_exists(conn, "corpus_labels"):
        return 0
    info = {row["name"]: row for row in conn.execute("PRAGMA table_info(corpus_labels)")}
    column = info.get("image_width")
    if column is None or not column["notnull"]:
        return 0
    return rebuild_from_schema(conn, "corpus_labels")


def unify_measurements_into_corpus(conn) -> dict:
    """Fold the old ``measurements`` table into the corpus, keeping every row.

    ``corpus_labels`` is the one place a measurement lives now, whichever surface
    made it - the Bench, the Dataset labeller, or a governed study capture. This
    copies every legacy ``measurements`` row into ``corpus_labels`` (attributed to
    the default dataset, its observation and frame recorded), then renames the old
    table to ``measurements_legacy`` - a **backup, never dropped** - and exposes
    ``measurements`` as a read-only view over the corpus (:data:`MEASUREMENTS_VIEW`).

    A legacy row whose frame is not on disk cannot carry the image size the corpus
    requires, so it is **left in the backup and counted**, never silently dropped
    (``mlContext.md`` law 8). Returns ``{"moved": n, "skipped": n}``.
    """
    if not table_exists(conn, "measurements"):
        conn.execute(MEASUREMENTS_VIEW)          # fresh database: the view is enough
        return {"moved": 0, "skipped": 0}

    from daphnia.db import repository        # lazy: repository is the write layer
    from daphnia.vision.images import ImageLoadError, header_size, image_size

    dataset_id = repository.ensure_default_dataset(conn)
    rows = conn.execute(
        "SELECT m.*, a.file_path AS asset_path FROM measurements m"
        " LEFT JOIN media_assets a ON a.asset_id = m.asset_id").fetchall()

    def size_of(path):
        """A frame's size without needing OpenCV; ``None`` when it cannot be read."""
        size = header_size(path)              # stdlib: the app starts without CV
        if size is not None:
            return size
        try:
            return image_size(path)           # a format the header reader cannot parse
        except (ImageLoadError, OSError, ValueError):
            return None

    moved = skipped = 0
    for row in rows:
        path = row["asset_path"]
        size = size_of(path) if path else None
        if size is None:
            skipped += 1
            continue
        width, height = size
        conn.execute(
            "INSERT OR IGNORE INTO corpus_labels ("
            " label_id, dataset_id, file_path, image_width, image_height,"
            " specimen_index, trait, p1_x, p1_y, p2_x, p2_y, length_px,"
            " profile_id, px_per_um, length_um, individual_index,"
            " width_px, height_px, area_px, width_um, height_um, area_um,"
            " uncertainty_pct, observation_id, asset_id, source, labeled_by,"
            " note, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["measurement_id"], dataset_id, str(path), int(width), int(height),
             row["specimen_index"], row["trait"], row["p1_x"], row["p1_y"],
             row["p2_x"], row["p2_y"], row["length_px"], row["profile_id"],
             row["px_per_um"], row["length_um"], row["individual_index"],
             row["width_px"], row["height_px"], row["area_px"], row["width_um"],
             row["height_um"], row["area_um"], row["uncertainty_pct"],
             row["observation_id"], row["asset_id"], row["source"],
             row["measured_by"], "migrated from measurements", row["created_at"]))
        moved += 1

    conn.execute("DROP INDEX IF EXISTS idx_measurement_asset_trait")
    conn.execute("ALTER TABLE measurements RENAME TO measurements_legacy")
    conn.execute(MEASUREMENTS_VIEW)
    return {"moved": moved, "skipped": skipped}


def refresh_measurements_view(conn) -> int:
    """(Re)create the ``measurements`` view when its shape is out of date.

    The view is the compatibility shim over ``corpus_labels``; when a column is added
    to it (the dataset scope, say), an existing view must be replaced - SQLite has no
    ``CREATE OR REPLACE VIEW``. Returns ``1`` when it was rebuilt, else ``0``.
    """
    if view_exists(conn, "measurements"):
        columns = {row["name"]
                   for row in conn.execute("PRAGMA table_info(measurements)")}
        if set(MEASUREMENTS_VIEW_COLUMNS) <= columns:
            return 0
        conn.execute("DROP VIEW measurements")
    conn.execute(MEASUREMENTS_VIEW)
    return 1


def migrate(conn) -> list:
    """Add columns and indexes introduced after a database was first created.

    A column carrying a REFERENCES clause is added with a NULL default, which is
    what SQLite requires for foreign keys added through ALTER TABLE. Tables whose
    *constraints* changed (not only their columns) are rebuilt first - see
    :func:`rescope_corpus_labels`.
    """
    applied = []
    rescoped = rescope_corpus_labels(conn)
    if rescoped:
        applied.append(f"corpus_labels: rebuilt with dataset scope ({rescoped} row(s) kept)")

    relaxed = relax_corpus_image_size(conn)
    if relaxed:
        applied.append(f"corpus_labels: image size is optional now ({relaxed} row(s) kept)")

    for table, column, definition in COLUMN_MIGRATIONS:
        if table_exists(conn, table) and not column_exists(conn, table, column):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            applied.append(f"{table}.{column}")

    moved = backfill_measurement_points(conn)
    if moved:
        applied.append(f"measurements: backfilled {moved} landmark row(s)")

    keyed = backfill_setup_key(conn)
    if keyed:
        applied.append(f"calibration_profiles: keyed {keyed} row(s) to their setup")

    budgeted = backfill_total_uncertainty(conn)
    if budgeted["scales"]:
        applied.append(f"corpus_labels: adopted the calibration's uncertainty for"
                       f" {budgeted['scales']} row(s)")
    if budgeted["budgets"]:
        applied.append(f"corpus_labels: derived the total uncertainty of"
                       f" {budgeted['budgets']} row(s)")

    removed = dedupe_measurements(conn)
    if removed:
        applied.append(f"measurements: removed {removed} duplicate row(s)")

    adopted = ensure_current_study(conn)
    if adopted:
        applied.append("studies: adopted 1 study as the current project")

    # The corpus is the one place a measurement lives now: fold the legacy
    # `measurements` table into it and expose `measurements` as a view.
    unified = unify_measurements_into_corpus(conn)
    if unified["moved"] or unified["skipped"]:
        note = f"measurements: {unified['moved']} row(s) folded into the corpus"
        if unified["skipped"]:
            note += (f"; {unified['skipped']} left in measurements_legacy - their "
                     "frame is not on disk, so the image size the corpus needs is unknown")
        applied.append(note)

    if refresh_measurements_view(conn):
        applied.append("measurements: view rebuilt over the corpus")

    for index in OBSOLETE_INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {index}")
    for statement in INDEXES:
        conn.execute(statement)
    return applied


def init_db(db_path=None) -> Path:
    """Create the database, its tables and any missing columns (idempotent)."""
    path = Path(db_path or default_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        conn.executescript(paths.SCHEMA_PATH.read_text(encoding="utf-8"))
        migrate(conn)
        conn.commit()
    finally:
        conn.close()
    return path


def table_names(conn) -> list:
    """Return the user table names currently present in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [row["name"] for row in rows]


if __name__ == "__main__":
    path = init_db()
    with connection(path) as conn:
        tables = table_names(conn)
    print(f"Initialised {path}")
    print("Tables: " + ", ".join(tables))
