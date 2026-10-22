"""The *Models* menu: load a checkpoint, and say which one is in force.

Which detector and which pose model the lab is using is one decision, and it used to be
made twice: the Predict surface had *Load detector* / *Open checkpoint...* / *Open pose
model...* buttons in its own panel, and no other surface could see the result. The menu
lives in the window **header** instead (`X-N13`), so it is there on every page, and every
load is recorded in the registry (:mod:`daphnia.models`) - *activate it in the database* is
the same act as loading it.

Two properties it keeps deliberately:

* **PyTorch comes in here, and only here** (and only when a checkpoint is actually
  loaded): the menu imports :mod:`daphnia.vision.rtmdet` / :mod:`daphnia.vision.rtmpose`
  inside the call, so a window with no model still starts without the ML stack
  (``SR-CORE-011``).
* **A failure is a message, not a traceback** - and not a silent empty view either: a
  missing checkpoint, a missing PyTorch or a damaged file is reported in the window's
  status bar with the fix the message names.
"""

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from daphnia import models, paths
from daphnia.db import connection
from daphnia.ui.widgets.header_bar import MODEL_MENUS

#: What a checkpoint file looks like to a file dialog.
CHECKPOINT_FILTER = "Checkpoints (*.pt *.pth);;All files (*)"


def hand_over(page, kind, model):
    """Give a freshly loaded ``model`` to ``page``; return the note to show instead.

    A page that can *use* a model declares ``adopt_models`` (the Predict surface does) and
    returns whether it took it; for any other page - and for a caller with no page at all,
    like the hub - the load is still **real**: the registry now says which model is active
    and the next surface to open it loads the same one. The caller then says what happened,
    because *a load nobody can see is a load nobody trusts*.

    Returns ``None`` when the page took the model, else the sentence to show.

    One implementation, three windows: the hub, a surface window and the study shell ask the
    same question, and none of them may answer it differently.
    """
    adopter = getattr(page, "adopt_models", None)
    if callable(adopter) and adopter(**{kind: model}):
        return None
    label = models.KIND_LABELS[kind]
    return (f"{label} loaded: {model.describe()}   -   registered as the active "
            f"{label.lower()}.")


def loader_for(kind):
    """``(the callable that loads this kind, the exception it documents)``.

    The import happens here, so the ML stack is only ever imported when a checkpoint is
    actually loaded - and the exception class comes with it, so a caller can report the
    *documented* failure without importing PyTorch itself. Shared by the *Models* menus and
    by the model overlay every canvas surface draws (`daphnia/ui/model_pass.py`).
    """
    if kind == models.KINDS[0]:
        from daphnia.vision.rtmdet.detect import ModelUnavailable, load_detector
        return load_detector, ModelUnavailable
    from daphnia.vision.rtmpose.detect import ModelUnavailable, load_pose_model
    return load_pose_model, ModelUnavailable


class ModelMenu(QObject):
    """The **Detector (RTMDet)** and **Pose (RTMPose)** menus of one window.

    One menu per kind, titled with the network's own name (`models.KIND_MODELS`): the
    detector and the pose model are two decisions with two checkpoints, two kinds of number
    and two reports, and a single *Models* menu made the operator read the same five entries
    twice to find out which half they belonged to.

    Signals:
        loaded(str kind, object model): a checkpoint was loaded - the window hands it to
            the page on screen (a page that can use a model declares ``adopt_models``).
        failed(str message): the load did not happen, and the message says why.
    """

    loaded = Signal(str, object)
    failed = Signal(str)

    def __init__(self, header_bar, parent=None, models_dir=None):
        super().__init__(parent)
        self.header_bar = header_bar
        self.window = parent
        self.directory = Path(models_dir) if models_dir is not None else models.models_dir()
        self.menus = {kind: header_bar.menu(title)
                      for kind, title in MODEL_MENUS.items()}
        # `loaded` is a *signal* (below): what this window has loaded is a dict, and the
        # two must not share a name - shadowing the signal broke every window once.
        self.loaded_paths = {}           # kind -> the checkpoint this window loaded
        self.refresh()

    # -- building ----------------------------------------------------------
    def refresh(self):
        """Read the training reports, then rebuild both menus from the **registry**.

        The reports first, because a checkpoint the lab trained five minutes ago is in force
        whether or not anybody has opened it in the app yet (`models.sync_from_reports`); the
        database is then the one place that says which checkpoint that is.

        Each menu shows two facts, and they are different ones:

        * a **check mark** on the checkpoint that is **in force** (the registry's active row);
        * a **check mark on the top entry** when this window has that kind **loaded** - so
          "which model is answering right now" is visible where the loading happens.

        Under them come the checkpoints **available in the models directory** (by name,
        `models.checkpoints_in`), registered or not: a file the lab just trained is one click
        away without a file dialog.
        """
        with connection() as conn:
            models.sync_from_reports(conn, self.directory)
            records = models.records(conn)
        by_kind = {kind: [row for row in records if row.kind == kind]
                   for kind in models.KINDS}

        for kind, menu in self.menus.items():
            menu.clear()
            label = models.KIND_LABELS[kind].lower()
            network = models.KIND_MODELS[kind]
            active = next((row for row in by_kind[kind] if row.is_active), None)
            loaded_path = self.loaded_paths.get(kind)

            entry = menu.addAction(f"Load the active {label} ({network})",
                                   lambda _c=False, k=kind: self.use_active(k))
            entry.setCheckable(True)
            entry.setChecked(bool(loaded_path))
            entry.setToolTip(
                f"{active.path}\n{active.detail}" if active is not None
                else f"Nothing registered yet - 'Open a {label} checkpoint...' (or a file "
                     f"from the list below) registers one.")
            menu.addAction(f"Open a {label} checkpoint...",
                           lambda _c=False, k=kind: self.load(k))
            menu.actions()[-1].setToolTip(f"Starts in {self.directory}")

            rows = self._checkpoints(kind, by_kind[kind])
            if rows:
                menu.addSection(f"checkpoints in {self.directory.name}/")
            for path, row in rows:
                choice = menu.addAction(self._entry_text(path, row, kind))
                choice.setCheckable(True)
                choice.setChecked(bool(row is not None and row.is_active))
                choice.setToolTip(self._entry_tooltip(path, row, kind))
                choice.triggered.connect(
                    lambda _c=False, k=kind, p=path: self.activate(k, p))
        self.header_bar.hide_empty_menus()
        return records

    def _checkpoints(self, kind, records):
        """``((path, registry row or None), ...)`` - what the directory holds, known or not.

        The files in the **default models directory** come first (sorted), then any registry
        row whose checkpoint lives somewhere else, so a model loaded from a USB stick is still
        offered under the name the registry knows it by.
        """
        known = {row.path: row for row in records}
        found = [(str(path), known.get(str(path))) for path in
                 models.checkpoints_in(kind, self.directory)]
        listed = {path for path, _row in found}
        found += [(path, row) for path, row in sorted(known.items())
                  if path not in listed]
        return found

    @staticmethod
    def _entry_text(path, row, kind) -> str:
        """What a checkpoint entry says: the file, what the registry knows, its state."""
        text = row.label if row is not None else Path(path).name
        if row is not None and row.is_active:
            text += "   (in force)"
        return text

    def _entry_tooltip(self, path, row, kind) -> str:
        """Everything an entry can hold: the network, the path, and what is known about it.

        A file the registry has not seen yet says so: it is offered because its **name**
        declares its kind (`models.KIND_PREFIXES`), and loading it is what registers it.
        """
        lines = [models.KIND_MODELS[kind], path]
        if row is None:
            lines.append("not in the registry yet - loading it registers it")
        else:
            if row.detail:
                lines.append(row.detail)
        if self.loaded_paths.get(kind) == path:
            lines.append("loaded in this window")
        return "\n".join(line for line in lines if line)

    # -- loading -----------------------------------------------------------
    def load(self, kind, path=None):
        """Load a checkpoint, register it as the active one, and announce it.

        ``path=None`` asks for the file; passing one (a default, a test) skips the dialog.
        Returns the registry record, or ``None`` when nothing was loaded.
        """
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self.window, f"Open the {models.KIND_LABELS[kind].lower()} checkpoint",
                str(paths.MODELS_DIR), CHECKPOINT_FILTER)
            if not path:
                return None
        loader, unavailable = loader_for(kind)
        try:
            model = loader(path)
        except unavailable as error:
            self.failed.emit(str(error))
            return None
        except Exception as error:                  # noqa: BLE001 - named, never silent
            self.failed.emit(f"{type(error).__name__}: {error}")
            return None

        with connection() as conn:
            record = models.register(
                conn, kind, path,
                **models.record_from_payload(kind, path, getattr(model, "payload", {})))
        self.loaded_paths[kind] = record.path    # this window is running *this* file now
        self.refresh()
        self.loaded.emit(kind, model)
        return record

    def use_active(self, kind):
        """Load the checkpoint the registry says is in force (no dialog)."""
        with connection() as conn:
            active = models.active(conn, kind)
        if active is None:
            label = models.KIND_LABELS[kind].lower()
            self.failed.emit(f"No active {label} is registered yet - open a {label} "
                             f"checkpoint, or pick a file from the menu's list.")
            return None
        return self.load(kind, active.path)

    def activate(self, kind, path):
        """Make a checkpoint the active one of its kind, and load it for this window.

        A checkpoint **found in the models directory** but not in the registry yet is loaded
        first - which registers it - so picking a file the lab has just trained is one click,
        not a "load it somewhere else first" errand (`models.checkpoints_in`).
        """
        with connection() as conn:
            known = models.record(conn, kind, path) is not None
        if known:
            with connection() as conn:
                models.activate_model(conn, kind, path)
        record = self.load(kind, path)
        self.refresh()                              # the check marks moved
        return record
