"""The hub window - the launcher that presents and opens the surfaces.

The study window (:class:`~daphnia.ui.main_window.MainWindow`) owns one guarded
run and nothing else. This window owns the *choice*: it renders
:mod:`daphnia.surfaces` and opens the window a surface needs - the study shell for
a surface that is built, or a plan for one that is specified but not yet built.

Launch the hub with ``python run.py``; launch a surface directly with
``python run.py study``, which never shows the hub at all.
"""

from functools import partial

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from daphnia import models, surfaces
from daphnia.ui.main_window import MainWindow
from daphnia.ui.model_menu import ModelMenu, hand_over
from daphnia.ui.pages.bench import Bench
from daphnia.ui.pages.calibration import Calibration
from daphnia.ui.pages.dataset import Dataset
from daphnia.ui.pages.home import Home
from daphnia.ui.pages.predict import Predict
from daphnia.ui.pages.study2 import RECORD_LEFT, Study2
from daphnia.ui.shell import APP_NAME, fit_to_screen
from daphnia.ui.widgets import GO_MENU, HeaderBar, StatusFooter

#: Preferred size of the hub window.
PREFERRED_SIZE = (980, 720)

#: Preferred size of a surface plan.
PLAN_SIZE = (780, 600)

#: Preferred size of the standalone calibration window.
CALIBRATION_SIZE = (1100, 720)

#: Preferred size of the standalone bench window.
BENCH_SIZE = (1180, 780)

#: Preferred size of the standalone dataset window.
DATASET_SIZE = (1180, 800)

#: Preferred size of the standalone predict window - the same shape as the dataset
#: window, because both are "a big picture plus a controls panel".
PREDICT_SIZE = (1240, 820)

#: Preferred size of the plate window. Wider than the others on purpose: a 96-well plate
#: is 12 columns across, and the wells have to stay round and readable side by side.
STUDY2_SIZE = (1320, 860)


class SurfaceWindow(QMainWindow):
    """A surface in its own window: the screen, plus the shared status footer.

    Two facts about the **machine** change while the operator works and both change
    what a number means - is the camera delivering frames, and which scale is in
    force. Every window carries them, and this is where that is decided once: the
    window is the screen above the footer, and ``Calibrate...`` in the footer hands
    over to the **Optics** surface (this window steps aside and comes back, so there
    is still only one window on screen - the same rule the hub follows when it opens
    a surface). Two screens already tell the operator to *"press Calibrate in the bar
    below"*; the bar is this one.

    A subclass declares three things: its :attr:`TITLE`, its preferred :attr:`SIZE`,
    and whether its screen has any use for :attr:`OFFERS_CALIBRATION`.
    """

    #: Window title suffix: ``DaphniaScope - <TITLE>``.
    TITLE = ""
    #: Preferred size, clamped to the screen by :func:`fit_to_screen`.
    SIZE = (1180, 780)
    #: Whether the footer offers ``Calibrate...``. The Optics window does not: it *is*
    #: the calibration, and its own panels are how one is taken.
    OFFERS_CALIBRATION = True
    #: Whether this window carries the *Detector (RTMDet)* / *Pose (RTMPose)* menus. A window
    #: whose screen cannot **use** a model leaves them out, because a menu that loads a
    #: checkpoint nothing acts on is a control with nothing behind it: the Bench displays its
    #: measurements and runs no model, so its window offers none (`SR-PREDICT-018`,
    #: `SR-BENCH-017`).
    OFFERS_MODELS = True
    #: This window's surface key. The *Go* menu offers every *other* surface, so a window
    #: never offers to open itself - the same rule the footer follows for Optics.
    KEY = ""

    def __init__(self, page, parent=None, title=None, key=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - {title or self.TITLE}")
        if key is not None:
            self.KEY = key

        self.page = page
        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(page, 1)

        # Named `status_footer` on purpose: that is the name the canvas workspace
        # already hides for full screen (`page_chrome`, `WINDOW_CHROME`), so a
        # full-screen look at the image leaves none of the chrome behind here either.
        self.status_footer = StatusFooter(show_calibrate=self.OFFERS_CALIBRATION)
        self.status_footer.calibrate_requested.connect(self.open_calibration)
        column.addWidget(self.status_footer)
        # The footer says which scale is in force; the Calibrate screen changes it (its
        # *Load...* and *Store this scale*), so the bar has to re-read it - otherwise
        # this window reports the old scale while the panel it hosts reports the new one.
        workflow_changed = getattr(page, "workflow_changed", None)
        if workflow_changed is not None:
            workflow_changed.connect(self.status_footer.refresh)

        # The header **is** the window's menu-bar area (`setMenuWidget`), so the window
        # has one top bar: the menus, and the facts that are not a measurement - which
        # frame, which model, the last message. Named `header_bar` for the same reason as
        # the footer: that is the name full screen hides (`WINDOW_CHROME`).
        self.header_bar = HeaderBar(title or self.TITLE)
        self.setMenuWidget(self.header_bar)
        self._build_menus()

        self.setCentralWidget(central)
        self._open = {}          # surface key -> the window this one handed over to
        self.follow_header(page)
        # *Detector (RTMDet)* / *Pose (RTMPose)*: which checkpoints are in force, loadable
        # from any window whose screen uses one (`X-N13`, `OFFERS_MODELS`).
        self.model_menu = None
        if self.OFFERS_MODELS:
            self.model_menu = ModelMenu(self.header_bar, self)
            self.model_menu.loaded.connect(self.adopt_model)
            self.model_menu.failed.connect(self.status_footer.show_note)
        self.header_bar.hide_empty_menus()
        fit_to_screen(self, self.SIZE)

    # -- models, from the header -------------------------------------------
    def adopt_model(self, kind, model):
        """Hand a freshly loaded model to the page on screen, or say what loaded it.

        The page's own answer decides: `hand_over` gives it the model when it can use one
        (the Predict surface does, and then shows it in its readout and the header), and
        returns the sentence to display when it cannot - a load nobody can see is a load
        nobody trusts.
        """
        note = hand_over(self.page, kind, model)
        if note:
            self.status_footer.show_note(note)
        return True

    # -- the header ---------------------------------------------------------
    def follow_header(self, page):
        """Show ``page``'s window facts in the header, and keep them following it.

        The page publishes them (`workspace.set_header`); this window renders them - so a
        screen states *what* a fact is and the window decides *where* it is read.
        """
        workspace = getattr(page, "workspace", None)
        if workspace is None:
            self.header_bar.set_facts([])
            self.header_bar.set_tools(None)
            return
        workspace.header_changed.connect(self.header_bar.set_facts)
        self.header_bar.set_facts(workspace.header)     # whatever it published already
        # The page's own tools, too: they used to float over the image (`X-N11`) and are
        # menus in this bar now (`X-N13`).
        self.header_bar.set_tools(workspace.tool_group)

    def _build_menus(self):
        """The two menus a surface window offers: where else to go, and the view."""
        self._build_go_menu()
        self._build_view_menu(self.page)

    def _build_go_menu(self):
        """*Go*: Home and every other surface, without a trip through the hub.

        The same step-aside the footer's *Calibrate...* uses
        (:meth:`open_surface_window`): this window hides, the chosen one opens, and this
        one comes back when that one closes - one window on screen, always. This window
        never offers itself (`KEY`).
        """
        entries = [(f"{surfaces.HUB_LABEL} - choose a surface",
                    partial(self.open_surface_window, surfaces.HUB_KEY))]
        entries += [(surface.label, partial(self.open_surface_window, surface.key))
                    for surface in surfaces.SURFACES
                    if surface.is_available and surface.key != self.KEY]
        self.header_bar.go_menu(entries)

    def _build_view_menu(self, page):
        """*View*, when this page has a canvas to act on (`X-N11`'s actions, in words)."""
        canvas = getattr(page, "canvas", None)
        workspace = getattr(page, "workspace", None)
        if canvas is None or workspace is None:
            return
        self.header_bar.view_menu(canvas, workspace)

    # -- handing over to another surface ------------------------------------
    def open_surface_window(self, key, checked=False):
        """Open another surface instead of this one, and come back when it closes.

        Generalised from the footer's *Calibrate...*, which is the same move for one
        specific surface. ``checked`` is Qt's own argument - a ``QAction`` passes its
        check state to the slot - and is not used.
        """
        existing = self._open.get(key)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return existing

        window = window_for(key)
        if window is None:
            return None
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(lambda _=None, name=key: self._surface_closed(name))
        self._open[key] = window

        self.hide()
        window.show()
        return window

    def _surface_closed(self, key):
        """The window we opened closed: re-read the bar and come back here."""
        self._open.pop(key, None)
        self.status_footer.refresh()
        self.show()

    @property
    def _calibration_window(self):
        """The Optics window, while this one has handed over to it.

        A property rather than a field of its own: *Calibrate...* in the footer and
        *Optics* in the *Go* menu are the same move, and two fields tracking one window is
        how they drift apart.
        """
        return self._open.get("optics")

    # -- calibration, from the bar ----------------------------------------
    def open_calibration(self):
        """Hand over to Optics; return here when the calibration window closes.

        Returns the calibration window, so a caller (or a test) can drive it.
        """
        return self.open_surface_window("optics")


class CalibrationWindow(SurfaceWindow):
    """Calibrate the optics on their own: no study, no step, and no phone needed.

    The calibration screen is shared with the study shell, but hosting it in its own
    window is what makes re-verifying possible **at any point** - after a bump,
    after a refocus, or after changing the objective - instead of only at step 3 of
    a workflow that may not even be running. It is the window the footer's
    *Calibrate...* opens from everywhere else, so it does not offer to open itself.
    """

    TITLE = "Calibrate"
    SIZE = CALIBRATION_SIZE
    OFFERS_CALIBRATION = False
    KEY = "optics"

    def __init__(self, parent=None):
        super().__init__(Calibration(), parent)


class BenchWindow(SurfaceWindow):
    """Measure without a study, on its own: no project, no plate, no phone.

    The screen is shared with the other surfaces; hosting it in its own window is
    what makes "a number right now" possible without walking a workflow - and what
    lets a result be attached to a study later instead of being thrown away.

    **No model menus.** The Bench displays measurements; a checkpoint is loaded and watched
    on the Predict surface (`SR-PREDICT-018`), so offering *Detector (RTMDet)* / *Pose
    (RTMPose)* here would be two menus whose only effect is on a registry this screen does not
    read.
    """

    TITLE = "Bench"
    SIZE = BENCH_SIZE
    KEY = "bench"
    OFFERS_MODELS = False

    def __init__(self, parent=None):
        super().__init__(Bench(), parent)


class DatasetWindow(SurfaceWindow):
    """Build the training corpus on its own: no study, no plate, no phone.

    The screen is shared with the other surfaces; hosting it in its own window is
    what makes "label frames at volume" possible without walking a workflow - and
    what keeps the corpus safe from the Study surface's auto-selection and pruning.
    """

    TITLE = "Dataset"
    SIZE = DATASET_SIZE
    KEY = "dataset"

    def __init__(self, parent=None):
        super().__init__(Dataset(), parent)


class PredictWindow(SurfaceWindow):
    """Watch the detector on its own: no study, no step, and no phone needed.

    The screen is shared with the other surfaces; hosting it in its own window is
    what makes "does the detector see the animals?" a question you can answer in
    five seconds - and what makes it possible to point it at a video file instead of
    a live camera.
    """

    TITLE = "Predict"
    SIZE = PREDICT_SIZE
    KEY = "predict"

    def __init__(self, parent=None):
        super().__init__(Predict(), parent)

    def closeEvent(self, event):
        """Stop the inference thread before the window (and the page) is destroyed.

        The window is opened with ``WA_DeleteOnClose``, so this is the last moment
        at which the worker can be joined - and a ``QThread`` destroyed while it is
        still running aborts the process.
        """
        self.page.stop()
        super().closeEvent(event)


class Study2Window(SurfaceWindow):
    """Capture a whole plate on its own: a study and a plate, no step order.

    The plate page is the acquisition half of the surface that will replace the governed
    Study pipeline, so it is hosted **beside** it - the same store, the same bursts, the
    same wells, and no gates to walk.
    """

    TITLE = "Study 2"
    SIZE = STUDY2_SIZE
    KEY = "study2"

    def __init__(self, parent=None):
        super().__init__(Study2(), parent)

    def closeEvent(self, event):
        """Store a running recording before the window (and the page) is destroyed.

        The window is opened with ``WA_DeleteOnClose``, so this is the last moment the page
        exists. A recording is written frame by frame *while it runs* and its rows are
        written when it stops: closing the window on one would leave a folder full of
        frames that no burst points at, forever. Stopping here stores what arrived, and the
        page says which ending that was (`SR-STUDY2-N06`).
        """
        self.page.stop_recording(reason=RECORD_LEFT)
        super().closeEvent(event)


#: The window each built surface opens. A surface *without* an entry has no window
#: yet, so its launch point shows its plan instead of pretending to work.
WINDOW_FACTORIES = {
    "study": MainWindow,
    "study2": Study2Window,
    "optics": CalibrationWindow,
    "bench": BenchWindow,
    "dataset": DatasetWindow,
    "predict": PredictWindow,
}


def window_for(key):
    """The window for a launch key: the hub, a surface shell, or its plan.

    Returns ``None`` for an unknown key, so a caller can report it.
    """
    if key == surfaces.HUB_KEY:
        return HubWindow()
    surface = surfaces.get(key)
    if surface is None:
        return None
    factory = WINDOW_FACTORIES.get(surface.key)
    return factory() if factory is not None else SurfacePlanWindow(surface)


class HubWindow(QMainWindow):
    """Launcher: the Home page, plus the surface window it opens."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - {surfaces.HUB_LABEL}")

        self.home = Home()
        self.home.surface_chosen.connect(self.open_surface)

        scroll = QScrollArea()
        scroll.setObjectName("body_scroll")      # the same look as the study shell
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.home)

        # The same status footer as every other window: the camera and the
        # calibration in force are facts about the machine, so both are visible
        # here - and both are actionable from here.
        self.status_footer = StatusFooter()
        self.status_footer.calibrate_requested.connect(self.calibrate)

        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(scroll, 1)
        column.addWidget(self.status_footer)
        self.setCentralWidget(central)

        # The same header every surface window carries: the menus - here *Go*, which is
        # the launcher's own job - and room for facts. The hub has none: it is not looking
        # at anything, so its bar is the name of the place and the way out of it.
        self.header_bar = HeaderBar(surfaces.HUB_LABEL)
        self.setMenuWidget(self.header_bar)
        menu = self.header_bar.menu(GO_MENU)
        for surface in surfaces.SURFACES:
            if surface.is_available:
                menu.addAction(surface.label,
                               partial(self.open_surface, surface.key))

        # The hub carries *Models* too: a checkpoint loaded here is registered as the
        # active one, so opening Predict afterwards already knows which model the lab
        # means (the hub is where a run is prepared, not only where it is started).
        self.model_menu = ModelMenu(self.header_bar, self)
        self.model_menu.loaded.connect(self._model_loaded)
        self.model_menu.failed.connect(self.status_footer.show_note)
        self.header_bar.hide_empty_menus()

        self._open = {}          # surface key -> the window currently showing it
        self._quitting = False
        fit_to_screen(self, PREFERRED_SIZE)

    def _model_loaded(self, kind, model):
        """A model was loaded from the hub: it is registered - and the bar says so.

        The hub looks at nothing, so there is no page to hand the model to: the note *is*
        the feedback, and the registry is what the next surface reads (`daphnia/models`).
        """
        self.status_footer.show_note(hand_over(None, kind, model))

    # -- calibration, from anywhere ----------------------------------------
    def calibrate(self):
        """Open the calibration screen on its own, leaving the hub behind."""
        return self.open_surface("optics")

    # -- opening a surface -------------------------------------------------
    def open_surface(self, key, checked=False):
        """Open a surface by key; returns its window, or ``None`` if unknown.

        The hub steps aside while a surface is open and comes back when that
        window is closed, so there is only ever one window on screen.
        ``checked`` is Qt's own argument (a ``QAction`` passes its check state); unused.
        """
        existing = self._open.get(key)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            self.hide()
            return existing

        window = window_for(key)
        if window is None:
            return None
        return self._show(key, window)

    def _show(self, key, window):
        """Show ``window`` instead of the hub, and come back when it closes."""
        # Closing the surface returns here instead of leaving nothing behind.
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(lambda _=None, name=key: self._surface_closed(name))
        self._open[key] = window

        self.hide()
        window.show()
        return window

    def _surface_closed(self, key):
        """A surface window closed: forget it and come back to the hub.

        The calibration and the project may both have changed while the surface was
        open, so both the footer and the tiles are re-read.
        """
        self._open.pop(key, None)
        if not self._quitting:
            self.status_footer.refresh()
            self.home.refresh()
            self.show()

    def closeEvent(self, event):
        """Remember that the hub is going away, so no surface resurrects it."""
        self._quitting = True
        super().closeEvent(event)


class SurfacePlanWindow(SurfaceWindow):
    """What a surface will do - for one that is specified but not built yet."""

    SIZE = PLAN_SIZE

    def __init__(self, surface, parent=None):
        # The plan is a page like any other, so the window it lives in is an ordinary
        # surface window: the same footer, the same camera and scale in the bar, the same
        # carrying header - and its *Go* menu names the surface.
        super().__init__(QWidget(), parent=parent, title=surface.label, key=surface.key)
        self.surface = surface

        column = QVBoxLayout(self.page)
        self.plan = QTextBrowser(self.page)
        self.plan.setObjectName("surface_plan")
        self.plan.setOpenExternalLinks(False)
        self.plan.setHtml(self._html())
        column.addWidget(self.plan)

        self.btn_back = QPushButton("Back to Home", self.page)
        self.btn_back.clicked.connect(self.close)
        column.addWidget(self.btn_back, alignment=Qt.AlignRight)

    def _html(self) -> str:
        """The surface's specification, in the words of its requirement doc."""
        surface = self.surface
        steps = "".join(f"<li>{item}</li>" for item in surface.next_steps)
        return f"""
        <h2>{surface.label}</h2>
        <p><b>{surface.tagline}</b></p>
        <p>{surface.summary}</p>
        <p><b>Status:</b> {surface.status_label}<br/>
           <b>Launch point:</b> <code>{surface.launch_point()}</code><br/>
           <b>Serves:</b> {surface.audience}<br/>
           <b>Needs first:</b> {surface.gate}</p>
        <h3>What to build next</h3>
        <ol>{steps}</ol>
        <p><b>Requirements:</b> <code>{surface.requirements}</code></p>
        <p><i>This surface is specified, not built yet - the Study surface works
        today. Nothing here is measured or stored.</i></p>
        """
