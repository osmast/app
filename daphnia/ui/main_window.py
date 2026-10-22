"""Main window - the step-based application shell.

There is no free navigation between tasks. :mod:`daphnia.workflow` defines the
order and the gates, and this window only offers Back/Continue: a step whose
requirements are unmet cannot be entered, so work cannot happen out of order -
no capturing without a study (which owns the image folder) and without an active
optical preset (without a scale an image can never be measured).
"""

from PySide6.QtWidgets import QFrame, QMainWindow, QScrollArea, QVBoxLayout

from daphnia import workflow
from daphnia.db import connection
from daphnia.streaming import shared_service
from daphnia.ui.loader import load_ui
from daphnia.ui.pages import (
    Acquire,
    Annotate,
    Calibration,
    Dashboard,
    Select,
    SetupExperiment,
    Visualize,
)
from daphnia.ui.model_menu import ModelMenu, hand_over
from daphnia.ui.shell import APP_NAME, ShellPage, StepBar, fit_to_screen
from daphnia.ui.widgets import HeaderBar, StatusFooter

#: Screen class per workflow route. Order and labels come from daphnia.workflow.
PAGE_CLASSES = {
    "dashboard": Dashboard,
    "experiment": SetupExperiment,
    "calibration": Calibration,
    "acquire": Acquire,
    "select": Select,
    "annotate": Annotate,
    "visualize": Visualize,
}

#: Screen shown on startup.
HOME_ROUTE = workflow.STEPS[0].route

#: Preferred startup size; clamped to the screen's available area.
PREFERRED_SIZE = (1180, 760)


class MainWindow(QMainWindow):
    """Shell window: one scrollable body plus the guarded step bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        load_ui('pages.ui', self)

        # The header **is** this window's menu-bar area (`setMenuWidget`): the menus, and
        # the facts that are not a measurement - which frame, the last message. There is
        # deliberately **no** *Go* menu here: free navigation between surfaces is the hub's
        # job, and the whole point of this window is the order (`daphnia/workflow.py`).
        # Named `header_bar`, so full screen hides it (`WINDOW_CHROME`).
        self.header_bar = HeaderBar("Study")
        self.setMenuWidget(self.header_bar)
        # *Models* works here too: a model loaded while labelling a corpus is registered
        # as active, and the Predict surface picks it up from the registry.
        self.model_menu = ModelMenu(self.header_bar, self)
        self.model_menu.loaded.connect(self._model_loaded)
        self.model_menu.failed.connect(self.status_footer_note)

        self.routes = {}        # route -> step index
        self.pages = {}         # route -> screen widget
        self._by_index = {}     # step index -> screen widget
        self._header_source = None      # the workspace of the page on screen

        for index, step in enumerate(workflow.STEPS):
            page = PAGE_CLASSES[step.route]()
            self.routes[step.route] = index
            self.pages[step.route] = page
            self._by_index[index] = page
            self.stackedWidget.addWidget(page)
            if isinstance(page, ShellPage):
                page.workflow_changed.connect(self.refresh_workflow)
            workspace = getattr(page, "workspace", None)
            if workspace is not None:
                workspace.header_changed.connect(self._on_header)

        self.step_bar = StepBar(workflow.STEPS)
        self.step_bar.btn_back.clicked.connect(self.go_back)
        self.step_bar.btn_continue.clicked.connect(self.go_continue)

        self._mount_shell()
        self._fit_to_screen()

        # Always start at Home; nothing is pre-selected, the operator walks the run.
        self.go_to_step(0)

    # -- navigation (guarded) ----------------------------------------------
    def go_to_step(self, index: int) -> bool:
        """Enter a step; refuses when its requirements are unmet."""
        index = max(0, min(index, len(workflow.STEPS) - 1))
        runtime = self._runtime_facts()
        with connection() as conn:
            allowed = workflow.can_enter(conn, index, runtime)
        if not allowed:
            self.refresh_workflow()
            return False

        self._notify(self.current_page(), "on_hidden")
        self.stackedWidget.setCurrentIndex(index)
        self._notify(self.current_page(), "on_shown")
        # The header carries the page on screen: its facts, its tools, its canvas.
        self._follow_header(self.current_page())
        self._mount_view_menu(self.current_page())
        self.header_bar.hide_empty_menus()      # a menu with nothing behind it: hidden
        # Enter a step at the top, so its top-left visual (the live stream) is
        # never left scrolled out of view by the previous screen.
        self.body_scroll.verticalScrollBar().setValue(0)
        self.refresh_workflow()
        return True

    def go_to(self, route: str) -> bool:
        """Route-name navigation, still subject to the gates."""
        return self.go_to_step(self.routes[route])

    def go_continue(self):
        """Advance one step (allowed only when the gates permit it)."""
        self.go_to_step(self.stackedWidget.currentIndex() + 1)

    def go_back(self):
        """Return to the previous step."""
        self.go_to_step(self.stackedWidget.currentIndex() - 1)

    def current_page(self):
        """The screen widget currently in the body."""
        return self._by_index.get(self.stackedWidget.currentIndex())

    # -- workflow state ----------------------------------------------------
    def refresh_workflow(self):
        """Re-evaluate the gates and update the step bar."""
        current = self.stackedWidget.currentIndex()
        last = len(workflow.STEPS) - 1
        runtime = self._runtime_facts()
        with connection() as conn:
            facts = workflow.facts(conn, runtime)
            reason = (workflow.blocking_reason(conn, current + 1, runtime)
                      if current < last else "")
        done = [bool(step.produces) and all(facts.get(fact) for fact in step.produces)
                for step in workflow.STEPS]
        self.step_bar.update_state(current, done, reason)
        # The shared footer may say something different after every step (a
        # calibration taken, the camera connected or dropped), so it is re-read with
        # the gates.
        self.status_footer.refresh()
        self.setWindowTitle(f"{APP_NAME} - {workflow.STEPS[current].label}")

    # -- shell -------------------------------------------------------------
    def status_footer_note(self, message):
        """Report something in the window's own bar (the footer is mounted after this)."""
        footer = getattr(self, "status_footer", None)
        if footer is not None:
            footer.show_note(message)

    def _model_loaded(self, kind, model):
        """A model was loaded while a study is open: the page takes it, or the bar says so.

        **This connection was missing**, so a load in the study shell did nothing at all on
        screen: the registry was updated and the page never heard about it. The study has no
        Predict step today, so the answer is normally the note - but a step that can use a
        model must still get it, and the load must never be silent.
        """
        note = hand_over(self.current_page(), kind, model)
        if note:
            self.status_footer_note(note)

    def _follow_header(self, page):
        """Show the page on screen in the header - its facts and its tools, and only its.

        Every screen publishes its own window facts (`workspace.set_header`) and builds its
        own toolbar (`workspace.add_tool`); a step that is not on screen must never write
        the bar, which is what :meth:`_on_header` checks. Switching steps therefore
        re-seeds rather than waits.
        """
        self._header_source = getattr(page, "workspace", None)
        if self._header_source is not None:
            self.header_bar.set_facts(self._header_source.header)
            self.header_bar.set_tools(self._header_source.tool_group)
            return
        # A step with no canvas keeps no facts, but it may still own menu items (File:
        # open a project, export a table) - `ShellPage.menu_tools`.
        tools = page.menu_tools() if hasattr(page, "menu_tools") else None
        self.header_bar.set_facts([])
        self.header_bar.set_tools(tools)

    def _on_header(self, kpis):
        """A page published its window facts: show them if it is the page on screen."""
        if self.sender() is self._header_source:
            self.header_bar.set_facts(kpis)

    def _mount_view_menu(self, page):
        """The *View* menu, acting on the canvas of the page on screen.

        One canvas per step, so the menu is **rebuilt** rather than accumulated: an action
        still bound to the previous step's canvas would zoom a widget nobody can see.
        """
        canvas = getattr(page, "canvas", None)
        workspace = getattr(page, "workspace", None)
        if canvas is not None and workspace is not None:
            self.header_bar.view_menu(canvas, workspace, rebuild=True)
            return
        # Nothing to view here: empty the menu rather than leave the last step's actions
        # in it (they are bound to a canvas that is no longer on screen), and let
        # `hide_empty_menus` take it out of the bar.
        self.header_bar.view_menu(None, None, rebuild=True)

    def _mount_shell(self):
        """Mount the shared scale footer and the step bar, then the screens."""
        footer = self.footer_container.layout()
        if footer is None:
            footer = QVBoxLayout(self.footer_container)
            footer.setContentsMargins(0, 0, 0, 0)
        # The same footer the hub and every other surface shows: the camera state and
        # the calibration in force, each actionable from wherever the operator is.
        self.status_footer = StatusFooter()
        self.status_footer.calibrate_requested.connect(self._open_calibration)
        footer.addWidget(self.status_footer)
        footer.addWidget(self.step_bar)
        self._mount_body_scroll()

    def _open_calibration(self):
        """Go back to the Calibrate step - allowed at any point in the run.

        Calibrating is a backwards move in the workflow, so it is always permitted
        while the optics are in use; if it is not permitted yet (no study, no
        connection), the step bar already says why.
        """
        index = workflow.index_of("calibration")
        if not self.go_to_step(index):
            with connection() as conn:
                reason = workflow.blocking_reason(conn, index,
                                                  self._runtime_facts())
            self.status_footer.show_note(reason or "Calibration is not reachable yet.")

    def _mount_body_scroll(self):
        """Wrap the stacked screens so the whole body scrolls as one region."""
        layout = self.centralwidget.layout()
        if layout is None:
            return

        position = layout.indexOf(self.stackedWidget)
        layout.removeWidget(self.stackedWidget)

        self.body_scroll = QScrollArea()
        self.body_scroll.setObjectName("body_scroll")
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QFrame.NoFrame)
        self.body_scroll.setWidget(self.stackedWidget)

        layout.insertWidget(max(position, 0), self.body_scroll)
        layout.setStretchFactor(self.body_scroll, 1)

    def _fit_to_screen(self):
        """Open at a size that fits the current screen."""
        fit_to_screen(self, PREFERRED_SIZE)

    @staticmethod
    def _runtime_facts() -> dict:
        """Facts that live outside the database: today only the phone connection."""
        return {"connection": shared_service().connected}

    @staticmethod
    def _notify(page, hook):
        callback = getattr(page, hook, None)
        if callable(callback):
            callback()
