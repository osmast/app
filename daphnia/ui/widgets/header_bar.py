"""The window header: the menus, and the facts that are not a measurement.

The canvas readout (:class:`~daphnia.ui.widgets.canvas_tools.KpiStack`) is for what
changes while your eyes are on the specimen - the number being produced, its scale, its
count, its caveat. Three other kinds of fact belong to the **window**, not to the
picture: *which* frame or file you are looking at, *which* model answered, and the
**last message**. They used to be cards over the image, competing with the number for
the same square centimetres of photograph - and a sentence lying on a specimen is read
by nobody.

They live here instead: a strip of ``label + value`` facts at the top of the window,
beside a real menu bar (a ``QMenuBar`` that is **not native**, so on macOS the bar is in
the window rather than in the system menu - a header that vanishes into the system menu
is a header nobody sees).

A header is **window chrome**: it sits on the window, not on the photograph, so its
colours are the scheme's tokens (exactly like the status footer) and full screen hides it
(``WINDOW_CHROME``). A long value is **elided**, never wrapped - a header that grows to
three lines is a panel that has moved to the top of the window - and the whole text stays
reachable in the value's tooltip, so a shortened path is still an honest one.

The facts are :class:`~daphnia.ui.widgets.canvas_tools.Kpi` records, the same ones the
cards on the canvas take: a page says *what a fact is* once, and decides where it is read.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMenuBar,
    QMessageBox,
    QWidget,
)

from daphnia.streaming import shared_service
from daphnia.ui.widgets.canvas_tools import (
    DEFAULT_TOOL_GROUP,
    ElidedLabel,
    Kpi,
    micro_caps_label,
)

#: Padding inside the header, and the gap between the menu bar, the title, the tools and
#: the facts.
HEADER_PADDING = (10, 2, 10, 2)
HEADER_SPACING = 12

#: The menus every window offers, named once and **in the order they are read**
#: (`docs/requirements/00-framework.md`, `X-N13`): what the window is for, then what it
#: opens, what it looks at, which model answers, how it is shown, where else to go, and
#: help. A page's own menu sits directly after *File*.
FILE_MENU = "File"
SOURCE_MENU = "Source"
#: One menu **per model kind**, titled with the network's own name (`daphnia.models`):
#: the detector is **RTMDet** (mmdetection lineage, `docs/rtmdet.md`) and the pose model is
#: **RTMPose** (mmpose lineage, `docs/rtmpose.md`). Two checkpoints, two kinds of number,
#: two reports - so two menus, not one list the operator has to sort out while reading it.
DETECTOR_MENU = "Detector (RTMDet)"
POSE_MENU = "Pose (RTMPose)"
MODEL_MENUS = {"detector": DETECTOR_MENU, "pose": POSE_MENU}
VIEW_MENU = "View"
GO_MENU = "Go"
HELP_MENU = "Help"

#: The app's menus, in that order. The page's own menus are inserted after *File*.
MENU_ORDER = (FILE_MENU, SOURCE_MENU, DETECTOR_MENU, POSE_MENU, VIEW_MENU, GO_MENU,
              HELP_MENU)


class HeaderFact(QWidget):
    """One fact in the header: a micro-caps heading and its value, side by side.

    The same :class:`Kpi` the cards on the canvas take, laid out for a bar rather than
    for a plate: heading, value, and - when the value needs one - a caveat flag in its
    tone's colour. It is read-only, takes no focus, and leaves the mouse alone.
    """

    def __init__(self, kpi: Kpi, parent=None):
        super().__init__(parent)
        if not kpi.label:
            raise ValueError(
                f"the header fact {kpi.key!r} has no heading. A bare value in a bar is a "
                f"number nobody can name: the canvas cards may drop their heading (where "
                f"position says the rest), a header fact may not. Label it.")
        self.kpi = kpi
        self.setObjectName("header_fact")
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.label = micro_caps_label(kpi.label.upper())
        self.label.setObjectName("header_fact_label")
        self.label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        row.addWidget(self.label, 0)

        self.value = ElidedLabel(str(kpi.value))
        self.value.setObjectName("header_fact_value")
        self.value.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        row.addWidget(self.value, 1)

        self.flag = None
        if kpi.flag:
            self.flag = micro_caps_label(kpi.flag.upper())
            self.flag.setObjectName("header_fact_flag")
            self.flag.setProperty("tone", kpi.tone or "note")
            self.flag.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            row.addWidget(self.flag, 0)

    def has_same_shape(self, kpi: Kpi) -> bool:
        """Whether ``kpi`` needs no new widget: only the caveat flag can appear."""
        return bool(kpi.flag) == (self.flag is not None)

    def update_kpi(self, kpi: Kpi):
        """Re-label this fact in place; the header follows a page every frame."""
        self.kpi = kpi
        self.label.setText(kpi.label.upper())
        self.value.setText(str(kpi.value))
        if self.flag is not None:
            self.flag.setText(kpi.flag.upper())


class HeaderBar(QWidget):
    """A window's top bar: the menus on the left, the facts that are not measurements.

    Mount it with ``QMainWindow.setMenuWidget`` and it *is* the window's menu-bar area -
    one bar, not a menu bar above a strip. A window builds its own menus through
    :meth:`menu`; the **facts** come from the page (`workspace.set_header`), because only
    the page knows which frame it is on and what just happened.

    The title says which window this is, in case two are open side by side; it is muted,
    because the facts are the part that changes.
    """

    def __init__(self, title="", parent=None, service=None):
        super().__init__(parent)
        self.setObjectName("header_bar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        # The same shared connection the footer drives: *Source* is where an operator
        # connects, and two buttons for one socket is how they disagree.
        self.service = service or shared_service()

        row = QHBoxLayout(self)
        row.setContentsMargins(*HEADER_PADDING)
        row.setSpacing(HEADER_SPACING)

        self.menu_bar = QMenuBar(self)
        self.menu_bar.setObjectName("header_menu")
        # In the window, not in the system menu: a header that macOS moves into the
        # global menu bar is a header that is not on screen at all.
        self.menu_bar.setNativeMenuBar(False)
        row.addWidget(self.menu_bar, 0)

        self.title = QLabel()
        self.title.setObjectName("header_title")
        row.addWidget(self.title, 0)

        # The page's own tool groups become **menus** (`set_tools`): the bar carries menu
        # items and nothing else - no buttons, no plates, nothing styled to look like a
        # control of its own.
        self._tools = None
        self._page_menus = []

        # The facts live in their own widget so the layout can give them all the room
        # the menus and the title leave, and elide whatever does not fit.
        self.facts_area = QWidget()
        self.facts_area.setObjectName("header_facts")
        self._facts_row = QHBoxLayout(self.facts_area)
        self._facts_row.setContentsMargins(0, 0, 0, 0)
        self._facts_row.setSpacing(HEADER_SPACING)
        row.addWidget(self.facts_area, 1)

        self._menus = {}
        self._facts = []
        self._fact_widgets = {}
        self._source_actions = []
        self._file_actions = []
        self.set_title(title)
        # The app's menus, in the order the application reads them. They are built here
        # (empty) so the order is the same in every window; the window fills *View*,
        # *Go* and *Models*, a page fills *File* and *Source*, and a menu that stays
        # empty is hidden (`hide_empty_menus`) rather than shown with nothing behind it.
        for name in MENU_ORDER:
            self.menu(name)
        self._source_menu = self._build_source_menu()
        self._file_menu = self._build_file_menu()
        self._build_help_menu()

    # -- File and Help -----------------------------------------------------
    def _build_file_menu(self) -> QMenu:
        """*File*: what the window opens, and how a run ends.

        The page's own *File* entries (`set_group(FILE_MENU)`: *Open a video...*, *Export
        ...*) land here, above the app's *Exit* - one place for "things from disk".
        """
        menu = self.menu(FILE_MENU)
        self._file_separator = menu.addSeparator()
        self._file_separator.setVisible(False)
        exit_action = menu.addAction("Exit")
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setMenuRole(QAction.MenuRole.QuitRole)
        exit_action.triggered.connect(self._quit)
        return menu

    def _build_help_menu(self) -> QMenu:
        """*Help*: the one page every window can explain, and where the data lives."""
        menu = self.menu(HELP_MENU)
        menu.addAction("How to calibrate (grid paper)", self._show_calibration_help)
        menu.addAction("Where the data lives", self._show_paths)
        return menu

    def _quit(self, _checked=False):
        """Leave the application - the whole window family, not just this one."""
        application = QApplication.instance()
        if application is not None:
            application.closeAllWindows()
            application.quit()

    def _show_calibration_help(self, _checked=False):
        """The calibration walkthrough (`daphnia/ui/help.py`), from any window."""
        from daphnia.ui.help import CalibrationHelpDialog

        CalibrationHelpDialog(self.window()).exec()

    def _show_paths(self, _checked=False):
        """Say where the database, the images and the checkpoints are."""
        from daphnia import paths

        QMessageBox.information(
            self.window(), "Where the data lives",
            f"Database:     {paths.DB_PATH}\n"
            f"Recordings:   {paths.RECORDINGS_DIR}\n"
            f"Datasets:     {paths.DATASETS_DIR}\n"
            f"Models:       {paths.MODELS_DIR}\n\n"
            f"Everything is a file on this machine: nothing leaves it.")

    def _fill_file_menu(self, actions):
        """Put the page's own file entries above the app's *Exit*."""
        for action in self._file_actions:
            self._file_menu.removeAction(action)
        self._file_actions = list(actions)
        for action in self._file_actions:
            self._file_menu.insertAction(self._file_separator, action)
        self._file_separator.setVisible(bool(self._file_actions))

    def hide_empty_menus(self):
        """Hide a menu with nothing behind it (a window with no *Go*, say).

        A menu that opens onto nothing is worse than no menu: it teaches the operator
        that the bar lies.
        """
        for name in MENU_ORDER:
            menu = self._menus.get(name)
            if menu is None:
                continue
            shown = [action for action in menu.actions()
                     if not action.isSeparator() and action.isVisible()]
            menu.menuAction().setVisible(bool(shown))

    # -- the Source menu ---------------------------------------------------
    def _build_source_menu(self) -> QMenu:
        """*Source*: connect the camera, and choose what the surface looks at.

        The **connection** entry belongs to the application - the one shared stream the
        whole app reads - so it is built here rather than by a page: its check mark *is*
        the camera state, and clicking it connects or disconnects from wherever the
        operator is standing. The page's own entries (`set_group(SOURCE_MENU)`: *Use the
        live frame*, *Open a video...*) follow under a separator.
        """
        menu = self.menu(SOURCE_MENU)
        self.connect_action = menu.addAction("Connect the live feed")
        self.connect_action.setCheckable(True)
        self.connect_action.triggered.connect(self._toggle_connection)
        self._separator = menu.addSeparator()
        self._separator.setVisible(False)       # until the page offers a source
        self.service.status.connect(self._follow_connection)
        self.service.opened.connect(self._follow_connection)
        self.service.closed.connect(self._follow_connection)
        self._follow_connection()
        return menu

    def _toggle_connection(self, _checked=False):
        """Connect or disconnect the shared stream - the footer's own move."""
        if self.service.state == self.service.DISCONNECTED:
            self.service.start()
        else:
            self.service.stop()
        self._follow_connection()

    def _follow_connection(self, *_args):
        """The check mark is the camera state, however it changed.

        The *detail* (connected but silent, and why) stays in the tooltip: a menu entry
        that grows a sentence is a sentence nobody reads, and the footer still spells the
        state out.
        """
        connected = self.service.state != self.service.DISCONNECTED
        self.connect_action.blockSignals(True)
        self.connect_action.setChecked(connected)
        self.connect_action.blockSignals(False)
        self.connect_action.setText(
            "Disconnect the live feed" if connected else "Connect the live feed")
        self.connect_action.setToolTip(
            f"{self.service.state_label()}\n"
            f"The phone publishes SRT; MediaMTX republishes the same path over RTSP, "
            f"which this reads.")

    def _fill_source_menu(self, actions):
        """Put the page's own source entries under the connection entry."""
        for action in self._source_actions:
            self._source_menu.removeAction(action)
        self._source_actions = list(actions)
        for action in self._source_actions:
            self._source_menu.addAction(action)
        self._separator.setVisible(bool(self._source_actions))

    # -- the menus ---------------------------------------------------------
    def menu(self, text) -> QMenu:
        """The menu under ``text``, created on first ask.

        Idempotent on purpose: a window (or a test) may ask for *View* twice and must not
        end up with two identical menus.
        """
        existing = self._menus.get(text)
        if existing is not None:
            return existing
        menu = self.menu_bar.addMenu(text)
        self._menus[text] = menu
        return menu

    def menus(self) -> tuple:
        """The menus on this bar, in the order they were added."""
        return tuple(self._menus)

    def go_menu(self, entries) -> QMenu:
        """The *Go* menu for ``(label, callback)`` pairs, laid out once.

        The callbacks belong to the **window** - it is the thing that can step aside -
        so the widget only decides what a *Go* menu looks like and the window decides
        what is in it. An empty ``entries`` leaves the menu as it is: a hub that has
        already offered the surfaces must not lose them on a second ask.
        """
        menu = self.menu(GO_MENU)
        if entries and not menu.actions():
            for label, callback in entries:
                menu.addAction(label, callback)
        return menu

    def view_menu(self, canvas, workspace, rebuild=False) -> QMenu:
        """The *View* menu for one canvas: the actions the tools carry (`X-N11`), in words.

        The entries are the same three the *View* cluster offers - so there is a route to
        them without aiming at a button over the picture - and **Full screen** is
        checkable and follows the view however it was entered or left, `Esc` included.
        ``rebuild`` is for the study shell, whose canvas is a different widget at every
        step: the actions must act on the canvas on screen.
        """
        menu = self.menu(VIEW_MENU)
        if rebuild and menu.actions():
            menu.clear()
        if canvas is None or workspace is None:
            # A step with no canvas offers no *View* at all: an action bound to the last
            # step's canvas zooms a widget nobody can see, and `hide_empty_menus` takes
            # the emptied menu out of the bar straight after.
            return menu
        if menu.actions():
            return menu
        menu.addAction("Zoom in", canvas.zoom_in)
        menu.addAction("Zoom out", canvas.zoom_out)
        menu.addAction("Fit the image", canvas.reset_view)
        pan = menu.addAction("Pan (hand tool)")
        pan.setCheckable(True)
        pan.triggered.connect(workspace.set_pan)
        workspace.pan_changed.connect(pan.setChecked)
        menu.addSeparator()
        full_screen = menu.addAction("Full screen")
        full_screen.setCheckable(True)
        full_screen.triggered.connect(workspace.toggle_full_screen)
        workspace.full_screen.changed.connect(full_screen.setChecked)
        return menu

    def show_menus(self, visible):
        """Show or hide the whole menu bar (a window with nothing to offer)."""
        self.menu_bar.setVisible(bool(visible))
        self.menu_bar.setMaximumWidth(16777215 if visible else 0)

    # -- the page's own menus ----------------------------------------------
    def set_tools(self, tools):
        """Turn the page's tool groups into **menus** - the bar carries menu items only.

        One menu per group the page named (*Measure*, *Frames*, *Label*...), rebuilt when
        the page hides, shows or renames an action (:attr:`ToolGroup.changed`) - so a group
        with nothing left to offer leaves the bar with it.
        """
        if self._tools is tools:
            return
        if self._tools is not None:
            self._tools.changed.disconnect(self._tools_changed)
        self._tools = tools
        if tools is not None:
            tools.changed.connect(self._tools_changed)
        self.render_tools()

    def render_tools(self):
        """Build the page's menus from its tool groups (idempotent, cheap)."""
        for menu in self._page_menus:
            self.menu_bar.removeAction(menu.menuAction())
            menu.deleteLater()
        self._page_menus = []
        # The app's *File* and *Source* menus are **shared** with the page on screen
        # (`_fill_file_menu`, `_fill_source_menu`), so they are emptied first: a step that
        # brings no source of its own must not inherit the last step's *Open a video...*.
        self._fill_source_menu([])
        self._fill_file_menu([])
        if self._tools is None:
            return
        for title, actions in self._tools.groups():
            shown = [action for action in actions if action.isVisible()]
            if title == SOURCE_MENU:
                # The page's source entries are not a menu of their own: they belong
                # under the app's connection entry.
                self._fill_source_menu(shown)
                continue
            if title == FILE_MENU:
                # ... and its file entries under the app's *Exit*.
                self._fill_file_menu(shown)
                continue
            if not shown:                       # nothing behind it: no menu
                continue
            menu = QMenu(title, self)
            for action in shown:
                menu.addAction(action)
            self._insert_menu(menu)
            self._page_menus.append(menu)

    def _insert_menu(self, menu):
        """Put a page menu **after File**, before the app's own menus: the work first."""
        existing = self.menu_bar.actions()
        offset = 1 + len(self._page_menus)      # File, then the page menus already placed
        if len(existing) > offset:
            self.menu_bar.insertMenu(existing[offset], menu)
        else:
            self.menu_bar.addMenu(menu)

    def _tools_changed(self):
        """The page changed its tools: rebuild - but never under an open menu."""
        if self.menu_bar.activeAction() is not None:
            # Pulling a menu out from under a click is worse than showing it one event
            # loop late, so the rebuild waits for the loop to be free.
            QTimer.singleShot(0, self.render_tools)
            return
        self.render_tools()

    @property
    def page_menus(self) -> tuple:
        """The titles of the menus built from the page's tools, in order."""
        return tuple(menu.title() for menu in self._page_menus)

    # -- the facts ---------------------------------------------------------
    def set_title(self, text):
        """Name the window this bar belongs to (muted; the facts are the loud part)."""
        text = str(text or "")
        self.title.setText(text)
        self.title.setVisible(bool(text))

    @property
    def facts(self) -> list:
        """The :class:`Kpi` facts the bar is showing, in the page's order."""
        return list(self._facts)

    def fact(self, key):
        """The fact widget under ``key``, or ``None``."""
        return self._fact_widgets.get(key)

    def set_facts(self, kpis):
        """Show exactly these facts, in this order (a blank value is dropped).

        Widgets are **reused** while the keys and their shape hold: the Predict surface
        re-publishes its facts on every frame it shows, and a header that rebuilds its
        widgets per frame is a header with a leak in it.
        """
        wanted = [kpi for kpi in kpis if str(kpi.value).strip()]
        keys = [kpi.key for kpi in wanted]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"two header facts share a key in {keys} - a fact is looked up by its key")
        if keys == list(self._fact_widgets) and all(
                self._fact_widgets[kpi.key].has_same_shape(kpi) for kpi in wanted):
            for kpi in wanted:
                self._fact_widgets[kpi.key].update_kpi(kpi)
        else:
            self._rebuild(wanted)
        self._facts = wanted

    def _rebuild(self, kpis):
        """Lay the facts out afresh - a new key, a lost one, or a new order."""
        while self._facts_row.count():
            item = self._facts_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._fact_widgets = {}
        for kpi in kpis:
            fact = HeaderFact(kpi, self.facts_area)
            self._fact_widgets[kpi.key] = fact
            self._facts_row.addWidget(fact, 1)

