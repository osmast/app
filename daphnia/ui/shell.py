"""Application shell: a scrollable body plus the guarded step bar.

Every screen fills the body with two regions::

    +--------------------------------------------+--------------------+
    | visual area, edge to edge, no margins      | controls panel     |
    | (live stream top-left, captured frame,     | fixed width,       |
    | table, ...) expands to fill                | padded             |
    +--------------------------------------------+--------------------+
    | step bar: 1 Overview 2 Setup ...   hint        [Back] [Continue] |
    +------------------------------------------------------------------+

There is no free navigation: the step bar offers only Back/Continue and keeps
Continue disabled while the next step's requirements are unmet (see
:mod:`daphnia.workflow`).
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from daphnia.ui.loader import load_ui
from daphnia.ui.widgets.canvas_workspace import CanvasWorkspace
from daphnia.ui.widgets.canvas_tools import ToolGroup

APP_NAME = "DaphniaScope"

#: Width of the right-hand controls panel, shared by every screen that has one.
#:
#: Measured, not guessed. The panel scrolls vertically and its horizontal scroll bar
#: is **off**, so a child wider than the viewport is unreachable rather than merely
#: ugly. With the panel at this width the viewport is ~400 px, which fits the content
#: of every shared screen (the densest of them asks for 345 px) and lets the
#: explanatory sentences wrap in two or three lines instead of six. A screen whose
#: content cannot shrink any further overrides it: the **Dataset** panel needs 480
#: (see ``daphnia/ui/pages/dataset.py``), because its dataset combo sizes itself to
#: its longest item.
#:
#: A shell window's own floor is therefore ``CONTROLS_WIDTH`` plus the canvas'
#: ``PANEL_MINIMUM`` - Qt enforces it from the layout, so ``WINDOW_MINIMUM`` below
#: stays the floor for the small windows that have no panel at all.
CONTROLS_WIDTH = 420

#: Padding inside the controls panel (the visual area has none).
CONTROLS_PADDING = 10

#: Smallest a top-level window may be opened at, and how much of the screen it
#: may occupy.
WINDOW_MINIMUM = (760, 500)
WINDOW_SCREEN_FRACTION = 0.9


def fit_to_screen(window, preferred) -> tuple:
    """Resize a top-level window to ``preferred``, clamped to the screen.

    Shared by every top-level window (the surface windows and the hub), so they
    all open at a size the current display can actually show. Returns the size
    that was applied.
    """
    width, height = preferred
    min_width, min_height = WINDOW_MINIMUM
    screen = window.screen() or QGuiApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        width = min(width, max(min_width,
                               int(available.width() * WINDOW_SCREEN_FRACTION)))
        height = min(height, max(min_height,
                                 int(available.height() * WINDOW_SCREEN_FRACTION)))
    window.resize(width, height)
    return width, height



class StepBar(QWidget):
    """Guarded step navigation: where you are, what blocks you, Back/Continue."""

    def __init__(self, steps, parent=None):
        super().__init__(parent)
        self.setObjectName("step_bar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        self.step_labels = []
        for position, step in enumerate(steps, start=1):
            label = QLabel(f"{position} {step.label}")
            label.setObjectName("step_label")
            layout.addWidget(label)
            self.step_labels.append(label)

        self.hint_label = QLabel()
        self.hint_label.setObjectName("step_hint")
        layout.addWidget(self.hint_label, 1)

        self.btn_back = QPushButton("Back")
        layout.addWidget(self.btn_back)

        self.btn_continue = QPushButton("Continue")
        layout.addWidget(self.btn_continue)

    def update_state(self, index, done, reason):
        """Mark the current step and enable movement only when it is allowed."""
        for position, label in enumerate(self.step_labels):
            if position == index:
                state = "current"
            elif done[position]:
                state = "done"
            else:
                state = "todo"
            if label.property("state") != state:
                label.setProperty("state", state)
                label.style().unpolish(label)
                label.style().polish(label)

        last = len(self.step_labels) - 1
        self.btn_back.setEnabled(index > 0)
        self.btn_continue.setVisible(index < last)
        self.btn_continue.setEnabled(index < last and not reason)
        self.hint_label.setText(reason)


class ShellPage(QWidget):
    """Standard screen body: visual area on the left, controls panel on the right.

    The visual area carries no margins at all, so a live stream runs flush to the
    window edges. The static controls live in the screen's ``.ui`` file (so they
    stay editable in Qt Designer); the visual widgets -- live stream, image
    canvas, table -- are code-built widgets added with :meth:`add_visual`.

    Signals:
        workflow_changed: emitted after the screen produced something the
            workflow gates depend on (a study, a preset, a capture, a
            measurement), so the step bar can re-evaluate Continue.
    """

    workflow_changed = Signal()

    def __init__(self, ui_file=None, split=True, controls_width=CONTROLS_WIDTH,
                 parent=None):
        super().__init__(parent)

        self._menu_tools = None      # the page's own menu items (no canvas: see below)
        self.visual_layout = QVBoxLayout()
        self.visual_layout.setContentsMargins(0, 0, 0, 0)
        self.visual_layout.setSpacing(0)

        self.visual = QWidget()
        self.visual.setObjectName("visual_area")
        self.visual.setLayout(self.visual_layout)
        self.visual.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if split:
            row = QHBoxLayout(self)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            row.addWidget(self.visual, 1)
            if ui_file:
                self.controls = self._load_controls(ui_file, controls_width)
                row.addWidget(self.controls, 0)
        else:
            # Screens without a visual: the .ui content fills the whole body.
            column = QVBoxLayout(self)
            column.setContentsMargins(CONTROLS_PADDING, CONTROLS_PADDING,
                                      CONTROLS_PADDING, CONTROLS_PADDING)
            if ui_file:
                column.addWidget(self._adopt(load_ui(ui_file)))

    # -- composition -------------------------------------------------------
    def add_visual(self, widget, stretch=1):
        """Place a widget in the left-hand visual area."""
        self.visual_layout.addWidget(widget, stretch)
        return widget

    def canvas_workspace(self, canvas):
        """Give an image canvas the tools that belong *on* it.

        One line per screen: the view tools (magnify, fit, pan), full screen, and the
        readout line are the same everywhere, and the screen then adds its own
        measuring tools with :meth:`CanvasWorkspace.add_tool`. The chrome to hide in
        full screen comes from :func:`daphnia.ui.widgets.page_chrome`, which knows
        this shell's layout, so an entry point needs to know nothing about it.
        """
        return CanvasWorkspace(self, canvas)

    def menu_tools(self) -> ToolGroup:
        """The page's own **menu items**, for a step that has no canvas.

        A screen with nothing to measure on -- the project list, the results table --
        still owns actions: open a project, export a table. They belong in the window's
        menus (`X-N13`), and they are declared exactly like a canvas page's tools::

            tools = self.menu_tools()
            tools.set_group(FILE_MENU)
            tools.add_tool(ToolSpec("export", "Export measurements..."), self.export)

        The window reads this group when the page publishes no workspace
        (``daphnia.ui.main_window``), so a step without a canvas is in the menu bar too.
        """
        if self._menu_tools is None:
            self._menu_tools = ToolGroup(self)
        return self._menu_tools

    def notify_workflow(self):
        """Tell the shell that gating facts may have changed."""
        self.workflow_changed.emit()

    # -- helpers -----------------------------------------------------------
    def _load_controls(self, ui_file, width):
        """Wrap the screen's .ui content in the right-hand controls panel.

        The controls scroll *inside* the panel: a tall form then never stretches
        the whole page, so the visual area (the live stream) always fits the
        viewport and stays at the top-left without the body scrolling.
        """
        root = self._adopt(load_ui(ui_file))

        scroll = QScrollArea()
        scroll.setObjectName("controls_scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(root)

        panel = QWidget()
        panel.setObjectName("controls_panel")
        panel.setFixedWidth(width)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(CONTROLS_PADDING, CONTROLS_PADDING,
                                  CONTROLS_PADDING, CONTROLS_PADDING)
        layout.addWidget(scroll)
        return panel

    def _adopt(self, root):
        """Expose every named widget of a .ui file as an attribute of the page."""
        for child in root.findChildren(QWidget):
            name = child.objectName()
            if name:
                setattr(self, name, child)
        return root
