"""A canvas, its on-image tools, and the full-screen view of it.

Every canvas surface - Calibrate, Annotate, Bench, Dataset, Predict - wants the same
three things, and each used to build them by hand in its own ``.ui`` file: the **view
tools** (magnify, fit, pan) that belong next to the image they act on, the **measuring
actions** the operator presses while looking at the specimen, and a **full-screen**
view in which the image is the only thing on screen.

:class:`CanvasWorkspace` is that bundle, in one line for the page and invisible to
the entry point:

    self.canvas = self.add_visual(ImageCanvas(max_points=2, ...))
    self.workspace = self.canvas_workspace(self.canvas)
    self.workspace.add_tool(ToolSpec("save", "Save", "..."), self.save_label)

The buttons themselves float on the canvas
(:mod:`daphnia.ui.widgets.canvas_tools`); this module adds what a *page* needs on
top of them: the five view tools every canvas gets, the page's own tools, the facts
the window header reads (``set_header``), and full screen.
"""

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget

from daphnia.ui.widgets.canvas_tools import (
    CanvasTools,
    Kpi,
)

#: Attributes a surface window may carry that are **chrome**: they say *where you
#: are* (the header, the step bar, the scale footer), not *what the image is*, so the
#: full-screen view hides them. Named once here, because Study, Bench, Dataset, Predict
#: and Optics all use the same names for them.
WINDOW_CHROME = ("header_bar", "step_bar", "status_footer")


def page_chrome(page, canvas) -> list:
    """The widgets that must go away for ``canvas`` to be the only thing visible.

    The page's controls panel, its sibling visuals (the live stream) and the window
    chrome (:data:`WINDOW_CHROME`). Every surface keeps those names, so one rule
    covers all of them - and a widget that is not there is simply not in the list.
    """
    chrome = []
    controls = getattr(page, "controls", None)
    if isinstance(controls, QWidget):
        chrome.append(controls)
    layout = getattr(page, "visual_layout", None)
    if layout is not None:
        for index in range(layout.count()):
            widget = layout.itemAt(index).widget()
            if isinstance(widget, QWidget) and widget is not canvas:
                chrome.append(widget)
    window = canvas.window()
    for name in WINDOW_CHROME:
        widget = getattr(window, name, None)
        if isinstance(widget, QWidget):
            chrome.append(widget)
    return chrome


class CanvasFullScreen(QObject):
    """Make one canvas the only thing on screen - and put everything back.

    The window is **not** rebuilt: its chrome is hidden and the window itself goes
    full screen, so the canvas keeps its points, its overlays, its tools and its
    magnification. ``Esc`` leaves; so does the tool button, which is the only way
    back on a device without a keyboard.

    Signals:
        changed(bool): the canvas just became the only thing on screen (True) or
            stopped being it (False). The tool button follows it either way, so a
            view entered or left by ``Esc`` can never leave the button lying.
    """

    changed = Signal(bool)

    def __init__(self, canvas, chrome=(), window=None, parent=None):
        super().__init__(parent if parent is not None else canvas)
        self.canvas = canvas
        self._chrome = list(chrome)
        self._window = window
        self._active = False
        self._hidden = []
        self._geometry = None
        self._maximized = False

        # Esc, and only while the canvas is full screen: outside it, Esc keeps its
        # own meaning (abandon the box being drawn).
        self._escape = QShortcut(QKeySequence("Esc"), canvas)
        self._escape.setContext(Qt.WindowShortcut)
        self._escape.setEnabled(False)
        self._escape.activated.connect(self.leave)

    @property
    def active(self) -> bool:
        """True while the canvas is the only thing on screen."""
        return self._active

    def window(self):
        """The window the canvas lives in.

        Resolved when it is wanted, never when the page is built: a page exists
        before it is mounted in its window, and what has to go full screen (and lose
        its step bar) is the *window*, not the page.
        """
        return self._window if self._window is not None else self.canvas.window()

    def enter(self) -> bool:
        """Hide everything but the canvas and go full screen; True when it worked."""
        if self._active:
            return False
        window = self.window()
        if window is None:
            return False
        self._geometry = window.saveGeometry()
        self._maximized = window.isMaximized()
        # Only what is visible now is put back later: a control the page had hidden
        # on purpose must stay hidden after the canvas comes back.
        self._hidden = [widget for widget in self._chrome_widgets(window)
                        if widget.isVisible()]
        for widget in self._hidden:
            widget.hide()
        window.showFullScreen()
        self._active = True
        self._escape.setEnabled(True)
        self.canvas.setFocus()
        self.changed.emit(True)
        return True

    def leave(self) -> bool:
        """Put the chrome and the window geometry back; True when it did."""
        if not self._active:
            return False
        self._active = False
        self._escape.setEnabled(False)
        window = self.window()
        if window is not None:
            window.showNormal()
            if self._geometry is not None:
                window.restoreGeometry(self._geometry)
            if self._maximized:
                window.showMaximized()
        for widget in self._hidden:
            widget.show()
        self._hidden = []
        self.canvas.setFocus()
        self.changed.emit(False)
        return True

    def toggle(self) -> bool:
        """Leave when full screen, enter otherwise; the state now in force."""
        if self._active:
            self.leave()
        else:
            self.enter()
        return self._active

    def _chrome_widgets(self, window) -> list:
        """Everything to hide: the page's chrome plus the window's own."""
        extra = [getattr(window, name, None) for name in WINDOW_CHROME]
        return self._chrome + [widget for widget in extra
                               if isinstance(widget, QWidget)]


class CanvasWorkspace(QObject):
    """One canvas: the tools on it, the page's measuring actions, full screen.

    A page creates the workspace in one line and then adds its own tools:

        self.workspace = self.canvas_workspace(self.canvas)
        self.workspace.add_tool(ToolSpec("save", "Save", "..."), self.save)

    It also owns the pieces the pages used to wire by hand: the **hand tool** (``Pan``),
    the **full-screen** view, and the facts that are *not* a measurement - the source, the
    model, the last message - which go into the window's **header** (``set_header``),
    because they belong to the window rather than to the picture. Nothing is drawn over
    the image by the workspace; a page's own draw tools are the one exception, and they
    are its decision (``set_draw_tools``, `X-N20`).
    """

    #: A page just published the facts that belong to the **window header** - the bar
    #: sets them, and the page does not need to know the window exists.
    header_changed = Signal(list)

    #: The hand tool was switched on or off - by the *View* menu or from code. The menu's
    #: check mark follows it (the tool itself drags the view without dropping a point).
    pan_changed = Signal(bool)

    def __init__(self, page, canvas, parent=None):
        super().__init__(parent if parent is not None else canvas)
        self.page = page
        self.canvas = canvas
        self.tools = CanvasTools(canvas)
        self._header = []
        self._pan_on = False
        self._tool_shortcuts = {}
        self.full_screen = CanvasFullScreen(canvas, chrome=page_chrome(page, canvas),
                                            parent=self)
        canvas.zoom_changed.connect(self._on_zoom_changed)
        self._on_zoom_changed(canvas.zoom)

    # -- building ----------------------------------------------------------
    def set_tool_shortcuts(self, shortcuts, toggle=False):
        """Give a page's own tools a **key** each - Photoshop's rule for a canvas (`X-N18`).

        ``shortcuts`` maps a key to a tool key, e.g. ``{"V": "select", "L": "line"}``. Pressing
        the key **ticks that tool's action**, which is exactly what clicking it in the header
        menu does - so switching between the tools of a canvas never costs a trip to the menu,
        and a page can offer its tools the way an image editor does. A key whose action is
        missing (or is not a toggle) is skipped, and the shortcut lives on the **page**, so it
        works wherever the focus is.

        ``toggle`` is for a page whose tools can be turned **off**: the key of the tool already
        in force then turns it off, which is the same state as clicking its own button again.
        *No tool* is a state like any other on such a page (the *normal view*, where a press
        selects), so it has to be one key away too - a state reachable only through the menu
        bar is a state nobody returns to.
        """
        for key, tool in shortcuts.items():
            action = self.tools.action(tool)
            if action is None or not action.isCheckable():
                continue
            shortcut = QShortcut(QKeySequence(key), self.page)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(
                lambda action=action, toggle=toggle: self._activate_tool(action, toggle))
            self._tool_shortcuts[key] = shortcut
        return dict(self._tool_shortcuts)

    def tool_shortcuts(self) -> dict:
        """``{key: QShortcut}`` - the tool keys this canvas answers to (`X-N18`)."""
        return dict(self._tool_shortcuts)

    @staticmethod
    def _activate_tool(action, toggle=False):
        """Tick a tool's action - the page's own handler does the rest.

        With ``toggle``, a tool that is **already** in force is turned off instead, so the key
        reaches the page's *no tool armed* state as well as arming one.
        """
        if action.isChecked():
            if toggle:
                action.setChecked(False)
            return
        action.setChecked(True)

    def set_group(self, title):
        """Name the header menu the tools added next belong to (*Measure*, *Frames*...)."""
        self.tools.set_group(title)

    def add_tool(self, spec, callback=None):
        """Add one of the page's own tools to the window header's menus.

        ``callback`` is optional so that a checkable tool can wire its own signal
        (``action.toggled``) instead of the plain trigger.
        """
        return self.tools.add_tool(spec, callback)

    # -- the page's side ---------------------------------------------------
    def action(self, key):
        """The tool action added under ``key``, or ``None``."""
        return self.tools.action(key)

    def set_checked(self, key, checked):
        """Set a toggle's state without running what the toggle does."""
        self.tools.set_checked(key, checked)

    def set_visible(self, key, visible):
        """Show or hide one tool - a state can have no use for it."""
        self.tools.set_visible(key, visible)

    def set_draw_tools(self, actions):
        """Put the page's **draw tools** on the image as a floating bar (`X-N20`).

        The bar takes the tool *actions* themselves (``self.action(key)``), so a button on the
        picture and its menu item are one tool, and it sits at the canvas' lower-left corner.
        """
        return self.tools.set_draw_tools(actions)

    @property
    def draw_bar(self):
        """The **draw tools** floating on the image, if the page put them there (`X-N20`)."""
        return self.tools.draw_bar

    @property
    def tool_group(self):
        """The page's actions, grouped - what a window turns into header menus."""
        return self.tools.actions

    def set_header(self, kpis):
        """Publish the facts that belong to the **window header**, not to the picture.

        Which frame you are on, which model answered, the last message: facts that do not
        change with every click and do not have to be next to the specimen. The window
        that owns this page puts them in its header bar - so a page states *what a fact
        is* once and decides only *where it is read*.

        This is the **only** place a page publishes a fact: the strip of cards that used
        to sit on the image is gone (2026-10-18), so a number a page computes is either
        in its own panel or in this bar.

        The **magnification** is added by the workspace itself (see :meth:`header`): it is
        a view fact, and it used to be the *View* cluster's own status line.
        """
        self._header = list(kpis)
        self.header_changed.emit(self.header)

    @property
    def header(self) -> list:
        """The header facts: the page's, plus the view fact the tools used to show.

        The magnification only appears while the image *is* magnified: at *fit* - the
        common state - the bar stays for the facts that are worth a slot.
        """
        facts = list(self._header)
        zoom = self.canvas.zoom
        if zoom > self.canvas.MIN_ZOOM:
            facts.append(Kpi("zoom", f"{zoom:.2f}x", label="Zoom"))
        return facts

    def set_pan(self, on):
        """Turn the hand tool on or off - from the *View* menu, or from code.

        The menu's check mark follows through :attr:`pan_changed`, so a page that switches
        the hand tool on by itself cannot leave the menu lying.
        """
        on = bool(on)
        if on == self._pan_on:
            return
        self._pan_on = on
        self._on_pan_toggled(on)
        self.pan_changed.emit(on)

    # -- full screen -------------------------------------------------------
    @property
    def full_screen_active(self) -> bool:
        """True while the canvas is the only thing on screen."""
        return self.full_screen.active

    def enter_full_screen(self) -> bool:
        """Hide everything but the canvas and go full screen."""
        return self.full_screen.enter()

    def leave_full_screen(self) -> bool:
        """Bring the chrome back."""
        return self.full_screen.leave()

    def toggle_full_screen(self) -> bool:
        """Enter full screen or leave it; the state now in force.

        The menu item is set from the result, not from the click: a window that refuses
        to go full screen must not be left with a check mark.
        """
        self.full_screen.toggle()
        return self.full_screen.active

    # -- wiring ------------------------------------------------------------
    def _on_pan_toggled(self, checked):
        """The hand tool: left-drag moves the view and never drops a point."""
        self.canvas.pan_mode = bool(checked)
        self.canvas.setCursor(Qt.OpenHandCursor if checked else Qt.ArrowCursor)

    def _on_zoom_changed(self, zoom):
        """Republish the header: the magnification is one of its facts (`header`)."""
        self.header_changed.emit(self.header)
