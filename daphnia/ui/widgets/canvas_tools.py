"""What is left *on* the image canvas, and the buttons that left it.

The measuring controls used to be two translucent, collapsible clusters floating over the
canvas - one at the top right (*View*: magnify, fit, pan, full screen), one at the lower
left (*Measure*: whatever the page acts with). They put the controls next to the specimen,
but they also put them *on* it; and the moment the window grew a **header**
(``daphnia/ui/widgets/header_bar.py``, ``X-N13``) there was a better place for them.

So the split is by *kind*, not by proximity:

* the **tools** are menu items the window header carries, with the view actions in its
  *View* menu - and a page's **draw tools** are *also* a floating bar on the image
  (:class:`CanvasDrawBar`, `X-N20`), in the lower-left corner, where the hand already is;
* the **facts** are ``label + value`` pairs in that same header, built from
  :class:`Kpi`: a page states *what a fact is* once, and the window reads it where it
  belongs (`HeaderFact`).

**Nothing else is drawn over the picture.** The numbers used to be a strip of cards pinned
to the canvas' top-left corner; that strip is gone (2026-10-18), because every fact it
carried is already in the panel the page reads and in the window header, and the specimen
is what the eyes are on. What is left over the photograph is the page's own draw tools and
nothing besides.

Nothing in this module knows about pages, windows or full screen. It is a canvas, a row of
buttons, and the geometry that keeps that row in its corner; the *bundle* a page actually
uses (its tools, its facts, full screen) is
:class:`~daphnia.ui.widgets.canvas_workspace.CanvasWorkspace`.

One detail that matters more than it looks: every tool button has **no focus policy**, so
clicking a tool never takes the keyboard away from the canvas - ``+``, ``-``, ``F`` and
the arrow keys keep working, which is what labelling at volume relies on.
"""

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QAction, QFont, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QWidget,
)

#: Distance between the draw bar and the canvas edge, in pixels. The corner is fixed
#: on purpose: a tool that moves is a tool nobody finds twice.
MARGIN = 12

#: Letter-spacing of a micro-label, in pixels. Qt's stylesheet has no
#: ``letter-spacing``, and the spacing is what makes a small capital a heading
#: instead of a whisper - so it is set on the font here, while the size, the weight
#: and the colour stay in the theme with every other typographic decision.
LABEL_LETTER_SPACING = 1.0

#: How loud a fact's flag is. A tone names a *state*; the colour reaches the header
#: fact from the theme's tokens through the stylesheet, never from a page.
TONES = ("", "warning", "danger")

#: The menu a page's tools land in when it names no group of its own.
DEFAULT_TOOL_GROUP = "Tools"


def micro_caps_label(text) -> QLabel:
    """A small capital heading: the one type decision a stylesheet cannot express.

    Qt's stylesheet subset has no ``letter-spacing``, and the spacing is what makes a
    small capital a heading instead of a whisper - so it is set on the font here, while
    the size, the weight and the colour stay in the theme with every other typographic
    decision. Used by the window header's facts (``header_bar.HeaderFact``), the one place
    a fact is read now.
    """
    label = QLabel(text)
    font = label.font()
    font.setLetterSpacing(QFont.AbsoluteSpacing, LABEL_LETTER_SPACING)
    label.setFont(font)
    return label


#: Narrowest a value may be squeezed before it elides to nothing at all. A value too
#: narrow to read is a value the operator would rather have in the tooltip.
VALUE_MIN_WIDTH = 60


class ElidedLabel(QLabel):
    """A one-line label that shortens its text with "..." instead of wrapping.

    ``QFontMetrics.elidedText`` is the only way to hold a long path, a sentence or a
    measurement to one line in Qt: a stylesheet has no ``text-overflow``, and
    ``setWordWrap(False)`` alone would let the widget grow and push its neighbours out of
    the window. The label can shrink below its text width (a small ``minimumSizeHint``)
    and what it no longer shows stays available: :meth:`full_text`, and as the tooltip.

    Used by every reader of a fact - the window header's ``label + value`` pairs - because
    each of them is one line high by design.
    """

    def __init__(self, text="", parent=None):
        super().__init__("", parent)
        self.setWordWrap(False)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._full = ""
        self.setText(text)

    def setText(self, text):                    # noqa: N802 - Qt's own name
        """Set the whole text; what is *drawn* is as much of it as there is room for."""
        self._full = str(text or "")
        self.setToolTip(self._full)
        self.updateGeometry()
        self.update()

    def full_text(self) -> str:
        """The whole text, whatever the width currently shows."""
        return self._full

    def sizeHint(self) -> QSize:                # noqa: N802 - Qt's own name
        """How wide the label *wants* to be: as wide as its words.

        ``QLabel`` would answer this from the text it was given - and it is given
        **nothing** (:meth:`setText` keeps the words here and paints them itself), so the
        answer used to be a sliver: every value was elided even with room to spare -
        *"the info box overlay is very small and needs to be as wide as image canvas"*.

        A label that can *shrink* (``minimumSizeHint``) is what keeps a narrow window
        working: the hint is where the words fit, not a demand.
        """
        hint = super().sizeHint()
        return QSize(self.fontMetrics().horizontalAdvance(self._full), hint.height())

    def minimumSizeHint(self) -> QSize:         # noqa: N802 - Qt's own name
        """A value may be squeezed to :data:`VALUE_MIN_WIDTH`, and elides past that."""
        hint = super().minimumSizeHint()
        return QSize(VALUE_MIN_WIDTH, hint.height())

    def paintEvent(self, event):
        """Draw the elided text in the palette's own colour.

        Painted rather than delegated to ``QLabel``: the base class wraps and clips
        instead of eliding, and a value cut off mid-word without an ellipsis reads as a
        shorter value rather than a shortened one.
        """
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        text = self.fontMetrics().elidedText(self._full, Qt.ElideRight, self.width())
        painter.drawText(self.rect(), int(self.alignment()), text)


@dataclass(frozen=True)
class Kpi:
    """One fact of a surface: a micro-heading and the value it names.

    A fact is one *thing*: a heading (``"Body length"``, shown as ``BODY LENGTH``), the
    value, and - when the value needs a caveat - a flag (``WEAK``, ``STALE``, ``UNSCALED``)
    with the tone that says how loud it is. A page states its facts as these records and
    the window header renders them; a value without a heading cannot be read there, which
    is why ``HeaderFact`` refuses one.

    ``key`` is what a fact is looked up by, and what keeps its words in the same place in
    the bar while they change.
    """

    key: str
    value: str
    label: str = ""
    flag: str = ""
    tone: str = ""

    def __post_init__(self):
        if self.tone not in TONES:
            raise ValueError(
                f"unknown tone {self.tone!r} for fact {self.key!r}; the tones are "
                f"{', '.join(repr(tone) for tone in TONES)} - a tone nothing selects "
                f"on would be styled by nothing at all")


@dataclass(frozen=True)
class ToolSpec:
    """One tool button: its key, its label, and whether it is a toggle.

    ``key`` is what the page looks the button up by (to hide it in a state that has no use
    for it, or to set a toggle's state).
    """

    key: str
    text: str
    tooltip: str = ""
    checkable: bool = False
    checked: bool = False


class ToolGroup(QObject):
    """The page's own actions, grouped for the window header's **menus**.

    A page says what a tool *is* (`ToolSpec`) and what it does (the callback); the header
    turns each group into a **menu**, so the bar carries menu items and nothing else - no
    buttons, no plates, nothing drawn over the photograph (`X-N13`).

    The actions exist from the moment the page adds them, before any window does, and they
    are real :class:`QAction` s: a page hides one for a state that cannot use it, checks a
    toggle, or renames one (the Dataset queue's *Crop*/*Frame*), and the menu it lands in
    follows - :attr:`changed` says when.
    """

    #: A group gained an action, or one was hidden, shown or renamed.
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups = {}        # title -> [QAction, ...]
        self._order = []         # the titles, in the order the page named them
        self._actions = {}       # key -> QAction
        self._title = DEFAULT_TOOL_GROUP

    # -- building ----------------------------------------------------------
    def set_group(self, title):
        """Name the menu the tools added from now on belong to.

        Called once per group, the way a page reads: *these are the measurements*, *these
        step through frames*.
        """
        title = str(title or DEFAULT_TOOL_GROUP)
        if title not in self._groups:
            self._groups[title] = []
            self._order.append(title)
        self._title = title

    def add_tool(self, spec: ToolSpec, callback=None) -> QAction:
        """Add one action to the current group, and return it.

        ``callback`` is optional so that a checkable tool can wire ``toggled`` itself,
        which is what a box-drawing tool needs.
        """
        if self._title not in self._groups:
            self.set_group(self._title)
        action = QAction(spec.text, self)
        action.setToolTip(spec.tooltip or spec.text)
        action.setCheckable(spec.checkable)
        action.setChecked(spec.checked)
        if callback is not None:
            action.triggered.connect(callback)
        self._groups[self._title].append(action)
        self._actions[spec.key] = action
        self.changed.emit()
        return action

    def groups(self) -> tuple:
        """``((title, (action, ...)), ...)`` - the menus a header should build, in order."""
        return tuple((title, tuple(self._groups[title])) for title in self._order
                     if self._groups[title])

    # -- the page's side ---------------------------------------------------
    def action(self, key: str):
        """The action added under ``key``, or ``None``."""
        return self._actions.get(key)

    @property
    def keys(self) -> tuple:
        """The tool keys, in the order they were added."""
        return tuple(self._actions)

    def set_checked(self, key: str, checked: bool):
        """Set a toggle's state without running what the toggle does."""
        action = self.action(key)
        if action is None:
            return
        action.blockSignals(True)
        action.setChecked(bool(checked))
        action.blockSignals(False)

    def set_visible(self, key: str, visible: bool):
        """Show or hide one action - a state can have no use for it.

        Hidden, never disabled: a dead menu item invites a choice that does nothing
        (``X-N10``). A group whose actions are all hidden leaves the menu bar, so the page
        never keeps a menu with nothing behind it.
        """
        action = self.action(key)
        if action is None:
            return
        action.setVisible(bool(visible))
        self.changed.emit()

#: The **draw tools** live on the image as a small floating bar (`X-N20`), pinned to the canvas'
#: lower-left corner. The buttons are the *same* actions the window header's menu carries - one
#: action, two affordances - so a page defines a tool once and offers it where the hand is
#: (on the picture) and where the menus are, and the two can never disagree about which is on.
class CanvasDrawBar(QWidget):
    """A page's **draw tools**, as buttons floating on the canvas."""

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("canvas_draw_bar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        row = QHBoxLayout(self)
        row.setContentsMargins(5, 4, 5, 4)
        row.setSpacing(2)
        self._buttons = []
        for action in actions:
            button = QToolButton(self)
            # One action, two affordances: text, tooltip, checkable and checked all follow the
            # action, so a key, this button and the menu item are the same tool.
            button.setDefaultAction(action)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            # Like every tool button, it takes no focus: clicking a tool must never take the
            # keyboard away from the canvas.
            button.setFocusPolicy(Qt.NoFocus)
            row.addWidget(button)
            self._buttons.append(button)
        self.adjustSize()

    @property
    def buttons(self) -> list:
        """The tool buttons on this bar, in the order the page gave them."""
        return list(self._buttons)


class CanvasTools(QObject):
    """What is left *on* one :class:`~daphnia.ui.widgets.image_canvas.ImageCanvas`.

    One thing: the page's **draw tools** (``draw_bar``, a :class:`CanvasDrawBar`), pinned
    to the lower-left corner, because that is the one control whose gesture happens *on*
    the picture (`X-N20`). What the page's other tools *are* lives in :attr:`actions` - a
    :class:`ToolGroup` the window header turns into menus (`HeaderBar.set_tools`) - and
    the facts a page publishes go to that header too, through
    :meth:`CanvasWorkspace.set_header`. **Nothing is drawn over the image but the draw
    bar.**

    The bar is a child of the canvas, so it stays visible in full screen, and it pins
    itself to the corner through an event filter - which keeps the canvas itself unaware
    of it.
    """

    def __init__(self, canvas, parent=None):
        super().__init__(parent if parent is not None else canvas)
        self.canvas = canvas

        # The page's own actions, grouped: their own object so a window can turn them
        # into menus once it exists - and a page can wire them before it does.
        self.actions = ToolGroup(self)

        # The **draw tools** (`X-N20`): a strip of buttons on the image, built when a page asks
        # for it with :meth:`set_draw_tools`. None until then.
        self.draw_bar = None

        canvas.installEventFilter(self)
        self._place()

    # -- tools -------------------------------------------------------------
    def set_group(self, title):
        """Name the header menu the tools added next belong to (see :class:`ToolGroup`)."""
        self.actions.set_group(title)

    def add_tool(self, spec: ToolSpec, callback=None) -> QAction:
        """Add one of the page's tools, and return the action itself."""
        return self.actions.add_tool(spec, callback)

    def action(self, key: str):
        """The action added under ``key``, or ``None``."""
        return self.actions.action(key)

    @property
    def keys(self) -> tuple:
        """Every tool key on this page, in the order they were added."""
        return self.actions.keys

    def set_checked(self, key: str, checked: bool):
        """Set a toggle's state without running what the toggle does."""
        self.actions.set_checked(key, checked)

    def set_visible(self, key: str, visible: bool):
        """Show or hide one action - a state can have no use for it.

        A group whose actions are all hidden leaves the menu bar, so a page whose tools do
        not apply to the current state (a live feed has no next frame) leaves the bar
        alone instead of keeping a menu with nothing behind it.
        """
        self.actions.set_visible(key, visible)

    # -- the draw tools ----------------------------------------------------
    def set_draw_tools(self, actions):
        """Put the page's **draw tools** on the image as a floating bar (`X-N20`).

        ``actions`` are the tool actions themselves, so a button here **is** the menu item - the
        same key, the same state, no second copy to keep in step. A canvas that only shows a
        picture has no draw tools and never builds the bar.
        """
        self.draw_bar = CanvasDrawBar(actions, self.canvas)
        self.draw_bar.show()
        self._place()
        return self.draw_bar

    # -- placement ---------------------------------------------------------
    def eventFilter(self, watched, event):
        """Re-pin the draw bar whenever the canvas is resized (or first shown)."""
        if watched is self.canvas and event.type() in (QEvent.Resize, QEvent.Show):
            self._place()
        return False

    def _place(self):
        """Put the draw bar where it belongs: the lower-left corner."""
        self._place_draw_bar()

    def _place_draw_bar(self):
        """Pin the **draw tools** to the lower-left corner, clear of the canvas' edge (`X-N20`)."""
        if self.draw_bar is None:
            return
        self.draw_bar.adjustSize()
        self.draw_bar.move(MARGIN, max(MARGIN,
                                       self.canvas.height() - self.draw_bar.height() - MARGIN))
