"""What lives *on* the image, and the full-screen view of it.

``canvas_tools`` and ``canvas_workspace`` are shared by every surface - Calibrate,
Annotate, Bench, Dataset and Predict - so they are tested here once, in the abstract: the
page's own **tools** (menu items the window header builds), the one thing still drawn on
the image (a page's **draw bar**, `X-N20`), the facts that go to the window **header**, and
the window chrome that full screen hides. What each screen *does* with them is
that screen's own test.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QAction, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from daphnia.ui import theme
from daphnia.ui.widgets import (
    CanvasWorkspace,
    HeaderBar,
    ImageCanvas,
    Kpi,
    ToolSpec,
)
from daphnia.ui.widgets.canvas_tools import MARGIN
from daphnia.ui.widgets.canvas_workspace import page_chrome

#: A white image: a pixel that is still pure white proves nothing was drawn there.
WHITE_IMAGE = np.full((240, 320, 3), 255, dtype=np.uint8)


class _FakePage(QWidget):
    """A page shaped like ``ShellPage``: a visual area plus a controls panel.

    Built by hand, so these tests need neither a ``.ui`` file nor a database - and the
    shape they exercise is the shape every surface has.
    """

    def __init__(self):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        self.visual_layout = QVBoxLayout()
        self.visual = QWidget()
        self.visual.setLayout(self.visual_layout)
        row.addWidget(self.visual, 1)

        self.controls = QWidget()
        row.addWidget(self.controls, 0)


class CanvasTestCase(unittest.TestCase):
    """Common setup: one offscreen application and one page shaped like a screen."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _page(self, size=(600, 400)):
        page = _FakePage()
        page.resize(*size)
        page.canvas = ImageCanvas()
        # The size the canvas will have once it is laid out. A widget that is never
        # shown keeps this geometry, and Qt sends no resize event until it is shown,
        # so a test can place the strip without an event loop.
        page.canvas.resize(*size)
        page.visual_layout.addWidget(page.canvas)
        # A sibling visual - the live stream on Bench, Dataset and Calibrate - which
        # full screen has to hide as well.
        page.stream = QWidget()
        page.visual_layout.addWidget(page.stream)
        self.addCleanup(page.deleteLater)
        return page

    def _workspace(self, size=(600, 400)):
        page = self._page(size)
        return page, CanvasWorkspace(page, page.canvas)

    def _press(self, widget, point):
        """Send one left press to ``widget`` itself (not to a child under it)."""
        event = QMouseEvent(QEvent.MouseButtonPress, QPointF(point), QPointF(point),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(widget, event)


class ToolGroupTests(CanvasTestCase):
    """The page's own tools: menu items a window builds menus from, nothing on the image."""

    def _menu(self, bar, title):
        """The menu the bar is **showing** under that title (a page menu is not in `_menus`)."""
        for action in bar.menu_bar.actions():
            if action.text() == title and action.menu() is not None:
                return action.menu()
        self.fail(f"the bar shows no menu named {title!r}")

    def test_the_page_names_its_groups_and_the_header_builds_them(self):
        page, workspace = self._workspace()
        workspace.set_group("Measure")
        workspace.add_tool(ToolSpec("save", "Save"), lambda: None)
        workspace.set_group("Frames")
        workspace.add_tool(ToolSpec("next", "Next >"), lambda: None)

        self.assertEqual([title for title, _actions in workspace.tool_group.groups()],
                         ["Measure", "Frames"])
        self.assertEqual(workspace.tools.keys, ("save", "next"))

        bar = HeaderBar("Test")
        bar.set_tools(workspace.tool_group)
        self.assertEqual(bar.page_menus, ("Measure", "Frames"))
        # The page's menus sit **after File** (`X-N13`: the bar reads File | the page's work |
        # Source | Detector | Pose | View | Go | Help), so they are found by name.
        menu = self._menu(bar, "Measure")
        self.assertEqual([action.text() for action in menu.actions()], ["Save"])

    def test_the_page_tools_are_menu_items_not_widgets_on_the_image(self):
        """`X-N13`: the bar carries menu items - no buttons, nothing over the picture."""
        page, workspace = self._workspace()
        workspace.add_tool(ToolSpec("save", "Save"), lambda: None)

        self.assertIsInstance(workspace.action("save"), QAction)
        self.assertEqual(page.canvas.findChildren(QToolButton), [],
                         "a tool button is drawn on the image")

    def test_a_tool_runs_its_callback(self):
        page, workspace = self._workspace()
        seen = []
        workspace.add_tool(ToolSpec("save", "Save"), lambda: seen.append("save"))
        workspace.action("save").trigger()
        self.assertEqual(seen, ["save"])

    def test_setting_a_toggle_does_not_run_it(self):
        """A page sets a toggle from code (a model that requires boxes) - and must not
        thereby *do* what the toggle does."""
        page, workspace = self._workspace()
        seen = []
        workspace.add_tool(ToolSpec("box", "Box", checkable=True),
                           lambda: seen.append("box"))
        workspace.set_checked("box", True)
        self.assertTrue(workspace.action("box").isChecked())
        self.assertEqual(seen, [])

    def test_a_tool_can_be_hidden_for_a_state_that_cannot_use_it(self):
        """Hidden, never disabled (`X-N10`) - and the menu goes with its last item."""
        page, workspace = self._workspace()
        workspace.set_group("Frames")
        workspace.add_tool(ToolSpec("next_frame", "Frame >"), lambda: None)
        bar = HeaderBar("Test")
        bar.set_tools(workspace.tool_group)
        self.assertEqual(bar.page_menus, ("Frames",))

        workspace.set_visible("next_frame", False)
        self.assertFalse(workspace.action("next_frame").isVisible())
        self.assertEqual(bar.page_menus, (), "a menu with nothing behind it stayed")

        workspace.set_visible("next_frame", True)
        self.assertTrue(workspace.action("next_frame").isVisible())
        self.assertEqual(bar.page_menus, ("Frames",))

    def test_a_page_can_rename_a_tool_and_the_menu_follows(self):
        """The Dataset queue's *Crop*/*Frame* switch is one action with two names."""
        page, workspace = self._workspace()
        workspace.set_group("Label")
        view = workspace.add_tool(ToolSpec("view", "Crop"), lambda: None)
        bar = HeaderBar("Test")
        bar.set_tools(workspace.tool_group)

        view.setText("Frame")
        self.app.processEvents()
        menu = self._menu(bar, "Label")
        self.assertEqual([action.text() for action in menu.actions()], ["Frame"])


class PanTests(CanvasTestCase):
    """The hand tool, and the magnification: the two view facts the tools used to show."""

    def test_the_hand_tool_is_switched_on_and_off(self):
        page, workspace = self._workspace()
        workspace.set_pan(True)
        self.assertTrue(page.canvas.pan_mode)
        self.assertEqual(page.canvas.cursor().shape(), Qt.OpenHandCursor)

        workspace.set_pan(False)
        self.assertFalse(page.canvas.pan_mode)
        self.assertEqual(page.canvas.cursor().shape(), Qt.ArrowCursor)

    def test_whoever_switches_the_hand_tool_says_so(self):
        """The *View* menu's check mark follows it, however it was switched."""
        page, workspace = self._workspace()
        seen = []
        workspace.pan_changed.connect(seen.append)

        workspace.set_pan(True)
        workspace.set_pan(True)                  # already on: no second report
        workspace.set_pan(False)
        self.assertEqual(seen, [True, False])

    def test_the_magnification_is_a_header_fact_only_while_magnified(self):
        page, workspace = self._workspace()
        page.canvas.set_source(WHITE_IMAGE)
        workspace.set_header([Kpi("source", "File: a.png", label="Source")])
        self.assertEqual([kpi.key for kpi in workspace.header], ["source"])

        page.canvas.zoom_in()
        self.assertEqual([kpi.key for kpi in workspace.header], ["source", "zoom"])
        zoom = [kpi for kpi in workspace.header if kpi.key == "zoom"][0]
        self.assertEqual(zoom.value, f"{ImageCanvas.ZOOM_STEP:.2f}x")

        page.canvas.reset_view()
        self.assertEqual([kpi.key for kpi in workspace.header], ["source"])


class KpiTests(CanvasTestCase):
    """A fact of a surface: the record the window header renders."""

    def test_a_flag_may_only_wear_a_tone_something_selects_on(self):
        """A tone the theme has no rule for is a colour that means nothing - refused."""
        with self.assertRaises(ValueError) as raised:
            Kpi("x", "1", flag="?", tone="loud")
        self.assertIn("loud", str(raised.exception))
        for tone in ("", "warning", "danger"):
            self.assertEqual(Kpi("x", "1", tone=tone).tone, tone)


class ChromeTests(CanvasTestCase):
    """Which widgets a full-screen canvas hides: everything that is not the image."""

    def test_the_chrome_is_the_panel_and_the_other_visuals(self):
        page, workspace = self._workspace()
        chrome = page_chrome(page, page.canvas)
        self.assertIn(page.controls, chrome)
        self.assertIn(page.stream, chrome)
        self.assertNotIn(page.canvas, chrome)

    def test_a_window_without_those_names_simply_has_less_chrome(self):
        """A surface window that keeps no step bar must not break the rule."""
        page, workspace = self._workspace()
        self.assertEqual(page_chrome(page, page.canvas), [page.controls, page.stream])

    def test_the_header_the_step_bar_and_the_footer_of_a_window_are_chrome_too(self):
        page, workspace = self._workspace()
        page.header_bar = QWidget()
        page.step_bar = QWidget()
        page.status_footer = QWidget()
        chrome = page_chrome(page, page.canvas)
        for widget in (page.header_bar, page.step_bar, page.status_footer):
            self.assertIn(widget, chrome)


class FullScreenTests(CanvasTestCase):
    """Full screen: the canvas alone - and every piece of chrome back afterwards."""

    def _shown(self):
        page, workspace = self._workspace()
        page.show()
        self.addCleanup(page.hide)
        return page, workspace

    def test_full_screen_hides_the_chrome_and_puts_it_back(self):
        page, workspace = self._shown()
        self.assertTrue(page.controls.isVisible())
        self.assertTrue(page.stream.isVisible())

        self.assertTrue(workspace.enter_full_screen())
        self.assertTrue(workspace.full_screen_active)
        self.assertFalse(page.controls.isVisible())
        self.assertFalse(page.stream.isVisible())      # the live feed goes as well
        self.assertTrue(page.canvas.isVisible())

        self.assertTrue(workspace.leave_full_screen())
        self.assertFalse(workspace.full_screen_active)
        self.assertTrue(page.controls.isVisible())
        self.assertTrue(page.stream.isVisible())

    def test_the_second_enter_is_a_no_op(self):
        page, workspace = self._shown()
        self.assertTrue(workspace.enter_full_screen())
        self.assertFalse(workspace.enter_full_screen())    # already full screen
        self.assertTrue(workspace.leave_full_screen())
        self.assertFalse(workspace.leave_full_screen())    # already back

    def test_chrome_the_page_had_hidden_stays_hidden(self):
        """Back means *back*: a control hidden on purpose must not reappear."""
        page, workspace = self._shown()
        page.controls.hide()
        workspace.enter_full_screen()
        workspace.leave_full_screen()
        self.assertFalse(page.controls.isVisible())

    def test_escape_leaves_full_screen_and_only_then(self):
        page, workspace = self._shown()
        escape = workspace.full_screen._escape
        self.assertFalse(escape.isEnabled())          # outside, Esc abandons a box

        workspace.enter_full_screen()
        self.assertTrue(escape.isEnabled())
        escape.activated.emit()
        self.assertFalse(workspace.full_screen_active)
        self.assertFalse(escape.isEnabled())

    def test_full_screen_keeps_the_picture_and_nothing_of_its_own(self):
        """`X-N12`: full screen hides the chrome and leaves the image alone.

        The numbers are **not** repeated on the image any more (2026-10-18) - they live in
        the panel and in the window header - so what has to hold is the other half of the
        rule: going full screen adds nothing to the canvas, and takes nothing from it.
        """
        page, workspace = self._shown()
        workspace.enter_full_screen()
        self.app.processEvents()
        self.assertFalse(page.controls.isVisible())
        self.assertEqual(page.canvas.findChildren(QWidget), [],
                         "full screen put something of its own over the picture")


class PaintTests(CanvasTestCase):
    """What is actually drawn **on the picture**, looked at in the pixels.

    The strip of cards used to be the one thing over the image, and a widget-tree test
    could not tell whether it was painted: this renders the whole canvas - parent first,
    children on top, exactly as Qt paints it - and looks at what is there. What may be
    there now is a page's own **draw bar** (`X-N20`) and nothing else.
    """

    def _painted(self, canvas, scheme, image=WHITE_IMAGE):
        """Render the canvas under one scheme's stylesheet and photograph it."""
        canvas.setStyleSheet(theme.build_stylesheet(scheme))
        canvas.set_source(image)               # fitted into the canvas
        return canvas.grab().toImage()

    def test_the_workspace_adds_nothing_to_the_picture(self):
        """In **both** schemes: what is on the canvas is the image, and only the image.

        Measured, not inspected: the same canvas is painted twice - once with a workspace
        built for it, once without - and the two photographs must match pixel for pixel. A
        card pinned to the corner used to make them differ.
        """
        for scheme in theme.SCHEMES:
            with self.subTest(scheme=scheme):
                plain = self._page((480, 320))
                plain.show()
                self.addCleanup(plain.hide)
                furnished = self._page((480, 320))
                furnished.show()
                self.addCleanup(furnished.hide)
                workspace = CanvasWorkspace(furnished, furnished.canvas)
                workspace.add_tool(ToolSpec("save", "Save"), lambda: None)

                self.assertEqual(self._painted(plain.canvas, scheme),
                                 self._painted(furnished.canvas, scheme),
                                 "the workspace drew something on the picture")

    def test_the_draw_bar_is_drawn_on_the_picture_and_nowhere_else(self):
        """`X-N20`: the tools that draw *on* the image sit on the image, lower-left.

        The frame is made the size of the canvas, so "the picture" and "the canvas" are
        the same rectangle and a pixel says which of the two was painted.
        """
        full = np.full((320, 480, 3), 255, dtype=np.uint8)
        page, workspace = self._workspace((480, 320))
        workspace.set_group("Measure")
        line = workspace.add_tool(ToolSpec("line", "Line"), lambda: None)
        page.show()
        self.addCleanup(page.hide)
        self.app.processEvents()

        before = self._painted(page.canvas, "dark", full)
        bar = workspace.set_draw_tools([line])
        self.app.processEvents()

        self.assertTrue(bar.isVisibleTo(page.canvas))
        self.assertEqual((bar.x(), bar.y()),
                         (MARGIN, page.canvas.height() - bar.height() - MARGIN))
        self.assertEqual([button.text() for button in bar.buttons], ["Line"])

        after = self._painted(page.canvas, "dark", full)
        centre = bar.mapTo(page.canvas, bar.rect().center())
        self.assertEqual(before.pixelColor(centre).name(), "#ffffff")
        self.assertNotEqual(after.pixelColor(centre).name(), "#ffffff",
                            "the draw bar is not painted over the picture")
        # The rest of the picture is untouched: the bar is a plate in the corner, not a
        # tint over the specimen.
        corner = (page.canvas.width() - MARGIN, MARGIN)
        self.assertEqual(after.pixelColor(*corner), before.pixelColor(*corner))
        # ... and every other tool of the page is a **menu item**, not a button on the
        # image (`X-N13`): the bar's own buttons are the only ones on the canvas.
        self.assertEqual(page.canvas.findChildren(QToolButton), bar.buttons)

    def test_a_page_that_asks_for_nothing_draws_nothing(self):
        """No card, no plate, no button: the image is the image (`X-N13`)."""
        page, workspace = self._workspace((480, 320))
        workspace.add_tool(ToolSpec("save", "Save"), lambda: None)
        page.show()
        self.addCleanup(page.hide)
        self.app.processEvents()

        self.assertEqual(page.canvas.findChildren(QToolButton), [])
        self.assertEqual([child.objectName() for child in page.canvas.children()
                          if isinstance(child, QWidget)], [])


if __name__ == "__main__":
    unittest.main()



