"""The ``.ui`` files: what Qt Designer may edit, and what the app needs from them.

Every screen's controls live in ``assets/ui/*.ui`` and are loaded at runtime by
:func:`daphnia.ui.loader.load_ui` (``QUiLoader``). That makes **Qt Designer** the tool
for them - ``make ui-designer FILE=bench.ui`` opens one - and this module is what keeps
a Designer edit from half-breaking a screen, because none of these failures raise an
error at runtime: the file still loads, and a control is simply dead or invisible.

* **A duplicate ``objectName``** is the trap that was already there: two widgets may
  not share a name, because ``load_ui`` exposes *one* attribute per name (`findChildren`
  order decides which). ``annotate.ui`` had two ``btn_new_specimen`` buttons, so the
  "New specimen" button inside the Subject group was never wired and never disabled -
  and both were on screen.
* **A promoted widget** (``<customwidget>``) arrives as a plain ``QWidget``: a hole in
  the panel, with no error.
* **A resource file** (``.qrc``) needs compiling; the app must start without that.

Nothing here needs the application: the first half is pure XML, and the second half
only the loader.
"""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget           # noqa: E402

from daphnia import paths                                    # noqa: E402
from daphnia.ui.loader import load_ui                        # noqa: E402

#: Every screen file, as the application finds them.
UI_FILES = sorted(paths.UI_DIR.glob("*.ui"))


class UiXmlTests(unittest.TestCase):
    """Rules that hold for the file itself, checked without starting Qt."""

    def test_there_are_screens_to_check(self):
        self.assertTrue(UI_FILES)

    def test_every_screen_is_well_formed_xml(self):
        for path in UI_FILES:
            with self.subTest(file=path.name):
                ElementTree.parse(path)                      # raises if malformed

    def test_no_widget_name_is_used_twice(self):
        """One name, one widget: the loader exposes one attribute per name."""
        for path in UI_FILES:
            root = ElementTree.parse(path).getroot()
            names = [widget.get("name") for widget in root.iter("widget")
                     if widget.get("name")]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            with self.subTest(file=path.name):
                self.assertEqual(
                    duplicates, [],
                    f"{path.name} declares {duplicates} more than once - load_ui would "
                    f"expose only one of them, and the other control would do nothing")

    def test_no_screen_promotes_a_widget(self):
        """QUiLoader cannot create a Python class; a promoted widget becomes a hole."""
        for path in UI_FILES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name):
                self.assertNotIn("<customwidget", text,
                                 "a promoted widget is not loadable by QUiLoader")

    def test_no_screen_needs_a_resource_file(self):
        """The app starts without a compiled .qrc, so icons come from code or a path."""
        for path in UI_FILES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name):
                self.assertNotIn("<resources>", text)
                self.assertNotIn(".qrc", text)

    def test_every_widget_class_is_a_qt_widget(self):
        """Only classes QUiLoader can instantiate - no project-specific widgets."""
        from PySide6 import QtWidgets

        for path in UI_FILES:
            root = ElementTree.parse(path).getroot()
            for widget in root.iter("widget"):
                name = widget.get("class") or ""
                with self.subTest(file=path.name, widget=name):
                    self.assertTrue(hasattr(QtWidgets, name),
                                    f"{name} is not a Qt widget class")


class UiLoadTests(unittest.TestCase):
    """What a *page* gets out of a Designer edit, using the app's own loader."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_every_screen_loads_the_way_the_app_loads_it(self):
        for path in UI_FILES:
            with self.subTest(file=path.name):
                widget = load_ui(path.name)              # QUiLoader, as at runtime
                self.assertIsInstance(widget, QWidget)
                self.assertTrue(widget.objectName(),
                                "the top-level widget needs a name: load_ui copies it "
                                "onto the page")

    def test_every_named_control_becomes_an_attribute_of_the_page(self):
        """The contract page code depends on: `self.btn_ok`, not `findChild`."""
        for path in UI_FILES:
            page = QWidget()                             # stands in for a ShellPage
            with self.subTest(file=path.name):
                loaded = load_ui(path.name, page)
                self.assertIsNotNone(loaded)
                expected = [widget.get("name")
                            for widget in ElementTree.parse(path).getroot().iter("widget")
                            if widget.get("name")]
                missing = [name for name in expected
                           if not isinstance(getattr(page, name, None), QWidget)]
                self.assertEqual(missing, [],
                                 f"{path.name}: {missing} did not reach the page")

    def test_a_screen_with_a_duplicate_name_would_lose_a_control(self):
        """The rule above, demonstrated - so nobody 'fixes' the check away.

        `annotate.ui` really had two `btn_new_specimen` buttons (one of them dead); both are
        gone now, the action being the *Measure* menu's *Next specimen*. The demonstration
        therefore writes its own two-widget screen: what is being shown is the *loader's*
        behaviour, not a screen's.
        """
        directory = tempfile.mkdtemp()
        (Path(directory) / "duplicate.ui").write_text(
            "<ui version=\"4.0\"><class>Form</class><widget class=\"QWidget\" name=\"Form\">"
            "<widget class=\"QPushButton\" name=\"btn_twice\"/>"
            "<widget class=\"QPushButton\" name=\"btn_twice\"/>"
            "</widget></ui>")
        page = QWidget()
        load_ui("duplicate.ui", page, ui_dir=directory)
        buttons = [child for child in page.findChildren(QWidget)
                   if child.objectName() == "btn_twice"]
        self.assertEqual(len(buttons), 2, "the file really does declare the name twice")
        self.assertIsInstance(getattr(page, "btn_twice", None), QWidget,
                             "one name must mean one widget, or the page attribute is a "
                             "coin toss between them")


if __name__ == "__main__":
    unittest.main()

