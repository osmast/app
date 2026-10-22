#!/usr/bin/env python3
"""Open one screen in Qt's own UI editor (Qt Designer), against *this* app.

    python scripts/edit_ui.py bench.ui            # or: make ui-designer FILE=bench.ui
    python scripts/edit_ui.py --check bench.ui    # verify only, open nothing
    python scripts/edit_ui.py --list              # verify every screen

The ``.ui`` files are plain Qt Designer XML and every one of them is loaded by the
application at runtime (``daphnia/ui/loader.py`` -> ``QUiLoader``), so Designer is the
right tool for the *controls* of a screen. Two things it will **not** show, because
they are code:

* the **visual area** - the live preview, the image canvas and the on-canvas tool
  clusters are built by ``ShellPage.add_visual`` / ``canvas_workspace`` and added at
  runtime. A screen opened here looks like a form; that form *is* the controls panel,
  and the empty area beside it in the running app is where the picture goes;
* the shell's post-load surgery - ``MainWindow`` wraps ``stackedWidget`` in a scroll
  area and mounts the footer + step bar into ``footer_container``.

Two rules keep a ``.ui`` file loadable at runtime, and both are checked here (and in
``tests/test_ui_files.py``) so a Designer edit cannot half-break a screen:

1. **plain Qt widgets only.** ``QUiLoader`` cannot create a Python class, so a
   *promoted* widget arrives as a plain ``QWidget`` - the screen then renders a hole
   where the control should be, with no error;
2. **no resources.** The app must start without a compiled ``.qrc``, so an icon added
   from a resource file would be missing on a machine that has not built it.

The file is loaded once with the *same loader the application uses* before the editor
opens, so "would this even open?" is answered before any time is spent editing.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daphnia import paths                                  # noqa: E402

#: Editor binaries, in the order they are looked for. ``Designer6`` is what the
#: conda-forge PySide6 package ships, and it matches the Qt the app runs on - another
#: Qt's Designer would still read the file (the XML is stable) but could write
#: properties this runtime does not know.
EDITORS = ("pyside6-designer", "Designer6", "designer")


def find_editor():
    """The Qt Designer to use: the environment's own first, then whatever is on PATH."""
    prefix = os.environ.get("CONDA_PREFIX")
    for name in EDITORS:
        if prefix:
            candidate = Path(prefix) / "bin" / name
            if candidate.is_file():
                return str(candidate)
        found = shutil.which(name)
        if found:
            return found
    return None


def resolve(ui_name) -> Path:
    """The ``.ui`` file for a bare name (``bench.ui``, ``bench``) or a path."""
    candidate = Path(ui_name)
    if candidate.is_absolute():
        return candidate
    if candidate.parent == Path("."):           # a bare name means "a screen by name"
        name = candidate.name
        return paths.UI_DIR / (name if name.endswith(".ui") else f"{name}.ui")
    return (paths.PROJECT_ROOT / candidate).resolve()


def inspect(path: Path) -> list:
    """Report what the app would get from this file; returns the problems found."""
    text = path.read_text(encoding="utf-8")
    problems = []
    if "<customwidget" in text:
        problems.append(
            "this file **promotes** a widget: QUiLoader cannot create a Python class, "
            "so the screen shows a plain QWidget there (see daphnia/ui/loader.py)")
    if "<resources>" in text or ".qrc" in text:
        problems.append(
            "this file uses a **resource file**: the app would need a compiled .qrc, "
            "which it deliberately does not have")

    from PySide6.QtWidgets import QApplication
    from daphnia.ui.loader import load_ui

    QApplication.instance() or QApplication([])
    widget = load_ui(path.name)
    if not widget.objectName():
        problems.append("the top-level widget has no objectName - the loader copies it "
                        "onto the page, so give it the screen's name")
    names = sorted(child.objectName() for child in widget.findChildren(object)
                   if hasattr(child, "objectName") and child.objectName())
    print(f"  {path.name}: {type(widget).__name__} \"{widget.objectName()}\", "
          f"{len(names)} named widgets")
    print(f"    {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ui", nargs="?", default="pages.ui",
                        help="the screen to open (a name in assets/ui, or a path)")
    parser.add_argument("--check", action="store_true",
                        help="verify the file and report problems; open nothing")
    parser.add_argument("--list", action="store_true",
                        help="verify every screen in assets/ui")
    args = parser.parse_args()

    targets = (sorted(paths.UI_DIR.glob("*.ui")) if args.list else [resolve(args.ui)])
    failed = False
    for path in targets:
        if not path.is_file():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 2
        problems = inspect(path)
        for problem in problems:
            print(f"    problem: {problem}", file=sys.stderr)
        failed = failed or bool(problems)

    if args.check or args.list:
        if failed:
            print("problems found (see above)")
            return 1
        print("every screen loads the way the app loads it")
        return 0

    editor = find_editor()
    if editor is None:
        print("error: no Qt Designer found. Install one with:\n"
              "  conda install -n z-app pyside6        # ships Designer6\n"
              "or open assets/ui/*.ui in Qt Creator.", file=sys.stderr)
        return 1
    print(f"opening {targets[0].name} in {editor}")
    os.execv(editor, [editor, str(targets[0])])           # replaces this process
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

