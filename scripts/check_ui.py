#!/usr/bin/env python3
"""Check every surface's canvas offscreen: paint it, go full screen, come back.

    python scripts/check_ui.py          # offscreen, temp database, no window shown

The unit tests cover the canvas tools in the abstract and each screen's own contract;
this is the crude end-to-end version a user would do by hand: build the **real**
windows (Calibrate, Bench, Dataset, Predict, and the Study shell's pages), let Qt
paint each canvas, enter the full-screen view, paint again, and leave it - asserting
that the panel really disappears and comes back. A crash *inside* `paintEvent` (an
exception in Qt's event loop) is the failure mode this catches, and the test suite
cannot see it.

Every window is built **twice**, once per theme scheme, so a widget that is only
painted correctly in one of them cannot hide behind the other.

Exit code 0 = every surface survived; 1 = the first problem, printed with its cause.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The application is driven without a display; a throwaway database keeps the run
# from reading (or writing) the real zoo.db.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DAPHNIA_DB",
                      str(Path(tempfile.mkdtemp(prefix="daphnia-ui-check-"))
                          / "check.db"))

from PySide6.QtWidgets import QApplication, QScrollArea   # noqa: E402

from daphnia import paths                             # noqa: E402
from daphnia.db import init_db                        # noqa: E402
from daphnia.ui.hub import (                          # noqa: E402
    BenchWindow,
    CalibrationWindow,
    DatasetWindow,
    MainWindow,
    PredictWindow,
    Study2Window,
)
from daphnia.ui import theme                           # noqa: E402

#: Every surface window, in the order of the hub.
WINDOWS = (CalibrationWindow, Study2Window, BenchWindow, DatasetWindow, PredictWindow,
           MainWindow)


def canvases(window):
    """``(name, page)`` for every page of a window that has a canvas."""
    pages = getattr(window, "pages", None)
    if pages is None:
        page = getattr(window, "page", None)
        pages = {"page": page} if page is not None else {}
    return [(name, page) for name, page in pages.items()
            if hasattr(page, "canvas") and hasattr(page, "workspace")]


def check_page(window, name, page, window_name, verbose):
    """Paint this page's canvas, go full screen, paint again, come back."""
    # In the Study shell only the current step is visible; show this one. The workflow
    # gates would refuse it without a study, which is not what this check is about.
    stacked = getattr(window, "stackedWidget", None)
    if stacked is not None:
        stacked.setCurrentWidget(page)
        QApplication.processEvents()

    painted = page.canvas.grab()
    assert not painted.isNull(), f"{window_name}.{name}: the canvas painted nothing"
    assert page.controls.isVisible(), f"{window_name}.{name}: the panel is not up"

    # The panel scrolls vertically only, so content wider than the viewport would be
    # unreachable - the one thing a user notices immediately and a unit test can miss.
    scroll = page.controls.findChild(QScrollArea, "controls_scroll")
    assert scroll is not None, f"{window_name}.{name}: the panel has no scroll area"
    assert scroll.widget().minimumSizeHint().width() <= scroll.viewport().width(), (
        f"{window_name}.{name}: the controls panel clips its content "
        f"({scroll.widget().minimumSizeHint().width()} px needed, "
        f"{scroll.viewport().width()} px available)")

    assert page.workspace.enter_full_screen(), f"{window_name}.{name}: no full screen"
    QApplication.processEvents()
    assert page.workspace.full_screen_active, f"{window_name}.{name}: not active"
    assert not page.controls.isVisible(), f"{window_name}.{name}: the panel stayed"
    # The status footer is window chrome too (`WINDOW_CHROME`), and it is on *every*
    # window now - full screen has to leave the image alone.
    footer = getattr(window, "status_footer", None)
    if footer is not None:
        assert not footer.isVisible(), f"{window_name}.{name}: the footer stayed up"
    # ... and so is the header, with its menus and its facts.
    header = getattr(window, "header_bar", None)
    if header is not None:
        assert not header.isVisible(), f"{window_name}.{name}: the header stayed up"
    page.canvas.grab()                     # paint it full screen as well

    assert page.workspace.leave_full_screen(), f"{window_name}.{name}: no way back"
    QApplication.processEvents()
    assert page.controls.isVisible(), f"{window_name}.{name}: the panel did not return"
    if footer is not None:
        assert footer.isVisible(), f"{window_name}.{name}: the footer did not return"
    if header is not None:
        assert header.isVisible(), f"{window_name}.{name}: the header did not return"

    if verbose:
        print(f"    {name:12} {painted.width():4} x {painted.height():<4} "
              f"{len(page.workspace.tools.keys)} tools   paint + full screen ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="only report failures")
    args = parser.parse_args()

    app = QApplication([])
    app.setApplicationName("DaphniaScope")
    paths.ensure_runtime_dirs()
    init_db()

    # Every surface is painted in **both** schemes. A widget that is only right in one
    # of them - an inline colour, a colour the template forgot - cannot hide here, and
    # the theme raises on a missing token instead of half-painting a window.
    checked = 0
    for scheme in theme.SCHEMES:
        theme.apply_theme(app, scheme)
        if not args.quiet:
            print(f"  theme: {scheme}")
        for factory in WINDOWS:
            window = factory()
            window.show()
            QApplication.processEvents()
            if not args.quiet:
                print(f"    {factory.__name__}")
            try:
                for name, page in canvases(window):
                    check_page(window, name, page, factory.__name__, not args.quiet)
                    checked += 1
            except AssertionError as failure:
                print(f"error: {failure}", file=sys.stderr)
                return 1
            finally:
                window.close()
                QApplication.processEvents()

    print(f"ok: {checked} canvas(es) painted in {' and '.join(theme.SCHEMES)}, "
          f"went full screen and came back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
