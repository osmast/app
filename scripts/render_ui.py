#!/usr/bin/env python3
"""Render every surface offscreen to PNGs, so a layout change can be *looked at*.

    python scripts/render_ui.py                 # -> recordings/ui-render/*.png
    python scripts/render_ui.py --out /tmp/ui   # somewhere throwaway
    python scripts/render_ui.py --calibrate-photo recordings/calibration/PROF_x.png

`make ui-check` proves a surface *paints*; it cannot show whether it is any good. This
does the crude thing a reviewer does by hand: build the real windows offscreen (no
display), grab them, and put the pictures somewhere they can be opened. It renders the
**live store's** state (a copy of it - the original is never opened for writing), because
the questions a reviewer has ("what does the panel say when five calibrations exist?")
are questions about the operator's data.

The Calibrate screen gets a second shot with a real photo loaded and the grid measured on
it, which is the state the daily act ends in; `--calibrate-photo` chooses the photo.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The application is driven without a display, and against a **copy** of the store, so a
# render can never change what it is showing. Set before PySide6 is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DEFAULT_PHOTO = "recordings/calibration/PROF_LEICADM500_2X_IPHONE_1.0X_INFINITY_20261018_133641.png"


def _isolate_store() -> str:
    """Copy the live database into a temp dir and point the app at the copy."""
    from daphnia import paths

    source = paths.DB_PATH
    target = Path(tempfile.mkdtemp(prefix="daphnia-render-")) / source.name
    if source.exists():
        shutil.copy(source, target)
    os.environ["DAPHNIA_DB"] = str(target)
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="output folder")
    parser.add_argument("--calibrate-photo", default=None,
                        help="photo to load and measure on the Calibrate screen")
    args = parser.parse_args()

    _isolate_store()

    from PySide6.QtWidgets import QApplication

    from daphnia import paths
    from daphnia.db import init_db
    from daphnia.ui import theme
    from daphnia.ui.hub import (BenchWindow, CalibrationWindow, DatasetWindow,
                                MainWindow, PredictWindow)

    out = Path(args.out).expanduser() if args.out else paths.RECORDINGS_DIR / "ui-render"
    out.mkdir(parents=True, exist_ok=True)
    photo = Path(args.calibrate_photo or DEFAULT_PHOTO).expanduser()
    if not photo.is_absolute():
        photo = paths.PROJECT_ROOT / photo

    app = QApplication([])
    app.setApplicationName("DaphniaScope")
    paths.ensure_runtime_dirs()
    init_db()

    written = []
    for scheme in theme.SCHEMES:
        theme.apply_theme(app, scheme)
        for factory in (CalibrationWindow, BenchWindow, DatasetWindow, PredictWindow,
                        MainWindow):
            window = factory()
            window.resize(1500, 900)
            window.show()
            QApplication.processEvents()
            shot = out / f"{scheme}_{factory.__name__}.png"
            window.grab().save(str(shot))
            written.append(shot)
            window.close()
            QApplication.processEvents()

    # The daily act, end to end: a real photo, then the grid measured on it - in **both**
    # schemes, because a colour that only works on one of them is exactly the defect a
    # reviewer is looking for here (the verdict wears a theme token).
    for scheme in theme.SCHEMES:
        theme.apply_theme(app, scheme)
        window = CalibrationWindow()
        window.resize(1500, 900)
        window.show()
        QApplication.processEvents()
        page = window.page
        if photo.exists():
            page.use_photo(str(photo))
            QApplication.processEvents()
            page.measure_grid()
            QApplication.processEvents()
            for name, label in (("grid", page.grid_label),
                                ("period", page.label_result),
                                ("drift", page.label_drift),
                                ("field", page.label_field_stop),
                                ("verdict", page.label_verdict),
                                ("distortion", page.label_distortion),
                                ("scale", page.uncertainty_label),
                                ("state", page.in_force_label),
                                ("status", page.status_label)):
                # Printed because a picture shows *where* the facts land and only this
                # shows *what* they say.
                print(f"  calibrate[{scheme}]/{name}: {label.text()}")
        shot = out / f"{scheme}_CalibrationWindow_measured.png"
        window.grab().save(str(shot))
        written.append(shot)
        window.close()
        QApplication.processEvents()

    print(f"written {len(written)} render(s) to {out}")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
