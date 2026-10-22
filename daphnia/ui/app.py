"""Application bootstrap: surface dispatch, QApplication, database, theme, window.

One repository, one package, one database - several **launch points**. This module
is the single place that turns a command-line argument into a window:

    python run.py             the Home hub, where a surface is chosen
    python run.py study       the governed study application
    python run.py study2      capture a whole plate, well by well
    python run.py bench       measure a specimen now, no study needed
    python run.py optics      calibrate the optics
    python run.py dataset     build and label the training corpus
    python run.py --help      the launch points, listed

A surface with no window yet opens its plan, never the wrong window, so a launch
point can never silently do the wrong thing.
"""

import sys

from PySide6.QtWidgets import QApplication

from daphnia import paths, surfaces
from daphnia.db import init_db
from daphnia.streaming import shared_service
from daphnia.ui.hub import window_for
from daphnia.ui.theme import apply_theme

APPLICATION_NAME = "DaphniaScope"
ORGANISATION_NAME = "ZHAW Aquaculture Systems Research Group"

#: Arguments that print the launch points instead of opening a window.
HELP_ARGS = ("-h", "--help", "help", "--list", "list")

#: Exit code for a launch key that names no surface.
UNKNOWN_SURFACE_EXIT = 2


def usage() -> str:
    """The launch points, for ``--help``."""
    lines = [
        f"{APPLICATION_NAME} - one backend, several surfaces.",
        "",
        "  python run.py              open Home and choose a surface",
    ]
    for surface in surfaces.SURFACES:
        mark = "" if surface.is_available else "   (specified, not built yet)"
        lines.append(
            f"  python run.py {surface.key:<8} {surface.tagline}{mark}")
    return "\n".join(lines)


def parse_launch(args) -> str:
    """The launch key in ``args``: a surface key, or the hub when none is given.

    Qt options (anything starting with ``-``) are skipped, so this stays usable
    even if the application later accepts its own flags.
    """
    for arg in args or []:
        text = str(arg).strip()
        if not text or text.startswith("-"):
            continue
        return text.lower()
    return surfaces.HUB_KEY


def create_app(argv=None) -> QApplication:
    """Create the QApplication with application metadata set."""
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APPLICATION_NAME)
    app.setApplicationDisplayName(APPLICATION_NAME)
    app.setOrganizationName(ORGANISATION_NAME)
    return app


def main(argv=None) -> int:
    """Open the requested surface; returns the process exit code."""
    argv = list(sys.argv if argv is None else argv)
    args = argv[1:]

    if any(arg in HELP_ARGS for arg in args):
        print(usage())
        return 0

    key = parse_launch(args)
    if key != surfaces.HUB_KEY and surfaces.get(key) is None:
        print(f"{APPLICATION_NAME}: unknown surface '{key}'.\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return UNKNOWN_SURFACE_EXIT

    # The launch key is consumed above, so Qt only sees its own arguments.
    app = create_app(argv[:1] or [APPLICATION_NAME])

    # Local state (datasets/, models/, recordings/) and the database.
    paths.ensure_runtime_dirs()
    init_db()

    # One theme for every window: the tokens, the palette and the stylesheet come
    # from daphnia/ui/theme.py, and the scheme follows the desktop unless
    # DAPHNIA_THEME pins it (light | dark | system).
    apply_theme(app)

    # Stop the reader thread before the interpreter tears the app down: a QThread
    # destroyed while still running aborts the process.
    app.aboutToQuit.connect(shared_service().stop)

    window = window_for(key)
    window.show()
    return app.exec()
