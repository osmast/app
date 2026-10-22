#!/usr/bin/env python3
"""Entry point for the desktop application.

    python run.py             Home hub - choose a surface
    python run.py study       the governed study application
    python run.py study2      capture a whole plate, well by well
    python run.py bench       measure a specimen now, no study needed
    python run.py optics      calibrate the optics
    python run.py dataset     build and label the training corpus
    python run.py --help      list the launch points

See :mod:`daphnia.surfaces` for what each surface is for, and
``docs/requirements/`` for what it must do.
"""

import sys

from daphnia.ui.app import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))

