"""Allow ``python -m daphnia`` as an alternative to ``python run.py``.

    python -m daphnia           the Home hub
    python -m daphnia study     a surface, by key (see daphnia.surfaces)
"""

import sys

from daphnia.ui.app import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))

