"""One module per step; each screen is a QWidget loaded from ``assets/ui``.

:class:`Home` is the exception: it is not a workflow step but the hub screen that
offers the surfaces (see :mod:`daphnia.surfaces`), so it is not registered in
:mod:`daphnia.workflow`.
"""

from daphnia.ui.pages.acquire import Acquire
from daphnia.ui.pages.annotate import Annotate
from daphnia.ui.pages.bench import Bench
from daphnia.ui.pages.calibration import Calibration
from daphnia.ui.pages.dashboard import Dashboard
from daphnia.ui.pages.dataset import Dataset
from daphnia.ui.pages.home import Home
from daphnia.ui.pages.predict import Predict
from daphnia.ui.pages.select import Select
from daphnia.ui.pages.setup_experiment import SetupExperiment
from daphnia.ui.pages.setup_plates import SetupPlates
from daphnia.ui.pages.study2 import Study2
from daphnia.ui.pages.visualize import Visualize

__all__ = [
    "Dashboard",
    "Home",
    "Bench",
    "Dataset",
    "Predict",
    "SetupExperiment",
    "SetupPlates",
    "Study2",
    "Calibration",
    "Acquire",
    "Select",
    "Annotate",
    "Visualize",
]
