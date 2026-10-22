"""Plate Layout & Treatment Builder - placeholder.

Will provide the multi-format canvas (6 / 24 / 96-well) and drag-and-drop
treatment assignment (medium, toxicant, concentration).
"""

from daphnia.ui.shell import ShellPage


class SetupPlates(ShellPage):
    def __init__(self, parent=None):
        super().__init__(ui_file='setup_plates.ui', split=False, parent=parent)
