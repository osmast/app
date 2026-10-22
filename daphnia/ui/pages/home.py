"""Home hub - choose a surface (what you want to do), not a screen.

The landing screen of the application. It offers every surface from
:mod:`daphnia.surfaces` as a tile, states whether each is usable today, and hands
the choice to :class:`~daphnia.ui.hub.HubWindow`, which owns the actual windows.
The page itself knows nothing about *how* a surface is opened, so the launch
points stay in one place.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from daphnia import surfaces
from daphnia.db import connection, queries
from daphnia.ui.shell import ShellPage


class Home(ShellPage):
    """The hub: one tile per surface, plus the state of the shared backend.

    Signals:
        surface_chosen: emitted with a surface key when its tile is activated.
    """

    surface_chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(ui_file='home.ui', split=False, parent=parent)

        self.tiles = {}        # surface key -> the tile widget
        self.buttons = {}      # surface key -> the tile's button
        self._build_tiles()
        self.refresh()

    def on_shown(self):
        self.refresh()

    # -- tiles -------------------------------------------------------------
    def _build_tiles(self):
        """One tile per surface, in the order the registry declares them."""
        layout = self.tiles_container.layout()
        for surface in surfaces.SURFACES:
            layout.addWidget(self._make_tile(surface))
        layout.addStretch(1)

    def _make_tile(self, surface) -> QGroupBox:
        """Build the tile for one surface."""
        tile = QGroupBox(surface.label)
        tile.setObjectName(f"tile_{surface.key}")
        column = QVBoxLayout(tile)

        tagline = QLabel(surface.tagline)
        tagline.setObjectName("tile_tagline")
        tagline.setWordWrap(True)

        summary = QLabel(surface.summary)
        summary.setWordWrap(True)

        meta = QLabel(f"Serves: {surface.audience}\nNeeds: {surface.gate}")
        meta.setObjectName("tile_meta")
        meta.setWordWrap(True)

        status = QLabel(f"{surface.status_label}   -   {surface.launch_point()}")
        status.setObjectName("tile_status")
        status.setWordWrap(True)

        for widget in (tagline, summary, meta, status):
            column.addWidget(widget)

        # Only a built surface is offered as "Open"; the others show their plan.
        button = QPushButton("Open" if surface.is_available else "View plan")
        button.setObjectName(f"open_{surface.key}")
        button.clicked.connect(
            lambda _checked=False, key=surface.key: self.surface_chosen.emit(key))

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button)
        column.addLayout(row)

        self.tiles[surface.key] = tile
        self.buttons[surface.key] = button
        # Expose it like a .ui widget, so the smoke tests can reach it.
        setattr(self, f"open_{surface.key}", button)
        return tile

    # -- state of the shared backend ---------------------------------------
    def refresh(self):
        """Report what only the hub knows: the surfaces and the current project.

        The optical scale, its uncertainty and its age live in the shared
        :class:`~daphnia.ui.widgets.StatusFooter`, together with the camera state, so
        they read identically in every window.
        """
        with connection() as conn:
            study = queries.active_study(conn)

        project = study["name"] if study is not None else "none"
        ready = len(surfaces.available())
        total = len(surfaces.SURFACES)
        self.status_label.setText(
            f"{total} surfaces: {ready} ready, {total - ready} specified.   "
            f"Current project: {project}")
