"""Select step - pick the best frame of each well / individual.

Selection is deliberately separate from annotation: here every burst of a target is
on the table at once and exactly **one** frame is kept - the one that will be
measured. The others go to the tray, so each well or individual yields exactly one
measurement.
"""

from PySide6.QtCore import QTimer

from daphnia import workflow
from daphnia.db import connection, queries, repository
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import ImageViewer, ThumbnailGallery


class Select(ShellPage):
    """Choose the best frame per target and discard the rest."""

    def __init__(self, parent=None):
        super().__init__(ui_file='select.ui', parent=parent)

        self.gallery = self.add_visual(ThumbnailGallery(selectable=True))
        self.gallery.opened.connect(self._open_fullscreen)
        self.gallery.decided.connect(self._on_decided)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_pick_sharpest.clicked.connect(self._pick_sharpest)
        self.btn_discard_all.clicked.connect(self._clear_selection)
        self.combo_target.currentIndexChanged.connect(self._load_selected)
        self.btn_prev_target.clicked.connect(lambda: self._step_target(-1))
        self.btn_next_target.clicked.connect(lambda: self._step_target(1))

        self._observation_id = None
        self.refresh()

    def on_shown(self):
        self.refresh()

    # -- targets -----------------------------------------------------------
    def refresh(self):
        with connection() as conn:
            study = queries.active_study(conn)
            targets = queries.list_targets(conn, study["study_id"]) if study else []

        previous = self._observation_id
        self.combo_target.blockSignals(True)
        self.combo_target.clear()
        for row in targets:
            self.combo_target.addItem(
                f"{queries.target_label(row)}  ({row['frames']} frames, "
                f"{row['kept']} kept)", row["observation_id"])
        # Stay on the individual the operator was reviewing, if it still exists.
        index = self.combo_target.findData(previous) if previous is not None else -1
        self.combo_target.setCurrentIndex(max(index, 0))
        self.combo_target.blockSignals(False)

        if self.combo_target.count() == 0:
            self._observation_id = None
            self.gallery.load([])
            self.summary_label.setText("Nothing captured yet.")
            self.status_label.setText(workflow.REASONS["observation"])
            self.btn_pick_sharpest.setEnabled(False)
            self.btn_discard_all.setEnabled(False)
            self._sync_nav()
            return
        self._load_selected()

    def _load_selected(self):
        self._observation_id = self.combo_target.currentData()
        self._reload_gallery()
        self._sync_nav()

    # -- navigation --------------------------------------------------------
    def _step_target(self, delta):
        """Move to the previous/next individual (back and forth)."""
        index = self.combo_target.currentIndex() + delta
        if 0 <= index < self.combo_target.count():
            self.combo_target.setCurrentIndex(index)

    def _sync_nav(self):
        """Enable the back/forward buttons and report where we are."""
        index = self.combo_target.currentIndex()
        count = self.combo_target.count()
        self.btn_prev_target.setEnabled(index > 0)
        self.btn_next_target.setEnabled(0 <= index < count - 1)
        self.position_label.setText(
            f"Individual {index + 1} of {count}" if count else "-")

    def _reload_gallery(self):
        assets = []
        if self._observation_id is not None:
            with connection() as conn:
                assets = queries.assets_for(conn, self._observation_id)
        self.gallery.load(assets)
        kept = sum(1 for row in assets if row["selection"] == "kept")
        discarded = sum(1 for row in assets if row["selection"] == "discarded")
        self.summary_label.setText(
            f"frames {len(assets)}   kept {kept}   discarded {discarded}")
        self.btn_pick_sharpest.setEnabled(bool(assets))
        self.btn_discard_all.setEnabled(bool(assets))

    # -- decisions ---------------------------------------------------------
    def _on_decided(self, asset_id, selection):
        """A ticked frame becomes the single best one of its target."""
        if self._observation_id is None:
            return
        with connection() as conn:
            if selection == "kept":
                repository.select_best_frame(conn, self._observation_id, asset_id)
            else:
                repository.set_asset_selection(conn, asset_id, selection)
        self.status_label.setText(
            f"Kept {asset_id} - the other frames of this target go to the tray."
            if selection == "kept" else f"Discarded {asset_id}.")
        # select_best_frame rewrites the sibling frames, so repaint after this event.
        QTimer.singleShot(0, self._reload_gallery)
        self.notify_workflow()

    def _pick_sharpest(self):
        if self._observation_id is None:
            return
        with connection() as conn:
            asset_id = repository.select_sharpest_frame(conn, self._observation_id)
        self.status_label.setText(
            f"Kept the sharpest frame ({asset_id})." if asset_id
            else "No frames to choose from.")
        self._reload_gallery()
        self.notify_workflow()

    def _clear_selection(self):
        if self._observation_id is None:
            return
        with connection() as conn:
            repository.clear_selection(conn, self._observation_id)
        self.status_label.setText("Selection cleared for this target.")
        self._reload_gallery()
        self.notify_workflow()

    def _open_fullscreen(self, path):
        ImageViewer(path, parent=self).exec()
