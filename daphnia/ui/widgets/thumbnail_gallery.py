"""Thumbnail gallery of the frames of a target.

Used in two modes:

* ``selectable=False`` - acquisition: just look at the bursts to decide whether
  another burst is needed (double-click opens a frame full screen).
* ``selectable=True``  - review: each frame carries a keep/discard checkbox, which
  drives which frames reach the annotation carousel.
"""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem

THUMB = QSize(160, 90)


def check_state(selection):
    """Selection -> checkbox state (pending is neither kept nor discarded)."""
    if selection == "kept":
        return Qt.Checked
    if selection == "discarded":
        return Qt.Unchecked
    return Qt.PartiallyChecked


def selection_of(state):
    """Checkbox state -> selection."""
    if state == Qt.Checked:
        return "kept"
    if state == Qt.Unchecked:
        return "discarded"
    return "pending"


class ThumbnailGallery(QListWidget):
    """Icon grid of frames, labelled ``b<burst> f<frame>``.

    Signals:
        opened(str): the file path of the double-clicked frame.
        decided(str, str): (asset_id, selection) after a checkbox change
            (only in selectable mode).
    """

    opened = Signal(str)
    decided = Signal(str, str)

    def __init__(self, selectable=False, parent=None):
        super().__init__(parent)
        self.selectable = selectable
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(THUMB)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setSpacing(6)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setMinimumHeight(THUMB.height() + 40)
        if selectable:
            self.itemChanged.connect(self._on_item_changed)

    # -- content -----------------------------------------------------------
    def load(self, assets):
        """Fill from ``media_assets`` rows (asset_id/file_path/burst_index/...)."""
        self.blockSignals(True)
        self.clear()
        for row in assets:
            item = QListWidgetItem()
            pixmap = QPixmap(row["file_path"])
            if not pixmap.isNull():
                item.setIcon(QIcon(pixmap.scaled(THUMB, Qt.KeepAspectRatio,
                                                 Qt.SmoothTransformation)))
            item.setText(f"b{row['burst_index'] or 0} f{row['frame_index'] or 0}")
            item.setData(Qt.UserRole, row["asset_id"])
            item.setData(Qt.UserRole + 1, row["file_path"])
            if self.selectable:
                # Deliberately NOT ItemIsUserTristate: with it a click cycles
                # pending -> part -> checked -> ... so ticking a frame would first
                # mark it as *bad*. A plain checkbox toggles kept <-> discarded,
                # and a frame stays 'undecided' until the first click.
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(check_state(row["selection"]))
            self.addItem(item)
        self.blockSignals(False)

    def asset_ids(self) -> list:
        """Ids of the frames currently shown, in order."""
        return [self.item(index).data(Qt.UserRole) for index in range(self.count())]

    # -- helpers -----------------------------------------------------------
    def set_all(self, selection):
        """Apply one decision to every frame (used by the bulk buttons)."""
        self.blockSignals(True)
        for index in range(self.count()):
            self.item(index).setCheckState(check_state(selection))
        self.blockSignals(False)

    # -- interaction -------------------------------------------------------
    def _on_item_changed(self, item):
        asset_id = item.data(Qt.UserRole)
        if asset_id:
            self.decided.emit(asset_id, selection_of(item.checkState()))

    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if item is not None:
            path = item.data(Qt.UserRole + 1)
            if path:
                self.opened.emit(path)
        super().mouseDoubleClickEvent(event)
