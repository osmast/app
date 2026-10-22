"""Full-screen inspection of a single frame."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout


class ImageViewer(QDialog):
    """Show one frame fitted to the screen; Esc or a click closes it."""

    def __init__(self, path, title="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title or str(path))
        self.setModal(True)
        # Colours come from the theme's canvas tokens (see assets/theme/*.qss), so
        # the viewer matches the canvas it opens from - in either scheme.
        self.setObjectName("image_viewer")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel()
        self.label.setObjectName("image_viewer_text")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        pixmap = QPixmap(str(path))
        screen = QGuiApplication.primaryScreen()
        if pixmap.isNull():
            self.label.setText(f"Could not load {path}")
            return

        available = screen.availableGeometry() if screen is not None else None
        if available is not None:
            target = QSize(int(available.width() * 0.95),
                           int(available.height() * 0.95))
            pixmap = pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)
        self.resize(pixmap.size())

    def mousePressEvent(self, event):
        """Any click dismisses the viewer."""
        self.accept()
