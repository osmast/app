"""Live preview of the shared :class:`~daphnia.streaming.StreamService`.

A preview is a *view*: it binds to the app's single stream connection, so the phone
is connected once (on the Setup step) and observed everywhere else. The preview
caps its own height at the stream's aspect ratio, so the image fills it exactly
instead of being letterboxed inside a taller box.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from daphnia.ui.widgets.image_canvas import (
    PANEL_MINIMUM,
    frame_to_qimage,
)

#: Smallest the preview may become - identical to the image canvas, so the stream
#: and image panels look the same from screen to screen.
DEFAULT_MINIMUM = PANEL_MINIMUM


class StreamView(QLabel):
    """Live preview bound to a :class:`~daphnia.streaming.StreamService`."""

    def __init__(self, service=None, minimum=DEFAULT_MINIMUM, lock_ratio=True,
                 hide_when_empty=True, parent=None):
        super().__init__(parent)
        # The panel's colours come from the theme, keyed on this name - never from an
        # inline stylesheet, which would keep the pane dark (or light) after the
        # desktop switched appearance.
        self.setObjectName("stream_view")
        self.lock_ratio = lock_ratio
        self.hide_when_empty = hide_when_empty
        #: Whether the page wants the *live preview* at all. A measuring page that has taken
        #: a frame goes to **image mode** and turns this off, so the frozen picture fills the
        #: area instead of playing video beside it (the camera keeps running - this is a view).
        self._preview_enabled = True
        self._service = None
        self._aspect = None

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(*minimum)
        # Ignored, so a large pixmap never pushes the window wider.
        policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        policy.setHeightForWidth(lock_ratio)
        self.setSizePolicy(policy)
        self.setText("Not connected")

        if service is not None:
            self.bind(service)

    # -- binding -----------------------------------------------------------
    def bind(self, service):
        """Observe ``service``: its frames, its aspect ratio and its state."""
        if service is self._service:
            return
        if self._service is not None:
            self._service.frames_changed.disconnect(self._on_frame)
            self._service.ratio_changed.disconnect(self._on_ratio)
            self._service.status.disconnect(self._on_status)
            self._service.closed.disconnect(self._on_closed)
        self._service = service
        service.frames_changed.connect(self._on_frame)
        service.ratio_changed.connect(self._on_ratio)
        service.status.connect(self._on_status)
        service.closed.connect(self._on_closed)
        self._aspect = service.aspect
        self._cap_height_to_ratio()
        self.show_state()
        self._on_frame()

    def show_state(self):
        """Say why the preview is empty - and step aside while there is nothing to show.

        "Not connected" is not good enough as a *state*: it does not distinguish
        "nobody tried" from "connecting", "connected but silent" or an outright
        failure. And an empty preview is not worth 260 px of the visual area, so with
        no frame the panel hides and the image keeps the whole space. The connection
        state is on the shared status footer either way.
        """
        frame = self._frame_of(self._service)
        if frame is None:
            self.clear()
            self.setText("Not connected" if self._service is None
                         else self._service.state_label())
        self._sync_visibility(frame is not None)

    @staticmethod
    def _frame_of(service):
        """The newest frame of ``service``, or ``None``."""
        return service.current_frame() if service is not None else None

    def set_preview_enabled(self, enabled):
        """Show the live preview (True) or stand aside for a **frozen image** (False).

        A page that has taken a frame (or opened a photo) is measuring an *image*, and the
        moving video beside it is noise - so it turns the preview off and the canvas fills the
        area. The stream itself is untouched: this is a view, and the shared camera keeps
        running, so *Use the live frame* still grabs the newest picture while the preview is
        hidden.
        """
        self._preview_enabled = bool(enabled)
        self.show_state()

    @property
    def preview_enabled(self) -> bool:
        """Whether this preview is allowed to be on screen (the page's choice)."""
        return self._preview_enabled

    def _sync_visibility(self, has_frame):
        """Occupy space only while the page wants the preview *and* there is a frame."""
        wanted = self._preview_enabled and (has_frame or not self.hide_when_empty)
        if wanted == self.isHidden():             # only act when it must change
            self.setVisible(wanted)

    def _on_status(self, *_args):
        self.show_state()

    def _on_closed(self):
        """A dropped connection must not leave a stale picture on screen."""
        self.clear()
        self.show_state()

    @property
    def service(self):
        """The stream this preview shows."""
        return self._service

    # -- aspect ratio ------------------------------------------------------
    @property
    def aspect(self):
        """width / height of the stream, or ``None`` before the first frame."""
        return self._aspect

    def hasHeightForWidth(self):
        """True once the stream's ratio is known and ratio locking is on."""
        return self.lock_ratio and self._aspect is not None

    def heightForWidth(self, width):
        """Height that shows the stream undistorted at ``width``.

        While the ratio is known the layout treats this as the item's *minimum*
        height, so it must never report 0 - before the first frame the widget's own
        minimum is the honest answer.
        """
        if self.hasHeightForWidth():
            return max(1, int(width / self._aspect))
        return self.minimumHeight()

    def _cap_height_to_ratio(self):
        """Keep the preview exactly as tall as its own ratio, never letterboxed."""
        if not self.hasHeightForWidth():
            return
        wanted = self.heightForWidth(self.width())
        if self.maximumHeight() != wanted:
            self.setMaximumHeight(wanted)

    # -- painting ----------------------------------------------------------
    def _on_ratio(self, aspect):
        self._aspect = aspect
        self._cap_height_to_ratio()
        self.updateGeometry()

    def _on_frame(self):
        frame = self._frame_of(self._service)
        if frame is None:
            self.show_state()
            return
        self._sync_visibility(True)
        pixmap = QPixmap.fromImage(frame_to_qimage(frame))
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._cap_height_to_ratio()
        self._on_frame()
