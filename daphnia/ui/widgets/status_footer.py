"""The shared status footer: the camera and the scale, in every window.

Two facts about the **machine** belong on every screen, because both change while
you work and both change what a number means:

* **Is the camera delivering frames?** An open socket is not a picture, so the bar
  reports four distinct states - not connected, connecting, connected but silent,
  and live - and offers Connect/Disconnect from wherever the operator is standing.
* **Which scale is in force, and how far can it be trusted?** The scale, its
  uncertainty, its age, and a Calibrate action.

Each of these was previously reachable from exactly one screen - the Setup step for
the connection, the Calibrate step for the scale. That is why an Optics window could
truthfully report "no connected camera" while the phone was streaming happily: the
window had simply never opened a reader.

The bar is deliberately dumb: it reports state and emits intent. The host decides
what *Calibrate* means for it.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from daphnia import settings
from daphnia.db import connection, repository
from daphnia.streaming import shared_service
from daphnia.vision.calibration import calibration_age_hours, is_stale

#: Padding inside the bar, identical in every window.
FOOTER_PADDING = (12, 4, 12, 4)


class StatusFooter(QWidget):
    """Camera state and scale, side by side, in every window.

    Signals:
        calibrate_requested: the operator wants to (re)calibrate now.
    """

    calibrate_requested = Signal()

    def __init__(self, context="", service=None, show_calibrate=True, parent=None):
        super().__init__(parent)
        self.setObjectName("status_footer")
        self.context = context
        self._note = ""
        self.service = service or shared_service()

        row = QHBoxLayout(self)
        row.setContentsMargins(*FOOTER_PADDING)
        row.setSpacing(8)

        self.camera_label = QLabel()
        self.camera_label.setObjectName("camera_state_label")
        row.addWidget(self.camera_label)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setObjectName("btn_connect_camera")
        self.btn_connect.setToolTip(
            "Open or close the connection to the phone's stream. The phone publishes "
            "SRT; MediaMTX republishes the same path over RTSP, which this reads.")
        self.btn_connect.clicked.connect(self.toggle_connection)
        row.addWidget(self.btn_connect)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Plain)
        row.addWidget(divider)

        self.scale_label = QLabel()
        self.scale_label.setObjectName("status_scale_label")
        self.scale_label.setWordWrap(True)
        row.addWidget(self.scale_label, 1)

        self.btn_calibrate = QPushButton("Calibrate...")
        self.btn_calibrate.setObjectName("btn_calibrate")
        self.btn_calibrate.setToolTip(
            "Measure the 1 mm grid again - do this whenever the mount, the focus or "
            "the objective could have moved.")
        self.btn_calibrate.clicked.connect(self.calibrate_requested.emit)
        # The Optics window is the calibration, so it does not offer to open it: it
        # keeps the camera and the scale, which are the other half of the bar.
        self.btn_calibrate.setVisible(bool(show_calibrate))
        row.addWidget(self.btn_calibrate)

        # The camera state changes on its own, so the bar follows the service
        # instead of being polled. `status` fires on every transition, including the
        # first frame - the moment there is actually something to see.
        self.service.status.connect(self._on_status)
        self.service.opened.connect(self._on_status)
        self.service.closed.connect(self._on_status)

        self.refresh()

    # -- camera ------------------------------------------------------------
    def toggle_connection(self):
        """Connect the shared service, or disconnect it - from any window."""
        if self.service.state == self.service.DISCONNECTED:
            self.service.start()
        else:
            self.service.stop()
        self.refresh()

    def _on_status(self, *_args):
        self.refresh()

    # -- content -----------------------------------------------------------
    def set_context(self, context):
        """Extra text that belongs to the host (current project, and so on)."""
        self.context = context or ""
        self.refresh()

    def show_note(self, note):
        """Say something that is not about the machine (cleared on refresh)."""
        self._note = note or ""
        self.refresh()

    def refresh(self):
        """Re-read the camera state and the calibration in force."""
        with connection() as conn:
            profile = repository.active_profile(conn)

        self.camera_label.setText(self.service.state_label())
        # While a connection exists the button offers the way out of it, including
        # mid-attempt: "Connecting ..." can be cancelled.
        self.btn_connect.setText(
            "Disconnect" if self.service.state != self.service.DISCONNECTED
            else "Connect")
        self.scale_label.setText(self.summary(profile, self.context, self._note))

    @staticmethod
    def summary(profile, context="", note="") -> str:
        """The scale part of the bar, also used directly by the tests.

        ``Optical scale: 147.3 px/mm +/- 0.50%, calibrated 3.8 h ago   |   context``
        """
        if profile is None:
            scale = "no active preset - calibrate before measuring"
        else:
            scale = (f"{profile.px_per_um * 1000.0:.1f} px/mm "
                     f"+/- {profile.uncertainty_pct:.2f}%")
            age = calibration_age_hours(profile.calibrated_at)
            if age is not None:
                scale += f", calibrated {age:.1f} h ago"
            if is_stale(profile.calibrated_at, settings.CALIBRATION_MAX_AGE_HOURS):
                scale += " - RE-VERIFY"
        parts = [f"Optical scale: {scale}"]
        if context:
            parts.append(context)
        if note:
            parts.append(note)
        return "   |   ".join(parts)
