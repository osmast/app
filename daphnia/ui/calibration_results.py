"""The Calibrate screen's **results panel**: what the detector found, on any target type.

discorpy answers *"is the target being seen?"* by plotting the reference points it extracted
over the image (its ``losa.save_plot_image`` / ``save_residual_plot``). This dialog is the
same idea, in the app: the photo with the detection drawn on it -

* the **reference points** the estimator actually measured from (rings), for a dot target or a
  checkerboard;
* the **lattice** it measured (amber lines, at the period and tilt it reported);
* the **field stop** it confined itself to (dashed ring);

- and beside it the numbers, with the **binarisation level** adjustable so the operator can see
  a bad Otsu split and fix it without leaving the panel. Switching the *pattern* re-runs the
  other detector, so the three target types can be compared on the same photo.

Nothing here measures anything: it calls the page's own ``measure`` / ``points`` callables, so
the panel draws and reports exactly what the Calibrate screen would store - one owner for the
mapping and for the estimator (`mlContext.md` law 1).

PySide6 and OpenCV only; the plotting function is pure NumPy/OpenCV so it can be tested with
no window at all.
"""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: BGR colours, kept beside the drawing they belong to. Amber is the lattice (the same amber
#: `ImageCanvas` draws the measured grid in - one thing, one colour across the app); the
#: reference points are a colour of their own so "what it measured from" cannot be confused
#: with "what it measured".
LATTICE_COLOR = (0, 191, 255)
POINT_COLOR = (60, 60, 220)
FIELD_STOP_COLOR = (0, 191, 255)

#: A reference point's ring, in pixels of the *photo*. Small, so a fine print's points do not
#: merge into a blob on screen.
POINT_RADIUS_PX = 4

#: How wide the photo is drawn in the panel. Fixed, so switching pattern does not resize the
#: window under the operator's cursor.
PANEL_IMAGE_WIDTH = 720


def _as_bgr(frame):
    """A 3-channel BGR view of a frame (a copy), or ``None`` when there is nothing to draw."""
    if frame is None or getattr(frame, "ndim", 0) < 2:
        return None
    import cv2

    image = frame[:, :, :3] if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image).copy()


def _draw_lattice(image, estimate):
    """Draw the measured lattice across the image, at its period and tilt."""
    import cv2

    period = float(getattr(estimate, "period_px", 0.0) or 0.0)
    if period <= 0:
        return
    angle = np.radians(float(getattr(estimate, "angle_deg", 0.0) or 0.0))
    normal = (float(np.cos(angle)), float(np.sin(angle)))
    direction = (-normal[1], normal[0])
    height, width = image.shape[:2]
    centre_x, centre_y = width / 2.0, height / 2.0
    reach = float(max(width, height))
    for index in range(-(int(reach / period) + 2), int(reach / period) + 3):
        offset = index * period
        base_x = centre_x + normal[0] * offset
        base_y = centre_y + normal[1] * offset
        first = (int(round(base_x - direction[0] * reach)),
                 int(round(base_y - direction[1] * reach)))
        second = (int(round(base_x + direction[0] * reach)),
                  int(round(base_y + direction[1] * reach)))
        cv2.line(image, first, second, LATTICE_COLOR, 1, cv2.LINE_AA)


def plot_detection(frame, points=None, estimate=None):
    """The photo with its detection drawn: ``(N, 2)`` source-pixel points, or ``None``.

    ``points`` are the reference points to ring (``None`` or empty for a line target, which has
    no points); ``estimate`` supplies the lattice and the field stop. Returns a BGR ``uint8``
    image, or ``None`` when there is no frame to draw - a caller shows "no photo" rather than a
    blank rectangle.
    """
    import cv2

    image = _as_bgr(frame)
    if image is None:
        return None
    if estimate is not None:
        stop = getattr(estimate, "field_stop", None)
        if stop is not None:
            centre = (int(round(stop.centre_x)), int(round(stop.centre_y)))
            cv2.circle(image, centre, int(round(stop.radius)), FIELD_STOP_COLOR, 1,
                       cv2.LINE_AA)
        _draw_lattice(image, estimate)
    if points is not None:
        for x, y in np.asarray(points, dtype=np.float64).reshape(-1, 2):
            centre = (int(round(float(x))), int(round(float(y))))
            cv2.circle(image, centre, POINT_RADIUS_PX, POINT_COLOR, 1, cv2.LINE_AA)
            cv2.circle(image, centre, 1, POINT_COLOR, -1, cv2.LINE_AA)
    return image


def summarise(estimate, pitch_um, threshold, reason="") -> str:
    """The panel's readout: what was measured, how sure, and on what cut.

    A **refusal** is a result, so it is reported like one (``reason``), never as an empty
    panel - the same rule the Calibrate screen follows.
    """
    if estimate is None:
        return ("<b>No grid measured.</b><br>" + reason) if reason \
            else "<b>No grid measured.</b>"
    scale = estimate.px_per_um()
    lines = [
        f"<b>{estimate.period_px:.3f} px</b> per printed cell "
        f"({pitch_um / 1000.0:g} mm)",
        f"S = <b>{scale * 1000.0:.1f} px/mm</b> ({scale:.6f} px/um)",
        f"{estimate.reading_count} {estimate.reading_label} · tilt "
        f"{estimate.angle_deg:+.1f}&deg;",
    ]
    if getattr(estimate, "spread_pct", 0.0):
        lines.append(f"drift &plusmn; {estimate.spread_pct:.1f} % across the field")
    stop = getattr(estimate, "field_stop", None)
    lines.append(f"field stop: paper circle r {stop.radius:.0f} px" if stop is not None
                 else "field stop: none - the whole frame is paper")
    if getattr(estimate, "coverage", 1.0) < 1.0:
        lines.append(f"measurable area: {estimate.coverage * 100:.0f} % of the frame")
    lines.append("binarisation: " + ("Otsu (auto)" if threshold is None
                                     else f"grey level {threshold:g} (set by hand)"))
    lines.extend(estimate.caveats)
    return "<br>".join(lines)


class CalibrationResultsDialog(QDialog):
    """What the detector found on this photo, of any target type - and the cut it used.

    It measures nothing itself: ``measure(frame, pattern, threshold)`` and
    ``points(frame, pattern, threshold)`` are the Calibrate page's own callables, so this panel
    and that screen can never disagree about what the detector saw. Changing the pattern or the
    binarisation level re-runs and redraws at once - which is what makes "adjust it in the
    frontend" a two-second act rather than another photo.
    """

    def __init__(self, frame, patterns, pitch_um, measure, points, pattern=None,
                 threshold=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration results")
        self.setModal(True)
        self.resize(1180, 660)

        self.frame = frame
        self.pitch_um = float(pitch_um)
        self.measure = measure
        self.points_for = points

        layout = QHBoxLayout(self)

        # -- the drawing ----------------------------------------------------
        left = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setMinimumSize(PANEL_IMAGE_WIDTH, 360)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setWordWrap(True)
        left.addWidget(self.image_label, stretch=1)
        legend = QLabel(
            "Rings: the reference points measured from · amber lines: the lattice it "
            "measured · amber ring: the field stop. This panel reads the photo on panel 2 - "
            "open one there first.")
        legend.setWordWrap(True)
        left.addWidget(legend)
        layout.addLayout(left, stretch=3)

        # -- the controls and the readout -----------------------------------
        right = QVBoxLayout()
        self.combo_pattern = QComboBox()
        for label, value in patterns:
            self.combo_pattern.addItem(label, value)
        right.addWidget(QLabel("Printed pattern"))
        right.addWidget(self.combo_pattern)

        self.check_threshold = QCheckBox("Binarisation: set by hand")
        self.check_threshold.setToolTip(
            "Off: the detector chooses the ink/paper cut with Otsu's method. On: it cuts at "
            "the level below - the escape hatch for a photo where Otsu's split lands wrongly. "
            "It does not cure the eyepiece's vignette: the paper's brightness varies across "
            "the field, so no single level is right everywhere.")
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 255.0)
        self.spin_threshold.setDecimals(1)
        self.spin_threshold.setValue(128.0)
        self.spin_threshold.setEnabled(False)
        right.addWidget(self.check_threshold)
        right.addWidget(self.spin_threshold)

        refresh = QPushButton("Re-measure")
        refresh.clicked.connect(self.refresh)
        right.addWidget(refresh)

        self.readout = QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(Qt.RichText)
        right.addWidget(self.readout)
        right.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        right.addWidget(close)
        layout.addLayout(right, stretch=2)

        # -- initial state --------------------------------------------------
        index = self.combo_pattern.findData(pattern) if pattern is not None else 0
        self.combo_pattern.setCurrentIndex(index if index >= 0 else 0)
        if threshold is not None:
            self.check_threshold.setChecked(True)
            self.spin_threshold.setValue(float(threshold))
        self.check_threshold.toggled.connect(self._on_threshold_toggled)
        self.spin_threshold.valueChanged.connect(self.refresh)
        self.combo_pattern.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def pattern(self):
        """The printed pattern currently being inspected."""
        return self.combo_pattern.currentData()

    def threshold(self):
        """The manual binarisation level, or ``None`` for "Otsu decides"."""
        return (float(self.spin_threshold.value())
                if self.check_threshold.isChecked() else None)

    def _on_threshold_toggled(self, checked):
        self.spin_threshold.setEnabled(checked)
        self.refresh()

    def refresh(self):
        """Re-measure on the current controls and redraw - the panel's whole behaviour."""
        if self.frame is None:
            self.image_label.setText(
                "No photo yet.\nPanel 2's 'Use live frame' or 'Open photo...' puts one here.")
            self.readout.setText("<b>No photo</b> to inspect.")
            return
        pattern, threshold = self.pattern(), self.threshold()
        reason, estimate, points = "", None, None
        try:
            estimate = self.measure(self.frame, pattern, threshold)
            points = self.points_for(self.frame, pattern, threshold)
        except ValueError as error:               # the estimator's own named refusal
            reason = str(error)
        self._show(plot_detection(self.frame, points=points, estimate=estimate))
        self.readout.setText(summarise(estimate, self.pitch_um, threshold, reason))

    def _show(self, overlay):
        """Draw the overlay in the label, scaled to fit - never stretched out of shape."""
        if overlay is None:
            self.image_label.setText("No photo to draw.")
            return
        from daphnia.ui.widgets.image_canvas import frame_to_qimage

        pixmap = QPixmap.fromImage(frame_to_qimage(overlay))
        target = self.image_label.size()
        if target.width() > 0 and target.height() > 0:
            pixmap = pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)


def summarise_probe(result) -> str:
    """The correction panel's readout: the model, its residual, and whether it is usable."""
    magnitude = result.magnitude_pct
    lines = [
        f"centre of distortion <b>({result.x_center:.1f}, {result.y_center:.1f})</b> px",
        "coefficients " + ", ".join(f"c{i}={value:+.3e}"
                                    for i, value in enumerate(result.coefficients)),
        f"line spacing {result.line_spacing_px[0]:.2f} / {result.line_spacing_px[1]:.2f} px "
        f"&middot; {result.lines[0]} + {result.lines[1]} lines kept "
        f"from {result.points[0]} + {result.points[1]} points",
        f"residual <b>{result.residual_before_px:.3f} px &rarr; "
        f"{result.residual_after_px:.3f} px</b> rms, worst line "
        f"{result.worst_line_px:.3f} px",
        "discorpy's own check: " + ("<b>sub-pixel</b> - it passes" if result.subpixel
                                    else "<b>FAILED</b> - not good enough to store"),
    ]
    if magnitude is not None:
        lines.append(f"magnification change at the frame edge: {magnitude:+.1f} %")
    lines.extend(f"<i>{note}</i>" for note in result.notes)
    return "<br>".join(lines)


class DistortionResultsDialog(QDialog):
    """Every figure of discorpy's calibration flow, **before and after** the correction.

    The step is slow (a background FFT, two profile searches, the grouping and the parabola
    regeneration), so it runs on a button rather than on every control change. A failure - no
    discorpy, no line family, a model that made the frame worse - is said in words; an empty
    panel would be blamed on the photo, which is the mistake the probe exists to prevent.
    """

    #: How wide each figure is drawn. Fixed, so the grid does not reflow as figures arrive.
    FIGURE_WIDTH = 420

    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Distortion probe (discorpy)")
        self.setModal(True)
        self.resize(980, 720)

        from PySide6.QtWidgets import QGridLayout, QScrollArea

        self.frame = frame
        layout = QVBoxLayout(self)
        explain = QLabel(
            "discorpy's calibration flow on the photo in panel 2, in the order of its own "
            "documentation (3.3.3 dot pattern, 3.3.4 line pattern): background correction, "
            "binarisation, the cross points, the lines they group into, the residual "
            "<b>before</b> the model, the corrected lines, the residual <b>after</b>, the "
            "unwarped frame and the difference. Nothing is stored and no frame is rewritten - "
            "the model is <i>not</i> storable yet (SR-OPTICS-032/033).")
        explain.setWordWrap(True)
        layout.addWidget(explain)

        row = QHBoxLayout()
        self.run_button = QPushButton("Run the probe")
        self.run_button.clicked.connect(self.run_probe)
        row.addWidget(self.run_button)
        self.status = QLabel("Not run yet.")
        self.status.setWordWrap(True)
        row.addWidget(self.status, stretch=1)
        layout.addLayout(row)

        self.readout = QLabel("")
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(Qt.RichText)
        layout.addWidget(self.readout)

        area = QScrollArea()
        area.setWidgetResizable(True)
        holder = QWidget()
        self.grid = QGridLayout(holder)
        area.setWidget(holder)
        layout.addWidget(area, stretch=1)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close, alignment=Qt.AlignRight)

        if frame is None:
            self.run_button.setEnabled(False)
            self.status.setText("No photo on panel 2 to probe.")

    def run_probe(self):
        """Run the flow once, then fill the grid - the panel's one action."""
        from PySide6.QtWidgets import QApplication

        from daphnia.vision.distortion import FIGURE_CAPTIONS, DistortionUnavailable, probe

        self.run_button.setEnabled(False)
        self.status.setText("Running discorpy's flow... (a few seconds)")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            result = probe(self.frame)
        except DistortionUnavailable as error:
            self.status.setText(str(error))
            self.readout.setText("")
            return
        except (ValueError, RuntimeError) as error:       # named, and shown
            self.status.setText(f"The probe could not finish: {error}")
            self.readout.setText("")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.run_button.setEnabled(True)

        self.status.setText("Done - each figure is the flow's own output on this frame, in "
                            "the order it produced them.")
        self.readout.setText(summarise_probe(result))
        for index, (name, caption) in enumerate(FIGURE_CAPTIONS):
            self._place(index, name, caption, result.figures.get(name))

    def _place(self, index, name, caption, image):
        """One figure and its caption, into the grid - left column first."""
        from daphnia.ui.widgets.image_canvas import frame_to_qimage

        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        picture = QLabel()
        picture.setAlignment(Qt.AlignCenter)
        if image is not None:
            pixmap = QPixmap.fromImage(frame_to_qimage(image))
            picture.setPixmap(pixmap.scaledToWidth(self.FIGURE_WIDTH, Qt.SmoothTransformation))
        else:
            picture.setText(f"{name}: not produced")
        column.addWidget(picture)
        title = QLabel(f"<b>{name}</b> - {caption}")
        title.setWordWrap(True)
        column.addWidget(title)
        self.grid.addWidget(holder, index // 2, index % 2)
