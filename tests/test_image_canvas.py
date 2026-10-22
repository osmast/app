"""Zoomable image canvas: zoom/pan keeps clicks mapped to image pixels."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter
from PySide6.QtWidgets import QApplication

from daphnia.ui import theme
from daphnia.ui.widgets import ImageCanvas, specimen_color
from daphnia.ui.widgets.image_canvas import (FIELD_STOP_COLOR, SPECIMEN_COLORS, TOOL_BOX,
                                             TOOL_KEYPOINTS, TOOL_LINE, TOOL_SELECT,
                                             TOOLS, box_at, moved_box, resize_handle,
                                             resized_box)

#: A 200 x 100 image; the widget centre always sits on the image centre (100, 50).
IMAGE = np.zeros((100, 200, 3), dtype=np.uint8)


class ZoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        return canvas

    def _center(self, canvas):
        return QPointF(canvas.width() / 2, canvas.height() / 2)

    def test_starts_fitted(self):
        self.assertEqual(self._canvas().zoom, ImageCanvas.MIN_ZOOM)

    def test_setting_a_frame_can_keep_the_operators_zoom(self):
        """A live view must not re-fit on every frame (the Predict surface)."""
        canvas = self._canvas()
        canvas.set_zoom(3.0)
        canvas.set_source(IMAGE, reset_view=False)
        self.assertAlmostEqual(canvas.zoom, 3.0)
        canvas.set_source(IMAGE)
        self.assertEqual(canvas.zoom, ImageCanvas.MIN_ZOOM)

    def test_zoom_in_magnifies_about_the_centre(self):
        canvas = self._canvas()
        canvas.zoom_in()
        self.assertGreater(canvas.zoom, 1.0)
        point = canvas._to_image(self._center(canvas))
        self.assertAlmostEqual(point[0], 100.0, delta=1.0)
        self.assertAlmostEqual(point[1], 50.0, delta=1.0)

    def test_clicks_stay_accurate_after_zooming(self):
        canvas = self._canvas()
        canvas.set_zoom(3.0)
        center = self._center(canvas)
        event = QMouseEvent(QEvent.MouseButtonPress, center, center,
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        canvas.mousePressEvent(event)
        self.assertEqual(len(canvas.points), 1)
        self.assertAlmostEqual(canvas.points[0][0], 100.0, delta=1.0)
        self.assertAlmostEqual(canvas.points[0][1], 50.0, delta=1.0)

    def test_reset_view_returns_to_fit(self):
        canvas = self._canvas()
        canvas.zoom_in()
        canvas.zoom_in()
        canvas.reset_view()
        self.assertEqual(canvas.zoom, ImageCanvas.MIN_ZOOM)

    def test_the_zoom_signal_reports_fit_and_the_factor(self):
        """The on-canvas View cluster *shows* the magnification from this signal."""
        canvas = self._canvas()
        seen = []
        canvas.zoom_changed.connect(seen.append)
        canvas.zoom_in()
        canvas.reset_view()
        self.assertEqual(seen, [ImageCanvas.ZOOM_STEP, ImageCanvas.MIN_ZOOM])

    def test_zoom_is_bounded(self):
        canvas = self._canvas()
        canvas.set_zoom(10_000.0)
        self.assertEqual(canvas.zoom, ImageCanvas.MAX_ZOOM)
        canvas.set_zoom(0.01)
        self.assertEqual(canvas.zoom, ImageCanvas.MIN_ZOOM)


class _StubGesture:
    """Minimal stand-in for a QNativeGestureEvent (its ctor is not public)."""

    def __init__(self, kind, value=0.0, position=None, delta=None):
        self._kind, self._value, self._position, self._delta = \
            kind, value, position, delta

    def gestureType(self):
        return self._kind

    def value(self):
        return self._value

    def position(self):
        return self._position

    def delta(self):
        return self._delta


class GestureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        return canvas

    def test_two_finger_pinch_zooms(self):
        canvas = self._canvas()
        center = QPointF(canvas.width() / 2, canvas.height() / 2)
        handled = canvas._native_gesture(
            _StubGesture(Qt.ZoomNativeGesture, value=0.5, position=center))
        self.assertTrue(handled)
        self.assertGreater(canvas.zoom, 1.0)

    def test_other_gestures_are_not_consumed(self):
        canvas = self._canvas()
        self.assertFalse(canvas._native_gesture(_StubGesture(Qt.TapGesture)))


class LatticeOverlayTests(unittest.TestCase):
    """The measured grid is drawn on the canvas, so detection is visible."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        return canvas

    def test_there_is_no_overlay_before_a_grid_is_measured(self):
        self.assertIsNone(self._canvas().lattice)

    def test_the_measured_grid_is_kept_for_drawing(self):
        canvas = self._canvas()
        canvas.set_lattice(147.3, 8.0)
        self.assertEqual(canvas.lattice, (147.3, 8.0))

    def test_a_new_frame_clears_the_overlay(self):
        """A stale lattice over a new photo would be a lie about what was measured."""
        canvas = self._canvas()
        canvas.set_lattice(147.3, 0.0)
        canvas.set_source(IMAGE)
        self.assertIsNone(canvas.lattice)

    def test_clearing_the_points_also_clears_the_overlay(self):
        canvas = self._canvas()
        canvas.set_lattice(147.3, 0.0)
        canvas.clear_points()
        self.assertIsNone(canvas.lattice)

    def test_a_period_of_zero_is_not_an_overlay(self):
        canvas = self._canvas()
        canvas.set_lattice(0.0)
        self.assertIsNone(canvas.lattice)

    def test_painting_with_an_overlay_does_not_fail(self):
        """The paint path is the part that has to work on screen."""
        canvas = self._canvas()
        canvas.set_lattice(37.5, 12.0)
        canvas.repaint()
        self.assertEqual(canvas.lattice, (37.5, 12.0))

    def test_painting_without_a_frame_is_still_safe(self):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_lattice(40.0, 0.0)
        canvas.repaint()
        self.assertIsNotNone(canvas)


class FieldStopOverlayTests(unittest.TestCase):
    """The paper's circle is drawn beside the lattice: *where* the scale was measurable.

    Through an eyepiece the paper is a disc in a black surround, so the lattice alone says
    what was measured and not where it was possible to measure it (`SR-OPTICS-026`). The
    overlay has the lattice's lifecycle exactly - a photo it did not come from must never
    carry it.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        return canvas

    def test_there_is_no_circle_before_one_is_measured(self):
        canvas = self._canvas()
        self.assertIsNone(canvas.field_stop)
        self.assertIsNone(canvas._field_stop)

    def test_the_circle_is_kept_for_drawing(self):
        canvas = self._canvas()
        canvas.set_field_stop(101.0, 202.0, 303.0)
        self.assertEqual(canvas.field_stop, (101.0, 202.0, 303.0))

    def test_a_new_frame_clears_the_circle(self):
        """Same rule as the lattice: it can never describe a photo it did not come from."""
        canvas = self._canvas()
        canvas.set_field_stop(101.0, 202.0, 303.0)
        canvas.set_source(IMAGE)
        self.assertIsNone(canvas.field_stop)

    def test_clearing_the_points_clears_the_circle_too(self):
        canvas = self._canvas()
        canvas.set_field_stop(101.0, 202.0, 303.0)
        canvas.clear_points()
        self.assertIsNone(canvas.field_stop)

    def test_a_radius_of_zero_is_not_a_circle(self):
        """A sheet photographed in air has no field stop - `None` is a real answer."""
        canvas = self._canvas()
        canvas.set_field_stop(10.0, 10.0, 0.0)
        self.assertIsNone(canvas.field_stop)
        canvas.set_field_stop(10.0, 10.0, 5.0)
        canvas.set_field_stop(None)
        self.assertIsNone(canvas.field_stop)

    def test_painting_the_circle_marks_the_canvas(self):
        """Counted in pixels: a circle that was measured but drawn nowhere is the failure.

        The canvas shows a 200x100 frame, so the ring is magnified - and _display_rect is
        what says where. Counting is the only way to test what was actually *reported*,
        which is the same rule the prediction overlay is checked by.
        """
        canvas = self._canvas()
        canvas.set_field_stop(100.0, 50.0, 30.0)
        canvas.repaint()
        image = canvas.grab().toImage()
        colour = QColor(FIELD_STOP_COLOR)
        rect = canvas._display_rect()
        scale = rect.width() / IMAGE.shape[1]
        centre = (rect.x() + 100.0 * scale, rect.y() + 50.0 * scale)
        radius = 30.0 * scale
        hit = 0
        for y in range(max(0, int(centre[1] - radius - 1)),
                       min(image.height(), int(centre[1] + radius + 2))):
            for x in range(max(0, int(centre[0] - radius - 1)),
                           min(image.width(), int(centre[0] + radius + 2))):
                pixel = image.pixelColor(x, y)
                near_ring = abs(((x - centre[0]) ** 2 + (y - centre[1]) ** 2) ** 0.5
                                - radius) <= 1.5
                if near_ring and abs(pixel.red() - colour.red()) < 40 and \
                        abs(pixel.green() - colour.green()) < 40 and \
                        abs(pixel.blue() - colour.blue()) < 40:
                    hit += 1
        self.assertGreater(hit, 0, "the measured circle is not on the canvas")


class _StubWheel:
    """Minimal stand-in for a QWheelEvent, whose constructor is awkward to fake."""

    def __init__(self, position, angle=0, pixel=0, angle_x=0, pixel_x=0,
                 modifiers=Qt.NoModifier):
        self._position = position
        self._angle = QPoint(angle_x, angle)
        self._pixel = QPoint(pixel_x, pixel)
        self._modifiers = modifiers

    def angleDelta(self):
        return self._angle

    def pixelDelta(self):
        return self._pixel

    def position(self):
        return self._position

    def modifiers(self):
        return self._modifiers

    def accept(self):
        pass


class InteractionTests(unittest.TestCase):
    """Zooming and panning that do not fight the operator."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        return canvas

    def _press(self, canvas, point, modifiers=Qt.NoModifier):
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, point, point, Qt.LeftButton,
            Qt.LeftButton, modifiers))

    def test_scroll_pans_and_never_zooms(self):
        """Zoom has its own gestures; scrolling must not change the magnification.

        An accidental zoom while clicking is an *error in the ground truth*, not an
        inconvenience, so a vertical two-finger scroll pans and leaves the zoom alone.
        """
        canvas = self._canvas()
        canvas.set_zoom(3.0)
        zoom_before = canvas.zoom
        pan_before = canvas._pan.y()
        canvas.wheelEvent(_StubWheel(QPointF(300, 200), angle=120))
        self.assertAlmostEqual(canvas.zoom, zoom_before)     # no zoom, ever
        self.assertNotEqual(canvas._pan.y(), pan_before)     # the view moved instead

    def test_command_scroll_does_not_zoom_either(self):
        """⌘ + scroll used to zoom; pinch, the buttons and + / - / F do now."""
        canvas = self._canvas()
        canvas.set_zoom(3.0)
        zoom_before = canvas.zoom
        canvas.wheelEvent(_StubWheel(QPointF(300, 200), angle=120,
                                     modifiers=Qt.ControlModifier))
        self.assertAlmostEqual(canvas.zoom, zoom_before)

    def test_a_pixel_delta_pans_in_both_axes(self):
        """A trackpad's small pixel deltas move the view, not the zoom."""
        canvas = self._canvas()
        canvas.set_zoom(3.0)
        canvas.wheelEvent(_StubWheel(QPointF(300, 200), pixel=30, pixel_x=20))
        self.assertNotEqual(canvas._pan.x(), 0.0)
        self.assertNotEqual(canvas._pan.y(), 0.0)

    def test_a_wheel_notch_pans_a_whole_step(self):
        """A wheel mouse sends angle notches; they are normalised into pixels."""
        canvas = self._canvas()
        canvas.set_zoom(3.0)
        canvas.wheelEvent(_StubWheel(QPointF(300, 200), angle=ImageCanvas.SCROLL_REFERENCE))
        self.assertAlmostEqual(canvas._pan.y(), ImageCanvas.WHEEL_PAN_STEP)

    def test_panning_only_happens_while_the_image_is_magnified(self):
        """There is nothing to pan to at fit zoom, so a scroll must not drift."""
        canvas = self._canvas()                              # starts fitted
        canvas.wheelEvent(_StubWheel(QPointF(300, 200), angle=120))
        self.assertEqual((canvas._pan.x(), canvas._pan.y()), (0.0, 0.0))

    def test_a_horizontal_scroll_pans_instead_of_zooming(self):
        """Two-finger side-to-side must move around, not zoom."""
        canvas = self._canvas()
        canvas.set_zoom(3.0)
        pan_before = canvas._pan.x()
        zoom_before = canvas.zoom
        canvas.wheelEvent(_StubWheel(QPointF(300, 200), angle_x=120))
        self.assertAlmostEqual(canvas.zoom, zoom_before)     # no zoom
        self.assertNotEqual(canvas._pan.x(), pan_before)     # the view moved

    def test_a_horizontal_pixel_delta_pans_too(self):
        canvas = self._canvas()
        canvas.set_zoom(3.0)
        canvas.wheelEvent(_StubWheel(QPointF(300, 200), pixel_x=40))
        self.assertNotEqual(canvas._pan.x(), 0.0)

    def test_annotated_specimens_can_be_numbered(self):
        canvas = self._canvas()
        self.assertTrue(canvas.show_annotation_index)
        canvas.set_annotations([(2, [((1.0, 2.0), (3.0, 4.0))])])
        canvas.repaint()
        self.assertEqual(canvas.annotations[0][0], 2)

    def test_command_drag_pans_without_placing_a_point(self):
        """The missing interaction: move the view without dropping a landmark."""
        canvas = self._canvas()
        canvas.set_zoom(3.0)
        center = QPointF(canvas.width() / 2, canvas.height() / 2)
        self._press(canvas, center, Qt.ControlModifier)
        self.assertEqual(canvas.points, [])                  # no landmark placed
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, center + QPointF(20, 0), center,
            Qt.NoButton, Qt.LeftButton, Qt.ControlModifier))
        self.assertNotEqual(canvas._pan.x(), 0.0)            # the view moved

    def test_pan_mode_left_drag_does_not_place_a_point(self):
        canvas = self._canvas()
        canvas.pan_mode = True
        center = QPointF(canvas.width() / 2, canvas.height() / 2)
        self._press(canvas, center)
        self.assertEqual(canvas.points, [])

    def test_a_plain_click_still_places_a_point(self):
        canvas = self._canvas()
        center = QPointF(canvas.width() / 2, canvas.height() / 2)
        self._press(canvas, center)
        self.assertEqual(len(canvas.points), 1)

    def test_annotations_are_kept_for_drawing(self):
        canvas = self._canvas()
        canvas.set_annotations([(1, [((1.0, 2.0), (3.0, 4.0))]),
                                {"index": 2,
                                 "segments": [((5.0, 6.0), (7.0, 8.0))]}])
        self.assertEqual([index for index, _ in canvas.annotations], [1, 2])
        canvas.set_annotations(None)
        self.assertEqual(canvas.annotations, [])

    def test_every_segment_can_carry_its_own_label(self):
        """A parallel list of labels, one per segment - the Bench's per-line naming.

        With it, the canvas draws each label above *its own* line instead of one sentence for
        the whole item at its first point; without it (Predict's case) the item's own ``text``
        is still drawn at the anchor, so nothing that worked stops working.
        """
        canvas = self._canvas()
        canvas.set_annotations([
            (1, [((10.0, 10.0), (30.0, 10.0)), ((10.0, 30.0), (30.0, 30.0))],
             "", ["length", "length"], ["1.1  40.0 um", "Gut: 55.0 um"]),
            (2, [((50.0, 10.0), (70.0, 10.0))], "2: 60.0 um", ["length"]),
        ])
        self.assertEqual(canvas.annotation_texts,
                         [(1, ["1.1  40.0 um", "Gut: 55.0 um"]), (2, [""])])
        self.assertEqual(canvas.annotations_with_text[1][2], "2: 60.0 um",
                         "an item's own label is still kept")
        canvas.repaint()                       # both paths paint without failing

    def test_a_per_segment_label_paints_where_the_line_is(self):
        """The label sits **above its own segment**, not collected at the first point."""
        canvas = self._canvas()
        canvas.resize(400, 300)
        canvas.set_annotations([
            (1, [((10.0, 10.0), (30.0, 10.0)), ((10.0, 60.0), (30.0, 60.0))],
             "", ["length", "length"], ["first", "second"]),
        ])
        canvas.repaint()
        painted = canvas.grab().toImage()
        first = self._label_plate(painted, canvas._to_screen((20.0, 10.0)))
        second = self._label_plate(painted, canvas._to_screen((20.0, 60.0)))
        self.assertIsNotNone(first, "the first line carries its own label")
        self.assertIsNotNone(second, "and so does the second, 50 px lower")
        self.assertNotEqual(first, second, "two labels, each above its own line")

    @staticmethod
    def _label_plate(image, point):
        """The **y** of the dark plate drawn above ``point``, or ``None`` when there is none.

        A label is drawn on a translucent black plate (`_paint_annotation_label`), which is
        what tells a label apart from the picture behind it: the test looks for it where the
        eye does, a few pixels above the anchor.
        """
        if point is None:                       # pragma: no cover - a canvas that lost the point
            return None
        x, y = int(point[0]), int(point[1])
        for offset in range(4, 30):             # the plate sits above the anchor
            if not (0 <= y - offset < image.height() and 0 <= x < image.width()):
                continue
            pixel = image.pixelColor(x, y - offset)
            if pixel.alpha() == 255 and pixel.red() < 60 and pixel.green() < 60:
                return y - offset
        return None

    def test_painting_overlays_does_not_fail(self):
        canvas = self._canvas()
        canvas.set_annotations([(1, [((1.0, 2.0), (3.0, 4.0))])])
        canvas.repaint()

    def test_points_are_connected_only_when_asked(self):
        """A landmark pass draws many same-landmark points - they must not link up."""
        canvas = self._canvas()
        self.assertTrue(canvas.connect_points)              # two points = one interval
        canvas.connect_points = False
        self.assertFalse(canvas.connect_points)

    def test_specimen_colours_are_stable_and_cycle(self):
        self.assertEqual(specimen_color(1), specimen_color(1))
        self.assertEqual(specimen_color(1),
                         specimen_color(1 + len(SPECIMEN_COLORS)))
        self.assertNotEqual(specimen_color(1), specimen_color(2))


class DoubleClickTests(unittest.TestCase):
    """A double click is a gesture, not two clicks: one point, reported once."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        return canvas

    def _center(self, canvas):
        return QPointF(canvas.width() / 2, canvas.height() / 2)

    def test_a_double_click_reports_the_gesture_and_places_no_second_point(self):
        """Qt sends the double click *instead of* a second press: one point, not two."""
        canvas = self._canvas()
        seen = []
        canvas.double_clicked.connect(lambda: seen.append(True))
        center = self._center(canvas)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, center, center, Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        canvas.mouseDoubleClickEvent(QMouseEvent(
            QEvent.MouseButtonDblClick, center, center, Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, center, center, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))
        self.assertEqual(len(canvas.points), 1)          # only the first press
        self.assertEqual(len(seen), 1)                   # the gesture was reported

    def test_a_double_click_aborts_a_box_preview(self):
        """The double click's first press opens a box; the gesture is not a box."""
        canvas = self._canvas()
        seen = []
        canvas.box_changed.connect(seen.append)
        center = self._center(canvas)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, center, center, Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        canvas.mouseDoubleClickEvent(QMouseEvent(
            QEvent.MouseButtonDblClick, center, center, Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, center + QPointF(1, 1), center,
            Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
        self.assertIsNone(canvas.box)
        self.assertTrue(all(box == [] for box in seen))  # nothing was reported as a box

    def test_a_double_click_while_panning_is_ignored(self):
        canvas = self._canvas()
        seen = []
        canvas.double_clicked.connect(lambda: seen.append(True))
        center = self._center(canvas)
        canvas.mouseDoubleClickEvent(QMouseEvent(
            QEvent.MouseButtonDblClick, center, center, Qt.LeftButton,
            Qt.LeftButton, Qt.ControlModifier))          # ⌘-drag is a pan
        self.assertEqual(seen, [])

    def test_a_double_click_without_an_image_is_ignored(self):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        seen = []
        canvas.double_clicked.connect(lambda: seen.append(True))
        canvas.mouseDoubleClickEvent(QMouseEvent(
            QEvent.MouseButtonDblClick, QPointF(10, 10), QPointF(10, 10),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(seen, [])


class BoxTests(unittest.TestCase):
    """Drawing a bounding box: a drag, not a click - and never a stray point."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        canvas.box_mode = True
        return canvas

    def _center(self, canvas):
        return QPointF(canvas.width() / 2, canvas.height() / 2)

    def _click(self, canvas, point):
        """A press-and-release at one spot: a click, not a drag."""
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, point, point, Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, point, point, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def _drag(self, canvas, start, end):
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, start, Qt.NoButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, start, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    def test_a_second_click_finishes_the_box(self):
        """Click a corner, then click the opposite one - no dragging needed."""
        canvas = self._canvas()
        seen = []
        canvas.box_changed.connect(seen.append)
        center = self._center(canvas)
        self._click(canvas, center + QPointF(-60, -30))
        self.assertTrue(canvas.box_pending)             # one corner is anchored
        self.assertTrue(seen == [] or seen == [[]])     # a corner is not a box
        self._click(canvas, center + QPointF(60, 30))   # the opposite corner
        self.assertFalse(canvas.box_pending)
        self.assertEqual(seen[-1], [(80.0, 40.0, 120.0, 60.0)])
        self.assertEqual(canvas.box, (80.0, 40.0, 120.0, 60.0))

    def test_the_anchor_is_where_the_first_click_was(self):
        canvas = self._canvas()
        center = self._center(canvas)
        self._click(canvas, center)
        self.assertAlmostEqual(canvas.box_origin[0], 100.0, delta=2.0)
        self.assertAlmostEqual(canvas.box_origin[1], 50.0, delta=2.0)

    def test_a_drag_from_the_anchored_corner_still_finishes_the_box(self):
        """The two ways mix: click a corner, then drag the opposite one out."""
        canvas = self._canvas()
        center = self._center(canvas)
        self._click(canvas, center + QPointF(-60, -30))
        self.assertTrue(canvas.box_pending)
        self._drag(canvas, center + QPointF(60, 30), center + QPointF(80, 40))
        self.assertFalse(canvas.box_pending)
        x1, y1, x2, y2 = canvas.box
        self.assertAlmostEqual((x1, y1), (80.0, 40.0), places=3)   # the click's corner
        self.assertAlmostEqual(x2, 126.666, places=2)              # the drag's corner
        self.assertAlmostEqual(y2, 63.333, places=2)

    def test_two_clicks_on_the_same_spot_do_not_make_a_box(self):
        """A twitch is still a twitch: the corner simply stays anchored."""
        canvas = self._canvas()
        seen = []
        canvas.box_changed.connect(seen.append)
        center = self._center(canvas)
        self._click(canvas, center)
        self._click(canvas, center + QPointF(1, 1))
        self.assertTrue(canvas.box_pending)
        x1, y1, x2, y2 = canvas.box
        self.assertLess(x2 - x1, ImageCanvas.MIN_BOX_PIXELS)
        self.assertLess(y2 - y1, ImageCanvas.MIN_BOX_PIXELS)
        self.assertTrue(all(box == [] for box in seen))

    def test_escape_abandons_a_half_drawn_box(self):
        canvas = self._canvas()
        center = self._center(canvas)
        self._click(canvas, center)
        self.assertTrue(canvas.box_pending)
        canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        self.assertFalse(canvas.box_pending)
        self.assertIsNone(canvas.box)

    def test_a_right_click_abandons_a_half_drawn_box(self):
        canvas = self._canvas()
        center = self._center(canvas)
        self._click(canvas, center)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, center, center, Qt.RightButton,
            Qt.RightButton, Qt.NoModifier))
        self.assertFalse(canvas.box_pending)

    def test_anchoring_reports_the_state_once_per_change(self):
        """The page hears "corner set / corner released", never "box stored"."""
        canvas = self._canvas()
        states = []
        canvas.box_anchor_changed.connect(states.append)
        center = self._center(canvas)
        self._click(canvas, center)
        self._click(canvas, center + QPointF(80, 40))
        self.assertEqual(states, [True, False])

    def test_a_drag_stores_a_box_in_image_pixels(self):
        canvas = self._canvas()
        seen = []
        canvas.box_changed.connect(seen.append)
        center = self._center(canvas)
        self._drag(canvas, center + QPointF(-60, -30), center + QPointF(60, 30))
        self.assertTrue(seen)                      # the box was reported
        x1, y1, x2, y2 = canvas.box
        self.assertAlmostEqual((x1 + x2) / 2, 100.0, delta=2.0)
        self.assertAlmostEqual((y1 + y2) / 2, 50.0, delta=2.0)
        self.assertGreater(x2 - x1, y2 - y1)       # wider than tall, as dragged

    def test_the_corners_are_normalised_whatever_the_drag_direction(self):
        canvas = self._canvas()
        center = self._center(canvas)
        self._drag(canvas, center + QPointF(60, 30), center + QPointF(-60, -30))
        x1, y1, x2, y2 = canvas.box
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)

    def test_a_drag_places_no_landmark(self):
        canvas = self._canvas()
        center = self._center(canvas)
        self._drag(canvas, center, center + QPointF(80, 40))
        self.assertEqual(canvas.points, [])

    def test_a_click_is_not_a_box(self):
        """A stray click must not store an unusable specimen: it anchors a corner.

        Superseded behaviour: a click used to leave nothing at all. Now it starts a
        two-click box, so the guarantee is that a *click* is never reported as a box -
        the corner simply waits for the opposite click (or an Escape).
        """
        canvas = self._canvas()
        seen = []
        canvas.box_changed.connect(seen.append)
        center = self._center(canvas)
        self._drag(canvas, center, center + QPointF(1, 1))
        self.assertTrue(canvas.box_pending)             # anchored, not stored
        x1, y1, x2, y2 = canvas.box
        self.assertLess(x2 - x1, ImageCanvas.MIN_BOX_PIXELS)
        self.assertEqual(seen, [])                      # nothing was reported at all

    def test_box_mode_off_places_points_again(self):
        canvas = self._canvas()
        canvas.box_mode = False
        center = self._center(canvas)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, center, center, Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(len(canvas.points), 1)

    def test_a_new_frame_clears_the_box(self):
        canvas = self._canvas()
        canvas.set_box((1.0, 2.0, 3.0, 4.0))
        canvas.set_source(IMAGE)
        self.assertIsNone(canvas.box)

    def test_a_box_dragged_past_the_image_is_pulled_back_onto_it(self):
        """A corner outside the frame cannot be stored: 11 boxes in the real corpus
        were, because the drag crossed the letterbox."""
        canvas = self._canvas()
        seen = []
        canvas.box_changed.connect(seen.append)
        center = self._center(canvas)
        # Well outside the 200 x 100 image, in every direction.
        self._drag(canvas, center + QPointF(-400, -300), center + QPointF(400, 300))
        x1, y1, x2, y2 = canvas.box
        self.assertGreaterEqual(x1, 0.0)
        self.assertGreaterEqual(y1, 0.0)
        self.assertLessEqual(x2, float(IMAGE.shape[1]))
        self.assertLessEqual(y2, float(IMAGE.shape[0]))
        self.assertEqual(seen[-1], [(x1, y1, x2, y2)])

    def test_a_click_outside_the_image_anchors_the_nearest_corner(self):
        """Click-click is the same gesture as a drag, so it is clamped the same."""
        canvas = self._canvas()
        center = self._center(canvas)
        self._click(canvas, center + QPointF(-400, -300))
        origin_x, origin_y = canvas.box_origin
        self.assertGreaterEqual(origin_x, 0.0)
        self.assertGreaterEqual(origin_y, 0.0)

    def test_a_stored_box_can_touch_the_border(self):
        """Clamping must not shrink a specimen that really reaches the edge."""
        canvas = self._canvas()
        center = self._center(canvas)
        self._drag(canvas, center + QPointF(-300, -200), center + QPointF(300, 200))
        x1, y1, x2, y2 = canvas.box
        self.assertAlmostEqual(x1, 0.0)
        self.assertAlmostEqual(y1, 0.0)
        self.assertAlmostEqual(x2, float(IMAGE.shape[1]))
        self.assertAlmostEqual(y2, float(IMAGE.shape[0]))

    def test_an_overlap_with_a_stored_box_is_reported(self):
        """Two boxes on one animal is the mistake a multi-animal frame invites."""
        canvas = self._canvas()
        canvas.set_boxes([(1, (10.0, 10.0, 60.0, 60.0))])
        self.assertEqual(canvas.overlaps_stored((12.0, 12.0, 58.0, 58.0)), 1)
        self.assertIsNone(canvas.overlaps_stored((60.0, 70.0, 90.0, 95.0)))

    def test_the_overlap_threshold_can_be_moved(self):
        canvas = self._canvas()
        canvas.set_boxes([(2, (10.0, 10.0, 60.0, 60.0))])
        # A barely-touching box is 0.1 IoU at most, so only a low bar reports it.
        self.assertIsNone(canvas.overlaps_stored((55.0, 55.0, 90.0, 90.0),
                                                 threshold=0.5))

    def test_the_selection_index_is_drawn_on_a_stored_box(self):
        """Several animals at once need their numbers, or the colours are meaningless."""
        canvas = self._canvas()
        self.assertTrue(canvas.show_annotation_index)


    def test_stored_boxes_are_kept_for_drawing(self):
        canvas = self._canvas()
        canvas.set_boxes([(1, (1.0, 2.0, 3.0, 4.0)),
                          {"index": 2, "box": (5.0, 6.0, 7.0, 8.0)}])
        self.assertEqual([index for index, _ in canvas.boxes], [1, 2])
        canvas.repaint()                       # painting overlays must not fail
        canvas.set_boxes(None)
        self.assertEqual(canvas.boxes, [])

    def test_clearing_the_box_reports_it(self):
        canvas = self._canvas()
        seen = []
        canvas.box_changed.connect(seen.append)
        canvas.set_box((1.0, 2.0, 3.0, 4.0))
        canvas.clear_box()
        self.assertEqual(seen, [[(1.0, 2.0, 3.0, 4.0)], []])
        self.assertEqual(canvas.boxes_drawn, [])


class PaintTests(unittest.TestCase):
    """Painting runs in the test's own frame, so a broken draw fails loudly.

    A Qt paint handler that raises leaves the painter active and takes the process
    down (observed: a `QPoint + QPointF` in the box overlay segfaulted the app while
    every test stayed green, because the error happened inside the event loop rather
    than in the test). These tests open a real painter and call the overlay painters
    directly, so an exception is an ordinary test failure.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        return canvas

    def _paint(self, canvas, method):
        surface = QImage(600, 400, QImage.Format_RGB32)
        painter = QPainter(surface)
        try:
            method(painter, QRect(0, 0, 600, 400))
            painter.end()
        finally:
            if painter.isActive():
                painter.end()

    def test_painting_annotated_specimens_does_not_raise(self):
        canvas = self._canvas()
        canvas.set_annotations([(1, [((1.0, 2.0), (30.0, 40.0))]),
                                (2, [((5.0, 6.0), (7.0, 8.0))])])
        self._paint(canvas, canvas._paint_annotations)

    def test_painting_stored_boxes_does_not_raise(self):
        """The overlay that crashed: a dashed box with its numbered corner label."""
        canvas = self._canvas()
        canvas.set_boxes([(1, (1.0, 2.0, 30.0, 40.0)),
                          (2, (5.0, 6.0, 70.0, 80.0))])
        self.assertTrue(canvas.show_annotation_index)
        self._paint(canvas, canvas._paint_boxes)

    def test_painting_the_box_being_drawn_does_not_raise(self):
        canvas = self._canvas()
        canvas.box_mode = True
        canvas.set_box((1.0, 2.0, 30.0, 40.0))
        self._paint(canvas, canvas._paint_boxes)

    def test_painting_a_pending_box_corner_does_not_raise(self):
        """Half a box is on screen between the two clicks - and it is painted."""
        canvas = self._canvas()
        canvas.box_mode = True
        canvas.anchor_box_corner((10.0, 20.0))
        self.assertTrue(canvas.box_pending)
        self._paint(canvas, canvas._paint_boxes)

    def test_painting_points_and_the_lattice_does_not_raise(self):
        canvas = self._canvas()
        canvas.set_points([(10.0, 10.0), (40.0, 40.0)])
        canvas.set_lattice(10.0, 0.0)
        self._paint(canvas, canvas._paint_points)
        self._paint(canvas, canvas._paint_lattice)

    def test_the_whole_widget_renders_with_every_overlay_at_once(self):
        """One frame with annotations, boxes and the active box: the real case."""
        canvas = self._canvas()
        canvas.set_annotations([(1, [((1.0, 2.0), (30.0, 40.0))])])
        canvas.set_boxes([(1, (1.0, 2.0, 30.0, 40.0))])
        canvas.set_box((40.0, 40.0, 90.0, 90.0))
        canvas.set_points([(10.0, 10.0)])
        canvas.repaint()
        rendered = canvas.grab()
        self.assertFalse(rendered.isNull())



class SourceTests(unittest.TestCase):
    """One entry point, whatever the source is - and only ever one image on screen.

    ``set_source`` is the canvas' only way in, so the five screens that show an image
    (Calibrate, Annotate, Bench, Dataset, Predict) do not each repeat the file reading
    and do not each carry a second, weaker loader. It accepts a numpy frame, a file
    path, a QImage/QPixmap, or None; anything else is a programming error.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name)
        self.canvas = ImageCanvas()
        self.canvas.resize(600, 400)
        self.addCleanup(self.canvas.deleteLater)

    def _file(self, frame=IMAGE, name="frame.png"):
        """A real image file on disk, written by Qt (no codec questions)."""
        from daphnia.ui.widgets import frame_to_qimage
        path = self.folder / name
        self.assertTrue(frame_to_qimage(frame).save(str(path)))
        return path

    def test_a_numpy_frame_is_shown_and_read_back(self):
        frame = self.canvas.set_source(IMAGE)
        self.assertIs(frame, IMAGE)                 # no copy: the page keeps its frame
        self.assertIs(self.canvas.frame, IMAGE)
        self.assertIsNone(self.canvas.refusal)

    def test_a_file_is_read_by_the_canvas_and_read_back(self):
        """The page gets the pixels back, so it never loads the file a second time."""
        path = self._file()
        frame = self.canvas.set_source(path)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, IMAGE.shape)
        self.assertIs(self.canvas.frame, frame)

    def test_a_qimage_and_a_qpixmap_are_accepted(self):
        from PySide6.QtGui import QPixmap
        from daphnia.ui.widgets import frame_to_qimage

        image = frame_to_qimage(IMAGE)
        self.assertEqual(self.canvas.set_source(image).shape, IMAGE.shape)
        self.assertEqual(self.canvas.set_source(QPixmap.fromImage(image)).shape,
                         IMAGE.shape)

    def test_none_clears_the_canvas(self):
        self.canvas.set_source(IMAGE)
        self.canvas.set_source(None)
        self.assertIsNone(self.canvas.frame)
        self.assertIsNone(self.canvas.refusal)
        self.assertIsNone(self.canvas._pixmap)

    def test_a_source_the_canvas_cannot_show_is_named_and_reported(self):
        """Not a stale picture, not a silent blank: the reason, on the canvas."""
        failures = []
        self.canvas.source_failed.connect(failures.append)
        self.canvas.set_source(IMAGE)

        missing = self.folder / "gone.png"
        self.assertIsNone(self.canvas.set_source(missing))
        self.assertIsNone(self.canvas.frame)              # the old image is gone
        self.assertIn("gone.png", self.canvas.refusal)    # ... and the reason names it
        self.assertEqual(failures, [self.canvas.refusal])

    def test_an_unreadable_file_keeps_only_the_refusal(self):
        broken = self.folder / "broken.png"
        broken.write_bytes(b"not an image at all")
        self.assertIsNone(self.canvas.set_source(broken,
                                                 allow_system_conversion=False))
        self.assertIn("broken.png", self.canvas.refusal)

    def test_a_source_of_the_wrong_kind_is_an_error_not_a_blank_canvas(self):
        with self.assertRaises(TypeError) as raised:
            self.canvas.set_source(42)
        self.assertIn("int", str(raised.exception))

    def test_the_new_image_never_inherits_the_old_ones_markings(self):
        """'Only ever one image': points, box, overlays and grid go with the image."""
        self.canvas.set_points([(10.0, 10.0), (20.0, 20.0)])
        self.canvas.set_box((1.0, 1.0, 30.0, 30.0))
        self.canvas.set_boxes([(1, (1.0, 1.0, 30.0, 30.0))])
        self.canvas.set_lattice(10.0, 0.0)

        self.canvas.set_source(np.full((20, 20, 3), 7, dtype=np.uint8))
        self.assertEqual(self.canvas.points, [])
        self.assertIsNone(self.canvas.box)
        self.assertEqual(self.canvas.boxes, [])
        self.assertIsNone(self.canvas.lattice)

    def test_a_refused_source_leaves_nothing_of_the_previous_one_either(self):
        self.canvas.set_points([(10.0, 10.0)])
        self.canvas.set_source(self.folder / "nope.png")
        self.assertEqual(self.canvas.points, [])

    def test_the_same_pixels_come_back_out_of_a_qimage(self):
        """frame -> QImage -> frame is the round trip the source union rests on."""
        from daphnia.ui.widgets import frame_to_qimage, qimage_to_frame

        colour = np.zeros((10, 20, 3), dtype=np.uint8)
        colour[:, :, 0] = 200                     # BGR: blue is the channel that varies
        colour[:, :, 1] = 100
        grey = np.arange(200, dtype=np.uint8).reshape(10, 20)
        for frame in (colour, grey):
            with self.subTest(shape=frame.shape):
                back = qimage_to_frame(frame_to_qimage(frame))
                self.assertTrue(np.array_equal(back, frame), f"{frame.shape} changed")

    def test_the_frame_owns_its_pixels_and_not_qt_s_buffer(self):
        """Qt frees an image with it: a view of that buffer is a use-after-free.

        This is not hypothetical - the grayscale branch returned a view once (a
        contiguous slice through `ascontiguousarray` is not a copy), and the symptom
        was a first row shifted by one byte, only sometimes, only after other images
        had been allocated.
        """
        from daphnia.ui.widgets import frame_to_qimage, qimage_to_frame

        for frame in (IMAGE, np.arange(200, dtype=np.uint8).reshape(10, 20)):
            with self.subTest(shape=frame.shape):
                back = qimage_to_frame(frame_to_qimage(frame))
                self.assertTrue(back.flags["OWNDATA"], "the frame does not own its data")
                self.assertTrue(np.array_equal(back, frame))

    def test_an_empty_qimage_is_refused_by_name(self):
        from daphnia.ui.widgets import qimage_to_frame
        from daphnia.vision.images import ImageLoadError

        with self.assertRaises(ImageLoadError):
            qimage_to_frame(QImage())

    def test_a_refusal_is_painted_not_just_recorded(self):
        """The canvas shows the reason where the image would be - never blank silence."""
        self.canvas.set_source(IMAGE)
        self.canvas.set_source(self.folder / "gone.png")
        painted = self.canvas.grab().toImage()
        self.assertFalse(painted.isNull())
        self.assertIsNone(self.canvas._pixmap)          # nothing of the old image
        self.assertIn("gone.png", self.canvas.refusal)


class BoxEditTests(unittest.TestCase):
    """Selecting and editing the boxes on a frame (`ImageCanvas.box_edit`).

    The maths is pure (`box_at`, `resize_handle`, `moved_box`, `resized_box`) and the
    interaction is driven with real mouse events, because the promise being kept is a
    *gesture*: click a box and it is yours to move, resize or remove.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self, boxes=((1, (20.0, 10.0, 60.0, 40.0)),)):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        canvas.set_boxes(boxes)
        canvas.box_edit = True
        return canvas

    def _centre_of(self, canvas, box):
        return canvas._screen_box(box).center()

    def _press(self, canvas, point):
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, point, point, Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))

    def _move(self, canvas, start, end):
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, start, Qt.NoButton, Qt.LeftButton,
            Qt.NoModifier))

    def _release(self, canvas, point):
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, point, point, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    # -- the maths ---------------------------------------------------------
    def test_the_innermost_box_wins(self):
        """A box drawn around another one is the common mistake: pick the animal."""
        big, small = (0, 0, 100, 100), (40, 40, 60, 60)
        self.assertEqual(box_at((50, 50), [big, small]), small)
        self.assertEqual(box_at((51, 51), [small, big]), small)
        self.assertIsNone(box_at((150, 50), [big, small]))
        self.assertEqual(box_at((105, 50), [big], tolerance=10), big)
        self.assertIsNone(box_at((105, 50), [big], tolerance=1))

    def test_corners_beat_edges_and_outside_is_nothing(self):
        box = (10, 10, 50, 50)
        self.assertEqual(resize_handle((10, 10), box, 5), "nw")
        self.assertEqual(resize_handle((50, 50), box, 5), "se")
        self.assertEqual(resize_handle((30, 10), box, 5), "n")
        self.assertEqual(resize_handle((50, 30), box, 5), "e")
        self.assertIsNone(resize_handle((30, 30), box, 5))     # the middle moves it
        self.assertIsNone(resize_handle((200, 200), box, 5))

    def test_moving_a_box_keeps_its_size_and_stays_in_the_frame(self):
        box = (10, 10, 50, 40)
        self.assertEqual(moved_box(box, 5, -5), (15, 5, 55, 35))
        # 40 x 30 inside a 200 x 100 frame: the far corner is where it stops.
        self.assertEqual(moved_box(box, 500, 500, bounds=(200, 100)), (160, 70, 200, 100))

    def test_resizing_clamps_to_the_frame_and_never_collapses(self):
        box = (10, 10, 50, 40)
        self.assertEqual(resized_box(box, "se", (80, 70)), (10, 10, 80, 70))
        # Dragged through itself: the box keeps the minimum size, on the side being pulled.
        self.assertEqual(resized_box(box, "nw", (60, 50), minimum=20), (30, 20, 50, 40))
        self.assertEqual(resized_box(box, "se", (900, 900), bounds=(200, 100)),
                         (10, 10, 200, 100))



    # -- the gesture -------------------------------------------------------
    def test_clicking_a_box_selects_it_and_the_page_hears_it(self):
        canvas = self._canvas()
        seen = []
        canvas.box_selected.connect(seen.append)
        centre = self._centre_of(canvas, (20.0, 10.0, 60.0, 40.0))
        self._press(canvas, centre)
        self._release(canvas, centre)
        self.assertEqual(canvas.selected_box, 1)
        self.assertEqual(seen, [1])

    def test_a_press_that_selects_places_no_point(self):
        """The whole point of the tool: a click on a box is never a label."""
        canvas = self._canvas()
        points = []
        canvas.points_changed.connect(points.append)
        centre = self._centre_of(canvas, (20.0, 10.0, 60.0, 40.0))
        self._press(canvas, centre)
        self._release(canvas, centre)
        self.assertEqual(canvas.points, [])
        self.assertEqual([entry for entry in points if entry], [])

    def test_dragging_a_selected_box_reports_the_new_geometry(self):
        canvas = self._canvas()
        edits = []
        canvas.box_edited.connect(lambda index, box: edits.append((index, box)))
        centre = self._centre_of(canvas, (20.0, 10.0, 60.0, 40.0))
        self._press(canvas, centre)
        self._move(canvas, centre, centre + QPointF(20, 10))
        self._release(canvas, centre + QPointF(20, 10))
        self.assertEqual(len(edits), 1)
        index, box = edits[0]
        self.assertEqual(index, 1)
        self.assertAlmostEqual(box[2] - box[0], 40.0, places=3)   # width kept
        self.assertGreater(box[0], 20.0)                          # and it moved right

    def test_a_drag_that_changes_nothing_is_not_reported(self):
        """A click is not an edit: nothing is stored by a selection."""
        canvas = self._canvas()
        edits = []
        canvas.box_edited.connect(lambda index, box: edits.append((index, box)))
        centre = self._centre_of(canvas, (20.0, 10.0, 60.0, 40.0))
        self._press(canvas, centre)
        self._release(canvas, centre)
        self.assertEqual(edits, [])

    def test_a_predicted_box_is_picked_not_edited(self):
        canvas = self._canvas(boxes=())
        canvas.set_predictions([{"box": (30.0, 20.0, 70.0, 50.0)}])
        picked = []
        canvas.prediction_selected.connect(picked.append)
        centre = self._centre_of(canvas, (30.0, 20.0, 70.0, 50.0))
        self._press(canvas, centre)
        self._release(canvas, centre)
        self.assertEqual(canvas.selected_prediction, 0)
        self.assertIsNone(canvas.selected_box)
        self.assertEqual(picked, [0])

    def test_delete_asks_the_page_to_remove_the_selected_box(self):
        canvas = self._canvas()
        asked = []
        canvas.box_delete_requested.connect(asked.append)
        centre = self._centre_of(canvas, (20.0, 10.0, 60.0, 40.0))
        self._press(canvas, centre)
        self._release(canvas, centre)
        canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
        self.assertEqual(asked, [1])

    def test_escape_lets_go_of_the_selection(self):
        canvas = self._canvas()
        centre = self._centre_of(canvas, (20.0, 10.0, 60.0, 40.0))
        self._press(canvas, centre)
        self._release(canvas, centre)
        canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        self.assertIsNone(canvas.selected_box)

    def test_a_box_can_be_grabbed_without_the_select_tool(self):
        """A nearby box is editable **right away** - no need to switch tools first (`X-N15`).

        This is the behaviour an operator asked for: hovering a box (or a point, or a model's
        box) highlights it and a click grabs it, even while the create tool is on. Empty space
        is still the page's own gesture, so measuring is unchanged.
        """
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        canvas.set_boxes([(1, (20.0, 10.0, 60.0, 40.0))])
        centre = canvas._screen_box((20.0, 10.0, 60.0, 40.0)).center()
        self._press(canvas, centre)
        self._release(canvas, centre)
        self.assertEqual(canvas.selected_box, 1, "the box under the press was grabbed")
        self.assertEqual(canvas.points, [], "and no point was placed on it")
        # Empty space is still the create gesture.
        canvas.clear_selection()
        self._press(canvas, QPointF(10.0, 260.0))
        self._release(canvas, QPointF(10.0, 260.0))
        self.assertEqual(len(canvas.points), 1, "empty space places its point")

    def test_nothing_to_pick_means_nothing_to_offer(self):
        canvas = ImageCanvas()
        self.assertFalse(canvas.has_selectable())
        canvas.set_boxes([(1, (1.0, 1.0, 2.0, 2.0))])
        self.assertTrue(canvas.has_selectable())


class EditAllTests(unittest.TestCase):
    """Any point, any box, several of them at once (`SR-DATASET-027`).

    Driven with real mouse events, because what is promised is a *gesture*: drag a model's box
    and it becomes an annotation, drag a point and only that point moves, select several and
    move or delete them together.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self, boxes=None):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        canvas.set_boxes(boxes if boxes is not None else
                         [(1, (10.0, 10.0, 40.0, 30.0)), (2, (60.0, 12.0, 90.0, 34.0))])
        canvas.box_edit = True
        return canvas

    def _at(self, canvas, box, dx=0.0, dy=0.0):
        point = canvas._screen_box(box).center()
        return QPointF(point.x() + dx, point.y() + dy)

    def _gesture(self, canvas, start, end, modifiers=Qt.NoModifier):
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton, modifiers))
        if end is None:
            canvas.mouseReleaseEvent(QMouseEvent(
                QEvent.MouseButtonRelease, start, start, Qt.LeftButton, Qt.NoButton,
                modifiers))
            return
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, start, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, start, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))

    def test_shift_click_adds_to_the_selection(self):
        canvas = self._canvas()
        self._gesture(canvas, self._at(canvas, (10.0, 10.0, 40.0, 30.0)), None)
        self._gesture(canvas, self._at(canvas, (60.0, 12.0, 90.0, 34.0)), None,
                      modifiers=Qt.ShiftModifier)
        self.assertEqual(canvas.selected_boxes, [1, 2])
        self.assertEqual(canvas.selected_box, 2, "the last one clicked is the primary")

    def test_shift_click_again_removes_it(self):
        canvas = self._canvas()
        self._gesture(canvas, self._at(canvas, (10.0, 10.0, 40.0, 30.0)), None)
        second = self._at(canvas, (60.0, 12.0, 90.0, 34.0))
        self._gesture(canvas, second, None, modifiers=Qt.ShiftModifier)
        self._gesture(canvas, second, None, modifiers=Qt.ShiftModifier)
        self.assertEqual(canvas.selected_boxes, [1])

    def test_the_whole_selection_moves_together(self):
        canvas = self._canvas()
        edits = []
        canvas.box_edited.connect(lambda index, box: edits.append((index, box)))
        first = self._at(canvas, (10.0, 10.0, 40.0, 30.0))
        second = self._at(canvas, (60.0, 12.0, 90.0, 34.0))
        self._gesture(canvas, first, None)
        self._gesture(canvas, second, None, modifiers=Qt.ShiftModifier)
        self._gesture(canvas, second, QPointF(second.x() + 12, second.y() + 8))
        self.assertEqual(sorted(index for index, _box in edits), [1, 2])
        for index, box in edits:                       # both moved, both kept their size
            original = canvas.boxes[index - 1][1]
            self.assertAlmostEqual(box[2] - box[0], original[2] - original[0], places=3)

    def test_a_marquee_selects_every_box_it_covers(self):
        canvas = self._canvas()
        top_left = canvas._to_screen((0.0, 0.0))
        bottom_right = canvas._to_screen((100.0, 40.0))
        self._gesture(canvas, QPointF(*top_left), QPointF(*bottom_right))
        self.assertEqual(canvas.selected_boxes, [1, 2])

    def test_a_marquee_over_nothing_clears_the_selection(self):
        canvas = self._canvas()
        self._gesture(canvas, self._at(canvas, (10.0, 10.0, 40.0, 30.0)), None)
        empty = canvas._to_screen((150.0, 80.0))
        self._gesture(canvas, QPointF(*empty), QPointF(empty[0] + 30, empty[1] + 20))
        self.assertEqual(canvas.selected_boxes, [])

    def test_delete_asks_for_every_selected_box(self):
        canvas = self._canvas()
        asked = []
        canvas.box_delete_requested.connect(asked.append)
        self._gesture(canvas, self._at(canvas, (10.0, 10.0, 40.0, 30.0)), None)
        self._gesture(canvas, self._at(canvas, (60.0, 12.0, 90.0, 34.0)), None,
                      modifiers=Qt.ShiftModifier)
        canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
        self.assertEqual(sorted(asked), [1, 2])



    def _painted(self, canvas, box, patch=None):
        """How many pixels of the **prediction** colour are inside this box's screen region.

        Counting pixels is the only way to test what was actually *reported*: "I see a ghost
        lagging behind" and "I see two boxes on one animal" are both complaints about paint.
        ``patch`` narrows the count to a square of that many pixels around the box centre, for
        the case where the box itself moved away and only *its old place* is being checked.
        """
        canvas.repaint()
        image = canvas.grab().toImage()
        color = theme.qcolor("prediction")
        screen = canvas._screen_box(box)
        if patch:
            centre = screen.center()
            left, top = int(centre.x()) - patch, int(centre.y()) - patch
            right, bottom = int(centre.x()) + patch, int(centre.y()) + patch
        else:
            left, top = int(screen.left()) - 3, int(screen.top()) - 3
            right, bottom = int(screen.right()) + 3, int(screen.bottom()) + 3
        count = 0
        for y in range(max(0, top), min(image.height(), bottom + 1)):
            for x in range(max(0, left), min(image.width(), right + 1)):
                pixel = image.pixelColor(x, y)
                if (abs(pixel.red() - color.red()) < 40 and
                        abs(pixel.green() - color.green()) < 40 and
                        abs(pixel.blue() - color.blue()) < 40):
                    count += 1
        return count

    def test_a_prediction_over_a_label_is_not_drawn(self):
        """One box per animal: once a stored label covers the same animal, the dotted copy of
        it is not painted any more - two rectangles for one animal carry no information."""
        canvas = self._canvas(boxes=[])
        canvas.set_predictions([{"box": (20.0, 15.0, 60.0, 45.0)}])
        alone = self._painted(canvas, (20.0, 15.0, 60.0, 45.0))
        self.assertGreater(alone, 0, "the model's box is on the frame to begin with")
        canvas.set_boxes([(1, (20.0, 15.0, 60.0, 45.0))])
        self.assertIsNotNone(canvas.overlaps_stored((20.0, 15.0, 60.0, 45.0)))
        self.assertEqual(self._painted(canvas, (20.0, 15.0, 60.0, 45.0)), 0)

    def test_a_prediction_over_another_animal_is_still_drawn(self):
        """The rule is "the *same* animal twice", not "no predictions once you have labelled":
        a second animal's suggestion is exactly what the operator still needs to see."""
        canvas = self._canvas(boxes=[(1, (20.0, 15.0, 60.0, 45.0))])
        canvas.set_predictions([{"box": (70.0, 20.0, 95.0, 40.0)}])
        self.assertIsNone(canvas.overlaps_stored((70.0, 20.0, 95.0, 40.0)))
        self.assertGreater(self._painted(canvas, (70.0, 20.0, 95.0, 40.0)), 0)

    def test_the_box_being_edited_has_no_ghost_behind_it(self):
        """The dragged model box is the only rectangle on screen: the dotted original at the
        old position is not painted while the drag lasts, so "the new position was not
        updated" cannot be what the operator sees."""
        canvas = self._canvas(boxes=[])
        canvas.set_predictions([{"box": (20.0, 15.0, 60.0, 45.0)}])
        start = self._at(canvas, (20.0, 15.0, 60.0, 45.0))
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, QPointF(start.x() + 120, start.y() + 90), start,
            Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(canvas.selected_prediction, 0)
        self.assertEqual(self._painted(canvas, (20.0, 15.0, 60.0, 45.0), patch=9), 0,
                         "the old, dotted box must not stay behind the drag")
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, QPointF(start.x() + 120, start.y() + 90), start,
            Qt.LeftButton, Qt.NoButton, Qt.NoModifier))

    def test_moving_a_point_asks_the_page_to_look_but_not_to_re_render(self):
        """A point being *moved* is not a point being *placed*.

        `points_changed` means "a click placed a landmark": a page that hears it during an edit
        drag re-renders its own copy of the pair, which is the point snapping back to where it
        was - the "I let go and the position is not updated" report (`SR-DATASET-027`).
        """
        canvas = self._canvas(boxes=[])
        canvas.set_points([(20.0, 20.0), (60.0, 40.0)])
        placed, moved, edited = [], [], []
        canvas.points_changed.connect(placed.append)
        canvas.points_moved.connect(lambda: moved.append(True))
        canvas.points_edited.connect(lambda: edited.append(True))
        start = canvas._to_screen((60.0, 40.0))
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(*start), QPointF(*start),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, QPointF(start[0] + 20, start[1] + 10), QPointF(*start),
            Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, QPointF(start[0] + 20, start[1] + 10), QPointF(*start),
            Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
        self.assertEqual(placed, [], "a move must never look like a placement")
        self.assertTrue(moved and edited)
        end = canvas.points[1]
        self.assertGreater(end[0], 60.0, "the point kept the position it was dropped at")
        self.assertGreater(end[1], 40.0)

    def test_a_click_on_a_prediction_selects_it_and_stores_nothing(self):
        """In *Select* a model's box is **selected**, not accepted; the draft stays empty."""
        canvas = self._canvas(boxes=[])
        canvas.set_predictions([{"box": (20.0, 15.0, 60.0, 45.0)}])
        reported = []
        canvas.box_changed.connect(reported.append)
        press = self._at(canvas, (20.0, 15.0, 60.0, 45.0))
        self._gesture(canvas, press, None)
        self.assertEqual(canvas.selected_prediction, 0)
        self.assertEqual(reported, [], "a click must not label")
        self.assertIsNone(canvas.box, "Select only selects - the measuring tools accept it")

    def test_backspace_removes_a_picked_prediction(self):
        """"The model is wrong about this one": nothing is stored, so the layer just drops it."""
        canvas = self._canvas(boxes=[])
        canvas.set_predictions([{"box": (20.0, 15.0, 60.0, 45.0)},
                                {"box": (70.0, 20.0, 95.0, 40.0)}])
        removed = []
        canvas.prediction_removed.connect(removed.append)
        press = self._at(canvas, (20.0, 15.0, 60.0, 45.0))
        self._gesture(canvas, press, None)
        canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Backspace, Qt.NoModifier))
        self.assertEqual(removed, [0])
        self.assertEqual(len(canvas.predictions), 1)
        self.assertEqual(canvas.predictions[0][0], (70.0, 20.0, 95.0, 40.0))
        self.assertIsNone(canvas.selected_prediction)

    def test_the_edit_mode_is_on_by_default(self):
        """A fresh canvas edits rather than creates: the mode is automatic (`SR-DATASET-027`)."""
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        canvas.set_boxes([(1, (20.0, 10.0, 60.0, 40.0))])
        self.assertFalse(canvas.box_edit, "the canvas is a widget: a page switches the mode on")
        centre = canvas._screen_box((20.0, 10.0, 60.0, 40.0)).center()
        canvas.box_edit = True                      # what every labelling page does at start-up
        self._gesture(canvas, centre, None)
        self.assertEqual(canvas.selected_box, 1)
        self.assertEqual(canvas.points, [], "and the click created nothing")

    def test_point_mode_turns_the_press_into_the_keypoint_gesture(self):
        """The crop loop's rule: a click on the animal **places** the point.

        The item's box covers the whole magnified crop, so the ordinary order (a press
        inside a box selects it) swallowed every click: the prompt says *"click the head"*
        and nothing happened - *"not able to correct the keypoint: it first selects the
        box"* (`SR-DATASET-025`).
        """
        canvas = self._canvas(boxes=[(1, (20.0, 10.0, 60.0, 40.0))])
        canvas.point_mode = True
        selected = []
        canvas.box_selected.connect(selected.append)
        centre = canvas._screen_box((20.0, 10.0, 60.0, 40.0)).center()

        self._gesture(canvas, QPointF(centre), None)

        self.assertEqual(len(canvas.points), 1, "the press placed the keypoint")
        self.assertIsNone(canvas.selected_box, "and it picked no box")
        self.assertEqual(selected, [])
        self.assertIsNone(canvas._marquee, "nor did empty space start a marquee")

    def test_point_mode_still_picks_up_a_point_to_drag(self):
        """Correcting a keypoint is **one** press: a point under the cursor wins."""
        canvas = self._canvas(boxes=[(1, (20.0, 10.0, 60.0, 40.0))])
        canvas.point_mode = True
        canvas.set_points([(30.0, 20.0), (55.0, 35.0)])
        start = canvas._to_screen((55.0, 35.0))

        self._gesture(canvas, QPointF(*start), QPointF(start[0] + 8, start[1] + 6))

        self.assertEqual(canvas.selected_point, 1)
        self.assertNotEqual(tuple(canvas.points[1]), (55.0, 35.0))
        self.assertEqual(tuple(canvas.points[0]), (30.0, 20.0), "the other point stayed")
        self.assertEqual(len(canvas.points), 2, "a drag moves a point, it places nothing")
        self.assertIsNone(canvas.selected_box)



    def test_a_model_box_dragged_becomes_a_draft_box(self):
        """A **measuring** tool promotes the prediction *silently*, and on the drop reports it
        **once** - with the final geometry - through `prediction_accepted` (`X-N19`)."""
        canvas = self._canvas(boxes=[])
        canvas.set_tool(TOOL_LINE)          # a measuring tool: the press accepts
        canvas.set_predictions([{"box": (20.0, 15.0, 60.0, 45.0)}])
        drawn, accepted = [], []
        canvas.box_changed.connect(drawn.append)
        canvas.prediction_accepted.connect(
            lambda index, box, points, moved: accepted.append((index, box, moved)))
        start = self._at(canvas, (20.0, 15.0, 60.0, 45.0))
        self._gesture(canvas, start, QPointF(start.x() + 10, start.y() + 6))
        self.assertEqual(drawn, [], "not reported as a drawn box as well - that would double it")
        self.assertEqual(len(accepted), 1, "one drag, one report")
        index, box, moved = accepted[0]
        self.assertEqual((index, moved), (0, True))
        self.assertAlmostEqual(box[2] - box[0], 40.0, places=3)
        self.assertGreater(box[0], 20.0)
        self.assertEqual(len(canvas.predictions), 1,
                         "the canvas does not remove it - the page does")

    def test_a_point_is_dragged_on_its_own(self):
        canvas = self._canvas(boxes=[])
        canvas.set_points([(30.0, 20.0), (60.0, 40.0)])
        seen, moved = [], []
        canvas.points_changed.connect(seen.append)
        canvas.points_moved.connect(lambda: moved.append(True))
        start = canvas._to_screen((60.0, 40.0))
        self._gesture(canvas, QPointF(*start), QPointF(start[0] + 20, start[1] + 10))
        self.assertEqual(canvas.selected_point, 1)
        self.assertNotEqual(tuple(canvas.points[1]), (60.0, 40.0))
        self.assertEqual(tuple(canvas.points[0]), (30.0, 20.0), "the other point stayed")
        self.assertTrue(moved, "the page hears every step of the drag")
        self.assertEqual(seen, [], "an edit is not a placement (`SR-DATASET-027`)")

    def test_delete_removes_a_selected_point(self):
        canvas = self._canvas(boxes=[])
        canvas.set_points([(30.0, 20.0), (60.0, 40.0)])
        start = canvas._to_screen((30.0, 20.0))
        self._gesture(canvas, QPointF(*start), None)
        canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
        self.assertEqual(len(canvas.points), 1)
        self.assertEqual(tuple(canvas.points[0]), (60.0, 40.0))
        self.assertIsNone(canvas.selected_point)



    def test_a_double_click_in_edit_mode_is_two_selections(self):
        """It must **not** walk to the next frame: that threw the selection away mid-correction."""
        canvas = self._canvas()
        walked = []
        canvas.double_clicked.connect(lambda: walked.append(True))
        centre = self._at(canvas, (10.0, 10.0, 40.0, 30.0))
        self._gesture(canvas, centre, None)
        canvas.mouseDoubleClickEvent(QMouseEvent(
            QEvent.MouseButtonDblClick, centre, centre, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        self.assertEqual(walked, [], "edit mode: a double click is not 'next frame'")
        self.assertEqual(canvas.selected_boxes, [1], "and the selection survives it")

    def test_the_double_click_still_walks_the_queue_in_creation_mode(self):
        """The labelling gesture has to keep working where it belongs (`X-N13`)."""
        canvas = self._canvas()
        canvas.box_edit = False
        canvas.box_mode = True
        walked = []
        canvas.double_clicked.connect(lambda: walked.append(True))
        canvas.mouseDoubleClickEvent(QMouseEvent(
            QEvent.MouseButtonDblClick, QPointF(10, 10), QPointF(10, 10), Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(walked, [True])

    def test_delete_selection_is_one_call_for_the_pages_shortcut(self):
        """The pages bind `Backspace` as a shortcut: the key must not need canvas focus."""
        canvas = self._canvas()
        asked = []
        canvas.box_delete_requested.connect(asked.append)
        self.assertFalse(canvas.delete_selection(), "nothing selected, nothing removed")
        centre = self._at(canvas, (10.0, 10.0, 40.0, 30.0))
        self._gesture(canvas, centre, None)
        self.assertTrue(canvas.delete_selection())
        self.assertEqual(asked, [1])
        self.assertEqual(canvas.selected_boxes, [])



class ToolTests(unittest.TestCase):
    """The canvas' interaction **tools**: one press, one meaning (`X-N13`).

    Editing used to be a single boolean that both selected and created, so a page could
    not offer *select* without turning *create* off. Each tool is driven here with real
    mouse events, because what is promised is a *gesture*.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self, boxes=None):
        canvas = ImageCanvas()
        canvas.resize(600, 400)
        canvas.set_source(IMAGE)
        canvas.set_boxes(boxes if boxes is not None else
                         [(1, (10.0, 10.0, 40.0, 30.0))])
        return canvas

    def _press(self, canvas, point, modifiers=Qt.NoModifier):
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, point, point, Qt.LeftButton, Qt.LeftButton,
            modifiers))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, point, point, Qt.LeftButton, Qt.NoButton,
            modifiers))

    def test_the_default_tool_is_a_measuring_line(self):
        """A fresh canvas measures: editing is a tool you choose, not the default."""
        canvas = self._canvas()
        self.assertEqual(canvas.tool, TOOL_LINE)
        self.assertFalse(canvas.box_edit, "and a click creates rather than selects")

    def test_every_tool_is_a_valid_setting(self):
        canvas = self._canvas()
        for tool in TOOLS:
            canvas.set_tool(tool)
            self.assertEqual(canvas.tool, tool)

    def test_part_all_drops_a_box_only_prediction(self):
        """`part="all"` must remove what is left - a **box-only** answer included.

        Found by the Bench's *From the model* list (`SR-BENCH-017`): a detector that answered
        without a pose left an item `part="all"` refused to remove, because the *keypoints* it
        was asked to clear were already absent - and the early return abandoned the whole call.
        """
        canvas = self._canvas()
        canvas.set_predictions([((10.0, 10.0, 60.0, 50.0), (), "")])       # no keypoints
        self.assertEqual(len(canvas.predictions), 1)
        canvas.drop_prediction(0, part="all")
        self.assertEqual(canvas.predictions, [],
                         "the box-only prediction stayed in the layer for good")

    def test_asking_for_a_part_already_gone_leaves_the_rest_alone(self):
        """A part that is already absent is not a failure of the call (`X-N17`)."""
        canvas = self._canvas()
        canvas.set_predictions([((10.0, 10.0, 60.0, 50.0), ((20.0, 20.0), (40.0, 30.0)), "")])
        canvas.drop_prediction(0, part="box")            # the box goes, the keypoints stay
        self.assertEqual(len(canvas.predictions), 1)
        canvas.drop_prediction(0, part="box")            # and again
        self.assertEqual(len(canvas.predictions), 1, "asking twice removed the item")
        canvas.drop_prediction(0, part="all")            # now nothing is left
        self.assertEqual(canvas.predictions, [])

    def test_a_line_tool_press_places_a_point(self):
        canvas = self._canvas()
        canvas.set_tool(TOOL_LINE)
        self._press(canvas, QPointF(30, 200))
        self.assertEqual(len(canvas.points), 1, "the click is the landmark")

    def test_select_tool_selects_and_creates_nothing(self):
        canvas = self._canvas()
        canvas.set_tool(TOOL_SELECT)
        self.assertTrue(canvas.box_edit)
        centre = canvas._screen_box((10.0, 10.0, 40.0, 30.0)).center()
        self._press(canvas, QPointF(centre))
        self.assertEqual(canvas.selected_box, 1)
        self.assertEqual(canvas.points, [], "a select press never places a point")

    def test_box_tool_draws_on_empty_space_and_still_edits(self):
        """The split the operator asked for: in box mode a box can be *made* and edited."""
        canvas = self._canvas()
        canvas.set_tool(TOOL_BOX)
        self.assertTrue(canvas.box_mode)
        self.assertTrue(canvas.box_edit, "the box tool also edits what is there")
        canvas.clear_box()
        self._press(canvas, QPointF(200, 200))     # empty space, no box under it
        self.assertTrue(canvas.box_pending, "empty space anchors a corner to draw from")

    def test_the_select_tool_marquees_on_empty_space(self):
        canvas = self._canvas()
        canvas.set_tool(TOOL_SELECT)
        start = QPointF(5, 200)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        self.assertIsNotNone(canvas._marquee, "the select tool marquees")
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, start, start, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))

    # -- editing is always available; the tools only draw (`X-N20`) --------
    def test_a_bare_canvas_leaves_editing_off(self):
        """The widget's default is unchanged: it is a page that turns click-to-edit on."""
        self.assertFalse(ImageCanvas().editing)

    def test_editing_picks_an_object_in_any_draw_tool(self):
        """With editing on, a press on a box edits it - the measuring tool included."""
        canvas = self._canvas()
        canvas.editing = True
        canvas.set_tool(TOOL_LINE)
        centre = canvas._screen_box((10.0, 10.0, 40.0, 30.0)).center()
        self._press(canvas, QPointF(centre))
        self.assertEqual(canvas.selected_box, 1, "touching the box selected it")
        self.assertEqual(canvas.points, [], "and the press drew nothing")

    def test_editing_leaves_empty_space_to_the_draw_tool(self):
        """Empty space is where the tool in force draws: editing must not steal it."""
        canvas = self._canvas()
        canvas.editing = True
        canvas.set_tool(TOOL_LINE)
        self._press(canvas, QPointF(200, 200))
        self.assertEqual(len(canvas.points), 1, "the line tool placed its point")
        self.assertIsNone(canvas._marquee, "no marquee ran")

    def test_editing_leaves_empty_space_to_the_box_tool(self):
        canvas = self._canvas()
        canvas.editing = True
        canvas.set_tool(TOOL_BOX)
        canvas.clear_box()
        self._press(canvas, QPointF(200, 200))
        self.assertTrue(canvas.box_pending, "the box tool anchored its corner")
        self.assertEqual(canvas.points, [])

    def test_a_marquee_on_empty_space_needs_shift(self):
        """A marquee is a *selection* gesture: with a draw tool it takes **Shift**."""
        canvas = self._canvas()
        canvas.editing = True
        canvas.set_tool(TOOL_LINE)
        start = QPointF(5, 200)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton,
            Qt.ShiftModifier))
        self.assertIsNotNone(canvas._marquee, "Shift on empty space marquees")
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, start, start, Qt.LeftButton, Qt.NoButton,
            Qt.ShiftModifier))

    def test_editing_makes_a_double_click_two_selections_not_next(self):
        """While editing is on a double click is not "next frame" - it is two selections."""
        canvas = self._canvas()
        canvas.editing = True
        canvas.set_tool(TOOL_LINE)
        asked = []
        canvas.double_clicked.connect(lambda: asked.append(True))
        centre = canvas._screen_box((10.0, 10.0, 40.0, 30.0)).center()
        canvas.mouseDoubleClickEvent(QMouseEvent(
            QEvent.MouseButtonDblClick, QPointF(centre), QPointF(centre), Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(asked, [], "the page was not asked to walk on")

    def test_a_real_press_reports_point_placed_but_set_points_does_not(self):
        """Only a landing click fires `point_placed` - what an auto-storing tool needs."""
        canvas = self._canvas()
        placed = []
        canvas.point_placed.connect(placed.append)
        canvas.set_points([(10.0, 10.0)])
        self.assertEqual(placed, [], "restoring a draft is not a placed point")
        self._press(canvas, QPointF(40, 200))
        self.assertEqual(placed, [[(10.0, 10.0), (canvas.points[-1])]])

    def test_a_skeleton_is_drawn_from_the_named_bones(self):
        """Painting a keypoint set with bones must not raise (two pairs, four points)."""
        canvas = self._canvas()
        canvas.point_names = ["head", "tail", "left", "right"]
        canvas.skeleton = [(0, 1), (2, 3)]
        canvas.set_points([(10.0, 10.0), (60.0, 10.0), (30.0, 5.0), (30.0, 25.0)])
        image = QImage(600, 400, QImage.Format_ARGB32)
        painter = QPainter(image)
        try:
            canvas._paint_points(painter, canvas._display_rect())
        finally:
            painter.end()

    # -- hover-to-grab: edit right away, without switching tools (`X-N15`) --
    def _move(self, canvas, point):
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, point, point, Qt.NoButton, Qt.NoButton, Qt.NoModifier))

    def test_hover_reports_what_the_next_press_would_grab(self):
        canvas = self._canvas()
        canvas.set_points([(120.0, 60.0)])
        self._move(canvas, QPointF(*canvas._to_screen((120.0, 60.0))))
        self.assertEqual(canvas.hover, ("point", 0), "a keypoint is grabbable")
        centre = canvas._screen_box((10.0, 10.0, 40.0, 30.0)).center()
        self._move(canvas, QPointF(centre))
        self.assertEqual(canvas.hover, ("box", 1), "a stored box is grabbable")
        self._move(canvas, QPointF(10.0, 260.0))          # empty space
        self.assertIsNone(canvas.hover)
        canvas.leaveEvent(QEvent(QEvent.Leave))
        self.assertIsNone(canvas.hover)

    def test_hover_shows_a_grab_cursor(self):
        canvas = self._canvas()
        centre = canvas._screen_box((10.0, 10.0, 40.0, 30.0)).center()
        self._move(canvas, QPointF(centre))
        self.assertEqual(canvas.cursor().shape(), Qt.OpenHandCursor)
        self._move(canvas, QPointF(10.0, 260.0))
        self.assertNotEqual(canvas.cursor().shape(), Qt.OpenHandCursor)

    def test_point_mode_hovers_only_points(self):
        """In the crop the box *is* the animal: only a point is grabbable there."""
        canvas = self._canvas()
        canvas.point_mode = True
        canvas.set_points([(120.0, 60.0)])
        centre = canvas._screen_box((10.0, 10.0, 40.0, 30.0)).center()
        self._move(canvas, QPointF(centre))
        self.assertIsNone(canvas.hover, "the crop's box is not advertised as grabbable")
        self._move(canvas, QPointF(*canvas._to_screen((120.0, 60.0))))
        self.assertEqual(canvas.hover, ("point", 0))

    def test_a_line_tool_press_grabs_a_nearby_point_instead_of_placing_one(self):
        """A keypoint is editable **right away** - the create tool does not block it."""
        canvas = self._canvas()
        canvas.set_tool(TOOL_LINE)
        canvas.set_points([(120.0, 60.0)])
        moved, placed = [], []
        canvas.points_moved.connect(lambda: moved.append(True))
        canvas.point_placed.connect(placed.append)
        start = QPointF(*canvas._to_screen((120.0, 60.0)))
        end = QPointF(start.x() + 8, start.y() + 6)
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, start, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, start, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))
        self.assertEqual(len(canvas.points), 1, "the point was grabbed, not replaced")
        self.assertEqual(moved, [True], "a grabbed point is a move")
        self.assertEqual(placed, [], "and nothing was placed")
        self.assertNotEqual(tuple(canvas.points[0]), (120.0, 60.0), "the point moved")

    def test_a_line_tool_press_grabs_a_prediction(self):
        canvas = self._canvas()
        canvas.set_tool(TOOL_LINE)
        canvas.set_predictions([((60.0, 12.0, 90.0, 34.0), [], "")])
        picked = []
        canvas.prediction_selected.connect(picked.append)
        centre = canvas._screen_box((60.0, 12.0, 90.0, 34.0)).center()
        self._press(canvas, QPointF(centre))
        self.assertEqual(picked, [0], "the model's box was picked to edit")
        self.assertEqual(canvas.points, [], "and no point was placed")

    def test_painting_the_hover_does_not_raise(self):
        canvas = self._canvas()
        canvas.set_points([(120.0, 60.0)])
        canvas._set_hover(("point", 0))
        image = QImage(600, 400, QImage.Format_ARGB32)
        painter = QPainter(image)
        try:
            canvas._paint_hover(painter, canvas._display_rect())
        finally:
            painter.end()

    # -- a stored measurement is editable (`X-N16`) ------------------------
    def _annotation(self, canvas):
        canvas.annotation_edit = True
        canvas.set_annotations([(1, [((120.0, 40.0), (160.0, 40.0))], "", ["length"])])

    def test_annotations_are_not_grabbable_unless_the_page_asks(self):
        """A page that never stores a measurement into an overlay keeps its clicks."""
        canvas = self._canvas()
        canvas.set_annotations([(1, [((120.0, 40.0), (160.0, 40.0))], "", ["length"])])
        at = QPointF(*canvas._to_screen((120.0, 40.0)))
        self._move(canvas, at)
        self.assertIsNone(canvas.hover)
        self._press(canvas, at)
        self.assertEqual(len(canvas.points), 1,
                         "the overlay swallowed nothing: the click is a normal placement")

    def test_a_stored_measurement_is_reported_and_grabbed(self):
        canvas = self._canvas()
        self._annotation(canvas)
        reported = []
        canvas.annotation_point_selected.connect(
            lambda specimen, segment, endpoint:
                reported.append((specimen, segment, endpoint)))
        # The page answers by loading the stored row back into the draft, synchronously.
        canvas.annotation_point_selected.connect(
            lambda *_args: canvas.set_points([(120.0, 40.0), (160.0, 40.0)]))
        at = QPointF(*canvas._to_screen((120.0, 40.0)))
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, at, at, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(reported, [(1, 0, 0)], "the press named the exact measurement")
        self.assertEqual(len(canvas.points), 2, "the page loaded the stored row")
        self.assertEqual(canvas.selected_point, 0, "and the same press grabbed the end")

    def test_hovering_a_stored_measurement_offers_a_grab(self):
        canvas = self._canvas()
        self._annotation(canvas)
        self._move(canvas, QPointF(*canvas._to_screen((160.0, 40.0))))
        self.assertEqual(canvas.hover, ("annotation", 1))

    def test_a_measurement_is_deleted_like_any_other_object(self):
        """`Backspace` after clicking a measurement asks the page to remove it (`X-N15`)."""
        canvas = self._canvas()
        self._annotation(canvas)
        canvas.annotation_point_selected.connect(
            lambda *_args: canvas.set_points([(120.0, 40.0), (160.0, 40.0)]))
        asked = []
        canvas.annotation_delete_requested.connect(
            lambda specimen, segment: asked.append((specimen, segment)))
        at = QPointF(*canvas._to_screen((120.0, 40.0)))
        self._press(canvas, at)
        self.assertTrue(canvas.delete_selection())
        self.assertEqual(asked, [(1, 0)], "the measured line, not a box or a point")

    # -- selecting several objects (`X-N15`) -------------------------------
    def test_several_predictions_can_be_selected(self):
        canvas = self._canvas()
        canvas.set_tool(TOOL_SELECT)
        canvas.set_predictions([((100.0, 15.0, 140.0, 35.0), (), ""),
                                ((150.0, 15.0, 190.0, 35.0), (), "")])
        first = QPointF(*canvas._to_screen((120.0, 25.0)))
        second = QPointF(*canvas._to_screen((170.0, 25.0)))
        self._press(canvas, first)
        self.assertEqual(canvas.selected_predictions, [0])
        self._press(canvas, second, Qt.ShiftModifier)
        self.assertEqual(canvas.selected_predictions, [0, 1],
                         "a prediction is an object you can select like any other")

    def test_backspace_removes_every_selected_prediction(self):
        canvas = self._canvas()
        canvas.set_tool(TOOL_SELECT)
        canvas.set_predictions([((100.0, 15.0, 140.0, 35.0), (), ""),
                                ((150.0, 15.0, 190.0, 35.0), (), "")])
        removed = []
        canvas.prediction_removed.connect(removed.append)
        self._press(canvas, QPointF(*canvas._to_screen((120.0, 25.0))))
        self._press(canvas, QPointF(*canvas._to_screen((170.0, 25.0))),
                    Qt.ShiftModifier)
        self.assertTrue(canvas.delete_selection())
        self.assertEqual(sorted(removed), [0, 1])
        self.assertEqual(canvas.predictions, [])

    def test_several_measurements_can_be_selected_and_deleted(self):
        canvas = self._canvas()
        canvas.set_tool(TOOL_SELECT)
        canvas.annotation_edit = True
        canvas.set_annotations([(1, [((120.0, 40.0), (160.0, 40.0))], "", ["length"]),
                                (2, [((175.0, 40.0), (195.0, 40.0))], "", ["length"])])
        asked = []
        canvas.annotation_delete_requested.connect(
            lambda specimen, segment: asked.append((specimen, segment)))
        self._press(canvas, QPointF(*canvas._to_screen((120.0, 40.0))))
        self._press(canvas, QPointF(*canvas._to_screen((175.0, 40.0))), Qt.ShiftModifier)
        self.assertEqual(canvas.selected_annotations, [(1, 0), (2, 0)])
        self.assertTrue(canvas.delete_selection())
        self.assertEqual(sorted(asked), [(1, 0), (2, 0)])

    def test_a_marquee_selects_boxes_measurements_and_predictions(self):
        canvas = self._canvas()          # a stored box at (10,10,40,30)
        canvas.set_tool(TOOL_SELECT)
        canvas.annotation_edit = True
        canvas.annotation_point_selected.connect(lambda *_a: None)
        canvas.set_annotations([(2, [((100.0, 10.0), (140.0, 10.0))], "", ["length"])])
        canvas.set_predictions([((150.0, 12.0, 190.0, 34.0), (), "")])
        start = QPointF(*canvas._to_screen((195.0, 85.0)))     # empty space
        end = QPointF(*canvas._to_screen((5.0, 5.0)))
        canvas.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, start, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))
        canvas.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, start, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        canvas.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, start, Qt.LeftButton, Qt.NoButton,
            Qt.NoModifier))
        self.assertEqual(canvas.selected_boxes, [1])
        self.assertEqual(canvas.selected_annotations, [(2, 0)])
        self.assertEqual(canvas.selected_predictions, [0])

    def test_box_edge_grab_picks_a_box_by_its_outline_only(self):
        """With the flag on, only the outline picks a box - the inside reaches what is in it."""
        canvas = self._canvas()          # a stored box at (10,10,40,30)
        canvas.set_tool(TOOL_SELECT)
        canvas.box_edge_grab = True
        screen = canvas._screen_box((10.0, 10.0, 40.0, 30.0))
        self._press(canvas, QPointF(screen.center()))
        self.assertEqual(canvas.selected_boxes, [], "the inside is not the box's line")
        self._press(canvas, QPointF(screen.topLeft()))
        self.assertEqual(canvas.selected_boxes, [1])

    # -- the tool follows the object (`X-N19`) -----------------------------
    def test_a_press_on_a_box_reports_the_box_tool(self):
        canvas = self._canvas()          # a stored box at (10,10,40,30)
        canvas.set_tool(TOOL_LINE)
        seen = []
        canvas.object_pressed.connect(seen.append)
        self._press(canvas, QPointF(*canvas._to_screen((25.0, 20.0))))
        self.assertEqual(seen, [TOOL_BOX], "the object says which tool owns it")

    def test_a_press_on_a_measurement_reports_the_measuring_tool(self):
        canvas = self._canvas()
        canvas.set_tool(TOOL_SELECT)
        self._annotation(canvas)
        canvas.annotation_point_selected.connect(
            lambda *_args: canvas.set_points([(120.0, 40.0), (160.0, 40.0)]))
        seen = []
        canvas.object_pressed.connect(seen.append)
        self._press(canvas, QPointF(*canvas._to_screen((120.0, 40.0))))
        self.assertEqual(seen, [TOOL_LINE])

    def test_a_press_on_a_predicted_box_reports_the_box_tool(self):
        canvas = self._canvas(boxes=[])
        canvas.set_tool(TOOL_LINE)
        canvas.set_predictions([{"box": (100.0, 15.0, 140.0, 35.0)}])
        seen = []
        canvas.object_pressed.connect(seen.append)
        self._press(canvas, QPointF(*canvas._to_screen((120.0, 25.0))))
        self.assertEqual(seen, [TOOL_BOX])

    def test_a_press_on_a_predicted_keypoint_reports_the_keypoints_tool(self):
        canvas = self._canvas(boxes=[])
        canvas.set_tool(TOOL_SELECT)
        canvas.set_predictions([{"box": (100.0, 15.0, 140.0, 35.0),
                                 "points": ((110.0, 20.0), (130.0, 30.0))}])
        seen = []
        canvas.object_pressed.connect(seen.append)
        self._press(canvas, QPointF(*canvas._to_screen((110.0, 20.0))))
        self.assertEqual(seen, [TOOL_KEYPOINTS])

    def test_set_tool_can_keep_the_work_the_press_started(self):
        """Following an object must not throw away the selection the press just made."""
        canvas = self._canvas()
        canvas.set_tool(TOOL_SELECT)
        canvas.select_box(1)
        canvas.set_tool(TOOL_BOX, keep=True)
        self.assertEqual(canvas.tool, TOOL_BOX)
        self.assertEqual(canvas.selected_boxes, [1], "the selection stayed")
        canvas.set_tool(TOOL_LINE)
        self.assertEqual(canvas.selected_boxes, [], "a *choice* still clears it")

    def test_painting_a_selected_prediction_does_not_raise(self):
        """A selected prediction must stay **visible** (it used to be hidden, so it looked
        like it could not be selected)."""
        canvas = self._canvas()
        canvas.set_predictions([((100.0, 15.0, 140.0, 35.0), ((110.0, 20.0), (130.0, 30.0)), "")])
        canvas.select_prediction(0)
        image = QImage(600, 400, QImage.Format_ARGB32)
        painter = QPainter(image)
        try:
            canvas._paint_predictions(painter, canvas._display_rect())
        finally:
            painter.end()


if __name__ == "__main__":
    unittest.main()
