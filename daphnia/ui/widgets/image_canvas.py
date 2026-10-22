"""Image display that reports clicks in *image* pixel coordinates."""

import os

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from daphnia.annotation import MIN_BOX_PIXELS
from daphnia.ui import theme
from daphnia.vision.frames import to_landscape  # re-exported by daphnia.ui.widgets
from daphnia.vision.images import ImageLoadError, load_image

MARKER_COLOR = "#00adb5"

#: Colours for specimens that are *already* annotated. They are deliberately
#: distinct from the active marker, so the points you are placing now stand out
#: from the ones you placed before - and each specimen keeps its own colour, so
#: several animals on one frame can be told apart without reading a label.
SPECIMEN_COLORS = (
    "#ff5c8a", "#8a5cff", "#5cff8a", "#ffd25c", "#5cd2ff", "#ff8a5c",
    "#c0ff5c", "#5cffd2",
)


def specimen_color(index: int) -> str:
    """A stable colour for a specimen index (1-based), cycling through the palette."""
    return SPECIMEN_COLORS[(int(index) - 1) % len(SPECIMEN_COLORS)]


#: The canvas' interaction **tools**: what one left press means. Editing used to be a
#: single boolean (``box_edit``) doing four jobs at once - select, move/resize, drag a
#: point, marquee - so a page could not offer *select* without also turning off
#: *create*, and Bench opened in an edit mode nobody asked for (one had to switch it
#: off before a single measurement could be placed). The tools name the jobs apart:
#:
#: * ``line`` - place two points and they are one measured line (never selects);
#: * ``keypoints`` - place the schema's anatomical keypoints (a *skeleton* of bones);
#: * ``box`` - draw a new bounding box, and edit the boxes already on the frame;
#: * ``select`` - pick what is under the press and edit it, and never create a label.
#:
#: A tool says what a press on **empty space** means, and that is all it means. With
#: ``editing`` on (`X-N20`) *touching* an object - selecting it, moving it, resizing it,
#: correcting it - is available in **every** tool, because that is the standard way to
#: work with what is on the screen; a page then offers the *draw* tools only (line,
#: keypoints, box) and never has to name *select* as a mode.
TOOL_LINE = "line"
TOOL_KEYPOINTS = "keypoints"
TOOL_BOX = "box"
TOOL_SELECT = "select"

#: Every tool, in the order a page offers them.
TOOLS = (TOOL_LINE, TOOL_KEYPOINTS, TOOL_BOX, TOOL_SELECT)

#: How a tool is phrased for an operator.
TOOL_LABELS = {
    TOOL_LINE: "Line",
    TOOL_KEYPOINTS: "Keypoints",
    TOOL_BOX: "Box",
    TOOL_SELECT: "Select",
}


#: How close to an edge or a corner a press must be to grab a box for editing, in **screen**
#: pixels: a handle is a finger, not a pixel, and a 12-pixel target is what makes a corner
#: corner-shaped for a mouse.
BOX_GRAB_PIXELS = 10

#: The eight ways a box can be resized - a corner or an edge - and which edges each moves.
RESIZE_HANDLES = {
    "nw": ("left", "top"), "n": ("top",), "ne": ("right", "top"),
    "w": ("left",), "e": ("right",),
    "sw": ("left", "bottom"), "s": ("bottom",), "se": ("right", "bottom"),
}


def _corners(box) -> tuple:
    """``(x1, y1, x2, y2)`` from a pair-of-points, a 4-tuple, **or** a ``QRectF``.

    The canvas stores boxes as numbers and draws them as Qt geometry, and the hit tests are
    used from both sides - so they read either, instead of every caller remembering which.
    """
    if hasattr(box, "left"):                       # a QRectF
        return (float(box.left()), float(box.top()),
                float(box.right()), float(box.bottom()))
    values = [float(value) for value in box]
    if len(values) == 4:
        return tuple(values)
    (ax, ay), (bx, by) = values[0], values[1]
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def _xy(point) -> tuple:
    """``(x, y)`` from a tuple, a list **or** a ``QPoint``/``QPointF``."""
    if hasattr(point, "x"):
        return (float(point.x()), float(point.y()))
    x, y = point
    return (float(x), float(y))


def box_at(point, boxes, tolerance=0.0):
    """Which box a press is **inside**: the innermost one, or ``None``.

    ``boxes`` is an iterable of ``(x1, y1, x2, y2)``; ``tolerance`` widens every box by that
    much (``+`` grows it), which is how a press just outside an edge still counts as *on* it.
    The **innermost** (smallest area) box wins: two boxes on one animal is a labelling error
    this surface reports, and picking the smaller one is what lets the operator edit the
    wrong box they just made instead of the big one around it.
    """
    if point is None:
        return None
    x, y = _xy(point)
    contained = []
    for box in boxes or ():
        x1, y1, x2, y2 = _corners(box)
        if (x1 - tolerance) <= x <= (x2 + tolerance) \
                and (y1 - tolerance) <= y <= (y2 + tolerance):
            contained.append((max(0.0, x2 - x1) * max(0.0, y2 - y1), box))
    if not contained:
        return None
    return min(contained, key=lambda item: item[0])[1]


def box_edge_at(point, boxes, tolerance=BOX_GRAB_PIXELS):
    """Which box's **outline** a press is on - the nearest edge within ``tolerance``, or ``None``.

    A box is grabbed by its **line**, not by its whole interior: a press deep inside belongs to
    what is *inside* the box (a keypoint, a measurement, a smaller box), which is what stops a
    rectangle drawn around an animal from swallowing every click meant for the objects in it.
    A press just outside an edge counts too, so the target is as forgiving as a finger needs.
    """
    if point is None:
        return None
    x, y = _xy(point)
    best = None
    for box in boxes or ():
        x1, y1, x2, y2 = _corners(box)
        if not ((x1 - tolerance) <= x <= (x2 + tolerance)
                and (y1 - tolerance) <= y <= (y2 + tolerance)):
            continue
        distance = min(abs(x - x1), abs(x - x2), abs(y - y1), abs(y - y2))
        if distance <= tolerance and (best is None or distance < best[0]):
            best = (distance, box)
    return None if best is None else best[1]


def resize_handle(point, box, tolerance) -> str:
    """Which resize handle of ``box`` a press is on: ``"nw"``, ``"e"``, ... or ``None``.

    Corners win over edges, edges over nothing: a press in the corner region is a corner
    grab even though it is also on two edges.
    """
    if point is None or box is None:
        return None
    x, y = _xy(point)
    x1, y1, x2, y2 = _corners(box)
    inside = ((x1 - tolerance) <= x <= (x2 + tolerance)
              and (y1 - tolerance) <= y <= (y2 + tolerance))
    if not inside:
        return None
    near = {"left": abs(x - x1) <= tolerance, "right": abs(x - x2) <= tolerance,
            "top": abs(y - y1) <= tolerance, "bottom": abs(y - y2) <= tolerance}
    horizontal = ("w" if near["left"] else "") + ("e" if near["right"] else "")
    vertical = ("n" if near["top"] else "") + ("s" if near["bottom"] else "")
    if horizontal and vertical:                       # a corner: "n"+"w" -> "nw"
        return (vertical[0] if "n" in vertical else "s") + horizontal[0]
    if horizontal or vertical:
        return horizontal or vertical
    return None


def moved_box(box, dx, dy, bounds=None) -> tuple:
    """``box`` shifted by ``(dx, dy)`` image pixels, kept inside ``bounds`` if given."""
    x1, y1, x2, y2 = _corners(box)
    width, height = x2 - x1, y2 - y1
    x1 += float(dx)
    y1 += float(dy)
    if bounds is not None:
        frame_w, frame_h = bounds
        x1 = max(0.0, min(x1, frame_w - width))
        y1 = max(0.0, min(y1, frame_h - height))
    return (x1, y1, x1 + width, y1 + height)


def resized_box(box, handle, point, minimum=1.0, bounds=None) -> tuple:
    """``box`` with the edges ``handle`` names dragged to ``point``.

    ``minimum`` is the smallest width/height kept, so a box can never be dragged through
    itself and out the other side - which is how an "edited" box becomes a zero-area one
    that no metric can score.
    """
    x1, y1, x2, y2 = _corners(box)
    x, y = _xy(point)
    edges = RESIZE_HANDLES.get(str(handle), ())
    if "left" in edges:
        x1 = x
    if "right" in edges:
        x2 = x
    if "top" in edges:
        y1 = y
    if "bottom" in edges:
        y2 = y
    if bounds is not None:
        frame_w, frame_h = bounds
        x1 = max(0.0, min(x1, frame_w))
        x2 = max(0.0, min(x2, frame_w))
        y1 = max(0.0, min(y1, frame_h))
        y2 = max(0.0, min(y2, frame_h))
    if x2 - x1 < minimum:
        if "left" in edges:
            x1 = x2 - minimum
        else:
            x2 = x1 + minimum
    if y2 - y1 < minimum:
        if "top" in edges:
            y1 = y2 - minimum
        else:
            y2 = y1 + minimum
    return (x1, y1, x2, y2)


def _normalised_box(box) -> tuple:
    """``(x1, y1, x2, y2)`` with the top-left corner first, whatever was passed.

    A box may arrive as ``(x1, y1, x2, y2)`` or as two ``(x, y)`` corners (a drag
    that started bottom-right is still the same rectangle), and callers should not
    have to care which.
    """
    if box is None:
        return None
    values = list(box)
    if len(values) == 4:
        ax, ay, bx, by = values
    else:
        (ax, ay), (bx, by) = values
    x1, x2 = sorted((float(ax), float(bx)))
    y1, y2 = sorted((float(ay), float(by)))
    return (x1, y1, x2, y2)


def box_iou(first, second) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes.

    Used to notice that the same animal was boxed twice, which is an easy mistake
    on a frame with several individuals and an expensive one: two boxes on one
    animal mean one animal is missing from the corpus.
    """
    if first is None or second is None:
        return 0.0
    ax1, ay1, ax2, ay2 = _normalised_box(first)
    bx1, by1, bx2, by2 = _normalised_box(second)
    overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = overlap_w * overlap_h
    if intersection <= 0.0:
        return 0.0
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1)
             - intersection)
    return intersection / union if union > 0 else 0.0


#: The measured grid is drawn in a different colour from the landmarks, because it
#: is evidence rather than input.
LATTICE_COLOR = "#ffb000"

#: The paper's circular field stop wears the same amber as the lattice it was measured
#: with: both are evidence of what the detector saw, and a second colour would suggest a
#: second measurement.
FIELD_STOP_COLOR = LATTICE_COLOR

#: The detected **reference points** are a colour of their own, so "what it measured from"
#: (the dots or cells it found) cannot be confused with "what it measured" (the amber
#: lattice fitted through them). Same hue family as `calibration_results.POINT_COLOR`.
REFERENCE_COLOR = "#dc3c3c"

#: Smallest a canvas or a stream panel may become. The *colours* of a panel are not
#: fixed here: they come from the theme (``canvas`` / ``canvas_text`` tokens), so a
#: light desktop no longer opens a black hole where the picture is, and a dark one
#: does not flash a white one. Asking the theme also means the panel changes with it
#: (see :meth:`ImageCanvas.event`).
PANEL_MINIMUM = (480, 260)


def frame_to_qimage(frame) -> QImage:
    """Convert a BGR or greyscale uint8 numpy frame into a (copied) QImage."""
    height, width = frame.shape[:2]
    if frame.ndim == 2:
        return QImage(frame.data, width, height, width, QImage.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
    return QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()


def qimage_to_frame(image: QImage) -> np.ndarray:
    """The numpy frame behind a ``QImage`` - the inverse of :func:`frame_to_qimage`.

    Greyscale comes back as ``(H, W)`` and anything else as BGR ``(H, W, 3)``, so a
    page that hands the canvas pixels it already has can still read them back as the
    same kind of array the pipeline produces. The result **own its pixels** (it is a
    copy): Qt frees an image's buffer with the image, and a numpy view of that buffer
    outliving it is a use-after-free that shows up as a shifted first row and nothing
    else - which is exactly how this function was wrong once.

    Raises:
        ImageLoadError: the image has no pixels. It is the same refusal a file that
            cannot be decoded gets, because from the canvas' side both mean one
            thing: there is no image to show.
    """
    if image.isNull() or image.width() <= 0 or image.height() <= 0:
        raise ImageLoadError("the image has no pixels (0 x 0)")
    if image.format() == QImage.Format_Grayscale8:
        rows = np.frombuffer(image.constBits(), dtype=np.uint8)
        rows = rows.reshape(image.height(), image.bytesPerLine())
        return rows[:, :image.width()].copy()          # numpy owns it (see above)
    rgb = image.convertToFormat(QImage.Format_RGB888)
    rows = np.frombuffer(rgb.constBits(), dtype=np.uint8)
    rows = rows.reshape(rgb.height(), rgb.bytesPerLine())
    pixels = rows[:, :rgb.width() * 3].reshape(rgb.height(), rgb.width(), 3)
    return pixels[:, :, ::-1].copy()                   # RGB -> BGR, and a copy


def _source_label(source) -> str:
    """How to name a source in a refusal: the file's own name, or its kind."""
    if isinstance(source, (str, os.PathLike)):
        return f"'{os.path.basename(os.fspath(source))}'"
    return f"that {type(source).__name__}"


def frame_from_source(source, allow_system_conversion=True):
    """The numpy frame behind **any** source a canvas may be given.

    One place decides what an image *is*, so no screen repeats the file reading and
    no screen has a second, weaker loader:

    * ``None`` - nothing (the canvas clears, and says so with its placeholder);
    * a numpy ``ndarray`` - already a frame (BGR or greyscale), as the pipeline and
      the stream produce;
    * a ``QImage`` or ``QPixmap`` - pixels the caller already has;
    * a ``str`` or ``os.PathLike`` - a file, read by
      :func:`daphnia.vision.images.load_image`, the one loader that also reads what a
      phone writes (``.HEIC``) and refuses a file by name when it cannot.

    Raises:
        ImageLoadError: a file that could not be read, or pixels with no size.
        TypeError: a source of a kind a canvas cannot show - a programming error, so
            it is raised rather than drawn.
    """
    if source is None:
        return None
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, QPixmap):
        return qimage_to_frame(source.toImage())
    if isinstance(source, QImage):
        return qimage_to_frame(source)
    if isinstance(source, (str, os.PathLike)):
        return load_image(source, allow_system_conversion=allow_system_conversion)
    raise TypeError(
        f"an image canvas cannot show {type(source).__name__} - give it a numpy "
        f"frame, a QImage/QPixmap, a file path, or None to clear it")


def fit_rect(image_width, image_height, box_width, box_height) -> QRect:
    """Largest rect with the image's aspect ratio that fits inside the box."""
    if image_width <= 0 or image_height <= 0 or box_width <= 0 or box_height <= 0:
        return QRect(0, 0, 0, 0)
    scale = min(box_width / image_width, box_height / image_height)
    width, height = int(image_width * scale), int(image_height * scale)
    return QRect((box_width - width) // 2, (box_height - height) // 2, width, height)


class ImageCanvas(QWidget):
    """Show one image, report clicks in *image* pixels, and let the operator zoom.

    The image is fitted (or magnified) into the widget and clicks are mapped back
    to the original pixels, so callers can feed the coordinates straight into the
    calibration/measurement maths at any magnification - which is what makes
    precise pitch-mark and landmark clicks possible.

    Interaction:
        * left click                        place the next landmark; in box mode it
                                            anchors a box corner (a second click, or
                                            a drag from it, finishes the box)
        * two-finger scroll                 pan, in both axes (never zooms)
        * two-finger pinch                  zoom towards the cursor
        * right/middle drag; **Command/Ctrl + left drag**; arrow keys   pan
        * ``+`` / ``-`` / ``F``             zoom in / out / fit the whole image
        * ``Esc``                           abandon a half-drawn box, or the selection
        * ``Delete``                        ask the page to remove the selected box

    **With ``box_edit`` on** (a page's *Select* tool) a press **selects and edits the boxes
    on the frame** instead of placing a label: inside a box moves it, an edge or a corner
    resizes it, and a drop reports the new geometry (`box_edited`) so the page can store it.
    A press inside a *predicted* box selects that animal (`prediction_selected`) - the model's
    answer is not a label, so it is chosen, not edited.

    **With ``editing`` on** (`X-N20`) that same pressing, moving, resizing and correcting is
    available in **every** tool, and the tool then only decides what a press on **empty space**
    does: *line*/*keypoints* place a point, *box* anchors a corner, and a *marquee* - a
    selection gesture, not a drawing one - takes **Shift** (or the *Select* tool). This is what
    a page with **draw** tools wants: clicking a box edits it, clicking empty space draws.

    **With ``point_mode`` on** (a page labelling one animal's keypoints on a crop) the press
    *is* the keypoint gesture: it picks up a point, or places the next one - a box is not a
    thing to move there, because in that work the box is the animal the landmarks belong to.

    Clicking more points than ``max_points`` starts a fresh point set.

    **Zoom is deliberate and pan is cheap.** Scrolling pans in both directions and
    cannot change the magnification; zooming is the pinch, the buttons or ``+``/
    ``-``/``F``. Since the click coordinates are what a model learns, an accidental
    zoom during labelling is an error, not an inconvenience. Holding Command (or
    Control) - or turning on ``pan_mode`` - while dragging pans without placing a
    point. Already-placed specimens are shown with :meth:`set_annotations`, each in
    its own colour (see :func:`specimen_color`); boxes stored for a frame are shown
    with :meth:`set_boxes`, and the box being drawn with :meth:`set_box`.
    """

    points_changed = Signal(list)

    #: Emitted when a **real press** placed a point (a click on the image), with the
    #: point list now on the canvas. ``points_changed`` also fires for a programmatic
    #: :meth:`set_points` (a page restoring a draft, a test driving the widget), so it
    #: cannot tell "the operator just clicked" - which is exactly what a tool that
    #: **auto-stores on the second click** has to know.
    point_placed = Signal(list)

    #: Emitted when the drawn bounding box changes: ``[(x1, y1, x2, y2)]`` or ``[]``.
    box_changed = Signal(list)

    #: Emitted on a double click - a queue-based page reads it as "next frame".
    double_clicked = Signal()

    #: Emitted when the interaction **tool** changes (``tool``). The page follows it to
    #: say what the next press will do - and so a tool offered in the header can never
    #: be left lying about which gesture is in force.
    tool_changed = Signal(str)

    #: Emitted when a press lands on a **stored object** the canvas can identify, with the tool
    #: that owns it (`TOOL_BOX`, `TOOL_LINE`, `TOOL_KEYPOINTS`). A page follows it by **switching
    #: its tool to that object's mode** without disturbing the press - so clicking a box in the
    #: Line tool both selects the box and arms the Box tool (`X-N19`).
    object_pressed = Signal(str)

    #: Emitted when a press lands on a **stored measurement** (an annotation) while
    #: ``annotation_edit`` is on: ``(specimen_index, segment_index, endpoint)``. The page
    #: loads exactly that row into the editable set - the canvas cannot, because a stored row
    #: is the page's data - and the *same* press then grabs the endpoint (see
    #: :meth:`_grab_annotation`), so a measurement drawn an hour ago is editable in one click
    #: (`X-N16`).
    annotation_point_selected = Signal(int, int, int)

    #: Emitted when the operator asks to remove a **measurement** (`Delete`/`Backspace`): the
    #: annotation under the last press, as ``(specimen_index, segment_index)``. The page
    #: decides - deleting a measurement is a data change (it drops the row on Dataset, or the
    #: item in *this view* on Bench), and the canvas stores nothing (`X-N15`).
    annotation_delete_requested = Signal(int, int)

    #: Emitted when the corner a box is being drawn from is anchored (True) or
    #: released (False): a page can say what the next click will do without the
    #: event ever being mistaken for "a box was made".
    box_anchor_changed = Signal(bool)

    #: Emitted when the magnification changes (fit = 1.0). The on-canvas tools read
    #: it to *show* the magnification, so the operator can see how much the pixels
    #: a landmark is placed between are stretched.
    zoom_changed = Signal(float)

    #: Emitted when a source could not be shown, with the reason (also drawn on the
    #: canvas). A page wires it to its own status label instead of catching the
    #: loader's error per file.
    source_failed = Signal(str)

    #: Emitted when a **stored** box is selected or deselected, with its specimen index
    #: (``-1`` when nothing is selected). The page follows it - the dataset surface sets
    #: the specimen dropdown to the animal whose box was clicked.
    box_selected = Signal(int)

    #: Emitted when a selected box was **edited**: ``(specimen index, (x1, y1, x2, y2))``.
    #: Only the page stores it - a drag is geometry until the drop.
    box_edited = Signal(int, tuple)

    #: Emitted when the operator asks to remove the selected box (`Delete`/`Backspace`).
    #: The page decides: deleting a label is a data change, and the canvas stores nothing.
    box_delete_requested = Signal(int)

    #: Emitted when a **predicted** box is selected (the index into the prediction layer).
    #: A prediction cannot be edited as a box - it is not a label - but *which animal the
    #: model was most sure about* is a real choice, and the page hands that one to the draft.
    prediction_selected = Signal(int)

    #: Emitted when a promotion of a model's box **finishes** - the press that grabbed a
    #: prediction was let go, whether or not it moved: ``(index, box, points, moved)``. The page
    #: creates the measurement/label **from this** - the final, dragged geometry - instead of from
    #: the untouched suggestion, which is what stops the "ghost at the old position" (`X-N15`).
    #: The prediction itself is only dropped by the canvas when it **moved** (`box_changed`
    #: follows); a page that accepts a *click* too drops it with :meth:`drop_prediction`.
    prediction_accepted = Signal(int, tuple, tuple, bool)

    #: Emitted when a drag of an **editable point** ends (`Backspace` in a pair too). The page
    #: stores the corrected label when it already exists - "letting go" is a correction, and a
    #: correction that needs a second key press is a correction the operator forgets to store
    #: (`SR-DATASET-027`).
    points_edited = Signal()

    #: Emitted while an **existing point is being moved** (`Edit`), on every step of the drag.
    #: The page only has to look again - the canvas already holds the new position. A page that
    #: treats this as a placement overwrites the drag and the point snaps back (`SR-DATASET-027`).
    points_moved = Signal()

    #: Emitted when the operator removes a **prediction** (`Backspace` on a picked model box):
    #: "the model is wrong about this one". Nothing is stored, so the page only drops it from
    #: the layer it drew (`SR-DATASET-027`).
    prediction_removed = Signal(int)

    #: Zoom range, relative to "fit" (1.0 shows the whole image).
    MIN_ZOOM = 1.0
    MAX_ZOOM = 40.0
    #: Multiplicative zoom applied by the ``+`` / ``-`` buttons and one wheel notch.
    ZOOM_STEP = 1.25
    #: A wheel/scroll delta that counts as one "notch" (Qt reports 120 per notch).
    SCROLL_REFERENCE = 120.0
    #: A wheel "notch" (120 units) moves the view this many widget pixels.
    WHEEL_PAN_STEP = 60.0

    #: Pixels moved per arrow-key press.
    PAN_STEP = 40
    #: A box smaller than this (in image pixels) is a mis-click, not a specimen.
    #: Aliased to :data:`daphnia.annotation.MIN_BOX_PIXELS` so the labeller, the
    #: export reader and the corpus repair all agree on "too small to be real".
    MIN_BOX_PIXELS = MIN_BOX_PIXELS

    def __init__(self, max_points=2, point_names=None, parent=None):
        super().__init__(parent)
        self.max_points = max_points
        self.point_names = list(point_names or [])
        self._pixmap = None
        self._frame = None          # the numpy frame of the image on screen (or None)
        self._refusal = None        # why there is no image (drawn instead of it)
        self._points = []
        #: ``[(index, [(p1, p2), ...], text, tags, texts)]`` - see `set_annotations`:
        #: the specimen, its segments, the item's own label, the per-segment tags, and
        #: the per-segment labels (`annotation_texts`).
        self._annotations = []
        self._lattice = None
        self._field_stop = None     # (centre_x, centre_y, radius) of the paper's circle
        self._reference_points = None   # the detected points of a grid measurement, (N, 2)
        self._zoom = self.MIN_ZOOM
        self._pan = QPointF(0.0, 0.0)
        self._panning = None
        self.pan_mode = False       # a hand tool: left-drag pans, clicks do not mark
        self.show_point_labels = True
        self.connect_points = True  # draw a line between the active points?
        #: Which pairs of the *active* points are joined by a **bone** - a tuple of
        #: ``(first_index, second_index)`` into :attr:`point_names`. Empty means "join
        #: the first two" (a line tool's single interval); a keypoint page sets the
        #: schema's skeleton so the head->tail and left->right bones are drawn instead
        #: of a polyline through every click (`annotation.KeypointSystem.edges`).
        self.skeleton = ()
        self.show_annotation_index = True   # number each annotated specimen
        self.show_annotation_text = True    # draw each annotation's label, if it has one
        self.box_mode = False       # a left-drag draws the specimen's bounding box
        # **`point_mode`** (a page labelling keypoints on a crop): the press *is* the
        # keypoint gesture - it picks up a point, or places the next one - and the boxes
        # on the frame are what the animal *is*, not a thing to move. In the crop loop the
        # item's box covers the whole picture, so with the ordinary order every click
        # selected that box and no keypoint could be placed at all ("not able to correct
        # the keypoint - it first selects the box"); a box's geometry is judged in *Frame*
        # view, where this mode is off (`SR-DATASET-025/027`).
        self.point_mode = False
        self._box = None            # the active box: (x1, y1, x2, y2) in image px
        self._boxes = []            # [(specimen_index, (x1, y1, x2, y2))] overlays
        self._predictions = []      # [(box, points, label)] - what a model guessed
        # Selecting and editing the boxes on the frame (a page's *Select* tool): with
        # `box_edit` on, a press picks a box instead of placing a label - see
        # `mousePressEvent`. `_selected` is the *stored* box being edited (a specimen
        # index), `_selected_prediction` the model's box that was picked.
        self.box_edit = False
        #: With **``editing``** on, **clicking an object is always available** - selecting,
        #: moving, resizing, correcting or deleting it - whatever draw tool is in force (`X-N20`).
        #: That is the *standard* way to touch what is on the canvas, so a page sets it once and
        #: the draw tools stay about **drawing**; a bare widget leaves it off.
        self.editing = False
        #: With **``box_edge_grab``** on, a stored box is grabbed by its **outline** only (the
        #: edge band of :func:`box_edge_at`), never by a press deep inside it. A page whose boxes
        #: are **measurements** (the Bench) turns this on, so a rectangle drawn around an animal
        #: no longer swallows the clicks meant for the keypoints and lines inside it; the corpus
        #: leaves it off, because there a box *is* the animal (`X-N17`).
        self.box_edge_grab = False
        #: The interaction tool in force when no box tool is chosen (`tool`). `line` and
        #: `keypoints` are create tools that place points; the getter above folds in
        #: `box_edit`/`box_mode` so `tool` always says what a press will do.
        self._tool = TOOL_LINE
        self._selected = None           # the stored-box index under edit, or None
        self._selection = []            # every selected stored box, primary last
        self._selected_prediction = None    # the model's box that was picked, or None
        self._selected_point = None     # the editable point being dragged, or None
        #: What is **under the cursor** right now - ``(kind, index)`` as
        #: :meth:`_hit_editable` reports it, or ``None``. It is what the cursor shape and the
        #: hover ring follow, so *"the thing I can grab"* is visible before the press
        #: (`X-N15`) rather than discovered by clicking.
        self._hover = None
        #: The **measurement** (annotation) the last press landed on, ``(specimen, segment)``,
        #: or ``None`` - the primary one, kept for compatibility. Every selected measurement is
        #: in :attr:`_selected_annotations`.
        self._selected_annotation = None
        #: Every selected **measurement** (annotation), as ``(specimen, segment)`` pairs.
        self._selected_annotations = []
        #: Every selected **prediction** index. Predictions are selectable objects too: a
        #: selection can hold boxes, measurements and predictions, and `Backspace` removes
        #: everything selected (`X-N15`).
        self._selected_predictions = []
        #: With **``annotation_edit``** on, the **stored measurements** drawn as annotations
        #: (a Bench line, a Dataset keypoint pair) are hittable: hovering one shows it and a
        #: press reports the exact ``(specimen, segment, endpoint)`` so the page can load that
        #: row and let it be corrected (`X-N16`). Off by default: a page that never stores a
        #: measurement into an overlay must not have one swallow its clicks.
        self.annotation_edit = False
        self._marquee = None            # the rubber band while it is being dragged
        self._drag = None               # what the press started - see `_begin_edit`
        self._box_origin = None     # the corner a drag started from
        self.placeholder = "No image"
        self.setMinimumSize(*PANEL_MINIMUM)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setToolTip("Pinch to zoom, two-finger scroll to pan, "
                        "\u2318/Ctrl-drag to pan, F to fit.")

    # -- the tool ----------------------------------------------------------
    @property
    def tool(self) -> str:
        """What a left press means right now - one of :data:`TOOLS`.

        A box or a selection *is* a tool the operator asked for, so the plain
        ``box_edit``/``box_mode`` flags (which older pages still set) are folded in here:
        with the box tool on it is :data:`TOOL_BOX`, with editing on it is
        :data:`TOOL_SELECT`, and otherwise it is the stored create tool (``line`` or
        ``keypoints``).
        """
        if self.box_mode:
            return TOOL_BOX
        if self.box_edit:
            return TOOL_SELECT
        return self._tool

    def set_tool(self, tool: str, keep=False):
        """Choose the interaction tool; a create tool is the fallback for an unknown key.

        This is what a page's *Tool* menu calls. It sets the two older flags the canvas
        already reads (``box_edit`` for select/edit, ``box_mode`` for drawing boxes), so
        one named tool drives the whole press-handling path - and it drops any selection
        the previous tool was holding, because a selection is a thing the *select* tool
        owns.

        ``keep`` is for a page that **follows an object the operator just clicked** (`X-N19`):
        the tool changes while the press's own work - the selection, the drag, the points
        already placed - is left exactly as it is, so one gesture can both select a box and arm
        the Box tool.

        These flags are for a page whose tools *are* the modes (the corpus' *Edit* tool). A page
        that offers **draw** tools leaves ``editing`` on instead (`X-N20`), and its tools then
        change only what a press on empty space draws.
        """
        tool = tool if tool in TOOLS else TOOL_LINE
        self._tool = tool
        self.box_edit = tool in (TOOL_SELECT, TOOL_BOX)
        self.box_mode = tool == TOOL_BOX
        if not keep:
            self.clear_selection()
        self.tool_changed.emit(self.tool)
        self.update()

    # -- content -----------------------------------------------------------
    def set_source(self, source, reset_view=True, allow_system_conversion=True):
        """Show **one** image, from whatever the caller has; returns its numpy frame.

        This is the canvas' only way in, so no screen repeats the file reading and no
        screen has a second, weaker loader. ``source`` may be:

        * a **numpy frame** (BGR or greyscale) - what the pipeline and the stream
          produce;
        * a **file path** - read by :func:`daphnia.vision.images.load_image`, the one
          loader that also reads what a phone writes (``.HEIC``);
        * a **``QImage``/``QPixmap``** - pixels the caller already has;
        * ``None`` - nothing, which *clears* the canvas to its placeholder.

        Whatever arrives, it replaces what was there: the previous image, its points,
        its box, its overlays and its measured grid are gone. A page therefore shows
        one image, and what is on the canvas is what the numbers belong to.

        A source that cannot be used does not leave a stale picture behind and does
        not blank the canvas in silence either: the reason is *drawn* where the image
        would be (see :attr:`refusal`) and emitted as :attr:`source_failed`, so a
        screen can put it in its own panel with
        ``canvas.source_failed.connect(self.status_label.setText)`` and no per-file
        ``try``/``except`` of its own.

        ``reset_view=False`` keeps the current zoom and pan: a live view needs that,
        because re-fitting many times a second would throw away the magnification the
        operator just chose.

        Returns:
            The numpy frame now on the canvas, or ``None`` when nothing is shown
            (cleared, or refused).

        Raises:
            TypeError: ``source`` is of a kind a canvas cannot show. That is a
                programming error, so it is raised rather than drawn.
        """
        try:
            frame = frame_from_source(
                source, allow_system_conversion=allow_system_conversion)
        except ImageLoadError as error:
            self._refuse(_source_label(source), str(error))
            if reset_view:
                self.reset_view()
            self.clear_box()
            self.set_boxes([])
            self.set_annotations([])
            self._reset()
            return None

        self._show(frame, reset_view)
        return frame

    def _show(self, frame, reset_view):
        """Put ``frame`` (or nothing) on the canvas, without any extra check."""
        self._frame = frame
        self._refusal = None
        self._pixmap = (None if frame is None
                        else QPixmap.fromImage(frame_to_qimage(frame)))
        if reset_view:
            self.reset_view()
        self.clear_box()
        self.set_boxes([])
        self.set_annotations([])      # another image's markings are not this one's
        self._reset()

    def _refuse(self, label, reason):
        """Record why nothing can be shown; the placeholder gives way to the reason.

        The message is the whole story - *what* could not be loaded and *why* - because
        it is also what a screen puts in its status line.
        """
        self._frame = None
        self._pixmap = None
        self._refusal = f"Could not load {label} - {reason}"
        self.source_failed.emit(self._refusal)

    @property
    def frame(self):
        """The numpy frame of the image on the canvas, or ``None`` when there is none.

        The same array ``set_source`` was given or loaded, so a page that shows a
        source and then needs its pixels (a crop, a detection, a grid) can read them
        back instead of loading the file a second time.
        """
        return self._frame

    @property
    def refusal(self):
        """Why nothing is shown, or ``None`` - also drawn on the canvas."""
        return self._refusal

    def set_points(self, points):
        """Set the markers directly (e.g. when re-opening an annotated frame)."""
        self._points = [(float(x), float(y)) for x, y in points][:self.max_points]
        self.points_changed.emit(self.points)
        self.update()

    def clear_points(self):
        self._reset()

    def _reset(self):
        self._points = []
        self._lattice = None
        self._field_stop = None
        self._reference_points = None
        self.points_changed.emit([])
        self.update()

    # -- overlay -----------------------------------------------------------
    def set_lattice(self, period_px, angle_deg=0.0):
        """Overlay the measured grid, so the scale can be *seen* and not just trusted.

        ``period_px`` is the spacing and ``angle_deg`` the rotation of the line
        family, both in image pixels - exactly what
        :func:`daphnia.vision.lattice.measure` reports. ``None`` (or 0) clears it.
        """
        self._lattice = None if not period_px else (float(period_px),
                                                    float(angle_deg))
        self.update()

    def set_field_stop(self, centre_x=None, centre_y=None, radius=None):
        """Overlay the circular paper area the scale was measured **inside**.

        A phone frame through an eyepiece has a hard black surround: the paper is a disc,
        and everything the detector reads comes from inside it. Drawing it is the other
        half of the evidence (`SR-OPTICS-026`) - the lattice says *what* was measured, the
        circle says *where it was possible to measure it*. Any argument left out (or a
        radius of 0) clears the overlay, so it has exactly the lifecycle of
        :meth:`set_lattice`: it can never describe a photo it did not come from.
        """
        self._field_stop = (None if not radius
                            else (float(centre_x), float(centre_y), float(radius)))
        self.update()

    @property
    def field_stop(self):
        """``(centre_x, centre_y, radius)`` of the drawn circle, or ``None``."""
        return self._field_stop

    @property
    def lattice(self):
        """``(period_px, angle_deg)`` of the drawn grid, or ``None``."""
        return self._lattice

    def set_reference_points(self, points):
        """Ring the **detected reference points** of a grid measurement on the photo.

        ``points`` is an iterable of ``(x, y)`` in image pixels - exactly what
        :func:`daphnia.vision.points.reference_points` returns - so pressing *Measure grid*
        leaves the evidence on screen: *what the measurement measured from*, not only the
        lattice fitted through it. ``None`` or empty clears it, so it has the lifecycle of
        :meth:`set_lattice` and can never describe a photo it did not come from.
        """
        # `points` is a numpy ``(N, 2)`` array as often as a list, and ``array or []`` raises:
        # the emptiness test must come first, and be on the *list* that was built.
        cleaned = [] if points is None else [(float(x), float(y)) for x, y in points]
        self._reference_points = cleaned or None
        self.update()

    @property
    def reference_points(self):
        """``[(x, y)]`` of the detected points drawn, or ``None``."""
        return self._reference_points

    def set_annotations(self, annotations):
        """Show the specimens already annotated on this frame, behind the active set.

        ``annotations`` is an iterable of ``(specimen_index, segments)`` pairs (or
        dicts with those keys), where ``segments`` is a list of ``(point_a, point_b)``
        interval pairs - the *measured* relationships. That is what makes the width
        draw as a left-to-right line and the length as a head-to-tail line, each
        specimen in its own :func:`specimen_color`. A point with no partner yet is
        passed as a zero-length segment and drawn as a dot.

        An item may carry a third element (or a ``"text"`` key): a **label** drawn
        above the specimen's first point, on a dark plate so it stays readable over a
        pale or a dark frame. The Predict surface uses it for the measured length, so
        the number sits on the animal it belongs to.

        A fourth element (or a ``"tags"`` key) is a parallel list of **opaque tags**, one
        per segment - how a page says *which measurement a segment is* (Bench passes the
        trait). With tags, a press on that segment is reported back as a concrete
        ``(specimen, segment, endpoint)`` (`annotation_point_selected`), and the page loads
        exactly that row to correct it (`X-N16`).

        A fifth element (or a ``"texts"`` key) is a parallel list of **per-segment
        labels**, one per segment. With it, every segment carries its **own** label, drawn
        above *that* segment (over its middle) instead of once for the whole item at its
        first point - which is what a page needs when each of an animal's numbers is its
        own measurement with its own name (the Bench, `SR-BENCH-024`). A ``None``/empty
        entry means *no label for that segment*; the item's own ``text`` is then only used
        for the segments that have none, so a page may mix the two.
        """
        cleaned = []
        for item in annotations or []:
            tags = ()
            texts = ()
            if isinstance(item, dict):
                index = item.get("index", 0)
                segments = item.get("segments") or []
                text = item.get("text") or ""
                tags = item.get("tags") or ()
                texts = item.get("texts") or ()
            elif len(item) >= 5:
                index, segments, text, tags, texts = item[0], item[1], item[2], item[3], item[4]
            elif len(item) >= 4:
                index, segments, text, tags = item[0], item[1], item[2], item[3]
            elif len(item) >= 3:
                index, segments, text = item[0], item[1], item[2]
            else:
                index, segments = item
                text = ""
            pairs = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                     for a, b in segments]
            tags = list(tags) + [None] * (len(pairs) - len(tags))
            texts = [str(value or "") for value in texts]
            texts += [""] * (len(pairs) - len(texts))
            cleaned.append((int(index), pairs, str(text or ""), tuple(tags),
                            tuple(texts)))
        self._annotations = cleaned
        self.update()

    @property
    def annotations(self) -> list:
        """The ``(specimen_index, segments)`` pairs currently drawn as overlays."""
        return [(index, pairs) for index, pairs, _text, _tags, _texts in self._annotations]

    @property
    def annotations_with_text(self) -> list:
        """The ``(specimen_index, segments, text)`` triples, labels included."""
        return [(index, pairs, text)
                for index, pairs, text, _tags, _texts in self._annotations]

    @property
    def annotation_texts(self) -> list:
        """The ``(specimen_index, [one label per segment])`` pairs - the per-line labels.

        Empty strings where a segment carries none, so the positions still line up with
        the segments of :attr:`annotations`.
        """
        return [(index, list(texts))
                for index, _pairs, _text, _tags, texts in self._annotations]

    # -- the bounding box --------------------------------------------------
    def set_box(self, box):
        """Set the active box (``(x1, y1, x2, y2)`` in image pixels, or ``None``)."""
        self._box = None if box is None else _normalised_box(box)
        self._set_box_origin(None)
        self.box_changed.emit(self.boxes_drawn)
        self.update()

    def clear_box(self):
        """Forget the active box and any half-drawn one (the overlays stay)."""
        self.set_box(None)

    def _set_box_origin(self, origin):
        """Anchor (or release) the corner the box is being drawn from, and say so."""
        had = self._box_origin is not None
        self._box_origin = origin
        if (origin is not None) != had:
            self.box_anchor_changed.emit(origin is not None)

    @property
    def box_pending(self) -> bool:
        """True when one corner is anchored and the opposite one is still to come."""
        return self._box_origin is not None

    @property
    def box_origin(self):
        """The anchored corner ``(x, y)`` in image pixels, or ``None``."""
        return self._box_origin

    @property
    def box(self):
        """The active box ``(x1, y1, x2, y2)`` in image pixels, or ``None``."""
        return self._box

    @property
    def boxes_drawn(self) -> list:
        """``[(x1, y1, x2, y2)]`` when a box is drawn, ``[]`` otherwise."""
        return [self._box] if self._box is not None else []

    def set_boxes(self, boxes):
        """Show the boxes already stored for this frame, behind the active one.

        ``boxes`` is an iterable of ``(specimen_index, (x1, y1, x2, y2))`` pairs (or
        dicts with ``index``/``box`` keys), drawn in each specimen's own colour - the
        same convention :meth:`set_annotations` uses for intervals.
        """
        cleaned = []
        for item in boxes or []:
            if isinstance(item, dict):
                index = item.get("index", 0)
                box = item.get("box")
            else:
                index, box = item
            if box is None:
                continue
            cleaned.append((int(index), _normalised_box(box)))
        self._boxes = cleaned
        self.update()

    @property
    def boxes(self) -> list:
        """The stored boxes currently drawn as overlays."""
        return list(self._boxes)

    def set_predictions(self, predictions):
        """Show what a **model** guessed on this frame, in its own layer (`SR-PREDICT-018`).

        ``predictions`` is an iterable of items with ``box``, ``points`` (keypoints in
        **source pixels**, in the model's own order) and an optional ``label`` - a
        :class:`~daphnia.ui.model_pass.Prediction`, a dict, or a plain
        ``(box, points, label)`` tuple.

        Drawn **dotted, in one colour** (`theme.qcolor("prediction")`), so it can never be
        mistaken for a stored label: those wear a specimen's own colour and a dashed outline,
        and the box being dragged wears the marker colour. A prediction is not ground truth
        (`mlContext.md` law 7) and it must not look like it. Nothing here is ever written to
        the database - clearing the layer is `set_predictions([])`.
        """
        cleaned = []
        for item in predictions or []:
            if isinstance(item, dict):
                box, points = item.get("box"), item.get("points") or ()
                text = item.get("label") or ""
            elif hasattr(item, "box"):
                box, points = item.box, getattr(item, "points", ())
                text = getattr(item, "label", "") or ""
            elif len(item) >= 3:
                box, points, text = item[0], item[1], item[2]
            else:
                box, points = item
                text = ""
            if box is None:
                continue
            cleaned.append((_normalised_box(box),
                            tuple((float(x), float(y)) for x, y in points),
                            str(text or "")))
        self._predictions = cleaned
        self.update()

    @property
    def predictions(self) -> list:
        """The model's boxes (and keypoints) currently drawn as overlays."""
        return list(self._predictions)

    # -- selecting and editing the boxes on the frame ----------------------
    def has_selectable(self) -> bool:
        """Whether a box is on the frame to pick (`box_edit`'s tool is offered then)."""
        return bool(self._boxes or self._predictions)

    @property
    def selected_box(self):
        """The **stored** box index being edited, or ``None``."""
        return self._selected

    @property
    def selected_prediction(self):
        """The index into the prediction layer that was picked, or ``None``."""
        return self._selected_prediction

    def select_box(self, index, additive=False):
        """Select a stored box by specimen index (``None`` deselects), and say so.

        ``additive`` (Shift/Cmd) **adds to** the selection instead of replacing it - the
        standard multi-selection gesture - and the box just clicked becomes the primary one:
        it is the one whose handles are drawn and whose specimen the page follows. Selecting a
        box clears any selected measurement or prediction, so one press means one thing.
        """
        self._selected_annotations = []
        self._selected_annotation = None
        self._selected_predictions = []
        if index is None:
            chosen = []
        else:
            index = int(index)
            if additive and index in self._selection:
                chosen = [item for item in self._selection if item != index]
            elif additive:
                chosen = [item for item in self._selection if item != index] + [index]
            else:
                chosen = [index]
        self._selection = chosen
        self._selected = chosen[-1] if chosen else None
        self._selected_prediction = None
        self.box_selected.emit(-1 if self._selected is None else self._selected)
        self.update()
        return self._selected

    @property
    def selected_boxes(self) -> list:
        """Every stored box index in the selection, in the order they were added."""
        return list(self._selection)

    def _primary(self):
        """The box whose handles are drawn: the last one added to the selection."""
        return self._selection[-1] if self._selection else None

    def select_point(self, index):
        """Pick one of the **editable points** (`_points`) so it can be dragged or removed."""
        self._selected_point = None if index is None else int(index)
        self.update()
        return self._selected_point

    @property
    def selected_point(self):
        """The editable point under edit, or ``None``."""
        return self._selected_point

    def select_prediction(self, index, additive=False):
        """Pick a model's box (``None`` deselects); ``additive`` (Shift/Cmd) adds or removes it.

        A prediction is a **selectable object**: several can be held at once (Shift/Cmd-click or
        a marquee) and `Backspace` removes them from the layer. Selecting a prediction clears any
        selected box or measurement, so one press means one thing.
        """
        self._selection = []
        self._selected = None
        self._selected_annotations = []
        self._selected_annotation = None
        if index is None or int(index) < 0:
            self._selected_predictions = []
        else:
            index = int(index)
            if additive and index in self._selected_predictions:
                self._selected_predictions = [item for item in self._selected_predictions
                                              if item != index]
            elif additive:
                self._selected_predictions = (
                    [item for item in self._selected_predictions if item != index] + [index])
            else:
                self._selected_predictions = [index]
        self._selected_prediction = (self._selected_predictions[-1]
                                     if self._selected_predictions else None)
        self.prediction_selected.emit(
            -1 if self._selected_prediction is None else self._selected_prediction)
        self.update()
        return self._selected_prediction

    @property
    def selected_predictions(self) -> list:
        """Every selected prediction index, in the order they were picked."""
        return list(self._selected_predictions)

    def select_annotation(self, specimen, segment, additive=False):
        """Select a **measurement** (a drawn annotation); ``additive`` adds or removes it.

        Several measurements can be selected at once, and `Backspace` asks the page to drop
        every one of them (`annotation_delete_requested`). Selecting a measurement clears any
        selected box or prediction.
        """
        self._selection = []
        self._selected = None
        self._selected_predictions = []
        self._selected_prediction = None
        entry = (int(specimen), int(segment))
        if additive and entry in self._selected_annotations:
            self._selected_annotations = [item for item in self._selected_annotations
                                          if item != entry]
        elif additive:
            self._selected_annotations = (
                [item for item in self._selected_annotations if item != entry] + [entry])
        else:
            self._selected_annotations = [entry]
        self._selected_annotation = (self._selected_annotations[-1]
                                     if self._selected_annotations else None)
        self.update()
        return self._selected_annotation

    @property
    def selected_annotations(self) -> list:
        """Every selected measurement, as ``(specimen, segment)`` pairs."""
        return list(self._selected_annotations)

    def clear_selection(self):
        """Forget the selection, and tell the page it is gone."""
        had_box = bool(self._selection)
        had_prediction = bool(self._selected_predictions)
        had_point = self._selected_point is not None
        had_annotation = bool(self._selected_annotations)
        self._selection = []
        self._selected = None
        self._selected_prediction = None
        self._selected_predictions = []
        self._selected_point = None
        self._selected_annotation = None
        self._selected_annotations = []
        if had_box:
            self.box_selected.emit(-1)
        if had_prediction:
            self.prediction_selected.emit(-1)
        if had_box or had_prediction or had_point or had_annotation:
            self.update()

    def _screen_box(self, box):
        """A box in image pixels as the rectangle it occupies in this widget."""
        if self._pixmap is None or box is None:
            return None
        rect = self._display_rect()
        scale = rect.width() / self._pixmap.width()
        x1, y1, x2, y2 = (float(value) for value in box)
        return QRectF(rect.x() + x1 * scale, rect.y() + y1 * scale,
                      (x2 - x1) * scale, (y2 - y1) * scale)

    def _image_bounds(self):
        """``(width, height)`` of the image on screen - the frame a box is kept inside."""
        if self._pixmap is None:
            return None
        return (float(self._pixmap.width()), float(self._pixmap.height()))

    def _to_screen(self, point):
        """An image pixel as a point in this widget, or ``None`` without an image."""
        if self._pixmap is None or point is None:
            return None
        rect = self._display_rect()
        scale = rect.width() / self._pixmap.width()
        return (rect.x() + float(point[0]) * scale, rect.y() + float(point[1]) * scale)

    def box_of(self, index):
        """The stored box of one specimen index, or ``None`` - the box being edited."""
        for specimen, box in self._boxes:
            if specimen == index:
                return box
        return None

    def _replace_box(self, index, box):
        """Put an edited box back into the overlay list (the page stores it on the drop)."""
        self._boxes = [(specimen, box if specimen == index else existing)
                       for specimen, existing in self._boxes]

    @staticmethod
    def _point_pair(position):
        """A mouse position as ``(x, y)`` floats: the hit tests work in either space."""
        return (float(position.x()), float(position.y()))

    def _hit_editable(self, position):
        """``(kind, index)`` of the editable thing under ``position``, or ``None``.

        The one place the *"what can I grab here?"* question is answered, so a press and a
        hover can never disagree about it. Front to back, exactly the paint order:

        * ``("point", i)`` - one of the active keypoints (``_points`` - a landmark, a spine);
        * ``("box", specimen)`` - a stored box on the frame;
        * ``("prediction", i)`` - a model's box in the prediction layer.
        """
        index = self._hit_point(position)
        if index is not None:
            return ("point", index)
        if self.annotation_edit:
            hit = self._hit_annotation(position)
            if hit is not None:
                return ("annotation", hit[0])
        if self.point_mode:
            # In the crop the box *is* the animal: a press that is not on a point places the
            # next landmark, so nothing else is grabbable there (`SR-DATASET-028`).
            return None
        index = self._hit_stored(position)
        if index is not None:
            return ("box", index)
        if self._predictions:
            index = self._hit_prediction(position)
            if index is not None:
                return ("prediction", index)
        return None

    @property
    def hover(self):
        """``(kind, index)`` of the editable thing under the cursor, or ``None``.

        What the next press would grab (`X-N15`): ``("point", i)``, ``("box", specimen)`` or
        ``("prediction", i)``. A page can read it; a test uses it to prove the hover feedback
        exists without inspecting pixels.
        """
        return self._hover

    def _set_hover(self, target):
        """Remember what is under the cursor, and show it (a ring, and a grab cursor)."""
        target = tuple(target) if target else None
        if target == self._hover:
            return
        self._hover = target
        self._update_hover_cursor()
        self.update()

    def _update_hover_cursor(self):
        """Show that the thing under the cursor can be grabbed - unless panning owns it."""
        if self.pan_mode or self._panning is not None:
            return
        if self._hover is None:
            self.unsetCursor()
        else:
            self.setCursor(Qt.OpenHandCursor)

    def _start_edit(self, position, modifiers=Qt.NoModifier) -> bool:
        """Start editing whatever is **under** ``position`` - or say there is nothing there.

        The shared half of a press. It is asked first with the *Select* tool on (where a
        marquee follows when nothing is under it) **and** with it off, so a create tool can
        still let the operator **click a nearby target to edit it right away** instead of
        switching tools first (`X-N15`). Returns whether the press was used, so the caller
        knows whether the tool's own gesture (place a point, anchor a box) may also run.

        The order of the questions is the order of what is on screen, front to back:

        1. a **handle** of the primary selected box - resize it;
        2. an editable **point** - drag that point (one keypoint at a time);
        3. a **stored measurement** (an annotation) - report it so the page loads that row, then
           drag the end (`annotation_edit`, `X-N16`);
        4. a **stored box** - click selects it (Shift/Cmd adds to the selection), dragging
           moves the **whole selection**;
        5. a **prediction** - it is picked *and* promoted to the editable draft, so dragging
           a model's box corrects it and letting go stores it as an annotation.

        With **``point_mode``** on (a page labelling keypoints on a crop) only the first two
        questions are asked: a press that is not on a point is a **placement** (the press is
        handed back to :meth:`mousePressEvent`), because there the box is the animal and the
        click is the landmark - see :attr:`point_mode`.
        """
        image_point = self._to_image(position)
        if image_point is None:
            return False
        where = self._point_pair(position)
        additive = bool(modifiers & (Qt.ShiftModifier | Qt.ControlModifier
                                     | Qt.MetaModifier))
        # 1. A handle of the primary selected box: resize it.
        primary = self._primary()
        if primary is not None:
            box = self.box_of(primary)
            if box is not None:
                handle = resize_handle(where, self._screen_box(box), BOX_GRAB_PIXELS)
                if handle:
                    self._drag = {"kind": "resize", "start": image_point, "handle": handle,
                                  "indices": [primary], "boxes": {primary: box},
                                  "grab": True}
                    return True
        # 2. An editable point: drag it (a keypoint is dragged one at a time).
        index = self._hit_point(position)
        if index is not None:
            self._selected_annotation = None
            self._drag = {"kind": "point", "start": image_point, "index": index,
                          "point": self._points[index], "grab": True}
            self.select_point(index)
            return True
        if self.point_mode:
            # The rest of the frame is where the landmarks go: not a box to pick, not a
            # marquee to drag, and the stored pair is consumed by the click-click loop rather
            # than reopened (`SR-DATASET-028`). `mousePressEvent` places the point.
            return False
        if self._grab_annotation(position, additive):
            # 3. A **stored measurement** drawn as an annotation: the page loaded its row on
            #    `annotation_point_selected`, and the end is now being dragged - one press to
            #    edit a line or a keypoint set that was stored earlier (`X-N16`). With Shift it
            #    is only added to (or removed from) the selection.
            self.object_pressed.emit(TOOL_LINE)
            return True
        # 4. A stored box: select or add to the selection, then move what is selected.
        index = self._hit_stored(position)
        if index is not None:
            if additive:
                self.select_box(index, additive=True)
                self.object_pressed.emit(TOOL_BOX)
                return True
            if index not in self._selection:
                self.select_box(index)
            self._drag = {"kind": "move", "start": image_point,
                          "indices": list(self._selection),
                          "boxes": {item: self.box_of(item) for item in self._selection},
                          "grab": True}
            self.object_pressed.emit(TOOL_BOX)
            return True
        # 5. A model's box: pick it **and** put it in the draft, so it can be corrected and
        #    stored - a prediction becomes an annotation by being dragged. The promotion is
        #    **silent** on purpose: `set_box` reports a completed box, and a press is not one -
        #    emitting here stored the untouched prediction as specimen N *and* the corrected one
        #    as N+1 (measured: one drag, two labels).
        if self._predictions:
            index = self._hit_prediction(position)
            if index is not None:
                # A prediction is an object you can **select** (`X-N15`): Shift/Cmd adds it to
                # the selection, and `Backspace` drops it from the layer. Only a **create** tool
                # turns it into a draft the same press can drag into a label (`X-N14`); in the
                # *Select* tool a press selects and nothing is promoted by merely clicking.
                select_tool = self.tool == TOOL_SELECT      # decided before the tool may switch
                self.select_prediction(index, additive=additive)
                item = self._predictions[index]
                # What the operator actually **aimed at** decides what the release makes: a press
                # on one of the model's keypoints makes a line, a press on the box itself makes a
                # box - a detection + pose prediction carries both (`X-N15`).
                aimed = self._prediction_points_at(item, position)
                self.object_pressed.emit(TOOL_KEYPOINTS if aimed else TOOL_BOX)
                if select_tool:
                    self.update()
                    return True
                box = item[0]
                self._box = _normalised_box(box) if box is not None else None
                self._set_box_origin(None)
                self._drag = {"kind": "draft", "start": image_point, "box": box,
                              "index": index, "points": aimed, "grab": True}
                self.update()
                return True
        # Nothing under the press.
        return False

    def _begin_edit(self, position, modifiers=Qt.NoModifier) -> bool:
        """The **Select** tool's press: edit what is under it, else start a marquee.

        The editing half is :meth:`_start_edit` (steps 1-4); this adds the *Select* tool's
        own answer to **empty space** - a marquee, which selects every stored box it covers.
        With the **box** tool on, empty space is where a *box* is drawn, so the press is
        handed back to :meth:`mousePressEvent` (a tool that could edit boxes but never make
        one would be useless).
        """
        if self._start_edit(position, modifiers):
            return True
        image_point = self._to_image(position)
        if image_point is None:
            return False
        if self.point_mode or self.box_mode:
            # The press is where the tool draws.
            return False
        additive = bool(modifiers & (Qt.ShiftModifier | Qt.ControlModifier
                                     | Qt.MetaModifier))
        if self.tool != TOOL_SELECT and not additive:
            # **Empty space belongs to the draw tool** (`X-N20`): a plain press draws a line, a
            # keypoint pair or a box. A marquee is a *selection* gesture, so it needs the
            # *Select* tool or **Shift** - never the ordinary press.
            return False
        self.clear_selection()
        self._drag = {"kind": "marquee", "start": image_point,
                      "rect": (image_point, image_point)}
        self._marquee = self._drag["rect"]
        return True

    def _hit_point(self, position):
        """The index of the editable point under ``position``, or ``None``."""
        where = self._point_pair(position)
        for index, (x, y) in enumerate(self._points):
            screen = self._to_screen((x, y))
            if screen is None:
                continue
            if ((screen[0] - where[0]) ** 2 + (screen[1] - where[1]) ** 2
                    <= BOX_GRAB_PIXELS ** 2):
                return index
        return None

    def _pick(self, position, items):
        """The index of the **innermost** box of ``items`` under ``position``, or ``None``.

        ``items`` is ``((index, box), ...)``. The innermost (smallest) box wins - the rule
        :func:`box_at` documents - because a box drawn around another one is the common
        mistake: a click on the animal *inside* the big box must pick the animal. A page that
        draws **measurements** as boxes (the Bench) turns :attr:`box_edge_grab` on instead, so a
        click inside the box can reach the objects in it.
        """
        where = self._point_pair(position)
        screens = [(index, self._screen_box(box)) for index, box in items]
        hit = box_at(where, [screen for _index, screen in screens], BOX_GRAB_PIXELS)
        if hit is None:
            return None
        for index, screen in screens:
            if screen == hit:
                return index
        return None

    def _hit_stored(self, position):
        """The specimen index of the stored box under ``position``, or ``None``.

        With :attr:`box_edge_grab` on (a page whose boxes are **measurements** with objects
        inside them - the Bench) only the **outline** counts, so a click inside the box reaches
        the keypoints and lines drawn in it; otherwise the whole interior picks the box (the
        corpus, where a box *is* the animal and there is nothing inside to click).
        """
        if not self.box_edge_grab:
            return self._pick(position, self._boxes)
        where = self._point_pair(position)
        screens = [(index, self._screen_box(box)) for index, box in self._boxes]
        hit = box_edge_at(where, [screen for _index, screen in screens], BOX_GRAB_PIXELS)
        if hit is not None:
            for index, screen in screens:
                if screen == hit:
                    return index
            return None
        # A press **inside** a box that is already selected moves the selection: the edge band
        # is how a box is *picked*, the interior is how a picked box is *dragged* (`X-N17`).
        chosen = [(index, screen) for index, screen in screens if index in self._selection]
        inside = box_at(where, [screen for _index, screen in chosen])
        if inside is None:
            return None
        for index, screen in chosen:
            if screen == inside:
                return index
        return None

    def _hit_prediction(self, position):
        """The index of the predicted box under ``position``, or ``None``.

        A prediction is grabbed **anywhere inside it** (it is the thing on screen), and *what*
        the press is aimed at - the box or one of its keypoints - is decided separately
        (:meth:`_prediction_points_at`), so "click the box and get a box" holds even when the
        prediction also carries RTMPose keypoints (`X-N15`).
        """
        index = self._pick(position,
                           [(index, item[0]) for index, item in enumerate(self._predictions)
                            if item[0] is not None])
        if index is not None:
            return index
        # A prediction whose **box** has been accepted is reached by its keypoints, so the
        # remaining half of a model's answer is still selectable (`X-N17`).
        where = self._point_pair(position)
        for index, item in enumerate(self._predictions):
            for point in (item[1] if len(item) > 1 else ()):
                at = self._to_screen(point)
                if at is None:
                    continue
                if ((at[0] - where[0]) ** 2 + (at[1] - where[1]) ** 2
                        <= BOX_GRAB_PIXELS ** 2):
                    return index
        return None

    def _prediction_points_at(self, item, position):
        """The keypoints of a prediction the press **actually landed on**, or ``()``.

        A model's answer carries both a box and its keypoints (RTMDet + RTMPose). What the
        operator aimed at decides what the release makes: a press **on a keypoint** means a
        line, a press on the **box** around it means a box measurement. Returning ``()`` for the
        box is what makes "click the box, get a box" true (`X-N15`).
        """
        points = item[1] if len(item) > 1 else ()
        if not points:
            return ()
        where = self._point_pair(position)
        for point in points:
            screen = self._to_screen(point)
            if screen is None:
                continue
            if ((screen[0] - where[0]) ** 2 + (screen[1] - where[1]) ** 2
                    <= BOX_GRAB_PIXELS ** 2):
                return tuple(points)
        return ()

    def _hit_annotation(self, position, tolerance=BOX_GRAB_PIXELS):
        """``(specimen, segment, endpoint)`` of a stored measurement's *end* under a press.

        Only the drawn **ends** count (a keypoint, a line's stop), and the nearest one wins,
        so a measurement is grabbed by an end and never by a fat middle that would fight the
        box around it. ``endpoint`` is 0 for the first point of the segment, 1 for the second.
        Nothing is returned unless ``annotation_edit`` is on.
        """
        if not self.annotation_edit or not self._annotations:
            return None
        where = self._point_pair(position)
        best = None
        for index, segments, _text, _tags, _texts in self._annotations:
            for segment, ((ax, ay), (bx, by)) in enumerate(segments):
                for endpoint, (px, py) in enumerate(((ax, ay), (bx, by))):
                    screen = self._to_screen((px, py))
                    if screen is None:
                        continue
                    distance = ((screen[0] - where[0]) ** 2
                                + (screen[1] - where[1]) ** 2)
                    if distance <= tolerance ** 2 and (best is None or distance < best[0]):
                        best = (distance, index, segment, endpoint)
        if best is None:
            return None
        return (best[1], best[2], best[3])

    def _grab_annotation(self, position, additive=False) -> bool:
        """A press on a **stored measurement**: let the page load it, then grab the end.

        The canvas cannot load a stored row (it is the page's data), so it reports
        ``annotation_point_selected`` and the page answers by putting that measurement back in
        the editable set. Qt delivers the signal **synchronously**, so the points are on the
        canvas by the time this returns - and the drag starts on the *same* press, which is
        what "edit it right away" means (`X-N16`).

        With **``additive``** (Shift/Cmd) the measurement is only **added to the selection** -
        its row is not loaded, because a press that extends a selection must not also reopen a
        measurement for editing.
        """
        hit = self._hit_annotation(position)
        if hit is None:
            return False
        specimen, segment, endpoint = hit
        if additive:
            self.select_annotation(specimen, segment, additive=True)
            return True
        self.annotation_point_selected.emit(specimen, segment, endpoint)
        self.select_annotation(specimen, segment)
        image_point = self._to_image(position)
        if image_point is None or endpoint >= len(self._points):
            # The page did not load the row (it has no handler, or the row is gone): the
            # measurement is still *selected* - `Backspace` can remove it - but the press
            # places nothing on top of a stored measurement.
            self.update()
            return True
        self._drag = {"kind": "point", "start": image_point, "index": endpoint,
                      "point": self._points[endpoint], "grab": True}
        self.select_point(endpoint)
        self.update()
        return True

    def _edit_drag(self, position) -> bool:
        """Follow whatever the press started: a move, a resize, a point or a marquee.

        Returns True while one is in progress, so the page's own gestures stay out of it.
        """
        if self._drag is None:
            return False
        point = self._to_image(position, clamp=True)
        if point is None:
            return True
        kind = self._drag.get("kind")
        if kind == "marquee":
            self._drag["rect"] = (self._drag["start"], point)
            self._marquee = self._drag["rect"]
            self.update()
            return True
        if kind == "point":
            if point is not None:
                self._points[self._drag["index"]] = (float(point[0]), float(point[1]))
                if self._drag.get("grab"):
                    # A **grabbed** point is being *moved*, not placed: `points_changed`
                    # means "a click placed a landmark", and a page that hears it for a move
                    # re-renders its own copy of the pair - which is the point snapping back
                    # to where it was (`SR-DATASET-027`). `points_moved` says only "look
                    # again", and the canvas keeps the position. A grab is a drag in **any**
                    # tool, so this is the drag's own flag, not `box_edit`.
                    self.points_moved.emit()
                else:
                    self.points_changed.emit(self.points)
                self.update()
            return True
        bounds = self._image_bounds()
        if kind == "draft":
            original = self._drag.get("box")
            if original is None:
                return True                     # a points-only prediction: nothing to move
            self._box = moved_box(original,
                                  float(point[0]) - float(self._drag["start"][0]),
                                  float(point[1]) - float(self._drag["start"][1]), bounds)
            self.update()
            return True
        if kind == "resize":
            index = self._drag["indices"][0]
            self._replace_box(index, resized_box(self._drag["boxes"][index],
                                                 self._drag["handle"], point,
                                                 minimum=float(self.MIN_BOX_PIXELS),
                                                 bounds=bounds))
            self.update()
            return True
        if kind == "move":
            # The **whole selection** moves together - which is what selecting several boxes
            # is for (`SR-DATASET-027`).
            dx = float(point[0]) - float(self._drag["start"][0])
            dy = float(point[1]) - float(self._drag["start"][1])
            for index, original in self._drag["boxes"].items():
                self._replace_box(index, moved_box(original, dx, dy, bounds))
            self.update()
            return True
        return False

    def _finish_edit(self) -> bool:
        """End a drag: report every box that moved, select a marquee's boxes, or nothing."""
        if self._drag is None:
            return False
        drag, self._drag = self._drag, None
        kind = drag.get("kind")
        if kind == "marquee":
            self._marquee = None
            self._select_marquee(drag["rect"])
            self.update()
            return True
        if kind == "point":
            self.points_edited.emit()      # the page commits a corrected label (if there is one)
            self.update()
            return True
        if kind == "draft":
            index = drag.get("index")
            original = drag.get("box")
            final = (tuple(self._box) if self._box is not None
                     else tuple(original) if original is not None else None)
            moved = original is not None and final != tuple(original)
            if index is not None:
                points = tuple(drag.get("points") or ())   # what the press aimed at
                # A promoted prediction is reported **once**, through `prediction_accepted`,
                # with the final geometry - never as a drawn box as well, or a page that acts on
                # both would make two of it (`X-N19`).
                self.prediction_accepted.emit(index, final, points, moved)
            return True
        if kind in ("move", "resize"):
            # The **moved geometry is captured before anything is emitted**: a page stores each
            # reported box and then re-reads the frame's boxes from the database, so emitting
            # as we go would make the second box snap back to where the *first* store left it
            # (measured: dragging two selected boxes moved one and reverted the other).
            moved = [(index, tuple(self.box_of(index) or original), tuple(original))
                     for index, original in drag["boxes"].items()]
            for index, updated, original in moved:
                if updated != original:
                    self.box_edited.emit(index, updated)
            return True
        return False

    def _select_marquee(self, rect):
        """Select every **stored box**, **measurement** and **prediction** the marquee covers.

        A marquee is a selection, never a label: it selects the objects whose own rectangle lies
        inside the dragged one, so the operator can take a whole row of animals - or their
        measurements - in one go. Nothing when it was a click, not a drag.
        """
        (x1, y1), (x2, y2) = rect
        left, right = sorted((float(x1), float(x2)))
        top, bottom = sorted((float(y1), float(y2)))
        if (right - left) < BOX_GRAB_PIXELS and (bottom - top) < BOX_GRAB_PIXELS:
            return                                          # a click, not a marquee

        def inside(box):
            return (left <= box[0] and top <= box[1]
                    and box[2] <= right and box[3] <= bottom)

        picked = [index for index, box in self._boxes if inside(box)]
        predictions = [index for index, item in enumerate(self._predictions)
                       if item[0] is not None and inside(item[0])]
        annotations = []
        for index, segments, _text, _tags, _texts in self._annotations:
            for segment, (a, b) in enumerate(segments):
                if (inside((min(a[0], b[0]), min(a[1], b[1]),
                            max(a[0], b[0]), max(a[1], b[1])))):
                    annotations.append((index, segment))
        if not (picked or predictions or annotations):
            return
        self._selection = picked
        self._selected = picked[-1] if picked else None
        self._selected_predictions = predictions
        self._selected_prediction = predictions[-1] if predictions else None
        self._selected_annotations = annotations
        self._selected_annotation = annotations[-1] if annotations else None
        if picked:
            self.box_selected.emit(self._selected)
        if predictions:
            self.prediction_selected.emit(self._selected_prediction)
        self.update()

    def overlaps_stored(self, box, threshold=0.5):
        """The stored specimen whose box overlaps ``box`` most, or ``None``.

        Two boxes on one animal is a labelling error - and with several
        individuals per frame it is the easy one to make, because nothing on
        screen forces you to count. The page uses this to **say so**; the label is
        still stored, because a label is never thrown away silently
        (``SR-DATASET-N03``) and a legitimate overlap (an animal in front of
        another) must remain possible.

        The **prediction layer** uses the same predicate to stay honest
        (`SR-PREDICT-018`): a model box whose animal already has a stored label is not
        information, so it is not painted - "one box per animal".
        """
        best, best_iou = None, 0.0
        for index, stored in self._boxes:
            value = box_iou(box, stored)
            if value > best_iou:
                best, best_iou = index, value
        return best if best is not None and best_iou >= threshold else None

    # -- accessors ---------------------------------------------------------
    @property
    def points(self) -> list:
        """Clicked points, in image pixels."""
        return list(self._points)

    @property
    def image_size(self):
        """(width, height) of the displayed image, or (0, 0)."""
        return (self._pixmap.width(), self._pixmap.height()) if self._pixmap else (0, 0)

    # -- zoom & pan --------------------------------------------------------
    @property
    def zoom(self) -> float:
        """Magnification relative to 'fit' (1.0 shows the whole image)."""
        return self._zoom

    def reset_view(self):
        """Zoom back to fit and centre the image."""
        self._zoom = self.MIN_ZOOM
        self._pan = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self):
        """Zoom in about the widget centre."""
        self._zoom_by(self.ZOOM_STEP)

    def zoom_out(self):
        """Zoom out about the widget centre."""
        self._zoom_by(1.0 / self.ZOOM_STEP)

    def set_zoom(self, zoom, anchor=None):
        """Set magnification relative to fit, zooming about ``anchor``."""
        self._zoom_to(zoom, anchor if anchor is not None else self._center())

    def _zoom_by(self, factor, anchor=None):
        """Multiply the zoom by ``factor``, keeping ``anchor`` fixed (centre)."""
        self._zoom_to(self._zoom * factor,
                      anchor if anchor is not None else self._center())

    def _zoom_to(self, zoom, anchor):
        """Change the zoom keeping the image point under ``anchor`` in place."""
        if self._pixmap is None:
            return
        focus = self._to_image(anchor)
        self._zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(zoom)))
        self._clamp_pan()
        if focus is not None:
            self._anchor_image_point(focus, anchor)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def _anchor_image_point(self, image_point, anchor):
        """Pan so ``image_point`` lands under the widget ``anchor`` again."""
        rect = self._display_rect()
        if rect.isEmpty():
            return
        scale = rect.width() / self._pixmap.width()
        base_x = (self.width() - rect.width()) // 2
        base_y = (self.height() - rect.height()) // 2
        self._pan.setX(anchor.x() - image_point[0] * scale - base_x)
        self._pan.setY(anchor.y() - image_point[1] * scale - base_y)
        self._clamp_pan()

    def _center(self):
        return QPointF(self.width() / 2.0, self.height() / 2.0)

    def _pan_by(self, dx, dy):
        self._pan += QPointF(dx, dy)
        self._clamp_pan()
        self.update()

    # -- geometry ----------------------------------------------------------
    def _display_size(self):
        fit = min(self.width() / self._pixmap.width(),
                  self.height() / self._pixmap.height())
        scale = fit * self._zoom
        return int(self._pixmap.width() * scale), int(self._pixmap.height() * scale)

    def _display_rect(self) -> QRect:
        if self._pixmap is None or self.width() <= 0 or self.height() <= 0:
            return QRect(0, 0, 0, 0)
        width, height = self._display_size()
        x = (self.width() - width) // 2 + int(self._pan.x())
        y = (self.height() - height) // 2 + int(self._pan.y())
        return QRect(x, y, width, height)

    def _clamp_pan(self):
        """Keep the image covering the widget when zoomed, centred when it fits."""
        if self._pixmap is None or self.width() <= 0 or self.height() <= 0:
            return
        width, height = self._display_size()
        if width <= self.width():
            self._pan.setX(0.0)
        else:
            base_x = (self.width() - width) // 2
            self._pan.setX(min(-base_x, max(self.width() - width - base_x,
                                            self._pan.x())))
        if height <= self.height():
            self._pan.setY(0.0)
        else:
            base_y = (self.height() - height) // 2
            self._pan.setY(min(-base_y, max(self.height() - height - base_y,
                                            self._pan.y())))

    def _to_image(self, position, clamp=False):
        """Map a widget position to image pixels.

        ``clamp`` keeps the result inside the image. A **box drag** that leaves the
        frame - going around an animal at the edge, out across the letterbox - must
        still describe the animal, so its corner is pulled back onto the image
        border instead of being stored as a coordinate that cannot exist. A stored
        ``y1 = -10`` is not a detail: a detector trained on it learns a rectangle
        that starts outside the picture, and the corpus looks corrupted for a
        reason nobody can see in the app. Landmark clicks are **not** clamped -
        a click outside the image is not a landmark.
        """
        rect = self._display_rect()
        if rect.isEmpty():
            return None
        scale = rect.width() / self._pixmap.width()
        x = (position.x() - rect.x()) / scale
        y = (position.y() - rect.y()) / scale
        if clamp:
            # A rectangle may reach the far border of a W x H raster, so the span
            # is [0, W] and not [0, W - 1].
            x = min(max(x, 0.0), float(self._pixmap.width()))
            y = min(max(y, 0.0), float(self._pixmap.height()))
        return (x, y)

    # -- interaction -------------------------------------------------------
    def event(self, event):
        """Route macOS trackpad gestures (pinch/pan) to the zoom & pan handlers.

        A **style change** (the desktop switched appearance, or ``DAPHNIA_THEME``
        pinned another scheme) also lands here: the panel colour comes from the
        theme rather than from a stylesheet rule, so this widget has to repaint
        itself or it would keep the old scheme until something else moved it.
        """
        if event.type() == QEvent.NativeGesture and self._native_gesture(event):
            return True
        if event.type() == QEvent.StyleChange:
            self.update()
        return super().event(event)

    def _native_gesture(self, event):
        """Handle a trackpad gesture; returns True when it was consumed."""
        if self._pixmap is None:
            return False
        kind = event.gestureType()
        if kind == Qt.ZoomNativeGesture:            # two-finger pinch
            anchor = event.position() if hasattr(event, "position") else self._center()
            factor = min(max(1.0 + event.value(), 0.5), 2.0)
            self._zoom_by(factor, anchor)
            return True
        if kind == Qt.PanNativeGesture and hasattr(event, "delta"):
            self._pan += event.delta()
            self._clamp_pan()
            self.update()
            return True
        return False

    def _wheel_delta(self, event) -> tuple:
        """``(dx, dy)`` in widget pixels: the pixel delta when there is one.

        A trackpad sends many small **pixel** deltas, a wheel mouse sends whole
        **angle** notches; both are normalised here so one code path pans for either.
        """
        pixel = event.pixelDelta()
        if pixel.x() or pixel.y():
            return float(pixel.x()), float(pixel.y())
        angle = event.angleDelta()
        scale = self.WHEEL_PAN_STEP / self.SCROLL_REFERENCE
        return float(angle.x()) * scale, float(angle.y()) * scale

    def wheelEvent(self, event):
        """A two-finger scroll **pans** (both axes) and *never* zooms.

        Zooming has its own gestures - the trackpad **pinch**, the *Zoom -/Fit/Zoom +*
        buttons and ``+``/``-``/``F`` - so scrolling a frame can never change the
        magnification by accident. That matters because the click coordinates are what
        the model learns: an unintended zoom is a landmark placed on the wrong pixel.
        The view follows the fingers: sideways moves sideways, up/down moves up/down.
        """
        if self._pixmap is None:
            return
        dx, dy = self._wheel_delta(event)
        if dx == 0 and dy == 0:
            return
        self._pan_by(dx, dy)
        event.accept()

    def _starts_pan(self, event) -> bool:
        """True when this press should pan instead of placing a landmark.

        Right/middle drag always pans. A **left** drag pans when the hand tool is on
        (``pan_mode``) or Command/Control is held, so an operator can move the view
        without accidentally dropping a point - the case that made precise zooming
        painful on a trackpad.
        """
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            return True
        if event.button() != Qt.LeftButton:
            return False
        if self.pan_mode:
            return True
        held = event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)
        return bool(held)

    def anchor_box_corner(self, point):
        """Start (or continue) a box from one corner - the click *or* the drag path.

        The anchored corner plus the cursor is the whole state, so both ways of
        drawing a box use the same code: **drag** from the first corner, or **click**
        one corner and then the opposite one. Releasing only stores a box the gesture
        actually made (see :meth:`mouseReleaseEvent`), so a click is never mistaken
        for a sliver of a box.
        """
        x, y = float(point[0]), float(point[1])
        if self._box_origin is None:
            self._set_box_origin((x, y))
            self._box = (x, y, x, y)
        else:
            self._box = (self._box_origin[0], self._box_origin[1], x, y)
        self.update()

    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        if self._starts_pan(event):
            if event.button() != Qt.LeftButton:
                # A right/middle click also cancels a half-drawn box: the gesture
                # that started it is abandoned, and the press still pans.
                self._set_box_origin(None)
            self._panning = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        edit = self.box_edit or self.editing
        if edit and self._begin_edit(event.position(), event.modifiers()):
            # Editing is available in *every* draw tool (`X-N20`): a press on an object
            # selects/moves/resizes it, and the gesture below (draw) must not also run.
            return
        if not edit and self._start_edit(event.position(), event.modifiers()):
            # A **create** tool still lets the operator grab what is under the press - a
            # nearby point, box or prediction - and edit it right away, without switching to
            # *Select* first (`X-N15`). Empty space (nothing under the press) falls through to
            # the tool's own gesture below, so measuring is unchanged.
            return
        # A box corner is pulled back inside the frame; a landmark just is a click.
        point = self._to_image(event.position(), clamp=self.box_mode)
        if point is None:
            return
        if self.box_mode:
            self.anchor_box_corner(point)
            return
        if len(self._points) >= self.max_points:
            self._points = []
        self._points.append((float(point[0]), float(point[1])))
        self.points_changed.emit(self.points)
        self.point_placed.emit(self.points)
        self.update()

    def mouseDoubleClickEvent(self, event):
        """A double click reports the gesture and draws nothing further.

        The **first** press of the double click has already placed its point (or opened
        a box preview), and Qt sends this event *instead of* a second press - so the
        handler must not place a second point, and it aborts a box preview the first
        press may have opened. What the gesture *means* is the page's decision: in the
        Dataset queue it walks to the next frame.

        **In *Edit* mode it means two selections, not "next"**: the queue-walking binding fired
        here while a box was being corrected, which advanced the frame and threw the selection
        away ("it randomly goes to next frame", "I have to click it twice").
        """
        if self._pixmap is None or event.button() != Qt.LeftButton:
            return
        if self._starts_pan(event) or self._panning is not None:
            return
        if self.box_edit or self.editing:
            # Editing is on (`X-N20`): the double click is two selections, never "next".
            event.accept()
            return
        self._set_box_origin(None)       # the gesture is "next", not "draw a sliver"
        if self._box is not None and not self.box_mode:
            self._box = None
        self.double_clicked.emit()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._edit_drag(event.position()):
            if self._hover is not None:
                self._set_hover(None)           # a drag owns the cursor, not the hover
            return
        if self._box_origin is not None:
            point = self._to_image(event.position(), clamp=True)
            if point is not None:
                self._box = (self._box_origin[0], self._box_origin[1],
                             float(point[0]), float(point[1]))
                self.update()
            return
        if self._panning is None:
            # Nothing is being dragged: show what the press would grab (`X-N15`).
            self._set_hover(self._hit_editable(event.position()))
            return
        self._pan += event.position() - self._panning
        self._panning = event.position()
        self._clamp_pan()
        self.update()

    def leaveEvent(self, event):
        """The cursor left the canvas: forget what was under it."""
        self._set_hover(None)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._finish_edit():
            return
        if self._panning is not None:
            self._panning = None
            self.unsetCursor()
        if self._box_origin is not None:
            box = _normalised_box(self._box)
            if box is not None and (box[2] - box[0]) >= self.MIN_BOX_PIXELS \
                    and (box[3] - box[1]) >= self.MIN_BOX_PIXELS:
                self._set_box_origin(None)
                self._box = box
                self.box_changed.emit(self.boxes_drawn)
            # Else: a **click**, not a drag, so the corner stays anchored and the
            # opposite one is set by the next click (or a drag from here). Nothing is
            # reported: a corner is not a box, and a mis-click must never be stored.
            self.update()

    def drop_prediction(self, index, part="all"):
        """Remove a prediction - or just its **box** or just its **keypoints** - from the layer.

        A model's answer is two things at once: the box RTMDet found and the keypoints RTMPose
        placed in it. They are **independent objects**: accepting the box must not take the
        keypoints away, and accepting the keypoints must not take the box away (`X-N17`). The
        item is only removed once *nothing* of it is left. Safe to call for a part that is
        already gone, for a part the prediction never had, and for an index that is out of
        range - and the canvas drops a **moved** draft itself, so a page that also drops a
        clicked one cannot remove the wrong suggestion.

        Each part is cleared **on its own** and the removal is decided afterwards. Asking for a
        part that is already gone must not abandon the whole call: a **box-only** prediction
        (the detector answered, the pose model did not) is exactly what `part="all"` is for, and
        returning early on its empty keypoints left it in the layer for good - found by the
        Bench's *From the model* list, whose one click drops the whole answer
        (`SR-BENCH-017`).
        """
        if index is None or int(index) < 0 or int(index) >= len(self._predictions):
            return
        index = int(index)
        box, points, label = self._predictions[index]
        if part in ("box", "all"):
            box = None
        if part in ("points", "all"):
            points = ()
        if box is None and not points:
            self._drop_prediction(index)          # nothing left: the item itself goes
            return
        self._predictions[index] = (box, points, label)
        self.update()

    def _drop_prediction(self, index):
        """Take one box out of the prediction layer, and say which: the layer is the canvas'."""
        self._predictions = [item for position, item in enumerate(self._predictions)
                             if position != index]
        # The selection indexes the same list, so it follows the removal and the shift.
        self._selected_predictions = [item - 1 if item > index else item
                                      for item in self._selected_predictions if item != index]
        if self._selected_prediction == index:
            self._selected_prediction = (self._selected_predictions[-1]
                                         if self._selected_predictions else None)
        elif self._selected_prediction is not None and self._selected_prediction > index:
            self._selected_prediction -= 1
        self.prediction_removed.emit(index)
        self.update()

    def delete_selection(self) -> bool:
        """Remove what is selected - the same rule as `Backspace` (`SR-DATASET-027`).

        Public, because the **pages bind it as a shortcut too**: a key that only works while the
        canvas happens to hold focus is a key the operator presses again and again ("press
        backspace multiple times till it disappears"). Returns whether anything was selected.

        * selected **stored boxes** - one `box_delete_requested` per box, and the *page* deletes
          the label (the canvas never touches the database);
        * selected **measurements** - one `annotation_delete_requested` per measurement, and the
          page drops those labels / view items;
        * selected **model boxes** - `prediction_removed` each: nothing was stored, so the layer
          drops them;
        * a selected **point** - the canvas removes it itself and reports `points_changed`.
        """
        if self._selected_annotations:
            for specimen, segment in list(self._selected_annotations):
                self.annotation_delete_requested.emit(int(specimen), int(segment))
            self._selected_annotations = []
            self._selected_annotation = None
            self.update()
            return True
        if self._selection:
            for index in list(self._selection):
                self.box_delete_requested.emit(index)
            self.clear_selection()
            return True
        if self._selected_predictions:
            # Highest index first: `_drop_prediction` renumbers the layer as it removes, so
            # dropping from the top keeps every later index valid.
            for index in sorted(self._selected_predictions, reverse=True):
                self._drop_prediction(index)
            self._selected_predictions = []
            self._selected_prediction = None
            self.update()
            return True
        if self._selected_point is not None:
            index = self._selected_point
            self._points = [point for position, point in enumerate(self._points)
                            if position != index]
            self.select_point(None)
            self.points_changed.emit(self.points)
            self.update()
            return True
        return False

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.zoom_out()
        elif key in (Qt.Key_0, Qt.Key_F):
            self.reset_view()
        elif key == Qt.Key_Left:
            self._pan_by(-self.PAN_STEP, 0)
        elif key == Qt.Key_Right:
            self._pan_by(self.PAN_STEP, 0)
        elif key == Qt.Key_Up:
            self._pan_by(0, -self.PAN_STEP)
        elif key == Qt.Key_Down:
            self._pan_by(0, self.PAN_STEP)
        elif key == Qt.Key_Escape:
            # `Esc` gives up the *current* thing first: a half-drawn box, then the
            # selection - the same order the eye is working in.
            if self._box_origin is not None or self._box is None:
                self.clear_box()                   # abandon a half-drawn box
            self.clear_selection()
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            # Removing something is a data change: the page hears the request and decides
            # (`SR-DATASET-027`). One implementation, shared with the pages' shortcut.
            if not self.delete_selection():
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clamp_pan()

    # -- painting ----------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("canvas"))
        if self._pixmap is None:
            # Nothing to show: the page's own wording ("No image") - or, when a source
            # was refused, the reason, wrapped so a long path is not cut off. A blank
            # canvas must never be the only answer to "why is there no picture?".
            painter.setPen(theme.qcolor("canvas_text"))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                             self._refusal or self.placeholder)
            return
        rect = self._display_rect()
        painter.drawPixmap(rect, self._pixmap)
        self._paint_lattice(painter, rect)
        self._paint_field_stop(painter, rect)
        self._paint_reference_points(painter, rect)
        self._paint_predictions(painter, rect)
        self._paint_annotations(painter, rect)
        self._paint_boxes(painter, rect)
        self._paint_points(painter, rect)
        self._paint_selected_annotations(painter, rect)
        self._paint_hover(painter, rect)
        self._paint_marquee(painter)

    def _paint_selected_annotations(self, painter, rect):
        """Ring the ends of every **selected measurement**, so a selection is visible (`X-N15`).

        A measurement is an object like a box: when it is selected (clicked, Shift-clicked or
        caught by the marquee) it must *look* selected, and `Backspace` removes it.
        """
        if not self._selected_annotations or self._pixmap is None:
            return
        pen = QPen(QColor(theme.qcolor("accent")))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        wanted = set(self._selected_annotations)
        for index, segments, _text, _tags, _texts in self._annotations:
            for segment, (a, b) in enumerate(segments):
                if (index, segment) not in wanted:
                    continue
                for point in (self._to_screen(a), self._to_screen(b)):
                    if point is not None:
                        painter.drawEllipse(QPointF(*point), 10, 10)

    def _paint_hover(self, painter, rect):
        """Draw what is **under the cursor**, so *"grab this"* is visible before the press.

        A ring on a keypoint, an outline on a box or a model's box - in the accent colour, so
        it reads as *the same* affordance the selection wears and never as a new label
        (`X-N15`). Drawn after the points so the ring is not hidden, and only while nothing is
        being dragged (the drag owns the cursor then).
        """
        if self._hover is None or self._pixmap is None or self._drag is not None:
            return
        kind, index = self._hover
        pen = QPen(QColor(theme.qcolor("accent")))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        if kind == "point":
            if index >= len(self._points):
                return
            screen = self._to_screen(self._points[index])
            if screen is not None:
                painter.drawEllipse(QPointF(*screen), 9, 9)
            return
        if kind == "annotation":
            # Ring every end of the specimen's stored measurement(s): the run of keypoints
            # the press would open for editing (`X-N16`).
            for specimen, segments, _text, _tags, _texts in self._annotations:
                if specimen != index:
                    continue
                for (ax, ay), (bx, by) in segments:
                    for point in self._to_screen((ax, ay)), self._to_screen((bx, by)):
                        if point is not None:
                            painter.drawEllipse(QPointF(*point), 9, 9)
            return
        if kind == "box":
            box = self.box_of(index)
        elif kind == "prediction":
            box = self._predictions[index][0] if index < len(self._predictions) else None
        else:
            return
        screen = self._screen_box(box) if box is not None else None
        if screen is not None:
            painter.drawRect(screen)

    def _paint_selection(self, painter, screen, primary=False):
        """Draw a **selected** box: every one outlined, the primary one with its handles.

        The dashed per-specimen outline says *where an animal is*; the solid accent outline says
        *this one is selected* - move it by its middle, resize it by an edge or a corner,
        remove it with ``Delete``. Handles are drawn on the **primary** box only (the last one
        clicked): eight sets of handles on eight boxes is a picture nobody can aim at. Drawn
        while the page's *Select* tool is on (the affordances appear exactly when they can be
        grabbed) **and** whenever something is selected - a box grabbed in a create tool is just
        as editable, and must look selected (`X-N15`).
        """
        if not self.box_edit and not self.editing and not self._selection:
            return
        color = QColor(theme.qcolor("accent"))
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(screen)
        if not primary:
            return
        painter.setBrush(color)
        size = 4.0
        for point in (screen.topLeft(), QPointF(screen.right(), screen.top()),
                      screen.bottomLeft(),
                      QPointF(screen.right(), screen.bottom()),
                      QPointF(screen.center().x(), screen.top()),
                      QPointF(screen.center().x(), screen.bottom()),
                      QPointF(screen.left(), screen.center().y()),
                      QPointF(screen.right(), screen.center().y())):
            painter.drawRect(QRectF(point.x() - size, point.y() - size, size * 2, size * 2))

    def _paint_marquee(self, painter):
        """Draw the rubber band while it is dragged - the selection being made, live."""
        if self._marquee is None:
            return
        first = self._to_screen(self._marquee[0])
        second = self._to_screen(self._marquee[1])
        if first is None or second is None:
            return
        rect = QRectF(QPointF(*first), QPointF(*second)).normalized()
        color = QColor(theme.qcolor("accent"))
        pen = QPen(color)
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(40)
        painter.setBrush(fill)
        painter.drawRect(rect)

    def _paint_predictions(self, painter, rect):
        """Draw the model's answer *under* everything the operator owns (`SR-PREDICT-018`).

        One colour, dotted, no specimen number - because a prediction has no identity and is
        nobody's stored label: it is a suggestion drawn behind the labels, the boxes and the
        points the human placed, so it can be seen, judged and ignored. The keypoints are
        drawn as the segments they are (the top-down pipeline's head-to-tail line) plus a dot
        each, and the label - when the surface filled one - sits on its own plate like a
        measurement's does.

        Two predictions are **not** drawn, because drawing them is the "ghost" and the "two
        boxes on one animal" an operator reports as a bug:

        * the one **being edited** (it is the box in their hand; a dotted copy left behind at the
          old position reads as "the new position was not updated");
        * the one **an existing label already covers** (IoU >= 0.5): once an animal has a real
          box, the same animal drawn twice carries no information.
        """
        if self._pixmap is None or not self._predictions:
            return
        scale = rect.width() / self._pixmap.width()
        color = QColor(theme.qcolor("prediction"))
        for index, (box, points, text) in enumerate(self._predictions):
            dragging = (self._drag is not None and self._drag.get("kind") == "draft"
                        and self._drag.get("index") == index
                        and self._drag.get("box") is not None)
            if dragging:
                # The one in the operator's hand: a dotted copy left behind at the old position
                # reads as "the new position was not updated".
                continue
            selected = index in self._selected_predictions
            pen = QPen(QColor(theme.qcolor("accent")) if selected else color)
            pen.setWidth(3 if selected else 2)
            pen.setStyle(Qt.SolidLine if selected else Qt.DotLine)
            if box is not None and (self.overlaps_stored(box) is None or selected):
                # Already labelled (IoU >= 0.5): the same animal drawn twice carries no
                # information - *unless it is selected*, which the operator must be able to see.
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                x1, y1, x2, y2 = box
                painter.drawRect(QRect(int(rect.x() + x1 * scale), int(rect.y() + y1 * scale),
                                       int((x2 - x1) * scale), int((y2 - y1) * scale)))
            if not points:
                continue
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QColor(theme.qcolor("accent")) if selected else color)
            anchor = None
            for position, (x, y) in enumerate(points):
                point = QPointF(rect.x() + x * scale, rect.y() + y * scale)
                if anchor is None:
                    anchor = point
                elif position:
                    painter.drawLine(anchor, point)
                painter.drawEllipse(point, 3, 3)
            if text and anchor is not None:
                self._paint_annotation_label(painter, anchor, text, color)

    def _paint_boxes(self, painter, rect):
        """Draw the stored boxes per specimen, then the one being drawn now.

        Stored boxes use each specimen's colour and a dashed outline, so a box reads
        as *where the animal is* and never hides the landmarks on top of it; the box
        being dragged is solid and in the marker colour.
        """
        if self._pixmap is None:
            return
        width, height = self._pixmap.width(), self._pixmap.height()
        scale = rect.width() / width

        def to_screen(box):
            x1, y1, x2, y2 = box
            return QRect(int(rect.x() + x1 * scale), int(rect.y() + y1 * scale),
                         int((x2 - x1) * scale), int((y2 - y1) * scale))

        for index, box in self._boxes:
            pen = QPen(QColor(specimen_color(index)))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            screen = to_screen(box)
            painter.drawRect(screen)
            if self.show_annotation_index:
                # Explicitly float arithmetic: QRect.topLeft() is a *QPoint*, and adding
                # a QPointF to it raises inside the paint event, which leaves the painter
                # active and takes the process down.
                corner = screen.topLeft()
                painter.drawText(QPointF(corner.x() + 3.0, corner.y() - 5.0), str(index))
            if index in self._selection:
                self._paint_selection(painter, screen, primary=(index == self._primary()))
        if self._box is not None:
            box = _normalised_box(self._box)
            pen = QPen(QColor(MARKER_COLOR))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(to_screen(box))
        if self.box_pending:
            # The anchored corner, so a started box is visible before it is finished.
            origin_x, origin_y = self._box_origin
            anchor = QPointF(rect.x() + origin_x * scale, rect.y() + origin_y * scale)
            painter.setPen(QPen(QColor(MARKER_COLOR)))
            painter.setBrush(QColor(MARKER_COLOR))
            painter.drawEllipse(anchor, 4, 4)

    def _paint_annotations(self, painter, rect):
        """Draw the already-annotated *intervals*, one colour per specimen.

        Colour *is* the specimen's identity here: the operator sees at a glance which
        animals are done and which are still missing - and each measured interval is
        drawn as the line it is (width across the body, length head to spine base),
        not as whatever consecutive points happened to be clicked. The specimen number
        is drawn at its first point, so a colour can be read back as an index - and
        where an annotation carries a **label** (the Predict surface's measured
        length), the label is drawn there instead, on a dark plate.

        A **per-segment label** (`set_annotations`' fifth element, the Bench's own
        labelling) is drawn above the *middle of its own segment*, so one animal's
        several numbers are each readable where they were measured rather than collected
        into one line at the first point.
        """
        if not self._annotations or self._pixmap is None:
            return
        scale = rect.width() / self._pixmap.width()
        for index, segments, text, _tags, texts in self._annotations:
            if not segments:
                continue
            color = QColor(specimen_color(index))
            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(color)
            anchor = None
            for position, ((ax, ay), (bx, by)) in enumerate(segments):
                a = QPointF(rect.x() + ax * scale, rect.y() + ay * scale)
                b = QPointF(rect.x() + bx * scale, rect.y() + by * scale)
                if anchor is None:
                    anchor = a
                if a != b:
                    painter.drawLine(a, b)
                painter.drawEllipse(a, 3, 3)
                painter.drawEllipse(b, 3, 3)
                label = texts[position] if position < len(texts) else ""
                if self.show_annotation_text and label:
                    self._paint_annotation_label(
                        painter, QPointF((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0),
                        label, color)
                    painter.setPen(pen)          # the label's own pen and brush end here
                    painter.setBrush(color)
            if anchor is None:
                continue
            if self.show_annotation_text and text:
                self._paint_annotation_label(painter, anchor, text, color)
            elif self.show_annotation_index:
                painter.setPen(QColor("#ffffff"))
                painter.drawText(anchor + QPointF(-3, -7), str(index))
                painter.setPen(pen)

    def _paint_annotation_label(self, painter, anchor, text, color):
        """Draw one annotation's label above its anchor, on a dark plate.

        The plate is not decoration: this label is a measured length written over a
        photograph that is pale in a lit well and near-black at the frame edge, and
        coloured text alone disappears on one of them. A translucent plate keeps the
        number legible on both and hides only a corner of the animal.
        """
        metrics = painter.fontMetrics()
        width = float(metrics.horizontalAdvance(text))
        height = float(metrics.height())
        box = QRectF(anchor.x() - 2.0, anchor.y() - height - 8.0,
                     width + 10.0, height + 4.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(box, 3.0, 3.0)
        painter.setPen(QColor(color))
        painter.drawText(QPointF(box.x() + 5.0, box.y() + height), text)

    def _paint_lattice(self, painter, rect):
        """Draw the measured grid across the image, at its measured spacing."""
        if self._lattice is None or self._pixmap is None:
            return
        period, angle_deg = self._lattice
        if period <= 0:
            return

        angle = np.radians(angle_deg)
        normal = (float(np.cos(angle)), float(np.sin(angle)))
        direction = (-normal[1], normal[0])          # lines run along this
        width, height = self._pixmap.width(), self._pixmap.height()
        centre_x, centre_y = width / 2.0, height / 2.0
        reach = float(max(width, height))
        scale = rect.width() / width

        pen = QPen(QColor(LATTICE_COLOR))
        pen.setWidth(1)
        painter.setPen(pen)
        count = int(reach / period) + 2
        for index in range(-count, count + 1):
            offset = index * period
            base_x = centre_x + normal[0] * offset
            base_y = centre_y + normal[1] * offset
            painter.drawLine(
                QPointF(rect.x() + (base_x - direction[0] * reach) * scale,
                        rect.y() + (base_y - direction[1] * reach) * scale),
                QPointF(rect.x() + (base_x + direction[0] * reach) * scale,
                        rect.y() + (base_y + direction[1] * reach) * scale))

    def _paint_field_stop(self, painter, rect):
        """Ring the paper's circle, dashed so it cannot be mistaken for a measurement.

        Drawn *over* the lattice, because it is the boundary of what the lattice may be
        read from: a solid ring would read as one more measured line on the paper.
        """
        if self._field_stop is None or self._pixmap is None:
            return
        centre_x, centre_y, radius = self._field_stop
        if radius <= 0:
            return
        scale = rect.width() / self._pixmap.width()
        pen = QPen(QColor(FIELD_STOP_COLOR))
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(
            QPointF(rect.x() + centre_x * scale, rect.y() + centre_y * scale),
            radius * scale, radius * scale)

    def _paint_reference_points(self, painter, rect):
        """Ring the **detected reference points** of a grid measurement.

        Drawn *under* the operator's own marks, like a prediction: these are evidence of what
        the detector saw, not something they placed. A ring (not a filled dot) so a fine print's
        points stay countable on screen, and a menu of hundreds costs no more than a few.
        """
        if not self._reference_points or self._pixmap is None:
            return
        scale = rect.width() / self._pixmap.width()
        pen = QPen(QColor(REFERENCE_COLOR))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for x, y in self._reference_points:
            painter.drawEllipse(QPointF(rect.x() + x * scale, rect.y() + y * scale), 3, 3)

    def _paint_points(self, painter, rect):
        if not self._points:
            return
        scale = rect.width() / self._pixmap.width()
        pen = QPen(QColor(MARKER_COLOR))
        pen.setWidth(2)
        screen = [QPointF(rect.x() + x * scale, rect.y() + y * scale)
                  for x, y in self._points]
        painter.setPen(pen)
        if self.skeleton:
            # A keypoint page draws the schema's **bones** (head->tail, left->right), not
            # a polyline through the click order - which would join the tail to a side
            # that no anatomy connects (`annotation.KeypointSystem.edges`).
            for first, second in self.skeleton:
                if first < len(screen) and second < len(screen):
                    painter.drawLine(screen[first], screen[second])
        elif self.connect_points and len(screen) >= 2:
            painter.drawLine(screen[0], screen[1])
        for index, point in enumerate(screen):
            painter.setBrush(QColor(MARKER_COLOR))
            painter.drawEllipse(point, 4, 4)
            if index == self._selected_point:
                # The point under edit wears a ring, so "drag *this* one" is visible
                # (`SR-DATASET-027`).
                ring = QPen(QColor(theme.qcolor("accent")))
                ring.setWidth(2)
                painter.setPen(ring)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(point, 8, 8)
                painter.setPen(pen)
            if not self.show_point_labels:
                continue
            painter.setPen(QColor("#ffffff"))
            name = self.point_names[index] if index < len(self.point_names) else str(index + 1)
            painter.drawText(point + QPointF(8, -8), name)
            painter.setPen(pen)
