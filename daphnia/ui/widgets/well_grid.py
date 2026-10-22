"""The plate as a picture: one lit well per well of the plate.

The plate page answers one question from across the bench - *which wells have an image
yet?* - so the plate is **drawn**, not listed: a grid of wells in the plate's own shape,
each lit by its state (:mod:`daphnia.wells`), the well the camera is over picked out by a
ring, and a legend that says what a light means.

Two consequences of painting rather than styling:

* a well is a circle with a rim, and ``QSS`` cannot draw either - so a tile is a widget
  with a ``paintEvent`` (:func:`paint_well`), and the tile's **shape** carries the
  information the way a list of labels never does;
* the colours are read from :mod:`daphnia.ui.theme` **at paint time**, so a desktop
  switch from light to dark repaints the plate correctly instead of leaving the wells in
  the other scheme (the trap ``StreamView`` names: never an inline stylesheet).

A page of the plate is a **shape**, not a list: :meth:`WellGrid.set_page` is given the
wells of one page and the plate's column count, and lays them out as the plate they came
from - so a 24-well page of a 96-well plate is two rows of it, and the operator can see
where the wells he is looking at sit on the plate he is holding.
"""

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImageReader,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

from daphnia import wells
from daphnia.ui import theme

#: State -> the theme token of a well's body. A state without a token would be a well
#: that cannot light up, so the two lists are kept side by side (and a test checks them).
STATE_TOKENS = {
    wells.PENDING: "well_empty",
    wells.CAPTURED: "well_captured",
    wells.MEASURED: "well_measured",
}

#: Bounds on one well *tile*, in pixels. Below the minimum a 96-well plate stops being
#: readable (the label needs the room); above the maximum the plate becomes a poster, and
#: the tiles are centred in the space instead of growing into it.
TILE_MINIMUM = 30
TILE_MAXIMUM = 118

#: Room reserved inside a tile for the well's label, under its circle.
LABEL_HEIGHT = 14

#: White space between a tile's edge and its circle.
WELL_PADDING = 3.0

#: How far outside the circle the "the camera is over this well" ring is drawn.
RING_OFFSET = 2.5

#: Margins and spacing of the grid itself.
GRID_MARGIN = 16
GRID_SPACING = 6

#: How wide the band of a well's *state* colour is when the well already has a picture.
#: The picture is drawn inside the circle and the state stays the rim around it, because
#: the two answer different questions: "what is in this well" and "where is this well in
#: the run".
BAND_WIDTH = 3.0

#: Thumbnails are decoded **at the size they are drawn** (a 12 MP frame is never turned
#: into a 12 MP pixmap to be shown at 80 px), and kept in a small cache: walking a plate
#: and turning pages re-draws the same wells, and re-reading a PNG per tile per repaint is
#: what would make a 96-well plate of captures feel broken.
#:
#: Measured on 2026-10-21 (M4, offscreen, this rig's 1280x720 PNGs): a full 96-well plate
#: is **1083 ms cold - 11.3 ms a well - and 0.01 ms cached**, which is why the cache exists
#: and why the page offers to draw the frames at all (`SR-STUDY2-012`).
THUMB_SIZE = 160
THUMB_CACHE_LIMIT = 160

_PIXMAP_CACHE = {}


def well_pixmap(path):
    """A small pixmap of the frame at ``path``, or ``None`` when it cannot be read.

    Decoding is scaled (``QImageReader.setScaledSize``), so the cost of drawing a plate of
    images is the *reading*, never a full-resolution pixmap per well. The cache is a plain
    dict with a cap: an unbounded one would hold a plate's worth of tiles forever, and a
    missing image is not an error - a well whose file has moved is still a captured well.
    """
    if not path:
        return None
    key = str(path)
    if key in _PIXMAP_CACHE:                        # a frame already read (or already lost)
        return _PIXMAP_CACHE[key]
    reader = QImageReader(key)
    if not reader.canRead():
        _PIXMAP_CACHE[key] = None                   # remembered: never read twice per run
        return None
    size = reader.size()
    if size.isValid() and size.width() and size.height():
        scale = max(size.width() / THUMB_SIZE, size.height() / THUMB_SIZE, 1.0)
        reader.setScaledSize(QSize(max(1, int(size.width() / scale)),
                                   max(1, int(size.height() / scale))))
    image = reader.read()
    pixmap = None if image.isNull() else QPixmap.fromImage(image)
    while len(_PIXMAP_CACHE) >= THUMB_CACHE_LIMIT:
        _PIXMAP_CACHE.pop(next(iter(_PIXMAP_CACHE)))    # oldest first: a plain FIFO cap
    _PIXMAP_CACHE[key] = pixmap
    return pixmap


def cover_source(pixmap, target):
    """The part of ``pixmap`` which fills ``target`` without distorting it (centre crop).

    A well is a circle and a frame is a rectangle: the honest fill is to enlarge the frame
    until it covers the circle and crop the rest, never to squash it into an oval - a
    distorted picture of a specimen is worse than no picture.
    """
    if not target.width() or not target.height() or not pixmap.width():
        return None
    scale = max(target.width() / pixmap.width(), target.height() / pixmap.height())
    width = min(pixmap.width(), int(target.width() / scale) + 1)
    height = min(pixmap.height(), int(target.height() / scale) + 1)
    return QRectF((pixmap.width() - width) / 2.0, (pixmap.height() - height) / 2.0,
                  width, height)


def paint_well(painter, rect, state, image=None, current=False, hovered=False) -> None:
    """Draw one well into ``rect``: body, its picture, rim, measured line, current ring.

    Shared by a tile and a legend swatch, so the key of the plate can never disagree with
    the plate it explains. ``painter`` must already have antialiasing on.

    ``image`` - the well's frame as a small pixmap, or ``None`` - is drawn **inside** the
    circle while the state's colour stays the band around it (`BAND_WIDTH`): the picture
    answers "what was captured here", the band answers "how far is this well in the run",
    and neither is dropped for the other.
    """
    body = QColor(theme.color(STATE_TOKENS.get(state, "well_empty")))
    if hovered and state != wells.PENDING:
        body = body.lighter(115)        # a lit well glows a little under the pointer
    painter.setBrush(body)
    rim = QPen(QColor(theme.color("well_rim")))
    rim.setWidthF(1.5)
    painter.setPen(rim)
    painter.drawEllipse(rect)

    if image is not None:
        inner = rect.adjusted(BAND_WIDTH, BAND_WIDTH, -BAND_WIDTH, -BAND_WIDTH)
        source = cover_source(image, inner)
        if source is not None and inner.width() > 2 and inner.height() > 2:
            clip = QPainterPath()
            clip.addEllipse(inner)
            painter.save()
            painter.setClipPath(clip)
            painter.drawPixmap(inner, image, source)
            painter.restore()
            painter.setBrush(Qt.NoBrush)
            painter.setPen(rim)
            painter.drawEllipse(inner)          # the hairline between picture and band

    if state == wells.MEASURED:
        # The one glyph on the plate: a measured well is drawn with the line that was
        # measured on it. It is the only state that means a number exists, and it is drawn
        # in the scheme's *ink* so it reads on the well's body in light and dark alike.
        line = QPen(QColor(theme.color("text")))
        line.setWidthF(max(1.5, rect.width() * 0.06))
        line.setCapStyle(Qt.RoundCap)
        painter.setPen(line)
        middle = rect.center()
        half = rect.width() * 0.18
        painter.drawLine(QPointF(middle.x() - half, middle.y()),
                         QPointF(middle.x() + half, middle.y()))

    if current:
        ring = QPen(QColor(theme.color("accent")))
        ring.setWidthF(2.0)
        painter.setPen(ring)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect.adjusted(-RING_OFFSET, -RING_OFFSET,
                                          RING_OFFSET, RING_OFFSET))


class WellTile(QWidget):
    """One well of the plate: a lit circle, its label, a ring while it is the current one.

    Signals:
        clicked(str): pressed with a left button - emitted with this tile's well label,
            so the grid (and the page above it) never has to know which tile it was.
    """

    clicked = Signal(str)

    def __init__(self, well_label, parent=None):
        super().__init__(parent)
        self.well_label = str(well_label)
        self._progress = wells.well_progress(self.well_label)
        self._image = None          # the well's frame as a small pixmap (or None)
        self._current = False
        self._hovered = False
        self.setObjectName("well_tile")
        # The grid decides a tile's size (it keeps the plate's own proportions), so a tile
        # never asks for a size of its own.
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self._refresh_tooltip()

    # -- state -------------------------------------------------------------
    @property
    def progress(self) -> wells.WellProgress:
        """What this well holds: the record the page shows and the tooltip repeats."""
        return self._progress

    def set_progress(self, progress_row):
        """Re-light the well from a :class:`~daphnia.wells.WellProgress`."""
        self._progress = progress_row
        self._refresh_tooltip()
        self.update()

    def set_image(self, image):
        """Draw ``image`` (a small pixmap of the well's frame) inside the well.

        ``None`` returns the well to its plain state colour - the honest picture of a well
        with no frame, and of one whose file can no longer be read.
        """
        self._image = image
        self.update()

    @property
    def image(self):
        """The pixmap drawn inside this well, or ``None``."""
        return self._image

    def set_current(self, current):
        """Mark this well as the one the camera is over (the ring, and a bold label)."""
        current = bool(current)
        if current != self._current:
            self._current = current
            self.update()

    @property
    def is_current(self) -> bool:
        """Whether this is the well the camera is over."""
        return self._current

    def _refresh_tooltip(self):
        """The well's own numbers, where the pointer already is."""
        self.setToolTip(f"{self._progress.describe()}\n"
                        f"{wells.STATE_HINTS[self._progress.state]}")

    # -- painting ----------------------------------------------------------
    def _well_rect(self) -> QRectF:
        """The circle: the largest one that fits the tile, leaving room for the label."""
        side = min(self.width() - 2 * WELL_PADDING,
                   self.height() - LABEL_HEIGHT - 2 * WELL_PADDING)
        side = max(4.0, float(side))
        top = (self.height() - LABEL_HEIGHT - side) / 2.0
        return QRectF((self.width() - side) / 2.0, top, side, side)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self._well_rect()
        paint_well(painter, rect, self._progress.state, image=self._image,
                   current=self._current, hovered=self._hovered)
        self._paint_label(painter, rect)

    def _paint_label(self, painter, rect):
        """The well's label under its circle: bold while it is the current well.

        The label is the well's **identity** (``A01``), so it is never elided or dropped -
        a plate whose wells cannot be named is a pattern, not a plate.
        """
        label_rect = QRectF(0, self.height() - LABEL_HEIGHT, self.width(), LABEL_HEIGHT)
        font = QFont(painter.font())
        font.setPointSizeF(max(6.5, min(10.0, rect.width() / 4.5)))
        font.setBold(self._current)
        painter.setFont(font)
        painter.setPen(QColor(theme.color("text" if self._current else "text_muted")))
        painter.drawText(label_rect,
                         int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                         self.well_label)

    # -- interaction -------------------------------------------------------
    def mousePressEvent(self, event):
        """A left press chooses the well - anywhere on the tile, not only on the circle."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.well_label)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)
        self._hovered = False
        self.update()
        super().leaveEvent(event)


class WellGrid(QWidget):
    """One page of a plate: wells drawn in the plate's own rows and columns.

    The page above it owns *which* wells are on screen (``daphnia.wells.pages``); this
    widget owns only how they look, so it can be exercised with a hand-written list.

    Signals:
        well_chosen(str): a well tile was clicked - with its label.
    """

    well_chosen = Signal(str)

    #: Smallest the drawing may become; a 96-well plate at the minimum is 12 x 30 px wide
    #: plus spacing, which still leaves the plate's shape readable.
    MINIMUM = QSize(320, 200)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("well_plate")
        self.setMinimumSize(self.MINIMUM)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(GRID_MARGIN, GRID_MARGIN, GRID_MARGIN, GRID_MARGIN)
        self._grid.setSpacing(GRID_SPACING)
        # The plate keeps its own size and is centred in whatever room the window gives
        # it: growing the tiles with the window is what turns a plate into a poster.
        self._grid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tiles = {}
        self._labels = ()
        self._columns = 0
        self._images = True         # draw each captured well's frame inside its circle

    # -- content -----------------------------------------------------------
    def set_page(self, labels, columns, counts=None, current=None, images=True):
        """Draw one page: ``labels`` in reading order, ``columns`` wells per row.

        ``counts`` is the mapping :func:`daphnia.db.queries.plate_well_counts` returns
        (well label -> row); a well that is not in it is drawn as empty, and the widget
        therefore needs no database at all. ``images`` draws each captured well's frame
        inside its circle (`SR-STUDY2-012`) - off, the plate is the pure state diagram.
        """
        self._clear()
        rows = counts or {}
        self._images = bool(images)
        self._labels = tuple(labels or ())
        self._columns = max(1, int(columns))
        for index, label in enumerate(self._labels):
            tile = WellTile(label, self)
            self._dress(tile, rows)
            tile.clicked.connect(self.well_chosen)
            self._grid.addWidget(tile, index // self._columns, index % self._columns)
            self._tiles[label] = tile
        self.set_current(current)
        self._resize_tiles()

    def set_progress(self, counts=None, current=None):
        """Re-light the wells from new counts, **without** rebuilding the grid.

        A burst does not change which wells are on the page, so re-lighting is what keeps
        a capture from blinking the plate (and throwing the pointer's hover away).
        """
        rows = counts or {}
        for label, tile in self._tiles.items():
            self._dress(tile, rows)
        if current is not None:
            self.set_current(current)

    def _dress(self, tile, rows):
        """Give one tile its record, and its picture when pictures are wanted."""
        row = rows.get(tile.well_label)
        tile.set_progress(wells.well_progress(tile.well_label, row))
        tile.set_image(self._image_of(row) if self._images else None)

    @staticmethod
    def _image_of(row):
        """The well's frame as a thumbnail, from a counts row (``None`` when there is none).

        The reading is tolerant on purpose: a row from a query that does not carry the
        column (an older caller, a hand-made mapping in a test) simply has no picture.
        """
        if row is None:
            return None
        try:
            path = row["image_path"]
        except (KeyError, IndexError):
            return None
        return well_pixmap(path)

    def set_current(self, well_label):
        """Ring the well the camera is over (and no other)."""
        for label, tile in self._tiles.items():
            tile.set_current(label == well_label)

    @property
    def tiles(self):
        """``{well label: tile}`` for the page on screen (also how a test reads it)."""
        return dict(self._tiles)

    @property
    def labels(self) -> tuple:
        """The wells of the page, in reading order."""
        return self._labels

    @property
    def columns(self) -> int:
        """How many wells one row of this page holds."""
        return self._columns

    @property
    def images(self) -> bool:
        """Whether the captured wells draw their frame."""
        return self._images

    def _clear(self):
        """Drop the tiles of the previous page."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._tiles = {}

    # -- geometry ----------------------------------------------------------
    def _shape(self):
        """``(rows, columns)`` of the page on screen."""
        if not self._labels or not self._columns:
            return 0, 0
        return -(-len(self._labels) // self._columns), self._columns

    def _resize_tiles(self):
        """Size the wells so the whole page fits the room, in the plate's proportions."""
        rows, columns = self._shape()
        if not rows:
            return
        room_width = (self.width() - 2 * GRID_MARGIN
                      - (columns - 1) * GRID_SPACING)
        room_height = (self.height() - 2 * GRID_MARGIN
                       - (rows - 1) * GRID_SPACING)
        side = min(room_width / columns, room_height / rows)
        side = max(TILE_MINIMUM, min(TILE_MAXIMUM, int(side)))
        for tile in self._tiles.values():
            tile.setFixedSize(QSize(side, side))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_tiles()


class WellLegend(QWidget):
    """The plate's key: one swatch per state, with how many wells are in it.

    Built in code because a legend is not a Designer control, and painted by the same
    :func:`paint_well` the tiles use - so the key and the plate it explains can never
    disagree about what a colour means.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("well_legend")
        row = QHBoxLayout(self)
        row.setContentsMargins(GRID_MARGIN, 4, GRID_MARGIN, 4)
        row.setSpacing(14)
        self.swatches = {}
        self.state_labels = {}
        for state in wells.STATES:
            swatch = WellSwatch(state, self)
            text = QLabel(f"{wells.STATE_LABELS[state]}  -", self)
            text.setObjectName("well_legend_label")
            text.setToolTip(wells.STATE_HINTS[state])
            row.addWidget(swatch)
            row.addWidget(text)
            row.addSpacing(6)
            self.swatches[state] = swatch
            self.state_labels[state] = text
        row.addStretch(1)
        self.plate_label = QLabel("No plate open.", self)
        self.plate_label.setObjectName("well_legend_label")
        row.addWidget(self.plate_label)

    def set_summary(self, counted, plate_text=None):
        """Show how many wells are in each state (a plate's summary, or one page's).

        The counts are the record's, never a guess, and a state with no wells still
        appears: "nothing measured yet" is exactly what the operator wants to see.
        """
        by_state = (counted or {}).get("by_state", {})
        for state, label in self.state_labels.items():
            label.setText(f"{wells.STATE_LABELS[state]}  {by_state.get(state, 0)}")
        if plate_text is not None:
            self.plate_label.setText(plate_text)


class WellSwatch(QWidget):
    """One dot of the legend: a well drawn with exactly the tiles' own painter."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setFixedSize(16, 16)
        self.setToolTip(wells.STATE_HINTS[state])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        paint_well(painter, rect, self.state)

