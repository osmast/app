"""Bench step - make a measurement now, without a study or a mode.

The fast path of the app: an image, two clicks, a number. Nothing here needs a project,
a plate or a phone, and nothing here asks the operator to *choose a mode first*: the
canvas' own **tools** (`daphnia/ui/widgets/image_canvas.py`) are what a press means, and
on the Bench they are exactly two - **Line** (click A, click B, and the line is a
measurement; several follow each other as A_1/B_1, A_2/B_2 ...) and **Box** (the
rectangle one individual stands in).

**This surface displays measurements, and nothing else.** Running a model is the Predict
surface's job (`SR-PREDICT-018`) and labelling one keypoint at a time is a **corpus**
workflow that lives on Dataset - so neither is offered here: there is no *Model* switch
and no *Keypoints* tool to reach for, and the tools that remain are the two ways of
*drawing* a distance or a box.

A measurement on the Bench lives in **the view of the image it was drawn on** until it is
saved (`SR-BENCH-022`): a view belongs to an image, so switching frames switches the view,
and stepping back to a frame brings its measurements back. Leaving the surface without
saving still loses what was not stored, and the panel says so. *Save this view to the
dataset* writes the rows into the open dataset (the same store every surface reads); *Save
this view to the study* writes them and attaches the image to a study in one go, so a
measurement taken here can belong to a project without being re-measured.

An image arrives three ways: a **file**, a single **live frame**, or a **burst** of live
frames (`SR-BENCH-023`) - a burst is written into a folder of its own under
``datasets/raw``, so each frame is its own image with its own view, the sharpest one can be
measured and the others stepped past, and the same folder can be brought back later with
*Import a folder...*.

**Every measurement is labelled where it was measured.** A line wears its own number and
value *above its middle*, and a measurement the operator has **named** wears that name
instead (`SR-BENCH-024`): a row's *Name* is typed in the panel, and the picture follows. One
animal's two lengths are therefore two readable lines, not one collected sentence at its
first point, and a measurement drawn outside every box is simply drawn and numbered
(`0.1`, `0.2` ...) - the words *no individual* are not written over the image, the panel is
where the groups are read (`SR-BENCH-016`).

**A box is an individual, and the measurements inside it are its own.** A frame can hold
several animals, and a number that cannot say *which* animal it belongs to cannot be
compared with anything. So a box drawn with the *Box* tool is the
**individual** the rest of the application already has - `observations.individual_index` on
the Study side, one box per specimen in the corpus, a `vision.tracking.Track` on the Predict
side - and every measurement whose **midpoint** falls inside it belongs to it
(`daphnia/vision/individuals.py` states the rule and `tests/test_individuals.py` proves it).
The list is therefore **grouped**: each individual with the size of its box and every number
measured on it, and a final *no individual* group for the measurements drawn outside every
box - kept and shown, never attached to the nearest animal. A box is still stored with its
width, height and area (trait ``bbox``), so *how big is this animal* is answered too, and
one animal may carry two ``length`` rows without the second erasing the first.

What the Bench *does* require is a scale, and it is honest about the alternative: with no
calibration the readout is in pixels and labelled **unscaled**, and only a scaled
measurement can be saved, because ``length_um`` without a scale would be a fabricated
number. Calibrating is one click away in the status footer, and works from a photo.
"""

from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QStyledItemDelegate,
                               QTreeWidgetItem)

from daphnia import annotation, maintenance, paths, settings, units
from daphnia.db import connection, queries, repository
from daphnia.study import ANATOMY, DEFAULT_ORGANISM_KIND
from daphnia.streaming import shared_service
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import (FILE_MENU, SOURCE_MENU, ImageCanvas, Kpi, StreamView,
                                TOOL_BOX, TOOL_LABELS, TOOL_LINE,
                                TOOL_SELECT, ToolSpec, specimen_color)
from daphnia.vision import individuals
from daphnia.vision.keypoints import Keypoints

#: What the measured distance *is*, in the study pipeline's own words. No study is
#: configured on the bench, so the default animal anatomy is used - the same definition
#: the study pipeline measures against, so a bench number and a study number mean the
#: same thing.
BENCH_ENDPOINT = ANATOMY[DEFAULT_ORGANISM_KIND]["endpoint"]

#: The endpoints of a **measured line** on the bench. The bench measures *a distance*,
#: not necessarily an anatomy: a colleague putting a ruler across a frame should not
#: have to call the ends "head" and "spine base". They are A and B, and several lines
#: on one frame are numbered A_1/B_1, A_2/B_2 ... by their specimen index. The *saved*
#: trait is still ``length`` with the real scale, so a bench line and a study length
#: remain the same kind of number (`projectbrief.md`, non-negotiable 1).
BENCH_LINE_POINTS = ("A", "B")

#: The only trait a *line* on the Bench produces. A bench measurement is a distance, and
#: every one of them is `length` with the scale in force - there is no trait to choose.
BENCH_TRAIT = "length"

#: The trait a **box** measurement travels under (`annotation.BOX_TRAIT`). A box is not a
#: distance between two landmarks - it is a rectangle - but it is one more thing measured on
#: a specimen, so it is stored as one more row of the same kind, with its geometry in the
#: width/height/area columns (`SR-BENCH-015`).
BENCH_BOX_TRAIT = annotation.BOX_TRAIT

#: Image files a **folder import** may hold - the same list the Dataset labeller
#: uses, so "which files are frames" is answered in one place.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".tif", ".tiff",
                  ".bmp", ".webp")

#: The **draw tools**, and the key each one wears: an image editor's rule, every tool is **one
#: key** away (`X-N18`). `L`ine and `B`ox - the two ways of *drawing* a measurement, and that
#: is all this surface draws. There is no *Keypoints* tool (one keypoint at a time is the
#: **corpus** workflow of the Dataset surface), no *Select* tool and no mode to switch:
#: **touching an object is always available** (`X-N20`), so the tools are about drawing and
#: nothing else. Either of them can also be turned **off** - and *no tool armed* is a state of
#: its own, the **normal view**, where a press selects what is under it and nothing is drawn.
BENCH_DRAW_TOOLS = (TOOL_LINE, TOOL_BOX)
BENCH_TOOL_KEYS = {
    TOOL_LINE: "L",
    TOOL_BOX: "B",
}

#: The column of ``list_results`` a measurement's **own name** is typed into
#: (`SR-BENCH-024`). The first two columns are a report - the number and the value - and are
#: not editable; this one is the operator's.
NAME_COLUMN = 2

#: What a row of the results tree says when the operator is asked what the *Name* column is
#: for. One sentence, in the place the names are typed.
NAME_HINT = ("Double-click a row's Name to call that measurement what you call it - the "
             "name is written above its line and saved with the row.")

#: What the canvas is set to when **no draw tool** is armed. The normal view is the *Select*
#: tool's behaviour - a press picks what is under it, or starts a marquee - so the page asks the
#: canvas for exactly that instead of inventing a fourth press-handling path. It is deliberately
#: **not** an entry of its own in the tool menu: nothing to draw is not a tool, it is the state
#: you are in before you reach for one (`X-N20`).
BENCH_NORMAL_VIEW = TOOL_SELECT

#: The role a row of `list_results` carries: the **row slot** of *this view* the row stands for
#: (its ``specimen``, the row key `(asset, slot, trait)`), which is how a typed name - or a click -
#: finds the measurement again. Never by position in the tree: the grouping decides that, and it
#: changes whenever a box is moved.
ROW_SPECIMEN = Qt.UserRole

#: The canvas index the measurements that belong to **no** individual are drawn and numbered
#: under. It is a real group - drawn, numbered and named - and it cannot collide with an
#: individual, whose numbers start at 1 (`vision.individuals`). The panel uses the same number
#: (`0.1`, `0.2` ...), so the picture and the list agree about a stray measurement too.
NO_INDIVIDUAL = 0


class _NameOnlyDelegate(QStyledItemDelegate):
    """A results row may be typed into **in its *Name* column and nowhere else** (`SR-BENCH-024`).

    The number and the value are a *report* - ``1.1``, ``2314.0 um (2.314 mm)`` - so an editor over
    them would invite a row that no longer says what was measured. The name is the operator's own,
    so it is the one cell that opens an editor; every other column refuses one, which is also what
    keeps a double click on a number from putting a box over it.
    """

    def createEditor(self, parent, option, index):
        if index.column() != NAME_COLUMN:
            return None
        return super().createEditor(parent, option, index)


class Bench(ShellPage):
    """One image, two clicks, one number - in this view, until you save it.

    Its controls panel needs the room the Dataset labeller's does: the dataset row
    holds a combo reading ``<name>  [<what it labels>]`` beside two buttons, and a
    QComboBox sizes itself to its longest item. The shared 420 px panel would clip it,
    so the measured 500 px (Dataset's own number) is used here too.
    """

    PANEL_WIDTH = 500

    def __init__(self, parent=None):
        super().__init__(ui_file='bench.ui', controls_width=self.PANEL_WIDTH,
                         parent=parent)

        self.service = shared_service()
        self.stream = self.add_visual(StreamView(self.service))
        self.canvas = self.add_visual(
            ImageCanvas(max_points=2, point_names=list(BENCH_LINE_POINTS)), stretch=3)
        self.canvas.points_changed.connect(self._on_points_changed)
        self.canvas.points_moved.connect(self._refresh_measure)
        # A **real click** placed a point: on the Bench the second one completes a line, so it
        # becomes a measurement in *this view* at once - no *Save* press between lines.
        self.canvas.point_placed.connect(self._on_point_placed)
        # A dropped point corrects the measurement it came from (`X-N16`).
        self.canvas.points_edited.connect(self._on_points_edited)
        # A **box** is a measurement too: drawn with the *Box* tool, and edited like any other.
        self.canvas.box_changed.connect(self._on_box_changed)
        self.canvas.box_edited.connect(self._on_box_edited)
        self.canvas.box_delete_requested.connect(self._on_box_delete_requested)
        # A line already drawn stays editable: the canvas reports which measurement a press
        # landed on, Bench loads that row back from *this view*, and the same press drags the
        # end (`X-N16`).
        self.canvas.annotation_edit = True
        self.canvas.annotation_point_selected.connect(self._on_annotation_point_selected)
        # A Bench box is a **measurement** with things inside it (the line/pair drawn on the
        # animal), so it is grabbed by its **outline** - a click inside reaches what is there
        # (`X-N17`).
        self.canvas.box_edge_grab = True
        # **Touching an object is not a mode** (`X-N20`): whatever draw tool is in force, a press
        # on a box or a measurement selects it, drags it, resizes it or removes it. The tool then
        # only decides what a press on *empty space* draws.
        self.canvas.editing = True
        # `Backspace` on a measurement removes it from this view (`X-N15`).
        self.canvas.annotation_delete_requested.connect(
            self._on_annotation_delete_requested)
        # The **tool follows the object that was clicked** (`X-N19`): a click on a box arms the
        # Box tool, on a measurement the measuring tool - in any mode.
        self.canvas.object_pressed.connect(self._on_object_pressed)

        # The page's own actions are menu items (`X-N13`): the tools that decide what a press
        # means, the measuring actions, and the *File* / *Source* entries.
        self.workspace = self.canvas_workspace(self.canvas)
        self._add_tools()
        self._add_measure_tools()
        self._add_source_tools()
        self._add_file_tools()

        self._frame = None            # the image being measured
        self._path = None             # the file behind it, when there is one
        #: **The views**, one per image: ``{image key: [items]}``. A view item is
        #: ``{"kind", "specimen", "trait", "p1", "p2", "name", ...}`` - a ``"box"`` item is an
        #: **individual** (its ``specimen`` is the index it answers to), a ``"line"`` item is
        #: one number measured on the frame, and ``"name"`` is the operator's own word for it
        #: (`SR-BENCH-024`: empty until they type one). Which individual a line belongs to is
        #: **derived from the geometry** every time the view is shown (`vision.individuals`),
        #: so moving a box takes the measurements inside it with it and no grouping can go stale.
        #:
        #: A view belongs to the **image** it was drawn on (`SR-BENCH-022`): switching frames
        #: switches the view, and switching back brings its measurements back, so a queue can
        #: be walked without losing what was measured on the frame behind. ``session`` is the
        #: view of the image on screen.
        self._views = {}
        self._key = ""                # the image on screen (set by `_install`)
        self._canvas_items = {}       # (index, segment) -> the measurement drawn there
        self._canvas_boxes = {}       # index -> the box item drawn there
        self._editing = None          # the session specimen whose pair is loaded for a drag
        self._loading = False         # guards programmatic canvas updates
        # The **dataset** this work is stored in - the same store every surface
        # writes, so a bench number is visible in the Dataset labeller and a study
        # measurement is visible here. A fresh database gets a default one.
        self._dataset = None
        # The **queue** of imported frames: a folder or a video brings in many, and
        # they are walked with A/D or a double click. One image is a queue of one.
        self._queue = []
        self._index = 0
        self._batch_id = None         # the recorded import this queue belongs to

        self.btn_save.clicked.connect(self._save_to_dataset)
        self.btn_save_study.clicked.connect(self._save_to_study)
        # The corpus choices - create a dataset, open another, rename one - are the
        # Dataset surface's own controls, so the two entry points cannot drift.
        self.combo_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        self.btn_new_dataset.clicked.connect(self.new_dataset)
        self.btn_rename_dataset.clicked.connect(self.rename_dataset)
        # The results tree's **name column** is the one cell an operator types into
        # (`SR-BENCH-024`): a name is written above that measurement's line on the picture, and
        # it is saved with the row. Nothing else in the tree is editable - the number and the
        # value are a report (`_NameOnlyDelegate`).
        self.list_results.setItemDelegate(_NameOnlyDelegate(self.list_results))
        self.list_results.itemChanged.connect(self._on_row_renamed)
        self.list_results.setToolTip(NAME_HINT)
        # `Backspace`/`Delete` remove whatever the *Select* tool has selected, wherever the
        # focus happens to be (`SR-DATASET-027`).
        for keys in (QKeySequence(Qt.Key_Backspace), QKeySequence.StandardKey.Delete):
            delete_shortcut = QShortcut(keys, self)
            delete_shortcut.activated.connect(self.canvas.delete_selection)
        # A **double click** on the image walks to the next frame of an imported queue,
        # and `A`/`D` (or the arrows) walk it either way - the same gesture the Dataset
        # labeller uses, so a folder or a video is measured frame by frame (`SR-BENCH-018`).
        self.canvas.double_clicked.connect(self.next_frame)
        for keys, delta in (("D", 1), ("Right", 1), ("A", -1), ("Left", -1)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(lambda delta=delta: self.step(delta))

        # The camera state decides whether a live frame is even available, so it is reported
        # here next to the image. A source the canvas could not read says so in the panel too.
        for signal in (self.service.status, self.service.opened, self.service.closed):
            signal.connect(self._refresh_image)
        self.canvas.source_failed.connect(self.status_label.setText)

        self._apply_tool()
        self._build_selectors()
        self.refresh()

    def on_shown(self):
        self.refresh()

    # -- the view of the image on screen -----------------------------------
    @property
    def session(self):
        """**This view** - the measurements made on the image on screen (`SR-BENCH-022`).

        A view belongs to an image, so this is the view of the frame on screen: walking a
        queue switches it, and stepping back restores what was measured on that frame. It
        is still a scratchpad in the sense that leaving without saving loses it (the panel
        says so) - but *switching frames* no longer does.
        """
        return self._views.setdefault(self._key, [])

    @session.setter
    def session(self, items):
        """Replace this image's view (used by Clear and by saving)."""
        self._views[self._key] = list(items)

    def view_of(self, key):
        """The measurements stored for one image key, or an empty list."""
        return self._views.get(key, [])

    def _view_key(self, path=None):
        """The identity of the image on screen: its file, or this live capture.

        A file is its own key, so reopening it - or stepping back to it in the queue -
        returns to the same view. A live frame that is not on disk yet is keyed by the
        moment it was taken, so two captures are two images with two views.
        """
        if path is not None:
            return str(Path(path))
        return f"live:{datetime.now().isoformat(timespec='microseconds')}"

    # -- the dataset (one corpus per model, shared with every entry point) --
    def _build_selectors(self):
        """Load the open dataset and list them all in the selector."""
        self._ensure_dataset()
        self._refresh_dataset_choices()

    def _ensure_dataset(self):
        """Load the open dataset, creating the default corpus on a fresh database."""
        with connection() as conn:
            dataset_id = repository.ensure_default_dataset(conn)
            self._dataset = queries.dataset_by_id(conn, dataset_id)
        self._refresh_dataset_label()

    def dataset(self):
        """The dataset row this work belongs to (never ``None`` once built)."""
        return self._dataset

    def dataset_id(self):
        """The id of the dataset a measurement belongs to, or ``None``."""
        return self._dataset["dataset_id"] if self._dataset else None

    def _refresh_dataset_label(self):
        """Show which corpus is open and what its model expects."""
        if self._dataset is None:
            self.dataset_label.setText("Dataset: -")
            return
        schema = annotation.get(self._dataset["schema_key"])
        self.dataset_label.setText(
            f"Dataset: {self._dataset['name']}\nModel: {schema.describe()}")

    def _refresh_dataset_choices(self, select=None):
        """List every dataset, the open one selected, each labelled with its model."""
        with connection() as conn:
            datasets = queries.list_datasets(conn)
        combo = self.combo_dataset
        # A combo sizes itself to its longest item, which would let a long dataset name
        # dictate the whole panel's width. The full name stays readable in the item's
        # tooltip and in the label below, so the box is capped instead.
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(20)
        combo.blockSignals(True)
        combo.clear()
        for row in datasets:
            schema = annotation.get(row["schema_key"])
            combo.addItem(f"{row['name']}  [{schema.label}]", row["dataset_id"])
            combo.setItemData(combo.count() - 1, schema.describe(), Qt.ToolTipRole)
        index = combo.findData(select) if select is not None else 0
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

    def _on_dataset_changed(self, *_):
        """Another dataset was picked: it becomes the one this work is stored in."""
        dataset_id = self.combo_dataset.currentData()
        if dataset_id and dataset_id != self.dataset_id():
            self.select_dataset(dataset_id)
            self.status_label.setText(
                f"Now working in '{self._dataset['name']}' - "
                f"{annotation.get(self._dataset['schema_key']).describe()}.")

    def select_dataset(self, dataset_id):
        """Open another dataset and make it the one this work is stored in."""
        with connection() as conn:
            repository.activate_dataset(conn, dataset_id)
        self._ensure_dataset()
        self._refresh_dataset_choices(select=dataset_id)
        self.refresh()
        return self._dataset

    def create_dataset(self, *, name, schema_key, organism="Daphnia magna",
                       note=None):
        """Create a corpus for one model and open it; returns its id."""
        with connection() as conn:
            dataset_id = repository.create_dataset(
                conn, name=name, schema_key=schema_key, organism=organism, note=note)
        self._ensure_dataset()
        self._refresh_dataset_choices(select=dataset_id)
        self.refresh()
        return dataset_id

    def new_dataset(self):
        """Ask for a new dataset (name + model) and open it."""
        from daphnia.ui.pages.dataset import DatasetDialog    # shared dialog
        dialog = DatasetDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return None
        dataset_id = self.create_dataset(
            name=dialog.dataset_name(), schema_key=dialog.schema_key(),
            organism=dialog.organism(), note=dialog.note())
        self.status_label.setText(
            f"Dataset '{self._dataset['name']}' created - "
            f"{annotation.get(self._dataset['schema_key']).describe()}.  "
            "Measurements are stored in it from now on.")
        return dataset_id

    def rename_dataset(self):
        """Rename the open dataset (its model and its labels stay as they are)."""
        from daphnia.ui.pages.dataset import DatasetDialog    # shared dialog
        dialog = DatasetDialog(self, dataset=self._dataset)
        if dialog.exec() != QDialog.Accepted:
            return None
        with connection() as conn:
            final = repository.rename_dataset(conn, self.dataset_id(),
                                              dialog.dataset_name())
        self._ensure_dataset()
        self._refresh_dataset_choices(select=self.dataset_id())
        self.status_label.setText(f"Dataset renamed to '{final}'.")
        return final


    # -- the tools ---------------------------------------------------------
    def tool(self) -> str:
        """The interaction tool in force on the canvas (`TOOL_LINE`, ...)."""
        return self.canvas.tool

    def _points(self) -> tuple:
        """The two landmark names of the measuring tool, in click order.

        The *Line* tool measures a distance, not an anatomy, so the ends are **A** and **B** - a
        colleague putting a ruler across a frame should not have to call the ends "head" and
        "spine base". The *saved* trait is still ``length`` with the real scale, so a bench line
        and a study length remain the same kind of number (``projectbrief.md``, non-negotiable 1).
        """
        return BENCH_LINE_POINTS

    def _endpoint(self) -> str:
        return BENCH_ENDPOINT

    def _prompt(self) -> str:
        if self.tool() == TOOL_LINE:
            return ("Click the A, then the B - the line joins this view the moment B lands.  "
                    "Keep clicking for A_2 / B_2 ... then save the view.")
        if self.tool() == TOOL_BOX:
            return ("Press a corner and drag the opposite one - and drag any box already on the "
                    "frame to move or resize it.")
        return ("No tool armed: click an object to select it, then drag it - or take a tool "
                "(L / B) to draw.  A click here never makes a measurement.")

    @staticmethod
    def _tool_tooltip(tool) -> str:
        """What each **draw** tool draws, in the operator's words (the header menu's tooltip)."""
        return {
            TOOL_LINE:
                "Measure a distance: click A, then B. The line joins this view at once - "
                "keep clicking for A_2/B_2, A_3/B_3 ...",
            TOOL_BOX:
                "Draw a box: press a corner, drag to the opposite one - the individual the "
                "numbers inside it belong to.",
        }.get(tool, "")

    def _add_tools(self):
        """The mutually-exclusive **draw tool** group - what a press on *empty space* draws.

        Each tool wears its **key** in the menu (`Line (L)`, `Box (B)`), the way an image
        editor's tools do, and the shortcut switches it from anywhere (`X-N18`) - and the *same*
        key turns the tool it names **off**, so *no tool armed* (the **normal view**, `X-N20`) is
        one key away too. The same actions are put **on the canvas** as a floating bar (`X-N20`),
        where the hand already is: one action, two affordances, so the bar and the menu can never
        disagree.
        """
        self.workspace.set_group("Draw")
        for tool in BENCH_DRAW_TOOLS:
            action = self.workspace.add_tool(
                ToolSpec(tool, self._tool_label(tool),
                         f"{self._tool_tooltip(tool)}  ({BENCH_TOOL_KEYS[tool]})",
                         checkable=True, checked=(tool == TOOL_LINE)))
            action.toggled.connect(
                lambda checked, key=tool: self._on_tool_changed(key, checked))
        self.workspace.set_tool_shortcuts(
            {key: tool for tool, key in BENCH_TOOL_KEYS.items()}, toggle=True)
        self.workspace.set_draw_tools([self.workspace.action(tool)
                                       for tool in BENCH_DRAW_TOOLS])

    @staticmethod
    def _tool_label(tool) -> str:
        """A tool's menu text, with the key that switches to it."""
        return f"{TOOL_LABELS[tool]} ({BENCH_TOOL_KEYS[tool]})"

    def _add_measure_tools(self):
        """The actions of *this view* - and **no model**: this surface displays measurements.

        A model is a checkpoint the lab loads and watches, which is the Predict surface's own
        job (`SR-PREDICT-018`); offering a *Model* switch here would make the Bench a second,
        quieter way to run one, with nothing in the panel saying so.
        """
        self.workspace.set_group("Measure")
        self.workspace.add_tool(
            ToolSpec("undo", "Undo", "Take back the last point, then the last measurement "
                                     "in this view."),
            lambda: self._tool_action(self.undo))
        self.workspace.add_tool(
            ToolSpec("clear", "Clear points", "Forget the points placed on this image - the "
                                              "view's measurements stay."),
            lambda: self._tool_action(self.clear_points))
        self.workspace.add_tool(
            ToolSpec("clear_view", "Clear this view", "Discard every measurement in this "
                                                      "view - saved ones stay saved."),
            lambda: self._tool_action(self.clear_view))
        self.workspace.add_tool(
            ToolSpec("refresh", "Refresh the list", "Re-read this view's measurements."),
            self.refresh)

    def _add_source_tools(self):
        self.workspace.set_group(SOURCE_MENU)
        self.workspace.add_tool(
            ToolSpec("use_frame", "Use the live frame",
                     "Measure what the camera sees right now."),
            self.use_live_frame)
        self.workspace.add_tool(
            ToolSpec("burst", "Take a burst",
                     f"Take {settings.BURST_SIZE} live frames into a folder of their own "
                     "under datasets/raw - each is its own image with its own view, A/D "
                     "walks them, and *Import a folder...* on it later gives the same "
                     "queue.  Pick the sharpest frame to measure the animal on."),
            self.burst)
        # The **live preview**: on while there is nothing to measure, off once a frame is
        # taken - the frozen image is then the whole visual area (`X-N16`). The camera keeps
        # running either way; *Use the live frame* still grabs it.
        preview_tool = self.workspace.add_tool(
            ToolSpec("live_preview", "Live preview",
                     "Show the live camera beside the image. Off after a frame is taken "
                     "(image mode) - the picture fills the area. The camera keeps running.",
                     checkable=True, checked=True))
        preview_tool.toggled.connect(self._on_live_preview_toggled)

    def _add_file_tools(self):
        self.workspace.set_group(FILE_MENU)
        self.workspace.add_tool(
            ToolSpec("open_photo", "Open a photo...",
                     "Measure any image file - no study, no plate, no phone needed."),
            self._open_photo)
        self.workspace.add_tool(
            ToolSpec("import_folder", "Import a folder...",
                     "Measure through every image under a folder, one after another."),
            self.import_folder)
        self.workspace.add_tool(
            ToolSpec("import_video", "Import a video...",
                     "Write a clip's frames to disk, then measure through them "
                     "(step with A/D or a double click)."),
            self._choose_video)
        self.workspace.add_tool(
            ToolSpec("export", "Export this dataset (CSV)",
                     "Write the saved measurements of the open dataset as a CSV file."),
            self._export)


    # -- the tool in force -------------------------------------------------
    def _on_tool_changed(self, tool, checked):
        """A **draw tool** was chosen - or turned **off**, which is the *normal view*.

        Switching tools is not a data change - *this view* is untouched - but it *does* drop
        the open points (they belong to the gesture that placed them) and any selection, so
        the next press cannot be misread.

        Unticking the tool in force is **not** an accident to be undone (it used to be: a radio
        group with nothing selected left the press undefined). It is the state the operator asks
        for by clicking the armed tool again: the **normal view**, where a press selects what is
        under it and nothing is drawn (`X-N20`) - and the canvas is given the *Select* tool's own
        behaviour for it (`BENCH_NORMAL_VIEW`), rather than a fourth press-handling path.
        """
        if self._loading:
            return
        if not checked:
            self.canvas.set_tool(BENCH_NORMAL_VIEW)
            self._apply_tool()
            self.status_label.setText(f"Normal view - {self._prompt()}")
            return
        for other in BENCH_DRAW_TOOLS:
            if other != tool:
                self.workspace.set_checked(other, False)
        self.canvas.set_tool(tool)
        self._apply_tool()
        self.status_label.setText(f"{TOOL_LABELS[tool]} - {self._prompt()}")

    def _apply_tool(self):
        """Show the points the tool in force places, then refresh the readout."""
        self._loading = True
        self.canvas.max_points = 2
        self.canvas.show_point_labels = True
        self.canvas.connect_points = True     # two points = one line
        self.canvas.skeleton = ()             # bench places a pair: one bone, 0 -> 1
        self.canvas.point_names = list(self._points())
        self.canvas.clear_points()
        self._loading = False
        self._editing = None
        self._refresh_keypoint_label()
        self._refresh_measure()
        self._refresh_overlays()

    def _on_object_pressed(self, tool):
        """The **tool follows the object that was clicked** (`X-N19`).

        Clicking a box arms the Box tool; clicking a measurement - or a model's keypoints -
        arms the measuring tool. It works in **any** mode, so the press that selects an object
        also leaves the right tool in force for it: the next press means what the object under
        the cursor needs, without a trip to the menu.

        The one exception is the **normal view** (`X-N20`): with no draw tool armed the operator
        has asked for a view where nothing is drawn, so a press **selects and arms nothing** -
        otherwise the normal view would last exactly one click.
        """
        if self.tool() == BENCH_NORMAL_VIEW:
            return
        if tool and tool != self.tool():
            self._switch_tool_quietly(tool)

    def _switch_tool_quietly(self, tool):
        """Change the tool **without disturbing the press** it followed (`X-N19`).

        Not :meth:`_on_tool_changed`: that is a *choice*, and it drops the open points and the
        selection. This keeps them, because the press that armed the tool is still working on
        exactly those objects.
        """
        for other in BENCH_DRAW_TOOLS:
            self.workspace.set_checked(other, other == tool)   # silent: no handler runs
        self.canvas.set_tool(tool, keep=True)
        self.canvas.point_names = list(self._points())
        self._refresh_keypoint_label()
        self._refresh_measure()
        self.status_label.setText(f"{TOOL_LABELS[tool]} - {self._prompt()}")

    def _refresh_keypoint_label(self):
        """Show the tool and its two landmarks - it is a display, not a control."""
        first, second = self._points()
        if self.tool() == TOOL_LINE:
            self.keypoint_label.setText(f"Line: {first} -> {second}")
        elif self.tool() == BENCH_NORMAL_VIEW:
            self.keypoint_label.setText("Normal view: nothing is drawn")
        else:
            self.keypoint_label.setText(f"Tool: {TOOL_LABELS[self.tool()]}")

    def _tool_action(self, action):
        """Run one tool, then repeat the panel's answer in the window **header**.

        A tool pressed on the image must answer where the facts are read: the number and
        the source live in the panel and in the header bar.
        """
        action()
        self._refresh_header()

    # -- measuring ---------------------------------------------------------
    def _on_points_changed(self, points):
        """A point was placed or cleared: refresh the live number."""
        self._refresh_measure()

    def _refresh_measure(self, *_):
        self.prompt_label.setText(self._prompt())
        self.length_label.setText(self._readout())
        self._refresh_header()

    def _readout(self) -> str:
        """The live number: micrometres when calibrated, pixels when not.

        Two uncertainties travel with it, because they answer different questions
        and only one of them was ever shown: the ``+/-`` after the length is the
        *calibration's* relative error, and ``total +/-`` is the whole budget of
        this length - that plus where the two landmarks were clicked and the pixel
        grid they had to land on (`units.total_uncertainty_um`,
        ``docs/accuracy.md`` §1). On this rig the click is the larger term, so a
        panel showing only the first would understate the number by 13x.
        """
        if self._frame is None:
            return "No image yet."
        points = self.canvas.points
        if len(points) < 2:
            placed = len(points)
            return (f"Click the second end to finish the line ({placed} of 2 points)."
                    if placed else self._prompt())
        keypoints = Keypoints(eye_apex=points[0], spine_base=points[1])
        profile = self._active_profile()
        owner = self._owner_text(points)
        if profile is None:
            return (f"{self._endpoint()}: {keypoints.length_px:.1f} px  (UNSCALED - no "
                    "calibration; save it and it would have no unit)" + owner)
        um = profile.to_um(keypoints.length_px)
        total = units.format_total_uncertainty(um, profile.px_per_um,
                                               profile.uncertainty_pct)
        return (f"{self._endpoint()}: "
                f"{units.format_with_uncertainty(um, profile.uncertainty_pct)}   "
                f"{total}   "
                f"({keypoints.length_px:.1f} px / {profile.px_per_um:.6f} px/um)" + owner)

    def _refresh_header(self):
        """Publish the facts the **window header** reads: which image, and the last message.

        The numbers are *not* among them: the length being measured is in the panel
        (``length_label``) and the individuals with their measurements are in the results
        tree, which is the list (`SR-BENCH-016`). What is on the image is this view's
        geometry - boxes, lines, points - and the draw bar in its corner; nothing else
        (`X-N13`).
        """
        self.workspace.set_header([
            Kpi("source", self.source_label.text(), label="Image"),
            Kpi("status", self.status_label.text(), label="Status"),
        ])

    # -- this view ---------------------------------------------------------
    def _on_point_placed(self, points):
        """A real click placed a point. The second one completes a line in *this view*.

        Only a genuine press reaches here (`point_placed`), so loading a stored row or a test
        calling `set_points` never adds a measurement by itself.
        """
        if self._loading or self._frame is None:
            return
        if self.tool() != TOOL_LINE:
            return
        if len(points) < 2:
            return
        self._add_measurement(points[0], points[1])

    def _add_measurement(self, p1, p2):
        """Append ``(p1, p2)`` to this view, then clear the canvas for the next line."""
        profile = self._active_profile()
        length_px = Keypoints(eye_apex=p1, spine_base=p2).length_px
        self.session.append({
            "kind": "line",
            "specimen": len(self.session) + 1,
            "trait": BENCH_TRAIT,
            "name": "",                     # the operator's own word for it (`SR-BENCH-024`)
            "p1": (float(p1[0]), float(p1[1])),
            "p2": (float(p2[0]), float(p2[1])),
            "length_px": length_px,
            "length_um": profile.to_um(length_px) if profile is not None else None,
        })
        self._loading = True
        self.canvas.clear_points()
        self._loading = False
        self._editing = None
        self._refresh_overlays()
        place = self._place_of(self.session[-1])       # the individual, and its own count
        if place is not None:
            self.canvas.select_annotation(*place)      # the new line is the selection
        self._refresh_results()
        self._refresh_measure()
        unit = ("unscaled (no calibration)" if profile is None
                else units.format_length(self.session[-1]["length_um"]))
        self.status_label.setText(
            f"Measurement {self._row_number(self.session[-1])} in this view: {unit}.  "
            "Not saved yet - save the view to keep it.")

    @staticmethod
    def _number_text(index, position=None) -> str:
        """``2`` for an individual, ``2.1`` for its first measurement, ``0.1`` for a stray one.

        One formatter for the canvas label, the panel rows and the status messages, so the three
        cannot disagree about what a number means (`SR-BENCH-016`).
        """
        who = str(NO_INDIVIDUAL) if index is None else str(index)
        return who if position is None else f"{who}.{position}"

    @staticmethod
    def _paint_row(row, index):
        """Give a panel row the **colour the canvas draws that individual in** (`specimen_color`).

        Colour is the identity the picture already uses, so a row and the object it describes are
        recognisable as the same thing without reading either one.
        """
        colour = QColor(specimen_color(NO_INDIVIDUAL if index is None else index))
        for column in (0, 1):
            row.setForeground(column, QBrush(colour))

    def _row_number(self, item) -> str:
        """The number the operator reads for one view item: ``2`` or ``2.1``.

        The **individual**, and - for a measurement - its own count inside it. The canvas draws
        the individual's number on the object and the panel lists the same number, so a line on
        the picture and a row in the list are recognisably the same thing (`SR-BENCH-016`); a
        measurement that belongs to nobody reads ``0.1``, the same group the canvas draws it in
        (:data:`NO_INDIVIDUAL`).
        """
        place = self._place_of(item)
        if place is None:
            return "-"
        index, segment = place
        return self._number_text(index, None if item.get("kind") == "box" else segment + 1)

    def _session_item(self, specimen):
        """The item of *this view* holding a **row slot**, or ``None``.

        The slot is the row key (`(asset, slot, trait)`), not the number the operator reads: this
        is how a line loaded for correction is found again (`_editing`), never how a canvas press
        is resolved (that is :meth:`_place_of`).
        """
        for item in self.session:
            if item["specimen"] == specimen:
                return item
        return None

    def _boxes(self):
        """This view's box items, in the order the individuals are numbered."""
        return [item for item in self.session if item.get("kind") == "box"]

    def _individuals(self):
        """The boxes in this view, as the **individuals** they stand for.

        A box *is* an individual (`vision.individuals`). The **number** it answers to is its
        position among the boxes - 1..N in the order they were drawn - so the operator reads
        *Individual 1, 2, 3* and never wonders where 2 went. It is *derived* here, like the
        grouping itself, and deliberately **not** the item's ``specimen`` slot: that one belongs
        to the row key (`(asset, slot, trait)`, which is what lets a box and the two lines in it
        hold three rows without colliding), and letting it name the animal made the individuals
        read 1, 3, 5 ... because every line consumed a number (`SR-BENCH-016`).
        """
        return [individuals.Individual(position,
                                      (item["p1"][0], item["p1"][1],
                                       item["p2"][0], item["p2"][1]))
                for position, item in enumerate(self._boxes(), start=1)]

    def _measured_count(self) -> int:
        """How many **numbers** this view holds - a box is an individual, not one of them."""
        return sum(1 for item in self.session if item.get("kind") != "box")

    def _individual_index(self, item):
        """Which individual a view item is **about**: a box is one, a line is *inside* one.

        ``None`` is a real answer - the measurement falls inside no box - and the list shows it
        as *No individual* rather than attaching it to the nearest animal
        (`vision.individuals`; ``mlContext.md`` law 8: a failure is named, never silent).
        """
        if item.get("kind") == "box":
            for position, box in enumerate(self._boxes(), start=1):
                if box is item:                  # identity: two boxes can hold equal geometry
                    return position
            return None
        found = individuals.individual_of(item["p1"], item["p2"], self._individuals())
        return found.index if found is not None else None

    def _grouped(self):
        """This view as the list reads it: ``[(individual number, box item, [measurements])]``.

        A box heads the numbers measured inside it; ``None`` heads the ones that belong to no
        individual. The grouping is **derived from the geometry every time**, so dragging a box
        over a line takes that line with it (`SR-BENCH-016`).
        """
        boxes = self._boxes()
        lines = [item for item in self.session if item.get("kind") != "box"]
        grouped = individuals.group(lines, self._individuals())
        rows = [(position, box, grouped.get(position, []))
                for position, box in enumerate(boxes, start=1)]
        if grouped.get(None):
            rows.append((None, None, grouped[None]))
        return rows

    @staticmethod
    def _group_title(number) -> str:
        """The header of one group: the individual, or the numbers that belong to nobody."""
        return "No individual" if number is None else f"Individual {number}"

    def _group_value(self, box, measurements) -> str:
        """The header's second column: the box's own size, and how many numbers sit under it."""
        count = f"{len(measurements)} measurement(s)"
        if box is None:
            return count
        return f"box {self._value_text(box)}   {count}"

    def _owner_text(self, points) -> str:
        """Which individual a pair on the canvas belongs to - said *while* it is placed.

        The operator sees whose number this is becoming **before** the second click lands, so a
        line that ends up outside every box is not a surprise in the list afterwards.
        """
        if len(points) < 2:
            return ""
        found = individuals.individual_of(points[0], points[1], self._individuals())
        if found is None:
            return "   -> no individual (no box contains it)"
        return f"   -> Individual {found.index}"

    # -- a measurement's own name (`SR-BENCH-024`) --------------------------
    def _name_of(self, item) -> str:
        """The operator's own word for one measurement, or ``""`` when it has none."""
        return str(item.get("name") or "").strip()

    def _on_row_renamed(self, item, column):
        """A row's *Name* was typed: the measurement behind that row takes the name.

        One handler for every row - a line and a box are both measurements of *this view* - and
        the tree is **not rebuilt** here: the editor that just committed belongs to the item, so
        re-making the rows would take the widget out from under Qt. The canvas, which has to
        follow the name, is redrawn instead (`_refresh_overlays`).
        """
        if self._loading or int(column) != NAME_COLUMN:
            return
        target = self._session_item(item.data(0, ROW_SPECIMEN))
        if target is None:
            return
        name = str(item.text(NAME_COLUMN) or "").strip()
        if name == self._name_of(target):
            return
        target["name"] = name
        item.setToolTip(NAME_COLUMN, name or NAME_HINT)
        self._refresh_overlays()
        self._refresh_measure()
        where = self._row_number(target)
        self.status_label.setText(
            f"Measurement {where} is named '{name}'." if name
            else f"Measurement {where} is unnamed again - it reads by its number ({where}).")

    def _value_text(self, item) -> str:
        """One measurement's value as a line the operator reads: a distance or a box."""
        if item.get("kind") == "box":
            if item["width_um"] is None:
                return (f"{item['width_px']:.1f} x {item['height_px']:.1f} px "
                        f"({item['area_px']:.0f} px^2, unscaled)")
            return (f"{units.format_length(item['width_um'])} x "
                    f"{units.format_length(item['height_um'])} = "
                    f"{item['area_um']:.0f} um^2")
        if item["length_um"] is None:
            return f"{item['length_px']:.1f} px (unscaled)"
        return units.format_length(item["length_um"])

    def _box_fields(self, box):
        """The measured geometry of a box: width, height, area (px and um) and the diagonal.

        The two corners are stored in ``p1``/``p2`` - so ``length_px`` is the **diagonal**,
        which is what keeps the ``NOT NULL`` distance columns honest for a box row
        (`SR-BENCH-015`).
        """
        profile = self._active_profile()
        x1, y1, x2, y2 = (float(value) for value in box)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        width, height = x2 - x1, y2 - y1
        diagonal = (width ** 2 + height ** 2) ** 0.5

        def um(value):
            return profile.to_um(value) if profile is not None else None

        return {
            "p1": (x1, y1), "p2": (x2, y2),
            "width_px": width, "height_px": height, "area_px": width * height,
            "width_um": um(width), "height_um": um(height),
            "area_um": (um(width) * um(height)) if profile is not None else None,
            "length_px": diagonal, "length_um": um(diagonal),
        }

    def _add_box(self, box):
        """Append a **box** measurement to this view, then clear the canvas box."""
        item = {"kind": "box", "specimen": len(self.session) + 1,
                "trait": BENCH_BOX_TRAIT,
                "name": ""}                 # the operator's own word for it (`SR-BENCH-024`)
        item.update(self._box_fields(box))
        self.session.append(item)
        self._loading = True
        self.canvas.clear_box()
        self._loading = False
        self._editing = None
        self._refresh_overlays()
        self.canvas.select_box(len(self._boxes()))    # the new box is the selection
        self._refresh_results()
        self._refresh_measure()
        self.status_label.setText(
            f"Individual {len(self._boxes())} in this view: {self._value_text(item)}.  Not "
            "saved yet - save the view to keep it.")

    def _on_box_changed(self, boxes):
        """The *Box* tool finished a box: it becomes a measurement in this view."""
        if self._loading or self._frame is None or not boxes:
            return
        if self.tool() != TOOL_BOX:
            return
        self._add_box(boxes[-1])

    def _on_box_edited(self, index, box):
        """A view box was dragged or resized: update that measurement in place.

        ``index`` is the **individual's number** - what the canvas draws and what the panel
        lists - so it is resolved through the map built with the overlays, never by row position.
        """
        if self._loading:
            return
        item = self._canvas_boxes.get(int(index))
        if item is None or item.get("kind") != "box":
            return
        item.update(self._box_fields(box))
        self._refresh_overlays()
        self._refresh_results()
        self._refresh_measure()
        self.status_label.setText(
            f"Individual {index} updated in this view: {self._value_text(item)}.  Not saved "
            "yet.")

    def _on_box_delete_requested(self, index):
        """`Backspace` on a **box** measurement: drop it from this view (nothing was saved).

        A box travels through the canvas' box layer, so the canvas asks with
        `box_delete_requested` - the same "the page decides" rule as the corpus boxes, and the
        measurement lands in the same view-removal as a line. The index is the individual's
        number, so the box it names is the one the overlays drew there.
        """
        self._remove_from_view(self._canvas_boxes.get(int(index)))

    def _on_annotation_delete_requested(self, index, segment):
        """`Backspace` on a measurement: drop it from this view (nothing was saved)."""
        self._remove_from_view(self._canvas_items.get((int(index), int(segment))))

    def _remove_from_view(self, item):
        """Remove one measurement from this view, keeping the rest's **row slots** 1..N.

        The slots are the row key (`(asset, slot, trait)`), so they are compacted on removal -
        otherwise the next measurement could be given a slot another one still holds, and the
        store would take the second for the first. The **individual** numbers are not touched:
        they are derived from the box order, so they renumber themselves.
        """
        if item is None:
            return
        kind = "Individual" if item.get("kind") == "box" else "Measurement"
        where = self._row_number(item)
        self.session = [entry for entry in self.session if entry is not item]
        for position, entry in enumerate(self.session, start=1):
            entry["specimen"] = position          # the row slots stay 1..N
        self._editing = None
        self.canvas.clear_points()
        self._refresh_overlays()
        self._refresh_results()
        self._refresh_measure()
        self.status_label.setText(f"Removed {kind.lower()} {where} from this view.")


    def _refresh_overlays(self):
        """Draw *this view*: every individual with its box, and **every line with its own label**.

        **The canvas is coloured and numbered by individual** (`SR-BENCH-016`). A box and the
        lines measured inside it wear the **same number and the same colour** - so the picture and
        the panel read as one thing - and a line is numbered by *whose it is*, never by a running
        count of rows. One annotation item per individual carries all of its segments, which is
        what makes a press report `(individual, segment)`: the page then loads exactly that line
        for correction (`X-N16`).

        Each segment carries **its own** label (`SR-BENCH-024`), written above that line's middle
        by the canvas: the operator's **name** for the measurement when they gave it one, else the
        number the panel lists it under and its value. So a measurement that belongs to no
        individual is drawn and numbered under :data:`NO_INDIVIDUAL` - ``0.1``, ``0.2`` - and
        nothing spells *no individual* over the photograph.

        The map from ``(index, segment)`` back to the view item is built **here**, with the
        drawing, so nothing has to be guessed from the order lines happen to be drawn in.
        """
        self._canvas_items = {}          # (index, segment) -> the measurement drawn there
        self._canvas_boxes = {}          # index -> the box item drawn there
        segments_by_index = {}
        tags_by_index = {}
        labels_by_index = {}
        for number, box, measurements in self._grouped():
            index = NO_INDIVIDUAL if number is None else number
            for segment, item in enumerate(measurements):
                self._canvas_items[(index, segment)] = item
                segments_by_index.setdefault(index, []).append((item["p1"], item["p2"]))
                tags_by_index.setdefault(index, []).append(item["trait"])
                labels_by_index.setdefault(index, []).append(self._line_label(item))
        for index, item in enumerate(self._boxes(), start=1):
            self._canvas_boxes[index] = item
        self.canvas.set_annotations(
            [(index, segments, "", tags_by_index[index], labels_by_index[index])
             for index, segments in segments_by_index.items()])
        self.canvas.set_boxes(
            [(index, (item["p1"][0], item["p1"][1], item["p2"][0], item["p2"][1]))
             for index, item in self._canvas_boxes.items()])

    def _line_label(self, item) -> str:
        """One measured line's label: **its name**, or the number the panel lists it under.

        A named measurement reads *Gut length: 2314.0 um* - the operator's own word, because a
        name they chose says more about what was measured than ``1.1`` does - and an unnamed one
        reads *1.1  2314.0 um*, the same number the panel's first column shows, so the picture and
        the list cannot disagree about which measurement a line is (`SR-BENCH-016`). The value
        itself is always there, in the shortest honest form (:meth:`_compact_value`): a name
        replaces the number, never the measurement.
        """
        value = self._compact_value(item)
        name = self._name_of(item)
        return f"{name}: {value}" if name else f"{self._row_number(item)}  {value}"

    def _compact_value(self, item) -> str:
        """One measurement in the shortest honest form - the canvas label and the overlay card.

        Used by both, so the picture and the overlay cannot drift apart; the full wording (mm
        beside µm, the calibration's own spread) stays in the panel's list and in the readout.
        """
        if item.get("kind") == "box":
            if item["width_um"] is None:
                return f"{item['width_px']:.0f} x {item['height_px']:.0f} px"
            return f"{item['width_um']:.0f} x {item['height_um']:.0f} um"
        if item["length_um"] is None:
            return f"{item['length_px']:.1f} px"
        return f"{item['length_um']:.1f} um"

    def _place_of(self, item):
        """Where a view item was **drawn**: ``(index, segment)``, or ``None``.

        Read from the map :meth:`_refresh_overlays` built, never recomputed - the canvas and the
        page must agree about which object a press meant.
        """
        for place, drawn in self._canvas_items.items():
            if drawn is item:
                return place
        for index, box in self._canvas_boxes.items():
            if box is item:
                return index, 0
        return None

    def _on_annotation_point_selected(self, index, segment, endpoint):
        """A drawn line was clicked: load *this view's* measurement so the end can be dragged.

        The correction is written back into the view when the point is let go
        (`_on_points_edited`) - nothing is stored until the view is saved. ``(index, segment)``
        is where the line was **drawn** - the individual and the measurement's own count - so the
        line the operator pressed is the line that is loaded, whatever order the rows hold.
        """
        item = self._canvas_items.get((int(index), int(segment)))
        if item is None:
            return
        self._loading = True
        self._editing = item["specimen"]              # the row slot, for `_on_points_edited`
        self.canvas.set_points([item["p1"], item["p2"]])
        self.canvas.select_point(endpoint)
        self._loading = False
        self._refresh_measure()
        self.status_label.setText(
            f"Editing measurement {self._row_number(item)} in this view - drag the point; it is "
            "updated when you let go.")


    def _on_points_edited(self):
        """A point was dragged and let go: update the measurement it came from in this view."""
        if self._loading or self._editing is None:
            return
        item = self._session_item(self._editing)
        points = self.canvas.points
        if item is None or len(points) < 2:
            return
        profile = self._active_profile()
        length_px = Keypoints(eye_apex=points[0], spine_base=points[1]).length_px
        item["p1"] = (float(points[0][0]), float(points[0][1]))
        item["p2"] = (float(points[1][0]), float(points[1][1]))
        item["length_px"] = length_px
        item["length_um"] = profile.to_um(length_px) if profile is not None else None
        self._refresh_overlays()
        self._refresh_results()
        self._refresh_measure()
        self.status_label.setText(
            f"Line {self._editing} updated in this view.  Not saved yet.")

    def undo(self):
        """Undo the last thing placed: a loose point first, then the last view measurement."""
        if self.canvas.points:
            self.canvas.clear_points()
            self.status_label.setText("Took back the point.")
            return
        if self.session:
            removed = self.session.pop()
            where = self._row_number(removed)
            self._editing = None
            self._refresh_overlays()
            self._refresh_results()
            self._refresh_measure()
            self.status_label.setText(f"Took back {where} from this view.")
            return
        self.status_label.setText("Nothing to take back.")

    def clear_points(self):
        """Forget the points placed on this image (never the view's measurements)."""
        self.canvas.clear_points()
        self._editing = None
        self._refresh_measure()

    def clear_view(self):
        """Discard *this view* - what was saved to the dataset stays saved."""
        removed = len(self.session)
        self.session = []
        self._editing = None
        self.canvas.clear_points()
        self._refresh_overlays()
        self._refresh_results()
        self._refresh_measure()
        self.status_label.setText(
            f"Cleared {removed} measurement(s) from this view.  Saved measurements stay "
            "saved." if removed else "This view is already empty.")

    # -- saving ------------------------------------------------------------
    def _active_profile(self):
        """The calibration in force, or ``None`` when nothing is calibrated."""
        with connection() as conn:
            return repository.active_profile(conn)

    def _no_scale_message(self) -> str:
        return ("No calibration, so there is no unit to store this in.  Press Calibrate in "
                "the bar below - it only needs a photo of the 1 mm grid - then measure again.")

    def _save_to_dataset(self):
        """Store every measurement in this view in the **open dataset**, then clear it."""
        name = self._dataset["name"] if self._dataset else "the dataset"
        self._persist_view(None, f"the dataset '{name}'")

    def _save_to_study(self):
        """Store this view's measurements and attach the image to the chosen study."""
        study_id = self.combo_study.currentData()
        if study_id is None:
            self.status_label.setText(
                "There is no study to save into yet.  Create one in the Study surface, then "
                "come back - or save to the dataset instead.")
            return
        self._persist_view(study_id, f"'{self.combo_study.currentText()}'")


    def _persist_view(self, study_id, where):
        """Write ``self.session`` to the store, optionally onto a study, then clear the view.

        The whole view is written through **one** observation (one image), so attaching it to
        a study attaches exactly this photo's measurements - and each measurement keeps the
        scale it was taken under (`projectbrief.md`, non-negotiable 1) **and the name the
        operator gave it** (`SR-BENCH-024`, stored in ``corpus_labels.label_name`` and carried
        by the Bench CSV).

        Every row is a **corpus label of a dataset**: the study's own when saving onto a
        study, and otherwise the dataset open at the top of the panel - so the same work is
        visible from the Dataset entry point and the CSV export reads one store, not two.
        """
        if not self.session:
            self.status_label.setText("There is nothing in this view to save yet.")
            return
        profile = self._active_profile()
        if profile is None:
            self.status_label.setText(self._no_scale_message())
            return
        path = self._ensure_file()
        if path is None:
            return
        dataset_id = None if study_id is not None else self.dataset_id()
        with connection() as conn:
            observation_id = repository.bench_observation(
                conn, file_path=path, profile_id=profile.profile_id)
            asset = repository.bench_asset(conn, observation_id)
            asset_id = asset["asset_id"] if asset is not None else None
            for item in self.session:
                box = None
                if item.get("kind") == "box":
                    box = (item["width_px"], item["height_px"], item["area_px"],
                           item["width_um"], item["height_um"], item["area_um"])
                repository.save_measurement(
                    conn, observation_id=observation_id, asset_id=asset_id,
                    profile=profile, trait=item["trait"],
                    specimen_index=item["specimen"], p1=item["p1"], p2=item["p2"],
                    box=box, individual_index=self._individual_index(item),
                    # The operator's own name for it travels with the row (`SR-BENCH-024`),
                    # so a named measurement is not re-numbered by being saved.
                    name=self._name_of(item) or None,
                    dataset_id=dataset_id)
            if study_id is not None:
                repository.attach_observation_to_study(
                    conn, observation_id=observation_id, study_id=study_id)
        saved = len(self.session)
        self.session = []
        self._editing = None
        self.refresh()
        self.status_label.setText(
            f"Saved {saved} measurement(s) on {Path(path).name} to {where}.")

    def _ensure_file(self):
        """A storable file for the current image, or ``None`` if it cannot be written.

        A live frame is written to ``datasets/raw`` the first time it is saved, so a saved
        bench result always has an image behind it - and a quick look that is never saved
        leaves nothing behind.
        """
        if self._path is not None:
            return self._path
        paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = paths.RAW_DIR / f"bench_{stamp}.png"
        if not cv2.imwrite(str(target), self._frame):
            self.status_label.setText(f"Could not write {target} - not saved.")
            return None
        self._path = target
        self.source_label.setText(
            f"Image: {target.name} (captured from the live frame)")
        return target

    # -- the image ---------------------------------------------------------
    def refresh(self):
        """Re-read the image, the scale in force, the overlays and this view's list."""
        self._refresh_image()
        self._refresh_overlays()
        self._refresh_results()

    def _refresh_image(self, *_):
        self.stream_hint_label.setText(
            f"Live feed: {self.service.state_label()}.  Open any photo file, or use "
            "the live frame.")
        if self._path is None and self._frame is None:
            self.source_label.setText("No image yet.")
        profile = self._active_profile()
        if profile is None:
            self.scale_label.setText(
                "Scale: none - results read in pixels only.  Press Calibrate in the "
                "bar below (a photo of the 1 mm grid is enough) to get micrometres.")
        else:
            self.scale_label.setText(
                f"Scale: {profile.px_per_um * 1000.0:.1f} px/mm "
                f"+/- {profile.uncertainty_pct:.2f}%")
        self._refresh_measure()


    def _open_photo(self):
        """Ask for a photo, then use it. The choosing and the loading stay separate."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open an image to measure", str(paths.DATASETS_DIR),
            "Images (*.png *.jpg *.jpeg *.heic *.heif *.tif *.tiff *.bmp *.webp)"
            ";;All files (*)")
        if path:
            self.open_image(path)

    def open_image(self, path):
        """Load this image file - any format the machine can read.

        Dialog-free on purpose, so the loading half can be driven directly.
        """
        self._install(self.canvas.set_source(path), Path(path))
        self._enter_image_mode()

    # -- the queue: a folder or a video brings many frames ------------------
    def import_folder(self):
        """Bring in every image under a folder, recursively, and walk them."""
        folder = QFileDialog.getExistingDirectory(
            self, "Import a folder of frames", str(paths.RECORDINGS_DIR))
        if not folder:
            return
        found = sorted(path for path in Path(folder).rglob("*")
                       if path.suffix.lower() in IMAGE_SUFFIXES)
        if not found:
            self.status_label.setText(f"No image files found under {folder}.")
            return
        self.add_paths(found, source=folder, kind="folder")
        self.status_label.setText(
            f"Imported {len(found)} frame(s) from {folder} - one after another.")

    def _choose_video(self):
        """Ask for a video file, then import its frames."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import a video to measure through", str(paths.RECORDINGS_DIR),
            "Video (*.mp4 *.mov *.m4v *.avi *.mkv *.ts);;All files (*)")
        if path:
            self.import_video(path)

    def import_video(self, path):
        """Write a clip's frames to disk under ``datasets/raw/<stem>/`` and queue them.

        A video is not a file you can measure - the frames are - so they are
        materialised once and each keeps its frame number in the file name, which is
        how the queue stays in frame order (``daphnia/vision/video.py``).
        """
        from daphnia.vision import video as video_module
        target = paths.RAW_DIR / Path(path).stem
        try:
            frames = video_module.extract_frames(path, target)
        except ValueError as error:
            self.status_label.setText(str(error))
            return
        if not frames:
            self.status_label.setText(
                f"No frames could be read from {Path(path).name}.")
            return
        self.add_paths(frames, source=str(path), kind="video")
        self.status_label.setText(
            f"Imported {len(frames)} frame(s) from {Path(path).name} into {target}.")

    def add_paths(self, paths_in, *, source=None, kind="folder"):
        """Replace the queue with these frames, in a fixed order, and show the first.

        The import is **recorded** (``import_batches``/``imported_frames``): where the
        frames came from, a video frame's number, and where the operator stopped - so
        a session is resumable and re-importing the same source is the same work.
        """
        unique = []
        for path in paths_in:
            text = str(Path(path))
            if text not in unique:
                unique.append(text)
        self._queue = [Path(text) for text in sorted(unique)]
        self._index = 0
        with connection() as conn:
            self._batch_id = repository.open_import(
                conn, kind=kind, paths=self._queue, source=source,
                dataset_id=self.dataset_id())
        self._show_current()

    def _show_current(self):
        """Load the frame at the current position of the queue."""
        if not self._queue:
            return
        self.open_image(str(self._queue[self._index]))
        self._refresh_position()

    def _refresh_position(self):
        """Say where in the queue the operator is, when there is more than one frame."""
        if len(self._queue) <= 1:
            return
        self.status_label.setText(
            f"Frame {self._index + 1} of {len(self._queue)}: "
            f"{self._queue[self._index].name}  (A/D or a double click walks the queue)")

    def step(self, delta):
        """Walk the imported frames by ``delta`` (clamped at both ends)."""
        if not self._queue:
            self.status_label.setText(
                "No frames in the queue - import a folder or a video to walk through.")
            return
        self._index = max(0, min(self._index + delta, len(self._queue) - 1))
        if getattr(self, "_batch_id", None):
            with connection() as conn:
                repository.resume_import(conn, self._batch_id, self._index)
        self._show_current()

    def next_frame(self):
        """A double click on the image walks to the next frame - and stores nothing."""
        self.step(1)

    def use_live_frame(self):
        """Measure what the camera sees now.

        Taking the frame **is** switching to image mode: the frozen picture is what will be
        measured, so the moving video steps aside (`X-N16`). The frame is written to disk only
        if the measurement is saved, so a quick look leaves nothing behind.
        """
        frame = self.service.current_frame()
        if frame is None:
            self.status_label.setText(
                f"No live frame.  {self.service.state_label()}  Connect in the bar "
                "below, or open a photo file instead.")
            return
        self.canvas.set_source(frame)
        self._install(frame, None)
        self._enter_image_mode()

    def burst(self):
        """Take a **burst** of live frames into a folder of its own, then open it.

        A burst is several *images*, not one, and it is written into
        ``datasets/raw/<burst>/`` - a real folder in the dataset tree - so the very same
        folder can be brought back later with *Import a folder...* and gives exactly the
        same queue and the same per-image views (`SR-BENCH-023`). Each frame keeps its own
        view, so the sharpest one can be measured and the others stepped past. Unlike a
        single *Use the live frame* - written only when a measurement is stored
        (`SR-BENCH-012`) - a burst is a deliberate capture, so its frames are on disk at
        once.
        """
        frames = self.service.recent_frames()
        if not frames:
            self.status_label.setText(
                f"No live frames.  {self.service.state_label()}  Connect in the bar "
                "below, or open a photo file instead.")
            return
        chosen = frames[-max(1, int(settings.BURST_SIZE)):]
        folder = paths.burst_dir(datetime.now().strftime("burst_%Y%m%d_%H%M%S"))
        folder.mkdir(parents=True, exist_ok=True)
        written = []
        for index, frame in enumerate(chosen):
            target = folder / f"f{index:02d}.png"
            if cv2.imwrite(str(target), frame):
                written.append(target)
        if not written:
            self.status_label.setText(
                f"Could not write the burst to {folder} - nothing captured.")
            return
        # The **same path a folder import takes**, with the burst's own folder as the
        # source - so a burst and an import produce the same queue, the same per-image
        # views and the same store, and the folder can be reopened later.
        self.add_paths(written, source=str(folder), kind="folder")
        self._enter_image_mode()
        self.status_label.setText(
            f"Took a burst of {len(written)} frame(s) into {folder.name}/ - A/D walks "
            "them, and *Import a folder...* on it later gives the same queue.")

    def _install(self, frame, path, key=None):
        """Install the image being measured - **and its own view** - or forget it.

        A view belongs to an image (`SR-BENCH-022`), so installing another image switches
        to *that* image's view: whatever was measured on the frame behind stays with it and
        comes back when the operator steps back to it. With ``frame=None`` there is *no*
        image and therefore nothing to measure.
        """
        self._frame = frame
        self._path = Path(path) if (frame is not None and path is not None) else None
        self._editing = None
        if frame is None:
            self._key = ""            # no image on screen, so no view
            self.canvas.clear_points()
            self.refresh()
            return
        self._key = key or self._view_key(self._path)
        self.canvas.clear_points()
        height, width = frame.shape[:2]
        name = self._path.name if self._path is not None else "live frame (not saved yet)"
        self.source_label.setText(f"Image: {name}  ({width} x {height})")
        kept = len(self.view_of(self._key))
        if kept:
            self.status_label.setText(
                f"This image already carries {kept} measurement(s) - they are in this "
                "view again.")
        # **Everything** of the new view: the drawing on the picture (`_refresh_overlays`)
        # as well as the list - without it the measurements of the frame stepped back to
        # were in the store but not on screen.
        self.refresh()

    def _enter_image_mode(self):
        """Hide the live preview: from here on, the thing on screen is an *image*.

        The toggle is set **silently** (`set_checked` does not re-run the handler), so the
        menu's check mark tells the truth about the state the page just chose without
        double-handling it.
        """
        self.stream.set_preview_enabled(False)
        self.workspace.set_checked("live_preview", False)

    def _on_live_preview_toggled(self, checked):
        """The *Live preview* switch: show the camera beside the image, or stand it down."""
        self.stream.set_preview_enabled(bool(checked))
        if checked:
            self.status_label.setText(
                "Live preview on - the camera is beside the image.  Take a frame with "
                "'Use the live frame'.")


    # -- this view's list and export ---------------------------------------
    def _refresh_results(self):
        """Show *this view*: the measurements made since the Bench was opened, not yet saved.

        What was saved to the dataset is a different list (the CSV export and the Dataset
        labeller read it);
        here the operator sees exactly what is at stake if they leave without saving. It is
        **grouped by individual** (`SR-BENCH-016`): each box heads the numbers measured inside
        it - with its own size as the group's fact - and a measurement drawn outside every box
        is listed under *No individual*, kept and visible rather than attached to the nearest
        animal.

        The third column is the measurement's **own name** (`SR-BENCH-024`) - the one cell an
        operator types into, while the number and the value are a report - and it is written
        above that line on the picture, so a name given here is visible where the measurement is.
        """
        # Rebuilding the rows re-writes every item, which is what `itemChanged` reports: the
        # guard says "this is the page, not the operator" (`_on_row_renamed`).
        previous = self._loading
        self._loading = True
        try:
            self.list_results.clear()
            for number, box, measurements in self._grouped():
                header = QTreeWidgetItem([self._group_title(number),
                                          self._group_value(box, measurements)])
                self._paint_row(header, number)
                if box is not None:
                    header.setData(0, ROW_SPECIMEN, box["specimen"])
                self.list_results.addTopLevelItem(header)
                for position, item in enumerate(measurements, start=1):
                    child = QTreeWidgetItem(
                        [f"{self._number_text(number, position)}   "
                         f"{annotation.trait_label(item['trait'])}",
                         self._value_text(item),
                         self._name_of(item)])
                    self._paint_row(child, number)
                    child.setData(0, ROW_SPECIMEN, item["specimen"])
                    child.setToolTip(NAME_COLUMN, self._name_of(item) or NAME_HINT)
                    child.setFlags(child.flags() | Qt.ItemIsEditable)
                    header.addChild(child)
                header.setExpanded(True)
        finally:
            self._loading = previous
        # Which image this view belongs to (`SR-BENCH-022`) - the operator must be able to
        # tell the frame's own measurements from the one behind's, especially in a queue.
        where = ""
        if len(self._queue) > 1:
            where = f"  Frame {self._index + 1} of {len(self._queue)}."
        elif self._path is not None:
            where = f"  On {self._path.name}."
        if self.session:
            self.count_label.setText(
                f"{len(self._individuals())} individual(s), {self._measured_count()} "
                "measurement(s) in this view - not saved yet." + where)
        else:
            self.count_label.setText("No measurements in this view yet." + where)
        self.unsaved_label.setVisible(bool(self.session))
        self.btn_save.setEnabled(bool(self.session))
        self.btn_save_study.setEnabled(bool(self.session))

        with connection() as conn:
            studies = queries.list_studies_with_progress(conn)
        self.combo_study.clear()
        for study in studies:
            self.combo_study.addItem(study["name"], study["study_id"])
        if not studies:
            self.combo_study.addItem("(no study exists yet)", None)

    def _export(self):
        """Write **this dataset's** saved measurements as CSV, landmark coordinates included."""
        dataset_id = self.dataset_id()
        with connection() as conn:
            if not queries.bench_count(conn, dataset_id=dataset_id):
                self.status_label.setText(
                    "Nothing saved in this dataset yet - save this view first.")
                return
        name = self._dataset["name"] if self._dataset else "dataset"
        suggested = str(paths.DATASETS_DIR / f"{name}_measurements.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export this dataset's results", suggested, "CSV (*.csv)")
        if not path:
            return
        with connection() as conn:
            written = maintenance.export_bench_csv(conn, path, dataset_id=dataset_id)
        self.status_label.setText(
            f"Exported {written} saved measurement(s) of this dataset, with their "
            f"landmark coordinates, to {path}.")

