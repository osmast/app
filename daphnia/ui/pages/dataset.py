"""Dataset step - build and label the training corpus.

The thesis surface: it produces the ground truth the keypoint model trains on, so
it does the opposite of the Study surface's "exactly one frame per target" - many
frames, labelled at volume, and **nothing** auto-selected or deleted.

**A dataset is one corpus per model.** Which model is chosen (see
:mod:`daphnia.annotation`) decides what a label *is*: two keypoints (head ->
tail), four (head, tail, sideL, sideR), the bounding box around the specimen, or
a combination. So "train another model" means *create another dataset* - the
labels of one corpus are never re-purposed for another model, and the same frame
may carry ground truth for several of them.

A label here is **scale-free**: it records the clicked keypoints (and the box) in
*source-image pixels*, so a frame can be labelled with no study, plate, connection
or calibration (``SR-DATASET-N01``). The calibration in force, when there is one,
is stored alongside so the label can also report a length in micrometres - but the
corpus never *requires* a scale, because the model learns pixels and archived or
third-party images must be usable too.

Frames come from a folder on disk (or a single file, or the live frame). The queue
is deterministic and walks with Previous/Next and the keyboard, because labelling
speed is the binding constraint on the corpus, not the maths.
"""

from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QWidget)

from daphnia import annotation, labeling, maintenance, paths, units
from daphnia.db import connection, queries, repository
from daphnia.streaming import shared_service
from daphnia.study import TRAITS
from daphnia.ui.model_pass import ModelRunner, primary
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import (
    FILE_MENU,
    SOURCE_MENU,
    ImageCanvas,
    Kpi,
    StreamView,
    ToolSpec,
)
from daphnia.vision.crop import crop_for_box
from daphnia.vision.export import ExportError
from daphnia.vision.keypoints import Keypoints

#: What each trait means anatomically, for the operator prompt. The trait keys come
#: from :mod:`daphnia.study`, so a corpus label and a study measurement always name
#: the same points - and a dataset can only offer the traits its model measures.
DATASET_TRAIT_HINTS = {
    "length": "anterior margin of the compound eye (head) -> base of the apical spine",
    "width": "left side -> right side, across the carapace",
    annotation.BOX_TRAIT: "drag a rectangle around the specimen",
}

#: How many points a landmark *pass* may hold - one per specimen on the frame.
PASS_MAX_POINTS = 500

#: Image files a corpus folder may hold.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".tif", ".tiff",
                  ".bmp", ".webp")

#: What the *Open boxes export* dialog accepts: the file an export carries its
#: boxes in, with the ``images/`` folder beside it.
BOXES_FILTER = "Boxes export (boxes.csv labels.json);;All files (*)"


def suggested_images_dir(source) -> Path:
    """Where an export's images usually are, given its boxes file or folder.

    An export keeps them in ``images/`` beside the file, so that is the suggestion -
    and when there is no such folder the source's own folder is offered instead. It is
    only ever a **suggestion** the dialog pre-fills: the operator confirms or changes
    it, because a corpus can perfectly well have its file and its pictures in two
    different places (``SR-DATASET-022``).
    """
    path = Path(source).expanduser()
    folder = path if path.is_dir() else path.parent
    sibling = folder / "images"
    return sibling if sibling.is_dir() else folder


class BoxesExportDialog(QDialog):
    """Ask for the two halves of an existing box corpus: the file, and the images.

    A ``boxes.csv`` names its frames **relative** to the export folder, so opening one
    means saying where the pictures are as well - and the operator may well have the
    file in one place and the frames in another, or want to point at the ``images/``
    folder they can actually see in a file browser. So both are asked for, in plain
    fields, with the images folder pre-filled from the file (never guessed silently):
    the corpus is a *view* over the original frames, and this dialog is where that is
    stated.
    """

    def __init__(self, parent=None, source=None, images_dir=None):
        super().__init__(parent)
        self.setWindowTitle("Open a boxes export")
        self.file_edit = QLineEdit(str(source) if source else "")
        self.file_edit.setPlaceholderText(
            "boxes.csv or labels.json - or the export folder itself")
        self.images_edit = QLineEdit(
            str(images_dir) if images_dir
            else (str(suggested_images_dir(source)) if source else ""))
        self.images_edit.setPlaceholderText(
            "the folder holding the frames the boxes were drawn on")

        form = QFormLayout(self)
        hint = QLabel(
            "Pick the boxes file of an existing export (or the export folder), and "
            "the folder its images are in - usually the <b>images/</b> folder beside "
            "it. Every box becomes one queued animal, shown as a magnified crop; the "
            "keypoints are stored on the <b>original</b> frame, never on the crop.")
        hint.setWordWrap(True)
        form.addRow(hint)
        form.addRow("Boxes file", self._row(self.file_edit, self._browse_file))
        form.addRow("Image folder", self._row(self.images_edit, self._browse_images))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _row(self, edit, callback):
        """A field and its *Browse...* button, side by side."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit)
        button = QPushButton("Browse...")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return row

    def _browse_file(self):
        """Choose the boxes file, and follow with its images folder when it is new."""
        start = self.file_edit.text().strip() or str(paths.DATASETS_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open the boxes file of an export", str(Path(start).parent
                                                          if Path(start).is_file()
                                                          else start),
            BOXES_FILTER)
        if not path:
            return
        self.file_edit.setText(path)
        # Only *fill in* the images field: an operator who typed a folder keeps it.
        if not self.images_edit.text().strip() or self._images_was_suggested:
            self.images_edit.setText(str(suggested_images_dir(path)))

    def _browse_images(self):
        """Choose the folder the frames are in."""
        start = (self.images_edit.text().strip()
                 or self.file_edit.text().strip() or str(paths.DATASETS_DIR))
        folder = QFileDialog.getExistingDirectory(
            self, "Choose the folder holding the frames", str(start))
        if folder:
            self.images_edit.setText(folder)

    @property
    def _images_was_suggested(self) -> bool:
        """True when the images field still holds what was suggested for the file."""
        current = self.images_edit.text().strip()
        source = self.file_edit.text().strip()
        if not current:
            return True
        return bool(source) and current == str(suggested_images_dir(source))

    def boxes_file(self) -> str:
        """The boxes file (or export folder) the operator chose."""
        return self.file_edit.text().strip()

    def images_dir(self) -> str:
        """The folder holding the frames, or an empty string when none was given."""
        return self.images_edit.text().strip()


def crop_box(crop, box) -> tuple:
    """A source-pixel box as **crop** pixels, for the overlay that shows the animal.

    The box is drawn so the operator sees the animal's extent and how much context
    the crop kept - and it goes through the same :class:`~daphnia.vision.crop.Crop`
    the clicks use, so the outline and the points can never disagree.
    """
    x1, y1 = crop.to_crop((box[0], box[1]))
    x2, y2 = crop.to_crop((box[2], box[3]))
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def source_box(crop, box) -> tuple:
    """A **dragged** crop-pixel box as source pixels - :func:`crop_box`'s way back.

    Both corners go through the one :class:`~daphnia.vision.crop.Crop` that owns the
    mapping, exactly as a click does (`mlContext.md` law 1): a drag is reported in the
    frame of reference the canvas is *showing*, and storing those numbers wrote a box
    three times too small at the crop's own offset. The corners are re-sorted, because a
    drag may cross itself.
    """
    x1, y1 = crop.invert((box[0], box[1]))
    x2, y2 = crop.invert((box[2], box[3]))
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


class DatasetDialog(QDialog):
    """Ask for the two things that define a corpus: its name and its model.

    Only the *model* decides what will be labelled later, which is why it is asked
    once, here, instead of being a per-label choice: a corpus whose labels mix two
    keypoint systems cannot train either model. Pass an existing ``dataset`` row to
    rename it instead - then the **model is locked**, because the schema is what the
    labels that already exist *mean*, and the honest way to another model is another
    dataset.
    """

    def __init__(self, parent=None, dataset=None, default_organism="Daphnia magna"):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("Rename dataset" if dataset is not None else "New dataset")
        self.organism_edit = QLineEdit(
            dataset["organism"] if dataset is not None else default_organism)
        self.name_edit = QLineEdit()
        self.model_combo = QComboBox()
        for key, label in annotation.options():
            self.model_combo.addItem(label, key)
        if dataset is not None:
            self.model_combo.setCurrentIndex(
                max(self.model_combo.findData(dataset["schema_key"]), 0))
            self.model_combo.setEnabled(False)
            self.model_combo.setToolTip(
                "Fixed when the dataset was created: changing the model would "
                "invalidate every label already in this corpus.")
            self.name_edit.setText(dataset["name"])
        else:
            self._suggest()
        self.note_edit = QPlainTextEdit(
            (dataset["note"] or "") if dataset is not None else "")
        self.note_edit.setPlaceholderText(
            "Optional: what this corpus is for, and what it must not be used for.")

        form = QFormLayout(self)
        form.addRow("Organism", self.organism_edit)
        form.addRow("Name", self.name_edit)
        form.addRow("Model", self.model_combo)
        form.addRow("Note", self.note_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _suggest(self):
        """Propose a name from the organism and the model, so Ok is one click away."""
        self.name_edit.setText(self.suggested_name())

    def suggested_name(self) -> str:
        """``<organism> - <model>``, the name a dataset gets unless it is changed."""
        organism = self.organism_edit.text().split()[0] if self.organism_edit.text() else ""
        label = annotation.get(self.schema_key()).label
        return f"{organism} - {label}" if organism else label

    def dataset_name(self) -> str:
        """The name the operator typed, with a sane fallback either way.

        A cleared field in *create* mode means "use the suggestion"; in *rename* mode
        it means "keep the name it already has" - never something the operator did not
        ask for.
        """
        text = self.name_edit.text().strip()
        if text:
            return text
        if self.dataset is not None:
            return self.dataset["name"]
        return self.suggested_name()

    def schema_key(self) -> str:
        """The model this corpus is for."""
        return self.model_combo.currentData() or annotation.DEFAULT_SCHEMA

    def organism(self) -> str:
        """The organism the corpus holds."""
        return self.organism_edit.text().strip() or "Daphnia magna"

    def note(self):
        """The optional note."""
        return self.note_edit.toPlainText().strip() or None


class Dataset(ShellPage):
    """A deterministic queue of frames, labelled keypoint by keypoint."""

    #: This screen's controls panel needs more room than the shared one, and the number
    #: is measured, not picked: the dataset row holds a combo reading
    #: ``<name>  [<what it labels>]`` beside two buttons, and a QComboBox sizes itself
    #: to its longest item - so the content's minimum width is ~458 px. Add the panel
    #: padding (2 x 10) and the vertical scroll bar (10) and 500 px is what fits it.
    #: The shared 420 px panel would clip 38 px of it, with no scroll bar to reach it.
    PANEL_WIDTH = 500

    def __init__(self, parent=None):
        super().__init__(ui_file='dataset.ui', controls_width=self.PANEL_WIDTH,
                         parent=parent)

        self.service = shared_service()
        self.stream = self.add_visual(StreamView(self.service))
        self.canvas = self.add_visual(
            ImageCanvas(max_points=2, point_names=list(annotation.default().keypoints)),
            stretch=3)
        self.canvas.points_changed.connect(self._on_points_changed)
        self.canvas.points_moved.connect(self._refresh_readout)
        self.canvas.box_changed.connect(self._on_box_changed)
        self.canvas.box_anchor_changed.connect(self._on_box_anchor_changed)
        self.canvas.double_clicked.connect(self.next_frame)
        # Editing a box that is already stored (`daphnia/ui/widgets/image_canvas.py`): the
        # canvas finds and moves it, the *page* stores what the drop produced, and the page
        # decides whether a `Delete` really removes a label.
        self.canvas.box_selected.connect(self._on_box_selected)
        self.canvas.box_edited.connect(self._on_box_edited)
        self.canvas.box_delete_requested.connect(self._on_box_delete)
        self.canvas.prediction_selected.connect(self._on_prediction_selected)
        self.canvas.prediction_removed.connect(self._on_prediction_removed)
        # A **dragged** model box becomes a label; the same rectangle must then leave the
        # prediction layer, or the dotted box left behind is the "ghost" an operator reports
        # (`SR-DATASET-027`). Only the **box** half leaves - the keypoints stay selectable.
        self.canvas.prediction_accepted.connect(self._on_prediction_accepted)
        # A dropped point **corrects** the label it came from: if this (specimen, trait) is
        # already stored, the new position is written without being asked twice
        # (`SR-DATASET-027`) - and without moving to the next specimen, because the other
        # point may be next.
        self.canvas.points_edited.connect(self._on_points_edited)
        # A **stored keypoint pair** stays editable after it is set: the canvas reports which
        # measurement a press landed on, the page loads that label back into the draft, and the
        # same press drags the point (`X-N16`).
        self.canvas.annotation_edit = True
        self.canvas.annotation_point_selected.connect(self._on_annotation_point_selected)
        # Labelling at volume happens on the image, so the controls that label do too:
        # they are where the eyes are, they are what full screen keeps, and the two
        # gestures that walk the queue (double click, A/D) stay available. The panel
        # keeps the corpus choices and the counts.
        self.workspace = self.canvas_workspace(self.canvas)
        self.workspace.set_group("Label")
        self.workspace.add_tool(
            ToolSpec("save", "Save", "Store this landmark pair in the corpus."),
            lambda: self._tool_action(self.save_label))
        self.workspace.add_tool(
            ToolSpec("undo", "Undo", "Take back the last point, box or stored label."),
            lambda: self._tool_action(self.undo_point))
        self.workspace.add_tool(
            ToolSpec("clear", "Clear points", "Forget the points placed on this "
                                              "frame - stored labels stay."),
            lambda: self._tool_action(self.clear_points))
        self.workspace.add_tool(
            ToolSpec("clear_frame", "Clear frame", "Delete every stored label of this "
                                                   "frame and forget the open points."),
            lambda: self._tool_action(self.clear_frame))
        # A checkable tool behaves exactly like the panel's check box did: its
        # ``toggled`` is what turns box drawing on and off.
        box_tool = self.workspace.add_tool(
            ToolSpec("box", "Box", "Label a bounding box around the specimen instead "
                                   "of keypoints (also the B key).", checkable=True))
        box_tool.toggled.connect(self._on_box_toggled)
        # The two answers of the review loop. They exist **only** while a box queue
        # is open, because "yes/no to this box" is a question only a box queue asks.
        self.workspace.add_tool(
            ToolSpec("accept", "Yes", "Keep this box as ground truth and go on (Y)."),
            lambda: self._tool_action(self.accept_box))
        self.workspace.add_tool(
            ToolSpec("reject", "No", "This box is not a specimen - remove it (N)."),
            lambda: self._tool_action(self.reject_box))
        # The two views of one queued animal: the whole frame with every box drawn
        # (where you are, and what the neighbours are), or just this animal magnified
        # (where the keypoints are placed). Both store on the original frame.
        crop_tool = self.workspace.add_tool(
            ToolSpec("view", "Crop", "Show only this animal, magnified - or the whole "
                                    "frame with every box (also the C key).",
                     checkable=True))
        crop_tool.toggled.connect(self._on_view_toggled)
        self.workspace.add_tool(
            ToolSpec("prev", "< Prev", "Previous frame in the queue (A)."),
            lambda: self.step(-1))
        self.workspace.add_tool(
            ToolSpec("next", "Next >", "Next frame in the queue (D, or a double "
                                       "click on the image)."),
            lambda: self.step(1))
        self.workspace.add_tool(
            ToolSpec("next_keypoint", "Next keypoint",
                     "Move on to the next keypoint without storing this one."),
            self.next_keypoint)
        self.workspace.add_tool(
            ToolSpec("refresh", "Refresh the counts", "Re-read the corpus and the queue."),
            self.refresh)
        # The model **on this frame**: with the switch on, every frame that arrives is run
        # through the models in force (RTMDet + RTMPose), the answer is drawn, and the
        # largest animal's box or keypoints land in the *editable* draft - so correcting the
        # model is the same gesture as labelling by hand, and nothing is stored until Save.
        # Off by default: the model is a tool the operator reaches for, not an uninvited
        # hand in the draft.
        model_tool = self.workspace.add_tool(
            ToolSpec("model", "Model",
                     "Predict the box (or the keypoints) of the animals on this frame with "
                     "the models in force, and put the largest one in the draft for you to "
                     "correct. Off: label by hand only.",
                     checkable=True, checked=False))
        model_tool.toggled.connect(self._on_model_toggled)
        # The **Select** tool (key `"edit"`), on by default (`SR-DATASET-027`): a click
        # **inside a box or on a point selects it** - it never creates a new label - so a
        # prediction can be corrected by dragging it and a wrong box removed with
        # `Backspace`/`Delete`. Turning it *off* is the creation mode: clicks place points
        # and drags draw boxes (`X-N14` names the gestures as tools; the key stays `"edit"`
        # so the pages and tests that already use it are unchanged).
        edit_tool = self.workspace.add_tool(
            ToolSpec("edit", "Select",
                     "The Select tool: a click on a box or a point selects it, a drag moves "
                     "it, an edge or a corner resizes it, Backspace/Delete removes it, and a "
                     "click never creates a new label. Off: the *Keypoints* tool creates.",
                     checkable=True, checked=True))
        edit_tool.toggled.connect(self._on_edit_toggled)
        self.canvas.box_edit = True

        # *File* and *Source*: the frames of a corpus come from files (a folder, an
        # image, a boxes export) or from the camera; the corpus itself is exported. Seven
        # panel buttons became menu items here (`X-N13`) - the panel keeps the choices.
        self.workspace.set_group(FILE_MENU)
        self.workspace.add_tool(
            ToolSpec("open_image", "Open an image...",
                     "Label one frame file, on its own."),
            self._open_image)
        self.workspace.add_tool(
            ToolSpec("import_folder", "Import a folder...",
                     "Queue every image in a folder and label them one by one."),
            self.import_folder)
        self.workspace.add_tool(
            ToolSpec("open_boxes", "Open a boxes export...",
                     "Work through the boxes of an existing export, one animal at a "
                     "time, in a magnified crop."),
            self.choose_boxes_export)
        self.workspace.add_tool(
            ToolSpec("export", "Export the dataset",
                     "Write this dataset - and only this dataset - as images, labels, "
                     "boxes, splits and a manifest."),
            self.export)
        self.workspace.set_group(SOURCE_MENU)
        self.workspace.add_tool(
            ToolSpec("use_frame", "Use the live frame",
                     "Label what the camera sees now; the frame is written only on save."),
            self.use_live_frame)

        self._queue = []        # the frames to label, in a fixed order
        self._index = 0
        self._frame = None      # the pixels of the current frame
        self._path = None       # the file behind them, when there is one
        self._labels = {}       # file_path -> {(specimen, trait): row}
        self._boxes = {}        # file_path -> {specimen: box row}
        self._annotation_tags = {}   # (specimen, segment) -> trait, for re-editing a label
        self._draft = labeling.Draft()   # partial points of a landmark pass
        self._loading = False   # guards programmatic canvas updates
        self._model_predictions = ()    # the last answer of the models in force
        # The models in force, run on the frame on screen (`daphnia/ui/model_pass.py`): the
        # page never chooses a checkpoint - the registry does, written by the *Detector
        # (RTMDet)* / *Pose (RTMPose)* menus.
        self.runner = ModelRunner(self)
        self._specimen = 1      # the specimen selected in the dropdown
        self._trait = TRAITS[0]   # the trait (two-point mode)
        self._landmark = None   # the landmark (landmark mode)
        self._schema = annotation.default()   # the chosen model's annotation schema
        self._dataset = None    # the dataset row being labelled
        self._box_mode = False  # dragging a bounding box instead of clicking points
        self._items = []        # the box queue: one *animal* per item, not one frame
        self._item_index = 0
        self._crop = None       # the mapping in force on the canvas (None = the frame)
        self._item_crop = None  # the crop of the current animal, always available
        self._crop_view = True  # frame view or crop view of the current animal
        self._item_frame = None  # the whole frame the current crop was taken from
        self._accepted = 0      # boxes a person said yes to in this session
        self._rejected = 0      # boxes a person said no to (a prediction that was wrong)
        self._last_boxes_source = None   # remembered, so the next open is one click
        self._last_images_dir = None

        # The corpus, its frames and its export are **menu items** now (`X-N13`): *File*
        # for the frames and the export, *Source* for the live frame, *Label* for the
        # actions of a labelling pass. What stays here is what a *choice* is: the dataset,
        # the mode, the specimen.
        self.combo_specimen.currentIndexChanged.connect(self._on_specimen_changed)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.combo_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        self.btn_new_dataset.clicked.connect(self.new_dataset)
        self.btn_rename_dataset.clicked.connect(self.rename_dataset)

        # Keyboard is how volume gets done: D/Right and A/Left walk the queue.
        for keys, step in (("D", 1), ("Right", 1), ("A", -1), ("Left", -1)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(lambda step=step: self.step(step))
        save_shortcut = QShortcut(QKeySequence("S"), self)
        save_shortcut.activated.connect(self.save_label)
        box_shortcut = QShortcut(QKeySequence("B"), self)
        box_shortcut.activated.connect(self.toggle_box_mode)
        # `Backspace`/`Delete` remove whatever the *Edit* mode has selected, wherever the focus
        # happens to be: the key must not depend on the canvas holding it (`SR-DATASET-027`).
        for keys in (QKeySequence(Qt.Key_Backspace), QKeySequence.StandardKey.Delete):
            delete_shortcut = QShortcut(keys, self)
            delete_shortcut.activated.connect(self.canvas.delete_selection)
        # The review loop is answered from the keyboard too: Y keeps a box, N drops it.
        accept_shortcut = QShortcut(QKeySequence("Y"), self)
        accept_shortcut.activated.connect(self.accept_box)
        reject_shortcut = QShortcut(QKeySequence("N"), self)
        reject_shortcut.activated.connect(self.reject_box)
        crop_shortcut = QShortcut(QKeySequence("C"), self)
        crop_shortcut.activated.connect(self.toggle_view)

        for signal in (self.service.status, self.service.opened,
                       self.service.closed):
            signal.connect(self._refresh_image)

        # A frame the canvas cannot read (a moved corpus, a bad file) is refused in the
        # panel too - named, with the reason, instead of an empty canvas.
        self.canvas.source_failed.connect(self.status_label.setText)

        self._build_selectors()
        self.refresh()

    def on_shown(self):
        self.refresh()

    # -- selectors / subject ----------------------------------------------
    def _build_selectors(self):
        self.combo_mode.clear()
        for mode in (labeling.PAIR_MODE, labeling.PASS_MODE):
            self.combo_mode.addItem(labeling.MODE_LABELS[mode], mode)
        self._ensure_dataset()
        self._refresh_dataset_choices()
        self._apply_schema()
        self._refresh_specimen_choices()
        self._on_mode_changed()      # show the controls the current mode uses

    # -- the dataset (one corpus per model) --------------------------------
    def _ensure_dataset(self):
        """Load the open dataset, creating the default corpus on a fresh database."""
        with connection() as conn:
            dataset_id = repository.ensure_default_dataset(conn)
            self._dataset = queries.dataset_by_id(conn, dataset_id)

    def dataset(self):
        """The dataset row being labelled (never ``None`` once the page is built)."""
        return self._dataset

    def dataset_id(self):
        """The id of the dataset a label belongs to, or ``None``."""
        return self._dataset["dataset_id"] if self._dataset else None

    def schema(self) -> annotation.AnnotationSchema:
        """The chosen model's annotation schema - what a label here *is*."""
        return self._schema

    def _refresh_dataset_choices(self, select=None):
        """List every dataset, the open one selected, each labelled with its model.

        The model is the whole difference between two datasets, so it belongs in the
        list - otherwise "Daphnia (2)" says nothing about what labelling it will ask
        for.
        """
        with connection() as conn:
            datasets = queries.list_datasets(conn)
        combo = self.combo_dataset
        combo.blockSignals(True)
        combo.clear()
        for row in datasets:
            schema = annotation.get(row["schema_key"])
            combo.addItem(f"{row['name']}  [{schema.label}]", row["dataset_id"])
            combo.setItemData(combo.count() - 1, schema.describe(), Qt.ToolTipRole)
        index = combo.findData(select) if select is not None else 0
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

    def select_dataset(self, dataset_id):
        """Open another dataset and re-point the labelling at its model."""
        with connection() as conn:
            repository.activate_dataset(conn, dataset_id)
        self._ensure_dataset()
        self._refresh_dataset_choices(select=dataset_id)
        self._apply_schema()
        self.refresh()
        return self._dataset

    def create_dataset(self, *, name, schema_key, organism="Daphnia magna",
                       note=None):
        """Create a corpus for one model and open it; returns its id.

        Modelling another target means another dataset - labels are never carried
        over, so a corpus always describes exactly one model's input/output. A name
        that is already taken is numbered instead of refused, and the status line
        reports the name that was actually used.
        """
        with connection() as conn:
            dataset_id = repository.create_dataset(
                conn, name=name, schema_key=schema_key, organism=organism, note=note)
        self._ensure_dataset()
        self._refresh_dataset_choices(select=dataset_id)
        self._apply_schema()
        self.refresh()
        return dataset_id

    def rename_dataset(self):
        """Rename the open dataset (its model and its labels stay as they are)."""
        dialog = DatasetDialog(self, dataset=self._dataset)
        if dialog.exec() != QDialog.Accepted:
            return None
        return self.set_dataset_name(dialog.dataset_name(), organism=dialog.organism(),
                                     note=dialog.note())

    def set_dataset_name(self, name, organism=None, note=None) -> str:
        """Apply a new name/organism/note to the open dataset; returns the final name."""
        with connection() as conn:
            final = repository.rename_dataset(conn, self.dataset_id(), name)
            if organism is not None or note is not None:
                conn.execute(
                    "UPDATE datasets SET organism = COALESCE(?, organism),"
                    " note = ? WHERE dataset_id = ?",
                    (organism, note, self.dataset_id()))
        self._ensure_dataset()
        self._refresh_dataset_choices(select=self.dataset_id())
        self._refresh_dataset_label()
        self.status_label.setText(f"Dataset renamed to '{final}'.")
        return final

    def new_dataset(self):
        """Ask for a new dataset (name + model) and open it."""
        dialog = DatasetDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return None
        dataset_id = self.create_dataset(
            name=dialog.dataset_name(), schema_key=dialog.schema_key(),
            organism=dialog.organism(), note=dialog.note())
        self.status_label.setText(
            f"Dataset '{self._dataset['name']}' created - {self._schema.describe()}.")
        return dataset_id

    def _apply_schema(self):
        """Point the labelling at the model's keypoint system (or at its boxes)."""
        key = self._dataset["schema_key"] if self._dataset else None
        self._schema = annotation.get(key)
        keypoints = self._schema.keypoints
        self._draft = labeling.Draft(intervals=labeling.intervals_for(self._schema),
                                     landmarks=keypoints)
        self._trait = self._traits()[0] if self._traits() else None
        self._landmark = keypoints[0] if keypoints else None
        self._loading = True
        self._box_mode = self._schema.is_detection_only
        self.canvas.box_mode = self._box_mode
        self.canvas.clear_box()
        # The Box tool follows the model the same way the panel's check box did.
        self.workspace.set_checked("box", self._box_mode)
        self._loading = False
        self._refresh_dataset_label()
        self._refresh_controls()
        self._on_subject_changed()      # the prompt, readout and overlays follow
        # A box queue belongs to the corpus it was opened for, and opening it is an
        # explicit act (*Open boxes export...*). Switching dataset therefore **leaves**
        # item mode rather than silently entering it: a dataset that merely has boxes
        # keeps meaning "label this frame", and another corpus's boxes are never left
        # on screen (`SR-DATASET-022`).
        if self._items:
            self._clear_item_state()

    def _refresh_dataset_label(self):
        """Show which corpus is open and what its model expects."""
        name = self._dataset["name"] if self._dataset else "none"
        self.dataset_label.setText(
            f"Dataset: {name}\nModel: {self._schema.describe()}")

    def _on_dataset_changed(self, *_):
        """Another dataset was picked: it becomes the one being labelled."""
        dataset_id = self.combo_dataset.currentData()
        if dataset_id and dataset_id != self.dataset_id():
            self.select_dataset(dataset_id)
            self.status_label.setText(
                f"Now labelling '{self._dataset['name']}' - "
                f"{self._schema.describe()}.")

    def _traits(self) -> tuple:
        """The traits this dataset's model measures (empty for boxes only)."""
        return tuple(self._schema.traits) if self._schema else ()

    def trait(self) -> str:
        """The trait being labelled (two-point mode)."""
        return self._trait

    def set_trait(self, name):
        """Choose the trait (also what the *Next keypoint* button cycles)."""
        traits = self._traits()
        if name in traits:
            self._trait = name
        self._on_subject_changed()

    def mode(self) -> str:
        """``'pair'`` (two points per specimen) or ``'pass'`` (landmark at a time)."""
        return self.combo_mode.currentData() or labeling.PAIR_MODE

    def landmark(self) -> str:
        """The landmark being placed (landmark mode)."""
        return self._landmark

    def set_landmark(self, name):
        """Choose the landmark pass; the auto-advance calls this too."""
        if name not in self._schema.keypoints:
            return
        self._landmark = name
        self._on_landmark_changed()

    def next_keypoint(self):
        """Step to the next keypoint/trait by hand.

        Auto-advance moves the pass on once a landmark covers every specimen, but the
        *first* switch (e.g. all heads -> the tails) is a judgement call, so it needs a
        button rather than an automatic rule. A dataset's model may have fewer points:
        two-keypoint corpora simply wrap at the tail.
        """
        if not self._schema.uses_keypoints:
            self.status_label.setText(
                "This dataset labels bounding boxes - drag one around a specimen.")
            return
        if self.mode() == labeling.PASS_MODE:
            following = (labeling.next_landmark(self._landmark,
                                                order=self._schema.keypoints)
                         or self._schema.keypoints[0])
            self.set_landmark(following)
            self.status_label.setText(f"Now placing '{self._landmark}'.")
        else:
            self.set_trait(labeling.next_trait(self._trait, traits=self._traits()))
            self.status_label.setText(
                f"Trait: {annotation.trait_label(self._trait)}.  {self._prompt()}")

    def _refresh_keypoint_label(self):
        """Show the current keypoint/trait - it is a display, not a control."""
        if not self._schema.uses_keypoints:
            self.keypoint_label.setText("Keypoint: none (boxes only)")
        elif self.mode() == labeling.PASS_MODE:
            self.keypoint_label.setText(f"Keypoint: {self._landmark}")
        else:
            first, second = self._points()
            self.keypoint_label.setText(
                f"Trait: {annotation.trait_label(self._trait)}  ({first} -> {second})")

    def specimen(self) -> int:
        """Which animal on the frame is being labelled (from the dropdown)."""
        return int(self.combo_specimen.currentData() or 1)

    def _stored_specimen_count(self) -> int:
        """How many specimens the frame is known to hold, from the store and the draft."""
        indices = set(self._draft.specimens())
        indices.update(specimen for (specimen, _trait) in self._stored())
        indices.update(self._stored_boxes())
        return max(indices) if indices else 0

    def _refresh_specimen_choices(self, select=None):
        """Rebuild the specimen dropdown as ``1..N``, where N is what the frame holds.

        The list grows as specimens are added, which is why the specimen is a dropdown
        rather than a button that has to be pressed to make a new one.
        """
        target = max(1, int(select if select is not None else self._specimen))
        total = max(self._stored_specimen_count(), target, 1)
        combo = self.combo_specimen
        combo.blockSignals(True)
        combo.clear()
        for index in range(1, total + 1):
            combo.addItem(str(index), index)
        combo.setCurrentIndex(target - 1)
        combo.blockSignals(False)
        self._specimen = target

    def _set_specimen(self, value):
        """Select a specimen, adding it to the dropdown if it is new.

        The canvas follows: in box mode the selected specimen's stored box becomes the
        editable one, in pair mode the clicked points are reset - so a programmatic
        selection behaves exactly like picking the number by hand.
        """
        self._refresh_specimen_choices(select=value)
        self._on_subject_changed()

    def _on_specimen_changed(self, *_):
        """A specimen was picked in the dropdown."""
        self._specimen = int(self.combo_specimen.currentData() or 1)
        self._on_subject_changed()

    def _points(self) -> tuple:
        """The two landmark names of the current trait, in click order."""
        if not self._schema.uses_keypoints:
            return ()
        return self._schema.keypoint_system.points_for(self._trait)

    def _prompt(self) -> str:
        """Tell the operator exactly what the current action expects."""
        if self._items:
            here = ("this animal is magnified" if self._crop_view
                    else "the whole frame with every box is shown")
            switch = ("C shows the whole frame with every box"
                      if self._crop_view
                      else "C magnifies just this animal")
            if not self._schema.uses_keypoints:
                return (f"Review this box: Yes keeps it as ground truth, No removes "
                        f"it.  {here.capitalize()}; {switch}.  Walk the queue with "
                        "Prev/Next (A/D).")
            first, second = self._points()
            return (f"Click the {first}, then the {second} "
                    f"({DATASET_TRAIT_HINTS.get(self.trait(), '')}) - the pair is "
                    f"stored as soon as both exist, and the next animal follows.  "
                    f"{here.capitalize()}; {switch}; Y/N answers the box itself.")
        if self._box_mode:
            if self.canvas.box_pending:
                return (f"Corner set - click the opposite corner (or drag from it) "
                        f"to finish the box for specimen {self.specimen()}.")
            return (f"Drag a rectangle around specimen {self.specimen()} - or click "
                    f"one corner and then the opposite one.")
        if not self._schema.uses_keypoints:
            return ("Drag a rectangle around each specimen; the box is stored as "
                    "soon as you let go.")
        if self.mode() == labeling.PASS_MODE:
            return (f"Click every specimen's {self.landmark()}; the interval saves "
                    f"itself once both ends exist.")
        first, second = self._points()
        return (f"Click the {first}, then the {second} "
                f"({DATASET_TRAIT_HINTS.get(self.trait(), '')}).")

    # -- state -------------------------------------------------------------
    def refresh(self):
        """Re-read the corpus counts, the queue's labelled set and the image."""
        self._reload_labels()
        self._refresh_stats()
        self._refresh_image()
        self._refresh_overlays()

    def _reload_labels(self):
        """Re-read the **open dataset's** labels and boxes for every queued frame.

        Scoped to one dataset on purpose: switching datasets must swap the ground
        truth the canvas shows, never merge two models' labels into one picture.
        """
        with connection() as conn:
            rows = queries.dataset_labels(conn, self.dataset_id())
            boxes = queries.dataset_boxes(conn, self.dataset_id())
        labels = {}
        for row in rows:
            labels.setdefault(row["file_path"], {})[
                (row["specimen_index"], row["trait"])] = row
        boxed = {}
        for row in boxes:
            boxed.setdefault(row["file_path"], {})[row["specimen_index"]] = row
        self._labels = labels
        self._boxes = boxed

    def _item_queue_summary(self) -> str:
        """How far through the box queue the operator is, and what it holds.

        Counted per animal, not per frame: the queue's unit is one box, and a
        frame's three animals are three items. The trait count is the one that says
        how much of the *corpus* is done.
        """
        trait = self.trait()
        done = 0
        if trait is not None:
            done = sum(1 for item in self._items
                       if (item["specimen"], trait)
                       in self._labels.get(str(item["path"]), {}))
            what = annotation.trait_label(trait)
        else:
            done = sum(1 for item in self._items
                       if item["source"] == "manual")
            what = "boxes"
        models = sum(1 for item in self._items if item["source"] == "model")
        return (f"{len(self._items)} animal(s) queued, {done} with {what}"
                + (f", {models} still a prediction" if models else "")
                + f".  Accepted {self._accepted}, rejected {self._rejected}.")

    def _refresh_stats(self):
        with connection() as conn:
            stats = queries.dataset_stats(conn, self.dataset_id())
        if self._items:
            self.queue_label.setText(self._item_queue_summary())
        else:
            labelled = sum(1 for path in self._queue if str(path) in self._labels)
            self.queue_label.setText(
                f"{len(self._queue)} frame(s) in the queue, {labelled} labelled."
                if self._queue else
                "No frames yet - import a folder, open an image, or use the live "
                "frame.")
        counts = ", ".join(f"{annotation.trait_label(trait)} {count}"
                           for trait, count in sorted(stats["by_trait"].items()))
        boxes = (f", {stats['boxes']} box(es)" if self._schema.exports_boxes else "")
        self.stats_label.setText(
            f"Dataset '{self._dataset['name'] if self._dataset else '-'}': "
            f"{stats['labels']} label(s){boxes} on {stats['images']} image(s)"
            + (f" - {counts}." if counts else "."))

    @staticmethod
    def _active_profile():
        """The calibration in force, or ``None`` when nothing is calibrated."""
        with connection() as conn:
            return repository.active_profile(conn)

    # -- the image ---------------------------------------------------------
    def _refresh_image(self, *_):
        self.stream_hint_label.setText(
            f"Live feed: {self.service.state_label()}.")
        profile = self._active_profile()
        if profile is None:
            self.scale_label.setText(
                "Scale: none - labels are stored in pixels.  A calibration is "
                "optional: it only adds the length in micrometres.")
        else:
            self.scale_label.setText(
                f"Scale: {profile.px_per_um * 1000.0:.1f} px/mm "
                f"+/- {profile.uncertainty_pct:.2f}%")
        self._refresh_readout()

    def _set_image(self, frame, path):
        """Adopt an image as the frame to label.

        The frame is already on the canvas (via ``set_source``, whatever the source
        was), so this only records it and re-reads what hangs off it.
        """
        self._frame = frame
        self._path = Path(path) if path is not None else None
        self._draft.clear()          # partial points belong to the frame they were placed on
        self.source_label.setText(
            f"Frame: {self._path.name}" if self._path is not None
            else "Live frame (written to disk when you save a label).")
        self._set_specimen(1)
        self._on_subject_changed()
        self.prelabel()              # the models in force answer on the frame just shown

    # -- the model on this frame -------------------------------------------
    def prelabel(self, announce=False):
        """Run the models in force on the frame on screen, if the switch is on.

        * A model that is **selected** is a model that answers: with the switch on (default),
          every frame that arrives is run through the top-down pipeline and the answer is
          drawn on it (`daphnia/ui/model_pass.py`).
        * The **largest** animal's answer becomes the *editable* draft - its box in box mode
          (drag it onto the animal you mean; releasing the drag stores it, exactly as
          drawing one by hand does) or its two keypoints in two-point mode (move them, then
          *Save*). Correcting a prediction is therefore the same gesture as labelling.
        * In **pass** mode, in an **item** queue and with the switch off, nothing is placed:
          the answer stays in the prediction layer, where it can be seen and ignored. A
          prediction is never stored by itself (`SR-PREDICT-N01`).
        * With nothing to report the page's **own** status line stands (`announce=True` is
          the switch being turned on by hand - then the reason is said out loud).
        """
        if not self.workspace.action("model").isChecked():
            return
        result = self.runner.run(self._frame, self._um_per_px())
        self.canvas.set_predictions(result.predictions)
        self._model_predictions = result.predictions
        best = primary(result.predictions)
        hint = ""
        self._loading = True
        if best is not None and not self._items:
            if self._box_mode and best.box is not None:
                self.canvas.set_box(best.box)       # drag it; letting go stores it
                hint = "Drag the predicted box onto the animal you mean and let go."
            elif best.points and self.mode() != labeling.PASS_MODE:
                self.canvas.set_points(list(best.points))
                hint = "Fix the predicted points, then Save."
        self._loading = False
        if not len(result) and not announce:
            return
        note = result.note + (f"  {hint}" if hint else "")
        current = self.status_label.text().strip()
        self.status_label.setText(f"{current}  {note}" if current else note)

    def _on_model_toggled(self, checked):
        """The *Model* switch: predict on this frame, or label by hand only."""
        if checked:
            self.prelabel(announce=True)
            return
        # Off means the page is *untouched* by the model: the answer and the draft it filled
        # go, so nothing half-predicted can be stored by a later click.
        self._loading = True
        self.canvas.set_predictions([])
        self.canvas.set_box(None)
        self.canvas.clear_points()
        self._loading = False
        self.status_label.setText("Model off - labelling by hand.")

    def _um_per_px(self):
        """The scale in force in micrometres per pixel, or ``None`` (labels stay px)."""
        profile = self._active_profile()
        if profile is None or not profile.px_per_um:
            return None
        return 1.0 / float(profile.px_per_um)

    # -- selecting and editing a stored box --------------------------------
    def _on_edit_toggled(self, checked):
        """The *Edit* mode: the frame's boxes are edited, or the page goes back to creating.

        On (the default) a click selects what is under it - a box, a model's box, a point - and
        never labels; off, clicks place points and drags draw boxes, which is how a frame with
        no labels yet is filled in (`SR-DATASET-027`).
        """
        self.canvas.box_edit = bool(checked)
        if checked:
            self.status_label.setText(
                "Edit mode: click a box or a point to select it, drag it to move or resize "
                "it, Backspace to remove it. Click empty space to deselect.")
        else:
            self.canvas.clear_selection()
            self.status_label.setText(
                f"Create mode: {self._prompt()}")

    def _on_box_selected(self, index):
        """A stored box was clicked: the panel follows it (`_set_specimen`)."""
        if index < 0:
            return
        self._set_specimen(index)
        x1, y1, x2, y2 = self.canvas.box_of(index) or (0, 0, 0, 0)
        self.status_label.setText(
            f"Specimen {index} selected ({x2 - x1:.0f} x {y2 - y1:.0f} px) - drag it, resize "
            f"an edge, or press Delete to remove it.")

    def _on_box_edited(self, index, box):
        """A box was moved or resized: store the corrected geometry for that specimen.

        This is the *same* write a drawn box performs (`save_corpus_box`, one row per
        specimen), because correcting a label is labelling - and the numbers are said out
        loud: a box that quietly changed size is a label nobody can audit.

        **The geometry is converted back to source pixels first.** The canvas reports what
        the operator dragged *in whatever frame it is showing*, and in the crop loop that
        is the magnified crop - so storing the numbers as they arrive wrote a box three
        times too small at the crop's own offset, with the crop's size as the image size
        (measured: the box of specimen 1 went from ``(100, 100, 200, 160)`` on a
        1280 x 720 frame to ``(82, 79, 382, 259)`` on a "450 x 330" one). One mapping
        object owns the way back (`mlContext.md` law 1), exactly as it does for a click.
        """
        if self._frame is None and self._path is None:
            self.status_label.setText("Nothing to edit yet - open an image first.")
            return
        path = self._ensure_file()
        if path is None:
            return
        source = self._source_box(box)
        width, height = self._source_size()
        with connection() as conn:
            repository.save_corpus_box(
                conn, file_path=str(path), image_size=(width, height), box=source,
                specimen_index=int(index), dataset_id=self.dataset_id())
        if self._items:
            # The queue holds the box it was built from, and the crop draws *that*: a
            # correction the queue did not hear would leave the view showing the old
            # rectangle while the corpus holds the new one (two boxes, one animal).
            for item in self._items:
                if item["path"] == Path(path) and int(item["specimen"]) == int(index):
                    item["box"] = source
        self._reload_labels()
        self._refresh_overlays()
        x1, y1, x2, y2 = source
        self.status_label.setText(
            f"Specimen {index}: box corrected to {x1:.0f}, {y1:.0f} - "
            f"{x2 - x1:.0f} x {y2 - y1:.0f} px on {Path(path).name}.")

    def _on_box_delete(self, index):
        """`Delete` on a selected box: remove that label from the corpus (the page decides)."""
        if self._path is None:
            self.status_label.setText(
                "This frame is not stored yet, so there is no box to remove.")
            return
        with connection() as conn:
            removed = repository.delete_corpus_box(
                conn, file_path=str(self._path), specimen_index=int(index),
                dataset_id=self.dataset_id())
        self._reload_labels()
        self._refresh_overlays()
        self.canvas.clear_selection()
        self.status_label.setText(
            f"Specimen {index}: box removed." if removed
            else f"Specimen {index} has no stored box - nothing to remove.")

    def _on_prediction_selected(self, index):
        """A **predicted** box was clicked: that animal becomes the one being labelled.

        A prediction is a suggestion, so picking one is not editing a label - it answers
        *which animal?* - and the answer goes into the editable draft exactly like the largest
        one does when the model runs.
        """
        if index < 0 or index >= len(self._model_predictions):
            return
        item = self._model_predictions[index]
        self._loading = True
        if self._box_mode:
            self.canvas.set_box(item.box)
        elif item.points:
            self.canvas.set_points(list(item.points))
        self._loading = False
        self.status_label.setText(
            f"Predicted animal {index + 1} of {len(self._model_predictions)} "
            f"(score {item.score:.2f}) is in the draft - correct it and Save.")

    def _on_prediction_accepted(self, index, box, points, moved):
        """A model's box was **dragged** into a label: store it, and take it out of the layer.

        Only a drag stores a label (`SR-DATASET-027`), so only a drag stores here - through the
        very same store-and-move-on a drawn box gets. The keypoints half of the answer, if the
        prediction carried any, stays selectable (`X-N17`).
        """
        if moved and box is not None:
            self._on_box_changed([box])
            self.canvas.drop_prediction(index, part="box")

    def _on_prediction_removed(self, index):
        """`Backspace` on a model's box: the model is wrong about that one - drop it here.

        Nothing was stored, so nothing is deleted: the frame's prediction layer simply stops
        showing it, and running the model again brings it back if the operator changes their mind.
        """
        if index < 0 or index >= len(self._model_predictions):
            return
        self._model_predictions = tuple(item for position, item in
                                        enumerate(self._model_predictions)
                                        if position != index)
        self.status_label.setText(
            f"Predicted animal {index + 1} removed from this frame - "
            f"{len(self._model_predictions)} left.")

    def _on_points_edited(self):
        """A point was dragged and let go: write the correction if the label already exists.

        Nothing happens for a label that is not stored yet - that one is *Save*'s job, and a
        half-placed pair must never be stored by a drag. A stored (specimen, trait) is a
        correction, and a correction needs no second key press (`SR-DATASET-027`).
        """
        if self._loading or self._box_mode or self._items:
            # The crop loop is *click, click, next animal*: it consumes the pair the moment both
            # ends exist and leaves no point on screen to drag, so it has no correction gesture
            # to answer (`SR-DATASET-027`).
            return
        points = self.canvas.points
        specimen, trait = self.specimen(), self.trait()
        if (specimen, trait) not in self._stored():
            return
        if len(points) < 2:
            return
        path = self._ensure_file()
        if path is None:
            return
        height, width = self._frame.shape[:2]
        profile = self._active_profile()
        with connection() as conn:
            repository.save_corpus_label(
                conn, file_path=str(path), image_size=(width, height),
                trait=trait, p1=points[0], p2=points[1], specimen_index=specimen,
                profile=profile, dataset_id=self.dataset_id())
        self._reload_labels()
        self._refresh_stats()
        self._refresh_overlays()
        self.status_label.setText(
            f"Specimen {specimen} {annotation.trait_label(trait)} corrected on "
            f"{Path(path).name} - stored.")

    def adopt_models(self, detector=None, pose=None) -> bool:
        """Take a model the window loaded from its menus, and run it on the frame.

        Taking it **switches the *Model* tool on** - that is what the menu click asked for -
        and reports the answer on the frame and in the status line.
        """
        if not self.runner.adopt_models(detector=detector, pose=pose):
            return False
        self.workspace.set_checked("model", True)
        self.prelabel(announce=True)
        return True

    # -- the queue ---------------------------------------------------------
    def import_folder(self):
        """Bring in every image under a folder, recursively."""
        folder = QFileDialog.getExistingDirectory(
            self, "Import a folder of frames", str(paths.RECORDINGS_DIR))
        if not folder:
            return
        found = sorted(path for path in Path(folder).rglob("*")
                       if path.suffix.lower() in IMAGE_SUFFIXES)
        if not found:
            self.status_label.setText(f"No image files found under {folder}.")
            return
        self.add_paths(found)
        self.status_label.setText(
            f"Imported {len(found)} frame(s) from {folder}.  {self._prompt()}")

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open an image to label", str(paths.RECORDINGS_DIR),
            "Images (*.png *.jpg *.jpeg *.heic *.heif *.tif *.tiff *.bmp *.webp)"
            ";;All files (*)")
        if path:
            self.add_paths([Path(path)])

    def add_paths(self, paths_in):
        """Replace the queue with these frames, in a fixed order, and show the first.

        A frame queue and a box queue are two different questions, so bringing frames
        in **leaves** the box queue behind: otherwise the position readout would mean
        two things at once and ``step()`` would walk the wrong list.
        """
        if self._items:
            self._clear_item_state()
        unique = []
        for path in paths_in:
            text = str(Path(path))
            if text not in unique:
                unique.append(text)
        self._queue = [Path(text) for text in sorted(unique)]
        self._index = 0
        self.refresh()
        self._show_current()

    def next_frame(self):
        """A double click on the image walks to the next frame - and stores nothing.

        The first click of the double click has *already* done its thing (a point in
        two-point mode, a landmark - possibly completing an interval - in landmark
        mode, a sliver of a box in box mode). Moving on must not carry that artefact
        into the corpus, so it is taken back first, using the same undo a mistaken
        click would get: what is dropped is exactly what the click added, and it is
        reported in the status line.
        """
        if not self._queue and not self._items:
            self.status_label.setText(
                "Nothing to walk: a single live frame is not a queue - import a "
                "folder or open an image to move through frames.")
            return
        before = self._item_index if self._items else self._index
        dropped = self._take_back_click()
        self.step(1)
        note = ("  The point the double click placed was taken back - a double click "
                "never labels." if dropped else "")
        after = self._item_index if self._items else self._index
        if after == before:
            note += ("  This is the last animal in the queue." if self._items
                     else "  This is the last frame in the queue.")
        self.status_label.setText(f"{self.position_label.text()}{note}")

    def _take_back_click(self) -> bool:
        """Undo what the double click's own first press did; True when it did anything."""
        if self._items:
            # Item mode: the press placed a keypoint of this animal. It is not a
            # label (the pair is what stores), but leaving it would put half an
            # interval on the next animal's crop.
            if self.canvas.points:
                self.canvas.clear_points()
                return True
            return False
        if self._box_mode:
            drew = self.canvas.box is not None
            self._loading = True
            self.canvas.clear_box()
            self._loading = False
            return drew
        if self.mode() == labeling.PASS_MODE:
            # A loose point, or an interval the click just completed - both are exactly
            # what Undo removes, and both are the click's own doing.
            if self._draft.last() is not None or self._stored():
                self._undo_pass()
                return True
            return False
        if self.canvas.points:
            self.canvas.clear_points()
            return True
        return False

    def step(self, delta):
        """Walk the queue by ``delta``: animals in item mode, frames otherwise."""
        if self._items:
            self._item_index = max(0, min(self._item_index + delta,
                                          len(self._items) - 1))
            self._show_item()
            return
        if not self._queue:
            self.status_label.setText("No frames in the queue yet.")
            return
        self._index = max(0, min(self._index + delta, len(self._queue) - 1))
        self._show_current()

    def _show_current(self):
        """Load the frame at the current position and show its stored label."""
        if not self._queue:
            return
        path = self._queue[self._index]
        frame = self.canvas.set_source(path)     # reads the file, or refuses it
        if frame is None:
            self._frame = None
            self._path = None
        else:
            self._set_image(frame, path)
        self.position_label.setText(
            f"Frame {self._index + 1} of {len(self._queue)}")
        self._refresh_stats()

    def use_live_frame(self):
        """Label what the camera sees now; the frame is written only on save."""
        frame = self.service.current_frame()
        if frame is None:
            self.status_label.setText(
                f"No live frame.  {self.service.state_label()}  Connect in the bar "
                "below, or import a folder instead.")
            return
        if self._items:
            self._clear_item_state()      # a live frame is a frame queue, not a box
        self._queue = []
        self._index = 0
        self.position_label.setText("Live frame")
        self.canvas.set_source(frame)
        self._set_image(frame, None)
        self._refresh_stats()

    # -- the box queue: one animal per item ---------------------------------
    def choose_boxes_export(self):
        """The button: ask for the boxes file **and** the images folder, then open them.

        Two fields rather than one dialog trick: a `boxes.csv` names its frames
        relative to the export folder, so "where are the pictures" is a real question
        the operator answers - and the answer they can see in a file browser is often
        the `images/` folder itself. The last file opened is remembered, so the second
        time is one click.

        If the open dataset has no box in its model, the operator is **offered the
        dataset that does** instead of being left at a wall - a question is allowed
        here because a person is here (the dialog-free core never asks).
        """
        dialog = BoxesExportDialog(self, source=self._last_boxes_source,
                                   images_dir=self._last_images_dir)
        if dialog.exec() != QDialog.Accepted:
            return None
        source = dialog.boxes_file()
        if not source:
            self.status_label.setText(
                "No boxes file given - point at the export's boxes.csv, or at the "
                "export folder itself.")
            return None
        self._last_boxes_source = source
        self._last_images_dir = dialog.images_dir()
        images_dir = dialog.images_dir() or None
        if not self._schema.exports_boxes and not self._offer_box_dataset():
            return {}
        return self.open_boxes_export(source, images_dir=images_dir)

    def open_boxes_export(self, source, images_dir=None) -> dict:
        """Import the boxes of an existing export, then queue them one animal each.

        The recursive annotator's front door: boxes that already exist - a
        detector's predictions, or boxes drawn in an earlier session - become items
        of *this* dataset, and each item is then answered yes/no and carries its
        keypoints. The queue is built from the **store**, never from the file, so a
        box added by hand is in it too: the corpus is the queue, and the animals the
        detector missed have to be addable (``SR-DATASET-022``).

        ``images_dir`` is where the frames are, when that is not beside the boxes file.

        A keypoint-only model is refused **in words**: it has nowhere to put a box,
        and the fix is another dataset (``X-N10``).
        """
        if not self._schema.exports_boxes:
            # Refused in words, **never** with a dialog: this method is the dialog-free
            # core that a test, a script and the offscreen UI check drive, and a modal
            # window there is a hang. The *button* is where a person is, and that is
            # where the offer to create the right dataset lives.
            self.status_label.setText(
                f"'{self._dataset['name']}' is a keypoint-only dataset "
                f"({self._schema.label}), so it has nowhere to put a box. Create a "
                "dataset whose model has one - 'Pose (4 points) + bounding box' or "
                "'Bounding boxes only' - and open the boxes there.")
            return {}
        try:
            with connection() as conn:
                result = maintenance.import_boxes(conn, self.dataset_id(), source,
                                                  images_dir=images_dir)
        except (ValueError, ExportError) as error:
            self.status_label.setText(f"Could not open those boxes - {error}")
            return {}
        self.build_box_queue()
        self._reload_labels()
        self._refresh_stats()
        self._show_item(0)
        note = ""
        if result["skipped"]:
            note += (f"  {result['skipped']} box(es) leave their image or are smaller "
                     f"than {annotation.MIN_BOX_PIXELS:g} px, so they were not "
                     "imported - the labeller refuses to make one, so a corpus must "
                     "not hold one.")
        if result["missing"]:
            note += (f"  {result['missing']} frame(s) are not on disk; their boxes are "
                     "kept and reported, never silently dropped.")
        where = (f" (frames from '{Path(result['images_dir']).name}/')"
                 if result.get("images_dir") else "")
        self.status_label.setText(
            f"Imported {result['boxes']} box(es) on {result['images']} frame(s) from "
            f"{Path(source).name}{where} - {len(self._items)} animal(s) queued, one at "
            f"a time.{note}")
        return result

    def switch_to_box_dataset(self, name=None) -> str:
        """Create and open the dataset that *can* hold a box, in one step.

        A corpus is one model, so a box cannot be put into a keypoint-only corpus -
        that part of the architecture is right. What is *not* right is leaving the
        operator at a wall: the intent ("review these boxes") is answerable by
        creating the dataset whose model has a box, so this does exactly that. A
        keypoint corpus gains the box (``pose_full_boxes``); a detection corpus is
        created as ``boxes``.
        """
        schema_key = "pose_full_boxes" if self._schema.uses_keypoints else "boxes"
        base = self._dataset["name"] if self._dataset else "Daphnia"
        return self.create_dataset(name=name or f"{base} + boxes",
                                   schema_key=schema_key)

    def _offer_box_dataset(self) -> bool:
        """Ask whether to create the box-capable dataset; True when it was created.

        Refusing a keypoint-only corpus is correct but useless on its own
        (``X-N10``: an action that cannot proceed says what is missing). Here it can
        also *do* something about it, in one click - which is the difference between a
        wall and a step. **Only the button calls this**: a modal window has no place in
        the dialog-free core that a test, a script or the offscreen UI check drives.
        """
        question = QMessageBox(self)
        question.setWindowTitle("This dataset cannot hold a box")
        question.setText(
            f"'{self._dataset['name']}' is a keypoint-only dataset "
            f"({self._schema.label}), so it has nowhere to put a bounding box.\n\n"
            "Create the dataset whose model has a box, and open these boxes there?")
        create = question.addButton("Create and open it", QMessageBox.AcceptRole)
        question.addButton("Cancel", QMessageBox.RejectRole)
        question.exec()
        if question.clickedButton() is not create:
            self.status_label.setText(
                "Nothing opened. A box needs a dataset whose model has one - make "
                "one with 'New dataset...' and open the boxes there.")
            return False
        self.switch_to_box_dataset()
        return True

    def build_box_queue(self):
        """Queue every stored box of the open dataset - one animal per item.

        Opening boxes replaces the *frame* queue: the two ask different questions
        ("which animal is this?" vs "what is on this frame?"), and mixing them would
        make the position readout mean two things at once. The canvas leaves box
        drawing and landmark passes behind, because in this mode the box is the
        **item**, not something to draw, and the crop holds exactly one animal.
        """
        with connection() as conn:
            rows = queries.dataset_boxes(conn, self.dataset_id())
        self._items = [
            {"path": Path(row["file_path"]), "specimen": int(row["specimen_index"]),
             "box": (row["x1"], row["y1"], row["x2"], row["y2"]),
             "source": row["source"],
             "size": (int(row["image_width"]), int(row["image_height"]))}
            for row in rows]
        self._queue = []
        self._index = 0
        self._item_index = 0
        self._crop = None
        self._item_crop = None
        # A new queue opens in the view its corpus is worked in (keypoints -> the
        # crop; boxes only -> the whole frame). The toggle then sticks across
        # animals, because the operator chose it.
        self._crop_view = self._default_crop_view()
        self._loading = True
        combo = self.combo_mode
        combo.blockSignals(True)
        combo.setCurrentIndex(max(combo.findData(labeling.PAIR_MODE), 0))
        combo.blockSignals(False)
        self._box_mode = False
        self.canvas.box_mode = False
        self.workspace.set_checked("box", False)
        self._loading = False
        self._refresh_controls()
        return self._items

    def item_mode(self) -> bool:
        """True while the queue is one **animal** at a time, not one frame."""
        return bool(self._items)

    def _current_item(self):
        """The animal being reviewed/labelled, or ``None`` in frame mode."""
        if not self._items:
            return None
        return self._items[self._item_index]

    def _crop_point(self, point) -> tuple:
        """A click on the crop as **source-image** pixels - the frame a label stores.

        The single conversion between what the operator sees and what the corpus
        records (``mlContext.md`` law 1). In frame view (and outside item mode) there
        is no crop in force, so the click already *is* a source pixel.
        """
        if self._crop is None:
            return (float(point[0]), float(point[1]))
        return self._crop.invert(point)

    def _source_box(self, box) -> tuple:
        """A **dragged** box as source-image pixels - the same way back a click takes.

        Both corners go through the crop in force, so a box corrected in the magnified
        crop is stored as the rectangle it *is* and not as the one that was drawn on the
        enlargement (:func:`source_box`).
        """
        if self._crop is None or box is None:
            return box
        return source_box(self._crop, box)

    def _source_size(self) -> tuple:
        """``(width, height)`` of the frame a label is stored against.

        In the crop loop the canvas holds the *crop*, so the frame the label belongs to is
        the one the crop was taken from - never the enlargement's own pixel count.
        """
        frame = (self._item_frame if self._crop is not None and self._item_frame is not None
                 else self._frame)
        height, width = frame.shape[:2]
        return (int(width), int(height))

    @staticmethod
    def _point_is_inside(point, item, tolerance=2.0) -> bool:
        """True when a source-pixel point falls on (or just outside) a box.

        A couple of pixels of slack on purpose: an operator placing the spine base
        *on* the animal's edge is right, and the box is not a mask.
        """
        x1, y1, x2, y2 = item["box"]
        return (x1 - tolerance <= float(point[0]) <= x2 + tolerance
                and y1 - tolerance <= float(point[1]) <= y2 + tolerance)

    def _default_crop_view(self) -> bool:
        """Which view a box queue opens in.

        A model that labels keypoints is *placing points on one animal*, and the
        magnified crop is where that is convenient. A detection-only corpus is
        *reviewing boxes*, and a box can only be judged inside its frame - so that one
        opens on the whole frame.
        """
        return bool(self._schema.uses_keypoints)

    def _show_item(self, index=None):
        """Show one queued animal: the magnified crop, or the frame with its boxes.

        The canvas holds one of two pictures, and :meth:`_apply_item_view` decides
        which - and, with it, **the frame of reference in force**: the crop window is a
        *view*, and every click is converted back to source pixels through the one
        :class:`~daphnia.vision.crop.Crop` that owns the mapping (`mlContext.md` law
        1). In frame view there is no crop, so a click already *is* a source pixel.
        """
        if not self._items:
            return
        if index is not None:
            self._item_index = max(0, min(int(index), len(self._items) - 1))
        item = self._items[self._item_index]
        frame = self.canvas.set_source(item["path"])   # reads the file, or refuses it
        if frame is None:
            # The canvas is empty, so no animal is being reviewed: drop the item's
            # frame and crop instead of leaving the previous animal's overlays and
            # prompts in place (the queue itself stays - it came from a file on disk).
            self._item_frame = None
            self._item_crop = None
            self._crop = None
            self._frame = None
            self._path = None
            self._refresh_overlays()
            self._refresh_readout()
            self._refresh_stats()
            return
        height, width = frame.shape[:2]
        try:
            crop = crop_for_box(item["box"], (width, height))
        except ValueError as error:
            self.status_label.setText(
                f"The box of specimen {item['specimen']} on {item['path'].name} "
                f"cannot be shown - {error}")
            return
        self._item_frame = frame
        self._item_crop = crop
        self._path = item["path"]
        if self._path is not None:
            # The panel used to keep saying "No image yet" while a picture was on the
            # canvas: in a box queue the frame *is* the item's file, so say it.
            self.source_label.setText(
                f"Frame: {self._path.name}   (animal {self._item_index + 1} of "
                f"{len(self._items)})")
        self._apply_item_view()
        self._set_specimen(item["specimen"])
        self.position_label.setText(
            f"Animal {self._item_index + 1} of {len(self._items)}"
            f"   specimen {item['specimen']}  [{item['source']}]  {item['path'].name}")
        self._refresh_overlays()
        self._refresh_stats()
        self._refresh_readout()

    def _apply_item_view(self):
        """Put the chosen view of the current animal on the canvas.

        Frame view shows the **whole frame with every box** - where this animal is,
        and what its neighbours are, which is what a box can only be judged against.
        Crop view shows the animal alone, magnified, which is where keypoints are
        convenient to place. Both keep the original frame as the coordinate space.
        """
        item = self._current_item()
        if item is None or self._item_frame is None:
            return
        self._crop = self._item_crop if self._crop_view else None
        self._frame = (self._crop.window(self._item_frame) if self._crop_view
                       else self._item_frame)
        # In the magnified crop of a keypoint corpus the press **is** the keypoint
        # gesture: the item's box covers the whole picture, so the ordinary order (a
        # press inside a box selects it) swallowed every click and no landmark could be
        # placed at all (`SR-DATASET-025`, `ImageCanvas.point_mode`). The box itself is
        # corrected in *Frame* view, where the mode is off and every box is visible.
        self.canvas.point_mode = bool(self._crop is not None
                                      and self._schema.uses_keypoints)
        self._loading = True
        self.canvas.set_source(self._frame)
        self._loading = False
        self._refresh_overlays()
        self._refresh_tool_labels()
        self.prompt_label.setText(self._prompt())
        self.prelabel()              # the models in force answer on the view just shown

    def _refresh_tool_labels(self):
        """Say **which view is on**, on the tool that switches it.

        The label follows the project's convention for a checkable tool: a checked
        ``Pan`` means the hand tool is *on*, so a checked ``Crop`` means the magnified
        crop is on - the button names the state, not the action (the tooltip and the
        prompt say what pressing it does).
        """
        button = self.workspace.action("view")
        if button is None:
            return
        button.setText("Crop" if self._crop_view else "Frame")
        self.workspace.set_checked("view", self._crop_view)

    def set_view(self, crop_view):
        """Show the current animal magnified (``True``) or in its whole frame."""
        self._crop_view = bool(crop_view)
        self._apply_item_view()
        self._refresh_readout()
        return self._crop_view

    def toggle_view(self):
        """`C` and the tool: switch between the crop and the whole frame."""
        if not self._items:
            self.status_label.setText(
                "The crop is a view of one box - open a boxes export first.")
            return None
        self.set_view(not self._crop_view)
        self.status_label.setText(
            "Showing the whole frame with its boxes - click a point on the animal "
            "being labelled (it is the solid box)." if self._crop_view is False
            else f"Showing this animal magnified - {self._item_crop.describe()}.")
        return self._crop_view

    def _on_view_toggled(self, checked):
        """The Crop/Frame tool was clicked; the canvas follows it."""
        if self._loading or not self._items:
            return
        self.set_view(checked)
        self._refresh_readout()

    def _store_item_points(self, points):
        """Store this animal's interval as soon as both points exist.

        The crop loop is *click, click, next animal*, so the pair is stored on the
        second click rather than waiting for *Save* - the same rule a landmark pass
        follows. Both points are converted to **source pixels** through the crop
        first, and only then stored, so the corpus never sees crop coordinates.
        """
        item = self._current_item()
        if item is None or len(points) < 2:
            self._refresh_readout()
            return None
        first = self._crop_point(points[0])
        second = self._crop_point(points[1])
        height, width = self._item_frame.shape[:2]
        profile = self._active_profile()
        with connection() as conn:
            repository.save_corpus_label(
                conn, file_path=item["path"], image_size=(width, height),
                trait=self.trait(), p1=first, p2=second,
                specimen_index=item["specimen"], profile=profile,
                dataset_id=self.dataset_id())
        self._reload_labels()
        self._refresh_stats()
        self._refresh_overlays()
        note = ""
        if item["source"] == "model":
            # A prediction is never left unanswered. The keypoints of a *box* are a
            # human act, but they are not a verdict on the rectangle the detector
            # drew - so the queue waits here for a Yes or a No instead of moving on
            # and leaving a `source='model'` row behind (SR-PREDICT-N01, law 7).
            self.canvas.clear_points()
            note = "  Answer the box itself: Y (yes) or N (no)."
        if not self._point_is_inside(first, item) or not self._point_is_inside(second,
                                                                             item):
            # A point outside the animal's own box is usually a click on the
            # neighbour - said out loud, and stored anyway, because the box may
            # simply be tight (a label is never dropped silently).
            note += ("  Check this one: a point is outside specimen %d's box, so it "
                     "may be a click on a neighbour." % item["specimen"])
        self.status_label.setText(
            f"Stored specimen {item['specimen']} "
            f"{annotation.trait_label(self.trait())} on {item['path'].name} - "
            f"({first[0]:.0f}, {first[1]:.0f}) -> ({second[0]:.0f}, {second[1]:.0f}) "
            f"in frame pixels.{note}")
        if item["source"] == "model":
            self._refresh_readout()
            return item
        self._advance_item()
        return item

    def accept_box(self):
        """Yes: this box is a specimen - the human accept that makes it ground truth.

        A box the detector proposed is stored as ``source='model'``, and only a
        person may turn it into ground truth (``SR-PREDICT-N01``, ``mlContext.md``
        law 7). Accepting therefore rewrites the row to ``source='manual'`` and names
        the model that proposed it - which is also what ``SR-DATASET-008`` needs to
        report how well the two agree. A box drawn by hand is ground truth already;
        accepting it just moves on.
        """
        item = self._current_item()
        if item is None:
            self.status_label.setText(
                "No box to answer - open a boxes export (or draw a box) first.")
            return None
        if item["source"] == "model":
            with connection() as conn:
                repository.set_corpus_box_source(
                    conn, file_path=item["path"], specimen_index=item["specimen"],
                    dataset_id=self.dataset_id(), source="manual",
                    note="accepted as ground truth by a person")
        self._accepted += 1
        self._reload_labels()
        self.status_label.setText(
            f"Accepted the box of specimen {item['specimen']} on "
            f"{item['path'].name} - it is ground truth now "
            f"({self._accepted} accepted this session).")
        self._advance_item()
        return item

    def reject_box(self):
        """No: this box is not a specimen - remove it, so no model learns it.

        A rejected box is **deleted**, not kept: a detector trained on a rectangle a
        person said no to would learn the wrong animal. The frame stays in the
        corpus and the count is reported. Keeping the rejection itself as a **hard
        negative** is worth doing and needs somewhere to put it - that is
        ``SR-DATASET-014``, and this surface does not pretend to have it.
        """
        item = self._current_item()
        if item is None:
            self.status_label.setText(
                "No box to answer - open a boxes export (or draw a box) first.")
            return None
        with connection() as conn:
            repository.delete_corpus_box(
                conn, file_path=item["path"], specimen_index=item["specimen"],
                dataset_id=self.dataset_id())
        self._rejected += 1
        self._reload_labels()
        self.status_label.setText(
            f"Rejected the box of specimen {item['specimen']} on "
            f"{item['path'].name} - removed from the corpus "
            f"({self._rejected} rejected this session).")
        self._rebuild_items_at(self._item_index)
        return item

    def _advance_item(self):
        """Move to the next queued animal, and say so at the end of the queue."""
        if not self._items:
            return
        if self._item_index >= len(self._items) - 1:
            self.canvas.clear_points()
            self.status_label.setText(
                self.status_label.text() + "  This is the last animal in the queue.")
            self._refresh_readout()
            return
        self._show_item(self._item_index + 1)

    def _rebuild_items_at(self, index):
        """Rebuild the queue (a box was removed) and show what is now at ``index``."""
        self.build_box_queue()
        if not self._items:
            self._clear_item_state()
            return
        self._show_item(min(max(int(index), 0), len(self._items) - 1))

    def _clear_item_state(self):
        """Leave item mode: the queue is empty, so nothing is being reviewed."""
        self._items = []
        self._item_index = 0
        self._crop = None
        self._item_crop = None
        self._item_frame = None
        self._crop_view = True
        self.canvas.point_mode = False      # the keypoint gesture leaves with the crop
        self.canvas.clear_points()
        self._loading = True
        self.canvas.clear_box()
        self.canvas.set_boxes([])
        self.canvas.set_annotations([])
        self._loading = False
        self._refresh_controls()
        self._refresh_readout()

    def _clear_item_label(self):
        """Item mode: clear **this animal's** stored intervals, not the frame's.

        Clearing the whole frame would delete the labels of the other animals on it,
        which the queue has not reviewed yet - so "clear" follows what is on screen.
        """
        item = self._current_item()
        if item is None:
            return 0
        removed = 0
        with connection() as conn:
            for trait in self._traits():
                removed += repository.delete_corpus_label(
                    conn, file_path=item["path"], specimen_index=item["specimen"],
                    trait=trait, dataset_id=self.dataset_id())
        self.canvas.clear_points()
        self._reload_labels()
        self._refresh_stats()
        self._refresh_overlays()
        self.status_label.setText(
            f"Cleared {removed} stored label(s) of specimen {item['specimen']} on "
            f"{item['path'].name} - the frame's other animals are untouched.")
        return removed

    def _refresh_item_overlays(self):
        """Draw the queue the way the current view allows.

        **Frame view**: every stored box of the frame, each in its specimen's colour,
        with the animal being labelled drawn **solid on top** - that is the "bounding
        box display" the corpus needs, and it is the only way to see that the box is
        actually around an animal and not beside one. **Crop view**: only the animal on
        screen - in a magnified crop a neighbour's edge looks like part of *this*
        animal, so drawing them would be worse than useless.
        """
        item = self._current_item()
        if item is None or self._item_frame is None:
            return
        if self._crop is None:
            self.canvas.set_annotations(
                self._annotation_items(self._known_segments()))
            self.canvas.set_boxes(
                [(index, (row["x1"], row["y1"], row["x2"], row["y2"]))
                 for index, row in sorted(self._stored_boxes().items())])
            self._set_active_box(item["box"])
            return
        segments = []
        for (specimen, _trait), row in self._stored().items():
            if specimen != item["specimen"]:
                continue
            segments.append((self._crop.to_crop((row["p1_x"], row["p1_y"])),
                             self._crop.to_crop((row["p2_x"], row["p2_y"]))))
        self.canvas.set_annotations(
            [(item["specimen"], segments)] if segments else [])
        self.canvas.set_boxes([(item["specimen"], crop_box(self._crop, item["box"]))])

    def _set_active_box(self, box):
        """Highlight one box as the *active* one, storing nothing.

        ``set_box`` is what the canvas draws in the marker colour on top of the
        overlays, so the animal being labelled is unmistakable among its neighbours -
        and ``_loading`` keeps the resulting ``box_changed`` from being read as a
        newly drawn box (which would store it).
        """
        self._loading = True
        self.canvas.set_box(box)
        self._loading = False

    # -- labelling ---------------------------------------------------------
    def _on_subject_changed(self, *_):
        """A control changed: refresh the prompt, the readout and the overlays.

        In *pair* mode the canvas shows the current specimen's interval (loaded from
        the store, or empty). In *pass* mode the canvas shows the whole current pass -
        one point per specimen for the chosen landmark. In *box* mode the canvas shows
        the current specimen's **stored** box when there is one, so correcting it is a
        redraw of the box you can already see.
        """
        if self._box_mode:
            self._show_box()
        elif self.mode() == labeling.PASS_MODE:
            self._show_pass()
        else:
            self._loading = True
            self.canvas.max_points = 2
            self.canvas.show_point_labels = True
            self.canvas.connect_points = True     # two points = one interval
            self.canvas.point_names = list(self._points())
            # The editable set holds only what is being clicked *now*; a stored interval
            # is an overlay (coloured and numbered), so Undo can still take it back and
            # re-measuring a specimen never starts from stale points.
            self.canvas.clear_points()
            self._loading = False
        self._refresh_keypoint_label()
        self.prompt_label.setText(self._prompt())
        self._refresh_readout()
        self._refresh_overlays()

    def _show_box(self):
        """Show the current specimen's stored box as the editable one - or none, in edit mode.

        The active box is what a drag replaces, so selecting a specimen with a box already
        stored shows that box: a correction is a redraw of the rectangle you can see, not a
        blind new one. The overlays keep showing every other specimen.

        **In *Edit* mode it is left empty**: the *selection* is what is being edited there (drawn
        in the accent colour), and siting the same rectangle a second time as the "active" box
        drew every selected animal **twice** - "we always see two duplicate boxes" - with no way
        to tell which of the two a drag would move.
        """
        if self.canvas.box_edit:
            self._loading = True
            self.canvas.max_points = 2
            self.canvas.point_names = []
            self.canvas.clear_points()
            self.canvas.clear_box()
            self._loading = False
            return
        stored = self._stored_boxes().get(self.specimen())
        self._loading = True
        self.canvas.max_points = 2
        self.canvas.point_names = []
        self.canvas.clear_points()
        self.canvas.set_box(
            (stored["x1"], stored["y1"], stored["x2"], stored["y2"])
            if stored is not None else None)
        self._loading = False

    def _show_pass(self):
        """Show the pass through the overlays, so each specimen keeps its own colour.

        The pass points are *not* the editable set: leaving the canvas empty means every
        placed point is drawn by :meth:`_refresh_overlays` in its specimen's colour from
        the moment it is placed, instead of all sharing the marker colour until the pass
        is switched.
        """
        self._loading = True
        self.canvas.max_points = PASS_MAX_POINTS
        self.canvas.show_point_labels = False
        self.canvas.connect_points = False     # same landmark, different specimens
        self.canvas.point_names = []
        self.canvas.clear_points()
        self._loading = False

    def _known_segments(self):
        """``{specimen_index: [(segment, trait), ...]}`` - the intervals known on this frame.

        Stored labels give the measured intervals (length = head->spine base, width =
        left->right) **tagged with their trait**, so a press on one can be mapped back to the
        row it came from (`X-N16`); a point placed in the open pass but not yet paired gives a
        zero-length segment with no tag, so it shows as a dot instead of a line drawn to a
        *different* specimen.
        """
        by_specimen = {}
        for (specimen, trait), row in self._stored().items():
            by_specimen.setdefault(specimen, []).append(
                (((row["p1_x"], row["p1_y"]), (row["p2_x"], row["p2_y"])), trait))
        for specimen, points in self._draft.points_by_specimen().items():
            for point in points:
                by_specimen.setdefault(specimen, []).append(((point, point), None))
        return by_specimen

    def _annotation_items(self, segments_by_specimen):
        """Turn ``{specimen: [(segment, tag), ...]}`` into ``set_annotations`` items.

        It also records ``(specimen, segment_index) -> tag`` in ``self._annotation_tags``, so a
        press on a drawn segment is reported back as the concrete stored row it is (`X-N16`).
        """
        self._annotation_tags = {}
        items = []
        for specimen, entries in sorted(segments_by_specimen.items()):
            segments = [segment for segment, _tag in entries]
            tags = [tag for _segment, tag in entries]
            for segment, tag in enumerate(tags):
                if tag is not None:
                    self._annotation_tags[(specimen, segment)] = tag
            items.append((specimen, segments, "", tags))
        return items

    def _on_annotation_point_selected(self, specimen, segment, endpoint):
        """A stored keypoint pair was clicked: load that label back so the point can be dragged.

        The trait comes from the tag the page put on the segment, and the two points come from
        the stored row - so *which* label is being corrected is never guessed (`X-N16`). The
        correction is written when the point is let go (`_on_points_edited`).
        """
        trait = self._annotation_tags.get((specimen, segment))
        if trait is None:
            return
        row = self._stored().get((int(specimen), trait))
        if row is None:
            return
        self._loading = True
        self._trait = trait
        self._set_specimen(specimen)          # clears the draft, as a real selection would
        self.canvas.set_points([(row["p1_x"], row["p1_y"]),
                                (row["p2_x"], row["p2_y"])])
        self.canvas.select_point(endpoint)
        self._loading = False
        self._refresh_keypoint_label()
        self.status_label.setText(
            f"Editing specimen {specimen} {annotation.trait_label(trait)} - drag the point; "
            "it is stored when you let go.")

    def _refresh_overlays(self):
        """Colour every specimen's known intervals *and* boxes, per specimen.

        Both overlays share ``specimen_color``, so a specimen's box and its landmarks
        are readable as the same animal at a glance. In **item** mode only the animal
        on screen is drawn, in crop pixels.
        """
        if self._items:
            self._refresh_item_overlays()
            # Even in the box queue the boxes on screen can be picked - the *Select* tool
            # appears with them, and leaves the bar when there is nothing to pick (`X-N10`).
            return
        known = self._known_segments()
        self.canvas.set_annotations(self._annotation_items(known))
        self.canvas.set_boxes(
            [(index, (row["x1"], row["y1"], row["x2"], row["y2"]))
             for index, row in sorted(self._stored_boxes().items())])

    def _stored(self) -> dict:
        """The stored labels of the current frame: ``{(specimen, trait): row}``."""
        return self._labels.get(str(self._path), {}) if self._path else {}

    def _stored_boxes(self) -> dict:
        """``{specimen_index: box row}`` - the boxes stored on the current frame."""
        return self._boxes.get(str(self._path), {}) if self._path else {}

    def _landmark_specimens(self, landmark) -> set:
        """Specimens already carrying ``landmark`` - stored or placed in this session."""
        present = set(self._draft.placed_landmarks(landmark))
        trait = self._schema.keypoint_system.trait_of(landmark)
        if trait is not None:
            present.update(specimen for (specimen, name) in self._stored()
                           if name == trait)
        return present

    def _next_specimen_for(self, landmark) -> int:
        """First specimen with no point for ``landmark`` yet - stored or in the draft.

        Stored labels count as done, so after the spine base of specimen 1 is stored the
        pass moves on to specimen 2 instead of pointing at 1 again.
        """
        present = self._landmark_specimens(landmark)
        index = 1
        while index in present:
            index += 1
        return index

    def _specimen_count(self) -> int:
        """How many specimens the frame holds - the highest index seen anywhere.

        The first pass defines it: once specimens 1..N carry a head, a later pass knows
        to stop at N instead of asking the operator to count.
        """
        indices = set(self._draft.specimens())
        indices.update(specimen for (specimen, _trait) in self._stored())
        return max(indices) if indices else 0

    def _landmarks_populated(self) -> set:
        """The landmarks that already have a point on this frame, stored or drafted."""
        present = set(self._draft.landmarks())
        intervals = labeling.intervals_for(self._schema)
        for (_specimen, trait) in self._stored():
            present.update(intervals.get(trait, ()))
        return present

    def next_landmark(self):
        """Advance to the next landmark pass by hand (no wrap past the end)."""
        following = labeling.next_landmark(self.landmark(),
                                           order=self._schema.keypoints)
        if following is None:
            self.status_label.setText(
                "No further landmark - this frame is fully labelled.")
            return
        self.set_landmark(following)
        self.status_label.setText(f"Now placing '{following}'.")

    def _advance_landmark_if_complete(self) -> bool:
        """Auto-move to the next landmark once the current pass covers every specimen.

        The specimen count is whatever the frame already holds (the first pass sets it).
        A pass is complete when its landmark sits on all ``1..N`` specimens **and** at
        least one other landmark exists - so the open-ended *first* pass is never cut
        short, and every later pass moves the operator on instead of making them cycle
        specimens by hand.
        """
        total = self._specimen_count()
        if total == 0 or len(self._landmarks_populated()) < 2:
            return False
        if not set(range(1, total + 1)) <= self._landmark_specimens(self.landmark()):
            return False
        following = labeling.next_landmark(self.landmark(),
                                           order=self._schema.keypoints)
        if following is None:
            self.status_label.setText(
                "All landmarks placed - this frame is fully labelled.")
            return False
        self.set_landmark(following)
        self.status_label.setText(
            f"'{following}' pass ready - the previous landmark is complete for all "
            f"{total} specimen(s).")
        return True

    def _on_points_changed(self, points):
        """A click in *pass* mode places the landmark for the next specimen.

        In **item** mode a second click completes this animal's interval and stores
        it straight away - the crop loop is click, click, next animal.
        """
        if self._loading or self._box_mode:
            return
        if self._items:
            if len(points) >= 2:
                self._store_item_points(points)
            else:
                self._refresh_readout()
            return
        if self.mode() == labeling.PASS_MODE and points:
            specimen = self._next_specimen_for(self.landmark())
            self._draft.place(specimen, self.landmark(), points[-1])
            self._commit_completed()
            self._loading = True
            self._set_specimen(self._next_specimen_for(self.landmark()))
            self._loading = False
            self._show_pass()
            self._refresh_overlays()
            self._advance_landmark_if_complete()
        self._refresh_readout()

    def _commit_completed(self) -> int:
        """Store every interval the pass just completed; returns how many."""
        done = self._draft.take_complete()
        if not done:
            return 0
        path = self._ensure_file()
        if path is None:
            return 0
        height, width = self._frame.shape[:2]
        profile = self._active_profile()
        with connection() as conn:
            for specimen, trait, p1, p2 in done:
                repository.save_corpus_label(
                    conn, file_path=str(path), image_size=(width, height),
                    trait=trait, p1=p1, p2=p2, specimen_index=specimen,
                    profile=profile, dataset_id=self.dataset_id())
        self._reload_labels()
        self._refresh_stats()
        self.status_label.setText(
            f"Stored {len(done)} interval(s) on {Path(path).name}.")
        return len(done)

    def _refresh_readout(self, *_):
        """Write the readout **in the panel**, and publish the window facts.

        The readout is the panel's ``readout_label``: the value being labelled - the
        thing being decided click by click - the queue position and the last message.
        It is *not* repeated over the picture any more (2026-10-18); the header carries
        the context (the queue, the message), because that belongs to the window rather
        than to the specimen.
        """
        self._refresh_panel_readout()
        self.workspace.set_header([
            Kpi("queue", self.position_label.text(), label="Queue"),
            Kpi("status", self.status_label.text(), label="Status"),
        ])

    def _refresh_panel_readout(self, *_):
        if self._box_mode or not self._schema.uses_keypoints:
            self.readout_label.setText(self._box_readout())
            return
        if self.mode() == labeling.PASS_MODE:
            self.readout_label.setText(self._pass_progress())
            return
        points = self.canvas.points
        if len(points) < 2:
            self.readout_label.setText(self._prompt())
            return
        # The canvas may be showing a **crop**, so a click is converted to source
        # pixels before it is measured or read: what is stored is what is reported
        # (mlContext.md law 1).
        first = self._crop_point(points[0])
        second = self._crop_point(points[1])
        keypoints = Keypoints(eye_apex=first, spine_base=second)
        profile = self._active_profile()
        label = annotation.trait_label(self.trait())
        if profile is None:
            self.readout_label.setText(
                f"{label}: {keypoints.length_px:.1f} px "
                f"(no scale - stored in pixels)")
            return
        um = profile.to_um(keypoints.length_px)
        total = units.format_total_uncertainty(um, profile.px_per_um,
                                               profile.uncertainty_pct)
        self.readout_label.setText(
            f"{label}: "
            f"{units.format_with_uncertainty(um, profile.uncertainty_pct)}   "
            f"{total}   "
            f"({keypoints.length_px:.1f} px)")

    def _box_readout(self) -> str:
        """What the box of the current specimen is, or what to do to make one."""
        item = self._current_item()
        if item is not None:
            x1, y1, x2, y2 = item["box"]
            crop = (f"   crop {self._crop.describe()}" if self._crop is not None else "")
            height, width = (self._item_frame.shape[:2]
                             if self._item_frame is not None else (0, 0))
            return (f"Box of specimen {item['specimen']}: {x2 - x1:.0f} x "
                    f"{y2 - y1:.0f} px at ({x1:.0f}, {y1:.0f})  [{item['source']}] - "
                    f"Yes keeps it as ground truth, No removes it.{crop}"
                    f"\nThe crop is only a view: labels are stored on the original "
                    f"{width} x {height} frame.")
        box = self.canvas.box
        stored = self._stored_boxes().get(self.specimen())
        if self.canvas.box_pending:
            origin_x, origin_y = self.canvas.box_origin
            live = "" if box is None else (
                f"  Now {box[2] - box[0]:.0f} x {box[3] - box[1]:.0f} px")
            return (f"Corner at ({origin_x:.0f}, {origin_y:.0f}) - click the "
                    f"opposite corner.{live}")
        if box is None and stored is not None:
            box = (stored["x1"], stored["y1"], stored["x2"], stored["y2"])
        if box is None:
            return self._prompt()
        x1, y1, x2, y2 = box
        return (f"Box of specimen {self.specimen()}: "
                f"{x2 - x1:.0f} x {y2 - y1:.0f} px at ({x1:.0f}, {y1:.0f})")

    def _pass_progress(self) -> str:
        """How many intervals the pass has, per trait - stored plus in progress."""
        stored = self._stored()
        parts = []
        for trait in self._traits():
            count = sum(1 for (_s, name) in stored if name == trait)
            parts.append(f"{annotation.trait_label(trait)} "
                         f"{count + self._draft.count(trait)}")
        total = self._specimen_count()
        suffix = f"   {total} specimen(s)" if total else ""
        boxes = len(self._stored_boxes())
        if self._schema.exports_boxes and boxes:
            parts.append(f"boxes {boxes}")
        return f"Pass: {' / '.join(parts)}{suffix}   (placing '{self.landmark()}')"

    def _next_specimen_missing_box(self) -> int:
        """First specimen with no stored box - so correcting box 2 does not jump to 3.

        The first boxed pass defines the count exactly as the first landmark pass does:
        the dropdown grows as specimens are added, and once every known specimen has a
        box the next free index is offered so more can be added.
        """
        present = set(self._stored_boxes())
        index = 1
        while index in present:
            index += 1
        return index

    def _next_specimen(self):
        """Start the next specimen (or the next one missing a label)."""
        if self._box_mode:
            self._set_specimen(self._next_specimen_missing_box())
            self.status_label.setText(self._prompt())
            return
        if self.mode() == labeling.PASS_MODE:
            self._set_specimen(self._next_specimen_for(self.landmark()))
            self.status_label.setText(self._prompt())
            return
        self._set_specimen(self._specimen + 1)
        self.canvas.clear_points()
        self.status_label.setText(f"Specimen {self.specimen()} - {self._prompt()}")

    # -- mode, hand tool, fixing a pass ------------------------------------
    def _refresh_controls(self):
        """Show only the controls the open dataset's model actually uses.

        A keypoint corpus explains itself with the trait/keypoint controls; a
        detection-only corpus has nothing to click but a box, so the keypoint
        controls are hidden instead of disabled - a dead control looks broken.

        In **item** mode the shape of the work is different again: the box *is* the
        item (so drawing another one is not offered), a crop holds exactly one animal
        (so the pass mode makes no sense), the pair stores itself, and the two
        questions that *are* asked are Yes and No.
        """
        uses_keypoints = self._schema.uses_keypoints
        items = self.item_mode()
        pass_mode = (not items) and self.mode() == labeling.PASS_MODE
        self.combo_mode.setVisible(uses_keypoints and not items)
        self.keypoint_label.setVisible(uses_keypoints)
        self.workspace.set_visible("next_keypoint", uses_keypoints and not items)
        self.workspace.set_visible("box", self._schema.exports_boxes and not items)
        # Landmark mode and box mode store each label as soon as it exists, so Save
        # is only meaningful in two-point keypoint mode - and in item mode the pair
        # is stored by the second click.
        self.workspace.set_visible("save", uses_keypoints and not pass_mode
                                   and not self._box_mode and not items)
        for key in ("accept", "reject", "view"):
            self.workspace.set_visible(key, items)
        self.btn_rename_dataset.setEnabled(self._dataset is not None)

    # -- mode, hand tool, fixing a pass ------------------------------------
    def _on_mode_changed(self, *_):
        """Switch modes, showing only the controls the new mode actually uses."""
        self._refresh_controls()
        self._on_subject_changed()
        self.status_label.setText(
            f"{labeling.MODE_LABELS[self.mode()]} - {self._prompt()}")

    # -- the bounding box --------------------------------------------------
    def toggle_box_mode(self):
        """`B` and the button: label boxes - refused when the model has none.

        A refusal is reported in words instead of doing nothing: the operator asked for
        something the open dataset's model cannot use, and the fix is another dataset.
        """
        if not self._schema.exports_boxes:
            self.status_label.setText(
                f"'{self._dataset['name']}' is a keypoint-only model - create a "
                "dataset with a bounding box to label boxes.")
            return None
        self.workspace.action("box").setChecked(
            not self.workspace.action("box").isChecked())
        self.status_label.setText(self._prompt())
        return self._box_mode

    def _on_box_toggled(self, checked):
        """Turn box drawing on or off; the canvas follows the model and the specimen."""
        if self._loading:
            return
        self._box_mode = bool(checked)
        self.canvas.box_mode = self._box_mode
        self._refresh_controls()
        self._on_subject_changed()
        self.status_label.setText(self._prompt())

    def _on_box_anchor_changed(self, anchored):
        """One corner was anchored (or released): say what the next click will do.

        A corner is *not* a label, so this never stores anything - it only keeps the
        prompt and the readout honest between the two clicks of a box.
        """
        self.prompt_label.setText(self._prompt())
        self._refresh_readout()

    def _on_box_changed(self, boxes):
        """A box was dragged: store it for the selected specimen and move on."""
        if self._loading or not boxes:
            return
        if self._frame is None and self._path is None:
            self.status_label.setText(
                "Nothing to label yet.  Import a folder or open an image first.")
            return
        path = self._ensure_file()
        if path is None:
            return
        height, width = self._frame.shape[:2]
        specimen = self.specimen()
        box = boxes[0]
        # The box is stored *before* the overlap is reported: two boxes on one animal
        # is worth saying out loud on a frame with several individuals, but a label
        # is never thrown away silently (SR-DATASET-N03) - the operator decides.
        overlapping = self.canvas.overlaps_stored(box)
        with connection() as conn:
            repository.save_corpus_box(
                conn, file_path=str(path), image_size=(width, height), box=box,
                specimen_index=specimen, dataset_id=self.dataset_id())
        self._reload_labels()
        self._refresh_stats()
        self._refresh_overlays()
        x1, y1, x2, y2 = box
        # Storing IS moving on: the next specimen missing a box becomes editable, and
        # its own box (if any) is sited. The box just stored is an overlay by then.
        self._next_specimen()
        self.status_label.setText(
            f"Stored the box of specimen {specimen} "
            f"({x2 - x1:.0f} x {y2 - y1:.0f} px) on {Path(path).name}."
            + (f"  Check this one: it overlaps specimen {overlapping}'s box, so it "
               "may be the same animal boxed twice." if overlapping else ""))

    def _on_landmark_changed(self, *_):
        """A new landmark pass: show it and jump to the first specimen missing it."""
        if self.mode() != labeling.PASS_MODE:
            return
        self._loading = True
        self._set_specimen(self._next_specimen_for(self.landmark()))
        self._loading = False
        self._show_pass()
        self._refresh_overlays()
        self._refresh_keypoint_label()
        self._refresh_readout()
        self.prompt_label.setText(self._prompt())

    def _tool_action(self, action):
        """Run one tool from the canvas, then repeat the panel's answer on the image.

        A tool pressed on the image must answer on the image: in full screen the
        panel - and every message in it - is hidden.
        """
        action()
        self._refresh_readout()

    def undo_point(self):
        """Undo the last thing placed on this frame - a loose point, a box, a row."""
        if self._box_mode:
            self._undo_box()
        elif self.mode() == labeling.PASS_MODE:
            self._undo_pass()
        else:
            self._undo_pair()

    def _undo_box(self):
        """Box mode: remove the highest stored box of this frame (the last drawn)."""
        boxes = self._stored_boxes()
        if not boxes:
            self.status_label.setText("No stored box to undo on this frame.")
            return
        specimen = max(boxes)
        self._delete_stored_box(specimen)
        self._set_specimen(specimen)
        self._refresh_overlays()
        self._refresh_readout()
        self.status_label.setText(f"Removed the box of specimen {specimen}.")

    def _delete_stored_box(self, specimen):
        """Delete one stored box and re-read what depends on it."""
        if self._path is not None:
            with connection() as conn:
                repository.delete_corpus_box(
                    conn, file_path=self._path, specimen_index=specimen,
                    dataset_id=self.dataset_id())
        self._reload_labels()
        self._refresh_stats()

    def _undo_pair(self):
        """Two-point mode: drop the clicked points, then the stored interval.

        In **item** mode the interval undone is the current animal's, never another
        animal's on the same frame - the queue is one animal at a time, so undo must
        be too.
        """
        if self.canvas.points:
            self.canvas.clear_points()
            self.status_label.setText("Cleared the clicked points.")
            return
        item = self._current_item()
        if item is not None:
            with connection() as conn:
                removed = repository.delete_corpus_label(
                    conn, file_path=item["path"], specimen_index=item["specimen"],
                    trait=self.trait(), dataset_id=self.dataset_id())
            self._reload_labels()
            self._refresh_stats()
            self._refresh_overlays()
            self.status_label.setText(
                f"Removed specimen {item['specimen']}'s stored "
                f"{annotation.trait_label(self.trait())}."
                if removed else
                f"Nothing stored for specimen {item['specimen']} "
                f"{annotation.trait_label(self.trait())}.")
            return
        trait = self.trait()
        specimens = [s for (s, name) in self._stored() if name == trait]
        if not specimens:
            self.status_label.setText(
                f"Nothing to undo for {annotation.trait_label(trait)}.")
            return
        specimen = max(specimens)
        self._delete_stored(specimen, trait)
        self._on_subject_changed()
        self.status_label.setText(
            f"Removed the stored {annotation.trait_label(trait)} of specimen {specimen}.")

    def _undo_pass(self):
        """Landmark mode: drop the last loose point, then the last stored interval."""
        last = self._draft.last()
        if last is not None:
            specimen, landmark = last
            self._draft.unplace(specimen, landmark)
            self._landmark = landmark
            self._refresh_keypoint_label()
            self._set_specimen(specimen)
            self._on_subject_changed()
            self.status_label.setText(f"Removed {landmark} of specimen {specimen}.")
            return

        # Everything is stored: undo the interval of the highest specimen. The pass may
        # already have auto-advanced to the next landmark, so fall back to whatever the
        # frame does have, not only the current landmark's trait.
        rows = self._stored()
        trait = self._schema.keypoint_system.trait_of(self.landmark())
        specimens = [s for (s, name) in rows if name == trait]
        if specimens:
            specimen = max(specimens)
        elif rows:
            specimen, trait = max(rows, key=lambda key: key[0])
        else:
            self.status_label.setText("Nothing to undo in this pass.")
            return
        self._landmark = labeling.intervals_for(self._schema)[trait][0]
        self._refresh_keypoint_label()
        self._delete_stored(specimen, trait)
        self._set_specimen(self._next_specimen_for(self._landmark))
        self._on_subject_changed()
        self.status_label.setText(
            f"Removed the stored {trait} of specimen {specimen}.")

    def _delete_stored(self, specimen, trait):
        """Delete one stored label and re-read what depends on it."""
        if self._path is not None:
            with connection() as conn:
                repository.delete_corpus_label(
                    conn, file_path=self._path, specimen_index=specimen,
                    trait=trait, dataset_id=self.dataset_id())
        self._reload_labels()
        self._refresh_stats()

    def clear_points(self):
        """Forget the points placed on this frame (never the stored labels)."""
        if self._box_mode:
            self._loading = True
            self.canvas.clear_box()
            self._loading = False
            self.status_label.setText(
                "Cleared the drawn box - the stored boxes stay.")
        elif self.mode() == labeling.PASS_MODE:
            self._draft.clear()
            self._show_pass()
            self._refresh_overlays()
            self.status_label.setText("Cleared the points placed on this frame.")
        else:
            self.canvas.clear_points()

    def clear_frame(self):
        """Forget **everything** on this frame: stored labels and the open points.

        A folder import can hold frames from different studies with different specimen
        counts, so a frame has to be re-doable from scratch - this removes the stored
        labels of the frame, not only the points currently drawn on the canvas.
        """
        if self._frame is None and self._path is None:
            self.status_label.setText("No frame to clear.")
            return
        if self._items:
            self._clear_item_label()
            return
        removed = 0
        removed_boxes = 0
        if self._path is not None:
            with connection() as conn:
                removed = repository.delete_corpus_labels_for_file(
                    conn, self._path, dataset_id=self.dataset_id())
                removed_boxes = repository.delete_corpus_boxes_for_file(
                    conn, self._path, dataset_id=self.dataset_id())
        self._draft.clear()
        self._loading = True
        self.canvas.clear_box()
        self._loading = False
        self._reload_labels()
        self._refresh_stats()
        self._on_subject_changed()
        self.status_label.setText(
            f"Cleared {removed} stored label(s), {removed_boxes} box(es) and the "
            f"open points on this frame - other datasets are untouched.")

    # -- storing -----------------------------------------------------------
    def save_label(self):
        """Store this landmark pair in the corpus - no study or scale needed.

        In *pass* mode and in *box* mode there is nothing to press: an interval is
        stored the moment its second landmark is placed, and a box the moment the
        drag is released, so Save just says that.
        """
        if self._box_mode:
            self.status_label.setText(
                self._prompt() + "  A box is stored as soon as you release the "
                "drag - there is nothing to press.")
            return
        if not self._schema.uses_keypoints:
            self.status_label.setText(
                "This dataset labels bounding boxes only - drag one around a "
                "specimen.")
            return
        if self.mode() == labeling.PASS_MODE:
            saved = self._commit_completed()
            self.status_label.setText(
                "One-landmark mode stores each interval as soon as both ends exist"
                + (f" - {saved} stored now." if saved else
                   " - place the matching landmark to finish an interval."))
            return
        if self._frame is None:
            self.status_label.setText(
                "Nothing to label yet.  Import a folder or open an image first.")
            return
        points = self.canvas.points
        if len(points) < 2:
            self.status_label.setText(self._prompt())
            return

        path = self._ensure_file()
        if path is None:
            return
        height, width = self._frame.shape[:2]
        specimen = self.specimen()
        profile = self._active_profile()
        with connection() as conn:
            repository.save_corpus_label(
                conn, file_path=str(path), image_size=(width, height),
                trait=self.trait(), p1=points[0], p2=points[1],
                specimen_index=specimen, profile=profile,
                dataset_id=self.dataset_id())
        scale = (f" and {profile.px_per_um:.4f} px/um" if profile else " in pixels")
        self._reload_labels()
        self._refresh_stats()
        self._refresh_overlays()
        self._next_specimen()       # save IS "next specimen" - no button for it
        self.status_label.setText(
            f"Labelled specimen {specimen} {annotation.trait_label(self.trait())} "
            f"on {Path(path).name} - stored{scale}.  Next specimen ready.")

    def _ensure_file(self):
        """A storable file for the current image, or ``None`` if it cannot be written.

        A live frame is written to ``datasets/raw`` the first time it is saved, so a
        stored label always has an image behind it - and a quick look that is never
        saved leaves nothing behind.
        """
        if self._path is not None:
            return self._path
        paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = paths.RAW_DIR / f"dataset_{stamp}.png"
        # The stamp has second resolution, so two frames saved in the same second
        # would share a path - and the corpus keys a label by that path, so the two
        # animals' labels would merge into one row. A free name is found instead.
        counter = 1
        while target.exists():
            counter += 1
            target = paths.RAW_DIR / f"dataset_{stamp}_{counter}.png"
        if not cv2.imwrite(str(target), self._frame):
            self.status_label.setText(f"Could not write {target} - not saved.")
            return None
        self._path = target
        self.source_label.setText(f"Frame: {target.name} (from the live frame)")
        return target

    # -- export ------------------------------------------------------------
    def _suggested_export_folder(self) -> Path:
        """Where this dataset's export goes unless the operator says otherwise.

        Named after the dataset and stamped, so exporting three models never mixes
        them up in one folder - and the *model* is part of the name, because two
        exports that differ only in their model are the one thing that must never be
        confused.
        """
        slug = "".join(ch if (ch.isalnum() or ch in "-_") else "_"
                       for ch in f"{self._dataset['name']}")
        slug = slug.strip("_") or "dataset"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return paths.ANNOTATIONS_DIR / f"{slug}_{stamp}"

    def _export_summary(self, result, folder) -> str:
        """The report after an export - including an empty split, if there is one.

        A small corpus cannot fill three buckets, and a silently empty validation
        set would make a "why is my score NaN" question later. Saying it here costs
        one sentence.
        """
        splits = result["splits"]
        empty = [name for name, count in splits.items() if not count]
        note = (f"  Note: the {'/'.join(sorted(empty))} split is still empty - the "
                "corpus is small, and the split is deterministic by image."
                if empty else "")
        return (f"Exported '{result['dataset']['name']}' "
                f"({result['schema']['label']}): {result['labels']} keypoint "
                f"label(s), {result['boxes']} box(es) and {result['images']} image(s) "
                f"to {folder} - train {splits['train']}, val {splits['val']}, "
                f"test {splits['test']} (see splits/ for the image lists).{note}")

    def export(self):
        """Write **this dataset** as images + a machine-readable label file.

        The export carries what the dataset's model needs - the named keypoints of its
        system and/or the bounding boxes - so a training run needs no other source of
        truth than the exported folder.
        """
        with connection() as conn:
            stats = queries.dataset_stats(conn, self.dataset_id())
        if not stats["labels"] and not stats["boxes"]:
            self.status_label.setText(
                f"Nothing to export in '{self._dataset['name']}' yet - label a "
                "frame first.")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Export the dataset to a folder",
            str(self._suggested_export_folder()))
        if not folder:
            return
        with connection() as conn:
            result = maintenance.export_dataset(conn, self.dataset_id(), folder)
        self.status_label.setText(self._export_summary(result, folder))

