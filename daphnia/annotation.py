"""Annotation schemas: what a model is trained to predict, and therefore what
an operator labels.

The app no longer has *one* annotation vocabulary. A **dataset** is the corpus for
one model, and the model decides what a label *is*:

* **two keypoints** - head -> tail, enough for a length model;
* **four keypoints** - head, tail and both sides, which also gives a width;
* **bounding box** - where the animal is, which is what a detector learns.

The landmarks keep the canonical names of :mod:`daphnia.study` (``head``,
``spine base``, ``left side``, ``right side``), so a corpus label and a study
measurement always name the same anatomy - but a *dataset* may also be exported
with the short names a training config expects (``tail``, ``sideL``, ``sideR``).

This module is deliberately **Qt-free** and free of any storage concern: the
database stores a schema *key*, the canvas asks for the keypoints, the exporter
asks for the export names and every part of the app therefore agrees.

The live bounding-box *prediction* (a model detecting specimens in real time) is
explicitly out of scope; what lives here is the *labelling* side that has to
exist before such a model can be trained.
"""

from dataclasses import dataclass

from daphnia.study import LANDMARK_TRAIT, TRAITS, TRAIT_LABELS

#: The canonical landmark names, taken from :mod:`daphnia.study` so no surface
#: invents its own wording (``X-N08``).
HEAD = "head"
TAIL = "spine base"
LEFT = "left side"
RIGHT = "right side"

#: The trait of an interval that spans the body, and the one that spans it across.
LENGTH = "length"
WIDTH = "width"

#: The trait a bounding box is stored under. It is a *pseudo* trait: a box is not a
#: distance between two landmarks, but it is one more thing an operator can record
#: for a specimen, so it travels with the same rows (there is no fourth concept).
BOX_TRAIT = "bbox"

#: How the box "trait" is phrased in the UI.
BOX_LABEL = "Bounding box"

#: Trait label lookup that also knows the box.
LABELS = dict(TRAIT_LABELS)
LABELS[BOX_TRAIT] = BOX_LABEL

#: A box smaller than this (in source-image pixels) is a mis-click, not a specimen.
#: The single source of truth: the canvas refuses to *make* one, the corpus export
#: drops one from training, and ``maintenance.clamp_corpus_boxes`` reports one that
#: a devious drag left outside the image. A box the size of a daphnid's eye at the
#: thesis' 1200 dpi target is many times larger, so nothing real is lost.
MIN_BOX_PIXELS = 3.0


def trait_label(trait) -> str:
    """Operator-facing label of a trait, including the bounding box."""
    return LABELS.get(trait, str(trait))


#: How a landmark is named in an exported data file. The corpus speaks the study's
#: anatomy; a training config speaks short names.
EXPORT_NAMES = {
    HEAD: "head",
    TAIL: "tail",
    LEFT: "sideL",
    RIGHT: "sideR",
}

#: The two intervals a four-keypoint system can measure.
INTERVAL_LENGTH = (LENGTH, HEAD, TAIL)
INTERVAL_WIDTH = (WIDTH, LEFT, RIGHT)


def export_name(landmark) -> str:
    """The short name of a landmark in an exported label file."""
    return EXPORT_NAMES.get(landmark, str(landmark).replace(" ", "_"))


@dataclass(frozen=True)
class KeypointSystem:
    """One keypoint vocabulary: which points exist, which pairs are *connected*,
    and which pairs measure what.

    ``edges`` is the **skeleton**: the pairs of keypoints that are anatomically
    joined. It is not decoration - it is what a pose model's connectivity means, it
    is what a viewer draws, and it is what makes a partly-labelled animal
    interpretable (two points that belong together are visibly one bone). It
    defaults to the measured intervals, because for a two- or four-point system the
    only connections are the ones being measured; a richer system names its own.
    """

    key: str                # 'head_tail'
    label: str              # 'head -> tail'
    keypoints: tuple        # the points to click, in click order
    intervals: tuple        # ((trait, first, second), ...) the measured distances
    hint: str               # what those points are, in anatomy
    edges: tuple = ()       # ((first, second), ...) the skeleton, click-order pairs

    @property
    def skeleton(self) -> tuple:
        """The connected keypoint pairs - ``edges`` when given, else the intervals."""
        if self.edges:
            return tuple((first, second) for first, second in self.edges)
        return tuple((first, second) for _trait, first, second in self.intervals)

    @property
    def traits(self) -> tuple:
        """The traits this system can measure, in UI order."""
        return tuple(trait for trait, _, _ in self.intervals)

    def uses(self, landmark) -> bool:
        """True when ``landmark`` belongs to this system."""
        return landmark in self.keypoints

    def points_for(self, trait) -> tuple:
        """``(first, second)`` landmarks of one trait, or the first interval."""
        for name, first, second in self.intervals:
            if name == trait:
                return first, second
        if not self.intervals:
            return ()
        return self.intervals[0][1], self.intervals[0][2]

    def trait_of(self, landmark):
        """The trait a landmark helps to measure, or ``None``."""
        for trait, first, second in self.intervals:
            if landmark in (first, second):
                return trait
        return None
#: The keypoint systems a dataset may be built for. Two points is enough for a
#: length model; four also gives the width, which is what the study measures today.
KEYPOINT_SYSTEMS = {
    "head_tail": KeypointSystem(
        key="head_tail",
        label="head -> tail (2 keypoints)",
        keypoints=(HEAD, TAIL),
        intervals=(INTERVAL_LENGTH,),
        hint="anterior margin of the compound eye (head) -> base of the apical "
             "spine (tail)",
        edges=((HEAD, TAIL),),
    ),
    "head_tail_sides": KeypointSystem(
        key="head_tail_sides",
        label="head, tail, sideL, sideR (4 keypoints)",
        keypoints=(HEAD, TAIL, LEFT, RIGHT),
        intervals=(INTERVAL_LENGTH, INTERVAL_WIDTH),
        hint="head -> tail for the length, left -> right across the carapace "
             "for the width",
        edges=((HEAD, TAIL), (LEFT, RIGHT)),
    ),
    "none": KeypointSystem(
        key="none",
        label="no keypoints",
        keypoints=(),
        intervals=(),
        hint="nothing to click but the specimen itself",
        edges=(),
    ),
}

#: The keypoint system used when an old dataset does not name one.
DEFAULT_SYSTEM = "head_tail_sides"


@dataclass(frozen=True)
class AnnotationSchema:
    """One model target: which keypoints (if any) and whether boxes are labelled.

    A dataset names exactly one of these, and everything downstream follows it: the
    canvas offers these points, the labels record them, the export writes them.
    """

    key: str                # 'pose_full'
    label: str              # 'Pose - head, tail & sides (4 points)'
    system: str             # key into KEYPOINT_SYSTEMS
    boxes: bool             # are bounding boxes part of the label?
    summary: str            # one sentence for the operator

    @property
    def keypoint_system(self) -> KeypointSystem:
        """The keypoint vocabulary of this schema."""
        return KEYPOINT_SYSTEMS.get(self.system, KEYPOINT_SYSTEMS["none"])

    @property
    def keypoints(self) -> tuple:
        """The points to click, in click order (empty for detection only)."""
        return self.keypoint_system.keypoints

    @property
    def intervals(self) -> tuple:
        """``((trait, first, second), ...)`` the keypoints measure."""
        return self.keypoint_system.intervals

    @property
    def edges(self) -> tuple:
        """The **skeleton**: the keypoint pairs that are anatomically connected."""
        return self.keypoint_system.skeleton

    def export_edges(self) -> tuple:
        """The skeleton with the short names a training config expects.

        A pose config wants the bones as pairs of keypoint names, so a corpus
        cannot be consumed without them (``SR-DATASET-023``). The names are the same
        export names the keypoints use, so ``('head', 'tail')`` here means exactly
        what it means in ``labels.json``.
        """
        return tuple((export_name(first), export_name(second))
                     for first, second in self.edges)

    @property
    def traits(self) -> tuple:
        """The traits this schema can measure, in UI order."""
        return self.keypoint_system.traits

    @property
    def uses_keypoints(self) -> bool:
        """True when the schema labels keypoints at all."""
        return bool(self.keypoints)

    @property
    def is_detection_only(self) -> bool:
        """True when a label is nothing but a bounding box."""
        return self.boxes and not self.uses_keypoints

    @property
    def exports_boxes(self) -> bool:
        """True when the exported dataset must carry bounding boxes."""
        return bool(self.boxes)

    def label_columns(self) -> tuple:
        """The columns one label row carries: keypoints and/or the box."""
        columns = []
        if self.uses_keypoints:
            columns.append("keypoints")
        if self.exports_boxes:
            columns.append("bbox")
        return tuple(columns)

    def describe(self) -> str:
        """One-line summary for a dataset row or a status line."""
        points = ("keypoints: " + ", ".join(self.keypoints)
                  if self.uses_keypoints else "no keypoints")
        parts = [points]
        if self.boxes:
            parts.append("bounding box")
        return f"{self.label}  ({'; '.join(parts)})"
#: The model targets a dataset may be built for, in the order the UI offers them.
SCHEMAS = (
    AnnotationSchema(
        key="pose_full",
        label="Pose - head, tail & sides (4 points)",
        system="head_tail_sides",
        boxes=False,
        summary="Four keypoints per specimen: the length and the width.",
    ),
    AnnotationSchema(
        key="pose_head_tail",
        label="Pose - head & tail (2 points)",
        system="head_tail",
        boxes=False,
        summary="Two keypoints per specimen: the body length only.",
    ),
    AnnotationSchema(
        key="pose_full_boxes",
        label="Pose (4 points) + bounding box",
        system="head_tail_sides",
        boxes=True,
        summary="Four keypoints and the box around each specimen.",
    ),
    AnnotationSchema(
        key="boxes",
        label="Bounding boxes only",
        system="none",
        boxes=True,
        summary="One box per specimen - the detector's target, no keypoints.",
    ),
)

#: The schema a new dataset starts with: today's vocabulary, so nothing changes
#: for a study that already measures head -> spine base and the sides.
DEFAULT_SCHEMA = "pose_full"


def get(key) -> AnnotationSchema:
    """The schema for a dataset's stored key (never ``None``)."""
    if key:
        wanted = str(key).strip()
        for schema in SCHEMAS:
            if schema.key == wanted:
                return schema
    return default()


def default() -> AnnotationSchema:
    """The default schema (``pose_full``)."""
    for schema in SCHEMAS:
        if schema.key == DEFAULT_SCHEMA:
            return schema
    return SCHEMAS[0]


def keys() -> tuple:
    """Every schema key, in UI order."""
    return tuple(schema.key for schema in SCHEMAS)


def options() -> tuple:
    """``(key, label)`` pairs for a model/schema dropdown."""
    return tuple((schema.key, schema.label) for schema in SCHEMAS)


def intervals(system=DEFAULT_SYSTEM) -> dict:
    """``{trait: (first, second)}`` of a keypoint system - the click order per trait."""
    system = KEYPOINT_SYSTEMS.get(system, KEYPOINT_SYSTEMS[DEFAULT_SYSTEM])
    return {trait: (first, second) for trait, first, second in system.intervals}


def landmark_choices(schema) -> tuple:
    """``(name, label)`` pairs for a landmark selector, in click order."""
    return tuple((name, f"{name}  ({trait_label(LANDMARK_TRAIT.get(name, LENGTH))})")
                 for name in schema.keypoints)


def all_traits() -> tuple:
    """Every trait the corpus can store (the study's, plus the box)."""
    return tuple(TRAITS) + (BOX_TRAIT,)