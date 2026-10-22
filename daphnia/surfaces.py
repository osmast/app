"""Surfaces: the intent-based launch points of the application.

One repository, one package, one store - **several surfaces**. A surface is what
an operator *means to do*, not which screen they open: measure one specimen right
now, build the training corpus, run a governed longitudinal study, or set up the
optics.

Each surface has its own launch point (``python run.py <key>``), its own gate
policy (see :mod:`daphnia.workflow` for the governed one) and its own system
requirements document under ``docs/requirements/SR-<SURFACE>.md``. The split is
deliberate: the project's audiences need different amounts of ceremony, and
forcing all of them through the study pipeline is what produced twelve throwaway
studies in a single afternoon.

This module is deliberately **Qt-free**, so the hub can render it, ``run.py`` can
dispatch on it and the tests can check it without a display.
"""

from dataclasses import dataclass, field

#: A surface that can be used today.
AVAILABLE = "available"

#: A surface that is specified (requirements written) but not built yet.
PLANNED = "planned"

STATUSES = (AVAILABLE, PLANNED)

#: How a status is phrased for an operator.
STATUS_LABELS = {
    AVAILABLE: "Ready to use",
    PLANNED: "Specified - not built yet",
}

#: The launcher itself. Not a surface: it is where surfaces are chosen.
HUB_KEY = "hub"
HUB_LABEL = "Home"


@dataclass(frozen=True)
class Surface:
    """One launch point: what it is for, how it is gated, what to build next."""

    key: str                    # launch argument, e.g. 'bench'
    label: str                  # 'Bench'
    tagline: str                # one short line for the tile
    summary: str                # two sentences for the tile body
    audience: str               # who needs it
    gate: str                   # what must hold before it can act
    status: str                 # AVAILABLE | PLANNED
    requirements: str           # path to its requirement document
    next_steps: tuple = field(default_factory=tuple)   # first things to build

    @property
    def is_available(self) -> bool:
        """True when this surface can be used today."""
        return self.status == AVAILABLE

    @property
    def status_label(self) -> str:
        """Operator-facing status text."""
        return STATUS_LABELS.get(self.status, self.status)

    def launch_point(self) -> str:
        """The command that opens this surface."""
        return f"python run.py {self.key}"


#: The surfaces, in the order the hub offers them: the end goal first, then the
#: audiences that are currently blocked.
SURFACES = (
    Surface(
        key="study",
        label="Study",
        tagline="Run a governed longitudinal study",
        summary=(
            "The full pipeline in order: set up the study, calibrate the optics, "
            "acquire, select the best frame, annotate, analyse. Every gate is "
            "enforced, so the numbers it produces are reproducible and defensible."
        ),
        audience="The thesis, and every reviewer of its results",
        gate="study -> connection -> calibration -> observation -> selection -> measurement",
        status=AVAILABLE,
        requirements="docs/requirements/SR-STUDY.md",
        next_steps=(
            "Declare a plan of timepoints and show what is due today",
            "One-action study creation, with the mode axes behind an Advanced drawer",
            "Treatment assignment on top of plates",
        ),
    ),
    Surface(
        key="study2",
        label="Study 2",
        tagline="Capture a whole plate, well by well",
        summary=(
            "The plate page: a 6-, 24- or 96-well plate drawn well by well, with each "
            "well lighting up as it is imaged. Take a burst on the well under the camera, "
            "move the plate, step to the next well. It is the acquisition half of what "
            "will replace the governed Study pipeline - today it captures, later it will "
            "measure each individual in the well."
        ),
        audience="The plate routine the end goal is about: many wells, one burst each",
        gate="a study + a plate + a well, the live connection and an active calibration",
        status=AVAILABLE,
        requirements="docs/requirements/SR-STUDY2.md",
        next_steps=(
            "Measure a well's individuals: RTMPose on the well's image, or a manual line",
            "Export the plate as a contact sheet image (the frames, well by well)",
            "A plan of timepoints, so the same plate can be walked again tomorrow",
            "Retire the Study surface once this one covers its gates",
        ),
    ),
    Surface(
        key="bench",
        label="Bench",
        tagline="Measure a specimen right now",
        summary=(
            "Open any image (or use the live frame), click two landmarks and the number is "
            "written above the line - name it what you call it, and one box tells which "
            "individual each number is about. No study, no plate and no phone required - the "
            "fastest path to a number, and a result can be attached to a study later instead "
            "of being thrown away."
        ),
        audience="Colleagues who want a value today, without setting up a project",
        gate="an image + a scale (an uncalibrated reading stays in pixels)",
        status=AVAILABLE,
        requirements="docs/requirements/SR-BENCH.md",
        next_steps=(
            "Several specimens per image work already; batch import is next",
            "Attach a whole bench session to a study in one action",
            "Read a saved view back for a further pass over the same image",
        ),
    ),
    Surface(
        key="dataset",
        label="Dataset",
        tagline="Build and label the training corpus",
        summary=(
            "Create a dataset per model - two keypoints, four, and/or bounding "
            "boxes - and label the anatomical points at volume, then export a "
            "machine-readable corpus. Many frames per specimen is the point here, "
            "not a problem."
        ),
        audience="The thesis: it produces the ground truth the model trains on",
        gate="frames exist - labels are optional at any moment",
        status=AVAILABLE,
        requirements="docs/requirements/SR-DATASET.md",
        next_steps=(
            "Pre-labels: accept or correct a model prediction (source = 'model')",
            "Real-time bounding-box prediction, now that boxes can be labelled",
            "Split by specimen and study once a first model exists",
            "Hard-negative and empty-well labelling as a first-class queue item",
        ),
    ),
    Surface(
        key="predict",
        label="Predict",
        tagline="Watch the detector find the animals",
        summary=(
            "Run the trained RTMDet on the live camera or on a video file and see "
            "its boxes in real time: one rectangle per animal, with its score. This "
            "is the Localize step - what a study will use to find a specimen before "
            "measuring it - and the honest way to see whether the detector is good "
            "enough yet."
        ),
        audience="The thesis: it turns 'is the detector good?' into something you watch",
        gate="a trained detector (models/rtmdet_daphnia.pt) - see `make train`",
        status=AVAILABLE,
        requirements="docs/requirements/SR-PREDICT.md",
        next_steps=(
            "Add the keypoint model (RTMPose) on top of the boxes",
            "Offer 'measure these boxes' - the Localize -> Predict hand-off",
            "Record a clip with the detections burned in, for the thesis figures",
        ),
    ),
    Surface(
        key="optics",
        label="Optics",
        tagline="Calibrate the optics",
        summary=(
            "Detect the printed 1 mm grid automatically and store the scale as a "
            "dated calibration: what it was, when it was verified, and how accurate "
            "the reference made it. Works from the stream or from a saved photo."
        ),
        audience="Whoever sets up the microscope, and every surface that needs a scale",
        gate="a reference photo of the 1 mm grid (from the stream or from disk)",
        status=AVAILABLE,
        requirements="docs/requirements/SR-OPTICS.md",
        next_steps=(
            "One active scale per optical setup, not one global preset",
            "Calibration history for a setup, with the proof images",
            "Warn when a preset's scale changes by more than a set factor",
        ),
    ),
)


def get(key) -> "Surface | None":
    """The surface for a launch key, or ``None``."""
    if not key:
        return None
    wanted = str(key).strip().lower()
    for surface in SURFACES:
        if surface.key == wanted:
            return surface
    return None


def keys() -> tuple:
    """Every surface key, in hub order."""
    return tuple(surface.key for surface in SURFACES)


def available() -> tuple:
    """Surfaces that can be used today."""
    return tuple(surface for surface in SURFACES if surface.is_available)


def planned() -> tuple:
    """Surfaces that are specified but not built yet."""
    return tuple(surface for surface in SURFACES if not surface.is_available)

