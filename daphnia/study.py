"""Study modes: what kind of project a study is, and what that implies for the UI.

A study is not one thing. Some run multi-well plates, some count single specimens;
some measure animals, some plants; some follow the same specimens over days, some
measure once. The three axes below are stored with the study, and every screen asks
the resulting :class:`StudySpec` what to show - so the Setup step configures the
rest of the app instead of each screen hard-coding assumptions.
"""

from dataclasses import dataclass

from daphnia import protocols

#: Where the specimens live: multi-well plates, or nothing but a counter.
TARGET_MODES = {
    "plates": "Multi-well plates",
    "individuals": "Single specimens (counter)",
}

#: What is measured - this decides the anatomy that gets annotated.
ORGANISM_KINDS = {
    "animal": "Animal",
    "plant": "Plant",
}

#: Whether the same specimens are followed over time.
TIMECOURSES = {
    "longitudinal": "Repeated over time",
    "single": "Measured once",
}

DEFAULT_TARGET_MODE = "individuals"
DEFAULT_ORGANISM_KIND = "animal"
DEFAULT_TIMECOURSE = "longitudinal"

#: The two landmarks per organism kind: what the operator clicks, and what the
#: distance between them means.
ANATOMY = {
    "animal": {
        "keypoints": ("head", "spine base"),
        "endpoint": "body length",
        "specimen": "individual",
    },
    "plant": {
        "keypoints": ("base", "tip"),
        "endpoint": "shoot length",
        "specimen": "plant",
    },
}


#: The traits measured on every specimen, in UI order. Each trait is one distance
#: between two landmarks: 'length' (head -> spine base) and 'width' (the sides).
TRAITS = ("length", "width")
TRAIT_LABELS = {"length": "Length", "width": "Width"}

#: Landmarks for the width trait - a pair of points across the body, left then right.
SIDE_POINTS = ("left side", "right side")

#: The anatomical landmarks clicked per trait, in click order. ``length`` is the
#: project's fixed body-length endpoint - the anterior margin of the compound eye
#: (the "head") to the base of the apical spine; ``width`` spans the carapace.
TRAIT_LANDMARKS = {
    "length": ANATOMY[DEFAULT_ORGANISM_KIND]["keypoints"],   # ('head', 'spine base')
    "width": SIDE_POINTS,
}

#: Every landmark, in click order: length's pair, then width's pair.
LANDMARKS = tuple(point for trait in TRAITS for point in TRAIT_LANDMARKS[trait])

#: Which trait each landmark belongs to.
LANDMARK_TRAIT = {point: trait for trait in TRAITS
                  for point in TRAIT_LANDMARKS[trait]}


@dataclass(frozen=True)
class StudySpec:
    """A study's configuration, as the screens need to read it."""

    study_id: str
    name: str
    organism: str
    organism_kind: str
    target_mode: str
    timecourse: str
    interval_hours: float
    destination: str
    protocol_key: str = protocols.DEFAULT_PROTOCOL

    @classmethod
    def from_row(cls, row) -> "StudySpec":
        """Build the spec from a ``studies`` row."""
        # ``protocol_key`` was added later; rows from an older database may lack it.
        keys = row.keys()
        return cls(
            study_id=row["study_id"],
            name=row["name"],
            organism=row["organism"],
            organism_kind=row["organism_kind"],
            target_mode=row["target_mode"],
            timecourse=row["timecourse"],
            interval_hours=row["measurement_interval_hours"],
            destination=row["destination"],
            protocol_key=(row["protocol_key"] if "protocol_key" in keys
                          else protocols.DEFAULT_PROTOCOL),
        )

    # -- what the screens ask ---------------------------------------------
    @property
    def uses_plates(self) -> bool:
        """True when the run iterates wells of a plate."""
        return self.target_mode == "plates"

    @property
    def is_longitudinal(self) -> bool:
        """True when the same specimens are measured again over time."""
        return self.timecourse == "longitudinal"

    @property
    def protocol(self) -> protocols.Protocol:
        """The acquisition protocol this study is run with."""
        return protocols.get(self.protocol_key)

    @property
    def anatomy(self) -> dict:
        """Anatomy dictionary of this organism kind."""
        return ANATOMY.get(self.organism_kind, ANATOMY[DEFAULT_ORGANISM_KIND])

    @property
    def keypoints(self) -> tuple:
        """The two landmarks to click, e.g. ``('head', 'spine base')``."""
        return self.anatomy["keypoints"]

    @property
    def endpoint(self) -> str:
        """What the measured distance means, e.g. ``'body length'``."""
        return self.anatomy["endpoint"]

    @property
    def specimen(self) -> str:
        """What one target is called, e.g. ``'individual'``."""
        return self.anatomy["specimen"]

    def points_for(self, trait: str) -> tuple:
        """Landmark names to click for ``trait`` on this study's organism."""
        return SIDE_POINTS if trait == "width" else self.keypoints

    def endpoint_for(self, trait: str) -> str:
        """What the measured distance means for ``trait``, e.g. ``'body width'``."""
        return "body width" if trait == "width" else self.endpoint

    def describe(self) -> str:
        """One-line summary of the mode, for the overview screen."""
        return " / ".join((
            ORGANISM_KINDS.get(self.organism_kind, self.organism_kind),
            TARGET_MODES.get(self.target_mode, self.target_mode),
            TIMECOURSES.get(self.timecourse, self.timecourse),
        ))

def stage_of(row) -> str:
    """Where a project stands, in the workflow's own vocabulary.

    ``row`` carries the counts from
    :func:`daphnia.db.queries.list_studies_with_progress` (``plates``, ``targets``,
    ``frames``, ``selected``, ``measured``), so a half-finished run reports exactly
    what is still missing.
    """
    if row["measured"]:
        return f"{row['measured']} measurement(s) annotated - ready for analytics"
    if row["selected"]:
        return f"{row['selected']} frame(s) selected - annotate them next"
    if row["targets"]:
        return f"{row['targets']} target(s) captured - select the best frames next"
    if row["plates"]:
        return "plates configured - ready to capture"
    return "just created - configure plates or start capturing"
