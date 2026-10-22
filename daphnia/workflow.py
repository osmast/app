"""Step-based workflow: the non-negotiable order of an experiment.

  1 Overview -> 2 Setup -> 3 Calibrate -> 4 Acquire -> 5 Select -> 6 Annotate
  -> 7 Analytics

Each step declares what it *produces* (a fact later steps can rely on) and what it
*requires* right now. A step can only be **entered** when all of that holds in the
database, so the order cannot be bypassed from the UI:

* ``study``       a study exists - it decides **where the images are saved**.
* ``connection``  the phone is connected. It is established once, on the Setup step,
                  and required again for Calibrate and Acquire, so nothing is
                  captured or calibrated off a stream that is not there.
* ``calibration`` an active optical preset exists - an image without a scale can
                  never be measured.
* ``observation`` at least one specimen was captured.
* ``selection``   exactly one best frame per well / individual was chosen.
* ``measurement`` at least one keypoint annotation was stored.

Selecting deliberately does **not** require the connection: once the frames are on
disk the phone can be put away and the work continues offline.
"""

from dataclasses import dataclass, field

from daphnia.db import queries, repository


@dataclass(frozen=True)
class Step:
    """One step of the run."""

    route: str
    label: str
    requires: tuple = field(default_factory=tuple)   # facts needed to enter
    produces: tuple = field(default_factory=tuple)   # facts created here


#: The workflow, in order.
STEPS = (
    Step("dashboard", "Overview"),
    Step("experiment", "Setup", produces=("study",)),
    Step("calibration", "Calibrate", requires=("connection",),
         produces=("calibration",)),
    Step("acquire", "Acquire", requires=("connection",), produces=("observation",)),
    Step("select", "Select", produces=("selection",)),
    Step("annotate", "Annotate", produces=("measurement",)),
    Step("visualize", "Analytics"),
)

#: Why a fact might be missing, phrased for the operator.
REASONS = {
    "study": "Set up the study first - it decides where the images are saved.",
    "connection": "Connect the phone on the Setup step first.",
    "calibration": "Calibrate the optics first - an image without a scale cannot be measured.",
    "observation": "Capture at least one specimen first.",
    "selection": "Select the best frame of each well before annotating.",
    "measurement": "Annotate at least one selected frame first.",
}

FACTS = ("study", "connection", "calibration", "observation", "selection",
         "measurement")


def facts(conn, runtime=None) -> dict:
    """Which non-negotiable facts currently hold for the **active project**.

    The run-progress facts are counted per study, never globally: a brand-new
    project must not inherit captures, selections or measurements from another
    project. The optical preset is hardware and is shared, so ``calibration`` stays
    global.

    ``runtime`` carries the facts that live outside the database - today only the
    phone connection, which the operator establishes once on the Setup step.
    """
    study = queries.active_study(conn)
    progress = queries.study_progress(conn, study["study_id"]) if study else {}
    state = {
        "study": study is not None,
        "connection": False,
        "calibration": repository.active_profile(conn) is not None,
        "observation": progress.get("targets", 0) > 0,
        "selection": progress.get("selected", 0) > 0,
        "measurement": progress.get("measured", 0) > 0,
    }
    if runtime:
        state.update(runtime)
    return state


def index_of(route: str) -> int:
    """Step index of a route name."""
    for position, step in enumerate(STEPS):
        if step.route == route:
            return position
    raise KeyError(route)


def required(index: int) -> tuple:
    """Facts that must hold before step ``index`` may be entered."""
    needed = []
    for step in STEPS[:index]:
        needed.extend(step.produces)
    needed.extend(STEPS[index].requires)

    unique = []
    for fact in needed:
        if fact not in unique:
            unique.append(fact)
    return tuple(unique)


def missing(conn, index: int, runtime=None) -> tuple:
    """Required facts for step ``index`` that do not hold yet."""
    current = facts(conn, runtime)
    return tuple(fact for fact in required(index) if not current.get(fact))


def can_enter(conn, index: int, runtime=None) -> bool:
    """True when step ``index`` may be entered."""
    return not missing(conn, index, runtime)


def blocking_reason(conn, index: int, runtime=None) -> str:
    """Operator-facing explanation of what blocks step ``index``."""
    gaps = missing(conn, index, runtime)
    return REASONS[gaps[0]] if gaps else ""


def furthest_step(conn, runtime=None) -> int:
    """Index of the furthest step that may be entered right now."""
    index = 0
    for candidate in range(1, len(STEPS)):
        if not can_enter(conn, candidate, runtime):
            break
        index = candidate
    return index
