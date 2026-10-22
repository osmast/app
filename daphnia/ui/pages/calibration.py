"""Calibrate step - one photo, one press, one result, one stored record.

This rig calibrates daily: a phone clamped to the eyepiece, 1 mm grid paper at the
specimen plane, the paper's circular field stop inside the frame. The panel is shaped
for that act, in the order of the path (`SR-OPTICS-024`):

1. **Optical chain** - the preset name the scale belongs to. One name is one optical
   chain (:func:`daphnia.ids.new_setup_key`), so the name *is* the identity; the chain
   key, the calibration in force for it and how many calibrations this device has stored
   are shown live, because that is what storing will replace. The panel is arranged the
   way a **device** and its **calibrations** both work - load one, or start a new one: at
   the head of the panel sit the preset's two switches (*Load...* makes an already-stored
   device's newest scale the one in force, *New* names a fresh chain), and below them the
   calibration's own pair does the same for the scale (*Load...* picks one of this
   device's stored calibrations and puts it in force with its proof photo, *New* clears
   the measurement to take a fresh one). Changing optics is a choice, not a
   re-measurement (`SR-OPTICS-006`).
2. **Target photo** - the frame the scale is measured from (the live frame, or any
   photo file), how big one printed square is, and **two clearly separate routes to one
   scale**: the automatic lattice measurement over the whole field, then - below a rule -
   the hand-clicked span as the fallback. The automatic one states its period, the bands it
   rests on, the **drift** across the field, the field stop it was measured inside and a
   verdict. Both routes only ever set **one scale for the whole frame**; neither corrects
   the lens, which is why the distortion is a property of the chain in **panel 1** and not a
   result of the measurement here. Everything measured is *drawn on the photo*
   (`SR-OPTICS-026`), so the evidence can be seen instead of trusted.
3. **Scale in force** - the resulting pixels per micrometre, its uncertainty **with the
   route it came from**, and what that means for a real measurement. A scale that is
   **already known** can be typed in instead, and then it is stored as the operator's
   claim, with the accuracy they state for it - never as a measurement. *Store this
   scale* writes a new dated calibration *against the preset from panel 1*;
   measurements already taken keep the one they were taken under (`SR-OPTICS-011`).

Two wordings are load-bearing and must not drift back (`SR-OPTICS-032`,
`SR-OPTICS-036`): the thing measured across the field is the **drift** (panel 2), and the
*lens distortion* is **not corrected** (panel 1) - it is never "0 %" and never "measured",
because no model exists and nothing rewrites a frame. Panel 1 is where that belongs: it is
a property of the optical chain, and it is **optional** - it only matters off-centre.
Nothing here is ever a disabled button: an action that cannot proceed says what is
missing. A greyed-out control swallows the click, which is how "Detect 1 mm grid" came
to look like it did nothing at all.
"""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QWidget,
)

from daphnia import paths, settings, units
from daphnia.db import connection, repository
from daphnia.ids import new_setup_key
from daphnia.streaming import shared_service
from daphnia.ui.calibration_results import (
    CalibrationResultsDialog,
    DistortionResultsDialog,
)
from daphnia.ui.help import CalibrationHelpDialog
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import ImageCanvas, Kpi, StreamView, ToolSpec
from daphnia.vision.calibration import (
    calibration_age_hours,
    is_stale,
    px_per_um,
    relative_uncertainty_pct,
)
from daphnia.vision.grid import GRID_PITCH_UM, PAPER_PITCH_TOLERANCE_PCT
from daphnia.vision.lattice import LatticeNotFound, measure as measure_lattice
from daphnia.vision.points import PointsNotFound, measure_points, reference_points


#: The default reference target. There is deliberately one *kind* of target - printed
#: grid paper - so there is nothing to select wrongly; how big one square of it is
#: lives in panel 2, because finer paper is the honest way to calibrate finer optics
#: (see :mod:`daphnia.vision.grid`).
TARGET_TYPE = "calibrated_grid_1mm"

#: The printed **pattern** the scale is measured from. The rig's own sheet is grid paper,
#: whose scale is the spacing of its printed *lines*; a printed dot target
#: (`assets/targets/`) is the point-based alternative, and the two need different
#: detectors (`daphnia.vision.lattice` reads lines, `daphnia.vision.points` reads dots).
#: The choice lives beside the *Printed square* field because both describe the sheet in
#: front of the lens - and because a detector that guessed wrong would report "no grid"
#: for a perfectly good photo.
PATTERN_LINES = "lines"
PATTERN_DOTS = "dots"
PATTERN_CHECKERBOARD = "checkerboard"
PATTERN_LABELS = (("Line grid (paper)", PATTERN_LINES),
                  ("Dot grid (printed target)", PATTERN_DOTS),
                  ("Checkerboard (printed target)", PATTERN_CHECKERBOARD))

#: How each pattern is named in a sentence (the *state* line and the manual prompt). Kept
#: beside the selector so the wording and the choices cannot drift apart.
PATTERN_WORDS = {PATTERN_LINES: "lines", PATTERN_DOTS: "dots",
                 PATTERN_CHECKERBOARD: "cells"}

#: The name a preset gets when the operator does not give one. Named rather than
#: invented: an unnamed preset still needs a chain key, and "preset" says so.
DEFAULT_PRESET_NAME = "preset"

#: Where a scale came from. A hand-entered scale is a **claim** the panel must never
#: present as a measurement, so the source travels with the number and is stored.
MANUAL_SOURCE = "hand-entered"

#: The two routes that *measure* the scale. They are named once because two places read
#: them: the arithmetic (`_pitch`) and the record (`_save_profile` writes the route into
#: the proof row's notes).
SOURCE_SPAN = "clicked span"
SOURCE_GRID = "detected grid"

#: The verdict panel 2 gives the measurement the operator just made. Four states, and
#: every one of them is reachable: nothing asked for yet, a lattice measured, a lattice
#: measured whose field moves more than the print can explain, and a refusal. Each one
#: says what it *means* and what to do - a verdict the operator has to decode is a
#: verdict nobody acts on.
VERDICT_NONE = "NOT MEASURED - press 'Measure grid' on a photo of the grid paper"
VERDICT_MEASURED = ("MEASURED - the scale holds across the whole field: the bands agree "
                    "to within the print's own tolerance")
VERDICT_WEAK = ("WEAK - the scale is right where it was measured, and less right towards "
                "the edge. Keep the specimen centred; see panel 1 for the distortion that "
                "is not corrected")
VERDICT_REFUSED = ("REFUSED - no lattice in this frame.  Fill the field with the paper, "
                   "or use a frame with the ring out of view")

#: The header's `STATE` fact. ``measuring`` is deliberately absent: the measurement runs
#: synchronously on the GUI thread, so a state that says "measuring" could never be
#: painted - and a state nobody can see is a lie in a bar that is read at a glance.
STATE_READY = "ready"
STATE_NO_LATTICE = "no lattice"
STATE_STORED = "stored"


def grid_label(pitch_um: float) -> str:
    """The reference target in words, for the square size the operator set."""
    return f"{pitch_um / 1000.0:g} mm grid ({pitch_um:g} um)"


def target_type(pitch_um: float) -> str:
    """The stored target kind: the 1 mm grid keeps its historical name."""
    return (TARGET_TYPE if abs(pitch_um - GRID_PITCH_UM) < 1e-9
            else f"calibrated_grid_{pitch_um:g}um")


#: An example length, so the accuracy is shown in micrometres and not only as a
#: percentage nobody can picture. Roughly the animals this project measures.
EXAMPLE_LENGTH_UM = 160.0

#: A recalibration that changes the scale by more than this factor either way is
#: almost always the wrong optics being saved under this preset name.
SCALE_JUMP_FACTOR = 2.0

#: Every calibration records these. The mount locks them, so they are constants rather
#: than fields: the preset name is the identity (`SR-OPTICS-004`).
CAMERA_ZOOM_RATIO = 1.0
FOCUS_MODE = "infinity_locked"


def scale_jump_is_suspicious(new_scale, previous_scale) -> bool:
    """True when ``new_scale`` differs from the previous one by more than 2x."""
    if not previous_scale or previous_scale <= 0 or new_scale <= 0:
        return False
    ratio = new_scale / previous_scale
    return ratio > SCALE_JUMP_FACTOR or ratio < 1.0 / SCALE_JUMP_FACTOR


class Calibration(ShellPage):
    """Preset -> reference -> scale, and the proof photo behind any earlier one.

    Its panel states a ``PANEL_WIDTH`` of its own rather than the shared
    ``CONTROLS_WIDTH``: the result block, the verdict sentence and the chain's distortion
    note all wrap in a 420 px panel, and a measured 500 px (the ``Dataset``/``Bench``
    precedent) unwraps about five lines - which is the difference between *Store this
    scale* being one small scroll below the fold and being on screen.
    """

    PANEL_WIDTH = 500

    def __init__(self, parent=None):
        # No width of its own for the *canvas*: the panel below is what grew. The content
        # asks for ~1131 px of height against an ~823 px viewport at a 900 px window.
        super().__init__(ui_file='calibration.ui', controls_width=self.PANEL_WIDTH,
                         parent=parent)

        self.service = shared_service()
        self.stream = self.add_visual(StreamView(self.service))
        self.canvas = self.add_visual(
            ImageCanvas(max_points=2, point_names=("first line", "last line")),
            stretch=3)
        self.canvas.points_changed.connect(self._on_points_changed)
        # The controls that act on the photo live *on* the photo: the zoom and pan
        # tools, full screen, and the readout - which is what keeps the screen usable
        # when the panel beside it is hidden.
        self.workspace = self.canvas_workspace(self.canvas)
        # The two actions of the daily act, in the window's menus: measuring the lattice,
        # then storing the scale it produced. They are **also** buttons on their panels -
        # `X-N13` says an action is a menu item, `SR-OPTICS-024` says this step's
        # conclusion lives on its panel - and this screen resolves that deliberately as
        # "both" (the one-line exception in `memory-bank/systemPatterns.md`). Both call
        # one method, so there is no second copy of the action.
        self.workspace.set_group("Calibration")
        self.workspace.add_tool(
            ToolSpec("detect", "Measure grid",
                     "Find the printed grid in the target photo - lines or dots - and "
                     "measure its period."),
            self.measure_grid)
        self.workspace.add_tool(
            ToolSpec("save_profile", "Store this scale",
                     "Store the scale - with its uncertainty and the route it came from - "
                     "as the calibration in force for the chosen chain."),
            self._save_profile)
        # The results panel: what the detector found, drawn on the photo, for either printed
        # pattern, with the binarisation level adjustable. It measures nothing of its own -
        # it previews this screen's own estimators.
        self.workspace.add_tool(
            ToolSpec("results", "Results...",
                     "Inspect what the detector found on this photo - the reference points, "
                     "the lattice and the field stop - for either printed pattern, and adjust "
                     "the binarisation level."),
            self._show_results)
        self.workspace.add_tool(
            ToolSpec("correction", "Distortion probe...",
                     "Run discorpy's calibration flow on this photo and show every figure it "
                     "makes, before and after the correction. Nothing is stored."),
            self._show_correction)

        # A choice belongs beside its panel (D8/D9): the target photo *is* the live frame
        # or a file, so those two buttons sit on panel 2, and the page declares no *File*
        # or *Source* entry of its own. Connecting the camera stays where the application
        # puts it - the one shared stream, in the *Source* menu.
        self.btn_use_frame.clicked.connect(self._capture_frame)
        self.btn_open_photo.clicked.connect(self._load_reference_image)
        self.btn_measure_grid.clicked.connect(self.measure_grid)
        self.btn_store_scale.clicked.connect(self._save_profile)
        # Panel 1's two switches: *Load...* changes which stored device is in force,
        # *New* starts a fresh chain. They sit at the top of the panel and the store at
        # its foot, so the daily act reads top-to-bottom: pick or start the device, then
        # measure, then store.
        self.btn_load_preset.clicked.connect(self._load_preset)
        self.btn_new_preset.clicked.connect(self._new_preset)

        self._frame = None          # the target photo being measured
        self._grid = None           # a detected lattice, when there is one
        self._points = None         # the detected reference points last measured, (N, 2)
        self._photo_name = None     # what that photo is called, for the header fact
        self._refused = None        # why the last measurement found no lattice
        self._stored = False        # the scale on screen has just been stored
        self._pending_label = None  # the *New calibration...* note, until it is stored

        # Panel 1: the preset name decides which calibration is valid - it *is* the
        # optical chain - so the chain key and the calibration in force are recomputed
        # whenever the name changes.
        self.profile_name.textChanged.connect(self._refresh_optics)
        # The calibration line is per device, so it follows the name too: renaming to
        # another rig shows *that* rig's stored calibrations, not the previous one's.
        self.profile_name.textChanged.connect(self._refresh_calibrations)

        # Panel 2: the target photo (its two buttons, wired above), *which printed pattern*
        # it holds, the size of one printed cell, and how the scale is measured from them.
        # Changing the square re-scales the detection already on screen - it never
        # re-detects (`SR-OPTICS-030`). Changing the *pattern* is different: a dot
        # measurement is not a line measurement, so the detection is dropped, not rescaled.
        self.spin_grid_mm.valueChanged.connect(self._on_grid_size_changed)
        self.spin_squares.valueChanged.connect(self._refresh_reference)
        # The pattern selector is created here rather than in the .ui: it sits in the
        # *Printed square* row's own form, one row above the square it describes. It is
        # added by name because ``load_ui`` exposes named **widgets** as attributes, not
        # named layouts - the form has to be found, not read off ``self``.
        self.combo_pattern = QComboBox()
        for label, value in PATTERN_LABELS:
            self.combo_pattern.addItem(label, value)
        self.combo_pattern.setToolTip(
            "The printed pattern in the target photo. 'Line grid' is ordinary graph paper "
            "(a square of printed *lines*); 'Dot grid' is a printed target from "
            "assets/targets (a field of round *dots*); 'Checkerboard' is the same kind of "
            "printed target, whose scale is the side of one square *cell*. "
            "The patterns need different detectors, so the choice must match the sheet in "
            "front of the lens.")
        form = self.reference_group.findChild(QFormLayout, "grid_size_form")
        form.addRow("Printed pattern", self.combo_pattern)
        self.combo_pattern.currentIndexChanged.connect(self._on_pattern_changed)

        # Binarisation: off = Otsu decides (the default), on = the operator's own cut. It sits
        # in the same form because it describes the same sheet, and it is **inert** on the line
        # grid - that estimator reads a grey profile and never binarises - so the control is
        # disabled there and says why, rather than accepting a number nothing reads.
        self.check_threshold = QCheckBox("set by hand")
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 255.0)
        self.spin_threshold.setDecimals(1)
        self.spin_threshold.setValue(128.0)
        self.spin_threshold.setSuffix(" grey")
        threshold_row = QWidget()
        threshold_layout = QHBoxLayout(threshold_row)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(self.check_threshold)
        threshold_layout.addWidget(self.spin_threshold)
        form.addRow("Binarisation", threshold_row)
        self.check_threshold.toggled.connect(self._on_threshold_toggled)
        self.spin_threshold.valueChanged.connect(self._refresh_reference)
        self._sync_threshold_controls()

        # The results panel, on the panel itself and not only in the window menu: after
        # 'Measure grid' the operator wants to *see* what was found, and a menu entry is
        # where that goes to be missed. It is inserted directly under the button it inspects.
        self.btn_results = QPushButton("Results...")
        self.btn_results.setToolTip(
            "Draw what the detector found on this photo - the reference points it measured "
            "from, the lattice, and the field stop - for either printed pattern, with the "
            "binarisation level adjustable.")
        self.btn_results.clicked.connect(self._show_results)
        # The distortion probe: discorpy's whole flow, and the figures it makes before and
        # after the correction. Its own panel, because it is a *probe* - slow, and nothing it
        # finds is stored yet (`SR-OPTICS-032`/`033`).
        self.btn_correction = QPushButton("Correction...")
        self.btn_correction.setToolTip(
            "Run discorpy's distortion-calibration flow on this photo and show every figure "
            "it makes: the normalised and binarised frames, the cross points, the grouped "
            "lines, the residual BEFORE the model, the corrected lines, the residual AFTER, "
            "the unwarped frame and the difference. Nothing is stored.")
        self.btn_correction.clicked.connect(self._show_correction)
        reference_layout = self.reference_group.layout()
        reference_layout.insertWidget(
            reference_layout.indexOf(self.btn_measure_grid) + 1, self.btn_results)
        reference_layout.insertWidget(
            reference_layout.indexOf(self.btn_measure_grid) + 2, self.btn_correction)

        # Panel 3: a scale the operator already knows, off by default. Its two fields
        # are enabled only while the box is ticked - a number nobody typed is not a
        # scale, and the panel says so rather than saving it.
        self.check_manual_scale.toggled.connect(self._on_manual_toggled)
        self.spin_manual_px_per_um.valueChanged.connect(self._refresh_reference)
        self.spin_manual_accuracy.valueChanged.connect(self._refresh_reference)
        self._on_manual_toggled(self.check_manual_scale.isChecked())

        # Panel 1 again: a calibration is loaded or created the same way a preset is.
        # *Load...* picks one of this device's stored calibrations and puts it in force
        # (with its proof photo); *New* clears the measurement to start a fresh one. The
        # loading is a `load_calibration` call - the dialog lives in the handler, so a
        # test never meets a modal.
        self.btn_load_calibration.clicked.connect(self._load_calibration)
        self.btn_new_calibration.clicked.connect(self._new_calibration)

        # The camera state belongs to panel 2 (it decides which photos exist), so it
        # is reported there rather than mixed into the scale panel.
        for signal in (self.service.status, self.service.opened, self.service.closed):
            signal.connect(self._refresh_reference)
        # A photo the canvas could not read is refused in the panel too, by name and
        # reason - see ImageCanvas.set_source.
        self.canvas.source_failed.connect(self.status_label.setText)

        self._refresh_all()

    def on_shown(self):
        self._refresh_all()

    def _refresh_all(self):
        """Re-read all three panels, in the order of the path."""
        self._refresh_optics()
        self._refresh_reference()
        self._refresh_calibrations()

    # -- panel 1: the preset the scale belongs to --------------------------
    def preset_name(self) -> str:
        """The name the operator gave this preset - what the scale belongs to."""
        return self.profile_name.text().strip() or DEFAULT_PRESET_NAME

    def setup_key(self) -> str:
        """The optical chain this preset names - what a calibration belongs to.

        The name **is** the identity (:func:`daphnia.ids.new_setup_key`): asking for
        the hardware a second time (microscope, objective, camera, zoom, focus) is how
        a description drifts out of step with the optics it claims to name, and one
        name keeping one scale history is what makes "which calibration is this animal
        in?" answerable at a glance.
        """
        return new_setup_key(self.preset_name())

    def _refresh_optics(self, *_):
        key = self.setup_key()
        self.setup_key_label.setText(key)
        with connection() as conn:
            active = repository.active_profile(conn)
        self.in_force_label.setText(self._in_force_text(active, key))
        # Which chain the scale in force belongs to: the value names the case ("other
        # chain"), and the key itself is one hover away - the panel is a list of values
        # a 500 px column can hold, not a paragraph.
        self.in_force_label.setToolTip(
            f"The calibration in force was measured on {active.setup_key}."
            if active is not None else "")

    def _in_force_text(self, active, key) -> str:
        """What is in force now, as **list values** - one of three cases, not a sentence.

        The empty name box is **not** "a different preset": nothing has been named yet,
        so the panel reports the scale *any* chain would use and stops claiming a chain
        that was never typed. `RE-VERIFY` is appended only where the quoted scale is the
        one this name would use - a stale row belonging to another chain is not this
        chain's problem, and the footer already flags age for the database as a whole.
        """
        if active is None:
            return "none yet"
        summary = (f"{active.px_per_um * 1000.0:.2f} px/mm "
                   f"± {active.uncertainty_pct:.2f} %")
        stale = is_stale(active.calibrated_at, settings.CALIBRATION_MAX_AGE_HOURS)
        if not self.profile_name.text().strip():
            return (f"{summary} · any chain · {(active.calibrated_at or '')[:16]}"
                    + (" · RE-VERIFY" if stale else ""))
        if active.setup_key == key:
            age = calibration_age_hours(active.calibrated_at)
            when = f" · {age:.1f} h ago" if age is not None else ""
            return f"{summary}{when}" + (" · RE-VERIFY" if stale else "")
        return f"{summary} · other chain"

    # -- switching the device in force -------------------------------------
    def stored_preset_names(self) -> list:
        """Every chain in the store, newest first, one name each - the *Load...* list.

        Distinct ``setup_key``s, so a chain calibrated five times is **one** entry,
        labelled with the name its newest calibration carries. Names and nothing else:
        the panel asks the operator to remember which name is which device, rather than
        spreading a scale, a date and a route across columns a 500 px panel cannot hold.
        """
        with connection() as conn:
            rows = repository.calibrations(conn)
        names, seen = [], set()
        for row in rows:                       # newest first
            if row["setup_key"] not in seen:
                seen.add(row["setup_key"])
                names.append(row["profile_name"])
        return names

    def _load_preset(self):
        """Pick a stored device **by name** and switch to it - panel 1's *Load...*.

        The chooser is a list of **names only** (`SR-OPTICS-006`): a preset name *is*
        its chain, so one string is the whole label and there is no scale or date column
        to read past. The switching itself is :meth:`load_preset`, which a caller (or a
        test) drives without a dialog - a modal chooser inside the method a test calls is
        how a suite hangs.
        """
        names = self.stored_preset_names()
        if not names:
            self.status_label.setText(
                "No preset to load yet.  Name this rig in 'Preset name', then put the "
                "grid paper in front of the camera and press 'Store this scale' to make "
                "the first one - 'Load...' lists what has been stored.")
            return
        name, accepted = QInputDialog.getItem(
            self, "Load a preset", "Device:", names, 0, False)
        if accepted and name:
            self.load_preset(name)

    def load_preset(self, name) -> bool:
        """Switch the device in force to the chain called ``name``.

        The scale that takes over is that chain's **newest** calibration (`SR-OPTICS-006`):
        loading never measures and never stores, it only chooses which stored scale the
        whole application measures with - so it is guarded exactly like storing, by the 2x
        check (`SR-OPTICS-008`), because switching a microscope for a phone *is* the 5x
        move that check exists to make deliberate. The name field follows, so the panel
        states which device is in force and the footer re-reads the scale.
        """
        name = (name or "").strip()
        if not name:
            return False
        key = new_setup_key(name)
        with connection() as conn:
            rows = repository.calibrations_for_setup(conn, key)
        if not rows:
            self.status_label.setText(
                f"No stored calibration for '{name}' yet.  Name this rig and press "
                "'Store this scale' to make one.")
            return False
        row = rows[0]                          # newest first
        if not self._confirm_scale(row["px_per_um"]):
            self.status_label.setText(
                f"Not switched to '{name}' - check the scale it would put in force, "
                "then try again.")
            return False
        self.profile_name.setText(name)        # the field states the device in force
        with connection() as conn:
            repository.activate_profile(conn, row["profile_id"])
        self._stored = False
        self._pending_label = None             # a switch drops any note aimed at a new one
        self.notify_workflow()                 # the gates and the footer re-read the scale
        self._refresh_all()
        self.status_label.setText(
            f"Switched to '{name}': {row['px_per_um'] * 1000.0:.1f} px/mm "
            f"± {row['uncertainty_pct']:.2f} % is now in force.  Every measurement from "
            "now on carries that.")
        return True

    def _new_preset(self):
        """Name a fresh device in a popup and start it - panel 1's *New*.

        The name is asked for **here** rather than typed into the panel afterwards, so
        that *New* answers the same shape of question as *Load...*: a popup, one string,
        nothing else. The starting itself is :meth:`new_preset`, which a caller (or a
        test) drives without a dialog - a modal chooser inside the method a test calls is
        how a suite hangs.
        """
        name, accepted = QInputDialog.getText(
            self, "New preset", "Name this device:", text="")
        if accepted:
            self.new_preset(name)

    def new_preset(self, name):
        """Start a fresh preset called ``name``: clear the name and the measurement.

        A new device begins by having no measured scale, so the operator names it and
        stores its scale. What is in force is deliberately left alone: naming a device is
        not switching to it (`SR-OPTICS-031`), and *Load...* is the one action that
        switches. An empty name is allowed - the panel's own field then carries it.
        """
        self._grid = None
        self._refused = None
        self._stored = False
        self._pending_label = None             # a *New* preset is not the one the note was for
        self.canvas.clear_points()             # the measurement and its overlays go too
        self.profile_name.setText((name or "").strip())
        self._refresh_all()
        if self.profile_name.text():
            self.status_label.setText(
                f"New preset '{self.profile_name.text()}'.  Measure the grid and press "
                "'Store this scale'.")
        else:
            self.status_label.setText(
                "New preset.  Name this rig in 'Preset name', then measure the grid and "
                "press 'Store this scale'.")

    # -- the calibration list, and its own Load... / New -------------------
    def stored_calibrations(self) -> list:
        """This **device**'s calibrations, newest first - what *Load...* offers here.

        One entry per calibration of the **current** chain (`setup_key`): a preset is
        re-verified daily, so it has many dated scales while the *preset* list has one
        name. The label is the operator's own note for the row when they gave one, and
        the date otherwise: the two ways a person tells two calibrations apart. The
        identity (`profile_id`) is what :meth:`_load_calibration` hands on.
        """
        with connection() as conn:
            rows = repository.calibrations_for_setup(conn, self.setup_key())
            active = repository.active_profile(conn)
        active_id = active.profile_id if active is not None else None
        entries = []
        for row in rows:
            when = row["last_verified_at"] or "unknown"
            name = row["label"] or when[:16]
            entries.append({
                "profile_id": row["profile_id"],
                "name": name,
                "in_force": row["profile_id"] == active_id,
                "label": (f"{name} · {row['px_per_um'] * 1000.0:.1f} px/mm"
                          + (" · (in force)" if row["profile_id"] == active_id else "")),
            })
        return entries

    def _refresh_calibrations(self, *_):
        """Panel 1's calibrations value: how many this device has, and the one in force."""
        entries = self.stored_calibrations()
        if not entries:
            self.history_label.setText("none stored")
            return
        in_force = next((entry["name"] for entry in entries if entry["in_force"]), None)
        # The count, then the loaded one if it is this device's - a value, not a sentence.
        self.history_label.setText(
            f"{len(entries)} stored"
            + (f" · in force: {in_force}" if in_force else " · none in force"))

    def _new_calibration(self):
        """Name a fresh calibration of **this** device in a popup - panel 1's *New*.

        The label is a note the operator will recognise later ("after refocus"), so it is
        asked for the way a preset's name is; the starting itself is :meth:`new_calibration`,
        which a caller (or a test) drives without a dialog.
        """
        label, accepted = QInputDialog.getText(
            self, "New calibration", f"Label for the next calibration of "
            f"'{self.preset_name()}':", text="")
        if accepted:
            self.new_calibration(label)

    def new_calibration(self, label):
        """Start a fresh calibration of this device, carrying ``label`` to the store.

        The preset name is left as it is - a new calibration belongs to the same chain and
        is created by measuring the grid again and pressing *Store this scale*. The label
        is kept until that store writes it onto the new row; nothing in force changes
        until then.
        """
        self._pending_label = (label or "").strip() or None
        self._grid = None
        self._refused = None
        self._stored = False
        self.canvas.clear_points()             # the measurement and its overlays go too
        self._refresh_all()
        note = f" ('{self._pending_label}')" if self._pending_label else ""
        self.status_label.setText(
            f"New calibration{note} for '{self.preset_name()}'.  Put the grid paper in "
            "front of the camera, press 'Measure grid', then 'Store this scale'.")

    def _load_calibration(self):
        """Pick one of this device's calibrations and load it - panel 1's *Load...*.

        The popup lists the **current** preset's calibrations by date and scale, so the
        list is short and its entries are the device's own history. The loading itself is
        :meth:`load_calibration`, which a caller (or a test) drives without a dialog - a
        modal chooser inside the method a test calls is how a suite hangs.
        """
        entries = self.stored_calibrations()
        if not entries:
            self.status_label.setText(
                f"No stored calibration for '{self.preset_name()}' yet.  Name the device "
                "above, measure the grid and press 'Store this scale' - 'Load...' lists "
                "what this device has already stored.")
            return
        labels = [entry["label"] for entry in entries]
        label, accepted = QInputDialog.getItem(
            self, "Load a calibration", f"Calibration of '{self.preset_name()}':",
            labels, 0, False)
        if accepted and label:
            self.load_calibration(entries[labels.index(label)]["profile_id"])

    def load_calibration(self, profile_id) -> bool:
        """Load one of this device's calibrations: it becomes the scale in force.

        A calibration is *chosen* here, not merely looked at: the picked row is activated
        (`repository.activate_profile`) and its **proof photo** is put on the canvas, so
        the scale the whole application measures with is the one whose evidence can be
        seen. Like *Load...* for a preset, it is guarded by the 2x check
        (`SR-OPTICS-008`) and stores nothing. A calibration with no proof photo
        (hand-entered) still activates; the panel only says there is no image to show.
        """
        with connection() as conn:
            profile = repository.profile_by_id(conn, profile_id)
            record = repository.proof_record(conn, profile_id)
        if profile is None:
            self.status_label.setText(
                f"There is no calibration {profile_id} in this database.")
            return False
        if not self._confirm_scale(profile.px_per_um):
            self.status_label.setText(
                f"Not loaded - check the scale {profile_id} would put in force, then "
                "try again.")
            return False

        with connection() as conn:
            repository.activate_profile(conn, profile_id)
        self._stored = False
        self._pending_label = None             # a load drops any note aimed at a new one
        self.notify_workflow()                 # the gates and the footer re-read the scale
        summary = (f"{profile.px_per_um * 1000.0:.1f} px/mm "
                   f"\u00b1 {profile.uncertainty_pct:.2f} %")
        when = profile.calibrated_at or "unknown"
        if record is None:
            self._refresh_all()
            self.status_label.setText(
                f"Loaded {profile_id}: {summary}, verified {when} - it is now the "
                "calibration in force.  It was entered by hand, so it has no proof photo "
                "to show.")
            return True

        self._set_reference(record["raw_image_path"],
                            Path(record["raw_image_path"]).name)
        self._refresh_all()
        if self._frame is None:
            self.status_label.setText(
                f"{self.canvas.refusal or 'The proof photo could not be loaded'}.  "
                f"{profile_id} is now the calibration in force ({summary}); only its "
                "photo could not be shown.")
            return True
        self.status_label.setText(
            f"Loaded {profile_id}: {summary}, verified {when}.  It is now the calibration "
            "in force, and this is the proof photo it was measured from.")
        return True

    # -- panel 2: the reference photo, and the grid in it -------------------
    def grid_pitch_um(self) -> float:
        """How wide one printed square of the reference grid is, in micrometres."""
        return self.spin_grid_mm.value() * 1000.0

    def span_um(self) -> float:
        """How far a hand-clicked span reaches, in micrometres."""
        return self.spin_squares.value() * self.grid_pitch_um()

    def _on_grid_size_changed(self, *_):
        """A different square size re-scales the same detection - no re-measuring.

        The detector measures a *period in pixels*; the square's true size is what
        turns that into a scale. So the estimate is kept and re-labelled with the new
        pitch instead of being thrown away, which is what makes "try the finer paper"
        a one-second experiment rather than another photo.
        """
        if self._grid is not None:
            self._grid = replace(self._grid, pitch_um=self.grid_pitch_um())
        self._refresh_reference()

    def target_pattern(self) -> str:
        """Which printed pattern the target photo holds: ``PATTERN_LINES`` or ``DOTS``.

        The default is the line grid, because that is the sheet this rig has always used;
        the selector only ever reports what the operator chose.
        """
        return self.combo_pattern.currentData() or PATTERN_LINES

    def binarisation_threshold(self):
        """The manual ink/paper cut in the frame's own grey units, or ``None`` (Otsu).

        ``None`` - the default - means the detector decides. A number is the operator's own
        cut, honoured only where it does something: the **line** estimator reads a grey
        profile and never binarises, so this is ``None`` for it.
        """
        if self.target_pattern() == PATTERN_LINES or not self.check_threshold.isChecked():
            return None
        return float(self.spin_threshold.value())

    def _on_threshold_toggled(self, checked):
        self._sync_threshold_controls()
        self._refresh_reference()

    def _sync_threshold_controls(self):
        """Offer the manual cut only where it acts, and say why when it cannot."""
        usable = self.target_pattern() != PATTERN_LINES
        self.check_threshold.setEnabled(usable)
        self.spin_threshold.setEnabled(usable and self.check_threshold.isChecked())
        self.check_threshold.setToolTip(
            "Off: the detector chooses the ink/paper cut with Otsu's method. On: it cuts at "
            "the level beside it - the escape hatch for a photo where Otsu's split lands "
            "wrongly. It does not cure the eyepiece's vignette: the paper's brightness varies "
            "across the field, so no single level is right everywhere."
            if usable else
            "The line-grid estimator reads a grey profile and never binarises, so there is no "
            "cut to set here. Choose a dot grid or a checkerboard pattern to use it.")

    def _measure_target(self, frame, pattern, threshold=None):
        """Run the estimator a printed pattern means - the one place that mapping lives.

        `measure_grid` and the results panel both come through here, so the screen and the
        panel cannot disagree about which detector a pattern selects.
        """
        if pattern == PATTERN_LINES:
            return measure_lattice(frame, pitch_um=self.grid_pitch_um())
        return measure_points(frame, pitch_um=self.grid_pitch_um(), pattern=pattern,
                              threshold=threshold)

    def _target_points(self, frame, pattern, threshold=None):
        """The reference points to **draw** for a pattern, or ``None`` for the line target.

        A line lattice is drawn from the estimate it produced; only the point targets have
        points of their own to ring. The points come from the vision layer, not from a copy
        here, so the drawing is of what the measurement actually used.
        """
        if pattern == PATTERN_LINES:
            return None
        return reference_points(frame, pattern=pattern, threshold=threshold)

    def _show_results(self):
        """The results panel: what was detected, of any pattern, with the cut adjustable."""
        CalibrationResultsDialog(
            frame=self._frame, patterns=PATTERN_LABELS, pitch_um=self.grid_pitch_um(),
            pattern=self.target_pattern(), threshold=self.binarisation_threshold(),
            measure=self._measure_target, points=self._target_points, parent=self).exec()

    def _show_correction(self):
        """The distortion probe: discorpy's whole flow, and the figures before and after."""
        DistortionResultsDialog(frame=self._frame, parent=self).exec()

    def _on_pattern_changed(self, *_):
        """A different printed pattern invalidates the detection - it is not a rescale.

        The period of a *dot* lattice is not the period of a *line* family, so keeping the
        previous number and re-labelling it (which is what a square-size change does) would
        file one target's measurement under another's. The verdict goes back to *not
        measured* and the operator presses 'Measure grid' again.
        """
        self._grid = None
        self._points = None
        self._refused = None
        self._sync_threshold_controls()
        self._refresh_reference()

    def _manual_prompt(self) -> str:
        word = PATTERN_WORDS.get(self.target_pattern(), "lines")
        mark = word[:-1] if word.endswith("s") else word
        return (f"Click the first and the last {mark} of "
                f"{self.spin_squares.value()} cell(s) = {self.span_um():g} um.")

    def _span_px(self) -> float:
        (x1, y1), (x2, y2) = self.canvas.points[0], self.canvas.points[1]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _on_points_changed(self, points):
        """A hand-clicked span takes precedence over a detected grid."""
        self._refresh_reference()

    def _refresh_reference(self, *_):
        """Panel 2, once per change: the photo's state, the result block, the header.

        One refresh method per panel, and this is panel 2's: it is called on a new photo,
        on a new square size, when hand marks change and when the camera changes state -
        never in a loop. The numbers it prints all come from :attr:`_grid` (the
        measurement) and the canvas draws the evidence; the page keeps no copy of either.
        """
        self.stream_hint_label.setText(
            f"Live feed: {self.service.state_label()} - 'Use live frame' needs it live; "
            "a photo file needs nothing.")
        self._refresh_grid_state()
        self._refresh_result()
        self.points_label.setText(
            self._manual_prompt() if len(self.canvas.points) < 2
            else f"Span clicked: {self._span_px():.1f} px.")
        self._refresh_scale()
        # Two facts that are *not* a measurement go to the **window header**, and the
        # lattice with the paper's circle is drawn *on the photo*, which is where evidence
        # belongs: nothing is written over the specimen (2026-10-18).
        self.workspace.set_header([
            Kpi("photo", self._photo_fact(), label="Photo"),
            Kpi("state", self._state_fact(), label="State"),
        ])

    def _photo_fact(self) -> str:
        """``name 1280x720`` - which frame the numbers on screen are about."""
        if self._frame is None:
            return "none"
        height, width = self._frame.shape[:2]
        return f"{self._photo_name or 'photo'} {width}x{height}"

    def _state_fact(self) -> str:
        """One word for where the daily act stands (see the ``STATE_*`` constants)."""
        if self._stored:
            return STATE_STORED
        if self._frame is not None and self._grid is None:
            return STATE_NO_LATTICE
        return STATE_READY

    def _refresh_grid_state(self):
        """Panel 2's one state sentence: which printed square the numbers rest on."""
        pitch_mm = self.grid_pitch_um() / 1000.0
        if self._grid is None:
            self.grid_label.setText(
                f"No lattice measured yet - one printed cell is set to {pitch_mm:g} mm.  "
                f"Press 'Measure grid' on a photo of the printed target "
                f"({PATTERN_WORDS.get(self.target_pattern(), 'lines')}).")
            self.canvas.set_lattice(None)
            self.canvas.set_field_stop(None)
            self.canvas.set_reference_points(None)
        else:
            grid = self._grid
            self.grid_label.setText(
                f"Measured on this photo: one printed cell = {pitch_mm:g} mm.")
            # Draw it: the overlay is the evidence that detection actually ran, the ringed
            # points are *what it measured from*, and the circle says *where* in the field
            # the scale was measured (`SR-OPTICS-026`).
            self.canvas.set_lattice(grid.period_px, grid.angle_deg)
            self.canvas.set_reference_points(self._points)
            stop = grid.field_stop
            self.canvas.set_field_stop(*(stop.centre_x, stop.centre_y, stop.radius)
                                       if stop else (None, None, None))
    def _refresh_result(self):
        """Panel 2's result block: what was measured, where, and what it is worth.

        `LatticeEstimate.warnings()` is read for its **fields**, never printed as a
        paragraph: the period goes to one line, the drift to its own, the field stop to a
        third, the judgement to the verdict - and the long caveats to a tooltip. That is
        the point of the block (D5): one fact, one home, and the visible home is never the
        truncated one.
        """
        grid = self._grid
        if grid is None:
            self.label_result.setText("Period: -")
            self.label_drift.setText("Drift: -")
            self.label_field_stop.setText("Field stop: -")
            if self._refused:
                # The verdict is short and the *reason* is long: the estimator names what
                # it found (one line family, too few bands), and that is where a search for
                # the reason should end - one hover away, not in a paragraph nobody can
                # find again (`SR-OPTICS-017`).
                self._set_verdict(VERDICT_REFUSED, "warning")
                self.label_verdict.setToolTip(self._refused)
            else:
                self._set_verdict(VERDICT_NONE)
                self.label_verdict.setToolTip(
                    "No measurement was asked for yet. 'Measure grid' reads the printed "
                    "lattice's period from this photo; 'Store this scale' records it.")
            return

        self.label_result.setText(
            f"Period: {grid.period_px:.3f} px/cell · "
            f"{grid.reading_count} {grid.reading_label} · "
            f"tilt {grid.angle_deg:+.1f}°")
        self.label_result.setToolTip(
            "One printed cell, in pixels: the median of the field's bands, and how many "
            "local readings it rests on. The tilt is the lattice's own rotation in this "
            "frame - rotating does not change the scale.")
        self.label_drift.setText(
            f"Drift: ± {grid.spread_pct:.1f} % across the field (centre → edge)")
        self.label_drift.setToolTip(self._drift_tooltip(grid))
        self.label_field_stop.setText(
            "Field stop: " + (
                f"paper circle ({grid.field_stop.centre_x:.0f}, "
                f"{grid.field_stop.centre_y:.0f}) r {grid.field_stop.radius:.0f} px · "
                f"{grid.coverage * 100:.0f} % of the frame"
                if grid.field_stop is not None
                else "none in view - the whole frame is paper"))
        self.label_field_stop.setToolTip(
            "The circular paper area the eyepiece's field stop leaves in this frame, and "
            "how much of the frame that is. The scale was measured inside it and nowhere "
            "else: the dark surround is not paper, and a ring inside the field is not the "
            "lattice.")
        if grid.spread_pct > PAPER_PITCH_TOLERANCE_PCT:
            # The one judgement this block makes, and the yardstick is the print's own
            # tolerance (`SR-OPTICS-036`): the drift line above already carries the
            # percentage, so the verdict says *what it means* and what to do - and points at
            # panel 1 for the correction that does not exist, because that is the difference
            # between a scale (which both routes set) and the lens distortion (which neither
            # of them touches).
            self._set_verdict(VERDICT_WEAK, "warning")
            self.label_verdict.setToolTip(
                f"The field's own readings move {grid.spread_pct:.1f} % from the centre of "
                f"the frame to its edge, which is more than the print's own "
                f"{PAPER_PITCH_TOLERANCE_PCT:.1f} %: the scale is exact where it was "
                "measured and less exact further out. Nothing is wrong with the number - "
                "it is a place in the frame, not an uncertainty - so keep the specimen "
                "close to the centre, or fit a distortion correction for this chain "
                "(nothing in this app does that yet, and panel 1 says so).")
        else:
            self._set_verdict(VERDICT_MEASURED, "ok")
            self.label_verdict.setToolTip(
                f"The field's own readings agree with each other to within the print's own "
                f"{PAPER_PITCH_TOLERANCE_PCT:.1f} %, so this one scale is good across the "
                "whole frame and the specimen can sit anywhere in it. The lens distortion "
                "is still not corrected (panel 1) - with a drift this small it does not "
                "change what you measure.")

    def _drift_tooltip(self, grid) -> str:
        """The long form of the drift line, including the estimator's own caveats.

        Two facts the panel must not fold into one: the drift is a **place** in the frame
        changing the scale, and the uncertainty is how well the centre was measured. The
        caveats are quoted from the estimate - they are what it measured on *this* photo -
        and they are worded as drift, never as "lens distortion, measured"
        (`SR-OPTICS-032`): no model exists here and nothing corrects a frame.
        """
        notes = ["The spread of the field's own local readings - the same bands that "
                 "produced the period - so this is the place in the frame changing the "
                 "scale, not an uncertainty of the number above. No correction is applied "
                 "to the frame: panel 1 says the distortion is not corrected."]
        notes.extend(grid.caveats)
        return "  ".join(notes)

    def _set_verdict(self, text, tone=""):
        """One line, one colour - and the colour survives a *change* of verdict.

        A dynamic property is read when the widget is polished, so the tone has to be
        re-polished by hand: without this the label keeps the first verdict's colour
        forever, which is worse than no colour at all.
        """
        self.label_verdict.setText(f"Verdict: {text}")
        self.label_verdict.setProperty("tone", tone)
        self.label_verdict.style().unpolish(self.label_verdict)
        self.label_verdict.style().polish(self.label_verdict)



    # -- panel 3: the scale ------------------------------------------------
    def manual_scale(self):
        """``(px_per_um, stated uncertainty_pct)`` for a typed scale, or ``None``.

        A typed scale is a **claim**, not a measurement: the operator says what the
        scale is and how accurate that is, and the panel repeats both as such. Zero
        (nothing typed) is not a scale, so it is refused here and the panel asks for a
        number instead of saving one.
        """
        if not self.check_manual_scale.isChecked():
            return None
        scale = float(self.spin_manual_px_per_um.value())
        if scale <= 0:
            return None
        return scale, float(self.spin_manual_accuracy.value())

    def _on_manual_toggled(self, checked: bool):
        """The two hand-entry fields only mean anything while the box is ticked."""
        self.spin_manual_px_per_um.setEnabled(checked)
        self.spin_manual_accuracy.setEnabled(checked)
        self._refresh_reference()

    def _pitch(self):
        """``(px_per_um, distance_px, uncertainty_pct, source)``, or ``None``.

        A typed scale wins over everything: it is the operator saying they already know
        the number. Then hand-clicked marks take precedence when they exist, so the
        manual span stays available as a fallback; otherwise the detected grid is the
        source.
        """
        manual = self.manual_scale()
        if manual is not None:
            scale, uncertainty = manual
            # One micrometre spans `scale` pixels: the proof row every calibration
            # carries, stated for the hand-entered value (there is no photo to file).
            return (scale, scale, uncertainty, MANUAL_SOURCE)
        points = self.canvas.points
        if len(points) >= 2:
            span = self.span_um()
            if span <= 0:
                return None
            distance_px = self._span_px()
            return (px_per_um(distance_px, span), distance_px,
                    relative_uncertainty_pct(distance_px), SOURCE_SPAN)
        if self._grid is not None:
            return (self._grid.px_per_um(), self._grid.mean_period_px,
                    self._grid.uncertainty_pct(), SOURCE_GRID)
        return None

    def _refresh_scale(self):
        """Panel 3: the number, the route it came from, and what it means.

        The route is part of the number (D6): the grid route's uncertainty **includes** the
        print's own tolerance, the span route's does not, and a hand-entered one is a claim
        rather than a measurement. One sentence says all three, and the advice lives in the
        tooltip - so nothing on the panel recommends the route that leaves half a percent
        out, which is what "weak: click a longer span" used to do on a 2.54 % grid.
        """
        result = self._pitch()
        if result is None:
            self.px_per_um_label.setText("S = - px/mm")
            self.uncertainty_label.setText("Scale uncertainty: -")
            self.uncertainty_label.setToolTip("")
            self.example_label.setText(self._no_scale_text())
            return

        scale, _distance_px, uncertainty, source = result
        self.px_per_um_label.setText(
            f"S = {scale * 1000.0:.1f} px/mm   ({scale:.6f} px/um)"
            + ("   - hand-entered" if source == MANUAL_SOURCE else ""))
        self.uncertainty_label.setText(
            f"Scale uncertainty: {self._uncertainty_text(source, uncertainty)}")
        self.uncertainty_label.setToolTip(self._uncertainty_advice(source, uncertainty))
        example = ("A measured length would read: "
                   f"{units.format_with_uncertainty(EXAMPLE_LENGTH_UM, uncertainty)}")
        if scale > 0:
            # The *whole* budget of that example, not only the scale's term: the click
            # and the pixel grid are decided by this very scale, so the panel that
            # chooses the scale is the right place to see what they cost
            # (`units.total_uncertainty_um`, `docs/accuracy.md` §1).
            example += ("   "
                        + units.format_total_uncertainty(EXAMPLE_LENGTH_UM, scale,
                                                         uncertainty))
        self.example_label.setText(example)

    def _uncertainty_text(self, source, uncertainty) -> str:
        """The uncertainty **with its route** - the three do not mean the same thing.

        `SR-OPTICS-018` puts the print's own tolerance in the number, and only the grid
        route actually does: the span is a measurement of the *photograph*, so naming what
        it leaves out is the difference between 0.29 % and 0.5 %+. The hand route has no
        fit at all, so it says whose claim the number is (`SR-OPTICS-029`).
        """
        if source == MANUAL_SOURCE:
            return (f"± {uncertainty:.2f} %  - you stated it, so it is a claim, not a "
                    "measurement.")
        if source == SOURCE_SPAN:
            return (f"± {uncertainty:.2f} % from the {SOURCE_SPAN}  - the print's own "
                    f"{PAPER_PITCH_TOLERANCE_PCT:.2f} % is not included.")
        statistical = (self._grid.statistical_uncertainty_pct()
                       if self._grid is not None else None)
        fit = (f"the bands' own fit ({statistical:.2f} %) plus "
               if statistical is not None else "")
        return (f"± {uncertainty:.2f} % from the {SOURCE_GRID}  - {fit}the print's own "
                f"{PAPER_PITCH_TOLERANCE_PCT:.2f} %.")

    def _uncertainty_advice(self, source, uncertainty) -> str:
        """What would actually improve this number - never the route that omits the print.

        On the grid route the print's tolerance is already *in* the number, so pointing at
        the span would trade an honest 2.54 % for a 0.29 % that leaves half a percent out
        (D6). The useful move there is to make the bands agree with each other - paper flat,
        in focus, filling the field - because that is where the rest of the number lives.
        """
        weak = uncertainty > settings.CALIBRATION_WEAK_UNCERTAINTY_PCT
        mark = (f"  Weak by the {settings.CALIBRATION_WEAK_UNCERTAINTY_PCT:.1f} % mark."
                if weak else "")
        if source == MANUAL_SOURCE:
            return ("It carries only what you stated for it. Measure the scale against a "
                    "stage micrometer, type that in, and it becomes a measurement.")
        if source == SOURCE_SPAN:
            return ("A longer span - more squares between the two clicks - is the direct "
                    "way to improve it." + mark)
        return ("The fit over the field is as good as this photo can give, and the print's "
                "own tolerance cannot be measured away. Put the paper flat, in focus and "
                "filling the field so the bands agree." + mark)

    def _no_scale_text(self) -> str:
        """Why there is no scale yet - naming which of the three routes is missing."""
        if self.check_manual_scale.isChecked():
            return ("Type the scale in px/um above - 0 is not a scale - or untick 'I know "
                    "the scale already' to measure the grid instead.")
        return ("Measure the grid, click a span, or set the scale by hand, and the "
                "scale appears here.")

    # -- actions -----------------------------------------------------------
    def _capture_frame(self):
        """Freeze the live frame as the target photo."""
        frame = self.service.current_frame()
        if frame is None:
            self.status_label.setText(self._no_live_frame())
            return
        self._set_reference(frame, "Live frame")

    def _no_live_frame(self) -> str:
        """Why there is no live frame, in the words of the connection itself."""
        return (f"No live frame.  {self.service.state_label()}  Press Connect in the "
                "bar below, or use 'Open photo...' instead.")

    def _load_reference_image(self):
        """Ask for a photo, then use it. The choosing and the loading are separate."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a photo of the grid",
            str(paths.CALIBRATION_IMAGES_DIR),
            "Images (*.png *.jpg *.jpeg *.heic *.heif *.tif *.tiff *.bmp *.webp)"
            ";;All files (*)")
        if path:
            self.use_photo(path)

    def use_photo(self, path):
        """Calibrate from this photo file - any format the machine can read.

        Dialog-free on purpose, so the loading half can be driven directly.
        """
        self._set_reference(path, Path(path).name)

    def _set_reference(self, source, name):
        """Install a target photo, discarding any measurement of an earlier one.

        The source may be the live frame or a file the operator picked; the canvas reads
        either. With ``frame=None`` the canvas refused it, and then there is **no** target
        at all - not the previous one, whose scale would then be measured against a photo
        nobody can see. The name travels with it, because the header fact ("which frame are
        these numbers about?") has to survive the status line being overwritten.
        """
        frame = self.canvas.set_source(source)   # also clears the lattice and the circle
        self._frame = frame
        self._photo_name = name if frame is not None else None
        self._grid = None
        self._points = None
        self._refused = None
        self._stored = False
        if frame is None:
            self._refresh_reference()
            return
        height, width = frame.shape[:2]
        self._refresh_reference()
        self.status_label.setText(
            f"{name} ({width} x {height}).  Press 'Measure grid'.")

    def measure_grid(self):
        """Measure the printed lattice in the target photo - one press, one result.

        Never disabled and never silent: with nothing to measure it says so rather than
        swallowing the click. A lattice that was found puts its numbers in panel 2's block;
        a **refusal** puts the short verdict there and the estimator's own reason (one line
        family, too few bands) in that verdict's tooltip, because a paragraph of diagnosis
        in the status line is where a reason goes to be lost.
        """
        if self._frame is None:
            self.status_label.setText(
                "No photo to measure yet.  Press 'Use live frame' if the camera is live, "
                "or 'Open photo...'.")
            return

        self.canvas.clear_points()            # the detection replaces manual marks
        self._stored = False
        # The printed pattern decides which estimator runs - a line family and a dot lattice
        # are different measurements, and neither detector can read the other's target
        # (`daphnia.vision.lattice` vs `daphnia.vision.points`). Both refuse with a reason,
        # which the verdict block quotes.
        pattern, threshold = self.target_pattern(), self.binarisation_threshold()
        try:
            self._grid = self._measure_target(self._frame, pattern, threshold)
        except (LatticeNotFound, PointsNotFound) as error:
            self._grid = None
            self._points = None
            self._refused = str(error)
            self._refresh_reference()
            self.status_label.setText("No grid found - see the verdict in panel 2.")
            return

        # The rings drawn on the photo are *what the period was measured from*: the same
        # detector on the same frame, pattern and cut - `reference_points` shares
        # `_detect_dots`/`_detect_cells` with `measure_points`, so this is deterministic and
        # not a second opinion about what is ink (`mlContext.md` law 1).
        self._points = self._target_points(self._frame, pattern, threshold)
        self._refused = None
        self._refresh_reference()
        self.status_label.setText(
            f"Grid measured: {self._grid.period_px:.3f} px per printed cell, from "
            f"{self._grid.reading_count} {self._grid.reading_label}.  "
            "Panel 3 shows the scale it gives.")

    def _show_help(self):
        CalibrationHelpDialog(self).exec()

    # -- storing: the scale goes back onto the preset of panel 1 -----------
    def _save_profile(self):
        """Store the scale in force for the chain in panel 1 - the step's conclusion.

        Reachable twice on purpose (`X-N13` vs `SR-OPTICS-024`): the panel 3 button and the
        *Calibration* menu's *Store this scale* call this one method, so the two entries can
        never drift apart.
        """
        result = self._pitch()
        if result is None:
            self.status_label.setText(
                "Nothing to store yet.  Put the grid paper in front of the camera and "
                "press 'Measure grid', click a span by hand, or set the scale by hand in "
                "panel 3.")
            return
        scale, distance_px, uncertainty_pct, source = result

        if not self._confirm_scale(scale):
            self.status_label.setText(
                "Calibration cancelled - check the scale, and the preset it is being "
                "saved under, in panel 1.")
            return

        setup_key = self.setup_key()
        measured = source != MANUAL_SOURCE

        # Keep the proof image for a **measured** scale: one file per calibration run,
        # holding the photo that actually produced the number rather than a
        # re-rendering of it. A hand-entered scale has no such photo, and saying so is
        # the point - there is nothing to file as evidence for it.
        image_path = None
        if measured:
            paths.CALIBRATION_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = paths.CALIBRATION_IMAGES_DIR / f"{setup_key}_{stamp}.png"
            cv2.imwrite(str(image_path), self._frame)

        with connection() as conn:
            calibration_id = repository.register_calibration(
                conn,
                setup_key=setup_key,
                profile_name=self.preset_name(),
                # The operator's own note for this calibration, typed at *New...*; the
                # hardware columns stay honest about not knowing anything more
                # (`SR-OPTICS-004`).
                label=self._pending_label,
                microscope_model="unknown",
                objective_magnification="unknown",
                camera_device="unknown",
                px_per_um=scale,
                uncertainty_pct=uncertainty_pct,
                camera_zoom_ratio=CAMERA_ZOOM_RATIO,
                focus_mode=FOCUS_MODE,
            )
            if measured:
                repository.insert_calibration_record(
                    conn,
                    profile_id=calibration_id,
                    target_type=target_type(self.grid_pitch_um()),
                    raw_image_path=image_path,
                    detected_distance_px=distance_px,
                    actual_distance_um=(self.span_um() if source == SOURCE_SPAN
                                        else self.grid_pitch_um()),
                    notes=f"{grid_label(self.grid_pitch_um())}, measured from the "
                          f"{source}",
                )

        # The order of these two lines is the header's `STATE`: "stored" is a fact about
        # the act that just happened, so it is set before the panels re-read themselves.
        self._stored = True
        self._pending_label = None      # the note has been written onto the new row
        self.notify_workflow()
        self._refresh_all()
        self.status_label.setText(
            self._saved_text(setup_key, scale, uncertainty_pct, source))

    @staticmethod
    def _saved_text(setup_key, scale, uncertainty_pct, source) -> str:
        """What was stored, and what it is worth - a claim is not a measurement."""
        claim = ("  It is hand-entered, so that accuracy is your claim rather than a "
                 "measured one." if source == MANUAL_SOURCE else "")
        return (f"Stored for {setup_key}: S = {scale * 1000.0:.1f} px/mm, "
                f"± {uncertainty_pct:.2f} %.  Every measurement taken from now on "
                f"carries that; older ones keep theirs.{claim}")

    def _confirm_scale(self, scale) -> bool:
        """Guard against a wrong scale: warn when it jumps by more than 2x.

        Storing a scale that is off by a factor scales every later measurement by that
        factor, without changing a single recorded value - and **switching** a preset
        does the same across the whole application - so this asks before either, naming
        both values (`SR-OPTICS-008`). It serves *Store this scale* and *Load...* alike,
        so its wording is about the change and not about which button caused it.
        """
        with connection() as conn:
            previous = repository.active_profile(conn)
        if not scale_jump_is_suspicious(scale, previous.px_per_um if previous else None):
            return True
        answer = QMessageBox.warning(
            self, "Scale changed a lot",
            f"The scale in force would change from {previous.px_per_um * 1000.0:.1f} "
            f"px/mm to {scale * 1000.0:.1f} px/mm - more than 2x.\n\nThat is what "
            "switching the objective, the phone or the mounting looks like - and also "
            "what a hand-entered value out by a factor looks like. Make it "
            "deliberate.\n\nChange it anyway?",
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
        return answer == QMessageBox.Ok

