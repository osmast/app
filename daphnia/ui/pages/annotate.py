"""Annotate step - measure every specimen and trait of each kept frame.

A photo can hold more than one specimen, and each specimen is measured for more
than one trait (length = head -> spine base, width = the two side points). This
carousel walks every kept frame and, within it, lets the operator move between
specimens and traits. Every (frame, specimen, trait) is one measurement row; the
landmarks double as ground truth for the prediction model.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut

from daphnia import units, workflow
from daphnia.db import connection, queries, repository
from daphnia.study import TRAITS, TRAIT_LABELS
from daphnia.ui.model_pass import ModelRunner, primary
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import ImageCanvas, Kpi, ToolSpec
from daphnia.vision.keypoints import Keypoints


class Annotate(ShellPage):
    """Carousel over all kept frames; several specimens and traits per frame."""

    def __init__(self, parent=None):
        super().__init__(ui_file='annotate.ui', parent=parent)

        self.canvas = self.add_visual(
            ImageCanvas(max_points=2, point_names=("head", "spine base")))
        self.canvas.points_changed.connect(self._update_readout)
        self.canvas.points_moved.connect(self._refresh_header)
        # Measuring happens on the image, so the controls that measure do too: the
        # landmarks are placed while looking at the specimen. The panel keeps the
        # *choices* (specimen, trait), the number and the list of what is already
        # measured.
        self.workspace = self.canvas_workspace(self.canvas)
        self.workspace.set_group("Measure")
        self.workspace.add_tool(
            ToolSpec("save", "Save", "Store the two landmarks of this specimen."),
            self._save_current)
        self.workspace.add_tool(
            ToolSpec("clear", "Clear points", "Forget the points clicked on this "
                                              "frame - stored measurements stay."),
            self.canvas.clear_points)
        self.workspace.add_tool(
            ToolSpec("prev", "< Prev", "Previous frame in the queue."),
            lambda: self._step(-1))
        self.workspace.add_tool(
            ToolSpec("next", "Next >", "Next frame in the queue."),
            lambda: self._step(1))
        self.workspace.add_tool(
            ToolSpec("new_specimen", "Next specimen",
                     "Start measuring the next animal in this frame."),
            self._new_specimen)
        self.workspace.add_tool(
            ToolSpec("refresh", "Refresh the list", "Re-read what is stored for this group."),
            self.refresh)
        # The model on this frame: with the switch on, the models in force answer on the
        # frame on screen and the largest animal's keypoints land in the editable draft.
        # Off by default - a tool the operator reaches for.
        model_tool = self.workspace.add_tool(
            ToolSpec("model", "Model",
                     "Predict the keypoints of the animals on this frame with the models in "
                     "force, and put the largest one in the draft for you to correct. Off: "
                     "measure by hand only.",
                     checkable=True, checked=False))
        model_tool.toggled.connect(self._on_model_toggled)
        # The **Select** tool (key `"edit"`), on by default (`SR-DATASET-027`): a click
        # inside a box selects it (never places a new point), so the model's keypoints can
        # be corrected and a wrong box or point removed with `Backspace`. Off is the
        # measuring mode for placing landmarks by hand.
        edit_tool = self.workspace.add_tool(
            ToolSpec("edit", "Select",
                     "The Select tool: a click on a box or a point selects it, a drag moves "
                     "it, Backspace/Delete removes it, and a click never places a new point. "
                     "Off: clicks place the landmarks by hand.",
                     checkable=True, checked=True))
        edit_tool.toggled.connect(self._on_edit_toggled)
        self.canvas.box_edit = True

        # The choices stay beside the panel's combos; the actions above are menu items.
        self.combo_trait.currentIndexChanged.connect(self._on_subject_changed)
        self.spin_specimen.valueChanged.connect(self._on_subject_changed)
        # A frame whose file has gone missing is refused in the panel too. The annotate
        # screen used to hand the path to a second, weaker loader that showed nothing
        # at all when it failed - now it names the file and the reason.
        self.canvas.source_failed.connect(self.status_label.setText)
        # Picking one of the model's animals on the frame (`Edit`).
        self.canvas.prediction_selected.connect(self._on_prediction_selected)
        self.canvas.prediction_removed.connect(self._on_prediction_removed)
        # A dropped point corrects the measurement it came from, if there is one (`SR-DATASET-027`).
        self.canvas.points_edited.connect(self._on_points_edited)
        # `Backspace`/`Delete` remove what is selected wherever the focus happens to be.
        for keys in (QKeySequence(Qt.Key_Backspace), QKeySequence.StandardKey.Delete):
            delete_shortcut = QShortcut(keys, self)
            delete_shortcut.activated.connect(self.canvas.delete_selection)

        self._spec = None
        self._rows = []
        self._loading = False        # guards programmatic canvas updates
        # The models in force, run on the frame on screen (`daphnia/ui/model_pass.py`): the
        # page never chooses a checkpoint - the registry does, written by the *Detector
        # (RTMDet)* / *Pose (RTMPose)* menus.
        self.runner = ModelRunner(self)
        self._model_predictions = ()    # the last answer of the models in force
        self.queue = []
        self.index = 0
        self._build_traits()
        self.refresh()

    # -- subject (specimen & trait) ----------------------------------------
    def _build_traits(self):
        self.combo_trait.clear()
        for trait in TRAITS:
            self.combo_trait.addItem(TRAIT_LABELS[trait], trait)

    def trait(self) -> str:
        """The trait currently being measured."""
        return self.combo_trait.currentData() or "length"

    def specimen(self) -> int:
        """The specimen currently being measured."""
        return int(self.spin_specimen.value())

    def on_shown(self):
        self.refresh()

    # -- queue -------------------------------------------------------------
    def refresh(self):
        with connection() as conn:
            spec = queries.active_study_spec(conn)
            queue = queries.kept_assets_all(conn, spec.study_id) if spec else []
        self._spec = spec
        self.canvas.point_names = list(spec.keypoints) if spec else ["head", "spine base"]
        self.queue = list(queue)
        self.index = min(self.index, max(len(self.queue) - 1, 0))
        self._show_current()

    def _show_current(self, *_):
        annotated = sum(1 for row in self.queue if row["measurements"])
        self.progress_label.setText(
            f"{annotated} / {len(self.queue)} frames measured" if self.queue
            else "queue empty")

        if not self.queue:
            self.canvas.set_source(None)
            self.task_label.setText("No selected frames yet.")
            self.measurement_label.setText("Not measured yet.")
            self.list_measurements.clear()
            self._rows = []
            self.status_label.setText(workflow.REASONS["selection"])
            self._set_enabled(False)
            self._refresh_header()
            return

        self._set_enabled(True)
        asset = self.queue[self.index]
        self.task_label.setText(
            f"{self.index + 1} / {len(self.queue)}   {queries.target_label(asset)}   "
            f"(burst {asset['burst_index'] or 0}, frame {asset['frame_index'] or 0})")
        self.canvas.set_source(asset["file_path"])     # one source, the same way in
        with connection() as conn:
            self._rows = queries.measurements_for_asset(conn, asset["asset_id"])
        self._sync_specimens()
        self._on_subject_changed()
        self.prelabel()              # the models in force answer on the frame just shown

    # -- the model on this frame -------------------------------------------
    def prelabel(self, announce=False):
        """Run the models in force on this frame, and hand the answer to the draft.

        The same rule as the other labelling surfaces (`daphnia/ui/model_pass.py`): a model
        that is *selected* answers on the frame on screen, the largest animal's keypoints
        become the editable pair (move them, then *Save* - the model's guess is never stored
        by itself), and every animal the model found stays drawn in the prediction layer.
        With nothing to report the page's own status line stands, unless the switch was just
        turned on by hand (`announce=True`).
        """
        if not self.workspace.action("model").isChecked() or not self.queue:
            return
        result = self.runner.run(self.canvas.frame, self._um_per_px())
        self.canvas.set_predictions(result.predictions)
        self._model_predictions = result.predictions
        best = primary(result.predictions)
        placed = (best is not None and len(best.points) >= 2
                  and self._stored(self.specimen(), self.trait()) is None)
        if placed:
            self._loading = True
            self.canvas.set_points(list(best.points))
            self._loading = False
        if not len(result) and not announce:
            return
        note = result.note + ("  Fix the predicted points, then Save." if placed else "")
        current = self.status_label.text().strip()
        self.status_label.setText(f"{current}  {note}" if current else note)

    def _on_model_toggled(self, checked):
        """The *Model* switch: predict on this frame, or measure by hand only."""
        if checked:
            self.prelabel(announce=True)
            return
        self._loading = True
        self.canvas.set_predictions([])
        self.canvas.clear_points()
        self._loading = False
        self.status_label.setText("Model off - measuring by hand.")

    def _um_per_px(self):
        """The scale in force in micrometres per pixel, or ``None`` (labels stay px)."""
        with connection() as conn:
            profile = repository.active_profile(conn)
        if profile is None or not profile.px_per_um:
            return None
        return 1.0 / float(profile.px_per_um)

    # -- picking one animal on the frame -----------------------------------
    def _on_edit_toggled(self, checked):
        """The *Edit* mode: click an animal's box to measure *that* one, or measure by hand."""
        self.canvas.box_edit = bool(checked)
        if checked:
            self.status_label.setText(
                "Edit mode: click an animal's box to make it the one being measured, drag a "
                "point to move it, Backspace to remove it.")
        else:
            self.canvas.clear_selection()
            self.status_label.setText("Create mode - measuring by hand.")

    def _on_prediction_selected(self, index):
        """A predicted box was clicked: that animal's points go into the draft to correct."""
        if index < 0 or index >= len(self._model_predictions):
            return
        item = self._model_predictions[index]
        if len(item.points) < 2:
            self.status_label.setText(
                f"Animal {index + 1} was found (score {item.score:.2f}), but the pose model "
                f"placed no points in its box - measure it by hand.")
            return
        self._loading = True
        self.canvas.set_points(list(item.points))
        self._loading = False
        self.status_label.setText(
            f"Animal {index + 1} of {len(self._model_predictions)} "
            f"(score {item.score:.2f}) is in the draft - correct the points, then Save.")

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
        """A point was dragged and let go: write the correction if this pair is already stored.

        A pair that is not stored yet waits for *Save* - placing a first point must never store
        anything. A stored one is a correction, and the operator should not have to press Save
        twice for the same row (`SR-DATASET-027`).
        """
        if self._loading or not self.queue or self._stored(self.specimen(), self.trait()) is None:
            return
        if len(self.canvas.points) < 2:
            return
        self._save_current()

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

    def _set_enabled(self, enabled):
        """Enable the panel's choices and the tools on the image together.

        The two sets mean the same thing: a screen with no frames has nothing to
        measure, so a *Save* on the canvas must be as dead as the one in the panel -
        and the tools that walk the queue follow its ends, exactly like the buttons.
        """
        for widget in (self.combo_trait, self.spin_specimen):
            widget.setEnabled(enabled)
        for key in ("save", "clear", "new_specimen", "refresh"):
            self.workspace.action(key).setEnabled(enabled)
        first = enabled and self.index > 0
        last = enabled and self.index < len(self.queue) - 1
        self.workspace.action("prev").setEnabled(first)
        self.workspace.action("next").setEnabled(last)

    def _sync_specimens(self):
        """Clamp the specimen spinner to the specimens already on this frame."""
        highest = max((row["specimen_index"] for row in self._rows), default=1)
        self.spin_specimen.blockSignals(True)
        self.spin_specimen.setMaximum(max(20, highest + 1))
        self.spin_specimen.setValue(min(max(self.specimen(), 1), highest))
        self.spin_specimen.blockSignals(False)

    def _on_subject_changed(self, *_):
        """Show the stored landmarks of this (specimen, trait), or clear them."""
        self.canvas.point_names = list(self._point_names())
        stored = self._stored(self.specimen(), self.trait())
        if stored is not None:
            self.canvas.set_points([(stored["p1_x"], stored["p1_y"]),
                                    (stored["p2_x"], stored["p2_y"])])
            self.measurement_label.setText(self._measurement_text(stored))
            self.status_label.setText(
                "This specimen/trait is measured - saving again updates it.")
        else:
            self.canvas.clear_points()
            self.measurement_label.setText(
                "Not measured yet - click the two landmarks and save.")
            self.status_label.setText(self._prompt())
        self._fill_list()
        self._refresh_header()

    def _stored(self, specimen, trait):
        for row in self._rows:
            if row["specimen_index"] == specimen and row["trait"] == trait:
                return row
        return None

    def _point_names(self):
        if self._spec is not None:
            return self._spec.points_for(self.trait())
        return ("head", "spine base")

    def _fill_list(self):
        """List the measurements already stored on this frame."""
        self.list_measurements.clear()
        for row in self._rows:
            label = TRAIT_LABELS.get(row["trait"], row["trait"])
            self.list_measurements.addItem(
                f"Specimen {row['specimen_index']}  {label}: "
                f"{row['length_um']:.1f} um")

    def _measurement_text(self, measurement) -> str:
        """Readout of a stored measurement, so the value is visible on this panel.

        The stored row carries its own budget, so it is *read*, not recomputed:
        the ``total +/-`` shown here is the number the export writes
        (``corpus_labels.total_uncertainty_um``, `docs/accuracy.md` §1). A row from
        before that column is derived from the numbers it did record - the same
        function, so the two can only differ if a constant changed.
        """
        keys = measurement.keys()
        uncertainty = (measurement["uncertainty_pct"]
                       if "uncertainty_pct" in keys else None)
        if uncertainty is None:
            length = units.format_length(measurement["length_um"])
        else:
            length = units.format_with_uncertainty(
                measurement["length_um"], uncertainty)
        total = self._stored_total_text(measurement, uncertainty)
        return (f"{self._endpoint()}: {length}{total}   "
                f"({measurement['length_px']:.1f} px at "
                f"{measurement['px_per_um']:.6f} px/um, "
                f"{measurement['source']} by {measurement['measured_by']})")

    @staticmethod
    def _stored_total_text(measurement, uncertainty) -> str:
        """``'   total +/- 75.1 um (3.8 %)'`` for a stored row, else ``''``.

        Nothing is printed when the row has no scale to derive a budget from: a
        label stored without a calibration has no micrometres, so it has no budget
        either - and inventing one would be the false precision `SR-OPTICS-005`
        exists to prevent.
        """
        keys = measurement.keys()
        px_per_um = measurement["px_per_um"] if "px_per_um" in keys else None
        length_um = measurement["length_um"] if "length_um" in keys else None
        if not px_per_um or not uncertainty or not length_um:
            return ""
        stored = (measurement["total_uncertainty_um"]
                  if "total_uncertainty_um" in keys else None)
        return "   " + units.format_total_uncertainty(
            length_um, px_per_um, uncertainty, total_um=stored)

    def _endpoint(self) -> str:
        return self._spec.endpoint_for(self.trait()) if self._spec else "length"

    def _step(self, delta):
        if not self.queue:
            return
        self.index = max(0, min(self.index + delta, len(self.queue) - 1))
        self.spin_specimen.blockSignals(True)
        self.spin_specimen.setValue(1)
        self.spin_specimen.blockSignals(False)
        self._show_current()

    def _new_specimen(self):
        """Move to the next specimen on this frame."""
        self.spin_specimen.setValue(min(self.spin_specimen.value() + 1,
                                        self.spin_specimen.maximum()))

    # -- annotation --------------------------------------------------------
    def _prompt(self) -> str:
        """Ask for the two landmarks of this study's organism."""
        first, second = self.canvas.point_names
        return f"Click the {first}, then the {second}."

    def _update_readout(self, points):
        with connection() as conn:
            profile = repository.active_profile(conn)
        if profile is None:
            self.length_label.setText(workflow.REASONS["calibration"])
        elif len(points) < 2:
            self.length_label.setText(self._prompt())
        else:
            keypoints = Keypoints(eye_apex=points[0], spine_base=points[1])
            um = profile.to_um(keypoints.length_px)
            total = units.format_total_uncertainty(um, profile.px_per_um,
                                                   profile.uncertainty_pct)
            self.length_label.setText(
                f"{self._endpoint()}: "
                f"{units.format_with_uncertainty(um, profile.uncertainty_pct)}   "
                f"{total}   "
                f"({keypoints.length_px:.1f} px / {profile.px_per_um:.6f} px/um)")
        self._refresh_header()

    def _refresh_header(self):
        """Publish the facts the **window header** reads: which frame, the last message.

        A screen where the operator measures used to repeat its number over the picture so
        the number survived full screen; that strip is **gone** (2026-10-18). The number
        lives in the panel (``length_label``), which is where it is read, and the header
        carries the context - the frame and the message - because those belong to the
        window rather than to the specimen.
        """
        self.workspace.set_header([
            Kpi("task", self.task_label.text(), label="Frame"),
            Kpi("status", self.status_label.text(), label="Status"),
        ])

    def _save_current(self):
        if not self.queue:
            return
        points = self.canvas.points
        if len(points) < 2:
            self.status_label.setText(self._prompt())
            return

        asset = self.queue[self.index]
        with connection() as conn:
            profile = repository.active_profile(conn)
            if profile is None:
                self.status_label.setText(workflow.REASONS["calibration"])
                return
            repository.save_measurement(
                conn,
                observation_id=asset["observation_id"],
                asset_id=asset["asset_id"],
                profile=profile,
                trait=self.trait(),
                specimen_index=self.specimen(),
                p1=points[0],
                p2=points[1],
            )
        label = TRAIT_LABELS.get(self.trait(), self.trait())
        self.status_label.setText(
            f"Saved specimen {self.specimen()} {label} on "
            f"{queries.target_label(asset)}.")
        self.notify_workflow()
        self._reload_rows()

    def _reload_rows(self):
        """Refresh the frame's measurements and the queue progress after a save."""
        asset = self.queue[self.index]
        with connection() as conn:
            self._rows = queries.measurements_for_asset(conn, asset["asset_id"])
        row = dict(asset)
        row["measurements"] = len(self._rows)
        self.queue[self.index] = row
        annotated = sum(1 for entry in self.queue if entry["measurements"])
        self.progress_label.setText(f"{annotated} / {len(self.queue)} frames measured")
        self._on_subject_changed()
