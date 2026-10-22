"""Overview step - pick a project and see where it stands.

The landing screen: the project list on the left (current one first, with its
progress), and a summary of the selected project on the right - including what the
workflow expects next, so a half-finished run says what is still missing.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QTableWidget, QTableWidgetItem

from daphnia import maintenance, workflow
from daphnia.db import connection, queries, repository
from daphnia.study import StudySpec, stage_of
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import FILE_MENU, ToolSpec

COLUMNS = ("project", "organism", "mode", "created", "targets", "frames", "measured")


class Dashboard(ShellPage):
    """Project chooser plus the selected project's status."""

    def __init__(self, parent=None):
        super().__init__(ui_file='dashboard.ui', parent=parent)

        self.table = self.add_visual(QTableWidget(0, len(COLUMNS)))
        self.table.setHorizontalHeaderLabels([c.capitalize() for c in COLUMNS])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._show_selected)

        self.btn_refresh.clicked.connect(self.refresh)

        # The project lifecycle is **File** material, not panel material (`X-N13`): the
        # bar carries *New* / *Open* / *Close* / *Prune*, and the panel shows the list
        # they act on. This step has no canvas, so its actions go through `menu_tools`.
        tools = self.menu_tools()
        tools.set_group(FILE_MENU)
        tools.add_tool(
            ToolSpec("new_project", "New project...",
                     "Walk to Setup and describe a new project."),
            self._new_project)
        self._open_action = tools.add_tool(
            ToolSpec("open_project", "Open the selected project",
                     "Make the project selected in the list the one in force."),
            self._open_selected)
        tools.add_tool(
            ToolSpec("close_project", "Close the project",
                     "Consolidate the selected project: prune it, export its "
                     "measurements, write its manifest, then put it away."),
            self._close_project)
        tools.add_tool(
            ToolSpec("prune", "Prune discarded frames...",
                     "Delete the discarded, unmeasured frames of the selected project "
                     "from disk."),
            self._prune_discarded)

        self._studies = []
        self.refresh()

    def on_shown(self):
        self.refresh()

    # -- content -----------------------------------------------------------
    def refresh(self):
        with connection() as conn:
            self._studies = queries.list_studies_with_progress(conn)
            profile = repository.active_profile(conn)
            spec = queries.active_study_spec(conn)
            reason = workflow.blocking_reason(conn, 1)

        self._fill_table()
        self.next_label.setText(f"Next: {reason}" if reason else "Workflow complete.")
        if profile is None:
            self.status_label.setText("No active optical preset - calibrate before measuring.")
        else:
            self.status_label.setText(
                f"Active preset {profile.profile_id}: "
                f"S = {profile.px_per_um * 1000.0:.1f} px/mm")
        self._show_selected()

    def _fill_table(self):
        self.table.setRowCount(len(self._studies))
        for row_index, row in enumerate(self._studies):
            values = (
                ("* " if row["is_current"] else "") + row["name"],
                row["organism"],
                StudySpec.from_row(row).describe(),
                row["created_at"][:10],
                row["targets"],
                row["frames"],
                row["measured"],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, row["study_id"])
                self.table.setItem(row_index, column, item)
        self.table.resizeColumnsToContents()

        # preselect the project being worked on
        for row_index, row in enumerate(self._studies):
            if row["is_current"]:
                self.table.selectRow(row_index)
                break

    def _selected_row(self):
        """The study row the operator highlighted, or ``None``."""
        items = self.table.selectedItems()
        if not items:
            return None
        study_id = items[0].data(Qt.UserRole)
        return next((row for row in self._studies if row["study_id"] == study_id),
                    None)

    def _show_selected(self):
        row = self._selected_row()
        if row is None:
            self.project_label.setText("No project yet - create one to begin.")
            self.stage_label.setText("")
            self.progress_label.setText("")
            self._open_action.setEnabled(False)
            return

        spec = StudySpec.from_row(row)
        self.project_label.setText(
            f"{spec.name}  ({spec.organism})\n"
            f"mode: {spec.describe()}\n"
            f"images -> {spec.destination}")
        self.stage_label.setText(stage_of(row))
        self.progress_label.setText(
            f"plates {row['plates']}   targets {row['targets']}   frames {row['frames']}   "
            f"selected {row['selected']}   measured {row['measured']}")
        # *Open* means nothing for the project already in force, so the menu item is
        # disabled - the list's own state is what the item acts on.
        self._open_action.setEnabled(not row["is_current"])

    # -- actions -----------------------------------------------------------
    def _open_selected(self):
        row = self._selected_row()
        if row is None or row["is_current"]:
            return
        with connection() as conn:
            repository.set_current_study(conn, row["study_id"])
        self.notify_workflow()               # the shell re-evaluates every gate
        self.refresh()

    def _new_project(self):
        """Hand over to the shell to walk to the Setup step."""
        window = self.window()
        go_to = getattr(window, "go_to", None)
        if callable(go_to):
            go_to("experiment")

    # -- data lifecycle ----------------------------------------------------
    def _prune_discarded(self):
        """Delete the discarded, un-measured frames of the selected project."""
        row = self._selected_row()
        if row is None:
            return
        confirm = QMessageBox.question(
            self, "Prune discarded frames",
            f"Delete the discarded burst frames of '{row['name']}' from disk?\n"
            "Kept frames and every measurement are left untouched.")
        if confirm != QMessageBox.Yes:
            return
        with connection() as conn:
            result = maintenance.prune_discarded(conn, row["study_id"])
        self.refresh()
        self.status_label.setText(
            f"Pruned {result['rows']} discarded frame(s); "
            f"{result['files']} file(s) removed from disk.")

    def _close_project(self):
        """Consolidate the selected project (prune, export, manifest) and close it."""
        row = self._selected_row()
        if row is None:
            return
        confirm = QMessageBox.question(
            self, "Close and consolidate",
            f"Consolidate '{row['name']}'?\n\nThis prunes discarded frames, exports "
            "the measurements as CSV, writes a manifest next to the images and "
            "closes the study.")
        if confirm != QMessageBox.Yes:
            return
        with connection() as conn:
            result = maintenance.consolidate_and_close(conn, row["study_id"])
        self.refresh()
        self.status_label.setText(
            f"Consolidated '{row['name']}': pruned {result['pruned']['rows']} frame(s), "
            f"exported {result['exported']} measurement(s), wrote the manifest.")
        self.notify_workflow()
