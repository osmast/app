"""Analytics step - the persisted measurements.

Charts, dose-response curves and export come later; the table already proves the
round trip: study -> calibrate -> capture -> select -> annotate -> store.
"""

from PySide6.QtWidgets import QFileDialog, QTableWidget, QTableWidgetItem

from daphnia import maintenance, units, workflow
from daphnia.db import connection, queries
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import FILE_MENU, ToolSpec

COLUMNS = ("created_at", "target", "specimen", "trait", "length_um", "length_mm",
           "length_px", "px_per_um", "source", "measured_by")


class Visualize(ShellPage):
    """Table of measurements with their target context."""

    def __init__(self, parent=None):
        super().__init__(ui_file='visualize.ui', parent=parent)

        self.table = self.add_visual(QTableWidget(0, len(COLUMNS)))
        self.table.setHorizontalHeaderLabels([c.replace('_', ' ') for c in COLUMNS])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.btn_refresh.clicked.connect(self.refresh)

        # The one thing this step produces for the outside world is a file, so it is a
        # *File* menu item (`X-N13`) rather than a button under the table it exports.
        tools = self.menu_tools()
        tools.set_group(FILE_MENU)
        tools.add_tool(
            ToolSpec("export_csv", "Export measurements (CSV)...",
                     "Write this study's measurements - and only this study's - as a CSV "
                     "file."),
            self._export_csv)
        self.refresh()

    def on_shown(self):
        self.refresh()

    def refresh(self):
        # Analytics is about *this* run: only the active study's measurements.
        with connection() as conn:
            study = queries.active_study(conn)
            rows = (queries.measurements_for_study(conn, study["study_id"])
                    if study is not None else [])

        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row["created_at"],
                queries.target_label(row),
                row["specimen_index"],
                row["trait"],
                f"{row['length_um']:.1f}",
                f"{units.to_mm(row['length_um']):.3f}",
                f"{row['length_px']:.1f}",
                f"{row['px_per_um']:.6f}",
                row["source"],
                row["measured_by"],
            )
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(str(value)))

        if study is None:
            self.summary_label.setText("No study yet - set one up on the Setup step.")
        else:
            self.summary_label.setText(
                f"{len(rows)} measurement(s) in '{study['name']}'" if rows
                else f"No measurements yet in '{study['name']}'.")

    def _export_csv(self):
        """Export the active study's measurements as a CSV file."""
        with connection() as conn:
            study = queries.active_study(conn)
        if study is None:
            self.status_label.setText(workflow.REASONS["study"])
            return

        suggested = f"{study['name'].replace(' ', '_')}_measurements.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export measurements", suggested, "CSV (*.csv)")
        if not path:
            return
        with connection() as conn:
            count = maintenance.export_measurements_csv(conn, study["study_id"], path)
        self.status_label.setText(f"Exported {count} measurement(s) to {path}")
