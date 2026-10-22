"""Setup step - the study, its plates and the phone connection.

Everything that has to be decided *once* happens here: the study (which owns the
image folder), the plate geometries, and the connection to the phone. Calibration
and acquisition reuse that connection instead of opening their own, so the phone is
connected exactly once per session.
"""

from PySide6.QtWidgets import QFileDialog

from daphnia import paths, protocols
from daphnia.study import (
    DEFAULT_ORGANISM_KIND,
    DEFAULT_TARGET_MODE,
    DEFAULT_TIMECOURSE,
    ORGANISM_KINDS,
    TARGET_MODES,
    TIMECOURSES,
)
from daphnia.db import connection, queries, repository
from daphnia.ids import new_id
from daphnia.templates import StudyTemplate
from daphnia.streaming import (
    lan_ip,
    publish_srt_url,
    read_rtsp_url,
    shared_service,
)
from daphnia.ui.shell import ShellPage
from daphnia.ui.widgets import StreamView


class SetupExperiment(ShellPage):
    """Create the study, add plates and connect the phone - once."""

    def __init__(self, parent=None):
        super().__init__(ui_file='setup_experiment.ui', parent=parent)

        # The one connection of the app: every other screen observes this service.
        self.service = shared_service()
        self.stream = self.add_visual(StreamView(self.service))
        self.service.status.connect(self.status_label.setText)
        self.service.opened.connect(self._on_service_state_changed)
        self.service.closed.connect(self._on_service_state_changed)

        self._destination = None
        # The connect/disconnect of *this* address stay beside the fields that carry it:
        # the host and the URL are the *choice*, so the button that applies them is part
        # of it, exactly like the study combo and its Attach in Bench. The header's
        # *Source* menu connects the stream that is already configured; this pair
        # configures it first - the walkthrough is in the header's *Help* menu.
        self.btn_choose_destination.clicked.connect(self._choose_destination)
        self.btn_create_study.clicked.connect(self._create_study)
        self.btn_add_plate.clicked.connect(self._add_plate)
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.edit_host.textChanged.connect(self._sync_urls)
        self.btn_load_template.clicked.connect(self._load_template)
        self.btn_save_template.clicked.connect(self._save_template)
        self.combo_protocol.currentIndexChanged.connect(self._sync_protocol_hint)

        self._fill_protocols()
        self._sync_protocol_hint()
        self.edit_host.setText(lan_ip())
        self._sync_urls()
        self._sync_connection()
        self.refresh()

    def on_shown(self):
        self.refresh()
        self._sync_connection()

    # -- connection --------------------------------------------------------
    def _sync_urls(self):
        """Keep the read URL and the phone's publish URL in step with the host."""
        host = self.edit_host.text().strip() or lan_ip()
        self.edit_url.setText(read_rtsp_url(host=host))
        self.publish_label.setText(publish_srt_url(host=host))

    def _connect(self):
        """Start the one shared stream from the URL this step publishes."""
        self.service.set_url(self.edit_url.text().strip())
        self.service.start()
        self._on_service_state_changed()

    def _disconnect(self):
        self.service.stop()
        self._on_service_state_changed()

    def _on_service_state_changed(self):
        """Follow the service, so the shell's gates re-evaluate on connect/drop."""
        self._sync_connection()
        self.notify_workflow()

    def _sync_connection(self):
        """Report the connection, and enable the button that applies this address."""
        connected = self.service.connected
        self.btn_connect.setEnabled(not connected)
        self.btn_disconnect.setEnabled(connected)
        self.connection_label.setText(
            f"Connected: {self.service.url}" if connected
            else "Not connected. Start MediaMTX on this Mac, then connect.")

    def _kind(self) -> str:
        """Selected organism kind value."""
        label = self.combo_kind.currentText()
        return next((key for key, text in ORGANISM_KINDS.items() if text == label),
                    DEFAULT_ORGANISM_KIND)

    def _target_mode(self) -> str:
        """Selected target mode value."""
        label = self.combo_target_mode.currentText()
        return next((key for key, text in TARGET_MODES.items() if text == label),
                    DEFAULT_TARGET_MODE)

    def _timecourse(self) -> str:
        """Selected timecourse value."""
        label = self.combo_timecourse.currentText()
        return next((key for key, text in TIMECOURSES.items() if text == label),
                    DEFAULT_TIMECOURSE)

    def refresh(self):
        with connection() as conn:
            spec = queries.active_study_spec(conn)
            plate_rows = queries.list_plates(conn, spec.study_id) if spec else []

        if spec is None:
            self.study_label.setText("No study yet.")
        else:
            self.study_label.setText(
                f"Active study: {spec.name} ({spec.organism})\n"
                f"mode: {spec.describe()}\n"
                f"protocol: {spec.protocol.label}\n"
                f"interval: every {spec.interval_hours:g} h\n"
                f"images: {spec.destination}")

        if plate_rows:
            self.plates_label.setText(
                "Plates: " + ", ".join(
                    f"{row['label']} ({row['format']}-well)" for row in plate_rows))
        else:
            self.plates_label.setText(
                "No plates yet - the individual counter works without one.")

        self._fill_templates()

    def _choose_destination(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Where should the images of this study be stored?")
        if folder:
            self._destination = folder
            self.destination_label.setText(f"Images: {folder}")

    def _create_study(self):
        name = self.study_name.text().strip()
        if not name:
            self.status_label.setText("Give the study a name first.")
            return

        # Pre-generate the id so the default destination is concrete, not a guess.
        study_id = new_id("STUDY")
        destination = self._destination or str(paths.study_dir(study_id))

        with connection() as conn:
            repository.create_study(
                conn,
                study_id=study_id,
                name=name,
                destination=destination,
                organism=self.organism.text().strip() or "Daphnia magna",
                organism_kind=self._kind(),
                target_mode=self._target_mode(),
                timecourse=self._timecourse(),
                measurement_interval_hours=self.interval_hours.value(),
                protocol_key=self._protocol_key(),
            )

        self.status_label.setText(
            f"Study {study_id} created - images go to {destination}")
        self.refresh()
        self.notify_workflow()

    # -- protocols & templates ---------------------------------------------
    def _protocol_key(self) -> str:
        """Selected acquisition protocol key."""
        return protocols.key_for_label(self.combo_protocol.currentText())

    def _fill_protocols(self):
        """Populate the protocol combo from :mod:`daphnia.protocols`."""
        self.combo_protocol.clear()
        for protocol in protocols.PROTOCOLS.values():
            self.combo_protocol.addItem(protocol.label, protocol.key)
        self.combo_protocol.setCurrentText(
            protocols.get(protocols.DEFAULT_PROTOCOL).label)

    def _sync_protocol_hint(self):
        """Show the trade-off of the selected protocol, so the choice is informed."""
        self.protocol_hint_label.setText(
            protocols.get(self._protocol_key()).tradeoff)

    def _fill_templates(self):
        """Repopulate the template combo with the latest version of each name."""
        with connection() as conn:
            rows = queries.list_study_templates(conn)
        self.combo_template.blockSignals(True)
        self.combo_template.clear()
        for row in rows:
            self.combo_template.addItem(
                StudyTemplate.from_row(row).label(), row["template_id"])
        self.combo_template.blockSignals(False)
        self.btn_load_template.setEnabled(bool(rows))

    def _load_template(self):
        """Apply the selected template to the study form."""
        template_id = self.combo_template.currentData()
        if template_id is None:
            self.status_label.setText("Save a template first.")
            return
        with connection() as conn:
            row = repository.template_by_id(conn, template_id)
        if row is None:
            return
        template = StudyTemplate.from_row(row)
        self.organism.setText(template.organism)
        self.combo_kind.setCurrentText(
            ORGANISM_KINDS.get(template.organism_kind,
                               ORGANISM_KINDS[DEFAULT_ORGANISM_KIND]))
        self.combo_target_mode.setCurrentText(
            TARGET_MODES.get(template.target_mode,
                             TARGET_MODES[DEFAULT_TARGET_MODE]))
        self.combo_timecourse.setCurrentText(
            TIMECOURSES.get(template.timecourse,
                            TIMECOURSES[DEFAULT_TIMECOURSE]))
        self.interval_hours.setValue(template.measurement_interval_hours)
        self.combo_protocol.setCurrentText(template.protocol.label)
        self.status_label.setText(
            f"Loaded template '{template.label()}' - give the study a name.")

    def _save_template(self):
        """Save the current form as a new version of a named template."""
        name = self.template_name.text().strip() or self.study_name.text().strip()
        if not name:
            self.status_label.setText(
                "Name the template (or the study) before saving it.")
            return
        with connection() as conn:
            repository.save_study_template(
                conn, name=name,
                organism=self.organism.text().strip() or "Daphnia magna",
                organism_kind=self._kind(),
                target_mode=self._target_mode(),
                timecourse=self._timecourse(),
                measurement_interval_hours=self.interval_hours.value(),
                protocol_key=self._protocol_key(),
            )
        self._fill_templates()
        self.combo_template.setCurrentIndex(0)     # newest name first
        self.template_name.clear()
        self.status_label.setText(f"Saved template '{name}' as a new version.")

    # -- plates ------------------------------------------------------------
    def _add_plate(self):
        format_ = self.combo_format.currentText()
        with connection() as conn:
            study = queries.active_study(conn)
            if study is None:
                self.status_label.setText("Create the study first.")
                return
            repository.create_plate(conn, study_id=study["study_id"], format=format_)
        self.status_label.setText(
            f"Added a {format_}-well plate - its wells are selectable while acquiring.")
        self.refresh()
