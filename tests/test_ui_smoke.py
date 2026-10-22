"""Headless smoke test: the shell builds and the step gates hold."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QScrollArea, QToolButton, QWidget

import numpy as np

from daphnia import models, protocols, surfaces, workflow
from daphnia.db import connection, init_db, queries, repository
from daphnia.streaming import StreamService
from daphnia.ui.hub import (
    BenchWindow,
    CalibrationWindow,
    DatasetWindow,
    HubWindow,
    PredictWindow,
    Study2Window,
    SurfacePlanWindow,
)
from daphnia.ui.main_window import MainWindow
from daphnia.ui.pages.bench import Bench
from daphnia.ui.widgets import HeaderBar, Kpi, StatusFooter
from daphnia.ui.widgets.canvas_workspace import WINDOW_CHROME
from daphnia.ui.widgets.header_bar import DETECTOR_MENU, POSE_MENU, SOURCE_MENU

#: The single *Models* menu this replaced (`X-N13`): a window that still shows it would be
#: reading the old bar.
MODELS_MENU_GONE = "Models"

TABLES = ("corpus_labels", "corpus_boxes", "media_assets", "observations",
          "imaging_sessions", "calibration_records", "calibration_profiles",
          "plates", "studies", "models")


class ScreenTestCase(unittest.TestCase):
    """A temporary database plus one QApplication, shared by the smoke tests."""

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "ui-test.db")
        init_db()
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        if cls._previous_db is None:
            os.environ.pop("DAPHNIA_DB", None)
        else:
            os.environ["DAPHNIA_DB"] = cls._previous_db

    def _clear_tables(self):
        """Every test starts from an empty database."""
        with connection() as conn:
            for table in TABLES:
                conn.execute(f"DELETE FROM {table}")


class ShellTests(ScreenTestCase):
    def setUp(self):
        self._clear_tables()
        self.window = MainWindow()

    def _study(self):
        with connection() as conn:
            repository.create_study(conn, name="T", destination=str(self._tmp.name))

    def _measured_study(self, name):
        """Create a study with one kept frame and one saved measurement."""
        with connection() as conn:
            study_id = repository.create_study(
                conn, name=name, destination=str(self._tmp.name))
            repository.upsert_profile(
                conn, profile_id="PROF_T", profile_name="t", microscope_model="m",
                objective_magnification="2x", camera_device="c", px_per_um=5.0)
            repository.activate_profile(conn, "PROF_T")
            session_id = repository.ensure_session(
                conn, study_id=study_id, profile_id="PROF_T")
            observation_id = repository.create_observation(
                conn, session_id=session_id, individual_index=1)
            asset_id = repository.add_media_asset(
                conn, observation_id=observation_id, file_path="f.png",
                sharpness=5.0, selection="kept")
            repository.save_manual_measurement(
                conn, observation_id=observation_id, asset_id=asset_id,
                profile=repository.active_profile(conn),
                eye_apex=(0.0, 0.0), spine_base=(50.0, 0.0))
        return study_id

    def _study_with_targets(self, frame_counts):
        """A study with one individual per count, each holding that many frames."""
        with connection() as conn:
            study_id = repository.create_study(
                conn, name="Select", destination=str(self._tmp.name))
            repository.upsert_profile(
                conn, profile_id="PROF_T", profile_name="t", microscope_model="m",
                objective_magnification="2x", camera_device="c", px_per_um=5.0)
            session_id = repository.ensure_session(
                conn, study_id=study_id, profile_id="PROF_T")
            for individual, frames in enumerate(frame_counts, start=1):
                observation_id = repository.create_observation(
                    conn, session_id=session_id, individual_index=individual)
                for frame_index in range(frames):
                    repository.add_media_asset(
                        conn, observation_id=observation_id,
                        file_path=f"f{frame_index}.png", frame_index=frame_index,
                        burst_index=1, sharpness=float(frame_index))
        return study_id

    # -- shell -------------------------------------------------------------
    def test_all_steps_are_registered(self):
        self.assertEqual(self.window.stackedWidget.count(), len(workflow.STEPS))

    def test_always_opens_at_home(self):
        self.assertEqual(self.window.stackedWidget.currentIndex(), 0)
        self.assertEqual(workflow.STEPS[0].route, "dashboard")

    def test_body_scrolls_as_one_region(self):
        self.assertIsInstance(self.window.body_scroll, QScrollArea)
        self.assertIs(self.window.body_scroll.widget(), self.window.stackedWidget)

    def test_every_screen_has_a_marginless_body(self):
        for step in workflow.STEPS:
            page = self.window.pages[step.route]
            self.assertIsNotNone(page.visual.layout())
            margins = page.visual.layout().contentsMargins()
            self.assertEqual((margins.left(), margins.top(),
                              margins.right(), margins.bottom()), (0, 0, 0, 0))

    # -- gates -------------------------------------------------------------
    def test_acquire_cannot_be_entered_before_setup_and_calibration(self):
        self.assertFalse(self.window.go_to_step(workflow.index_of("acquire")))
        self.assertEqual(self.window.stackedWidget.currentIndex(), 0)

        # Standing on Setup the step bar spells out what is missing.
        self.window.go_continue()
        self.window.go_continue()
        self.assertEqual(self.window.stackedWidget.currentIndex(),
                         workflow.index_of("experiment"))
        self.assertIn("study", self.window.step_bar.hint_label.text().lower())

    def test_continue_from_home_reaches_setup(self):
        self.assertTrue(self.window.step_bar.btn_continue.isEnabled())
        self.window.go_continue()
        self.assertEqual(self.window.stackedWidget.currentIndex(),
                         workflow.index_of("experiment"))

    def _pretend_connected(self):
        """The service is never started in tests, so fake the runtime fact."""
        self.window._runtime_facts = lambda: {"connection": True}

    def test_setup_cannot_advance_without_a_study_and_a_connection(self):
        self.window.go_continue()
        self.assertFalse(self.window.step_bar.btn_continue.isEnabled())
        self.assertIn("study", self.window.step_bar.hint_label.text().lower())

        self._study()
        self.window.refresh_workflow()
        # a study alone is not enough: calibrating needs the phone connection
        self.assertFalse(self.window.step_bar.btn_continue.isEnabled())
        self.assertIn("connect", self.window.step_bar.hint_label.text().lower())

        self._pretend_connected()
        self.window.refresh_workflow()
        self.assertTrue(self.window.step_bar.btn_continue.isEnabled())

    def test_acquire_refuses_without_the_connection(self):
        self._study()
        with connection() as conn:
            repository.upsert_profile(
                conn, profile_id="PROF_T", profile_name="t", microscope_model="m",
                objective_magnification="2x", camera_device="c", px_per_um=5.0)
            repository.activate_profile(conn, "PROF_T")
        self.window.refresh_workflow()

        self.window.go_to_step(workflow.index_of("experiment"))
        self.assertFalse(self.window.go_to_step(workflow.index_of("acquire")))
        self.assertIn("connect", self.window.step_bar.hint_label.text().lower())

    def test_continue_stops_at_the_next_gate(self):
        self._study()
        self._pretend_connected()
        self.window.refresh_workflow()
        self.window.go_continue()          # home  -> setup
        self.window.go_continue()          # setup -> calibrate
        self.assertEqual(self.window.stackedWidget.currentIndex(),
                         workflow.index_of("calibration"))

        self.window.go_continue()          # calibrate -> acquire needs a scale
        self.assertEqual(self.window.stackedWidget.currentIndex(),
                         workflow.index_of("calibration"))
        self.assertIn("calibrate", self.window.step_bar.hint_label.text().lower())

    def test_back_returns_towards_home(self):
        self.window.go_continue()
        self.window.go_back()
        self.assertEqual(self.window.stackedWidget.currentIndex(), 0)

    # -- protocols & templates (feature wiring) ----------------------------
    def test_setup_offers_the_configured_protocols(self):
        page = self.window.pages["experiment"]
        labels = [page.combo_protocol.itemText(i)
                  for i in range(page.combo_protocol.count())]
        self.assertEqual(labels, protocols.labels())
        self.assertTrue(page.protocol_hint_label.text())

    def test_acquire_defaults_its_burst_to_the_study_protocol(self):
        with connection() as conn:
            repository.create_study(
                conn, name="T", destination=str(self._tmp.name),
                protocol_key="careful")
        page = self.window.pages["acquire"]
        page.refresh_targets()
        self.assertEqual(page.spin_burst.value(),
                         protocols.get("careful").burst_size)

    def test_every_stream_preview_is_ratio_locked(self):
        """The Calibrate preview must behave like the others, not a thin strip."""
        for route in ("experiment", "calibration", "acquire"):
            stream = self.window.pages[route].stream
            self.assertTrue(stream.lock_ratio,
                            f"{route} preview must ratio-lock like the others")

    def test_overview_and_analytics_expose_the_lifecycle_controls(self):
        """The project lifecycle and the export are **menu items** now (`X-N13`).

        The buttons left the panels when the bar took them: what this checks is that the
        actions are still reachable - and that the old panel attributes are gone, so a page
        cannot quietly keep a second copy of a menu item.
        """
        dashboard = self.window.pages["dashboard"]
        visualize = self.window.pages["visualize"]
        for key in ("new_project", "open_project", "close_project", "prune"):
            self.assertIsNotNone(dashboard.menu_tools().action(key), key)
        self.assertIsNotNone(visualize.menu_tools().action("export_csv"))
        self.assertTrue(hasattr(visualize, "status_label"))
        for key in ("btn_prune", "btn_close_project", "btn_export_csv"):
            self.assertFalse(hasattr(dashboard if key != "btn_export_csv" else visualize, key))

    def test_analytics_only_lists_the_active_study(self):
        self._measured_study("First")
        self._measured_study("Second")            # becomes the current project
        page = self.window.pages["visualize"]
        page.refresh()
        self.assertEqual(page.table.rowCount(), 1)
        with connection() as conn:               # the global list still holds both
            self.assertEqual(len(queries.list_measurements(conn)), 2)

    def test_annotate_panel_shows_the_saved_measurement(self):
        self._measured_study("Annotate me")
        page = self.window.pages["annotate"]
        page.refresh()
        self.assertEqual(len(page.queue), 1)
        text = page.measurement_label.text()
        self.assertIn("um", text)
        self.assertIn("mm", text)                 # both units are reported

    def test_annotate_lists_each_measurement_of_a_frame(self):
        self._measured_study("Annotate me")
        with connection() as conn:
            asset = conn.execute(
                "SELECT asset_id, observation_id FROM media_assets LIMIT 1").fetchone()
            repository.save_measurement(
                conn, observation_id=asset["observation_id"],
                asset_id=asset["asset_id"], profile=repository.active_profile(conn),
                trait="width", specimen_index=2, p1=(0.0, 0.0), p2=(20.0, 0.0))
        page = self.window.pages["annotate"]
        page.refresh()
        self.assertEqual(page.combo_trait.count(), 2)      # Length / Width
        self.assertEqual(page.list_measurements.count(), 2)  # one frame, two rows
        page.spin_specimen.setValue(2)
        page.combo_trait.setCurrentIndex(1)                # Width
        self.assertIn("width", page.measurement_label.text().lower())

    def test_select_shows_every_frame_and_navigates_back_and_forth(self):
        self._study_with_targets([1, 3, 2])
        page = self.window.pages["select"]
        page.refresh()
        self.assertEqual(page.combo_target.count(), 3)
        self.assertEqual(page.gallery.count(), 1)          # Individual 1 has one frame
        self.assertIn("1 of 3", page.position_label.text())
        self.assertFalse(page.btn_prev_target.isEnabled())

        page.btn_next_target.click()                        # forward one individual
        self.assertEqual(page.gallery.count(), 3)           # all of its frames
        self.assertIn("2 of 3", page.position_label.text())

        page.btn_prev_target.click()                        # and back
        self.assertEqual(page.gallery.count(), 1)
        self.assertIn("1 of 3", page.position_label.text())

    def test_controls_panel_scrolls_instead_of_the_whole_body(self):
        """A tall form must not push the live stream out of view."""
        from PySide6.QtWidgets import QScrollArea
        experiment = self.window.pages["experiment"]
        self.assertIsNotNone(
            experiment.controls.findChild(QScrollArea, "controls_scroll"))
        self.assertLess(experiment.controls.sizeHint().height(), 600)

    def test_every_panel_is_wide_enough_for_what_is_in_it(self):
        """The contract behind ``CONTROLS_WIDTH``, measured on every built surface.

        The controls panel scrolls **vertically only** (its horizontal scroll bar is
        off), so a child wider than the viewport is not just clipped - it is
        unreachable. Each surface therefore states its panel's width - the shared
        ``CONTROLS_WIDTH``, or its own ``PANEL_WIDTH`` where the screen needs a different
        one (Dataset needs **more** for a combo that sizes itself to its longest item,
        Predict takes **less** and gives it to the picture the detector is judged on) - and
        the content it holds has to fit inside whichever it is. What no surface may do is
        be narrower than its own content.
        """
        from PySide6.QtWidgets import QScrollArea

        from daphnia.ui.hub import (BenchWindow, CalibrationWindow, DatasetWindow,
                                    PredictWindow, Study2Window)
        from daphnia.ui.shell import CONTROLS_WIDTH

        for factory in (CalibrationWindow, BenchWindow, DatasetWindow,
                        PredictWindow, Study2Window, MainWindow):
            window = factory()
            self.addCleanup(window.close)
            window.resize(1200, 760)
            window.show()
            self.app.processEvents()

            pages = getattr(window, "pages", None) or {"page": window.page}
            for name, page in pages.items():
                panel = getattr(page, "controls", None)
                if panel is None:               # the hub has no panel at all
                    continue
                stacked = getattr(window, "stackedWidget", None)
                if stacked is not None:         # in the Study shell, lay this one out
                    stacked.setCurrentWidget(page)
                    self.app.processEvents()
                with self.subTest(window=factory.__name__, page=name):
                    scroll = panel.findChild(QScrollArea, "controls_scroll")
                    self.assertIsNotNone(scroll, "the panel lost its scroll area")
                    expected = getattr(type(page), "PANEL_WIDTH", CONTROLS_WIDTH)
                    self.assertEqual(
                        panel.width(), expected,
                        f"{factory.__name__}.{name}: the panel is {panel.width()} px, "
                        f"not the {expected} px this screen states")
                    self.assertLessEqual(
                        scroll.widget().minimumSizeHint().width(),
                        scroll.viewport().width(),
                        f"{factory.__name__}.{name}: the panel clips its own content "
                        f"- widen it (see CONTROLS_WIDTH, or PANEL_WIDTH here)")

    def test_stream_and_image_panels_share_the_same_floor(self):
        from daphnia.ui.widgets import ImageCanvas, StreamView
        self.assertEqual(StreamView().minimumSize(), ImageCanvas().minimumSize())

    def test_the_view_menu_is_the_canvas_route_on_every_canvas_screen(self):
        """The view actions are **menu items** now, not buttons over the picture.

        The window's *View* menu acts on the canvas **on screen** (zoom, fit, the hand
        tool, full screen), and nothing of it is drawn on the image (`X-N13`).
        """
        for route in ("calibration", "annotate"):
            page = self.window.pages[route]
            self.window._mount_view_menu(page)
            menu = self.window.header_bar._menus["View"]
            titles = [action.text() for action in menu.actions() if action.text()]
            self.assertEqual(titles, ["Zoom in", "Zoom out", "Fit the image",
                                      "Pan (hand tool)", "Full screen"], route)
            self.assertEqual(page.canvas.findChildren(QToolButton), [],
                             f"{route}: a tool button is still drawn on the image")
            # Nothing duplicates them in the panel any more.
            for name in ("btn_zoom_in", "btn_zoom_out", "btn_zoom_fit"):
                self.assertFalse(hasattr(page, name), f"{route}.{name}")


class HubTests(ScreenTestCase):
    """The hub: one tile per surface, and the window each tile opens."""

    def setUp(self):
        self._clear_tables()

    def _hub(self):
        """A hub that is always closed again, so no window outlives the test."""
        hub = HubWindow()
        self.addCleanup(hub.close)
        return hub

    def test_the_hub_offers_every_surface_as_a_tile(self):
        hub = self._hub()
        self.assertEqual(set(hub.home.tiles), set(surfaces.keys()))
        for key in surfaces.keys():
            self.assertTrue(hasattr(hub.home, f"open_{key}"), key)

    def test_the_study_tile_opens_the_study_shell(self):
        """The button, not the method: the tile is wired to the launcher."""
        hub = self._hub()
        hub.home.open_study.click()
        self.assertIsInstance(hub._open["study"], MainWindow)

    def test_the_dataset_tile_opens_the_dataset_window(self):
        hub = self._hub()
        hub.home.open_dataset.click()
        self.assertIsInstance(hub._open["dataset"], DatasetWindow)

    def test_the_predict_tile_opens_the_predict_window(self):
        """The new surface is offered by the hub the moment the registry has it."""
        hub = self._hub()
        hub.home.open_predict.click()
        window = hub._open["predict"]
        self.assertIsInstance(window, PredictWindow)
        window.close()

    def test_opening_a_surface_twice_reuses_its_window(self):
        hub = self._hub()
        self.assertIs(hub.open_surface("dataset"), hub.open_surface("dataset"))

    def test_an_unknown_surface_opens_nothing(self):
        hub = self._hub()
        self.assertIsNone(hub.open_surface("nope"))

    def test_every_built_surface_reads_open(self):
        hub = self._hub()
        for key in surfaces.keys():
            self.assertEqual(hub.home.buttons[key].text(), "Open", key)

    def test_the_hub_reports_the_project(self):
        hub = self._hub()
        self.assertIn("Current project: none", hub.home.status_label.text())

    def test_the_hub_counts_the_surfaces(self):
        hub = self._hub()
        self.assertIn(
            f"{len(surfaces.SURFACES)} surfaces: {len(surfaces.available())} ready",
            hub.home.status_label.text())

    # -- the shared footer (the same bar in every window) ------------------
    def test_the_hub_shows_the_shared_status_footer(self):
        hub = self._hub()
        self.assertEqual(hub.status_footer.scale_label.objectName(), "status_scale_label")
        self.assertIn("no active preset", hub.status_footer.scale_label.text())

    def test_the_footer_reports_the_camera_state(self):
        """An open socket is not a picture, so the bar says which it is."""
        hub = self._hub()
        self.assertIn("not connected", hub.status_footer.camera_label.text().lower())
        self.assertEqual(hub.status_footer.btn_connect.text(), "Connect")

    def test_the_study_window_shows_the_same_footer(self):
        """One bar, every surface: that is what makes 'calibrate anywhere' true."""
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key="PROF_S", profile_name="s", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=0.783, uncertainty_pct=1.36)
        hub = self._hub()
        window = hub.open_surface("study")
        footer = window.status_footer
        self.assertIn("783.0 px/mm", footer.scale_label.text())
        self.assertIn("+/- 1.36%", footer.scale_label.text())
        window.close()

    def test_the_footer_reports_the_age_of_the_calibration(self):
        with connection() as conn:
            repository.register_calibration(
                conn, setup_key="PROF_S", profile_name="s", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=0.783, uncertainty_pct=1.36)
        hub = self._hub()
        self.assertIn("calibrated", hub.status_footer.scale_label.text())

    def test_the_footer_flags_a_calibration_that_may_have_moved(self):
        """Recalibrating is allowed at any point, so stale must be visible."""
        with connection() as conn:
            calibration_id = repository.register_calibration(
                conn, setup_key="PROF_S", profile_name="s", microscope_model="m",
                objective_magnification="2x", camera_device="c",
                px_per_um=0.783, uncertainty_pct=1.0)
            conn.execute(
                "UPDATE calibration_profiles SET last_verified_at = ?"
                " WHERE profile_id = ?", ("2000-01-01 00:00:00", calibration_id))
        hub = self._hub()
        self.assertIn("RE-VERIFY", hub.status_footer.scale_label.text())

    def test_the_footer_calibrates_from_the_hub_without_a_study(self):
        """'Calibrate at any point' includes the hub, with no study at all."""
        hub = self._hub()
        hub.status_footer.btn_calibrate.click()
        self.assertEqual(set(hub._open), {"optics"})


class HeaderBarTests(ScreenTestCase):
    """Every window's header: the menus, and the facts that are not measurements.

    The canvas cards carry the numbers (`X-N12`); the header carries what belongs to the
    **window** - which frame, which model, the last message - so the two never compete for
    the photograph. It is chrome, hidden by name in full screen.
    """

    def setUp(self):
        self._clear_tables()

    def _window(self, factory):
        window = factory()
        self.addCleanup(window.close)
        window.show()
        self.app.processEvents()
        return window

    def _menu(self, window, name):
        """``{action text: action}`` for one of the header's menus."""
        return {action.text(): action
                for action in window.header_bar._menus[name].actions()
                if action.text()}

    def test_every_window_carries_the_header(self):
        for factory in (HubWindow, CalibrationWindow, BenchWindow, DatasetWindow,
                        PredictWindow, Study2Window, MainWindow):
            with self.subTest(window=factory.__name__):
                window = self._window(factory)
                self.assertEqual(window.header_bar.objectName(), "header_bar")
                # In the window, not in the system menu: a header macOS moves into the
                # global bar is a header nobody sees.
                self.assertFalse(window.header_bar.menu_bar.isNativeMenuBar())
                self.assertTrue(window.header_bar.title.text())

    def test_the_header_is_chrome_full_screen_hides(self):
        """`WINDOW_CHROME` hides `header_bar` - and puts it back."""
        self.assertIn("header_bar", WINDOW_CHROME)
        window = self._window(BenchWindow)
        page = window.page
        self.assertTrue(window.header_bar.isVisible())

        self.assertTrue(page.workspace.enter_full_screen())
        self.app.processEvents()
        self.assertFalse(window.header_bar.isVisible())

        self.assertTrue(page.workspace.leave_full_screen())
        self.app.processEvents()
        self.assertTrue(window.header_bar.isVisible())

    def test_the_facts_are_the_pages_and_they_leave_the_canvas(self):
        """One home per fact: what is not a measurement is in the window, never on the image."""
        window = self._window(PredictWindow)
        page = window.page
        page.source_label.setText("File: clip.mp4")
        page.status_label.setText("Ready.")
        page._index = 12                      # a video frame, so the fact moves
        page._refresh_header()

        self.assertEqual([kpi.key for kpi in window.header_bar.facts],
                         ["model", "status"])
        self.assertEqual(page.canvas.findChildren(QWidget), [],
                         "a fact was drawn over the picture")

    def test_the_source_menu_is_where_the_camera_is_connected(self):
        """The camera **string** left the bar; the *Source* menu is the way in.

        The entry's check mark *is* the state - a connected stream is checked - and the
        footer spells the detail out. Menu and footer drive one shared connection, so two
        ways to connect cannot disagree about the socket.
        """
        for factory in (HubWindow, BenchWindow, PredictWindow):
            with self.subTest(window=factory.__name__):
                window = self._window(factory)
                entry = window.header_bar.connect_action
                self.assertTrue(entry.isCheckable())
                self.assertIn("live feed", entry.text().lower())
                self.assertFalse(entry.isChecked())      # nothing to connect to here
                self.assertIs(window.header_bar.service, window.status_footer.service)
                self.assertIn(SOURCE_MENU, window.header_bar.menus())

    def test_the_pages_own_source_entries_sit_in_that_menu(self):
        """`Connect` and `Open a video...` are one decision in one place (`X-N13`)."""
        window = self._window(PredictWindow)
        menu = window.header_bar._menus[SOURCE_MENU]
        titles = [action.text() for action in menu.actions() if action.text()]
        self.assertEqual(titles, ["Connect the live feed", "Use the live feed",
                                  "Open a video file...", "Open a single image..."])
        self.assertNotIn(SOURCE_MENU, window.header_bar.page_menus)

    def test_the_model_menus_are_on_every_window_whose_screen_uses_one(self):
        """Both model menus are on every window that can **use** a model, with their entries.

        Which checkpoints appear is the **registry's** answer, filled from the training
        reports (`models.sync_from_reports`) - a checkpoint the lab trained is offered
        whether or not anybody has opened it in the app yet.

        The Bench is the exception, and is checked as one below: its screen displays
        measurements and runs no model (`SR-BENCH-017`), so a checkpoint menu there would be a
        control with nothing behind it.
        """
        for factory in (HubWindow, DatasetWindow, PredictWindow, MainWindow):
            with self.subTest(window=factory.__name__):
                window = self._window(factory)
                self.assertIsNotNone(window.model_menu)
                titles = window.header_bar.menus()
                # Not merely created: **shown**, because the window fills them with entries
                # (`hide_empty_menus` hides a menu with nothing behind it).
                self.assertTrue(window.header_bar._menus[DETECTOR_MENU]
                                .menuAction().isVisible())
                self.assertTrue(window.header_bar._menus[POSE_MENU]
                                .menuAction().isVisible())
                self.assertIn(DETECTOR_MENU, titles)
                self.assertIn(POSE_MENU, titles)
                self.assertNotIn(MODELS_MENU_GONE, titles)
                detector_titles = [action.text() for action in
                                   window.header_bar._menus[DETECTOR_MENU].actions()
                                   if action.text()]
                pose_titles = [action.text() for action in
                               window.header_bar._menus[POSE_MENU].actions()
                               if action.text()]
                self.assertIn("Open a detector checkpoint...", detector_titles)
                self.assertIn("Open a pose model checkpoint...", pose_titles)
                # Each menu carries its own half and nothing of the other's.
                self.assertNotIn("Open a pose model checkpoint...", detector_titles)
                self.assertNotIn("Open a detector checkpoint...", pose_titles)

    def test_the_bench_window_offers_no_model_menu(self):
        """The surface that only displays measurements carries no checkpoint menus.

        `OFFERS_MODELS` is the window's own statement of what its screen does: the Bench runs
        no model, so *Detector (RTMDet)* / *Pose (RTMPose)* are absent rather than present and
        inert, and the page exposes nothing a menu could hand a checkpoint to.
        """
        window = self._window(BenchWindow)
        self.assertFalse(BenchWindow.OFFERS_MODELS)
        self.assertIsNone(window.model_menu)
        # The bar offers them nowhere: an empty menu is **hidden** rather than shown with
        # nothing behind it (`hide_empty_menus`), so the operator never reads a checkpoint
        # menu that this screen would not act on.
        self.assertFalse(window.header_bar._menus[DETECTOR_MENU].menuAction().isVisible())
        self.assertFalse(window.header_bar._menus[POSE_MENU].menuAction().isVisible())
        self.assertNotIn(MODELS_MENU_GONE, window.header_bar.menus())
        self.assertFalse(hasattr(window.page, "adopt_models"))

    def test_the_model_menus_list_what_is_in_the_models_folder(self):
        """A checkpoint in the models folder is offered by name, registered or not.

        Clicking one **loads it and registers it** (verified against real checkpoints by
        hand); with a stand-in file that is not a checkpoint, what must happen is the same
        thing that happens for any broken file: a **named failure**, and no state change
        (`mlContext.md` law 8).
        """
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        (directory / "rtmdet_fresh.pt").write_bytes(b"not really a checkpoint")
        window = self._window(DatasetWindow)          # a screen that uses a model
        window.model_menu.directory = directory
        window.model_menu.refresh()

        actions = {action.text(): action
                   for action in window.header_bar._menus[DETECTOR_MENU].actions()
                   if action.text()}
        self.assertIn("rtmdet_fresh.pt", actions)
        self.assertIn("not in the registry yet", actions["rtmdet_fresh.pt"].toolTip())

        failures = []
        window.model_menu.failed.connect(failures.append)
        actions["rtmdet_fresh.pt"].trigger()
        self.assertTrue(failures, "a checkpoint that will not load must say so")
        self.assertIn("rtmdet_fresh.pt", failures[0])

    def test_loading_from_the_menu_switches_the_page_model_on(self):
        """The complaint: the menu loaded a model and the page stayed off, so nothing appeared.

        Tested on the **Dataset** screen, whose page can use a model (the Bench cannot: it
        displays measurements and carries no model menu at all).
        """
        from daphnia.ui.pages.dataset import Dataset
        with connection() as conn:
            models.register(conn, "detector", "/tmp/rtmdet_stub.pt")
        page = Dataset()
        self.assertFalse(page.workspace.action("model").isChecked())
        self.assertTrue(page.adopt_models(detector=object()))
        self.assertTrue(page.workspace.action("model").isChecked(),
                        "a model taken from the menu is a decision to use it")

        """*Activate it in the database* is visible in the menu, on every window."""
        with connection() as conn:
            registered = models.register(conn, "detector", "/tmp/rtmdet_stub.pt",
                                         preset="tiny", classes=("daphnia",))
        window = self._window(DatasetWindow)
        actions = {action.text(): action
                   for action in window.header_bar._menus[DETECTOR_MENU].actions()
                   if action.text()}

        self.assertIn("Load the active detector (RTMDet)", actions)
        choice = [action for text, action in actions.items()
                  if text.startswith("rtmdet_stub.pt")][0]
        self.assertTrue(choice.isCheckable())
        self.assertTrue(choice.isChecked())
        self.assertIn("daphnia", choice.toolTip())
        # ... and activating another one is one click: the registry follows.
        with connection() as conn:
            self.assertEqual(models.active(conn, "detector").path, registered.path)

    def test_the_numbers_are_in_the_panel_and_the_context_in_the_header(self):
        """`X-N13`: the length is read in the panel; the header carries the window's facts.

        The canvas holds the page's own **draw tools** and nothing else (`X-N20`): the card
        that used to repeat a number over the specimen is gone (2026-10-18).
        """
        window = self._window(BenchWindow)
        page = window.page
        page.status_label.setText("Ready to measure.")
        page._refresh_header()

        self.assertEqual([kpi.key for kpi in window.header_bar.facts],
                         ["source", "status"])
        self.assertEqual([child.objectName() for child in page.canvas.children()
                          if isinstance(child, QWidget)], ["canvas_draw_bar"])

    def test_only_the_page_on_screen_may_write_the_header(self):
        """The study shell swaps pages; a hidden step must not write the bar."""
        window = self._window(MainWindow)
        annotate = window.pages["annotate"]
        calibration = window.pages["calibration"]
        window._header_source = annotate.workspace

        annotate.workspace.set_header([Kpi("frame", "1 / 40", label="Frame")])
        self.assertEqual([kpi.key for kpi in window.header_bar.facts], ["frame"])

        calibration.workspace.set_header([Kpi("grid", "hook", label="Grid")])
        self.assertEqual([kpi.key for kpi in window.header_bar.facts], ["frame"],
                         "a step that is not on screen wrote the header")

    def test_the_go_menu_opens_a_surface_and_comes_back(self):
        window = self._window(BenchWindow)
        actions = self._menu(window, "Go")
        self.assertNotIn("Bench", actions)          # never offers to open itself
        self.assertIn("Home - choose a surface", actions)

        actions["Predict"].trigger()
        self.app.processEvents()
        predict = window._open["predict"]
        self.assertIsInstance(predict, PredictWindow)
        self.assertTrue(predict.isVisible())
        self.assertFalse(window.isVisible())        # one window at a time

        predict.close()                             # WA_DeleteOnClose
        for _ in range(20):
            self.app.processEvents()
        self.assertIsNone(window._open.get("predict"))
        self.assertTrue(window.isVisible())

    def test_the_view_menu_mirrors_the_tools_and_follows_full_screen(self):
        window = self._window(BenchWindow)
        page = window.page
        page.canvas.set_source(np.full((240, 320, 3), 255, dtype=np.uint8))
        self.app.processEvents()
        actions = self._menu(window, "View")
        self.assertEqual(list(actions),
                         ["Zoom in", "Zoom out", "Fit the image", "Pan (hand tool)",
                          "Full screen"])

        zoom = page.canvas.zoom
        actions["Zoom in"].trigger()
        self.assertGreater(page.canvas.zoom, zoom)

        actions["Pan (hand tool)"].trigger()
        self.assertTrue(page.canvas.pan_mode)
        actions["Pan (hand tool)"].trigger()
        self.assertFalse(page.canvas.pan_mode)

        actions["Full screen"].trigger()
        self.app.processEvents()
        self.assertTrue(page.workspace.full_screen_active)
        self.assertTrue(actions["Full screen"].isChecked())

        page.workspace.leave_full_screen()          # the way Esc leaves it
        self.app.processEvents()
        self.assertFalse(actions["Full screen"].isChecked())

    def test_a_long_value_elides_instead_of_widening_the_bar(self):
        bar = HeaderBar("Test")
        bar.resize(300, 30)
        bar.set_facts([Kpi("source", "x" * 400, label="Source")])
        bar.show()
        self.addCleanup(bar.hide)
        self.app.processEvents()

        value = bar.fact("source").value
        self.assertEqual(value.full_text(), "x" * 400)      # nothing is thrown away
        self.assertLessEqual(value.width(), bar.width())    # and nothing overflows
        elided = value.fontMetrics().elidedText(value.full_text(), Qt.ElideRight,
                                                value.width())
        self.assertTrue(elided.endswith("…"), elided)

    def test_a_header_fact_must_say_what_it_names(self):
        """A bare value in a bar is a number nobody can name - refused, not drawn."""
        with self.assertRaises(ValueError) as raised:
            HeaderBar().set_facts([Kpi("bare", "12")])
        self.assertIn("heading", str(raised.exception))


class WindowFooterTests(ScreenTestCase):
    """Every window of the application shows the shared bar - not just the hub.

    Two facts about the machine decide what a number means: is the camera delivering
    frames, and which scale is in force. They belong on every window, and the screens
    already point at the bar (*"press Calibrate in the bar below"* on Bench and
    Calibrate) - so this is a contract, not a decoration.
    """

    def setUp(self):
        self._clear_tables()

    def _windows(self):
        windows = [factory() for factory in (HubWindow, CalibrationWindow, BenchWindow,
                                             DatasetWindow, PredictWindow, Study2Window,
                                             MainWindow)]
        # The plan window belongs to the family too: it is what a launch point shows
        # when no window is registered for it. (Which surface it describes does not
        # matter here - the footer is the point.)
        windows.append(SurfacePlanWindow(surfaces.SURFACES[-1]))
        for window in windows:
            self.addCleanup(window.close)
        return windows

    def test_every_window_carries_the_footer(self):
        for window in self._windows():
            with self.subTest(window=type(window).__name__):
                footer = window.status_footer
                self.assertIsInstance(footer, StatusFooter)
                self.assertEqual(footer.camera_label.objectName(), "camera_state_label")
                self.assertIn("Optical scale", footer.scale_label.text())

    def test_the_bar_is_the_name_full_screen_hides(self):
        """`WINDOW_CHROME` hides `status_footer`, so the name is part of the contract."""
        self.assertIn("status_footer", WINDOW_CHROME)
        for window in self._windows():
            with self.subTest(window=type(window).__name__):
                self.assertEqual(window.status_footer.objectName(), "status_footer")

    def test_the_optics_window_does_not_offer_to_open_itself(self):
        """It *is* the calibration; every other window offers the way to it."""
        for window in self._windows():
            with self.subTest(window=type(window).__name__):
                offered = not window.status_footer.btn_calibrate.isHidden()
                self.assertEqual(offered, not isinstance(window, CalibrationWindow))

    def test_calibrate_hands_over_to_optics_and_comes_back(self):
        """Calibrating is a screen, not a dialog: this window steps aside and returns."""
        from daphnia.ui.hub import BenchWindow

        bench = BenchWindow()
        self.addCleanup(bench.close)
        bench.show()
        self.app.processEvents()

        bench.status_footer.btn_calibrate.click()
        self.app.processEvents()
        calibration = bench._calibration_window
        self.assertIsInstance(calibration, CalibrationWindow)
        self.assertTrue(calibration.isVisible())
        self.assertFalse(bench.isVisible())       # one window at a time, as in the hub

        calibration.close()                       # WA_DeleteOnClose
        for _ in range(20):
            self.app.processEvents()
        self.assertTrue(bench.isVisible(), "the window did not come back")
        self.assertIsNone(bench._calibration_window)

    def test_a_second_press_does_not_open_a_second_calibration_window(self):
        from daphnia.ui.hub import BenchWindow

        bench = BenchWindow()
        self.addCleanup(bench.close)
        first = bench.open_calibration()
        second = bench.open_calibration()
        self.assertIs(first, second)
        first.close()


class StatusFooterTests(ScreenTestCase):
    """The shared bar: camera state and scale, with both actions available."""

    def setUp(self):
        self._clear_tables()

    def _footer(self):
        """A footer over an isolated service, so no test touches the network."""
        service = StreamService(url="rtsp://test/cam", orientation="native")
        footer = StatusFooter(service=service)
        self.addCleanup(footer.deleteLater)
        return footer

    def test_it_reports_the_camera_as_not_connected(self):
        footer = self._footer()
        self.assertIn("not connected", footer.camera_label.text().lower())
        self.assertEqual(footer.btn_connect.text(), "Connect")

    def test_it_distinguishes_silence_from_a_picture(self):
        footer = self._footer()
        footer.service._on_connected()
        self.assertIn("no frames", footer.camera_label.text().lower())
        self.assertEqual(footer.btn_connect.text(), "Disconnect")

        footer.service._on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(footer.camera_label.text(), "Camera live")

    def test_the_connect_button_opens_the_stream_without_a_setup_step(self):
        """The whole point: the camera can be opened from any window."""
        footer = self._footer()
        started = []
        # No network in tests - only the intent is asserted.
        footer.service.start = lambda *a, **k: started.append("start")
        footer.btn_connect.click()
        self.assertEqual(started, ["start"])

    def test_the_button_offers_the_way_out_of_a_connection(self):
        footer = self._footer()
        stopped = []
        footer.service.stop = lambda *a, **k: stopped.append("stop")
        footer.service._state = StreamService.CONNECTING   # state is a property
        footer.btn_connect.click()
        self.assertEqual(stopped, ["stop"])


if __name__ == "__main__":
    unittest.main()

