"""Surfaces: the registry, the launch dispatch and the surface windows.

A surface is a launch point (see :mod:`daphnia.surfaces`). The registry is data, so
most of it is checkable without a display; only the windows need a QApplication.
"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daphnia import paths, surfaces
from daphnia.db import init_db
from daphnia.ui import app as launcher
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

#: Packages allowed to depend on Qt. Everything else must import headless, so it
#: stays usable from scripts and notebooks (SR-CORE-011).
QT_DEPENDENT = ("ui", "streaming")


class RegistryTests(unittest.TestCase):
    """What the hub renders and run.py dispatches on."""

    def test_keys_are_unique(self):
        self.assertEqual(len(set(surfaces.keys())), len(surfaces.SURFACES))

    def test_every_surface_is_fully_described(self):
        for surface in surfaces.SURFACES:
            with self.subTest(surface.key):
                self.assertTrue(surface.label)
                self.assertTrue(surface.tagline)
                self.assertTrue(surface.summary)
                self.assertTrue(surface.audience)
                self.assertTrue(surface.gate)
                self.assertTrue(surface.next_steps)
                self.assertIn(surface.status, surfaces.STATUSES)
                self.assertTrue(surface.status_label)

    def test_every_surface_traces_to_a_requirement_document(self):
        """A surface with no requirements is a wish, not a surface."""
        for surface in surfaces.SURFACES:
            with self.subTest(surface.key):
                doc = paths.PROJECT_ROOT / surface.requirements
                self.assertTrue(doc.exists(), surface.requirements)
                self.assertEqual(doc.name, f"SR-{surface.key.upper()}.md")
                self.assertIn(surface.key.upper(), doc.read_text(encoding="utf-8"))

    def test_available_and_planned_partition_the_registry(self):
        grouped = [s.key for s in surfaces.available() + surfaces.planned()]
        self.assertEqual(sorted(grouped), sorted(surfaces.keys()))
        self.assertEqual(len(grouped), len(set(grouped)))    # counted once each

    def test_these_are_the_surfaces_that_work(self):
        """Keep this honest: only these have a real window behind them.

        ``study2`` is the plate page - the acquisition half of the surface that will
        replace ``study`` - and it is listed next to it for exactly that reason.
        """
        self.assertEqual([s.key for s in surfaces.available()],
                         ["study", "study2", "bench", "dataset", "predict", "optics"])

    def test_get_is_lenient_about_case_and_whitespace(self):
        self.assertEqual(surfaces.get("  Study ").key, "study")

    def test_get_rejects_unknown_and_empty_keys(self):
        for value in ("nope", "", "  ", None):
            with self.subTest(value=value):
                self.assertIsNone(surfaces.get(value))

    def test_the_launch_point_is_the_command_that_opens_it(self):
        self.assertEqual(surfaces.get("bench").launch_point(),
                         "python run.py bench")


class PurityTests(unittest.TestCase):
    """The Qt-free guarantee that keeps the domain reusable (SR-CORE-011)."""

    def test_the_registry_imports_no_qt(self):
        source = Path(surfaces.__file__).read_text(encoding="utf-8")
        self.assertNotIn("PySide6", source)

    def test_only_ui_and_streaming_may_depend_on_qt(self):
        package = paths.PROJECT_ROOT / "daphnia"
        offenders = []
        for path in sorted(package.rglob("*.py")):
            relative = path.relative_to(package)
            if relative.parts[0] in QT_DEPENDENT:
                continue
            if "PySide6" in path.read_text(encoding="utf-8"):
                offenders.append(str(relative))
        self.assertEqual(offenders, [])


class LaunchTests(unittest.TestCase):
    """run.py's dispatch: a launch key becomes exactly one surface."""

    def test_no_argument_launches_the_hub(self):
        self.assertEqual(launcher.parse_launch([]), surfaces.HUB_KEY)
        self.assertEqual(launcher.parse_launch(None), surfaces.HUB_KEY)

    def test_a_surface_key_is_returned_lowercased(self):
        self.assertEqual(launcher.parse_launch(["Study"]), "study")

    def test_options_are_skipped(self):
        self.assertEqual(launcher.parse_launch(["--verbose", "bench"]), "bench")

    def test_help_arguments_are_recognised(self):
        for arg in ("-h", "--help", "help", "--list"):
            with self.subTest(arg=arg):
                self.assertIn(arg, launcher.HELP_ARGS)

    def test_usage_lists_every_launch_point(self):
        text = launcher.usage()
        for surface in surfaces.SURFACES:
            with self.subTest(surface.key):
                self.assertIn(f"python run.py {surface.key}", text)
        self.assertIn("open Home", text)


class WindowTests(unittest.TestCase):
    """Which window a launch key opens - never the wrong one."""

    @classmethod
    def setUpClass(cls):
        cls._previous_db = os.environ.get("DAPHNIA_DB")
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["DAPHNIA_DB"] = str(Path(cls._tmp.name) / "surfaces-test.db")
        init_db()
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        if cls._previous_db is None:
            os.environ.pop("DAPHNIA_DB", None)
        else:
            os.environ["DAPHNIA_DB"] = cls._previous_db

    def test_the_hub_key_opens_the_hub(self):
        window = launcher.window_for(surfaces.HUB_KEY)
        self.assertIsInstance(window, HubWindow)
        window.close()

    def test_a_built_surface_opens_its_shell(self):
        window = launcher.window_for("study")
        self.assertIsInstance(window, MainWindow)
        window.close()

    def test_the_optics_surface_opens_the_calibration_window(self):
        """Calibrating must be possible on its own, with no study and no phone."""
        window = launcher.window_for("optics")
        self.assertIsInstance(window, CalibrationWindow)
        self.assertIsNotNone(window.page)
        window.close()

    def test_the_plate_surface_opens_the_plate_window(self):
        """The plate page is its own surface: no step order, one plate at a time."""
        window = launcher.window_for("study2")
        self.assertIsInstance(window, Study2Window)
        self.assertIsNotNone(window.page)
        window.close()

    def test_the_bench_surface_opens_the_bench_window(self):
        """Measuring must be possible on its own, with no study at all."""
        window = launcher.window_for("bench")
        self.assertIsInstance(window, BenchWindow)
        self.assertIsNotNone(window.page)
        window.close()

    def test_the_dataset_surface_opens_the_dataset_window(self):
        """Labelling must be possible on its own, with no study at all."""
        window = launcher.window_for("dataset")
        self.assertIsInstance(window, DatasetWindow)
        self.assertIsNotNone(window.page)
        window.close()

    def test_the_predict_surface_opens_the_predict_window(self):
        """Watching the detector must be possible on its own, with no study."""
        window = launcher.window_for("predict")
        self.assertIsInstance(window, PredictWindow)
        self.assertIsNotNone(window.page)
        window.close()

    def test_a_planned_surface_opens_its_plan(self):
        """A surface with no window still renders its specification, never another's.

        Every surface has a window today, so the plan window is exercised directly
        with a registry entry - the guarantee (`SR-CORE-014`) must survive the next
        surface being added as `planned`.
        """
        window = SurfacePlanWindow(surfaces.get("dataset"))
        text = window.plan.toPlainText()
        self.assertIn("Dataset", text)
        self.assertIn("SR-DATASET.md", text)
        window.close()

    def test_an_unknown_key_opens_nothing(self):
        self.assertIsNone(launcher.window_for("nope"))


if __name__ == "__main__":
    unittest.main()

