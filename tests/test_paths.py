"""Project paths: where the code, the assets and the operator's data live.

The distinction these tests protect is the one a **packaged** app depends on:
everything the app *writes* hangs off :data:`daphnia.paths.DATA_ROOT`, and one
environment variable moves all of it - so a `.exe` in Program Files or a `.app`
inside a signed disk image never has to write into its own read-only folder.
"""

import importlib
import os
import tempfile
import unittest
from pathlib import Path

from daphnia import paths

#: The variables these tests set, so each one is put back exactly as it was.
VARIABLES = ("DAPHNIA_DATA_DIR", "DAPHNIA_DB")


class DataRootTests(unittest.TestCase):
    """`DAPHNIA_DATA_DIR` moves the generated state; the code stays where it is."""

    def setUp(self):
        self._previous = {name: os.environ.get(name) for name in VARIABLES}

        def restore():
            for name, value in self._previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            importlib.reload(paths)          # back to the checkout's own layout

        self.addCleanup(restore)

    def _reload(self, **env):
        for name in VARIABLES:
            os.environ.pop(name, None)
        os.environ.update(env)
        return importlib.reload(paths)

    def test_the_default_root_is_the_checkout(self):
        module = self._reload()
        self.assertEqual(module.DATA_ROOT, module.PROJECT_ROOT)
        self.assertEqual(module.DB_PATH, module.PROJECT_ROOT / "zoo.db")
        self.assertEqual(module.RECORDINGS_DIR,
                         module.PROJECT_ROOT / "recordings")

    def test_the_data_dir_moves_every_generated_path(self):
        """The pictures, the weights and the database travel together."""
        folder = "/tmp/daphniascope-data"
        module = self._reload(DAPHNIA_DATA_DIR=folder)
        for path in (module.DB_PATH, module.RECORDINGS_DIR, module.DATASETS_DIR,
                     module.MODELS_DIR, module.CALIBRATION_IMAGES_DIR,
                     module.RAW_DIR):
            with self.subTest(path=str(path)):
                self.assertTrue(str(path).startswith(folder), path)
        # ... and the *code's* resources do not: the UI files are shipped with the app.
        self.assertEqual(module.UI_DIR, module.PROJECT_ROOT / "assets" / "ui")
        self.assertEqual(module.SCHEMA_PATH,
                         module.PACKAGE_DIR / "db" / "schema.sql")

    def test_the_database_can_still_be_pointed_at_a_single_file(self):
        """`DAPHNIA_DB` is the older, narrower override and it still wins."""
        module = self._reload(DAPHNIA_DATA_DIR="/tmp/data",
                              DAPHNIA_DB="/tmp/one.db")
        from daphnia.db.connection import default_db_path
        self.assertEqual(default_db_path(), Path("/tmp/one.db"))
        # The data root still governs the frames and the weights.
        self.assertTrue(str(module.RECORDINGS_DIR).startswith("/tmp/data"))

    def test_a_packaged_app_can_be_told_where_its_state_lives(self):
        """`ensure_runtime_dirs()` creates the tree the app needs, wherever it is."""
        with tempfile.TemporaryDirectory() as folder:
            module = self._reload(DAPHNIA_DATA_DIR=folder)
            module.ensure_runtime_dirs()
            for directory in (module.DATASETS_DIR, module.RECORDINGS_DIR,
                              module.MODELS_DIR, module.CALIBRATION_IMAGES_DIR):
                with self.subTest(directory=str(directory)):
                    self.assertTrue(directory.is_dir(), directory)


if __name__ == "__main__":
    unittest.main()
