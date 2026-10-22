"""Runtime loader for Qt Designer ``.ui`` files built on PySide6.

This is the PySide6 equivalent of ``PyQt5.uic.loadUi``: it attaches the widgets
declared in a ``.ui`` file to an existing instance and exposes every named child
as an attribute, so code such as::

    uic.loadUi('pages.ui', self)
    self.stackedWidget.setCurrentIndex(1)

can be written as::

    load_ui('pages.ui', self)
    self.stackedWidget.setCurrentIndex(1)
"""

from pathlib import Path

from PySide6.QtCore import QFile, QIODevice
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from daphnia import paths


def load_ui(ui_file: str, base_instance=None, ui_dir=None):
    """Populate ``base_instance`` from a ``.ui`` file in ``assets/ui``.

    Mirrors ``uic.loadUi(ui_file, base_instance)``: the loaded widgets are
    parented to ``base_instance`` and every widget with an ``objectName`` becomes
    an attribute of ``base_instance``.

    Omit ``base_instance`` to get the top-level widget instead (useful when a
    page wants to nest the loaded form itself).
    """
    path = Path(ui_dir or paths.UI_DIR) / ui_file

    loader = QUiLoader()
    handle = QFile(str(path))
    if not handle.open(QIODevice.ReadOnly):
        raise FileNotFoundError(f"Could not open UI file: {path}")
    try:
        loaded = loader.load(handle)
    finally:
        handle.close()

    if base_instance is None:
        return loaded

    # uic.loadUi also applies the .ui's top-level object name to the target.
    if not base_instance.objectName() and loaded.objectName():
        base_instance.setObjectName(loaded.objectName())

    if isinstance(loaded, QMainWindow) and isinstance(base_instance, QMainWindow):
        # A QMainWindow cannot simply be nested in a layout, so transfer the
        # loaded window's contents (central widget, menu bar, status bar) into
        # the existing window instead.
        if not base_instance.windowTitle() and loaded.windowTitle():
            base_instance.setWindowTitle(loaded.windowTitle())
        central = loaded.centralWidget()
        if central is not None:
            base_instance.setCentralWidget(central)
        if loaded.menuBar() is not None:
            base_instance.setMenuBar(loaded.menuBar())
        if loaded.statusBar() is not None:
            base_instance.setStatusBar(loaded.statusBar())
    else:
        # Let the loaded widget fill the whole base widget.
        layout = QVBoxLayout(base_instance)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(loaded)

    # Expose every named child widget as an attribute, like uic.loadUi does.
    for child in base_instance.findChildren(QWidget):
        name = child.objectName()
        if name:
            setattr(base_instance, name, child)

    return loaded
