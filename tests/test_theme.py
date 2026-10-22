"""The unified theme: tokens, both schemes, the palette, and the stylesheet.

This is the test that keeps "some screens follow the system setting and some do not"
from coming back. It checks the three things that made the old theme a patchwork:

* every scheme defines the *same* tokens, so a colour can never be missing from one
  of them (the old stylesheet sprayed light colours onto a dark platform palette);
* **no widget class the application uses is left out** of the template, so nothing
  falls through to whatever the desktop hands Qt (group boxes, tables, lists, text
  browsers and message boxes used to);
* the template refers to no image **files** - the old one pointed its check marks and
  arrows at Adwaita icons that only exist on Linux, which is why the tick and the
  combo arrow were simply missing on macOS.
"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

from daphnia import paths
from daphnia.ui import theme

#: The widget classes this application puts on screen (from ``assets/ui/*.ui`` plus
#: the ones the pages build in code). Every one of them has to be addressed by the
#: template or by the palette - this list is what "the whole application" means.
WIDGETS = (
    "QMainWindow", "QDialog", "QLabel", "QPushButton", "QToolButton", "QLineEdit",
    "QTextEdit", "QPlainTextEdit", "QTextBrowser", "QSpinBox", "QDoubleSpinBox",
    "QComboBox", "QCheckBox", "QRadioButton", "QGroupBox", "QListWidget",
    "QTableWidget", "QHeaderView", "QScrollArea", "QScrollBar", "QSlider",
    "QProgressBar", "QMenuBar", "QMenu", "QTabWidget", "QStatusBar", "QToolTip",
    "QStackedWidget",
)

#: The application's own widgets, keyed on ``objectName``.
OBJECT_NAMES = (
    "step_bar", "step_label", "step_hint", "controls_panel", "controls_scroll",
    "body_scroll", "status_footer", "camera_state_label", "status_scale_label",
    "header_bar", "header_menu", "header_title", "header_facts", "header_fact",
    "header_fact_label", "header_fact_value", "header_fact_flag",
    "canvas_draw_bar", "label_verdict",
    "stream_view", "image_viewer", "image_viewer_text",
    "surface_plan", "tile_tagline", "tile_meta", "tile_status", "visual_area",
    "well_plate", "well_legend_label",
)


def _contrast(first: str, second: str) -> float:
    """WCAG contrast ratio between two ``#rrggbb`` colours."""
    def luminance(value):
        channels = [int(value[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
        linear = [(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
                  for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    a, b = luminance(first), luminance(second)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)



class TokenTests(unittest.TestCase):
    """One token set, two schemes, nothing missing."""

    def test_the_two_schemes_define_the_same_tokens(self):
        light = set(theme.tokens("light"))
        dark = set(theme.tokens("dark"))
        self.assertEqual(light, dark,
                         "a token missing from one scheme is a hole in the other - "
                         "add it to LIGHT and DARK")
        self.assertTrue(light)

    def test_the_tools_over_a_photograph_do_not_follow_the_scheme(self):
        """They lie on the picture, not on a surface: identical in both schemes."""
        for name in theme.PHOTO_OVERLAY:
            self.assertEqual(theme.color(name, "light"), theme.color(name, "dark"),
                             f"{name} sits over a photograph and must not change")

    def test_an_unknown_token_names_itself_and_the_place_to_fix_it(self):
        with self.assertRaises(theme.ThemeError) as raised:
            theme.color("not_a_token", "dark")
        message = str(raised.exception)
        self.assertIn("not_a_token", message)
        self.assertIn("theme.py", message)

    def test_every_token_is_a_colour_a_painter_could_use(self):
        """Except the translucent overlay ones, which only a stylesheet can express."""
        for scheme in theme.SCHEMES:
            for name in theme.tokens(scheme):
                if theme.PHOTO_OVERLAY.get(name, "").startswith("rgba"):
                    continue
                with self.subTest(scheme=scheme, token=name):
                    self.assertTrue(theme.qcolor(name, scheme).isValid())

    def test_an_rgba_token_is_refused_by_the_painter_with_its_name(self):
        with self.assertRaises(theme.ThemeError) as raised:
            theme.qcolor("overlay_bg", "dark")
        self.assertIn("overlay_bg", str(raised.exception))

    def test_text_has_readable_contrast_in_both_schemes(self):
        """The bug being fixed was #272727 text on a #323232 window (1.1:1)."""
        for scheme in theme.SCHEMES:
            for background, foreground in (("window", "text"), ("surface", "text"),
                                           ("surface", "text_muted"),
                                           ("surface_alt", "text")):
                with self.subTest(scheme=scheme, background=background,
                                  foreground=foreground):
                    ratio = _contrast(theme.color(background, scheme),
                                      theme.color(foreground, scheme))
                    self.assertGreaterEqual(
                        ratio, 4.0,
                        f"{scheme}: {foreground} on {background} is {ratio:.2f}:1")


class SchemeTests(unittest.TestCase):
    """Which scheme gets applied, and what the environment may say."""

    def setUp(self):
        self._previous = os.environ.get(theme.ENV_VAR)
        os.environ.pop(theme.ENV_VAR, None)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._previous is None:
            os.environ.pop(theme.ENV_VAR, None)
        else:
            os.environ[theme.ENV_VAR] = self._previous

    def test_an_explicit_request_wins(self):
        self.assertEqual(theme.resolve_scheme("dark"), "dark")
        self.assertEqual(theme.resolve_scheme("LIGHT"), "light")

    def test_the_environment_pins_the_scheme_when_nothing_is_asked_for(self):
        os.environ[theme.ENV_VAR] = "dark"
        self.assertEqual(theme.resolve_scheme(), "dark")
        self.assertEqual(theme.resolve_scheme("light"), "light")

    def test_system_means_the_desktop(self):
        os.environ[theme.ENV_VAR] = "dark"
        self.assertEqual(theme.resolve_scheme(theme.SYSTEM), theme.system_scheme())

    def test_a_typo_is_an_error_not_the_other_theme(self):
        for value in ("drak", "night", "1", "dark mode"):
            with self.subTest(value=value):
                os.environ[theme.ENV_VAR] = value
                with self.assertRaises(theme.ThemeError) as raised:
                    theme.resolve_scheme()
                self.assertIn(theme.ENV_VAR, str(raised.exception))
                self.assertIn("light, dark", str(raised.exception))

    def test_the_system_scheme_is_one_of_the_two(self):
        self.assertIn(theme.system_scheme(), theme.SCHEMES)


class StylesheetTests(unittest.TestCase):
    """The template: substituted, complete, and free of missing icon files."""

    def test_both_schemes_render_a_stylesheet_without_a_placeholder_left(self):
        for scheme in theme.SCHEMES:
            with self.subTest(scheme=scheme):
                stylesheet = theme.build_stylesheet(scheme)
                self.assertNotRegex(stylesheet, r"%[a-z_]+%")
                self.assertGreater(len(stylesheet), 2000)

    def test_the_two_schemes_are_not_the_same_stylesheet(self):
        self.assertNotEqual(theme.build_stylesheet("light"),
                            theme.build_stylesheet("dark"))

    def test_no_stylesheet_points_at_an_image_file(self):
        """The old theme's check mark was an Adwaita PNG that macOS does not have."""
        for scheme in theme.SCHEMES:
            with self.subTest(scheme=scheme):
                self.assertNotIn("url(", theme.build_stylesheet(scheme))

    def test_the_indicator_is_left_to_the_style(self):
        """Styling `::indicator` is what removed the tick; it has to stay unstyled."""
        for scheme in theme.SCHEMES:
            with self.subTest(scheme=scheme):
                self.assertNotIn("::indicator", theme.build_stylesheet(scheme))

    def test_every_widget_the_application_uses_is_styled(self):
        stylesheet = " ".join(theme.build_stylesheet(s) for s in theme.SCHEMES)
        for widget in WIDGETS:
            with self.subTest(widget=widget):
                self.assertIn(widget, stylesheet,
                              f"{widget} is not themed - add a rule to the template")

    def test_every_object_name_of_our_own_widgets_is_styled(self):
        stylesheet = theme.build_stylesheet("dark")
        for name in OBJECT_NAMES:
            with self.subTest(object_name=name):
                self.assertIn(f"#{name}", stylesheet,
                              f"#{name} is not themed - add a rule to the template")

    def test_a_template_that_uses_an_undefined_token_is_reported(self):
        broken = self._temporary("broken.qss")
        broken.write_text("QWidget { color: %no_such_colour%; }", encoding="utf-8")
        with self.assertRaises(theme.ThemeError) as raised:
            theme.build_stylesheet("dark", template_path=broken)
        message = str(raised.exception)
        self.assertIn("no_such_colour", message)
        self.assertIn("dark", message)

    def test_a_template_that_uses_two_undefined_tokens_names_both(self):
        broken = self._temporary("two.qss")
        broken.write_text("%one% %two%", encoding="utf-8")
        with self.assertRaises(theme.ThemeError) as raised:
            theme.build_stylesheet("light", template_path=broken)
        self.assertIn("one", str(raised.exception))
        self.assertIn("two", str(raised.exception))

    def test_a_missing_template_names_the_file_and_the_fix(self):
        missing = self._temporary("gone.qss")
        with self.assertRaises(theme.ThemeError) as raised:
            theme.build_stylesheet("light", template_path=missing)
        self.assertIn("gone.qss", str(raised.exception))

    def _temporary(self, name) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="daphnia-theme-test-"))
        return directory / name


class PaletteTests(unittest.TestCase):
    """The half of the theme the stylesheet does not reach."""

    def test_the_palette_carries_the_token_colours(self):
        for scheme in theme.SCHEMES:
            palette = theme.build_palette(scheme)
            with self.subTest(scheme=scheme):
                for role, token in ((QPalette.ColorRole.Window, "window"),
                                    (QPalette.ColorRole.Base, "surface_alt"),
                                    (QPalette.ColorRole.Text, "text"),
                                    (QPalette.ColorRole.ButtonText, "text"),
                                    (QPalette.ColorRole.Highlight, "accent")):
                    self.assertEqual(palette.color(role).name(),
                                     theme.color(token, scheme))

    def test_a_disabled_widget_does_not_keep_the_full_text_colour(self):
        for scheme in theme.SCHEMES:
            palette = theme.build_palette(scheme)
            with self.subTest(scheme=scheme):
                disabled = palette.color(QPalette.ColorGroup.Disabled,
                                         QPalette.ColorRole.Text)
                self.assertEqual(disabled.name(), theme.color("text_disabled", scheme))

    def test_a_dark_palette_is_dark_and_a_light_one_is_light(self):
        self.assertLess(theme.build_palette("dark").color(
            QPalette.ColorRole.Window).lightness(), 128)
        self.assertGreater(theme.build_palette("light").color(
            QPalette.ColorRole.Window).lightness(), 128)



class AppliedThemeTests(unittest.TestCase):
    """What ``apply_theme`` does to a running application."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        theme.apply_theme(self.app, theme.SYSTEM)

    def test_applying_a_scheme_sets_the_palette_and_the_stylesheet(self):
        for scheme in theme.SCHEMES:
            with self.subTest(scheme=scheme):
                theme.apply_theme(self.app, scheme)
                self.assertEqual(theme.active_scheme(), scheme)
                self.assertEqual(self.app.styleSheet(), theme.build_stylesheet(scheme))
                self.assertEqual(
                    self.app.palette().color(QPalette.ColorRole.Window).name(),
                    theme.color("window", scheme))

    def test_the_style_draws_what_a_stylesheet_cannot(self):
        """Fusion is what draws the check mark, the combo arrow and the spin buttons."""
        theme.apply_theme(self.app, "dark")
        # Qt wraps the style in a stylesheet proxy the moment a stylesheet is in
        # force (that proxy reports no name of its own), so the base style is only
        # observable with the sheet out of the way - which is also proof that the
        # stylesheet really is applied.
        self.assertTrue(self.app.styleSheet())
        self.app.setStyleSheet("")
        self.assertEqual(self.app.style().objectName().lower(),
                         theme.STYLE.lower())
        self.assertIn(theme.STYLE, QStyleFactory.keys())

    def test_switching_scheme_switches_the_other_half_too(self):
        theme.apply_theme(self.app, "light")
        theme.apply_theme(self.app, "dark")
        self.assertEqual(theme.active_tokens()["text"], theme.DARK["text"])
        self.assertEqual(theme.color("text"), theme.color("text", "dark"))

    def test_a_pinned_scheme_ignores_a_desktop_change(self):
        theme.apply_theme(self.app, "light")
        theme._on_system_scheme_changed()       # the desktop switched appearance
        self.assertEqual(theme.active_scheme(), "light")

    def test_a_followed_scheme_takes_the_desktop_s_new_choice(self):
        theme.apply_theme(self.app, theme.SYSTEM)
        theme._on_system_scheme_changed()
        self.assertEqual(theme.active_scheme(), theme.system_scheme())

    def test_the_system_is_followed_once_per_application(self):
        theme.apply_theme(self.app, theme.SYSTEM)
        theme.apply_theme(self.app, theme.SYSTEM)
        self.assertTrue(getattr(self.app, "_daphnia_following_system_scheme", False))

    def test_an_unknown_scheme_is_refused(self):
        with self.assertRaises(theme.ThemeError):
            theme.apply_theme(self.app, "midnight")


class RepositoryTests(unittest.TestCase):
    """The template is a tracked file, and the old theme is gone for good."""

    def test_the_template_is_where_the_theme_looks_for_it(self):
        self.assertTrue((paths.THEME_DIR / theme.TEMPLATE_FILE).is_file())

    def test_the_superseded_theme_is_not_in_the_repository(self):
        """One theme system, or the next reader wonders which file is in force."""
        self.assertEqual(sorted(p.name for p in paths.THEME_DIR.glob("*.qss")),
                         [theme.TEMPLATE_FILE])

    def test_no_screen_colours_itself_with_an_inline_stylesheet(self):
        """One place decides colour: the theme.

        A widget-level ``setStyleSheet`` is a colour that survives a scheme change -
        which is how one screen becomes the odd one out. The image viewer and the
        stream panel used to be exactly that (``#101010`` and ``#121212`` by hand),
        so this is a regression guard, not a style preference.
        """
        ui_root = paths.PROJECT_ROOT / "daphnia" / "ui"
        offenders = sorted(
            path.relative_to(ui_root).as_posix()
            for path in ui_root.rglob("*.py")
            if path.name != "theme.py"
            and "setStyleSheet" in path.read_text(encoding="utf-8"))
        self.assertEqual(
            offenders, [],
            f"these modules colour a widget directly: {offenders} - put the colour in "
            f"daphnia/ui/theme.py (a token, or an objectName rule in the template)")


if __name__ == "__main__":
    unittest.main()

