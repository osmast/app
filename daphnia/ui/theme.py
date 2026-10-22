"""Application theming: one token set, two schemes, one look for every window.

The application used to load a single **light** stylesheet
(``assets/theme/MacOS.qss``, GTRONICK) on top of whatever palette the operating
system handed Qt. On macOS in Dark Mode that is not a theme, it is a patchwork:

* the stylesheet paints ``#272727`` label text onto the platform's ``#323232``
  window - about 1.1:1 contrast, i.e. invisible;
* every widget the stylesheet never mentions (``QGroupBox``, ``QTableWidget``,
  ``QListWidget``, ``QTextBrowser``, ``QMessageBox``, menus, tool tips) keeps the
  *system's* dark look, which is why half of every window went dark while the
  buttons and fields stayed light;
* the scroll-area and tab-widget blocks in that file are dark even in a light
  theme, and check boxes and combo boxes had **no tick and no arrow** - their
  icons pointed at Adwaita files that only exist on Linux.

The unified system has three moving parts, and no widget colours itself:

* **Tokens** (the dictionaries below) are the only place a colour is written down.
  ``light`` and ``dark`` must define the *same* keys: a key present in one and
  missing from the other is a bug ``tests/test_theme.py`` fails on, never a
  fallback.
* **The palette** built from those tokens (:func:`build_palette`) colours
  everything the stylesheet does not address - group boxes, tables, lists, text
  browsers, message boxes, menus, tool tips, disabled and selected states. That is
  the half that used to follow the system while the other half did not.
* **The template** (``assets/theme/daphniascope.qss``) carries *structure* -
  padding, radii, hairlines, the app's own ``objectName`` blocks - and the few
  colours that are a decision rather than a surface: the primary action, the step
  states, and the translucent tools that float *over* a photograph.

The scheme follows the **system** by default and switches when the desktop does
(``QStyleHints.colorSchemeChanged``), so the whole application is dark on a dark
desktop and light on a light one - in every window, including the canvases and the
image viewer. Pin it instead with ``DAPHNIA_THEME=light|dark`` or
``theme.set_scheme("dark")``; an unrecognised value is an error that names itself
and the values it accepts.
"""

import os
import re
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette

from daphnia import paths

#: The stylesheet template. ``%token%`` is not valid QSS, which is what makes it a
#: safe placeholder - the file itself is full of ``{}``, ``#``, ``:`` and ``@``.
TEMPLATE_FILE = "daphniascope.qss"
TOKEN = re.compile(r"%([a-z][a-z0-9_]*)%")

#: The schemes a caller may ask for, and the value that means "ask the system".
SCHEMES = ("light", "dark")
SYSTEM = "system"

#: Environment variable that pins the scheme (``DAPHNIA_THEME=light|dark|system``).
ENV_VAR = "DAPHNIA_THEME"

#: The style that draws what a stylesheet cannot: check marks, combo and spin
#: arrows, focus rings, frame grooves. Fusion is used in *both* schemes on purpose -
#: it is the one style that draws those primitives identically on macOS, Linux and
#: Windows, and a themed application has already replaced the native look of most
#: controls, so mixing native and themed controls is exactly the patchwork to avoid.
STYLE = "Fusion"


class ThemeError(RuntimeError):
    """A theme problem to fix rather than to guess around (no silent fallback)."""


# ---------------------------------------------------------------------------
# Tokens
#
# The tools that float *over* a photograph are deliberately identical in both
# schemes: the window around a picture may follow the desktop, but a panel lying on
# top of the picture has to be readable over a bright frame and a dark one alike -
# and the image is the one thing that must not change meaning when the desktop
# switches from light to dark.
# ---------------------------------------------------------------------------
PHOTO_OVERLAY = {
    "overlay_bg": "rgba(18, 18, 18, 200)",
    "overlay_border": "rgba(255, 255, 255, 45)",
    "overlay_text": "#d7d7d7",
    "overlay_text_strong": "#ffffff",
    "overlay_button": "rgba(255, 255, 255, 26)",
    "overlay_button_hover": "rgba(255, 255, 255, 60)",
    "overlay_button_border": "rgba(255, 255, 255, 55)",
    "overlay_checked": "rgba(17, 104, 228, 220)",
    "overlay_checked_border": "#6aa6ff",
    # The plate a page's **draw tools** sit on, over the picture (CanvasDrawBar): the
    # same plate as the tools themselves, a little lighter, so a bar of buttons on a
    # bright frame never reads as a panel of its own.
    "overlay_card_bg": "rgba(18, 18, 18, 190)",
}

#: Light scheme. Kept close to what the application looked like before - the point
#: of this change is one system, not a new coat of paint.
LIGHT = {
    # surfaces
    "window": "#ececec",
    "surface": "#f7f7f7",
    "surface_alt": "#ffffff",
    "canvas": "#d4d4d4",
    # lines
    "border": "#c8cdd2",
    "border_strong": "#9aa1a8",
    "shadow": "#b0b4b8",
    # text
    "text": "#272727",
    "text_muted": "#6b6b6b",
    "text_disabled": "#a3a3a3",
    "canvas_text": "#5f6368",
    # accent and interaction states
    "accent": "#1168e4",
    "accent_hover": "#1b7bf5",
    "accent_text": "#ffffff",
    "hover": "#e3e8ed",
    "pressed": "#d5dbe1",
    # status
    "ok": "#027f7f",
    "warning": "#a0522d",
    "danger": "#c0392b",
    # The **verdict** of a measurement the operator just made (Calibrate's panel 2) gets
    # its own named pair: the sentence "we measured it and it holds" versus "we measured it
    # and the field moves" is a judgement of that screen, and naming its colours means the
    # verdict can diverge from the step bar's states without either moving for the other.
    "verdict_ok": "#027f7f",
    "verdict_warn": "#a0522d",
    # what a **model** guessed, on a photograph (`SR-PREDICT-018`): a prediction is not a
    # label, so it may not wear a stored label's colour - one colour, both schemes, and it
    # is deliberately the same in light and dark (like the photo overlays, it is read
    # *against the picture*, never against the window).
    "prediction": "#00b0c4",
    # The wells of a plate (``daphnia/ui/widgets/well_grid.py``). They are the one place the
    # application draws a *lamp* rather than a surface: a plate is read from across the
    # bench, so the fill of a well has to say "empty", "has a picture" or "has a number"
    # without any text. Their own names for the same reason as ``verdict_ok``: a well's
    # state is a judgement of the plate page and may diverge from the status tokens.
    "well_empty": "#e2e5e9",
    "well_captured": "#9dc0ea",
    "well_measured": "#6fc3a6",
    "well_rim": "#b8bec6",
    **PHOTO_OVERLAY,
}

#: Dark scheme. Every value is chosen against ``window``/``surface`` for readable
#: contrast (nothing near-invisible for body text) - the failure being fixed.
DARK = {
    "window": "#202124",
    "surface": "#2a2b2e",
    "surface_alt": "#17181a",
    "canvas": "#141416",
    "border": "#3d4045",
    "border_strong": "#5b6067",
    "shadow": "#0b0b0c",
    "text": "#e8eaed",
    "text_muted": "#9aa0a6",
    "text_disabled": "#6b7076",
    "canvas_text": "#8b9095",
    "accent": "#2f81f7",
    "accent_hover": "#4c95f9",
    "accent_text": "#ffffff",
    "hover": "#34373c",
    "pressed": "#3d4147",
    "ok": "#4ec9b0",
    "warning": "#e0a458",
    "danger": "#ff6b6b",
    "verdict_ok": "#4ec9b0",
    "verdict_warn": "#e0a458",
    "prediction": "#00b0c4",
    # The wells of a plate: darker in the dark scheme, and further apart in lightness, so
    # an empty well and a captured one stay distinguishable at 30 px.
    "well_empty": "#34373c",
    "well_captured": "#3d6ba8",
    "well_measured": "#3f8f78",
    "well_rim": "#4c5157",
    **PHOTO_OVERLAY,
}

SCHEME_TOKENS = {"light": LIGHT, "dark": DARK}

#: What is currently applied, what was *asked* for (``None`` == follow the system),
#: and the application being themed, so a live system change can re-apply.
_ACTIVE_SCHEME = None
_REQUESTED_SCHEME = None
_APP = None


# ---------------------------------------------------------------------------
# Reading the desktop's choice
# ---------------------------------------------------------------------------
def system_scheme(app=None) -> str:
    """The scheme the desktop is asking for: ``'light'`` or ``'dark'``.

    Qt reports it through ``QStyleHints.colorScheme()``. When that is ``Unknown`` -
    a headless or remote platform - the window colour of the current palette decides
    instead, and with no application object at all the answer is ``'light'``: a
    headless process has no desktop to follow.
    """
    application = app if app is not None else QGuiApplication.instance()
    if application is None:
        return "light"
    scheme = application.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return "dark"
    if scheme == Qt.ColorScheme.Light:
        return "light"
    window = application.palette().color(QPalette.ColorRole.Window)
    return "dark" if window.lightness() < 128 else "light"


def resolve_scheme(requested=None, app=None) -> str:
    """The scheme to apply: an explicit request, else ``DAPHNIA_THEME``, else the desktop.

    Raises:
        ThemeError: the request (or the environment variable) names no scheme. A typo
            must not silently give the *other* theme - that is the class of bug this
            module exists to remove.
    """
    choice = requested if requested is not None else os.environ.get(ENV_VAR)
    if choice is None:
        return system_scheme(app)
    choice = str(choice).strip().lower()
    if choice in ("", SYSTEM, "auto", "default"):
        return system_scheme(app)
    if choice not in SCHEMES:
        source = ("the 'scheme' argument" if requested is not None
                  else f"${ENV_VAR}")
        raise ThemeError(
            f"{source} names no theme: {choice!r}. Use one of "
            f"{', '.join(SCHEMES)}, or '{SYSTEM}' to follow the desktop.")
    return choice


# ---------------------------------------------------------------------------
# Tokens -> stylesheet, tokens -> palette
# ---------------------------------------------------------------------------
def tokens(scheme) -> dict:
    """Every token of ``scheme``; raises :class:`ThemeError` for an unknown scheme."""
    key = str(scheme).strip().lower()
    if key not in SCHEME_TOKENS:
        raise ThemeError(
            f"unknown theme scheme {scheme!r} - the known ones are "
            f"{', '.join(SCHEMES)}.")
    return dict(SCHEME_TOKENS[key])


def active_scheme() -> str:
    """The scheme in force.

    Before the first :func:`apply_theme` this is the scheme the application *would*
    apply on this machine - which is what painter code needs while a widget is being
    built by a test that never called :func:`apply_theme`.
    """
    return _ACTIVE_SCHEME or resolve_scheme(None, _APP)


def active_tokens() -> dict:
    """The tokens of the scheme in force (see :func:`active_scheme`)."""
    return tokens(active_scheme())


def color(name, scheme=None) -> str:
    """One token, ready to be written into QSS (a hex or ``rgba(...)`` string).

    Painter code wants :func:`qcolor` instead: ``QColor`` does not parse ``rgba(...)``.
    """
    values = tokens(scheme) if scheme is not None else active_tokens()
    if name not in values:
        known = ", ".join(sorted(values))
        raise ThemeError(
            f"no token {name!r} in the theme (scheme '{scheme or active_scheme()}'). "
            f"Add it to LIGHT and DARK in daphnia/ui/theme.py. Known tokens: {known}")
    return values[name]


def qcolor(name, scheme=None) -> QColor:
    """One token as a :class:`QColor`, for painting.

    Raises:
        ThemeError: the token is a CSS ``rgba(...)`` value that ``QColor`` cannot
            parse - such a token belongs in the stylesheet, not in a painter.
    """
    value = color(name, scheme)
    parsed = QColor(value)
    if not parsed.isValid():
        raise ThemeError(
            f"token {name!r} is {value!r}, which QColor cannot read - use it in the "
            f"stylesheet, or give the painter a plain hex token.")
    return parsed


def build_stylesheet(scheme, template_path=None) -> str:
    """The QSS for ``scheme``: the template with every ``%token%`` substituted.

    Two failures raise instead of producing a stylesheet with a hole in it: a token
    the template uses that the scheme does not define, and a template file that is
    not there (it is tracked, so the fix is to restore it).
    """
    values = tokens(scheme)
    path = Path(template_path or (paths.THEME_DIR / TEMPLATE_FILE))
    if not path.is_file():
        raise ThemeError(
            f"theme template {path} is missing - it is a tracked file; restore it "
            f"(git checkout assets/theme/{TEMPLATE_FILE}).")
    template = path.read_text(encoding="utf-8")

    unknown = []

    def substitute(match):
        name = match.group(1)
        if name not in values:
            unknown.append(name)
            return match.group(0)
        return values[name]

    stylesheet = TOKEN.sub(substitute, template)
    if unknown:
        raise ThemeError(
            f"{path.name} uses token(s) {', '.join(sorted(set(unknown)))} that the "
            f"'{scheme}' scheme does not define. Add them to LIGHT and DARK in "
            f"daphnia/ui/theme.py.")
    leftover = TOKEN.search(stylesheet)
    if leftover is not None:
        raise ThemeError(
            f"{path.name}: {leftover.group(0)!r} was not substituted (scheme "
            f"'{scheme}') - a token name is lower case letters, digits and _.")
    return stylesheet


def build_palette(scheme) -> QPalette:
    """The palette for ``scheme``: the colours of every widget QSS does not address.

    This is the half of the theme that used to come from the desktop, which is why
    group boxes, tables, lists, text browsers and message boxes went dark on a dark
    Mac while the QSS-painted buttons stayed light. Now both halves come from the
    same tokens.
    """
    values = tokens(scheme)
    role = QPalette.ColorRole
    group = QPalette.ColorGroup

    palette = QPalette()
    for role_name, token in (
        (role.Window, "window"),
        (role.WindowText, "text"),
        (role.Base, "surface_alt"),
        (role.AlternateBase, "surface"),
        (role.Text, "text"),
        (role.Button, "surface"),
        (role.ButtonText, "text"),
        (role.BrightText, "danger"),
        (role.Highlight, "accent"),
        (role.HighlightedText, "accent_text"),
        (role.ToolTipBase, "surface_alt"),
        (role.ToolTipText, "text"),
        (role.PlaceholderText, "text_muted"),
        (role.Link, "accent"),
        (role.LinkVisited, "accent"),
        # The frame shades: group-box and scroll-bar grooves, sunken borders.
        (role.Light, "surface_alt"),
        (role.Midlight, "surface"),
        (role.Mid, "border"),
        (role.Dark, "border_strong"),
        (role.Shadow, "shadow"),
    ):
        palette.setColor(role_name, QColor(values[token]))

    # Disabled: the same surfaces, muted text, and no bright accent to click.
    for role_name in (role.WindowText, role.Text, role.ButtonText,
                      role.PlaceholderText):
        palette.setColor(group.Disabled, role_name, QColor(values["text_disabled"]))
    for role_name, token in ((role.Base, "surface"),
                             (role.Button, "surface"),
                             (role.Highlight, "border_strong"),
                             (role.HighlightedText, "text_disabled"),
                             (role.Window, "window")):
        palette.setColor(group.Disabled, role_name, QColor(values[token]))

    return palette


# ---------------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------------
def apply_theme(app, scheme=None, follow_system=True, style=STYLE) -> str:
    """Theme the whole application with ``scheme``; returns the scheme applied.

    ``scheme`` is ``'light'``, ``'dark'`` or ``'system'``/``None`` (the default), and
    ``DAPHNIA_THEME`` says the same thing without touching the code. ``follow_system``
    re-applies the theme when the desktop switches appearance, but only while the
    scheme is *not* pinned - a caller who asked for light keeps light.

    The style is set first (Fusion draws the primitives a stylesheet cannot), then the
    palette, then the stylesheet: the palette has to be in place before QSS re-polishes
    every widget, so that the widgets QSS does not address get the same colours.
    """
    if style and hasattr(app, "setStyle") and app.style().objectName().lower() != style.lower():
        app.setStyle(style)

    global _APP, _REQUESTED_SCHEME
    _APP = app
    _REQUESTED_SCHEME = None if scheme in (None, SYSTEM) else _checked_scheme(scheme)

    applied = _install(app, resolve_scheme(_REQUESTED_SCHEME, app))
    if follow_system:
        _follow_system(app)
    return applied


def set_scheme(scheme, app=None) -> str:
    """Switch the running application to ``scheme`` (``light`` | ``dark`` | ``system``).

    The entry point for anything that offers the choice to a user (a menu, a
    setting); the theme is re-applied at once and future desktop changes are followed
    again when ``scheme`` is ``'system'``.
    """
    target = app if app is not None else (_APP or QGuiApplication.instance())
    if target is None:
        raise ThemeError(
            "no QApplication to theme - call apply_theme(app) first, or pass the "
            "application explicitly.")
    return apply_theme(target, scheme)


def _checked_scheme(scheme) -> str:
    """``scheme`` as a pinned scheme name, or :class:`ThemeError`."""
    key = str(scheme).strip().lower()
    if key not in SCHEMES:
        raise ThemeError(
            f"unknown theme scheme {scheme!r} - the known ones are "
            f"{', '.join(SCHEMES)} (or '{SYSTEM}' to follow the desktop).")
    return key


def _install(app, scheme) -> str:
    """Apply one concrete scheme to ``app`` (palette first, then stylesheet)."""
    global _ACTIVE_SCHEME
    _ACTIVE_SCHEME = scheme
    app.setPalette(build_palette(scheme))
    app.setStyleSheet(build_stylesheet(scheme))
    return scheme


def _follow_system(app) -> None:
    """Re-apply the theme when the desktop switches light <-> dark (once per app)."""
    if getattr(app, "_daphnia_following_system_scheme", False):
        return
    change = getattr(app.styleHints(), "colorSchemeChanged", None)
    if change is None:                      # Qt older than 6.5: nothing to follow
        return
    change.connect(lambda *_args: _on_system_scheme_changed())
    app._daphnia_following_system_scheme = True


def _on_system_scheme_changed() -> None:
    """The desktop changed appearance: re-apply, unless the scheme was pinned."""
    if _APP is None or _REQUESTED_SCHEME is not None:
        return
    _install(_APP, resolve_scheme(None, _APP))


