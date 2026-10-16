"""
theme.py — QSS theming for the PySide6 rewrite (v2.6 preview).

A single QSS template parameterised by a small palette (accent + neutrals).
Changing the accent re-applies the stylesheet live — no restart, unlike the
CustomTkinter version where a theme change needed a restart.
"""
from __future__ import annotations

import os
import sys


def assets_dir() -> str:
    """Dossier assets/ — résolu correctement en dev ET en exe gelé (PyInstaller).
    En frozen onedir, les datas sont sous sys._MEIPASS (_internal/)."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets")


def asset(name: str) -> str:
    """Chemin absolu d'un fichier dans assets/."""
    return os.path.join(assets_dir(), name)


# Absolute path to assets/ (forward slashes — Qt QSS url() wants Unix-style paths).
_ASSETS = assets_dir().replace("\\", "/")


# Dark palette aligned to the CustomTkinter "green" dark theme (neutral grays),
# so the Qt port matches the v2.5.8 window background instead of the bluish black.
#   CTk: window gray14 #242424 · frame gray17 #2b2b2b · frame-top gray20 #333333
#        entry #343638 · border gray28 ~#4a4a4a · text #DCE4EE
_DARK = {
    "bg":       "#242424",
    "surface":  "#2b2b2b",
    "surface2": "#343638",
    "border":   "#4a4a4a",
    "text":     "#dce4ee",
    "text_dim": "#9aa3b0",
    "tabbar":   "#333333",
    # Special-dark panels (kept identical to the original look in dark mode).
    "panel":       "#1a1a2e",   # AI / pipeline cards
    "panel_deep":  "#0d1117",   # losses box
    "panel_deep2": "#111827",   # redux-only losses sub-box
    # Training-tab monitoring console (GPU/SYSTEM panels + stat cards).
    "console":      "#181818",
    "console_card": "#222222",
    "console_bd":   "#333333",
    # Onglet Dégradations — fond bleu derrière previews + pages de réglages.
    "deg_blue":     "#1b2a4a",
}

_LIGHT = {
    "bg":       "#f3f4f7",
    "surface":  "#ffffff",
    "surface2": "#e9ebf0",
    "border":   "#cfd4df",
    "text":     "#1a1d24",
    "text_dim": "#5b6273",
    "tabbar":   "#dde1ea",
    # Light equivalents of the special-dark panels (readable on the light theme).
    "panel":       "#ffffff",
    "panel_deep":  "#fbfcfe",
    "panel_deep2": "#eef1f6",
    "console":      "#ffffff",
    "console_card": "#f4f6fb",
    "console_bd":   "#cdd3df",
    # Onglet Dégradations — fond bleu clair (lisible en thème clair).
    "deg_blue":     "#e7eefb",
}

# Accent swatches for the customisation panel.
PRESET_ACCENTS = {
    "Green":  "#2ecc71",
    "Blue":   "#3b8ed0",
    "Purple": "#8e6cd0",
    "Orange": "#e67e22",
    "Pink":   "#e84393",
    "Teal":   "#16c0a8",
    "Red":    "#e74c3c",
}

DEFAULT_ACCENT = "#2ecc71"


def _clamp(v: int) -> int:
    return max(0, min(255, v))


def _shift(hex_color: str, factor: float) -> str:
    """Lighten (factor>1) or darken (factor<1) a #rrggbb colour."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (_clamp(int(c * factor)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


_QSS = """
* {{ font-family: "Segoe UI", "Segoe UI Emoji"; font-size: {fs}px; color: {text}; }}
QMainWindow, QWidget {{ background: {bg}; }}
/* Labels never paint their own background (avoids dark boxes behind text on panels) */
QLabel {{ background: transparent; }}
/* Tooltips: accent background (dimmed), white text, supports rich-text (colored ↑/↓) */
QToolTip {{ background: {accent_dim}; color: #ffffff; border: 1px solid {accent};
           border-radius: 5px; padding: 5px 7px; font-size: {fs}px; }}

/* Top nav bar (custom, centred) — CTk segmented-button look */
QFrame#navbar {{ background: {tabbar}; border-radius: 11px; }}
QPushButton#topnav {{ background: transparent; color: {text_dim}; padding: 9px 18px;
    border-radius: 8px; font-weight: 600; border: none; }}
QPushButton#topnav:hover {{ background: {surface2}; color: {text}; }}
QPushButton#topnav:checked {{ background: {accent}; color: #ffffff; }}

/* (QTabWidget kept for any nested use) */
QTabWidget::pane {{ border: 1px solid {border}; background: {surface}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ background: {surface2}; color: {text_dim}; padding: 8px 18px; margin: 2px; border-radius: 6px; font-weight: 600; }}
QTabBar::tab:selected {{ background: {accent}; color: #ffffff; }}

QPushButton {{ background: {accent}; color: #ffffff; border: none; padding: 8px 16px; border-radius: 7px; font-weight: 600; }}
QPushButton:hover {{ background: {accent_hi}; }}
QPushButton:pressed {{ background: {accent_lo}; }}
QPushButton:disabled {{ background: {surface2}; color: {text_dim}; }}
QPushButton#ghost {{ background: {surface2}; color: {text}; }}
QPushButton#ghost:hover {{ background: {border}; }}
QPushButton#warn {{ background: #e67e22; color: #ffffff; }}
QPushButton#warn:hover {{ background: #d35400; }}
QPushButton#warn:pressed {{ background: #ba4a00; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {surface2}; border: 1px solid {border}; border-radius: 6px; padding: 3px 7px; color: {text}; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border: 1px solid {accent}; }}

/* QComboBox styled like the CTk green CTkOptionMenu (accent bg, white text) */
QComboBox {{ background: {accent}; color: #ffffff; border: none; border-radius: 6px;
    padding: 3px 8px; font-weight: 600; min-height: 18px; }}
QComboBox:hover {{ background: {accent_hi}; }}
QComboBox:focus {{ background: {accent_hi}; }}
QComboBox:disabled {{ background: {surface2}; color: {text_dim}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{ image: url("{assets}/arrow_down.svg"); width: 10px; height: 6px; margin-right: 6px; }}
QComboBox::down-arrow:disabled {{ image: url("{assets}/arrow_down.svg"); }}
QComboBox QAbstractItemView {{ background: {surface2}; color: {text}; border: 1px solid {border};
    selection-background-color: {accent}; selection-color: #fff; outline: none; }}

QGroupBox {{ border: 1px solid {border}; border-radius: 10px; margin-top: 14px; padding: 10px; background: {surface}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px; color: {accent}; font-weight: 700; }}
QGroupBox#boxG {{ border: 1px solid #3b8ed0; background: transparent; }}
QGroupBox#boxG::title {{ color: #3b8ed0; }}
QGroupBox#boxD {{ border: 1px solid #e74c3c; background: transparent; }}
QGroupBox#boxD::title {{ color: #e74c3c; }}

QSlider::groove:horizontal {{ height: 6px; background: {surface2}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {accent}; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }}
QSlider::handle:horizontal:hover {{ background: {accent_hi}; }}

QCheckBox::indicator, QRadioButton::indicator {{ width: 18px; height: 18px; border: 1px solid {border};
    border-radius: 5px; background: {surface2}; }}
QCheckBox::indicator:checked {{ background: {accent}; border: 1px solid {accent};
    image: url("{assets}/check.svg"); }}
QRadioButton::indicator:checked {{ background: {accent}; border: 1px solid {accent}; }}
QRadioButton::indicator {{ border-radius: 9px; }}

QProgressBar {{ background: {surface2}; border: none; border-radius: 7px; text-align: center; height: 16px; color: {text}; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 7px; }}

QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {border}; border-radius: 6px; min-height: 32px; }}
QScrollBar::handle:vertical:hover {{ background: {accent}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

QLabel#title {{ font-size: 26px; font-weight: 800; color: {accent}; }}
QLabel#subtitle {{ color: {text_dim}; font-size: 13px; }}
QLabel#h2 {{ font-size: 16px; font-weight: 700; color: #3b8ed0; }}
QFrame#card {{ background: {surface}; border: 1px solid {border}; border-radius: 12px; }}
QFrame#sep {{ background: {border}; max-height: 1px; }}

/* ── Theme-aware panels (objectNames) — replace hardcoded dark backgrounds so the
      light theme stays readable. Driven by the palette so they adapt live. ── */
/* Dégradations tab */
QTabWidget#degtabs::pane {{ background: {deg_blue}; border: 1px solid {border}; border-radius: 8px; }}
QTabWidget#degtabs::tab-bar {{ alignment: center; }}
QWidget#degcol {{ background: {deg_blue}; }}
QWidget#degcol QLabel, QWidget#degcol QSlider, QWidget#degcol QCheckBox {{ background: transparent; }}
QWidget#degcol QLineEdit {{ background: transparent; border: 1px solid {border}; border-radius: 5px; }}
QFrame#degprev {{ background: {deg_blue}; border: 1px solid {border}; border-radius: 8px; }}
QFrame#degprev QLabel, QFrame#degprev QCheckBox {{ background: transparent; }}
QPushButton#sevbtn {{ background: {surface2}; color: {text}; padding: 4px 12px; border-radius: 6px; font-weight: 600; }}
QPushButton#sevbtn:hover {{ background: {border}; }}
/* Generic panels reused across tools/run/queue/assistant pages */
QFrame#pcard, QFrame#pcard2, QFrame#ppmod, QFrame#rcard, QFrame#qcard, QFrame#genprev {{
    background: {surface}; border: 1px solid {border}; border-radius: 8px; }}
/* Config tab — special panels (dark look preserved in dark, readable in light) */
QFrame#navpanel {{ background: {surface}; border: 1px solid {border}; border-radius: 8px; }}
QFrame#lossbox {{ background: {panel_deep}; border-radius: 8px; }}
QFrame#lossredux {{ background: {panel_deep2}; border-radius: 6px; }}
QFrame#lossbox QLabel, QFrame#lossbox QCheckBox {{ background: transparent; }}
QFrame#lossbox QLabel:disabled, QFrame#lossbox QCheckBox:disabled {{ color: {text_dim}; }}
QFrame#aiCard {{ background: {panel}; border-radius: 8px; }}
/* Training tab — monitoring console (dark preserved in dark, light & readable in light).
   Labels inherit the theme text colour; coloured stat values keep their own colours. */
QFrame#mon {{ background: {console}; border: 1px solid {console_bd}; border-radius: 8px; }}
QFrame#files {{ background: {console_card}; border: 1px solid {console_bd}; border-radius: 8px; }}
QFrame#statcard {{ background: {console_card}; border: 1px solid {console_bd}; border-radius: 5px; }}
QFrame#infobar {{ background: {console_card}; border: 1px solid {console_bd}; border-radius: 6px; }}
QFrame#mon QLabel, QFrame#statcard QLabel, QFrame#infobar QLabel, QFrame#files QLabel {{
    background: transparent; }}
/* Sidebar nav buttons (Config + Outils) — theme-aware text (was palette(text) → invisible
   on the light theme), accent highlight when selected. */
QPushButton#nav {{ background: transparent; text-align: left; padding: 9px 12px;
    border-radius: 6px; font-weight: 600; color: {text}; border: none; }}
QPushButton#nav:hover {{ background: {surface2}; }}
QPushButton#nav:checked {{ background: {accent}; color: #ffffff; }}
"""


def build_qss(accent: str = DEFAULT_ACCENT, dark: bool = True, scale: float = 1.0) -> str:
    pal = dict(_DARK if dark else _LIGHT)
    pal["accent"] = accent
    pal["accent_hi"] = _shift(accent, 1.15)
    pal["accent_lo"] = _shift(accent, 0.85)
    pal["accent_dim"] = _shift(accent, 0.45)  # tooltip background — accent, less intense
    pal["fs"] = max(9, round(13 * scale))
    pal["assets"] = _ASSETS
    return _QSS.format(**pal)


def apply_theme(app, accent: str = DEFAULT_ACCENT, dark: bool = True, scale: float = 1.0) -> None:
    """Apply the theme to a QApplication — call again any time to re-theme live."""
    app.setStyleSheet(build_qss(accent, dark, scale))


def enhance_button_hover(root) -> None:
    """Add :hover / :pressed feedback to every flat colour-styled QPushButton under `root`
    that doesn't already define them. Lets buttons that set an inline `background:#hex`
    (green/red/orange/purple action buttons) get the same press feedback as themed ones,
    without editing each call site. Idempotent — skips buttons already wrapped or stateful."""
    import re
    from PySide6.QtWidgets import QPushButton
    for b in root.findChildren(QPushButton):
        ss = (b.styleSheet() or "").strip()
        if not ss or ":hover" in ss or ":pressed" in ss or "QPushButton" in ss:
            continue
        m = re.search(r'background\s*:\s*(#[0-9a-fA-F]{6})', ss)
        if not m:
            continue
        bg = m.group(1)
        hi = _shift(bg, 1.12)
        lo = _shift(bg, 0.84)
        body = ss if ss.endswith(";") else ss + ";"
        b.setStyleSheet(f"QPushButton{{{body}}}"
                        f"QPushButton:hover{{background:{hi};}}"
                        f"QPushButton:pressed{{background:{lo}; padding-top:1px;}}")
