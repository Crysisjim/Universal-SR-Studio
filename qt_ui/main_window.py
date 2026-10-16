"""
main_window.py — PySide6 main window (v2.6.0).

Goal of this skeleton:
  * prove the Qt shell is smooth (native rendering — no per-widget redraw lag,
    no freeze on window focus/restore, no resize stutter),
  * showcase the live theming (accent colour picker, instant, no restart),
  * prove the existing GUI-agnostic backend (src/core/) is reused as-is.

It is NOT a full port — tabs other than Customisation / System are placeholders
to be filled in tab by tab.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QColorDialog, QGroupBox, QGridLayout, QComboBox, QFrame,
    QPlainTextEdit, QSizePolicy,
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

from qt_ui import theme
from qt_ui.i18n import tr


class MainWindow(QMainWindow):
    def __init__(self, settings, app):
        super().__init__()
        self.settings = settings
        self.app = app
        self._accent = settings.get("qt_accent", theme.DEFAULT_ACCENT) or theme.DEFAULT_ACCENT
        # Theme mode: "dark" (default — the app was designed dark-first), "light", or "auto"
        # (follow the Windows system theme). Resolve to an effective dark bool.
        self._theme_mode = (settings.get("qt_theme_mode", "") or "").lower()
        if self._theme_mode not in ("dark", "light", "auto"):
            # migrate the legacy qt_dark flag; default to dark
            self._theme_mode = "light" if settings.get("qt_dark", "true") == "false" else "dark"
        self._dark = self._resolve_dark()
        try:
            self._scale = float(settings.get("qt_scale", "1.0") or 1.0)
        except (TypeError, ValueError):
            self._scale = 1.0

        self.setWindowTitle("Universal SR Studio  (Qt — v2.6.0)")
        # Restore the last window geometry (size + position); fall back to the CTk default.
        _geo = settings.get("qt_win_geometry", "")
        if _geo:
            try:
                from PySide6.QtCore import QByteArray
                self.restoreGeometry(QByteArray.fromBase64(_geo.encode("ascii")))
            except Exception:
                self.resize(1600, 900)
        else:
            self.resize(1600, 900)

        # Custom top nav bar (centred, pill style) + stacked pages. Guarantees the
        # main tabs are horizontally centred in the window (QTabWidget alignment is
        # unreliable across styles).
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(10)

        navrow = QHBoxLayout()
        navrow.addStretch(1)
        self._navbar = QFrame(); self._navbar.setObjectName("navbar")
        self._navlay = QHBoxLayout(self._navbar)
        self._navlay.setContentsMargins(5, 5, 5, 5)
        self._navlay.setSpacing(3)
        navrow.addWidget(self._navbar)
        navrow.addStretch(1)
        outer.addLayout(navrow)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self._nav_buttons = []

        # Same names + order + emoji as the CustomTkinter app (app.py).
        from qt_ui.i18n import tr
        self._add_page("🧙 " + tr("Assistant", "Assistant"), self._make("qt_ui.tabs.assistant_tab", "AssistantTab"))
        self._add_page("📝 " + tr("Configuration", "Configuration"), self._make("qt_ui.tabs.config_tab", "ConfigTab"))
        self._add_page("🚀 " + tr("Entraînement", "Training"), self._make("qt_ui.tabs.run_tab", "RunTab"))
        self._add_page("🔧 " + tr("Outils", "Tools"), self._make("qt_ui.tabs.tools_tab", "ToolsTab"))
        self._add_page("📋 " + tr("File d'attente", "Queue"), self._make("qt_ui.tabs.queue_tab", "QueueTab"))
        self._settings_index = len(self._nav_buttons)  # index de la page Réglages (pour le startup fresh)
        self._add_page("⚙️ " + tr("Réglages", "Settings"), self._make("qt_ui.tabs.settings_tab", "SettingsTab",
                                                                      theming=self))
        self._add_page("🌐 " + tr("Partage", "Sharing"), self._make("qt_ui.tabs.partage_tab", "PartageTab"))
        self._add_page("📖 " + tr("Wiki", "Wiki"), self._make("qt_ui.tabs.wiki_tab", "WikiTab"))

        self._apply_theme()
        # give every inline colour-styled button hover/pressed feedback (app-wide, once)
        try:
            theme.enhance_button_hover(self)
        except Exception:
            pass
        # Fresh install (aucun moteur installé) → ouvre sur Réglages (Moteurs) ;
        # sinon ouvre sur l'Assistant.
        try:
            from src.core import engine_installer as _ei
            _fresh = not (_ei.is_engine_installed("TraiNNer-Redux") or _ei.is_engine_installed("NeoSR"))
        except Exception:
            _fresh = False
        self._select(self._settings_index if _fresh else 0)

    # ── theming controller (used by the Settings tab) ────────────────────────
    def accent(self) -> str:
        return self._accent

    def is_dark(self) -> bool:
        return self._dark

    def set_accent(self, hexv: str):
        self._accent = hexv
        self.settings.set("qt_accent", hexv)
        self._apply_theme()

    def set_dark(self, dark: bool):
        # the saved mode too: qt_theme_mode is read first at startup, so a choice made
        # here was undone by the previous dark / light / auto mode after a restart
        self._theme_mode = "dark" if dark else "light"
        self.settings.set("qt_theme_mode", self._theme_mode)
        self._dark = bool(dark)
        self.settings.set("qt_dark", "true" if dark else "false")
        self._apply_theme()

    def scale(self) -> float:
        return getattr(self, "_scale", 1.0)

    def set_scale(self, value: float):
        self._scale = float(value)
        self.settings.set("qt_scale", str(self._scale))
        self._apply_theme()

    # ── helpers ────────────────────────────────────────────────────────────
    def _make(self, module: str, cls: str, **kwargs) -> QWidget:
        """Import + construct a ported tab; fall back to a placeholder on failure."""
        try:
            mod = __import__(module, fromlist=[cls])
            return getattr(mod, cls)(self.settings, **kwargs)
        except Exception as e:
            ph = self._placeholder(cls.replace("Tab", ""))
            ph.layout().addWidget(QLabel(f"(indisponible : {e})"))
            return ph

    def _add_page(self, title: str, widget: QWidget):
        idx = self.stack.count()
        b = QPushButton(title)
        b.setObjectName("topnav")
        b.setCheckable(True)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(lambda _=False, i=idx: self._select(i))
        self._navlay.addWidget(b)
        self._nav_buttons.append(b)
        self.stack.addWidget(widget)

    def _select(self, index: int):
        self.stack.setCurrentIndex(index)
        for i, b in enumerate(self._nav_buttons):
            b.setChecked(i == index)

    def _apply_theme(self):
        theme.apply_theme(self.app, self._accent, self._dark, getattr(self, "_scale", 1.0))

    def _running_trainer(self):
        """The RunTab's TrainingRunner while a training process is alive, else None."""
        try:
            from qt_ui.tabs.run_tab import RunTab
            rt = self.findChild(RunTab)
        except Exception:
            return None, None
        r = getattr(rt, "runner", None) if rt is not None else None
        if r is not None and getattr(r, "is_running", False):
            return rt, r
        return None, None

    def closeEvent(self, event):
        # A training run outlives the window (separate python process): closing the app by
        # mistake used to leave it running unseen, with no way to stop it from the UI.
        if not getattr(self, "_uss_close_ok", False):
            rt, runner = self._running_trainer()
            if runner is not None:
                from PySide6.QtWidgets import QMessageBox
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setWindowTitle(tr("Entraînement en cours", "Training in progress"))
                box.setText(tr("Un entraînement est en cours. Que voulez-vous faire ?",
                               "A training run is in progress. What do you want to do?"))
                box.setInformativeText(tr(
                    "« Sauvegarder et quitter » fait un arrêt propre (modèle et état sauvés, "
                    "reprise possible) puis ferme l'application.\n"
                    "« Laisser tourner » ferme la fenêtre mais l'entraînement continue en "
                    "arrière-plan (visible seulement dans le Gestionnaire des tâches).",
                    "\"Save and quit\" stops cleanly (model and state saved, resumable) and then "
                    "closes the application.\n"
                    "\"Keep running\" closes the window but training goes on in the background "
                    "(only visible in Task Manager)."))
                b_save = box.addButton(tr("Sauvegarder et quitter", "Save and quit"), QMessageBox.AcceptRole)
                b_keep = box.addButton(tr("Laisser tourner", "Keep running"), QMessageBox.DestructiveRole)
                b_cancel = box.addButton(tr("Annuler", "Cancel"), QMessageBox.RejectRole)
                box.setDefaultButton(b_cancel)
                box.setEscapeButton(b_cancel)
                box.exec()
                clicked = box.clickedButton()
                if clicked is b_save:
                    event.ignore()
                    self._close_after_soft_stop(rt, runner)
                    return
                if clicked is not b_keep:
                    event.ignore()
                    return
        # Persist window geometry (size + position) for the next session.
        try:
            self.settings.set("qt_win_geometry", bytes(self.saveGeometry().toBase64()).decode("ascii"))
        except Exception:
            pass
        super().closeEvent(event)

    def _close_after_soft_stop(self, rt, runner):
        """Soft stop (same path as the Stop button: save then exit), wait for the process
        to end, then close. The runner kills the process itself after 120 s without exit."""
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QProgressDialog
        try:
            if not getattr(runner, "stop_requested", False):
                rt._stop()
        except Exception:
            pass
        dlg = QProgressDialog(tr("Sauvegarde de l'entraînement en cours…\n"
                                 "L'application se fermera ensuite.",
                                 "Saving the training run…\nThe application will close afterwards."),
                              None, 0, 0, self)
        dlg.setWindowTitle(tr("Arrêt propre", "Clean stop"))
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.show()
        self._uss_stop_timer = QTimer(self)

        def _poll():
            if getattr(runner, "is_running", False):
                return
            self._uss_stop_timer.stop()
            dlg.close()
            try:
                runner.kill_monitoring_tools()
            except Exception:
                pass
            self._uss_close_ok = True
            self.close()

        self._uss_stop_timer.timeout.connect(_poll)
        self._uss_stop_timer.start(500)

    def _placeholder(self, name: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        title = QLabel(name)
        title.setObjectName("title")
        sub = QLabel(tr("Onglet à porter depuis la version CustomTkinter.\n"
                     "Le backend (src/core/) est déjà réutilisable tel quel.", "Tab to port from the CustomTkinter version.\nThe backend (src/core/) is already reusable as-is."))
        sub.setObjectName("subtitle")
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addStretch(1)
        return w

    def _hsep(self) -> QFrame:
        f = QFrame()
        f.setObjectName("sep")
        f.setFrameShape(QFrame.HLine)
        return f

    # ── System tab (backend reuse demo) ──────────────────────────────────────
    def _system_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        title = QLabel("System / GPU")
        title.setObjectName("title")
        lay.addWidget(title)
        lay.addWidget(QLabel(tr("Données fournies par le backend existant "
                             "(src/core/compute_estimator.py) — preuve de réutilisation.", "Data provided by the existing backend (src/core/compute_estimator.py) — proof of reuse.")))

        box = QGroupBox("GPU & PyTorch")
        gl = QVBoxLayout(box)
        self._gpu_text = QPlainTextEdit()
        self._gpu_text.setReadOnly(True)
        self._gpu_text.setMinimumHeight(180)
        gl.addWidget(self._gpu_text)
        btn = QPushButton(tr("Détecter", "Detect"))
        btn.clicked.connect(self._detect_gpu)
        gl.addWidget(btn, alignment=Qt.AlignLeft)
        lay.addWidget(box)
        lay.addStretch(1)
        self._detect_gpu()
        return w

    def _detect_gpu(self):
        lines = []
        try:
            from src.core.compute_estimator import get_pytorch_recommendation
            rec = get_pytorch_recommendation()
            lines.append(f"GPU          : {rec.get('gpu_name', '?')}")
            lines.append(f"Génération   : {rec.get('gpu_gen', '?')}")
            lines.append(f"Tensor Cores : {rec.get('has_tensor_cores', '?')}")
            lines.append(f"PyTorch reco : {rec.get('torch_version', '?')}  ({rec.get('cuda_tag', '?')})")
            lines.append(f"Wheel        : {rec.get('whl_url', '?')}")
            note = rec.get("note", "")
            if note:
                lines.append(f"\n{note}")
        except Exception as e:
            lines.append(f"(backend indisponible ici : {e})")
        self._gpu_text.setPlainText("\n".join(lines))

    # ── Customisation tab (the showcase) ─────────────────────────────────────
    def _customisation_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        title = QLabel(tr("Customisation", "Customization"))
        title.setObjectName("title")
        lay.addWidget(title)
        lay.addWidget(QLabel(tr("Couleur d'accent appliquée EN DIRECT (pas de redémarrage) — "
                             "impossible proprement en CustomTkinter.", "Accent color applied LIVE (no restart) — not cleanly possible in CustomTkinter.")))

        # Accent presets
        box = QGroupBox("Couleur d'accent")
        gl = QGridLayout(box)
        gl.addWidget(QLabel("Presets :"), 0, 0)
        col = 1
        for label, hexv in theme.PRESET_ACCENTS.items():
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, h=hexv: self._set_accent(h))
            b.setStyleSheet(f"background:{hexv}; color:#fff;")
            gl.addWidget(b, 0, col)
            col += 1
        pick = QPushButton(tr("Couleur perso…", "Custom color…"))
        pick.setObjectName("ghost")
        pick.clicked.connect(self._pick_accent)
        gl.addWidget(pick, 1, 1, 1, 2)
        self._accent_lbl = QLabel(self._accent)
        gl.addWidget(self._accent_lbl, 1, 3)
        lay.addWidget(box)

        # Light / dark
        box2 = QGroupBox(tr("Apparence", "Appearance"))
        hl = QHBoxLayout(box2)
        hl.addWidget(QLabel("Mode :"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([tr("Sombre", "Dark"), tr("Clair", "Light"), tr("Auto (système)", "Auto (system)")])
        self._mode_combo.setToolTip(tr("Sombre = thème par défaut (l'app a été pensée pour le sombre).\n"
                                    "Clair = thème clair. Auto = suit le thème système Windows.", "Dark = default theme (the app was designed dark-first).\nLight = light theme. Auto = follows the Windows system theme."))
        self._mode_combo.setCurrentIndex({"dark": 0, "light": 1, "auto": 2}.get(self._theme_mode, 0))
        self._mode_combo.currentIndexChanged.connect(self._set_mode)
        hl.addWidget(self._mode_combo)
        hl.addStretch(1)
        lay.addWidget(box2)

        # Live demo of themed widgets
        demo = QGroupBox(tr("Aperçu live", "Live preview"))
        dl = QHBoxLayout(demo)
        dl.addWidget(QPushButton(tr("Bouton", "Button")))
        g = QPushButton("Ghost"); g.setObjectName("ghost"); dl.addWidget(g)
        cb = QComboBox(); cb.addItems(["Option A", "Option B"]); dl.addWidget(cb)
        dl.addStretch(1)
        lay.addWidget(demo)
        lay.addStretch(1)
        return w

    def _set_accent(self, hexv: str):
        self._accent = hexv
        self._accent_lbl.setText(hexv)
        self.settings.set("qt_accent", hexv)
        self._apply_theme()

    def _pick_accent(self):
        c = QColorDialog.getColor(QColor(self._accent), self, "Couleur d'accent")
        if c.isValid():
            self._set_accent(c.name())

    @staticmethod
    def _windows_is_dark() -> bool:
        """Read the Windows app theme (AppsUseLightTheme: 0 = dark, 1 = light). Defaults to
        dark if it can't be read (non-Windows or registry error)."""
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                               r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            val, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
            winreg.CloseKey(k)
            return int(val) == 0
        except Exception:
            return True

    def _resolve_dark(self) -> bool:
        if self._theme_mode == "light":
            return False
        if self._theme_mode == "auto":
            return self._windows_is_dark()
        return True  # "dark" (default)

    def _set_mode(self, idx: int):
        self._theme_mode = ("dark", "light", "auto")[idx] if 0 <= idx <= 2 else "dark"
        self.settings.set("qt_theme_mode", self._theme_mode)
        self._dark = self._resolve_dark()
        self.settings.set("qt_dark", "true" if self._dark else "false")  # legacy compat
        self._apply_theme()
        try:
            theme.enhance_button_hover(self)
        except Exception:
            pass
