"""
settings_tab.py — Réglages (PySide6) — sub-tabbed, faithful to CTk tab_settings:

  Apparence · Moteurs & Système · Langue & Notif. · Clés API · Galerie & Patch TB · À Propos

Apparence is its own tab. Moteurs & Système groups, per engine, the dependency table +
install/update controls, plus AIDA64 / UI-fluide toggles, Python-portable status and the
GPU/PyTorch advisory. Git actions run via a QProcess console; advisory + dependency
versions read in QThreads.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QGroupBox,
    QPushButton, QFileDialog, QPlainTextEdit, QComboBox, QColorDialog, QScrollArea,
    QFrame, QDialog, QMessageBox, QTabWidget, QCheckBox, QSlider,
)
from PySide6.QtGui import QColor, QFont, QTextCursor

from qt_ui.i18n import tr
from PySide6.QtCore import Qt, QProcess, QThread, Signal

from qt_ui import theme

_EMOJI = 'font-family:"Segoe UI Emoji"; font-size:13px; padding:0;'
_REPOS = {
    "NeoSR": "https://github.com/muslll/neosr",
    "TraiNNer-Redux": "https://github.com/the-database/traiNNer-redux",
}
_DEPS = [
    ("torch", "torch"), ("torchvision", "torchvision"), ("cv2", "cv2"),
    ("numpy", "numpy"), ("yaml", "yaml"), ("scipy", "scipy"), ("pywt", "pywt"),
    ("tqdm", "tqdm"), ("rich", "rich"), ("tensorboard", "tensorboard"),
    ("PIL", "PIL"), ("ema_pytorch", "ema_pytorch"),
]
_API_PROVIDERS = [
    ("OpenRouter (Gratuit)", "api_key_openrouter", "https://openrouter.ai/settings/keys"),
    # GitHub Models fully retired 2026-07-30 (playground/catalog/inference API/BYOK all shut
    # down) — row removed, provider no longer usable. See src/core/ai_providers.py.
    ("Google (Gemini)", "api_key_google", "https://aistudio.google.com/apikey"),
    ("Anthropic (Claude)", "api_key_anthropic", "https://console.anthropic.com/settings/keys"),
    ("OpenAI (ChatGPT)", "api_key_openai", "https://platform.openai.com/api-keys"),
    ("xAI (Grok)", "api_key_xai", "https://console.x.ai"),
    ("DeepSeek", "api_key_deepseek", "https://platform.deepseek.com/api_keys"),
    ("NVIDIA NIM", "api_key_nvidia", "https://build.nvidia.com/settings/api-key"),
]
_NOTIF = [
    ("notif_win11_upscale", tr("Upscale image terminé", "Image upscale done")), ("notif_win11_batch", tr("Batch upscale terminé", "Batch upscale done")),
    ("notif_win11_errors", tr("Erreurs & crashes", "Errors & crashes")), ("notif_win11_training", tr("Fin d'entraînement", "Training end")),
]
_EVENT_SOUNDS = [
    ("snd_error", tr("Son d'erreur grave (crash, erreur critique)", "Serious error sound (crash, critical error)")),
    ("snd_warning", tr("Son d'avertissement (erreur non fatale dans un batch)", "Warning sound (non-fatal error in a batch)")),
    ("snd_about", tr("Son de la section À Propos", "About section sound")),
]
_GAL_AUTO_DEFAULTS = {"gal_auto_stop": "true", "gal_auto_ngrok": "true"}
_GAL_AUTO = [
    ("gal_auto_start", tr("Auto-démarrer la galerie au lancement de l'entraînement", "Auto-start gallery when training launches")),
    ("gal_auto_stop", tr("Auto-arrêter la galerie à l'arrêt de l'entraînement", "Auto-stop gallery when training stops")),
    ("gal_auto_tb", tr("Auto-démarrer TensorBoard au lancement (démarre le serveur TB)", "Auto-start TensorBoard on launch (starts the TB server)")),
    ("gal_auto_open", tr("Auto-ouvrir le navigateur sur localhost:6006 quand TB prêt", "Auto-open browser at localhost:6006 when TB ready")),
    ("gal_auto_ngrok", tr("Autoriser le lancement automatique de Ngrok (monitoring.auto_ngrok)", "Allow automatic Ngrok launch (monitoring.auto_ngrok)")),
]


def _folder_size(path):
    total = 0
    try:
        for d, _, fs in os.walk(path):
            for fn in fs:
                fp = os.path.join(d, fn)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        return f"{total / (1024 * 1024):.1f} MB"
    except Exception:
        return "0 MB"


class _AboutBg(QWidget):
    """À Propos background: about_bg.jpg darkened (~0.3 brightness), cover-fit."""
    def __init__(self, img_path, parent=None):
        super().__init__(parent)
        from PySide6.QtGui import QPixmap
        self._pm = QPixmap(img_path) if (img_path and os.path.isfile(img_path)) else QPixmap()

    def paintEvent(self, ev):
        from PySide6.QtGui import QPainter, QColor
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1a1a2e"))
        if not self._pm.isNull():
            scaled = self._pm.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            p.drawPixmap(-x, -y, scaled)
            p.fillRect(self.rect(), QColor(10, 10, 25, 178))  # darken ≈ brightness 0.3
        p.end()


def _aida64_detected() -> bool:
    """AIDA64 lancé ? (process aida64.exe). Sert à auto-activer le toggle au 1er run."""
    try:
        import subprocess
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq aida64.exe"],
                           capture_output=True, text=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return "aida64.exe" in (r.stdout or "").lower()
    except Exception:
        return False


class _ConsoleDialog(QDialog):
    def __init__(self, title, program, args, cwd=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title); self.resize(820, 520)
        self._running = True
        v = QVBoxLayout(self)
        warn = QLabel(tr("⚠️  Installation en cours — NE FERMEZ PAS cette fenêtre "
                         "(téléchargements en cours, ça peut être long).",
                         "⚠️  Installation in progress — DO NOT CLOSE this window "
                         "(downloads in progress, this can take a while)."))
        warn.setWordWrap(True)
        warn.setStyleSheet("background:#7a3b00;color:#ffd9a8;border-radius:6px;padding:6px 10px;font-weight:bold;")
        v.addWidget(warn)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFont(QFont("Consolas", 10))
        self.log.setMaximumBlockCount(5000)
        v.addWidget(self.log, 1)
        self.btn = QPushButton(tr("⏳ Installation en cours…", "⏳ Installing…")); self.btn.setObjectName("ghost")
        self.btn.setEnabled(False); self.btn.clicked.connect(self.accept)
        v.addWidget(self.btn)
        self.proc = QProcess(self); self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._out)
        self.proc.finished.connect(self._on_finished)
        if cwd and os.path.isdir(cwd):
            self.proc.setWorkingDirectory(cwd)
        self.log.appendPlainText(f">> {program} {' '.join(args)}\n")
        self.proc.start(program, args)

    def _on_finished(self, code, _status):
        self._running = False
        self.log.appendPlainText(tr(f"\n== Terminé (code {code}) ==", f"\n== Done (code {code}) =="))
        ok = (code == 0)
        self.btn.setText(tr("✅ Fermer", "✅ Close") if ok else tr("❌ Fermer (voir erreurs)", "❌ Close (see errors)"))
        self.btn.setEnabled(True)
        # l'avertissement n'a plus lieu d'être
        warn = self.findChild(QLabel)
        if warn:
            warn.setText(tr("✅ Terminé — tu peux fermer.", "✅ Done — you can close.") if ok else tr("❌ Échec — voir le log ci-dessus.", "❌ Failed — see the log above."))
            warn.setStyleSheet(("background:#1e4d2b;color:#bff5c8;" if ok else "background:#5a1f1f;color:#ffc9c9;")
                               + "border-radius:6px;padding:6px 10px;font-weight:bold;")

    def _out(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        # Gère les barres de progression pip (retours chariot \r sans \n) :
        # un \r réécrit la ligne courante au lieu d'empiler des doublons.
        cur = self.log.textCursor()
        cur.movePosition(QTextCursor.End)
        i = 0
        while i < len(data):
            ch = data[i]
            if ch == "\r":
                nxt = data[i + 1] if i + 1 < len(data) else ""
                if nxt != "\n":  # \r seul = réécriture de ligne (progress bar)
                    cur.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                    cur.removeSelectedText()
                # \r\n traité comme un simple \n au tour suivant
            elif ch == "\n":
                cur.insertText("\n")
            else:
                cur.insertText(ch)
            i += 1
        self.log.setTextCursor(cur)
        self.log.ensureCursorVisible()

    def closeEvent(self, ev):
        if self._running:
            r = QMessageBox.question(
                self, tr("Installation en cours", "Installation in progress"),
                tr("Une installation est EN COURS. La fermer maintenant laissera "
                   "l'environnement incomplet (à réinstaller).\n\nFermer quand même ?",
                   "An installation is IN PROGRESS. Closing it now will leave "
                   "the environment incomplete (to be reinstalled).\n\nClose anyway?"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                ev.ignore(); return
            try:
                self.proc.kill()
            except Exception:
                pass
        ev.accept()


class _AdvisoryWorker(QThread):
    done = Signal(dict)

    def run(self):
        try:
            from src.core.compute_estimator import get_pytorch_recommendation
            self.done.emit(get_pytorch_recommendation() or {})
        except Exception as e:
            self.done.emit({"gpu_name": "Indisponible", "note": str(e)})


class _DepsWorker(QThread):
    done = Signal(dict)

    def __init__(self, py):
        super().__init__(); self.py = py

    def run(self):
        out = {}
        if self.py and os.path.isfile(self.py):
            import subprocess
            # Resolve versions via package metadata first (reliable for rich /
            # ema_pytorch which don't expose __version__), then fall back to import.
            _dist = {"cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml",
                     "pywt": "PyWavelets", "ema_pytorch": "ema-pytorch"}
            code = ("import importlib, importlib.metadata as M\n"
                    "_d=%r\n"
                    "for m in %r:\n"
                    "  v=None\n"
                    "  for name in ([_d[m]] if m in _d else [])+[m]:\n"
                    "    try: v=M.version(name); break\n"
                    "    except Exception: pass\n"
                    "  if v is None:\n"
                    "    try:\n"
                    "      mod=importlib.import_module(m); v=getattr(mod,'__version__','ok')\n"
                    "    except Exception: v='X'\n"
                    "  print(m, v)\n") % (_dist, [m for _, m in _DEPS])
            try:
                r = subprocess.run([self.py, "-c", code], capture_output=True, text=True, timeout=40,
                                   creationflags=0x08000000 if os.name == "nt" else 0)
                for line in r.stdout.splitlines():
                    p = line.split()
                    if len(p) >= 2:
                        out[p[0]] = p[1]
            except Exception:
                pass
        self.done.emit(out)


class SettingsTab(QWidget):
    def __init__(self, settings, theming=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.theming = theming
        self.api_fields = {}
        self._dep_labels = {}

        root = QVBoxLayout(self); root.setContentsMargins(8, 8, 8, 8)
        self.tabs = QTabWidget(); self.tabs.setObjectName("settabs")
        self.tabs.addTab(self._tab_engines_system(), tr("Moteurs && Système", "Engines && System"))
        self.tabs.addTab(self._tab_appearance(), tr("Apparence && API", "Appearance && API"))
        self.tabs.addTab(self._tab_lang_notif(), tr("Langue && Notif.", "Language && Notif."))
        self.tabs.addTab(self._tab_gallery(), tr("Galerie && Patch TB", "Gallery && TB Patch"))
        self._about_index = self.tabs.count()
        self.tabs.addTab(self._tab_about(), tr("À Propos", "About"))
        self.tabs.setCurrentIndex(0)  # ouvre sur Moteurs & Système
        self.tabs.currentChanged.connect(self._on_subtab_changed)
        root.addWidget(self.tabs)
        self.setStyleSheet("QTabWidget#settabs::tab-bar { alignment: center; }"
                           " QGroupBox { background: transparent; }")

        self._refresh_paths()
        self._adv = _AdvisoryWorker(); self._adv.done.connect(self._render_advisory); self._adv.start()
        self._deps = _DepsWorker(self._engine_python()); self._deps.done.connect(self._render_deps); self._deps.start()
        # Clean shutdown: wait for the worker threads before the app quits
        # (avoids "QThread destroyed while running" if the window closes early).
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_workers)

    def _stop_workers(self):
        for w in (getattr(self, "_adv", None), getattr(self, "_deps", None)):
            try:
                if w is not None and w.isRunning():
                    w.wait(8000)
            except Exception:
                pass

    def _scroll(self, inner):
        s = QScrollArea(); s.setWidgetResizable(True); s.setFrameShape(QFrame.NoFrame); s.setWidget(inner)
        return s

    # ── Tab : Apparence ───────────────────────────────────────────────────────────
    def _tab_appearance(self) -> QWidget:
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(12)
        if self.theming is None:
            v.addWidget(QLabel(tr("Thème non disponible.", "Theme unavailable."))); v.addStretch(1); return self._scroll(inner)
        box = QGroupBox(tr("Apparence / Thème (appliqué en direct)", "Appearance / Theme (applied live)")); bv = QVBoxLayout(box)
        prow = QHBoxLayout(); prow.addWidget(QLabel(tr("Accent :", "Accent:")))
        for label, hexv in theme.PRESET_ACCENTS.items():
            b = QPushButton(label); b.setStyleSheet(f"background:{hexv}; color:#fff;")
            b.clicked.connect(lambda _=False, h=hexv: self._set_accent(h)); prow.addWidget(b)
        prow.addStretch(1); bv.addLayout(prow)
        row2 = QHBoxLayout()
        pick = QPushButton(tr("Couleur perso…", "Custom color…")); pick.setObjectName("ghost"); pick.clicked.connect(self._pick_accent)
        row2.addWidget(pick)
        self._accent_lbl = QLabel(self.theming.accent()); row2.addWidget(self._accent_lbl); row2.addStretch(1)
        row2.addWidget(QLabel(tr("Mode :", "Mode:")))
        self._mode = QComboBox(); self._mode.addItems([tr("Sombre", "Dark"), tr("Clair", "Light")]); self._mode.setFixedWidth(110)
        self._mode.setCurrentIndex(0 if self.theming.is_dark() else 1)
        self._mode.currentIndexChanged.connect(lambda i: self.theming.set_dark(i == 0))
        row2.addWidget(self._mode); bv.addLayout(row2)
        # Taille interface (zoom global — avantage Qt, live)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel(tr("Taille interface :", "UI size:")))
        self._scale = QComboBox()
        self._scales = [("90 %", 0.9), ("100 %", 1.0), ("110 %", 1.1), ("125 %", 1.25), ("150 %", 1.5)]
        self._scale.addItems([s for s, _ in self._scales]); self._scale.setFixedWidth(110)
        cur = getattr(self.theming, "scale", lambda: 1.0)()
        self._scale.setCurrentIndex(min(range(len(self._scales)), key=lambda i: abs(self._scales[i][1] - cur)))
        self._scale.currentIndexChanged.connect(self._on_scale)
        # F24: says what it scales
        self._scale.setToolTip(tr("Agrandit le texte de l'interface. Les largeurs fixes de certains "
                                  "champs et boutons ne changent pas.",
                                  "Enlarges the interface text. Some fields and buttons keep a fixed width."))
        row3.addWidget(self._scale); row3.addStretch(1)
        reset = QPushButton(tr("↺ Réinitialiser l'apparence", "↺ Reset appearance")); reset.setObjectName("ghost")
        reset.setStyleSheet('font-family:"Segoe UI Emoji"; font-size:13px;'); reset.clicked.connect(self._reset_appearance)
        row3.addWidget(reset)
        bv.addLayout(row3)
        v.addWidget(box)
        v.addWidget(self._api_section())
        v.addStretch(1)
        return self._scroll(inner)

    def _on_scale(self, i):
        if hasattr(self.theming, "set_scale"):
            self.theming.set_scale(self._scales[i][1])

    def _reset_appearance(self):
        self._set_accent(theme.DEFAULT_ACCENT)
        if hasattr(self.theming, "set_dark"):
            self.theming.set_dark(True); self._mode.setCurrentIndex(0)
        if hasattr(self.theming, "set_scale"):
            self.theming.set_scale(1.0); self._scale.setCurrentIndex(1)

    def _set_accent(self, hexv):
        self.theming.set_accent(hexv); self._accent_lbl.setText(hexv); self.settings.set("qt_accent", hexv)

    def _pick_accent(self):
        c = QColorDialog.getColor(QColor(self.theming.accent()), self, tr("Couleur d'accent", "Accent color"))
        if c.isValid():
            self._set_accent(c.name())

    # ── Tab : Moteurs & Système ───────────────────────────────────────────────────
    def _tab_engines_system(self) -> QWidget:
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(12)
        # System config row
        sysb = QGroupBox(tr("Configuration Système & Runtime", "System & Runtime Configuration")); sv = QVBoxLayout(sysb)
        r = QHBoxLayout()
        r.addWidget(QLabel(tr("Python (venv moteur) :", "Python (engine venv):")))
        self.py_edit = QLineEdit(self._engine_python()); r.addWidget(self.py_edit, 1)
        pb = QPushButton("📂"); pb.setObjectName("ghost"); pb.setFixedWidth(40); pb.setStyleSheet(_EMOJI)
        pb.setFixedHeight(self.py_edit.sizeHint().height())
        pb.clicked.connect(self._pick_py); r.addWidget(pb)
        # F16: every engine action uses the shared venv when it exists — the field shows
        # that path instead of accepting a value nothing reads
        try:
            from src.core import engine_paths as _ep
            _venv = _ep.shared_venv_python()
        except Exception:
            _venv = ""
        if _venv and os.path.isfile(_venv):
            self.py_edit.setReadOnly(True); pb.setEnabled(False)
            _tip = tr("Python du venv partagé, utilisé par l'entraînement, l'inférence et l'installation "
                      "des dépendances. Non modifiable tant que ce venv existe.",
                      "Shared venv Python, used for training, inference and dependency installs. "
                      "Not editable while this venv exists.")
            self.py_edit.setToolTip(_tip); pb.setToolTip(_tip)
        r.addSpacing(16)
        self.chk_aida = QCheckBox(tr("Activer AIDA64", "Enable AIDA64"))
        # Auto-activation au 1er run si AIDA64 est détecté (process en cours d'exécution).
        _aida_known = self.settings.get("use_aida64", "") in ("true", "false")
        _aida_on = self.settings.get("use_aida64") == "true"
        if not _aida_known and _aida64_detected():
            _aida_on = True
            self.settings.set("use_aida64", "true")
            self.chk_aida.setToolTip(tr("AIDA64 détecté → activé automatiquement.", "AIDA64 detected → enabled automatically."))
        self.chk_aida.setChecked(_aida_on)
        self.chk_aida.stateChanged.connect(lambda: self.settings.set("use_aida64", "true" if self.chk_aida.isChecked() else "false"))
        r.addWidget(self.chk_aida)
        # « UI fluide » retiré en QT : sans effet (Qt rend nativement fluide ;
        # le flag perf_smooth_ui n'était lu que par l'ancienne UI CustomTkinter).
        sv.addLayout(r)
        # Python portable status (le PORTABLE runtimes/python-3.12.9, pas le venv).
        from src.core import engine_installer as _ei
        ok = _ei.is_portable_python_installed()
        st = QLabel((tr("✅ Python Portable (Prêt)", "✅ Portable Python (Ready)") if ok else tr("❌ Python Portable manquant", "❌ Portable Python missing")))
        st.setStyleSheet(f"color:{'#2ecc71' if ok else '#e74c3c'}; font-weight:700;"); sv.addWidget(st)
        v.addWidget(sysb)
        # GPU advisory
        adv = QGroupBox("GPU & PyTorch"); av = QVBoxLayout(adv)
        self._adv_body = QLabel(tr("Détection GPU…", "GPU Detection…")); self._adv_body.setStyleSheet("color:gray;"); self._adv_body.setWordWrap(True)
        av.addWidget(self._adv_body)
        # Boutons : python portable + (ré)install du PyTorch CUDA assorti au GPU.
        gpu_btns = QHBoxLayout()
        bpy = QPushButton(tr("📥 Python portable", "📥 Portable Python")); bpy.setStyleSheet(_EMOJI)
        bpy.setToolTip(tr("Télécharger le Python 3.12 portable (runtimes/) — requis avant d'installer un moteur.", "Download portable Python 3.12 (runtimes/) — required before installing an engine."))
        bpy.clicked.connect(self._install_portable_python); gpu_btns.addWidget(bpy)
        btorch = QPushButton(tr("🔧 Installer / MAJ PyTorch (GPU)", "🔧 Install / Update PyTorch (GPU)")); btorch.setStyleSheet(_EMOJI)
        btorch.setToolTip(tr("Réinstalle le build PyTorch CUDA assorti à ce GPU (cu128 Blackwell / cu126 …) dans le venv partagé.", "Reinstalls the CUDA PyTorch build matching this GPU (cu128 Blackwell / cu126 …) into the shared venv."))
        btorch.clicked.connect(self._install_torch_gpu); gpu_btns.addWidget(btorch)
        gpu_btns.addStretch(1)
        av.addLayout(gpu_btns)
        v.addWidget(adv)
        # Two engine columns (deps table + install controls) inside one framed box
        eg = QGroupBox(tr("Centre d'installation des moteurs", "Engine Installation Center")); egv = QVBoxLayout(eg)
        cols = QHBoxLayout(); cols.setSpacing(12)
        cols.addWidget(self._engine_column("NeoSR"), 1)
        cols.addWidget(self._engine_column("TraiNNer-Redux"), 1)
        egv.addLayout(cols)
        # Shared-venv pip dependency actions.
        dep_row = QHBoxLayout()
        dep_row.addWidget(QLabel(tr("Dépendances (venv partagé) :", "Dependencies (shared venv):")))
        bmiss = QPushButton(tr("⬇ Installer les manquants", "⬇ Install missing"))
        bmiss.setStyleSheet('background:#e67e22;color:#fff;border-radius:6px;padding:5px 12px; font-family:"Segoe UI Emoji"; font-size:13px;')
        bmiss.clicked.connect(lambda: self._pip_deps(missing_only=True)); dep_row.addWidget(bmiss)
        ball = QPushButton(tr("♻ Réinstaller toutes les deps", "♻ Reinstall all deps"))
        ball.setStyleSheet('background:#9b59b6;color:#fff;border-radius:6px;padding:5px 12px; font-family:"Segoe UI Emoji"; font-size:13px;')
        ball.clicked.connect(lambda: self._pip_deps(missing_only=False)); dep_row.addWidget(ball)
        dep_row.addStretch(1)
        egv.addLayout(dep_row)
        v.addWidget(eg)
        v.addWidget(self._paths_box())
        v.addStretch(1)
        return self._scroll(inner)

    def _actions_chemins(self) -> QGroupBox:
        box = QGroupBox(tr("Actions moteurs & Chemins", "Engine Actions & Paths")); v = QVBoxLayout(box)
        for name in ("NeoSR", "TraiNNer-Redux"):
            path = self._engine_path(name)
            installed = os.path.isfile(os.path.join(path, "train.py"))
            r = QHBoxLayout()
            lbl = QLabel(name); lbl.setStyleSheet("color:#3b8ed0; font-weight:700;"); lbl.setFixedWidth(130)
            r.addWidget(lbl)
            if installed:
                pull = QPushButton("🔄 PULL"); pull.setStyleSheet('background:#2980b9;color:#fff;border-radius:6px;padding:5px 10px; font-family:"Segoe UI Emoji"; font-size:13px;')
                pull.clicked.connect(lambda _=False, p=path, n=name: self._update_engine(n, p)); pull.setToolTip(self._update_tip(name)); r.addWidget(pull)
                rb = QPushButton(tr("⚠️ Forcer Réinstallation", "⚠️ Force Reinstall")); rb.setStyleSheet('background:#c0392b;color:#fff;border-radius:6px;padding:5px 10px; font-family:"Segoe UI Emoji"; font-size:13px;')
                rb.clicked.connect(lambda _=False, n=name, p=path: self._reinstall(n, p)); r.addWidget(rb)
                r.addStretch(1)
                op = QPushButton("📂"); op.setObjectName("ghost"); op.setFixedWidth(40); op.setStyleSheet(_EMOJI)
                from qt_ui.widgets import same_height; same_height(pull, rb, op)   # same height as PULL / Reinstall
                op.clicked.connect(lambda _=False, p=path: os.startfile(p) if os.path.isdir(p) else None); r.addWidget(op)
            else:
                inst = QPushButton(tr("⬇ INSTALLER", "⬇ INSTALL")); inst.setStyleSheet('background:#e67e22;color:#fff;border-radius:6px;padding:5px 12px; font-family:"Segoe UI Emoji"; font-size:13px;')
                inst.clicked.connect(lambda _=False, n=name, p=path: self._install(n, p)); r.addWidget(inst); r.addStretch(1)
            v.addLayout(r)
        self.paths_view = QPlainTextEdit(); self.paths_view.setReadOnly(True); self.paths_view.setMaximumHeight(120)
        self.paths_view.setFont(QFont("Consolas", 10)); v.addWidget(self.paths_view)
        ref = QPushButton(tr("Rafraîchir", "Refresh")); ref.setObjectName("ghost"); ref.clicked.connect(self._refresh_paths); v.addWidget(ref)
        return box

    def _engine_column(self, name) -> QFrame:
        path = self._engine_path(name)
        installed = os.path.isfile(os.path.join(path, "train.py"))
        col = QFrame(); col.setObjectName("engcol")
        col.setStyleSheet("QFrame#engcol{background:transparent;border:1px solid #3a3a4a;border-radius:8px;}")
        v = QVBoxLayout(col); v.setContentsMargins(12, 10, 12, 10); v.setSpacing(6)
        head = QHBoxLayout()
        ttl = QLabel(name); ttl.setStyleSheet("color:#3b8ed0; font-weight:700; font-size:15px;"); head.addWidget(ttl); head.addStretch(1)
        stt = QLabel(tr("✅ INSTALLÉ", "✅ INSTALLED") if installed else tr("❌ ABSENT", "❌ MISSING"))
        stt.setStyleSheet(f"color:{'#2ecc71' if installed else '#e74c3c'}; font-weight:700;"); head.addWidget(stt)
        v.addLayout(head)
        if installed:
            info = QLabel(tr("Taille : ", "Size: ") + f"{_folder_size(path)}"); info.setStyleSheet("color:gray; font-size:11px;"); v.addWidget(info)
        # deps grid (system tools + packages)
        g = QGridLayout(); g.setVerticalSpacing(3); v.addLayout(g)
        row = 0
        for tool in ("git", "ffmpeg", "nvidia-smi"):
            if tool == "ffmpeg":
                try:  # system ffmpeg, or the portable one the engine install brings
                    from src.core.lq_generator import resolve_ffmpeg
                    present = resolve_ffmpeg() is not None
                except Exception:
                    present = shutil.which(tool) is not None
            else:
                present = shutil.which(tool) is not None
            n = QLabel(tool.upper()); n.setFont(QFont("Consolas", 9))
            if tool == "git" and not present:
                # optional: engines are downloaded as source archives without it
                val = QLabel(tr("optionnel", "optional")); val.setStyleSheet("color:gray;")
            else:
                val = QLabel("☑ OK" if present else "✕")
                val.setStyleSheet("color:#2ecc71;" if present else "color:#e74c3c;")
            val.setFont(QFont("Consolas", 9)); val.setAlignment(Qt.AlignRight)
            g.addWidget(n, row, 0); g.addWidget(val, row, 1); row += 1
        for disp, mod in _DEPS:
            n = QLabel(disp.upper()); n.setFont(QFont("Consolas", 9))
            val = QLabel("…"); val.setFont(QFont("Consolas", 9)); val.setAlignment(Qt.AlignRight); val.setStyleSheet("color:#2ecc71;")
            self._dep_labels.setdefault(mod, []).append(val)
            g.addWidget(n, row, 0); g.addWidget(val, row, 1); row += 1
        g.setColumnStretch(1, 1)
        # mini action frame for THIS engine (PULL / Reinstall / open / install)
        act = QFrame(); act.setStyleSheet("QFrame{border:1px solid #2a2a3a;border-radius:6px;}")
        ah = QHBoxLayout(act); ah.setContentsMargins(8, 6, 8, 6)
        if installed:
            pull = QPushButton("🔄 PULL"); pull.setStyleSheet('background:#2980b9;color:#fff;border-radius:6px;padding:5px 10px; font-family:"Segoe UI Emoji"; font-size:13px;')
            pull.clicked.connect(lambda _=False, p=path, n=name: self._update_engine(n, p)); pull.setToolTip(self._update_tip(name)); ah.addWidget(pull)
            rb = QPushButton(tr("⚠️ Forcer Réinstallation", "⚠️ Force Reinstall")); rb.setStyleSheet('background:#c0392b;color:#fff;border-radius:6px;padding:5px 10px; font-family:"Segoe UI Emoji"; font-size:13px;')
            rb.clicked.connect(lambda _=False, n=name, p=path: self._reinstall(n, p)); ah.addWidget(rb)
            ah.addStretch(1)
            op = QPushButton("📂"); op.setObjectName("ghost"); op.setFixedWidth(40); op.setStyleSheet(_EMOJI)
            op.clicked.connect(lambda _=False, p=path: os.startfile(p) if os.path.isdir(p) else None); ah.addWidget(op)
            from qt_ui.widgets import same_height; same_height(pull, rb, op)
        else:
            inst = QPushButton(tr("⬇ INSTALLER", "⬇ INSTALL")); inst.setStyleSheet('background:#e67e22;color:#fff;border-radius:6px;padding:5px 12px; font-family:"Segoe UI Emoji"; font-size:13px;')
            inst.clicked.connect(lambda _=False, n=name, p=path: self._install(n, p)); ah.addWidget(inst); ah.addStretch(1)
        v.addWidget(act)
        v.addStretch(1)
        return col

    def _engine_path(self, name):
        try:
            from src.core import engine_paths as ep
            return ep.neosr_path() if name == "NeoSR" else ep.redux_path()
        except Exception:
            _h = os.path.expanduser("~")
            _b = os.path.join(_h, "USRS")
            return os.path.join(_b, name)

    def _render_advisory(self, rec):
        if not rec or not rec.get("gpu_name"):
            self._adv_body.setText(tr("(GPU non détecté)", "(GPU not detected)")); return
        has_tc = rec.get("has_tensor_cores", False)
        _tc = tr("✅ Tensor Cores", "✅ Tensor Cores") if has_tc else tr("⚠️ Sans Tensor Cores", "⚠️ No Tensor Cores")
        lines = [f"🖥️  {rec.get('gpu_name','?')}   {_tc}   {rec.get('gpu_gen','')}"]
        if rec.get("torch_version"):
            lines.append(tr("PyTorch recommandé : ", "Recommended PyTorch: ") + f"{rec['torch_version']} + CUDA {rec.get('cuda_tag','?')}")
        if rec.get("features"):
            lines.append("✅ " + "  ·  ".join(rec["features"]))
        for lim in rec.get("limitations", []):
            lines.append(f"⚠️  {lim}")
        if rec.get("note"):
            lines.append(rec["note"])
        if rec.get("driver_warning"):
            lines.append("❌ " + rec["driver_warning"])
        self._adv_body.setStyleSheet("color:#e74c3c;" if rec.get("driver_warning") else "color:#ddd;")
        self._adv_body.setText("\n".join(lines))

    def _render_deps(self, vers):
        self._dep_versions = dict(vers)
        for mod, lbls in self._dep_labels.items():
            v = vers.get(mod, "X")
            for lbl in lbls:
                lbl.setText(v if v not in ("?", "X") else "—")
                lbl.setStyleSheet("color:#2ecc71;" if v not in ("?", "X") else "color:#e74c3c;")

    # import name → pip package name
    _PIP_NAMES = {
        "torch": "torch", "torchvision": "torchvision", "cv2": "opencv-python",
        "numpy": "numpy", "yaml": "pyyaml", "scipy": "scipy", "pywt": "PyWavelets",
        "tqdm": "tqdm", "rich": "rich", "tensorboard": "tensorboard",
        "PIL": "pillow", "ema_pytorch": "ema-pytorch",
    }

    def _pip_deps(self, missing_only=True):
        py = self._engine_python()
        if not py or not os.path.isfile(py):
            QMessageBox.critical(self, tr("Dépendances", "Dependencies"), tr("Python du venv moteur introuvable.", "Engine venv Python not found.")); return
        vers = getattr(self, "_dep_versions", {})
        # torch/torchvision are managed by the engine install (CUDA build) — never touch here.
        skip = {"torch", "torchvision"}
        if missing_only:
            mods = [m for _, m in _DEPS if m not in skip and vers.get(m, "X") in ("X", "?")]
            if not mods:
                QMessageBox.information(self, tr("Dépendances", "Dependencies"), tr("Aucune dépendance manquante. ✅", "No missing dependency. ✅")); return
            pkgs = [self._PIP_NAMES.get(m, m) for m in mods]
            args = ["-m", "pip", "install"] + pkgs
            title = tr("Installer dépendances manquantes", "Install missing dependencies")
        else:
            pkgs = [self._PIP_NAMES.get(m, m) for _, m in _DEPS if m not in skip]
            if QMessageBox.question(self, tr("Réinstaller les dépendances", "Reinstall dependencies"),
                                    tr(f"Réinstaller/mettre à jour {len(pkgs)} paquets (hors torch) ?",
                                       f"Reinstall/update {len(pkgs)} packages (excluding torch)?")) != QMessageBox.Yes:
                return
            # F27: --force-reinstall, otherwise a package already at the target version
            # is left as it is (no-deps: torch is not touched through a dependency)
            args = ["-m", "pip", "install", "--upgrade", "--force-reinstall", "--no-deps"] + pkgs
            title = tr("Réinstaller toutes les dépendances", "Reinstall all dependencies")
        _ConsoleDialog(title, py, args, None, self).exec()
        self._reassert_cuda_torch(py)
        # refresh the dep table after the install window closes
        self._deps = _DepsWorker(py); self._deps.done.connect(self._render_deps); self._deps.start()

    def _reassert_cuda_torch(self, py):
        """Some deps (ema-pytorch, timm…) can silently pull a CPU-only torch build as
        a transitive dependency. Reassert the CUDA build LAST, exactly like the CTk
        app's install_missing_deps_only did — otherwise the venv can regress to CPU
        torch after "Installer/Réinstaller dépendances" without any visible error."""
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=8)
            has_cuda = r.returncode == 0
        except Exception:
            has_cuda = False
        if not has_cuda:
            return
        try:
            from src.core.compute_estimator import get_pytorch_recommendation
            rec = get_pytorch_recommendation()
            torch_pkg, tv_pkg = rec["install_pkgs"][0], rec["install_pkgs"][1]
            whl_url = rec["whl_url"]
        except Exception:
            torch_pkg, tv_pkg, whl_url = "torch==2.7.0", "torchvision==0.22.0", "https://download.pytorch.org/whl/cu126"
        subprocess.run([py, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], capture_output=True)
        args = ["-m", "pip", "install", torch_pkg, tv_pkg, "--index-url", whl_url, "--no-cache-dir"]
        _ConsoleDialog(tr("Réassertion PyTorch CUDA", "Reasserting CUDA PyTorch"), py, args, None, self).exec()

    def _update_tip(self, name):
        from src.core import engine_paths as _ep
        pin = _ep.REDUX_GIT_PIN if name == "TraiNNer-Redux" else _ep.NEOSR_GIT_PIN
        return (tr(f"Récupère et repositionne le moteur sur la version testée ({pin[:8]}).",
                   f"Fetches and puts the engine back on the tested version ({pin[:8]}).") if pin else
                tr("git pull de la branche suivie.", "git pull of the tracked branch."))

    def _update_engine(self, name, path):
        """F26: Redux is installed on a tested commit (detached HEAD); a bare `git pull`
        failed there, or left the tested version on a branch. Update = fetch + checkout
        of that same commit, like the installer."""
        from src.core import engine_paths as _ep
        pin = _ep.REDUX_GIT_PIN if name == "TraiNNer-Redux" else _ep.NEOSR_GIT_PIN
        if pin:
            self._git(["cmd", "/c", f"git fetch origin && git checkout {pin}"], path, f"Update {name}")
        else:
            self._git(["git", "pull"], path, f"Update {name}")

    def _git(self, args, cwd, title):
        # Installs made without git (source archive) have no .git: "update" then means
        # re-running the installer, which re-downloads the tested source version.
        if not shutil.which("git") or not os.path.isfile(os.path.join(cwd, ".git", "HEAD")):
            name = "NeoSR" if "neosr" in os.path.basename(os.path.normpath(cwd)).lower() else "TraiNNer-Redux"
            self._install(name, cwd)
            return
        _ConsoleDialog(title, args[0], args[1:], cwd, self).exec()

    def _refresh_deps_table(self):
        self._deps = _DepsWorker(self._engine_python()); self._deps.done.connect(self._render_deps); self._deps.start()

    def _rebuild_engines_tab(self):
        """Reconstruit l'onglet « Moteurs & Système » après une install → statuts
        python portable / moteurs INSTALLÉ / deps à jour SANS redémarrer l'app."""
        try:
            idx = 0
            self._dep_labels = {}        # repart à zéro (sinon accumule des labels morts)
            self.tabs.removeTab(idx)
            self.tabs.insertTab(idx, self._tab_engines_system(), tr("Moteurs && Système", "Engines && System"))
            self.tabs.setCurrentIndex(idx)
            self._refresh_paths()
            # relance les workers (le rebuild ne fait que construire les widgets vides)
            self._adv = _AdvisoryWorker(); self._adv.done.connect(self._render_advisory); self._adv.start()
            self._deps = _DepsWorker(self._engine_python()); self._deps.done.connect(self._render_deps); self._deps.start()
        except Exception:
            self._refresh_deps_table()

    def _install_portable_python(self):
        """Télécharge le Python 3.12 portable (1er pas du bootstrap)."""
        from src.core import engine_installer as ei
        if ei.is_portable_python_installed():
            if QMessageBox.question(self, tr("Python portable", "Portable Python"),
                                    tr("Python portable déjà présent. Réinstaller ?", "Portable Python already present. Reinstall?")) != QMessageBox.Yes:
                return
        res = ei.build_portable_python_script()
        if res is None:
            QMessageBox.critical(self, tr("Python introuvable", "Python not found"),
                                 tr("Aucun Python système trouvé pour amorcer l'installation.\n"
                                    "Installe Python 3.12 depuis python.org, puis relance.",
                                    "No system Python found to bootstrap the install.\n"
                                    "Install Python 3.12 from python.org, then relaunch."))
            return
        cmd, cwd = res
        _ConsoleDialog(tr("Installation Python portable", "Portable Python installation"), cmd[0], cmd[1:], cwd, self).exec()

    def _install_torch_gpu(self):
        """(Ré)installe le build PyTorch CUDA assorti au GPU détecté."""
        from src.core import engine_installer as ei
        if not ei.is_venv_ready():
            QMessageBox.warning(self, tr("Venv absent", "Venv missing"),
                                tr("Le venv partagé n'existe pas encore. Installe d'abord un moteur.", "The shared venv does not exist yet. Install an engine first."))
            return
        if QMessageBox.question(self, tr("PyTorch GPU", "GPU PyTorch"),
                                tr("Réinstaller le PyTorch CUDA assorti à ce GPU dans le venv partagé ?\n"
                                   "(désinstalle le torch courant puis télécharge le build GPU, ~2.8 GB)",
                                   "Reinstall the CUDA PyTorch build matching this GPU into the shared venv?\n"
                                   "(uninstalls current torch then downloads the GPU build, ~2.8 GB)")
                                ) != QMessageBox.Yes:
            return
        res = ei.build_torch_install_script()
        if res is None:
            QMessageBox.critical(self, tr("Erreur", "Error"), tr("Venv introuvable.", "Venv not found.")); return
        cmd, cwd = res
        _ConsoleDialog(tr("Installer / MAJ PyTorch (GPU)", "Install / Update PyTorch (GPU)"), cmd[0], cmd[1:], cwd, self).exec()
        self._rebuild_engines_tab()

    def _install(self, name, path):
        """Bootstrap complet (comme le CTk) : python portable (si absent) →
        venv partagé + dépendances + PyTorch CUDA GPU-matched + clone du moteur.
        Le 2e moteur skip le torch déjà installé."""
        from src.core import engine_installer as ei
        # git is optional: without it the installer downloads the engine source archive.
        try:
            from src.core.compute_estimator import get_pytorch_recommendation
            _rec = get_pytorch_recommendation() or {}
        except Exception:
            _rec = {}
        if not _rec.get("compute_sm") and not shutil.which("nvidia-smi"):
            if QMessageBox.question(self, tr("GPU NVIDIA", "NVIDIA GPU"),
                                    tr("Aucun GPU NVIDIA détecté (nvidia-smi introuvable).\n"
                                       "L'entraînement demande une carte NVIDIA avec un driver récent.\n\nInstaller quand même ?",
                                       "No NVIDIA GPU detected (nvidia-smi not found).\n"
                                       "Training needs an NVIDIA card with a recent driver.\n\nInstall anyway?")) != QMessageBox.Yes:
                return
        elif _rec.get("driver_warning"):
            if QMessageBox.question(self, tr("Driver NVIDIA trop ancien", "NVIDIA driver too old"),
                                    _rec["driver_warning"] + "\n\n" +
                                    tr("Mets à jour le driver (nvidia.com/drivers) puis relance l'installation.\n\nInstaller quand même ?",
                                       "Update the driver (nvidia.com/drivers), then run the install again.\n\nInstall anyway?")) != QMessageBox.Yes:
                return
        # 0) bundle A : copier le python embarqué si présent (skip download) + dossiers workspace
        ei.ensure_portable_python_bundled()
        ei.ensure_workspace_folders()
        # 1) python portable d'abord si toujours absent (mode dev / pas de bundle)
        if not ei.is_portable_python_installed():
            res = ei.build_portable_python_script()
            if res is None:
                QMessageBox.critical(self, tr("Python introuvable", "Python not found"),
                                     tr("Aucun Python système pour amorcer.\nInstalle Python 3.12 depuis python.org, puis relance.",
                                        "No system Python to bootstrap.\nInstall Python 3.12 from python.org, then relaunch."))
                return
            cmd, cwd = res
            _ConsoleDialog(tr("Installation Python portable", "Portable Python installation"), cmd[0], cmd[1:], cwd, self).exec()
            if not ei.is_portable_python_installed():
                QMessageBox.warning(self, "Python portable", tr("Installation interrompue ou échouée — voir le log.", "Installation interrupted or failed — see the log.")); return
        # 2) moteur (venv + deps + torch + clone)
        res = ei.build_engine_install_script(name)
        if res is None:
            QMessageBox.critical(self, "Install", tr("Python portable introuvable après installation.", "Portable Python not found after installation.")); return
        cmd, cwd = res
        _ConsoleDialog(tr(f"Install {name}", f"Install {name}"), cmd[0], cmd[1:], cwd, self).exec()
        self._rebuild_engines_tab()

    def _check_user_data_protection(self, path) -> bool:
        """Refuse to delete an engine folder that still holds trained-model data
        (datasets/experiments/datsheet/models/weights), mirroring the CTk guard."""
        if not os.path.exists(path):
            return True
        protected = ("datasets", "experiments", "datsheet", "models", "weights")
        found = [i for i in os.listdir(path) if i.lower() in protected]
        if found:
            QMessageBox.warning(self, tr("PROTECTION DONNÉES", "DATA PROTECTION"),
                                tr(f"Suppression BLOQUÉE : dossiers sensibles détectés ({found}).\n"
                                   "Déplace/archive ces dossiers ailleurs si tu veux vraiment réinstaller.",
                                   f"Deletion BLOCKED: sensitive folders detected ({found}).\n"
                                   "Move/archive those folders elsewhere if you really want to reinstall."))
            return False
        return True

    def _reinstall(self, name, path):
        if QMessageBox.question(self, tr("Réinstaller", "Reinstall"),
                                tr(f"Supprimer et recloner {name} ?\n(le venv partagé est conservé)",
                                   f"Delete and re-clone {name}?\n(the shared venv is kept)")) != QMessageBox.Yes:
            return
        if not self._check_user_data_protection(path):
            return
        shutil.rmtree(path, ignore_errors=True); self._install(name, path)

    def _paths_box(self) -> QGroupBox:
        box = QGroupBox(tr("Chemins USRS", "USRS Paths")); v = QVBoxLayout(box)
        self.paths_view = QPlainTextEdit(); self.paths_view.setReadOnly(True); self.paths_view.setMaximumHeight(120)
        self.paths_view.setFont(QFont("Consolas", 10)); v.addWidget(self.paths_view)
        ref = QPushButton(tr("Rafraîchir", "Refresh")); ref.setObjectName("ghost"); ref.clicked.connect(self._refresh_paths); v.addWidget(ref)
        return box

    def _refresh_paths(self):
        lines = []
        try:
            from src.core import engine_paths as ep
            lines.append(f"Base    : {ep.base_engine_path()}")
            lines.append(f"Runtimes: {ep.runtimes_path()}")
            lines.append(f"NeoSR   : {ep.neosr_path()}  ({'ok' if os.path.isdir(ep.neosr_path()) else 'absent'})")
            lines.append(f"Redux   : {ep.redux_path()}  ({'ok' if os.path.isdir(ep.redux_path()) else 'absent'})")
            lines.append(f"Python  : {ep.shared_venv_python()}")
        except Exception as e:
            lines.append(f"(engine_paths indisponible : {e})")
        self.paths_view.setPlainText("\n".join(lines))

    # ── Tab : Langue & Notif ──────────────────────────────────────────────────────
    def _tab_lang_notif(self) -> QWidget:
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(12)
        lb = QGroupBox("Langue / Language"); lh = QHBoxLayout(lb); lh.addWidget(QLabel(tr("Langue :", "Language:")))
        self.lang = QComboBox(); self.lang.addItems(["🇫🇷 Français", "🇬🇧 English"]); self.lang.setFixedWidth(160)
        self.lang.setCurrentIndex(0 if self.settings.get("qt_lang", "fr") == "fr" else 1)
        self.lang.currentIndexChanged.connect(self._set_lang)
        lh.addWidget(self.lang)
        lh.addWidget(QLabel(tr("(redémarrage pour tout appliquer)", "(restart to fully apply)")))
        lh.addStretch(1); v.addWidget(lb)

        nb = QGroupBox("Notifications Windows 11"); nv = QVBoxLayout(nb)
        self.notif = {}
        for key, label in _NOTIF:
            c = QCheckBox(label); c.setChecked(self.settings.get(key, "true") == "true")
            c.stateChanged.connect(lambda _=0, k=key, cb=c: self.settings.set(k, "true" if cb.isChecked() else "false"))
            self.notif[key] = c; nv.addWidget(c)
        tb = QPushButton(tr("Tester une notification", "Test a notification")); tb.setStyleSheet('background:#2980b9; color:#fff; border-radius:6px; padding:5px;')
        tb.clicked.connect(self._test_notif); nv.addWidget(tb)
        v.addWidget(nb)

        sb = QGroupBox(tr("Son de fin d'entraînement", "Training-end sound")); sv = QVBoxLayout(sb)
        self.snd_on = QCheckBox(tr("Jouer un son quand un entraînement se termine", "Play a sound when a training finishes"))
        self.snd_on.setChecked(self.settings.get("sound_on", "true") == "true")
        self.snd_on.stateChanged.connect(lambda: self.settings.set("sound_on", "true" if self.snd_on.isChecked() else "false"))
        sv.addWidget(self.snd_on)
        vr = QHBoxLayout(); vr.addWidget(QLabel(tr("Volume :", "Volume:")))
        self.vol = QSlider(Qt.Horizontal); self.vol.setRange(0, 100); self.vol.setValue(int(self.settings.get("sound_vol", "76") or 76))
        self.vol_lbl = QLabel(f"{self.vol.value()}%")
        self.vol.valueChanged.connect(lambda x: (self.vol_lbl.setText(f"{x}%"), self.settings.set("sound_vol", str(x))))
        vr.addWidget(self.vol, 1); vr.addWidget(self.vol_lbl); sv.addLayout(vr)
        ts = QPushButton(tr("🔊 Tester le son", "🔊 Test sound")); ts.setStyleSheet('background:#27ae60;color:#fff;border-radius:6px;padding:5px; font-family:"Segoe UI Emoji"; font-size:13px;')
        ts.clicked.connect(lambda _=False: self._test_sound("success")); sv.addWidget(ts)
        info = QLabel(tr("Fichier son : assets/success.wav", "Sound file: assets/success.wav")); info.setStyleSheet("color:gray; font-size:11px;"); sv.addWidget(info)
        v.addWidget(sb)

        eb = QGroupBox(tr("Sons d'événements", "Event sounds")); ev = QVBoxLayout(eb)
        self.evt = {}
        for key, label in _EVENT_SOUNDS:
            row = QHBoxLayout()
            c = QCheckBox(label); c.setChecked(self.settings.get(key, "true") == "true")
            c.stateChanged.connect(lambda _=0, k=key, cb=c: self.settings.set(k, "true" if cb.isChecked() else "false"))
            self.evt[key] = c; row.addWidget(c); row.addStretch(1)
            t = QPushButton("🔊 Test"); t.setFixedWidth(70)
            t.setStyleSheet('background:#27ae60;color:#fff;border-radius:6px;padding:3px; font-family:"Segoe UI Emoji"; font-size:13px;')
            # each Test plays ITS sound (they all played success.wav)
            t.clicked.connect(lambda _=False, k=key: self._test_sound(
                {"snd_error": "error", "snd_warning": "warning", "snd_about": "about"}.get(k, "success")))
            row.addWidget(t)
            ev.addLayout(row)
        ef = QLabel(tr("Fichiers : assets/error.WAV · warning.WAV · about.WAV", "Files: assets/error.WAV · warning.WAV · about.WAV")); ef.setStyleSheet("color:gray; font-size:11px;"); ev.addWidget(ef)
        v.addWidget(eb)
        v.addStretch(1)
        return self._scroll(inner)

    def _on_subtab_changed(self, idx):
        """Plays about.WAV when the À Propos sub-tab is opened — CTk bound this to
        the tab's <Map> event; the Qt port never wired it at all."""
        if idx == getattr(self, "_about_index", -1):
            try:
                from qt_ui.notify import play_sound
                play_sound(self.settings, "about", "snd_about")
            except Exception:
                pass

    def _set_lang(self, i):
        code = "fr" if i == 0 else "en"
        self.settings.set("qt_lang", code); self.settings.set("language", code)
        try:
            from qt_ui.i18n import set_lang
            set_lang(code)
        except Exception:
            pass
        QMessageBox.information(self, "Langue / Language",
                               "Langue enregistrée. Redémarre l'app pour l'appliquer partout.\n"
                               "Language saved. Restart the app to apply everywhere.")

    def _test_notif(self):
        try:
            from qt_ui.notify import notify
            notify(self.settings, "Universal SR Studio", "Notification de test.")
        except Exception:
            QMessageBox.information(self, "Notification", tr("Test (toast non branché).", "Test (toast not connected)."))

    def _test_sound(self, name="success"):
        """Plays the given sound at the current volume (same path as the real events)."""
        from qt_ui.theme import asset
        p = asset(f"{name}.WAV")
        if not (os.path.isfile(p) or os.path.isfile(p[:-4] + ".wav")):
            QMessageBox.information(self, tr("Son", "Sound"), tr("Fichier son introuvable.", "Sound file not found."))
            return
        from qt_ui.notify import play_sound
        play_sound(self.settings, name, force=True)

    # ── Clés API (section dans l'onglet Apparence & API) ─────────────────────────
    def _api_section(self) -> QGroupBox:
        box = QGroupBox(tr("Clés API pour la Vérification AI", "API Keys for AI Review")); v = QVBoxLayout(box)
        sub = QLabel(tr("Stockées localement (user_settings.json) — jamais envoyées ailleurs qu'au fournisseur choisi.", "Stored locally (user_settings.json) — never sent anywhere except the chosen provider."))
        sub.setStyleSheet("color:gray;"); v.addWidget(sub)
        g = QGridLayout(); v.addLayout(g)
        for i, (label, key, url) in enumerate(_API_PROVIDERS):
            from qt_ui.widgets import provider_label
            lb = QLabel(provider_label(label)); lb.setStyleSheet("font-weight:600;"); g.addWidget(lb, i, 0)
            e = QLineEdit(self.settings.get(key, "") or ""); e.setEchoMode(QLineEdit.Password)
            e.setPlaceholderText("sk-… / xai-… / AIza…"); self.api_fields[key] = e; g.addWidget(e, i, 1)
            lk = QLabel(f'<a href="{url}" style="color:#3b8ed0;">{url}</a>'); lk.setOpenExternalLinks(True)
            g.addWidget(lk, i, 2)
        g.setColumnStretch(1, 1)
        save = QPushButton(tr("Sauvegarder les clés API", "Save API keys")); save.clicked.connect(self._save_api)
        v.addWidget(save, alignment=Qt.AlignCenter)
        return box

    def _save_api(self):
        for key, e in self.api_fields.items():
            self.settings.set(key, e.text().strip())
        QMessageBox.information(self, tr("Clés API", "API Keys"), tr("Clés sauvegardées.", "Keys saved."))

    # ── Tab : Galerie & Patch TB ──────────────────────────────────────────────────
    def _tab_gallery(self) -> QWidget:
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(12)
        a = QGroupBox(tr("A. Serveur Galerie Web (sans modifier NeoSR)", "A. Web Gallery Server (without modifying NeoSR)")); av = QVBoxLayout(a)
        av.addWidget(QLabel(tr("Mini-serveur HTTP sur un dossier d'images. Mobile, auto-refresh, zoom. Ngrok optionnel.", "Mini HTTP server on an image folder. Mobile, auto-refresh, zoom. Ngrok optional.")))
        r = QHBoxLayout(); r.addWidget(QLabel(tr("Dossier :", "Folder:")))
        self.gal_dir = QLineEdit(self.settings.get("gallery_dir", "") or ""); r.addWidget(self.gal_dir, 1)
        gb = QPushButton("📂"); gb.setObjectName("ghost"); gb.setFixedWidth(40); gb.setStyleSheet(_EMOJI)
        gb.setFixedHeight(self.gal_dir.sizeHint().height())
        gb.clicked.connect(lambda: self.gal_dir.setText(QFileDialog.getExistingDirectory(self, tr("Dossier images", "Images folder")) or self.gal_dir.text()))
        r.addWidget(gb); av.addLayout(r)
        r2 = QHBoxLayout(); r2.addWidget(QLabel(tr("Port :", "Port:")))
        self.gal_port = QLineEdit(self.settings.get("gallery_port", "8765") or "8765"); self.gal_port.setFixedWidth(80)
        r2.addWidget(self.gal_port); r2.addSpacing(16)
        self.gal_ngrok = QCheckBox(tr("Tunnel Ngrok (accès distant)", "Ngrok Tunnel (remote access)"))
        self.gal_ngrok.setChecked(self.settings.get("gallery_ngrok") == "true"); r2.addWidget(self.gal_ngrok)
        r2.addStretch(1); av.addLayout(r2)
        self.gal_auto = {}
        for key, label in _GAL_AUTO:
            # opt-in for what launches something (they were wired to nothing before, with a
            # "true" default: wiring them as-is would start servers for everyone)
            c = QCheckBox(label); c.setChecked(self.settings.get(key, _GAL_AUTO_DEFAULTS.get(key, "false")) == "true")
            c.stateChanged.connect(lambda _=0, k=key, cb=c: self.settings.set(k, "true" if cb.isChecked() else "false"))
            self.gal_auto[key] = c; av.addWidget(c)
        br = QHBoxLayout()
        start = QPushButton(tr("▶ Démarrer", "▶ Start")); start.setStyleSheet('background:#27ae60;color:#fff;border-radius:6px;padding:5px 12px;')
        start.clicked.connect(self._gallery_start); br.addWidget(start)
        stop = QPushButton(tr("⏹ Arrêter", "⏹ Stop")); stop.setStyleSheet('background:#c0392b;color:#fff;border-radius:6px;padding:5px 12px;')
        stop.clicked.connect(self._gallery_stop); br.addWidget(stop)
        op = QPushButton(tr("🌐 Ouvrir", "🌐 Open")); op.setStyleSheet('background:#2980b9;color:#fff;border-radius:6px;padding:5px 12px; font-family:"Segoe UI Emoji"; font-size:13px;')
        # F21: open the URL the server really uses (port 0 = allocated port)
        op.clicked.connect(lambda: __import__("webbrowser").open(getattr(self, "_gal_url", "")
                                                                 or f"http://localhost:{self.gal_port.text()}")); br.addWidget(op)
        br.addStretch(1); av.addLayout(br)
        self.gal_state = QLabel(tr("État : Arrêté", "Status: Stopped")); self.gal_state.setStyleSheet("color:gray;"); av.addWidget(self.gal_state)
        qr_row = QHBoxLayout()
        self.gal_qr_lbl = QLabel(); self.gal_qr_lbl.setFixedSize(140, 140); qr_row.addWidget(self.gal_qr_lbl)
        self.gal_qr_text = QLabel(""); self.gal_qr_text.setStyleSheet("color:#3498db;"); self.gal_qr_text.setWordWrap(True)
        qr_row.addWidget(self.gal_qr_text, 1); av.addLayout(qr_row)
        v.addWidget(a)

        b = QGroupBox(tr("B. Patch NeoSR/Redux pour images TensorBoard", "B. NeoSR/Redux Patch for TensorBoard images")); bv = QVBoxLayout(b)
        bv.addWidget(QLabel(tr("Modifie le code moteur pour envoyer les images de validation dans TensorBoard. Réversible (.usr_bak).", "Modifies engine code to send validation images to TensorBoard. Reversible (.usr_bak).")))
        er = QHBoxLayout(); er.addWidget(QLabel(tr("Engine :", "Engine:")))
        self.patch_engine = QLineEdit(self._engine_path("NeoSR")); er.addWidget(self.patch_engine, 1)
        eb = QPushButton("📂"); eb.setObjectName("ghost"); eb.setFixedWidth(40); eb.setStyleSheet(_EMOJI)
        eb.setFixedHeight(self.patch_engine.sizeHint().height())
        eb.clicked.connect(lambda: self.patch_engine.setText(QFileDialog.getExistingDirectory(self, tr("Racine moteur", "Engine root")) or self.patch_engine.text()))
        er.addWidget(eb); bv.addLayout(er)
        sc = QHBoxLayout(); sc.addWidget(QLabel(tr("Raccourcis :", "Shortcuts:")))
        for nm in ("NeoSR", "TraiNNer-Redux"):
            sbn = QPushButton(nm); sbn.setObjectName("ghost")
            sbn.clicked.connect(lambda _=False, n=nm: self.patch_engine.setText(self._engine_path(n))); sc.addWidget(sbn)
        chk = QPushButton(tr("🔎 Vérifier statut", "🔎 Check status")); chk.setStyleSheet('background:#2980b9;color:#fff;border-radius:6px;padding:4px 10px; font-family:"Segoe UI Emoji"; font-size:13px;')
        from qt_ui.widgets import same_height; same_height(sbn, chk)   # same height as the shortcut buttons
        chk.clicked.connect(self._tb_patch_status); sc.addWidget(chk); sc.addStretch(1); bv.addLayout(sc)
        self.patch_status = QLabel(tr("○ Statut inconnu — cliquez sur Vérifier statut.", "○ Unknown status — click Check status."))
        self.patch_status.setStyleSheet("color:gray; font-size:11px;"); bv.addWidget(self.patch_status)
        pr = QHBoxLayout()
        ap = QPushButton(tr("✅ Appliquer le Patch", "✅ Apply Patch")); ap.setStyleSheet('background:#27ae60;color:#fff;border-radius:6px;padding:5px 12px; font-family:"Segoe UI Emoji"; font-size:13px;')
        ap.clicked.connect(lambda: self._tb_patch(True)); pr.addWidget(ap)
        rp = QPushButton(tr("✕ Retirer le Patch", "✕ Remove Patch")); rp.setStyleSheet('background:#c0392b;color:#fff;border-radius:6px;padding:5px 12px;')
        rp.clicked.connect(lambda: self._tb_patch(False)); pr.addWidget(rp); pr.addStretch(1)
        bv.addLayout(pr)
        v.addWidget(b); v.addStretch(1)
        return self._scroll(inner)

    def _gallery_start(self):
        self.settings.set("gallery_dir", self.gal_dir.text()); self.settings.set("gallery_port", self.gal_port.text())
        self.settings.set("gallery_ngrok", "true" if self.gal_ngrok.isChecked() else "false")   # F21
        d = self.gal_dir.text().strip()
        if not os.path.isdir(d):
            self.gal_state.setText(tr("État : ❌ dossier invalide", "Status: ❌ invalid folder")); return
        try:
            from src.core.gallery_server import get_server
            port = int(self.gal_port.text() or 8765)
            res = get_server().start(d, port=port, with_ngrok=self.gal_ngrok.isChecked()) or {}
            # GalleryServer.start() returns {ok, port, local_url, ngrok_url, error} — never
            # an "url" key. The old code only ever read .get("url", <hardcoded fallback>),
            # so a real failure (server already running, port busy, dir vanished) still
            # showed "✅ en ligne" with a guessed URL that may not even be listening
            # (2026-09-17 fix).
            if not res.get("ok"):
                self.gal_state.setText(tr("État : ❌ ", "Status: ❌ ") + str(res.get('error', tr('échec inconnu', 'unknown failure'))))
                return
            url = res.get("ngrok_url") or res.get("local_url") or f"http://localhost:{res.get('port') or port}"
            self._gal_url = url
            state = tr("État : ✅ en ligne — ", "Status: ✅ online — ") + str(url)
            if res.get("ngrok_warning"):   # F21: the tunnel failure was dropped
                state += tr("  ⚠ Ngrok : ", "  ⚠ Ngrok: ") + str(res["ngrok_warning"])
            self.gal_state.setText(state)
            self._show_gallery_qr(url)
        except Exception as e:
            self.gal_state.setText(tr("État : ❌ ", "Status: ❌ ") + str(e))

    def _show_gallery_qr(self, url):
        """Renders a scannable QR code for the gallery URL (was silently dropped in
        the Qt port — no way to open the mobile gallery without typing the URL)."""
        self.gal_qr_lbl.clear(); self.gal_qr_text.setText("")
        try:
            from src.core.qr_code import generate_qr_image, is_qrcode_available
            if not is_qrcode_available():
                return
            qr_path = os.path.join(os.path.expanduser("~"), ".usr_studio_qr.png")
            if generate_qr_image(url, qr_path, box_size=6):
                from PySide6.QtGui import QPixmap
                pm = QPixmap(qr_path)
                if not pm.isNull():
                    self.gal_qr_lbl.setPixmap(pm.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.gal_qr_text.setText(tr(f"📱 Scannez avec votre téléphone\n\nURL : {url}",
                                                 f"📱 Scan with your phone\n\nURL: {url}"))
        except Exception:
            pass

    def _gallery_stop(self):
        try:
            from src.core.gallery_server import get_server
            get_server().stop()
        except Exception:
            pass
        self.gal_state.setText(tr("État : Arrêté", "Status: Stopped"))

    def _tb_patch_status(self):
        try:
            from src.core.tb_image_patch import get_patch_status
            st = get_patch_status(self.patch_engine.text().strip())
            patched = st.get("patched") if isinstance(st, dict) else st
            self.patch_status.setText((tr("✅ Patché (images TensorBoard actives)", "✅ Patched (TensorBoard images active)") if patched
                                       else tr("○ Pas patché (prêt à l'emploi)", "○ Not patched (ready to use)")) + f"  ·  {st}")
        except Exception as e:
            self.patch_status.setText(f"❌ {e}")

    def _tb_patch(self, apply):
        root = self.patch_engine.text().strip()
        if not os.path.isdir(root):
            QMessageBox.warning(self, "Patch TB", tr("Dossier moteur invalide.", "Invalid engine folder.")); return
        try:
            if apply:
                from src.core.tb_image_patch import patch_engine
                ok, msg, *_ = patch_engine(root)  # (ok, msg, file): 3 values, 2 were unpacked
            else:
                from src.core.tb_image_patch import find_validation_file, unpatch_file
                f = find_validation_file(root)
                ok, msg = unpatch_file(f) if f else (False, tr("Fichier cible introuvable.", "Target file not found."))
            QMessageBox.information(self, "Patch TB", ("✅ " if ok else "❌ ") + str(msg))
            self._tb_patch_status()
        except Exception as e:
            QMessageBox.critical(self, "Patch TB", str(e))

    # ── Tab : À Propos (fidèle à la v2.5.8) ───────────────────────────────────────
    @staticmethod
    def _sec(text) -> QLabel:
        s = QLabel(text); s.setAlignment(Qt.AlignCenter)
        s.setStyleSheet("color:#888; font-style:italic; background:transparent;")
        return s

    def _thanks_rows(self, layout, rows, name_color, name_w=200):
        for name, url, desc in rows:
            r = QHBoxLayout(); r.setContentsMargins(20, 0, 20, 0)
            if url:
                lbl = QLabel(f'<a href="{url}" style="color:{name_color}; text-decoration:underline;">{name}</a>')
                lbl.setOpenExternalLinks(True)
            else:
                lbl = QLabel(name); lbl.setStyleSheet(f"color:{name_color}; font-weight:700;")
            lbl.setFixedWidth(name_w); lbl.setStyleSheet(lbl.styleSheet() + " background:transparent;")
            r.addWidget(lbl)
            d = QLabel(desc); d.setStyleSheet("color:#888; background:transparent;"); r.addWidget(d, 1)
            layout.addLayout(r)

    def _tab_about(self) -> QWidget:
        from PySide6.QtGui import QPixmap
        from qt_ui.theme import asset
        bg = ""
        for ext in ("about_bg.jpg", "about_bg.png"):
            cand = asset(ext)
            if os.path.isfile(cand):
                bg = cand; break
        inner = _AboutBg(bg)
        v = QVBoxLayout(inner); v.setContentsMargins(40, 18, 40, 18); v.setSpacing(5)

        logo = asset("icon.png")
        if os.path.isfile(logo):
            pm = QPixmap(logo)
            if not pm.isNull():
                lg = QLabel(); lg.setPixmap(pm.scaledToHeight(72, Qt.SmoothTransformation))
                lg.setAlignment(Qt.AlignCenter); lg.setStyleSheet("background:transparent;"); v.addWidget(lg)
        title = QLabel("Universal SR Studio"); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#3498db; font-size:24px; font-weight:800; background:transparent;"); v.addWidget(title)
        ver = QLabel("v2.6.0 — Super-Resolution Training Suite"); ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet("color:#888; background:transparent;"); v.addWidget(ver)
        sep = QFrame(); sep.setFixedHeight(2); sep.setStyleSheet("background:#3498db;"); v.addWidget(sep)
        desc = QLabel(tr("Application de configuration et d'entraînement de modèles de Super-Resolution "
                         "(NeoSR & TraiNNer-Redux).\n\nFonctionnalités : wizard guide, configuration avancée, "
                         "monitoring GPU temps réel, outils de conversion et comparaison.",
                         "Configuration and training app for Super-Resolution models "
                         "(NeoSR & TraiNNer-Redux).\n\nFeatures: guided wizard, advanced configuration, "
                         "real-time GPU monitoring, conversion and comparison tools."))
        desc.setAlignment(Qt.AlignCenter); desc.setWordWrap(True); desc.setStyleSheet("color:#888; background:transparent;")
        v.addWidget(desc)

        # ── Auteurs & Credits (panneau sombre #111128) ──
        cred = QFrame(); cred.setObjectName("credpanel")
        cred.setStyleSheet("QFrame#credpanel{background:rgba(17,17,40,0.80); border:1px solid #333355; border-radius:6px;}")
        cv = QVBoxLayout(cred); cv.setContentsMargins(15, 12, 15, 12); cv.setSpacing(7)
        ch = QLabel(tr("Auteurs & Credits", "Authors & Credits")); ch.setAlignment(Qt.AlignCenter)
        ch.setStyleSheet("color:#e74c3c; font-weight:700; font-size:14px; background:transparent;"); cv.addWidget(ch)
        for who, role in [
                ("CrysisJim", tr("Concept, Design, Direction, Tests", "Concept, Design, Direction, Testing")),
                ("Gemini 3.1 Pro", tr("Google — Base du code, Architecture initiale", "Google — Code base, initial architecture")),
                ("Grok 4.3", tr("xAI — Base du code, Contributions initiales", "xAI — Code base, initial contributions")),
                ("Claude Opus 5.5", tr("Anthropic — Architecture, Code principal, Refactoring",
                                       "Anthropic — Architecture, Main code, Refactoring")),
                ("GPT-6.1 Sol", tr("OpenAI — Audits de code, renfort sur les gros chantiers",
                                   "OpenAI — Code audits, extra hands on big work"))]:
            r = QHBoxLayout()
            a = QLabel(who); a.setFixedWidth(150); a.setStyleSheet("color:#fff; font-weight:700; background:transparent;")
            r.addWidget(a)
            rl = QLabel(role); rl.setStyleSheet("color:#888; background:transparent;"); r.addWidget(rl, 1)
            cv.addLayout(r)
        v.addWidget(cred)

        # ── Remerciements (panneau sombre #0d1117) ──
        th = QFrame(); th.setObjectName("thankspanel")
        th.setStyleSheet("QFrame#thankspanel{background:rgba(13,17,23,0.80); border:1px solid #2a4a2a; border-radius:6px;}")
        tv = QVBoxLayout(th); tv.setContentsMargins(12, 12, 12, 12); tv.setSpacing(6)
        tt = QLabel(tr("🙏 Remerciements", "🙏 Acknowledgements")); tt.setAlignment(Qt.AlignCenter)
        tt.setStyleSheet("color:#2ecc71; font-weight:700; font-size:13px; background:transparent;"); tv.addWidget(tt)
        tv.addWidget(self._sec(tr("— Moteurs d'entraînement —", "— Training engines —")))
        self._thanks_rows(tv, [
            ("neosr-project", "https://github.com/neosr-project/neosr", tr("NeoSR — moteur SR moderne (PyTorch)", "NeoSR — modern SR engine (PyTorch)")),
            ("the-database", "https://github.com/the-database/traiNNer-redux", tr("traiNNer-redux — fork avancé avec OTF, ECO, SpanF...", "traiNNer-redux — advanced fork with OTF, ECO, SpanF..."))],
            "#3498db", 150)
        tv.addWidget(self._sec(tr("— Architectures & Losses communautaires —", "— Community Architectures & Losses —")))
        self._thanks_rows(tv, [
            ("umzi2 (Umzi)", "https://github.com/umzi2", "SMoSR · SpanC (spanpp) · GFISRv2 · SparK Perceptual Loss"),
            ("zhengchen1999 / NeoSR team", "https://github.com/neosr-project/neosr", "CATANet (Token Aggregation + LRSA)"),
            ("XPixelGroup / XLAB", "https://github.com/XPixelGroup", "SwinIR · HAT · DRCT · RCAN · RealESRGAN"),
            ("xinntao / Real-ESRGAN", "https://github.com/xinntao/Real-ESRGAN", tr("RealESRGAN · ESRGAN · RRDB — base de l'écosystème SR", "RealESRGAN · ESRGAN · RRDB — foundation of the SR ecosystem")),
            ("BasicSR community", "https://github.com/XPixelGroup/BasicSR", tr("Framework BasicSR (losses, métriques, pipeline)", "BasicSR framework (losses, metrics, pipeline)")),
            ("2minkyulee", "https://github.com/2minkyulee/ECO", "ECO Training Mode (AAAI 2024)"),
            ("CF2lter", "", tr("WireSR (« 1x_Wired ») — anti-crénelage / traits fins, inclus avec son accord",
                               "WireSR (“1x_Wired”) — anti-aliasing / thin lines, included with his permission")),
            ("stinkybread", "https://github.com/stinkybread/FDAT-RD", tr("FDAT-RD — FDAT pour l'anime", "FDAT-RD — FDAT for anime")),
            ("Phhofm", "https://github.com/Phhofm", "ParagonSR · ParagonSR2 · AetherNet · HEART · NERVE"),
            ("Kim2091", "https://github.com/Kim2091", "TFDAT v2"),
            ("Rift73", "https://github.com/Rift73", "WSR · DUAL"),
            ("enhancr", "https://github.com/enhancr/figsr", "FIGSR")],
            "#e67e22", 200)
        tv.addWidget(self._sec(tr("— IA ayant participé à ce projet —", "— AI that contributed to this project —")))
        self._thanks_rows(tv, [
            ("Claude Opus 5.5  (Anthropic)", "", tr("Code principal, refactoring, intégration", "Main code, refactoring, integration")),
            ("GPT-6.1 Sol  (OpenAI)", "", tr("Audits de code, renfort sur les gros chantiers", "Code audits, extra hands on big work")),
            ("Gemini 3.1 Pro  (Google)", "", tr("Base du code, architecture initiale", "Code base, initial architecture")),
            ("Grok  (xAI)", "", tr("Contributions initiales", "Initial contributions"))],
            "#9b59b6", 220)
        tv.addWidget(self._sec(tr("— Outils & économies de tokens —", "— Tools & token savings —")))
        self._thanks_rows(tv, [
            ("Caveman (Julius Brussee)", "https://github.com/JuliusBrussee/caveman", tr("Mode réponse ultra-compact — économies de tokens majeures 😊", "Ultra-compact response mode — major token savings 😊")),
            ("Graphify", "https://github.com/safishamsi/graphify", tr("Graphes de connaissance — contexte persistant, moins de relecture", "Knowledge graphs — persistent context, less re-reading")),
            ("Obsidian", "https://obsidian.md", tr("Coffre de notes — mémoire inter-sessions, tokens économisés 😊", "Note vault — cross-session memory, tokens saved 😊"))],
            "#f39c12", 200)
        mc = QLabel(tr("Merci à toute la communauté SR open-source 💙", "Thanks to the whole open-source SR community 💙")); mc.setAlignment(Qt.AlignCenter)
        mc.setStyleSheet("color:#555577; font-style:italic; background:transparent; padding-top:6px;"); tv.addWidget(mc)
        v.addWidget(th)

        row = QHBoxLayout(); row.addStretch(1); row.addWidget(QLabel(tr("Moteurs SR :", "SR Engines:")))
        for label, url in (("NeoSR", "https://github.com/neosr-project/neosr"),
                           ("TraiNNer-Redux", "https://github.com/the-database/traiNNer-redux")):
            lk = QLabel(f'<a href="{url}" style="color:#3498db;">{label}</a>'); lk.setOpenExternalLinks(True)
            lk.setStyleSheet("background:transparent;"); row.addWidget(lk)
        row.addStretch(1); v.addLayout(row)
        lic = QLabel(tr("2025-2026 — Licence MIT", "2025-2026 — MIT License")); lic.setAlignment(Qt.AlignCenter); lic.setStyleSheet("color:#888; background:transparent;")
        v.addWidget(lic); v.addStretch(1)
        return self._scroll(inner)

    # ── python path ───────────────────────────────────────────────────────────────
    def _engine_python(self):
        try:
            from src.core import engine_paths as ep
            return ep.shared_venv_python() or ""
        except Exception:
            return self.settings.get("python_path", "")

    def _pick_py(self):
        path, _ = QFileDialog.getOpenFileName(self, "python.exe", "", "Python (python.exe);;Tous (*.*)")
        if path:
            self.py_edit.setText(path); self.settings.set("python_path", path)
