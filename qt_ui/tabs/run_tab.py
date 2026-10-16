"""
run_tab.py — Entraînement (PySide6 port of RunTab) — faithful to the v2.5.8 layout:

  ┌ GPU monitor ┐ ┌ Python/Script/Config + Load ┐ ┌ SYSTÈME monitor ┐
  │ 2×5 stat dashboard (durée/iter/vitesse/ETA/epoch · lossG/lossD/lr/PSNR/SSIM) │
  │ info bar : Architecture · Scale · Batch · Patch · Accumulate                 │
  │ console : progress + [IA][LR][Serveurs][Métriques][Validation][clear]        │
  │ log                                                                          │
  └ DÉMARRER · Éteindre PC · STOP & SAVE ┘

Training runs through QProcess (native, no UI freeze); GPU/CPU are polled via a
QTimer (pynvml + psutil when available). Log lines feed the stat cards live.
"""
from __future__ import annotations

import os
import re
import sys
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QLineEdit,
    QPlainTextEdit, QProgressBar, QFileDialog, QFrame, QCheckBox, QMessageBox,
    QSizePolicy,
)
from PySide6.QtCore import QProcess, Qt, QTimer, Signal
from PySide6.QtGui import QTextCursor, QFont, QColor, QTextCharFormat, QTextBlockFormat
from qt_ui.i18n import tr

# ANSI SGR code → log color (matches the CTk tag palette so the log looks identical).
_ANSI_RE = re.compile(r'\x1b\[([0-9;]*)m')
_ANSI_COLORS = {
    "36": "#3498db",       # cyan  — timestamps / VRAM / INFO
    "31": "#e74c3c",       # red   — G losses
    "35": "#9b59b6",       # purple — D losses
    "32": "#2ecc71",       # green — iter / checkmarks / saves
    "33": "#f1c40f",       # yellow — speed / interrupts
    "34": "#3498db",       # blue  — SSIM
    "95": "#ff69b4",       # pink  — grad norms
    "38;5;208": "#e67e22", # orange — epoch
    "38;5;130": "#c0874e", # brown  — learning rate
    "96": "#1abc9c",       # teal   — PSNR (matches BEST PSNR card)
}


def _bar_color_qss(color):
    return f"QProgressBar{{background:#2a2a2a;border:none;border-radius:3px;}}" \
           f"QProgressBar::chunk{{background:{color};border-radius:3px;}}"


# Two-phase training (delayed GAN): phase badge colours (PSNR green, GAN mauve)
_PHASE_PSNR = "#27ae60"
_PHASE_GAN = "#9b59b6"
# Phase degradations (PSNR -> GAN OTF profile): orange, like the degradation box of the log
_DEG_COLOR = "#e67e22"
# Phase losses (PSNR -> GAN loss weights): cyan
_LOSS_COLOR = "#17a2b8"


def _main_bar_qss(color):
    """Run progress bar: thicker, with the percentage written in the middle."""
    return ("QProgressBar{background:#2a2a2a;border:none;border-radius:4px;"
            "color:#fff;font-weight:700;font-size:11px;text-align:center;}"
            f"QProgressBar::chunk{{background:{color};border-radius:4px;}}")



def _load_any_config(path):
    """YAML (traiNNer-redux) or TOML (NeoSR) config -> dict."""
    if str(path).lower().endswith(".toml"):
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f) or {}
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

class RunTab(QWidget):
    # cross-thread signals: TrainingRunner calls callbacks from its worker thread
    _log_sig = Signal(str)
    _fin_sig = Signal(int)
    # public: emitted once a run is fully finalized (exit code, user_stopped) — the Queue
    # tab chains its next config on it.
    run_finished = Signal(int, bool)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.proc: QProcess | None = None   # legacy QProcess (unused when runner present)
        self.runner = None                  # src.core.runner.TrainingRunner instance
        self.entries = {}
        self.start_time = None
        self.target_iters = 100000
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.hist = {"iter": [], "loss": [], "loss_d": [], "psnr": [], "ssim": [], "lr": []}
        # USRS texture metrics history (usrs_tex_mae...) + best (val, iter) per key
        from src.core.val_extra_metrics import KEYS as _EXTRA_KEYS
        for _k in _EXTRA_KEYS:
            self.hist[_k] = []
        self._extra_best = {}
        self._last_iter = 0
        self._user_stopped = False
        self._hist_id = None
        self._last_speed = None       # last parsed it/s, fed to record_training_end
        self._vram_peak_mb = 0        # peak VRAM sampled during this run
        self._power_samples = []      # GPU power (W) samples during this run
        try:
            from src.core.log_format import ReduxLogFormatter
            self._fmt = ReduxLogFormatter()
        except Exception:
            self._fmt = None

        root = QVBoxLayout(self); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        # ── 1. top zone : GPU | files | CPU ──
        top = QHBoxLayout(); top.setSpacing(6); root.addLayout(top)
        top.addWidget(self._gpu_panel())
        top.addWidget(self._files_panel(), 1)
        top.addWidget(self._cpu_panel())

        # ── 2. dashboard ──
        root.addWidget(self._dashboard())
        # ── 3. info bar ──
        root.addWidget(self._info_bar())
        # ── 4. console ──
        root.addWidget(self._console(), 1)
        # ── 5. bottom buttons ──
        root.addLayout(self._bottom())

        # Monitoring panels (mon / files / statcard / infobar) are theme-driven via the
        # global QSS (objectNames) so they adapt to dark/light. Nothing hardcoded here.

        self._poll = QTimer(self); self._poll.setInterval(1500)
        self._poll.timeout.connect(self._poll_stats); self._poll.start()
        QTimer.singleShot(300, self._auto_load)
        QTimer.singleShot(300, self._poll_stats)
        self._tick = QTimer(self); self._tick.setInterval(1000); self._tick.timeout.connect(self._update_timer)
        self._log_sig.connect(self._on_runner_log)
        self._fin_sig.connect(self._on_finished)

    # ── panels ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_color(p):
        """Bar colour by load level (faithful to CTk get_load_color)."""
        if p < 30: return "#3498db"
        if p < 60: return "#2ecc71"
        if p < 85: return "#e67e22"
        return "#e74c3c"

    def _bar_row(self, layout, label, color):
        row = QHBoxLayout(); row.setSpacing(5)
        lbl = QLabel(label); lbl.setFixedWidth(58); lbl.setFont(QFont("Consolas", 9)); row.addWidget(lbl)
        bar = QProgressBar(); bar.setRange(0, 100); bar.setValue(0); bar.setTextVisible(False)
        bar.setFixedHeight(8); bar.setStyleSheet(_bar_color_qss(color)); row.addWidget(bar, 1)
        val = QLabel("--"); val.setFixedWidth(66); val.setAlignment(Qt.AlignRight)
        val.setFont(QFont("Consolas", 9)); row.addWidget(val)
        layout.addLayout(row)
        return bar, val

    def _set_bar(self, bar, val_lbl, pct, text):
        bar.setValue(int(max(0, min(100, pct))))
        bar.setStyleSheet(_bar_color_qss(self._load_color(pct)))
        val_lbl.setText(text)

    def _gpu_panel(self) -> QFrame:
        f = QFrame(); f.setObjectName("mon"); f.setFixedWidth(338); f.setMinimumHeight(248)
        v = QVBoxLayout(f); v.setContentsMargins(16, 12, 16, 12); v.setSpacing(8)
        t = QLabel("GPU"); t.setAlignment(Qt.AlignCenter); t.setStyleSheet("color:#888; font-weight:700; font-size:13px;")
        v.addWidget(t)
        self.lbl_gpu = QLabel("…"); self.lbl_gpu.setAlignment(Qt.AlignCenter)
        self.lbl_gpu.setStyleSheet("color:#2ecc71; font-weight:700;"); self.lbl_gpu.setWordWrap(True)
        v.addWidget(self.lbl_gpu); v.addSpacing(3)
        self.bar_vram, self.val_vram = self._bar_row(v, "VRAM", "#3498db")
        self.bar_util, self.val_util = self._bar_row(v, "Load", "#e67e22")
        self.bar_mc, self.val_mc = self._bar_row(v, "MemCtrl", "#9b59b6")
        self.bar_pcie, self.val_pcie = self._bar_row(v, "PCIe", "#1abc9c")
        self.bar_gpw, self.val_gpw = self._bar_row(v, "Power", "#e74c3c")
        v.addStretch(1)
        self.lbl_gtemp = QLabel("-- °C"); self.lbl_gtemp.setAlignment(Qt.AlignCenter)
        self.lbl_gtemp.setStyleSheet("font-size:16px; font-weight:700;"); v.addWidget(self.lbl_gtemp)
        return f

    def _cpu_panel(self) -> QFrame:
        f = QFrame(); f.setObjectName("mon"); f.setFixedWidth(338); f.setMinimumHeight(248)
        v = QVBoxLayout(f); v.setContentsMargins(16, 12, 16, 12); v.setSpacing(8)
        t = QLabel(tr("SYSTÈME", "SYSTEM")); t.setAlignment(Qt.AlignCenter); t.setStyleSheet("color:#888; font-weight:700; font-size:13px;")
        v.addWidget(t)
        name = self._cpu_name(); col = "#e67e22" if ("AMD" in name.upper() or "RYZEN" in name.upper()) else "#3498db"
        self.lbl_cpu = QLabel(name); self.lbl_cpu.setAlignment(Qt.AlignCenter)
        self.lbl_cpu.setStyleSheet(f"color:{col}; font-weight:700;"); self.lbl_cpu.setWordWrap(True)
        v.addWidget(self.lbl_cpu); v.addSpacing(3)
        self.bar_cpu, self.val_cpu = self._bar_row(v, "CPU", "#2ecc71")
        self.bar_ram, self.val_ram = self._bar_row(v, "RAM", "#3498db")
        self.bar_disk, self.val_disk = self._bar_row(v, "DISK", "#95a5a6")
        self.bar_freq, self.val_freq = self._bar_row(v, "Freq", "#f39c12")
        self.bar_cpw, self.val_cpw = self._bar_row(v, "Power", "#c0392b")
        v.addStretch(1)
        self.lbl_ctemp = QLabel("-- °C"); self.lbl_ctemp.setAlignment(Qt.AlignCenter)
        self.lbl_ctemp.setStyleSheet("font-size:16px; font-weight:700;"); v.addWidget(self.lbl_ctemp)
        return f

    def _files_panel(self) -> QFrame:
        f = QFrame(); f.setObjectName("files")
        g = QGridLayout(f); g.setContentsMargins(10, 8, 10, 8); g.setHorizontalSpacing(4)
        for i, (label, key) in enumerate((("Python", "python_path"), ("Script", "script_path"),
                                          ("Config", "config_path"))):
            lb = QLabel(label); lb.setFixedWidth(50); lb.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            g.addWidget(lb, i, 0)
            default = self.settings.get(key, "") or ""
            if key == "python_path":
                default = self._shared_venv_python_or(default)
            e = QLineEdit(default); e.setFont(QFont("Consolas", 10))
            self.entries[key] = e; g.addWidget(e, i, 1)
            if key == "python_path" and default and self._shared_venv_python_or("") == default:
                # F01: the shared venv always wins — show it as the Python actually used
                e.setReadOnly(True)
                e.setToolTip(tr("Python réellement utilisé : le venv partagé des moteurs (non modifiable tant qu'il existe).",
                                "Python actually used: the engines' shared venv (not editable while it exists)."))
            if key == "config_path":
                # auto-point the Script line at the right engine (.toml→NeoSR / .yml→Redux)
                e.editingFinished.connect(self._align_script)
            b = QPushButton("📂"); b.setFixedWidth(38); b.setObjectName("ghost")
            b.setStyleSheet('font-family:"Segoe UI Emoji"; font-size:13px; padding:0;')
            b.setFixedHeight(e.sizeHint().height())
            b.clicked.connect(lambda _=False, ed=e, k=key: self._browse(ed, k))
            if key == "python_path" and e.isReadOnly():
                b.setEnabled(False)
            g.addWidget(b, i, 2)
        load = QPushButton("Load"); load.setFixedWidth(56)
        load.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        load.setStyleSheet('QPushButton{background:#3a3f4b; color:#fff; border:1px solid #4a4f5b;'
                           ' border-radius:6px; font-weight:700; padding:0;}'
                           ' QPushButton:hover{background:#464d5c;}'
                           ' QPushButton:pressed{background:#2f343e;}')
        load.clicked.connect(lambda: self._browse(self.entries["config_path"], "config_path"))
        g.addWidget(load, 0, 3, 3, 1)
        g.setColumnStretch(1, 1)
        return f

    def _dashboard(self) -> QFrame:
        wrap = QFrame()
        g = QGridLayout(wrap); g.setContentsMargins(2, 0, 2, 0); g.setSpacing(5)
        # color=None → value inherits the theme text colour (readable on dark AND light cards);
        # the saturated colors below read fine on both.
        # NOTE: the first tuple element is the internal lookup key (self.cards[...] used
        # throughout this file to update values) — keep it stable. The second is the
        # translated display label shown on the card.
        cards = [
            ("DURÉE", tr("DURÉE", "DURATION"), "00:00:00", None),
            ("ITÉRATION", tr("ITÉRATION", "ITERATION"), "0 / --", "#2ecc71"),
            ("VITESSE", tr("VITESSE", "SPEED"), "-- it/s", "#f1c40f"),
            ("FIN ESTIMÉE", tr("FIN ESTIMÉE", "ETA"), "--:--:--", None),
            ("EPOCH", "EPOCH", "0", "#e67e22"),
            ("LOSS G", "LOSS G", "--", "#e74c3c"), ("LOSS D", "LOSS D", "--", "#9b59b6"),
            ("LR", "LR", "--", None),
            ("BEST PSNR", "BEST PSNR", "-- dB", "#1abc9c"), ("BEST SSIM", "BEST SSIM", "--", "#3498db"),
        ]
        # USRS texture metrics (3rd row): texture / edges / grain / HF detail / HF sharpness
        from src.core import val_extra_metrics as _vem
        cards += [(k, _vem.card_label(k), "--", _vem.color(k)) for k in _vem.KEYS]
        _card_tips = {
            "DURÉE": tr("Temps écoulé depuis le début de l'entraînement.", "Time elapsed since training started."),
            "ITÉRATION": tr("Itération courante / total prévu (total_iter de la config).", "Current iteration / total planned (config's total_iter)."),
            "VITESSE": tr("Vitesse d'entraînement (itérations par seconde).", "Training speed (iterations per second)."),
            "FIN ESTIMÉE": tr("Temps restant estimé avant la fin (eta renvoyé par le moteur).", "Estimated time remaining (eta reported by the engine)."),
            "EPOCH": tr("Époque courante (passages complets sur le dataset).", "Current epoch (full passes over the dataset)."),
            "LOSS G": tr("Perte totale du Générateur (doit globalement descendre/se stabiliser).", "Total Generator loss (should generally decrease/stabilize)."),
            "LOSS D": tr("Perte du Discriminateur (GAN). ~0.2-0.3 = équilibre sain.", "Discriminator loss (GAN). ~0.2-0.3 = healthy balance."),
            "LR": tr("Learning rate courant du générateur.", "Current generator learning rate."),
            "BEST PSNR": tr("Meilleur PSNR de validation atteint (+ haut = mieux).", "Best validation PSNR reached (higher = better)."),
            "BEST SSIM": tr("Meilleur SSIM de validation atteint (proche de 1 = mieux).", "Best validation SSIM reached (closer to 1 = better)."),
        }
        for _k in _vem.KEYS:
            _card_tips[_k] = _vem.tooltip(_k) + tr(
                "\n\nCarte : meilleure valeur atteinte (@ itération)." if _k != "usrs_hf_ratio"
                else "\n\nCarte : dernière valeur, colorée selon la zone (bleu doux / vert net / rouge sur-accentué).",
                "\n\nCard: best value reached (@ iteration)." if _k != "usrs_hf_ratio"
                else "\n\nCard: last value, colored by zone (blue soft / green sharp / red oversharpened).")
        self.cards = {}
        for i, (key, disp, val, color) in enumerate(cards):
            r, c = divmod(i, 5)
            card = QFrame(); card.setObjectName("statcard"); card.setMinimumHeight(58)
            card.setToolTip(_card_tips.get(key, ""))
            cv = QVBoxLayout(card); cv.setContentsMargins(6, 8, 6, 8); cv.setSpacing(3)
            tt = QLabel(disp); tt.setAlignment(Qt.AlignCenter); tt.setStyleSheet("color:gray; font-size:10px; font-weight:700;")
            lv = QLabel(val); lv.setAlignment(Qt.AlignCenter)
            lv.setStyleSheet((f"color:{color}; " if color else "") + "font-weight:700; font-size:15px;")
            lv.setWordWrap(True)
            cv.addStretch(1); cv.addWidget(tt); cv.addWidget(lv); cv.addStretch(1)
            self.cards[key] = lv
            g.addWidget(card, r, c)
        for c in range(5):
            g.setColumnStretch(c, 1)
        return wrap

    def _info_bar(self) -> QFrame:
        f = QFrame(); f.setObjectName("infobar")
        h = QHBoxLayout(f); h.setContentsMargins(8, 4, 8, 4)
        self.info = {}
        for label in ("Architecture", "Scale", "Batch Size", "Patch/LQ Size", "Accumulate"):
            cell = QHBoxLayout()
            l = QLabel(f"{label} : "); l.setStyleSheet("color:gray;")
            v = QLabel("--"); v.setStyleSheet("color:#3b8ed0; font-weight:700;")
            self.info[label] = v
            cell.addWidget(l); cell.addWidget(v); cell.addStretch(1)
            h.addLayout(cell, 1)
        return f

    def _console(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)
        head = QHBoxLayout()
        # Two-phase training (delayed GAN): current phase + GAN auto start counter.
        # Hidden for a single-phase run.
        self.lbl_phase = QLabel(); self.lbl_phase.hide()
        self.lbl_phase.setToolTip(tr(
            "Entraînement en 2 phases : PSNR (sans GAN) puis GAN.\n"
            "Le compteur indique les validations consécutives sans gain de PSNR\n"
            "(le GAN démarre quand il est plein, pas avant l'itération minimale).",
            "Two-phase training: PSNR (no GAN), then GAN.\n"
            "The counter shows the consecutive validations without a PSNR gain\n"
            "(the GAN starts when it is full, not before the minimum iteration)."))
        head.addWidget(self.lbl_phase)
        # Phase degradations (only when a separate GAN profile is active): profile 1 (PSNR),
        # ramp 1 -> 2 with its percentage, profile 2 (GAN).
        self.lbl_deg = QLabel(); self.lbl_deg.hide()
        self.lbl_deg.setToolTip(tr(
            "Dégradations par phase : profil 1 (PSNR) avant le GAN, puis rampe linéaire\n"
            "vers le profil 2 (GAN) après son démarrage.",
            "Phase degradations: profile 1 (PSNR) before the GAN, then a linear ramp\n"
            "to profile 2 (GAN) after it starts."))
        head.addWidget(self.lbl_deg)
        # Phase losses (only when a separate GAN loss profile is active)
        self.lbl_lphase = QLabel(); self.lbl_lphase.hide()
        self.lbl_lphase.setToolTip(tr(
            "Losses par phase : profil 1 (PSNR) avant le GAN, puis rampe linéaire des poids\n"
            "vers le profil 2 (GAN) après son démarrage.",
            "Phase losses: profile 1 (PSNR) before the GAN, then a linear weight ramp\n"
            "to profile 2 (GAN) after it starts."))
        head.addWidget(self.lbl_lphase)
        head.addStretch(1)
        _ai_lbl = tr("🤖 IA", "🤖 AI")
        _lr_lbl = tr("📉 LR", "📉 LR")
        _srv_lbl = tr("📡 Serveurs", "📡 Servers")
        _met_lbl = tr("📊 Métriques", "📊 Metrics")
        _val_lbl = tr("🔍 Validation", "🔍 Validation")
        _btn_tips = {
            _ai_lbl: tr("Analyser l'entraînement avec une IA (détecte sur/sous-apprentissage, conseils).", "Analyze the training with an AI (detects over/underfitting, advice)."),
            _lr_lbl: tr("Visualiser la courbe du learning rate selon le scheduler (+ marqueur d'itération).", "View the learning rate curve per the scheduler (+ iteration marker)."),
            _srv_lbl: tr("Galerie d'images + TensorBoard (liens, QR codes, démarrage).", "Image gallery + TensorBoard (links, QR codes, startup)."),
            _met_lbl: tr("Courbes Loss G / Loss D / PSNR en temps réel.", "Real-time Loss G / Loss D / PSNR curves."),
            _val_lbl: tr("Comparer LQ (entrée) vs SR (sortie) des images de validation.", "Compare LQ (input) vs SR (output) of validation images."),
        }
        for txt, color, slot in ((_ai_lbl, "#e67e22", self._ai), (_lr_lbl, "#9b59b6", self._lr),
                                 (_srv_lbl, "#16a085", self._servers),
                                 (_met_lbl, "#2980b9", self._metrics),
                                 (_val_lbl, "#8e44ad", self._validation)):
            b = QPushButton(txt); b.setFixedHeight(26)
            b.setStyleSheet(f'background:{color}; color:#fff; border:none; border-radius:6px; '
                            f'padding:2px 10px; font-family:"Segoe UI Emoji"; font-weight:600; font-size:13px;')
            b.setToolTip(_btn_tips.get(txt, "")); b.clicked.connect(slot); head.addWidget(b)
        clr = QPushButton("🧹"); clr.setObjectName("ghost"); clr.setFixedWidth(34); clr.setFixedHeight(26)
        clr.setStyleSheet('font-family:"Segoe UI Emoji"; font-size:13px; padding:0;')
        clr.setToolTip(tr("Nettoyer le log", "Clear log"))
        clr.clicked.connect(self._clear_log); head.addWidget(clr)
        v.addLayout(head)

        self.chk_resume = QCheckBox(tr("Auto-resume (reprendre au dernier checkpoint)", "Auto-resume (continue from the last checkpoint)")); self.chk_resume.setChecked(True)
        self.chk_resume.setToolTip(tr("Au DÉMARRAGE : si l'expérience a déjà un checkpoint (.state), l'entraînement repart\n"
                                   "de là au lieu de recommencer à zéro (après un crash, relancez simplement).\n"
                                   "Pas de relance automatique pendant votre absence.",
                                   "At START: if the experiment already has a checkpoint (.state), training continues\n"
                                   "from it instead of starting over (after a crash, just start again).\n"
                                   "No automatic relaunch while you are away."))
        v.addWidget(self.chk_resume)

        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        self.progress.setTextVisible(True); self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setFixedHeight(18); self.progress.setFormat("0%")
        self.progress.setStyleSheet(_main_bar_qss("#2ecc71")); v.addWidget(self.progress)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(8000)
        # Wrap to the widget width (no horizontal scroll). A small monospace font keeps a
        # full iter record on a single line in practice (like the CTk app), and only a
        # truly over-long line wraps. The global theme QSS (`* { font-family:"Segoe UI" }`)
        # OVERRIDES setFont(), so a widget-level QSS forces the monospace + size.
        self.log.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.log.setObjectName("trainlog")
        # Pin the log to its original dark background in BOTH themes (its ANSI colors —
        # green/orange/white box-drawing — are tuned for a dark console and would be
        # unreadable on a light background). Widget-level QSS so it never follows the theme.
        self.log.setStyleSheet('QPlainTextEdit#trainlog{font-family:"Consolas","Cascadia Mono","Courier New";'
                               ' font-size:11px; background:#343638; color:#dce4ee;'
                               ' border:1px solid #4a4a4a; border-radius:6px;}')
        _logfont = QFont("Consolas", 9)
        _logfont.setStyleHint(QFont.Monospace); _logfont.setFixedPitch(True)
        self.log.setFont(_logfont); v.addWidget(self.log, 1)
        return w

    def _bottom(self) -> QHBoxLayout:
        h = QHBoxLayout()
        self.btn_start = QPushButton(tr("▶  DÉMARRER", "▶  START")); self.btn_start.setFixedHeight(40)
        self.btn_start.setToolTip(tr("Lancer l'entraînement (prépare le moteur : patches, injection custom,\n"
                                  "dégradations OTF, copie de travail de la config).", "Start training (prepares the engine: patches, custom injection,\nOTF degradations, working copy of the config)."))
        self.btn_start.setStyleSheet(
            'QPushButton{background:#27ae60; color:#fff; border:none; border-radius:8px;'
            ' font-size:14px; font-weight:700;}'
            'QPushButton:hover{background:#2ecc71;}'
            'QPushButton:pressed{background:#1e8449; padding-top:2px;}'  # visual "press" feedback
            'QPushButton:disabled{background:#3a4a40; color:#8aa;}')
        self.btn_start.clicked.connect(self._start); h.addWidget(self.btn_start, 1)
        self.chk_shutdown = QCheckBox(tr("Éteindre PC à la fin", "Shutdown PC when done")); h.addWidget(self.chk_shutdown)
        self.chk_shutdown.setToolTip(tr("Éteint l'ordinateur 60 s après un entraînement terminé normalement (une fenêtre\n"
                                     "permet d'annuler). Pas d'arrêt après un STOP manuel ou un crash.",
                                     "Shuts the computer down 60 s after a training that finished normally (a window\n"
                                     "lets you cancel). No shutdown after a manual STOP or a crash."))
        self.btn_stop = QPushButton("⏹  STOP & SAVE"); self.btn_stop.setFixedHeight(40); self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip(tr("Clic 1 = arrêt propre (sauvegarde modèle + état avant de stopper).\n"
                                 "Clic 2 = arrêt forcé immédiat (KILL, sans sauvegarde).", "Click 1 = clean stop (saves model + state before stopping).\nClick 2 = immediate force stop (KILL, no save)."))
        self.btn_stop.setStyleSheet(
            'QPushButton{background:#d35b58; color:#fff; border:none; border-radius:8px;'
            ' font-size:14px; font-weight:700;}'
            'QPushButton:hover{background:#e06b68;}'
            'QPushButton:pressed{background:#a83a37; padding-top:2px;}'  # visual "press" feedback
            'QPushButton:disabled{background:#4a3a3a; color:#a88;}')
        self.btn_stop.clicked.connect(self._stop); h.addWidget(self.btn_stop, 1)
        return h

    # ── files / config ──────────────────────────────────────────────────────────
    def _browse(self, entry, key):
        if "config" in key:
            flt = "Config (*.toml *.yml *.yaml);;Tous (*.*)"
        elif "python" in key:
            flt = "Exe (*.exe);;Tous (*.*)"
        else:
            flt = "Python (*.py);;Tous (*.*)"
        start = ""
        if "config" in key:
            from src.core import engine_paths as _ep
            cand = os.path.join(_ep.base_engine_path(), "Option Custom")
            if os.path.isdir(cand):
                start = cand
        p, _ = QFileDialog.getOpenFileName(self, tr("Fichier", "File"), start, flt)
        if p:
            entry.setText(p); self.settings.set(key, p)
            if key == "config_path":
                self._read_config(p)
                self._align_script()

    def _shared_venv_python_or(self, fallback: str) -> str:
        """Prefer the ONE shared venv python (single source of truth, see engine_paths.py)
        over whatever stale/manually-picked value might be sitting in settings — a stale
        entry (e.g. pointing at the bootstrap portable python instead of the venv) still
        "exists on disk" so a plain os.path.isfile check never catches it, and training
        then fails with a cryptic `ModuleNotFoundError: No module named 'torch'`."""
        try:
            from src.core import engine_paths as ep
            venv_py = ep.shared_venv_python()
            if venv_py and os.path.isfile(venv_py):
                return venv_py
        except Exception:
            pass
        return fallback

    def _auto_load(self):
        for key in ("python_path", "script_path", "config_path"):
            val = self.settings.get(key, "")
            if key == "python_path":
                val = self._shared_venv_python_or(val)
            if val and key in self.entries:
                self.entries[key].setText(val)
        cfg = self.entries["config_path"].text().strip()
        if cfg and os.path.isfile(cfg):
            self._read_config(cfg)

    def _read_config(self, path):
        try:
            arch = scale = bs = ps = acc = "?"
            if path.endswith(".toml"):
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib
                with open(path, "rb") as fh:
                    d = tomllib.load(fh)
                tr = d.get("datasets", {}).get("train", {})
                bs = tr.get("batch_size", "?"); ps = tr.get("patch_size", "?"); acc = tr.get("accumulate", 1)
            else:
                import yaml
                with open(path, "r", encoding="utf-8") as fh:
                    d = yaml.safe_load(fh)
                tr = d.get("datasets", {}).get("train", {})
                bs = tr.get("batch_size_per_gpu", "?"); ps = tr.get("gt_size", tr.get("lq_size", "?"))
                acc = tr.get("accum_iter", "?")
            arch = d.get("network_g", {}).get("type", "?"); scale = d.get("scale", "?")
            self.info["Architecture"].setText(str(arch).upper())
            self.info["Scale"].setText(f"x{scale}")
            self.info["Batch Size"].setText(str(bs))
            self.info["Patch/LQ Size"].setText(str(ps))
            self.info["Accumulate"].setText(str(acc))
            # NeoSR keeps total_iter under [logger]; traiNNer-redux under train:. Without the
            # logger lookup a NeoSR run kept showing the previous config's total (e.g. 0/400
            # for a 300-iteration run), which also skewed the progress bar and ETA.
            ti = (d.get("total_iter") or d.get("train", {}).get("total_iter")
                  or d.get("logger", {}).get("total_iter"))
            if ti:
                self.target_iters = int(float(ti))
                self.cards["ITÉRATION"].setText(f"0 / {self.target_iters}")
            self._phase_from_config(d)
        except Exception:
            pass

    # ── training (via TrainingRunner — same patches/prep as the CTk app) ──────────
    def _resolve_paths(self):
        py = self.entries["python_path"].text().strip()
        py = self._shared_venv_python_or(py)
        if not py or not os.path.isfile(py):
            py = self.settings.get("python_path", "python") or "python"
        self._align_script()
        cfg = self.entries["config_path"].text().strip()
        script = self.entries["script_path"].text().strip()
        return py, script, cfg

    def _align_script(self):
        """Point the Script line at the train.py matching the config's engine:
        .toml → NeoSR, .yml/.yaml → Redux. Runs on config change and before start, so a
        NeoSR .toml never gets launched with redux's train.py (msgspec.yaml would crash)."""
        cfg = self.entries["config_path"].text().strip()
        if not cfg:
            return
        ext = os.path.splitext(cfg)[1].lower()
        want_redux = ext in (".yml", ".yaml")
        script = self.entries["script_path"].text().strip()
        cur_redux = ("redux" in script.lower() or "trainner" in script.lower())
        cur_neosr = ("neosr" in script.lower() and not cur_redux)
        mismatch = bool(script) and ((want_redux and cur_neosr) or (not want_redux and cur_redux))
        # A saved script from the *other* app engine root (~/USRS vs ~/USRS-EXPERIMENTAL,
        # e.g. a user_settings.json carried over from another build) would run the wrong
        # engine: older archs missing ("No object named 'nerve'") and experiments written
        # in the wrong folder. Custom paths outside both roots are left alone.
        try:
            from src.core import engine_paths as _ep
            _home = os.path.normcase(os.path.abspath(_ep.user_home()))
            _sp = os.path.normcase(os.path.abspath(script)) if script else ""
            _base = os.path.normcase(os.path.abspath(_ep.base_engine_path()))
            _roots = [os.path.normcase(os.path.join(_home, n)) for n in ("USRS", "USRS-EXPERIMENTAL")]
            if _sp and any(_sp.startswith(r + os.sep) for r in _roots) and not _sp.startswith(_base + os.sep):
                mismatch = True
        except Exception:
            pass
        if script and os.path.isfile(script) and not mismatch:
            return
        try:
            from src.core import engine_paths as ep
            order = [ep.redux_path()] if want_redux else [ep.neosr_path()]
            for root in order:
                ts = os.path.join(root, "train.py")
                if os.path.isfile(ts):
                    if mismatch and script:
                        self._append("[" + tr("Moteur", "Engine") + "] " + tr(
                            f"Config {ext} → script ré-aligné sur {ts}",
                            f"Config {ext} → script re-aligned to {ts}") + "\n")
                    self.entries["script_path"].setText(ts)
                    self.settings.set("script_path", ts)
                    break
            else:
                if mismatch:
                    self.entries["script_path"].setText("")
                self._append("[" + tr("Moteur", "Engine") + "] " + tr(
                    f"Config {ext} : le moteur {'traiNNer-redux' if want_redux else 'NeoSR'} n'est pas installé "
                    "(Réglages › Moteurs).",
                    f"{ext} config: the {'traiNNer-redux' if want_redux else 'NeoSR'} engine is not installed "
                    "(Settings › Engines).") + "\n")
        except Exception:
            pass

    def is_busy(self) -> bool:
        return self.runner is not None and bool(getattr(self.runner, "is_running", False))

    def external_start(self, config_path: str, suppress_shutdown: bool = False) -> bool:
        """Start a run from another tab (Queue): load the config into the Training tab
        and press START, so the run gets the full dashboard / formatted log / badges.
        Returns True if the run was launched."""
        if self.is_busy() or not config_path or not os.path.isfile(config_path):
            return False
        self.suppress_shutdown = bool(suppress_shutdown)
        self.entries["config_path"].setText(config_path)
        self.settings.set("config_path", config_path)
        self._read_config(config_path)
        self._align_script()
        self._start()
        return self.is_busy()

    def _start(self):
        if self.runner is not None and getattr(self.runner, "is_running", False):
            return
        py, script, cfg = self._resolve_paths()
        if not (cfg and os.path.isfile(cfg)) or not (script and os.path.isfile(script)) or not (py and os.path.isfile(py)):
            QMessageBox.warning(self, tr("Entraînement", "Training"), tr("Python / script / config introuvable.", "Python / script / config not found."))
            return
        try:
            from src.core.runner import TrainingRunner
        except Exception as e:
            QMessageBox.critical(self, tr("Entraînement", "Training"), tr(f"Backend runner indisponible : {e}", f"Backend runner unavailable: {e}"))
            return
        if self.runner is None:
            self.runner = TrainingRunner()
        self.start_time = time.time(); self._tick.start()
        self._last_iter = 0
        self._user_stopped = False  # set by _stop() → drives the yellow "interrupted" banner
        self._soft_stop_done = False
        self.btn_stop.setText("⏹  STOP & SAVE")
        # reset metrics graph history + best scores for the new run
        for _k in self.hist:
            self.hist[_k] = []
        self._last_speed = None; self._vram_peak_mb = 0; self._power_samples = []
        self._reset_dashboard()
        if self._fmt is not None:
            self._fmt.reset()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self._append(">> " + tr("Préparation moteur (patches + injection custom)…", "Preparing engine (patches + custom injection)…") + "\n")
        # Refresh the dashboard cards (Architecture/Scale/Batch/Patch/Accumulate) from
        # the config file about to be launched — previously only done by the "Load"
        # button, so editing the YAML then hitting Start showed stale values even
        # though train.py itself always read the current file correctly.
        self._read_config(cfg)
        # Generation tag: the previous run's stdout-reading thread can still have a
        # few buffered lines in flight (already read from the pipe, not yet emitted)
        # at the exact moment Stop → Start happen in quick succession. Both runs share
        # the same self._log_sig/_on_runner_log, so without this a stray leftover line
        # from the OLD run (e.g. "iter: 400") lands on the freshly-reset dashboard and
        # briefly shows stale numbers until the NEW run's first real line overwrites it.
        self._run_generation = getattr(self, "_run_generation", 0) + 1
        _gen = self._run_generation
        # callbacks run in the runner's worker thread → marshal to UI via signals
        def log_cb(s):
            if self._run_generation == _gen:
                self._log_sig.emit(str(s))
        fin_cb = lambda *a: self._fin_sig.emit(int(a[0]) if a and isinstance(a[0], int) else 0)
        try:
            self._show_training_estimation(cfg)
            self._log_otf_summary(cfg)
            self._set_val_dirs(cfg)
            self.runner.monitor_prefs = self._monitor_prefs()
            self._gallery_auto(start=True)
            self.runner.start_training(py, script, cfg, log_cb, fin_cb,
                                       auto_resume=self.chk_resume.isChecked())
            self._record_training_start(cfg, script)
        except Exception as e:
            self._append(f"\n❌ {e}\n"); self._on_finished(1)

    def _pref(self, key, default):
        return str(self.settings.get(key, default)).lower() == "true"

    def _monitor_prefs(self):
        """Settings › Gallery automations for the runner (they were read by nothing)."""
        return {"auto_tb": self._pref("gal_auto_tb", "false"),
                "open_browser": self._pref("gal_auto_open", "false"),
                "ngrok_allowed": self._pref("gal_auto_ngrok", "true")}

    def _gallery_auto(self, start):
        """Auto-start the gallery at launch / auto-stop it at the end (only a gallery this
        tab started itself is stopped)."""
        try:
            from src.core.gallery_server import get_server
            srv = get_server()
            if start and self._pref("gal_auto_start", "false"):
                d = self.settings.get("gallery_dir", "") or ""
                port = int(self.settings.get("gallery_port", "8765") or 8765)
                res = srv.start(d, port=port, with_ngrok=False) or {}
                self._gallery_started = bool(res.get("ok"))
                self._append("[" + tr("Galerie", "Gallery") + "] " + (
                    res.get("local_url", "") if res.get("ok") else str(res.get("error", "?"))) + "\n")
            elif not start and getattr(self, "_gallery_started", False) and self._pref("gal_auto_stop", "true"):
                srv.stop()
                self._gallery_started = False
        except Exception as e:
            self._append(f"[{tr('Galerie', 'Gallery')}] {e}\n")

    def _set_val_dirs(self, config_path):
        """Give the log formatter the validation dataset folder so the VALIDATION box
        lists ALL images (not only the few names captured from tqdm)."""
        if self._fmt is None:
            return
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return

        def _dir(ds):
            for k in ("dataroot_lq", "lq_path", "dataroot_gt", "gt_path"):
                v = ds.get(k, "")
                if isinstance(v, list):
                    v = v[0] if v else ""
                v = str(v).strip()
                if v:
                    return v
            return ""

        lq = gt = ""
        datasets = cfg.get("datasets", {})
        entries = datasets.values() if isinstance(datasets, dict) else (datasets if isinstance(datasets, list) else [])
        for ds in entries:
            if isinstance(ds, dict) and str(ds.get("name", "")).lower().startswith("val"):
                d = _dir(ds)
                if d:
                    lq = d
                    break
        try:
            self._fmt.set_val_dirs(lq, gt)
        except Exception:
            pass

    def _show_training_estimation(self, config_path):
        """Render the training duration estimation box at launch (parity with CTk)."""
        try:
            import yaml
            from src.core.compute_estimator import estimate_training_time, format_estimation_message, run_params_from_config
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            ng = cfg.get("network_g", {})
            arch = ng.get("type", "omnisr_net") if isinstance(ng, dict) else (ng if isinstance(ng, str) else "omnisr_net")
            scale = int(cfg.get("scale", 4) or 4)
            tds = cfg.get("datasets", {}).get("train", {}) if isinstance(cfg.get("datasets"), dict) else {}
            batch = int(tds.get("batch_size_per_gpu") or tds.get("batch_size") or cfg.get("batch_size", 4) or 4)
            # Same estimate as the Config tab (losses, discriminators, checkpointing). The
            # patch is in LQ pixels: lq_size as is, gt_size divided by the scale (the old
            # code also divided lq_size, so lq 128 in 2x was estimated as patch 64).
            prm = run_params_from_config(cfg)
            accum, total_iter = prm["accumulate"], prm["total_iter"]
            est = estimate_training_time(**prm)
            # epochs from dataset image count
            dataset_line = ""
            try:
                gt = tds.get("dataroot_gt", [])
                gt = [gt] if isinstance(gt, str) else (gt or [])
                ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                n = sum(1 for d in gt if d and os.path.isdir(d)
                        for fn in os.scandir(d) if fn.is_file() and os.path.splitext(fn.name)[1].lower() in ext)
                if n > 0:
                    epochs = max(1, round(total_iter * batch * max(1, accum) / n))
                    dataset_line = f"Dataset train : {n} images → ~{epochs} epochs\n"
            except Exception:
                pass
            msg = format_estimation_message(est) if est else ""
            # strip wide emojis that would break monospace box alignment
            for _e in ("✅", "❌", "⚠️", "⚠", "🔍", "🚀"):
                msg = (msg or "").replace(_e, "")
            box = self._box(tr("[Estimation] Durée prévue de l'entraînement :", "[Estimate] Expected training duration:"),
                            [l for l in ((dataset_line + (msg or "")).splitlines()) if l.strip()],
                            border="\x1b[32m")  # green frame
            self._append_ansi(box + "\n")
        except Exception:
            pass

    def _log_otf_summary(self, config_path):
        """Render the active/inactive OTF degradations box at launch (parity with CTk)."""
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return
        ds = cfg.get("datasets", {}).get("train", {}) if isinstance(cfg.get("datasets"), dict) else {}
        deg = cfg.get("degradations", {}) if isinstance(cfg.get("degradations"), dict) else {}

        # key aliases (redux uses blur_prob2; some configs use second_blur_prob)
        _aliases = {"second_blur_prob": ["second_blur_prob", "blur_prob2"]}

        def gv(k):
            for kk in _aliases.get(k, [k]):
                for src in (cfg, deg, ds):
                    if isinstance(src, dict) and kk in src:
                        return src[kk]
            return 0.0

        # 8 base Real-ESRGAN OTF slots
        base_slots = [
            ("blur_prob", tr("Flou 1", "Blur 1")), ("gaussian_noise_prob", tr("Bruit Gaussien 1", "Gaussian Noise 1")),
            ("gray_noise_prob", tr("Bruit Gris 1", "Gray Noise 1")), ("jpeg_prob", "JPEG 1"),
            ("second_blur_prob", tr("Flou 2", "Blur 2")), ("gaussian_noise_prob2", tr("Bruit Gaussien 2", "Gaussian Noise 2")),
            ("gray_noise_prob2", tr("Bruit Gris 2", "Gray Noise 2")), ("final_sinc_prob", tr("Sinc final", "Final Sinc")),
        ]
        # every custom degradation the engine can apply (label for each *_prob key)
        custom_slots = [
            ("posterize_prob", "Posterize"), ("banding_prob", "Banding"),
            ("smooth_banding_prob", tr("Banding zones lisses", "Smooth Banding")),
            ("chroma_prob", "Chroma Sub"), ("ca_prob", tr("Aberration Chromatique", "Chromatic Aberration")),
            ("halation_prob", "Halation"), ("salt_pepper_prob", "Salt & Pepper"),
            ("vhs_prob", "VHS / Analog"), ("disc_blur_prob", tr("Flou Bokeh", "Bokeh Blur")),
            ("vignette_prob", tr("Vignettage", "Vignetting")), ("quantize_depth_prob", tr("Profondeur Bit", "Bit Depth")),
            ("coupled_optical_prob", tr("Cluster Optique", "Optics Cluster")), ("coupled_vintage_prob", tr("Cluster Vintage", "Vintage Cluster")),
            ("aliasing_prob", "Aliasing"), ("interlace_weave_prob", "Interlace Weave"),
            ("interlace_flicker_prob", "Interlace Flicker"), ("interlace_blend_prob", "Interlace Blend"),
            ("film_grain_prob", "Film Grain"), ("oversharp_prob", "Oversharp"),
            ("scanlines_prob", "Scanlines"), ("screentone_prob", "Screentone"),
            ("dithering_prob", "Dithering"), ("pixelate_prob", "Pixelate"),
            ("sin_prob", tr("Sinusoïdal", "Sinusoidal")), ("subsampling_prob", tr("Sous-échantillonnage Chroma", "Chroma Subsampling")),
            ("color_level_prob", tr("Niveaux de Couleur", "Color Levels")), ("wtp_halo_prob", "Halo / Ringing"),
            ("saturation_prob", "Saturation"), ("shift_prob", tr("Décalage Pixel", "Pixel Shift")),
        ]

        def _prob(k):
            try:
                return float(gv(k) or 0)
            except Exception:
                return 0.0

        base_active = [(lbl, _prob(k)) for k, lbl in base_slots if _prob(k) > 0]
        base_inactive = [lbl for k, lbl in base_slots if _prob(k) <= 0]
        custom_active = [(lbl, _prob(k)) for k, lbl in custom_slots if _prob(k) > 0]
        n_custom_inactive = sum(1 for k, _ in custom_slots if _prob(k) <= 0)
        total_active = len(base_active) + len(custom_active)

        body = [f"✓ {lbl} (p={p:.2f})" for lbl, p in base_active]
        if custom_active:
            body.append(tr("— Custom —", "— Custom —"))
            body += [f"✓ {lbl} (custom) (p={p:.2f})" for lbl, p in custom_active]
        if base_inactive:
            body.append(tr("(base inactives)", "(base inactive)"))
            body += [f"- {lbl}" for lbl in base_inactive]
        if n_custom_inactive:
            body.append(tr(f"(+ {n_custom_inactive} dégradations custom inactives)", f"(+ {n_custom_inactive} inactive custom degradations)"))
        # orange frame (matches the OTF degradations colour), green checkmarks
        box = self._box(tr(f"Dégradations OTF actives : {total_active}", f"Active OTF Degradations: {total_active}"), body, border="\x1b[38;5;208m")
        box = box.replace("✓", "\x1b[32m✓\x1b[0m")
        self._append_ansi(box + "\n")

    @staticmethod
    def _box(title, lines, border=None):
        """Render a monospace box (╔═╗ … ╚═╝). `border` = ANSI code to colorize the
        frame lines only (zero-width → alignment preserved); the inside stays default."""
        title = "  " + title.strip() + "  "
        body = ["  " + str(l).rstrip() for l in lines]
        width = max([len(title)] + [len(b) for b in body] + [40])
        bar = "═" * width
        out = [f"╔{bar}╗", f"║{title.ljust(width)}║", f"╠{bar}╣"]
        out += [f"║{b.ljust(width)}║" for b in body]
        out.append(f"╚{bar}╝")
        text = "\n".join(out)
        if border:
            X = "\x1b[0m"
            for ch in "╔╗╚╝╠╣═║":
                text = text.replace(ch, f"{border}{ch}{X}")
        return text

    def _persist_best_metrics(self):
        """Push the current best PSNR/SSIM into the Historique DB as they improve.
        update_training_progress() existed in training_history.py but was never
        actually called from anywhere — best_psnr/best_ssim stayed at their DEFAULT 0
        for every real training run, which is why History's "Stats par Architecture"
        was always empty (query filters WHERE best_psnr > 0) even with 60+ completed
        trainings in the DB (2026-09-17 fix). Best-effort: never let a DB hiccup
        interrupt live training."""
        try:
            if not getattr(self, "_hist_id", None):
                return
            from src.core.training_history import update_training_progress
            update_training_progress(
                self._hist_id, current_iter=self._last_iter,
                best_psnr=self.best_psnr or None, best_ssim=self.best_ssim or None)
        except Exception:
            pass

    def _record_training_start(self, config_path, script_path):
        """Log this training run in the Historique DB (faithful to the CTk app)."""
        self._hist_id = None
        try:
            from src.core.training_history import record_training_start
            cfg = _load_any_config(config_path)
            ng = cfg.get("network_g", {})
            arch = ng.get("type", "unknown") if isinstance(ng, dict) else (ng if isinstance(ng, str) else "unknown")
            train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
            tds = cfg.get("datasets", {}).get("train", {}) if isinstance(cfg.get("datasets"), dict) else {}
            batch = tds.get("batch_size_per_gpu") or tds.get("batch_size") or cfg.get("batch_size", 0)
            patch = tds.get("lq_size") or tds.get("gt_size") or cfg.get("gt_size", 0)
            ds_path = tds.get("dataroot_gt", "")
            if isinstance(ds_path, list):
                ds_path = ds_path[0] if ds_path else ""
            optim_g = train.get("optim_g", {}) if isinstance(train.get("optim_g"), dict) else {}
            losses_cfg = train.get("losses", {})
            gan_mode = 1 if (cfg.get("network_d") or cfg.get("net_d") or
                             (isinstance(losses_cfg, dict) and any("gan" in k.lower() for k in losses_cfg))) else 0
            if isinstance(losses_cfg, dict):
                losses_str = ",".join(k for k, v in losses_cfg.items() if v)
            elif isinstance(losses_cfg, list):
                losses_str = ",".join(str(x) for x in losses_cfg)
            else:
                losses_str = ""
            gpu_name = ""
            try:
                from src.core.compute_estimator import detect_gpu
                gpu_name, _ = detect_gpu()
            except Exception:
                pass
            self._hist_id = record_training_start(
                name=cfg.get("name", os.path.splitext(os.path.basename(config_path))[0]),
                engine="NeoSR" if "neosr" in script_path.lower() else "Redux",
                architecture=str(arch), scale=int(cfg.get("scale", 4) or 4),
                total_iter=int(float(train.get("total_iter") or cfg.get("total_iter")
                                     or (cfg.get("logger") or {}).get("total_iter") or 0)),  # NeoSR: [logger]
                config_path=config_path, dataset_path=str(ds_path),
                batch_size=int(batch or 0), patch_size=int(patch or 0),
                lr=str(optim_g.get("lr", "")), optimizer=str(optim_g.get("type", "")),
                gpu_name=gpu_name, gan_mode=gan_mode, losses=losses_str)
        except Exception as e:
            self._append(tr(f"[Historique] Avertissement : {e}\n", f"[History] Warning: {e}\n"))

    def _reset_dashboard(self):
        """Reset every dashboard card + progress bar to their idle state. Without
        this, a card kept showing the PREVIOUS run's number until a fresh log line
        overwrote it — for a run that gets stuck before its first "iter:" line (e.g.
        an OOM/VRAM-thrash hang right at startup), that made a training that had
        never produced a single iteration look like real (frozen) progress from the
        prior run. Called on every _start(), and from the 🧹 Clear log button so
        clearing the log doesn't leave stale cards behind either."""
        if self._fmt is not None:
            self._fmt.reset()
        self.best_psnr = 0.0; self.best_ssim = 0.0
        self.cards["BEST PSNR"].setText("-- dB"); self.cards["BEST SSIM"].setText("--")
        self._extra_best = {}
        try:
            from src.core import val_extra_metrics as _vem
            for _k in _vem.KEYS:
                self.cards[_k].setText("--")
                self.cards[_k].setStyleSheet(f"color:{_vem.color(_k)}; font-weight:700; font-size:15px;")
        except Exception:
            pass
        self.cards["ITÉRATION"].setText(f"0 / {self.target_iters}")
        self.cards["EPOCH"].setText("0")
        self.cards["LOSS G"].setText("--"); self.cards["LOSS D"].setText("--")
        self.cards["LR"].setText("--"); self.cards["FIN ESTIMÉE"].setText("--:--:--")
        self.cards["VITESSE"].setText("-- it/s")
        self._set_progress_pct(0)

    def _clear_log(self):
        self.log.clear()
        # Only reset the dashboard if nothing is currently running — clearing the
        # log mid-training shouldn't wipe the live stats out from under the user.
        if not (self.runner is not None and getattr(self.runner, "is_running", False)):
            self._reset_dashboard()

    # ── two-phase training (delayed GAN): phase badge + GAN auto start counter ────
    _GAN_AUTO_RE = re.compile(r"GAN auto start: when .*? over (\d+) validation.*?not before iteration (\d+)")
    _GAN_DELAYED_RE = re.compile(r"GAN delayed: .*?start at iteration (\d+)")
    _GAN_COUNT_RE = re.compile(r"GAN auto start: PSNR .*?(\d+)/(\d+) validation")
    _GAN_STARTED_RE = re.compile(r"GAN starts at iteration (\d+)")
    _DEG_ON_RE = re.compile(r"(?:Phase degradations|Dégradations par phase) PSNR \S+ GAN activ")
    _DEG_STATE_RE = re.compile(r"OTF degradations: (PSNR profile|transition|GAN profile) \(GAN profile (\d+)%\)")
    _LPHASE_ON_RE = re.compile(r"(?:Phase losses|Losses par phase) PSNR \S+ GAN activ")
    _LPHASE_STATE_RE = re.compile(r"Losses: (PSNR profile|transition|GAN profile) \(GAN profile (\d+)%\)")
    _RESUME_RE = re.compile(r"\[Auto-Resume\] Resuming from|Resuming training from epoch")
    # After a resume past the GAN start nothing re-announces it: the phase profile
    # line or the discriminator losses of an iteration record give it away.
    _GAN_RUNNING_RE = re.compile(r"OTF degradations: (?:GAN profile|transition)|l_g_gan|l_d_real")

    def _phase_from_config(self, d):
        """Two-phase mode from the train options (traiNNer-redux delayed GAN patch)."""
        t = (d or {}).get("train") or {}
        self._phase = None; self._gan_from = None; self._gan_count = None
        self._phase_tail = ""; self._phase_resumed = False
        self._deg = None   # None = no phase degradations; else GAN-profile percentage
        self.lbl_deg.hide()
        self._lphase = None   # None = no phase losses; else GAN-profile percentage
        self.lbl_lphase.hide()
        self._gan_min = self._gan_fixed = None
        if (d or {}).get("network_d") and (t.get("gan_auto_start") or int(t.get("gan_start_iter") or 0) > 0):
            self._phase = "psnr"
            if t.get("gan_auto_start"):
                self._gan_count = (0, int(t.get("gan_auto_patience") or 3))
                self._gan_min = int(t.get("gan_auto_min_iter") or 0)
            else:
                self._gan_fixed = int(t.get("gan_start_iter") or 0)
        self._show_phase()

    def _parse_phase(self, text):
        # The engine console colours numbers (rich ANSI codes) and wraps long records over
        # several lines/chunks: match on de-coloured, whitespace-collapsed text with a
        # rolling tail, and only act on matches that end in the new text.
        # rich also right-aligns the source "file.py:123" on the first line of a wrapped
        # record, i.e. in the middle of the message: drop those markers.
        norm = re.sub(r"\s+", " ", re.sub(r"[\w.-]+\.py:\d+", " ", _ANSI_RE.sub("", text))).strip()
        if not norm:
            return
        prev = getattr(self, "_phase_tail", "")
        buf = f"{prev} {norm}" if prev else norm
        new_from = len(prev) + 1 if prev else 0
        self._phase_tail = buf[-400:]
        events = []
        for kind, rx in (("auto", self._GAN_AUTO_RE), ("delayed", self._GAN_DELAYED_RE),
                         ("count", self._GAN_COUNT_RE), ("started", self._GAN_STARTED_RE),
                         ("running", self._GAN_RUNNING_RE), ("resume", self._RESUME_RE),
                         ("deg_on", self._DEG_ON_RE), ("deg", self._DEG_STATE_RE),
                         ("lphase_on", self._LPHASE_ON_RE), ("lphase", self._LPHASE_STATE_RE)):
            events += [(m.start(), kind, m) for m in rx.finditer(buf) if m.end() > new_from]
        if not events:
            return
        changed = False
        for _, kind, m in sorted(events, key=lambda e: e[0]):
            if kind == "resume":
                self._phase_resumed = True
                continue
            if kind in ("lphase_on", "lphase"):
                self._lphase = int(m.group(2)) if kind == "lphase" else (self._lphase or 0)
                self._show_badge(self.lbl_lphase, self._lphase, _LOSS_COLOR,
                                 tr("Losses", "Losses"))
                continue
            if kind in ("deg_on", "deg"):
                self._deg = int(m.group(2)) if kind == "deg" else (self._deg or 0)
                self._show_deg()
                continue
            if getattr(self, "_phase", None) is None:
                if kind not in ("auto", "delayed"):
                    continue
                self._phase = "psnr"; self._gan_from = None
                self._gan_count = self._gan_min = self._gan_fixed = None
            if kind == "auto":
                # the real counter lives in the resumed training state: unknown until
                # the next validation reports it
                self._gan_count = ("?" if getattr(self, "_phase_resumed", False) else 0,
                                   int(m.group(1)))
                self._gan_min = int(m.group(2))
            elif kind == "delayed":
                self._gan_fixed = int(m.group(1))
            elif kind == "count":
                self._gan_count = (int(m.group(1)), int(m.group(2)))
            elif kind == "started":
                self._phase = "gan"; self._gan_from = int(m.group(1))
            elif kind == "running" and self._phase == "psnr":
                self._phase = "gan"
            else:
                continue
            changed = True
        if changed:
            self._show_phase()

    def _show_badge(self, lab, pct, color, name):
        """Two-profile badge: profile 1 (PSNR), ramp 1 -> 2 with its %, profile 2 (GAN)."""
        if pct is None:
            lab.hide()
            return
        if pct <= 0:
            txt = tr(f"{name} : profil 1 (PSNR)", f"{name}: profile 1 (PSNR)")
        elif pct >= 100:
            txt = tr(f"{name} : profil 2 (GAN)", f"{name}: profile 2 (GAN)")
        else:
            txt = tr(f"{name} : 1 → 2  ·  {pct} %", f"{name}: 1 → 2  ·  {pct}%")
        lab.setText(txt)
        lab.setFixedHeight(26)
        lab.setStyleSheet(f"background:{color}; color:#fff; border-radius:6px; "
                          f"padding:2px 10px; font-weight:700; font-size:12px;")
        lab.show()

    def _show_deg(self):
        pct = getattr(self, "_deg", None)
        if pct is None:
            self.lbl_deg.hide()
            return
        if pct <= 0:
            txt = tr("Dégradations : profil 1 (PSNR)", "Degradations: profile 1 (PSNR)")
        elif pct >= 100:
            txt = tr("Dégradations : profil 2 (GAN)", "Degradations: profile 2 (GAN)")
        else:
            txt = tr(f"Dégradations : 1 → 2  ·  {pct} %", f"Degradations: 1 → 2  ·  {pct}%")
        self.lbl_deg.setText(txt)
        self.lbl_deg.setFixedHeight(26)
        self.lbl_deg.setStyleSheet(f"background:{_DEG_COLOR}; color:#fff; border-radius:6px; "
                                   f"padding:2px 10px; font-weight:700; font-size:12px;")
        self.lbl_deg.show()

    def _show_phase(self):
        phase = getattr(self, "_phase", None)
        if phase is None:
            self.lbl_phase.hide()
            self.progress.setStyleSheet(_main_bar_qss("#2ecc71"))
            return
        sp = lambda n: f"{n:,}".replace(",", " ")
        if phase == "gan":
            color = _PHASE_GAN
            txt = tr("Phase : GAN", "Phase: GAN")
            if self._gan_from:
                txt += tr(f"  ·  depuis {sp(self._gan_from)}", f"  ·  since {sp(self._gan_from)}")
        else:
            color = _PHASE_PSNR
            txt = tr("Phase : PSNR", "Phase: PSNR")
            if self._gan_count:
                txt += f"  ·  GAN auto {self._gan_count[0]}/{self._gan_count[1]}"
                if self._gan_min:
                    txt += f"  (≥ {sp(self._gan_min)})"
            elif self._gan_fixed:
                txt += tr(f"  ·  GAN à {sp(self._gan_fixed)}", f"  ·  GAN at {sp(self._gan_fixed)}")
        self.lbl_phase.setText(txt)
        self.lbl_phase.setFixedHeight(26)
        self.lbl_phase.setStyleSheet(f"background:{color}; color:#fff; border-radius:6px; "
                                     f"padding:2px 10px; font-weight:700; font-size:12px;")
        self.lbl_phase.show()
        self.progress.setStyleSheet(_main_bar_qss(color))

    def _set_progress_pct(self, pct):
        """Updates the in-app progress bar AND the Windows taskbar icon overlay —
        the CTk app had the latter, the Qt port never wired it at all (useful when
        the window is minimized/behind other apps during a long training)."""
        self.progress.setValue(pct)
        it = getattr(self, "_last_iter", 0) or 0
        self.progress.setFormat(f"{pct}%   ·   {it:,} / {self.target_iters:,}".replace(",", " ")
                                if it else f"{pct}%")
        try:
            from qt_ui.taskbar import init_taskbar, set_taskbar_progress
            init_taskbar(self)
            set_taskbar_progress(pct / 100.0, "normal")
        except Exception:
            pass

    # USS status lines come from the runner/UI, not the engine — never let the log
    # formatter's buffer/draining swallow them (it eats non-timestamp lines).
    _STATUS_RE = re.compile(r'^\s*(\[[A-Za-zÀ-ÿ]|>>|==|---|> Commande|🎛|🔍|✅|⚠|❌|\[SUCCES\]|\[STOP\])')

    def _on_runner_log(self, text):
        try:
            self._parse_phase(text)
        except Exception:
            pass
        if self._fmt is not None and self._STATUS_RE.match(text):
            # flush any pending engine record first so the last iter still shows
            try:
                flushed = self._fmt.flush()
                if flushed:
                    self._append_ansi(flushed)
            except Exception:
                pass
            self._append(text)
            return
        if self._fmt is not None:
            try:
                self._fmt.current_iter = self._last_iter
                rendered = self._fmt.feed(text)
                if rendered:
                    self._append_ansi(rendered)
                # keep best PSNR/SSIM cards in sync with the formatter's tracking
                _improved = False
                if self._fmt.best_psnr > self.best_psnr:
                    self.best_psnr = self._fmt.best_psnr
                    self.cards["BEST PSNR"].setText(f"{self.best_psnr:.4f} dB\n(@ {self._fmt.best_psnr_iter})")
                    _improved = True
                if self._fmt.best_ssim > self.best_ssim:
                    self.best_ssim = self._fmt.best_ssim
                    self.cards["BEST SSIM"].setText(f"{self.best_ssim:.4f}\n(@ {self._fmt.best_ssim_iter})")
                    _improved = True
                if _improved:
                    self._persist_best_metrics()
                # drain any new PSNR/SSIM validation readings into the Metrics graph
                # history (the formatter only tracks the single best value itself)
                if self._fmt.psnr_points:
                    self.hist["psnr"].extend(self._fmt.psnr_points)
                    self._fmt.psnr_points.clear()
                if self._fmt.ssim_points:
                    self.hist["ssim"].extend(self._fmt.ssim_points)
                    self._fmt.ssim_points.clear()
                if getattr(self._fmt, "extra_points", None):
                    self._drain_extra_metrics(self._fmt.extra_points)
                self._apply_metrics(self._fmt.metrics)
            except Exception:
                self._append(text)
                self._parse_stats(text)
        else:
            self._append(text)
            self._parse_stats(text)

    def _drain_extra_metrics(self, pts):
        """USRS texture metrics: move the formatter's readings into the graph history and
        refresh the 5 cards (best value @ iter; HF sharpness = last value, zone color)."""
        from src.core import val_extra_metrics as _vem
        for k, lst in list(pts.items()):
            if not lst:
                continue
            self.hist.setdefault(k, []).extend(lst)
            pts[k] = []
            card = self.cards.get(k)
            if card is None:
                continue
            if k == "usrs_hf_ratio":
                it, v = lst[-1]
                col, word = _vem.ratio_zone(v)
                card.setText(f"{v:.3f}\n({word} @ {it})")
                card.setStyleSheet(f"color:{col}; font-weight:700; font-size:15px;")
                continue
            lower = _vem.better(k) == "lower"
            for it, v in lst:
                b = self._extra_best.get(k)
                if b is None or (v < b[0] if lower else v > b[0]):
                    self._extra_best[k] = (v, it)
            v, it = self._extra_best[k]
            card.setText((f"{v:.3f}" if k.startswith("usrs_hf") else f"{v:.2f}") + f"\n(@ {it})")

    def _apply_metrics(self, m):
        """Fill dashboard cards + the metrics graph history from the formatter's parsed
        iter record (robust — the record is already joined to a single line)."""
        if not m:
            return
        cur = m.get("iter")
        if cur is not None:
            self._last_iter = cur
            self.cards["ITÉRATION"].setText(f"{cur} / {self.target_iters}")
            pct = min(100, int(cur / max(self.target_iters, 1) * 100))
            self._set_progress_pct(pct)
            if not self.hist["iter"] or self.hist["iter"][-1] != cur:
                self.hist["iter"].append(cur)
        if "epoch" in m:
            self.cards["EPOCH"].setText(str(m["epoch"]))
        if "perf" in m:
            self.cards["VITESSE"].setText(f"{m['perf']} it/s")
            try:
                self._last_speed = float(m["perf"])
            except (TypeError, ValueError):
                pass
        if "lr" in m:
            self.cards["LR"].setText(str(m["lr"]))
        if "eta" in m:
            self.cards["FIN ESTIMÉE"].setText(str(m["eta"]))
        if "loss_g" in m:
            self.cards["LOSS G"].setText(self._fmt_loss(m["loss_g"]))
            if cur is not None:
                try:
                    self.hist["loss"].append((cur, float(m["loss_g"])))
                except (ValueError, TypeError):
                    pass
        if "loss_d" in m:
            self.cards["LOSS D"].setText(self._fmt_loss(m["loss_d"]))
            if cur is not None:
                try:
                    self.hist["loss_d"].append((cur, float(m["loss_d"])))
                except (ValueError, TypeError):
                    pass

    @staticmethod
    def _fmt_loss(v):
        """Human-readable loss (0.0285) instead of scientific (2.85e-02)."""
        try:
            f = float(v)
        except (ValueError, TypeError):
            return str(v)
        if f == 0:
            return "0.0000"
        if 1e-4 <= abs(f) < 1e4:
            return f"{f:.4f}"
        return f"{f:.2e}"

    # Tighter line height (% of natural) so the vertical box-drawing chars (║) touch
    # between rows instead of looking dashed. Cosmetic only.
    _LINE_HEIGHT_PCT = 91

    def _tighten(self, cur):
        bf = QTextBlockFormat()
        bf.setLineHeight(float(self._LINE_HEIGHT_PCT),
                         QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        cur.mergeBlockFormat(bf)

    def _append_ansi(self, text):
        """Render ANSI-coded log text into the QPlainTextEdit with per-token colors,
        preserving the monospace alignment of validation boxes / iter records."""
        sb = self.log.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        cur = self.log.textCursor(); cur.movePosition(QTextCursor.End)
        self._tighten(cur)
        color = None
        parts = _ANSI_RE.split(text)  # [text, code, text, code, ...]
        for idx, part in enumerate(parts):
            if idx % 2 == 1:  # SGR code
                if part in ("0", ""):
                    color = None
                else:
                    color = _ANSI_COLORS.get(part) or _ANSI_COLORS.get(part.split(";")[0])
            elif part:
                fmt = QTextCharFormat()
                # carry the widget's monospace font explicitly — an empty QTextCharFormat
                # otherwise inherits the proportional app font and breaks box alignment.
                fmt.setFont(self.log.font())
                if color:
                    fmt.setForeground(QColor(color))
                cur.insertText(part, fmt)
        # trim to keep the widget light (mirror CTk's 3000/2000 policy)
        if self.log.blockCount() > 3000:
            c2 = self.log.textCursor(); c2.movePosition(QTextCursor.Start)
            c2.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, self.log.blockCount() - 2000)
            c2.removeSelectedText()
        if at_bottom:
            self.log.moveCursor(QTextCursor.End)

    def _on_finished(self, code=0):
        self._gallery_auto(start=False)
        try:
            from qt_ui.taskbar import set_taskbar_progress
            set_taskbar_progress(1.0, "none" if (code == 0 and not self._user_stopped) else
                                 ("paused" if self._user_stopped else "error"))
        except Exception:
            pass
        # flush any buffered engine record before the closing banner
        if self._fmt is not None:
            try:
                f = self._fmt.flush()
                if f:
                    self._append_ansi(f)
            except Exception:
                pass
        # Outcome → colored banner: green ✅ clean finish, yellow ⏸ user interruption,
        # red ✗ error/crash. A user soft-stop saves gracefully (code 0) but is still an
        # interruption, so _user_stopped takes priority over the exit code.
        if self._user_stopped:
            outcome = ("33", "⏸", tr("Entraînement interrompu par l'utilisateur — modèle et états sauvegardés.", "Training interrupted by the user — model and states saved."))
        elif code == 0 and getattr(self.runner, "save_failed", False):
            outcome = ("33", "⚠", tr("Processus terminé, mais des sauvegardes ont échoué (disque plein ?) — reprise depuis le dernier état complet.",
                                     "Process finished, but some saves failed (disk full?) — resume restarts from the last complete state."))
        elif code == 0:
            outcome = ("32", "✅", tr("Processus terminé — modèle et états sauvegardés.", "Process finished — model and states saved."))
        else:
            outcome = ("31", "✗", tr(f"Entraînement échoué / crash (code {code}).", f"Training failed / crashed (code {code})."))
        _col, _icon, _msg = outcome
        self._append_ansi(f"\n\x1b[{_col}m[INFO] {_msg}\x1b[0m\n")
        # finalize the Historique row (was left "running" forever otherwise)
        try:
            from src.core.training_history import record_training_end
            status = "interrupted" if self._user_stopped else ("completed" if code == 0 else "failed")
            avg_pw = (sum(self._power_samples) / len(self._power_samples)) if self._power_samples else 0.0
            record_training_end(self._hist_id or 0, status=status,
                                avg_speed=self._last_speed, vram_peak_mb=self._vram_peak_mb,
                                power_avg_w=avg_pw)
        except Exception:
            pass
        # auto-stop the gallery server if it was running (parity with CTk end behaviour)
        try:
            from src.core.gallery_server import get_server
            srv = get_server()
            if getattr(srv, "httpd", None):
                srv.stop(); self._append("[" + tr("Serveurs", "Servers") + "] " + tr("Galerie auto-arrêtée.", "Gallery auto-stopped.") + "\n")
        except Exception:
            pass
        _tail = "" if code == 0 else f" (code {code})"
        self._append_ansi(f"\x1b[{_col}m== {_icon} " + tr("Terminé", "Finished") + f"{_tail} ==\x1b[0m\n")
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self._tick.stop()
        self.btn_stop.setText("⏹  STOP & SAVE"); self._soft_stop_done = False
        try:
            from qt_ui.notify import notify, play_sound
            if self._user_stopped:
                notify(self.settings, tr("Entraînement interrompu", "Training interrupted"),
                       tr("Arrêt demandé — modèle et états sauvegardés.", "Stop requested — model and states saved."), "notif_win11_training")
            elif code == 0:
                notify(self.settings, tr("Entraînement terminé", "Training finished"), tr("La session d'entraînement est terminée.", "The training session has finished."),
                       "notif_win11_training")
                play_sound(self.settings, "success", "sound_on")
            else:
                notify(self.settings, tr("Entraînement échoué", "Training failed"), tr(f"Crash / code de sortie {code}.", f"Crash / exit code {code}."), "notif_win11_errors")
                play_sound(self.settings, "error", "snd_error")
        except Exception:
            pass
        # A manual STOP & SAVE also exits with code 0 (clean shutdown of the training
        # process), so this used to prompt "Shut down PC?" after the user just asked
        # to pause, not finish — self._user_stopped already exists to distinguish the
        # two (drives the yellow "interrupted" banner a few lines above) but wasn't
        # checked here (2026-09-17 fix).
        # Queue: tell it the run is over. A queued run that is not the last one never
        # triggers the auto shutdown (one-shot flag set by external_start).
        _no_shutdown = bool(getattr(self, "suppress_shutdown", False))
        self.suppress_shutdown = False
        try:
            self.run_finished.emit(int(code), bool(self._user_stopped))
        except Exception:
            pass
        if self.chk_shutdown.isChecked() and code == 0 and not self._user_stopped and not _no_shutdown:
            # Scheduled right away (60 s) with a NON-blocking cancel box: the old modal
            # question defaulted to "No" and waited for a click, so an unattended PC never
            # shut down — the whole point of the option.
            self._append(tr("\n[ATTENTION] ARRÊT AUTOMATIQUE DANS 60 SECONDES...\n",
                            "\n[WARNING] AUTO SHUTDOWN IN 60 SECONDS...\n"))
            if sys.platform == "win32":
                os.system("shutdown /s /t 60")
            else:
                os.system("shutdown -h +1")
            box = QMessageBox(QMessageBox.Warning, tr("Arrêt automatique", "Auto shutdown"),
                              tr("Le training est terminé : le PC s'éteint dans 60 secondes.",
                                 "Training is complete: the PC shuts down in 60 seconds."), parent=self)
            cancel = box.addButton(tr("Annuler l'arrêt", "Cancel the shutdown"), QMessageBox.RejectRole)

            def _cancel_shutdown(btn, _c=cancel):
                if btn is _c:
                    os.system("shutdown /a" if sys.platform == "win32" else "shutdown -c")
                    self._append(tr("[INFO] Arrêt automatique annulé par l'utilisateur.\n",
                                    "[INFO] Auto shutdown cancelled by user.\n"))
            box.buttonClicked.connect(_cancel_shutdown)
            box.setModal(False); box.show()
            self._shutdown_box = box

    def _stop(self):
        self._user_stopped = True
        already = getattr(self, "_soft_stop_done", False)
        if self.runner is not None:
            try:
                # 1st click → soft stop (save), 2nd click → KILL. The runner enforces this
                # via its stop_requested flag; we mirror it on the button label/log.
                self.runner.stop_training(lambda s: self._log_sig.emit(str(s)))
            except Exception as e:
                self._append(f"\n❌ stop : {e}\n")
        if not already:
            self._soft_stop_done = True
            self.btn_stop.setText(tr("⏹  FORCER KILL", "⏹  FORCE KILL"))
            self._append("\n== " + tr("Arrêt demandé (soft stop) — re-cliquez pour forcer le KILL", "Stop requested (soft stop) — click again to force KILL") + " ==\n")
        else:
            self._append("\n== " + tr("KILL forcé demandé", "Force KILL requested") + " ==\n")
        self._tick.stop()

    def _append(self, text):
        sb = self.log.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        cur = self.log.textCursor(); cur.movePosition(QTextCursor.End)
        self._tighten(cur); cur.insertText(text)
        if at_bottom:
            self.log.moveCursor(QTextCursor.End)

    def _parse_stats(self, text):
        cur_iter = None
        m = re.search(r'iter:\s*([\d,]+)', text)
        if m:
            cur_iter = int(m.group(1).replace(",", ""))
            self._last_iter = cur_iter
            self.cards["ITÉRATION"].setText(f"{cur_iter} / {self.target_iters}")
            pct = min(100, int(cur_iter / max(self.target_iters, 1) * 100))
            self._set_progress_pct(pct)
            self.hist["iter"].append(cur_iter)
        m = re.search(r'l_g_total[:=\s]+([\d.eE+-]+)', text) or re.search(r'loss[:=\s]+([\d.eE+-]+)', text)
        if m:
            self.cards["LOSS G"].setText(m.group(1))
            if cur_iter is not None:
                try:
                    self.hist["loss"].append((cur_iter, float(m.group(1))))
                except ValueError:
                    pass
        m = re.search(r'l_d[:=\s]+([\d.eE+-]+)', text)
        if m:
            self.cards["LOSS D"].setText(m.group(1))
        m = re.search(r'lr[:=\s(]+([\d.eE+-]+)', text)
        if m:
            self.cards["LR"].setText(m.group(1))
        m = re.search(r'psnr[:=\s]+([\d.]+)', text, re.I)
        if m:
            p = float(m.group(1))
            # PSNR usually arrives in the boxed validation summary block, a separate
            # log chunk with no "iter:" of its own — fall back to the last seen
            # iteration so the point isn't silently dropped (was: 0 pts in Metrics).
            psnr_iter = cur_iter if cur_iter is not None else self._last_iter
            if psnr_iter:
                self.hist["psnr"].append((psnr_iter, p))
            if p > self.best_psnr:
                self.best_psnr = p; self.cards["BEST PSNR"].setText(f"{p:.2f} dB")
                self._persist_best_metrics()
        m = re.search(r'ssim[:=\s]+([\d.]+)', text, re.I)
        if m:
            s = float(m.group(1))
            if s > self.best_ssim:
                self.best_ssim = s; self.cards["BEST SSIM"].setText(f"{s:.4f}")
                self._persist_best_metrics()

    def _update_timer(self):
        if self.start_time:
            el = int(time.time() - self.start_time)
            self.cards["DURÉE"].setText(f"{el // 3600:02d}:{(el % 3600) // 60:02d}:{el % 60:02d}")

    # ── monitoring (pynvml + psutil) ──────────────────────────────────────────────
    def _cpu_name(self):
        try:
            if os.name == "nt":
                import winreg
                k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            pass
        import platform
        return platform.processor() or "CPU"

    # ── AIDA64 registry sensors (CPU temp / power / clock) ────────────────────────
    def _aida_enabled(self) -> bool:
        """Settings › "Enable AIDA64" (the readers ignored it)."""
        return str(self.settings.get("use_aida64", "true")).lower() != "false"

    def _aida_temp(self) -> str:
        """CPU temp via AIDA64 SensorValues (Value.TCPU/TCPUPKG/Tcore1)."""
        if os.name != "nt" or not self._aida_enabled():
            return ""
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\FinalWire\AIDA64\SensorValues")
            i = 0
            while True:
                try:
                    name, data, _ = winreg.EnumValue(k, i)
                    if name in ("Value.TCPU", "Value.TCPUPKG", "Value.Tcore1"):
                        return f"{data} °C"
                    i += 1
                except OSError:
                    break
        except Exception:
            pass
        return ""

    def _cpu_temp_fallback(self) -> str:
        """CPU temp fallback: OpenHardwareMonitor WMI → MSAcpi ThermalZone (wmic)."""
        if os.name != "nt":
            return ""
        try:
            import wmi  # type: ignore
            w = wmi.WMI(namespace="root/OpenHardwareMonitor")
            for s in w.Sensor():
                if getattr(s, "SensorType", "") == "Temperature" and "CPU" in getattr(s, "Name", ""):
                    return f"{s.Value:.0f} °C"
        except Exception:
            pass
        try:
            import subprocess
            res = subprocess.run(
                ["wmic", r"/namespace:\\root\WMI", "PATH",
                 "MSAcpi_ThermalZoneTemperature", "get", "CurrentTemperature", "/value"],
                capture_output=True, text=True, timeout=2,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in res.stdout.splitlines():
                if "CurrentTemperature=" in line:
                    raw = int(line.split("=")[1].strip())
                    celsius = (raw / 10.0) - 273.15
                    if 0 < celsius < 120:
                        return f"{celsius:.0f} °C"
        except Exception:
            pass
        return ""

    def _aida_watts(self):
        """CPU package power (W) via AIDA64 (Value.PCPUPKG/PCPU/PCPUIA)."""
        if os.name != "nt" or not self._aida_enabled():
            return None
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\FinalWire\AIDA64\SensorValues")
            i = 0
            while True:
                try:
                    name, data, _ = winreg.EnumValue(k, i)
                    if name in ("Value.PCPUPKG", "Value.PCPU", "Value.PCPUIA"):
                        return float(data)
                    i += 1
                except OSError:
                    break
        except Exception:
            pass
        return None

    def _cpu_freq_mhz(self):
        """CPU clock (MHz): AIDA64 SCPUCLK → range scan → NtPowerInformation → psutil."""
        cur = None
        if os.name == "nt" and self._aida_enabled():
            try:
                import winreg
                ak = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\FinalWire\AIDA64\SensorValues")
                try:
                    val, _ = winreg.QueryValueEx(ak, "Value.SCPUCLK")
                    v = float(val)
                    if 500.0 < v < 8000.0:
                        cur = int(v)
                except (FileNotFoundError, OSError, ValueError, TypeError):
                    pass
                if cur is None:
                    cands, i = [], 0
                    while True:
                        try:
                            _nm, dat, _ = winreg.EnumValue(ak, i)
                            try:
                                v = float(dat)
                                if 2500.0 < v < 6000.0:
                                    cands.append(v)
                            except (ValueError, TypeError):
                                pass
                            i += 1
                        except OSError:
                            break
                    if cands:
                        cur = int(max(cands))
                winreg.CloseKey(ak)
            except Exception:
                pass
            if cur is None:
                try:
                    import ctypes

                    class _PPI(ctypes.Structure):
                        _fields_ = [("Number", ctypes.c_ulong), ("MaxMhz", ctypes.c_ulong),
                                    ("CurrentMhz", ctypes.c_ulong), ("MhzLimit", ctypes.c_ulong),
                                    ("MaxIdleState", ctypes.c_ulong), ("CurrentIdleState", ctypes.c_ulong)]
                    n = os.cpu_count() or 1
                    buf = (_PPI * n)()
                    if ctypes.windll.ntdll.NtPowerInformation(11, None, 0, buf, ctypes.sizeof(buf)) == 0:
                        cur = max(buf[j].CurrentMhz for j in range(n))
                except Exception:
                    pass
        if cur is None:
            try:
                import psutil
                f = psutil.cpu_freq()
                if f:
                    cur = f.current
            except Exception:
                pass
        return cur

    def _poll_stats(self):
        # GPU
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # silence pynvml/nvidia-ml-py FutureWarning
                import pynvml
            if not getattr(self, "_nvml_init", False):
                pynvml.nvmlInit(); self._nvml_h = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_init = True
            h = self._nvml_h
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            self.lbl_gpu.setText(name)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h); util = pynvml.nvmlDeviceGetUtilizationRates(h)
            vpct = int(mem.used / mem.total * 100)
            self._set_bar(self.bar_vram, self.val_vram, vpct, f"{mem.used//(1024**3)}.{(mem.used%(1024**3))//(1024**2)//100}/{mem.total//(1024**3)}GB")
            if self.runner is not None and getattr(self.runner, "is_running", False):
                self._vram_peak_mb = max(self._vram_peak_mb, int(mem.used / (1024 ** 2)))
            self._set_bar(self.bar_util, self.val_util, util.gpu, f"{util.gpu}%")
            self._set_bar(self.bar_mc, self.val_mc, util.memory, f"{util.memory}%")
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, 0); self.lbl_gtemp.setText(f"{temp} °C")
            except Exception:
                pass
            try:
                pcie = pynvml.nvmlDeviceGetPcieThroughput(h, 0) / 1024.0  # TX KB/s → MB/s
                self._set_bar(self.bar_pcie, self.val_pcie, min(100, pcie / 80), f"{pcie:.0f}MB/s")
            except Exception:
                pass
            try:
                pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                self._set_bar(self.bar_gpw, self.val_gpw, pw / max(cap, 1) * 100, f"{pw:.0f}/{cap:.0f}W")
                if self.runner is not None and getattr(self.runner, "is_running", False):
                    self._power_samples.append(pw)
                    if len(self._power_samples) > 600:  # cap memory on very long runs
                        self._power_samples = self._power_samples[-600:]
            except Exception:
                pass
        except Exception:
            pass
        # CPU / system
        try:
            import psutil
            c = psutil.cpu_percent(); self._set_bar(self.bar_cpu, self.val_cpu, c, f"{c:.1f}%")
            r = psutil.virtual_memory(); self._set_bar(self.bar_ram, self.val_ram, r.percent, f"{r.percent:.1f}%")
            d = psutil.disk_usage(os.path.abspath(os.sep)); self._set_bar(self.bar_disk, self.val_disk, d.percent, f"{d.percent:.1f}%")
            # CPU freq — AIDA64 SCPUCLK → NtPowerInformation → psutil
            try:
                mhz = self._cpu_freq_mhz()
                if mhz:
                    ghz = float(mhz) / 1000.0
                    if not hasattr(self, "_cpu_freq_max_ghz"):
                        fmax = psutil.cpu_freq()
                        base = (fmax.max / 1000.0) if (fmax and fmax.max) else 5.5
                        self._cpu_freq_max_ghz = max(base * 1.4, 4.5)
                    if ghz > self._cpu_freq_max_ghz:
                        self._cpu_freq_max_ghz = ghz * 1.05
                    self._set_bar(self.bar_freq, self.val_freq,
                                  min(100.0, ghz / max(self._cpu_freq_max_ghz, 0.1) * 100), f"{ghz:.2f}GHz")
            except Exception:
                pass
            # CPU power (AIDA64 registry)
            try:
                w = self._aida_watts()
                if w is not None:
                    CPU_WATT_MAX = 200.0
                    self._set_bar(self.bar_cpw, self.val_cpw, min(100.0, w / CPU_WATT_MAX * 100), f"{w:.0f}W")
            except Exception:
                pass
            # CPU temp (AIDA64 → fallback)
            try:
                t = self._aida_temp() or self._cpu_temp_fallback()
                if t:
                    self.lbl_ctemp.setText(t)
            except Exception:
                pass
        except Exception:
            pass

    # ── header buttons → dialogs ──────────────────────────────────────────────────
    def _cfg_dict(self):
        path = self.entries["config_path"].text().strip()
        if not path or not os.path.isfile(path):
            return {}
        try:
            if path.endswith(".toml"):
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib
                with open(path, "rb") as fh:
                    return tomllib.load(fh)
            import yaml
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception:
            return {}

    def _servers(self):
        from qt_ui.tabs.run_dialogs import ServersDialog
        ServersDialog(self.settings, port="6006",
                     script_path=self.entries["script_path"].text().strip(),
                     config_path=self.entries["config_path"].text().strip()).exec()

    def _metrics(self):
        from qt_ui.tabs.run_dialogs import MetricsDialog
        MetricsDialog(self.hist, self).exec()

    def _lr(self):
        from qt_ui.tabs.run_dialogs import LRDialog
        LRDialog(self._cfg_dict(), self).exec()

    def _ai(self):
        from qt_ui.tabs.run_dialogs import AIDialog
        # the whole log (the dialog applies its own "last N lines / whole" choice — it was cut
        # to 6000 characters here first) and the config CONTENT (only the path was sent)
        log = self.log.toPlainText()
        cfg_path = self.entries["config_path"].text().strip()
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg_text = fh.read()
        except Exception:
            cfg_text = cfg_path
        AIDialog(self.settings, log_text=log, config_text=cfg_text, parent=self).exec()

    def _validation(self):
        from qt_ui.tabs.run_dialogs import ValidationDialog
        ValidationDialog(self.settings, self).exec()
