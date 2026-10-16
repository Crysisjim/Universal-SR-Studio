"""
config_tab.py — Configuration tab (PySide6 port) — faithful to the CustomTkinter
tab_config (v2.5.8): left sidebar nav (STUDIO CONFIG + 8 pages + Load/Save/Test),
right panel with a VRAM-estimator bar on top + swapping pages (QStackedWidget).

Pages: Général · Architecture (capability bars + dynamic arch params + GAN) ·
Datasets (mode + paths + augmentations + reprise + ECO) · Entraînement
(hyperparams sliders + scheduler + log/validation + metrics) · Dégradations(stub) ·
Système/Avancé · Vérification AI(stub) · Pipeline PSNR→GAN(stub).

Reuses the backend (ConfigHandler.generate_config, compute_estimator.estimate_vram,
descriptions.get_arch_families / ARCH_PROFILES / REDUX_ARCH_FIELDS / ARCH_FIELDS).
"""
from __future__ import annotations

import os
import math
import platform

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox,
    QGroupBox, QPushButton, QCheckBox, QStackedWidget, QFrame, QFileDialog,
    QMessageBox, QScrollArea, QProgressBar, QSlider, QDialog, QPlainTextEdit,
)
from PySide6.QtGui import QPixmap, QPainter, QColor, QRadialGradient, QPolygonF
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QPointF

from qt_ui.widgets import LabeledSlider, hsep
from qt_ui.i18n import tr

# Video/temporal archs (traiNNer/archs/arch_info.py::TEMPORAL_ARCHS) — the only
# archs that can consume a "sequence" dataset (PairedVideoDataset / clip windows).
_TEMPORAL_ARCHS = {"tscunet", "temporalspan", "temporalspanv2", "tfdat", "tfdatv2", "motsr"}

# (label, key) — same order as the v2.5.8 CTk sidebar
_NAV = [
    (tr("Architecture", "Architecture"), "net"), (tr("Datasets", "Datasets"), "data"),
    (tr("Entraînement", "Training"), "train"), (tr("Dégradations", "Degradations"), "deg"),
    (tr("Système / Avancé", "System / Advanced"), "adv"),
    (tr("Vérification AI", "AI Verification"), "verify"),
    (tr("Pipeline PSNR→GAN", "Pipeline PSNR→GAN"), "pipeline"),
]

_LOSSES = [
    ("loss_pixel", "Pixel (L1/Charbonnier)"), ("loss_mssim", "MS-SSIM"),
    ("loss_percep", "Perceptual"), ("loss_ldl", "LDL"), ("loss_dists", "DISTS"),
    ("loss_ff", "Focal Frequency"), ("loss_color", "Color"), ("loss_luma", "Luma"),
    ("loss_hsluv", "HSLuv"), ("loss_cosim", "Cosim"), ("loss_gv", "Gradient Var."),
    ("loss_contextual", "Contextual"), ("loss_percep_anime", "Percep. Anime"),
    ("loss_linedistiller", "LineDistiller ★"), ("loss_percep_fp16", "Percep. FP16 ★"),
    ("loss_aesop", "AESOP ★"), ("loss_bcedice", "BCE+Dice ★"),
]

_VGG_LAYERS = [("conv1_2", "0.0"), ("conv2_2", "0.0"), ("conv3_2", "0.0"), ("conv3_4", "0.0"),
               ("conv4_2", "0.0"), ("conv4_4", "0.0"), ("conv5_2", "0.0"), ("conv5_4", "1.0")]

# Loss fields disabled per engine (faithful to CTk apply_engine_logic)
_NEOSR_ONLY = [
    "eco_init", "eco_iters", "eco_schedule",
    "loss_msswd", "weight_loss_msswd",
    "loss_consistency", "weight_loss_consistency",
    "consistency_blur", "consistency_cosim", "consistency_saturation", "consistency_brightness",
    "loss_ncc", "weight_loss_ncc",
    "loss_kl", "weight_loss_kl",
    "loss_wavelet", "weight_loss_wavelet", "wavelet_init",
    "loss_fdl", "weight_loss_fdl", "fdl_model",
    "mssim_window_size", "mssim_sigma", "mssim_k1", "mssim_k2",
]
_REDUX_ONLY = [
    "pretrain_eco", "eco_mode_kind", "eco_end_ratio",
    "loss_edge", "weight_loss_edge",
    "loss_hsluv", "weight_loss_hsluv", "hsluv_hue_weight", "hsluv_sat_weight", "hsluv_lum_weight",
    "loss_cosim", "weight_loss_cosim", "cosim_lambda",
    "loss_color", "weight_loss_color", "color_criterion",
    "loss_gv", "weight_loss_gv", "gv_patch_size", "gv_criterion",
    "loss_luma", "weight_loss_luma", "luma_criterion",
    "loss_contextual", "weight_loss_contextual", "ctx_distance_type", "ctx_band_width",
    "loss_spark", "weight_loss_spark", "spark_criterion", "spark_path", "spark_max_score",
    "loss_percep_anime", "weight_loss_percep_anime", "percep_anime_criterion",
    "loss_linedistiller", "weight_loss_linedistiller", "linedistiller_criterion",
    "loss_percep_fp16", "weight_loss_percep_fp16", "percep_fp16_criterion",
    "loss_aesop", "weight_loss_aesop", "aesop_criterion", "aesop_ae_path",
    "loss_bcedice", "weight_loss_bcedice", "bcedice_weight_bce", "bcedice_weight_dice",
]
# Fully engine-locked loss rows (grey whole row incl. labels)
_ROW_LOCK_NEOSR = ["loss_msswd", "loss_consistency", "loss_ncc", "loss_kl", "loss_wavelet", "loss_fdl"]
_ROW_LOCK_REDUX = ["loss_edge", "loss_hsluv", "loss_cosim", "loss_color", "loss_gv", "loss_luma", "loss_contextual",
                   "loss_linedistiller", "loss_percep_fp16", "loss_aesop", "loss_bcedice",
                   "loss_percep_anime", "loss_spark"]

_PERF_LABELS = [
    tr("Netteté (Sharpness)", "Sharpness"), tr("Texture/Détail", "Texture/Detail"),
    tr("Fidélité (PSNR)", "Fidelity (PSNR)"), tr("Vitesse (Inference)", "Speed (Inference)"),
    tr("Légèreté VRAM", "VRAM Lightness"), tr("Efficacité Anime", "Anime Efficiency"),
    tr("Efficacité Réaliste", "Realistic Efficiency"),
]


_GAN_THR = [(0.008, "🔵"), (0.08, "🟢"), (0.15, "🟡"), (0.35, "🟠"), (0.85, "🔴")]
_LR_THR = [(2e-5, "🔵"), (8e-5, "🟢"), (2e-4, "🟡"), (5e-4, "🟠"), (2e-3, "🔴")]


class _IntensityDot(QWidget):
    """Intensity marker painted with QPainter. The colour emoji it replaces (🔵🟢🟡🟠🔴💥)
    was always clipped at the bottom by Qt's colour-font rendering, whatever the label
    size. Same setText(emoji) API, so callers and the thresholds stay unchanged."""
    _COLORS = {"🔵": "#3b82f6", "🟢": "#22c55e", "🟡": "#facc15",
               "🟠": "#f97316", "🔴": "#ef4444", "💥": "#ef4444"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._key = "🟢"
        self.setFixedSize(24, 24)

    def setText(self, key):
        if key != self._key:
            self._key = key
            self.update()

    def text(self):
        return self._key

    def paintEvent(self, _e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        c = QColor(self._COLORS.get(self._key, "#22c55e"))
        cx, cy = self.width() / 2, self.height() / 2
        if self._key == "💥":  # maximum: yellow burst behind the red ball
            pts = []
            for i in range(16):
                r = 11.5 if i % 2 == 0 else 7.0
                a = i * math.pi / 8
                pts.append(QPointF(cx + r * math.cos(a), cy + r * math.sin(a)))
            p.setPen(Qt.NoPen); p.setBrush(QColor("#facc15")); p.drawPolygon(QPolygonF(pts))
        r = 7.0
        g = QRadialGradient(QPointF(cx - 2.5, cy - 2.5), r * 1.6)
        g.setColorAt(0.0, c.lighter(160)); g.setColorAt(0.55, c); g.setColorAt(1.0, c.darker(140))
        p.setPen(Qt.NoPen); p.setBrush(g); p.drawEllipse(QPointF(cx, cy), r, r)
        p.end()


class _LogSlider(QWidget):
    """Log-scale slider faithful to CTk: leading label · colour-emoji that changes
    with intensity (💥 at max) · '− intensité +' caption · log slider · value entry.
    All controls aligned on the middle grid row. Used for GAN weight and LRs."""
    valueChangedF = Signal(float)

    def __init__(self, label, value, lo=0.001, hi=1.0, fmt="{:.3f}",
                 thresholds=_GAN_THR, width=200, parent=None):
        super().__init__(parent)
        self._lo, self._hi, self._fmt, self._thr = lo, hi, fmt, thresholds
        g = QGridLayout(self); g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8); g.setVerticalSpacing(1)
        cap = QLabel(tr("−  intensité  +", "−  intensity  +")); cap.setAlignment(Qt.AlignCenter)
        cap.setStyleSheet("color:#9aa3b0; font-style:italic; font-size:11px;")
        g.addWidget(cap, 0, 2)
        g.addWidget(QLabel(label), 1, 0)
        self.dot = _IntensityDot()
        g.addWidget(self.dot, 1, 1)
        self.sld = QSlider(Qt.Horizontal); self.sld.setRange(0, 1000); self.sld.setMinimumWidth(width)
        g.addWidget(self.sld, 1, 2)
        self.entry = QLineEdit(); self.entry.setFixedWidth(72)
        g.addWidget(self.entry, 1, 3)
        spacer = QWidget(); spacer.setFixedHeight(15); g.addWidget(spacer, 2, 2)
        g.setColumnStretch(2, 1)
        self.sld.valueChanged.connect(self._from_slider)
        self.entry.editingFinished.connect(self._from_entry)
        self.set_value(value)

    def _emoji(self, v):
        for mx, e in self._thr:
            if v < mx:
                return e
        return "💥"

    def _pos_to_val(self, p):
        return self._lo * (self._hi / self._lo) ** (p / 1000.0)

    def _val_to_pos(self, v):
        v = max(self._lo, min(self._hi, v))
        return round(1000 * math.log(v / self._lo) / math.log(self._hi / self._lo))

    def _from_slider(self, p):
        v = self._pos_to_val(p)
        self.entry.blockSignals(True); self.entry.setText(self._fmt.format(v)); self.entry.blockSignals(False)
        self.dot.setText(self._emoji(v)); self.valueChangedF.emit(v)

    def _from_entry(self):
        try:
            v = float(self.entry.text())
        except ValueError:
            return
        self.sld.blockSignals(True); self.sld.setValue(self._val_to_pos(v)); self.sld.blockSignals(False)
        self.dot.setText(self._emoji(v)); self.valueChangedF.emit(v)

    def set_value(self, v):
        self.sld.blockSignals(True); self.sld.setValue(self._val_to_pos(v)); self.sld.blockSignals(False)
        self.entry.setText(self._fmt.format(v)); self.dot.setText(self._emoji(v))

    def value(self):
        try:
            return float(self.entry.text())
        except ValueError:
            return self._lo


class _SliderBox(QWidget):
    """Integer slider with a value entry on the right and a label above — faithful
    to CTk add_slider_param (Batch Size / Accumulate / Patch Size)."""
    valueChangedF = Signal(float)

    def __init__(self, label, lo, hi, step, value, parent=None):
        super().__init__(parent)
        self._lo, self._hi, self._step = lo, hi, step
        self._steps = max(1, round((hi - lo) / step))
        col = QVBoxLayout(self); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(2)
        col.addWidget(QLabel(label))
        row = QHBoxLayout(); row.setSpacing(6)
        self.sld = QSlider(Qt.Horizontal); self.sld.setRange(0, self._steps)
        row.addWidget(self.sld, 1)
        self.entry = QLineEdit(); self.entry.setFixedWidth(60); row.addWidget(self.entry)
        col.addLayout(row)
        self.sld.valueChanged.connect(self._from_slider)
        self.entry.editingFinished.connect(self._from_entry)
        self.set_value(value)

    def _to_int(self, v):
        return max(0, min(self._steps, round((v - self._lo) / self._step)))

    def _to_val(self, i):
        return self._lo + i * self._step

    def _from_slider(self, i):
        v = self._to_val(i)
        self.entry.blockSignals(True); self.entry.setText(str(int(round(v)))); self.entry.blockSignals(False)
        self.valueChangedF.emit(v)

    def _from_entry(self):
        try:
            v = float(self.entry.text())
        except ValueError:
            return
        self.sld.blockSignals(True); self.sld.setValue(self._to_int(v)); self.sld.blockSignals(False)
        self.valueChangedF.emit(v)

    def set_value(self, v):
        self.sld.blockSignals(True); self.sld.setValue(self._to_int(v)); self.sld.blockSignals(False)
        self.entry.setText(str(int(round(v))))

    def value(self):
        try:
            return float(self.entry.text())
        except ValueError:
            return self._lo


class _PerfBars(QFrame):
    """Qt port of components/performance_bars.PerformanceBars — 7 labelled bars
    coloured by score, reusing ARCH_PROFILES + the exact bonus/malus maths."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        g = QGridLayout(self); g.setContentsMargins(12, 10, 12, 10)
        g.setHorizontalSpacing(10); g.setVerticalSpacing(4)
        self.bars = []
        for i, text in enumerate(_PERF_LABELS):
            lbl = QLabel(text); lbl.setStyleSheet("font-weight:700;")
            lbl.setMinimumWidth(130)
            g.addWidget(lbl, i, 0)
            bar = QProgressBar(); bar.setRange(0, 100); bar.setValue(0)
            bar.setTextVisible(False); bar.setFixedHeight(12)
            g.addWidget(bar, i, 1)
            self.bars.append(bar)
        legend = QLabel(tr("Calcul dynamique basé sur tous les paramètres.", "Dynamic calculation based on all parameters."))
        legend.setStyleSheet("color:gray; font-style:italic;")
        legend.setAlignment(Qt.AlignCenter)
        g.addWidget(legend, len(_PERF_LABELS), 0, 1, 2)
        g.setColumnStretch(1, 1)

    @staticmethod
    def _num(values, key, default=0, cast=float):
        if key not in values:
            return default
        try:
            v = values[key]
            if isinstance(v, str):
                v = v.lower()
                if v in ("true", "on", "1"):
                    return 1
                if v in ("false", "off", "0", ""):
                    return 0
            return cast(v)
        except Exception:
            return default

    def update_stats(self, arch, is_gan, disc_type, values, vram_frac=None):
        """Capability bars. VRAM Lightness comes from the estimator (vram_frac = estimated
        GB / card GB) when given, so it agrees with the VRAM gauge. Speed is the
        inference speed of the architecture only (training speed is shown in the VRAM
        line, from the same estimator)."""
        try:
            from src.core.descriptions import ARCH_PROFILES
        except Exception:
            ARCH_PROFILES = {}
        base = list(ARCH_PROFILES.get(arch, [5, 5, 5, 5, 5, 5, 5]))
        sharp = tex = fid = 0.0
        malus_vram = malus_speed = 0.0

        # Fallback VRAM heuristic, only without an estimate. The patch is in LQ pixels,
        # so a bigger scale means a bigger HR output (more VRAM), not less.
        scale = self._num(values, "scale", 4, int)
        bs = self._num(values, "batch_size", 4)
        patch = self._num(values, "patch_size", 64)
        if scale >= 4:
            malus_vram += 1.0
        if bs > 4:
            malus_vram += (bs - 4) * 0.2
        if patch > 64:
            malus_vram += (patch - 64) * 0.05
        if self._num(values, "use_amp", 0):
            malus_vram -= 1.5; malus_speed -= 1.0

        win = self._num(values, "dyn_window_size", 0)
        if win > 8:
            malus_vram += (win - 8) * 0.1; sharp += 0.5; tex += 0.5; fid += 0.25
        feat = self._num(values, "dyn_num_feat", 64)
        if feat > 64:
            cx = (feat - 64) / 32
            malus_vram += cx * 0.8; malus_speed += cx * 1.0
            sharp += cx * 0.4; tex += cx * 0.4; fid += cx * 0.2

        if is_gan:
            # Stronger GAN weight: more sharpness/texture, less fidelity (0.1 = reference).
            w = self._num(values, "dyn_gan_weight", 0.1)
            f = max(0.3, min(2.0, w / 0.1)) if w > 0 else 0.0
            sharp += 2 * f; tex += 2 * f; fid -= 2 * f
            if self._num(values, "use_disc2"):
                tex += 0.5; fid -= 0.3

        # Losses (quality effect only; their VRAM/speed cost is in the estimator).
        for key, ds, dt, df in (
                ("loss_linedistiller", 0.6, 0.0, 0.2), ("loss_percep_anime", 0.3, 0.6, 0.0),
                ("loss_percep", 0.3, 0.5, 0.0), ("loss_ldl", 0.4, 0.0, 0.0),
                ("loss_ff", 0.2, 0.3, 0.0), ("loss_dists", 0.2, 0.5, 0.2),
                ("loss_contextual", 0.0, 0.6, -0.2), ("loss_spark", 0.3, 0.2, 0.0),
                ("loss_fdl", 0.2, 0.5, 0.0), ("loss_edge", 0.3, 0.0, 0.0),
                ("loss_mssim", 0.0, 0.0, 0.4), ("loss_color", 0.0, 0.0, 0.2),
                ("loss_wavelet", 0.2, 0.2, 0.0), ("loss_msswd", 0.0, 0.3, 0.0)):
            if self._num(values, key):
                sharp += ds; tex += dt; fid += df

        if vram_frac is not None:
            vram_score = 10.0 - 9.0 * max(0.0, min(1.0, vram_frac))
        else:
            vram_score = base[4] - malus_vram
        final = [base[0] + sharp, base[1] + tex, base[2] + fid,
                 base[3] - malus_speed, vram_score, base[5], base[6]]
        for i, bar in enumerate(self.bars):
            val = max(1.0, min(10.0, final[i]))
            bar.setValue(int(val * 10))
            col = "#2ecc71" if val >= 7.5 else "#f1c40f" if val >= 4.5 else "#e74c3c"
            bar.setStyleSheet(f"QProgressBar::chunk{{background:{col};border-radius:2px;}}")


class ConfigTab(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.fields: dict[str, object] = {}
        self._total_vram = 11.0

        self._vram_timer = QTimer(self); self._vram_timer.setSingleShot(True)
        self._vram_timer.setInterval(300); self._vram_timer.timeout.connect(self._do_vram)

        root = QHBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ── left sidebar ──
        nav = QFrame(); nav.setObjectName("sidebar"); nav.setFixedWidth(210)
        nl = QVBoxLayout(nav); nl.setContentsMargins(14, 16, 14, 16); nl.setSpacing(5)
        title = QLabel("STUDIO CONFIG"); title.setObjectName("h2"); nl.addWidget(title)
        nl.addSpacing(8)
        self.stack = QStackedWidget()
        self._navbtns = []
        # nav buttons live inside a subtle panel (background, like the CTk sidebar frame)
        navpanel = QFrame(); navpanel.setObjectName("navpanel")
        pl = QVBoxLayout(navpanel); pl.setContentsMargins(6, 8, 6, 8); pl.setSpacing(4)
        for label, key in _NAV:
            b = QPushButton(label); b.setObjectName("nav"); b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self._show(k))
            pl.addWidget(b); self._navbtns.append((key, b))
        nl.addWidget(navpanel)
        nl.addStretch(1)
        for txt, slot, color in ((tr("📂 Charger Config", "📂 Load Config"), self._load, "ghost"),
                                 (tr("💾 Sauvegarder", "💾 Save"), self._save, ""),
                                 (tr("🚀 Sauver & Tester", "🚀 Save & Test"), self._save_test, "warn")):
            bt = QPushButton(txt)
            if color:
                bt.setObjectName(color)
            bt.clicked.connect(slot); nl.addWidget(bt)
        root.addWidget(nav)

        # ── right panel ──
        right = QWidget(); rl = QVBoxLayout(right)
        rl.setContentsMargins(20, 16, 20, 16); rl.setSpacing(12)
        vbox = QWidget()
        vl = QVBoxLayout(vbox); vl.setContentsMargins(2, 0, 2, 0); vl.setSpacing(5)
        self.lbl_vram = QLabel(tr("VRAM : calcul…", "VRAM: calculating…")); self.lbl_vram.setStyleSheet("font-weight:700; background:transparent;")
        vl.addWidget(self.lbl_vram)
        self.prog_vram = QProgressBar(); self.prog_vram.setRange(0, 100); self.prog_vram.setValue(0)
        self.prog_vram.setTextVisible(False); self.prog_vram.setFixedHeight(10)
        vl.addWidget(self.prog_vram)
        rl.addWidget(vbox)
        rl.addWidget(self.stack, 1)
        root.addWidget(right, 1)

        self._pages = {
            "net": self._page_network(),
            "data": self._page_datasets(), "train": self._page_train(),
            "deg": self._deg_page(),
            "adv": self._page_advanced(),
            "verify": self._page_verify(),
            "pipeline": self._page_pipeline(),
        }
        for _label, key in _NAV:
            self.stack.addWidget(self._pages[key])
        self._show("net")

        # navpanel / lossbox / lossredux / aiCard backgrounds are driven by the global
        # theme QSS (objectNames) so they adapt to dark/light. Only structural rules here.
        # #nav buttons are styled by the global theme QSS (theme-aware text + accent).
        self.setStyleSheet("""
            QFrame#sidebar { background: transparent; }
            QLabel#h2 { background: transparent; }
        """)

        self._on_engine_change()
        self._schedule_vram()
        self._apply_field_tooltips()

    def _apply_field_tooltips(self):
        """Restore the rich per-field tooltips from descriptions.py (FR/EN, formatted
        with ↑/↓ colors) — was the bulk of the CTk app's tooltips, missing in the port."""
        from qt_ui.i18n import tip_for
        for key, w in self.fields.items():
            try:
                t = tip_for(key)
                if t and hasattr(w, "setToolTip"):
                    w.setToolTip(t)
            except Exception:
                pass

    # ── small helpers ───────────────────────────────────────────────────────────
    def _scroll(self, inner: QWidget) -> QScrollArea:
        s = QScrollArea(); s.setWidgetResizable(True); s.setFrameShape(QFrame.NoFrame)
        s.setWidget(inner); return s

    def _header(self, layout, text):
        h = QLabel(text); h.setObjectName("h2"); layout.addWidget(h)

    def _entry(self, grid, r, c, key, label, default=""):
        grid.addWidget(QLabel(label + " :"), r, c)
        e = QLineEdit(str(default)); self.fields[key] = e
        e.textChanged.connect(self._schedule_vram)
        grid.addWidget(e, r, c + 1)

    def _combo(self, grid, r, c, key, label, values, default):
        grid.addWidget(QLabel(label + " :"), r, c)
        cb = QComboBox(); cb.addItems(values); cb.setCurrentText(default)
        self.fields[key] = cb
        cb.currentIndexChanged.connect(self._schedule_vram)
        grid.addWidget(cb, r, c + 1); return cb

    def _browse_start_dir(self, ed, key):
        """Where a path picker should open: the folder already typed, else a sensible
        dataset default derived from the field key (train vs val)."""
        cur = (ed.text() or "").strip()
        if cur:
            d = cur if os.path.isdir(cur) else os.path.dirname(cur)
            if d and os.path.isdir(d):
                return d
        try:
            from src.core.engine_paths import base_engine_path
            sub = "val" if "val" in key.lower() else "train"
            cand = os.path.join(base_engine_path(), "datasets", sub)
            if os.path.isdir(cand):
                return cand
            ds = os.path.join(base_engine_path(), "datasets")
            return ds if os.path.isdir(ds) else ""
        except Exception:
            return ""

    def _path(self, grid, r, key, label, pick_file=False, flt="Tous (*.*)", tooltip=""):
        lb = QLabel(label + " :")
        if tooltip:
            lb.setToolTip(tooltip)
        grid.addWidget(lb, r, 0)
        e = QLineEdit(); self.fields[key] = e
        if tooltip:
            e.setToolTip(tooltip)
        # restore last-used path so dataset/pretrain paths persist across app launches
        _saved = str(self.settings.get(f"cfg_path_{key}", "") or "")
        if _saved:
            e.setText(_saved)
        e.editingFinished.connect(lambda ed=e, k=key: self.settings.set(f"cfg_path_{k}", ed.text()))
        grid.addWidget(e, r, 1)
        b = QPushButton("…"); b.setFixedWidth(40)
        if tooltip:
            b.setToolTip(tooltip)

        def _pick(_=False, ed=e, k=key, pf=pick_file):
            if pf:
                chosen = QFileDialog.getOpenFileName(self, tr("Fichier", "File"), self._browse_start_dir(ed, k), flt)[0]
            else:
                chosen = QFileDialog.getExistingDirectory(self, tr("Dossier", "Folder"), self._browse_start_dir(ed, k))
            if chosen:
                ed.setText(chosen)
                self.settings.set(f"cfg_path_{k}", chosen)
        b.clicked.connect(_pick)
        grid.addWidget(b, r, 2)

    def _aug(self, layout, key, label, tooltip=""):
        """Checkbox + probability slider + value (MixUp/CutMix/ResizeMix/CutBlur)."""
        row = QHBoxLayout()
        chk = QCheckBox(label); chk.setMinimumWidth(110); self.fields[key] = chk
        if tooltip:
            chk.setToolTip(tooltip)
        row.addWidget(chk)
        sld = LabeledSlider("", 0.0, 1.0, 0.15, 0.05, "{:.2f}")
        if tooltip:
            sld.setToolTip(tooltip)
        self.fields[f"{key}_prob"] = sld
        row.addWidget(sld, 1)
        layout.addLayout(row)

    # ── pages ───────────────────────────────────────────────────────────────────
    def _general_rows(self, v):
        """Les 3 rangées 'Paramètres Généraux' (fusionnées dans Architecture)."""
        self._header(v, tr("Paramètres Généraux", "General Settings"))

        # Rangée Moteur | Scale | Seed | Déterministe — label colle son champ (4px), air entre groupes
        r1 = QHBoxLayout(); r1.setSpacing(4); v.addLayout(r1)
        r1.addWidget(QLabel(tr("Moteur :", "Engine:"))); r1.addSpacing(2)
        self.engine_combo = QComboBox(); self.engine_combo.addItems(["TraiNNer-Redux", "NeoSR"])
        self.engine_combo.setCurrentText(self.settings.get("engine", "TraiNNer-Redux"))
        self.engine_combo.setFixedWidth(150); self.fields["engine"] = self.engine_combo
        self.engine_combo.currentIndexChanged.connect(self._on_engine_change)
        r1.addWidget(self.engine_combo); r1.addSpacing(26)
        r1.addWidget(QLabel(tr("Scale :", "Scale:"))); r1.addSpacing(2)
        scale = QComboBox(); scale.addItems(["1", "2", "3", "4", "8"]); scale.setCurrentText("4")
        scale.setFixedWidth(64); self.fields["scale"] = scale
        scale.currentIndexChanged.connect(self._schedule_vram)
        r1.addWidget(scale); r1.addSpacing(26)
        r1.addWidget(QLabel(tr("Seed :", "Seed:"))); r1.addSpacing(2)
        seed = QLineEdit("10"); seed.setFixedWidth(70); self.fields["manual_seed"] = seed
        r1.addWidget(seed); r1.addSpacing(26)
        self.fields["deterministic"] = QCheckBox(tr("Déterministe", "Deterministic"))
        r1.addWidget(self.fields["deterministic"]); r1.addStretch(1)

        # Rangée Nom — pleine largeur
        r2 = QHBoxLayout(); r2.setSpacing(4); v.addLayout(r2)
        r2.addWidget(QLabel(tr("Nom Expérience :", "Experiment Name:"))); r2.addSpacing(2)
        name = QLineEdit("experiment"); self.fields["name"] = name
        name.textChanged.connect(self._schedule_vram)
        r2.addWidget(name, 1)

        # Rangée Dossier — pleine largeur + bouton
        r3 = QHBoxLayout(); r3.setSpacing(4); v.addLayout(r3)
        r3.addWidget(QLabel(tr("Dossier Expériences :", "Experiments Folder:"))); r3.addSpacing(2)
        exp = QLineEdit(); self.fields["custom_exp_path"] = exp
        exp.setToolTip(tr("Dossier où le moteur écrit checkpoints/logs/TensorBoard pour cette expérience.\nLaisser vide = dossier « experiments » par défaut du moteur.",
                          "Folder where the engine writes checkpoints/logs/TensorBoard for this experiment.\nLeave empty = the engine's default “experiments” folder."))
        r3.addWidget(exp, 1)
        bexp = QPushButton("…"); bexp.setFixedWidth(40)
        bexp.clicked.connect(lambda _=False: exp.setText(
            QFileDialog.getExistingDirectory(self, tr("Dossier", "Folder")) or exp.text()))
        r3.addWidget(bexp)
        hint = QLabel(tr("Si laissé vide : dossier « experiments » par défaut du moteur choisi.",
                         "If left empty: the chosen engine's default “experiments” folder is used."))
        hint.setStyleSheet("color:#888; font-size:11px;"); hint.setWordWrap(True)
        v.addWidget(hint)

    def _page_network(self) -> QWidget:
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(6, 6, 6, 6); v.setSpacing(14)
        # Paramètres Généraux (fusionnés ici depuis l'ancien onglet Général)
        self._general_rows(v)
        v.addWidget(hsep())
        self._header(v, tr("Générateur (Modèle)", "Generator (Model)"))
        top = QHBoxLayout(); v.addLayout(top)

        # left: family + arch combos (label hugs its combo, air between groups)
        left = QVBoxLayout(); left.setSpacing(3); top.addLayout(left)
        left.addWidget(QLabel(tr("Famille :", "Family:")))
        self.fam_combo = QComboBox(); self.fam_combo.addItem(tr("TOUTES", "ALL")); self.fam_combo.setFixedWidth(220)
        self.fam_combo.setToolTip(tr("Filtre la liste Architecture par catégorie (transformer, léger, CNN…).\n« TOUTES » affiche toutes les archs disponibles pour le moteur choisi.",
                                     "Filters the Architecture list by category (transformer, lightweight, CNN…).\n“ALL” shows every arch available for the chosen engine."))
        self.fam_combo.currentTextChanged.connect(self._filter_archs)
        left.addWidget(self.fam_combo)
        left.addSpacing(14)
        left.addWidget(QLabel(tr("Architecture :", "Architecture:")))
        self.arch_combo = QComboBox(); self.fields["arch"] = self.arch_combo; self.arch_combo.setFixedWidth(220)
        self.arch_combo.currentTextChanged.connect(self._on_arch_change)
        left.addWidget(self.arch_combo)
        left.addStretch(1)
        top.addSpacing(16)

        # right: capability bars
        self.perf = _PerfBars(); top.addWidget(self.perf, 1)

        # dynamic per-arch params box (blue border, like CTk frame_dynamic_g)
        self.param_box = QGroupBox(""); self.param_box.setObjectName("boxG")
        self.param_grid = QGridLayout(self.param_box)
        v.addWidget(self.param_box)

        self._header(v, tr("Discriminateur (GAN)", "Discriminator (GAN)"))
        ganhdr = QHBoxLayout(); v.addLayout(ganhdr)
        self.fields["use_gan"] = QCheckBox(tr("Activer GAN", "Enable GAN"))
        self.fields["use_gan"].stateChanged.connect(self._on_gan_toggle)
        ganhdr.addWidget(self.fields["use_gan"]); ganhdr.addSpacing(28)
        self.fields["adaptive_d"] = QCheckBox("Adaptive D (GAN)")
        self.fields["adaptive_d"].setToolTip(tr(
            "Ajuste dynamiquement la fréquence d'entraînement du discriminateur selon l'équilibre G/D "
            "(évite qu'un des deux domine trop l'autre).",
            "Dynamically adjusts how often the discriminator trains based on the G/D balance "
            "(prevents either side from overpowering the other)."))
        ganhdr.addWidget(self.fields["adaptive_d"]); ganhdr.addStretch(1)

        self.gan_opts = QWidget(); go = QVBoxLayout(self.gan_opts); go.setContentsMargins(0, 6, 0, 0); go.setSpacing(8)
        row = QHBoxLayout(); row.setSpacing(4); go.addLayout(row)

        def _pair(label, field, gap=22):
            lbl = QLabel(label); row.addWidget(lbl)
            row.addSpacing(2); row.addWidget(field); row.addSpacing(gap)

        self.disc_combo = QComboBox(); self.disc_combo.setFixedWidth(150)
        self.fields["net_d_type"] = self.disc_combo
        self.disc_combo.currentTextChanged.connect(self._on_disc_change)
        _pair("Type :", self.disc_combo)
        self.gan_type_combo = QComboBox(); self.gan_type_combo.addItems(["vanilla", "wgan", "hinge", "lsgan"])
        self.gan_type_combo.setCurrentText("vanilla"); self.gan_type_combo.setFixedWidth(90)
        self.fields["gan_type"] = self.gan_type_combo
        self.gan_type_combo.currentIndexChanged.connect(self._schedule_vram)
        _pair("Mode :", self.gan_type_combo)
        rl = QLineEdit("1.0"); rl.setFixedWidth(52); self.fields["real_label_val"] = rl
        _pair("Real Label :", rl)
        fl = QLineEdit("0.0"); fl.setFixedWidth(52); self.fields["fake_label_val"] = fl
        _pair("Fake Label :", fl)
        self.fields["dyn_gan_weight"] = _LogSlider("Loss Weight GAN :", 0.05, width=200)
        self.fields["dyn_gan_weight"].valueChangedF.connect(self._schedule_vram)
        row.addWidget(self.fields["dyn_gan_weight"])
        row.addStretch(1)

        # Début différé du GAN (Redux uniquement, patchs moteur gan_start_iter /
        # gan_auto_* / gan_restart_lr) : itération fixe ou auto sur plateau du PSNR.
        self.gan_start_box = QWidget(); row2 = QHBoxLayout(self.gan_start_box)
        row2.setContentsMargins(0, 0, 0, 0); row2.setSpacing(4); go.addWidget(self.gan_start_box)

        def _pair2(label, field, tip, gap=22):
            lbl = QLabel(label); lbl.setToolTip(tip); field.setToolTip(tip)
            row2.addWidget(lbl); row2.addSpacing(2); row2.addWidget(field); row2.addSpacing(gap)

        gs = QLineEdit("0"); gs.setFixedWidth(70); self.fields["gan_start_iter"] = gs
        _pair2(tr("Début GAN (iter) :", "GAN start (iter):"), gs, tr(
            "Itération à laquelle le GAN démarre.\n"
            "Avant : le générateur apprend seulement avec ses autres losses (pas de loss GAN, "
            "discriminateur(s) en pause).\n"
            "- 0 : GAN actif dès le début (comportement standard).\n"
            "- 30000 à 100000 : conseillé pour un modèle entraîné de zéro.\n"
            "En mode Auto : démarrage au plus tard (0 = pas de limite).",
            "Iteration at which the GAN starts.\n"
            "Before it, the generator learns from its other losses only (no GAN loss, "
            "discriminator(s) paused).\n"
            "- 0: GAN active from the start (standard behaviour).\n"
            "- 30000 to 100000: recommended when training from scratch.\n"
            "In Auto mode: latest start (0 = no limit)."))
        ga = QCheckBox(tr("Auto (plateau PSNR)", "Auto (PSNR plateau)")); self.fields["gan_auto_start"] = ga
        ga.setToolTip(tr(
            "Démarre le GAN tout seul quand le PSNR de validation ne monte plus.\n"
            "Nécessite la validation avec la métrique PSNR. Sans elle, le GAN démarre à "
            "« Début GAN » (ou à « Iter min »).",
            "Starts the GAN by itself when the validation PSNR stops improving.\n"
            "Needs validation with the PSNR metric. Without it, the GAN starts at "
            "“GAN start” (or at “Min iter”)."))
        row2.addWidget(ga); row2.addSpacing(16)
        gp = QLineEdit("3"); gp.setFixedWidth(40); self.fields["gan_auto_patience"] = gp
        _pair2(tr("Patience :", "Patience:"), gp, tr(
            "Nombre de validations de suite sans gain de PSNR avant de démarrer le GAN.\n"
            "Avec Val Freq 2500 et une patience de 3 : 7500 itérations sans progrès.",
            "Number of validations in a row without a PSNR gain before the GAN starts.\n"
            "With Val Freq 2500 and a patience of 3: 7500 iterations without progress."), gap=16)
        gd = QLineEdit("0.05"); gd.setFixedWidth(50); self.fields["gan_auto_min_delta"] = gd
        _pair2(tr("Gain min (dB) :", "Min gain (dB):"), gd, tr(
            "Plus petit gain de PSNR compté comme un progrès. Conseillé : 0.03 à 0.1.",
            "Smallest PSNR gain counted as progress. Recommended: 0.03 to 0.1."), gap=16)
        gm = QLineEdit("20000"); gm.setFixedWidth(70); self.fields["gan_auto_min_iter"] = gm
        _pair2(tr("Iter min :", "Min iter:"), gm, tr(
            "Le GAN ne démarre jamais avant cette itération (évite un faux plateau au début).",
            "The GAN never starts before this iteration (avoids a false plateau at the start)."))
        gr = QCheckBox(tr("Relancer le planning LR", "Restart LR schedule")); self.fields["gan_restart_lr"] = gr
        gr.setChecked(True)
        gr.setToolTip(tr(
            "Au démarrage du GAN, le LR revient à sa valeur de départ et les milestones "
            "comptent à partir du démarrage du GAN, comme un 2e training séparé.\n"
            "Ex. : milestones 75000 / 112500 = 75000 et 112500 itérations après le début du GAN.\n"
            "Itérations totales = phase sans GAN + phase GAN.\n"
            "Décoché : les milestones comptent depuis l'itération 0.",
            "When the GAN starts, the LR goes back to its initial value and the milestones "
            "count from the GAN start, like a separate 2nd training run.\n"
            "E.g. milestones 75000 / 112500 = 75000 and 112500 iterations after the GAN start.\n"
            "Total iterations = phase without GAN + GAN phase.\n"
            "Unchecked: the milestones count from iteration 0."))
        row2.addWidget(gr); row2.addStretch(1)
        for w in (gs, gp, gd, gm):
            w.textChanged.connect(self._refresh_gan_start)
        ga.stateChanged.connect(self._refresh_gan_start)
        self._refresh_gan_start()

        go.addSpacing(14)
        self.disc_box = QGroupBox(""); self.disc_box.setObjectName("boxD")
        self.disc_grid = QGridLayout(self.disc_box); self.disc_grid.setContentsMargins(10, 14, 10, 10)
        go.addWidget(self.disc_box)

        # --- 2e discriminateur optionnel (support générique DualFormer-style :
        # n'importe quelle paire parmi les 10 discriminateurs Redux, pas
        # seulement spatial+spectral). Uniquement pris en charge côté moteur
        # Redux (network_d2/gan_opt2/optim_d2) -- masqué pour NeoSR.
        go.addSpacing(14)
        self.disc2_box = QGroupBox(""); self.disc2_box.setObjectName("boxD")
        d2v = QVBoxLayout(self.disc2_box); d2v.setContentsMargins(10, 14, 10, 10); d2v.setSpacing(6)
        d2hdr = QHBoxLayout(); d2v.addLayout(d2hdr)
        self.fields["use_disc2"] = QCheckBox(tr("Activer un 2e discriminateur", "Enable a 2nd discriminator"))
        self.fields["use_disc2"].setToolTip(tr(
            "Entraîne un second discriminateur SIMULTANÉMENT au premier, chacun avec sa "
            "propre loss et son propre optimiseur (network_d2/gan_opt2/optim_d2 côté "
            "traiNNer-redux). Recommandé par le papier DualFormer pour combiner un "
            "discriminateur spatial et un discriminateur spectral, mais fonctionne avec "
            "n'importe quelle paire des 10 discriminateurs Redux disponibles.\n\n"
            "Règle générale : associer des discriminateurs de domaines DIFFÉRENTS "
            "(spatial / fréquence / perceptuel pré-entraîné) apporte de la valeur ; "
            "associer deux discriminateurs du MÊME domaine coûte cher en VRAM/temps "
            "pour peu de gain (redondance).\n\n"
            "✅ Prouvé (papier DualFormer) : Spatial (VGG-Style/UNet-SN/PatchGAN-SN/"
            "MultiPatchGAN-SN/DUNet/MetaGAN2/DocRes-Disc) + Spectral (DualFormer) — "
            "domaines complémentaires (pixel vs fréquence), le spectral traque les "
            "artefacts périodiques/damier.\n"
            "🟡 Raisonnement solide mais non testé ici : Spatial + Projected GAN — combo "
            "répandu en littérature GAN générale (features pré-entraînées captent "
            "l'incohérence sémantique/perceptuelle).\n"
            "🟡 Expérimental, aucun précédent connu : Spectral + Projected GAN.\n"
            "❌ Déconseillé : NeXtSRGAN + (VGG-Style/DocRes-Disc/UNet-SN/PatchGAN-SN/"
            "MultiPatchGAN-SN/MetaGAN2/DUNet) — NeXtSRGAN est architecturalement quasi "
            "identique à VGG-Style, donc combiner les deux double le coût VRAM/temps "
            "pour un gain quasi nul (redondance de domaine).",
            "Trains a second discriminator SIMULTANEOUSLY with the first, each with its "
            "own loss and optimizer (network_d2/gan_opt2/optim_d2 in traiNNer-redux). "
            "Recommended by the DualFormer paper to combine a spatial and a spectral "
            "discriminator, but works with any pair among the 10 available Redux "
            "discriminators.\n\n"
            "General rule: pairing discriminators from DIFFERENT domains (spatial / "
            "frequency / pretrained-perceptual) adds value; pairing two discriminators "
            "from the SAME domain costs VRAM/time for little gain (redundancy).\n\n"
            "✅ Proven (DualFormer paper): Spatial (VGG-Style/UNet-SN/PatchGAN-SN/"
            "MultiPatchGAN-SN/DUNet/MetaGAN2/DocRes-Disc) + Spectral (DualFormer) — "
            "complementary domains (pixel vs frequency), spectral tracks periodic/"
            "checkerboard artifacts.\n"
            "🟡 Sound reasoning but untested here: Spatial + Projected GAN — common combo "
            "in general GAN literature (pretrained features catch semantic/perceptual "
            "incoherence).\n"
            "🟡 Experimental, no known precedent: Spectral + Projected GAN.\n"
            "❌ Not recommended: NeXtSRGAN + (VGG-Style/DocRes-Disc/UNet-SN/PatchGAN-SN/"
            "MultiPatchGAN-SN/MetaGAN2/DUNet) — NeXtSRGAN is architecturally near-"
            "identical to VGG-Style, so combining both doubles VRAM/time cost for "
            "near-zero gain (domain redundancy)."))
        self.fields["use_disc2"].stateChanged.connect(self._on_disc2_toggle)
        d2hdr.addWidget(self.fields["use_disc2"]); d2hdr.addStretch(1)

        self.disc2_opts = QWidget(); d2o = QVBoxLayout(self.disc2_opts); d2o.setContentsMargins(0, 4, 0, 0); d2o.setSpacing(6)
        row2 = QHBoxLayout(); row2.setSpacing(4); d2o.addLayout(row2)
        row2.addWidget(QLabel("Type D2 :"))
        self.disc2_combo = QComboBox(); self.disc2_combo.setFixedWidth(150)
        self.fields["net_d2_type"] = self.disc2_combo
        self.disc2_combo.currentTextChanged.connect(self._on_disc2_change)
        row2.addSpacing(2); row2.addWidget(self.disc2_combo); row2.addSpacing(22)
        self.fields["gan_weight_d2"] = _LogSlider("Loss Weight D2 :", 0.05, width=200)
        self.fields["gan_weight_d2"].valueChangedF.connect(self._schedule_vram)
        row2.addWidget(self.fields["gan_weight_d2"])
        row2.addStretch(1)

        # Avertissement ACTIF (pas juste tooltip) — visible en permanence sous
        # les sélecteurs dès qu'une paire D1/D2 redondante est choisie.
        self.lbl_disc_pair_warn = QLabel("")
        self.lbl_disc_pair_warn.setWordWrap(True)
        self.lbl_disc_pair_warn.setVisible(False)
        d2o.addWidget(self.lbl_disc_pair_warn)

        d2o.addSpacing(10)
        self.disc2_grid_box = QGroupBox(""); self.disc2_grid_box.setObjectName("boxD")
        self.disc2_grid = QGridLayout(self.disc2_grid_box); self.disc2_grid.setContentsMargins(10, 14, 10, 10)
        d2o.addWidget(self.disc2_grid_box)
        d2v.addWidget(self.disc2_opts)
        self.disc2_opts.setVisible(False)
        go.addWidget(self.disc2_box)

        v.addWidget(self.gan_opts)
        self.gan_opts.setVisible(False)
        # Logo moteur (NeoSR / traiNNer-redux) — sous le cadre GAN
        v.addSpacing(18)
        self.banner = QLabel(); self.banner.setAlignment(Qt.AlignCenter); self.banner.setMinimumHeight(120)
        v.addWidget(self.banner)
        v.addStretch(1)
        return self._scroll(inner)

    def _page_datasets(self) -> QWidget:
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(6, 6, 6, 6); v.setSpacing(12)
        self._header(v, tr("Datasets & Chemins", "Datasets & Paths"))
        # Mode — combo compact collé à son label
        mrow = QHBoxLayout(); mrow.setSpacing(4); v.addLayout(mrow)
        mrow.addWidget(QLabel(tr("Mode de Dataset :", "Dataset Mode:"))); mrow.addSpacing(2)
        mcb = QComboBox(); mcb.addItems(["otf", "bicubic", "paired"]); mcb.setCurrentText("otf")
        mcb.setFixedWidth(140); self.fields["dataset_mode"] = mcb
        self._dataset_mode_tooltip = tr(
            "otf : dégradations réalistes générées à la volée depuis les HQ seules (recommandé, pas besoin de LQ).\n"
            "bicubic : LQ = HQ réduite en bicubique seule (simple, moins réaliste).\n"
            "paired : utilise des paires HQ/LQ déjà fabriquées (dossiers Train/Val LQ requis).\n"
            "sequence : n'apparaît que pour les archs vidéo/temporelles (tscunet, temporalspan, "
            "temporalspanv2, tfdat) — utilise PairedVideoDataset : clips de plusieurs frames "
            "consécutives (voir « Frames par clip »), GT = frame centrale du clip uniquement, "
            "LQ ET HQ requis (dossiers Train/Val LQ + HQ), convention de nommage "
            "<préfixe_scène>_XXXX.png obligatoire.",
            "otf: realistic degradations generated on the fly from HQ only (recommended, no LQ needed).\n"
            "bicubic: LQ = HQ downscaled with bicubic only (simple, less realistic).\n"
            "paired: uses already-made HQ/LQ pairs (Train/Val LQ folders required).\n"
            "sequence: only appears for video/temporal archs (tscunet, temporalspan, temporalspanv2, "
            "tfdat) — uses PairedVideoDataset: clips of several consecutive frames (see \"Frames per "
            "clip\"), GT = clip's center frame only, LQ AND HQ both required (Train/Val LQ + HQ "
            "folders), <scene_prefix>_XXXX.png naming convention mandatory.")
        mcb.setToolTip(self._dataset_mode_tooltip)
        mcb.currentIndexChanged.connect(self._on_dataset_mode_change)
        mcb.currentIndexChanged.connect(self._schedule_vram)
        self.dataset_mode_combo = mcb
        mrow.addWidget(mcb)
        # "Séquence vidéo" (PairedVideoDataset) — only meaningful for temporal archs
        # (tscunet/temporalspan/temporalspanv2/tfdat). Added/removed from the combo
        # dynamically in _on_arch_change so non-temporal archs can't select it.
        mrow.addSpacing(16)
        self.clip_size_label = QLabel(tr("Frames par clip :", "Frames per clip:"))
        mrow.addWidget(self.clip_size_label)
        mrow.addSpacing(2)
        cse = QLineEdit("5"); cse.setFixedWidth(50)
        self.fields["clip_size"] = cse
        cse.textChanged.connect(self._schedule_vram)
        cse.setToolTip(tr(
            "Nombre de frames consécutives par clip (doit être impair, doit correspondre au nombre de "
            "frames attendu par le réseau — 5 par défaut pour ces archs).\n"
            "Convention de nommage requise par PairedVideoDataset/SingleVideoDataset :\n"
            "  <prefixe_scène>_XXXX.png  (ex: scene1_0001.png, scene1_0002.png, ...)\n"
            "Le préfixe = tout ce qui précède le premier '_' dans le nom de fichier ; les frames d'une même "
            "scène doivent partager le même préfixe et être triables par ordre alphabétique (numérotation "
            "zero-padded). Un dossier Train LQ + un dossier Train HQ avec les MÊMES noms de fichiers pour "
            "la frame centrale de chaque clip.",
            "Number of consecutive frames per clip (must be odd, must match the frame count expected by "
            "the network — 5 by default for these archs).\n"
            "Naming convention required by PairedVideoDataset/SingleVideoDataset:\n"
            "  <scene_prefix>_XXXX.png  (e.g. scene1_0001.png, scene1_0002.png, ...)\n"
            "The prefix = everything before the first '_' in the filename; frames from the same scene must "
            "share the same prefix and sort alphabetically (zero-padded numbering). One Train LQ folder + "
            "one Train HQ folder with the SAME filenames for each clip's center frame."))
        self.clip_size_edit = cse
        mrow.addWidget(cse)
        mrow.addStretch(1)
        self._on_dataset_mode_change()  # hidden by default (dataset_mode starts at "otf")
        v.addLayout(self._grid_paths())

        self._header(v, "Augmentations (Train)")
        arow = QHBoxLayout()
        self.fields["use_hflip"] = QCheckBox(tr("HFlip (Miroir H)", "HFlip (H Mirror)")); self.fields["use_hflip"].setChecked(True)
        self.fields["use_hflip"].setToolTip(tr(
            "Retourne aléatoirement l'image horizontalement pendant l'entraînement (augmentation classique, sans risque).",
            "Randomly flips the image horizontally during training (classic, safe augmentation)."))
        self.fields["use_rot"] = QCheckBox(tr("Rot90 (Rotations)", "Rot90 (Rotations)")); self.fields["use_rot"].setChecked(True)
        self.fields["use_rot"].setToolTip(tr(
            "Applique des rotations aléatoires par pas de 90° (0/90/180/270) — multiplie la variété du dataset sans déformer l'image.",
            "Applies random 90° rotations (0/90/180/270) — multiplies dataset variety without distorting the image."))
        self.fields["use_shuffle"] = QCheckBox(tr("Shuffle (Mélanger)", "Shuffle")); self.fields["use_shuffle"].setChecked(True)
        self.fields["use_shuffle"].setToolTip(tr(
            "Mélange l'ordre des images à chaque epoch pour éviter que le modèle mémorise l'ordre du dataset.",
            "Shuffles the image order every epoch so the model doesn't memorize the dataset order."))
        arow.addWidget(self.fields["use_hflip"]); arow.addWidget(self.fields["use_rot"])
        arow.addWidget(self.fields["use_shuffle"]); arow.addStretch(1)
        v.addLayout(arow)
        self._aug(v, "use_mixup", "MixUp", tr(
            "Mélange deux images (HQ+LQ) linéairement selon un ratio aléatoire. Réduit l'overfitting, régularise le modèle.\nProbabilité = fréquence d'application par batch.",
            "Linearly blends two image pairs (HQ+LQ) with a random ratio. Reduces overfitting, regularizes the model.\nProbability = how often it's applied per batch."))
        self._aug(v, "use_cutmix", "CutMix", tr(
            "Colle un patch rectangulaire d'une image sur une autre. Force le modèle à généraliser sur des textures locales variées.",
            "Pastes a rectangular patch from one image onto another. Forces the model to generalize across varied local textures."))
        self._aug(v, "use_resizemix", "ResizeMix", tr(
            "Comme CutMix mais redimensionne le patch collé — introduit de la variation d'échelle en plus de la variation spatiale.",
            "Like CutMix but resizes the pasted patch — adds scale variation on top of spatial variation."))
        self._aug(v, "use_cutblur", "CutBlur", tr(
            "Floute un patch rectangulaire de l'image HQ pour simuler des zones de netteté variable (utile pour la SR réaliste, évite le sur-affinage uniforme).",
            "Blurs a rectangular patch of the HQ image to simulate uneven sharpness (useful for realistic SR, avoids uniform over-sharpening)."))
        drow = QHBoxLayout()
        self.fields["aug_debug"] = QCheckBox("Aug. Debug"); drow.addWidget(self.fields["aug_debug"])
        self.fields["aug_debug"].setToolTip(tr(
            "Sauvegarde les paires augmentées (MoA) dans un dossier debug pour vérifier les augmentations.",
            "Saves augmented (MoA) pairs to a debug folder to verify augmentations."))
        drow.addWidget(QLabel(tr("Limite :", "Limit:")))
        self.fields["aug_debug_limit"] = QLineEdit("100"); self.fields["aug_debug_limit"].setFixedWidth(70)
        drow.addWidget(self.fields["aug_debug_limit"]); drow.addSpacing(20)
        # OTF debug: redux saves the degraded LQ each iter → lets you SEE OTF + custom degs.
        self.fields["high_order_degradations_debug"] = QCheckBox("OTF Debug")
        self.fields["high_order_degradations_debug"].setToolTip(tr(
            "Sauvegarde les LQ dégradées à la volée (OTF + custom) dans experiments/<nom>/debug — "
            "pour VÉRIFIER visuellement que les dégradations sont bien appliquées.",
            "Saves the on-the-fly degraded LQ (OTF + custom) to experiments/<name>/debug — "
            "to visually VERIFY degradations are applied."))
        drow.addWidget(self.fields["high_order_degradations_debug"])
        drow.addWidget(QLabel(tr("Limite :", "Limit:")))
        self.fields["high_order_degradations_debug_limit"] = QLineEdit("100")
        self.fields["high_order_degradations_debug_limit"].setFixedWidth(70)
        drow.addWidget(self.fields["high_order_degradations_debug_limit"]); drow.addStretch(1)
        v.addLayout(drow)

        self._header(v, tr("Reprise / Pretrain", "Resume / Pretrain"))
        gr = QGridLayout(); v.addLayout(gr)
        self._path(gr, 0, "resume_state", "Resume State (.state)", pick_file=True, flt=tr("State (*.state);;Tous (*.*)", "State (*.state);;All (*.*)"),
                   tooltip=tr("Reprend un entraînement interrompu depuis un fichier .state (optimiseur + itération) — laisse vide pour repartir de zéro.",
                              "Resumes an interrupted training run from a .state file (optimizer + iteration) — leave empty to start fresh."))
        self._path(gr, 1, "pretrain_model", "Pretrain Model (.pth)", pick_file=True,
                   flt=tr("Modèles (*.pth *.safetensors);;Tous (*.*)", "Models (*.pth *.safetensors);;All (*.*)"),
                   tooltip=tr("Charge les poids d'un modèle existant comme point de départ (fine-tuning) au lieu d'entraîner from scratch.",
                              "Loads an existing model's weights as a starting point (fine-tuning) instead of training from scratch."))
        self._path(gr, 2, "pretrain_model_d", "Pretrain D (.pth)", pick_file=True,
                   flt=tr("Modèles (*.pth *.safetensors);;Tous (*.*)", "Models (*.pth *.safetensors);;All (*.*)"),
                   tooltip=tr("Charge les poids d'un discriminateur existant (GAN uniquement) — laisse vide pour repartir d'un D aléatoire.",
                              "Loads an existing discriminator's weights (GAN only) — leave empty to start from a random D."))
        self._path(gr, 3, "pretrain_model_d2", "Pretrain D2 (.pth)", pick_file=True,
                   flt=tr("Modèles (*.pth *.safetensors);;Tous (*.*)", "Models (*.pth *.safetensors);;All (*.*)"),
                   tooltip=tr("traiNNer-redux : poids du 2e discriminateur (network_d2), utilisés seulement si "
                              "« Activer un 2e discriminateur » est coché — laisse vide pour un D2 aléatoire.",
                              "traiNNer-redux: weights of the second discriminator (network_d2), used only when "
                              "\"Enable a 2nd discriminator\" is ticked — leave empty for a random D2."))

        self._header(v, tr("ECO (Empirical Centroid-oriented Optimization)",
                           "ECO (Empirical Centroid-oriented Optimization)"))
        ge = QGridLayout(); v.addLayout(ge)
        self.fields["use_eco"] = QCheckBox(tr("Activer ECO", "Enable ECO")); ge.addWidget(self.fields["use_eco"], 0, 0, 1, 3)
        self._path(ge, 1, "pretrain_eco", "Teacher ECO (.pth)", pick_file=True,
                   flt=tr("Modèles (*.pth *.safetensors);;Tous (*.*)", "Models (*.pth *.safetensors);;All (*.*)"),
                   tooltip=tr("traiNNer-redux : poids du réseau « teacher » (même architecture que le modèle "
                              "entraîné, ex. un modèle déjà entraîné). Requis si ECO est activé.\n"
                              "NeoSR : pas de teacher séparé (champ grisé).",
                              "traiNNer-redux: weights of the 'teacher' network (same architecture as the "
                              "trained model, e.g. an already trained model). Required when ECO is on.\n"
                              "NeoSR: no separate teacher (field greyed out)."))
        ge2 = QGridLayout(); v.addLayout(ge2)
        self._combo(ge2, 0, 0, "eco_mode_kind", tr("Mode ECO (Redux)", "ECO mode (Redux)"), ["full", "hr_only"], "full")
        self._entry(ge2, 0, 2, "eco_end_ratio", tr("Fin du mélange (Redux)", "Blend end (Redux)"), "0.75")
        self._entry(ge2, 1, 0, "eco_init", tr("Début (NeoSR, it)", "Start (NeoSR, it)"), "15000")
        self._entry(ge2, 1, 2, "eco_iters", tr("Durée (NeoSR, it)", "Length (NeoSR, it)"), "80000")
        self._combo(ge2, 2, 0, "eco_schedule", tr("Courbe (NeoSR)", "Curve (NeoSR)"), ["sigmoid", "linear"], "sigmoid")
        v.addStretch(1)
        return self._scroll(inner)

    def _grid_paths(self) -> QGridLayout:
        g = QGridLayout()
        self._path(g, 0, "dataroot_gt", "Train HQ (GT)",
                   tooltip=tr("Dossier des images haute qualité (Ground Truth) utilisées pour l'entraînement. Obligatoire.",
                              "Folder of high-quality (Ground Truth) images used for training. Required."))
        self._path(g, 1, "val_gt", "Val HQ (GT)",
                   tooltip=tr("Dossier HQ utilisé pour la validation (calcul PSNR/SSIM pendant l'entraînement). Doit être séparé du dataset d'entraînement.",
                              "HQ folder used for validation (PSNR/SSIM computed during training). Should be separate from the training dataset."))
        self._path(g, 2, "val_lq", "Val LQ (Optionnel)",
                   tooltip=tr("Dossier LQ apparié à Val HQ (mode paired uniquement). En mode otf/bicubic, laisse vide — les LQ de validation sont générées automatiquement.",
                              "LQ folder paired with Val HQ (paired mode only). In otf/bicubic mode, leave empty — validation LQs are generated automatically."))
        self._path(g, 3, "dataroot_lq", "Train LQ (Optionnel)",
                   tooltip=tr("Dossier LQ apparié à Train HQ (mode paired uniquement). En mode otf/bicubic, laisse vide — les LQ d'entraînement sont générées à la volée.",
                              "LQ folder paired with Train HQ (paired mode only). In otf/bicubic mode, leave empty — training LQs are generated on the fly."))
        return g

    def _field_grid(self, grid, r, c, key, label, default=""):
        """Label-over-entry cell (CTk add_param_grid)."""
        cell = QWidget(); cl = QVBoxLayout(cell); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(2)
        cl.addWidget(QLabel(label))
        e = QLineEdit(str(default)); self.fields[key] = e
        e.textChanged.connect(self._schedule_vram); cl.addWidget(e)
        grid.addWidget(cell, r, c)

    def _combo_grid(self, grid, r, c, key, label, values, default):
        cell = QWidget(); cl = QVBoxLayout(cell); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(2)
        cl.addWidget(QLabel(label))
        cb = QComboBox(); cb.addItems(values); cb.setCurrentText(default); self.fields[key] = cb
        cb.currentIndexChanged.connect(self._schedule_vram); cl.addWidget(cb)
        grid.addWidget(cell, r, c)

    def _page_train(self) -> QWidget:
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(6, 6, 6, 6); v.setSpacing(14)
        self._header(v, tr("Hyperparamètres", "Hyperparameters"))
        # Batch / Accumulate / Patch — sliders + value box (3 colonnes)
        g = QGridLayout(); g.setHorizontalSpacing(22); v.addLayout(g)
        self.fields["batch_size"] = _SliderBox("Batch Size", 1, 64, 1, 4)
        self.fields["batch_size"].valueChangedF.connect(self._schedule_vram)
        g.addWidget(self.fields["batch_size"], 0, 0)
        self.fields["accumulate"] = _SliderBox("Accumulate", 1, 32, 1, 1)
        g.addWidget(self.fields["accumulate"], 0, 1)
        self.fields["patch_size"] = _SliderBox("Patch Size", 32, 512, 8, 64)
        self.fields["patch_size"].valueChangedF.connect(self._schedule_vram)
        g.addWidget(self.fields["patch_size"], 0, 2)
        for c in range(3):
            g.setColumnStretch(c, 1)
        # Total Iter / Warmup Iter / Warmup Steps
        g2 = QGridLayout(); g2.setHorizontalSpacing(22); v.addLayout(g2)
        self._field_grid(g2, 0, 0, "total_iter", "Total Iter", "150000")
        self._field_grid(g2, 0, 1, "warmup_iter", "Warmup Iter", "-1")
        self._field_grid(g2, 0, 2, "warmup_steps", "Warmup Steps", "-1")
        for c in range(3):
            g2.setColumnStretch(c, 1)
        # Optimiseur | Scheduler | Sched. Free
        orow = QHBoxLayout(); orow.setSpacing(4); v.addLayout(orow)
        orow.addWidget(QLabel(tr("Optimiseur G :", "G Optimizer:"))); orow.addSpacing(2)
        oc = QComboBox(); oc.setFixedWidth(140); self.fields["optim_g"] = self.optim_combo = oc
        oc.currentIndexChanged.connect(self._schedule_vram)
        orow.addWidget(oc); orow.addSpacing(26)
        orow.addWidget(QLabel(tr("Scheduler :", "Scheduler:"))); orow.addSpacing(2)
        sc = QComboBox(); sc.setFixedWidth(180); self.fields["scheduler"] = self.sched_combo = sc
        orow.addWidget(sc); orow.addSpacing(26)
        self.fields["sched_free"] = QCheckBox("Sched. Free"); orow.addWidget(self.fields["sched_free"])
        orow.addStretch(1)
        # LR Générateur | LR Discriminateur — log sliders avec emoji + value box
        lrrow = QHBoxLayout(); lrrow.setSpacing(20); v.addLayout(lrrow)
        self.fields["lr"] = _LogSlider(tr("LR Générateur :", "Generator LR:"), 5e-5, 1e-6, 1e-3, "{:.0e}", _LR_THR, width=160)
        lrrow.addWidget(self.fields["lr"], 1)
        self.fields["lr_d"] = _LogSlider(tr("LR Discriminateur :", "Discriminator LR:"), 5e-5, 1e-6, 1e-3, "{:.0e}", _LR_THR, width=160)
        lrrow.addWidget(self.fields["lr_d"], 1)

        self._header(v, tr("Paramètres Scheduler", "Scheduler Parameters"))
        gs = QGridLayout(); gs.setHorizontalSpacing(22); v.addLayout(gs)
        self._field_grid(gs, 0, 0, "milestones", "Milestones (Multi)", "75000, 112500")
        self._field_grid(gs, 0, 1, "gamma", "Gamma (Multi)", "0.5")
        self._field_grid(gs, 1, 0, "t_max", "T_max (Cosine)", "300000")
        self._field_grid(gs, 1, 1, "eta_min", "Eta Min (Cosine)", "1e-7")
        for c in range(2):
            gs.setColumnStretch(c, 1)

        self._header(v, "Log & Validation")
        gl = QGridLayout(); gl.setHorizontalSpacing(22); v.addLayout(gl)
        self._field_grid(gl, 0, 0, "val_freq", "Val Freq", "5000")
        self._field_grid(gl, 0, 1, "save_freq", "Save Freq", "5000")
        self._field_grid(gl, 0, 2, "print_freq", "Print Freq", "100")
        self._combo_grid(gl, 1, 0, "save_img", "Save Img", ["true", "false"], "true")
        self._field_grid(gl, 1, 1, "tile_val", "Tile (Val)", "200")
        self._field_grid(gl, 1, 2, "tile_pad", "Tile Pad", "32")
        for c in range(3):
            gl.setColumnStretch(c, 1)
        mrow = QHBoxLayout(); mrow.addWidget(QLabel(tr("Métriques :", "Metrics:"))); mrow.addSpacing(8)
        self._metric_checks_neosr_only = []
        for key, label, on in (("metric_psnr", "PSNR", True), ("metric_ssim", "SSIM", True),
                               ("metric_dists", "DISTS", False), ("metric_topiq", "TOPIQ", False)):
            chk = QCheckBox(label); chk.setChecked(on); self.fields[key] = chk; mrow.addWidget(chk)
            if key in ("metric_dists", "metric_topiq"):
                # NeoSR-only : Redux ignore ces métriques (val.metrics figé à psnr/ssim
                # côté moteur) — griser plutôt que laisser une case sans effet.
                chk.setToolTip(tr("NeoSR uniquement (ignoré par TraiNNer-Redux).",
                                   "NeoSR only (ignored by TraiNNer-Redux)."))
                self._metric_checks_neosr_only.append(chk)
        # USRS texture metrics (Redux): texture / edges / grain MAE + HF detail / sharpness
        chk = QCheckBox(tr("Texture / Netteté (USRS)", "Texture / Sharpness (USRS)")); chk.setChecked(True)
        chk.setToolTip(tr(
            "TraiNNer-Redux : ajoute à chaque validation 5 mesures qui séparent ce que le PSNR "
            "mélange — Texture (erreur sur les zones texturées), Contours (erreur sur les traits), "
            "Grain (erreur sur les aplats : grain/bruit inventé), Détail HF (les fins détails sont-ils "
            "au bon endroit) et Netteté HF (plus doux ou plus net que la GT).\n"
            "Affichées dans les cartes de l'onglet Run, la boîte de validation, TensorBoard et la "
            "fenêtre Métriques (cases à cocher). Coût : ~0,2 s par image de validation, sur le CPU.",
            "TraiNNer-Redux: adds 5 measures at each validation that split what PSNR averages away — "
            "Texture (error on textured areas), Edges (error on lines), Grain (error on flat areas: "
            "invented grain/noise), HF detail (is fine detail in the right place) and HF sharpness "
            "(softer or crisper than the GT).\n"
            "Shown in the Run tab cards, the validation box, TensorBoard and the Metrics window "
            "(check boxes). Cost: ~0.2 s per validation image, on the CPU."))
        self.fields["metric_texture"] = chk; mrow.addWidget(chk)
        mrow.addStretch(1); v.addLayout(mrow)
        v.addStretch(1)
        return self._scroll(inner)

    # ── losses helpers ──────────────────────────────────────────────────────────
    def _open_popup(self, title, specs):
        """Build a (non-modal) options dialog and register its widgets in fields."""
        dlg = QDialog(self); dlg.setWindowTitle(title)
        v = QVBoxLayout(dlg)
        for key, label, kind, default, opts in specs:
            r = QHBoxLayout(); r.addWidget(QLabel(label + " :"))
            if kind == "combo":
                w = QComboBox(); w.addItems(opts); w.setCurrentText(str(default))
            elif kind == "check":
                w = QCheckBox(); w.setChecked(bool(default))
            else:
                w = QLineEdit(str(default)); w.setFixedWidth(90)
            self.fields[key] = w; r.addWidget(w); r.addStretch(1); v.addLayout(r)
        b = QPushButton(tr("Fermer", "Close")); b.clicked.connect(dlg.accept); v.addWidget(b)
        return dlg

    def _gear(self, title, specs, label="⚙️", width=34) -> QPushButton:
        dlg = self._open_popup(title, specs)   # registers fields immediately
        btn = QPushButton(label); btn.setObjectName("ghost")
        btn.setFixedWidth(width); btn.setFixedHeight(26)
        btn.setStyleSheet('font-family:"Segoe UI Emoji"; font-size:13px; padding:0 4px;')
        btn.clicked.connect(lambda: (dlg.show(), dlg.raise_(), dlg.activateWindow()))
        return btn

    def _loss_row(self, layout, key, label, weight, extras=(), checked=False, lw=140):
        row = QHBoxLayout(); row.setSpacing(6)
        ws = []  # all widgets of this row (for engine-based greying)

        def _add(w):
            row.addWidget(w); ws.append(w); return w

        # Badge slot in front of every row (empty when the loss uses no pretrained
        # network) so the Weight columns stay aligned.
        _add(self._loss_badge(key))
        chk = QCheckBox(label); chk.setMinimumWidth(lw); chk.setChecked(checked)
        chk.stateChanged.connect(self._schedule_vram); self.fields[key] = chk
        _add(chk)
        _add(QLabel("Weight:"))
        e = QLineEdit(str(weight)); e.setFixedWidth(48); self.fields[f"weight_{key}"] = e
        _add(e)
        for ex in extras:
            kind = ex[0]
            if kind == "combo":
                _, k, lab, choices, dflt = ex
                if lab:
                    _add(QLabel(lab + ":"))
                cb = QComboBox(); cb.addItems(choices); cb.setCurrentText(dflt); cb.setFixedWidth(118)
                self.fields[k] = cb; _add(cb)
            elif kind == "entry":
                _, k, lab, dflt, w = ex
                if lab:
                    _add(QLabel(lab + ":"))
                en = QLineEdit(str(dflt)); en.setFixedWidth(w); self.fields[k] = en; _add(en)
            elif kind == "check":
                _, k, lab, dflt = ex
                c = QCheckBox(lab); c.setChecked(bool(dflt)); self.fields[k] = c; _add(c)
            elif kind == "gear":
                _, title, specs = ex
                _add(self._gear(title, specs))
            elif kind == "vgg":
                specs = [(f"percep_vgg_{l}", l, "entry", dw, None) for l, dw in _VGG_LAYERS]
                _add(self._gear(tr("Poids couches VGG  (0.0 = désactivée)", "VGG layer weights  (0.0 = disabled)"), specs,
                                label="▾ Couches VGG", width=150))
            elif kind == "path":
                k, ph = ex[1], ex[2]
                flt = ex[3] if len(ex) > 3 else tr("PTH (*.pth);;Tous (*.*)", "PTH (*.pth);;All (*.*)")
                en = QLineEdit(); en.setPlaceholderText(ph); en.setFixedWidth(110); self.fields[k] = en
                _add(en)
                bb = QPushButton("📂"); bb.setObjectName("ghost"); bb.setFixedWidth(40); bb.setFixedHeight(26)
                bb.setStyleSheet('font-family:"Segoe UI Emoji"; font-size:13px; padding:0;')
                bb.clicked.connect(lambda _=False, ed=en, f=flt: ed.setText(
                    QFileDialog.getOpenFileName(self, tr("Poids", "Weights"), "", f)[0] or ed.text()))
                _add(bb)
        row.addStretch(0)
        layout.addLayout(row)
        self._loss_rows[key] = ws

    @staticmethod
    def _loss_badge(key: str) -> QLabel:
        """Fixed-width badge naming what the loss's pretrained network was trained on."""
        from src.core.descriptions import LOSS_NET_BADGE
        kind = LOSS_NET_BADGE.get(key)
        lab = QLabel(); lab.setFixedWidth(52); lab.setAlignment(Qt.AlignCenter)
        if kind == "anime":
            lab.setText("Anime")
            lab.setToolTip(tr("Spécialisée anime : son réseau a été entraîné sur de l'anime / du dessin.",
                              "Anime-specific: its network was trained on anime / drawings."))
            lab.setStyleSheet("background:#8e44ad; color:white; border-radius:8px; "
                              "font-size:10px; font-weight:700; padding:1px 0;")
        elif kind == "photo":
            lab.setText("Photo")
            lab.setToolTip(tr("Réseau entraîné sur des photos (ImageNet) : marche sur l'anime, "
                              "mais ses repères sont photographiques.",
                              "Network trained on photos (ImageNet): works on anime, "
                              "but its references are photographic."))
            lab.setStyleSheet("background:#2e86de; color:white; border-radius:8px; "
                              "font-size:10px; font-weight:700; padding:1px 0;")
        else:
            lab.setStyleSheet("background:transparent;")
        return lab

    # ── phase losses (PSNR / GAN profile, traiNNer-redux) ─────────────────────────
    def _lphase_row(self) -> QHBoxLayout:
        self._lphase_profiles = [None, None]
        self._lphase_active = 0
        row = QHBoxLayout(); row.setSpacing(6)
        self.chk_lphase = QCheckBox(tr("Profil GAN différent", "Separate GAN profile"))
        self.chk_lphase.setToolTip(tr(
            "Deux profils de losses : un pour la phase PSNR (avant le GAN), un pour la phase "
            "GAN. Après le démarrage du GAN (auto ou itération fixe), le moteur passe des poids "
            "du profil PSNR à ceux du profil GAN par une rampe linéaire sur « Transition » "
            "itérations.\n"
            "Une loss décochée dans un profil a un poids 0 dans cette phase : elle n'est pas "
            "calculée. Toutes les losses des deux profils sont chargées dès le départ (la VRAM "
            "affichée est celle du run complet, pas de surprise au passage en GAN).\n"
            "Exemple : losses structurelles seules en phase PSNR, losses perceptuelles "
            "ajoutées en phase GAN. traiNNer-redux uniquement.",
            "Two loss profiles: one for the PSNR phase (before the GAN), one for the GAN "
            "phase. After the GAN start (auto or fixed iteration), the engine moves from the "
            "PSNR profile weights to the GAN profile weights with a linear ramp over "
            "'Transition' iterations.\n"
            "A loss unchecked in a profile has weight 0 in that phase: it is not computed. "
            "Every loss of both profiles is loaded from the start (the VRAM shown is the full "
            "run's, no surprise at the GAN start).\n"
            "Example: structural losses only in the PSNR phase, perceptual losses added in "
            "the GAN phase. traiNNer-redux only."))
        row.addWidget(self.chk_lphase)
        lab = QLabel(tr("Profil édité :", "Edited profile:")); row.addWidget(lab)
        self.lphase_combo = QComboBox()
        self.lphase_combo.addItems([tr("Phase PSNR", "PSNR phase"), tr("Phase GAN", "GAN phase")])
        self.lphase_combo.setFixedWidth(130); self.lphase_combo.setEnabled(False)
        row.addWidget(self.lphase_combo)
        row.addWidget(QLabel(tr("Transition (it) :", "Transition (it):")))
        self.lphase_ramp = QLineEdit("5000"); self.lphase_ramp.setFixedWidth(70)
        self.lphase_ramp.setToolTip(tr(
            "Itérations de la rampe des poids PSNR → GAN après le démarrage du GAN "
            "(0 = bascule immédiate ; une rampe courte évite d'ajouter un 2e choc à celui du "
            "discriminateur).",
            "Iterations of the PSNR → GAN weight ramp after the GAN start (0 = immediate "
            "switch; a short ramp avoids a second shock on top of the discriminator's)."))
        self.lphase_ramp.setEnabled(False)
        row.addWidget(self.lphase_ramp)
        self.lphase_info = QLabel(""); self.lphase_info.setStyleSheet("color:#888; font-size:11px;")
        row.addWidget(self.lphase_info)
        row.addStretch(1)
        self.chk_lphase.toggled.connect(self._on_lphase_toggled)
        self.lphase_combo.currentIndexChanged.connect(self._on_lphase_switch)
        self._lphase_lock = [self.chk_lphase, lab]
        return row

    def _lphase_keys(self):
        return [k for k in getattr(self, "_loss_rows", {}) if "gan" not in k]

    def _lphase_snapshot(self) -> dict:
        """{loss key: [checked, weight text]} of the Losses panel as displayed."""
        snap = {}
        for k in self._lphase_keys():
            c, w = self.fields.get(k), self.fields.get(f"weight_{k}")
            if isinstance(c, QCheckBox) and isinstance(w, QLineEdit):
                snap[k] = [bool(c.isChecked()), w.text()]
        return snap

    def _lphase_restore(self, snap):
        for k, (on, wt) in (snap or {}).items():
            c, w = self.fields.get(k), self.fields.get(f"weight_{k}")
            if isinstance(c, QCheckBox):
                c.setChecked(bool(on))
            if isinstance(w, QLineEdit):
                w.setText(str(wt))
        self._schedule_vram()

    def _on_lphase_toggled(self, on):
        self.lphase_combo.setEnabled(on); self.lphase_ramp.setEnabled(on)
        if not on and self._lphase_active == 1:
            self.lphase_combo.setCurrentIndex(0)
        self._update_lphase_info()
        self._schedule_vram()

    def _on_lphase_switch(self, idx):
        if idx == self._lphase_active:
            return
        self._lphase_profiles[self._lphase_active] = self._lphase_snapshot()
        if self._lphase_profiles[idx] is None:
            # first visit: start the GAN profile as a copy of the PSNR one
            self._lphase_profiles[idx] = self._lphase_snapshot()
        self._lphase_restore(self._lphase_profiles[idx])
        self._lphase_active = idx
        self._update_lphase_info()

    def _update_lphase_info(self):
        if not self.chk_lphase.isChecked():
            self.lphase_info.setText("")
        elif self._lphase_active == 1:
            self.lphase_info.setText(tr("Édition du profil GAN", "Editing the GAN profile"))
            self.lphase_info.setStyleSheet("color:#17a2b8; font-size:11px; font-weight:700;")
        else:
            self.lphase_info.setText(tr("Édition du profil PSNR", "Editing the PSNR profile"))
            self.lphase_info.setStyleSheet("color:#2ecc71; font-size:11px; font-weight:700;")

    def lphase_profiles(self):
        """(on, ramp, PSNR snapshot, GAN snapshot), whichever profile is displayed."""
        cur = self._lphase_snapshot()
        prof = list(self._lphase_profiles)
        prof[self._lphase_active] = cur
        p1 = prof[0] if prof[0] is not None else cur
        p2 = prof[1] if prof[1] is not None else p1
        return self.chk_lphase.isChecked(), self.lphase_ramp.text(), p1, p2

    def reset_lphase(self):
        """Back to a single loss profile (called before loading a config)."""
        if self._lphase_active == 1:
            self.lphase_combo.blockSignals(True); self.lphase_combo.setCurrentIndex(0)
            self.lphase_combo.blockSignals(False)
            if self._lphase_profiles[0] is not None:
                self._lphase_restore(self._lphase_profiles[0])
            self._lphase_active = 0
        self._lphase_profiles = [None, None]
        self.chk_lphase.setChecked(False)
        self.lphase_ramp.setText("5000")

    def apply_lphase(self, p2: dict, ramp=None, on=True):
        """GAN profile from a snapshot; the PSNR profile is what the panel shows."""
        self.reset_lphase()
        self._lphase_profiles = [self._lphase_snapshot(), dict(p2)]
        if ramp not in (None, ""):
            self.lphase_ramp.setText(str(ramp))
        self.chk_lphase.setChecked(bool(on))

    def _lphase_from_flat(self, flat2: dict) -> dict:
        snap = {}
        for k in self._lphase_keys():
            w = self.fields.get(f"weight_{k}")
            wt = flat2.get(f"weight_{k}")
            snap[k] = [str(flat2.get(k, "false")).lower() in ("true", "1", "on", "yes"),
                       str(wt) if wt is not None else (w.text() if isinstance(w, QLineEdit) else "1.0")]
        return snap

    def _loss_panel(self) -> QWidget:
        self._loss_rows = {}
        box = QFrame(); box.setObjectName("lossbox")
        h = QHBoxLayout(box); h.setContentsMargins(10, 8, 10, 8); h.setSpacing(12)
        left = QVBoxLayout(); left.setSpacing(2); h.addLayout(left)
        legend = QLabel(tr("Anime = réseau entraîné sur de l'anime  ·  Photo = réseau entraîné sur des photos  ·  "
                           "sans badge = calcul sans réseau appris (universel)",
                           "Anime = network trained on anime  ·  Photo = network trained on photos  ·  "
                           "no badge = no learned network (universal)"))
        legend.setStyleSheet("color:#8a8f98; font-size:11px; background:transparent;")
        left.addWidget(legend)
        left.addLayout(self._lphase_row())
        # ── colonne gauche : losses standards ──
        self._loss_row(left, "loss_pixel", "Pixel L1 / Charbonnier", 1.0, checked=True, extras=[
            ("combo", "pixel_reduction", "Mode", ["mean", "sum"], "mean"),
            ("combo", "pixel_criterion", "Type", ["L1Loss", "MSELoss", "HuberLoss", "charbonnierloss", "chc_loss"], "L1Loss"),
        ])
        self._loss_row(left, "loss_percep", "Perceptual (VGG)", 1.0, checked=True, extras=[
            ("combo", "percep_criterion", "", ["huber", "l1", "l2", "charbonnier", "chc_loss"], "huber"),
            ("vgg",),
        ])
        self._loss_row(left, "loss_fdl", "FDL (Freq)", 1.0, extras=[
            ("combo", "fdl_model", "", ["vgg", "dinov2", "resnet", "effnet"], "vgg")])
        self._loss_row(left, "loss_mssim", "MS-SSIM", 1.0, extras=[
            ("gear", "MS-SSIM Options (NeoSR)", [
                ("mssim_window_size", "Window Size", "entry", "11", None),
                ("mssim_sigma", "Sigma", "entry", "1.5", None),
                ("mssim_k1", "K1", "entry", "0.01", None),
                ("mssim_k2", "K2", "entry", "0.03", None)])])
        self._loss_row(left, "loss_dists", "DISTS", 1.0)
        self._loss_row(left, "loss_msswd", "MS-SWD", 1.0)
        self._loss_row(left, "loss_ff", "Focal Freq", 1.0, extras=[("entry", "ff_alpha", "α", "1.0", 50)])
        self._loss_row(left, "loss_consistency", "Consistency", 1.0, extras=[
            ("gear", "Consistency Options (NeoSR)", [
                ("consistency_blur", "Blur", "check", True, None),
                ("consistency_cosim", "CoSim", "check", True, None),
                ("consistency_saturation", "Saturation", "check", True, None),
                ("consistency_brightness", "Brightness", "check", True, None)])])
        self._loss_row(left, "loss_ncc", "NCC", 1.0)
        self._loss_row(left, "loss_kl", "KL Divergence", 1.0)
        self._loss_row(left, "loss_ldl", "LDL", 1.0, extras=[
            ("combo", "ldl_criterion", "", ["charbonnier", "l1", "l2", "huber"], "charbonnier"),
            ("entry", "ldl_ksize", "ksize", "7", 45)])
        self._loss_row(left, "loss_edge", "Edge Loss", 0.05)
        self._loss_row(left, "loss_wavelet", "Wavelet Guided", 1.0, extras=[
            ("entry", "wavelet_init", "Init", "10000", 55)])
        _w = self.fields.get("loss_wavelet")
        if _w is not None:
            _w.setToolTip(((_w.toolTip() + "\n\n") if _w.toolTip() else "")
                          + tr("NeoSR : nécessite le GAN (le moteur refuse sinon).",
                               "NeoSR: requires the GAN (the engine refuses otherwise)."))
        left.addStretch(1)

        # ── colonne droite : Redux uniquement ──
        rcol = QFrame(); rcol.setObjectName("lossredux"); rcol.setFixedWidth(720)
        right = QVBoxLayout(rcol); right.setSpacing(2)
        hr = QLabel(tr("⚡ Redux uniquement", "⚡ Redux only")); hr.setStyleSheet("color:#3b8ed0; font-weight:700; background:transparent;")
        right.addWidget(hr)
        self._loss_row(right, "loss_hsluv", "HSLuv", 1.0, lw=120, extras=[
            ("entry", "hsluv_hue_weight", "H", "0.33", 48),
            ("entry", "hsluv_sat_weight", "S", "0.33", 48),
            ("entry", "hsluv_lum_weight", "L", "0.33", 48)])
        self._loss_row(right, "loss_cosim", "Cosim", 1.0, lw=120, extras=[
            ("entry", "cosim_lambda", "λ", "5", 42)])
        self._loss_row(right, "loss_color", "Color", 1.0, lw=120, extras=[
            ("combo", "color_criterion", "", ["l1", "l2", "charbonnier"], "l1")])
        self._loss_row(right, "loss_gv", "Grad Variance", 1.0, lw=120, extras=[
            ("entry", "gv_patch_size", "P", "16", 38),
            ("combo", "gv_criterion", "", ["charbonnier", "l1", "l2"], "charbonnier")])
        self._loss_row(right, "loss_luma", "Luma", 1.0, lw=120, extras=[
            ("combo", "luma_criterion", "", ["l1", "l2", "charbonnier"], "l1")])
        self._loss_row(right, "loss_contextual", "Contextual", 1.0, lw=120, extras=[
            ("combo", "ctx_distance_type", "", ["cosine", "l1", "l2"], "cosine"),
            ("entry", "ctx_band_width", "BW", "0.5", 42)])
        self._loss_row(right, "loss_linedistiller", "LineDistiller", 1.0, lw=120, extras=[
            ("combo", "linedistiller_criterion", "", ["l1", "charbonnier"], "l1")])
        self._loss_row(right, "loss_percep_fp16", "Percep. FP16", 1.0, lw=120, extras=[
            ("combo", "percep_fp16_criterion", "", ["charbonnier", "l1", "fd", "pd", "fd+l1", "pd+l1"], "charbonnier")])
        self._loss_row(right, "loss_aesop", "AESOP", 1.0, lw=120, extras=[
            ("combo", "aesop_criterion", "", ["charbonnier", "l1"], "charbonnier"),
            ("path", "aesop_ae_path", tr("autoencodeur .safetensors", "autoencoder .safetensors"),
             "Safetensors (*.safetensors);;All (*.*)")])
        # AESOPLoss (traiNNer/losses/aesop_loss.py) also needs decoder_opt, scale and
        # pretrain_network_ae: the generator writes the trained arch as decoder_opt (the
        # paper's recipe: the AE decoder starts from the PSNR model of the same arch), the
        # config scale, and this path. Without a path the config is refused at save time
        # instead of crashing the launch (the row used to be greyed out here, but the
        # Redux engine switch re-enabled it, so it could be ticked and crash).
        self._loss_row(right, "loss_bcedice", "BCE+Dice", 1.0, lw=120, extras=[
            ("entry", "bcedice_weight_bce", "BCE", "1.0", 42),
            ("entry", "bcedice_weight_dice", "Dice", "1.0", 42)])
        self._loss_row(right, "loss_percep_anime", "Percep. Anime", 1.0, lw=120, extras=[
            ("combo", "percep_anime_criterion", "", ["l1", "charbonnier"], "l1")])
        self._loss_row(right, "loss_spark", "SparK (Percep)", 0.2, lw=120, extras=[
            ("combo", "spark_criterion", "", ["fd", "charbonnier"], "fd"),
            ("path", "spark_path", "epoch290.pth (opt)"),
            ("entry", "spark_max_score", "Clamp", "50.0", 52)])
        right.addStretch(1)
        h.addWidget(rcol)
        h.addStretch(1)
        return box

    def _sync_bf16(self, *_):
        """BF16 is the AMP dtype (both engines); NeoSR refuses it without AMP."""
        amp, bf = self.fields.get("use_amp"), self.fields.get("bfloat16")
        if amp is None or bf is None:
            return
        bf.setEnabled(amp.isChecked())
        if not amp.isChecked():
            bf.setChecked(False)
        bf.setToolTip(tr("Type de l'AMP : bfloat16 au lieu de float16 (RTX 30+). Cochez d'abord AMP.",
                         "AMP dtype: bfloat16 instead of float16 (RTX 30+). Tick AMP first."))

    def _page_advanced(self) -> QWidget:
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(6, 6, 6, 6); v.setSpacing(10)
        self._header(v, tr("Optimisations Système", "System Optimizations"))
        for k, txt in (("use_amp", "Use AMP (FP16)"), ("bfloat16", "Use BF16 (RTX 30+)"),
                       ("fast_matmul", "Fast MatMul (TF32)"), ("compile", "Torch Compile (Linux)"),
                       ("channels_last", "Channels Last (memory format)"),
                       ("match_lq_colors", "Match Colors (LQ->GT)")):
            self.fields[k] = QCheckBox(txt)
            if k == "use_amp":
                self.fields[k].stateChanged.connect(self._schedule_vram)
                self.fields[k].toggled.connect(self._sync_bf16)
            if k == "compile" and platform.system() == "Windows":
                # torch.compile relies on Triton as its backend compiler, which has
                # no official/complete Windows support (torch._dynamo.is_dynamo_supported()
                # is False on Windows) -> training crashes with NotImplementedError as
                # soon as a compiled step actually runs. Can't be fixed engine-side
                # (real PyTorch/Triton platform limitation), so disable the option here
                # instead of letting the user enable something that will crash training.
                self.fields[k].setChecked(False)
                self.fields[k].setEnabled(False)
                self.fields[k].setToolTip(tr(
                    "Torch Compile n'est pas supporté sous Windows (Triton n'a pas de "
                    "support Windows complet) — activez-le uniquement sous Linux.",
                    "Torch Compile is not supported on Windows (Triton has no complete "
                    "Windows support) — only enable this on Linux."
                ))
            v.addWidget(self.fields[k])
        self._sync_bf16()
        gc = QHBoxLayout(); v.addLayout(gc)
        self.fields["grad_clip"] = QCheckBox("Gradient Clipping"); gc.addWidget(self.fields["grad_clip"])
        gc.addWidget(QLabel(tr("Max Norm :", "Max Norm:")))
        self.fields["grad_clip_max_norm"] = QLineEdit("100"); self.fields["grad_clip_max_norm"].setFixedWidth(70)
        gc.addWidget(self.fields["grad_clip_max_norm"]); gc.addStretch(1)
        g2 = QHBoxLayout(); v.addLayout(g2)
        g2.addWidget(QLabel(tr("Num GPU :", "GPU Num:")))
        self.fields["num_gpu"] = QLineEdit("1"); self.fields["num_gpu"].setFixedWidth(50)
        g2.addWidget(self.fields["num_gpu"]); g2.addSpacing(20)
        g2.addWidget(QLabel(tr("Mode SAM :", "SAM Mode:")))
        self.fields["sam"] = QComboBox(); self.fields["sam"].addItems(["none", "sam", "fsam"])
        self.fields["sam"].setFixedWidth(90); g2.addWidget(self.fields["sam"])
        g2.addWidget(QLabel(tr("Init :", "Init:")))
        self.fields["sam_init"] = QLineEdit("-1"); self.fields["sam_init"].setFixedWidth(50)
        g2.addWidget(self.fields["sam_init"]); g2.addSpacing(20)
        g2.addWidget(QLabel(tr("EMA :", "EMA:")))
        self.fields["ema"] = QLineEdit("0.999"); self.fields["ema"].setFixedWidth(70)
        g2.addWidget(self.fields["ema"]); g2.addStretch(1)

        self._header(v, tr("Fonctions de Perte (Losses)", "Loss Functions"))
        v.addWidget(self._loss_panel())

        self._header(v, "Monitoring")
        m1 = QHBoxLayout(); v.addLayout(m1)
        self.fields["auto_tensorboard"] = QCheckBox(tr("Serveur TB", "TB Server")); m1.addWidget(self.fields["auto_tensorboard"])
        btn_tb = QPushButton("♻"); btn_tb.setObjectName("warn")
        btn_tb.setFixedWidth(38); btn_tb.setFixedHeight(26)
        btn_tb.setStyleSheet('font-family:"Segoe UI Emoji","Segoe UI Symbol"; font-size:14px; padding:0;')
        btn_tb.setToolTip(tr("Relancer TensorBoard", "Restart TensorBoard"))
        btn_tb.clicked.connect(self._restart_tb); m1.addWidget(btn_tb)
        m1.addWidget(QLabel(tr("Port :", "Port:")))
        self.fields["port_tb"] = QLineEdit("6006"); self.fields["port_tb"].setFixedWidth(60)
        m1.addWidget(self.fields["port_tb"]); m1.addStretch(1)
        self.fields["use_tb_logger"] = QCheckBox(tr("Générer Logs", "Generate Logs")); self.fields["use_tb_logger"].setChecked(True)
        v.addWidget(self.fields["use_tb_logger"])
        self.fields["auto_ngrok"] = QCheckBox(tr("Tunnel Ngrok", "Ngrok Tunnel")); v.addWidget(self.fields["auto_ngrok"])

        self._header(v, tr("Chargement & Perceptual", "Loading & Perceptual"))
        ld = QHBoxLayout(); v.addLayout(ld)
        ld.addWidget(QLabel(tr("Prefetch Mode :", "Prefetch Mode:")))
        self.fields["prefetch_mode"] = QComboBox(); self.fields["prefetch_mode"].addItems(["cuda", "cpu", "None"])
        self.fields["prefetch_mode"].setFixedWidth(100); ld.addWidget(self.fields["prefetch_mode"])
        ld.addSpacing(20); ld.addWidget(QLabel(tr("Workers/GPU :", "Workers/GPU:")))
        self.fields["num_worker_per_gpu"] = QLineEdit("4"); self.fields["num_worker_per_gpu"].setFixedWidth(50)
        ld.addWidget(self.fields["num_worker_per_gpu"]); ld.addStretch(1)
        v.addStretch(1)
        return self._scroll(inner)

    def _deg_page(self) -> QWidget:
        try:
            from qt_ui.tabs.degradations import DegradationsTab
            self.deg_tab = DegradationsTab(self.settings)
            return self.deg_tab
        except Exception as e:
            return self._placeholder(tr("Dégradations", "Degradations"), tr(f"(indisponible : {e})", f"(unavailable: {e})"))

    def _config_text(self) -> str:
        import json
        return json.dumps(self._data(), indent=2, ensure_ascii=False, default=str)

    # provider display → (settings key, [models])
    try:
        from src.core.ai_providers import AI_MODELS as _AI_MODELS
    except Exception:
        _AI_MODELS = {"OpenRouter (Gratuit)": ("api_key_openrouter", ["openrouter/free"])}

    def _ai_card(self, title, color):
        """Dark rounded panel with a colored bold title; returns the inner layout."""
        f = QFrame(); f.setObjectName("aiCard")  # bg from theme QSS (dark/light aware)
        cv = QVBoxLayout(f); cv.setContentsMargins(12, 10, 12, 10); cv.setSpacing(4)
        t = QLabel(title); t.setStyleSheet(f"color:{color}; font-weight:700; font-size:13px; background:transparent;")
        cv.addWidget(t)
        return f, cv

    def _page_verify(self) -> QWidget:
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(8)
        self._header(v, tr("Vérification de Configuration par IA", "AI Configuration Review"))
        sub = QLabel(tr("Envoyez votre configuration à une IA pour analyse avant l'entraînement.", "Send your configuration to an AI for analysis before training."))
        sub.setStyleSheet("color:#888;"); v.addWidget(sub)

        # ── Templates de Configuration ──
        f_tpl, tl = self._ai_card(tr("Templates de Configuration", "Configuration Templates"), "#9b59b6")
        d = QLabel(tr("Charge une configuration pré-réglée pour des cas d'usage typiques.", "Loads a preset configuration for typical use cases."))
        d.setStyleSheet("color:#888; background:transparent;"); tl.addWidget(d)
        row = QHBoxLayout(); lt = QLabel(tr("Template :", "Template:")); lt.setFixedWidth(120); lt.setStyleSheet("background:transparent;")
        row.addWidget(lt)
        self.tpl_choice = QComboBox()
        try:
            from src.core.config_templates import list_templates
            self.tpl_choice.addItems([tr("(aucun)", "(none)")] + list(list_templates()))
        except Exception:
            self.tpl_choice.addItem(tr("(aucun)", "(none)"))
        self.tpl_choice.setFixedWidth(260); row.addWidget(self.tpl_choice)
        btpl = QPushButton(tr("Charger Template", "Load Template"))
        btpl.setStyleSheet("background:#9b59b6; color:#fff; border:none; border-radius:6px; padding:6px 12px; font-weight:600;")
        btpl.clicked.connect(self._load_template); row.addWidget(btpl); row.addStretch(1)
        tl.addLayout(row); v.addWidget(f_tpl)

        # ── Configuration API ──
        f_api, al = self._ai_card(tr("Configuration API", "API Configuration"), "#3498db")
        r1 = QHBoxLayout(); l1 = QLabel(tr("Fournisseur AI :", "AI Provider:")); l1.setFixedWidth(120); l1.setStyleSheet("background:transparent;")
        r1.addWidget(l1)
        from qt_ui.widgets import KeyedCombo
        self.ai_provider = KeyedCombo(list(self._AI_MODELS.keys()))
        self.ai_provider.setFixedWidth(220); r1.addWidget(self.ai_provider); r1.addStretch(1); al.addLayout(r1)

        r2 = QHBoxLayout(); l2 = QLabel(tr("Clé API :", "API Key:")); l2.setFixedWidth(120); l2.setStyleSheet("background:transparent;")
        r2.addWidget(l2)
        self.ai_key = QLineEdit(); self.ai_key.setEchoMode(QLineEdit.Password); self.ai_key.setFixedWidth(360)
        r2.addWidget(self.ai_key)
        hint = QLabel(tr("(depuis Réglages)", "(from Settings)")); hint.setStyleSheet("color:#888; background:transparent;"); r2.addWidget(hint)
        r2.addStretch(1); al.addLayout(r2)

        r3 = QHBoxLayout(); l3 = QLabel(tr("Modèle :", "Model:")); l3.setFixedWidth(120); l3.setStyleSheet("background:transparent;")
        r3.addWidget(l3)
        self.ai_model = QComboBox(); self.ai_model.setEditable(True); self.ai_model.setFixedWidth(300)
        from qt_ui.widgets import combo_open_on_click
        combo_open_on_click(self.ai_model)  # click anywhere opens the list
        r3.addWidget(self.ai_model); r3.addStretch(1); al.addLayout(r3)

        self.ai_model_desc = QLabel(""); self.ai_model_desc.setWordWrap(True)
        self.ai_model_desc.setStyleSheet("color:#888; background:transparent;"); al.addWidget(self.ai_model_desc)

        r_test = QHBoxLayout()
        self.btn_test_api = QPushButton(tr("🔌 Tester la Connexion API", "🔌 Test API Connection"))
        self.btn_test_api.setStyleSheet('background:#16a085; color:#fff; border:none; border-radius:6px; padding:6px 12px; '
                                        'font-weight:600; font-family:"Segoe UI Emoji"; font-size:13px;')
        self.btn_test_api.clicked.connect(self._test_api); r_test.addWidget(self.btn_test_api)
        self.lbl_test_result = QLabel(""); self.lbl_test_result.setStyleSheet("color:#888; background:transparent;")
        r_test.addWidget(self.lbl_test_result, 1); al.addLayout(r_test)

        r_cache = QHBoxLayout()
        self.ai_use_cache = QCheckBox(tr("Utiliser le cache (éviter les appels redondants)", "Use cache (avoid redundant calls)")); self.ai_use_cache.setChecked(True)
        r_cache.addWidget(self.ai_use_cache); r_cache.addStretch(1)
        bclr = QPushButton(tr("Vider Cache", "Clear Cache")); bclr.setStyleSheet("background:#666; color:#fff; border:none; border-radius:6px; padding:6px 12px;")
        bclr.clicked.connect(self._clear_ai_cache); r_cache.addWidget(bclr); al.addLayout(r_cache)
        v.addWidget(f_api)

        # provider/model wiring (after widgets exist)
        self.ai_provider.keyChanged.connect(self._on_ai_provider)
        self.ai_model.currentTextChanged.connect(self._on_ai_model)
        self._on_ai_provider(self.ai_provider.currentText())

        # ── L'IA vérifiera ──
        f_sc, sl = self._ai_card(tr("L'IA vérifiera :", "The AI will check:"), "#e67e22")
        for c in [tr("Cohérence architecture / scale / batch / patch", "Architecture / scale / batch / patch consistency"),
                  tr("Compatibilité des losses avec le moteur", "Loss compatibility with the engine"),
                  tr("Recommandations LR / scheduler / optimiseur", "LR / scheduler / optimizer recommendations"),
                  tr("Estimation VRAM et risques de OOM", "VRAM estimation and OOM risks"),
                  tr("Bonnes pratiques PSNR vs GAN", "PSNR vs GAN best practices"),
                  tr("Erreurs courantes et pièges", "Common mistakes and pitfalls")]:
            lc = QLabel(f"  • {c}"); lc.setStyleSheet("color:#888; background:transparent;"); sl.addWidget(lc)
        v.addWidget(f_sc)

        # ── Mode d'envoi ──
        self.ai_send_mode = QComboBox()
        self.ai_send_mode.addItems([tr("Texte résumé (léger)", "Summary text (light)"), tr("Config complète (détaillé)", "Full config (detailed)")])
        self.ai_send_mode.setCurrentText(tr("Config complète (détaillé)", "Full config (detailed)"))
        self.ai_send_mode.setFixedWidth(250); v.addWidget(self.ai_send_mode)

        # ── Contexte supplémentaire ──
        f_ctx, cl = self._ai_card(tr("Contexte supplémentaire (optionnel)", "Additional context (optional)"), "#27ae60")
        def _ctx_combo(label, values):
            r = QHBoxLayout(); lb = QLabel(label); lb.setFixedWidth(140); lb.setStyleSheet("background:transparent;")
            r.addWidget(lb)
            cb = QComboBox(); cb.addItems(values); cb.setFixedWidth(220); r.addWidget(cb); r.addStretch(1)
            cl.addLayout(r); return cb
        self.ai_ctx_dataset = _ctx_combo(tr("Type de dataset :", "Dataset type:"),
            [tr("(auto-detect)", "(auto-detect)"), tr("Anime / Illustration", "Anime / Illustration"),
             tr("Photo réaliste", "Realistic photo"), tr("Texte / Document", "Text / Document"),
             tr("Médical / Scientifique", "Medical / Scientific"), tr("Jeux vidéo / Pixel art", "Video game / Pixel art"),
             tr("Mixte", "Mixed")])
        self.ai_ctx_goal = _ctx_combo(tr("But du modèle :", "Model goal:"),
            [tr("(non spécifié)", "(unspecified)"), tr("PSNR (fidélité pixel)", "PSNR (pixel fidelity)"),
             tr("GAN (qualité perçue)", "GAN (perceived quality)"), tr("PSNR puis GAN (pipeline)", "PSNR then GAN (pipeline)"),
             "Denoise", "Deblur", tr("Compression artifact", "Compression artifact"), "Debanding"])
        self.ai_ctx_time = _ctx_combo(tr("Temps alloué :", "Time allotted:"),
            [tr("(non spécifié)", "(unspecified)"), tr("< 2 heures", "< 2 hours"), tr("2-6 heures", "2-6 hours"),
             tr("6-24 heures", "6-24 hours"), tr("1-3 jours", "1-3 days"), tr("3-7 jours", "3-7 days"),
             tr("> 1 semaine", "> 1 week")])
        r_ds = QHBoxLayout(); lds = QLabel(tr("Taille dataset :", "Dataset size:")); lds.setFixedWidth(140); lds.setStyleSheet("background:transparent;")
        r_ds.addWidget(lds)
        self.ai_ctx_ds_size = QLineEdit(); self.ai_ctx_ds_size.setPlaceholderText(tr("ex: 5000 images, 50Go...", "e.g. 5000 images, 50GB...")); self.ai_ctx_ds_size.setFixedWidth(180)
        r_ds.addWidget(self.ai_ctx_ds_size)
        bauto = QPushButton(tr("Auto", "Auto")); bauto.setFixedWidth(55)
        bauto.setStyleSheet("background:#2c6e49; color:#fff; border:none; border-radius:6px; padding:4px;")
        bauto.clicked.connect(self._autodetect_ds_size); r_ds.addWidget(bauto); r_ds.addStretch(1); cl.addLayout(r_ds)

        ln = QLabel(tr("Questions / notes supplémentaires :", "Additional questions / notes:")); ln.setStyleSheet("color:#888; background:transparent;"); cl.addWidget(ln)
        self.ai_ctx_notes = QPlainTextEdit(); self.ai_ctx_notes.setFixedHeight(60); cl.addWidget(self.ai_ctx_notes)
        self.ai_include_sysinfo = QCheckBox(tr("Inclure infos système (CPU, GPU, RAM, stockage)", "Include system info (CPU, GPU, RAM, storage)")); self.ai_include_sysinfo.setChecked(True)
        cl.addWidget(self.ai_include_sysinfo); v.addWidget(f_ctx)

        # ── Bouton analyse ──
        self.btn_analyze = QPushButton(tr("Analyser ma Configuration", "Analyze my Configuration"))
        self.btn_analyze.setStyleSheet("background:#8e44ad; color:#fff; border:none; border-radius:8px; padding:11px; font-weight:700; font-size:14px;")
        self.btn_analyze.clicked.connect(self._check_config_ai); v.addWidget(self.btn_analyze)

        # ── Résultat ──
        self.ai_result_box = QPlainTextEdit(); self.ai_result_box.setReadOnly(True); self.ai_result_box.setMinimumHeight(260)
        v.addWidget(self.ai_result_box, 1)

        s = QScrollArea(); s.setWidgetResizable(True); s.setFrameShape(QFrame.NoFrame); s.setWidget(inner)
        return s

    # ── Vérification IA : handlers ────────────────────────────────────────────────
    def _on_ai_provider(self, name):
        skey, models = self._AI_MODELS.get(name, ("", []))
        self.ai_model.blockSignals(True)
        self.ai_model.clear(); self.ai_model.addItems(models)
        try:
            from src.core.ai_models_metadata import get_provider_default_model
            default = get_provider_default_model(name)
            if default in models:
                self.ai_model.setCurrentText(default)
        except Exception:
            pass
        self.ai_model.blockSignals(False)
        if skey:
            self.ai_key.setText(self.settings.get(skey, "") or self.settings.get("ai_api_key", "") or "")
        self._on_ai_model(self.ai_model.currentText())

    def _on_ai_model(self, model_id):
        try:
            from src.core.ai_models_metadata import get_model_description
            self.ai_model_desc.setText(get_model_description(model_id) or "")
        except Exception:
            self.ai_model_desc.setText("")

    def _test_api(self):
        provider = self.ai_provider.currentText(); key = self.ai_key.text().strip(); model = self.ai_model.currentText()
        if not key:
            self.lbl_test_result.setText(tr("❌ Clé API vide", "❌ Empty API key")); self.lbl_test_result.setStyleSheet("color:#e74c3c; background:transparent;")
            return
        self.lbl_test_result.setText(tr("⏳ Test en cours...", "⏳ Testing...")); self.lbl_test_result.setStyleSheet("color:#f39c12; background:transparent;")
        self.btn_test_api.setEnabled(False)

        class _TestWorker(QThread):
            done = Signal(bool, str)
            def run(self_):
                try:
                    from src.core.ai_cache import test_api_connection
                    ok, msg = test_api_connection(provider, key, model)
                    self_.done.emit(bool(ok), str(msg))
                except Exception as e:
                    self_.done.emit(False, str(e))

        def _fin(ok, msg):
            icon = "✅" if ok else "❌"; color = "#2ecc71" if ok else "#e74c3c"
            if len(msg) > 100:
                msg = msg[:100] + "..."
            self.lbl_test_result.setText(f"{icon} {msg}"); self.lbl_test_result.setStyleSheet(f"color:{color}; background:transparent;")
            self.btn_test_api.setEnabled(True)
            self._test_worker = None
        self._test_worker = _TestWorker(); self._test_worker.done.connect(_fin); self._test_worker.start()

    def _clear_ai_cache(self):
        try:
            from src.core.ai_cache import cache_stats, clear_cache
            stats = cache_stats()
            if stats.get("count", 0) == 0:
                QMessageBox.information(self, "Cache", tr("Le cache est déjà vide.", "The cache is already empty.")); return
            if QMessageBox.question(self, tr("Vider le cache", "Clear cache"),
                                    tr(f"Supprimer {stats['count']} réponses cachées ({stats.get('size_kb', 0):.1f} KB) ?",
                                       f"Delete {stats['count']} cached responses ({stats.get('size_kb', 0):.1f} KB)?")) == QMessageBox.Yes:
                n = clear_cache(); QMessageBox.information(self, "Cache", tr(f"{n} réponses supprimées.", f"{n} responses deleted."))
        except Exception as e:
            QMessageBox.critical(self, "Cache", str(e))

    def _load_template(self):
        choice = self.tpl_choice.currentText()
        if not choice or choice == tr("(aucun)", "(none)"):
            QMessageBox.information(self, "Template", tr("Sélectionnez un template.", "Select a template.")); return
        try:
            from src.core.config_templates import get_template
            tpl = get_template(choice)
        except Exception as e:
            QMessageBox.critical(self, "Template", str(e)); return
        if not tpl:
            QMessageBox.warning(self, "Template", tr(f"Template '{choice}' non trouvé.", f"Template '{choice}' not found.")); return
        if QMessageBox.question(self, tr("Charger Template", "Load Template"),
                                tr(f"Charger '{choice}' ?\n\n{tpl.get('_description', '')}\n\nLes valeurs actuelles seront écrasées.",
                                   f"Load '{choice}'?\n\n{tpl.get('_description', '')}\n\nCurrent values will be overwritten.")) != QMessageBox.Yes:
            return
        applied = self._apply_flat_config({k: v for k, v in tpl.items() if not k.startswith("_")})
        QMessageBox.information(self, "Template", tr(f"Template '{choice}' chargé ({applied} champs appliqués).", f"Template '{choice}' loaded ({applied} fields applied)."))

    def _apply_flat_config(self, flat: dict) -> int:
        """Reuse the loader path to push a flat dict into the widgets. Returns the number
        of keys that actually matched a real widget (see _set_field), not len(flat) —
        a dict full of unrecognized keys (e.g. a mismatched importer) now honestly
        reports 0/low instead of claiming every key was applied (2026-09-17 fix)."""
        try:
            return self._populate(flat)
        except Exception:
            return 0

    def _autodetect_ds_size(self):
        folder = ""
        try:
            folder = (self._data().get("dataroot_gt", "") or "").strip()
        except Exception:
            pass
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Auto-detect",
                                tr("Configurez d'abord un dossier Train HQ (GT) valide dans l'onglet Datasets.",
                                   "First set a valid Train HQ (GT) folder in the Datasets tab."))
            return
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        try:
            n = sum(1 for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts)
            self.ai_ctx_ds_size.setText(f"{n} images")
        except Exception as e:
            QMessageBox.critical(self, "Auto-detect", str(e))

    def _build_ai_prompt(self) -> str:
        data = self._data()
        engine = data.get("engine", self._engine()); arch = data.get("arch", "omnisr")
        ct = f"Moteur: {engine}\nArchitecture: {arch}\nScale: {data.get('scale', '4')}\n"
        ct += f"Batch Size: {data.get('batch_size', '4')}\nPatch Size: {data.get('patch_size', '64')}\n"
        ct += f"Total Iter: {data.get('total_iter', '150000')}\n"
        ct += f"Optimizer: {data.get('optim_g', 'AdamW')}\nLR: {data.get('lr', '5e-5')}\n"
        ct += f"Scheduler: {data.get('scheduler', 'MultiStepLR')}\n"
        ct += f"AMP: {data.get('use_amp', 'false')}\nGAN: {data.get('use_gan', 'false')}\n"
        if str(data.get("use_gan", "false")).lower() == "true":
            ct += f"Disc Type: {data.get('net_d_type', 'unet')}\nGAN Weight: {data.get('dyn_gan_weight', '0.05')}\n"
            ct += f"GAN Type: {data.get('gan_type', 'bce')}\n"
        losses = [f"{k}={data.get('weight_' + k, '1.0')}" for k in data
                  if k.startswith("loss_") and str(data[k]).lower() == "true"]
        if losses:
            ct += f"Losses: {', '.join(losses)}\n"
        if self.ai_send_mode.currentText() == tr("Config complète (détaillé)", "Full config (detailed)"):
            ct += f"\nConfig complète (JSON):\n{self._config_text()}\n"
        ctx = []
        if self.ai_ctx_dataset.currentText() != tr("(auto-detect)", "(auto-detect)"):
            ctx.append(f"Type de dataset: {self.ai_ctx_dataset.currentText()}")
        if self.ai_ctx_goal.currentText() != tr("(non spécifié)", "(unspecified)"):
            ctx.append(f"But: {self.ai_ctx_goal.currentText()}")
        if self.ai_ctx_time.currentText() != tr("(non spécifié)", "(unspecified)"):
            ctx.append(f"Temps alloué: {self.ai_ctx_time.currentText()}")
        if self.ai_ctx_ds_size.text().strip():
            ctx.append(f"Taille dataset: {self.ai_ctx_ds_size.text().strip()}")
        notes = self.ai_ctx_notes.toPlainText().strip()
        if notes:
            ctx.append(f"Notes utilisateur: {notes}")
        if self.ai_include_sysinfo.isChecked():
            si = self._collect_sysinfo()
            if si:
                ctx.append(f"Système: {si}")
        ctx_block = ("\n\nContexte supplémentaire:\n" + "\n".join(f"- {c}" for c in ctx)) if ctx else ""
        return (f"Tu es un expert en Super-Resolution et en entraînement de modèles IA. "
                f"Voici ma configuration {engine} avec le modèle {arch}. "
                f"Analyse-la et donne-moi des recommandations concrètes.{ctx_block}\n\n```\n{ct}```")

    def _collect_sysinfo(self) -> str:
        parts = []
        try:
            import psutil
            parts.append(f"RAM {psutil.virtual_memory().total // (1024**3)}GB")
            parts.append(f"CPU {psutil.cpu_count(logical=True)} threads")
        except Exception:
            pass
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml
            pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            vram = pynvml.nvmlDeviceGetMemoryInfo(h).total // (1024**3)
            parts.append(f"GPU {name} {vram}GB"); pynvml.nvmlShutdown()
        except Exception:
            pass
        return ", ".join(parts)

    def _check_config_ai(self):
        provider = self.ai_provider.currentText(); key = self.ai_key.text().strip(); model = self.ai_model.currentText()
        if not key:
            QMessageBox.critical(self, tr("Erreur", "Error"),
                                 tr("Veuillez entrer votre clé API.\n\nVous pouvez sauvegarder vos clés dans\nRéglages › Apparence & API.",
                                    "Please enter your API key.\n\nYou can save your keys in\nSettings › Appearance & API."))
            return
        skey = self._AI_MODELS.get(provider, ("", []))[0]
        if skey:
            self.settings.set(skey, key)
        prompt = self._build_ai_prompt()
        # cache lookup
        if self.ai_use_cache.isChecked():
            try:
                from src.core.ai_cache import get_cached_response
                cached = get_cached_response(provider, model, prompt)
                if cached:
                    self.ai_result_box.setPlainText(f"[CACHE HIT — pas d'appel API]\n\n{cached}"); return
            except Exception:
                pass
        self.ai_result_box.setPlainText("Analyse en cours... Veuillez patienter.")
        self.btn_analyze.setEnabled(False)
        use_cache = self.ai_use_cache.isChecked()

        from qt_ui.tabs.run_dialogs import _AIWorker
        self._ai_worker = _AIWorker(provider, model, key, prompt)

        def _fin(result):
            if use_cache and result and not result.startswith("❌"):
                try:
                    from src.core.ai_cache import store_response
                    store_response(provider, model, prompt, result)
                except Exception:
                    pass
            self.ai_result_box.setPlainText(result); self.btn_analyze.setEnabled(True); self._ai_worker = None
        self._ai_worker.done.connect(_fin); self._ai_worker.start()

    _LR_PSNR_TIP = tr("Learning Rate phase PSNR.\n"
                      "↑ plus élevé = convergence + rapide mais instable · ↓ plus bas = + lent, + sûr.\n"
                      "  5e-4 🔥 rapide · 3e-4 ✅ standard · 1e-4 🐢 sûr · 5e-5 🐌 fine-tuning",
                      "PSNR-phase learning rate.\n"
                      "↑ Higher = faster but unstable · ↓ Lower = slower but safer.\n"
                      "  5e-4 🔥 fast · 3e-4 ✅ standard · 1e-4 🐢 safe · 5e-5 🐌 fine-tuning")
    _LR_GAN_TIP = tr("Learning Rate phase GAN (2–5× plus bas que PSNR).\n"
                     "↑ trop haut = le discriminateur domine · ↓ = équilibre G/D plus stable.\n"
                     "  1e-4 ✅ standard GAN · 5e-5 🟢 conservateur · 2e-5 🐢 fine-tuning",
                     "GAN-phase learning rate (2–5× lower than PSNR).\n"
                     "↑ Too high = the discriminator dominates · ↓ Lower = more stable G/D balance.\n"
                     "  1e-4 ✅ standard GAN · 5e-5 🟢 conservative · 2e-5 🐢 fine-tuning")

    def _page_pipeline(self) -> QWidget:
        inner = QWidget(); v = QVBoxLayout(inner); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(8)
        self._header(v, tr("Pipeline Automatisé PSNR → GAN", "Automated PSNR → GAN Pipeline"))
        self._pipe = {}

        hint, hl = self._ai_card(tr("Un seul run suffit ?", "Is one run enough?"), "#f1c40f")
        hlb = QLabel("  " + tr(
            "Avec traiNNer-redux, si les réglages ne changent pas entre les deux phases, préférez "
            "Architecture → Discriminateur → « Début GAN » + « Auto (plateau PSNR) » : un seul "
            "training, bascule automatique sur plateau du PSNR. Cette page sert quand chaque phase "
            "a ses propres réglages (LR, losses, batch, patch) ou pour NeoSR.",
            "With traiNNer-redux, if the settings stay the same in both phases, prefer "
            "Architecture → Discriminator → “GAN start” + “Auto (PSNR plateau)”: one training, "
            "automatic switch on a PSNR plateau. This page is for phases with their own settings "
            "(LR, losses, batch, patch) or for NeoSR."))
        hlb.setStyleSheet("color:#888; background:transparent;"); hlb.setWordWrap(True); hl.addWidget(hlb)
        v.addWidget(hint)

        info, il = self._ai_card(tr("Comment ça marche / How it works", "How it works"), "#3498db")
        for txt in [tr("1. Configure ton modèle dans Architecture / Datasets / Entraînement.",
                       "1. Configure your model in Architecture / Datasets / Training."),
                    tr("2. Ici, définis seulement les SURCHARGES par phase (iters, LR, losses).",
                       "2. Here, only define the per-phase OVERRIDES (iters, LR, losses)."),
                    tr("3. Phase 1 (PSNR) entraîne sans GAN → fidélité structurelle.",
                       "3. Phase 1 (PSNR) trains without GAN → structural fidelity."),
                    tr("4. Phase 2 (GAN) active le discriminateur et part du modèle final de la "
                       "Phase 1 (Redux : branché automatiquement ; NeoSR : à régler dans Datasets → "
                       "Pretrain Model une fois la Phase 1 finie). Les deux configs peuvent être "
                       "ajoutées à la file d'attente, qui les enchaîne.",
                       "4. Phase 2 (GAN) enables the discriminator and starts from Phase 1's final "
                       "model (Redux: wired automatically; NeoSR: set Datasets → Pretrain Model once "
                       "Phase 1 is done). Both configs can be added to the Queue, which chains them."),
                    tr("5. « Test Rapide » génère 2 configs de 2000 iters pour valider la chaîne.",
                       "5. \u201cQuick Test\u201d generates 2 configs of 2000 iters to validate the chain.")]:
            lb = QLabel("  " + txt); lb.setStyleSheet("color:#888; background:transparent;"); lb.setWordWrap(True); il.addWidget(lb)
        v.addWidget(info)

        # ── Phase 1 : PSNR ──
        p1, pl = self._ai_card(tr("Phase 1 — PSNR (Fidélité / Fidelity)", "Phase 1 — PSNR (Fidelity)"), "#2ecc71")
        self._pipe["psnr_iter"] = self._pipe_field(pl, tr("Itérations :", "Iterations:"), "150000",
                                                   tr("Itérations phase PSNR (surcharge total_iter) / PSNR-phase iterations.",
                                                      "PSNR-phase iterations (overrides total_iter)."))
        self._pipe["psnr_lr"] = self._pipe_field(pl, "Learning Rate :", "5e-4", self._LR_PSNR_TIP)
        self._pipe["psnr_batch"] = self._pipe_field(pl, "Batch Size :", "", tr("Vide = valeur des pages principales / empty = use main config.", "Empty = use the main config value."))
        self._pipe["psnr_patch"] = self._pipe_field(pl, tr("Patch Size (GT) :", "Patch Size (GT):"), "", tr("Vide = valeur des pages principales / empty = use main config.", "Empty = use the main config value."))
        lr = QHBoxLayout(); ll = QLabel(tr("Losses :", "Losses:")); ll.setFixedWidth(120); ll.setStyleSheet("background:transparent;"); lr.addWidget(ll)
        for key, label, chk in (("psnr_pixel", "Pixel (L1)", True), ("psnr_percep", "Perceptual", False),
                                ("psnr_mssim", "MS-SSIM", False), ("psnr_ldl", "LDL", False), ("psnr_ff", "Focal Freq", False)):
            c = QCheckBox(label); c.setChecked(chk); self._pipe[key] = c; lr.addWidget(c)
        lr.addStretch(1); pl.addLayout(lr)
        orow = QHBoxLayout(); ol = QLabel(tr("Optim / Sched :", "Optim / Sched:")); ol.setFixedWidth(120); ol.setStyleSheet("background:transparent;"); orow.addWidget(ol)
        self._pipe["psnr_optim"] = QComboBox(); self._pipe["psnr_optim"].addItems([tr("(depuis config)", "(from config)"), "adan", "adamw", "adam"])
        self._pipe["psnr_optim"].setFixedWidth(140); orow.addWidget(self._pipe["psnr_optim"])
        self._pipe["psnr_sched"] = QComboBox(); self._pipe["psnr_sched"].addItems([tr("(depuis config)", "(from config)"), "CosineAnnealing", "MultiStepLR"])
        self._pipe["psnr_sched"].setFixedWidth(160); orow.addWidget(self._pipe["psnr_sched"]); orow.addStretch(1); pl.addLayout(orow)
        v.addWidget(p1)

        # ── Phase 2 : GAN ──
        p2, gl = self._ai_card(tr("Phase 2 — GAN (Textures réalistes / Realistic textures)", "Phase 2 — GAN (Realistic textures)"), "#e74c3c")
        self._pipe["gan_iter"] = self._pipe_field(gl, tr("Itérations :", "Iterations:"), "100000", tr("Itérations phase GAN. 50K–200K typique / GAN-phase iterations.", "GAN-phase iterations. 50K-200K typical."))
        self._pipe["gan_lr"] = self._pipe_field(gl, "Learning Rate :", "1e-4", self._LR_GAN_TIP)
        self._pipe["gan_weight"] = self._pipe_field(gl, tr("Poids GAN :", "GAN Weight:"), "0.1",
                                                    tr("Poids du loss adversarial / GAN loss weight. 0.05–0.3. ↑ = textures + marquées mais + d'artefacts.", "Adversarial loss weight. 0.05-0.3. Higher = stronger textures but more artifacts."))
        self._pipe["gan_percep_w"] = self._pipe_field(gl, tr("Poids Perceptual :", "Perceptual Weight:"), "1.0", tr("Poids perceptual en phase GAN / perceptual weight.", "Perceptual weight in the GAN phase."))
        self._pipe["gan_pixel_w"] = self._pipe_field(gl, tr("Poids Pixel :", "Pixel Weight:"), "0.01", tr("Poids pixel en GAN (très faible) / pixel weight (very low).", "Pixel weight in GAN (very low)."))
        drow = QHBoxLayout(); dl = QLabel(tr("Disc / GAN type :", "Disc / GAN type:")); dl.setFixedWidth(120); dl.setStyleSheet("background:transparent;"); drow.addWidget(dl)
        self._pipe["disc"] = QComboBox(); self._pipe["disc"].addItems([tr("(depuis config)", "(from config)"), "unet", "dunet", "patchgan", "metagan", "ea2fpn"])
        self._pipe["disc"].setFixedWidth(140); drow.addWidget(self._pipe["disc"])
        self._pipe["gan_type"] = QComboBox(); self._pipe["gan_type"].addItems([tr("(depuis config)", "(from config)"), "bce", "mse", "huber"])
        self._pipe["gan_type"].setFixedWidth(120); drow.addWidget(self._pipe["gan_type"]); drow.addStretch(1); gl.addLayout(drow)
        er = QHBoxLayout(); el = QLabel(tr("Losses extra :", "Extra Losses:")); el.setFixedWidth(120); el.setStyleSheet("background:transparent;"); er.addWidget(el)
        for key, label, chk in (("gan_pixel", "Pixel", True), ("gan_percep", "Perceptual", True), ("gan_ldl", "LDL", False),
                                ("gan_dists", "DISTS", False), ("gan_ff", "Focal Freq", False), ("gan_mssim", "MS-SSIM", False)):
            c = QCheckBox(label); c.setChecked(chk); self._pipe[key] = c; er.addWidget(c)
        er.addStretch(1); gl.addLayout(er)
        v.addWidget(p2)

        br = QHBoxLayout()
        bt = QPushButton(tr("⚡ Test Rapide (2K + 2K iters)", "⚡ Quick Test (2K + 2K iters)"))
        bt.setStyleSheet('background:#e67e22; color:#fff; border:none; border-radius:8px; padding:9px 14px; font-weight:700; font-family:"Segoe UI Emoji";')
        bt.clicked.connect(lambda: self._gen_pipeline(test_mode=True)); br.addWidget(bt)
        bg = QPushButton(tr("🚀 Générer les configs PSNR + GAN", "🚀 Generate PSNR + GAN configs"))
        bg.setStyleSheet('background:#27ae60; color:#fff; border:none; border-radius:8px; padding:9px 14px; font-weight:700; font-family:"Segoe UI Emoji";')
        bg.clicked.connect(lambda: self._gen_pipeline(test_mode=False)); br.addWidget(bg)
        self._pipe["to_queue"] = QCheckBox(tr("Ajouter les 2 phases à la file d'attente", "Add both phases to the Queue"))
        self._pipe["to_queue"].setChecked(True)
        self._pipe["to_queue"].setToolTip(tr(
            "Ajoute Phase 1 puis Phase 2 à l'onglet File d'attente : « Lancer la queue » les enchaîne. "
            "Si la Phase 1 échoue, la Phase 2 s'arrête aussitôt (modèle de départ absent).",
            "Adds Phase 1 then Phase 2 to the Queue tab: “Run queue” chains them. "
            "If Phase 1 fails, Phase 2 stops right away (missing starting model)."))
        br.addWidget(self._pipe["to_queue"]); br.addStretch(1)
        v.addLayout(br)

        # ── Ou : deux fichiers d'options complets (Redux) ──
        pf, fl = self._ai_card(tr("Ou : enchaîner deux fichiers d'options complets (Redux)",
                                  "Or: chain two full option files (Redux)"), "#9b59b6")
        flb = QLabel("  " + tr(
            "Chaque phase garde tous ses réglages (losses, GAN, dégradations…). L'app écrit une "
            "copie « .chained » de la Phase 2 qui part du modèle final de la Phase 1 (ton fichier "
            "n'est pas modifié), puis ajoute les deux à la file d'attente.",
            "Each phase keeps all its settings (losses, GAN, degradations…). The app writes a "
            "“.chained” copy of Phase 2 that starts from Phase 1's final model (your file is not "
            "changed), then adds both to the Queue."))
        flb.setStyleSheet("color:#888; background:transparent;"); flb.setWordWrap(True); fl.addWidget(flb)
        for key, label in (("file1", tr("Phase 1 (PSNR) :", "Phase 1 (PSNR):")),
                           ("file2", tr("Phase 2 (GAN) :", "Phase 2 (GAN):"))):
            r = QHBoxLayout(); lb = QLabel(label); lb.setFixedWidth(120); lb.setStyleSheet("background:transparent;"); r.addWidget(lb)
            e = QLineEdit(); e.setPlaceholderText("….yml"); self._pipe[key] = e; r.addWidget(e, 1)
            b = QPushButton("…"); b.setFixedWidth(40)
            b.clicked.connect(lambda _=False, ed=e: self._pick_pipe_file(ed)); r.addWidget(b)
            fl.addLayout(r)
        bc = QPushButton(tr("🔗 Enchaîner ces 2 fichiers", "🔗 Chain these 2 files"))
        bc.setStyleSheet('background:#8e44ad; color:#fff; border:none; border-radius:8px; padding:9px 14px; font-weight:700; font-family:"Segoe UI Emoji";')
        bc.clicked.connect(self._chain_pipe_files)
        cr = QHBoxLayout(); cr.addWidget(bc); cr.addStretch(1); fl.addLayout(cr)
        v.addWidget(pf)

        self._pipe_out = QLabel(""); self._pipe_out.setStyleSheet("color:#888;"); self._pipe_out.setWordWrap(True)
        v.addWidget(self._pipe_out); v.addStretch(1)

        s = QScrollArea(); s.setWidgetResizable(True); s.setFrameShape(QFrame.NoFrame); s.setWidget(inner)
        return s

    def _pick_pipe_file(self, edit):
        start = os.path.dirname(edit.text().strip()) if edit.text().strip() else ""
        if not start:
            try:
                from src.core.engine_paths import base_engine_path
                start = os.path.join(base_engine_path(), "Option Custom", "trainner_redux")
            except Exception:
                start = ""
        path, _ = QFileDialog.getOpenFileName(self, tr("Fichier d'options", "Option file"), start,
                                              "Config Redux (*.yml *.yaml)")
        if path:
            edit.setText(path)

    def _chain_pipe_files(self):
        """Phase 1 + Phase 2 from two full Redux option files: writes <phase2>.chained.yml
        starting from phase 1's final model, then queues both."""
        import yaml
        out = []
        try:
            f1 = self._pipe["file1"].text().strip(); f2 = self._pipe["file2"].text().strip()
            for f in (f1, f2):
                if not f or not os.path.isfile(f) or os.path.splitext(f)[1].lower() not in (".yml", ".yaml"):
                    raise ValueError(tr(f"Fichier Redux (.yml) introuvable : {f or '(vide)'}",
                                        f"Redux file (.yml) not found: {f or '(empty)'}"))
            with open(f1, encoding="utf-8") as fh:
                c1 = yaml.safe_load(fh) or {}
            with open(f2, encoding="utf-8") as fh:
                c2 = yaml.safe_load(fh) or {}
            n1 = str(c1.get("name") or "").strip()
            if not n1:
                raise ValueError(tr("La Phase 1 n'a pas de « name ».", "Phase 1 has no “name”."))
            if str(c2.get("name") or "").strip() == n1:
                raise ValueError(tr("Les deux phases ont le même nom d'expérience : la Phase 2 écraserait la Phase 1.",
                                    "Both phases have the same experiment name: Phase 2 would overwrite Phase 1."))
            if c1.get("network_g") != c2.get("network_g") or c1.get("scale") != c2.get("scale"):
                raise ValueError(tr("Architecture ou scale différents entre les deux phases : le modèle de la Phase 1 ne se chargerait pas.",
                                    "Architecture or scale differ between the phases: Phase 1's model would not load."))
            t1 = c1.get("train") or {}
            t2 = c2.setdefault("train", {})
            losses1 = [str(l.get("type", "")).lower() for l in (t1.get("losses") or []) if isinstance(l, dict)]
            losses2 = [str(l.get("type", "")).lower() for l in (t2.get("losses") or []) if isinstance(l, dict)]
            if "ganloss" in losses1:
                out.append(tr("⚠ La Phase 1 contient déjà une loss GAN.", "⚠ Phase 1 already has a GAN loss."))
            if "ganloss" not in losses2:
                out.append(tr("⚠ La Phase 2 n'a pas de loss GAN.", "⚠ Phase 2 has no GAN loss."))
            from src.core.engine_paths import redux_path
            ema = float(t1.get("ema_decay", 0) or 0) > 0
            model = os.path.join(redux_path(), "experiments", n1, "models",
                                 f"net_g{'_ema' if ema else ''}_latest.safetensors").replace("\\", "/")
            path = c2.setdefault("path", {}) or {}
            c2["path"] = path
            path["pretrain_network_g"] = model
            path["resume_state"] = None
            # Phase 2 is the GAN phase: no delayed GAN start inside it.
            for k in ("gan_start_iter", "gan_auto_start", "gan_auto_patience",
                      "gan_auto_min_delta", "gan_auto_min_iter", "gan_restart_lr"):
                t2.pop(k, None)
            b2, e2 = os.path.splitext(f2)
            chained = b2 + ".chained" + e2
            with open(chained, "w", encoding="utf-8") as fh:
                yaml.safe_dump(c2, fh, sort_keys=False, allow_unicode=True)
            out.insert(0, tr(f"✅ Phase 2 enchaînée → {chained}", f"✅ Phase 2 chained → {chained}"))
            out.insert(1, tr(f"↪ Elle part de : {model}", f"↪ It starts from: {model}"))
            from qt_ui.tabs.queue_tab import QueueTab
            from PySide6.QtWidgets import QListWidgetItem
            q = self.window().findChild(QueueTab)
            if q is not None:
                q.list.addItem(QListWidgetItem(f1)); q.list.addItem(QListWidgetItem(chained))
                out.append(tr("📋 Ajoutées à la file d'attente (onglet File d'attente → Lancer la queue).",
                              "📋 Added to the Queue (Queue tab → Run queue)."))
            self._pipe_out.setText("\n".join(out)); self._pipe_out.setStyleSheet("color:#2ecc71;")
        except Exception as e:
            self._pipe_out.setText(f"❌ {e}"); self._pipe_out.setStyleSheet("color:#e74c3c;")

    def _pipe_field(self, layout, label, default, tip):
        r = QHBoxLayout(); lb = QLabel(label); lb.setFixedWidth(120); lb.setStyleSheet("background:transparent;")
        lb.setToolTip(tip); r.addWidget(lb)
        e = QLineEdit(default); e.setFixedWidth(120); e.setPlaceholderText(tr("(depuis config)", "(from config)")); e.setToolTip(tip)
        r.addWidget(e); r.addStretch(1); layout.addLayout(r)
        return e

    def _gen_pipeline(self, test_mode=False):
        folder = QFileDialog.getExistingDirectory(self, tr("Dossier de sortie des configs", "Configs output folder"))
        if not folder:
            return
        try:
            from src.core.config_handler import ConfigHandler
            base = self._data()
            redux = "Redux" in self._engine()
            ext = ".yml" if redux else ".toml"
            name = base.get("name", "experiment")
            P = self._pipe

            def _txt(w):
                return w.text().strip() if hasattr(w, "text") else ""

            def _combo(w):
                t = w.currentText(); return "" if t.startswith("(") else t

            # Phase 1 — PSNR (no GAN)
            d1 = dict(base)
            d1["use_gan"] = "false"; d1["name"] = f"{'TEST_' if test_mode else ''}{name}_PSNR"
            d1["resume_state"] = ""
            d1["total_iter"] = "2000" if test_mode else (_txt(P["psnr_iter"]) or base.get("total_iter", "150000"))
            if _txt(P["psnr_lr"]):
                d1["lr"] = _txt(P["psnr_lr"])
            if _txt(P["psnr_batch"]):
                d1["batch_size"] = _txt(P["psnr_batch"])
            if _txt(P["psnr_patch"]):
                d1["patch_size"] = _txt(P["psnr_patch"])
            for key, flag in (("psnr_pixel", "loss_pixel"), ("psnr_percep", "loss_percep"), ("psnr_mssim", "loss_mssim"),
                              ("psnr_ldl", "loss_ldl"), ("psnr_ff", "loss_ff")):
                d1[flag] = "true" if P[key].isChecked() else "false"
            if _combo(P["psnr_optim"]):
                d1["optim_g"] = _combo(P["psnr_optim"])
            if _combo(P["psnr_sched"]):
                d1["scheduler"] = _combo(P["psnr_sched"])

            # Phase 2 — GAN
            d2 = dict(base)
            d2["use_gan"] = "true"; d2["name"] = f"{'TEST_' if test_mode else ''}{name}_GAN"
            d2["resume_state"] = ""
            # The pipeline already separates the phases: no delayed GAN start inside phase 2.
            d2["gan_start_iter"] = "0"; d2["gan_auto_start"] = "false"
            p1_model = ""
            if redux:
                # Redux writes the final model as experiments/<name>/models/net_g[_ema]_latest
                # under the engine folder the Queue runs (engine_paths.redux_path()).
                try:
                    from src.core.engine_paths import redux_path
                    _ema = float(str(base.get("ema", "0.999") or 0)) > 0
                except Exception:
                    _ema = True
                    redux_path = None
                if redux_path is not None:
                    p1_model = os.path.join(redux_path(), "experiments", d1["name"], "models",
                                            f"net_g{'_ema' if _ema else ''}_latest.safetensors").replace("\\", "/")
                    d2["pretrain_model"] = p1_model
            d2["total_iter"] = "2000" if test_mode else (_txt(P["gan_iter"]) or "100000")
            if _txt(P["gan_lr"]):
                d2["lr"] = _txt(P["gan_lr"])
            if _txt(P["gan_weight"]):
                d2["dyn_gan_weight"] = _txt(P["gan_weight"])
            for key, flag in (("gan_pixel", "loss_pixel"), ("gan_percep", "loss_percep"), ("gan_ldl", "loss_ldl"),
                              ("gan_dists", "loss_dists"), ("gan_ff", "loss_ff"), ("gan_mssim", "loss_mssim")):
                d2[flag] = "true" if P[key].isChecked() else "false"
            if _txt(P.get("gan_percep_w")):
                d2["weight_loss_percep"] = _txt(P["gan_percep_w"])
            if _txt(P.get("gan_pixel_w")):
                d2["weight_loss_pixel"] = _txt(P["gan_pixel_w"])
            if _combo(P["disc"]):
                d2["net_d_type"] = _combo(P["disc"])
            if _combo(P["gan_type"]):
                d2["gan_type"] = _combo(P["gan_type"])

            results = []
            for tag, data in (("PSNR", d1), ("GAN", d2)):
                path = os.path.join(folder, f"{data['name']}{ext}")
                ch = ConfigHandler(); ch.set_engine("TraiNNer-Redux" if redux else "NeoSR")
                ok, msg = ch.generate_config(data, path)
                results.append(f"{'✅' if ok else '❌'} {tag} → {path}" + ("" if ok else f"  ({msg})"))
            if p1_model:
                results.append(tr(f"↪ Phase 2 part de : {p1_model}", f"↪ Phase 2 starts from: {p1_model}"))
            else:
                results.append(tr("↪ Phase 2 : régler Datasets → Pretrain Model après la Phase 1.",
                                  "↪ Phase 2: set Datasets → Pretrain Model after Phase 1."))
            if P["to_queue"].isChecked() and all(r.startswith("✅") for r in results[:2]):
                try:
                    from qt_ui.tabs.queue_tab import QueueTab
                    from PySide6.QtWidgets import QListWidgetItem
                    q = self.window().findChild(QueueTab)
                    if q is not None:
                        for data in (d1, d2):
                            q.list.addItem(QListWidgetItem(os.path.join(folder, f"{data['name']}{ext}")))
                        results.append(tr("📋 Ajoutées à la file d'attente (onglet File d'attente → Lancer la queue).",
                                          "📋 Added to the Queue (Queue tab → Run queue)."))
                except Exception as e:
                    results.append(tr(f"⚠ File d'attente : {e}", f"⚠ Queue: {e}"))
            self._pipe_out.setText("\n".join(results)); self._pipe_out.setStyleSheet("color:#2ecc71;")
            try:
                os.startfile(folder)
            except Exception:
                pass
        except Exception as e:
            self._pipe_out.setText(f"❌ {e}"); self._pipe_out.setStyleSheet("color:#e74c3c;")

    def _placeholder(self, title, msg) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(8, 8, 8, 8)
        t = QLabel(title); t.setObjectName("title"); v.addWidget(t)
        s = QLabel(msg); s.setObjectName("subtitle"); v.addWidget(s)
        v.addStretch(1); return w

    # ── logic ───────────────────────────────────────────────────────────────────
    def _show(self, key):
        for i, (k, b) in enumerate(self._navbtns):
            b.setChecked(k == key)
            if k == key:
                self.stack.setCurrentIndex(i)

    def _engine(self) -> str:
        return self.engine_combo.currentText()

    def _on_engine_change(self, *_):
        is_redux = "Redux" in self._engine()
        self.settings.set("engine", self._engine())
        try:
            from src.core.descriptions import get_arch_families
            fams = get_arch_families("redux" if is_redux else "neosr")
            self._families = fams
            self.fam_combo.blockSignals(True)
            self.fam_combo.clear(); self.fam_combo.addItem(tr("TOUTES", "ALL"))
            self.fam_combo.addItems(sorted(fams.keys()))
            self.fam_combo.blockSignals(False)
            self._filter_archs("TOUTES")
        except Exception:
            self.arch_combo.clear()
            self.arch_combo.addItems(["omnisr", "span", "esrgan", "compact", "gfisrv2"])
        self._load_banner(is_redux)
        self._refresh_disc(is_redux)
        self._refresh_optim_sched(is_redux)
        self._apply_loss_engine(is_redux)
        self._apply_metric_engine(is_redux)
        self._on_arch_change()

    def _apply_metric_engine(self, is_redux):
        """Grise DISTS/TOPIQ en mode Redux : le moteur y fige val.metrics à psnr/ssim
        uniquement, ces cases n'ont donc aucun effet côté Redux."""
        for chk in getattr(self, "_metric_checks_neosr_only", []):
            chk.setEnabled(not is_redux)

    def _apply_loss_engine(self, is_redux):
        """Grey out losses incompatible with the selected engine (NeoSR-only when
        Redux is active, and vice-versa) — fields AND row labels."""
        for k in _NEOSR_ONLY:
            w = self.fields.get(k)
            if w is not None:
                w.setEnabled(not is_redux)
        for k in _REDUX_ONLY:
            w = self.fields.get(k)
            if w is not None:
                w.setEnabled(is_redux)
        # whole-row greying (labels included) for fully engine-locked losses
        rows = getattr(self, "_loss_rows", {})
        for k in _ROW_LOCK_NEOSR:
            for w in rows.get(k, []):
                w.setEnabled(not is_redux)
        for k in _ROW_LOCK_REDUX:
            for w in rows.get(k, []):
                w.setEnabled(is_redux)
        # phase losses: traiNNer-redux only
        if getattr(self, "chk_lphase", None) is not None:
            if not is_redux and self.chk_lphase.isChecked():
                self.chk_lphase.setChecked(False)
            for w in self._lphase_lock:
                w.setEnabled(is_redux)
        self._apply_criterion_engine(is_redux)

    # Engine-dependent option lists for the pixel/perceptual/LDL criterion combos —
    # a value valid on one engine's backend can be rejected by the other's (e.g.
    # Redux only accepts "charbonnierloss"/"l1loss"/"mseloss" for pixel_criterion).
    _CRITERION_OPTIONS = {
        "pixel_criterion": {
            True: ["charbonnierloss", "l1loss", "mseloss"],
            False: ["L1Loss", "MSELoss", "HuberLoss", "chc_loss"],
        },
        "percep_criterion": {
            True: ["charbonnier", "l1", "fd", "pd", "fd+l1", "pd+l1"],
            # neosr vgg_perceptual_loss accepts l1 / l2 / huber / chc only ("charbonnier" and
            # "chc_loss" crashed at launch: NotImplementedError: criterion not supported)
            False: ["huber", "l1", "l2", "chc"],
        },
        "ldl_criterion": {
            True: ["charbonnier", "l1", "l2", "huber"],
            False: ["l1", "l2", "huber", "chc"],
        },
    }

    def _apply_criterion_engine(self, is_redux):
        for key, by_engine in self._CRITERION_OPTIONS.items():
            cb = self.fields.get(key)
            if cb is None:
                continue
            opts = by_engine[is_redux]
            cur = cb.currentText()
            cb.blockSignals(True)
            cb.clear(); cb.addItems(opts)
            cb.setCurrentText(cur if cur in opts else opts[0])
            cb.blockSignals(False)

    def _refresh_optim_sched(self, is_redux):
        """Engine-dependent optimizer / scheduler / scale lists (what the engine
        actually supports — REDUX vs NeoSR)."""
        try:
            from src.core import descriptions as D
            triples = [
                (getattr(self, "optim_combo", None), D.REDUX_OPTIMIZERS if is_redux else D.NEOSR_OPTIMIZERS),
                (getattr(self, "sched_combo", None), D.REDUX_SCHEDULERS if is_redux else D.NEOSR_SCHEDULERS),
                (self.fields.get("scale"), D.REDUX_SCALES if is_redux else D.NEOSR_SCALES),
            ]
        except Exception:
            return
        for combo, items in triples:
            if combo is None:
                continue
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear(); combo.addItems([str(x) for x in items])
            combo.setCurrentText(cur if cur in [str(x) for x in items] else str(items[0]))
            combo.blockSignals(False)

    def _filter_archs(self, fam):
        archs = []
        try:
            if fam in ("TOUTES", "ALL") or not hasattr(self, "_families"):
                for lst in getattr(self, "_families", {}).values():
                    archs += list(lst)
                archs = sorted(set(archs))
            else:
                archs = list(self._families.get(fam, []))
        except Exception:
            pass
        if not archs:
            archs = ["omnisr", "span", "esrgan", "compact", "gfisrv2"]
        cur = self.arch_combo.currentText()
        self.arch_combo.blockSignals(True)
        self.arch_combo.clear(); self.arch_combo.addItems(archs)
        if cur in archs:
            self.arch_combo.setCurrentText(cur)
        self.arch_combo.blockSignals(False)
        self._on_arch_change()

    def _arch_fields(self, arch):
        is_redux = "Redux" in self._engine()
        try:
            from src.core import descriptions as D
            table = D.REDUX_ARCH_FIELDS if is_redux else D.ARCH_FIELDS
            # No num_feat fallback: many archs (artcnn, …) don't accept `num_feat` and the
            # engine then crashes with "unexpected keyword argument". For archs without a
            # field definition, inject NO params → the arch factory uses its own defaults.
            return table.get(arch) or table.get("default") or []
        except Exception:
            return []

    def _on_dataset_mode_change(self, *_):
        """Show the clip_size field only when 'sequence' (PairedVideoDataset) is the
        active dataset mode — irrelevant for otf/bicubic/paired."""
        is_seq = self.fields.get("dataset_mode") is not None and self.fields["dataset_mode"].currentText() == "sequence"
        if hasattr(self, "clip_size_label"):
            self.clip_size_label.setVisible(is_seq)
            self.clip_size_edit.setVisible(is_seq)

    _NON_SEQUENCE_MODES = ["otf", "bicubic", "paired"]
    _DATASET_PATH_KEYS = ("dataroot_gt", "val_gt", "val_lq", "dataroot_lq")

    def _swap_dataset_paths(self, to_video: bool):
        """Video/temporal Train+Val HQ/LQ folders are never the same folders as a
        normal arch's — cache each group separately (persisted in settings, so it
        survives app restarts) and swap the QLineEdit contents on transition,
        instead of leaving the previous mode's stale paths in place."""
        leaving = "img" if to_video else "seq"
        entering = "seq" if to_video else "img"
        for k in self._DATASET_PATH_KEYS:
            w = self.fields.get(k)
            if w is None:
                continue
            self.settings.set(f"cfg_path_{k}__{leaving}", w.text())
            restored = str(self.settings.get(f"cfg_path_{k}__{entering}", "") or "")
            w.setText(restored)
            self.settings.set(f"cfg_path_{k}", restored)

    def _update_dataset_mode_for_arch(self, arch):
        """Restrict the dataset-mode combo to exactly 'sequence' for video/temporal
        archs (TEMPORAL_ARCHS in traiNNer/archs/arch_info.py) — otf/bicubic/paired
        are meaningless for them (no PairedVideoDataset equivalent), so they're
        removed rather than just supplemented. Restores the normal 3-option combo
        when switching back to a non-temporal arch. Also swaps Train/Val HQ/LQ
        paths between the video and non-video path caches (see _swap_dataset_paths)."""
        if not hasattr(self, "dataset_mode_combo"):
            return
        mcb = self.dataset_mode_combo
        is_temporal = arch in _TEMPORAL_ARCHS
        is_seq_only = mcb.count() == 1 and mcb.itemText(0) == "sequence"
        if is_temporal and not is_seq_only:
            self._swap_dataset_paths(to_video=True)
            mcb.blockSignals(True)
            mcb.clear()
            mcb.addItem("sequence")
            mcb.setCurrentText("sequence")
            mcb.blockSignals(False)
            mcb.setEnabled(False)
            mcb.setToolTip(tr(
                "Verrouillé sur « sequence » — seul mode compatible avec les archs vidéo/temporelles.\n\n",
                "Locked to \"sequence\" — the only mode compatible with video/temporal archs.\n\n")
                + getattr(self, "_dataset_mode_tooltip", ""))
        elif not is_temporal and is_seq_only:
            self._swap_dataset_paths(to_video=False)
            mcb.blockSignals(True)
            mcb.clear()
            mcb.addItems(self._NON_SEQUENCE_MODES)
            mcb.setCurrentText("otf")
            mcb.blockSignals(False)
            mcb.setEnabled(True)
            mcb.setToolTip(getattr(self, "_dataset_mode_tooltip", ""))
        self._set_moa_enabled(not is_temporal)
        self._on_dataset_mode_change()

    def _set_moa_enabled(self, enabled: bool):
        """Grey out the MoA batch augmentations (MixUp/CutMix/ResizeMix/CutBlur) for
        video/temporal archs — they assume a (N,C,H,W) lq tensor and are incompatible
        with PairedVideoDataset's (N,T,C,H,W) clips (see config_handler.py's
        is_sequence force-to-"none" for the actual safety net; this is the visual cue)."""
        _reason = tr(
            "Désactivé : incompatible avec le mode séquence (les augmentations MoA supposent une "
            "image seule, pas un clip multi-frames).",
            "Disabled: incompatible with sequence mode (MoA augmentations assume a single image, "
            "not a multi-frame clip).")
        for _key in ("use_mixup", "use_cutmix", "use_resizemix", "use_cutblur"):
            for _fkey in (_key, f"{_key}_prob"):
                _w = self.fields.get(_fkey)
                if _w is not None:
                    _w.setEnabled(enabled)
                    # keep the normal description so it comes back once a non-temporal
                    # arch is selected again (it used to stay on the "disabled" reason)
                    _orig = _w.property("_uss_moa_tip")
                    if _orig is None:
                        _orig = _w.toolTip()
                        if _orig != _reason:
                            _w.setProperty("_uss_moa_tip", _orig)
                    if not enabled:
                        _w.setToolTip(_reason)
                    elif _orig is not None and _w.toolTip() == _reason:
                        _w.setToolTip(_orig)

    def _on_arch_change(self, *_):
        # rebuild the dynamic per-arch parameter box
        arch = self.arch_combo.currentText() or "omnisr"
        self._update_dataset_mode_for_arch(arch)
        # Archs à scale imposé (cascadedgaze 1x, MoTSR 2x) : aligne le Scale tout de suite.
        try:
            from src.core.descriptions import arch_fixed_scale
            _fx = arch_fixed_scale(arch)
            _sw = self.fields.get("scale")
            if _fx and _sw is not None and hasattr(_sw, "findText"):
                for _t in (str(_fx), f"{_fx}x", f"×{_fx}"):
                    if _sw.findText(_t) >= 0:
                        _sw.setCurrentText(_t)
                        break
        except Exception:
            pass
        # clear previous dyn_ fields + layout
        for k in [k for k in self.fields if k.startswith("dyn_") and k != "dyn_gan_weight"]:
            self.fields.pop(k, None)
        self._clear_grid(self.param_grid)
        self.param_box.setTitle(tr(f"Paramètres — {arch}", f"Parameters — {arch}"))
        specs = self._arch_fields(arch)
        for i, spec in enumerate(specs):
            label = spec.get("label", spec.get("key", "param"))
            skey = spec.get("key", f"p{i}")
            key = "dyn_" + skey
            default = spec.get("default", "")
            self.param_grid.addWidget(self._param_cell(key, label, default, slider_ok=True, spec=spec), 0, i)
            # dynamic param fields are keyed "dyn_<name>" which never matches
            # descriptions.py (arch-specific tooltips live under "<arch>_<name>",
            # e.g. "figsr_dim" — a few generic names like "gc"/"upsampler" are also
            # defined bare). _apply_field_tooltips() only runs once at init, before
            # any arch is picked, so these must be applied here on every rebuild.
            self._apply_dyn_tooltip(key, f"{arch}_{skey}", skey, spec.get("tip_key"))
        self.param_grid.setColumnStretch(len(specs), 1)
        self._schedule_vram()

    def _apply_dyn_tooltip(self, field_key, arch_specific_key, generic_key, spec_key=None):
        # Priority: the field's own tip_key (descriptions.py) > "<arch>_<param>" > "<param>".
        # The tip_key used to be ignored, so every arch whose tooltip key didn't happen to be
        # "<arch>_<param>" (TFDAT v2, MoTSR, MambaIR, FDAT-RD…) showed a generic tooltip.
        try:
            from qt_ui.i18n import tip_for
            t = (tip_for(spec_key) if spec_key else "") or tip_for(arch_specific_key) or tip_for(generic_key)
        except Exception:
            t = ""
        w = self.fields.get(field_key)
        if t and w is not None and hasattr(w, "setToolTip"):
            w.setToolTip(t)

    def _clear_grid(self, grid):
        """Remove all widgets AND reset column stretches (stale stretches from a
        previous arch left empty gaps between fields)."""
        while grid.count():
            it = grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for c in range(grid.columnCount() + 2):
            grid.setColumnStretch(c, 0)

    def _param_cell(self, key, label, default, slider_ok=False, spec=None) -> QWidget:
        """A compact label-over-field cell for the dynamic param rows (one row,
        left-aligned), matching the CTk dynamic-frame look. Renders a green combo
        when the field spec declares type='combobox'."""
        cell = QWidget(); cl = QVBoxLayout(cell); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(3)
        is_num = isinstance(default, (int, float))
        if spec and spec.get("type") == "combobox":
            cl.addWidget(QLabel(label))
            cb = QComboBox(); cb.addItems([str(c) for c in spec.get("choices", [])])
            cb.setCurrentText(str(default)); cb.setFixedWidth(160)
            cb.currentIndexChanged.connect(self._schedule_vram)
            self.fields[key] = cb; cl.addWidget(cb)
        elif str(default).lower() in ("true", "false"):
            cl.addWidget(QLabel(label))
            cb = QComboBox(); cb.addItems(["true", "false"]); cb.setCurrentText(str(default).lower())
            cb.setFixedWidth(90); cb.currentIndexChanged.connect(self._schedule_vram)
            self.fields[key] = cb; cl.addWidget(cb)
        elif slider_ok and is_num and ("feat" in key.lower() or "dim" in key.lower()):
            # step=1 (pas 8) : un pas de 8 snappait la valeur par défaut de l'arch sur la
            # grille (man/rgt 180→176, emt 60→64) → valeurs que l'arch refuse (conv groupée,
            # dim÷num_heads…). Pas de 1 = la valeur exacte de l'arch est préservée.
            sld = LabeledSlider(label, 16, max(256, int(default) * 2), float(default), 1, "{:.0f}")
            sld.setFixedWidth(200); sld.valueChangedF.connect(self._schedule_vram)
            self.fields[key] = sld; cl.addWidget(sld)
        else:
            cl.addWidget(QLabel(label))
            e = QLineEdit(str(default)); e.textChanged.connect(self._schedule_vram)
            e.setFixedWidth(90 if is_num else 140)
            self.fields[key] = e; cl.addWidget(e)
        return cell

    def _refresh_gan_start(self, *_):
        """Auto fields only matter with Auto; LR restart only with a delayed GAN."""
        f = self.fields
        auto = f["gan_auto_start"].isChecked()
        for k in ("gan_auto_patience", "gan_auto_min_delta", "gan_auto_min_iter"):
            f[k].setEnabled(auto)
        try:
            delayed = auto or int(f["gan_start_iter"].text() or 0) > 0
        except ValueError:
            delayed = False
        f["gan_restart_lr"].setEnabled(delayed)

    def _on_gan_toggle(self, *_):
        if hasattr(self, "gan_opts"):
            self.gan_opts.setVisible(self.fields["use_gan"].isChecked())
        self._schedule_vram()

    def _refresh_disc(self, is_redux):
        if not hasattr(self, "disc_combo"):
            return
        try:
            from src.core import descriptions as D
            disc = list(D.REDUX_DISC_LIST if is_redux else D.NEOSR_DISC_LIST)
            gans = list(D.REDUX_GAN_TYPES if is_redux else D.NEOSR_GAN_TYPES)
            disp = [D.DISC_DISPLAY_NAMES.get(x, x) for x in disc]
        except Exception:
            disc = ["unet", "patchgan"]; gans = ["vanilla", "bce"]; disp = disc
        cur = self.disc_combo.currentText()
        self.disc_combo.blockSignals(True)
        self.disc_combo.clear(); self.disc_combo.addItems(disp)
        if cur in disp:
            self.disc_combo.setCurrentText(cur)
        self.disc_combo.blockSignals(False)
        if self.disc_combo.currentText() != cur:
            # the selection moved (the previous D is not in this engine's list) while the
            # signal was blocked: rebuild its parameter fields, else the old D's dd_ params
            # were written for the new D ("DUnet got an unexpected keyword 'n_blocks'")
            self._on_disc_change()
        self.gan_type_combo.blockSignals(True)
        gcur = self.gan_type_combo.currentText()
        self.gan_type_combo.clear(); self.gan_type_combo.addItems(gans)
        if gcur in gans:
            self.gan_type_combo.setCurrentText(gcur)
        self.gan_type_combo.blockSignals(False)
        # 2e discriminateur : uniquement supporté côté moteur Redux
        # (network_d2/gan_opt2/optim_d2). Masqué pour NeoSR, jamais visible.
        if hasattr(self, "disc2_box"):
            self.disc2_box.setVisible(is_redux)
        if hasattr(self, "gan_start_box"):
            self.gan_start_box.setVisible(is_redux)
            if not is_redux and "use_disc2" in self.fields:
                self.fields["use_disc2"].setChecked(False)
        if is_redux:
            self._refresh_disc2()
        self._on_disc_change()

    def _refresh_disc2(self):
        if not hasattr(self, "disc2_combo"):
            return
        try:
            from src.core import descriptions as D
            disc = list(D.REDUX_DISC_LIST)
            disp = [D.DISC_DISPLAY_NAMES.get(x, x) for x in disc]
        except Exception:
            disc = ["unetdiscriminatorsn"]; disp = disc
        cur = self.disc2_combo.currentText()
        self.disc2_combo.blockSignals(True)
        self.disc2_combo.clear(); self.disc2_combo.addItems(disp)
        if cur in disp:
            self.disc2_combo.setCurrentText(cur)
        elif "Spectral (DualFormer)" in disp:
            self.disc2_combo.setCurrentText("Spectral (DualFormer)")
        self.disc2_combo.blockSignals(False)
        self._on_disc2_change()

    def _disc2_internal(self) -> str:
        disp = self.disc2_combo.currentText() if hasattr(self, "disc2_combo") else ""
        try:
            from src.core.descriptions import DISC_INTERNAL_NAMES
            return DISC_INTERNAL_NAMES.get(disp, disp)
        except Exception:
            return disp

    def _on_disc2_toggle(self, *_):
        if hasattr(self, "disc2_opts"):
            self.disc2_opts.setVisible(self.fields["use_disc2"].isChecked())
        self._on_disc2_change()
        self._schedule_vram()

    def _on_disc2_change(self, *_):
        if not hasattr(self, "disc2_grid"):
            return
        disc2 = self._disc2_internal() or "unetdiscriminatorsn"
        for k in [k for k in self.fields if k.startswith("dd2_")]:
            self.fields.pop(k, None)
        self._clear_grid(self.disc2_grid)
        try:
            from src.core.descriptions import DISC_FIELDS
            specs = DISC_FIELDS.get(disc2, [])
        except Exception:
            specs = []
        for i, spec in enumerate(specs):
            label = spec.get("label", spec.get("key", "p"))
            skey = spec.get("key", f"p{i}")
            key = "dd2_" + skey
            default = spec.get("default", "")
            self.disc2_grid.addWidget(self._param_cell(key, label, default, slider_ok=True, spec=spec), 0, i)
            self._apply_dyn_tooltip(key, f"{disc2}_{skey}", skey, spec.get("tip_key"))
        self.disc2_grid.setColumnStretch(len(specs), 1)
        self._check_disc_pair_warning()
        self._schedule_vram()

    # Discriminateurs quasi identiques à NeXtSRGAN (même pile conv VGG128-style,
    # cf. tooltip NeXtSRGAN et DISC_VRAM_OVERHEAD_GB) -- redondants s'ils sont
    # combinés avec lui : coût VRAM/temps doublé pour un gain quasi nul.
    _NEXTSRGAN_REDUNDANT_WITH = {
        "vggstylediscriminator", "docres_discriminator", "unetdiscriminatorsn",
        "patchgandiscriminatorsn", "multiscalepatchgandiscriminatorsn",
        "metagan2", "dunet",
    }

    def _check_disc_pair_warning(self, *_):
        """Avertissement ACTIF (label visible en permanence, pas juste un
        tooltip au survol) quand la paire D1/D2 choisie est une redondance
        connue. N'empêche pas la sélection (expérimental, l'utilisateur reste
        libre) -- informe seulement, clairement et sans dépendre du survol."""
        if not hasattr(self, "lbl_disc_pair_warn"):
            return
        use_gan = bool(self.fields.get("use_gan") and self.fields["use_gan"].isChecked())
        use_disc2 = bool(self.fields.get("use_disc2") and self.fields["use_disc2"].isChecked())
        if not use_gan:
            self.lbl_disc_pair_warn.setVisible(False)
            return

        # SpectralDiscriminator : patch_size doit diviser le crop (lq_size*scale),
        # sinon EinopsError au 1er iter (vécu en réel avec gt_size=80/patch_size=32).
        # On ne corrige plus rien automatiquement (préférence explicite : garder
        # patch_size tel quel) -- on prévient activement à la place. Vérifié pour D1
        # toujours (dès que le GAN est actif) et D2 seulement s'il est activé.
        bad_patch = self._spectral_patch_size_mismatch(check_d2=use_disc2)
        if bad_patch:
            which, ps, crop = bad_patch
            divisors = [d for d in (128, 96, 64, 48, 32, 16) if d % ps == 0] or [ps]
            self.lbl_disc_pair_warn.setText(tr(
                f"⚠️ {which} (Spectral) : patch_size={ps} ne divise pas le crop actuel "
                f"({crop}px). Ajuste \"Patch Size\" (crop réseau) à un multiple de {ps} "
                f"(ex. {', '.join(str(d) for d in divisors)}), ou baisse patch_size du "
                f"discriminateur. Sinon un filet de sécurité au runtime réduira "
                f"patch_size automatiquement pour éviter le crash (perte de qualité "
                f"spectrale).",
                f"⚠️ {which} (Spectral): patch_size={ps} does not divide the current "
                f"crop ({crop}px). Adjust \"Patch Size\" (network crop) to a multiple "
                f"of {ps} (e.g. {', '.join(str(d) for d in divisors)}), or lower the "
                f"discriminator's patch_size. Otherwise a runtime safety net will "
                f"shrink patch_size automatically to avoid the crash (lower spectral "
                f"quality)."))
            self.lbl_disc_pair_warn.setStyleSheet("color:#e74c3c; font-weight:bold; padding:4px; background:rgba(231,76,60,0.12); border-radius:4px;")
            self.lbl_disc_pair_warn.setVisible(True)
            return

        if not use_disc2:
            self.lbl_disc_pair_warn.setVisible(False)
            return
        d1 = (self._disc_internal() or "").lower()
        d2 = (self._disc2_internal() or "").lower()
        other = d2 if d1 == "nextsrgandiscriminator" else (d1 if d2 == "nextsrgandiscriminator" else None)
        is_redundant = other is not None and other in self._NEXTSRGAN_REDUNDANT_WITH
        if is_redundant:
            self.lbl_disc_pair_warn.setText(tr(
                "⚠️ Paire redondante : NeXtSRGAN est architecturalement quasi identique à "
                "VGG-Style/DocRes-Disc/UNet-SN/PatchGAN-SN/MultiPatchGAN-SN/MetaGAN2/DUNet "
                "(même domaine spatial). Combiner les deux double le coût VRAM/temps pour "
                "un gain quasi nul. Choisis plutôt un discriminateur de domaine différent "
                "(ex. Spectral ou Projected GAN) pour un vrai bénéfice.",
                "⚠️ Redundant pair: NeXtSRGAN is architecturally near-identical to "
                "VGG-Style/DocRes-Disc/UNet-SN/PatchGAN-SN/MultiPatchGAN-SN/MetaGAN2/DUNet "
                "(same spatial domain). Combining both doubles VRAM/time cost for near-zero "
                "gain. Pick a discriminator from a different domain instead (e.g. Spectral "
                "or Projected GAN) for an actual benefit."))
            self.lbl_disc_pair_warn.setStyleSheet("color:#e67e22; font-weight:bold; padding:4px; background:rgba(230,126,34,0.12); border-radius:4px;")
            self.lbl_disc_pair_warn.setVisible(True)
        elif d1 == d2:
            self.lbl_disc_pair_warn.setText(tr(
                "⚠️ D1 et D2 sont le même discriminateur : aucune diversité de domaine, "
                "double le coût pour rien.",
                "⚠️ D1 and D2 are the same discriminator: no domain diversity, doubles cost for nothing."))
            self.lbl_disc_pair_warn.setStyleSheet("color:#e67e22; font-weight:bold; padding:4px; background:rgba(230,126,34,0.12); border-radius:4px;")
            self.lbl_disc_pair_warn.setVisible(True)
        else:
            self.lbl_disc_pair_warn.setVisible(False)

    def _spectral_patch_size_mismatch(self, check_d2: bool):
        """Retourne (label, patch_size, crop) si D1 (ou D2 si check_d2) est un
        SpectralDiscriminator dont patch_size ne divise pas le crop réseau actuel
        (lq_size * scale), sinon None. Lu directement depuis les widgets (pas besoin
        de _values()/generate_config, ça doit réagir en live pendant l'édition)."""
        try:
            scale = int(float(self.fields["scale"].currentText())) if "scale" in self.fields else 1
        except Exception:
            scale = 1
        try:
            crop = int(self.fields["patch_size"].value()) * max(scale, 1)
        except Exception:
            return None
        candidates = [("D1", self._disc_internal(), "dd_patch_size")]
        if check_d2:
            candidates.append(("D2", self._disc2_internal(), "dd2_patch_size"))
        for label, disc, key in candidates:
            if (disc or "").lower() != "spectraldiscriminator":
                continue
            w = self.fields.get(key)
            if w is None:
                continue
            try:
                ps = int(float(w.text() if hasattr(w, "text") else w.value()))
            except Exception:
                continue
            if ps > 0 and crop % ps != 0:
                return (label, ps, crop)
        return None

    def _disc_internal(self) -> str:
        disp = self.disc_combo.currentText() if hasattr(self, "disc_combo") else "unet"
        try:
            from src.core.descriptions import DISC_INTERNAL_NAMES
            return DISC_INTERNAL_NAMES.get(disp, disp)
        except Exception:
            return disp

    def _on_disc_change(self, *_):
        if not hasattr(self, "disc_grid"):
            return
        disc = self._disc_internal() or "unet"
        for k in [k for k in self.fields if k.startswith("dd_")]:
            self.fields.pop(k, None)
        self._clear_grid(self.disc_grid)
        self.disc_box.setTitle(tr(f"Paramètres Disc. — {self.disc_combo.currentText()}", f"Disc. Parameters — {self.disc_combo.currentText()}"))
        try:
            from src.core.descriptions import DISC_FIELDS
            specs = DISC_FIELDS.get(disc, [])
        except Exception:
            specs = []
        for i, spec in enumerate(specs):
            label = spec.get("label", spec.get("key", "p"))
            skey = spec.get("key", f"p{i}")
            key = "dd_" + skey
            default = spec.get("default", "")
            self.disc_grid.addWidget(self._param_cell(key, label, default, slider_ok=True, spec=spec), 0, i)
            self._apply_dyn_tooltip(key, f"{disc}_{skey}", skey, spec.get("tip_key"))
        self.disc_grid.setColumnStretch(len(specs), 1)
        self._check_disc_pair_warning()
        self._schedule_vram()

    def _load_banner(self, is_redux):
        from qt_ui.theme import asset
        fname = "traiNNer redux.png" if is_redux else "neosr.png"
        path = asset(fname)
        if os.path.isfile(path):
            pm = QPixmap(path)
            if not pm.isNull():
                self.banner.setPixmap(pm.scaledToHeight(140, Qt.SmoothTransformation)); return
        self.banner.setText(f"[ {('TraiNNer-Redux' if is_redux else 'NeoSR')} ]")

    def _schedule_vram(self, *_):
        self._vram_timer.start()

    def _values(self) -> dict:
        out = {}
        for key, w in self.fields.items():
            if isinstance(w, QCheckBox):
                out[key] = "true" if w.isChecked() else "false"
            elif isinstance(w, QComboBox):
                out[key] = w.currentText()
            elif isinstance(w, (LabeledSlider, _LogSlider, _SliderBox)):
                out[key] = w.value()
            elif isinstance(w, QLineEdit):
                out[key] = w.text()
        return out

    def _do_vram(self):
        # Le slider "Patch Size" (lq_size) et les champs dd_/dd2_patch_size ne sont
        # reliés qu'à _schedule_vram -> _do_vram, pas à _check_disc_pair_warning
        # directement ; on la déclenche ici pour que l'avertissement SpectralDiscriminator
        # patch_size/crop reste à jour en live pendant l'édition.
        self._check_disc_pair_warning()
        vals = self._values()
        arch = self.arch_combo.currentText() or "omnisr"
        is_gan = self.fields["use_gan"].isChecked() if "use_gan" in self.fields else False
        est = None
        # VRAM + training speed: same estimate as the Run tab (losses, discriminators,
        # checkpointing, scale, OTF), from the config this form would generate.
        try:
            from src.core.compute_estimator import estimate_training_time, run_params_from_config
            cfg = None
            if "Redux" in self._engine():
                from src.core.config_handler import ConfigHandler
                ch = ConfigHandler(); ch.set_engine("TraiNNer-Redux")
                _dd = self._data()
                cfg = ch._gen_redux(_dd, None, _return_cfg=True)
                _lp2 = _dd.get("loss_phase2_data")
                if isinstance(cfg, dict) and isinstance(_lp2, dict):
                    # phase losses: the losses of both profiles are loaded from the start
                    _d2 = dict(_dd); _d2.update(_lp2)
                    _cfg2 = ch._gen_redux(_d2, None, _return_cfg=True)
                    if isinstance(_cfg2, dict):
                        _ls = cfg["train"]["losses"]
                        _have = {str(_l.get("type")) for _l in _ls if isinstance(_l, dict)}
                        for _l in (_cfg2.get("train") or {}).get("losses") or []:
                            if isinstance(_l, dict) and str(_l.get("type")) not in _have:
                                _ls.append(_l)
            if isinstance(cfg, dict):
                prm = run_params_from_config(cfg)
            else:
                prm = self._estimate_params_from_form(vals, arch, is_gan)
            res = estimate_training_time(**prm)
            est = res["vram_gb"]
            pct = min(100, int(est / max(self._total_vram, 0.1) * 100))
            col = "#2ecc71" if pct < 70 else ("#e67e22" if pct < 95 else "#e74c3c")
            self.lbl_vram.setText(tr(
                f"VRAM Estimée : {est:.1f} GB / {self._total_vram:.1f} GB ({pct}%)  ·  ~{res['itps']:.2f} it/s",
                f"Estimated VRAM: {est:.1f} GB / {self._total_vram:.1f} GB ({pct}%)  ·  ~{res['itps']:.2f} it/s")
                + (tr("  ⚠ dépasse la VRAM du GPU : manque de mémoire probable (réduire batch / patch)",
                      "  ⚠ exceeds the GPU VRAM: out of memory likely (lower batch / patch)")
                   if est > self._total_vram else ""))
            self.prog_vram.setValue(pct)
            self.prog_vram.setStyleSheet(f"QProgressBar::chunk{{background:{col};border-radius:5px;}}")
        except Exception as e:
            self.lbl_vram.setText(tr(f"VRAM : (indispo : {e})", f"VRAM: (unavailable: {e})"))
        # capability bars
        try:
            frac = est / max(self._total_vram, 0.1) if est is not None else None
            self.perf.update_stats(arch, is_gan, self._disc_internal(), vals, vram_frac=frac)
        except Exception:
            pass

    def _estimate_params_from_form(self, vals, arch, is_gan) -> dict:
        """estimate_training_time() kwargs straight from the form (NeoSR engine)."""
        num = _PerfBars._num
        losses = []
        for key, typ in (("loss_percep", "perceptualloss"), ("loss_dists", "distsloss"),
                         ("loss_ldl", "ldlloss"), ("loss_mssim", "mssimloss"),
                         ("loss_contextual", "contextualloss"), ("loss_fdl", "fdlloss"),
                         ("loss_ff", "ffloss"), ("loss_edge", "sobeledgeloss")):
            if num(vals, key):
                losses.append(typ)
        discs = []
        if is_gan:
            discs.append(self._disc_internal())
            if self.fields.get("use_disc2") and self.fields["use_disc2"].isChecked():
                discs.append(self._disc2_internal())
        _ck = self.fields.get("dyn_use_checkpoint")
        return {
            "architecture": arch,
            "batch_size": int(num(vals, "batch_size", 4)),
            "patch_size": int(num(vals, "patch_size", 64)),
            "scale": int(num(vals, "scale", 4)),
            "accumulate": int(num(vals, "accumulate", 1) or 1),
            "total_iter": int(num(vals, "total_iter", 100000)),
            "use_amp": str(vals.get("use_amp", "false")).lower() in ("true", "1", "on"),
            "amp_bf16": str(vals.get("bfloat16", "false")).lower() in ("true", "1", "on"),
            "losses": losses, "discs": discs,
            "use_checkpoint": bool(_ck is not None and hasattr(_ck, "currentText") and _ck.currentText() == "true"),
            "otf": str(vals.get("dataset_mode", "paired")).lower() in ("otf", "realesrgandataset", "bicubic"),
        }

    # ── collect + actions ──
    def _data(self) -> dict:
        d = {}
        d.update(getattr(self, "_resize_modes", None) or {})
        for key, w in self.fields.items():
            if isinstance(w, QCheckBox):
                d[key] = "true" if w.isChecked() else "false"
            elif isinstance(w, QComboBox):
                d[key] = w.currentText()
            elif isinstance(w, (LabeledSlider, _LogSlider, _SliderBox)):
                d[key] = w.value()
            elif isinstance(w, QLineEdit):
                d[key] = w.text()
        # merge the Dégradations tab settings (OTF generator) into the config
        try:
            if getattr(self, "deg_tab", None) is not None:
                for k, v in self.deg_tab.collect_degradations().items():
                    d.setdefault(k, v)
                # GAN-phase profile (phase degradations): config_handler keeps only the
                # keys that differ from the PSNR profile (deg_phase2).
                _p2 = self.deg_tab.collect_phase2()
                if _p2:
                    d["deg_phase2_data"] = _p2
                    d["deg_phase2_ramp"] = self.deg_tab.phase2_ramp()
        except Exception:
            pass
        # Phase losses (redux): the PSNR profile goes in the usual keys whatever profile
        # the panel shows; the GAN profile in loss_phase2_data (config_handler keeps the
        # weights that change, as loss_phase2).
        try:
            if getattr(self, "chk_lphase", None) is not None and self.chk_lphase.isChecked() \
                    and "Redux" in self._engine():
                _lon, _lramp, _lp1, _lp2 = self.lphase_profiles()
                for _k, (_c, _w) in _lp1.items():
                    d[_k] = "true" if _c else "false"
                    d[f"weight_{_k}"] = _w
                _l2 = {}
                for _k, (_c, _w) in _lp2.items():
                    _l2[_k] = "true" if _c else "false"
                    _l2[f"weight_{_k}"] = _w
                d["loss_phase2_data"] = _l2
                d["loss_phase2_ramp"] = _lramp
        except Exception:
            pass
        # --- ECO key mapping: the UI uses use_eco / pretrain_eco, but
        # config_handler.generate_config reads eco_mode + eco_pretrain_path.
        # Without this remap, enabling ECO produced a config with NO eco block.
        # (the "Mode ECO (Low Memory)" box of System Optimizations was the same ECO — NeoSR's
        # train.eco key — under a wrong "saves VRAM" text; merged into the Datasets ECO section)
        _eco_on = str(d.get("use_eco", "false")).lower() in ("true", "1", "yes", "on")
        d["eco_mode"] = "true" if _eco_on else "false"
        if d.get("pretrain_eco") and not d.get("eco_pretrain_path"):
            d["eco_pretrain_path"] = d["pretrain_eco"]
        # schedule_free: UI key is sched_free, generate_config reads schedule_free.
        if "schedule_free" not in d:
            d["schedule_free"] = str(d.get("sched_free", "false"))
        # aug_debug → moa_debug (generate_config reads moa_debug, not aug_debug).
        if "moa_debug" not in d:
            d["moa_debug"] = str(d.get("aug_debug", "false"))
        if "moa_debug_limit" not in d and d.get("aug_debug_limit"):
            d["moa_debug_limit"] = d["aug_debug_limit"]
        # MoA augmentations: UI uses use_mixup / use_mixup_prob, but generate_config reads
        # aug_mixup / prob_aug_mixup to build use_moa + moa_augs + moa_probs. Without this
        # remap, use_moa stayed false even with every augmentation checked.
        for _a in ("mixup", "cutmix", "resizemix", "cutblur"):
            d[f"aug_{_a}"] = str(d.get(f"use_{_a}", "false"))
            _p = d.get(f"use_{_a}_prob")
            if _p is not None:
                d[f"prob_aug_{_a}"] = _p
        # net_d_type : le combo contient le nom d'AFFICHAGE ("UNet") → convertir vers
        # la clé registry ("unet") attendue par le moteur, sinon KeyError au train GAN
        # ("No object named 'UNet' found in 'arch' registry!").
        if d.get("net_d_type"):
            d["net_d_type"] = self._disc_internal()
        if d.get("net_d2_type"):
            d["net_d2_type"] = self._disc2_internal()
        # tile_val → tile : le widget UI ("Tile (Val)") est enregistré sous tile_val,
        # mais generate_config lit "tile" (val.tile_size en Redux, val.tile en NeoSR).
        # Sans ce remap le tiling de validation était TOUJOURS ignoré (image entière).
        if "tile" not in d and d.get("tile_val"):
            d["tile"] = d["tile_val"]
        return d

    def _restart_tb(self):
        import sys as _sys, os as _os, subprocess
        port = self.fields["port_tb"].text() if "port_tb" in self.fields else "6006"
        py = self.settings.get("python_path", _sys.executable) or _sys.executable
        try:
            from src.core import engine_paths as _ep
            _venv_py = _ep.shared_venv_python()
            if _venv_py and _os.path.isfile(_venv_py):
                py = _venv_py
        except Exception:
            pass
        is_redux = "Redux" in self._engine()
        try:
            base = getattr(self.settings, "redux_path" if is_redux else "neosr_path", None)
        except Exception:
            base = None
        logdir = _os.path.join(base, "tb_logger") if (is_redux and base) else \
                 (_os.path.join(base, "experiments", "tb_logger") if base else _os.path.join(_os.getcwd(), "tb_logger"))
        try:
            if _sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/IM", "tensorboard.exe"], capture_output=True)
            _os.makedirs(logdir, exist_ok=True)
            cmd = [py, "-m", "tensorboard.main", "--logdir", logdir, f"--port={port}", "--bind_all"]
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if _sys.platform == "win32" else 0)
            QMessageBox.information(self, "TensorBoard", tr(f"Serveur relancé sur le port {port}.\nLogdir : {logdir}", f"Server restarted on port {port}.\nLogdir: {logdir}"))
        except Exception as e:
            QMessageBox.critical(self, tr("Erreur", "Error"), str(e))

    def _check_val_scale(self) -> bool:
        """Avertit si les LQ de validation ne sont pas les GT réduites du scale (set de
        restauration 1x utilisé pour un modèle x2…). Renvoie False si l'utilisateur annule."""
        try:
            from src.core.dataset_tools import val_scale_mismatch
            scale = int(float(self.fields["scale"].currentText().lstrip("×x"))) if "scale" in self.fields else 1
            gt = self.fields["val_gt"].text().strip() if "val_gt" in self.fields else ""
            lq = self.fields["val_lq"].text().strip() if "val_lq" in self.fields else ""
            if not gt or not lq:
                return True
            bad = val_scale_mismatch(gt, lq, scale)
            if not bad:
                return True
            name, g, l = bad[0]
            r = QMessageBox.warning(
                self, tr("Validation : tailles LQ/GT", "Validation: LQ/GT sizes"),
                tr(f"Les images LQ de validation ne sont pas les GT réduites ×{scale}.\n"
                   f"Exemple : {name} — GT {g[0]}×{g[1]}, LQ {l[0]}×{l[1]} "
                   f"(attendu {g[0] // scale}×{g[1] // scale}).\n\n"
                   f"La sortie sera remise à la taille de la GT avant le calcul : le PSNR/SSIM "
                   f"et le démarrage auto du GAN ne mesureront plus l'upscale ×{scale}.\n"
                   f"Générez les LQ avec Outils → Générateur LQ (réduction ×{scale}).\n\n"
                   f"Sauvegarder quand même ?",
                   f"The validation LQ images are not the GT downscaled ×{scale}.\n"
                   f"Example: {name} — GT {g[0]}×{g[1]}, LQ {l[0]}×{l[1]} "
                   f"(expected {g[0] // scale}×{g[1] // scale}).\n\n"
                   f"The output will be resized to the GT size before scoring: PSNR/SSIM "
                   f"and the automatic GAN start will no longer measure the ×{scale} upscale.\n"
                   f"Generate the LQ with Tools → LQ Generator (×{scale} downscale).\n\n"
                   f"Save anyway?"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            return r == QMessageBox.Yes
        except Exception:
            return True

    def _check_patch_min(self) -> bool:
        """Avertit si la taille de patch est sous le minimum recommandé pour l'arch choisie
        (transformers à fenêtre qui crashent sinon). Renvoie False si l'utilisateur annule."""
        try:
            from src.core.descriptions import arch_min_patch, arch_fixed_scale, arch_min_scale, arch_allowed_scales
            arch = self.arch_combo.currentText()
            sw = self.fields.get("scale")
            cur_scale = sw.currentText().strip() if sw is not None and hasattr(sw, "currentText") else ""
            # Scales EXPLICITEMENT supportées (ni min ni max simple, ex msdan = {2,3,4}
            # seulement → UnboundLocalError sur x_up pour scale 1 ou 8).
            allowed = arch_allowed_scales(arch, self._engine())
            if allowed and cur_scale and cur_scale.lstrip("×x").isdigit() and int(cur_scale.lstrip("×x")) not in allowed:
                allowed_str = "/".join(str(s) for s in sorted(allowed))
                r = QMessageBox.warning(
                    self, tr("Scale non supporté", "Unsupported scale"),
                    tr(f"L'architecture « {arch} » ne supporte que les scales {allowed_str} "
                       f"(refuse scale {cur_scale}).\n\n"
                       f"En l'état, l'entraînement plante au démarrage.\n"
                       f"Passe le Scale à {allowed_str}.\n\nSauvegarder quand même ?",
                       f"Architecture “{arch}” only supports scale {allowed_str} "
                       f"(rejects scale {cur_scale}).\n\n"
                       f"As-is, training crashes at start-up.\n"
                       f"Set Scale to {allowed_str}.\n\nSave anyway?"),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if r != QMessageBox.Yes:
                    return False
            # Scale MINIMUM (ex realcugan = 2/3/4 only → ValueError si scale 1).
            mns = arch_min_scale(arch)
            if mns and cur_scale and cur_scale.lstrip("×x").isdigit() and int(cur_scale.lstrip("×x")) < mns:
                r = QMessageBox.warning(
                    self, tr("Scale trop petit", "Scale too small"),
                    tr(f"L'architecture « {arch} » exige un scale ≥ {mns} (refuse scale {cur_scale}).\n"
                       f"Passe le Scale à {mns}, 3 ou 4.\n\nSauvegarder quand même ?",
                       f"Architecture “{arch}” requires scale ≥ {mns} (rejects scale {cur_scale}).\n"
                       f"Set Scale to {mns}, 3 or 4.\n\nSave anyway?"),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if r != QMessageBox.Yes:
                    return False
            # Scale imposé (ex cascadedgaze = restauration 1x only → assert scale==1 sinon crash).
            fx = arch_fixed_scale(arch)
            if fx:
                sw = self.fields.get("scale")
                cur_scale = sw.currentText().strip() if sw is not None and hasattr(sw, "currentText") else ""
                if cur_scale and cur_scale not in (str(fx), f"{fx}x", f"×{fx}"):
                    r = QMessageBox.warning(
                        self, tr("Scale incompatible", "Incompatible scale"),
                        tr(f"L'architecture « {arch} » fonctionne UNIQUEMENT en scale {fx}"
                           + (" (restauration, pas d'agrandissement)" if fx == 1 else "")
                           + f".\nTon réglage : scale {cur_scale}.\n\n"
                           f"En l'état, l'entraînement plante (assert scale == {fx}).\n"
                           f"Passe le Scale à {fx}.\n\nSauvegarder quand même ?",
                           f"Architecture “{arch}” works ONLY at scale {fx}"
                           + (" (restoration, no upscaling)" if fx == 1 else "")
                           + f".\nYour setting: scale {cur_scale}.\n\n"
                           f"As-is, training crashes (assert scale == {fx}).\n"
                           f"Set Scale to {fx}.\n\nSave anyway?"),
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if r != QMessageBox.Yes:
                        return False
            if not self._check_linedistiller_patch(cur_scale):
                return False
            if arch.startswith("fdat_rd_"):
                if not self._check_fdat_rd_patch(cur_scale):
                    return False
            mn = arch_min_patch(arch)
            if not mn:
                return True
            w = self.fields.get("patch_size")
            patch = int(w.value()) if w is not None and hasattr(w, "value") else 0
            if patch and patch < mn:
                r = QMessageBox.warning(
                    self, tr("Taille de patch trop petite", "Patch size too small"),
                    tr(f"L'architecture « {arch} » nécessite une taille de patch (lq_size) d'au moins "
                       f"~{mn} px.\nTon réglage actuel : {patch} px.\n\n"
                       f"En dessous, l'entraînement plante (erreur de forme / reshape).\n"
                       f"Recommandé : passe Patch Size à {mn} (multiple de 16).\n\n"
                       f"Sauvegarder quand même ?",
                       f"Architecture \u201c{arch}\u201d requires a patch size (lq_size) of at least "
                       f"~{mn} px.\nYour current setting: {patch} px.\n\n"
                       f"Below that, training crashes (shape/reshape error).\n"
                       f"Recommended: set Patch Size to {mn} (multiple of 16).\n\n"
                       f"Save anyway?"),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                return r == QMessageBox.Yes
        except Exception:
            pass
        return True

    def _linedistiller_on(self) -> bool:
        """LineDistiller enabled in the PSNR profile or in the GAN profile (phase losses):
        the engine builds it in both cases."""
        try:
            if "Redux" not in self._engine():
                return False
            d = self._data()
            if str(d.get("loss_linedistiller", "")).lower() == "true":
                return True
            p2 = d.get("loss_phase2_data")
            return isinstance(p2, dict) and str(p2.get("loss_linedistiller", "")).lower() == "true"
        except Exception:
            return False

    def _check_linedistiller_patch(self, cur_scale: str) -> bool:
        """LineDistiller (traiNNer-redux) runs a U-Net whose skip connections only line up
        when the HR patch (lq_size x scale) is a multiple of 16: HR 120 crashes at the first
        iteration (The size of tensor a (15) must match the size of tensor b (14)), HR 112
        and 128 work. Returns False if the user cancels."""
        if not self._linedistiller_on():
            return True
        s = cur_scale.lstrip("×x").rstrip("x")
        scale = int(s) if s.isdigit() else 2
        w = self.fields.get("patch_size")
        patch = int(w.value()) if w is not None and hasattr(w, "value") else 0
        if not patch or (patch * scale) % 16 == 0:
            return True
        step = 16 // math.gcd(16, max(1, scale))      # lq_size step: 8 at 2x, 16 at 1x/3x
        lo = max(step, patch // step * step)
        hi = lo + step
        r = QMessageBox.warning(
            self, tr("Patch incompatible avec LineDistiller", "Patch incompatible with LineDistiller"),
            tr(f"LineDistiller demande une taille HR (Patch Size × scale) multiple de 16.\n"
               f"Ton réglage : {patch} × {scale} = {patch * scale} px.\n\n"
               f"En l'état, l'entraînement plante dès la première itération "
               f"(The size of tensor a must match the size of tensor b).\n\n"
               f"Recommandé : Patch Size {lo} ou {hi}, ou désactive LineDistiller.\n\n"
               f"Sauvegarder quand même ?",
               f"LineDistiller needs an HR size (Patch Size × scale) that is a multiple of 16.\n"
               f"Your setting: {patch} × {scale} = {patch * scale} px.\n\n"
               f"As-is, training crashes at the first iteration "
               f"(The size of tensor a must match the size of tensor b).\n\n"
               f"Recommended: Patch Size {lo} or {hi}, or disable LineDistiller.\n\n"
               f"Save anyway?"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return r == QMessageBox.Yes

    def _check_fdat_rd_patch(self, cur_scale: str) -> bool:
        """FDAT-RD : avertit si lq_size n'est pas un multiple de l'alignement de l'archi
        (60 en 2x unshuffle, sinon 30). Sinon le modèle n'apprend jamais un vrai bord
        d'image et laisse une ligne sombre en bas/à droite hors de l'app (chaiNNer, ONNX)
        sur les tailles multiples de cet alignement (720×480, 1920×1080).
        Renvoie False si l'utilisateur annule."""
        from src.core.descriptions import fdat_rd_patch_multiple
        s = cur_scale.lstrip("×x").rstrip("x")
        scale = int(s) if s.isdigit() else 2
        uw = self.fields.get("dyn_unshuffle_mod")
        unshuffle = (uw.currentText().strip().lower() == "true"
                     if uw is not None and hasattr(uw, "currentText") else True)
        mult = fdat_rd_patch_multiple(scale, unshuffle)
        w = self.fields.get("patch_size")
        patch = int(w.value()) if w is not None and hasattr(w, "value") else 0
        if not patch or patch % mult == 0:
            return True
        step = mult
        if self._linedistiller_on():
            # LineDistiller also needs lq_size x scale % 16 == 0: suggest sizes that
            # satisfy both (2x full-res: lcm(30, 8) = 120).
            ld = 16 // math.gcd(16, max(1, scale))
            step = mult * ld // math.gcd(mult, ld)
        lo = max(step, patch // step * step)
        hi = lo + step if lo < patch else lo
        sugg = f"{hi}" if lo == hi else f"{hi} ({tr('ou', 'or')} {lo})"
        r = QMessageBox.warning(
            self, tr("Patch FDAT-RD non aligné", "FDAT-RD patch not aligned"),
            tr(f"FDAT-RD aligne son entrée sur un multiple de {mult} px "
               f"(scale {scale}, unshuffle_mod {'true' if unshuffle else 'false'}).\n"
               f"Ton Patch Size : {patch} px.\n\n"
               f"Avec un patch non multiple de {mult}, le modèle n'apprend jamais un vrai "
               f"bord d'image : hors de l'app (chaiNNer, ONNX), les images dont la taille est "
               f"un multiple de {mult} (720×480, 1920×1080…) auront une ligne sombre en bas "
               f"et à droite. Dans l'app, l'archi corrige ce défaut.\n\n"
               f"Recommandé : Patch Size {sugg}.\n\nSauvegarder quand même ?",
               f"FDAT-RD aligns its input to a multiple of {mult} px "
               f"(scale {scale}, unshuffle_mod {'true' if unshuffle else 'false'}).\n"
               f"Your Patch Size: {patch} px.\n\n"
               f"With a patch that is not a multiple of {mult}, the model never learns a real "
               f"image border: outside the app (chaiNNer, ONNX), images whose size is a "
               f"multiple of {mult} (720×480, 1920×1080…) get a dark line on the bottom and "
               f"right edge. Inside the app, the arch corrects this defect.\n\n"
               f"Recommended: Patch Size {sugg}.\n\nSave anyway?"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return r == QMessageBox.Yes

    def _check_eco_teacher(self) -> bool:
        """Avertit si ECO est activé sans checkpoint teacher (.pth) renseigné.

        traiNNer-redux exige network_g_teacher + path.pretrain_network_g_teacher
        quand train.eco.enabled est vrai (sr_model.py::__init__) : sans checkpoint,
        l'entraînement plante immédiatement avec
        AssertionError: pretrain_network_g_teacher must be defined when using
        network_g_teacher. Renvoie False si l'utilisateur annule."""
        try:
            eco_on = self.fields.get("use_eco")
            if eco_on is None or not hasattr(eco_on, "isChecked") or not eco_on.isChecked():
                return True
            if "redux" not in self.engine_combo.currentText().lower():
                return True  # NeoSR ECO has no separate teacher
            pretrain = self.fields.get("pretrain_eco")
            has_path = bool(pretrain is not None and hasattr(pretrain, "text") and pretrain.text().strip())
            if not has_path:
                r = QMessageBox.warning(
                    self, tr("ECO sans teacher", "ECO without teacher"),
                    tr("ECO est activé mais aucun checkpoint teacher (.pth) n'est renseigné.\n\n"
                       "En l'état, l'entraînement plante au démarrage "
                       "(AssertionError: pretrain_network_g_teacher must be defined).\n"
                       "Renseigne « Pretrain ECO (.pth) » ou désactive ECO.\n\n"
                       "Sauvegarder quand même ?",
                       "ECO is enabled but no teacher checkpoint (.pth) is set.\n\n"
                       "As-is, training crashes at startup "
                       "(AssertionError: pretrain_network_g_teacher must be defined).\n"
                       "Set “Pretrain ECO (.pth)” or disable ECO.\n\n"
                       "Save anyway?"),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if r != QMessageBox.Yes:
                    return False
        except Exception:
            pass
        return True

    def _save(self, quiet=False):
        # Default to the engine's Option Custom subfolder + pre-fill the experiment name
        # and the right extension (.yml for Redux, .toml for NeoSR) — like the CTk app.
        if not quiet and not self._check_patch_min():
            return None
        if not quiet and not self._check_val_scale():
            return None
        if not quiet and not self._check_eco_teacher():
            return None
        is_redux = "Redux" in self._engine()
        ext = ".yml" if is_redux else ".toml"
        name = (self.fields["name"].text().strip() if "name" in self.fields else "") or "experiment"
        try:
            from src.core.engine_paths import base_engine_path
            base = os.path.join(base_engine_path(), "Option Custom",
                                "trainner_redux" if is_redux else "neosr")
            os.makedirs(base, exist_ok=True)
        except Exception:
            base = os.path.expanduser("~")
        default_path = os.path.join(base, name + ext)
        flt = "Config Redux (*.yml *.yaml)" if is_redux else "Config NeoSR (*.toml)"
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer la config", default_path, flt)
        if not path:
            return None
        try:
            from src.core.config_handler import ConfigHandler
            ch = ConfigHandler()
            ch.set_engine("TraiNNer-Redux" if is_redux else "NeoSR")
            ok, msg = ch.generate_config(self._data(), path)
            # Sidecar with the exact UI state → lossless reload (MoA prob normalization is
            # lossy in the engine config). NOT embedded in the .yml because Redux options use
            # msgspec StrictStruct(forbid_unknown_fields=True) and would reject an extra key.
            if ok:
                self._write_ui_sidecar(path)
            if quiet and ok:
                return path
            QMessageBox.information(self, "Config", ("✅ " if ok else "❌ ") + str(msg))
            return path if ok else None
        except Exception as e:
            QMessageBox.critical(self, tr("Erreur", "Error"), tr(f"Échec : {e}", f"Failed: {e}"))
            return None

    def _save_test(self):
        """Save the config, push it to the Entraînement tab with the engine-matching
        train script, switch there and launch — the « Sauver & Tester » one-click flow."""
        if not self._check_patch_min():   # avertit avant de lancer (patch trop petit = crash)
            return
        if not self._check_val_scale():
            return
        path = self._save(quiet=True)
        if not path:
            return
        try:
            from qt_ui.tabs.run_tab import RunTab
            mw = self.window()
            rt = mw.findChild(RunTab)
            if rt is None:
                QMessageBox.information(self, tr("Tester", "Test"), tr(f"Config enregistrée :\n{path}\n\n"
                                       "Onglet Entraînement introuvable — lance manuellement.",
                                       f"Config saved:\n{path}\n\n"
                                       "Training tab not found — launch manually.")); return
            rt.entries["config_path"].setText(path)
            # the train script MUST match the config's engine (.yml→Redux, .toml→NeoSR),
            # otherwise redux's msgspec.yaml parser chokes on a NeoSR .toml (and vice versa).
            is_redux = "Redux" in self._engine()
            try:
                from src.core import engine_paths as ep
                root = ep.redux_path() if is_redux else ep.neosr_path()
                ts = os.path.join(root, "train.py")
                if os.path.isfile(ts):
                    rt.entries["script_path"].setText(ts)
            except Exception:
                pass
            # switch the main nav to the page hosting the RunTab, then start
            stack = getattr(mw, "stack", None)
            if stack is not None:
                for i in range(stack.count()):
                    pg = stack.widget(i)
                    if pg is rt or pg.findChild(RunTab) is rt:
                        mw._select(i); break
            rt._start()
        except Exception as e:
            QMessageBox.critical(self, "Tester", tr(f"Config enregistrée mais lancement échoué : {e}", f"Config saved but launch failed: {e}"))

    def _load(self):
        # open in the engine's Option Custom subfolder by default (like the CTk app)
        is_redux = "Redux" in self._engine()
        start_dir = ""
        try:
            from src.core.engine_paths import base_engine_path
            cand = os.path.join(base_engine_path(), "Option Custom",
                                "trainner_redux" if is_redux else "neosr")
            start_dir = cand if os.path.isdir(cand) else os.path.dirname(cand)
        except Exception:
            pass
        path, _ = QFileDialog.getOpenFileName(self, tr("Charger config", "Load config"), start_dir,
                                              tr("Config (*.yml *.yaml *.toml);;Tous (*.*)", "Config (*.yml *.yaml *.toml);;All (*.*)"))
        if not path:
            return
        try:
            from src.core.config_handler import ConfigHandler
            ok, flat = ConfigHandler().load_config(path)
        except Exception as e:
            QMessageBox.critical(self, tr("Charger", "Load"), tr(f"Échec : {e}", f"Failed: {e}")); return
        if not ok:
            QMessageBox.critical(self, tr("Charger", "Load"), str(flat)); return
        self._populate(flat)
        self._read_ui_sidecar(path)  # exact UI state overlay if a .uss.json sidecar exists
        QMessageBox.information(self, tr("Charger", "Load"), tr(f"Config chargée : {os.path.basename(path)}", f"Config loaded: {os.path.basename(path)}"))

    # Config keys with no widget: kept from the loaded config and sent back on save.
    _RESIZE_MODE_KEYS = tuple(f"resize_mode_{k}{s}" for s in ("", "2", "3") for k in ("list", "prob")) \
        + ("poisson_scale_range", "poisson_scale_range2")

    def _populate(self, flat: dict):
        """Fill the form from a flat config dict (symmetric to generate_config)."""
        # No widget for the resize modes: keep the loaded values (hidden passthrough) so a
        # save does not reset them to the default mode mix.
        self._resize_modes = {k: flat[k] for k in self._RESIZE_MODE_KEYS if flat.get(k)}
        # 1. engine first (rebuilds arch/disc/optim/loss lists)
        eng = flat.get("_detected_engine") or flat.get("engine")
        if eng:
            self.engine_combo.blockSignals(True)
            self.engine_combo.setCurrentText("TraiNNer-Redux" if "redux" in str(eng).lower() else "NeoSR")
            self.engine_combo.blockSignals(False)
            self._on_engine_change()
        # 2. arch + disc (rebuild dynamic param boxes)
        if flat.get("arch"):
            self.arch_combo.setCurrentText(str(flat["arch"]))
        if flat.get("net_d_type") and hasattr(self, "disc_combo"):
            try:
                from src.core.descriptions import DISC_DISPLAY_NAMES
                self.disc_combo.setCurrentText(DISC_DISPLAY_NAMES.get(str(flat["net_d_type"]), str(flat["net_d_type"])))
            except Exception:
                pass
        if flat.get("net_d2_type") and hasattr(self, "disc2_combo"):
            try:
                from src.core.descriptions import DISC_DISPLAY_NAMES
                self.disc2_combo.setCurrentText(DISC_DISPLAY_NAMES.get(str(flat["net_d2_type"]), str(flat["net_d2_type"])))
                flat.setdefault("use_disc2", "true")
            except Exception:
                pass
        # reverse of the _data() key remaps so loading restores the toggles
        if "eco_mode" in flat and "use_eco" not in flat:
            flat["use_eco"] = flat["eco_mode"]
        if flat.get("eco_pretrain_path") and "pretrain_eco" not in flat:
            flat["pretrain_eco"] = flat["eco_pretrain_path"]
        if "schedule_free" in flat and "sched_free" not in flat:
            flat["sched_free"] = flat["schedule_free"]
        # moa_debug (config) → aug_debug (UI field)
        if "moa_debug" in flat and "aug_debug" not in flat:
            flat["aug_debug"] = flat["moa_debug"]
        if "moa_debug_limit" in flat and "aug_debug_limit" not in flat:
            flat["aug_debug_limit"] = flat["moa_debug_limit"]
        # MoA augmentation checkboxes are stored as a moa_augs list in the config
        _augs = flat.get("moa_augs") or flat.get("aug_list") or []
        if isinstance(_augs, str):
            _augs = [a.strip() for a in _augs.replace(",", " ").split()]
        if _augs:
            _augs_l = [str(a).lower() for a in _augs]
            for _aug in ("mixup", "cutmix", "resizemix", "cutblur"):
                flat.setdefault(f"use_{_aug}", "true" if _aug in _augs_l else "false")
        # 3. all remaining fields by type
        _applied = 0
        for key, val in flat.items():
            if self._set_field(key, val):
                _applied += 1
        # 4. degradations live in the separate Dégradations tab → push them there too
        # (counted into _applied too — they were silently excluded before, making the
        # "N champs appliqués" message understate what actually landed, 2026-09-17 fix)
        try:
            if getattr(self, "deg_tab", None) is not None:
                _applied += sum(1 for k in flat if k in self.deg_tab._setters)
                self.deg_tab.reset_phase()
                self.deg_tab.apply_degradations(flat)
                if isinstance(flat.get("_deg_phase2_flat"), dict):
                    self.deg_tab.apply_phase2(flat["_deg_phase2_flat"], flat.get("deg_phase2_ramp"))
        except Exception:
            pass
        try:
            self.reset_lphase()
            if isinstance(flat.get("_loss_phase2_flat"), dict):
                self.apply_lphase(self._lphase_from_flat(flat["_loss_phase2_flat"]),
                                  flat.get("loss_phase2_ramp"))
        except Exception:
            pass
        self._schedule_vram()
        self._last_applied_count = _applied
        return _applied

    @staticmethod
    def _sidecar_path(config_path: str) -> str:
        return str(config_path) + ".uss.json"

    def _write_ui_sidecar(self, config_path: str):
        """Persist the exact UI state next to the config (lossless reload)."""
        try:
            import json
            with open(self._sidecar_path(config_path), "w", encoding="utf-8") as f:
                json.dump(self._ui_state(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _read_ui_sidecar(self, config_path: str):
        """Overlay the exact UI state from a .uss.json sidecar, if present. Applied after
        _populate so sliders/values match exactly what was saved (engine config is lossy
        for MoA probs after normalization — the sidecar is not)."""
        try:
            import json
            sp = self._sidecar_path(config_path)
            if not os.path.exists(sp):
                return
            with open(sp, "r", encoding="utf-8") as f:
                ui = json.load(f)
            if not isinstance(ui, dict):
                return
            for key, val in ui.items():
                if key in ("_deg_enabled", "_deg_phase2", "_loss_phase2"):
                    continue
                self._set_field(key, val)
            if getattr(self, "deg_tab", None) is not None:
                self.deg_tab.apply_degradations(ui)
                # restore exact enable state (overrides apply_degradations' prob>0 heuristic)
                deg_en = ui.get("_deg_enabled")
                if isinstance(deg_en, dict):
                    for k, chk in self.deg_tab._checks.items():
                        if k in deg_en:
                            chk.setChecked(bool(deg_en[k]))
                _p2 = ui.get("_deg_phase2")
                if isinstance(_p2, dict) and isinstance(_p2.get("profile"), dict):
                    self.deg_tab.apply_phase2_snapshot(_p2["profile"], _p2.get("ramp"), _p2.get("on", True))
            _lp = ui.get("_loss_phase2")
            if isinstance(_lp, dict) and isinstance(_lp.get("profile"), dict):
                self.apply_lphase(_lp["profile"], _lp.get("ramp"), _lp.get("on", True))
            self._schedule_vram()
        except Exception:
            pass

    def _set_field(self, key, val) -> bool:
        """Set one self.fields widget from a raw value (handles every widget type).
        Returns True iff a matching widget was actually found and set — callers use this
        to report a real "N fields applied" count instead of len(flat), which previously
        counted every key in the source dict even when most had no matching widget
        (silently dropped, but the UI message said "applied") (2026-09-17 fix)."""
        w = self.fields.get(key)
        if w is None:
            return False
        # YAML bools load as Python False/True: str() gives "False", which matched no
        # "false"/"true" item, so the combobox silently kept its default.
        if isinstance(val, bool):
            val = "true" if val else "false"
        try:
            if isinstance(w, QCheckBox):
                w.setChecked(str(val).lower() in ("true", "1", "on", "yes"))
            elif isinstance(w, QComboBox):
                idx = w.findText(str(val))
                if idx < 0:
                    low = str(val).strip().lower()
                    idx = next((i for i in range(w.count()) if w.itemText(i).strip().lower() == low), -1)
                if idx >= 0:
                    w.setCurrentIndex(idx)
                else:
                    w.setCurrentText(str(val))
            elif isinstance(w, (LabeledSlider, _LogSlider, _SliderBox)):
                w.set_value(float(val)) if hasattr(w, "set_value") else w.setValue(float(val))
            elif isinstance(w, QLineEdit):
                w.setText(str(val))
            else:
                return False
            return True
        except Exception:
            return False

    def _ui_state(self) -> dict:
        """Capture the exact current UI values (raw, by field key) so a saved config
        can be reloaded losslessly. Includes the Dégradations tab values."""
        st = {}
        for key, w in self.fields.items():
            try:
                if isinstance(w, QCheckBox):
                    st[key] = bool(w.isChecked())
                elif isinstance(w, QComboBox):
                    st[key] = w.currentText()
                elif isinstance(w, (LabeledSlider, _LogSlider, _SliderBox)):
                    st[key] = float(w.value())
                elif isinstance(w, QLineEdit):
                    st[key] = w.text()
            except Exception:
                pass
        try:
            if getattr(self, "deg_tab", None) is not None:
                for dk, dv in self.deg_tab.collect_degradations().items():
                    st[dk] = dv
                _on, _ramp, _snap = self.deg_tab.phase2_state()
                if _snap:
                    st["_deg_phase2"] = {"on": _on, "ramp": _ramp, "profile": _snap}
                # explicit enable state per degradation (a slider value alone can't tell
                # an enabled 0.15 from a disabled 0.15)
                st["_deg_enabled"] = {k: bool(chk.isChecked())
                                      for k, chk in self.deg_tab._checks.items()}
        except Exception:
            pass
        try:
            if getattr(self, "chk_lphase", None) is not None:
                _lon, _lramp, _lp1, _lp2 = self.lphase_profiles()
                if _lon or self._lphase_profiles[1] is not None:
                    for _k, (_c, _w) in _lp1.items():
                        st[_k] = bool(_c)
                        st[f"weight_{_k}"] = _w
                    st["_loss_phase2"] = {"on": _lon, "ramp": _lramp, "profile": _lp2}
        except Exception:
            pass
        return st
