"""
post_process.py — Post-Processing tab (PySide6).

Spatial chain (Deband · Line darken · Line thin · Colour · Resize · Sharpen ·
Edge cleanup) runs live on a single preview image via src.core.post_process_chain.
Temporal modules (Temporal Fix · Undistort) apply only in batch (multi-frame) via
src.core.post_proc_session. Each module is a collapsible card with an enable toggle
and a per-module Reset. The live preview is debounced so heavy sliders don't freeze.
"""
from __future__ import annotations

import io
import os
import threading

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QCheckBox,
    QComboBox, QFrame, QScrollArea, QSplitter, QFileDialog, QSizePolicy, QLineEdit,
    QProgressBar, QPlainTextEdit, QToolButton,
)
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon
from PySide6.QtCore import Qt, QTimer, QThread, Signal

import os as _os
_ASSETS = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "assets")
_GEAR_ICON = _os.path.join(_ASSETS, "gear.svg")
_ARROW_DOWN = _os.path.join(_ASSETS, "arrow_down_g.svg")
_ARROW_RIGHT = _os.path.join(_ASSETS, "arrow_right.svg")
_CW = 150  # uniform combo width across all modules

from qt_ui.widgets import LabeledSlider
from qt_ui.i18n import tr

_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}


def pil_to_pixmap(im) -> QPixmap:
    im = im.convert("RGB")
    qim = QImage(im.tobytes("raw", "RGB"), im.width, im.height, im.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qim.copy())


class _PreviewView(QLabel):
    """Preview: hold-left-click shows the original, mouse wheel zooms."""
    def __init__(self):
        super().__init__(tr("Ouvre une image pour commencer.", "Open an image to start."))
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(380, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("QLabel{ background:#0b0d12; border-radius:10px; }")
        self.setToolTip(tr("Clic gauche maintenu = image source · molette = zoom", "Hold left click = source image · wheel = zoom"))
        self._result_pm = None; self._src_pm = None
        self._zoom = 1.0; self._showing_src = False

    def set_images(self, result_pm, src_pm):
        self._result_pm = result_pm; self._src_pm = src_pm; self._render()

    def clear_images(self):
        self._result_pm = None; self._src_pm = None

    def _active(self):
        return self._src_pm if (self._showing_src and self._src_pm) else self._result_pm

    def _render(self):
        pm = self._active()
        if pm is None:
            return
        w = max(50, int(self.width() * self._zoom)); h = max(50, int(self.height() * self._zoom))
        super().setPixmap(pm.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def wheelEvent(self, e):
        if self._active() is None:
            return
        self._zoom = max(0.2, min(8.0, self._zoom * (1.15 if e.angleDelta().y() > 0 else 1 / 1.15)))
        self._render()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._src_pm is not None:
            self._showing_src = True; self._render()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._showing_src = False; self._render()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._render()


class _Module(QFrame):
    """Collapsible card: header (enable · title · ⚙ · Reset · ▼/▶) + hideable body.
    Transparent so no dark box sits behind the text/sliders. Click the title (or the
    ▼/▶ button) to collapse/expand."""
    _settings_ref = None  # set by PostProcessTab before building modules

    def __init__(self, title, on_change, reset_fn, gear_fn=None, key=None, settings=None):
        super().__init__()
        self._settings = settings if settings is not None else _Module._settings_ref
        self._key = key or "".join(c.lower() if c.isalnum() else "" for c in title)[:24]
        settings = self._settings; key = self._key
        self.setObjectName("ppmod")
        self.setStyleSheet("QFrame#ppmod{background:transparent; border:1px solid #3a3a4a; border-radius:8px;}")
        outer = QVBoxLayout(self); outer.setContentsMargins(8, 6, 8, 6); outer.setSpacing(4)
        hdr = QHBoxLayout(); hdr.setSpacing(6)
        self.enable = QCheckBox(); self.enable.stateChanged.connect(on_change)
        hdr.addWidget(self.enable)
        self._title = QPushButton(title); self._title.setFlat(True); self._title.setCursor(Qt.PointingHandCursor)
        self._title.setStyleSheet("QPushButton{font-weight:700; background:transparent; border:none; text-align:left; padding:0;}")
        self._title.clicked.connect(self._collapse); hdr.addWidget(self._title)
        hdr.addStretch(1)
        if gear_fn is not None:
            g = QPushButton(); g.setFixedSize(26, 22); g.setToolTip(tr('Réglages avancés du module', 'Advanced module settings'))
            if _os.path.isfile(_GEAR_ICON):
                g.setIcon(QIcon(_GEAR_ICON))
            else:
                g.setText(tr("Régl.", "Set."))
            g.setStyleSheet('background:#2ecc71; border:none; border-radius:4px;')
            g.clicked.connect(gear_fn); hdr.addWidget(g)
        rst = QPushButton("Reset"); rst.setFixedHeight(22)
        rst.setToolTip(tr('Réinitialiser ce module à ses valeurs par défaut', 'Reset this module to defaults'))
        rst.setStyleSheet('background:#444; color:#fff; border:none; border-radius:4px; padding:1px 10px; font-size:11px;')
        rst.clicked.connect(reset_fn); hdr.addWidget(rst)
        # Collapse toggle via SVG icon (renders immediately, no resize needed).
        self._toggle = QPushButton(); self._toggle.setFixedSize(24, 22); self._toggle.setCursor(Qt.PointingHandCursor)
        self._toggle.setToolTip(tr("Replier / déplier ce module", "Collapse / expand this module"))
        self._toggle.setStyleSheet('QPushButton{background:transparent; border:none;}')
        self._toggle.clicked.connect(self._collapse); hdr.addWidget(self._toggle)
        hdr.addSpacing(8)  # keep Reset + arrow off the scrollbar edge (visible without resizing)
        outer.addLayout(hdr)
        self.body = QWidget()
        self.bv = QVBoxLayout(self.body); self.bv.setContentsMargins(2, 2, 2, 2); self.bv.setSpacing(3)
        outer.addWidget(self.body)
        # restore persisted collapse state
        collapsed = False
        if settings is not None and key:
            v = settings.get(f"pp_collapsed_{key}", False)
            collapsed = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        self._collapsed = bool(collapsed)
        self.body.setVisible(not collapsed)
        self._set_arrow(not collapsed)

    def _set_arrow(self, expanded):
        ic = _ARROW_DOWN if expanded else _ARROW_RIGHT
        if _os.path.isfile(ic):
            self._toggle.setIcon(QIcon(ic)); self._toggle.setText("")
        else:
            self._toggle.setText("▾" if expanded else "▸")  # small triangles = monochrome (no emoji)

    def _collapse(self):
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool):
        self._collapsed = bool(collapsed)
        self.body.setVisible(not collapsed); self._set_arrow(not collapsed)
        if self._settings is not None and self._key:
            self._settings.set(f"pp_collapsed_{self._key}", collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def isChecked(self):
        return self.enable.isChecked()


class _BatchWorker(QThread):
    line = Signal(str)
    progress = Signal(float)
    preview = Signal(bytes, bytes)
    done = Signal(int, int)

    def __init__(self, src, out, settings, tf_cfg, out_opts, stop_event):
        super().__init__(); self.src, self.out, self.settings = src, out, settings
        self.tf_cfg = tf_cfg; self.out_opts = out_opts; self.stop = stop_event

    def run(self):
        # F40: every exit emits done (the buttons stayed locked when the output folder
        # could not be created or listed)
        self.notes = []
        try:
            self._run()
        except Exception as e:
            self.line.emit(f"❌ {e}")
            self.notes.append(str(e))
            self.done.emit(0, 0)

    def _run(self):
        from PIL import Image
        from src.core.post_process_chain import apply_chain
        os.makedirs(self.out, exist_ok=True)
        if os.path.isfile(self.src):
            files = [self.src]
        else:
            files = [os.path.join(self.src, f) for f in sorted(os.listdir(self.src))
                     if os.path.splitext(f)[1].lower() in _IMG_EXT]
        total = len(files)
        if total == 0:
            self.line.emit(tr("Aucune image trouvée.", "No image found.")); self.done.emit(0, 0); return

        # Optional temporal post-proc subprocess (Temporal Fix / Undistort).
        _pp = None; _tf_active = _ud_active = False
        cfg = self.tf_cfg
        if cfg.get("tf") or cfg.get("ud"):
            try:
                from src.core.quick_upscale import _TRAINNER_VENV_PY
                from src.core.post_proc_session import PostProcSession
                _pp = PostProcSession(venv_py=_TRAINNER_VENV_PY, log=self.line.emit)
                if _pp.start():
                    if cfg.get("tf"):
                        try:
                            _pp.init_tf(mode=cfg["tf_mode"], strength=cfg["tf_str"], window=cfg["tf_win"], precision=cfg["tf_prec"])
                            _tf_active = True; self.line.emit("[TemporalFix] " + tr("activé", "enabled"))
                        except Exception as e:
                            self.notes.append(f"Temporal Fix : {e}"); self.line.emit(f"❌ [TemporalFix] {e}")
                    if cfg.get("ud"):
                        try:
                            _pp.init_ud(mode=cfg["ud_mode"], strength=cfg["ud_str"], window=cfg["ud_win"],
                                        precision=cfg.get("ud_prec", cfg["tf_prec"]))
                            _ud_active = True; self.line.emit("[Undistort] " + tr("activé", "enabled"))
                        except Exception as e:
                            self.notes.append(f"Undistort : {e}"); self.line.emit(f"❌ [Undistort] {e}")
                    _pp.set_quality(self.out_opts.get("quality", 95))
                    if _tf_active and _ud_active:
                        _pp.chain_ud_tf(True)   # F33: UD output feeds TF
                else:
                    _pp = None
                    self.notes.append(tr("post-proc temporel non démarré", "temporal post-proc not started"))
            except Exception as e:
                self.notes.append(str(e))
                self.line.emit(tr(f"[Post-proc temporel] indisponible ({e})", f"[Temporal post-proc] unavailable ({e})")); _pp = None

        done = 0
        try:
            for i, fp in enumerate(files):
                if self.stop.is_set():
                    self.line.emit(tr("Arrêt demandé.", "Stop requested.")); break
                try:
                    img = Image.open(fp).convert("RGB")
                    aa = self.settings.get("_aa_vs")
                    chain_st = self.settings
                    if aa:
                        import tempfile as _tf
                        from src.core import vs_aa
                        ti = _tf.mktemp(suffix=".png"); to = _tf.mktemp(suffix=".png")
                        ok = False
                        try:
                            img.save(ti)
                            ok, _m = vs_aa.antialias_image(ti, to, aa["method"], aa["device"],
                                                           aa["strength"], aa["passes"], log=self.line.emit)
                            if ok and os.path.isfile(to):
                                img = Image.open(to).convert("RGB")
                        finally:
                            for _p in (ti, to):
                                try:
                                    os.remove(_p)
                                except Exception:
                                    pass
                        # VapourSynth/StaxRip absent (or failed) — fall back to the
                        # native ONNX-free nnedi3 port (aa_nnedi3_native.py). Only
                        # covers "nnedi3", not "eedi3" (different, non-neural algorithm,
                        # not ported — still requires VapourSynth).
                        native_ok = False
                        if not ok and "eedi3" not in aa["method"].lower():
                            try:
                                import numpy as _np
                                from src.core.aa_nnedi3_native import antialias_image_native
                                self.line.emit(tr("VapourSynth indisponible — repli nnedi3 natif (plus lent, sans VapourSynth).",
                                                  "VapourSynth unavailable — falling back to native nnedi3 (slower, no VapourSynth)."))
                                arr = _np.asarray(img, dtype=_np.float32) / 255.0
                                arr = antialias_image_native(arr, strength=aa["strength"], passes=aa["passes"])
                                img = Image.fromarray((arr * 255.0).clip(0, 255).astype(_np.uint8), mode="RGB")
                                native_ok = True
                            except Exception as _e:
                                self.line.emit(f"[AA natif] {_e}")
                        if not ok and not native_ok:
                            # F35: no VapourSynth result and no native fallback (EEDI3, or the
                            # fallback failed): the directional AA of the chain runs instead
                            # of no anti-aliasing at all
                            chain_st = dict(self.settings); chain_st["aa_enabled"] = True
                            _n = tr("AA VapourSynth échoué — AA directionnel de la chaîne utilisé",
                                    "VapourSynth AA failed — the chain's directional AA used")
                            if _n not in self.notes:
                                self.notes.append(_n); self.line.emit("⚠ " + _n)
                    out_img = apply_chain(img.copy(), chain_st)
                    oo = self.out_opts
                    op = os.path.join(self.out, os.path.splitext(os.path.basename(fp))[0] + oo["ext"])
                    save_kw = {}
                    if oo["bit16"]:
                        # F36: a real 16-bit file (Pillow has no RGB uint16 mode). The chain
                        # works in 8 bits: the 16-bit file holds 8-bit precision.
                        import numpy as _np
                        from src.core.image_io import write16
                        write16(op, _np.asarray(out_img.convert("RGB"), _np.float32) / 255.0)
                    else:
                        if oo["fmt"] in ("JPEG", "WEBP"):
                            save_kw["quality"] = oo["quality"]
                        out_img.save(op, **save_kw)
                    if _ud_active:
                        _pp.push_ud(op)
                    if _tf_active and not _ud_active:
                        _pp.push_tf(op)
                    if i % max(1, total // 20) == 0:
                        b1 = io.BytesIO(); img.save(b1, "PNG"); b2 = io.BytesIO(); out_img.save(b2, "PNG")
                        self.preview.emit(b1.getvalue(), b2.getvalue())
                    done += 1
                    self.line.emit(f"[{i+1}/{total}] {os.path.basename(fp)}")
                except Exception as e:
                    self.line.emit(f"  ⚠ {os.path.basename(fp)}: {e}")
                self.progress.emit((i + 1) / total)
        finally:
            if _pp:
                try:
                    if _ud_active:
                        _pp.flush_ud()
                    if _tf_active:
                        _pp.flush_tf()
                except Exception as e:
                    self.notes.append(f"flush : {e}"); self.line.emit(f"❌ flush : {e}")
                try:
                    _pp.stop()
                except Exception:
                    pass
        self.done.emit(done, total)


class PostProcessTab(QWidget):
    # Sidebar-badge hooks (qt_ui/tabs/tools_tab.py) — see QuickUpscalePage.
    toolProgress = Signal(int)
    toolFinished = Signal(bool)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._src_img = None
        self._result = None
        self._worker = None
        self._stop = None

        # Temporal Fix / Undistort settings (shared dict with the dialogs).
        self._tf = {
            "strength": float(settings.get("pp_tf_strength", 0.5)),
            "window": int(settings.get("pp_tf_window", 7)),
            "precision": str(settings.get("pp_tf_precision", "float32")),
            "tempfix_mode": str(settings.get("pp_tf_mode", "classic")),
            "undistort_strength": float(settings.get("pp_ud_strength", 0.35)),
            "undistort_window": int(settings.get("pp_ud_window", 5)),
            "undistort_precision": str(settings.get("pp_ud_precision", "float32")),
            "undistort_mode": str(settings.get("pp_ud_mode", "classic")),
        }

        self._loading = True  # block persistence writes while building/restoring
        self._debounce = QTimer(self); self._debounce.setSingleShot(True)
        self._debounce.setInterval(320)  # let heavy sliders settle before re-rendering
        self._debounce.timeout.connect(self._render)

        root = QVBoxLayout(self); root.setContentsMargins(16, 12, 16, 12); root.setSpacing(6)

        # ── header + GPU panel ──
        head = QHBoxLayout()
        lc = QVBoxLayout(); lc.setSpacing(0)
        t = QLabel("Post Processing"); t.setObjectName("title"); lc.addWidget(t)
        sub = QLabel(tr("Chaîne de post-traitement : Temporal Fix, Undistort, couleur, netteté, redimensionnement.",
                        "Post-processing chain: Temporal Fix, Undistort, color, sharpening, resizing."))
        sub.setStyleSheet("color:#888;"); lc.addWidget(sub)
        head.addLayout(lc); head.addStretch(1)
        try:
            from qt_ui.tabs.tools_pages.quick_upscale import _GpuMiniPanel
            self.gpu_panel = _GpuMiniPanel(); head.addWidget(self.gpu_panel, 0, Qt.AlignTop | Qt.AlignRight)
        except Exception:
            self.gpu_panel = None
        root.addLayout(head)

        # ── paths + options (above the image) ──
        g = QGridLayout(); g.setVerticalSpacing(4); g.setHorizontalSpacing(6); root.addLayout(g)
        g.addWidget(QLabel(tr("Source (image ou dossier) :", "Source (image or folder):")), 0, 0)
        self.src_edit = QLineEdit(settings.get("pp_src", "") or ""); g.addWidget(self.src_edit, 0, 1)
        self.src_edit.editingFinished.connect(self._preview_from_source)
        srow = QWidget(); sh = QHBoxLayout(srow); sh.setContentsMargins(0, 0, 0, 0); sh.setSpacing(4)
        bi = QPushButton(tr("Image", "Image")); bi.setFixedSize(70, 28); bi.clicked.connect(self._pick_image)
        bf = QPushButton(tr("Dossier", "Folder")); bf.setFixedSize(80, 28); bf.clicked.connect(self._pick_folder)
        sh.addStretch(1); sh.addWidget(bi); sh.addWidget(bf); g.addWidget(srow, 0, 2)
        g.addWidget(QLabel(tr("Dossier sortie :", "Output folder:")), 1, 0)
        self.out_edit = QLineEdit(settings.get("pp_out", "") or ""); g.addWidget(self.out_edit, 1, 1)
        bo = QPushButton("…"); bo.setObjectName("ghost"); bo.setFixedSize(40, 28)
        bo.clicked.connect(lambda: self._pick_dir(self.out_edit)); g.addWidget(bo, 1, 2, Qt.AlignRight)
        g.setColumnStretch(1, 1); g.setColumnMinimumWidth(2, 160)

        opt = QHBoxLayout(); root.addLayout(opt); opt.addStretch(1)
        self.same_folder = QCheckBox(tr("Même dossier que source", "Same folder as source")); self.same_folder.setChecked(self._b("pp_same", False))
        self.same_folder.toggled.connect(self._on_same_folder)
        opt.addWidget(self.same_folder)
        self.subfolder = QCheckBox(tr('Sous-dossier "postproc/"', 'Sub-folder "postproc/"')); self.subfolder.setChecked(self._b("pp_subfolder", True))
        opt.addWidget(self.subfolder)
        self.show_src = QCheckBox(tr("Voir source", "Show source")); self.show_src.stateChanged.connect(lambda *_: self._render())
        opt.addWidget(self.show_src); opt.addStretch(1)
        self._bo = bo  # keep ref to enable/disable
        self.src_edit.textChanged.connect(self._sync_output)
        self._on_same_folder(self.same_folder.isChecked())

        # ── split: modules (left) / preview (right) ──
        split = QSplitter(Qt.Horizontal); root.addWidget(split, 1)
        ctl_host = QWidget(); ctl_host.setStyleSheet("QGroupBox{background:transparent;}")
        self._ctl = QVBoxLayout(ctl_host); self._ctl.setContentsMargins(2, 2, 4, 2); self._ctl.setSpacing(6)
        # top toggle: collapse / expand every module at once (label follows the state)
        _hdr = QHBoxLayout(); _hdr.addStretch(1)
        self.btn_toggle_all = QPushButton(); self.btn_toggle_all.setFixedSize(160, 24)
        self.btn_toggle_all.setStyleSheet('QPushButton{background:#3a3f4b; color:#fff; border:none;'
                                          ' border-radius:5px; padding:2px 12px; font-weight:600; text-align:center;}'
                                          'QPushButton:hover{background:#454b59;}')
        self.btn_toggle_all.clicked.connect(self._toggle_all)
        _hdr.addWidget(self.btn_toggle_all); _hdr.addSpacing(8)
        self._ctl.addLayout(_hdr)
        _Module._settings_ref = self.settings  # modules read this to persist collapse state
        self._build_tempfix()
        self._build_undistort()
        self._build_antialias()
        self._build_deband()
        self._build_linedark()
        self._build_linethin()
        self._build_color()
        self._build_resize()
        self._build_sharpen()
        self._build_edgeclean()
        self._build_output()
        self._ctl.addStretch(1)
        self._update_toggle_all_label()
        scroll = QScrollArea(); scroll.setWidget(ctl_host); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400); scroll.setMaximumWidth(480); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # reserve scrollbar space so headers never hide under it
        split.addWidget(scroll)

        rpanel = QWidget(); rv = QVBoxLayout(rpanel); rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(6)
        self._preview = _PreviewView()
        rv.addWidget(self._preview, 1)
        rr = QHBoxLayout()
        self.btn_run = QPushButton(tr("▶ Lancer Post Processing", "▶ Run Post Processing"))
        self.btn_run.setStyleSheet(
            'QPushButton{background:#2ecc71; color:#fff; border:none; border-radius:6px; padding:8px;'
            ' font-weight:700; font-family:"Segoe UI Emoji"; font-size:13px;}'
            'QPushButton:hover{background:#3ee07f;}'
            'QPushButton:pressed{background:#22a35a; padding-top:10px;}'
            'QPushButton:disabled{background:#2f4a3a; color:#8aa;}')
        self.btn_run.clicked.connect(self._run); rr.addWidget(self.btn_run, 1)
        self.btn_stop = QPushButton("⏹ Stop"); self.btn_stop.setEnabled(False); self.btn_stop.setFixedWidth(100)
        self.btn_stop.setStyleSheet(
            'QPushButton{background:#e74c3c; color:#fff; border:none; border-radius:6px; padding:8px;'
            ' font-weight:700; font-family:"Segoe UI Emoji"; font-size:13px;}'
            'QPushButton:hover{background:#ef6657;}'
            'QPushButton:pressed{background:#b83a2e; padding-top:10px;}'
            'QPushButton:disabled{background:#4a3a3a; color:#a88;}')
        self.btn_stop.clicked.connect(self._request_stop); rr.addWidget(self.btn_stop)
        rv.addLayout(rr)
        self.prog = QProgressBar(); self.prog.setRange(0, 100); rv.addWidget(self.prog)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(90); self.log.setFont(QFont("Consolas", 9))
        rv.addWidget(self.log)
        split.addWidget(rpanel); split.setStretchFactor(1, 1)

        self._restore_state()
        self._loading = False

    # ── persistence : remember every module parameter across restarts / tab changes ──
    def _persistables(self):
        """(settings_key, widget) for all module controls. Built lazily once all exist."""
        reg = []
        for name in ("tempfix_en", "undistort_en", "aa_en", "deband_en", "linedark_en", "linethin_en",
                     "color_en", "resize_en", "sharp_en", "edge_en", "output_en",
                     "lt_luma", "edge_hot", "edge_repair", "px_aspect",
                     "aa_method", "aa_device", "resize_mode", "resize_algo", "sharp_mode", "out_fmt", "out_depth",
                     "px_w", "px_h",
                     "s_aa_str", "s_aa_pass", "s_dy", "s_dcb", "s_dcr", "s_dgy", "s_dgc", "s_drg",
                     "s_ldstr", "s_ldlo", "s_ldhi", "s_ltdepth", "s_ltblur", "s_ltthr",
                     "s_bri", "s_con", "s_sat", "s_gam", "s_lr", "s_lg", "s_lb",
                     "s_scale", "s_mult", "s_strength", "s_radius", "s_cas", "s_edge", "out_quality"):
            w = getattr(self, name, None)
            if w is not None:
                reg.append((f"pp_{name}", w))
        return reg

    def _save_state(self):
        if getattr(self, "_loading", False):
            return
        for key, w in self._persistables():
            try:
                if isinstance(w, QCheckBox):
                    self.settings.set(key, w.isChecked())
                elif isinstance(w, QComboBox):
                    self.settings.set(key, w.currentText())
                elif isinstance(w, QLineEdit):
                    self.settings.set(key, w.text())
                else:  # LabeledSlider
                    self.settings.set(key, str(w.value()))
            except Exception:
                pass

    def _restore_state(self):
        for key, w in self._persistables():
            v = self.settings.get(key, None)
            if v is None or v == "":
                continue
            try:
                if isinstance(w, QCheckBox):
                    w.setChecked(v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on"))
                elif isinstance(w, QComboBox):
                    if w.findText(str(v)) >= 0:
                        w.setCurrentText(str(v))
                elif isinstance(w, QLineEdit):
                    w.setText(str(v))
                else:
                    w.setValue(float(v))
            except Exception:
                pass
        # refresh mode-dependent visibility after restore
        try:
            self._on_resize_mode(self.resize_mode.currentIndex())
        except Exception:
            pass

    # ── helpers ──
    def _b(self, key, default):
        v = self.settings.get(key, default)
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")

    def _all_modules(self):
        out = []
        for i in range(self._ctl.count()):
            w = self._ctl.itemAt(i).widget()
            if isinstance(w, _Module):
                out.append(w)
        return out

    def _toggle_all(self):
        mods = self._all_modules()
        # if anything is expanded → collapse all; otherwise expand all
        collapse = any(not m.is_collapsed() for m in mods)
        for m in mods:
            m.set_collapsed(collapse)
        self._update_toggle_all_label()

    def _update_toggle_all_label(self):
        mods = self._all_modules()
        all_collapsed = bool(mods) and all(m.is_collapsed() for m in mods)
        # Small triangles ▾ / ▸ (U+25BE / U+25B8) render as monochrome text glyphs on
        # Windows — unlike ▼/▶ where ▶ gets an emoji (coloured) presentation.
        if all_collapsed:
            self.btn_toggle_all.setText("▾ " + tr("Développer tout", "Expand all"))
        else:
            self.btn_toggle_all.setText("▸ " + tr("Réduire tout", "Collapse all"))

    def _changed(self, *_):
        self._debounce.start()
        self._save_state()

    def _slider(self, parent_body, *args, **kw):
        s = LabeledSlider(*args, **kw); s.valueChangedF.connect(self._changed); parent_body.addWidget(s)
        return s

    # ── module builders (each returns nothing; stores widgets on self) ──
    def _build_tempfix(self):
        m = _Module(tr("Temporal Fix (batch vidéo)", "Temporal Fix (video batch)"), self._changed, self._reset_tf, gear_fn=self._open_tf, key="tempfix")
        self.tempfix_en = m.enable
        lb = QLabel(tr("⚠ S'applique uniquement en batch (séquence d'images).\nRéglages via ⚙.",
                       "⚠ Applies only in batch (image sequence).\nSettings via ⚙."))
        lb.setStyleSheet("color:#888;"); m.bv.addWidget(lb)
        self._ctl.addWidget(m); self.m_tempfix = m

    def _build_undistort(self):
        m = _Module(tr("Undistort (batch vidéo)", "Undistort (video batch)"), self._changed, self._reset_ud, gear_fn=self._open_ud, key="undistort")
        self.undistort_en = m.enable
        lb = QLabel(tr("⚠ S'applique uniquement en batch (séquence d'images).\nRéglages via ⚙.",
                       "⚠ Applies only in batch (image sequence).\nSettings via ⚙."))
        lb.setStyleSheet("color:#888;"); m.bv.addWidget(lb)
        self._ctl.addWidget(m); self.m_undistort = m

    _TF_DEFAULTS = {"strength": 0.5, "window": 7, "precision": "float32", "tempfix_mode": "classic"}
    _UD_DEFAULTS = {"undistort_strength": 0.35, "undistort_window": 5, "undistort_precision": "float32",
                    "undistort_mode": "classic"}
    # F29: the dialogs write ups_* keys (Quick Upscale); this page keeps its own pp_* keys
    _PP_KEYS = {"strength": "pp_tf_strength", "window": "pp_tf_window", "precision": "pp_tf_precision",
                "tempfix_mode": "pp_tf_mode", "undistort_strength": "pp_ud_strength",
                "undistort_window": "pp_ud_window", "undistort_precision": "pp_ud_precision",
                "undistort_mode": "pp_ud_mode"}

    def _save_tf(self):
        for k, sk in self._PP_KEYS.items():
            if k in self._tf:
                self.settings.set(sk, str(self._tf[k]))

    def _reset_tf(self):   # F28
        self._tf.update(self._TF_DEFAULTS); self._save_tf()

    def _reset_ud(self):   # F28
        self._tf.update(self._UD_DEFAULTS); self._save_tf()

    def _open_tf(self):
        from qt_ui.tabs.tools_pages.quick_upscale import TempFixDialog
        snap = {k: self.settings.get(k) for k in ("ups_tempfix_strength", "ups_tempfix_window",
                                                   "ups_tempfix_precision", "ups_tempfix_mode")}
        if TempFixDialog(self.settings, self._tf, self).exec():
            self._save_tf()
        for k, v in snap.items():   # the Quick Upscale settings stay untouched
            if v is not None:
                self.settings.set(k, v)

    def _open_ud(self):
        from qt_ui.tabs.tools_pages.quick_upscale import UndistortDialog
        snap = {k: self.settings.get(k) for k in ("ups_undistort_strength", "ups_undistort_window",
                                                   "ups_undistort_precision", "ups_undistort_mode")}
        if UndistortDialog(self.settings, self._tf, self).exec():
            self._save_tf()
        for k, v in snap.items():
            if v is not None:
                self.settings.set(k, v)

    def _build_deband(self):
        m = _Module(tr("Deband / Antibanding", "Deband / Antibanding"), self._changed, lambda: self._reset([
            (self.s_dy, 12), (self.s_dcb, 10), (self.s_dcr, 10), (self.s_dgy, 0), (self.s_dgc, 0), (self.s_drg, 15)]), key="deband")
        self.deband_en = m.enable
        self.s_dy = self._slider(m.bv, tr("Force luma (y)", "Luma strength (y)"), 0, 64, 12, 1, "{:.0f}",
                                 hint=tr('Intensité du débanding sur la luminance. ↑ = bandes mieux lissées mais risque de perte de détail · ↓ = effet subtil.', 'Luma debanding strength. ↑ = smoother bands but may lose detail · ↓ = subtle.'))
        self.s_dcb = self._slider(m.bv, "Force Cb", 0, 64, 10, 1, "{:.0f}",
                                  hint=tr('Débanding du canal chroma bleu (Cb).', 'Blue-chroma (Cb) debanding strength.'))
        self.s_dcr = self._slider(m.bv, "Force Cr", 0, 64, 10, 1, "{:.0f}",
                                  hint=tr('Débanding du canal chroma rouge (Cr).', 'Red-chroma (Cr) debanding strength.'))
        self.s_dgy = self._slider(m.bv, tr("Grain luma", "Luma grain"), 0, 64, 0, 1, "{:.0f}",
                                  hint=tr('Grain de dither anti-bandes sur la luma. ↑ = masque mieux les bandes, ajoute du bruit.', 'Anti-band dither grain on luma. ↑ = hides bands better, adds noise.'))
        self.s_dgc = self._slider(m.bv, tr("Grain chroma", "Chroma Grain"), 0, 64, 0, 1, "{:.0f}",
                                  hint=tr('Grain de dither sur la chroma.', 'Dither grain on chroma.'))
        self.s_drg = self._slider(m.bv, tr("Rayon", "Radius"), 3, 30, 15, 1, "{:.0f}",
                                  hint=tr("Distance d'échantillonnage. ↑ = corrige de larges dégradés (halos possibles) · ↓ = local, micro-carrés.", 'Sampling distance. ↑ = fixes wide gradients (possible halos) · ↓ = local.'))
        self._ctl.addWidget(m)

    def _build_linedark(self):
        m = _Module(tr("Renforcement lignes", "Line darkening"), self._changed, lambda: self._reset([
            (self.s_ldstr, 0.2), (self.s_ldlo, 4), (self.s_ldhi, 28)]), key="linedark")
        self.linedark_en = m.enable
        self.s_ldstr = self._slider(m.bv, tr("Force", "Strength"), 0.0, 1.0, 0.2, 0.01,
                                    hint=tr('Assombrit les lignes/contours (look plus « dessiné »). ↑ = lignes plus marquées · ↓ = subtil.', "Darkens lines/edges (more 'drawn' look). ↑ = bolder lines · ↓ = subtle."))
        self.s_ldlo = self._slider(m.bv, tr("Seuil bas", "Low threshold"), 0, 255, 4, 1, "{:.0f}",
                                   hint=tr('Seuil de détection des contours faibles. ↓ = capte + de lignes fines.', 'Low edge-detection threshold. ↓ = catches more thin lines.'))
        self.s_ldhi = self._slider(m.bv, tr("Seuil haut", "High threshold"), 0, 255, 28, 1, "{:.0f}",
                                   hint=tr('Seuil des contours forts. ↑ = ne renforce que les lignes franches.', 'High edge threshold. ↑ = only reinforces strong lines.'))
        self._ctl.addWidget(m)

    def _build_linethin(self):
        m = _Module(tr("Amincissement lignes (aWarpSharp2)", "Line thinning (aWarpSharp2)"), self._changed, lambda: self._reset([
            (self.s_ltdepth, 6.0), (self.s_ltblur, 1), (self.s_ltthr, 0.05), (self.lt_luma, True)]), key="linethin")
        self.linethin_en = m.enable
        self.s_ltdepth = self._slider(m.bv, tr("Profondeur warp", "Warp depth"), 0.0, 20.0, 6.0, 0.1,
                                      hint=tr('Intensité du warp vers les contours (amincit + affûte les lignes). ↑ = lignes plus fines mais halos si trop fort · ↓ = doux.', 'Warp strength toward edges (thins + sharpens lines). ↑ = thinner lines but halos if too high · ↓ = gentle.'))
        self.s_ltblur = self._slider(m.bv, tr("Flou pré-warp", "Pre-warp blur"), 0, 5, 1, 1, "{:.0f}",
                                     hint=tr('Lissage du champ de warp. ↑ = warp + doux/stable · ↓ = plus agressif.', 'Warp-field blur. ↑ = smoother/stable warp · ↓ = more aggressive.'))
        self.s_ltthr = self._slider(m.bv, tr("Seuil edge", "Edge threshold"), 0.01, 0.5, 0.05, 0.01,
                                    hint=tr('Sensibilité de détection des bords. ↓ = capte + de bords faibles.', 'Edge sensitivity. ↓ = catches more weak edges.'))
        self.lt_luma = QCheckBox(tr("Luma uniquement (recommandé)", "Luma only (recommended)")); self.lt_luma.setChecked(True)
        self.lt_luma.stateChanged.connect(self._changed); m.bv.addWidget(self.lt_luma)
        self._ctl.addWidget(m)

    def _build_color(self):
        m = _Module(tr("Correction couleur", "Color correction"), self._changed, lambda: self._reset([
            (self.s_bri, 1.0), (self.s_con, 1.0), (self.s_sat, 1.0), (self.s_gam, 1.0),
            (self.s_lr, 1.0), (self.s_lg, 1.0), (self.s_lb, 1.0)]), key="color")
        self.color_en = m.enable
        self.s_bri = self._slider(m.bv, tr("Luminosité", "Brightness"), 0.3, 3.0, 1.0, 0.01,
                                  hint=tr('1.0 = neutre. ↑ = image plus claire · ↓ = plus sombre.', '1.0 = neutral. ↑ = brighter · ↓ = darker.'))
        self.s_con = self._slider(m.bv, tr("Contraste", "Contrast"), 0.3, 3.0, 1.0, 0.01,
                                  hint=tr('1.0 = neutre. ↑ = noirs/blancs plus marqués · ↓ = image plate.', '1.0 = neutral. ↑ = punchier · ↓ = flatter.'))
        self.s_sat = self._slider(m.bv, "Saturation", 0.0, 2.0, 1.0, 0.01,
                                  hint=tr('0 = niveaux de gris · 1 = neutre · 2 = couleurs saturées.', '0 = grayscale · 1 = neutral · 2 = vivid.'))
        self.s_gam = self._slider(m.bv, "Gamma", 0.2, 4.0, 1.0, 0.01,
                                  hint=tr('Courbe de tons. ↑ = éclaircit les tons moyens · ↓ = assombrit.', 'Tone curve. ↑ = lifts midtones · ↓ = darkens.'))
        self.s_lr = self._slider(m.bv, tr("Rouge", "Red"), 0.5, 1.5, 1.0, 0.01, hint=tr('Gain du canal Rouge (balance des blancs).', 'Red channel gain (white balance).'))
        self.s_lg = self._slider(m.bv, tr("Vert", "Green"), 0.5, 1.5, 1.0, 0.01, hint=tr('Gain du canal Vert.', 'Green channel gain.'))
        self.s_lb = self._slider(m.bv, tr("Bleu", "Blue"), 0.5, 1.5, 1.0, 0.01, hint=tr('Gain du canal Bleu.', 'Blue channel gain.'))
        self._ctl.addWidget(m)

    def _build_resize(self):
        m = _Module(tr("Redimensionnement", "Resize"), self._changed, self._reset_resize, key="resize")
        self.resize_en = m.enable
        r0 = QHBoxLayout(); r0.addWidget(QLabel(tr("Mode :", "Mode:")))
        self.resize_mode = QComboBox(); self.resize_mode.addItems([tr("Pourcentage", "Percentage"), tr("Multiplicateur", "Multiplier"), tr("Résolution cible", "Target resolution")])
        self.resize_mode.setFixedWidth(_CW); self.resize_mode.currentIndexChanged.connect(self._on_resize_mode)
        r0.addWidget(self.resize_mode); r0.addStretch(1); m.bv.addLayout(r0)
        r1 = QHBoxLayout(); r1.addWidget(QLabel(tr("Algo :", "Algo:")))
        self.resize_algo = QComboBox()
        self.resize_algo.addItems(["LANCZOS", "lanczos4", "spline16", "spline36", "spline64",
                                   "mitchell", "catmull", "bspline", "BICUBIC", "BILINEAR", "NEAREST"])
        self.resize_algo.setFixedWidth(_CW); self.resize_algo.currentIndexChanged.connect(self._changed)
        r1.addWidget(self.resize_algo); r1.addStretch(1); m.bv.addLayout(r1)
        # mode-specific controls
        self.s_scale = self._slider(m.bv, tr("Échelle (%)", "Scale (%)"), 10, 400, 100, 1, "{:.0f}")
        self.s_mult = self._slider(m.bv, "Multiplicateur (×)", 0.1, 4.0, 1.0, 0.05)
        self._px_row = QWidget(); pxl = QHBoxLayout(self._px_row); pxl.setContentsMargins(0, 0, 0, 0)
        pxl.addWidget(QLabel(tr("L :", "W:")))
        self.px_w = QLineEdit("1920"); self.px_w.setFixedWidth(60); self.px_w.textChanged.connect(self._changed); pxl.addWidget(self.px_w)
        pxl.addWidget(QLabel(tr("H :", "H:")))
        self.px_h = QLineEdit("1080"); self.px_h.setFixedWidth(60); self.px_h.textChanged.connect(self._changed); pxl.addWidget(self.px_h)
        self.px_aspect = QCheckBox(tr("Garder ratio", "Keep aspect")); self.px_aspect.setChecked(True); self.px_aspect.stateChanged.connect(self._changed)
        pxl.addWidget(self.px_aspect); pxl.addStretch(1)
        m.bv.addWidget(self._px_row)
        self._ctl.addWidget(m)
        self._on_resize_mode(0)

    def _on_resize_mode(self, idx):
        self.s_scale.setVisible(idx == 0)
        self.s_mult.setVisible(idx == 1)
        self._px_row.setVisible(idx == 2)
        self._changed()

    def _reset_resize(self):
        self.resize_mode.setCurrentIndex(0); self.resize_algo.setCurrentIndex(0)
        self.s_scale.setValue(100); self.s_mult.setValue(1.0)
        self.px_w.setText("1920"); self.px_h.setText("1080"); self.px_aspect.setChecked(True)

    def _build_sharpen(self):
        m = _Module(tr("Netteté (Sharpen)", "Sharpen"), self._changed, lambda: (self.sharp_mode.setCurrentIndex(0), self._reset([
            (self.s_strength, 1.0), (self.s_radius, 1.5), (self.s_cas, 0.5)])), key="sharpen")
        self.sharp_en = m.enable
        row = QHBoxLayout(); row.addWidget(QLabel(tr("Mode :", "Mode:")))
        self.sharp_mode = QComboBox(); self.sharp_mode.addItems(["UnsharpMask", "CAS"]); self.sharp_mode.setFixedWidth(_CW)
        self.sharp_mode.currentIndexChanged.connect(self._changed); row.addWidget(self.sharp_mode); row.addStretch(1)
        m.bv.addLayout(row)
        self.s_strength = self._slider(m.bv, tr("Force", "Strength"), 0.0, 5.0, 1.0, 0.05,
                                       hint=tr('Intensité de la netteté (UnsharpMask). ↑ = + net mais halos/bruit · ↓ = doux.', 'Sharpening strength (UnsharpMask). ↑ = sharper but halos/noise · ↓ = soft.'))
        self.s_radius = self._slider(m.bv, tr("Rayon (Unsharp)", "Radius (Unsharp)"), 0.3, 6.0, 1.5, 0.1,
                                     hint=tr('Rayon du masque flou. ↑ = halos plus larges · ↓ = netteté fine.', 'Unsharp mask radius. ↑ = wider halos · ↓ = fine detail.'))
        self.s_cas = self._slider(m.bv, tr("Netteté CAS", "CAS Sharpness"), 0.0, 1.0, 0.5, 0.01,
                                  hint=tr('Netteté adaptative au contraste (CAS, moins de halos). ↑ = + net.', 'Contrast-adaptive sharpening (CAS, fewer halos). ↑ = sharper.'))
        self._ctl.addWidget(m)

    def _build_edgeclean(self):
        m = _Module(tr("Edge Cleanup", "Edge Cleanup"), self._changed, lambda: self._reset([(self.s_edge, 8), (self.edge_hot, True), (self.edge_repair, False)]), key="edgeclean")
        self.edge_en = m.enable
        self.s_edge = self._slider(m.bv, tr("Force", "Strength"), 0, 20, 8, 1, "{:.0f}",
                                   hint=tr('Répare le ringing/aliasing près des bords (fin de chaîne). ↑ = nettoyage + fort (peut adoucir) · ↓ = léger.', 'Repairs ringing/aliasing near edges (end of chain). ↑ = stronger cleanup (may soften) · ↓ = light.'))
        self.edge_hot = QCheckBox(tr("Supprimer pixels chauds", "Remove hot pixels")); self.edge_hot.setChecked(True)
        self.edge_hot.setToolTip(tr('Remplace les pixels isolés trop clairs ou trop sombres près des bords.', 'Replaces isolated too-bright or too-dark pixels near edges.'))
        self.edge_hot.stateChanged.connect(self._changed); m.bv.addWidget(self.edge_hot)
        self.edge_repair = QCheckBox(tr("Mode Repair (clamp min/max — EdgeCleaner)", "Repair mode (clamp min/max — EdgeCleaner)"))
        self.edge_repair.stateChanged.connect(self._changed); m.bv.addWidget(self.edge_repair)
        self._ctl.addWidget(m)

    def _build_antialias(self):
        m = _Module(tr("Antialiasing (edge-directed)", "Antialiasing (edge-directed)"), self._changed, lambda: (self.aa_method.setCurrentIndex(0), self.aa_device.setCurrentIndex(0), self._reset([(self.s_aa_str, 0.5), (self.s_aa_pass, 1)])), key="antialias")
        self.aa_en = m.enable
        r = QHBoxLayout(); r.addWidget(QLabel(tr("Méthode :", "Method:")))
        self.aa_method = QComboBox(); self.aa_method.addItems([tr("EEDIaa (rapide)", "EEDIaa (fast)"), "NNEDI3aa", "EEDI3"])
        self.aa_method.setFixedWidth(_CW); self.aa_method.currentIndexChanged.connect(self._changed)
        r.addWidget(self.aa_method); r.addStretch(1); m.bv.addLayout(r)
        r2 = QHBoxLayout(); r2.addWidget(QLabel(tr("Device :", "Device:")))
        self.aa_device = QComboBox(); self.aa_device.addItems(["CPU", "GPU"]); self.aa_device.setFixedWidth(_CW)
        self.aa_device.currentIndexChanged.connect(self._changed)
        r2.addWidget(self.aa_device); r2.addStretch(1); m.bv.addLayout(r2)
        self.s_aa_str = self._slider(m.bv, tr("Force", "Strength"), 0.0, 1.0, 0.5, 0.01,
                                     hint=tr("Intensité de l'anti-aliasing. ↑ = escaliers mieux lissés mais image + douce · ↓ = subtil.", 'Anti-aliasing strength. ↑ = smoother jaggies but softer image · ↓ = subtle.'))
        self.s_aa_pass = self._slider(m.bv, "Passes", 1, 3, 1, 1, "{:.0f}",
                                      hint=tr('Nombre de passes. ↑ = lissage + fort (plus lent).', 'Number of passes. ↑ = stronger smoothing (slower).'))
        self.aa_note = QLabel(""); self.aa_note.setStyleSheet("color:#888;"); self.aa_note.setWordWrap(True); m.bv.addWidget(self.aa_note)
        self.aa_method.currentIndexChanged.connect(self._update_aa_note); self.aa_device.currentIndexChanged.connect(self._update_aa_note)
        self._update_aa_note()
        self._ctl.addWidget(m)

    def _update_aa_note(self, *_):
        method = self.aa_method.currentText()
        if method.startswith("EEDIaa"):
            self.aa_note.setText(tr("Approximation directionnelle (CPU, temps réel).", "Directional approximation (CPU, realtime)."))
            return
        try:
            from src.core import vs_aa
            ok = vs_aa.available()
        except Exception:
            ok = False
        if ok:
            self.aa_note.setText(tr("nnedi3/eedi3 RÉEL via VapourSynth (StaxRip) — appliqué au lancement (batch). "
                                 "Aperçu = approximation rapide.\nReal nnedi3/eedi3 via VapourSynth — applied on Run (batch). "
                                 "Preview = fast approximation.", "Real nnedi3/eedi3 via VapourSynth (StaxRip) — applied on Run (batch). Preview = fast approximation."))
        else:
            self.aa_note.setText(tr("⚠ VapourSynth (StaxRip) introuvable → repli sur l'approximation directionnelle.",
                                    "⚠ VapourSynth (StaxRip) not found → falls back to the directional approximation."))

    def _aa_config(self):
        if not self.aa_en.isChecked():
            return {"use_vs": False}
        method = self.aa_method.currentText()
        if method.startswith("EEDIaa"):
            return {"use_vs": False}
        try:
            from src.core import vs_aa
            if not vs_aa.available():
                return {"use_vs": False}
        except Exception:
            return {"use_vs": False}
        return {"use_vs": True, "method": "eedi3" if "EEDI3" in method else "nnedi3",
                "device": self.aa_device.currentText(), "strength": self.s_aa_str.value(),
                "passes": int(self.s_aa_pass.value())}

    def _build_output(self):
        m = _Module(tr("Sortie (profondeur / format)", "Output (depth / format)"), self._changed, lambda: (self.out_fmt.setCurrentIndex(0), self.out_depth.setCurrentIndex(0), self.out_quality.setValue(95)), key="output")
        self.output_en = m.enable
        r = QHBoxLayout(); r.addWidget(QLabel(tr("Format :", "Format:")))
        self.out_fmt = QComboBox(); self.out_fmt.addItems(["PNG", "JPEG", "WEBP", "TIFF", "BMP"]); self.out_fmt.setFixedWidth(_CW)
        self.out_fmt.currentIndexChanged.connect(self._changed)
        r.addWidget(self.out_fmt); r.addStretch(1); m.bv.addLayout(r)
        r2 = QHBoxLayout(); r2.addWidget(QLabel(tr("Profondeur :", "Depth:")))
        self.out_depth = QComboBox(); self.out_depth.addItems(["8 bits", "16 bits (PNG/TIFF)"]); self.out_depth.setFixedWidth(_CW)
        self.out_depth.setToolTip(tr("16 bits : fichier PNG/TIFF 16 bits réel ; la chaîne calcule en 8 bits, la précision reste celle du 8 bits.",
                                     "16 bits: a real 16-bit PNG/TIFF file; the chain computes in 8 bits, so the precision stays 8-bit."))
        self.out_depth.currentIndexChanged.connect(self._changed)
        r2.addWidget(self.out_depth); r2.addStretch(1); m.bv.addLayout(r2)
        self.out_quality = self._slider(m.bv, tr("Qualité (JPEG/WEBP)", "Quality (JPEG/WEBP)"), 60, 100, 95, 1, "{:.0f}")
        self._ctl.addWidget(m)

    @staticmethod
    def _reset(pairs):
        for sld, val in pairs:
            if isinstance(val, bool):      # checkboxes (F38)
                sld.setChecked(val)
            else:
                sld.setValue(val)

    # ── pickers ──
    def _pick_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Image", self.src_edit.text() or "",
                                           "Images (*.png *.jpg *.jpeg *.webp *.tiff *.tif *.bmp);;Tous (*.*)")
        if p:
            self.src_edit.setText(p); self.settings.set("pp_src", p); self._load_preview(p)

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, tr("Dossier source", "Source folder"), self.src_edit.text() or "")
        if d:
            self.src_edit.setText(d); self.settings.set("pp_src", d); self._preview_from_source()

    def _preview_from_source(self, *_):
        """F39: the preview follows the batch source (a file, or the first image of a
        folder) instead of keeping an older image on screen."""
        src = self.src_edit.text().strip().strip('"')
        if os.path.isdir(src):
            imgs = sorted(f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in _IMG_EXT)
            src = os.path.join(src, imgs[0]) if imgs else ""
        if src and os.path.isfile(src):
            self._load_preview(src)

    def _pick_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, tr("Dossier", "Folder"), edit.text() or "")
        if d:
            edit.setText(d); self.settings.set("pp_out", d)

    def _on_same_folder(self, on):
        self.settings.set("pp_same", bool(on))
        self.out_edit.setEnabled(not on)
        self._bo.setEnabled(not on)
        if on:
            self._sync_output()

    def _sync_output(self, *_):
        if not self.same_folder.isChecked():
            return
        src = self.src_edit.text().strip()
        if src:
            self.out_edit.setText(src if os.path.isdir(src) else os.path.dirname(src))

    def _load_preview(self, path):
        try:
            from PIL import Image
            self._src_img = Image.open(path).convert("RGB"); self._render()
        except Exception as e:
            self._preview.setText(tr(f"Erreur ouverture : {e}", f"Open error: {e}"))

    # ── settings dict for apply_chain ──
    def _output_opts(self) -> dict:
        """How the batch worker writes each frame (not part of the spatial chain)."""
        if not self.output_en.isChecked():
            return {"fmt": "PNG", "ext": ".png", "bit16": False, "quality": 95}
        fmt = self.out_fmt.currentText()
        _ext = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "TIFF": ".tif", "BMP": ".bmp"}.get(fmt, ".png")
        return {"fmt": fmt, "ext": _ext, "bit16": self.out_depth.currentIndex() == 1 and fmt in ("PNG", "TIFF"),
                "quality": int(self.out_quality.value())}

    def _settings(self) -> dict:
        return {
            "aa_enabled": self.aa_en.isChecked(),
            "aa_strength": self.s_aa_str.value(), "aa_passes": int(self.s_aa_pass.value()),
            "deband_enabled": self.deband_en.isChecked(),
            "deband_y": int(self.s_dy.value()), "deband_cb": int(self.s_dcb.value()),
            "deband_cr": int(self.s_dcr.value()), "deband_grainy": int(self.s_dgy.value()),
            "deband_grainc": int(self.s_dgc.value()), "deband_range": int(self.s_drg.value()),
            "linedark_enabled": self.linedark_en.isChecked(),
            "linedark_strength": self.s_ldstr.value(), "linedark_lo": int(self.s_ldlo.value()),
            "linedark_hi": int(self.s_ldhi.value()),
            "linethin_enabled": self.linethin_en.isChecked(),
            "linethin_depth": self.s_ltdepth.value(), "linethin_blur": int(self.s_ltblur.value()),
            "linethin_thresh": self.s_ltthr.value(), "linethin_luma": self.lt_luma.isChecked(),
            "edgeclean_enabled": self.edge_en.isChecked(),
            "edgeclean_strength": int(self.s_edge.value()), "edgeclean_hot": self.edge_hot.isChecked(),
            "edgeclean_repair": self.edge_repair.isChecked(),
            "color_enabled": self.color_en.isChecked(),
            "brightness": self.s_bri.value(), "contrast": self.s_con.value(),
            "saturation": self.s_sat.value(), "gamma": self.s_gam.value(),
            "level_r": self.s_lr.value(), "level_g": self.s_lg.value(), "level_b": self.s_lb.value(),
            "resize_enabled": self.resize_en.isChecked(),
            "resize_mode": {0: "percent", 1: "multiplier", 2: "pixels"}.get(self.resize_mode.currentIndex(), "percent"),
            "resize_scale": int(self.s_scale.value()), "resize_multiplier": self.s_mult.value(),
            "resize_width": int(self.px_w.text() or 0) if self.px_w.text().strip().isdigit() else 0,
            "resize_height": int(self.px_h.text() or 0) if self.px_h.text().strip().isdigit() else 0,
            "resize_aspect": self.px_aspect.isChecked(),
            "resize_method": self.resize_algo.currentText(),
            "sharpen_enabled": self.sharp_en.isChecked(),
            "sharpen_mode": "cas" if self.sharp_mode.currentText() == "CAS" else "unsharp",
            "sharpen_strength": self.s_strength.value(), "sharpen_radius": self.s_radius.value(),
            "sharpen_threshold": 3, "cas_sharpness": self.s_cas.value(),
        }

    # ── live preview ──
    def _render(self):
        if self._src_img is None:
            return
        try:
            if self.show_src.isChecked():
                img = self._src_img
            else:
                from src.core.post_process_chain import apply_chain
                img = apply_chain(self._src_img.copy(), self._settings())
            self._result = img
            self._preview.set_images(pil_to_pixmap(img), pil_to_pixmap(self._src_img))
        except Exception as e:
            self._preview.setText(tr(f"Erreur traitement : {e}", f"Processing error: {e}"))

    # ── batch ──
    def _request_stop(self):
        if self._stop:
            self._stop.set()
        self.btn_stop.setEnabled(False); self.btn_stop.setText(tr("Arrêt…", "Stopping…"))

    def _run(self):
        if self._worker and self._worker.isRunning():
            return
        src = self.src_edit.text().strip()
        if not src or not os.path.exists(src):
            self.log.setPlainText(tr("❌ Source invalide.", "❌ Invalid source.")); return
        base = (src if os.path.isdir(src) else os.path.dirname(src)) if self.same_folder.isChecked() else self.out_edit.text().strip()
        if not base:
            self.log.setPlainText(tr("❌ Dossier de sortie requis.", "❌ Output folder required.")); return
        out = os.path.join(base, "postproc") if self.subfolder.isChecked() else base
        self.settings.set("pp_src", src); self.settings.set("pp_out", self.out_edit.text().strip())
        self.settings.set("pp_same", self.same_folder.isChecked()); self.settings.set("pp_subfolder", self.subfolder.isChecked())
        self.log.clear(); self.prog.setValue(0)
        self.btn_run.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_stop.setText("⏹ Stop")
        self.toolProgress.emit(0)
        self._stop = threading.Event()
        tf_cfg = {
            "tf": self.tempfix_en.isChecked(), "ud": self.undistort_en.isChecked(),
            "tf_mode": self._tf["tempfix_mode"], "tf_str": self._tf["strength"],
            "tf_win": self._tf["window"], "tf_prec": self._tf["precision"],
            "ud_mode": self._tf["undistort_mode"], "ud_str": self._tf["undistort_strength"],
            "ud_win": self._tf["undistort_window"],
            "ud_prec": self._tf.get("undistort_precision", "float32"),
        }
        st = self._settings()
        aa = self._aa_config()
        if aa.get("use_vs"):
            st["aa_enabled"] = False  # disable the directional AA; VapourSynth handles it
            st["_aa_vs"] = aa
        self._worker = _BatchWorker(src, out, st, tf_cfg, self._output_opts(), self._stop)
        self._worker.line.connect(self.log.appendPlainText)
        self._worker.progress.connect(self._on_progress)
        self._worker.preview.connect(self._batch_preview)
        self._worker.done.connect(self._batch_done)
        self._worker.start()

    def _on_progress(self, f):
        self.prog.setValue(int(f * 100))
        self.toolProgress.emit(int(f * 100))

    def _batch_preview(self, b_src, b_out):
        from PIL import Image
        try:
            self._result = Image.open(io.BytesIO(b_out)).convert("RGB")
            src = Image.open(io.BytesIO(b_src)).convert("RGB")
            self._preview.set_images(pil_to_pixmap(self._result), pil_to_pixmap(src))
        except Exception:
            pass

    def _batch_done(self, n, total):
        self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_stop.setText("⏹ Stop")
        notes = list(getattr(self._worker, "notes", []) or [])
        ok = bool(total) and n == total and not notes
        if total:
            self.log.appendPlainText((tr("✅ Terminé", "✅ Finished") if ok else tr("⚠ Terminé partiellement", "⚠ Partially finished"))
                                     + f" — {n}/{total} images.")
        else:
            self.log.appendPlainText(tr("Aucune image.", "No image."))
        for x in notes:
            self.log.appendPlainText(tr(f"  ⚠ non appliqué : {x}", f"  ⚠ not applied: {x}"))
        self._worker = None; self._stop = None
        self.toolFinished.emit(ok)
        try:
            from qt_ui.notify import notify
            notify(self.settings, tr("Post Processing terminé", "Post Processing finished"), f"{n}/{total} images", "notif_win11_batch")
        except Exception:
            pass
