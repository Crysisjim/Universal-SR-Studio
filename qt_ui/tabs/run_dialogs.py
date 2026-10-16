"""
run_dialogs.py — popup windows for the Entraînement tab: Serveurs (TensorBoard),
LR schedule, Métriques (Loss/PSNR curves), Validation (LQ vs SR), Analyse IA.

Charts are drawn with a small self-contained QPainter widget (no matplotlib dep).
"""
from __future__ import annotations

import os
import re
import math
import webbrowser

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QComboBox,
    QCheckBox, QPlainTextEdit, QWidget, QFileDialog, QRadioButton, QButtonGroup,
    QMessageBox, QScrollArea, QSlider, QGridLayout,
)
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QFont
from PySide6.QtCore import Qt, QThread, Signal, QTimer

from qt_ui.i18n import tr


def remember_geometry(dlg, settings, key):
    """Give a dialog minimize/maximize buttons and remember its size, position and
    maximized state in the settings (key), restored on the next opening. Saved when
    the dialog closes (Close button, Esc or the window's X), so a manual resize counts."""
    dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowMinMaxButtonsHint)
    if settings is None:
        return
    try:
        from PySide6.QtCore import QByteArray
        saved = settings.get(key, "") or ""
        if saved:
            dlg.restoreGeometry(QByteArray.fromBase64(saved.encode("ascii")))
    except Exception:
        pass

    def _save(*_):
        try:
            settings.set(key, bytes(dlg.saveGeometry().toBase64()).decode("ascii"))
        except Exception:
            pass
    dlg.finished.connect(_save)


# ── tiny line chart ───────────────────────────────────────────────────────────
class LineChart(QWidget):
    def __init__(self, series, title="", xlabel="", xticks=None, parent=None):
        super().__init__(parent)
        # series: (label, color, [(x,y)...]) or (label, color, points, dashed)
        self.series = series; self.title = title; self.xlabel = xlabel; self.xticks = xticks
        self.marker_x = None  # current-iteration marker (red dot + vertical line)
        self.setMinimumSize(620, 380)
        self.setStyleSheet("background:#15151f; border-radius:6px;")

    def set_series(self, series):
        self.series = series; self.update()

    def set_marker(self, x):
        self.marker_x = x; self.update()

    @staticmethod
    def _kfmt(x):
        return f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"

    def paintEvent(self, _):
        from PySide6.QtCore import Qt as _Qt
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h, m = self.width(), self.height(), 50
        top = 34 if self.title else m
        pts_all = [pt for s in self.series for pt in s[2]]
        if self.title:
            p.setPen(QColor("#3b8ed0")); p.setFont(QFont("Roboto", 12, QFont.Bold))
            p.drawText(0, 6, w, 24, _Qt.AlignCenter, self.title)
        if not pts_all:
            p.setPen(QColor("#888")); p.drawText(self.rect(), _Qt.AlignCenter, tr("Pas de données pour le moment", "No data yet")); return
        xs = [x for x, _ in pts_all]; ys = [y for _, y in pts_all]
        xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
        if xmax == xmin: xmax += 1
        if ymax == ymin: ymax += 1e-6

        def X(x): return m + (x - xmin) / (xmax - xmin) * (w - 2 * m)
        def Y(y): return h - m - (y - ymin) / (ymax - ymin) * (h - top - m)

        p.setPen(QPen(QColor("#2a2a3a"), 1))
        for i in range(5):
            yy = top + i * (h - top - m) / 4
            p.drawLine(m, yy, w - m, yy)
        # x grid ticks
        ticks = self.xticks or [xmin, (xmin + xmax) / 2, xmax]
        p.setFont(QFont("Consolas", 8))
        for tx in ticks:
            if xmin <= tx <= xmax:
                gx = X(tx)
                p.setPen(QPen(QColor("#2a2a3a"), 1, _Qt.DashLine)); p.drawLine(int(gx), top, int(gx), h - m)
                p.setPen(QColor("#888")); p.drawText(int(gx) - 16, h - m + 16, self._kfmt(tx))
        p.setPen(QPen(QColor("#555"), 1)); p.drawLine(m, h - m, w - m, h - m); p.drawLine(m, top, m, h - m)

        leg_y = top + 12
        for s in self.series:
            label, color, pts = s[0], s[1], s[2]; dashed = len(s) > 3 and s[3]
            if not pts:
                continue
            pen = QPen(QColor(color), 2)
            if dashed:
                pen.setStyle(_Qt.DotLine)
            p.setPen(pen)
            prev = None
            for x, y in pts:
                cx, cy = X(x), Y(y)
                if prev:
                    p.drawLine(int(prev[0]), int(prev[1]), int(cx), int(cy))
                prev = (cx, cy)
            # right-aligned on the plot edge (a fixed x of w-m-140 let long labels, e.g.
            # "HF sharpness  (ideal ~0.75-1.0)", run past the chart border)
            p.setPen(QColor(color))
            p.drawText(max(m + 4, w - m - p.fontMetrics().horizontalAdvance(label)), leg_y, label); leg_y += 16

        # current-iteration marker: vertical red line + dot + label.
        # Label used to sit at the very top (top+4), right on the same row as the
        # title/legend -> unreadable, blended into the clutter. Moved to just above
        # the x-axis instead (clear of title/legend AND of the axis tick labels
        # below it), with a small background box so it stays legible even when the
        # line crosses plotted curves.
        if self.marker_x is not None and xmin <= self.marker_x <= xmax:
            mxp = X(self.marker_x)
            p.setPen(QPen(QColor("#e74c3c"), 1, _Qt.DashLine))
            p.drawLine(int(mxp), top, int(mxp), h - m)
            p.setBrush(QColor("#e74c3c")); p.setPen(QColor("#e74c3c"))
            p.drawEllipse(int(mxp) - 4, top - 4, 8, 8)
            p.setFont(QFont("Consolas", 8))
            label = f"iter {self._kfmt(self.marker_x)}"
            fm = p.fontMetrics()
            lbl_w = fm.horizontalAdvance(label)
            lbl_x = int(mxp) + 6
            # Flip to the left of the line when it would overflow the right edge.
            if lbl_x + lbl_w + 4 > w - m:
                lbl_x = int(mxp) - lbl_w - 10
            lbl_y = h - m - 8
            p.setPen(_Qt.NoPen); p.setBrush(QColor(30, 30, 46, 200))
            p.drawRect(lbl_x - 3, lbl_y - fm.ascent() - 2, lbl_w + 6, fm.height() + 4)
            p.setPen(QColor("#e74c3c")); p.setBrush(_Qt.NoBrush)
            p.drawText(lbl_x, lbl_y, label)

        p.setPen(QColor("#888")); p.setFont(QFont("Consolas", 8))
        p.drawText(4, top + 6, f"{ymax:.2g}"); p.drawText(4, h - m, f"{ymin:.2g}")
        if self.xlabel:
            p.drawText(0, h - 16, w, 14, _Qt.AlignCenter, self.xlabel)


def _pixmap(im):
    im = im.convert("RGB")
    qi = QImage(im.tobytes("raw", "RGB"), im.width, im.height, im.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qi.copy())


# ── Serveurs (TensorBoard) ────────────────────────────────────────────────────
class ServersDialog(QDialog):
    def __init__(self, settings, port="6006", logdir=None, script_path="", config_path="", parent=None):
        super().__init__(parent)
        self.settings = settings; self.port = port; self.logdir = logdir
        self.script_path = script_path; self.config_path = config_path
        self.setWindowTitle(tr("Serveurs — Galerie & TensorBoard", "Servers — Gallery & TensorBoard")); self.resize(460, 600); self.setStyleSheet("QDialog{background:#1e1e2e;}")
        v = QVBoxLayout(self)
        # ── Galerie images ──
        gp = settings.get("gallery_port", "8765") or "8765"
        self.gurl = f"http://localhost:{gp}"
        gt = QLabel(tr("🖼 Galerie Images", "🖼 Image Gallery")); gt.setStyleSheet("color:#2ecc71; font-weight:700; font-size:14px;")
        gt.setAlignment(Qt.AlignCenter); v.addWidget(gt)
        # Prefer the public ngrok URL (shareable / scannable on a phone) if the gallery is up.
        try:
            from src.core.gallery_server import get_server
            _srv = get_server()
            if getattr(_srv, "ngrok_url", ""):
                self.gurl = _srv.ngrok_url
        except Exception:
            pass
        gl = QLabel(self.gurl); gl.setStyleSheet("color:#3b8ed0;"); gl.setAlignment(Qt.AlignCenter); v.addWidget(gl)
        _gqr = self._qr_label(self.gurl)
        if _gqr:
            v.addWidget(_gqr, 0, Qt.AlignCenter)
        gr = QHBoxLayout(); v.addLayout(gr)
        gb1 = QPushButton(tr("🌐 Ouvrir", "🌐 Open")); gb1.clicked.connect(lambda: webbrowser.open(self.gurl)); gr.addWidget(gb1)
        gb2 = QPushButton(tr("▶ Démarrer la galerie", "▶ Start gallery")); gb2.setObjectName("ghost"); gb2.clicked.connect(self._start_gallery); gr.addWidget(gb2)
        v.addSpacing(10)
        # ── TensorBoard ──
        self.url = f"http://localhost:{port}"
        t = QLabel("📊 TensorBoard Data"); t.setStyleSheet("color:#e67e22; font-weight:700; font-size:14px;")
        t.setAlignment(Qt.AlignCenter); v.addWidget(t)
        lk = QLabel(self.url); lk.setStyleSheet("color:#3b8ed0;"); lk.setAlignment(Qt.AlignCenter); v.addWidget(lk)
        _tqr = self._qr_label(self.url)
        if _tqr:
            v.addWidget(_tqr, 0, Qt.AlignCenter)
        tbr = QHBoxLayout(); v.addLayout(tbr)
        b1 = QPushButton(tr("🌐 Ouvrir", "🌐 Open")); b1.clicked.connect(lambda: webbrowser.open(self.url)); tbr.addWidget(b1)
        b2 = QPushButton(tr("▶ Lancer / Relancer TensorBoard", "▶ Start / Restart TensorBoard")); b2.setObjectName("warn"); b2.clicked.connect(self._restart); tbr.addWidget(b2)
        v.addStretch(1)
        c = QPushButton(tr("Fermer", "Close")); c.setObjectName("ghost"); c.clicked.connect(self.accept); v.addWidget(c)

    def _qr_label(self, url):
        """Return a QLabel holding a QR code image for `url`, or None if unavailable."""
        try:
            from src.core.qr_code import generate_qr_image, is_qrcode_available
            if not is_qrcode_available():
                return None
            import tempfile
            p = os.path.join(tempfile.gettempdir(), f"uss_qr_{abs(hash(url)) % 10**8}.png")
            if not generate_qr_image(url, p, box_size=6):
                return None
            pm = QPixmap(p)
            if pm.isNull():
                return None
            lbl = QLabel()
            lbl.setPixmap(pm.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            lbl.setAlignment(Qt.AlignCenter)
            # white QR on a blue frame (stays scannable — modules keep their white quiet zone)
            lbl.setStyleSheet("background:#2980b9; padding:10px; border-radius:8px;")
            return lbl
        except Exception:
            return None

    def _start_gallery(self):
        d = self.settings.get("gallery_dir", "") or ""
        try:
            from src.core.gallery_server import get_server
            gp = int(self.settings.get("gallery_port", "8765") or 8765)
            res = get_server().start(d, port=gp, with_ngrok=False) or {}
            if not res.get("ok"):  # the result was ignored and the browser opened anyway
                QMessageBox.warning(self, tr("Galerie", "Gallery"), str(res.get("error") or "?"))
                return
            url = res.get("local_url") or self.gurl
            self.gurl = url
            webbrowser.open(url)
        except Exception as e:
            QMessageBox.warning(self, tr("Galerie", "Gallery"), str(e))

    def _logdir_map(self):
        """Build a {label: path} map covering both engine layouts (Redux: {root}/tb_logger,
        NeoSR: {root}/experiments/{exp_name}/tb_logger), probing the active engine root
        AND the ~/USRS engine locations for a centralized multi-engine view — exact
        port of CTk's _open_tensorboard resolution (the Qt dialog previously only ever
        looked at cwd/tb_logger, which almost never matched the real training logdir)."""
        import yaml as _yaml

        def _probe(root, label, out):
            if not root or not os.path.isdir(root):
                return
            direct = os.path.join(root, "tb_logger")
            if os.path.isdir(direct):
                out[label] = direct; return
            exp_name = None
            if self.config_path and os.path.isfile(self.config_path):
                try:
                    if self.config_path.lower().endswith(".toml"):  # NeoSR configs are TOML
                        import tomllib as _tl
                        with open(self.config_path, "rb") as f:
                            exp_name = (_tl.load(f) or {}).get("name", "")
                    else:
                        with open(self.config_path, "r", encoding="utf-8") as f:
                            exp_name = (_yaml.safe_load(f) or {}).get("name", "")
                except Exception:
                    pass
            if exp_name:
                # NeoSR writes experiments/tb_logger/<name>; older layouts used
                # experiments/<name>/tb_logger
                for nested in (os.path.join(root, "experiments", "tb_logger", exp_name),
                               os.path.join(root, "experiments", exp_name, "tb_logger")):
                    if os.path.isdir(nested):
                        out[label] = nested; return
            exp_base = os.path.join(root, "experiments")
            if os.path.isdir(os.path.join(exp_base, "tb_logger")):
                out[label] = os.path.join(exp_base, "tb_logger"); return
            if os.path.isdir(exp_base):
                for ename in sorted(os.listdir(exp_base)):
                    cand = os.path.join(exp_base, ename, "tb_logger")
                    if os.path.isdir(cand):
                        out[label] = cand; return

        out = {}
        if self.script_path and os.path.isfile(self.script_path):
            _probe(os.path.dirname(self.script_path), "ActiveEngine", out)
        from src.core.engine_paths import base_engine_path as _ebp
        usrs = _ebp()
        _probe(os.path.join(usrs, "traiNNer-redux"), "traiNNer", out)
        _probe(os.path.join(usrs, "neosr"), "NeoSR", out)
        # dedupe identical paths under different labels
        seen, deduped = {}, {}
        for label, path in out.items():
            norm = os.path.normcase(os.path.normpath(path))
            if norm not in seen:
                seen[norm] = label; deduped[label] = path
        return deduped

    def _restart(self):
        import sys, subprocess
        py = self.settings.get("python_path", sys.executable) or sys.executable
        try:
            from src.core import engine_paths as _ep
            _venv_py = _ep.shared_venv_python()
            if _venv_py and os.path.isfile(_venv_py):
                py = _venv_py
        except Exception:
            pass
        logdir_map = self._logdir_map()
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/IM", "tensorboard.exe"], capture_output=True)
            if logdir_map:
                if len(logdir_map) == 1:
                    tb_args = ["--logdir", next(iter(logdir_map.values()))]
                else:
                    tb_args = ["--logdir_spec", ",".join(f"{lbl}:{p}" for lbl, p in logdir_map.items())]
            else:
                logdir = self.logdir or os.path.join(os.getcwd(), "tb_logger")
                os.makedirs(logdir, exist_ok=True)
                tb_args = ["--logdir", logdir]
            subprocess.Popen([py, "-m", "tensorboard.main", *tb_args, f"--port={self.port}", "--bind_all"],
                             creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            QMessageBox.information(self, "TensorBoard", tr(f"Relancé sur {self.url}", f"Restarted on {self.url}"))
        except Exception as e:
            QMessageBox.critical(self, tr("Erreur", "Error"), str(e))


# ── LR schedule ───────────────────────────────────────────────────────────────
class LRDialog(QDialog):
    """The learning rate the configured scheduler will actually give (type, milestones,
    gamma, T_max, eta_min, warmup read from the config). It used to draw the same two
    generic curves (MultiStep + Cosine) whatever the config said."""

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Scheduler LR — Visualisation", "LR Scheduler — Visualization")); self.resize(720, 480); self.setStyleSheet("QDialog{background:#1e1e2e;}")
        v = QVBoxLayout(self)
        p = self._params(cfg)
        pts = self._curve(p)
        _g = p.get("gan_restart", 0)
        ticks = sorted({0, p["total"], *[m for m in p["milestones"] if 0 < m < p["total"]],
                        *([_g] + [_g + m for m in p["milestones"] if _g + m < p["total"]] if _g else [])})
        self.chart = LineChart(
            [(p["label"], "#2ecc71", pts)],
            title=tr("Évolution du Learning Rate", "Learning Rate Evolution"), xlabel=tr("Itérations", "Iterations"), xticks=ticks)
        v.addWidget(self.chart, 1)
        if p["note"]:
            n = QLabel(p["note"]); n.setStyleSheet("color:#e67e22;"); n.setWordWrap(True); v.addWidget(n)
        c = QPushButton(tr("Fermer", "Close")); c.setObjectName("ghost"); c.clicked.connect(self.accept); v.addWidget(c)
        # live red marker tracking the current training iteration (reads parent run_tab)
        self._parent = parent
        self._sync_marker()
        self._mtimer = QTimer(self); self._mtimer.setInterval(1000)
        self._mtimer.timeout.connect(self._sync_marker); self._mtimer.start()

    def _sync_marker(self):
        it = getattr(self._parent, "_last_iter", 0) or 0
        if it > 0:
            self.chart.set_marker(it)

    @staticmethod
    def _params(cfg):
        """Scheduler of a traiNNer-redux (train.scheduler) or NeoSR (train.scheduler) config,
        or of the flat Config-tab dict."""
        tr_ = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
        sch = tr_.get("scheduler", {}) if isinstance(tr_.get("scheduler"), dict) else {}
        def num(*vals, cast=float, default=None):
            for v in vals:
                if v not in (None, ""):
                    try:
                        return cast(v)
                    except (TypeError, ValueError):
                        pass
            return default
        total = num(tr_.get("total_iter"), cfg.get("total_iter"), cast=lambda x: int(float(x)), default=150000)
        optg = tr_.get("optim_g", {}) if isinstance(tr_.get("optim_g"), dict) else {}
        base = num(optg.get("lr"), cfg.get("lr"), default=2e-4)
        kind = str(sch.get("type") or cfg.get("scheduler") or "MultiStepLR").lower()
        ms = sch.get("milestones", cfg.get("milestones")) or []
        if isinstance(ms, str):
            ms = [int(x) for x in re.findall(r"\d+", ms)]
        ms = [int(m) for m in ms]
        gamma = num(sch.get("gamma"), cfg.get("gamma"), default=0.5)
        t_max = num(sch.get("T_max"), sch.get("t_max"), cfg.get("t_max"), cast=lambda x: int(float(x)), default=total)
        eta_min = num(sch.get("eta_min"), cfg.get("eta_min"), default=0.0)
        periods = [int(x) for x in (sch.get("periods") or [])]
        step_size = num(sch.get("step_size"), cast=lambda x: int(float(x)), default=max(1, total // 10))
        warm = num(tr_.get("warmup_iter"), cfg.get("warmup_iter"), cast=lambda x: int(float(x)), default=-1)
        note = ""
        if "plateau" in kind:
            note = tr("ReduceLROnPlateau dépend de la validation : la baisse ne peut pas être prévue (LR de départ affiché).",
                      "ReduceLROnPlateau depends on validation: its drops cannot be predicted (starting LR shown).")
        if cfg.get("schedule_free") or optg.get("schedule_free"):
            note = tr("Schedule-Free : l'optimiseur règle son LR lui-même (pas de scheduler).",
                      "Schedule-Free: the optimizer sets its own LR (no scheduler).")
        # traiNNer-redux gan_restart_lr: at the GAN start every scheduler restarts from the
        # initial LR and its milestones count from there
        gan_start = num(tr_.get("gan_start_iter"), cfg.get("gan_start_iter"), cast=lambda x: int(float(x)), default=0)
        restart = bool(tr_.get("gan_restart_lr", str(cfg.get("gan_restart_lr", "")).lower() == "true"))
        has_gan = bool(cfg.get("network_d") or str(cfg.get("use_gan", "")).lower() == "true")
        gan_restart = gan_start if (restart and has_gan and gan_start and 0 < gan_start < total) else 0
        if gan_restart and (tr_.get("gan_auto_start") or str(cfg.get("gan_auto_start", "")).lower() == "true"):
            note = (note + "\n" if note else "") + tr(
                f"GAN auto : redémarrage du LR montré à l'itération {gan_start} (au plus tard) — il a lieu "
                "plus tôt si le PSNR plafonne avant.",
                f"Auto GAN: the LR restart is shown at iteration {gan_start} (the latest) — it happens "
                "earlier if the PSNR plateaus before.")
        return {"total": max(1, total), "base": base, "kind": kind, "milestones": ms, "gamma": gamma,
                "gan_restart": gan_restart,
                "t_max": max(1, t_max), "eta_min": eta_min, "periods": periods, "step_size": max(1, step_size),
                "warmup": warm, "note": note, "label": sch.get("type") or cfg.get("scheduler") or "MultiStepLR"}

    @staticmethod
    def _curve(p):
        total, base, kind = p["total"], p["base"], p["kind"]
        pts = []; step = max(1, total // 400)
        for i_abs in range(0, total + 1, step):
            g = p.get("gan_restart", 0)
            i = i_abs - g if g and i_abs >= g else i_abs  # schedule restarted at the GAN start
            if "warmrestart" in kind or (p["periods"] and "cosine" in kind):
                per = p["periods"] or [p["t_max"]]
                j, k = i, 0
                while k < len(per) - 1 and j >= per[k]:
                    j -= per[k]; k += 1
                T = max(1, per[min(k, len(per) - 1)])
                lr = p["eta_min"] + (base - p["eta_min"]) * (1 + math.cos(math.pi * min(j, T) / T)) / 2
            elif "cosine" in kind:
                # torch CosineAnnealingLR keeps following the cosine after T_max
                lr = p["eta_min"] + (base - p["eta_min"]) * (1 + math.cos(math.pi * i / p["t_max"])) / 2
            elif "exponential" in kind:
                lr = base * (p["gamma"] ** i)
            elif "steplr" in kind and "multi" not in kind:
                lr = base * (p["gamma"] ** (i // p["step_size"]))
            elif "plateau" in kind or "constant" in kind or "none" in kind:
                lr = base
            else:  # MultiStepLR
                lr = base * (p["gamma"] ** sum(1 for m in p["milestones"] if i >= m))
            if p["warmup"] and p["warmup"] > 0 and i_abs < p["warmup"]:
                lr = lr * i_abs / p["warmup"]
            pts.append((i_abs, lr))
        return pts


# ── Métriques ─────────────────────────────────────────────────────────────────
class MetricsDialog(QDialog):
    """Training metrics: a row of check boxes picks the charts, shown side by side in a
    grid (2 or 3 columns) rather than stacked, so many charts stay on one screen. The
    choice is remembered in the settings (metrics_dialog_charts). A checked chart with
    no data yet (Loss D during the PSNR phase…) is hidden until its first point."""

    _SETTING = "metrics_dialog_charts"
    _DEFAULT = ("loss", "loss_d", "psnr")

    def __init__(self, hist, parent=None):
        super().__init__(parent)
        from qt_ui.i18n import rich_tip
        from src.core import val_extra_metrics as vem
        self.hist = hist
        self._settings = getattr(parent, "settings", None)
        self.setWindowTitle(tr("Métriques d'Entraînement", "Training Metrics")); self.resize(1240, 820)
        self.setStyleSheet("QDialog{background:#1e1e2e;}")
        # key -> (label, color, tooltip)
        self._defs = {
            "loss": ("Loss G", "#e74c3c", tr("Perte totale du générateur (doit globalement descendre puis se stabiliser).",
                                             "Total generator loss (should go down, then level off).")),
            "loss_d": ("Loss D", "#9b59b6", tr("Perte du discriminateur (phase GAN). Stable = équilibre G/D sain ; vers 0 = D écrase G.",
                                               "Discriminator loss (GAN phase). Stable = healthy G/D balance; towards 0 = D crushes G.")),
            "psnr": ("PSNR", "#1abc9c", tr("PSNR de validation (dB) : fidélité globale à la GT, + haut = mieux. Baisse un peu en phase GAN, c'est normal.",
                                          "Validation PSNR (dB): overall fidelity to the GT, higher = better. Drops a little in the GAN phase, that's normal.")),
            "ssim": ("SSIM", "#3498db", tr("SSIM de validation : similarité de structure avec la GT, proche de 1 = mieux.",
                                          "Validation SSIM: structural similarity to the GT, closer to 1 = better.")),
        }
        for k in vem.KEYS:
            self._defs[k] = (vem.short_label(k), vem.color(k), vem.tooltip(k))
        # small reading hint next to the curve name inside each chart
        lower, higher = tr("plus bas = mieux", "lower is better"), tr("plus haut = mieux", "higher is better")
        self._hints = {"loss": lower, "loss_d": tr("stable = mieux", "stable is better"),
                       "psnr": higher, "ssim": higher,
                       "usrs_hf_ratio": tr("idéal ~0,75-1,0", "ideal ~0.75-1.0")}
        for k in vem.KEYS:
            self._hints.setdefault(k, lower if vem.better(k) == "lower" else higher)
        saved = ""
        try:
            saved = self._settings.get(self._SETTING, "") if self._settings is not None else ""
        except Exception:
            saved = ""
        on = [k for k in (saved or "").split(",") if k in self._defs] or list(self._DEFAULT)

        v = QVBoxLayout(self)
        topbar = QHBoxLayout(); v.addLayout(topbar)
        self.stat = QLabel("—"); self.stat.setStyleSheet("color:#888;")
        topbar.addWidget(self.stat); topbar.addStretch(1)
        self.overlay = QCheckBox(tr("Superposer G + D", "Overlay G + D")); topbar.addWidget(self.overlay)
        self.overlay.setToolTip(rich_tip(tr("Affiche Loss G et Loss D sur un seul graphique.", "Shows Loss G and Loss D on a single chart.")))
        self.overlay.toggled.connect(self._relayout)
        refr = QPushButton(tr("⟳ Rafraîchir", "⟳ Refresh")); refr.setStyleSheet(
            'background:#2980b9; color:#fff; border:none; border-radius:6px; padding:5px 12px; font-family:"Segoe UI Emoji";')
        refr.clicked.connect(self._refresh); topbar.addWidget(refr)

        checks = QHBoxLayout(); checks.setSpacing(12); v.addLayout(checks)
        lbl = QLabel(tr("Graphiques :", "Charts:")); lbl.setStyleSheet("color:#aaa; font-weight:700;")
        checks.addWidget(lbl)
        self.checks = {}
        for k, (label, color, tip) in self._defs.items():
            cb = QCheckBox(label); cb.setChecked(k in on)
            cb.setStyleSheet(f"QCheckBox{{color:{color}; font-weight:700;}}")
            cb.setToolTip(rich_tip(tip))
            cb.toggled.connect(self._relayout)
            self.checks[k] = cb; checks.addWidget(cb)
        checks.addStretch(1)

        self.grid_host = QWidget(); self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0); self.grid.setSpacing(8)
        v.addWidget(self.grid_host, 1)
        self._empty = QLabel(tr("Pas encore de données pour les graphiques cochés.",
                                "No data yet for the checked charts."))
        self._empty.setAlignment(Qt.AlignCenter); self._empty.setStyleSheet("color:#888; font-size:14px;")
        self._empty.hide()
        self._laid = None
        self.charts = {}
        for k, (label, color, tip) in self._defs.items():
            ch = LineChart([], title=label); ch.setMinimumSize(320, 220); ch.setToolTip(rich_tip(tip))
            self.charts[k] = ch
        self.chart_combined = LineChart([], title="Loss G + Loss D"); self.chart_combined.setMinimumSize(320, 220)

        c = QPushButton(tr("Fermer", "Close")); c.setObjectName("ghost"); c.clicked.connect(self.accept); v.addWidget(c)
        remember_geometry(self, self._settings, "metrics_dialog_geometry")
        self._relayout()
        # auto-refresh every 5 s (the dialog runs modal but the Qt event loop keeps
        # delivering the training thread's log signals, so hist keeps growing live).
        self._timer = QTimer(self); self._timer.setInterval(5000)
        self._timer.timeout.connect(self._refresh); self._timer.start()

    def _checked(self):
        return [k for k, cb in self.checks.items() if cb.isChecked()]

    def _shown(self):
        """Checked charts that have data, with Loss G + D merged when overlaid."""
        keys = [k for k in self._checked() if self.hist.get(k)]
        if self.overlay.isChecked() and "loss" in keys and "loss_d" in keys:
            i = keys.index("loss")
            keys = [k for k in keys if k not in ("loss", "loss_d")]
            keys.insert(i, "_combined")
        return keys

    def _relayout(self, *_):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.hide()
        keys = self._shown()
        self._laid = keys
        cols = 1 if len(keys) <= 1 else 2 if len(keys) <= 4 else 3
        for i, k in enumerate(keys):
            w = self.chart_combined if k == "_combined" else self.charts[k]
            r, c = divmod(i, cols)
            self.grid.addWidget(w, r, c); w.show()
        if not keys:
            self.grid.addWidget(self._empty, 0, 0); self._empty.show()
        for c in range(3):
            self.grid.setColumnStretch(c, 1 if c < max(cols, 1) else 0)
        try:
            if self._settings is not None:
                self._settings.set(self._SETTING, ",".join(self._checked()))
        except Exception:
            pass
        self._fill()

    def _refresh(self):
        # a chart gets its first point (Loss D at the GAN start…) -> show it
        if self._shown() != self._laid:
            self._relayout()
        else:
            self._fill()

    def _fill(self):
        for k in self._laid or []:
            if k == "_combined":
                self.chart_combined.set_series([("Loss G", "#e74c3c", self.hist.get("loss", [])),
                                                ("Loss D", "#9b59b6", self.hist.get("loss_d", []))])
                continue
            label, color, _tip = self._defs[k]
            name = f"{label}  ({self._hints[k]})" if k in self._hints else label
            self.charts[k].set_series([(name, color, self.hist.get(k, []))])
        counts = []
        for k in self._checked():
            n = len(self.hist.get(k, []))
            counts.append(f"{self._defs[k][0]}: {n}" if n else
                          f"{self._defs[k][0]}: " + tr("masqué (pas de données)", "hidden (no data)"))
        self.stat.setText("  |  ".join(counts) + "  ·  auto-refresh 5s")


class _ZoomImage(QLabel):
    """Image label in a QScrollArea. Linked partners (LQ + SR) share zoom and view:
    wheel = zoom under the cursor, middle button held = pan, both images at once.
    Both images are given at the same pixel size, so their geometry matches."""
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self._pm = None
        self._scale = 1.0
        self.partner = None
        self.scroll = None          # the QScrollArea holding this label (set by the dialog)
        self._drag = None

    def set_image(self, pm):
        self._pm = pm; self._render()

    def set_scale(self, s):
        self._scale = max(0.05, min(16.0, s)); self._render()

    def fit(self):
        """Scale so the whole image fits the visible pane."""
        if self._pm is None or self.scroll is None:
            return
        vp = self.scroll.viewport().size()
        if vp.width() < 50 or vp.height() < 50:
            return
        self.set_scale(min(vp.width() / self._pm.width(), vp.height() / self._pm.height()))

    def _render(self):
        if self._pm is None:
            return
        w = max(1, int(self._pm.width() * self._scale))
        # Smooth below 2x; nearest from 2x up, to see the real pixels when zooming in.
        # (Nearest at a small non-integer fit factor such as 1.06x duplicated one pixel row in
        # ~16 and made the validation images look aliased / jagged compared to the real output.)
        mode = Qt.SmoothTransformation if self._scale < 2.0 else Qt.FastTransformation
        self.setPixmap(self._pm.scaledToWidth(w, mode))
        self.adjustSize()

    def _views(self):
        return [z for z in (self, self.partner) if z is not None and z.scroll is not None]

    def wheelEvent(self, e):
        if self._pm is None or self.scroll is None:
            return
        vp_pos = self.scroll.viewport().mapFromGlobal(e.globalPosition().toPoint())
        h, v = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        old_w, old_h = max(1, self.width()), max(1, self.height())
        rx = (h.value() + vp_pos.x()) / old_w
        ry = (v.value() + vp_pos.y()) / old_h
        s = self._scale * (1.15 if e.angleDelta().y() > 0 else 1 / 1.15)
        for z in self._views():                # zoom both images simultaneously
            z.set_scale(s)
        for z in self._views():                # keep the point under the cursor in place
            z.scroll.horizontalScrollBar().setValue(int(rx * z.width() - vp_pos.x()))
            z.scroll.verticalScrollBar().setValue(int(ry * z.height() - vp_pos.y()))
        e.accept()

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._drag = e.globalPosition().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            p = e.globalPosition().toPoint()
            dx, dy = p.x() - self._drag.x(), p.y() - self._drag.y()
            self._drag = p
            for z in self._views():            # pan both images together
                hb, vb = z.scroll.horizontalScrollBar(), z.scroll.verticalScrollBar()
                hb.setValue(hb.value() - dx); vb.setValue(vb.value() - dy)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self._drag is not None:
            self._drag = None
            self.unsetCursor()
            e.accept()
            return
        super().mouseReleaseEvent(e)


# ── Validation (LQ vs SR) ─────────────────────────────────────────────────────
class ValidationDialog(QDialog):
    _EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    # Image shown on opening: the one the user last picked in the list (remembered),
    # else this default when the validation set has it, else the newest snapshot.
    _IMG_SETTING = "val_preview_image"
    _DEFAULT_IMG = "tsurune084"

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._parent = parent
        self.val_lq_dir = ""
        self._last_sr = None; self._last_lq = None
        self.setWindowTitle(tr("Prévisualisation Validation", "Validation Preview")); self.resize(1100, 640); self.setStyleSheet("QDialog{background:#1e1e2e;}")
        self.finished.connect(lambda *_: self._reload_timer.stop())
        v = QVBoxLayout(self)
        top = QHBoxLayout(); v.addLayout(top)
        top.addWidget(QLabel(tr("Dossier validation (SR) :", "Validation folder (SR):")))
        self.folder = QLineEdit(); top.addWidget(self.folder, 1)
        self._manual = False  # a folder typed / picked by the user wins over auto-detection
        self.folder.editingFinished.connect(self._typed)
        b = QPushButton("📂"); b.setFixedWidth(40); b.setStyleSheet('font-family:"Segoe UI Emoji"; font-size:13px;')
        b.clicked.connect(self._pick); top.addWidget(b)
        top.addWidget(QLabel(tr("Image :", "Image:")))
        self.combo = QComboBox(); self.combo.setMinimumWidth(200); self.combo.setFixedHeight(30)
        self.combo.currentTextChanged.connect(self._show); top.addWidget(self.combo)
        self.combo.textActivated.connect(self._remember_image)
        self.combo.setToolTip(tr(
            "Image affichée. Celle que vous choisissez ici est retenue et réaffichée à la "
            "prochaine ouverture (par défaut : tsurune084, sinon la plus récente).",
            "Image shown. The one you pick here is remembered and shown again next time "
            "(default: tsurune084, else the newest one)."))
        rb = QPushButton(tr("🔄 Recharger", "🔄 Reload")); rb.setFixedHeight(30)
        rb.setStyleSheet('font-family:"Segoe UI Emoji"; padding:0 10px;')
        rb.setToolTip(tr("Recharger la dernière image de validation produite", "Reload the last produced validation image"))
        rb.clicked.connect(self._reload); top.addWidget(rb)
        self.chk_auto = QCheckBox(tr("Auto-reload", "Auto-reload"))
        self.chk_auto.setToolTip(tr("Recharge automatiquement toutes les 3s pendant l'entraînement.",
                                    "Automatically reloads every 3s during training."))
        self.chk_auto.stateChanged.connect(self._toggle_auto_reload); top.addWidget(self.chk_auto)
        self.chk_compare = QCheckBox(tr("Mode comparaison", "Compare mode"))
        self.chk_compare.setToolTip(tr("Superpose LQ/SR avec un curseur au lieu du côte-à-côte.",
                                       "Overlays LQ/SR with a slider instead of side-by-side."))
        self.chk_compare.stateChanged.connect(self._toggle_compare); top.addWidget(self.chk_compare)

        hint = QLabel(tr("💡 Molette = zoom · clic molette maintenu = déplacer (les deux images ensemble)",
                         "💡 Wheel = zoom · hold middle button = pan (both images together)"))
        hint.setStyleSheet("color:#888;"); hint.setAlignment(Qt.AlignCenter)
        v.addWidget(hint)
        self._reload_timer = QTimer(self); self._reload_timer.setInterval(3000)
        self._reload_timer.timeout.connect(self._reload)

        self._cmp_slider = QSlider(Qt.Horizontal); self._cmp_slider.setRange(0, 100); self._cmp_slider.setValue(50)
        self._cmp_slider.valueChanged.connect(self._render_compare); self._cmp_slider.hide()
        v.addWidget(self._cmp_slider)
        self._cmp_lbl = QLabel(); self._cmp_lbl.setAlignment(Qt.AlignCenter); self._cmp_lbl.hide()
        v.addWidget(self._cmp_lbl, 1)

        self._imgs_container = QWidget(); imgs = QHBoxLayout(self._imgs_container)
        v.addWidget(self._imgs_container, 1)
        self.lq = _ZoomImage(); self.lq.setStyleSheet("color:#e67e22;")
        self.sr = _ZoomImage(); self.sr.setStyleSheet("color:#2ecc71;")
        self.lq.partner = self.sr; self.sr.partner = self.lq  # synchronized zoom
        self.lq_name = QLabel(""); self.lq_name.setAlignment(Qt.AlignCenter); self.lq_name.setStyleSheet("color:#888;")
        self.sr_name = QLabel(""); self.sr_name.setAlignment(Qt.AlignCenter); self.sr_name.setStyleSheet("color:#888;")
        for title, lbl, nm, col in ((tr("LQ (Entrée)", "LQ (Input)"), self.lq, self.lq_name, "#e67e22"),
                                    (tr("SR (Sortie)", "SR (Output)"), self.sr, self.sr_name, "#2ecc71")):
            c = QVBoxLayout()
            tt = QLabel(title); tt.setAlignment(Qt.AlignCenter); tt.setStyleSheet(f"color:{col}; font-weight:700;")
            # scroll area so zooming scrolls inside a fixed pane instead of resizing the window
            sa = QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(lbl)
            sa.setAlignment(Qt.AlignCenter)
            sa.setStyleSheet("QScrollArea{border:none; background:transparent;}")
            lbl.scroll = sa
            c.addWidget(tt); c.addWidget(sa, 1); c.addWidget(nm); imgs.addLayout(c, 1)
        for a, b in ((self.lq.scroll, self.sr.scroll), (self.sr.scroll, self.lq.scroll)):
            a.horizontalScrollBar().valueChanged.connect(b.horizontalScrollBar().setValue)
            a.verticalScrollBar().valueChanged.connect(b.verticalScrollBar().setValue)
        cl = QPushButton(tr("Fermer", "Close")); cl.setObjectName("ghost"); cl.clicked.connect(self.accept); v.addWidget(cl)
        remember_geometry(self, self.settings, "validation_dialog_geometry")
        self._auto_detect()

    @staticmethod
    def _stem(name):
        """Image identity across iterations: 'tsurune084_700.png' / 'tsurune084' -> 'tsurune084'."""
        s = os.path.splitext(name)[0]
        return re.sub(r"_\d+$", "", s)

    def _remember_image(self, name):
        try:
            if self.settings is not None and name:
                self.settings.set(self._IMG_SETTING, self._stem(name))
        except Exception:
            pass

    def _preferred(self, items):
        """Item to select: remembered image > tsurune084 > None (caller picks the newest).
        In a flat folder the newest snapshot of that image wins."""
        try:
            saved = (self.settings.get(self._IMG_SETTING, "") if self.settings is not None else "") or ""
        except Exception:
            saved = ""
        for want in (saved, self._DEFAULT_IMG):
            if not want:
                continue
            hits = [i for i in items if self._stem(i) == want]
            if hits:
                folder = self.folder.text().strip()
                try:
                    return max(hits, key=lambda i: os.path.getmtime(os.path.join(folder, i)))
                except OSError:
                    return hits[-1]
        return None

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self._fit_both)

    def _fit_both(self):
        """Fit the SR image to its pane and give the LQ the very same scale."""
        if self.sr._pm is not None:
            self.sr.fit()
            if self.lq._pm is not None:
                self.lq.set_scale(self.sr._scale)

    def _auto_detect(self):
        """Auto-fill the validation visualization folder of the currently-training
        experiment + select the most recently produced image. Probes the active
        engine root AND the ~/USRS engine locations (not just the currently
        selected script) so this still works right after switching engines."""
        try:
            cfg_path = self._parent.entries["config_path"].text().strip() if self._parent else ""
            script = self._parent.entries["script_path"].text().strip() if self._parent else ""
            if not (cfg_path and os.path.isfile(cfg_path)):
                return
            if cfg_path.lower().endswith(".toml"):  # NeoSR: TOML (it was read with YAML)
                import tomllib
                with open(cfg_path, "rb") as f:
                    cfg = tomllib.load(f) or {}
            else:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            name = cfg.get("name", "")
            # val LQ dir (for the LQ side)
            ds = cfg.get("datasets", {})
            entries = ds.values() if isinstance(ds, dict) else (ds if isinstance(ds, list) else [])
            for d in entries:
                if isinstance(d, dict) and str(d.get("name", "")).lower().startswith("val"):
                    lq = d.get("dataroot_lq", "")
                    self.val_lq_dir = (lq[0] if isinstance(lq, list) and lq else lq) or ""
                    break
            from src.core.engine_paths import base_engine_path as _ebp
            usrs = _ebp()
            roots = []
            if script:
                roots.append(os.path.dirname(script))
            roots += [os.path.join(usrs, "traiNNer-redux"), os.path.join(usrs, "neosr")]
            for engine_dir in roots:
                vis = os.path.join(engine_dir, "experiments", name, "visualization")
                if os.path.isdir(vis):
                    self.folder.setText(vis)
                    self._load(vis)
                    return
        except Exception:
            pass

    def _pick(self):
        p = QFileDialog.getExistingDirectory(self, tr("Dossier images validation", "Validation images folder"))
        if p:
            self.folder.setText(p); self.val_lq_dir = ""; self._manual = True; self._load(p)

    def _typed(self):
        p = self.folder.text().strip()
        if p and os.path.isdir(p):
            self._manual = True; self._load(p)

    def _reload(self):
        """Reload: the user's own folder if one was chosen (auto-detection used to replace
        it), else the running experiment's visualization folder."""
        p = self.folder.text().strip()
        if self._manual and p and os.path.isdir(p):
            cur = self.combo.currentText()
            self._load(p)
            if cur and self.combo.findText(cur) >= 0:
                self.combo.setCurrentText(cur)
        else:
            self._auto_detect()

    def _latest_png(self, d):
        """Newest .png in a dir (redux saves <name>_<iter>.png snapshots)."""
        # not the LQ / GT copies some engines save next to the SR snapshots
        pngs = [os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in self._EXT
                and not re.search(r"_(lq|lr|gt|hr)$", os.path.splitext(f)[0], re.IGNORECASE)]
        return max(pngs, key=os.path.getmtime) if pngs else None

    def _load(self, folder):
        # redux layout: visualization/<imgname>/<imgname>_<iter>.png (subfolders)
        subdirs = [f for f in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, f))]
        self._nested = bool(subdirs)
        self.combo.blockSignals(True); self.combo.clear()
        if subdirs:
            self.combo.addItems(subdirs)
            # remembered / default image, else the image whose latest snapshot is the newest
            sel = self._preferred(subdirs)
            if sel is None:
                newest = 0
                for s in subdirs:
                    lp = self._latest_png(os.path.join(folder, s))
                    if lp and os.path.getmtime(lp) > newest:
                        newest, sel = os.path.getmtime(lp), s
            if sel:
                self.combo.setCurrentText(sel)
        else:
            files = [f for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in self._EXT]
            self.combo.addItems(files)
            sel = self._preferred(files)
            if sel:
                self.combo.setCurrentText(sel)
        self.combo.blockSignals(False)
        if self.combo.count():
            self._show(self.combo.currentText())

    def _show(self, name):
        folder = self.folder.text().strip()
        if not name or not folder:
            return
        from PIL import Image
        try:
            if getattr(self, "_nested", False):
                srp = self._latest_png(os.path.join(folder, name))
            else:
                srp = os.path.join(folder, name)
            if not srp or not os.path.isfile(srp):
                self.sr.setText(tr("(SR introuvable)", "(SR not found)")); return
            sr = Image.open(srp).convert("RGB")
            self.sr.set_image(_pixmap(sr))
            self.sr_name.setText(os.path.basename(srp))
            self._last_sr, self._last_lq = sr, None
            # LQ: from val LQ dataset folder (same image name). Also tries Redux's
            # "{name}_lr.png" naming convention and prefix matching (base names don't
            # always match exactly) — the previous exact-stem-only match missed those.
            stem = name if self._nested else os.path.splitext(name)[0]
            stem = re.sub(r"_lr$", "", stem, flags=re.IGNORECASE)
            lqp = None
            for base in (self.val_lq_dir, os.path.join(os.path.dirname(folder), "LQ")):
                if base and os.path.isdir(base):
                    by_stem = {}
                    for f in os.listdir(base):
                        s = os.path.splitext(f)[0]
                        by_stem[s.lower()] = os.path.join(base, f)
                    sl = stem.lower()
                    if sl in by_stem:
                        lqp = by_stem[sl]
                    else:
                        for s_l, fp in by_stem.items():
                            if s_l.startswith(sl) or sl.startswith(s_l):
                                lqp = fp; break
                if lqp:
                    break
            if lqp:
                lq = Image.open(lqp).convert("RGB")
                if lq.size != sr.size:   # same displayed size as the SR, real LQ pixels
                    lq = lq.resize(sr.size, Image.NEAREST)
                self.lq.set_image(_pixmap(lq))
                self.lq_name.setText(os.path.basename(lqp))
                self._last_lq = lq
            else:
                self.lq.setText(tr("(LQ introuvable)", "(LQ not found)")); self.lq_name.setText("")
        except Exception as e:
            self.sr.setText(f"❌ {e}")
        self._fit_both()
        if self.chk_compare.isChecked():
            self._render_compare()

    def _toggle_auto_reload(self, *_):
        if self.chk_auto.isChecked():
            self._reload_timer.start()
        else:
            self._reload_timer.stop()

    def _toggle_compare(self, *_):
        on = self.chk_compare.isChecked()
        self._imgs_container.setVisible(not on)
        self._cmp_slider.setVisible(on)
        self._cmp_lbl.setVisible(on)
        if on:
            self._render_compare()

    def _render_compare(self, *_):
        """Overlay LQ|SR with a vertical divider driven by the slider (same technique
        as the Comparateur tool page) — CTk offered this compare mode, the Qt port
        only ever had side-by-side."""
        sr, lq = getattr(self, "_last_sr", None), getattr(self, "_last_lq", None)
        if sr is None:
            return
        from PIL import Image
        if lq is None:  # without the LQ the "comparison" was SR against SR
            self._cmp_lbl.setText(tr("(LQ introuvable : comparaison impossible)", "(LQ not found: nothing to compare)"))
            return
        base = lq.resize(sr.size)
        comp = base.copy()
        x = int(sr.width * self._cmp_slider.value() / 100)
        if x > 0:
            comp.paste(sr.crop((0, 0, x, sr.height)), (0, 0))
        comp.thumbnail((max(200, self._cmp_lbl.width()), max(200, self._cmp_lbl.height())), Image.LANCZOS)
        self._cmp_lbl.setPixmap(_pixmap(comp))


def ai_request(provider, model, api_key, prompt):
    """Call an AI chat API (faithful to the CTk logic) and return the text reply."""
    import urllib.request
    import urllib.error
    import json as _json
    headers = {"Content-Type": "application/json", "User-Agent": "UniversalSRStudio/Qt"}
    if "OpenRouter" in provider:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"; headers["HTTP-Referer"] = "https://github.com/Universal-SR-Studio"
    elif "Anthropic" in provider:
        url = "https://api.anthropic.com/v1/messages"
        headers["x-api-key"] = api_key; headers["anthropic-version"] = "2023-06-01"
    elif "OpenAI" in provider or "ChatGPT" in provider:
        url = "https://api.openai.com/v1/chat/completions"; headers["Authorization"] = f"Bearer {api_key}"
    elif "Google" in provider or "Gemini" in provider:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    elif "xAI" in provider or "Grok" in provider:
        url = "https://api.x.ai/v1/chat/completions"; headers["Authorization"] = f"Bearer {api_key}"
    elif "DeepSeek" in provider:
        url = "https://api.deepseek.com/chat/completions"; headers["Authorization"] = f"Bearer {api_key}"
    elif "NVIDIA" in provider:
        url = "https://integrate.api.nvidia.com/v1/chat/completions"; headers["Authorization"] = f"Bearer {api_key}"
    else:
        return "❌ " + tr("Fournisseur non supporté.", "Unsupported provider.")
    if "generativelanguage" in url:
        body = _json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    else:
        # OpenAI reasoning models (GPT-5+ / GPT-6, o-series) reject max_tokens and count
        # their reasoning inside max_completion_tokens: 2000 could leave an empty answer.
        _lim = ({"max_completion_tokens": 16000} if ("OpenAI" in provider or "ChatGPT" in provider)
                else {"max_tokens": 4000})
        body = _json.dumps({"model": model, **_lim,
                            "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = _json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return "❌ " + tr(f"Erreur HTTP {e.code} :\n{(e.read().decode() if e.fp else '')[:600]}", f"HTTP Error {e.code}:\n{(e.read().decode() if e.fp else '')[:600]}")
    except Exception as e:
        return "❌ " + tr(f"Erreur réseau : {e}", f"Network error: {e}")
    if "content" in result and isinstance(result["content"], list):
        return result["content"][0].get("text", str(result))
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    if "candidates" in result:
        parts = result["candidates"][0].get("content", {}).get("parts", [])
        return parts[0].get("text", str(result)) if parts else str(result)
    return str(result)


class _AIWorker(QThread):
    done = Signal(str)

    def __init__(self, provider, model, key, prompt):
        super().__init__(); self.provider, self.model, self.key, self.prompt = provider, model, key, prompt

    def run(self):
        try:
            self.done.emit(ai_request(self.provider, self.model, self.key, self.prompt))
        except Exception as e:
            self.done.emit(f"❌ {e}")


# ── Analyse IA ────────────────────────────────────────────────────────────────
class AIDialog(QDialog):
    # provider → (settings key, [models]) — shared source of truth
    try:
        from src.core.ai_providers import AI_MODELS as _PROVIDERS
    except Exception:
        _PROVIDERS = {"OpenRouter (Gratuit)": ("api_key_openrouter", ["openrouter/free"])}

    @property
    def _MODELS(self):
        return {k: v[1] for k, v in self._PROVIDERS.items()}

    def _on_provider(self, name):
        models = self._PROVIDERS.get(name, ("", []))[1]
        self.model.clear(); self.model.addItems(models)
        saved = self.settings.get("ai_model", "")
        if saved and saved in models:
            self.model.setCurrentText(saved)
        # auto-load the saved per-provider API key
        skey = self._PROVIDERS.get(name, ("", []))[0]
        if skey and hasattr(self, "key"):
            self.key.setText(self.settings.get(skey, "") or "")

    def __init__(self, settings, log_text="", config_text="", parent=None):
        super().__init__(parent)
        self.settings = settings; self.log_text = log_text; self.config_text = config_text
        self.setWindowTitle(tr("Analyse IA — Entraînement", "AI Analysis — Training")); self.resize(760, 720)
        v = QVBoxLayout(self)
        t = QLabel(tr("🤖 Analyse IA de l'entraînement", "🤖 AI training analysis")); t.setStyleSheet("color:#e67e22; font-weight:700; font-size:14px;")
        t.setAlignment(Qt.AlignCenter); v.addWidget(t)
        row = QHBoxLayout(); v.addLayout(row); row.addWidget(QLabel("Fournisseur :"))
        from qt_ui.widgets import KeyedCombo
        self.provider = KeyedCombo(list(self._MODELS.keys()))
        self.provider.keyChanged.connect(self._on_provider)
        row.addWidget(self.provider, 1)
        row2 = QHBoxLayout(); v.addLayout(row2); row2.addWidget(QLabel(tr("Modèle :", "Model:")))
        self.model = QComboBox(); self.model.setEditable(True)
        from qt_ui.widgets import combo_open_on_click
        combo_open_on_click(self.model)  # click anywhere opens the list (still typable)
        row2.addWidget(self.model, 1)
        v.addWidget(QLabel(tr("Clé API :", "API Key:")))
        self.key = QLineEdit(self.settings.get("ai_api_key", "")); self.key.setEchoMode(QLineEdit.Password)
        v.addWidget(self.key)
        # Log d'entraînement : N dernières lignes / tout
        v.addWidget(QLabel(tr("Log d'entraînement :", "Training log:")))
        lr = QHBoxLayout(); v.addLayout(lr)
        self.rb_last = QRadioButton(tr("Dernières", "Last")); self.rb_last.setChecked(True)
        self.log_lines = QLineEdit("200"); self.log_lines.setFixedWidth(60)
        self.rb_all = QRadioButton(tr("Tout le log (60 000 derniers caractères max)", "Whole log (last 60,000 characters max)"))
        grp = QButtonGroup(self); grp.addButton(self.rb_last); grp.addButton(self.rb_all)
        lr.addWidget(self.rb_last); lr.addWidget(self.log_lines); lr.addWidget(QLabel(tr("lignes", "lines")))
        lr.addSpacing(16); lr.addWidget(self.rb_all); lr.addStretch(1)
        self._on_provider(self.provider.currentText())
        v.addWidget(QLabel(tr("Question à l'IA :", "Question to the AI:")))
        self.question = QPlainTextEdit(tr("Que penses-tu de mon entraînement pour le moment ? "
                                       "Y a-t-il des signes de sur-apprentissage, sous-apprentissage ou problèmes à corriger ?",
                                       "What do you think of my training so far? "
                                       "Are there signs of overfitting, underfitting, or issues to fix?"))
        self.question.setFixedHeight(70); v.addWidget(self.question)
        v.addWidget(QLabel(tr("Réponse IA :", "AI Response:")))
        self.answer = QPlainTextEdit(); self.answer.setReadOnly(True); v.addWidget(self.answer, 1)
        bot = QHBoxLayout(); v.addLayout(bot)
        send = QPushButton(tr("📨 Envoyer", "📨 Send")); send.setObjectName("warn"); send.clicked.connect(self._send)
        bot.addWidget(send, 1)
        cp = QPushButton(tr("📋 Copier", "📋 Copy")); cp.setObjectName("ghost")
        cp.clicked.connect(lambda: self.answer.selectAll() or self.answer.copy()); bot.addWidget(cp)
        # dark background on the TEXT inputs/outputs only — the combos keep the global
        # accent style (the user wants provider/model dropdowns in the accent colour).
        for _wb in (self.key, self.log_lines, self.question, self.answer):
            _wb.setStyleSheet(_wb.styleSheet() + "background:#1e1e2e; color:#dce4ee;")

    def _send(self):
        provider = self.provider.currentText()
        key = self.key.text().strip()
        if not key:
            QMessageBox.warning(self, tr("Analyse IA", "AI Analysis"), tr("Renseignez une clé API.", "Enter an API key.")); return
        # persist key against its provider slot (shared with the Settings tab)
        skey = self._PROVIDERS.get(provider, ("", []))[0]
        if skey:
            self.settings.set(skey, key)
        self.settings.set("ai_model", self.model.currentText().strip())
        log = self.log_text
        if self.rb_last.isChecked():
            try:
                n = int(self.log_lines.text())
                log = "\n".join(log.splitlines()[-n:])
            except ValueError:
                pass
        # the LAST part of the log is the useful one (it was cut from the start: [:12000])
        log = log[-60000:] if self.rb_all.isChecked() else log[-20000:]
        prompt = (f"{self.question.toPlainText()}\n\n"
                  f"=== CONFIG ===\n{self.config_text[:16000]}\n\n=== LOG ===\n{log}")
        self.answer.setPlainText("⏳ " + tr("Envoi à l'IA…", "Sending to the AI…"))
        self._aiw = _AIWorker(provider, self.model.currentText().strip(), key, prompt)
        self._aiw.done.connect(lambda r: self.answer.setPlainText(r))
        self._aiw.start()
