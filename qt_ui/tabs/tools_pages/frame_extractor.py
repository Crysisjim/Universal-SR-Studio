"""
frame_extractor.py — Tools › Extraction de Frames (PySide6).

Extracts still frames from a video via src.core.lq_generator (extract_frame_at /
extract_frames / probe_video), in one of three modes:
  - a single frame by number,
  - one frame every N seconds,
  - every frame.
Output format/bit-depth reuses lq_generator.EXTRACT_FORMATS (png8/png16/tiff8/tiff16/jpg) —
the same genuinely-16-bit pipeline already used by the LQ Generator and Quick Upscale.
"""
from __future__ import annotations

import os
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QFileDialog,
    QRadioButton, QButtonGroup, QStackedWidget, QProgressBar,
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QFont, QPixmap

from qt_ui.i18n import tr

_VIDEO_FILTER = ("Videos (*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.ts *.wmv *.flv);;"
                  + tr("Tous les fichiers", "All files") + " (*)")

_FORMAT_ITEMS = [
    ("png8", "PNG (8-bit)"),
    ("png16", "PNG (16-bit)"),
    ("tiff8", "TIFF (8-bit)"),
    ("tiff16", "TIFF (16-bit)"),
    ("jpg", "JPEG (8-bit)"),
    ("bmp", "BMP (8-bit)"),
]


class _ProbeWorker(QThread):
    result = Signal(dict)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            from src.core.lq_generator import probe_video
            self.result.emit(probe_video(self.video_path))
        except Exception:
            self.result.emit({})


class _ExtractWorker(QThread):
    line = Signal(str)
    progress = Signal(int)
    preview = Signal(str)
    done = Signal(bool, str, str)  # ok, info, preview_path ("" if none)

    def __init__(self, mode, video_path, out_dir, prefix, frame_number, interval_s,
                 img_format, jpg_quality):
        super().__init__()
        self.mode = mode
        self.video_path = video_path
        self.out_dir = out_dir
        self.prefix = prefix
        self.frame_number = frame_number
        self.interval_s = interval_s
        self.img_format = img_format
        self.jpg_quality = jpg_quality
        self._stop_requested = False
        self._last_preview = None

    def stop(self):
        self._stop_requested = True

    def _on_progress(self, n):
        self.progress.emit(n)
        # Best-effort live preview: peek at the most recently modified output file so
        # far. Piggybacks on ffmpeg's own progress cadence (~every 0.5s in practice, see
        # extract_frames) instead of polling the filesystem on a separate timer.
        try:
            entries = [os.path.join(self.out_dir, f) for f in os.listdir(self.out_dir)]
            entries = [p for p in entries if os.path.isfile(p)]
            if entries:
                latest = max(entries, key=os.path.getmtime)
                if latest != self._last_preview:
                    self._last_preview = latest
                    self.preview.emit(latest)
        except Exception:
            pass

    def run(self):
        try:
            from src.core.lq_generator import extract_frame_at, extract_frames
            os.makedirs(self.out_dir, exist_ok=True)
            if self.mode == "single":
                self.line.emit(tr(f"Extraction de la frame {self.frame_number}…",
                                   f"Extracting frame {self.frame_number}…"))
                out_path = os.path.join(self.out_dir, f"{self.prefix}_{self.frame_number:06d}")
                p = extract_frame_at(self.video_path, out_path, self.frame_number,
                                      img_format=self.img_format, jpg_quality=self.jpg_quality,
                                      cancel_check=lambda: self._stop_requested)
                self.line.emit(tr(f"✅ Frame extraite : {p}", f"✅ Frame extracted: {p}"))
                self.done.emit(True, self.out_dir, p)
            else:
                fps = 0.0 if self.mode == "all" else (1.0 / self.interval_s if self.interval_s > 0 else 0.0)
                paths = extract_frames(self.video_path, self.out_dir, self.prefix, fps=fps,
                                        progress=self._on_progress,
                                        log=self.line.emit,
                                        img_format=self.img_format, jpg_quality=self.jpg_quality,
                                        cancel_check=lambda: self._stop_requested)
                if self._stop_requested:
                    # B30: a stopped extraction is not a success (frames kept are partial)
                    msg = tr(f"Arrêté — {len(paths)} frame(s) partielle(s) dans : {self.out_dir}",
                             f"Stopped — {len(paths)} partial frame(s) in: {self.out_dir}")
                    self.line.emit("⏹ " + msg)
                    self.done.emit(False, msg, paths[-1] if paths else "")
                    return
                self.line.emit(tr(f"✅ {len(paths)} frame(s) extraite(s) dans : {self.out_dir}",
                                   f"✅ {len(paths)} frame(s) extracted to: {self.out_dir}"))
                self.done.emit(bool(paths), self.out_dir, paths[-1] if paths else "")
        except InterruptedError as e:
            self.line.emit(f"⏹ {e}")
            self.done.emit(False, str(e), "")
        except Exception as e:
            self.line.emit(f"❌ {e}")
            self.done.emit(False, str(e), "")


class FrameExtractorPage(QWidget):
    toolProgress = Signal(int)   # 0-100, wired to the Tools nav sidebar's spinner badge
    toolFinished = Signal(bool)  # True = success (green badge), False = failure (red badge)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.worker = None
        self.probe_worker = None
        self._probe_info = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        t = QLabel(tr("Extraction de Frames", "Frame Extraction")); t.setObjectName("title")
        root.addWidget(t)
        root.addWidget(QLabel(tr(
            "Extrait des images fixes depuis une vidéo — une frame précise, une frame "
            "toutes les X secondes, ou toutes les frames.",
            "Extracts still frames from a video — a specific frame, one frame every X "
            "seconds, or every frame.")))

        # ── Vidéo source ──
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Vidéo :", "Video:")))
        self.edit_video = QLineEdit()
        self.edit_video.setToolTip(tr("Fichier vidéo source.", "Source video file."))
        self.edit_video.editingFinished.connect(self._probe)
        row.addWidget(self.edit_video, 1)
        b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedWidth(40)
        b.clicked.connect(self._pick_video); row.addWidget(b)
        root.addLayout(row)

        self.lbl_info = QLabel(tr("(sélectionnez une vidéo pour voir ses infos)",
                                   "(select a video to see its info)"))
        self.lbl_info.setStyleSheet("color:#888;")
        root.addWidget(self.lbl_info)

        # ── Mode ──
        mode_row = QHBoxLayout()
        self.rb_single = QRadioButton(tr("Frame N°", "Frame #"))
        self.rb_interval = QRadioButton(tr("1 frame toutes les (s)", "1 frame every (s)"))
        self.rb_all = QRadioButton(tr("Toutes les frames", "Every frame"))
        self.rb_interval.setChecked(True)
        self._mode_group = QButtonGroup(self)
        for rb in (self.rb_single, self.rb_interval, self.rb_all):
            self._mode_group.addButton(rb)
            mode_row.addWidget(rb)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        self._mode_stack = QStackedWidget()
        # index 0: single frame number
        w0 = QWidget(); l0 = QHBoxLayout(w0); l0.setContentsMargins(0, 0, 0, 0)
        l0.addWidget(QLabel(tr("Numéro de frame :", "Frame number:")))
        self.spin_frame = QSpinBox(); self.spin_frame.setRange(1, 10_000_000); self.spin_frame.setValue(1)
        l0.addWidget(self.spin_frame); l0.addStretch(1)
        self._mode_stack.addWidget(w0)
        # index 1: interval seconds
        w1 = QWidget(); l1 = QHBoxLayout(w1); l1.setContentsMargins(0, 0, 0, 0)
        l1.addWidget(QLabel(tr("Intervalle (secondes) :", "Interval (seconds):")))
        self.spin_interval = QDoubleSpinBox(); self.spin_interval.setRange(0.04, 3600.0)
        self.spin_interval.setValue(5.0); self.spin_interval.setDecimals(2)
        l1.addWidget(self.spin_interval); l1.addStretch(1)
        self._mode_stack.addWidget(w1)
        # index 2: all frames — nothing to configure
        w2 = QWidget(); l2 = QHBoxLayout(w2); l2.setContentsMargins(0, 0, 0, 0)
        l2.addWidget(QLabel(tr("Extrait chaque frame de la vidéo (peut être volumineux).",
                                "Extracts every frame of the video (can be large).")))
        l2.addStretch(1)
        self._mode_stack.addWidget(w2)
        self._mode_stack.setCurrentIndex(1)
        root.addWidget(self._mode_stack)

        self.rb_single.toggled.connect(lambda on: on and self._mode_stack.setCurrentIndex(0))
        self.rb_interval.toggled.connect(lambda on: on and self._mode_stack.setCurrentIndex(1))
        self.rb_all.toggled.connect(lambda on: on and self._mode_stack.setCurrentIndex(2))

        # ── Sortie ──
        row2 = QHBoxLayout()
        row2.addWidget(QLabel(tr("Dossier de sortie :", "Output folder:")))
        self.edit_out = QLineEdit()
        row2.addWidget(self.edit_out, 1)
        b2 = QPushButton("…"); b2.setObjectName("ghost"); b2.setFixedWidth(40)
        b2.clicked.connect(self._pick_out); row2.addWidget(b2)
        root.addLayout(row2)

        self.chk_subfolder = QCheckBox(tr("Créer un sous-dossier nommé d'après la vidéo",
                                           "Create a subfolder named after the video"))
        self.chk_subfolder.setChecked(True)
        self.chk_subfolder.setToolTip(tr(
            "Coché : les frames vont dans <dossier de sortie>/<nom de la vidéo>/.\n"
            "Décoché : les frames vont directement dans le dossier de sortie.",
            "Checked: frames go into <output folder>/<video name>/.\n"
            "Unchecked: frames go directly into the output folder."))
        root.addWidget(self.chk_subfolder)

        # ── Format ──
        row3 = QHBoxLayout()
        row3.addWidget(QLabel(tr("Format :", "Format:")))
        self.combo_format = QComboBox()
        for key, label in _FORMAT_ITEMS:
            self.combo_format.addItem(label, key)
        row3.addWidget(self.combo_format)
        row3.addSpacing(16)
        self.lbl_quality = QLabel(tr("Qualité JPEG :", "JPEG quality:"))
        self.spin_quality = QSpinBox(); self.spin_quality.setRange(1, 100); self.spin_quality.setValue(95)
        row3.addWidget(self.lbl_quality); row3.addWidget(self.spin_quality)
        row3.addStretch(1)
        self.combo_format.currentIndexChanged.connect(self._sync_quality_visibility)
        self._sync_quality_visibility()
        root.addLayout(row3)

        # ── Action ──
        row4 = QHBoxLayout()
        self.btn = QPushButton(tr("Extraire", "Extract"))
        self.btn.clicked.connect(self._run)
        row4.addWidget(self.btn)
        self.btn_stop = QPushButton("⏹ " + tr("Arrêter", "Stop"))
        self.btn_stop.setVisible(False)
        self.btn_stop.setStyleSheet(
            'QPushButton{background:#e74c3c; color:#fff; border:none; border-radius:6px; padding:9px;'
            ' font-weight:700; font-family:"Segoe UI Emoji"; font-size:13px;}'
            'QPushButton:hover{background:#ef6657;}'
            'QPushButton:pressed{background:#b83a2e; padding-top:11px;}')
        self.btn_stop.clicked.connect(self._stop)
        row4.addWidget(self.btn_stop)
        row4.addStretch(1)
        root.addLayout(row4)

        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        prog_row.addWidget(self.progress_bar, 1)
        self.lbl_eta = QLabel("")
        self.lbl_eta.setStyleSheet("color:#888;")
        self.lbl_eta.setFixedWidth(90)
        prog_row.addWidget(self.lbl_eta)
        root.addLayout(prog_row)

        bottom = QHBoxLayout()
        self.out = QPlainTextEdit(); self.out.setReadOnly(True)
        self.out.setFont(QFont("Consolas", 10)); self.out.setMaximumBlockCount(5000)
        bottom.addWidget(self.out, 2)

        self.preview_label = QLabel(tr("(aperçu)", "(preview)"))
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border:1px solid #3a3a4a; border-radius:8px; color:#888;")
        self.preview_label.setMinimumSize(280, 160)
        self.preview_label.setScaledContents(False)
        bottom.addWidget(self.preview_label, 1)
        root.addLayout(bottom, 1)

    def _sync_quality_visibility(self):
        is_jpg = self.combo_format.currentData() == "jpg"
        self.lbl_quality.setVisible(is_jpg)
        self.spin_quality.setVisible(is_jpg)

    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Choisir une vidéo", "Choose a video"), "", _VIDEO_FILTER)
        if p:
            self.edit_video.setText(p)
            self._probe()

    def _pick_out(self):
        p = QFileDialog.getExistingDirectory(self, tr("Dossier de sortie", "Output folder"))
        if p:
            self.edit_out.setText(p)

    def _probe(self):
        video = self.edit_video.text().strip()
        if not video or not os.path.isfile(video):
            return
        self.lbl_info.setText(tr("Analyse de la vidéo…", "Analyzing video…"))
        self.probe_worker = _ProbeWorker(video)
        self.probe_worker.result.connect(self._probe_done)
        self.probe_worker.start()

    def _probe_done(self, info: dict):
        self._probe_info = info or {}
        if not info or not info.get("fps"):
            self.lbl_info.setText(tr("(infos indisponibles — ffprobe introuvable ?)",
                                      "(info unavailable — ffprobe not found?)"))
            return
        fps = info["fps"]; dur = info["duration"]; nb = info["nb_frames"]
        w, h = info["width"], info["height"]
        m, s = divmod(int(dur), 60)
        h_, m = divmod(m, 60)
        self.lbl_info.setText(
            f"{w}×{h} · {fps:.2f} fps · {h_:02d}:{m:02d}:{s:02d} · "
            + tr(f"≈{nb} frames", f"≈{nb} frames"))
        if nb:
            self.spin_frame.setRange(1, max(1, nb))

    def _run(self):
        video = self.edit_video.text().strip()
        out_dir = self.edit_out.text().strip()
        if not video or not os.path.isfile(video):
            self.out.appendPlainText(tr("❌ Vidéo introuvable.", "❌ Video not found."))
            return
        if not out_dir:
            self.out.appendPlainText(tr("❌ Choisissez un dossier de sortie.", "❌ Choose an output folder."))
            return
        prefix = os.path.splitext(os.path.basename(video))[0]
        prefix = "".join(c if (c.isalnum() or c in "-_") else "_" for c in prefix) or "frame"
        target_dir = os.path.join(out_dir, prefix) if self.chk_subfolder.isChecked() else out_dir

        mode = "single" if self.rb_single.isChecked() else ("all" if self.rb_all.isChecked() else "interval")
        img_format = self.combo_format.currentData()
        jpg_quality = self.spin_quality.value()

        self.out.clear()
        self.lbl_eta.setText("")
        self._extract_start = time.monotonic()
        self.preview_label.setText(tr("(aperçu)", "(preview)")); self.preview_label.setPixmap(QPixmap())
        # ffmpeg's "frame=" progress counter (fed to `progress`) tracks DECODED frames,
        # not just the ones written out — even with an fps filter dropping most of them,
        # ffmpeg still decodes every frame of the video to apply it. So the probed total
        # frame count is the right ceiling for both "every frame" and "every N seconds",
        # not the (much smaller) expected output count.
        nb = self._probe_info.get("nb_frames") or 0
        if mode == "single" or not nb:
            self.progress_bar.setRange(0, 0)  # indeterminate — no reliable total to show
        else:
            self.progress_bar.setRange(0, nb)
            self.progress_bar.setValue(0)
        self.btn.setEnabled(False); self.btn.setText(tr("Extraction…", "Extracting…"))
        self.btn_stop.setVisible(True); self.btn_stop.setEnabled(True)
        self.toolProgress.emit(0)
        self.worker = _ExtractWorker(
            mode, video, target_dir, prefix,
            self.spin_frame.value(), self.spin_interval.value(),
            img_format, jpg_quality)
        self.worker.line.connect(self.out.appendPlainText)
        self.worker.progress.connect(self._on_progress)
        self.worker.preview.connect(self._show_preview)
        self.worker.done.connect(self._done)
        self.worker.start()

    def _on_progress(self, n: int):
        self.progress_bar.setValue(n)
        maxv = self.progress_bar.maximum()
        if maxv > 0:
            self.toolProgress.emit(min(99, int(n * 100 / maxv)))
            elapsed = time.monotonic() - self._extract_start
            if n > 0 and elapsed > 1.0:
                rate = n / elapsed
                remaining_s = max(0.0, (maxv - n) / rate) if rate > 0 else 0.0
                m, s = divmod(int(remaining_s), 60)
                self.lbl_eta.setText(tr(f"ETA {m:02d}:{s:02d}", f"ETA {m:02d}:{s:02d}"))

    def _stop(self):
        if self.worker:
            self.btn_stop.setEnabled(False); self.btn_stop.setText(tr("Arrêt…", "Stopping…"))
            self.worker.stop()

    def _show_preview(self, path: str):
        if not path or not os.path.isfile(path):
            return
        pix = QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(self.preview_label.width() or 280, self.preview_label.height() or 160,
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(pix)

    def _done(self, ok: bool, info: str, preview_path: str):
        self.btn.setEnabled(True); self.btn.setText(tr("Extraire", "Extract"))
        self.btn_stop.setVisible(False); self.btn_stop.setText("⏹ " + tr("Arrêter", "Stop"))
        self.lbl_eta.setText("")
        if ok and self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(self.progress_bar.maximum())
        else:
            self.progress_bar.setRange(0, 1); self.progress_bar.setValue(1 if ok else 0)
        if ok and preview_path and os.path.isfile(preview_path):
            self._show_preview(preview_path)
        elif not ok:
            self.preview_label.setText(tr("(aperçu indisponible)", "(preview unavailable)"))
        try:
            from qt_ui.notify import notify
            if ok:
                notify(self.settings, tr("Extraction terminée", "Extraction finished"),
                       tr(f"Frames extraites dans : {info}", f"Frames extracted to: {info}"),
                       "notif_win11_batch")
            elif getattr(self.worker, "_stop_requested", False):
                notify(self.settings, tr("Extraction arrêtée", "Extraction stopped"), info, "notif_win11_batch")
            else:
                notify(self.settings, tr("Extraction échouée", "Extraction failed"),
                       info, "notif_win11_errors")
        except Exception:
            pass
        self.toolFinished.emit(bool(ok))
        self.worker = None
