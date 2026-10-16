"""
quick_upscale.py — Tools › Quick Upscale (PySide6) — faithful port of the CustomTkinter
create_page_upscale (v2.5.9): model/source/output rows, output options (same-folder,
subfolder, model-name, serialization, format/bitdepth/quality), scale/tile/tile-pad/AMP,
Color Fix (+method +⚙), Temporal Fix (+⚙ TF), Undistort (+⚙ UD), persistent subprocess,
skip frames (threshold/blocks), run/stop/progress/timing, log + side-by-side preview.

Heavy lifting stays in src.core.quick_upscale (upscale_image / PersistentBatchSession /
dandere_should_skip) and src.core.post_proc_session (PostProcSession). All inference runs
in a QThread; settings dialogs are QDialogs.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog, QFrame,
    QDialog, QSlider, QMessageBox, QSizePolicy, QSplitter, QRadioButton, QButtonGroup,
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QFont, QPixmap
from qt_ui.i18n import tr

_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
_EXT_MAP = {"PNG": ".png", "JPEG": ".jpg", "JPG": ".jpg", "WEBP": ".webp", "TIFF": ".tif", "BMP": ".bmp"}
# Video-file input — works with ANY model now. Sequence/temporal archs (see
# _SEQUENCE_ARCHS in src.core.quick_upscale: tscunet/temporalspan/temporalspanv2/
# tfdat) go through _video_sequence() (sliding-window temporal inference).
# Any other (non-temporal) model goes through _video_nonsequence(): frames are
# extracted then treated as a classic frame-by-frame folder, each frame
# independent (no sliding window). Both paths reuse
# src.core.lq_generator.extract_frames / resolve_ffmpeg as-is for extraction.
_VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".mts"}


def _mkdtemp_frames(base_dir: str | None, log=None) -> str:
    """mkdtemp() for the extracted-frames folder, honoring the user's chosen base
    dir (Quick Upscale's "Dossier temp." field) instead of always defaulting to
    the system temp folder — a long video can need several GB, and the system
    temp drive (often C:) may not have room. Falls back to the system default
    (with a warning) if the chosen folder doesn't exist and can't be created."""
    if base_dir:
        try:
            os.makedirs(base_dir, exist_ok=True)
            return tempfile.mkdtemp(prefix="usrs_quickups_frames_", dir=base_dir)
        except Exception as e:
            if log:
                log(tr(f"⚠ Dossier temp. « {base_dir} » invalide ({e}), retour au dossier système par défaut.",
                       f"⚠ Temp folder '{base_dir}' invalid ({e}), falling back to the system default."))
    return tempfile.mkdtemp(prefix="usrs_quickups_frames_")


def _detect_video_fps(ffmpeg_exe: str, video_path: str) -> float | None:
    """Best-effort source framerate via `ffmpeg -i <video>` stderr parsing (no
    ffprobe dependency — imageio-ffmpeg only bundles ffmpeg). Returns None if
    it can't be determined; caller must fall back to a sane default."""
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        r = subprocess.run([ffmpeg_exe, "-i", video_path], capture_output=True,
                           text=True, timeout=20, **kwargs)
        out = (r.stderr or "") + (r.stdout or "")
        m = re.search(r"(\d+(?:\.\d+)?)\s*fps", out)
        if m:
            fps = float(m.group(1))
            if fps > 0:
                return fps
    except Exception:
        pass
    return None


# ── Reassembly encoding options ─────────────────────────────────────────────
# Container -> allowed encoder keys. WebM cannot mux H.264/H.265/NVENC streams
# (verified: ffmpeg's webm muxer only accepts vp8/vp9/av1 video), so it gets its
# own single CPU codec (libvpx-vp9) instead of silently producing an unplayable
# file or a container/codec mismatch error at run time.
CONTAINER_EXT = {"MP4": ".mp4", "MKV": ".mkv", "WebM": ".webm"}
CONTAINER_ENCODERS = {
    "MP4": ["libx264", "libx265", "h264_nvenc", "hevc_nvenc", "av1_nvenc"],
    "MKV": ["libx264", "libx265", "h264_nvenc", "hevc_nvenc", "av1_nvenc"],
    # ffmpeg's webm muxer also accepts av1 (not just vp8/vp9) — av1_nvenc included here too.
    "WebM": ["libvpx-vp9", "av1_nvenc"],
}
ENCODER_LABELS = {
    "libx264": tr("libx264 (CPU H.264)", "libx264 (CPU H.264)"),
    "libx265": tr("libx265 (CPU H.265/HEVC)", "libx265 (CPU H.265/HEVC)"),
    "h264_nvenc": tr("h264_nvenc (GPU H.264)", "h264_nvenc (GPU H.264)"),
    "hevc_nvenc": tr("hevc_nvenc (GPU H.265/HEVC)", "hevc_nvenc (GPU H.265/HEVC)"),
    "av1_nvenc": tr("av1_nvenc (GPU AV1)", "av1_nvenc (GPU AV1)"),
    "libvpx-vp9": tr("libvpx-vp9 (CPU VP9)", "libvpx-vp9 (CPU VP9)"),
}
# x264/x265 presets (standard libx264/libx265 -preset values, unchanged names).
CPU_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
# NVENC presets: verified via `ffmpeg -h encoder=h264_nvenc` on this machine (ffmpeg
# build bundles both legacy names and p1..p7 — p1..p7 is the modern/recommended set,
# confirmed accepted at run time in the encode smoke test).
NVENC_PRESETS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
NVENC_PRESET_LABELS = {
    "p1": tr("p1 — le plus rapide", "p1 — fastest"), "p2": "p2", "p3": "p3",
    "p4": tr("p4 — équilibré (défaut)", "p4 — balanced (default)"), "p5": "p5", "p6": "p6",
    "p7": tr("p7 — le plus lent (meilleure qualité)", "p7 — slowest (best quality)"),
}
# libvpx-vp9 uses -deadline instead of a numeric preset.
VP9_PRESETS = ["realtime", "good", "best"]

_NVENC_PROBE_CACHE: dict[str, bool] = {}


def probe_nvenc_encoders(ffmpeg_exe: str) -> dict[str, bool]:
    """Actually test-encode 2 frames with each NVENC codec (not just parse
    `ffmpeg -encoders`, which lists codecs ffmpeg was *compiled* with regardless of
    whether the GPU/driver can run them — verified on the dev machine: av1_nvenc is
    listed but fails at run time on its Pascal GPU with 'No capable devices found').
    This is deliberately probed PER-MACHINE, never hardcoded off: AV1 NVENC needs an
    Ada/Blackwell-class GPU (RTX 40/50 series) — a dev machine lacking it must not
    remove the option for users who do have one. Cached per ffmpeg_exe path for the
    process lifetime (each probe costs ~0.2-0.5s)."""
    cache_key = ffmpeg_exe or ""
    cached = _NVENC_PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = {}
    for codec in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
        ok = False
        if ffmpeg_exe and os.path.isfile(ffmpeg_exe):
            try:
                kwargs = {}
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                # 256x256, not something tiny like 64x64: verified NVENC (hevc_nvenc on this
                # machine) rejects frames below its minimum encode size with "Frame dimensions
                # are less than the minimum supported value" — too small would read as
                # "unavailable" for an encoder that actually works.
                cmd = [ffmpeg_exe, "-y", "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=5:duration=0.2",
                       "-c:v", codec, "-f", "null", "-"]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, **kwargs)
                ok = r.returncode == 0
            except Exception:
                ok = False
        result[codec] = ok
    _NVENC_PROBE_CACHE[cache_key] = result
    return result


def _reassemble_video(ffmpeg_exe: str, frames_folder: str, fps: float, out_video_path: str,
                      encoder: str = "libx264", quality_mode: str = "crf", crf: float = 23.0,
                      bitrate_mbps: float = 8.0, preset: str = "medium",
                      log=None, source_video_for_audio: str | None = None,
                      files: list | None = None) -> tuple[bool, str]:
    """Encode every image in `frames_folder` (natural-sorted) into `out_video_path` at
    `fps`, using `encoder`/`quality_mode`/`crf`/`bitrate_mbps`/`preset`. Defaults
    (libx264/crf/23/medium) reproduce the exact previous hardcoded behavior
    (libx264 + yuv420p) when the caller doesn't override anything. Folder-of-frames
    output remains the primary deliverable regardless of this succeeding or not.

    `source_video_for_audio`: when given (the original video the frames were
    extracted from), muxes its audio track into the reassembled video by stream-copy
    (no re-encode). Previously this function had no audio handling at all — the
    reassembled video was always silent even when the source had sound, with no
    warning anywhere in the UI (2026-09-17 fix). If the source has no audio stream,
    ffmpeg's optional-map (`?`) just drops the -map silently and the video is still
    produced (video-only), matching the old (silent) behavior rather than failing."""
    # `files`: the frames this run produced, in order (A30). Without it, every image of
    # the folder — older results included — went into the video.
    if files is not None:
        files = [os.path.basename(f) for f in files if os.path.isfile(os.path.join(frames_folder, os.path.basename(f)))]
    else:
        files = sorted([f for f in os.listdir(frames_folder)
                        if os.path.splitext(f)[1].lower() in _EXT], key=_natural_sort_key)
    if not files:
        return False, tr("Aucune frame upscalée à réassembler.", "No upscaled frame to reassemble.")
    # ffmpeg needs a single printf-style pattern — the upscaled frames are named
    # "<orig-stem>_upscaled<ext>", which isn't a fixed-width numeric sequence, so
    # build a temporary concat list instead of relying on -i pattern%d.
    list_path = os.path.join(frames_folder, "_concat_list.txt")
    try:
        with open(list_path, "w", encoding="utf-8") as fh:
            for f in files:
                p = os.path.join(frames_folder, f).replace("\\", "/").replace("'", "'\\''")
                fh.write(f"file '{p}'\nduration {1.0 / fps:.6f}\n")
            # ffconcat needs the last frame repeated (duration is ignored on the last entry)
            p = os.path.join(frames_folder, files[-1]).replace("\\", "/").replace("'", "'\\''")
            fh.write(f"file '{p}'\n")

        cmd = [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", list_path]
        _has_audio_src = bool(source_video_for_audio and os.path.isfile(source_video_for_audio))
        if _has_audio_src:
            cmd += ["-i", source_video_for_audio]
        cmd += ["-r", f"{fps}"]
        is_nvenc = encoder.endswith("_nvenc")
        is_vp9 = encoder == "libvpx-vp9"
        cmd += ["-c:v", encoder]
        if is_nvenc:
            cmd += ["-preset", preset]
            if quality_mode == "bitrate":
                cmd += ["-rc", "vbr", "-b:v", f"{bitrate_mbps}M"]
            else:
                cmd += ["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"]
            cmd += ["-pix_fmt", "yuv420p"]
        elif is_vp9:
            cmd += ["-deadline", preset, "-cpu-used", "2"]
            if quality_mode == "bitrate":
                cmd += ["-b:v", f"{bitrate_mbps}M"]
            else:
                cmd += ["-crf", str(int(crf)), "-b:v", "0"]
            cmd += ["-pix_fmt", "yuv420p"]
        else:  # libx264 / libx265 (CPU)
            cmd += ["-preset", preset]
            if quality_mode == "bitrate":
                cmd += ["-b:v", f"{bitrate_mbps}M"]
            else:
                cmd += ["-crf", str(int(crf))]
            cmd += ["-pix_fmt", "yuv420p"]
        if _has_audio_src:
            # video from input 0 (the frame concat), audio from input 1 (source video) if
            # it has one ("?" = optional map, no error if no audio stream). "-shortest"
            # avoids a longer source-audio track outliving the reassembled video.
            cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy", "-shortest"]
        cmd += [out_video_path]

        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, **kwargs)
        if r.returncode != 0:
            tail = "\n".join((r.stderr or "").splitlines()[-8:])
            return False, f"ffmpeg (réassemblage) a échoué (code {r.returncode}).\n{tail}"
        return True, out_video_path
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass


def _natural_sort_key(s: str):
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _b(settings, key, default=False) -> bool:
    v = settings.get(key, default)
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "on")


_ST_FMT = {  # safetensors dtype → (struct char, byte size)
    "U8": ("B", 1), "I8": ("b", 1), "BOOL": ("?", 1),
    "I16": ("h", 2), "U16": ("H", 2), "F16": ("e", 2), "BF16": ("H", 2),
    "I32": ("i", 4), "U32": ("I", 4), "F32": ("f", 4),
    "I64": ("q", 8), "U64": ("Q", 8), "F64": ("d", 8),
}


def _detect_multiscale(model_path):
    """Return sorted list of trained scales for MetaIGConv models (SpanPP/SpanC family),
    or None. Torch-free for .safetensors; venv subprocess for .pth."""
    try:
        if model_path.endswith(".safetensors"):
            import json, struct
            with open(model_path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n).decode("utf-8", "replace"))
            if "MetaIGConv" not in hdr:
                return None
            meta = hdr["MetaIGConv"]; dtype = str(meta.get("dtype", "")).upper()
            ch, esz = _ST_FMT.get(dtype, ("i", 4))
            offs = meta.get("data_offsets", [0, 0]); nbytes = offs[1] - offs[0]
            count = nbytes // esz
            if count <= 0:
                return None
            with open(model_path, "rb") as f2:
                f2.seek(8 + n + offs[0]); raw = f2.read(nbytes)
            vals = struct.unpack("<" + ch * count, raw)
            return sorted({int(v) for v in vals})
        else:
            from src.core import engine_paths as ep
            py = ep.shared_venv_python() or ep.resolve_engine_python(ep.redux_path())
            if not py or not os.path.isfile(py):
                return None
            import subprocess
            script = ("import torch\n"
                      f"ck=torch.load({model_path!r},map_location='cpu',weights_only=False)\n"
                      "sd=None\n"
                      "for k in ('params_ema','params_g','params','model','state_dict'):\n"
                      "    if isinstance(ck,dict) and k in ck: sd=ck[k]; break\n"
                      "if sd is None and isinstance(ck,dict) and any(str(x).endswith('.weight') for x in ck): sd=ck\n"
                      "if sd and 'MetaIGConv' in sd:\n"
                      "    print(','.join(str(int(v.item())) for v in sd['MetaIGConv']))\n")
            r = subprocess.run([py, "-c", script], capture_output=True, text=True, timeout=30,
                               creationflags=0x08000000 if os.name == "nt" else 0)
            if r.returncode == 0 and r.stdout.strip():
                return sorted({int(x) for x in r.stdout.strip().split(",")})
    except Exception:
        return None
    return None


class _MultiScaleDialog(QDialog):
    """Asks which output scale to use for a multi-scale (MetaIGConv) checkpoint."""
    def __init__(self, scales, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Scale multi-échelle", "Multi-scale")); self.setModal(True)
        self.chosen = max(scales)
        v = QVBoxLayout(self)
        v.addWidget(QLabel(tr(f"Modèle multi-échelle détecté\nScales disponibles : "
                           f"{' / '.join(str(s) + '×' for s in scales)}\n\nChoisissez le scale de sortie :",
                           f"Multi-scale model detected\nAvailable scales: "
                           f"{' / '.join(str(s) + '×' for s in scales)}\n\nChoose the output scale:")))
        self._grp = QButtonGroup(self); self._rbs = []
        for s in scales:
            rb = QRadioButton(f"{s}×"); rb.setChecked(s == max(scales))
            self._grp.addButton(rb); v.addWidget(rb); self._rbs.append((s, rb))
        b = QPushButton(tr("Confirmer", "Confirm")); b.setStyleSheet("background:#2ecc71; color:#fff; padding:7px; font-weight:600; border-radius:6px;")
        b.clicked.connect(self._ok); v.addWidget(b)

    def _ok(self):
        for s, rb in self._rbs:
            if rb.isChecked():
                self.chosen = s; break
        self.accept()


# ════════════════════════════════════════════════════════════════════════════════
#  GPU mini-panel (🖥 GPU | Load | VRAM | Temp) — top-right of the header
# ════════════════════════════════════════════════════════════════════════════════
class _GpuMiniPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gpupanel")
        self.setStyleSheet("QFrame#gpupanel{background:#111827; border-radius:8px;}")
        h = QHBoxLayout(self); h.setContentsMargins(10, 6, 10, 6); h.setSpacing(8)
        title = QLabel("🖥 GPU")
        title.setStyleSheet('color:#3B8ED0; font-weight:700; font-family:"Segoe UI Emoji"; font-size:11px; background:transparent;')
        h.addWidget(title)
        self.bars = {}
        for name, key, color, w in (("Load", "load", "#3498db", 36), ("VRAM", "vram", "#9b59b6", 64), ("Temp", "temp", "#2ecc71", 38)):
            lb = QLabel(name); lb.setFont(QFont("Consolas", 9)); lb.setStyleSheet("color:#94a3b8; background:transparent;")
            h.addWidget(lb)
            bar = QProgressBar(); bar.setRange(0, 100); bar.setTextVisible(False); bar.setFixedSize(110, 8)
            self._style_bar(bar, color); h.addWidget(bar)
            val = QLabel("—"); val.setFixedWidth(w); val.setFont(QFont("Consolas", 9)); val.setStyleSheet("color:#CBD5E1; background:transparent;")
            h.addWidget(val)
            self.bars[key] = (bar, val)
        self._init = False
        self._timer = QTimer(self); self._timer.timeout.connect(self._poll); self._timer.start(2000)
        QTimer.singleShot(200, self._poll)

    def _style_bar(self, bar, color):
        bar.setStyleSheet(f"QProgressBar{{background:#1e293b; border:none; border-radius:4px;}} "
                          f"QProgressBar::chunk{{background:{color}; border-radius:4px;}}")

    def _set(self, key, pct, text, color):
        bar, val = self.bars[key]; bar.setValue(int(max(0, min(100, pct)))); self._style_bar(bar, color); val.setText(text)

    def _poll(self):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml
            if not self._init:
                pynvml.nvmlInit(); self._h = pynvml.nvmlDeviceGetHandleByIndex(0); self._init = True
            h = self._h
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, 0)
            self._set("load", util, f"{util}%", "#e74c3c" if util > 80 else "#f39c12" if util > 60 else "#3498db")
            # mem.used/mem.total are ctypes-backed fields (nvidia-ml-py) — isolated in its
            # own try so a bad read (seen as a raw unformatted byte count instead of
            # "X.X/Y.YG" in the packaged .exe, not reproducible calling the same library
            # directly — a ctypes/frozen-binary quirk, not a units bug) degrades to "—"
            # instead of leaking a meaningless number, and never takes Load/Temp down
            # with it (2026-09-28 fix).
            try:
                used_b, total_b = int(mem.used), int(mem.total)
                if total_b <= 0 or used_b < 0 or used_b > total_b:
                    raise ValueError(f"implausible VRAM reading: used={used_b} total={total_b}")
                vf = int(used_b / total_b * 100)
                self._set("vram", vf, f"{used_b / 1024**3:.1f}/{total_b / 1024**3:.1f}G",
                          "#e74c3c" if vf > 90 else "#f39c12" if vf > 70 else "#9b59b6")
            except Exception:
                self._set("vram", 0, "—", "#9b59b6")
            self._set("temp", int(temp / 120 * 100), f"{temp}°C",
                      "#e74c3c" if temp > 85 else "#f39c12" if temp > 70 else "#2ecc71")
        except Exception:
            for key, (_b, val) in self.bars.items():
                val.setText("no GPU" if key == "vram" else "—")

    def stop(self):
        self._timer.stop()
        if self._init:
            try:
                import pynvml; pynvml.nvmlShutdown()
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════════════════
#  Settings dialogs (Color Fix / Temporal Fix / Undistort)
# ════════════════════════════════════════════════════════════════════════════════
class _SliderRow(QHBoxLayout):
    """Slider + live value label (float, n steps)."""
    def __init__(self, lo, hi, steps, init, fmt="{:.2f}"):
        super().__init__()
        self.fmt = fmt; self.lo, self.hi, self.steps = lo, hi, steps
        self.sld = QSlider(Qt.Horizontal); self.sld.setRange(0, steps); self.sld.setFixedWidth(220)
        self.sld.setValue(round((init - lo) / (hi - lo) * steps))
        self.lbl = QLabel(fmt.format(init)); self.lbl.setFixedWidth(40)
        self.sld.valueChanged.connect(lambda _: self.lbl.setText(fmt.format(self.value())))
        self.addWidget(self.sld); self.addWidget(self.lbl)

    def value(self):
        return self.lo + (self.hi - self.lo) * self.sld.value() / self.steps


class ColorFixDialog(QDialog):
    def __init__(self, settings, cf, method, parent=None):
        super().__init__(parent)
        self.settings = settings; self.cf = cf; self.method = method
        self.setWindowTitle(tr("Réglages Color Fix", "Color Fix Settings")); self.setModal(True)
        v = QVBoxLayout(self)
        t = QLabel(f"Color Fix — {method}"); t.setStyleSheet("color:#3B8ED0; font-weight:700; font-size:14px;")
        v.addWidget(t)

        v.addWidget(QLabel(tr("Intensité (0 = désactivé, 1 = correction complète) :", "Strength (0 = off, 1 = full correction):")))
        self.r_str = _SliderRow(0.0, 1.0, 20, float(cf.get("strength", 1.0)), "{:.1f}")
        v.addLayout(self.r_str)
        sep = QFrame(); sep.setFixedHeight(1); sep.setStyleSheet("background:#555;"); v.addWidget(sep)

        self.wav = self.rad = self.fast = None
        if method == "wavelet":
            v.addWidget(QLabel(tr("Niveaux wavelet (1-10) — rayon = 2^N px (recommandé 3-5) :", "Wavelet levels (1-10) — radius = 2^N px (recommended 3-5):")))
            self.r_wav = _SliderRow(1, 10, 9, int(cf.get("wavelets", 4)), "{:.0f}")
            v.addLayout(self.r_wav)
        elif method == "guided":
            info = QLabel(tr(
                "Modèle aniCC (Bendel, PACC V4) : recolore la sortie d'après la source, y compris les "
                "dérives de couleur non uniformes. Garde le détail de l'upscale.\n"
                "Téléchargé automatiquement (~14 Mo) au premier usage. Licence CC BY-NC-SA 4.0.",
                "aniCC model (Bendel, PACC V4): re-colours the output from the source, including "
                "non-uniform colour shifts. Keeps the upscale's detail.\n"
                "Downloaded automatically (~14 MB) on first use. License CC BY-NC-SA 4.0."))
            info.setWordWrap(True); v.addWidget(info)
        elif method != "average":
            # A14: LAB / histogram / colormap have no radius, no fast mode and run on CPU
            info = QLabel(tr("Pas de réglage spécifique pour cette méthode (calcul CPU).",
                             "No specific setting for this method (CPU computation)."))
            info.setWordWrap(True); v.addWidget(info)
        else:
            v.addWidget(QLabel(tr("Radius (5-100) — 32=local, 96=global (recommandé 20-50) :", "Radius (5-100) — 32=local, 96=global (recommended 20-50):")))
            self.r_rad = _SliderRow(5, 100, 19, int(cf.get("radius", 32)), "{:.0f}")
            v.addLayout(self.r_rad)
            self.fast = QCheckBox(tr("Mode rapide (fast) — downscale+upscale, ~10× plus vite", "Fast mode — downscale+upscale, ~10× faster"))
            self.fast.setChecked(bool(cf.get("fast", False))); v.addWidget(self.fast)

        v.addWidget(QLabel(tr("Canaux à corriger :", "Channels to fix:")))
        prow = QHBoxLayout(); planes = cf.get("planes", [0, 1, 2])
        self.ch_r = QCheckBox("R"); self.ch_g = QCheckBox("G"); self.ch_b = QCheckBox("B")
        self.ch_r.setChecked(0 in planes); self.ch_g.setChecked(1 in planes); self.ch_b.setChecked(2 in planes)
        for c in (self.ch_r, self.ch_g, self.ch_b):
            prow.addWidget(c)
        prow.addStretch(1); v.addLayout(prow)

        v.addWidget(QLabel(tr("Accélération :", "Acceleration:")))
        self._dev_ui = [tr("Auto (CUDA si dispo)", "Auto (CUDA if available)"), "CPU", "CUDA", tr("TRT (torch.compile, RTX recommandé)", "TRT (torch.compile, RTX recommended)")]
        self._dev_map = {"auto": 0, "cpu": 1, "cuda": 2, "trt": 3}
        self._dev_key = ["auto", "cpu", "cuda", "trt"]
        self.dev = QComboBox(); self.dev.addItems(self._dev_ui)
        self.dev.setCurrentIndex(self._dev_map.get(cf.get("device", "auto"), 0)); v.addWidget(self.dev)
        if method in ("lab", "hist", "colormap"):
            self.dev.setEnabled(False)
            self.dev.setToolTip(tr("Méthode calculée sur CPU (NumPy/PIL) quel que soit ce choix.",
                                   "This method runs on CPU (NumPy/PIL) whatever this choice."))
        elif method == "guided":
            self.dev.setToolTip(tr("TRT équivaut ici à CUDA (pas de torch.compile pour aniCC).",
                                   "TRT means CUDA here (no torch.compile for aniCC)."))

        v.addWidget(QLabel(tr("Image de référence (optionnel — PNG/TIFF 16-bit) :", "Reference image (optional — PNG/TIFF 16-bit):")))
        rrow = QHBoxLayout()
        self.ref = QLineEdit(cf.get("ref", "")); self.ref.setPlaceholderText(tr("Aucune (utilise LQ par défaut)", "None (uses LQ by default)"))
        rrow.addWidget(self.ref, 1)
        bref = QPushButton("📂"); bref.setFixedWidth(40); bref.clicked.connect(self._browse_ref); rrow.addWidget(bref)
        bclr = QPushButton("✕"); bclr.setFixedWidth(34); bclr.clicked.connect(lambda: self.ref.clear()); rrow.addWidget(bclr)
        v.addLayout(rrow)

        brow = QHBoxLayout()
        ok = QPushButton(tr("Appliquer", "Apply")); ok.setStyleSheet("background:#2ecc71; color:#fff; font-weight:600; padding:7px;")
        ok.clicked.connect(self._apply)
        ca = QPushButton(tr("Annuler", "Cancel")); ca.setStyleSheet("background:#e74c3c; color:#fff; font-weight:600; padding:7px;")
        ca.clicked.connect(self.reject)
        brow.addWidget(ok); brow.addWidget(ca); v.addLayout(brow)

    def _browse_ref(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Image de référence", "Reference image"), "",
                                           tr("Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp *.webp);;Tous (*.*)", "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp *.webp);;All (*.*)"))
        if p:
            self.ref.setText(p)

    def _apply(self):
        planes = [c for c, w in ((0, self.ch_r), (1, self.ch_g), (2, self.ch_b)) if w.isChecked()]
        if not planes:   # A15: no channel ticked used to mean all three
            QMessageBox.warning(self, tr("Canaux", "Channels"),
                                tr("Cochez au moins un canal (R, G ou B).", "Tick at least one channel (R, G or B)."))
            return
        ref = self.ref.text().strip()
        if ref and not os.path.isfile(ref):   # A35: a missing reference was silently ignored
            QMessageBox.warning(self, tr("Référence", "Reference"),
                                tr(f"Image de référence introuvable :\n{ref}", f"Reference image not found:\n{ref}"))
            return
        self.cf["strength"] = round(self.r_str.value(), 2)
        self.settings.set("ups_colorfix_strength", str(self.cf["strength"]))
        if self.method == "wavelet":
            self.cf["wavelets"] = int(round(self.r_wav.value()))
            self.settings.set("ups_colorfix_wavelets", str(self.cf["wavelets"]))
        elif self.method == "average":
            self.cf["radius"] = int(round(self.r_rad.value())); self.cf["fast"] = bool(self.fast.isChecked())
            self.settings.set("ups_colorfix_radius", str(self.cf["radius"]))
            self.settings.set("ups_colorfix_fast", self.cf["fast"])
        self.cf["planes"] = planes; self.settings.set("ups_colorfix_planes", planes)
        dk = self._dev_key[self.dev.currentIndex()]; self.cf["device"] = dk
        self.settings.set("ups_colorfix_device", dk)
        self.cf["ref"] = ref; self.settings.set("ups_colorfix_ref", ref)
        self.accept()


def _parse_mode(m):
    if m == "classic":
        return "classic", "pytorch"
    for pfx, bk in (("model_", "pytorch"), ("ort_", "ort"), ("trt_", "trt"), ("cpu_", "cpu")):
        if m.startswith(pfx):
            return m[len(pfx):], bk
    return "classic", "pytorch"


def _build_mode(variant, backend):
    if variant == "classic":
        return "classic"
    return {"pytorch": "model_", "ort": "ort_", "trt": "trt_", "cpu": "cpu_"}.get(backend, "model_") + variant


class _DLModelRow(QHBoxLayout):
    """Status label + Download button for a model_manager key (TF/UD weights)."""
    def __init__(self, parent_dialog, key_getter):
        super().__init__()
        self.dlg = parent_dialog; self.key_getter = key_getter; self.worker = None
        self.status = QLabel(""); self.status.setStyleSheet("color:#888;")
        self.btn = QPushButton(tr("Télécharger", "Download")); self.btn.setFixedWidth(120); self.btn.clicked.connect(self._dl)
        self.addWidget(self.status, 1); self.addWidget(self.btn)

    def refresh(self):
        key = self.key_getter()
        if not key:
            self.status.setText(tr("Classic — aucun poids nécessaire.", "Classic — no weights needed.")); self.status.setStyleSheet("color:#888;")
            self.btn.setEnabled(False); return
        try:
            from src.core.model_manager import get_manager, MODELS
            info = MODELS[key]
            if get_manager().is_ready(key):
                self.status.setText(f"✓ {info['filename']} ({info['size_mb']:.1f} MB)")
                self.status.setStyleSheet("color:#2ecc71;"); self.btn.setEnabled(False); self.btn.setText(tr("Téléchargé ✓", "Downloaded ✓"))
            else:
                self.status.setText(f"⬇ {info['filename']} ({info['size_mb']:.1f} MB)")
                self.status.setStyleSheet("color:#e67e22;"); self.btn.setEnabled(True); self.btn.setText(tr("Télécharger", "Download"))
        except Exception as e:
            self.status.setText(tr(f"Erreur: {e}", f"Error: {e}")); self.status.setStyleSheet("color:#e74c3c;")

    def _dl(self):
        key = self.key_getter()
        if not key:
            return
        self.btn.setEnabled(False); self.btn.setText(tr("Téléchargement…", "Downloading…"))
        self.status.setText("0 %"); self.status.setStyleSheet("color:#3498db;")

        row = self

        class _DLW(QThread):
            prog = Signal(int, int); done = Signal(bool, str)
            def run(self_):
                try:
                    from src.core.model_manager import get_manager
                    get_manager().download(key, progress_cb=lambda dl, tot: self_.prog.emit(dl, tot))
                    self_.done.emit(True, "")
                except Exception as e:
                    self_.done.emit(False, str(e))

        def _p(dl, tot):
            pct = int(dl * 100 / tot) if tot else 0
            row.status.setText(f"{pct}%  ({dl // 1024}KB / {tot // 1024}KB)")

        def _d(ok, msg):
            if ok:
                row.refresh()
            else:
                row.status.setText(tr(f"Erreur: {msg}", f"Error: {msg}")); row.status.setStyleSheet("color:#e74c3c;")
                row.btn.setEnabled(True); row.btn.setText(tr("Réessayer", "Retry"))
            row.worker = None

        self.worker = _DLW(); self.worker.prog.connect(_p); self.worker.done.connect(_d); self.worker.start()


_VARIANTS_TF = {"classic": tr("Classic (rapide / fallback)", "Classic (fast / fallback)"),
                "s1": tr("s1  (~2 MB, léger)", "s1  (~2 MB, light)"),
                "s2": tr("s2 ★ (~2 MB, recommandé)", "s2 ★ (~2 MB, recommended)"),
                "s3": tr("s3  (~2 MB, fort)", "s3  (~2 MB, strong)")}
_VARIANTS_UD = {"classic": tr("Classic (rapide / fallback)", "Classic (fast / fallback)"), "tmt": "TMT (~8 MB PTH / ~5 MB ONNX)"}
_BACKENDS = {"pytorch": "PyTorch CUDA", "ort": "OnnxRuntime ⚡ (CUDA)",
             "trt": "TRT + OnnxRuntime ⚡⚡", "cpu": "CPU (PyTorch)"}


class TempFixDialog(QDialog):
    def __init__(self, settings, tf, parent=None):
        super().__init__(parent)
        self.settings = settings; self.tf = tf
        self.setWindowTitle(tr("Réglages Temporal Fix", "Temporal Fix Settings")); self.setModal(True)
        v = QVBoxLayout(self)
        t = QLabel("Temporal Fix — Post-processing"); t.setStyleSheet("color:#9B59B6; font-weight:700; font-size:14px;")
        v.addWidget(t)
        v.addWidget(QLabel(tr("Réduction du scintillement SR sur séquences vidéo.\n"
                           "Blend adaptatif : zones statiques lissées, zones mobiles conservées.", "Reduces SR flicker on video sequences.\nAdaptive blend: static areas smoothed, moving areas preserved.")))

        v.addWidget(QLabel(tr("Intensité (0 = désactivé, 1 = maximum — 0.4-0.6 recommandé) :", "Strength (0 = off, 1 = maximum — 0.4-0.6 recommended):")))
        self.r_str = _SliderRow(0.0, 1.0, 20, float(tf.get("strength", 0.5)))
        v.addLayout(self.r_str)

        wrow = QHBoxLayout(); wrow.addWidget(QLabel(tr("Taille fenêtre (frames) :", "Window size (frames):")))
        self.win = QComboBox(); self.win.addItems(["5", "7", "9"]); self.win.setCurrentText(str(tf.get("window", 7)))
        wrow.addWidget(self.win)
        self.lat = QLabel(tr(f"→ latence {int(tf.get('window', 7)) // 2} frames", f"→ latency {int(tf.get('window', 7)) // 2} frames")); self.lat.setStyleSheet("color:#888;")
        self.win.currentTextChanged.connect(lambda x: self.lat.setText(tr(f"→ latence {int(x) // 2} frames", f"→ latency {int(x) // 2} frames")))
        wrow.addWidget(self.lat); wrow.addStretch(1); v.addLayout(wrow)

        prow = QHBoxLayout(); prow.addWidget(QLabel(tr("Précision :", "Precision:")))
        self.prec = QComboBox(); self.prec.addItems(["float32", "float16"]); self.prec.setCurrentText(tf.get("precision", "float32"))
        prow.addWidget(self.prec); prow.addStretch(1); v.addLayout(prow)

        sep = QFrame(); sep.setFixedHeight(1); sep.setStyleSheet("background:#555;"); v.addWidget(sep)
        h = QLabel(tr("Algorithme Temporal Fix :", "Temporal Fix algorithm:")); h.setStyleSheet("font-weight:700;"); v.addWidget(h)
        var0, bk0 = _parse_mode(tf.get("tempfix_mode", "classic"))
        self._vrev = {v: k for k, v in _VARIANTS_TF.items()}; self._brev = {v: k for k, v in _BACKENDS.items()}
        vr = QHBoxLayout(); vr.addWidget(QLabel("Variante:"))
        self.variant = QComboBox(); self.variant.addItems(list(_VARIANTS_TF.values()))
        self.variant.setCurrentText(_VARIANTS_TF.get(var0, _VARIANTS_TF["classic"])); vr.addWidget(self.variant, 1); v.addLayout(vr)
        br = QHBoxLayout(); br.addWidget(QLabel("Backend:"))
        self.backend = QComboBox(); self.backend.addItems(list(_BACKENDS.values()))
        self.backend.setCurrentText(_BACKENDS.get(bk0, _BACKENDS["pytorch"])); br.addWidget(self.backend, 1); v.addLayout(br)

        self.dl = _DLModelRow(self, self._model_key); v.addLayout(self.dl)
        self.variant.currentTextChanged.connect(self._on_sel); self.backend.currentTextChanged.connect(self._on_sel)
        self._on_sel()

        brow = QHBoxLayout()
        ok = QPushButton(tr("Appliquer", "Apply")); ok.setStyleSheet("background:#2ecc71; color:#fff; font-weight:600; padding:7px;")
        ok.clicked.connect(self._apply)
        ca = QPushButton(tr("Annuler", "Cancel")); ca.setStyleSheet("background:#e74c3c; color:#fff; font-weight:600; padding:7px;")
        ca.clicked.connect(self.reject)
        brow.addWidget(ok); brow.addWidget(ca); v.addLayout(brow)

    def _model_key(self):
        var = self._vrev.get(self.variant.currentText(), "classic")
        bk = self._brev.get(self.backend.currentText(), "pytorch")
        if var == "classic":
            return ""
        return f"temporalfix_{var}_onnx" if bk in ("ort", "trt") else f"temporalfix_{var}"

    def _on_sel(self, *_):
        var = self._vrev.get(self.variant.currentText(), "classic")
        bk = self._brev.get(self.backend.currentText(), "pytorch")
        self.backend.setEnabled(var != "classic"); self.dl.refresh()
        # A18: the neural models are trained on 7 frames; A21: the ONNX models are FP16
        neural = var != "classic"
        if neural:
            self.win.setCurrentText("7")
        self.win.setEnabled(not neural)
        self.win.setToolTip(tr("Fixée à 7 par le modèle neuronal.", "Fixed to 7 by the neural model.") if neural else "")
        forced = "float16" if neural and bk in ("ort", "trt") else ("float32" if neural and bk == "cpu" else None)
        self._lock_prec(forced, tr("Modèle ONNX FP16 : précision fixe.", "FP16 ONNX model: fixed precision.")
                        if forced == "float16" else tr("Backend CPU : float32.", "CPU backend: float32."),
                        tr("float16 : GPU CUDA uniquement (sinon float32).", "float16: CUDA GPU only (float32 otherwise)."))

    def _lock_prec(self, forced, why, tip=""):
        """Force the precision combo while an algorithm imposes it, restore the user's
        choice afterwards."""
        if forced:
            if self.prec.isEnabled():
                self._prec_user = self.prec.currentText()
            self.prec.setCurrentText(forced)
        elif not self.prec.isEnabled() and hasattr(self, "_prec_user"):
            self.prec.setCurrentText(self._prec_user)
        self.prec.setEnabled(not forced)
        self.prec.setToolTip(why if forced else tip)

    def _apply(self):
        self.tf["strength"] = round(self.r_str.value(), 2); self.tf["window"] = int(self.win.currentText())
        self.tf["precision"] = self.prec.currentText()
        var = self._vrev.get(self.variant.currentText(), "classic"); bk = self._brev.get(self.backend.currentText(), "pytorch")
        self.tf["tempfix_mode"] = _build_mode(var, bk)
        self.settings.set("ups_tempfix_strength", str(self.tf["strength"]))
        self.settings.set("ups_tempfix_window", str(self.tf["window"]))
        self.settings.set("ups_tempfix_precision", self.tf["precision"])
        self.settings.set("ups_tempfix_mode", self.tf["tempfix_mode"])
        self.accept()


class UndistortDialog(QDialog):
    def __init__(self, settings, tf, parent=None):
        super().__init__(parent)
        self.settings = settings; self.tf = tf
        self.setWindowTitle(tr("Réglages Undistort", "Undistort Settings")); self.setModal(True)
        v = QVBoxLayout(self)
        t = QLabel(tr("Undistort — Correction de jitter HF", "Undistort — HF jitter correction")); t.setStyleSheet("color:#27AE60; font-weight:700; font-size:14px;")
        v.addWidget(t)
        v.addWidget(QLabel(tr("Réduit le 'shimmer' sur les contours (jitter haute fréquence).\n"
                           "Médiane temporelle du composant HF. Complémentaire à Temporal Fix.", "Reduces 'shimmer' on edges (high-frequency jitter).\nTemporal median of the HF component. Complementary to Temporal Fix.")))

        v.addWidget(QLabel(tr("Intensité Undistort (0.3–0.5 recommandé) :", "Undistort strength (0.3–0.5 recommended):")))
        self.r_str = _SliderRow(0.0, 1.0, 20, float(tf.get("undistort_strength", 0.35)))
        v.addLayout(self.r_str)

        wrow = QHBoxLayout(); wrow.addWidget(QLabel(tr("Fenêtre Undistort (frames) :", "Undistort window (frames):")))
        self.win = QComboBox(); self.win.addItems(["3", "5", "7"]); self.win.setCurrentText(str(tf.get("undistort_window", 5)))
        wrow.addWidget(self.win); wrow.addStretch(1); v.addLayout(wrow)

        prow = QHBoxLayout(); prow.addWidget(QLabel(tr("Précision :", "Precision:")))
        self.prec = QComboBox(); self.prec.addItems(["float32", "float16"]); self.prec.setCurrentText(tf.get("undistort_precision", "float32"))
        prow.addWidget(self.prec); prow.addStretch(1); v.addLayout(prow)

        h = QLabel(tr("Algorithme Undistort :", "Undistort algorithm:")); h.setStyleSheet("font-weight:700;"); v.addWidget(h)
        var0, bk0 = _parse_mode(tf.get("undistort_mode", "classic"))
        self._vrev = {v: k for k, v in _VARIANTS_UD.items()}; self._brev = {v: k for k, v in _BACKENDS.items()}
        vr = QHBoxLayout(); vr.addWidget(QLabel("Variante:"))
        self.variant = QComboBox(); self.variant.addItems(list(_VARIANTS_UD.values()))
        self.variant.setCurrentText(_VARIANTS_UD.get(var0, _VARIANTS_UD["classic"])); vr.addWidget(self.variant, 1); v.addLayout(vr)
        br = QHBoxLayout(); br.addWidget(QLabel("Backend:"))
        self.backend = QComboBox(); self.backend.addItems(list(_BACKENDS.values()))
        self.backend.setCurrentText(_BACKENDS.get(bk0, _BACKENDS["pytorch"])); br.addWidget(self.backend, 1); v.addLayout(br)

        self.dl = _DLModelRow(self, self._model_key); v.addLayout(self.dl)
        self.variant.currentTextChanged.connect(self._on_sel); self.backend.currentTextChanged.connect(self._on_sel)
        self._on_sel()

        brow = QHBoxLayout()
        ok = QPushButton(tr("Appliquer", "Apply")); ok.setStyleSheet("background:#2ecc71; color:#fff; font-weight:600; padding:7px;")
        ok.clicked.connect(self._apply)
        ca = QPushButton(tr("Annuler", "Cancel")); ca.setStyleSheet("background:#e74c3c; color:#fff; font-weight:600; padding:7px;")
        ca.clicked.connect(self.reject)
        brow.addWidget(ok); brow.addWidget(ca); v.addLayout(brow)

    def _model_key(self):
        var = self._vrev.get(self.variant.currentText(), "classic")
        bk = self._brev.get(self.backend.currentText(), "pytorch")
        if var == "classic":
            return ""
        return "undistort_tmt_onnx" if bk in ("ort", "trt") else "undistort_tmt"

    def _on_sel(self, *_):
        var = self._vrev.get(self.variant.currentText(), "classic")
        bk = self._brev.get(self.backend.currentText(), "pytorch")
        self.backend.setEnabled(var != "classic"); self.dl.refresh()
        # A18: TMT works on 10-frame batches, the window combo is Classic only;
        # A20/A21: TMT ONNX is FP16, TMT PyTorch runs float32
        neural = var != "classic"
        self.win.setEnabled(not neural)
        self.win.setToolTip(tr("TMT traite des lots de 10 frames : fenêtre Classic uniquement.",
                               "TMT processes 10-frame batches: Classic window only.") if neural else "")
        TempFixDialog._lock_prec(self, ("float16" if bk in ("ort", "trt") else "float32") if neural else None,
                                 tr("Fixée par le modèle TMT.", "Fixed by the TMT model."))

    def _apply(self):
        self.tf["undistort_strength"] = round(self.r_str.value(), 2); self.tf["undistort_window"] = int(self.win.currentText())
        self.tf["undistort_precision"] = self.prec.currentText()
        var = self._vrev.get(self.variant.currentText(), "classic"); bk = self._brev.get(self.backend.currentText(), "pytorch")
        self.tf["undistort_mode"] = _build_mode(var, bk)
        self.settings.set("ups_undistort_strength", str(self.tf["undistort_strength"]))
        self.settings.set("ups_undistort_window", str(self.tf["undistort_window"]))
        self.settings.set("ups_undistort_precision", self.tf["undistort_precision"])
        self.settings.set("ups_undistort_mode", self.tf["undistort_mode"])
        self.accept()


class EncodeOptionsDialog(QDialog):
    """Réassemblage vidéo — conteneur / encodeur / qualité / preset. Backs
    `_reassemble_video()`'s new parameters. Defaults reproduce the exact previous
    hardcoded behavior (MP4 / libx264 / CRF 23 / medium) untouched."""
    def __init__(self, settings, enc, parent=None):
        super().__init__(parent)
        self.settings = settings; self.enc = enc
        self.setWindowTitle(tr("Options d'encodage (réassemblage)", "Encoding options (reassembly)"))
        self.setModal(True)
        v = QVBoxLayout(self)
        t = QLabel(tr("Réassemblage vidéo — options d'encodage", "Video reassembly — encoding options"))
        t.setStyleSheet("color:#3498DB; font-weight:700; font-size:14px;")
        v.addWidget(t)

        try:
            from src.core.lq_generator import resolve_ffmpeg
            self._ffmpeg_exe = resolve_ffmpeg(None)
        except Exception:
            self._ffmpeg_exe = None
        self._nvenc_ok = probe_nvenc_encoders(self._ffmpeg_exe) if self._ffmpeg_exe else {}

        crow = QHBoxLayout(); crow.addWidget(QLabel(tr("Conteneur :", "Container:")))
        self.container = QComboBox(); self.container.addItems(list(CONTAINER_EXT.keys()))
        self.container.setCurrentText(enc.get("container", "MP4"))
        crow.addWidget(self.container); crow.addStretch(1); v.addLayout(crow)

        erow = QHBoxLayout(); erow.addWidget(QLabel(tr("Encodeur :", "Encoder:")))
        self.encoder = QComboBox()
        erow.addWidget(self.encoder, 1); v.addLayout(erow)
        self.enc_note = QLabel(""); self.enc_note.setStyleSheet("color:#888; font-size:11px;")
        self.enc_note.setWordWrap(True); v.addWidget(self.enc_note)

        qrow = QHBoxLayout(); qrow.addWidget(QLabel(tr("Mode qualité :", "Quality mode:")))
        self.mode_group = QButtonGroup(self)
        self.rb_crf = QRadioButton(tr("CRF (recommandé)", "CRF (recommended)"))
        self.rb_bitrate = QRadioButton(tr("Bitrate ciblé", "Target bitrate"))
        self.mode_group.addButton(self.rb_crf); self.mode_group.addButton(self.rb_bitrate)
        (self.rb_bitrate if enc.get("quality_mode") == "bitrate" else self.rb_crf).setChecked(True)
        qrow.addWidget(self.rb_crf); qrow.addWidget(self.rb_bitrate); qrow.addStretch(1); v.addLayout(qrow)

        self.crf_row = QHBoxLayout()
        self.crf_lbl = QLabel(tr("CRF (0=sans perte … 51=pire) :", "CRF (0=lossless … 51=worst):"))
        self.crf_row.addWidget(self.crf_lbl)
        self.crf_sld = _SliderRow(0.0, 51.0, 51, float(enc.get("crf", 23)), fmt="{:.0f}")
        self.crf_row.addLayout(self.crf_sld); v.addLayout(self.crf_row)

        self.br_row = QHBoxLayout()
        self.br_row.addWidget(QLabel(tr("Bitrate (Mbps) :", "Bitrate (Mbps):")))
        self.bitrate = QLineEdit(str(enc.get("bitrate_mbps", 8)))
        self.bitrate.setFixedWidth(60)
        self.br_row.addWidget(self.bitrate); self.br_row.addStretch(1); v.addLayout(self.br_row)

        prow = QHBoxLayout(); prow.addWidget(QLabel(tr("Preset :", "Preset:")))
        self.preset = QComboBox(); prow.addWidget(self.preset, 1); v.addLayout(prow)

        self.rb_crf.toggled.connect(self._sync_quality_mode)
        self.container.currentTextChanged.connect(self._sync_encoders)
        self.encoder.currentTextChanged.connect(self._sync_preset)
        self._sync_encoders(select=enc.get("encoder", "libx264"))
        self._sync_quality_mode()

        brow = QHBoxLayout()
        ok = QPushButton(tr("Appliquer", "Apply")); ok.setStyleSheet("background:#2ecc71; color:#fff; font-weight:600; padding:7px;")
        ok.clicked.connect(self._apply)
        ca = QPushButton(tr("Annuler", "Cancel")); ca.setStyleSheet("background:#e74c3c; color:#fff; font-weight:600; padding:7px;")
        ca.clicked.connect(self.reject)
        brow.addWidget(ok); brow.addWidget(ca); v.addLayout(brow)

    def _current_container(self):
        return self.container.currentText()

    def _sync_encoders(self, *_args, select=None):
        cont = self._current_container()
        keys = CONTAINER_ENCODERS.get(cont, ["libx264"])
        self.encoder.blockSignals(True)
        self.encoder.clear()
        notes = []
        for k in keys:
            is_nvenc = k.endswith("_nvenc")
            available = (not is_nvenc) or self._nvenc_ok.get(k, False)
            label = ENCODER_LABELS.get(k, k)
            if is_nvenc and not available:
                label += tr(" — indisponible sur ce GPU/pilote", " — unavailable on this GPU/driver")
                notes.append(k)
            self.encoder.addItem(label, k)
            if is_nvenc and not available:
                idx = self.encoder.count() - 1
                item_model = self.encoder.model()
                item_model.item(idx).setEnabled(False)
        target = select or self.enc.get("encoder")
        target_idx = self.encoder.findData(target) if target else -1
        if target_idx < 0 or not self.encoder.model().item(target_idx).isEnabled():
            # first enabled item
            target_idx = 0
            for i in range(self.encoder.count()):
                if self.encoder.model().item(i).isEnabled():
                    target_idx = i
                    break
        self.encoder.setCurrentIndex(target_idx)
        self.encoder.blockSignals(False)
        if notes:
            self.enc_note.setText(tr(
                "NVENC testé réellement (pas seulement listé par ffmpeg -encoders) : "
                "indisponible = a échoué au test d'encodage sur cette machine.",
                "NVENC actually test-encoded (not just listed by ffmpeg -encoders): "
                "unavailable = failed the encode test on this machine."))
        else:
            self.enc_note.setText("")
        self._sync_preset()

    def _sync_quality_label(self):
        # A33: NVENC runs -cq (constant quality, 0 = automatic, not lossless); VP9 CRF
        # tops at 63. Same slider, honest label per encoder.
        k = self.encoder.currentData() or ""
        if k.endswith("_nvenc"):
            self.rb_crf.setText(tr("CQ (qualité constante)", "CQ (constant quality)"))
            self.crf_lbl.setText(tr("CQ NVENC (1=meilleur … 51=pire, 0=auto) :", "NVENC CQ (1=best … 51=worst, 0=auto):"))
        elif k == "libvpx-vp9":
            self.rb_crf.setText(tr("CRF (recommandé)", "CRF (recommended)"))
            self.crf_lbl.setText(tr("CRF VP9 (0=meilleur … 51 ici, max 63) :", "VP9 CRF (0=best … 51 here, max 63):"))
        else:
            self.rb_crf.setText(tr("CRF (recommandé)", "CRF (recommended)"))
            if k == "libx265":   # x265 needs -x265-params lossless=1; CRF 0 is not lossless
                self.crf_lbl.setText(tr("CRF (0=meilleur … 51=pire) :", "CRF (0=best … 51=worst):"))
            else:
                self.crf_lbl.setText(tr("CRF (0=sans perte … 51=pire) :", "CRF (0=lossless … 51=worst):"))

    def _sync_preset(self, *_args):
        self._sync_quality_label()
        key = self.encoder.currentData() or "libx264"
        self.preset.blockSignals(True)
        self.preset.clear()
        if key.endswith("_nvenc"):
            for p in NVENC_PRESETS:
                self.preset.addItem(NVENC_PRESET_LABELS.get(p, p), p)
            default = "p4"
        elif key == "libvpx-vp9":
            for p in VP9_PRESETS:
                self.preset.addItem(p, p)
            default = "good"
        else:
            for p in CPU_PRESETS:
                self.preset.addItem(p, p)
            default = "medium"
        saved = self.enc.get("preset")
        idx = self.preset.findData(saved) if saved else -1
        self.preset.setCurrentIndex(idx if idx >= 0 else max(0, self.preset.findData(default)))
        self.preset.blockSignals(False)

    def _sync_quality_mode(self, *_a):
        crf_on = self.rb_crf.isChecked()
        for i in range(self.crf_row.count()):
            w = self.crf_row.itemAt(i)
            if w.widget():
                w.widget().setVisible(crf_on)
            elif w.layout():
                for j in range(w.layout().count()):
                    w2 = w.layout().itemAt(j).widget()
                    if w2:
                        w2.setVisible(crf_on)
        for i in range(self.br_row.count()):
            w = self.br_row.itemAt(i)
            if w.widget():
                w.widget().setVisible(not crf_on)

    def _apply(self):
        self.enc["container"] = self.container.currentText()
        self.enc["encoder"] = self.encoder.currentData() or "libx264"
        self.enc["quality_mode"] = "bitrate" if self.rb_bitrate.isChecked() else "crf"
        self.enc["crf"] = round(self.crf_sld.value())
        try:
            self.enc["bitrate_mbps"] = max(0.1, float(self.bitrate.text().strip() or "8"))
        except ValueError:
            self.enc["bitrate_mbps"] = 8.0
        self.enc["preset"] = self.preset.currentData() or "medium"
        self.settings.set("ups_enc_container", self.enc["container"])
        self.settings.set("ups_enc_encoder", self.enc["encoder"])
        self.settings.set("ups_enc_quality_mode", self.enc["quality_mode"])
        self.settings.set("ups_enc_crf", str(self.enc["crf"]))
        self.settings.set("ups_enc_bitrate_mbps", str(self.enc["bitrate_mbps"]))
        self.settings.set("ups_enc_preset", self.enc["preset"])
        self.accept()


# ════════════════════════════════════════════════════════════════════════════════
#  Worker — faithful port of run_upscale (single image + folder/persistent/skip/TF/UD)
# ════════════════════════════════════════════════════════════════════════════════
def _read01(path):
    """Image file as float32 RGB in [0, 1], 16-bit files included."""
    from src.core.image_io import read_float01
    return read_float01(path)


class _UpscaleWorker(QThread):
    line = Signal(str)
    progress = Signal(float)          # 0..1
    timing = Signal(str)
    preview = Signal(str, str)        # in_path, out_path
    done = Signal(bool, str)          # ok, summary

    def __init__(self, o, stop_event):
        super().__init__(); self.o = o; self.stop = stop_event

    def run(self):
        o = self.o
        cb = lambda m: self.line.emit(str(m))
        try:
            from src.core.quick_upscale import upscale_image
        except Exception as e:
            self.done.emit(False, tr(f"Module quick_upscale indisponible : {e}", f"quick_upscale module unavailable: {e}")); return
        try:
            os.makedirs(o["actual_out"], exist_ok=True)
            if o.get("video_input"):
                try:
                    from src.core.quick_upscale import _SEQUENCE_ARCHS
                except Exception:
                    _SEQUENCE_ARCHS = set()
                if o.get("video_arch") in _SEQUENCE_ARCHS:
                    self._video_sequence(cb)
                else:
                    self._video_nonsequence(upscale_image, cb)
            elif os.path.isfile(o["input"]):
                self._single(upscale_image, cb)
            elif os.path.isdir(o["input"]):
                try:
                    from src.core.quick_upscale import _SEQUENCE_ARCHS
                except Exception:
                    _SEQUENCE_ARCHS = set()
                if o.get("video_arch") in _SEQUENCE_ARCHS:
                    self._folder_sequence(cb)   # A01: frames folder + temporal model
                else:
                    self._folder(upscale_image, cb)
            else:
                self.done.emit(False, tr(f"Chemin introuvable : {o['input']}", f"Path not found: {o['input']}"))
        except Exception as e:
            self.done.emit(False, str(e))

    def _ups_kwargs(self, cb, pcb):
        o = self.o
        return dict(scale=o["scale"], tile_size=o["tile"], tile_pad=o["tile_pad"], use_amp=o["amp"],
                    callback=cb, progress_callback=pcb, out_format=o["out_format"],
                    bit_depth=o["bit_depth"], quality=o["quality"], stop_event=self.stop,
                    color_fix=o["color_fix"], color_fix_wavelets=o["cf_wavelets"],
                    color_fix_radius=o["cf_radius"], color_fix_fast=o["cf_fast"],
                    color_fix_strength=o["cf_strength"], color_fix_planes=o["cf_planes"],
                    color_fix_device=o["cf_device"], color_fix_ref=o["cf_ref"])

    def _single(self, upscale_image, cb):
        o = self.o; self.timing.emit("")

        def _fmt(s):
            s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
            return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s" if m else f"{sec}s"

        base = os.path.splitext(os.path.basename(o["input"]))[0]
        out_path = os.path.join(o["actual_out"], f"{base}{o['out_suffix']}{o['out_ext']}")
        if os.path.normcase(os.path.abspath(out_path)) == os.path.normcase(os.path.abspath(o["input"])):
            msg = tr("Sortie refusée : le fichier de sortie écraserait l'image source.",
                     "Output refused: the output file would overwrite the source image.")
            cb("❌ " + msg); self.done.emit(False, msg); return
        t0 = time.monotonic()

        # "Subprocess persistant" was silently ignored here — _folder() honors it
        # (spawns a real GPU-venv subprocess via PersistentBatchSession) but this
        # single-image path went straight to in-process upscale_image(), which
        # in a dev run uses whatever torch the launching interpreter has (often
        # CPU-only) instead of the engine venv's CUDA torch. Same fallback
        # pattern as _folder() now applied here for consistency.
        _session = None
        if o.get("persistent") and not o["model"].lower().endswith(".onnx"):
            try:
                from src.core.quick_upscale import make_persistent_session
                _session = make_persistent_session(o["model"], tile_size=o["tile"], tile_pad=o["tile_pad"],
                                                   use_amp=o["amp"], log=cb, scale_hint=o["scale"])
                if _session is not None:
                    # A10/A11: the worker writes the chosen format, bit depth and quality
                    _session.save_opts = {"format": o["out_format"], "bit_depth": o["bit_depth"],
                                          "quality": o["quality"]}
                if _session is not None:
                    if not _session.start():
                        cb(tr("[Persistent] Démarrage échoué — mode normal utilisé", "[Persistent] Startup failed — normal mode used"))
                        _session = None
            except Exception as e:
                cb(tr(f"[Persistent] Erreur ({e}) — mode normal", f"[Persistent] Error ({e}) — normal mode"))
                _session = None

        if _session is not None and getattr(_session, "is_ready", False):
            ok, msg, _extra = _session.infer(o["input"], out_path)
            if _extra.get("note"):  # worker auto-tile after a CUDA OOM
                cb(_extra["note"])
            # Subprocess persistant ne fait pas le Color Fix -> appliqué ici (numpy/PIL, pas besoin de torch).
            if ok and o.get("color_fix", "none") != "none" and o.get("cf_strength", 0.0) > 0.0:
                try:
                    from src.core.quick_upscale import color_fix_image as _cfi, load_color_fix_ref as _cfref
                    import numpy as _np
                    from PIL import Image as _CImg
                    _sr = _read01(out_path)
                    _lq = _CImg.open(o["input"]).convert("RGB")
                    _pl = o["cf_planes"] if o.get("cf_planes") is not None else [0, 1, 2]
                    _sr = _cfi(_sr, _lq, method=o["color_fix"], wavelets=o["cf_wavelets"],
                               radius=o["cf_radius"], fast=o["cf_fast"], strength=o["cf_strength"],
                               planes=_pl, device=o["cf_device"],
                               ref_pil=_cfref(o.get("cf_ref"), cb))
                    from src.core.quick_upscale import _save_output as _so
                    _so(_sr.clip(0, 1), out_path, o["out_format"], o["bit_depth"], o["quality"])
                except Exception as _e:
                    cb(tr(f"  [ColorFix] Erreur : {_e}", f"  [ColorFix] Error: {_e}"))
            self.progress.emit(1.0)
            _session.stop()
        else:
            ok, msg = upscale_image(o["model"], o["input"], out_path,
                                    **self._ups_kwargs(cb, lambda v: self.progress.emit(float(v))))
        el = time.monotonic() - t0
        if ok:
            self.progress.emit(1.0); cb(f"-> {out_path}"); self.preview.emit(o["input"], out_path)
            self.timing.emit(tr(f"Terminé — {_fmt(el)}", f"Finished — {_fmt(el)}"))
            self.done.emit(True, os.path.basename(out_path))
        else:
            cb(tr(f"ERREUR : {msg}", f"ERROR: {msg}")); self.done.emit(False, msg)

    def _drop_tmp(self, tmp_dir, cb):
        if self.o.get("keep_frames"):
            cb(tr(f"Frames extraites conservées : {tmp_dir}", f"Extracted frames kept: {tmp_dir}"))
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _folder_sequence(self, cb):
        """Folder of numbered frames + temporal model (tscunet/temporalspan/
        temporalspanv2/tfdat…): the frames go straight to the sequence pipeline,
        like an extracted video. Before, each frame went through upscale_image(),
        which refuses these archs."""
        o = self.o
        try:
            from src.core.quick_upscale import _sequence_video_upscale_folder
        except Exception as e:
            self.done.emit(False, tr(f"Module indisponible : {e}", f"Module unavailable: {e}")); return
        if os.path.normcase(os.path.abspath(o["actual_out"])) == os.path.normcase(os.path.abspath(o["input"])):
            msg = tr("Sortie refusée : avec un modèle temporel, le dossier de sortie doit être différent "
                     "du dossier des frames.",
                     "Output refused: with a temporal model the output folder must differ from the frames folder.")
            cb("❌ " + msg); self.done.emit(False, msg); return
        n = len([f for f in os.listdir(o["input"]) if os.path.splitext(f)[1].lower() in _EXT])
        if n < 2:
            msg = tr(f"Dossier : {n} frame(s) — {o['video_arch']} nécessite une séquence.",
                     f"Folder: {n} frame(s) — {o['video_arch']} needs a sequence.")
            cb("❌ " + msg); self.done.emit(False, msg); return
        cb(tr(f"🎞 {n} frames → pipeline séquence ({o['video_arch']})",
              f"🎞 {n} frames -> sequence pipeline ({o['video_arch']})"))
        t0 = time.monotonic()

        def _seq_progress(msg: str) -> None:
            m = re.match(r"^\[(\d+)/(\d+)\]", str(msg))
            if m:
                self.progress.emit(int(m.group(1)) / max(1, int(m.group(2))))

        try:
            success, total, errors = _sequence_video_upscale_folder(
                model_path=o["model"], input_folder=o["input"], output_folder=o["actual_out"],
                detected_arch=o["video_arch"], scale=o["scale"], log=cb,
                progress_callback=_seq_progress, stop_event=self.stop,
                color_fix=o["color_fix"], color_fix_wavelets=o["cf_wavelets"],
                color_fix_radius=o["cf_radius"], color_fix_fast=o["cf_fast"],
                color_fix_strength=o["cf_strength"], color_fix_device=o["cf_device"],
                out_bit_depth=o["bit_depth"] if o["bit_depth"] in (8, 16) else 8,
            )
        except Exception as e:
            self.done.emit(False, tr(f"Upscale séquence échoué : {e}", f"Sequence upscale failed: {e}")); return
        for err in errors:
            cb(tr(f"  ERREUR : {err}", f"  ERROR: {err}"))
        el = int(time.monotonic() - t0)
        self.timing.emit(tr(f"Terminé — {el}s total.", f"Finished — {el}s total."))
        self.progress.emit(1.0)
        cb(tr(f"Terminé : {success}/{total} frames upscalées -> {o['actual_out']}",
              f"Finished: {success}/{total} frames upscaled -> {o['actual_out']}"))
        stopped = tr(" — arrêté", " — stopped") if self.stop.is_set() and success < total else ""
        self.done.emit(success == total and total > 0, f"{success}/{total} OK{stopped}")

    def _video_sequence(self, cb):
        """Entrée vidéo pour les 4 archs séquence (tscunet/temporalspan/temporalspanv2/
        tfdat) : extrait les frames (lq_generator.extract_frames, réutilisé tel quel)
        dans un dossier temporaire, lance _sequence_video_upscale_folder() dessus
        (pipeline d'inférence séquence existant, inchangé), puis nettoie le dossier
        temporaire (sauf si « Garder les frames extraites » est coché)."""
        o = self.o

        def _fmt(s):
            s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
            return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s" if m else f"{sec}s"

        try:
            from src.core.lq_generator import resolve_ffmpeg, extract_frames
            from src.core.quick_upscale import _sequence_video_upscale_folder, _peek_arch_from_model_path
        except Exception as e:
            self.done.emit(False, tr(f"Module indisponible : {e}", f"Module unavailable: {e}")); return

        video_path = o["input"]
        ffmpeg_exe = resolve_ffmpeg(None)
        if not ffmpeg_exe:
            self.done.emit(False, tr(
                "ffmpeg introuvable — installez ffmpeg, le paquet Python 'imageio-ffmpeg', ou "
                "placez un binaire portable dans bin/ffmpeg.exe.",
                "ffmpeg not found — install ffmpeg, the 'imageio-ffmpeg' Python package, or "
                "place a portable binary in bin/ffmpeg.exe.")); return

        detected_arch = o.get("video_arch") or _peek_arch_from_model_path(o["model"])

        tmp_dir = _mkdtemp_frames(o.get("temp_dir"), log=cb)
        cb(tr(f"🎬 Extraction des frames → {tmp_dir}", f"🎬 Extracting frames -> {tmp_dir}"))
        prefix = os.path.splitext(os.path.basename(video_path))[0]
        batch_start = time.monotonic()
        try:
            def _extract_progress(n):
                if n % 30 == 0:
                    cb(tr(f"  … {n} frames extraites", f"  ... {n} frames extracted"))

            frame_paths = extract_frames(video_path, tmp_dir, prefix, fps=0,
                                         ffmpeg_path=ffmpeg_exe, progress=_extract_progress, log=cb,
                                         img_format=o.get("img_format", "png8"),
                                         jpg_quality=o.get("jpg_quality", 95),
                                         cancel_check=self.stop.is_set)
        except Exception as e:
            self._drop_tmp(tmp_dir, cb)
            self.done.emit(False, tr(f"Extraction ffmpeg échouée : {e}", f"ffmpeg extraction failed: {e}")); return

        if self.stop.is_set():   # A29: Stop during extraction ends the run here
            self._drop_tmp(tmp_dir, cb)
            self.done.emit(False, tr("Arrêté pendant l'extraction.", "Stopped during extraction.")); return
        n_frames = len(frame_paths)
        if n_frames < 2:
            self._drop_tmp(tmp_dir, cb)
            self.done.emit(False, tr(f"Extraction : seulement {n_frames} frame(s) — {detected_arch} "
                                     f"nécessite une séquence.",
                                     f"Extraction: only {n_frames} frame(s) — {detected_arch} "
                                     f"needs a sequence.")); return
        cb(tr(f"✔ {n_frames} frames extraites.", f"✔ {n_frames} frames extracted."))

        def _seq_progress(msg: str) -> None:
            m = re.match(r"^\[(\d+)/(\d+)\]", str(msg))
            if m:
                self.progress.emit(int(m.group(1)) / max(1, int(m.group(2))))

        try:
            success, total, errors = _sequence_video_upscale_folder(
                model_path=o["model"], input_folder=tmp_dir, output_folder=o["actual_out"],
                detected_arch=detected_arch, scale=o["scale"], log=cb,
                progress_callback=_seq_progress, stop_event=self.stop,
                color_fix=o["color_fix"], color_fix_wavelets=o["cf_wavelets"],
                color_fix_radius=o["cf_radius"], color_fix_fast=o["cf_fast"],
                color_fix_strength=o["cf_strength"], color_fix_device=o["cf_device"],
                out_bit_depth=o.get("vid_out_bit_depth", 8),
            )
        except Exception as e:
            self._drop_tmp(tmp_dir, cb)
            self.done.emit(False, tr(f"Upscale séquence échoué : {e}", f"Sequence upscale failed: {e}")); return

        for err in errors:
            cb(tr(f"  ERREUR : {err}", f"  ERROR: {err}"))

        reassembled_msg = ""
        if success > 0 and o.get("reassemble_video") and self.stop.is_set():
            cb(tr("Réassemblage vidéo ignoré : traitement arrêté.", "Video reassembly skipped: processing stopped."))
        elif success > 0 and o.get("reassemble_video"):
            fps = _detect_video_fps(ffmpeg_exe, video_path) or 25.0
            cb(tr(f"🎞 Réassemblage vidéo ({fps:.3f} fps détecté)…",
                 f"🎞 Reassembling video ({fps:.3f} fps detected)..."))
            enc = o.get("enc") or {}
            container = enc.get("container", "MP4")
            ext = CONTAINER_EXT.get(container, ".mp4")
            out_video = os.path.join(o["actual_out"], f"{prefix}_upscaled{ext}")
            ok_r, msg_r = _reassemble_video(
                ffmpeg_exe, o["actual_out"], fps, out_video,
                encoder=enc.get("encoder", "libx264"),
                quality_mode=enc.get("quality_mode", "crf"),
                crf=enc.get("crf", 23), bitrate_mbps=enc.get("bitrate_mbps", 8.0),
                preset=enc.get("preset", "medium"), log=cb,
                source_video_for_audio=video_path,
                files=[f"{os.path.splitext(os.path.basename(fp))[0]}_upscaled.png" for fp in frame_paths])
            if ok_r:
                cb(tr(f"✔ Vidéo réassemblée -> {out_video}", f"✔ Video reassembled -> {out_video}"))
                reassembled_msg = f" + {os.path.basename(out_video)}"
            else:
                cb(tr(f"⚠ Réassemblage vidéo échoué : {msg_r}", f"⚠ Video reassembly failed: {msg_r}"))

        self._drop_tmp(tmp_dir, cb)

        el = time.monotonic() - batch_start
        self.timing.emit(tr(f"Terminé — {_fmt(el)} total.", f"Finished — {_fmt(el)} total."))
        self.progress.emit(1.0)
        cb(tr(f"Terminé : {success}/{total} frames upscalées -> {o['actual_out']}",
             f"Finished: {success}/{total} frames upscaled -> {o['actual_out']}"))
        reassembly_failed = bool(o.get("reassemble_video")) and success > 0 and not reassembled_msg and not self.stop.is_set()
        stopped = tr(" — arrêté", " — stopped") if self.stop.is_set() and success < total else ""
        self.done.emit(success == total and total > 0 and not reassembly_failed,
                       f"{success}/{total} OK{reassembled_msg}{stopped}"
                       + (tr(" — réassemblage échoué", " — reassembly failed") if reassembly_failed else ""))

    def _video_nonsequence(self, upscale_image, cb):
        """Entrée vidéo pour un modèle NON-séquence (n'importe laquelle des ~70 archs
        classiques hors tscunet/temporalspan/temporalspanv2/tfdat) : extrait les frames
        (mêmes mécanismes que _video_sequence — lq_generator.extract_frames/resolve_ffmpeg,
        mêmes options Format extraction PNG8/PNG16/JPEG) dans un dossier temporaire, puis les
        traite comme un dossier classique via self._folder() — chaque frame indépendamment,
        PAS de fenêtre glissante (le modèle n'est pas temporel) — avec toutes les options
        normales déjà collectées (tuilage/color-fix/persistent/dandere/tempfix/undistort/
        format de sortie/bit-depth/qualité). Réassemble en vidéo si demandé (réutilise
        _reassemble_video/EncodeOptionsDialog, même mécanisme que le mode séquence). Nettoie
        le dossier temporaire sauf si « Garder les frames extraites » est coché.

        Note 16-bit : le format de sortie/bit-depth utilisés ici sont les options générales
        (self.bitdepth -> o["bit_depth"]), pas le combo « Sortie séquence 8/16-bit » (celui-ci
        reste spécifique au pipeline séquence). _save_output() — utilisé par
        upscale_image()/self._folder() pour TOUT chemin non-séquence, vidéo ou pas — écrit
        maintenant un vrai PNG/TIFF 16-bit via pyvips (fix 2026-09-16, vérifié binaire :
        IHDR bitdepth=16 réel). Ce chemin en hérite comme le mode dossier classique."""
        o = self.o

        def _fmt(s):
            s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
            return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s" if m else f"{sec}s"

        try:
            from src.core.lq_generator import resolve_ffmpeg, extract_frames
        except Exception as e:
            self.done.emit(False, tr(f"Module indisponible : {e}", f"Module unavailable: {e}")); return

        video_path = o["input"]
        ffmpeg_exe = resolve_ffmpeg(None)
        if not ffmpeg_exe:
            self.done.emit(False, tr(
                "ffmpeg introuvable — installez ffmpeg, le paquet Python 'imageio-ffmpeg', ou "
                "placez un binaire portable dans bin/ffmpeg.exe.",
                "ffmpeg not found — install ffmpeg, the 'imageio-ffmpeg' Python package, or "
                "place a portable binary in bin/ffmpeg.exe.")); return

        tmp_dir = _mkdtemp_frames(o.get("temp_dir"), log=cb)
        cb(tr(f"🎬 Extraction des frames → {tmp_dir}", f"🎬 Extracting frames -> {tmp_dir}"))
        prefix = os.path.splitext(os.path.basename(video_path))[0]
        batch_start = time.monotonic()
        try:
            def _extract_progress(n):
                if n % 30 == 0:
                    cb(tr(f"  … {n} frames extraites", f"  ... {n} frames extracted"))

            frame_paths = extract_frames(video_path, tmp_dir, prefix, fps=0,
                                         ffmpeg_path=ffmpeg_exe, progress=_extract_progress, log=cb,
                                         img_format=o.get("img_format", "png8"),
                                         jpg_quality=o.get("jpg_quality", 95),
                                         cancel_check=self.stop.is_set)
        except Exception as e:
            self._drop_tmp(tmp_dir, cb)
            self.done.emit(False, tr(f"Extraction ffmpeg échouée : {e}", f"ffmpeg extraction failed: {e}")); return

        if self.stop.is_set():   # A29: Stop during extraction ends the run here
            self._drop_tmp(tmp_dir, cb)
            self.done.emit(False, tr("Arrêté pendant l'extraction.", "Stopped during extraction.")); return
        n_frames = len(frame_paths)
        if n_frames < 1:
            self._drop_tmp(tmp_dir, cb)
            self.done.emit(False, tr("Extraction : aucune frame extraite.",
                                     "Extraction: no frame extracted.")); return
        cb(tr(f"✔ {n_frames} frames extraites.", f"✔ {n_frames} frames extracted."))
        cb(tr("Modèle non-temporel — chaque frame traitée indépendamment (pas de fenêtre glissante).",
             "Non-temporal model — each frame processed independently (no sliding window)."))

        try:
            success, total, errors = self._folder(upscale_image, cb, input_folder=tmp_dir, emit_done=False)
        except Exception as e:
            self._drop_tmp(tmp_dir, cb)
            self.done.emit(False, tr(f"Upscale échoué : {e}", f"Upscale failed: {e}")); return

        reassembled_msg = ""
        if success > 0 and o.get("reassemble_video") and self.stop.is_set():
            cb(tr("Réassemblage vidéo ignoré : traitement arrêté.", "Video reassembly skipped: processing stopped."))
        elif success > 0 and o.get("reassemble_video"):
            fps = _detect_video_fps(ffmpeg_exe, video_path) or 25.0
            cb(tr(f"🎞 Réassemblage vidéo ({fps:.3f} fps détecté)…",
                 f"🎞 Reassembling video ({fps:.3f} fps detected)..."))
            enc = o.get("enc") or {}
            container = enc.get("container", "MP4")
            ext = CONTAINER_EXT.get(container, ".mp4")
            out_video = os.path.join(o["actual_out"], f"{prefix}_upscaled{ext}")
            ok_r, msg_r = _reassemble_video(
                ffmpeg_exe, o["actual_out"], fps, out_video,
                encoder=enc.get("encoder", "libx264"),
                quality_mode=enc.get("quality_mode", "crf"),
                crf=enc.get("crf", 23), bitrate_mbps=enc.get("bitrate_mbps", 8.0),
                preset=enc.get("preset", "medium"), log=cb,
                source_video_for_audio=video_path, files=list(getattr(self, "_produced", [])))
            if ok_r:
                cb(tr(f"✔ Vidéo réassemblée -> {out_video}", f"✔ Video reassembled -> {out_video}"))
                reassembled_msg = f" + {os.path.basename(out_video)}"
            else:
                cb(tr(f"⚠ Réassemblage vidéo échoué : {msg_r}", f"⚠ Video reassembly failed: {msg_r}"))

        self._drop_tmp(tmp_dir, cb)

        el = time.monotonic() - batch_start
        self.timing.emit(tr(f"Terminé — {_fmt(el)} total.", f"Finished — {_fmt(el)} total."))
        self.progress.emit(1.0)
        cb(tr(f"Terminé : {success}/{total} frames upscalées -> {o['actual_out']}",
             f"Finished: {success}/{total} frames upscaled -> {o['actual_out']}"))
        reassembly_failed = bool(o.get("reassemble_video")) and success > 0 and not reassembled_msg and not self.stop.is_set()
        stopped = tr(" — arrêté", " — stopped") if self.stop.is_set() and success < total else ""
        self.done.emit(success == total and total > 0 and not reassembly_failed,
                       f"{success}/{total} OK{reassembled_msg}{stopped}"
                       + (tr(" — réassemblage échoué", " — reassembly failed") if reassembly_failed else ""))

    def _folder(self, upscale_image, cb, input_folder=None, emit_done=True):
        o = self.o
        src_folder = input_folder if input_folder is not None else o["input"]
        files = sorted([fn for fn in os.listdir(src_folder) if os.path.splitext(fn)[1].lower() in _EXT],
                       key=_natural_sort_key)
        total = len(files)
        if total == 0:
            cb(tr("Aucune image trouvée dans le dossier.", "No image found in the folder."))
            if emit_done:
                self.done.emit(False, tr("Aucune image trouvée", "No image found"))
            return 0, 0, [tr("Aucune image trouvée", "No image found")]
        pad_w = max(5, len(str(o["serialize_start"] + total - 1))) if o["serialize"] else 5
        # Destinations decided up front: never a source file (same folder + serialization
        # could write 00000.png over a source), and two sources with the same stem
        # (a.jpg + a.png) no longer write the same output.
        _dest, _seen = {}, {}
        for i, fname in enumerate(files):
            base = os.path.splitext(fname)[0]
            if o["serialize"]:
                name = f"{o['serialize_start'] + i:0{pad_w}d}{o['out_ext']}"
            else:
                name = f"{base}{o['out_suffix']}{o['out_ext']}"
            key = name.lower()
            if key in _seen and not o["serialize"]:
                name = f"{base}_{os.path.splitext(fname)[1][1:]}{o['out_suffix']}{o['out_ext']}"
                key = name.lower()
            _seen[key] = fname
            _dest[fname] = os.path.join(o["actual_out"], name)
        _srcs = {os.path.normcase(os.path.abspath(os.path.join(src_folder, f))) for f in files}
        _clash = [d for d in _dest.values() if os.path.normcase(os.path.abspath(d)) in _srcs]
        if _clash:
            msg = tr(f"Sortie refusée : {len(_clash)} fichier(s) de sortie écraseraient des images source "
                     f"(ex. {os.path.basename(_clash[0])}). Choisissez un autre dossier ou cochez « Sous-dossier upscaled/ ».",
                     f"Output refused: {len(_clash)} output file(s) would overwrite source images "
                     f"(e.g. {os.path.basename(_clash[0])}). Choose another folder or tick 'Sub-folder upscaled/'.")
            cb("❌ " + msg)
            if emit_done:
                self.done.emit(False, msg)
            return 0, total, [msg]
        success = 0; errors = []; self._produced = []
        batch_start = time.monotonic(); img_times = []
        dstats = {"skipped": 0}

        # optional heavy helpers (persistent batch / skip frames / post-proc)
        _session = _pp = None
        _pp_started = _tf_active = _ud_active = False
        _tf_lat = _ud_lat = 0
        _pil_to_float = _float_to_pil = dandere_should_skip = None
        prev_lq = prev_sr = None
        try:
            from src.core.quick_upscale import (PersistentBatchSession, _TRAINNER_VENV_PY,
                                                dandere_should_skip as _dss, _pil_to_float as _p2f,
                                                _float_to_pil as _f2p)
            dandere_should_skip, _pil_to_float, _float_to_pil = _dss, _p2f, _f2p
        except Exception as e:
            if o["persistent"] or o["dandere"]:
                cb(tr(f"[Avancé] helpers indisponibles ({e}) — mode normal", f"[Advanced] helpers unavailable ({e}) — normal mode"))
            _TRAINNER_VENV_PY = None

        if o["persistent"] and o["model"].lower().endswith(".onnx"):
            cb(tr("[Persistent] Non disponible pour ONNX — une session ONNX par image",
                  "[Persistent] Not available for ONNX — one ONNX session per image"))
        elif o["persistent"] and _TRAINNER_VENV_PY:
            try:
                from src.core.quick_upscale import make_persistent_session
                _session = make_persistent_session(o["model"], tile_size=o["tile"], tile_pad=o["tile_pad"],
                                                   use_amp=o["amp"], log=cb, scale_hint=o["scale"])
                if _session is not None:
                    # A10/A11: the worker writes the chosen format, bit depth and quality
                    _session.save_opts = {"format": o["out_format"], "bit_depth": o["bit_depth"],
                                          "quality": o["quality"]}
                if _session is not None and not _session.start():
                    cb(tr("[Persistent] Démarrage échoué — mode normal utilisé", "[Persistent] Startup failed — normal mode used")); _session = None
            except Exception as e:
                cb(tr(f"[Persistent] Erreur ({e}) — mode normal", f"[Persistent] Error ({e}) — normal mode")); _session = None

        if (o["tempfix"] or o["undistort"]) and _TRAINNER_VENV_PY:
            try:
                from src.core.post_proc_session import PostProcSession
                _pp = PostProcSession(venv_py=_TRAINNER_VENV_PY, log=cb); _pp_started = _pp.start()
            except Exception as e:
                cb(tr(f"[Post-proc] Indisponible ({e})", f"[Post-proc] Unavailable ({e})")); _pp = None
        if o["tempfix"] and _pp_started:
            try:
                _tf_lat = _pp.init_tf(mode=o["tf_mode"], strength=o["tf_strength"],
                                      window=o["tf_window"], precision=o["tf_precision"]); _tf_active = True
                cb(tr(f"[TemporalFix] Activé — mode={o['tf_mode']} fenêtre={o['tf_window']} (latence {_tf_lat}f)", f"[TemporalFix] Enabled — mode={o['tf_mode']} window={o['tf_window']} (latency {_tf_lat}f)"))
            except Exception as e:
                cb(tr(f"[TemporalFix] Init échoué ({e})", f"[TemporalFix] Init failed ({e})"))
        if o["undistort"] and _pp_started:
            try:
                _ud_lat = _pp.init_ud(mode=o["ud_mode"], strength=o["ud_strength"],
                                      window=o["ud_window"], precision=o["ud_precision"]); _ud_active = True
                cb(tr(f"[Undistort] Activé — mode={o['ud_mode']} fenêtre={o['ud_window']} (latence {_ud_lat}f)", f"[Undistort] Enabled — mode={o['ud_mode']} window={o['ud_window']} (latency {_ud_lat}f)"))
            except Exception as e:
                cb(tr(f"[Undistort] Init échoué ({e})", f"[Undistort] Init failed ({e})"))
        _chained = False
        if _pp_started and (_tf_active or _ud_active):
            try:
                _pp.set_quality(o["quality"])
                if _tf_active and _ud_active:
                    # A25: Undistort output feeds Temporal Fix inside the worker
                    _pp.chain_ud_tf(True); _chained = True
            except Exception as e:
                cb(tr(f"[Post-proc] Configuration échouée ({e})", f"[Post-proc] Setup failed ({e})"))

        def _fmt(s):
            s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
            return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s" if m else f"{sec}s"

        _ppc = {"tf": 0, "ud": 0}   # compteurs frames passées en Temporal Fix / Undistort

        def _upd_timing(done, dur):
            img_times.append(dur); recent = img_times[-5:]; avg = sum(recent) / len(recent)
            fps = 1.0 / avg if avg > 0 else 0.0
            eta = avg * (total - done); el = time.monotonic() - batch_start
            ds = f"   {dstats['skipped']} skip↩" if (o["dandere"] and dstats["skipped"]) else ""
            # Ajout à DROITE (sans retirer l'existant) : compteurs post-proc si actifs.
            pp = ""
            if _tf_active:
                pp += f"   TF:{_ppc['tf']}/{total}"
            if _ud_active:
                pp += f"   UD:{_ppc['ud']}/{total}"
            self.timing.emit(tr(f"Écoulé : {_fmt(el)}   ETA : {_fmt(eta)}   "
                             f"Vitesse : {fps:.2f} img/s{ds}   [{done}/{total}]{pp}",
                             f"Elapsed: {_fmt(el)}   ETA: {_fmt(eta)}   "
                             f"Speed: {fps:.2f} img/s{ds}   [{done}/{total}]{pp}"))

        self.timing.emit("")
        try:
            for i, fname in enumerate(files):
                if self.stop.is_set():
                    cb(tr("Arrêt demandé par l'utilisateur.", "Stop requested by the user.")); break
                in_path = os.path.join(src_folder, fname); base = os.path.splitext(fname)[0]
                out_path = _dest[fname]
                cb(f"[{i + 1}/{total}] {fname}")

                curr_lq = None
                if o["dandere"] and _pil_to_float:
                    try:
                        from PIL import Image as _Im
                        curr_lq = _pil_to_float(_Im.open(in_path))
                    except Exception:
                        pass

                if (o["dandere"] and prev_lq is not None and curr_lq is not None and prev_sr is not None
                        and dandere_should_skip):
                    if dandere_should_skip(prev_lq, curr_lq, block_size=o["dandere_blocks"], threshold=o["dandere_thr"]):
                        cb(tr(f"  [Skip frames] tous blocs < {o['dandere_thr']:.4f}", f"  [Skip frames] all blocks < {o['dandere_thr']:.4f}"))
                        try:
                            _float_to_pil(prev_sr).save(out_path)
                        except Exception as e:
                            # A26: a frame that was not written is an error, not a success
                            errors.append(f"{fname}: skip-frame save failed ({e})")
                            self.progress.emit((i + 1) / total); _upd_timing(i + 1, 0.05); continue
                        dstats["skipped"] += 1; success += 1  # prev_lq stays the LQ of prev_sr (A27)
                        self._produced.append(out_path)
                        self.preview.emit(in_path, out_path)
                        # A27: the reused frame still goes through Undistort / Temporal Fix,
                        # otherwise the post-processed sequence has holes
                        if _ud_active:
                            try:
                                _pp.push_ud(out_path); _ppc["ud"] += 1
                            except Exception as e:
                                cb(tr(f"  [Undistort] Erreur {i} : {e}", f"  [Undistort] Error {i}: {e}")); _ud_active = False
                        if _tf_active and _chained and _ud_active:
                            _ppc["tf"] += 1   # forwarded by push_ud
                        elif _tf_active:
                            try:
                                _pp.push_tf(out_path); _ppc["tf"] += 1
                            except Exception as e:
                                cb(tr(f"  [TemporalFix] Erreur {i} : {e}", f"  [TemporalFix] Error {i}: {e}")); _tf_active = False
                        self.progress.emit((i + 1) / total); _upd_timing(i + 1, 0.05); continue

                s0, s1 = i / total, (i + 1) / total
                t0 = time.monotonic(); ok, msg = False, ""
                if _session is not None and getattr(_session, "is_ready", False):
                    ok, msg, _extra = _session.infer(in_path, out_path)
                    if _extra.get("note"):  # worker auto-tile after a CUDA OOM
                        cb(_extra["note"])
                    # Le subprocess persistant ne fait PAS le Color Fix → on l'applique ici
                    # (color_fix_image = numpy/PIL, pas besoin de torch en-process).
                    if ok and o.get("color_fix", "none") != "none" and o.get("cf_strength", 0.0) > 0.0:
                        try:
                            from src.core.quick_upscale import color_fix_image as _cfi, load_color_fix_ref as _cfref
                            import numpy as _np
                            from PIL import Image as _CImg
                            _sr = _read01(out_path)
                            _lq = _CImg.open(in_path).convert("RGB")
                            _pl = o["cf_planes"] if o.get("cf_planes") is not None else [0, 1, 2]
                            _sr = _cfi(_sr, _lq, method=o["color_fix"], wavelets=o["cf_wavelets"],
                                       radius=o["cf_radius"], fast=o["cf_fast"], strength=o["cf_strength"],
                                       planes=_pl, device=o["cf_device"],
                               ref_pil=_cfref(o.get("cf_ref"), cb))
                            from src.core.quick_upscale import _save_output as _so
                            _so(_sr.clip(0, 1), out_path, o["out_format"], o["bit_depth"], o["quality"])
                        except Exception as _e:
                            cb(tr(f"  [ColorFix] Erreur {i} : {_e}", f"  [ColorFix] Error {i}: {_e}"))
                else:
                    ok, msg = upscale_image(o["model"], in_path, out_path,
                                            **self._ups_kwargs(cb, lambda v: self.progress.emit(s0 + v * (s1 - s0))))
                dur = time.monotonic() - t0

                if ok:
                    success += 1; self.preview.emit(in_path, out_path); self._produced.append(out_path)
                    if o["dandere"] and _pil_to_float:
                        try:
                            from PIL import Image as _Im
                            prev_sr = _pil_to_float(_Im.open(out_path))
                        except Exception:
                            pass
                    if _ud_active:
                        try:
                            _pp.push_ud(out_path); _ppc["ud"] += 1
                        except Exception as e:
                            cb(tr(f"  [Undistort] Erreur {i} : {e}", f"  [Undistort] Error {i}: {e}")); _ud_active = False
                    if _tf_active and _chained and _ud_active:
                        _ppc["tf"] += 1   # forwarded by push_ud
                    elif _tf_active:
                        try:
                            _pp.push_tf(out_path); _ppc["tf"] += 1
                        except Exception as e:
                            cb(tr(f"  [TemporalFix] Erreur {i} : {e}", f"  [TemporalFix] Error {i}: {e}")); _tf_active = False
                else:
                    errors.append(f"{fname}: {msg}")
                    # show it now (one line) — the full message is still listed at the end
                    _last = str(msg).strip().splitlines()[-1] if str(msg).strip() else str(msg)
                    cb(tr(f"  ⚠ {fname} : {_last}", f"  ⚠ {fname}: {_last}"))
                if o["dandere"]:
                    prev_lq = curr_lq
                self.progress.emit(s1); _upd_timing(i + 1, dur)
        finally:
            if _session is not None:
                _session.stop()
            if _pp_started:
                if _ud_active and _ud_lat > 0:
                    cb(tr("[Undistort] Flush des dernières frames…", "[Undistort] Flushing the last frames…"))
                    try:
                        _pp.flush_ud()
                    except Exception:
                        pass
                if _tf_active and _tf_lat > 0:
                    cb(tr("[TemporalFix] Flush des dernières frames…", "[TemporalFix] Flushing the last frames…"))
                    try:
                        _pp.flush_tf()
                    except Exception:
                        pass
                _pp.stop()

        if o["dandere"]:
            pct = dstats["skipped"] / total * 100 if total else 0
            cb(tr(f"[Skip frames] {dstats['skipped']}/{total} frames ({pct:.0f}% GPU économisé)", f"[Skip frames] {dstats['skipped']}/{total} frames ({pct:.0f}% GPU saved)"))
        el = time.monotonic() - batch_start
        fps_final = success / el if el > 0 else 0
        self.timing.emit(tr(f"Terminé — {_fmt(el)} total, {fps_final:.2f} img/s moy.", f"Finished — {_fmt(el)} total, {fps_final:.2f} img/s avg."))
        cb(tr(f"Terminé : {success}/{total} images traitées.", f"Finished: {success}/{total} images processed."))
        for err in errors:
            cb(tr(f"  ERREUR : {err}", f"  ERROR: {err}"))
        if emit_done:
            # A28: a stopped run is reported as stopped, not as a plain partial result
            stopped = tr(" — arrêté", " — stopped") if self.stop.is_set() and success < total else ""
            self.done.emit(success == total, f"{success}/{total} OK{stopped}")
        return success, total, errors


# ════════════════════════════════════════════════════════════════════════════════
#  Page
# ════════════════════════════════════════════════════════════════════════════════
class QuickUpscalePage(QWidget):
    # Sidebar-badge hooks (qt_ui/tabs/tools_tab.py): emitted while a run is active
    # so the Tools nav shows live progress on the "Quick Upscale" entry even when
    # another tool page is currently displayed (the worker keeps running in the
    # background regardless of which page is on screen).
    toolProgress = Signal(int)   # 0-100
    toolFinished = Signal(bool)  # True = success (green badge), False = failure (red badge)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings; self.worker = None
        self._stop_event = None; self._last_preview = None
        self._cf = {
            "wavelets": int(settings.get("ups_colorfix_wavelets", 4)),
            "radius": int(settings.get("ups_colorfix_radius", 32)),
            "fast": _b(settings, "ups_colorfix_fast", False),
            "planes": settings.get("ups_colorfix_planes", [0, 1, 2]) if isinstance(settings.get("ups_colorfix_planes", [0, 1, 2]), list) else [0, 1, 2],
            "device": str(settings.get("ups_colorfix_device", "auto")),
            "ref": str(settings.get("ups_colorfix_ref", "")),
            "strength": float(settings.get("ups_colorfix_strength", 1.0)),
        }
        self._tf = {
            "strength": float(settings.get("ups_tempfix_strength", 0.5)),
            "window": int(settings.get("ups_tempfix_window", 7)),
            "precision": str(settings.get("ups_tempfix_precision", "float32")),
            "tempfix_mode": str(settings.get("ups_tempfix_mode", "classic")),
            "undistort_strength": float(settings.get("ups_undistort_strength", 0.35)),
            "undistort_window": int(settings.get("ups_undistort_window", 5)),
            "undistort_mode": str(settings.get("ups_undistort_mode", "classic")),
            "undistort_precision": str(settings.get("ups_undistort_precision", "float32")),
        }

        root = QVBoxLayout(self); root.setContentsMargins(16, 14, 16, 12); root.setSpacing(4)
        hdr = QHBoxLayout(); root.addLayout(hdr)
        left = QVBoxLayout(); left.setSpacing(0)
        t = QLabel("Quick Upscale"); t.setObjectName("title"); left.addWidget(t)
        sub = QLabel(tr("Upscaler une image ou un dossier avec un modèle entraîné.",
                        "Upscale an image or a folder with a trained model."))
        sub.setStyleSheet("color:#888;"); left.addWidget(sub)
        hdr.addLayout(left); hdr.addStretch(1)
        self.gpu_panel = _GpuMiniPanel()
        hdr.addWidget(self.gpu_panel, 0, Qt.AlignTop | Qt.AlignRight)

        # ── paths : label fixe + champ extensible (colle aux boutons) + boutons à droite ──
        _LBL_W = 210

        def _path_row(label_text, field, buttons):
            r = QHBoxLayout(); r.setSpacing(6)
            lb = QLabel(label_text); lb.setFixedWidth(_LBL_W)
            r.addWidget(lb); r.addWidget(field, 1)
            for b in buttons:
                r.addWidget(b)
            root.addLayout(r)

        self.model = QLineEdit(settings.get("ups_last_model", "") or "")
        self.model.setToolTip(tr("Modèle SR à utiliser pour l'upscale (.pth, .safetensors ou .onnx).",
                                 "SR model used for upscaling (.pth, .safetensors or .onnx)."))
        bm = QPushButton("…"); bm.setObjectName("ghost"); bm.setFixedSize(40, 28); bm.clicked.connect(self._pick_model)
        _path_row(tr("Modèle (.pth/.safetensors/.onnx) :", "Model (.pth/.safetensors/.onnx):"), self.model, [bm])

        self.inp = QLineEdit(settings.get("ups_last_input", "") or "")
        self.inp.setToolTip(tr("Image unique, dossier d'images (traitement par lot), OU fichier vidéo "
                                "(n'importe quel modèle — les frames sont extraites automatiquement ; "
                                "les 4 archs séquence tscunet/temporalspan/temporalspanv2/tfdat sont "
                                "traitées en clip temporel, tout autre modèle frame par frame).",
                                "Single image, folder of images (batch), OR a video file (any model — "
                                "frames are extracted automatically; the 4 sequence archs "
                                "tscunet/temporalspan/temporalspanv2/tfdat are processed as a temporal "
                                "clip, any other model frame by frame)."))
        self.inp.textChanged.connect(self._sync_output)
        bi = QPushButton(tr("Image", "Image")); bi.setFixedSize(80, 28); bi.clicked.connect(self._pick_image)
        bi.setToolTip(tr("Choisir une image unique.", "Pick a single image."))
        bf = QPushButton(tr("Dossier", "Folder")); bf.setFixedSize(86, 28); bf.clicked.connect(self._pick_input_folder)
        bf.setToolTip(tr("Choisir un dossier (traitement par lot de toutes les images).",
                          "Pick a folder (batch process all images)."))
        bv = QPushButton(tr("Vidéo", "Video")); bv.setFixedSize(80, 28); bv.clicked.connect(self._pick_video)
        bv.setToolTip(tr("Choisir un fichier vidéo — fonctionne avec n'importe quel modèle. Les frames "
                          "sont extraites automatiquement (ffmpeg) : les 4 archs séquence "
                          "(tscunet/temporalspan/temporalspanv2/tfdat) sont upscalées en clip temporel, "
                          "tout autre modèle frame par frame (comme un dossier classique).",
                          "Pick a video file — works with any model. Frames are extracted "
                          "automatically (ffmpeg): the 4 sequence archs "
                          "(tscunet/temporalspan/temporalspanv2/tfdat) are upscaled as a temporal clip, "
                          "any other model frame by frame (like a classic folder)."))
        _path_row(tr("Source (image, dossier ou vidéo) :", "Source (image, folder or video):"), self.inp, [bi, bf, bv])

        self.out = QLineEdit(settings.get("ups_last_output", "") or "")
        self.out.setToolTip(tr("Dossier où enregistrer les images upscalées.",
                                "Folder where upscaled images are saved."))
        self.bo = QPushButton("…"); self.bo.setObjectName("ghost"); self.bo.setFixedSize(40, 28); self.bo.clicked.connect(self._pick_output_folder)
        _path_row(tr("Dossier sortie :", "Output folder:"), self.out, [self.bo])

        # ── output options row (centered) ──
        root.addSpacing(8)
        opt = QHBoxLayout(); opt.setSpacing(14); root.addLayout(opt); opt.addStretch(1)
        self.same_folder = QCheckBox(tr("Même dossier que la source", "Same folder as source")); self.same_folder.setChecked(_b(settings, "ups_same_folder", False))
        self.same_folder.setToolTip(tr("Écrit le résultat à côté de l'image source (ignore le dossier de sortie).",
                                       "Writes the result next to the source image (ignores the output folder)."))
        self.same_folder.toggled.connect(self._on_same_folder); opt.addWidget(self.same_folder)
        self.subfolder = QCheckBox(tr('Sous-dossier "upscaled/" (garde le nom original)', 'Sub-folder "upscaled/" (keep original name)')); self.subfolder.setChecked(_b(settings, "ups_subfolder", True))
        self.subfolder.setToolTip(tr("Crée un sous-dossier « upscaled/ » et garde le nom de fichier d'origine.",
                                     "Creates an 'upscaled/' sub-folder and keeps the original filename."))
        opt.addWidget(self.subfolder)
        self.modelname = QCheckBox(tr("Nom du modèle dans le fichier", "Model name in filename")); self.modelname.setChecked(_b(settings, "ups_modelname", False))
        self.modelname.setToolTip(tr("Ajoute le nom du modèle au nom de fichier (utile pour comparer plusieurs modèles).",
                                     "Appends the model name to the filename (handy to compare several models)."))
        opt.addWidget(self.modelname)
        opt.addWidget(self._vsep())
        self.serialize = QCheckBox(tr("Sérialisation", "Serialization")); self.serialize.setChecked(_b(settings, "ups_serialize", False))
        self.serialize.setToolTip(tr("Renomme les sorties en numéros séquentiels (0001, 0002…) à partir du « Début ».",
                                     "Renames outputs as sequential numbers (0001, 0002…) starting from 'Start'."))
        self.serialize.toggled.connect(self._on_serialize); opt.addWidget(self.serialize)
        opt.addWidget(QLabel(tr("Début :", "Start:")))
        self.serialize_start = QLineEdit(str(settings.get("ups_serialize_start", "0"))); self.serialize_start.setFixedWidth(55)
        opt.addWidget(self.serialize_start)
        opt.addWidget(self._vsep())
        opt.addWidget(QLabel(tr("Format sortie :", "Output format:")))
        self.fmt = QComboBox(); self.fmt.addItems(["PNG", "JPEG", "WEBP", "TIFF", "BMP"]); self.fmt.setFixedWidth(80)
        self.fmt.setCurrentText(settings.get("ups_format", "PNG")); self.fmt.currentTextChanged.connect(self._on_format)
        self.fmt.setToolTip(tr("Format des images de sortie. PNG/TIFF = sans perte · WEBP = bon compromis · JPEG = avec perte.",
                               "Output image format. PNG/TIFF = lossless · WEBP = good tradeoff · JPEG = lossy."))
        opt.addWidget(self.fmt)
        self.bitdepth = QComboBox(); self.bitdepth.addItems(["8 bits", "16 bits"]); self.bitdepth.setFixedWidth(80)
        self.bitdepth.setCurrentText(settings.get("ups_bitdepth", "8 bits"))
        self.bitdepth.setToolTip(tr("Profondeur par canal. 16 bits = dégradés plus fins / moins de banding (PNG/TIFF), fichiers plus lourds.",
                                    "Per-channel depth. 16-bit = smoother gradients / less banding (PNG/TIFF), larger files."))
        opt.addWidget(self.bitdepth)
        opt.addWidget(QLabel(tr("Qualité :", "Quality:")))
        self.quality = QComboBox(); self.quality.addItems(["70", "75", "80", "85", "90", "95", "100"]); self.quality.setFixedWidth(60)
        self.quality.setCurrentText(str(settings.get("ups_quality", "95")))
        self.quality.setToolTip(tr("Qualité de compression pour JPEG/WEBP (100 = max). Sans effet sur PNG/TIFF.",
                                   "Compression quality for JPEG/WEBP (100 = max). No effect on PNG/TIFF."))
        opt.addWidget(self.quality)
        opt.addStretch(1)

        # ── scale / tile / amp / colorfix / tf / ud (centered) ──
        root.addSpacing(8)
        o2 = QHBoxLayout(); o2.setSpacing(10); root.addLayout(o2); o2.addStretch(1)
        o2.addWidget(QLabel(tr("Scale :", "Scale:")))
        self.scale = QComboBox(); self.scale.addItems(["Auto", "1", "2", "3", "4", "8"]); self.scale.setFixedWidth(70)
        self.scale.setCurrentText(settings.get("ups_scale", "Auto")); o2.addWidget(self.scale)
        self.scale.setToolTip(tr("Facteur d'agrandissement. « Auto » = lu depuis le modèle (recommandé).\n"
                                 "Upscale factor. 'Auto' = read from the model (recommended).",
                                 "Upscale factor. 'Auto' = read from the model (recommended)."))
        o2.addWidget(QLabel(tr("Tile :", "Tile:")))
        self.tile = QComboBox(); self.tile.addItems(["Auto", "128", "192", "256", "384", "512", "768", "1024", "0 (pas de tile)"])
        self.tile.setFixedWidth(110); self.tile.setCurrentText(settings.get("ups_tile", "Auto")); o2.addWidget(self.tile)
        self.tile.setToolTip(tr("Découpe l'image en tuiles pour économiser la VRAM.\n"
                                "↓ tuile = moins de VRAM mais plus lent et risque de coutures · "
                                "↑ tuile = plus rapide mais + de VRAM · 0 = image entière d'un coup.\n"
                                "Splits the image into tiles to save VRAM. ↓ tile = less VRAM but slower / seam risk · "
                                "↑ tile = faster but more VRAM · 0 = whole image at once.",
                                "Splits the image into tiles to save VRAM. ↓ tile = less VRAM but slower / seam risk · "
                                "↑ tile = faster but more VRAM · 0 = whole image at once."))
        o2.addWidget(QLabel(tr("Tile Pad :", "Tile Pad:")))
        self.tile_pad = QComboBox(); self.tile_pad.addItems(["Auto", "8", "16", "32", "48", "64"]); self.tile_pad.setFixedWidth(80)
        self.tile_pad.setCurrentText(settings.get("ups_tile_pad", "Auto")); o2.addWidget(self.tile_pad)
        self.tile_pad.setToolTip(tr("Recouvrement entre tuiles pour masquer les coutures.\n"
                                    "↑ = coutures mieux cachées mais plus lent · ↓ = plus rapide mais coutures possibles.\n"
                                    "Overlap between tiles to hide seams. ↑ = better seams but slower · ↓ = faster but seams may show.",
                                    "Overlap between tiles to hide seams. ↑ = better seams but slower · ↓ = faster but seams may show."))
        self.amp = QCheckBox("AMP (FP16)"); self.amp.setChecked(_b(settings, "ups_amp", True)); o2.addWidget(self.amp)
        self.amp.setToolTip(tr("Précision mixte FP16 : ~2× plus rapide + moitié moins de VRAM, qualité quasi identique.\n"
                               "À désactiver seulement si artefacts/NaN sur certains modèles.\n"
                               "FP16 mixed precision: ~2× faster + half the VRAM, near-identical quality. "
                               "Disable only if a model shows artifacts/NaN.",
                               "FP16 mixed precision: ~2× faster + half the VRAM, near-identical quality. "
                               "Disable only if a model shows artifacts/NaN."))
        o2.addWidget(self._vsep())
        self.colorfix = QCheckBox("Color Fix"); self.colorfix.setChecked(_b(settings, "ups_colorfix", False))
        self.colorfix.setToolTip(tr("Réaligne les couleurs de la sortie sur la source (corrige les dérives de teinte de certains modèles).\n"
                                    "Re-aligns output colors to the source (fixes hue/tint drift some models introduce).",
                                    "Re-aligns output colors to the source (fixes hue/tint drift some models introduce)."))
        self.colorfix.toggled.connect(self._on_colorfix); o2.addWidget(self.colorfix)
        o2.addWidget(QLabel(tr("Méthode :", "Method:")))
        self.cf_method = QComboBox(); self.cf_method.addItems(["wavelet", "average", "lab", "hist", "colormap", "guided"]); self.cf_method.setFixedWidth(100)
        self.cf_method.setCurrentText(settings.get("ups_colorfix_method", "wavelet"))
        self.cf_method.currentTextChanged.connect(lambda m: self.settings.set("ups_colorfix_method", m))
        o2.addWidget(self.cf_method)
        self.cf_btn = QPushButton("⚙ ColorFix…"); self._slim_btn(self.cf_btn)
        self.cf_btn.clicked.connect(self._open_colorfix); o2.addWidget(self.cf_btn)
        o2.addWidget(self._vsep())
        self.tempfix = QCheckBox("Temporal Fix"); self.tempfix.setChecked(_b(settings, "ups_tempfix", False)); o2.addWidget(self.tempfix)
        self.tempfix.setToolTip(tr("Réduit le scintillement entre frames sur une séquence vidéo (batch d'images).",
                                   "Reduces frame-to-frame flicker on a video sequence (image batch)."))
        self.tf_btn = QPushButton("⚙ TF…"); self._slim_btn(self.tf_btn)
        self.tf_btn.clicked.connect(self._open_tf); o2.addWidget(self.tf_btn)
        o2.addWidget(self._vsep())
        self.undistort = QCheckBox("Undistort"); self.undistort.setChecked(_b(settings, "ups_undistort", False)); o2.addWidget(self.undistort)
        self.undistort.setToolTip(tr("Réduit le 'shimmer' / jitter haute fréquence sur les contours (séquence vidéo).",
                                     "Reduces high-frequency shimmer/jitter on edges (video sequence)."))
        self.ud_btn = QPushButton("⚙ UD…"); self._slim_btn(self.ud_btn)
        self.ud_btn.clicked.connect(self._open_ud); o2.addWidget(self.ud_btn)
        o2.addStretch(1)

        # ── persistent + skip frames (centered) ──
        root.addSpacing(8)
        o3 = QHBoxLayout(); o3.setSpacing(12); root.addLayout(o3); o3.addStretch(1)
        self.persistent = QCheckBox(tr("Subprocess persistant", "Persistent subprocess")); self.persistent.setChecked(_b(settings, "ups_persistent", False))
        o3.addWidget(self.persistent); o3.addWidget(self._vsep())
        self.dandere = QCheckBox("Skip frames"); self.dandere.setChecked(_b(settings, "ups_dandere", False))
        self.dandere.toggled.connect(self._on_dandere); o3.addWidget(self.dandere)
        _dandere_thr_tip = tr(
            "Skip frames : une frame est traitée comme identique à la précédente (upscale sauté, "
            "réutilise le résultat précédent) seulement si TOUS ses blocs ont un écart (MAE) sous ce "
            "seuil. Plus haut = plus de frames sautées (plus rapide) mais risque de rater un petit "
            "mouvement (ex. juste les lèvres qui bougent). Plus bas = plus prudent, moins de frames "
            "sautées. Défaut (papier) : 0.010.",
            "Skip frames: a frame is treated as identical to the previous one (upscale skipped, "
            "reuses the previous result) only if ALL its blocks have a diff (MAE) below this "
            "threshold. Higher = more frames skipped (faster) but risks missing small motion (e.g. "
            "just the lips moving). Lower = more cautious, fewer frames skipped. Default (paper): "
            "0.010.")
        _dandere_blk_tip = tr(
            "Skip frames : taille (en pixels) des blocs carrés utilisés pour comparer une frame à la "
            "précédente. Comparer par bloc plutôt que sur l'image entière évite qu'un mouvement "
            "localisé (ex. juste un visage qui tourne) soit noyé dans la moyenne globale et "
            "faussement considéré comme \"pas de changement\". Plus petit = détection plus fine du "
            "mouvement local mais plus lent à calculer ; plus grand = plus rapide mais moins précis. "
            "Défaut : 14.",
            "Skip frames: size (in pixels) of the square blocks used to compare a frame to the "
            "previous one. Comparing block-by-block instead of on the whole image prevents localized "
            "motion (e.g. just a face turning) from being averaged away and wrongly treated as \"no "
            "change\". Smaller = finer local-motion detection but slower to compute; larger = faster "
            "but less precise. Default: 14.")
        _lbl_thr = QLabel(tr("Seuil :", "Threshold:")); _lbl_thr.setToolTip(_dandere_thr_tip)
        o3.addWidget(_lbl_thr)
        self.dandere_thr = QLineEdit(str(settings.get("ups_dandere_threshold", "0.010"))); self.dandere_thr.setFixedWidth(55)
        self.dandere_thr.setToolTip(_dandere_thr_tip)
        o3.addWidget(self.dandere_thr)
        _lbl_blk = QLabel(tr("Blocs :", "Blocks:")); _lbl_blk.setToolTip(_dandere_blk_tip)
        o3.addWidget(_lbl_blk)
        self.dandere_blk = QLineEdit(str(settings.get("ups_dandere_block_size", "14"))); self.dandere_blk.setFixedWidth(45)
        self.dandere_blk.setToolTip(_dandere_blk_tip)
        o3.addWidget(self.dandere_blk)
        o3.addWidget(self._vsep())
        self.keep_frames = QCheckBox(tr("Garder les frames extraites", "Keep extracted frames"))
        self.keep_frames.setChecked(_b(settings, "ups_keep_frames", False))
        self.keep_frames.setToolTip(tr("Entrée vidéo uniquement : conserve le dossier temporaire de frames "
                                       "extraites au lieu de le supprimer après l'upscale.",
                                       "Video input only: keeps the temporary extracted-frames folder "
                                       "instead of deleting it after the upscale."))
        o3.addWidget(self.keep_frames)
        self.reassemble_video = QCheckBox(tr("Réassembler en vidéo (ffmpeg)", "Reassemble to video (ffmpeg)"))
        self.reassemble_video.setChecked(_b(settings, "ups_reassemble_video", False))
        self.reassemble_video.setToolTip(tr("Entrée vidéo uniquement : en plus du dossier de frames upscalées, "
                                            "réencode une vidéo (conteneur/encodeur/qualité réglables via "
                                            "« ⚙ Encodage… ») au framerate de la source détecté via ffmpeg.",
                                            "Video input only: in addition to the folder of upscaled frames, "
                                            "re-encodes a video (container/encoder/quality adjustable via "
                                            "'⚙ Encoding…') at the source framerate detected via ffmpeg."))
        o3.addWidget(self.reassemble_video)
        self.enc_btn = QPushButton(tr("⚙ Encodage…", "⚙ Encoding…")); self._slim_btn(self.enc_btn)
        self.enc_btn.clicked.connect(self._open_enc); o3.addWidget(self.enc_btn)
        o3.addWidget(self._vsep())
        o3.addWidget(QLabel(tr("Dossier temp. :", "Temp folder:")))
        self.temp_dir = QLineEdit(str(settings.get("ups_temp_dir", "") or ""))
        self.temp_dir.setFixedWidth(160)
        self.temp_dir.setPlaceholderText(tr("(par défaut système)", "(system default)"))
        self.temp_dir.setToolTip(tr(
            "Entrée vidéo uniquement : dossier où sont extraites les frames temporaires avant "
            "upscale. Vide = dossier temp système par défaut (souvent sur le disque C:, peut "
            "manquer d'espace pour une longue vidéo). Choisis un disque avec assez de place libre.",
            "Video input only: folder where temporary frames are extracted before upscaling. "
            "Empty = system default temp folder (often on the C: drive, can run out of space for "
            "a long video). Pick a drive with enough free space."))
        self.temp_dir.textChanged.connect(lambda t: self.settings.set("ups_temp_dir", t.strip()))
        o3.addWidget(self.temp_dir)
        self.temp_dir_btn = QPushButton("…"); self._slim_btn(self.temp_dir_btn)
        self.temp_dir_btn.setFixedWidth(28)
        self.temp_dir_btn.clicked.connect(self._pick_temp_dir)
        o3.addWidget(self.temp_dir_btn)
        o3.addWidget(self._vsep())
        o3.addWidget(QLabel(tr("Format extraction :", "Extraction format:")))
        self.vid_format = QComboBox()
        # Same key set as LQ Generator's video-mode combo (qt_ui/tabs/tools_pages/gen_lq.py):
        # TIFF is left out for consistency even though it wouldn't break anything here
        # (these frames are always temporary: extracted -> upscaled -> deleted, unless
        # "Keep extracted frames" is checked — no PairedVideoDataset involved on this path).
        self._vid_format_keys = ["png8", "png16", "jpg"]
        self.vid_format.addItems([
            tr("PNG 8-bit (par défaut)", "PNG 8-bit (default)"),
            tr("PNG 16-bit", "PNG 16-bit"),
            "JPEG",
        ])
        _fmt_saved = str(settings.get("ups_vid_format", "png8") or "png8")
        self.vid_format.setCurrentIndex(
            self._vid_format_keys.index(_fmt_saved) if _fmt_saved in self._vid_format_keys else 0)
        self.vid_format.setToolTip(tr(
            "Entrée vidéo uniquement : format des frames extraites avant upscale (toujours temporaires, "
            "supprimées après — sauf si « Garder les frames extraites » est coché).",
            "Video input only: format of the frames extracted before upscaling (always temporary, "
            "deleted afterwards — unless 'Keep extracted frames' is checked)."))
        self.vid_format.setFixedWidth(150)
        self.vid_format.currentIndexChanged.connect(self._on_vid_format_change)
        o3.addWidget(self.vid_format)
        self.vid_jpg_q_label = QLabel(tr("Qualité JPEG :", "JPEG quality:"))
        o3.addWidget(self.vid_jpg_q_label)
        self.vid_jpg_quality = QSlider(Qt.Horizontal)
        self.vid_jpg_quality.setRange(1, 100)
        self.vid_jpg_quality.setValue(int(settings.get("ups_vid_jpg_quality", 95) or 95))
        self.vid_jpg_quality.setFixedWidth(100)
        self.vid_jpg_q_value = QLabel(str(self.vid_jpg_quality.value())); self.vid_jpg_q_value.setFixedWidth(28)
        self.vid_jpg_quality.valueChanged.connect(
            lambda v: (self.vid_jpg_q_value.setText(str(v)), settings.set("ups_vid_jpg_quality", v)))
        o3.addWidget(self.vid_jpg_quality); o3.addWidget(self.vid_jpg_q_value)
        o3.addWidget(QLabel(tr("Sortie séquence :", "Sequence output:")))
        self.vid_out_bitdepth = QComboBox()
        # Bit-depth of the frames WRITTEN by the sequence (temporal) upscale
        # pipeline — tscunet/temporalspan/temporalspanv2/tfdat only. 8-bit is
        # the default and reproduces the exact previous behavior byte-for-byte.
        # 16-bit writes genuine 16-bit-per-channel PNGs (via pyvips) — real
        # gain only if the extracted frames above are also "PNG 16-bit" AND
        # the source video itself carries more than 8-bit precision; on an
        # 8-bit source it just avoids re-quantization noise from the model's
        # own float32 math, a smaller but still real benefit.
        self.vid_out_bitdepth.addItems([
            tr("8-bit (par défaut)", "8-bit (default)"),
            tr("16-bit (PNG16 réel)", "16-bit (real PNG16)"),
        ])
        _outbd_saved = str(settings.get("ups_vid_out_bitdepth", "8") or "8")
        self.vid_out_bitdepth.setCurrentIndex(1 if _outbd_saved == "16" else 0)
        self.vid_out_bitdepth.setToolTip(tr(
            "Entrée vidéo uniquement (archs séquence) : profondeur des frames upscalées écrites "
            "sur disque. 16-bit = PNG16 réel (via pyvips), pas une conversion cosmétique — "
            "utile pour un master haute précision (moins de banding). Le réassemblage vidéo "
            "final (ffmpeg) reste en 8-bit (yuv420p) même si les frames intermédiaires sont "
            "en 16-bit — seul le dossier de frames upscalées profite pleinement du 16-bit.",
            "Video input only (sequence archs): bit depth of the upscaled frames written to "
            "disk. 16-bit = real PNG16 (via pyvips), not a cosmetic conversion — useful for a "
            "high-precision master (less banding). The final video reassembly (ffmpeg) stays "
            "8-bit (yuv420p) even if the intermediate frames are 16-bit — only the folder of "
            "upscaled frames fully benefits from the 16-bit output."))
        self.vid_out_bitdepth.setFixedWidth(160)
        self.vid_out_bitdepth.currentIndexChanged.connect(
            lambda idx: settings.set("ups_vid_out_bitdepth", "16" if idx == 1 else "8"))
        o3.addWidget(self.vid_out_bitdepth)
        self._on_vid_format_change(self.vid_format.currentIndex())
        o3.addStretch(1)

        self._enc = {
            "container": settings.get("ups_enc_container", "MP4") or "MP4",
            "encoder": settings.get("ups_enc_encoder", "libx264") or "libx264",
            "quality_mode": settings.get("ups_enc_quality_mode", "crf") or "crf",
            "crf": int(float(settings.get("ups_enc_crf", 23) or 23)),
            "bitrate_mbps": float(settings.get("ups_enc_bitrate_mbps", 8.0) or 8.0),
            "preset": settings.get("ups_enc_preset", "medium") or "medium",
        }

        # ── run / stop / progress ──
        rr = QHBoxLayout(); root.addLayout(rr)
        self.btn_run = QPushButton(tr("⚡ Lancer Upscale", "⚡ Run Upscale"))
        self.btn_run.setToolTip(tr("Lance l'upscale de l'image ou du dossier source avec le modèle sélectionné.", "Runs the upscale of the source image or folder with the selected model."))
        self.btn_run.setStyleSheet(
            'QPushButton{background:#2ecc71; color:#fff; border:none; border-radius:6px; padding:9px;'
            ' font-weight:700; font-family:"Segoe UI Emoji"; font-size:13px;}'
            'QPushButton:hover{background:#3ee07f;}'
            'QPushButton:pressed{background:#22a35a; padding-top:11px;}'
            'QPushButton:disabled{background:#2f4a3a; color:#8aa;}')
        self.btn_run.clicked.connect(self._run); rr.addWidget(self.btn_run, 1)
        self.btn_stop = QPushButton("⏹ Stop"); self.btn_stop.setEnabled(False); self.btn_stop.setFixedWidth(110)
        self.btn_stop.setToolTip(tr("Arrête l'upscale en cours.", "Stops the current upscale."))
        self.btn_stop.setStyleSheet(
            'QPushButton{background:#e74c3c; color:#fff; border:none; border-radius:6px; padding:9px;'
            ' font-weight:700; font-family:"Segoe UI Emoji"; font-size:13px;}'
            'QPushButton:hover{background:#ef6657;}'
            'QPushButton:pressed{background:#b83a2e; padding-top:11px;}'
            'QPushButton:disabled{background:#4a3a3a; color:#a88;}')
        self.btn_stop.clicked.connect(self._request_stop); rr.addWidget(self.btn_stop)

        pr = QHBoxLayout(); root.addLayout(pr)
        self.prog = QProgressBar(); self.prog.setRange(0, 100); pr.addWidget(self.prog, 1)
        self.prog_pct = QLabel("0%"); self.prog_pct.setFixedWidth(42); self.prog_pct.setAlignment(Qt.AlignRight)
        pr.addWidget(self.prog_pct)
        self.timing = QLabel(""); self.timing.setStyleSheet("color:#9B59B6;"); self.timing.setFont(QFont("Consolas", 9))
        root.addWidget(self.timing)

        # ── split: log (top) / preview (bottom) ──
        split = QSplitter(Qt.Vertical)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFont(QFont("Consolas", 10))
        self.log.setMaximumBlockCount(8000); split.addWidget(self.log)
        prev = QFrame(); prev.setObjectName("prevpanel"); prev.setStyleSheet("QFrame#prevpanel{background:#111827; border-radius:6px;}")
        ph = QHBoxLayout(prev)
        self.prev_in, self.res_in = self._prev_col(ph, "Source", "#e67e22")
        self.prev_out, self.res_out = self._prev_col(ph, tr("Résultat", "Result"), "#2ecc71")
        split.addWidget(prev); split.setSizes([170, 240])
        root.addWidget(split, 1)

        self._on_format(self.fmt.currentText()); self._on_same_folder(self.same_folder.isChecked())
        self._on_serialize(self.serialize.isChecked()); self._on_colorfix(self.colorfix.isChecked())
        self._on_dandere(self.dandere.isChecked())
        self.inp.textChanged.connect(self._sync_applicability)
        self.model.textChanged.connect(self._sync_applicability)
        self._sync_applicability()

    # A06 / A07: options the chosen pipeline does not use are greyed out with the reason,
    # instead of staying clickable and silently ignored.
    _SINGLE_DEAD = ("serialize", "serialize_start", "tempfix", "tf_btn", "undistort", "ud_btn",
                    "dandere", "dandere_thr", "dandere_blk")
    _TEMPORAL_DEAD = ("serialize", "serialize_start", "modelname", "tempfix", "tf_btn", "undistort",
                      "ud_btn", "dandere", "dandere_thr", "dandere_blk", "persistent", "tile",
                      "tile_pad", "amp")

    def _temporal_model(self, model):
        """Sequence arch of a .safetensors model (header only — cheap); None otherwise."""
        if not model.lower().endswith(".safetensors") or not os.path.isfile(model):
            return None
        key = (model, os.path.getmtime(model))
        if getattr(self, "_arch_key", None) != key:
            try:
                from src.core.quick_upscale import _peek_arch_from_model_path, _SEQUENCE_ARCHS
                a = _peek_arch_from_model_path(model)
                self._arch_val = a if a in _SEQUENCE_ARCHS else None
            except Exception:
                self._arch_val = None
            self._arch_key = key
        return self._arch_val

    def _sync_applicability(self, *_):
        inp = self.inp.text().strip(); model = self.model.text().strip()
        is_video = os.path.isfile(inp) and os.path.splitext(inp)[1].lower() in _VIDEO_EXT
        is_single = os.path.isfile(inp) and not is_video
        temporal = (is_video or os.path.isdir(inp)) and self._temporal_model(model)
        if temporal:
            dead, why = self._TEMPORAL_DEAD, tr(f"Non utilisé par le pipeline séquence ({temporal}).",
                                                f"Not used by the sequence pipeline ({temporal}).")
        elif is_single:
            dead, why = self._SINGLE_DEAD, tr("Non utilisé pour une image seule.", "Not used for a single image.")
        else:
            dead, why = (), ""
        names = set(self._SINGLE_DEAD) | set(self._TEMPORAL_DEAD)
        for n in names:
            w = getattr(self, n, None)
            if w is None:
                continue
            if not hasattr(w, "_base_tip"):
                w._base_tip = w.toolTip()
            off = n in dead
            w.setEnabled(not off)
            w.setToolTip((why + ("\n" + w._base_tip if w._base_tip else "")) if off else w._base_tip)
        # re-apply the dependent states of the controls that stayed enabled
        if "serialize" not in dead:
            self._on_serialize(self.serialize.isChecked())
        if "dandere" not in dead:
            self._on_dandere(self.dandere.isChecked())

    # ── small ui helpers ──
    def _vsep(self):
        f = QFrame(); f.setFrameShape(QFrame.VLine); f.setStyleSheet("color:#555;"); return f

    def _slim_btn(self, b):
        """Make a ⚙ settings button match the height of the combos next to it."""
        b.setFixedHeight(28)
        b.setStyleSheet('QPushButton{font-family:"Segoe UI Emoji"; font-size:12px; padding:2px 10px;}')

    def _prev_col(self, parent_layout, label, color):
        col = QWidget(); cv = QVBoxLayout(col); cv.setContentsMargins(4, 4, 4, 4); cv.setSpacing(2)
        lt = QLabel(label); lt.setAlignment(Qt.AlignCenter)
        lt.setStyleSheet(f"color:{color}; font-weight:700; background:transparent;")
        cv.addWidget(lt)
        img = QLabel("—"); img.setAlignment(Qt.AlignCenter); img.setStyleSheet("background:#1e293b; border-radius:4px;")
        img.setMinimumHeight(80); img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cv.addWidget(img, 1)
        res = QLabel(""); res.setAlignment(Qt.AlignCenter); res.setFont(QFont("Consolas", 9))
        res.setStyleSheet(f"color:{color}; background:transparent;")
        cv.addWidget(res); parent_layout.addWidget(col, 1)
        return img, res

    # ── pickers ──
    def _pick_model(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Modèle", "Model"), self.model.text() or "",
                                           tr("Modèles (*.safetensors *.pth *.onnx);;Tous (*.*)", "Models (*.safetensors *.pth *.onnx);;All (*.*)"))
        if p:
            self.model.setText(p); self.settings.set("ups_last_model", p)

    def _pick_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Image", self.inp.text() or "",
                                           "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif);;Tous (*.*)")
        if p:
            self.inp.setText(p); self.settings.set("ups_last_input", p)

    def _pick_input_folder(self):
        d = QFileDialog.getExistingDirectory(self, tr("Dossier source", "Source folder"), self.inp.text() or "")
        if d:
            self.inp.setText(d); self.settings.set("ups_last_input", d)

    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(
            self, tr("Vidéo", "Video"), self.inp.text() or "",
            tr("Vidéos (*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.wmv *.flv *.ts *.mts);;Tous (*.*)",
               "Videos (*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.wmv *.flv *.ts *.mts);;All (*.*)"))
        if p:
            self.inp.setText(p); self.settings.set("ups_last_input", p)

    def _pick_output_folder(self):
        d = QFileDialog.getExistingDirectory(self, tr("Dossier sortie", "Output folder"), self.out.text() or "")
        if d:
            self.out.setText(d); self.settings.set("ups_last_output", d)

    def _pick_temp_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("Dossier temporaire (frames)", "Temp folder (frames)"),
                                              self.temp_dir.text() or "")
        if d:
            self.temp_dir.setText(d); self.settings.set("ups_temp_dir", d)

    # ── toggles ──
    def _on_same_folder(self, on):
        self.settings.set("ups_same_folder", bool(on))
        self.out.setEnabled(not on); self.bo.setEnabled(not on)
        if on:
            self._sync_output()

    def _sync_output(self, *_):
        if not self.same_folder.isChecked():
            return
        inp = self.inp.text().strip()
        if inp:
            self.out.setText(inp if os.path.isdir(inp) else os.path.dirname(inp))

    def _on_serialize(self, on):
        self.serialize_start.setEnabled(bool(on))

    def _on_format(self, fmt):
        self.bitdepth.setEnabled(fmt in ("PNG", "TIFF"))
        if fmt not in ("PNG", "TIFF"):
            self.bitdepth.setCurrentText("8 bits")
        self.quality.setEnabled(fmt in ("JPEG", "WEBP"))

    def _on_colorfix(self, on):
        self.cf_method.setEnabled(bool(on)); self.cf_btn.setEnabled(bool(on))

    def _on_dandere(self, on):
        self.dandere_thr.setEnabled(bool(on)); self.dandere_blk.setEnabled(bool(on))

    def _open_colorfix(self):
        ColorFixDialog(self.settings, self._cf, self.cf_method.currentText(), self).exec()

    def _open_tf(self):
        TempFixDialog(self.settings, self._tf, self).exec()

    def _open_ud(self):
        UndistortDialog(self.settings, self._tf, self).exec()

    def _open_enc(self):
        EncodeOptionsDialog(self.settings, self._enc, self).exec()

    def _on_vid_format_change(self, idx):
        key = self._vid_format_keys[idx] if 0 <= idx < len(self._vid_format_keys) else "png8"
        self.settings.set("ups_vid_format", key)
        is_jpg = key == "jpg"
        self.vid_jpg_q_label.setVisible(is_jpg)
        self.vid_jpg_quality.setVisible(is_jpg)
        self.vid_jpg_q_value.setVisible(is_jpg)

    # ── run ──
    def _request_stop(self):
        if self._stop_event:
            self._stop_event.set()
        self.btn_stop.setEnabled(False); self.btn_stop.setText(tr("Arrêt...", "Stopping..."))

    def _run(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, tr("Upscale en cours", "Upscale in progress"), tr("Un upscale est déjà en cours. Cliquez sur Stop et attendez.", "An upscale is already running. Click Stop and wait."))
            return
        # absolute paths: the runners / persistent workers run with another working
        # directory, so a relative path validated here pointed elsewhere for them
        _abs = lambda p: os.path.abspath(p) if p else p
        model = _abs(self.model.text().strip()); inp = _abs(self.inp.text().strip())
        same = self.same_folder.isChecked(); subf = self.subfolder.isChecked()
        base_out = (inp if os.path.isdir(inp) else os.path.dirname(inp)) if same else _abs(self.out.text().strip())
        actual_out = os.path.join(base_out, "upscaled") if subf else base_out
        if not model or not os.path.isfile(model):
            QMessageBox.critical(self, tr("Erreur", "Error"), tr("Sélectionnez un modèle valide.", "Select a valid model.")); return
        if not inp or not os.path.exists(inp):
            QMessageBox.critical(self, tr("Erreur", "Error"), tr("Sélectionnez une source valide.", "Select a valid source.")); return
        if not base_out:
            QMessageBox.critical(self, tr("Erreur", "Error"), tr("Sélectionnez un dossier de sortie.", "Select an output folder.")); return

        # ── Entrée vidéo : fonctionne avec n'importe quel modèle. Les 4 archs séquence
        # (tscunet/temporalspan/temporalspanv2/tfdat — voir _SEQUENCE_ARCHS dans
        # src.core.quick_upscale) reçoivent un clip via _video_sequence() (fenêtre
        # glissante temporelle). Tout autre modèle (non-temporel) passe par
        # _video_nonsequence() : les frames sont extraites puis traitées indépendamment,
        # comme un dossier classique (pas de fenêtre glissante).
        video_input = os.path.isfile(inp) and os.path.splitext(inp)[1].lower() in _VIDEO_EXT
        video_arch = None
        if video_input or os.path.isdir(inp):
            try:
                from src.core.quick_upscale import _peek_arch_from_model_path
                video_arch = _peek_arch_from_model_path(model)
            except Exception:
                video_arch = None

        add_name = self.modelname.isChecked()
        model_tag = f"_{os.path.splitext(os.path.basename(model))[0]}" if add_name else ""
        out_suffix = model_tag if subf else f"{model_tag}_UP"
        ser_raw = self.serialize_start.text().strip(); ser_start = int(ser_raw) if ser_raw.isdigit() else 0

        # persist
        s = self.settings
        s.set("ups_last_model", model); s.set("ups_last_input", inp)
        if not same:
            s.set("ups_last_output", base_out)
        for k, w in (("ups_subfolder", subf), ("ups_modelname", add_name), ("ups_serialize", self.serialize.isChecked()),
                     ("ups_amp", self.amp.isChecked()), ("ups_colorfix", self.colorfix.isChecked()),
                     ("ups_persistent", self.persistent.isChecked()), ("ups_dandere", self.dandere.isChecked()),
                     ("ups_tempfix", self.tempfix.isChecked()), ("ups_undistort", self.undistort.isChecked()),
                     ("ups_keep_frames", self.keep_frames.isChecked()),
                     ("ups_reassemble_video", self.reassemble_video.isChecked())):
            s.set(k, w)
        s.set("ups_serialize_start", str(ser_start))
        s.set("ups_scale", self.scale.currentText()); s.set("ups_tile", self.tile.currentText())
        s.set("ups_tile_pad", self.tile_pad.currentText()); s.set("ups_format", self.fmt.currentText())
        s.set("ups_bitdepth", self.bitdepth.currentText()); s.set("ups_quality", self.quality.currentText())
        s.set("ups_dandere_threshold", self.dandere_thr.text().strip())
        s.set("ups_dandere_block_size", self.dandere_blk.text().strip())

        def _num(combo, default):
            t = combo.currentText()
            if t.startswith("Auto"):
                return -1 if default == "tile" else 0
            if t.startswith("0"):
                return 0
            try:
                return int(t)
            except ValueError:
                return 0
        scale = 0 if self.scale.currentText() == "Auto" else int(self.scale.currentText())
        # Multi-scale model (SpanPP / SpanC / MetaIGConv): ask which scale to use when Auto.
        if scale == 0 and os.path.isfile(model):
            ms = _detect_multiscale(model)
            if ms and len(ms) > 1:
                dlg = _MultiScaleDialog(ms, self)
                if dlg.exec() != QDialog.Accepted:
                    return  # closing the dialog cancels (it launched with the max scale)
                scale = dlg.chosen
        # A04: an explicit scale the model cannot produce was silently replaced by the
        # model's own factor. Say it before launching.
        if scale and os.path.isfile(model):
            try:
                from src.core.quick_upscale import peek_model_scale
                ms = _detect_multiscale(model)
                native = 0 if ms else peek_model_scale(model)
            except Exception:
                ms, native = None, 0
            if ms and scale not in ms:
                QMessageBox.warning(self, tr("Échelle", "Scale"),
                                    tr(f"Ce modèle multi-échelle supporte : {', '.join(f'x{v}' for v in ms)}.",
                                       f"This multi-scale model supports: {', '.join(f'x{v}' for v in ms)}."))
                return
            if native and native != scale:
                r = QMessageBox.question(self, tr("Échelle", "Scale"),
                                         tr(f"Le modèle est x{native} : l'échelle x{scale} demandée ne peut pas être produite.\n"
                                            f"Continuer en x{native} ?",
                                            f"The model is x{native}: the requested x{scale} cannot be produced.\n"
                                            f"Continue at x{native}?"))
                if r != QMessageBox.Yes:
                    return
                scale = native
        tile = -1 if self.tile.currentText() == "Auto" else (0 if self.tile.currentText().startswith("0") else int(self.tile.currentText()))
        if tile == -1:
            # Auto = VRAM-adaptive tiling (exact port of the CTk Auto path) — never
            # hardcode 256 here, that silently drops the adaptive behavior.
            try:
                import torch
                if torch.cuda.is_available():
                    free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                    tile = 512 if free_mb > 5000 else 256
                else:
                    tile = 256
            except Exception:
                tile = 256
        tile_pad = max(8, tile // 8) if self.tile_pad.currentText() == "Auto" else int(self.tile_pad.currentText())
        try:
            thr = float(self.dandere_thr.text().strip() or "0.010")
        except ValueError:
            thr = 0.010
        try:
            blk = max(4, int(self.dandere_blk.text().strip() or "14"))
        except ValueError:
            blk = 14
        bit_depth = 16 if self.bitdepth.currentText() == "16 bits" else 8
        try:
            quality = int(self.quality.currentText())
        except ValueError:
            quality = 95
        out_ext = _EXT_MAP.get(self.fmt.currentText().upper(), ".png")

        opts = {
            "model": model, "input": inp, "actual_out": actual_out, "out_suffix": out_suffix, "out_ext": out_ext,
            "scale": scale, "tile": tile, "tile_pad": tile_pad, "amp": self.amp.isChecked(),
            "out_format": self.fmt.currentText(), "bit_depth": bit_depth, "quality": quality,
            "serialize": self.serialize.isChecked(), "serialize_start": ser_start,
            "color_fix": self.cf_method.currentText() if self.colorfix.isChecked() else "none",
            "cf_wavelets": self._cf["wavelets"], "cf_radius": self._cf["radius"], "cf_fast": self._cf["fast"],
            "cf_strength": self._cf["strength"], "cf_planes": self._cf["planes"],
            "cf_device": self._cf["device"], "cf_ref": self._cf["ref"],
            "persistent": self.persistent.isChecked(), "dandere": self.dandere.isChecked(),
            "dandere_thr": thr, "dandere_blocks": blk,
            "tempfix": self.tempfix.isChecked(), "tf_mode": self._tf["tempfix_mode"],
            "tf_strength": self._tf["strength"], "tf_window": self._tf["window"], "tf_precision": self._tf["precision"],
            "undistort": self.undistort.isChecked(), "ud_mode": self._tf["undistort_mode"],
            "ud_strength": self._tf["undistort_strength"], "ud_window": self._tf["undistort_window"],
            "ud_precision": self._tf["undistort_precision"],
            "video_input": video_input, "video_arch": video_arch,
            "keep_frames": self.keep_frames.isChecked(),
            "reassemble_video": self.reassemble_video.isChecked(),
            "img_format": self._vid_format_keys[self.vid_format.currentIndex()],
            "jpg_quality": self.vid_jpg_quality.value(),
            "temp_dir": self.temp_dir.text().strip() or None,
            "vid_out_bit_depth": 16 if self.vid_out_bitdepth.currentIndex() == 1 else 8,
            "enc": dict(self._enc),
        }

        self._is_batch = os.path.isdir(inp) or video_input
        self.log.clear(); self.prog.setValue(0); self.prog_pct.setText("0%"); self.timing.setText("")
        self.btn_run.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_stop.setText("⏹ Stop")
        self.toolProgress.emit(0)
        self._stop_event = threading.Event()
        self.worker = _UpscaleWorker(opts, self._stop_event)
        self.worker.line.connect(self._log)
        self.worker.progress.connect(self._set_progress)
        self.worker.timing.connect(self.timing.setText)
        self.worker.preview.connect(self._update_preview)
        self.worker.done.connect(self._done)
        self.worker.start()

    def _log(self, msg):
        self.log.appendPlainText(msg)

    def _set_progress(self, v):
        p = int(v * 100); self.prog.setValue(p); self.prog_pct.setText(f"{p}%")
        self.toolProgress.emit(p)

    def _update_preview(self, in_path, out_path):
        self._last_preview = (in_path, out_path)
        try:
            self.settings.set("last_upscale_before", in_path)
            self.settings.set("last_upscale_after", out_path)
        except Exception:
            pass
        for path, lbl, res in ((in_path, self.prev_in, self.res_in), (out_path, self.prev_out, self.res_out)):
            pm = QPixmap(path)
            if not pm.isNull():
                res.setText(f"{pm.width()} × {pm.height()} px")
                lbl.setPixmap(pm.scaled(lbl.width(), lbl.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._last_preview:
            self._update_preview(*self._last_preview)

    def _done(self, ok, msg):
        self.log.appendPlainText(("✅ " if ok else "⚠ ") + msg)
        self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_stop.setText("⏹ Stop")
        self.worker = None; self._stop_event = None
        self.toolFinished.emit(bool(ok))
        try:
            from qt_ui.notify import notify, play_sound
            batch = getattr(self, "_is_batch", False)
            if ok:
                if batch:
                    notify(self.settings, tr(f"Batch : {msg}", f"Batch: {msg}"), tr("Upscale terminé", "Upscale finished"), "notif_win11_batch")
                    import re as _re
                    _m = _re.match(r"\s*(\d+)\s*/\s*(\d+)", str(msg))
                    if _m and int(_m.group(1)) < int(_m.group(2)):  # some images failed
                        play_sound(self.settings, "warning", "snd_warning")
                else:
                    notify(self.settings, tr("Upscale terminé", "Upscale finished"), msg, "notif_win11_upscale")
            else:
                notify(self.settings, tr("Upscale échoué", "Upscale failed"), msg, "notif_win11_errors")
                play_sound(self.settings, "error", "snd_error")
        except Exception:
            pass
