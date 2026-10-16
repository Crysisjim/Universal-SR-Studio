"""
gen_lq.py — Tools › Générateur Dataset LQ (PySide6) — faithful port of the v2.5.8
CTk create_page_generator: HQ/LQ paths, scale + resample method, 5 degradation tabs
(Basique / Couleur / Vidéo / Avancé / Pipeline, ~30 degradations with sliders),
Lancer la Génération + progress, and a Before/After preview (load test image, apply
degradations live, adjustable height). All degradation math lives in
src.core.lq_generator; settings persist per-widget.
"""
from __future__ import annotations

import io
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog, QFrame,
    QTabWidget, QScrollArea, QSlider, QSizePolicy,
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QFont, QPixmap

from qt_ui.i18n import tr

# resample-method name → PIL constant (resolved lazily so import never needs PIL).
# "DPID" isn't a PIL constant — _resize_downscale() in src/core/lq_generator.py
# special-cases that string and routes it to umzi2/pepedpid instead.
_METHODS = ["BICUBIC", "BILINEAR", "LANCZOS", "NEAREST", "BOX", "DPID"]


def _pil_method(name):
    if name == "DPID":
        return "dpid"
    from PIL import Image
    return {"BICUBIC": Image.BICUBIC, "BILINEAR": Image.BILINEAR, "LANCZOS": Image.LANCZOS,
            "NEAREST": Image.NEAREST, "BOX": Image.BOX}.get(name, Image.BICUBIC)


# ── degradation layout: tab → [(chk_key, label, [(slider_label, key, default, lo, hi, step)])] ──
# NOTE: dict keys (tab titles) are also used as QTabWidget labels directly — translated
# in place since nothing compares against these strings elsewhere in the file.
_TABS = {
    tr("🔧 Basique", "🔧 Basic"): [
        ("gen_blur", tr("Flou gaussien", "Gaussian Blur"), [("σ", "gen_blur_sigma", 1.0, 0.1, 8.0, 0.1)]),
        ("gen_noise", tr("Bruit gaussien", "Gaussian Noise"), [("σ", "gen_noise_sigma", 10, 1, 50, 1)]),
        ("gen_jpeg", tr("Compression JPEG", "JPEG Compression"), [(tr("Qualité", "Quality"), "gen_jpeg_q", 50, 1, 99, 1)]),
        ("gen_color_jitter", "Color Jitter (±10%)", []),
        ("gen_sharpen", tr("Netteté", "Sharpness"), [(tr("Intensité", "Intensity"), "gen_sharpen_amt", 1.5, 1.0, 3.0, 0.1)]),
    ],
    tr("🎨 Couleur", "🎨 Color"): [
        ("gen_posterize", tr("Postérisation", "Posterization"), [("bits", "gen_posterize_bits", 4, 2, 8, 1)]),
        ("gen_banding", "Banding", [(tr("Niveaux", "Levels"), "gen_banding_levels", 32, 8, 256, 8)]),
        ("gen_ca", tr("Aberration chrom.", "Chrom. Aberration"), [("Pixels", "gen_ca_shift", 2, 1, 10, 1)]),
        ("gen_disc_blur", tr("Flou disque (bokeh)", "Disc Blur (bokeh)"), [(tr("Rayon", "Radius"), "gen_disc_blur_r", 4.0, 1.0, 12.0, 0.5)]),
        ("gen_vignette", "Vignette", [(tr("Force", "Strength"), "gen_vignette_str", 0.4, 0.1, 1.0, 0.05)]),
        ("gen_halo", "Halo (wtp_halo)", [(tr("Force", "Strength"), "gen_halo_str", 0.3, 0.1, 1.0, 0.05),
                                         (tr("Rayon", "Radius"), "gen_halo_r", 6, 2, 15, 1)]),
        ("gen_saturation", "Saturation", [(tr("Facteur", "Factor"), "gen_sat_factor", 0.8, 0.0, 2.0, 0.05)]),
        ("gen_color_levels", tr("Niveaux couleur", "Color Levels"), [(tr("Seuil haut", "High Threshold"), "gen_col_high", 230, 200, 255, 5),
                                                 (tr("Seuil bas", "Low Threshold"), "gen_col_low", 10, 0, 40, 2)]),
        ("gen_quantize_depth", "Quantize depth", [("bits", "gen_qdepth_bits", 6, 4, 8, 1)]),
    ],
    tr("📺 Vidéo", "📺 Video"): [
        ("gen_chroma_sub", tr("Sous-éch. chroma", "Chroma Subsampling"), []),
        ("gen_aliasing", "Aliasing", [(tr("Force", "Strength"), "gen_aliasing_str", 0.75, 0.5, 1.0, 0.05)]),
        ("gen_interlace_weave", "Interlace weave", [(tr("Force", "Strength"), "gen_weave_str", 0.8, 0.1, 1.0, 0.05)]),
        ("gen_interlace_flicker", "Flicker", [("Amplitude", "gen_flicker_amp", 0.22, 0.01, 0.5, 0.01)]),
        ("gen_interlace_blend", "Field blend", [(tr("Mélange", "Blend"), "gen_blend_mix", 0.55, 0.1, 1.0, 0.05)]),
        ("gen_scanlines", "Scanlines CRT", [(tr("Période", "Period"), "gen_scanlines_period", 3, 2, 8, 1),
                                            (tr("Noirceur", "Darkness"), "gen_scanlines_dark", 0.35, 0.1, 0.6, 0.05)]),
        ("gen_vhs", "VHS / Analog", [(tr("Force", "Strength"), "gen_vhs_str", 0.3, 0.1, 1.0, 0.05)]),
        ("gen_screentone", "Screentone", [(tr("Taille", "Size"), "gen_screen_sz", 6, 3, 20, 1)]),
        ("gen_dithering", "Dithering", [(tr("Couleurs", "Colors"), "gen_dither_col", 4, 4, 64, 4)]),
        ("gen_sinusoidal", tr("Sinusoïdal", "Sinusoidal"), [(tr("Période", "Period"), "gen_sin_period", 300, 50, 600, 10),
                                          ("Amplitude", "gen_sin_alpha", 0.2, 0.05, 0.5, 0.05)]),
        ("gen_pixel_shift", "Pixel Shift", [("Pixels", "gen_pxsh_amt", 3, 1, 20, 1)]),
    ],
    tr("⚙ Avancé", "⚙ Advanced"): [
        ("gen_film_grain", "Film Grain", [("σ", "gen_grain_sigma", 0.08, 0.01, 0.3, 0.01),
                                          (tr("Taille", "Size"), "gen_grain_size", 1, 1, 3, 1)]),
        ("gen_oversharp", "Oversharpening", [(tr("Force", "Strength"), "gen_oversharp_amt", 1.4, 0.5, 3.0, 0.1)]),
        ("gen_motion_blur", "Motion Blur", [("Pixels", "gen_mb_px", 7, 3, 31, 2)]),
        ("gen_codec", tr("Artefacts codec (double JPEG)", "Codec artifacts (double JPEG)"), [("JPEG 1", "gen_codec_q1", 30, 1, 70, 1),
                                                ("JPEG 2", "gen_codec_q2", 60, 1, 99, 1)]),
        ("gen_salt_pepper", tr("Sel & Poivre", "Salt & Pepper"), [(tr("Quantité", "Amount"), "gen_sp_amt", 0.01, 0.001, 0.1, 0.001)]),
        ("gen_halation", tr("Halation (film)", "Halation (film)"), [(tr("Force", "Strength"), "gen_hal_str", 0.2, 0.05, 0.8, 0.05)]),
        ("gen_autocrop", "Auto-crop patches", [(tr("Taille", "Size"), "gen_crop_sz", 256, 64, 512, 32)]),
    ],
}


# Explanatory tooltip per degradation checkbox (beginner-friendly, one effect each).
_GEN_TIPS = {
    "gen_blur": tr("Flou gaussien : adoucit l'image (simule une mise au point ratée ou un upscale).\nσ ↑ = plus flou.",
                   "Gaussian blur: softens the image (simulates a missed focus or an upscale).\nσ ↑ = blurrier."),
    "gen_noise": tr("Bruit gaussien : grain aléatoire (simule capteur, haute sensibilité ISO).\nσ ↑ = plus de grain.",
                    "Gaussian noise: random grain (simulates a sensor, high ISO sensitivity).\nσ ↑ = more grain."),
    "gen_jpeg": tr("Compression JPEG : artefacts en blocs 8×8 + halos sur les contours.\nQualité ↓ = plus d'artefacts.",
                   "JPEG compression: 8×8 block artifacts + edge halos.\nQuality ↓ = more artifacts."),
    "gen_color_jitter": tr("Variation aléatoire ±10% de luminosité / contraste / saturation, ±5% de teinte.",
                           "Random ±10% variation of brightness / contrast / saturation, ±5% hue."),
    "gen_sharpen": tr("Sur-accentuation : renforce les contours (simule un filtre de netteté agressif).",
                      "Oversharpening: strengthens edges (simulates an aggressive sharpen filter)."),
    "gen_posterize": tr("Postérisation : réduit le nombre de couleurs (aplats, dégradés en escalier).\nbits ↓ = effet plus marqué.",
                        "Posterization: reduces the number of colors (flat areas, stepped gradients).\nbits ↓ = stronger effect."),
    "gen_banding": tr("Banding : bandes visibles dans les dégradés (manque de profondeur de couleur).",
                      "Banding: visible bands in gradients (lack of color depth)."),
    "gen_ca": tr("Aberration chromatique : franges colorées (rouge/cyan) décalées sur les bords.",
                "Chromatic aberration: colored fringes (red/cyan) offset at the edges."),
    "gen_disc_blur": tr("Flou de disque (bokeh) : flou circulaire, simule un objectif ouvert / hors-focus.",
                        "Disc blur (bokeh): circular blur, simulates a wide-open / out-of-focus lens."),
    "gen_vignette": tr("Vignette : assombrissement progressif des coins de l'image.",
                       "Vignette: progressive darkening of the image corners."),
    "gen_halo": tr("Halo (wtp_halo) : auréole claire autour des zones contrastées (sur-traitement).",
                  "Halo (wtp_halo): light halo around high-contrast areas (over-processing)."),
    "gen_saturation": tr("Saturation : <1 désature (vers gris), >1 sature. Simule réglages couleur.",
                         "Saturation: <1 desaturates (toward gray), >1 saturates. Simulates color settings."),
    "gen_color_levels": tr("Niveaux de couleur : compresse la plage tonale (noirs relevés, blancs écrêtés).",
                           "Color levels: compresses the tonal range (lifted blacks, clipped whites)."),
    "gen_quantize_depth": tr("Réduction de profondeur de bits par canal (banding couleur plus fort).",
                             "Reduces bit depth per channel (stronger color banding)."),
    "gen_chroma_sub": tr("Sous-échantillonnage chroma (4:2:0) : perte de détail couleur (comme la vidéo).",
                         "Chroma subsampling (4:2:0): loss of color detail (like video)."),
    "gen_aliasing": tr("Aliasing : crénelage / escaliers sur les diagonales (anti-aliasing insuffisant).",
                       "Aliasing: jagged/stair-stepped diagonals (insufficient anti-aliasing)."),
    "gen_interlace_weave": tr("Entrelacement weave : peigne sur les zones en mouvement (vidéo entrelacée).",
                              "Interlace weave: combing on moving areas (interlaced video)."),
    "gen_interlace_flicker": tr("Scintillement entre trames paires/impaires (vieux signal TV).",
                                "Flicker between even/odd fields (old TV signal)."),
    "gen_interlace_blend": tr("Mélange des deux trames (déentrelacement par fusion → léger ghosting).",
                              "Blends both fields (deinterlacing by merging → slight ghosting)."),
    "gen_scanlines": tr("Scanlines CRT : lignes sombres horizontales (effet écran cathodique).",
                        "CRT scanlines: dark horizontal lines (cathode-ray screen effect)."),
    "gen_vhs": tr("VHS / analogique : bruit, traînées, instabilité de couleur (cassette).",
                 "VHS / analog: noise, trails, color instability (tape)."),
    "gen_screentone": tr("Screentone / trame d'impression (points réguliers, style manga/comic).",
                         "Screentone / print halftone (regular dots, manga/comic style)."),
    "gen_dithering": tr("Tramage : simule plus de couleurs avec un motif de points (palette réduite).",
                        "Dithering: simulates more colors with a dot pattern (reduced palette)."),
    "gen_sinusoidal": tr("Ondulation sinusoïdale : vagues de luminosité (style wtp), sans déformation géométrique.",
                         "Sinusoidal wave: brightness waves (wtp style), no geometric distortion."),
    "gen_pixel_shift": tr("Décalage de pixels : léger déplacement aléatoire (désalignement capteur).",
                          "Pixel shift: slight random displacement (sensor misalignment)."),
    "gen_film_grain": tr("Grain argentique film : bruit organique structuré (pellicule).",
                         "Film grain: structured organic noise (film stock)."),
    "gen_oversharp": tr("Sur-netteté extrême : halos blancs marqués autour des contours.",
                        "Extreme oversharpening: pronounced white halos around edges."),
    "gen_motion_blur": tr("Flou de mouvement directionnel (sujet ou caméra qui bouge).",
                          "Directional motion blur (moving subject or camera)."),
    "gen_codec": tr("Artefacts codec : deux compressions JPEG successives (approximation des blocs + flou "
                    "d'un flux vidéo ; ce n'est pas un vrai encodage H.264).",
                    "Codec artifacts: two successive JPEG compressions (approximates the blocks + blur of "
                    "a video stream; not a real H.264 encode)."),
    "gen_salt_pepper": tr("Sel & poivre : pixels blancs/noirs isolés (bruit impulsionnel).",
                          "Salt & pepper: isolated white/black pixels (impulse noise)."),
    "gen_halation": tr("Halation film : diffusion lumineuse autour des hautes lumières (glow rougeâtre).",
                       "Film halation: light diffusion around highlights (reddish glow)."),
    "gen_autocrop": tr("Auto-crop : découpe les images en patchs carrés (taille réglable) au lieu de les dégrader entières.",
                       "Auto-crop: splits images into square patches (adjustable size) instead of degrading them whole."),
}


class _GenSlider(QWidget):
    """Compact slider + value box (int or float depending on step)."""
    changed = Signal()

    def __init__(self, lo, hi, step, init):
        super().__init__()
        self.lo, self.hi, self.step = lo, hi, step
        self.is_int = isinstance(step, int) or float(step).is_integer()
        self.n = max(1, round((hi - lo) / step))
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(3)
        self.sld = QSlider(Qt.Horizontal); self.sld.setRange(0, self.n); self.sld.setFixedWidth(90)
        self.box = QLineEdit(); self.box.setFixedWidth(46); self.box.setAlignment(Qt.AlignCenter)
        h.addWidget(self.sld); h.addWidget(self.box)
        self.set_value(init)
        self.sld.valueChanged.connect(self._from_slider)
        self.box.editingFinished.connect(self._from_box)

    def _fmt(self, v):
        return str(int(round(v))) if self.is_int else f"{round(v, 4):.4g}"

    def value(self):
        try:
            v = float(self.box.text())
        except ValueError:
            v = self.lo
        return int(round(v)) if self.is_int else v

    def set_value(self, v):
        v = max(self.lo, min(self.hi, float(v)))
        self.box.setText(self._fmt(v))
        self.sld.blockSignals(True); self.sld.setValue(round((v - self.lo) / self.step)); self.sld.blockSignals(False)

    def _from_slider(self, _):
        v = self.lo + self.sld.value() * self.step
        self.box.setText(self._fmt(v)); self.changed.emit()

    def _from_box(self):
        self.set_value(self.value()); self.changed.emit()


class _PreviewWorker(QThread):
    done = Signal(bytes)
    fail = Signal(str)

    def __init__(self, pil_img, opts):
        super().__init__(); self.pil, self.opts = pil_img, opts

    def run(self):
        try:
            from src.core.lq_generator import degrade_one
            out = degrade_one(self.pil, self.opts)
            buf = io.BytesIO(); out.save(buf, format="PNG")
            self.done.emit(buf.getvalue())
        except Exception as e:
            self.fail.emit(str(e))


class _BatchWorker(QThread):
    line = Signal(str)
    progress = Signal(float, int, int)
    preview = Signal(bytes, bytes)
    done = Signal(int, int)

    def __init__(self, hq, lq, opts):
        super().__init__(); self.hq, self.lq, self.opts = hq, lq, opts

    def run(self):
        try:
            from src.core.lq_generator import process_folder

            def _prev(hq_snap, lq_snap):
                b1 = io.BytesIO(); hq_snap.save(b1, format="PNG")
                b2 = io.BytesIO(); lq_snap.save(b2, format="PNG")
                self.preview.emit(b1.getvalue(), b2.getvalue())

            n, total = process_folder(self.hq, self.lq, self.opts,
                                      progress=lambda f, d, t: self.progress.emit(f, d, t),
                                      preview=_prev, log=self.line.emit)
            self.done.emit(n, total)
        except Exception as e:
            self.line.emit(f"❌ {e}"); self.done.emit(0, 0)


class _VideoBatchWorker(QThread):
    """Runs src.core.lq_generator.process_video() off-thread: extract frames (ffmpeg) →
    reuse process_folder() for degradation (Train, then Val if requested)."""
    line = Signal(str)
    progress = Signal(str, object, object, object)  # phase, frac, done, total
    preview = Signal(bytes, bytes)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, video_path, hq_dir, lq_dir, opts, prefix, fps, val_count, val_hq_dir, val_lq_dir,
                 val_position="end", img_format="png8", jpg_quality=95):
        super().__init__()
        self.video_path, self.hq_dir, self.lq_dir, self.opts = video_path, hq_dir, lq_dir, opts
        self.prefix, self.fps = prefix, fps
        self.val_count, self.val_hq_dir, self.val_lq_dir = val_count, val_hq_dir, val_lq_dir
        self.val_position = val_position
        self.img_format, self.jpg_quality = img_format, jpg_quality

    def run(self):
        try:
            from src.core.lq_generator import process_video

            def _prev(hq_snap, lq_snap):
                b1 = io.BytesIO(); hq_snap.save(b1, format="PNG")
                b2 = io.BytesIO(); lq_snap.save(b2, format="PNG")
                self.preview.emit(b1.getvalue(), b2.getvalue())

            def _prog(phase, frac, done, total):
                self.progress.emit(phase, frac, done, total)

            result = process_video(
                self.video_path, self.hq_dir, self.lq_dir, self.opts,
                prefix=self.prefix, fps=self.fps,
                val_count=self.val_count, val_hq_dir=self.val_hq_dir, val_lq_dir=self.val_lq_dir,
                val_position=self.val_position,
                progress=_prog, preview=_prev, log=self.line.emit,
                img_format=self.img_format, jpg_quality=self.jpg_quality,
            )
            self.done.emit(result)
        except Exception as e:
            self.line.emit(f"❌ {e}")
            self.failed.emit(str(e))


class GenLQPage(QWidget):
    # Sidebar-badge hooks (qt_ui/tabs/tools_tab.py) — see QuickUpscalePage for
    # the pattern. toolBusy covers the image-extraction phase of the video path
    # (ffmpeg, no known %); toolProgress covers everything with a real fraction.
    toolProgress = Signal(int)
    toolBusy = Signal()
    toolFinished = Signal(bool)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._w = {}            # gen_* key → widget
        self._prev_pil = None   # loaded test image (PIL)
        self._worker = None
        self._prev_h = int(settings.get("gen_prev_h", 280) or 280)

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget(); scroll.setWidget(inner); outer.addWidget(scroll)
        v = QVBoxLayout(inner); v.setContentsMargins(20, 16, 20, 16); v.setSpacing(6)

        # header + GPU panel
        hdr = QHBoxLayout(); v.addLayout(hdr)
        lc = QVBoxLayout(); lc.setSpacing(0)
        t = QLabel(tr("Générateur Dataset LQ", "LQ Dataset Generator")); t.setObjectName("title"); lc.addWidget(t)
        sub = QLabel(tr("Générez des données basse qualité avec toutes les dégradations connues.", "Generate low-quality data with all known degradations."))
        sub.setStyleSheet("color:#888;"); lc.addWidget(sub)
        hdr.addLayout(lc); hdr.addStretch(1)
        try:
            from qt_ui.tabs.tools_pages.quick_upscale import _GpuMiniPanel
            self.gpu_panel = _GpuMiniPanel(); hdr.addWidget(self.gpu_panel, 0, Qt.AlignTop | Qt.AlignRight)
        except Exception:
            self.gpu_panel = None

        # mode selector — Images (existing per-image pipeline, unchanged) vs Vidéo
        # (extract frames with ffmpeg, then reuse the exact same degradation pipeline).
        mode_row = QHBoxLayout(); v.addLayout(mode_row)
        mode_row.addWidget(QLabel(tr("Mode :", "Mode:")))
        self.mode = QComboBox(); self.mode.addItems([tr("🖼 Images", "🖼 Images"), tr("🎬 Vidéo", "🎬 Video")])
        self.mode.setCurrentIndex(1 if str(settings.get("gen_mode_idx", "0")) == "1" else 0)
        self.mode.setFixedWidth(140)
        self.mode.setToolTip(tr("Images : dégrade un dossier d'images HQ existant (mode historique).\n"
                             "Vidéo : extrait les frames d'un fichier vidéo (ffmpeg) puis applique les mêmes dégradations.", "Images: degrades an existing HQ image folder (legacy mode).\nVideo: extracts frames from a video file (ffmpeg) then applies the same degradations."))
        self.mode.currentIndexChanged.connect(self._on_mode_change)
        mode_row.addWidget(self.mode); mode_row.addStretch(1)

        # paths — Images mode
        self.img_paths_box = QWidget(); img_v = QVBoxLayout(self.img_paths_box)
        img_v.setContentsMargins(0, 0, 0, 0); img_v.setSpacing(4)
        v.addWidget(self.img_paths_box)
        self.hq = self._path_row(img_v, "Source (HQ) :", "gen_hq")
        self.hq.setToolTip(tr("Dossier d'images haute qualité (la référence). Elles ne sont jamais modifiées.", "High-quality image folder (the reference). They are never modified."))
        self.lq = self._path_row(img_v, "Destination (LQ) :", "gen_lq")
        self.lq.setToolTip(tr("Dossier où écrire les images dégradées (basse qualité) générées.", "Folder to write the generated degraded (low-quality) images."))

        # paths — Vidéo mode
        self.vid_paths_box = QWidget(); vid_v = QVBoxLayout(self.vid_paths_box)
        vid_v.setContentsMargins(0, 0, 0, 0); vid_v.setSpacing(4)
        v.addWidget(self.vid_paths_box)

        self.vid_file = self._file_row(vid_v, tr("Fichier vidéo :", "Video file:"), "gen_vid_path")
        self.vid_file.setToolTip(tr("Fichier vidéo source. Toutes ses frames (ou un sous-échantillon, voir FPS) "
                                 "seront extraites comme images HQ.", "Source video file. All its frames (or a subsample, see FPS) will be extracted as HQ images."))

        prow2 = QHBoxLayout(); vid_v.addLayout(prow2)
        prow2.addWidget(QLabel(tr("Préfixe scène :", "Scene prefix:")))
        self.vid_prefix = QLineEdit(settings.get("gen_vid_prefix", "") or "")
        self.vid_prefix.setPlaceholderText(tr("(auto = nom du fichier)", "(auto = filename)"))
        self.vid_prefix.setToolTip(tr(
            "Préfixe utilisé pour nommer les frames : <préfixe>_00001.png, <préfixe>_00002.png…\n"
            "Convention requise par PairedVideoDataset : le préfixe ne doit PAS contenir de « _ » "
            "(tout ce qui suit le premier « _ » dans le nom de fichier est ignoré).\n"
            "Laisser vide = nom du fichier vidéo, nettoyé automatiquement.",
            "Prefix used to name frames: <prefix>_00001.png, <prefix>_00002.png…\n"
            "Convention required by PairedVideoDataset: the prefix must NOT contain \"_\" "
            "(everything after the first \"_\" in the filename is ignored).\n"
            "Leave empty = the video's filename, auto-cleaned."))
        self.vid_prefix.textChanged.connect(lambda x: settings.set("gen_vid_prefix", x))
        prow2.addWidget(self.vid_prefix, 1)
        prow2.addSpacing(12)
        prow2.addWidget(QLabel(tr("FPS extraction :", "Extraction FPS:")))
        self.vid_fps = QLineEdit(str(settings.get("gen_vid_fps", "0") or "0")); self.vid_fps.setFixedWidth(50)
        self.vid_fps.setToolTip(tr("0 = extrait TOUTES les frames de la vidéo (recommandé pour un dataset vidéo cohérent).\n"
                               "Sinon, sous-échantillonne à N images/seconde.", "0 = extracts ALL frames from the video (recommended for a coherent video dataset).\nOtherwise, subsamples to N frames/second."))
        self.vid_fps.textChanged.connect(lambda x: settings.set("gen_vid_fps", x))
        prow2.addWidget(self.vid_fps); prow2.addStretch(1)

        frow = QHBoxLayout(); vid_v.addLayout(frow)
        frow.addWidget(QLabel(tr("Format extraction :", "Extraction format:")))
        self.vid_format = QComboBox()
        # Keys map 1:1 to src.core.lq_generator.EXTRACT_FORMATS. TIFF8/TIFF16 are
        # intentionally NOT offered here: PairedVideoDataset (traiNNer, external repo)
        # filters frames with f.lower().endswith((".png",".jpg",".jpeg")) — verified by
        # direct read of traiNNer/data/paired_video_dataset.py — so any .tiff HQ frame
        # would be silently invisible to training. No workaround is offered since a
        # dataset that "trains" on zero frames with no error is the worst failure mode
        # for this tool. PNG16 stays: verified by direct test (extracted a real 16-bit
        # PNG, ran it through traiNNer's own PairedVideoDataset.__getitem__) that
        # vipsimfrompath()+img2tensor()/tensor2float32() (torch.iinfo-based normalization)
        # correctly propagate the full 16-bit range to a [0,1] float32 GT tensor — the
        # extra precision is NOT lost for the actual training consumer. (This app's own
        # internal PIL-based tools — degradation preview, Quick Upscale, the *_runner.py
        # upscale runners — DO truncate a 16-bit file to 8-bit on load; that limitation is
        # real but irrelevant to what PairedVideoDataset reads for training.)
        self._vid_format_keys = ["png8", "png16", "jpg"]
        self.vid_format.addItems([
            tr("PNG 8-bit (par défaut)", "PNG 8-bit (default)"),
            tr("PNG 16-bit", "PNG 16-bit"),
            "JPEG",
        ])
        _fmt_saved = str(settings.get("gen_vid_format", "png8") or "png8")
        self.vid_format.setCurrentIndex(
            self._vid_format_keys.index(_fmt_saved) if _fmt_saved in self._vid_format_keys else 0)
        self.vid_format.setToolTip(tr(
            "Format des frames HQ extraites de la vidéo (les frames LQ restent toujours en PNG 8-bit, "
            "inchangé).\nPNG 8-bit : comportement historique.\nPNG 16-bit : préserve la pleine précision "
            "de la vidéo source sur le disque et jusqu'au tenseur GT de PairedVideoDataset (vérifié) — "
            "mais les outils internes de cette appli (aperçu, Quick Upscale, runners) tronquent à 8-bit "
            "s'ils rouvrent ce fichier.\nJPEG : plus compact, avec perte (réglez la qualité ci-contre).",
            "Format for the HQ frames extracted from the video (LQ frames always stay 8-bit PNG, "
            "unchanged).\nPNG 8-bit: previous/default behavior.\nPNG 16-bit: preserves the source "
            "video's full precision on disk and all the way to PairedVideoDataset's GT tensor "
            "(verified) — but this app's own internal tools (preview, Quick Upscale, runners) "
            "truncate to 8-bit if they reopen this file.\nJPEG: smaller, lossy (set quality alongside)."))
        self.vid_format.setFixedWidth(160)
        self.vid_format.currentIndexChanged.connect(self._on_vid_format_change)
        frow.addWidget(self.vid_format)

        frow.addSpacing(12)
        self.vid_jpg_q_label = QLabel(tr("Qualité JPEG :", "JPEG quality:"))
        frow.addWidget(self.vid_jpg_q_label)
        self.vid_jpg_quality = QSlider(Qt.Horizontal)
        self.vid_jpg_quality.setRange(1, 100)
        self.vid_jpg_quality.setValue(int(settings.get("gen_vid_jpg_quality", 95) or 95))
        self.vid_jpg_quality.setFixedWidth(120)
        self.vid_jpg_quality.setToolTip(tr("1 = pire qualité / plus petit fichier, 100 = meilleure qualité.",
                                            "1 = worst quality / smallest file, 100 = best quality."))
        self.vid_jpg_q_value = QLabel(str(self.vid_jpg_quality.value())); self.vid_jpg_q_value.setFixedWidth(28)
        self.vid_jpg_quality.valueChanged.connect(
            lambda v: (self.vid_jpg_q_value.setText(str(v)), settings.set("gen_vid_jpg_quality", v)))
        frow.addWidget(self.vid_jpg_quality)
        frow.addWidget(self.vid_jpg_q_value)
        frow.addStretch(1)
        self._on_vid_format_change(self.vid_format.currentIndex())

        self.vid_hq = self._path_row(vid_v, tr("Frames HQ (sortie) :", "HQ frames (output):"), "gen_vid_hq")
        self.vid_hq.setToolTip(tr("Dossier où écrire les frames HQ extraites de la vidéo (créé si absent).", "Folder to write the HQ frames extracted from the video (created if missing)."))
        self.vid_lq = self._path_row(vid_v, tr("Frames LQ (sortie) :", "LQ frames (output):"), "gen_vid_lq")
        self.vid_lq.setToolTip(tr("Dossier où écrire les frames LQ dégradées (créé si absent).", "Folder to write the degraded LQ frames (created if missing)."))

        vrow = QHBoxLayout(); vid_v.addLayout(vrow)
        self.vid_val_chk = QCheckBox(tr("Générer un set de validation", "Generate a validation set"))
        self.vid_val_chk.setChecked(self._b("gen_vid_val_enabled", False))
        self.vid_val_chk.setToolTip(tr("Réserve les X dernières frames extraites (portion de fin de vidéo, cohérente "
                                    "temporellement) pour la validation, séparément du set d'entraînement.", "Reserves the last X extracted frames (a temporally coherent tail portion of the video) for validation, separate from the training set."))
        self.vid_val_chk.toggled.connect(self._on_vid_val_toggle)
        vrow.addWidget(self.vid_val_chk)
        vrow.addSpacing(8)
        vrow.addWidget(QLabel(tr("X images :", "X images:")))
        self.vid_val_count = QLineEdit(str(settings.get("gen_vid_val_count", "20") or "20")); self.vid_val_count.setFixedWidth(50)
        self.vid_val_count.setToolTip(tr("Nombre de frames (en fin de vidéo) réservées à la validation.", "Number of frames (from the end of the video) reserved for validation."))
        self.vid_val_count.textChanged.connect(lambda x: settings.set("gen_vid_val_count", x))
        vrow.addWidget(self.vid_val_count)
        vrow.addSpacing(12)
        vrow.addWidget(QLabel(tr("Position :", "Position:")))
        self.vid_val_pos = QComboBox()
        self.vid_val_pos.addItems([tr("Fin", "End"), tr("Début", "Start"), tr("Aléatoire", "Random")])
        _pos_saved = str(settings.get("gen_vid_val_pos", "end") or "end")
        self.vid_val_pos.setCurrentIndex({"end": 0, "start": 1, "random": 2}.get(_pos_saved, 0))
        self.vid_val_pos.setFixedWidth(100)
        self.vid_val_pos.setToolTip(tr(
            "Où prélever le bloc de frames de validation dans la vidéo (toujours un bloc CONTIGU — "
            "PairedVideoDataset a besoin de vraies frames consécutives, pas d'images éparpillées) :\n"
            "Fin : dernières frames (défaut) — risque : générique de fin, écran noir.\n"
            "Début : premières frames — risque : logo/intro.\n"
            "Aléatoire : bloc pris à une position tirée au hasard dans toute la vidéo — évite "
            "systématiquement intro/outro non représentatifs. Contrepartie : contrairement à Fin/Début "
            "(coupe nette en bord de séquence), retirer un bloc du milieu laisse une « couture » dans "
            "les frames d'entraînement restantes (les deux frames de part et d'autre du bloc retiré "
            "deviennent \"adjacentes\" pour le dataset) — quelques clips d'entraînement near cette "
            "couture mélangeront deux moments sans rapport. Négligeable sur un épisode entier, mais "
            "réel.",
            "Where to pull the validation frame block from in the video (always a CONTIGUOUS block — "
            "PairedVideoDataset needs real consecutive frames, not scattered stills):\n"
            "End: last frames (default) — risk: end credits, black screen.\n"
            "Start: first frames — risk: logo/intro.\n"
            "Random: block taken from a randomly-picked position anywhere in the video — avoids "
            "systematically landing on unrepresentative intro/outro content. Trade-off: unlike End/Start "
            "(a clean cut at a sequence boundary), removing a middle block leaves a \"seam\" in the "
            "remaining Train frames (the two frames flanking the removed block become \"adjacent\" to "
            "the dataset) — a few Train clips near that seam will splice two unrelated moments. "
            "Negligible on a full episode, but real."))
        self.vid_val_pos.currentIndexChanged.connect(
            lambda i: settings.set("gen_vid_val_pos", ["end", "start", "random"][i]))
        vrow.addWidget(self.vid_val_pos)
        vrow.addStretch(1)

        self.vid_val_hq = self._path_row(vid_v, tr("Val HQ (sortie) :", "Val HQ (output):"), "gen_vid_val_hq")
        self.vid_val_hq.setToolTip(tr("Dossier de validation HQ. Laisser vide = <Frames HQ>_val.", "Validation HQ folder. Leave empty = <HQ frames>_val."))
        self.vid_val_lq = self._path_row(vid_v, tr("Val LQ (sortie) :", "Val LQ (output):"), "gen_vid_val_lq")
        self.vid_val_lq.setToolTip(tr("Dossier de validation LQ. Laisser vide = <Frames LQ>_val.", "Validation LQ folder. Leave empty = <LQ frames>_val."))

        self.vid_ffmpeg_status = QLabel(""); self.vid_ffmpeg_status.setStyleSheet("color:#888;")
        vid_v.addWidget(self.vid_ffmpeg_status)

        self.vid_log = QPlainTextEdit(); self.vid_log.setReadOnly(True); self.vid_log.setMaximumBlockCount(3000)
        self.vid_log.setFont(QFont("Consolas", 9)); self.vid_log.setFixedHeight(100)
        vid_v.addWidget(self.vid_log)

        self._sync_vid_val_visibility()
        self._on_mode_change()

        # scale + method
        sm = QHBoxLayout(); v.addLayout(sm)
        sm.addWidget(QLabel("Scale :"))
        self.scale = QComboBox(); self.scale.addItems(["1", "2", "3", "4", "6", "8"])
        self.scale.setCurrentText(str(settings.get("gen_scale", "4"))); self.scale.setFixedWidth(60)
        self.scale.setToolTip(tr("Facteur de réduction HQ→LQ. Scale 4 = la LQ fait 1/4 de la taille\n"
                              "(le modèle apprendra à la ré-agrandir ×4). Scale 1 = même taille (dégradation seule).", "HQ→LQ reduction factor. Scale 4 = the LQ is 1/4 the size\n(the model will learn to upscale it ×4). Scale 1 = same size (degradation only)."))
        self.scale.currentTextChanged.connect(lambda x: settings.set("gen_scale", x))
        sm.addWidget(self.scale); sm.addSpacing(18)
        sm.addWidget(QLabel(tr("Méthode :", "Method:")))
        self.method = QComboBox(); self.method.addItems(_METHODS)
        self.method.setCurrentText(str(settings.get("gen_method", "BICUBIC"))); self.method.setFixedWidth(110)
        self.method.setToolTip(tr("Algorithme de redimensionnement HQ→LQ.\n"
                               "BICUBIC = doux et naturel (défaut) · LANCZOS = plus net · "
                               "BILINEAR = plus flou · NEAREST = pixelisé (pixel art) · BOX = moyenne.", "HQ→LQ resampling algorithm.\nBICUBIC = soft and natural (default) · LANCZOS = sharper · BILINEAR = blurrier · NEAREST = pixelated (pixel art) · BOX = average."))
        self.method.currentTextChanged.connect(lambda x: settings.set("gen_method", x))
        sm.addWidget(self.method); sm.addSpacing(12)

        # DPID lambda — only meaningful when Méthode = DPID (detail-preserving downscale,
        # umzi2/pepedpid). 0.0 = max lissage, 1.0 = max préservation détail, 0.5 recommandé.
        self.dpid_lbl = QLabel(tr("λ DPID :", "DPID λ:"))
        self.dpid_sld = QSlider(Qt.Horizontal); self.dpid_sld.setRange(0, 100); self.dpid_sld.setFixedWidth(90)
        self.dpid_sld.setValue(int(round(float(settings.get("gen_dpid_lambda", 0.5)) * 100)))
        self.dpid_val = QLabel(f"{self.dpid_sld.value() / 100:.2f}")
        _dpid_tip = tr("Coefficient λ du downscale DPID (umzi2/pepedpid, Rust).\n"
                      "0.0 = lissage maximal (image douce) · 1.0 = préservation maximale du détail (image nette).\n"
                      "0.5 recommandé (compromis). Actif seulement si Méthode = DPID.",
                      "λ coefficient for the DPID downscale (umzi2/pepedpid, Rust-backed).\n"
                      "0.0 = max smoothing (soft image) · 1.0 = max detail preservation (sharp image).\n"
                      "0.5 recommended (balance). Only used when Method = DPID.")
        self.dpid_lbl.setToolTip(_dpid_tip); self.dpid_sld.setToolTip(_dpid_tip)

        def _on_dpid_lambda(v):
            self.dpid_val.setText(f"{v / 100:.2f}")
            settings.set("gen_dpid_lambda", v / 100)
        self.dpid_sld.valueChanged.connect(_on_dpid_lambda)

        def _sync_dpid_visibility(method_name):
            active = (method_name == "DPID")
            for w in (self.dpid_lbl, self.dpid_sld, self.dpid_val):
                w.setVisible(active)
        self.method.currentTextChanged.connect(_sync_dpid_visibility)
        _sync_dpid_visibility(self.method.currentText())

        sm.addWidget(self.dpid_lbl); sm.addWidget(self.dpid_sld); sm.addWidget(self.dpid_val)
        sm.addStretch(1)

        # tabs
        tabs = QTabWidget(); v.addWidget(tabs)
        for tab_name, rows in _TABS.items():
            tabs.addTab(self._build_tab(rows), tab_name)
        tabs.addTab(self._build_pipeline_tab(), "🎲 Pipeline")

        # run
        self.btn = QPushButton(tr("▶  Lancer la Génération", "▶  Run Generation"))
        self.btn.setToolTip(tr("Applique les dégradations cochées à toutes les images du dossier Source (HQ)\n"
                            "et écrit les versions basse qualité dans le dossier Destination (LQ).\n"
                            "Idéal pour fabriquer une paire HQ/LQ d'entraînement en mode 'paired'.", "Applies the checked degradations to all images in the Source (HQ) folder\nand writes the low-quality versions into the Destination (LQ) folder.\nIdeal to build an HQ/LQ training pair in 'paired' mode."))
        self.btn.setStyleSheet(
            "QPushButton{background:#E67E22; color:#fff; border:none; border-radius:6px; padding:9px; font-weight:700; font-size:14px;}"
            "QPushButton:hover{background:#f0871f;}"
            "QPushButton:pressed{background:#c4661a; padding-top:11px;}"
            "QPushButton:disabled{background:#5a4636; color:#caa;}")
        self.btn.clicked.connect(self._run); v.addWidget(self.btn)
        self.prog = QProgressBar(); self.prog.setRange(0, 100); v.addWidget(self.prog)
        self.status = QLabel(tr("En attente…", "Waiting…")); self.status.setAlignment(Qt.AlignCenter); self.status.setStyleSheet("color:#888;")
        v.addWidget(self.status)

        # preview
        ph = QLabel(tr("Aperçu Avant / Après", "Before / After Preview")); ph.setObjectName("title"); ph.setStyleSheet("font-size:18px; font-weight:800; color:#3b8ed0;")
        v.addWidget(ph)
        prow = QHBoxLayout(); v.addLayout(prow)
        b_load = QPushButton(tr("📂 Charger image de test", "📂 Load test image")); b_load.setStyleSheet('background:#2980b9; color:#fff; border:none; border-radius:6px; padding:6px 12px; font-family:"Segoe UI Emoji"; font-size:13px;')
        b_load.setToolTip(tr("Charge une image pour tester les dégradations sans lancer toute la génération.", "Loads an image to test degradations without running the full generation."))
        b_load.clicked.connect(self._load_preview); prow.addWidget(b_load)
        b_apply = QPushButton(tr("⚡ Appliquer dégradations", "⚡ Apply degradations")); b_apply.setStyleSheet('background:#16a085; color:#fff; border:none; border-radius:6px; padding:6px 12px; font-family:"Segoe UI Emoji"; font-size:13px;')
        b_apply.setToolTip(tr("Applique les dégradations cochées à l'image de test et montre le résultat\n"
                           "Avant/Après. Recommence pour voir l'effet du hasard (probabilités, bruit).", "Applies the checked degradations to the test image and shows the\nBefore/After result. Repeat to see the effect of randomness (probabilities, noise)."))
        b_apply.clicked.connect(self._apply_preview); prow.addWidget(b_apply)
        self.prev_status = QLabel(""); self.prev_status.setStyleSheet("color:#888;"); prow.addWidget(self.prev_status, 1)

        hrow = QHBoxLayout(); v.addLayout(hrow)
        hrow.addWidget(QLabel(tr("Hauteur aperçu :", "Preview height:")))
        self.h_lbl = QLabel(f"{self._prev_h} px"); self.h_lbl.setFixedWidth(60); hrow.addWidget(self.h_lbl)
        self.h_sld = QSlider(Qt.Horizontal); self.h_sld.setRange(120, 700); self.h_sld.setValue(self._prev_h); self.h_sld.setFixedWidth(220)
        self.h_sld.valueChanged.connect(self._on_height); hrow.addWidget(self.h_sld); hrow.addStretch(1)

        panels = QHBoxLayout(); v.addLayout(panels)
        self.prev_before, self._before_box = self._preview_panel(tr("Avant (HQ original)", "Before (original HQ)"), "#3498db", is_before=True)
        self.prev_after, self._after_box = self._preview_panel(tr("Après (LQ dégradé)", "After (degraded LQ)"), "#e67e22", is_before=False)
        panels.addWidget(self._before_box, 1); panels.addWidget(self._after_box, 1)

    # ── builders ──────────────────────────────────────────────────────────────
    def _path_row(self, layout, label, key):
        r = QHBoxLayout(); r.setSpacing(6)
        lb = QLabel(label); lb.setFixedWidth(150); r.addWidget(lb)
        e = QLineEdit(self.settings.get(key, "") or ""); r.addWidget(e, 1)
        e.textChanged.connect(lambda x, k=key: self.settings.set(k, x))
        b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedSize(40, 28)
        b.clicked.connect(lambda _=False, ed=e: self._pick_dir(ed)); r.addWidget(b)
        layout.addLayout(r)
        return e

    def _pick_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, tr("Dossier", "Folder"), edit.text() or "")
        if d:
            edit.setText(d)

    def _file_row(self, layout, label, key):
        r = QHBoxLayout(); r.setSpacing(6)
        lb = QLabel(label); lb.setFixedWidth(150); r.addWidget(lb)
        e = QLineEdit(self.settings.get(key, "") or ""); r.addWidget(e, 1)
        e.textChanged.connect(lambda x, k=key: self.settings.set(k, x))
        b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedSize(40, 28)
        b.clicked.connect(lambda _=False, ed=e: self._pick_video(ed)); r.addWidget(b)
        layout.addLayout(r)
        return e

    def _pick_video(self, edit):
        p, _ = QFileDialog.getOpenFileName(
            self, tr("Fichier vidéo", "Video file"), edit.text() or "",
            tr("Vidéos (*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.flv *.wmv);;Tous (*.*)",
               "Videos (*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.flv *.wmv);;All (*.*)"))
        if p:
            edit.setText(p)

    def _on_mode_change(self, _idx=None):
        self.settings.set("gen_mode_idx", str(self.mode.currentIndex()))
        is_video = self.mode.currentIndex() == 1
        self.img_paths_box.setVisible(not is_video)
        self.vid_paths_box.setVisible(is_video)
        if is_video:
            self._refresh_ffmpeg_status()

    def _on_vid_format_change(self, idx):
        key = self._vid_format_keys[idx] if 0 <= idx < len(self._vid_format_keys) else "png8"
        self.settings.set("gen_vid_format", key)
        is_jpg = key == "jpg"
        self.vid_jpg_q_label.setVisible(is_jpg)
        self.vid_jpg_quality.setVisible(is_jpg)
        self.vid_jpg_q_value.setVisible(is_jpg)

    def _on_vid_val_toggle(self, checked):
        self.settings.set("gen_vid_val_enabled", bool(checked))
        self._sync_vid_val_visibility()

    def _sync_vid_val_visibility(self):
        on = self.vid_val_chk.isChecked()
        for w in (self.vid_val_count, self.vid_val_hq, self.vid_val_lq):
            w.setEnabled(on)

    def _refresh_ffmpeg_status(self):
        try:
            from src.core.lq_generator import resolve_ffmpeg
            p = resolve_ffmpeg()
        except Exception:
            p = None
        if p:
            self.vid_ffmpeg_status.setText(tr(f"✔ ffmpeg trouvé : {p}", f"✔ ffmpeg found: {p}"))
            self.vid_ffmpeg_status.setStyleSheet("color:#2ecc71;")
        else:
            self.vid_ffmpeg_status.setText(tr(
                "⚠ ffmpeg introuvable — installez-le et ajoutez-le au PATH, installez le paquet "
                "Python 'imageio-ffmpeg', ou placez un binaire portable dans bin/ffmpeg.exe.",
                "⚠ ffmpeg not found — install it and add it to PATH, install the 'imageio-ffmpeg' "
                "Python package, or place a portable binary in bin/ffmpeg.exe."))
            self.vid_ffmpeg_status.setStyleSheet("color:#e67e22;")

    def _build_tab(self, rows):
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame)
        w = QWidget(); sc.setWidget(w)
        # Grid with fixed-width columns so every slider/value-box lines up vertically:
        # col0 checkbox · (col1 label1, col2 slider1) · (col3 label2, col4 slider2) · stretch
        g = QGridLayout(w); g.setVerticalSpacing(5); g.setHorizontalSpacing(6); g.setContentsMargins(6, 6, 6, 6)
        for i, (chk_key, label, sliders) in enumerate(rows):
            cb = QCheckBox(label); cb.setFixedWidth(170)
            if chk_key in _GEN_TIPS:
                cb.setToolTip(_GEN_TIPS[chk_key])
            cb.setChecked(self._b(chk_key, False)); cb.toggled.connect(lambda val, k=chk_key: self.settings.set(k, bool(val)))
            self._w[chk_key] = cb
            g.addWidget(cb, i, 0)
            for j, (slabel, skey, default, lo, hi, step) in enumerate(sliders):
                lab = QLabel(slabel + " :"); lab.setFixedWidth(78); lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                lab.setStyleSheet("color:#888;")
                g.addWidget(lab, i, 1 + j * 2)
                init = self.settings.get(skey, default)
                try:
                    init = float(init)
                except (TypeError, ValueError):
                    init = float(default)
                sld = _GenSlider(lo, hi, step, init)
                sld.changed.connect(lambda k=skey: self.settings.set(k, self._w[k].value()))
                self._w[skey] = sld
                g.addWidget(sld, i, 2 + j * 2)
        g.setColumnStretch(5, 1)
        g.setRowStretch(len(rows), 1)
        return sc

    def _build_pipeline_tab(self):
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame)
        w = QWidget(); sc.setWidget(w)
        v = QVBoxLayout(w); v.setSpacing(6)
        h1 = QLabel(tr("Répétitions du pipeline", "Pipeline repetitions")); h1.setStyleSheet("font-weight:700;"); v.addWidget(h1)
        r1 = QHBoxLayout(); r1.addWidget(QLabel(tr("Passes :", "Passes:")))
        self.passes = QComboBox(); self.passes.addItems(["1", "2", "3", "4", "5"])
        self.passes.setCurrentText(str(self.settings.get("gen_passes", "1"))); self.passes.setFixedWidth(60)
        self.passes.currentTextChanged.connect(lambda x: self.settings.set("gen_passes", x))
        self._w["gen_passes"] = self.passes
        r1.addWidget(self.passes); r1.addStretch(1); v.addLayout(r1)

        h2 = QLabel(tr("Probabilité par dégradation", "Probability per degradation")); h2.setStyleSheet("font-weight:700; padding-top:8px;"); v.addWidget(h2)
        d = QLabel(tr("Probabilité globale d'appliquer chaque dégradation activée (0=jamais, 1=toujours).\n"
                   "Utile pour créer de la variété : même config → résultats différents.", "Global probability of applying each enabled degradation (0=never, 1=always).\nUseful to create variety: same config → different results."))
        d.setStyleSheet("color:#888;"); v.addWidget(d)
        r2 = QHBoxLayout(); r2.addWidget(QLabel(tr("Probabilité :", "Probability:")))
        init = self.settings.get("gen_prob", 1.0)
        try:
            init = float(init)
        except (TypeError, ValueError):
            init = 1.0
        self.prob = _GenSlider(0.0, 1.0, 0.05, init)
        self.prob.changed.connect(lambda: self.settings.set("gen_prob", self.prob.value()))
        self._w["gen_prob"] = self.prob
        r2.addWidget(self.prob); r2.addStretch(1); v.addLayout(r2)

        h3 = QLabel(tr("Mode probabiliste", "Probability mode")); h3.setStyleSheet("font-weight:700; padding-top:8px;"); v.addWidget(h3)
        r3 = QHBoxLayout()
        self.prob_mode = QComboBox(); self.prob_mode.addItems([
            tr("Indépendant (par dégradation)", "Independent (per degradation)"),
            tr("Par image (1 tirage)", "Per image (1 draw)")])
        # persist by INDEX (0/1), never by display text — the text is language-dependent
        # and a value saved in one language wouldn't match the combo after switching.
        self.prob_mode.setCurrentIndex(1 if str(self.settings.get("gen_prob_mode_idx", "0")) == "1" else 0)
        self.prob_mode.setFixedWidth(280)
        self.prob_mode.currentIndexChanged.connect(lambda i: self.settings.set("gen_prob_mode_idx", str(i)))
        self._w["gen_prob_mode"] = self.prob_mode
        r3.addWidget(self.prob_mode); r3.addStretch(1); v.addLayout(r3)
        v.addStretch(1)
        return sc

    def _preview_panel(self, title, color, is_before=False):
        box = QFrame(); box.setObjectName("genprev")
        box.setStyleSheet(f"QFrame#genprev{{border:1px solid {color}; border-radius:4px;}}")  # bg from theme
        bv = QVBoxLayout(box); bv.setContentsMargins(4, 4, 4, 4); bv.setSpacing(2)
        lt = QLabel(title); lt.setAlignment(Qt.AlignCenter); lt.setStyleSheet(f"color:{color}; font-weight:700; background:transparent;")
        bv.addWidget(lt)
        img = QLabel(tr("Aucune image", "No image") if is_before else tr("Appuyer sur ⚡", "Press ⚡"))
        img.setAlignment(Qt.AlignCenter); img.setStyleSheet("color:#555; background:transparent;")
        img.setMinimumHeight(self._prev_h); img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bv.addWidget(img, 1)
        return img, box

    # ── helpers / state ─────────────────────────────────────────────────────────
    def _b(self, key, default):
        v = self.settings.get(key, default)
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("1", "true", "yes", "on")

    def _collect_opts(self):
        W = self._w

        def _raw(key, default):
            # sliders / spin boxes (.value), combos (.currentText) and line edits (.text):
            # "passes" is a combo, so .value() raised and the 2-5 passes always ran as 1
            w = W.get(key)
            if w is None:
                return default
            for attr in ("value", "currentText", "text"):
                f = getattr(w, attr, None)
                if callable(f):
                    return f()
            return default

        def _fv(key, default):
            try:
                return float(_raw(key, default))
            except Exception:
                return float(default)

        def _iv(key, default):
            try:
                return int(round(float(_raw(key, default))))
            except Exception:
                return int(default)

        def _bv(key):
            try:
                return bool(W[key].isChecked())
            except Exception:
                return False

        return {
            "scale": int(self.scale.currentText()),
            "method": _pil_method(self.method.currentText()),
            "dpid_lambda": self.dpid_sld.value() / 100,
            "blur": _bv("gen_blur"), "blur_sigma": _fv("gen_blur_sigma", 1.0),
            "noise": _bv("gen_noise"), "noise_sigma": _fv("gen_noise_sigma", 10),
            "jpeg": _bv("gen_jpeg"), "jpeg_q": _iv("gen_jpeg_q", 50),
            "color_jitter": _bv("gen_color_jitter"),
            "sharpen": _bv("gen_sharpen"), "sharpen_amt": _fv("gen_sharpen_amt", 1.5),
            "posterize": _bv("gen_posterize"), "posterize_bits": _iv("gen_posterize_bits", 4),
            "banding": _bv("gen_banding"), "banding_levels": _iv("gen_banding_levels", 32),
            "ca": _bv("gen_ca"), "ca_shift": _iv("gen_ca_shift", 2),
            "disc_blur": _bv("gen_disc_blur"), "disc_blur_r": _fv("gen_disc_blur_r", 4.0),
            "vignette": _bv("gen_vignette"), "vignette_str": _fv("gen_vignette_str", 0.4),
            "halo": _bv("gen_halo"), "halo_str": _fv("gen_halo_str", 0.3), "halo_r": _iv("gen_halo_r", 6),
            "saturation": _bv("gen_saturation"), "sat_factor": _fv("gen_sat_factor", 0.8),
            "color_levels": _bv("gen_color_levels"),
            "col_high": _fv("gen_col_high", 230), "col_low": _fv("gen_col_low", 10),
            "quantize_depth": _bv("gen_quantize_depth"), "qdepth_bits": _iv("gen_qdepth_bits", 6),
            "chroma_sub": _bv("gen_chroma_sub"),
            "aliasing": _bv("gen_aliasing"), "aliasing_str": _fv("gen_aliasing_str", 0.75),
            "interlace_weave": _bv("gen_interlace_weave"), "weave_str": _fv("gen_weave_str", 0.8),
            "interlace_flicker": _bv("gen_interlace_flicker"), "flicker_amp": _fv("gen_flicker_amp", 0.22),
            "interlace_blend": _bv("gen_interlace_blend"), "blend_mix": _fv("gen_blend_mix", 0.55),
            "scanlines": _bv("gen_scanlines"),
            "scanlines_period": _iv("gen_scanlines_period", 3), "scanlines_dark": _fv("gen_scanlines_dark", 0.35),
            "vhs": _bv("gen_vhs"), "vhs_str": _fv("gen_vhs_str", 0.3),
            "screentone": _bv("gen_screentone"), "screen_sz": _iv("gen_screen_sz", 6),
            "dithering": _bv("gen_dithering"), "dither_col": _iv("gen_dither_col", 4),
            "sinusoidal": _bv("gen_sinusoidal"),
            "sin_period": _fv("gen_sin_period", 300), "sin_alpha": _fv("gen_sin_alpha", 0.2),
            "pixel_shift": _bv("gen_pixel_shift"), "pxsh_amt": _iv("gen_pxsh_amt", 3),
            "film_grain": _bv("gen_film_grain"),
            "grain_sigma": _fv("gen_grain_sigma", 0.08), "grain_size": _iv("gen_grain_size", 1),
            "oversharp": _bv("gen_oversharp"), "oversharp_amt": _fv("gen_oversharp_amt", 1.4),
            "motion_blur": _bv("gen_motion_blur"), "mb_px": _iv("gen_mb_px", 7),
            "codec": _bv("gen_codec"), "codec_q1": _iv("gen_codec_q1", 30), "codec_q2": _iv("gen_codec_q2", 60),
            "salt_pepper": _bv("gen_salt_pepper"), "sp_amt": _fv("gen_sp_amt", 0.01),
            "halation": _bv("gen_halation"), "hal_str": _fv("gen_hal_str", 0.2),
            "autocrop": _bv("gen_autocrop"), "crop_sz": _iv("gen_crop_sz", 256),
            "passes": _iv("gen_passes", 1),
            "prob": _fv("gen_prob", 1.0),
            "prob_per_image": self.prob_mode.currentIndex() == 1,
        }

    # ── preview ───────────────────────────────────────────────────────────────
    def _on_height(self, h):
        self._prev_h = h; self.h_lbl.setText(f"{h} px")
        self.prev_before.setMinimumHeight(h); self.prev_after.setMinimumHeight(h)
        self.settings.set("gen_prev_h", h)
        self._rescale_previews()

    def _rescale_previews(self):
        for lbl in (self.prev_before, self.prev_after):
            pm = lbl.property("_pm")
            if pm:
                lbl.setPixmap(pm.scaled(lbl.width(), lbl.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _set_preview(self, lbl, pm):
        lbl.setProperty("_pm", pm)
        lbl.setPixmap(pm.scaled(lbl.width(), lbl.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev); self._rescale_previews()

    def _load_preview(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Charger image test", "Load test image"), "",
                                           tr("Images (*.png *.jpg *.jpeg *.bmp *.webp);;Tous (*.*)", "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All (*.*)"))
        if not p:
            return
        try:
            from PIL import Image
            self._prev_pil = Image.open(p).convert("RGB")
            pm = QPixmap(p)
            self._set_preview(self.prev_before, pm)
            self.prev_status.setText(f"{os.path.basename(p)} — {self._prev_pil.width}×{self._prev_pil.height}")
            self.prev_status.setStyleSheet("color:#2ecc71;")
        except Exception as e:
            self.prev_status.setText(str(e)); self.prev_status.setStyleSheet("color:#e74c3c;")

    def _apply_preview(self):
        if self._prev_pil is None:
            self.prev_status.setText(tr("Charger une image d'abord.", "Load an image first.")); self.prev_status.setStyleSheet("color:#e67e22;")
            return
        self.prev_status.setText(tr("Traitement…", "Processing…")); self.prev_status.setStyleSheet("color:#3498db;")
        self._pw = _PreviewWorker(self._prev_pil, self._collect_opts())
        self._pw.done.connect(self._preview_ready); self._pw.fail.connect(self._preview_fail); self._pw.start()

    def _preview_ready(self, data):
        pm = QPixmap(); pm.loadFromData(data)
        self._set_preview(self.prev_after, pm)
        self.prev_status.setText(tr("Aperçu mis à jour.", "Preview updated.")); self.prev_status.setStyleSheet("color:#2ecc71;")
        self._pw = None

    def _preview_fail(self, msg):
        self.prev_status.setText(msg); self.prev_status.setStyleSheet("color:#e74c3c;"); self._pw = None

    # ── run batch ─────────────────────────────────────────────────────────────
    def _run(self):
        if self.mode.currentIndex() == 1:
            self._run_video()
            return
        hq, lq = self.hq.text().strip(), self.lq.text().strip()
        if not hq or not os.path.isdir(hq):
            self.status.setText(tr("Dossier source (HQ) invalide.", "Invalid source (HQ) folder.")); self.status.setStyleSheet("color:#e74c3c;")
            return
        if not lq:
            self.status.setText(tr("Dossier destination (LQ) requis.", "Destination (LQ) folder required.")); self.status.setStyleSheet("color:#e74c3c;")
            return
        self.settings.set("gen_hq", hq); self.settings.set("gen_lq", lq)
        self.btn.setEnabled(False); self.btn.setText(tr("Génération…", "Generating…")); self.prog.setValue(0)
        self.status.setStyleSheet("color:#888;")
        self.toolProgress.emit(0)
        self._worker = _BatchWorker(hq, lq, self._collect_opts())
        # backend messages (per-image errors, the matched-HQ folder): the Images mode has no
        # log box, they were dropped and a crash read as "No image found"
        self._img_msgs = []
        self._worker.line.connect(self._img_msgs.append)
        self._worker.progress.connect(self._on_progress)
        self._worker.preview.connect(self._on_batch_preview)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_progress(self, frac, done, total):
        self.prog.setValue(int(frac * 100)); self.status.setText(f"{done}/{total}")
        self.toolProgress.emit(int(frac * 100))

    def _on_batch_preview(self, b_before, b_after):
        p1 = QPixmap(); p1.loadFromData(b_before); self._set_preview(self.prev_before, p1)
        p2 = QPixmap(); p2.loadFromData(b_after); self._set_preview(self.prev_after, p2)

    def _on_done(self, n, total):
        self.btn.setEnabled(True); self.btn.setText(tr("▶  Lancer la Génération", "▶  Run Generation"))
        msgs = list(getattr(self, "_img_msgs", []) or [])
        fatal = [m for m in msgs if m.startswith("❌")]
        errs = [m for m in msgs if "⚠" in m]
        info = [m for m in msgs if m.startswith("ℹ")]
        if fatal:
            txt = fatal[-1]
        elif total:
            txt = tr(f"Terminé — {n}/{total} images traitées.", f"Done — {n}/{total} images processed.")
            if errs:
                txt += tr(f"  {len(errs)} erreur(s), ex. : {errs[0].strip()}", f"  {len(errs)} error(s), e.g.: {errs[0].strip()}")
        else:
            txt = tr("Aucune image trouvée.", "No image found.")
        if info:
            txt += "\n" + info[-1]
        self.status.setText(txt); self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#2ecc71;" if (n and not fatal and not errs) else
                                  "color:#e74c3c;" if fatal else "color:#e67e22;")
        self._worker = None
        self.toolFinished.emit(bool(n))
        try:
            from qt_ui.notify import notify
            notify(self.settings, tr("Génération LQ terminée", "LQ generation finished"), f"{n}/{total} images", "notif_win11_batch")
        except Exception:
            pass

    # ── run video ─────────────────────────────────────────────────────────────
    def _run_video(self):
        video = self.vid_file.text().strip()
        hq_out = self.vid_hq.text().strip()
        lq_out = self.vid_lq.text().strip()
        if not video or not os.path.isfile(video):
            self.status.setText(tr("Fichier vidéo invalide.", "Invalid video file.")); self.status.setStyleSheet("color:#e74c3c;")
            return
        if not hq_out or not lq_out:
            self.status.setText(tr("Dossiers de sortie HQ/LQ requis.", "Output HQ/LQ folders required.")); self.status.setStyleSheet("color:#e74c3c;")
            return
        try:
            from src.core.lq_generator import resolve_ffmpeg
            if not resolve_ffmpeg():
                self.status.setText(tr("ffmpeg introuvable — voir le message sous les champs vidéo.", "ffmpeg not found — see the message under the video fields."))
                self.status.setStyleSheet("color:#e74c3c;")
                self._refresh_ffmpeg_status()
                return
        except Exception:
            pass

        prefix = self.vid_prefix.text().strip() or None
        try:
            fps = float(self.vid_fps.text().strip() or "0")
        except ValueError:
            fps = 0.0

        val_count = 0
        val_hq_dir = val_lq_dir = None
        if self.vid_val_chk.isChecked():
            try:
                val_count = max(0, int(float(self.vid_val_count.text().strip() or "0")))
            except ValueError:
                val_count = 0
            val_hq_dir = self.vid_val_hq.text().strip() or None
            val_lq_dir = self.vid_val_lq.text().strip() or None
        val_position = ["end", "start", "random"][self.vid_val_pos.currentIndex()]
        img_format = self._vid_format_keys[self.vid_format.currentIndex()]
        jpg_quality = self.vid_jpg_quality.value()

        self.settings.set("gen_vid_path", video)
        self.settings.set("gen_vid_hq", hq_out)
        self.settings.set("gen_vid_lq", lq_out)

        self.vid_log.clear()
        self.btn.setEnabled(False); self.btn.setText(tr("Génération…", "Generating…"))
        self.prog.setRange(0, 0)  # indeterminate while ffmpeg extracts
        self.status.setText(tr("Extraction des frames…", "Extracting frames…")); self.status.setStyleSheet("color:#888;")
        self.toolBusy.emit()

        self._vworker = _VideoBatchWorker(video, hq_out, lq_out, self._collect_opts(),
                                          prefix, fps, val_count, val_hq_dir, val_lq_dir,
                                          val_position=val_position,
                                          img_format=img_format, jpg_quality=jpg_quality)
        self._vworker.line.connect(self.vid_log.appendPlainText)
        self._vworker.progress.connect(self._on_vid_progress)
        self._vworker.preview.connect(self._on_batch_preview)
        self._vworker.done.connect(self._on_vid_done)
        self._vworker.failed.connect(self._on_vid_failed)
        self._vworker.start()

    def _on_vid_progress(self, phase, frac, done, total):
        if phase == "extract":
            self.prog.setRange(0, 0)
            self.status.setText(tr(f"Extraction… {done} frames", f"Extracting… {done} frames"))
            self.toolBusy.emit()
        elif phase in ("train", "val"):
            self.prog.setRange(0, 100)
            f = float(frac) if frac is not None else 0.0
            self.prog.setValue(int(f * 100))
            label = tr("Dégradation (Train)", "Degrading (Train)") if phase == "train" else tr("Dégradation (Validation)", "Degrading (Validation)")
            self.status.setText(f"{label} — {done}/{total}")
            self.toolProgress.emit(int(f * 100))

    def _on_vid_done(self, result):
        self.btn.setEnabled(True); self.btn.setText(tr("▶  Lancer la Génération", "▶  Run Generation"))
        self.prog.setRange(0, 100); self.prog.setValue(100)
        msg = tr(
            f"Terminé — {result.get('frames_extracted', 0)} frames extraites, "
            f"{result.get('train_processed', 0)}/{result.get('train_total', 0)} Train, "
            f"{result.get('val_processed', 0)}/{result.get('val_total', 0)} Val.",
            f"Done — {result.get('frames_extracted', 0)} frames extracted, "
            f"{result.get('train_processed', 0)}/{result.get('train_total', 0)} Train, "
            f"{result.get('val_processed', 0)}/{result.get('val_total', 0)} Val.")
        tp, tt = result.get("train_processed", 0), result.get("train_total", 0)
        vp, vt = result.get("val_processed", 0), result.get("val_total", 0)
        complete = tt > 0 and tp == tt and vp == vt
        if not complete:
            msg += tr("  Des frames n'ont pas été dégradées — voir le journal.",
                      "  Some frames were not degraded — see the log.")
        self.status.setText(msg)
        self.status.setStyleSheet("color:#2ecc71;" if complete else "color:#e67e22;" if tp else "color:#e74c3c;")
        self._vworker = None
        self.toolFinished.emit(complete)
        try:
            from qt_ui.notify import notify
            notify(self.settings, tr("Génération LQ (vidéo) terminée", "LQ generation (video) finished"), msg, "notif_win11_batch")
        except Exception:
            pass

    def _on_vid_failed(self, msg):
        self.btn.setEnabled(True); self.btn.setText(tr("▶  Lancer la Génération", "▶  Run Generation"))
        self.prog.setRange(0, 100); self.prog.setValue(0)
        self.status.setText(msg); self.status.setStyleSheet("color:#e74c3c;")
        self._vworker = None
        self.toolFinished.emit(False)
