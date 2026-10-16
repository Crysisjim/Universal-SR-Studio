"""
degradations.py — Configuration › Dégradations (PySide6 port of create_page_degradations).

Faithful to the CTk v2.5.8 page: preset row (Light/Medium/Heavy/Overkill) + Deg.Custom
+ VHS/Analog toggles + custom-severity buttons, a 6-tab notebook (Stage 1 / Stage 2 /
Custom / Custom 2 / Custom 3 / Custom 4) each with two scrollable columns of controls
(checkbox + slider/range/entry/combo), and a live OTF preview (HQ → LQ) wired to the
existing backend src.core.otf_preview.apply_otf_pipeline (run off-thread).
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QCheckBox, QSlider, QTabWidget, QScrollArea, QFrame, QFileDialog,
    QMessageBox, QSizePolicy, QDialog, QGraphicsView, QGraphicsScene,
)
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtCore import Qt, QThread, Signal, QEvent

from qt_ui.i18n import tr


class _ZoomView(QGraphicsView):
    """Single-image viewer: mouse-wheel zoom, drag pan, double-click reset."""
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self); self.setScene(self._scene)
        self._item = self._scene.addPixmap(pixmap)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.black)

    def wheelEvent(self, ev):
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def mouseDoubleClickEvent(self, ev):
        self.resetTransform(); self.fitInView(self._item, Qt.KeepAspectRatio)

# ── control specs per tab/column ────────────────────────────────────────────
# tuple forms:
#   ("h",   title)                                   -> section header
#   ("s",   label, key, default, smax, step)         -> slider + value box
#   ("r",   label, key, "min,max")                   -> range (two entries)
#   ("e",   label, key, default)                     -> plain entry
#   ("c",   label, key, [choices], default)          -> combo
_TABS = [
    ("  Stage 1  ", {
        "left": [
            ("h", "Stage 1 (Base)"),
            ("s", "Blur Prob", "blur_prob", 0.5, 1.0, 0.01),
            ("r", "Blur Sigma", "blur_sigma", "0.2,3.0"),
            ("s", "Kernel Size", "blur_kernel_size", 21, 31, 2),
            ("s", "Sinc Prob", "sinc_prob", 0.0, 1.0, 0.01),
            ("r", "Betag Range", "betag_range", "0.5,4"),
            ("r", "Betap Range", "betap_range", "1,2"),
            ("s", "Noise Prob", "gaussian_noise_prob", 0.5, 1.0, 0.01),
            ("r", "Noise Range", "noise_range", "1,30"),
            ("e", "Resize Prob", "resize_prob", "[0.2, 0.7, 0.1]"),
            ("r", "Resize Range", "resize_range", "0.5,1.5"),
        ],
        "right": [
            ("h", "Compression"),
            ("s", "JPEG Prob", "jpeg_prob", 1.0, 1.0, 0.01),
            ("r", "JPEG Range", "jpeg_range", "30,95"),
            ("s", "Gray Noise", "gray_noise_prob", 0.4, 1.0, 0.01),
            ("s", "Sinc (Ringing)", "final_sinc_prob", 0.8, 1.0, 0.01),
        ],
    }),
    ("  Stage 2  ", {
        "left": [
            ("h", tr("Stage 2 — Blur / Bruit", "Stage 2 — Blur / Noise")),
            ("s", "2nd Blur Prob", "second_blur_prob", 0.5, 1.0, 0.01),
            ("r", "Blur Sigma 2", "blur_sigma2", "0.2,1.5"),
            ("s", "Kernel Size 2", "blur_kernel_size2", 21, 31, 2),
            ("s", "Sinc Prob 2", "sinc_prob2", 0.0, 1.0, 0.01),
            ("r", "Betag Range 2", "betag_range2", "0.5,4"),
            ("r", "Betap Range 2", "betap_range2", "1,2"),
            ("s", "Noise Prob 2", "gaussian_noise_prob2", 0.5, 1.0, 0.01),
            ("r", "Noise Range 2", "noise_range2", "1,25"),
            ("s", "Gray Noise 2", "gray_noise_prob2", 0.0, 1.0, 0.01),
        ],
        "right": [
            ("h", "Stage 2 — Compression"),
            ("e", "Resize Prob 2", "resize_prob2", "[0.3, 0.4, 0.3]"),
            ("r", "Resize Range 2", "resize_range2", "0.3,1.2"),
            ("s", "JPEG Prob 2", "jpeg_prob2", 1.0, 1.0, 0.01),
            ("r", "JPEG Range 2", "jpeg_range2", "30,95"),
        ],
    }),
    ("  Custom   ", {
        "left": [
            ("h", tr("Quantification", "Quantization")),
            ("s", "Posterize Prob", "posterize_prob", 0.0, 1.0, 0.01),
            ("r", "Posterize Bits", "posterize_bits_range", "3,6"),
            ("s", "Banding Prob", "banding_prob", 0.0, 1.0, 0.01),
            ("r", "Banding Levels", "banding_levels_range", "16,64"),
            ("s", tr("Banding zones lisses", "Smooth-area Banding"), "smooth_banding_prob", 0.0, 1.0, 0.01),
            ("r", tr("Niveaux luma (zones lisses)", "Luma Levels (smooth)"), "smooth_banding_levels_range", "48,128"),
            ("s", tr("Part tons sombres", "Dark-tone Share"), "smooth_banding_dark_prob", 0.5, 1.0, 0.05),
            ("h", "VHS / Analog"),
            ("s", "VHS Prob", "vhs_prob", 0.0, 1.0, 0.01),
            ("r", "VHS Strength", "vhs_strength_range", "0.1,0.5"),
            ("h", tr("Flou Bokeh / Défocus", "Bokeh Blur / Defocus")),
            ("s", "Disc Blur Prob", "disc_blur_prob", 0.0, 1.0, 0.01),
            ("r", tr("Rayon (px)", "Radius (px)"), "disc_blur_radius_range", "2,8"),
            ("h", tr("Vignettage", "Vignetting")),
            ("s", "Vignette Prob", "vignette_prob", 0.0, 1.0, 0.01),
            ("r", tr("Intensité", "Intensity"), "vignette_strength_range", "0.2,0.6"),
            ("r", tr("Rayon net", "Sharp Radius"), "vignette_radius_range", "0.45,0.75"),
        ],
        "right": [
            ("h", tr("Optique / Analog", "Optics / Analog")),
            ("s", "Chroma Sub Prob", "chroma_prob", 0.0, 1.0, 0.01),
            ("s", "Chrom. Aber. Prob", "ca_prob", 0.0, 1.0, 0.01),
            ("r", "CA Shift (px)", "ca_shift_range", "1,5"),
            ("s", "Halation Prob", "halation_prob", 0.0, 1.0, 0.01),
            ("r", "Halation Strength", "halation_strength_range", "0.05,0.3"),
            ("s", "Salt&Pepper Prob", "salt_pepper_prob", 0.0, 1.0, 0.01),
            ("r", "S&P Amount", "salt_pepper_amount_range", "0.001,0.05"),
            ("h", tr("Profondeur Bit (Quant.)", "Bit Depth (Quant.)")),
            ("s", "Quant. Depth Prob", "quantize_depth_prob", 0.0, 1.0, 0.01),
            ("r", "Bits (4–7)", "quantize_depth_bits_range", "4,7"),
            ("h", tr("Clusters Couplés (sr_degrade)", "Coupled Clusters (sr_degrade)")),
            ("s", tr("Optique Prob", "Optics Prob"), "coupled_optical_prob", 0.0, 1.0, 0.01),
            ("s", "Vintage Prob", "coupled_vintage_prob", 0.0, 1.0, 0.01),
        ],
    }),
    (" Custom 2  ", {
        "left": [
            ("h", tr("Aliasing / Lignes", "Aliasing / Lines")),
            ("s", "Aliasing Prob", "aliasing_prob", 0.0, 1.0, 0.01),
            ("r", "Aliasing Scale", "aliasing_scale_range", "0.5,0.85"),
            ("h", tr("Entrelacement — Weave", "Interlacing — Weave")),
            ("s", "Weave Prob", "interlace_weave_prob", 0.0, 1.0, 0.01),
            ("r", "Weave Strength", "interlace_weave_strength_range", "0.5,1.0"),
            ("h", tr("Entrelacement — Flicker", "Interlacing — Flicker")),
            ("s", "Flicker Prob", "interlace_flicker_prob", 0.0, 1.0, 0.01),
            ("r", "Flicker Strength", "interlace_flicker_strength_range", "0.1,0.4"),
            ("h", tr("Entrelacement — Blend", "Interlacing — Blend")),
            ("s", "Blend Prob", "interlace_blend_prob", 0.0, 1.0, 0.01),
            ("r", "Blend Strength", "interlace_blend_strength_range", "0.3,1.0"),
        ],
        "right": [
            ("h", tr("Grain Cinéma", "Film Grain")),
            ("s", "Film Grain Prob", "film_grain_prob", 0.0, 1.0, 0.01),
            ("r", "Grain Strength", "film_grain_strength_range", "0.03,0.12"),
            ("r", "Grain Size (px)", "film_grain_size_range", "1,2"),
            ("h", tr("Sur-Netteté (Halos)", "Oversharpening (Halos)")),
            ("s", "Oversharp Prob", "oversharp_prob", 0.0, 1.0, 0.01),
            ("r", "Oversharp Strength", "oversharp_strength_range", "0.5,2.0"),
            ("h", "Scanlines CRT"),
            ("s", "Scanlines Prob", "scanlines_prob", 0.0, 1.0, 0.01),
            ("r", "Scanlines Strength", "scanlines_strength_range", "0.2,0.5"),
            ("r", "Scanlines Spacing", "scanlines_spacing_range", "2,4"),
        ],
    }),
    (" Custom 3  ", {
        "left": [
            ("h", "Screentone / Halftone"),
            ("s", "Screentone Prob", "screentone_prob", 0.0, 1.0, 0.01),
            ("r", "Dot Size", "screentone_dot_size", "7,15"),
            ("r", "Angle", "screentone_angle", "0,90"),
            ("c", tr("Type de point", "Dot Type"), "screentone_dot_type", ["circle", "ellipse", "square", "cross"], "circle"),
            ("c", tr("Espace couleur", "Color Space"), "screentone_color_space", ["rgb", "cmyk", "hsv", "gray"], "rgb"),
            ("h", "Dithering"),
            ("s", "Dithering Prob", "dithering_prob", 0.0, 1.0, 0.01),
            ("r", tr("Canaux couleur", "Color Channels"), "dithering_color_ch", "2,8"),
            ("c", tr("Algorithme", "Algorithm"), "dithering_type",
             ["quantize", "floyd_steinberg", "jarvis_judice", "stucki", "atkinson", "ordered"], "floyd_steinberg"),
        ],
        "right": [
            ("h", tr("Pixelisation", "Pixelation")),
            ("s", "Pixelate Prob", "pixelate_prob", 0.0, 1.0, 0.01),
            ("r", "Block Size", "pixelate_size", "2,16"),
            ("h", tr("Sinusoïdal (ondulations)", "Sinusoidal (waves)")),
            ("s", "Sin Prob", "sin_prob", 0.0, 1.0, 0.01),
            ("r", tr("Fréquence", "Frequency"), "sin_shape", "100,600"),
            ("r", "Amplitude", "sin_alpha", "0.1,0.4"),
            ("r", tr("Biais", "Bias"), "sin_bias", "0.8,1.2"),
            ("c", "Orientation", "sin_orientation", ["random", "horizontal", "vertical"], "random"),
            ("h", tr("Sous-Échantillonnage Chroma", "Chroma Subsampling")),
            ("s", "Subsamp Prob", "subsampling_prob", 0.0, 1.0, 0.01),
            ("c", "Format", "subsampling_format", ["4:4:4", "4:2:2", "4:2:0", "4:1:1"], "4:2:0"),
            ("c", "Standard YUV", "subsampling_yuv", ["601", "709", "2020"], "601"),
        ],
    }),
    (" Custom 4  ", {
        "left": [
            ("h", tr("Niveaux de Couleur", "Color Levels")),
            ("s", "Color Level Prob", "color_level_prob", 0.0, 1.0, 0.01),
            ("r", "Output High", "color_level_high", "220,255"),
            ("r", "Output Low", "color_level_low", "0,35"),
            ("r", "Gamma", "color_level_gamma", "0.7,1.5"),
            ("h", "Halo / Ringing (wtp)"),
            ("s", "Halo Prob", "wtp_halo_prob", 0.0, 1.0, 0.01),
            ("r", "Halo Strength", "wtp_halo_strength", "0.1,0.5"),
            ("r", "Halo Radius", "wtp_halo_radius", "3,12"),
        ],
        "right": [
            ("h", tr("Saturation", "Saturation")),
            ("s", "Saturation Prob", "saturation_prob", 0.0, 1.0, 0.01),
            ("r", "Saturation", "saturation_range", "0.3,1.8"),
            ("h", tr("Décalage Pixel (Shift)", "Pixel Shift")),
            ("s", "Shift Prob", "shift_prob", 0.0, 1.0, 0.01),
            ("r", "Amplitude (px)", "shift_range", "1,8"),
            ("c", tr("Axe", "Axis"), "shift_axis", ["random", "horizontal", "vertical", "both"], "random"),
            ("h", tr("⚙ Avancé (Redux natif)", "⚙ Advanced (Redux native)")),
            ("s", "ThickLines Prob", "thicklines_prob", 0.0, 1.0, 0.01),
            ("b", tr("LQ USM (netteté LQ)", "LQ USM (LQ sharpening)"), "lq_usm", False),
            ("r", "USM Radius", "lq_usm_radius_range", "1,25"),
        ],
    }),
    (" Custom 5  ", {
        "left": [
            ("h", tr("DCT Codec-Block (4×4, table radiale)", "DCT Codec-Block (4×4, radial table)")),
            ("s", "DCT Block Prob", "dct_block_prob", 0.0, 1.0, 0.01),
            ("r", tr("Qualité (1=fort, 100=léger)", "Quality (1=heavy, 100=light)"), "dct_block_quality_range", "20,80"),
            ("r", tr("Taille de bloc", "Block size"), "dct_block_size_range", "4,4"),
            ("r", tr("Facteur chroma (plus grossier)", "Chroma factor (coarser)"), "dct_block_chroma_scale_range", "1.5,2.5"),
            ("c", "Standard YUV", "dct_block_yuv", ["601", "709", "2020"], "601"),
        ],
        "right": [],
    }),
]

# Severity → per-key probability curve (exact port of CTk _apply_severity_preset's _T
# table). Index order: none, mild, medium, heavy, extreme. Only touches probs, not ranges.
_SEVERITY_IDX = {"none": 0, "mild": 1, "medium": 2, "heavy": 3, "extreme": 4}
_SEVERITY_CURVES = {
    "posterize_prob":         (0, 0.1,  0.2,  0.4,  0.7),
    "banding_prob":           (0, 0.1,  0.25, 0.4,  0.7),
    "smooth_banding_prob":    (0, 0.1,  0.2,  0.35, 0.5),
    "chroma_prob":            (0, 0.15, 0.3,  0.5,  0.8),
    "ca_prob":                (0, 0.1,  0.2,  0.4,  0.6),
    "halation_prob":          (0, 0.1,  0.15, 0.3,  0.5),
    "salt_pepper_prob":       (0, 0.05, 0.15, 0.3,  0.5),
    "vhs_prob":               (0, 0.0,  0.1,  0.3,  0.5),
    "aliasing_prob":          (0, 0.0,  0.1,  0.25, 0.4),
    "interlace_weave_prob":   (0, 0.0,  0.05, 0.2,  0.4),
    "interlace_flicker_prob": (0, 0.0,  0.05, 0.15, 0.3),
    "interlace_blend_prob":   (0, 0.0,  0.05, 0.15, 0.3),
    "film_grain_prob":        (0, 0.2,  0.35, 0.5,  0.7),
    "oversharp_prob":         (0, 0.1,  0.2,  0.35, 0.5),
    "scanlines_prob":         (0, 0.0,  0.05, 0.2,  0.4),
    "screentone_prob":        (0, 0.0,  0.05, 0.15, 0.3),
    "dithering_prob":         (0, 0.0,  0.05, 0.15, 0.3),
    "pixelate_prob":          (0, 0.0,  0.05, 0.1,  0.25),
    "sin_prob":               (0, 0.0,  0.05, 0.1,  0.2),
    "subsampling_prob":       (0, 0.1,  0.2,  0.35, 0.6),
    "color_level_prob":       (0, 0.1,  0.2,  0.35, 0.5),
    "wtp_halo_prob":          (0, 0.05, 0.15, 0.3,  0.5),
    "saturation_prob":        (0, 0.1,  0.2,  0.35, 0.5),
    "shift_prob":             (0, 0.0,  0.0,  0.1,  0.3),
    "disc_blur_prob":         (0, 0.1,  0.2,  0.35, 0.5),
    "vignette_prob":          (0, 0.15, 0.3,  0.45, 0.6),
    "quantize_depth_prob":    (0, 0.0,  0.1,  0.25, 0.4),
    "coupled_optical_prob":   (0, 0.0,  0.1,  0.2,  0.4),
    "coupled_vintage_prob":   (0, 0.0,  0.0,  0.15, 0.3),
}

# Preset (Light/Medium/Heavy/Overkill) — exact port of CTk apply_deg_preset's _PRESETS.
# Touches both Stage1/2 OTF probs+ranges AND every custom degradation prob+range.
_CUSTOM_KEYS = {
    "posterize_prob", "posterize_bits_range", "banding_prob", "banding_levels_range",
    "chroma_prob", "ca_prob", "ca_shift_range", "halation_prob", "halation_strength_range",
    "salt_pepper_prob", "salt_pepper_amount_range", "vhs_prob", "vhs_strength_range",
    "aliasing_prob", "aliasing_scale_range",
    "interlace_weave_prob", "interlace_weave_strength_range",
    "interlace_flicker_prob", "interlace_flicker_strength_range",
    "interlace_blend_prob", "interlace_blend_strength_range",
    "film_grain_prob", "film_grain_strength_range", "film_grain_size_range",
    "oversharp_prob", "oversharp_strength_range",
    "scanlines_prob", "scanlines_strength_range", "scanlines_spacing_range",
}
_VHS_KEYS = {"vhs_prob", "vhs_strength_range"}
_PRESETS = {
    "Light": {
        "blur_prob": 0.2, "blur_sigma": "[0.2, 1.5]",
        "gaussian_noise_prob": 0.1, "noise_range": "[1, 10]",
        "gray_noise_prob": 0.1,
        "jpeg_prob": 0.5, "jpeg_range": "[80, 99]",
        "second_blur_prob": 0.3, "blur_sigma2": "[0.2, 1.0]",
        "gaussian_noise_prob2": 0.1, "noise_range2": "[1, 8]",
        "gray_noise_prob2": 0.0, "jpeg_range2": "[75, 99]",
        "final_sinc_prob": 0.3,
        "posterize_prob": 0.0, "posterize_bits_range": "[4, 8]",
        "banding_prob": 0.0, "banding_levels_range": "[32, 64]",
        "chroma_prob": 0.0,
        "ca_prob": 0.0, "ca_shift_range": "[1, 3]",
        "halation_prob": 0.0, "halation_strength_range": "[0.05, 0.15]",
        "salt_pepper_prob": 0.0, "salt_pepper_amount_range": "[0.001, 0.01]",
        "vhs_prob": 0.0, "vhs_strength_range": "[0.1, 0.3]",
        "aliasing_prob": 0.0, "aliasing_scale_range": "[0.7, 0.9]",
        "interlace_weave_prob": 0.0, "interlace_weave_strength_range": "[0.5, 1.0]",
        "interlace_flicker_prob": 0.0, "interlace_flicker_strength_range": "[0.1, 0.3]",
        "interlace_blend_prob": 0.0, "interlace_blend_strength_range": "[0.3, 0.7]",
        "film_grain_prob": 0.0, "film_grain_strength_range": "[0.03, 0.08]", "film_grain_size_range": "[1, 2]",
        "oversharp_prob": 0.0, "oversharp_strength_range": "[0.5, 1.5]",
        "scanlines_prob": 0.0, "scanlines_strength_range": "[0.2, 0.4]", "scanlines_spacing_range": "[2, 4]",
    },
    "Medium": {
        "blur_prob": 0.5, "blur_sigma": "[0.2, 3.0]",
        "gaussian_noise_prob": 0.3, "noise_range": "[1, 20]",
        "gray_noise_prob": 0.2,
        "jpeg_prob": 0.8, "jpeg_range": "[50, 95]",
        "second_blur_prob": 0.5, "blur_sigma2": "[0.2, 1.5]",
        "gaussian_noise_prob2": 0.3, "noise_range2": "[1, 15]",
        "gray_noise_prob2": 0.1, "jpeg_range2": "[50, 95]",
        "final_sinc_prob": 0.5,
        "posterize_prob": 0.0, "posterize_bits_range": "[4, 8]",
        "banding_prob": 0.0, "banding_levels_range": "[16, 64]",
        "chroma_prob": 0.1,
        "ca_prob": 0.0, "ca_shift_range": "[1, 3]",
        "halation_prob": 0.0, "halation_strength_range": "[0.05, 0.2]",
        "salt_pepper_prob": 0.05, "salt_pepper_amount_range": "[0.001, 0.02]",
        "vhs_prob": 0.0, "vhs_strength_range": "[0.1, 0.3]",
        "aliasing_prob": 0.1, "aliasing_scale_range": "[0.65, 0.9]",
        "interlace_weave_prob": 0.0, "interlace_weave_strength_range": "[0.5, 1.0]",
        "interlace_flicker_prob": 0.0, "interlace_flicker_strength_range": "[0.1, 0.3]",
        "interlace_blend_prob": 0.0, "interlace_blend_strength_range": "[0.3, 0.7]",
        "film_grain_prob": 0.15, "film_grain_strength_range": "[0.03, 0.1]", "film_grain_size_range": "[1, 2]",
        "oversharp_prob": 0.1, "oversharp_strength_range": "[0.5, 1.5]",
        "scanlines_prob": 0.0, "scanlines_strength_range": "[0.2, 0.4]", "scanlines_spacing_range": "[2, 4]",
    },
    "Heavy": {
        "blur_prob": 0.7, "blur_sigma": "[1.0, 4.0]",
        "gaussian_noise_prob": 0.5, "noise_range": "[10, 40]",
        "gray_noise_prob": 0.4,
        "jpeg_prob": 1.0, "jpeg_range": "[30, 80]",
        "second_blur_prob": 0.8, "blur_sigma2": "[0.5, 2.5]",
        "gaussian_noise_prob2": 0.5, "noise_range2": "[5, 25]",
        "gray_noise_prob2": 0.3, "jpeg_range2": "[30, 80]",
        "final_sinc_prob": 0.7,
        "posterize_prob": 0.2, "posterize_bits_range": "[3, 8]",
        "banding_prob": 0.3, "banding_levels_range": "[16, 64]",
        "chroma_prob": 0.3,
        "ca_prob": 0.2, "ca_shift_range": "[1, 5]",
        "halation_prob": 0.2, "halation_strength_range": "[0.1, 0.3]",
        "salt_pepper_prob": 0.1, "salt_pepper_amount_range": "[0.005, 0.03]",
        "vhs_prob": 0.1, "vhs_strength_range": "[0.2, 0.5]",
        "aliasing_prob": 0.2, "aliasing_scale_range": "[0.55, 0.85]",
        "interlace_weave_prob": 0.1, "interlace_weave_strength_range": "[0.5, 1.0]",
        "interlace_flicker_prob": 0.05, "interlace_flicker_strength_range": "[0.1, 0.35]",
        "interlace_blend_prob": 0.05, "interlace_blend_strength_range": "[0.3, 0.8]",
        "film_grain_prob": 0.3, "film_grain_strength_range": "[0.05, 0.15]", "film_grain_size_range": "[1, 3]",
        "oversharp_prob": 0.2, "oversharp_strength_range": "[0.8, 2.5]",
        "scanlines_prob": 0.05, "scanlines_strength_range": "[0.25, 0.5]", "scanlines_spacing_range": "[2, 4]",
    },
    "Overkill": {
        "blur_prob": 0.9, "blur_sigma": "[2.0, 6.0]",
        "gaussian_noise_prob": 0.8, "noise_range": "[20, 60]",
        "gray_noise_prob": 0.8,
        "jpeg_prob": 1.0, "jpeg_range": "[10, 60]",
        "second_blur_prob": 1.0, "blur_sigma2": "[1.0, 4.0]",
        "gaussian_noise_prob2": 0.8, "noise_range2": "[10, 40]",
        "gray_noise_prob2": 0.6, "jpeg_range2": "[10, 50]",
        "final_sinc_prob": 0.9,
        "posterize_prob": 0.5, "posterize_bits_range": "[2, 6]",
        "banding_prob": 0.6, "banding_levels_range": "[8, 32]",
        "chroma_prob": 0.6,
        "ca_prob": 0.4, "ca_shift_range": "[2, 7]",
        "halation_prob": 0.4, "halation_strength_range": "[0.15, 0.4]",
        "salt_pepper_prob": 0.2, "salt_pepper_amount_range": "[0.01, 0.05]",
        "vhs_prob": 0.3, "vhs_strength_range": "[0.3, 0.7]",
        "aliasing_prob": 0.35, "aliasing_scale_range": "[0.5, 0.8]",
        "interlace_weave_prob": 0.2, "interlace_weave_strength_range": "[0.7, 1.0]",
        "interlace_flicker_prob": 0.15, "interlace_flicker_strength_range": "[0.2, 0.5]",
        "interlace_blend_prob": 0.15, "interlace_blend_strength_range": "[0.5, 1.0]",
        "film_grain_prob": 0.5, "film_grain_strength_range": "[0.06, 0.18]", "film_grain_size_range": "[1, 4]",
        "oversharp_prob": 0.35, "oversharp_strength_range": "[1.0, 3.0]",
        "scanlines_prob": 0.15, "scanlines_strength_range": "[0.3, 0.6]", "scanlines_spacing_range": "[2, 4]",
    },
}


def pil_to_pixmap(im) -> QPixmap:
    im = im.convert("RGB")
    qim = QImage(im.tobytes("raw", "RGB"), im.width, im.height, im.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qim.copy())


class _PreviewWorker(QThread):
    done = Signal(object, object, str)
    fail = Signal(str)

    def __init__(self, src, cfg, scale):
        super().__init__(); self.src, self.cfg, self.scale = src, cfg, scale

    def run(self):
        try:
            from PIL import Image
            from src.core.otf_preview import apply_otf_pipeline
            hq = Image.open(self.src).convert("RGB")
            if max(hq.size) > 768:
                hq.thumbnail((768, 768), Image.LANCZOS)
            lq, log = apply_otf_pipeline(hq, self.cfg, scale=self.scale)
            self.done.emit(hq, lq, str(log))
        except Exception as e:
            self.fail.emit(str(e))


class _BatchWorker(QThread):
    """Generate OTF LQ variants for every image of a folder (Tester GT Val)."""
    progress = Signal(str)
    done = Signal(int, str)

    def __init__(self, folder, cfg, scale, variants):
        super().__init__()
        self.folder, self.cfg, self.scale, self.variants = folder, cfg, scale, variants

    def run(self):
        try:
            from PIL import Image
            from src.core.otf_preview import apply_otf_pipeline
        except Exception as e:
            self.done.emit(0, f"❌ {e}"); return
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        out = self.folder.rstrip("/\\") + "_LQ_preview"
        os.makedirs(out, exist_ok=True)
        files = [f for f in sorted(os.listdir(self.folder))
                 if os.path.splitext(f)[1].lower() in exts]
        n = 0
        for f in files:
            try:
                hq = Image.open(os.path.join(self.folder, f)).convert("RGB")
                for vi in range(max(1, self.variants)):
                    lq, _ = apply_otf_pipeline(hq, self.cfg, scale=self.scale)
                    stem, ext = os.path.splitext(f)
                    name = f"{stem}_v{vi}{ext}" if self.variants > 1 else f
                    lq.save(os.path.join(out, name))
                    n += 1
                self.progress.emit(f"… {f}")
            except Exception as e:
                self.progress.emit(f"❌ {f} : {e}")
        self.done.emit(n, out)


class DegradationsTab(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._getters: dict[str, callable] = {}
        self._setters: dict[str, callable] = {}  # key -> set the widget value (for load)
        self._checks: dict[str, QCheckBox] = {}
        self._sliders: dict[str, tuple] = {}  # key -> (slider, entry, lo, hi, step)
        self.worker = None

        self._last_hq = self._last_lq = None

        # Phase degradations: profile 0 = PSNR phase, 1 = GAN phase. The widgets show
        # the active profile; the other one is kept as a snapshot.
        self._profiles = [None, None]
        self._active = 0

        root = QVBoxLayout(self); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)
        root.addLayout(self._preset_row())
        root.addLayout(self._phase_row())

        self.tabs = QTabWidget(); self.tabs.setObjectName("degtabs")
        for title, cols in _TABS:
            self.tabs.addTab(self._build_tab(cols), title)
        root.addWidget(self.tabs, 2)

        root.addWidget(self._preview_panel(), 3)

        # Panel backgrounds + severity buttons are now driven by the global theme QSS
        # (objectNames degtabs / degcol / degprev / sevbtn) so they adapt to dark/light.
        # Disable mouse-wheel on sliders + combos in this tab (avoid accidental edits)
        for sld in self.findChildren(QSlider):
            sld.installEventFilter(self)
        for cb in self.findChildren(QComboBox):
            cb.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel and isinstance(obj, (QSlider, QComboBox)):
            return True  # eat the wheel event
        return super().eventFilter(obj, ev)

    # ── preset / severity row ─────────────────────────────────────────────────
    def _preset_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel(tr("Preset :", "Preset:")))
        self.preset = QComboBox(); self.preset.addItems(["Light", "Medium", "Heavy", "Overkill"])
        self.preset.setCurrentText("Medium"); self.preset.setFixedWidth(120)
        self.preset.currentTextChanged.connect(self._apply_preset)
        row.addWidget(self.preset)
        self.chk_custom = QCheckBox(tr("Deg. Custom", "Custom Deg.")); self.chk_custom.setChecked(True); row.addWidget(self.chk_custom)
        self.chk_vhs = QCheckBox("VHS/Analog"); row.addWidget(self.chk_vhs)
        sep = QLabel("│"); sep.setStyleSheet("color:#555;"); row.addWidget(sep)
        lbl = QLabel(tr("⚡ Sévérité custom :", "⚡ Custom severity:")); lbl.setStyleSheet("color:#9b59b6; font-weight:700;")
        row.addWidget(lbl)
        for label, key in (("none", "none"), ("mild", "mild"), ("med.", "medium"),
                           ("heavy", "heavy"), ("max", "extreme")):
            b = QPushButton(label); b.setObjectName("sevbtn")
            b.clicked.connect(lambda _=False, k=key: self._apply_severity(k))
            row.addWidget(b)
        row.addStretch(1)
        self.deg_info_lbl = QLabel(""); self.deg_info_lbl.setStyleSheet("color:#888; font-size:11px;")
        row.addWidget(self.deg_info_lbl)
        return row

    # ── phase profiles (PSNR / GAN) ───────────────────────────────────────────
    def _phase_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(6)
        self.chk_phase2 = QCheckBox(tr("Profil GAN différent", "Separate GAN profile"))
        self.chk_phase2.setToolTip(tr(
            "Deux profils de dégradations OTF : un pour la phase PSNR (avant le GAN), un pour "
            "la phase GAN. Après le démarrage du GAN (auto ou itération fixe), le moteur passe "
            "du profil PSNR au profil GAN par une rampe linéaire sur « Transition » itérations.\n"
            "Actif seulement avec un GAN (démarrage auto ou gan_start fixe ; sinon le profil GAN "
            "s'applique dès le début). Changent pendant le run : blur 1/2, resize, bruit, gray "
            "noise, JPEG, thicklines et toutes les dégradations custom. Restent sur le profil "
            "PSNR : formes de noyaux (sigma, taille, sinc) et final sinc.",
            "Two OTF degradation profiles: one for the PSNR phase (before the GAN), one for the "
            "GAN phase. After the GAN start (auto or fixed iteration), the engine moves from "
            "the PSNR profile to the GAN profile with a linear ramp over 'Transition' iterations.\n"
            "Only active with a GAN (auto start or fixed gan_start; otherwise the GAN profile "
            "applies from the start). Changed during the run: blur 1/2, resize, noise, gray "
            "noise, JPEG, thicklines and every custom degradation. Stay on the PSNR profile: "
            "kernel shapes (sigma, size, sinc) and final sinc."))
        row.addWidget(self.chk_phase2)
        row.addWidget(QLabel(tr("Profil édité :", "Edited profile:")))
        self.phase_combo = QComboBox()
        self.phase_combo.addItems([tr("Phase PSNR", "PSNR phase"), tr("Phase GAN", "GAN phase")])
        self.phase_combo.setFixedWidth(130); self.phase_combo.setEnabled(False)
        row.addWidget(self.phase_combo)
        row.addWidget(QLabel(tr("Transition (it) :", "Transition (it):")))
        self.ramp_edit = QLineEdit("15000"); self.ramp_edit.setFixedWidth(70)
        self.ramp_edit.setToolTip(tr(
            "Itérations de la rampe profil PSNR → profil GAN après le démarrage du GAN "
            "(conseillé 10-20k ; 0 = bascule immédiate).",
            "Iterations of the PSNR → GAN profile ramp after the GAN start "
            "(10-20k advised; 0 = immediate switch)."))
        self.ramp_edit.setEnabled(False)
        row.addWidget(self.ramp_edit)
        self.phase_info = QLabel(""); self.phase_info.setStyleSheet("color:#888; font-size:11px;")
        row.addWidget(self.phase_info)
        row.addStretch(1)
        self.chk_phase2.toggled.connect(self._on_phase2_toggled)
        self.phase_combo.currentIndexChanged.connect(self._on_phase_switch)
        return row

    def _snapshot(self) -> dict:
        return {"v": self._collect(), "c": {k: c.isChecked() for k, c in self._checks.items()}}

    def _restore(self, snap: dict):
        for k, v in snap.get("v", {}).items():
            setter = self._setters.get(k)
            if setter is not None:
                try:
                    setter(v)
                except Exception:
                    pass
        for k, on in snap.get("c", {}).items():
            if k in self._checks:
                self._checks[k].setChecked(bool(on))

    def _on_phase2_toggled(self, on):
        self.phase_combo.setEnabled(on); self.ramp_edit.setEnabled(on)
        if not on and self._active == 1:
            self.phase_combo.setCurrentIndex(0)
        self._update_phase_info()

    def _on_phase_switch(self, idx):
        if idx == self._active:
            return
        self._profiles[self._active] = self._snapshot()
        if self._profiles[idx] is None:
            # first visit: start the GAN profile as a copy of the PSNR one
            self._profiles[idx] = self._snapshot()
        self._restore(self._profiles[idx])
        self._active = idx
        self._update_phase_info()

    def _update_phase_info(self):
        if not self.chk_phase2.isChecked():
            self.phase_info.setText("")
        elif self._active == 1:
            self.phase_info.setText(tr("Édition du profil GAN", "Editing the GAN profile"))
            self.phase_info.setStyleSheet("color:#e67e22; font-size:11px; font-weight:700;")
        else:
            self.phase_info.setText(tr("Édition du profil PSNR", "Editing the PSNR profile"))
            self.phase_info.setStyleSheet("color:#2ecc71; font-size:11px; font-weight:700;")

    def reset_phase(self):
        """Back to a single PSNR profile (called before loading a config)."""
        if self._active == 1:
            self.phase_combo.blockSignals(True); self.phase_combo.setCurrentIndex(0)
            self.phase_combo.blockSignals(False)
            if self._profiles[0] is not None:
                self._restore(self._profiles[0])
            self._active = 0
        self._profiles = [None, None]
        self.chk_phase2.setChecked(False)
        self.ramp_edit.setText("15000")

    def apply_phase2(self, flat2: dict, ramp=None):
        """Load the GAN profile from a flat config (PSNR profile overlaid with deg_phase2)."""
        if self._active == 1:
            self.phase_combo.setCurrentIndex(0)
        snap0 = self._snapshot()
        self.apply_degradations(flat2)
        self._profiles[1] = self._snapshot()
        self._restore(snap0)
        self._profiles[0] = None
        self.chk_phase2.setChecked(True)
        if ramp is not None:
            self.ramp_edit.setText(str(int(float(ramp))))

    def apply_phase2_snapshot(self, snap: dict, ramp=None, on=True):
        """Restore the GAN profile from the UI sidecar (exact widget values)."""
        if self._active == 1:
            self.phase_combo.setCurrentIndex(0)
        self._profiles[1] = snap
        self.chk_phase2.setChecked(bool(on))
        if ramp is not None:
            self.ramp_edit.setText(str(ramp))

    def phase2_state(self):
        """(on, ramp, GAN-profile snapshot or None) for the UI sidecar."""
        snap = self._snapshot() if self._active == 1 else self._profiles[1]
        return self.chk_phase2.isChecked(), self.ramp_edit.text(), snap

    def collect_phase2(self):
        """GAN-profile values (UI keys), or None when no separate GAN profile is set."""
        if not self.chk_phase2.isChecked():
            return None
        snap = self._snapshot() if self._active == 1 else self._profiles[1]
        return dict(snap["v"]) if snap else None

    def phase2_ramp(self) -> int:
        try:
            return max(0, int(float(self.ramp_edit.text())))
        except ValueError:
            return 15000

    # ── tab + column builders ─────────────────────────────────────────────────
    def _build_tab(self, cols) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(4, 4, 4, 4); h.setSpacing(8)
        h.addWidget(self._column(cols.get("left", [])), 1)
        h.addWidget(self._column(cols.get("right", [])), 1)
        return w

    def _column(self, specs) -> QScrollArea:
        inner = QWidget(); inner.setObjectName("degcol")
        v = QVBoxLayout(inner); v.setContentsMargins(6, 6, 6, 6); v.setSpacing(3)
        for spec in specs:
            kind = spec[0]
            if kind == "h":
                lbl = QLabel(spec[1]); lbl.setObjectName("h2"); v.addWidget(lbl)
            elif kind == "s":
                v.addLayout(self._slider_ctl(*spec[1:]))
            elif kind == "r":
                v.addLayout(self._range_ctl(*spec[1:]))
            elif kind == "e":
                v.addLayout(self._entry_ctl(*spec[1:]))
            elif kind == "c":
                v.addLayout(self._combo_ctl(*spec[1:]))
            elif kind == "b":
                v.addLayout(self._bool_ctl(*spec[1:]))
        v.addStretch(1)
        s = QScrollArea(); s.setWidgetResizable(True); s.setFrameShape(QFrame.NoFrame); s.setWidget(inner)
        return s

    def _chk(self, label, key) -> QCheckBox:
        chk = QCheckBox(label); chk.setChecked(True); self._checks[key] = chk
        try:
            from qt_ui.i18n import tip_for
            t = tip_for(key)
            if t:
                chk.setToolTip(t)
        except Exception:
            pass
        return chk

    def _slider_ctl(self, label, key, default, smax, step):
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(self._chk(label, key))
        steps = max(1, round(smax / step))
        sld = QSlider(Qt.Horizontal); sld.setRange(0, steps)
        sld.setValue(round(float(default) / step))
        row.addWidget(sld, 1)
        e = QLineEdit(self._fmt(default, step)); e.setFixedWidth(56); row.addWidget(e)
        is_int = step >= 1

        def on_sld(i):
            val = i * step
            e.blockSignals(True); e.setText(str(int(val)) if is_int else f"{val:.2f}"); e.blockSignals(False)

        def on_entry():
            try:
                sld.blockSignals(True); sld.setValue(round(float(e.text()) / step)); sld.blockSignals(False)
            except ValueError:
                pass
        sld.valueChanged.connect(on_sld); e.editingFinished.connect(on_entry)
        self._sliders[key] = (sld, e, step)
        self._getters[key] = lambda e=e: e.text()
        self._setters[key] = lambda v, k=key: self._set_slider(k, v)
        return row

    def _range_ctl(self, label, key, default):
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(self._chk(label, key)); row.addStretch(1)
        lo, _, hi = default.partition(",")
        e_min = QLineEdit(lo.strip()); e_min.setFixedWidth(50); row.addWidget(e_min)
        arrow = QLabel("→"); arrow.setStyleSheet("color:#888;"); row.addWidget(arrow)
        e_max = QLineEdit(hi.strip()); e_max.setFixedWidth(50); row.addWidget(e_max)
        self._getters[key] = lambda a=e_min, b=e_max: f"{a.text()},{b.text()}"

        def _set_range(v, a=e_min, b=e_max):
            s = str(v).strip().strip("[]() ")
            parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
            if len(parts) >= 2:
                a.setText(parts[0]); b.setText(parts[1])
        self._setters[key] = _set_range
        return row

    def _entry_ctl(self, label, key, default):
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(self._chk(label, key)); row.addStretch(1)
        e = QLineEdit(str(default)); e.setFixedWidth(110); row.addWidget(e)
        self._getters[key] = lambda e=e: e.text()
        self._setters[key] = lambda v, e=e: e.setText(str(v))
        return row

    def _combo_ctl(self, label, key, choices, default):
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel(label)); row.addStretch(1)
        cb = QComboBox(); cb.addItems(choices); cb.setCurrentText(default); cb.setFixedWidth(150)
        row.addWidget(cb)
        self._getters[key] = lambda cb=cb: cb.currentText()
        self._setters[key] = lambda v, cb=cb: cb.setCurrentText(str(v))
        return row

    def _bool_ctl(self, label, key, default=False):
        """A plain on/off control (the checkbox IS the value) — for booleans like lq_usm
        that aren't probabilities. Getter returns 'true'/'false'."""
        row = QHBoxLayout(); row.setSpacing(6)
        chk = QCheckBox(label)
        chk.setChecked(bool(default))
        self._checks[key] = chk
        try:
            from qt_ui.i18n import tip_for
            t = tip_for(key)
            if t:
                chk.setToolTip(t)
        except Exception:
            pass
        row.addWidget(chk); row.addStretch(1)
        self._getters[key] = lambda c=chk: "true" if c.isChecked() else "false"
        self._setters[key] = lambda v, c=chk: c.setChecked(str(v).lower() in ("true", "1", "yes", "on"))
        self._bool_keys = getattr(self, "_bool_keys", set()); self._bool_keys.add(key)
        return row

    def apply_degradations(self, flat: dict):
        """Restore every degradation control from a loaded config (called by Config load).
        A degradation's enable checkbox follows its *_prob being > 0."""
        for key, setter in self._setters.items():
            if key in flat and flat[key] is not None and str(flat[key]) != "":
                try:
                    setter(flat[key])
                except Exception:
                    pass
        _bool_keys = getattr(self, "_bool_keys", set())
        for key, chk in self._checks.items():
            if key in _bool_keys:
                # on/off control: the checkbox is the value itself, not a prob gate
                if key in flat:
                    chk.setChecked(str(flat[key]).lower() in ("true", "1", "yes", "on"))
                continue
            prob = flat.get(key)
            if prob is None and not key.endswith("_prob"):
                prob = flat.get(key + "_prob")
            try:
                chk.setChecked(float(prob) > 0)
            except (TypeError, ValueError):
                pass

    @staticmethod
    def _fmt(v, step):
        return str(int(v)) if step >= 1 else f"{float(v):.2f}"

    # ── preset / severity logic ───────────────────────────────────────────────
    def _set_slider(self, key, value):
        if key in self._sliders:
            sld, e, step = self._sliders[key]
            e.setText(self._fmt(value, step))
            sld.blockSignals(True); sld.setValue(round(float(value) / step)); sld.blockSignals(False)

    def _apply_preset(self, name):
        """Fill every Stage1/2 + custom-degradation prob AND range from the preset
        (exact port of CTk apply_deg_preset), gated by the Deg.Custom/VHS checkboxes."""
        use_custom = self.chk_custom.isChecked()
        use_vhs = self.chk_vhs.isChecked()
        for key, val in _PRESETS.get(name, {}).items():
            if key in _CUSTOM_KEYS and not use_custom:
                continue
            if key in _VHS_KEYS and not use_vhs:
                continue
            setter = self._setters.get(key)
            if setter is None:
                continue
            try:
                setter(val)
            except Exception:
                pass
        flags = []
        if not use_custom:
            flags.append(tr("sans custom", "no custom"))
        if not use_vhs:
            flags.append(tr("sans VHS", "no VHS"))
        suffix = f" ({', '.join(flags)})" if flags else ""
        if hasattr(self, "deg_info_lbl"):
            self.deg_info_lbl.setText(tr(f"Preset : {name}{suffix}", f"Preset: {name}{suffix}"))

    def _apply_severity(self, level):
        """Fill every custom-degradation probability with its own per-key severity
        curve (exact port of CTk _apply_severity_preset's _T table) — not a flat value."""
        idx = _SEVERITY_IDX.get(level, 0)
        for key, curve in _SEVERITY_CURVES.items():
            setter = self._setters.get(key)
            if setter is None:
                continue
            try:
                setter(curve[idx])
            except Exception:
                pass

    # ── live preview ──────────────────────────────────────────────────────────
    def _preview_panel(self) -> QWidget:
        box = QFrame(); box.setObjectName("degprev")
        v = QVBoxLayout(box); v.setContentsMargins(12, 10, 12, 12); v.setSpacing(6)
        t = QLabel(tr("🔍 Aperçu en direct des dégradations", "🔍 Live degradation preview"))
        t.setStyleSheet("color:#9b59b6; font-weight:700;"); v.addWidget(t)
        v.addWidget(QLabel(tr("Charge une image source, applique les paramètres ci-dessus pour voir le résultat.", "Loads a source image, applies the settings above to see the result.")))

        ctl = QHBoxLayout(); ctl.setSpacing(6); v.addLayout(ctl)
        _H = 30  # uniform control height for the whole preview toolbar
        ctl.addWidget(QLabel(tr("Image :", "Image:")))
        self.src = QLineEdit(); self.src.setMinimumWidth(360); self.src.setFixedHeight(_H)
        last = self.settings.get("last_preview_image", "")
        if last and os.path.isfile(last):
            self.src.setText(last)
        ctl.addWidget(self.src, 1)
        b = QPushButton("…"); b.setFixedWidth(48); b.setFixedHeight(_H); b.clicked.connect(self._browse); ctl.addWidget(b)
        ctl.addWidget(QLabel(tr("Scale :", "Scale:")))
        self.scale = QComboBox(); self.scale.addItems(["1", "2", "3", "4", "8"]); self.scale.setCurrentText("4")
        self.scale.setFixedWidth(60); self.scale.setFixedHeight(_H); ctl.addWidget(self.scale)
        gen = QPushButton(tr("🎲 Régénérer", "🎲 Regenerate")); gen.setFixedHeight(_H); gen.clicked.connect(self._refresh); ctl.addWidget(gen)
        zoom = QPushButton("🔍 Zoom"); zoom.setObjectName("ghost"); zoom.setFixedHeight(_H); zoom.clicked.connect(self._zoom)
        ctl.addWidget(zoom)
        ctl.addWidget(QLabel(tr("Var :", "Var:")))
        self.variants = QLineEdit("1"); self.variants.setFixedWidth(40); self.variants.setFixedHeight(_H); ctl.addWidget(self.variants)
        gtv = QPushButton(tr("📁 Tester GT Val", "📁 Test GT Val")); gtv.setObjectName("ghost"); gtv.setFixedHeight(_H); gtv.clicked.connect(self._test_gt_val)
        ctl.addWidget(gtv)
        ctl.addStretch(1)

        imgs = QHBoxLayout(); imgs.setSpacing(14); v.addLayout(imgs, 1)
        self.hq_lbl = QLabel(tr("(Aucune image)", "(No image)")); self.hq_lbl.setAlignment(Qt.AlignCenter)
        self.hq_lbl.setMinimumHeight(220); self.hq_lbl.setStyleSheet("color:#666;")
        self.lq_lbl = QLabel(""); self.lq_lbl.setAlignment(Qt.AlignCenter)
        self.lq_lbl.setMinimumHeight(220)
        self.hq_res = QLabel(""); self.lq_res = QLabel("")
        for col, title, lbl, res in (("#2ecc71", tr("Source (HQ)", "Source (HQ)"), self.hq_lbl, self.hq_res),
                                     ("#e67e22", tr("Résultat OTF (LQ)", "OTF Result (LQ)"), self.lq_lbl, self.lq_res)):
            c = QVBoxLayout()
            tt = QLabel(title); tt.setStyleSheet(f"color:{col}; font-weight:700;"); tt.setAlignment(Qt.AlignCenter)
            res.setStyleSheet("color:#888; font-size:11px;"); res.setAlignment(Qt.AlignCenter)
            c.addWidget(tt); c.addWidget(res); c.addWidget(lbl, 1)
            imgs.addLayout(c, 1)
        self.log_lbl = QLabel(""); self.log_lbl.setStyleSheet("color:#888; font-size:11px;")
        self.log_lbl.setAlignment(Qt.AlignCenter)
        v.addWidget(self.log_lbl)
        return box

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Image source", "Source image"), "",
                                           tr("Images (*.png *.jpg *.jpeg *.webp *.bmp);;Tous (*.*)", "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All (*.*)"))
        if p:
            self.src.setText(p); self.settings.set("last_preview_image", p); self._refresh()

    def _collect(self) -> dict:
        cfg = {}
        for k, getter in self._getters.items():
            val = getter()
            try:
                cfg[k] = float(val)
            except (ValueError, TypeError):
                cfg[k] = val
        return cfg

    def _refresh(self):
        src = self.src.text().strip()
        if not src or not os.path.isfile(src):
            QMessageBox.warning(self, tr("Aperçu", "Preview"), tr("Sélectionnez d'abord une image source valide.", "Select a valid source image first."))
            return
        # Universal SR Studio fix: if a previous preview worker is still
        # running (e.g. rapid slider tweaks while calibrating), reassigning
        # self.worker below would drop the only Python reference to that
        # still-running QThread — it gets garbage-collected mid-flight,
        # which is a fatal, traceback-less Qt crash (BEX64 in Qt6Core.dll,
        # confirmed via Windows Event Log). Ignore the request instead;
        # the pending preview will still land via _show/_fail once done.
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            return
        try:
            scale = int(self.scale.currentText())
        except ValueError:
            scale = 4
        self.log_lbl.setText(tr("⏳ Génération…", "⏳ Generating…"))
        self.worker = _PreviewWorker(src, self._collect(), scale)
        self.worker.done.connect(self._show)
        self.worker.fail.connect(lambda e: self.log_lbl.setText(f"❌ {e}"))
        self.worker.start()

    def _show(self, hq, lq, log):
        from PIL import Image
        self._last_hq, self._last_lq = hq, lq
        target = (self.hq_lbl.width() or 360, self.hq_lbl.height() or 220)
        hq_t = hq.copy(); hq_t.thumbnail(target, Image.LANCZOS)
        lq_t = lq.copy().resize(hq_t.size, Image.NEAREST)
        self.hq_lbl.setPixmap(pil_to_pixmap(hq_t))
        self.lq_lbl.setPixmap(pil_to_pixmap(lq_t))
        self.hq_res.setText(f"{hq.size[0]}×{hq.size[1]}")
        self.lq_res.setText(f"{lq.size[0]}×{lq.size[1]}")
        self.log_lbl.setText(log[:200])
        self.worker = None

    def _zoom(self):
        if self._last_lq is None:
            QMessageBox.information(self, "Zoom", tr("Génère d'abord un aperçu (Régénérer).", "Generate a preview first (Regenerate).")); return
        w, h = self._last_lq.size
        dlg = QDialog(self)
        dlg.setWindowTitle(tr(f"Zoom — LQ après OTF  ({w}×{h})   molette = zoom · glisser = pan · double-clic = reset",
                              f"Zoom — LQ after OTF  ({w}×{h})   wheel = zoom · drag = pan · double-click = reset"))
        dlg.resize(min(1100, w + 80), min(720, h + 120))
        lay = QVBoxLayout(dlg); lay.setContentsMargins(0, 0, 0, 0)
        view = _ZoomView(pil_to_pixmap(self._last_lq))
        lay.addWidget(view)
        dlg.show()
        view.fitInView(view._item, Qt.KeepAspectRatio)
        dlg.exec()

    def _test_gt_val(self):
        # Use the configured Val GT folder directly; only ask if unset
        folder = self.settings.get("ds_val_gt", "") or ""
        if not folder or not os.path.isdir(folder):
            folder = QFileDialog.getExistingDirectory(self, tr("Dossier GT (validation)", "GT folder (validation)"))
        if not folder:
            return
        try:
            scale = int(self.scale.currentText())
        except ValueError:
            scale = 4
        try:
            variants = max(1, int(self.variants.text()))
        except ValueError:
            variants = 1
        self.log_lbl.setText(tr("⏳ Génération batch GT Val…", "⏳ GT Val batch generation…"))
        self.worker = _BatchWorker(folder, self._collect(), scale, variants)
        self.worker.progress.connect(lambda s: self.log_lbl.setText(s))
        self.worker.done.connect(self._batch_done)
        self.worker.start()

    def _batch_done(self, n, out):
        self.log_lbl.setText(tr(f"✅ {n} LQ générés → {out}", f"✅ {n} LQ generated → {out}"))
        self.worker = None
        try:
            os.startfile(out)  # open the output folder (Windows)
        except Exception:
            pass

    # ── exposed for config save ────────────────────────────────────────────────
    def collect_degradations(self) -> dict:
        """PSNR-profile values (the only profile when no separate GAN profile is set)."""
        if self._active == 1 and self._profiles[0] is not None:
            return dict(self._profiles[0]["v"])
        return self._collect()
