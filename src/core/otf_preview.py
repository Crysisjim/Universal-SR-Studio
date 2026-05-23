"""
otf_preview.py — Apply OTF degradations to sample images for preview.

Mimics the NeoSR/Redux degradation pipeline (in pure PIL/numpy, no torch needed)
so the user can see what the LQ images will look like before launching training.
"""
import os
import io
import random
from typing import Tuple

try:
    from PIL import Image, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def _parse_range(val, default=(0.0, 1.0)) -> Tuple[float, float]:
    """Parse '[a, b]' or [a, b] into a tuple."""
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return (float(val[0]), float(val[1]))
    if isinstance(val, str):
        v = val.strip().lstrip("[").rstrip("]").replace(" ", "")
        try:
            parts = [float(x) for x in v.split(",") if x]
            if len(parts) >= 2:
                return (parts[0], parts[1])
        except Exception:
            pass
    return default


def apply_blur(img, sigma_range=(0.2, 3.0), kernel_size=21):
    """Apply gaussian blur."""
    sigma = random.uniform(*sigma_range)
    if sigma < 0.01:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_noise(img, noise_range=(1, 30), gray_prob=0.4):
    """Apply gaussian noise."""
    if not NUMPY_AVAILABLE:
        return img
    arr = np.array(img).astype(np.float32)
    sigma = random.uniform(*noise_range)
    if random.random() < gray_prob:
        # Gray noise — same value across channels
        noise = np.random.normal(0, sigma, arr.shape[:2])
        for c in range(arr.shape[2] if arr.ndim == 3 else 1):
            arr[..., c] = arr[..., c] + noise
    else:
        noise = np.random.normal(0, sigma, arr.shape)
        arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_jpeg(img, quality_range=(30, 95)):
    """Apply JPEG compression."""
    q = random.randint(int(quality_range[0]), int(quality_range[1]))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_resize(img, target_size, scale_range=(0.5, 1.5)):
    """Resize down to scale*target_size then back up."""
    w, h = img.size
    s = random.uniform(*scale_range)
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    method = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
    return img.resize((new_w, new_h), method)


def apply_posterize(img, bits_range=(3, 6)):
    """Reduce per-channel bit depth."""
    from PIL import ImageOps
    bits = random.randint(int(bits_range[0]), int(bits_range[1]))
    return ImageOps.posterize(img, bits)


def apply_banding(img, levels_range=(16, 64)):
    """Quantize to a small palette."""
    levels = random.randint(int(levels_range[0]), int(levels_range[1]))
    return img.quantize(colors=levels, method=Image.Quantize.FASTOCTREE).convert("RGB")


def apply_chroma_subsampling(img):
    """Simulate 4:2:0 chroma subsampling (PIL YCbCr round-trip)."""
    if not NUMPY_AVAILABLE:
        return img
    ycbcr = img.convert("YCbCr")
    arr = np.array(ycbcr).astype(np.float32)
    h, w = arr.shape[:2]
    for ch in (1, 2):  # Cb, Cr
        ch_img = Image.fromarray(arr[:, :, ch].astype(np.uint8))
        dw, dh = max(1, w // 2), max(1, h // 2)
        ch_img = ch_img.resize((dw, dh), Image.BOX).resize((w, h), Image.NEAREST)
        arr[:, :, ch] = np.array(ch_img).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "YCbCr").convert("RGB")


def apply_chromatic_aberration(img, shift_range=(1, 5)):
    """Shift R and B channels in opposite horizontal directions."""
    if not NUMPY_AVAILABLE:
        return img
    shift = random.randint(int(shift_range[0]), int(shift_range[1]))
    arr = np.array(img).copy()
    arr[:, :, 0] = np.roll(arr[:, :, 0], shift, axis=1)   # R right
    arr[:, :, 2] = np.roll(arr[:, :, 2], -shift, axis=1)  # B left
    return Image.fromarray(arr)


def apply_halation(img, strength_range=(0.05, 0.3)):
    """Film halation: bright areas bleed warm glow."""
    if not NUMPY_AVAILABLE:
        return img
    strength = random.uniform(*strength_range)
    arr = np.array(img).astype(np.float32)
    lum = arr.mean(axis=2)
    bright_mask = (lum > 0.65 * 255).astype(np.float32)
    warm = arr.copy()
    warm[:, :, 2] *= 0.5  # reduce blue in glow
    for c in range(3):
        warm[:, :, c] *= bright_mask
    warm_img = Image.fromarray(np.clip(warm, 0, 255).astype(np.uint8))
    glow = warm_img.filter(ImageFilter.GaussianBlur(radius=11))
    glow_arr = np.array(glow).astype(np.float32)
    result = np.clip(arr + glow_arr * strength, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def apply_salt_pepper(img, amount_range=(0.001, 0.05)):
    """Random black and white pixels."""
    if not NUMPY_AVAILABLE:
        return img
    amount = random.uniform(*amount_range)
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    n_salt = int(h * w * amount * 0.5)
    n_pepper = int(h * w * amount * 0.5)
    coords = [np.random.randint(0, dim, n_salt) for dim in (h, w)]
    arr[coords[0], coords[1], :] = 255
    coords = [np.random.randint(0, dim, n_pepper) for dim in (h, w)]
    arr[coords[0], coords[1], :] = 0
    return Image.fromarray(arr)


def apply_vhs(img, strength_range=(0.1, 0.5)):
    """VHS/analog artifacts: chroma bleed + scanline dropout."""
    if not NUMPY_AVAILABLE:
        return img
    strength = random.uniform(*strength_range)
    arr = np.array(img).copy().astype(np.float32)
    w = arr.shape[1]
    h = arr.shape[0]
    shift = max(1, int(w * 0.015 * strength))
    arr[:, :, 1] = np.roll(arr[:, :, 1], shift, axis=1)
    arr[:, :, 2] = np.roll(arr[:, :, 2], shift * 2, axis=1)
    dropout_prob = strength * 0.25
    for row in range(h):
        if random.random() < dropout_prob:
            arr[row, :, :] *= 0.4
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ─── Custom 2 degradations ────────────────────────────────────────────────────

def apply_aliasing_pil(img, scale_range=(0.5, 0.85)):
    """Nearest-neighbor downscale+upscale → staircase aliasing on diagonal lines."""
    scale = random.uniform(*scale_range)
    w, h = img.size
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = img.resize((sw, sh), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def apply_interlace_weave_pil(img, strength_range=(0.5, 1.0)):
    """Weave interlacing: replace odd scanlines with vertically-shifted field → comb teeth."""
    if not NUMPY_AVAILABLE:
        return img
    strength = random.uniform(*strength_range)
    arr = np.array(img).astype(np.float32)
    shifted = np.roll(arr, 1, axis=0)
    result = arr.copy()
    result[1::2] = arr[1::2] * (1 - strength) + shifted[1::2] * strength
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def apply_interlace_flicker_pil(img, strength_range=(0.1, 0.4)):
    """Field flicker: alternate lines brighter / darker (CRT 50/60 Hz artifact)."""
    if not NUMPY_AVAILABLE:
        return img
    strength = random.uniform(*strength_range)
    arr = np.array(img).astype(np.float32)
    arr[0::2] = arr[0::2] * (1.0 + strength)
    arr[1::2] = arr[1::2] * (1.0 - strength)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_interlace_blend_pil(img, strength_range=(0.3, 1.0)):
    """Field blending: ghosting between fields (bob deinterlace artifact)."""
    if not NUMPY_AVAILABLE:
        return img
    strength = random.uniform(*strength_range)
    shift = random.choice([1, 2, 3])
    arr = np.array(img).astype(np.float32)
    shifted = np.roll(arr, shift, axis=0)
    result = arr * (1.0 - strength * 0.4) + shifted * (strength * 0.4)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def apply_film_grain_pil(img, strength_range=(0.03, 0.12), size_range=(1, 2)):
    """Luminance-dependent structured film grain (peaks at midtones)."""
    if not NUMPY_AVAILABLE:
        return img
    strength = random.uniform(*strength_range)
    grain_size = random.randint(int(size_range[0]), max(int(size_range[0]), int(size_range[1])))
    arr = np.array(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    grain_weight = 1.0 - (2.0 * luma - 1.0) ** 2
    if grain_size > 1:
        nh, nw = max(1, h // grain_size), max(1, w // grain_size)
        noise_s = np.random.normal(0, 1, (nh, nw, 3)).astype(np.float32)
        # Upsample via PIL
        noise_img = Image.fromarray(
            np.clip((noise_s * 127.5 + 127.5), 0, 255).astype(np.uint8)
        ).resize((w, h), Image.BILINEAR)
        noise = (np.array(noise_img).astype(np.float32) - 127.5) / 127.5
    else:
        noise = np.random.normal(0, 1, (h, w, 3)).astype(np.float32)
    result = arr + noise * grain_weight[..., np.newaxis] * strength
    return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))


def apply_oversharpening_pil(img, strength_range=(0.5, 2.0)):
    """Unsharp mask halos (USM over-sharpening artifact)."""
    strength = random.uniform(*strength_range)
    radius = 1.0
    percent = int(strength * 150)
    threshold = 2
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def apply_scanlines_pil(img, spacing_range=(2, 4), strength_range=(0.2, 0.5)):
    """CRT scanlines: darken every N-th horizontal line."""
    if not NUMPY_AVAILABLE:
        return img
    spacing = random.randint(int(spacing_range[0]), max(int(spacing_range[0]), int(spacing_range[1])))
    strength = random.uniform(*strength_range)
    arr = np.array(img).astype(np.float32)
    arr[::spacing] = arr[::spacing] * (1.0 - strength)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ─── Custom 3 degradations ────────────────────────────────────────────────────

def apply_screentone(img, dot_size: int = 8, angle_deg: float = 45.0,
                     dot_type: str = "circle", color_space: str = "rgb"):
    """Halftone screentone pattern (manga/print style).

    Points sombres = large, points clairs = petits.
    dot_type : circle | ellipse | square | cross
    color_space : rgb | gray (gray → luminance uniquement)
    """
    if not NUMPY_AVAILABLE:
        return img
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    dot_size = max(2, int(dot_size))
    angle = float(angle_deg) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xr = xs * cos_a + ys * sin_a
    yr = -xs * sin_a + ys * cos_a

    # Position normalisée dans la cellule [-0.5, 0.5)
    cx = (xr % dot_size) / dot_size - 0.5
    cy = (yr % dot_size) / dot_size - 0.5

    if dot_type in ("circle", "ellipse"):
        dist = np.sqrt(cx ** 2 + cy ** 2)
    elif dot_type == "square":
        dist = np.maximum(np.abs(cx), np.abs(cy))
    elif dot_type == "cross":
        dist = np.minimum(np.abs(cx), np.abs(cy))
    else:
        dist = np.sqrt(cx ** 2 + cy ** 2)

    luma = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) / 255.0
    # Zones sombres → gros points (radius grand)
    radius = (1.0 - luma) * 0.46
    mask = (dist < radius).astype(np.float32)[..., np.newaxis]

    if color_space == "gray":
        result = arr * (1.0 - mask)
    else:
        result = arr * (1.0 - mask)  # noircit les points

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def apply_dithering(img, n_colors: int = 4, dither_type: str = "floyd_steinberg"):
    """Réduction de couleurs avec algorithme de tramage.

    dither_type : floyd_steinberg | ordered | quantize | atkinson | jarvis_judice | stucki
    """
    n_colors = max(2, int(n_colors))
    if dither_type == "ordered":
        # Tramage ordonné : quantize sans diffusion PIL
        return img.quantize(colors=n_colors,
                            dither=getattr(Image, "Dither", None) and
                                   Image.Dither.NONE or 0).convert("RGB")
    elif dither_type in ("floyd_steinberg", "atkinson", "jarvis_judice", "stucki"):
        return img.quantize(colors=n_colors,
                            dither=getattr(Image, "Dither", None) and
                                   Image.Dither.FLOYDSTEINBERG or 1).convert("RGB")
    else:
        return img.quantize(colors=n_colors,
                            dither=getattr(Image, "Dither", None) and
                                   Image.Dither.NONE or 0).convert("RGB")


def apply_pixelate(img, block_size: int = 8):
    """Pixelisation par sous-échantillonnage nearest-neighbor."""
    w, h = img.size
    block_size = max(2, int(block_size))
    small = img.resize((max(1, w // block_size), max(1, h // block_size)), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def apply_sinusoidal(img, shape_range=(100.0, 600.0), alpha_range=(0.1, 0.4),
                     bias_range=(0.8, 1.2), orientation: str = "aléatoire"):
    """Ondulation sinusoïdale de luminosité (wtp-style).

    shape  = période en pixels (100–600)
    alpha  = amplitude en fraction de 255 (0.1–0.4)
    bias   = déphasage [0, 2π] via bruit aléatoire sur la plage
    orientation : horizontal | vertical | aléatoire
    """
    if not NUMPY_AVAILABLE:
        return img
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    period = random.uniform(float(shape_range[0]), float(shape_range[1]))
    alpha  = random.uniform(float(alpha_range[0]), float(alpha_range[1]))
    bias   = random.uniform(float(bias_range[0]), float(bias_range[1])) * np.pi

    if orientation == "aléatoire":
        orientation = random.choice(["horizontal", "vertical"])

    freq = 2.0 * np.pi / max(period, 1.0)
    if orientation == "horizontal":
        wave = np.sin(freq * np.arange(w) + bias)          # (W,)
        wave = wave[np.newaxis, :, np.newaxis]              # (1, W, 1)
    else:
        wave = np.sin(freq * np.arange(h) + bias)          # (H,)
        wave = wave[:, np.newaxis, np.newaxis]              # (H, 1, 1)

    result = arr + wave * alpha * 255.0 * 0.5
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def apply_subsampling_wtp(img, subsampling_format: str = "4:2:0",
                          yuv_matrix: str = "601"):
    """Sous-échantillonnage chroma configurable (wtp-style).

    yuv_matrix 601/709/2020 : change les coefficients RGB→YCbCr.
    """
    if not NUMPY_AVAILABLE:
        return img

    # Matrices RGB→YCbCr (BT.601, BT.709, BT.2020)
    matrices = {
        "601":  [0.299,  0.587,  0.114],
        "709":  [0.2126, 0.7152, 0.0722],
        "2020": [0.2627, 0.6780, 0.0593],
    }
    kr, kg, kb = matrices.get(str(yuv_matrix), matrices["601"])

    # Facteurs H×V par canal (Cb, Cr)
    fmt_map = {
        "4:4:4": [(1, 1), (1, 1)],
        "4:2:2": [(2, 1), (2, 1)],
        "4:2:0": [(2, 2), (2, 2)],
        "4:1:1": [(4, 1), (4, 1)],
    }
    factors = fmt_map.get(subsampling_format, [(2, 2), (2, 2)])

    # Conversion YCbCr manuelle si matrice non-601
    if yuv_matrix in ("709", "2020"):
        arr = np.array(img).astype(np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        y  = kr * r + kg * g + kb * b
        cb = (b - y) / (2 * (1 - kb)) + 128
        cr = (r - y) / (2 * (1 - kr)) + 128
        ycbcr = np.stack([y, cb, cr], axis=2)
    else:
        ycbcr = np.array(img.convert("YCbCr")).astype(np.float32)

    h, w = ycbcr.shape[:2]
    for ci, (fw, fh) in zip([1, 2], factors):
        if fw == 1 and fh == 1:
            continue
        ch_img = Image.fromarray(ycbcr[:, :, ci].astype(np.uint8))
        dw, dh = max(1, w // fw), max(1, h // fh)
        ch_img = ch_img.resize((dw, dh), Image.BOX).resize((w, h), Image.NEAREST)
        ycbcr[:, :, ci] = np.array(ch_img).astype(np.float32)

    if yuv_matrix in ("709", "2020"):
        y, cb, cr = ycbcr[..., 0], ycbcr[..., 1] - 128, ycbcr[..., 2] - 128
        r = np.clip(y + 2 * (1 - kr) * cr, 0, 255)
        b = np.clip(y + 2 * (1 - kb) * cb, 0, 255)
        g = np.clip((y - kr * r - kb * b) / max(kg, 1e-6), 0, 255)
        result = np.stack([r, g, b], axis=2)
        return Image.fromarray(result.astype(np.uint8))
    else:
        return Image.fromarray(
            np.clip(ycbcr, 0, 255).astype(np.uint8), "YCbCr").convert("RGB")


# ─── Custom 4 degradations ────────────────────────────────────────────────────

def apply_color_levels(img, high_range=(220.0, 255.0), low_range=(0.0, 35.0),
                       gamma_range=(0.7, 1.5)):
    """Ajustement niveaux de couleur (style Photoshop Levels).

    Réduit la plage de sortie (Output High/Low) + correction gamma.
    """
    if not NUMPY_AVAILABLE:
        return img
    arr = np.array(img).astype(np.float32)
    out_high = random.uniform(float(high_range[0]), float(high_range[1]))
    out_low  = random.uniform(float(low_range[0]),  float(low_range[1]))
    gamma    = random.uniform(float(gamma_range[0]), float(gamma_range[1]))

    # Gamma sur [0,1]
    arr_n = arr / 255.0
    if abs(gamma - 1.0) > 1e-4:
        arr_n = np.power(np.clip(arr_n, 0, 1), 1.0 / gamma)
    # Sortie dans [out_low, out_high]
    arr_n = arr_n * (out_high - out_low) + out_low
    return Image.fromarray(np.clip(arr_n, 0, 255).astype(np.uint8))


def apply_wtp_halo(img, strength_range=(0.1, 0.5), radius_range=(3.0, 12.0)):
    """Halo / ringing artifact autour des bords (wtp-style).

    Amplifie la différence entre l'image et sa version floutée.
    """
    if not NUMPY_AVAILABLE:
        return img
    strength = random.uniform(float(strength_range[0]), float(strength_range[1]))
    radius   = random.uniform(float(radius_range[0]),   float(radius_range[1]))
    blurred  = img.filter(ImageFilter.GaussianBlur(radius=max(1, int(radius))))
    arr  = np.array(img).astype(np.float32)
    blur = np.array(blurred).astype(np.float32)
    result = arr + (arr - blur) * strength
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def apply_saturation_adj(img, factor_range=(0.3, 1.8)):
    """Ajustement de la saturation via PIL ImageEnhance."""
    from PIL import ImageEnhance
    factor = random.uniform(float(factor_range[0]), float(factor_range[1]))
    return ImageEnhance.Color(img).enhance(factor)


def apply_pixel_shift(img, shift_range=(1.0, 8.0), axis: str = "aléatoire"):
    """Décalage de pixels le long d'un axe (glitch effect).

    axis : horizontal | vertical | les deux | aléatoire
    """
    if not NUMPY_AVAILABLE:
        return img
    arr = np.array(img).astype(np.float32)
    shift = int(random.uniform(float(shift_range[0]), float(shift_range[1])))
    if axis == "aléatoire":
        axis = random.choice(["horizontal", "vertical", "les deux"])
    if axis in ("horizontal", "les deux"):
        arr = np.roll(arr, shift, axis=1)
    if axis in ("vertical", "les deux"):
        arr = np.roll(arr, shift, axis=0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_otf_pipeline(img, config: dict, scale: int = 4) -> Tuple:
    """
    Apply the full OTF degradation pipeline (2 stages + final).

    Reads degradations from either the top level OR a nested [degradations] table
    (NeoSR/Real-ESRGAN style). Custom keys (banding/posterize) are at top level.

    Returns (lq_image, applied_steps_log).
    """
    if not PIL_AVAILABLE:
        return img, ["PIL not available"]
    if img.mode != "RGB":
        img = img.convert("RGB")

    # NeoSR/Real-ESRGAN configs nest degradations under [degradations] table.
    # We merge: top-level keys take precedence, then [degradations].
    deg = dict(config.get("degradations") or {})
    # Override with top-level keys if present (USR Studio templates use top-level)
    for k, v in config.items():
        if k != "degradations" and not isinstance(v, dict):
            deg[k] = v

    log = []
    out = img

    # ===== Stage 1 =====
    # NeoSR/Real-ESRGAN style: stage 1 blur is ALWAYS applied (no blur_prob toggle).
    # USR Studio templates may set blur_prob; if absent, default to 1.0 (always blur).
    blur1_prob = float(deg.get("blur_prob", 1.0 if "blur_sigma" in deg else 0.0))
    if random.random() < blur1_prob:
        sigma_range = _parse_range(deg.get("blur_sigma", "[0.2, 3.0]"), (0.2, 3.0))
        out = apply_blur(out, sigma_range)
        log.append(f"Blur1 sigma~U{sigma_range}")

    if random.random() < float(deg.get("gaussian_noise_prob", 0.0)):
        noise_range = _parse_range(deg.get("noise_range", "[1, 30]"), (1, 30))
        gray_prob = float(deg.get("gray_noise_prob", 0.4))
        out = apply_noise(out, noise_range, gray_prob)
        log.append(f"Noise1 sigma~U{noise_range}")

    if random.random() < float(deg.get("jpeg_prob", 0.0)):
        q_range = _parse_range(deg.get("jpeg_range", "[30, 95]"), (30, 95))
        out = apply_jpeg(out, q_range)
        log.append(f"JPEG1 q~U{q_range}")

    # ===== Stage 2 =====
    if random.random() < float(deg.get("second_blur_prob", 0.0)):
        sigma_range = _parse_range(deg.get("blur_sigma2", "[0.2, 1.5]"), (0.2, 1.5))
        out = apply_blur(out, sigma_range)
        log.append(f"Blur2 sigma~U{sigma_range}")

    if random.random() < float(deg.get("gaussian_noise_prob2", 0.0)):
        noise_range = _parse_range(deg.get("noise_range2", "[1, 25]"), (1, 25))
        gray_prob = float(deg.get("gray_noise_prob2", 0.4))
        out = apply_noise(out, noise_range, gray_prob)
        log.append(f"Noise2 sigma~U{noise_range}")

    if random.random() < float(deg.get("jpeg_prob", 0.0)):
        q_range = _parse_range(deg.get("jpeg_range2", "[30, 95]"), (30, 95))
        out = apply_jpeg(out, q_range)
        log.append(f"JPEG2 q~U{q_range}")

    # ===== Custom degradations =====
    if random.random() < float(deg.get("posterize_prob", 0.0)):
        bits_range = _parse_range(deg.get("posterize_bits_range", "[3, 6]"), (3, 6))
        out = apply_posterize(out, bits_range)
        log.append(f"Posterize bits~U{bits_range}")

    if random.random() < float(deg.get("banding_prob", 0.0)):
        lev_range = _parse_range(deg.get("banding_levels_range", "[16, 64]"), (16, 64))
        out = apply_banding(out, lev_range)
        log.append(f"Banding levels~U{lev_range}")

    if random.random() < float(deg.get("chroma_prob", 0.0)):
        out = apply_chroma_subsampling(out)
        log.append("ChromaSub 4:2:0")

    if random.random() < float(deg.get("ca_prob", 0.0)):
        shift_range = _parse_range(deg.get("ca_shift_range", "[1, 5]"), (1, 5))
        out = apply_chromatic_aberration(out, shift_range)
        log.append(f"ChromAber shift~U{shift_range}")

    if random.random() < float(deg.get("halation_prob", 0.0)):
        str_range = _parse_range(deg.get("halation_strength_range", "[0.05, 0.3]"), (0.05, 0.3))
        out = apply_halation(out, str_range)
        log.append(f"Halation str~U{str_range}")

    if random.random() < float(deg.get("salt_pepper_prob", 0.0)):
        amt_range = _parse_range(deg.get("salt_pepper_amount_range", "[0.001, 0.05]"), (0.001, 0.05))
        out = apply_salt_pepper(out, amt_range)
        log.append(f"SaltPepper amt~U{amt_range}")

    if random.random() < float(deg.get("vhs_prob", 0.0)):
        str_range = _parse_range(deg.get("vhs_strength_range", "[0.1, 0.5]"), (0.1, 0.5))
        out = apply_vhs(out, str_range)
        log.append(f"VHS str~U{str_range}")

    # ===== Custom 2 degradations =====
    if random.random() < float(deg.get("aliasing_prob", 0.0)):
        sc_range = _parse_range(deg.get("aliasing_scale_range", "[0.5, 0.85]"), (0.5, 0.85))
        out = apply_aliasing_pil(out, sc_range)
        log.append(f"Aliasing scale~U{sc_range}")

    if random.random() < float(deg.get("interlace_weave_prob", 0.0)):
        st_range = _parse_range(deg.get("interlace_weave_strength_range", "[0.5, 1.0]"), (0.5, 1.0))
        out = apply_interlace_weave_pil(out, st_range)
        log.append(f"Interlace-Weave str~U{st_range}")

    if random.random() < float(deg.get("interlace_flicker_prob", 0.0)):
        st_range = _parse_range(deg.get("interlace_flicker_strength_range", "[0.1, 0.4]"), (0.1, 0.4))
        out = apply_interlace_flicker_pil(out, st_range)
        log.append(f"Interlace-Flicker str~U{st_range}")

    if random.random() < float(deg.get("interlace_blend_prob", 0.0)):
        st_range = _parse_range(deg.get("interlace_blend_strength_range", "[0.3, 1.0]"), (0.3, 1.0))
        out = apply_interlace_blend_pil(out, st_range)
        log.append(f"Interlace-Blend str~U{st_range}")

    if random.random() < float(deg.get("film_grain_prob", 0.0)):
        st_range = _parse_range(deg.get("film_grain_strength_range", "[0.03, 0.12]"), (0.03, 0.12))
        sz_range = _parse_range(deg.get("film_grain_size_range", "[1, 2]"), (1, 2))
        out = apply_film_grain_pil(out, st_range, sz_range)
        log.append(f"FilmGrain str~U{st_range} size~U{sz_range}")

    if random.random() < float(deg.get("oversharp_prob", 0.0)):
        st_range = _parse_range(deg.get("oversharp_strength_range", "[0.5, 2.0]"), (0.5, 2.0))
        out = apply_oversharpening_pil(out, st_range)
        log.append(f"Oversharp str~U{st_range}")

    if random.random() < float(deg.get("scanlines_prob", 0.0)):
        sp_range = _parse_range(deg.get("scanlines_spacing_range", "[2, 4]"), (2, 4))
        st_range = _parse_range(deg.get("scanlines_strength_range", "[0.2, 0.5]"), (0.2, 0.5))
        out = apply_scanlines_pil(out, sp_range, st_range)
        log.append(f"Scanlines sp~U{sp_range} str~U{st_range}")

    # ===== Custom 3 degradations =====
    if random.random() < float(deg.get("screentone_prob", 0.0)):
        ds_range  = _parse_range(deg.get("screentone_dot_size",  "[7, 15]"), (7, 15))
        ang_range = _parse_range(deg.get("screentone_angle",     "[0, 90]"), (0, 90))
        ds  = int(random.uniform(ds_range[0],  ds_range[1]))
        ang = random.uniform(ang_range[0], ang_range[1])
        dtp = str(deg.get("screentone_dot_type",    "circle"))
        csp = str(deg.get("screentone_color_space", "rgb"))
        out = apply_screentone(out, dot_size=ds, angle_deg=ang, dot_type=dtp, color_space=csp)
        log.append(f"Screentone size={ds} angle={ang:.0f}° type={dtp}")

    if random.random() < float(deg.get("dithering_prob", 0.0)):
        nc_range = _parse_range(deg.get("dithering_color_ch", "[2, 8]"), (2, 8))
        n_colors = int(random.uniform(nc_range[0], nc_range[1]))
        dtype    = str(deg.get("dithering_type", "floyd_steinberg"))
        out = apply_dithering(out, n_colors=n_colors, dither_type=dtype)
        log.append(f"Dithering type={dtype} colors={n_colors}")

    if random.random() < float(deg.get("pixelate_prob", 0.0)):
        ps_range = _parse_range(deg.get("pixelate_size", "[2, 16]"), (2, 16))
        bs = int(random.uniform(ps_range[0], ps_range[1]))
        out = apply_pixelate(out, block_size=bs)
        log.append(f"Pixelate block={bs}")

    if random.random() < float(deg.get("sin_prob", 0.0)):
        sh_range  = _parse_range(deg.get("sin_shape",       "[100, 600]"), (100, 600))
        al_range  = _parse_range(deg.get("sin_alpha",       "[0.1, 0.4]"), (0.1, 0.4))
        bi_range  = _parse_range(deg.get("sin_bias",        "[0.8, 1.2]"), (0.8, 1.2))
        orient    = str(deg.get("sin_orientation", "aléatoire"))
        out = apply_sinusoidal(out, shape_range=sh_range, alpha_range=al_range,
                               bias_range=bi_range, orientation=orient)
        log.append(f"Sinusoidal shape~U{sh_range} alpha~U{al_range} orient={orient}")

    if random.random() < float(deg.get("subsampling_prob", 0.0)):
        fmt = str(deg.get("subsampling_format", "4:2:0"))
        yuv = str(deg.get("subsampling_yuv",    "601"))
        out = apply_subsampling_wtp(out, subsampling_format=fmt, yuv_matrix=yuv)
        log.append(f"Subsampling {fmt} YUV-{yuv}")

    # ===== Custom 4 degradations =====
    if random.random() < float(deg.get("color_level_prob", 0.0)):
        hi_range  = _parse_range(deg.get("color_level_high",  "[220, 255]"), (220, 255))
        lo_range  = _parse_range(deg.get("color_level_low",   "[0, 35]"),    (0,   35))
        ga_range  = _parse_range(deg.get("color_level_gamma", "[0.7, 1.5]"), (0.7, 1.5))
        out = apply_color_levels(out, high_range=hi_range, low_range=lo_range,
                                 gamma_range=ga_range)
        log.append(f"ColorLevels hi~U{hi_range} lo~U{lo_range} γ~U{ga_range}")

    if random.random() < float(deg.get("wtp_halo_prob", 0.0)):
        st_range = _parse_range(deg.get("wtp_halo_strength", "[0.1, 0.5]"), (0.1, 0.5))
        ra_range = _parse_range(deg.get("wtp_halo_radius",   "[3, 12]"),    (3,   12))
        out = apply_wtp_halo(out, strength_range=st_range, radius_range=ra_range)
        log.append(f"WTP-Halo str~U{st_range} r~U{ra_range}")

    if random.random() < float(deg.get("saturation_prob", 0.0)):
        sat_range = _parse_range(deg.get("saturation_range", "[0.3, 1.8]"), (0.3, 1.8))
        out = apply_saturation_adj(out, factor_range=sat_range)
        log.append(f"Saturation factor~U{sat_range}")

    if random.random() < float(deg.get("shift_prob", 0.0)):
        sh_range = _parse_range(deg.get("shift_range", "[1, 8]"), (1, 8))
        sh_axis  = str(deg.get("shift_axis", "aléatoire"))
        out = apply_pixel_shift(out, shift_range=sh_range, axis=sh_axis)
        log.append(f"PixelShift range~U{sh_range} axis={sh_axis}")

    # ===== Final downscale to LR size =====
    if scale > 1:
        w, h = img.size
        lr_w, lr_h = max(1, w // scale), max(1, h // scale)
        out = out.resize((lr_w, lr_h), Image.BICUBIC)
    # scale=1 means same-size LR/HQ (e.g. for deband/denoise tasks)

    return out, log


def generate_preview_samples(image_paths: list, config: dict, scale: int = 4,
                              n_samples_per_image: int = 1) -> list:
    """
    Generate preview samples for a list of source images.

    Returns list of dicts: {"hq_path": str, "lq_image": PIL.Image, "log": list}
    """
    samples = []
    for hq_path in image_paths:
        if not os.path.exists(hq_path):
            continue
        try:
            img = Image.open(hq_path).convert("RGB")
            for _ in range(max(1, n_samples_per_image)):
                lq, log = apply_otf_pipeline(img, config, scale)
                samples.append({
                    "hq_path": hq_path,
                    "hq_image": img,
                    "lq_image": lq,
                    "log": log,
                })
        except Exception as e:
            print(f"[OTF Preview] Erreur {hq_path}: {e}")
    return samples
