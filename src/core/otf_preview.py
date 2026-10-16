"""
otf_preview.py — Apply OTF degradations to sample images for preview.

Mimics the NeoSR/Redux degradation pipeline (in pure PIL/numpy, no torch needed)
so the user can see what the LQ images will look like before launching training.
"""
import os
import io
import math
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

try:
    from scipy import special as _scipy_special
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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


def _parse_three(val, default=(0.2, 0.7, 0.1)) -> Tuple[float, float, float]:
    """Parse '[up, down, keep]' or [a, b, c] into a 3-tuple (resize_prob/resize_prob2)."""
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        return (float(val[0]), float(val[1]), float(val[2]))
    if isinstance(val, str):
        v = val.strip().lstrip("[").rstrip("]").replace(" ", "")
        try:
            parts = [float(x) for x in v.split(",") if x]
            if len(parts) >= 3:
                return (parts[0], parts[1], parts[2])
        except Exception:
            pass
    return default


# ─── Faithful blur-kernel synthesis ────────────────────────────────────────────
# Exact (numpy) port of traiNNer-redux's traiNNer/data/degradations.py kernel
# generators — used so the live preview matches what RealESRGANModel.feed_data
# actually convolves onto the LQ during training (sinc/jinc ringing filters +
# iso/aniso/generalized/plateau mixed-gaussian kernels), not just a flat sigma blur.

_KERNEL_LIST = ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"]
_KERNEL_PROB = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]  # fixed mix (config_handler.py hardcoded default)
_KERNEL_SIZES = list(range(5, 18, 2))  # kernel_range [5, 17] expanded, matches the real dataset


def _mesh_grid(kernel_size):
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))).reshape(kernel_size, kernel_size, 2)
    return xy


def _sigma_matrix2(sig_x, sig_y, theta):
    d = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    u = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return u @ d @ u.T


def _bivariate_gaussian(kernel_size, sig_x, sig_y, theta, isotropic=True):
    grid = _mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]]) if isotropic else _sigma_matrix2(sig_x, sig_y, theta)
    inv = np.linalg.inv(sigma_matrix)
    k = np.exp(-0.5 * np.sum(np.dot(grid, inv) * grid, 2))
    return k / np.sum(k)


def _bivariate_generalized_gaussian(kernel_size, sig_x, sig_y, theta, beta, isotropic=True):
    grid = _mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]]) if isotropic else _sigma_matrix2(sig_x, sig_y, theta)
    inv = np.linalg.inv(sigma_matrix)
    k = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inv) * grid, 2), beta))
    return k / np.sum(k)


def _bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, isotropic=True):
    grid = _mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]]) if isotropic else _sigma_matrix2(sig_x, sig_y, theta)
    inv = np.linalg.inv(sigma_matrix)
    k = np.reciprocal(np.power(np.sum(np.dot(grid, inv) * grid, 2), beta) + 1)
    return k / np.sum(k)


def _random_mixed_kernel(kernel_size, sigma_range, betag_range, betap_range):
    """Exact port of random_mixed_kernels: picks iso/aniso/generalized/plateau shape
    per the Studio's fixed kernel_list/kernel_prob mix, using UI sigma/beta ranges."""
    rotation_range = (-math.pi, math.pi)
    sig_x = random.uniform(*sigma_range)
    kernel_type = random.choices(_KERNEL_LIST, _KERNEL_PROB)[0]
    if kernel_type == "iso":
        k = _bivariate_gaussian(kernel_size, sig_x, sig_x, 0, isotropic=True)
    elif kernel_type == "aniso":
        sig_y = random.uniform(*sigma_range); rot = random.uniform(*rotation_range)
        k = _bivariate_gaussian(kernel_size, sig_x, sig_y, rot, isotropic=False)
    elif kernel_type in ("generalized_iso", "generalized_aniso"):
        iso = kernel_type == "generalized_iso"
        sig_y = sig_x if iso else random.uniform(*sigma_range)
        rot = 0 if iso else random.uniform(*rotation_range)
        beta = random.uniform(betag_range[0], 1) if random.random() < 0.5 else random.uniform(1, betag_range[1])
        k = _bivariate_generalized_gaussian(kernel_size, sig_x, sig_y, rot, beta, isotropic=iso)
    else:  # plateau_iso / plateau_aniso
        iso = kernel_type == "plateau_iso"
        sig_y = sig_x if iso else random.uniform(*sigma_range)
        rot = 0 if iso else random.uniform(*rotation_range)
        beta = random.uniform(betap_range[0], 1) if random.random() < 0.5 else random.uniform(1, betap_range[1])
        k = _bivariate_plateau(kernel_size, sig_x, sig_y, rot, beta, isotropic=iso)
    return k, kernel_type


def _circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """Exact port of circular_lowpass_kernel (2D sinc/jinc ringing filter). Uses
    scipy's Bessel J1 when available (bit-faithful to the training engine); falls
    back to a windowed separable sinc (visually similar ringing, not bit-exact)
    when scipy isn't installed."""
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    center = (kernel_size - 1) / 2
    if SCIPY_AVAILABLE:
        ax = np.arange(kernel_size) - center
        xx, yy = np.meshgrid(ax, ax)
        r = np.sqrt(xx ** 2 + yy ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = cutoff * _scipy_special.j1(cutoff * r) / (2 * np.pi * r)
        ci = int(round(center))
        kernel[ci, ci] = cutoff ** 2 / (4 * np.pi)
    else:
        ax = np.arange(kernel_size) - center
        win = np.hanning(kernel_size) if kernel_size > 1 else np.ones(1)
        sinc1d = np.sinc(cutoff / np.pi * ax) * win
        kernel = np.outer(sinc1d, sinc1d)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad, pad), (pad, pad)))
    return kernel


def apply_kernel_pil(img, kernel):
    """Convolve an arbitrary 2D kernel (sinc or mixed-gaussian) onto an RGB image
    via FFT (numpy only — no scipy/torch needed for the convolution itself)."""
    if not NUMPY_AVAILABLE:
        return img
    arr = np.array(img).astype(np.float64)
    kh, kw = kernel.shape
    ph_pad, pw_pad = kh // 2, kw // 2
    padded = np.pad(arr, ((ph_pad, ph_pad), (pw_pad, pw_pad), (0, 0)), mode="reflect")
    ph, pw = padded.shape[:2]
    kpad = np.zeros((ph, pw), dtype=np.float64)
    ky0, kx0 = ph // 2 - kh // 2, pw // 2 - kw // 2
    kpad[ky0:ky0 + kh, kx0:kx0 + kw] = kernel
    kpad = np.fft.ifftshift(kpad)
    fk = np.fft.rfft2(kpad)
    out = np.empty_like(padded)
    for c in range(padded.shape[2]):
        fc = np.fft.rfft2(padded[..., c])
        out[..., c] = np.fft.irfft2(fc * fk, s=(ph, pw))
    out = out[ph_pad:ph - ph_pad, pw_pad:pw - pw_pad, :]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def apply_poisson_noise(img, scale_range=(0.05, 2.0), gray_prob=0.4):
    """Poisson (shot) noise — the training pipeline always applies EITHER gaussian
    OR poisson noise each stage (never neither), chosen via gaussian_noise_prob."""
    if not NUMPY_AVAILABLE:
        return img
    arr = np.array(img).astype(np.float64) / 255.0
    scale = max(1e-3, random.uniform(*scale_range))
    if random.random() < gray_prob:
        luma = np.clip(0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2], 0, 1)
        noisy = np.random.poisson(luma * 255.0 * scale) / (255.0 * scale)
        delta = (noisy - luma)[..., np.newaxis]
        arr = arr + delta
    else:
        vals = np.clip(arr, 0, 1)
        noisy = np.random.poisson(vals * 255.0 * scale) / (255.0 * scale)
        arr = arr + (noisy - vals)
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))


_RESIZE_MODE_PIL = {}


def _resize_mode_map():
    global _RESIZE_MODE_PIL
    if not _RESIZE_MODE_PIL and PIL_AVAILABLE:
        _RESIZE_MODE_PIL = {
            "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
            "nearest-exact": Image.NEAREST, "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
        }
    return _RESIZE_MODE_PIL


def _pick_resize_mode(deg, list_key, prob_key, default_list=None):
    modes = deg.get(list_key) or (default_list or ["bilinear", "bicubic", "nearest-exact", "lanczos"])
    if isinstance(modes, str):
        modes = [m.strip().strip("\"'") for m in modes.strip("[]").split(",") if m.strip()]
    probs = deg.get(prob_key)
    if isinstance(probs, str):
        try:
            probs = [float(x) for x in probs.strip("[]").split(",") if x.strip()]
        except Exception:
            probs = None
    if not probs or len(probs) != len(modes):
        probs = [1.0] * len(modes)
    choice = str(random.choices(modes, probs)[0]).lower()
    return _resize_mode_map().get(choice, Image.BICUBIC)


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
    bits = random.randint(int(bits_range[0]), int(bits_range[1]))
    if not NUMPY_AVAILABLE or bits >= 8:
        from PIL import ImageOps
        return ImageOps.posterize(img, min(8, bits))
    # round to 2**bits levels like the training hook (PIL posterize truncates = darker)
    lv = 2 ** bits - 1
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.round(arr * lv) / lv
    return Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8))


def apply_lq_usm(img, radius_range=(1, 25)):
    """Unsharp Mask on the LQ (preview of Redux's lq_usm) — over-sharpening / edge halos."""
    from PIL import ImageFilter
    r = random.uniform(float(radius_range[0]), float(radius_range[1]))
    return img.filter(ImageFilter.UnsharpMask(radius=max(0.5, r), percent=150, threshold=2))


def apply_thicklines(img):
    """Approximate Redux's ThickLines: thicken dark strokes/edges (min-filter erodes the
    bright background → dark lines grow). Faithful-enough preview for anime / line-art."""
    from PIL import ImageFilter
    return img.filter(ImageFilter.MinFilter(3))


def apply_banding(img, levels_range=(16, 64)):
    """Per-channel quantization to N levels — same math as the training hook
    (the old palette quantize looked nothing like what the model was trained on)."""
    levels = max(2, random.randint(int(levels_range[0]), int(levels_range[1])))
    if not NUMPY_AVAILABLE or levels >= 256:
        return img
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.round(arr * (levels - 1)) / (levels - 1)
    return Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8))


def apply_smooth_banding(img, levels_range=(48, 128), dark_prob=0.5):
    """Codec-style luma banding restricted to flat areas — mirror of the training hook's
    apply_smooth_banding (7x7 mean of |dx|+|dy|, linear-light variant for dark tones)."""
    if not NUMPY_AVAILABLE:
        return img, False
    from PIL import ImageFilter
    linear = random.random() < float(dark_prob)
    lv = max(2, random.randint(int(levels_range[0]), int(levels_range[1])))
    if linear:
        lv *= 6
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gx = np.zeros_like(y); gx[:, :-1] = np.abs(y[:, 1:] - y[:, :-1])
    gy = np.zeros_like(y); gy[:-1, :] = np.abs(y[1:, :] - y[:-1, :])
    g = gx + gy
    # 7x7 box mean (separable, edge-normalised like count_include_pad=False)
    k = np.ones(7, dtype=np.float32)
    ones = np.ones_like(g)

    def _box(a):
        a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, a)
        return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, a)

    g = _box(g) / _box(ones)
    mask = np.clip((0.012 - g) / 0.008, 0.0, 1.0)
    if linear:
        yq = np.clip(np.round(np.clip(y, 0, 1) ** 2.2 * lv) / lv, 0, 1) ** (1.0 / 2.2)
    else:
        yq = np.round(y * lv) / lv
    out = arr + (np.clip(yq - y, -4.0 / 255.0, 4.0 / 255.0) * mask)[..., None]
    return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)), linear


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


# ─── NEW: DiscBlur / Vignette / QuantizeDepth (sr_degrade inspired) ──────────

def apply_disc_blur_pil(img, radius_range=(2.0, 8.0)):
    """Bokeh/defocus blur approximating a disk kernel (lens out-of-focus)."""
    radius = random.uniform(*radius_range)
    try:
        from scipy.ndimage import convolve
        r = max(1, int(round(radius)))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        k = ((xx * xx + yy * yy) <= r * r).astype(np.float32)
        k /= k.sum()
        a = np.asarray(img.convert("RGB"), np.float32)
        out = np.stack([convolve(a[..., c], k, mode="reflect") for c in range(3)], -1)
        return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
    except ImportError:
        # no OpenCV: Gaussian approximation of the disc
        return img.filter(ImageFilter.GaussianBlur(radius=radius * 0.7))


def apply_vignette_pil(img, strength_range=(0.2, 0.6), radius_range=(0.45, 0.75)):
    """Darken edges to simulate lens vignetting."""
    if not NUMPY_AVAILABLE:
        return img
    strength = random.uniform(*strength_range)
    radius = random.uniform(*radius_range)
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_dist = max((cx ** 2 + cy ** 2) ** 0.5, 1.0)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dist_norm = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max_dist
    falloff = np.clip((dist_norm - radius) / max(1.0 - radius, 1e-6), 0.0, 1.0)
    vignette = (1.0 - strength * falloff)[..., np.newaxis]
    return Image.fromarray(np.clip(arr * vignette, 0, 255).astype(np.uint8))


def apply_quantize_depth_pil(img, bits_range=(4, 7)):
    """Uniform sub-8-bit depth reduction (even steps, different from posterize)."""
    if not NUMPY_AVAILABLE:
        return img
    bits = random.randint(int(bits_range[0]), int(bits_range[1]))
    levels = 2 ** bits - 1
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.round(arr * levels) / levels  # round like the training hook (floor darkened)
    return Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8))


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
    img = img.convert("RGB")
    if dither_type in ("floyd_steinberg", "atkinson", "jarvis_judice", "stucki"):
        # Image.quantize(colors=..., dither=...) ignores `dither` without a palette:
        # build the palette first, then map with Floyd-Steinberg error diffusion.
        pal = img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
        return img.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG).convert("RGB")
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


def _dct_matrix_np(n: int):
    i = np.arange(n).reshape(1, -1).astype(np.float64)
    k = np.arange(n).reshape(-1, 1).astype(np.float64)
    C = np.cos(np.pi / n * (i + 0.5) * k)
    alpha = np.full((n, 1), np.sqrt(2.0 / n))
    alpha[0, 0] = np.sqrt(1.0 / n)
    return (C * alpha)


def _radial_quant_table_np(n: int, quality: float):
    q = max(1.0, min(100.0, float(quality)))
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    scale = max(1.0, scale)
    u = np.arange(n).reshape(1, -1).astype(np.float64)
    v = np.arange(n).reshape(-1, 1).astype(np.float64)
    r = np.sqrt(u * u + v * v)
    return np.maximum(1.0, np.round((1.0 + r) * scale / 100.0))


def apply_dct_block_degrade(img, block_size: int = 4, quality: float = 50.0,
                            chroma_scale: float = 2.0, yuv_matrix: str = "601"):
    """Preview twin of otf_custom_degradations.apply_dct_block_degrade_t (GPU
    training hook) — same 4×4-block-DCT + radial-quantisation-table technique
    (our own implementation inspired by ArtCNN v1.6.2's changelog, not a
    verified port — see that function's docstring), numpy-only for the UI
    preview. img: PIL.Image RGB."""
    if not NUMPY_AVAILABLE:
        return img
    n = max(2, int(block_size))
    matrices = {"601": [0.299, 0.587, 0.114], "709": [0.2126, 0.7152, 0.0722],
               "2020": [0.2627, 0.6780, 0.0593]}
    kr, kg, kb = matrices.get(str(yuv_matrix), matrices["601"])

    arr = np.array(img.convert("RGB")).astype(np.float64)
    h, w = arr.shape[:2]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2 * (1 - kb)) + 128.0
    cr = (r - y) / (2 * (1 - kr)) + 128.0

    Cmat = _dct_matrix_np(n)
    Ct = Cmat.T
    q_luma = _radial_quant_table_np(n, quality)
    q_chroma = np.maximum(1.0, q_luma * max(1.0, float(chroma_scale)))

    def _transform(plane, q_table):
        ph = (n - plane.shape[0] % n) % n
        pw = (n - plane.shape[1] % n) % n
        p = np.pad(plane, ((0, ph), (0, pw)), mode="edge") if (ph or pw) else plane
        Hp, Wp = p.shape
        blocks = p.reshape(Hp // n, n, Wp // n, n).transpose(0, 2, 1, 3).reshape(-1, n, n)
        coeff = Cmat[None] @ blocks @ Ct[None]
        coeff = np.round(coeff / q_table[None]) * q_table[None]
        rec = Ct[None] @ coeff @ Cmat[None]
        rec = rec.reshape(Hp // n, Wp // n, n, n).transpose(0, 2, 1, 3).reshape(Hp, Wp)
        return rec[:h, :w]

    y2 = _transform(y, q_luma)
    cb2 = _transform(cb, q_chroma) - 128.0
    cr2 = _transform(cr, q_chroma) - 128.0

    r2 = y2 + 2 * (1 - kr) * cr2
    b2 = y2 + 2 * (1 - kb) * cb2
    g2 = (y2 - kr * r2 - kb * b2) / max(kg, 1e-6)
    out = np.stack([r2, g2, b2], axis=2).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


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


def _apply_stage_blur(img, deg, stage):
    """Blur1/Blur2: exact selection logic from realesrgan_dataset.py — sinc_prob
    chance of a ringing sinc/jinc kernel, else a mixed iso/aniso/generalized/plateau
    gaussian-family kernel (using the UI's Blur Sigma + Betag/Betap Range)."""
    suf = "" if stage == 1 else "2"
    ks = random.choice(_KERNEL_SIZES)
    if random.random() < float(deg.get(f"sinc_prob{suf}", 0.0)):
        omega_c = random.uniform(math.pi / 3, math.pi) if ks < 13 else random.uniform(math.pi / 5, math.pi)
        kernel = _circular_lowpass_kernel(omega_c, ks)
        return apply_kernel_pil(img, kernel), f"Sinc{stage} ks={ks} ωc={omega_c:.2f}"
    default_sigma = (0.2, 3.0) if stage == 1 else (0.2, 1.5)
    sigma_range = _parse_range(deg.get(f"blur_sigma{suf}", str(list(default_sigma))), default_sigma)
    betag_range = _parse_range(deg.get(f"betag_range{suf}", "[0.5, 4]"), (0.5, 4))
    betap_range = _parse_range(deg.get(f"betap_range{suf}", "[1, 2]"), (1, 2))
    kernel, ktype = _random_mixed_kernel(ks, sigma_range, betag_range, betap_range)
    return apply_kernel_pil(img, kernel), f"Blur{stage} {ktype} ks={ks} sigma~U{sigma_range}"


def _apply_stage_resize(img, deg, stage, ori_size, scale):
    """Resize1 (relative to current size) / Resize2 (absolute, relative to
    ori_size/scale) — exact up/down/keep logic from feed_data."""
    suf = "" if stage == 1 else "2"
    default_probs = (0.2, 0.7, 0.1) if stage == 1 else (0.3, 0.4, 0.3)
    up, down, keep = _parse_three(deg.get(f"resize_prob{suf}"), default_probs)
    updown = random.choices(["up", "down", "keep"], [up, down, keep])[0]
    default_range = (0.5, 1.5) if stage == 1 else (0.3, 1.2)
    rng = _parse_range(deg.get(f"resize_range{suf}", str(list(default_range))), default_range)
    if updown == "up":
        factor = random.uniform(1.0, max(1.0, rng[1]))
    elif updown == "down":
        factor = random.uniform(min(rng[0], 1.0), 1.0)
    else:
        factor = 1.0
    if abs(factor - 1.0) < 1e-6:
        return img, None
    mode = _pick_resize_mode(deg, f"resize_mode_list{suf}", f"resize_mode_prob{suf}")
    if stage == 1:
        w, h = img.size
        nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    else:
        ori_w, ori_h = ori_size
        nw = max(1, round(ori_w / scale * factor))
        nh = max(1, round(ori_h / scale * factor))
    return img.resize((nw, nh), mode), f"Resize{stage} {updown} x{factor:.2f} → {nw}x{nh}"


def _apply_stage_noise(img, deg, stage):
    """Noise always fires each stage — gaussian OR poisson, never neither
    (matches feed_data's if/else, not an independent skip-gate)."""
    suf = "" if stage == 1 else "2"
    gray_prob = float(deg.get(f"gray_noise_prob{suf}", 0.4))
    if random.random() < float(deg.get(f"gaussian_noise_prob{suf}", 0.0)):
        default_range = (1, 30) if stage == 1 else (1, 25)
        noise_range = _parse_range(deg.get(f"noise_range{suf}", str(list(default_range))), default_range)
        return apply_noise(img, noise_range, gray_prob), f"Noise{stage} gauss sigma~U{noise_range}"
    default_range = (0.05, 2.0) if stage == 1 else (0.05, 2.5)
    poisson_range = _parse_range(deg.get(f"poisson_scale_range{suf}", str(list(default_range))), default_range)
    return apply_poisson_noise(img, poisson_range, gray_prob), f"Noise{stage} poisson scale~U{poisson_range}"


def apply_otf_pipeline(img, config: dict, scale: int = 4) -> Tuple:
    """
    Apply the full OTF degradation pipeline, ported step-for-step from
    RealESRGANModel.feed_data (traiNNer/models/realesrgan_model.py): LQ-USM +
    ThickLines -> Blur1(sinc/mixed) -> Resize1 -> Noise1 -> JPEG1 -> Blur2 ->
    Resize2 -> Noise2 -> [50/50: resize-back+FinalSinc vs JPEG2] -> custom
    degradations (posterize..coupled_vintage, in the exact order the training-time
    USR_STUDIO_CUSTOM_HOOK applies them, i.e. strictly LAST).

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
    ori_w, ori_h = img.size

    # ===== Native Redux, always first: LQ-USM then ThickLines =====
    if str(deg.get("lq_usm", "false")).lower() in ("true", "1", "yes", "on"):
        usm_range = _parse_range(deg.get("lq_usm_radius_range", "[1, 25]"), (1, 25))
        out = apply_lq_usm(out, usm_range)
        log.append(f"LQ-USM radius~U{usm_range}")
    if random.random() < float(deg.get("thicklines_prob", 0.0)):
        out = apply_thicklines(out)
        log.append("ThickLines")

    # ===== Stage 1 : blur -> resize -> noise -> jpeg =====
    # NeoSR/Real-ESRGAN style: stage 1 blur is ALWAYS applied (no blur_prob toggle).
    # USR Studio templates may set blur_prob; if absent, default to 1.0 (always blur).
    blur1_prob = float(deg.get("blur_prob", 1.0 if "blur_sigma" in deg else 0.0))
    if random.random() < blur1_prob and NUMPY_AVAILABLE:
        out, blog = _apply_stage_blur(out, deg, 1)
        log.append(blog)

    out, rlog = _apply_stage_resize(out, deg, 1, (ori_w, ori_h), scale)
    if rlog:
        log.append(rlog)

    if NUMPY_AVAILABLE:
        out, nlog = _apply_stage_noise(out, deg, 1)
        log.append(nlog)

    if random.random() < float(deg.get("jpeg_prob", 0.0)):
        q_range = _parse_range(deg.get("jpeg_range", "[30, 95]"), (30, 95))
        out = apply_jpeg(out, q_range)
        log.append(f"JPEG1 q~U{q_range}")

    # ===== Stage 2 : blur -> resize (absolute, ori/scale) -> noise =====
    if random.random() < float(deg.get("second_blur_prob", 0.0)) and NUMPY_AVAILABLE:
        out, blog = _apply_stage_blur(out, deg, 2)
        log.append(blog)

    out, rlog = _apply_stage_resize(out, deg, 2, (ori_w, ori_h), scale)
    if rlog:
        log.append(rlog)

    if NUMPY_AVAILABLE:
        out, nlog = _apply_stage_noise(out, deg, 2)
        log.append(nlog)

    # ===== Final : 50/50 order {resize-back + FinalSinc} vs {JPEG2} =====
    target_w, target_h = max(1, ori_w // scale), max(1, ori_h // scale)
    resize_mode3 = _pick_resize_mode(deg, "resize_mode_list3", "resize_mode_prob3")

    def _final_resize_sinc(im):
        im2 = im.resize((target_w, target_h), resize_mode3)
        if random.random() < float(deg.get("final_sinc_prob", 0.0)) and NUMPY_AVAILABLE:
            ks = random.choice(_KERNEL_SIZES)
            omega_c = random.uniform(math.pi / 3, math.pi)
            kernel = _circular_lowpass_kernel(omega_c, ks, pad_to=21)
            im2 = apply_kernel_pil(im2, kernel)
            return im2, f"FinalSinc ks={ks} ωc={omega_c:.2f}"
        return im2, f"FinalResize → {target_w}x{target_h}"

    def _final_jpeg(im):
        if random.random() < float(deg.get("jpeg_prob2", deg.get("jpeg_prob", 0.0))):
            q_range = _parse_range(deg.get("jpeg_range2", "[30, 95]"), (30, 95))
            return apply_jpeg(im, q_range), f"JPEG2 q~U{q_range}"
        return im, None

    if random.random() < 0.5:
        out, l1 = _final_resize_sinc(out); log.append(l1)
        out, l2 = _final_jpeg(out)
        if l2:
            log.append(l2)
    else:
        out, l2 = _final_jpeg(out)
        if l2:
            log.append(l2)
        out, l1 = _final_resize_sinc(out); log.append(l1)

    # ===== Custom degradations — run LAST, exactly like USR_STUDIO_CUSTOM_HOOK =====
    if random.random() < float(deg.get("posterize_prob", 0.0)):
        bits_range = _parse_range(deg.get("posterize_bits_range", "[3, 6]"), (3, 6))
        out = apply_posterize(out, bits_range)
        log.append(f"Posterize bits~U{bits_range}")

    if random.random() < float(deg.get("banding_prob", 0.0)):
        lev_range = _parse_range(deg.get("banding_levels_range", "[16, 64]"), (16, 64))
        out = apply_banding(out, lev_range)
        log.append(f"Banding levels~U{lev_range}")

    if random.random() < float(deg.get("smooth_banding_prob", 0.0)):
        sb_range = _parse_range(deg.get("smooth_banding_levels_range", "[48, 128]"), (48, 128))
        out, _lin = apply_smooth_banding(out, sb_range, float(deg.get("smooth_banding_dark_prob", 0.5)))
        log.append(f"Smooth banding levels~U{sb_range}{' (dark/linear)' if _lin else ''}")

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

    if random.random() < float(deg.get("dct_block_prob", 0.0)):
        q_lo, q_hi = _parse_range(deg.get("dct_block_quality_range", "[20, 80]"), (20, 80))
        bs_lo, bs_hi = _parse_range(deg.get("dct_block_size_range", "[4, 4]"), (4, 4))
        cs_lo, cs_hi = _parse_range(deg.get("dct_block_chroma_scale_range", "[1.5, 2.5]"), (1.5, 2.5))
        yuv_d = str(deg.get("dct_block_yuv", "601"))
        bs = random.randint(int(bs_lo), max(int(bs_lo), int(bs_hi)))
        q = random.uniform(q_lo, q_hi)
        cs = random.uniform(cs_lo, cs_hi)
        out = apply_dct_block_degrade(out, block_size=bs, quality=q, chroma_scale=cs, yuv_matrix=yuv_d)
        log.append(f"DCT-Block {bs}x{bs} quality={q:.0f} chroma×{cs:.2f} YUV-{yuv_d}")

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

    # ===== NEW: DiscBlur / Vignette / QuantizeDepth =====
    if random.random() < float(deg.get("disc_blur_prob", 0.0)):
        r_range = _parse_range(deg.get("disc_blur_radius_range", "[2, 8]"), (2.0, 8.0))
        out = apply_disc_blur_pil(out, r_range)
        log.append(f"DiscBlur r~U{r_range}")

    if random.random() < float(deg.get("vignette_prob", 0.0)):
        st_range = _parse_range(deg.get("vignette_strength_range", "[0.2, 0.6]"), (0.2, 0.6))
        ra_range = _parse_range(deg.get("vignette_radius_range", "[0.45, 0.75]"), (0.45, 0.75))
        out = apply_vignette_pil(out, st_range, ra_range)
        log.append(f"Vignette str~U{st_range}")

    if random.random() < float(deg.get("quantize_depth_prob", 0.0)):
        b_range = _parse_range(deg.get("quantize_depth_bits_range", "[4, 7]"), (4, 7))
        out = apply_quantize_depth_pil(out, b_range)
        log.append(f"QuantizeDepth bits~U{b_range}")

    # ===== Coupled clusters (fire-together groups) =====
    if random.random() < float(deg.get("coupled_optical_prob", 0.0)):
        # Cheap-lens: disc_blur + vignette + CA all fire together
        r_range  = _parse_range(deg.get("disc_blur_radius_range", "[2, 8]"), (2.0, 8.0))
        out = apply_disc_blur_pil(out, r_range)
        st_range = _parse_range(deg.get("vignette_strength_range", "[0.2, 0.6]"), (0.2, 0.6))
        ra_range = _parse_range(deg.get("vignette_radius_range", "[0.45, 0.75]"), (0.45, 0.75))
        out = apply_vignette_pil(out, st_range, ra_range)
        ca_range = _parse_range(deg.get("ca_shift_range", "[1, 5]"), (1, 5))
        out = apply_chromatic_aberration(out, ca_range)
        log.append("Coupled-Optical (disc_blur+vignette+CA)")

    if random.random() < float(deg.get("coupled_vintage_prob", 0.0)):
        # Vintage: VHS + banding + film_grain fire together
        vhs_range = _parse_range(deg.get("vhs_strength_range", "[0.1, 0.5]"), (0.1, 0.5))
        out = apply_vhs(out, vhs_range)
        b_range = _parse_range(deg.get("banding_levels_range", "[16, 64]"), (16, 64))
        out = apply_banding(out, b_range)
        fg_range = _parse_range(deg.get("film_grain_strength_range", "[0.03, 0.12]"), (0.03, 0.12))
        sz_range = _parse_range(deg.get("film_grain_size_range", "[1, 2]"), (1, 2))
        out = apply_film_grain_pil(out, fg_range, sz_range)
        log.append("Coupled-Vintage (VHS+banding+grain)")

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
