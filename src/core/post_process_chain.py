"""
post_process_chain.py — GUI-agnostic Post-Processing pipeline.

Extracted from tab_tools._pp_apply_pil_chain so BOTH the CustomTkinter UI and the
new PySide6 UI call the same processing code. Pure PIL + numpy + line_filters.

Chain order: Deband → Line Darken → Line Thin → Color (+RGB levels) → Resize
             → Sharpen (Unsharp/CAS) → Edge Cleanup.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _to_arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def apply_chain(pil_img: Image.Image, settings: dict) -> Image.Image:
    """Apply the full post-processing chain. `settings` is a flat dict of the
    same keys the CustomTkinter tab uses (deband_*, color_enabled, resize_*, …)."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # 0a. Deband
    if settings.get("deband_enabled"):
        from src.core.line_filters import deband as _deband
        pil_img = _to_pil(_deband(
            _to_arr(pil_img),
            y=int(settings.get("deband_y", 12)), cb=int(settings.get("deband_cb", 10)),
            cr=int(settings.get("deband_cr", 10)), grainy=int(settings.get("deband_grainy", 0)),
            grainc=int(settings.get("deband_grainc", 0)), range_=int(settings.get("deband_range", 15))))

    # 0b. Line Darken
    if settings.get("linedark_enabled"):
        from src.core.line_filters import line_darken as _ldk
        pil_img = _to_pil(_ldk(
            _to_arr(pil_img),
            strength=float(settings.get("linedark_strength", 0.2)),
            lowthresh=int(settings.get("linedark_lo", 4)),
            highthresh=int(settings.get("linedark_hi", 28))))

    # 0a2. Antialiasing (edge-directed)
    if settings.get("aa_enabled"):
        from src.core.line_filters import antialias as _aa
        pil_img = _to_pil(_aa(
            _to_arr(pil_img),
            strength=float(settings.get("aa_strength", 0.5)),
            passes=int(settings.get("aa_passes", 1))))

    # 0c. Line Thin
    if settings.get("linethin_enabled"):
        from src.core.line_filters import line_thin as _lth
        pil_img = _to_pil(_lth(
            _to_arr(pil_img),
            depth=float(settings.get("linethin_depth", 6.0)),
            blur=int(settings.get("linethin_blur", 1)),
            thresh=float(settings.get("linethin_thresh", 0.05)),
            luma_only=bool(settings.get("linethin_luma", True))))

    # 1. Color correction (+ per-channel R/G/B levels)
    if settings.get("color_enabled"):
        b = float(settings.get("brightness", 1.0)); c = float(settings.get("contrast", 1.0))
        s = float(settings.get("saturation", 1.0)); g = float(settings.get("gamma", 1.0))
        if abs(b - 1.0) > 1e-3: pil_img = ImageEnhance.Brightness(pil_img).enhance(b)
        if abs(c - 1.0) > 1e-3: pil_img = ImageEnhance.Contrast(pil_img).enhance(c)
        if abs(s - 1.0) > 1e-3: pil_img = ImageEnhance.Color(pil_img).enhance(s)
        if abs(g - 1.0) > 1e-3:
            arr = _to_arr(pil_img)
            pil_img = _to_pil(np.clip(np.power(arr, 1.0 / g), 0.0, 1.0))
        lr = float(settings.get("level_r", 1.0)); lg = float(settings.get("level_g", 1.0))
        lb = float(settings.get("level_b", 1.0))
        if any(abs(x - 1.0) > 1e-3 for x in (lr, lg, lb)):
            arr = _to_arr(pil_img)
            arr[..., 0] = np.clip(arr[..., 0] * lr, 0.0, 1.0)
            arr[..., 1] = np.clip(arr[..., 1] * lg, 0.0, 1.0)
            arr[..., 2] = np.clip(arr[..., 2] * lb, 0.0, 1.0)
            pil_img = _to_pil(arr)

    # 2. Resize
    if settings.get("resize_enabled"):
        _PIL_MAP = {"LANCZOS": Image.Resampling.LANCZOS, "BICUBIC": Image.Resampling.BICUBIC,
                    "BILINEAR": Image.Resampling.BILINEAR, "NEAREST": Image.Resampling.NEAREST}
        method = _PIL_MAP.get(settings.get("resize_method", "LANCZOS"), Image.Resampling.LANCZOS)
        w, h = pil_img.size
        nw, nh = w, h
        mode = settings.get("resize_mode", "percent")
        if mode == "percent":
            p = int(settings.get("resize_scale", 100)); nw, nh = max(1, round(w * p / 100)), max(1, round(h * p / 100))
        elif mode == "multiplier":
            m = float(settings.get("resize_multiplier", 1.0)); nw, nh = max(1, round(w * m)), max(1, round(h * m))
        elif mode == "pixels":
            tw, th = int(settings.get("resize_width", 0)), int(settings.get("resize_height", 0))
            if tw > 0 and th > 0:
                if settings.get("resize_aspect", True):
                    r = min(tw / w, th / h); nw, nh = max(1, round(w * r)), max(1, round(h * r))
                else:
                    nw, nh = tw, th
            elif tw > 0: nw, nh = tw, max(1, round(h * tw / w))
            elif th > 0: nh, nw = th, max(1, round(w * th / h))
        if (nw, nh) != (w, h):
            _rm = str(settings.get("resize_method", "LANCZOS")).lower()
            if _rm in ("lanczos4", "spline16", "spline36", "spline64", "mitchell", "catmull", "bspline"):
                try:
                    from src.core.line_filters import resample as _resample
                    pil_img = _to_pil(_resample(_to_arr(pil_img), nw, nh, _rm))
                except Exception:
                    pil_img = pil_img.resize((nw, nh), method)
            else:
                pil_img = pil_img.resize((nw, nh), method)

    # 3. Sharpen
    if settings.get("sharpen_enabled"):
        if settings.get("sharpen_mode", "unsharp") == "cas":
            from src.core.line_filters import cas
            pil_img = _to_pil(cas(_to_arr(pil_img), float(settings.get("cas_sharpness", 0.5))))
        else:
            percent = max(0, int(float(settings.get("sharpen_strength", 1.0)) * 100))
            pil_img = pil_img.filter(ImageFilter.UnsharpMask(
                radius=float(settings.get("sharpen_radius", 1.5)), percent=percent,
                threshold=int(settings.get("sharpen_threshold", 3))))

    # 4. Edge Cleanup
    if settings.get("edgeclean_enabled"):
        from src.core.line_filters import edge_cleanup as _ecl
        pil_img = _to_pil(_ecl(_to_arr(pil_img),
                               strength=int(settings.get("edgeclean_strength", 8)),
                               hot=bool(settings.get("edgeclean_hot", True)),
                               repair=bool(settings.get("edgeclean_repair", False))))

    return pil_img
