"""
line_filters.py — Line-art / edge post-processing filters (numpy/scipy, in-process).

Pure-numpy filters used by the Post Processing PIL chain (no GPU, no extra deps
beyond what the build already ships: numpy, scipy, pillow).
Each function takes/returns a float32 array [H, W, 3] in [0, 1].

Modules (VapourSynth equivalents in parentheses):
  cas          Contrast Adaptive Sharpening (AMD FidelityFX CAS)          (core.cas.CAS)
  line_darken  Line darkening / reinforcement                              (Hysteria)
  line_thin    Line thinning via edge warp                                 (aWarpSharp2)
  edge_cleanup Edge repair + ringing removal                               (havsfunc EdgeCleaner)
  deband       Banding removal + optional grain injection                  (neo_f3kdb)

v2.5.8 — Universal SR Studio
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _pad_edge(arr: np.ndarray) -> np.ndarray:
    """3x3 neighbour padding (edge-replicate mode)."""
    return np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")


def _luma(arr: np.ndarray) -> np.ndarray:
    """BT.601 luma, shape [H, W, 1]."""
    return (0.299 * arr[..., 0:1] + 0.587 * arr[..., 1:2] + 0.114 * arr[..., 2:3])


# ─────────────────────────────────────────────────────────────────────────────
# CAS — Contrast Adaptive Sharpening (AMD FidelityFX, sharpen-only path)
# Ref VS : core.cas.CAS(clip, sharpness)
# ─────────────────────────────────────────────────────────────────────────────

def cas(arr: np.ndarray, sharpness: float = 0.5) -> np.ndarray:
    """
    Contrast Adaptive Sharpening (FidelityFX CAS).

    Per-channel, contrast-aware sharpening: low-contrast flat areas get less
    sharpening (less halo/noise boost) than edges.

    Parameters
    ----------
    arr        : float32 [H, W, 3] in [0, 1]
    sharpness  : 0..1   0 = mild, 1 = strong
    """
    s = float(np.clip(sharpness, 0.0, 1.0))
    p = _pad_edge(arr)
    a = p[:-2, :-2]; b = p[:-2, 1:-1]; c = p[:-2, 2:]
    d = p[1:-1, :-2]; e = arr;          f = p[1:-1, 2:]
    g = p[2:, :-2];  h = p[2:, 1:-1];  i = p[2:, 2:]

    mn = np.minimum(np.minimum(np.minimum(d, e), np.minimum(f, b)), h)
    mn = mn + np.minimum(mn, np.minimum(np.minimum(a, c), np.minimum(g, i)))
    mx = np.maximum(np.maximum(np.maximum(d, e), np.maximum(f, b)), h)
    mx = mx + np.maximum(mx, np.maximum(np.maximum(a, c), np.maximum(g, i)))

    rcp_mx = 1.0 / np.maximum(mx, _EPS)
    amp = np.sqrt(np.clip(np.minimum(mn, 2.0 - mx) * rcp_mx, 0.0, 1.0))

    peak = -1.0 / (8.0 + (5.0 - 8.0) * s)
    w = amp * peak
    rcp_w = 1.0 / (1.0 + 4.0 * w)
    out = (b * w + d * w + f * w + h * w + e) * rcp_w
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Line Darken — Hysteria-faithful line reinforcement
# Ref VS : hysteria.Hysteria(clip, strength=0.2, usemask=True,
#                             lowthresh=4, highthresh=28, showmask=False)
#
# Algorithm ported from hysteria.py (StaxRip VapourSynth):
#   1. Cartoon-mode edge detection (matrix [0,-2,1,0,1,0,0,0,0])
#   2. Two-threshold hysteresis: grow strong edges (>hi) into weak (>lo)
#   3. Inflate -> blur x2 -> deflate the mask
#   4. Darkening = (Inflate(lum) - lum) * strength, clamped by minchg/maxchg
#   5. Skip pixels where luma > luma_cap (protect highlights)
#   6. Masked merge at mask locations only
# ─────────────────────────────────────────────────────────────────────────────

def line_darken(
    arr: np.ndarray,
    strength: float = 0.2,
    lowthresh: int = 4,
    highthresh: int = 28,
    luma_cap: int = 191,
    maxchg: int = 255,
    minchg: int = 0,
) -> np.ndarray:
    """
    Hysteria-faithful line reinforcement.

    Parameters
    ----------
    arr        : float32 [H, W, 3] in [0, 1]
    strength   : darkening strength (0 = none, 1 = full)
    lowthresh  : hysteresis lower threshold (8-bit, 0-255)
    highthresh : hysteresis upper threshold (8-bit, 0-255)
    luma_cap   : don't darken pixels brighter than this (8-bit, 0-255)
    maxchg     : maximum darkening per pixel (8-bit, 0-255)
    minchg     : minimum darkening to apply; 0 = apply all (8-bit, 0-255)
    """
    try:
        from scipy.ndimage import (maximum_filter, minimum_filter,
                                   uniform_filter, binary_dilation)
    except ImportError:
        return arr.copy()

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength < 1e-4:
        return arr.copy()

    lo  = float(np.clip(lowthresh,  0, 255)) / 255.0
    hi  = float(np.clip(highthresh, 0, 255)) / 255.0
    cap = float(np.clip(luma_cap,   0, 255)) / 255.0
    mx  = float(np.clip(maxchg,     0, 255)) / 255.0
    mn  = float(np.clip(minchg,     0, 255)) / 255.0
    if hi <= lo:
        hi = lo + _EPS

    lum = _luma(arr)[..., 0]   # [H, W]

    # 1. Cartoon-mode directional gradient (hysteria.py matrix [0,-2,1,0,1,0,0,0,0])
    #    edge[r,c] = max(0, -2*lum[r-1,c] + lum[r-1,c+1] + lum[r,c])
    p    = np.pad(lum, 1, mode='edge')
    edge = np.clip(-2.0 * p[:-2, 1:-1] + p[:-2, 2:] + p[1:-1, 1:-1], 0.0, 1.0)

    # 2. Hysteresis: grow strong edges into weak candidates
    strong     = edge >= hi
    candidates = edge >= lo
    mask = strong.copy()
    for _ in range(20):
        expanded = binary_dilation(mask, iterations=1) & candidates
        if np.array_equal(expanded, mask):
            break
        mask = expanded

    # 3. Inflate -> blur x2 -> deflate (smooth/widen the mask)
    themask = maximum_filter(mask.astype(np.float32), size=3)
    themask = uniform_filter(themask, size=3)
    themask = uniform_filter(themask, size=3)
    themask = minimum_filter(themask, size=3)

    # 4. Darkening amount = (local_max - pixel) * strength
    local_max = maximum_filter(lum, size=3)
    diffs     = (local_max - lum) * strength
    diffs     = np.where(lum > cap, 0.0, diffs)   # skip bright pixels (luma_cap)
    diffs     = np.clip(diffs, mn, mx)

    darkened_lum = lum - diffs

    # 5. Masked merge
    result_lum = lum * (1.0 - themask) + darkened_lum * themask

    # 6. Propagate luma shift proportionally to all RGB channels (preserves hue)
    ratio = np.where(lum > _EPS, result_lum / (lum + _EPS), 1.0)
    ratio = np.clip(ratio, 0.0, 1.0)[..., np.newaxis]   # [H, W, 1]
    return np.clip(arr * ratio, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Line Thinning — aWarpSharp2-style warp sharpening
# Ref VS : core.warp.AWarpSharp2(clip, depth=6, blur=1, planes=[0])
#
# Key fix: `thresh` is now a SOFT SCALE for a tanh mask, not a binary cutoff.
# The previous default of 0.55 (binary) produced a zero mask on anime content
# where typical luma gradient magnitudes are 0.05-0.2 in [0,1] range.
# ─────────────────────────────────────────────────────────────────────────────

def antialias(arr: np.ndarray, strength: float = 0.5, passes: int = 1) -> np.ndarray:
    """Edge-directed anti-aliasing (FXAA/SangNom-lite, CPU).

    Smooths jagged edges by blending each high-gradient pixel ALONG its edge
    tangent (perpendicular to the luma gradient), leaving flat areas untouched.
    Not a neural nnedi3/eedi3 — a fast deterministic approximation for staircase
    artefacts on diagonals. `strength` 0-1, `passes` 1-3.

    arr: float32 [H, W, 3] in [0, 1].
    """
    try:
        from scipy.ndimage import map_coordinates, uniform_filter
    except ImportError:
        return arr.copy()
    s = float(np.clip(strength, 0.0, 1.0))
    if s < 0.01:
        return arr.copy()
    out = arr.astype(np.float32).copy()
    H, W = out.shape[:2]
    rows = np.arange(H, dtype=np.float64); cols = np.arange(W, dtype=np.float64)
    gr, gc = np.meshgrid(rows, cols, indexing="ij")
    for _ in range(max(1, int(passes))):
        lum = _luma(out)[..., 0]
        p = np.pad(lum, 1, mode="edge")
        gx = (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5
        gy = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
        gmag = np.hypot(gx, gy)
        # tangent = perpendicular to gradient, normalised
        norm = gmag + 1e-6
        tx = (-gy / norm); ty = (gx / norm)
        # edge mask: only AA where there's a real edge
        thr = float(np.clip(gmag.mean() * 1.3, 0.01, 0.5))
        emask = np.clip(gmag / (thr + 1e-6), 0.0, 1.0)[..., np.newaxis] * s
        # sample ±1px along the tangent and average with the centre
        for sign in (1.0, -1.0):
            coords = np.stack([np.clip(gr + sign * ty, 0, H - 1),
                               np.clip(gc + sign * tx, 0, W - 1)], axis=0)
            samp = np.stack([map_coordinates(out[..., ch], coords, order=1, mode="nearest")
                             for ch in range(3)], axis=-1)
            out = out * (1.0 - 0.5 * emask) + samp * (0.5 * emask)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def line_thin(
    arr: np.ndarray,
    depth: float = 6.0,
    blur: int = 1,
    thresh: float = 0.05,
    luma_only: bool = True,
) -> np.ndarray:
    """
    aWarpSharp2-faithful edge warp (warpsharp / line thinning).

    Faithful pipeline (like VapourSynth warp.AWarpSharp2):
      1. Build an edge mask = Sobel magnitude of luma, amplified + clamped.
      2. Blur the mask (the warp field — this spread is what makes it work).
      3. Warp every plane ALONG the gradient of the blurred mask. Pixels are
         pulled toward the edge ridges, which compresses transitions: lines get
         sharper and thinner. (The old version warped the luma gradient and
         pushed pixels toward the dark line centre, which thickened lines.)

    Parameters
    ----------
    arr       : float32 [H, W, 3] in [0, 1]
    depth     : warp strength (slider 0-20; aWarpSharp 'depth'). 0 = none.
    blur      : number of mask blur passes (aWarpSharp 'blur'/chroma). Higher = smoother warp.
    thresh    : edge sensitivity — lower = more/weaker edges caught (maps to mask gain).
    luma_only : kept for API compat; the warp field is always luma-derived and all
                planes are warped together (faithful behaviour).
    """
    try:
        from scipy.ndimage import uniform_filter, map_coordinates
    except ImportError:
        return arr.copy()

    depth = float(np.clip(depth, 0.0, 20.0))
    if depth < 0.1:
        return arr.copy()

    H, W = arr.shape[:2]
    lum = _luma(arr)[..., 0]

    # 1. Sobel edge mask, amplified by a gain derived from `thresh`
    #    (lower thresh -> higher gain -> weaker edges also caught).
    p    = np.pad(lum, 1, mode='edge')
    sx   = (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5
    sy   = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
    gain = float(np.clip(0.6 / max(float(thresh), 0.02), 4.0, 30.0))
    mask = np.clip(np.hypot(sx, sy) * gain, 0.0, 1.0).astype(np.float32)

    # 2. Blur the mask (warp field). At least one pass; more = smoother.
    for _ in range(max(1, int(blur))):
        mask = uniform_filter(mask, size=3)

    # 3. Gradient of the blurred mask → warp direction (toward edge ridges).
    pm  = np.pad(mask, 1, mode='edge')
    mgx = (pm[1:-1, 2:] - pm[1:-1, :-2]) * 0.5
    mgy = (pm[2:, 1:-1] - pm[:-2, 1:-1]) * 0.5

    # Map slider 0-20 to a SMALL sub-pixel displacement. The previous ×4 gain
    # produced ~50 px warps → bright overshoot fringes ("white halos"). aWarpSharp
    # only nudges pixels a fraction toward the ridge, so keep the displacement low.
    eff = float(depth) * 0.6   # depth 20 → ~12 px max (typical use 3-8 → 2-5 px)
    dx  = (eff * mgx).astype(np.float64)   # +gradient = toward edge → sharpen/thin
    dy  = (eff * mgy).astype(np.float64)

    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing='ij')
    coords = np.stack([np.clip(grid_r + dy, 0.0, H - 1),
                       np.clip(grid_c + dx, 0.0, W - 1)], axis=0)

    out = arr.copy()
    for ch in range(3):
        out[..., ch] = np.clip(
            map_coordinates(arr[..., ch], coords, order=1, mode='nearest'),
            0.0, 1.0).astype(np.float32)

    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cleanup — havsfunc EdgeCleaner-style repair
# Ref VS : havsfunc.EdgeCleaner(clip, strength=8, rmode=17, hot=True)
# ─────────────────────────────────────────────────────────────────────────────

def edge_cleanup(
    arr: np.ndarray,
    strength: int = 8,
    hot: bool = True,
    repair: bool = False,
) -> np.ndarray:
    """
    havsfunc EdgeCleaner-style edge repair.

    Repairs ringing / aliasing artifacts near edges by:
    1. Building an edge mask from luma gradient.
    2. Blending toward 3x3 median at edge regions (rmode=17 approximation).
    3. Optionally clamping hot-pixel outliers near edges.

    Parameters
    ----------
    arr      : float32 [H, W, 3] in [0, 1]
    strength : repair intensity 0-20
    hot      : if True, clamp local bright outliers near edges
    """
    try:
        from scipy.ndimage import median_filter, uniform_filter
    except ImportError:
        return arr

    s = float(np.clip(strength, 0, 20)) / 20.0
    if s < 0.01:
        return arr.copy()

    lum = _luma(arr)[..., 0]
    p   = np.pad(lum, 1, mode='edge')
    gx  = (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5
    gy  = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
    gmag = np.hypot(gx, gy)
    # the centred gradient is 0 on an isolated pixel itself: dilate so the defect
    # pixel is inside the mask, not only its neighbours
    from scipy.ndimage import maximum_filter as _mx
    gmag = _mx(gmag, size=3)

    thr   = float(np.clip(gmag.mean() * 1.5, 0.01, 0.5))
    emask = np.clip(gmag / (thr + _EPS), 0.0, 1.0)[..., np.newaxis]   # [H, W, 1]

    repaired = np.stack([
        median_filter(arr[..., ch], size=3) for ch in range(3)
    ], axis=-1)

    out = arr * (1.0 - emask * s) + repaired * (emask * s)

    if hot:
        blurred  = uniform_filter(arr, size=(3, 3, 1))
        sigma    = uniform_filter(np.abs(arr - blurred), size=(3, 3, 1))
        # F37: too dark as well as too bright
        hot_mask = (np.abs(arr - blurred) > 3.0 * sigma + _EPS) * emask
        out      = np.where(hot_mask > 0.5, blurred, out)

    # Repair mode (RemoveGrain rmode=17-style): clamp each pixel to the min/max of
    # its 3×3 neighbourhood at edge regions, killing residual ringing/overshoot.
    if repair:
        try:
            from scipy.ndimage import minimum_filter, maximum_filter
            # F37: the 8 neighbours only — with the pixel itself in the window the
            # clamp could never change it
            fp = np.ones((3, 3), bool); fp[1, 1] = False
            nmin = np.stack([minimum_filter(out[..., ch], footprint=fp) for ch in range(3)], axis=-1)
            nmax = np.stack([maximum_filter(out[..., ch], footprint=fp) for ch in range(3)], axis=-1)
            clamped = np.clip(out, nmin, nmax)
            out = out * (1.0 - emask) + clamped * emask
        except Exception:
            pass

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Deband — neo_f3kdb-style banding removal + optional grain
# Ref VS : core.neo_f3kdb.Deband(clip, y=12, cb=10, cr=10, grainy=0, grainc=0)
#
# Key fix: previous implementation used uniform_filter(size=2*range_+1) which
# replaced pixels with a HEAVY BLUR regardless of whether they were banded.
# This caused the color washout + extreme blur reported by the user.
#
# Correct f3kdb approach: for each pixel, sample 4 cardinal neighbours at
# distance `range_`. If ALL 4 are within `threshold` of the center pixel,
# the pixel is in a smooth banded region -> replace with their average.
# Otherwise the pixel is left UNCHANGED (no blur applied).
# ─────────────────────────────────────────────────────────────────────────────

def deband(
    arr: np.ndarray,
    y: int = 12,
    cb: int = 10,
    cr: int = 10,
    grainy: int = 0,
    grainc: int = 0,
    range_: int = 15,
    seed: int = 0,
) -> np.ndarray:
    """
    neo_f3kdb-style banding removal with sparse cardinal sampling.

    Only pixels in flat/banded regions (all 4 cardinal samples within threshold)
    are modified; textured/edge regions are left untouched.

    Parameters
    ----------
    arr     : float32 [H, W, 3] in [0, 1]
    y       : luma threshold (0-64, 8-bit scale). 0 = disabled.
    cb      : Cb chroma threshold.
    cr      : Cr chroma threshold.
    grainy  : luma grain strength (0-64).
    grainc  : chroma grain strength (0-64).
    range_  : sampling distance in pixels (f3kdb default = 15).
    seed    : RNG seed for grain.
    """
    out  = arr.copy()
    H, W = arr.shape[:2]
    r    = max(1, int(range_))

    # Convert RGB -> simplified YCbCr (BT.601 luma + colour differences)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    Y_  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb_ = B - Y_
    Cr_ = R - Y_

    def _deband_plane(plane: np.ndarray, strength: int) -> np.ndarray:
        if strength <= 0:
            return plane.copy()
        # Fixed flat-region detection threshold; the slider controls the blend
        # INTENSITY (not a hard threshold), so 1 = very subtle, 64 = strong, and
        # the change scales smoothly. The old code did a binary replace (full avg
        # vs full center) which was harsh and produced hard "micro-square" edges.
        thr = 18.0 / 255.0
        amt = min(1.0, float(strength) / 48.0)
        pad = r
        p   = np.pad(plane, pad, mode='edge')    # [H+2r, W+2r]
        c   = p[pad:pad+H,     pad:pad+W]        # center
        u   = p[0:H,           pad:pad+W]        # up r pixels
        d   = p[2*pad:2*pad+H, pad:pad+W]        # down r pixels
        l   = p[pad:pad+H,     0:W]              # left r pixels
        ri  = p[pad:pad+H,     2*pad:2*pad+W]    # right r pixels
        avg = (u + d + l + ri) * 0.25
        # Flatness in [0,1]: 1 where all neighbours match the centre, fading to 0
        # at the threshold edge — a soft mask avoids hard banding of the deband.
        maxdiff = np.maximum.reduce([np.abs(u - c), np.abs(d - c),
                                     np.abs(l - c), np.abs(ri - c)])
        flat  = np.clip(1.0 - maxdiff / thr, 0.0, 1.0)
        blend = amt * flat
        return (c + (avg - c) * blend).astype(np.float32)

    Y2  = _deband_plane(Y_,  y)
    Cb2 = _deband_plane(Cb_, cb)
    Cr2 = _deband_plane(Cr_, cr)

    # Optional grain injection (anti-banding dither). Scaled gently (/512) so a
    # setting of 1 is a true subtle dither (~0.5/255 sigma) rather than visible noise.
    rng = np.random.default_rng(seed)
    if grainy > 0:
        Y2  = Y2  + rng.normal(0.0, float(grainy) / 512.0, (H, W)).astype(np.float32)
    if grainc > 0:
        gc  = rng.normal(0.0, float(grainc) / 512.0, (H, W)).astype(np.float32)
        Cb2 = Cb2 + gc
        Cr2 = Cr2 + gc * 0.5

    # Back to RGB (exact inverse of simplified YCbCr above)
    # R = Y + Cr,  B = Y + Cb,  G = Y - (0.299*Cr + 0.114*Cb) / 0.587
    out[..., 0] = np.clip(Y2 + Cr2,                          0.0, 1.0)
    out[..., 1] = np.clip(Y2 - 0.5093 * Cr2 - 0.1942 * Cb2, 0.0, 1.0)
    out[..., 2] = np.clip(Y2 + Cb2,                          0.0, 1.0)
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# High-quality separable resampling — sinc (Lanczos-4), spline16, and cubic
# variants not available in Pillow. Weights are always normalised so windowed
# kernels (Lanczos) integrate to 1. Verified by partition-of-unity tests.
# ─────────────────────────────────────────────────────────────────────────────

def _sinc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return np.where(np.abs(x) < 1e-8, 1.0, np.sin(np.pi * x) / (np.pi * x))


def _lanczos(x: np.ndarray, a: int = 4) -> np.ndarray:
    x = np.abs(x)
    return np.where(x < a, _sinc(x) * _sinc(x / a), 0.0)


def _cubic(x: np.ndarray, B: float, C: float) -> np.ndarray:
    """Generalised Mitchell-Netravali cubic (B, C parameterisation)."""
    x = np.abs(np.asarray(x, np.float64)); x2 = x * x; x3 = x2 * x
    w = np.zeros_like(x); m1 = x < 1; m2 = (x >= 1) & (x < 2)
    w[m1] = ((12 - 9 * B - 6 * C) * x3[m1] + (-18 + 12 * B + 6 * C) * x2[m1] + (6 - 2 * B)) / 6.0
    w[m2] = ((-B - 6 * C) * x3[m2] + (6 * B + 30 * C) * x2[m2]
             + (-12 * B - 48 * C) * x[m2] + (8 * B + 24 * C)) / 6.0
    return w


def _poly3(x, c0, c1, c2, c3):
    return c0 + x * (c1 + x * (c2 + x * c3))


# Spline16/36/64 — exact zimg (VapourSynth) "Panorama Tools" coefficients.
# Verified by partition-of-unity tests. poly3 = Horner form c0+x(c1+x(c2+x*c3)).
def _spline16(x: np.ndarray) -> np.ndarray:
    x = np.abs(np.asarray(x, np.float64)); w = np.zeros_like(x)
    m1 = x < 1; m2 = (x >= 1) & (x < 2)
    w[m1] = _poly3(x[m1],     1.0, -1/5,    -9/5,  1.0)
    w[m2] = _poly3(x[m2] - 1, 0.0, -7/15,    4/5, -1/3)
    return w


def _spline36(x: np.ndarray) -> np.ndarray:
    x = np.abs(np.asarray(x, np.float64)); w = np.zeros_like(x)
    m1 = x < 1; m2 = (x >= 1) & (x < 2); m3 = (x >= 2) & (x < 3)
    w[m1] = _poly3(x[m1],     1.0, -3/209,    -453/209, 13/11)
    w[m2] = _poly3(x[m2] - 1, 0.0, -156/209,   270/209, -6/11)
    w[m3] = _poly3(x[m3] - 2, 0.0,  26/209,    -45/209,  1/11)
    return w


def _spline64(x: np.ndarray) -> np.ndarray:
    x = np.abs(np.asarray(x, np.float64)); w = np.zeros_like(x)
    m1 = x < 1; m2 = (x >= 1) & (x < 2); m3 = (x >= 2) & (x < 3); m4 = (x >= 3) & (x < 4)
    w[m1] = _poly3(x[m1],     1.0, -3/2911,    -6387/2911, 49/41)
    w[m2] = _poly3(x[m2] - 1, 0.0, -2328/2911,  4032/2911, -24/41)
    w[m3] = _poly3(x[m3] - 2, 0.0,  582/2911,  -1008/2911,  6/41)
    w[m4] = _poly3(x[m4] - 3, 0.0,  -97/2911,    168/2911, -1/41)
    return w


# name -> (kernel function, support radius)
RESAMPLE_KERNELS = {
    "lanczos4":  (lambda x: _lanczos(x, 4),    4),   # sharp windowed sinc
    "spline16":  (_spline16,                   2),   # spline family, low ringing
    "spline36":  (_spline36,                   3),   # popular downscale (≈Lanczos3, fewer artifacts)
    "spline64":  (_spline64,                   4),   # sharper spline, 8 taps
    "mitchell":  (lambda x: _cubic(x, 1/3, 1/3), 2), # balanced, good for downscale
    "catmull":   (lambda x: _cubic(x, 0.0, 0.5), 2), # sharp cubic
    "bspline":   (lambda x: _cubic(x, 1.0, 0.0), 2), # smooth spline
}


def _axis_weights(n_in: int, n_out: int, K, support: int) -> np.ndarray:
    """Build the [n_out, n_in] resampling weight matrix for one axis."""
    scale = n_in / float(n_out)
    fs = support * max(scale, 1.0)            # widen support for anti-aliased downscale
    centers = (np.arange(n_out) + 0.5) * scale - 0.5
    M = np.zeros((n_out, n_in), np.float64)
    for i in range(n_out):
        c = centers[i]
        idx = np.arange(int(np.floor(c - fs)), int(np.ceil(c + fs)) + 1)
        w = K((idx - c) / max(scale, 1.0))
        s = w.sum()
        if s != 0:
            w = w / s
        np.add.at(M[i], np.clip(idx, 0, n_in - 1), w)   # edge-replicate at borders
    return M


def resample(arr: np.ndarray, out_w: int, out_h: int, method: str = "lanczos4") -> np.ndarray:
    """High-quality separable resize. arr float32 [H, W, 3] in [0, 1]."""
    if method not in RESAMPLE_KERNELS:
        return arr
    K, support = RESAMPLE_KERNELS[method]
    H, W = arr.shape[:2]
    Mx = _axis_weights(W, out_w, K, support)
    My = _axis_weights(H, out_h, K, support)
    tmp = np.einsum("ij,hjc->hic", Mx, arr.astype(np.float64))
    out = np.einsum("ij,jwc->iwc", My, tmp)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
