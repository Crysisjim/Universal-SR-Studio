"""USRS texture metrics for traiNNer-redux validation (injected by USRS into traiNNer/metrics/).

Five numbers that split what PSNR averages away, computed on the luma (Y) of the
validation output (SR) against the validation GT, in 0-255 units:

- usrs_tex_mae    MAE on textured areas (micro-detail fidelity, lower = better)
- usrs_edge_mae   MAE on the strongest edges / line art (lower = better)
- usrs_flat_mae   MAE on flat areas (invented grain / noise / banding, lower = better)
- usrs_hf_corr    correlation of the high frequencies on textured areas (higher = better)
- usrs_hf_ratio   high-frequency energy SR / GT on textured areas (sharpness:
                  < 1 softer than the GT, > 1 sharper than the GT)

Masks come from the Sobel gradient of the GT luma: edges = top 8 %, flat = locally
weak gradient (bottom 35 %), texture = dense medium gradient away from edges and flat
areas. High frequencies = luma minus its Gaussian blur (sigma 1.5).

traiNNer calls each metric separately with the same (img, img2) arrays, so the five
values are computed once per image pair and cached. CPU only, ~0.2 s per 1080p image.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy import ndimage

from traiNNer.utils.registry import METRIC_REGISTRY

_CACHE: dict[str, dict[str, float]] = {}


def _key(img: np.ndarray, img2: np.ndarray) -> str:
    h = hashlib.md5()
    for a in (img, img2):
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a[::31, ::29]).tobytes())
    return h.hexdigest()


def _luma(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    if a.ndim == 2:
        return a
    if a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):  # CHW -> HWC
        a = np.moveaxis(a, 0, -1)
    if a.shape[-1] == 1:
        return a[..., 0]
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def compute_texture_metrics(img: np.ndarray, img2: np.ndarray) -> dict[str, float]:
    """img = SR, img2 = GT (uint8 or float, same scale for both). Values in 0-255 units."""
    k = _key(img, img2)
    hit = _CACHE.get(k)
    if hit is not None:
        return hit
    ys, yg = _luma(img), _luma(img2)
    if yg.max() <= 1.0 and ys.max() <= 1.0:  # float [0, 1] inputs
        ys, yg = ys * 255.0, yg * 255.0
    h, w = min(ys.shape[0], yg.shape[0]), min(ys.shape[1], yg.shape[1])
    ys, yg = ys[:h, :w], yg[:h, :w]
    gg = np.hypot(ndimage.sobel(yg, 0), ndimage.sobel(yg, 1))
    strong = gg > np.percentile(gg, 92)
    flat = ndimage.uniform_filter(gg, 9) < np.percentile(gg, 35)
    tex = (ndimage.uniform_filter((gg > np.percentile(gg, 55)).astype(np.float32), 15) > 0.45) \
        & ~ndimage.binary_dilation(strong, iterations=4) & ~flat
    d = np.abs(ys - yg)

    def mae(m: np.ndarray) -> float:
        return float(d[m].mean()) if m.any() else float("nan")

    res = {"usrs_tex_mae": mae(tex), "usrs_edge_mae": mae(strong), "usrs_flat_mae": mae(flat),
           "usrs_hf_corr": float("nan"), "usrs_hf_ratio": float("nan")}
    if tex.sum() > 50:
        hs = ys - ndimage.gaussian_filter(ys, 1.5)
        hg = yg - ndimage.gaussian_filter(yg, 1.5)
        a, b = hs[tex], hg[tex]
        if a.std() > 1e-6 and b.std() > 1e-6:
            res["usrs_hf_corr"] = float(np.corrcoef(a, b)[0, 1])
        res["usrs_hf_ratio"] = float(np.abs(a).mean() / max(1e-6, float(np.abs(b).mean())))
    if len(_CACHE) > 8:
        _CACHE.clear()
    _CACHE[k] = res
    return res


def _value(name: str, img: np.ndarray, img2: np.ndarray) -> float:
    v = compute_texture_metrics(img, img2)[name]
    # traiNNer averages over the val set: a NaN would poison the mean -> 0 instead
    return 0.0 if v != v else v


@METRIC_REGISTRY.register()
def calculate_usrs_tex_mae(img: np.ndarray, img2: np.ndarray, **kwargs: Any) -> float:
    return _value("usrs_tex_mae", img, img2)


@METRIC_REGISTRY.register()
def calculate_usrs_edge_mae(img: np.ndarray, img2: np.ndarray, **kwargs: Any) -> float:
    return _value("usrs_edge_mae", img, img2)


@METRIC_REGISTRY.register()
def calculate_usrs_flat_mae(img: np.ndarray, img2: np.ndarray, **kwargs: Any) -> float:
    return _value("usrs_flat_mae", img, img2)


@METRIC_REGISTRY.register()
def calculate_usrs_hf_corr(img: np.ndarray, img2: np.ndarray, **kwargs: Any) -> float:
    return _value("usrs_hf_corr", img, img2)


@METRIC_REGISTRY.register()
def calculate_usrs_hf_ratio(img: np.ndarray, img2: np.ndarray, **kwargs: Any) -> float:
    return _value("usrs_hf_ratio", img, img2)
