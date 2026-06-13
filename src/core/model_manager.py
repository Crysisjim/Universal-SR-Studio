"""
src/core/model_manager.py
═══════════════════════════════════════════════════════════════════════════════
ModelManager — registre + téléchargement des poids neuraux (download-on-demand).

Poids stockés dans :  <project_root>/runtimes/weights/<model_name>.pth
Téléchargement depuis GitHub raw (pifroggi repos, licences MIT/custom).

Usage
-----
    from src.core.model_manager import ModelManager
    mgr = ModelManager()
    if not mgr.is_ready("temporalfix_s2"):
        mgr.download("temporalfix_s2", progress_cb=lambda p, t: ...)
    path = mgr.weight_path("temporalfix_s2")

v2.5.7 — Universal SR Studio
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    # ── Temporal Fix — PyTorch PTH (pifroggi/vs_temporalfix — custom license) ──
    "temporalfix_s1": {
        "filename": "temporalfix_s1_v1.1.pth",
        "url":      "https://github.com/pifroggi/vs_temporalfix/raw/main/vs_temporalfix/models/temporalfix_s1_v1.1.pth",
        "size_mb":  1.92,
        "label":    "TemporalFix s1 PTH (léger, ~2 MB)",
        "credits":  "pifroggi — https://github.com/pifroggi/vs_temporalfix",
        "format":   "pth",
        "variant":  "s1",
    },
    "temporalfix_s2": {
        "filename": "temporalfix_s2_v1.pth",
        "url":      "https://github.com/pifroggi/vs_temporalfix/raw/main/vs_temporalfix/models/temporalfix_s2_v1.pth",
        "size_mb":  1.92,
        "label":    "TemporalFix s2 PTH (recommandé, ~2 MB)",
        "credits":  "pifroggi — https://github.com/pifroggi/vs_temporalfix",
        "format":   "pth",
        "variant":  "s2",
    },
    "temporalfix_s3": {
        "filename": "temporalfix_s3_v1.pth",
        "url":      "https://github.com/pifroggi/vs_temporalfix/raw/main/vs_temporalfix/models/temporalfix_s3_v1.pth",
        "size_mb":  1.92,
        "label":    "TemporalFix s3 PTH (fort, ~2 MB)",
        "credits":  "pifroggi — https://github.com/pifroggi/vs_temporalfix",
        "format":   "pth",
        "variant":  "s3",
    },
    # ── Temporal Fix — ONNX fp16 (op18, dynamic axes) ─────────────────────
    "temporalfix_s1_onnx": {
        "filename": "temporalfix_s1_v1.1_op18_fp16.onnx",
        "url":      "https://github.com/pifroggi/vs_temporalfix/raw/main/vs_temporalfix/models/temporalfix_s1_v1.1_op18_fp16.onnx",
        "size_mb":  1.14,
        "label":    "TemporalFix s1 ONNX fp16 (~1.1 MB)",
        "credits":  "pifroggi — https://github.com/pifroggi/vs_temporalfix",
        "format":   "onnx",
        "variant":  "s1",
    },
    "temporalfix_s2_onnx": {
        "filename": "temporalfix_s2_v1_op18_fp16.onnx",
        "url":      "https://github.com/pifroggi/vs_temporalfix/raw/main/vs_temporalfix/models/temporalfix_s2_v1_op18_fp16.onnx",
        "size_mb":  1.14,
        "label":    "TemporalFix s2 ONNX fp16 (~1.1 MB, recommandé)",
        "credits":  "pifroggi — https://github.com/pifroggi/vs_temporalfix",
        "format":   "onnx",
        "variant":  "s2",
    },
    "temporalfix_s3_onnx": {
        "filename": "temporalfix_s3_v1_op18_fp16.onnx",
        "url":      "https://github.com/pifroggi/vs_temporalfix/raw/main/vs_temporalfix/models/temporalfix_s3_v1_op18_fp16.onnx",
        "size_mb":  1.14,
        "label":    "TemporalFix s3 ONNX fp16 (~1.1 MB, fort)",
        "credits":  "pifroggi — https://github.com/pifroggi/vs_temporalfix",
        "format":   "onnx",
        "variant":  "s3",
    },
    # ── Undistort — PyTorch PTH (xg416 / pifroggi, MIT) ────────────────────
    "undistort_tmt": {
        "filename": "undistort_dynamic_1st_stage.pth",
        "url":      "https://github.com/pifroggi/vs_undistort/raw/main/vs_undistort/dynamic_1st_stage.pth",
        "size_mb":  8.13,
        "label":    "Undistort TMT PTH (~8 MB)",
        "credits":  "xg416 / pifroggi — https://github.com/pifroggi/vs_undistort",
        "format":   "pth",
        "variant":  "tmt",
    },
    # ── Undistort — ONNX fp16 (op19, dynamic shapes) ──────────────────────
    "undistort_tmt_onnx": {
        "filename": "undistort_dynamic_1st_stage_op19_fp16.onnx",
        "url":      "https://github.com/pifroggi/vs_undistort/raw/main/vs_undistort/dynamic_1st_stage_op19_fp16_dynamic.onnx",
        "size_mb":  4.67,
        "label":    "Undistort TMT ONNX fp16 (~4.7 MB, recommandé GPU)",
        "credits":  "xg416 / pifroggi — https://github.com/pifroggi/vs_undistort",
        "format":   "onnx",
        "variant":  "tmt",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ModelManager
# ─────────────────────────────────────────────────────────────────────────────

class ModelManager:
    """
    Download-on-demand manager for neural post-processing weights.

    Parameters
    ----------
    weights_dir : str | Path | None
        Override default cache location.
        Default: <project_root>/runtimes/weights/
    """

    def __init__(self, weights_dir: Optional[str | Path] = None) -> None:
        if weights_dir is None:
            # Resolve relative to this file's location: src/core/ → project root
            _here = Path(__file__).resolve().parent.parent.parent
            weights_dir = _here / "runtimes" / "weights"
        self._dir = Path(weights_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Queries ──────────────────────────────────────────────────────────────

    def weight_path(self, model_name: str) -> Path:
        """Return the local path for a model's weight file."""
        info = self._get(model_name)
        return self._dir / info["filename"]

    def is_ready(self, model_name: str) -> bool:
        """True if weight file exists and is non-empty."""
        p = self.weight_path(model_name)
        return p.is_file() and p.stat().st_size > 1024

    def label(self, model_name: str) -> str:
        return MODELS[model_name]["label"]

    def size_mb(self, model_name: str) -> float:
        return MODELS[model_name]["size_mb"]

    def credits(self, model_name: str) -> str:
        return MODELS[model_name]["credits"]

    # ── Download ─────────────────────────────────────────────────────────────

    def download(
        self,
        model_name: str,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Download weight file for `model_name`.

        Parameters
        ----------
        model_name : str
            Key from MODELS registry.
        progress_cb : callable(downloaded_bytes, total_bytes) | None
            Called periodically during download.

        Returns
        -------
        Path to the downloaded file.
        """
        info = self._get(model_name)
        dest = self.weight_path(model_name)
        url  = info["url"]
        tmp  = dest.with_suffix(".part")

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Universal-SR-Studio/2.5.7"},
        )

        with urllib.request.urlopen(request, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 65536  # 64 KB

            with open(tmp, "wb") as fout:
                while True:
                    data = resp.read(chunk)
                    if not data:
                        break
                    fout.write(data)
                    downloaded += len(data)
                    if progress_cb is not None:
                        progress_cb(downloaded, total)

        # Atomic rename
        if dest.exists():
            dest.unlink()
        tmp.rename(dest)
        return dest

    def delete(self, model_name: str) -> None:
        """Remove cached weight file."""
        p = self.weight_path(model_name)
        if p.is_file():
            p.unlink()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get(self, model_name: str) -> dict:
        if model_name not in MODELS:
            raise KeyError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODELS.keys())}"
            )
        return MODELS[model_name]


# ── Module-level singleton (lazy init) ───────────────────────────────────────

_default_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    """Return the default ModelManager (created on first call)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelManager()
    return _default_manager
