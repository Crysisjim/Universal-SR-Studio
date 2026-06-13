"""
src/core/ort_session.py
═══════════════════════════════════════════════════════════════════════════════
Module-level OnnxRuntime InferenceSession cache.

Problem Chaiinner / every ONNX pipeline:
  Creating InferenceSession is slow (1-10s on GPU — CUDA context init, TRT
  engine build).  If recreated per frame or per upscale job, the first image
  of every batch "freezes" for several seconds.

Fix:
  Sessions stored in a PROCESS-WIDE dict keyed by (onnx_path, providers).
  Even if TemporalFixProcessor / UndistortProcessor instances are garbage-
  collected between jobs, the ORT session stays warm in GPU VRAM.

  Only evict if user explicitly calls evict() or the process exits.

Usage:
    from src.core.ort_session import get_ort_session
    sess = get_ort_session("/path/model.onnx", backend="ort")
    out  = sess.run(None, {inp_name: inp_np})

TRT engine cache:
  TRT engines are written to <weights_dir>/trt_engines/ so they are NOT
  rebuilt on every launch — only on first use per GPU model.

v2.5.7 — Universal SR Studio
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ONNX_OK = False
try:
    import onnxruntime as ort
    _ONNX_OK = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Provider helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_onnx_available() -> bool:
    return _ONNX_OK


def available_providers() -> List[str]:
    if not _ONNX_OK:
        return []
    return ort.get_available_providers()


def providers_for_backend(backend: str) -> List[str]:
    """
    Map user backend name → ORT provider chain (most preferred first).

    backend
    -------
    "trt"  → TensorRT EP → CUDA EP → CPU EP
    "ort"  → CUDA EP → CPU EP          (default, GPU auto)
    "cuda" → CUDA EP → CPU EP          (alias for ort)
    "cpu"  → CPU EP only

    Providers not present on the system are still included; ORT will log a
    warning and skip them, falling through to the next available one.
    CPU EP is always present and acts as final fallback.
    """
    if backend in ("trt", "tensorrt"):
        return ["TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider"]
    elif backend in ("ort", "cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:  # "cpu"
        return ["CPUExecutionProvider"]


def detect_active_backend(onnx_path: str) -> Optional[str]:
    """Return the backend key of a cached session for onnx_path, or None."""
    return _SESSION_CACHE.active_backend(onnx_path)


# ─────────────────────────────────────────────────────────────────────────────
# Session cache
# ─────────────────────────────────────────────────────────────────────────────

class _OrtSessionCache:
    """Thread-safe, process-wide InferenceSession registry."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple, "ort.InferenceSession"] = {}
        self._lock  = threading.Lock()

    def get(self,
            onnx_path: str,
            providers: List[str]) -> "ort.InferenceSession":
        """Return cached session, creating it on first call."""
        if not _ONNX_OK:
            raise ImportError(
                "onnxruntime-gpu not found.\n"
                "Install: pip install onnxruntime-gpu"
            )
        key = (onnx_path, tuple(providers))
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self._build(onnx_path, providers)
            return self._cache[key]

    def evict(self, onnx_path: Optional[str] = None) -> None:
        """Remove cached session(s).  None = clear everything."""
        with self._lock:
            if onnx_path is None:
                self._cache.clear()
            else:
                gone = [k for k in self._cache if k[0] == onnx_path]
                for k in gone:
                    del self._cache[k]

    def active_backend(self, onnx_path: str) -> Optional[str]:
        with self._lock:
            for (path, providers), _ in self._cache.items():
                if path == onnx_path:
                    # Reverse-map first provider to backend name
                    p0 = providers[0]
                    if p0 == "TensorrtExecutionProvider":
                        return "trt"
                    elif p0 == "CUDAExecutionProvider":
                        return "ort"
                    else:
                        return "cpu"
            return None

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build(onnx_path: str, providers: List[str]) -> "ort.InferenceSession":
        trt_cache = str(Path(onnx_path).parent / "trt_engines")
        Path(trt_cache).mkdir(parents=True, exist_ok=True)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern        = True
        opts.enable_cpu_mem_arena      = True
        # Uncomment to reduce ORT verbosity:
        # opts.log_severity_level = 3

        # Per-provider options
        prov_opts = []
        for p in providers:
            if p == "TensorrtExecutionProvider":
                prov_opts.append({
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path":   trt_cache,
                    "trt_fp16_enable":         True,
                    "trt_max_workspace_size":  2 << 30,  # 2 GB
                })
            elif p == "CUDAExecutionProvider":
                prov_opts.append({
                    "cudnn_conv_use_max_workspace": "1",
                })
            else:
                prov_opts.append({})

        return ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=providers,
            provider_options=prov_opts,
        )


# Module-level singleton
_SESSION_CACHE = _OrtSessionCache()


def get_ort_session(onnx_path: str, backend: str = "ort") -> "ort.InferenceSession":
    """
    Get (or create and cache) an ORT session.

    Parameters
    ----------
    onnx_path : str
        Absolute path to .onnx file.
    backend : str
        "ort" | "cuda"  → CUDA EP → CPU EP  (default)
        "trt"           → TRT EP → CUDA EP → CPU EP
        "cpu"           → CPU EP only

    Returns
    -------
    ort.InferenceSession  (loaded once, returned instantly on subsequent calls)
    """
    providers = providers_for_backend(backend)
    return _SESSION_CACHE.get(onnx_path, providers)
