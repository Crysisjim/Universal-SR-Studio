"""
temporal_fix.py — Temporal consistency post-processing (v2.5.7)

Modes
-----
classic                 Pure PyTorch, no weights. Motion-adaptive blend.
                        Window 5/7/9. Fast, always available.

model_s1/s2/s3          PyTorch PTH inference (pifroggi neural model).
                        ~2 MB weights. Window=7 fixed. GPU auto.

ort_s1/s2/s3            OnnxRuntime ONNX fp16 inference.
                        CUDA EP + CPU EP fallback. Fastest on GPU.
                        ✓ Session cached process-wide → no reload between jobs.

trt_s1/s2/s3            OnnxRuntime + TensorRT EP.
                        First run builds engine (~1 min). Subsequent: instant.
                        Requires CUDA + TRT libs.

cpu_s1/s2/s3            PyTorch CPU inference (no GPU needed).

API
---
    proc = TemporalFixProcessor(window=7, strength=0.5, mode="ort_s2")
    for frame_np in frames:            # float32 [H,W,3]  [0,1]
        out = proc.push(frame_np)
        if out is not None:
            save(out)
    for out in proc.flush():
        save(out)

v2.5.7 — Universal SR Studio
"""
from __future__ import annotations
from collections import deque
from typing import Optional, Iterator
import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# Probe: verify GPU actually executes kernels (sm_61 / Pascal fails on PyTorch 2.5+)
_CUDA_OK = False
if _TORCH_AVAILABLE:
    try:
        _t = torch.zeros(1, device="cuda")
        _ = _t + _t
        del _t
        _CUDA_OK = True
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Mode → model-key mapping
# ─────────────────────────────────────────────────────────────────────────────

# Modes that require PTH weights
_PTH_MODES = {"model_s1": "temporalfix_s1",
              "model_s2": "temporalfix_s2",
              "model_s3": "temporalfix_s3",
              "cpu_s1":   "temporalfix_s1",
              "cpu_s2":   "temporalfix_s2",
              "cpu_s3":   "temporalfix_s3"}

# Modes that require ONNX weights
_ORT_MODES  = {"ort_s1":   "temporalfix_s1_onnx",
               "ort_s2":   "temporalfix_s2_onnx",
               "ort_s3":   "temporalfix_s3_onnx",
               "trt_s1":   "temporalfix_s1_onnx",
               "trt_s2":   "temporalfix_s2_onnx",
               "trt_s3":   "temporalfix_s3_onnx"}

_NEURAL_MODES = set(_PTH_MODES) | set(_ORT_MODES)


def _resolve_device(pref: str, mode: str) -> str:
    if mode.startswith("cpu_"):
        return "cpu"
    if pref in ("auto", "cuda"):
        return "cuda" if _CUDA_OK else "cpu"
    return pref


def _np_to_tensor(arr: np.ndarray, device: str,
                  dtype: "torch.dtype") -> "torch.Tensor":
    t = torch.from_numpy(np.ascontiguousarray(arr).transpose(2, 0, 1)).unsqueeze(0)
    return t.to(device=device, dtype=dtype)


def _tensor_to_np(t: "torch.Tensor") -> np.ndarray:
    return np.clip(t.squeeze(0).permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────

class TemporalFixProcessor:
    """
    Sliding-window temporal consistency filter.

    Parameters
    ----------
    window : int        Odd 5/7/9.  Latency = window // 2.  Ignored for neural.
    strength : float    0-1.  Classic: blend weight.  Neural: mix orig ↔ model.
    precision : str     'float32' | 'float16'
    device : str        'auto' | 'cuda' | 'cpu'
    motion_threshold : float   Classic mode variance threshold.
    mode : str          See module docstring.
    """

    def __init__(
        self,
        window: int = 7,
        strength: float = 0.5,
        precision: str = "float32",
        device: str = "auto",
        motion_threshold: float = 0.008,
        mode: str = "classic",
    ) -> None:
        self.mode     = mode
        self.strength = float(strength)
        self.motion_threshold = float(motion_threshold)

        self._device = _resolve_device(device, mode)
        if _TORCH_AVAILABLE:
            self._dtype = (torch.float16
                           if precision == "float16"
                           else torch.float32)
        else:
            self._dtype = None

        # Neural modes use fixed window=7
        if mode in _NEURAL_MODES:
            window = 7

        if window % 2 == 0:
            raise ValueError(f"window must be odd, got {window}")
        self.window = window
        self.half   = window // 2

        self._buf: deque   = deque()
        self._n_pushed     = 0
        self._n_yielded    = 0

        # PTH model (lazy)
        self._pth_model    = None
        self._pth_model_hw = None

        # ORT state (session retrieved from global cache — never stored per-instance
        # to avoid holding a reference that would keep it hot in one instance but
        # create a second session in another)
        self._ort_path: Optional[str] = None   # resolved once at first use
        self._ort_inp_name: Optional[str] = None
        self._ort_dynamic_hw: Optional[bool] = None  # None = not yet checked

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, frame_np: np.ndarray) -> Optional[np.ndarray]:
        if not _TORCH_AVAILABLE or self.strength <= 0.0:
            return self._push_passthrough(frame_np)

        t = _np_to_tensor(frame_np, self._device, self._dtype)
        self._buf.append(t)
        self._n_pushed += 1

        if len(self._buf) < self.window:
            return None
        while len(self._buf) > self.window:
            self._buf.popleft()

        if self.mode in _ORT_MODES:
            result_t = self._process_centre_ort()
        elif self.mode in _PTH_MODES:
            result_t = self._process_centre_pth()
        else:
            result_t = self._process_centre_classic()

        self._buf.popleft()
        self._n_yielded += 1
        return _tensor_to_np(result_t)

    def flush(self) -> Iterator[np.ndarray]:
        if not self._buf:
            return
        last = self._buf[-1]
        n_pending = len(self._buf)
        for _ in range(n_pending):
            while len(self._buf) < self.window:
                self._buf.append(last)
            if not _TORCH_AVAILABLE or self.strength <= 0.0:
                raw = self._buf[0]
                yield (_tensor_to_np(raw)
                       if hasattr(raw, "cpu")   # torch tensor; numpy also has .squeeze()
                       else np.clip(raw, 0.0, 1.0))
            elif self.mode in _ORT_MODES:
                yield _tensor_to_np(self._process_centre_ort())
            elif self.mode in _PTH_MODES:
                yield _tensor_to_np(self._process_centre_pth())
            else:
                yield _tensor_to_np(self._process_centre_classic())
            self._buf.popleft()

    def reset(self) -> None:
        self._buf.clear()
        self._n_pushed = 0
        self._n_yielded = 0

    @property
    def latency(self) -> int:
        return self.half

    # ── Classic ───────────────────────────────────────────────────────────────

    def _push_passthrough(self, frame_np: np.ndarray) -> Optional[np.ndarray]:
        self._buf.append(frame_np)
        self._n_pushed += 1
        if len(self._buf) > self.window:
            oldest = self._buf.popleft()
            self._n_yielded += 1
            return oldest
        if len(self._buf) == self.window:
            centre = self._buf[self.half]
            self._n_yielded += 1
            return centre
        return None

    def _process_centre_classic(self) -> "torch.Tensor":
        frames  = list(self._buf)[:self.window]
        centre  = frames[self.half]
        stacked = torch.cat(frames, dim=0)
        t_mean  = stacked.mean(dim=0, keepdim=True)
        t_var   = stacked.var(dim=0, keepdim=True)
        t_var_lum = t_var.max(dim=1, keepdim=True).values
        thr = self.motion_threshold
        mask = torch.sigmoid((t_var_lum - thr) / (thr * 0.3 + 1e-8))
        blended = t_mean * (1.0 - mask) + centre * mask
        return (centre * (1.0 - self.strength) + blended * self.strength).clamp(0, 1)

    # ── PyTorch PTH ───────────────────────────────────────────────────────────

    def _load_pth(self, h: int, w: int) -> None:
        from src.core.model_manager import get_manager
        from src.core.models.temporalfix_arch import temporalfix_arch

        model_key = _PTH_MODES[self.mode]
        mgr = get_manager()
        if not mgr.is_ready(model_key):
            raise RuntimeError(
                f"Poids '{model_key}' non téléchargés. "
                "Ouvrez ⚙ Post-proc… → Télécharger."
            )
        weight_path = str(mgr.weight_path(model_key))
        model = temporalfix_arch(fixed_hw=None, conf_thresh=0.6, min_support=1,
                                 gate_slope=12.0, count_slope=4.0)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()

        use_fp16 = (self._dtype == torch.float16
                    and self._device == "cuda"
                    and torch.cuda.get_device_capability()[0] >= 7)
        if use_fp16:
            model.half()
        model.to(self._device)
        self._pth_model    = model
        self._pth_model_hw = (h, w)

    def _process_centre_pth(self) -> "torch.Tensor":
        frames = list(self._buf)[:7]
        h, w   = frames[0].shape[-2], frames[0].shape[-1]
        if self._pth_model is None or self._pth_model_hw != (h, w):
            self._load_pth(h, w)

        stack  = torch.stack([f.squeeze(0) for f in frames], dim=0).unsqueeze(0)
        centre = frames[3].to(torch.float32)
        with torch.inference_mode():
            corrected = self._pth_model(
                stack.to(self._device, dtype=self._dtype)
            ).to(torch.float32)

        if self.strength < 1.0:
            corrected = centre * (1.0 - self.strength) + corrected * self.strength
        return corrected.clamp(0, 1)

    # ── OnnxRuntime (ORT / TRT) ───────────────────────────────────────────────

    def _get_ort_backend(self) -> str:
        """Map mode prefix to ort_session backend key."""
        if self.mode.startswith("trt_"):
            return "trt"
        elif self.mode.startswith("cpu_"):
            return "cpu"
        return "ort"  # ort_s* → CUDA EP

    def _ensure_ort_ready(self) -> None:
        """Resolve ONNX path and meta on first call."""
        if self._ort_path is not None:
            return
        from src.core.model_manager import get_manager
        model_key = _ORT_MODES[self.mode]
        mgr = get_manager()
        if not mgr.is_ready(model_key):
            raise RuntimeError(
                f"Poids ONNX '{model_key}' non téléchargés. "
                "Ouvrez ⚙ Post-proc… → Télécharger."
            )
        self._ort_path = str(mgr.weight_path(model_key))

    def _get_ort_session(self):
        from src.core.ort_session import get_ort_session
        return get_ort_session(self._ort_path, backend=self._get_ort_backend())

    def _ort_get_meta(self, session) -> tuple:
        """Return (input_name, dynamic_hw) — checked once, cached."""
        if self._ort_inp_name is not None:
            return self._ort_inp_name, self._ort_dynamic_hw
        inp = session.get_inputs()[0]
        self._ort_inp_name  = inp.name
        shape = inp.shape  # e.g. [1, 21, -1, -1] or ['batch', 21, 'H', 'W']
        # Dynamic if last two dims are symbolic (str) or -1
        h_dim = shape[-2] if len(shape) >= 2 else -1
        w_dim = shape[-1] if len(shape) >= 1 else -1
        self._ort_dynamic_hw = (
            isinstance(h_dim, str) or h_dim in (-1, None) or
            isinstance(w_dim, str) or w_dim in (-1, None)
        )
        return self._ort_inp_name, self._ort_dynamic_hw

    def _process_centre_ort(self) -> "torch.Tensor":
        self._ensure_ort_ready()
        frames = list(self._buf)[:7]
        h, w   = frames[0].shape[-2], frames[0].shape[-1]
        centre = frames[3].to(torch.float32)

        # Build [1, 21, H, W] float16 numpy input (7 frames × 3 ch merged)
        # ONNX op18 export uses 4D input [B, 7*3, H, W], not 5D [B, 7, 3, H, W]
        stack_np = np.stack(
            [f.squeeze(0).cpu().numpy() for f in frames], axis=0
        )[np.newaxis].astype(np.float16)  # [1, 7, 3, H, W]
        stack_np = stack_np.reshape(1, stack_np.shape[1] * stack_np.shape[2], h, w)  # [1, 21, H, W]

        session = self._get_ort_session()
        inp_name, dynamic_hw = self._ort_get_meta(session)

        if dynamic_hw:
            out_np = session.run(None, {inp_name: stack_np})[0]  # [1, 3, H, W] fp16
        else:
            # Static ONNX shape → tile to match and reassemble
            exp_shape = session.get_inputs()[0].shape
            tile_h = int(exp_shape[-2])
            tile_w = int(exp_shape[-1])
            out_np = self._ort_tiled(session, inp_name, stack_np, h, w, tile_h, tile_w)

        corrected = torch.from_numpy(out_np.astype(np.float32)).squeeze(0)  # [3, H, W]
        corrected = corrected.unsqueeze(0).to(centre.device)  # [1, 3, H, W] — same device as centre

        if self.strength < 1.0:
            corrected = centre * (1.0 - self.strength) + corrected * self.strength
        return corrected.clamp(0, 1)

    @staticmethod
    def _ort_tiled(session, inp_name: str, stack_np: np.ndarray,
                   h: int, w: int, tile_h: int, tile_w: int) -> np.ndarray:
        """Tile inference for fixed-shape ONNX.  stack_np: [1,21,H,W] fp16 (7*3 merged)."""
        import math
        n_h = math.ceil(h / tile_h)
        n_w = math.ceil(w / tile_w)
        out = np.zeros((1, 3, h, w), dtype=np.float32)
        cnt = np.zeros((1, 1, h, w), dtype=np.float32)

        for ih in range(n_h):
            for iw in range(n_w):
                y0 = min(ih * tile_h, h - tile_h)
                x0 = min(iw * tile_w, w - tile_w)
                y1, x1 = y0 + tile_h, x0 + tile_w
                tile = stack_np[..., y0:y1, x0:x1]  # [1,21,th,tw]
                res  = session.run(None, {inp_name: tile})[0]   # [1,3,th,tw]
                out[..., y0:y1, x0:x1] += res.astype(np.float32)
                cnt[..., y0:y1, x0:x1] += 1.0

        return (out / cnt).astype(np.float16)
