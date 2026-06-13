"""
src/core/undistort.py — Temporal HF consistency correction (v2.5.7)

Modes
-----
classic                 Pure PyTorch, no weights.
                        LF/HF Gaussian split → temporal HF median → blend.
                        Window 3/5/7, latency = window//2.

model_tmt               PyTorch PTH (TMT — xg416/pifroggi, ~8 MB).
                        Non-overlapping batch of tmt_window frames.
                        Latency = tmt_window frames (default 10).

ort_tmt                 OnnxRuntime ONNX fp16 (dynamic shapes).
                        CUDA EP → CPU EP fallback.
                        Session cached process-wide → no reload between jobs.

trt_tmt                 OnnxRuntime + TensorRT EP.
                        First run builds TRT engine (~2-5 min). Subsequent: instant.

API (identical for all modes)
-----------------------------
    proc = UndistortProcessor(mode="ort_tmt", strength=0.4)
    for frame_np in frames:
        out = proc.push(frame_np)
        if out: save(out)
    for out in proc.flush():
        save(out)

v2.5.7 — Universal SR Studio
"""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Iterator, List, Optional

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# Probe: verify GPU actually executes kernels (sm_61 / Pascal fails on PyTorch 2.5+)
_CUDA_OK = False
if _TORCH_OK:
    try:
        _t = torch.zeros(1, device="cuda")
        _ = _t + _t
        del _t
        _CUDA_OK = True
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────

_ORT_TMT_MODES = {"ort_tmt", "trt_tmt"}
_PTH_TMT_MODES = {"model_tmt"}
_NEURAL_MODES  = _ORT_TMT_MODES | _PTH_TMT_MODES


class UndistortProcessor:
    """
    Temporal HF consistency corrector for SR output sequences.

    Parameters
    ----------
    window          Classic: temporal window (odd ≥ 3).  Latency = window // 2.
    strength        0-1. Correction intensity.  Rec 0.3-0.5.
    precision       'float32' | 'float16'
    device          'auto' | 'cuda' | 'cpu'
    gaussian_sigma  Classic only: LF kernel sigma.
    mode            'classic' | 'model_tmt' | 'ort_tmt' | 'trt_tmt'
    tmt_window      Neural: batch size.  Default 10.  Latency = tmt_window.
    tmt_patch       Neural PTH tiling size.  Default 256.
    """

    def __init__(
        self,
        window: int = 5,
        strength: float = 0.35,
        precision: str = "float32",
        device: str = "auto",
        gaussian_sigma: float = 1.5,
        mode: str = "classic",
        tmt_window: int = 10,
        tmt_patch: int = 256,
    ) -> None:
        if not _TORCH_OK:
            raise ImportError("UndistortProcessor requires PyTorch.")

        self.mode     = mode
        self.strength = float(max(0.0, min(1.0, strength)))
        self._dtype   = torch.float16 if precision == "float16" else torch.float32

        if device == "auto":
            self._device = torch.device("cuda" if _CUDA_OK else "cpu")
        elif device == "cuda":
            self._device = torch.device("cuda" if _CUDA_OK else "cpu")
        else:
            self._device = torch.device("cpu")

        # ── Classic mode ─────────────────────────────────────────────────────
        if window % 2 == 0:
            window += 1
        if window < 3:
            window = 3
        self.window  = window
        self.latency = window // 2   # overridden for neural modes below

        self._kernel = self._make_gaussian_kernel(
            sigma=gaussian_sigma, ksize=7
        ).to(self._device, dtype=self._dtype)

        self._buf: deque     = deque(maxlen=window)
        self._pending: deque = deque()

        # ── Neural / ORT batch mode ──────────────────────────────────────────
        self._tmt_window = max(2, tmt_window)
        self._tmt_patch  = tmt_patch
        self._raw_batch: List[np.ndarray] = []
        self._output_q: deque             = deque()

        if mode in _NEURAL_MODES:
            self.latency = self._tmt_window  # overwrite

        # PTH model (lazy)
        self._tmt_model = None

        # ORT state
        self._ort_path: Optional[str]      = None
        self._ort_inp_name: Optional[str]  = None
        self._ort_dynamic_hw               = None  # checked once

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, frame_np: np.ndarray) -> Optional[np.ndarray]:
        if self.mode in _NEURAL_MODES:
            return self._push_neural(frame_np)
        return self._push_classic(frame_np)

    def flush(self) -> Iterator[np.ndarray]:
        if self.mode in _NEURAL_MODES:
            yield from self._flush_neural()
        else:
            yield from self._flush_classic()

    def reset(self) -> None:
        self._buf.clear()
        self._pending.clear()
        self._raw_batch.clear()
        self._output_q.clear()

    # ── Classic ───────────────────────────────────────────────────────────────

    def _push_classic(self, frame_np: np.ndarray) -> Optional[np.ndarray]:
        t = self._to_tensor(frame_np)
        self._buf.append(t)
        self._pending.append(frame_np)
        if len(self._buf) < self.window:
            return None
        result_t = self._process_classic(list(self._buf))
        self._pending.popleft()
        return self._to_numpy(result_t)

    def _flush_classic(self) -> Iterator[np.ndarray]:
        if not self._pending or not self._buf:
            return
        last_tensor = self._buf[-1]
        while self._pending:
            buf_list = list(self._buf)
            while len(buf_list) < self.window:
                buf_list.append(last_tensor)
            result_t = self._process_classic(buf_list[:self.window])
            self._pending.popleft()
            self._buf.append(last_tensor)
            yield self._to_numpy(result_t)

    def _process_classic(self, frames: list) -> "torch.Tensor":
        centre  = frames[self.latency]
        lf_c    = self._lowpass(centre)
        hf_c    = (centre - lf_c).to(torch.float32)
        if self.strength <= 0.0:
            return centre.to(torch.float32).cpu()
        hf_fp32   = [(f - self._lowpass(f)).to(torch.float32) for f in frames]
        hf_stack  = torch.cat(hf_fp32, dim=0)
        hf_median = hf_stack.median(dim=0).values.unsqueeze(0)
        hf_cor    = hf_c * (1.0 - self.strength) + hf_median * self.strength
        return torch.clamp(lf_c.to(torch.float32) + hf_cor, 0.0, 1.0).cpu()

    def _lowpass(self, t: "torch.Tensor") -> "torch.Tensor":
        pad = self._kernel.shape[-1] // 2
        return F.conv2d(t.to(self._device, dtype=self._dtype),
                        self._kernel, padding=pad, groups=t.shape[1])

    # ── Neural batch (shared push / flush for PTH + ORT) ─────────────────────

    def _push_neural(self, frame_np: np.ndarray) -> Optional[np.ndarray]:
        self._raw_batch.append(frame_np)
        if len(self._raw_batch) >= self._tmt_window:
            self._run_neural_batch(self._raw_batch)
            self._raw_batch = []
        if self._output_q:
            return self._output_q.popleft()
        return None

    def _flush_neural(self) -> Iterator[np.ndarray]:
        while self._output_q:
            yield self._output_q.popleft()
        if self._raw_batch:
            n_real = len(self._raw_batch)
            last = self._raw_batch[-1]
            padded = list(self._raw_batch)
            while len(padded) < self._tmt_window:
                padded.append(last)
            self._run_neural_batch(padded)
            self._raw_batch = []
            for _ in range(n_real):
                if self._output_q:
                    yield self._output_q.popleft()
        while self._output_q:
            yield self._output_q.popleft()

    def _run_neural_batch(self, frames: List[np.ndarray]) -> None:
        """Dispatch to PTH or ORT inference, fill _output_q."""
        if self.mode in _ORT_TMT_MODES:
            corrected = self._infer_ort(frames)
        else:
            corrected = self._infer_pth(frames)

        for i, orig in enumerate(frames):
            c = corrected[i]
            if self.strength < 1.0:
                c = orig * (1.0 - self.strength) + c * self.strength
            self._output_q.append(np.clip(c, 0.0, 1.0).astype(np.float32))

    # ── PTH inference ─────────────────────────────────────────────────────────

    def _get_tmt_model(self):
        if self._tmt_model is not None:
            return self._tmt_model
        from src.core.model_manager import get_manager
        from src.core.models.unet3d_tmt import DetiltUNet3DS
        mgr = get_manager()
        if not mgr.is_ready("undistort_tmt"):
            raise RuntimeError(
                "Poids 'undistort_tmt' non téléchargés. "
                "Ouvrez ⚙ Post-proc… → Télécharger."
            )
        weight_path = str(mgr.weight_path("undistort_tmt"))
        model = DetiltUNet3DS(num_channels=3, feat_channels=[64, 256, 256, 512],
                              norm="LN", conv_type="dw", residual="conv",
                              interpolation="bilinear")
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        use_fp16 = (self._dtype == torch.float16
                    and self._device.type == "cuda"
                    and torch.cuda.get_device_capability()[0] >= 7)
        if use_fp16:
            model.half()
        model.to(self._device)
        self._tmt_model = model
        return model

    def _infer_pth(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        from src.core.models.tmt_inference import test_spatial_overlap
        model = self._get_tmt_model()
        tensors = [torch.from_numpy(np.ascontiguousarray(f).transpose(2, 0, 1))
                   for f in frames]
        batch = torch.stack(tensors, dim=0).to(torch.float32)  # [T,C,H,W]
        input_blk = batch.unsqueeze(0)                          # [1,T,C,H,W]
        with torch.inference_mode():
            recovered = test_spatial_overlap(
                input_blk, model,
                patch_height=self._tmt_patch,
                patch_width=self._tmt_patch,
                min_overlap=32,
                scales=[True, True, True],
                tile_device=torch.device("cpu"),
            )
        recovered = recovered.squeeze(0).float()  # [T,C,H,W]
        return [recovered[i].permute(1, 2, 0).numpy() for i in range(len(frames))]

    # ── ORT inference ─────────────────────────────────────────────────────────

    def _ort_backend(self) -> str:
        if self.mode == "trt_tmt":
            return "trt"
        return "ort"

    def _ensure_ort(self) -> None:
        if self._ort_path is not None:
            return
        from src.core.model_manager import get_manager
        mgr = get_manager()
        if not mgr.is_ready("undistort_tmt_onnx"):
            raise RuntimeError(
                "Poids ONNX 'undistort_tmt_onnx' non téléchargés. "
                "Ouvrez ⚙ Post-proc… → Télécharger."
            )
        self._ort_path = str(mgr.weight_path("undistort_tmt_onnx"))

    def _get_ort_session(self):
        from src.core.ort_session import get_ort_session
        return get_ort_session(self._ort_path, backend=self._ort_backend())

    def _ort_meta(self, session) -> tuple:
        if self._ort_inp_name is not None:
            return self._ort_inp_name, self._ort_dynamic_hw
        inp = session.get_inputs()[0]
        self._ort_inp_name = inp.name
        shape = inp.shape
        # undistort ONNX is dynamic (_dynamic in filename)
        h_dim = shape[-2] if len(shape) >= 2 else -1
        w_dim = shape[-1] if len(shape) >= 1 else -1
        self._ort_dynamic_hw = (
            isinstance(h_dim, str) or h_dim in (-1, None) or
            isinstance(w_dim, str) or w_dim in (-1, None)
        )
        return self._ort_inp_name, self._ort_dynamic_hw

    def _infer_ort(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        OnnxRuntime inference for a batch.
        Input: [1, T, C, H, W] fp16
        Output: [1, T, C, H, W] fp16
        Undistort ONNX has dynamic shapes → no tiling needed.
        """
        self._ensure_ort()
        T = len(frames)
        # Stack [T, C, H, W] → [1, T, C, H, W] fp16
        stack = np.stack(
            [np.ascontiguousarray(f).transpose(2, 0, 1) for f in frames],
            axis=0,
        )[np.newaxis].astype(np.float16)   # [1, T, C, H, W]

        session  = self._get_ort_session()
        inp_name, dynamic_hw = self._ort_meta(session)

        out_np = session.run(None, {inp_name: stack})[0]  # [1, T, C, H, W] fp16
        out    = out_np.astype(np.float32)[0]             # [T, C, H, W]

        return [out[i].transpose(1, 2, 0) for i in range(T)]  # list of [H,W,C]

    # ── Tensor helpers ────────────────────────────────────────────────────────

    def _to_tensor(self, frame_np: np.ndarray) -> "torch.Tensor":
        return (torch.from_numpy(np.ascontiguousarray(frame_np))
                .permute(2, 0, 1).unsqueeze(0)
                .to(self._device, dtype=self._dtype))

    def _to_numpy(self, t: "torch.Tensor") -> np.ndarray:
        return (t.squeeze(0).permute(1, 2, 0).cpu().float().numpy())

    @staticmethod
    def _make_gaussian_kernel(sigma: float = 1.5, ksize: int = 7) -> "torch.Tensor":
        coords = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        g      = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
        g      = g / g.sum()
        k2d    = g.unsqueeze(1) * g.unsqueeze(0)
        k2d    = k2d / k2d.sum()
        return k2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
