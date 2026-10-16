"""
GFISRv2 Upscale — Standalone.
Layout mirrors Quick Upscale tab from Universal SR Studio.
"""
import os
import sys
import json
import subprocess
import threading
import queue
import math
import time
import traceback
import contextlib
from tkinter import filedialog, StringVar, IntVar, BooleanVar, PanedWindow as _PanedWindow

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND_OK = True
except ImportError:
    _DND_OK = False

import customtkinter as ctk
from PIL import Image as _PILImage

# ── Path setup ─────────────────────────────────────────────────────────────
_FROZEN = getattr(sys, "_MEIPASS", None) is not None

if _FROZEN:
    _HERE    = sys._MEIPASS
    _EXE_DIR = os.path.dirname(sys.executable)
    _ROOT    = _EXE_DIR
else:
    _HERE    = os.path.dirname(os.path.abspath(__file__))
    _EXE_DIR = _HERE
    _ROOT    = os.path.abspath(os.path.join(_HERE, "..", ".."))  # tools/gfisrv2_upscale -> project root

_ARCH_DIR = os.path.join(_ROOT, "src", "core", "custom_archs")
_WORKER   = os.path.join(_HERE, "gfisrv2_worker.py") if not _FROZEN else None

for _p in (_HERE, _ARCH_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Portable deps: the CUDA PyTorch wheel (torch + numpy + deps) is installed here by the
# "Install Dependencies" dialog. torch is NOT bundled in the exe, so _deps is the only
# torch on the path — no shadow conflicts. _deps takes priority so it overrides anything.
_DEPS = os.path.join(_EXE_DIR, "_deps")
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

_SETTINGS_FILE = os.path.join(_EXE_DIR, "gfisrv2_settings.json")

# ── Theme ──────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
_ACCENT = "#3B8ED0"
_GREEN  = "#2ecc71"
_RED    = "#e74c3c"
_RED_HV = "#c0392b"
_GRAY   = "gray60"
_PANEL  = ("#E8E8E8", "#111827")


def _strip_dnd(s: str) -> str:
    s = s.strip()
    return s[1:-1] if s.startswith("{") and s.endswith("}") else s


def _load_ckpt(path: str) -> dict:
    """Load model checkpoint — supports .pth/.pt and .safetensors."""
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("params_ema", "params_g", "params", "state_dict"):
            if k in ckpt:
                return ckpt[k]
    return ckpt


# ── Inference helpers ──────────────────────────────────────────────────────

def _tile_inference(model, t, tile: int, pad: int, scale: int):
    import torch
    B, C, H, W = t.shape
    out = t.new_zeros(B, C, H * scale, W * scale)
    tx = math.ceil(W / tile)
    ty = math.ceil(H / tile)
    for y in range(ty):
        for x in range(tx):
            ox, oy = x * tile, y * tile
            x0 = max(ox - pad, 0); x1 = min(ox + tile + pad, W)
            y0 = max(oy - pad, 0); y1 = min(oy + tile + pad, H)
            ti = t[:, :, y0:y1, x0:x1]
            with torch.no_grad():
                to_ = model(ti)
            ix0 = (ox - x0) * scale; iy0 = (oy - y0) * scale
            ex = min(ox * scale + tile * scale, out.shape[3])
            ey = min(oy * scale + tile * scale, out.shape[2])
            aw = ex - ox * scale; ah = ey - oy * scale
            out[:, :, oy * scale:ey, ox * scale:ex] = \
                to_[:, :, iy0:iy0 + ah, ix0:ix0 + aw]
    return out


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
_EXT_MAP  = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif"}


def _detect_scale_from_ckpt(ckpt: dict) -> int:
    if "upscale.MetaUpsample" in ckpt:
        try:
            import torch
            meta = ckpt["upscale.MetaUpsample"]
            if hasattr(meta, "item"):
                pass
            else:
                s = int(meta[2].item() if hasattr(meta[2], "item") else meta[2])
                if s in (1, 2, 3, 4, 8):
                    return s
        except Exception:
            pass
    return 1


def _resolve_device(msg_q: queue.Queue):
    """Detect best available device: CUDA (if arch supported) → CPU."""
    def _put(*a): msg_q.put(a)
    import torch
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability()
            cap_str = f"sm_{cap[0]}{cap[1]}"
            supported = torch.cuda.get_arch_list()
            if supported and cap_str not in supported:
                _put("log", f"GPU {name} ({cap_str}) not in this torch build "
                            f"({', '.join(supported)}) -> CPU. "
                            f"Install the matching CUDA build via Install Dependencies.")
                return "cpu"
            _put("log", f"CUDA: {name} ({cap_str})")
            return "cuda"
        except Exception as e:
            _put("log", f"CUDA detect error: {e} -> CPU")
            return "cpu"
    _put("log", "No CUDA GPU detected -> CPU (slow). "
                "Install the CUDA build via Install Dependencies for GPU speed.")
    return "cpu"


def _load_model(params: dict, device, msg_q: queue.Queue):
    """Load and return (model, scale). Sends log to msg_q."""
    def _put(*a): msg_q.put(a)
    from gfisrv2_arch import GFISRV2
    _put("log", f"Loading model [{device}]...")
    ckpt  = _load_ckpt(params["model"])
    scale = params["scale"]
    if scale == 0:
        scale = _detect_scale_from_ckpt(ckpt)
        _put("log", f"Scale auto-detected: {scale}x")
    model = GFISRV2(
        dim=params["dim"], n_blocks=params["n_blocks"],
        mid_dim=params["mid_dim"], scale=scale,
        upsampler=params["upsampler"],
    )
    model.load_state_dict(ckpt, strict=True)
    model.eval().to(device)
    _put("log", f"Model OK  dim={params['dim']} n_blocks={params['n_blocks']} scale={scale}x")
    return model, scale


def _infer_images(files: list, out_dir: str, model, device, scale: int, params: dict,
                  msg_q: queue.Queue, stop_ev: threading.Event):
    """Process a list of image files with an already-loaded model."""
    import torch
    import numpy as np
    from PIL import Image as _PIL
    def _put(*a): msg_q.put(a)

    total = len(files)
    if total == 0:
        _put("error", "No image files found"); return
    os.makedirs(out_dir, exist_ok=True)

    fmt_up      = params["format"].upper()
    bits        = 16 if "16" in params.get("bits", "8") else 8
    use_amp     = params["amp"] and str(device) == "cuda"
    tile        = params["tile"]
    pad         = params["tile_pad"] if params["tile_pad"] > 0 else max(tile // 8, 8)
    ser_idx     = params["serialize_start"]
    model_stem  = os.path.splitext(os.path.basename(params["model"]))[0]
    out_ext     = _EXT_MAP.get(fmt_up, ".png")

    for i, fpath in enumerate(files):
        if stop_ev.is_set():
            _put("log", "Stopped."); break
        name = os.path.basename(fpath)
        _put("progress", i, total, name)

        img = _PIL.open(fpath).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        ctx = torch.autocast("cuda") if use_amp else contextlib.nullcontext()
        with ctx, torch.no_grad():
            out = _tile_inference(model, t, tile, pad, scale) if tile > 0 else model(t)

        # .float() is REQUIRED: under AMP the output is float16, and a float16
        # `* 65535` overflows (max 65504) -> inf -> uint16 cast = 0 = black image.
        arr_out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()

        kw = {}
        if bits == 16 and fmt_up in ("PNG", "TIFF"):
            arr16 = (arr_out * 65535.0).round().clip(0, 65535).astype(np.uint16)
            h_img, w_img = arr16.shape[:2]
            arr_be = arr16.astype('>u2')
            img_out = _PIL.frombuffer("RGB", (w_img, h_img), bytes(arr_be.data), "raw", "RGB;16B", 0, 1)
            if fmt_up == "TIFF":
                kw["compression"] = "lzw"
        else:
            img_out = _PIL.fromarray((arr_out * 255).astype("uint8"))
            if fmt_up in ("JPEG", "WEBP"):
                kw["quality"] = params["quality"]
            elif fmt_up == "TIFF":
                kw["compression"] = "lzw"

        stem = os.path.splitext(name)[0]
        if params["serialize"]:
            out_name = f"{ser_idx:05d}{out_ext}"
            ser_idx += 1
        elif params["modelname"]:
            out_name = f"{stem}_{model_stem}{out_ext}"
        else:
            out_name = stem + out_ext

        out_path = os.path.join(out_dir, out_name)
        img_out.save(out_path, format=fmt_up, **kw)
        _put("preview", fpath, out_path)

    _put("progress", total, total, "done")


def _collect_files(inp: str) -> list:
    return sorted(
        os.path.join(inp, f) for f in os.listdir(inp)
        if os.path.splitext(f)[1].lower() in _IMG_EXTS
    ) if os.path.isdir(inp) else [inp]


def _require_torch():
    """Import torch with a friendly error if it hasn't been installed into _deps yet."""
    try:
        import torch
        return torch
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "PyTorch is not installed yet. Click 'Install Dependencies' (top-right) to "
            "download the CUDA build (~2.7 GB) into the _deps/ folder, then restart the app."
        ) from e


def _infer_inline(params: dict, msg_q: queue.Queue, stop_ev: threading.Event):
    """Single-run inline inference for frozen exe (non-persistent)."""
    def _put(*a): msg_q.put(a)
    torch = None
    model = None
    try:
        _put("log", "Loading torch...")
        torch = _require_torch()
        _put("log", f"torch {torch.__version__}")
        device = _resolve_device(msg_q)
        model, scale = _load_model(params, device, msg_q)
        _infer_images(_collect_files(params["input"]), params["output"],
                      model, device, scale, params, msg_q, stop_ev)
        _put("done", None)
    except Exception as e:
        _put("error", f"{type(e).__name__}: {e}")
        _put("error", traceback.format_exc())
        _put("done", None)
    finally:
        # Release GPU memory back to the driver after a single run.
        try:
            del model
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def _infer_persistent(model_params: dict, job_q: queue.Queue,
                      msg_q: queue.Queue, stop_ev: threading.Event):
    """Persistent inline worker — load model once, process jobs from job_q.
    job_q items: params dict (image job) | None (stop sentinel).
    """
    def _put(*a): msg_q.put(a)
    torch = None
    model = None
    try:
        _put("log", "Loading torch (persistent)...")
        torch = _require_torch()
        _put("log", f"torch {torch.__version__}")
        device = _resolve_device(msg_q)
        model, scale = _load_model(model_params, device, msg_q)
        _put("log", "Persistent worker ready — model in memory.")

        while not stop_ev.is_set():
            try:
                job = job_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                _infer_images(_collect_files(job["input"]), job["output"],
                              model, device, scale, job, msg_q, stop_ev)
            except Exception as je:
                _put("error", f"{type(je).__name__}: {je}")
                _put("error", traceback.format_exc())
            # Free the inference scratch VRAM after each batch (keep the model loaded).
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            _put("done", None)

    except Exception as e:
        _put("error", f"{type(e).__name__}: {e}")
        _put("error", traceback.format_exc())
        _put("done", None)
    finally:
        # Free the model + GPU memory when the persistent worker exits (stop/close).
        try:
            del model
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                _put("log", "Persistent worker stopped — VRAM released.")
        except Exception:
            pass


# ── GPU Monitor widget ─────────────────────────────────────────────────────

class _GpuMonitor(ctk.CTkFrame):
    """3-bar compact GPU/VRAM/CPU monitor for top-right header area."""

    def __init__(self, parent, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        self._handle = None
        self._init_nvml()
        self._bars: dict  = {}
        self._lbls: dict  = {}
        self._build()
        self._tick()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._handle = None

    def _build(self):
        for name, color in (("GPU", "#3498db"), ("VRAM", "#9b59b6"), ("CPU", "#2ecc71")):
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=name, width=34,
                         font=ctk.CTkFont(size=9), text_color="gray55",
                         anchor="w").pack(side="left")
            bar = ctk.CTkProgressBar(row, width=88, height=7, corner_radius=2,
                                     progress_color=color)
            bar.set(0)
            bar.pack(side="left", padx=2)
            lbl = ctk.CTkLabel(row, text="—", width=44,
                               font=ctk.CTkFont(size=9), text_color="gray55", anchor="e")
            lbl.pack(side="left")
            self._bars[name] = bar
            self._lbls[name] = lbl

    def _tick(self):
        try:
            if self._handle is not None:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._bars["GPU"].set(util.gpu / 100.0)
                self._lbls["GPU"].configure(text=f"{util.gpu}%")
                pct = mem.used / max(mem.total, 1)
                gb  = mem.used / 1073741824
                tot = mem.total / 1073741824
                self._bars["VRAM"].set(pct)
                self._lbls["VRAM"].configure(text=f"{gb:.1f}/{tot:.0f}G")
        except Exception:
            pass
        try:
            import psutil
            cpu = psutil.cpu_percent() / 100.0
            self._bars["CPU"].set(cpu)
            self._lbls["CPU"].configure(text=f"{cpu*100:.0f}%")
        except ImportError:
            pass
        self.after(1000, self._tick)


# ── App base ───────────────────────────────────────────────────────────────

if _DND_OK:
    class _Base(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self):
            super().__init__()
            self.TkdndVersion = TkinterDnD._require(self)
else:
    class _Base(ctk.CTk):
        pass


class GFISRApp(_Base):

    def __init__(self):
        super().__init__()
        self.title("GFISRv2 — Quick Upscale")
        self.geometry("1100x740")
        self.minsize(860, 620)

        self._proc    = None
        self._stop_ev = threading.Event()
        self._msg_q:  queue.Queue = queue.Queue()
        self._running = False
        self._t0      = 0.0
        self._prev_refs = []
        # Persistent subprocess state (dev/non-frozen)
        self._persistent_proc  = None
        self._persistent_model = None
        # Persistent inline worker state (frozen exe)
        self._pworker_thread = None
        self._pworker_jobq   = None
        self._pworker_model  = None

        # Vars
        self.v_model     = StringVar()
        self.v_input     = StringVar()
        self.v_output    = StringVar()
        self.v_same_src  = BooleanVar(value=True)
        self.v_subfolder = BooleanVar(value=True)
        self.v_modelname = BooleanVar(value=False)
        self.v_serialize = BooleanVar(value=False)
        self.v_ser_start = StringVar(value="0")
        self.v_format    = StringVar(value="PNG")
        self.v_bits      = StringVar(value="16 bits")
        self.v_quality   = StringVar(value="95")
        self.v_scale     = StringVar(value="Auto")
        self.v_tile      = StringVar(value="Auto")
        self.v_tile_pad  = StringVar(value="Auto")
        self.v_amp        = BooleanVar(value=True)
        self.v_persistent = BooleanVar(value=False)
        # Arch (hidden by default)
        self.v_dim       = IntVar(value=48)
        self.v_n_blocks  = IntVar(value=24)
        self.v_mid_dim   = IntVar(value=32)
        self.v_upsampler = StringVar(value="dysample")

        self._load_settings()
        if getattr(self, "_saved_geometry", None):
            try:
                self.geometry(self._saved_geometry)
            except Exception:
                pass

        self.v_model.trace_add("write", self._on_model_change)
        self.v_input.trace_add("write", self._sync_output)
        self.v_same_src.trace_add("write", self._sync_output)
        self.v_subfolder.trace_add("write", self._sync_output)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll()

    # ── Build ────────────────────────────────────────────────────────────

    def _build_ui(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=14, pady=10)

        # Title row: left=title, right=GPU bars + Install button
        hdr = ctk.CTkFrame(main, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, 6))

        title_col = ctk.CTkFrame(hdr, fg_color="transparent")
        title_col.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(title_col, text="Quick Upscale", font=ctk.CTkFont("Roboto", 24, "bold"),
                     text_color=_ACCENT, anchor="w").pack(fill="x")
        ctk.CTkLabel(title_col, text="GFISRv2 — deband / deblur / upscale — standalone",
                     font=ctk.CTkFont(size=12), text_color=_GRAY, anchor="w").pack(fill="x")

        right_col = ctk.CTkFrame(hdr, fg_color="transparent")
        right_col.pack(side="right", padx=(16, 0))
        _tip(ctk.CTkButton(right_col, text="⬇ Install Dependencies", width=165, height=26,
                      fg_color="gray28", hover_color="gray38",
                      command=self._install_deps_dialog),
             "Download PyTorch (CUDA) into _deps/ next to the exe — auto-picks the build for your GPU. "
             "Needed once before the first upscale.").pack(anchor="e", pady=(2, 6))
        self._gpu_monitor = _GpuMonitor(right_col)
        self._gpu_monitor.pack(anchor="e")

        # ── Model ──
        mr = ctk.CTkFrame(main, fg_color="transparent")
        mr.pack(fill="x", pady=1)
        ctk.CTkLabel(mr, text="Model (.pth / .safetensors):", width=220, anchor="w").pack(side="left")
        e_model = ctk.CTkEntry(mr, textvariable=self.v_model)
        e_model.pack(side="left", fill="x", expand=True, padx=4)
        _Tooltip(e_model, "GFISRv2 model file (.safetensors/.pth/.pt). Only GFISRv2-architecture "
                          "models work here. Drag & drop supported.")
        ctk.CTkButton(mr, text="...", width=32, command=self._pick_model).pack(side="left")
        self._bind_dnd(e_model, self.v_model)

        # ── Source ──
        ir = ctk.CTkFrame(main, fg_color="transparent")
        ir.pack(fill="x", pady=1)
        ctk.CTkLabel(ir, text="Source (image or folder):", width=220, anchor="w").pack(side="left")
        e_input = ctk.CTkEntry(ir, textvariable=self.v_input)
        e_input.pack(side="left", fill="x", expand=True, padx=4)
        _Tooltip(e_input, "A single image or a folder (batch). All images in the folder are processed. "
                          "Drag & drop supported.")
        ctk.CTkButton(ir, text="Image",  width=60, command=self._pick_image).pack(side="left", padx=(0, 2))
        ctk.CTkButton(ir, text="Folder", width=70, command=self._pick_folder_input).pack(side="left")
        self._bind_dnd(e_input, self.v_input)

        # ── Output ──
        outr = ctk.CTkFrame(main, fg_color="transparent")
        outr.pack(fill="x", pady=1)
        ctk.CTkLabel(outr, text="Output folder:", width=220, anchor="w").pack(side="left")
        e_out = ctk.CTkEntry(outr, textvariable=self.v_output)
        e_out.pack(side="left", fill="x", expand=True, padx=4)
        _Tooltip(e_out, "Where results are written. Auto-filled when 'Same folder as source' is on.")
        ctk.CTkButton(outr, text="...", width=32, command=self._pick_folder_output).pack(side="left")
        self._bind_dnd(e_out, self.v_output)

        # ── Output options row ──
        op1 = self._fixed_row(main, 32)
        _spacer(op1)
        _tip(ctk.CTkCheckBox(op1, text="Same folder as source", variable=self.v_same_src,
                        width=175, command=self._sync_output),
             "Write results next to the source files instead of a separate output folder.").pack(side="left", padx=(0, 10))
        self._cb_subfolder = ctk.CTkCheckBox(op1, text='"upscaled/" subfolder',
                                              variable=self.v_subfolder, width=165,
                                              command=self._sync_output)
        self._cb_subfolder.pack(side="left", padx=(0, 10))
        _Tooltip(self._cb_subfolder,
                 'Put results in an "upscaled/" subfolder of the source folder (avoids overwriting originals).')
        _tip(ctk.CTkCheckBox(op1, text="Model name in file", variable=self.v_modelname,
                        width=150),
             "Append the model name to each output filename (e.g. frame_GFISRV2.png).").pack(side="left", padx=(0, 10))
        _vsep(op1)
        _tip(ctk.CTkCheckBox(op1, text="Serialization", variable=self.v_serialize,
                        width=120, command=self._on_serialize_toggle),
             "Rename outputs as a zero-padded sequence (00000, 00001, ...) starting at 'Start'.").pack(side="left", padx=(0, 4))
        ctk.CTkLabel(op1, text="Start:").pack(side="left", padx=(0, 3))
        self._ser_entry = ctk.CTkEntry(op1, textvariable=self.v_ser_start, width=52)
        self._ser_entry.pack(side="left", padx=(0, 10))
        _Tooltip(self._ser_entry, "First index used by serialized numbering.")
        _vsep(op1)
        ctk.CTkLabel(op1, text="Format:").pack(side="left", padx=(0, 4))
        _tip(ctk.CTkOptionMenu(op1, variable=self.v_format,
                          values=["PNG", "JPEG", "WEBP", "TIFF", "BMP"],
                          width=90, command=self._on_format_change),
             "Output image format. PNG/TIFF are lossless and support 16-bit. JPEG/WEBP are lossy.").pack(side="left", padx=(0, 6))
        ctk.CTkLabel(op1, text="Bits:").pack(side="left", padx=(0, 4))
        self._opt_bits = ctk.CTkOptionMenu(op1, variable=self.v_bits,
                                            values=["8 bits", "16 bits"],
                                            width=80)
        self._opt_bits.pack(side="left", padx=(0, 6))
        _Tooltip(self._opt_bits,
                 "Bit depth per channel. 16-bit keeps more gradient precision (best for deband); PNG/TIFF only.")
        ctk.CTkLabel(op1, text="Quality:").pack(side="left", padx=(0, 4))
        _tip(ctk.CTkOptionMenu(op1, variable=self.v_quality,
                          values=["75", "80", "85", "90", "95", "100"],
                          width=70),
             "Compression quality for lossy formats (JPEG/WEBP). Higher = better but larger.").pack(side="left")
        _spacer(op1)
        self._on_serialize_toggle()

        # ── Scale / Tile / AMP row ──
        op2 = self._fixed_row(main, 32)
        _spacer(op2)
        _tip(ctk.CTkCheckBox(op2, text="Persistent", variable=self.v_persistent,
                        width=100),
             "Keep the model loaded in VRAM between batches (faster repeated runs). "
             "Scratch VRAM is freed after each batch; the model is freed on Stop.").pack(side="left", padx=(0, 12))
        _vsep(op2)
        ctk.CTkLabel(op2, text="Scale:").pack(side="left")
        _tip(ctk.CTkOptionMenu(op2, variable=self.v_scale,
                          values=["Auto", "1", "2", "3", "4", "8"],
                          width=80),
             "Output scale factor. 'Auto' reads it from the model (this deband model is 1x).").pack(side="left", padx=5)
        ctk.CTkLabel(op2, text="Tile:").pack(side="left", padx=(12, 0))
        _tip(ctk.CTkOptionMenu(op2, variable=self.v_tile,
                          values=["Auto", "128", "192", "256", "384", "512", "768",
                                  "0 (no tile)"],
                          width=115),
             "Split large images into tiles to fit VRAM. Smaller tiles = less VRAM but slower. "
             "'0 (no tile)' processes the whole image at once.").pack(side="left", padx=5)
        ctk.CTkLabel(op2, text="Tile Pad:").pack(side="left", padx=(12, 0))
        _tip(ctk.CTkOptionMenu(op2, variable=self.v_tile_pad,
                          values=["Auto", "8", "16", "24", "32", "48", "64"],
                          width=90),
             "Overlap (in pixels) between tiles to hide seams. Higher = fewer artifacts, slightly slower.").pack(side="left", padx=5)
        _tip(ctk.CTkCheckBox(op2, text="AMP (FP16)", variable=self.v_amp,
                        width=100),
             "Automatic mixed precision (half float). Faster + less VRAM on CUDA, "
             "tiny quality trade-off. CUDA only.").pack(side="left", padx=12)
        _vsep(op2)
        _tip(ctk.CTkButton(op2, text="Arch params...", width=115, height=26,
                      fg_color="gray30", hover_color="gray40",
                      command=self._open_arch_dialog),
             "Manually set the GFISRv2 architecture params (dim, n_blocks, mid_dim, upsampler). "
             "Normally auto-detected from the model.").pack(side="left", padx=(0, 6))
        _tip(ctk.CTkButton(op2, text="Auto-detect", width=95, height=26,
                      fg_color="gray30", hover_color="gray40",
                      command=self._autodetect),
             "Read the architecture params + scale directly from the selected model file.").pack(side="left", padx=(0, 6))
        _tip(ctk.CTkButton(op2, text="⚙ GPU Setup", width=100, height=26,
                      fg_color="gray30", hover_color="gray40",
                      command=self._gpu_setup_dialog),
             "Show GPU / driver / PyTorch status and whether CUDA is active.").pack(side="left")
        _spacer(op2)

        # ── Run / Stop ──
        run_row = ctk.CTkFrame(main, fg_color="transparent")
        run_row.pack(fill="x", pady=(12, 3))
        self._btn_run = ctk.CTkButton(
            run_row, text="⚡ Run Upscale", fg_color=_GREEN,
            hover_color="#27ae60", font=ctk.CTkFont(size=13, weight="bold"),
            command=self._run)
        self._btn_run.pack(side="left", fill="x", expand=True, padx=(0, 5))
        _Tooltip(self._btn_run, "Process the source with the selected model and settings.")
        self._btn_stop = ctk.CTkButton(
            run_row, text="⏹ Stop", fg_color=_RED, hover_color=_RED_HV,
            width=110, state="disabled", command=self._stop)
        self._btn_stop.pack(side="left", padx=(0, 8))
        _Tooltip(self._btn_stop, "Stop the current run and free all GPU memory (model + scratch).")

        # ── Progress ──
        pr = ctk.CTkFrame(main, fg_color="transparent")
        pr.pack(fill="x")
        self._pbar = ctk.CTkProgressBar(pr)
        self._pbar.pack(side="left", fill="x", expand=True)
        self._pbar.set(0)
        self._lbl_pct = ctk.CTkLabel(pr, text="0%", width=42, anchor="e")
        self._lbl_pct.pack(side="left", padx=(6, 0))

        self._lbl_timing = ctk.CTkLabel(main, text="", font=ctk.CTkFont("Consolas", 11),
                                         text_color="#9B59B6", anchor="w")
        self._lbl_timing.pack(fill="x", pady=(1, 0))

        # ── PanedWindow: log | preview ──
        paned = _PanedWindow(main, orient="vertical", sashwidth=6, sashrelief="flat",
                              bg="#1a1a1a", bd=0)
        paned.pack(fill="both", expand=True, pady=5)

        log_host = ctk.CTkFrame(paned, fg_color="transparent")
        paned.add(log_host, height=130, minsize=60)
        self._log = ctk.CTkTextbox(log_host, font=ctk.CTkFont("Consolas", 11),
                                    fg_color=("#f0f0f0", "#0d1520"), text_color="#a8c0d8")
        self._log.pack(fill="both", expand=True)

        prev_host = ctk.CTkFrame(paned, fg_color=_PANEL, corner_radius=6)
        paned.add(prev_host, minsize=80)
        for side, attr, label in [("left", "_prev_in", "Source"),
                                   ("right", "_prev_out", "Result")]:
            col = ctk.CTkFrame(prev_host, fg_color="transparent")
            col.pack(side=side, fill="both", expand=True, padx=4, pady=4)
            ctk.CTkLabel(col, text=label, font=ctk.CTkFont(size=10), text_color=_GRAY).pack()
            lbl = ctk.CTkLabel(col, text="—", fg_color=("#D0D0D0", "#1e293b"), corner_radius=4)
            lbl.pack(fill="both", expand=True)
            setattr(self, attr, lbl)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _fixed_row(self, parent, h: int):
        f = ctk.CTkFrame(parent, fg_color="transparent", height=h)
        f.pack_propagate(False)
        f.pack(fill="x", pady=(2, 0))
        return f

    def _bind_dnd(self, widget, var: StringVar):
        if not _DND_OK:
            return
        def _drop(ev, v=var):
            v.set(_strip_dnd(ev.data))
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", _drop)

    def _on_format_change(self, fmt: str):
        supports_16 = fmt in ("PNG", "TIFF")
        self._opt_bits.configure(state="normal" if supports_16 else "disabled")
        if not supports_16:
            self.v_bits.set("8 bits")

    def _on_model_change(self, *_):
        mp = self.v_model.get().strip()
        if mp and os.path.isfile(mp):
            if hasattr(self, "_autodetect_job"):
                try: self.after_cancel(self._autodetect_job)
                except Exception: pass
            self._autodetect_job = self.after(600, self._autodetect)
            # Kill persistent workers on model change so next run reloads fresh
            if self._persistent_model and self._persistent_model != mp:
                self._kill_persistent()
            if self._pworker_model and self._pworker_model != mp:
                self._stop_pworker()

    def _stop_pworker(self):
        """Stop the persistent inline worker (frozen exe mode)."""
        if self._pworker_jobq:
            self._pworker_jobq.put(None)  # stop sentinel
        self._pworker_jobq   = None
        self._pworker_thread = None
        self._pworker_model  = None

    def _kill_persistent(self):
        if self._persistent_proc and self._persistent_proc.poll() is None:
            try:
                self._persistent_proc.stdin.write('{"quit":true}\n')
                self._persistent_proc.stdin.flush()
            except Exception:
                pass
            try:
                self._persistent_proc.terminate()
            except Exception:
                pass
        self._persistent_proc  = None
        self._persistent_model = None

    def _parse_worker_line(self, line: str) -> bool:
        """Parse one stdout line from worker. Returns True when DONE received."""
        if line.startswith("PROGRESS:"):
            pts = line.split(":", 3)
            if len(pts) >= 3:
                self._msg_q.put(("progress", int(pts[1]), max(int(pts[2]), 1),
                                 pts[3] if len(pts) > 3 else ""))
        elif line.startswith("PREVIEW:"):
            parts = line[8:].split("|", 1)
            if len(parts) == 2 and os.path.isfile(parts[0]) and os.path.isfile(parts[1]):
                self._msg_q.put(("preview", parts[0], parts[1]))
        elif line.startswith("LOG:"):
            self._msg_q.put(("log", line[4:]))
        elif line.startswith("ERROR:"):
            self._msg_q.put(("error", line[6:]))
        elif line == "DONE":
            self._msg_q.put(("done", None))
            self._running = False
            return True
        elif line:
            self._msg_q.put(("log", line))
        return False

    def _gpu_setup_dialog(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("GPU Setup — PyTorch / CUDA")
        dlg.geometry("560x420")
        dlg.resizable(True, True)
        dlg.grab_set()

        info_var = StringVar(value="Detecting GPU...")
        ctk.CTkLabel(dlg, textvariable=info_var,
                     font=ctk.CTkFont("Consolas", 11),
                     justify="left", anchor="w", wraplength=520).pack(
                     fill="x", padx=12, pady=(12, 6))

        sep = ctk.CTkFrame(dlg, height=1, fg_color="gray40")
        sep.pack(fill="x", padx=12, pady=4)

        hint = ctk.CTkLabel(dlg, text="", text_color=_GRAY,
                            font=ctk.CTkFont(size=11),
                            justify="left", anchor="w", wraplength=520)
        hint.pack(fill="x", padx=12, pady=(4, 6))

        def _detect():
            lines, tip = [], ""
            # GPU info via nvidia-smi (works even before torch is installed)
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=8, creationflags=0x08000000)
                row = [l.strip() for l in r.stdout.splitlines() if l.strip()]
                if row:
                    name, cap, drv = [s.strip() for s in row[0].split(",")]
                    lines.append(f"GPU     : {name}")
                    lines.append(f"Compute : {cap}   Driver : {drv}")
                else:
                    lines.append("GPU     : no NVIDIA GPU detected")
            except Exception:
                lines.append("GPU     : nvidia-smi not available (no NVIDIA driver?)")

            # torch status
            try:
                import torch
                lines.append(f"PyTorch : {torch.__version__}")
                if torch.cuda.is_available():
                    cap = torch.cuda.get_device_capability()
                    cap_str = f"sm_{cap[0]}{cap[1]}"
                    supported = torch.cuda.get_arch_list()
                    if cap_str in supported:
                        lines.append(f"Status  : [OK] CUDA active ({cap_str})")
                    else:
                        lines.append(f"Status  : [!!] {cap_str} not in torch build {supported}")
                        tip = "Installed torch does not support this GPU. Reinstall via Install Dependencies."
                else:
                    lines.append("Status  : torch present but CUDA not available (CPU mode)")
                    tip = "Running on CPU (slow)."
            except Exception:
                lines.append("PyTorch : not installed")
                tip = ("PyTorch is not installed. Click 'Install Dependencies' (top-right) to "
                       "download the CUDA build (~2.7 GB) — then restart. The app runs on CPU "
                       "until then.")

            info_var.set("\n".join(lines))
            hint.configure(text=tip)

        threading.Thread(target=_detect, daemon=True).start()
        ctk.CTkButton(dlg, text="Close", command=dlg.destroy).pack(pady=10)

    def _install_deps_dialog(self):
        """Install GPU dependencies (torch-directml etc.) into _deps/ next to exe."""
        import shutil

        dlg = ctk.CTkToplevel(self)
        dlg.title("Install Dependencies")
        dlg.geometry("600x680")
        dlg.resizable(True, True)
        dlg.grab_set()

        deps_dir = os.path.join(_EXE_DIR, "_deps")

        ctk.CTkLabel(dlg,
            text="Installs GPU packages into  _deps/  next to the exe.\n"
                 "No system-wide changes. Restart the app after installing.",
            font=ctk.CTkFont(size=11), text_color=_GRAY,
            justify="left", anchor="w", wraplength=560).pack(fill="x", padx=14, pady=(12, 2))

        ctk.CTkLabel(dlg,
            text=f"Target: {deps_dir}",
            font=ctk.CTkFont("Consolas", 10), text_color="#9B59B6",
            anchor="w").pack(fill="x", padx=14, pady=(0, 8))

        sep = ctk.CTkFrame(dlg, height=1, fg_color="gray40")
        sep.pack(fill="x", padx=12, pady=2)

        # Plain frame — no scroll needed for 4 packages
        scroll = ctk.CTkFrame(dlg, fg_color="transparent")
        scroll.pack(fill="x", padx=10, pady=4)

        log_box = ctk.CTkTextbox(dlg, height=90, font=ctk.CTkFont("Consolas", 10))
        log_box.pack(fill="x", padx=10, pady=(0, 4))
        log_box.configure(state="disabled")

        def _log_dlg(msg):
            log_box.configure(state="normal")
            log_box.insert("end", msg + "\n")
            log_box.see("end")
            log_box.configure(state="disabled")

        def _pip_prefix():
            """Return the command prefix to run pip, or None. Prefers the bundled
            embeddable Python 3.12 (no system Python needed, guaranteed cp312 match);
            falls back to a system Python of the same version."""
            # 1) Bundled embeddable Python + pip.pyz (true standalone)
            embed = os.path.join(_HERE, "pyembed", "python.exe")
            pippyz = os.path.join(_HERE, "pip.pyz")
            if os.path.isfile(embed) and os.path.isfile(pippyz):
                return [embed, pippyz]
            # 2) Dev mode: current interpreter
            if not _FROZEN:
                return [sys.executable, "-m", "pip"]
            # 3) System Python matching the frozen runtime version
            cur = f"{sys.version_info.major}.{sys.version_info.minor}"
            py_launcher = shutil.which("py")
            if py_launcher:
                try:
                    r = subprocess.run([py_launcher, f"-{cur}", "--version"],
                                       capture_output=True, text=True, timeout=5,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
                    if r.returncode == 0:
                        return [py_launcher, f"-{cur}", "-m", "pip"]
                except Exception:
                    pass
            for name in (f"python{cur}", f"python{sys.version_info.major}", "python", "python3"):
                p = shutil.which(name)
                if p:
                    return [p, "-m", "pip"]
            return None

        def _pkg_included(import_name: str) -> bool:
            """True if the package is already bundled inside the exe."""
            try:
                __import__(import_name)
                return True
            except Exception:
                return False

        # ── CUDA PyTorch card ────────────────────────────────────────────
        card = ctk.CTkFrame(scroll, fg_color=("gray88", "gray18"), corner_radius=6)
        card.pack(fill="x", pady=3, padx=2)

        title_lbl = ctk.CTkLabel(card, text="PyTorch CUDA  (NVIDIA GPU acceleration)",
                                 font=ctk.CTkFont(size=12, weight="bold"),
                                 anchor="w")
        title_lbl.pack(fill="x", padx=8, pady=(8, 1))
        gpu_lbl = ctk.CTkLabel(card, text="Detecting GPU...", text_color=_GRAY,
                               font=ctk.CTkFont(size=11), anchor="w",
                               wraplength=560, justify="left")
        gpu_lbl.pack(fill="x", padx=8)
        ctk.CTkLabel(card,
                     text="Downloads torch 2.7.0 + numpy (~2.7 GB) into _deps/. "
                          "The right build is auto-selected from your GPU; override below if needed.",
                     text_color=_GRAY, font=ctk.CTkFont(size=10), anchor="w",
                     wraplength=560, justify="left").pack(fill="x", padx=8, pady=(2, 4))

        _PYPI_INDEX = "https://pypi.org/simple"
        # variant label -> pytorch wheel tag ; "auto" resolved from the GPU compute capability
        _VARIANTS = {
            "Auto-detect  (recommended)":              "auto",
            "CUDA 12.6  —  GTX 9xx … RTX 40xx (sm 50-90)": "cu126",
            "CUDA 12.8  —  RTX 50xx Blackwell (sm 100+)":  "cu128",
            "CPU only  —  no NVIDIA GPU (slow)":           "cpu",
        }
        variant_var = StringVar(value="Auto-detect  (recommended)")

        sel_row = ctk.CTkFrame(card, fg_color="transparent")
        sel_row.pack(fill="x", padx=8, pady=(0, 2))
        ctk.CTkLabel(sel_row, text="Build:", width=44, anchor="w").pack(side="left")
        ctk.CTkOptionMenu(sel_row, variable=variant_var, values=list(_VARIANTS.keys()),
                          width=420).pack(side="left")

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=(4, 8))
        cmd_lbl = ctk.CTkLabel(btn_row, text="",
                               font=ctk.CTkFont("Consolas", 10), text_color="gray45",
                               anchor="w")
        cmd_lbl.pack(side="left", fill="x", expand=True)
        install_btn = ctk.CTkButton(btn_row, text="Install", width=90, height=28,
                                    fg_color=_ACCENT, hover_color="#2980b9")
        install_btn.pack(side="right")

        _detected_cap = {"cap": None}

        def _resolve_tag() -> str:
            """Return the pytorch wheel tag for the current selection (auto uses GPU cap)."""
            tag = _VARIANTS.get(variant_var.get(), "auto")
            if tag != "auto":
                return tag
            cap = _detected_cap["cap"]
            if not cap:
                return "cu126"
            try:
                major, minor = [int(x) for x in cap.split(".")]
                sm = major * 10 + minor
            except Exception:
                return "cu126"
            if sm >= 100:   # Blackwell (RTX 50xx)
                return "cu128"
            if sm >= 50:    # Maxwell .. Hopper (GTX 9xx .. RTX 40xx, incl. GTX 1080 Ti sm_61)
                return "cu126"
            return "cpu"    # too old for modern torch CUDA

        def _torch_installed() -> bool:
            # Check the folder only — avoid importing torch (heavy) just to update a button.
            return os.path.isdir(os.path.join(deps_dir, "torch"))

        def _detect_gpu():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=8,
                    creationflags=0x08000000)
                lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
                if not lines:
                    return None, None
                name, cap = [s.strip() for s in lines[0].split(",")]
                return name, cap
            except Exception:
                return None, None

        def _pip_system(args, on_done):
            """Run pip via the bundled embeddable Python (or a system Python fallback),
            streaming output to the log box. The frozen exe cannot run pip in-process
            reliably (distlib resource finder), so we use an external interpreter."""
            prefix = _pip_prefix()
            if not prefix:
                _log_dlg("ERROR: no Python available to run pip. (The bundled embeddable "
                         "Python is missing — reinstall the app, or install Python 3.12.)")
                on_done(1)
                return
            cmd = prefix + args
            def _run():
                rc = 1
                try:
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        creationflags=0x08000000)  # CREATE_NO_WINDOW
                    for line in proc.stdout:
                        dlg.after(0, _log_dlg, line.rstrip())
                    rc = proc.wait()
                except Exception as e:
                    dlg.after(0, _log_dlg, f"ERROR: {e}")
                dlg.after(0, on_done, rc)
            threading.Thread(target=_run, daemon=True).start()

        def _update_cmd(*_):
            tag = _resolve_tag()
            cmd_lbl.configure(text=f"pip install --target _deps/ torch==2.7.0 numpy  ({tag})")
        variant_var.trace_add("write", _update_cmd)

        def _do_install():
            tag = _resolve_tag()
            install_btn.configure(state="disabled", text="Installing…")
            os.makedirs(deps_dir, exist_ok=True)
            _log_dlg(f"Installing PyTorch ({tag}) into {deps_dir} ...")
            _log_dlg("Downloading ~2.7 GB — this can take several minutes.")
            args = [
                "install", "--upgrade", "--only-binary=:all:",
                "--disable-pip-version-check", "--no-input",
                f"--target={deps_dir}",
                "--index-url", f"https://download.pytorch.org/whl/{tag}",
                "--extra-index-url", _PYPI_INDEX,
                "torch==2.7.0", "numpy",
            ]
            def _done(rc):
                if rc == 0:
                    _log_dlg("Done. Restart the app — the GPU will be used automatically.")
                    install_btn.configure(text="✓ Installed", fg_color=_GREEN)
                    if deps_dir not in sys.path:
                        sys.path.insert(0, deps_dir)
                else:
                    _log_dlg(f"pip failed (code {rc}). Check your connection and retry.")
                    install_btn.configure(state="normal", text="Retry")
            _pip_system(args, _done)

        install_btn.configure(command=_do_install)

        def _refresh_gpu():
            name, cap = _detect_gpu()
            _detected_cap["cap"] = cap
            dlg.after(0, _update_cmd)
            if name:
                txt = f"GPU: {name}  (compute {cap})  →  auto: {_resolve_tag()}"
            else:
                txt = ("No NVIDIA GPU detected. CUDA install will still work if you have an "
                       "NVIDIA card; otherwise the app runs on CPU (slow).")
            gpu_lbl.configure(text=txt)
            if _torch_installed():
                install_btn.configure(text="✓ Installed", fg_color=_GREEN, state="disabled")
                title_lbl.configure(text_color=_GREEN)
        threading.Thread(target=_refresh_gpu, daemon=True).start()

        ctk.CTkButton(dlg, text="Close", command=dlg.destroy).pack(pady=6)

    def _on_serialize_toggle(self):
        state = "normal" if self.v_serialize.get() else "disabled"
        self._ser_entry.configure(state=state)

    # ── Pickers ───────────────────────────────────────────────────────────

    def _pick_model(self):
        p = filedialog.askopenfilename(
            title="Select GFISRv2 model",
            filetypes=[("Model files", "*.pth *.safetensors *.pt"), ("All", "*.*")])
        if p:
            self.v_model.set(p)

    def _pick_image(self):
        p = filedialog.askopenfilename(
            title="Select input image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff"), ("All", "*.*")])
        if p:
            self.v_input.set(p)

    def _pick_folder_input(self):
        p = filedialog.askdirectory(title="Select input folder")
        if p:
            self.v_input.set(p)

    def _pick_folder_output(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.v_output.set(p)
            self.v_same_src.set(False)

    def _sync_output(self, *_):
        if not self.v_same_src.get():
            return
        inp = self.v_input.get().strip()
        if not inp:
            return
        base = inp if os.path.isdir(inp) else os.path.dirname(inp)
        out  = os.path.join(base, "upscaled") if self.v_subfolder.get() else base
        self.v_output.set(out)

    # ── Arch dialog ───────────────────────────────────────────────────────

    def _open_arch_dialog(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("GFISRv2 Arch params")
        dlg.geometry("340x220")
        dlg.resizable(False, False)
        dlg.grab_set()
        ctk.CTkLabel(dlg, text="Auto-detected on model load. Override manually if needed.",
                     text_color=_GRAY, font=ctk.CTkFont(size=10), wraplength=310).pack(pady=(10, 6))

        def _row(lbl, var, width=70):
            r = ctk.CTkFrame(dlg, fg_color="transparent")
            r.pack(fill="x", padx=20, pady=3)
            ctk.CTkLabel(r, text=lbl, width=90, anchor="w").pack(side="left")
            ctk.CTkEntry(r, textvariable=var, width=width).pack(side="left")

        _row("dim:", self.v_dim)
        _row("n_blocks:", self.v_n_blocks)
        _row("mid_dim:", self.v_mid_dim)
        r = ctk.CTkFrame(dlg, fg_color="transparent")
        r.pack(fill="x", padx=20, pady=3)
        ctk.CTkLabel(r, text="upsampler:", width=90, anchor="w").pack(side="left")
        ctk.CTkOptionMenu(r, variable=self.v_upsampler,
                          values=["dysample", "pixelshuffledirect", "pixelshuffle",
                                  "nearest+conv", "transpose+conv"],
                          width=175).pack(side="left")
        ctk.CTkButton(dlg, text="Close", command=dlg.destroy).pack(pady=8)

    # ── Auto-detect ───────────────────────────────────────────────────────

    def _autodetect(self):
        mp = self.v_model.get().strip()
        if not mp or not os.path.isfile(mp):
            return
        self._log_msg(f"Auto-detect: {os.path.basename(mp)}")
        def _run():
            import re
            try:
                ckpt = _load_ckpt(mp)

                # Scale from MetaUpsample (only in .pth, not in raw safetensors)
                if "upscale.MetaUpsample" in ckpt:
                    try:
                        meta = ckpt["upscale.MetaUpsample"]
                        s = int(meta[2].item() if hasattr(meta[2], "item") else meta[2])
                        if s in (1, 2, 3, 4, 8):
                            self.v_scale.set(str(s))
                            self._log_msg(f"  scale = {s}x")
                    except Exception:
                        pass

                # dim from in_to_dim conv weight output channels
                for key in ("in_to_dim.weight", "in_to_dim.1.weight"):
                    if key in ckpt:
                        w = ckpt[key]
                        d = int(w.shape[0] if hasattr(w, "shape") else w["shape"][0])
                        self.v_dim.set(d)
                        self._log_msg(f"  dim = {d}")
                        break

                # n_blocks = index of first trailing Conv2d in gfisr_body
                # gfisr_body layout: [GatedCNNBlock x n_blocks] + [Conv2d, SiLU, Conv2d]
                # Direct-child Conv2d keys: "gfisr_body.N.weight" / "gfisr_body.N.bias"
                # GatedCNNBlock keys have deeper nesting: "gfisr_body.N.fc1.weight" etc.
                direct_body = set()
                for k in ckpt:
                    if re.fullmatch(r"gfisr_body\.\d+\.(weight|bias)", k):
                        direct_body.add(int(k.split(".")[1]))
                if direct_body:
                    nb = min(direct_body)  # first trailing Conv2d index = n_blocks
                    self.v_n_blocks.set(nb)
                    self._log_msg(f"  n_blocks = {nb}")

                # upsampler heuristic from upscale keys
                upscale_keys = [k for k in ckpt if k.startswith("upscale.")]
                if any("offset" in k or "mask" in k for k in upscale_keys):
                    self.v_upsampler.set("dysample")
                    self._log_msg("  upsampler = dysample")
                elif any("conv_last" in k for k in upscale_keys):
                    self.v_upsampler.set("pixelshuffle")
                    self._log_msg("  upsampler = pixelshuffle")
                elif upscale_keys:
                    self.v_upsampler.set("pixelshuffledirect")
                    self._log_msg("  upsampler = pixelshuffledirect (guess)")

                self._log_msg("Auto-detect OK.")
            except Exception as e:
                self._log_msg(f"Auto-detect error: {e}")
        threading.Thread(target=_run, daemon=True).start()

    # ── Log / progress helpers ────────────────────────────────────────────

    def _log_msg(self, msg: str):
        def _do():
            self._log.configure(state="normal")
            self._log.insert("end", msg + "\n")
            self._log.see("end")
            self._log.configure(state="disabled")
        self.after(0, _do)

    def _update_preview(self, in_path: str, out_path: str):
        def _do():
            try:
                refs = []
                for widget, fpath in [(self._prev_in, in_path), (self._prev_out, out_path)]:
                    w = max(widget.winfo_width() - 8, 100)
                    h = max(widget.winfo_height() - 24, 60)
                    img = _PILImage.open(fpath).convert("RGB")
                    img.thumbnail((w, h))
                    ci = ctk.CTkImage(light_image=img, dark_image=img,
                                      size=(img.width, img.height))
                    widget.configure(image=ci, text="")
                    refs.append(ci)
                self._prev_refs = refs
            except Exception:
                pass
        self.after(0, _do)

    # ── Queue polling ─────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                item = self._msg_q.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._log_msg(item[1])
                elif kind == "error":
                    self._log_msg(f"[ERR] {item[1]}")
                elif kind == "progress":
                    _, cur, tot, name = item
                    pct = cur / max(tot, 1)
                    self.after(0, lambda p=pct, n=f"{int(pct*100)}%": (
                        self._pbar.set(p), self._lbl_pct.configure(text=n)))
                    elapsed = time.time() - self._t0
                    if cur > 0:
                        eta = elapsed / cur * (tot - cur)
                        self.after(0, lambda e=elapsed, et=eta, c=cur, t=tot, n=name:
                            self._lbl_timing.configure(
                                text=f"Elapsed: {e:.0f}s   ETA: {et:.0f}s   [{c}/{t}]  {n}"))
                    else:
                        self.after(0, lambda n=name:
                            self._lbl_timing.configure(text=f"Loading…  {n}"))
                elif kind == "preview":
                    self._update_preview(item[1], item[2])
                elif kind == "done":
                    elapsed = time.time() - self._t0
                    self._log_msg(f"Done in {elapsed:.1f}s")
                    self.after(0, lambda e=elapsed: self._lbl_timing.configure(
                        text=f"Done in {e:.1f}s"))
                    self.after(0, lambda: (self._pbar.set(1.0),
                                           self._lbl_pct.configure(text="100%")))
                    self._running = False
                    self.after(0, self._on_done)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ── Run / Stop ────────────────────────────────────────────────────────

    def _resolve_tile(self) -> int:
        v = self.v_tile.get()
        if "auto" in v.lower():
            return 256
        if "0" in v and "no" in v.lower():
            return 0
        try:
            return int(v.split()[0])
        except Exception:
            return 256

    def _resolve_tile_pad(self, tile: int) -> int:
        v = self.v_tile_pad.get()
        if "auto" in v.lower():
            return max(tile // 8, 8) if tile > 0 else 32
        try:
            return int(v.split()[0])
        except Exception:
            return 32

    def _resolve_scale(self) -> int:
        v = self.v_scale.get()
        if "auto" in v.lower():
            return 0  # 0 = auto-detect from ckpt
        try:
            return int(v)
        except Exception:
            return 1

    def _collect(self) -> dict:
        tile = self._resolve_tile()
        return {
            "model":         self.v_model.get().strip(),
            "input":         self.v_input.get().strip(),
            "output":        self.v_output.get().strip(),
            "scale":         self._resolve_scale(),
            "tile":          tile,
            "tile_pad":      self._resolve_tile_pad(tile),
            "amp":           self.v_amp.get(),
            "format":        self.v_format.get(),
            "bits":          self.v_bits.get(),
            "quality":       int(self.v_quality.get()),
            "modelname":     self.v_modelname.get(),
            "serialize":     self.v_serialize.get(),
            "serialize_start": int(self.v_ser_start.get() or "0"),
            "dim":           self.v_dim.get(),
            "n_blocks":      self.v_n_blocks.get(),
            "mid_dim":       self.v_mid_dim.get(),
            "upsampler":     self.v_upsampler.get(),
        }

    def _run(self):
        p = self._collect()
        if not p["model"] or not os.path.isfile(p["model"]):
            self._log_msg("ERROR: select a valid model file (.pth/.safetensors).")
            return
        if not p["input"] or not os.path.exists(p["input"]):
            self._log_msg("ERROR: select a valid input image or folder.")
            return
        if not p["output"]:
            self._sync_output()
            p["output"] = self.v_output.get().strip()
        if not p["output"]:
            self._log_msg("ERROR: specify an output folder.")
            return

        # Clear log
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

        self._log_msg(f"Model:  {os.path.basename(p['model'])}")
        self._log_msg(f"Input:  {p['input']}")
        self._log_msg(f"Output: {p['output']}")
        scale_str = "Auto" if p["scale"] == 0 else f"{p['scale']}x"
        tile_str  = f"{p['tile']}" if p["tile"] > 0 else "no tile"
        self._log_msg(f"Scale: {scale_str}  Tile: {tile_str}  Pad: {p['tile_pad']}  AMP: {p['amp']}")

        self._running = True
        self._t0 = time.time()
        self._stop_ev.clear()
        self._btn_run.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._pbar.set(0)
        self._lbl_pct.configure(text="0%")
        self._lbl_timing.configure(text="")

        if _FROZEN:
            if self.v_persistent.get():
                # Frozen + persistent: long-lived thread keeps model in memory
                reuse = (self._pworker_thread and
                         self._pworker_thread.is_alive() and
                         self._pworker_model == p["model"])
                if not reuse:
                    self._stop_pworker()
                    self._pworker_jobq   = queue.Queue()
                    self._pworker_model  = p["model"]
                    # Queue the first job before starting so thread finds it immediately
                    self._pworker_jobq.put(p)
                    self._pworker_thread = threading.Thread(
                        target=_infer_persistent,
                        args=(p, self._pworker_jobq, self._msg_q, self._stop_ev),
                        daemon=True)
                    self._pworker_thread.start()
                else:
                    # Reuse: just send new job — model stays loaded
                    self._pworker_jobq.put(p)
            else:
                threading.Thread(target=_infer_inline, args=(p, self._msg_q, self._stop_ev),
                                 daemon=True).start()
        else:
            threading.Thread(target=self._subprocess_run, args=(p,), daemon=True).start()

    def _subprocess_run(self, p: dict):
        import json
        scale_arg     = "auto" if p["scale"] == 0 else str(p["scale"])
        use_persistent = self.v_persistent.get()

        # Base command — input omitted here, added below for non-persistent
        base_cmd = [
            sys.executable, _WORKER,
            "--model",    p["model"],
            "--output",   p["output"],
            "--scale",    scale_arg,
            "--tile",     str(p["tile"]),
            "--tile_pad", str(p["tile_pad"]),
            "--amp",      "1" if p["amp"] else "0",
            "--modelname","1" if p["modelname"] else "0",
            "--serialize","1" if p["serialize"] else "0",
            "--ser_start",str(p["serialize_start"]),
            "--dim",      str(p["dim"]),
            "--n_blocks", str(p["n_blocks"]),
            "--mid_dim",  str(p["mid_dim"]),
            "--upsampler",p["upsampler"],
            "--format",   p["format"],
            "--bits",     "16" if "16" in p.get("bits", "8") else "8",
            "--quality",  str(p["quality"]),
        ]

        # ── Persistent mode ──────────────────────────────────────────────────
        if use_persistent:
            reuse = (self._persistent_proc and
                     self._persistent_proc.poll() is None and
                     self._persistent_model == p["model"])

            if not reuse:
                # Kill any old persistent proc then start fresh
                self._kill_persistent()
                cmd = base_cmd + ["--persistent"]
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace")
                except Exception as e:
                    self._msg_q.put(("error", f"Popen failed: {e}"))
                    self._msg_q.put(("done", None))
                    return
                self._persistent_proc  = proc
                self._persistent_model = p["model"]

                # Read startup lines until READY
                for raw in proc.stdout:
                    line = raw.rstrip()
                    if line == "READY":
                        self._msg_q.put(("log", "Persistent worker ready."))
                        break
                    self._parse_worker_line(line)
                else:
                    # stdout closed before READY → startup failed
                    self._msg_q.put(("error", "Persistent worker failed to start (READY not received)."))
                    self._msg_q.put(("done", None))
                    self._persistent_proc = None
                    return

            proc = self._persistent_proc
            req  = json.dumps({"input": p["input"], "output": p["output"]}) + "\n"
            try:
                proc.stdin.write(req)
                proc.stdin.flush()
            except Exception as e:
                self._msg_q.put(("error", f"stdin write failed: {e}"))
                self._msg_q.put(("done", None))
                self._persistent_proc = None
                return

            # Read worker output until DONE (keep proc alive after)
            try:
                for raw in proc.stdout:
                    if self._parse_worker_line(raw.rstrip()):
                        return  # DONE — proc stays alive for next Run
            except Exception as e:
                self._msg_q.put(("error", str(e)))
                self._msg_q.put(("done", None))
                self._persistent_proc = None
            return

        # ── Normal (non-persistent) mode ─────────────────────────────────────
        cmd = base_cmd + ["--input", p["input"]]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace")
            for raw in self._proc.stdout:
                self._parse_worker_line(raw.rstrip())
            self._proc.wait()
            rc = self._proc.returncode
            if rc != 0 and self._running:
                self._msg_q.put(("error", f"Worker exit {rc}"))
                self._msg_q.put(("done", None))
        except Exception as e:
            self._msg_q.put(("error", str(e)))
            self._msg_q.put(("done", None))
        finally:
            self._proc    = None
            self._running = False

    def _on_done(self):
        self._btn_run.configure(state="normal")
        self._btn_stop.configure(state="disabled")

    def _stop(self):
        self._stop_ev.set()
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._kill_persistent()  # subprocess persistant
        self._stop_pworker()     # inline persistant (frozen)
        self._log_msg("Stop.")
        self.after(300, self._on_done)

    def _on_close(self):
        self._save_settings()
        self._kill_persistent()
        self._stop_pworker()
        self._stop()
        self.destroy()

    def _load_settings(self):
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            return
        def _sv(var, key):
            if key in s:
                try: var.set(s[key])
                except Exception: pass
        _sv(self.v_model,     "model")
        _sv(self.v_input,     "input")
        _sv(self.v_output,    "output")
        _sv(self.v_format,    "format")
        _sv(self.v_bits,      "bits")
        _sv(self.v_quality,   "quality")
        _sv(self.v_scale,     "scale")
        _sv(self.v_tile,      "tile")
        _sv(self.v_tile_pad,  "tile_pad")
        _sv(self.v_amp,       "amp")
        _sv(self.v_persistent,"persistent")
        _sv(self.v_same_src,  "same_src")
        _sv(self.v_subfolder, "subfolder")
        _sv(self.v_modelname, "modelname")
        _sv(self.v_serialize, "serialize")
        _sv(self.v_ser_start, "ser_start")
        _sv(self.v_dim,       "dim")
        _sv(self.v_n_blocks,  "n_blocks")
        _sv(self.v_mid_dim,   "mid_dim")
        _sv(self.v_upsampler, "upsampler")
        self._saved_geometry = s.get("geometry")

    def _save_settings(self):
        try:
            s = {
                "model":      self.v_model.get(),
                "input":      self.v_input.get(),
                "output":     self.v_output.get(),
                "format":     self.v_format.get(),
                "bits":       self.v_bits.get(),
                "quality":    self.v_quality.get(),
                "scale":      self.v_scale.get(),
                "tile":       self.v_tile.get(),
                "tile_pad":   self.v_tile_pad.get(),
                "amp":        self.v_amp.get(),
                "persistent": self.v_persistent.get(),
                "same_src":   self.v_same_src.get(),
                "subfolder":  self.v_subfolder.get(),
                "modelname":  self.v_modelname.get(),
                "serialize":  self.v_serialize.get(),
                "ser_start":  self.v_ser_start.get(),
                "dim":        self.v_dim.get(),
                "n_blocks":   self.v_n_blocks.get(),
                "mid_dim":    self.v_mid_dim.get(),
                "upsampler":  self.v_upsampler.get(),
                "geometry":   self.geometry(),
            }
            with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
        except Exception:
            pass


# ── Widget helpers ─────────────────────────────────────────────────────────

def _spacer(parent):
    f = ctk.CTkFrame(parent, fg_color="transparent", width=1, height=1)
    f.pack_propagate(False)
    f.pack(side="left", fill="x", expand=True)

def _vsep(parent):
    ctk.CTkFrame(parent, width=1, fg_color="gray40").pack(
        side="left", fill="y", padx=(8, 8), pady=3)


class _Tooltip:
    """Lightweight hover tooltip for any widget (English help text)."""
    def __init__(self, widget, text, delay=450):
        self.widget, self.text, self.delay = widget, text, delay
        self._tip = None
        self._after = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _=None):
        self._cancel()
        try:
            self._after = self.widget.after(self.delay, self._show)
        except Exception:
            pass

    def _show(self):
        if self._tip is not None:
            return
        try:
            x = self.widget.winfo_rootx() + 14
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except Exception:
            return
        self._tip = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass
        lbl = ctk.CTkLabel(tw, text=self.text, fg_color=("gray92", "gray16"),
                           text_color=("gray10", "gray90"), corner_radius=6,
                           wraplength=340, justify="left",
                           font=ctk.CTkFont(size=11))
        lbl.pack(padx=8, pady=5)

    def _hide(self, _=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None

    def _cancel(self):
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None


def _tip(widget, text):
    """Attach a tooltip and return the widget (for inline use)."""
    _Tooltip(widget, text)
    return widget


# ── Entry point ────────────────────────────────────────────────────────────

def _run_diag():
    """Headless self-test of the torch/CUDA stack. Enable with env GFISRV2_DIAG=1.
    Writes result to _diag.log next to the exe. Kept for field debugging."""
    import traceback as _tb
    out = []
    def p(*a): out.append(" ".join(str(x) for x in a))
    def _chain(e):
        parts, cur = [], e
        while cur:
            parts.append("".join(_tb.format_exception(type(cur), cur, cur.__traceback__)))
            cur = cur.__cause__ or cur.__context__
        return "\n--- caused by ---\n".join(parts)

    def _w():
        try:
            import torch
            p("torch OK:", torch.__version__, "_C:", hasattr(torch, "_C"),
              "cuda:", torch.cuda.is_available())
            if torch.cuda.is_available():
                p("  GPU:", torch.cuda.get_device_name(0), torch.cuda.get_arch_list())
        except Exception as e:
            p("torch FAILED:"); p(_chain(e))
    th = threading.Thread(target=_w); th.start(); th.join()
    with open(os.path.join(_EXE_DIR, "_diag.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main():
    if os.environ.get("GFISRV2_DIAG"):
        _run_diag()
        return
    app = GFISRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
