"""
quick_upscale.py — Inférence rapide universelle (NeoSR / Redux).
Supporte le tiling pour économiser la VRAM et la détection automatique
de l'architecture depuis les poids du modèle.
"""
import os
import sys
import math
import time
import threading
import traceback
from typing import Optional, Tuple, Callable

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    from PIL import Image
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False


# ─── Architecture detection from state_dict ─────────────────────

# Key signatures to identify architecture from weight keys
ARCH_SIGNATURES = {
    "spanplus": ["feats.0.eval_conv", "upsampler.offset"],  # SPANPlus (traiNNer-redux, DySample upsampler)
    "span": ["feats.0.block", "feats.0.sk"],
    "compact": ["body.0.weight", "body.2.weight"],
    "omnisr": ["residual_layer", "osag"],
    "realplksr": ["feats.0.lk.conv", "dysample"],
    "esrgan": ["body.0.rdb1", "RRDB_trunk"],
    "swinir": ["layers.0.residual_group.blocks", "conv_after_body"],
    "hat": ["layers.0.residual_group.blocks", "conv_before_upsample", "relative_coords_table"],
    "dat": ["layers.0.blocks", "before_RG"],
    "drct": ["layers.0.swin1", "layers.0.adjust1"],
    "rcan": ["body.0.body.0.body.0.weight"],
    "cugan": ["unet1.conv1.conv.0.weight"],
    "esc": ["plk_filter"],
    "ninasr": ["head.0.bias", "body.0.body.2.body"],
    "lmlt":   ["feats.0.lhsb", "to_img.0.weight"],
    "eimn":   ["block1.0.attn.region", "block1.0.layer_scale_1"],
}


def detect_arch_from_state(state_dict: dict) -> Optional[str]:
    """Attempt to detect architecture from state_dict key patterns."""
    keys_str = " ".join(state_dict.keys())
    for arch, signatures in ARCH_SIGNATURES.items():
        if all(sig in keys_str for sig in signatures):
            return arch
    return None


def extract_weights(checkpoint: dict) -> dict:
    """Extract generator weights from various checkpoint formats."""
    for key in ("params_ema", "params_g", "params", "model", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    # If top-level keys look like weight tensors, return as-is
    if any(k.endswith(".weight") for k in checkpoint.keys()):
        return checkpoint
    return checkpoint


def detect_scale_from_state(state_dict: dict) -> int:
    """Try to infer scale factor from upsampling layers."""
    for key in state_dict:
        if "upsample" in key.lower() or "upconv" in key.lower() or "pixel_shuffle" in key.lower():
            tensor = state_dict[key]
            if hasattr(tensor, "shape") and len(tensor.shape) == 4:
                out_ch = tensor.shape[0]
                in_ch = tensor.shape[1]
                if out_ch > in_ch:
                    ratio = out_ch / in_ch
                    # pixel_shuffle: out_channels = in_channels * scale^2
                    scale = int(math.sqrt(ratio))
                    if scale in (2, 3, 4, 8):
                        return scale
    # DySample upsampler (SPANPlus): offset channels = 4 * 2 * scale^2 = 8 * scale^2
    # init_pos / offset.weight both use the same channel count
    if "upsampler.offset.weight" in state_dict:
        offset_ch = state_dict["upsampler.offset.weight"].shape[0]
        scale_sq = offset_ch / 8.0
        s = int(math.sqrt(scale_sq))
        if s >= 1 and s * s == int(scale_sq):
            return s
    # ESC DySample upsampler: to_img.offset.weight channels = groups(4) * 2 * scale^2 = 8 * scale^2
    if "to_img.offset.weight" in state_dict:
        offset_ch = state_dict["to_img.offset.weight"].shape[0]
        scale_sq = offset_ch / 8.0
        s = int(math.sqrt(scale_sq))
        if s >= 1 and s * s == int(scale_sq):
            return s
    # conv upsampler (SPANPlus 1x without DySample): always 1x
    if "upsampler.weight" in state_dict and "upsampler.bias" in state_dict:
        if "upsampler.offset" not in " ".join(state_dict.keys()):
            return 1
    # ninasr tail: tail.1 is Conv2d with out_ch = 3 * scale^2
    # eimn tail:   tail.0 is Conv2d with out_ch = 3 * scale^2
    # lmlt upsample: to_img.0 is Conv2d with out_ch = 3 * scale^2
    for probe_key in ("tail.1.weight", "tail.0.weight", "to_img.0.weight"):
        if probe_key in state_dict:
            ch = state_dict[probe_key].shape[0]
            s = int(round(math.sqrt(ch / 3)))
            if s >= 1 and s * s * 3 == ch:
                return s
    return 4  # Default


# ─── Tiled inference ─────────────────────────────────────────────

def tile_inference(model, img_tensor: "torch.Tensor", tile_size: int = 256,
                   tile_pad: int = 32, scale: int = 4,
                   stop_event=None) -> "torch.Tensor":
    """
    Process an image in overlapping tiles to save VRAM.
    Checks stop_event between tiles — raises RuntimeError("stopped") if set.

    Args:
        model: PyTorch model (callable)
        img_tensor: Input tensor [1, C, H, W]
        tile_size: Size of each tile (pixels)
        tile_pad: Overlap padding
        scale: Upscale factor
        stop_event: threading.Event — checked between tiles

    Returns:
        Output tensor [1, C, H*scale, W*scale]
    """
    batch, channel, height, width = img_tensor.shape
    out_h = height * scale
    out_w = width * scale
    output = img_tensor.new_zeros((batch, channel, out_h, out_w))
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            # Check stop between tiles — immediate abort
            if stop_event and stop_event.is_set():
                raise RuntimeError("stopped")
            # Input tile bounds (with padding)
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            input_start_x = max(ofs_x - tile_pad, 0)
            input_end_x = min(ofs_x + tile_size + tile_pad, width)
            input_start_y = max(ofs_y - tile_pad, 0)
            input_end_y = min(ofs_y + tile_size + tile_pad, height)

            # Input tile
            input_tile = img_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

            # Process tile
            with torch.no_grad():
                output_tile = model(input_tile)

            # Output tile bounds
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # Remove padding from output
            output_start_x_tile = (ofs_x - input_start_x) * scale
            output_end_x_tile = output_start_x_tile + tile_size * scale
            output_start_y_tile = (ofs_y - input_start_y) * scale
            output_end_y_tile = output_start_y_tile + tile_size * scale

            # Clamp to actual tile size
            output_end_x_tile = min(output_end_x_tile, output_tile.shape[3])
            output_end_y_tile = min(output_end_y_tile, output_tile.shape[2])

            # Target region in output
            target_end_x = min(ofs_x * scale + tile_size * scale, out_w)
            target_end_y = min(ofs_y * scale + tile_size * scale, out_h)
            actual_w = target_end_x - ofs_x * scale
            actual_h = target_end_y - ofs_y * scale

            output[:, :, ofs_y * scale:target_end_y, ofs_x * scale:target_end_x] = \
                output_tile[:, :, output_start_y_tile:output_start_y_tile + actual_h,
                            output_start_x_tile:output_start_x_tile + actual_w]

    return output


# ─── Image save helper ───────────────────────────────────────────

# Extension mapping for output format
_FMT_EXT = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "WEBP": ".webp",
    "TIFF": ".tif",
    "BMP": ".bmp",
}


def _save_output(
    float_arr: "np.ndarray",   # [H, W, 3] float32 in [0, 1]
    output_path: str,
    out_format: str = "PNG",
    bit_depth: int = 8,
    quality: int = 95,
) -> None:
    """Save an upscaled float32 [H,W,3] array to disk with format and bit-depth control."""
    fmt = out_format.upper()
    save_kwargs: dict = {}

    if bit_depth == 16 and fmt in ("PNG", "TIFF"):
        arr16 = (float_arr * 65535.0).round().clip(0, 65535).astype(np.uint16)
        h, w = arr16.shape[:2]
        # Big-endian 16-bit RGB (required by PNG 16-bit spec)
        arr_be = arr16.byteswap().newbyteorder()
        img = Image.frombuffer("RGB", (w, h), bytes(arr_be.data), "raw", "RGB;16B", 0, 1)
        if fmt == "TIFF":
            save_kwargs["compression"] = "lzw"
    else:
        arr8 = (float_arr * 255.0).round().clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr8)
        if fmt in ("JPEG", "JPG"):
            save_kwargs["quality"] = quality
            save_kwargs["subsampling"] = 0
        elif fmt == "WEBP":
            save_kwargs["quality"] = quality
        elif fmt == "TIFF":
            save_kwargs["compression"] = "lzw"

    pil_fmt = "JPEG" if fmt in ("JPEG", "JPG") else fmt
    img.save(output_path, format=pil_fmt, **save_kwargs)


# ─── ONNX inference ──────────────────────────────────────────────

def _onnx_upscale(
    model_path: str,
    input_path: str,
    output_path: str,
    log: Callable,
    progress_callback: Optional[Callable] = None,
    out_format: str = "PNG",
    bit_depth: int = 8,
    quality: int = 95,
) -> "Tuple[bool, str]":
    """
    Run super-resolution inference via ONNX Runtime.
    Supports any SR model exported to ONNX with input [1, 3, H, W]
    and output [1, 3, H*scale, W*scale].
    Requires: pip install onnxruntime-gpu  (or onnxruntime for CPU-only)
    """
    def _prog(v):
        if progress_callback:
            progress_callback(v)

    try:
        import onnxruntime as ort
    except ImportError:
        return False, (
            "onnxruntime non installé.\n"
            "Installez avec : pip install onnxruntime-gpu\n"
            "(ou onnxruntime pour CPU uniquement)"
        )

    if not IMAGING_AVAILABLE:
        return False, "Pillow/NumPy non installé"

    log("ONNX: chargement du modele...")
    _prog(0.08)

    providers = []
    try:
        import torch
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
    except ImportError:
        pass
    providers.append("CPUExecutionProvider")

    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        return False, f"Impossible de charger le modele ONNX : {e}"

    active = sess.get_providers()
    log(f"ONNX: providers actifs = {active}")
    _prog(0.25)

    # Load image
    log(f"Lecture : {os.path.basename(input_path)}")
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = np.transpose(img_np, (2, 0, 1))[np.newaxis]  # [1, 3, H, W]

    log(f"ONNX: inference {img.width}x{img.height}...")
    _prog(0.50)

    inp_name = sess.get_inputs()[0].name
    try:
        out_np = sess.run(None, {inp_name: img_t})[0]  # [1, 3, H*s, W*s]
    except Exception as e:
        return False, f"Erreur inference ONNX : {e}"

    _prog(0.90)

    # Convert and save
    out_np = out_np.squeeze(0).clip(0, 1)         # [3, H, W]
    out_np = np.transpose(out_np, (1, 2, 0))       # [H, W, 3] float32 [0,1]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    _save_output(out_np, output_path, out_format, bit_depth, quality)

    _prog(1.0)
    log(f"ONNX OK : {output_path}")
    return True, f"ONNX upscale réussi : {os.path.basename(output_path)}"


# ─── Subprocess helper : non-blocking reader + immediate kill ─────

def _run_proc_with_stop(proc, progress_map, log, progress_callback, stop_event):
    """
    Drive a subprocess non-blockingly so stop_event is honoured immediately,
    even when the subprocess is silent during heavy GPU computation.

    Strategy:
      - stdout is drained by a daemon reader thread (never blocks the main thread).
      - Main thread polls proc.poll() every 50 ms and checks stop_event.
      - On stop: proc.kill() is called instantly, function returns False.

    Returns:
      (killed: bool)  — True if process was killed by stop_event.
    """
    def _reader():
        try:
            for line in proc.stdout:
                stripped = line.rstrip()
                log(stripped)
                if progress_callback and progress_map:
                    for prefix, pct in progress_map:
                        if prefix in stripped:
                            progress_callback(pct)
                            break
        except Exception:
            pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    # Poll until the process exits or stop is requested
    while proc.poll() is None:
        if stop_event and stop_event.is_set():
            try:
                proc.kill()
            except Exception:
                pass
            t.join(timeout=1.0)
            return True   # killed
        time.sleep(0.05)

    t.join(timeout=2.0)
    return False  # natural exit


# ─── Universal subprocess fallback (spandrel + SpanPlus) ─────────

_TRAINNER_VENV_PY = os.path.join(
    os.path.expanduser("~"), "IA_Engine", "traiNNer-redux", ".venv", "Scripts", "python.exe"
)
_NEOSR_VENV_PY = os.path.join(
    os.path.expanduser("~"), "IA_Engine", "neosr", ".venv", "Scripts", "python.exe"
)
# Keep old spanplus_runner for compatibility; prefer universal_runner
_UNIVERSAL_RUNNER = os.path.join(os.path.dirname(__file__), "universal_runner.py")
_SPANPLUS_RUNNER  = os.path.join(os.path.dirname(__file__), "spanplus_runner.py")
_NEOSR_RUNNER         = os.path.join(os.path.dirname(__file__), "neosr_runner.py")
_NEOSR_GENERAL_RUNNER = os.path.join(os.path.dirname(__file__), "neosr_general_runner.py")

# Archs qui appellent net_opt() au niveau module → subprocess neosr_general_runner obligatoire
_NEOSR_GENERAL_ARCHS = {"ninasr", "lmlt", "eimn", "drct"}

# ─── Model cache ─────────────────────────────────────────────────
# Clé : chemin absolu canonique du modèle.
# Valeur : {"model": obj|None, "scale": int, "arch": str, "device": str,
#            "is_subprocess": bool, "subprocess_type": str}
# Évite de recharger le modèle à chaque image lors d'un batch.
_MODEL_CACHE: dict = {}


def clear_model_cache() -> int:
    """
    Libère tous les modèles en cache et vide la VRAM.
    Retourne le nombre d'entrées supprimées.
    Appeler depuis l'UI après un batch pour récupérer la VRAM.
    """
    global _MODEL_CACHE
    n = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return n


def _spanplus_subprocess_infer(
    model_path: str,
    input_path: str,
    output_path: str,
    log: Callable,
    progress_callback: Optional[Callable] = None,
    stop_event=None,
    tile_size: int = 0,
    tile_pad: int = 32,
    use_amp: bool = False,
) -> "Tuple[bool, str]":
    """
    Run inference in the traiNNer-redux venv subprocess.
    Uses universal_runner (spandrel-first) when available,
    falls back to spanplus_runner for legacy compatibility.
    progress_callback(float 0.0-1.0) is called as runner lines arrive.
    """
    import subprocess

    venv_py = _TRAINNER_VENV_PY
    # Prefer universal runner (handles SPAN, ESRGAN, etc. via spandrel)
    runner = _UNIVERSAL_RUNNER if os.path.isfile(_UNIVERSAL_RUNNER) else _SPANPLUS_RUNNER

    if not os.path.isfile(venv_py):
        return False, (
            f"traiNNer-redux venv Python introuvable :\n{venv_py}\n"
            f"Vérifiez que traiNNer-redux est installé dans ~/IA_Engine/traiNNer-redux."
        )
    if not os.path.isfile(runner):
        return False, f"Runner SPANPlus introuvable : {runner}"

    # Map known runner output prefixes to approximate progress values
    # Covers both universal_runner and spanplus_runner output lines
    _PROGRESS_MAP = [
        ("[Runner] Device",      0.06),
        ("[Runner] Chargement",  0.20),
        ("[Runner] spandrel OK", 0.45),
        ("[Runner] SpanPlus",    0.45),
        ("[Runner] Modele",      0.60),
        ("[Runner] Inference",   0.70),
        ("[Runner] Sauvegarde",  0.92),
    ]

    # Pass tile/amp params as argv so universal_runner can use them
    cmd = [venv_py, runner, model_path, input_path, output_path,
           str(tile_size), str(tile_pad), "1" if use_amp else "0"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        killed = _run_proc_with_stop(proc, _PROGRESS_MAP, log, progress_callback, stop_event)
        proc.stdout.close()
        if killed:
            return False, "Arrêt demandé par l'utilisateur."
    except Exception as e:
        return False, f"Erreur lancement subprocess SPANPlus : {e}"

    if proc.returncode == 0:
        if progress_callback:
            progress_callback(1.0)
        return True, f"SPANPlus (subprocess) : {os.path.basename(output_path)}"
    return False, f"Subprocess SPANPlus a échoué (code {proc.returncode})"


# ─── NeoSR subprocess (ESC arch) ─────────────────────────────────

def _neosr_subprocess_infer(
    model_path: str,
    input_path: str,
    output_path: str,
    log: Callable,
    progress_callback: Optional[Callable] = None,
    stop_event=None,
) -> "Tuple[bool, str]":
    """
    Run ESC inference in the neosr venv subprocess.
    neosr_runner.py patches net_opt before importing esc_arch
    so the module-level net_opt() call does not require sys.argv -opt.
    """
    import subprocess

    venv_py = _NEOSR_VENV_PY
    runner  = _NEOSR_RUNNER

    if not os.path.isfile(venv_py):
        return False, (
            f"neosr venv Python introuvable :\n{venv_py}\n"
            f"Vérifiez que neosr est installé dans ~/IA_Engine/neosr."
        )
    if not os.path.isfile(runner):
        return False, f"Runner neosr introuvable : {runner}"

    _PROGRESS_MAP = [
        ("[NeoSR] Device",    0.06),
        ("[NeoSR] ESC params", 0.20),
        ("[NeoSR] ESC chargé", 0.50),
        ("[NeoSR] Inférence",  0.70),
        ("[NeoSR] Sauvegarde", 0.92),
    ]

    cmd = [venv_py, runner, model_path, input_path, output_path]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        killed = _run_proc_with_stop(proc, _PROGRESS_MAP, log, progress_callback, stop_event)
        proc.stdout.close()
        if killed:
            return False, "Arrêt demandé par l'utilisateur."
    except Exception as e:
        return False, f"Erreur lancement subprocess neosr : {e}"

    if proc.returncode == 0:
        if progress_callback:
            progress_callback(1.0)
        return True, f"ESC (neosr subprocess) : {os.path.basename(output_path)}"
    return False, f"Subprocess neosr ESC a échoué (code {proc.returncode})"


# ─── NeoSR general subprocess (ninasr, lmlt, eimn, drct, …) ─────

def _neosr_general_subprocess_infer(
    model_path: str,
    input_path: str,
    output_path: str,
    log: Callable,
    progress_callback: Optional[Callable] = None,
    stop_event=None,
) -> "Tuple[bool, str]":
    """
    Run inference via neosr_general_runner.py in the neosr venv subprocess.
    Handles net_opt() patching and arch detection internally.
    Used for archs in _NEOSR_GENERAL_ARCHS (ninasr, lmlt, eimn, drct).
    """
    import subprocess

    venv_py = _NEOSR_VENV_PY
    runner  = _NEOSR_GENERAL_RUNNER

    if not os.path.isfile(venv_py):
        return False, (
            f"neosr venv Python introuvable :\n{venv_py}\n"
            f"Vérifiez que neosr est installé dans ~/IA_Engine/neosr."
        )
    if not os.path.isfile(runner):
        return False, f"neosr_general_runner introuvable : {runner}"

    _PROGRESS_MAP = [
        ("[NeoSR] Device",     0.06),
        ("[NeoSR] Arch",       0.20),
        ("[NeoSR] Modele",     0.45),
        ("[NeoSR] Inference",  0.70),
        ("[NeoSR] Sauvegarde", 0.92),
    ]

    cmd = [venv_py, runner, model_path, input_path, output_path]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        killed = _run_proc_with_stop(proc, _PROGRESS_MAP, log, progress_callback, stop_event)
        proc.stdout.close()
        if killed:
            return False, "Arrêt demandé par l'utilisateur."
    except Exception as e:
        return False, f"Erreur lancement subprocess neosr general : {e}"

    if proc.returncode == 0:
        if progress_callback:
            progress_callback(1.0)
        return True, f"neosr general ({os.path.basename(output_path)})"
    return False, f"Subprocess neosr general a echoue (code {proc.returncode})"


# ─── Main upscale function ───────────────────────────────────────

def upscale_image(
    model_path: str,
    input_path: str,
    output_path: str,
    scale: int = 0,
    tile_size: int = 256,
    tile_pad: int = 32,
    use_amp: bool = True,
    callback: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
    out_format: str = "PNG",
    bit_depth: int = 8,
    quality: int = 95,
    stop_event=None,
) -> Tuple[bool, str]:
    """
    Upscale a single image using a trained SR model.
    
    Args:
        model_path: Path to .pth or .safetensors model
        input_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscale factor (0 = auto-detect)
        tile_size: Tile size for inference (0 = no tiling)
        tile_pad: Tile overlap padding
        use_amp: Use mixed precision (FP16)
        callback: Optional progress callback(message: str)
        
    Returns:
        (success: bool, message: str)
    """
    if not os.path.isfile(model_path):
        return False, f"Modèle introuvable : {model_path}"
    if not os.path.isfile(input_path):
        return False, f"Image introuvable : {input_path}"

    def log(msg):
        if callback:
            callback(msg)
        try:
            print(f"[Upscale] {msg}")
        except OSError:
            pass

    # ── ONNX short-circuit (no PyTorch needed) ──
    if model_path.lower().endswith(".onnx"):
        return _onnx_upscale(model_path, input_path, output_path, log, progress_callback,
                             out_format=out_format, bit_depth=bit_depth, quality=quality)

    if not TORCH_AVAILABLE:
        return False, "PyTorch non installé"
    if not IMAGING_AVAILABLE:
        return False, "Pillow/NumPy non installé"

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Device : {device}")

        # ── Cache check ──
        _cache_key   = os.path.realpath(model_path)
        _cached      = _MODEL_CACHE.get(_cache_key)
        _cache_device = str(device)

        if _cached is not None and _cached.get("device") == _cache_device:
            # --- Cache hit ---
            if _cached.get("is_subprocess"):
                detected_arch = _cached["arch"]
                if scale == 0:
                    scale = _cached["scale"]
                stype = _cached.get("subprocess_type", "trainner")
                log(f"[Cache] {detected_arch} (subprocess {stype}, scale {scale}x)")
                if stype == "esc":
                    return _neosr_subprocess_infer(
                        model_path, input_path, output_path, log,
                        progress_callback, stop_event=stop_event)
                elif stype == "neosr_general":
                    return _neosr_general_subprocess_infer(
                        model_path, input_path, output_path, log,
                        progress_callback, stop_event=stop_event)
                else:
                    return _spanplus_subprocess_infer(
                        model_path, input_path, output_path, log,
                        progress_callback, stop_event=stop_event,
                        tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp)
            model         = _cached["model"]
            detected_arch = _cached["arch"]
            if scale == 0:
                scale = _cached["scale"]
            log(f"[Cache] Modèle réutilisé : {detected_arch}, scale {scale}x")
        else:
            # --- Cache miss : chargement complet ---
            log(f"Chargement du modèle : {os.path.basename(model_path)}")

            if model_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path, device="cpu")
                except ImportError:
                    return False, "Module safetensors requis : pip install safetensors"
            else:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                state_dict = extract_weights(checkpoint)

            # ── Detect architecture ──
            detected_arch = detect_arch_from_state(state_dict)
            if scale == 0:
                scale = detect_scale_from_state(state_dict)
            log(f"Architecture détectée : {detected_arch or 'inconnue'}, scale : {scale}x")

            def _store_subprocess(stype: str) -> None:
                _MODEL_CACHE[_cache_key] = {
                    "model": None, "scale": scale,
                    "arch": detected_arch or stype,
                    "device": _cache_device,
                    "is_subprocess": True,
                    "subprocess_type": stype,
                }

            # ESC : net_opt() au niveau module, subprocess neosr venv obligatoire
            if detected_arch == "esc":
                log("ESC détecté → subprocess neosr venv...")
                _store_subprocess("esc")
                return _neosr_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event
                )

            # ninasr/lmlt/eimn/drct : net_opt() au niveau module → neosr_general_runner
            if detected_arch in _NEOSR_GENERAL_ARCHS:
                log(f"{detected_arch} → subprocess neosr général...")
                _store_subprocess("neosr_general")
                return _neosr_general_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event
                )

            # ── Try to instantiate model in-process ──
            model = None

            # Try neosr imports
            if detected_arch and model is None:
                try:
                    if detected_arch == "span":
                        from neosr.archs.span_arch import span
                        model = span(scale=scale)
                    elif detected_arch == "omnisr":
                        from neosr.archs.omnisr_arch import omnisr
                        model = omnisr(scale=scale)
                    elif detected_arch == "realplksr":
                        from neosr.archs.realplksr_arch import realplksr
                        model = realplksr(scale=scale)
                    elif detected_arch == "compact":
                        from neosr.archs.srvgg_arch import srvgg
                        model = srvgg(scale=scale)
                    elif detected_arch == "esrgan":
                        from neosr.archs.rrdbnet_arch import rrdbnet
                        model = rrdbnet(scale=scale)
                except ImportError:
                    pass

            # Try traiNNer-redux imports
            if detected_arch and model is None:
                try:
                    redux_path = os.path.join(os.path.expanduser("~"), "IA_Engine", "traiNNer-redux")
                    if os.path.isdir(redux_path) and redux_path not in sys.path:
                        sys.path.insert(0, redux_path)
                    if detected_arch == "spanplus":
                        from traiNNer.archs.spanplus_arch import SpanPlus
                        fc = state_dict.get("feats.0.eval_conv.weight",
                                            state_dict.get("feats.0.sk.weight", None))
                        fc = fc.shape[0] if fc is not None else 48
                        up = "dys" if "upsampler.offset.weight" in state_dict else "conv"
                        model = SpanPlus(feature_channels=fc, upscale=scale, upsampler=up)
                        log(f"SPANPlus instancié (fc={fc}, scale={scale}x, upsampler={up})")
                    elif detected_arch == "span":
                        from traiNNer.archs.span_arch import SPAN
                        model = SPAN(scale=scale)
                    elif detected_arch == "omnisr":
                        from traiNNer.archs.omnisr_arch import OmniSR
                        model = OmniSR(upscale=scale)
                    elif detected_arch == "compact":
                        from traiNNer.archs.srvgg_arch import SRVGGNetCompact
                        model = SRVGGNetCompact(upscale=scale)
                except ImportError:
                    pass

            if model is None:
                # Fallback 1: TorchScript
                try:
                    model = torch.jit.load(model_path, map_location=device)
                    log("Modèle chargé comme TorchScript")
                except Exception:
                    pass

            if model is None:
                # Fallback 2: universal subprocess (spandrel in traiNNer venv)
                _store_subprocess("trainner")
                log(f"Tentative via subprocess traiNNer-redux venv ({detected_arch or 'arch inconnue'})...")
                return _spanplus_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event,
                    tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp
                )

            if not isinstance(model, torch.jit.ScriptModule):
                model.load_state_dict(state_dict, strict=False)

            model = model.to(device).eval()
            log("Modèle chargé et prêt")

            # Mise en cache pour les appels suivants
            _MODEL_CACHE[_cache_key] = {
                "model": model, "scale": scale,
                "arch": detected_arch or "unknown",
                "device": _cache_device,
                "is_subprocess": False,
            }

        # ── Load image ──
        log(f"Lecture : {os.path.basename(input_path)}")
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

        # ── Inference ──
        log(f"Inférence ({img.width}x{img.height} → {img.width*scale}x{img.height*scale})")
        
        amp_ctx = torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda"))
        
        # Check stop before starting heavy inference
        if stop_event and stop_event.is_set():
            return False, "Arrêt demandé par l'utilisateur."

        with torch.no_grad(), amp_ctx:
            if tile_size > 0 and (img.width > tile_size or img.height > tile_size):
                log(f"Mode tiling : {tile_size}px, padding {tile_pad}px")
                output = tile_inference(model, img_tensor, tile_size, tile_pad, scale,
                                        stop_event=stop_event)
            else:
                output = model(img_tensor)

        # ── Save output ──
        output = output.squeeze(0).clamp(0, 1).cpu().float().numpy()
        output = np.transpose(output, (1, 2, 0))   # [H, W, 3] float32 [0,1]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        _save_output(output, output_path, out_format, bit_depth, quality)

        log(f"✅ Sauvegardé : {output_path}")
        return True, f"Upscale réussi : {os.path.basename(output_path)}"

    except RuntimeError as e:
        if "stopped" in str(e):
            return False, "Arrêt demandé par l'utilisateur."
        return False, f"Erreur runtime : {e}\n{traceback.format_exc()}"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if tile_size > 0:
            return False, (
                f"VRAM insuffisante même avec tiling ({tile_size}px).\n"
                f"Réduisez tile_size ou utilisez le CPU."
            )
        return False, (
            "VRAM insuffisante. Activez le tiling (tile_size=256) ou réduisez la taille."
        )
    except Exception as e:
        return False, f"Erreur : {e}\n{traceback.format_exc()}"


def upscale_folder(
    model_path: str,
    input_folder: str,
    output_folder: str,
    scale: int = 0,
    tile_size: int = 256,
    tile_pad: int = 32,
    use_amp: bool = True,
    callback: Optional[Callable] = None,
) -> Tuple[int, int, list]:
    """
    Upscale all images in a folder.
    
    Returns:
        (success_count, total_count, errors: list[str])
    """
    if not os.path.isdir(input_folder):
        return 0, 0, [f"Dossier introuvable : {input_folder}"]

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    files = [f for f in os.listdir(input_folder)
             if os.path.splitext(f)[1].lower() in exts]
    
    if not files:
        return 0, 0, ["Aucune image trouvée"]

    os.makedirs(output_folder, exist_ok=True)
    
    success = 0
    errors = []
    
    for i, fname in enumerate(sorted(files)):
        in_path = os.path.join(input_folder, fname)
        base, ext = os.path.splitext(fname)
        out_path = os.path.join(output_folder, f"{base}_upscaled.png")
        
        if callback:
            callback(f"[{i+1}/{len(files)}] {fname}")
        
        ok, msg = upscale_image(
            model_path, in_path, out_path,
            scale=scale, tile_size=tile_size, tile_pad=tile_pad,
            use_amp=use_amp, callback=callback
        )
        
        if ok:
            success += 1
        else:
            errors.append(f"{fname}: {msg}")
    
    return success, len(files), errors
