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
    # SMoSR: Self-Modulate SR (umzi2) — MetaUpsample buffer is very distinctive
    "smosr":  ["blocks_2.0.body", "end_block.0.body", "upsampler.MetaUpsample"],
    # SpanF: variant SPAN avec SPAB1 blocks et conv_near depthwise
    "spanf":  ["block_1.conv1.eval_conv", "conv_near.weight", "block_5.conv1"],
    # SpanC: SPAN reparamétrisable + IGConv multi-scale (MetaIGConv buffer)
    "spanc":  ["block_1.conv_a.eval_conv", "upsampler.coord_map", "MetaIGConv"],
    # GFISRv2: GatedCNN body + UniUpsampleV3 MetaUpsample
    "gfisrv2": ["gfisr_body.0.fc1", "upscale.MetaUpsample", "in_to_dim.weight"],
    # CATANet: Token Aggregation Block (IRCA+IASA) + LRSA (NeoSR, mars 2025)
    # Clés réelles vérifiées sur state_dict : irca_attn / iasa_attn + first_conv
    "catanet": ["blocks.0.0.irca_attn.to_k.weight", "blocks.0.0.iasa_attn.to_q.weight", "first_conv.weight"],
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
    # SMoSR: MetaUpsample buffer stores [254, method_idx, scale, in_dim, out_dim, mid_dim, group, rep]
    if "upsampler.MetaUpsample" in state_dict:
        meta = state_dict["upsampler.MetaUpsample"]
        try:
            s = int(meta[2].item())
            if s in (1, 2, 3, 4, 8):
                return s
        except Exception:
            pass
    # GFISRv2: UniUpsampleV3 stores same MetaUpsample at upscale.MetaUpsample
    if "upscale.MetaUpsample" in state_dict:
        meta = state_dict["upscale.MetaUpsample"]
        try:
            s = int(meta[2].item())
            if s in (1, 2, 3, 4, 8):
                return s
        except Exception:
            pass
    # SpanC: MetaIGConv buffer stores list of scale values as uint8 tensor
    # eval_base_scale cannot be recovered from state_dict — use min of scale_list as default
    if "MetaIGConv" in state_dict:
        meta = state_dict["MetaIGConv"]
        try:
            scales = [int(v.item()) for v in meta]
            if scales:
                # Return max scale (most common use case)
                s = max(scales)
                if s in (1, 2, 3, 4, 8):
                    return s
        except Exception:
            pass
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


# ─── Color Fix (post-upscale color drift correction) ────────────────────────

def _cf_resolve_device(device_pref: str) -> str:
    """
    Retourne le device effectif selon préférence et disponibilité.
    'auto'  → cuda si dispo, sinon cpu
    'cuda'  → force GPU (fallback cpu si absent)
    'trt'   → torch.compile max-autotune sur GPU (fallback cuda, puis cpu)
    'cpu'   → toujours PIL
    """
    if device_pref in ("cuda", "trt"):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return device_pref   # préserve la distinction cuda/trt
        return "cpu"
    if device_pref == "auto":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cpu"


# ── Cache pour la version torch.compile du BoxBlur (TRT mode) ────────────────
_cf_trt_fn = None   # type: ignore


def _cf_get_trt_boxblur():
    """
    Retourne _cf_boxblur_gpu compilé avec torch.compile(mode='max-autotune').
    Premier appel déclenche l'autotuning Triton (~3 s sur RTX) — résultat mis en cache.
    Fallback transparent vers _cf_boxblur_gpu si torch.compile non disponible.
    """
    global _cf_trt_fn
    if _cf_trt_fn is None:
        if TORCH_AVAILABLE:
            try:
                _cf_trt_fn = torch.compile(
                    _cf_boxblur_gpu,
                    mode="max-autotune",
                    fullgraph=False,
                )
            except Exception:
                _cf_trt_fn = _cf_boxblur_gpu   # torch < 2.0 → fallback
        else:
            _cf_trt_fn = _cf_boxblur_gpu
    return _cf_trt_fn


def _cf_boxblur_gpu(t: "torch.Tensor", r: int, c: int) -> "torch.Tensor":
    """
    BoxBlur séparable 1D sur GPU (PyTorch F.conv2d).
    Entrée/sortie : [1, C, H, W] float32 sur CUDA.
    Utilise replicate-padding → comportement bord identique à PIL BoxBlur.
    """
    import torch.nn.functional as F
    r = max(1, int(r))
    k = 2 * r + 1
    dev = t.device
    # Filtres séparables : horizontal puis vertical
    kh = torch.ones(c, 1, 1, k, device=dev, dtype=torch.float32) / k
    kv = torch.ones(c, 1, k, 1, device=dev, dtype=torch.float32) / k
    t = F.pad(t, (r, r, 0, 0), mode="replicate")
    t = F.conv2d(t, kh, groups=c)
    t = F.pad(t, (0, 0, r, r), mode="replicate")
    t = F.conv2d(t, kv, groups=c)
    return t


def color_fix_image(
    sr_np: "np.ndarray",
    lq_pil: "Image.Image",
    method: str = "wavelet",
    wavelets: int = 4,
    radius: int = 10,
    fast: bool = False,
    strength: float = 1.0,
    planes: "Optional[list]" = None,
    device: str = "auto",
    ref_pil: "Optional[Image.Image]" = None,
) -> "np.ndarray":
    """
    Fix color drift post-upscale. Equivalent to vs_colorfix (pifroggi).

    method='wavelet': ATWT (à trous wavelet transform) multi-niveaux.
        Décompose SR et REF en N niveaux de fréquences progressives.
        Remplace uniquement le résidu basse-fréquence (couleur globale) du SR par celui du REF.
        Préserve intégralement le détail haute-fréquence du SR (textures, contours).
        wavelets=4 → rayon effectif 2^4=16px. Equiv. vs_colorfix.wavelet(wavelets=4).
        Recommandé : 3–5.

    method='average': region-average single-level (equivalent vs_colorfix.average).
        fast=False → BoxBlur (précis). fast=True → downscale+upscale (10× plus rapide).
        Recommandé : radius 5–30.

    device: 'auto'  → CUDA si dispo sinon CPU.
            'cuda'  → force GPU PyTorch F.conv2d.
            'trt'   → torch.compile max-autotune (Triton autotuning, ~3s warmup).
            'cpu'   → toujours PIL.
        GPU (~10-30× plus rapide que PIL sur 4K). Pas de VapourSynth requis.

    ref_pil: image de référence alternative (vs LQ par défaut).
        Recommandé : source haute qualité (PNG/TIFF 16-bit) pour éviter le banding.
        Si None → utilise lq_pil.

    planes : canaux à corriger [0,1,2]=RGB (défaut). Ex: [1,2]=GB seulement.
    strength: blend [0.0=off, 1.0=correction complète].

    Returns: float32 [H, W, 3] in [0, 1]
    """
    from PIL import ImageFilter as _IF

    if planes is None:
        planes = [0, 1, 2]
    if strength <= 0.0 or not planes:
        return sr_np

    h, w = sr_np.shape[:2]
    nc = sr_np.shape[2]  # nombre de canaux (3 pour RGB)

    # ── Source de référence (LQ par défaut, ou image externe) ────────────────
    _src_pil = ref_pil if ref_pil is not None else lq_pil
    lq_up = np.array(_src_pil.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0

    # ── Résolution du device ──────────────────────────────────────────────────
    resolved = _cf_resolve_device(device)
    use_gpu  = (resolved in ("cuda", "trt") and TORCH_AVAILABLE)
    _boxblur = _cf_get_trt_boxblur() if resolved == "trt" else _cf_boxblur_gpu

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _boxblur_pil(arr: "np.ndarray", r: int) -> "np.ndarray":
        pil = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))
        return np.array(pil.filter(_IF.BoxBlur(max(1, r))), dtype=np.float32) / 255.0

    def _np_to_t(arr: "np.ndarray") -> "torch.Tensor":
        """[H,W,C] numpy → [1,C,H,W] CUDA float32."""
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to("cuda", dtype=torch.float32)

    def _t_to_np(t: "torch.Tensor") -> "np.ndarray":
        """[1,C,H,W] CUDA → [H,W,C] numpy float32."""
        return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().float().numpy()

    # ── WAVELET ───────────────────────────────────────────────────────────────
    if method == "wavelet":
        if use_gpu:
            try:
                sr_t = _np_to_t(sr_np)
                lq_t = _np_to_t(lq_up)
                for level in range(max(1, int(wavelets))):
                    r = 2 ** level  # 1, 2, 4, 8, 16, 32...
                    sr_t = _boxblur(sr_t, r, nc)
                    lq_t = _boxblur(lq_t, r, nc)
                correction = _t_to_np(lq_t - sr_t)
            except Exception:
                # Fallback PIL si CUDA plante (OOM, etc.)
                use_gpu = False

        if not use_gpu:
            sr_low = sr_np.copy()
            lq_low = lq_up.copy()
            for level in range(max(1, int(wavelets))):
                r = 2 ** level
                sr_low = _boxblur_pil(sr_low, r)
                lq_low = _boxblur_pil(lq_low, r)
            correction = lq_low - sr_low

    # ── AVERAGE ───────────────────────────────────────────────────────────────
    elif method == "average":
        if fast:
            # Downscale+upscale (chaiNNer-style) — toujours CPU (déjà très rapide)
            factor = max(2, int(radius))
            sm_w, sm_h = max(1, w // factor), max(1, h // factor)
            sr_dn = Image.fromarray((sr_np * 255.0).clip(0, 255).astype(np.uint8)).resize(
                (sm_w, sm_h), Image.BILINEAR)
            lq_dn = _src_pil.resize((sm_w, sm_h), Image.BILINEAR)
            sr_low = np.array(sr_dn.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
            lq_low = np.array(lq_dn.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        elif use_gpu:
            try:
                sr_t = _np_to_t(sr_np)
                lq_t = _np_to_t(lq_up)
                sr_t = _boxblur(sr_t, int(radius), nc)
                lq_t = _boxblur(lq_t, int(radius), nc)
                correction = _t_to_np(lq_t - sr_t)
            except Exception:
                use_gpu = False

        if not use_gpu and not fast:
            sr_low = _boxblur_pil(sr_np, int(radius))
            lq_low = _boxblur_pil(lq_up, int(radius))
            correction = lq_low - sr_low

        if fast:
            correction = lq_low - sr_low

    else:
        return sr_np

    fixed = sr_np.copy()
    for c in planes:
        fixed[:, :, c] = np.clip(sr_np[:, :, c] + correction[:, :, c], 0.0, 1.0)

    if strength < 1.0:
        fixed = sr_np * (1.0 - strength) + fixed * strength
    return np.clip(fixed, 0.0, 1.0)


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

# Archs à exécuter via subprocess traiNNer-redux même si importables en-process.
# Raison : le kernel CUDA compilé dans le venv traiNNer est requis.
# - spanplus : DySample update_params() → F.conv2d kernel absent du torch système (Python 3.14 / sm_61)
# - smosr, gfisrv2 : MetaUpsample (famille DySample) → même problème probable
# - spanc  : MetaIGConv coord-based → incertain, subprocess par sécurité
# - spanf  : conv depthwise standard, mais subprocess pour cohérence venv
_TRAINNER_SUBPROCESS_ARCHS = {"spanplus", "smosr", "gfisrv2", "spanc", "spanf"}

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
    color_fix: str = "none",
    color_fix_wavelets: int = 4,
    color_fix_radius: int = 32,
    color_fix_fast: bool = False,
    color_fix_strength: float = 1.0,
    color_fix_planes: Optional[list] = None,
    color_fix_device: str = "auto",
    color_fix_ref: str = "",
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

        # ── Color Fix post-subprocess helper ──────────────────────────────────────
        def _post_colorfix(sub_result):
            """Applique Color Fix sur le fichier sauvegardé par un subprocess."""
            _ok, _info = sub_result
            if (_ok and color_fix != "none" and color_fix_strength > 0.0
                    and IMAGING_AVAILABLE and os.path.isfile(output_path)):
                try:
                    _sr_arr = (np.array(Image.open(output_path).convert("RGB"))
                               .astype(np.float32) / 255.0)
                    _lq_pil = Image.open(input_path).convert("RGB")
                    _planes = color_fix_planes if color_fix_planes is not None else [0, 1, 2]
                    log(f"Color Fix ✓ [{color_fix}] strength={color_fix_strength:.2f}")
                    _sr_arr = color_fix_image(_sr_arr, _lq_pil,
                                              method=color_fix,
                                              wavelets=color_fix_wavelets,
                                              radius=color_fix_radius,
                                              fast=color_fix_fast,
                                              strength=color_fix_strength,
                                              planes=_planes,
                                              device=color_fix_device)
                    _save_output(_sr_arr, output_path, out_format, bit_depth, quality)
                except Exception as _cf_e:
                    log(f"Color Fix subprocess : erreur ({_cf_e})")
            return _ok, _info

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
                    return _post_colorfix(_spanplus_subprocess_infer(
                        model_path, input_path, output_path, log,
                        progress_callback, stop_event=stop_event,
                        tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp))
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

            if detected_arch in _TRAINNER_SUBPROCESS_ARCHS:
                log(f"{detected_arch} → subprocess traiNNer-redux venv (kernel CUDA requis)...")
                _store_subprocess("trainner")
                return _post_colorfix(_spanplus_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event,
                    tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp
                ))

            # ── Try to instantiate model in-process ──
            model = None

            # Try neosr imports
            if detected_arch and model is None:
                try:
                    neosr_path = os.path.join(os.path.expanduser("~"), "IA_Engine", "neosr")
                    if os.path.isdir(neosr_path) and neosr_path not in sys.path:
                        sys.path.insert(0, neosr_path)
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
                    elif detected_arch == "catanet":
                        # net_opt() doit être patché avant l'import (lecture module-level)
                        import neosr.archs.arch_util as _cat_au
                        _saved_net_opt = _cat_au.net_opt
                        _cat_au.net_opt = lambda: (scale, True)
                        try:
                            from neosr.archs.catanet_arch import catanet as _CATANet
                            w = state_dict.get("first_conv.weight")
                            dim = int(w.shape[0]) if w is not None else 40
                            model = _CATANet(dim=dim, upscale=scale)
                            log(f"CATANet instancié (dim={dim}, scale={scale}x)")
                        finally:
                            _cat_au.net_opt = _saved_net_opt
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
                    elif detected_arch == "smosr":
                        from traiNNer.archs.smosr_arch import SMoSR
                        w = state_dict.get("blocks_1.0.body.0.W")
                        dim = int(w.shape[0]) if w is not None else 48
                        n_mb = len([k for k in state_dict if k.startswith("blocks_2.") and k.endswith(".body.0.W")])
                        n_mb = n_mb if n_mb > 0 else 3
                        meta = state_dict.get("upsampler.MetaUpsample")
                        if meta is not None and scale == 0:
                            try:
                                scale = max(1, int(meta[2].item()))
                            except Exception:
                                scale = 4
                        model = SMoSR(scale=max(1, scale), dim=dim, n_mb=n_mb)
                        log(f"SMoSR instancié (dim={dim}, n_mb={n_mb}, scale={scale}x)")
                    elif detected_arch == "spanf":
                        from traiNNer.archs.spanf_arch import spanf
                        w = state_dict.get("block_1.conv1.eval_conv.weight")
                        fc = int(w.shape[0]) if w is not None else 32
                        model = spanf(feature_channels=fc, scale=max(1, scale))
                        log(f"SpanF instancié (fc={fc}, scale={scale}x)")
                    elif detected_arch == "spanc":
                        from traiNNer.archs.spanpp_arch import SpanC
                        meta = state_dict.get("MetaIGConv")
                        if meta is not None:
                            scale_list = tuple(int(v.item()) for v in meta)
                        else:
                            scale_list = (2, 4)
                        w = state_dict.get("conv0.eval_conv.weight")
                        fc = int(w.shape[0]) if w is not None else 48
                        model = SpanC(feature_channels=fc, scale_list=scale_list,
                                      eval_base_scale=max(1, scale) if scale in scale_list else scale_list[0])
                        log(f"SpanC instancié (fc={fc}, scales={scale_list})")
                    elif detected_arch == "gfisrv2":
                        from traiNNer.archs.gfisrv2_arch import GFISRV2
                        w = state_dict.get("in_to_dim.weight")
                        dim = int(w.shape[0]) if w is not None else 48
                        n_blocks = len([k for k in state_dict if k.startswith("gfisr_body.") and k.endswith(".fc1.weight")])
                        n_blocks = n_blocks if n_blocks > 0 else 24
                        meta = state_dict.get("upscale.MetaUpsample")
                        if meta is not None and scale == 0:
                            try:
                                scale = max(1, int(meta[2].item()))
                            except Exception:
                                scale = 4
                        model = GFISRV2(scale=max(1, scale), dim=dim, n_blocks=n_blocks)
                        log(f"GFISRv2 instancié (dim={dim}, n_blocks={n_blocks}, scale={scale}x)")
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
                return _post_colorfix(_spanplus_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event,
                    tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp
                ))

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
            try:
                if tile_size > 0 and (img.width > tile_size or img.height > tile_size):
                    log(f"Mode tiling : {tile_size}px, padding {tile_pad}px")
                    output = tile_inference(model, img_tensor, tile_size, tile_pad, scale,
                                            stop_event=stop_event)
                else:
                    output = model(img_tensor)
            except Exception as _cuda_e:
                _esc = str(_cuda_e)
                if "no kernel image" in _esc or "cudaErrorNoKernelImageForDevice" in _esc:
                    log("⚠ Kernel CUDA incompatible → fallback CPU (plus lent)")
                    _model_cpu = model.cpu()
                    _tensor_cpu = img_tensor.cpu()
                    with torch.amp.autocast("cpu", enabled=False):
                        if tile_size > 0 and (img.width > tile_size or img.height > tile_size):
                            output = tile_inference(_model_cpu, _tensor_cpu, tile_size, tile_pad,
                                                    scale, stop_event=stop_event)
                        else:
                            output = _model_cpu(_tensor_cpu)
                    # output reste sur CPU — .cpu() ligne suivante est no-op
                    if device.type == "cuda":
                        model.to(device)  # remet le modèle sur GPU pour les appels suivants
                else:
                    raise

        # ── Save output ──
        output = output.squeeze(0).clamp(0, 1).cpu().float().numpy()
        output = np.transpose(output, (1, 2, 0))   # [H, W, 3] float32 [0,1]

        # ── Color Fix (post-process) ──
        if color_fix != "none" and color_fix_strength > 0.0 and IMAGING_AVAILABLE:
            _planes = color_fix_planes if color_fix_planes is not None else [0, 1, 2]
            _cf_dev = _cf_resolve_device(color_fix_device)

            # ── Chargement image de référence (optionnel) ──
            _ref_pil = None
            if color_fix_ref and os.path.isfile(color_fix_ref):
                try:
                    _ref_pil = Image.open(color_fix_ref).convert("RGB")
                    log(f"Color Fix ref : {os.path.basename(color_fix_ref)}")
                except Exception as _e:
                    log(f"Color Fix ref : impossible de charger ({_e}) — LQ utilisé")

            _ref_label = f" ref={os.path.basename(color_fix_ref)}" if _ref_pil else ""
            if color_fix == "wavelet":
                log(f"Color Fix [wavelet] wavelets={color_fix_wavelets} "
                    f"strength={color_fix_strength:.2f} planes={_planes} "
                    f"device={_cf_dev}{_ref_label}")
            else:
                log(f"Color Fix [average] radius={color_fix_radius} fast={color_fix_fast} "
                    f"strength={color_fix_strength:.2f} planes={_planes} "
                    f"device={_cf_dev}{_ref_label}")

            if _cf_dev == "trt":
                log("Color Fix TRT : premier appel → autotuning Triton (~3 s), "
                    "résultat mis en cache pour les prochaines images.")

            output = color_fix_image(output, img,
                                     method=color_fix,
                                     wavelets=color_fix_wavelets,
                                     radius=color_fix_radius,
                                     fast=color_fix_fast,
                                     strength=color_fix_strength,
                                     planes=_planes,
                                     device=color_fix_device,
                                     ref_pil=_ref_pil)

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
