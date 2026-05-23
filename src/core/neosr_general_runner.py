"""
neosr_general_runner.py - Inference subprocess for neosr architectures.

Handles archs that call net_opt() at module level (ninasr, lmlt, eimn, drct_s, etc.)
by patching arch_util.net_opt BEFORE any arch import.

Usage:
    <neosr_venv_python> neosr_general_runner.py <model_path> <input_path> <output_path>

Exit 0 = success, 1 = error. All output to stdout (UTF-8).
"""
import sys
import os
import math
import re
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

NEOSR_PATH = os.path.join(os.path.expanduser("~"), "IA_Engine", "neosr")
if os.path.isdir(NEOSR_PATH) and NEOSR_PATH not in sys.path:
    sys.path.insert(0, NEOSR_PATH)
    os.chdir(NEOSR_PATH)

# Stub neosr packages to prevent __init__.py from running
# (avoids parse_options() / rng() module-level calls requiring -opt argv)
import types as _t
for _n, _p in [
    ('neosr',       os.path.join(NEOSR_PATH, 'neosr')),
    ('neosr.archs', os.path.join(NEOSR_PATH, 'neosr', 'archs')),
]:
    if _n not in sys.modules:
        _m = _t.ModuleType(_n)
        _m.__path__ = [_p]
        _m.__package__ = _n
        sys.modules[_n] = _m
if 'neosr' in sys.modules and 'neosr.archs' in sys.modules:
    sys.modules['neosr'].archs = sys.modules['neosr.archs']
del _t, _n, _p, _m

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"[NeoSR] ERREUR import: {e}", flush=True)
    sys.exit(1)


# ── Architecture signatures (key substrings → arch name) ──────────────────────
ARCH_SIGNATURES = {
    "ninasr": ["head.0.bias", "body.0.body.2.body"],
    "lmlt":   ["feats.0.lhsb", "to_img.0.weight"],
    "eimn":   ["block1.0.attn.region", "block1.0.layer_scale_1"],
    "drct":   ["layers.0.swin1", "layers.0.adjust1"],
    "span":   ["feats.0.block", "feats.0.sk"],
    "omnisr": ["residual_layer", "osag"],
    "realplksr": ["feats.0.lk.conv", "dysample"],
    "compact":   ["body.0.weight", "body.2.weight"],
    "esrgan":    ["body.0.rdb1", "RRDB_trunk"],
    "rcan":      ["body.0.body.0.body.0.weight"],
}


def detect_arch(state_dict: dict) -> str | None:
    keys_str = " ".join(state_dict.keys())
    for arch, sigs in ARCH_SIGNATURES.items():
        if all(s in keys_str for s in sigs):
            return arch
    return None


def detect_scale(state_dict: dict) -> int:
    # ninasr tail.1, eimn tail.0, lmlt to_img.0 → out_ch = 3 * scale^2
    for probe in ("tail.1.weight", "tail.0.weight", "to_img.0.weight"):
        w = state_dict.get(probe)
        if w is not None:
            ch = w.shape[0]
            s = int(round(math.sqrt(ch / 3)))
            if s >= 1 and s * s * 3 == ch:
                return s
    # compact arch: last body conv → out_ch = 3 * scale^2
    # body layout: [first_conv, act, (conv,act)*num_conv, last_conv]
    # last_conv shape = [3*scale^2, num_feat, 3, 3]
    compact_conv_idxs = sorted(
        int(k.split(".")[1]) for k in state_dict
        if k.startswith("body.") and k.endswith(".weight")
        and len(state_dict[k].shape) == 4
    )
    if compact_conv_idxs:
        last_body_key = f"body.{compact_conv_idxs[-1]}.weight"
        w = state_dict.get(last_body_key)
        if w is not None:
            out_ch = w.shape[0]
            s = int(round(math.sqrt(out_ch / 3)))
            if s >= 1 and s * s * 3 == out_ch:
                return s
    # drct / swinir-style pixel_shuffle upsamplers
    for k, v in state_dict.items():
        kl = k.lower()
        if ("upsample" in kl or "upconv" in kl) and hasattr(v, "shape") and len(v.shape) == 4:
            out_ch, in_ch = v.shape[0], v.shape[1]
            if out_ch > in_ch:
                s = int(math.sqrt(out_ch / in_ch))
                if s in (2, 3, 4, 8):
                    return s
    return 1  # default: 1x for neosr benchmark models


def _load_state_dict(model_path: str) -> dict:
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(model_path, device="cpu")
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    for key in ("params_ema", "params_g", "params", "model", "state_dict"):
        if key in ck:
            return ck[key]
    if any(k.endswith(".weight") for k in ck.keys()):
        return ck
    return ck


def build_model(arch: str, state_dict: dict, scale: int):
    """Instantiate arch and load weights. Returns model on CPU."""
    import neosr.archs.arch_util as _au
    _au.net_opt = lambda: (scale, True)

    if arch == "ninasr":
        from neosr.archs.ninasr_arch import ninasr
        model = ninasr(scale=scale)

    elif arch == "lmlt":
        from neosr.archs.lmlt_arch import lmlt
        model = lmlt(upscaling_factor=scale)

    elif arch == "eimn":
        from neosr.archs.eimn_arch import eimn
        head_w = state_dict.get("head.0.weight")
        embed_dims = int(head_w.shape[0]) if head_w is not None else 64
        num_stages = sum(
            1 for k in state_dict
            if k.startswith("block") and k.endswith(".region.weight")
        )
        model = eimn(
            embed_dims=embed_dims, scale=scale,
            num_stages=num_stages if num_stages > 0 else 16,
        )

    elif arch == "drct":
        from neosr.archs.drct_arch import drct, drct_s
        cf_w = state_dict.get("conv_first.weight")
        if cf_w is not None and cf_w.shape[0] <= 60:
            model = drct_s(upscale=scale)
        else:
            model = drct(upscale=scale)

    elif arch == "span":
        from neosr.archs.span_arch import span
        model = span(scale=scale)

    elif arch == "omnisr":
        from neosr.archs.omnisr_arch import omnisr
        model = omnisr(scale=scale)

    elif arch == "realplksr":
        from neosr.archs.realplksr_arch import realplksr
        model = realplksr(scale=scale)

    elif arch == "compact":
        from neosr.archs.compact_arch import compact
        # Detect num_feat and num_conv from state_dict to match saved model exactly
        w0 = state_dict.get("body.0.weight")
        num_feat = int(w0.shape[0]) if w0 is not None else 64
        # body layout: [first_conv, first_act, (conv, act)*num_conv, last_conv]
        # → last conv is at index 2 + 2*num_conv  → num_conv = (last_idx - 2) // 2
        conv_idxs = sorted(
            int(k.split(".")[1]) for k in state_dict
            if k.startswith("body.") and k.endswith(".weight")
            and len(state_dict[k].shape) == 4  # Conv2d weights are 4D
        )
        last_idx = conv_idxs[-1] if conv_idxs else 34
        num_conv = max(0, (last_idx - 2) // 2)
        model = compact(num_feat=num_feat, num_conv=num_conv, upscale=scale)

    elif arch == "esrgan":
        from neosr.archs.rrdbnet_arch import rrdbnet
        model = rrdbnet(scale=scale)

    else:
        raise ValueError(f"Arch non supportée dans ce runner: {arch}")

    model.load_state_dict(state_dict, strict=True)
    return model


def run(model_path: str, input_path: str, output_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NeoSR] Device: {device}", flush=True)

    state_dict = _load_state_dict(model_path)
    arch = detect_arch(state_dict)
    scale = detect_scale(state_dict)
    print(f"[NeoSR] Arch détectée: {arch or 'inconnue'}, scale: {scale}x", flush=True)

    if arch is None:
        raise ValueError("Architecture inconnue — impossible de charger le modèle.")

    model = build_model(arch, state_dict, scale)
    model = model.to(device).eval()
    print(f"[NeoSR] Modèle chargé: {arch}", flush=True)

    img = Image.open(input_path).convert("RGB")
    print(f"[NeoSR] Inference {img.width}x{img.height} -> {img.width*scale}x{img.height*scale}...", flush=True)

    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

    import torch.nn.functional as F_pad
    _, _, H, W = img_t.shape
    ws = getattr(model, 'window_size', 1)
    if isinstance(ws, (list, tuple)):
        ws = ws[0]

    def _infer_padded(tile: "torch.Tensor") -> "torch.Tensor":
        """Run model on a single tile, padding to window_size multiple, then crop."""
        _, _, th, tw = tile.shape
        ph = (ws - th % ws) % ws
        pw = (ws - tw % ws) % ws
        if ph > 0 or pw > 0:
            tile = F_pad.pad(tile, (0, pw, 0, ph), mode='reflect')
        with torch.no_grad():
            out = model(tile)
        return out[:, :, :th * scale, :tw * scale]

    TILE = 512
    PAD  = 32
    # For large images with big window_size (drct, swinir, hat), go straight to tiling
    # to avoid OOM on the full-image attempt.
    FORCE_TILE = ws >= 16 and H * W > 512 * 512
    if FORCE_TILE:
        print(f"[NeoSR] Large image + window_size={ws} -> tiling {TILE}px directly", flush=True)
    # Try full-image inference first; fall back to tiling on OOM
    if not FORCE_TILE:
        try:
            out = _infer_padded(img_t)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
                raise
            torch.cuda.empty_cache()
            FORCE_TILE = True
            print(f"[NeoSR] OOM on full image — switching to {TILE}px tiling", flush=True)

    if FORCE_TILE:
        out = torch.zeros((1, 3, H * scale, W * scale), device=img_t.device)
        for y0 in range(0, H, TILE):
            for x0 in range(0, W, TILE):
                x1_src = max(0, x0 - PAD)
                y1_src = max(0, y0 - PAD)
                x2_src = min(W, x0 + TILE + PAD)
                y2_src = min(H, y0 + TILE + PAD)
                tile = img_t[:, :, y1_src:y2_src, x1_src:x2_src]
                tile_out = _infer_padded(tile)
                # Determine paste region (remove pad overlap)
                ox = (x0 - x1_src) * scale
                oy = (y0 - y1_src) * scale
                ow = min(TILE, W - x0) * scale
                oh = min(TILE, H - y0) * scale
                out[:, :, y0*scale:y0*scale+oh, x0*scale:x0*scale+ow] = \
                    tile_out[:, :, oy:oy+oh, ox:ox+ow]
                torch.cuda.empty_cache()

    out_np = out.squeeze(0).clamp(0, 1).cpu().float().numpy()
    out_np = np.transpose(out_np, (1, 2, 0))
    out_np = (out_np * 255.0).round().astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(out_np).save(output_path)
    print(f"[NeoSR] Sauvegardé: {output_path}", flush=True)

    del model, img_t, out


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: neosr_general_runner.py <model_path> <input_path> <output_path>")
        sys.exit(1)
    _model_path, _input_path, _output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.argv = [sys.argv[0]]
    try:
        run(_model_path, _input_path, _output_path)
    except Exception as e:
        print(f"[NeoSR] ERREUR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
