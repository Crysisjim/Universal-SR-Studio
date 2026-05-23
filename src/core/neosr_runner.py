"""
neosr_runner.py - ESC/neosr inference subprocess.

Handles ESC arch which calls net_opt() at module level (requires -opt argv).
Patches net_opt in arch_util BEFORE importing esc_arch to avoid the crash.

Usage:
    <neosr_venv_python> neosr_runner.py <model_path> <input_path> <output_path>

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

# Stub neosr and neosr.archs packages before any neosr import.
#
# Without stubs:
#   import neosr.archs.arch_util
#   → neosr/__init__.py: from neosr.data import *
#   → neosr/data/__init__.py: import neosr.data.otf_dataset
#   → otf_dataset.py: from neosr.data.degradations import ...
#   → degradations.py: rng = rng()          ← module-level call
#   → rng.py: opt, __ = parse_options(...)   ← requires -opt in sys.argv
#   → ValueError: Didn't get a config
#
# With stubs: Python uses our empty module objects instead of running
# __init__.py, then finds arch files via __path__ and imports them directly.
# neosr/archs/__init__.py is also skipped to prevent it from auto-importing
# esc_arch (which calls net_opt() at module level before our patch).
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
# Link neosr.archs as attribute of neosr stub so dotted access works
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


def _infer_esc_params(state_dict: dict) -> dict:
    """Infer ESC constructor parameters from state_dict keys and shapes."""
    # dim: proj = Conv2d(3, dim, 3, 1, 1)  →  proj.weight.shape = [dim, 3, 3, 3]
    proj_w = state_dict.get("proj.weight")
    dim = proj_w.shape[0] if proj_w is not None else 64

    # is_fp variant uses lk_channel/lk_spatial; standard uses plk_filter
    is_fp = "lk_channel" in state_dict

    # pdim: plk_filter.shape[0] (standard) or lk_channel.shape[0] (is_fp)
    if is_fp:
        pdim_t = state_dict.get("lk_channel")
    else:
        pdim_t = state_dict.get("plk_filter")
    pdim = pdim_t.shape[0] if pdim_t is not None else 16

    # n_blocks: count unique block indices in keys like "blocks.N.*"
    block_indices = set()
    for k in state_dict:
        m = re.match(r"^blocks\.(\d+)\.", k)
        if m:
            block_indices.add(int(m.group(1)))
    n_blocks = len(block_indices) if block_indices else 5

    # realsr: presence of skip connection (skip.0.weight)
    realsr = "skip.0.weight" in state_dict

    # use_dysample: DySample offset conv present as to_img.offset.weight
    use_dysample = "to_img.offset.weight" in state_dict

    # upscaling_factor: infer from DySample offset channels (groups=4 → 8*scale²)
    scale = 1
    if use_dysample:
        offset_ch = state_dict["to_img.offset.weight"].shape[0]
        scale_sq = offset_ch / 8.0
        s = int(round(math.sqrt(scale_sq)))
        if s >= 1 and s * s == int(scale_sq):
            scale = s
    elif is_fp:
        # pixel_shuffle path: to_img = Conv2d(dim, 3*scale², 3, 1, 1)
        to_img_w = state_dict.get("to_img.weight")
        if to_img_w is not None:
            out_ch = to_img_w.shape[0]
            scale_sq = out_ch / 3.0
            s = int(round(math.sqrt(scale_sq)))
            if s >= 1 and s * s == int(scale_sq):
                scale = s

    return {
        "dim": dim,
        "pdim": pdim,
        "n_blocks": n_blocks,
        "is_fp": is_fp,
        "realsr": realsr,
        "use_dysample": use_dysample,
        "upscaling_factor": scale,
    }


def run(model_path: str, input_path: str, output_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NeoSR] Device : {device}", flush=True)

    state_dict = _load_state_dict(model_path)
    params = _infer_esc_params(state_dict)
    scale = params["upscaling_factor"]
    print(f"[NeoSR] ESC params : dim={params['dim']}, pdim={params['pdim']}, "
          f"n_blocks={params['n_blocks']}, scale={scale}x, "
          f"dysample={params['use_dysample']}, realsr={params['realsr']}", flush=True)

    # Patch net_opt BEFORE importing esc_arch.
    # esc_arch calls `upscale, __ = net_opt()` at module level which requires -opt in sys.argv.
    import neosr.archs.arch_util as _au
    _au.net_opt = lambda: (scale, True)

    from neosr.archs.esc_arch import esc

    model = esc(
        dim=params["dim"],
        pdim=params["pdim"],
        n_blocks=params["n_blocks"],
        upscaling_factor=scale,
        is_fp=params["is_fp"],
        use_dysample=params["use_dysample"],
        realsr=params["realsr"],
        attn_type="sdpa",
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    print(f"[NeoSR] ESC chargé.", flush=True)

    # Inference
    img = Image.open(input_path).convert("RGB")
    out_w = img.width * scale
    out_h = img.height * scale
    print(f"[NeoSR] Inférence {img.width}x{img.height} -> {out_w}x{out_h}...", flush=True)

    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)

    out = out.squeeze(0).clamp(0, 1).cpu().float().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = (out * 255.0).round().astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(out).save(output_path)
    print(f"[NeoSR] Sauvegarde : {output_path}", flush=True)

    del model, img_t, out


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: neosr_runner.py <model_path> <input_path> <output_path>", flush=True)
        sys.exit(1)
    # Save our args and strip sys.argv BEFORE any neosr import.
    # neosr's options parser reads sys.argv at import time and fails on unknown positional args.
    _model_path, _input_path, _output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.argv = [sys.argv[0]]
    try:
        run(_model_path, _input_path, _output_path)
    except Exception as e:
        print(f"[NeoSR] ERREUR : {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
