"""
redux_inference_runner.py — Inference pour toute arch traiNNer-redux.

Stratégie :
  1. Essaie spandrel (auto-détecte l'arch depuis le checkpoint)
  2. Si UnsupportedModelError → fallback traiNNer ARCH_REGISTRY (utilise arch_name)

Usage:
    <venv_python> redux_inference_runner.py <model_path> <input_path> <output_path> [arch_name]

Exit 0 = ok, 1 = erreur. Tout sur stdout (UTF-8, traceback inclus).
"""
import sys
import traceback as _tb
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_model_path  = sys.argv[1] if len(sys.argv) > 1 else ""
_input_path  = sys.argv[2] if len(sys.argv) > 2 else ""
_output_path = sys.argv[3] if len(sys.argv) > 3 else ""
_arch_hint   = sys.argv[4] if len(sys.argv) > 4 else ""

# ── traiNNer path ──────────────────────────────────────────────────────────────
import os
_REDUX = os.path.join(os.path.expanduser("~"), "IA_Engine", "traiNNer-redux")
if os.path.isdir(_REDUX) and _REDUX not in sys.path:
    sys.path.insert(0, _REDUX)


def _load_state_dict_raw(model_path: str) -> dict:
    """Charge le state dict brut depuis .safetensors ou .pth."""
    import torch
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


def _detect_scale_from_state_dict(state_dict: dict) -> int:
    """Estime le scale depuis les clés du state dict."""
    import math
    # Cherche les couches pixel-shuffle / depth_to_space
    for k, v in state_dict.items():
        kl = k.lower()
        if not hasattr(v, "shape"):
            continue
        # depth_to_space bias: shape [3*scale²]
        if ("depth_to_space" in kl or "pixel_shuffle" in kl) and "bias" in kl and len(v.shape) == 1:
            ch = v.shape[0]
            s = int(round(math.sqrt(ch / 3)))
            if s >= 1 and s * s * 3 == ch:
                return s
        # Upsample conv: out_ch = 3*scale² or in_ch*scale²
        if ("upsample" in kl or "upconv" in kl or "to_img" in kl or "tail" in kl) \
                and len(v.shape) == 4:
            out_ch = v.shape[0]
            s = int(round(math.sqrt(out_ch / 3)))
            if s >= 1 and s * s * 3 == out_ch:
                return s
    return 1  # défaut scale=1 (bench traiNNer toujours scale=1)


def _build_from_registry(arch_name: str, state_dict: dict):
    """Charge l'arch depuis le registry traiNNer, en détectant le scale."""
    import traiNNer.archs  # populate registry
    from traiNNer.utils.registry import ARCH_REGISTRY
    if arch_name not in ARCH_REGISTRY._obj_map:
        raise ValueError(f"Arch '{arch_name}' introuvable dans ARCH_REGISTRY. "
                         f"Archs dispo: {sorted(ARCH_REGISTRY._obj_map.keys())[:20]}")
    cls = ARCH_REGISTRY.get(arch_name)
    scale = _detect_scale_from_state_dict(state_dict)
    print(f"[Runner] Scale détecté: {scale}x", flush=True)
    # Essaie avec scale détecté, fallback sans arg
    try:
        model = cls(scale=scale)
    except TypeError:
        model = cls()
    # Chargement strict d'abord, puis non-strict si ça échoue
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[Runner] strict=True échoué ({e}), essai strict=False…", flush=True)
        model.load_state_dict(state_dict, strict=False)
    return model


def _tile_inference(model, tensor, tile_size: int = 512, overlap: int = 64):
    """Tiled inference pour grandes images."""
    import torch
    _, c, h, w = tensor.shape
    if h <= tile_size and w <= tile_size:
        return model(tensor)

    stride = tile_size - overlap
    t_h, t_w = min(tile_size, h), min(tile_size, w)
    with torch.no_grad():
        test_out = model(tensor[:, :, :t_h, :t_w])
    scale_h = test_out.shape[2] / t_h
    scale_w = test_out.shape[3] / t_w

    out_h, out_w = int(h * scale_h), int(w * scale_w)
    output = torch.zeros(1, c, out_h, out_w, device=tensor.device, dtype=tensor.dtype)
    weight = torch.zeros(1, 1, out_h, out_w, device=tensor.device, dtype=tensor.dtype)

    ys = list(range(0, h - tile_size, stride)) + [h - tile_size] if h > tile_size else [0]
    xs = list(range(0, w - tile_size, stride)) + [w - tile_size] if w > tile_size else [0]
    for y1 in ys:
        for x1 in xs:
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(y1 + tile_size, h), min(x1 + tile_size, w)
            with torch.no_grad():
                out_tile = model(tensor[:, :, y1:y2, x1:x2])
            oy1, oy2 = int(y1 * scale_h), int(y2 * scale_h)
            ox1, ox2 = int(x1 * scale_w), int(x2 * scale_w)
            output[:, :, oy1:oy2, ox1:ox2] += out_tile
            weight[:, :, oy1:oy2, ox1:ox2] += 1.0

    return output / weight.clamp(min=1.0)


def run(model_path: str, input_path: str, output_path: str, arch_hint: str = "") -> None:
    import torch
    import numpy as np
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Runner] Device: {device}", flush=True)

    # ── 1. Essai spandrel ─────────────────────────────────────────────────────
    model = None
    try:
        import spandrel
        loader = spandrel.ModelLoader(device=device)
        model_info = loader.load_from_file(model_path)
        model = model_info.model.eval()
        print(f"[Runner] Arch (spandrel): {type(model).__name__}", flush=True)
    except Exception as sp_err:
        sp_name = type(sp_err).__name__
        print(f"[Runner] spandrel échoué ({sp_name}), fallback traiNNer registry…", flush=True)

    # ── 2. Fallback traiNNer registry ─────────────────────────────────────────
    if model is None:
        if not arch_hint:
            raise RuntimeError(
                "spandrel ne reconnaît pas l'arch et aucun arch_hint fourni. "
                "Passer le nom d'arch en 4ème argument."
            )
        state_dict = _load_state_dict_raw(model_path)
        model = _build_from_registry(arch_hint, state_dict).to(device).eval()
        print(f"[Runner] Arch (traiNNer registry): {arch_hint}", flush=True)

    # ── Charger l'image ───────────────────────────────────────────────────────
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # ── Inférence ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        output = _tile_inference(model, tensor, tile_size=512, overlap=64)

    # ── Sauvegarder ───────────────────────────────────────────────────────────
    out_arr = output.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).float().cpu().numpy()
    out_img = Image.fromarray((out_arr * 255.0).round().astype(np.uint8))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)
    print(f"[Runner] Sauvegardé: {output_path}", flush=True)


if __name__ == "__main__":
    try:
        run(_model_path, _input_path, _output_path, _arch_hint)
    except Exception as e:
        print(f"[Runner] ERREUR: {type(e).__name__}: {e}", flush=True)
        _tb.print_exc(file=sys.stdout)  # traceback sur stdout (capturé par le bench)
        sys.exit(1)
