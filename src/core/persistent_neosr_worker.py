"""
persistent_neosr_worker.py — Subprocess ESC longue durée (modèle chargé une fois).

Protocole JSON-lines via stdin/stdout :
  <- {"cmd": "init", "model": "<path>", "tile_size": 256, "tile_pad": 32, "use_amp": false}
  -> {"status": "ready", "arch": "ESC", "scale": 2}
  <- {"cmd": "infer", "input": "<path>", "output": "<path>"}
  -> {"status": "ok"}  |  {"status": "error", "msg": "..."}
  <- {"cmd": "quit"}
  -> (exit 0)

Lancement :
  <neosr_venv_python> persistent_neosr_worker.py
"""
import sys
import os
import json
import re
import math
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

NEOSR_PATH = os.path.join(os.path.expanduser("~"), "IA_Engine", "neosr")
if os.path.isdir(NEOSR_PATH) and NEOSR_PATH not in sys.path:
    sys.path.insert(0, NEOSR_PATH)
    os.chdir(NEOSR_PATH)

# Stub neosr packages to prevent net_opt() crash on import
import types as _t
for _n, _p in [
    ("neosr",       os.path.join(NEOSR_PATH, "neosr")),
    ("neosr.archs", os.path.join(NEOSR_PATH, "neosr", "archs")),
]:
    if _n not in sys.modules:
        _m = _t.ModuleType(_n)
        _m.__path__ = [_p]
        _m.__package__ = _n
        sys.modules[_n] = _m
if "neosr" in sys.modules and "neosr.archs" in sys.modules:
    sys.modules["neosr"].archs = sys.modules["neosr.archs"]
del _t, _n, _p, _m

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(json.dumps({"status": "fatal", "msg": f"Import error: {e}"}), flush=True)
    sys.exit(1)

# ── Global state ──────────────────────────────────────────────────────────────
_model = None
_scale: int = 1
_device = None
_tile_size: int = 0
_tile_pad: int = 32
_use_amp: bool = False
_last_hw: "tuple | None" = None   # (H, W) frame précédente — détecte changement de résolution


def _emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _load_state_dict(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params_ema", "params_g", "params", "model", "state_dict"):
        if key in ck:
            return ck[key]
    if any(k.endswith(".weight") for k in ck.keys()):
        return ck
    return ck


def _infer_esc_params(sd: dict) -> dict:
    proj_w = sd.get("proj.weight")
    dim = proj_w.shape[0] if proj_w is not None else 64

    is_fp = "lk_channel" in sd
    pdim_t = sd.get("lk_channel") if is_fp else sd.get("plk_filter")
    pdim = pdim_t.shape[0] if pdim_t is not None else 16

    block_indices = set()
    for k in sd:
        m = re.match(r"^blocks\.(\d+)\.", k)
        if m:
            block_indices.add(int(m.group(1)))
    n_blocks = len(block_indices) if block_indices else 5

    realsr = "skip.0.weight" in sd
    use_dysample = "to_img.offset.weight" in sd

    scale = 1
    if use_dysample:
        offset_ch = sd["to_img.offset.weight"].shape[0]
        scale_sq = offset_ch / 8.0
        s = int(round(math.sqrt(scale_sq)))
        if s >= 1 and s * s == int(scale_sq):
            scale = s
    elif is_fp:
        to_img_w = sd.get("to_img.weight")
        if to_img_w is not None:
            out_ch = to_img_w.shape[0]
            scale_sq = out_ch / 3.0
            s = int(round(math.sqrt(scale_sq)))
            if s >= 1 and s * s == int(scale_sq):
                scale = s

    return {
        "dim": dim, "pdim": pdim, "n_blocks": n_blocks,
        "is_fp": is_fp, "realsr": realsr,
        "use_dysample": use_dysample, "upscaling_factor": scale,
    }


def _tile_infer(model, img_t, tile: int, pad: int, scale: int):
    b, c, h, w = img_t.shape
    if tile == 0 or (h <= tile and w <= tile):
        with torch.inference_mode():
            return model(img_t)
    out = torch.zeros(b, c, h * scale, w * scale, device=img_t.device, dtype=img_t.dtype)
    tiles_x = math.ceil(w / tile)
    tiles_y = math.ceil(h / tile)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x1 = max(tx * tile - pad, 0)
            y1 = max(ty * tile - pad, 0)
            x2 = min((tx + 1) * tile + pad, w)
            y2 = min((ty + 1) * tile + pad, h)
            with torch.inference_mode():
                tile_out = model(img_t[:, :, y1:y2, x1:x2])
            ox1 = (tx * tile - x1) * scale
            oy1 = (ty * tile - y1) * scale
            out_x1 = tx * tile * scale
            out_y1 = ty * tile * scale
            out_x2 = min(out_x1 + tile * scale, w * scale)
            out_y2 = min(out_y1 + tile * scale, h * scale)
            out[:, :, out_y1:out_y2, out_x1:out_x2] = tile_out[
                :, :, oy1:oy1 + (out_y2 - out_y1), ox1:ox1 + (out_x2 - out_x1)]
    return out


def cmd_init(payload: dict) -> None:
    global _model, _scale, _device, _tile_size, _tile_pad, _use_amp, _last_hw
    _last_hw = None
    model_path = payload["model"]
    _tile_size = int(payload.get("tile_size", 256))
    _tile_pad  = int(payload.get("tile_pad", 32))
    _use_amp   = bool(payload.get("use_amp", False))

    if not os.path.isfile(model_path):
        _emit({"status": "error", "msg": f"Modèle introuvable : {model_path}"})
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # neosr force cudnn.benchmark=True à l'import. Pour l'inférence multi-résolution
    # c'est nocif : chaque nouvelle taille d'image relance un autotune cuDNN complet
    # (workspace transitoire énorme → spike VRAM + stall), puis fragmente l'allocateur
    # et ralentit tout le reste du batch. On le désactive pour rester stable.
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        sd = _load_state_dict(model_path)
    except Exception as e:
        _emit({"status": "error", "msg": f"Chargement state_dict : {e}"})
        return

    try:
        params = _infer_esc_params(sd)
        _scale = params["upscaling_factor"]
    except Exception as e:
        _emit({"status": "error", "msg": f"Détection paramètres ESC : {e}"})
        return

    try:
        import neosr.archs.arch_util as _au
        _au.net_opt = lambda: (_scale, True)
        from neosr.archs.esc_arch import esc
        _model = esc(
            dim=params["dim"],
            pdim=params["pdim"],
            n_blocks=params["n_blocks"],
            upscaling_factor=_scale,
            is_fp=params["is_fp"],
            use_dysample=params["use_dysample"],
            realsr=params["realsr"],
            attn_type="sdpa",
        )
        _model.load_state_dict(sd, strict=True)
        _model = _model.to(_device).eval()
    except Exception as e:
        _emit({"status": "error", "msg": f"Chargement modèle ESC : {e}\n{traceback.format_exc()}"})
        return

    _emit({"status": "ready", "arch": "ESC", "scale": _scale})


def cmd_infer(payload: dict) -> None:
    global _last_hw
    if _model is None:
        _emit({"status": "error", "msg": "Modèle non initialisé."})
        return

    input_path  = payload["input"]
    output_path = payload["output"]

    try:
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_t  = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(_device)

        # Changement de résolution → défragmente l'allocateur avant l'inférence
        # (évite buildup VRAM réservée + thrashing cudaMalloc sur batch mixte).
        cur_hw = (img_t.shape[2], img_t.shape[3])
        if _device.type == "cuda" and _last_hw is not None and cur_hw != _last_hw:
            torch.cuda.empty_cache()
        _last_hw = cur_hw

        if _use_amp and _device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                out_t = _tile_infer(_model, img_t, _tile_size, _tile_pad, _scale)
        else:
            out_t = _tile_infer(_model, img_t, _tile_size, _tile_pad, _scale)

        out = out_t.squeeze(0).clamp(0, 1).cpu().float().numpy()
        out = (np.transpose(out, (1, 2, 0)) * 255.0).round().astype(np.uint8)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        Image.fromarray(out).save(output_path)
        _emit({"status": "ok"})
    except Exception as e:
        _emit({"status": "error", "msg": f"{e}\n{traceback.format_exc()}"})


def main_loop() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            _emit({"status": "error", "msg": f"JSON invalide : {e}"})
            continue

        cmd = payload.get("cmd", "")
        if cmd == "init":
            cmd_init(payload)
        elif cmd == "infer":
            cmd_infer(payload)
        elif cmd == "quit":
            sys.exit(0)
        else:
            _emit({"status": "error", "msg": f"Commande inconnue : {cmd!r}"})


if __name__ == "__main__":
    main_loop()
