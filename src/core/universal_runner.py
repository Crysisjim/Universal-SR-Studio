"""
universal_runner.py - General-purpose SR inference subprocess.

Tries spandrel first (handles SPAN, ESRGAN, RealPLKSR, SwinIR, etc.).
Falls back to manual SpanPlus instantiation for traiNNer-redux models
that spandrel does not yet support.

Usage:
    <traiNNer_venv_python> universal_runner.py <model_path> <input_path> <output_path>

Exit 0 = success, 1 = error. All output to stdout (UTF-8).
"""
import sys
import os
import math
import traceback

# Force UTF-8 so Unicode chars don't crash on cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TRAINNER_PATH = os.path.join(os.path.expanduser("~"), "IA_Engine", "traiNNer-redux")
if os.path.isdir(TRAINNER_PATH) and TRAINNER_PATH not in sys.path:
    sys.path.insert(0, TRAINNER_PATH)
    os.chdir(TRAINNER_PATH)

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"[Runner] ERREUR import: {e}", flush=True)
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


def run(model_path: str, input_path: str, output_path: str,
        tile_size: int = 0, tile_pad: int = 32, use_amp: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Runner] Device : {device} | tile={tile_size}px | amp={use_amp}", flush=True)

    model = None
    scale = 1

    # ── 1. Try spandrel (handles SPAN, ESRGAN, RealPLKSR, SwinIR, HAT, …) ──
    try:
        import spandrel
        print(f"[Runner] Chargement via spandrel : {os.path.basename(model_path)}", flush=True)
        loader = spandrel.ModelLoader(device=device)
        descriptor = loader.load_from_file(model_path)
        model = descriptor.model
        scale = descriptor.scale
        print(f"[Runner] spandrel OK : {type(descriptor).__name__}, scale={scale}x", flush=True)
    except Exception as e:
        print(f"[Runner] spandrel echec ({e}), tentative manuelle...", flush=True)

    # ── 2. Fallback: manual SpanPlus (traiNNer-redux, not yet in spandrel) ──
    if model is None:
        try:
            from traiNNer.archs.spanplus_arch import SpanPlus
            state_dict = _load_state_dict(model_path)

            fc_t = state_dict.get("feats.0.eval_conv.weight",
                                  state_dict.get("feats.0.sk.weight", None))
            fc = fc_t.shape[0] if fc_t is not None else 48

            upsampler = "conv"
            if "upsampler.offset.weight" in state_dict:
                upsampler = "dys"
                ch = state_dict["upsampler.offset.weight"].shape[0]
                s = int(math.sqrt(ch / 8.0))
                if s >= 1 and s * s == int(ch / 8.0):
                    scale = s

            model = SpanPlus(feature_channels=fc, upscale=scale, upsampler=upsampler)
            model.load_state_dict(state_dict, strict=True)
            print(f"[Runner] SpanPlus manuel : fc={fc}, scale={scale}x, up={upsampler}", flush=True)
        except Exception as e:
            print(f"[Runner] SpanPlus manuel echec : {e}", flush=True)
            model = None

    if model is None:
        print("[Runner] ERREUR : Impossible d'instancier le modele.", flush=True)
        sys.exit(1)

    model = model.to(device).eval()
    print("[Runner] Modele pret.", flush=True)

    # ── Inference ──
    img = Image.open(input_path).convert("RGB")
    print(f"[Runner] Inference {img.width}x{img.height} -> {img.width * scale}x{img.height * scale}...",
          flush=True)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

    amp_ctx = torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda"))

    def _infer_tile(img_tensor, ts, tp):
        """Tiled inference with overlap to save VRAM."""
        _, _, H, W = img_tensor.shape
        out = torch.zeros((1, 3, H * scale, W * scale), device=img_tensor.device)
        for y0 in range(0, H, ts):
            for x0 in range(0, W, ts):
                x1s = max(0, x0 - tp); y1s = max(0, y0 - tp)
                x2s = min(W, x0 + ts + tp); y2s = min(H, y0 + ts + tp)
                tile = img_tensor[:, :, y1s:y2s, x1s:x2s]
                with torch.no_grad(), amp_ctx:
                    tile_out = model(tile)
                ox = (x0 - x1s) * scale; oy = (y0 - y1s) * scale
                ow = min(ts, W - x0) * scale; oh = min(ts, H - y0) * scale
                out[:, :, y0*scale:y0*scale+oh, x0*scale:x0*scale+ow] = \
                    tile_out[:, :, oy:oy+oh, ox:ox+ow]
                torch.cuda.empty_cache()
        return out

    _, _, H, W = img_t.shape
    use_tiling = tile_size > 0 and (W > tile_size or H > tile_size)
    if use_tiling:
        print(f"[Runner] Tiling {tile_size}px (pad={tile_pad}px)", flush=True)
        out = _infer_tile(img_t, tile_size, tile_pad)
    else:
        with torch.no_grad(), amp_ctx:
            out = model(img_t)

    # ── Save ──
    out = out.squeeze(0).clamp(0, 1).cpu().float().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = (out * 255.0).round().astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(out).save(output_path, quality=95)
    print(f"[Runner] Sauvegarde : {output_path}", flush=True)

    del model, img_t, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: universal_runner.py <model_path> <input_path> <output_path> [tile_size] [tile_pad] [use_amp]", flush=True)
        sys.exit(1)
    _tile_size = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    _tile_pad  = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    _use_amp   = sys.argv[6] == "1" if len(sys.argv) > 6 else False
    try:
        run(sys.argv[1], sys.argv[2], sys.argv[3],
            tile_size=_tile_size, tile_pad=_tile_pad, use_amp=_use_amp)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
