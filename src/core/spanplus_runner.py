"""
spanplus_runner.py - Standalone SPANPlus inference runner.

Called as a subprocess by quick_upscale.py when the main program's Python
environment doesn't have spandrel (required by SPANPlus / DySample).

Usage:
    <traiNNer_venv_python> spanplus_runner.py <model_path> <input_path> <output_path>

All output is printed to stdout for the parent process to capture.
Exit code 0 = success, 1 = error.
"""
import sys
import os
import math
import traceback

# Force UTF-8 output so special chars don't crash on cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TRAINNER_PATH = os.path.join(os.path.expanduser("~"), "IA_Engine", "traiNNer-redux")
if TRAINNER_PATH not in sys.path:
    sys.path.insert(0, TRAINNER_PATH)
if os.path.isdir(TRAINNER_PATH):
    os.chdir(TRAINNER_PATH)

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"[Runner] ❌ Import error: {e}", flush=True)
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


def run(model_path: str, input_path: str, output_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Runner] Device : {device}", flush=True)

    # --- Load weights ---
    print(f"[Runner] Chargement : {os.path.basename(model_path)}", flush=True)
    state_dict = _load_state_dict(model_path)

    # --- Auto-detect feature_channels ---
    fc_t = state_dict.get("feats.0.eval_conv.weight",
                          state_dict.get("feats.0.sk.weight", None))
    fc = fc_t.shape[0] if fc_t is not None else 48

    # --- Auto-detect scale & upsampler ---
    upscale = 1
    upsampler = "conv"
    if "upsampler.offset.weight" in state_dict:
        upsampler = "dys"
        ch = state_dict["upsampler.offset.weight"].shape[0]
        s = int(math.sqrt(ch / 8.0))
        if s >= 1 and s * s == int(ch / 8.0):
            upscale = s

    print(f"[Runner] SPANPlus fc={fc}, scale={upscale}x, upsampler={upsampler}", flush=True)

    # --- Instantiate model ---
    from traiNNer.archs.spanplus_arch import SpanPlus
    model = SpanPlus(feature_channels=fc, upscale=upscale, upsampler=upsampler).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("[Runner] Modèle prêt.", flush=True)

    # --- Load image ---
    img = Image.open(input_path).convert("RGB")
    print(f"[Runner] Inférence {img.width}x{img.height} → {img.width*upscale}x{img.height*upscale}...", flush=True)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

    # --- Inference ---
    with torch.no_grad():
        out = model(img_t)

    # --- Save ---
    out = out.squeeze(0).clamp(0, 1).cpu().float().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = (out * 255.0).round().astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(out).save(output_path, quality=95)
    print(f"[Runner] ✅ Sauvegardé : {output_path}", flush=True)

    del model, img_t, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spanplus_runner.py <model_path> <input_path> <output_path>", flush=True)
        sys.exit(1)

    _, model_path, input_path, output_path = sys.argv

    try:
        run(model_path, input_path, output_path)
    except Exception:
        print("[Runner] ❌ ERREUR :", flush=True)
        traceback.print_exc()
        sys.exit(1)
