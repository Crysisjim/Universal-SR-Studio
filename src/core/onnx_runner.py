"""
onnx_runner.py — Standalone ONNX super-resolution inference subprocess.

Run inside an engine venv (which has onnxruntime-gpu) because the frozen
portable exe does not bundle onnxruntime. Mirrors _onnx_upscale() in
quick_upscale.py.

Usage:
    python onnx_runner.py <model.onnx> <input> <output> [out_format] [bit_depth] [quality]
"""
import sys
import os

# Force UTF-8 stdout so emoji/accents don't crash on cp1252
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _save_output(arr, path, out_format, bit_depth, quality):
    import numpy as np
    from PIL import Image
    out_format = (out_format or "PNG").upper()
    if bit_depth == 16 and out_format in ("PNG", "TIFF"):
        a16 = (arr.clip(0, 1) * 65535.0).round().astype(np.uint16)
        # PIL doesn't support uint16 RGB directly — use cv2 (always in venv)
        try:
            import cv2
            bgr = a16[:, :, ::-1] if a16.ndim == 3 else a16  # RGB→BGR for cv2
            cv2.imwrite(path, bgr)
            return
        except Exception:
            pass
        # Fallback: save as 8-bit if cv2 unavailable
        a8 = (arr.clip(0, 1) * 255.0).round().astype(np.uint8)
        Image.fromarray(a8).save(path)
        return
    a8 = (arr.clip(0, 1) * 255.0).round().astype(np.uint8)
    im = Image.fromarray(a8)
    if out_format in ("JPG", "JPEG"):
        im.save(path, quality=int(quality))
    elif out_format == "WEBP":
        im.save(path, quality=int(quality))
    else:
        im.save(path)


def main():
    if len(sys.argv) < 4:
        print("Usage: onnx_runner.py <model.onnx> <input> <output> [fmt] [bits] [quality]", flush=True)
        sys.exit(2)

    model_path, input_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    out_format = sys.argv[4] if len(sys.argv) > 4 else "PNG"
    bit_depth = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    quality = int(sys.argv[6]) if len(sys.argv) > 6 else 95

    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNX] ERREUR : onnxruntime non installé dans le venv.", flush=True)
        sys.exit(1)
    import numpy as np
    from PIL import Image

    providers = []
    try:
        import torch
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
    except Exception:
        pass
    providers.append("CPUExecutionProvider")

    print("[ONNX] Chargement du modele...", flush=True)
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"[ONNX] ERREUR chargement : {e}", flush=True)
        sys.exit(1)
    print(f"[ONNX] Providers actifs = {sess.get_providers()}", flush=True)

    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = np.transpose(img_np, (2, 0, 1))[np.newaxis]  # [1,3,H,W]

    inp_name = sess.get_inputs()[0].name
    if "float16" in sess.get_inputs()[0].type:
        img_t = img_t.astype(np.float16)

    print(f"[ONNX] Inference {img.width}x{img.height}...", flush=True)
    try:
        out_np = sess.run(None, {inp_name: img_t})[0]
    except Exception as e:
        print(f"[ONNX] ERREUR inference : {e}", flush=True)
        sys.exit(1)

    out_np = out_np.astype(np.float32).squeeze(0).clip(0, 1)  # [3,H,W]
    out_np = np.transpose(out_np, (1, 2, 0))                  # [H,W,3]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    _save_output(out_np, output_path, out_format, bit_depth, quality)
    print(f"[ONNX] Sauvegarde : {output_path}", flush=True)


if __name__ == "__main__":
    main()
