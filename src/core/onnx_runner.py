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


def _imwrite_u(path, arr):
    """cv2.imwrite that also works with non-ASCII Windows paths (imencode + tofile)."""
    import os as _os
    import cv2 as _cv2
    ext = _os.path.splitext(path)[1] or ".png"
    ok, buf = _cv2.imencode(ext, arr)
    if not ok:
        return False
    buf.tofile(path)
    return True


def _save_output(arr, path, out_format, bit_depth, quality):
    import numpy as np
    from PIL import Image
    out_format = (out_format or "PNG").upper()
    if bit_depth == 16 and out_format in ("PNG", "TIFF"):
        a16 = (arr.clip(0, 1) * 65535.0).round().astype(np.uint16)
        # PIL doesn't support uint16 RGB directly — use cv2 (always in venv)
        import cv2
        bgr = a16[:, :, ::-1] if a16.ndim == 3 else a16  # RGB→BGR for cv2
        if not _imwrite_u(path, np.ascontiguousarray(bgr)):
            raise RuntimeError(f"16-bit write failed: {path}")
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

    # torch BEFORE onnxruntime: onnxruntime-gpu then reuses the CUDA 12 / cuDNN 9 DLLs
    # shipped with torch (no separate CUDA install needed).
    providers = []
    try:
        import torch
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
    except Exception:
        pass
    providers.append("CPUExecutionProvider")
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNX] ERREUR : onnxruntime non installé dans le venv.", flush=True)
        sys.exit(1)
    try:
        ort.set_default_logger_severity(4)  # native ORT log = UTF-16 text -> garbage in the UI log
    except Exception:
        pass
    import numpy as np
    from PIL import Image

    print("[ONNX] Chargement du modele...", flush=True)
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 4
        sess = ort.InferenceSession(model_path, so, providers=providers)
    except Exception as e:
        print(f"[ONNX] ERREUR chargement : {e}", flush=True)
        sys.exit(1)
    print(f"[ONNX] Providers actifs = {sess.get_providers()}", flush=True)
    if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in sess.get_providers():
        print(f"[ONNX] ⚠ GPU refusé par onnxruntime {ort.__version__} → calcul sur CPU (lent). "
              "onnxruntime-gpu >= 1.27 est compilé pour CUDA 13 (incompatible avec PyTorch CUDA 12 "
              "et les GPU Pascal) : Paramètres → « Installer / Réinstaller dépendances » remet la "
              "bonne version.", flush=True)

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
