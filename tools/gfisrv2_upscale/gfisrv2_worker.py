"""
GFISRv2 inference worker — subprocess.
Reports PROGRESS/LOG/ERROR/PREVIEW/DONE on stdout.
Persistent mode: load model once, read image requests from stdin as JSON lines.
"""
import sys
import os
import math
import argparse
import contextlib

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_HERE, "..", ".."))
_ARCH_DIR = os.path.join(_ROOT, "src", "core", "custom_archs")

_BUNDLE = getattr(sys, "_MEIPASS", None)
if _BUNDLE:
    sys.path.insert(0, _BUNDLE)
elif os.path.isdir(_ARCH_DIR):
    sys.path.insert(0, _ARCH_DIR)
else:
    sys.path.insert(0, _HERE)

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _log(msg):       print(f"LOG:{msg}",              flush=True)
def _err(msg):       print(f"ERROR:{msg}",            flush=True)
def _prog(c, t, n=""): print(f"PROGRESS:{c}:{t}:{n}", flush=True)
def _prev(a, b):     print(f"PREVIEW:{a}|{b}",        flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True)
    p.add_argument("--input",      default="")   # optional in persistent mode
    p.add_argument("--output",     required=True)
    p.add_argument("--scale",      default="auto")
    p.add_argument("--tile",       type=int, default=256)
    p.add_argument("--tile_pad",   type=int, default=32)
    p.add_argument("--amp",        type=int, default=1)
    p.add_argument("--modelname",  type=int, default=0)
    p.add_argument("--serialize",  type=int, default=0)
    p.add_argument("--ser_start",  type=int, default=0)
    p.add_argument("--dim",        type=int, default=48)
    p.add_argument("--n_blocks",   type=int, default=24)
    p.add_argument("--mid_dim",    type=int, default=32)
    p.add_argument("--upsampler",  default="dysample")
    p.add_argument("--format",     default="PNG")
    p.add_argument("--bits",       type=int, default=8)
    p.add_argument("--quality",    type=int, default=95)
    p.add_argument("--persistent", action="store_true", default=False)
    return p.parse_args()


def _load_ckpt(path: str) -> dict:
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("params_ema", "params_g", "params", "state_dict"):
            if k in ckpt:
                return ckpt[k]
    return ckpt


def _detect_scale(ckpt: dict) -> int:
    if "upscale.MetaUpsample" in ckpt:
        try:
            meta = ckpt["upscale.MetaUpsample"]
            s = int(meta[2].item() if hasattr(meta[2], "item") else meta[2])
            if s in (1, 2, 3, 4, 8):
                return s
        except Exception:
            pass
    return 1


def _get_device():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            cap = torch.cuda.get_device_capability()
            cap_str = f"sm_{cap[0]}{cap[1]}"
            supported = torch.cuda.get_arch_list()
            if supported and cap_str not in supported:
                _log(f"GPU {cap_str} incompatible PyTorch (supporte {supported}) → CPU/DirectML")
                device = "cpu"
        except Exception:
            pass
    if device == "cpu":
        try:
            import torch_directml
            dml_dev = torch_directml.device()
            _log(f"DirectML: {dml_dev}")
            return dml_dev
        except ImportError:
            pass
    return device


def tile_inference(model, t, tile: int, pad: int, scale: int):
    import torch
    B, C, H, W = t.shape
    out = t.new_zeros(B, C, H * scale, W * scale)
    tx  = math.ceil(W / tile)
    ty  = math.ceil(H / tile)
    for y in range(ty):
        for x in range(tx):
            ox, oy = x * tile, y * tile
            x0 = max(ox - pad, 0); x1 = min(ox + tile + pad, W)
            y0 = max(oy - pad, 0); y1 = min(oy + tile + pad, H)
            ti = t[:, :, y0:y1, x0:x1]
            with torch.no_grad():
                to_ = model(ti)
            ix0 = (ox - x0) * scale; iy0 = (oy - y0) * scale
            ex  = min(ox * scale + tile * scale, out.shape[3])
            ey  = min(oy * scale + tile * scale, out.shape[2])
            aw  = ex - ox * scale; ah = ey - oy * scale
            out[:, :, oy * scale:ey, ox * scale:ex] = \
                to_[:, :, iy0:iy0 + ah, ix0:ix0 + aw]
    return out


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
_EXT_MAP  = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif"}


def _save_image(arr_out, out_path: str, fmt: str, bits: int, quality: int):
    """Save float32 HxWx3 (0-1) array. NumPy 2.0 compatible."""
    import numpy as np
    from PIL import Image

    fmt_up = fmt.upper()
    save_kw = {}
    if bits == 16 and fmt_up in ("PNG", "TIFF"):
        arr16 = (arr_out * 65535.0).round().clip(0, 65535).astype(np.uint16)
        h, w = arr16.shape[:2]
        # .astype('>u2') converts to big-endian uint16 (NumPy 1.x and 2.x)
        arr_be = arr16.astype('>u2')
        img = Image.frombuffer("RGB", (w, h), bytes(arr_be.data), "raw", "RGB;16B", 0, 1)
        if fmt_up == "TIFF":
            save_kw["compression"] = "lzw"
    else:
        arr8 = (arr_out * 255.0).round().clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr8)
        if fmt_up in ("JPEG", "WEBP"):
            save_kw["quality"] = quality
        elif fmt_up == "TIFF":
            save_kw["compression"] = "lzw"

    img.save(out_path, format=fmt_up, **save_kw)


def _process_file(fpath: str, out_dir: str, model, device, scale: int, args, ser_idx: int) -> int:
    """Infer one image. Returns updated ser_idx."""
    import torch
    import numpy as np
    from PIL import Image

    name = os.path.basename(fpath)
    img = Image.open(fpath).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    use_amp = bool(args.amp) and str(device) == "cuda"
    tile    = args.tile
    pad     = args.tile_pad if args.tile_pad > 0 else max(tile // 8, 8)

    ctx = torch.autocast("cuda") if use_amp else contextlib.nullcontext()
    with ctx, torch.no_grad():
        out = tile_inference(model, t, tile, pad, scale) if tile > 0 else model(t)

    arr_out    = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out_ext    = _EXT_MAP.get(args.format.upper(), ".png")
    stem       = os.path.splitext(name)[0]
    model_stem = os.path.splitext(os.path.basename(args.model))[0]

    if args.serialize:
        out_name = f"{ser_idx:05d}{out_ext}"
        ser_idx += 1
    elif args.modelname:
        out_name = f"{stem}_{model_stem}{out_ext}"
    else:
        out_name = stem + out_ext

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    _save_image(arr_out, out_path, args.format, args.bits, args.quality)
    _prev(fpath, out_path)
    return ser_idx


def main():
    args = _parse()
    _prog(0, 1, "init")

    try:
        import torch
        from gfisrv2_arch import GFISRV2

        device = _get_device()
        _log(f"Device: {device}")
        _log(f"Loading model: {os.path.basename(args.model)}")

        ckpt  = _load_ckpt(args.model)
        scale_raw = args.scale.strip().lower()
        scale = _detect_scale(ckpt) if scale_raw == "auto" else int(scale_raw)
        _log(f"Scale: {scale}x")

        model = GFISRV2(
            dim=args.dim, n_blocks=args.n_blocks,
            mid_dim=args.mid_dim, scale=scale,
            upsampler=args.upsampler,
        )
        model.load_state_dict(ckpt, strict=True)
        model.eval().to(device)
        _log(f"Model OK  dim={args.dim} n_blocks={args.n_blocks} scale={scale}x")

        if args.persistent:
            # ── Persistent mode ────────────────────────────────────────────
            # Model loaded once. Read JSON requests from stdin, print DONE after each.
            print("READY", flush=True)
            import json
            ser_idx = args.ser_start
            for raw in sys.stdin:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    req = json.loads(raw)
                except json.JSONDecodeError as je:
                    _err(f"JSON decode error: {je}")
                    print("DONE", flush=True)
                    continue
                if req.get("quit"):
                    break
                inp     = req.get("input", "")
                out_dir = req.get("output", args.output)
                if not inp or not os.path.exists(inp):
                    _err(f"Input not found: {inp}")
                    print("DONE", flush=True)
                    continue
                try:
                    files = sorted(
                        os.path.join(inp, f) for f in os.listdir(inp)
                        if os.path.splitext(f)[1].lower() in _IMG_EXTS
                    ) if os.path.isdir(inp) else [inp]

                    total = len(files)
                    for i, fpath in enumerate(files):
                        _prog(i, total, os.path.basename(fpath))
                        ser_idx = _process_file(fpath, out_dir, model, device, scale, args, ser_idx)
                    _prog(total, total, "done")
                except Exception as e:
                    import traceback
                    _err(f"{type(e).__name__}: {e}")
                    _err(traceback.format_exc())
                print("DONE", flush=True)

        else:
            # ── Normal batch mode ──────────────────────────────────────────
            inp = args.input
            if not inp:
                _err("--input required in non-persistent mode"); sys.exit(1)

            files = sorted(
                os.path.join(inp, f) for f in os.listdir(inp)
                if os.path.splitext(f)[1].lower() in _IMG_EXTS
            ) if os.path.isdir(inp) else [inp]

            total = len(files)
            if total == 0:
                _err("No image files found"); sys.exit(1)

            ser_idx = args.ser_start
            for i, fpath in enumerate(files):
                _prog(i, total, os.path.basename(fpath))
                ser_idx = _process_file(fpath, args.output, model, device, scale, args, ser_idx)

            _prog(total, total, "done")
            print("DONE", flush=True)

    except Exception as e:
        import traceback
        _err(f"{type(e).__name__}: {e}")
        _err(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
