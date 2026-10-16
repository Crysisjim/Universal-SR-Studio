"""Output writer shared by the one-image inference runners (universal, spanplus,
neosr, neosr_general). Loose module next to them: the engine venv imports it from the
runner's own folder.

The Quick Upscale tab passes its format / bit depth / quality in the USS_SAVE_OPTS
environment variable (JSON). Without it the runner writes an 8-bit image in the format
of the output extension, as before."""
import json
import os

import numpy as np
from PIL import Image


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


def infer_opts():
    """Tile / tile pad / AMP chosen in the Quick Upscale tab (USS_INFER_OPTS, JSON).
    tile 0 = whole image; None = the runner's own default."""
    try:
        o = json.loads(os.environ.get("USS_INFER_OPTS") or "{}")
    except ValueError:
        o = {}
    tile = o.get("tile")
    return {"tile": None if tile is None else int(tile), "tile_pad": int(o.get("tile_pad") or 32),
            "amp": bool(o.get("amp"))}


def save_float(out_np, output_path, default_quality=95):
    """out_np: float32 [H, W, 3] in [0, 1]."""
    try:
        opts = json.loads(os.environ.get("USS_SAVE_OPTS") or "{}")
    except ValueError:
        opts = {}
    fmt = str(opts.get("format") or "").upper()
    bit_depth = int(opts.get("bit_depth") or 8)
    quality = int(opts.get("quality") or default_quality)
    low = output_path.lower()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if bit_depth == 16 and (fmt in ("PNG", "TIFF") or low.endswith((".png", ".tif", ".tiff"))):
        import cv2
        arr16 = (out_np * 65535.0).round().clip(0, 65535).astype(np.uint16)
        if not _imwrite_u(output_path, np.ascontiguousarray(arr16[:, :, ::-1])):
            raise RuntimeError(f"cv2.imwrite failed: {output_path}")
        return
    img = Image.fromarray((out_np * 255.0).round().clip(0, 255).astype(np.uint8))
    kw = {}
    if fmt in ("JPEG", "JPG") or low.endswith((".jpg", ".jpeg")):
        kw = {"quality": quality, "subsampling": 0}
    elif fmt == "WEBP" or low.endswith(".webp"):
        kw = {"quality": quality}
    elif fmt == "TIFF" or low.endswith((".tif", ".tiff")):
        kw = {"compression": "tiff_lzw"}
    img.save(output_path, **kw)
