"""16-bit aware image reading / writing / resizing for the app process.

The portable app bundles pyvips (libvips) but not OpenCV (~126 MB). These helpers use
OpenCV when it is importable (dev runs, engine venv), else pyvips, else Pillow (8-bit).
Arrays are float32 RGB in [0, 1]."""
import numpy as np


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


def read_float01(path):
    """Image file as float32 RGB [H, W, 3] in [0, 1]; 16-bit files keep their precision."""
    try:
        import cv2
        a = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if a is not None:
            if a.ndim == 2:
                a = np.stack([a] * 3, -1)
            a = a[:, :, :3][:, :, ::-1]
            return a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)
    except ImportError:
        pass
    try:
        import pyvips
        v = pyvips.Image.new_from_file(path, access="sequential")
        a = v.numpy()
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        a = a[:, :, :3]
        if a.dtype == np.uint16:
            return a.astype(np.float32) / 65535.0
        if a.dtype == np.uint8:
            return a.astype(np.float32) / 255.0
    except Exception:
        pass
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def write16(path, arr01):
    """Write float32 RGB [0, 1] as a real 16-bit PNG/TIFF. Raises when no writer can."""
    a16 = np.ascontiguousarray((np.clip(arr01, 0.0, 1.0) * 65535.0).round().astype(np.uint16))
    try:
        import cv2
        if not _imwrite_u(path, np.ascontiguousarray(a16[:, :, ::-1])):
            raise RuntimeError(f"16-bit write failed: {path}")
        return
    except ImportError:
        pass
    import pyvips
    h, w, c = a16.shape
    v = pyvips.Image.new_from_memory(a16.tobytes(), w, h, c, "ushort")
    if path.lower().endswith((".tif", ".tiff")):
        v.write_to_file(path, compression="lzw")
    else:
        v.write_to_file(path)


def resize_float(arr, size):
    """Bicubic resize of a float32 [H, W, C] array to size = (w, h), full precision."""
    w, h = size
    try:
        import cv2
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)
    except ImportError:
        pass
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(arr[:, :, c].astype(np.float32), "F")
                                .resize((w, h), Image.BICUBIC), np.float32)
                     for c in range(arr.shape[2])], -1)
