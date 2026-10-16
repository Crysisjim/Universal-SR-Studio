"""
bundle_vapoursynth.py — copy the minimal VapourSynth runtime + AA plugins from a
detected install (StaxRip) into vendor/vapoursynth/ so the portable build runs real
nnedi3 / EEDI3 anti-aliasing with NO external dependency.

vs_aa.py searches vendor/vapoursynth/ FIRST (before StaxRip / PATH), so once this
script has populated it the app is fully self-contained.

Usage:
    python tools/bundle_vapoursynth.py                 # auto-detect source (StaxRip)
    python tools/bundle_vapoursynth.py --src "<path to FrameServer\\VapourSynth>" --plugins "<path to Plugins\\VS>"
    python tools/bundle_vapoursynth.py --no-imwri      # skip the 24 MB ImageMagick reader (needs a raw I/O path — experimental)

Honest size note:
    runtime (python3xx.dll + vapoursynth.dll + stdlib + VSPipe.exe) ~30 MB
    + nnedi3 weights .bin                                            ~13 MB
    + znedi3 (CPU) + NNEDI3CL (GPU OpenCL) + EEDI3m                  ~4 MB
    + libimwri (image I/O, ImageMagick)                             ~24 MB
    → ~70 MB with imwri (recommended/reliable), ~47 MB without (experimental).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

_DEF_SRC = [
    r"C:\Program Files\StaxRip\Apps\FrameServer\VapourSynth",
    r"C:\Program Files (x86)\StaxRip\Apps\FrameServer\VapourSynth",
]
_DEF_PLUGINS = [
    r"C:\Program Files\StaxRip\Apps\Plugins\VS",
    r"C:\Program Files (x86)\StaxRip\Apps\Plugins\VS",
]

# Installer leftovers safe to drop (the portable runtime structure must stay intact —
# VapourSynth needs its portable.vs marker, python ._pth, stdlib zip, plugin dirs, etc.,
# so we copy the WHOLE tree minus these clearly-unused download artifacts).
_RUNTIME_DROP = (
    "vapoursynth64-portable", "python-3", "get-pip.py", "pfm-",
)

# Plugin DLLs to bundle (+ the weights file).
_PLUGIN_FILES = [
    ("nnedi3", "vsznedi3.dll"),
    ("nnedi3", "NNEDI3CL.dll"),
    ("nnedi3", "nnedi3_weights.bin"),
    ("EEDI3m", "EEDI3m.dll"),
]
_IMWRI = ("LibImwri", "libimwri.dll")


def _find(cands):
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


def _copy_runtime(src, dst):
    """Copy the entire portable runtime tree (preserving structure so VSScript inits),
    skipping only the unused installer download artifacts."""
    total = 0
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_dir = os.path.join(dst, rel) if rel != "." else dst
        for fn in files:
            if any(s in fn.lower() for s in _RUNTIME_DROP):
                continue
            os.makedirs(out_dir, exist_ok=True)
            sp = os.path.join(root, fn)
            try:
                shutil.copy2(sp, os.path.join(out_dir, fn))
                total += os.path.getsize(sp)
            except Exception:
                pass
    return total


def _copy_plugins(plugins_root, dst, with_imwri):
    os.makedirs(dst, exist_ok=True)
    total = 0
    wanted = list(_PLUGIN_FILES) + ([_IMWRI] if with_imwri else [])
    for sub, name in wanted:
        # search recursively (folder layout varies)
        found = None
        for root, _dirs, files in os.walk(plugins_root):
            low = {f.lower(): f for f in files}
            if name.lower() in low:
                found = os.path.join(root, low[name.lower()])
                break
        if found:
            shutil.copy2(found, os.path.join(dst, name))
            total += os.path.getsize(found)
            print(f"  + {name}  ({os.path.getsize(found)/1e6:.1f} MB)")
        else:
            print(f"  ! introuvable : {name}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="FrameServer/VapourSynth folder")
    ap.add_argument("--plugins", default=None, help="Plugins/VS folder")
    ap.add_argument("--no-imwri", action="store_true", help="skip libimwri (saves ~24 MB)")
    args = ap.parse_args()

    src = args.src or _find(_DEF_SRC)
    plugins = args.plugins or _find(_DEF_PLUGINS)
    if not src or not os.path.isdir(src):
        print("VapourSynth runtime introuvable. Passe --src.", file=sys.stderr); return 2
    if not plugins or not os.path.isdir(plugins):
        print("Plugins VS introuvables. Passe --plugins.", file=sys.stderr); return 2

    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vendor = os.path.join(app_root, "vendor", "vapoursynth")
    print(f"Source runtime : {src}")
    print(f"Source plugins : {plugins}")
    print(f"Destination    : {vendor}\n")

    print("Runtime…")
    t1 = _copy_runtime(src, vendor)
    print(f"  runtime ~{t1/1e6:.1f} MB")
    print("Plugins…")
    t2 = _copy_plugins(plugins, os.path.join(vendor, "plugins"), with_imwri=not args.no_imwri)
    print(f"\nTotal bundle ~{(t1 + t2)/1e6:.1f} MB -> {vendor}")
    print("vs_aa.py utilisera ce bundle en priorite (avant StaxRip/PATH).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
