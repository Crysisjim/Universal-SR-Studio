# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — GFISRv2 Upscale standalone (CUDA edition).

DESIGN : torch / torchvision / numpy ne sont PAS bundlés (trop gros + la variante
CUDA dépend du GPU cible). L'utilisateur les télécharge via le bouton
"Install Dependencies" qui installe la bonne wheel CUDA dans _deps/ à côté de l'exe
(auto-détection de la compute capability via nvidia-smi).

  - GTX 10xx (Pascal, sm_61) -> cu118
  - RTX 20xx+ (sm_75+)        -> cu126

Le bundle ne contient que la GUI (customtkinter, PIL, tkinterdnd2), einops,
safetensors et l'arch GFISRv2. La GUI démarre sans torch ; l'inférence requiert
l'install CUDA (un clic).

Build (depuis un env SANS torch lourd — torch est exclu de toute façon) :
    cd tools\\gfisrv2_upscale
    pyinstaller gfisrv2_standalone.spec --distpath ../../dist --workpath ../../build_tmp/gfisrv2
"""
import os
import customtkinter as _ctk

HERE     = os.path.dirname(os.path.abspath(SPEC))
ROOT     = os.path.abspath(os.path.join(HERE, "..", ".."))
ARCH_DIR = os.path.join(ROOT, "src", "core", "custom_archs")
CTK_DIR  = os.path.join(os.path.dirname(_ctk.__file__), "assets")

_HIDDEN = [
    "gfisrv2_arch",          # compiled from ARCH_DIR via pathex
    "customtkinter",
    "tkinterdnd2",
    "PIL",
    "PIL.Image",
    "PIL.ImageOps",
    "PIL.ImageTk",
    "einops",
    "contextlib",
    "safetensors",
    "pynvml",
    "psutil",
]

# torch + numpy are loaded at runtime from _deps/ but are NOT analyzed at build time
# (excluded), so PyInstaller would only bundle the stdlib modules the GUI itself uses.
# torch/numpy need much more stdlib (glob, ctypes, multiprocessing, ...). Force the
# COMPLETE standard library into the bundle so the runtime CUDA torch can import.
import sys as _sys
_STDLIB = sorted(m for m in getattr(_sys, "stdlib_module_names", set())
                 if not m.startswith("_"))
_HIDDEN += _STDLIB
print(f"[SPEC] bundling full stdlib ({len(_STDLIB)} modules) for runtime torch")

_EXTRA_DATAS, _EXTRA_BINARIES = [], []

# Bundle an embeddable Python 3.12 + pip.pyz so the app can install the CUDA PyTorch
# wheel into _deps/ with NO system Python required (true standalone). The embeddable
# is cp312 → matches this exe → installs cp312 wheels that load correctly.
_VENDOR  = os.path.join(HERE, "_vendor")
_PYEMBED = os.path.join(_VENDOR, "pyembed")
_PIPPYZ  = os.path.join(_VENDOR, "pip.pyz")
if os.path.isdir(_PYEMBED) and os.path.isfile(_PIPPYZ):
    _EXTRA_DATAS += [(_PYEMBED, "pyembed"), (_PIPPYZ, ".")]
    print("[SPEC] embeddable Python + pip.pyz bundled (standalone installer)")
else:
    print("[SPEC] WARNING: _vendor/pyembed or pip.pyz missing — install needs system Python")

block_cipher = None

a = Analysis(
    [os.path.join(HERE, "gfisrv2_app.py")],
    # pathex includes ARCH_DIR so PyInstaller compiles gfisrv2_arch.py into the bundle
    pathex=[HERE, ROOT, ARCH_DIR],
    binaries=_EXTRA_BINARIES,
    datas=[
        # customtkinter themes/icons
        (CTK_DIR, "customtkinter/assets"),
        # worker script bundled as data (used only in dev mode, harmless in exe)
        (os.path.join(HERE, "gfisrv2_worker.py"), "."),
    ] + _EXTRA_DATAS,
    hiddenimports=_HIDDEN,
    hookspath=[],
    runtime_hooks=[],
    # torch/torchvision/numpy are NOT bundled — installed into _deps/ at runtime
    # (CUDA wheel chosen per-GPU). Excluding them keeps the exe light and avoids
    # bundling a wrong-variant torch. safetensors.torch is imported lazily.
    # NOTE: do NOT exclude "unittest" — torch._dispatch.python imports unittest.mock.
    excludes=["tkinter.test", "torch", "torchvision", "torchaudio", "numpy"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="gfisrv2_upscale",
    debug=False,
    strip=False,
    upx=False,
    console=False,     # windowed — no console popup
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="gfisrv2_upscale",
)
