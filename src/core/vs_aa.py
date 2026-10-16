"""
vs_aa.py — real nnedi3 / EEDI3 anti-aliasing via VapourSynth (StaxRip plugins).

Uses the VapourSynth runtime bundled with StaxRip (VSPipe.exe + plugins) to run the
classic edge-directed-interpolation AA on a single image:
  • nnedi3  → znedi3 (CPU, weights .bin) or NNEDI3CL (OpenCL GPU)
  • EEDI3   → eedi3m (CPU)

No Python ABI coupling: we generate a `.vpy` script and run it through VSPipe as a
subprocess, writing the AA'd PNG via imwri. If VapourSynth/plugins aren't found, the
caller falls back to the pure-numpy directional AA in line_filters.antialias.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

def _app_root():
    """Repo/app root (parent of src/). Lets the portable build ship a bundled
    VapourSynth under vendor/vapoursynth/ with zero external dependency."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Bundled VapourSynth (portable build) is searched FIRST, then StaxRip, then PATH.
_VENDOR = os.path.join(_app_root(), "vendor", "vapoursynth")
_DEFAULT_ROOTS = [
    _VENDOR,
    os.path.join(_VENDOR, "core"),
    r"C:\Program Files\StaxRip\Apps\FrameServer\VapourSynth",
    r"C:\Program Files (x86)\StaxRip\Apps\FrameServer\VapourSynth",
]
_DEFAULT_PLUGINS = [
    os.path.join(_VENDOR, "plugins"),
    _VENDOR,
    r"C:\Program Files\StaxRip\Apps\Plugins\VS",
    r"C:\Program Files (x86)\StaxRip\Apps\Plugins\VS",
]

_vspipe = None
_plugins = None
_override_vspipe = None
_override_plugins = None


def configure(vspipe: str | None = None, plugins: str | None = None) -> None:
    """Point the AA backend at a specific VapourSynth (e.g. a path the user set in
    Réglages, or a VapourSynth bundled with the portable build). Resets detection."""
    global _override_vspipe, _override_plugins, _vspipe, _plugins
    _override_vspipe = vspipe or None
    _override_plugins = plugins or None
    _vspipe = _plugins = None


def init_from_settings(settings) -> None:
    try:
        configure(settings.get("vs_aa_vspipe", "") or None, settings.get("vs_aa_plugins", "") or None)
    except Exception:
        pass


def _detect():
    global _vspipe, _plugins
    if _vspipe is not None:
        return
    import shutil
    # 1) explicit override (settings / portable bundle)
    cands = []
    if _override_vspipe:
        cands.append(_override_vspipe)
    # 2) env var (portable launcher can set this)
    if os.environ.get("USRS_VSPIPE"):
        cands.append(os.environ["USRS_VSPIPE"])
    # 3) standalone VapourSynth on PATH
    w = shutil.which("vspipe") or shutil.which("VSPipe")
    if w:
        cands.append(w)
    # 4) StaxRip defaults
    cands += [os.path.join(r, "VSPipe.exe") for r in _DEFAULT_ROOTS]
    for p in cands:
        if p and os.path.isfile(p):
            _vspipe = p
            break

    pcands = []
    if _override_plugins:
        pcands.append(_override_plugins)
    if os.environ.get("USRS_VS_PLUGINS"):
        pcands.append(os.environ["USRS_VS_PLUGINS"])
    pcands += _DEFAULT_PLUGINS
    for d in pcands:
        if d and os.path.isdir(d):
            _plugins = d
            break


def available() -> bool:
    _detect()
    return bool(_vspipe and _plugins)


def _plugin(*names):
    """Return the first existing plugin DLL path among `names` (searched recursively)."""
    if not _plugins:
        return None
    for root, _dirs, files in os.walk(_plugins):
        low = {f.lower(): f for f in files}
        for n in names:
            if n.lower() in low:
                return os.path.join(root, low[n.lower()])
    return None


def _imwri_dll():
    return _plugin("imwri.dll", "vsimwri.dll", "libimwri.dll")


def antialias_image(in_path: str, out_path: str, method: str = "nnedi3",
                    device: str = "cpu", strength: float = 1.0, passes: int = 1,
                    log=None) -> tuple[bool, str]:
    """AA `in_path` → `out_path` with the requested method/device. Returns (ok, msg)."""
    _detect()
    if not available():
        return False, "VapourSynth (StaxRip) introuvable."
    imwri = _imwri_dll()
    if not imwri:
        return False, "Plugin imwri introuvable."

    method = (method or "nnedi3").lower()
    gpu = str(device).lower().startswith("gpu")
    if "eedi3" in method:
        dll = _plugin("EEDI3m.dll", "EEDI3.dll")
        ns, fn = "eedi3m", "EEDI3"
    elif gpu:
        dll = _plugin("NNEDI3CL.dll")
        ns, fn = "nnedi3cl", "NNEDI3CL"
    else:
        dll = _plugin("vsznedi3.dll", "libnnedi3.dll")
        ns = "znedi3" if dll and "znedi3" in os.path.basename(dll).lower() else "nnedi3"
        fn = "nnedi3"
    if not dll:
        return False, f"Plugin AA introuvable pour {method}/{device}."

    strength = max(0.0, min(1.0, float(strength)))
    passes = max(1, int(passes))
    vpy = _VPY.format(imwri=imwri.replace("\\", "\\\\"), aadll=dll.replace("\\", "\\\\"),
                      ns=ns, fn=fn, inp=in_path.replace("\\", "\\\\"),
                      outp=out_path.replace("\\", "\\\\"), passes=passes, strength=strength)
    tf = tempfile.NamedTemporaryFile("w", suffix=".vpy", delete=False, encoding="utf-8")
    tf.write(vpy); tf.close()
    try:
        r = subprocess.run([_vspipe, tf.name, os.devnull], capture_output=True, text=True,
                           timeout=120, creationflags=0x08000000 if os.name == "nt" else 0)
        if log and r.stderr.strip():
            log(r.stderr.strip()[-300:])
        if r.returncode != 0:
            return False, (r.stderr or "VSPipe a échoué")[-300:]
        if not os.path.isfile(out_path):
            return False, "Sortie non produite."
        return True, "ok"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.remove(tf.name)
        except Exception:
            pass


# nnedi3/eedi3 antialiasing: double each axis via EDI, resize back with the 0.5px
# field shift, transpose, repeat — the canonical single-rate AA recipe.
_VPY = '''import vapoursynth as vs
core = vs.core
core.std.LoadPlugin(r"{imwri}")
core.std.LoadPlugin(r"{aadll}")
src = core.imwri.Read(r"{inp}")
rgb = core.resize.Bicubic(src, format=vs.RGB24)
work = core.resize.Bicubic(rgb, format=vs.YUV444P16, matrix_s="709")

def edi(c):
    return core.{ns}.{fn}(c, field=1, dh=True)

def aa1(c):
    w, h = c.width, c.height
    c = edi(c)
    c = core.resize.Spline36(c, w, h, src_top=0.5)
    c = core.std.Transpose(c)
    c = edi(c)
    c = core.resize.Spline36(c, h, w, src_top=0.5)
    c = core.std.Transpose(c)
    return c

aa = work
for _ in range({passes}):
    aa = aa1(aa)

if {strength} < 0.999:
    aa = core.std.Merge(work, aa, weight={strength})

out = core.resize.Bicubic(aa, format=vs.RGB24, matrix_in_s="709")
out = core.imwri.Write(out, "PNG", r"{outp}", overwrite=True)
out.set_output()
'''
