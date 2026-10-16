"""
lq_generator.py — GUI-agnostic LQ dataset generation (faithful to the v2.5.8 CTk
`_apply_gen_degradations` / `_process_gen`). Both the Qt and CTk fronts build the same
`opts` dict (keys below) and call these functions; all degradation math lives here.

opts keys: scale (int), method (PIL resample const), plus per-degradation bool flags +
params (see the v2.5.8 _collect_gen_opts). Pipeline: passes (int), prob (float),
prob_per_image (bool), autocrop/crop_sz.
"""
from __future__ import annotations

import io as _io
import os
import random
import re as _re
import shutil as _shutil
import subprocess as _subprocess
import time as _time


def _resize_downscale(img, target_w, target_h, method, dpid_lambda=0.5):
    """Downscale `img` (PIL.Image) to (target_w, target_h). `method` is either a
    normal PIL resample constant, or the sentinel string "dpid" — Detail-Preserving
    Image Downscaling (umzi2/pepedpid, Rust-backed rewrite of "Rapid, Detail-
    Preserving Image Downscaling", Weber et al.) which keeps edges/texture sharper
    than bicubic at the cost of a slightly different (less "soft") look. Optional
    dependency (pepedpid + pepeline) — raises a clear error if missing rather than
    silently falling back, so a misconfigured env doesn't produce a mismatched dataset.
    """
    if method != "dpid":
        return img.resize((target_w, target_h), method)
    try:
        import numpy as np
        from pepedpid import dpid_resize
    except ImportError as e:
        raise ImportError(
            "Downscale method 'DPID' requires the 'pepedpid' and 'pepeline' packages "
            f"(pip install pepedpid pepeline). Missing: {e}"
        ) from e
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    out = dpid_resize(arr, int(target_h), int(target_w), float(dpid_lambda))
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(out, mode="RGB")


def apply_degradations(img, opts):
    """Apply the enabled degradations (respecting per-degradation / per-image probability)
    to a single PIL.Image and return the degraded image."""
    import numpy as np
    from PIL import Image, ImageFilter as _IF, ImageEnhance as _IE, ImageOps as _IO

    prob = opts.get("prob", 1.0)
    prob_per_image = opts.get("prob_per_image", False)
    apply_all = (random.random() < prob) if prob_per_image else None

    def _should(flag):
        if not flag:
            return False
        if prob_per_image:
            return apply_all
        return random.random() < prob

    # ── Basique ──
    if _should(opts.get("sharpen")):
        # sharpen_amt (slider "Intensité", 1.0-3.0) was previously ignored — every run
        # used PIL's fixed ImageFilter.SHARPEN regardless of the slider value.
        amt = float(opts.get("sharpen_amt", 1.5))
        img = img.filter(_IF.UnsharpMask(radius=2, percent=int(amt * 100), threshold=3))
    if _should(opts.get("blur")):
        img = img.filter(_IF.GaussianBlur(radius=opts["blur_sigma"]))
    if _should(opts.get("noise")):
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, opts["noise_sigma"], arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    if _should(opts.get("color_jitter")):
        img = _IE.Brightness(img).enhance(random.uniform(0.9, 1.1))
        img = _IE.Contrast(img).enhance(random.uniform(0.9, 1.1))
        img = _IE.Color(img).enhance(random.uniform(0.9, 1.1))
        hsv = np.array(img.convert("HSV"))
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + random.randint(-13, 13)) % 256   # hue ±5 %
        img = Image.fromarray(hsv, "HSV").convert("RGB")
    if _should(opts.get("jpeg")):
        buf = _io.BytesIO(); img.save(buf, format="JPEG", quality=opts["jpeg_q"]); buf.seek(0)
        img = Image.open(buf).convert("RGB")

    # ── Couleur ──
    if _should(opts.get("posterize")):
        img = _IO.posterize(img, max(1, min(8, opts["posterize_bits"])))
    if _should(opts.get("banding")):
        levels = max(2, min(256, opts["banding_levels"]))
        img = img.quantize(colors=levels, method=Image.Quantize.FASTOCTREE).convert("RGB")
    if _should(opts.get("saturation")):
        img = _IE.Color(img).enhance(opts["sat_factor"])

    try:
        from src.core.otf_preview import (
            apply_chromatic_aberration, apply_disc_blur_pil, apply_vignette_pil,
            apply_wtp_halo, apply_color_levels, apply_quantize_depth_pil,
            apply_vhs, apply_aliasing_pil, apply_interlace_weave_pil,
            apply_interlace_flicker_pil, apply_interlace_blend_pil,
            apply_scanlines_pil, apply_screentone, apply_dithering,
            apply_sinusoidal, apply_subsampling_wtp, apply_pixel_shift,
            apply_film_grain_pil, apply_oversharpening_pil,
            apply_salt_pepper, apply_halation,
        )
        _otf_ok = True
    except Exception:
        _otf_ok = False

    if _otf_ok:
        if _should(opts.get("ca")):
            img = apply_chromatic_aberration(img, shift_range=(opts["ca_shift"], opts["ca_shift"]))
        if _should(opts.get("disc_blur")):
            img = apply_disc_blur_pil(img, radius_range=(opts["disc_blur_r"], opts["disc_blur_r"]))
        if _should(opts.get("vignette")):
            img = apply_vignette_pil(img, strength_range=(opts["vignette_str"], opts["vignette_str"]))
        if _should(opts.get("halo")):
            img = apply_wtp_halo(img, strength_range=(opts["halo_str"], opts["halo_str"]),
                                 radius_range=(opts["halo_r"], opts["halo_r"]))
        if _should(opts.get("color_levels")):
            img = apply_color_levels(img, high_range=(opts["col_high"], opts["col_high"]),
                                     low_range=(opts["col_low"], opts["col_low"]))
        if _should(opts.get("quantize_depth")):
            img = apply_quantize_depth_pil(img, bits_range=(opts["qdepth_bits"], opts["qdepth_bits"]))
        # Vidéo
        if _should(opts.get("chroma_sub")):
            img = apply_subsampling_wtp(img, subsampling_format="4:2:0")
        if _should(opts.get("aliasing")):
            img = apply_aliasing_pil(img, scale_range=(opts["aliasing_str"], opts["aliasing_str"]))
        if _should(opts.get("interlace_weave")):
            img = apply_interlace_weave_pil(img, strength_range=(opts["weave_str"], opts["weave_str"]))
        if _should(opts.get("interlace_flicker")):
            img = apply_interlace_flicker_pil(img, strength_range=(opts["flicker_amp"], opts["flicker_amp"]))
        if _should(opts.get("interlace_blend")):
            img = apply_interlace_blend_pil(img, strength_range=(opts["blend_mix"], opts["blend_mix"]))
        if _should(opts.get("scanlines")):
            img = apply_scanlines_pil(img, spacing_range=(opts["scanlines_period"], opts["scanlines_period"]),
                                      strength_range=(opts["scanlines_dark"], opts["scanlines_dark"]))
        if _should(opts.get("vhs")):
            img = apply_vhs(img, strength_range=(opts["vhs_str"], opts["vhs_str"]))
        if _should(opts.get("screentone")):
            img = apply_screentone(img, dot_size=opts["screen_sz"])
        if _should(opts.get("dithering")):
            img = apply_dithering(img, n_colors=opts["dither_col"])
        if _should(opts.get("sinusoidal")):
            img = apply_sinusoidal(img, shape_range=(opts["sin_period"], opts["sin_period"]),
                                   alpha_range=(opts["sin_alpha"], opts["sin_alpha"]))
        if _should(opts.get("pixel_shift")):
            img = apply_pixel_shift(img, shift_range=(opts["pxsh_amt"], opts["pxsh_amt"]))
        # Avancé
        if _should(opts.get("film_grain")):
            img = apply_film_grain_pil(img, strength_range=(opts["grain_sigma"], opts["grain_sigma"]),
                                       size_range=(opts["grain_size"], opts["grain_size"]))
        if _should(opts.get("oversharp")):
            img = apply_oversharpening_pil(img, strength_range=(opts["oversharp_amt"], opts["oversharp_amt"]))
        if _should(opts.get("salt_pepper")):
            img = apply_salt_pepper(img, amount_range=(opts["sp_amt"], opts["sp_amt"]))
        if _should(opts.get("halation")):
            img = apply_halation(img, strength_range=(opts["hal_str"], opts["hal_str"]))

    # Motion blur — directional line kernel via FFT convolution.
    # NOTE: PIL's ImageFilter.Kernel only accepts (3,3) or (5,5) — it silently/hard
    # raises ValueError for anything else, so this degradation crashed on every
    # non-trivial mb_px (default 7, slider goes to 31). Reuses the FFT-convolution
    # helper from otf_preview.py (same one used for sinc/mixed-kernel blur) so any
    # kernel size works.
    if _should(opts.get("motion_blur")):
        ksize = max(3, int(opts["mb_px"]) | 1)
        try:
            import numpy as _np
            from src.core.otf_preview import apply_kernel_pil
            kernel = _np.zeros((ksize, ksize), dtype=_np.float64)
            kernel[ksize // 2, :] = 1.0 / ksize
            img = apply_kernel_pil(img, kernel)
        except Exception:
            img = img.filter(_IF.BoxBlur(max(1, ksize // 4)))  # never-crash fallback

    # H.264 codec simulation: double JPEG pass
    if _should(opts.get("codec")):
        buf = _io.BytesIO(); img.save(buf, format="JPEG", quality=opts["codec_q1"]); buf.seek(0)
        img = Image.open(buf).convert("RGB")
        buf2 = _io.BytesIO(); img.save(buf2, format="JPEG", quality=opts["codec_q2"]); buf2.seek(0)
        img = Image.open(buf2).convert("RGB")

    return img


def degrade_one(pil_img, opts):
    """Full single-image pipeline for the preview: downscale by `scale` then apply
    `passes` rounds of degradations. Returns the LQ PIL image."""
    from PIL import Image  # noqa: F401
    img = pil_img.copy()
    scale = opts["scale"]
    img = _resize_downscale(img, max(1, img.width // scale), max(1, img.height // scale),
                            opts["method"], opts.get("dpid_lambda", 0.5))
    for _ in range(max(1, opts.get("passes", 1))):
        img = apply_degradations(img, opts)
    return img


def paired_hq_dir(lq):
    """Where the matching HQ images go when the HQ side must change (auto-crop patches, or
    sizes not divisible by the scale): the source HQ folder is never modified."""
    return os.path.normpath(lq).rstrip("\\/") + "_HQ"


def process_folder(hq, lq, opts, progress=None, preview=None, log=None, same_ext=False, only=None):
    """Batch-generate LQ images from an HQ folder. Callbacks:
      progress(frac, done, total) · preview(hq_snap, lq_snap) · log(msg).
    The engines need HQ = LQ x scale exactly. When the HQ side has to change (auto-crop,
    or a size not divisible by the scale), the matching HQ images are written to
    paired_hq_dir(lq) and that folder is the GT folder to train with (logged).
    same_ext: name the LQ exactly like the HQ (video datasets match full file names).
    Returns (processed_count, total)."""
    from PIL import Image
    import shutil
    if os.path.normcase(os.path.abspath(hq)) == os.path.normcase(os.path.abspath(lq)):
        raise ValueError("Les dossiers HQ et LQ sont identiques : les HQ seraient écrasées. / "
                         "HQ and LQ folders are the same: the HQ images would be overwritten.")
    os.makedirs(lq, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = [x for x in os.listdir(hq) if os.path.splitext(x)[1].lower() in exts]
    if only is not None:   # B27: only the frames of this run, not the other scenes of the folder
        only = {os.path.basename(x) for x in only}
        files = [x for x in files if x in only]
    total = len(files)
    if total == 0:
        if log:
            log("Aucune image trouvée. / No image found.")
        return 0, 0
    scale = max(1, int(opts["scale"]))
    passes = max(1, int(opts.get("passes", 1)))
    autocrop = bool(opts.get("autocrop"))
    csz = (max(scale, int(opts.get("crop_sz", 256))) // scale) * scale  # patch = multiple of scale
    # does any HQ need a matching copy? (auto-crop, or a size not divisible by the scale)
    need_pairs = autocrop
    if not need_pairs and scale > 1:
        for fname in files:
            try:
                with Image.open(os.path.join(hq, fname)) as _im:
                    if _im.width % scale or _im.height % scale:
                        need_pairs = True
                        break
            except Exception:
                pass
    pair_dir = paired_hq_dir(lq) if need_pairs else None
    if pair_dir:
        os.makedirs(pair_dir, exist_ok=True)
    live_every = max(1, total // 20)
    done = 0
    for i, fname in enumerate(files):
        try:
            img = Image.open(os.path.join(hq, fname)).convert("RGB")
            changed = False
            if autocrop and img.width >= csz and img.height >= csz:
                x0 = random.randint(0, img.width - csz)
                y0 = random.randint(0, img.height - csz)
                img = img.crop((x0, y0, x0 + csz, y0 + csz)); changed = True
            w2, h2 = (img.width // scale) * scale, (img.height // scale) * scale
            if (w2, h2) != img.size:
                img = img.crop((0, 0, w2, h2)); changed = True
            hq_snap = img.copy()
            if pair_dir:
                dst_hq = os.path.join(pair_dir, fname)
                if changed:
                    img.save(dst_hq)
                else:
                    shutil.copy2(os.path.join(hq, fname), dst_hq)
            lq_img = _resize_downscale(img, max(1, img.width // scale), max(1, img.height // scale),
                                       opts["method"], opts.get("dpid_lambda", 0.5))
            for _ in range(passes):
                lq_img = apply_degradations(lq_img, opts)
            if preview and i % live_every == 0:
                preview(hq_snap, lq_img.copy())
            stem, ext = os.path.splitext(fname)
            if same_ext and ext.lower() in (".jpg", ".jpeg"):
                # same full name as the HQ frame; near-lossless so the JPEG adds no artifact
                lq_img.save(os.path.join(lq, fname), quality=100, subsampling=0)
            elif same_ext:
                lq_img.save(os.path.join(lq, fname))
            else:
                lq_img.save(os.path.join(lq, stem + ".png"))
            done += 1
        except Exception as e:
            if log:
                log(f"  ⚠ {fname}: {e}")
        if progress:
            progress((i + 1) / total, i + 1, total)
    if pair_dir and log:
        log(f"ℹ HQ appariées (recadrées au multiple de l'échelle / patchs) : {pair_dir} — "
            f"à utiliser comme dossier GT. / Matched HQ (cropped to a multiple of the scale / "
            f"patches): {pair_dir} — use it as the GT folder.")
    return done, total


# ── Mode "Entrée Vidéo" ──────────────────────────────────────────────────────
# Extracts frames from a video file (ffmpeg), then reuses process_folder() above
# for the HQ→LQ degradation pass — same pipeline, same opts dict, nothing duplicated.
# Frame naming follows the convention expected by traiNNer/data/paired_video_dataset.py
# (PairedVideoDataset): "<scene_prefix>_XXXXX.png", where scene_prefix = filename
# split on the FIRST "_" — so the prefix itself must never contain an underscore.
# CAVEAT (img_format != "png8") — VERIFIED against the actual traiNNer-redux checkout
# (traiNNer/data/paired_video_dataset.py), not guessed:
#   - PairedVideoDataset.__init__ lists frames with
#     `f.lower().endswith((".png", ".jpg", ".jpeg"))` — .tiff/.tif is NOT in that tuple.
#     So TIFF HQ frames (tiff8/tiff16) are silently invisible to it: the scan finds 0
#     files for that scene, 0 clips get built, and training silently sees nothing for
#     that video with no error raised anywhere. This is why the UI (gen_lq.py /
#     quick_upscale.py) does not offer tiff8/tiff16 as extraction-format choices at all.
#   - PNG16 IS visible (extension is still ".png") and its extra precision is NOT lost
#     for training: confirmed by instantiating a real PairedVideoDataset on a folder of
#     genuinely 16-bit PNG frames and inspecting __getitem__'s output tensor — dtype
#     float32, full [0,1] range, no truncation. PairedVideoDataset reads frames via
#     vipsimfrompath() (pyvips, verified to load these files as real uint16 arrays) and
#     normalizes via tensor2float32()/torch.iinfo(tensor.dtype).max (not a hardcoded
#     255), so the pipeline this app's frames actually feed at training time keeps the
#     full bit depth. LQ frames are always re-saved as 8-bit ".png" by process_folder()
#     regardless of img_format (see img.save(...) a few lines up) — LQ precision is
#     unaffected by this choice either way.
#   - Separately (see EXTRACT_FORMAT_16BIT_WARNING below): THIS APP's own internal
#     consumers of these files (the degradation step that builds the LQ frames,
#     Quick Upscale, the *_runner.py upscale runners, any other `Image.open(...)` in
#     this repo) go through Pillow, which truncates a 16-bit source to 8-bit on load
#     with no warning. That is a real, separate limitation — but it does not apply to
#     PairedVideoDataset (the actual training-time reader), which never uses Pillow.

def sanitize_scene_prefix(name: str) -> str:
    """Turn an arbitrary string (e.g. a video filename) into a safe scene prefix:
    no underscore (PairedVideoDataset splits on the first "_" to get the prefix),
    no path separators / spaces, ASCII letters+digits+hyphen only. Never empty."""
    name = os.path.splitext(str(name))[0] if "." in str(name) else str(name)
    name = _re.sub(r"[^A-Za-z0-9\-]+", "-", name.strip())
    name = name.strip("-")
    return name or "scene"


def resolve_ffmpeg(ffmpeg_path: str | None = None) -> str | None:
    """Locate an ffmpeg executable, in priority order:
      1. an explicit `ffmpeg_path` if given and valid,
      2. a portable binary bundled with the app (bin/ffmpeg.exe or runtimes/ffmpeg/ffmpeg.exe
         relative to the project root — NOT shipped by default, see the LQ Generator's
         video-mode status line / project README for how to add one),
      3. the `imageio-ffmpeg` Python package, if installed (it bundles its own portable
         ffmpeg binary — the simplest zero-config option, add it to requirements.txt to enable),
      4. `ffmpeg` on the system PATH (shutil.which), same convention already used by
         qt_ui/tabs/settings_tab.py's dependency check.
    Returns the resolved path, or None if nothing is found (caller must handle gracefully —
    never raise here, this is a probe used both by the pipeline and by the UI status line)."""
    if ffmpeg_path and os.path.isfile(ffmpeg_path):
        return ffmpeg_path
    try:
        # .../src/core/lq_generator.py -> project root is two levels up from src/
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for _rel in ("bin/ffmpeg.exe", "runtimes/ffmpeg/ffmpeg.exe", "runtimes/ffmpeg/bin/ffmpeg.exe"):
            _p = os.path.join(_root, *_rel.split("/"))
            if os.path.isfile(_p):
                return _p
    except Exception:
        pass
    try:
        import imageio_ffmpeg
        _p = imageio_ffmpeg.get_ffmpeg_exe()
        if _p and os.path.isfile(_p):
            return _p
    except Exception:
        pass
    _sys_ff = _shutil.which("ffmpeg")
    if _sys_ff:
        return _sys_ff
    # 5. imageio-ffmpeg installed in the engine venv by the engine installer
    try:
        import glob as _glob
        try:
            from src.core import engine_paths as _ep
        except ImportError:
            import engine_paths as _ep
        _venv = _ep.shared_venv_dir()
        for _p in sorted(_glob.glob(os.path.join(_venv, "Lib", "site-packages", "imageio_ffmpeg",
                                                 "binaries", "ffmpeg*.exe")), reverse=True):
            if os.path.isfile(_p):
                return _p
    except Exception:
        pass
    return None


def resolve_ffprobe(ffmpeg_path: str | None = None) -> str | None:
    """Locate ffprobe next to whatever resolve_ffmpeg() found (same install, same
    bundling story — portable ffmpeg binaries always ship both). Returns None if
    ffmpeg itself couldn't be resolved or ffprobe isn't next to it; callers fall back
    to a duration-less UI rather than fail extraction over a missing probe tool."""
    exe = resolve_ffmpeg(ffmpeg_path)
    if not exe:
        return None
    base = os.path.dirname(exe)
    name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    cand = os.path.join(base, name)
    if os.path.isfile(cand):
        return cand
    return _shutil.which(name)


def probe_video(video_path: str, ffmpeg_path: str | None = None) -> dict:
    """Return {fps, duration, nb_frames, width, height} for `video_path`, best-effort.
    `nb_frames` is fps*duration rounded (container frame counts are frequently wrong/
    absent for common containers, so this estimate is what the UI shows as a guide for
    picking a frame number — not used for the actual extraction, which lets ffmpeg do
    its own frame counting). Returns an all-zero dict (never raises) if ffprobe is
    unavailable or the file can't be read — the UI just skips the frame-count hint."""
    out = {"fps": 0.0, "duration": 0.0, "nb_frames": 0, "width": 0, "height": 0}
    if not os.path.isfile(video_path):
        return out
    exe = resolve_ffprobe(ffmpeg_path)
    if not exe:
        return _probe_with_ffmpeg(video_path, ffmpeg_path, out)
    cmd = [exe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,width,height:format=duration",
           "-of", "json", video_path]
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = _subprocess.CREATE_NO_WINDOW
    try:
        r = _subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                           encoding="utf-8", errors="replace", **kwargs)
        import json as _json
        data = _json.loads(r.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        rate = str(stream.get("r_frame_rate", "0/1"))
        if "/" in rate:
            num, den = rate.split("/", 1)
            fps = float(num) / float(den) if float(den) else 0.0
        else:
            fps = float(rate or 0)
        duration = float(fmt.get("duration", 0) or 0)
        out["fps"] = fps
        out["duration"] = duration
        out["nb_frames"] = int(round(fps * duration)) if fps and duration else 0
        out["width"] = int(stream.get("width", 0) or 0)
        out["height"] = int(stream.get("height", 0) or 0)
    except Exception:
        pass
    return out


def _probe_with_ffmpeg(video_path: str, ffmpeg_path: str | None, out: dict) -> dict:
    """Fallback when there is no ffprobe (imageio-ffmpeg ships ffmpeg only): read
    duration / fps / size from the stream banner `ffmpeg -i` prints on stderr."""
    import re as _re
    exe = resolve_ffmpeg(ffmpeg_path)
    if not exe:
        return out
    kwargs = {"creationflags": _subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        r = _subprocess.run([exe, "-hide_banner", "-i", video_path], capture_output=True, text=True,
                            timeout=20, encoding="utf-8", errors="replace", **kwargs)
        err = r.stderr or ""
        m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
        if m:
            out["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        v = next((ln for ln in err.splitlines() if "Video:" in ln), "")
        m = _re.search(r"(\d+(?:\.\d+)?)\s*fps", v) or _re.search(r"(\d+(?:\.\d+)?)\s*tbr", v)
        if m:
            out["fps"] = float(m.group(1))
        m = _re.search(r",\s*(\d{2,5})x(\d{2,5})", v)
        if m:
            out["width"], out["height"] = int(m.group(1)), int(m.group(2))
        if out["fps"] and out["duration"]:
            out["nb_frames"] = int(round(out["fps"] * out["duration"]))
    except Exception:
        pass
    return out


def extract_frame_at(video_path: str, out_path: str, frame_number: int,
                      img_format: str = "png8", jpg_quality: int = 95,
                      ffmpeg_path: str | None = None, cancel_check=None) -> str:
    """Extract a single frame (1-based `frame_number`, matching what a human means by
    "frame N") to `out_path`. Uses ffmpeg's select filter rather than a `-ss` seek —
    seeking is duration-estimate-based and can land off-by-one-or-more on VFR/odd-GOP
    sources, whereas `select='eq(n\\,N-1)'` walks the real decoded frame index. Raises
    RuntimeError if ffmpeg is missing or produces nothing.

    `cancel_check`, if given, is polled while ffmpeg runs (Popen + wait loop rather than
    a single blocking subprocess.run, so a cancel can actually interrupt it); on a stop
    request the process is killed and InterruptedError is raised."""
    exe = resolve_ffmpeg(ffmpeg_path)
    if not exe:
        raise RuntimeError(
            "ffmpeg introuvable. Installez ffmpeg (https://ffmpeg.org/) et ajoutez-le au PATH, "
            "installez le paquet Python 'imageio-ffmpeg', ou placez un binaire portable dans "
            "'bin/ffmpeg.exe' (ou 'runtimes/ffmpeg/ffmpeg.exe') à la racine du projet."
        )
    n = max(1, int(frame_number))
    fmt_key = (img_format or "png8").lower()
    ext, pix_fmt = EXTRACT_FORMATS.get(fmt_key, EXTRACT_FORMATS["png8"])
    if not out_path.lower().endswith(ext):
        out_path = os.path.splitext(out_path)[0] + ext
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [exe, "-y", "-i", video_path, "-vf", f"select=eq(n\\,{n - 1})", "-fps_mode", "vfr", "-frames:v", "1"]
    if pix_fmt:
        cmd += ["-pix_fmt", pix_fmt]
    if fmt_key == "jpg":
        cmd += ["-q:v", str(_jpg_quality_to_ffmpeg_qv(jpg_quality))]
    cmd += [out_path]
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = _subprocess.CREATE_NO_WINDOW
    proc = _subprocess.Popen(cmd, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
                             universal_newlines=True, bufsize=1, encoding="utf-8", errors="replace", **kwargs)
    # Must actively drain stdout while polling — cancel_check used to live in a separate
    # poll+sleep loop that never read the pipe, so on a big/slow source ffmpeg's own
    # output (esp. with -stats) filled the OS pipe buffer and blocked writing, deadlocking
    # against this loop's `time.sleep` until the 120s timeout killed it (found via a real
    # extraction on a full 1080p episode — frame 200 alone hit the timeout every time,
    # 2026-09-28 fix). Same read-and-check-per-line pattern as extract_frames(), which
    # never had this problem because it always drains the pipe.
    tail = []
    cancelled = False
    start = _time.monotonic()
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                if len(tail) > 20:
                    tail.pop(0)
            if cancel_check and cancel_check():
                cancelled = True
                proc.terminate()
                break
            if _time.monotonic() - start > 120:
                proc.terminate()
                proc.wait()
                raise RuntimeError("ffmpeg a dépassé le délai de 120s.")
    proc.wait()
    if cancelled:
        raise InterruptedError("Extraction annulée par l'utilisateur.")
    if proc.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(f"ffmpeg a échoué (code {proc.returncode}).\n" + "\n".join(tail[-8:]))
    return out_path


#: Supported extraction formats -> (file extension, ffmpeg -pix_fmt or None for default).
#: PNG8/TIFF8 are 8-bit (unchanged behavior). PNG16/TIFF16 are genuine 16-bit-per-channel
#: outputs at the ffmpeg level (verified: ffprobe reports pix_fmt=rgb48be/rgb48le, and
#: OpenCV's cv2.imread(..., IMREAD_UNCHANGED) reads them back as real uint16 data spanning
#: the full 0-65535 range) — BUT see the WARNING below, this pipeline cannot consume that
#: extra precision downstream.
EXTRACT_FORMATS = {
    "png8":  (".png",  None),        # default, unchanged: no explicit -pix_fmt
    "png16": (".png",  "rgb48be"),   # ffmpeg's PNG encoder requires big-endian rgb48
    "tiff8": (".tiff", "rgb24"),
    "tiff16": (".tiff", "rgb48le"),  # ffmpeg's TIFF encoder accepts little-endian rgb48
    "jpg":   (".jpg",  None),        # quality via -q:v, see extract_frames()
    "bmp":   (".bmp",  None),        # 8-bit only — ffmpeg's BMP encoder has no 16-bit mode
}

#: BIT-DEPTH NOTE (verified by direct test, not assumed): a genuinely 16-bit PNG/TIFF
#: produced by ffmpeg (rgb48be/rgb48le, verified via ffprobe AND pyvips reading it back
#: as real uint16 data, full 0-65535 range) splits into two different outcomes depending
#: on who reads it back:
#:   - traiNNer-redux's PairedVideoDataset (the actual training-time consumer of these
#:     HQ/GT frames) reads via pyvips (vipsimfrompath()) and normalizes with
#:     tensor2float32()/torch.iinfo(...).max — verified end-to-end by instantiating it on
#:     a real 16-bit-frame folder: the __getitem__ tensor comes back float32, full [0,1]
#:     range, no truncation. The extra precision IS preserved for training.
#:   - Every reader INSIDE this app (`Image.open(...).convert("RGB")` in
#:     lq_generator.py's own process_folder/apply_degradations, quick_upscale.py's frame
#:     loaders, dataset_tools.py, otf_preview.py, post_proc_worker.py, every
#:     *_runner.py) uses Pillow, which has no first-class 3-channel 16-bit-per-channel
#:     mode. Confirmed by direct test: `PIL.Image.open(path)` on the same 16-bit file
#:     comes back as mode "RGB", dtype uint8 — Pillow SILENTLY truncates to 8-bit on
#:     load, no error/warning. So this app's own degradation/preview/upscale tools do
#:     lose the extra precision the moment they reopen the file with PIL — that's a real,
#:     separate limitation from the training path above.
EXTRACT_FORMAT_16BIT_WARNING = (
    "Le format 16-bit produit bien des fichiers 16-bit sur le disque (vérifié), et cette "
    "précision est bien préservée jusqu'au tenseur d'entraînement de PairedVideoDataset "
    "(vérifié : lit via pyvips, normalise via torch.iinfo — pas de troncature). En "
    "revanche, Pillow (utilisé partout ailleurs DANS CETTE APPLI : dégradation, "
    "chargement Quick Upscale, tous les runners d'upscale) n'a pas de mode RGB 16-bit et "
    "tronque silencieusement à 8-bit dès qu'il rouvre un tel fichier — donc l'aperçu, "
    "Quick Upscale et les runners internes ne voient que 8 bits même sur un fichier "
    "16-bit."
)


def _jpg_quality_to_ffmpeg_qv(quality):
    """Map a user-facing "qualité 1-100" (100 = best) to ffmpeg's mjpeg -q:v scale,
    which is inverted and runs roughly 2 (best) .. 31 (worst). Clamped to [2, 31]."""
    try:
        q = float(quality)
    except (TypeError, ValueError):
        q = 90.0
    q = max(1.0, min(100.0, q))
    # linear map: 100 -> 2, 1 -> 31
    qv = round(31 - (q - 1.0) * (31 - 2) / 99.0)
    return max(2, min(31, int(qv)))


def extract_frames(video_path, out_dir, prefix, fps=0, ffmpeg_path=None, digits=5,
                    progress=None, log=None, img_format="png8", jpg_quality=95,
                    cancel_check=None):
    """Extract every frame (or every 1/fps seconds if fps > 0) of `video_path` into
    `out_dir` as "<prefix>_%0{digits}d.<ext>", numbered sequentially from 1 (so they sort
    alphabetically — required by PairedVideoDataset). `progress(n)` is called with the
    running frame count as ffmpeg reports it (no total — ffmpeg doesn't know it upfront
    without a slower pre-pass, so this is intentionally indeterminate). Returns the sorted
    list of written frame paths. Raises RuntimeError if ffmpeg is missing or fails.

    `img_format` is one of EXTRACT_FORMATS's keys ("png8" default/unchanged, "png16",
    "tiff8", "tiff16", "jpg"). `jpg_quality` (1-100, 100=best) only applies to "jpg" —
    see EXTRACT_FORMAT_16BIT_WARNING for the 16-bit formats' known downstream limitation.

    `cancel_check`, if given, is polled after each output line; when it returns truthy the
    ffmpeg process is terminated and the frames written so far are returned (not raised —
    a user-requested stop isn't a failure).
    """
    os.makedirs(out_dir, exist_ok=True)
    exe = resolve_ffmpeg(ffmpeg_path)
    if not exe:
        raise RuntimeError(
            "ffmpeg introuvable. Installez ffmpeg (https://ffmpeg.org/) et ajoutez-le au PATH, "
            "installez le paquet Python 'imageio-ffmpeg', ou placez un binaire portable dans "
            "'bin/ffmpeg.exe' (ou 'runtimes/ffmpeg/ffmpeg.exe') à la racine du projet."
        )
    prefix = sanitize_scene_prefix(prefix)
    fmt_key = (img_format or "png8").lower()
    ext, pix_fmt = EXTRACT_FORMATS.get(fmt_key, EXTRACT_FORMATS["png8"])
    pattern = os.path.join(out_dir, f"{prefix}_%0{digits}d{ext}")
    cmd = [exe, "-y", "-i", video_path]
    try:
        fps_f = float(fps or 0)
    except (TypeError, ValueError):
        fps_f = 0.0
    if fps_f > 0:
        cmd += ["-vf", f"fps={fps_f}"]
    if pix_fmt:
        cmd += ["-pix_fmt", pix_fmt]
    if fmt_key == "jpg":
        cmd += ["-q:v", str(_jpg_quality_to_ffmpeg_qv(jpg_quality))]
    cmd += [
        # -fps_mode passthrough (no auto duplicate/drop-frame sync to a target rate):
        # without it, "every frame" extraction on real-world anime/VFR sources
        # (frame timestamps that don't match a constant rate) makes ffmpeg's default
        # sync logic insert duplicate frames for the image2 muxer, which then fails
        # a few dozen frames in with "Error submitting a packet to the muxer: Invalid
        # argument" (dup=2 visible in its own stats right before the crash) — found
        # from a real failed extraction, not a hypothetical (2026-09-28 fix).
        # (-fps_mode, not the old -vsync alias: current ffmpeg builds removed -vsync
        # entirely — "Unrecognized option 'vsync'" — found 2026-10-16 after a winget
        # ffmpeg update; -fps_mode exists since ffmpeg 5.1.)
        "-fps_mode", "passthrough",
        # -progress pipe:1: ffmpeg's machine-readable progress report, newline-terminated
        # and flushed after every update. Without it, the human "-stats" line updates via
        # bare '\r' and gets buffered for a long time when stdout is a pipe (not a TTY) —
        # the log/progress bar/preview looked frozen until the WHOLE extraction finished,
        # even though it was progressing fine underneath (2026-09-28 fix).
        "-progress", "pipe:1",
        "-start_number", "1", pattern,
    ]

    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = _subprocess.CREATE_NO_WINDOW
    proc = _subprocess.Popen(cmd, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
                             universal_newlines=True, bufsize=1, encoding="utf-8", errors="replace", **kwargs)
    frame_re = _re.compile(r"frame=\s*(\d+)")
    last = -1
    tail = []
    cancelled = False
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            if len(tail) > 20:
                tail.pop(0)
            m = frame_re.search(line)
            if m:
                n = int(m.group(1))
                if n != last:
                    last = n
                    if progress:
                        progress(n)
            if cancel_check and cancel_check():
                cancelled = True
                proc.terminate()
                break
    proc.wait()
    if cancelled:
        if log:
            log("⏹ Extraction annulée par l'utilisateur.")
    elif proc.returncode != 0:
        detail = "\n".join(tail[-8:])
        raise RuntimeError(f"ffmpeg a échoué (code {proc.returncode}).\n{detail}")

    files = sorted(f for f in os.listdir(out_dir)
                   if f.startswith(prefix + "_") and f.lower().endswith(ext))
    if log and not files:
        log("⚠ ffmpeg n'a produit aucune frame.")
    return [os.path.join(out_dir, f) for f in files]


def process_video(video_path, hq_dir, lq_dir, opts, prefix=None, fps=0,
                   val_count=0, val_hq_dir=None, val_lq_dir=None,
                   val_position="end", ffmpeg_path=None, digits=5,
                   progress=None, preview=None, log=None,
                   img_format="png8", jpg_quality=95):
    """Full "Entrée Vidéo" pipeline:
      1. extract every frame of `video_path` into `hq_dir` (HQ), named
         "<scene_prefix>_XXXXX.png" (scene_prefix from `prefix` or the video's filename);
      2. if val_count > 0, move `val_count` extracted frames — a single CONTIGUOUS
         chunk (never scattered: PairedVideoDataset needs Val frames to form real
         consecutive clips, not disjoint stills) — out of `hq_dir` into `val_hq_dir`.
         `val_position` picks where that chunk sits in the sequence:
           "end"    — last frames (default, previous behavior; risk: end credits/
                      black frames/static end card on a full episode).
           "start"  — first frames (risk: intro/logo card).
           "random" — a uniformly random contiguous window anywhere in the
                      sequence — avoids always landing on non-representative
                      intro/outro content. Trade-off: unlike "end"/"start" (a clean
                      cut at a sequence boundary), carving a chunk out of the
                      middle leaves a seam in the remaining Train frames — the
                      list PairedVideoDataset builds from `hq_dir`'s directory
                      listing treats the two frames flanking the removed chunk as
                      adjacent, so a handful of Train clips near that one seam
                      will silently splice two temporally-unrelated moments. Given
                      it's one seam out of a whole episode's clips, this is a
                      minor/acceptable cost for avoiding an unrepresentative Val
                      set — but it's a real trade-off, not free.
      3. degrade the remaining Train frames hq_dir -> lq_dir, and the Val frames
         val_hq_dir -> val_lq_dir (if any), by calling process_folder() — the exact
         same degradation pipeline used by the image-folder mode, unchanged.

    `progress(phase, frac, done, total)` is called with phase in
    {"extract", "train", "val"}; frac/total are None during "extract" (indeterminate).
    Returns a summary dict (frame/processed counts + resolved output folders).
    """
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"Fichier vidéo introuvable : {video_path}")
    if not hq_dir or not lq_dir:
        raise ValueError("Dossiers de sortie HQ et LQ requis.")

    scene_prefix = sanitize_scene_prefix(prefix or os.path.basename(video_path))
    try:
        _vc = max(0, int(val_count or 0))
    except (TypeError, ValueError):
        _vc = 0
    # B26: the four folders must be distinct and not nested — HQ = LQ overwrote the HQ
    # frames, a Val folder inside Train got listed as Train.
    _dirs = {"HQ": hq_dir, "LQ": lq_dir}
    if _vc > 0:
        _dirs["Val HQ"] = val_hq_dir or (hq_dir.rstrip("/\\") + "_val")
        _dirs["Val LQ"] = val_lq_dir or (lq_dir.rstrip("/\\") + "_val")
    _abs = {k: os.path.normcase(os.path.abspath(v)) for k, v in _dirs.items()}
    _keys = list(_abs)
    for i, a in enumerate(_keys):
        for b in _keys[i + 1:]:
            pa, pb = _abs[a], _abs[b]
            if pa == pb or pa.startswith(pb + os.sep) or pb.startswith(pa + os.sep):
                raise ValueError(f"Dossiers {a} et {b} identiques ou imbriqués : {_dirs[a]} / {_dirs[b]}")
    # B27: frames of an earlier extraction with the same prefix would be mixed in
    _ext = EXTRACT_FORMATS.get((img_format or "png8").lower(), EXTRACT_FORMATS["png8"])[0]
    if os.path.isdir(hq_dir):
        _old = [f for f in os.listdir(hq_dir) if f.startswith(scene_prefix + "_") and f.lower().endswith(_ext)]
        if _old:
            raise ValueError(f"Le dossier HQ contient déjà {len(_old)} frame(s) « {scene_prefix}_* » "
                             f"d'une extraction précédente. Choisissez un dossier vide ou un autre préfixe.")
    os.makedirs(hq_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)

    try:
        val_count = max(0, int(val_count or 0))
    except (TypeError, ValueError):
        val_count = 0
    if val_count > 0:
        val_hq_dir = val_hq_dir or (hq_dir.rstrip("/\\") + "_val")
        val_lq_dir = val_lq_dir or (lq_dir.rstrip("/\\") + "_val")
        os.makedirs(val_hq_dir, exist_ok=True)
        os.makedirs(val_lq_dir, exist_ok=True)

    if log:
        log(f"🎬 Extraction des frames → {hq_dir} (préfixe « {scene_prefix} »)")

    def _extract_progress(n):
        if progress:
            progress("extract", None, n, None)

    frame_paths = extract_frames(video_path, hq_dir, scene_prefix, fps=fps,
                                  ffmpeg_path=ffmpeg_path, digits=digits,
                                  progress=_extract_progress, log=log,
                                  img_format=img_format, jpg_quality=jpg_quality)
    n_extracted = len(frame_paths)
    if log:
        log(f"✔ {n_extracted} frames extraites.")
    if n_extracted == 0:
        raise RuntimeError("Aucune frame extraite (vidéo vide, illisible, ou échec ffmpeg).")

    n_val_moved = 0
    if val_count > 0:
        # B28: Train keeps at least one clip (PairedVideoDataset needs clip_size
        # consecutive frames; 7 covers the usual clip sizes 3/5/7)
        _min_train = 7
        if n_extracted - val_count < _min_train:
            val_count = max(0, n_extracted - _min_train)
            if log:
                log(f"⚠ Validation réduite à {val_count} frame(s) : le Train garde au moins {_min_train} frames (un clip).")
        if 0 < val_count < _min_train and log:
            log(f"⚠ {val_count} frame(s) de validation : moins qu'un clip de {_min_train} — la validation vidéo peut être vide.")
        _pos = (val_position or "end").lower()
        if val_count <= 0:
            val_files = []
        elif _pos == "start":
            val_files = frame_paths[:val_count]
        elif _pos == "random":
            import random as _random
            _max_start = max(0, n_extracted - val_count)
            _start = _random.randint(0, _max_start)
            val_files = frame_paths[_start:_start + val_count]
            if log:
                log(f"🎲 Position validation aléatoire : frames {_start} à {_start + val_count} / {n_extracted}.")
        else:  # "end" (default)
            val_files = frame_paths[-val_count:]
        for src_p in val_files:
            _shutil.move(src_p, os.path.join(val_hq_dir, os.path.basename(src_p)))
        n_val_moved = len(val_files)
        if log and n_val_moved:
            log(f"↪ {n_val_moved} frames déplacées vers le set de validation ({val_hq_dir}).")

    if log:
        log("🧪 Dégradation HQ→LQ (Train)…")

    def _train_progress(frac, done, total):
        if progress:
            progress("train", frac, done, total)

    _val_names = {os.path.basename(p) for p in (val_files if val_count > 0 else [])}
    n_train, total_train = process_folder(hq_dir, lq_dir, opts,
                                          progress=_train_progress, preview=preview, log=log,
                                          same_ext=True,
                                          only=[p for p in frame_paths if os.path.basename(p) not in _val_names])

    n_val_done, total_val = 0, 0
    if val_count > 0:
        if log:
            log("🧪 Dégradation HQ→LQ (Validation)…")

        def _val_progress(frac, done, total):
            if progress:
                progress("val", frac, done, total)

        n_val_done, total_val = process_folder(val_hq_dir, val_lq_dir, opts,
                                               progress=_val_progress, log=log, same_ext=True)

    return {
        "prefix": scene_prefix,
        "frames_extracted": n_extracted,
        "train_hq": hq_dir, "train_lq": lq_dir,
        "train_processed": n_train, "train_total": total_train,
        "val_hq": val_hq_dir, "val_lq": val_lq_dir,
        "val_moved": n_val_moved, "val_processed": n_val_done, "val_total": total_val,
    }
