"""
persistent_upscale_worker.py — Subprocess worker longue durée pour batch upscale.

Protocole JSON-lines via stdin/stdout :
  <- {"cmd": "init", "model": "<path>", "tile_size": 256, "tile_pad": 32, "use_amp": true}
  -> {"status": "ready", "arch": "<arch>", "scale": 2}
  <- {"cmd": "infer", "input": "<path>", "output": "<path>"}
  -> {"status": "ok", "msg": "..."}   |  {"status": "error", "msg": "..."}
  <- {"cmd": "quit"}
  -> (exit 0)

Le modèle est chargé UNE FOIS à "init" et reste en VRAM pour tout le batch.
Chaque "infer" est ~2-3s plus rapide qu'un subprocess neuf (pas de rechargement).

Lancement :
  <venv_python> persistent_upscale_worker.py
"""
import sys
import os
import json
import traceback
import math
import re

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add traiNNer-redux to path (worker runs in traiNNer venv)
_HOME = os.path.expanduser("~")
try:
    from src.core.engine_paths import base_engine_path as _ebp
except ImportError:  # venv runner: engine_paths.py ships loose next to this file
    from engine_paths import base_engine_path as _ebp
_BASE = _ebp()
TRAINNER_PATH = os.path.join(_BASE, "traiNNer-redux")
if os.path.isdir(TRAINNER_PATH) and TRAINNER_PATH not in sys.path:
    sys.path.insert(0, TRAINNER_PATH)
    os.chdir(TRAINNER_PATH)

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(json.dumps({"status": "fatal", "msg": f"Import error: {e}"}), flush=True)
    sys.exit(1)


# ─── State globals ────────────────────────────────────────────────
_model = None
_scale: int = 1
_arch: str = "unknown"
_device = None
_tile_size: int = 0
_tile_pad: int = 32
_use_amp: bool = False
_last_hw: "tuple | None" = None   # (H, W) of previous frame — detect resolution change


# stdout carries the JSON protocol only: any other print() (an arch constructor, e.g.
# DocRes "Enabling Auto-Coordinate Injection") goes to stderr instead of breaking it.
_PROTO_OUT = sys.stdout
sys.stdout = sys.stderr


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


def _emit(obj: dict) -> None:
    _PROTO_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTO_OUT.flush()


def _load_state_dict(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params_ema", "params_g", "params", "model", "state_dict"):
        if key in ck:
            return ck[key]
    return ck


def _detect_scale(sd: dict) -> int:
    # Pixel-shuffle layers: out_ch = in_ch * scale^2
    for k, v in sd.items():
        if "upsample" in k.lower() or "upconv" in k.lower() or "pixel_shuffle" in k.lower():
            if hasattr(v, "shape") and v.ndim == 4:
                out_ch, in_ch = v.shape[0], v.shape[1]
                if out_ch > in_ch:
                    ratio = out_ch / in_ch
                    s = int(math.sqrt(ratio))
                    if s in (2, 3, 4, 8):
                        return s
    # DySample upsampler (SPANPlus): offset.weight channels = 4*2*scale^2 = 8*scale^2
    if "upsampler.offset.weight" in sd:
        offset_ch = sd["upsampler.offset.weight"].shape[0]
        scale_sq = offset_ch / 8.0
        s = int(math.sqrt(scale_sq))
        if s >= 1 and s * s == int(scale_sq):
            return s
    # SpanC/SpanPP: MetaIGConv buffer stores list of scale values (uint8)
    if "MetaIGConv" in sd:
        try:
            scales = [int(v.item()) for v in sd["MetaIGConv"]]
            if scales:
                s = max(scales)  # default to max scale (real upscale intent)
                if s in (1, 2, 3, 4, 8):
                    return s
        except Exception:
            pass
    # Legacy: check any upsampler conv shape ratio vs conv_first
    for k, v in sd.items():
        if "upsampler" in k and hasattr(v, "shape") and v.ndim == 4:
            c_in = sd.get("conv_first.weight", torch.zeros(1, 1, 1, 1)).shape[1]
            c_out = v.shape[0]
            if c_in > 0:
                ratio = c_out / c_in
                for s in [4, 3, 2]:
                    if abs(ratio - s * s) < 0.5:
                        return s
    return 4


def _tile_inference(model, img_t, tile: int, pad: int, scale: int) -> "torch.Tensor":
    b, c, h, w = img_t.shape
    if tile == 0 or (h <= tile and w <= tile):
        return model(img_t)
    out = torch.zeros(b, c, h * scale, w * scale, device=img_t.device, dtype=img_t.dtype)
    tiles_x = math.ceil(w / tile)
    tiles_y = math.ceil(h / tile)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x1 = max(tx * tile - pad, 0)
            y1 = max(ty * tile - pad, 0)
            x2 = min((tx + 1) * tile + pad, w)
            y2 = min((ty + 1) * tile + pad, h)
            tile_in = img_t[:, :, y1:y2, x1:x2]
            with torch.inference_mode():
                tile_out = model(tile_in)
            ox1 = (tx * tile - x1) * scale
            oy1 = (ty * tile - y1) * scale
            ox2 = ox1 + tile * scale
            oy2 = oy1 + tile * scale
            out_x1 = tx * tile * scale
            out_y1 = ty * tile * scale
            out_x2 = min(out_x1 + tile * scale, w * scale)
            out_y2 = min(out_y1 + tile * scale, h * scale)
            clip_x = out_x2 - out_x1
            clip_y = out_y2 - out_y1
            out[:, :, out_y1:out_y2, out_x1:out_x2] = tile_out[
                :, :, oy1:oy1 + clip_y, ox1:ox1 + clip_x]
    return out


def _tile_inference_dandere(
    model, img_t, prev_img_t, prev_sr_t, tile: int, pad: int, scale: int, threshold: float
) -> "tuple":
    """
    Dandere2x tile-by-tile inference.

    For each tile: compare the PADDED LQ region (exactly what the model sees)
    between current and previous frame.
      - MAE ≤ threshold → tile unchanged → model would produce identical SR
        → copy tile from prev_sr (no seam: same padded input = same output)
      - MAE >  threshold → tile changed → run GPU on this tile only

    Returns: (out_tensor [b,c,H*s,W*s], n_changed int, n_total int)
    """
    b, c, h, w = img_t.shape
    # Initialise with prev_sr — unchanged tiles keep their value automatically
    out = prev_sr_t.clone()
    tiles_x = math.ceil(w / tile)
    tiles_y = math.ceil(h / tile)
    n_changed = 0
    n_total = tiles_x * tiles_y

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x1 = max(tx * tile - pad, 0)
            y1 = max(ty * tile - pad, 0)
            x2 = min((tx + 1) * tile + pad, w)
            y2 = min((ty + 1) * tile + pad, h)

            curr_patch = img_t[:, :, y1:y2, x1:x2]
            prev_patch = prev_img_t[:, :, y1:y2, x1:x2]

            # Compare padded region (same context the model will use)
            if float((curr_patch - prev_patch).abs().mean()) <= threshold:
                # Tile unchanged → prev_sr tile already set from clone
                continue

            n_changed += 1
            with torch.inference_mode():
                tile_out = model(curr_patch)

            ox1 = (tx * tile - x1) * scale
            oy1 = (ty * tile - y1) * scale
            out_x1 = tx * tile * scale
            out_y1 = ty * tile * scale
            out_x2 = min(out_x1 + tile * scale, w * scale)
            out_y2 = min(out_y1 + tile * scale, h * scale)
            clip_x = out_x2 - out_x1
            clip_y = out_y2 - out_y1
            out[:, :, out_y1:out_y2, out_x1:out_x2] = tile_out[
                :, :, oy1:oy1 + clip_y, ox1:ox1 + clip_x]

    return out, n_changed, n_total


def cmd_init(payload: dict) -> None:
    global _model, _scale, _arch, _device, _tile_size, _tile_pad, _use_amp, _last_hw
    _last_hw = None
    model_path = payload["model"]
    _tile_size = int(payload.get("tile_size", 256))
    _tile_pad  = int(payload.get("tile_pad", 32))
    _use_amp   = bool(payload.get("use_amp", False))

    if not os.path.isfile(model_path):
        _emit({"status": "error", "msg": f"Model not found: {model_path}"})
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # traiNNer/NeoSR force cudnn.benchmark=True on import (training speed).
    # For variable-resolution INFERENCE batches this is harmful: every new
    # input shape triggers a full cuDNN autotune (huge transient workspace →
    # VRAM spike + long stall), then fragments the caching allocator and slows
    # the rest of the batch. Disable it so mixed-resolution folders stay stable.
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        sd = _load_state_dict(model_path)
    except Exception as e:
        _emit({"status": "error", "msg": f"Load state_dict failed: {e}"})
        return

    _scale = _detect_scale(sd)
    _scale_hint = int(payload.get("scale_hint", 0))  # user-selected scale (0=auto)

    # 0) FIGSR manual — MUST run before GFISRV2 (figsr also has in_to_dim.weight but
    #    uses gfisr_body_half/_2 + cat_to_dim instead of gfisr_body).
    if "cat_to_dim.weight" in sd and "gfisr_body_half.0.fc1.weight" in sd:
        try:
            from traiNNer.archs.figsr_arch import FIGSR, SampleMods
            _w   = sd.get("in_to_dim.weight")
            _dim = int(_w.shape[0]) if _w is not None else 48
            _nb  = (len([k for k in sd if k.startswith("gfisr_body_half.") and k.endswith(".fc1.weight")])
                    + len([k for k in sd if k.startswith("gfisr_body_half_2.") and k.endswith(".fc1.weight")]))
            _nb  = _nb if _nb > 0 else 24
            _figsr_scale = 1
            _up = "pixelshuffledirect"
            _mid = 32
            _meta = sd.get("upscale.MetaUpsample")
            if _meta is not None:
                try:
                    _figsr_scale = max(1, int(_meta[2].item()))
                    _up  = list(SampleMods.__args__)[int(_meta[1].item())]
                    _mid = int(_meta[5].item())
                except Exception:
                    pass
            if _scale_hint > 0:
                _figsr_scale = _scale_hint
            _model = FIGSR(scale=max(1, _figsr_scale), dim=_dim, n_blocks=_nb,
                           upsampler=_up, mid_dim=_mid).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch  = "FIGSR"
            _scale = max(1, _figsr_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": "figsr-manual"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"FIGSR load failed: {_e}"})
            return

    # 1) GFISRV2 manual — MUST run before spandrel (spandrel misidentifies it as SPAN)
    # Mirror universal_runner.py: no upsampler/mid_dim params, strict=False.
    # Scale MUST be inferred from the checkpoint, NOT from _detect_scale():
    #   - _detect_scale() returns 4 (default) for GFISRV2 because its upscale keys
    #     don't match pixel-shuffle patterns → GFISRV2(scale=4,...) creates wrong upscale
    #   - upscale.0.weight [3, 48, 3, 3] → out_ch=3 = RGB = scale 1
    #   - upscale.MetaUpsample → stores scale at index 2 (DySample upsamplers)
    # Require gfisr_body.0 specifically (not just in_to_dim, which figsr also has).
    if "gfisr_body.0.fc1.weight" in sd:
        try:
            from traiNNer.archs.gfisrv2_arch import GFISRV2
            _w   = sd.get("in_to_dim.weight")
            _dim = int(_w.shape[0]) if _w is not None else 48
            _nb  = len([k for k in sd if k.startswith("gfisr_body.") and k.endswith(".fc1.weight")])
            _nb  = _nb if _nb > 0 else 24

            # ── Proper scale detection for GFISRV2 ──────────────────────
            _gfisrv2_scale = 1  # safe default
            _meta = sd.get("upscale.MetaUpsample")
            _up0  = sd.get("upscale.0.weight")
            if _meta is not None:
                try:
                    _gfisrv2_scale = max(1, int(_meta[2].item()))
                except Exception:
                    _gfisrv2_scale = 4
            elif _up0 is not None and _up0.ndim == 4:
                _out_ch = _up0.shape[0]
                if _out_ch == 3:
                    _gfisrv2_scale = 1          # direct RGB output → 1x model
                else:
                    # pixel-shuffle: out_ch = dim * scale^2
                    _sq = _out_ch / max(1.0, float(_dim))
                    _s  = int(math.sqrt(_sq))
                    if _s >= 1 and _s * _s == int(_sq) and _s in (1, 2, 3, 4, 8):
                        _gfisrv2_scale = _s

            _model = GFISRV2(scale=max(1, _gfisrv2_scale), dim=_dim, n_blocks=_nb).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch  = "GFISRV2"
            _scale = max(1, _gfisrv2_scale)  # CRITICAL: update global — _tile_inference uses it
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": "gfisrv2-manual"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"GFISRV2 load failed: {_e}"})
            return  # ne pas tomber sur SPAN avec des poids incompatibles

    # 2) SpanPP manual — AVANT spandrel : spandrel chargerait SpanPP comme SPAN (scale fixe),
    #    ce qui ignore scale_hint. On intercepte ici pour respecter le scale choisi par l'user.
    if "block_1.c1_r.conv3.eval_conv.weight" in sd and "upsampler.amplitude" in sd:
        try:
            import math as _math
            from traiNNer.archs.spanpp_arch import SpanC
            _meta = sd.get("MetaIGConv")
            _scale_list = tuple(int(v.item()) for v in _meta) if _meta is not None else (1, 2)
            _w  = sd.get("block_1.c1_r.conv3.eval_conv.weight")
            _fc = int(_w.shape[0]) if _w is not None else 48
            # latent_layers: compter les Conv2d dans query_kernel (nb_total - 1 finale)
            _qk_keys = [k for k in sd if k.startswith("upsampler.query_kernel.") and k.endswith(".weight")]
            _latent_layers = max(4, len(_qk_keys) - 1) if _qk_keys else 4
            # implicit_dim depuis query_kernel.0
            _qk0 = sd.get("upsampler.query_kernel.0.weight")
            _implicit_dim = int(_qk0.shape[0]) if _qk0 is not None else 256
            # ig_kernel_size depuis freq shape[0] = fc * k^2
            _freq = sd.get("upsampler.freq")
            if _freq is not None:
                _k2 = _freq.shape[0] / _fc
                _ig_k = max(1, int(round(_math.sqrt(_k2))))
            else:
                _ig_k = 3
            # Prefer user-selected scale if valid, else traiNNer-redux's eval_base_scale
            # default (2) — max (4) made 2x-trained models come out 4x
            _requested = _scale_hint if _scale_hint > 0 else (2 if 2 in _scale_list else min(_scale_list))
            _eval_scale = _requested if _requested in _scale_list else min(_scale_list)
            _model = SpanC(feature_channels=_fc, scale_list=_scale_list,
                           eval_base_scale=_eval_scale,
                           ig_kernel_size=_ig_k,
                           implicit_dim=_implicit_dim,
                           latent_layers=_latent_layers)
            _model.load_state_dict(sd, strict=False)  # MUST be before .eval()
            _model = _model.to(_device).eval()  # eval() after weights loaded → eval_convs correct
            _arch  = "SpanPP"
            _scale = _eval_scale
            _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": "spanpp-manual"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"SpanPP load failed: {_e}"})
            return

    # 2b) SMoSR manual — before spandrel (spandrel misidentifies it)
    if ("short.weight" in sd and "upsampler.MetaUpsample" in sd
            and any(k.startswith("blocks_2.") for k in sd)):
        try:
            import re as _re
            from traiNNer.archs.smosr_arch import SMoSR, SampleMods
            _short = sd.get("short.weight")
            _cscale = max(1, int(round(math.sqrt(_short.shape[0] / 3.0)))) if _short is not None else 2
            _upw = sd.get("upsampler.0.eval_conv.weight")
            _dim = (int(_upw.shape[1]) - 3 * _cscale * _cscale) if _upw is not None else 48
            _idx = set()
            for _k in sd:
                _m = _re.match(r"blocks_2\.(\d+)\.", _k)
                if _m:
                    _idx.add(int(_m.group(1)))
            _nmb = len(_idx) if _idx else 3
            _rep = any(".eval_conv." in _k for _k in sd)
            _up = "pixelshuffledirect"
            _mid = 32
            _meta = sd.get("upsampler.MetaUpsample")
            if _meta is not None:
                try:
                    _up = list(SampleMods.__args__)[int(_meta[1].item())]
                    _mid = int(_meta[5].item())
                except Exception:
                    pass
            _model = SMoSR(dim=_dim, scale=max(1, _cscale), rep=_rep, n_mb=_nmb,
                           upsampler=_up, upsampler_mid_dim=_mid)
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device).eval()
            _arch = "SMoSR"
            _scale = _scale_hint if _scale_hint > 0 else _cscale
            _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": "smosr-manual"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"SMoSR load failed: {_e}"})
            return

    # 2c) SpanF manual — before spandrel
    if ("block_1.c1_r.eval_conv.weight" in sd and "conv_near.weight" in sd
            and "conv_cat.weight" in sd):
        try:
            from traiNNer.archs.spanf_arch import spanf
            _w = sd.get("block_1.c1_r.eval_conv.weight")
            _fc = int(_w.shape[0]) if _w is not None else 32
            _c2 = sd.get("conv_2.eval_conv.weight")
            _ssc = 1
            if _c2 is not None:
                _out = _c2.shape[0]
                _s = int(math.sqrt(_out / 3.0))
                _ssc = _s if _s >= 1 and _s * _s == int(_out / 3.0) else 1
            _scale = _scale_hint if _scale_hint > 0 else _ssc
            _model = spanf(num_in_ch=3, num_out_ch=3, feature_channels=_fc, scale=max(1, _scale))
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device).eval()
            _arch = "SpanF"
            _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": "spanf-manual"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"SpanF load failed: {_e}"})
            return

    # 3) Spandrel (universel — pour tout ce qui n'est pas GFISRV2/SpanPP)
    try:
        if "groups.0.blocks.2.attn.dictionary" in sd:  # FDAT-RD: spandrel would take it for FDAT
            raise RuntimeError("skip spandrel")
        import spandrel
        try:
            import spandrel_extra_arches
            spandrel_extra_arches.install()
        except Exception:
            pass
        model_desc = spandrel.ModelLoader(device=_device).load_from_file(model_path)
        _model = model_desc.model.eval().to(_device)
        _scale = getattr(model_desc, "scale", _scale)
        _arch  = type(_model).__name__
        _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": "spandrel"})
        return
    except Exception:
        pass

    # 3a) HEART / NERVE (traiNNer-redux master) — hyperparams read from the weights
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import arch_infer
        _ai = arch_infer.detect(sd)
        if _ai and _ai not in arch_infer.IMAGE_ARCHS:
            _emit({"status": "error", "msg": f"{_ai.upper()} est un modèle vidéo : utilisez l'entrée Vidéo de Quick Upscale."})
            return
        if _ai:
            _m, _sc, _kw = arch_infer.build(_ai, sd)
            _model = _m.eval().to(_device)
            _scale = _sc
            _arch = _ai.upper()
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"{_ai}-manual dim={_kw.get('dim', _kw.get('embed_dim'))}"})
            return
    except Exception as _e:
        _emit({"status": "error", "msg": f"HEART/NERVE load failed: {_e}"})
        return

    # 3b) RCAN manual — spandrel mis-detects n_feats for scale-1 (deband) RCAN
    # because it reads n_feats from the absent pixelshuffle tail.0 → size mismatch.
    if ("body.0.body.0.body.3.conv_du.0.weight" in sd
            and "tail.1.weight" in sd
            and ("head.0.weight" in sd or "head.1.weight" in sd)):
        try:
            from spandrel.architectures.RCAN.__arch.rcan_arch import RCAN
            from spandrel.util import get_seq_len
            _hi = 0 if "head.0.weight" in sd else 1
            _nf = int(sd[f"head.{_hi}.weight"].shape[0])
            _nc = int(sd["tail.1.weight"].shape[0])
            _ks = int(sd[f"head.{_hi}.weight"].shape[-1])
            _nrg = get_seq_len(sd, "body") - 1
            _nrb = get_seq_len(sd, "body.0.body") - 1
            _red = _nf // int(sd["body.0.body.0.body.3.conv_du.0.weight"].shape[0])
            # Scale from upsampler presence: no tail.0.* keys → 1x (deband, no pixelshuffle).
            # _detect_scale() mis-reports because scale-1 RCAN has no upscale keys.
            _has_up = any(k.startswith("tail.0.") for k in sd)
            _rcan_scale = (_scale if _scale and _scale > 0 else 2) if _has_up else 1
            _model = RCAN(scale=_rcan_scale, n_resgroups=_nrg, n_resblocks=_nrb,
                          n_colors=_nc, rgb_range=255, norm=("sub_mean.weight" in sd),
                          kernel_size=_ks, n_feats=_nf, reduction=_red, res_scale=1,
                          act_mode="relu", unshuffle_mod=(_hi == 1)).eval()
            _model.load_state_dict(sd, strict=True)
            _model = _model.to(_device)
            _scale = _rcan_scale
            _arch = "RCAN"
            _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": "rcan-manual"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"RCAN load failed: {_e}"})
            return

    # 4) SPANPlus manual (DySample ou conv upsampler) — guard: need feats.* keys
    _fc_w = sd.get("feats.0.eval_conv.weight", sd.get("feats.0.sk.weight", None))
    if _fc_w is not None:
        try:
            # strict rebuild from the weights (was strict=False with blocks=[4]: the
            # _s / _st / _sts variants loaded partly and produced garbage silently)
            import arch_infer
            _model, _scale, _kw = arch_infer.build("spanplus", sd)
            _model = _model.eval().to(_device)
            _arch  = "SPANPlus"
            _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": "spanplus-fallback"})
            return
        except Exception:
            pass

    # ── ParagonSR v1 (Phhofm) — conv-first, Magic Kernel Sharp 1x upsampler ──
    # Signature: conv_fuse.weight + magic_upsampler.* prefix keys
    if "conv_fuse.weight" in sd and any(k.startswith("magic_upsampler.") for k in sd):
        try:
            from traiNNer.archs.paragonsr_arch import ParagonSR
            _ci = sd.get("conv_in.weight")
            _num_feat = int(_ci.shape[0]) if _ci is not None else 28
            _body_g = set(); _body_b = set()
            for _k in sd:
                if _k.startswith("body.") and ".blocks." in _k:
                    _pts = _k.split(".")
                    if len(_pts) >= 4:
                        _body_g.add(_pts[1]); _body_b.add(_pts[3])
            _num_groups = max(len(_body_g), 1)
            _num_blocks = max(len(_body_b), 1)
            # Detect ffn_expansion from project_in_g ratio
            _pig = sd.get("body.0.blocks.0.transformer.project_in_g.weight")
            _ffn_exp = round(_pig.shape[0] / _pig.shape[1], 4) if (_pig is not None and _pig.shape[1] > 0) else 1.5
            # The scale is not in the weights (parameter-free Magic Kernel upsampler): the
            # user's choice, else 4x like the single-image runner.
            _psr_scale = _scale_hint if _scale_hint > 0 else 4
            _model = ParagonSR(scale=max(1, _psr_scale), num_feat=_num_feat,
                               num_groups=_num_groups, num_blocks=_num_blocks,
                               ffn_expansion=_ffn_exp).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "ParagonSR"
            _scale = max(1, _psr_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"paragonsr-manual feat={_num_feat} g={_num_groups} b={_num_blocks}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"ParagonSR load failed: {_e}"})
            return

    # ── ParagonSR v2 (Phhofm) — dual-path architecture ───────────────────────
    # Signature: detail_gain + conv_in/body at root (no "base." prefix — that's
    # the MagicKernelSharp2021Upsample submodule, not the main path).
    if "detail_gain" in sd and "conv_in.weight" in sd and any(k.startswith("body.") for k in sd):
        try:
            from traiNNer.archs.paragonsr2_arch import ParagonSR2
            import re as _re2
            _ci2 = sd.get("conv_in.weight")
            _nf2 = int(_ci2.shape[0]) if _ci2 is not None else 64
            _keys2 = list(sd.keys())
            _keys2_str = " ".join(_keys2)
            # Detect variant from block signature keys first (photo/pro both
            # default to num_feat=64, so the threshold alone can't tell them apart).
            # NOTE: check gated_conv (ultimate) BEFORE token_ca (pro) — ultimate blocks
            # also carry a token_ca submodule in their staged curriculum, so checking
            # token_ca first would misclassify ultimate checkpoints as pro.
            if ".gated_conv." in _keys2_str:
                _var2 = "ultimate"
            elif ".token_ca." in _keys2_str or ".channel_attn." in _keys2_str:
                _var2 = "pro"
            elif ".proj." in _keys2_str and ".gate." in _keys2_str:
                _var2 = "stream"
            else:
                _var2 = "realtime" if _nf2 <= 16 else "photo"
            _grp2, _blk2 = set(), set()
            for _k in _keys2:
                _m = _re2.match(r"body\.(\d+)\.blocks\.(\d+)\.", _k)
                if _m:
                    _grp2.add(int(_m.group(1)))
                    if int(_m.group(1)) == 0:
                        _blk2.add(int(_m.group(2)))
            _ng2 = len(_grp2) if _grp2 else 4
            _nb2 = len(_blk2) if _blk2 else 4
            # up.0 = conv num_feat -> num_feat * scale^2 (pixel shuffle): the scale is in the
            # weights (it used to default to 1, so 2x/4x models ran at 1x).
            _up2 = sd.get("up.0.weight")
            _s2 = 0
            if _up2 is not None and _up2.shape[1] > 0:
                _r2 = int(round((_up2.shape[0] / _up2.shape[1]) ** 0.5))
                _s2 = _r2 if _r2 >= 1 and _r2 * _r2 * _up2.shape[1] == _up2.shape[0] else 0
            _psr2_scale = _scale_hint if _scale_hint > 0 else (_s2 or 4)
            _model = ParagonSR2(scale=max(1, _psr2_scale), num_feat=_nf2,
                                 num_groups=_ng2, num_blocks=_nb2, variant=_var2).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "ParagonSR2"
            _scale = max(1, _psr2_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"paragonsr2-manual feat={_nf2} var={_var2} g={_ng2} b={_nb2}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"ParagonSR2 load failed: {_e}"})
            return

    # ── MambaIR (ECCV2024) — VSSBlock's SS2D signature ───────────────────────
    if any(".self_attention.x_proj_weight" in k for k in sd):
        try:
            import re as _re
            from traiNNer.archs.mambair_arch import MambaIR
            _cf = sd.get("conv_first.weight")
            _embed_dim = int(_cf.shape[0]) if _cf is not None else 96
            _grp = {}
            for _k in sd:
                _mm = _re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", _k)
                if _mm:
                    _i, _j = int(_mm.group(1)), int(_mm.group(2))
                    _grp.setdefault(_i, set()).add(_j)
            _depths = tuple(len(_grp[i]) for i in sorted(_grp)) if _grp else (6, 6, 6, 6)
            _alogs = sd.get("layers.0.residual_group.blocks.0.self_attention.A_logs")
            _d_state = int(_alogs.shape[-1]) if _alogs is not None else 16
            _d_inner = int(_alogs.shape[0] // 4) if _alogs is not None else _embed_dim * 2
            _mlp_ratio = round(_d_inner / max(1, _embed_dim), 4)
            _resi = "1conv" if "layers.0.conv.weight" in sd else "3conv"
            if "conv_up1.weight" in sd:
                _up = "nearest+conv"; _up_scale = 4
            elif "conv_before_upsample.0.weight" in sd:
                _up = "pixelshuffle"
                _stages = sorted({int(m.group(1)) for k in sd
                                  if (m := _re.match(r"^upsample\.(\d+)\.weight$", k))})
                if len(_stages) == 1:
                    _outc = sd[f"upsample.{_stages[0]}.weight"].shape[0]
                    _up_scale = 3 if _outc == 9 * 64 else 2
                else:
                    _up_scale = 2 ** max(1, len(_stages))
            elif "upsample.0.weight" in sd:
                _up = "pixelshuffledirect"
                _outc = sd["upsample.0.weight"].shape[0]
                _s = round(math.sqrt(_outc / 3.0))
                _up_scale = _s if _s >= 1 and _s * _s * 3 == _outc else 4
            else:
                _up = ""; _up_scale = 1
            _mamba_scale = _scale_hint if _scale_hint > 0 else _up_scale
            _model = MambaIR(embed_dim=_embed_dim, depths=_depths, d_state=_d_state,
                             mlp_ratio=_mlp_ratio, upscale=max(1, _mamba_scale),
                             upsampler=_up, resi_connection=_resi)
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device).eval()
            _arch = "MambaIR"
            _scale = max(1, _mamba_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"mambair-manual embed_dim={_embed_dim} depths={_depths}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"MambaIR load failed: {_e}"})
            return

    # ── MambaIRv2 (CVPR2025) — ASSM/AttentiveLayer signature ─────────────────
    if (any(".embeddingA.weight" in k for k in sd)
            and any(".assm.selectiveScan.x_proj_weight" in k for k in sd)):
        try:
            import re as _re
            from traiNNer.archs.mambairv2_arch import MambaIRv2
            _cf = sd.get("conv_first.weight")
            _embed_dim = int(_cf.shape[0]) if _cf is not None else 48
            _grp = {}
            for _k in sd:
                _mm = _re.match(r"^layers\.(\d+)\.residual_group\.layers\.(\d+)\.", _k)
                if _mm:
                    _i, _j = int(_mm.group(1)), int(_mm.group(2))
                    _grp.setdefault(_i, set()).add(_j)
            _depths = tuple(len(_grp[i]) for i in sorted(_grp)) if _grp else (6, 6, 6, 6)
            _alogs = sd.get("layers.0.residual_group.layers.0.assm.selectiveScan.A_logs")
            _d_state = int(_alogs.shape[-1]) if _alogs is not None else 8
            _inproj = sd.get("layers.0.residual_group.layers.0.assm.in_proj.0.weight")
            _mlp_ratio = round(int(_inproj.shape[0]) / max(1, _embed_dim), 4) if _inproj is not None else 1.5
            _embB = sd.get("layers.0.residual_group.layers.0.assm.embeddingB.weight")
            _num_tokens = int(_embB.shape[0]) if _embB is not None else 64
            _inner_rank = int(_embB.shape[1]) if _embB is not None else 32
            _rpbt = sd.get("layers.0.residual_group.layers.0.win_mhsa.relative_position_bias_table")
            if _rpbt is not None:
                _num_heads = int(_rpbt.shape[-1])
                _window_size = max(2, round((math.sqrt(_rpbt.shape[0]) + 1) / 2))
            else:
                _num_heads, _window_size = 4, 16
            _dw = sd.get("layers.0.residual_group.layers.0.convffn1.dwconv.depthwise_conv.0.weight")
            _convffn_ks = int(_dw.shape[-1]) if _dw is not None else 5
            _resi = "1conv" if "layers.0.conv.weight" in sd else "3conv"
            if "conv_up1.weight" in sd:
                _up = "nearest+conv"; _up_scale = 4
            elif "conv_before_upsample.0.weight" in sd:
                _up = "pixelshuffle"
                _stages = sorted({int(m.group(1)) for k in sd
                                  if (m := _re.match(r"^upsample\.(\d+)\.weight$", k))})
                if len(_stages) == 1:
                    _outc = sd[f"upsample.{_stages[0]}.weight"].shape[0]
                    _up_scale = 3 if _outc == 9 * 64 else 2
                else:
                    _up_scale = 2 ** max(1, len(_stages))
            elif "upsample.0.weight" in sd:
                _up = "pixelshuffledirect"
                _outc = sd["upsample.0.weight"].shape[0]
                _s = round(math.sqrt(_outc / 3.0))
                _up_scale = _s if _s >= 1 and _s * _s * 3 == _outc else 4
            else:
                _up = ""; _up_scale = 1
            _mamba_scale = _scale_hint if _scale_hint > 0 else _up_scale
            _model = MambaIRv2(embed_dim=_embed_dim, depths=_depths,
                               num_heads=(_num_heads,) * len(_depths),
                               d_state=_d_state, mlp_ratio=_mlp_ratio, window_size=_window_size,
                               inner_rank=_inner_rank, num_tokens=_num_tokens,
                               convffn_kernel_size=_convffn_ks, upscale=max(1, _mamba_scale),
                               upsampler=_up, resi_connection=_resi)
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device).eval()
            _arch = "MambaIRv2"
            _scale = max(1, _mamba_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"mambairv2-manual embed_dim={_embed_dim} depths={_depths}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"MambaIRv2 load failed: {_e}"})
            return

    # ── ArtCNN ─────────────────────────────────────────────────────────────
    if "depth_to_space.upscale.0.weight" in sd and "res_block.0.conv.0.0.weight" in sd:
        try:
            import re as _re
            from traiNNer.archs.artcnn_arch import ArtCNN
            _w0 = sd.get("conv0.weight")
            _filters = int(_w0.shape[0]) if _w0 is not None else 96
            _in_ch = int(_w0.shape[1]) if _w0 is not None else 3
            _ks = int(_w0.shape[-1]) if _w0 is not None else 3
            _nb = len({int(m.group(1)) for k in sd
                       if (m := _re.match(r"^res_block\.(\d+)\.conv\.0\.0\.weight$", k))})
            _nb = _nb if _nb > 0 else 16
            _up_w = sd.get("depth_to_space.upscale.0.weight")
            if _up_w is not None and _in_ch > 0:
                _sq = _up_w.shape[0] / _in_ch
                _s = round(math.sqrt(_sq))
                _det_scale = _s if _s >= 1 and _s * _s == round(_sq) else 4
            else:
                _det_scale = 4
            _art_scale = _scale_hint if _scale_hint > 0 else _det_scale
            _model = ArtCNN(in_ch=_in_ch, scale=max(1, _art_scale), filters=_filters,
                            n_block=_nb, kernel_size=_ks).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "ArtCNN"
            _scale = max(1, _art_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"artcnn-manual filters={_filters} n_block={_nb}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"ArtCNN load failed: {_e}"})
            return

    # ── LKFMixer ───────────────────────────────────────────────────────────
    if any(k.startswith("layers.layers.") and ".FDB.c1_d.weight" in k for k in sd):
        try:
            import re as _re
            from traiNNer.archs.lkfmixer_arch import LKFMixer
            _w0 = sd.get("conv_first.weight")
            _channels = int(_w0.shape[0]) if _w0 is not None else 40
            _in_channels = int(_w0.shape[1]) if _w0 is not None else 3
            _nblk = len({int(m.group(1)) for k in sd
                         if (m := _re.match(r"^layers\.layers\.(\d+)\.FDB\.c1_d\.weight$", k))})
            _nblk = _nblk if _nblk > 0 else 6
            _lk_w = sd.get("layers.layers.0.FDB.c1_r.PLKB.DWConv_Kx1.weight")
            _large_kernel = int(_lk_w.shape[2]) if _lk_w is not None else 31
            _split_ch = int(_lk_w.shape[0]) if _lk_w is not None else 10
            _split_factor = round(_split_ch / max(1, _channels), 4)
            _up_w = sd.get("upsampler.upsampleOneStep.0.weight")
            _out_channels = 3
            if _up_w is not None:
                _sq = _up_w.shape[0] / _out_channels
                _s = round(math.sqrt(_sq))
                _det_scale = _s if _s >= 1 and _s * _s == round(_sq) else 4
            else:
                _det_scale = 4
            _lkf_scale = _scale_hint if _scale_hint > 0 else _det_scale
            _model = LKFMixer(in_channels=_in_channels, channels=_channels, out_channels=_out_channels,
                              upscale=max(1, _lkf_scale), num_block=_nblk, large_kernel=_large_kernel,
                              split_factor=_split_factor).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "LKFMixer"
            _scale = max(1, _lkf_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"lkfmixer-manual channels={_channels} num_block={_nblk}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"LKFMixer load failed: {_e}"})
            return

    # ── RTMoSR ─────────────────────────────────────────────────────────────
    if "to_img.0.conv2.weight" in sd and any(k.startswith("body.") and ".fc1." in k for k in sd) \
            and any(".conv.0.poll.1." in k for k in sd):
        try:
            import re as _re
            from traiNNer.archs.rtmosr_arch import RTMoSR
            if "to_feat.1.conv2.weight" in sd:
                _feat_w = sd["to_feat.1.conv2.weight"]
                _unshuffle = max(1, round(math.sqrt(_feat_w.shape[1] / 3.0)))
            else:
                _feat_w = sd.get("to_feat.conv2.weight")
                _unshuffle = 0
            _dim = int(_feat_w.shape[0]) if _feat_w is not None else 32
            _img_w = sd.get("to_img.0.conv2.weight")
            _scale_internal = round(math.sqrt(_img_w.shape[0] / 3.0)) if _img_w is not None else 2
            _det_scale = (4 // _unshuffle) if _unshuffle > 0 else _scale_internal
            _rtm_scale = _scale_hint if _scale_hint > 0 else max(1, _det_scale)
            _nblocks = len({int(m.group(1)) for k in sd
                            if (m := _re.match(r"^body\.(\d+)\.norm\.scale$", k))})
            _nblocks = _nblocks if _nblocks > 0 else 2
            _fc1_w = sd.get("body.0.fc1.conv2.weight")
            _ffn_exp = round((_fc1_w.shape[0] / 2.0) / _dim, 4) if (_fc1_w is not None and _dim > 0) else 2.0
            _dccm = "body.0.fc2.conv2.weight" in sd
            _se = "body.0.conv.2.squeezing.0.weight" in sd
            _model = RTMoSR(scale=max(1, _rtm_scale), dim=_dim, ffn_expansion=_ffn_exp,
                            n_blocks=_nblocks, unshuffle_mod=(_unshuffle > 0), dccm=_dccm, se=_se).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "RTMoSR"
            _scale = max(1, _rtm_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"rtmosr-manual dim={_dim} n_blocks={_nblocks}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"RTMoSR load failed: {_e}"})
            return

    # ── MoSRv2 ─────────────────────────────────────────────────────────────
    if "to_img.MetaUpsample" in sd and any(k.startswith("gblocks.") for k in sd):
        try:
            import re as _re
            from traiNNer.archs.mosrv2_arch import MoSRv2
            from traiNNer.archs.arch_util import SampleMods3
            if "gblocks.0.weight" in sd:
                _in_w = sd["gblocks.0.weight"]
                _unshuffle = 0
            else:
                _in_w = sd.get("gblocks.1.weight")
                _unshuffle = max(1, round(math.sqrt(_in_w.shape[1] / 3.0))) if _in_w is not None else 0
            _dim = int(_in_w.shape[0]) if _in_w is not None else 64
            _nblock = len({int(m.group(1)) for k in sd
                           if (m := _re.match(r"^gblocks\.(\d+)\.gamma$", k))})
            _nblock = _nblock if _nblock > 0 else 24
            _norm_key = next((k for k in sd if _re.match(r"^gblocks\.\d+\.norm\.(scale|weight)$", k)), None)
            _rms_norm = bool(_norm_key and _norm_key.endswith(".scale"))
            _fc1_w = None
            for k, v in sd.items():
                if _re.match(r"^gblocks\.\d+\.fc1\.weight$", k):
                    _fc1_w = v
                    break
            _exp_ratio = round((_fc1_w.shape[0] / 2.0) / _dim, 4) if (_fc1_w is not None and _dim > 0) else 1.5
            _meta = sd.get("to_img.MetaUpsample")
            _mid_dim = 32
            _upsampler = "pixelshuffledirect"
            _scale_internal = 4
            if _meta is not None:
                try:
                    _upsampler = list(SampleMods3.__args__)[int(_meta[1].item())]
                    _scale_internal = max(1, int(_meta[2].item()))
                    _mid_dim = int(_meta[5].item())
                except Exception:
                    pass
            _det_scale = (4 // _unshuffle) if _unshuffle > 0 else _scale_internal
            _mosr_scale = _scale_hint if _scale_hint > 0 else max(1, _det_scale)
            _model = MoSRv2(in_ch=3, scale=max(1, _mosr_scale), n_block=_nblock, dim=_dim,
                            upsampler=_upsampler, expansion_ratio=_exp_ratio, mid_dim=_mid_dim,
                            unshuffle_mod=(_unshuffle > 0), rms_norm=_rms_norm).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "MoSRv2"
            _scale = max(1, _mosr_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"mosrv2-manual dim={_dim} n_block={_nblock}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"MoSRv2 load failed: {_e}"})
            return

    # ── Sebica ─────────────────────────────────────────────────────────────
    if "tail.0.weight" in sd and any(".channel_attention_forward." in k for k in sd):
        try:
            from traiNNer.archs.sebica_arch import Sebica
            _head_w = sd.get("head.0.weight")
            _n_val = int(_head_w.shape[0]) if _head_w is not None else 16
            _tail_w = sd.get("tail.0.weight")
            _det_scale = round(math.sqrt(_tail_w.shape[0] / 3.0)) if _tail_w is not None else 4
            _seb_scale = _scale_hint if _scale_hint > 0 else max(1, _det_scale)
            _model = Sebica(scale=max(1, _seb_scale), N=_n_val).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "Sebica"
            _scale = max(1, _seb_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": f"sebica-manual N={_n_val}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"Sebica load failed: {_e}"})
            return

    # ── LMLT ───────────────────────────────────────────────────────────────
    if ("to_feat.weight" in sd
            and any(re.match(r"^feats\.\d+\.norm1\.weight$", k) for k in sd)
            and any(".lhsb." in k for k in sd)):
        try:
            import re as _re
            from traiNNer.archs.lmlt_arch import LMLT
            _feat_w = sd.get("to_feat.weight")
            _dim = int(_feat_w.shape[0]) if _feat_w is not None else 60
            _nblocks = len({int(m.group(1)) for k in sd
                            if (m := _re.match(r"^feats\.(\d+)\.norm1\.weight$", k))})
            _nblocks = _nblocks if _nblocks > 0 else 8
            _ccm_w = sd.get("feats.0.ccm.ccm.0.weight")
            _ffn_scale = round(_ccm_w.shape[0] / _dim, 4) if (_ccm_w is not None and _dim > 0) else 2.0
            _img_w = sd.get("to_img.0.weight")
            _det_scale = round(math.sqrt(_img_w.shape[0] / 3.0)) if _img_w is not None else 4
            _lmlt_scale = _scale_hint if _scale_hint > 0 else max(1, _det_scale)
            _model = LMLT(dim=_dim, n_blocks=_nblocks, ffn_scale=_ffn_scale, scale=max(1, _lmlt_scale)).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "LMLT"
            _scale = max(1, _lmlt_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"lmlt-manual dim={_dim} n_blocks={_nblocks}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"LMLT load failed: {_e}"})
            return

    # ── MAN ────────────────────────────────────────────────────────────────
    if any(k.startswith("body.") and ".body_t." in k for k in sd) and "tail.0.weight" in sd \
            and "sub_mean.weight" in sd:
        try:
            import re as _re
            from traiNNer.archs.man_arch import MAN
            _head_w = sd.get("head.weight")
            _n_feats = int(_head_w.shape[0]) if _head_w is not None else 180
            _nrg = len({int(m.group(1)) for k in sd
                        if (m := _re.match(r"^body\.(\d+)\.body_t\.", k))})
            _nrg = _nrg if _nrg > 0 else 1
            _nrb = len({int(m.group(1)) for k in sd
                        if (m := _re.match(r"^body\.0\.body\.(\d+)\.", k))})
            _nrb = _nrb if _nrb > 0 else 36
            _tail_w = sd.get("tail.0.weight")
            _det_scale = round(math.sqrt(_tail_w.shape[0] / 3.0)) if _tail_w is not None else 2
            _man_scale = _scale_hint if _scale_hint > 0 else max(1, _det_scale)
            _model = MAN(n_resblocks=_nrb, n_resgroups=_nrg, n_feats=_n_feats,
                        scale=max(1, _man_scale)).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "MAN"
            _scale = max(1, _man_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"man-manual n_feats={_n_feats} nrg={_nrg} nrb={_nrb}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"MAN load failed: {_e}"})
            return

    # ── EIMN ───────────────────────────────────────────────────────────────
    if (any(k.startswith("block1.") and ".mlp.DFFM." in k for k in sd)
            and any(re.match(r"^norm\d+\.weight$", k) for k in sd)):
        try:
            import re as _re
            from traiNNer.archs.eimn_arch import EIMN
            _head_w = sd.get("head.0.weight")
            _embed_dims = int(_head_w.shape[0]) if _head_w is not None else 64
            _num_stages = len({int(m.group(1)) for k in sd
                               if (m := _re.match(r"^norm(\d+)\.weight$", k))})
            _num_stages = _num_stages if _num_stages > 0 else 16
            _depths = len({int(m.group(1)) for k in sd
                           if (m := _re.match(r"^block1\.(\d+)\.", k))})
            _depths = _depths if _depths > 0 else 1
            _mlp_w = sd.get("block1.0.mlp.linear_in.weight")
            if _mlp_w is not None and _embed_dims > 0:
                _hidden = int(_mlp_w.shape[0] // 2)
                _mlp_ratios = (_hidden + 0.5) / _embed_dims  # +0.5 epsilon: int(dim*ratio) truncates safely back to _hidden
            else:
                _mlp_ratios = 2.66
            _tail_w = sd.get("tail.0.weight")
            _det_scale = round(math.sqrt(_tail_w.shape[0] / 3.0)) if _tail_w is not None else 2
            _eimn_scale = _scale_hint if _scale_hint > 0 else max(1, _det_scale)
            _model = EIMN(embed_dims=_embed_dims, scale=max(1, _eimn_scale), depths=_depths,
                         mlp_ratios=_mlp_ratios, num_stages=_num_stages).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "EIMN"
            _scale = max(1, _eimn_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"eimn-manual embed_dims={_embed_dims} num_stages={_num_stages}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"EIMN load failed: {_e}"})
            return

    # ── ELAN ───────────────────────────────────────────────────────────────
    if "head.0.weight" in sd and "tail.0.weight" in sd \
            and any(".modules_lfe.lfe_0." in k for k in sd) \
            and any(".modules_gmsa.gmsa_0." in k for k in sd):
        try:
            import re as _re
            from traiNNer.archs.elan_arch import ELAN
            _head_w = sd.get("head.0.weight")
            _c_elan = int(_head_w.shape[0]) if _head_w is not None else 180
            _n_body = len({int(m.group(1)) for k in sd
                          if (m := _re.match(r"^body\.(\d+)\.", k))})
            _n_body = _n_body if _n_body > 0 else 36
            _n_share = len({int(m.group(1)) for k in sd
                            if (m := _re.match(r"^body\.0\.modules_lfe\.lfe_(\d+)\.", k))})
            _n_share = (_n_share - 1) if _n_share > 0 else 0
            _m_elan = _n_body * (1 + _n_share)
            _tail_w = sd.get("tail.0.weight")
            _det_scale = round(math.sqrt(_tail_w.shape[0] / 3.0)) if _tail_w is not None else 4
            _elan_scale = _scale_hint if _scale_hint > 0 else max(1, _det_scale)
            _model = ELAN(scale=max(1, _elan_scale), m_elan=_m_elan, c_elan=_c_elan, n_share=_n_share).eval()
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device)
            _arch = "ELAN"
            _scale = max(1, _elan_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"elan-manual c_elan={_c_elan} m_elan={_m_elan}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"ELAN load failed: {_e}"})
            return

    # ── GateRV3 — SISR span branch + gated U-Net (in_to_dim/sisr_cat_conv) ───
    if "in_to_dim.weight" in sd and "sisr_cat_conv.weight" in sd:
        try:
            import re as _re
            from traiNNer.archs.gaterv3_arch import GateRV3
            from traiNNer.archs.arch_util import SampleMods
            _idm = sd.get("in_to_dim.weight")
            _dim = int(_idm.shape[0]) if _idm is not None else 32
            _in_ch = int(_idm.shape[1]) if _idm is not None else 3
            _enc = {}
            for _k in sd:
                _mm = _re.match(r"^gater_encode\.(\d+)\.gated\.(\d+)\.", _k)
                if _mm:
                    _enc.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
            _enc_blocks = tuple(len(_enc[i]) for i in sorted(_enc)) if _enc else (2, 2, 4, 6)
            _dec = {}
            for _k in sd:
                _mm = _re.match(r"^decode\.(\d+)\.gated\.(\d+)\.", _k)
                if _mm:
                    _dec.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
            _dec_blocks = tuple(len(_dec[i]) for i in sorted(_dec)) if _dec else (2, 2, 2, 2)
            _lat_idx = {int(m.group(1)) for k in sd if (m := _re.match(r"^latent\.(\d+)\.", k))}
            _num_latent = len(_lat_idx) if _lat_idx else 8
            _sisr_idx = {int(m.group(1)) for k in sd if (m := _re.match(r"^span_n_b\.(\d+)\.", k))}
            _sisr_blocks = len(_sisr_idx) if _sisr_idx else 4
            _attention = any(".token_mix.qkv.weight" in k for k in sd)
            _up = "pixelshuffle"; _up_mid_dim = 48; _det_scale = 2
            _meta = sd.get("dim_to_in.MetaUpsample")
            if _meta is not None:
                try:
                    _up = list(SampleMods.__args__)[int(_meta[1].item())]
                    _det_scale = max(1, int(_meta[2].item()))
                    _up_mid_dim = int(_meta[5].item())
                except Exception:
                    pass
            elif "dim_to_in.weight" in sd:
                _det_scale = 1
            _gaterv3_scale = _scale_hint if _scale_hint > 0 else _det_scale
            _model = GateRV3(in_ch=_in_ch, dim=_dim, enc_blocks=_enc_blocks, dec_blocks=_dec_blocks,
                             num_latent=_num_latent, scale=max(1, _gaterv3_scale), upsample=_up,
                             upsample_mid_dim=_up_mid_dim, attention=_attention,
                             sisr_blocks=_sisr_blocks)
            _model.load_state_dict(sd, strict=True)
            _model = _model.to(_device).eval()
            _arch = "GateRV3"
            _scale = max(1, _gaterv3_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"gaterv3-manual dim={_dim} enc={_enc_blocks} dec={_dec_blocks}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"GateRV3 load failed: {_e}"})
            return

    # ── HiT-SNG — NGram context window partition (unique to SNG variant) ─────
    if any(".ngram_window_partition.ngram_context.unigram_embed.weight" in k for k in sd):
        try:
            import re as _re
            from traiNNer.archs.hit_sng_arch import HiT_SNG as _HiTCls
            _hit_arch_name = "HiT-SNG"
            _cf = sd.get("conv_first.weight")
            _embed_dim = int(_cf.shape[0]) if _cf is not None else 60
            _in_chans = int(_cf.shape[1]) if _cf is not None else 3
            _grp = {}
            for _k in sd:
                _mm = _re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", _k)
                if _mm:
                    _grp.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
            _depths = tuple(len(_grp[i]) for i in sorted(_grp)) if _grp else (6, 6, 6, 6)
            _headw = sd.get("layers.0.residual_group.blocks.0.correlation.pos.pos3.2.weight")
            _num_heads_v = int(_headw.shape[0]) if _headw is not None else 6
            _num_heads = (_num_heads_v,) * len(_depths)
            _mlpw = sd.get("layers.0.residual_group.blocks.0.mlp.fc1.weight")
            _mlp_ratio = round(_mlpw.shape[0] / max(1, _embed_dim), 4) if _mlpw is not None else 2.0
            _resi = "1conv" if "layers.0.conv.weight" in sd else "3conv"
            if "conv_up1.weight" in sd:
                _up = "nearest+conv"; _up_scale = 4
            elif "conv_before_upsample.0.weight" in sd:
                _up = "pixelshuffle"
                _stages = sorted({int(m.group(1)) for k in sd
                                  if (m := _re.match(r"^upsample\.(\d+)\.weight$", k))})
                if len(_stages) == 1:
                    _outc = sd[f"upsample.{_stages[0]}.weight"].shape[0]
                    _up_scale = 3 if _outc == 9 * 64 else 2
                else:
                    _up_scale = 2 ** max(1, len(_stages))
            elif "upsample.0.weight" in sd:
                _up = "pixelshuffledirect"
                _outc = sd["upsample.0.weight"].shape[0]
                _s = round(math.sqrt(_outc / 3.0))
                _up_scale = _s if _s >= 1 and _s * _s * 3 == _outc else 4
            else:
                _up = ""; _up_scale = 1
            _hit_scale = _scale_hint if _scale_hint > 0 else _up_scale
            _model = _HiTCls(upscale=max(1, _hit_scale), in_chans=_in_chans, embed_dim=_embed_dim,
                             depths=_depths, num_heads=_num_heads, mlp_ratio=_mlp_ratio,
                             upsampler=_up, resi_connection=_resi)
            # base_win_size/hier_win_ratios non dérivables des shapes → défauts du registre ;
            # strict=False pour tolérer un éventuel mismatch de spatial_linear.
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device).eval()
            _arch = _hit_arch_name
            _scale = max(1, _hit_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"hit_sng-manual embed_dim={_embed_dim} depths={_depths}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"HiT-SNG load failed: {_e}"})
            return

    # ── HiT-SIR — Spatial-Channel Correlation (SCC/DFE), no NGram context ────
    if any(".correlation.qv.linear.weight" in k for k in sd) and "conv_first.weight" in sd:
        try:
            import re as _re
            from traiNNer.archs.hit_sir_arch import HiT_SIR as _HiTCls
            _hit_arch_name = "HiT-SIR"
            _cf = sd.get("conv_first.weight")
            _embed_dim = int(_cf.shape[0]) if _cf is not None else 60
            _in_chans = int(_cf.shape[1]) if _cf is not None else 3
            _grp = {}
            for _k in sd:
                _mm = _re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", _k)
                if _mm:
                    _grp.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
            _depths = tuple(len(_grp[i]) for i in sorted(_grp)) if _grp else (6, 6, 6, 6)
            _headw = sd.get("layers.0.residual_group.blocks.0.correlation.pos.pos3.2.weight")
            _num_heads_v = int(_headw.shape[0]) if _headw is not None else 6
            _num_heads = (_num_heads_v,) * len(_depths)
            _mlpw = sd.get("layers.0.residual_group.blocks.0.mlp.fc1.weight")
            _mlp_ratio = round(_mlpw.shape[0] / max(1, _embed_dim), 4) if _mlpw is not None else 2.0
            _resi = "1conv" if "layers.0.conv.weight" in sd else "3conv"
            if "conv_up1.weight" in sd:
                _up = "nearest+conv"; _up_scale = 4
            elif "conv_before_upsample.0.weight" in sd:
                _up = "pixelshuffle"
                _stages = sorted({int(m.group(1)) for k in sd
                                  if (m := _re.match(r"^upsample\.(\d+)\.weight$", k))})
                if len(_stages) == 1:
                    _outc = sd[f"upsample.{_stages[0]}.weight"].shape[0]
                    _up_scale = 3 if _outc == 9 * 64 else 2
                else:
                    _up_scale = 2 ** max(1, len(_stages))
            elif "upsample.0.weight" in sd:
                _up = "pixelshuffledirect"
                _outc = sd["upsample.0.weight"].shape[0]
                _s = round(math.sqrt(_outc / 3.0))
                _up_scale = _s if _s >= 1 and _s * _s * 3 == _outc else 4
            else:
                _up = ""; _up_scale = 1
            _hit_scale = _scale_hint if _scale_hint > 0 else _up_scale
            _model = _HiTCls(upscale=max(1, _hit_scale), in_chans=_in_chans, embed_dim=_embed_dim,
                             depths=_depths, num_heads=_num_heads, mlp_ratio=_mlp_ratio,
                             upsampler=_up, resi_connection=_resi)
            _model.load_state_dict(sd, strict=False)
            _model = _model.to(_device).eval()
            _arch = _hit_arch_name
            _scale = max(1, _hit_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"hit_sir-manual embed_dim={_embed_dim} depths={_depths}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"HiT-SIR load failed: {_e}"})
            return

    # ── CascadedGaze — NAFNet-style UNet with GlobalContextExtractor blocks ──
    if ("intro.weight" in sd and "ending.weight" in sd
            and any(".GCE.depthwise_separable_convs." in k for k in sd)):
        try:
            import re as _re
            from traiNNer.archs.cgnet_arch import CascadedGaze
            _introw = sd.get("intro.weight")
            _width = int(_introw.shape[0]) if _introw is not None else 60
            _img_channel = int(_introw.shape[1]) if _introw is not None else 3
            _enc = {}
            for _k in sd:
                _mm = _re.match(r"^encoders\.(\d+)\.(\d+)\.", _k)
                if _mm:
                    _enc.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
            _enc_blk_nums = [len(_enc[i]) for i in sorted(_enc)] if _enc else [2, 2, 4, 6]
            _dec = {}
            for _k in sd:
                _mm = _re.match(r"^decoders\.(\d+)\.(\d+)\.", _k)
                if _mm:
                    _dec.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
            _dec_blk_nums = [len(_dec[i]) for i in sorted(_dec)] if _dec else [2, 2, 2, 2]
            _mid_idx = {int(m.group(1)) for k in sd if (m := _re.match(r"^middle_blks\.(\d+)\.", k))}
            _middle_blk_num = len(_mid_idx) if _mid_idx else 10
            _stage_ids = sorted(_enc) if _enc else list(range(4))
            _gce_convs_nums = []
            for i in _stage_ids:
                _has3 = f"encoders.{i}.0.GCE.depthwise_separable_convs.2.pointwise.weight" in sd
                _gce_convs_nums.append(3 if _has3 else 2)
            if not _gce_convs_nums:
                _gce_convs_nums = [3, 3, 2, 2]
            # CascadedGaze est une arch restauration/débruitage — scale toujours 1
            # (assert dans le constructeur) ; scale_hint UI ignoré pour cette arch.
            _model = CascadedGaze(scale=1, img_channel=_img_channel, width=_width,
                                  middle_blk_num=_middle_blk_num, enc_blk_nums=_enc_blk_nums,
                                  dec_blk_nums=_dec_blk_nums, GCE_CONVS_nums=_gce_convs_nums)
            _model.load_state_dict(sd, strict=True)
            _model = _model.to(_device).eval()
            _arch = "CascadedGaze"
            _scale = 1
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"cascadedgaze-manual width={_width} enc={_enc_blk_nums}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"CascadedGaze load failed: {_e}"})
            return

    # ── ESCRealM — ESC-style plk_filter + UniUpsampleV3 (distinct from neosr's
    #    "esc" arch, which never has a to_img.MetaUpsample buffer) — DOIT tourner
    #    avant le bloc _NEOSR_ONLY plus bas (qui rejette tout "plk_filter") ──────
    if "plk_filter" in sd and "to_img.MetaUpsample" in sd:
        try:
            import re as _re
            from traiNNer.archs.escreal_arch import ESCRealM
            from traiNNer.archs.arch_util import SampleMods3
            if "proj.1.weight" in sd:
                _unshuffle_mod = True
                _pw = sd["proj.1.weight"]
                _dim = int(_pw.shape[0])
                _in_ch2 = int(_pw.shape[1])
                _unshuffle_factor = max(1, round(math.sqrt(_in_ch2 / 3.0)))
                _exposed_scale = max(1, 4 // _unshuffle_factor)
            else:
                _unshuffle_mod = False
                _pw = sd.get("proj.weight")
                _dim = int(_pw.shape[0]) if _pw is not None else 64
                _exposed_scale = None
            _plk = sd.get("plk_filter")
            _pdim = int(_plk.shape[0]) if _plk is not None else 16
            _kernel_size = int(_plk.shape[-1]) if _plk is not None else 13
            _blk_idx = {int(m.group(1)) for k in sd if (m := _re.match(r"^blocks\.(\d+)\.", k))}
            _n_blocks = len(_blk_idx) if _blk_idx else 10
            _conv_idx = {int(m.group(1)) for k in sd
                         if (m := _re.match(r"^blocks\.0\.pconvs\.(\d+)\.", k))}
            _conv_blocks = len(_conv_idx) if _conv_idx else 5
            _rpb = sd.get("blocks.0.attn.relative_position_bias")
            if _rpb is not None:
                _num_heads = int(_rpb.shape[0])
                _window_size = max(2, int(round((math.sqrt(_rpb.shape[1]) + 1) / 2)))
            else:
                _num_heads, _window_size = 4, 32
            _cffw = sd.get("blocks.0.convffns.0.proj.weight")
            _exp_ratio = round(_cffw.shape[0] / max(1, _dim), 4) if _cffw is not None else 2.0
            _up = "transpose+conv"; _mid_dim = 64; _scale_internal = 4
            _meta = sd.get("to_img.MetaUpsample")
            if _meta is not None:
                try:
                    _up = list(SampleMods3.__args__)[int(_meta[1].item())]
                    _scale_internal = max(1, int(_meta[2].item()))
                    _mid_dim = int(_meta[5].item())
                except Exception:
                    pass
            _det_scale = _exposed_scale if _exposed_scale is not None else _scale_internal
            _escrealm_scale = _scale_hint if _scale_hint > 0 else _det_scale
            # attn_type: jamais "Flex" (triton, absent / non supporté sous Windows et Pascal).
            # Mêmes poids et même calcul dans les 3 modes : "SDPA" sur GPU (noyau économe,
            # moitié de la VRAM de "Naive"), "Naive" sur CPU.
            _model = ESCRealM(dim=_dim, pdim=_pdim, kernel_size=_kernel_size, n_blocks=_n_blocks,
                              conv_blocks=_conv_blocks, window_size=_window_size, num_heads=_num_heads,
                              scale=max(1, _escrealm_scale), exp_ratio=_exp_ratio,
                              attn_type="SDPA" if str(_device).startswith("cuda") else "Naive",
                              mid_dim=_mid_dim, upsampler=_up, unshuffle_mod=_unshuffle_mod)
            _model.load_state_dict(sd, strict=True)
            _model = _model.to(_device).eval()
            _arch = "ESCRealM"
            _scale = max(1, _escrealm_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"escrealm-manual dim={_dim} n_blocks={_n_blocks} up={_up}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"ESCRealM load failed: {_e}"})
            return

    # ── EMT — Efficient Mixed Transformer (TransformerGroup sa_list/mlp_list) ─
    if "sub_mean.weight" in sd and "head.weight" in sd and any(".sa_list." in k for k in sd):
        try:
            import re as _re
            from traiNNer.archs.emt_arch import EMT
            _headw = sd.get("head.weight")
            _dim = int(_headw.shape[0]) if _headw is not None else 60
            _num_in_ch = int(_headw.shape[1]) if _headw is not None else 3
            _blk_idx = {int(m.group(1)) for k in sd if (m := _re.match(r"^body\.(\d+)\.", k))}
            _n_blocks = len(_blk_idx) if _blk_idx else 6
            _layer_idx = {int(m.group(1)) for k in sd
                          if (m := _re.match(r"^body\.0\.sa_list\.(\d+)\.", k))}
            _n_layers = len(_layer_idx) if _layer_idx else 6
            _n_gtls = len([j for j in _layer_idx
                           if f"body.0.sa_list.{j}.attn.0.weight" in sd]) if _layer_idx else 2
            _n_gtls = max(1, _n_gtls)
            _mlpw = sd.get("body.0.mlp_list.0.fc1.weight")
            _mlp_ratio = round(_mlpw.shape[0] / max(1, _dim)) if _mlpw is not None else 2
            _tidx = sorted({int(m.group(1)) for k in sd
                            if (m := _re.match(r"^tail\.(\d+)\.weight$", k))})
            if len(_tidx) >= 2:
                _up = "pixelshuffle"
                _stage0_w = sd[f"tail.{_tidx[0]}.weight"]
                _ratio0 = round(_stage0_w.shape[0] / max(1, _dim))
                _det_scale = 3 if _ratio0 == 9 else 2 ** max(1, len(_tidx) - 1)
            else:
                _up = "pixelshuffledirect"
                _w0 = sd.get("tail.0.weight")
                _det_scale = round(math.sqrt(_w0.shape[0] / 3.0)) if _w0 is not None else 4
            _emt_scale = _scale_hint if _scale_hint > 0 else _det_scale
            # num_heads/window_list/shift_list n'affectent aucune shape de poids — non
            # dérivables du state_dict. SWSA rearrange (qv=2,head=num_heads) exige que
            # dim soit divisible par 2*num_heads sous peine de crash au forward.
            _num_heads = next((h for h in (3, 4, 5, 6, 2, 1, 8, 10, 12, 15, 20)
                               if _dim % (2 * h) == 0), 1)
            _model = EMT(scale=max(1, _emt_scale), num_in_ch=_num_in_ch, num_out_ch=3,
                        upsampler=_up, dim=_dim, n_blocks=_n_blocks, n_layers=_n_layers,
                        n_GTLs=_n_gtls, num_heads=_num_heads, mlp_ratio=_mlp_ratio)
            _model.load_state_dict(sd, strict=True)
            _model = _model.to(_device).eval()
            _arch = "EMT"
            _scale = max(1, _emt_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"emt-manual dim={_dim} n_blocks={_n_blocks} n_layers={_n_layers}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"EMT load failed: {_e}"})
            return

    # ── FlexNet — short_cut ConvBlock + LinearPipeline/MetaPipeline body ─────
    if ("in_to_feat.weight" in sd and any(k.startswith("short_cut.") for k in sd)
            and any(k.startswith("pipeline.") for k in sd)):
        try:
            import re as _re
            from traiNNer.archs.flexnet_arch import FlexNet
            _c11w = sd.get("short_cut.conv11.weight")
            _dim = int(_c11w.shape[0]) if _c11w is not None else 64
            _inp_channels = int(_c11w.shape[1]) if _c11w is not None else 3
            if any(k.startswith("pipeline.att.") for k in sd):
                _pipeline_type = "linear"
                _stg = {}
                for _k in sd:
                    _mm = _re.match(r"^pipeline\.att\.(\d+)\.t_blocks\.(\d+)\.", _k)
                    if _mm:
                        _stg.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                _num_blocks = tuple(len(_stg[i]) for i in sorted(_stg)) if _stg else (6, 6, 6, 6, 6, 6)
                _ffnw = sd.get("pipeline.att.0.t_blocks.0.ffn.key.weight")
            else:
                _pipeline_type = "meta"
                _stg = {}
                for _k in sd:
                    _mm = _re.match(r"^pipeline\.enc(\d+)\.0\.t_blocks\.(\d+)\.", _k)
                    if _mm:
                        _stg.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                _num_blocks = tuple(len(_stg[i]) for i in sorted(_stg)) if _stg else (4, 6, 6, 8)
                _ffnw = sd.get("pipeline.enc0.0.t_blocks.0.ffn.key.weight")
            _hidden_rate = round(_ffnw.shape[0] / max(1, _dim)) if _ffnw is not None else 4
            _channel_norm = any(".ffn.key_norm.weight" in k for k in sd)
            _ws_buf = sd.get("window_size")
            try:
                _window_size = int(_ws_buf.item()) if _ws_buf is not None else 8
            except Exception:
                _window_size = 8
            if "to_img.offset.weight" in sd:
                _up = "dysample"
                _offw = sd["to_img.offset.weight"]
                _det_scale = max(1, round(math.sqrt(_offw.shape[0] / 8.0)))
            elif "to_img.1.0.weight" in sd:
                _up = "nearest+conv"
                _sf = sd.get("scale_factor")
                try:
                    _det_scale = max(1, int(_sf.item())) if _sf is not None else 4
                except Exception:
                    _det_scale = 4
            else:
                _up = "pixelshuffle"
                _imw = sd.get("to_img.0.weight")
                _det_scale = max(1, round(math.sqrt(_imw.shape[0] / 3.0))) if _imw is not None else 4
            _flexnet_scale = _scale_hint if _scale_hint > 0 else _det_scale
            _model = FlexNet(inp_channels=_inp_channels, out_channels=3, scale=max(1, _flexnet_scale),
                             dim=_dim, num_blocks=_num_blocks, window_size=_window_size,
                             hidden_rate=_hidden_rate, channel_norm=_channel_norm,
                             pipeline_type=_pipeline_type, upsampler=_up)
            _model.load_state_dict(sd, strict=True)
            _model = _model.to(_device).eval()
            _arch = "FlexNet"
            _scale = max(1, _flexnet_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"flexnet-manual dim={_dim} pipeline={_pipeline_type}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"FlexNet load failed: {_e}"})
            return

    # ── SRFormerv2 (traiNNer native) — mixe SwinTransformerBlock (qkv) et
    #    PSA_Block (kv/q) — DOIT tourner avant SRFormer (signature plus stricte) ─
    if ("layers.0.residual_group.blocks.0.attn.qkv.weight" in sd
            and "layers.0.residual_group.blocks.1.attn.kv.weight" in sd):
        try:
            import re as _re
            from traiNNer.archs.srformerv2_arch import SRFormerv2
            _cf = sd.get("conv_first.weight")
            _embed_dim = int(_cf.shape[0]) if _cf is not None else 240
            _in_chans = int(_cf.shape[1]) if _cf is not None else 3
            _grp = {}
            for _k in sd:
                _mm = _re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", _k)
                if _mm:
                    _grp.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
            # Nb de blocs bruts par layer = depth + 3 (3 SwinTransformerBlock toujours
            # insérés aux positions [0,3,6] par srformerv2_arch.py, quel que soit depth).
            _depths = tuple(max(1, len(_grp[i]) - 3) for i in sorted(_grp)) if _grp else (4, 4, 4, 4, 4, 4)
            _psa_idx = None
            for _j in sorted(_grp.get(0, [])):
                if f"layers.0.residual_group.blocks.{_j}.attn.kv.weight" in sd:
                    _psa_idx = _j
                    break
            if _psa_idx is not None:
                _kvw = sd[f"layers.0.residual_group.blocks.{_psa_idx}.attn.kv.weight"]
                _squeeze_dim = int(_kvw.shape[0] // 2)
                _rpbt = sd.get(f"layers.0.residual_group.blocks.{_psa_idx}.attn.relative_position_bias_table")
                _num_heads_v = int(_rpbt.shape[1]) if _rpbt is not None else 8
                _window_size = int(round(math.sqrt(_rpbt.shape[0]))) + 1 if _rpbt is not None else 36
                _mlpw = sd.get(f"layers.0.residual_group.blocks.{_psa_idx}.mlp.fc1.weight")
                _mlp_ratio = round(_mlpw.shape[0] / max(1, _embed_dim)) if _mlpw is not None else 2
            else:
                _squeeze_dim, _num_heads_v, _window_size, _mlp_ratio = 60, 8, 36, 2
            _num_heads = (_num_heads_v,) * len(_depths)
            _resi = "1conv" if "conv_after_body.weight" in sd else "3conv"
            if "conv_before_upsample.0.weight" in sd:
                _up = "pixelshuffle"
                _stages = sorted({int(m.group(1)) for k in sd
                                  if (m := _re.match(r"^upsample\.(\d+)\.weight$", k))})
                if len(_stages) == 1:
                    _outc = sd[f"upsample.{_stages[0]}.weight"].shape[0]
                    _det_scale = 3 if round(_outc / 64) == 9 else 2
                else:
                    _det_scale = 2 ** max(1, len(_stages))
            elif "upsample.0.weight" in sd:
                _up = "pixelshuffledirect"
                _outc = sd["upsample.0.weight"].shape[0]
                _s = round(math.sqrt(_outc / max(1, _in_chans)))
                _det_scale = _s if _s >= 1 else 4
            elif "conv_hr.weight" in sd:
                _up = "nearest+conv"; _det_scale = 4
            else:
                _up = ""; _det_scale = 2
            _srformerv2_scale = _scale_hint if _scale_hint > 0 else _det_scale
            # img_size : patch_size==1 toujours → patches_resolution==img_size ; dérivé de la
            # vraie shape d'attn_mask (buffer persistant d'un bloc PSA shift>0) — une shape
            # fausse fait planter le load strict.
            _img_size = _window_size
            for _k in sd:
                if _re.match(r"^layers\.0\.residual_group\.blocks\.\d+\.attn_mask$", _k):
                    _nw = sd[_k].shape[0]
                    _img_size = _window_size * max(1, round(math.sqrt(_nw)))
                    break
            _model = SRFormerv2(upscale=max(1, _srformerv2_scale), in_chans=_in_chans,
                                embed_dim=_embed_dim, squeeze_dim=_squeeze_dim, depths=_depths,
                                num_heads=_num_heads, window_size=_window_size, img_size=_img_size,
                                mlp_ratio=_mlp_ratio, upsampler=_up, resi_connection=_resi)
            _model.load_state_dict(sd, strict=True)
            _model = _model.to(_device).eval()
            _arch = "SRFormerv2"
            _scale = max(1, _srformerv2_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"srformerv2-manual embed_dim={_embed_dim} depths={_depths}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"SRFormerv2 load failed: {_e}"})
            return

    # ── SRFormer (spandrel_extra_arches-compatible) — tous les blocs sont PSA_Block ─
    if ("layers.0.residual_group.blocks.0.attn.kv.weight" in sd
            and "layers.0.residual_group.blocks.0.attn.q.weight" in sd):
        try:
            import re as _re
            from spandrel_extra_arches.architectures.SRFormer import SRFormer
            _cf = sd.get("conv_first.weight")
            _embed_dim = int(_cf.shape[0]) if _cf is not None else 180
            _in_chans = int(_cf.shape[1]) if _cf is not None else 3
            _grp = {}
            for _k in sd:
                _mm = _re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", _k)
                if _mm:
                    _grp.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
            _depths = tuple(len(_grp[i]) for i in sorted(_grp)) if _grp else (6, 6, 6, 6, 6, 6)
            _heads_list = []
            for i in (sorted(_grp) if _grp else range(len(_depths))):
                _rpbt = sd.get(f"layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table")
                _heads_list.append(int(_rpbt.shape[1]) if _rpbt is not None else 6)
            _num_heads = tuple(_heads_list) if _heads_list else (6,) * len(_depths)
            _rpbt0 = sd.get("layers.0.residual_group.blocks.0.attn.relative_position_bias_table")
            _window_size = int(round(math.sqrt(_rpbt0.shape[0]))) + 1 if _rpbt0 is not None else 24
            _mlpw = sd.get("layers.0.residual_group.blocks.0.mlp.fc1.weight")
            _mlp_ratio = round(_mlpw.shape[0] / max(1, _embed_dim), 4) if _mlpw is not None else 2.0
            _ape = "absolute_pos_embed" in sd
            _patch_norm = "patch_embed.norm.weight" in sd
            _qkv_bias = "layers.0.residual_group.blocks.0.attn.q.bias" in sd
            _resi = "1conv" if "conv_after_body.weight" in sd else "3conv"
            if "conv_hr.weight" in sd:
                _up = "nearest+conv"; _det_scale = 4
            elif "conv_before_upsample.0.weight" in sd:
                _up = "pixelshuffle"
                _stages = sorted({int(m.group(1)) for k in sd
                                  if (m := _re.match(r"^upsample\.(\d+)\.weight$", k))})
                if len(_stages) == 1:
                    _outc = sd[f"upsample.{_stages[0]}.weight"].shape[0]
                    _det_scale = 3 if round(_outc / 64) == 9 else 2
                else:
                    _det_scale = 2 ** max(1, len(_stages))
            elif "upsample.0.weight" in sd:
                _up = "pixelshuffledirect"
                _outc = sd["upsample.0.weight"].shape[0]
                _s = round(math.sqrt(_outc / max(1, _in_chans)))
                _det_scale = _s if _s >= 1 else 4
            else:
                _up = ""; _det_scale = 1
            _srformer_scale = _scale_hint if _scale_hint > 0 else _det_scale
            # img_size : patch_size==1 toujours → patches_resolution==img_size ; dérivé de la
            # vraie shape d'attn_mask (buffer persistant d'un bloc PSA shift>0) — une shape
            # fausse fait planter le load strict.
            _img_size = _window_size
            _found_mask = False
            for _k in sd:
                if _re.match(r"^layers\.0\.residual_group\.blocks\.\d+\.attn_mask$", _k):
                    _nw = sd[_k].shape[0]
                    _img_size = _window_size * max(1, round(math.sqrt(_nw)))
                    _found_mask = True
                    break
            if not _found_mask and _ape:
                _ape_t = sd.get("absolute_pos_embed")
                if _ape_t is not None:
                    _img_size = int(round(math.sqrt(_ape_t.shape[1])))
            _model = SRFormer(upscale=max(1, _srformer_scale), in_chans=_in_chans, img_size=_img_size,
                              embed_dim=_embed_dim, depths=_depths, num_heads=_num_heads,
                              window_size=_window_size, mlp_ratio=_mlp_ratio, qkv_bias=_qkv_bias,
                              ape=_ape, patch_norm=_patch_norm, upsampler=_up,
                              resi_connection=_resi)
            _model.load_state_dict(sd, strict=True)
            _model = _model.to(_device).eval()
            _arch = "SRFormer"
            _scale = max(1, _srformer_scale)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"srformer-manual embed_dim={_embed_dim} depths={_depths}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"SRFormer load failed: {_e}"})
            return

    # ── DocRes (Restormer-style, traiNNer native) — restauration de document
    #    (deskew/nettoyage scan), scale fixe = 1 (jamais d'upscale) ─────────
    if ("patch_embed.proj.weight" in sd
            and "encoder_level1.0.attn.temperature" in sd
            and "refinement.0.attn.temperature" in sd):
        try:
            import re as _re
            from traiNNer.archs.docres_arch import DocRes
            _pew = sd.get("patch_embed.proj.weight")
            _dim = int(_pew.shape[0]) if _pew is not None else 48
            _embed_ch = int(_pew.shape[1]) if _pew is not None else 5
            _inp_channels = 3 if _embed_ch == 5 else _embed_ch
            _outw = sd.get("output.weight")
            _out_channels = int(_outw.shape[0]) if _outw is not None else 3
            _bias = "encoder_level1.0.attn.qkv.bias" in sd
            _ln_type = "WithBias" if "encoder_level1.0.norm1.body.bias" in sd else "BiasFree"

            def _docres_count(_prefix):
                _idx = {int(m.group(1)) for k in sd if (m := _re.match(rf"^{_prefix}\.(\d+)\.", k))}
                return len(_idx)

            _n1 = _docres_count("encoder_level1") or 2
            _n2 = _docres_count("encoder_level2") or 3
            _n3 = _docres_count("encoder_level3") or 3
            _n4 = _docres_count("latent") or 4
            _num_blocks = [_n1, _n2, _n3, _n4]
            _num_refinement = _docres_count("refinement") or 4

            def _docres_heads(_prefix):
                _t = sd.get(f"{_prefix}.0.attn.temperature")
                return int(_t.shape[0]) if _t is not None else 1

            _heads = [_docres_heads("encoder_level1"), _docres_heads("encoder_level2"),
                      _docres_heads("encoder_level3"), _docres_heads("latent")]

            # See universal_runner.py's docres branch for why a single-stage back-solve
            # is insufficient: narrow the valid-factor interval across all 4 stages
            # (dim doubles each stage) and take the midpoint — reproduces every stage's
            # hidden_features exactly.
            _ffn_pairs = []
            for _pfx, _mult in (("encoder_level1", 1), ("encoder_level2", 2),
                                 ("encoder_level3", 4), ("latent", 8)):
                _w = sd.get(f"{_pfx}.0.ffn.project_in.weight")
                if _w is not None:
                    _ffn_pairs.append((_dim * _mult, int(_w.shape[0]) // 2))
            if _ffn_pairs:
                _lo = max(h / d for d, h in _ffn_pairs)
                _hi = min((h + 1) / d for d, h in _ffn_pairs)
                _ffn_exp = (_lo + _hi) / 2.0 if _lo < _hi else _lo
            else:
                _ffn_exp = 2.66

            _model = DocRes(inp_channels=_inp_channels, out_channels=_out_channels, dim=_dim,
                             num_blocks=_num_blocks, num_refinement_blocks=_num_refinement,
                             heads=_heads, ffn_expansion_factor=_ffn_exp, bias=_bias,
                             LayerNorm_type=_ln_type, dual_pixel_task=True)
            _model.load_state_dict(sd, strict=True)
            _model = _model.to(_device).eval()
            _arch = "DocRes"
            _scale = 1  # restauration document uniquement, jamais d'upscale (comme cascadedgaze)
            _emit({"status": "ready", "arch": _arch, "scale": _scale,
                   "backend": f"docres-manual dim={_dim} num_blocks={_num_blocks} heads={_heads}"})
            return
        except Exception as _e:
            _emit({"status": "error", "msg": f"DocRes load failed: {_e}"})
            return

    # ── Block NeoSR-only architectures BEFORE the SPAN fallback ──────────────
    # These archs exist only in the NeoSR engine (different venv from traiNNer).
    # strict=False would let SPAN load their weights silently → wrong results.
    # Emit a clear error so the session fails and the caller falls back to
    # the normal NeoSR upscale path.
    _NEOSR_ONLY = {
        "ESC":     "plk_filter",                              # Efficient Scale-invariant Context
        "CATANet": "blocks.0.0.irca_attn.to_k.weight",       # CATANet (NeoSR, Mar 2025)
        "SpanC":   "block_1.conv_a.eval_conv",                # SpanC (NeoSR MetaIGConv)
    }
    for _neosr_arch, _neosr_key in _NEOSR_ONLY.items():
        if _neosr_key in sd:
            _emit({
                "status": "error",
                "msg": (
                    f"'{_neosr_arch}' est une arch NeoSR-only — "
                    f"non supporté par le subprocess persistant (venv traiNNer). "
                    f"Décochez 'Subprocess persistant' pour ce modèle."
                ),
            })
            return

    # 5) SPAN generic fallback — ONLY if model has SPAN-signature keys.
    # Without this guard, non-SPAN archs (RCAN, HAT, SwinIR…) silently load into SPAN
    # via strict=False → wrong output (black/red images, wrong scale).
    _has_span_keys = "feats.0.sk.weight" in sd or "feats.0.eval_conv.weight" in sd
    if not _has_span_keys:
        _emit({
            "status": "error",
            "msg": (
                "Arch non reconnue dans le subprocess persistant (venv traiNNer). "
                "Décochez 'Subprocess persistant' pour ce modèle."
            ),
        })
        return
    try:
        from traiNNer.archs.span_arch import SPAN
        _fc_w = sd.get("feats.0.sk.weight", None)
        _fc   = int(_fc_w.shape[0]) if _fc_w is not None else 48
        _model = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=_fc, upscale=_scale,
                      bias=True, norm=False, img_range=255.,
                      rgb_mean=(0.4488, 0.4371, 0.4040)).eval()
        _model.load_state_dict(sd, strict=False)
        _model = _model.to(_device)
        _arch  = "SPAN"
        _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": "span-fallback"})
        return
    except Exception:
        pass

    _emit({"status": "error", "msg": "Could not load model (spandrel + SPANPlus + SPAN all failed)"})


def cmd_infer(payload: dict) -> None:
    global _model, _scale, _device, _tile_size, _tile_pad, _use_amp, _last_hw
    if _model is None:
        _emit({"status": "error", "msg": "Model not loaded — send init first"})
        return

    input_path        = payload["input"]
    output_path       = payload["output"]
    prev_input_path   = payload.get("prev_input")   # dandere tile mode
    prev_output_path  = payload.get("prev_output")  # dandere tile mode
    dandere_threshold = float(payload.get("dandere_threshold", 0.02))

    if not os.path.isfile(input_path):
        _emit({"status": "error", "msg": f"Input not found: {input_path}"})
        return

    try:
        img = Image.open(input_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(_device)

        # Resolution change → defragment caching allocator before inference.
        # Prevents reserved-VRAM buildup + cudaMalloc thrashing when frame
        # sizes vary within a batch (e.g. 1080p clip + a 480p outlier).
        cur_hw = (t.shape[2], t.shape[3])
        if _device.type == "cuda" and _last_hw is not None and cur_hw != _last_hw:
            torch.cuda.empty_cache()
        _last_hw = cur_hw

        amp_ok = _use_amp and _device.type == "cuda"

        # ── Dandere2x tile mode ───────────────────────────────────────
        # Requires tiling (tile_size > 0), prev_input, and prev_output.
        # Compares padded LQ tiles between frames; only sends changed tiles
        # to the GPU; unchanged tiles are copied from prev SR output.
        # No seam risk: same padded input → deterministic model → same SR.
        if (prev_input_path and prev_output_path and _tile_size > 0
                and os.path.isfile(prev_input_path)
                and os.path.isfile(prev_output_path)):
            try:
                prev_img = Image.open(prev_input_path).convert("RGB")
                prev_arr = np.array(prev_img).astype(np.float32) / 255.0
                prev_t   = torch.from_numpy(prev_arr).permute(2, 0, 1).unsqueeze(0).to(_device)

                prev_sr_img = Image.open(prev_output_path).convert("RGB")
                prev_sr_arr = np.array(prev_sr_img).astype(np.float32) / 255.0
                prev_sr_t   = torch.from_numpy(prev_sr_arr).permute(2, 0, 1).unsqueeze(0).to(_device)

                if amp_ok:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out, n_changed, n_total = _tile_inference_dandere(
                            _model, t, prev_t, prev_sr_t,
                            _tile_size, _tile_pad, _scale, dandere_threshold)
                else:
                    out, n_changed, n_total = _tile_inference_dandere(
                        _model, t, prev_t, prev_sr_t,
                        _tile_size, _tile_pad, _scale, dandere_threshold)

                out_np  = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().float().numpy()
                _write_output(out_np, output_path, payload.get("save"))
                pct = round(n_changed / n_total * 100, 1) if n_total > 0 else 100.0
                _emit({"status": "ok", "msg": os.path.basename(output_path),
                       "dandere_changed": n_changed, "dandere_total": n_total,
                       "dandere_pct": pct})
                return
            except Exception:
                # Fallback to normal inference if dandere tile mode fails
                pass

        # ── Normal inference ──────────────────────────────────────────
        # CUDA out of memory (attention archs such as MambaIRv2 with Tile 0 on a 1080p
        # frame want ~8 GB in one block) -> free VRAM, retry with tiles of 256 then 128 and keep
        # the size that fits for the rest of the batch (only the first image pays).
        tile = _tile_size
        note = ""
        while True:
            try:
                with torch.inference_mode():
                    if amp_ok:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out = _tile_inference(_model, t, tile, _tile_pad, _scale)
                    else:
                        out = _tile_inference(_model, t, tile, _tile_pad, _scale)
                break
            except torch.cuda.OutOfMemoryError:
                out = None
                if _device.type == "cuda":
                    torch.cuda.empty_cache()
                # 256 first: on Windows a size that "fits" only thanks to the driver's
                # shared-memory fallback runs ~2x slower (MambaIRv2 1080p: 512 = 73 s,
                # 256 = 37 s, 128 = 54 s per frame on a 1080 Ti).
                nxt = 256 if (tile == 0 or tile > 256) else tile // 2
                if nxt < 128:
                    raise
                tile = nxt
        if tile != _tile_size:
            note = (f"[Auto-tile] Mémoire GPU insuffisante en Tile {_tile_size or 0} "
                    f"→ tuiles de {tile} px pour la suite du lot.")
            _tile_size = tile

        out_np  = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().float().numpy()
        _write_output(out_np, output_path, payload.get("save"))
        reply = {"status": "ok", "msg": os.path.basename(output_path)}
        if note:
            reply["note"] = note
        _emit(reply)

    except Exception as e:
        _emit({"status": "error", "msg": f"Inference error: {traceback.format_exc()}"})



def _write_output(out_np, output_path, save=None):
    """Write a float32 [H,W,3] array in [0,1] with the format / bit depth / quality
    chosen in the Quick Upscale tab (payload "save"). Without "save" the file is an
    8-bit image in the format of its extension, as before."""
    save = save or {}
    fmt = str(save.get("format") or "").upper()
    bit_depth = int(save.get("bit_depth") or 8)
    quality = int(save.get("quality") or 95)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if bit_depth == 16 and (fmt in ("PNG", "TIFF") or output_path.lower().endswith((".png", ".tif", ".tiff"))):
        import cv2
        arr16 = (out_np * 65535.0).round().clip(0, 65535).astype(np.uint16)
        if not _imwrite_u(output_path, np.ascontiguousarray(arr16[:, :, ::-1])):
            raise RuntimeError(f"cv2.imwrite failed: {output_path}")
        return
    img = Image.fromarray((out_np * 255.0).round().clip(0, 255).astype(np.uint8))
    kw = {}
    if fmt in ("JPEG", "JPG") or output_path.lower().endswith((".jpg", ".jpeg")):
        kw = {"quality": quality, "subsampling": 0}
    elif fmt == "WEBP" or output_path.lower().endswith(".webp"):
        kw = {"quality": quality}
    elif fmt == "TIFF" or output_path.lower().endswith((".tif", ".tiff")):
        kw = {"compression": "tiff_lzw"}
    img.save(output_path, **kw)


def cmd_quit() -> None:
    global _model, _device
    _model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _emit({"status": "bye"})
    sys.exit(0)


def main() -> None:
    """Read JSON lines from stdin, dispatch commands."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"status": "error", "msg": f"JSON parse error: {e}"})
            continue

        cmd = msg.get("cmd", "")
        if cmd == "init":
            cmd_init(msg)
        elif cmd == "infer":
            cmd_infer(msg)
        elif cmd == "quit":
            cmd_quit()
        else:
            _emit({"status": "error", "msg": f"Unknown command: {cmd}"})


if __name__ == "__main__":
    main()
