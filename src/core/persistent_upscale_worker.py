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

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add traiNNer-redux to path (worker runs in traiNNer venv)
TRAINNER_PATH = os.path.join(os.path.expanduser("~"), "IA_Engine", "traiNNer-redux")
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


def _emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


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
    global _model, _scale, _arch, _device, _tile_size, _tile_pad, _use_amp
    model_path = payload["model"]
    _tile_size = int(payload.get("tile_size", 256))
    _tile_pad  = int(payload.get("tile_pad", 32))
    _use_amp   = bool(payload.get("use_amp", False))

    if not os.path.isfile(model_path):
        _emit({"status": "error", "msg": f"Model not found: {model_path}"})
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        sd = _load_state_dict(model_path)
    except Exception as e:
        _emit({"status": "error", "msg": f"Load state_dict failed: {e}"})
        return

    _scale = _detect_scale(sd)
    _scale_hint = int(payload.get("scale_hint", 0))  # user-selected scale (0=auto)

    # 1) GFISRV2 manual — MUST run before spandrel (spandrel misidentifies it as SPAN)
    # Mirror universal_runner.py: no upsampler/mid_dim params, strict=False.
    # Scale MUST be inferred from the checkpoint, NOT from _detect_scale():
    #   - _detect_scale() returns 4 (default) for GFISRV2 because its upscale keys
    #     don't match pixel-shuffle patterns → GFISRV2(scale=4,...) creates wrong upscale
    #   - upscale.0.weight [3, 48, 3, 3] → out_ch=3 = RGB = scale 1
    #   - upscale.MetaUpsample → stores scale at index 2 (DySample upsamplers)
    _GFISRV2_KEYS = {"in_to_dim.weight", "gfisr_body.0.fc1.weight"}
    if any(k in sd for k in _GFISRV2_KEYS):
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

    # 2) Spandrel (universel — pour tout ce qui n'est pas GFISRV2)
    try:
        import spandrel
        model_desc = spandrel.ModelLoader(device=_device).load_from_file(model_path)
        _model = model_desc.model.eval().to(_device)
        _scale = getattr(model_desc, "scale", _scale)
        _arch  = type(_model).__name__
        _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": "spandrel"})
        return
    except Exception:
        pass

    # 3) SPANPlus manual (DySample ou conv upsampler)
    try:
        from traiNNer.archs.spanplus_arch import SpanPlus
        _fc_w = sd.get("feats.0.eval_conv.weight", sd.get("feats.0.sk.weight", None))
        _fc   = int(_fc_w.shape[0]) if _fc_w is not None else 48
        _up   = "dys" if "upsampler.offset.weight" in sd else "conv"
        _model = SpanPlus(feature_channels=_fc, upscale=_scale, upsampler=_up).eval()
        _model.load_state_dict(sd, strict=False)
        _model = _model.to(_device)
        _arch  = "SPANPlus"
        _emit({"status": "ready", "arch": _arch, "scale": _scale, "backend": "spanplus-fallback"})
        return
    except Exception:
        pass

    # 4) SpanPP manual (blocs SPAB c1_r/c2_r/c3_r + IGConv Fourier upsampler)
    # Distinct de SPANPlus (feats.*) et SpanC-NeoSR (conv_a). Exclusif traiNNer-redux.
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
            # Prefer user-selected scale if valid, otherwise auto-detect
            _requested = _scale_hint if _scale_hint > 0 else _scale
            _eval_scale = max(1, _requested) if _requested in _scale_list else _scale_list[0]
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

    # 3) SPAN generic fallback
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
    global _model, _scale, _device, _tile_size, _tile_pad, _use_amp
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
                out_img = Image.fromarray((out_np * 255).round().astype(np.uint8))
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                out_img.save(output_path)
                pct = round(n_changed / n_total * 100, 1) if n_total > 0 else 100.0
                _emit({"status": "ok", "msg": os.path.basename(output_path),
                       "dandere_changed": n_changed, "dandere_total": n_total,
                       "dandere_pct": pct})
                return
            except Exception:
                # Fallback to normal inference if dandere tile mode fails
                pass

        # ── Normal inference ──────────────────────────────────────────
        with torch.inference_mode():
            if amp_ok:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = _tile_inference(_model, t, _tile_size, _tile_pad, _scale)
            else:
                out = _tile_inference(_model, t, _tile_size, _tile_pad, _scale)

        out_np  = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().float().numpy()
        out_img = Image.fromarray((out_np * 255).round().astype(np.uint8))
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        out_img.save(output_path)
        _emit({"status": "ok", "msg": os.path.basename(output_path)})

    except Exception as e:
        _emit({"status": "error", "msg": f"Inference error: {traceback.format_exc()}"})


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
