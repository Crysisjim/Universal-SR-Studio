"""
model_convert.py — GUI-agnostic model conversion (PTH / FP16 / BF16 / SafeTensors / ONNX).

Reuses the logic validated for the GFISRv2 standalone + tab_tools fixes:
  * SafeTensors save uses .clone() (avoids Windows os error 1224 on mmap'd source),
  * ONNX export of GFISRv2 goes through the arch's custom FFT `symbolic` with
    dynamo=False (legacy exporter — no onnxscript dependency).

Callable from both the CustomTkinter and PySide6 UIs. `log` is an optional callback
receiving progress strings.
"""
from __future__ import annotations

import os


def _load_state(pth: str):
    import torch
    if pth.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(pth, device="cpu")
    state = torch.load(pth, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for k in ("params_ema", "params_g", "params", "model", "state_dict"):
            if k in state:
                return state[k]
    return state


def _parse_size(value):
    """'960x540' / '960*540' / '960 540' -> (960, 540); anything else -> None."""
    if not value:
        return None
    import re
    m = re.fullmatch(r"\s*(\d+)\s*[x×*, ]\s*(\d+)\s*", str(value))
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    return (w, h) if w > 0 and h > 0 else None


def _check_dynamic_onnx(onnx_path, prec, log):
    """Run the "dynamic" ONNX at a size other than the 64x64 export dummy. If shapes
    were baked in by the tracer, warn instead of silently shipping a file that only
    works at 64x64. Best-effort: skipped when onnxruntime isn't importable."""
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception:
        return
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 4  # the expected failure is reported below, not by ORT
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        x = np.random.rand(1, 3, 72, 96).astype(np.float16 if prec == "fp16" else np.float32)
        ro = ort.RunOptions()
        ro.log_severity_level = 4
        sess.run(None, {"input": x}, ro)
        log("   ✓ ONNX dynamique vérifié (fonctionne à une autre taille que 64x64).")
    except Exception:
        log("   ⚠ Cet ONNX ne fonctionne qu'en 64x64 : l'architecture (attention à fenêtres) "
            "fige les formes au tracing. Ré-exportez avec une « taille fixe » égale à la "
            "résolution d'entrée réelle (ex. 960x540).")


def _cast_state(state, precision):
    """Return a state dict with floating tensors cast to the requested precision."""
    import torch
    if precision == "fp16":
        return {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
    if precision == "bf16":
        return {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in state.items()}
    return state  # fp32 — unchanged


def _build_model(pth, state, opts, log):
    """Load a torch nn.Module for ONNX export. Tries spandrel (≈100 architectures);
    falls back to the bundled custom GFISRv2 exporter. Returns (model, scale) or (None, 1)."""
    import torch
    # 1) spandrel — broad architecture coverage (auto-detect from the checkpoint)
    try:
        if "groups.0.blocks.2.attn.dictionary" in state:  # FDAT-RD: spandrel would take it for FDAT
            raise RuntimeError("FDAT-RD")
        from spandrel import ModelLoader
        try:
            import spandrel_extra_arches
            spandrel_extra_arches.install()
        except Exception:
            pass
        desc = ModelLoader().load_from_file(pth)
        model = desc.model.eval()
        scale = int(getattr(desc, "scale", 1) or 1)
        arch_name = getattr(getattr(desc, "architecture", None), "name", "?")
        log(f"   arch (spandrel) : {arch_name}  scale={scale}x")
        return model, scale
    except Exception as e:
        log(f"   spandrel n'a pas reconnu l'architecture ({type(e).__name__}) — essai des archs traiNNer récentes (HEART, NERVE, WireSR, FDAT-RD, MoTSR, WSR, DUAL)…")

    # 1b) HEART / NERVE (traiNNer-redux master) — hyperparams read from the weights
    try:
        import sys as _sys
        from src.core import arch_infer
        arch = arch_infer.detect(state)
        if arch in arch_infer.SEQUENCE_ARCHS:
            log(f"   {arch} est un modèle vidéo — exporté par l'export séquence.")
            return None, 1
        if arch:
            from src.core import engine_paths as _ep
            _tr = _ep.redux_path()
            if os.path.isdir(_tr) and _tr not in _sys.path:
                _sys.path.insert(0, _tr)
            model, scale, kw = arch_infer.build(arch, state)
            log(f"   arch (traiNNer) : {arch.upper()}  scale={scale}x")
            return model.eval(), scale
    except Exception as e:
        log(f"   HEART/NERVE : construction impossible ({e})")

    # 2) fallback — bundled GFISRv2 custom arch
    try:
        import re, sys as _sys
        arch_dir = opts.get("arch_dir") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "custom_archs")
        if arch_dir not in _sys.path:
            _sys.path.insert(0, arch_dir)
        from gfisrv2_arch import GFISRV2
        dim = 48
        for k in ("in_to_dim.weight", "in_to_dim.1.weight"):
            if k in state:
                dim = int(state[k].shape[0]); break
        body = [int(k.split(".")[1]) for k in state
                if re.fullmatch(r"gfisr_body\.\d+\.(weight|bias)", k)]
        nb = min(body) if body else 24
        sc = 1
        if "upscale.MetaUpsample" in state:
            try:
                sc = int(state["upscale.MetaUpsample"][2].item())
            except Exception:
                pass
        uk = [k for k in state if k.startswith("upscale.")]
        ups = "pixelshuffle" if any("conv_last" in k for k in uk) else "dysample"
        if "upscale.MetaUpsample" in state:
            try:  # [version, method index, scale, ...] — the method actually trained
                import gfisrv2_arch as _g
                ups = list(_g.SampleMods.__args__)[int(state["upscale.MetaUpsample"][1].item())]
            except Exception:
                pass
        log(f"   arch (custom) : GFISRv2 dim={dim} n_blocks={nb} scale={sc}x upsampler={ups}")
        m = GFISRV2(dim=dim, n_blocks=nb, mid_dim=32, scale=sc, upsampler=ups)
        m.load_state_dict(state, strict=True); m.eval()
        return m, sc
    except Exception as e:
        log(f"   ❌ Construction du modèle impossible : {e}")
        return None, 1


def _sequence_arch(state):
    """Video arch name (tscunet, temporalspan(v2), tfdat, tfdatv2, motsr) or None."""
    try:
        from src.core import quick_upscale as qu
        a = qu.detect_arch_from_state(state)
        return a if a in qu._SEQUENCE_ARCHS else None
    except Exception:
        return None


class _ClipStack:
    """Wraps a (B,T,C,H,W) video model behind the vs-mlrt / VapourKit layout:
    one input (1, T*3, H, W), frames stacked on channels, centre frame out."""

    @staticmethod
    def make(model, clip_size):
        import torch

        class ClipStack(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, x):
                b, tc, h, w = x.shape
                return self.model(x.reshape(b, clip_size, tc // clip_size, h, w))

        return ClipStack().eval()


def _export_sequence_onnx(arch, state, base, out_dir, opts, precision, opset, log) -> bool:
    """ONNX export of the video archs.
    MoTSR: the author's two graphs (tools/export_onnx.py) — `seed` (image_only, first
    frame of each scene) and `recurrent` (7 inputs: older_lr, older_hr, previous_lr,
    previous_hr, current_lr, next_lr, later_lr).
    Others: one graph, input (1, T*3, H, W) like the official bndl / VapourKit files."""
    import sys as _sys
    import numpy as np
    import torch
    from src.core import quick_upscale as qu
    model, clip = qu._build_sequence_model(arch, state, 0, log)
    model = model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prec = "fp32" if precision == "bf16" or (precision == "fp16" and dev == "cpu") else precision
    if prec != precision:
        log(f"   ⚠ {precision.upper()} impossible ici — export en FP32.")
    model = model.to(dev)
    if prec == "fp16":
        model = model.half()
    dt = torch.float16 if prec == "fp16" else torch.float32
    fixed = _parse_size(opts.get("onnx_size"))
    # Not a multiple of 16 on purpose: traces the arch's own reflect-pad branch, so the
    # dynamic graph also accepts sizes that need padding.
    w, h = fixed if fixed else (76, 68)
    sfx = ("_fp16" if prec == "fp16" else "") + (f"_{w}x{h}" if fixed else "")
    dyn = not fixed

    def _ort_check(path, feeds):
        try:
            import onnxruntime as ort
            so = ort.SessionOptions(); so.log_severity_level = 4
            s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            s.run(None, feeds)
            log("   ✓ ONNX vérifié avec onnxruntime à une autre taille.")
        except ImportError:
            pass
        except Exception as e:
            log(f"   ⚠ L'ONNX dynamique refuse une autre taille ({type(e).__name__}) — "
                f"ré-exportez avec une « taille fixe » (ex. 960x540).")

    npdt = np.float16 if prec == "fp16" else np.float32
    with torch.no_grad():
        if arch == "motsr":
            lr = torch.rand(1, 3, h, w, device=dev, dtype=dt)
            hr = model.image_only(lr)

            class Seed(torch.nn.Module):
                def __init__(s):
                    super().__init__(); s.m = model

                def forward(s, frame):
                    return s.m.image_only(frame)

            class Recurrent(torch.nn.Module):
                def __init__(s):
                    super().__init__(); s.m = model

                def forward(s, a, b, c, d, e, f, g):
                    return s.m.forward_step(a, b, c, d, e, f, g)

            seed_p = os.path.join(out_dir, f"{base}{sfx}_seed.onnx")
            rec_p = os.path.join(out_dir, f"{base}{sfx}_recurrent.onnx")
            kw = dict(opset_version=max(opset, 17), dynamo=False)
            torch.onnx.export(Seed(), lr, seed_p, input_names=["input"], output_names=["output"],
                              dynamic_axes={"input": {2: "height", 3: "width"},
                                            "output": {2: "out_height", 3: "out_width"}} if dyn else None, **kw)
            names = ["older_lr", "older_hr", "previous_lr", "previous_hr", "current_lr", "next_lr", "later_lr"]
            axes = {n: {2: "hr_height" if i in (1, 3) else "lr_height",
                        3: "hr_width" if i in (1, 3) else "lr_width"} for i, n in enumerate(names)}
            axes["output"] = {2: "out_height", 3: "out_width"}
            torch.onnx.export(Recurrent(), (lr, hr, lr, hr, lr, lr, lr), rec_p, input_names=names,
                              output_names=["output"], dynamic_axes=axes if dyn else None, **kw)
            log(f"  ✅ {os.path.basename(seed_p)}  (1re frame de chaque scène)")
            log(f"  ✅ {os.path.basename(rec_p)}  (7 entrées : older/previous LR+HR sortie, current, next, later)")
            if dyn:
                l2 = np.random.rand(1, 3, 72, 96).astype(npdt)
                h2 = np.random.rand(1, 3, 144, 192).astype(npdt)
                _ort_check(rec_p, dict(zip(names, [l2, h2, l2, h2, l2, l2, l2])))
            return True
        wrap = _ClipStack.make(model, clip)
        dummy = torch.rand(1, clip * 3, h, w, device=dev, dtype=dt)
        p = os.path.join(out_dir, f"{base}_1x{clip * 3}x{'HxW_dyn' if dyn else f'{h}x{w}'}{sfx}.onnx")
        torch.onnx.export(wrap, dummy, p, opset_version=opset, dynamo=False,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {2: "height", 3: "width"},
                                        "output": {2: "out_height", 3: "out_width"}} if dyn else None)
        log(f"  ✅ {os.path.basename(p)}  (entrée 1x{clip * 3}xHxW = {clip} frames empilées, "
            f"sortie = frame centrale — format vs-mlrt / VapourKit)")
        if dyn:
            _ort_check(p, {"input": np.random.rand(1, clip * 3, 96, 128).astype(npdt)})
    return True


def convert(pth: str, out_dir: str, opts: dict, log=None) -> bool:
    """Convert `pth` to the formats enabled in `opts`. Returns True on overall success.

    Precision model: fp16/bf16 are NOT separate outputs — they are the *precision*
    applied to every selected format. So:
        pth/onnx/safetensors/ncnn/trt selected, nothing else → FP32
        + fp16 → that format in FP16   ·   + bf16 → that format in BF16
    opts keys (bool): pth, onnx, safetensors, ncnn, trt, fp16, bf16, tf32
    opts: opset (int), arch (str), scale (str)
    """
    def _log(m):
        if log:
            log(m)

    if not os.path.isfile(pth):
        _log("❌ Modèle source introuvable.")
        return False
    if str(opts.get("onnx_size") or "").strip() and not _parse_size(opts.get("onnx_size")):
        _log(f"❌ Taille fixe ONNX invalide : « {opts.get('onnx_size')} » (attendu : 960x540, ou vide "
             "pour une taille dynamique).")
        return False
    os.makedirs(out_dir, exist_ok=True)

    try:
        import torch
    except ImportError:
        # Portable exe: no torch in-process — run this same module in the engine venv.
        return _convert_in_venv(pth, out_dir, opts, _log)

    # Precision (fp16 wins over bf16 if both ticked) + suffix.
    precision = "fp16" if opts.get("fp16") else ("bf16" if opts.get("bf16") else "fp32")
    psfx = "" if precision == "fp32" else f"_{precision}"
    # Real output formats (fp16/bf16 are precision flags, not formats).
    reals = [f for f in ("pth", "safetensors", "onnx", "ncnn", "trt") if opts.get(f)]
    if not reals:
        reals = ["pth"]
    _log(f"Précision : {precision.upper()}  ·  Formats : {', '.join(reals)}")

    # TF32 = faster matmul on Ampere+ during the conversion compute (not a storage dtype).
    try:
        _tf32 = bool(opts.get("tf32"))
        torch.backends.cuda.matmul.allow_tf32 = _tf32
        torch.backends.cudnn.allow_tf32 = _tf32
        if _tf32:
            _log("TF32 activé (accélère le calcul de conversion sur RTX 3000+).")
    except Exception:
        pass

    _log(f"Chargement : {os.path.basename(pth)}")
    try:
        state = _load_state(pth)
    except Exception as e:
        _log(f"❌ Chargement échoué : {e}")
        return False
    if not isinstance(state, dict):
        _log("❌ state_dict introuvable.")
        return False

    base = os.path.splitext(os.path.basename(pth))[0]
    ok = True
    cast = _cast_state(state, precision)
    _src = os.path.normcase(os.path.abspath(pth))

    def _dest(name):
        """Output path; never the source file itself (same folder + same format + FP32
        gave the same name, and the save replaced the original)."""
        p = os.path.join(out_dir, name)
        if os.path.normcase(os.path.abspath(p)) == _src:
            stem, ext = os.path.splitext(name)
            p = os.path.join(out_dir, f"{stem}_converted{ext}")
        return p

    # ── weight-only formats ──
    if "pth" in reals:
        _log(f"→ Sauvegarde PTH ({precision})…")
        try:
            _p = _dest(f"{base}{psfx}.pth")
            torch.save(cast, _p)
            _log(f"  ✅ {os.path.basename(_p)}")
        except Exception as e:
            ok = False; _log(f"  ❌ {e}")

    if "safetensors" in reals:
        _log(f"→ Sauvegarde SafeTensors ({precision})…")
        try:
            from safetensors.torch import save_file
            # .clone() detaches from any mmap of the source (Windows os error 1224);
            # keep the chosen precision (don't force float()).
            safe = {k: v.contiguous().clone() for k, v in cast.items()}
            _p = _dest(f"{base}{psfx}.safetensors")
            save_file(safe, _p)
            _log(f"  ✅ {os.path.basename(_p)}")
        except Exception as e:
            ok = False; _log(f"  ❌ {e}")

    # ── graph formats (ONNX → NCNN/TRT) ──
    if any(f in reals for f in ("onnx", "ncnn", "trt")):
        opset = int(opts.get("opset", 17))
        _log(f"→ Export ONNX (opset {opset})…")
        seq = _sequence_arch(state)
        try:
            if seq:
                _log(f"   arch vidéo : {seq}")
                ok = _export_sequence_onnx(seq, state, base, out_dir, opts, precision, opset, _log) and ok
                model = None
            else:
                model, _scale = _build_model(pth, state, opts, _log)
            if seq:
                pass
            elif model is None:
                ok = False
            else:
                use_cuda = torch.cuda.is_available()
                dev = "cuda" if use_cuda else "cpu"
                model = model.to(dev)
                # ONNX precision: fp16 supported; bf16 isn't well supported by ONNX → fp32.
                onnx_prec = precision
                if precision == "bf16":
                    _log("   ⚠ ONNX ne supporte pas BF16 — export en FP32.")
                    onnx_prec = "fp32"
                if onnx_prec == "fp16" and not use_cuda:
                    _log("   ⚠ FP16 ONNX nécessite CUDA — export en FP32 (CPU).")
                    onnx_prec = "fp32"
                if onnx_prec == "fp16":
                    model = model.half()
                # Fixed size ("960x540"): static graph at exactly that resolution. Needed
                # for window-attention archs (FDAT, SRFormer, SwinIR, HAT…): the legacy
                # tracer freezes their pad/reshape shapes at the dummy size, so a
                # "dynamic" export only really works at 64x64.
                fixed = _parse_size(opts.get("onnx_size"))
                if fixed:
                    w, h = fixed
                    _log(f"   Taille d'entrée fixe : {w}x{h} (graphe statique)")
                else:
                    w, h = 64, 64
                dummy = torch.randn(1, 3, h, w, device=dev)
                if onnx_prec == "fp16":
                    dummy = dummy.half()
                suffix = ("_fp16" if onnx_prec == "fp16" else "") + (f"_{w}x{h}" if fixed else "")
                onnx_path = os.path.join(out_dir, f"{base}{suffix}.onnx")
                export_kwargs = dict(opset_version=opset, dynamo=False,
                                     input_names=["input"], output_names=["output"])
                if not fixed:
                    export_kwargs["dynamic_axes"] = {"input": {0: "b", 2: "h", 3: "w"},
                                                     "output": {0: "b", 2: "h", 3: "w"}}
                # no_grad: tracing with autograd on keeps every activation alive —
                # FDAT at 960x540 asked for 25 GB of VRAM without it, ~3 GB with it.
                with torch.no_grad():
                    torch.onnx.export(model, dummy, onnx_path, **export_kwargs)
                _log(f"  ✅ {os.path.basename(onnx_path)}")
                if not fixed:
                    _check_dynamic_onnx(onnx_path, onnx_prec, _log)
        except Exception as e:
            ok = False
            import traceback
            _log(f"  ❌ ONNX: {e}")
            _log(traceback.format_exc()[-400:])
        if "ncnn" in reals:
            _log("⚠ NCNN : aucun fichier .param/.bin généré ici — convertir le .onnx "
                 "ci-dessus avec onnx2ncnn (outil externe, non inclus).")
        if "trt" in reals:
            _log("⚠ TensorRT : aucun fichier .engine généré ici — convertir le .onnx "
                 "ci-dessus avec trtexec --onnx=model.onnx --saveEngine=model.trt "
                 "(SDK TensorRT requis, non inclus).")

    # Was logged unconditionally regardless of `ok` — a failed export (e.g. ONNX
    # exception above) still showed "✅ Conversion terminée." right under the ❌ error
    # (2026-09-17 fix).
    if ok:
        _log("✅ Conversion terminée.")
    else:
        _log("❌ Conversion terminée avec des erreurs (voir ci-dessus).")
    return ok


def _convert_in_venv(pth, out_dir, opts, log) -> bool:
    """Run convert() in the engine venv (the portable GUI has no torch) and stream its log."""
    import json
    import subprocess
    import tempfile
    try:
        from src.core import engine_paths as ep
        py = ep.resolve_engine_python(ep.redux_path())
    except Exception as e:
        log(f"❌ PyTorch indisponible et moteur introuvable ({e}). Installez le moteur traiNNer-redux.")
        return False
    if not py or not os.path.isfile(py):
        log("❌ PyTorch indisponible : installez le moteur traiNNer-redux (onglet Paramètres).")
        return False
    with tempfile.TemporaryDirectory(prefix="usrs_conv_") as td:
        args = os.path.join(td, "args.json")
        with open(args, "w", encoding="utf-8") as f:
            json.dump({"pth": pth, "out_dir": out_dir, "opts": opts}, f)
        proc = subprocess.Popen([py, os.path.abspath(__file__), args], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result = False
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("RESULT "):
                result = line[7:].strip() == "1"
            elif line.strip() and "Warning" not in line and not line.startswith("  warnings.warn"):
                log(line)
        proc.wait()
        return result and proc.returncode == 0


if __name__ == "__main__":
    import json
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _here = os.path.dirname(os.path.abspath(__file__))
    # Append (never prepend): in the portable build the folder holding src/ is
    # PyInstaller's _internal/, whose own extension modules would shadow the venv's.
    for _p in (_here, os.path.dirname(os.path.dirname(_here))):
        if _p not in sys.path:
            sys.path.append(_p)
    with open(sys.argv[1], encoding="utf-8") as _f:
        _a = json.load(_f)
    try:
        from src.core import engine_paths as _ep
        _r = _ep.redux_path()
        if os.path.isdir(_r) and _r not in sys.path:
            sys.path.append(_r)
    except Exception:
        pass
    _ok = convert(_a["pth"], _a["out_dir"], _a["opts"], log=lambda m: print(m, flush=True))
    print("RESULT " + ("1" if _ok else "0"), flush=True)
