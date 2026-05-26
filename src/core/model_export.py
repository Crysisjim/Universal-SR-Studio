"""
model_export.py — Export trained models to formats compatible with
inference apps like video2x, QualityScaler, RealCUGAN, IEU.

Supported targets:
- ONNX (universal, used by QualityScaler, video2x via NVIDIA TensorRT)
- TorchScript (.pt) — broader compatibility
- safetensors (.safetensors) — secure, faster loading
"""
import os
import sys
import json


def export_to_onnx(pth_path: str, onnx_path: str, scale: int = 4,
                   patch_size: int = 64, opset: int = 17) -> tuple:
    """
    Export a .pth checkpoint to ONNX format.

    Returns (success: bool, message: str)
    """
    if not os.path.exists(pth_path):
        return (False, f"Fichier .pth non trouve: {pth_path}")
    try:
        import torch
    except ImportError:
        return (False, "PyTorch non disponible. pip install torch")

    try:
        # Load the checkpoint
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            if "params" in ckpt:
                state_dict = ckpt["params"]
            elif "params_ema" in ckpt:
                state_dict = ckpt["params_ema"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # We can't always reconstruct the architecture without knowing the type.
        # User must provide a load_model function or we use generic export.
        return (False, "Export ONNX necessite l'architecture exacte. "
                       "Utilisez les outils de NeoSR (scripts/) pour cette etape.")

    except Exception as e:
        return (False, f"Erreur: {e}")


def convert_pth_to_safetensors(pth_path: str, st_path: str) -> tuple:
    """Convert .pth to .safetensors (secure, faster)."""
    if not os.path.exists(pth_path):
        return (False, "Fichier source non trouve")
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        return (False, f"Dependances manquantes: {e}\n  pip install torch safetensors")

    try:
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
        # Extract state_dict
        if isinstance(ckpt, dict):
            if "params" in ckpt:
                sd = ckpt["params"]
            elif "params_ema" in ckpt:
                sd = ckpt["params_ema"]
            elif "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt

        # Filter only tensor values
        sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

        save_file(sd, st_path)
        size_mb = os.path.getsize(st_path) / (1024 * 1024)
        return (True, f"OK — {len(sd)} tenseurs, {size_mb:.1f} MB")
    except Exception as e:
        return (False, f"Erreur conversion: {e}")


def detect_model_format(model_path: str) -> dict:
    """
    Inspect a model file and return its format/architecture info.

    Returns dict with: format, num_params, total_params, architecture_hints,
    inferred_scale, layer_breakdown, etc.
    """
    if not os.path.exists(model_path):
        return {"error": "Fichier non trouve"}

    ext = os.path.splitext(model_path)[1].lower()
    info = {"format": ext, "size_mb": os.path.getsize(model_path) / (1024 * 1024)}

    try:
        if ext == ".onnx":
            try:
                import onnx
                model = onnx.load(model_path)
                info["onnx_version"] = model.ir_version
                info["opset"] = [op.version for op in model.opset_import]
                info["inputs"] = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                                   for i in model.graph.input]
                info["outputs"] = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                                    for o in model.graph.output]
            except ImportError:
                info["error"] = "onnx non installe (pip install onnx)"
        elif ext == ".pth" or ext == ".pt":
            import torch
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict):
                # Try to find the actual state_dict
                if "params" in ckpt:
                    sd = ckpt["params"]
                    info["wrapper"] = "neosr/basicsr (params)"
                elif "params_ema" in ckpt:
                    sd = ckpt["params_ema"]
                    info["wrapper"] = "neosr/basicsr (params_ema)"
                elif "state_dict" in ckpt:
                    sd = ckpt["state_dict"]
                    info["wrapper"] = "PyTorch state_dict"
                else:
                    sd = ckpt
                    info["wrapper"] = "raw"
                # Capture other top-level keys (iteration count, optim, etc.)
                if isinstance(ckpt, dict) and any(k != "params" for k in ckpt):
                    other = {k: type(v).__name__ for k, v in ckpt.items()
                             if k not in ("params", "params_ema", "state_dict")}
                    if other:
                        info["other_keys"] = other
            else:
                sd = ckpt
                info["wrapper"] = "raw"

            # Count params + per-layer breakdown
            total = 0
            tensors = 0
            arch_keys = []
            layer_groups = {}  # prefix -> (count, total_params)
            param_dtypes = {}
            for k, v in sd.items():
                if hasattr(v, "numel"):
                    n = v.numel()
                    total += n
                    tensors += 1
                    if tensors <= 5:
                        arch_keys.append(k)
                    # Group by top-level prefix (e.g. "body.0", "model.1.sub")
                    parts = k.split(".")
                    if len(parts) >= 2:
                        prefix = parts[0]
                    else:
                        prefix = k
                    if prefix not in layer_groups:
                        layer_groups[prefix] = [0, 0]
                    layer_groups[prefix][0] += 1
                    layer_groups[prefix][1] += n
                    # dtype
                    if hasattr(v, "dtype"):
                        dt = str(v.dtype)
                        param_dtypes[dt] = param_dtypes.get(dt, 0) + 1

            info["total_params"] = total
            info["num_tensors"] = tensors
            info["sample_keys"] = arch_keys
            info["layer_groups"] = sorted(
                [(k, c, p) for k, (c, p) in layer_groups.items()],
                key=lambda x: -x[2]
            )[:8]
            info["dtypes"] = param_dtypes

            # Architecture hints from key names (case-insensitive substring + structural patterns)
            keys_list = list(sd.keys())
            keys_str = " ".join(keys_list).lower()
            sample_keys_str = " ".join(keys_list[:30]).lower()
            hints = []

            # ── Specific architectures (most distinctive first) ──
            if "rrdb" in keys_str or "sub.0.rdb" in keys_str:
                hints.append("RRDBNet (ESRGAN/Real-ESRGAN)")
            if "swin" in keys_str or ("patch_embed" in keys_str and "rstb" in keys_str):
                hints.append("SwinIR")
            if "rcab" in keys_str or "rcan" in keys_str:
                hints.append("RCAN")
            if "omni" in keys_str:
                hints.append("OmniSR")
            if any(k in keys_str for k in ["hat.", "hab.", "ocab"]):
                hints.append("HAT")
            if "drct" in keys_str:
                hints.append("DRCT")
            if "span" in keys_str or "spab" in keys_str:
                hints.append("SPAN")
            if "compact" in keys_str or "realesr-general" in keys_str:
                hints.append("RealESRGAN-Compact")
            if "atd" in keys_str:
                hints.append("ATD")
            if "plksr" in keys_str:
                hints.append("PLKSR")
            if "dat" in keys_str and "spatial_block" in keys_str:
                hints.append("DAT")
            if "blocks_2.0.body" in keys_str and "upsampler.MetaUpsample" in keys_str:
                hints.append("SMoSR (Self-Modulate SR)")
            if "conv_near.weight" in keys_str and "block_5.conv1" in keys_str:
                hints.append("SpanF (SPAN simplifié, SPAB1 blocks)")
            if "MetaIGConv" in keys_str and "upsampler.coord_map" in keys_str:
                hints.append("SpanC (multi-scale IGConv, reparamétrisable)")
            if "gfisr_body.0.fc1" in keys_str and "upscale.MetaUpsample" in keys_str:
                hints.append("GFISRv2 (GatedCNN + Fourier-inspired)")
            if "blocks.0.0.irca_attn.to_k.weight" in keys_str and "first_conv.weight" in keys_str:
                hints.append("CATANet (Token Aggregation + LRSA, NeoSR)")

            # ── Generic patterns when architecture-specific keys are absent ──
            if not hints:
                if "model.0.weight" in keys_str and "model.1.sub" in keys_str:
                    hints.append("RRDBNet (ESRGAN old-style — model.X.sub.X.RDB...)")
                elif "model.0.weight" in keys_str and "model.1.weight" in keys_str:
                    if "body" in keys_str:
                        hints.append("EDSR/SRResNet (sequential 'body')")
                    else:
                        hints.append("Sequential (MSRResNet/SRResNet ESRGAN-light)")
                elif "body.0.weight" in keys_str and "body.1.weight" in keys_str:
                    hints.append("EDSR/MSRResNet (body.N.weight)")
                elif "conv_first" in keys_str and "upconv" in keys_str:
                    hints.append("ESRGAN-style (conv_first + upconv)")
                elif "head" in keys_str and "tail" in keys_str:
                    hints.append("RCAN-like (head/body/tail)")

            # ── Detect upscale factor from upsampling layers ──
            inferred_scale = None
            try:
                # Look for a final pixelshuffle / upconv layer to deduce scale
                for k, v in sd.items():
                    if hasattr(v, "shape"):
                        # Conv weights with output channels = scale^2 * in_channels suggest pixelshuffle
                        if any(s in k.lower() for s in ["upsampler", "upsample", "upconv1", "tail"]):
                            shape = list(v.shape)
                            if len(shape) == 4:  # Conv2d weight
                                out_ch, in_ch = shape[0], shape[1]
                                # Common: output = scale^2 * 3 (RGB) or scale^2 * in_ch
                                for s in [2, 3, 4, 8]:
                                    if out_ch == 3 * (s ** 2) or out_ch == in_ch * (s ** 2):
                                        inferred_scale = s
                                        break
                            if inferred_scale:
                                break
            except Exception:
                pass

            if inferred_scale:
                info["inferred_scale"] = f"x{inferred_scale}"

            # ── Check for discriminator (GAN networks have D.X keys) ──
            is_discriminator = any(k.startswith(("D.", "net_d.", "discriminator.")) for k in keys_list)
            if is_discriminator:
                hints.append("⚠ Discriminator detected (not a generator)")

            info["architecture_hints"] = hints if hints else ["unknown — voir 'Cles d'exemple'"]
        elif ext == ".safetensors":
            try:
                from safetensors import safe_open
                total = 0
                tensors_count = 0
                with safe_open(model_path, framework="pt") as f:
                    keys = list(f.keys())
                    for k in keys:
                        t = f.get_tensor(k)
                        total += t.numel()
                        tensors_count += 1
                info["num_tensors"] = tensors_count
                info["total_params"] = total
                info["sample_keys"] = keys[:5]
            except ImportError:
                info["error"] = "safetensors non installe (pip install safetensors)"
    except Exception as e:
        info["error"] = str(e)

    return info


def format_model_info(info: dict) -> str:
    """Format detect_model_format output as detailed readable text."""
    if "error" in info:
        return f"❌ {info['error']}"
    lines = []
    lines.append("═══ FICHIER ═══")
    lines.append(f"  Format        : {info.get('format', '?')}")
    lines.append(f"  Taille        : {info.get('size_mb', 0):.2f} MB")
    if "wrapper" in info:
        lines.append(f"  Wrapper       : {info['wrapper']}")

    if "total_params" in info:
        params_m = info["total_params"] / 1_000_000
        lines.append("")
        lines.append("═══ MODELE ═══")
        lines.append(f"  Parametres    : {params_m:.3f} M ({info['total_params']:,})")
        lines.append(f"  Tenseurs      : {info['num_tensors']}")
        # Estimated VRAM at FP32
        vram_mb_fp32 = (info["total_params"] * 4) / (1024 * 1024)
        lines.append(f"  VRAM ~ FP32   : {vram_mb_fp32:.1f} MB (poids seulement)")
        if vram_mb_fp32 < 100:
            lines.append(f"                  ↳ Modele leger, OK pour real-time inference")
        elif vram_mb_fp32 < 500:
            lines.append(f"                  ↳ Modele moyen, GPU recommande pour inference rapide")
        else:
            lines.append(f"                  ↳ Modele lourd, GPU haut de gamme requis")

    if "architecture_hints" in info:
        lines.append("")
        lines.append("═══ ARCHITECTURE ═══")
        lines.append(f"  Detection     : {', '.join(info['architecture_hints'])}")
        if info.get("inferred_scale"):
            lines.append(f"  Scale infere  : {info['inferred_scale']}")

    if "dtypes" in info and info["dtypes"]:
        lines.append("")
        lines.append("═══ PRECISION ═══")
        for dt, count in info["dtypes"].items():
            lines.append(f"  {dt:20s}: {count} tenseurs")

    if "layer_groups" in info and info["layer_groups"]:
        lines.append("")
        lines.append("═══ TOP MODULES (par nb de parametres) ═══")
        for prefix, count, params in info["layer_groups"]:
            params_m = params / 1_000_000
            pct = 100.0 * params / info["total_params"] if info.get("total_params") else 0
            lines.append(f"  {prefix:20s}: {count:4d} tenseurs, {params_m:6.3f} M ({pct:5.1f}%)")

    if "sample_keys" in info:
        lines.append("")
        lines.append("═══ CLES D'EXEMPLE ═══")
        for k in info["sample_keys"]:
            lines.append(f"  {k}")

    if "other_keys" in info and info["other_keys"]:
        lines.append("")
        lines.append("═══ METADATA CHECKPOINT ═══")
        for k, v in info["other_keys"].items():
            lines.append(f"  {k:20s}: {v}")

    if "onnx_version" in info:
        lines.append("")
        lines.append("═══ ONNX ═══")
        lines.append(f"  IR version    : {info['onnx_version']}")
        lines.append(f"  Opset         : {info.get('opset', '?')}")
        if "inputs" in info:
            lines.append(f"  Inputs        :")
            for n, s in info["inputs"]:
                lines.append(f"    - {n}: {s}")
        if "outputs" in info:
            lines.append(f"  Outputs       :")
            for n, s in info["outputs"]:
                lines.append(f"    - {n}: {s}")

    return "\n".join(lines)
