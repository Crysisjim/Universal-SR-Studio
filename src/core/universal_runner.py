"""
universal_runner.py - General-purpose SR inference subprocess.

Tries spandrel first (handles SPAN, ESRGAN, RealPLKSR, SwinIR, etc.).
Falls back to manual SpanPlus instantiation for traiNNer-redux models
that spandrel does not yet support.

Usage:
    <traiNNer_venv_python> universal_runner.py <model_path> <input_path> <output_path>

Exit 0 = success, 1 = error. All output to stdout (UTF-8).
"""
import sys
import os
import re
import math
import traceback

# Force UTF-8 so Unicode chars don't crash on cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    print(f"[Runner] ERREUR import: {e}", flush=True)
    sys.exit(1)


def _load_state_dict(model_path: str) -> dict:
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(model_path, device="cpu")
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    for key in ("params_ema", "params_g", "params", "model", "state_dict"):
        if key in ck:
            return ck[key]
    if any(k.endswith(".weight") for k in ck.keys()):
        return ck
    return ck


def run(model_path: str, input_path: str, output_path: str,
        tile_size: int = 0, tile_pad: int = 32, use_amp: bool = False,
        scale_hint: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Runner] Device : {device} | tile={tile_size}px | amp={use_amp} | scale_hint={scale_hint}", flush=True)

    model = None
    scale = 1

    # ── 0. Peek keys: RCAN scale-1 (deband) → spandrel mis-detects n_feats from the
    #    absent pixelshuffle tail.0 → noisy size-mismatch dump. Skip spandrel for it. ──
    skip_spandrel = False
    try:
        if model_path.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(model_path, framework="pt") as _f:
                _peek = set(_f.keys())
        else:
            _peek = set(_load_state_dict(model_path).keys())
        if "groups.0.blocks.2.attn.dictionary" in _peek and "groups.0.blocks.0.attn.rel_pos_bias" in _peek:
            skip_spandrel = True
            print("[Runner] FDAT-RD détecté — spandrel ignoré (le prendrait pour un FDAT).", flush=True)
        if ("body.0.body.0.body.3.conv_du.0.weight" in _peek
                and "tail.1.weight" in _peek
                and not any(k.startswith("tail.0.") for k in _peek)):
            skip_spandrel = True
            print("[Runner] RCAN 1x détecté — spandrel ignoré (construction manuelle).", flush=True)
    except Exception:
        pass

    # ── 1. Try spandrel (handles SPAN, ESRGAN, RealPLKSR, SwinIR, HAT, …) ──
    if not skip_spandrel:
        try:
            import spandrel
            # Without install(), every spandrel_extra_arches arch (SRFormer, …) is
            # rejected with an empty UnsupportedModelError, and the fallback cascade then
            # rebuilt it with the wrong runner/scale.
            try:
                import spandrel_extra_arches
                spandrel_extra_arches.install()
            except Exception:
                pass
            print(f"[Runner] Chargement via spandrel : {os.path.basename(model_path)}", flush=True)
            loader = spandrel.ModelLoader(device=device)
            descriptor = loader.load_from_file(model_path)
            model = descriptor.model
            scale = descriptor.scale
            print(f"[Runner] spandrel OK : {descriptor.architecture.name}, scale={scale}x", flush=True)
        except Exception as e:
            if type(e).__name__ == "UnsupportedModelError":
                print("[Runner] Architecture non reconnue par spandrel, tentative manuelle...", flush=True)
            else:
                print(f"[Runner] spandrel echec ({type(e).__name__}: {e}), tentative manuelle...", flush=True)

    # ── 1b. HEART / NERVE (traiNNer-redux master) — hyperparams read from the weights ──
    if model is None:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import arch_infer
            _sd = _load_state_dict(model_path)
            _ai = arch_infer.detect(_sd)
            if _ai == "motsr":
                print("[Runner] MoTSR est un modèle VIDÉO (5 frames + récurrence) : utilisez "
                      "l'entrée Vidéo ou un dossier de frames dans Quick Upscale.", flush=True)
            elif _ai:
                model, scale, _kw = arch_infer.build(_ai, _sd)
                print(f"[Runner] {_ai.upper()} (traiNNer-redux) : "
                      + ", ".join(f"{k}={v}" for k, v in _kw.items()), flush=True)
        except Exception as e:
            print(f"[Runner] HEART/NERVE : chargement échoué ({type(e).__name__}: {str(e)[:300]})",
                  flush=True)
            model = None

    # ── 2. Fallback: manual traiNNer-redux exotic archs (not yet in spandrel) ──
    if model is None:
        try:
            state_dict = _load_state_dict(model_path)

            # Détection arch depuis les clés du state_dict
            detected = None
            if "feats.0.eval_conv.weight" in state_dict and "upsampler.offset.weight" in state_dict:
                detected = "spanplus"
            elif "blocks_2.0.body" in " ".join(state_dict.keys()) and "upsampler.MetaUpsample" in state_dict:
                detected = "smosr"
            elif "cat_to_dim.weight" in state_dict and "gfisr_body_half.0.fc1.weight" in state_dict:
                detected = "figsr"
            elif "gfisr_body.0.fc1.weight" in state_dict and "upscale.MetaUpsample" in state_dict:
                detected = "gfisrv2"
            elif "block_1.c1_r.conv3.eval_conv.weight" in state_dict and "upsampler.amplitude" in state_dict:
                detected = "spanpp"
            elif "block_1.conv_a.eval_conv.weight" in state_dict and "MetaIGConv" in state_dict:
                detected = "spanc"
            elif ("block_1.c1_r.eval_conv.weight" in state_dict
                  and "conv_near.weight" in state_dict
                  and "conv_cat.weight" in state_dict):
                detected = "spanf"
            elif "feats.0.eval_conv.weight" in state_dict or "feats.0.sk.weight" in state_dict:
                detected = "spanplus"  # SPANPlus sans DySample (1x conv upsampler)
            elif ("body.0.body.0.body.3.conv_du.0.weight" in state_dict
                  and "tail.1.weight" in state_dict
                  and ("head.0.weight" in state_dict or "head.1.weight" in state_dict)):
                # RCAN — spandrel mis-detects n_feats for scale-1 (deband) models
                # because it reads n_feats from the absent pixelshuffle tail.0.
                detected = "rcan"
            elif ("conv_fuse.weight" in state_dict
                  and any(k.startswith("magic_upsampler.") for k in state_dict)):
                # ParagonSR v1 (Phhofm) — Magic Kernel Sharp upsampler signature
                detected = "paragonsr"
            elif ("detail_gain" in state_dict
                  and "conv_in.weight" in state_dict
                  and any(k.startswith("body.") for k in state_dict)):
                # ParagonSR v2 (Phhofm) — dual-path architecture
                detected = "paragonsr2"
            elif any(".self_attention.x_proj_weight" in k for k in state_dict):
                # MambaIR (ECCV2024) — VSSBlock's SS2D signature
                detected = "mambair"
            elif any(".embeddingA.weight" in k for k in state_dict) and \
                    any(".assm.selectiveScan.x_proj_weight" in k for k in state_dict):
                # MambaIRv2 (CVPR2025) — ASSM/AttentiveLayer signature
                detected = "mambairv2"
            elif "depth_to_space.upscale.0.weight" in state_dict and "res_block.0.conv.0.0.weight" in state_dict:
                detected = "artcnn"
            elif any(k.startswith("layers.layers.") and ".FDB.c1_d.weight" in k for k in state_dict):
                detected = "lkfmixer"
            elif "to_img.0.conv2.weight" in state_dict and any(k.startswith("body.") and ".fc1." in k for k in state_dict) \
                    and any(".conv.0.poll.1." in k for k in state_dict):
                detected = "rtmosr"
            elif "to_img.MetaUpsample" in state_dict and any(k.startswith("gblocks.") for k in state_dict):
                detected = "mosrv2"
            elif "tail.0.weight" in state_dict and any(".channel_attention_forward." in k for k in state_dict):
                detected = "sebica"
            elif "to_feat.weight" in state_dict and any(re.match(r"^feats\.\d+\.norm1\.weight$", k) for k in state_dict) \
                    and any(".lhsb." in k for k in state_dict):
                detected = "lmlt"
            elif any(re.match(r"^body\.\d+\.body_t\.", k) for k in state_dict) and "tail.0.weight" in state_dict \
                    and "sub_mean.weight" in state_dict:
                detected = "man"
            elif any(re.match(r"^norm\d+\.weight$", k) for k in state_dict) and any(k.startswith("block1.") and ".mlp.DFFM." in k for k in state_dict):
                detected = "eimn"
            elif "head.0.weight" in state_dict and "tail.0.weight" in state_dict \
                    and any(".modules_lfe.lfe_0." in k for k in state_dict) \
                    and any(".modules_gmsa.gmsa_0." in k for k in state_dict):
                detected = "elan"
            elif "in_to_dim.weight" in state_dict and "sisr_cat_conv.weight" in state_dict:
                # GateRV3 — SISR span branch + gated U-Net (in_to_dim/sisr_cat_conv signature)
                detected = "gaterv3"
            elif any(".ngram_window_partition.ngram_context.unigram_embed.weight" in k for k in state_dict):
                # HiT-SNG — NGram context window partition (unique to SNG variant)
                detected = "hit_sng"
            elif any(".correlation.qv.linear.weight" in k for k in state_dict) and "conv_first.weight" in state_dict:
                # HiT-SIR — Spatial-Channel Correlation (SCC/DFE) signature, no NGram context
                detected = "hit_sir"
            elif ("intro.weight" in state_dict and "ending.weight" in state_dict
                  and any(".GCE.depthwise_separable_convs." in k for k in state_dict)):
                # CascadedGaze — NAFNet-style UNet with GlobalContextExtractor blocks
                detected = "cascadedgaze"
            elif "plk_filter" in state_dict and "to_img.MetaUpsample" in state_dict:
                # ESCRealM — ESC-style plk_filter + UniUpsampleV3 (distinct from neosr's "esc" arch,
                # which never has a to_img.MetaUpsample buffer)
                detected = "escrealm"
            elif "sub_mean.weight" in state_dict and "head.weight" in state_dict \
                    and any(".sa_list." in k for k in state_dict):
                # EMT — Efficient Mixed Transformer (TransformerGroup sa_list/mlp_list signature)
                detected = "emt"
            elif "in_to_feat.weight" in state_dict and any(k.startswith("short_cut.") for k in state_dict) \
                    and any(k.startswith("pipeline.") for k in state_dict):
                # FlexNet — short_cut ConvBlock + LinearPipeline/MetaPipeline body
                detected = "flexnet"
            elif "layers.0.residual_group.blocks.0.attn.qkv.weight" in state_dict \
                    and "layers.0.residual_group.blocks.1.attn.kv.weight" in state_dict:
                # SRFormerv2 (traiNNer native) — mixes SwinTransformerBlock (qkv) + PSA_Block (kv/q)
                detected = "srformerv2"
            elif "layers.0.residual_group.blocks.0.attn.kv.weight" in state_dict \
                    and "layers.0.residual_group.blocks.0.attn.q.weight" in state_dict:
                # SRFormer (spandrel_extra_arches-compatible) — all blocks are PSA_Block
                detected = "srformer"
            elif ("patch_embed.proj.weight" in state_dict
                  and "encoder_level1.0.attn.temperature" in state_dict
                  and "refinement.0.attn.temperature" in state_dict):
                # DocRes (Restormer-style document restoration, traiNNer native) — scale fixe 1x
                detected = "docres"

            if detected == "spanplus":
                # blocks / feature_channels / upsampler (dys, ps, conv) from the weights —
                # covers spanplus_s / _st / _sts, which the defaults (blocks=[4]) failed to load
                import arch_infer
                model, scale, _kw = arch_infer.build("spanplus", state_dict)
                print(f"[Runner] SpanPlus manuel : fc={_kw['feature_channels']}, blocks={_kw['blocks']}, "
                      f"scale={scale}x, up={_kw['upsampler']}", flush=True)

            elif detected == "smosr":
                from traiNNer.archs.smosr_arch import SMoSR, SampleMods
                # Construction scale comes from short.weight = in_ch * scale^2 (reliable).
                # MetaUpsample[2] is doubled for this arch (self.scale = scale*2) — don't use it.
                _short = state_dict.get("short.weight")
                _cscale = max(1, int(round(math.sqrt(_short.shape[0] / 3.0)))) if _short is not None else 2
                # dim from upsampler input: in = dim + in_ch*scale^2
                _upw = state_dict.get("upsampler.0.eval_conv.weight")
                if _upw is not None:
                    dim = int(_upw.shape[1]) - 3 * _cscale * _cscale
                else:
                    w = state_dict.get("blocks_1.0.body.0.W")
                    dim = int(w.shape[0]) if w is not None else 48
                # n_mb: distinct blocks_2.N indices
                _idx = set()
                for _k in state_dict:
                    _mm = re.match(r"blocks_2\.(\d+)\.", _k)
                    if _mm:
                        _idx.add(int(_mm.group(1)))
                n_mb = len(_idx) if _idx else 3
                _rep = any(".eval_conv." in _k for _k in state_dict)
                _up = "pixelshuffledirect"
                _mid = 32
                meta = state_dict.get("upsampler.MetaUpsample")
                if meta is not None:
                    try:
                        _up = list(SampleMods.__args__)[int(meta[1].item())]
                        _mid = int(meta[5].item())
                    except Exception:
                        pass
                model = SMoSR(dim=dim, scale=max(1, _cscale), rep=_rep, n_mb=n_mb,
                              upsampler=_up, upsampler_mid_dim=_mid)
                model.load_state_dict(state_dict, strict=False)
                # Real upscale = _cscale (model.scale attribute is doubled/misleading for SMoSR)
                scale = scale_hint if scale_hint > 0 else _cscale
                print(f"[Runner] SMoSR manuel : dim={dim}, n_mb={n_mb}, scale={scale}x, "
                      f"rep={_rep}, up={_up}", flush=True)

            elif detected == "gfisrv2":
                from traiNNer.archs.gfisrv2_arch import GFISRV2
                w = state_dict.get("in_to_dim.weight")
                dim = int(w.shape[0]) if w is not None else 48
                n_blocks = len([k for k in state_dict if k.startswith("gfisr_body.") and k.endswith(".fc1.weight")])
                n_blocks = n_blocks if n_blocks > 0 else 24
                _msc = 0
                meta = state_dict.get("upscale.MetaUpsample")
                if meta is not None:
                    try:
                        _msc = max(1, int(meta[2].item()))
                    except Exception:
                        _msc = 0
                scale = scale_hint if scale_hint > 0 else (_msc if _msc > 0 else 4)
                model = GFISRV2(scale=max(1, scale), dim=dim, n_blocks=n_blocks)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] GFISRv2 manuel : dim={dim}, n_blocks={n_blocks}, scale={scale}x", flush=True)

            elif detected == "figsr":
                from traiNNer.archs.figsr_arch import FIGSR, SampleMods
                w = state_dict.get("in_to_dim.weight")
                dim = int(w.shape[0]) if w is not None else 48
                n_blocks = (len([k for k in state_dict if k.startswith("gfisr_body_half.") and k.endswith(".fc1.weight")])
                            + len([k for k in state_dict if k.startswith("gfisr_body_half_2.") and k.endswith(".fc1.weight")]))
                n_blocks = n_blocks if n_blocks > 0 else 24
                upsampler = "pixelshuffledirect"
                mid_dim = 32
                _msc = 0
                meta = state_dict.get("upscale.MetaUpsample")
                if meta is not None:
                    try:
                        _msc = max(1, int(meta[2].item()))
                        upsampler = list(SampleMods.__args__)[int(meta[1].item())]
                        mid_dim = int(meta[5].item())
                    except Exception:
                        pass
                scale = scale_hint if scale_hint > 0 else (_msc if _msc > 0 else 4)
                model = FIGSR(scale=max(1, scale), dim=dim, n_blocks=n_blocks,
                              upsampler=upsampler, mid_dim=mid_dim)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] FIGSR manuel : dim={dim}, n_blocks={n_blocks}, scale={scale}x, up={upsampler}", flush=True)

            elif detected in ("spanc", "spanpp"):
                from traiNNer.archs.spanpp_arch import SpanC
                meta = state_dict.get("MetaIGConv")
                if meta is not None:
                    scale_list = tuple(int(v.item()) for v in meta)
                else:
                    scale_list = (1, 2) if detected == "spanpp" else (2, 4)
                if detected == "spanpp":
                    w = state_dict.get("block_1.c1_r.conv3.eval_conv.weight")
                else:
                    w = state_dict.get("conv0.eval_conv.weight")
                fc = int(w.shape[0]) if w is not None else 48
                # Detect IGConv params from checkpoint
                _qk_keys = [k for k in state_dict if k.startswith("upsampler.query_kernel.") and k.endswith(".weight")]
                latent_layers = max(4, len(_qk_keys) - 1) if _qk_keys else 4
                _qk0 = state_dict.get("upsampler.query_kernel.0.weight")
                implicit_dim = int(_qk0.shape[0]) if _qk0 is not None else 256
                _freq = state_dict.get("upsampler.freq")
                if _freq is not None:
                    _k2 = _freq.shape[0] / fc
                    ig_k = max(1, int(round(math.sqrt(_k2))))
                else:
                    ig_k = 3
                # Honor user-selected scale if valid, else min (smallest = current auto behavior)
                if scale_hint > 0 and scale_hint in scale_list:
                    eval_base = scale_hint
                else:
                    eval_base = min(scale_list)
                model = SpanC(feature_channels=fc, scale_list=scale_list, eval_base_scale=eval_base,
                              ig_kernel_size=ig_k, implicit_dim=implicit_dim, latent_layers=latent_layers)
                model.load_state_dict(state_dict, strict=False)
                scale = eval_base  # update scale for inference dimensions
                print(f"[Runner] {'SpanPP' if detected == 'spanpp' else 'SpanC'} manuel : fc={fc}, scales={scale_list}, eval={eval_base}x", flush=True)

            elif detected == "spanf":
                from traiNNer.archs.spanf_arch import spanf
                w = state_dict.get("block_1.c1_r.eval_conv.weight")
                fc = int(w.shape[0]) if w is not None else 32
                # conv_2 output = num_out_ch * scale^2 → infer scale (scale_hint overrides)
                _msc = 0
                _c2 = state_dict.get("conv_2.eval_conv.weight")
                if _c2 is not None:
                    _out = _c2.shape[0]
                    _s = int(math.sqrt(_out / 3.0))
                    _msc = _s if _s >= 1 and _s * _s == int(_out / 3.0) else 0
                scale = scale_hint if scale_hint > 0 else (_msc if _msc > 0 else 1)
                model = spanf(num_in_ch=3, num_out_ch=3, feature_channels=fc, scale=max(1, scale))
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] SpanF manuel : fc={fc}, scale={scale}x", flush=True)

            elif detected == "paragonsr":
                # ParagonSR v1 (Phhofm) — conv-first, Magic Kernel Sharp upsampler.
                # Detect ALL params from state_dict to ensure correct shapes.
                # NOTE: paragonsr_anime factory hardcodes num_feat=28 and ignores YAML's
                # num_feat kwarg. Always detect from conv_in.weight, never trust YAML.
                from traiNNer.archs.paragonsr_arch import ParagonSR
                _ci = state_dict.get("conv_in.weight")
                num_feat = int(_ci.shape[0]) if _ci is not None else 28
                # Detect num_groups and num_blocks from body.G.blocks.B.* key structure
                _body_g = set()
                _body_b = set()
                for _k in state_dict:
                    if _k.startswith("body.") and ".blocks." in _k:
                        _parts = _k.split(".")
                        if len(_parts) >= 4:
                            _body_g.add(_parts[1]); _body_b.add(_parts[3])
                num_groups = max(len(_body_g), 1)
                num_blocks  = max(len(_body_b), 1)
                # Detect ffn_expansion from GatedFFN project_in_g shape ratio
                # project_in_g: [hidden, in_ch, 1, 1] where hidden = in_ch * ffn_expansion
                _pig = state_dict.get("body.0.blocks.0.transformer.project_in_g.weight")
                if _pig is not None and _pig.shape[1] > 0:
                    _ffn_expansion = round(_pig.shape[0] / _pig.shape[1], 4)
                else:
                    _ffn_expansion = 1.5  # paragonsr_anime default
                # The scale is not in the weights (parameter-free Magic Kernel upsampler):
                # it comes from the user's choice. `scale` was left at 1 here, so every
                # ParagonSR model ran at 1x.
                scale = scale_hint if scale_hint > 0 else 4
                if scale_hint <= 0:
                    print("[Runner] ParagonSR : scale absent des poids — 4x par défaut, choisissez "
                          "le scale du modèle dans Quick Upscale.", flush=True)
                model = ParagonSR(scale=max(1, scale), num_feat=num_feat,
                                  num_groups=num_groups, num_blocks=num_blocks,
                                  ffn_expansion=_ffn_expansion)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] ParagonSR manuel : feat={num_feat}, groups={num_groups}, "
                      f"blocks={num_blocks}, ffn_exp={_ffn_expansion}, scale={scale}x", flush=True)

            elif detected == "paragonsr2":
                # ParagonSR2 (Phhofm) — dual-path, selective attention.
                # Detect num_feat from conv_in.weight shape.
                from traiNNer.archs.paragonsr2_arch import ParagonSR2
                _ci = state_dict.get("conv_in.weight")
                num_feat = int(_ci.shape[0]) if _ci is not None else 64
                _keys = list(state_dict.keys())
                _keys_str = " ".join(_keys)
                # detect variant from block signature keys first (unambiguous),
                # fall back to num_feat thresholds (photo/pro share num_feat=64 by default).
                # NOTE: check gated_conv (ultimate) BEFORE token_ca (pro) — ultimate blocks
                # also carry a token_ca submodule in their staged curriculum, so checking
                # token_ca first would misclassify ultimate checkpoints as pro.
                if ".gated_conv." in _keys_str:
                    _variant = "ultimate"
                elif ".token_ca." in _keys_str or ".channel_attn." in _keys_str:
                    _variant = "pro"
                elif ".proj." in _keys_str and ".gate." in _keys_str:
                    _variant = "stream"
                else:
                    _variant = "realtime" if num_feat <= 16 else "photo"
                # detect num_groups / num_blocks from body.{g}.blocks.{b} key indices
                _group_idx = set()
                _block_idx = set()
                for k in _keys:
                    m = re.match(r"body\.(\d+)\.blocks\.(\d+)\.", k)
                    if m:
                        _group_idx.add(int(m.group(1)))
                        if int(m.group(1)) == 0:
                            _block_idx.add(int(m.group(2)))
                num_groups = len(_group_idx) if _group_idx else 4
                num_blocks = len(_block_idx) if _block_idx else 4
                # up.0 = conv num_feat -> num_feat * scale^2 before the pixel shuffle.
                _up = state_dict.get("up.0.weight")
                _s2 = 0
                if _up is not None and _up.shape[1] > 0:
                    _r = round(math.sqrt(_up.shape[0] / _up.shape[1]))
                    _s2 = _r if _r >= 1 and _r * _r * _up.shape[1] == _up.shape[0] else 0
                scale = scale_hint if scale_hint > 0 else (_s2 or 4)
                model = ParagonSR2(scale=max(1, scale), num_feat=num_feat,
                                    num_groups=num_groups, num_blocks=num_blocks,
                                    variant=_variant)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] ParagonSR2 manuel : feat={num_feat}, variant={_variant}, "
                      f"groups={num_groups}, blocks={num_blocks}, scale={scale}x", flush=True)

            elif detected == "rcan":
                from spandrel.architectures.RCAN.__arch.rcan_arch import RCAN
                from spandrel.util import get_seq_len
                head_idx = 0 if "head.0.weight" in state_dict else 1
                n_feats = state_dict[f"head.{head_idx}.weight"].shape[0]
                n_colors = state_dict["tail.1.weight"].shape[0]
                kernel_size = state_dict[f"head.{head_idx}.weight"].shape[-1]
                n_resgroups = get_seq_len(state_dict, "body") - 1
                n_resblocks = get_seq_len(state_dict, "body.0.body") - 1
                reduction = n_feats // state_dict["body.0.body.0.body.3.conv_du.0.weight"].shape[0]
                unshuffle_mod = head_idx == 1
                rcan_scale = scale_hint if scale_hint > 0 else 1
                model = RCAN(scale=rcan_scale, n_resgroups=n_resgroups, n_resblocks=n_resblocks,
                             n_colors=n_colors, rgb_range=255,
                             norm=("sub_mean.weight" in state_dict), kernel_size=kernel_size,
                             n_feats=n_feats, reduction=reduction, res_scale=1,
                             act_mode="relu", unshuffle_mod=unshuffle_mod)
                model.load_state_dict(state_dict, strict=True)
                scale = rcan_scale
                print(f"[Runner] RCAN manuel : n_feats={n_feats}, nrg={n_resgroups}, "
                      f"nrb={n_resblocks}, red={reduction}, scale={scale}x", flush=True)

            elif detected == "mambair":
                from traiNNer.archs.mambair_arch import MambaIR
                _cf = state_dict.get("conv_first.weight")
                embed_dim = int(_cf.shape[0]) if _cf is not None else 96
                # depths per residual group: layers.{i}.residual_group.blocks.{j}...
                _grp = {}
                for _k in state_dict:
                    _mm = re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", _k)
                    if _mm:
                        _i, _j = int(_mm.group(1)), int(_mm.group(2))
                        _grp.setdefault(_i, set()).add(_j)
                depths = tuple(len(_grp[i]) for i in sorted(_grp)) if _grp else (6, 6, 6, 6)
                # d_state / mlp_ratio from the first SS2D block's A_logs (K=4 stacked, then flattened)
                _alogs = state_dict.get("layers.0.residual_group.blocks.0.self_attention.A_logs")
                d_state = int(_alogs.shape[-1]) if _alogs is not None else 16
                d_inner = int(_alogs.shape[0] // 4) if _alogs is not None else embed_dim * 2
                mlp_ratio = round(d_inner / max(1, embed_dim), 4)
                # resi_connection: 1conv = Conv2d directly ("layers.0.conv.weight"),
                # 3conv = Sequential ("layers.0.conv.0.weight")
                resi_connection = "1conv" if "layers.0.conv.weight" in state_dict else "3conv"
                # upsampler + scale from reconstruction-module signature
                if "conv_up1.weight" in state_dict:
                    upsampler = "nearest+conv"
                    up_scale = 4
                elif "conv_before_upsample.0.weight" in state_dict:
                    upsampler = "pixelshuffle"
                    _stages = sorted({int(m.group(1)) for k in state_dict
                                      if (m := re.match(r"^upsample\.(\d+)\.weight$", k))})
                    if len(_stages) == 1:
                        _outc = state_dict[f"upsample.{_stages[0]}.weight"].shape[0]
                        up_scale = 3 if _outc == 9 * 64 else 2
                    else:
                        up_scale = 2 ** max(1, len(_stages))
                elif "upsample.0.weight" in state_dict:
                    upsampler = "pixelshuffledirect"
                    _outc = state_dict["upsample.0.weight"].shape[0]
                    _s = round(math.sqrt(_outc / 3.0))
                    up_scale = _s if _s >= 1 and _s * _s * 3 == _outc else 4
                else:
                    upsampler = ""
                    up_scale = 1
                scale = scale_hint if scale_hint > 0 else up_scale
                model = MambaIR(embed_dim=embed_dim, depths=depths, d_state=d_state,
                                mlp_ratio=mlp_ratio, upscale=max(1, scale),
                                upsampler=upsampler, resi_connection=resi_connection)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] MambaIR manuel : embed_dim={embed_dim}, depths={depths}, "
                      f"d_state={d_state}, mlp_ratio={mlp_ratio}, up={upsampler}, "
                      f"resi={resi_connection}, scale={scale}x", flush=True)

            elif detected == "mambairv2":
                from traiNNer.archs.mambairv2_arch import MambaIRv2
                _cf = state_dict.get("conv_first.weight")
                embed_dim = int(_cf.shape[0]) if _cf is not None else 48
                # depths / num_layers: layers.{i}.residual_group.layers.{j}...
                _grp = {}
                for _k in state_dict:
                    _mm = re.match(r"^layers\.(\d+)\.residual_group\.layers\.(\d+)\.", _k)
                    if _mm:
                        _i, _j = int(_mm.group(1)), int(_mm.group(2))
                        _grp.setdefault(_i, set()).add(_j)
                depths = tuple(len(_grp[i]) for i in sorted(_grp)) if _grp else (6, 6, 6, 6)
                _alogs = state_dict.get("layers.0.residual_group.layers.0.assm.selectiveScan.A_logs")
                d_state = int(_alogs.shape[-1]) if _alogs is not None else 8
                _inproj = state_dict.get("layers.0.residual_group.layers.0.assm.in_proj.0.weight")
                mlp_ratio = round(int(_inproj.shape[0]) / max(1, embed_dim), 4) if _inproj is not None else 1.5
                _embB = state_dict.get("layers.0.residual_group.layers.0.assm.embeddingB.weight")
                num_tokens = int(_embB.shape[0]) if _embB is not None else 64
                inner_rank = int(_embB.shape[1]) if _embB is not None else 32
                _rpbt = state_dict.get("layers.0.residual_group.layers.0.win_mhsa.relative_position_bias_table")
                if _rpbt is not None:
                    num_heads = int(_rpbt.shape[-1])
                    window_size = max(2, round((math.sqrt(_rpbt.shape[0]) + 1) / 2))
                else:
                    num_heads, window_size = 4, 16
                _dw = state_dict.get("layers.0.residual_group.layers.0.convffn1.dwconv.depthwise_conv.0.weight")
                convffn_kernel_size = int(_dw.shape[-1]) if _dw is not None else 5
                resi_connection = "1conv" if "layers.0.conv.weight" in state_dict else "3conv"
                if "conv_up1.weight" in state_dict:
                    upsampler = "nearest+conv"; up_scale = 4
                elif "conv_before_upsample.0.weight" in state_dict:
                    upsampler = "pixelshuffle"
                    _stages = sorted({int(m.group(1)) for k in state_dict
                                      if (m := re.match(r"^upsample\.(\d+)\.weight$", k))})
                    if len(_stages) == 1:
                        _outc = state_dict[f"upsample.{_stages[0]}.weight"].shape[0]
                        up_scale = 3 if _outc == 9 * 64 else 2
                    else:
                        up_scale = 2 ** max(1, len(_stages))
                elif "upsample.0.weight" in state_dict:
                    upsampler = "pixelshuffledirect"
                    _outc = state_dict["upsample.0.weight"].shape[0]
                    _s = round(math.sqrt(_outc / 3.0))
                    up_scale = _s if _s >= 1 and _s * _s * 3 == _outc else 4
                else:
                    upsampler = ""; up_scale = 1
                scale = scale_hint if scale_hint > 0 else up_scale
                model = MambaIRv2(embed_dim=embed_dim, depths=depths, num_heads=(num_heads,) * len(depths),
                                  d_state=d_state, mlp_ratio=mlp_ratio, window_size=window_size,
                                  inner_rank=inner_rank, num_tokens=num_tokens,
                                  convffn_kernel_size=convffn_kernel_size, upscale=max(1, scale),
                                  upsampler=upsampler, resi_connection=resi_connection)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] MambaIRv2 manuel : embed_dim={embed_dim}, depths={depths}, "
                      f"d_state={d_state}, heads={num_heads}, win={window_size}, "
                      f"tokens={num_tokens}, rank={inner_rank}, up={upsampler}, "
                      f"scale={scale}x", flush=True)

            elif detected == "artcnn":
                from traiNNer.archs.artcnn_arch import ArtCNN
                w0 = state_dict.get("conv0.weight")
                filters = int(w0.shape[0]) if w0 is not None else 96
                in_ch = int(w0.shape[1]) if w0 is not None else 3
                kernel_size = int(w0.shape[-1]) if w0 is not None else 3
                n_block = len({int(m.group(1)) for k in state_dict
                               if (m := re.match(r"^res_block\.(\d+)\.conv\.0\.0\.weight$", k))})
                n_block = n_block if n_block > 0 else 16
                up_w = state_dict.get("depth_to_space.upscale.0.weight")
                if up_w is not None and in_ch > 0:
                    _sq = up_w.shape[0] / in_ch
                    _s = round(math.sqrt(_sq))
                    det_scale = _s if _s >= 1 and _s * _s == round(_sq) else 4
                else:
                    det_scale = 4
                scale = scale_hint if scale_hint > 0 else det_scale
                model = ArtCNN(in_ch=in_ch, scale=max(1, scale), filters=filters,
                                n_block=n_block, kernel_size=kernel_size)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] ArtCNN manuel : filters={filters}, n_block={n_block}, "
                      f"k={kernel_size}, scale={scale}x", flush=True)

            elif detected == "lkfmixer":
                from traiNNer.archs.lkfmixer_arch import LKFMixer
                w0 = state_dict.get("conv_first.weight")
                channels = int(w0.shape[0]) if w0 is not None else 40
                in_channels = int(w0.shape[1]) if w0 is not None else 3
                num_block = len({int(m.group(1)) for k in state_dict
                                  if (m := re.match(r"^layers\.layers\.(\d+)\.FDB\.c1_d\.weight$", k))})
                num_block = num_block if num_block > 0 else 6
                lk_w = state_dict.get("layers.layers.0.FDB.c1_r.PLKB.DWConv_Kx1.weight")
                large_kernel = int(lk_w.shape[2]) if lk_w is not None else 31
                split_channels = int(lk_w.shape[0]) if lk_w is not None else 10
                split_factor = round(split_channels / max(1, channels), 4)
                up_w = state_dict.get("upsampler.upsampleOneStep.0.weight")
                out_channels = 3
                if up_w is not None:
                    _sq = up_w.shape[0] / out_channels
                    _s = round(math.sqrt(_sq))
                    det_scale = _s if _s >= 1 and _s * _s == round(_sq) else 4
                else:
                    det_scale = 4
                scale = scale_hint if scale_hint > 0 else det_scale
                model = LKFMixer(in_channels=in_channels, channels=channels, out_channels=out_channels,
                                  upscale=max(1, scale), num_block=num_block, large_kernel=large_kernel,
                                  split_factor=split_factor)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] LKFMixer manuel : channels={channels}, num_block={num_block}, "
                      f"large_kernel={large_kernel}, split_factor={split_factor}, scale={scale}x", flush=True)

            elif detected == "rtmosr":
                from traiNNer.archs.rtmosr_arch import RTMoSR
                if "to_feat.1.conv2.weight" in state_dict:
                    _feat_w = state_dict["to_feat.1.conv2.weight"]
                    _in_dim = int(_feat_w.shape[1])
                    unshuffle = max(1, round(math.sqrt(_in_dim / 3.0)))
                else:
                    _feat_w = state_dict.get("to_feat.conv2.weight")
                    unshuffle = 0
                dim = int(_feat_w.shape[0]) if _feat_w is not None else 32
                _img_w = state_dict.get("to_img.0.conv2.weight")
                _scale_internal = round(math.sqrt(_img_w.shape[0] / 3.0)) if _img_w is not None else 2
                det_scale = (4 // unshuffle) if unshuffle > 0 else _scale_internal
                scale = scale_hint if scale_hint > 0 else max(1, det_scale)
                n_blocks = len({int(m.group(1)) for k in state_dict
                                if (m := re.match(r"^body\.(\d+)\.norm\.scale$", k))})
                n_blocks = n_blocks if n_blocks > 0 else 2
                _fc1_w = state_dict.get("body.0.fc1.conv2.weight")
                ffn_expansion = round((_fc1_w.shape[0] / 2.0) / dim, 4) if (_fc1_w is not None and dim > 0) else 2.0
                dccm = "body.0.fc2.conv2.weight" in state_dict
                se = "body.0.conv.2.squeezing.0.weight" in state_dict
                unshuffle_mod = unshuffle > 0
                model = RTMoSR(scale=max(1, scale), dim=dim, ffn_expansion=ffn_expansion,
                                n_blocks=n_blocks, unshuffle_mod=unshuffle_mod, dccm=dccm, se=se)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] RTMoSR manuel : dim={dim}, n_blocks={n_blocks}, "
                      f"ffn_exp={ffn_expansion}, dccm={dccm}, se={se}, scale={scale}x", flush=True)

            elif detected == "mosrv2":
                from traiNNer.archs.mosrv2_arch import MoSRv2
                from traiNNer.archs.arch_util import SampleMods3
                if "gblocks.0.weight" in state_dict:
                    _in_w = state_dict["gblocks.0.weight"]
                    unshuffle = 0
                else:
                    _in_w = state_dict.get("gblocks.1.weight")
                    unshuffle = max(1, round(math.sqrt(_in_w.shape[1] / 3.0))) if _in_w is not None else 0
                dim = int(_in_w.shape[0]) if _in_w is not None else 64
                in_ch = 3
                n_block = len({int(m.group(1)) for k in state_dict
                                if (m := re.match(r"^gblocks\.(\d+)\.gamma$", k))})
                n_block = n_block if n_block > 0 else 24
                _gamma_key = next((k for k in state_dict if re.match(r"^gblocks\.\d+\.norm\.(scale|weight)$", k)), None)
                rms_norm = bool(_gamma_key and _gamma_key.endswith(".scale"))
                _fc1_w = None
                for k, v in state_dict.items():
                    if re.match(r"^gblocks\.\d+\.fc1\.weight$", k):
                        _fc1_w = v
                        break
                expansion_ratio = round((_fc1_w.shape[0] / 2.0) / dim, 4) if (_fc1_w is not None and dim > 0) else 1.5
                meta = state_dict.get("to_img.MetaUpsample")
                mid_dim = 32
                upsampler = "pixelshuffledirect"
                _scale_internal = 4
                if meta is not None:
                    try:
                        upsampler = list(SampleMods3.__args__)[int(meta[1].item())]
                        _scale_internal = max(1, int(meta[2].item()))
                        mid_dim = int(meta[5].item())
                    except Exception:
                        pass
                det_scale = (4 // unshuffle) if unshuffle > 0 else _scale_internal
                scale = scale_hint if scale_hint > 0 else max(1, det_scale)
                model = MoSRv2(in_ch=in_ch, scale=max(1, scale), n_block=n_block, dim=dim,
                                upsampler=upsampler, expansion_ratio=expansion_ratio, mid_dim=mid_dim,
                                unshuffle_mod=(unshuffle > 0), rms_norm=rms_norm)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] MoSRv2 manuel : dim={dim}, n_block={n_block}, "
                      f"exp_ratio={expansion_ratio}, up={upsampler}, scale={scale}x", flush=True)

            elif detected == "sebica":
                from traiNNer.archs.sebica_arch import Sebica
                _head_w = state_dict.get("head.0.weight")
                n_val = int(_head_w.shape[0]) if _head_w is not None else 16
                _tail_w = state_dict.get("tail.0.weight")
                det_scale = round(math.sqrt(_tail_w.shape[0] / 3.0)) if _tail_w is not None else 4
                scale = scale_hint if scale_hint > 0 else max(1, det_scale)
                model = Sebica(scale=max(1, scale), N=n_val)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] Sebica manuel : N={n_val}, scale={scale}x", flush=True)

            elif detected == "lmlt":
                from traiNNer.archs.lmlt_arch import LMLT
                _feat_w = state_dict.get("to_feat.weight")
                dim = int(_feat_w.shape[0]) if _feat_w is not None else 60
                n_blocks = len({int(m.group(1)) for k in state_dict
                                if (m := re.match(r"^feats\.(\d+)\.norm1\.weight$", k))})
                n_blocks = n_blocks if n_blocks > 0 else 8
                _ccm_w = state_dict.get("feats.0.ccm.ccm.0.weight")
                ffn_scale = round(_ccm_w.shape[0] / dim, 4) if (_ccm_w is not None and dim > 0) else 2.0
                _img_w = state_dict.get("to_img.0.weight")
                det_scale = round(math.sqrt(_img_w.shape[0] / 3.0)) if _img_w is not None else 4
                scale = scale_hint if scale_hint > 0 else max(1, det_scale)
                model = LMLT(dim=dim, n_blocks=n_blocks, ffn_scale=ffn_scale, scale=max(1, scale))
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] LMLT manuel : dim={dim}, n_blocks={n_blocks}, "
                      f"ffn_scale={ffn_scale}, scale={scale}x", flush=True)

            elif detected == "man":
                from traiNNer.archs.man_arch import MAN
                _head_w = state_dict.get("head.weight")
                n_feats = int(_head_w.shape[0]) if _head_w is not None else 180
                n_resgroups = len({int(m.group(1)) for k in state_dict
                                    if (m := re.match(r"^body\.(\d+)\.body_t\.", k))})
                n_resgroups = n_resgroups if n_resgroups > 0 else 1
                n_resblocks = len({int(m.group(1)) for k in state_dict
                                    if (m := re.match(r"^body\.0\.body\.(\d+)\.", k))})
                n_resblocks = n_resblocks if n_resblocks > 0 else 36
                _tail_w = state_dict.get("tail.0.weight")
                det_scale = round(math.sqrt(_tail_w.shape[0] / 3.0)) if _tail_w is not None else 2
                scale = scale_hint if scale_hint > 0 else max(1, det_scale)
                model = MAN(n_resblocks=n_resblocks, n_resgroups=n_resgroups, n_feats=n_feats,
                            scale=max(1, scale))
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] MAN manuel : n_feats={n_feats}, n_resgroups={n_resgroups}, "
                      f"n_resblocks={n_resblocks}, scale={scale}x", flush=True)

            elif detected == "eimn":
                from traiNNer.archs.eimn_arch import EIMN
                _head_w = state_dict.get("head.0.weight")
                embed_dims = int(_head_w.shape[0]) if _head_w is not None else 64
                num_stages = len({int(m.group(1)) for k in state_dict
                                   if (m := re.match(r"^norm(\d+)\.weight$", k))})
                num_stages = num_stages if num_stages > 0 else 16
                depths = len({int(m.group(1)) for k in state_dict
                               if (m := re.match(r"^block1\.(\d+)\.", k))})
                depths = depths if depths > 0 else 1
                _mlp_w = state_dict.get("block1.0.mlp.linear_in.weight")
                if _mlp_w is not None and embed_dims > 0:
                    _hidden = int(_mlp_w.shape[0] // 2)
                    mlp_ratios = (_hidden + 0.5) / embed_dims  # +0.5 epsilon: int(dim*ratio) truncates safely back to _hidden
                else:
                    mlp_ratios = 2.66
                _tail_w = state_dict.get("tail.0.weight")
                det_scale = round(math.sqrt(_tail_w.shape[0] / 3.0)) if _tail_w is not None else 2
                scale = scale_hint if scale_hint > 0 else max(1, det_scale)
                model = EIMN(embed_dims=embed_dims, scale=max(1, scale), depths=depths,
                             mlp_ratios=mlp_ratios, num_stages=num_stages)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] EIMN manuel : embed_dims={embed_dims}, num_stages={num_stages}, "
                      f"depths={depths}, mlp_ratios={mlp_ratios}, scale={scale}x", flush=True)

            elif detected == "elan":
                from traiNNer.archs.elan_arch import ELAN
                _head_w = state_dict.get("head.0.weight")
                c_elan = int(_head_w.shape[0]) if _head_w is not None else 180
                n_body = len({int(m.group(1)) for k in state_dict
                               if (m := re.match(r"^body\.(\d+)\.", k))})
                n_body = n_body if n_body > 0 else 36
                n_share = len({int(m.group(1)) for k in state_dict
                                if (m := re.match(r"^body\.0\.modules_lfe\.lfe_(\d+)\.", k))})
                n_share = (n_share - 1) if n_share > 0 else 0
                m_elan = n_body * (1 + n_share)
                _tail_w = state_dict.get("tail.0.weight")
                det_scale = round(math.sqrt(_tail_w.shape[0] / 3.0)) if _tail_w is not None else 4
                scale = scale_hint if scale_hint > 0 else max(1, det_scale)
                model = ELAN(scale=max(1, scale), m_elan=m_elan, c_elan=c_elan, n_share=n_share)
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] ELAN manuel : c_elan={c_elan}, m_elan={m_elan}, scale={scale}x", flush=True)

            elif detected == "gaterv3":
                from traiNNer.archs.gaterv3_arch import GateRV3
                from traiNNer.archs.arch_util import SampleMods
                _idm = state_dict.get("in_to_dim.weight")
                dim = int(_idm.shape[0]) if _idm is not None else 32
                in_ch = int(_idm.shape[1]) if _idm is not None else 3
                _enc = {}
                for _k in state_dict:
                    _mm = re.match(r"^gater_encode\.(\d+)\.gated\.(\d+)\.", _k)
                    if _mm:
                        _enc.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                enc_blocks = tuple(len(_enc[i]) for i in sorted(_enc)) if _enc else (2, 2, 4, 6)
                _dec = {}
                for _k in state_dict:
                    _mm = re.match(r"^decode\.(\d+)\.gated\.(\d+)\.", _k)
                    if _mm:
                        _dec.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                dec_blocks = tuple(len(_dec[i]) for i in sorted(_dec)) if _dec else (2, 2, 2, 2)
                _lat_idx = {int(m.group(1)) for k in state_dict if (m := re.match(r"^latent\.(\d+)\.", k))}
                num_latent = len(_lat_idx) if _lat_idx else 8
                _sisr_idx = {int(m.group(1)) for k in state_dict if (m := re.match(r"^span_n_b\.(\d+)\.", k))}
                sisr_blocks = len(_sisr_idx) if _sisr_idx else 4
                attention = any(".token_mix.qkv.weight" in k for k in state_dict)
                upsample = "pixelshuffle"; upsample_mid_dim = 48; det_scale = 2
                _meta = state_dict.get("dim_to_in.MetaUpsample")
                if _meta is not None:
                    try:
                        upsample = list(SampleMods.__args__)[int(_meta[1].item())]
                        det_scale = max(1, int(_meta[2].item()))
                        upsample_mid_dim = int(_meta[5].item())
                    except Exception:
                        pass
                elif "dim_to_in.weight" in state_dict:
                    det_scale = 1
                scale = scale_hint if scale_hint > 0 else det_scale
                model = GateRV3(in_ch=in_ch, dim=dim, enc_blocks=enc_blocks, dec_blocks=dec_blocks,
                                 num_latent=num_latent, scale=max(1, scale), upsample=upsample,
                                 upsample_mid_dim=upsample_mid_dim, attention=attention,
                                 sisr_blocks=sisr_blocks)
                model.load_state_dict(state_dict, strict=True)
                print(f"[Runner] GateRV3 manuel : dim={dim}, enc={enc_blocks}, dec={dec_blocks}, "
                      f"latent={num_latent}, sisr={sisr_blocks}, attn={attention}, up={upsample}, "
                      f"scale={scale}x", flush=True)

            elif detected in ("hit_sir", "hit_sng"):
                if detected == "hit_sir":
                    from traiNNer.archs.hit_sir_arch import HiT_SIR as _HiTCls
                else:
                    from traiNNer.archs.hit_sng_arch import HiT_SNG as _HiTCls
                _cf = state_dict.get("conv_first.weight")
                embed_dim = int(_cf.shape[0]) if _cf is not None else 60
                in_chans = int(_cf.shape[1]) if _cf is not None else 3
                _grp = {}
                for _k in state_dict:
                    _mm = re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", _k)
                    if _mm:
                        _grp.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                depths = tuple(len(_grp[i]) for i in sorted(_grp)) if _grp else (6, 6, 6, 6)
                _headw = state_dict.get(
                    "layers.0.residual_group.blocks.0.correlation.pos.pos3.2.weight")
                num_heads_v = int(_headw.shape[0]) if _headw is not None else 6
                num_heads = (num_heads_v,) * len(depths)
                _mlpw = state_dict.get("layers.0.residual_group.blocks.0.mlp.fc1.weight")
                mlp_ratio = round(_mlpw.shape[0] / max(1, embed_dim), 4) if _mlpw is not None else 2.0
                resi_connection = "1conv" if "layers.0.conv.weight" in state_dict else "3conv"
                if "conv_up1.weight" in state_dict:
                    upsampler = "nearest+conv"; up_scale = 4
                elif "conv_before_upsample.0.weight" in state_dict:
                    upsampler = "pixelshuffle"
                    _stages = sorted({int(m.group(1)) for k in state_dict
                                      if (m := re.match(r"^upsample\.(\d+)\.weight$", k))})
                    if len(_stages) == 1:
                        _outc = state_dict[f"upsample.{_stages[0]}.weight"].shape[0]
                        up_scale = 3 if _outc == 9 * 64 else 2
                    else:
                        up_scale = 2 ** max(1, len(_stages))
                elif "upsample.0.weight" in state_dict:
                    upsampler = "pixelshuffledirect"
                    _outc = state_dict["upsample.0.weight"].shape[0]
                    _s = round(math.sqrt(_outc / 3.0))
                    up_scale = _s if _s >= 1 and _s * _s * 3 == _outc else 4
                else:
                    upsampler = ""; up_scale = 1
                scale = scale_hint if scale_hint > 0 else up_scale
                model = _HiTCls(upscale=max(1, scale), in_chans=in_chans, embed_dim=embed_dim,
                                 depths=depths, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                 upsampler=upsampler, resi_connection=resi_connection)
                # base_win_size/hier_win_ratios non dérivables des shapes (indépendants des poids
                # pour la plupart des sous-modules) — valeurs par défaut du registre conservées ;
                # strict=False pour tolérer un éventuel mismatch de spatial_linear si le modèle a
                # été entraîné avec un base_win_size/hier_win_ratios non standard.
                model.load_state_dict(state_dict, strict=False)
                print(f"[Runner] {detected.upper()} manuel : embed_dim={embed_dim}, depths={depths}, "
                      f"heads={num_heads_v}, mlp_ratio={mlp_ratio}, up={upsampler}, "
                      f"resi={resi_connection}, scale={scale}x", flush=True)

            elif detected == "cascadedgaze":
                from traiNNer.archs.cgnet_arch import CascadedGaze
                _introw = state_dict.get("intro.weight")
                width = int(_introw.shape[0]) if _introw is not None else 60
                img_channel = int(_introw.shape[1]) if _introw is not None else 3
                _enc = {}
                for _k in state_dict:
                    _mm = re.match(r"^encoders\.(\d+)\.(\d+)\.", _k)
                    if _mm:
                        _enc.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                enc_blk_nums = [len(_enc[i]) for i in sorted(_enc)] if _enc else [2, 2, 4, 6]
                _dec = {}
                for _k in state_dict:
                    _mm = re.match(r"^decoders\.(\d+)\.(\d+)\.", _k)
                    if _mm:
                        _dec.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                dec_blk_nums = [len(_dec[i]) for i in sorted(_dec)] if _dec else [2, 2, 2, 2]
                _mid_idx = {int(m.group(1)) for k in state_dict
                            if (m := re.match(r"^middle_blks\.(\d+)\.", k))}
                middle_blk_num = len(_mid_idx) if _mid_idx else 10
                _stage_ids = sorted(_enc) if _enc else list(range(4))
                GCE_CONVS_nums = []
                for i in _stage_ids:
                    _has3 = f"encoders.{i}.0.GCE.depthwise_separable_convs.2.pointwise.weight" in state_dict
                    GCE_CONVS_nums.append(3 if _has3 else 2)
                if not GCE_CONVS_nums:
                    GCE_CONVS_nums = [3, 3, 2, 2]
                # CascadedGaze est une arch restauration/débruitage — scale toujours 1 (assert
                # dans le constructeur), le scale_hint UI est ignoré pour cette arch.
                scale = 1
                model = CascadedGaze(scale=1, img_channel=img_channel, width=width,
                                      middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums,
                                      dec_blk_nums=dec_blk_nums, GCE_CONVS_nums=GCE_CONVS_nums)
                model.load_state_dict(state_dict, strict=True)
                print(f"[Runner] CascadedGaze manuel : width={width}, enc={enc_blk_nums}, "
                      f"dec={dec_blk_nums}, middle={middle_blk_num}, gce={GCE_CONVS_nums}, "
                      f"scale=1x (fixe)", flush=True)

            elif detected == "escrealm":
                from traiNNer.archs.escreal_arch import ESCRealM
                from traiNNer.archs.arch_util import SampleMods3
                if "proj.1.weight" in state_dict:
                    unshuffle_mod = True
                    _pw = state_dict["proj.1.weight"]
                    dim = int(_pw.shape[0])
                    _in_ch = int(_pw.shape[1])
                    _unshuffle_factor = max(1, round(math.sqrt(_in_ch / 3.0)))
                    _exposed_scale = max(1, 4 // _unshuffle_factor)
                else:
                    unshuffle_mod = False
                    _pw = state_dict.get("proj.weight")
                    dim = int(_pw.shape[0]) if _pw is not None else 64
                    _exposed_scale = None
                _plk = state_dict.get("plk_filter")
                pdim = int(_plk.shape[0]) if _plk is not None else 16
                kernel_size = int(_plk.shape[-1]) if _plk is not None else 13
                _blk_idx = {int(m.group(1)) for k in state_dict if (m := re.match(r"^blocks\.(\d+)\.", k))}
                n_blocks = len(_blk_idx) if _blk_idx else 10
                _conv_idx = {int(m.group(1)) for k in state_dict
                             if (m := re.match(r"^blocks\.0\.pconvs\.(\d+)\.", k))}
                conv_blocks = len(_conv_idx) if _conv_idx else 5
                _rpb = state_dict.get("blocks.0.attn.relative_position_bias")
                if _rpb is not None:
                    num_heads = int(_rpb.shape[0])
                    window_size = max(2, int(round((math.sqrt(_rpb.shape[1]) + 1) / 2)))
                else:
                    num_heads, window_size = 4, 32
                _cffw = state_dict.get("blocks.0.convffns.0.proj.weight")
                exp_ratio = round(_cffw.shape[0] / max(1, dim), 4) if _cffw is not None else 2.0
                upsampler = "transpose+conv"; mid_dim = 64; _scale_internal = 4
                _meta = state_dict.get("to_img.MetaUpsample")
                if _meta is not None:
                    try:
                        upsampler = list(SampleMods3.__args__)[int(_meta[1].item())]
                        _scale_internal = max(1, int(_meta[2].item()))
                        mid_dim = int(_meta[5].item())
                    except Exception:
                        pass
                det_scale = _exposed_scale if _exposed_scale is not None else _scale_internal
                scale = scale_hint if scale_hint > 0 else det_scale
                # attn_type: never "Flex" (needs triton, missing / unsupported on Windows and
                # Pascal). Same weights and maths in all three modes: "SDPA" on GPU (memory-
                # efficient kernel, half the VRAM of "Naive"), "Naive" on CPU.
                model = ESCRealM(dim=dim, pdim=pdim, kernel_size=kernel_size, n_blocks=n_blocks,
                                  conv_blocks=conv_blocks, window_size=window_size, num_heads=num_heads,
                                  scale=max(1, scale), exp_ratio=exp_ratio,
                                  attn_type="SDPA" if device.type == "cuda" else "Naive",
                                  mid_dim=mid_dim, upsampler=upsampler, unshuffle_mod=unshuffle_mod)
                model.load_state_dict(state_dict, strict=True)
                print(f"[Runner] ESCRealM manuel : dim={dim}, pdim={pdim}, n_blocks={n_blocks}, "
                      f"conv_blocks={conv_blocks}, heads={num_heads}, win={window_size}, "
                      f"up={upsampler}, unshuffle={unshuffle_mod}, scale={scale}x", flush=True)

            elif detected == "emt":
                from traiNNer.archs.emt_arch import EMT
                _headw = state_dict.get("head.weight")
                dim = int(_headw.shape[0]) if _headw is not None else 60
                num_in_ch = int(_headw.shape[1]) if _headw is not None else 3
                _blk_idx = {int(m.group(1)) for k in state_dict if (m := re.match(r"^body\.(\d+)\.", k))}
                n_blocks = len(_blk_idx) if _blk_idx else 6
                _layer_idx = {int(m.group(1)) for k in state_dict
                              if (m := re.match(r"^body\.0\.sa_list\.(\d+)\.", k))}
                n_layers = len(_layer_idx) if _layer_idx else 6
                n_GTLs = len([j for j in _layer_idx
                              if f"body.0.sa_list.{j}.attn.0.weight" in state_dict]) if _layer_idx else 2
                n_GTLs = max(1, n_GTLs)
                _mlpw = state_dict.get("body.0.mlp_list.0.fc1.weight")
                mlp_ratio = round(_mlpw.shape[0] / max(1, dim)) if _mlpw is not None else 2
                _tidx = sorted({int(m.group(1)) for k in state_dict
                                if (m := re.match(r"^tail\.(\d+)\.weight$", k))})
                if len(_tidx) >= 2:
                    upsampler = "pixelshuffle"
                    _stage0_w = state_dict[f"tail.{_tidx[0]}.weight"]
                    _ratio0 = round(_stage0_w.shape[0] / max(1, dim))
                    det_scale = 3 if _ratio0 == 9 else 2 ** max(1, len(_tidx) - 1)
                else:
                    upsampler = "pixelshuffledirect"
                    _w0 = state_dict.get("tail.0.weight")
                    det_scale = round(math.sqrt(_w0.shape[0] / 3.0)) if _w0 is not None else 4
                scale = scale_hint if scale_hint > 0 else det_scale
                # num_heads/window_list/shift_list n'affectent aucune shape de poids (juste le
                # regroupement/partitionnement au forward) — non dérivables du state_dict.
                # SWSA fait attn(dim*2) puis rearrange par (qv=2, head=num_heads) : num_heads
                # DOIT diviser dim*2/2=dim exactement sous peine de crash au forward. On essaie
                # la valeur registre (3) puis les diviseurs usuels par ordre de préférence.
                num_heads = next((h for h in (3, 4, 5, 6, 2, 1, 8, 10, 12, 15, 20)
                                   if dim % (2 * h) == 0), 1)
                model = EMT(scale=max(1, scale), num_in_ch=num_in_ch, num_out_ch=3, upsampler=upsampler,
                            dim=dim, n_blocks=n_blocks, n_layers=n_layers, n_GTLs=n_GTLs,
                            num_heads=num_heads, mlp_ratio=mlp_ratio)
                model.load_state_dict(state_dict, strict=True)
                print(f"[Runner] EMT manuel : dim={dim}, n_blocks={n_blocks}, n_layers={n_layers}, "
                      f"n_GTLs={n_GTLs}, mlp_ratio={mlp_ratio}, up={upsampler}, scale={scale}x",
                      flush=True)

            elif detected == "flexnet":
                from traiNNer.archs.flexnet_arch import FlexNet
                _c11w = state_dict.get("short_cut.conv11.weight")
                dim = int(_c11w.shape[0]) if _c11w is not None else 64
                inp_channels = int(_c11w.shape[1]) if _c11w is not None else 3
                if any(k.startswith("pipeline.att.") for k in state_dict):
                    pipeline_type = "linear"
                    _stg = {}
                    for _k in state_dict:
                        _mm = re.match(r"^pipeline\.att\.(\d+)\.t_blocks\.(\d+)\.", _k)
                        if _mm:
                            _stg.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                    num_blocks = tuple(len(_stg[i]) for i in sorted(_stg)) if _stg else (6, 6, 6, 6, 6, 6)
                    _ffnw = state_dict.get("pipeline.att.0.t_blocks.0.ffn.key.weight")
                else:
                    pipeline_type = "meta"
                    _stg = {}
                    for _k in state_dict:
                        _mm = re.match(r"^pipeline\.enc(\d+)\.0\.t_blocks\.(\d+)\.", _k)
                        if _mm:
                            _stg.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                    num_blocks = tuple(len(_stg[i]) for i in sorted(_stg)) if _stg else (4, 6, 6, 8)
                    _ffnw = state_dict.get("pipeline.enc0.0.t_blocks.0.ffn.key.weight")
                hidden_rate = round(_ffnw.shape[0] / max(1, dim)) if _ffnw is not None else 4
                channel_norm = any(".ffn.key_norm.weight" in k for k in state_dict)
                _ws_buf = state_dict.get("window_size")
                try:
                    window_size = int(_ws_buf.item()) if _ws_buf is not None else 8
                except Exception:
                    window_size = 8
                if "to_img.offset.weight" in state_dict:
                    upsampler = "dysample"
                    _offw = state_dict["to_img.offset.weight"]
                    det_scale = max(1, round(math.sqrt(_offw.shape[0] / 8.0)))
                elif "to_img.1.0.weight" in state_dict:
                    upsampler = "nearest+conv"
                    _sf = state_dict.get("scale_factor")
                    try:
                        det_scale = max(1, int(_sf.item())) if _sf is not None else 4
                    except Exception:
                        det_scale = 4
                else:
                    upsampler = "pixelshuffle"
                    _imw = state_dict.get("to_img.0.weight")
                    det_scale = max(1, round(math.sqrt(_imw.shape[0] / 3.0))) if _imw is not None else 4
                scale = scale_hint if scale_hint > 0 else det_scale
                model = FlexNet(inp_channels=inp_channels, out_channels=3, scale=max(1, scale), dim=dim,
                                 num_blocks=num_blocks, window_size=window_size, hidden_rate=hidden_rate,
                                 channel_norm=channel_norm, pipeline_type=pipeline_type,
                                 upsampler=upsampler)
                model.load_state_dict(state_dict, strict=True)
                print(f"[Runner] FlexNet manuel : dim={dim}, pipeline={pipeline_type}, "
                      f"num_blocks={num_blocks}, window={window_size}, hidden_rate={hidden_rate}, "
                      f"channel_norm={channel_norm}, up={upsampler}, scale={scale}x", flush=True)

            elif detected == "srformer":
                from spandrel_extra_arches.architectures.SRFormer import SRFormer
                _cf = state_dict.get("conv_first.weight")
                embed_dim = int(_cf.shape[0]) if _cf is not None else 180
                in_chans = int(_cf.shape[1]) if _cf is not None else 3
                _grp = {}
                for _k in state_dict:
                    _mm = re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", _k)
                    if _mm:
                        _grp.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                depths = tuple(len(_grp[i]) for i in sorted(_grp)) if _grp else (6, 6, 6, 6, 6, 6)
                _heads_list = []
                for i in (sorted(_grp) if _grp else range(len(depths))):
                    _rpbt = state_dict.get(
                        f"layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table")
                    _heads_list.append(int(_rpbt.shape[1]) if _rpbt is not None else 6)
                num_heads = tuple(_heads_list) if _heads_list else (6,) * len(depths)
                _rpbt0 = state_dict.get("layers.0.residual_group.blocks.0.attn.relative_position_bias_table")
                window_size = int(round(math.sqrt(_rpbt0.shape[0]))) + 1 if _rpbt0 is not None else 24
                _mlpw = state_dict.get("layers.0.residual_group.blocks.0.mlp.fc1.weight")
                mlp_ratio = round(_mlpw.shape[0] / max(1, embed_dim), 4) if _mlpw is not None else 2.0
                ape = "absolute_pos_embed" in state_dict
                patch_norm = "patch_embed.norm.weight" in state_dict
                qkv_bias = "layers.0.residual_group.blocks.0.attn.q.bias" in state_dict
                resi_connection = "1conv" if "conv_after_body.weight" in state_dict else "3conv"
                if "conv_hr.weight" in state_dict:
                    upsampler = "nearest+conv"; det_scale = 4
                elif "conv_before_upsample.0.weight" in state_dict:
                    upsampler = "pixelshuffle"
                    _stages = sorted({int(m.group(1)) for k in state_dict
                                      if (m := re.match(r"^upsample\.(\d+)\.weight$", k))})
                    if len(_stages) == 1:
                        _outc = state_dict[f"upsample.{_stages[0]}.weight"].shape[0]
                        det_scale = 3 if round(_outc / max(1, embed_dim if False else 64)) == 9 else 2
                    else:
                        det_scale = 2 ** max(1, len(_stages))
                elif "upsample.0.weight" in state_dict:
                    upsampler = "pixelshuffledirect"
                    _outc = state_dict["upsample.0.weight"].shape[0]
                    _s = round(math.sqrt(_outc / max(1, in_chans)))
                    det_scale = _s if _s >= 1 else 4
                else:
                    upsampler = ""; det_scale = 1
                scale = scale_hint if scale_hint > 0 else det_scale
                # img_size : patch_size==1 toujours (registre), donc patches_resolution==img_size.
                # attn_mask (bloc PSA shift>0) donne nW = (patches_resolution/window_size)^2 —
                # dérivation EXACTE, bien plus fiable qu'un défaut arbitraire (attn_mask est un
                # buffer persistant : une shape fausse fait planter le load_state_dict strict).
                img_size = window_size
                for _k in state_dict:
                    if re.match(r"^layers\.0\.residual_group\.blocks\.\d+\.attn_mask$", _k):
                        _nw = state_dict[_k].shape[0]
                        img_size = window_size * max(1, round(math.sqrt(_nw)))
                        break
                else:
                    if ape:
                        _ape = state_dict.get("absolute_pos_embed")
                        if _ape is not None:
                            img_size = int(round(math.sqrt(_ape.shape[1])))
                model = SRFormer(upscale=max(1, scale), in_chans=in_chans, img_size=img_size,
                                  embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                  window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  ape=ape, patch_norm=patch_norm, upsampler=upsampler,
                                  resi_connection=resi_connection)
                model.load_state_dict(state_dict, strict=True)
                print(f"[Runner] SRFormer manuel : embed_dim={embed_dim}, depths={depths}, "
                      f"heads={num_heads}, window={window_size}, mlp_ratio={mlp_ratio}, "
                      f"up={upsampler}, resi={resi_connection}, scale={scale}x", flush=True)

            elif detected == "srformerv2":
                from traiNNer.archs.srformerv2_arch import SRFormerv2
                _cf = state_dict.get("conv_first.weight")
                embed_dim = int(_cf.shape[0]) if _cf is not None else 240
                in_chans = int(_cf.shape[1]) if _cf is not None else 3
                _grp = {}
                for _k in state_dict:
                    _mm = re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", _k)
                    if _mm:
                        _grp.setdefault(int(_mm.group(1)), set()).add(int(_mm.group(2)))
                # Nb de blocs bruts par layer = depth + 3 (3 SwinTransformerBlock toujours insérés
                # aux positions [0,3,6] par le code de srformerv2_arch.py, quel que soit depth).
                depths = tuple(max(1, len(_grp[i]) - 3) for i in sorted(_grp)) if _grp else (4, 4, 4, 4, 4, 4)
                _psa_idx = None
                for _j in sorted(_grp.get(0, [])):
                    if f"layers.0.residual_group.blocks.{_j}.attn.kv.weight" in state_dict:
                        _psa_idx = _j
                        break
                if _psa_idx is not None:
                    _kvw = state_dict[f"layers.0.residual_group.blocks.{_psa_idx}.attn.kv.weight"]
                    squeeze_dim = int(_kvw.shape[0] // 2)
                    _rpbt = state_dict.get(
                        f"layers.0.residual_group.blocks.{_psa_idx}.attn.relative_position_bias_table")
                    num_heads_v = int(_rpbt.shape[1]) if _rpbt is not None else 8
                    window_size = int(round(math.sqrt(_rpbt.shape[0]))) + 1 if _rpbt is not None else 36
                    _mlpw = state_dict.get(
                        f"layers.0.residual_group.blocks.{_psa_idx}.mlp.fc1.weight")
                    mlp_ratio = round(_mlpw.shape[0] / max(1, embed_dim)) if _mlpw is not None else 2
                else:
                    squeeze_dim, num_heads_v, window_size, mlp_ratio = 60, 8, 36, 2
                num_heads = (num_heads_v,) * len(depths)
                resi_connection = "1conv" if "conv_after_body.weight" in state_dict else "3conv"
                if "conv_before_upsample.0.weight" in state_dict:
                    upsampler = "pixelshuffle"
                    _stages = sorted({int(m.group(1)) for k in state_dict
                                      if (m := re.match(r"^upsample\.(\d+)\.weight$", k))})
                    if len(_stages) == 1:
                        _outc = state_dict[f"upsample.{_stages[0]}.weight"].shape[0]
                        det_scale = 3 if round(_outc / 64) == 9 else 2
                    else:
                        det_scale = 2 ** max(1, len(_stages))
                elif "upsample.0.weight" in state_dict:
                    upsampler = "pixelshuffledirect"
                    _outc = state_dict["upsample.0.weight"].shape[0]
                    _s = round(math.sqrt(_outc / max(1, in_chans)))
                    det_scale = _s if _s >= 1 else 4
                elif "conv_hr.weight" in state_dict:
                    upsampler = "nearest+conv"; det_scale = 4
                else:
                    upsampler = ""; det_scale = 2
                scale = scale_hint if scale_hint > 0 else det_scale
                # img_size : patch_size==1 toujours (registre) → patches_resolution==img_size ;
                # dérivé de la vraie shape d'attn_mask (buffer persistant d'un bloc PSA shift>0),
                # bien plus fiable qu'un défaut arbitraire (une shape fausse fait planter le load
                # strict).
                img_size = window_size
                for _k in state_dict:
                    if re.match(r"^layers\.0\.residual_group\.blocks\.\d+\.attn_mask$", _k):
                        _nw = state_dict[_k].shape[0]
                        img_size = window_size * max(1, round(math.sqrt(_nw)))
                        break
                model = SRFormerv2(upscale=max(1, scale), in_chans=in_chans, embed_dim=embed_dim,
                                    squeeze_dim=squeeze_dim, depths=depths, num_heads=num_heads,
                                    window_size=window_size, img_size=img_size, mlp_ratio=mlp_ratio,
                                    upsampler=upsampler, resi_connection=resi_connection)
                model.load_state_dict(state_dict, strict=True)
                print(f"[Runner] SRFormerv2 manuel : embed_dim={embed_dim}, depths={depths}, "
                      f"squeeze_dim={squeeze_dim}, heads={num_heads_v}, window={window_size}, "
                      f"mlp_ratio={mlp_ratio}, up={upsampler}, scale={scale}x", flush=True)

            elif detected == "docres":
                from traiNNer.archs.docres_arch import DocRes
                _pew = state_dict.get("patch_embed.proj.weight")
                dim = int(_pew.shape[0]) if _pew is not None else 48
                _embed_ch = int(_pew.shape[1]) if _pew is not None else 5
                # embed_channels==5 → arch's forced auto-coord injection (inp_channels==3
                # trained), otherwise inp_channels == embed_ch directly (no coords).
                inp_channels = 3 if _embed_ch == 5 else _embed_ch
                _outw = state_dict.get("output.weight")
                out_channels = int(_outw.shape[0]) if _outw is not None else 3
                bias = "encoder_level1.0.attn.qkv.bias" in state_dict
                LayerNorm_type = "WithBias" if "encoder_level1.0.norm1.body.bias" in state_dict else "BiasFree"

                def _docres_count(prefix):
                    idx = {int(m.group(1)) for k in state_dict if (m := re.match(rf"^{prefix}\.(\d+)\.", k))}
                    return len(idx)

                n1 = _docres_count("encoder_level1") or 2
                n2 = _docres_count("encoder_level2") or 3
                n3 = _docres_count("encoder_level3") or 3
                n4 = _docres_count("latent") or 4
                num_blocks = [n1, n2, n3, n4]
                num_refinement_blocks = _docres_count("refinement") or 4

                def _docres_heads(prefix):
                    t = state_dict.get(f"{prefix}.0.attn.temperature")
                    return int(t.shape[0]) if t is not None else 1

                heads = [_docres_heads("encoder_level1"), _docres_heads("encoder_level2"),
                         _docres_heads("encoder_level3"), _docres_heads("latent")]

                # hidden_features_k = int(dim*2**k * ffn_expansion_factor) is computed
                # per-stage (dim doubles each stage) with a SINGLE shared float factor —
                # back-solving from one stage's hidden_features alone does not reproduce
                # the other stages exactly (floor-rounding drifts differently at each dim
                # scale). Instead, narrow an interval of valid factors across every stage:
                # hidden_k/dim_k <= factor < (hidden_k+1)/dim_k, intersect all 4, pick the
                # midpoint. Verified to reproduce every stage's hidden_features exactly.
                _ffn_pairs = []
                for _pfx, _mult in (("encoder_level1", 1), ("encoder_level2", 2),
                                     ("encoder_level3", 4), ("latent", 8)):
                    _w = state_dict.get(f"{_pfx}.0.ffn.project_in.weight")
                    if _w is not None:
                        _ffn_pairs.append((dim * _mult, int(_w.shape[0]) // 2))
                if _ffn_pairs:
                    _lo = max(h / d for d, h in _ffn_pairs)
                    _hi = min((h + 1) / d for d, h in _ffn_pairs)
                    ffn_expansion_factor = (_lo + _hi) / 2.0 if _lo < _hi else _lo
                else:
                    ffn_expansion_factor = 2.66

                model = DocRes(inp_channels=inp_channels, out_channels=out_channels, dim=dim,
                                num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                                heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                LayerNorm_type=LayerNorm_type, dual_pixel_task=True)
                model.load_state_dict(state_dict, strict=True)
                # DocRes = restauration document (deskew/nettoyage), jamais d'upscale — comme
                # cascadedgaze (ARCH_FIXED_SCALE=1 côté UI) : ignore scale_hint.
                scale = 1
                print(f"[Runner] DocRes manuel : dim={dim}, num_blocks={num_blocks}, heads={heads}, "
                      f"refine={num_refinement_blocks}, ffn_exp={ffn_expansion_factor:.3f}, "
                      f"bias={bias}, norm={LayerNorm_type}, scale={scale}x", flush=True)

        except Exception as e:
            _msg = str(e)
            # strict load_state_dict errors list every key — hundreds of lines that
            # bury the actual verdict in the log.
            if len(_msg) > 600:
                _msg = _msg[:600] + " […]"
            print(f"[Runner] Fallback traiNNer manuel echec : {_msg}", flush=True)
            model = None

    if model is None:
        print("[Runner] ERREUR : architecture non reconnue — ni spandrel, ni les archs "
              "traiNNer-redux/custom connues ne correspondent à ce modèle.", flush=True)
        try:
            _sd = _load_state_dict(model_path)
            _tops = sorted({k.split(".")[0] for k in _sd.keys()})
            print(f"[Runner] Signature des clés (pour identification) : {', '.join(_tops[:15])}",
                  flush=True)
            if any(t in _tops for t in ("flow_net", "temporal_fusion", "align_refine")):
                print("[Runner] Ce modèle semble être un modèle VIDÉO multi-frames "
                      "(flux optique / fusion temporelle), pas un modèle image.", flush=True)
        except Exception:
            pass
        sys.exit(1)

    model = model.to(device).eval()
    print("[Runner] Modele pret.", flush=True)

    # ── Inference ──
    img = Image.open(input_path).convert("RGB")
    print(f"[Runner] Inference {img.width}x{img.height} -> {img.width * scale}x{img.height * scale}...",
          flush=True)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

    amp_ctx = torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda"))

    def _infer_tile(img_tensor, ts, tp):
        """Tiled inference with overlap to save VRAM."""
        _, _, H, W = img_tensor.shape
        out = torch.zeros((1, 3, H * scale, W * scale), device=img_tensor.device)
        for y0 in range(0, H, ts):
            for x0 in range(0, W, ts):
                x1s = max(0, x0 - tp); y1s = max(0, y0 - tp)
                x2s = min(W, x0 + ts + tp); y2s = min(H, y0 + ts + tp)
                tile = img_tensor[:, :, y1s:y2s, x1s:x2s]
                with torch.no_grad(), amp_ctx:
                    tile_out = model(tile)
                ox = (x0 - x1s) * scale; oy = (y0 - y1s) * scale
                ow = min(ts, W - x0) * scale; oh = min(ts, H - y0) * scale
                out[:, :, y0*scale:y0*scale+oh, x0*scale:x0*scale+ow] = \
                    tile_out[:, :, oy:oy+oh, ox:ox+ow]
                torch.cuda.empty_cache()
        return out

    _, _, H, W = img_t.shape
    use_tiling = tile_size > 0 and (W > tile_size or H > tile_size)
    if use_tiling:
        print(f"[Runner] Tiling {tile_size}px (pad={tile_pad}px)", flush=True)
        out = _infer_tile(img_t, tile_size, tile_pad)
    else:
        with torch.no_grad(), amp_ctx:
            out = model(img_t)

    # ── Save ──
    out = out.squeeze(0).clamp(0, 1).cpu().float().numpy()
    out = np.transpose(out, (1, 2, 0))
    from runner_save import save_float
    save_float(out, output_path)
    print(f"[Runner] Sauvegarde : {output_path}", flush=True)

    del model, img_t, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: universal_runner.py <model_path> <input_path> <output_path> [tile_size] [tile_pad] [use_amp] [scale_hint]", flush=True)
        sys.exit(1)
    _tile_size  = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    _tile_pad   = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    _use_amp    = sys.argv[6] == "1" if len(sys.argv) > 6 else False
    _scale_hint = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    try:
        run(sys.argv[1], sys.argv[2], sys.argv[3],
            tile_size=_tile_size, tile_pad=_tile_pad, use_amp=_use_amp,
            scale_hint=_scale_hint)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
