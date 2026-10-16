"""
quick_upscale.py — Inférence rapide universelle (NeoSR / Redux).
Supporte le tiling pour économiser la VRAM et la détection automatique
de l'architecture depuis les poids du modèle.
"""
import os
import sys
import math
import re
import time
import json
import threading
import traceback
from typing import Optional, Tuple, Callable

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    from PIL import Image
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False

try:
    import pyvips
    PYVIPS_AVAILABLE = True
except ImportError:
    PYVIPS_AVAILABLE = False


# ─── Architecture detection from state_dict ─────────────────────

# Key signatures to identify architecture from weight keys
ARCH_SIGNATURES = {
    # ── WSR / DUAL (Rift73) — built by arch_infer.py (preset read from the shapes) ──
    "wsr": ["stem.sk.weight", "context_stem.sk.weight", "gates.0.project.weight", "reconstruct.weight"],
    # SPANF3 (traiNNer-redux): SPAN blocks + att_conv + Conv3XC to_pixel (no upsampler.0)
    "spanf3": ["conv_cat.weight", "to_pixel.sk.weight", "block_1.att_conv.weight"],
    "dual": ["stem.direct.expand.weight", "stem.sequential.reduce.weight"],
    # ── HEART / NERVE (traiNNer-redux master, Phhofm) — first: HEART shares
    # "conv_first"/"layers.N.residual_group.blocks"/"conv_after_body" with the
    # SwinIR/HAT/HMA family below, ".attn.to_hidden" (RIB bias) is unique to it.
    # Model construction is delegated to arch_infer.py (shapes → kwargs).
    "heart": ["conv_first.weight", ".attn.to_hidden", "residual_group.blocks"],
    # NeoSR-trained KRGN / MAN / MoSRv2 (no MetaUpsample, unlike traiNNer-redux's) —
    # rebuilt by neosr_general_runner.py::build_extra (strict load, scale tried).
    "krgn": ["head_conv.weight", "features_fusion_module", "sub_mean.weight"],
    "man_neosr": ["head.weight", ".LKA.", "tail.0.weight"],  # + no sub_mean (see excludes)
    # DAT-iLN (traiNNer-redux) — built by arch_infer.py (preset matched on the shapes)
    "dat_iln": ["before_RG.norm.weight", "conv_first.weight", "layers.0.blocks.0.attn.attns."],
    # FDAT-RD (stinkybread) — FDAT + token dictionary; must win over FDAT (its checkpoints
    # also match FDAT's keys). Built by arch_infer.py.
    "fdat_rd": [".attn.dictionary", ".attn.rel_pos_bias", "upsampler.MetaUpsample"],
    # MoTSR (xyether) — 5-frame + recurrent video model, sequence pipeline only.
    "motsr": ["temporal.compress.weight", "feedback_older.0.weight"],
    "nerve": ["stem.weight", "blocks.0.conv1.weight", "upsampler.0.weight"],
    # WireSR (CF2lter, "1x_Wired" anti-aliasing) — built by arch_infer.py.
    "wiresr": ["blocks.0.mix.stripe_h.weight", "blocks.0.mix.diag_m.weight", "upsampler.MetaUpsample"],
    # ── v2.7.0 batch : 8 archs NeoSR supplémentaires (Quick Upscale, subprocess
    # général neosr_general_runner.py) ──────────────────────────────────────────
    # Placées EN TOUT PREMIER dans le dict : "hit_srf" partage ".correlation.qv.
    # linear." + "conv_first.weight" avec la signature "hit_sir" plus bas (les deux
    # familles HiT-SR ont un module "correlation" nommé pareil) — doit gagner la
    # course. "hma" partage "layers.0.residual_group.blocks" + "conv_after_body"
    # avec "swinir" plus bas (HMA dérive de SwinIR) — même raison.
    # PlainUSR — head/tail conv + Block(MBConv/PConv/LocalAttention). Ne couvre que
    # les checkpoints "training-time" (non reparamétrés via switch_to_deploy() —
    # c'est le cas normal des checkpoints net_g/params_ema utilisés par cette app).
    "plainusr": ["head.weight", "block1.body.0.expand_conv.weight", "tail.0.weight"],
    # ASID — IDSG_A (block0) + IDSG (block1/block2/...) avec ESA + pixelshuffle_block.
    "asid": ["block0.res_end.weight", "block1.esa.conv_f.weight"],
    # GRFormer — GRSA (Group-wise Residual self-attention), ESRPB_MLP + q1/k1/v1
    # (Group Residual Linear) sont uniques à cette arch.
    "grformer": ["layers.0.blocks.0.attn.ESRPB_MLP.0.weight",
                 "layers.0.blocks.0.attn.q1.weight"],
    # DCT (Dual Compression Transformer) — swsa (Adaptive_Window_Attention) + lla
    # (Linear_Attention) dans les blocs pairs.
    "dct": ["layers.0.blocks.0.swsa.attns.0.pos.pos_proj.weight",
            "layers.0.blocks.0.lla.qkv.weight"],
    # FlexNet — déjà signalé plus bas (in_to_feat.weight/short_cut.conv11.weight)
    # pour le routing traiNNer — structure de poids identique côté neosr (vérifié).
    # Pas de nouvelle entrée : "flexnet" est ajouté à _NEOSR_GENERAL_ARCHS, ce qui
    # fait gagner ce runner sur _TRAINNER_SUBPROCESS_ARCHS pour les .safetensors.
    # HiT-SRF (famille HiT-SR) — RHTB/residual_group (comme swinir/hat/drct) avec
    # un module "correlation" (SCC) partageant ".correlation.qv.linear." avec
    # hit_sir/hit_sng plus bas → doit être détecté AVANT ces deux entrées.
    "hit_srf": ["layers.0.residual_group.blocks.0.correlation.pos.pos3.2.weight",
                "layers.0.residual_group.blocks.0.mlp.dwconv.depthwise_conv.0.weight"],
    # HMA (Hybrid Multi-axis Aggregation) — RHTB/residual_group avec FusedConv (SE)
    # alterné avec FAB (window attention classique) + GAB (grid attention globale).
    # relative_position_index_SA (buffer top-level) est unique à HMA.
    "hma": ["relative_position_index_SA",
            "layers.0.residual_group.blocks.0.fused_conv.weight"],

    # ── v2.6.1 batch : 9 archs natives traiNNer-redux supplémentaires ──────────
    # Placées EN PREMIER dans le dict : plusieurs d'entre elles partagent des
    # substrings génériques ("layers.0.residual_group.blocks" + "conv_after_body",
    # "plk_filter") avec des signatures déjà présentes plus bas (swinir/hat/esc) —
    # detect_arch_from_state() retourne le PREMIER match trouvé dans l'ordre
    # d'insertion du dict, donc ces entrées doivent gagner la course.
    # HiT-SNG AVANT HiT-SIR : les deux partagent ".correlation.qv.linear.weight"
    # (module DFE commun), seul SNG a la fenêtre NGram — ordre = désambiguïsation.
    # NB: les signatures ci-dessous ne doivent PAS finir par ".weight"/".bias" — sinon
    # detect_arch_from_state() bascule en mode "clé EXACTE" (voir sa docstring) et un
    # motif partiel (préfixé par un index de layer/bloc variable) ne matchera jamais.
    "hit_sng": [".correlation.qv.linear.",
                ".ngram_window_partition.ngram_context.unigram_embed."],
    "hit_sir": [".correlation.qv.linear.", "conv_first.weight"],
    "gaterv3": ["in_to_dim.weight", "sisr_cat_conv.weight"],
    "cascadedgaze": ["intro.weight", "ending.weight", ".GCE.depthwise_separable_convs."],
    # ESCRealM (traiNNer) vs "esc" (neosr, plus bas) : ESCRealM utilise UniUpsampleV3
    # (to_img.MetaUpsample), le ESC neosr n'a jamais ce buffer sur son to_img.
    "escrealm": ["plk_filter", "to_img.MetaUpsample"],
    "emt": ["sub_mean.weight", "head.weight", ".sa_list."],
    "flexnet": ["in_to_feat.weight", "short_cut.conv11.weight"],
    # SRFormerv2 (traiNNer, mixe SwinTransformerBlock "qkv" + PSA_Block "kv"/"q")
    # AVANT SRFormer (spandrel_extra_arches, tous les blocs sont PSA_Block "kv"/"q"
    # sans jamais de "qkv") — les deux signatures sont mutuellement exclusives par
    # construction (bloc 0 diffère), mais on garde l'ordre pour rester explicite.
    "srformerv2": ["layers.0.residual_group.blocks.0.attn.qkv.weight",
                   "layers.0.residual_group.blocks.1.attn.kv.weight"],
    "srformer": ["layers.0.residual_group.blocks.0.attn.kv.weight",
                 "layers.0.residual_group.blocks.0.attn.q.weight"],

    "spanplus": ["feats.0.eval_conv", "upsampler.offset"],  # SPANPlus (traiNNer-redux, DySample upsampler)
    "span": ["feats.0.block", "feats.0.sk"],
    "compact": ["body.0.weight", "body.2.weight"],
    "omnisr": ["residual_layer", "osag"],
    "realplksr": ["feats.0.lk.conv", "dysample"],
    "esrgan": ["body.0.rdb1", "RRDB_trunk"],
    "swinir": ["layers.0.residual_group.blocks", "conv_after_body"],
    "hat": ["layers.0.residual_group.blocks", "conv_before_upsample", "relative_coords_table"],
    "dat": ["layers.0.blocks", "before_RG"],
    "drct": ["layers.0.swin1", "layers.0.adjust1"],
    "rcan": ["body.0.body.0.body.0.weight"],
    "cugan": ["unet1.conv1.conv.0.weight"],
    "esc": ["plk_filter"],
    "ninasr": ["head.0.bias", "body.0.body.2.body"],
    "lmlt":   ["feats.0.lhsb", "to_img.0.weight"],
    "eimn":   ["block1.0.attn.region", "block1.0.layer_scale_1"],
    # SMoSR: Self-Modulate SR (umzi2) — MetaUpsample buffer is very distinctive
    "smosr":  ["blocks_2.0.body", "end_block.0.body", "upsampler.MetaUpsample"],
    # SpanF: variant SPAN avec SPAB1 blocks et conv_near depthwise
    "spanf":  ["block_1.conv1.eval_conv", "conv_near.weight", "block_5.conv1"],
    # SpanC: SPAN reparamétrisable + IGConv multi-scale (MetaIGConv buffer)
    "spanc":  ["block_1.conv_a.eval_conv", "upsampler.coord_map", "MetaIGConv"],
    # SpanPP: SpanC avec blocs SPAB (c1_r/c2_r/c3_r) + upsampler IGConv Fourier (amplitude/freq/query_kernel)
    # Classe SpanC dans spanpp_arch.py — même arch, blocs internes différents de spanc
    "spanpp": ["block_1.c1_r.conv3.eval_conv.weight", "upsampler.amplitude", "MetaIGConv"],
    # GFISRv2: GatedCNN body + UniUpsampleV3 MetaUpsample
    "gfisrv2": ["gfisr_body.0.fc1", "upscale.MetaUpsample", "in_to_dim.weight"],
    # CATANet: Token Aggregation Block (IRCA+IASA) + LRSA (NeoSR, mars 2025)
    # Clés réelles vérifiées sur state_dict : irca_attn / iasa_attn + first_conv
    "catanet": ["blocks.0.0.irca_attn.to_k.weight", "blocks.0.0.iasa_attn.to_q.weight", "first_conv.weight"],

    # ── v2.5.6 : custom archs ────────────────────────────────────────────────
    # FIGSR (Fourier-Inception-Gated SR, umzi2) — gfisr_body_half + cat_to_dim + fu (FourierUnit)
    "figsr": ["gfisr_body_half", "cat_to_dim", "fu"],
    # ParagonSR v1 (Phhofm) — Magic Kernel Sharp upsampler (very distinctive) + conv_fuse
    "paragonsr": ["magic_upsampler", "conv_fuse"],
    # ParagonSR2 (Phhofm) — dual-path: detail_gain scalar + conv_in/body at root (no "base." prefix — that's the upsampler submodule)
    "paragonsr2": ["detail_gain", "conv_in.weight"],
    # MambaIR (ECCV2024) — VSSBlock's SS2D signature
    "mambair": [".self_attention.x_proj_weight"],
    # MambaIRv2 (CVPR2025) — ASSM/AttentiveLayer signature
    "mambairv2": [".embeddingA.weight", ".assm.selectiveScan.x_proj_weight"],
    # AetherNet (Phhofm, NeoSR) — conv_after_body + stages + conv_before_upsample
    "aethernet": ["conv_after_body", "stages", "conv_before_upsample"],
    # HASN (NeoSR "📦 Autres") — 6 fixed conv_1/block_1../block_6/conv_2/upsampler
    "hasn": ["block_1.dwconv.conv.weight", "block_6.dwconv.conv.weight", "esa.conv_f"],
    # MSDAN (NeoSR "📦 Autres") — BSConv_first/DFEB/upsampler.BSConv_2/BSConv_3.
    # Weights are IDENTICAL regardless of trained scale (upscale_factor is a
    # plain int, never a saved buffer) — scale_hint (UI dropdown) is REQUIRED.
    "msdan": ["BSConv_first.pw.weight", "upsampler.BSConv_2.pw.weight", "upsampler.BSConv_3.pw.weight"],
    # CFSR (NeoSR "📦 Autres") — LargeKernelConv reparam (lkb_reparam) + ConvMod attn.
    # First two sigs are EXACT top-level keys (layers.0.blocks.0....) — must match
    # detect_arch_from_state()'s "ends in .weight/.bias → exact key" rule.
    "cfsr": ["layers.0.blocks.0.attn.a.2.lkb_reparam.weight", "layers.0.blocks.0.mlp.fc2.weight",
             "conv_after_body.weight"],

    # ── v2.5.5 : nouveaux moteurs ─────────────────────────────────────────────
    # OSEDiff : One-Step Efficient Diffusion SR (diffusion UNet + VAE, SD-based)
    # Clés typiques : unet.conv_in, vae.encoder.conv_in, time_embedding ou time_embed
    "osediff": ["unet.conv_in.weight", "vae.encoder.conv_in.weight"],
    # VOSR : diffusion-based SR (CVPR 2026 track) — possible DM-style keys
    # Alternate: scheduler.alphas_cumprod (if checkpoint contains scheduler)
    "vosr": ["model.diffusion_model.input_blocks.0.0.weight", "model.diffusion_model.out.2.weight"],
    # TVT : Transformer Video Tokens — video SR, temporal attention
    "tvt": ["temporal_attn.0.norm1.weight", "frame_token.weight"],
    # DAM-VSR : Dual Attention Mechanism Video SR — temporal + spatial dual attn
    "dam_vsr": ["dual_attn.temporal_attn.to_q.weight", "flow_warp.weight"],

    # ── v2.5.8 : archs traiNNer-redux custom (Quick Upscale, batch 1) ──────────
    # NOTE: "lmlt"/"eimn" ci-dessus visent déjà neosr_general_runner (structure de
    # poids identique entre traiNNer-redux et neosr pour ces deux archs) — pas de
    # doublon ici, le bloc manuel traiNNer de universal_runner.py/persistent_
    # upscale_worker.py pour lmlt/eimn n'est donc jamais atteint par ce routing
    # (inoffensif, juste redondant).
    "artcnn": ["depth_to_space.upscale.0.weight", "res_block.0.conv.0.0.weight"],
    "lkfmixer": ["layers.layers.0.FDB.c1_d.weight", "upsampler.upsampleOneStep.0.weight"],
    "rtmosr": [".conv.0.poll.1.alpha", "to_img.0.alpha"],
    "mosrv2": ["to_img.MetaUpsample", "gblocks."],
    "mosrv2_neosr": ["gblocks.", "to_img.0.weight"],   # NeoSR's (checked after the MetaUpsample one)
    "sebica": ["body.0.attn.channel_attention_forward.1.weight"],
    "man": ["body.0.body_t.att.0.weight", "add_mean.weight"],
    "elan": [".modules_lfe.lfe_0.", ".modules_gmsa.gmsa_0."],

    # ── v2.6.0 : archs vidéo/temporelles traiNNer-redux (Quick Upscale séquence) ──
    # traiNNer/archs/arch_info.py::TEMPORAL_ARCHS = {tscunet, temporalspan,
    # temporalspanv2, tfdat}. Toutes prennent (B,T,C,H,W) et sortent (B,C,H,W)
    # (frame centrale du clip). Contrairement à TVT/DAM-VSR (_VIDEO_ARCHS,
    # inférence frame unique via fallback spandrel), ces 4 archs ont un vrai
    # chemin d'inférence multi-frames dédié — voir _SEQUENCE_ARCHS ci-dessous.
    # TSCUNet (classe enregistrée directement, style SCUNet/Swin transformer) —
    # "m_head.0.weight" (nn.Sequential, distinct de temporalspanv2's "m_head.weight"
    # nu) + le buffer d'attention SCUNet-transformer très distinctif.
    # m_layers.0.m_head.0.weight: per-frame heads, absent from the image SCUNet
    # (scunet_aaf6aa), which the bare m_head.0.weight signature took for TSCUNet.
    "tscunet": ["m_head.0.weight", "m_layers.0.m_head.0.weight",
                "trans_block.msa.relative_position_params"],
    # TemporalSPAN — fusion précoce : conv dédiée par frame (center + history)
    "temporalspan": ["center_conv.eval_conv.weight", "history_convs.0.eval_conv.weight"],
    # TemporalSPANv2 — m_head est un nn.Conv2d nu (PAS de ".0.", à la différence
    # de tscunet) + blocs SPAB internes préfixés "m_layers.N.blocks.M.c1_r...".
    "temporalspanv2": ["m_head.weight", "m_layers.0.blocks.0.c1_r.eval_conv.weight"],
    # TFDAT — flow_net (optical flow léger) + temporal_fuse, uniques à cette arch.
    "tfdat": ["flow_net.flow_pred.weight", "temporal_fuse.0.weight", "conv_first.weight"],
    # TFDAT v2 (Kim2091, branch tfdat-v2 — e.g. 2x_bndl_animefilm_v3): pyramid flow
    # decoder + confidence-gated fusion. Incompatible with v1 checkpoints.
    "tfdatv2": ["flow_net.dec1.0.weight", "temporal_fusion.fuse.0.weight", "align_refine.body.0.weight"],

    # ── DocRes (Restormer-style document restoration, traiNNer native) ────────
    # 4-stage encoder/decoder U-Net (encoder_level1..3/latent + decoder_level1..3
    # + refinement), scale FIXE 1x (deskew/nettoyage scan, jamais d'upscale).
    # docres_base/docres_large share the same class — dim/blocks/heads all
    # derived from real tensor shapes, no separate signature needed per variant.
    "docres": ["patch_embed.proj.weight", "encoder_level1.0.attn.temperature",
               "refinement.0.attn.temperature"],
}

# Modèles à renvoyer avec message spécialisé (pas d'inférence image directe)
_DIFFUSION_ARCHS = {"osediff", "vosr"}          # besoins runtime diffusion
_VIDEO_ARCHS     = {"tvt", "dam_vsr"}            # besoin entrée multi-frames (pas de chemin dédié — fallback frame unique)
# Archs vidéo/temporelles AVEC un vrai chemin d'inférence multi-frames (fenêtre
# glissante, dossier/batch uniquement — voir _build_sequence_model /
# _sequence_video_upscale_folder plus bas).
_SEQUENCE_ARCHS  = {"tscunet", "temporalspan", "temporalspanv2", "tfdat", "tfdatv2", "motsr"}


def _engine_base() -> str:
    """Racine des moteurs (engine_paths.base_engine_path : ~/USRS, ou la racine de dev isolée)."""
    try:
        from src.core.engine_paths import base_engine_path as _ebp
    except ImportError:  # venv runner: engine_paths.py ships loose next to this file
        from engine_paths import base_engine_path as _ebp
    return _ebp()


# Keys that rule an arch out even when its signature matches (a more specific arch
# shares the same prefix and is checked later in ARCH_SIGNATURES).
_ARCH_SIGNATURE_EXCLUDES = {
    "swinir": (".self_attention.x_proj_weight",),  # MambaIR (same residual_group.blocks + conv_after_body)
    "dat": ("before_RG.norm.weight",),       # DAT-iLN (LayerNorm instead of conv before RG)
    "man_neosr": ("sub_mean.weight",),        # traiNNer-redux MAN has MeanShift layers
}


def detect_arch_from_state(state_dict: dict) -> Optional[str]:
    """Attempt to detect architecture from state_dict key patterns.

    Signatures ending in '.weight' or '.bias' require an EXACT key match
    (prevents false positives, e.g. RCAN's 'body.0.body.0.body.0.weight'
    matching compact's 'body.0.weight' as a substring).
    All other signatures (prefix/module-name patterns) use substring match.
    """
    keys     = set(state_dict.keys())
    keys_str = " ".join(keys)
    for arch, signatures in ARCH_SIGNATURES.items():
        if any(x in keys_str for x in _ARCH_SIGNATURE_EXCLUDES.get(arch, ())):
            continue
        match = True
        for sig in signatures:
            if sig.endswith(".weight") or sig.endswith(".bias"):
                # Full parameter name → must be an exact key
                if sig not in keys:
                    match = False
                    break
            else:
                # Partial pattern (module prefix, class name, buffer) → substring ok
                if sig not in keys_str:
                    match = False
                    break
        if match:
            return arch
    return None


def extract_weights(checkpoint: dict) -> dict:
    """Extract generator weights from various checkpoint formats."""
    for key in ("params_ema", "params_g", "params", "model", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    # If top-level keys look like weight tensors, return as-is
    if any(k.endswith(".weight") for k in checkpoint.keys()):
        return checkpoint
    return checkpoint


def detect_scale_from_state(state_dict: dict, default: int = 4) -> int:
    """Try to infer scale factor from upsampling layers. Returns `default` when no rule
    matches (default=0 lets the caller know the scale is unknown)."""
    # Archs rebuilt by arch_infer (HEART/NERVE/WireSR/MoTSR/TFDAT v1-v2) know their real
    # scale; the generic shuffle heuristic below reads TFDAT's internal x4 upsampler.
    try:
        try:
            from src.core import arch_infer as _ai
        except ImportError:
            import arch_infer as _ai
        _a = _ai.detect(state_dict)
        if _a:
            _kw = _ai.describe_state(_a, state_dict)
            if _kw and _kw.get("scale"):
                return int(_kw["scale"])
    except Exception:
        pass
    # DocRes (document restoration) is always 1x; its Restormer-style convs read 8x
    if "refinement.0.attn.temperature" in state_dict:
        return 1
    # MoSRv2 / RTMoSR unshuffle_mod: a 2x model unshuffles the input by 2 and then runs
    # an internal x4 upsampler (MetaUpsample / to_img read 4). The unshuffled input conv
    # (3 * u^2 channels) gives the real scale: 4 // u.
    _in_w = None
    if "gblocks.1.weight" in state_dict and "gblocks.0.weight" not in state_dict:
        _in_w = state_dict["gblocks.1.weight"]          # MoSRv2 (gblocks.0 = PixelUnshuffle)
    elif "to_feat.1.conv2.weight" in state_dict:
        _in_w = state_dict["to_feat.1.conv2.weight"]    # RTMoSR (to_feat.0 = PixelUnshuffle)
    if _in_w is not None and len(_in_w.shape) == 4:
        _u = int(round(math.sqrt(int(_in_w.shape[1]) / 3)))
        if _u in (2, 4) and 3 * _u * _u == int(_in_w.shape[1]):
            return 4 // _u
    # pixel_shuffle convs: out_channels = in_channels * factor^2. A 4x upsampler is often
    # two x2 stages (upsample.0 / upsample.2, HAT/SwinIR/DAT…): the stages multiply.
    # (The first stage alone used to be returned, reading every such 4x model as 2x.)
    _stages = []
    for key in state_dict:
        if "upsample" in key.lower() or "upconv" in key.lower() or "pixel_shuffle" in key.lower():
            tensor = state_dict[key]
            if hasattr(tensor, "shape") and len(tensor.shape) == 4:
                out_ch = tensor.shape[0]
                in_ch = tensor.shape[1]
                if out_ch > in_ch:
                    f = int(round(math.sqrt(out_ch / in_ch)))
                    if f in (2, 3, 4, 8) and f * f * in_ch == out_ch:
                        _stages.append(f)
    if _stages:
        scale = 1
        for f in _stages:
            scale *= f
        if scale in (2, 3, 4, 8):
            return scale
    # DySample upsampler (SPANPlus): offset channels = 4 * 2 * scale^2 = 8 * scale^2
    # init_pos / offset.weight both use the same channel count
    if "upsampler.offset.weight" in state_dict:
        offset_ch = state_dict["upsampler.offset.weight"].shape[0]
        scale_sq = offset_ch / 8.0
        s = int(math.sqrt(scale_sq))
        if s >= 1 and s * s == int(scale_sq):
            return s
    # ESC DySample upsampler: to_img.offset.weight channels = groups(4) * 2 * scale^2 = 8 * scale^2
    if "to_img.offset.weight" in state_dict:
        offset_ch = state_dict["to_img.offset.weight"].shape[0]
        scale_sq = offset_ch / 8.0
        s = int(math.sqrt(scale_sq))
        if s >= 1 and s * s == int(scale_sq):
            return s
    # SMoSR: MetaUpsample buffer stores [254, method_idx, scale, in_dim, out_dim, mid_dim, group, rep]
    if "upsampler.MetaUpsample" in state_dict:
        meta = state_dict["upsampler.MetaUpsample"]
        try:
            s = int(meta[2].item())
            if s in (1, 2, 3, 4, 8):
                return s
        except Exception:
            pass
    # GFISRv2: UniUpsampleV3 stores same MetaUpsample at upscale.MetaUpsample
    if "upscale.MetaUpsample" in state_dict:
        meta = state_dict["upscale.MetaUpsample"]
        try:
            s = int(meta[2].item())
            if s in (1, 2, 3, 4, 8):
                return s
        except Exception:
            pass
    # SpanC: MetaIGConv buffer stores list of scale values as uint8 tensor
    # eval_base_scale cannot be recovered from state_dict — use min of scale_list as default
    if "MetaIGConv" in state_dict:
        meta = state_dict["MetaIGConv"]
        try:
            scales = [int(v.item()) for v in meta]
            if scales:
                # IGConv decodes any scale of the list; traiNNer-redux validates at its
                # eval_base_scale default (2). Max (4) was returned before: 2x-trained
                # SpanPP models came out 4x. The UI scale still overrides this.
                s = 2 if 2 in scales else min(scales)
                if s in (1, 2, 3, 4, 8):
                    return s
        except Exception:
            pass
    # conv upsampler (SPANPlus 1x without DySample): always 1x
    if "upsampler.weight" in state_dict and "upsampler.bias" in state_dict:
        if "upsampler.offset" not in " ".join(state_dict.keys()):
            return 1
    # ninasr tail: tail.1 is Conv2d with out_ch = 3 * scale^2
    # eimn tail:   tail.0 is Conv2d with out_ch = 3 * scale^2
    # lmlt upsample: to_img.0 is Conv2d with out_ch = 3 * scale^2
    for probe_key in ("tail.1.weight", "tail.0.weight", "to_img.0.weight"):
        if probe_key in state_dict:
            ch = state_dict[probe_key].shape[0]
            s = int(round(math.sqrt(ch / 3)))
            # s == 1 (3 channels) is also the plain last conv after an upsampler (RCAN's
            # tail.1): not proof of a 1x model.
            if s >= 2 and s * s * 3 == ch:
                return s
    # Generic: the last conv of a pixel-shuffle upsampler outputs 3 * scale^2 channels
    # (12 in 2x, 48 in 4x). Without it, every 2x model whose last layer has another name
    # (depth_to_space, dim_to_in, to_img.0.conv1, tail.2, up.0, conv_near…) or fewer
    # outputs than inputs (12 < 48) was read as 4x and failed to load.
    # Only output-like layer names (safetensors keys are sorted alphabetically, so "the
    # last key" is not the output layer: a 48-channel body conv read as 4x), and only
    # when every such layer agrees.
    _out_name = re.compile(r"(^|\.)(up\w*|tail|conv_last|last\w*|to_img|depth_to_space|"
                           r"dim_to_in|conv_near|conv_out|out)(\.|$)")
    found = set()
    for k, t in state_dict.items():
        # out == in channels = a feature layer (CRAFT's 48->48 "last_fc" read a 1x model as 4x)
        if (k.endswith("weight") and hasattr(t, "shape") and len(t.shape) == 4
                and t.shape[1] != 3 and int(t.shape[0]) != int(t.shape[1]) and _out_name.search(k)):
            s = {12: 2, 27: 3, 48: 4, 192: 8}.get(int(t.shape[0]))
            if s:
                found.add(s)
    if len(found) == 1:
        return found.pop()
    # SRVGGNetCompact (compact / ultracompact / superultracompact): body.N, the last
    # conv outputs 3 * scale^2.
    _body = [(int(m.group(1)), t) for k, t in state_dict.items()
             if (m := re.match(r"^body\.(\d+)\.weight$", k)) and hasattr(t, "shape") and len(t.shape) == 4]
    if _body and "body.0.weight" in state_dict and not any(k.startswith("conv_first") for k in state_dict):
        s = {3: 1, 12: 2, 27: 3, 48: 4, 192: 8}.get(int(max(_body, key=lambda x: x[0])[1].shape[0]))
        if s:
            return s
    return default


class _ShapeOnly:
    """Header-only stand-in for a tensor: .shape, no data (scale detection)."""
    def __init__(self, shape):
        self.shape = tuple(shape)


def peek_model_scale(model_path: str) -> int:
    """Scale of a .safetensors model from its header (large tensors are never read;
    tensors of 16 values or less are, for the scale metadata some archs store).
    0 = unknown (.pth / .onnx / no rule)."""
    try:
        if not model_path.lower().endswith(".safetensors") or not os.path.isfile(model_path):
            return 0
        from safetensors import safe_open
        sd = {}
        with safe_open(model_path, framework="numpy") as f:
            for k in f.keys():
                shp = f.get_slice(k).get_shape()
                n = 1
                for d in shp:
                    n *= int(d)
                sd[k] = f.get_tensor(k) if n <= 16 else _ShapeOnly(shp)
        return int(detect_scale_from_state(sd, default=0) or 0)
    except Exception:
        return 0


def _peek_arch_from_model_path(model_path: str) -> Optional[str]:
    """Lightweight architecture detection from a checkpoint on disk, without
    doing the full in-process model load. Reads only the safetensors header
    (key names/shapes, zero tensor data) or the full torch checkpoint for
    .pth (metadata isn't separable there). Used by BOTH upscale_image() and
    upscale_folder() so the "does this arch need a sequence/multi-frame
    dataset" gate is evaluated identically in both call paths — the app hit
    this exact class of bug twice already this session (a detection added to
    a runner that the caller never routed to): a single shared function makes
    that impossible to reproduce here.
    Returns None on any failure (caller must treat that as "unknown/normal").
    """
    try:
        if not os.path.isfile(model_path):
            return None
        if model_path.lower().endswith(".safetensors"):
            import struct as _struct, json as _json
            with open(model_path, "rb") as _mf:
                _hlen = _struct.unpack("<Q", _mf.read(8))[0]
                _hmeta = _json.loads(_mf.read(_hlen))
            _keys = set(k for k in _hmeta if k != "__metadata__")
            return detect_arch_from_state({k: None for k in _keys})  # type: ignore
        if not TORCH_AVAILABLE:
            # Portable build: no torch in-process. The engine venv lists the key names
            # (A02: a temporal .pth video was otherwise processed frame by frame).
            import subprocess as _sp, json as _json
            _py = next((p for p in (_TRAINNER_VENV_PY, _NEOSR_VENV_PY) if p and os.path.isfile(p)), None)
            if not _py:
                return None
            _code = ("import sys,json,torch;ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False)\n"
                     "for k in ('params_ema','params','state_dict','model'):\n"
                     "    if isinstance(ck,dict) and isinstance(ck.get(k),dict): ck=ck[k]; break\n"
                     "print(json.dumps([k for k in ck.keys() if isinstance(k,str)]))")
            _r = _sp.run([_py, "-c", _code, model_path], capture_output=True, text=True, timeout=120,
                         creationflags=0x08000000 if sys.platform == "win32" else 0)
            if _r.returncode != 0:
                return None
            _keys = _json.loads(_r.stdout.strip().splitlines()[-1])
            return detect_arch_from_state({k: None for k in _keys})  # type: ignore
        _ck = torch.load(model_path, map_location="cpu", weights_only=False)
        _sd = extract_weights(_ck)
        return detect_arch_from_state(_sd)
    except Exception:
        return None


# ─── Color Fix (post-upscale color drift correction) ────────────────────────

def _cf_resolve_device(device_pref: str) -> str:
    """
    Retourne le device effectif selon préférence et disponibilité.
    'auto'  → cuda si dispo, sinon cpu
    'cuda'  → force GPU (fallback cpu si absent)
    'trt'   → torch.compile max-autotune sur GPU (fallback cuda, puis cpu)
    'cpu'   → toujours PIL
    """
    if device_pref in ("cuda", "trt"):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return device_pref   # préserve la distinction cuda/trt
        return "cpu"
    if device_pref == "auto":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cpu"


# ── Cache pour la version torch.compile du BoxBlur (TRT mode) ────────────────
_cf_trt_fn = None   # type: ignore


def _cf_get_trt_boxblur():
    """
    Retourne _cf_boxblur_gpu compilé avec torch.compile(mode='max-autotune').
    Premier appel déclenche l'autotuning Triton (~3 s sur RTX) — résultat mis en cache.
    Fallback transparent vers _cf_boxblur_gpu si torch.compile non disponible.
    """
    global _cf_trt_fn
    if _cf_trt_fn is None:
        if TORCH_AVAILABLE:
            try:
                _cf_trt_fn = torch.compile(
                    _cf_boxblur_gpu,
                    mode="max-autotune",
                    fullgraph=False,
                )
            except Exception:
                _cf_trt_fn = _cf_boxblur_gpu   # torch < 2.0 → fallback
        else:
            _cf_trt_fn = _cf_boxblur_gpu
    return _cf_trt_fn


def _cf_boxblur_gpu(t: "torch.Tensor", r: int, c: int) -> "torch.Tensor":
    """
    BoxBlur séparable 1D sur GPU (PyTorch F.conv2d).
    Entrée/sortie : [1, C, H, W] float32 sur CUDA.
    Utilise replicate-padding → comportement bord identique à PIL BoxBlur.
    """
    import torch.nn.functional as F
    r = max(1, int(r))
    k = 2 * r + 1
    dev = t.device
    # Filtres séparables : horizontal puis vertical
    kh = torch.ones(c, 1, 1, k, device=dev, dtype=torch.float32) / k
    kv = torch.ones(c, 1, k, 1, device=dev, dtype=torch.float32) / k
    t = F.pad(t, (r, r, 0, 0), mode="replicate")
    t = F.conv2d(t, kh, groups=c)
    t = F.pad(t, (0, 0, r, r), mode="replicate")
    t = F.conv2d(t, kv, groups=c)
    return t


def color_fix_image(
    sr_np: "np.ndarray",
    lq_pil: "Image.Image",
    method: str = "wavelet",
    wavelets: int = 4,
    radius: int = 10,
    fast: bool = False,
    strength: float = 1.0,
    planes: "Optional[list]" = None,
    device: str = "auto",
    ref_pil: "Optional[Image.Image]" = None,
) -> "np.ndarray":
    """
    Fix color drift post-upscale. Equivalent to vs_colorfix (pifroggi).

    method='wavelet': ATWT (à trous wavelet transform) multi-niveaux.
        Décompose SR et REF en N niveaux de fréquences progressives.
        Remplace uniquement le résidu basse-fréquence (couleur globale) du SR par celui du REF.
        Préserve intégralement le détail haute-fréquence du SR (textures, contours).
        wavelets=4 → rayon effectif 2^4=16px. Equiv. vs_colorfix.wavelet(wavelets=4).
        Recommandé : 3–5.

    method='average': region-average single-level (equivalent vs_colorfix.average).
        fast=False → BoxBlur (précis). fast=True → downscale+upscale (10× plus rapide).
        Recommandé : radius 5–30.

    device: 'auto'  → CUDA si dispo sinon CPU.
            'cuda'  → force GPU PyTorch F.conv2d.
            'trt'   → torch.compile max-autotune (Triton autotuning, ~3s warmup).
            'cpu'   → toujours PIL.
        GPU (~10-30× plus rapide que PIL sur 4K). Pas de VapourSynth requis.

    ref_pil: image de référence alternative (vs LQ par défaut).
        Recommandé : source haute qualité (PNG/TIFF 16-bit) pour éviter le banding.
        Si None → utilise lq_pil.

    planes : canaux à corriger [0,1,2]=RGB (défaut). Ex: [1,2]=GB seulement.
    strength: blend [0.0=off, 1.0=correction complète].

    Returns: float32 [H, W, 3] in [0, 1]
    """
    from PIL import ImageFilter as _IF

    if planes is None:
        planes = [0, 1, 2]
    if strength <= 0.0 or not planes:
        return sr_np

    h, w = sr_np.shape[:2]
    nc = sr_np.shape[2]  # nombre de canaux (3 pour RGB)

    # ── Source de référence (LQ par défaut, ou image externe) ────────────────
    if isinstance(ref_pil, np.ndarray):
        # 16-bit reference loaded as float (load_color_fix_ref): wavelet / average keep
        # its precision; LAB / histogram / colormap work on its 8-bit version.
        from src.core.image_io import resize_float
        lq_up = np.clip(resize_float(ref_pil, (w, h)), 0.0, 1.0).astype(np.float32)
        _src_pil = Image.fromarray((ref_pil * 255.0).round().clip(0, 255).astype(np.uint8))
    else:
        _src_pil = ref_pil if ref_pil is not None else lq_pil
        lq_up = np.array(_src_pil.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0

    # ── Résolution du device ──────────────────────────────────────────────────
    resolved = _cf_resolve_device(device)
    use_gpu  = (resolved in ("cuda", "trt") and TORCH_AVAILABLE)
    _boxblur = _cf_get_trt_boxblur() if resolved == "trt" else _cf_boxblur_gpu

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _boxblur_pil(arr: "np.ndarray", r: int) -> "np.ndarray":
        pil = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))
        return np.array(pil.filter(_IF.BoxBlur(max(1, r))), dtype=np.float32) / 255.0

    def _np_to_t(arr: "np.ndarray") -> "torch.Tensor":
        """[H,W,C] numpy → [1,C,H,W] CUDA float32."""
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to("cuda", dtype=torch.float32)

    def _t_to_np(t: "torch.Tensor", clamp: bool = True) -> "np.ndarray":
        """[1,C,H,W] CUDA → [H,W,C] numpy float32. clamp=False for SIGNED data (the
        colour correction LQ − SR: clamping it to [0, 1] dropped every negative
        correction, so the GPU Color Fix could never darken)."""
        t = t.squeeze(0).permute(1, 2, 0)
        return (t.clamp(0, 1) if clamp else t).cpu().float().numpy()

    # ── WAVELET ───────────────────────────────────────────────────────────────
    if method == "wavelet":
        if use_gpu:
            try:
                sr_t = _np_to_t(sr_np)
                lq_t = _np_to_t(lq_up)
                for level in range(max(1, int(wavelets))):
                    r = 2 ** level  # 1, 2, 4, 8, 16, 32...
                    sr_t = _boxblur(sr_t, r, nc)
                    lq_t = _boxblur(lq_t, r, nc)
                correction = _t_to_np(lq_t - sr_t, clamp=False)
            except Exception:
                # Fallback PIL si CUDA plante (OOM, etc.)
                use_gpu = False

        if not use_gpu:
            sr_low = sr_np.copy()
            lq_low = lq_up.copy()
            for level in range(max(1, int(wavelets))):
                r = 2 ** level
                sr_low = _boxblur_pil(sr_low, r)
                lq_low = _boxblur_pil(lq_low, r)
            correction = lq_low - sr_low

    # ── AVERAGE ───────────────────────────────────────────────────────────────
    elif method == "average":
        if fast:
            # Downscale+upscale (chaiNNer-style) — toujours CPU (déjà très rapide)
            factor = max(2, int(radius))
            sm_w, sm_h = max(1, w // factor), max(1, h // factor)
            sr_dn = Image.fromarray((sr_np * 255.0).clip(0, 255).astype(np.uint8)).resize(
                (sm_w, sm_h), Image.BILINEAR)
            lq_dn = _src_pil.resize((sm_w, sm_h), Image.BILINEAR)
            sr_low = np.array(sr_dn.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
            lq_low = np.array(lq_dn.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        elif use_gpu:
            try:
                sr_t = _np_to_t(sr_np)
                lq_t = _np_to_t(lq_up)
                sr_t = _boxblur(sr_t, int(radius), nc)
                lq_t = _boxblur(lq_t, int(radius), nc)
                correction = _t_to_np(lq_t - sr_t, clamp=False)
            except Exception:
                use_gpu = False

        if not use_gpu and not fast:
            sr_low = _boxblur_pil(sr_np, int(radius))
            lq_low = _boxblur_pil(lq_up, int(radius))
            correction = lq_low - sr_low

        if fast:
            correction = lq_low - sr_low

    # ── LAB (vs_colorfix v2.0) ─────────────────────────────────────────────────
    # Corrects chroma (a*, b*) in CIE-LAB space while preserving SR luminance (L*).
    # Best for: stylized content, strong saturation drift, when luminance is correct.
    elif method == "lab":
        try:
            from PIL import ImageCms as _cms
            _srgb  = _cms.createProfile("sRGB")
            _lab_p = _cms.createProfile("LAB")
            _to_lab   = _cms.buildTransformFromOpenProfiles(_srgb, _lab_p, "RGB", "LAB")
            _to_rgb   = _cms.buildTransformFromOpenProfiles(_lab_p, _srgb, "LAB", "RGB")

            sr_u8  = (sr_np * 255.0).clip(0, 255).astype(np.uint8)
            lq_u8  = np.array(_src_pil.resize((w, h), Image.BICUBIC))

            sr_pil = Image.fromarray(sr_u8)
            lq_pil_r = Image.fromarray(lq_u8)

            sr_lab  = np.array(_cms.applyTransform(sr_pil,  _to_lab)).astype(np.float32)
            lq_lab  = np.array(_cms.applyTransform(lq_pil_r, _to_lab)).astype(np.float32)

            # Replace a* and b* from ref, keep L* from SR
            mixed_lab = sr_lab.copy()
            mixed_lab[:, :, 1] = lq_lab[:, :, 1]  # a*
            mixed_lab[:, :, 2] = lq_lab[:, :, 2]  # b*

            mixed_u8  = np.array(_cms.applyTransform(
                Image.fromarray(mixed_lab.clip(0, 255).astype(np.uint8)), _to_rgb))
            # the correction is the LAB round-trip delta, added to the full-precision SR
            delta = (mixed_u8.astype(np.float32) - sr_u8.astype(np.float32)) / 255.0
            fixed_lab = sr_np.copy()
            for c in planes:
                fixed_lab[:, :, c] = sr_np[:, :, c] + delta[:, :, c]

            if strength < 1.0:
                fixed_lab = sr_np * (1.0 - strength) + fixed_lab * strength
            return np.clip(fixed_lab, 0.0, 1.0)
        except Exception:
            # Fallback: simple channel-mean correction if ImageCms unavailable
            sr_mean = sr_np.mean(axis=(0, 1))
            lq_mean = lq_up.mean(axis=(0, 1))
            fixed_lab = sr_np.copy()
            for c in planes:
                if sr_mean[c] > 1e-6:
                    fixed_lab[:, :, c] = np.clip(sr_np[:, :, c] * (lq_mean[c] / sr_mean[c]), 0, 1)
            if strength < 1.0:
                fixed_lab = sr_np * (1.0 - strength) + fixed_lab * strength
            return np.clip(fixed_lab, 0.0, 1.0)

    # ── HISTOGRAM matching (vs_colorfix v2.0) ──────────────────────────────────
    # Matches the full histogram (CDF) of each channel from ref to SR.
    # Best for: dramatic color palette differences, strong stylization.
    elif method == "hist":
        fixed_hist = sr_np.copy()
        for c in planes:
            src_ch  = sr_np[:, :, c].ravel()
            ref_ch  = lq_up[:, :, c].ravel()
            # Build CDFs
            bins = 256
            src_hist, edges = np.histogram(src_ch, bins=bins, range=(0.0, 1.0))
            ref_hist, _     = np.histogram(ref_ch, bins=bins, range=(0.0, 1.0))
            src_cdf = np.cumsum(src_hist).astype(np.float64)
            ref_cdf = np.cumsum(ref_hist).astype(np.float64)
            if src_cdf[-1] > 0:
                src_cdf /= src_cdf[-1]
            if ref_cdf[-1] > 0:
                ref_cdf /= ref_cdf[-1]
            # Mapping: for each src bin, find ref bin with matching CDF
            lut = np.interp(src_cdf, ref_cdf, np.linspace(0.0, 1.0, bins))
            # Apply LUT
            src_idx = np.clip((src_ch * (bins - 1)).astype(int), 0, bins - 1)
            fixed_hist[:, :, c] = lut[src_idx].reshape(h, w)

        if strength < 1.0:
            fixed_hist = sr_np * (1.0 - strength) + fixed_hist * strength
        return np.clip(fixed_hist, 0.0, 1.0)

    # ── COLORMAP / LUT quantile (vs_colorfix v2.0 style) ────────────────────────
    # Quantile-based sparse LUT mapping from SR colors to reference colors.
    # Faster than full histogram matching — 64 quantile samples, interp LUT.
    # Best for: fast global correction, large images, mild stylization drift.
    elif method == "colormap":
        n_q = 64
        fixed_cmap = sr_np.copy()
        for c in planes:
            src_ch = sr_np[:, :, c].ravel().astype(np.float64)
            ref_ch = lq_up[:, :, c].ravel().astype(np.float64)
            quants = np.linspace(0.0, 1.0, n_q)
            src_q  = np.quantile(src_ch, quants)
            ref_q  = np.quantile(ref_ch, quants)
            # Ensure src_q strictly increasing for interp
            src_q, idx = np.unique(src_q, return_index=True)
            ref_q = ref_q[idx]
            mapped = np.interp(src_ch, src_q, ref_q).clip(0.0, 1.0)
            fixed_cmap[:, :, c] = mapped.reshape(h, w)

        if strength < 1.0:
            fixed_cmap = sr_np * (1.0 - strength) + fixed_cmap * strength
        return np.clip(fixed_cmap, 0.0, 1.0)

    # ── GUIDED (aniCC / PACC V4, Bendel) ─────────────────────────────────────
    # Learned colour transfer from the LR reference, handles non-uniform shifts.
    # Model weights downloaded on first use (guided_color_fix.ensure_weights).
    elif method == "guided":
        try:
            if TORCH_AVAILABLE:
                try:
                    from src.core.guided_color_fix import run_guided
                except ImportError:
                    from guided_color_fix import run_guided
                guided = run_guided(sr_np, _src_pil, device=device)
            else:
                guided = _guided_color_fix_subprocess(sr_np, _src_pil, device=device)
            correction = guided - sr_np
        except Exception as e:
            raise RuntimeError(f"Color Fix guided (aniCC) indisponible : {e}") from e

    else:
        return sr_np

    fixed = sr_np.copy()
    for c in planes:
        fixed[:, :, c] = np.clip(sr_np[:, :, c] + correction[:, :, c], 0.0, 1.0)

    if strength < 1.0:
        fixed = sr_np * (1.0 - strength) + fixed * strength
    return np.clip(fixed, 0.0, 1.0)


# ─── Tiled inference ─────────────────────────────────────────────

def tile_inference(model, img_tensor: "torch.Tensor", tile_size: int = 256,
                   tile_pad: int = 32, scale: int = 4,
                   stop_event=None) -> "torch.Tensor":
    """
    Process an image in overlapping tiles to save VRAM.
    Checks stop_event between tiles — raises RuntimeError("stopped") if set.

    Args:
        model: PyTorch model (callable)
        img_tensor: Input tensor [1, C, H, W]
        tile_size: Size of each tile (pixels)
        tile_pad: Overlap padding
        scale: Upscale factor
        stop_event: threading.Event — checked between tiles

    Returns:
        Output tensor [1, C, H*scale, W*scale]
    """
    batch, channel, height, width = img_tensor.shape
    out_h = height * scale
    out_w = width * scale
    output = img_tensor.new_zeros((batch, channel, out_h, out_w))
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            # Check stop between tiles — immediate abort
            if stop_event and stop_event.is_set():
                raise RuntimeError("stopped")
            # Input tile bounds (with padding)
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            input_start_x = max(ofs_x - tile_pad, 0)
            input_end_x = min(ofs_x + tile_size + tile_pad, width)
            input_start_y = max(ofs_y - tile_pad, 0)
            input_end_y = min(ofs_y + tile_size + tile_pad, height)

            # Input tile
            input_tile = img_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

            # Process tile
            with torch.no_grad():
                output_tile = model(input_tile)

            # Output tile bounds
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # Remove padding from output
            output_start_x_tile = (ofs_x - input_start_x) * scale
            output_end_x_tile = output_start_x_tile + tile_size * scale
            output_start_y_tile = (ofs_y - input_start_y) * scale
            output_end_y_tile = output_start_y_tile + tile_size * scale

            # Clamp to actual tile size
            output_end_x_tile = min(output_end_x_tile, output_tile.shape[3])
            output_end_y_tile = min(output_end_y_tile, output_tile.shape[2])

            # Target region in output
            target_end_x = min(ofs_x * scale + tile_size * scale, out_w)
            target_end_y = min(ofs_y * scale + tile_size * scale, out_h)
            actual_w = target_end_x - ofs_x * scale
            actual_h = target_end_y - ofs_y * scale

            output[:, :, ofs_y * scale:target_end_y, ofs_x * scale:target_end_x] = \
                output_tile[:, :, output_start_y_tile:output_start_y_tile + actual_h,
                            output_start_x_tile:output_start_x_tile + actual_w]

    return output


# ─── Image save helper ───────────────────────────────────────────

# Extension mapping for output format
_FMT_EXT = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "WEBP": ".webp",
    "TIFF": ".tif",
    "BMP": ".bmp",
}


def _save_output(
    float_arr: "np.ndarray",   # [H, W, 3] float32 in [0, 1]
    output_path: str,
    out_format: str = "PNG",
    bit_depth: int = 8,
    quality: int = 95,
) -> None:
    """Save an upscaled float32 [H,W,3] array to disk with format and bit-depth control."""
    fmt = out_format.upper()
    save_kwargs: dict = {}

    if bit_depth == 16 and fmt in ("PNG", "TIFF"):
        # NOTE: the old Pillow trick here (Image.frombuffer("RGB", ..., "raw",
        # "RGB;16B", ...)) never actually wrote 16-bit — Pillow has no native
        # 3x16-bit-per-channel RGB image mode, so the "RGB;16B" raw decoder
        # silently downsamples into the 8-bit "RGB" mode while decoding. The
        # file it produced was confirmed 8-bit on disk (IHDR bitdepth=8) even
        # though the NumPy-2.0 byteswap/newbyteorder crash was "fixed" earlier
        # — that fix only stopped the crash, it didn't make the output real
        # 16-bit. Same root cause already diagnosed and solved for the
        # video/sequence pipeline (_save_sequence_frame, see the long comment
        # above it) — reusing that proven pyvips-based writer here instead.
        if not PYVIPS_AVAILABLE:
            raise RuntimeError(
                "Sortie 16-bit demandée mais pyvips n'est pas installé "
                "(pip install pyvips pyvips-binary). Pillow ne peut pas "
                "écrire un vrai PNG/TIFF 16-bit par canal pour du RGB."
            )
        arr16 = (float_arr * 65535.0).round().clip(0, 65535).astype(np.uint16)
        h, w, c = arr16.shape
        arr16 = np.ascontiguousarray(arr16)
        vimg = pyvips.Image.new_from_memory(arr16.tobytes(), w, h, c, "ushort")
        if fmt == "TIFF":
            vimg.write_to_file(output_path, compression="lzw")
        else:
            vimg.write_to_file(output_path)
        return
    else:
        arr8 = (float_arr * 255.0).round().clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr8)
        if fmt in ("JPEG", "JPG"):
            save_kwargs["quality"] = quality
            save_kwargs["subsampling"] = 0
        elif fmt == "WEBP":
            save_kwargs["quality"] = quality
        elif fmt == "TIFF":
            save_kwargs["compression"] = "lzw"

    pil_fmt = "JPEG" if fmt in ("JPEG", "JPG") else fmt
    img.save(output_path, format=pil_fmt, **save_kwargs)


# ─── ONNX inference ──────────────────────────────────────────────

def _onnx_upscale(
    model_path: str,
    input_path: str,
    output_path: str,
    log: Callable,
    progress_callback: Optional[Callable] = None,
    out_format: str = "PNG",
    bit_depth: int = 8,
    quality: int = 95,
) -> "Tuple[bool, str]":
    """
    Run super-resolution inference via ONNX Runtime.
    Supports any SR model exported to ONNX with input [1, 3, H, W]
    and output [1, 3, H*scale, W*scale].
    Requires: pip install onnxruntime-gpu  (or onnxruntime for CPU-only)
    """
    def _prog(v):
        if progress_callback:
            progress_callback(v)

    try:
        import onnxruntime as ort
    except ImportError:
        return False, (
            "onnxruntime non installé.\n"
            "Installez avec : pip install onnxruntime-gpu\n"
            "(ou onnxruntime pour CPU uniquement)"
        )

    if not IMAGING_AVAILABLE:
        return False, "Pillow/NumPy non installé"

    log("ONNX: chargement du modele...")
    _prog(0.08)

    providers = []
    try:
        import torch
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
    except ImportError:
        pass
    providers.append("CPUExecutionProvider")

    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        return False, f"Impossible de charger le modele ONNX : {e}"

    active = sess.get_providers()
    log(f"ONNX: providers actifs = {active}")
    _prog(0.25)

    # Load image
    log(f"Lecture : {os.path.basename(input_path)}")
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = np.transpose(img_np, (2, 0, 1))[np.newaxis]  # [1, 3, H, W]

    log(f"ONNX: inference {img.width}x{img.height}...")
    _prog(0.50)

    inp_name = sess.get_inputs()[0].name
    # Cast input to dtype expected by the model (float32 or float16)
    _inp_type = sess.get_inputs()[0].type  # e.g. 'tensor(float)' or 'tensor(float16)'
    if "float16" in _inp_type:
        img_t = img_t.astype(np.float16)
    # else keep float32 (default)
    try:
        out_np = sess.run(None, {inp_name: img_t})[0]  # [1, 3, H*s, W*s]
    except Exception as e:
        err = str(e)
        if "newbyteorder" in err:
            return False, (
                "Erreur inference ONNX : onnxruntime incompatible avec NumPy 2.0.\n"
                "Fix : pip install \"onnxruntime-gpu>=1.18.0,<1.27\"\n"
                f"(détail : {err})"
            )
        return False, f"Erreur inference ONNX : {e}"
    out_np = out_np.astype(np.float32)  # always convert output back to float32

    _prog(0.90)

    # Convert and save
    out_np = out_np.squeeze(0).clip(0, 1)         # [3, H, W]
    out_np = np.transpose(out_np, (1, 2, 0))       # [H, W, 3] float32 [0,1]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    _save_output(out_np, output_path, out_format, bit_depth, quality)

    _prog(1.0)
    log(f"ONNX OK : {output_path}")
    return True, f"ONNX upscale réussi : {os.path.basename(output_path)}"


def _onnx_subprocess_infer(model_path, input_path, output_path, log, progress_callback,
                           stop_event=None, out_format="PNG", bit_depth=8, quality=95):
    """Run ONNX inference in an engine venv (onnxruntime not bundled in frozen exe)."""
    import subprocess
    venv_py = next((p for p in [_TRAINNER_VENV_PY, _NEOSR_VENV_PY] if os.path.exists(p)), None)
    if not venv_py:
        return False, "Aucun venv moteur trouvé pour l'inférence ONNX."
    if not os.path.isfile(_ONNX_RUNNER):
        return False, f"Runner ONNX introuvable : {_ONNX_RUNNER}"
    _PROGRESS_MAP = [("[ONNX] Chargement", 0.10), ("[ONNX] Providers", 0.25),
                     ("[ONNX] Inference", 0.55), ("[ONNX] Sauvegarde", 0.92)]
    cmd = [venv_py, _ONNX_RUNNER, model_path, input_path, output_path,
           str(out_format), str(bit_depth), str(quality)]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
        killed = _run_proc_with_stop(proc, _PROGRESS_MAP, log, progress_callback, stop_event)
        proc.stdout.close()
        if killed:
            return False, "Arrêt demandé par l'utilisateur."
    except Exception as e:
        return False, f"Erreur lancement subprocess ONNX : {e}"
    if proc.returncode == 0:
        if progress_callback:
            progress_callback(1.0)
        return True, f"ONNX (subprocess) : {os.path.basename(output_path)}"
    return False, f"Subprocess ONNX a échoué (code {proc.returncode})"


# ─── Subprocess helper : non-blocking reader + immediate kill ─────

def _run_proc_with_stop(proc, progress_map, log, progress_callback, stop_event):
    """
    Drive a subprocess non-blockingly so stop_event is honoured immediately,
    even when the subprocess is silent during heavy GPU computation.

    Strategy:
      - stdout is drained by a daemon reader thread (never blocks the main thread).
      - Main thread polls proc.poll() every 50 ms and checks stop_event.
      - On stop: proc.kill() is called instantly, function returns False.

    Returns:
      (killed: bool)  — True if process was killed by stop_event.
    """
    import re as _re
    _ansi = _re.compile(r"\x1b\[[0-9;]*m|\[[0-9;]{1,6}m")

    def _reader():
        try:
            for line in proc.stdout:
                # Native libraries (onnxruntime…) may write UTF-16 text and ANSI colours:
                # shown raw they turn into garbage in the UI log.
                stripped = _ansi.sub("", line.replace("\x00", "")).rstrip()
                if not stripped.strip():
                    continue
                log(stripped)
                if progress_callback and progress_map:
                    for prefix, pct in progress_map:
                        if prefix in stripped:
                            progress_callback(pct)
                            break
        except Exception:
            pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    # Poll until the process exits or stop is requested
    while proc.poll() is None:
        if stop_event and stop_event.is_set():
            try:
                proc.kill()
            except Exception:
                pass
            t.join(timeout=1.0)
            return True   # killed
        time.sleep(0.05)

    t.join(timeout=2.0)
    return False  # natural exit


# ─── Universal subprocess fallback (spandrel + SpanPlus) ─────────

# v2.5.6: shared runtimes/.venv (legacy per-engine .venv kept as fallback).
from src.core import engine_paths as _ep
_TRAINNER_VENV_PY = _ep.resolve_engine_python(_ep.redux_path())
_NEOSR_VENV_PY = _ep.resolve_engine_python(_ep.neosr_path())
# Keep old spanplus_runner for compatibility; prefer universal_runner
_UNIVERSAL_RUNNER = os.path.join(os.path.dirname(__file__), "universal_runner.py")
_SPANPLUS_RUNNER  = os.path.join(os.path.dirname(__file__), "spanplus_runner.py")
_NEOSR_RUNNER              = os.path.join(os.path.dirname(__file__), "neosr_runner.py")
_NEOSR_GENERAL_RUNNER      = os.path.join(os.path.dirname(__file__), "neosr_general_runner.py")
_PERSISTENT_NEOSR_WORKER   = os.path.join(os.path.dirname(__file__), "persistent_neosr_worker.py")
_ONNX_RUNNER               = os.path.join(os.path.dirname(__file__), "onnx_runner.py")

# ─── Persistent NeoSR (ESC) session ─────────────────────────────────────────


def _read_json_reply(proc, who: str) -> dict:
    """Next JSON line from a persistent worker. Non-JSON lines (a print() inside an
    arch constructor, e.g. DocRes) are skipped instead of failing the session."""
    while True:
        resp_line = proc.stdout.readline()
        if not resp_line:
            raise RuntimeError(f"{who}: worker process closed stdout unexpectedly")
        resp_line = resp_line.strip()
        if resp_line.startswith("{"):
            try:
                return json.loads(resp_line)
            except ValueError:
                pass


def _drain_stderr(proc, tail) -> None:
    """Read the worker's stderr in the background: an unread PIPE fills up (warnings,
    tracebacks) and blocks the worker. The last lines are kept for error messages."""
    import threading

    def _run():
        try:
            for line in proc.stderr:
                tail.append(line.rstrip())
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


class PersistentNeoSRSession:
    """
    Garde le modèle ESC chargé dans le venv neosr entre chaque image d'un batch.
    Même protocole JSON-lines que PersistentBatchSession.
    Gain : ~3-8s par image (pas de rechargement neosr + ESC à chaque subprocess).

    Usage :
        with PersistentNeoSRSession(venv_py, model_path, tile_size, tile_pad) as s:
            ok, msg = s.infer(input_path, output_path)
    """

    def __init__(self, venv_py: str, model_path: str,
                 tile_size: int = 256, tile_pad: int = 32, use_amp: bool = False,
                 log: Optional[Callable] = None, scale_hint: int = 0):
        self._venv_py    = venv_py
        self._model_path = model_path
        self._tile_size  = tile_size
        self._tile_pad   = tile_pad
        self._use_amp    = use_amp
        self._log        = log or (lambda m: None)
        self._scale_hint = scale_hint  # required for msdan (scale not recoverable from weights)
        self._proc       = None
        self.scale: int  = 1
        self._ready      = False

    def _send(self, obj: dict) -> dict:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        return _read_json_reply(self._proc, "PersistentNeoSR")

    def start(self) -> bool:
        import subprocess as _sp
        if not os.path.isfile(self._venv_py):
            self._log(f"[ESC-Persistent] venv Python introuvable : {self._venv_py}")
            return False
        if not os.path.isfile(_PERSISTENT_NEOSR_WORKER):
            self._log(f"[ESC-Persistent] Worker introuvable : {_PERSISTENT_NEOSR_WORKER}")
            return False

        _env = os.environ.copy()
        _env["PYTHONIOENCODING"] = "utf-8"
        _env["PYTHONUTF8"] = "1"

        neosr_dir = os.path.join(_engine_base(), "neosr")
        cwd = neosr_dir if os.path.isdir(neosr_dir) else os.path.dirname(self._venv_py)
        try:
            self._proc = _sp.Popen(
                [self._venv_py, _PERSISTENT_NEOSR_WORKER],
                stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE,
                text=True, encoding="utf-8", errors="replace",
                env=_env, cwd=cwd,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
        except Exception as e:
            self._log(f"[ESC-Persistent] Échec lancement : {e}")
            return False
        from collections import deque
        self._err_tail = deque(maxlen=40)
        _drain_stderr(self._proc, self._err_tail)

        try:
            resp = self._send({
                "cmd": "init",
                "model": self._model_path,
                "tile_size": self._tile_size,
                "tile_pad":  self._tile_pad,
                "use_amp":   self._use_amp,
                "scale_hint": self._scale_hint,
            })
        except Exception as e:
            _last = [l for l in getattr(self, "_err_tail", []) if l.strip()][-1:]
            self._log(f"[ESC-Persistent] Erreur init : {e}" + (f" — {_last[0][:200]}" if _last else ""))
            self.stop()
            return False

        if resp.get("status") == "ready":
            self.scale  = resp.get("scale", 1)
            self._ready = True
            self._log(f"[ESC-Persistent] Prêt — ESC {self.scale}× (modèle chargé une fois)")
            return True
        self._log(f"[ESC-Persistent] Init refusé : {resp.get('msg', '?')}")
        self.stop()
        return False

    def infer(self, input_path: str, output_path: str) -> Tuple[bool, str]:
        if not self._ready:
            return False, "Session non prête"
        try:
            payload = {"cmd": "infer", "input": input_path, "output": output_path}
            if getattr(self, "save_opts", None):
                payload["save"] = self.save_opts
            resp = self._send(payload)
            if resp.get("status") == "ok":
                return True, "ok"
            return False, resp.get("msg", "erreur inconnue")
        except Exception as e:
            self._ready = False
            return False, str(e)

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        self._ready = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ─── Custom arch injection ───────────────────────────────────────
# These archs (gfisrv2, smosr, spanpp/SpanC, figsr) are NOT in the official
# traiNNer-redux / neosr repos. We bundle them and copy them into the engine's
# archs/ folder before inference, re-applying on every run (survives git pull,
# fresh installs). Same pattern as runner.py _patch_* methods.
_CUSTOM_ARCHS_DIR = os.path.join(os.path.dirname(_UNIVERSAL_RUNNER), "custom_archs")
# Every .py shipped in custom_archs/ (MoTSR, WireSR, TFDAT v2, ParagonSR + resampler,
# Mamba, discriminators…). Listing the folder instead of hard-coding names: a new
# custom arch file can no longer be forgotten here (several were, before 2026-10).
_CUSTOM_ARCH_FILES = sorted(
    f for f in (os.listdir(_CUSTOM_ARCHS_DIR) if os.path.isdir(_CUSTOM_ARCHS_DIR) else [])
    if f.endswith(".py") and not f.startswith("__"))
# Archs held back from public builds until their author agrees (WireSR): present only
# on the developer's machine, injected the same way when the folder exists.
_PENDING_ARCHS_DIR = os.path.join(os.path.dirname(_CUSTOM_ARCHS_DIR), "custom_archs_pending")
_PENDING_ARCH_FILES = sorted(
    f for f in (os.listdir(_PENDING_ARCHS_DIR) if os.path.isdir(_PENDING_ARCHS_DIR) else [])
    if f.endswith(".py") and not f.startswith("__"))

# NeoSR-only custom archs (not shared with traiNNer-redux — aether_arch.py imports
# from neosr.utils.registry, aether_core.py has no neosr dependency but lives
# alongside it). Same injection pattern as _CUSTOM_ARCH_FILES, neosr archs/ only.
# Source: runner.py injects the same pair at training-launch time (see its
# "custom_neosr_archs" reference) — this mirrors that for Quick Upscale inference,
# which does not go through runner.py and would otherwise never get this file.
_CUSTOM_NEOSR_ARCHS_DIR = os.path.join(os.path.dirname(_UNIVERSAL_RUNNER), "custom_neosr_archs")
_CUSTOM_NEOSR_ARCH_FILES = ["aether_core.py", "aether_arch.py"]


def _peek_safetensors_keys(path: str) -> list:
    """Read tensor names from a .safetensors header without torch (8-byte len + JSON)."""
    import json, struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    return [k for k in header.keys() if k != "__metadata__"]


# Cache: model_path → "neosr" | "neosr_general" | "trainner" (frozen routing decision)
_FROZEN_ROUTE_CACHE = {}


# Format / bit depth / quality of the current upscale_image() call, handed to the
# one-image runner subprocesses in USS_SAVE_OPTS (read by runner_save.save_float).
_SAVE_OPTS: dict = {}
# Tile / tile pad / AMP of the current call, for the NeoSR runners (USS_INFER_OPTS).
_INFER_OPTS: dict = {}


def _runner_env(**extra):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **extra}
    if _SAVE_OPTS:
        env["USS_SAVE_OPTS"] = json.dumps(_SAVE_OPTS)
    if _INFER_OPTS:
        env["USS_INFER_OPTS"] = json.dumps(_INFER_OPTS)
    return env


def _read_float01(path):
    """Saved output as float32 RGB [0, 1], keeping 16-bit precision."""
    from src.core.image_io import read_float01
    return read_float01(path)


def _cf_try(sr, lq, log, **kw):
    """color_fix_image that logs a failure and returns the uncorrected image."""
    try:
        return color_fix_image(sr, lq, **kw)
    except Exception as e:
        if log:
            log(f"⚠ Color Fix échoué ({e}) — sortie sans correction")
        return sr


def load_color_fix_ref(path, log=None):
    """Color Fix reference: None when empty, float32 RGB array [0, 1] (16-bit kept)
    otherwise. A path that does not exist is reported instead of silently using the
    LQ image (A35)."""
    if not path:
        return None
    if not os.path.isfile(path):
        if log:
            log(f"Color Fix ref : fichier introuvable ({path}) — LQ utilisé")
        return None
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    try:
        ref = _read_float01(path)
    except Exception as e:
        if log:
            log(f"Color Fix ref : impossible de charger ({e}) — LQ utilisé")
        return None
    _REF_CACHE.clear(); _REF_CACHE[key] = ref   # one reference per batch
    return ref


_REF_CACHE: dict = {}


def _apply_color_fix_file(output_path, input_path, color_fix, color_fix_wavelets,
                          color_fix_radius, color_fix_fast, color_fix_strength,
                          color_fix_planes, color_fix_device, out_format, bit_depth, quality, log,
                          color_fix_ref=""):
    """Applique Color Fix sur le fichier de sortie déjà sauvé par un subprocess
    (inférence venv : le PNG écrit est brut, sans color fix). color_fix_image =
    numpy/PIL → pas besoin de torch en-process. Sert au mode portable ET aux runners."""
    if not (color_fix and color_fix != "none" and color_fix_strength > 0.0
            and IMAGING_AVAILABLE and os.path.isfile(output_path)):
        return
    try:
        _sr = _read_float01(output_path)
        _lq = Image.open(input_path).convert("RGB")
        _pl = color_fix_planes if color_fix_planes is not None else [0, 1, 2]
        log(f"Color Fix ✓ [{color_fix}] strength={color_fix_strength:.2f}")
        _sr = color_fix_image(_sr, _lq, method=color_fix, wavelets=color_fix_wavelets,
                              radius=color_fix_radius, fast=color_fix_fast,
                              strength=color_fix_strength, planes=_pl, device=color_fix_device,
                              ref_pil=load_color_fix_ref(color_fix_ref, log))
        _save_output(_sr, output_path, out_format, bit_depth, quality)
    except Exception as _e:
        log(f"Color Fix : erreur ({_e})")


def _frozen_route_infer(model_path, input_path, output_path, log, progress_callback,
                        stop_event=None, tile_size=0, tile_pad=32, use_amp=False, scale=0):
    """Route inference to the correct engine venv in frozen mode (no in-process torch).

    safetensors: peek keys torch-free → detect arch → route neosr/traiNNer.
    .pth / unknown: try traiNNer, fall back to neosr venv (covers ESC and neosr-only .pth).
    """
    _key = os.path.realpath(model_path)
    _cached = _FROZEN_ROUTE_CACHE.get(_key)

    def _trainner():
        return _spanplus_subprocess_infer(
            model_path, input_path, output_path, log, progress_callback,
            stop_event=stop_event, tile_size=tile_size, tile_pad=tile_pad,
            use_amp=use_amp, scale_hint=scale)

    def _neosr():
        return _neosr_subprocess_infer(
            model_path, input_path, output_path, log, progress_callback,
            stop_event=stop_event, use_amp=use_amp)

    def _neosr_general():
        return _neosr_general_subprocess_infer(
            model_path, input_path, output_path, log, progress_callback,
            stop_event=stop_event, scale_hint=scale)

    # Use cached routing if known (avoids repeating the fallback cascade per frame)
    if _cached == "neosr" and os.path.exists(_NEOSR_VENV_PY):
        return _neosr()
    if _cached == "neosr_general" and os.path.exists(_NEOSR_VENV_PY):
        return _neosr_general()
    if _cached == "trainner":
        return _trainner()

    # Detect arch torch-free for safetensors → route directly
    _arch = None
    if model_path.lower().endswith(".safetensors"):
        try:
            _keys = _peek_safetensors_keys(model_path)
            _arch = detect_arch_from_state({k: None for k in _keys})
        except Exception:
            pass

    if _arch == "esc" and os.path.exists(_NEOSR_VENV_PY):
        _FROZEN_ROUTE_CACHE[_key] = "neosr";  return _neosr()
    if _arch in _NEOSR_GENERAL_ARCHS and os.path.exists(_NEOSR_VENV_PY):
        _FROZEN_ROUTE_CACHE[_key] = "neosr_general";  return _neosr_general()

    # Default: traiNNer venv
    _res = _trainner()
    if _res[0]:
        _FROZEN_ROUTE_CACHE[_key] = "trainner"
        return _res

    # traiNNer failed — for non-safetensors (e.g. ESC .pth) try neosr venv fallback
    _stopped = isinstance(_res[1], str) and ("Arrêt" in _res[1] or "arrêt" in _res[1])
    if (not _stopped and not model_path.lower().endswith(".safetensors")
            and os.path.exists(_NEOSR_VENV_PY)):
        log("Échec venv traiNNer → tentative venv neosr...")
        _r2 = _neosr()
        if _r2[0]:
            _FROZEN_ROUTE_CACHE[_key] = "neosr";  return _r2
        _r3 = _neosr_general()
        if _r3[0]:
            _FROZEN_ROUTE_CACHE[_key] = "neosr_general";  return _r3
    return _res


def ensure_custom_archs(log=None) -> None:
    """Copy bundled custom arch .py into the engine archs/ folders if missing or changed.

    The custom archs are traiNNer-redux archs: copied into traiNNer-redux only. They used
    to be copied into neosr too, where several import traiNNer at module level; NeoSR
    imports every *_arch.py of its archs/ folder, so one of them broke every NeoSR arch
    (training and inference). Copies left there by older versions are disabled
    (renamed *.usrs_disabled, nothing deleted). Idempotent.
    """
    if not os.path.isdir(_CUSTOM_ARCHS_DIR):
        return
    base = _engine_base()
    neosr_archs = os.path.join(base, "neosr", "neosr", "archs")
    if os.path.isdir(neosr_archs):
        for fname in _CUSTOM_ARCH_FILES + _PENDING_ARCH_FILES:
            stray = os.path.join(neosr_archs, fname)
            if fname in _CUSTOM_NEOSR_ARCH_FILES or not os.path.isfile(stray):
                continue
            try:
                os.replace(stray, stray + ".usrs_disabled")
                if log:
                    log(f"[ARCH] {fname} retiré de neosr (arch traiNNer-redux) → {fname}.usrs_disabled")
            except Exception as _e:
                if log:
                    log(f"[ARCH] {fname} : désactivation dans neosr impossible : {_e}")
    targets = [
        os.path.join(base, "traiNNer-redux", "traiNNer", "archs"),
    ]
    for archs_dir in targets:
        if not os.path.isdir(archs_dir):
            continue
        for src_dir, fname in ([(_CUSTOM_ARCHS_DIR, f) for f in _CUSTOM_ARCH_FILES]
                               + [(_PENDING_ARCHS_DIR, f) for f in _PENDING_ARCH_FILES]):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(archs_dir, fname)
            if not os.path.isfile(src):
                continue
            try:
                need_copy = True
                if os.path.isfile(dst):
                    with open(src, "rb") as a, open(dst, "rb") as b:
                        need_copy = a.read() != b.read()
                if need_copy:
                    import shutil
                    shutil.copy2(src, dst)
                    if log:
                        log(f"[ARCH] {fname} injecté dans {os.path.basename(os.path.dirname(archs_dir))}")
            except Exception as _e:
                if log:
                    log(f"[ARCH] Échec injection {fname} : {_e}")

    # NeoSR-only custom archs (aethernet) — neosr archs/ only, not traiNNer-redux.
    if os.path.isdir(_CUSTOM_NEOSR_ARCHS_DIR):
        neosr_archs_dir = os.path.join(base, "neosr", "neosr", "archs")
        if os.path.isdir(neosr_archs_dir):
            for fname in _CUSTOM_NEOSR_ARCH_FILES:
                src = os.path.join(_CUSTOM_NEOSR_ARCHS_DIR, fname)
                dst = os.path.join(neosr_archs_dir, fname)
                if not os.path.isfile(src):
                    continue
                try:
                    need_copy = True
                    if os.path.isfile(dst):
                        with open(src, "rb") as a, open(dst, "rb") as b:
                            need_copy = a.read() != b.read()
                    if need_copy:
                        import shutil
                        shutil.copy2(src, dst)
                        if log:
                            log(f"[ARCH] {fname} injecté dans neosr")
                except Exception as _e:
                    if log:
                        log(f"[ARCH] Échec injection {fname} : {_e}")

    # Small targeted patches to upstream engine files (bugs in files the official
    # engine ships, not whole custom files) — distinct from the whole-file injection
    # above. Self-heals across engine re-clones. See engine_patches.py.
    try:
        from src.core.engine_patches import apply_all_known_engines
        apply_all_known_engines(base, log)
    except Exception as _e:
        if log:
            log(f"[PATCH] Échec application des patchs moteur : {_e}")

# Archs qui appellent net_opt() au niveau module → subprocess neosr_general_runner obligatoire
# _NEOSR_GENERAL_ARCHS now defined earlier (includes aethernet)

# Archs à exécuter via subprocess traiNNer-redux même si importables en-process.
# Raison : le kernel CUDA compilé dans le venv traiNNer est requis.
# - spanplus : DySample update_params() → F.conv2d kernel absent du torch système (Python 3.14 / sm_61)
# - smosr, gfisrv2 : MetaUpsample (famille DySample) → même problème probable
# - spanc  : MetaIGConv coord-based → incertain, subprocess par sécurité
# - spanf  : conv depthwise standard, mais subprocess pour cohérence venv
_TRAINNER_SUBPROCESS_ARCHS = {"spanplus", "smosr", "gfisrv2", "spanc", "spanf",
                              "figsr", "paragonsr", "paragonsr2", "mambair", "mambairv2",
                              # v2.5.8 batch 1 (lmlt/eimn omis : déjà routés vers neosr_general,
                              # structure de poids partagée avec neosr — cf. ARCH_SIGNATURES)
                              "artcnn", "lkfmixer", "rtmosr", "mosrv2", "sebica", "man", "elan",
                              # v2.6.1 batch : 9 archs natives traiNNer-redux supplémentaires
                              # (universal_runner.py + persistent_upscale_worker.py)
                              "gaterv3", "hit_sir", "hit_sng", "cascadedgaze", "escrealm",
                              "emt", "flexnet", "srformer", "srformerv2",
                              # DocRes (docres_base/docres_large) — manual reconstruction only
                              # lives in universal_runner.py/persistent_upscale_worker.py, no
                              # in-process elif branch here (would be dead code, same as the
                              # other entries in this set — see the routing gate above).
                              "docres"}
# AetherNet = NeoSR arch (net_opt() at module level) → route via neosr subprocess.
# Source (aether_core.py / aether_arch.py) is NOT bundled with the neosr repo — it
# lives in src/core/custom_neosr_archs/ and is copied into neosr/neosr/archs/ by
# ensure_custom_archs() before every inference call (same pattern as mambair etc.
# for traiNNer-redux). Without that injection this entry would break — do not
# remove ensure_custom_archs()'s _CUSTOM_NEOSR_ARCH_FILES copy step.
_NEOSR_GENERAL_ARCHS = {"ninasr", "lmlt", "eimn", "drct", "aethernet", "hasn", "msdan", "cfsr",
                         # NeoSR CATANet / KRGN / MAN / MoSRv2 (neosr_general_runner build_extra);
                         # CATANet's in-process path never worked (NeoSR parses argv at import).
                         "catanet", "krgn", "man_neosr", "mosrv2_neosr",
                         # v2.7.0 batch : 8 archs NeoSR supplémentaires (voir ARCH_SIGNATURES
                         # ci-dessus + neosr_general_runner.py::build_model() /
                         # persistent_neosr_worker.py::_build_general_model() pour les branches).
                         # "flexnet" a aussi une entrée traiNNer historique dans
                         # _TRAINNER_SUBPROCESS_ARCHS plus bas — ce set gagne pour les .safetensors
                         # (voir la fonction de routing) et sert de repli pour les .pth.
                         # "srformer" couvre aussi le preset srformer_medium (même classe,
                         # hyperparamètres différents mais déduits des shapes réelles — pas
                         # de clé "srformer_medium" séparée nécessaire dans ARCH_SIGNATURES).
                         "plainusr", "asid", "grformer", "dct", "flexnet", "hit_srf", "hma", "srformer"}

# ─── Model cache ─────────────────────────────────────────────────
# Clé : chemin absolu canonique du modèle.
# Valeur : {"model": obj|None, "scale": int, "arch": str, "device": str,
#            "is_subprocess": bool, "subprocess_type": str}
# Évite de recharger le modèle à chaque image lors d'un batch.
_MODEL_CACHE: dict = {}


def clear_model_cache() -> int:
    """
    Libère tous les modèles en cache et vide la VRAM.
    Retourne le nombre d'entrées supprimées.
    Appeler depuis l'UI après un batch pour récupérer la VRAM.
    """
    global _MODEL_CACHE
    n = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return n


# ─── Persistent batch subprocess session ─────────────────────────────────────
_PERSISTENT_WORKER = os.path.join(os.path.dirname(__file__), "persistent_upscale_worker.py")


class PersistentBatchSession:
    """
    Garde un subprocess venv Python vivant pendant tout le batch.
    Protocole JSON-lines : chaque commande = 1 ligne JSON sur stdin,
    chaque réponse = 1 ligne JSON sur stdout.

    Gain : le modèle est chargé UNE seule fois → ~2-5s économisés par image.
    Sur 30 000 frames : 16-25h de gain vs un subprocess par image.

    Usage :
        with PersistentBatchSession(venv_py, model_path, tile_size, tile_pad, use_amp) as s:
            ok, msg, _ = s.infer(input_path, output_path)
    """

    def __init__(self, venv_py: str, model_path: str,
                 tile_size: int = 256, tile_pad: int = 32, use_amp: bool = False,
                 log: Optional[Callable] = None, scale_hint: int = 0):
        self._venv_py    = venv_py
        self._model_path = model_path
        self._tile_size  = tile_size
        self._tile_pad   = tile_pad
        self._use_amp    = use_amp
        self._log        = log or (lambda m: None)
        self._scale_hint = scale_hint  # user-selected scale (0=auto)
        self._proc       = None
        self.arch: str   = "unknown"
        self.scale: int  = 2
        self._ready      = False

    def _send(self, obj: dict) -> dict:
        """Send one JSON command, receive one JSON response."""
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        return _read_json_reply(self._proc, "PersistentBatch")

    def start(self) -> bool:
        """Launch the persistent worker subprocess and init the model."""
        import subprocess as _sp
        import json

        if not os.path.isfile(self._venv_py):
            self._log(f"[PersistentBatch] venv Python introuvable : {self._venv_py}")
            return False
        if not os.path.isfile(_PERSISTENT_WORKER):
            self._log(f"[PersistentBatch] Worker introuvable : {_PERSISTENT_WORKER}")
            return False

        _env = os.environ.copy()
        _env["PYTHONIOENCODING"] = "utf-8"
        _env["PYTHONUTF8"] = "1"

        engine_dir = os.path.dirname(os.path.dirname(os.path.dirname(self._venv_py)))
        try:
            self._proc = _sp.Popen(
                [self._venv_py, _PERSISTENT_WORKER],
                stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE,
                text=True, encoding="utf-8", errors="replace",
                env=_env, cwd=engine_dir,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
        except Exception as e:
            self._log(f"[PersistentBatch] Échec lancement : {e}")
            return False
        from collections import deque
        self._err_tail = deque(maxlen=40)
        _drain_stderr(self._proc, self._err_tail)

        # Send init command
        try:
            resp = self._send({
                "cmd": "init",
                "model": self._model_path,
                "tile_size": self._tile_size,
                "tile_pad":  self._tile_pad,
                "use_amp":   self._use_amp,
                "scale_hint": self._scale_hint,  # user-selected scale (0=auto)
            })
        except Exception as e:
            _last = [l for l in getattr(self, "_err_tail", []) if l.strip()][-1:]
            self._log(f"[PersistentBatch] Erreur init : {e}" + (f" — {_last[0][:200]}" if _last else ""))
            self.stop()
            return False

        if resp.get("status") == "ready":
            self.arch  = resp.get("arch",  "unknown")
            self.scale = resp.get("scale", 2)
            self._ready = True
            self._log(f"[PersistentBatch] Prêt — arch={self.arch} scale={self.scale}×")
            return True
        else:
            self._log(f"[PersistentBatch] Init refusé : {resp.get('msg', '?')}")
            self.stop()
            return False

    def infer(self, input_path: str, output_path: str,
              prev_input: Optional[str] = None,
              prev_output: Optional[str] = None,
              dandere_threshold: float = 0.02) -> Tuple[bool, str, dict]:
        """
        Run inference on one image. Model stays loaded in VRAM.
        If prev_input + prev_output are provided AND tile_size > 0 was set at init,
        the worker uses dandere2x tile compositing (only changed tiles go to GPU).
        Returns: (ok, msg, extra) where extra may contain dandere stats.
        """
        if not self._ready or self._proc is None:
            return False, "Session non démarrée", {}
        if self._proc.poll() is not None:
            return False, f"Worker terminé prématurément (code {self._proc.returncode})", {}
        payload: dict = {"cmd": "infer", "input": input_path, "output": output_path}
        if getattr(self, "save_opts", None):
            payload["save"] = self.save_opts
        if prev_input and prev_output:
            payload["prev_input"] = prev_input
            payload["prev_output"] = prev_output
            payload["dandere_threshold"] = dandere_threshold
        try:
            resp = self._send(payload)
        except Exception as e:
            return False, f"Erreur communication worker : {e}", {}
        ok  = resp.get("status") == "ok"
        msg = resp.get("msg", "")
        extra = {k: resp[k] for k in ("dandere_changed", "dandere_total", "dandere_pct", "note")
                 if k in resp}
        return ok, msg, extra

    def stop(self) -> None:
        """Graceful shutdown — free VRAM, terminate process."""
        import json
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
        except Exception:
            pass
        finally:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
            self._ready = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    @property
    def is_ready(self) -> bool:
        return self._ready


def _persistent_route(model_path: str) -> str:
    """Which persistent worker runs this model: "neosr" (ESC + the NeoSR-only archs,
    neosr venv), "trainner" (traiNNer-redux subprocess archs), or "" (spandrel archs,
    which the traiNNer worker also loads). Reads only the key names."""
    try:
        if model_path.endswith(".safetensors"):
            # Read only the header (key names + dtypes/shapes) — zero tensor data
            import struct as _struct
            with open(model_path, "rb") as _mf:
                _hlen = _struct.unpack("<Q", _mf.read(8))[0]
                _hmeta = json.loads(_mf.read(_hlen))
            _sd_keys = set(k for k in _hmeta if k != "__metadata__")
        else:
            _ck = torch.load(model_path, map_location="cpu", weights_only=False)
            _sd = _ck if any(k.endswith(".weight") for k in _ck.keys()) else (
                _ck.get("params_ema") or _ck.get("params") or _ck)
            _sd_keys = set(_sd.keys())
        arch = detect_arch_from_state({k: None for k in _sd_keys}) or ""  # type: ignore
    except Exception:
        return ""
    # persistent_neosr_worker.py handles ESC and the NeoSR "general" archs
    # (ninasr/lmlt/eimn/drct/aethernet/hasn/msdan/catanet/krgn…)
    if arch == "esc" or arch in _NEOSR_GENERAL_ARCHS:
        return "neosr"
    # + the archs rebuilt by arch_infer (HEART/NERVE/WireSR/FDAT-RD/DAT-iLN/WSR/DUAL),
    # which also run in the traiNNer subprocess
    if arch in _TRAINNER_SUBPROCESS_ARCHS | {"heart", "nerve", "wiresr", "fdat_rd", "dat_iln",
                                              "wsr", "dual", "spanf3"}:
        return "trainner"
    return ""


class _NeoSRSessionAdapter:
    """PersistentNeoSRSession behind the PersistentBatchSession interface
    (infer returns ok, msg, extra) so callers can use either one."""

    def __init__(self, session: "PersistentNeoSRSession"):
        self._s = session

    def start(self) -> bool:
        return self._s.start()

    def infer(self, input_path: str, output_path: str, *_, **__) -> Tuple[bool, str, dict]:
        ok, msg = self._s.infer(input_path, output_path)
        return ok, msg, {}

    def stop(self) -> None:
        self._s.stop()

    @property
    def is_ready(self) -> bool:
        return bool(getattr(self._s, "_ready", False))

    @property
    def scale(self) -> int:
        return self._s.scale

    @property
    def save_opts(self):
        return getattr(self._s, "save_opts", None)

    @save_opts.setter
    def save_opts(self, v):
        self._s.save_opts = v


def make_persistent_session(model_path: str, tile_size: int = 256, tile_pad: int = 32,
                            use_amp: bool = False, log: Optional[Callable] = None,
                            scale_hint: int = 0):
    """Persistent session for the Quick Upscale tab: the NeoSR worker for NeoSR-only
    archs, the traiNNer worker otherwise. Before, the tab always opened the traiNNer
    worker, so ESC / GRFormer / CATANet… failed to start and ran one subprocess per
    image. Returns an unstarted session, or None when no engine venv is installed."""
    if _persistent_route(model_path) == "neosr" and os.path.isfile(_NEOSR_VENV_PY):
        return _NeoSRSessionAdapter(PersistentNeoSRSession(
            venv_py=_NEOSR_VENV_PY, model_path=model_path, tile_size=tile_size,
            tile_pad=tile_pad, use_amp=use_amp, log=log, scale_hint=scale_hint))
    if not _TRAINNER_VENV_PY:
        return None
    return PersistentBatchSession(venv_py=_TRAINNER_VENV_PY, model_path=model_path,
                                  tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp,
                                  log=log, scale_hint=scale_hint)


# ─── Dandere2x helpers ────────────────────────────────────────────────────────

def _dandere_block_mae(prev_arr: "np.ndarray", curr_arr: "np.ndarray",
                        block_size: int) -> "np.ndarray":
    """
    Vectorized per-block MAE using numpy reshape (no Python loops).
    Returns float32 [bh, bw] — mean absolute error per block.
    ~100x faster than a block-level Python loop.
    """
    h, w = prev_arr.shape[:2]
    ch, cw = curr_arr.shape[:2]
    if (ch, cw) != (h, w):
        # Resolution change between frames (different source clip / scene).
        # No temporal reuse possible → every block counts as changed (max MAE).
        bh = math.ceil(max(h, ch) / block_size)
        bw = math.ceil(max(w, cw) / block_size)
        return np.full((bh, bw), np.inf, dtype=np.float32)
    bh = math.ceil(h / block_size)
    bw = math.ceil(w / block_size)
    ph, pw = bh * block_size, bw * block_size
    # Pad to exact multiple of block_size
    p = np.pad(prev_arr.astype(np.float32), ((0, ph - h), (0, pw - w), (0, 0)), mode='edge')
    c = np.pad(curr_arr.astype(np.float32), ((0, ph - h), (0, pw - w), (0, 0)), mode='edge')
    # Reshape to [bh, block_size, bw, block_size, 3] then transpose → [bh, bw, bs, bs, 3]
    pb = p.reshape(bh, block_size, bw, block_size, 3).transpose(0, 2, 1, 3, 4)
    cb = c.reshape(bh, block_size, bw, block_size, 3).transpose(0, 2, 1, 3, 4)
    return np.abs(cb - pb).mean(axis=(2, 3, 4))   # [bh, bw]


def dandere_compute_diff(
    prev_arr: "np.ndarray",
    curr_arr: "np.ndarray",
    block_size: int = 16,
    threshold: float = 0.02,
    search_window: int = 0,
) -> "tuple":
    """
    Block-level difference detection.

    search_window=0 (default): fully vectorized numpy — fast (microseconds).
    search_window>0: motion-compensated Python loops — slow but handles pans.
    For block mode the fast path is always used (no compositing at SR level).

    Returns:
        changed_mask  : bool [bh, bw]           True = block changed
        motion_vectors: int32 [bh, bw, 2]       (dy, dx) — zeros when no MV
    """
    h, w = prev_arr.shape[:2]
    # Resolution change between consecutive frames → no temporal reuse.
    # Force full re-process: every block changed, no motion vectors.
    if curr_arr.shape[:2] != (h, w):
        ch, cw = curr_arr.shape[:2]
        bh = math.ceil(max(h, ch) / block_size)
        bw = math.ceil(max(w, cw) / block_size)
        return np.ones((bh, bw), dtype=bool), np.zeros((bh, bw, 2), dtype=np.int32)
    bh = math.ceil(h / block_size)
    bw = math.ceil(w / block_size)
    motion_vectors = np.zeros((bh, bw, 2), dtype=np.int32)

    if search_window == 0:
        # Fast vectorized path — no motion vectors
        mae = _dandere_block_mae(prev_arr, curr_arr, block_size)
        return mae > threshold, motion_vectors

    # Slow motion-search path (kept for reference / future use)
    changed_mask = np.ones((bh, bw), dtype=bool)
    for by in range(bh):
        for bx in range(bw):
            y0 = by * block_size;  x0 = bx * block_size
            y1 = min(y0 + block_size, h);  x1 = min(x0 + block_size, w)
            bH = y1 - y0;  bW = x1 - x0
            curr_block = curr_arr[y0:y1, x0:x1]
            best_mae = threshold;  best_dy = 0;  best_dx = 0;  found = False
            for dy in range(-search_window, search_window + 1):
                for dx in range(-search_window, search_window + 1):
                    py0 = y0 + dy;  px0 = x0 + dx
                    py1 = py0 + bH;  px1 = px0 + bW
                    if py0 < 0 or px0 < 0 or py1 > h or px1 > w:
                        continue
                    mae = np.mean(np.abs(curr_block - prev_arr[py0:py1, px0:px1]))
                    if mae < best_mae:
                        best_mae, best_dy, best_dx, found = mae, dy, dx, True
            if found:
                changed_mask[by, bx] = False
                motion_vectors[by, bx] = [best_dy, best_dx]
    return changed_mask, motion_vectors


def dandere_compose(
    prev_sr: "np.ndarray",
    curr_sr: "np.ndarray",
    diff_mask: "np.ndarray",
    motion_vectors: "np.ndarray | None" = None,
    block_size: int = 16,
    scale: int = 4,
    feather: int = 0,
) -> "np.ndarray":
    """
    [LEGACY — not used in block mode with global SR upscalers]

    Compositing at SR level creates visible seams when the SR model has a
    global receptive field (SPANPlus, SPAN, SwinIR…): the same LQ block
    produces slightly different SR output depending on surrounding context,
    so mixing blocks from two SR runs always shows a texture discontinuity.

    Kept for completeness / future tile-based upscaler integration.
    Current block mode skips frames with 0% changed blocks instead.
    """
    bsz    = block_size * scale
    h,  w  = curr_sr.shape[:2]
    hp, wp = prev_sr.shape[:2]
    bh, bw = diff_mask.shape
    result = curr_sr.copy()
    for by in range(bh):
        for bx in range(bw):
            if diff_mask[by, bx]:
                continue
            dst_y0 = by * bsz;  dst_x0 = bx * bsz
            dst_y1 = min(dst_y0 + bsz, h);  dst_x1 = min(dst_x0 + bsz, w)
            dy_sr = int(motion_vectors[by, bx, 0]) * scale if motion_vectors is not None else 0
            dx_sr = int(motion_vectors[by, bx, 1]) * scale if motion_vectors is not None else 0
            src_y0 = dst_y0 + dy_sr;  src_x0 = dst_x0 + dx_sr
            src_y1 = src_y0 + (dst_y1 - dst_y0);  src_x1 = src_x0 + (dst_x1 - dst_x0)
            if src_y0 < 0 or src_x0 < 0 or src_y1 > hp or src_x1 > wp:
                continue
            result[dst_y0:dst_y1, dst_x0:dst_x1] = prev_sr[src_y0:src_y1, src_x0:src_x1]
    return result


def dandere_should_skip(
    prev_arr: "np.ndarray",
    curr_arr: "np.ndarray",
    block_size: int = 32,
    threshold: float = 0.02,
) -> bool:
    """
    Block-level skip decision — more accurate than global MAE.

    Returns True only if EVERY block's MAE is below threshold.
    A character turning their head changes only the face blocks (maybe 5% of
    image) but those blocks have high MAE → max() > threshold → don't skip.
    Global MAE would give ~0.002 for the same case → wrongly skipped.

    block_size=32: coarser blocks → faster; still catches local motion.
    """
    mae = _dandere_block_mae(prev_arr, curr_arr, block_size)
    return bool(mae.max() <= threshold)


def dandere_frame_similarity(
    prev_arr: "np.ndarray",
    curr_arr: "np.ndarray",
    detect_global_motion: bool = False,
    motion_check_px: int = 3,
) -> "tuple":
    """
    [LEGACY] Returns (mean_diff, has_global_motion).
    Prefer dandere_should_skip() for skip-mode decisions.
    """
    if curr_arr.shape != prev_arr.shape:
        # Resolution change → never skip; treat as fully different frame.
        return float("inf"), False
    mean_diff = float(np.abs(curr_arr - prev_arr).mean())
    return mean_diff, False


def _pil_to_float(pil: "Image.Image") -> "np.ndarray":
    return np.array(pil.convert("RGB"), dtype=np.float32) / 255.0


def _float_to_pil(arr: "np.ndarray") -> "Image.Image":
    return Image.fromarray((arr.clip(0, 1) * 255).round().astype(np.uint8))


def _spanplus_subprocess_infer(
    model_path: str,
    input_path: str,
    output_path: str,
    log: Callable,
    progress_callback: Optional[Callable] = None,
    stop_event=None,
    tile_size: int = 0,
    tile_pad: int = 32,
    use_amp: bool = False,
    scale_hint: int = 0,
) -> "Tuple[bool, str]":
    """
    Run inference in the traiNNer-redux venv subprocess.
    Uses universal_runner (spandrel-first) when available,
    falls back to spanplus_runner for legacy compatibility.
    progress_callback(float 0.0-1.0) is called as runner lines arrive.
    """
    import subprocess

    venv_py = _TRAINNER_VENV_PY
    # Prefer universal runner (handles SPAN, ESRGAN, etc. via spandrel)
    runner = _UNIVERSAL_RUNNER if os.path.isfile(_UNIVERSAL_RUNNER) else _SPANPLUS_RUNNER

    if not os.path.isfile(venv_py):
        return False, (
            f"traiNNer-redux venv Python introuvable :\n{venv_py}\n"
            f"Vérifiez que traiNNer-redux est installé dans {os.path.join(_engine_base(), 'traiNNer-redux')}."
        )
    if not os.path.isfile(runner):
        return False, f"Runner d'inférence introuvable : {runner}"

    # Map known runner output prefixes to approximate progress values
    # Covers both universal_runner and spanplus_runner output lines
    _PROGRESS_MAP = [
        ("[Runner] Device",      0.06),
        ("[Runner] Chargement",  0.20),
        ("[Runner] spandrel OK", 0.45),
        ("[Runner] SpanPlus",    0.45),
        ("[Runner] Modele",      0.60),
        ("[Runner] Inference",   0.70),
        ("[Runner] Sauvegarde",  0.92),
    ]

    # Pass tile/amp/scale params as argv so universal_runner can use them
    cmd = [venv_py, runner, model_path, input_path, output_path,
           str(tile_size), str(tile_pad), "1" if use_amp else "0", str(scale_hint)]
    # TRAINNER_ROOT lets universal_runner add the traiNNer-redux source to sys.path
    # so that `from traiNNer.archs.*` imports work without editable install.
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_runner_env(TRAINNER_ROOT=_ep.redux_path()),
            creationflags=0x08000000 if sys.platform == "win32" else 0,  # CREATE_NO_WINDOW
        )
        killed = _run_proc_with_stop(proc, _PROGRESS_MAP, log, progress_callback, stop_event)
        proc.stdout.close()
        if killed:
            return False, "Arrêt demandé par l'utilisateur."
    except Exception as e:
        return False, f"Erreur lancement du subprocess d'inférence (venv traiNNer) : {e}"

    if proc.returncode == 0:
        if progress_callback:
            progress_callback(1.0)
        return True, f"Upscale (subprocess) : {os.path.basename(output_path)}"
    return False, (f"Échec du chargement/inférence du modèle (code {proc.returncode}) — "
                   f"architecture probablement non supportée, voir le log ci-dessus.")


# ─── NeoSR subprocess (ESC arch) ─────────────────────────────────

def _neosr_subprocess_infer(
    model_path: str,
    input_path: str,
    output_path: str,
    log: Callable,
    progress_callback: Optional[Callable] = None,
    stop_event=None,
    use_amp: bool = False,
) -> "Tuple[bool, str]":
    """
    Run ESC inference in the neosr venv subprocess.
    neosr_runner.py patches net_opt before importing esc_arch
    so the module-level net_opt() call does not require sys.argv -opt.
    """
    import subprocess

    venv_py = _NEOSR_VENV_PY
    runner  = _NEOSR_RUNNER

    if not os.path.isfile(venv_py):
        return False, (
            f"neosr venv Python introuvable :\n{venv_py}\n"
            f"Vérifiez que neosr est installé dans {os.path.join(_engine_base(), 'neosr')}."
        )
    if not os.path.isfile(runner):
        return False, f"Runner neosr introuvable : {runner}"

    _PROGRESS_MAP = [
        ("[NeoSR] Device",    0.06),
        ("[NeoSR] ESC params", 0.20),
        ("[NeoSR] ESC chargé", 0.50),
        ("[NeoSR] Inférence",  0.70),
        ("[NeoSR] Sauvegarde", 0.92),
    ]

    # neosr_runner.py accepts exactly 3 positional args (model, input, output) — no amp flag
    cmd = [venv_py, runner, model_path, input_path, output_path]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_runner_env(),
            creationflags=0x08000000 if sys.platform == "win32" else 0,  # CREATE_NO_WINDOW
        )
        killed = _run_proc_with_stop(proc, _PROGRESS_MAP, log, progress_callback, stop_event)
        proc.stdout.close()
        if killed:
            return False, "Arrêt demandé par l'utilisateur."
    except Exception as e:
        return False, f"Erreur lancement subprocess neosr : {e}"

    if proc.returncode == 0:
        if progress_callback:
            progress_callback(1.0)
        return True, f"ESC (neosr subprocess) : {os.path.basename(output_path)}"
    return False, f"Subprocess neosr ESC a échoué (code {proc.returncode})"


# ─── NeoSR general subprocess (ninasr, lmlt, eimn, drct, …) ─────

def _neosr_general_subprocess_infer(
    model_path: str,
    input_path: str,
    output_path: str,
    log: Callable,
    progress_callback: Optional[Callable] = None,
    stop_event=None,
    scale_hint: int = 0,
) -> "Tuple[bool, str]":
    """
    Run inference via neosr_general_runner.py in the neosr venv subprocess.
    Handles net_opt() patching and arch detection internally.
    Used for archs in _NEOSR_GENERAL_ARCHS (ninasr, lmlt, eimn, drct, aethernet,
    hasn, msdan). scale_hint: user-selected scale — REQUIRED for msdan, whose
    checkpoint weights are structurally identical regardless of the scale it
    was trained at (upscale_factor is a plain Python int, never a saved
    buffer/shape) — see Changelog Dev session 135 for the full diagnosis.
    """
    import subprocess

    venv_py = _NEOSR_VENV_PY
    runner  = _NEOSR_GENERAL_RUNNER

    if not os.path.isfile(venv_py):
        return False, (
            f"neosr venv Python introuvable :\n{venv_py}\n"
            f"Vérifiez que neosr est installé dans {os.path.join(_engine_base(), 'neosr')}."
        )
    if not os.path.isfile(runner):
        return False, f"neosr_general_runner introuvable : {runner}"

    _PROGRESS_MAP = [
        ("[NeoSR] Device",     0.06),
        ("[NeoSR] Arch",       0.20),
        ("[NeoSR] Modele",     0.45),
        ("[NeoSR] Inference",  0.70),
        ("[NeoSR] Sauvegarde", 0.92),
    ]

    cmd = [venv_py, runner, model_path, input_path, output_path, str(max(0, scale_hint))]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_runner_env(),
            creationflags=0x08000000 if sys.platform == "win32" else 0,  # CREATE_NO_WINDOW
        )
        killed = _run_proc_with_stop(proc, _PROGRESS_MAP, log, progress_callback, stop_event)
        proc.stdout.close()
        if killed:
            return False, "Arrêt demandé par l'utilisateur."
    except Exception as e:
        return False, f"Erreur lancement subprocess neosr general : {e}"

    if proc.returncode == 0:
        if progress_callback:
            progress_callback(1.0)
        return True, f"neosr general ({os.path.basename(output_path)})"
    return False, f"Subprocess neosr general a echoue (code {proc.returncode})"


# ─── Main upscale function ───────────────────────────────────────

def upscale_image(
    model_path: str,
    input_path: str,
    output_path: str,
    scale: int = 0,
    tile_size: int = 256,
    tile_pad: int = 32,
    use_amp: bool = True,
    callback: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
    out_format: str = "PNG",
    bit_depth: int = 8,
    quality: int = 95,
    stop_event=None,
    color_fix: str = "none",
    color_fix_wavelets: int = 4,
    color_fix_radius: int = 32,
    color_fix_fast: bool = False,
    color_fix_strength: float = 1.0,
    color_fix_planes: Optional[list] = None,
    color_fix_device: str = "auto",
    color_fix_ref: str = "",
) -> Tuple[bool, str]:
    """
    Upscale a single image using a trained SR model.
    
    Args:
        model_path: Path to .pth or .safetensors model
        input_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscale factor (0 = auto-detect)
        tile_size: Tile size for inference (0 = no tiling)
        tile_pad: Tile overlap padding
        use_amp: Use mixed precision (FP16)
        callback: Optional progress callback(message: str)
        
    Returns:
        (success: bool, message: str)
    """
    if not os.path.isfile(model_path):
        return False, f"Modèle introuvable : {model_path}"
    if not os.path.isfile(input_path):
        return False, f"Image introuvable : {input_path}"
    global _SAVE_OPTS, _INFER_OPTS
    _SAVE_OPTS = {"format": out_format, "bit_depth": bit_depth, "quality": quality}
    _INFER_OPTS = {"tile": int(tile_size or 0), "tile_pad": int(tile_pad), "amp": bool(use_amp)}

    def log(msg):
        if callback:
            callback(msg)
        try:
            print(f"[Upscale] {msg}")
        except OSError:
            pass

    # Warn if user picked a discriminator (net_d) — cannot be used for upscaling
    _base = os.path.basename(model_path).lower()
    if _base.startswith("net_d") or _base.startswith("net_d_"):
        return False, ("Ce fichier est un discriminateur (net_d) — il ne peut pas servir à l'upscale.\n"
                       "Utilisez le générateur : net_g_*.safetensors ou net_g_ema_*.safetensors.")

    # Ensure custom archs (gfisrv2, smosr, spanpp, figsr) are present in engine before inference
    ensure_custom_archs(log)

    # ── Video/temporal archs (tscunet/temporalspan/temporalspanv2/tfdat) : une
    # image seule n'a pas de sens (le réseau attend un clip de plusieurs frames
    # consécutives). Message clair, pas de crash — utilisez le mode dossier/batch.
    _peeked_arch = _peek_arch_from_model_path(model_path)
    if _peeked_arch in _SEQUENCE_ARCHS:
        return False, (
            f"{_peeked_arch} est une architecture vidéo/temporelle — elle attend un clip de "
            f"plusieurs frames consécutives en entrée, pas une image seule.\n"
            f"Utilisez le mode Dossier/Batch (une séquence d'images numérotées dans un dossier)."
        )

    # ── ONNX short-circuit ──
    if model_path.lower().endswith(".onnx"):
        # Frozen exe has no onnxruntime → run in engine venv subprocess
        if getattr(sys, 'frozen', False):
            ensure_custom_archs(log)
            return _onnx_subprocess_infer(model_path, input_path, output_path, log,
                                          progress_callback, stop_event=stop_event,
                                          out_format=out_format, bit_depth=bit_depth, quality=quality)
        return _onnx_upscale(model_path, input_path, output_path, log, progress_callback,
                             out_format=out_format, bit_depth=bit_depth, quality=quality)

    # Lazy torch retry — module-level TORCH_AVAILABLE may be False in PyInstaller portable
    # if quick_upscale is imported before PyInstaller finishes bootstrapping torch in thread.
    global TORCH_AVAILABLE
    if not TORCH_AVAILABLE:
        try:
            import torch as _t; import torch.nn.functional as _F  # noqa: F401,E401
            TORCH_AVAILABLE = True
        except ImportError:
            pass
    if not TORCH_AVAILABLE:
        # In frozen portable build: torch not bundled in exe.
        # Route to the correct venv subprocess (no in-process torch needed).
        if getattr(sys, 'frozen', False):
            log("Mode portable : inference via subprocess venv (torch non dispo en-process)")
            _res = _frozen_route_infer(
                model_path, input_path, output_path, log, progress_callback,
                stop_event=stop_event, tile_size=tile_size, tile_pad=tile_pad,
                use_amp=use_amp, scale=scale,
            )
            # Le subprocess sauve le PNG brut → Color Fix appliqué ici (tous modèles, portable).
            if _res and _res[0]:
                _apply_color_fix_file(output_path, input_path, color_fix, color_fix_wavelets,
                                      color_fix_radius, color_fix_fast, color_fix_strength,
                                      color_fix_planes, color_fix_device, out_format, bit_depth, quality, log,
                                      color_fix_ref=color_fix_ref)
            return _res
        return False, "PyTorch non installé"
    if not IMAGING_AVAILABLE:
        return False, "Pillow/NumPy non installé"

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Device : {device}")

        # ── Color Fix post-subprocess helper ──────────────────────────────────────
        def _post_colorfix(sub_result):
            """Applique Color Fix sur le fichier sauvegardé par un subprocess."""
            _ok, _info = sub_result
            if (_ok and color_fix != "none" and color_fix_strength > 0.0
                    and IMAGING_AVAILABLE and os.path.isfile(output_path)):
                try:
                    _sr_arr = _read_float01(output_path)
                    _lq_pil = Image.open(input_path).convert("RGB")
                    _planes = color_fix_planes if color_fix_planes is not None else [0, 1, 2]
                    log(f"Color Fix ✓ [{color_fix}] strength={color_fix_strength:.2f}")
                    _sr_arr = color_fix_image(_sr_arr, _lq_pil,
                                              method=color_fix,
                                              wavelets=color_fix_wavelets,
                                              radius=color_fix_radius,
                                              fast=color_fix_fast,
                                              strength=color_fix_strength,
                                              planes=_planes,
                                              device=color_fix_device,
                                              ref_pil=load_color_fix_ref(color_fix_ref, log))
                    _save_output(_sr_arr, output_path, out_format, bit_depth, quality)
                except Exception as _cf_e:
                    log(f"Color Fix subprocess : erreur ({_cf_e})")
            return _ok, _info

        # ── Cache check ──
        _cache_key   = os.path.realpath(model_path)
        _cached      = _MODEL_CACHE.get(_cache_key)
        _cache_device = str(device)

        if _cached is not None and _cached.get("device") == _cache_device:
            # --- Cache hit ---
            if _cached.get("is_subprocess"):
                detected_arch = _cached["arch"]
                _user_scale = scale
                if scale == 0:
                    scale = _cached["scale"]
                stype = _cached.get("subprocess_type", "trainner")
                log(f"[Cache] {detected_arch} (subprocess {stype}, scale {scale}x)")
                if stype == "esc":
                    return _post_colorfix(_neosr_subprocess_infer(
                        model_path, input_path, output_path, log,
                        progress_callback, stop_event=stop_event, use_amp=use_amp))
                elif stype == "neosr_general":
                    return _post_colorfix(_neosr_general_subprocess_infer(
                        model_path, input_path, output_path, log,
                        progress_callback, stop_event=stop_event, scale_hint=(scale if _user_scale else _cached.get("hint", scale))))
                else:
                    return _post_colorfix(_spanplus_subprocess_infer(
                        model_path, input_path, output_path, log,
                        progress_callback, stop_event=stop_event,
                        tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp, scale_hint=(scale if _user_scale else _cached.get("hint", scale))))
            model         = _cached["model"]
            detected_arch = _cached["arch"]
            if scale == 0:
                scale = _cached["scale"]
            log(f"[Cache] Modèle réutilisé : {detected_arch}, scale {scale}x")
        else:
            # --- Cache miss : chargement complet ---
            log(f"Chargement du modèle : {os.path.basename(model_path)}")

            if model_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path, device="cpu")
                except ImportError:
                    return False, "Module safetensors requis : pip install safetensors"
            else:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                state_dict = extract_weights(checkpoint)

            # ── Detect architecture ──
            detected_arch = detect_arch_from_state(state_dict)
            # Hint for the subprocess runners: the user's scale, or the detected one, or 0
            # (unknown) so each runner reads the scale from the arch's own layers. The 4x
            # default used to be sent as the hint and overrode the runners' detection.
            _known_scale = scale or detect_scale_from_state(state_dict, default=0)
            if scale == 0:
                scale = _known_scale or 4
            _hint = _known_scale
            if not _hint and detected_arch == "msdan":
                _hint = scale  # MSDAN's scale cannot be read from its weights
            log(f"Architecture détectée : {detected_arch or 'inconnue'}, scale : "
                f"{scale}x{'' if _known_scale else ' (non détecté)'}")

            def _store_subprocess(stype: str) -> None:
                _MODEL_CACHE[_cache_key] = {
                    "model": None, "scale": scale, "hint": _hint,
                    "arch": detected_arch or stype,
                    "device": _cache_device,
                    "is_subprocess": True,
                    "subprocess_type": stype,
                }

            # ── Diffusion SR (OSEDiff, VOSR) : nécessite runtime diffusion ──────────
            # Ces modèles utilisent un UNet SD + VAE — inférence itérative non supportée
            # par le pipeline tile_inference. Tentative via spandrel subprocess ;
            # si spandrel ne les supporte pas, l'erreur sera claire.
            if detected_arch in _DIFFUSION_ARCHS:
                log(f"⚠ {detected_arch.upper()} détecté — modèle diffusion SR. "
                    f"Tentative via spandrel subprocess (traiNNer venv)...")
                _store_subprocess("trainner")
                return _post_colorfix(_spanplus_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event,
                    tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp, scale_hint=_hint
                ))

            # ── Video SR (TVT, DAM-VSR) : nécessite entrée multi-frames ──────────
            # Inférence image par image non supportée nativement — spandrel subprocess.
            # Un mode vidéo dédié sera ajouté en v2.6.0.
            if detected_arch in _VIDEO_ARCHS:
                log(f"⚠ {detected_arch.upper()} détecté — modèle Video SR (multi-frames). "
                    f"Inférence frame unique via spandrel subprocess...")
                _store_subprocess("trainner")
                return _post_colorfix(_spanplus_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event,
                    tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp, scale_hint=_hint
                ))

            # HEART / NERVE (traiNNer-redux master) : construits par arch_infer.py dans
            # le runner traiNNer — même chemin qu'en mode portable, une seule implémentation.
            if detected_arch in ("heart", "nerve", "wiresr", "fdat_rd", "dat_iln", "wsr", "dual",
                                 "spanf3"):
                log(f"{detected_arch.upper()} détecté → subprocess traiNNer (arch_infer)...")
                _store_subprocess("trainner")
                return _post_colorfix(_spanplus_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event,
                    tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp, scale_hint=_hint
                ))

            # ESC : net_opt() au niveau module, subprocess neosr venv obligatoire
            if detected_arch == "esc":
                log("ESC détecté → subprocess neosr venv...")
                _store_subprocess("esc")
                return _post_colorfix(_neosr_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event, use_amp=use_amp
                ))

            # ninasr/lmlt/eimn/drct : net_opt() au niveau module → neosr_general_runner
            if detected_arch in _NEOSR_GENERAL_ARCHS:
                log(f"{detected_arch} → subprocess neosr général...")
                _store_subprocess("neosr_general")
                return _post_colorfix(_neosr_general_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event, scale_hint=_hint
                ))

            if detected_arch in _TRAINNER_SUBPROCESS_ARCHS:
                log(f"{detected_arch} → subprocess traiNNer-redux venv (kernel CUDA requis)...")
                _store_subprocess("trainner")
                return _post_colorfix(_spanplus_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event,
                    tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp, scale_hint=_hint
                ))

            # ── Try to instantiate model in-process ──
            model = None

            # Try neosr imports
            if detected_arch and model is None:
                try:
                    neosr_path = os.path.join(_engine_base(), "neosr")
                    if os.path.isdir(neosr_path) and neosr_path not in sys.path:
                        sys.path.insert(0, neosr_path)
                    if detected_arch == "span":
                        from neosr.archs.span_arch import span
                        model = span(scale=scale)
                    elif detected_arch == "omnisr":
                        from neosr.archs.omnisr_arch import omnisr
                        model = omnisr(scale=scale)
                    elif detected_arch == "realplksr":
                        from neosr.archs.realplksr_arch import realplksr
                        model = realplksr(scale=scale)
                    elif detected_arch == "compact":
                        from neosr.archs.srvgg_arch import srvgg
                        model = srvgg(scale=scale)
                    elif detected_arch == "esrgan":
                        from neosr.archs.rrdbnet_arch import rrdbnet
                        model = rrdbnet(scale=scale)
                    elif detected_arch == "catanet":
                        # net_opt() doit être patché avant l'import (lecture module-level)
                        import neosr.archs.arch_util as _cat_au
                        _saved_net_opt = _cat_au.net_opt
                        _cat_au.net_opt = lambda: (scale, True)
                        try:
                            from neosr.archs.catanet_arch import catanet as _CATANet
                            w = state_dict.get("first_conv.weight")
                            dim = int(w.shape[0]) if w is not None else 40
                            model = _CATANet(dim=dim, upscale=scale)
                            log(f"CATANet instancié (dim={dim}, scale={scale}x)")
                        finally:
                            _cat_au.net_opt = _saved_net_opt
                except Exception:
                    # NeoSR can raise ValueError (parse_options), not just ImportError
                    pass

            # Try traiNNer-redux imports
            if detected_arch and model is None:
                try:
                    redux_path = os.path.join(_engine_base(), "traiNNer-redux")
                    if os.path.isdir(redux_path) and redux_path not in sys.path:
                        sys.path.insert(0, redux_path)
                    if detected_arch == "spanplus":
                        from traiNNer.archs.spanplus_arch import SpanPlus
                        fc = state_dict.get("feats.0.eval_conv.weight",
                                            state_dict.get("feats.0.sk.weight", None))
                        fc = fc.shape[0] if fc is not None else 48
                        up = "dys" if "upsampler.offset.weight" in state_dict else "conv"
                        model = SpanPlus(feature_channels=fc, upscale=scale, upsampler=up)
                        log(f"SPANPlus instancié (fc={fc}, scale={scale}x, upsampler={up})")
                    elif detected_arch == "span":
                        from traiNNer.archs.span_arch import SPAN
                        model = SPAN(scale=scale)
                    elif detected_arch == "omnisr":
                        from traiNNer.archs.omnisr_arch import OmniSR
                        model = OmniSR(upscale=scale)
                    elif detected_arch == "compact":
                        from traiNNer.archs.srvgg_arch import SRVGGNetCompact
                        # Auto-detect num_feat + num_conv from checkpoint
                        # body.0.weight: [num_feat, in_ch, 3, 3]  → out_ch = num_feat
                        # body weight keys total = num_conv + 1 (input + intermediates + output)
                        _cw0 = state_dict.get("body.0.weight")
                        _c_nf = int(_cw0.shape[0]) if _cw0 is not None else 64
                        # Only count real conv kernels (4D weights) — body also holds PReLU
                        # activations, which get their own "body.N.weight" key (1D, one value
                        # per channel) that was previously counted as if it were a conv layer,
                        # roughly doubling num_conv and corrupting every reload of a trained
                        # compact/ultracompact/superultracompact checkpoint (2026-09-17 fix,
                        # found by actually loading a freshly-trained superultracompact model).
                        _c_bw = [k for k in state_dict
                                 if k.startswith("body.") and k.endswith(".weight") and state_dict[k].dim() == 4]
                        _c_nc = max(1, len(_c_bw) - 2) if _c_bw else 16
                        model = SRVGGNetCompact(num_feat=_c_nf, num_conv=_c_nc, upscale=scale)
                        log(f"SRVGGNetCompact instancié (num_feat={_c_nf}, num_conv={_c_nc}, scale={scale}x)")
                    elif detected_arch == "smosr":
                        from traiNNer.archs.smosr_arch import SMoSR
                        w = state_dict.get("blocks_1.0.body.0.W")
                        dim = int(w.shape[0]) if w is not None else 48
                        n_mb = len([k for k in state_dict if k.startswith("blocks_2.") and k.endswith(".body.0.W")])
                        n_mb = n_mb if n_mb > 0 else 3
                        meta = state_dict.get("upsampler.MetaUpsample")
                        if meta is not None and scale == 0:
                            try:
                                scale = max(1, int(meta[2].item()))
                            except Exception:
                                scale = 4
                        model = SMoSR(scale=max(1, scale), dim=dim, n_mb=n_mb)
                        log(f"SMoSR instancié (dim={dim}, n_mb={n_mb}, scale={scale}x)")
                    elif detected_arch == "spanf":
                        from traiNNer.archs.spanf_arch import spanf
                        w = state_dict.get("block_1.conv1.eval_conv.weight")
                        fc = int(w.shape[0]) if w is not None else 32
                        model = spanf(feature_channels=fc, scale=max(1, scale))
                        log(f"SpanF instancié (fc={fc}, scale={scale}x)")
                    elif detected_arch in ("spanc", "spanpp"):
                        from traiNNer.archs.spanpp_arch import SpanC
                        meta = state_dict.get("MetaIGConv")
                        if meta is not None:
                            scale_list = tuple(int(v.item()) for v in meta)
                        else:
                            scale_list = (1, 2) if detected_arch == "spanpp" else (2, 4)
                        # spanc: feature_channels depuis conv0.eval_conv
                        # spanpp: feature_channels depuis block_1.c1_r.conv3.eval_conv (pas de conv0)
                        if detected_arch == "spanpp":
                            w = state_dict.get("block_1.c1_r.conv3.eval_conv.weight")
                        else:
                            w = state_dict.get("conv0.eval_conv.weight")
                        fc = int(w.shape[0]) if w is not None else 48
                        # Détecter latent_layers depuis query_kernel (défaut=4 mais peut varier)
                        # Indices pairs = Conv2d, dernier = sortie(3ch) → latent_layers = nb_conv - 1
                        _qk_keys = [k for k in state_dict if k.startswith("upsampler.query_kernel.") and k.endswith(".weight")]
                        _latent_layers = max(4, len(_qk_keys) - 1) if _qk_keys else 4
                        # implicit_dim depuis la forme de query_kernel.0
                        _qk0 = state_dict.get("upsampler.query_kernel.0.weight")
                        _implicit_dim = int(_qk0.shape[0]) if _qk0 is not None else 256
                        # ig_kernel_size depuis freq: shape[0] = fc * k^2
                        _freq = state_dict.get("upsampler.freq")
                        if _freq is not None:
                            import math as _math
                            _k2 = _freq.shape[0] / fc
                            _ig_k = max(1, int(round(_math.sqrt(_k2))))
                        else:
                            _ig_k = 3
                        _eval_scale = max(1, scale) if scale in scale_list else scale_list[0]
                        model = SpanC(feature_channels=fc, scale_list=scale_list,
                                      eval_base_scale=_eval_scale,
                                      ig_kernel_size=_ig_k,
                                      implicit_dim=_implicit_dim,
                                      latent_layers=_latent_layers)
                        log(f"{'SpanPP' if detected_arch == 'spanpp' else 'SpanC'} instancié (fc={fc}, scales={scale_list}, latent={_latent_layers}, idim={_implicit_dim})")
                    elif detected_arch == "gfisrv2":
                        from traiNNer.archs.gfisrv2_arch import GFISRV2
                        w = state_dict.get("in_to_dim.weight")
                        dim = int(w.shape[0]) if w is not None else 48
                        n_blocks = len([k for k in state_dict if k.startswith("gfisr_body.") and k.endswith(".fc1.weight")])
                        n_blocks = n_blocks if n_blocks > 0 else 24
                        meta = state_dict.get("upscale.MetaUpsample")
                        if meta is not None and scale == 0:
                            try:
                                scale = max(1, int(meta[2].item()))
                            except Exception:
                                scale = 4
                        model = GFISRV2(scale=max(1, scale), dim=dim, n_blocks=n_blocks)
                        log(f"GFISRv2 instancié (dim={dim}, n_blocks={n_blocks}, scale={scale}x)")
                except ImportError:
                    pass

            if model is None:
                # Fallback 1: TorchScript
                try:
                    model = torch.jit.load(model_path, map_location=device)
                    log("Modèle chargé comme TorchScript")
                except Exception:
                    pass

            if model is None:
                # Fallback 2: universal subprocess (spandrel in traiNNer venv)
                _store_subprocess("trainner")
                log(f"Tentative via subprocess traiNNer-redux venv ({detected_arch or 'arch inconnue'})...")
                return _post_colorfix(_spanplus_subprocess_infer(
                    model_path, input_path, output_path, log, progress_callback,
                    stop_event=stop_event,
                    tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp, scale_hint=_hint
                ))

            if not isinstance(model, torch.jit.ScriptModule):
                model.load_state_dict(state_dict, strict=False)

            try:
                model = model.to(device).eval()
            except Exception as _cuda_err:
                # cudaErrorNoKernelImageForDevice ou autre erreur CUDA (Pascal incompatible)
                # → fallback subprocess traiNNer venv (compilé pour sm_61)
                _err_str = str(_cuda_err)
                if "no kernel image" in _err_str or "cudaErrorNoKernelImageForDevice" in _err_str or "AcceleratorError" in _err_str:
                    log(f"GPU incompatible pour {detected_arch or 'arch'} (Pascal/Maxwell) → subprocess traiNNer")
                    _store_subprocess("trainner")
                    return _post_colorfix(_spanplus_subprocess_infer(
                        model_path, input_path, output_path, log, progress_callback,
                        stop_event=stop_event,
                        tile_size=tile_size, tile_pad=tile_pad, use_amp=use_amp, scale_hint=_hint
                    ))
                raise  # autre erreur CUDA → propage
            log("Modèle chargé et prêt")

            # Mise en cache pour les appels suivants
            _MODEL_CACHE[_cache_key] = {
                "model": model, "scale": scale,
                "arch": detected_arch or "unknown",
                "device": _cache_device,
                "is_subprocess": False,
            }

        # ── Load image ──
        log(f"Lecture : {os.path.basename(input_path)}")
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

        # ── Inference ──
        log(f"Inférence ({img.width}x{img.height} → {img.width*scale}x{img.height*scale})")
        
        amp_ctx = torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda"))
        
        # Check stop before starting heavy inference
        if stop_event and stop_event.is_set():
            return False, "Arrêt demandé par l'utilisateur."

        with torch.no_grad(), amp_ctx:
            try:
                if tile_size > 0 and (img.width > tile_size or img.height > tile_size):
                    log(f"Mode tiling : {tile_size}px, padding {tile_pad}px")
                    output = tile_inference(model, img_tensor, tile_size, tile_pad, scale,
                                            stop_event=stop_event)
                else:
                    output = model(img_tensor)
            except Exception as _cuda_e:
                _esc = str(_cuda_e)
                if "no kernel image" in _esc or "cudaErrorNoKernelImageForDevice" in _esc:
                    log("⚠ Kernel CUDA incompatible → fallback CPU (plus lent)")
                    _model_cpu = model.cpu()
                    _tensor_cpu = img_tensor.cpu()
                    with torch.amp.autocast("cpu", enabled=False):
                        if tile_size > 0 and (img.width > tile_size or img.height > tile_size):
                            output = tile_inference(_model_cpu, _tensor_cpu, tile_size, tile_pad,
                                                    scale, stop_event=stop_event)
                        else:
                            output = _model_cpu(_tensor_cpu)
                    # output reste sur CPU — .cpu() ligne suivante est no-op
                    if device.type == "cuda":
                        model.to(device)  # remet le modèle sur GPU pour les appels suivants
                else:
                    raise

        # ── Save output ──
        output = output.squeeze(0).clamp(0, 1).cpu().float().numpy()
        output = np.transpose(output, (1, 2, 0))   # [H, W, 3] float32 [0,1]

        # ── Color Fix (post-process) ──
        if color_fix != "none" and color_fix_strength > 0.0 and IMAGING_AVAILABLE:
            _planes = color_fix_planes if color_fix_planes is not None else [0, 1, 2]
            _cf_dev = _cf_resolve_device(color_fix_device)

            # ── Chargement image de référence (optionnel) ──
            _ref_pil = load_color_fix_ref(color_fix_ref, log)
            if _ref_pil is not None:
                log(f"Color Fix ref : {os.path.basename(color_fix_ref)}")

            _ref_label = f" ref={os.path.basename(color_fix_ref)}" if _ref_pil is not None else ""
            if color_fix == "wavelet":
                log(f"Color Fix [wavelet] wavelets={color_fix_wavelets} "
                    f"strength={color_fix_strength:.2f} planes={_planes} "
                    f"device={_cf_dev}{_ref_label}")
            elif color_fix == "average":
                log(f"Color Fix [average] radius={color_fix_radius} fast={color_fix_fast} "
                    f"strength={color_fix_strength:.2f} planes={_planes} "
                    f"device={_cf_dev}{_ref_label}")
            else:
                log(f"Color Fix [{color_fix}] strength={color_fix_strength:.2f} "
                    f"planes={_planes}{_ref_label}")

            if _cf_dev == "trt":
                log("Color Fix TRT : premier appel → autotuning Triton (~3 s), "
                    "résultat mis en cache pour les prochaines images.")

            output = _cf_try(output, img, log,
                                     method=color_fix,
                                     wavelets=color_fix_wavelets,
                                     radius=color_fix_radius,
                                     fast=color_fix_fast,
                                     strength=color_fix_strength,
                                     planes=_planes,
                                     device=color_fix_device,
                                     ref_pil=_ref_pil)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        _save_output(output, output_path, out_format, bit_depth, quality)

        log(f"✅ Sauvegardé : {output_path}")
        return True, f"Upscale réussi : {os.path.basename(output_path)}"

    except RuntimeError as e:
        if "stopped" in str(e):
            return False, "Arrêt demandé par l'utilisateur."
        return False, f"Erreur runtime : {e}\n{traceback.format_exc()}"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if tile_size > 0:
            return False, (
                f"VRAM insuffisante même avec tiling ({tile_size}px).\n"
                f"Réduisez tile_size ou utilisez le CPU."
            )
        return False, (
            "VRAM insuffisante. Activez le tiling (tile_size=256) ou réduisez la taille."
        )
    except Exception as e:
        return False, f"Erreur : {e}\n{traceback.format_exc()}"


# ─── Video/temporal archs (tscunet/temporalspan/temporalspanv2/tfdat) ───────
# These 4 archs take (B,T,C,H,W) and output a single (B,C,H,W) frame — the
# center of the input clip. Reconstruction of the network's hyperparameters
# from the checkpoint's weight shapes follows the exact same style as the
# other custom archs above (spanplus/smosr/spanc/gfisrv2 in-process blocks).

# ── True 16-bit-capable I/O for the video/sequence pipeline only ────────────
# (_sequence_video_upscale_folder below). NOT used by the single-image /
# classic-folder upscale paths — those keep their existing PIL-only behavior
# untouched, on purpose (out of scope for this pass).
#
# Why pyvips instead of Pillow for the loader: Pillow has no first-class 3x16
# bit-per-channel RGB mode, so `Image.open(...).convert("RGB")` silently
# truncates a genuine 16-bit source to 8-bit on load (verified). pyvips reads
# the real on-disk sample format (uchar/ushort/etc.) and `.numpy()` returns it
# with the matching numpy dtype, no truncation — this is the exact same
# read pattern traiNNer-redux's PairedVideoDataset uses at training time
# (vipsimfrompath()) for the same reason. Verified byte-identical to Pillow
# for actual 8-bit sources (uint8 arrays compare equal), so the default path
# is unaffected.
#
# Why pyvips instead of Pillow for the *writer* when 16-bit output is asked:
# verified directly — Pillow's `Image.frombuffer("RGB", size, data, "raw",
# "RGB;16B", 0, 1)` trick (used by the generic `_save_output()` above for its
# bit_depth==16 branch) does NOT produce a genuine 16-bit-per-channel PNG; the
# "RGB;16B" raw *decoder* downsamples 16-bit input into an 8-bit "RGB" image
# while decoding — reopening the file this writes shows it landed on disk as
# plain 8-bit (confirmed: pyvips reports format "uchar", not "ushort", after
# round-tripping through that codepath). pyvips's `Image.new_from_memory(...).
# write_to_file(...)` with an actual uint16 buffer does write a real 16-bit
# PNG (confirmed: reloading it back reports format "ushort" and every sample
# matches the source array exactly).

_FMT_MAXVAL = {
    "uchar": 255.0, "char": 127.0,
    "ushort": 65535.0, "short": 32767.0,
    "uint": 4294967295.0, "int": 2147483647.0,
    "float": 1.0, "double": 1.0,
}


def _load_sequence_frame(path: str) -> "np.ndarray":
    """Load one frame as float32 [H, W, 3] in [0, 1], normalized by the REAL
    max value of its on-disk sample format (255 for 8-bit, 65535 for 16-bit,
    etc.) — not a hardcoded /255. Falls back to the old Pillow (8-bit-only)
    loader if pyvips isn't installed, so the function degrades gracefully
    rather than hard-failing a batch that used to work."""
    if not PYVIPS_AVAILABLE:
        img = Image.open(path).convert("RGB")
        return np.asarray(img).astype(np.float32) / 255.0
    v = pyvips.Image.new_from_file(path, access="sequential")
    if v.bands == 1:
        v = v.bandjoin([v, v])  # gray -> 3 bands, same as PIL's .convert("RGB")
    elif v.bands >= 4:
        v = v[:3]  # drop alpha (and any extra bands) — no compositing, matches
        # PIL's .convert("RGB") which also just discards alpha unmixed.
    elif v.bands == 2:
        v = v[:1].bandjoin([v[:1], v[:1]])
    arr = v.numpy()
    maxval = _FMT_MAXVAL.get(v.format, 255.0)
    return arr.astype(np.float32) / maxval


def _save_sequence_frame(float_arr: "np.ndarray", out_path: str, bit_depth: int = 8) -> None:
    """Save one upscaled frame. bit_depth=8 (default) reproduces the exact
    previous behavior byte-for-byte (Pillow uint8 PNG). bit_depth=16 writes a
    genuine 16-bit-per-channel PNG via pyvips (see note above on why Pillow
    can't do this for RGB)."""
    if bit_depth == 16 and PYVIPS_AVAILABLE:
        arr16 = (np.clip(float_arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        h, w, c = arr16.shape
        arr16 = np.ascontiguousarray(arr16)
        vimg = pyvips.Image.new_from_memory(arr16.tobytes(), w, h, c, "ushort")
        vimg.write_to_file(out_path)
        return
    arr8 = (np.clip(float_arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8).save(out_path)

def _build_sequence_model(detected_arch: str, state_dict: dict, scale: int, log):
    """Instantiate one of the 4 video/temporal archs from its state_dict.
    Returns (model, clip_size). Raises on failure — never masks a shape
    mismatch on our side. One exception: TFDAT's own load_state_dict()
    (traiNNer/archs/tfdat_arch.py) hardcodes strict=False internally
    regardless of what's passed — a pre-existing quirk of that arch in the
    engine, not something introduced here; we independently verify the key
    sets match afterwards and raise ourselves if they don't, so the strict
    guarantee still holds end-to-end.
    """
    redux_path = os.path.join(_engine_base(), "traiNNer-redux")
    if os.path.isdir(redux_path) and redux_path not in sys.path:
        sys.path.insert(0, redux_path)

    if detected_arch == "motsr":
        # Shared shape→kwargs rules (arch_infer.py), engine-registered factory.
        try:
            from src.core import arch_infer
        except ImportError:
            import arch_infer
        model, _sc, kw = arch_infer.build("motsr", state_dict)
        log("MoTSR instancié (" + ", ".join(f"{k}={v}" for k, v in kw.items()) + ") — mode récurrent")
        return model, 5

    if detected_arch == "tscunet":
        from traiNNer.archs.tscunet_arch import TSCUNet
        # TSCUNet self-configures (in_nc/dim/out_nc/clip_size/nb/scale/residual/
        # sigma) from state_dict shapes/key-counts and self-loads with
        # strict=True internally (see TSCUNet.__init__, state= kwarg).
        model = TSCUNet(state=state_dict)
        clip_size = int(model.clip_size)
        log(f"TSCUNet instancié depuis le checkpoint (dim={model.dim}, clip_size={clip_size}, scale={model.scale}x)")
        return model, clip_size

    if detected_arch == "temporalspan":
        from traiNNer.archs.temporal_span_arch import temporalspan
        fc = int(state_dict["center_conv.eval_conv.weight"].shape[0])
        hc = int(state_dict["history_convs.0.eval_conv.weight"].shape[0])
        n_hist = len([k for k in state_dict
                      if k.startswith("history_convs.") and k.endswith(".eval_conv.weight")])
        num_frames = n_hist + 1
        # scale 0 = "Auto": the scale is not stored as such — the first one whose strict
        # load succeeds (it was built 1x before, and a 2x/4x model failed to load)
        last = None
        for sc in ([scale] if scale > 0 else [2, 4, 1, 3, 8]):
            try:
                model = temporalspan(num_in_ch=3, num_out_ch=3, num_frames=num_frames,
                                     feature_channels=fc, scale=sc, history_channels=hc)
                model.load_state_dict(state_dict, strict=True)
                scale = sc
                break
            except Exception as e:
                last = e
        else:
            raise last
        log(f"TemporalSPAN instancié (feature_channels={fc}, history_channels={hc}, "
            f"num_frames={num_frames}, scale={scale}x)")
        return model, num_frames

    if detected_arch == "temporalspanv2":
        from traiNNer.archs.temporal_span_v2_arch import temporalspanv2
        fc = int(state_dict["m_head.weight"].shape[0])
        n_layers = len({int(k.split(".")[1]) for k in state_dict if k.startswith("m_layers.")})
        num_frames = 2 * n_layers + 1
        up_out = int(state_dict["m_upsample.0.weight"].shape[0])
        eff_scale = max(1, scale) if scale else max(1, round((up_out / fc) ** 0.5))
        # TemporalSPANBlock._forward_impl() indexes self.blocks[0..5] by hand —
        # num_blocks is effectively hardcoded to 6 regardless of what's passed;
        # we still read the checkpoint's real count for the log/sanity check.
        n_blocks = len({int(k.split(".")[3]) for k in state_dict if k.startswith("m_layers.0.blocks.")})
        if n_blocks != 6:
            log(f"⚠ TemporalSPANv2 : {n_blocks} blocs SPAB détectés (attendu 6 — "
                f"TemporalSPANBlock.forward() est câblé en dur pour 6). Le modèle "
                f"pourrait ne pas se charger correctement.")
        residual = "m_res.weight" in state_dict
        model = temporalspanv2(num_in_ch=3, num_out_ch=3, num_frames=num_frames,
                                feature_channels=fc, scale=eff_scale,
                                num_blocks=max(1, n_blocks), residual=residual)
        model.load_state_dict(state_dict, strict=True)
        log(f"TemporalSPANv2 instancié (feature_channels={fc}, num_frames={num_frames}, "
            f"num_blocks={n_blocks}, residual={residual}, scale={eff_scale}x)")
        return model, num_frames

    if detected_arch in ("tfdat", "tfdatv2"):
        # Shared rules (arch_infer.py): real scale from conv_first (MetaUpsample stores the
        # internal x4 for scale <= 2), block pattern / ffn / aim / upsampler from the
        # weights, v2 clip_size from temporal_fusion.fuse. Key sets checked there.
        try:
            from src.core import arch_infer
        except ImportError:
            import arch_infer
        model, _sc, kw = arch_infer.build(detected_arch, state_dict)
        name = "TFDAT v2" if detected_arch == "tfdatv2" else "TFDAT"
        log(f"{name} instancié (" + ", ".join(f"{k}={v}" for k, v in kw.items()) + ")")
        return model, int(kw["clip_size"])

    raise ValueError(f"Arch séquence inconnue : {detected_arch}")


def _guided_color_fix_subprocess(sr_np, ref_pil, device="auto"):
    """Guided Color Fix (aniCC) for the torch-less portable GUI: run
    guided_color_fix.py in the engine venv, read back the float32 result (.npy)."""
    import subprocess
    import tempfile
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guided_color_fix.py")
    with tempfile.TemporaryDirectory(prefix="usrs_gcf_") as td:
        sr_p, lr_p, out_p = (os.path.join(td, n) for n in ("sr.npy", "lr.png", "out.npy"))
        np.save(sr_p, np.clip(sr_np, 0, 1).astype(np.float32))   # no 8-bit round trip
        ref_pil.convert("RGB").save(lr_p)
        r = subprocess.run([_TRAINNER_VENV_PY, script, sr_p, lr_p, out_p, str(device)], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0 or not os.path.isfile(out_p):
            raise RuntimeError((r.stderr or r.stdout or "échec").strip()[-300:])
        return np.load(out_p)


def _sequence_subprocess(model_path, input_folder, output_folder, detected_arch, scale, log,
                         progress_callback, stop_event, color_fix, color_fix_wavelets,
                         color_fix_radius, color_fix_fast, color_fix_strength,
                         color_fix_device, out_bit_depth):
    """Run _sequence_video_upscale_folder in the engine venv via sequence_runner.py.
    Streams its log lines, forwards Stop through a sentinel file, returns the same
    (success, total, errors) tuple as the in-process call."""
    import subprocess
    import tempfile
    runner = os.path.join(os.path.dirname(__file__), "sequence_runner.py")
    if not os.path.isfile(_TRAINNER_VENV_PY):
        return 0, 0, [f"Python du venv moteur introuvable : {_TRAINNER_VENV_PY}"]
    if not os.path.isfile(runner):
        return 0, 0, [f"Runner séquence introuvable : {runner}"]
    tmp = tempfile.mkdtemp(prefix="usrs_seq_")
    stop_file = os.path.join(tmp, "stop")
    args_file = os.path.join(tmp, "args.json")
    with open(args_file, "w", encoding="utf-8") as f:
        json.dump({"model_path": model_path, "input_folder": input_folder,
                   "output_folder": output_folder, "detected_arch": detected_arch,
                   "scale": scale, "stop_file": stop_file, "color_fix": color_fix,
                   "color_fix_wavelets": color_fix_wavelets, "color_fix_radius": color_fix_radius,
                   "color_fix_fast": color_fix_fast, "color_fix_strength": color_fix_strength,
                   "color_fix_device": color_fix_device, "out_bit_depth": out_bit_depth}, f)
    log("Mode portable : pipeline séquence dans le venv moteur (subprocess)")
    result = None
    try:
        proc = subprocess.Popen(
            [_TRAINNER_VENV_PY, runner, args_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=0x08000000 if sys.platform == "win32" else 0)
        for line in proc.stdout:
            line = line.rstrip()
            if stop_event is not None and stop_event.is_set() and not os.path.exists(stop_file):
                open(stop_file, "w").close()
            if line.startswith("RESULT "):
                try:
                    result = json.loads(line[7:])
                except Exception:
                    pass
                continue
            if line:
                log(line)
                if callable(progress_callback):
                    try:
                        progress_callback(line)
                    except Exception:
                        pass
        proc.wait()
    except Exception as e:
        return 0, 0, [f"Lancement du runner séquence échoué : {e}"]
    finally:
        try:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass
    if not result:
        return 0, 0, [f"Le runner séquence s'est arrêté sans résultat (code {proc.returncode})."]
    return int(result.get("success", 0)), int(result.get("total", 0)), list(result.get("errors", []))


def _sequence_video_upscale_folder(
    model_path: str,
    input_folder: str,
    output_folder: str,
    detected_arch: str,
    scale: int,
    log,
    progress_callback: Optional[Callable] = None,
    stop_event=None,
    color_fix: str = "none",
    color_fix_wavelets: int = 4,
    color_fix_radius: int = 32,
    color_fix_fast: bool = False,
    color_fix_strength: float = 1.0,
    color_fix_device: str = "auto",
    out_bit_depth: int = 8,
) -> Tuple[int, int, list]:
    """Sliding-window sequence inference for the 4 video/temporal archs.

    Loads every LQ frame in the folder, and for each frame position builds a
    clip_size-frame window centered on it (duplicating the first/last frame
    to fill the window at the start/end of the sequence — the standard
    boundary strategy for sliding-window video SR, since there's no real
    neighbor frame beyond the sequence edges), runs the model once per
    output frame, and saves one upscaled frame per input frame.

    No tiling in this pass (whole-frame only) — tiling a multi-frame window
    consistently across all clip_size frames is materially more involved
    than single-image tiling and was out of budget for this session; frames
    must fit in memory/VRAM untiled.

    Bit depth: frames are loaded via pyvips (`_load_sequence_frame`),
    preserving their real on-disk precision (8 or 16-bit) instead of
    Pillow's silent 8-bit truncation — the model itself still runs in
    float32 [0, 1] regardless. `out_bit_depth` (8 default, or 16) controls
    only the *output* write (`_save_sequence_frame`): 8 reproduces the exact
    previous byte-for-byte behavior, 16 writes genuine 16-bit-per-channel
    PNGs. When color_fix is active with out_bit_depth=16, the fix is applied
    in-memory on the float tensor before the single 16-bit save (no lossy
    8-bit round-trip through disk) — for out_bit_depth=8 the previous
    save-then-reopen-then-refix codepath (`_apply_color_fix_file`) is left
    untouched, so that default case is unchanged.
    """
    errors: list = []
    if not TORCH_AVAILABLE:
        if getattr(sys, "frozen", False):
            # Portable exe: no torch in-process → same pipeline, run by the engine venv.
            return _sequence_subprocess(
                model_path, input_folder, output_folder, detected_arch, scale, log,
                progress_callback, stop_event, color_fix, color_fix_wavelets,
                color_fix_radius, color_fix_fast, color_fix_strength, color_fix_device,
                out_bit_depth)
        return 0, 0, ["PyTorch non installé"]
    if not IMAGING_AVAILABLE:
        return 0, 0, ["Pillow/NumPy non installé"]
    if out_bit_depth == 16 and not PYVIPS_AVAILABLE:
        return 0, 0, ["Sortie 16-bit demandée mais pyvips n'est pas installé "
                       "(pip install pyvips pyvips-binary)."]
    if not os.path.isdir(input_folder):
        return 0, 0, [f"Dossier introuvable : {input_folder}"]

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    import re as _re
    def _nat_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in _re.split(r'(\d+)', s)]
    files = sorted((f for f in os.listdir(input_folder)
                    if os.path.splitext(f)[1].lower() in exts), key=_nat_key)
    if len(files) < 2:
        return 0, 0, [f"{detected_arch} nécessite une séquence de plusieurs frames "
                       f"(au moins 2 trouvées : {len(files)})."]

    os.makedirs(output_folder, exist_ok=True)

    def _cb(msg: str) -> None:
        if callable(progress_callback):
            try:
                progress_callback(msg)
            except Exception:
                pass
        log(msg)

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    try:
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(model_path, device="cpu")
        else:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = extract_weights(checkpoint)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _cb(f"Device : {device}")
        model, clip_size = _build_sequence_model(detected_arch, state_dict, scale, _cb)
        model = model.to(device).eval()
    except Exception as e:
        return 0, 0, [f"Chargement du modèle séquence échoué : {e}\n{traceback.format_exc()}"]

    half = clip_size // 2
    n = len(files)

    # Preload every frame as a float tensor (folder/batch mode assumes a
    # bounded sequence — same assumption the rest of upscale_folder makes).
    # _load_sequence_frame normalizes by the frame's REAL dtype max (255 for
    # 8-bit, 65535 for 16-bit, ...) instead of a hardcoded /255.
    frame_tensors = []
    try:
        for f in files:
            arr = _load_sequence_frame(os.path.join(input_folder, f))
            frame_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    except Exception as e:
        return 0, 0, [f"Lecture des frames échouée : {e}"]

    # MoTSR is recurrent: each frame also consumes the two previous RAW (unclamped)
    # SR outputs. Same algorithm as the author's motsr_video.py (verified 0.0 diff):
    # scene cuts = mean |thumbnail(64x36) diff| > 0.30 reset the state; inside a
    # scene the 5-LR window is clamped to the scene; the state is seeded with
    # image_only() of the scene's first frame.
    is_motsr = detected_arch == "motsr"
    if is_motsr:
        import torch.nn.functional as _Fm
        scene_start = [0] * n
        _prev_thumb, _start = None, 0
        for _i in range(n):
            _th = _Fm.interpolate(frame_tensors[_i].unsqueeze(0), size=(36, 64),
                                  mode="bilinear", align_corners=False)
            if _prev_thumb is not None and float((_th - _prev_thumb).abs().mean()) > 0.30:
                _start = _i
            scene_start[_i] = _start
            _prev_thumb = _th
        scene_end = [0] * n
        _end = n
        for _i in range(n - 1, -1, -1):
            scene_end[_i] = _end
            if scene_start[_i] == _i:
                _end = _i
        _cuts = sum(1 for _i in range(1, n) if scene_start[_i] == _i)
        _cb(f"MoTSR : {n} frames, {_cuts} coupure(s) de scène détectée(s) (état récurrent réinitialisé)")
        _state = {"older": None, "previous": None}
        _gpu = {}  # frame index -> device tensor, only the current 5-frame window

    def _motsr_step(i):
        left, right = scene_start[i], scene_end[i]
        for _k in [k for k in _gpu if k < i - 2]:
            del _gpu[_k]
        def lr(k):
            k = min(max(k, left), right - 1)
            if k not in _gpu:
                _gpu[k] = frame_tensors[k].unsqueeze(0).to(device)
            return _gpu[k]
        with torch.no_grad():
            if i == left:
                seed = model.image_only(lr(left))
                _state["older"], _state["previous"] = seed, seed
            pred = model.forward_step(lr(i - 2), _state["older"], lr(i - 1), _state["previous"],
                                      lr(i), lr(i + 1), lr(i + 2))
        if not torch.isfinite(pred).all():
            raise RuntimeError("prédiction non finie (NaN/Inf)")
        _state["older"], _state["previous"] = _state["previous"], pred
        return pred

    success = 0
    for i, fname in enumerate(files):
        if _stopped():
            errors.append("Arrêt demandé par l'utilisateur.")
            break
        _cb(f"[{i + 1}/{n}] {fname} (clip_size={clip_size})")
        try:
            if is_motsr:
                out = _motsr_step(i)
            else:
                # Boundary strategy: clamp the window index into [0, n-1], which
                # duplicates the first/last frame for windows that would
                # otherwise run off the sequence edge.
                window_idx = [min(max(i + j - half, 0), n - 1) for j in range(clip_size)]
                window = torch.stack([frame_tensors[k] for k in window_idx], dim=0)
                window = window.unsqueeze(0).to(device)  # (1, T, C, H, W)
                with torch.no_grad():
                    out = model(window)
            out = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
            base, _ = os.path.splitext(fname)
            out_path = os.path.join(output_folder, f"{base}_upscaled.png")
            do_color_fix = color_fix != "none" and color_fix_strength > 0.0
            if out_bit_depth == 16:
                # In-memory color fix (if any) then a single genuine 16-bit
                # save — avoids the 8-bit round-trip through disk that the
                # default (8-bit) codepath below still uses unchanged.
                if do_color_fix:
                    lq_pil = Image.open(os.path.join(input_folder, fname)).convert("RGB")
                    _cb(f"Color Fix ✓ [{color_fix}] strength={color_fix_strength:.2f}")
                    out = _cf_try(out, lq_pil, _cb, method=color_fix,
                                  wavelets=color_fix_wavelets, radius=color_fix_radius,
                                  fast=color_fix_fast, strength=color_fix_strength,
                                  planes=[0, 1, 2], device=color_fix_device)
                _save_sequence_frame(out, out_path, bit_depth=16)
            else:
                # Unchanged from before: save 8-bit, then color-fix by
                # reopening the saved file (existing shared helper).
                _save_sequence_frame(out, out_path, bit_depth=8)
                if do_color_fix:
                    _apply_color_fix_file(out_path, os.path.join(input_folder, fname),
                                          color_fix, color_fix_wavelets, color_fix_radius,
                                          color_fix_fast, color_fix_strength, None,
                                          color_fix_device, "PNG", 8, 95, _cb)
            success += 1
        except Exception as e:
            errors.append(f"{fname}: {e}")
            _cb(f"  Erreur : {e}")

    return success, n, errors


def upscale_folder(
    model_path: str,
    input_folder: str,
    output_folder: str,
    scale: int = 0,
    tile_size: int = 256,
    tile_pad: int = 32,
    use_amp: bool = True,
    callback: Optional[Callable] = None,
    stop_event=None,
    # ── Serialization ──
    serialize: bool = False,
    serialize_start: int = 0,
    # ── Color Fix ──
    color_fix: str = "none",
    color_fix_wavelets: int = 4,
    color_fix_radius: int = 32,
    color_fix_fast: bool = False,
    color_fix_strength: float = 1.0,
    color_fix_device: str = "auto",
    # ── Persistent batch subprocess (v2.5.5) ──
    persistent_batch: bool = False,
    # ── Dandere2x (v2.5.5) ──
    dandere_mode: bool = False,
    dandere_block_size: int = 16,
    dandere_threshold: float = 0.02,
    dandere_full_skip: bool = False,  # skip entire frame if global diff < threshold
    dandere_full_skip_threshold: float = 0.005,
    # ── Temporal Fix (v2.5.7) ──
    temporal_fix: bool = False,
    temporal_fix_strength: float = 0.5,
    temporal_fix_window: int = 7,
    temporal_fix_precision: str = "float32",
    temporal_fix_device: str = "auto",
) -> Tuple[int, int, list]:
    """
    Upscale all images in a folder.

    persistent_batch: Keep model loaded in VRAM across all frames (subprocess archs only).
                      Eliminates ~2-5s model reload per image. Major gain on large batches.

    dandere_mode: Dandere2x-inspired block skip.
      dandere_full_skip=False (default): block-level compositing — upscale every frame,
        but copy unchanged blocks from previous SR output (saves color fix + I/O time).
      dandere_full_skip=True: skip entire frames that are nearly identical to the previous
        (global MSE < dandere_full_skip_threshold). Zero upscale cost for those frames.
        Best for anime with many static shots.

    Returns:
        (success_count, total_count, errors: list[str])
    """
    if not os.path.isdir(input_folder):
        return 0, 0, [f"Dossier introuvable : {input_folder}"]

    # Ensure custom archs present before batch (covers persistent worker too)
    ensure_custom_archs(callback if callable(callback) else None)

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    raw_files = [f for f in os.listdir(input_folder)
                 if os.path.splitext(f)[1].lower() in exts]

    if not raw_files:
        return 0, 0, ["Aucune image trouvée"]

    # Natural sort (frame_1 < frame_2 < … < frame_10)
    import re as _re
    def _nat_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in _re.split(r'(\d+)', s)]
    files = sorted(raw_files, key=_nat_key)

    os.makedirs(output_folder, exist_ok=True)

    def _cb(msg: str) -> None:
        if callback:
            callback(msg)

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    # Determine if the model is a subprocess arch (needs persistent worker).
    # Use all state_dict keys (metadata only for safetensors — no tensor data loaded).
    _is_subprocess_arch = _is_esc_arch = _is_neosr_general_arch = False
    if os.path.isfile(model_path) and persistent_batch:
        _route = _persistent_route(model_path)
        _is_subprocess_arch = _route == "trainner"
        _is_esc_arch = _is_neosr_general_arch = _route == "neosr"

    # ── Persistent batch session (traiNNer subprocess archs) ──────────────────
    _session: Optional[PersistentBatchSession] = None
    if persistent_batch and _is_subprocess_arch and not str(model_path).lower().endswith(".onnx"):
        _session = PersistentBatchSession(
            venv_py=_TRAINNER_VENV_PY,
            model_path=model_path,
            tile_size=tile_size,
            tile_pad=tile_pad,
            use_amp=use_amp,
            log=_cb,
            scale_hint=scale,  # user-selected scale (0=auto)
        )
        if not _session.start():
            _session = None
            _cb("[PersistentBatch] Démarrage échoué — mode normal utilisé")

    # ── Persistent ESC/NeoSR-general session (neosr venv) ─────────────────────
    _esc_session: Optional[PersistentNeoSRSession] = None
    if persistent_batch and (_is_esc_arch or _is_neosr_general_arch) and os.path.isfile(_NEOSR_VENV_PY):
        _esc_session = PersistentNeoSRSession(
            venv_py=_NEOSR_VENV_PY,
            model_path=model_path,
            tile_size=tile_size,
            tile_pad=tile_pad,
            use_amp=use_amp,
            log=_cb,
            scale_hint=scale,  # required for msdan (scale not recoverable from weights)
        )
        if not _esc_session.start():
            _esc_session = None
            _cb("[ESC-Persistent] Démarrage échoué — subprocess par image utilisé")

    success = 0
    errors  = []
    skipped_dandere = 0

    # Dandere2x state
    _prev_lq_arr: Optional["np.ndarray"] = None  # previous input as float32 [H,W,3]
    _prev_sr_arr: Optional["np.ndarray"] = None  # previous SR output as float32

    # Temporal Fix state (v2.5.7)
    _tf_proc = None
    # Maps output filenames to their final disk paths (needed for latency writes)
    _tf_pending: list = []  # list of (out_path,) for frames not yet written
    if temporal_fix and IMAGING_AVAILABLE:
        try:
            from src.core.temporal_fix import TemporalFixProcessor
            _tf_proc = TemporalFixProcessor(
                window=temporal_fix_window,
                strength=temporal_fix_strength,
                precision=temporal_fix_precision,
                device=temporal_fix_device,
            )
            _cb(f"[TemporalFix] Activé — fenêtre={temporal_fix_window}, "
                f"intensité={temporal_fix_strength:.2f}, "
                f"latence={_tf_proc.latency} frames")
        except Exception as _tfe:
            _cb(f"[TemporalFix] Import échoué ({_tfe}) — désactivé")

    total = len(files)
    try:
        for i, fname in enumerate(files):
            if _stopped():
                errors.append("Arrêt demandé par l'utilisateur.")
                break

            in_path = os.path.join(input_folder, fname)
            base, _ = os.path.splitext(fname)
            if serialize:
                out_name = f"{serialize_start + i:05d}.png"
            else:
                out_name = f"{base}_upscaled.png"
            out_path = os.path.join(output_folder, out_name)

            _cb(f"[{i+1}/{total}] {fname}")

            # ── Dandere2x — load current LQ frame ────────────────────────────
            _curr_lq_arr: Optional["np.ndarray"] = None
            if dandere_mode and IMAGING_AVAILABLE:
                try:
                    _curr_lq_arr = _pil_to_float(Image.open(in_path))
                except Exception:
                    pass

            # ── Dandere2x — full-frame skip ────────────────────────────────
            if (dandere_mode and dandere_full_skip and
                    _prev_lq_arr is not None and _curr_lq_arr is not None and
                    _prev_sr_arr is not None):
                diff = dandere_frame_similarity(_prev_lq_arr, _curr_lq_arr)
                if diff < dandere_full_skip_threshold:
                    # Frame virtually identical → copy previous SR output
                    _cb(f"  [Dandere2x] Skip (diff={diff:.4f} < {dandere_full_skip_threshold})")
                    skipped_dandere += 1
                    if IMAGING_AVAILABLE:
                        _float_to_pil(_prev_sr_arr).save(out_path)
                    _prev_lq_arr = _curr_lq_arr
                    success += 1
                    continue

            # ── Normal upscale ────────────────────────────────────────────────
            if _esc_session is not None:
                # Persistent ESC session (neosr venv — model stays loaded)
                ok, msg = _esc_session.infer(in_path, out_path)
                if not ok:
                    _cb(f"  [ESC-Persistent] Erreur : {msg} — fallback subprocess")
                    ok, msg = upscale_image(
                        model_path, in_path, out_path,
                        scale=scale, tile_size=tile_size, tile_pad=tile_pad,
                        use_amp=use_amp, callback=callback,
                        stop_event=stop_event,
                        color_fix=color_fix,
                        color_fix_wavelets=color_fix_wavelets,
                        color_fix_radius=color_fix_radius,
                        color_fix_fast=color_fix_fast,
                        color_fix_strength=color_fix_strength,
                        color_fix_device=color_fix_device,
                    )
            elif _session is not None:
                # Persistent traiNNer subprocess session (infer returns ok, msg, extra)
                ok, msg, _extra = _session.infer(in_path, out_path)
                if _extra.get("note"):
                    _cb(f"  {_extra['note']}")
                if not ok:
                    _cb(f"  [PersistentBatch] Erreur : {msg.strip().splitlines()[-1] if msg.strip() else msg}"
                        " — fallback subprocess")
                    ok, msg = upscale_image(
                        model_path, in_path, out_path,
                        scale=scale, tile_size=tile_size, tile_pad=tile_pad,
                        use_amp=use_amp, callback=callback,
                        stop_event=stop_event,
                        color_fix=color_fix,
                        color_fix_wavelets=color_fix_wavelets,
                        color_fix_radius=color_fix_radius,
                        color_fix_fast=color_fix_fast,
                        color_fix_strength=color_fix_strength,
                        color_fix_device=color_fix_device,
                    )
            else:
                ok, msg = upscale_image(
                    model_path, in_path, out_path,
                    scale=scale, tile_size=tile_size, tile_pad=tile_pad,
                    use_amp=use_amp, callback=callback,
                    stop_event=stop_event,
                    color_fix=color_fix,
                    color_fix_wavelets=color_fix_wavelets,
                    color_fix_radius=color_fix_radius,
                    color_fix_fast=color_fix_fast,
                    color_fix_strength=color_fix_strength,
                    color_fix_device=color_fix_device,
                )

            if ok:
                success += 1
                # ── Dandere2x — block compositing ──────────────────────────────
                if (dandere_mode and not dandere_full_skip and
                        _prev_lq_arr is not None and _curr_lq_arr is not None and
                        _prev_lq_arr.shape == _curr_lq_arr.shape and
                        IMAGING_AVAILABLE):
                    try:
                        _curr_sr_arr = _pil_to_float(Image.open(out_path))
                        if _prev_sr_arr is not None:
                            _diff_mask = dandere_compute_diff(
                                _prev_lq_arr, _curr_lq_arr,
                                block_size=dandere_block_size,
                                threshold=dandere_threshold,
                            )
                            changed_pct = _diff_mask.mean() * 100
                            _cb(f"  [Dandere2x] {changed_pct:.1f}% blocs changés")
                            _composed = dandere_compose(
                                _prev_sr_arr, _curr_sr_arr,
                                _diff_mask,
                                block_size=dandere_block_size,
                                scale=scale if scale > 0 else 4,
                            )
                            _float_to_pil(_composed).save(out_path)
                            _prev_sr_arr = _composed
                        else:
                            _prev_sr_arr = _curr_sr_arr
                    except Exception as _de:
                        _cb(f"  [Dandere2x] Erreur compositing : {_de}")
                elif ok and IMAGING_AVAILABLE:
                    # First frame or no dandere — just cache SR
                    if dandere_mode:
                        try:
                            _prev_sr_arr = _pil_to_float(Image.open(out_path))
                        except Exception:
                            pass

                # ── Temporal Fix — push frame into sliding window ──────────────
                if _tf_proc is not None and IMAGING_AVAILABLE:
                    _tf_pending.append(out_path)
                    try:
                        frame_np = _pil_to_float(Image.open(out_path))
                        tf_result = _tf_proc.push(frame_np)
                        if tf_result is not None:
                            # The processed frame corresponds to the oldest pending path
                            _write_path = _tf_pending.pop(0)
                            _float_to_pil(tf_result).save(_write_path)
                    except Exception as _tfe2:
                        _cb(f"  [TemporalFix] Erreur frame {i}: {_tfe2}")
                        _tf_pending.clear()
                        _tf_proc = None  # disable on error

            else:
                errors.append(f"{fname}: {msg}")

            # Update dandere2x LQ reference
            if dandere_mode:
                _prev_lq_arr = _curr_lq_arr

    finally:
        if _session is not None:
            _session.stop()
        if _esc_session is not None:
            _esc_session.stop()

        # ── Temporal Fix — flush remaining frames ─────────────────────────────
        if _tf_proc is not None and _tf_pending and IMAGING_AVAILABLE:
            _cb(f"[TemporalFix] Flush des {len(_tf_pending)} dernières frames…")
            try:
                for tf_result in _tf_proc.flush():
                    if not _tf_pending:
                        break
                    _write_path = _tf_pending.pop(0)
                    _float_to_pil(tf_result).save(_write_path)
            except Exception as _tff:
                _cb(f"[TemporalFix] Erreur flush : {_tff}")

    if skipped_dandere > 0:
        _cb(f"[Dandere2x] {skipped_dandere}/{total} frames sautées (identiques)")

    return success, total, errors
