"""
neosr_general_runner.py - Inference subprocess for neosr architectures.

Handles archs that call net_opt() at module level (ninasr, lmlt, eimn, drct_s, etc.)
by patching arch_util.net_opt BEFORE any arch import.

Usage:
    <neosr_venv_python> neosr_general_runner.py <model_path> <input_path> <output_path>

Exit 0 = success, 1 = error. All output to stdout (UTF-8).
"""
import sys
import os
import math
import re
import traceback

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
NEOSR_PATH = os.path.join(_BASE, "neosr")
if os.path.isdir(NEOSR_PATH) and NEOSR_PATH not in sys.path:
    sys.path.insert(0, NEOSR_PATH)
    os.chdir(NEOSR_PATH)

# Stub neosr packages to prevent __init__.py from running
# (avoids parse_options() / rng() module-level calls requiring -opt argv)
import types as _t
for _n, _p in [
    ('neosr',       os.path.join(NEOSR_PATH, 'neosr')),
    ('neosr.archs', os.path.join(NEOSR_PATH, 'neosr', 'archs')),
]:
    if _n not in sys.modules:
        _m = _t.ModuleType(_n)
        _m.__path__ = [_p]
        _m.__package__ = _n
        sys.modules[_n] = _m
if 'neosr' in sys.modules and 'neosr.archs' in sys.modules:
    sys.modules['neosr'].archs = sys.modules['neosr.archs']
del _t, _n, _p, _m

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"[NeoSR] ERREUR import: {e}", flush=True)
    sys.exit(1)


# ── Architecture signatures (key substrings → arch name) ──────────────────────
ARCH_SIGNATURES = {
    # ── v2.7.0 batch : 8 archs NeoSR supplémentaires — placées en premier, avant
    # tout ce qui partage un substring générique ("layers.0.residual_group.blocks"
    # + "conv_after_body" avec swinir/hat plus bas, ".correlation.qv.linear." avec
    # hit_sir/hit_sng côté quick_upscale.py) — mêmes signatures que
    # quick_upscale.py::ARCH_SIGNATURES (voir ses commentaires pour le détail des
    # collisions évitées).
    "plainusr": ["head.weight", "block1.body.0.expand_conv.weight", "tail.0.weight"],
    "asid": ["block0.res_end.weight", "block1.esa.conv_f.weight"],
    "grformer": ["layers.0.blocks.0.attn.ESRPB_MLP.0.weight",
                 "layers.0.blocks.0.attn.q1.weight"],
    "dct": ["layers.0.blocks.0.swsa.attns.0.pos.pos_proj.weight",
            "layers.0.blocks.0.lla.qkv.weight"],
    "flexnet": ["in_to_feat.weight", "short_cut.conv11.weight"],
    "hit_srf": ["layers.0.residual_group.blocks.0.correlation.pos.pos3.2.weight",
                "layers.0.residual_group.blocks.0.mlp.dwconv.depthwise_conv.0.weight"],
    "hma": ["relative_position_index_SA",
            "layers.0.residual_group.blocks.0.fused_conv.weight"],
    # srformer (also covers the srformer_medium preset — same class, hyperparams
    # deduced from real shapes below).
    "srformer": ["layers.0.residual_group.blocks.0.attn.kv.weight",
                 "layers.0.residual_group.blocks.0.attn.q.weight"],

    "ninasr": ["head.0.bias", "body.0.body.2.body"],
    "lmlt":   ["feats.0.lhsb", "to_img.0.weight"],
    "eimn":   ["block1.0.attn.region", "block1.0.layer_scale_1"],
    "drct":   ["layers.0.swin1", "layers.0.adjust1"],
    "span":   ["feats.0.block", "feats.0.sk"],
    "omnisr": ["residual_layer", "osag"],
    "realplksr": ["feats.0.lk.conv", "dysample"],
    "compact":   ["body.0.weight", "body.2.weight"],
    "esrgan":    ["body.0.rdb1", "RRDB_trunk"],
    "rcan":      ["body.0.body.0.body.0.weight"],
    # AetherNet (Phhofm, custom neosr arch — injected via ensure_custom_archs()).
    # Same signature as quick_upscale.py's ARCH_SIGNATURES entry.
    "aethernet": ["conv_after_body", "stages", "conv_before_upsample"],
    # HASN — 6 fixed blocks, upscale=upscale default at module level → net_opt() required.
    "hasn": ["block_1.dwconv.conv.weight", "block_6.dwconv.conv.weight", "esa.conv_f"],
    # MSDAN — BSConv_first/DFEB body + upsampler.BSConv_2/BSConv_3 (both always
    # present regardless of trained scale — see build_model() for why scale
    # MUST come from scale_hint, not from the weights).
    "msdan": ["BSConv_first.pw.weight", "upsampler.BSConv_2.pw.weight", "upsampler.BSConv_3.pw.weight"],
    # CFSR — LargeKernelConv reparam (lkb_reparam) + ConvMod attn.
    "cfsr": ["layers.0.blocks.0.attn.a.2.lkb_reparam.weight", "layers.0.blocks.0.mlp.fc2.weight",
             "conv_after_body.weight"],
}


def detect_arch(state_dict: dict) -> str | None:
    keys_str = " ".join(state_dict.keys())
    for arch, sigs in ARCH_SIGNATURES.items():
        if all(s in keys_str for s in sigs):
            return arch
    return None


def detect_scale(state_dict: dict) -> int:
    # AetherNet registers its scale as an exact int64 buffer — read it directly.
    _scale_t = state_dict.get("scale_tensor")
    if _scale_t is not None:
        try:
            return int(_scale_t.item())
        except Exception:
            pass
    # ninasr tail.1, eimn tail.0, lmlt to_img.0, hasn upsampler.0 → out_ch = 3 * scale^2
    # plainusr: tail.0 (already covered). asid: up.0. flexnet: to_img.0 (already
    # covered). grformer/dct/hit_srf: upsample.0 (already covered, UpsampleOneStep).
    for probe in ("tail.1.weight", "tail.0.weight", "to_img.0.weight", "upsampler.0.weight",
                  "upsample.0.weight", "up.0.weight"):
        w = state_dict.get(probe)
        if w is not None:
            ch = w.shape[0]
            s = int(round(math.sqrt(ch / 3)))
            if s >= 1 and s * s * 3 == ch:
                return s
    # compact arch: last body conv → out_ch = 3 * scale^2
    # body layout: [first_conv, act, (conv,act)*num_conv, last_conv]
    # last_conv shape = [3*scale^2, num_feat, 3, 3]
    compact_conv_idxs = sorted(
        int(k.split(".")[1]) for k in state_dict
        if k.startswith("body.") and k.endswith(".weight")
        and len(state_dict[k].shape) == 4
    )
    if compact_conv_idxs:
        last_body_key = f"body.{compact_conv_idxs[-1]}.weight"
        w = state_dict.get(last_body_key)
        if w is not None:
            out_ch = w.shape[0]
            s = int(round(math.sqrt(out_ch / 3)))
            if s >= 1 and s * s * 3 == out_ch:
                return s
    # drct / swinir / hma-style pixel_shuffle upsamplers. A 4x upsampler is two x2
    # stages (upsample.0 / upsample.2): the stages multiply. The first stage alone was
    # returned before, so 4x DRCT / HMA / SRFormer models were rebuilt as 2x.
    _stages = []
    for k, v in state_dict.items():
        kl = k.lower()
        if ("upsample" in kl or "upconv" in kl) and hasattr(v, "shape") and len(v.shape) == 4:
            out_ch, in_ch = v.shape[0], v.shape[1]
            if out_ch > in_ch:
                s = int(round(math.sqrt(out_ch / in_ch)))
                if s in (2, 3, 4, 8) and s * s * in_ch == out_ch:
                    _stages.append(s)
    if _stages:
        s = math.prod(_stages)
        if s in (2, 3, 4, 8):
            return s
    return 1  # default: 1x for neosr benchmark models



# ── NeoSR archs rebuilt from their defaults (2026-10-29) ─────────────────────────
# catanet / krgn / man / mosrv2 trained with NeoSR: the sizes readable from the shapes
# are used, the rest stays at the NeoSR defaults, and the scale is the one whose STRICT
# load succeeds (a wrong rebuild never loads silently). Before, these models were sent
# to the traiNNer runner (which has no such NeoSR archs) and failed.
_EXTRA_ARCH_SIGNATURES = {
    "catanet": ["first_conv.weight", "irca_attn.to_k.weight"],
    "krgn": ["head_conv.weight", "features_fusion_module", "sub_mean.weight"],
    "man_neosr": ["head.weight", ".LKA.", "tail.0.weight"],
    "mosrv2_neosr": ["gblocks.", "to_img.0.weight"],
}


def _detect_extra_arch(sd) -> "str | None":
    keys_str = " ".join(sd.keys())
    if "MetaUpsample" in keys_str:  # traiNNer-redux MoSRv2 / RTMoSR, not NeoSR's
        return None
    for arch, sigs in _EXTRA_ARCH_SIGNATURES.items():
        if arch == "man_neosr" and "sub_mean.weight" in sd:
            continue  # traiNNer-redux MAN (MeanShift layers)
        if all(s in keys_str for s in sigs):
            return arch
    return None


def _make_extra(arch, sd, scale, unshuffle=True):
    import neosr.archs.arch_util as _au
    _au.net_opt = lambda: (scale, True)
    if arch == "catanet":
        from neosr.archs.catanet_arch import catanet
        return catanet(dim=int(sd["first_conv.weight"].shape[0]), upscale=scale)
    if arch == "krgn":
        from neosr.archs.krgn_arch import krgn
        n_resgroups = len({k.split(".")[1] for k in sd if k.startswith("body.")})
        return krgn(n_feats=int(sd["head_conv.weight"].shape[0]),
                    n_resgroups=n_resgroups or 9, scale=scale)
    if arch == "man_neosr":
        from neosr.archs.man_arch import man
        groups = {k.split(".")[1] for k in sd if re.match(r"^body\.\d+\.body\.\d+\.", k)}
        blocks = {k.split(".")[3] for k in sd if k.startswith("body.0.body.")}
        return man(n_resblocks=len(blocks) or 36, n_resgroups=len(groups) or 1,
                   n_feats=int(sd["head.weight"].shape[0]), scale=scale)
    if arch == "mosrv2_neosr":
        from neosr.archs.mosrv2_arch import mosrv2
        dims = [int(v.shape[0]) for k, v in sd.items()
                if re.match(r"^gblocks\.\d+\.weight$", k) and len(v.shape) == 4]
        n_block = len({k.split(".")[1] for k in sd if k.startswith("gblocks.") and ".gamma" in k})
        return mosrv2(scale=scale, dim=dims[0] if dims else 64, n_block=n_block or 24,
                      unshuffle_mod=unshuffle)
    raise ValueError(arch)


def build_extra(arch, sd, scale_hint=0):
    """(model, scale) — first scale (hint first) whose strict load succeeds."""
    tried = []
    for s in [scale_hint] + [2, 4, 1, 3, 8]:
        if s <= 0 or s in tried:
            continue
        tried.append(s)
        for unsh in ((True, False) if arch == "mosrv2_neosr" else (True,)):
            try:
                m = _make_extra(arch, sd, s, unsh)
                m.load_state_dict(sd, strict=True)
                return m, s
            except Exception:
                continue
    raise RuntimeError(f"{arch} : aucun scale ({tried}) ne charge ces poids avec les "
                       "réglages NeoSR par défaut (hyper-paramètres modifiés ?)")

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


def build_model(arch: str, state_dict: dict, scale: int):
    """Instantiate arch and load weights. Returns model on CPU."""
    import neosr.archs.arch_util as _au
    _au.net_opt = lambda: (scale, True)

    if arch == "ninasr":
        from neosr.archs.ninasr_arch import ninasr
        model = ninasr(scale=scale)

    elif arch == "lmlt":
        from neosr.archs.lmlt_arch import lmlt
        # dim / n_blocks from the weights (lmlt_tiny 36, lmlt_large 84, base 60)
        _ti = state_dict.get("to_img.0.weight")
        _nb = len({k.split(".")[1] for k in state_dict if re.match(r"^feats\.\d+\.", k)}) or 8
        model = lmlt(upscaling_factor=scale, dim=int(_ti.shape[1]) if _ti is not None else 60, n_blocks=_nb)

    elif arch == "eimn":
        from neosr.archs.eimn_arch import eimn
        head_w = state_dict.get("head.0.weight")
        embed_dims = int(head_w.shape[0]) if head_w is not None else 64
        num_stages = sum(
            1 for k in state_dict
            if k.startswith("block") and k.endswith(".region.weight")
        )
        model = eimn(
            embed_dims=embed_dims, scale=scale,
            num_stages=num_stages if num_stages > 0 else 16,
        )

    elif arch == "drct":
        from neosr.archs.drct_arch import drct
        # embed_dim and layer count from the weights: covers drct_s (48, 4 x 2 blocks),
        # drct (180, 6 layers), and traiNNer-redux drct_l / drct_xl (12 / 14 layers),
        # which the neosr defaults (6 layers) failed to load.
        cf_w = state_dict.get("conv_first.weight")
        _dim = int(cf_w.shape[0]) if cf_w is not None else 180
        _n = len({k.split(".")[1] for k in state_dict if re.match(r"^layers\.\d+\.", k)}) or 6
        _d = 2 if _dim <= 60 else 6
        model = drct(upscale=scale, embed_dim=_dim, depths=(_d,) * _n, num_heads=(6,) * _n)

    elif arch == "aethernet":
        # AetherNet (Phhofm) — fully parameterized reconstruction from state_dict
        # tensor shapes (embed_dim/depths/kernels/mlp_ratio/attn flags), not from
        # the preset name — covers all 7 UI variants (aether_mobile..extreme) and
        # any custom fine-tune with a different depth/width combo.
        # res_scale is a plain python float (not a tensor) — it does not affect
        # load_state_dict() but does affect forward() math; the 7 official presets
        # only use two values (0.2 for the two narrowest — mobile/tiny — 0.1 for
        # the rest), which embed_dim<=64 distinguishes exactly.
        from neosr.archs.aether_core import AetherNet
        cf_w = state_dict.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 64

        stage_blocks: dict = {}
        for k in state_dict:
            m = re.match(r"^stages\.(\d+)\.(\d+)\.", k)
            if m:
                si, bi = int(m.group(1)), int(m.group(2))
                stage_blocks.setdefault(si, set()).add(bi)
        num_stages = len(stage_blocks) if stage_blocks else 4
        depths = tuple(len(stage_blocks.get(i, set())) for i in range(num_stages)) or (4, 4, 4, 4)

        lk_w = state_dict.get("stages.0.0.conv.lk_conv.weight")
        fused_init = lk_w is None
        if fused_init:
            lk_w = state_dict.get("stages.0.0.conv.fused_conv.weight")
        lk_kernel = int(lk_w.shape[2]) if lk_w is not None else 13
        sk_w = state_dict.get("stages.0.0.conv.sk_conv.weight")
        sk_kernel = int(sk_w.shape[2]) if sk_w is not None else 5

        gate_w = state_dict.get("stages.0.0.ffn.conv_gate.weight")
        mlp_ratio = (float(gate_w.shape[0]) / embed_dim) if (gate_w is not None and embed_dim) else 1.5

        use_channel_attn = any(k.startswith("stages.0.0.channel_attn.") for k in state_dict)
        use_spatial_attn = any(k.startswith("stages.0.0.spatial_attn.") for k in state_dict)
        norm_type = "deployment" if "stages.0.0.norm.running_mean" in state_dict else "layernorm"
        res_scale = 0.2 if embed_dim <= 64 else 0.1

        model = AetherNet(
            embed_dim=embed_dim, depths=depths, lk_kernel=lk_kernel, sk_kernel=sk_kernel,
            fused_init=fused_init, mlp_ratio=mlp_ratio, use_channel_attn=use_channel_attn,
            use_spatial_attn=use_spatial_attn, norm_type=norm_type, scale=scale,
            res_scale=res_scale,
        )

    elif arch == "hasn":
        from neosr.archs.hasn_arch import hasn
        w = state_dict.get("conv_1.weight")
        feature_channels = int(w.shape[0]) if w is not None else 52
        model = hasn(feature_channels=feature_channels, upscale=scale)

    elif arch == "span":
        from neosr.archs.span_arch import span
        model = span(scale=scale)

    elif arch == "omnisr":
        from neosr.archs.omnisr_arch import omnisr
        model = omnisr(scale=scale)

    elif arch == "realplksr":
        from neosr.archs.realplksr_arch import realplksr
        model = realplksr(scale=scale)

    elif arch == "compact":
        from neosr.archs.compact_arch import compact
        # Detect num_feat and num_conv from state_dict to match saved model exactly
        w0 = state_dict.get("body.0.weight")
        num_feat = int(w0.shape[0]) if w0 is not None else 64
        # body layout: [first_conv, first_act, (conv, act)*num_conv, last_conv]
        # → last conv is at index 2 + 2*num_conv  → num_conv = (last_idx - 2) // 2
        conv_idxs = sorted(
            int(k.split(".")[1]) for k in state_dict
            if k.startswith("body.") and k.endswith(".weight")
            and len(state_dict[k].shape) == 4  # Conv2d weights are 4D
        )
        last_idx = conv_idxs[-1] if conv_idxs else 34
        num_conv = max(0, (last_idx - 2) // 2)
        model = compact(num_feat=num_feat, num_conv=num_conv, upscale=scale)

    elif arch == "esrgan":
        from neosr.archs.rrdbnet_arch import rrdbnet
        model = rrdbnet(scale=scale)

    elif arch == "msdan":
        # MSDAN — upsampler.__init__ builds ALL of BSConv_2/BSConv1_2/BSConv_3/
        # BSConv1_3/up_2/up_3 unconditionally, regardless of upscale_factor —
        # the state_dict is byte-identical whether the checkpoint was trained
        # at x2, x3 or x4. Only `channels`/`num_DFEB` are recoverable from
        # weight shapes; `scale` MUST come from `scale` (the caller's
        # scale_hint — see run()/__main__ below), never guessed. Only 2/3/4
        # are valid (forward() silently returns None for anything else).
        from neosr.archs.msdan_arch import msdan
        w = state_dict.get("BSConv_first.pw.weight")
        channels = int(w.shape[0]) if w is not None else 48
        idxs = {
            int(m.group(1)) for k in state_dict
            if (m := re.match(r"^layers\.(\d+)\.EBFB\.", k))
        }
        num_dfeb = len(idxs) if idxs else 8
        msdan_scale = scale if scale in (2, 3, 4) else 4
        model = msdan(channels=channels, num_DFEB=num_dfeb, upscale_factor=msdan_scale)

    elif arch == "cfsr":
        # CFSR — merge_all() runs unconditionally at the end of __init__ (fuses
        # MLP's sobel/laplacian branches into pos.weight/bias and deletes the
        # separate params), so a freshly-constructed instance's state_dict
        # already matches the checkpoint's merged shape — no special-casing
        # needed, unlike a "trained-unmerged vs deployed-merged" arch would.
        from neosr.archs.cfsr_arch import cfsr
        cf_w = state_dict.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 48

        layer_blocks: dict = {}
        for k in state_dict:
            m = re.match(r"^layers\.(\d+)\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_blocks.setdefault(li, set()).add(bi)
        num_layers = len(layer_blocks) if layer_blocks else 2
        depths = tuple(len(layer_blocks.get(i, set())) for i in range(num_layers)) or (6, 6)

        dw_w = state_dict.get("layers.0.blocks.0.attn.a.2.lkb_reparam.weight")
        dw_size = int(dw_w.shape[2]) if dw_w is not None else 9
        fc1_w = state_dict.get("layers.0.blocks.0.mlp.fc1.weight")
        mlp_ratio = (int(round(fc1_w.shape[0] / embed_dim))
                     if (fc1_w is not None and embed_dim) else 4)
        model = cfsr(embed_dim=embed_dim, depths=depths, dw_size=dw_size,
                     mlp_ratio=mlp_ratio, upscale=scale)

    elif arch == "plainusr":
        # PlainUSR — n_feat pilote la branche (n_feat<=32 / >=80 / sinon) EXACTEMENT
        # comme __init__ (pas de devinette : c'est la même comparaison numérique).
        # Ne couvre que les checkpoints "training-time" (structure MBConv/PConv non
        # fusionnée) — un checkpoint exporté après switch_to_deploy() (Conv2d
        # fusionnés) n'a pas les clés "expand_conv"/"fea_conv" attendues et
        # échouera à la détection (limitation connue, documentée).
        from neosr.archs.plainusr_arch import plainusr
        head_w = state_dict.get("head.weight")
        n_feat = int(head_w.shape[0]) - 3 if head_w is not None else 64
        b1_w = state_dict.get("block1.body.0.expand_conv.weight")
        im0 = int(b1_w.shape[1]) if b1_w is not None else 64
        if n_feat <= 32:
            bm_w = state_dict.get("blockm.body.0.expand_conv.weight")
            im1 = int(bm_w.shape[1]) if bm_w is not None else 48
            im_feat = (im0, im1, 32)  # 3rd elem unused by this branch
            attn_feat = 16  # blockm uses LocalAttention(speed=True) — f not in weights
        else:
            b2_w = state_dict.get("block2.body.0.expand_conv.weight")
            im1 = int(b2_w.shape[1]) if b2_w is not None else 48
            bm_w = state_dict.get("blockm.body.0.expand_conv.weight")
            im2 = int(bm_w.shape[1]) if bm_w is not None else 32
            im_feat = (im0, im1, im2)
            attn_idx = 6 if n_feat >= 80 else 4
            af_w = state_dict.get(f"block1.body.{attn_idx}.body.0.weight")
            attn_feat = int(af_w.shape[0]) if af_w is not None else 16
        model = plainusr(n_feat=n_feat, im_feat=im_feat, attn_feat=attn_feat, scale=scale)

    elif arch == "asid":
        # ASID — window_size ne modifie AUCUNE shape de poids dans cette arch
        # (uniquement le pas d'attention par fenêtre au forward) → non déductible,
        # valeur par défaut (8) conservée. flash_attn : idem (bascule juste
        # scaled_dot_product_attention vs calcul manuel, pas de paramètre créé).
        from neosr.archs.asid_arch import asid
        in_w = state_dict.get("input.weight")
        num_feat = int(in_w.shape[0]) if in_w is not None else 48
        block_idxs = {
            int(m.group(1)) for k in state_dict
            if (m := re.match(r"^block0\.residual_layer\.(\d+)\.", k))
        }
        block_num = len(block_idxs) if block_idxs else 1
        d8 = "block7.esa.conv_f.weight" in state_dict
        pe = "block0.residual_layer.0.SLA1.attn.rel_pos_bias.weight" in state_dict
        bias = "input.bias" in state_dict
        model = asid(num_feat=num_feat, block_num=block_num, window_size=8,
                     scale=scale, flash_attn=True, d8=d8, pe=pe, bias=bias)

    elif arch == "grformer":
        # GRFormer — window_size = [8,32] par défaut, déduit EXACTEMENT de la shape
        # du buffer relative_position_bias_table (1, 2*Wh-1, 2*Ww-1, 2).
        from neosr.archs.grformer_arch import grformer
        cf_w = state_dict.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 60

        layer_depths: dict = {}
        for k in state_dict:
            m = re.match(r"^layers\.(\d+)\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_depths.setdefault(li, set()).add(bi)
        num_layers = len(layer_depths) if layer_depths else 4
        depths = tuple(len(layer_depths.get(i, set())) for i in range(num_layers)) or (6,) * 4

        num_heads = []
        for i in range(num_layers):
            ls_w = state_dict.get(f"layers.{i}.blocks.0.attn.logit_scale")
            num_heads.append(int(ls_w.shape[0]) if ls_w is not None else 3)
        num_heads = tuple(num_heads)

        rpb = state_dict.get("layers.0.blocks.0.attn.relative_position_bias_table")
        if rpb is not None:
            window_size = [int((rpb.shape[1] + 1) / 2), int((rpb.shape[2] + 1) / 2)]
        else:
            window_size = [8, 32]

        fc1_w = state_dict.get("layers.0.blocks.0.mlp.fc1.weight")
        mlp_ratio = (fc1_w.shape[0] / embed_dim) if (fc1_w is not None and embed_dim) else 2.0

        model = grformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                         window_size=window_size, mlp_ratio=mlp_ratio, upscale=scale)

    elif arch == "dct":
        # DCT — split_size déduit EXACTEMENT des buffers rpe_biases/relative_position_
        # index de la 1ere Window_Attention (voir dérivation dans le commentaire du
        # code appelant / rapport). shift_size n'affecte aucune shape (non déductible,
        # valeur par défaut conservée). reso déduit du buffer attn_mask_0/1 quand
        # présent (blocs shift), sinon défaut 64.
        from neosr.archs.dct_arch import dct
        cf_w = state_dict.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 80

        layer_depths: dict = {}
        for k in state_dict:
            m = re.match(r"^layers\.(\d+)\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_depths.setdefault(li, set()).add(bi)
        num_layers = len(layer_depths) if layer_depths else 1
        depth = [len(layer_depths.get(i, set())) for i in range(num_layers)] or [20]

        # num_heads : lla.dconv.weight.shape[0] = num_heads // 2 EXACTEMENT (une
        # seule division entière, cf. DCTB: Linear_Attention(num_heads=num_heads//2)
        # puis Linear_Attention.dconv a autant de groupes que ce num_heads//2).
        # Éviter la voie swsa.attns.pos3 (double division //2//2 — perd de
        # l'information pour les valeurs impaires en cours de route, ex. 6 -> 3 -> 1).
        num_heads = []
        for i in range(num_layers):
            dconv_w = state_dict.get(f"layers.{i}.blocks.0.lla.dconv.weight")
            num_heads.append(2 * int(dconv_w.shape[0]) if dconv_w is not None else 8)

        rpe = state_dict.get("layers.0.blocks.0.swsa.attns.0.rpe_biases")
        rpi = state_dict.get("layers.0.blocks.0.swsa.attns.0.relative_position_index")
        if rpe is not None and rpi is not None:
            p = int(rpi.shape[0])  # H_sp * W_sp
            n = int(rpe.shape[0])  # (2*H_sp-1) * (2*W_sp-1)
            s = (4 * p - n + 1) // 2  # H_sp + W_sp
            disc = s * s - 4 * p
            if disc >= 0:
                sq = int(round(disc ** 0.5))
                h_sp = (s - sq) // 2
                w_sp = (s + sq) // 2
                split_size = [max(1, h_sp), max(1, w_sp)] if h_sp > 0 else [2, 4]
            else:
                split_size = [2, 4]
        else:
            split_size = [2, 4]

        fc1_w = state_dict.get("layers.0.blocks.0.ffn.fc1.weight")
        expansion_factor = (fc1_w.shape[0] / embed_dim) if (fc1_w is not None and embed_dim) else 4.0

        resi_connection = "3conv" if "layers.0.conv.0.weight" in state_dict else "1conv"

        reso = 64
        for k, v in state_dict.items():
            if k.endswith("swsa.attn_mask_0") and hasattr(v, "shape"):
                nw = int(v.shape[0])
                area = nw * split_size[0] * split_size[1]
                r = int(round(area ** 0.5))
                if r * r == area:
                    reso = r
                break

        model = dct(img_size=reso, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                    split_size=split_size, expansion_factor=expansion_factor, upscale=scale,
                    resi_connection=resi_connection)

    elif arch == "flexnet":
        from neosr.archs.flexnet_arch import flexnet
        dim_w = state_dict.get("in_to_feat.weight")
        dim = int(dim_w.shape[0]) if dim_w is not None else 64
        ws_t = state_dict.get("window_size")
        window_size = int(ws_t.item()) if ws_t is not None else 8
        pipeline_type = "meta" if any(k.startswith("pipeline.enc0.") for k in state_dict) else "linear"

        if pipeline_type == "linear":
            stage_blocks: dict = {}
            for k in state_dict:
                m = re.match(r"^pipeline\.att\.(\d+)\.t_blocks\.(\d+)\.", k)
                if m:
                    si, bi = int(m.group(1)), int(m.group(2))
                    stage_blocks.setdefault(si, set()).add(bi)
        else:
            stage_blocks = {}
            for k in state_dict:
                m = re.match(r"^pipeline\.enc(\d)\.0\.t_blocks\.(\d+)\.", k)
                if m:
                    si, bi = int(m.group(1)), int(m.group(2))
                    stage_blocks.setdefault(si, set()).add(bi)
        num_stages = len(stage_blocks) if stage_blocks else (6 if pipeline_type == "linear" else 4)
        num_blocks = tuple(len(stage_blocks.get(i, set())) for i in range(num_stages)) \
            or ((6,) * 6 if pipeline_type == "linear" else (8,) * 4)

        ffn_key_w = None
        for k, v in state_dict.items():
            # a block working at the base dim (meta pipelines also hold 2x-8x dims,
            # and key order is alphabetical: dec0 came first, hidden_rate read 16)
            if (k.startswith("pipeline.") and k.endswith("ffn.key.weight")
                    and v.ndim == 2 and int(v.shape[1]) == dim):
                ffn_key_w = v
                break
        hidden_rate = int(round(ffn_key_w.shape[0] / dim)) if (ffn_key_w is not None and dim) else 4
        channel_norm = any(k.startswith("pipeline.") and k.endswith("ffn.key_norm.weight")
                            for k in state_dict)
        flash_attn = "flash_attn" in state_dict

        if "scale_factor" in state_dict:
            upsampler = "n+c"
        elif "to_img.offset.weight" in state_dict:
            upsampler = "dys"
        else:
            upsampler = "ps"

        model = flexnet(dim=dim, num_blocks=num_blocks, window_size=window_size,
                        hidden_rate=hidden_rate, channel_norm=channel_norm,
                        pipeline_type=pipeline_type, upsampler=upsampler,
                        flash_attn=flash_attn, scale=scale)

    elif arch == "hit_srf":
        # HiT-SRF — base_win_size/hier_win_ratios non déductibles des shapes (idem
        # hit_sir/hit_sng dans universal_runner.py, cf. son commentaire) : valeurs
        # par défaut conservées, strict=False en repli si un checkpoint a été
        # entraîné avec des valeurs non standard (mismatch possible sur les buffers
        # dépendant de la fenêtre hiérarchique — sans impact fonctionnel réel, le
        # module recalcule ses masques dynamiquement à l'inférence).
        from neosr.archs.hitsrf_arch import hit_srf
        cf_w = state_dict.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 60

        layer_depths: dict = {}
        for k in state_dict:
            m = re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_depths.setdefault(li, set()).add(bi)
        num_layers = len(layer_depths) if layer_depths else 4
        depths = tuple(len(layer_depths.get(i, set())) for i in range(num_layers)) or (6,) * 4

        num_heads = []
        for i in range(num_layers):
            head_w = state_dict.get(f"layers.{i}.residual_group.blocks.0.correlation.pos.pos3.2.weight")
            num_heads.append(int(head_w.shape[0]) if head_w is not None else 6)
        num_heads = tuple(num_heads)

        mlp_w = state_dict.get("layers.0.residual_group.blocks.0.mlp.fc1.weight")
        mlp_ratio = (mlp_w.shape[0] / embed_dim) if (mlp_w is not None and embed_dim) else 2.0

        resi_connection = "1conv" if "layers.0.conv.weight" in state_dict else "3conv"

        if "conv_up1.weight" in state_dict:
            upsampler = "nearest+conv"
        elif "conv_before_upsample.0.weight" in state_dict:
            upsampler = "pixelshuffle"
        elif "upsample.0.weight" in state_dict:
            upsampler = "pixelshuffledirect"
        else:
            upsampler = ""

        model = hit_srf(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, upscale=scale, upsampler=upsampler,
                        resi_connection=resi_connection)
        model.load_state_dict(state_dict, strict=False)
        return model

    elif arch == "hma":
        # HMA — window_size déduit EXACTEMENT du buffer relative_position_index_SA
        # (shape = window_size^2 x window_size^2). interval_size (grille GAB) ne
        # modifie aucune shape de poids → non déductible, défaut (4) conservé.
        from neosr.archs.hma_arch import hma
        cf_w = state_dict.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 60

        rpi_sa = state_dict.get("relative_position_index_SA")
        if rpi_sa is not None:
            window_size = max(1, int(round(rpi_sa.shape[0] ** 0.5)))
        else:
            window_size = 8

        layer_depths: dict = {}
        for k in state_dict:
            m = re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_depths.setdefault(li, set()).add(bi)
        num_layers = len(layer_depths) if layer_depths else 4
        # Le ModuleList "blocks" alterne FusedConv+FAB (i pair) / FAB seul (i impair)
        # → 1.5 entrées par itération de depth en moyenne (exact pour depth pair,
        # cas de tous les presets officiels — cf. commentaire de build_model()).
        depths = tuple(max(2, round(len(layer_depths.get(i, set())) * 2 / 3))
                        for i in range(num_layers)) or (6,) * 4

        num_heads = []
        for i in range(num_layers):
            rpb_w = state_dict.get(f"layers.{i}.residual_group.blocks.1.attn.relative_position_bias_table")
            num_heads.append(int(rpb_w.shape[1]) if rpb_w is not None else 6)
        num_heads = tuple(num_heads)

        mlp_w = state_dict.get("layers.0.residual_group.blocks.1.mlp.fc1.weight")
        mlp_ratio = (mlp_w.shape[0] / embed_dim) if (mlp_w is not None and embed_dim) else 2.0

        resi_connection = "1conv" if "layers.0.conv.weight" in state_dict else "identity"

        model = hma(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                   window_size=window_size, interval_size=4, mlp_ratio=mlp_ratio,
                   upscale=scale, resi_connection=resi_connection)

    elif arch == "srformer":
        # SRFormer (couvre aussi le preset srformer_medium — même classe, seuls les
        # hyperparamètres par défaut diffèrent, tous déduits ici des shapes réelles).
        from neosr.archs.srformer_arch import srformer
        cf_w = state_dict.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 60

        layer_depths: dict = {}
        for k in state_dict:
            m = re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_depths.setdefault(li, set()).add(bi)
        num_layers = len(layer_depths) if layer_depths else 4
        depths = tuple(len(layer_depths.get(i, set())) for i in range(num_layers)) or (6,) * 4

        # PSA.relative_position_bias_table shape = ((2*(ws//2)-1)^2, num_heads) —
        # num_heads directly from dim 1 (per layer); window_size back-solved from
        # layer 0 assuming a square window (srformer default 16, only value used
        # by both official presets — window_size is shared across all layers).
        rpb0_w = state_dict.get("layers.0.residual_group.blocks.0.attn.relative_position_bias_table")
        if rpb0_w is not None:
            permuted_ws = int(round((rpb0_w.shape[0] ** 0.5 + 1) / 2))
            window_size = max(2, permuted_ws * 2)
        else:
            window_size = 16
        num_heads = []
        for i in range(num_layers):
            rpb_w = state_dict.get(f"layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table")
            num_heads.append(int(rpb_w.shape[1]) if rpb_w is not None else 6)
        num_heads = tuple(num_heads)

        mlp_w = state_dict.get("layers.0.residual_group.blocks.0.mlp.fc1.weight")
        mlp_ratio = (mlp_w.shape[0] / embed_dim) if (mlp_w is not None and embed_dim) else 2.0

        resi_connection = "1conv" if "layers.0.conv.weight" in state_dict else "3conv"

        if "conv_up1.weight" in state_dict:
            upsampler = "nearest+conv"
        elif "conv_before_upsample.0.weight" in state_dict:
            upsampler = "pixelshuffle"
        elif "upsample.0.weight" in state_dict:
            upsampler = "pixelshuffledirect"
        else:
            upsampler = ""

        model = srformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                         window_size=window_size, mlp_ratio=mlp_ratio, upscale=scale,
                         upsampler=upsampler, resi_connection=resi_connection)

    else:
        raise ValueError(f"Arch non supportée dans ce runner: {arch}")

    state_dict = _drop_stale_buffers(model, state_dict)
    model.load_state_dict(state_dict, strict=not any(
        "attn_mask" in k for k in model.state_dict() if k not in state_dict))
    return model


def _drop_stale_buffers(model, state_dict: dict) -> dict:
    """Saved attention masks / position indexes are recomputed from the image size the
    model is built for: drop the ones whose shape differs (SRFormer saved at another
    img_size failed with "size mismatch for ...attn_mask")."""
    own = model.state_dict()
    return {k: v for k, v in state_dict.items()
            if not (("attn_mask" in k or "relative_position_index" in k) and k in own
                    and tuple(own[k].shape) != tuple(v.shape))}


def run(model_path: str, input_path: str, output_path: str, scale_hint: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NeoSR] Device: {device}", flush=True)

    state_dict = _load_state_dict(model_path)
    extra = _detect_extra_arch(state_dict)
    arch = extra or detect_arch(state_dict)
    if extra:
        model, scale = build_extra(extra, state_dict, scale_hint)
        print(f"[NeoSR] Arch détectée: {extra}, scale: {scale}x (chargement strict)", flush=True)
    elif arch == "msdan":
        # scale is NOT recoverable from msdan weights (see build_model()) —
        # the UI's scale_hint is mandatory, detect_scale() would just guess 1.
        scale = scale_hint if scale_hint > 0 else 4
        print(f"[NeoSR] msdan: scale non déductible des poids — scale_hint={scale_hint or '(absent, défaut 4)'}", flush=True)
    else:
        scale = scale_hint if scale_hint > 0 else detect_scale(state_dict)
    if not extra:
        print(f"[NeoSR] Arch détectée: {arch or 'inconnue'}, scale: {scale}x", flush=True)

    if arch is None:
        raise ValueError("Architecture inconnue — impossible de charger le modèle.")

    if not extra:
        model = build_model(arch, state_dict, scale)
    model = model.to(device).eval()
    print(f"[NeoSR] Modèle chargé: {arch}", flush=True)

    img = Image.open(input_path).convert("RGB")
    print(f"[NeoSR] Inference {img.width}x{img.height} -> {img.width*scale}x{img.height*scale}...", flush=True)

    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)

    import torch.nn.functional as F_pad
    _, _, H, W = img_t.shape
    ws = getattr(model, 'window_size', 1)
    if isinstance(ws, (list, tuple)):
        ws = ws[0]

    def _infer_padded(tile: "torch.Tensor") -> "torch.Tensor":
        """Run model on a single tile, padding to window_size multiple, then crop."""
        _, _, th, tw = tile.shape
        ph = (ws - th % ws) % ws
        pw = (ws - tw % ws) % ws
        if ph > 0 or pw > 0:
            tile = F_pad.pad(tile, (0, pw, 0, ph), mode='reflect')
        with torch.no_grad(), torch.autocast("cuda", enabled=_amp and tile.is_cuda):
            out = model(tile)
        return out[:, :, :th * scale, :tw * scale].float()

    from runner_save import infer_opts
    _io = infer_opts()
    _amp = _io["amp"]
    TILE = _io["tile"] if _io["tile"] else 512
    PAD  = _io["tile_pad"]
    # For large images with big window_size (drct, swinir, hat), go straight to tiling
    # to avoid OOM on the full-image attempt. A tile size chosen in the tab tiles always.
    FORCE_TILE = bool(_io["tile"]) or (ws >= 16 and H * W > 512 * 512)
    if FORCE_TILE:
        print(f"[NeoSR] Tiling {TILE}px (pad {PAD})" + ("" if _io["tile"] else f" — large image + window_size={ws}"), flush=True)
    # Try full-image inference first; fall back to tiling on OOM
    if not FORCE_TILE:
        try:
            out = _infer_padded(img_t)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
                raise
            torch.cuda.empty_cache()
            FORCE_TILE = True
            print(f"[NeoSR] OOM on full image — switching to {TILE}px tiling", flush=True)

    if FORCE_TILE:
        out = torch.zeros((1, 3, H * scale, W * scale), device=img_t.device)
        for y0 in range(0, H, TILE):
            for x0 in range(0, W, TILE):
                x1_src = max(0, x0 - PAD)
                y1_src = max(0, y0 - PAD)
                x2_src = min(W, x0 + TILE + PAD)
                y2_src = min(H, y0 + TILE + PAD)
                tile = img_t[:, :, y1_src:y2_src, x1_src:x2_src]
                tile_out = _infer_padded(tile)
                # Determine paste region (remove pad overlap)
                ox = (x0 - x1_src) * scale
                oy = (y0 - y1_src) * scale
                ow = min(TILE, W - x0) * scale
                oh = min(TILE, H - y0) * scale
                out[:, :, y0*scale:y0*scale+oh, x0*scale:x0*scale+ow] = \
                    tile_out[:, :, oy:oy+oh, ox:ox+ow]
                torch.cuda.empty_cache()

    out_np = out.squeeze(0).clamp(0, 1).cpu().float().numpy()
    out_np = np.transpose(out_np, (1, 2, 0))
    from runner_save import save_float
    save_float(out_np, output_path)
    print(f"[NeoSR] Sauvegardé: {output_path}", flush=True)

    del model, img_t, out


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Usage: neosr_general_runner.py <model_path> <input_path> <output_path> [scale_hint]")
        sys.exit(1)
    _model_path, _input_path, _output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    _scale_hint = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    sys.argv = [sys.argv[0]]
    try:
        run(_model_path, _input_path, _output_path, scale_hint=_scale_hint)
    except Exception as e:
        print(f"[NeoSR] ERREUR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
