"""
persistent_neosr_worker.py — Subprocess neosr longue durée (modèle chargé une fois).

Protocole JSON-lines via stdin/stdout :
  <- {"cmd": "init", "model": "<path>", "tile_size": 256, "tile_pad": 32, "use_amp": false}
  -> {"status": "ready", "arch": "ESC", "scale": 2}
  <- {"cmd": "infer", "input": "<path>", "output": "<path>"}
  -> {"status": "ok"}  |  {"status": "error", "msg": "..."}
  <- {"cmd": "quit"}
  -> (exit 0)

Archs supportées : ESC (historique) + ninasr/lmlt/eimn/drct (archs "générales" —
même détection/reconstruction que neosr_general_runner.py, adaptée au protocole
persistant : le modèle reste chargé en VRAM entre les commandes "infer").

Lancement :
  <neosr_venv_python> persistent_neosr_worker.py
"""
import sys
import os
import json
import re
import math
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

# Stub neosr packages to prevent net_opt() crash on import
import types as _t
for _n, _p in [
    ("neosr",       os.path.join(NEOSR_PATH, "neosr")),
    ("neosr.archs", os.path.join(NEOSR_PATH, "neosr", "archs")),
]:
    if _n not in sys.modules:
        _m = _t.ModuleType(_n)
        _m.__path__ = [_p]
        _m.__package__ = _n
        sys.modules[_n] = _m
if "neosr" in sys.modules and "neosr.archs" in sys.modules:
    sys.modules["neosr"].archs = sys.modules["neosr.archs"]
del _t, _n, _p, _m

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(json.dumps({"status": "fatal", "msg": f"Import error: {e}"}), flush=True)
    sys.exit(1)

# ── Global state ──────────────────────────────────────────────────────────────
_model = None
_scale: int = 1
_arch: str = "unknown"
_window_size: int = 1   # >1 pour drct (attention par fenêtre) → padding par tuile requis
_device = None
_tile_size: int = 0
_tile_pad: int = 32
_use_amp: bool = False
_last_hw: "tuple | None" = None   # (H, W) frame précédente — détecte changement de résolution

# ── Archs "générales" (ninasr, lmlt, eimn, drct) — même détection que
# neosr_general_runner.py (subprocess un-shot). net_opt() est appelé au niveau
# module par ces archs → doit être patché AVANT l'import, comme pour ESC.
_GENERAL_ARCH_SIGNATURES = {
    # ── v2.7.0 batch : 8 archs NeoSR supplémentaires — placées en premier (mêmes
    # raisons/collisions que neosr_general_runner.py::ARCH_SIGNATURES et
    # quick_upscale.py::ARCH_SIGNATURES — voir leurs commentaires).
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
    "srformer": ["layers.0.residual_group.blocks.0.attn.kv.weight",
                 "layers.0.residual_group.blocks.0.attn.q.weight"],

    "ninasr": ["head.0.bias", "body.0.body.2.body"],
    "lmlt":   ["feats.0.lhsb", "to_img.0.weight"],
    "eimn":   ["block1.0.attn.region", "block1.0.layer_scale_1"],
    "drct":   ["layers.0.swin1", "layers.0.adjust1"],
    # AetherNet (Phhofm, custom neosr arch — injected via ensure_custom_archs()).
    # Same signature as quick_upscale.py's ARCH_SIGNATURES / neosr_general_runner.py.
    "aethernet": ["conv_after_body", "stages", "conv_before_upsample"],
    # HASN — 6 fixed blocks.
    "hasn": ["block_1.dwconv.conv.weight", "block_6.dwconv.conv.weight", "esa.conv_f"],
    # MSDAN — see neosr_general_runner.py::build_model() for why scale MUST
    # come from scale_hint (payload), never guessed from the weights.
    "msdan": ["BSConv_first.pw.weight", "upsampler.BSConv_2.pw.weight", "upsampler.BSConv_3.pw.weight"],
    # CFSR — LargeKernelConv reparam (lkb_reparam) + ConvMod attn. First two sigs
    # are EXACT full keys (no substring-only match needed here, unlike
    # quick_upscale.py's detect_arch_from_state — this file's matcher is
    # substring-only already, so the full path also works as a literal match).
    "cfsr": ["layers.0.blocks.0.attn.a.2.lkb_reparam.weight", "layers.0.blocks.0.mlp.fc2.weight",
             "conv_after_body.weight"],
}


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


def _detect_general_arch(sd: dict) -> "str | None":
    keys_str = " ".join(sd.keys())
    for arch, sigs in _GENERAL_ARCH_SIGNATURES.items():
        if all(s in keys_str for s in sigs):
            return arch
    return None


def _detect_general_scale(sd: dict) -> int:
    # AetherNet registers its scale as an exact int64 buffer — read it directly.
    _scale_t = sd.get("scale_tensor")
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
        w = sd.get(probe)
        if w is not None:
            ch = w.shape[0]
            s = int(round(math.sqrt(ch / 3)))
            if s >= 1 and s * s * 3 == ch:
                return s
    # drct / swinir / hma-style pixel_shuffle upsamplers. A 4x upsampler is two x2
    # stages (upsample.0 / upsample.2): the stages multiply. The first stage alone was
    # returned before, so 4x DRCT / HMA / SRFormer models were rebuilt as 2x.
    _stages = []
    for k, v in sd.items():
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
    return 4


def _build_general_model(arch: str, sd: dict, scale: int):
    """Instancie ninasr/lmlt/eimn/drct avec les vrais kwargs déduits du state_dict.
    Miroir de neosr_general_runner.py::build_model() (mêmes archs, même heuristique)."""
    import neosr.archs.arch_util as _au
    _au.net_opt = lambda: (scale, True)

    if arch == "ninasr":
        from neosr.archs.ninasr_arch import ninasr
        return ninasr(scale=scale), 1

    if arch == "lmlt":
        from neosr.archs.lmlt_arch import lmlt
        # dim / n_blocks from the weights (lmlt_tiny 36, lmlt_large 84, base 60)
        _ti = sd.get("to_img.0.weight")
        _nb = len({k.split(".")[1] for k in sd if re.match(r"^feats\.\d+\.", k)}) or 8
        return lmlt(upscaling_factor=scale, dim=int(_ti.shape[1]) if _ti is not None else 60, n_blocks=_nb), 1

    if arch == "eimn":
        from neosr.archs.eimn_arch import eimn
        head_w = sd.get("head.0.weight")
        embed_dims = int(head_w.shape[0]) if head_w is not None else 64
        num_stages = sum(
            1 for k in sd if k.startswith("block") and k.endswith(".region.weight")
        )
        return eimn(
            embed_dims=embed_dims, scale=scale,
            num_stages=num_stages if num_stages > 0 else 16,
        ), 1

    if arch == "drct":
        from neosr.archs.drct_arch import drct
        # embed_dim and layer count from the weights: covers drct_s (48, 4 x 2 blocks),
        # drct (180, 6 layers), and traiNNer-redux drct_l / drct_xl (12 / 14 layers),
        # which the neosr defaults (6 layers) failed to load.
        cf_w = sd.get("conv_first.weight")
        _dim = int(cf_w.shape[0]) if cf_w is not None else 180
        _n = len({k.split(".")[1] for k in sd if re.match(r"^layers\.\d+\.", k)}) or 6
        _d = 2 if _dim <= 60 else 6
        model = drct(upscale=scale, embed_dim=_dim, depths=(_d,) * _n, num_heads=(6,) * _n)
        return model, int(getattr(model, "window_size", 1))

    if arch == "aethernet":
        # Mirror of neosr_general_runner.py's aethernet branch — see its comments.
        from neosr.archs.aether_core import AetherNet
        cf_w = sd.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 64

        stage_blocks: dict = {}
        for k in sd:
            m = re.match(r"^stages\.(\d+)\.(\d+)\.", k)
            if m:
                si, bi = int(m.group(1)), int(m.group(2))
                stage_blocks.setdefault(si, set()).add(bi)
        num_stages = len(stage_blocks) if stage_blocks else 4
        depths = tuple(len(stage_blocks.get(i, set())) for i in range(num_stages)) or (4, 4, 4, 4)

        lk_w = sd.get("stages.0.0.conv.lk_conv.weight")
        fused_init = lk_w is None
        if fused_init:
            lk_w = sd.get("stages.0.0.conv.fused_conv.weight")
        lk_kernel = int(lk_w.shape[2]) if lk_w is not None else 13
        sk_w = sd.get("stages.0.0.conv.sk_conv.weight")
        sk_kernel = int(sk_w.shape[2]) if sk_w is not None else 5

        gate_w = sd.get("stages.0.0.ffn.conv_gate.weight")
        mlp_ratio = (float(gate_w.shape[0]) / embed_dim) if (gate_w is not None and embed_dim) else 1.5

        use_channel_attn = any(k.startswith("stages.0.0.channel_attn.") for k in sd)
        use_spatial_attn = any(k.startswith("stages.0.0.spatial_attn.") for k in sd)
        norm_type = "deployment" if "stages.0.0.norm.running_mean" in sd else "layernorm"
        res_scale = 0.2 if embed_dim <= 64 else 0.1

        model = AetherNet(
            embed_dim=embed_dim, depths=depths, lk_kernel=lk_kernel, sk_kernel=sk_kernel,
            fused_init=fused_init, mlp_ratio=mlp_ratio, use_channel_attn=use_channel_attn,
            use_spatial_attn=use_spatial_attn, norm_type=norm_type, scale=scale,
            res_scale=res_scale,
        )
        return model, 1  # no window attention → no per-tile padding needed

    if arch == "hasn":
        from neosr.archs.hasn_arch import hasn
        w = sd.get("conv_1.weight")
        feature_channels = int(w.shape[0]) if w is not None else 52
        return hasn(feature_channels=feature_channels, upscale=scale), 1

    if arch == "msdan":
        # Mirror of neosr_general_runner.py's msdan branch — see its comments.
        # `scale` here is already the caller's scale_hint (cmd_init below
        # forces this for msdan, since detect_general_scale() can't recover
        # it from the weights).
        from neosr.archs.msdan_arch import msdan
        w = sd.get("BSConv_first.pw.weight")
        channels = int(w.shape[0]) if w is not None else 48
        idxs = {
            int(m.group(1)) for k in sd
            if (m := re.match(r"^layers\.(\d+)\.EBFB\.", k))
        }
        num_dfeb = len(idxs) if idxs else 8
        msdan_scale = scale if scale in (2, 3, 4) else 4
        return msdan(channels=channels, num_DFEB=num_dfeb, upscale_factor=msdan_scale), 1

    if arch == "cfsr":
        # Mirror of neosr_general_runner.py's cfsr branch — see its comments
        # (merge_all() runs unconditionally in __init__, no unmerged/merged
        # state ambiguity to handle).
        from neosr.archs.cfsr_arch import cfsr
        cf_w = sd.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 48

        layer_blocks: dict = {}
        for k in sd:
            m = re.match(r"^layers\.(\d+)\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_blocks.setdefault(li, set()).add(bi)
        num_layers = len(layer_blocks) if layer_blocks else 2
        depths = tuple(len(layer_blocks.get(i, set())) for i in range(num_layers)) or (6, 6)

        dw_w = sd.get("layers.0.blocks.0.attn.a.2.lkb_reparam.weight")
        dw_size = int(dw_w.shape[2]) if dw_w is not None else 9
        fc1_w = sd.get("layers.0.blocks.0.mlp.fc1.weight")
        mlp_ratio = (int(round(fc1_w.shape[0] / embed_dim))
                     if (fc1_w is not None and embed_dim) else 4)
        return cfsr(embed_dim=embed_dim, depths=depths, dw_size=dw_size,
                    mlp_ratio=mlp_ratio, upscale=scale), 1

    if arch == "plainusr":
        # Mirror of neosr_general_runner.py's plainusr branch — see its comments.
        from neosr.archs.plainusr_arch import plainusr
        head_w = sd.get("head.weight")
        n_feat = int(head_w.shape[0]) - 3 if head_w is not None else 64
        b1_w = sd.get("block1.body.0.expand_conv.weight")
        im0 = int(b1_w.shape[1]) if b1_w is not None else 64
        if n_feat <= 32:
            bm_w = sd.get("blockm.body.0.expand_conv.weight")
            im1 = int(bm_w.shape[1]) if bm_w is not None else 48
            im_feat = (im0, im1, 32)
            attn_feat = 16
        else:
            b2_w = sd.get("block2.body.0.expand_conv.weight")
            im1 = int(b2_w.shape[1]) if b2_w is not None else 48
            bm_w = sd.get("blockm.body.0.expand_conv.weight")
            im2 = int(bm_w.shape[1]) if bm_w is not None else 32
            im_feat = (im0, im1, im2)
            attn_idx = 6 if n_feat >= 80 else 4
            af_w = sd.get(f"block1.body.{attn_idx}.body.0.weight")
            attn_feat = int(af_w.shape[0]) if af_w is not None else 16
        # PlainUSR n'a pas de padding externe requis (fast_forward opère directement
        # sur x, aucun window/patch — pas de contrainte de multiple de résolution).
        return plainusr(n_feat=n_feat, im_feat=im_feat, attn_feat=attn_feat, scale=scale), 1

    if arch == "asid":
        # Mirror of neosr_general_runner.py's asid branch — see its comments.
        # check_image_size() interne à asid.forward() gère déjà le padding →
        # pas besoin de padding externe (ws=1).
        from neosr.archs.asid_arch import asid
        in_w = sd.get("input.weight")
        num_feat = int(in_w.shape[0]) if in_w is not None else 48
        block_idxs = {
            int(m.group(1)) for k in sd
            if (m := re.match(r"^block0\.residual_layer\.(\d+)\.", k))
        }
        block_num = len(block_idxs) if block_idxs else 1
        d8 = "block7.esa.conv_f.weight" in sd
        pe = "block0.residual_layer.0.SLA1.attn.rel_pos_bias.weight" in sd
        bias = "input.bias" in sd
        return asid(num_feat=num_feat, block_num=block_num, window_size=8,
                    scale=scale, flash_attn=True, d8=d8, pe=pe, bias=bias), 1

    if arch == "grformer":
        # Mirror of neosr_general_runner.py's grformer branch — see its comments.
        # check_image_size() interne gère déjà le padding (ws=1).
        from neosr.archs.grformer_arch import grformer
        cf_w = sd.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 60

        layer_depths: dict = {}
        for k in sd:
            m = re.match(r"^layers\.(\d+)\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_depths.setdefault(li, set()).add(bi)
        num_layers = len(layer_depths) if layer_depths else 4
        depths = tuple(len(layer_depths.get(i, set())) for i in range(num_layers)) or (6,) * 4

        num_heads = []
        for i in range(num_layers):
            ls_w = sd.get(f"layers.{i}.blocks.0.attn.logit_scale")
            num_heads.append(int(ls_w.shape[0]) if ls_w is not None else 3)
        num_heads = tuple(num_heads)

        rpb = sd.get("layers.0.blocks.0.attn.relative_position_bias_table")
        if rpb is not None:
            window_size = [int((rpb.shape[1] + 1) / 2), int((rpb.shape[2] + 1) / 2)]
        else:
            window_size = [8, 32]

        fc1_w = sd.get("layers.0.blocks.0.mlp.fc1.weight")
        mlp_ratio = (fc1_w.shape[0] / embed_dim) if (fc1_w is not None and embed_dim) else 2.0

        return grformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                        window_size=window_size, mlp_ratio=mlp_ratio, upscale=scale), 1

    if arch == "dct":
        # Mirror of neosr_general_runner.py's dct branch — see its comments.
        # Adaptive_Window_Attention.forward() pad interne (F.pad par appel) — ws=1.
        from neosr.archs.dct_arch import dct
        cf_w = sd.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 80

        layer_depths: dict = {}
        for k in sd:
            m = re.match(r"^layers\.(\d+)\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_depths.setdefault(li, set()).add(bi)
        num_layers = len(layer_depths) if layer_depths else 1
        depth = [len(layer_depths.get(i, set())) for i in range(num_layers)] or [20]

        num_heads = []
        for i in range(num_layers):
            dconv_w = sd.get(f"layers.{i}.blocks.0.lla.dconv.weight")
            num_heads.append(2 * int(dconv_w.shape[0]) if dconv_w is not None else 8)

        rpe = sd.get("layers.0.blocks.0.swsa.attns.0.rpe_biases")
        rpi = sd.get("layers.0.blocks.0.swsa.attns.0.relative_position_index")
        if rpe is not None and rpi is not None:
            p = int(rpi.shape[0])
            n = int(rpe.shape[0])
            s = (4 * p - n + 1) // 2
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

        fc1_w = sd.get("layers.0.blocks.0.ffn.fc1.weight")
        expansion_factor = (fc1_w.shape[0] / embed_dim) if (fc1_w is not None and embed_dim) else 4.0

        resi_connection = "3conv" if "layers.0.conv.0.weight" in sd else "1conv"

        reso = 64
        for k, v in sd.items():
            if k.endswith("swsa.attn_mask_0") and hasattr(v, "shape"):
                nw = int(v.shape[0])
                area = nw * split_size[0] * split_size[1]
                r = int(round(area ** 0.5))
                if r * r == area:
                    reso = r
                break

        return dct(img_size=reso, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                   split_size=split_size, expansion_factor=expansion_factor, upscale=scale,
                   resi_connection=resi_connection), 1

    if arch == "flexnet":
        # Mirror of neosr_general_runner.py's flexnet branch — see its comments.
        # check_img_size() interne gère déjà le padding (ws=1).
        from neosr.archs.flexnet_arch import flexnet
        dim_w = sd.get("in_to_feat.weight")
        dim = int(dim_w.shape[0]) if dim_w is not None else 64
        ws_t = sd.get("window_size")
        window_size = int(ws_t.item()) if ws_t is not None else 8
        pipeline_type = "meta" if any(k.startswith("pipeline.enc0.") for k in sd) else "linear"

        if pipeline_type == "linear":
            stage_blocks: dict = {}
            for k in sd:
                m = re.match(r"^pipeline\.att\.(\d+)\.t_blocks\.(\d+)\.", k)
                if m:
                    si, bi = int(m.group(1)), int(m.group(2))
                    stage_blocks.setdefault(si, set()).add(bi)
        else:
            stage_blocks = {}
            for k in sd:
                m = re.match(r"^pipeline\.enc(\d)\.0\.t_blocks\.(\d+)\.", k)
                if m:
                    si, bi = int(m.group(1)), int(m.group(2))
                    stage_blocks.setdefault(si, set()).add(bi)
        num_stages = len(stage_blocks) if stage_blocks else (6 if pipeline_type == "linear" else 4)
        num_blocks = tuple(len(stage_blocks.get(i, set())) for i in range(num_stages)) \
            or ((6,) * 6 if pipeline_type == "linear" else (8,) * 4)

        ffn_key_w = None
        for k, v in sd.items():
            # a block working at the base dim (meta pipelines also hold 2x-8x dims,
            # and key order is alphabetical: dec0 came first, hidden_rate read 16)
            if (k.startswith("pipeline.") and k.endswith("ffn.key.weight")
                    and v.ndim == 2 and int(v.shape[1]) == dim):
                ffn_key_w = v
                break
        hidden_rate = int(round(ffn_key_w.shape[0] / dim)) if (ffn_key_w is not None and dim) else 4
        channel_norm = any(k.startswith("pipeline.") and k.endswith("ffn.key_norm.weight")
                            for k in sd)
        flash_attn = "flash_attn" in sd

        if "scale_factor" in sd:
            upsampler = "n+c"
        elif "to_img.offset.weight" in sd:
            upsampler = "dys"
        else:
            upsampler = "ps"

        return flexnet(dim=dim, num_blocks=num_blocks, window_size=window_size,
                       hidden_rate=hidden_rate, channel_norm=channel_norm,
                       pipeline_type=pipeline_type, upsampler=upsampler,
                       flash_attn=flash_attn, scale=scale), 1

    if arch == "hit_srf":
        # Mirror of neosr_general_runner.py's hit_srf branch — see its comments
        # (base_win_size/hier_win_ratios non déductibles, strict=False en repli
        # géré par l'appelant cmd_init ci-dessous). check_image_size() interne à
        # HierarchicalTransformerBlock gère déjà le padding (ws=1).
        from neosr.archs.hitsrf_arch import hit_srf
        cf_w = sd.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 60

        layer_depths: dict = {}
        for k in sd:
            m = re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_depths.setdefault(li, set()).add(bi)
        num_layers = len(layer_depths) if layer_depths else 4
        depths = tuple(len(layer_depths.get(i, set())) for i in range(num_layers)) or (6,) * 4

        num_heads = []
        for i in range(num_layers):
            head_w = sd.get(f"layers.{i}.residual_group.blocks.0.correlation.pos.pos3.2.weight")
            num_heads.append(int(head_w.shape[0]) if head_w is not None else 6)
        num_heads = tuple(num_heads)

        mlp_w = sd.get("layers.0.residual_group.blocks.0.mlp.fc1.weight")
        mlp_ratio = (mlp_w.shape[0] / embed_dim) if (mlp_w is not None and embed_dim) else 2.0

        resi_connection = "1conv" if "layers.0.conv.weight" in sd else "3conv"

        if "conv_up1.weight" in sd:
            upsampler = "nearest+conv"
        elif "conv_before_upsample.0.weight" in sd:
            upsampler = "pixelshuffle"
        elif "upsample.0.weight" in sd:
            upsampler = "pixelshuffledirect"
        else:
            upsampler = ""

        model = hit_srf(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, upscale=scale, upsampler=upsampler,
                        resi_connection=resi_connection)
        return model, 1  # strict=False handled by caller (base_win_size non déductible)

    if arch == "hma":
        # Mirror of neosr_general_runner.py's hma branch — see its comments.
        # HMA n'a AUCUN padding interne (pas de check_image_size) → ws=window_size
        # réel requis pour que le worker persistant pad/tile correctement.
        from neosr.archs.hma_arch import hma
        cf_w = sd.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 60

        rpi_sa = sd.get("relative_position_index_SA")
        window_size = max(1, int(round(rpi_sa.shape[0] ** 0.5))) if rpi_sa is not None else 8

        layer_depths: dict = {}
        for k in sd:
            m = re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_depths.setdefault(li, set()).add(bi)
        num_layers = len(layer_depths) if layer_depths else 4
        depths = tuple(max(2, round(len(layer_depths.get(i, set())) * 2 / 3))
                        for i in range(num_layers)) or (6,) * 4

        num_heads = []
        for i in range(num_layers):
            rpb_w = sd.get(f"layers.{i}.residual_group.blocks.1.attn.relative_position_bias_table")
            num_heads.append(int(rpb_w.shape[1]) if rpb_w is not None else 6)
        num_heads = tuple(num_heads)

        mlp_w = sd.get("layers.0.residual_group.blocks.1.mlp.fc1.weight")
        mlp_ratio = (mlp_w.shape[0] / embed_dim) if (mlp_w is not None and embed_dim) else 2.0

        resi_connection = "1conv" if "layers.0.conv.weight" in sd else "identity"

        model = hma(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                   window_size=window_size, interval_size=4, mlp_ratio=mlp_ratio,
                   upscale=scale, resi_connection=resi_connection)
        return model, window_size

    if arch == "srformer":
        # Mirror of neosr_general_runner.py's srformer branch — see its comments
        # (covers srformer_medium preset too). check_image_size() interne gère
        # déjà le padding (ws=1).
        from neosr.archs.srformer_arch import srformer
        cf_w = sd.get("conv_first.weight")
        embed_dim = int(cf_w.shape[0]) if cf_w is not None else 60

        layer_depths: dict = {}
        for k in sd:
            m = re.match(r"^layers\.(\d+)\.residual_group\.blocks\.(\d+)\.", k)
            if m:
                li, bi = int(m.group(1)), int(m.group(2))
                layer_depths.setdefault(li, set()).add(bi)
        num_layers = len(layer_depths) if layer_depths else 4
        depths = tuple(len(layer_depths.get(i, set())) for i in range(num_layers)) or (6,) * 4

        rpb0_w = sd.get("layers.0.residual_group.blocks.0.attn.relative_position_bias_table")
        if rpb0_w is not None:
            permuted_ws = int(round((rpb0_w.shape[0] ** 0.5 + 1) / 2))
            window_size = max(2, permuted_ws * 2)
        else:
            window_size = 16
        num_heads = []
        for i in range(num_layers):
            rpb_w = sd.get(f"layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table")
            num_heads.append(int(rpb_w.shape[1]) if rpb_w is not None else 6)
        num_heads = tuple(num_heads)

        mlp_w = sd.get("layers.0.residual_group.blocks.0.mlp.fc1.weight")
        mlp_ratio = (mlp_w.shape[0] / embed_dim) if (mlp_w is not None and embed_dim) else 2.0

        resi_connection = "1conv" if "layers.0.conv.weight" in sd else "3conv"

        if "conv_up1.weight" in sd:
            upsampler = "nearest+conv"
        elif "conv_before_upsample.0.weight" in sd:
            upsampler = "pixelshuffle"
        elif "upsample.0.weight" in sd:
            upsampler = "pixelshuffledirect"
        else:
            upsampler = ""

        return srformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                        window_size=window_size, mlp_ratio=mlp_ratio, upscale=scale,
                        upsampler=upsampler, resi_connection=resi_connection), 1

    raise ValueError(f"Arch non supportée dans ce worker: {arch}")


# stdout carries the JSON protocol only: any other print() (an arch constructor, e.g.
# DocRes "Enabling Auto-Coordinate Injection") goes to stderr instead of breaking it.
_PROTO_OUT = sys.stdout
sys.stdout = sys.stderr


def _emit(obj: dict) -> None:
    _PROTO_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTO_OUT.flush()


def _drop_stale_buffers(model, state_dict: dict) -> dict:
    """Saved attention masks / position indexes are recomputed from the image size the
    model is built for: drop the ones whose shape differs (SRFormer saved at another
    img_size failed with "size mismatch for ...attn_mask")."""
    own = model.state_dict()
    return {k: v for k, v in state_dict.items()
            if not (("attn_mask" in k or "relative_position_index" in k) and k in own
                    and tuple(own[k].shape) != tuple(v.shape))}



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

def _load_state_dict(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params_ema", "params_g", "params", "model", "state_dict"):
        if key in ck:
            return ck[key]
    if any(k.endswith(".weight") for k in ck.keys()):
        return ck
    return ck


def _infer_esc_params(sd: dict) -> dict:
    proj_w = sd.get("proj.weight")
    dim = proj_w.shape[0] if proj_w is not None else 64

    is_fp = "lk_channel" in sd
    pdim_t = sd.get("lk_channel") if is_fp else sd.get("plk_filter")
    pdim = pdim_t.shape[0] if pdim_t is not None else 16

    block_indices = set()
    for k in sd:
        m = re.match(r"^blocks\.(\d+)\.", k)
        if m:
            block_indices.add(int(m.group(1)))
    n_blocks = len(block_indices) if block_indices else 5

    realsr = "skip.0.weight" in sd
    use_dysample = "to_img.offset.weight" in sd

    scale = 1
    if use_dysample:
        offset_ch = sd["to_img.offset.weight"].shape[0]
        scale_sq = offset_ch / 8.0
        s = int(round(math.sqrt(scale_sq)))
        if s >= 1 and s * s == int(scale_sq):
            scale = s
    elif is_fp:
        to_img_w = sd.get("to_img.weight")
        if to_img_w is not None:
            out_ch = to_img_w.shape[0]
            scale_sq = out_ch / 3.0
            s = int(round(math.sqrt(scale_sq)))
            if s >= 1 and s * s == int(scale_sq):
                scale = s

    return {
        "dim": dim, "pdim": pdim, "n_blocks": n_blocks,
        "is_fp": is_fp, "realsr": realsr,
        "use_dysample": use_dysample, "upscaling_factor": scale,
    }


def _call_model_ws(model, x: "torch.Tensor", scale: int, ws: int) -> "torch.Tensor":
    """Run model on x, padding to a multiple of window_size when ws>1
    (drct: forward() does NOT pad internally, unlike lmlt's check_img_size).
    Crops the result back to the exact expected h*scale, w*scale."""
    import torch.nn.functional as _F
    _, _, th, tw = x.shape
    if ws > 1:
        ph = (ws - th % ws) % ws
        pw = (ws - tw % ws) % ws
        if ph > 0 or pw > 0:
            x = _F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.inference_mode():
        out = model(x)
    return out[:, :, :th * scale, :tw * scale]


def _tile_infer(model, img_t, tile: int, pad: int, scale: int, ws: int = 1):
    b, c, h, w = img_t.shape
    if tile == 0 or (h <= tile and w <= tile):
        return _call_model_ws(model, img_t, scale, ws)
    out = torch.zeros(b, c, h * scale, w * scale, device=img_t.device, dtype=img_t.dtype)
    tiles_x = math.ceil(w / tile)
    tiles_y = math.ceil(h / tile)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x1 = max(tx * tile - pad, 0)
            y1 = max(ty * tile - pad, 0)
            x2 = min((tx + 1) * tile + pad, w)
            y2 = min((ty + 1) * tile + pad, h)
            tile_out = _call_model_ws(model, img_t[:, :, y1:y2, x1:x2], scale, ws)
            ox1 = (tx * tile - x1) * scale
            oy1 = (ty * tile - y1) * scale
            out_x1 = tx * tile * scale
            out_y1 = ty * tile * scale
            out_x2 = min(out_x1 + tile * scale, w * scale)
            out_y2 = min(out_y1 + tile * scale, h * scale)
            out[:, :, out_y1:out_y2, out_x1:out_x2] = tile_out[
                :, :, oy1:oy1 + (out_y2 - out_y1), ox1:ox1 + (out_x2 - out_x1)]
    return out


def cmd_init(payload: dict) -> None:
    global _model, _scale, _arch, _window_size, _device, _tile_size, _tile_pad, _use_amp, _last_hw
    _last_hw = None
    _window_size = 1
    model_path = payload["model"]
    _tile_size = int(payload.get("tile_size", 256))
    _tile_pad  = int(payload.get("tile_pad", 32))
    _use_amp   = bool(payload.get("use_amp", False))
    _scale_hint = int(payload.get("scale_hint", 0))

    if not os.path.isfile(model_path):
        _emit({"status": "error", "msg": f"Modèle introuvable : {model_path}"})
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # neosr force cudnn.benchmark=True à l'import. Pour l'inférence multi-résolution
    # c'est nocif : chaque nouvelle taille d'image relance un autotune cuDNN complet
    # (workspace transitoire énorme → spike VRAM + stall), puis fragmente l'allocateur
    # et ralentit tout le reste du batch. On le désactive pour rester stable.
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        sd = _load_state_dict(model_path)
    except Exception as e:
        _emit({"status": "error", "msg": f"Chargement state_dict : {e}"})
        return

    # ── ESC (arch historique de ce worker) — inchangé ──────────────────────────
    if "plk_filter" in sd:
        try:
            params = _infer_esc_params(sd)
            _scale = params["upscaling_factor"]
        except Exception as e:
            _emit({"status": "error", "msg": f"Détection paramètres ESC : {e}"})
            return

        try:
            import neosr.archs.arch_util as _au
            _au.net_opt = lambda: (_scale, True)
            from neosr.archs.esc_arch import esc
            _model = esc(
                dim=params["dim"],
                pdim=params["pdim"],
                n_blocks=params["n_blocks"],
                upscaling_factor=_scale,
                is_fp=params["is_fp"],
                use_dysample=params["use_dysample"],
                realsr=params["realsr"],
                attn_type="sdpa",
            )
            _model.load_state_dict(sd, strict=True)
            _model = _model.to(_device).eval()
        except Exception as e:
            _emit({"status": "error", "msg": f"Chargement modèle ESC : {e}\n{traceback.format_exc()}"})
            return

        _arch = "ESC"
        _emit({"status": "ready", "arch": _arch, "scale": _scale})
        return

    # ── NeoSR archs rebuilt from their defaults (catanet / krgn / man / mosrv2) ──
    _extra = _detect_extra_arch(sd)
    if _extra is not None:
        try:
            _m, _scale = build_extra(_extra, sd, _scale_hint)
            _model = _m.to(_device).eval()
        except Exception as e:
            _emit({"status": "error", "msg": f"Chargement modèle {_extra} : {e}"})
            return
        _arch = _extra
        _emit({"status": "ready", "arch": _arch, "scale": _scale})
        return

    # ── Archs "générales" (ninasr, lmlt, eimn, drct) ───────────────────────────
    general_arch = _detect_general_arch(sd)
    if general_arch is not None:
        if general_arch == "msdan":
            # scale is NOT recoverable from msdan weights — see build_model()
            # comments in neosr_general_runner.py for the full diagnosis.
            _scale = _scale_hint if _scale_hint > 0 else 4
        else:
            _scale = _scale_hint if _scale_hint > 0 else _detect_general_scale(sd)
        try:
            _model, ws = _build_general_model(general_arch, sd, _scale)
            # hit_srf : base_win_size/hier_win_ratios non déductibles des shapes
            # (cf. _build_general_model) — strict=False en repli, sans impact
            # fonctionnel réel (masques hiérarchiques recalculés dynamiquement).
            _strict = general_arch != "hit_srf"
            _sd2 = _drop_stale_buffers(_model, sd)
            if len(_sd2) != len(sd):
                _strict = False  # recomputed attention masks only
            _model.load_state_dict(_sd2, strict=_strict)
            _model = _model.to(_device).eval()
            _window_size = ws
        except Exception as e:
            _emit({"status": "error",
                   "msg": f"Chargement modèle {general_arch} : {e}\n{traceback.format_exc()}"})
            return

        _arch = general_arch
        _emit({"status": "ready", "arch": _arch, "scale": _scale})
        return

    # ── Arch inconnue de ce worker persistant ──────────────────────────────────
    _emit({
        "status": "error",
        "msg": (
            "Architecture NeoSR non reconnue par le subprocess persistant "
            "(ESC/ninasr/lmlt/eimn/drct uniquement). "
            "Décochez 'Subprocess persistant' pour ce modèle."
        ),
    })


def cmd_infer(payload: dict) -> None:
    global _last_hw
    if _model is None:
        _emit({"status": "error", "msg": "Modèle non initialisé."})
        return

    input_path  = payload["input"]
    output_path = payload["output"]

    try:
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_t  = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(_device)

        # Changement de résolution → défragmente l'allocateur avant l'inférence
        # (évite buildup VRAM réservée + thrashing cudaMalloc sur batch mixte).
        cur_hw = (img_t.shape[2], img_t.shape[3])
        if _device.type == "cuda" and _last_hw is not None and cur_hw != _last_hw:
            torch.cuda.empty_cache()
        _last_hw = cur_hw

        if _use_amp and _device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                out_t = _tile_infer(_model, img_t, _tile_size, _tile_pad, _scale, _window_size)
        else:
            out_t = _tile_infer(_model, img_t, _tile_size, _tile_pad, _scale, _window_size)

        out = out_t.squeeze(0).clamp(0, 1).cpu().float().numpy()
        _write_output(np.transpose(out, (1, 2, 0)), output_path, payload.get("save"))
        _emit({"status": "ok"})
    except Exception as e:
        _emit({"status": "error", "msg": f"{e}\n{traceback.format_exc()}"})



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


def main_loop() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            _emit({"status": "error", "msg": f"JSON invalide : {e}"})
            continue

        cmd = payload.get("cmd", "")
        if cmd == "init":
            cmd_init(payload)
        elif cmd == "infer":
            cmd_infer(payload)
        elif cmd == "quit":
            sys.exit(0)
        else:
            _emit({"status": "error", "msg": f"Commande inconnue : {cmd!r}"})


if __name__ == "__main__":
    main_loop()
