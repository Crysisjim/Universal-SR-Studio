"""
arch_infer.py — rebuild HEART / NERVE (traiNNer-redux master, Phhofm) from a bare
state_dict. spandrel doesn't know these archs, so every inference tool (Quick Upscale
in-process, the portable-mode runner, the persistent worker, the ONNX converter) needs
the exact constructor kwargs, read back from the tensor shapes.

Pure-Python on purpose (only reads `.shape`): this file is shipped loose next to the
venv runner scripts and imported by them as well as by the app, so the inference rules
live in one place.

Not recoverable from the weights, left at the official defaults:
  - HEART window_size (32): the RIB bias works on normalized window coordinates, so the
    weights load at any window size — a model trained with a non-default window_size
    would load fine but attend over the wrong window.
"""
from __future__ import annotations

import math
import re


#: Archs that take a single image. MoTSR is a video model (5 frames + recurrence) and
#: only runs through Quick Upscale's sequence pipeline.
IMAGE_ARCHS = {"heart", "nerve", "wiresr", "fdat_rd", "dat_iln", "scunet_aaf6aa", "wsr", "dual",
               "spanf3"}


#: Video archs built here for Quick Upscale's sequence pipeline.
SEQUENCE_ARCHS = {"motsr", "tfdat", "tfdatv2"}


def detect(state_dict) -> str | None:
    """Arch name if the key layout matches one handled here, else None."""
    keys = state_dict.keys()
    # FDAT-RD before anything FDAT-like: its checkpoints also satisfy FDAT's detection.
    if ("groups.0.blocks.0.attn.rel_pos_bias" in state_dict
            and "groups.0.blocks.2.attn.dictionary" in state_dict
            and "upsampler.MetaUpsample" in state_dict
            and ("conv_first.weight" in state_dict or "conv_first.1.weight" in state_dict)):
        return "fdat_rd"
    # WSR (Rift73): wide residual trunk + LR/4 context stream (+ Halo attention).
    if ("stem.sk.weight" in state_dict and "context_stem.sk.weight" in state_dict
            and "reconstruct.weight" in state_dict and "gates.0.project.weight" in state_dict):
        return "wsr"
    # DUAL (Rift73): dual-context transformer, two-path stem.
    if ("stem.direct.expand.weight" in state_dict and "stem.sequential.reduce.weight" in state_dict
            and any(k.startswith("groups.0.layers.") for k in keys)):
        return "dual"
    if ("flow_net.dec1.0.weight" in state_dict and "temporal_fusion.fuse.0.weight" in state_dict
            and "align_refine.body.0.weight" in state_dict):
        return "tfdatv2"
    if ("flow_net.flow_pred.weight" in state_dict and "temporal_fuse.0.weight" in state_dict
            and "conv_first.weight" in state_dict):
        return "tfdat"
    if ("temporal.compress.weight" in state_dict and "feedback_older.0.weight" in state_dict
            and "gblocks.1.weight" in state_dict):
        return "motsr"
    if ("blocks.0.mix.stripe_h.weight" in state_dict and "blocks.0.mix.diag_m.weight" in state_dict
            and "upsampler.MetaUpsample" in state_dict
            and ("conv_first.weight" in state_dict or "conv_first.1.weight" in state_dict)):
        return "wiresr"
    if ("conv_first.weight" in state_dict
            and any(k.endswith(".attn.to_hidden") and ".residual_group.blocks." in k for k in keys)):
        return "heart"
    if ("stem.weight" in state_dict and "upsampler.0.weight" in state_dict
            and "blocks.0.conv1.weight" in state_dict
            and len(state_dict["stem.weight"].shape) == 4
            and state_dict["stem.weight"].shape[2] == 5):
        return "nerve"
    # DAT-iLN (traiNNer-redux): DAT with a LayerNorm before the residual groups.
    if ("before_RG.norm.weight" in state_dict and "conv_first.weight" in state_dict
            and any(k.startswith("layers.0.blocks.0.attn.attns.") for k in keys)):
        return "dat_iln"
    # SCUNet_aaf6aa (traiNNer-redux): SCUNet + residual conv + RRDB upsampler. spandrel
    # loads it as plain SCUNet and fails on the extra layers.
    if ("m_head.0.weight" in state_dict and "m_tail.0.weight" in state_dict
            and any(k.startswith(("m_upsample.", "m_res.")) for k in keys)):
        return "scunet_aaf6aa"
    # SPANF3 (traiNNer-redux, AnimeJaNai V3.1 family): SPAN blocks with att_conv and a
    # Conv3XC to_pixel head. Plain SPAN has upsampler.0 instead; spandrel knows neither.
    if ("conv_cat.weight" in state_dict and "to_pixel.sk.weight" in state_dict
            and "block_1.att_conv.weight" in state_dict):
        return "spanf3"
    return None


# DAT-iLN presets of the engine. Their hyper-parameters (split sizes, expansion…) are
# not all readable from the shapes, so the preset whose state_dict matches exactly wins.
_DAT_ILN_PRESETS = ("dat_iln", "dat_iln_s", "dat_iln_2", "dat_iln_light")


def _pixelshuffle_scale(sd, key: str, default: int = 0) -> int:
    """Scale from a pixel-shuffle conv (out = 3 * scale^2)."""
    if key in sd:
        ch = int(sd[key].shape[0])
        s = int(round(math.sqrt(ch / 3)))
        if s >= 1 and 3 * s * s == ch:
            return s
    return default


def _build_dat_iln(state_dict):
    import traiNNer.archs.dat_iln_arch as m
    want = {k: tuple(v.shape) for k, v in state_dict.items()}
    s0 = _pixelshuffle_scale(state_dict, "upsample.0.weight")
    scales = [s0] if s0 else [2, 4, 1, 3]
    upsampler = "pixelshuffle" if "conv_before_upsample.0.weight" in state_dict else "pixelshuffledirect"
    unshuffle = int(state_dict["conv_first.weight"].shape[1]) > 3
    for name in _DAT_ILN_PRESETS:
        for s in scales:
            try:
                net = getattr(m, name)(scale=s, upsampler=upsampler, unshuffle_mod=unshuffle)
            except Exception:
                continue
            if {k: tuple(v.shape) for k, v in net.state_dict().items()} == want:
                net.load_state_dict(dict(state_dict), strict=True)
                return net, s, {"scale": s, "preset": name, "upsampler": upsampler}
    raise RuntimeError("DAT-iLN : aucun preset du moteur (dat_iln / _s / _2 / _light) ne "
                       "correspond à ces poids (hyper-paramètres modifiés ?)")


def nerve_kwargs(sd) -> dict:
    dim = int(sd["stem.weight"].shape[0])
    in_ch = int(sd["stem.weight"].shape[1])
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    head_out = int(sd["upsampler.0.weight"].shape[0])
    out_ch = in_ch
    scale = int(round(math.sqrt(head_out / out_ch)))
    return {"scale": scale, "in_ch": in_ch, "out_ch": out_ch, "dim": dim, "n_blocks": n_blocks}


def motsr_kwargs(sd) -> dict:
    """MoTSR (xyether, MoSRv2 backbone). Only scale 2 / RGB / unshuffle head exist."""
    dim = int(sd["gblocks.1.weight"].shape[0])
    tail = None
    for k, v in sd.items():
        m = re.fullmatch(r"gblocks\.(\d+)\.weight", k)
        if m and tuple(v.shape) == (2 * dim, dim, 3, 3):
            tail = int(m.group(1))
            break
    n_block = (tail - 2) if tail is not None else 24
    hidden = int(sd["gblocks.2.fc2.weight"].shape[1])
    kw = {
        "scale": 2,
        "dim": dim,
        "n_block": n_block,
        "expansion_ratio": hidden / dim,
        "rms_norm": "gblocks.2.norm.scale" in sd,
        "temporal_dim": int(sd["temporal.compress.weight"].shape[0]),
        "feedback_dim": int(sd["feedback.0.weight"].shape[0]),
    }
    meta = sd.get("to_img.MetaUpsample")
    if meta is not None and hasattr(meta, "tolist"):
        kw["mid_dim"] = int(meta.tolist()[5])
    return kw


# Order of WireSR's SampleMods3 Literal — MetaUpsample stores an index into it.
_WIRESR_UPSAMPLERS = ["conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv",
                      "dysample", "transpose+conv", "lda", "pa_up"]


def wiresr_kwargs(sd) -> dict:
    """Port of the author's spandrel WireSRArch.load() (spandrel2.zip)."""
    _ver, up_idx, internal_scale, dim, out_ch, mid_dim, _grp = [int(v) for v in sd["upsampler.MetaUpsample"].tolist()]
    if "conv_first.1.weight" in sd:
        unshuffle_mod = True
        in_ch = out_ch
        unshuffle = math.isqrt(int(sd["conv_first.1.weight"].shape[1]) // in_ch)
        scale = 4 // unshuffle
    else:
        unshuffle_mod = False
        in_ch = int(sd["conv_first.weight"].shape[1])
        scale = internal_scale
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    attn_every, ca_heads = 0, 4
    for i in range(n_blocks):
        if f"blocks.{i}.ca.qkv.weight" in sd:
            attn_every = i + 1
            ca_heads = int(sd[f"blocks.{i}.ca.temp"].shape[0])
            break
    return {
        "scale": scale, "in_ch": in_ch, "out_ch": out_ch, "dim": dim, "n_blocks": n_blocks,
        "stripe_kernel": int(sd["blocks.0.mix.stripe_h.weight"].shape[3]),
        "diag_kernel": int(sd["blocks.0.mix.diag_m.weight"].shape[2]),
        "square_kernel": int(sd["blocks.0.mix.square.weight"].shape[2]),
        "expansion": int(sd["blocks.0.ffn.net.0.weight"].shape[0]) / dim,
        "attn_every": attn_every, "ca_heads": ca_heads,
        "upsampler": _WIRESR_UPSAMPLERS[up_idx], "mid_dim": mid_dim,
        "unshuffle_mod": unshuffle_mod,
    }


def _tfdat_common(sd, conv_first_key: str) -> dict:
    """Hyperparameters shared by TFDAT v1/v2 (FDAT backbone + UniUpsampleV3)."""
    embed_dim = int(sd[conv_first_key].shape[0])
    in_flat = int(sd[conv_first_key].shape[1])
    meta = [int(v) for v in sd["upsampler.MetaUpsample"].tolist()] if "upsampler.MetaUpsample" in sd else None
    out_ch = meta[4] if meta else 3
    # scale <= 2 runs on pixel_unshuffle'd input (4 // scale) with an internal x4
    # upsampler: MetaUpsample then stores 4, not the real scale.
    if in_flat > out_ch:
        unshuffle = math.isqrt(in_flat // out_ch)
        scale = 4 // unshuffle
        in_ch = in_flat // unshuffle ** 2
    else:
        in_ch = in_flat
        scale = meta[2] if meta else 4
    n_groups = len({k.split(".")[1] for k in sd if k.startswith("groups.")})
    blocks = sorted({int(k.split(".")[3]) for k in sd if k.startswith("groups.0.blocks.")})
    kinds = ["spatial" if f"groups.0.blocks.{b}.attn.bias" in sd else "channel" for b in blocks]
    pattern = kinds
    for n in range(1, len(kinds) + 1):
        if len(kinds) % n == 0 and kinds == kinds[:n] * (len(kinds) // n):
            pattern = kinds[:n]
            break
    num_heads, window_size = 4, 8
    for b, k in zip(blocks, kinds):
        if k == "spatial":
            bias = sd[f"groups.0.blocks.{b}.attn.bias"]
            num_heads, window_size = int(bias.shape[0]), math.isqrt(int(bias.shape[1]))
            break
        num_heads = int(sd[f"groups.0.blocks.{b}.attn.temp"].shape[0])
    b0 = f"groups.0.blocks.{blocks[0]}."
    kw = {
        "num_in_ch": in_ch, "num_out_ch": out_ch, "scale": scale, "embed_dim": embed_dim,
        "num_groups": n_groups, "depth_per_group": len(blocks) // len(pattern),
        "num_heads": num_heads, "window_size": window_size,
        "ffn_expansion_ratio": int(sd[b0 + "ffn.fc1.weight"].shape[0]) / embed_dim,
        "aim_reduction_ratio": embed_dim // int(sd[b0 + "inter.cg.1.weight"].shape[0]),
        "group_block_pattern": pattern,
    }
    if meta:
        kw["upsampler_type"] = _WIRESR_UPSAMPLERS[meta[1]]
        kw["mid_dim"] = meta[5]
    return kw


def fdat_rd_kwargs(sd) -> dict:
    """Port of the author's spandrel FDATRDArch.load() (stinkybread/FDAT-RD)."""
    _v, up_idx, scale, embed_dim, out_ch, mid_dim, _g = [int(v) for v in sd["upsampler.MetaUpsample"].tolist()]
    if "conv_first.1.weight" in sd:
        in_ch = out_ch
        scale = 4 // math.isqrt(int(sd["conv_first.1.weight"].shape[1]) // in_ch)
        unshuffle_mod = True
    else:
        in_ch = int(sd["conv_first.weight"].shape[1])
        unshuffle_mod = False
    n_groups = len({k.split(".")[1] for k in sd if k.startswith("groups.")})
    n_blocks = len({k.split(".")[3] for k in sd if k.startswith("groups.0.blocks.")})

    def btype(i):
        b = f"groups.0.blocks.{i}.attn."
        if b + "dictionary" in sd:
            return "dictionary"
        if b + "temp" in sd:
            return "channel"
        return "spatial"

    types = [btype(i) for i in range(n_blocks)]
    period = next(p for p in range(1, n_blocks + 1)
                  if n_blocks % p == 0 and all(types[i] == types[i % p] for i in range(n_blocks)))
    d_idx = types.index("dictionary")
    return {
        "num_in_ch": in_ch, "num_out_ch": out_ch, "scale": scale, "embed_dim": embed_dim,
        "num_groups": n_groups, "depth_per_group": n_blocks // period,
        "num_heads": int(sd["groups.0.blocks.0.attn.rel_pos_bias"].shape[0]),
        "split_size": (10, 30),  # not stored in the weights; the arch only accepts (10, 30)
        "num_dict_tokens": int(sd[f"groups.0.blocks.{d_idx}.attn.dictionary"].shape[0]),
        "ffn_expansion_ratio": int(sd["groups.0.blocks.0.ffn.fc1.weight"].shape[0]) / embed_dim,
        "aim_reduction_ratio": embed_dim // int(sd["groups.0.blocks.0.inter.cg.1.weight"].shape[0]),
        "group_block_pattern": types[:period],
        "upsampler_type": _WIRESR_UPSAMPLERS[up_idx], "mid_dim": mid_dim,
        "unshuffle_mod": unshuffle_mod,
    }


# WSR presets (width, Halo attention): the registered factories carry the rest.
_WSR_PRESETS = {(64, True): "wsr", (32, True): "wsr_compact", (32, False): "wsr_ultracompact",
                (16, False): "wsr_superultracompact", (8, False): "wsr_mobile"}


def wsr_kwargs(sd) -> dict:
    width = int(sd["stem.sk.weight"].shape[0])
    attn = "attention.query.weight" in sd
    return {"scale": _pixelshuffle_scale(sd, "reconstruct.weight", 1),
            "preset": _WSR_PRESETS.get((width, attn), "wsr" if attn else "wsr_ultracompact")}


# DUAL presets by trunk width; Light = XS geometry + PixelUnshuffle2 input packing.
_DUAL_PRESETS = {128: "dual_xs", 160: "dual_s", 192: "dual_m", 224: "dual_l", 256: "dual_xl"}


def spanf3_kwargs(sd) -> dict:
    """SPANF3: width from conv_cat, depth from the block count. unshuffle_mod (2x / 1x)
    packs the input with PixelUnshuffle(4 // scale) and always upsamples x4 inside."""
    fc = int(sd["conv_cat.weight"].shape[0])
    nb = len({k.split(".")[0] for k in sd if re.match(r"^block_\d+\.", k)})
    unshuffle = "conv_1.1.sk.weight" in sd
    in_ch = int(sd["conv_1.1.sk.weight" if unshuffle else "conv_1.sk.weight"].shape[1])
    up_in = int(round(math.sqrt(int(sd["to_pixel.sk.weight"].shape[0]) / 3)))
    if unshuffle:
        u = int(round(math.sqrt(in_ch / 3)))
        scale = max(1, 4 // max(1, u))
    else:
        scale = up_in
    return {"feature_channels": fc, "num_blocks": nb, "scale": scale, "unshuffle_mod": unshuffle}


def dual_kwargs(sd) -> dict:
    width = int(sd["stem.sequential.reduce.weight"].shape[0])
    light = int(sd["stem.direct.expand.weight"].shape[1]) == 12
    # The tail grows with the internal scale: 2 entries at 1x, +1 per doubling.
    n_tail = len({k.split(".")[1] for k in sd if k.startswith("tail.")})
    internal = 2 ** max(0, n_tail - 2)
    return {"scale": max(1, internal // (2 if light else 1)),
            "preset": "dual_light" if light else _DUAL_PRESETS.get(width, "dual_xs")}


def tfdat_kwargs(sd) -> dict:
    kw = _tfdat_common(sd, "conv_first.weight")
    kw["flow_base_ch"] = int(sd["flow_net.enc.0.weight"].shape[0])
    kw["clip_size"] = 5  # v1 averages the warped neighbours: not stored in the weights
    return kw


def tfdatv2_kwargs(sd) -> dict:
    kw = _tfdat_common(sd, "conv_first.0.weight")
    kw["flow_base_ch"] = int(sd["flow_net.enc1.0.weight"].shape[0])
    kw["clip_size"] = int(sd["temporal_fusion.fuse.0.weight"].shape[1]) // kw["embed_dim"]
    return kw


def _heart_scale(sd, num_feat) -> int:
    ups = sorted(int(m.group(1)) for k in sd
                 for m in [re.fullmatch(r"upsample\.(\d+)\.weight", k)] if m)
    if not ups:
        return 1
    first_out = int(sd[f"upsample.{ups[0]}.weight"].shape[0])
    if first_out == 9 * num_feat:
        return 3
    return 2 ** len(ups)


def heart_kwargs(sd) -> dict:
    embed_dim = int(sd["conv_first.weight"].shape[0])
    in_chans = int(sd["conv_first.weight"].shape[1])
    layer_ids = sorted({int(k.split(".")[1]) for k in sd if k.startswith("layers.")})
    depths, num_heads = [], []
    attn_idx = []
    rank = rib_hidden_dim = rib_n_freqs = None
    for li in layer_ids:
        pre = f"layers.{li}.residual_group.blocks."
        blocks = sorted({int(k[len(pre):].split(".")[0]) for k in sd if k.startswith(pre)})
        depths.append(len(blocks))
        heads = None
        for b in blocks:
            q = sd.get(f"{pre}{b}.attn.to_q")
            if q is not None:
                if li == layer_ids[0]:
                    attn_idx.append(b)
                if heads is None:
                    heads = int(q.shape[0])
                    rank = int(q.shape[2])
                    rib_hidden_dim = int(q.shape[1])
                    rib_n_freqs = (int(sd[f"{pre}{b}.attn.to_hidden"].shape[0]) - 2) // 4
        num_heads.append(heads if heads is not None else 6)
    attention_freq = (attn_idx[1] - attn_idx[0]) if len(attn_idx) > 1 else max(1, depths[0])

    b0 = f"layers.{layer_ids[0]}.residual_group.blocks.0."
    compress_ratio = embed_dim // int(sd[b0 + "conv_block.cab.0.weight"].shape[0])
    squeeze_factor = embed_dim // int(sd[b0 + "conv_block.cab.3.attention.1.weight"].shape[0])
    mlp_ratio = int(sd[b0 + "mlp.fc1.weight"].shape[0]) / embed_dim
    num_feat = int(sd["conv_before_upsample.0.weight"].shape[0])
    kw = {
        "scale": _heart_scale(sd, num_feat),
        "in_chans": in_chans,
        "embed_dim": embed_dim,
        "depths": depths,
        "num_heads": num_heads,
        "compress_ratio": compress_ratio,
        "squeeze_factor": squeeze_factor,
        "mlp_ratio": mlp_ratio,
        "num_feat": num_feat,
        "attention_freq": attention_freq,
        "ape": "absolute_pos_embed" in sd,
        "patch_norm": "patch_embed.norm.weight" in sd,
        "resi_connection": "1conv" if f"layers.{layer_ids[0]}.conv.weight" in sd else "identity",
    }
    if rank is not None:
        kw.update(rank=rank, rib_hidden_dim=rib_hidden_dim, rib_n_freqs=rib_n_freqs)
    return kw


class _Shape:
    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = tuple(shape)


def shapes_from_file(path: str) -> dict:
    """{name: object with .shape} without torch — enough for detect()/*_kwargs().
    safetensors: header only (the frozen GUI has no torch). .pth: needs torch."""
    if path.lower().endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, framework="np") as f:
            return {k: (f.get_tensor(k) if k.endswith("MetaUpsample")
                        else _Shape(f.get_slice(k).get_shape())) for k in f.keys()}
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params_ema", "params_g", "params", "model", "state_dict"):
        if isinstance(ck, dict) and isinstance(ck.get(key), dict):
            ck = ck[key]
            break
    return {k: (v if k.endswith("MetaUpsample") else _Shape(v.shape))
            for k, v in ck.items() if hasattr(v, "shape")}


_KWARGS = {"heart": "heart_kwargs", "nerve": "nerve_kwargs", "motsr": "motsr_kwargs",
           "wiresr": "wiresr_kwargs", "tfdat": "tfdat_kwargs", "tfdatv2": "tfdatv2_kwargs",
           "fdat_rd": "fdat_rd_kwargs", "scunet_aaf6aa": "scunet_aaf6aa_kwargs",
           "wsr": "wsr_kwargs", "dual": "dual_kwargs", "spanf3": "spanf3_kwargs"}


def describe_state(arch: str, sd) -> dict | None:
    """kwargs for an already-detected arch (sd may hold real tensors or _Shape)."""
    fn = _KWARGS.get(arch)
    return globals()[fn](sd) if fn else None


def describe(path: str):
    """(arch, kwargs) for a file whose arch is handled here, else (None, None)."""
    sd = shapes_from_file(path)
    arch = detect(sd)
    if arch == "heart":
        return arch, heart_kwargs(sd)
    if arch == "nerve":
        return arch, nerve_kwargs(sd)
    if arch == "motsr":
        return arch, motsr_kwargs(sd)
    if arch == "wiresr":
        return arch, wiresr_kwargs(sd)
    if arch == "fdat_rd":
        return arch, fdat_rd_kwargs(sd)
    if arch == "tfdat":
        return arch, tfdat_kwargs(sd)
    if arch == "tfdatv2":
        return arch, tfdatv2_kwargs(sd)
    if arch == "wsr":
        return arch, wsr_kwargs(sd)
    if arch == "dual":
        return arch, dual_kwargs(sd)
    if arch == "spanf3":
        return arch, spanf3_kwargs(sd)
    return None, None


def spanplus_kwargs(sd) -> dict:
    """SPANPlus (traiNNer-redux) — spanplus / _s / _st / _sts differ by feature_channels,
    blocks and upsampler; all three are readable from the keys and shapes."""
    fc_t = sd.get("feats.0.eval_conv.weight", sd.get("feats.0.sk.weight"))
    fc = int(fc_t.shape[0]) if fc_t is not None else 48
    spabs = sorted({int(k.split(".")[1]) for k in sd if re.match(r"^feats\.[1-9]\d*\.", k)})
    blocks = [len({k.split(".")[3] for k in sd if k.startswith(f"feats.{i}.block_n.")})
              for i in spabs] or [4]
    if "upsampler.offset.weight" in sd:
        # DySample: offset out = 2 * groups(4) * scale^2
        ch = int(sd["upsampler.offset.weight"].shape[0])
        up, scale = "dys", int(round(math.sqrt(ch / 8)))
    elif "upsampler.0.weight" in sd:
        up, scale = "ps", _pixelshuffle_scale(sd, "upsampler.0.weight", 4)
    else:
        up, scale = "conv", 1
    return {"feature_channels": fc, "blocks": blocks, "upsampler": up, "scale": scale}


def scunet_aaf6aa_kwargs(sd) -> dict:
    """Same rule as SCUNet_aaf6aa(state=...): 2 ** (number of up-convs - 1)."""
    n = len([k for k in sd if re.match(r"m_upsample\.0\.up\.[0-9]+\.weight", k)])
    return {"scale": 2 ** max(0, n - 1)}


def build(arch: str, state_dict):
    """Instantiate + strict-load HEART/NERVE. Needs traiNNer-redux (master) importable.
    Returns (model, scale, kwargs)."""
    if arch == "dat_iln":
        return _build_dat_iln(state_dict)
    if arch == "scunet_aaf6aa":
        from traiNNer.archs.scunet_aaf6aa_arch import SCUNet_aaf6aa
        model = SCUNet_aaf6aa(state=state_dict)   # the class reads its own config from it
        model.load_state_dict(dict(state_dict), strict=True)
        return model, int(model.scale), {"scale": int(model.scale)}
    if arch == "spanplus":
        from traiNNer.archs.spanplus_arch import SpanPlus
        kw = spanplus_kwargs(state_dict)
        model = SpanPlus(feature_channels=kw["feature_channels"], blocks=kw["blocks"],
                         upscale=kw["scale"], upsampler=kw["upsampler"])
        model.load_state_dict(dict(state_dict), strict=True)
        return model, kw["scale"], kw
    if arch == "spanf3":
        from traiNNer.archs.spanf3_arch import spanf3
        kw = spanf3_kwargs(state_dict)
        model = spanf3(**kw)
        model.load_state_dict(dict(state_dict), strict=True)
        return model, kw["scale"], kw
    if arch in ("wsr", "dual"):
        # Rift73 archs: preset factories (geometry not all readable from the shapes).
        try:
            import importlib
            mod = importlib.import_module(f"traiNNer.archs.{arch}_arch")
        except ImportError:
            raise RuntimeError(
                f"{arch.upper()} reconnu, mais le fichier de son architecture est absent de "
                f"cette installation (réinstallez l'app). / {arch.upper()} detected, but its "
                "architecture file is missing from this installation (reinstall the app).")
        kw = wsr_kwargs(state_dict) if arch == "wsr" else dual_kwargs(state_dict)
        extra = {"use_checkpoint": False} if arch == "dual" else {}
        model = getattr(mod, kw["preset"])(scale=kw["scale"], **extra)
        model.load_state_dict(dict(state_dict), strict=True)
        return model, kw["scale"], kw
    if arch == "nerve":
        from traiNNer.archs.nerve_arch import nerve as factory
        kw = nerve_kwargs(state_dict)
    elif arch == "heart":
        from traiNNer.archs.heart_arch import heart as factory
        kw = heart_kwargs(state_dict)
    elif arch == "motsr":
        from traiNNer.archs.motsr_arch import motsr as factory
        kw = motsr_kwargs(state_dict)
    elif arch == "wiresr":
        try:
            from traiNNer.archs.wiresr_arch import wiresr as factory
        except ImportError:
            raise RuntimeError(
                "WireSR reconnu, mais le fichier de son architecture est absent de cette "
                "installation (réinstallez l'app). / WireSR detected, but its architecture "
                "file is missing from this installation (reinstall the app).")
        kw = wiresr_kwargs(state_dict)
    elif arch == "fdat_rd":
        try:
            from traiNNer.archs.fdat_rd_arch import FDATRD as factory
        except ImportError:
            raise RuntimeError(
                "FDAT-RD reconnu, mais le fichier de son architecture est absent de cette "
                "installation (réinstallez l'app). / FDAT-RD detected, but its architecture "
                "file is missing from this installation (reinstall the app).")
        kw = fdat_rd_kwargs(state_dict)
    elif arch in ("tfdat", "tfdatv2"):
        # Both classes force strict=False in their own load_state_dict (to accept FDAT
        # pretrains) — check the key sets ourselves so a wrong rebuild never loads silently.
        if arch == "tfdat":
            from traiNNer.archs.tfdat_arch import TFDAT as factory
            kw = tfdat_kwargs(state_dict)
        else:
            from traiNNer.archs.tfdatv2_arch import TFDAT as factory
            kw = tfdatv2_kwargs(state_dict)
        model = factory(**kw)
        own = set(model.state_dict()) - {"upsampler.MetaUpsample"}
        ckpt = set(state_dict) - {"upsampler.MetaUpsample"}
        if own != ckpt:
            raise RuntimeError(f"{arch.upper()} : reconstruction incomplète — manquantes "
                               f"{sorted(own - ckpt)[:3]}, inattendues {sorted(ckpt - own)[:3]}")
        model.load_state_dict(dict(state_dict))
        bad = [k for k, v in model.state_dict().items()
               if k in state_dict and tuple(v.shape) != tuple(state_dict[k].shape)]
        if bad:
            raise RuntimeError(f"{arch.upper()} : formes incompatibles {bad[:3]}")
        return model, kw["scale"], kw
    else:
        raise ValueError(f"arch non gérée par arch_infer : {arch}")
    model = factory(**kw)
    model.load_state_dict(state_dict, strict=True)
    return model, kw["scale"], kw
