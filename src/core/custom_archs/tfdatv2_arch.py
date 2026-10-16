# TFDAT v2 — official source, Kim2091/traiNNer-redux-1 branch "tfdat-v2" (commit of
# 2026-07-05, "TFDAT v2: rework arch, add annealed flow supervision"), reference repo
# https://github.com/Kim2091/TFDAT (Apache-2.0, see licenses/TFDAT-Apache-2.0.txt).
# Unchanged except the registry name: "tfdatv2" instead of "tfdat", so it lives next to
# TFDAT v1 ("tfdat", engine file) — v1 and v2 checkpoints are incompatible.
# Used by e.g. 2x_bndl_animefilm_v3 (Bendel, 9-frame clips).

# Based on FDAT: https://github.com/stinkybread/fdat/blob/main/fdat.py
# ruff: noqa
# type: ignore
from math import gcd

import torch
import torch.nn.functional as F  # noqa: N812
from spandrel.__helpers.model_descriptor import StateDict
from spandrel.util.timm import DropPath
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from torch.nn.modules.module import _IncompatibleKeys  # type: ignore

from traiNNer.archs.arch_util import SampleMods3, UniUpsampleV3
from traiNNer.utils.registry import ARCH_REGISTRY


def warp_frame(x: Tensor, flow: Tensor) -> Tensor:
    """Warp frame x according to optical flow.

    Uses gather-based bilinear interpolation for DirectML/ONNX compatibility.
    Avoids F.grid_sample which has limited DirectML support.

    The sampling coordinate is decomposed into an exact integer base grid plus
    the flow's floor/fraction. The fractional interpolation weights are computed
    at flow magnitude (small numbers) instead of absolute pixel positions, so
    sub-pixel precision survives fp16 even at 4K resolutions, where absolute
    coordinates above 1024 would otherwise lose their fractional part entirely.

    Args:
        x: Input frame/features (B, C, H, W)
        flow: Optical flow (B, 2, H, W) in pixel units

    Returns:
        Warped frame (B, C, H, W), border-replicated outside the frame
    """
    B, C, H, W = x.shape

    # Exact integer base grid
    yy = torch.arange(H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
    xx = torch.arange(W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)

    fx = flow[:, 0:1]
    fy = flow[:, 1:2]
    fx0 = fx.floor()
    fy0 = fy.floor()
    # Fractional parts: small-magnitude values, fp16-safe
    dx = fx - fx0
    dy = fy - fy0

    # Integer sampling coordinates (exact int64 arithmetic)
    x0 = xx + fx0.long()
    y0 = yy + fy0.long()
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp with Min/Max (int-friendly in TensorRT, unlike Clip)
    zero = torch.zeros(1, dtype=torch.long, device=x.device)
    max_x = torch.full((1,), W - 1, dtype=torch.long, device=x.device)
    max_y = torch.full((1,), H - 1, dtype=torch.long, device=x.device)
    x0 = torch.minimum(torch.maximum(x0, zero), max_x)
    x1 = torch.minimum(torch.maximum(x1, zero), max_x)
    y0 = torch.minimum(torch.maximum(y0, zero), max_y)
    y1 = torch.minimum(torch.maximum(y1, zero), max_y)

    # Linear indices into flattened spatial dims
    x_flat = x.reshape(B, C, H * W)
    idx_00 = (y0 * W + x0).view(B, 1, H * W).expand(B, C, H * W)
    idx_01 = (y0 * W + x1).view(B, 1, H * W).expand(B, C, H * W)
    idx_10 = (y1 * W + x0).view(B, 1, H * W).expand(B, C, H * W)
    idx_11 = (y1 * W + x1).view(B, 1, H * W).expand(B, C, H * W)

    p00 = x_flat.gather(-1, idx_00).view(B, C, H, W)
    p01 = x_flat.gather(-1, idx_01).view(B, C, H, W)
    p10 = x_flat.gather(-1, idx_10).view(B, C, H, W)
    p11 = x_flat.gather(-1, idx_11).view(B, C, H, W)

    w00 = (1 - dx) * (1 - dy)
    w01 = dx * (1 - dy)
    w10 = (1 - dx) * dy
    w11 = dx * dy

    return w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11


def flow_smoothness(flow: Tensor, img: Tensor, edge_lambda: float = 10.0) -> Tensor:
    """First-order edge-aware smoothness prior for optical flow.

    Penalizes flow gradients except across image edges, where motion
    discontinuities are expected. Used only for the training-time auxiliary
    loss; never part of the inference graph.
    """
    fdx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean(1, keepdim=True)
    fdy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean(1, keepdim=True)
    idx = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(1, keepdim=True)
    idy = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(1, keepdim=True)
    return (fdx * torch.exp(-edge_lambda * idx)).mean() + (
        fdy * torch.exp(-edge_lambda * idy)
    ).mean()


def _flow_head(in_ch: int, hidden_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_ch, 2, 3, 1, 1),
    )


class FlowPyramidNet(nn.Module):
    """Lightweight coarse-to-fine optical flow (SPyNet-style, 3 levels).

    A shared encoder builds a 1/2, 1/4, 1/8 feature pyramid per frame (each
    frame is encoded once and reused across pairs). Flow is estimated at 1/8
    resolution and refined twice: at each finer level the neighbor features are
    warped by the upsampled flow and a residual correction is predicted. This
    handles much larger and more precise motion than a single-scale predictor.

    All flow heads are zero-initialized so alignment starts as identity.
    TensorRT and ONNX compatible.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 3
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c3, c3, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Coarsest level predicts absolute flow; finer levels predict residuals
        self.dec3 = _flow_head(c3 * 2, c3)
        self.dec2 = _flow_head(c2 * 2 + 2, c2)
        self.dec1 = _flow_head(c1 * 2 + 2, c1)

    def zero_init_flow_heads(self) -> None:
        for dec in (self.dec1, self.dec2, self.dec3):
            nn.init.zeros_(dec[-1].weight)
            nn.init.zeros_(dec[-1].bias)

    def encode(self, frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        f1 = self.enc1(frame)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        return f1, f2, f3

    @staticmethod
    def _up2(flow: Tensor) -> Tensor:
        return (
            F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=False)
            * 2.0
        )

    def forward(
        self,
        pyr_src: tuple[Tensor, Tensor, Tensor],
        pyr_tgt: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """Estimate flow such that warp_frame(src, flow) aligns to tgt.

        Args:
            pyr_src: Feature pyramid of the source (neighbor) frame
            pyr_tgt: Feature pyramid of the target (center) frame

        Returns:
            flow: Optical flow (B, 2, H, W) in pixel units at full resolution
        """
        f1s, f2s, f3s = pyr_src
        f1t, f2t, f3t = pyr_tgt

        # 1/8 resolution: absolute flow
        flow3 = self.dec3(torch.cat([f3s, f3t], dim=1))

        # 1/4 resolution: warp + residual
        up2 = self._up2(flow3)
        f2s_w = warp_frame(f2s, up2)
        flow2 = up2 + self.dec2(torch.cat([f2s_w, f2t, up2], dim=1))

        # 1/2 resolution: warp + residual
        up1 = self._up2(flow2)
        f1s_w = warp_frame(f1s, up1)
        flow1 = up1 + self.dec1(torch.cat([f1s_w, f1t, up1], dim=1))

        return self._up2(flow1)


class AlignmentRefineBlock(nn.Module):
    """Feature-space alignment cleanup after pixel-space warping.

    Pixel-space warping is exact at full resolution but leaves small residual
    misalignments. This block sees the warped-neighbor and center features
    together and predicts a residual correction. Zero-initialized: starts as
    identity. Shared across all neighbors.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

    def zero_init_last(self) -> None:
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, warped: Tensor, center: Tensor) -> Tensor:
        return warped + self.body(torch.cat([warped, center], dim=1))


class TemporalAttentionFusion(nn.Module):
    """Confidence-gated temporal fusion (TSA-style, export-friendly).

    Each aligned neighbor gets a per-pixel confidence in [0, 1] predicted from
    (a) the embedded feature similarity with the center frame and (b) the
    photometric warp error. Occluded or misaligned regions are suppressed
    instead of ghosting into the output. Center and gated neighbor features are
    then concatenated (preserving per-frame sub-pixel detail) and fused, with a
    squeeze-excite gate for globally consistent exposure across frames.
    """

    def __init__(self, dim: int, clip_size: int, reduction_ratio: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.emb_center = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.emb_neighbor = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.conf = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * clip_size, dim, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def zero_init_conf(self) -> None:
        # sigmoid(0) = 0.5: neighbors start half-trusted
        nn.init.zeros_(self.conf[-1].weight)
        nn.init.zeros_(self.conf[-1].bias)

    def forward(
        self, center: Tensor, neighbors: list[Tensor], errs: list[Tensor]
    ) -> Tensor:
        emb_c = self.emb_center(center)
        scale = self.dim**-0.5
        gated = [center]
        for feat, err in zip(neighbors, errs):
            sim = (emb_c * self.emb_neighbor(feat)).sum(dim=1, keepdim=True) * scale
            w = torch.sigmoid(self.conf(torch.cat([sim, err], dim=1)))
            gated.append(feat * w)
        f = self.fuse(torch.cat(gated, dim=1))
        return f * self.se(f)


# --- FDAT Components ---
class FastSpatialWindowAttention(nn.Module):
    """Window attention with optional half-window shift (Swin-style).

    Alternating shifted blocks let detail propagate across window boundaries,
    removing the 8px blocking grid on fine texture. The shift is a torch.roll
    (exports as Slice+Concat, TensorRT-safe). No attention mask is used: the
    only affected windows are the wrap-around ones at the image border, a
    negligible fraction for SR-sized inputs, and masking is what makes shifted
    windows export-hostile.
    """

    def __init__(self, dim, window_size=8, num_heads=4, qkv_bias=False, shift=False) -> None:
        super().__init__()
        self.dim, self.ws, self.nh = dim, window_size, num_heads
        self.shift = window_size // 2 if shift else 0
        self.scale = (dim // num_heads) ** -0.5
        self.qkv, self.proj = (
            nn.Linear(dim, dim * 3, bias=qkv_bias),
            nn.Linear(dim, dim),
        )
        self.bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size, window_size * window_size)
        )
        trunc_normal_(self.bias, std=0.02)

    def forward(self, x, H, W):
        B, L, C = x.shape

        # Calculate padded dimensions explicitly for TensorRT compatibility
        H_pad = ((H + self.ws - 1) // self.ws) * self.ws
        W_pad = ((W + self.ws - 1) // self.ws) * self.ws
        pad_b = H_pad - H
        pad_r = W_pad - W

        # Reshape to spatial, pad, then process
        x = x.view(B, H, W, C)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # Partition into windows with explicit window counts
        nH, nW = H_pad // self.ws, W_pad // self.ws
        x = (
            x.view(B, nH, self.ws, nW, self.ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B * nH * nW, self.ws * self.ws, C)
        )

        # Attention
        qkv = (
            self.qkv(x)
            .view(-1, self.ws * self.ws, 3, self.nh, C // self.nh)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale @ k.transpose(-2, -1)) + self.bias
        x = (
            (F.softmax(attn, dim=-1) @ v)
            .transpose(1, 2)
            .reshape(-1, self.ws * self.ws, C)
        )

        # Reverse window partition
        x = (
            self.proj(x)
            .view(B, nH, nW, self.ws, self.ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, H_pad, W_pad, C)
        )

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        return x.view(B, H * W, C)


class FastChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False) -> None:
        super().__init__()
        self.nh = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv, self.proj = (
            nn.Linear(dim, dim * 3, bias=qkv_bias),
            nn.Linear(dim, dim),
        )

    def forward(self, x, H, W):  # H, W are unused but kept for API consistency
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Pre-scale by 1/sqrt(N) before normalize: mathematically a no-op
        # (normalize is scale-invariant) but keeps the internal sum-of-squares
        # over N tokens inside fp16 range at high resolutions.
        inv_n = N**-0.5
        q = F.normalize(q.transpose(-2, -1) * inv_n, dim=-1, eps=1e-4)
        k = F.normalize(k.transpose(-2, -1) * inv_n, dim=-1, eps=1e-4)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.temp, dim=-1)
        return self.proj(
            (attn @ v.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        )


class SimplifiedAIM(nn.Module):
    def __init__(self, dim, reduction_ratio=8) -> None:
        super().__init__()
        self.sg = nn.Sequential(nn.Conv2d(dim, 1, 1, bias=False), nn.Sigmoid())
        self.cg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, attn_feat, conv_feat, interaction_type, H, W):
        B, L, C = attn_feat.shape
        if interaction_type == "spatial_modulates_channel":
            sm = (
                self.sg(attn_feat.transpose(1, 2).view(B, C, H, W))
                .view(B, 1, L)
                .transpose(1, 2)
            )
            return attn_feat + (conv_feat * sm)
        else:
            cm = (
                self.cg(conv_feat.transpose(1, 2).view(B, C, H, W))
                .view(B, C, 1)
                .transpose(1, 2)
            )
            return (attn_feat * cm) + conv_feat


class SimplifiedFFN(nn.Module):
    def __init__(self, dim, expansion_ratio=2.0, drop=0.0) -> None:
        super().__init__()
        hd = int(dim * expansion_ratio)
        self.fc1, self.act, self.fc2 = (
            nn.Linear(dim, hd, False),
            nn.GELU(),
            nn.Linear(hd, dim, False),
        )
        self.drop = nn.Dropout(drop)
        self.smix = nn.Conv2d(hd, hd, 3, 1, 1, groups=hd, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.drop(self.act(self.fc1(x)))
        x_s = (
            self.smix(x.transpose(1, 2).view(B, x.shape[-1], H, W))
            .view(B, x.shape[-1], L)
            .transpose(1, 2)
        )
        return self.drop(self.fc2(x_s))


class SimplifiedDATBlock(nn.Module):
    def __init__(
        self, dim, nh, ws, ffn_exp, aim_re, btype, dp, qkv_b=False, shift=False
    ) -> None:
        super().__init__()
        self.btype = btype
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = (
            FastSpatialWindowAttention(dim, ws, nh, qkv_b, shift=shift)
            if btype == "spatial"
            else FastChannelAttention(dim, nh, qkv_b)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), nn.GELU()
        )
        self.inter = SimplifiedAIM(dim, aim_re)
        self.dp = DropPath(dp) if dp > 0.0 else nn.Identity()
        self.ffn = SimplifiedFFN(dim, ffn_exp)

    def _conv_fwd(self, x, H, W):
        B, L, C = x.shape
        return (
            self.conv(x.transpose(1, 2).view(B, C, H, W)).view(B, C, L).transpose(1, 2)
        )

    def forward(self, x, H, W):
        n1 = self.n1(x)
        itype = (
            "channel_modulates_spatial"
            if self.btype == "spatial"
            else "spatial_modulates_channel"
        )
        fused = self.inter(self.attn(n1, H, W), self._conv_fwd(n1, H, W), itype, H, W)
        x = x + self.dp(fused)
        x = x + self.dp(self.ffn(self.n2(x), H, W))
        return x


class SimplifiedResidualGroup(nn.Module):
    def __init__(self, dim, depth, nh, ws, ffn_exp, aim_re, pattern, dp_rates) -> None:
        super().__init__()
        # Alternate plain/shifted windows across the spatial blocks of the group
        blocks = []
        spatial_idx = 0
        for i in range(depth):
            btype = pattern[i % len(pattern)]
            shift = False
            if btype == "spatial":
                shift = spatial_idx % 2 == 1
                spatial_idx += 1
            blocks.append(
                SimplifiedDATBlock(
                    dim, nh, ws, ffn_exp, aim_re, btype, dp_rates[i], shift=shift
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_seq = x.view(B, C, H * W).transpose(1, 2).contiguous()
        for block in self.blocks:
            x_seq = block(x_seq, H, W)
        return self.conv(x_seq.transpose(1, 2).view(B, C, H, W)) + x


class TFDAT(nn.Module):
    """Temporal FDAT for Video Super-Resolution (v2).

    Input: (B, T, C, H, W) - batch of video clips with T frames (T must be odd
        and equal to clip_size). A 4D input (B, T*C, H, W) as produced by
        vsmlrt-style channel stacking is also accepted.
    Output: (B, C, H*scale, W*scale) - upscaled center frame

    Pipeline:
        1. A shared pyramid encoder + coarse-to-fine flow net aligns each
           neighbor to the center frame at full resolution (fp16-robust warp).
        2. Every frame is featurized independently (no pixel averaging), and
           warped-neighbor features get a zero-init residual alignment cleanup.
        3. Confidence-gated temporal fusion suppresses occluded/misaligned
           regions and preserves per-frame sub-pixel detail.
        4. FDAT spatial/channel-attention backbone + upsampler reconstruct the
           residual on top of a bilinear-upscaled center frame, anchoring the
           output to the input for temporal stability.

    TensorRT, ONNX and fp16 compatible.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 2,
        clip_size: int = 5,
        embed_dim: int = 64,
        num_groups: int = 4,
        depth_per_group: int = 3,
        num_heads: int = 4,
        window_size: int = 8,
        ffn_expansion_ratio: float = 2.0,
        aim_reduction_ratio: int = 8,
        group_block_pattern: list[str] | None = None,
        drop_path_rate: float = 0.1,
        mid_dim: int = 64,
        upsampler_type: SampleMods3 = "pixelshuffle",
        flow_base_ch: int = 32,
        flow_aux_weight: float = 0.1,
        flow_aux_smooth_weight: float = 0.1,
        flow_aux_anneal_iters: int = 100000,
    ) -> None:
        if group_block_pattern is None:
            group_block_pattern = ["spatial", "channel"]
        if clip_size % 2 == 0:
            raise ValueError("clip_size must be odd")
        super().__init__()

        self.clip_size = clip_size
        self.upscale = scale
        self.num_in_ch = num_in_ch
        self.res_connect = num_in_ch == num_out_ch

        # Training-only auxiliary flow supervision (photometric + smoothness).
        # Computed during forward when training, consumed by trainers that
        # support get_aux_losses(). Never active in eval/export.
        self.flow_aux_weight = flow_aux_weight
        self.flow_aux_smooth_weight = flow_aux_smooth_weight
        self.flow_aux_anneal_iters = flow_aux_anneal_iters
        self.flow_aux_loss: Tensor | None = None

        # Unshuffle setup for 1x and 2x scale
        self.unshuffle = 1
        internal_scale = scale
        if scale <= 2:
            self.unshuffle = 4 // max(scale, 1) if scale > 0 else 4
            internal_scale = 4
            conv_in_ch = num_in_ch * self.unshuffle**2
        else:
            conv_in_ch = num_in_ch

        # Pyramid optical flow (operates on original resolution)
        self.flow_net = FlowPyramidNet(num_in_ch, flow_base_ch)

        # Shared per-frame shallow feature extraction
        self.conv_first = nn.Sequential(
            nn.Conv2d(conv_in_ch, embed_dim, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True),
        )

        # Feature-space alignment cleanup (shared across neighbors)
        self.align_refine = AlignmentRefineBlock(embed_dim)

        # Confidence-gated temporal fusion
        self.temporal_fusion = TemporalAttentionFusion(
            embed_dim, clip_size, aim_reduction_ratio
        )

        # Main FDAT backbone
        ad = depth_per_group * len(group_block_pattern)
        td = num_groups * ad
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, td)]

        self.groups = nn.Sequential(
            *[
                SimplifiedResidualGroup(
                    embed_dim,
                    ad,
                    num_heads,
                    window_size,
                    ffn_expansion_ratio,
                    aim_reduction_ratio,
                    group_block_pattern,
                    dpr[i * ad : (i + 1) * ad],
                )
                for i in range(num_groups)
            ]
        )

        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.upsampler = UniUpsampleV3(
            upsampler_type, internal_scale, embed_dim, num_out_ch, mid_dim, 4
        )

        # HR refinement tail: a narrow zero-init residual cleanup at output
        # resolution, right where upsampling artifacts are visible
        self.hr_refine = nn.Sequential(
            nn.Conv2d(num_out_ch, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, num_out_ch, 3, 1, 1),
        )

        # Alignment requirement: LCM of (unshuffle * window_size) and the flow
        # pyramid stride (three stride-2 levels -> divisible by 8). After
        # unshuffle, feature dims must still be divisible by window_size, so
        # the input must be divisible by (unshuffle * window_size).
        def lcm(a, b):
            return a * b // gcd(a, b)

        self.align = lcm(self.unshuffle * window_size, 8)

        self.apply(self._init_weights)

        # Zero-inits must run after the general init so they aren't overwritten
        self.flow_net.zero_init_flow_heads()
        self.align_refine.zero_init_last()
        self.temporal_fusion.zero_init_conf()
        nn.init.zeros_(self.hr_refine[-1].weight)
        nn.init.zeros_(self.hr_refine[-1].bias)

    def load_state_dict(
        self,
        state_dict: StateDict,
        strict: bool = False,
        *args,
        **kwargs,
    ) -> _IncompatibleKeys:
        state_dict["upsampler.MetaUpsample"] = self.upsampler.MetaUpsample
        # Always use strict=False to allow loading FDAT pretrained weights
        return super().load_state_dict(state_dict, strict=False, *args, **kwargs)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm | nn.GroupNorm):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _to_feat(self, img: Tensor) -> Tensor:
        if self.unshuffle > 1:
            img = F.pixel_unshuffle(img, self.unshuffle)
        return self.conv_first(img)

    def get_aux_losses(self, current_iter: int) -> dict[str, Tensor]:
        """Auxiliary training losses from the most recent forward pass.

        Called by trainers that support generator-internal losses (see
        traiNNer-redux SRModel). The flow supervision weight anneals linearly
        to zero over flow_aux_anneal_iters: early on it gives the flow net a
        direct learning signal, later the task loss takes over so flow isn't
        forced to minimize photometric error in occluded regions forever.
        """
        losses: dict[str, Tensor] = {}
        if self.flow_aux_loss is not None:
            if self.flow_aux_anneal_iters > 0:
                ramp = max(0.0, 1.0 - current_iter / self.flow_aux_anneal_iters)
            else:
                ramp = 1.0
            weight = self.flow_aux_weight * ramp
            if weight > 0:
                losses["l_g_flow_aux"] = self.flow_aux_loss * weight
        return losses

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, T, C, H, W), or (B, T*C, H, W) with frames
               stacked along the channel dimension (vsmlrt-style)

        Returns:
            Upscaled center frame (B, C, H*scale, W*scale)
        """
        if x.ndim == 4:
            b, tc, h, w = x.shape
            x = x.reshape(b, self.clip_size, tc // self.clip_size, h, w)
        B, T, C, H, W = x.shape
        if T != self.clip_size:
            raise ValueError(f"expected clip_size={self.clip_size} frames, got {T}")
        center_idx = T // 2

        # Compute padding for alignment (must satisfy flow net, window attention, and unshuffle)
        pad_h = (self.align - H % self.align) % self.align
        pad_w = (self.align - W % self.align) % self.align
        if pad_h > 0 or pad_w > 0:
            # Reshape to 4D for padding (reflect mode doesn't support 5D with pad size 4)
            x = x.reshape(B * T, C, H, W)
            # Use replicate if padding >= dimension size (reflect requires padding < dim)
            pad_mode = "reflect" if pad_h < H and pad_w < W else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
            x = x.view(B, T, C, H + pad_h, W + pad_w)

        # Extract center frame
        center = x[:, center_idx]  # (B, C, H_pad, W_pad)

        # Align neighboring frames to center using coarse-to-fine optical flow.
        # The center pyramid is encoded once and reused for every pair.
        pyr_center = self.flow_net.encode(center)
        warped_neighbors = []
        warp_errs = []
        self.flow_aux_loss = None
        compute_aux = self.training and self.flow_aux_weight > 0
        aux = None
        for t in range(T):
            if t == center_idx:
                continue
            neighbor = x[:, t]
            flow = self.flow_net(self.flow_net.encode(neighbor), pyr_center)
            warped = warp_frame(neighbor, flow)
            warped_neighbors.append(warped)
            # Photometric error: explicit occlusion/misalignment signal
            err = (warped - center).abs().mean(dim=1, keepdim=True)
            warp_errs.append(err)
            if compute_aux:
                pair_aux = err.mean() + self.flow_aux_smooth_weight * flow_smoothness(
                    flow, center
                )
                aux = pair_aux if aux is None else aux + pair_aux
        if aux is not None:
            self.flow_aux_loss = aux / (T - 1)

        # Per-frame feature extraction (no pixel averaging: sub-pixel detail
        # from each frame survives into fusion)
        center_feat = self._to_feat(center)
        neighbor_feats = [
            self.align_refine(self._to_feat(w), center_feat) for w in warped_neighbors
        ]

        # Bring error maps to feature resolution
        if self.unshuffle > 1:
            warp_errs = [F.avg_pool2d(e, self.unshuffle) for e in warp_errs]

        # Confidence-gated temporal fusion
        fused = self.temporal_fusion(center_feat, neighbor_feats, warp_errs)

        # Main backbone with residual
        x_deep = self.conv_after(self.groups(fused))
        out = self.upsampler(x_deep + fused)

        # Base-frame residual: anchors the output to the input frame, which
        # stabilizes the result temporally and shrinks the residual the
        # network has to represent (helps fp16 dynamic range).
        if self.res_connect:
            if self.upscale > 1:
                base = F.interpolate(
                    center,
                    scale_factor=self.upscale,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                base = center
            out = out + base

        # HR-space residual cleanup (identity at init)
        out = out + self.hr_refine(out)

        # Crop to original size (accounting for scale)
        return out[:, :, : H * self.upscale, : W * self.upscale]


# --- Model Variants ---

@ARCH_REGISTRY.register()
def tfdatv2(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    scale: int = 4,
    clip_size: int = 5,
    embed_dim: int = 120,
    num_groups: int = 4,
    depth_per_group: int = 3,
    num_heads: int = 4,
    window_size: int = 8,
    ffn_expansion_ratio: float = 2.0,
    aim_reduction_ratio: int = 8,
    group_block_pattern: list[str] | None = None,
    drop_path_rate: float = 0.1,
    mid_dim: int = 64,
    upsampler_type: SampleMods3 = "pixelshuffle",
    flow_base_ch: int = 32,
    flow_aux_weight: float = 0.1,
    flow_aux_smooth_weight: float = 0.1,
    flow_aux_anneal_iters: int = 100000,
) -> TFDAT:
    return TFDAT(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
        clip_size=clip_size,
        embed_dim=embed_dim,
        num_groups=num_groups,
        depth_per_group=depth_per_group,
        num_heads=num_heads,
        window_size=window_size,
        ffn_expansion_ratio=ffn_expansion_ratio,
        aim_reduction_ratio=aim_reduction_ratio,
        group_block_pattern=group_block_pattern,
        drop_path_rate=drop_path_rate,
        mid_dim=mid_dim,
        upsampler_type=upsampler_type,
        flow_base_ch=flow_base_ch,
        flow_aux_weight=flow_aux_weight,
        flow_aux_smooth_weight=flow_aux_smooth_weight,
        flow_aux_anneal_iters=flow_aux_anneal_iters,
    )
