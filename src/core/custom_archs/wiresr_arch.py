# WireSR v2 — thin-structure-oriented SR / anti-aliasing network, by CF2lter
# (model release "1x_Wired", Enhance Everything! Discord). Source: the author's
# spandrel2.zip (Google Drive folder 1-eHwEveHLhP4yhU1rbjCJD3XuJWuiTU4), fetched 2026-10-16.
# LICENSE: included with the author's permission (granted 2026-10-26), with credit to
# CF2lter — see licenses/WireSR-PERMISSION.txt. The 1x_Wired *weights* stay
# CC BY-NC-SA 4.0 (not bundled).
# Changes (Universal SR Studio): the relative spandrel import of DySample is made
# absolute; the traiNNer registration factory `wiresr` is appended at the end.
"""
WireSR v2: thin-structure-oriented SR / anti-aliasing network.

"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys  # type: ignore

from spandrel.__helpers.model_descriptor import StateDict
from spandrel.util import store_hyperparameters
from spandrel.util.timm import DropPath, trunc_normal_

from spandrel.architectures.__arch_helpers.dysample import DySample

SampleMods = Literal[
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
]

SampleMods3 = Literal[SampleMods, "transpose+conv", "lda", "pa_up"]


class LayerNorm(nn.Module):
    """Channels-first LayerNorm, used by LDA_AQU."""

    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x):
        if x.is_contiguous(memory_format=torch.channels_last):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.dim, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LDA_AQU(nn.Module):
    def __init__(
        self,
        in_channels=48,
        reduction_factor=4,
        nh=1,
        scale_factor=2.0,
        k_e=3,
        k_u=3,
        n_groups=2,
        range_factor=11,
        rpb=True,
    ) -> None:
        super().__init__()
        self.k_u = k_u
        self.num_head = nh
        self.scale_factor = scale_factor
        self.n_groups = n_groups
        self.offset_range_factor = range_factor

        self.attn_dim = in_channels // (reduction_factor * self.num_head)
        self.scale = self.attn_dim**-0.5
        self.rpb = rpb
        self.hidden_dim = in_channels // reduction_factor
        self.proj_q = nn.Conv2d(
            in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.proj_k = nn.Conv2d(
            in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.group_channel = in_channels // (reduction_factor * self.n_groups)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.group_channel,
                self.group_channel,
                3,
                1,
                1,
                groups=self.group_channel,
                bias=False,
            ),
            LayerNorm(self.group_channel),
            nn.SiLU(),
            nn.Conv2d(self.group_channel, 2 * k_u**2, k_e, 1, k_e // 2),
        )
        self.layer_norm = LayerNorm(in_channels)

        self.pad = int((self.k_u - 1) / 2)
        base = np.arange(-self.pad, self.pad + 1).astype(np.float32)
        base_y = np.repeat(base, self.k_u)
        base_x = np.tile(base, self.k_u)
        base_offset = np.stack([base_y, base_x], axis=1).flatten()
        base_offset = torch.tensor(base_offset).view(1, -1, 1, 1)
        self.register_buffer("base_offset", base_offset, persistent=False)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    1, self.num_head, 1, self.k_u**2, self.hidden_dim // self.num_head
                )
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def get_offset(self, offset, Hout, Wout):
        B, _, _, _ = offset.shape
        device = offset.device
        row_indices = torch.arange(Hout, device=device)
        col_indices = torch.arange(Wout, device=device)
        row_indices, col_indices = torch.meshgrid(row_indices, col_indices)
        index_tensor = torch.stack((row_indices, col_indices), dim=-1).view(
            1, Hout, Wout, 2
        )
        offset = rearrange(
            offset, "b (kh kw d) h w -> b kh h kw w d", kh=self.k_u, kw=self.k_u
        )
        offset = offset + index_tensor.view(1, 1, Hout, 1, Wout, 2)
        offset = offset.contiguous().view(B, self.k_u * Hout, self.k_u * Wout, 2)

        offset[..., 0] = 2 * offset[..., 0] / (Hout - 1) - 1
        offset[..., 1] = 2 * offset[..., 1] / (Wout - 1) - 1
        offset = offset.flip(-1)
        return offset

    def extract_feats(self, x, offset, ks=3):
        out = nn.functional.grid_sample(
            x,
            offset,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        out = rearrange(out, "b c (ksh h) (ksw w) -> b (ksh ksw) c h w", ksh=ks, ksw=ks)
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        out_H, out_W = int(H * self.scale_factor), int(W * self.scale_factor)
        v = x
        x = self.layer_norm(x)
        q = self.proj_q(x)
        k = self.proj_k(x)

        q = torch.nn.functional.interpolate(
            q, (out_H, out_W), mode="bilinear", align_corners=True
        )
        q_off = q.view(B * self.n_groups, -1, out_H, out_W)
        pred_offset = self.conv_offset(q_off)
        offset = pred_offset.tanh().mul(self.offset_range_factor) + self.base_offset.to(
            x.dtype
        )

        k = k.view(B * self.n_groups, self.hidden_dim // self.n_groups, H, W)
        v = v.view(B * self.n_groups, C // self.n_groups, H, W)
        offset = self.get_offset(offset, out_H, out_W)
        k = self.extract_feats(k, offset=offset)
        v = self.extract_feats(v, offset=offset)

        q = rearrange(q, "b (nh c) h w -> b nh (h w) () c", nh=self.num_head)
        k = rearrange(k, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        v = rearrange(v, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        k = rearrange(k, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)
        v = rearrange(v, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)

        if self.rpb:
            k = k + self.relative_position_bias_table

        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, "b nh (h w) t c -> b (nh c) (t h) w", h=out_H)
        return out


class PA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x):
        return x.mul(self.conv(x))


class UniUpsampleV3(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods3 = "pa_up",
        scale: int = 2,
        in_dim: int = 48,
        out_dim: int = 3,
        mid_dim: int = 48,
        group: int = 4,  # Only DySample
    ) -> None:
        m: list[nn.Module] = []

        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend(
                [nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)]
            )
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)]
                    )
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
            m.append(DySample(mid_dim, out_dim, scale, group))
        elif upsample == "transpose+conv":
            if scale == 2:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1))
            elif scale == 3:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 3, 3, 0))
            elif scale == 4:
                m.extend(
                    [
                        nn.ConvTranspose2d(in_dim, in_dim, 4, 2, 1),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2, 3, 4"
                )
            m.append(nn.Conv2d(out_dim, out_dim, 3, 1, 1))
        elif upsample == "lda":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
            m.append(LDA_AQU(mid_dim, scale_factor=scale))
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "pa_up":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                            PA(mid_dim),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ]
                    )
                    in_dim = mid_dim
            elif scale == 3:
                m.extend(
                    [
                        nn.Upsample(scale_factor=3),
                        nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                        PA(mid_dim),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        else:
            raise ValueError(
                f"An invalid Upsample was selected. Please choose one of {SampleMods3}"
            )
        super().__init__(*m)

        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    3,  # Block version, if you change something, please number from the end so that you can distinguish between authorized changes and third parties
                    list(SampleMods3.__args__).index(upsample),  # UpSample method index
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )


# --- WireSR components ---


class RMSNorm(nn.Module):
    """Channels-first RMSNorm. Statistics are computed in fp32 and only the
    gain is cast back, so it is safe under pure fp16. No mean subtraction and
    no variance division -> flat regions stay flat."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, 1, 1))
        self.offset = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=1, keepdim=True) + self.eps)
        return self.scale * (x * inv_rms.to(x.dtype)) + self.offset


class DiagConv(nn.Module):
    """Depthwise stripe along the main or anti diagonal: a k x k kernel with
    all off-diagonal taps masked to zero. Gives diagonal lines the same long
    collinear context that h/v stripes give flat ones. Zero-padded (no
    wrap-around, unlike roll/shift tricks) and export-safe (the mask is a
    plain multiply; masked taps get zero grad, weight decay is harmless)."""

    def __init__(self, dim: int, kernel: int = 17, flip: bool = False) -> None:
        super().__init__()
        assert kernel % 2 == 1, "DiagConv kernel must be odd"
        self.kernel = kernel
        width = torch.empty(dim, 1, kernel, kernel)
        self.weight = nn.Parameter(width)
        trunc_normal_(self.weight, std=0.02)
        mask = torch.zeros(1, 1, kernel, kernel)
        rows = list(range(kernel))
        cols = list(range(kernel - 1, -1, -1)) if flip else rows
        mask[0, 0, rows, cols] = 1.0
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            x,
            self.weight * self.mask,
            padding=self.kernel // 2,
            groups=x.size(1),
        )


class StripeConv(nn.Module):
    """Six-way criss-cross depthwise mixing on a channel split:
    horizontal stripe (1xK), vertical stripe (Kx1), main diagonal, anti
    diagonal, 2D square (kxk: arbitrary orientations, junctions, curves),
    identity (features that pass through untouched).
    Elongated structures get ~K px of collinear context per layer in all four
    cardinal orientations."""

    def __init__(
        self,
        dim: int,
        stripe_kernel: int = 61,
        diag_kernel: int = 17,
        square_kernel: int = 17,
    ) -> None:
        super().__init__()
        assert dim % 8 == 0, "StripeConv dim must be divisible by 8"
        assert stripe_kernel % 2 == 1 and square_kernel % 2 == 1
        q, e = dim // 4, dim // 8
        self.split = [q, q, e, e, e, dim - 2 * q - 3 * e]
        p, ps = stripe_kernel // 2, square_kernel // 2
        self.stripe_h = nn.Conv2d(
            q, q, (1, stripe_kernel), padding=(0, p), groups=q, bias=False
        )
        self.stripe_v = nn.Conv2d(
            q, q, (stripe_kernel, 1), padding=(p, 0), groups=q, bias=False
        )
        self.diag_m = DiagConv(e, diag_kernel, flip=False)
        self.diag_a = DiagConv(e, diag_kernel, flip=True)
        self.square = nn.Conv2d(e, e, square_kernel, padding=ps, groups=e, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        xh, xv, xdm, xda, xs, xi = torch.split(x, self.split, dim=1)
        return torch.cat(
            [
                self.stripe_h(xh),
                self.stripe_v(xv),
                self.diag_m(xdm),
                self.diag_a(xda),
                self.square(xs),
                xi,
            ],
            dim=1,
        )


class GlobalStripeGate(nn.Module):
    """Row/column global context gate.
    A 2px line contributes ~0 to an average but dominates a max over its
    row/column, so max-pooled stripe descriptors give global line awareness
    without any attention windows."""

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(dim // reduction, 8)
        self.row_mlp = nn.Sequential(
            nn.Conv2d(dim * 2, mid, 1, bias=False),
            nn.Mish(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self.col_mlp = nn.Sequential(
            nn.Conv2d(dim * 2, mid, 1, bias=False),
            nn.Mish(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        row = torch.cat(
            [x.amax(dim=3, keepdim=True), x.mean(dim=3, keepdim=True)], dim=1
        )  # (B, 2C, H, 1)
        col = torch.cat(
            [x.amax(dim=2, keepdim=True), x.mean(dim=2, keepdim=True)], dim=1
        )  # (B, 2C, 1, W)
        return x * self.row_mlp(row) * self.col_mlp(col)


class ChannelAttn(nn.Module):
    """XCiT/FDAT-style channel attention -- global periodic-pattern reasoning
    (chain-link moire suppression). Cheap: attn map is (C/h) x (C/h).
    Computed in fp32 for softmax stability under pure-fp16 inference."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.heads = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x).float().reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.unbind(dim=1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temp.float()).softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w).to(x.dtype)
        return self.proj(out)


class ConvFFN(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.Mish(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Mish(inplace=True),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EA(nn.Module):
    """Element-wise attention (RealPLKSR), slimmed: dw3x3 + 1x1 instead of a
    dense 3x3. ~8x cheaper, keeps spatial gating (the stripe mixer / FFN
    already provide dense spatial context)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.f(x)


class WireBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        stripe_kernel: int,
        diag_kernel: int,
        square_kernel: int,
        expansion: float,
        use_ca: bool,
        ca_heads: int,
        drop_path: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mix = StripeConv(dim, stripe_kernel, diag_kernel, square_kernel)
        self.fuse = nn.Conv2d(dim, dim, 1)
        self.gate = GlobalStripeGate(dim)

        self.ca = ChannelAttn(dim, ca_heads) if use_ca else None
        self.norm_ca = RMSNorm(dim) if use_ca else None

        self.norm2 = RMSNorm(dim)
        self.ffn = ConvFFN(dim, expansion)
        self.ea = EA(dim)

        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))
        self.gamma3 = (
            nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))
            if use_ca
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # stripe mixing (h/v/diag/square/identity) + global row/col gate
        x = x + self.drop_path(
            self.gamma1 * self.gate(self.fuse(self.mix(self.norm1(x))))
        )
        # optional channel attention (moire suppression)
        if self.ca is not None:
            x = x + self.drop_path(self.gamma3 * self.ca(self.norm_ca(x)))
        # conv FFN with element-wise edge gate
        f = self.ffn(self.norm2(x))
        x = x + self.drop_path(self.gamma2 * (f * self.ea(f)))
        return x


@store_hyperparameters()
class WireSR(nn.Module):
    """WireSR v2 -- thin-structure-oriented SR / anti-aliasing network."""

    hyperparameters = {}

    def __init__(
        self,
        *,
        scale: int = 4,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 24,
        stripe_kernel: int = 61,
        diag_kernel: int = 17,
        square_kernel: int = 17,
        expansion: float = 2.0,
        attn_every: int = 5,
        ca_heads: int = 4,
        drop_path_rate: float = 0.0,
        layer_scale_init: float = 1e-2,
        upsampler: SampleMods3 = "pixelshuffle",
        mid_dim: int = 64,
        unshuffle_mod: bool = False,  # keep False for AA! unshuffling destroys 1px lines
    ) -> None:
        super().__init__()
        self.scale = scale
        self.pad = 0
        internal_scale = scale

        if unshuffle_mod and scale <= 2:
            unshuffle = 4 // scale
            internal_scale = 4
            self.pad = unshuffle
            self.conv_first = nn.Sequential(
                nn.PixelUnshuffle(unshuffle),
                nn.Conv2d(in_ch * unshuffle**2, dim, 3, 1, 1),
            )
        else:
            self.conv_first = nn.Conv2d(in_ch, dim, 3, 1, 1)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        self.blocks = nn.Sequential(
            *[
                WireBlock(
                    dim=dim,
                    stripe_kernel=stripe_kernel,
                    diag_kernel=diag_kernel,
                    square_kernel=square_kernel,
                    expansion=expansion,
                    use_ca=attn_every > 0 and (i % attn_every == attn_every - 1),
                    ca_heads=ca_heads,
                    drop_path=dpr[i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(n_blocks)
            ]
        )
        self.conv_after = nn.Conv2d(dim, dim, 3, 1, 1)

        upsampler_type: SampleMods3 = "conv" if internal_scale == 1 else upsampler
        self.upsampler = UniUpsampleV3(
            upsampler_type, internal_scale, dim, out_ch, mid_dim, 4
        )

        # global residual: the model only learns the correction (great for AA stability)
        self.short: nn.Module = (
            nn.Upsample(scale_factor=scale, mode="bilinear")
            if scale > 1
            else nn.Identity()
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        # DiagConv initializes its own masked weight in __init__
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def load_state_dict(
        self,
        state_dict: StateDict,
        *args,  # noqa: ANN002
        **kwargs,
    ) -> _IncompatibleKeys:
        state_dict["upsampler.MetaUpsample"] = self.upsampler.MetaUpsample
        return super().load_state_dict(state_dict, *args, **kwargs)

    def check_img_size(self, x: Tensor, h: int, w: int) -> Tensor:
        if self.pad == 0:
            return x
        mod_pad_h = (self.pad - h % self.pad) % self.pad
        mod_pad_w = (self.pad - w % self.pad) % self.pad
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x: Tensor) -> Tensor:
        _b, _c, h, w = x.shape
        x = self.check_img_size(x, h, w)
        feat = self.conv_first(x)
        out = self.conv_after(self.blocks(feat)) + feat
        out = self.upsampler(out) + self.short(x)
        if self.pad:
            out = out[:, :, : h * self.scale, : w * self.scale]
        return out

# ── Universal SR Studio: traiNNer-redux registration ─────────────────────────────
try:
    from traiNNer.utils.registry import ARCH_REGISTRY
    _REG = ARCH_REGISTRY.register()
except Exception:  # plain import (inference runners) without traiNNer
    def _REG(f):
        return f


@_REG
def wiresr(
    scale: int = 4,
    dim: int = 64,
    n_blocks: int = 24,
    stripe_kernel: int = 61,
    diag_kernel: int = 17,
    square_kernel: int = 17,
    expansion: float = 2.0,
    attn_every: int = 5,
    ca_heads: int = 4,
    drop_path_rate: float = 0.0,
    layer_scale_init: float = 1e-2,
    upsampler: str = "pixelshuffle",
    mid_dim: int = 64,
    unshuffle_mod: bool = False,
    in_ch: int = 3,
    out_ch: int = 3,
) -> WireSR:
    return WireSR(
        scale=scale, in_ch=in_ch, out_ch=out_ch, dim=dim, n_blocks=n_blocks,
        stripe_kernel=stripe_kernel, diag_kernel=diag_kernel, square_kernel=square_kernel,
        expansion=expansion, attn_every=attn_every, ca_heads=ca_heads,
        drop_path_rate=drop_path_rate, layer_scale_init=layer_scale_init,
        upsampler=upsampler, mid_dim=mid_dim, unshuffle_mod=unshuffle_mod,
    )
