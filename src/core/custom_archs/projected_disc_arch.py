"""Projected GAN discriminator: a small trainable classification head run on
top of multi-scale features extracted by a FROZEN, ImageNet-pretrained
backbone, instead of learning feature extraction from scratch.

Ported from the official repository (MIT license)
https://github.com/autonomousvision/projected-gan, which accompanies
"Projected GANs Converge Faster" (Sauer, Chen, Schwarz & Geiger, NeurIPS
2021, https://arxiv.org/pdf/2111.01007). Source files fetched and read
directly from ``pg_modules/`` (not reconstructed from the paper text):
``projector.py`` (``F_RandomProj`` / the "feature projector"),
``discriminator.py`` (``SingleDisc`` / ``MultiScaleD`` / the classification
head), and ``blocks.py`` (the conv/norm building blocks the head uses).
Everything ported is 100% plain PyTorch -- no compiled CUDA ops (unlike the
same repo's StyleGAN2/FastGAN *generators*, which depend on
``torch_utils/ops/``; the generator side is irrelevant here since only the
discriminator is being used, as traiNNer-redux brings its own generators).

What was ported vs. dropped from the reference, and why:
  * ``F_RandomProj`` -- ported faithfully. Feeds the input through a frozen
    ``timm`` ``tf_efficientnet_lite0`` (ImageNet-pretrained) split into 4
    stages (``layer0..layer3``), then optionally 1x1 "cross-channel mixing"
    (CCM) convs and/or a top-down "cross-scale mixing" (CSM) feature-pyramid
    pass (``FeatureFusionBlock``) -- this is the ``proj_type`` 0/1/2 switch
    in the paper. The backbone weights are frozen (``requires_grad=False``,
    ``eval()`` forced in ``.train()``); only the CCM/CSM convs and the
    classification head below are trained.
  * ``SingleDisc`` / ``MultiScaleD`` -- ported faithfully as the trainable
    classification head: a small stack of spectral-norm ``DownBlock``
    (optionally ``DownBlockPatch`` for a PatchGAN-style receptive field)
    per feature scale, each ending in a 4x4 conv to a single logit map;
    ``MultiScaleD`` runs one such mini-discriminator per selected backbone
    scale and concatenates their per-scale logits along dim 1.
  * ``SingleDiscCond`` / class-conditional projection discriminator (the
    ``cond=1`` path, using ``CCBN``/embedding/``c_dim``) -- DROPPED. It
    needs a class-label tensor ``c`` passed alongside the image
    (``forward(x, c)``); traiNNer-redux's training loop only ever calls
    ``net_d(x)`` with a single image tensor (see ``SRModel`` in
    ``traiNNer/models/sr_model.py``), so there is no class label to
    condition on. This port is therefore always unconditional
    (``cond=0``/``SingleDisc``, matching the paper's default unconditional
    setting for most datasets).
  * ``DiffAugment`` (``pg_modules/diffaug.py``, ``color,translation,cutout``
    policy) -- ported faithfully as an optional pre-backbone augmentation
    (``use_diffaug``), matching the reference's default-on behavior.
  * ``nn.quantized.FloatFunctional().add(a, b)`` calls inside
    ``FeatureFusionBlock``/``ResidualConvUnit`` in the reference -- replaced
    with plain ``a + b``. ``FloatFunctional`` only matters for quantization
    observer/dtype bookkeeping; in plain float32/float16 training (the only
    mode traiNNer-redux uses) it is behaviorally identical to a normal add.
    ``ResidualConvUnit`` itself is unused by the CSM path taken here (the
    reference only wires it into a DPT-style depth decoder, not
    ``F_RandomProj``) and was not ported.
  * Multiple named backbones (the paper explores other timm backbones for
    ablations) -- only ``tf_efficientnet_lite0`` was ported/verified, since
    that is the paper's actual default and the only one exercised in this
    integration.

HONESTY NOTE (generality vs. the other Redux discriminators): unlike
Spectral (DualFormer) or NeXtSRGAN, which are SR-specific discriminator
designs, Projected GAN is a *general-purpose* GAN-training technique (its
paper targets unconditional/class-conditional image synthesis benchmarks
such as FFHQ/LSUN/ImageNet, not super-resolution). Its frozen backbone was
pretrained for ImageNet classification, not for judging SR artifacts
specifically, so it may be less targeted at typical SR failure modes
(ringing, checkerboarding, texture hallucination) than a discriminator
built for SR. It is included here as a faster-converging, generalist
alternative -- not a like-for-like replacement for the SR-specific
discriminators.

NEW DEPENDENCY: this file requires ``timm`` (already listed in this
engine's ``pyproject.toml`` -- see ``[project].dependencies``) to load the
frozen ``tf_efficientnet_lite0`` backbone. The FIRST forward pass with a
given backbone downloads its ImageNet-pretrained weights (~15-20 MB) from
the timm/HuggingFace model hub into the local timm cache
(``~/.cache/huggingface`` or ``TORCH_HOME``/``TIMM_HOME``) -- this requires
an internet connection once; subsequent runs reuse the cached weights
offline.
"""

from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY

# ============================================================================
# blocks.py subset (conv/norm/CSM building blocks actually used by the
# discriminator path -- generator-only blocks such as GLU/Swish/SEBlock/
# UpBlock*/NoiseInjection/CCBN/InitLayer were not ported).
# ============================================================================


def _conv2d(*args: object, **kwargs: object) -> nn.Module:
    return spectral_norm(nn.Conv2d(*args, **kwargs))  # pyright: ignore[reportArgumentType]


def _norm_layer(c: int) -> nn.Module:
    # Reference default is mode="batch"; the "group" mode exists in the
    # reference but is never actually selected anywhere in the discriminator
    # path, so it was not exposed.
    return nn.BatchNorm2d(c)


class _SeparableConv2d(nn.Module):
    """Depthwise 3x3 (padding=1, fixed) + pointwise 1x1 conv, both spectral-normed."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = _conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=False,
            padding=1,
        )
        self.pointwise = _conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class _DownBlock(nn.Module):
    """4x4 stride-2 conv -> BatchNorm -> LeakyReLU(0.2), halving H and W.

    ``separable=True`` swaps the conv for a depthwise-separable 3x3 conv
    followed by 2x2 average pooling (reference's alternate, cheaper path).
    """

    def __init__(self, in_planes: int, out_planes: int, separable: bool = False) -> None:
        super().__init__()
        if not separable:
            self.main = nn.Sequential(
                _conv2d(in_planes, out_planes, 4, 2, 1),
                _norm_layer(out_planes),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            self.main = nn.Sequential(
                _SeparableConv2d(in_planes, out_planes, 3),
                _norm_layer(out_planes),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.AvgPool2d(2, 2),
            )

    def forward(self, feat: Tensor) -> Tensor:
        return self.main(feat)


class _DownBlockPatch(nn.Module):
    """``_DownBlock`` followed by an extra 1x1 conv -> BN -> LeakyReLU.

    Used when ``patch=True``: keeps the receptive field small/local
    (PatchGAN-style) instead of collapsing all the way to a global 8x8/4x4
    feature map before the final logit conv.
    """

    def __init__(self, in_planes: int, out_planes: int, separable: bool = False) -> None:
        super().__init__()
        self.main = nn.Sequential(
            _DownBlock(in_planes, out_planes, separable),
            _conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            _norm_layer(out_planes),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, feat: Tensor) -> Tensor:
        return self.main(feat)


class _FeatureFusionBlock(nn.Module):
    """Top-down feature-pyramid fusion step used by cross-scale mixing (CSM).

    Adds the (optional) coarser-scale input to the current-scale input,
    bilinearly upsamples 2x, then projects channels with a 1x1 conv. This is
    the "scale mixing" building block: chaining these top-to-bottom (see
    ``_ProjectedFeatureExtractor``) fuses coarse semantic features into
    finer spatial scales.
    """

    def __init__(self, features: int, expand: bool = False) -> None:
        super().__init__()
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        out = x if skip is None else x + skip
        out = nn.functional.interpolate(
            out, scale_factor=2, mode="bilinear", align_corners=True
        )
        return self.out_conv(out)


# ============================================================================
# projector.py port: frozen timm backbone + optional CCM/CSM projection head.
# ============================================================================


def _split_efficientnet_lite0(model: nn.Module) -> nn.Module:
    """Split a timm ``tf_efficientnet_lite0`` into 4 feature stages.

    Matches the reference's ``_make_efficientnet``: groups the model's
    ``blocks`` (a ``timm`` EfficientNet's MBConv stage list) into 4 chunks
    at fixed indices tuned for this specific backbone's stage boundaries.
    """
    # Reference (older timm) exposed a separate ``model.act1`` after
    # ``model.bn1``. Current timm fuses BN + activation into ``bn1``
    # (a ``BatchNormAct2d``) and no longer has a standalone ``act1``
    # attribute, so we only chain what actually exists on this backbone.
    stem: list[nn.Module] = [model.conv_stem, model.bn1]
    if hasattr(model, "act1"):
        stem.append(model.act1)

    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(*stem, *model.blocks[0:2])
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained


class _ProjectedFeatureExtractor(nn.Module):
    """Frozen ``tf_efficientnet_lite0`` backbone (+ optional CCM/CSM heads).

    Args mirror the reference ``F_RandomProj``:
        cout: Output channel width of the CCM/CSM projection convs
            (ignored if ``proj_type == 0``).
        expand: If True, CCM output channels grow ``cout, cout*2, cout*4,
            cout*8`` across the 4 scales instead of a flat ``cout``
            everywhere; CSM mirrors this on the way back down.
        proj_type: 0 = raw backbone features only (no projection), 1 =
            cross-channel mixing only (1x1 convs per scale), 2 = cross-scale
            mixing (adds the top-down feature-pyramid fusion on top of CCM).
        pretrained: Load ImageNet-pretrained backbone weights (downloaded
            via timm on first use). Disabling this defeats the entire point
            of Projected GAN (a strong pretrained prior is the paper's whole
            mechanism) -- exposed only for offline/CI testing, not
            recommended for real training.
    """

    def __init__(
        self,
        cout: int = 64,
        expand: bool = True,
        proj_type: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        assert proj_type in (0, 1, 2), "proj_type must be 0, 1, or 2"
        self.proj_type = proj_type

        import timm

        backbone = timm.create_model("tf_efficientnet_lite0", pretrained=pretrained)
        self.pretrained = _split_efficientnet_lite0(backbone)

        # Freeze the backbone -- only the CCM/CSM projection convs (if any)
        # and the classification head (built separately, see
        # ProjectedDiscriminator) are ever trained.
        for p in self.pretrained.parameters():
            p.requires_grad = False

        # Reference fixes the internal resolution assumption to 256
        # regardless of the caller-provided im_res (see _make_projector: "the
        # best results are achieved by fixing this to 256").
        im_res = 256
        self.resolutions = [im_res // 4, im_res // 8, im_res // 16, im_res // 32]
        self.channels = self._calc_channels()

        self.ccm: nn.ModuleDict | None = None
        self.csm: nn.ModuleDict | None = None
        if proj_type >= 1:
            out_channels = (
                [cout, cout * 2, cout * 4, cout * 8] if expand else [cout] * 4
            )
            self.ccm = nn.ModuleDict(
                {
                    f"layer{i}": nn.Conv2d(
                        self.channels[i], out_channels[i], kernel_size=1, bias=True
                    )
                    for i in range(4)
                }
            )
            self.channels = out_channels

        if proj_type == 2:
            assert self.csm is None
            c = self.channels
            self.csm = nn.ModuleDict(
                {
                    "layer3": _FeatureFusionBlock(c[3], expand=expand),
                    "layer2": _FeatureFusionBlock(c[2], expand=expand),
                    "layer1": _FeatureFusionBlock(c[1], expand=expand),
                    "layer0": _FeatureFusionBlock(c[0], expand=False),
                }
            )
            self.channels = [c[0], c[0], c[0] * 2, c[0] * 4] if expand else [cout] * 4
            # CSM upsamples 2x per fusion step, so the effective feature
            # resolution doubles relative to the raw CCM/backbone output.
            self.resolutions = [r * 2 for r in self.resolutions]

    def _calc_channels(self) -> list[int]:
        import torch

        with torch.no_grad():
            was_training = self.pretrained.training
            self.pretrained.eval()
            tmp = torch.zeros(1, 3, 224, 224)
            channels = []
            tmp = self.pretrained.layer0(tmp)
            channels.append(tmp.shape[1])
            tmp = self.pretrained.layer1(tmp)
            channels.append(tmp.shape[1])
            tmp = self.pretrained.layer2(tmp)
            channels.append(tmp.shape[1])
            tmp = self.pretrained.layer3(tmp)
            channels.append(tmp.shape[1])
            self.pretrained.train(was_training)
        return channels

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        out0 = self.pretrained.layer0(x)
        out1 = self.pretrained.layer1(out0)
        out2 = self.pretrained.layer2(out1)
        out3 = self.pretrained.layer3(out2)
        out = {"0": out0, "1": out1, "2": out2, "3": out3}

        if self.proj_type == 0:
            return out

        assert self.ccm is not None
        out = {
            "0": self.ccm["layer0"](out["0"]),
            "1": self.ccm["layer1"](out["1"]),
            "2": self.ccm["layer2"](out["2"]),
            "3": self.ccm["layer3"](out["3"]),
        }

        if self.proj_type == 1:
            return out

        assert self.csm is not None
        out3_mixed = self.csm["layer3"](out["3"])
        out2_mixed = self.csm["layer2"](out3_mixed, out["2"])
        out1_mixed = self.csm["layer1"](out2_mixed, out["1"])
        out0_mixed = self.csm["layer0"](out1_mixed, out["0"])
        return {"0": out0_mixed, "1": out1_mixed, "2": out2_mixed, "3": out3_mixed}


# ============================================================================
# discriminator.py port: unconditional classification head (SingleDisc /
# MultiScaleD). SingleDiscCond / CCBN-based conditioning were dropped (see
# module docstring).
# ============================================================================

_CHANNEL_DICT = {
    4: 512,
    8: 512,
    16: 256,
    32: 128,
    64: 64,
    128: 64,
    256: 32,
    512: 16,
    1024: 8,
}


class _SingleDisc(nn.Module):
    """One mini-discriminator: downsamples ``start_sz -> end_sz`` then a 4x4->1 conv."""

    def __init__(
        self,
        nc: int,
        start_sz: int,
        end_sz: int = 8,
        separable: bool = False,
        patch: bool = False,
    ) -> None:
        super().__init__()
        # interpolate for start sizes that are not powers of two
        if start_sz not in _CHANNEL_DICT:
            sizes = sorted(_CHANNEL_DICT.keys())
            start_sz = min(sizes, key=lambda s: abs(s - start_sz))

        nfc = dict(_CHANNEL_DICT)
        nfc[start_sz] = nc

        layers: list[nn.Module] = []
        db_cls = _DownBlockPatch if patch else _DownBlock
        sz = start_sz
        while sz > end_sz:
            layers.append(db_cls(nfc[sz], nfc[sz // 2], separable))
            sz = sz // 2

        layers.append(_conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class _MultiScaleD(nn.Module):
    """Runs one ``_SingleDisc`` per selected backbone scale, concatenates logits."""

    def __init__(
        self,
        channels: list[int],
        resolutions: list[int],
        num_discs: int = 4,
        separable: bool = False,
        patch: bool = False,
    ) -> None:
        super().__init__()
        assert num_discs in (1, 2, 3, 4)

        disc_in_channels = channels[:num_discs]
        disc_in_res = resolutions[:num_discs]

        mini_discs = {}
        for i, (cin, res) in enumerate(zip(disc_in_channels, disc_in_res, strict=True)):
            start_sz = 16 if patch else res
            mini_discs[str(i)] = _SingleDisc(
                nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch
            )
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features: dict[str, Tensor]) -> Tensor:
        import torch

        all_logits = [
            disc(features[k]).view(features[k].size(0), -1)
            for k, disc in self.mini_discs.items()
        ]
        return torch.cat(all_logits, dim=1)


# ============================================================================
# diffaug.py port (color,translation,cutout policy only -- the only policy
# the reference ever actually applies by default).
# ============================================================================


def _diffaugment(x: Tensor) -> Tensor:
    import torch

    # color
    x = x + (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    )
    x_mean = x.mean(dim=1, keepdim=True)
    x = (
        x - x_mean
    ) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (
        x - x_mean
    ) * (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    ) + x_mean

    # translation
    ratio = 0.125
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(
        -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
    )
    translation_y = torch.randint(
        -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = nn.functional.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(
        0, 3, 1, 2
    )

    # cutout
    ratio = 0.2
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(
        0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    offset_y = torch.randint(
        0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)

    return x.contiguous()


# ============================================================================
# Public arch: frozen feature extractor + trainable multi-scale head.
# ============================================================================


@ARCH_REGISTRY.register()
class ProjectedDiscriminator(nn.Module):
    """Projected GAN discriminator (see module docstring for the full port notes).

    Args:
        cout: Channel width for the CCM/CSM projection convs (only used
            when ``proj_type >= 1``). Paper/reference default: 64.
        expand: Grow CCM/CSM channel width across scales (``cout, 2*cout,
            4*cout, 8*cout``) instead of a flat ``cout`` at every scale.
            Reference default: True.
        proj_type: Feature projection mode -- "none" (raw backbone features,
            fastest, weakest), "ccm" (1x1 cross-channel mixing per scale),
            or "csm" (adds cross-scale/top-down feature-pyramid fusion on
            top of ccm -- reference default, empirically the strongest).
        num_discs: How many of the 4 backbone scales get their own
            mini-discriminator (1 = coarsest/deepest scale only, 4 = all).
            Reference default: 4 (technically the reference's own
            ``MultiScaleD`` example configs mostly use up to 3, but 4 is the
            architecture's supported maximum and this port's default to
            keep every backbone scale supervised).
        separable: Use depthwise-separable convs + avgpool instead of plain
            strided convs in the downsampling head (cheaper, reference's
            optional lightweight mode). Reference default: False.
        patch: Use PatchGAN-style local downblocks (stops downsampling
            earlier per scale, larger relative receptive field per
            spatial position) instead of collapsing to a single global
            decision per scale. Reference default: False.
        use_diffaug: Apply the reference's differentiable augmentation
            (random brightness/saturation/contrast + translation + cutout)
            to the input before the backbone -- data-efficient GAN training
            trick from Zhao et al., helps prevent discriminator overfitting
            on small datasets. Reference default: True.
        interp224: Resize the input to 224x224 before the backbone (the
            backbone/paper's native resolution). Reference default: True;
            disabling risks feeding the frozen ImageNet backbone
            out-of-distribution input sizes.
        pretrained_backbone: Load ImageNet-pretrained backbone weights
            (downloaded via timm on first use, ~15-20 MB, requires internet
            the first time). Disabling defeats the entire point of
            Projected GAN -- exposed only for offline/CI smoke testing.
    """

    def __init__(
        self,
        cout: int = 64,
        expand: bool = True,
        proj_type: str = "csm",
        num_discs: int = 4,
        separable: bool = False,
        patch: bool = False,
        use_diffaug: bool = True,
        interp224: bool = True,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()

        proj_type_map = {"none": 0, "ccm": 1, "csm": 2}
        if proj_type not in proj_type_map:
            raise ValueError(
                f"proj_type must be one of {sorted(proj_type_map)}, got {proj_type!r}"
            )

        self.use_diffaug = use_diffaug
        self.interp224 = interp224

        self.feature_network = _ProjectedFeatureExtractor(
            cout=cout,
            expand=expand,
            proj_type=proj_type_map[proj_type],
            pretrained=pretrained_backbone,
        )
        self.discriminator = _MultiScaleD(
            channels=self.feature_network.channels,
            resolutions=self.feature_network.resolutions,
            num_discs=num_discs,
            separable=separable,
            patch=patch,
        )

    def train(self, mode: bool = True) -> "ProjectedDiscriminator":
        # The backbone is always frozen/eval -- only the CCM/CSM projection
        # convs (part of feature_network but NOT the backbone itself) and
        # the classification head are ever trained. Matches the reference's
        # ProjectedDiscriminator.train() override.
        self.feature_network.pretrained.eval()
        modules_to_train = [self.discriminator]
        if self.feature_network.ccm is not None:
            modules_to_train.append(self.feature_network.ccm)
        if self.feature_network.csm is not None:
            modules_to_train.append(self.feature_network.csm)
        for m in modules_to_train:
            m.train(mode)
        return self

    def eval(self) -> "ProjectedDiscriminator":
        return self.train(False)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_diffaug:
            x = _diffaugment(x)
        if self.interp224:
            x = nn.functional.interpolate(
                x, size=224, mode="bilinear", align_corners=False
            )
        features = self.feature_network(x)
        return self.discriminator(features)
