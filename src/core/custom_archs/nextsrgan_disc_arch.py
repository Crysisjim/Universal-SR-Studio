"""NeXtSRGAN discriminator ("ConvNeXt-based discriminator" per the paper).

Ported from the official TensorFlow implementation
(https://github.com/PomKlementieff/NeXtSRGAN — note: that repo has NO license file,
so this port is listed as "permission pending" in THIRD_PARTY_NOTICES.md), function
``DiscriminatorVGG128`` in ``modules/nextsrgan.py``, which accompanies
"NeXtSRGAN: single image super-resolution based on the enhanced generative
adversarial network with ConvNeXt" (Klementieff, Kim & Han, The Visual
Computer, 2024, https://link.springer.com/article/10.1007/s00371-024-03797-2).

Architecture honesty note: despite being called "ConvNeXt-based" (paper
Fig. 6(a)), the *verified* reference TF source (fetched directly from the
repo, not inferred from the paper text) is architecturally a standard
ESRGAN/BasicSR-style VGG128 discriminator stack -- 5 stages of
[3x3 stride-1 conv, 4x4 stride-2 conv, BatchNorm], doubling the channel
width up to ``num_feat * 8`` -- with exactly two departures from the
classic VGG128 discriminator (compare ``VGGStyleDiscriminator`` in
``discriminator_arch.py``):
  1. GELU activations in stages 1-4 instead of LeakyReLU (the paper's nod
     to ConvNeXt, which favors GELU over ReLU-family activations).
  2. A ``GlobalAveragePooling2D`` + single ``Dense(1)`` head instead of
     ``Flatten`` + 2 FC layers (100 -> 1), removing the fixed-128x128-input
     constraint.
It does NOT use any of the actual ConvNeXt building blocks: no patchify
stem, no depthwise/grouped convolutions, no LayerNorm, no per-stage
variable depth or stochastic depth, no inverted bottleneck / 4x MLP
expansion. This port is intentionally faithful to what the reference TF
code actually computes, not to what the "ConvNeXt-based" name implies.

Stage-by-stage structure of the reference (also reproduced 1:1 below,
including the fact that stage 0 has no activation between its two convs,
unlike stages 1-4):
  stage0: conv3x3 s1 (bias) -> num_feat      | conv4x4 s2 (no bias) -> num_feat      | BN
  stage1: conv3x3 s1 (no bias) -> num_feat*2 | GELU | conv4x4 s2 (no bias) -> num_feat*2 | BN
  stage2: conv3x3 s1 (no bias) -> num_feat*4 | GELU | conv4x4 s2 (no bias) -> num_feat*4 | BN
  stage3: conv3x3 s1 (no bias) -> num_feat*8 | GELU | conv4x4 s2 (no bias) -> num_feat*8 | BN
  stage4: conv3x3 s1 (no bias) -> num_feat*8 | GELU | conv4x4 s2 (no bias) -> num_feat*8 | BN
  head:   GlobalAveragePool2d -> Linear(num_feat*8, 1)

No pretrained weights exist for this discriminator (the TF repo ships
none, and a GAN discriminator is always trained from scratch together with
its generator), so there is no weight-portability concern here -- only the
architecture is being replicated.
"""

from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from traiNNer.archs.arch_util import default_init_weights
from traiNNer.utils.registry import ARCH_REGISTRY


class _NeXtSRGANDiscStage(nn.Module):
    """One discriminator stage: conv3x3 s1 [-> GELU] -> conv4x4 s2 -> BN.

    ``use_activation=False`` reproduces the reference's stage 0, which has
    no activation between its two convs (all other stages set it True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_activation: bool,
        first_conv_bias: bool,
        norm: nn.Module,
    ) -> None:
        super().__init__()
        self.conv_s1 = norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=first_conv_bias,
            )
        )
        self.act = nn.GELU() if use_activation else nn.Identity()
        self.conv_s2 = norm(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        # BatchNorm is skipped entirely when spectral norm is requested,
        # matching how spectral_disc_arch.py treats normalization choices
        # as mutually exclusive rather than stacking BN with SN.
        self.bn = nn.Identity() if norm is spectral_norm else nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_s1(x)
        x = self.act(x)
        x = self.conv_s2(x)
        return self.bn(x)


@ARCH_REGISTRY.register()
class NeXtSRGANDiscriminator(nn.Module):
    """NeXtSRGAN's "ConvNeXt-based" VGG128-style discriminator (see module docstring).

    Args:
        num_in_ch: Input channel count (3 for RGB).
        num_feat: Base channel width of the conv stack (paper/reference default: 64).
        use_msra_init: MSRA/Kaiming init for conv & linear layers (traiNNer convention;
            the reference TF code instead uses scaled Glorot/VarianceScaling init).
        use_spectral_norm: Wrap every conv/linear layer with spectral norm and drop
            BatchNorm (the reference uses plain BatchNorm with no spectral norm;
            this is an opt-in traiNNer extra, consistent with spectral_disc_arch.py).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_feat: int = 64,
        use_msra_init: bool = True,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        norm: nn.Module = spectral_norm if use_spectral_norm else nn.Identity()

        self.stage0 = _NeXtSRGANDiscStage(
            num_in_ch, num_feat, use_activation=False, first_conv_bias=True, norm=norm
        )
        self.stage1 = _NeXtSRGANDiscStage(
            num_feat, num_feat * 2, use_activation=True, first_conv_bias=False, norm=norm
        )
        self.stage2 = _NeXtSRGANDiscStage(
            num_feat * 2, num_feat * 4, use_activation=True, first_conv_bias=False, norm=norm
        )
        self.stage3 = _NeXtSRGANDiscStage(
            num_feat * 4, num_feat * 8, use_activation=True, first_conv_bias=False, norm=norm
        )
        self.stage4 = _NeXtSRGANDiscStage(
            num_feat * 8, num_feat * 8, use_activation=True, first_conv_bias=False, norm=norm
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = norm(nn.Linear(num_feat * 8, 1))

        if use_msra_init:
            default_init_weights(
                [
                    self.stage0,
                    self.stage1,
                    self.stage2,
                    self.stage3,
                    self.stage4,
                    self.linear,
                ]
            )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.stage0(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.pool(feat).flatten(1)
        return self.linear(feat)
