"""Spectral discriminator: a VGG-style discriminator run on the Fourier
spectrum of the input instead of the raw spatial image.

Ported/adapted from the official DualFormer repository
(https://github.com/Luciennnnnnn/DualFormer, MIT license), source file
``basicsr/archs/discriminator2_arch.py`` (class ``VggDiscriminatorYN``),
which accompanies "On the Effectiveness of Spectral Discriminators for
Perceptual Quality Improvement" (ICCV 2023, https://arxiv.org/abs/2307.12027).

The paper's own VGG-style discriminator body (``_VggBlockDiscriminator``
below) is not a novel network — traiNNer already has an equivalent
(``VGGStyleDiscriminator`` in ``discriminator_arch.py``). The actual
contribution being ported is the *spectral preprocessing*
(``traiNNer.utils.spectrum.spectrum_transform``) applied ahead of it, so
this file wires that preprocessing into a discriminator via
``SpectralDiscriminator`` — the class actually meant to be selected as
``network_d`` for GAN training.

Per the paper, the best results come from running this spectral
discriminator *simultaneously* with a normal spatial-domain discriminator
(their losses combined) — not from replacing the spatial discriminator
with this one. traiNNer-redux's training loop only supports a single
``network_d``, so this is currently only usable as a (single)
discriminator on its own, which is a meaningfully weaker setup than the
paper's dual-discriminator recipe. See the ``spectral_disc`` tooltip in
the Studio UI for the same caveat.
"""

import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from traiNNer.archs.arch_util import default_init_weights
from traiNNer.utils.registry import ARCH_REGISTRY
from traiNNer.utils.spectrum import spectrum_transform


def _discriminator_block(
    in_channels: int, out_channels: int, norm: nn.Module
) -> list[nn.Module]:
    """One VGG-style block: 2 convs (stride 1 then 2) halving H and W."""
    return [
        norm(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            )
        ),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        norm(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True
            )
        ),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    ]


class _VggBlockDiscriminator(nn.Module):
    """VGG-style conv stack + adaptive pooling + 2 FC layers -> 1 logit.

    Unlike traiNNer's existing ``VGGStyleDiscriminator``, this uses no
    BatchNorm (matches the reference: BN is avoided for WGAN-GP
    compatibility) and adaptive pooling to a fixed 4x4 grid before the FC
    head, so it accepts any input spatial size divisible enough to survive
    4 stride-2 downsamples (>= ~16px), not just a fixed 128/256.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_feat: int = 64,
        use_msra_init: bool = True,
        pool: str = "avg",
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        norm: nn.Module = spectral_norm if use_spectral_norm else nn.Identity()

        layers: list[nn.Module] = []
        layers.extend(_discriminator_block(num_in_ch, num_feat, norm))
        layers.extend(_discriminator_block(num_feat, num_feat * 2, norm))
        layers.extend(_discriminator_block(num_feat * 2, num_feat * 4, norm))
        layers.extend(_discriminator_block(num_feat * 4, num_feat * 8, norm))

        if pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d(output_size=(4, 4)))
        elif pool == "max":
            layers.append(nn.AdaptiveMaxPool2d(output_size=(4, 4)))
        else:
            raise NotImplementedError("pool must be one of: avg, max")

        layers.extend(
            [
                nn.Flatten(),
                norm(nn.Linear(num_feat * 8 * 4 * 4, 100)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                norm(nn.Linear(100, 1)),
            ]
        )

        if use_msra_init:
            default_init_weights(layers)  # pyright: ignore[reportArgumentType]

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


@ARCH_REGISTRY.register()
class SpectralDiscriminator(nn.Module):
    """VGG-style discriminator applied to the Fourier spectrum of the input.

    Forward pass: ``spectrum_transform`` turns the input image into a
    (patch-wise) normalized power/magnitude spectrum with the same spatial
    resolution, then a VGG-style conv stack scores it — see module
    docstring for the paper reference and the "use alongside a spatial
    discriminator" caveat.

    Args:
        num_in_ch: Input channel count (3 for RGB); becomes 1 if
            ``reduction="gray"`` or ``reduction="mean"``, or is doubled if
            ``spec_type="full"`` (magnitude + phase channels concatenated) —
            handled automatically, no need to change this manually except
            for non-RGB inputs.
        num_feat: Base channel width of the conv stack.
        patch_size: Side length of the square patches the FFT is computed
            on. The input H and W must be divisible by this. 32 matches
            the paper's default for common SR training patch sizes.
        reduction: "none" (keep all channels), "gray" (luma before FFT), or
            "mean" (average the per-channel spectra after FFT).
        window: "hann" to window each patch before the FFT (reduces edge
            artifacts in the spectrum), or None.
        fix_normalize: Per-sample vs per-batch spectrum normalization.
        spec_type: "power", "magnitude", or "full" (magnitude + phase).
        scale: "linear" or "log" power scaling (only used for spec_type="power").
        use_msra_init: MSRA/Kaiming init for conv & linear layers.
        pool: "avg" or "max" adaptive pooling before the FC head.
        use_spectral_norm: Wrap conv/linear layers with spectral norm.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_feat: int = 64,
        patch_size: int = 32,
        reduction: str = "none",
        window: str | None = None,
        fix_normalize: bool = False,
        spec_type: str = "power",
        scale: str = "linear",
        use_msra_init: bool = True,
        pool: str = "avg",
        use_spectral_norm: bool = False,
        input_noise: float = 0.0,
    ) -> None:
        super().__init__()
        # [USS] input_noise (sigma in 8-bit levels, e.g. 2.0 = 2/255): Gaussian instance
        # noise added to EVERY input (real and fake) in train mode only. With scale="log"
        # the spectrum shows the sub-LSB noise floor (dither of the 10->8-bit GT extraction,
        # residual grain): without noise the D separates GT vs G output at ~98 % from that
        # floor alone and, as D2, would push G to ADD grain. Measured (GT vs WSR v2 135k):
        # held-out acc 0.975 (none), 0.956 (8-bit quantize), 0.863 (1/255), 0.706 (2/255).
        self.input_noise = float(input_noise or 0.0)
        # UI/config layers may pass the string "none" (case-insensitive) as a
        # human-typed sentinel for "no window" rather than YAML/TOML null ->
        # Python None, e.g. traiNNer-redux config generated by Universal SR
        # Studio's field defaults. Normalize here so both forms work.
        if isinstance(window, str) and window.strip().lower() == "none":
            window = None

        self.patch_size = patch_size
        self.reduction = reduction
        self.window = window
        self.fix_normalize = fix_normalize
        self.spec_type = spec_type
        self.scale = scale

        spec_ch = num_in_ch
        if reduction in ("gray", "mean"):
            spec_ch = 1
        if spec_type == "full":
            spec_ch *= 2

        self.disc = _VggBlockDiscriminator(
            num_in_ch=spec_ch,
            num_feat=num_feat,
            use_msra_init=use_msra_init,
            pool=pool,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Garde-fou : spectrum_transform() exige H et W divisibles par patch_size,
        # sinon EinopsError et crash immédiat du training (vécu en réel : patch_size=32
        # sur des crops 80x80). Le YAML n'est pas toujours régénéré via l'UI (Start
        # Training relance directement le fichier existant sur disque) donc on ne peut
        # pas compter uniquement sur la correction faite à la génération de config
        # (config_handler.py) — on la refait ici, au runtime, en dernier recours.
        ps = self.patch_size
        h, w = x.shape[2], x.shape[3]
        if h % ps != 0 or w % ps != 0:
            for cand in (64, 32, 16, 8, 4, 2, 1):
                if cand <= ps and h % cand == 0 and w % cand == 0:
                    ps = cand
                    break
            else:
                ps = 1
            if not getattr(self, "_warned_patch_size", False):
                print(
                    f"[SpectralDiscriminator] patch_size={self.patch_size} ne divise pas "
                    f"l'entrée {h}x{w} -> corrigé à {ps} pour ce run (ajuste ta config pour "
                    f"faire taire cet avertissement)."
                )
                self._warned_patch_size = True
        if self.training and self.input_noise > 0:
            x = x + torch.randn_like(x) * (self.input_noise / 255.0)
        spec = spectrum_transform(
            x,
            patch_size=ps,
            reduction=self.reduction,
            window=self.window,
            fix_normalize=self.fix_normalize,
            spec_type=self.spec_type,
            scale=self.scale,
        )
        return self.disc(spec)
