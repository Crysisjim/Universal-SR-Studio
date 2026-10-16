"""Spectral (Fourier-domain) preprocessing for discriminator inputs.

Ported from the official DualFormer / "On the Effectiveness of Spectral
Discriminators for Perceptual Quality Improvement" repository
(https://github.com/Luciennnnnnn/DualFormer, MIT license), source file
``basicsr/utils/spectrum.py``. The paper (ICCV 2023,
https://arxiv.org/abs/2307.12027) shows that running a standard
discriminator on the Fourier spectrum of image patches, in *addition* to a
normal spatial-domain discriminator, improves perceptual GAN training.

Only ``spectrum_transform`` and its direct dependencies are ported here
(the reference file also has unused/experimental variants — reduced
spectra, azimuthal averaging, masking, etc. — that are not needed for the
"spectral discriminator" use case and are intentionally left out).

Pure ``torch.fft`` + ``einops`` — no compiled/CUDA-specific dependency.
"""

import einops
import torch
from torch import Tensor
from torch.fft import fftn, fftshift


def add_window(data: Tensor, window: str | None = None) -> Tensor:
    """Apply a Hann window over the last two (spatial) dims of ``data``.

    Args:
        data: Tensor shaped (N, H, W).
        window: ``"hann"`` to apply a 2D Hann window, ``None`` to skip.
    """
    if window is not None:
        if window == "hann":
            h_win = torch.hann_window(
                data.shape[1], periodic=False, dtype=data.dtype, device=data.device
            )
            w_win = torch.hann_window(
                data.shape[2], periodic=False, dtype=data.dtype, device=data.device
            )
            win = torch.matmul(h_win.unsqueeze(dim=1), w_win.unsqueeze(dim=0))
            data = data * win
        else:
            raise NotImplementedError(f"Does not support window: {window}")
    return data


def complex_mag(x: Tensor) -> Tensor:
    return torch.sqrt(x.real**2 + x.imag**2 + 1e-7)


def complex_phase(x: Tensor) -> Tensor:
    return torch.arctan(x.imag / (x.real + 1e-7))


def get_power_spectrum(
    data: Tensor, window: str | None = None, scale: str = "linear"
) -> Tensor:
    """Power spectrum |F|^2 of a (N, H, W) tensor, DC-centered."""
    if (data.ndim - 1) != 2:
        raise AttributeError("Data must be 2d.")

    if window is not None:
        data = add_window(data, window=window)

    freq = fftn(data, dim=(1, 2))
    freq = fftshift(freq, dim=(1, 2))
    power_spec = freq.real**2 + freq.imag**2

    if scale == "log":
        power_spec = 10 * torch.log10(power_spec + 1e-7)

    return power_spec


def get_magnitude_spectrum(data: Tensor, window: str | None = None) -> Tensor:
    """Magnitude spectrum |F| of a (N, H, W) tensor, DC-centered."""
    if (data.ndim - 1) != 2:
        raise AttributeError("Data must be 2d.")

    if window is not None:
        data = add_window(data, window=window)

    freq = fftn(data, dim=(1, 2))
    freq = fftshift(freq, dim=(1, 2))
    return complex_mag(freq)


def get_phase_spectrum(data: Tensor, window: str | None = None) -> Tensor:
    """Phase spectrum of a (N, H, W) tensor, DC-centered."""
    if (data.ndim - 1) != 2:
        raise AttributeError("Data must be 2d.")

    if window is not None:
        data = add_window(data, window=window)

    freq = fftn(data, dim=(1, 2))
    freq = fftshift(freq, dim=(1, 2))
    return complex_phase(freq)


def normalize(x: Tensor, fix_normalize: bool = False) -> Tensor:
    """Min-max normalize to [0, 1].

    ``fix_normalize=True`` normalizes per-sample (over all dims but batch);
    ``fix_normalize=False`` normalizes over the whole batch with a single
    global min/max (matches the reference implementation's default).
    """
    dims = list(range(x.dim()))[1:]
    if fix_normalize:
        x = (x - x.amin(dims, keepdim=True)) / (
            x.amax(dims, keepdim=True) - x.amin(dims, keepdim=True) + 1e-5
        )
    else:
        x = (x - x.min()) / (x.max() - x.min() + 1e-5)
    return x


def spectrum_transform(
    x: Tensor,
    patch_size: int,
    reduction: str = "none",
    window: str | None = None,
    fix_normalize: bool = False,
    spec_type: str = "power",
    scale: str = "linear",
) -> Tensor:
    """Patch-wise Fourier-spectrum transform.

    Splits ``x`` (B, C, H, W) into non-overlapping ``patch_size`` x
    ``patch_size`` patches, computes the per-patch Fourier spectrum
    (power/magnitude/phase), normalizes it, and reassembles the patches
    back into a (B, C', H, W) image — so the spatial resolution is
    unchanged and the result can be fed straight into a standard
    image discriminator.

    Args:
        x: Input image batch, (B, C, H, W). H and W must be divisible by
            ``patch_size``.
        patch_size: Side length of the square patches the spectrum is
            computed on.
        reduction: ``"none"`` (keep all input channels), ``"gray"``
            (convert to luma before the transform), or ``"mean"`` (average
            the per-channel spectra into one channel after the transform).
        window: ``"hann"`` to window each patch before the FFT (reduces
            spectral leakage from the patch edges), or ``None``.
        fix_normalize: Per-sample vs. per-batch min-max normalization.
        spec_type: ``"power"``, ``"magnitude"``, or ``"full"`` (magnitude
            concatenated with phase, doubling the channel count).
        scale: ``"linear"`` or ``"log"`` (only used for ``spec_type="power"``).

    Returns:
        Tensor (B, C', H, W) with the same spatial size as the input,
        values normalized to [0, 1].
    """
    h, w = x.shape[2] // patch_size, x.shape[3] // patch_size

    x = einops.rearrange(
        x, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=patch_size, p2=patch_size
    )

    if reduction == "gray":
        x = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x = x.unsqueeze(dim=1)

    if spec_type == "power":
        x = get_power_spectrum(x.flatten(0, 1), window=window, scale=scale).unflatten(
            0, (x.shape[0], x.shape[1])
        )
        if reduction == "mean":
            x = x.mean(dim=1, keepdim=True).float()
        x = normalize(x, fix_normalize=fix_normalize)
    elif spec_type == "magnitude":
        x = get_magnitude_spectrum(x.flatten(0, 1), window=window).unflatten(
            0, (x.shape[0], x.shape[1])
        )
        if reduction == "mean":
            x = x.mean(dim=1, keepdim=True).float()
        x = normalize(x, fix_normalize=fix_normalize)
    elif spec_type == "full":
        mag_spectrum = get_magnitude_spectrum(
            x.flatten(0, 1), window=window
        ).unflatten(0, (x.shape[0], x.shape[1]))
        mag_spectrum = normalize(mag_spectrum, fix_normalize=fix_normalize)
        phase_spectrum = get_phase_spectrum(x.flatten(0, 1), window=window).unflatten(
            0, (x.shape[0], x.shape[1])
        )
        phase_spectrum = normalize(phase_spectrum, fix_normalize=fix_normalize)
        x = torch.cat([mag_spectrum, phase_spectrum], dim=1)
    else:
        raise NotImplementedError(f"Does not support: {spec_type}")

    x = einops.rearrange(x, "(b h w) c p1 p2 -> b c (h p1) (w p2)", h=h, w=w)
    return x
