"""ICNR initialization shared by PixelShuffle heads (required by pulse_arch.py).

Added locally: upstream PULSE (HF Phips/PULSE) imports `traiNNer.utils.icnr`,
which does not exist in this checkout. Same implementation as the private
helper in nerve_arch.py. Init-only: no effect once a checkpoint is loaded.
"""

import math

import torch
from torch import nn


def icnr_init(conv: nn.Conv2d, upscale: int) -> None:
    """ICNR initialization for a Conv2d that feeds a PixelShuffle(upscale).

    Every sub-pixel phase starts from the same base kernel, so the upsampling
    head cannot begin with a checkerboard bias.
    """
    out_c = conv.out_channels
    sub = upscale * upscale
    assert out_c % sub == 0
    base_c = out_c // sub
    base = nn.init.kaiming_uniform_(conv.weight[:base_c].clone(), a=math.sqrt(5))
    with torch.no_grad():
        conv.weight.data.zero_()
        for i in range(sub):
            conv.weight.data[i::sub] = base
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
