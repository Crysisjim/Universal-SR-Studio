"""
guided_color_fix.py — "Guided" Color Fix: Bendel's aniCC / PACC V4 model.

A learned colour corrector: it takes the upscaled image (structure reference) and the
low-res source (colour reference) and re-colours the upscale like the source, while
keeping its detail — unlike wavelet/average, it also handles non-uniform colour shifts.

Model + network code: Bendel (labx), https://huggingface.co/labx/2x_aniCC
(CC BY-NC-SA 4.0). The PACCV4 classes below are the ones from the author's chaiNNer
node (GuidedColorFix-chaiNNer.zip, guided_color_fix.py), unchanged; the chaiNNer node
glue is replaced by run_guided() / the CLI.

The model needs HR = exactly 2x LR: any upscale is handled by resizing the reference
to HR/2 (bicubic) and padding odd HR sizes by one pixel.

Weights are downloaded on first use to <engine root>/models/colorfix/.

CLI (used by the portable GUI, which has no torch):
    <engine venv python> guided_color_fix.py <sr.png> <lr.png> <out.npy>
writes the corrected image as float32 HxWx3 in [0, 1].
"""
from __future__ import annotations

import math
import os
import sys

WEIGHTS_URL = "https://huggingface.co/labx/2x_aniCC/resolve/main/2x_bndl_anicc.safetensors"
WEIGHTS_NAME = "2x_bndl_anicc.safetensors"


def default_weights_path() -> str:
    try:
        from src.core.engine_paths import base_engine_path
    except ImportError:
        from engine_paths import base_engine_path
    return os.path.join(base_engine_path(), "models", "colorfix", WEIGHTS_NAME)


def _safe_print(msg) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:  # cp1252 console
        print(str(msg).encode("ascii", "replace").decode(), flush=True)


def ensure_weights(path: str | None = None, log=_safe_print) -> str:
    """Return the weights path, downloading them (~14 MB) the first time."""
    path = path or default_weights_path()
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import urllib.request
    log(f"Guided Color Fix : téléchargement du modèle aniCC (Bendel, ~14 Mo) -> {path}")
    tmp = path + ".part"
    with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)
    return path


# ─── PACC V4 (verbatim from the author's chaiNNer node) ─────────────────────────
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402,N812
from torch import Tensor, nn  # noqa: E402


class ConvAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                              dilation=dilation, bias=True)
        self.act = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvAct(channels, channels, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.conv1(x))


class ContextStage(nn.Module):
    def __init__(self, channels: int, n_blocks: int) -> None:
        super().__init__()
        dilations = (1, 2, 4, 2)
        self.body = nn.Sequential(*[DilatedResBlock(channels, dilation=dilations[i % len(dilations)])
                                    for i in range(n_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class StructureStats(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    @staticmethod
    def luma(x: Tensor) -> Tensor:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def edge_map(self, x: Tensor) -> Tensor:
        luma = self.luma(x)
        grad_x = F.conv2d(luma, self.sobel_x.to(dtype=x.dtype), padding=1)
        grad_y = F.conv2d(luma, self.sobel_y.to(dtype=x.dtype), padding=1)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)


class GlobalColorAlign(nn.Module):
    def __init__(self, channels: int, n_knots: int = 8) -> None:
        super().__init__()
        self.n_knots = n_knots
        knots = torch.linspace(1.0 / (n_knots + 1), n_knots / (n_knots + 1), n_knots)
        self.register_buffer("knots", knots, persistent=False)
        eye = torch.eye(3).reshape(1, 9)
        self.register_buffer("eye", eye, persistent=False)
        n_params = 9 + 3 + 3 * n_knots
        self.mlp = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, n_params))

    def forward(self, pooled: Tensor, hr: Tensor) -> Tensor:
        b, _, h, w = hr.shape
        params = self.mlp(pooled)
        mat = (params[:, :9] + self.eye.to(dtype=hr.dtype)).view(b, 3, 3)
        bias = params[:, 9:12].view(b, 3, 1, 1)
        slopes = params[:, 12:].view(b, 3, self.n_knots) / self.n_knots
        y = torch.bmm(mat, hr.flatten(2)).view(b, 3, h, w) + bias
        curve = torch.zeros_like(y)
        for k in range(self.n_knots):
            curve = curve + slopes[:, :, k].view(b, 3, 1, 1) * F.relu(y - self.knots[k].to(dtype=hr.dtype))
        return y + curve


class PaletteRouter(nn.Module):
    def __init__(self, kv_channels: int, q_channels: int, out_channels: int,
                 num_slots: int = 32, slot_dim: int = 64) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, slot_dim, 16, 16) * 0.02)
        self.to_k = nn.Conv2d(kv_channels, slot_dim, 1, bias=True)
        self.to_v = nn.Conv2d(kv_channels, slot_dim, 1, bias=True)
        self.to_q = nn.Conv2d(q_channels, slot_dim, 1, bias=True)
        self.to_out = nn.Conv2d(slot_dim, out_channels, 1, bias=True)

    def forward(self, kv_feat: Tensor, q_feat: Tensor, reliability: Tensor) -> Tensor:
        b, _, h_kv, w_kv = kv_feat.shape
        _, _, h_q, w_q = q_feat.shape
        scale = 1.0 / math.sqrt(self.slot_dim)
        pos_kv = F.interpolate(self.pos, size=(h_kv, w_kv), mode="bilinear", align_corners=False)
        pos_q = F.interpolate(self.pos, size=(h_q, w_q), mode="bilinear", align_corners=False)
        k = (self.to_k(kv_feat) + pos_kv).flatten(2)
        v = self.to_v(kv_feat).flatten(2).transpose(1, 2)
        log_rel = torch.log(reliability.flatten(2) + 1e-4)
        slots = self.slots.expand(b, -1, -1)
        gather_attn = torch.softmax(torch.bmm(slots, k) * scale + log_rel, dim=-1)
        slot_vals = torch.bmm(gather_attn, v)
        q = (self.to_q(q_feat) + pos_q).flatten(2).transpose(1, 2)
        route_attn = torch.softmax(torch.bmm(q, slots.transpose(1, 2)) * scale, dim=-1)
        routed = torch.bmm(route_attn, slot_vals)
        routed = routed.transpose(1, 2).reshape(b, self.slot_dim, h_q, w_q)
        return self.to_out(routed)


class GuidedCoeffUpsampler(nn.Module):
    def __init__(self, sharpness: float = 20.0) -> None:
        super().__init__()
        self.sharpness = sharpness

    @staticmethod
    def _shift(x: Tensor, dy: int, dx: int) -> Tensor:
        _, _, h, w = x.shape
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        y0 = 1 + dy
        x0 = 1 + dx
        return x_pad[:, :, y0: y0 + h, x0: x0 + w]

    def forward(self, coeffs: Tensor, guide_hr: Tensor, guide_lr: Tensor) -> Tensor:
        size = guide_hr.shape[-2:]
        num = torch.zeros(coeffs.shape[0], coeffs.shape[1], *size, dtype=coeffs.dtype, device=coeffs.device)
        den = torch.zeros(coeffs.shape[0], 1, *size, dtype=coeffs.dtype, device=coeffs.device)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c_up = F.interpolate(self._shift(coeffs, dy, dx), size=size, mode="bilinear", align_corners=False)
                g_up = F.interpolate(self._shift(guide_lr, dy, dx), size=size, mode="bilinear", align_corners=False)
                weight = torch.exp(-self.sharpness * (guide_hr - g_up).abs().mean(dim=1, keepdim=True))
                num = num + c_up * weight
                den = den + weight
        return num / (den + 1e-6)


class PACCV4(nn.Module):
    def __init__(self, *, feat: int, n_blocks: int, num_slots: int, slot_dim: int, n_knots: int,
                 coeff_scale: float = 0.5, refine_bound: float = 0.1, jbu_sharpness: float = 20.0) -> None:
        super().__init__()
        self.coeff_scale = coeff_scale
        self.refine_bound = refine_bound
        self.stats = StructureStats()
        self.in_channels = 3 + 3 + 3 + 1 + 1 + 1 + 1
        self.enc0 = nn.Sequential(ConvAct(self.in_channels, feat), ContextStage(feat, n_blocks))
        self.down1 = nn.Conv2d(feat, feat * 2, 3, stride=2, padding=1, bias=True)
        self.enc1 = nn.Sequential(nn.PReLU(feat * 2), ContextStage(feat * 2, n_blocks))
        self.down2 = nn.Conv2d(feat * 2, feat * 4, 3, stride=2, padding=1, bias=True)
        self.enc2 = nn.Sequential(nn.PReLU(feat * 4), ContextStage(feat * 4, n_blocks + 1))
        self.reliability_head = nn.Sequential(ConvAct(feat * 4, feat), nn.Conv2d(feat, 1, 1, bias=True))
        self.global_align = GlobalColorAlign(feat * 4, n_knots=n_knots)
        self.app_enc0 = nn.Sequential(ConvAct(5, feat), ContextStage(feat, 1))
        self.app_down = nn.Conv2d(feat, feat * 2, 3, stride=2, padding=1, bias=True)
        self.app_enc1 = nn.Sequential(nn.PReLU(feat * 2), ContextStage(feat * 2, n_blocks))
        self.router = PaletteRouter(kv_channels=feat * 4 + 3, q_channels=feat * 2, out_channels=feat * 2,
                                    num_slots=num_slots, slot_dim=slot_dim)
        self.dec4 = nn.Sequential(ConvAct(feat * 4 + 3, feat * 4), ContextStage(feat * 4, n_blocks))
        self.dec4_up = nn.Conv2d(feat * 4, feat * 2, 3, padding=1, bias=True)
        self.fuse2 = nn.Sequential(ConvAct(feat * 6, feat * 2), ContextStage(feat * 2, n_blocks))
        self.coeff_head = nn.Sequential(ConvAct(feat * 2, feat * 2), nn.Conv2d(feat * 2, 12, 3, padding=1, bias=True))
        self.upsampler = GuidedCoeffUpsampler(sharpness=jbu_sharpness)
        self.refine = nn.Sequential(ConvAct(7, feat), ContextStage(feat, max(n_blocks - 1, 1)),
                                    nn.Conv2d(feat, 3, 3, padding=1, bias=True))

    def forward(self, hr: Tensor, lr: Tensor) -> Tensor:
        b, _, h_hr, w_hr = hr.shape
        lr_up = F.interpolate(lr, size=(h_hr, w_hr), mode="bicubic", align_corners=False)
        hr_luma = self.stats.luma(hr)
        hr_edge = self.stats.edge_map(hr)
        lr_edge = self.stats.edge_map(lr_up)
        edge_diff = (lr_edge - hr_edge).abs()
        x = torch.cat([hr, lr_up, lr_up - hr, hr_luma, hr_edge, lr_edge, edge_diff], dim=1)
        x0 = self.enc0(x)
        x1 = self.enc1(self.down1(x0))
        evid = self.enc2(self.down2(x1))
        reliability = torch.sigmoid(self.reliability_head(evid))
        rel_sum = reliability.sum(dim=(2, 3))
        pooled = (evid * reliability).sum(dim=(2, 3)) / (rel_sum + 1e-4)
        hr_base = self.global_align(pooled, hr)
        obs = F.interpolate(lr_up - hr_base, size=evid.shape[-2:], mode="bilinear", align_corners=False)
        kv_feat = torch.cat([evid, obs], dim=1)
        app = self.app_enc0(torch.cat([hr_base, hr_luma, hr_edge], dim=1))
        app = self.app_enc1(self.app_down(app))
        routed = self.router(kv_feat, app, reliability)
        d4 = self.dec4(kv_feat)
        d2 = F.interpolate(d4, size=app.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.fuse2(torch.cat([self.dec4_up(d2), routed, app], dim=1))
        coeffs = torch.tanh(self.coeff_head(d2)) * self.coeff_scale
        guide_lr = F.interpolate(hr_base, size=coeffs.shape[-2:], mode="bilinear", align_corners=False)
        m = self.upsampler(coeffs, hr_base, guide_lr)
        m = m.view(b, 3, 4, h_hr, w_hr)
        hr_aug = torch.cat([hr_base, torch.ones_like(hr_base[:, :1])], dim=1)
        delta = (m * hr_aug.unsqueeze(1)).sum(dim=2)
        out_coarse = hr_base + delta
        refine_in = torch.cat([out_coarse, hr, hr_edge], dim=1)
        refine = torch.tanh(self.refine(refine_in)) * self.refine_bound
        return out_coarse + refine


def _count_blocks(state_dict, prefix: str) -> int:
    return len({k[len(prefix):].split(".", 1)[0] for k in state_dict
                if k.startswith(prefix) and k[len(prefix):].split(".", 1)[0].isdigit()})


def build_model(state_dict) -> PACCV4:
    if not ("router.slots" in state_dict and "global_align.mlp.2.weight" in state_dict):
        raise ValueError("Ce fichier n'est pas un modèle PACC V4 (aniCC).")
    n_params = int(state_dict["global_align.mlp.2.weight"].shape[0])
    model = PACCV4(feat=int(state_dict["enc0.0.conv.weight"].shape[0]),
                   n_blocks=_count_blocks(state_dict, "enc0.1.body."),
                   num_slots=int(state_dict["router.slots"].shape[1]),
                   slot_dim=int(state_dict["router.slots"].shape[2]),
                   n_knots=(n_params - 12) // 3)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


# ─── App glue ──────────────────────────────────────────────────────────────────
_CACHE: dict = {}


def _model(device: str, weights: str | None, log):
    key = (device, weights)
    if key not in _CACHE:
        from safetensors.torch import load_file
        _CACHE[key] = build_model(load_file(ensure_weights(weights, log), device="cpu")).to(device)
    return _CACHE[key]


def run_guided(sr, ref, device: str = "auto", weights: str | None = None, log=_safe_print):
    """sr: float32 HxWx3 [0,1] (the upscale). ref: float32 hxwx3 [0,1] or PIL image at
    any size (the colour reference — normally the LQ source). Returns float32 HxWx3."""
    import numpy as np
    if device in ("auto", "trt", "cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = sr.shape[:2]
    ph, pw = h % 2, w % 2
    hr = torch.from_numpy(np.ascontiguousarray(sr, dtype=np.float32)).permute(2, 0, 1)[None]
    if ph or pw:
        hr = F.pad(hr, (0, pw, 0, ph), mode="replicate")
    if not isinstance(ref, np.ndarray):
        ref = np.asarray(ref.convert("RGB"), dtype=np.float32) / 255.0
    lr = torch.from_numpy(np.ascontiguousarray(ref, dtype=np.float32)).permute(2, 0, 1)[None]
    size = (hr.shape[2] // 2, hr.shape[3] // 2)
    if tuple(lr.shape[2:]) != size:
        lr = F.interpolate(lr, size=size, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    model = _model(device, weights, log)
    with torch.inference_mode():
        out = model(hr.to(device), lr.to(device)).float().clamp(0, 1)
    return out[0, :, :h, :w].permute(1, 2, 0).cpu().numpy()


def main(argv) -> int:
    import numpy as np
    from PIL import Image
    if argv[1].lower().endswith(".npy"):
        sr = np.load(argv[1]).astype(np.float32)
    else:
        sr = np.asarray(Image.open(argv[1]).convert("RGB"), dtype=np.float32) / 255.0
    device = argv[4] if len(argv) > 4 else "auto"
    out = run_guided(sr, Image.open(argv[2]), device=device, log=lambda m: print(m, flush=True))
    np.save(argv[3], out.astype(np.float32))
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _here = os.path.dirname(os.path.abspath(__file__))
    for _p in (_here, os.path.dirname(os.path.dirname(_here))):
        if _p not in sys.path:
            sys.path.append(_p)
    sys.exit(main(sys.argv))
