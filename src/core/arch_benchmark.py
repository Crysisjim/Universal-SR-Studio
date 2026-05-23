"""
arch_benchmark.py - Benchmark neosr architectures.

Runs each arch for N iterations (default 2500) with identical config.
Records: avg it/s, peak VRAM, PSNR/SSIM. After each arch, runs a quick
upscale test on the saved checkpoint.

Errors are caught per-arch — benchmark continues to the next.
All results are timestamped; existing logs are never overwritten.

Usage:
    python arch_benchmark.py [options]

    --n-iter 2500          Iterations per arch (default: 2500)
    --timeout 3600         Max seconds per arch before kill (default: 3600)
    --archs esc,omnisr     Comma-separated list (default: all)
    --train-gt PATH        Override train HR dataset path
    --val-gt PATH          Override val GT path
    --val-lq PATH          Override val LQ path
    --amp                  Enable mixed-precision (use_amp = true)
    --bf16                 Enable bfloat16
    --output-dir PATH      Results directory (default: ~/IA_Engine/benchmark_results)
    --keep-toml            Don't delete temp TOML files after run
    --no-upscale           Skip quick upscale test after each arch
    --enable-val           Enable val/PSNR (needs val_lq at same scale as archs)
    --laptop               Laptop mode: test 4 precision modes per arch
                           (NORMAL, FP16-AMP, BF16-AMP, TF32) — auto-prompted if omitted
    --desktop              Desktop mode: single NORMAL run per arch
"""
import sys
import os
import re
import time
import json
import threading
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

NEOSR_PATH     = Path.home() / "IA_Engine" / "neosr"
NEOSR_PYTHON   = NEOSR_PATH / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT   = NEOSR_PATH / "train.py"

TRAINNER_PATH   = Path.home() / "IA_Engine" / "traiNNer-redux"
TRAINNER_PYTHON = TRAINNER_PATH / ".venv" / "Scripts" / "python.exe"

_THIS_DIR = Path(__file__).parent
NEOSR_RUNNER         = _THIS_DIR / "neosr_runner.py"
NEOSR_GENERAL_RUNNER = _THIS_DIR / "neosr_general_runner.py"
UNIVERSAL_RUNNER     = _THIS_DIR / "universal_runner.py"

DEFAULT_TRAIN_GT = str(Path.home() / "IA_Engine" / "datasets" / "train" / "HR")
DEFAULT_VAL_GT   = str(Path.home() / "IA_Engine" / "datasets" / "val" / "GT")
DEFAULT_VAL_LQ   = str(Path.home() / "IA_Engine" / "datasets" / "val" / "LQ")
DEFAULT_TEST_IMG = ""  # Set via --train-gt or leave empty to skip upscale test

# Precision modes for laptop (RTX) benchmark
# TF32: fast_matmul=true  → torch matmul+cudnn both use TF32 cores
# FP16: use_amp=true, bfloat16=false
# BF16: use_amp=true, bfloat16=true
PRECISION_MODES = [
    {"label": "NORMAL", "use_amp": False, "use_bf16": False, "fast_matmul": False},
    {"label": "FP16",   "use_amp": True,  "use_bf16": False, "fast_matmul": False},
    {"label": "BF16",   "use_amp": True,  "use_bf16": True,  "fast_matmul": False},
    {"label": "TF32",   "use_amp": False, "use_bf16": False, "fast_matmul": True},
]
DESKTOP_MODE = [
    {"label": "NORMAL", "use_amp": False, "use_bf16": False, "fast_matmul": False},
]

# ── Arch definitions ───────────────────────────────────────────────────────────
# All archs benchmarked at scale=1 (restoration/deband use case).
# NOTE: val (PSNR) is DISABLED by default because the user's val_lq is at 4x.
#   Use --enable-val with a matching 1x val dataset to re-enable.
# Factory-based archs (hat_s, dat_s, swinir_small, etc.): only pass params NOT
# already set by the factory to avoid "got multiple values for keyword" errors.

# ── Phase 1 DONE (desktop GTX 1080 Ti — 2026-05-15) ──────────────────────────
# These archs passed phase 1. Disabled to skip on next run.
# Full results: benchmark_results/benchmark_20260515_230809.txt
# Results summary:
#   span       5.368 it/s  1.70 GB  PSNR 33.50  upscale:ok
#   compact   11.861 it/s  0.73 GB  PSNR 53.50  upscale:ok
#   safmn      8.445 it/s  1.19 GB  PSNR 37.72  upscale:ok
#   esc        1.196 it/s  5.30 GB  PSNR 37.00  upscale:ok
#   plksr      1.947 it/s  2.90 GB  PSNR 43.15  upscale:ok
#   realplksr  1.877 it/s  3.18 GB  PSNR 22.07  upscale:ok
#   rcan       1.324 it/s  6.72 GB  PSNR 38.24  upscale:ok
#   ditn       1.237 it/s  4.86 GB  PSNR 31.51  upscale:ok
ARCHS_PHASE1_DONE = {
    "span": {
        "scale": 1,
        "desc": "SPAN (feature_channels=48, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "span"\n'
            'num_in_ch = 3\n'
            'num_out_ch = 3\n'
            'feature_channels = 48\n'
            'upscale = 1'
        ),
    },
    "compact": {
        "scale": 1,
        "desc": "RealESRGAN Compact (num_feat=64, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "compact"\n'
            'num_in_ch = 3\n'
            'num_out_ch = 3\n'
            'num_feat = 64\n'
            'num_block = 16\n'
            'upscale = 1'
        ),
    },
    "safmn": {
        "scale": 1,
        "desc": "SAFMN (dim=36, n_blocks=8, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "safmn"\n'
            'dim = 36\n'
            'n_blocks = 8\n'
            'upscaling_factor = 1'
        ),
    },
    "esc": {
        "scale": 1,
        "desc": "ESC (dim=64, n_blocks=5, DySample 1x)",
        "is_esc": True,
        "network_g": (
            'type = "esc"\n'
            'dim = 64\n'
            'n_blocks = 5\n'
            'window_size = 32\n'
            'attn_type = "sdpa"\n'
            'use_dysample = true'
        ),
    },
    "plksr": {
        "scale": 1,
        "desc": "PLKSR (dim=64, n_blocks=28, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "plksr"\n'
            'dim = 64\n'
            'n_blocks = 28\n'
            'upscaling_factor = 1'
        ),
    },
    "realplksr": {
        "scale": 1,
        "desc": "RealPLKSR (dim=64, n_blocks=28, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "realplksr"\n'
            'dim = 64\n'
            'n_blocks = 28\n'
            'upscaling_factor = 1'
        ),
    },
    "rcan": {
        "scale": 1,
        "desc": "RCAN (n_resgroups=10, n_feats=64, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "rcan"\n'
            'n_resgroups = 10\n'
            'n_resblocks = 20\n'
            'n_feats = 64\n'
            'reduction = 16\n'
            'upscale = 1'
        ),
    },
    "ditn": {
        "scale": 1,
        "desc": "DITN (embed_dim=60, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "ditn"\n'
            'upscale = 1\n'
            'num_in_ch = 3\n'
            'num_out_ch = 3\n'
            'img_size = 64\n'
            'patch_size = 1\n'
            'embed_dim = 60\n'
            'isa_num_head = 6\n'
            'transposed_num_head = 6\n'
            'num_ISABs = 4\n'
            'num_TransGroups = 4\n'
            'window_size = 16\n'
            'mlp_ratio = 2.0'
        ),
    },
}

# ── Phase 2 — archs to test (fixed names/params + new archs) ──────────────────
# Archs that errored in phase 1 with corrected type names / params,
# plus new archs not present in the original run.
ARCHS_TO_TEST = {
    # ── Lightweight CNN ────────────────────────────────────────────────────────
    "ninasr": {
        "scale": 1,
        "desc": "NinaSR (n_feats=32, n_resblocks=26, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "ninasr"\n'
            'n_feats = 32\n'
            'n_resblocks = 26\n'
            'n_colors = 3\n'
            'scale = 1'
        ),
    },
    "lmlt": {
        "scale": 1,
        "desc": "LMLT (dim=60, n_blocks=8, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "lmlt"\n'
            'dim = 60\n'
            'n_blocks = 8\n'
            'upscaling_factor = 1'
        ),
    },
    # ── CNN medium ─────────────────────────────────────────────────────────────
    "omnisr": {
        "scale": 1,
        "desc": "OmniSR (num_feat=64, window=8, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "omnisr"\n'
            'num_feat = 64\n'
            'upsampling = 1\n'
            'window_size = 8'
        ),
    },
    "esrgan": {
        "scale": 1,
        "desc": "ESRGAN/RRDB (num_feat=64, num_block=23, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "esrgan"\n'
            'num_in_ch = 3\n'
            'num_out_ch = 3\n'
            'num_feat = 64\n'
            'num_block = 23\n'
            'num_grow_ch = 32'
        ),
    },
    # ── CNN heavy ──────────────────────────────────────────────────────────────
    "man": {
        "scale": 1,
        "desc": "MAN (n_feats=180, n_resblocks=36, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "man"\n'
            'n_resblocks = 36\n'
            'n_resgroups = 1\n'
            'n_feats = 180\n'
            'n_colors = 3\n'
            'scale = 1'
        ),
    },
    # ── Transformer light ──────────────────────────────────────────────────────
    "swinir_small": {
        "scale": 1,
        "desc": "SwinIR-S (embed_dim=60, factory defaults, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "swinir_small"\n'
            'upscale = 1'
        ),
    },
    "eimn": {
        "scale": 1,
        "desc": "EIMN (embed_dims=48, num_stages=28, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "eimn"\n'
            'embed_dims = 48\n'
            'scale = 1\n'
            'num_stages = 28'
        ),
    },
    # ── Transformer medium ─────────────────────────────────────────────────────
    "hat_s": {
        "scale": 1,
        "desc": "HAT-S (embed=144, img_size=96, batch=2/accum=2, 1x)",
        "is_esc": False,
        "batch_size": 2,
        "accumulate": 2,
        "network_g": (
            'type = "hat_s"\n'
            'upscale = 1\n'
            'img_size = 96'
        ),
    },
    "dat_s": {
        "scale": 1,
        "desc": "DAT-S (factory defaults: embed=180, img_size=96, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "dat_s"\n'
            'upscale = 1\n'
            'img_size = 96'
        ),
    },
    "dat_2": {
        "scale": 1,
        "desc": "DAT-2 (factory defaults: embed=180, dual-attn, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "dat_2"\n'
            'upscale = 1'
        ),
    },
    "srformer_medium": {
        "scale": 1,
        "desc": "SRFormer-M (factory defaults: embed=180, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "srformer_medium"\n'
            'upscale = 1'
        ),
    },
    "grformer_medium": {
        "scale": 1,
        # NOTE: factory hardcodes img_size=64 before **kwargs → cannot override without TypeError.
        # grformer recalculates attention mask dynamically when input != img_size → no crash.
        # window_size=[8,32]: 96 divisible by both → OK.
        "desc": "GRFormer-M (factory defaults, batch=2/accum=2, 1x)",
        "is_esc": False,
        "batch_size": 2,
        "accumulate": 2,
        "network_g": (
            'type = "grformer_medium"\n'
            'upscale = 1'
        ),
    },
    "rgt_s": {
        "scale": 1,
        "desc": "RGT-S (factory defaults, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "rgt_s"\n'
            'upscale = 1'
        ),
    },
    # ── Transformer heavy ──────────────────────────────────────────────────────
    "hat_m": {
        "scale": 1,
        "desc": "HAT-M (embed=180, img_size=96, batch=2/accum=2, 1x)",
        "is_esc": False,
        "batch_size": 2,
        "accumulate": 2,
        "network_g": (
            'type = "hat_m"\n'
            'upscale = 1\n'
            'img_size = 96'
        ),
    },
    "swinir_medium": {
        "scale": 1,
        "desc": "SwinIR-M (factory defaults: embed=180, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "swinir_medium"\n'
            'upscale = 1'
        ),
    },
    "atd": {
        "scale": 1,
        # window_size=16 → 256 tokens/window → softmax NaN. window_size=8 (64 tokens) stable.
        "desc": "ATD (embed_dim=90, img_size=96, window_size=8, batch=2/accum=2, 1x)",
        "is_esc": False,
        "batch_size": 2,
        "accumulate": 2,
        "network_g": (
            'type = "atd"\n'
            'upscale = 1\n'
            'img_size = 96\n'
            'window_size = 8\n'
            'embed_dim = 90'
        ),
    },
    "drct_s": {
        "scale": 1,
        "desc": "DRCT-S (factory defaults, 1x) — lighter than DRCT",
        "is_esc": False,
        "network_g": (
            'type = "drct_s"\n'
            'upscale = 1'
        ),
    },
    # ── Other ──────────────────────────────────────────────────────────────────
    "moesr": {
        "scale": 1,
        "desc": "MoESR (dim=48, n_blocks=24, conv upsampler, 1x)",
        "is_esc": False,
        "network_g": (
            'type = "moesr"\n'
            'in_ch = 3\n'
            'out_ch = 3\n'
            'scale = 1\n'
            'dim = 48\n'
            'n_blocks = 24\n'
            'n_block = 4\n'
            'upsampler = "conv"'
        ),
    },
}

# ── TOML template ──────────────────────────────────────────────────────────────
_TOML_TEMPLATE = """\
name = "Bench_{bench_name}"
model_type = "otf"
scale = {scale}
num_gpu = 1
manual_seed = 42
use_amp = {use_amp}
bfloat16 = {use_bf16}
fast_matmul = {fast_matmul}
compile = false

[monitoring]
auto_tensorboard = false
port = 6006
auto_ngrok = false

[degradations]
resize_prob = [ 0.2, 0.7, 0.1,]
resize_range = [ 0.4, 1.5,]
gaussian_noise_prob = 0.18
noise_range = [ 1, 10,]
poisson_scale_range = [ 0.05, 2.0,]
gray_noise_prob = 0.1
blur_kernel_size = 21
kernel_list = [ "iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso",]
kernel_prob = [ 0.45, 0.25, 0.12, 0.03, 0.12, 0.03,]
sinc_prob = 0.1
blur_sigma = [ 0.2, 1.5,]
betag_range = [ 0.5, 4,]
betap_range = [ 1, 2,]
second_blur_prob = 0.27
resize_prob2 = [ 0.3, 0.4, 0.3,]
resize_range2 = [ 0.6, 1.2,]
gaussian_noise_prob2 = 0.13
noise_range2 = [ 1, 8,]
poisson_scale_range2 = [ 0.05, 2.5,]
gray_noise_prob2 = 0.1
blur_kernel_size2 = 21
kernel_list2 = [ "iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso",]
kernel_prob2 = [ 0.45, 0.25, 0.12, 0.03, 0.12, 0.03,]
sinc_prob2 = 0.1
blur_sigma2 = [ 0.2, 1.0,]
betag_range2 = [ 0.5, 4,]
betap_range2 = [ 1, 2,]
jpeg_range = [ 80, 99,]
jpeg_range2 = [ 75, 99,]
final_sinc_prob = 0.8
jpeg_prob = 0.5
posterize_prob = 0.21
posterize_bits_range = [ 3, 6,]
banding_prob = 0.59
banding_levels_range = [ 16, 64,]
chroma_prob = 0.0
ca_prob = 0.0
ca_shift_range = [ 1, 3,]
halation_prob = 0.0
halation_strength_range = [ 0.05, 0.15,]
salt_pepper_prob = 0.0
salt_pepper_amount_range = [ 0.001, 0.01,]
vhs_prob = 0.0
vhs_strength_range = [ 0.1, 0.2,]

[network_g]
{network_g}

[path]
strict_load_g = false

[train]
total_iter = {n_iter}
n_iter = {n_iter}
warmup_iter = 0
ema = 0.999
grad_clip = false
match_lq_colors = false

[logger]
total_iter = {n_iter}
print_freq = {print_freq}
save_checkpoint_freq = {n_iter}
use_tb_logger = false

[val]
val_freq = {val_freq}
save_img = false
pbar = false
tile = 256
tile_pad = 32

[datasets.train]
type = "otf"
name = "BenchTrain"
dataroot_gt = "{train_gt}"
num_worker_per_gpu = 2
prefetch_mode = "cpu"
batch_size = {batch_size}
accumulate = {accumulate}
patch_size = 96
use_shuffle = true
use_hflip = true
use_rot = true

[datasets.val]
name = "BenchVal"
type = "paired"
dataroot_gt = "{val_gt}"
dataroot_lq = "{val_lq}"

[datasets.val.io_backend]
type = "disk"

[datasets.train.io_backend]
type = "disk"

[train.optim_g]
type = "AdamW_SF"
lr = 4.79e-5
weight_decay = 0
betas = [ 0.9, 0.99,]
schedule_free = true
warmup_steps = -1

[train.scheduler]
type = "MultiStepLR"
milestones = [ 999999,]
gamma = 0.5

[train.pixel_opt]
type = "chc_loss"
loss_weight = 1.0
reduction = "mean"

[train.ldl_opt]
type = "ldl_loss"
loss_weight = 1.0

[val.metrics.psnr]
type = "calculate_psnr"
crop_border = 1
test_y_channel = true

[val.metrics.ssim]
type = "calculate_ssim"
crop_border = 1
test_y_channel = true
"""

# Lines to print live from subprocess stdout (case-insensitive)
_LIVE_KEYWORDS = ('it/s', 'gpu mem', 'psnr', 'ssim', 'error', 'erreur',
                  'traceback', 'eta:', 'warning', 'started', 'finish',
                  'saving', 'checkpoint', 'validation')


def _generate_toml(bench_name: str, arch_cfg: dict, n_iter: int,
                   train_gt: str, val_gt: str, val_lq: str,
                   use_amp: bool, use_bf16: bool, fast_matmul: bool = False,
                   enable_val: bool = False) -> str:
    # Val disabled by default: archs are incompatible with a 4x val_lq dataset.
    # Use --enable-val with a matching 1x val dataset to measure PSNR.
    val_freq = max(200, n_iter // 2) if enable_val else n_iter * 1000
    print_freq = max(50, n_iter // 50)
    batch_size = arch_cfg.get("batch_size", 4)
    accumulate = arch_cfg.get("accumulate", 1)
    return _TOML_TEMPLATE.format(
        bench_name=bench_name,
        scale=arch_cfg["scale"],
        use_amp="true" if use_amp else "false",
        use_bf16="true" if use_bf16 else "false",
        fast_matmul="true" if fast_matmul else "false",
        network_g=arch_cfg["network_g"],
        n_iter=n_iter,
        print_freq=print_freq,
        val_freq=val_freq,
        train_gt=train_gt.replace("\\", "/"),
        val_gt=val_gt.replace("\\", "/"),
        val_lq=val_lq.replace("\\", "/"),
        batch_size=batch_size,
        accumulate=accumulate,
    )


# ── Metric parsers ─────────────────────────────────────────────────────────────
_RE_ITPS = re.compile(r"([\d.]+)\s*it/s", re.IGNORECASE)
_RE_VRAM = re.compile(r"(?:GPU\s+mem|VRAM)[:\s]+([\d.]+)\s*GB", re.IGNORECASE)
_RE_PSNR = re.compile(r"psnr[:\s]+([\d.]+)", re.IGNORECASE)
_RE_SSIM = re.compile(r"ssim[:\s]+([\d.]+)", re.IGNORECASE)
_RE_ITER = re.compile(r"iter[:\s]+(\d+)", re.IGNORECASE)


_NVIDIA_SMI_CANDIDATES = [
    "nvidia-smi",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
]
_nvidia_smi_exe: str | None = None


def _find_nvidia_smi() -> str | None:
    global _nvidia_smi_exe
    if _nvidia_smi_exe is not None:
        return _nvidia_smi_exe
    for candidate in _NVIDIA_SMI_CANDIDATES:
        try:
            r = subprocess.run(
                [candidate, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                _nvidia_smi_exe = candidate
                return candidate
        except Exception:
            continue
    _nvidia_smi_exe = ""
    return None


def _nvidia_smi_vram_mb() -> float | None:
    exe = _find_nvidia_smi()
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


# ── Upscale test ───────────────────────────────────────────────────────────────
def _run_upscale_test(arch_name: str, model_path: Path, test_image: Path,
                      output_dir: Path, is_esc: bool,
                      mode_label: str = "NORMAL") -> dict:
    """Run a quick upscale on the benchmark checkpoint. Returns status dict."""
    out_img = output_dir / "upscale_tests" / f"{mode_label}_Bench_{arch_name}_{model_path.stem}.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)

    if not test_image.exists():
        return {"status": "skip", "reason": f"test image not found: {test_image}"}

    if is_esc:
        py, runner = NEOSR_PYTHON, NEOSR_RUNNER
    elif NEOSR_GENERAL_RUNNER.exists():
        # Use neosr venv for neosr-native archs (ninasr, lmlt, eimn, drct_s, etc.)
        py, runner = NEOSR_PYTHON, NEOSR_GENERAL_RUNNER
    else:
        py, runner = TRAINNER_PYTHON, UNIVERSAL_RUNNER

    if not py.exists():
        return {"status": "skip", "reason": f"venv not found: {py}"}
    if not runner.exists():
        return {"status": "skip", "reason": f"runner not found: {runner}"}

    print(f"    [upscale] {arch_name}: {model_path.name} → {out_img.name}", flush=True)
    try:
        r = subprocess.run(
            [str(py), str(runner), str(model_path), str(test_image), str(out_img)],
            capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        if r.returncode == 0 and out_img.exists():
            sz = out_img.stat().st_size
            print(f"    [upscale] OK — {out_img.name} ({sz // 1024} KB)", flush=True)
            return {"status": "ok", "output": str(out_img), "size_kb": sz // 1024}
        else:
            reason = (r.stdout + r.stderr).strip()[-400:]
            print(f"    [upscale] ERREUR — {reason[:120]}", flush=True)
            return {"status": "error", "reason": reason}
    except subprocess.TimeoutExpired:
        print(f"    [upscale] timeout (300s)", flush=True)
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── Per-arch benchmark ─────────────────────────────────────────────────────────
def run_benchmark_arch(
    arch_name: str,
    arch_cfg: dict,
    n_iter: int,
    timeout_sec: int,
    train_gt: str,
    val_gt: str,
    val_lq: str,
    use_amp: bool,
    use_bf16: bool,
    fast_matmul: bool,
    keep_toml: bool,
    output_dir: Path,
    test_image: Path,
    do_upscale: bool,
    mode_label: str = "NORMAL",
    enable_val: bool = False,
) -> dict:
    # bench_name uniquely identifies the experiment (arch + precision mode)
    bench_name = f"{arch_name}_{mode_label}" if mode_label != "NORMAL" else arch_name
    result = {
        "arch": arch_name,
        "mode": mode_label,
        "desc": arch_cfg.get("desc", ""),
        "scale": arch_cfg["scale"],
        "n_iter": n_iter,
        "use_amp": use_amp,
        "use_bf16": use_bf16,
        "fast_matmul": fast_matmul,
        "status": "error",
        "error": None,
        "avg_itps": None,
        "peak_vram_gb": None,
        "peak_vram_smi_gb": None,
        "psnr_readings": {},
        "ssim_readings": {},
        "elapsed_sec": 0,
        "iters_completed": 0,
        "upscale_test": None,
    }

    toml_path = output_dir / f"_bench_{bench_name}.toml"
    try:
        toml_text = _generate_toml(bench_name, arch_cfg, n_iter,
                                    train_gt, val_gt, val_lq, use_amp, use_bf16,
                                    fast_matmul=fast_matmul, enable_val=enable_val)
        toml_path.write_text(toml_text, encoding="utf-8")
    except Exception as e:
        result["error"] = f"TOML generation failed: {e}"
        return result

    itps_readings: list[float] = []
    # neosr_vram: values from neosr's own log (arch allocation only — preferred)
    # smi_vram:   values from nvidia-smi (system total — fallback)
    neosr_vram_readings: list[float] = []
    smi_vram_readings: list[float] = []
    psnr_map: dict[int, float] = {}
    ssim_map: dict[int, float] = {}
    current_iter = [0]
    stdout_lines: list[str] = []

    stop_vram = threading.Event()

    def vram_monitor():
        while not stop_vram.is_set():
            v = _nvidia_smi_vram_mb()
            if v is not None:
                smi_vram_readings.append(v / 1024.0)
            stop_vram.wait(4)

    vram_thread = threading.Thread(target=vram_monitor, daemon=True)
    vram_thread.start()

    t_start = time.time()
    proc = None
    try:
        proc = subprocess.Popen(
            [str(NEOSR_PYTHON), str(TRAIN_SCRIPT), "-opt", str(toml_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(NEOSR_PATH),
        )

        def stdout_reader():
            for line in proc.stdout:
                stripped = line.rstrip()
                stdout_lines.append(stripped)

                m = _RE_ITPS.search(stripped)
                if m:
                    itps_readings.append(float(m.group(1)))
                m = _RE_VRAM.search(stripped)
                if m:
                    neosr_vram_readings.append(float(m.group(1)))
                m = _RE_ITER.search(stripped)
                if m:
                    current_iter[0] = int(m.group(1))
                # PSNR/SSIM only from validation output lines
                lower = stripped.lower()
                if "psnr" in lower and ("val" in lower or "metric" in lower or "#" in lower):
                    mp = _RE_PSNR.search(stripped)
                    ms = _RE_SSIM.search(stripped)
                    it = current_iter[0]
                    if mp:
                        psnr_map[it] = float(mp.group(1))
                    if ms:
                        ssim_map[it] = float(ms.group(1))
                    print(f"  [{arch_name}] {stripped}", flush=True)
                elif any(k in lower for k in _LIVE_KEYWORDS):
                    print(f"  [{arch_name}] {stripped}", flush=True)

        reader_thread = threading.Thread(target=stdout_reader, daemon=True)
        reader_thread.start()

        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            print(f"  [{arch_name}] timeout ({timeout_sec}s) — killing process", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        reader_thread.join(timeout=10)
        exit_code = proc.returncode

    except Exception as e:
        result["error"] = f"Subprocess launch failed: {e}"
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        stop_vram.set()
        if not keep_toml and toml_path.exists():
            try:
                toml_path.unlink()
            except Exception:
                pass
        return result
    finally:
        stop_vram.set()
        if not keep_toml and toml_path.exists():
            try:
                toml_path.unlink()
            except Exception:
                pass

    elapsed = time.time() - t_start
    result["elapsed_sec"] = round(elapsed, 1)
    result["iters_completed"] = current_iter[0]

    if itps_readings:
        drop = max(1, len(itps_readings) // 10)
        stable = itps_readings[drop:] if len(itps_readings) > drop + 2 else itps_readings
        result["avg_itps"] = round(sum(stable) / len(stable), 4)

    # peak_vram_gb    = neosr's own reported allocation (arch-specific, preferred)
    # peak_vram_smi_gb = nvidia-smi system total (all processes on GPU)
    if neosr_vram_readings:
        result["peak_vram_gb"] = round(max(neosr_vram_readings), 2)
    elif smi_vram_readings:
        result["peak_vram_gb"] = round(max(smi_vram_readings), 2)
    if smi_vram_readings:
        result["peak_vram_smi_gb"] = round(max(smi_vram_readings), 2)

    result["psnr_readings"] = {str(k): round(v, 4) for k, v in sorted(psnr_map.items())}
    result["ssim_readings"] = {str(k): round(v, 4) for k, v in sorted(ssim_map.items())}

    if exit_code == 0 or current_iter[0] >= n_iter - 1:
        result["status"] = "ok"
        # Iter regex sometimes misses final log line — clamp to n_iter on clean exit
        if exit_code == 0 and current_iter[0] < n_iter // 2:
            result["iters_completed"] = n_iter
    elif not itps_readings:
        result["status"] = "error"
        result["error"] = f"exit_code={exit_code}\n" + "\n".join(stdout_lines[-25:])
    else:
        result["status"] = "timeout"

    # ── Quick upscale test ─────────────────────────────────────────────────────
    if do_upscale and result["status"] in ("ok", "timeout"):
        # neosr saves to {NEOSR_PATH}/experiments/Bench_{bench_name}/models/
        models_dir = NEOSR_PATH / "experiments" / f"Bench_{bench_name}" / "models"
        model_path = models_dir / f"net_g_{n_iter}.pth"
        if not model_path.exists():
            # Try closest saved checkpoint
            candidates = sorted(models_dir.glob("net_g_*.pth")) if models_dir.exists() else []
            model_path = candidates[-1] if candidates else None

        if model_path and model_path.exists():
            result["upscale_test"] = _run_upscale_test(
                arch_name, model_path, test_image, output_dir,
                is_esc=arch_cfg.get("is_esc", False),
                mode_label=mode_label,
            )
        else:
            result["upscale_test"] = {"status": "skip", "reason": "model checkpoint not found"}
            print(f"  [{arch_name}] checkpoint not found at {models_dir}", flush=True)

    return result


# ── Report ─────────────────────────────────────────────────────────────────────
def _format_report(results: list[dict], gpu_name: str, ts: str) -> str:
    multi_mode = len({r.get("mode", "NORMAL") for r in results}) > 1
    lines = [
        "=" * 90,
        f"  neosr Architecture Benchmark  —  {ts}",
        f"  GPU: {gpu_name}",
        "=" * 90,
        "",
    ]
    if multi_mode:
        lines.append(
            f"{'Arch':<14} {'Mode':<8} {'Status':<9} {'it/s':>7} {'VRAM GB':>8} {'SMI GB':>7}"
            f" {'PSNR@end':>9} {'SSIM@end':>9} {'Iters':>6}"
        )
        lines.append("-" * 98)
    else:
        lines.append(
            f"{'Arch':<14} {'Status':<9} {'it/s':>7} {'VRAM GB':>8} {'SMI GB':>7}"
            f" {'PSNR@end':>9} {'SSIM@end':>9} {'Iters':>6}"
        )
        lines.append("-" * 86)

    for r in results:
        itps  = f"{r['avg_itps']:.3f}" if r["avg_itps"] else "  —  "
        vram  = f"{r['peak_vram_gb']:.2f}" if r["peak_vram_gb"] else "  —  "
        smi   = f"{r['peak_vram_smi_gb']:.2f}" if r.get("peak_vram_smi_gb") else "  —  "
        psnr_vals = list(r["psnr_readings"].values())
        ssim_vals = list(r["ssim_readings"].values())
        psnr = f"{psnr_vals[-1]:.2f}" if psnr_vals else "  —  "
        ssim = f"{ssim_vals[-1]:.4f}" if ssim_vals else "  —  "
        iters = str(r["iters_completed"])
        ut = r.get("upscale_test") or {}
        up_status = ut.get("status", "—")
        mode = r.get("mode", "NORMAL")
        if multi_mode:
            lines.append(
                f"{r['arch']:<14} {mode:<8} {r['status']:<9} {itps:>7} {vram:>8} {smi:>7}"
                f" {psnr:>9} {ssim:>9} {iters:>6}   upscale:{up_status}"
            )
        else:
            lines.append(
                f"{r['arch']:<14} {r['status']:<9} {itps:>7} {vram:>8} {smi:>7}"
                f" {psnr:>9} {ssim:>9} {iters:>6}   upscale:{up_status}"
            )
    lines.append("")
    lines.append(f"{'Arch':<14}  Description")
    lines.append("-" * 78)
    for r in results:
        lines.append(f"{r['arch']:<14}  {r['desc']}")
    lines.append("")

    # Error details
    for r in results:
        if r["status"] == "error" and r["error"]:
            lines.append(f"[ERROR — {r['arch']}]")
            lines.append(r["error"][:600])
            lines.append("")

    # PSNR progression table
    psnr_archs = [r for r in results if r["psnr_readings"]]
    if psnr_archs:
        lines.append("PSNR progression:")
        for r in psnr_archs:
            pts = "  ".join(f"@{k}={v:.2f}" for k, v in r["psnr_readings"].items())
            lines.append(f"  {r['arch']:<14} {pts}")
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)


# ── Machine type selection ─────────────────────────────────────────────────────
def _ask_machine_type(args_laptop: bool, args_desktop: bool) -> str:
    """Returns 'laptop' or 'desktop'. Prompts if neither flag was given."""
    if args_laptop and not args_desktop:
        return "laptop"
    if args_desktop and not args_laptop:
        return "desktop"
    # Interactive prompt
    print("\n" + "=" * 60)
    print("  Machine type?")
    print("  1. laptop  — RTX (test 4 precision modes per arch)")
    print("  2. desktop — GTX (single NORMAL run per arch)")
    print("=" * 60)
    while True:
        choice = input("  Choice [1/2] or [laptop/desktop]: ").strip().lower()
        if choice in ("1", "laptop", "l"):
            return "laptop"
        if choice in ("2", "desktop", "d"):
            return "desktop"
        print("  Please enter 1 or 2.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="neosr arch benchmark")
    parser.add_argument("--n-iter",     type=int,  default=2500)
    parser.add_argument("--timeout",    type=int,  default=3600)
    parser.add_argument("--archs",      type=str,  default="")
    parser.add_argument("--train-gt",   type=str,  default=DEFAULT_TRAIN_GT)
    parser.add_argument("--val-gt",     type=str,  default=DEFAULT_VAL_GT)
    parser.add_argument("--val-lq",     type=str,  default=DEFAULT_VAL_LQ)
    parser.add_argument("--amp",        action="store_true",
                        help="(desktop override) force AMP — ignored in laptop mode")
    parser.add_argument("--bf16",       action="store_true",
                        help="(desktop override) force BF16 — ignored in laptop mode")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path.home() / "IA_Engine" / "benchmark_results"))
    parser.add_argument("--keep-toml",  action="store_true")
    parser.add_argument("--no-upscale",  action="store_true")
    parser.add_argument("--enable-val",  action="store_true",
                        help="Enable validation/PSNR — requires val_lq at same scale as archs")
    parser.add_argument("--test-image", type=str, default=DEFAULT_TEST_IMG)
    parser.add_argument("--laptop",  action="store_true",
                        help="Laptop mode: 4 precision modes per arch (NORMAL, FP16, BF16, TF32)")
    parser.add_argument("--desktop", action="store_true",
                        help="Desktop mode: single NORMAL run per arch")
    args = parser.parse_args()

    if not NEOSR_PYTHON.exists():
        print(f"[Benchmark] ERROR: neosr venv not found: {NEOSR_PYTHON}")
        sys.exit(1)
    if not TRAIN_SCRIPT.exists():
        print(f"[Benchmark] ERROR: train.py not found: {TRAIN_SCRIPT}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_image = Path(args.test_image)

    # Machine type selection (prompt if neither flag given)
    machine = _ask_machine_type(args.laptop, args.desktop)
    precision_modes = PRECISION_MODES if machine == "laptop" else DESKTOP_MODE
    # In desktop mode, --amp / --bf16 flags still override the single NORMAL mode
    if machine == "desktop":
        precision_modes = [{"label": "NORMAL", "use_amp": args.amp,
                            "use_bf16": args.bf16, "fast_matmul": False}]

    # Detect GPU
    gpu_name = "Unknown"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            gpu_name = r.stdout.strip().split("\n")[0].replace("NVIDIA GeForce ", "").strip()
    except Exception:
        pass

    # Select archs
    if args.archs:
        selected = [a.strip() for a in args.archs.split(",") if a.strip()]
        archs = {k: v for k, v in ARCHS_TO_TEST.items() if k in selected}
        unknown = [a for a in selected if a not in ARCHS_TO_TEST]
        if unknown:
            print(f"[Benchmark] WARNING: unknown arch(es): {unknown}")
    else:
        archs = ARCHS_TO_TEST

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    mode_names = [m["label"] for m in precision_modes]
    total_runs = len(archs) * len(precision_modes)
    print(f"\n{'=' * 70}")
    print(f"  neosr Benchmark — {machine.upper()} mode — GPU: {gpu_name}")
    print(f"  {ts}")
    print(f"  Archs  : {list(archs.keys())}")
    print(f"  Modes  : {mode_names}  ({total_runs} total runs)")
    print(f"  N iter : {args.n_iter}  |  Timeout: {args.timeout}s  |  Val/PSNR: {args.enable_val}")
    print(f"  Test image: {test_image.name}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}\n")

    all_results = []
    run_count = 0
    for arch_name, arch_cfg in archs.items():
        for mode in precision_modes:
            run_count += 1
            label = mode["label"]
            print(
                f"\n[Benchmark] {'─' * 8} [{run_count}/{total_runs}] "
                f"{arch_name} [{label}]: {arch_cfg['desc']} {'─' * 8}",
                flush=True,
            )
            result = run_benchmark_arch(
                arch_name=arch_name,
                arch_cfg=arch_cfg,
                n_iter=args.n_iter,
                timeout_sec=args.timeout,
                train_gt=args.train_gt,
                val_gt=args.val_gt,
                val_lq=args.val_lq,
                use_amp=mode["use_amp"],
                use_bf16=mode["use_bf16"],
                fast_matmul=mode["fast_matmul"],
                keep_toml=args.keep_toml,
                output_dir=output_dir,
                test_image=test_image,
                do_upscale=not args.no_upscale,
                mode_label=label,
                enable_val=args.enable_val,
            )
            all_results.append(result)

            summary = f"  → [{label}] status={result['status']}"
            if result["avg_itps"]:
                summary += f"  it/s={result['avg_itps']:.3f}"
            if result["peak_vram_gb"]:
                summary += f"  VRAM={result['peak_vram_gb']:.2f}GB"
            if result["psnr_readings"]:
                last_psnr = list(result["psnr_readings"].values())[-1]
                summary += f"  PSNR={last_psnr:.2f}"
            if result["status"] == "error":
                summary += f"\n  ERROR: {(result['error'] or '')[:200]}"
            print(summary, flush=True)

            # Save partial results after each run so nothing is lost on crash
            _save_results(all_results, output_dir, ts_file, gpu_name, ts, partial=True)

    # Final save
    _save_results(all_results, output_dir, ts_file, gpu_name, ts, partial=False)


def _save_results(results: list[dict], output_dir: Path, ts_file: str,
                  gpu_name: str, ts: str, partial: bool) -> None:
    suffix = "_partial" if partial else ""
    json_path = output_dir / f"benchmark_{ts_file}{suffix}.json"
    txt_path  = output_dir / f"benchmark_{ts_file}{suffix}.txt"

    json_path.write_text(
        json.dumps({"gpu": gpu_name, "timestamp": ts, "results": results}, indent=2),
        encoding="utf-8",
    )
    report = _format_report(results, gpu_name, ts)
    txt_path.write_text(report, encoding="utf-8")

    if not partial:
        # Append to cumulative history log (never overwritten)
        history_path = output_dir / "benchmark_history.txt"
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(report)
            f.write("\n\n")
        print(report, flush=True)
        print(f"\nResults saved:")
        print(f"  {json_path}")
        print(f"  {txt_path}")
        print(f"  {history_path}  (cumulative)")


if __name__ == "__main__":
    main()
