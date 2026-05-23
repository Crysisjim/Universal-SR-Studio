"""
feature_benchmark.py — Feature Coverage Benchmark (NeoSR, GTX 1080 Ti)

Teste chaque fonction configurable de Universal SR Studio en utilisant
compact comme modèle de base (11.86 it/s, 0.73 GB).

Une seule variable change par test — permet d'isoler l'impact réel.
Mesures : it/s, VRAM (PyTorch + SMI), PSNR@fin, statut upscale.

Usage:
    python feature_benchmark.py [options]

    --n-iter 2500          Itérations par test (défaut : 2500)
    --timeout 1800         Timeout par test en secondes (défaut : 1800)
    --category losses      Catégorie seulement (losses/optimizer/scheduler/
                           gan/augmentation/precision/system)
    --tests baseline,loss_mse  Tests nommés seulement (virgule)
    --no-upscale           Passer le test upscale
    --output-dir PATH      Dossier résultats (défaut : ~/IA_Engine/benchmark_results/feature_bench)
    --train-gt PATH        Dataset training GT
    --keep-toml            Conserver les TOML temporaires
    --list                 Lister tous les tests disponibles et quitter
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

# ── Paths ──────────────────────────────────────────────────────────────────────
NEOSR_PATH   = Path.home() / "IA_Engine" / "neosr"
NEOSR_PYTHON = NEOSR_PATH / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = NEOSR_PATH / "train.py"

_THIS_DIR            = Path(__file__).parent
NEOSR_GENERAL_RUNNER = _THIS_DIR / "neosr_general_runner.py"

DEFAULT_TRAIN_GT = str(Path.home() / "IA_Engine" / "datasets" / "train" / "HR")
DEFAULT_VAL_GT   = str(Path.home() / "IA_Engine" / "datasets" / "val" / "GT")
DEFAULT_VAL_LQ   = str(Path.home() / "IA_Engine" / "datasets" / "val" / "LQ")
DEFAULT_TEST_IMG = ""  # Set via --train-gt or leave empty to skip upscale test

# ── compact network_g (base arch for all tests) ────────────────────────────────
_COMPACT_NETWORK_G = (
    'type = "compact"\n'
    'num_in_ch = 3\n'
    'num_out_ch = 3\n'
    'num_feat = 64\n'
    'num_block = 16\n'
    'upscale = 1'
)

# ── Fixed TOML header (degradations + network_g + datasets + logger + val) ─────
_TOML_FIXED = """\
name = "FBench_{test_name}"
model_type = "otf"
scale = 1
num_gpu = 1
manual_seed = 42
use_amp = {use_amp}
bfloat16 = {use_bf16}
fast_matmul = {fast_matmul}
compile = {compile}

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

[network_g]
{network_g}

[path]
strict_load_g = false

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
name = "FBenchTrain"
dataroot_gt = "{train_gt}"
num_worker_per_gpu = 2
prefetch_mode = "cpu"
batch_size = {batch_size}
accumulate = {accumulate}
patch_size = {patch_size}
use_shuffle = true
use_hflip = true
use_rot = true
{augmentation_lines}
[datasets.val]
name = "FBenchVal"
type = "paired"
dataroot_gt = "{val_gt}"
dataroot_lq = "{val_lq}"

[datasets.val.io_backend]
type = "disk"

[datasets.train.io_backend]
type = "disk"

[val.metrics.psnr]
type = "calculate_psnr"
crop_border = 1
test_y_channel = true

[val.metrics.ssim]
type = "calculate_ssim"
crop_border = 1
test_y_channel = true
"""

# ── Default optimizer / scheduler blocks ──────────────────────────────────────
_DEFAULT_OPTIM = """\
[train.optim_g]
type = "AdamW"
lr = 2e-4
weight_decay = 0
betas = [ 0.9, 0.99,]
"""

_DEFAULT_SCHEDULER = """\
[train.scheduler]
type = "CosineAnnealing"
T_max = {n_iter}
eta_min = 1e-7
"""

_DEFAULT_PIXEL_OPT = """\
[train.pixel_opt]
type = "L1Loss"
loss_weight = 1.0
reduction = "mean"
"""

# ── Feature test registry ──────────────────────────────────────────────────────
# Each test is a dict:
#   name        : str   — unique identifier (also used as bench_name)
#   desc        : str   — human-readable description
#   category    : str   — losses / optimizer / scheduler / gan / augmentation / precision / system
#   requires_amp: bool  — skip on Pascal (GTX 10xx, no AMP)
#   requires_compile: bool — skip on Pascal
#   pixel_opt   : str   — TOML block [train.pixel_opt] (default: L1Loss)
#   extra_losses: list  — additional TOML blocks (perceptual, ldl, fdl, dists, gan…)
#   optim       : str   — TOML block [train.optim_g] (default: AdamW)
#   scheduler   : str or None — TOML block [train.scheduler] (None = omit)
#   network_d   : str or None — TOML block [network_d]
#   optim_d     : str or None — TOML block [train.optim_d]
#   use_amp     : bool  (default False)
#   use_bf16    : bool  (default False)
#   fast_matmul : bool  (default False)
#   compile_    : bool  (default False)
#   batch_size  : int   (default 4)
#   accumulate  : int   (default 1)
#   patch_size  : int   (default 96)
#   warmup_iter : int   (default 0)
#   ema         : float (default 0.999)
#   grad_clip   : bool  (default False)
#   augmentation: str or None — lines appended to [datasets.train]

def _mk(name, desc, category, **kw):
    base = dict(
        name=name, desc=desc, category=category,
        requires_amp=False, requires_compile=False,
        pixel_opt=None, extra_losses=[],
        optim=None, scheduler=None,
        network_d=None, optim_d=None,
        use_amp=False, use_bf16=False, fast_matmul=False, compile_=False,
        batch_size=4, accumulate=1, patch_size=96,
        warmup_iter=0, ema=0.999, grad_clip=False,
        augmentation=None,
    )
    base.update(kw)
    return base


FEATURE_TESTS = [
    # ── BASELINE ──────────────────────────────────────────────────────────────
    _mk("baseline",
        "Baseline : compact, L1Loss, AdamW, CosineAnnealing, OTF, batch=4",
        "baseline"),

    # ── LOSSES — pixel_opt variations ─────────────────────────────────────────
    _mk("loss_l1_sum",
        "L1Loss reduction=sum",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "L1Loss"\nloss_weight = 1.0\nreduction = "sum"\n'),

    _mk("loss_mse",
        "MSELoss",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "MSELoss"\nloss_weight = 1.0\nreduction = "mean"\n'),

    _mk("loss_huber",
        "HuberLoss",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "HuberLoss"\nloss_weight = 1.0\n'),

    _mk("loss_chc",
        "CHC loss (Charbonnier)",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "chc_loss"\nloss_weight = 1.0\nreduction = "mean"\n'),

    _mk("loss_mssim",
        "MS-SSIM loss",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "mssim_loss"\nloss_weight = 1.0\n'),

    _mk("loss_ncc",
        "NCC loss (Normalized Cross-Correlation)",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "ncc_loss"\nloss_weight = 1.0\n'),

    _mk("loss_ff",
        "FF loss (Focal Frequency)",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "ff_loss"\nloss_weight = 1.0\n'),

    _mk("loss_consistency_chc",
        "Consistency loss criterion=chc (Oklab chroma + CIE L*)",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "consistency_loss"\nloss_weight = 1.0\ncriterion = "chc"\n'),

    _mk("loss_consistency_l1",
        "Consistency loss criterion=l1",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "consistency_loss"\nloss_weight = 1.0\ncriterion = "l1"\n'),

    _mk("loss_dists",
        "DISTS loss (weight=0.5)",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "dists_loss"\nloss_weight = 0.5\n'),

    _mk("loss_ldl",
        "LDL loss criterion=huber",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "ldl_loss"\nloss_weight = 1.0\ncriterion = "huber"\n'),

    # VGG perceptual — criterion variations
    _mk("loss_vgg_huber_conv44",
        "VGG perceptual criterion=huber layer=conv4_4",
        "losses",
        pixel_opt=(
            '[train.pixel_opt]\ntype = "vgg_perceptual_loss"\nloss_weight = 1.0\n'
            'criterion = "huber"\n\n'
            '[train.pixel_opt.layer_weights]\n"conv4_4" = 1.0\n'
        )),

    _mk("loss_vgg_l1_conv44",
        "VGG perceptual criterion=l1 layer=conv4_4",
        "losses",
        pixel_opt=(
            '[train.pixel_opt]\ntype = "vgg_perceptual_loss"\nloss_weight = 1.0\n'
            'criterion = "l1"\n\n'
            '[train.pixel_opt.layer_weights]\n"conv4_4" = 1.0\n'
        )),

    _mk("loss_vgg_l2_conv44",
        "VGG perceptual criterion=l2 (MSELoss) layer=conv4_4",
        "losses",
        pixel_opt=(
            '[train.pixel_opt]\ntype = "vgg_perceptual_loss"\nloss_weight = 1.0\n'
            'criterion = "l2"\n\n'
            '[train.pixel_opt.layer_weights]\n"conv4_4" = 1.0\n'
        )),

    _mk("loss_vgg_multilayer",
        "VGG perceptual conv3_4+conv4_4+conv5_4",
        "losses",
        pixel_opt=(
            '[train.pixel_opt]\ntype = "vgg_perceptual_loss"\nloss_weight = 1.0\n'
            'criterion = "huber"\n\n'
            '[train.pixel_opt.layer_weights]\n"conv3_4" = 0.5\n"conv4_4" = 1.0\n"conv5_4" = 0.5\n'
        )),

    _mk("loss_fdl_vgg",
        "FDL loss model=vgg",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "fdl_loss"\nloss_weight = 1.0\nmodel = "vgg"\n'),

    _mk("loss_fdl_dinov2",
        "FDL loss model=dinov2 (heavy — peut timeout)",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "fdl_loss"\nloss_weight = 1.0\nmodel = "dinov2"\n'),

    # Combined losses
    _mk("loss_l1_ldl",
        "L1Loss + LDL loss (stack classique qualité)",
        "losses",
        extra_losses=[
            '[train.ldl_opt]\ntype = "ldl_loss"\nloss_weight = 1.0\ncriterion = "huber"\n'
        ]),

    _mk("loss_l1_vgg_ldl",
        "L1Loss + VGG perceptual + LDL (stack qualité complet)",
        "losses",
        extra_losses=[
            (
                '[train.perceptual_opt]\ntype = "vgg_perceptual_loss"\nloss_weight = 1.0\n'
                'criterion = "huber"\n\n'
                '[train.perceptual_opt.layer_weights]\n"conv4_4" = 1.0\n'
            ),
            '[train.ldl_opt]\ntype = "ldl_loss"\nloss_weight = 1.0\ncriterion = "huber"\n',
        ]),

    _mk("loss_chc_ldl",
        "CHC + LDL (stack bench arch_benchmark)",
        "losses",
        pixel_opt='[train.pixel_opt]\ntype = "chc_loss"\nloss_weight = 1.0\nreduction = "mean"\n',
        extra_losses=[
            '[train.ldl_opt]\ntype = "ldl_loss"\nloss_weight = 1.0\n'
        ]),

    # ── OPTIMIZERS ────────────────────────────────────────────────────────────
    _mk("optim_adam",
        "Adam (lr=2e-4)",
        "optimizer",
        optim='[train.optim_g]\ntype = "Adam"\nlr = 2e-4\nweight_decay = 0\nbetas = [ 0.9, 0.99,]\n'),

    # AdamW = baseline, already tested as "baseline"

    _mk("optim_nadam",
        "NAdam (lr=2e-4)",
        "optimizer",
        optim='[train.optim_g]\ntype = "NAdam"\nlr = 2e-4\nweight_decay = 0\nbetas = [ 0.9, 0.99,]\n'),

    _mk("optim_adan",
        "Adan (lr=2e-4, 3 betas)",
        "optimizer",
        optim='[train.optim_g]\ntype = "Adan"\nlr = 2e-4\nweight_decay = 0\nbetas = [ 0.98, 0.92, 0.99,]\n'),

    _mk("optim_adamw_win",
        "AdamW_Win (variante accélérée)",
        "optimizer",
        optim='[train.optim_g]\ntype = "AdamW_Win"\nlr = 2e-4\nweight_decay = 0\nbetas = [ 0.9, 0.99,]\n'),

    _mk("optim_adamw_sf",
        "AdamW_SF (Schedule-Free — pas de scheduler)",
        "optimizer",
        optim=(
            '[train.optim_g]\ntype = "AdamW_SF"\nlr = 2e-4\nweight_decay = 0\n'
            'betas = [ 0.9, 0.99,]\nschedule_free = true\nwarmup_steps = 200\n'
        ),
        scheduler="none"),  # SF ne veut pas de scheduler externe

    _mk("optim_adan_sf",
        "Adan_SF (Schedule-Free Adan)",
        "optimizer",
        optim=(
            '[train.optim_g]\ntype = "Adan_SF"\nlr = 2e-4\nweight_decay = 0\n'
            'betas = [ 0.98, 0.92, 0.99,]\nschedule_free = true\nwarmup_steps = 200\n'
        ),
        scheduler="none"),

    _mk("optim_soap_sf",
        "SOAP_SF (Schedule-Free SOAP, lr=1e-3)",
        "optimizer",
        optim=(
            '[train.optim_g]\ntype = "SOAP_SF"\nlr = 1e-3\nweight_decay = 0\n'
            'betas = [ 0.95, 0.95,]\nschedule_free = true\nwarmup_steps = 200\n'
        ),
        scheduler="none"),

    # ── SCHEDULERS ────────────────────────────────────────────────────────────
    _mk("sched_multistep",
        "MultiStepLR milestones=[1250, 2000] gamma=0.5",
        "scheduler",
        scheduler='[train.scheduler]\ntype = "MultiStepLR"\nmilestones = [ 1250, 2000,]\ngamma = 0.5\n'),

    _mk("sched_cosine",
        "CosineAnnealing T_max=2500 (baseline scheduler)",
        "scheduler"),  # same as baseline, reference point

    _mk("sched_multistep_late",
        "MultiStepLR milestones=[2000] gamma=0.1 (une seule baisse)",
        "scheduler",
        scheduler='[train.scheduler]\ntype = "MultiStepLR"\nmilestones = [ 2000,]\ngamma = 0.1\n'),

    # ── GAN ───────────────────────────────────────────────────────────────────
    _mk("gan_bce_unet",
        "GAN bce + discriminateur unet",
        "gan",
        pixel_opt='[train.pixel_opt]\ntype = "L1Loss"\nloss_weight = 0.01\nreduction = "mean"\n',
        extra_losses=[
            '[train.gan_opt]\ntype = "gan_loss"\ngan_type = "bce"\nloss_weight = 0.05\n'
            'real_label_val = 1.0\nfake_label_val = 0.0\n'
        ],
        network_d=(
            '[network_d]\ntype = "unet"\nnum_in_ch = 3\nnum_feat = 64\nskip_connection = true\n'
        ),
        optim_d='[train.optim_d]\ntype = "Adam"\nlr = 2e-4\nweight_decay = 0\nbetas = [ 0.9, 0.99,]\n'),

    _mk("gan_bce_dunet",
        "GAN bce + discriminateur dunet",
        "gan",
        pixel_opt='[train.pixel_opt]\ntype = "L1Loss"\nloss_weight = 0.01\nreduction = "mean"\n',
        extra_losses=[
            '[train.gan_opt]\ntype = "gan_loss"\ngan_type = "bce"\nloss_weight = 0.05\n'
            'real_label_val = 1.0\nfake_label_val = 0.0\n'
        ],
        network_d='[network_d]\ntype = "dunet"\nin_ch = 3\ndim = 64\n',
        optim_d='[train.optim_d]\ntype = "Adam"\nlr = 2e-4\nweight_decay = 0\nbetas = [ 0.9, 0.99,]\n'),

    _mk("gan_bce_patchgan",
        "GAN bce + discriminateur patchgan",
        "gan",
        pixel_opt='[train.pixel_opt]\ntype = "L1Loss"\nloss_weight = 0.01\nreduction = "mean"\n',
        extra_losses=[
            '[train.gan_opt]\ntype = "gan_loss"\ngan_type = "bce"\nloss_weight = 0.05\n'
            'real_label_val = 1.0\nfake_label_val = 0.0\n'
        ],
        network_d='[network_d]\ntype = "patchgan"\nnum_in_ch = 3\nnum_feat = 64\n',
        optim_d='[train.optim_d]\ntype = "Adam"\nlr = 2e-4\nweight_decay = 0\nbetas = [ 0.9, 0.99,]\n'),

    _mk("gan_bce_metagan",
        "GAN bce + discriminateur metagan",
        "gan",
        pixel_opt='[train.pixel_opt]\ntype = "L1Loss"\nloss_weight = 0.01\nreduction = "mean"\n',
        extra_losses=[
            '[train.gan_opt]\ntype = "gan_loss"\ngan_type = "bce"\nloss_weight = 0.05\n'
            'real_label_val = 1.0\nfake_label_val = 0.0\n'
        ],
        network_d='[network_d]\ntype = "metagan"\nin_ch = 3\n',
        optim_d='[train.optim_d]\ntype = "Adam"\nlr = 2e-4\nweight_decay = 0\nbetas = [ 0.9, 0.99,]\n'),

    _mk("gan_bce_ea2fpn",
        "GAN bce + discriminateur ea2fpn",
        "gan",
        pixel_opt='[train.pixel_opt]\ntype = "L1Loss"\nloss_weight = 0.01\nreduction = "mean"\n',
        extra_losses=[
            '[train.gan_opt]\ntype = "gan_loss"\ngan_type = "bce"\nloss_weight = 0.05\n'
            'real_label_val = 1.0\nfake_label_val = 0.0\n'
        ],
        network_d='[network_d]\ntype = "ea2fpn"\nnum_in_ch = 3\nnum_feat = 64\n',
        optim_d='[train.optim_d]\ntype = "Adam"\nlr = 2e-4\nweight_decay = 0\nbetas = [ 0.9, 0.99,]\n'),

    # ── AUGMENTATIONS ─────────────────────────────────────────────────────────
    _mk("aug_none",
        "Pas d'augmentation (hflip+rot seulement)",
        "augmentation",
        augmentation='augmentation = [ "none",]\naug_prob = [ 1.0,]\n'),

    _mk("aug_mixup",
        "Mixup uniquement",
        "augmentation",
        augmentation='augmentation = [ "none", "mixup",]\naug_prob = [ 0.5, 0.5,]\n'),

    _mk("aug_cutmix",
        "CutMix uniquement",
        "augmentation",
        augmentation='augmentation = [ "none", "cutmix",]\naug_prob = [ 0.5, 0.5,]\n'),

    _mk("aug_resizemix",
        "ResizeMix uniquement",
        "augmentation",
        augmentation='augmentation = [ "none", "resizemix",]\naug_prob = [ 0.5, 0.5,]\n'),

    _mk("aug_cutblur",
        "CutBlur uniquement",
        "augmentation",
        augmentation='augmentation = [ "none", "cutblur",]\naug_prob = [ 0.4, 0.6,]\n'),

    _mk("aug_all",
        "Toutes augmentations (none+mixup+cutmix+resizemix+cutblur)",
        "augmentation",
        augmentation='augmentation = [ "none", "mixup", "cutmix", "resizemix", "cutblur",]\n'
                     'aug_prob = [ 0.4, 0.15, 0.15, 0.15, 0.35,]\n'),

    # ── PRECISION (skip on Pascal GTX 10xx) ───────────────────────────────────
    _mk("prec_fp16",
        "AMP FP16 (requires RTX 20xx+)",
        "precision",
        requires_amp=True,
        use_amp=True, use_bf16=False),

    _mk("prec_bf16",
        "AMP BF16 (requires RTX 30xx+)",
        "precision",
        requires_amp=True,
        use_amp=True, use_bf16=True),

    _mk("prec_tf32",
        "TF32 fast_matmul (requires RTX 30xx+)",
        "precision",
        requires_amp=True,
        fast_matmul=True),

    # ── SYSTEM SETTINGS ───────────────────────────────────────────────────────
    _mk("sys_warmup_500",
        "warmup_iter=500 (warm-up LR)",
        "system",
        warmup_iter=500),

    _mk("sys_ema_off",
        "EMA désactivé (ema=0)",
        "system",
        ema=0.0),

    _mk("sys_ema_09",
        "EMA conservateur (ema=0.9)",
        "system",
        ema=0.9),

    _mk("sys_grad_clip",
        "Gradient clipping activé",
        "system",
        grad_clip=True),

    _mk("sys_batch2_accum2",
        "batch=2/accumulate=2 (effective batch=4, moins de VRAM)",
        "system",
        batch_size=2, accumulate=2),

    _mk("sys_batch8",
        "batch=8/accumulate=1 (plus stable, plus de VRAM)",
        "system",
        batch_size=8, accumulate=1),

    _mk("sys_patch64",
        "patch_size=64 (moins de VRAM, contexte réduit)",
        "system",
        patch_size=64),

    _mk("sys_patch128",
        "patch_size=128 (plus de VRAM, meilleur contexte)",
        "system",
        patch_size=128,
        batch_size=2),

    _mk("sys_compile",
        "torch.compile=true (requires RTX + PyTorch 2.x)",
        "system",
        requires_compile=True,
        compile_=True),
]

# Index by name for --tests filter
_TEST_BY_NAME = {t["name"]: t for t in FEATURE_TESTS}

# ── Metric parsers ─────────────────────────────────────────────────────────────
_RE_ITPS = re.compile(r"([\d.]+)\s*it/s", re.IGNORECASE)
_RE_VRAM = re.compile(r"(?:GPU\s+mem|VRAM)[:\s]+([\d.]+)\s*GB", re.IGNORECASE)
_RE_PSNR = re.compile(r"psnr[:\s]+([\d.]+)", re.IGNORECASE)
_RE_SSIM = re.compile(r"ssim[:\s]+([\d.]+)", re.IGNORECASE)
_RE_ITER = re.compile(r"iter[:\s]+(\d+)", re.IGNORECASE)

_LIVE_KEYWORDS = ('it/s', 'gpu mem', 'psnr', 'ssim', 'error', 'erreur',
                  'traceback', 'eta:', 'warning', 'saving', 'checkpoint', 'validation')

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
            r = subprocess.run([candidate, "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                _nvidia_smi_exe = candidate
                return candidate
        except Exception:
            continue
    _nvidia_smi_exe = ""
    return None


def _smi_vram_mb() -> float | None:
    exe = _find_nvidia_smi()
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def _detect_gpu_cc() -> tuple[int, int]:
    """Retourne la compute capability du GPU (major, minor). (0,0) si inconnu."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:
        pass
    return (0, 0)


def _is_pascal() -> bool:
    cc = _detect_gpu_cc()
    return cc[0] == 6  # sm_61 = Pascal (GTX 10xx)


# ── TOML generation ────────────────────────────────────────────────────────────
def _generate_feature_toml(test: dict, n_iter: int,
                            train_gt: str, val_gt: str, val_lq: str) -> str:
    val_freq = n_iter * 1000  # désactivé
    print_freq = max(50, n_iter // 50)
    aug_lines = (test["augmentation"] + "\n") if test["augmentation"] else ""

    parts = [_TOML_FIXED.format(
        test_name=test["name"],
        use_amp="true" if test["use_amp"] else "false",
        use_bf16="true" if test["use_bf16"] else "false",
        fast_matmul="true" if test["fast_matmul"] else "false",
        compile="true" if test["compile_"] else "false",
        network_g=_COMPACT_NETWORK_G,
        n_iter=n_iter,
        print_freq=print_freq,
        val_freq=val_freq,
        train_gt=train_gt.replace("\\", "/"),
        val_gt=val_gt.replace("\\", "/"),
        val_lq=val_lq.replace("\\", "/"),
        batch_size=test["batch_size"],
        accumulate=test["accumulate"],
        patch_size=test["patch_size"],
        augmentation_lines=aug_lines,
    )]

    # [train] section (scalar params)
    parts.append(
        f"[train]\n"
        f"total_iter = {n_iter}\n"
        f"n_iter = {n_iter}\n"
        f"warmup_iter = {test['warmup_iter']}\n"
        f"ema = {test['ema']}\n"
        f"grad_clip = {'true' if test['grad_clip'] else 'false'}\n"
        f"match_lq_colors = false\n\n"
    )

    # [train.optim_g]
    optim_block = test["optim"] or _DEFAULT_OPTIM
    parts.append(optim_block + "\n")

    # [train.scheduler] — omit for schedule-free optimizers
    sched = test["scheduler"]
    if sched is None:
        # default cosine
        parts.append(_DEFAULT_SCHEDULER.format(n_iter=n_iter) + "\n")
    elif sched == "none":
        # schedule-free: use dummy MultiStepLR at milestone >> n_iter
        parts.append(
            f"[train.scheduler]\ntype = \"MultiStepLR\"\nmilestones = [ {n_iter * 999},]\ngamma = 1.0\n\n"
        )
    else:
        parts.append(sched + "\n")

    # [train.pixel_opt]
    pixel_block = test["pixel_opt"] or _DEFAULT_PIXEL_OPT
    parts.append(pixel_block + "\n")

    # Extra loss blocks (ldl, perceptual, fdl, dists, gan_opt, etc.)
    for extra in test["extra_losses"]:
        parts.append(extra + "\n")

    # [network_d] (GAN discriminator)
    if test["network_d"]:
        parts.append(test["network_d"] + "\n")

    # [train.optim_d] (GAN discriminator optimizer)
    if test["optim_d"]:
        parts.append(test["optim_d"] + "\n")

    return "\n".join(parts)


# ── Upscale test ───────────────────────────────────────────────────────────────
def _run_upscale_test(test_name: str, model_path: Path,
                      test_image: Path, output_dir: Path) -> dict:
    out_img = output_dir / "upscale_tests" / f"FBench_{test_name}_{model_path.stem}.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)

    if not test_image.exists():
        return {"status": "skip", "reason": f"image introuvable: {test_image}"}
    if not NEOSR_PYTHON.exists():
        return {"status": "skip", "reason": "neosr venv introuvable"}
    if not NEOSR_GENERAL_RUNNER.exists():
        return {"status": "skip", "reason": f"runner introuvable: {NEOSR_GENERAL_RUNNER}"}

    print(f"    [upscale] {test_name} → {out_img.name}", flush=True)
    try:
        r = subprocess.run(
            [str(NEOSR_PYTHON), str(NEOSR_GENERAL_RUNNER),
             str(model_path), str(test_image), str(out_img)],
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
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── Per-test benchmark ─────────────────────────────────────────────────────────
def run_feature_test(test: dict, n_iter: int, timeout_sec: int,
                     train_gt: str, val_gt: str, val_lq: str,
                     keep_toml: bool, output_dir: Path,
                     test_image: Path, do_upscale: bool) -> dict:
    name = test["name"]
    result = {
        "name": name, "desc": test["desc"], "category": test["category"],
        "status": "error", "error": None,
        "avg_itps": None, "peak_vram_gb": None, "peak_vram_smi_gb": None,
        "psnr_readings": {}, "ssim_readings": {},
        "elapsed_sec": 0, "iters_completed": 0, "upscale_test": None,
    }

    toml_path = output_dir / f"_fbench_{name}.toml"
    try:
        toml_text = _generate_feature_toml(test, n_iter, train_gt, val_gt, val_lq)
        toml_path.write_text(toml_text, encoding="utf-8")
    except Exception as e:
        result["error"] = f"TOML generation failed: {e}"
        return result

    itps_readings: list[float] = []
    neosr_vram: list[float] = []
    smi_vram: list[float] = []
    psnr_map: dict[int, float] = {}
    ssim_map: dict[int, float] = {}
    current_iter = [0]
    stdout_lines: list[str] = []

    stop_vram = threading.Event()

    def vram_monitor():
        while not stop_vram.is_set():
            v = _smi_vram_mb()
            if v is not None:
                smi_vram.append(v / 1024.0)
            stop_vram.wait(4)

    vram_thread = threading.Thread(target=vram_monitor, daemon=True)
    vram_thread.start()

    t_start = time.time()
    proc = None
    try:
        proc = subprocess.Popen(
            [str(NEOSR_PYTHON), str(TRAIN_SCRIPT), "-opt", str(toml_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
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
                    neosr_vram.append(float(m.group(1)))
                m = _RE_ITER.search(stripped)
                if m:
                    current_iter[0] = int(m.group(1))
                lower = stripped.lower()
                if "psnr" in lower and ("val" in lower or "metric" in lower or "#" in lower):
                    mp = _RE_PSNR.search(stripped)
                    ms = _RE_SSIM.search(stripped)
                    it = current_iter[0]
                    if mp:
                        psnr_map[it] = float(mp.group(1))
                    if ms:
                        ssim_map[it] = float(ms.group(1))
                    print(f"  [{name}] {stripped}", flush=True)
                elif any(k in lower for k in _LIVE_KEYWORDS):
                    print(f"  [{name}] {stripped}", flush=True)

        reader = threading.Thread(target=stdout_reader, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            print(f"  [{name}] timeout ({timeout_sec}s) — killing", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        reader.join(timeout=10)
        exit_code = proc.returncode

    except Exception as e:
        result["error"] = f"Subprocess launch failed: {e}"
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
        stop_vram.set()
        if not keep_toml and toml_path.exists():
            toml_path.unlink(missing_ok=True)
        return result
    finally:
        stop_vram.set()
        if not keep_toml and toml_path.exists():
            toml_path.unlink(missing_ok=True)

    elapsed = time.time() - t_start
    result["elapsed_sec"] = round(elapsed, 1)
    result["iters_completed"] = current_iter[0]

    if itps_readings:
        drop = max(1, len(itps_readings) // 10)
        stable = itps_readings[drop:] if len(itps_readings) > drop + 2 else itps_readings
        result["avg_itps"] = round(sum(stable) / len(stable), 4)

    if neosr_vram:
        result["peak_vram_gb"] = round(max(neosr_vram), 2)
    elif smi_vram:
        result["peak_vram_gb"] = round(max(smi_vram), 2)
    if smi_vram:
        result["peak_vram_smi_gb"] = round(max(smi_vram), 2)

    result["psnr_readings"] = {str(k): round(v, 4) for k, v in sorted(psnr_map.items())}
    result["ssim_readings"] = {str(k): round(v, 4) for k, v in sorted(ssim_map.items())}

    if exit_code == 0 or current_iter[0] >= n_iter - 1:
        result["status"] = "ok"
        if exit_code == 0 and current_iter[0] < n_iter // 2:
            result["iters_completed"] = n_iter
    elif not itps_readings:
        result["status"] = "error"
        result["error"] = f"exit_code={exit_code}\n" + "\n".join(stdout_lines[-25:])
    else:
        result["status"] = "timeout"

    # Upscale test
    if do_upscale and result["status"] in ("ok", "timeout"):
        models_dir = NEOSR_PATH / "experiments" / f"FBench_{name}" / "models"
        model_path = models_dir / f"net_g_{n_iter}.pth"
        if not model_path.exists():
            candidates = sorted(models_dir.glob("net_g_*.pth")) if models_dir.exists() else []
            model_path = candidates[-1] if candidates else None
        if model_path and model_path.exists():
            result["upscale_test"] = _run_upscale_test(
                name, model_path, test_image, output_dir)
        else:
            result["upscale_test"] = {"status": "skip",
                                      "reason": f"checkpoint introuvable dans {models_dir}"}

    return result


# ── Report ─────────────────────────────────────────────────────────────────────
def _format_report(results: list[dict], gpu_name: str, ts: str, n_iter: int) -> str:
    lines = [
        "=" * 100,
        f"  Feature Coverage Benchmark  —  {ts}",
        f"  GPU: {gpu_name}  |  Base arch: compact  |  N_iter: {n_iter}",
        "=" * 100,
        "",
        f"{'Test':<26} {'Cat':<14} {'Status':<9} {'it/s':>7} {'VRAM GB':>8} {'SMI GB':>7}"
        f" {'PSNR@end':>9} {'Iters':>6}",
        "-" * 92,
    ]
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    for cat, rlist in by_cat.items():
        lines.append(f"\n  [{cat.upper()}]")
        for r in rlist:
            itps = f"{r['avg_itps']:.3f}" if r["avg_itps"] else "  —  "
            vram = f"{r['peak_vram_gb']:.2f}" if r["peak_vram_gb"] else "  —  "
            smi  = f"{r['peak_vram_smi_gb']:.2f}" if r.get("peak_vram_smi_gb") else "  —  "
            psnr_vals = list(r["psnr_readings"].values())
            psnr = f"{psnr_vals[-1]:.2f}" if psnr_vals else "  —  "
            iters = str(r["iters_completed"])
            ut = r.get("upscale_test") or {}
            up_status = ut.get("status", "—")
            lines.append(
                f"  {r['name']:<24} {r['category']:<14} {r['status']:<9} {itps:>7}"
                f" {vram:>8} {smi:>7} {psnr:>9} {iters:>6}   upscale:{up_status}"
            )

    lines.append("\n" + "=" * 100)
    lines.append("\nDescriptions :")
    lines.append("-" * 70)
    for r in results:
        lines.append(f"  {r['name']:<26} {r['desc']}")

    # Errors
    errors = [r for r in results if r["status"] == "error" and r.get("error")]
    if errors:
        lines.append("\n\nERREURS :")
        lines.append("-" * 70)
        for r in errors:
            lines.append(f"\n[ERROR — {r['name']}]")
            lines.append(str(r.get("error", ""))[:800])

    lines.append("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Feature Coverage Benchmark NeoSR")
    ap.add_argument("--n-iter",     type=int,   default=2500)
    ap.add_argument("--timeout",    type=int,   default=1800)
    ap.add_argument("--category",   type=str,   default=None,
                    help="Filtre par catégorie (losses/optimizer/scheduler/gan/augmentation/precision/system)")
    ap.add_argument("--tests",      type=str,   default=None,
                    help="Tests spécifiques (virgule séparés)")
    ap.add_argument("--no-upscale", action="store_true")
    ap.add_argument("--output-dir", type=str,   default=None)
    ap.add_argument("--train-gt",   type=str,   default=DEFAULT_TRAIN_GT)
    ap.add_argument("--val-gt",     type=str,   default=DEFAULT_VAL_GT)
    ap.add_argument("--val-lq",     type=str,   default=DEFAULT_VAL_LQ)
    ap.add_argument("--test-image", type=str,   default=DEFAULT_TEST_IMG)
    ap.add_argument("--keep-toml",  action="store_true")
    ap.add_argument("--list",       action="store_true",
                    help="Lister tous les tests disponibles et quitter")
    args = ap.parse_args()

    if args.list:
        print(f"\n{'Test':<26} {'Catégorie':<14} {'AMP req':>7}  Description")
        print("-" * 90)
        cats: dict[str, list] = {}
        for t in FEATURE_TESTS:
            cats.setdefault(t["category"], []).append(t)
        for cat, tlist in cats.items():
            print(f"\n  [{cat.upper()}]")
            for t in tlist:
                amp = "oui" if t["requires_amp"] else ""
                print(f"  {t['name']:<24} {cat:<14} {amp:>7}  {t['desc']}")
        print(f"\nTotal : {len(FEATURE_TESTS)} tests")
        return

    # GPU detection
    cc = _detect_gpu_cc()
    is_pascal = (cc[0] == 6)
    gpu_name = "Inconnu"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    print(f"GPU : {gpu_name}  (sm_{cc[0]}{cc[1]})")
    if is_pascal:
        print("Pascal détecté → tests AMP/BF16/compile marqués skip\n")

    # Checks
    if not NEOSR_PYTHON.exists():
        print(f"[ERREUR] neosr venv introuvable : {NEOSR_PYTHON}", file=sys.stderr)
        sys.exit(1)
    if not TRAIN_SCRIPT.exists():
        print(f"[ERREUR] train.py introuvable : {TRAIN_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path.home() / "IA_Engine" / "benchmark_results" / "feature_bench")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter tests
    tests_to_run = FEATURE_TESTS
    if args.tests:
        names = {n.strip() for n in args.tests.split(",")}
        tests_to_run = [t for t in tests_to_run if t["name"] in names]
        if not tests_to_run:
            print(f"[ERREUR] Aucun test trouvé parmi : {args.tests}", file=sys.stderr)
            sys.exit(1)
    if args.category:
        tests_to_run = [t for t in tests_to_run if t["category"] == args.category]
        if not tests_to_run:
            print(f"[ERREUR] Aucun test dans la catégorie '{args.category}'", file=sys.stderr)
            sys.exit(1)

    test_image = Path(args.test_image)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"feature_bench_{ts}.txt"
    json_file   = output_dir / f"feature_bench_{ts}.json"

    n_skip = sum(1 for t in tests_to_run
                 if (t["requires_amp"] or t["requires_compile"]) and is_pascal)
    n_run  = len(tests_to_run) - n_skip
    eta_min = n_run * args.n_iter / 11.86 / 60
    print(f"Tests sélectionnés : {len(tests_to_run)}"
          f" ({n_skip} skippés Pascal) → {n_run} à lancer")
    print(f"ETA estimée : ~{eta_min:.0f} min (base compact 11.86 it/s)")
    print(f"Résultats   : {result_file}\n")

    results: list[dict] = []
    for i, test in enumerate(tests_to_run, 1):
        name = test["name"]
        skip_reason = None
        if (test["requires_amp"] or test["requires_compile"]) and is_pascal:
            skip_reason = f"Pascal ne supporte pas {'AMP' if test['requires_amp'] else 'compile'}"

        if skip_reason:
            print(f"\n[{i}/{len(tests_to_run)}] SKIP {name} — {skip_reason}")
            results.append({
                "name": name, "desc": test["desc"], "category": test["category"],
                "status": "skip", "error": skip_reason,
                "avg_itps": None, "peak_vram_gb": None, "peak_vram_smi_gb": None,
                "psnr_readings": {}, "ssim_readings": {},
                "elapsed_sec": 0, "iters_completed": 0, "upscale_test": None,
            })
            continue

        print(f"\n[{i}/{len(tests_to_run)}] {name}", flush=True)
        print(f"  {test['desc']}", flush=True)
        print(f"  cat={test['category']} | batch={test['batch_size']}"
              f"/accum={test['accumulate']} | patch={test['patch_size']}"
              f" | ema={test['ema']} | warmup={test['warmup_iter']}", flush=True)

        r = run_feature_test(
            test=test, n_iter=args.n_iter, timeout_sec=args.timeout,
            train_gt=args.train_gt, val_gt=args.val_gt, val_lq=args.val_lq,
            keep_toml=args.keep_toml,
            output_dir=output_dir, test_image=test_image,
            do_upscale=not args.no_upscale,
        )
        results.append(r)

        icon = "✅" if r["status"] == "ok" else ("⏱" if r["status"] == "timeout" else "❌")
        itps = f"{r['avg_itps']:.3f} it/s" if r["avg_itps"] else "n/a"
        vram = f"{r['peak_vram_smi_gb']:.2f} GB SMI" if r.get("peak_vram_smi_gb") else "n/a"
        psnr_vals = list(r["psnr_readings"].values())
        psnr = f"PSNR={psnr_vals[-1]:.2f}" if psnr_vals else ""
        print(f"  {icon} {r['status']:<7} {itps}  {vram}  {psnr}", flush=True)

        # Partial save after each test
        json_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        report = _format_report(results, gpu_name, ts, args.n_iter)
        result_file.write_text(report, encoding="utf-8")

    # Final report
    print("\n" + "=" * 70)
    print(f"RÉSUMÉ — {len(results)} tests")
    print("=" * 70)
    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")
    skip_count = sum(1 for r in results if r["status"] == "skip")
    print(f"  ✅ OK     : {ok_count}")
    print(f"  ❌ Erreur : {err_count}")
    print(f"  ⏭ Skip   : {skip_count}")
    print(f"\nRésultats sauvegardés dans : {result_file}")
    print(f"JSON                       : {json_file}")


if __name__ == "__main__":
    main()
