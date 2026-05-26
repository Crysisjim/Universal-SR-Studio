"""
redux_feature_benchmark.py — traiNNer-Redux Feature Coverage Benchmark
Laptop : RTX 3070 Ti (8 GB VRAM) — AMP BF16 — compact comme arch de base

Teste ~34 features (losses, optimizers, schedulers, precision) une par une.
Objectif : vérifier compatibilité + mesurer impact sur it/s.

Resume : state JSON sauvegardé après chaque test.

Usage:
    python redux_feature_benchmark.py [options]

    --n-iter 500           Itérations par test (défaut : 500)
    --timeout 900          Timeout par test en secondes (défaut : 900)
    --tests loss_l1,...    Tester seulement ces features
    --output-dir PATH      Dossier résultats (défaut : ~/IA_Engine/benchmark_results/redux_feat)
    --train-gt PATH        Dataset training GT
    --reset                Remettre à zéro (ignorer état existant)
    --list                 Lister tous les tests disponibles
"""
import sys
import os
import re
import time
import json
import shutil
import threading
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
REDUX_PATH   = Path.home() / "IA_Engine" / "traiNNer-redux"
TRAIN_SCRIPT = REDUX_PATH / "train.py"

# ── Chemin racine du Studio (pour importer quick_upscale dans le subprocess) ──
_STUDIO_ROOT = Path(__file__).resolve().parent.parent.parent  # Universal SR Studio DEV/

# ── Modèle perso de référence pour les tests ECO + upscale ────────────────────
_PERSONAL_MODEL = (
    Path.home() / "IA_Engine" / "Final model" /
    "Crysisjim SPANPlus Deband_HARD 1.0" /
    "Crysisjim SPANPlus Deband_HARD 1.0.safetensors"
)
_DEFAULT_TEST_IMG = (
    Path.home() / "IA_Engine" / "datasets" / "val" / "LQ" / "Overlord (1).png"
)


def _find_venv_python(base: Path) -> Path:
    """Cherche python.exe dans les emplacements courants du venv (Windows + Linux)."""
    candidates = [
        base / ".venv" / "Scripts" / "python.exe",   # Windows .venv (géré par Universal SR Studio)
        base / "venv"  / "Scripts" / "python.exe",   # Windows venv
        base / ".venv" / "bin"     / "python",        # Linux .venv
        base / "venv"  / "bin"     / "python",        # Linux venv
        base / ".venv" / "bin"     / "python3",
        base / "venv"  / "bin"     / "python3",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(sys.executable)


REDUX_PYTHON = _find_venv_python(REDUX_PATH)

DEFAULT_TRAIN_GT = str(Path.home() / "IA_Engine/datasets/train/HR")
DEFAULT_VAL_GT   = str(Path.home() / "IA_Engine/datasets/val/GT")
DEFAULT_VAL_LQ   = str(Path.home() / "IA_Engine/datasets/val/LQ")

# ── YAML fragments ─────────────────────────────────────────────────────────────
_OPTIM_ADAMW = """\
    type: AdamW
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]"""

_OPTIM_ADAM = """\
    type: Adam
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]"""

_OPTIM_ADAN = """\
    type: Adan
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.98, 0.92, 0.99]"""

_OPTIM_ADAMWSF = """\
    type: AdamWScheduleFree
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]
    warmup_steps: 100"""

_OPTIM_ADANSF = """\
    type: AdanScheduleFree
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.98, 0.92, 0.99]
    warmup_steps: 100"""

_SCHED_MULTISTEP = """\
    type: MultiStepLR
    milestones: [999999]
    gamma: 0.5"""

# ── Loss blocks (4-space indented for YAML list) ───────────────────────────────
def _l(loss_type: str, weight: float = 1.0, extra: str = "") -> str:
    return f"    - type: {loss_type}\n      loss_weight: {weight}\n{extra}"

_LOSSES: dict[str, str] = {
    # ── Sprint 19/20 — SparK Perceptual Loss (InceptionNext features) ──────────
    "spark_fd":         _l("SparkLoss", extra="      criterion: fd\n"),
    "spark_charbonnier":_l("SparkLoss", extra="      criterion: charbonnier\n"),
    # pixel
    "charbonnier": _l("charbonnierloss"),
    "l1":          _l("l1loss"),
    "mse":         _l("mseloss"),
    "fft":         _l("fftloss"),
    "luma":        _l("lumaloss", extra="      criterion: l1\n"),
    "color":       _l("colorloss", extra="      criterion: l1\n      scale: 4\n"),
    "hsluv":       _l("hsluvloss", extra="      criterion: l1\n"),
    "average":     _l("averageloss", extra="      criterion: l1\n      scale: 4\n"),
    "bicubic":     _l("bicubicloss", extra="      criterion: l1\n      scale: 4\n"),
    "psnr":        _l("psnrloss", extra="      reduction: mean\n      to_y: false\n"),
    # ssim
    "ssim":        _l("ssimloss"),
    "mssim":       _l("mssimloss"),
    "msssiml1":    _l("msssiml1loss"),
    # perceptual
    "perc_conv":   (
        "    - type: perceptualloss\n      loss_weight: 1.0\n"
        "      layer_weights:\n        conv1_2: 0.1\n        conv2_2: 0.1\n"
        "        conv3_4: 1.0\n        conv4_4: 1.0\n        conv5_4: 1.0\n"
        "      criterion: l1\n"
    ),
    "perc_relu": (
        "    - type: perceptualloss\n      loss_weight: 1.0\n"
        "      layer_weights:\n        relu1_2: 0.1\n        relu2_2: 0.1\n"
        "        relu3_4: 1.0\n        relu4_4: 1.0\n        relu5_4: 1.0\n"
        "      criterion: pd\n"
    ),
    # quality
    "ncc":   _l("nccloss"),
    "cosim": _l("cosimloss"),
    "flip":  _l("fliploss"),
    "gv":    _l("gradientvarianceloss", extra="      patch_size: 16\n      criterion: charbonnier\n"),
    "ff":    _l("ffloss", extra="      alpha: 1.0\n      ave_spectrum: true\n"),
    # advanced (need pretrain weights)
    "dists":  _l("distsloss"),
    "adists": _l("adistsloss"),
    "ctx": (
        "    - type: contextualloss\n      loss_weight: 0.05\n"
        "      layer_weights:\n        conv3_2: 1.0\n        conv4_2: 1.0\n"
        "      max_1d_size: 64\n      distance_type: cosine\n      band_width: 0.5\n"
    ),
}

# ── Feature test registry ──────────────────────────────────────────────────────
def _f(name, desc, category, loss_key=None, optim_block=None, sched_fn=None,
       use_amp=True, amp_bf16=True, channels_last=True,
       fast_matmul=False, use_compile=False,
       extra_train_fields="", extra_path_fields="",
       dataset_mode="paired", lq_size_override=None,
       eco_pretrain_auto=False, eco_pretrain_path="",
       keep_exp_dir=False, upscale_model_path=""):
    """
    extra_train_fields  : lignes YAML supplémentaires sous train: (ex: '  eco: true\n  eco_alpha: 0.05\n')
    extra_path_fields   : lignes YAML supplémentaires sous path: (ex: '  eco_pretrain_g: /path\n')
    dataset_mode        : 'paired' (défaut) | 'bicubic' (pairedimagedataset, LQ=GT)
    lq_size_override    : remplace base_lq_size si défini (utile pour losses qui nécessitent lq≥128)
    eco_pretrain_auto   : si True, cherche auto le checkpoint RFeat_dataset_bicubic pour eco_pretrain_g
    eco_pretrain_path   : chemin direct vers un .safetensors/.pth à utiliser comme eco_pretrain_g
    keep_exp_dir        : si True, ne supprime PAS exp_dir après le run (utile quand un test downstream en a besoin)
    upscale_model_path  : si non vide, lance un upscale final avec ce modèle sur l'image de test
    """
    return dict(
        name=name, desc=desc, category=category,
        loss_key=loss_key or "charbonnier",
        optim_block=optim_block or _OPTIM_ADAMW,
        sched_fn=sched_fn,        # callable(n_iter)->str, or None for default
        use_amp=use_amp, amp_bf16=amp_bf16,
        channels_last=channels_last,
        fast_matmul=fast_matmul, use_compile=use_compile,
        extra_train_fields=extra_train_fields,
        extra_path_fields=extra_path_fields,
        dataset_mode=dataset_mode,
        lq_size_override=lq_size_override,
        eco_pretrain_auto=eco_pretrain_auto,
        eco_pretrain_path=eco_pretrain_path,
        keep_exp_dir=keep_exp_dir,
        upscale_model_path=upscale_model_path,
    )


def _sched_cosine(n_iter: int) -> str:
    half = max(n_iter // 2, 1)
    return (
        f"    type: CosineAnnealingRestartLR\n"
        f"    periods: [{half}, {half}]\n"
        f"    restart_weights: [1, 0.5]\n"
        f"    eta_min: !!float 1e-7"
    )


def _sched_knee(n_iter: int) -> str:
    return (
        f"    type: KneeLR\n"
        f"    peak_lr: !!float 2e-4\n"
        f"    total_steps: {n_iter}\n"
        f"    explore_ratio: 0.5"
    )


FEATURE_LIST = [
    # ── Losses pixel ─────────────────────────────────────────────────────────
    _f("loss_charbonnier", "CharbonnierLoss (baseline)",        "losses_pixel", "charbonnier"),
    _f("loss_l1",          "L1Loss",                             "losses_pixel", "l1"),
    _f("loss_mse",         "MSELoss (L2)",                       "losses_pixel", "mse"),
    _f("loss_fft",         "FFTLoss (frequency domain L1)",      "losses_pixel", "fft"),
    _f("loss_luma",        "LumaLoss L1 (Y channel)",            "losses_pixel", "luma"),
    _f("loss_color",       "ColorLoss L1 (UV chrominance)",      "losses_pixel", "color"),
    _f("loss_hsluv",       "HSLuvLoss L1 (HSLuv color space)",   "losses_pixel", "hsluv"),
    _f("loss_average",     "AverageLoss L1 (avg pool)",          "losses_pixel", "average"),
    _f("loss_bicubic",     "BicubicLoss L1 (bicubic downsample)","losses_pixel", "bicubic"),
    _f("loss_psnr",        "PSNRLoss (differentiable PSNR)",     "losses_pixel", "psnr"),
    # ── Losses SSIM ──────────────────────────────────────────────────────────
    _f("loss_ssim",     "SSIMLoss",                "losses_ssim", "ssim"),
    _f("loss_mssim",    "MSSIMLoss (multi-scale)", "losses_ssim", "mssim"),
    _f("loss_msssiml1", "MSSSIML1Loss (MS-SSIM+L1)","losses_ssim","msssiml1"),
    # ── Losses perceptual ────────────────────────────────────────────────────
    _f("loss_perc_conv", "PerceptualLoss conv layers l1 (VGG19)", "losses_perceptual", "perc_conv"),
    _f("loss_perc_relu", "PerceptualLoss relu layers pd (VGG19)", "losses_perceptual", "perc_relu"),
    # ── Losses quality ───────────────────────────────────────────────────────
    _f("loss_ncc",   "NCCLoss (Normalized Cross-Corr)",  "losses_quality", "ncc"),
    _f("loss_cosim", "CosimLoss (Cosine Similarity)",    "losses_quality", "cosim"),
    _f("loss_flip",  "FLIPLoss (NVIDIA FLIP perceptual)","losses_quality", "flip"),
    _f("loss_gv",    "GradientVarianceLoss (patch GV)",  "losses_quality", "gv"),
    _f("loss_ff",    "FFLoss (Focal Frequency)",          "losses_quality", "ff"),
    # ── Losses advanced (VGG/pretrain) ───────────────────────────────────────
    _f("loss_dists",  "DISTSLoss (structure+texture, VGG16)",  "losses_advanced", "dists"),
    _f("loss_adists", "ADISTSLoss (attention DISTS, VGG16)",   "losses_advanced", "adists"),
    _f("loss_ctx",    "ContextualLoss (unaligned, VGG19)",     "losses_advanced", "ctx"),
    # ── Optimizers ───────────────────────────────────────────────────────────
    _f("optim_adam",   "Adam optimizer",              "optimizer", optim_block=_OPTIM_ADAM),
    _f("optim_adan",   "Adan optimizer",              "optimizer", optim_block=_OPTIM_ADAN),
    _f("optim_adamwsf","AdamWScheduleFree (no sched)","optimizer", optim_block=_OPTIM_ADAMWSF),
    _f("optim_adansf", "AdanScheduleFree (no sched)", "optimizer", optim_block=_OPTIM_ADANSF),
    # ── Schedulers ───────────────────────────────────────────────────────────
    _f("sched_cosine", "CosineAnnealingRestartLR (2 cycles)", "scheduler", sched_fn=_sched_cosine),
    _f("sched_knee",   "KneeLR (warmup→linear decay)",         "scheduler", sched_fn=_sched_knee),
    # ── Precision variants ───────────────────────────────────────────────────
    _f("prec_bf16_nocl",  "BF16 AMP + channels_last=false",
       "precision", channels_last=False),
    _f("prec_fp16",       "FP16 AMP (amp_bf16=false)",
       "precision", amp_bf16=False),
    _f("prec_noamp",      "No AMP (fp32 training)",
       "precision", use_amp=False, amp_bf16=False),
    _f("prec_fastmatmul", "BF16 AMP + fast_matmul=true",
       "precision", fast_matmul=True),
    _f("prec_compile",    "BF16 AMP + torch.compile (default mode)",
       "precision", use_compile=True),

    # ── Sprint 19/20 — Nouvelles features ────────────────────────────────────

    # SparK Perceptual Loss (InceptionNext features, Redux uniquement)
    # spark_fd: InceptionNext downsample aggressif → besoin lq≥128 (sinon kernel 4>spatial 3)
    _f("loss_spark_fd",   "SparkLoss Fourier Domain (InceptionNext backbone)",
       "losses_advanced", "spark_fd", lq_size_override=128),
    _f("loss_spark_charb","SparkLoss Charbonnier (InceptionNext backbone)",
       "losses_advanced", "spark_charbonnier"),

    # Bicubic dataset — run AVANT eco_training pour générer un checkpoint compact réutilisable
    _f("dataset_bicubic", "Bicubic dataset mode (pairedimagedataset, LQ=GT)",
       "training_mode",
       dataset_mode="bicubic",
       keep_exp_dir=True),   # checkpoint gardé pour eco_pretrain_auto

    # ECO Training Mode — blending weights with pretrained reference
    # eco_pretrain_auto=True : cherche auto le checkpoint RFeat_dataset_bicubic généré ci-dessus
    # Si absent : ECO log warning, training continue sans blend (comportement de fallback)
    _f("eco_training",    "ECO Training Mode (α=0.05, weight blending — pretrain=RFeat_dataset_bicubic)",
       "training_mode",
       extra_train_fields="  eco: true\n  eco_alpha: 0.05\n",
       eco_pretrain_auto=True),

    # ── Tests perso : SPANPlus Deband_HARD comme modèle de référence ─────────
    # bicubic_personal : même que dataset_bicubic mais avec upscale final via modèle perso
    _f("bicubic_personal",
       "Bicubic dataset (pairedimagedataset) + upscale final SPANPlus Deband_HARD",
       "training_mode",
       dataset_mode="bicubic",
       keep_exp_dir=True,
       upscale_model_path=str(_PERSONAL_MODEL)),

    # eco_personal : ECO blending avec modèle perso comme pretrain + upscale final
    # Note : arch compact (SPAN) vs SPANPlus — si mismatch shape → status=error attendu
    _f("eco_personal",
       "ECO (α=0.05) — pretrain=SPANPlus Deband_HARD + upscale final",
       "training_mode",
       extra_train_fields="  eco: true\n  eco_alpha: 0.05\n",
       eco_pretrain_path=str(_PERSONAL_MODEL),
       upscale_model_path=str(_PERSONAL_MODEL)),
]

_FEAT_BY_NAME = {f["name"]: f for f in FEATURE_LIST}

# ── YAML template ──────────────────────────────────────────────────────────────
_YAML_TEMPLATE = """\
name: RFeat_{name}
scale: 1
use_amp: {use_amp}
amp_bf16: {amp_bf16}
use_channels_last: {channels_last}
fast_matmul: {fast_matmul}
use_compile: {use_compile}
num_gpu: auto
manual_seed: 42

datasets:
  train:
    name: RFeatTrain
    type: {dataset_type}
    dataroot_gt: ['{train_gt}']
    dataroot_lq: ['{train_gt}']
    lq_size: {lq_size}
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 2
    batch_size_per_gpu: {batch_size}
    accum_iter: 1
  val:
    name: RFeatVal
    type: pairedimagedataset
    dataroot_gt: ['{val_gt}']
    dataroot_lq: ['{val_lq}']

network_g:
  type: {base_arch}

path:
  param_key_g: ~
  strict_load_g: true
  resume_state: ~
{extra_path_fields}
train:
  ema_decay: 0.999
  ema_power: 0.75
  grad_clip: false
{extra_train_fields}  optim_g:
{optim_block}
  scheduler:
{sched_block}
  total_iter: {n_iter}
  warmup_iter: -1
  losses:
{losses_block}
val:
  val_enabled: true
  val_freq: {n_iter}
  save_img: false
  tile_size: 256
  tile_overlap: 32
  metrics_enabled: true
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false

logger:
  print_freq: {print_freq}
  save_checkpoint_freq: {n_iter}
  save_checkpoint_format: safetensors
  use_tb_logger: false
"""

# ── Output regexes ─────────────────────────────────────────────────────────────
_RE_ITPS = re.compile(r"\[performance:\s*([\d.]+)\s*it/s\]", re.IGNORECASE)
_RE_VRAM = re.compile(r"\[peak VRAM:\s*([\d.]+)\s*GB\]", re.IGNORECASE)
_RE_PSNR = re.compile(r"#\s+psnr\s*:\s*([\d.]+)", re.IGNORECASE)
_RE_ITER = re.compile(r"iter:\s*([\d,]+)", re.IGNORECASE)

# ── Console display ──────────────────────────────────────────────────────────
_RE_ITER_DISP = re.compile(
    r"\[epoch:\s*(\d+),\s*iter:\s*([\d,]+),\s*lr:\(([\d.e+\-]+)\)\]"
    r".*?\[performance:\s*([\d.]+)\s*it/s\]"
    r".*?\[eta:\s*([^\]]+)\]"
    r".*?\[peak VRAM:\s*([\d.]+)\s*GB\]"
    r"(.*)",
    re.IGNORECASE | re.DOTALL,
)
_RE_TS_DISP   = re.compile(r"\[(?:\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})\]")
_RE_LOSS_DISP = re.compile(r"\bl_(\w+):\s*([\d.e+\-]+)")
_RE_VAL_DISP  = re.compile(r"#\s+(psnr|ssim)\s*:\s*([\d.]+)", re.IGNORECASE)

_C0 = "\033[0m"    # reset
_CB = "\033[1m"    # bold
_CD = "\033[2m"    # dim
_CC = "\033[96m"   # cyan   — arch prefix
_CG = "\033[92m"   # green  — it/s, DONE ok
_CY = "\033[93m"   # yellow — iter, epoch
_CL = "\033[94m"   # blue   — VRAM
_CR = "\033[91m"   # red    — errors
_CW = "\033[1;97m" # bold white — header


def _display_training_line(raw: str, arch: str) -> None:
    """Reformate iter/validation ; supprime le bruit de démarrage."""
    vm = _RE_VAL_DISP.search(raw)
    if vm:
        ts_m = _RE_TS_DISP.search(raw)
        ts   = ts_m.group(1) if ts_m else ""
        print(f"{_CC}[{arch}]{_C0} {_CD}{ts}{_C0} | "
              f"[ {_CG}{vm.group(1).upper()}: {float(vm.group(2)):.4f}{_C0} ]", flush=True)
        return
    m = _RE_ITER_DISP.search(raw)
    if not m:
        raw_s = raw.strip()
        if "Start training from epoch" in raw_s:
            ts_m = _RE_TS_DISP.search(raw)
            ts   = ts_m.group(1) if ts_m else ""
            print(f"{_CC}[{arch}]{_C0} {_CD}{ts}{_C0} | {_CD}Démarrage entraînement...{_C0}",
                  flush=True)
        elif raw_s and ("error" in raw_s.lower() or "exception" in raw_s.lower()) \
                and "logging error" not in raw_s.lower():
            print(f"{_CC}[{arch}]{_C0} {_CR}[ERR]{_C0} {raw_s[:250]}", flush=True)
        return
    epoch, it, lr, itps, eta, vram, rest = m.groups()
    it_int = int(it.replace(',', ''))
    ts_m   = _RE_TS_DISP.search(raw)
    ts     = ts_m.group(1) if ts_m else ""
    losses = _RE_LOSS_DISP.findall(rest)
    loss_str = " ".join(f"[ {_CD}l_{n}: {v}{_C0} ]" for n, v in losses)
    out = (
        f"{_CC}[{arch}]{_C0} {_CD}{ts}{_C0} | "
        f"[ epoch: {_CY}{int(epoch):2d}{_C0} ] "
        f"[ iter: {_CY}{it_int:5d}{_C0} ] "
        f"[ {_CG}{float(itps):.3f} it/s{_C0} ] "
        f"[ {_CD}lr: {lr}{_C0} ] "
        f"[ {_CD}eta: {eta.strip()}{_C0} ] "
        f"[ {_CL}VRAM: {vram} GB{_C0} ]"
    )
    if loss_str:
        out += " " + loss_str
    print(out, flush=True)


def _find_eco_pretrain(base_arch: str = "compact") -> str:
    """
    Cherche le dernier checkpoint safetensors/pth généré par RFeat_dataset_bicubic.
    Utilisé par eco_pretrain_auto pour tester le vrai blending ECO avec un pretrain réel.
    Retourne le chemin absolu (str) ou "" si non trouvé.
    """
    exp_dir = REDUX_PATH / "experiments" / "RFeat_dataset_bicubic" / "models"
    if not exp_dir.exists():
        return ""
    # Cherche dans models/ et models/resume_models/
    # Préférer net_g_ema (plus smooth), puis net_g; safetensors avant pth; plus récent en premier
    candidates = sorted(
        list(exp_dir.glob("net_g_*.safetensors"))
        + list(exp_dir.glob("net_g_*.pth"))
        + list((exp_dir / "resume_models").glob("net_g_*.safetensors") if (exp_dir / "resume_models").exists() else [])
        + list((exp_dir / "resume_models").glob("net_g_*.pth") if (exp_dir / "resume_models").exists() else []),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return ""
    # Préférer EMA si dispo parmi les candidats
    ema = [c for c in candidates if "ema" in c.stem]
    return str(ema[0]) if ema else str(candidates[0])


def _make_yaml(cfg: dict, n_iter: int, train_gt: str, val_gt: str, val_lq: str,
               base_arch: str = "compact", base_channels_last: bool = True,
               base_lq_size: int = 96, base_batch: int = 8) -> str:
    sched_block = cfg["sched_fn"](n_iter) if cfg["sched_fn"] else _SCHED_MULTISTEP
    # Precision tests override channels_last explicitly; others use base_channels_last
    cl = cfg["channels_last"]
    if cfg["category"] != "precision":
        cl = base_channels_last

    # Bicubic OTF mode: use pairedimagedataset (realesrgandataset not in Redux schema)
    # high_order_degradation removed — not a valid field in traiNNer-redux msgspec schema
    # Per-feature lq_size_override (e.g. spark_fd needs lq≥128 due to InceptionNext downsampling)
    effective_lq = cfg.get("lq_size_override") or base_lq_size

    # ECO pretrain — chemin direct ou auto-detect
    extra_path = cfg.get("extra_path_fields", "")
    if cfg.get("eco_pretrain_path"):
        # Chemin direct fourni (ex: modèle perso)
        eco_ckpt = cfg["eco_pretrain_path"]
        if Path(eco_ckpt).exists():
            print(f"\033[96m[eco_pretrain_path]\033[0m Pretrain direct : {eco_ckpt}", flush=True)
            extra_path = extra_path + f"  eco_pretrain_g: '{eco_ckpt}'\n"
        else:
            print(f"\033[93m[eco_pretrain_path]\033[0m Fichier introuvable : {eco_ckpt}"
                  " → ECO blending désactivé", flush=True)
    elif cfg.get("eco_pretrain_auto"):
        eco_ckpt = _find_eco_pretrain(base_arch)
        if eco_ckpt:
            print(f"\033[96m[eco_pretrain_auto]\033[0m Checkpoint trouvé : {eco_ckpt}", flush=True)
            extra_path = extra_path + f"  eco_pretrain_g: '{eco_ckpt}'\n"
        else:
            print(f"\033[93m[eco_pretrain_auto]\033[0m Aucun checkpoint RFeat_dataset_bicubic trouvé"
                  " → ECO blending désactivé (fallback)", flush=True)

    return _YAML_TEMPLATE.format(
        name=cfg["name"],
        use_amp=str(cfg["use_amp"]).lower(),
        amp_bf16=str(cfg["amp_bf16"]).lower(),
        channels_last=str(cl).lower(),
        fast_matmul=str(cfg["fast_matmul"]).lower(),
        use_compile=str(cfg["use_compile"]).lower(),
        dataset_type="pairedimagedataset",
        train_gt=train_gt,
        val_gt=val_gt,
        val_lq=val_lq,
        optim_block=cfg["optim_block"],
        sched_block=sched_block,
        losses_block=_LOSSES[cfg["loss_key"]],
        extra_train_fields=cfg.get("extra_train_fields", ""),
        extra_path_fields=extra_path,
        base_arch=base_arch,
        lq_size=effective_lq,
        batch_size=base_batch,
        n_iter=n_iter,
        print_freq=max(50, n_iter // 10),
    )


def _smi_thread(stop_event: threading.Event, readings: list) -> None:
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                timeout=5,
            ).decode().strip()
            mb = max(int(x) for x in out.splitlines() if x.strip().isdigit())
            readings.append(mb / 1024.0)
        except Exception:
            pass
        stop_event.wait(5.0)


def _run_upscale_with_model(model_path: str, output_dir: Path) -> tuple[bool, str]:
    """
    Lance un upscale de l'image de test (_DEFAULT_TEST_IMG) avec le modèle fourni.
    Utilise quick_upscale via subprocess pour éviter les conflits d'import.
    Retourne (ok, chemin_sortie_ou_message_erreur).
    """
    test_img = _DEFAULT_TEST_IMG
    if not test_img.exists():
        return False, f"Image de test introuvable : {test_img}"
    if not Path(model_path).exists():
        return False, f"Modèle introuvable : {model_path}"

    out_name = f"upscale_{Path(model_path).stem}.png"
    out_path = output_dir / out_name
    # Échappe les backslashes pour le script Python inline
    studio_root = str(_STUDIO_ROOT).replace("\\", "\\\\")
    model_esc   = str(model_path).replace("\\", "\\\\")
    img_esc     = str(test_img).replace("\\", "\\\\")
    out_esc     = str(out_path).replace("\\", "\\\\")

    script = (
        f"import sys; sys.path.insert(0, r'{_STUDIO_ROOT}')\n"
        f"from src.core.quick_upscale import upscale_image\n"
        f"ok, msg = upscale_image(r'{model_path}', r'{test_img}', r'{out_path}', "
        f"scale=0, tile_size=256, tile_pad=32, use_amp=True)\n"
        f"print('UPSCALE_OK' if ok else f'UPSCALE_ERR: {{msg}}')\n"
    )
    _up_env = os.environ.copy()
    _up_env["PYTHONIOENCODING"] = "utf-8"
    _up_env["PYTHONUTF8"] = "1"
    try:
        res = subprocess.run(
            [str(REDUX_PYTHON), "-c", script],
            capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace",
            env=_up_env,
            cwd=str(REDUX_PATH),
        )
        combined = res.stdout + res.stderr
        if "UPSCALE_OK" in combined:
            print(f"\033[92m[upscale_final]\033[0m OK → {out_path.name}", flush=True)
            return True, str(out_path)
        else:
            err = combined.strip()[-300:]
            print(f"\033[91m[upscale_final]\033[0m Échec : {err[:200]}", flush=True)
            return False, err
    except subprocess.TimeoutExpired:
        return False, "Upscale timeout (>180s)"
    except Exception as ex:
        return False, str(ex)


def run_feature_test(cfg: dict, n_iter: int, timeout: int,
                     train_gt: str, val_gt: str, val_lq: str,
                     output_dir: Path,
                     base_arch: str = "compact",
                     base_channels_last: bool = True,
                     base_lq_size: int = 96,
                     base_batch: int = 8) -> dict:
    name = cfg["name"]
    result = dict(
        name=name, desc=cfg["desc"], category=cfg["category"],
        status="error", error="",
        avg_itps=None, peak_vram_gb=None, peak_vram_smi_gb=None,
        psnr_final=None, elapsed_sec=0.0, iters_completed=0,
    )

    yaml_str = _make_yaml(cfg, n_iter, train_gt, val_gt, val_lq, base_arch, base_channels_last,
                          base_lq_size, base_batch)
    yaml_path = output_dir / f"_tmp_RFeat_{name}.yml"
    yaml_path.write_text(yaml_str, encoding="utf-8")
    exp_dir = REDUX_PATH / "experiments" / f"RFeat_{name}"

    # Nettoie le dossier expérience résiduel d'un run précédent (--reset ne le fait pas)
    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)
        print(f"[reset] {exp_dir.name} supprimé (run précédent)", flush=True)

    smi_readings: list = []
    stop_ev = threading.Event()
    th = threading.Thread(target=_smi_thread, args=(stop_ev, smi_readings), daemon=True)
    th.start()

    t0 = time.time()
    itps_readings: list = []
    vram_readings: list = []
    last_iter = 0
    psnr_val: float | None = None

    try:
        _env = os.environ.copy()
        _env["PYTHONIOENCODING"] = "utf-8"
        _env["PYTHONUTF8"] = "1"
        _env["NO_COLOR"] = "1"
        _env["PYTHONUNBUFFERED"] = "1"
        _env["COLUMNS"] = "200"
        proc = subprocess.Popen(
            [str(REDUX_PYTHON), str(TRAIN_SCRIPT), "-opt", str(yaml_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REDUX_PATH), text=True,
            encoding="utf-8", errors="replace", bufsize=1,
            env=_env,
        )
        print(f"{_CC}[{name}]{_C0} {_CD}Chargement dataset et modèle...{_C0}", flush=True)
        output_lines: list[str] = []
        try:
            for line in proc.stdout:  # type: ignore
                line = line.replace('\r', '')
                output_lines.append(line)
                _display_training_line(line, name)
                m = _RE_ITPS.search(line)
                if m:
                    itps_readings.append(float(m.group(1)))
                m = _RE_VRAM.search(line)
                if m:
                    vram_readings.append(float(m.group(1)))
                m = _RE_PSNR.search(line)
                if m:
                    psnr_val = float(m.group(1))
                m = _RE_ITER.search(line)
                if m:
                    last_iter = int(m.group(1).replace(',', ''))
                if time.time() - t0 > timeout:
                    proc.kill()
                    result["status"] = "timeout"
                    break
        except Exception:
            pass

        rc = proc.wait(timeout=30)
        result["iters_completed"] = last_iter
        result["elapsed_sec"] = round(time.time() - t0, 1)

        if result["status"] != "timeout":
            if rc == 0:
                if last_iter == 0:
                    result["status"] = "error"
                    tail = "".join(output_lines[-30:])
                    result["error"] = (
                        f"Training crashed silently (0 iters in {result['elapsed_sec']}s). "
                        "Likely cause: CUDA incompatibility (GPU sm_61/Pascal not supported by PyTorch 2.7+). "
                        "Fix: reinstall torch<=2.6.0+cu124.\n"
                        f"Output:\n{tail}"
                    )
                else:
                    result["status"] = "ok"
            else:
                result["status"] = "error"
                tail = "".join(output_lines[-30:])
                result["error"] = f"exit_code={rc}\n{tail}"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    finally:
        stop_ev.set()
        th.join(timeout=10)

    if itps_readings:
        data = itps_readings[5:] if len(itps_readings) > 8 else itps_readings
        result["avg_itps"] = round(sum(data) / len(data), 3)
    if vram_readings:
        result["peak_vram_gb"] = round(max(vram_readings), 2)
    if smi_readings:
        result["peak_vram_smi_gb"] = round(max(smi_readings), 2)
    if psnr_val is not None:
        result["psnr_final"] = round(psnr_val, 4)

    if exp_dir.exists() and not cfg.get("keep_exp_dir", False):
        shutil.rmtree(exp_dir, ignore_errors=True)
    elif exp_dir.exists() and cfg.get("keep_exp_dir", False):
        print(f"[keep_exp_dir] {exp_dir.name} conservé (utilisé par eco_pretrain_auto)", flush=True)
    yaml_path.unlink(missing_ok=True)

    # ── Upscale final avec modèle perso (optionnel) ───────────────────────────
    upscale_mdl = cfg.get("upscale_model_path", "")
    if upscale_mdl and result["status"] == "ok":
        print(f"\033[96m[upscale_final]\033[0m Lancement upscale avec {Path(upscale_mdl).name}...",
              flush=True)
        up_ok, up_info = _run_upscale_with_model(upscale_mdl, output_dir)
        result["upscale_ok"] = up_ok
        result["upscale_path"] = up_info if up_ok else ""
        result["upscale_error"] = "" if up_ok else up_info

    return result


def _update_coffre(result: dict, coffre_path: Path | None = None) -> None:
    if coffre_path is None or not coffre_path.exists():
        return
    try:
        text = coffre_path.read_text(encoding="utf-8")
        section = "## traiNNer-Redux — Benchmark Features (Laptop RTX 3070 Ti)"
        row = (
            f"| {result['name']} | {result['category']} | {result['status']} | "
            f"{result['avg_itps'] or '—'} | {result.get('psnr_final') or '—'} | {result['desc']} |"
        )
        if section not in text:
            header = (
                f"\n\n{section}\n"
                "GPU: RTX 3070 Ti Laptop (8 GB) — BF16 — compact arch — scale=1\n\n"
                "| Feature | Category | Status | it/s | PSNR | Description |\n"
                "|---|---|---|---|---|---|\n"
            )
            text += header + row + "\n"
        else:
            pat = re.compile(rf"\|\s*{re.escape(result['name'])}\s*\|.*")
            if pat.search(text):
                text = pat.sub(row, text)
            else:
                idx = text.find("| Feature | Category |")
                if idx >= 0:
                    nl2 = text.find("\n", text.find("\n", idx) + 1)
                    text = text[:nl2 + 1] + row + "\n" + text[nl2 + 1:]
                else:
                    text += row + "\n"
        coffre_path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"[COFFRE] Erreur: {e}", flush=True)


def _format_report(results: list, gpu: str, n_iter: int, base_arch: str = "compact") -> str:
    lines = [
        "=" * 100,
        f"  traiNNer-Redux Feature Benchmark  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  GPU: {gpu}  |  N_iter: {n_iter}  |  Arch: {base_arch}  |  AMP: BF16",
        "=" * 100,
        "",
        f"{'Feature':<25} {'Cat':<22} {'Status':<8}  {'it/s':>8}  {'VRAM':>6}  {'PSNR':>8}  {'Iters':>6}",
        "-" * 90,
    ]
    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    for cat in sorted(by_cat):
        lines.append(f"\n  [{cat}]")
        for r in by_cat[cat]:
            itps = f"{r['avg_itps']:.3f}" if r.get("avg_itps") else "  —"
            vram = f"{r.get('peak_vram_smi_gb') or r.get('peak_vram_gb') or 0:.2f}"
            psnr = f"{r['psnr_final']:.2f}" if r.get("psnr_final") else " —"
            lines.append(
                f"  {r['name']:<23} {r['category']:<22} {r['status']:<8}  {itps:>8}  "
                f"{vram:>5}  {psnr:>8}  {r['iters_completed']:>6}"
            )

    errors = [r for r in results if r["status"] in ("error", "timeout")]
    if errors:
        lines += ["", "", "ERREURS :", "-" * 70]
        for r in errors:
            lines.append(f"\n[{r['status'].upper()} — {r['name']}]")
            lines.append(r.get("error", "")[:600])

    lines.append("=" * 100)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter",     type=int, default=500)
    ap.add_argument("--timeout",    type=int, default=900)
    ap.add_argument("--tests",      type=str, default="")
    ap.add_argument("--output-dir", type=str, default="")
    ap.add_argument("--train-gt",   type=str, default=DEFAULT_TRAIN_GT)
    ap.add_argument("--val-gt",     type=str, default=DEFAULT_VAL_GT)
    ap.add_argument("--val-lq",     type=str, default=DEFAULT_VAL_LQ)
    ap.add_argument("--reset",      action="store_true")
    ap.add_argument("--no-amp",      action="store_true",
                    help="Désactive AMP (use_amp=false, amp_bf16=false) pour tous les tests. "
                         "Utiliser sur desktop si GPU < Ampere ou sans support BF16.")
    ap.add_argument("--coffre-path", type=str, default="",
                    help="Chemin vers la note Obsidian pour mise à jour auto. "
                         "Vide = pas de mise à jour (défaut).")
    ap.add_argument("--list",       action="store_true")
    ap.add_argument("--base-arch",          type=str, default="compact",
                    help="Arch de base pour tous les tests features (défaut: compact). "
                         "Passé automatiquement par redux_arch_benchmark.py --auto-chain.")
    ap.add_argument("--base-channels-last", type=str, default="true",
                    help="channels_last de l'arch de base (true/false). Défaut: true.")
    ap.add_argument("--base-lq-size",       type=int, default=96,
                    help="lq_size de l'arch de base en pixels (défaut: 96).")
    ap.add_argument("--base-batch",         type=int, default=8,
                    help="batch_size_per_gpu de l'arch de base (défaut: 8).")
    args = ap.parse_args()

    if args.list:
        cats: dict = {}
        for f in FEATURE_LIST:
            cats.setdefault(f["category"], []).append(f)
        for cat, items in sorted(cats.items()):
            print(f"\n  [{cat}]")
            for f in items:
                print(f"    {f['name']:<25}  {f['desc']}")
        return

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ── Validation de l'environnement ────────────────────────────────────────
    print(f"[INFO] traiNNer-redux : {REDUX_PATH}", flush=True)
    print(f"[INFO] Python venv    : {REDUX_PYTHON}", flush=True)
    if not REDUX_PATH.exists():
        print(f"[ERREUR] Dossier traiNNer-redux introuvable : {REDUX_PATH}", flush=True)
        sys.exit(1)
    if not REDUX_PYTHON.exists():
        print(f"[ERREUR] Venv Python introuvable sous {REDUX_PATH}", flush=True)
        sys.exit(1)

    # --no-amp : on utilise un sous-dossier séparé pour ne pas mélanger desktop/laptop
    _suffix = "_noamp" if args.no_amp else ""
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path.home() / "IA_Engine" / "benchmark_results" / f"redux_feat{_suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / f"redux_feat{_suffix}_state.json"

    if args.no_amp:
        print("[INFO] Mode --no-amp : use_amp=false, amp_bf16=false forcé sur tous les tests.",
              flush=True)

    if args.reset and state_path.exists():
        state_path.unlink()
        print("[RESET] État effacé.", flush=True)

    state: dict = {}
    if state_path.exists() and not args.reset:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            print(f"[RESUME] {len(state.get('completed', {}))} tests déjà complétés.", flush=True)
        except Exception:
            state = {}

    state.setdefault("completed", {})
    state.setdefault("results", [])

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5,
        ).decode().strip().splitlines()[0]
    except Exception:
        gpu = "Unknown GPU"
    print(f"[INFO] GPU: {gpu}", flush=True)
    state["gpu"] = gpu
    state["n_iter"] = args.n_iter

    coffre = Path(args.coffre_path) if args.coffre_path else None
    if coffre:
        print(f"[INFO] Coffre: {coffre}", flush=True)

    base_arch = args.base_arch
    base_cl   = args.base_channels_last.lower() not in ("false", "0", "no")
    base_lq   = args.base_lq_size
    base_bat  = args.base_batch
    print(f"[INFO] Base arch: {base_arch}  channels_last={base_cl}  "
          f"lq_size={base_lq}  batch={base_bat}", flush=True)

    filt = [x.strip() for x in args.tests.split(",") if x.strip()]
    todo = [f for f in FEATURE_LIST if not filt or f["name"] in filt]
    todo = [f for f in todo if f["name"] not in state["completed"]]

    n_done = len(state["completed"])
    total  = len(todo) + n_done
    print(f"[INFO] {len(todo)} tests à faire, {n_done} déjà complétés.", flush=True)

    for i, cfg in enumerate(todo, start=1):
        name = cfg["name"]
        # Appliquer --no-amp : override use_amp/amp_bf16 sur tous les tests
        if args.no_amp:
            cfg = dict(cfg, use_amp=False, amp_bf16=False)

        print(f"\n{'='*60}", flush=True)
        idx_str = f"[{n_done + i}/{total}] {name} [{cfg['category'].upper()}]"
        dashes  = "─" * max(0, 72 - len(idx_str) - 14)
        print(f"\n{_CW}[Benchmark]{_C0} {_CD}────{_C0} {idx_str} {dashes}", flush=True)
        amp_info = "" if not args.no_amp else "  [no-amp]"
        print(f"  {cfg['desc']}{amp_info}", flush=True)

        r = run_feature_test(
            cfg=cfg, n_iter=args.n_iter, timeout=args.timeout,
            train_gt=args.train_gt, val_gt=args.val_gt, val_lq=args.val_lq,
            output_dir=out_dir,
            base_arch=base_arch, base_channels_last=base_cl,
            base_lq_size=base_lq, base_batch=base_bat,
        )

        state["completed"][name] = True
        existing = next((x for x in state["results"] if x["name"] == name), None)
        if existing:
            existing.update(r)
        else:
            state["results"].append(r)

        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        _update_coffre(r, coffre)

        st = r["status"]
        itps = f"{r['avg_itps']:.3f} it/s" if r.get("avg_itps") else "—"
        psnr = f"PSNR={r['psnr_final']:.2f}" if r.get("psnr_final") else ""
        _st_color = _CG if st == "ok" else _CR
        print(f"{_CW}[DONE]{_C0} {name}: {_st_color}{st}{_C0}  {itps}  {psnr}", flush=True)

    all_results = state["results"]
    if all_results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt = _format_report(all_results, gpu, args.n_iter, base_arch)
        rpt = out_dir / f"redux_feat_{ts}.txt"
        rpt.write_text(txt, encoding="utf-8")
        js  = out_dir / f"redux_feat_{ts}.json"
        js.write_text(json.dumps({"gpu": gpu, "timestamp": ts,
                                   "results": all_results}, indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        print(f"\n[RAPPORT] {rpt}", flush=True)
        print(txt, flush=True)


if __name__ == "__main__":
    main()
