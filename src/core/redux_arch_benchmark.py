"""
redux_arch_benchmark.py — traiNNer-Redux Architecture Benchmark

Teste les architectures traiNNer-Redux en plusieurs modes de précision :
  normal (FP32) / fp16 (AMP FP16) / bf16 (AMP BF16) / tf32 (fast_matmul)

À la fin, lance automatiquement redux_feature_benchmark.py avec l'arch la
plus rapide en BF16 (ou normal si BF16 non actif) comme base.

Resume : state JSON sauvegardé après chaque test (clé : "arch__mode").

Usage:
    python redux_arch_benchmark.py [options]

    --n-iter 2500                  Itérations par test (défaut : 2500)
    --timeout 3600                 Timeout par test en secondes (défaut : 3600)
    --tests compact,span           Tester seulement ces archs
    --modes normal,fp16,bf16,tf32  Modes à tester (défaut : voir DEFAULT_MODES)
    --output-dir PATH              Dossier résultats
    --train-gt PATH                Dataset training GT
    --no-upscale                   Passer le test upscale
    --reset                        Remettre à zéro (ignorer état existant)
    --list                         Lister tous les archs disponibles
    --coffre-path PATH             Note Obsidian (optionnel)
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

_THIS_DIR              = Path(__file__).parent
NEOSR_GENERAL_RUNNER   = _THIS_DIR / "neosr_general_runner.py"
NEOSR_PYTHON           = Path.home() / "IA_Engine" / "neosr" / ".venv" / "Scripts" / "python.exe"
REDUX_INFERENCE_RUNNER = _THIS_DIR / "redux_inference_runner.py"


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
    return Path(sys.executable)   # fallback : Python courant (avertissement dans main)


REDUX_PYTHON = _find_venv_python(REDUX_PATH)

DEFAULT_TRAIN_GT = str(Path.home() / "IA_Engine/datasets/train/HR")
DEFAULT_VAL_GT   = str(Path.home() / "IA_Engine/datasets/val/GT")
DEFAULT_VAL_LQ   = str(Path.home() / "IA_Engine/datasets/val/LQ")
DEFAULT_TEST_IMG = str(Path.home() / "IA_Engine/datasets/val/LQ/Overlord (1).png")

# ── Modes de précision ─────────────────────────────────────────────────────────
_PRECISION_MODES = {
    "normal": dict(use_amp="false", amp_bf16="false", fast_matmul="false"),
    "fp16":   dict(use_amp="true",  amp_bf16="false", fast_matmul="false"),
    "bf16":   dict(use_amp="true",  amp_bf16="true",  fast_matmul="false"),
    "tf32":   dict(use_amp="false", amp_bf16="false", fast_matmul="true"),
}

# ← "normal" pour desktop | "normal,fp16,bf16,tf32" pour laptop
DEFAULT_MODES = "normal"

# ── YAML base template ─────────────────────────────────────────────────────────
_YAML_TEMPLATE = """\
name: RBench_{run_name}
scale: 1
use_amp: {use_amp}
amp_bf16: {amp_bf16}
use_channels_last: {channels_last}
fast_matmul: {fast_matmul}
use_compile: false
num_gpu: auto
manual_seed: 42

datasets:
  train:
    name: RBenchTrain
    type: pairedimagedataset
    dataroot_gt: ['{train_gt}']
    dataroot_lq: ['{train_gt}']
    lq_size: {lq_size}
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 2
    batch_size_per_gpu: {batch_size}
    accum_iter: {accum_iter}
  val:
    name: RBenchVal
    type: pairedimagedataset
    dataroot_gt: ['{val_gt}']
    dataroot_lq: ['{val_lq}']

network_g:
  type: {arch_type}
{extra_ng_params}
path:
  param_key_g: ~
  strict_load_g: true
  resume_state: ~

train:
  ema_decay: 0.999
  ema_power: 0.75
  grad_clip: false
  optim_g:
    type: AdamW
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]
  scheduler:
    type: MultiStepLR
    milestones: [999999]
    gamma: 0.5
  total_iter: {n_iter}
  warmup_iter: -1
  losses:
    - type: charbonnierloss
      loss_weight: 1.0

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

# ── Arch registry ──────────────────────────────────────────────────────────────
def _a(name, desc, tier, arch_type=None, lq_size=64, batch=8, accum=1,
       channels_last=True, extra_ng=""):
    return dict(
        name=name, desc=desc, tier=tier,
        arch_type=arch_type or name,
        lq_size=lq_size, batch=batch, accum=accum,
        channels_last=channels_last,
        extra_ng=extra_ng,
    )


ARCH_LIST = [
    # ── TIER 1 — Ultra-light/fast (< 1.5 GB VRAM) ─────────────────────────────
    _a("superultracompact", "SuperUltraCompact (srvgg 8f16)", 1, lq_size=96, batch=8, channels_last=True),
    _a("ultracompact",      "UltraCompact (srvgg 16f32)",     1, lq_size=96, batch=8, channels_last=True),
    _a("compact",           "Compact (srvgg 64f16) — baseline", 1, lq_size=96, batch=8, channels_last=True),

    _a("artcnn_r8f64",   "ArtCNN R8F64",   1, lq_size=64, batch=8, channels_last=True),
    _a("span_s",         "SPAN-S",         1, lq_size=64, batch=8, channels_last=False),
    _a("safmn",          "SAFMN",          1, lq_size=64, batch=8, channels_last=False),
    _a("rtmosr",         "RTMoSR",         1, lq_size=64, batch=8, channels_last=False),
    _a("mosr_t",         "MoSR-T",         1, lq_size=64, batch=8, channels_last=True),
    _a("mosrv2",         "MoSRV2",         1, lq_size=64, batch=8, channels_last=True),
    _a("realplksr_tiny", "RealPLKSR-Tiny", 1, lq_size=64, batch=8, channels_last=True),

    # ── TIER 2 — Medium (1.5–4 GB VRAM) ──────────────────────────────────────
    _a("artcnn_r16f96", "ArtCNN R16F96",    2, lq_size=64, batch=8, channels_last=True),
    _a("safmn_l",       "SAFMN-L",          2, lq_size=64, batch=8, channels_last=False),
    _a("rtmosr_l",      "RTMoSR-L",         2, lq_size=64, batch=8, channels_last=False),
    _a("plksr_tiny",    "PLKSR-Tiny",       2, lq_size=64, batch=8, channels_last=True),
    _a("plksr",         "PLKSR",            2, lq_size=64, batch=4, channels_last=True),
    _a("realplksr",     "RealPLKSR",        2, lq_size=64, batch=4, channels_last=True),
    _a("man_tiny",      "MAN-Tiny",         2, lq_size=64, batch=8, channels_last=False),
    _a("man_light",     "MAN-Light",        2, lq_size=64, batch=4, channels_last=False),
    _a("elan_light",    "ELAN-Light",       2, lq_size=64, batch=8, channels_last=False),
    _a("span",          "SPAN",             2, lq_size=64, batch=8, channels_last=False),
    _a("spanplus_s",    "SPANPlus-S",       2, lq_size=64, batch=8, channels_last=False),
    _a("lmlt_tiny",     "LMLT-Tiny",        2, lq_size=64, batch=8, channels_last=True),
    _a("ditn_real",     "DITN-Real",        2, lq_size=64, batch=8, channels_last=False),
    _a("esrgan_lite",   "ESRGAN-Lite",      2, lq_size=64, batch=8, channels_last=True),
    _a("esrgan",        "ESRGAN (RRDB)",    2, lq_size=64, batch=4, channels_last=True),
    _a("omnisr",        "OmniSR",           2, lq_size=64, batch=4, channels_last=False),
    _a("sebica",        "Sebica",           2, lq_size=64, batch=8, channels_last=True),
    _a("seemore_t",     "SeemoRe-T",        2, lq_size=64, batch=8, channels_last=False),
    _a("lkfmixer_t",    "LKFMixer-T",       2, lq_size=64, batch=8, channels_last=False),
    _a("gaterv3_s",     "GaterV3-S",        2, lq_size=64, batch=8, channels_last=True),
    _a("flexnet",       "FlexNet",          2, lq_size=64, batch=8, channels_last=False),

    # ── TIER 3 — Heavy (4–8 GB) ───────────────────────────────────────────────
    _a("man",            "MAN (full)",       3, lq_size=64, batch=4, accum=1, channels_last=False),
    _a("swinir_s",       "SwinIR-S",         3, lq_size=48, batch=4, accum=1, channels_last=True),
    _a("eimn_a",         "EIMN-A",           3, lq_size=48, batch=4, accum=1, channels_last=False),
    _a("drct",           "DRCT (use_checkpoint)", 3, lq_size=48, batch=2, accum=2,
       channels_last=True, extra_ng="  use_checkpoint: true\n"),
    _a("srformer_light", "SRFormer-Light (use_checkpoint)", 3, lq_size=48, batch=2, accum=2,
       channels_last=True, extra_ng="  use_checkpoint: true\n"),

    # ── VARIANTS NON BENCHMARKÉS — Ajoutés 2026-05-19 ────────────────────────
    # ArtCNN variants (plus petits que r8f64)
    _a("artcnn_r8f48",  "ArtCNN R8F48",     1, lq_size=64, batch=8, channels_last=True),
    _a("artcnn_r3f24",  "ArtCNN R3F24",     1, lq_size=64, batch=8, channels_last=True),

    # RTMoSR Ultra-Light
    _a("rtmosr_ul",     "RTMoSR-UL",        1, lq_size=64, batch=8, channels_last=False),

    # MoSR full (plus grand que mosr_t)
    _a("mosr",          "MoSR (full)",      2, lq_size=64, batch=8, channels_last=True),

    # Sebica Mini
    _a("sebica_mini",   "Sebica-Mini",      1, lq_size=64, batch=8, channels_last=True),

    # SPANPlus variants
    _a("spanplus",      "SPANPlus",         2, lq_size=64, batch=8, channels_last=False),
    _a("spanplus_st",   "SPANPlus-ST",      2, lq_size=64, batch=8, channels_last=False),
    _a("spanplus_sts",  "SPANPlus-STS",     2, lq_size=64, batch=8, channels_last=False),

    # EIMN variants (base et large)
    _a("eimn",          "EIMN (base)",      2, lq_size=48, batch=4, channels_last=False),
    _a("eimn_l",        "EIMN-L",           3, lq_size=48, batch=4, channels_last=False),

    # LMLT variants (base et large)
    _a("lmlt_base",     "LMLT-Base",        2, lq_size=64, batch=8, channels_last=True),
    _a("lmlt_large",    "LMLT-Large",       3, lq_size=48, batch=4, channels_last=True),

    # LKFMixer variants
    _a("lkfmixer_b",    "LKFMixer-B",       2, lq_size=64, batch=4, channels_last=False),
    _a("lkfmixer_l",    "LKFMixer-L",       3, lq_size=48, batch=2, accum=2, channels_last=False),

    # GaterV3 full
    _a("gaterv3",       "GaterV3 (full)",   2, lq_size=64, batch=8, channels_last=True),

    # DITN base (pas real)
    _a("ditn",          "DITN (base)",      2, lq_size=64, batch=4, channels_last=False),

    # ELAN full (plus lourd que elan_light)
    _a("elan",          "ELAN (full)",      3, lq_size=48, batch=4, channels_last=False),

    # SwinIR-M (plus lourd que SwinIR-S)
    _a("swinir_m",      "SwinIR-M",         3, lq_size=48, batch=2, accum=2,
       channels_last=True, extra_ng="  use_checkpoint: true\n"),

    # SRFormerV2 (différent de srformer_light qui crash Pascal)
    _a("srformerv2",    "SRFormerV2",       3, lq_size=48, batch=2, accum=2,
       channels_last=True, extra_ng="  use_checkpoint: true\n"),

    # DRCT-XL (très lourd)
    _a("drct_xl",       "DRCT-XL (use_checkpoint)", 3, lq_size=48, batch=1, accum=4,
       channels_last=True, extra_ng="  use_checkpoint: true\n"),

    # RealPLKSR-Large
    _a("realplksr_large", "RealPLKSR-Large", 3, lq_size=64, batch=4, channels_last=True),

    # Swin2SR variants
    _a("swin2sr_s",     "Swin2SR-S",        2, lq_size=48, batch=4, channels_last=False),
    _a("swin2sr_m",     "Swin2SR-M",        3, lq_size=48, batch=2, accum=2,
       channels_last=False, extra_ng="  use_checkpoint: true\n"),

    # DAT variants
    _a("dat_s",         "DAT-S",            3, lq_size=48, batch=2, accum=2,
       channels_last=False, extra_ng="  use_checkpoint: true\n"),
    _a("dat_light",     "DAT-Light",        2, lq_size=64, batch=4, channels_last=False),

    # ── Sprint 19/20 — Nouvelles architectures communautaires ─────────────────
    # SpanF : SPAN simplifié, blocs SPAB1, très léger
    _a("spanf",         "SpanF (fc=32)",        1, lq_size=64, batch=8, channels_last=False,
       extra_ng="  feature_channels: 32\n"),

    # SpanPP / SpanC : multi-scale IGConv, reparamétrisable
    _a("spanc",         "SpanPP/SpanC (fc=48, s×1)", 1, lq_size=64, batch=8, channels_last=False,
       extra_ng="  feature_channels: 48\n  scale_list: [1]\n  eval_base_scale: 1\n"),

    # GFISRv2 : GatedCNN + FFT-inspired multi-upsampler
    _a("gfisrv2",       "GFISRv2 (dim=48)",     1, lq_size=64, batch=8, channels_last=False,
       extra_ng="  dim: 48\n  n_blocks: 24\n"),

    # SMoSR : léger Self-Modulation, compétitif SPAN-S
    _a("smosr",         "SMoSR (dim=48)",       1, lq_size=64, batch=8, channels_last=False,
       extra_ng="  dim: 48\n  n_mb: 3\n"),
]

_ARCH_BY_NAME = {a["name"]: a for a in ARCH_LIST}

# ── Output regex ───────────────────────────────────────────────────────────────
_RE_ITPS = re.compile(r"\[performance:\s*([\d.]+)\s*it/s\]", re.IGNORECASE)
_RE_VRAM = re.compile(r"\[peak VRAM:\s*([\d.]+)\s*GB\]",    re.IGNORECASE)
_RE_PSNR = re.compile(r"#\s+psnr\s*:\s*([\d.]+)",           re.IGNORECASE)
_RE_SSIM = re.compile(r"#\s+ssim\s*:\s*([\d.]+)",           re.IGNORECASE)
_RE_ITER = re.compile(r"iter:\s*([\d,]+)",                    re.IGNORECASE)

# ── Console display ──────────────────────────────────────────────────────────
_C0 = "\033[0m"    # reset
_CB = "\033[1m"    # bold
_CD = "\033[2m"    # dim
_CC = "\033[96m"   # cyan   — arch prefix
_CG = "\033[92m"   # green  — it/s, DONE ok
_CY = "\033[93m"   # yellow — iter, epoch
_CL = "\033[94m"   # blue   — VRAM
_CR = "\033[91m"   # red    — errors
_CW = "\033[1;97m" # bold white — header

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


def _make_yaml(cfg: dict, mode_name: str, n_iter: int,
               train_gt: str, val_gt: str, val_lq: str) -> str:
    mode = _PRECISION_MODES[mode_name]
    return _YAML_TEMPLATE.format(
        run_name=f"{cfg['name']}_{mode_name}",
        use_amp=mode["use_amp"],
        amp_bf16=mode["amp_bf16"],
        fast_matmul=mode["fast_matmul"],
        channels_last=str(cfg["channels_last"]).lower(),
        arch_type=cfg["arch_type"],
        extra_ng_params=cfg["extra_ng"],
        lq_size=cfg["lq_size"],
        batch_size=cfg["batch"],
        accum_iter=cfg["accum"],
        n_iter=n_iter,
        print_freq=min(100, n_iter // 10),
        train_gt=train_gt,
        val_gt=val_gt,
        val_lq=val_lq,
    )


def _smi_vram_thread(stop_event: threading.Event, readings: list) -> None:
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


def run_arch_test(cfg: dict, mode_name: str, n_iter: int, timeout: int,
                  train_gt: str, val_gt: str, val_lq: str,
                  do_upscale: bool, test_img: str, output_dir: Path) -> dict:
    name     = cfg["name"]
    run_name = f"{name}_{mode_name}"
    result   = dict(
        name=name, mode=mode_name, desc=cfg["desc"], tier=cfg["tier"],
        status="error", error="",
        avg_itps=None, peak_vram_gb=None, peak_vram_smi_gb=None,
        psnr_readings={}, ssim_readings={},
        elapsed_sec=0.0, iters_completed=0,
        upscale_test=None,
    )

    yaml_str  = _make_yaml(cfg, mode_name, n_iter, train_gt, val_gt, val_lq)
    yaml_path = output_dir / f"_tmp_RBench_{run_name}.yml"
    yaml_path.write_text(yaml_str, encoding="utf-8")
    exp_dir   = REDUX_PATH / "experiments" / f"RBench_{run_name}"

    smi_readings: list = []
    stop_event  = threading.Event()
    smi_thread  = threading.Thread(
        target=_smi_vram_thread, args=(stop_event, smi_readings), daemon=True
    )
    smi_thread.start()

    t0            = time.time()
    itps_readings: list = []
    vram_readings: list = []
    last_iter     = 0

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
                    result["psnr_readings"][last_iter] = float(m.group(1))
                m = _RE_SSIM.search(line)
                if m:
                    result["ssim_readings"][last_iter] = float(m.group(1))
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
        result["elapsed_sec"]     = round(time.time() - t0, 1)

        if result["status"] != "timeout":
            result["status"] = "ok" if rc == 0 else "error"
            if rc != 0:
                result["error"] = f"exit_code={rc}\n" + "".join(output_lines[-30:])
            elif last_iter == 0 and result["elapsed_sec"] < 60:
                result["status"] = "error"
                result["error"] = (
                    f"Training crashed silently (0 iters in {result['elapsed_sec']}s). "
                    "Likely cause: CUDA incompatibility (GPU sm_61/Pascal not supported "
                    "by PyTorch 2.7+). Fix: reinstall torch<=2.6.0+cu124.\n"
                    "Output:\n" + "".join(output_lines[-25:])
                )

    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)
    finally:
        stop_event.set()
        smi_thread.join(timeout=10)

    if itps_readings:
        data = itps_readings[5:] if len(itps_readings) > 8 else itps_readings
        result["avg_itps"] = round(sum(data) / len(data), 3)
    if vram_readings:
        result["peak_vram_gb"] = round(max(vram_readings), 2)
    if smi_readings:
        result["peak_vram_smi_gb"] = round(max(smi_readings), 2)

    # Upscale : seulement sur bf16 (ou normal si bf16 absent) pour économiser du temps
    if do_upscale and result["status"] == "ok":
        if not test_img or not Path(test_img).exists():
            print(f"{_CC}[{name}]{_C0} {_CR}[UPSCALE]{_C0} image test introuvable: {test_img!r}", flush=True)
            result["upscale_test"] = {"status": "no_img"}
        else:
            models_dir = exp_dir / "models"
            ckpt = None
            if models_dir.exists():
                for ext in [".safetensors", ".pth"]:
                    candidates = sorted(models_dir.glob(f"net_g_*{ext}"))
                    if candidates:
                        ckpt = candidates[-1]
                        break
            if ckpt:
                print(f"{_CC}[{name}]{_C0} {_CD}Test upscale:{_C0} {ckpt.name}", flush=True)
                up_out = output_dir / "upscale_tests" / f"RBench_{run_name}_{ckpt.stem}.png"
                up_out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    r = subprocess.run(
                        [str(REDUX_PYTHON), str(REDUX_INFERENCE_RUNNER),
                         str(ckpt), test_img, str(up_out), name],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=180,
                    )
                    if r.returncode == 0 and up_out.exists():
                        result["upscale_test"] = {
                            "status": "ok", "output": str(up_out),
                            "size_kb": round(up_out.stat().st_size / 1024),
                        }
                        print(f"{_CC}[{name}]{_C0} {_CG}[UPSCALE OK]{_C0} → {up_out.name}", flush=True)
                    else:
                        result["upscale_test"] = {"status": "error", "stderr": r.stdout[-500:]}
                        print(f"{_CC}[{name}]{_C0} {_CR}[UPSCALE ERR]{_C0} {r.stdout[-200:]}", flush=True)
                except Exception as e:
                    result["upscale_test"] = {"status": "error", "error": str(e)}
                    print(f"{_CC}[{name}]{_C0} {_CR}[UPSCALE ERR]{_C0} {e}", flush=True)
            else:
                print(f"{_CC}[{name}]{_C0} {_CR}[UPSCALE]{_C0} aucun checkpoint dans {models_dir}", flush=True)
                result["upscale_test"] = {"status": "no_ckpt"}

    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)
    yaml_path.unlink(missing_ok=True)
    return result


def _update_coffre(result: dict, all_results: list, coffre_path: "Path | None" = None) -> None:
    if coffre_path is None or not coffre_path.exists():
        return
    try:
        text   = coffre_path.read_text(encoding="utf-8")
        header = "## traiNNer-Redux — Benchmark Architectures (Laptop RTX 3070 Ti)"
        row    = _format_coffre_row(result)

        if header not in text:
            text += (
                f"\n\n{header}\n"
                "GPU: RTX 3070 Ti Laptop (8 GB) — 2500 iters — scale=1\n\n"
                "| Arch | Mode | Tier | it/s | VRAM GB | PSNR@end | Upscale | Description |\n"
                "|---|---|---|---|---|---|---|---|\n"
            ) + row + "\n"
        else:
            pat = re.compile(
                rf"\|\s*{re.escape(result['name'])}\s*\|\s*{re.escape(result['mode'])}\s*\|.*"
            )
            if pat.search(text):
                text = pat.sub(row, text)
            else:
                anchor = "| Arch | Mode | Tier | it/s |"
                idx = text.find(anchor)
                if idx >= 0:
                    nl2 = text.find("\n", text.find("\n", idx) + 1)
                    text = text[:nl2 + 1] + row + "\n" + text[nl2 + 1:]
                else:
                    text += row + "\n"

        coffre_path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"[COFFRE] Erreur: {e}", flush=True)


def _format_coffre_row(r: dict) -> str:
    itps  = f"{r['avg_itps']:.3f}" if r.get("avg_itps") else "—"
    vram  = f"{r.get('peak_vram_smi_gb') or r.get('peak_vram_gb') or 0:.2f}"
    psnrs = list(r.get("psnr_readings", {}).values())
    psnr  = f"{psnrs[-1]:.2f}" if psnrs else "—"
    ups   = r.get("upscale_test")
    up_s  = ups.get("status", "—") if ups else "—"
    return (
        f"| {r['name']} | {r.get('mode','?')} | {r['tier']} | {itps} | "
        f"{vram} | {psnr} | {up_s} | {r['desc']} |"
    )


def _format_report(results: list, gpu: str, n_iter: int, modes: list) -> str:
    modes_str = " / ".join(m.upper() for m in modes)
    lines = [
        "=" * 115,
        f"  traiNNer-Redux Architecture Benchmark  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  GPU: {gpu}  |  N_iter: {n_iter}  |  Modes: {modes_str}",
        "=" * 115,
    ]

    # Group results by arch name
    arch_map: dict = {}
    for r in results:
        arch_map.setdefault(r["name"], {})[r.get("mode", "?")] = r

    W   = 22   # arch name column width
    COL = 24   # per-mode column width

    hdr = f"\n  {'Arch':<{W}} {'Tier':>4}  {'Status':<8}"
    for m in modes:
        hdr += f"  {m.upper():<{COL}}"
    lines.append(hdr)
    lines.append("-" * 115)

    by_tier: dict = {}
    for arch_name, modes_dict in arch_map.items():
        first = next(iter(modes_dict.values()))
        by_tier.setdefault(first["tier"], []).append((arch_name, modes_dict))

    for tier in sorted(by_tier):
        lines.append(f"\n  [TIER {tier}]")
        for arch_name, modes_dict in by_tier[tier]:
            first    = next(iter(modes_dict.values()))
            statuses = [modes_dict[m]["status"] for m in modes if m in modes_dict]
            worst    = ("ok" if all(s == "ok" for s in statuses)
                        else "timeout" if "timeout" in statuses else "error")
            line = f"  {arch_name:<{W}} {first['tier']:>4}  {worst:<8}"
            for m in modes:
                r = modes_dict.get(m)
                if r is None:
                    line += f"  {'—':<{COL}}"
                else:
                    itps  = f"{r['avg_itps']:.2f}" if r.get("avg_itps") else "—"
                    vram  = f"{r.get('peak_vram_smi_gb') or r.get('peak_vram_gb') or 0:.1f}G"
                    psnrs = list(r.get("psnr_readings", {}).values())
                    psnr  = f"{psnrs[-1]:.1f}" if psnrs else "—"
                    cell  = f"{itps} it/s {vram} P:{psnr}"
                    line += f"  {cell:<{COL}}"
            lines.append(line)

    lines += ["", "=" * 115, "", "Descriptions :", "-" * 70]
    seen: set = set()
    for r in results:
        if r["name"] not in seen:
            lines.append(f"  {r['name']:<25} tier={r['tier']}  {r['desc']}")
            seen.add(r["name"])

    errors = [r for r in results if r["status"] in ("error", "timeout")]
    if errors:
        lines += ["", "", "ERREURS :", "-" * 70]
        for r in errors:
            lines.append(f"\n[{r['status'].upper()} — {r['name']} / {r.get('mode','?')}]")
            lines.append(r.get("error", "")[:800])

    lines.append("=" * 115)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter",      type=int, default=2500)
    ap.add_argument("--timeout",     type=int, default=3600)
    ap.add_argument("--tests",       type=str, default="")
    ap.add_argument("--modes",       type=str, default=DEFAULT_MODES,
                    help=f"Modes séparés par virgule. Défaut: '{DEFAULT_MODES}'")
    ap.add_argument("--output-dir",  type=str, default="")
    ap.add_argument("--train-gt",    type=str, default=DEFAULT_TRAIN_GT)
    ap.add_argument("--val-gt",      type=str, default=DEFAULT_VAL_GT)
    ap.add_argument("--val-lq",      type=str, default=DEFAULT_VAL_LQ)
    ap.add_argument("--test-img",    type=str, default=DEFAULT_TEST_IMG)
    ap.add_argument("--no-upscale",  action="store_true")
    ap.add_argument("--coffre-path", type=str, default="")
    ap.add_argument("--reset",       action="store_true")
    ap.add_argument("--list",        action="store_true")
    args = ap.parse_args()

    if args.list:
        for a in ARCH_LIST:
            print(f"  {a['name']:<25} tier={a['tier']}  {a['desc']}")
        return

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ── Validation de l'environnement ────────────────────────────────────────
    print(f"[INFO] traiNNer-redux : {REDUX_PATH}", flush=True)
    print(f"[INFO] Python venv    : {REDUX_PYTHON}", flush=True)
    if not REDUX_PATH.exists():
        print(f"[ERREUR] Dossier traiNNer-redux introuvable : {REDUX_PATH}", flush=True)
        print("         Vérifiez que traiNNer-redux est installé dans ~/IA_Engine/traiNNer-redux/",
              flush=True)
        sys.exit(1)
    if not TRAIN_SCRIPT.exists():
        print(f"[ERREUR] train.py introuvable : {TRAIN_SCRIPT}", flush=True)
        sys.exit(1)
    if not REDUX_PYTHON.exists():
        print(f"[ERREUR] Venv Python introuvable. Chemins essayés :", flush=True)
        for p in [REDUX_PATH / d / s / "python.exe"
                  for d in ("venv", ".venv") for s in ("Scripts", "bin")]:
            print(f"         {p}  {'✓' if p.exists() else '✗'}", flush=True)
        print("         Créez le venv : cd ~/IA_Engine/traiNNer-redux && python -m venv venv",
              flush=True)
        sys.exit(1)
    print(f"[INFO] train.py       : {TRAIN_SCRIPT}  ✓", flush=True)

    # Validate modes
    active_modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in active_modes if m not in _PRECISION_MODES]
    if invalid:
        print(f"[ERROR] Modes inconnus: {invalid}. Valides: {list(_PRECISION_MODES)}")
        sys.exit(1)
    print(f"[INFO] Modes actifs: {active_modes}", flush=True)

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path.home() / "IA_Engine" / "benchmark_results" / "redux_bench"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "redux_bench_state.json"

    if args.reset and state_path.exists():
        state_path.unlink()
        print("[RESET] État effacé.", flush=True)

    state: dict = {}
    if state_path.exists() and not args.reset:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            n_prev = len(state.get("completed", {}))
            print(f"[RESUME] {n_prev} tests déjà complétés.", flush=True)
        except Exception:
            state = {}

    state.setdefault("completed", {})
    state.setdefault("results", [])

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=5,
        ).decode().strip().splitlines()[0]
    except Exception:
        gpu = "Unknown GPU"
    print(f"[INFO] GPU: {gpu}", flush=True)
    state["gpu"]    = gpu
    state["n_iter"] = args.n_iter

    coffre = Path(args.coffre_path) if args.coffre_path else None
    if coffre:
        print(f"[INFO] Coffre: {coffre}", flush=True)

    arch_filter = [x.strip() for x in args.tests.split(",") if x.strip()]
    arch_todo   = [a for a in ARCH_LIST if not arch_filter or a["name"] in arch_filter]

    # Build todo: (cfg, mode_name) pairs not yet completed
    todo: list = []
    for cfg in arch_todo:
        for mode_name in active_modes:
            if f"{cfg['name']}__{mode_name}" not in state["completed"]:
                todo.append((cfg, mode_name))

    n_done  = len(state["completed"])
    n_total = len(todo) + n_done
    print(f"[INFO] {len(todo)} tests à faire (archs×modes), {n_done} déjà complétés.", flush=True)

    for i, (cfg, mode_name) in enumerate(todo, start=1):
        name      = cfg["name"]
        state_key = f"{name}__{mode_name}"
        mode_cfg  = _PRECISION_MODES[mode_name]

        idx_str = f"[{n_done + i}/{n_total}] {name} [{mode_name.upper()}]  (tier {cfg['tier']})"
        dashes  = "─" * max(0, 72 - len(idx_str) - 14)
        print(f"\n{_CW}[Benchmark]{_C0} {_CD}────{_C0} {idx_str} {dashes}", flush=True)
        print(f"  lq={cfg['lq_size']}  batch={cfg['batch']}  accum={cfg['accum']}  "
              f"cl={cfg['channels_last']}  amp={mode_cfg['use_amp']}  "
              f"bf16={mode_cfg['amp_bf16']}  fm={mode_cfg['fast_matmul']}", flush=True)

        # Upscale seulement sur bf16 (ou normal si bf16 non actif) pour économiser du temps
        do_up = (not args.no_upscale) and (
            mode_name == "bf16" or
            (mode_name == "normal" and "bf16" not in active_modes)
        )

        r = run_arch_test(
            cfg=cfg, mode_name=mode_name,
            n_iter=args.n_iter, timeout=args.timeout,
            train_gt=args.train_gt, val_gt=args.val_gt, val_lq=args.val_lq,
            do_upscale=do_up, test_img=args.test_img, output_dir=out_dir,
        )

        state["completed"][state_key] = True
        existing = next(
            (x for x in state["results"]
             if x["name"] == name and x.get("mode") == mode_name),
            None,
        )
        if existing:
            existing.update(r)
        else:
            state["results"].append(r)

        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        _update_coffre(r, state["results"], coffre)

        itps  = f"{r['avg_itps']:.3f} it/s" if r.get("avg_itps") else "—"
        psnrs = list(r.get("psnr_readings", {}).values())
        psnr  = f"PSNR={psnrs[-1]:.2f}" if psnrs else ""
        ups   = r.get("upscale_test")
        up_s  = ups.get("status", "—") if ups else "—"
        up_str = f"  upscale={_CG}{up_s}{_C0}" if up_s == "ok" else (f"  upscale={_CR}{up_s}{_C0}" if ups else "")
        _st_color = _CG if r["status"] == "ok" else _CR
        print(f"{_CW}[DONE]{_C0} {name}/{mode_name}: {_st_color}{r['status']}{_C0}  {itps}  {psnr}{up_str}", flush=True)

    # ── Final report ──────────────────────────────────────────────────────────
    all_results = state["results"]
    if all_results:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt = _format_report(all_results, gpu, args.n_iter, active_modes)
        rpt = out_dir / f"redux_bench_{ts}.txt"
        rpt.write_text(txt, encoding="utf-8")
        js  = out_dir / f"redux_bench_{ts}.json"
        js.write_text(
            json.dumps({"gpu": gpu, "timestamp": ts,
                        "modes": active_modes, "results": all_results},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[RAPPORT] {rpt}", flush=True)
        print(f"[JSON]    {js}",    flush=True)
        print(txt, flush=True)

    # ── Auto-chain vers feature bench ─────────────────────────────────────────
    feat_script = _THIS_DIR / "redux_feature_benchmark.py"
    if not feat_script.exists():
        print(f"[AUTO-CHAIN] Script introuvable: {feat_script} — skipping.", flush=True)
    else:
        ok_results = [r for r in all_results if r.get("status") == "ok" and r.get("avg_itps")]
        if not ok_results:
            print("[AUTO-CHAIN] Aucun résultat OK — feature bench non lancé.", flush=True)
        else:
            # Préférer BF16 pour le classement de vitesse ; fallback sur tous modes
            bf16_ok      = [r for r in ok_results if r.get("mode") == "bf16"]
            ref_results  = bf16_ok if bf16_ok else ok_results
            fastest      = max(ref_results, key=lambda r: r["avg_itps"])
            fastest_name = fastest["name"]
            fastest_cfg  = _ARCH_BY_NAME.get(fastest_name, {})
            fastest_cl   = str(fastest_cfg.get("channels_last", True)).lower()
            fastest_lq   = str(fastest_cfg.get("lq_size", 96))
            fastest_bat  = str(fastest_cfg.get("batch", 8))
            ref_mode     = fastest.get("mode", "?")

            print(f"\n{'='*65}", flush=True)
            print(f"[AUTO-CHAIN] Arch la plus rapide ({ref_mode.upper()}): {fastest_name}  "
                  f"({fastest['avg_itps']:.3f} it/s)", flush=True)
            print(f"[AUTO-CHAIN] → feature bench  arch={fastest_name}  "
                  f"cl={fastest_cl}  lq={fastest_lq}  bat={fastest_bat}", flush=True)

            feat_cmd = [
                sys.executable, str(feat_script),
                "--base-arch",          fastest_name,
                "--base-channels-last", fastest_cl,
                "--base-lq-size",       fastest_lq,
                "--base-batch",         fastest_bat,
                "--train-gt",           args.train_gt,
                "--val-gt",             args.val_gt,
                "--val-lq",             args.val_lq,
            ]
            # Feature bench : BF16 par défaut ; --no-amp si bf16 absent des modes testés
            if "bf16" not in active_modes:
                feat_cmd.append("--no-amp")
            if args.coffre_path:
                feat_cmd += ["--coffre-path", args.coffre_path]

            print(f"[AUTO-CHAIN] CMD: {' '.join(feat_cmd)}", flush=True)
            subprocess.run(feat_cmd, cwd=str(REDUX_PATH))


if __name__ == "__main__":
    main()
