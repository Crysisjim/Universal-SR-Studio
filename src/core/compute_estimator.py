"""
compute_estimator.py — Estimate training duration based on hardware + config.

Uses empirical baselines for common GPU/architecture combos to predict
the total training time. Helps avoid surprises like "this will take 6 days".
"""
import subprocess


# Empirical baselines: it/s for various (GPU, arch, batch, patch, scale)
# These are rough averages — better than nothing but not exact.
# Sources: NeoSR community benchmarks, user reports.
BASELINE_ITPS = {
    # GPU model -> base it/s for OmniSR @ batch=4, patch=64, scale=4
    "RTX 4090": 0.85,
    "RTX 4080": 0.62,
    "RTX 4070 Ti Super": 0.48,
    "RTX 4070 Ti": 0.45,
    "RTX 4070 Super": 0.42,
    "RTX 4070": 0.38,
    "RTX 3090": 0.55,
    "RTX 3080 Ti": 0.48,
    "RTX 3080": 0.42,
    "RTX 3070 Ti": 0.34,
    "RTX 3070": 0.30,
    "RTX 3060 Ti": 0.26,
    "RTX 3060": 0.22,
    "RTX 2080 Ti": 0.30,
    "RTX 2080": 0.25,
    "GTX 1080 Ti": 0.18,
    "GTX 1080": 0.14,
    "GTX 1070 Ti": 0.12,
    "GTX 1070": 0.11,
    # Laptop GPUs (mobile TDP — normal mode slower, BF16 comparable)
    "RTX 3070 Ti Laptop GPU": 0.28,
    "RTX 3060 Laptop GPU": 0.20,
    "RTX 3080 Laptop GPU": 0.35,
    "RTX 4060 Laptop GPU": 0.28,
    "RTX 4070 Laptop GPU": 0.34,
    "RTX 4080 Laptop GPU": 0.45,
}

# Architecture multiplier (relative to OmniSR which is the baseline)
# Higher value = faster training (more it/s).
ARCH_MULTIPLIER = {
    "omnisr_net": 1.0,
    "omnisr": 1.0,
    "rcan": 0.7,           # Slower (more params)
    "swinir_s": 0.55,
    "swinir_m": 0.40,
    "swinir_l": 0.25,
    "swinir": 0.40,
    "esrgan": 1.2,
    "rrdbnet": 1.2,
    "msrresnet": 1.5,
    "edsr": 0.9,
    "span": 16.0,          # SPAN family — measured ~16x faster than OmniSR baseline on same GPU
    "spanplus": 16.0,
    "spannet": 16.0,
    "compactnet": 1.8,
    "compact": 1.8,
    "realesrgancompact": 1.8,
    "hat": 0.30,
    "hat_s": 0.45,
    "hat_l": 0.20,
    "drct": 0.20,
    "drct_l": 0.20,
    "atd": 0.25,
    "plksr": 1.1,
    "realplksr": 1.3,
    "dat": 0.30,
    "dat_2": 0.28,
    # ESC: measured 1.05 it/s on GTX 1080 Ti @ batch=4, patch=96, scale=1
    # Normalized to baseline (batch=4, patch=64, scale=4): 1.05 / (64/96 * 4^0.25 * 0.18) ≈ 6.2
    "esc": 6.2,
    "esc_light": 8.5,      # lighter variant estimate
    "esc_large": 4.0,      # heavier variant estimate
    # --- Redux bench measurements (2026-05-19, GTX 1080 Ti, scale=1 normalized) ---
    # Measured at scale=1 batch=4 patch=96. Scale=4 multipliers are estimates.
    "rtmosr_l": 1.8,       # 9.45 it/s — same class as compact
    "rtmosr": 1.8,
    "rtmosr_ul": 2.0,      # ultra-light variant, estimated
    "mosr": 1.5,           # heavier than mosr_t
    "mosr_t": 1.8,         # 9.43 it/s
    "mosrv2": 1.8,         # 9.37 it/s
    "eimn": 1.25,
    "eimn_a": 1.8,         # 9.42 it/s
    "eimn_l": 1.5,
    "span_s": 18.0,        # span family — fast like span but smaller
    "spanplus_s": 18.0,
    "spanplus_st": 18.0,
    "spanplus_sts": 18.0,
    "artcnn_r16f96": 1.85, # 9.41 it/s
    "artcnn_r8f64": 1.77,  # 9.09 it/s
    "artcnn_r8f48": 1.70,
    "artcnn_r3f24": 1.50,
    "plksr_tiny": 1.80,    # 9.39 it/s
    "ditn_real": 1.80,     # 9.36 it/s (high VRAM though: ~4 GB SMI)
    "ditn": 1.20,
    "esrgan_lite": 1.82,   # 9.35 it/s
    "realplksr_tiny": 1.80, # 9.34 it/s
    "realplksr_large": 1.0,
    "lkfmixer_t": 1.80,    # 9.34 it/s
    "lkfmixer_b": 1.40,
    "lkfmixer_l": 1.20,
    "lmlt": 1.50,
    "lmlt_tiny": 1.80,     # 9.32 it/s
    "lmlt_base": 1.50,
    "lmlt_large": 1.20,
    "gaterv3": 1.50,
    "gaterv3_s": 1.80,     # 9.25 it/s
    "man": 0.16,           # 1.20 it/s — very slow, heavy model
    "man_tiny": 1.80,      # 9.25 it/s — tiny variant is compact-class speed
    "man_light": 0.55,     # 4.61 it/s
    "safmn": 1.79,         # 9.24 it/s
    "safmn_l": 0.60,       # 4.79 it/s — large variant slow
    "seemore_t": 1.79,     # 9.22 it/s
    "ultracompact": 1.90,  # 9.18 it/s — compact family
    "superultracompact": 2.0, # 8.34 it/s — lightest compact
    "sebica": 1.70,        # 8.69 it/s
    "sebica_mini": 1.90,
    "elan": 0.70,
    "elan_light": 0.80,    # 5.64 it/s
    "drct_s": 0.22,        # lighter than drct but still transformer-heavy
    "drct_xl": 0.15,
    "swin2sr_s": 0.55,
    "swin2sr_m": 0.40,
    "swin2sr_l": 0.25,
    "flexnet": 0.0,        # CRASHES on Pascal + PyTorch 2.7 — incompatible
    "srformer_light": 0.0, # idem
    "dat_s": 0.28,
    "dat_light": 0.50,
    "hat_m": 0.28,
    "omnisr": 1.0,         # baseline ref — measured 7.62 it/s at scale=1
}

# Base VRAM (GB) at batch=4, patch=64 — used by estimate_vram()
# Derived from neosr community reports. Scale: vram ~ base * (batch/4) * (patch/64)^1.5 * (0.75 if amp)
ARCH_BASE_VRAM = {
    "esc": 3.8,            # measured 6.97 GB @ patch=96; back-calc: 6.97/(96/64)^1.5 ≈ 3.8
    "esc_light": 2.8,
    "esc_large": 5.5,
    "omnisr_net": 3.5,
    "omnisr": 3.5,
    "span": 2.5,
    "spanplus": 2.5,
    "spannet": 2.5,
    "plksr": 3.0,
    "realplksr": 3.2,
    "swinir_s": 4.5,
    "swinir_m": 6.0,
    "swinir_l": 9.0,
    "swinir": 6.0,
    "esrgan": 3.0,
    "rrdbnet": 3.0,
    "msrresnet": 2.2,
    "edsr": 4.0,
    "compact": 1.5,
    "compactnet": 1.5,
    "realesrgancompact": 1.5,
    "hat_s": 5.0,
    "hat": 7.5,
    "hat_l": 11.0,
    "atd": 6.5,
    "dat": 5.5,
    "dat_2": 5.0,
    "drct": 3.0,           # measured 3.32 GB SMI at patch=96 → ~1.81 GB base, ×1.5 scale=4 → ~3.0
    "drct_l": 6.0,
    "drct_xl": 8.0,
    "drct_s": 4.0,
    "rcan": 4.5,           # base standard (n_resgroups=10, n_resblocks=20, n_feats=64)
    "rcan_l": 4.8,         # n_feats=96 → légèrement plus lourd que rcan standard
    "rcan_unshuffle": 4.6, # variante pixel-unshuffle

    # Redux bench 2026-05-19 (measured SMI at patch=96 → back-calc to patch=64 × scale-factor ~1.5)
    "rtmosr_l": 1.2,
    "rtmosr": 1.8,
    "rtmosr_ul": 0.8,
    "mosr_t": 1.6,
    "mosr": 2.5,
    "mosrv2": 1.3,
    "eimn_a": 1.6,
    "eimn": 2.5,
    "span_s": 2.0,
    "spanplus_s": 1.5,
    "artcnn_r16f96": 1.8,
    "artcnn_r8f64": 1.4,
    "artcnn_r8f48": 1.2,
    "artcnn_r3f24": 1.0,
    "plksr_tiny": 1.8,
    "ditn_real": 4.0,      # VRAM élevée: 4.13 GB SMI à scale=1 → prudence
    "ditn": 5.0,
    "esrgan_lite": 1.3,
    "realplksr_tiny": 2.0,
    "realplksr_large": 5.0,
    "lkfmixer_t": 2.0,
    "lkfmixer_b": 3.5,
    "lkfmixer_l": 5.0,
    "lmlt_tiny": 1.8,
    "lmlt": 3.0,
    "gaterv3_s": 2.2,
    "gaterv3": 3.5,
    "man": 8.0,            # 10.59 GB SMI à scale=1 — très lourd !
    "man_tiny": 1.6,
    "man_light": 2.5,
    "safmn": 1.8,
    "safmn_l": 4.5,        # 5.47 GB SMI à scale=1
    "seemore_t": 2.2,
    "ultracompact": 1.5,
    "superultracompact": 1.2,
    "sebica": 1.3,
    "sebica_mini": 1.0,
    "elan": 4.0,
    "elan_light": 3.5,     # 4.07 GB SMI à scale=1
    "swinir_s": 1.5,       # 1.76 GB SMI à scale=1 (beaucoup moins que swinir_m)
    "swin2sr_s": 2.0,
    "swin2sr_m": 5.0,
}


# BF16 Tensor Core boost per arch.
# Measured from traiNNer-redux bench (RTX 3070 Ti Laptop, batch=4, patch=96)
# and NeoSR bench (RTX 3070 Ti Laptop, batch=4, scale=1).
# Conservative — use traiNNer-redux values as lower bound.
# Only applied when GPU has native BF16 Tensor Cores (Ampere RTX 30xx+).
TENSOR_CORE_BF16_BOOST = {
    "plksr":          1.21,  # Measured +21% traiNNer-redux BF16
    "plksr_tiny":     1.15,  # Estimated — same family
    "realplksr":      1.20,  # NeoSR: +22%
    "realplksr_tiny": 1.15,
    "esc":            1.35,  # NeoSR: +51% — large matrix ops
    "esc_light":      1.25,
    "swinir_small":   1.15,  # NeoSR swinir_small: +22%
    "swinir_s":       1.15,
    "swinir_m":       1.10,
    "safmn_l":        1.13,  # Measured +13% traiNNer-redux BF16
    "ditn":           1.10,
    "ditn_real":      1.10,
    "span":           1.05,  # NeoSR: +9%
    "span_s":         1.05,
    "spanplus":       1.05,
    "spanplus_s":     1.05,
    "man":            1.10,  # FP16/BF16 required (Normal OOM) — estimate
    "man_light":      1.08,
    "man_tiny":       1.03,
    "rcan":           1.05,
    "omnisr":         1.03,  # NeoSR: minimal gain at batch=4
    # Most CNN-light archs: little or no boost at batch=4/patch<=128
    # compact / safmn / span / etc: ~1.0
}


def has_bf16_tensor_cores(gpu_name: str) -> bool:
    """True if GPU has native BF16 Tensor Cores — Ampere (RTX 30xx) or newer."""
    bf16_patterns = [
        "RTX 30", "RTX 40", "RTX 50",          # GeForce Ampere / Ada / Blackwell
        "A100", "A10G", "A40", "A30",            # Datacenter Ampere
        "H100", "H200", "H800",                  # Hopper
        "L4", "L40",                              # Ada datacenter
    ]
    gn = gpu_name or ""
    return any(p in gn for p in bf16_patterns)


def lookup_tensor_core_boost(arch_name: str) -> float:
    """Return BF16 Tensor Core speed multiplier for this arch (1.0 = no boost)."""
    if not arch_name:
        return 1.0
    a = str(arch_name).lower().strip()
    if a in TENSOR_CORE_BF16_BOOST:
        return TENSOR_CORE_BF16_BOOST[a]
    for key in sorted(TENSOR_CORE_BF16_BOOST.keys(), key=len, reverse=True):
        if key in a:
            return TENSOR_CORE_BF16_BOOST[key]
    return 1.0  # No boost for unlisted archs


_gpu_cache = {"name": None}


def lookup_arch_vram(arch_name: str) -> float:
    """Base VRAM (GB) at batch=4, patch=64 for given arch. Falls back to 3.5 GB."""
    if not arch_name:
        return 3.5
    a = str(arch_name).lower().strip()
    if a in ARCH_BASE_VRAM:
        return ARCH_BASE_VRAM[a]
    for key in sorted(ARCH_BASE_VRAM.keys(), key=len, reverse=True):
        if key in a:
            return ARCH_BASE_VRAM[key]
    return 3.5


def estimate_vram(
    architecture: str,
    batch_size: int = 4,
    patch_size: int = 64,
    use_amp: bool = False,
) -> float:
    """Estimate peak VRAM (GB). Formula: base * (batch/4) * (patch/64)^1.5 * amp_factor."""
    base = lookup_arch_vram(architecture)
    batch_factor = batch_size / 4.0
    patch_factor = (patch_size / 64.0) ** 1.5
    amp_factor = 0.72 if use_amp else 1.0
    return round(base * batch_factor * patch_factor * amp_factor, 2)


def detect_gpu_name() -> str:
    """Detect the primary GPU name via nvidia-smi (cached after first call)."""
    if _gpu_cache["name"] is not None:
        return _gpu_cache["name"]
    name = "Unknown"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            name = r.stdout.strip().split("\n")[0].replace("NVIDIA GeForce ", "").strip()
    except Exception:
        pass
    # Try torch as fallback
    if name == "Unknown":
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "").strip()
        except Exception:
            pass
    _gpu_cache["name"] = name
    return name


def lookup_arch_multiplier(arch_name: str) -> float:
    """Find architecture multiplier with fuzzy matching (case insensitive, partial)."""
    if not arch_name:
        return 1.0
    a = str(arch_name).lower().strip()
    # Direct match
    if a in ARCH_MULTIPLIER:
        return ARCH_MULTIPLIER[a]
    # Partial match (longest first to prioritize specific variants)
    for key in sorted(ARCH_MULTIPLIER.keys(), key=len, reverse=True):
        if key in a:
            return ARCH_MULTIPLIER[key]
    return 1.0


def lookup_baseline_itps(gpu_name: str) -> float:
    """Find the baseline it/s for this GPU. Returns 0.15 (conservative) if unknown."""
    if not gpu_name or gpu_name == "Unknown":
        return 0.15
    # Direct match
    for key, val in BASELINE_ITPS.items():
        if key in gpu_name:
            return val
    return 0.15  # Fallback


def estimate_training_time(
    gpu_name: str = None,
    architecture: str = "omnisr_net",
    batch_size: int = 4,
    patch_size: int = 64,
    scale: int = 4,
    accumulate: int = 1,
    total_iter: int = 100000,
    use_amp: bool = False,
    amp_bf16: bool = False,
) -> dict:
    """
    Estimate the total training duration.

    use_amp    : AMP enabled (fp16 or bf16)
    amp_bf16   : True = BF16 mode (enables Tensor Core boost on Ampere+)

    Returns dict with: estimated_seconds, eta_str, itps, confidence (low/medium/high)
    """
    if gpu_name is None:
        gpu_name = detect_gpu_name()

    base_itps = lookup_baseline_itps(gpu_name)
    arch_mult = lookup_arch_multiplier(architecture)

    # Scaling factors:
    # - Larger batch -> slower it/s (~1/batch_ratio)
    # - Larger patch -> much slower (quadratic in patch size)
    # - Higher scale -> bigger output, slower
    # - Accumulate -> linear slowdown
    batch_factor = 4.0 / max(1, batch_size)  # baseline batch=4
    # Patch scaling: linear — training is memory-bandwidth bound, not FLOPs bound
    patch_factor = 64.0 / max(8, patch_size)
    # Scale effect is mild once patch size is fixed; exponent 0.25 avoids large swings
    scale_factor = (4.0 / max(1, scale)) ** 0.25
    accum_factor = 1.0 / max(1, accumulate)  # Effective it/s decreases with accumulate

    # BF16 Tensor Core boost — only on Ampere+ GPUs with BF16 AMP active
    tc_boost = 1.0
    if use_amp and amp_bf16 and has_bf16_tensor_cores(gpu_name):
        tc_boost = lookup_tensor_core_boost(architecture)

    estimated_itps = (
        base_itps * arch_mult * batch_factor * patch_factor
        * scale_factor * accum_factor * tc_boost
    )
    estimated_itps = max(0.001, estimated_itps)

    total_seconds = total_iter / estimated_itps

    # Format
    days = int(total_seconds // 86400)
    hours = int((total_seconds % 86400) // 3600)
    minutes = int((total_seconds % 3600) // 60)
    if days > 0:
        eta_str = f"{days}j {hours}h {minutes}min"
    elif hours > 0:
        eta_str = f"{hours}h {minutes}min"
    else:
        eta_str = f"{minutes}min"

    # Confidence: high if GPU is in our table, low otherwise
    confidence = "high" if gpu_name in str(BASELINE_ITPS) else "medium"
    if base_itps == 0.15:  # Fallback was used
        confidence = "low"

    vram_gb = estimate_vram(architecture, batch_size, patch_size, use_amp=use_amp)

    return {
        "estimated_seconds": total_seconds,
        "eta_str": eta_str,
        "itps": estimated_itps,
        "confidence": confidence,
        "gpu": gpu_name,
        "base_itps": base_itps,
        "vram_gb": vram_gb,
        "tc_boost": tc_boost,
        "amp_mode": ("bf16" if (use_amp and amp_bf16) else "fp16" if use_amp else "off"),
    }


def get_pytorch_recommendation(gpu_name: str = None) -> dict:
    """
    Return recommended PyTorch version + install info based on detected GPU.

    Keys returned:
      gpu_name, gpu_gen, has_tensor_cores,
      torch_version, torchvision_version, cuda_tag, whl_url,
      install_pkgs  (list of "pkg==ver" strings to pip install),
      features      (list of available training features),
      limitations   (list of known limitations for this GPU class),
      note          (short UI-friendly string)
    """
    if gpu_name is None:
        gpu_name = detect_gpu_name()

    gn = gpu_name or ""

    # --- Ada Lovelace / Hopper / Blackwell (RTX 40xx, RTX 50xx, H100…) ---
    if any(p in gn for p in ["RTX 40", "RTX 50", "H100", "H200", "H800", "L4", "L40"]):
        gen = "Ada Lovelace / Hopper"
        has_tc = True
        tv = "2.7.0"; vv = "0.22.0"; ctag = "cu126"; min_tv = "2.5.0"
        features = ["BF16 Tensor Cores (natif)", "FP16 AMP", "TF32 (fast_matmul)", "FP8 (Ada)"]
        limits = ["torch.compile requiert WSL2 sur Windows"]
        note = "GPU moderne — PyTorch 2.7 + CUDA 12.6 recommande"
        upgrade_reason = "Mettre a jour pour BF16 / TF32 / FP8 optimaux"

    # --- Ampere (RTX 30xx, A100, A40…) ---
    elif any(p in gn for p in ["RTX 30", "A100", "A10G", "A40", "A30"]):
        gen = "Ampere"
        has_tc = True
        tv = "2.7.0"; vv = "0.22.0"; ctag = "cu126"; min_tv = "2.5.0"
        features = ["BF16 Tensor Cores (natif)", "FP16 AMP", "TF32 (fast_matmul)"]
        limits = [
            "torch.compile requiert WSL2 sur Windows",
            "fast_matmul (TF32) : verifier que traiNNer-redux utilise API PyTorch ≥2.7",
        ]
        note = "GPU Ampere — PyTorch 2.7 + CUDA 12.6 recommande"
        upgrade_reason = "Mettre a jour pour BF16 / TF32 optimaux"

    # --- Turing (RTX 20xx) ---
    elif any(p in gn for p in ["RTX 20"]):
        gen = "Turing"
        has_tc = True
        tv = "2.6.0"; vv = "0.21.0"; ctag = "cu124"; min_tv = "2.3.0"
        features = ["FP16 AMP", "Tensor Cores (FP16/INT8)"]
        limits = [
            "BF16 : support partiel sur Turing (pas natif)",
            "TF32 non supporte sur Turing",
            "torch.compile requiert WSL2 sur Windows",
        ]
        note = "GPU Turing — PyTorch 2.6 + CUDA 12.4 recommande"
        upgrade_reason = "Mettre a jour pour FP16 AMP stable"

    # --- Pascal / Maxwell / Kepler (GTX 10xx, GTX 9xx, GTX 7xx, GT…) ---
    else:
        gen = "Pascal / Maxwell / Kepler"
        has_tc = False
        # Recommend latest stable — works fine on Pascal, just without Tensor Core features.
        # Do NOT recommend downgrading — any PyTorch ≥2.3 is fine.
        tv = "2.7.0"; vv = "0.22.0"; ctag = "cu126"; min_tv = "2.3.0"
        features = ["FP32 standard", "FP16 AMP (partiel)"]
        limits = [
            "Pas de Tensor Cores → BF16 et TF32 non disponibles",
            "Ne pas activer : use_amp BF16, fast_matmul, tf32",
            "torch.compile non disponible",
            "AMP FP16 moins stable que sur Ampere+",
        ]
        note = "GPU sans Tensor Cores — toute version PyTorch ≥2.3 convient"
        upgrade_reason = "Mettre a jour vers la derniere version stable"

    whl_url = f"https://download.pytorch.org/whl/{ctag}"
    return {
        "gpu_name":             gpu_name,
        "gpu_gen":              gen,
        "has_tensor_cores":     has_tc,
        "torch_version":        tv,          # recommended target version
        "torchvision_version":  vv,
        "cuda_tag":             ctag,
        "min_torch_version":    min_tv,      # minimum acceptable version
        "whl_url":              whl_url,
        "install_pkgs":         [f"torch=={tv}", f"torchvision=={vv}"],
        "features":             features,
        "limitations":          limits,
        "note":                 note,
        "upgrade_reason":       upgrade_reason,
    }


def format_estimation_message(est: dict) -> str:
    """Format an estimation dict as a human-readable message."""
    confidence_emoji = {"high": "✅", "medium": "🟡", "low": "🟠"}
    emoji = confidence_emoji.get(est["confidence"], "❓")
    vram_str = f"\nVRAM estimee : ~{est['vram_gb']:.1f} GB" if "vram_gb" in est else ""
    # Show Tensor Core boost if active
    tc_str = ""
    tc_boost = est.get("tc_boost", 1.0)
    amp_mode = est.get("amp_mode", "off")
    if amp_mode != "off":
        if tc_boost > 1.01:
            tc_str = f"\nAMP {amp_mode.upper()} : Tensor Cores actifs (+{(tc_boost-1)*100:.0f}%)"
        else:
            tc_str = f"\nAMP {amp_mode.upper()} : actif"
    return (
        f"GPU detecte : {est['gpu']}\n"
        f"Vitesse estimee : {est['itps']:.3f} it/s\n"
        f"Duree estimee : {est['eta_str']}"
        f"{vram_str}"
        f"{tc_str}\n"
        f"Confiance : {emoji} {est['confidence']}"
    )
