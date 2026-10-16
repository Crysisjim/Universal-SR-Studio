"""
compute_estimator.py — Estimate training duration based on hardware + config.

Uses empirical baselines for common GPU/architecture combos to predict
the total training time. Helps avoid surprises like "this will take 6 days".
"""
import subprocess


def _t(fr: str, en: str) -> str:
    # UI QT (PySide6) : la langue vit dans qt_ui.i18n.
    try:
        from qt_ui.i18n import get_lang
        if get_lang() == "en":
            return en
    except Exception:
        pass
    # Repli : ancien translator CTk.
    try:
        from src.core.translations import get_translator
        tr = get_translator()
        if tr and getattr(tr, 'language', 'fr') == 'en':
            return en
    except Exception:
        pass
    return fr


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
    # WSR / DUAL (Rift73): real 150-it runs 2026-11-10 (measured_vram.py drives the
    # estimate); these fallback multipliers are fitted on the same runs (ratio real/est).
    "wsr_mobile": 28.0, "wsr_superultracompact": 34.0, "wsr_ultracompact": 30.0,
    "wsr_compact": 24.0, "wsr": 21.0,
    "dual_light": 4.2, "dual_xs": 1.8, "dual_s": 0.78, "dual_m": 0.75,
    "dual_l": 0.44, "dual_xl": 0.34,
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
    # ── Sprint 19/20 — Nouvelles architectures ──────────────────────────────────
    # CATANet (dim=40, NeoSR): measured 0.528 it/s @ GTX 1080 Ti, batch=4, patch=96, scale=1
    #   normalized: 0.528 / (0.18 * (64/96) * (4/1)^0.25) = 0.528 / 0.170 ≈ 3.1
    "catanet": 3.1,
    # SMoSR: measured 1.012 it/s @ GTX 1080 Ti, batch=8, patch=128, scale=1 (dim=72)
    # normalized: 1.012 / (0.18 * 0.5 * 0.5 * 1.414) = 15.9 → 16.0
    "smosr": 16.0,
    # SpanF: SPAN simplifié fc=32 — très léger, estimé légèrement + rapide que SPAN
    "spanf": 18.0,
    # SpanC / SpanPP: multi-scale IGConv, légèrement + lourd que SPAN
    "spanc": 14.0,
    "spanpp": 14.0,
    # GFISRv2: GatedCNN + FFT multi-upsampler, classe compact/intermédiaire
    "gfisrv2": 8.0,
    # FIGSR: Fourier-Inception-Gated, même famille que gfisrv2 (~même coût)
    "figsr": 8.0,
    # ── ParagonSR v1 (Phhofm) — conv-first, Magic Kernel Sharp upsampler ──────
    # Measured: paragonsr_m (72) @ 1x, batch=4, patch=96, GTX 1080 Ti → 11 it/s, 0.62 GB VRAM
    # Speed: 11 / (0.18 * (64/96) * (4/1)^0.25) = 64.7 → ~65.0
    # VRAM base: 0.62 / (96/64)^1.5 = 0.337 GB at batch=4, patch=64
    "paragonsr_nano":  75.0,  # num_feat=24, minimal
    "paragonsr_tiny":  70.0,  # num_feat=40
    "paragonsr_xs":    68.0,  # num_feat=48
    "paragonsr_s":     66.0,  # num_feat=64
    "paragonsr_m":     65.0,  # num_feat=72, measured ~11 it/s @ 1x
    "paragonsr_base":  52.0,  # num_feat=88, estimated ~9 it/s @ 1x
    "paragonsr_large": 36.0,  # num_feat=128
    "paragonsr_xl":    22.0,  # num_feat=192
    "paragonsr_xxl":   12.0,  # num_feat=256
    "paragonsr":       65.0,  # generic fallback
    # ── ParagonSR v2 (Phhofm) — similar class, slightly heavier ────────────
    "paragonsr2_nano":  65.0,
    "paragonsr2_tiny":  58.0,
    "paragonsr2_xs":    52.0,
    "paragonsr2_s":     45.0,
    "paragonsr2_m":     38.0,
    "paragonsr2_base":  30.0,
    "paragonsr2_pro":   22.0,
    "paragonsr2_ultimate": 15.0,
    "paragonsr2_ultimate_v2": 12.0,
    "paragonsr2":       38.0,  # generic fallback
    # ── Real-training calibration (2026-10-26), GTX 1080 Ti idle, scale 2, batch 4, FP32,
    # real OTF data, losses charbonnier+LDL+color+FocalFreq+PerceptualAnime, no GAN.
    # Short runs (200 it) under-read the steady speed: fdat_rd_medium read 1.226 it/s vs
    # 1.64 it/s in a full run, so every short-run value is scaled by 1.64/1.226 = 1.34.
    # Multiplier = real it/s / (0.18 * 64/patch * (4/2)^0.25).
    # Caveat: with these losses the fixed cost (OTF degradations + perceptual loss)
    # dominates: fdat_rd_medium ran at 1.26 it/s at patch 64 vs 1.23 at patch 96, so the
    # linear patch factor above is optimistic for small patches.
    # (Previous HEART/NERVE/WireSR values were isolated fwd+bwd bench ratios and
    # underestimated real speed 8x to 30x.)
    "nerve": 82.9,           # patch 96: 8.845 it/s short run -> 11.8 it/s
    "heart": 2.94,           # patch 64 (96 does not fit 11 GB): 0.470 -> 0.63 it/s
    "wiresr": 11.7,          # patch 96, scale 2: 1.242 -> 1.66 it/s
    "fdat_rd_small": 14.6,   # patch 96, unshuffle: 1.555 -> 2.08 it/s
    "fdat_rd_medium": 11.5,  # patch 96, unshuffle: full run 1.64 it/s (1.39 with UNet GAN)
    "fdat_rd_large": 4.6,    # patch 96, unshuffle: 0.489 -> 0.65 it/s
    # MoTSR: real training 2026-10-29 (sequence dataset, 5-frame clips, LQ 64, batch 4,
    # charbonnier + FocalFreq + color, no GAN): 10.6 it/s, 0.8 GB. The old value (1.6,
    # isolated bench ratio vs OmniSR) predicted 0.36 it/s, 30x too slow.
    "motsr": 40.2,
    # ── AetherNet (Phhofm) — NeoSR, structural reparameterization ──────────
    # Estimates only — no direct bench measurement yet.
    "aether_mobile":  60.0,   # 64ch, 2×4 blocks — ultra-light
    "aether_tiny":    50.0,   # 64ch, 3×3 blocks
    "aether_small":   35.0,   # 96ch, 4×4 blocks
    "aether_medium":  22.0,   # 128ch, 4×4 + channel attn
    "aether_large":   12.0,   # 180ch, 5×5 + spatial+channel attn
    "aether_pro":      8.0,   # 192ch, lk_kernel=15
    "aether_extreme":  3.5,   # 256ch, 5×12 + lk_kernel=21
    # ── MambaIR (ECCV2024, csguoh/MambaIR port) — real CUDA kernel (mamba_ssm) ──
    # Measured: embed_dim=96, depths=[6,6,6,6], d_state=16, mlp_ratio=2, 1x,
    # batch=2, patch=80, accumulate=1, GTX 1080 Ti → 0.842 it/s (real, from
    # train.py's own log line, not GPU-monitor sidebar).
    # Speed: 0.842 / (0.18 * (4/2) * (64/80) * (4/1)^0.25) = 0.842/0.4073 ≈ 2.07
    # Caveat: this uses the estimator's existing linear patch_factor (64/patch);
    # MambaIR's scan cost scales with sequence length L=H*W (closer to quadratic
    # in patch side), so predictions at patch sizes far from 80 may be optimistic.
    # ── MambaIRv2 (CVPR2025) — real training run, USRS "test mambairV2 v1.2" ──
    # embed_dim=96, depths=[5,5,5,5], num_heads=[4,4,4,4], d_state=8, window=16,
    # inner_rank=32, num_tokens=64, mlp_ratio=1.5, upsampler=pixelshuffledirect,
    # 1x, batch=2, patch=80, accumulate=1, GTX 1080 Ti — SAME config as mambair's
    # calibration point above, so directly comparable.
    # First segment (iter 41165→45000, concurrent x265 encode running in
    # background — see feedback_gpu_contention memory) measured only 0.843-1.15
    # it/s (contended). After the x265 encode stopped, iter 45000→113512 settled
    # into a stable 1.27-1.32 it/s band (contention-free) — used for calibration:
    # Speed: 1.30 / (0.18 * (2/2) * (80/64→64/80=0.8) * 1^0.25) = 1.30/0.144 ≈ 9.03
    "mambair": 2.07,
    "mambairv2": 9.03,
}

# Base VRAM (GB) at batch=4, patch=64 — used by estimate_vram()
# Derived from neosr community reports. Scale: vram ~ base * (batch/4) * (patch/64)^1.5 * (0.75 if amp)
ARCH_BASE_VRAM = {
    # WSR / DUAL (Rift73) — provisional (base 1 = 4.2 GB at b6 LQ128 2x). DUAL measured
    # generator-only: light b6 LQ128 9.5 GB; xs 2x light at b2 LQ64 (b6 LQ128 ran out of
    # memory). WSR not measured yet.
    "wsr_mobile": 0.4, "wsr_superultracompact": 0.5, "wsr_ultracompact": 0.7,
    "wsr_compact": 0.9, "wsr": 1.2,
    "dual_light": 2.5, "dual_xs": 5.0, "dual_s": 6.0, "dual_m": 7.2, "dual_l": 9.0, "dual_xl": 11.0,
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
    # ── Sprint 19/20 ─────────────────────────────────────────────────────────────
    # Calibration : base = training_vram / ((batch/4) * (patch/64)^1.5)
    # training_vram = torch.cuda.memory_reserved() (log neosr/traiNNer)
    # CATANet (dim=40): log neosr [ VRAM: 4.67 GB ] @ batch=4, patch=96
    #   base = 4.67 / (1 * (96/64)^1.5) = 4.67 / 1.837 ≈ 2.54
    "catanet": 2.54,
    # SMoSR dim=72: log traiNNer [ VRAM: 9.05 GB ] = torch.cuda.memory_reserved()
    #   @ batch=8, patch=128. base = 9.05 / (2 * 2.828) ≈ 1.60
    "smosr": 1.6,
    # SpanF fc=72: log traiNNer [ VRAM: 5.43 GB ] = torch.cuda.memory_reserved()
    #   @ batch=8, patch=128. Widget GPU Windows (2.9 GB) = WDDM sous-compte CUDA.
    #   base = 5.43 / (2 * 2.828) ≈ 0.96
    "spanf": 0.96,
    # SpanC / SpanPP : multi-scale IGConv — estimé ~2× spanf
    "spanc": 1.9,
    "spanpp": 1.9,
    # GFISRv2 : GatedCNN + FFT — estimé, classe compact+
    "gfisrv2": 2.2,
    # FIGSR : Fourier-Inception-Gated — estimé, ~même que gfisrv2
    "figsr": 2.2,
    # ── ParagonSR v1 (Phhofm) — base VRAM batch=4, patch=64 ───────────────
    # Measured: paragonsr_m @ 1x, batch=4, patch=96 → 0.62 GB → base = 0.62/(96/64)^1.5 = 0.34 GB
    # paragonsr_base @ 1x, batch=10, patch=128 → 2.13 GB → base ≈ 0.30 GB (consistent)
    # NOTE: paragonsr is 1x-optimized (no upsampling layers) → much lighter than 4x arches
    "paragonsr_nano":   0.22,
    "paragonsr_tiny":   0.26,
    "paragonsr_xs":     0.30,
    "paragonsr_s":      0.32,
    "paragonsr_m":      0.34,  # measured at 1x batch=4 patch=96 → base=0.34 GB
    "paragonsr_base":   0.42,  # num_feat=88, slightly heavier
    "paragonsr_large":  0.65,  # num_feat=128
    "paragonsr_xl":     1.2,   # num_feat=192
    "paragonsr_xxl":    2.2,   # num_feat=256
    "paragonsr":        0.34,  # generic fallback
    # ── ParagonSR v2 ──────────────────────────────────────────────────────
    "paragonsr2_nano":   0.28,
    "paragonsr2_tiny":   0.35,
    "paragonsr2_xs":     0.42,
    "paragonsr2_s":      0.55,
    "paragonsr2_m":      0.70,
    "paragonsr2_base":   0.90,
    "paragonsr2_pro":    1.30,
    "paragonsr2_ultimate": 1.80,
    "paragonsr2_ultimate_v2": 2.20,
    "paragonsr2":        0.70,  # generic fallback
    # ── Real-training peak VRAM (same runs as the speed calibration, 2026-10-26),
    # base = peak / (patch/64)^1.5. These include ~4-5 GB of loss networks + OTF
    # buffers (NERVE, a tiny model, peaks at 5.4 GB), which barely depends on the patch:
    # fdat_rd_medium peaked at 7.84 GB at patch 64 vs 8.03 GB at patch 96.
    "nerve": 2.9,            # 5.41 GB at patch 96
    "heart": 9.5,            # 9.51 GB at patch 64
    "wiresr": 4.4,           # 8.10 GB at patch 96
    "fdat_rd_small": 3.9,    # 7.25 GB at patch 96
    "fdat_rd_medium": 4.4,   # 8.12 GB at patch 96 (full run)
    "fdat_rd_large": 4.7,    # 8.61 GB at patch 96
    "motsr": 1.08,           # 0.8 GB real (see ARCH_MULTIPLIER), no perceptual loss
    # ── AetherNet (Phhofm) — NeoSR, structural reparameterization ────────
    "aether_mobile":   0.35,
    "aether_tiny":     0.40,
    "aether_small":    0.60,
    "aether_medium":   0.90,
    "aether_large":    1.50,
    "aether_pro":      1.80,
    "aether_extreme":  3.00,
    # ── MambaIR (ECCV2024, csguoh/MambaIR port) — real CUDA kernel (mamba_ssm) ──
    # Measured: embed_dim=96, depths=[6,6,6,6], batch=2, patch=80, 1x, GTX 1080 Ti
    # → log traiNNer [ VRAM: 7.83 GB ] (torch.cuda.memory_reserved(), full GAN
    # pipeline: generator + EMA copy + Adam optimizer states + custom OTF hook).
    # base = 7.83 / ((2/4) * (80/64)^1.5) = 7.83 / 0.699 ≈ 11.2 GB — genuinely
    # heavy (Adam optimizer state alone doubles generator memory on top of the
    # scan itself). embed_dim=114/patch=96/batch=4 OOM'd an 11GB card, consistent.
    # ── MambaIRv2 (CVPR2025) — real training run, USRS "test mambairV2 v1.2" ──
    # Same config as mambair's calibration point (batch=2, patch=80, 1x).
    # log traiNNer [ VRAM: 7.30 GB ] steady-state (torch.cuda.memory_reserved()).
    # base = 7.30 / ((2/4) * (80/64)^1.5) = 7.30 / 0.699 ≈ 10.44 GB — lighter
    # than mambair v1 (11.2 GB) despite similar embed_dim, consistent with v2's
    # attentive-scan design trading some compute for less optimizer-state bulk.
    "mambair": 11.2,
    "mambairv2": 10.44,
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


_gpu_cache = {"name": None, "driver": 0.0, "sm": 0}


def detect_gpu_info() -> dict:
    """Query nvidia-smi once for name, driver version and compute capability.

    Returns:
        name        — GPU display name (NVIDIA prefix stripped)
        driver      — driver version as float (e.g. 572.83), 0.0 if unknown
        sm          — compute capability as int (e.g. 120 for 12.0, 89 for 8.9)
        has_nvidia  — True if a NVIDIA GPU was found
    """
    if _gpu_cache["name"] is not None:
        return {
            "name":       _gpu_cache["name"],
            "driver":     _gpu_cache["driver"],
            "sm":         _gpu_cache["sm"],
            "has_nvidia": _gpu_cache["name"] != "Unknown",
        }

    name = "Unknown"
    driver = 0.0
    sm = 0

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split("\n")[0].split(", ")
            if parts:
                name = parts[0].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").strip()
            if len(parts) >= 2:
                try:
                    driver = float(parts[1].strip())
                except ValueError:
                    pass
            if len(parts) >= 3:
                cap = parts[2].strip()          # e.g. "12.0" or "8.9"
                try:
                    major, minor = cap.split(".")
                    sm = int(major) * 10 + int(minor)
                except Exception:
                    pass
    except Exception:
        pass

    # Fallback: torch (if already installed in the host env)
    if name == "Unknown":
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").strip()
                major, minor = torch.cuda.get_device_capability(0)
                sm = major * 10 + minor
        except Exception:
            pass

    _gpu_cache["name"]   = name
    _gpu_cache["driver"] = driver
    _gpu_cache["sm"]     = sm
    return {"name": name, "driver": driver, "sm": sm, "has_nvidia": name != "Unknown"}


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
    losses=None,
    discs=None,
    use_checkpoint: bool = False,
    scale: int = 2,
    unshuffle_mod: bool = True,
) -> float:
    """Estimate peak VRAM (GB), as the training log shows it (torch reserved memory).

    FDAT-RD: measured model (FDAT_RD_GEN). Other archs: base * (batch/4) *
    (patch/64)^1.5. losses / discs (engine type names) then add their measured cost
    (LOSS_COST / DISC_COST) relative to the losses the arch was calibrated with;
    losses=None = the calibration losses."""
    amp_factor = 0.72 if use_amp else 1.0
    bf = batch_size / 4.0
    a = str(architecture).lower()
    hr = max(8, patch_size) * max(1, scale)
    if a in FDAT_RD_GEN:
        full = not unshuffle_mod and scale < 3
        eff = _gen_patch(a, patch_size, scale, unshuffle_mod)
        px = (eff / 60.0) ** 2 if full else (eff / 120.0) ** 2
        gen = FDAT_RD_GEN[a][0] * (bf * px) ** _ACT_EXP
        if use_checkpoint:
            gen *= 1.0 - CHECKPOINT_GEN_VRAM_SAVING
        base = FDAT_RD_FIXED_GB + gen + FDAT_RD_LOSS_ACT_GB * (bf * (hr / 240.0) ** 2) ** _ACT_EXP
        base += VRAM_LOG_OFFSET_GB
        if full:
            base += FDAT_RD_FULLRES_VAL_GB * FDAT_RD_GEN[a][0] / FDAT_RD_GEN["fdat_rd_medium"][0]
    elif a in _MEASURED_VRAM:
        # Real campaign measure (charbonnier + VGG perceptual = the generic calibration
        # losses): fixed part + activations scaled by batch x LQ pixels. Batch 1 vs 2 on
        # escrealm (3.26 / 4.10 GB) is predicted at 3.23 GB from the batch-2 figure.
        gb, b_meas = _MEASURED_VRAM[a]
        act = max(0.05, gb - MEASURED_FIXED_GB)
        base = MEASURED_FIXED_GB + act * ((batch_size / b_meas) * (patch_size / 64.0) ** 2) ** _ACT_EXP
        if use_checkpoint:
            base = MEASURED_FIXED_GB + (base - MEASURED_FIXED_GB) * 0.66
    else:
        base = lookup_arch_vram(architecture) * bf * (patch_size / 64.0) ** 1.5
        if use_checkpoint:
            # Recomputes the generator activations; loss networks and D are not
            # checkpointed (real run: 8.13 -> 5.39 GB, x0.66).
            base *= 0.66
    extra = _losses_cost(a, losses, bf, patch_size, scale)[0]
    extra += _discs_cost(discs, bf, patch_size, scale)[0]
    return round(max(0.3, base + extra) * amp_factor, 2)


# ── Bench 2026-10-29 (GTX 1080 Ti, fdat_rd_medium, batch 4, LQ 120 -> HR 240, FP32,
# cudnn.benchmark, synthetic data, 40 it after a 10 it warm-up). VRAM = steady-state
# reserved memory after the warm-up; the training log also keeps the cuDNN autotune
# cache and the OTF buffers, hence VRAM_LOG_OFFSET_GB, fitted on the real runs:
# calibration losses 8.12 GB (LQ 96), + UNet-SN 8.61 GB (LQ 120), + LineDistiller
# 9.41 GB. Each addition was measured on top of the calibration losses
# (charbonnier + FocalFreq + color + LDL + PerceptualAnime); the costs add up (LD +
# UNet-SN measured 6.59 GB vs 6.50 summed, 1.271 s/it vs 1.287 summed).
VRAM_LOG_OFFSET_GB = 3.6
# Measured archs (measured_vram.py): fixed reserved memory of a run (CUDA context,
# cuDNN workspace, VGG, optimizer) — the smallest archs read 2.2-2.3 GB at batch 2.
MEASURED_FIXED_GB = 2.2
try:
    from src.core.measured_vram import MEASURED_VRAM as _MEASURED_VRAM, MEASURED_ITPS as _MEASURED_ITPS
except ImportError:
    try:
        from measured_vram import MEASURED_VRAM as _MEASURED_VRAM, MEASURED_ITPS as _MEASURED_ITPS
    except ImportError:
        _MEASURED_VRAM, _MEASURED_ITPS = {}, {}
# Measured speeds: fixed time per iteration (data, optimizer, OTF) — the fastest archs
# run at ~22 it/s (0.045 s), so the fixed share is small.
MEASURED_T_FIX = 0.03
# Real-CUGAN fast: pixel-unshuffle on the input, the UNets run at half the LQ size, so
# about 1/4 of the generator work of the measured (non-fast) run. Real run 2026-11-10
# (GTX 1080 Ti, batch 8, LQ 128, 2x): 4.1 it/s PSNR phase, 1.32 it/s GAN phase.
REALCUGAN_FAST_WORK = 0.25
_ACT_EXP = 0.88  # activations grow a bit less than linearly with batch x pixels
# FDAT-RD: generator activations (GB at batch 4, padded patch 120) and bench time
# per iteration of the generator (s). medium measured; small/large fitted on their
# real runs (7.25 / 8.61 GB and 2.08 / 0.65 it/s at LQ 96).
FDAT_RD_GEN = {
    "fdat_rd_small": (2.85, 0.228),
    "fdat_rd_medium": (3.67, 0.385),
    "fdat_rd_large": (4.20, 1.514),
}
# Without pixel-unshuffle (unshuffle_mod: false, the engine default) the body runs at the
# full LQ resolution, padded to a multiple of 30: LQ 64 -> 90, LQ 96 -> 120. The
# generator figures above then scale with (body / 60)^2 (60 = body of the unshuffle
# bench point). Time factor fitted on the bench (G only, batch 4, checkpointed: medium
# 1.60 / 3.02 s/it at LQ 64 / 96, small 1.11 / 2.18) and on the v4 run (medium, LQ 64,
# 2 discriminators: 1.47 s/it in the PSNR phase, ~1.6 expected with the GAN); large
# not measured (medium factor). VRAM: the activations follow the same model (bench
# without checkpoint at LQ 120: 11.9 GB, spilled); validation then runs the whole val
# images at full resolution and the log keeps that peak: +3.36 GB on fdat_rd_medium
# (v4 val set, 20 LQ images; 5.96 GB training -> 8.73 GB after the first validation).
FDAT_RD_FULLRES_TIME = {
    "fdat_rd_small": 2.02,
    "fdat_rd_medium": 1.69,
    "fdat_rd_large": 1.69,
}
FDAT_RD_FULLRES_VAL_GB = 3.36   # fdat_rd_medium; other sizes scaled by FDAT_RD_GEN VRAM
FDAT_RD_FIXED_GB = 0.4       # weights, optimizer, loss networks (steady state)
FDAT_RD_LOSS_ACT_GB = 0.6    # loss activations at HR 240, batch 4
FDAT_RD_T0 = 0.059           # fixed time per iteration (s, bench)
FDAT_RD_T_LOSS = 0.466       # calibration losses at HR 240, batch 4 (s, bench)
# Bench -> real speed: the real run is faster than the bench (1.64 it/s at LQ 96
# vs 1/0.742 s predicted by the bench model).
BENCH_TO_REAL_TIME = 0.822
# Gradient checkpointing (use_checkpoint): drops 90 % of the generator activations
# (bench 4.67 -> 1.38 GB), 11 % slower (1.118 -> 0.997 it/s).
CHECKPOINT_GEN_VRAM_SAVING = 0.9
CHECKPOINT_SPEED = 0.89
# Paired dataset vs OTF degradations (real 400 it runs, same config): +3 % speed,
# -0.1 GB. "bicubic" mode is OTF with light settings: same cost as OTF.
PAIRED_SPEED = 1.028
PAIRED_VRAM_SAVING_GB = 0.1

# Per loss: (extra GB, extra bench time per iteration in s), at batch 4 / HR 240.
# Keys: engine type names, lowercase without "_".
LOSS_COST = {
    "perceptualloss": (0.53, 0.094),        # VGG-19
    "perceptualanimeloss": (0.44, 0.041),   # ResNet-50 Danbooru
    "linedistillerloss": (0.67, 0.112),     # real run: +0.80 GB, 1.11 -> 1.03 it/s
    "distsloss": (0.65, 0.110),
    "adistsloss": (3.33, 1.486),            # very heavy: x2.6 slower
    "sparkloss": (0.58, 0.102),
    "contextualloss": (0.02, 0.037),
    "msssiml1loss": (0.02, 0.343),
    "mssimloss": (0.03, 0.044),
    "fliploss": (0.04, 0.079),
    "gradientvarianceloss": (0.03, 0.035),
    "sobeledgeloss": (0.02, 0.0),
    "structuretextureloss": (0.55, 0.0),
    # measured ~0 (pixel-space losses): cosim, hsluv, luma, ncc, fft, average, bicubic
    # LDL runs the EMA generator forward: its time is a share of the generator's.
    "ldlloss": (0.11, 0.0),
}
# Criterion extras on top of LOSS_COST (bench 2026-10-29, vs the same loss with its
# default criterion). VGG "fd" (feature distribution) is very heavy: +5.7 GB and
# 0.88 -> 0.58 it/s. The other criteria (l1 / l2 / charbonnier / huber, pd) of every
# loss measured within the noise (< 3 %).
LOSS_CRITERION_COST = {
    ("perceptualloss", "fd"): (5.14, 0.589),
    ("perceptualloss", "fd+l1"): (5.36, 0.873),
    ("perceptualloss", "pd+l1"): (0.31, 0.0),
}
LDL_GEN_TIME_SHARE = 0.66   # fdat_rd_medium: 0.256 s of 0.385 s generator time
# Losses included in each calibration (their cost is already in the base).
_CALIB_LOSSES_REAL = ("ldlloss", "perceptualanimeloss")   # 2026-10-26/29 real runs
_CALIB_LOSSES_GENERIC = ("perceptualloss",)               # older community figures
_REAL_CALIBRATED = {"nerve", "heart", "wiresr", "fdat_rd_small", "fdat_rd_medium",
                    "fdat_rd_large"}
_LOSS_ALIASES = {
    # perceptualfp16loss = the same VGG-19 (bench: same cost), criteria included
    "vggperceptualloss": "perceptualloss", "perceptualfp16loss": "perceptualloss",
    "distsloss": "distsloss", "ldlloss": "ldlloss", "mssimloss": "mssimloss",
    "msssiml1loss": "msssiml1loss", "edgeloss": "sobeledgeloss", "fdlloss": "perceptualloss",
}

# Per discriminator: (extra GB, extra bench time per iteration in s), at batch 4 /
# HR 240. GB = allocated memory: the reserved figure (UNet-SN +1.16 GB) overstated
# the real run (+0.36 GB at LQ 120, +0.07 GB at LQ 96).
DISC_COST = {
    # UNet-SN time fitted on the real run (1.64 -> 1.39 it/s at LQ 96); the bench
    # read 0.281 s, which put LQ 120 + UNet at 1.02 it/s instead of the real 1.10.
    "unetdiscriminatorsn": (0.52, 0.209),
    "patchgandiscriminatorsn": (0.06, 0.072),
    "multiscalepatchgandiscriminatorsn": (0.16, 0.101),
    "metagan2": (0.07, 0.161),
    "metagan3": (0.13, 0.124),
    "nextsrgandiscriminator": (0.23, 0.136),
    "spectraldiscriminator": (0.12, 0.129),
    "projecteddiscriminator": (0.44, 0.150),
    "vggstylediscriminator": (0.45, 0.148),    # measured at HR 256 (fixed input size)
    "dunet": (0.57, 0.209),
    "docresdiscriminator": (0.10, 0.028),
    # NeoSR discriminators (built from the NeoSR modules, same bench)
    "unet": (0.52, 0.220),
    "patchgan": (0.06, 0.03),
    "metagan": (0.09, 0.056),
    "ea2fpn": (0.27, 0.064),                   # measured at HR 256 (needs HR % 32 == 0)
    "default": (0.3, 0.15),                    # unknown type
}


def _gen_patch(architecture, patch_size: int, scale: int, unshuffle_mod: bool = True) -> int:
    """Patch the generator really runs on. FDAT-RD pads its input to a multiple of 60
    (2x unshuffle, 120 in 1x, else 30): LQ 64, 96 and 120 all run at 120, LQ 128 at 180
    (bench 2026-10-29: same speed and VRAM for 64/96/120, x3 slower at 128). Without
    unshuffle the multiple is 30 (LQ 64 -> 90)."""
    if str(architecture).lower().startswith("fdat_rd"):
        try:
            from src.core.descriptions import fdat_rd_patch_multiple
            m = fdat_rd_patch_multiple(scale, unshuffle_mod)
        except Exception:
            m = (60 if scale == 2 else 120 if scale == 1 else 30) if unshuffle_mod else 30
        return -(-max(1, patch_size) // m) * m
    return patch_size


def _norm_type(t) -> str:
    return str(t or "").lower().replace("_", "").strip()


def _act_factor(batch_factor: float, patch_size: int, scale: int) -> tuple:
    """(VRAM factor, time factor) of a loss network / discriminator: both grow with the
    HR pixels per batch. (1, 1) = the bench point (batch 4, LQ 120, scale 2)."""
    hr = max(8, patch_size) * max(1, scale)
    px = batch_factor * (hr / 240.0) ** 2
    return px ** _ACT_EXP, px


def _losses_cost(architecture, losses, batch_factor, patch_size, scale):
    """(extra GB, extra bench seconds per iteration) of `losses` vs the losses the arch
    was calibrated with. LDL's time is returned apart by _ldl_share()."""
    if losses is None:
        return 0.0, 0.0
    have = _loss_names(losses)
    ref = _CALIB_LOSSES_REAL if str(architecture).lower() in _REAL_CALIBRATED \
        else _CALIB_LOSSES_GENERIC
    vf, tf = _act_factor(batch_factor, patch_size, scale)
    gb = sec = 0.0
    for name, (g, t) in LOSS_COST.items():
        sign = 1 if (name in have and name not in ref) else -1 if (name in ref and name not in have) else 0
        gb += sign * g * vf
        sec += sign * t * tf
    # Criterion extras ("type:criterion" entries), on top of the loss's own cost.
    for l in losses:
        name, _, crit = str(l).partition(":")
        name = _LOSS_ALIASES.get(_norm_type(name), _norm_type(name))
        g, t = LOSS_CRITERION_COST.get((name, crit.strip().lower()), (0.0, 0.0))
        gb += g * vf
        sec += t * tf
    return gb, sec


def _loss_names(losses) -> set:
    """Normalized loss type names ("type" or "type:criterion" entries)."""
    out = set()
    for l in losses or ():
        n = _norm_type(str(l).partition(":")[0])
        out.add(_LOSS_ALIASES.get(n, n))
    return out


def _ldl_change(architecture, losses) -> int:
    """+1 = LDL added vs the calibration, -1 = removed, 0 = unchanged."""
    if losses is None:
        return 0
    have = _loss_names(losses)
    ref = _CALIB_LOSSES_REAL if str(architecture).lower() in _REAL_CALIBRATED \
        else _CALIB_LOSSES_GENERIC
    return (1 if "ldlloss" in have else 0) - (1 if "ldlloss" in ref else 0)


def _discs_cost(discs, batch_factor, patch_size, scale):
    """(extra GB, extra bench seconds per iteration) of the discriminators (D1, D2)."""
    gb = sec = 0.0
    vf, tf = _act_factor(batch_factor, patch_size, scale)
    for d in discs or ():
        if not d:
            continue
        g, t = DISC_COST.get(_norm_type(d), DISC_COST["default"])
        gb += g * vf
        sec += t * tf
    return gb, sec


def estimate_disc_vram_overhead(disc_type: str | None) -> float:
    """Extra VRAM (GB) of one discriminator at the bench point (batch 4, HR 240).
    0.0 if disc_type is falsy (no GAN). Kept for callers; estimate_vram(discs=...)
    scales it with the patch."""
    if not disc_type:
        return 0.0
    return DISC_COST.get(_norm_type(disc_type), DISC_COST["default"])[0]


def detect_gpu_name() -> str:
    """Detect the primary GPU name (cached). Delegates to detect_gpu_info()."""
    return detect_gpu_info()["name"]


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
    losses=None,
    discs=None,
    use_checkpoint: bool = False,
    otf: bool = True,
    unshuffle_mod: bool = True,
    fast: bool = False,
    gan_start_iter: int = 0,
    psnr_losses=None,
) -> dict:
    """
    Estimate the total training duration.

    use_amp    : AMP enabled (fp16 or bf16)
    amp_bf16   : True = BF16 mode (enables Tensor Core boost on Ampere+)
    losses     : engine loss types (e.g. ["charbonnierloss", "PerceptualAnimeLoss"]);
                 None = the losses the arch was calibrated with
    discs      : discriminator types used by the GAN (D1, D2); empty/None = no GAN
    unshuffle_mod: FDAT-RD pixel-unshuffle (scale 1-2). Off = body at full LQ
                 resolution, several times slower and heavier.
    fast       : Real-CUGAN fast variant (generator on the pixel-unshuffled input).
    gan_start_iter: iteration where the GAN starts (0 = from the start). Before it, the
                 run has no discriminator and only psnr_losses (None = losses without the
                 GAN loss): the ETA adds both phases.
    otf        : OTF degradations (realesrgandataset). The calibration runs used OTF; a
                 paired dataset measured 3 % faster and 0.1 GB lighter (real run,
                 fdat_rd_medium, 2026-10-29: 1.195 vs 1.162 it/s, 9.24 vs 9.34 GB).

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

    bf = batch_size / 4.0
    a = str(architecture).lower()
    gpu_time = 0.18 / max(0.01, base_itps)  # bench GPU (1080 Ti) -> this GPU
    _, loss_sec = _losses_cost(a, losses, bf, patch_size, scale)
    _, disc_sec = _discs_cost(discs, bf, patch_size, scale)
    ldl = _ldl_change(a, losses)
    if a in FDAT_RD_GEN:
        # Measured time model: fixed + generator (padded patch) + losses (HR pixels).
        eff = _gen_patch(a, patch_size, scale, unshuffle_mod)
        hr = max(8, patch_size) * max(1, scale)
        if not unshuffle_mod and scale < 3:
            t_gen = FDAT_RD_GEN[a][1] * bf * (eff / 60.0) ** 2 * FDAT_RD_FULLRES_TIME[a]
        else:
            t_gen = FDAT_RD_GEN[a][1] * bf * (eff / 120.0) ** 2
        if use_checkpoint:
            t_gen /= CHECKPOINT_SPEED
        t = (FDAT_RD_T0 + t_gen + FDAT_RD_T_LOSS * bf * (hr / 240.0) ** 2
             + loss_sec + disc_sec + ldl * LDL_GEN_TIME_SHARE * t_gen)
        t *= BENCH_TO_REAL_TIME * gpu_time
        estimated_itps = accum_factor * tc_boost / max(1e-4, t)
    elif a in _MEASURED_ITPS:
        # Real 150-iteration runs (GTX 1080 Ti, batch 2, LQ 64, 2x, charbonnier + VGG
        # perceptual = the generic calibration losses). The variable time follows the
        # pixels: half generator (LQ), half losses (HR, grows with the scale).
        itps_m, b_m = _MEASURED_ITPS[a]
        var = max(0.002, 1.0 / itps_m - MEASURED_T_FIX)
        work = (batch_size / b_m) * (max(8, patch_size) / 64.0) ** 2 * (0.5 + 0.5 * (max(1, scale) / 2.0) ** 2)
        if a == "realcugan" and fast:
            work *= REALCUGAN_FAST_WORK
        t = MEASURED_T_FIX + var * work
        if use_checkpoint:
            t = MEASURED_T_FIX + (t - MEASURED_T_FIX) / CHECKPOINT_SPEED
        t += (loss_sec + disc_sec) * BENCH_TO_REAL_TIME * gpu_time + ldl * 0.3 * var * work * 0.5
        # other GPUs: ratio of the baseline speeds to the 1080 Ti the runs used
        t *= lookup_baseline_itps("GTX 1080 Ti") / max(0.01, base_itps)
        estimated_itps = accum_factor * tc_boost / max(1e-4, t)
    else:
        estimated_itps = (
            base_itps * arch_mult * batch_factor * patch_factor
            * scale_factor * accum_factor * tc_boost
        )
        if use_checkpoint:
            estimated_itps *= CHECKPOINT_SPEED
        # Loss networks and discriminators add their own time per iteration, whatever
        # the generator (an EMA forward for LDL: a share of the generator time).
        t = 1.0 / max(1e-4, estimated_itps / accum_factor)
        t_gen = t * 0.5
        t += (loss_sec + disc_sec) * BENCH_TO_REAL_TIME * gpu_time + ldl * 0.3 * t_gen
        estimated_itps = accum_factor / max(1e-4, t)
    if not otf:
        estimated_itps *= PAIRED_SPEED
    estimated_itps = max(0.001, estimated_itps)

    total_seconds = total_iter / estimated_itps
    # Delayed GAN: PSNR phase without discriminators (and without the GAN-only losses).
    gs = min(max(0, int(gan_start_iter or 0)), int(total_iter))
    if gs > 0 and discs:
        if psnr_losses is None:
            psnr_losses = [l for l in (losses or []) if "gan" not in str(l).lower()]
        p_itps = estimate_training_time(
            gpu_name=gpu_name, architecture=architecture, batch_size=batch_size,
            patch_size=patch_size, scale=scale, accumulate=accumulate, total_iter=gs,
            use_amp=use_amp, amp_bf16=amp_bf16, losses=psnr_losses, discs=[],
            use_checkpoint=use_checkpoint, otf=otf, unshuffle_mod=unshuffle_mod,
            fast=fast)["itps"]
        total_seconds = gs / p_itps + (total_iter - gs) / estimated_itps
        estimated_itps = total_iter / max(1e-6, total_seconds)

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

    vram_gb = estimate_vram(architecture, batch_size, patch_size, use_amp=use_amp,
                            losses=losses, discs=discs, use_checkpoint=use_checkpoint,
                            scale=scale, unshuffle_mod=unshuffle_mod)
    if not otf:
        vram_gb = round(max(0.3, vram_gb - PAIRED_VRAM_SAVING_GB), 2)

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


def run_params_from_config(cfg: dict) -> dict:
    """estimate_training_time() kwargs from a traiNNer-redux config dict (the YAML the
    Config tab generates or the Run tab launches), so both tabs give the same estimate."""
    ng = cfg.get("network_g", {})
    arch = ng.get("type", "omnisr") if isinstance(ng, dict) else (ng or "omnisr")
    scale = int(cfg.get("scale", 4) or 4)
    dsets = cfg.get("datasets") if isinstance(cfg.get("datasets"), dict) else {}
    tds = dsets.get("train", {}) if isinstance(dsets.get("train"), dict) else {}
    # lq_size is in LQ pixels; gt_size in HR pixels.
    if tds.get("lq_size"):
        patch = int(tds["lq_size"])
    else:
        patch = int(tds.get("gt_size") or cfg.get("gt_size", 64) or 64) // max(1, scale)
    train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    # "type:criterion" when a criterion is set (perceptual fd costs far more than l1).
    losses = [str(l.get("type", "")) + (f":{l['criterion']}" if l.get("criterion") else "")
              for l in (train.get("losses") or []) if isinstance(l, dict)]
    discs = []
    if any("gan" in l.lower() for l in losses):
        discs = [d.get("type") for d in (cfg.get("network_d"), cfg.get("network_d2"))
                 if isinstance(d, dict) and d.get("type")]
    # Delayed GAN start (gan_start_iter; gan_auto_start = between gan_auto_min_iter and
    # gan_start_iter, midpoint). PSNR phase losses: weight 0 in loss_phase2 = skipped.
    gan_start = int(train.get("gan_start_iter") or 0)
    if gan_start > 0 and _as_bool(train.get("gan_auto_start", False)):
        gan_start = (gan_start + min(gan_start, int(train.get("gan_auto_min_iter") or 20000))) // 2
    psnr_losses = None
    lp2 = cfg.get("loss_phase2") if isinstance(cfg.get("loss_phase2"), dict) else {}
    if gan_start > 0:
        off = {str(k).lower() for k, v in lp2.items()
               if isinstance(v, (list, tuple)) and v and not float(v[0] or 0)}
        psnr_losses = [l for l in losses if "gan" not in l.lower()
                       and l.split(":")[0].lower() not in off]
    return {
        "architecture": arch,
        "fast": _as_bool(ng.get("fast", False)) if isinstance(ng, dict) else False,
        "gan_start_iter": gan_start,
        "psnr_losses": psnr_losses,
        "batch_size": int(tds.get("batch_size_per_gpu") or tds.get("batch_size") or 4),
        "patch_size": patch,
        "scale": scale,
        "accumulate": int(tds.get("accum_iter") or tds.get("accumulate") or 1),
        # NeoSR templates keep total_iter under [logger] only.
        "total_iter": int(float(train.get("total_iter") or cfg.get("total_iter")
                                or (cfg.get("logger") or {}).get("total_iter") or 100000)),
        "use_amp": bool(cfg.get("use_amp", False)),
        "amp_bf16": bool(cfg.get("amp_bf16", False) or cfg.get("bfloat16", False)),
        "losses": losses,
        "discs": discs,
        "use_checkpoint": bool(isinstance(ng, dict) and ng.get("use_checkpoint")),
        "otf": bool(cfg.get("high_order_degradation", False)),
        # FDAT-RD: missing key = engine default (off). Other archs ignore it.
        "unshuffle_mod": _as_bool(ng.get("unshuffle_mod", False)) if isinstance(ng, dict) else False,
    }


def _as_bool(v) -> bool:
    """YAML bool or its text form ("false" from the UI combobox)."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return bool(v)


def get_pytorch_recommendation(gpu_name: str = None) -> dict:
    """Return recommended PyTorch version + install info based on detected GPU.

    Uses compute capability (sm) + driver version for precise selection when
    nvidia-smi is available; falls back to GPU name patterns otherwise.

    Keys returned:
      gpu_name, gpu_gen, has_tensor_cores,
      torch_version, torchvision_version, cuda_tag, whl_url,
      install_pkgs  (list of "pkg==ver" strings to pip install),
      features, limitations, note, driver_warning (str or "")
    """
    info = detect_gpu_info()
    if gpu_name is None:
        gpu_name = info["name"]

    sm     = info["sm"]        # compute capability as int (e.g. 120, 89, 61)
    driver = info["driver"]    # driver version as float (e.g. 572.83), 0.0 = unknown
    gn     = gpu_name or ""
    driver_warning = ""

    # ── Select wheel tag ──────────────────────────────────────────────────────
    # Primary: use sm (compute capability). Fallback to name patterns when sm==0.

    # Min driver versions required (Windows) for each CUDA runtime:
    #   cu128 (CUDA 12.8): driver >= 570.xx  — Blackwell native (sm_120+)
    #   cu126 (CUDA 12.6): driver >= 527.41  — Maxwell through Hopper
    #   cu118 (CUDA 11.8): driver >= 452.39  — older Pascal / Kepler

    # --- Blackwell sm_100+ (B100/B200 = sm_100, RTX 50xx / RTX PRO = sm_120) ---
    # torch 2.7.0+cu128 is the first build with Blackwell kernels; cu126 has none.
    if sm >= 100 or (sm == 0 and any(p in gn for p in ["RTX 50", "RTX 60", "RTX PRO", "B100", "B200"])):
        gen = "Blackwell"
        has_tc = True
        tv = "2.7.0"; vv = "0.22.0"; ctag = "cu128"; min_tv = "2.7.0"
        features = [_t("BF16 Tensor Cores (natif)", "BF16 Tensor Cores (native)"),
                    "FP16 AMP", "TF32 (fast_matmul)", "FP8 (Blackwell)"]
        limits = [_t("torch.compile requiert WSL2 sur Windows",
                     "torch.compile requires WSL2 on Windows")]
        note = _t("GPU Blackwell — PyTorch 2.7 + CUDA 12.8 requis (sm_120)",
                  "Blackwell GPU — PyTorch 2.7 + CUDA 12.8 required (sm_120)")
        upgrade_reason = _t("cu128 requis pour support natif Blackwell",
                             "cu128 required for native Blackwell support")
        if driver > 0 and driver < 570:
            driver_warning = _t(
                f"Driver NVIDIA {driver:.0f} détecté. CUDA 12.8 (cu128) nécessite driver ≥ 570. "
                "Mettre à jour le driver NVIDIA.",
                f"NVIDIA driver {driver:.0f} detected. CUDA 12.8 (cu128) requires driver ≥ 570. "
                "Update your NVIDIA driver.")

    # --- Hopper sm_90 (H100, H200…) ---
    elif sm == 90 or (sm == 0 and any(p in gn for p in ["H100", "H200", "H800"])):
        gen = "Hopper"
        has_tc = True
        tv = "2.7.0"; vv = "0.22.0"; ctag = "cu126"; min_tv = "2.5.0"
        features = [_t("BF16 Tensor Cores (natif)", "BF16 Tensor Cores (native)"),
                    "FP16 AMP", "TF32 (fast_matmul)", "FP8"]
        limits = [_t("torch.compile requiert WSL2 sur Windows",
                     "torch.compile requires WSL2 on Windows")]
        note = _t("GPU Hopper — PyTorch 2.7 + CUDA 12.6 recommande",
                  "Hopper GPU — PyTorch 2.7 + CUDA 12.6 recommended")
        upgrade_reason = _t("Mettre a jour pour BF16 / TF32 / FP8 optimaux",
                             "Update for optimal BF16 / TF32 / FP8")

    # --- Ada Lovelace sm_89 (RTX 40xx, L40…) ---
    elif sm == 89 or (sm == 0 and any(p in gn for p in ["RTX 40", "L4", "L40"])):
        gen = "Ada Lovelace"
        has_tc = True
        tv = "2.7.0"; vv = "0.22.0"; ctag = "cu126"; min_tv = "2.5.0"
        features = [_t("BF16 Tensor Cores (natif)", "BF16 Tensor Cores (native)"),
                    "FP16 AMP", "TF32 (fast_matmul)", "FP8 (Ada)"]
        limits = [_t("torch.compile requiert WSL2 sur Windows",
                     "torch.compile requires WSL2 on Windows")]
        note = _t("GPU Ada Lovelace — PyTorch 2.7 + CUDA 12.6 recommande",
                  "Ada Lovelace GPU — PyTorch 2.7 + CUDA 12.6 recommended")
        upgrade_reason = _t("Mettre a jour pour BF16 / TF32 / FP8 optimaux",
                             "Update for optimal BF16 / TF32 / FP8")

    # --- Ampere sm_80/86 (RTX 30xx, A100, A40…) ---
    elif sm in range(80, 89) or (sm == 0 and any(p in gn for p in ["RTX 30", "A100", "A10G", "A40", "A30"])):
        gen = "Ampere"
        has_tc = True
        tv = "2.7.0"; vv = "0.22.0"; ctag = "cu126"; min_tv = "2.5.0"
        features = [_t("BF16 Tensor Cores (natif)", "BF16 Tensor Cores (native)"),
                    "FP16 AMP", "TF32 (fast_matmul)"]
        limits = [
            _t("torch.compile requiert WSL2 sur Windows",
               "torch.compile requires WSL2 on Windows"),
            _t("fast_matmul (TF32) : verifier que traiNNer-redux utilise API PyTorch ≥2.7",
               "fast_matmul (TF32): verify traiNNer-redux uses PyTorch ≥2.7 API"),
        ]
        note = _t("GPU Ampere — PyTorch 2.7 + CUDA 12.6 recommande",
                  "Ampere GPU — PyTorch 2.7 + CUDA 12.6 recommended")
        upgrade_reason = _t("Mettre a jour pour BF16 / TF32 optimaux",
                             "Update for optimal BF16 / TF32")

    # --- Turing sm_75 (RTX 20xx, GTX 16xx) ---
    elif sm == 75 or (sm == 0 and any(p in gn for p in ["RTX 20", "GTX 16"])):
        gen = "Turing"
        has_tc = True
        tv = "2.7.0"; vv = "0.22.0"; ctag = "cu126"; min_tv = "2.3.0"
        features = ["FP16 AMP", "Tensor Cores (FP16/INT8)"]
        limits = [
            _t("BF16 : support partiel sur Turing (pas natif)",
               "BF16: partial support on Turing (not native)"),
            _t("TF32 non supporte sur Turing", "TF32 not supported on Turing"),
            _t("torch.compile requiert WSL2 sur Windows",
               "torch.compile requires WSL2 on Windows"),
        ]
        note = _t("GPU Turing — PyTorch 2.7 + CUDA 12.6 recommande",
                  "Turing GPU — PyTorch 2.7 + CUDA 12.6 recommended")
        upgrade_reason = _t("Mettre a jour pour FP16 AMP stable",
                             "Update for stable FP16 AMP")

    # --- Pascal / Volta sm_60-72 (GTX 10xx, GTX Titan V) ---
    elif sm in range(60, 75) or (sm == 0 and any(p in gn for p in ["GTX 10", "GTX 9", "GTX 7", "Titan V"])):
        gen = "Pascal / Volta"
        has_tc = False
        # cu126 works with Pascal (sm_60+) if driver >= 527 — which ships with modern
        # GPU driver packages even for older GPUs. Downgrade to cu118 only if driver
        # is confirmed old (< 527).
        if driver > 0 and driver < 527:
            ctag = "cu118"; tv = "2.7.0"; vv = "0.22.0"; min_tv = "2.3.0"
            driver_warning = _t(
                f"Driver NVIDIA {driver:.0f} détecté. Cu118 utilisé (driver ≥ 527 requis pour cu126). "
                "Mettre à jour le driver pour de meilleures performances.",
                f"NVIDIA driver {driver:.0f} detected. Using cu118 (driver ≥ 527 required for cu126). "
                "Update your driver for better performance.")
        else:
            ctag = "cu126"; tv = "2.7.0"; vv = "0.22.0"; min_tv = "2.3.0"
        features = ["FP32 standard", _t("FP16 AMP (partiel)", "FP16 AMP (partial)")]
        limits = [
            _t("Pas de Tensor Cores → BF16 et TF32 non disponibles",
               "No Tensor Cores → BF16 and TF32 unavailable"),
            _t("Ne pas activer : use_amp BF16, fast_matmul, tf32",
               "Do not enable: use_amp BF16, fast_matmul, tf32"),
            _t("torch.compile non disponible", "torch.compile unavailable"),
            _t("AMP FP16 moins stable que sur Ampere+", "AMP FP16 less stable than on Ampere+"),
        ]
        note = _t("GPU Pascal — PyTorch recommande selon version driver",
                  "Pascal GPU — PyTorch selected based on driver version")
        upgrade_reason = _t("Mettre a jour vers la derniere version stable",
                             "Update to the latest stable version")

    # --- Maxwell sm_50-53 (GTX 9xx) still in torch 2.7.0+cu126; Kepler (< sm_50) is
    # not supported by any current PyTorch build. ---
    elif 35 <= sm <= 53:
        gen = "Kepler / Maxwell"
        has_tc = False
        tv = "2.7.0"; vv = "0.22.0"; ctag = "cu126" if sm >= 50 else "cu118"; min_tv = "2.0.0"
        features = ["FP32 standard"]
        limits = [
            _t("Pas de Tensor Cores", "No Tensor Cores"),
            _t("Support CUDA limité — cu118 uniquement", "Limited CUDA support — cu118 only"),
        ]
        note = _t("GPU Kepler/Maxwell — cu118 uniquement",
                  "Kepler/Maxwell GPU — cu118 only")
        upgrade_reason = ""

    # --- Fallback (sm=0 = nvidia-smi absent, or very old GPU) ---
    else:
        gen = "Pascal / Maxwell / Kepler"
        has_tc = False
        tv = "2.7.0"; vv = "0.22.0"; ctag = "cu126"; min_tv = "2.3.0"
        features = ["FP32 standard", _t("FP16 AMP (partiel)", "FP16 AMP (partial)")]
        limits = [
            _t("Pas de Tensor Cores → BF16 et TF32 non disponibles",
               "No Tensor Cores → BF16 and TF32 unavailable"),
            _t("Ne pas activer : use_amp BF16, fast_matmul, tf32",
               "Do not enable: use_amp BF16, fast_matmul, tf32"),
            _t("torch.compile non disponible", "torch.compile unavailable"),
            _t("AMP FP16 moins stable que sur Ampere+", "AMP FP16 less stable than on Ampere+"),
        ]
        note = _t("GPU sans Tensor Cores — toute version PyTorch ≥2.3 convient",
                  "GPU without Tensor Cores — any PyTorch ≥2.3 works")
        upgrade_reason = _t("Mettre a jour vers la derniere version stable",
                             "Update to the latest stable version")

    whl_url = f"https://download.pytorch.org/whl/{ctag}"
    return {
        "gpu_name":             gpu_name,
        "gpu_gen":              gen,
        "has_tensor_cores":     has_tc,
        "torch_version":        tv,
        "torchvision_version":  vv,
        "cuda_tag":             ctag,
        "min_torch_version":    min_tv,
        "whl_url":              whl_url,
        "install_pkgs":         [f"torch=={tv}", f"torchvision=={vv}"],
        "features":             features,
        "limitations":          limits,
        "note":                 note,
        "upgrade_reason":       upgrade_reason,
        "driver_warning":       driver_warning,
        "driver_version":       driver,
        "compute_sm":           sm,
    }


def format_estimation_message(est: dict) -> str:
    """Format an estimation dict as a human-readable message."""
    confidence_emoji = {"high": "✅", "medium": "🟡", "low": "🟠"}
    emoji = confidence_emoji.get(est["confidence"], "❓")
    vram_str = f"\n{_t('VRAM estimee', 'Estimated VRAM')} : ~{est['vram_gb']:.1f} GB" if "vram_gb" in est else ""
    # Show Tensor Core boost if active
    tc_str = ""
    tc_boost = est.get("tc_boost", 1.0)
    amp_mode = est.get("amp_mode", "off")
    if amp_mode != "off":
        if tc_boost > 1.01:
            tc_str = f"\nAMP {amp_mode.upper()} : {_t('Tensor Cores actifs', 'Tensor Cores active')} (+{(tc_boost-1)*100:.0f}%)"
        else:
            tc_str = f"\nAMP {amp_mode.upper()} : {_t('actif', 'active')}"
    return (
        f"{_t('GPU detecte', 'Detected GPU')} : {est['gpu']}\n"
        f"{_t('Vitesse estimee', 'Estimated speed')} : {est['itps']:.3f} it/s\n"
        f"{_t('Duree estimee', 'Estimated duration')} : {est['eta_str']}"
        f"{vram_str}"
        f"{tc_str}\n"
        f"{_t('Confiance', 'Confidence')} : {emoji} {est['confidence']}"
    )
