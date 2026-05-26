import toml
import yaml
import os
import sys
import ast

try:
    import tomllib
except ImportError:
    import toml as tomllib

# UI display names that differ from neosr loss registry keys.
_LOSS_TYPE_MAP = {
    "chc": "chc_loss",
}

# Archs whose __init__ accepts neither num_in_ch nor **kwargs — injecting
# these keys causes TypeError at train time.
_ARCHS_NO_CHANNELS = {
    "catanet", "cfsr", "cugan", "dunet", "eimn", "esc", "flexnet", "hasn",
    "krgn", "lmlt", "man", "metagan", "moesr", "mosrv2", "msdan", "ninasr",
    "plainusr", "plksr", "safmn", "vgg",
}


class ConfigHandler:
    def __init__(self):
        self.current_engine = "NeoSR"

    def set_engine(self, engine_name):
        self.current_engine = engine_name

    @staticmethod
    def detect_engine(path):
        """Auto-detect engine from file extension and content."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".toml":
            return "NeoSR"
        elif ext in (".yml", ".yaml"):
            return "TraiNNer-Redux"
        # Fallback: check content
        try:
            with open(path, "r", encoding="utf-8") as f:
                head = f.read(500)
            if "model_type" in head or "[network_g]" in head:
                return "NeoSR"
            if "high_order_degradation" in head or "batch_size_per_gpu" in head or "lq_size" in head:
                return "TraiNNer-Redux"
        except Exception:
            pass
        return "NeoSR"

    def load_config(self, path):
        try:
            # Auto-detect engine from file
            detected = self.detect_engine(path)
            self.current_engine = detected

            if path.endswith(".toml"):
                with open(path, "rb") as f:
                    data = tomllib.load(f)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            flat = self._flatten_config(data)
            flat["_detected_engine"] = detected
            return True, flat
        except Exception as e:
            return False, f"Erreur de lecture : {e}"

    def _flatten_config(self, cfg):
        # --- FIX IMPORTANT: safe_num DOIT être défini ICI ---
        def safe_num(k, default, type_cast=float):
            val = cfg.get(k)
            if val is None: val = cfg.get("degradations", {}).get(k)
            if val is None: return default
            try: return type_cast(val)
            except Exception: return default

        def l2s(val): return str(val).replace("[", "").replace("]", "").replace(" ", "")

        def unwrap_path(val):
            """Extract a clean path string from a value that may be a list, a stringified list, or a string.
            Handles malformed values like ["['C:/path']"] from previously broken saves.
            """
            if val is None or val == "":
                return ""
            # Step 1: list → first element
            while isinstance(val, list):
                val = val[0] if val else ""
            # Step 2: stringified list like "['C:/path']" or '["C:/path"]'
            s = str(val).strip()
            for _ in range(3):  # Iterate to undo multiple wrappings
                if s.startswith("[") and s.endswith("]"):
                    inner = s[1:-1].strip()
                    # Strip wrapping quotes
                    if (inner.startswith("'") and inner.endswith("'")) or \
                       (inner.startswith('"') and inner.endswith('"')):
                        inner = inner[1:-1]
                    s = inner.strip()
                else:
                    break
            return s

        flat = {}
        # GENERAL
        flat["name"] = cfg.get("name", "")
        flat["scale"] = str(cfg.get("scale", 4))
        flat["model_type"] = cfg.get("model_type", "image")
        
        flat["use_amp"] = str(cfg.get("use_amp", False)).lower()
        flat["bfloat16"] = str(cfg.get("bfloat16", False)).lower()
        flat["fast_matmul"] = str(cfg.get("fast_matmul", False)).lower()
        flat["compile"] = str(cfg.get("compile", False)).lower()
        # manual_seed → dérive aussi l'état du checkbox "deterministic" dans l'UI
        # NeoSR : manual_seed absent/None/0 = non-déterministe; toute autre valeur = déterministe
        # Redux : champ "deterministic" explicite dans le YAML (lu plus bas)
        _raw_seed = cfg.get("manual_seed")
        if _raw_seed is None or str(_raw_seed) in ("0", "None", "null", ""):
            flat["manual_seed"] = "0"
            flat["deterministic"] = "false"
        else:
            flat["manual_seed"] = str(_raw_seed)
            flat["deterministic"] = "true"
        # Redux : si "deterministic" explicitement dans le config, il prime
        if "deterministic" in cfg:
            flat["deterministic"] = str(cfg["deterministic"]).lower()
        flat["num_gpu"] = str(cfg.get("num_gpu", 1))
        
        mon = cfg.get("monitoring", {})
        flat["auto_tensorboard"] = str(mon.get("auto_tensorboard", False)).lower()
        flat["port_tb"] = str(mon.get("port", 6006))
        flat["auto_ngrok"] = str(mon.get("auto_ngrok", False)).lower()
        
        net_g = cfg.get("network_g", {})
        # Registry key → UI display name (inverse of _ARCH_DISPLAY_TO_REGISTRY)
        _REGISTRY_TO_DISPLAY = {"spanc": "spanpp"}
        _raw_arch = net_g.get("type", "omnisr")
        flat["arch"] = _REGISTRY_TO_DISPLAY.get(_raw_arch, _raw_arch)
        for k, v in net_g.items():
            if k not in ["type", "num_in_ch", "num_out_ch", "upsampling", "scale", "upscale"]:
                flat[f"dyn_{k}"] = v

        net_d = cfg.get("network_d", {})
        if net_d:
            flat["net_d_type"] = net_d.get("type", "unet")
            for k, v in net_d.items():
                if k not in ["type", "num_in_ch", "in_ch"]:
                    flat[f"dynd_{k}"] = v

        ds_train = cfg.get("datasets", {}).get("train", {})
        # Normalize Redux dataset types -> internal names
        # Redux uses: realesrgandataset (OTF), pairedimagedataset
        # NeoSR uses: otf, paired
        raw_type = str(ds_train.get("type", "paired")).lower()
        if "realesrgan" in raw_type or "otf" in raw_type:
            # Detect bicubic-only OTF from absence of high_order_degradation or gaussian_noise_prob == 0
            _is_bicubic = (not cfg.get("high_order_degradation", True)
                           and float(ds_train.get("gaussian_noise_prob", cfg.get("gaussian_noise_prob", 0.65))) < 0.01)
            flat["dataset_mode"] = "bicubic" if _is_bicubic else "otf"
        elif "bicubic" in raw_type:
            flat["dataset_mode"] = "bicubic"
        elif "paired" in raw_type:
            flat["dataset_mode"] = "paired"
        elif "single" in raw_type:
            flat["dataset_mode"] = "single"
        else:
            flat["dataset_mode"] = raw_type  # Keep as-is
        gt = ds_train.get("dataroot_gt", "")
        flat["dataroot_gt"] = unwrap_path(gt)
        # Support both NeoSR (batch_size) and Redux (batch_size_per_gpu)
        flat["batch_size"] = ds_train.get("batch_size", ds_train.get("batch_size_per_gpu", 4))
        flat["accumulate"] = ds_train.get("accumulate", ds_train.get("accum_iter", 1))
        # Support both NeoSR (patch_size) and Redux (lq_size)
        flat["patch_size"] = ds_train.get("patch_size", ds_train.get("lq_size", 64))
        flat["num_worker_per_gpu"] = ds_train.get("num_worker_per_gpu", 4)
        flat["prefetch_mode"] = ds_train.get("prefetch_mode", "cuda")
        flat["use_hflip"] = str(ds_train.get("use_hflip", True)).lower()
        flat["use_rot"] = str(ds_train.get("use_rot", True)).lower()
        
        augs = ds_train.get("augmentation", [])
        probs = ds_train.get("aug_prob", [])
        aug_map = {"mixup": "aug_mixup", "cutmix": "aug_cutmix", "resizemix": "aug_resizemix", "cutblur": "aug_cutblur"}
        idx = 0
        for a in augs:
            if a == "none": idx += 1; continue
            if a in aug_map:
                flat[aug_map[a]] = "true"
                if idx < len(probs): flat[f"prob_{aug_map[a]}"] = probs[idx]
            idx += 1

        ds_val = cfg.get("datasets", {}).get("val", {})
        # Validation paths can be list-form in Redux: dataroot_gt: ['path']
        # Or even doubly-wrapped from broken saves: dataroot_gt: ["['path']"]
        flat["val_gt"] = unwrap_path(ds_val.get("dataroot_gt", ""))
        flat["val_lq"] = unwrap_path(ds_val.get("dataroot_lq", ""))

        paths = cfg.get("path", {})
        # YAML null becomes Python None — convert to empty string for the entry widget
        rs = paths.get("resume_state")
        flat["resume_state"] = "" if rs is None else str(rs)
        pm = paths.get("pretrain_network_g")
        flat["pretrain_model"] = "" if pm is None else str(pm)

        train = cfg.get("train", {})
        logger = cfg.get("logger", {})
        
        iter_val = logger.get("total_iter") or train.get("total_iter") or train.get("n_iter") or 100000
        flat["total_iter"] = iter_val
        flat["warmup_iter"] = train.get("warmup_iter", -1)
        flat["grad_clip"] = str(train.get("grad_clip", False)).lower()
        # NeoSR uses "ema", Redux uses "ema_decay" — read both for round-trip fidelity
        flat["ema"] = str(train.get("ema", train.get("ema_decay", 0.999)))
        flat["sam"] = train.get("sam", "none")
        flat["sam_init"] = train.get("sam_init", -1)
        flat["eco_mode"] = str(train.get("eco", False)).lower()
        eco_pt = paths.get("eco_pretrain_g")
        flat["eco_pretrain_path"] = "" if eco_pt is None else str(eco_pt)
        flat["match_lq_colors"] = str(train.get("match_lq_colors", False)).lower()
        
        opt_g = train.get("optim_g", {})
        flat["optim_g"] = opt_g.get("type", "AdamW")
        flat["lr"] = opt_g.get("lr", 5e-5)
        # Fix schedule_free
        flat["schedule_free"] = str(opt_g.get("schedule_free", False)).lower()
        flat["warmup_steps"] = opt_g.get("warmup_steps", -1)
        
        opt_d = train.get("optim_d", {})
        flat["lr_d"] = opt_d.get("lr", 5e-5)
        
        sched = train.get("scheduler", {})
        flat["scheduler"] = sched.get("type", "MultiStepLR")
        if "milestones" in sched:
            flat["milestones"] = ",".join(map(str, sched["milestones"]))
        # Read T_max so the CosineAnnealingLR widget restores correctly
        if "T_max" in sched:
            flat["t_max"] = str(sched["T_max"])
        elif "t_max" in sched:
            flat["t_max"] = str(sched["t_max"])
        
        def get_w(key, default=1.0):
            opt = train.get(key)
            return opt.get("loss_weight", default) if isinstance(opt, dict) else default

        if "pixel_opt" in train:
            flat["loss_pixel"] = "true"; flat["weight_loss_pixel"] = train["pixel_opt"].get("loss_weight", 1.0)
            flat["pixel_reduction"] = train["pixel_opt"].get("reduction", "mean")
            flat["pixel_criterion"] = train["pixel_opt"].get("type", "L1Loss")
        else:
            flat["loss_pixel"] = "false"

        flat["loss_wavelet"] = "true" if train.get("wavelet_guided") else "false"
        flat["wavelet_init"] = train.get("wavelet_init", 10000)
        flat["weight_loss_wavelet"] = get_w("wavelet_opt")

        flat["loss_fdl"] = "true" if "fdl_opt" in train else "false"
        flat["weight_loss_fdl"] = get_w("fdl_opt")
        flat["fdl_model"] = train.get("fdl_opt", {}).get("model", "vgg")

        flat["loss_ldl"] = "true" if "ldl_opt" in train else "false"
        flat["weight_loss_ldl"] = get_w("ldl_opt")
        flat["ldl_criterion"] = train.get("ldl_opt", {}).get("criterion", "l1")
        flat["ldl_ksize"]     = train.get("ldl_opt", {}).get("ksize", 7)

        flat["loss_consistency"] = "true" if "consistency_opt" in train else "false"
        flat["weight_loss_consistency"] = get_w("consistency_opt")
        flat["consistency_blur"]       = str(train.get("consistency_opt", {}).get("use_blur",       True)).lower()
        flat["consistency_cosim"]      = str(train.get("consistency_opt", {}).get("use_cosim",      True)).lower()
        flat["consistency_saturation"] = str(train.get("consistency_opt", {}).get("use_saturation", True)).lower()
        flat["consistency_brightness"] = str(train.get("consistency_opt", {}).get("use_brightness", True)).lower()

        flat["loss_edge"] = "true" if "edge_opt" in train else "false"
        flat["weight_loss_edge"] = get_w("edge_opt", 0.05)
        flat["edge_criterion"] = train.get("edge_opt", {}).get("criterion", "l1")
        flat["edge_corner"] = str(train.get("edge_opt", {}).get("corner", False)).lower()

        flat["loss_mssim"] = "true" if "mssim_opt" in train else "false"
        flat["weight_loss_mssim"] = get_w("mssim_opt")
        flat["mssim_window_size"] = train.get("mssim_opt", {}).get("window_size", 11)
        flat["mssim_sigma"]       = train.get("mssim_opt", {}).get("sigma", 1.5)
        flat["mssim_k1"]          = train.get("mssim_opt", {}).get("K1", 0.01)
        flat["mssim_k2"]          = train.get("mssim_opt", {}).get("K2", 0.03)

        flat["loss_dists"] = "true" if "dists_opt" in train else "false"
        flat["weight_loss_dists"] = get_w("dists_opt")
        
        flat["loss_msswd"] = "true" if "msswd_opt" in train else "false"
        flat["weight_loss_msswd"] = get_w("msswd_opt")

        flat["loss_ff"] = "true" if "ff_opt" in train else "false"
        flat["weight_loss_ff"] = get_w("ff_opt")
        flat["ff_alpha"] = train.get("ff_opt", {}).get("alpha", 1.0)
        
        flat["loss_ncc"] = "true" if "ncc_opt" in train else "false"
        flat["weight_loss_ncc"] = get_w("ncc_opt")
        
        flat["loss_kl"] = "true" if "kl_opt" in train else "false"
        flat["weight_loss_kl"] = get_w("kl_opt")

        # Redux-only losses
        flat["loss_hsluv"] = "true" if "hsluv_opt" in train else "false"
        flat["weight_loss_hsluv"] = get_w("hsluv_opt")
        flat["hsluv_hue_weight"] = train.get("hsluv_opt", {}).get("hue_weight",        0.33)
        flat["hsluv_sat_weight"] = train.get("hsluv_opt", {}).get("saturation_weight", 0.33)
        flat["hsluv_lum_weight"] = train.get("hsluv_opt", {}).get("lightness_weight",  0.33)

        flat["loss_cosim"] = "true" if "cosim_opt" in train else "false"
        flat["weight_loss_cosim"] = get_w("cosim_opt")
        flat["cosim_lambda"] = train.get("cosim_opt", {}).get("cosim_lambda", 5)

        flat["loss_color"] = "true" if "color_opt" in train else "false"
        flat["weight_loss_color"] = get_w("color_opt")
        flat["color_criterion"] = train.get("color_opt", {}).get("criterion", "l1")

        flat["loss_gv"] = "true" if "gv_opt" in train else "false"
        flat["weight_loss_gv"] = get_w("gv_opt")
        flat["gv_patch_size"] = train.get("gv_opt", {}).get("patch_size", 16)
        flat["gv_criterion"]  = train.get("gv_opt", {}).get("criterion", "charbonnier")

        flat["loss_luma"] = "true" if "luma_opt" in train else "false"
        flat["weight_loss_luma"] = get_w("luma_opt")
        flat["luma_criterion"] = train.get("luma_opt", {}).get("criterion", "l1")

        flat["loss_contextual"] = "true" if "contextual_opt" in train else "false"
        flat["weight_loss_contextual"] = get_w("contextual_opt")
        flat["ctx_distance_type"] = train.get("contextual_opt", {}).get("distance_type", "cosine")
        flat["ctx_band_width"]    = train.get("contextual_opt", {}).get("band_width",    0.5)

        flat["loss_percep"] = "true" if "perceptual_opt" in train else "false"
        percep = train.get("perceptual_opt", {})
        flat["weight_loss_percep"] = percep.get("loss_weight", 1.0)
        flat["percep_criterion"] = percep.get("criterion", "charbonnier")
        _VGG_ALL = ["conv1_2","conv2_2","conv3_2","conv3_4","conv4_2","conv4_4","conv5_2","conv5_4"]
        _lw = percep.get("layer_weights", {})
        for _l in _VGG_ALL:
            flat[f"percep_vgg_{_l}"] = _lw.get(_l, 0.0)
        # Legacy single-key for Redux + UI compat
        flat["percep_layer"] = "conv5_4"
        for k in _lw:
            flat["percep_layer"] = k; break

        if "gan_opt" in train:
            flat["use_gan"] = "true"
            flat["gan_loss_weight"] = train["gan_opt"].get("loss_weight", 0.05)
            flat["gan_type"] = train["gan_opt"].get("gan_type", "bce")
            flat["real_label_val"] = train["gan_opt"].get("real_label_val", 1.0)
            flat["fake_label_val"] = train["gan_opt"].get("fake_label_val", 0.0)
        else:
            flat["use_gan"] = "false"

        # --- Redux losses (list format) ---
        losses_list = train.get("losses", [])
        if losses_list and isinstance(losses_list, list):
            for loss in losses_list:
                if not isinstance(loss, dict):
                    continue
                lt = loss.get("type", "").lower()
                lw = loss.get("loss_weight", 0)
                if lw == 0:
                    continue  # Disabled loss
                if "charbonnier" in lt or "l1loss" in lt or "mseloss" in lt:
                    flat["loss_pixel"] = "true"
                    flat["weight_loss_pixel"] = lw
                    flat["pixel_criterion"] = loss.get("type", "charbonnierloss")
                elif "mssim" in lt:
                    flat["loss_mssim"] = "true"
                    flat["weight_loss_mssim"] = lw
                elif "perceptual" in lt:
                    flat["loss_percep"] = "true"
                    flat["weight_loss_percep"] = lw
                    flat["percep_criterion"] = loss.get("criterion", "charbonnier")
                elif "hsluv" in lt:
                    flat["loss_hsluv"] = "true"
                    flat["weight_loss_hsluv"] = lw
                    flat["hsluv_hue_weight"] = loss.get("hue_weight",        0.33)
                    flat["hsluv_sat_weight"] = loss.get("saturation_weight", 0.33)
                    flat["hsluv_lum_weight"] = loss.get("lightness_weight",  0.33)
                elif "cosim" in lt:
                    flat["loss_cosim"] = "true"
                    flat["weight_loss_cosim"] = lw
                    flat["cosim_lambda"] = loss.get("cosim_lambda", 5)
                elif "ganloss" in lt:
                    flat["use_gan"] = "true"
                    flat["gan_loss_weight"] = lw
                    flat["gan_type"] = loss.get("gan_type", "vanilla")
                elif "dists" in lt:
                    flat["loss_dists"] = "true"
                    flat["weight_loss_dists"] = lw
                elif "ldl" in lt:
                    flat["loss_ldl"] = "true"
                    flat["weight_loss_ldl"] = lw
                    flat["ldl_criterion"] = loss.get("criterion", "l1")
                    flat["ldl_ksize"]     = loss.get("ksize", 7)
                elif "ffloss" in lt or "focalfrequency" in lt:
                    flat["loss_ff"] = "true"
                    flat["weight_loss_ff"] = lw
                    flat["ff_alpha"] = loss.get("alpha", 1.0)
                elif "colorloss" in lt or "color_loss" in lt:
                    flat["loss_color"] = "true"
                    flat["weight_loss_color"] = lw
                    flat["color_criterion"] = loss.get("criterion", "l1")
                elif "gradientvariance" in lt or "gv_loss" in lt:
                    flat["loss_gv"] = "true"
                    flat["weight_loss_gv"] = lw
                    flat["gv_patch_size"] = loss.get("patch_size", 16)
                    flat["gv_criterion"]  = loss.get("criterion", "charbonnier")
                elif "lumaloss" in lt or "luma_loss" in lt:
                    flat["loss_luma"] = "true"
                    flat["weight_loss_luma"] = lw
                    flat["luma_criterion"] = loss.get("criterion", "l1")
                elif "contextual" in lt:
                    flat["loss_contextual"] = "true"
                    flat["weight_loss_contextual"] = lw
                    flat["ctx_distance_type"] = loss.get("distance_type", "cosine")
                    flat["ctx_band_width"]    = loss.get("band_width",    0.5)
                elif "spark" in lt:
                    flat["loss_spark"] = "true"
                    flat["weight_loss_spark"] = lw
                    flat["spark_criterion"]   = loss.get("criterion", "fd")
                    flat["spark_path"]        = loss.get("path", "")

        flat["print_freq"] = logger.get("print_freq", 100)
        flat["save_freq"] = logger.get("save_checkpoint_freq", 5000)
        flat["use_tb_logger"] = str(logger.get("use_tb_logger", True)).lower()

        # Read monitoring settings (auto_tensorboard, auto_ngrok) from YAML if present
        mon = cfg.get("monitoring", {})
        flat["auto_tensorboard"] = str(mon.get("auto_tensorboard", False)).lower()
        flat["port_tb"] = str(mon.get("port", 6006))
        flat["auto_ngrok"] = str(mon.get("auto_ngrok", False)).lower()

        val = cfg.get("val", {})
        flat["val_freq"] = val.get("val_freq", 5000)
        flat["tile"] = val.get("tile", val.get("tile_size", 200))
        flat["tile_pad"] = val.get("tile_pad", val.get("tile_overlap", 32))
        flat["save_img"] = str(val.get("save_img", True)).lower()
        flat["val_enabled"] = str(val.get("val_enabled", True)).lower()

        metrics = val.get("metrics", {})
        flat["metric_psnr"] = "true" if "psnr" in metrics else "false"
        flat["metric_ssim"] = "true" if "ssim" in metrics else "false"
        flat["metric_lpips"] = "true" if "lpips" in metrics else "false"
        flat["metric_niqe"] = "true" if "niqe" in metrics else "false"
        flat["metric_dists"] = "true" if "dists" in metrics else "false"

        degs = cfg.get("degradations", {})
        # Redux puts degradation params at top level
        if not degs and cfg.get("high_order_degradation"):
            degs = {k: v for k, v in cfg.items() if k.startswith(("resize_", "gaussian_", "noise_", "poisson_", "gray_", "blur_", "sinc_", "betag_", "betap_", "jpeg_", "final_", "second_", "queue_"))}
        # Also check in train dataset for Redux kernel params
        ds_degs = cfg.get("datasets", {}).get("train", {})
        for dk in ("blur_kernel_size", "kernel_list", "kernel_prob", "kernel_range", "sinc_prob",
                    "blur_sigma", "betag_range", "betap_range",
                    "blur_kernel_size2", "kernel_list2", "kernel_prob2", "kernel_range2",
                    "sinc_prob2", "blur_sigma2", "betag_range2", "betap_range2",
                    "final_sinc_prob", "final_kernel_range"):
            if dk in ds_degs and dk not in degs:
                degs[dk] = ds_degs[dk]
        
        flat["resize_prob"] = l2s(degs.get("resize_prob", "0.2, 0.7, 0.1"))
        flat["resize_range"] = l2s(degs.get("resize_range", "0.5, 1.5"))
        flat["gaussian_noise_prob"] = degs.get("gaussian_noise_prob", 0.5)
        flat["noise_range"] = l2s(degs.get("noise_range", "1, 30"))
        flat["poisson_scale_range"] = l2s(degs.get("poisson_scale_range", "0.05, 3.0"))
        flat["gray_noise_prob"] = degs.get("gray_noise_prob", 0.4)
        
        flat["blur_prob"] = degs.get("blur_prob", 0.35)
        flat["blur_kernel_size"] = degs.get("blur_kernel_size", 21)
        flat["blur_sigma"] = l2s(degs.get("blur_sigma", "0.2, 3.0"))
        flat["sinc_prob"] = degs.get("sinc_prob", 0.1)
        flat["betag_range"] = l2s(degs.get("betag_range", "0.5, 4.0"))
        flat["betap_range"] = l2s(degs.get("betap_range", "1, 2"))
        
        # blur_prob2 is the Redux engine key; second_blur_prob is the studio widget key
        flat["second_blur_prob"] = degs.get("second_blur_prob", degs.get("blur_prob2", 0.8))
        flat["resize_prob2"] = l2s(degs.get("resize_prob2", "0.3, 0.4, 0.3"))
        flat["resize_range2"] = l2s(degs.get("resize_range2", "0.3, 1.2"))
        flat["gaussian_noise_prob2"] = degs.get("gaussian_noise_prob2", 0.5)
        flat["noise_range2"] = l2s(degs.get("noise_range2", "1, 25"))
        flat["poisson_scale_range2"] = l2s(degs.get("poisson_scale_range2", [0.05, 2.5]))
        flat["gray_noise_prob2"] = degs.get("gray_noise_prob2", 0.4)
        
        # UTILISATION DU SAFE_NUM DÉFINI PLUS HAUT
        flat["blur_kernel_size2"] = safe_num("blur_kernel_size2", 21, int)
        flat["blur_sigma2"] = l2s(degs.get("blur_sigma2", "0.2, 1.5"))
        flat["sinc_prob2"] = degs.get("sinc_prob2", 0.1)
        flat["betag_range2"] = l2s(degs.get("betag_range2", "0.5, 4.0"))
        flat["betap_range2"] = l2s(degs.get("betap_range2", "1, 2"))
        
        flat["jpeg_range"] = l2s(degs.get("jpeg_range", "30, 95"))
        flat["jpeg_range2"] = l2s(degs.get("jpeg_range2", "30, 95"))
        
        flat["final_sinc_prob"] = safe_num("final_sinc_prob", 0.8, float)
        flat["jpeg_prob"] = degs.get("jpeg_prob", 1.0)

        # Custom degradations (banding & posterize)
        flat["posterize_prob"] = safe_num("posterize_prob", 0.0, float)
        flat["posterize_bits_range"] = l2s(degs.get("posterize_bits_range", "3, 6"))
        flat["banding_prob"] = safe_num("banding_prob", 0.0, float)
        flat["banding_levels_range"] = l2s(degs.get("banding_levels_range", "16, 64"))

        flat["chroma_prob"] = safe_num("chroma_prob", 0.0, float)
        flat["ca_prob"] = safe_num("ca_prob", 0.0, float)
        flat["ca_shift_range"] = l2s(degs.get("ca_shift_range", "1, 5"))
        flat["halation_prob"] = safe_num("halation_prob", 0.0, float)
        flat["halation_strength_range"] = l2s(degs.get("halation_strength_range", "0.05, 0.3"))
        flat["salt_pepper_prob"] = safe_num("salt_pepper_prob", 0.0, float)
        flat["salt_pepper_amount_range"] = l2s(degs.get("salt_pepper_amount_range", "0.001, 0.05"))
        flat["vhs_prob"] = safe_num("vhs_prob", 0.0, float)
        flat["vhs_strength_range"] = l2s(degs.get("vhs_strength_range", "0.1, 0.5"))
        flat["aliasing_prob"] = safe_num("aliasing_prob", 0.0, float)
        flat["aliasing_scale_range"] = l2s(degs.get("aliasing_scale_range", "0.5, 0.85"))
        flat["interlace_weave_prob"] = safe_num("interlace_weave_prob", 0.0, float)
        flat["interlace_weave_strength_range"] = l2s(degs.get("interlace_weave_strength_range", "0.5, 1.0"))
        flat["interlace_flicker_prob"] = safe_num("interlace_flicker_prob", 0.0, float)
        flat["interlace_flicker_strength_range"] = l2s(degs.get("interlace_flicker_strength_range", "0.1, 0.4"))
        flat["interlace_blend_prob"] = safe_num("interlace_blend_prob", 0.0, float)
        flat["interlace_blend_strength_range"] = l2s(degs.get("interlace_blend_strength_range", "0.3, 1.0"))
        flat["film_grain_prob"] = safe_num("film_grain_prob", 0.0, float)
        flat["film_grain_strength_range"] = l2s(degs.get("film_grain_strength_range", "0.03, 0.12"))
        flat["film_grain_size_range"] = l2s(degs.get("film_grain_size_range", "1, 2"))
        flat["oversharp_prob"] = safe_num("oversharp_prob", 0.0, float)
        flat["oversharp_strength_range"] = l2s(degs.get("oversharp_strength_range", "0.5, 2.0"))
        flat["scanlines_prob"] = safe_num("scanlines_prob", 0.0, float)
        flat["scanlines_strength_range"] = l2s(degs.get("scanlines_strength_range", "0.2, 0.5"))
        flat["scanlines_spacing_range"] = l2s(degs.get("scanlines_spacing_range", "2, 4"))

        return flat

    def generate_config(self, data, save_path):
        """Fonction appelée par l'interface - DOIT toujours retourner (bool, message)"""
        try:
            if "Redux" in self.current_engine:
                return self._gen_redux(data, save_path)
            else:
                return self._generate_neosr_toml(data, save_path)
        except Exception as e:
            return False, f"Erreur génération : {e}"

    def _gen_redux(self, data, save_path):
        """Génération complète pour TraiNNer-Redux (.yml)"""
        try:
            def safe_num(k, d, t=float):
                try: return t(float(data.get(k, d)))
                except Exception: return d
            def safe_bool(k, d):
                val = str(data.get(k, str(d))).lower()
                return val in ["true", "1", "yes", "on"]
            def safe_list(k, default):
                val = data.get(k, default)
                if isinstance(val, list): return val
                if isinstance(val, str):
                    val = val.strip().strip("[]")
                    try: return [float(x.strip()) for x in val.split(",") if x.strip()]
                    except Exception: return default
                return default

            scale = safe_num("scale", 4, int)
            total_iter = safe_num("total_iter", 100000, int)
            exp_name = data.get("name", "experiment").strip()
            arch = data.get("arch", "omnisr")
            lq_size = safe_num("patch_size", 64, int)
            bs = safe_num("batch_size", 4, int)
            _ds_mode = data.get("dataset_mode", "paired")
            is_otf = _ds_mode in ("otf", "realesrgandataset", "bicubic")

            config = {
                "name": exp_name,
                "scale": scale,
                "use_amp": safe_bool("use_amp", True),
                "amp_bf16": safe_bool("bfloat16", False),
                "use_channels_last": False,
                "fast_matmul": safe_bool("fast_matmul", False),
                "use_compile": safe_bool("compile", False),
                "compile_mode": "default",
                "num_gpu": "auto",
                "deterministic": safe_bool("deterministic", False),
            }

            if is_otf:
                _is_bicubic_mode = _ds_mode == "bicubic"
                config["high_order_degradation"] = not _is_bicubic_mode
                config["high_order_degradations_debug"] = False
                config["lq_usm"] = False
                # First degradation
                config["thicklines_prob"] = 0.0
                config["blur_prob"] = safe_num("blur_prob", 0.0, float)
                config["resize_prob"] = safe_list("resize_prob", [0.2, 0.7, 0.1])
                config["resize_mode_list"] = ["bilinear", "bicubic", "nearest-exact", "lanczos"]
                config["resize_mode_prob"] = [0.25, 0.25, 0.25, 0.25]
                config["resize_range"] = safe_list("resize_range", [0.4, 1.5])
                config["gaussian_noise_prob"] = safe_num("gaussian_noise_prob", 0.0, float)
                config["noise_range"] = safe_list("noise_range", [1, 25])
                config["poisson_scale_range"] = safe_list("poisson_scale_range", [0.05, 2.0])
                config["gray_noise_prob"] = safe_num("gray_noise_prob", 0.0, float)
                config["jpeg_prob"] = safe_num("jpeg_prob", 1.0, float)
                config["jpeg_range"] = safe_list("jpeg_range", [30, 95])
                # Second degradation
                # Engine key is blur_prob2 (not second_blur_prob — that's the studio widget key)
                config["blur_prob2"] = safe_num("second_blur_prob", 0.0, float)
                config["resize_prob2"] = safe_list("resize_prob2", [0.3, 0.4, 0.3])
                config["resize_mode_list2"] = ["bilinear", "bicubic", "nearest-exact", "lanczos"]
                config["resize_mode_prob2"] = [0.25, 0.25, 0.25, 0.25]
                config["resize_range2"] = safe_list("resize_range2", [0.6, 1.2])
                config["gaussian_noise_prob2"] = safe_num("gaussian_noise_prob2", 0.0, float)
                config["noise_range2"] = safe_list("noise_range2", [1, 25])
                config["poisson_scale_range2"] = safe_list("poisson_scale_range2", [0.05, 2.5])
                config["gray_noise_prob2"] = safe_num("gray_noise_prob2", 0.0, float)
                config["jpeg_prob2"] = safe_num("jpeg_prob2", 1.0, float)
                config["jpeg_range2"] = safe_list("jpeg_range2", [30, 95])
                # Final resize
                config["resize_mode_list3"] = ["bilinear", "bicubic", "nearest-exact", "lanczos"]
                config["resize_mode_prob3"] = [0.25, 0.25, 0.25, 0.25]
                config["queue_size"] = 120
                # Custom degradations (patched into engine at runtime) — only write when non-zero
                p_prob = safe_num("posterize_prob", 0.0, float)
                if p_prob > 0:
                    config["posterize_prob"] = p_prob
                    config["posterize_bits_range"] = safe_list("posterize_bits_range", [3, 6])
                b_prob = safe_num("banding_prob", 0.0, float)
                if b_prob > 0:
                    config["banding_prob"] = b_prob
                    config["banding_levels_range"] = safe_list("banding_levels_range", [16, 64])
                c_prob = safe_num("chroma_prob", 0.0, float)
                if c_prob > 0:
                    config["chroma_prob"] = c_prob
                ca_prob = safe_num("ca_prob", 0.0, float)
                if ca_prob > 0:
                    config["ca_prob"] = ca_prob
                    config["ca_shift_range"] = safe_list("ca_shift_range", [1, 5])
                h_prob = safe_num("halation_prob", 0.0, float)
                if h_prob > 0:
                    config["halation_prob"] = h_prob
                    config["halation_strength_range"] = safe_list("halation_strength_range", [0.05, 0.3])
                sp_prob = safe_num("salt_pepper_prob", 0.0, float)
                if sp_prob > 0:
                    config["salt_pepper_prob"] = sp_prob
                    config["salt_pepper_amount_range"] = safe_list("salt_pepper_amount_range", [0.001, 0.05])
                v_prob = safe_num("vhs_prob", 0.0, float)
                if v_prob > 0:
                    config["vhs_prob"] = v_prob
                    config["vhs_strength_range"] = safe_list("vhs_strength_range", [0.1, 0.5])
                a_prob = safe_num("aliasing_prob", 0.0, float)
                if a_prob > 0:
                    config["aliasing_prob"] = a_prob
                    config["aliasing_scale_range"] = safe_list("aliasing_scale_range", [0.5, 0.85])
                for _k, _def in [
                    ("interlace_weave_prob",    0.0), ("interlace_flicker_prob", 0.0),
                    ("interlace_blend_prob",    0.0), ("film_grain_prob",        0.0),
                    ("oversharp_prob",          0.0), ("scanlines_prob",         0.0),
                ]:
                    _pv = safe_num(_k, 0.0, float)
                    if _pv > 0:
                        config[_k] = _pv
                for _k, _def in [
                    ("interlace_weave_strength_range",   [0.5, 1.0]),
                    ("interlace_flicker_strength_range", [0.1, 0.4]),
                    ("interlace_blend_strength_range",   [0.3, 1.0]),
                    ("film_grain_strength_range",        [0.03, 0.12]),
                    ("film_grain_size_range",            [1, 2]),
                    ("oversharp_strength_range",         [0.5, 2.0]),
                    ("scanlines_strength_range",         [0.2, 0.5]),
                    ("scanlines_spacing_range",          [2, 4]),
                ]:
                    _base = _k.replace("_strength_range", "_prob").replace("_size_range", "_prob").replace("_spacing_range", "_prob")
                    if safe_num(_base, 0.0, float) > 0:
                        config[_k] = safe_list(_k, _def)

            # Datasets
            train_type = "realesrgandataset" if is_otf else "pairedimagedataset"
            gt_path = data.get("dataroot_gt", "datasets/train/dataset1/hr").replace("\\", "/")
            train_ds = {
                "name": "Train Dataset",
                "type": train_type,
                "dataroot_gt": [gt_path],
                "lq_size": lq_size,
                "use_hflip": True,
                "use_rot": True,
                "num_worker_per_gpu": safe_num("num_worker_per_gpu", 8, int),
                "batch_size_per_gpu": bs,
                "accum_iter": safe_num("accumulate", 1, int),
            }
            if not is_otf:
                lq_path = data.get("dataroot_lq", "")
                if lq_path:
                    train_ds["dataroot_lq"] = [lq_path.replace("\\", "/")]
            else:
                # Kernel params for OTF
                train_ds.update({
                    "blur_kernel_size": safe_num("blur_kernel_size", 12, int),
                    "kernel_list": ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
                    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                    "kernel_range": [5, 17],
                    "sinc_prob": safe_num("sinc_prob", 0.0, float),
                    "blur_sigma": safe_list("blur_sigma", [0.2, 2]),
                    "betag_range": safe_list("betag_range", [0.5, 4]),
                    "betap_range": safe_list("betap_range", [1, 2]),
                    "blur_kernel_size2": safe_num("blur_kernel_size2", 12, int),
                    "kernel_list2": ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
                    "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                    "kernel_range2": [5, 17],
                    "sinc_prob2": safe_num("sinc_prob2", 0.0, float),
                    "blur_sigma2": safe_list("blur_sigma2", [0.2, 1]),
                    "betag_range2": safe_list("betag_range2", [0.5, 4]),
                    "betap_range2": safe_list("betap_range2", [1, 2]),
                    "final_sinc_prob": safe_num("final_sinc_prob", 0.0, float),
                    "final_kernel_range": [5, 17],
                })

            val_gt = data.get("val_gt", "datasets/val/dataset1/hr").replace("\\", "/")
            val_lq = data.get("val_lq", "datasets/val/dataset1/lr").replace("\\", "/")
            config["datasets"] = {
                "train": train_ds,
                "val": {
                    "name": "Val Dataset",
                    "type": "pairedimagedataset",
                    "dataroot_gt": [val_gt] if val_gt else ["datasets/val/dataset1/hr"],
                    "dataroot_lq": [val_lq] if val_lq else ["datasets/val/dataset1/lr"],
                },
            }

            # Network — dyn_ keys are arch params EXCEPT these UI-only ones:
            _DYN_NOT_NET = {"gan_weight"}
            # Bug fix: TraiNNer-Redux spanplus_arch expects abbreviated upsampler names
            _UPSAMPLER_ALIASES = {
                "dysample": "dys", "dysample++": "dys",
                "pixelshuffle": "ps", "pixel_shuffle": "ps",
                "convolution": "conv",
            }
            # UI display name → traiNNer ARCH_REGISTRY key (lowercase class name)
            _ARCH_DISPLAY_TO_REGISTRY = {
                "spanpp": "spanc",  # repo=spanpp, class=SpanC, registry="spanc"
            }
            arch_type = _ARCH_DISPLAY_TO_REGISTRY.get(arch, arch)
            net_g = {"type": arch_type}
            for k, v in data.items():
                if k.startswith("dyn_") and v is not None and str(v).strip() != "":
                    field = k[4:]
                    if field in _DYN_NOT_NET:
                        continue
                    if field == "upsampler":
                        v = _UPSAMPLER_ALIASES.get(str(v).lower().strip(), v)
                    # Parse list/tuple notation: "(2, 4)", "[2, 4]", "2, 4" → [2, 4]
                    # SpanPP uses tuple notation (2, 4) for scale_list
                    if isinstance(v, str):
                        _s = v.strip()
                        # Convert tuple (...) or bare "a, b" to bracket [...]
                        if _s.startswith("(") and _s.endswith(")"):
                            _s = "[" + _s[1:-1] + "]"
                        elif not _s.startswith("[") and "," in _s:
                            _s = "[" + _s + "]"
                        if _s.startswith("[") and _s.endswith("]"):
                            try:
                                inner = _s[1:-1]
                                parsed = [int(float(x.strip())) if float(x.strip()).is_integer()
                                          else float(x.strip())
                                          for x in inner.split(",") if x.strip()]
                                net_g[field] = parsed
                                continue
                            except Exception:
                                pass
                    try:
                        f = float(v)
                        net_g[field] = int(f) if f.is_integer() else f
                    except (ValueError, TypeError):
                        net_g[field] = v
            config["network_g"] = net_g

            use_gan = safe_bool("use_gan", False)
            if use_gan:
                config["network_d"] = {"type": data.get("net_d_type", "dunet")}

            # Path
            config["path"] = {
                "param_key_g": None,
                "strict_load_g": True,
                "resume_state": None,
            }
            pretrain = data.get("pretrain_model", "")
            if pretrain:
                config["path"]["pretrain_network_g"] = pretrain.replace("\\", "/")
            resume = data.get("resume_state", "")
            if resume:
                config["path"]["resume_state"] = resume.replace("\\", "/")
            eco_pretrain = data.get("eco_pretrain_path", "")
            if eco_pretrain:
                config["path"]["eco_pretrain_g"] = eco_pretrain.replace("\\", "/")

            # Train
            lr = safe_num("lr", 5e-4, float)
            milestones = []
            ms_str = data.get("milestones", "")
            if ms_str:
                try: milestones = [int(float(x.strip())) for x in str(ms_str).split(",") if x.strip()]
                except Exception: pass
            if not milestones:
                step = total_iter // 4
                milestones = [step * i for i in range(1, 4)]

            sched_type = data.get("scheduler", "MultiStepLR")
            sched_lower = sched_type.lower()
            if "cosineannealinglr" in sched_lower:
                sched_section = {"type": sched_type, "T_max": total_iter, "eta_min": 1e-7}
            elif "exponential" in sched_lower:
                sched_section = {"type": sched_type, "gamma": 0.99}
            elif "plateau" in sched_lower:
                sched_section = {"type": sched_type, "patience": 10, "factor": 0.5}
            elif "steplr" in sched_lower and "multi" not in sched_lower:
                sched_section = {"type": sched_type, "step_size": max(1000, total_iter // 10), "gamma": 0.5}
            else:  # MultiStepLR default
                sched_section = {"type": sched_type, "milestones": milestones, "gamma": 0.5}

            train_section = {
                "ema_decay": safe_num("ema", 0.999, float),
                "ema_power": 0.75,
                "grad_clip": safe_bool("grad_clip", False),
                "eco": safe_bool("eco_mode", False),
                "optim_g": {
                    "type": data.get("optim_g", "AdamW"),
                    "lr": lr,
                    "weight_decay": 0,
                    "betas": [0.9, 0.99],
                },
                "scheduler": sched_section,
                "total_iter": total_iter,
                "warmup_iter": safe_num("warmup_iter", -1, int),
            }

            if use_gan:
                train_section["optim_d"] = {
                    "type": "AdamW",
                    "lr": safe_num("lr_d", lr, float),
                    "weight_decay": 0,
                    "betas": [0.9, 0.99],
                }

            # Losses (Redux list format)
            losses = []
            if safe_bool("loss_pixel", True):
                losses.append({"type": data.get("pixel_criterion", "charbonnierloss"), "loss_weight": safe_num("weight_loss_pixel", 1.0, float)})
            if safe_bool("loss_mssim", False):
                losses.append({"type": "mssimloss", "loss_weight": safe_num("weight_loss_mssim", 0.5, float)})
            if safe_bool("loss_percep", False):
                # Bug fix: only these criterion values are valid for TraiNNer-Redux perceptualloss
                _VALID_PERCEP_CRIT = {"charbonnier", "l1", "pd+l1", "fd+l1", "pd", "fd"}
                crit = data.get("percep_criterion", "charbonnier")
                if crit not in _VALID_PERCEP_CRIT:
                    crit = "l1"
                losses.append({"type": "perceptualloss", "criterion": crit,
                               "loss_weight": safe_num("weight_loss_percep", 0.01, float)})
            if safe_bool("loss_ldl", False):
                losses.append({"type": "ldlloss", "loss_weight": safe_num("weight_loss_ldl", 1.0, float), "criterion": data.get("ldl_criterion", "l1"), "ksize": safe_num("ldl_ksize", 7, int)})
            if safe_bool("loss_dists", False):
                losses.append({"type": "distsloss", "loss_weight": safe_num("weight_loss_dists", 0.3, float)})
            if safe_bool("loss_ff", False):
                losses.append({"type": "ffloss", "loss_weight": safe_num("weight_loss_ff", 0.2, float), "alpha": safe_num("ff_alpha", 1.0, float)})
            if safe_bool("loss_hsluv", False):
                losses.append({"type": "hsluvloss", "loss_weight": safe_num("weight_loss_hsluv", 1.0, float), "hue_weight": safe_num("hsluv_hue_weight", 0.33, float), "saturation_weight": safe_num("hsluv_sat_weight", 0.33, float), "lightness_weight": safe_num("hsluv_lum_weight", 0.33, float)})
            if safe_bool("loss_cosim", False):
                losses.append({"type": "cosimloss", "loss_weight": safe_num("weight_loss_cosim", 1.0, float), "cosim_lambda": safe_num("cosim_lambda", 5, float)})
            if safe_bool("loss_color", False):
                losses.append({"type": "colorloss", "loss_weight": safe_num("weight_loss_color", 1.0, float), "criterion": data.get("color_criterion", "l1")})
            if safe_bool("loss_gv", False):
                losses.append({"type": "gradientvarianceloss", "loss_weight": safe_num("weight_loss_gv", 1.0, float), "patch_size": safe_num("gv_patch_size", 16, int), "criterion": data.get("gv_criterion", "charbonnier")})
            if safe_bool("loss_luma", False):
                losses.append({"type": "lumaloss", "loss_weight": safe_num("weight_loss_luma", 1.0, float), "criterion": data.get("luma_criterion", "l1")})
            if safe_bool("loss_contextual", False):
                losses.append({"type": "contextualloss", "loss_weight": safe_num("weight_loss_contextual", 1.0, float), "distance_type": data.get("ctx_distance_type", "cosine"), "band_width": safe_num("ctx_band_width", 0.5, float)})
            if safe_bool("loss_spark", False):
                _spark_entry = {"type": "SparkLoss", "loss_weight": safe_num("weight_loss_spark", 1.0, float), "criterion": data.get("spark_criterion", "fd")}
                _spark_path = (data.get("spark_path") or "").strip()
                if _spark_path:
                    _spark_entry["path"] = _spark_path
                losses.append(_spark_entry)
            if use_gan:
                losses.append({"type": "ganloss", "gan_type": data.get("gan_type", "vanilla"),
                               "loss_weight": safe_num("gan_loss_weight", 0.1, float)})
            if not losses:
                losses.append({"type": "charbonnierloss", "loss_weight": 1.0})
            train_section["losses"] = losses

            config["train"] = train_section

            # Val
            config["val"] = {
                "val_enabled": safe_bool("val_enabled", True),
                "val_freq": safe_num("val_freq", 1000, int),
                "save_img": safe_bool("save_img", True),
                "tile_size": safe_num("tile", 0, int),
                "tile_overlap": safe_num("tile_pad", 8, int),
                "metrics_enabled": True,
                "metrics": {
                    "psnr": {"type": "calculate_psnr", "crop_border": 4, "test_y_channel": False},
                    "ssim": {"type": "calculate_ssim", "crop_border": 4, "test_y_channel": False},
                },
            }

            # Logger
            config["logger"] = {
                "print_freq": safe_num("print_freq", 100, int),
                "save_checkpoint_freq": safe_num("save_freq", 1000, int),
                "save_checkpoint_format": "safetensors",
                "use_tb_logger": safe_bool("use_tb_logger", True),
            }

            # Monitoring — persisted so checkboxes restore on reload
            config["monitoring"] = {
                "auto_tensorboard": safe_bool("auto_tensorboard", False),
                "port": safe_num("port_tb", 6006, int),
                "auto_ngrok": safe_bool("auto_ngrok", False),
            }

            class _NullTilde(yaml.Dumper):
                pass
            _NullTilde.add_representer(
                type(None),
                lambda dumper, _: dumper.represent_scalar("tag:yaml.org,2002:null", "~")
            )
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, Dumper=_NullTilde, sort_keys=False,
                          allow_unicode=True, default_flow_style=False)
            return True, f"Fichier Redux généré : {os.path.basename(save_path)}"
        except Exception as e:
            return False, f"Erreur Redux : {e}"

    def _generate_neosr_toml(self, data, path):
        def safe_num(k, d, t):
            try: return t(float(data.get(k, d)))
            except Exception: return d
        def safe_bool(k, d):
            val = str(data.get(k, str(d))).lower()
            return val in ["true", "1", "yes", "on"]
        
        def safe_list(k, default):
            val = data.get(k, default)
            if isinstance(val, list): return val
            if isinstance(val, str):
                val = val.strip().strip("[]")
                try: return [float(x.strip()) if '.' in x else int(x.strip()) for x in val.split(',') if x.strip()]
                except Exception: return default
            return default

        def _convert_val(val):
            if isinstance(val, str) and val.startswith("["):
                try: return [int(x) if x.isdigit() else float(x) for x in val.strip("[]").split(",")]
                except Exception: return val
            if isinstance(val, float): return int(val) if val.is_integer() else val
            try:
                if "." in str(val): return float(val)
                return int(val)
            except Exception:
                return True if str(val).lower()=="true" else False if str(val).lower()=="false" else val

        scale = safe_num("scale", 4, int)
        total_iter = safe_num("total_iter", 100000, int)

        # SCHEDULER
        sched_type = data.get("scheduler", "MultiStepLR")
        sched_config = {"type": sched_type}
        sched_lower = str(sched_type).lower()
        if "multistep" in sched_lower:
            ms = data.get("milestones", "75000, 112500")
            try:
                sched_config["milestones"] = [int(x) for x in str(ms).split(",")]
            except Exception:
                sched_config["milestones"] = [75000, 112500]
            sched_config["gamma"] = 0.5
        elif "cosineannealingrestarts" in sched_lower or "cosine" in sched_lower and "warm" in sched_lower:
            sched_config["periods"] = [total_iter // 4] * 4
            sched_config["restart_weights"] = [1, 0.5, 0.25, 0.125]
            sched_config["eta_min"] = 1e-7
        elif "cosine" in sched_lower:
            # CosineAnnealingLR / CosineAnnealing — uses T_max + eta_min, NOT milestones
            sched_config["T_max"] = total_iter
            sched_config["eta_min"] = 1e-7
        elif "step" in sched_lower:
            # StepLR
            sched_config["step_size"] = max(1000, total_iter // 10)
            sched_config["gamma"] = 0.5
        elif "linear" in sched_lower:
            sched_config["start_factor"] = 1.0
            sched_config["end_factor"] = 0.01
            sched_config["total_iters"] = total_iter

        # OPTIMIZER
        optim_type = data.get("optim_g", "AdamW")
        betas_vals = [0.9, 0.99]
        weight_decay_val = 0
        if "Adan" in optim_type:
            betas_vals = [0.98, 0.92, 0.99]
            weight_decay_val = 0.02

        # METRICS
        val_metrics = {}
        if safe_bool("metric_psnr", True): val_metrics["psnr"] = {"type": "calculate_psnr", "crop_border": scale, "test_y_channel": True}
        if safe_bool("metric_ssim", False): val_metrics["ssim"] = {"type": "calculate_ssim", "crop_border": scale, "test_y_channel": True}
        if safe_bool("metric_lpips", False): val_metrics["lpips"] = {"type": "calculate_lpips", "crop_border": scale}
        if safe_bool("metric_niqe", False): val_metrics["niqe"] = {"type": "calculate_niqe", "crop_border": scale}
        if safe_bool("metric_dists", False): val_metrics["dists"] = {"type": "calculate_dists", "better": "lower"}

        # AUGMENTATIONS
        augs = ["none"]; probs = [0.4]
        if safe_bool("aug_mixup", False): augs.append("mixup"); probs.append(round(safe_num("prob_aug_mixup", 0.15, float), 2))
        if safe_bool("aug_cutmix", False): augs.append("cutmix"); probs.append(round(safe_num("prob_aug_cutmix", 0.15, float), 2))
        if safe_bool("aug_resizemix", False): augs.append("resizemix"); probs.append(round(safe_num("prob_aug_resizemix", 0.15, float), 2))
        if safe_bool("aug_cutblur", False): augs.append("cutblur"); probs.append(round(safe_num("prob_aug_cutblur", 0.15, float), 2))

        dataset_mode = data.get("dataset_mode", "otf")
        model_type = "otf" if dataset_mode == "otf" else "image"

        config = {
            "name": data.get("name", "experiment").strip(),
            "model_type": model_type, "scale": scale, 
            "num_gpu": safe_num("num_gpu", 1, int),
            # manual_seed = None → NeoSR désactive le mode déterministe (torch.backends.cudnn.benchmark=True)
            # manual_seed = N   → NeoSR active le mode déterministe + reproducibilité
            "manual_seed": (safe_num("manual_seed", 10, int)
                            if safe_bool("deterministic", True) and
                               str(data.get("manual_seed", "10")).strip() not in ("0", "", "None")
                            else None),
            "use_amp": safe_bool("use_amp", False), "bfloat16": safe_bool("bfloat16", False),
            "fast_matmul": safe_bool("fast_matmul", False),
            "compile": safe_bool("compile", False),
            "monitoring": {
                "auto_tensorboard": safe_bool("auto_tensorboard", False), 
                "port": safe_num("port_tb", 6006, int),
                "auto_ngrok": safe_bool("auto_ngrok", False)
            },
            "degradations": {
                "resize_prob": safe_list("resize_prob", [0.2, 0.7, 0.1]),
                "resize_range": safe_list("resize_range", [0.5, 1.5]),
                "gaussian_noise_prob": safe_num("gaussian_noise_prob", 0.5, float),
                "noise_range": safe_list("noise_range", [1, 25]),
                "poisson_scale_range": safe_list("poisson_scale_range", [0.05, 2.0]),
                "gray_noise_prob": safe_num("gray_noise_prob", 0.4, float),
                "blur_kernel_size": safe_num("blur_kernel_size", 21, int),
                "kernel_list": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
                "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                "sinc_prob": safe_num("sinc_prob", 0.1, float),
                "blur_sigma": safe_list("blur_sigma", [0.2, 3]),
                "betag_range": safe_list("betag_range", [0.5, 4]),
                "betap_range": safe_list("betap_range", [1, 2]),
                
                "second_blur_prob": safe_num("second_blur_prob", 0.8, float),
                "resize_prob2": safe_list("resize_prob2", [0.3, 0.4, 0.3]),
                "resize_range2": safe_list("resize_range2", [0.3, 1.2]),
                "gaussian_noise_prob2": safe_num("gaussian_noise_prob2", 0.5, float),
                "noise_range2": safe_list("noise_range2", [1, 25]),
                "poisson_scale_range2": safe_list("poisson_scale_range2", [0.05, 2.5]),
                "gray_noise_prob2": safe_num("gray_noise_prob2", 0.4, float),
                "blur_kernel_size2": safe_num("blur_kernel_size2", 21, int),
                "kernel_list2": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
                "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                "sinc_prob2": safe_num("sinc_prob2", 0.1, float),
                "blur_sigma2": safe_list("blur_sigma2", [0.2, 1.5]),
                "betag_range2": safe_list("betag_range2", [0.5, 4]),
                "betap_range2": safe_list("betap_range2", [1, 2]),

                "jpeg_range": safe_list("jpeg_range", [30, 95]),
                "jpeg_range2": safe_list("jpeg_range2", [30, 95]),
                "final_sinc_prob": safe_num("final_sinc_prob", 0.8, float),
                "jpeg_prob": safe_num("jpeg_prob", 1.0, float),
                "posterize_prob": safe_num("posterize_prob", 0.0, float),
                "posterize_bits_range": safe_list("posterize_bits_range", [3, 6]),
                "banding_prob": safe_num("banding_prob", 0.0, float),
                "banding_levels_range": safe_list("banding_levels_range", [16, 64]),
                "chroma_prob": safe_num("chroma_prob", 0.0, float),
                "ca_prob": safe_num("ca_prob", 0.0, float),
                "ca_shift_range": safe_list("ca_shift_range", [1, 5]),
                "halation_prob": safe_num("halation_prob", 0.0, float),
                "halation_strength_range": safe_list("halation_strength_range", [0.05, 0.3]),
                "salt_pepper_prob": safe_num("salt_pepper_prob", 0.0, float),
                "salt_pepper_amount_range": safe_list("salt_pepper_amount_range", [0.001, 0.05]),
                "vhs_prob": safe_num("vhs_prob", 0.0, float),
                "vhs_strength_range": safe_list("vhs_strength_range", [0.1, 0.5]),
                "aliasing_prob": safe_num("aliasing_prob", 0.0, float),
                "aliasing_scale_range": safe_list("aliasing_scale_range", [0.5, 0.85]),
                "interlace_weave_prob": safe_num("interlace_weave_prob", 0.0, float),
                "interlace_weave_strength_range": safe_list("interlace_weave_strength_range", [0.5, 1.0]),
                "interlace_flicker_prob": safe_num("interlace_flicker_prob", 0.0, float),
                "interlace_flicker_strength_range": safe_list("interlace_flicker_strength_range", [0.1, 0.4]),
                "interlace_blend_prob": safe_num("interlace_blend_prob", 0.0, float),
                "interlace_blend_strength_range": safe_list("interlace_blend_strength_range", [0.3, 1.0]),
                "film_grain_prob": safe_num("film_grain_prob", 0.0, float),
                "film_grain_strength_range": safe_list("film_grain_strength_range", [0.03, 0.12]),
                "film_grain_size_range": safe_list("film_grain_size_range", [1, 2]),
                "oversharp_prob": safe_num("oversharp_prob", 0.0, float),
                "oversharp_strength_range": safe_list("oversharp_strength_range", [0.5, 2.0]),
                "scanlines_prob": safe_num("scanlines_prob", 0.0, float),
                "scanlines_strength_range": safe_list("scanlines_strength_range", [0.2, 0.5]),
                "scanlines_spacing_range": safe_list("scanlines_spacing_range", [2, 4]),
            },
            "datasets": {
                "train": {
                    "type": dataset_mode, "name": "TrainSet", "dataroot_gt": data.get("dataroot_gt", ""),
                    "num_worker_per_gpu": safe_num("num_worker_per_gpu", 4, int), "prefetch_mode": data.get("prefetch_mode", "cuda"),
                    "batch_size": safe_num("batch_size", 4, int), 
                    "accumulate": safe_num("accumulate", 1, int),
                    "patch_size": safe_num("patch_size", 64, int),
                    "use_shuffle": True, "use_hflip": safe_bool("use_hflip", True), "use_rot": safe_bool("use_rot", True),
                    "augmentation": augs, "aug_prob": probs
                },
                "val": {
                    "name": "ValSet", "type": "paired", "dataroot_gt": data.get("val_gt", ""), "dataroot_lq": data.get("val_lq", ""), "io_backend": {"type": "disk"}
                }
            },
            "network_g": (lambda a: {"type": a, **({} if a in _ARCHS_NO_CHANNELS else {"num_in_ch": 3, "num_out_ch": 3})})(data.get("arch", "omnisr")),
            "path": {"strict_load_g": False, "resume_state": data.get("resume_state", "").strip(), "pretrain_network_g": data.get("pretrain_model", "").strip()},
            "train": {
                "total_iter": total_iter, "n_iter": total_iter,
                "warmup_iter": safe_num("warmup_iter", -1, int),
                "ema": safe_num("ema", 0.999, float),
                "grad_clip": safe_bool("grad_clip", False),
                "eco": safe_bool("eco_mode", False),
                "match_lq_colors": safe_bool("match_lq_colors", False),
                "optim_g": {
                    "type": optim_type, "lr": safe_num("lr", 5e-5, float), 
                    "weight_decay": weight_decay_val, "betas": betas_vals
                },
                "scheduler": sched_config,
            },
            "logger": {"total_iter": total_iter, "print_freq": safe_num("print_freq", 100, int), "save_checkpoint_freq": safe_num("save_freq", 5000, int), "use_tb_logger": safe_bool("use_tb_logger", True)},
            "val": {"val_freq": safe_num("val_freq", 5000, int), "save_img": safe_bool("save_img", True), "pbar": True, "tile": safe_num("tile", 200, int), "tile_pad": safe_num("tile_pad", 32, int), "metrics": val_metrics}
        }

        # Gestion du chemin d'expérience personnalisé
        if data.get("custom_exp_path"):
            config["path"]["results_root"] = data.get("custom_exp_path").replace("\\", "/")

        # Gestion du dossier LQ (seulement si Paired et chemin rempli)
        if data.get("dataroot_lq"):
            config["datasets"]["train"]["dataroot_lq"] = data.get("dataroot_lq").replace("\\", "/")

        # --- FIX: GESTION DE SCHEDULE_FREE ---
        if safe_bool("schedule_free", False):
            config["train"]["optim_g"]["schedule_free"] = True
            config["train"]["optim_g"]["warmup_steps"] = safe_num("warmup_steps", -1, int)
        else:
            config["train"]["optim_g"].pop("schedule_free", None)

        # LOSSES DYNAMIC ADDITION
        if safe_bool("loss_pixel", True):
            _pix_t = data.get("pixel_criterion", "L1Loss")
            config["train"]["pixel_opt"] = {"type": _LOSS_TYPE_MAP.get(_pix_t, _pix_t), "loss_weight": safe_num("weight_loss_pixel", 1.0, float), "reduction": data.get("pixel_reduction", "mean")}

        if safe_bool("loss_wavelet", False): 
            config["train"]["wavelet_guided"] = True
            config["train"]["wavelet_init"] = safe_num("wavelet_init", 10000, int)
            config["train"]["wavelet_opt"] = {"type": "WaveletLoss", "loss_weight": safe_num("weight_loss_wavelet", 1.0, float)}

        if safe_bool("loss_fdl", False): config["train"]["fdl_opt"] = {"type": "fdl_loss", "loss_weight": safe_num("weight_loss_fdl", 1.0, float), "model": data.get("fdl_model", "vgg")}
        if safe_bool("loss_ff", False): config["train"]["ff_opt"] = {"type": "ff_loss", "loss_weight": safe_num("weight_loss_ff", 0.2, float), "alpha": safe_num("ff_alpha", 1.0, float)}
        if safe_bool("loss_ldl", False): config["train"]["ldl_opt"] = {"type": "ldl_loss", "loss_weight": safe_num("weight_loss_ldl", 1.0, float), "criterion": data.get("ldl_criterion", "l1"), "ksize": safe_num("ldl_ksize", 7, int)}
        if safe_bool("loss_consistency", False): config["train"]["consistency_opt"] = {"type": "consistency_loss", "loss_weight": safe_num("weight_loss_consistency", 1.0, float), "use_blur": safe_bool("consistency_blur", True), "use_cosim": safe_bool("consistency_cosim", True), "use_saturation": safe_bool("consistency_saturation", True), "use_brightness": safe_bool("consistency_brightness", True)}
        if safe_bool("loss_edge", False): config["train"]["edge_opt"] = {"type": "EdgeLoss", "loss_weight": safe_num("weight_loss_edge", 0.05, float), "criterion": data.get("edge_criterion", "l1"), "corner": safe_bool("edge_corner", False)}

        if safe_bool("loss_mssim", False): config["train"]["mssim_opt"] = {"type": "mssim_loss", "loss_weight": safe_num("weight_loss_mssim", 1.0, float), "window_size": safe_num("mssim_window_size", 11, int), "sigma": safe_num("mssim_sigma", 1.5, float), "K1": safe_num("mssim_k1", 0.01, float), "K2": safe_num("mssim_k2", 0.03, float)}
        if safe_bool("loss_dists", False): config["train"]["dists_opt"] = {"type": "dists_loss", "loss_weight": safe_num("weight_loss_dists", 1.0, float)}
        if safe_bool("loss_msswd", False): config["train"]["msswd_opt"] = {"type": "msswd_loss", "loss_weight": safe_num("weight_loss_msswd", 1.0, float)}

        if safe_bool("loss_percep", False):
            _vgg_all = ["conv1_2","conv2_2","conv3_2","conv3_4","conv4_2","conv4_4","conv5_2","conv5_4"]
            _vgg_lw = {l: safe_num(f"percep_vgg_{l}", 0.0, float) for l in _vgg_all}
            _vgg_lw = {l: w for l, w in _vgg_lw.items() if w > 0}
            if not _vgg_lw:  # fallback: single-layer legacy
                _vgg_lw = {data.get("percep_layer", "conv5_4"): 1.0}
            config["train"]["perceptual_opt"] = {"type": "vgg_perceptual_loss", "loss_weight": safe_num("weight_loss_percep", 1.0, float), "criterion": data.get("percep_criterion", "huber"), "layer_weights": _vgg_lw}

        # ARCH SPECIFIC
        arch = data.get("arch", "omnisr")
        if arch == "omnisr": config["network_g"]["upsampling"] = scale
        elif arch == "span": config["network_g"]["upscale"] = scale
        elif arch in ["esrgan", "hat", "realplksr", "dat"]: config["network_g"]["scale"] = scale
        for k, v in data.items():
            if k.startswith("dyn_") and k != "dyn_gan_weight": config["network_g"][k.replace("dyn_", "")] = _convert_val(v)

        # GAN
        if safe_bool("use_gan", False):
            config["train"]["gan_opt"] = {"type": "gan_loss", "gan_type": data.get("gan_type", "bce"), "loss_weight": safe_num("dyn_gan_weight", 0.05, float), "real_label_val": safe_num("real_label_val", 1.0, float), "fake_label_val": safe_num("fake_label_val", 0.0, float)}
            d_type = data.get("net_d_type", "unet"); d_config = {"type": d_type}
            if d_type in ["dunet", "metagan"]: d_config["in_ch"] = 3
            elif d_type != "ea2fpn": d_config["num_in_ch"] = 3
            config["network_d"] = d_config
            for k, v in data.items():
                if k.startswith("dynd_"): config["network_d"][k.replace("dynd_", "")] = _convert_val(v)
            config["train"]["optim_d"] = {"type": data.get("optim_g", "AdamW"), "lr": safe_num("lr_d", 5e-5, float), "weight_decay": 0, "betas": [0.9, 0.99]}

        if not config["path"]["resume_state"]: del config["path"]["resume_state"]
        if not config["path"]["pretrain_network_g"]: del config["path"]["pretrain_network_g"]

        try:
            with open(path, "w", encoding="utf-8") as f: toml.dump(config, f)
            return True, f"Fichier NeoSR généré : {os.path.basename(path)}"
        except Exception as e:
            return False, f"Erreur écriture : {e}"