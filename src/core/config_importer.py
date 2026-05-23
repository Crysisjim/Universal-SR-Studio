"""
config_importer.py — Import standard YAML configs (Real-ESRGAN, SwinIR, BasicSR)
and convert them to the Universal SR Studio format.

Common community configs follow a similar structure based on BasicSR.
"""
import os
import yaml


def detect_config_format(config_path: str) -> str:
    """Detect which config family this YAML belongs to."""
    if not os.path.exists(config_path):
        return "unknown"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return "unknown"

    # Look at network_g.type to identify the format
    net_g = data.get("network_g", {})
    net_type = net_g.get("type", "").lower() if isinstance(net_g, dict) else ""

    if "rrdb" in net_type or "esrgan" in net_type:
        return "real_esrgan"
    if "swinir" in net_type:
        return "swinir"
    if "rcan" in net_type:
        return "rcan"
    if "hat" in net_type:
        return "hat"
    if "omnisr" in net_type:
        return "omnisr"  # Already our format
    return "basicsr"  # Generic BasicSR


def import_yaml_config(config_path: str) -> dict:
    """
    Import a community YAML config and convert it to USR Studio format.

    Returns a dict compatible with our load_action() expectations.
    """
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[Import] Erreur YAML: {e}")
        return {}

    fmt = detect_config_format(config_path)
    result = {}

    # ===== Network =====
    net_g = data.get("network_g", {})
    if isinstance(net_g, dict):
        net_type = net_g.get("type", "")
        # Map community names to our names
        type_map = {
            "RRDBNet": "esrgan",
            "MSRResNet": "esrgan",
            "SwinIR": "swinir_medium",
            "RCAN": "rcan",
            "HAT": "hat",
            "OmniSR": "omnisr",
        }
        result["network_g"] = type_map.get(net_type, net_type.lower())
        result["scale"] = net_g.get("scale") or net_g.get("upscale") or data.get("scale", 4)

    # ===== Optimization =====
    train = data.get("train", {})
    optim_g = train.get("optim_g", {})
    if isinstance(optim_g, dict):
        # Map optimizer types
        opt_type = optim_g.get("type", "Adam")
        opt_map = {"Adam": "Adam", "AdamW": "AdamW", "SGD": "SGD"}
        result["optim_g"] = opt_map.get(opt_type, opt_type)
        result["lr"] = str(optim_g.get("lr", 2e-4))

    optim_d = train.get("optim_d", {})
    if isinstance(optim_d, dict):
        result["lr_d"] = str(optim_d.get("lr", 1e-4))

    # Scheduler
    scheduler = train.get("scheduler", {})
    if isinstance(scheduler, dict):
        sched_type = scheduler.get("type", "MultiStepLR")
        result["scheduler"] = sched_type

    result["total_iter"] = train.get("total_iter") or train.get("niter", 100000)
    result["ema_decay"] = train.get("ema_decay", 0.999)

    # ===== Losses =====
    if "pixel_opt" in train:
        result["loss_l1"] = True
        result["loss_l1_weight"] = train["pixel_opt"].get("loss_weight", 1.0)
    if "perceptual_opt" in train:
        result["loss_perceptual"] = True
        result["loss_perceptual_weight"] = train["perceptual_opt"].get("perceptual_weight", 1.0)
    if "gan_opt" in train:
        result["loss_gan"] = True
        result["loss_gan_weight"] = train["gan_opt"].get("loss_weight", 0.1)
    if "ldl_opt" in train:
        result["loss_ldl"] = True
    if "color_opt" in train:
        result["loss_color"] = True

    # ===== Datasets =====
    datasets = data.get("datasets", {})
    train_ds = datasets.get("train", {})
    if isinstance(train_ds, dict):
        ds_type = train_ds.get("type", "")
        if "RealESRGAN" in ds_type or "OTF" in ds_type or "Realesrgan" in ds_type:
            result["dataset_mode"] = "otf"
        else:
            result["dataset_mode"] = "paired"

        result["batch_size_per_gpu"] = train_ds.get("batch_size_per_gpu", 4)
        result["patch_size"] = train_ds.get("gt_size") or train_ds.get("patch_size", 64)
        if result.get("scale") and result.get("dataset_mode") == "paired":
            # gt_size is at HR scale, patch_size in our app is LR
            result["patch_size"] = result["patch_size"] // result["scale"]

        # Paths
        result["ds_train_gt"] = train_ds.get("dataroot_gt", "")
        result["ds_train_lq"] = train_ds.get("dataroot_lq", "")

    val_ds = datasets.get("val", {})
    if isinstance(val_ds, dict):
        result["ds_val_gt"] = val_ds.get("dataroot_gt", "")
        result["ds_val_lq"] = val_ds.get("dataroot_lq", "")

    # ===== OTF degradations (Real-ESRGAN style) =====
    # Real-ESRGAN puts these at the top level OR under datasets.train
    deg_source = data.get("degradations", {})
    if not deg_source:
        deg_source = train_ds if isinstance(train_ds, dict) else {}

    deg_keys = [
        "blur_kernel_size", "kernel_list", "kernel_prob", "kernel_range",
        "sinc_prob", "blur_sigma", "betag_range", "betap_range",
        "blur_kernel_size2", "kernel_list2", "kernel_prob2", "kernel_range2",
        "sinc_prob2", "blur_sigma2", "betag_range2", "betap_range2",
        "final_sinc_prob", "final_kernel_range",
        "gaussian_noise_prob", "noise_range", "poisson_scale_range", "gray_noise_prob",
        "gaussian_noise_prob2", "noise_range2", "poisson_scale_range2", "gray_noise_prob2",
        "jpeg_prob", "jpeg_range", "jpeg_prob2", "jpeg_range2",
        "resize_prob", "resize_range", "resize_prob2", "resize_range2",
        "blur_prob", "second_blur_prob",
    ]
    for key in deg_keys:
        if key in deg_source:
            result[key] = deg_source[key]

    # ===== Meta =====
    result["_imported_from"] = os.path.basename(config_path)
    result["_format"] = fmt
    result["name"] = data.get("name", "imported_config")

    return result


def get_import_summary(config_path: str) -> str:
    """Get a human-readable summary of what would be imported."""
    if not os.path.exists(config_path):
        return f"Fichier non trouve: {config_path}"
    fmt = detect_config_format(config_path)
    result = import_yaml_config(config_path)
    if not result:
        return "Erreur lors de l'import."
    lines = [
        f"Format detecte : {fmt}",
        f"Nom : {result.get('name', '?')}",
        f"Architecture : {result.get('network_g', '?')}",
        f"Scale : x{result.get('scale', '?')}",
        f"Mode dataset : {result.get('dataset_mode', '?')}",
        f"Optimiseur : {result.get('optim_g', '?')} (lr={result.get('lr', '?')})",
        f"Iterations : {result.get('total_iter', '?')}",
        f"Batch : {result.get('batch_size_per_gpu', '?')}",
        f"Patch : {result.get('patch_size', '?')}",
    ]
    losses = []
    if result.get("loss_l1"):
        losses.append("L1")
    if result.get("loss_perceptual"):
        losses.append("Perceptual")
    if result.get("loss_gan"):
        losses.append("GAN")
    lines.append(f"Losses : {', '.join(losses) if losses else 'aucun'}")
    return "\n".join(lines)
