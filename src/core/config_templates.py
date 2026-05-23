"""
config_templates.py — Pre-configured training templates for common scenarios.

Each template returns a dict that can be loaded into the Config tab via
the same data structure as load_action() expects.
"""


def get_template_anime_4x_psnr() -> dict:
    """Anime/Illustration 4x — PSNR phase (good starting point)."""
    return {
        "_template_name": "Anime 4x PSNR",
        "_description": "Optimise pour anime/illustrations. Phase PSNR pour fidelite pixel.",
        # Architecture
        "network_g": "omnisr",
        "scale": 4,
        "ema_decay": 0.999,
        # Optimization
        "optim_g": "AdamW_SF",
        "lr": "3e-4",
        "scheduler": "MultiStepLR",
        "total_iter": 150000,
        # Losses (PSNR-focused)
        "loss_l1": True,
        "loss_l1_weight": 1.0,
        "loss_perceptual": False,
        "loss_gan": False,
        "loss_ff": False,
        # Dataset
        "dataset_mode": "otf",
        "patch_size": 64,
        "batch_size_per_gpu": 4,
        "accumulate": 6,
        # OTF degradations - light for anime
        "blur_prob": 0.3,
        "blur_sigma": "[0.2, 1.5]",
        "gaussian_noise_prob": 0.2,
        "noise_range": "[1, 15]",
        "second_blur_prob": 0.2,
        "gaussian_noise_prob2": 0.1,
        "jpeg_prob": 0.7,
        "jpeg_range": "[60, 95]",
        "jpeg_range2": "[70, 95]",
        "final_sinc_prob": 0.5,
        "gray_noise_prob": 0.1,
    }


def get_template_anime_4x_gan() -> dict:
    """Anime/Illustration 4x — GAN phase (after PSNR pretraining)."""
    return {
        "_template_name": "Anime 4x GAN",
        "_description": "Phase GAN apres PSNR. A utiliser avec un .pth pretrained.",
        "network_g": "omnisr",
        "scale": 4,
        "ema_decay": 0.999,
        "optim_g": "AdamW",
        "lr": "1e-4",
        "lr_d": "5e-5",
        "scheduler": "MultiStepLR",
        "total_iter": 80000,
        "loss_l1": True,
        "loss_l1_weight": 0.5,
        "loss_perceptual": True,
        "loss_perceptual_weight": 1.0,
        "loss_gan": True,
        "loss_gan_weight": 0.1,
        "loss_ff": True,
        "loss_ff_weight": 0.1,
        "dataset_mode": "otf",
        "patch_size": 64,
        "batch_size_per_gpu": 4,
        "accumulate": 6,
        "blur_prob": 0.4,
        "blur_sigma": "[0.2, 2.0]",
        "gaussian_noise_prob": 0.3,
        "noise_range": "[1, 20]",
        "jpeg_prob": 1.0,
        "jpeg_range": "[40, 90]",
        "final_sinc_prob": 0.6,
    }


def get_template_realistic_2x() -> dict:
    """Photo/Realistic 2x upscale."""
    return {
        "_template_name": "Photo Realiste 2x",
        "_description": "Photos realistes scale 2x. Degradations agressives style real-world.",
        "network_g": "rcan",
        "scale": 2,
        "ema_decay": 0.999,
        "optim_g": "AdamW",
        "lr": "2e-4",
        "scheduler": "CosineAnnealingLR",
        "total_iter": 200000,
        "loss_l1": True,
        "loss_l1_weight": 1.0,
        "loss_perceptual": False,
        "loss_gan": False,
        "dataset_mode": "otf",
        "patch_size": 64,
        "batch_size_per_gpu": 8,
        "accumulate": 4,
        "blur_prob": 0.7,
        "blur_sigma": "[0.2, 3.0]",
        "gaussian_noise_prob": 0.5,
        "noise_range": "[1, 30]",
        "second_blur_prob": 0.5,
        "blur_sigma2": "[0.2, 1.5]",
        "gaussian_noise_prob2": 0.4,
        "noise_range2": "[1, 25]",
        "jpeg_prob": 1.0,
        "jpeg_range": "[30, 95]",
        "jpeg_range2": "[30, 95]",
        "final_sinc_prob": 0.8,
        "gray_noise_prob": 0.4,
    }


def get_template_realistic_4x() -> dict:
    """Photo/Realistic 4x upscale."""
    return {
        "_template_name": "Photo Realiste 4x",
        "_description": "Photos realistes scale 4x. Pipeline complet realesrgan.",
        "network_g": "rcan",
        "scale": 4,
        "ema_decay": 0.999,
        "optim_g": "AdamW",
        "lr": "2e-4",
        "scheduler": "MultiStepLR",
        "total_iter": 250000,
        "loss_l1": True,
        "loss_l1_weight": 1.0,
        "loss_perceptual": False,
        "loss_gan": False,
        "dataset_mode": "otf",
        "patch_size": 64,
        "batch_size_per_gpu": 4,
        "accumulate": 6,
        "blur_prob": 0.8,
        "blur_sigma": "[0.2, 3.0]",
        "gaussian_noise_prob": 0.5,
        "noise_range": "[1, 30]",
        "second_blur_prob": 0.6,
        "blur_sigma2": "[0.2, 1.5]",
        "gaussian_noise_prob2": 0.5,
        "noise_range2": "[1, 25]",
        "jpeg_prob": 1.0,
        "jpeg_range": "[30, 95]",
        "jpeg_range2": "[30, 95]",
        "final_sinc_prob": 0.8,
        "gray_noise_prob": 0.4,
    }


def get_template_pixel_art_4x() -> dict:
    """Pixel art / retro game scaling."""
    return {
        "_template_name": "Pixel Art 4x",
        "_description": "Pour pixel art / retro games. Pas de blur, focus sur edges nets.",
        "network_g": "swinir_small",
        "scale": 4,
        "ema_decay": 0.9999,
        "optim_g": "Adam",
        "lr": "5e-4",
        "scheduler": "MultiStepLR",
        "total_iter": 100000,
        "loss_l1": True,
        "loss_l1_weight": 1.0,
        "loss_perceptual": False,
        "loss_gan": False,
        "dataset_mode": "paired",  # Pixel art needs paired (no blur OTF)
        "patch_size": 64,
        "batch_size_per_gpu": 8,
        "accumulate": 2,
        # OTF off for pixel art
        "blur_prob": 0.0,
        "gaussian_noise_prob": 0.0,
        "second_blur_prob": 0.0,
        "gaussian_noise_prob2": 0.0,
        "jpeg_prob": 0.0,
        "final_sinc_prob": 0.0,
        "gray_noise_prob": 0.0,
    }


def get_template_video_compressed_4x() -> dict:
    """Heavily compressed video frames (anime DVD/BluRay rip)."""
    return {
        "_template_name": "Video Compressee 4x",
        "_description": "Anime/Video tres compressee (banding, posterisation, JPEG fort).",
        "network_g": "omnisr",
        "scale": 4,
        "ema_decay": 0.999,
        "optim_g": "AdamW",
        "lr": "2e-4",
        "scheduler": "MultiStepLR",
        "total_iter": 200000,
        "loss_l1": True,
        "loss_l1_weight": 1.0,
        "loss_perceptual": True,
        "loss_perceptual_weight": 0.5,
        "dataset_mode": "otf",
        "patch_size": 64,
        "batch_size_per_gpu": 4,
        "accumulate": 6,
        "blur_prob": 0.6,
        "blur_sigma": "[0.5, 2.5]",
        "gaussian_noise_prob": 0.4,
        "noise_range": "[5, 25]",
        "second_blur_prob": 0.4,
        "gaussian_noise_prob2": 0.3,
        "jpeg_prob": 1.0,
        "jpeg_range": "[20, 70]",  # Aggressive JPEG
        "jpeg_range2": "[20, 70]",
        "final_sinc_prob": 0.9,  # High ringing
        "gray_noise_prob": 0.2,
        # Custom degradations (banding/posterize)
        "posterize_prob": 0.3,
        "posterize_bits_range": "[3, 5]",
        "banding_prob": 0.4,
        "banding_levels_range": "[16, 48]",
    }


# Registry of all templates
TEMPLATES = {
    "Anime 4x PSNR": get_template_anime_4x_psnr,
    "Anime 4x GAN": get_template_anime_4x_gan,
    "Photo Realiste 2x": get_template_realistic_2x,
    "Photo Realiste 4x": get_template_realistic_4x,
    "Pixel Art 4x": get_template_pixel_art_4x,
    "Video Compressee 4x": get_template_video_compressed_4x,
}


def list_templates() -> list:
    """Return list of available template names."""
    return list(TEMPLATES.keys())


def get_template(name: str) -> dict:
    """Get a template by name. Returns empty dict if not found."""
    fn = TEMPLATES.get(name)
    if fn is None:
        return {}
    return fn()
