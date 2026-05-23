"""
wizard_advanced.py — Moteur du Wizard Intelligent pour Universal SR Studio.
Gère la détection GPU, les questions adaptatives, la validation et la génération
de configs pour NeoSR (.toml) ET TraiNNer-Redux (.yml).
"""
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum


# ─── GPU Detection ───────────────────────────────────────────────

# Known GPU capabilities (compute capability → features)
GPU_FEATURES = {
    # Pascal (GTX 10xx) — sm_61
    "6.1": {"amp_fp16": False, "amp_bf16": False, "compile": False, "channels_last": False,
             "note": "Pascal (GTX 10xx) — AMP non supporté, training en FP32 uniquement"},
    # Turing (RTX 20xx) — sm_75
    "7.5": {"amp_fp16": True, "amp_bf16": False, "compile": True, "channels_last": True,
             "note": "Turing (RTX 20xx) — AMP FP16 supporté"},
    # Ampere (RTX 30xx) — sm_80/86
    "8.0": {"amp_fp16": True, "amp_bf16": True, "compile": True, "channels_last": True,
             "note": "Ampere (RTX 30xx) — AMP FP16 + BF16 supportés"},
    "8.6": {"amp_fp16": True, "amp_bf16": True, "compile": True, "channels_last": True,
             "note": "Ampere (RTX 30xx) — AMP FP16 + BF16 supportés"},
    "8.9": {"amp_fp16": True, "amp_bf16": True, "compile": True, "channels_last": True,
             "note": "Ada Lovelace (RTX 40xx) — Toutes les optimisations supportées"},
    # Fallback
    "default": {"amp_fp16": False, "amp_bf16": False, "compile": False, "channels_last": False,
                "note": "GPU inconnu — AMP désactivé par précaution"},
}


@dataclass
class GPUInfo:
    """Informations détaillées sur le GPU détecté."""
    name: str = "Inconnu"
    total_vram_gb: float = 0.0
    compute_capability: tuple = (0, 0)
    driver_version: str = ""

    @property
    def cc_str(self) -> str:
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    @property
    def features(self) -> dict:
        return GPU_FEATURES.get(self.cc_str, GPU_FEATURES.get(
            f"{self.compute_capability[0]}.0", GPU_FEATURES["default"]))

    @property
    def supports_amp(self) -> bool:
        return self.features.get("amp_fp16", False)

    @property
    def supports_bf16(self) -> bool:
        return self.features.get("amp_bf16", False)

    @property
    def supports_compile(self) -> bool:
        return self.features.get("compile", False)

    def is_suitable_for_training(self) -> bool:
        return self.total_vram_gb >= 4.0

    def get_recommended_batch_size(self) -> int:
        if self.total_vram_gb >= 24: return 8
        elif self.total_vram_gb >= 12: return 4
        elif self.total_vram_gb >= 8: return 2
        else: return 1

    def get_recommended_patch_size(self) -> int:
        if self.total_vram_gb >= 24: return 128
        elif self.total_vram_gb >= 12: return 96
        elif self.total_vram_gb >= 8: return 64
        else: return 48

    def get_gpu_summary(self) -> str:
        lines = [
            f"GPU : {self.name}",
            f"VRAM : {self.total_vram_gb:.1f} GB",
            f"Compute Capability : sm_{self.cc_str.replace('.', '')}",
            f"AMP FP16 : {'✅' if self.supports_amp else '❌ Non supporté'}",
            f"AMP BF16 : {'✅' if self.supports_bf16 else '❌'}",
            f"torch.compile : {'✅' if self.supports_compile else '❌'}",
            "",
            self.features.get("note", ""),
        ]
        return "\n".join(lines)


class GPUDetector:
    """Détection automatique du GPU via PyTorch ou nvidia-smi."""

    @staticmethod
    def detect_gpu() -> Optional[GPUInfo]:
        info = GPUDetector._try_torch()
        if info:
            return info
        info = GPUDetector._try_nvidia_smi()
        if info:
            return info
        return None

    @staticmethod
    def _try_torch() -> Optional[GPUInfo]:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                cc = torch.cuda.get_device_capability(0)
                return GPUInfo(
                    name=name, total_vram_gb=round(vram, 1),
                    compute_capability=cc,
                    driver_version=torch.version.cuda or ""
                )
        except Exception:
            pass
        return None

    @staticmethod
    def _try_nvidia_smi() -> Optional[GPUInfo]:
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                   "--format=csv,noheader,nounits"]
            if sys.platform == "win32":
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, startupinfo=si)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 4:
                    name = parts[0].strip()
                    vram_mb = float(parts[1].strip())
                    driver = parts[2].strip()
                    cc_parts = parts[3].strip().split(".")
                    cc = (int(cc_parts[0]), int(cc_parts[1])) if len(cc_parts) == 2 else (0, 0)
                    return GPUInfo(name=name, total_vram_gb=round(vram_mb / 1024, 1),
                                  compute_capability=cc, driver_version=driver)
        except Exception:
            pass
        return None


# ─── Question System ─────────────────────────────────────────────

class QuestionType(Enum):
    CHOICE = "choice"
    SCALE = "scale"
    PATH = "path"
    NUMBER = "number"
    TEXT = "text"
    BOOL = "bool"
    INFO = "info"  # Informational panel, no answer needed


@dataclass
class Question:
    id: str
    text: str
    type: QuestionType
    options: List[str] = field(default_factory=list)
    default: Any = None
    help_text: str = ""
    recommendation: str = ""
    validation: Optional[Callable] = None
    skip_condition: Optional[Callable] = None

    def validate(self, answer: Any) -> bool:
        if self.type == QuestionType.INFO:
            return True
        if self.validation:
            try:
                return self.validation(answer)
            except Exception:
                return False
        if self.type == QuestionType.PATH:
            return isinstance(answer, str) and len(answer.strip()) > 0
        if self.type == QuestionType.NUMBER:
            try:
                float(answer)
                return True
            except (ValueError, TypeError):
                return False
        return True


# ─── Engine-specific knowledge ───────────────────────────────────

# NeoSR architectures
# NeoSR architectures (base names, no _s/_m/_l variants)
NEOSR_ARCHITECTURES = [
    "omnisr", "span", "spanplus", "realplksr", "plksr", "compact", "ultracompact",
    "esrgan", "swinir", "swinir_small", "swinir_medium", "hat", "dat_s", "srformer_medium", "drct",
    "atd", "cugan", "rcan", "safmn", "lmlt", "eimn", "moesr", "flexnet",
    "artcnn_r16f96", "hit_srf", "mosrv2", "craft", "man",
]

# TraiNNer-Redux architectures (from traiNNer-redux wiki arch_reference)
REDUX_ARCHITECTURES = [
    # Recommended
    "omnisr", "span", "realplksr", "compact", "esrgan",
    # Lightweight
    "span_s", "span_f32", "spanplus", "spanplus_s", "superultracompact", "ultracompact",
    "safmn", "safmn_l", "realcugan", "plksr", "plksr_tiny", "realplksr_large",
    "rtmosr", "rtmosr_l", "mosr", "mosr_t", "mosrv2", "moesr2",
    "lmlt_base", "lmlt_large", "lmlt_tiny", "dis_balanced", "dis_fast",
    # Transformers
    "hat_s", "hat_m", "hat_l", "dat", "dat_2", "dat_light", "dat_s",
    "swinir_s", "swinir_m", "swinir_l", "swin2sr_s", "swin2sr_l",
    "srformer", "srformerv2", "drct", "drct_l", "drct_xl",
    "atd", "atd_light", "rgt", "rgt_s",
    # GAN / Restoration
    "esrgan_lite", "rcan", "rcan_l", "rcan_unshuffle",
    "artcnn_r16f96", "artcnn_r8f48", "artcnn_r3f24",
    # Video / Temporal
    "temporalspan", "temporalspanv2", "tscunet",
    # Others
    "autoencoder", "cascadedgaze", "craft", "dctlsa", "ditn_real",
    "dwt", "dwt_s", "eimn_a", "eimn_l", "elan", "elan_light", "emt",
    "escrealm", "escrealm_xl", "fdat", "fdat_light", "fdat_xl",
    "flexnet", "metaflexnet", "gaterv3", "grl_b", "grl_s", "grl_t",
    "hit_sir", "hit_sng", "hit_srf", "lkfmixer_b", "lkfmixer_l", "lkfmixer_t",
    "man", "man_light", "man_tiny", "metagan3",
    "scunet_aaf6aa", "sebica", "sebica_mini", "seemore_t",
]

# NeoSR loss types — from neosr wiki
NEOSR_LOSSES = {
    "L1Loss": {"type": "L1Loss", "loss_weight": 1.0, "reduction": "mean"},
    "MSELoss": {"type": "MSELoss", "loss_weight": 1.0, "reduction": "mean"},
    "HuberLoss": {"type": "HuberLoss", "loss_weight": 1.0},
    "chc": {"type": "chc", "loss_weight": 1.0},
    "vgg_perceptual_loss": {"type": "vgg_perceptual_loss", "loss_weight": 1.0, "criterion": "huber"},
    "dists_loss": {"type": "dists_loss", "loss_weight": 0.5},
    "gan_loss": {"type": "gan_loss", "gan_type": "bce", "loss_weight": 0.3},
    "ldl_loss": {"type": "ldl_loss", "loss_weight": 1.0, "criterion": "huber"},
    "fdl_loss": {"type": "fdl_loss", "loss_weight": 1.0, "model": "dinov2"},
    "ff_loss": {"type": "ff_loss", "loss_weight": 1.0},
    "gw_loss": {"type": "gw_loss", "loss_weight": 1.0, "criterion": "chc_loss"},
    "mssim_loss": {"type": "mssim_loss", "loss_weight": 1.0},
    "ncc_loss": {"type": "ncc_loss", "loss_weight": 1.0},
    "kl_loss": {"type": "kl_loss", "loss_weight": 1.0},
    "consistency_loss": {"type": "consistency_loss", "loss_weight": 1.0},
    "msswd_loss": {"type": "msswd_loss", "loss_weight": 1.0},
}

# Redux loss types (list-based) — from traiNNer-redux wiki
REDUX_LOSSES = {
    "charbonnierloss": {"type": "charbonnierloss", "loss_weight": 1.0},
    "l1loss": {"type": "l1loss", "loss_weight": 1.0},
    "mseloss": {"type": "mseloss", "loss_weight": 1.0},
    "mssimloss": {"type": "mssimloss", "loss_weight": 1.0},
    "perceptualloss": {"type": "perceptualloss", "criterion": "charbonnier", "loss_weight": 0.01},
    "hsluvloss": {"type": "hsluvloss", "criterion": "l1", "loss_weight": 1.0},
    "cosimloss": {"type": "cosimloss", "cosim_lambda": 5, "loss_weight": 1.0},
    "colorloss": {"type": "colorloss", "criterion": "l1", "loss_weight": 1.0},
    "lumaloss": {"type": "lumaloss", "criterion": "l1", "loss_weight": 1.0},
    "ganloss": {"type": "ganloss", "gan_type": "vanilla", "loss_weight": 0.1},
    "ldlloss": {"type": "ldlloss", "criterion": "l1", "loss_weight": 1.0},
    "distsloss": {"type": "distsloss", "loss_weight": 0.5},
    "adistsloss": {"type": "adistsloss", "window_size": 21, "loss_weight": 1.0},
    "ffloss": {"type": "ffloss", "loss_weight": 1.0},
    "fftloss": {"type": "fftloss", "loss_weight": 1.0},
    "gradientvarianceloss": {"type": "gradientvarianceloss", "patch_size": 16, "loss_weight": 1.0},
    "contextualloss": {"type": "contextualloss", "loss_weight": 1.0},
    "nccloss": {"type": "nccloss", "loss_weight": 1.0},
    "bicubicloss": {"type": "bicubicloss", "criterion": "l1", "loss_weight": 1.0},
    "fliploss": {"type": "fliploss", "loss_weight": 1.0},
    "multiscaleganloss": {"type": "multiscaleganloss", "gan_type": "vanilla", "loss_weight": 1.0},
}

# Architecture recommendations
ARCH_RECOMMENDATIONS = {
    "anime": {"qualite": "compact", "vitesse": "omnisr", "equilibre": "span"},
    "photo": {"qualite": "hat_m", "vitesse": "realcugan", "equilibre": "realplksr"},
    "mixte": {"qualite": "span", "vitesse": "omnisr", "equilibre": "realplksr"},
}

# NeoSR discriminators (from neosr wiki)
NEOSR_DISCRIMINATORS = ["unet", "dunet", "patchgan", "metagan", "ea2fpn"]

# Redux discriminators (from traiNNer-redux wiki)
REDUX_DISCRIMINATORS = ["dunet", "unetdiscriminatorsn", "metagan2", "patchgandiscriminatorsn", "multiscalepatchgandiscriminatorsn", "vggstylediscriminator"]


# ─── Safe numeric helpers ────────────────────────────────────────

def _safe_int(val, default=0):
    try: return int(float(val))
    except (ValueError, TypeError): return default

def _safe_float(val, default=0.0):
    try: return float(val)
    except (ValueError, TypeError): return default


# ─── Wizard Engine ───────────────────────────────────────────────

class WizardEngine:
    """Moteur principal du wizard — gère l'état, les questions et la config."""

    def __init__(self):
        self.gpu_info: Optional[GPUInfo] = GPUDetector.detect_gpu()
        self.answers: Dict[str, Any] = {}
        self.questions: List[Question] = self._build_questions()
        self.current_step: int = 0

    @property
    def total_steps(self) -> int:
        return len([q for q in self.questions if not self._should_skip(q)])

    def _should_skip(self, question: Question) -> bool:
        if question.skip_condition:
            try:
                return question.skip_condition(self.answers)
            except Exception:
                return False
        return False

    def get_visible_questions(self) -> List[Question]:
        return [q for q in self.questions if not self._should_skip(q)]

    def get_question(self, index: int) -> Optional[Question]:
        visible = self.get_visible_questions()
        return visible[index] if 0 <= index < len(visible) else None

    def set_answer(self, question_id: str, value: Any):
        self.answers[question_id] = value

    def get_answer(self, question_id: str, default: Any = None) -> Any:
        return self.answers.get(question_id, default)

    def get_recommendation(self, question_id: str) -> str:
        gpu = self.gpu_info
        vram = gpu.total_vram_gb if gpu else 0

        if question_id == "gpu_info" and gpu:
            return gpu.get_gpu_summary()

        if question_id == "engine":
            return ("NeoSR : format TOML, dégradations Real-ESRGAN intégrées, optimiseurs avancés (Adan).\n"
                    "TraiNNer-Redux : format YAML, plus d'architectures, losses modernes (HSLuv, CoSim).")

        if question_id == "batch_size":
            rec = gpu.get_recommended_batch_size() if gpu else 2
            return f"💡 Recommandé pour {gpu.name if gpu else 'votre GPU'} ({vram:.0f} GB) : {rec}"

        if question_id == "patch_size":
            rec = gpu.get_recommended_patch_size() if gpu else 64
            engine = self.get_answer("engine", "NeoSR")
            note = "(= lq_size dans Redux)" if "Redux" in str(engine) else "(= patch_size)"
            return f"💡 Recommandé : {rec} {note}"

        if question_id == "use_amp":
            if gpu and not gpu.supports_amp:
                return (f"⚠️ ATTENTION : Votre {gpu.name} (sm_{gpu.cc_str.replace('.', '')}) "
                        f"ne supporte PAS AMP FP16 avec le PyTorch actuel.\n"
                        f"AMP sera désactivé automatiquement. Training en FP32 (plus lent, plus de VRAM).")
            elif gpu and vram < 12:
                return "💡 Fortement recommandé — réduit la VRAM ~30%"
            return "💡 Recommandé — accélère le training sans perte de qualité"

        if question_id == "use_gan":
            if vram < 8:
                return "⚠️ GAN déconseillé avec moins de 8 GB VRAM (ajoute ~30%)"
            return "💡 Le GAN améliore la netteté perçue mais augmente VRAM ~30% et complexité"

        if question_id == "arch":
            content = self.get_answer("content_type", "mixte")
            objective = self.get_answer("objective", "equilibre")
            rec = ARCH_RECOMMENDATIONS.get(content, {}).get(objective, "span")
            return f"💡 Recommandé pour {content}/{objective} : {rec}"

        if question_id == "dataset_mode":
            return ("Paired : vous fournissez les images HQ + LQ pré-générées.\n"
                    "OTF (On-The-Fly) : seules les images HQ sont requises, "
                    "le moteur génère les dégradations pendant le training.")

        return ""

    def estimate_training_time(self) -> Dict[str, Any]:
        iters = _safe_int(self.get_answer("iterations", 100000), 100000)
        bs = _safe_int(self.get_answer("batch_size", 4), 4)
        patch = _safe_int(self.get_answer("patch_size", 64), 64)
        use_gan = self.get_answer("use_gan", False)
        use_amp = self.get_answer("use_amp", True)

        # Force AMP off if GPU doesn't support it
        if self.gpu_info and not self.gpu_info.supports_amp:
            use_amp = False

        base_speed = 2.0
        if self.gpu_info:
            vram = self.gpu_info.total_vram_gb
            if vram >= 24: base_speed = 5.0
            elif vram >= 12: base_speed = 3.0
            elif vram >= 8: base_speed = 1.5
            else: base_speed = 0.8

        speed = base_speed
        speed *= (4 / max(bs, 1))
        speed *= (64 / max(patch, 32)) ** 2
        if use_gan: speed *= 0.7
        if use_amp: speed *= 1.3

        total_seconds = iters / max(speed, 0.01)
        hours = total_seconds / 3600

        return {
            "iterations": iters,
            "estimated_speed": round(speed, 1),
            "estimated_hours": round(hours, 1),
            "estimated_days": round(hours / 24, 1),
            "readable": f"~{hours:.0f}h" if hours < 48 else f"~{hours/24:.1f} jours",
        }

    # ─── Config Generation ───────────────────────────────────

    def generate_config(self) -> Dict[str, Any]:
        """Génère la config dans le format du moteur choisi."""
        engine = str(self.get_answer("engine", "NeoSR"))
        print(f"[Wizard] Generating config for engine: {engine}")
        print(f"[Wizard] Answers: engine={engine}, arch={self.get_answer('arch')}, "
              f"optimizer={self.get_answer('optimizer')}, scheduler={self.get_answer('scheduler')}")
        if "Redux" in engine:
            return self._generate_redux_config()
        else:
            return self._generate_neosr_config()

    def get_config_extension(self) -> str:
        engine = str(self.get_answer("engine", "NeoSR"))
        return ".yml" if "Redux" in engine else ".toml"

    def _generate_neosr_config(self) -> Dict[str, Any]:
        """Génère une config NeoSR au format TOML (dict plat avec sections)."""
        a = self.answers
        arch = str(a.get("arch", "span"))
        scale = _safe_int(a.get("scale", 4), 4)
        bs = _safe_int(a.get("batch_size", 4), 4)
        patch = _safe_int(a.get("patch_size", 64), 64)
        iters = _safe_int(a.get("iterations", 100000), 100000)
        use_gan = a.get("use_gan", False)
        use_amp = a.get("use_amp", True)
        lr = _safe_float(a.get("learning_rate", 2e-4), 2e-4)
        exp_name = str(a.get("experiment_name", f"sr_{arch}_{scale}x"))
        dataset_mode = str(a.get("dataset_mode", "otf"))
        dataroot_gt = str(a.get("dataset_gt", "datasets/train/HR"))
        dataroot_lq = str(a.get("dataset_lq", ""))

        # Force AMP off on unsupported GPUs
        if self.gpu_info and not self.gpu_info.supports_amp:
            use_amp = False

        config = {
            "name": exp_name,
            "model_type": dataset_mode,
            "scale": scale,
            "num_gpu": 1,
            "manual_seed": 10,
            "use_amp": use_amp,
            "bfloat16": False,
            "fast_matmul": False,
            "compile": False,
            "monitoring": {
                "auto_tensorboard": True,
                "port": 6006,
                "auto_ngrok": False,
            },
            "network_g": {
                "type": arch,
                "num_in_ch": 3,
                "num_out_ch": 3,
                "scale": scale,
            },
            "path": {"strict_load_g": False},
            "train": {
                "total_iter": iters,
                "n_iter": iters,
                "warmup_iter": min(iters // 20, 5000) if iters > 10000 else -1,
                "ema": 0.999,
                "grad_clip": True,
                "optim_g": {
                    "type": a.get("optimizer", "AdamW"),
                    "lr": lr,
                    "weight_decay": 0,
                    "betas": [0.98, 0.92, 0.99] if "Adan" in str(a.get("optimizer", "")) else [0.9, 0.99],
                },
                "scheduler": {
                    "type": a.get("scheduler", "CosineAnnealing"),
                    "T_max": iters,
                    "eta_min": 1e-7,
                },
                "pixel_opt": {"type": "L1Loss", "loss_weight": 1.0, "reduction": "mean"},
            },
            "logger": {
                "total_iter": iters,
                "print_freq": 100,
                "save_checkpoint_freq": max(iters // 20, 2000),
                "use_tb_logger": True,
            },
            "val": {
                "val_freq": max(iters // 20, 1000),
                "save_img": True,
                "pbar": True,
                "tile": 128,
                "tile_pad": 64,
            },
            "datasets": {
                "train": {
                    "type": dataset_mode,
                    "name": "TrainSet",
                    "dataroot_gt": dataroot_gt,
                    "num_worker_per_gpu": 4,
                    "prefetch_mode": "cuda",
                    "batch_size": bs,
                    "accumulate": max(1, 8 // bs),
                    "patch_size": patch,
                    "use_shuffle": True,
                    "augmentation": ["none", "mixup", "cutmix", "resizemix", "cutblur"],
                    "aug_prob": [0.4, 0.2, 0.2, 0.2, 0.4],
                },
                "val": {
                    "name": "ValSet",
                    "type": "paired",
                    "dataroot_gt": "datasets/val/GT",
                    "dataroot_lq": "datasets/val/LQ",
                    "io_backend": {"type": "disk"},
                },
            },
        }

        # Paired mode: add LQ path
        if dataset_mode == "paired" and dataroot_lq:
            config["datasets"]["train"]["dataroot_lq"] = dataroot_lq

        # OTF degradations
        if dataset_mode == "otf":
            config["degradations"] = {
                "resize_prob": [0.2, 0.7, 0.1],
                "resize_range": [0.5, 1.5],
                "gaussian_noise_prob": 0.3,
                "noise_range": [1, 25],
                "poisson_scale_range": [0.05, 2.0],
                "gray_noise_prob": 0.4,
                "blur_kernel_size": 21,
                "kernel_list": ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
                "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                "sinc_prob": 0.1,
                "blur_sigma": [0.2, 2.75],
                "betag_range": [0.5, 4],
                "betap_range": [1, 2],
                "second_blur_prob": 0.57,
                "resize_prob2": [0.3, 0.4, 0.3],
                "resize_range2": [0.3, 1.2],
                "gaussian_noise_prob2": 0.32,
                "noise_range2": [5, 25],
                "poisson_scale_range2": [0.05, 2.5],
                "gray_noise_prob2": 0.4,
                "blur_kernel_size2": 21,
                "kernel_list2": ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
                "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                "sinc_prob2": 0.1,
                "blur_sigma2": [0.2, 2.75],
                "betag_range2": [0.5, 4],
                "betap_range2": [1, 2],
                "jpeg_range": [30, 95],
                "jpeg_range2": [60, 95],
                "final_sinc_prob": 0.75,
                "jpeg_prob": 0.78,
            }

        # Losses
        objective = str(a.get("objective", "equilibre"))
        if objective in ("qualite", "equilibre"):
            config["train"]["perceptual_opt"] = {
                "type": "vgg_perceptual_loss", "loss_weight": 1.0,
                "criterion": "huber", "layer_weights": {"conv4_4": 1.0}
            }
            config["train"]["ldl_opt"] = {"type": "ldl_loss", "loss_weight": 1.0}
            config["train"]["fdl_opt"] = {"type": "fdl_loss", "loss_weight": 0.5, "model": "vgg"}

        # GAN
        if use_gan:
            config["train"]["pixel_opt"]["loss_weight"] = 0.01
            config["train"]["gan_opt"] = {
                "type": "gan_loss", "gan_type": "bce",
                "loss_weight": 0.05, "real_label_val": 1.0, "fake_label_val": 0.0,
            }
            config["network_d"] = {
                "type": str(a.get("discriminator_type", "unet")),
                "num_in_ch": 3, "num_feat": 64, "skip_connection": True,
            }
            config["train"]["optim_d"] = {
                "type": "Adam", "lr": lr, "weight_decay": 0, "betas": [0.9, 0.99],
            }

        # Metrics
        config["val"]["metrics"] = {
            "psnr": {"type": "calculate_psnr", "crop_border": 4, "test_y_channel": True},
            "ssim": {"type": "calculate_ssim", "crop_border": 4, "test_y_channel": True},
        }

        # Schedule-Free optimizer handling
        opt_type = str(a.get("optimizer", "AdamW"))
        if opt_type.endswith("_SF"):
            config["train"]["optim_g"]["schedule_free"] = True
            config["train"]["optim_g"]["warmup_steps"] = min(iters // 10, 2000)
            # SF optimizers don't use external scheduler
            config["train"]["scheduler"] = {"type": "MultiStepLR", "milestones": [iters], "gamma": 1.0}

        return config

    def _generate_redux_config(self) -> Dict[str, Any]:
        """Génère une config TraiNNer-Redux au format YAML."""
        a = self.answers
        arch = str(a.get("arch", "omnisr"))
        scale = _safe_int(a.get("scale", 4), 4)
        bs = _safe_int(a.get("batch_size", 4), 4)
        lq_size = _safe_int(a.get("patch_size", 64), 64)
        iters = _safe_int(a.get("iterations", 100000), 100000)
        use_gan = a.get("use_gan", False)
        use_amp = a.get("use_amp", True)
        lr = _safe_float(a.get("learning_rate", 5e-4), 5e-4)
        exp_name = str(a.get("experiment_name", f"4x_{arch}"))
        dataset_mode = str(a.get("dataset_mode", "otf"))
        dataroot_gt = str(a.get("dataset_gt", "datasets/train/dataset1/hr"))
        dataroot_lq = str(a.get("dataset_lq", ""))

        # Force AMP off on unsupported GPUs
        if self.gpu_info and not self.gpu_info.supports_amp:
            use_amp = False

        config = {
            "name": exp_name,
            "scale": scale,
            "use_amp": use_amp,
            "amp_bf16": False,
            "use_channels_last": False,
            "fast_matmul": False,
            "use_compile": False,
            "compile_mode": "default",
            "num_gpu": "auto",
        }

        # OTF mode
        is_otf = (dataset_mode == "otf")
        if is_otf:
            config["high_order_degradation"] = True
            config["high_order_degradations_debug"] = False
            config["lq_usm"] = False
            config.update({
                "resize_prob": [0.2, 0.7, 0.1],
                "resize_mode_list": ["bilinear", "bicubic", "nearest-exact", "lanczos"],
                "resize_mode_prob": [0.25, 0.25, 0.25, 0.25],
                "resize_range": [0.4, 1.5],
                "gaussian_noise_prob": 0.0,
                "noise_range": [0, 0],
                "poisson_scale_range": [0, 0],
                "gray_noise_prob": 0.0,
                "jpeg_prob": 1.0,
                "jpeg_range": [75, 95],
                "blur_prob": 0.0,
                "resize_prob2": [0.3, 0.4, 0.3],
                "resize_mode_list2": ["bilinear", "bicubic", "nearest-exact", "lanczos"],
                "resize_mode_prob2": [0.25, 0.25, 0.25, 0.25],
                "resize_range2": [0.6, 1.2],
                "gaussian_noise_prob2": 0,
                "noise_range2": [0, 0],
                "poisson_scale_range2": [0, 0],
                "gray_noise_prob2": 0.0,
                "jpeg_prob2": 1.0,
                "jpeg_range2": [75, 95],
                "queue_size": 120,
            })

        # Datasets
        train_type = "realesrgandataset" if is_otf else "pairedimagedataset"
        train_ds = {
            "name": "Train Dataset",
            "type": train_type,
            "dataroot_gt": [dataroot_gt],
            "lq_size": lq_size,
            "use_hflip": True,
            "use_rot": True,
            "num_worker_per_gpu": 8,
            "batch_size_per_gpu": bs,
            "accum_iter": 1,
        }
        if not is_otf and dataroot_lq:
            train_ds["dataroot_lq"] = [dataroot_lq]

        if is_otf:
            train_ds["blur_kernel_size"] = 12
            train_ds["kernel_list"] = ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"]
            train_ds["kernel_prob"] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
            train_ds["kernel_range"] = [5, 17]
            train_ds["sinc_prob"] = 0.0
            train_ds["blur_sigma"] = [0.2, 2]
            train_ds["betag_range"] = [0.5, 4]
            train_ds["betap_range"] = [1, 2]
            train_ds["blur_kernel_size2"] = 12
            train_ds["kernel_list2"] = train_ds["kernel_list"]
            train_ds["kernel_prob2"] = train_ds["kernel_prob"]
            train_ds["kernel_range2"] = [5, 17]
            train_ds["sinc_prob2"] = 0.0
            train_ds["blur_sigma2"] = [0.2, 1]
            train_ds["betag_range2"] = [0.5, 4]
            train_ds["betap_range2"] = [1, 2]
            train_ds["final_sinc_prob"] = 0.0
            train_ds["final_kernel_range"] = [5, 17]

        config["datasets"] = {
            "train": train_ds,
            "val": {
                "name": "Val Dataset",
                "type": "pairedimagedataset",
                "dataroot_gt": ["datasets/val/dataset1/hr"],
                "dataroot_lq": ["datasets/val/dataset1/lr"],
            },
        }

        # Network
        config["network_g"] = {"type": arch}

        if use_gan:
            disc_type = str(a.get("discriminator_type", "dunet"))
            config["network_d"] = {"type": disc_type}

        # Path
        config["path"] = {
            "param_key_g": None,
            "strict_load_g": True,
            "resume_state": None,
        }

        # Training
        milestones = []
        step = iters // 4
        for i in range(1, 4):
            milestones.append(step * i)

        train_section = {
            "ema_decay": 0.999,
            "grad_clip": False,
            "optim_g": {
                "type": a.get("optimizer", "AdamW"),
                "lr": lr,
                "weight_decay": 0,
                "betas": [0.9, 0.99],
            },
            "scheduler": {
                "type": a.get("scheduler", "MultiStepLR"),
                "milestones": milestones,
                "gamma": 0.5,
            },
            "total_iter": iters,
            "warmup_iter": -1,
        }

        if not use_gan:
            train_section["ema_power"] = 0.75

        # Losses (Redux uses list format)
        losses = []
        objective = str(a.get("objective", "equilibre"))
        if use_gan:
            losses.append({"type": "mssimloss", "loss_weight": 0.5})
            losses.append({"type": "perceptualloss", "criterion": "charbonnier", "loss_weight": 0.01})
            losses.append({"type": "hsluvloss", "criterion": "charbonnier", "loss_weight": 1.0})
            losses.append({"type": "cosimloss", "loss_weight": 1.0})
            losses.append({"type": "ganloss", "gan_type": "vanilla", "loss_weight": 0.1})
            # Optim D
            train_section["optim_d"] = {
                "type": "AdamW", "lr": lr, "weight_decay": 0, "betas": [0.9, 0.99],
            }
        else:
            losses.append({"type": "charbonnierloss", "loss_weight": 1.0})

        train_section["losses"] = losses

        # MoA (augmentations)
        if use_gan:
            train_section["use_moa"] = False
            train_section["moa_augs"] = ["none", "mixup", "cutmix", "resizemix", "cutblur"]
            train_section["moa_probs"] = [0.4, 0.084, 0.084, 0.084, 0.348]

        config["train"] = train_section

        # Validation
        config["val"] = {
            "val_enabled": False,
            "val_freq": max(iters // 20, 1000),
            "save_img": True,
            "tile_size": 0,
            "tile_overlap": 8,
            "metrics_enabled": True,
            "metrics": {
                "psnr": {"type": "calculate_psnr", "crop_border": 4, "test_y_channel": False},
                "ssim": {"type": "calculate_ssim", "crop_border": 4, "test_y_channel": False},
            },
        }

        # Logger
        config["logger"] = {
            "print_freq": 100,
            "save_checkpoint_freq": max(iters // 20, 1000),
            "save_checkpoint_format": "safetensors",
            "use_tb_logger": True,
        }

        return config

    # ─── Questions ───────────────────────────────────────────

    def _build_questions(self) -> List[Question]:
        gpu = self.gpu_info

        questions = [
            # 0. GPU Info (informational)
            Question(
                id="gpu_info", text="Détection de votre matériel",
                type=QuestionType.INFO,
                help_text=(gpu.get_gpu_summary() if gpu else
                           "⚠️ Aucun GPU NVIDIA détecté.\nLe training SR nécessite un GPU NVIDIA avec CUDA.\n"
                           "Vérifiez votre installation PyTorch + CUDA."),
                default="ok",
            ),
            # 1. Engine
            Question(
                id="engine",
                text="Quel moteur d'entraînement voulez-vous utiliser ?",
                type=QuestionType.CHOICE,
                options=["NeoSR", "TraiNNer-Redux"],
                default="NeoSR",
                help_text=(
                    "NeoSR : format TOML, dégradations Real-ESRGAN, optimiseurs avancés (Adan, Adam).\n"
                    "  → Fichiers : .toml\n\n"
                    "TraiNNer-Redux : format YAML, architectures modernes (TemporalSPAN), losses avancées.\n"
                    "  → Fichiers : .yml\n\n"
                    "Les deux moteurs produisent des modèles compatibles (PyTorch .pth/.safetensors)."
                ),
            ),
            # 2. Content type
            Question(
                id="content_type",
                text="Quel type de contenu voulez-vous upscaler ?",
                type=QuestionType.CHOICE,
                options=["anime", "photo", "mixte"],
                default="mixte",
                help_text=(
                    "Anime : lignes nettes, aplats de couleur, peu de textures naturelles.\n"
                    "Photo : textures riches, dégradés subtils, détails fins.\n"
                    "Mixte : les deux types, bon compromis polyvalent."
                ),
            ),
            # 3. Scale
            Question(
                id="scale",
                text="Quel facteur d'upscale ?",
                type=QuestionType.CHOICE,
                options=[],  # Filled dynamically based on engine
                default="4",
                help_text=(
                    "1x : même résolution (restauration/denoise uniquement).\n"
                    "2x : double la résolution (ex: 480p → 960p).\n"
                    "3x : triple (plus rare, certaines architectures seulement).\n"
                    "4x : quadruple (ex: 480p → 1920p). Le plus courant.\n"
                    "8x : octuple (Redux uniquement, très lourd en VRAM)."
                ),
            ),
            # 4. Objective
            Question(
                id="objective",
                text="Quel est votre objectif principal ?",
                type=QuestionType.CHOICE,
                options=["qualite", "vitesse", "equilibre"],
                default="equilibre",
                help_text=(
                    "Qualité : meilleur résultat visuel, modèle plus lourd, training plus long.\n"
                    "Vitesse : modèle léger pour inférence rapide (temps réel, vidéo).\n"
                    "Équilibre : bon compromis qualité/vitesse pour la majorité des cas."
                ),
            ),
            # 5. Architecture
            Question(
                id="arch",
                text="Quelle architecture réseau utiliser ?",
                type=QuestionType.CHOICE,
                options=[],  # Filled dynamically based on engine
                default="span",
                help_text=(
                    "L'architecture définit la structure du réseau de neurones.\n"
                    "SPAN/Compact : léger, rapide (6-8 GB VRAM)\n"
                    "OmniSR/RealPLKSR : moyen, bon rapport qualité/vitesse (8-11 GB)\n"
                    "HAT/DAT : lourd, haute qualité (12-24 GB)\n"
                    "ESRGAN : classique, flexible, compatible GAN"
                ),
            ),
            # 6. Dataset mode
            Question(
                id="dataset_mode",
                text="Mode de dataset ?",
                type=QuestionType.CHOICE,
                options=["otf", "paired"],
                default="otf",
                help_text=(
                    "OTF (On-The-Fly) : Vous ne fournissez que les images HQ.\n"
                    "  Le moteur génère les dégradations (bruit, flou, compression) pendant le training.\n"
                    "  ✅ Plus simple, plus varié, recommandé pour débuter.\n\n"
                    "Paired : Vous fournissez des paires HQ + LQ pré-générées.\n"
                    "  ✅ Contrôle total sur les dégradations, reproductible."
                ),
            ),
            # 7. Dataset GT
            Question(
                id="dataset_gt",
                text="Dossier des images haute qualité (GT/HQ) :",
                type=QuestionType.PATH,
                help_text="Le dossier contenant vos images sources en haute résolution pour le training.",
            ),
            # 8. Dataset LQ (only for paired mode)
            Question(
                id="dataset_lq",
                text="Dossier des images basse qualité (LQ) :",
                type=QuestionType.PATH,
                help_text="Le dossier des images dégradées correspondantes.",
                skip_condition=lambda a: a.get("dataset_mode", "otf") != "paired",
            ),
            # 9. Batch size
            Question(
                id="batch_size",
                text="Batch size (images simultanées par itération) :",
                type=QuestionType.NUMBER,
                default=str(gpu.get_recommended_batch_size() if gpu else 4),
                options=["1", "2", "4", "8"],
                help_text=(
                    "Nombre d'images traitées en parallèle à chaque itération.\n"
                    "Plus gros = convergence plus stable mais plus de VRAM.\n"
                    "Si VRAM insuffisante, réduisez ou utilisez l'accumulation de gradients."
                ),
                validation=lambda v: 1 <= _safe_int(v) <= 32,
            ),
            # 10. Patch size
            Question(
                id="patch_size",
                text="Patch/LQ size (taille des crops d'entraînement) :",
                type=QuestionType.NUMBER,
                default=str(gpu.get_recommended_patch_size() if gpu else 64),
                options=["48", "64", "96", "128"],
                help_text=(
                    "Taille des sous-images extraites pour le training.\n"
                    "Plus grand = le réseau voit plus de contexte = meilleure qualité.\n"
                    "Mais consomme beaucoup plus de VRAM (quadratique).\n"
                    "NeoSR : c'est 'patch_size'. Redux : c'est 'lq_size'."
                ),
                validation=lambda v: _safe_int(v) in (32, 48, 64, 96, 128, 192, 256),
            ),
            # 11. Iterations
            Question(
                id="iterations",
                text="Nombre d'itérations de training :",
                type=QuestionType.NUMBER,
                default="100000",
                options=["50000", "100000", "200000", "500000"],
                help_text=(
                    "Plus d'itérations = meilleur résultat mais plus long.\n"
                    "50K : test rapide / fine-tuning court.\n"
                    "100K-200K : training standard.\n"
                    "500K+ : entraînement complet from scratch (plusieurs jours)."
                ),
                validation=lambda v: 1000 <= _safe_int(v) <= 2000000,
            ),
            # 12. GAN
            Question(
                id="use_gan",
                text="Utiliser un GAN (discriminateur) ?",
                type=QuestionType.BOOL,
                default=False,
                help_text=(
                    "Le GAN (Generative Adversarial Network) ajoute un réseau « juge » qui pousse\n"
                    "le générateur à produire des images plus réalistes et nettes.\n\n"
                    "✅ Avantages : textures plus nettes, détails plus fins\n"
                    "❌ Inconvénients : +30% VRAM, training plus instable, risque d'artefacts\n\n"
                    "Recommandé en 2ème phase (fine-tune) après un training PSNR initial."
                ),
            ),
            # 13. Discriminator type (skip if no GAN)
            Question(
                id="discriminator_type",
                text="Type de discriminateur :",
                type=QuestionType.CHOICE,
                options=[],  # Filled dynamically
                default="unet",
                help_text="Le type de réseau discriminateur pour le GAN.",
                skip_condition=lambda a: not a.get("use_gan", False),
            ),
            # 14. AMP
            Question(
                id="use_amp",
                text="Utiliser Mixed Precision (AMP) ?",
                type=QuestionType.BOOL,
                default=True if (gpu and gpu.supports_amp) else False,
                help_text=(
                    "AMP (Automatic Mixed Precision) utilise FP16 au lieu de FP32.\n"
                    "Réduit la VRAM ~30% et accélère le training ~20-40%.\n\n"
                    "⚠️ Nécessite un GPU avec Tensor Cores (RTX 20xx+, sm_75+).\n"
                    "Les GPU Pascal (GTX 10xx) ne supportent PAS AMP."
                ),
            ),
            # 15. Learning rate
            Question(
                id="learning_rate",
                text="Learning rate :",
                type=QuestionType.NUMBER,
                default="0.0002",
                options=["0.0001", "0.0002", "0.0005", "0.001"],
                help_text=(
                    "Vitesse d'apprentissage du réseau.\n"
                    "1e-4 (0.0001) : standard, stable pour la plupart des cas.\n"
                    "2e-4 (0.0002) : bon défaut pour Adam/AdamW.\n"
                    "5e-4 (0.0005) : agressif, convergence rapide, risque d'instabilité.\n"
                    "1e-3 (0.001) : recommandé pour SOAP_SF (NeoSR) uniquement.\n"
                    "1e-5 : conservateur, pour fine-tuning d'un modèle existant."
                ),
                validation=lambda v: 1e-7 <= _safe_float(v) <= 0.01,
            ),
            # 15b. Optimizer
            Question(
                id="optimizer",
                text="Optimiseur :",
                type=QuestionType.CHOICE,
                options=[],  # Filled dynamically based on engine
                default="AdamW",
                help_text=(
                    "L'optimiseur contrôle comment les poids du réseau sont mis à jour.\n\n"
                    "NeoSR :\n"
                    "  Adam/AdamW : standards, fiables, bons pour débuter.\n"
                    "  Adan : convergence rapide, bon pour le SR. 3 betas au lieu de 2.\n"
                    "  AdamW_Win : variante accélérée d'AdamW.\n"
                    "  AdamW_SF / Adan_SF / SOAP_SF : Schedule-Free — pas besoin de scheduler.\n"
                    "    ⚠️ Nécessite 'schedule_free = true' dans la config.\n\n"
                    "TraiNNer-Redux :\n"
                    "  Adam/AdamW : standards PyTorch, fiables.\n"
                    "  NAdam : Adam avec momentum de Nesterov.\n"
                    "  RAdam : Adam rectifié, plus stable au démarrage."
                ),
            ),
            # 15c. Scheduler
            Question(
                id="scheduler",
                text="Scheduler (planification du learning rate) :",
                type=QuestionType.CHOICE,
                options=[],  # Filled dynamically based on engine
                default="MultiStepLR",
                help_text=(
                    "Le scheduler réduit le learning rate pendant le training pour affiner la convergence.\n\n"
                    "MultiStepLR : baisse le LR par paliers (milestones). Simple et efficace.\n"
                    "CosineAnnealing : descente en cosinus, plus douce. Bon pour les longs trainings.\n\n"
                    "⚠️ Si vous utilisez un optimiseur Schedule-Free (SF),\n"
                    "le scheduler est ignoré (il est intégré dans l'optimiseur)."
                ),
                skip_condition=lambda a: str(a.get("optimizer", "")).endswith("_SF"),
            ),
            # 16. Experiment name
            Question(
                id="experiment_name",
                text="Nom de l'expérience :",
                type=QuestionType.TEXT,
                default="my_sr_model",
                help_text="Ce nom sera utilisé pour les dossiers de checkpoints, logs et résultats.",
                validation=lambda v: len(str(v).strip()) > 0,
            ),
        ]

        return questions

    def get_arch_options(self) -> List[str]:
        """Retourne les architectures disponibles pour le moteur sélectionné."""
        engine = str(self.get_answer("engine", "NeoSR"))
        if "Redux" in engine:
            return REDUX_ARCHITECTURES
        return NEOSR_ARCHITECTURES

    def get_discriminator_options(self) -> List[str]:
        """Retourne les discriminateurs disponibles pour le moteur sélectionné."""
        engine = str(self.get_answer("engine", "NeoSR"))
        if "Redux" in engine:
            return REDUX_DISCRIMINATORS
        return NEOSR_DISCRIMINATORS

    def get_scale_options(self) -> List[str]:
        """Retourne les scales disponibles pour le moteur sélectionné."""
        engine = str(self.get_answer("engine", "NeoSR"))
        if "Redux" in engine:
            return ["1", "2", "3", "4", "8"]
        return ["1", "2", "3", "4", "6", "8"]

    def get_optimizer_options(self) -> List[str]:
        """Retourne les optimiseurs disponibles pour le moteur sélectionné."""
        engine = str(self.get_answer("engine", "NeoSR"))
        if "Redux" in engine:
            return ["Adam", "AdamW", "NAdam", "SGD", "RAdam", "Adadelta", "Adagrad"]
        return ["Adam", "AdamW", "NAdam", "Adan", "AdamW_Win", "AdamW_SF", "Adan_SF", "SOAP_SF"]

    def get_scheduler_options(self) -> List[str]:
        """Retourne les schedulers disponibles pour le moteur sélectionné."""
        engine = str(self.get_answer("engine", "NeoSR"))
        if "Redux" in engine:
            return ["MultiStepLR", "CosineAnnealingLR", "StepLR", "ExponentialLR", "ReduceLROnPlateau"]
        return ["MultiStepLR", "CosineAnnealing"]


# ─── Config File Detection ───────────────────────────────────────

def detect_engine_from_file(filepath: str) -> str:
    """Détecte le moteur (NeoSR ou TraiNNer-Redux) à partir d'un fichier config."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".toml":
        return "NeoSR"
    elif ext in (".yml", ".yaml"):
        return "TraiNNer-Redux"
    # Try reading content
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read(500)
        if "model_type" in content or "[network_g]" in content or "[train." in content:
            return "NeoSR"
        if "high_order_degradation" in content or "lq_size" in content or "batch_size_per_gpu" in content:
            return "TraiNNer-Redux"
    except Exception:
        pass
    return "NeoSR"  # Default
