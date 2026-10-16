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


# neosr "presets" : factory functions auto-suffisantes qui hardcodent toutes les
# dimensions (embed_dim/depths/num_heads/img_size…) puis appellent l'arch de base
# via **kwargs. Injecter ces params depuis l'UI provoque
# "got multiple values for keyword argument 'embed_dim'/'img_size'". → on n'injecte
# AUCUN param dyn_ pour ces archs (ni canaux). hat_*/aether_* prennent `scale` en
# kwarg ; swinir_*/srformer_medium lisent l'échelle via net_opt() (global) → pas de scale.
_NEOSR_PRESET_ARCHS = {
    "hat_s", "hat_m", "hat_l",
    "swinir_small", "swinir_medium", "srformer_medium",
    "aether_mobile", "aether_tiny", "aether_small", "aether_medium",
    "aether_large", "aether_pro", "aether_extreme",
}
_NEOSR_PRESET_NEED_SCALE = {
    "hat_s", "hat_m", "hat_l",
    "aether_mobile", "aether_tiny", "aether_small", "aether_medium",
    "aether_large", "aether_pro", "aether_extreme",
}


def _clamp_scale_for_arch(arch, scale):
    """Force le scale aux contraintes de l'arch (sinon assert au train).
    cascadedgaze→scale figé 1, realcugan→scale min 2, etc. Source de vérité app :
    l'utilisateur ne peut pas générer une config qui plante sur ces archs."""
    try:
        from src.core.descriptions import arch_fixed_scale, arch_min_scale
    except Exception:
        try:
            from .descriptions import arch_fixed_scale, arch_min_scale
        except Exception:
            return scale
    try:
        fx = arch_fixed_scale(arch)
        if fx:
            return int(fx)
        mn = arch_min_scale(arch)
        if mn and int(scale) < int(mn):
            return int(mn)
    except Exception:
        pass
    return scale

# Archs whose __init__ accepts neither num_in_ch nor **kwargs — injecting
# these keys causes TypeError at train time.
_ARCHS_NO_CHANNELS = {
    "catanet", "cfsr", "cugan", "dunet", "eimn", "esc", "flexnet", "hasn",
    "krgn", "lmlt", "man", "metagan", "moesr", "mosrv2", "msdan", "ninasr",
    "plainusr", "plksr", "safmn", "vgg",
    # ParagonSR (v1) and ParagonSR2 factory functions use `in_chans` not `num_in_ch`.
    # They're fully self-contained presets — pass only `scale`, no channel args.
    "paragonsr_anime", "paragonsr_nano", "paragonsr_tiny", "paragonsr_xs",
    "paragonsr_s", "paragonsr_m", "paragonsr_l", "paragonsr_xl",
    "paragonsr2_realtime", "paragonsr2_stream", "paragonsr2_photo",
    "paragonsr2_pro", "paragonsr2_ultimate", "paragonsr2_ultimate_v2",
    # AetherNet (Phhofm) — NeoSR arch, factory functions accept scale + **kwargs.
    # AetherNet uses `in_chans` internally, not `num_in_ch` — skip channel injection.
    "aether_mobile", "aether_tiny", "aether_small",
    "aether_medium", "aether_large", "aether_pro", "aether_extreme",
}

# ── Config readability: group top-level keys into labelled sections ──────────────
# Reordering top-level keys is safe (msgspec ignores order); the injected "# ═══"
# headers are YAML/TOML comments → stripped on load, so the engine never sees them.
# A strict re-parse gate (see _sectioned_dump) guarantees zero functional change.
_REDUX_SECTIONS = [
    ("Général", ["name", "scale", "num_gpu", "manual_seed", "deterministic", "queue_size"]),
    ("Performance / AMP", ["use_amp", "amp_bf16", "use_channels_last", "fast_matmul",
                            "use_compile", "compile_mode"]),
    ("Dégradations — Activation", ["high_order_degradation", "lq_usm", "lq_usm_radius_range",
                                    "thicklines_prob"]),
    ("Dégradations OTF — Stage 1", ["blur_prob", "blur_kernel_size", "kernel_list", "kernel_prob",
        "blur_sigma", "betag_range", "betap_range", "sinc_prob",
        "resize_prob", "resize_mode_list", "resize_mode_prob", "resize_range",
        "gaussian_noise_prob", "noise_range", "poisson_scale_range", "gray_noise_prob",
        "jpeg_prob", "jpeg_range"]),
    ("Dégradations OTF — Stage 2", ["blur_prob2", "blur_kernel_size2", "kernel_list2", "kernel_prob2",
        "blur_sigma2", "betag_range2", "betap_range2", "sinc_prob2",
        "resize_prob2", "resize_mode_list2", "resize_mode_prob2", "resize_range2",
        "gaussian_noise_prob2", "noise_range2", "poisson_scale_range2", "gray_noise_prob2",
        "jpeg_prob2", "jpeg_range2",
        "final_sinc_prob", "resize_mode_list3", "resize_mode_prob3"]),
    ("Dégradations OTF — Debug", ["high_order_degradations_debug", "high_order_degradations_debug_limit"]),
    ("Datasets", ["datasets"]),
    ("Réseau Générateur", ["network_g"]),
    ("Réseau Discriminateur", ["network_d", "network_d2"]),
    ("Chemins / Reprise", ["path"]),
    ("Entraînement (losses, optim, scheduler)", ["train"]),
    ("Validation", ["val"]),
    ("Logger", ["logger"]),
    ("USS Monitoring (retiré pour le moteur)", ["monitoring"]),
]
_NEOSR_SECTIONS = [
    ("Général", ["name", "scale", "model_type", "num_gpu", "manual_seed", "deterministic",
                 "use_amp", "bfloat16", "compile", "fast_matmul", "queue_size"]),
    ("Dégradations OTF", ["high_order_degradation", "lq_usm", "lq_usm_radius_range"]),
    ("Datasets", ["datasets"]),
    ("Réseau Générateur", ["network_g"]),
    ("Réseau Discriminateur", ["network_d"]),
    ("Chemins / Reprise", ["path"]),
    ("Entraînement", ["train"]),
    ("Validation", ["val"]),
    ("Logger", ["logger"]),
]


def _build_section_index(config, sections):
    """Reorder `config` keys per `sections`, returning (ordered_dict, first_key→title).
    Custom degradation keys are folded into a 'Dégradations Custom' group; any key not
    covered anywhere lands in 'Autres' so NOTHING is ever dropped."""
    try:
        from src.core.otf_custom_degradations import _ALL_CUSTOM_KEYS as _CUSTOM
    except Exception:
        _CUSTOM = []
    groups = list(sections)
    # insert the custom-degradation group right after the OTF debug section if present
    if _CUSTOM:
        idx = next((i for i, (t, _) in enumerate(groups) if "Debug" in t), len(groups) - 7)
        groups.insert(idx + 1, ("Dégradations Custom", list(_CUSTOM)))
    ordered, used, first_of = {}, set(), {}
    for title, keys in groups:
        present = [k for k in keys if k in config and k not in used]
        if present:
            first_of[present[0]] = title
            for k in present:
                ordered[k] = config[k]; used.add(k)
    leftover = [k for k in config if k not in used]
    if leftover:
        first_of[leftover[0]] = "Autres"
        for k in leftover:
            ordered[k] = config[k]
    return ordered, first_of


def _ascii_title(s):
    """Transliterate a section title to pure ASCII. The redux engine's copy_opt_file()
    reopens the config with the OS default encoding (cp1252 on Windows) and CRASHES on
    bytes undefined there (e.g. the 3rd byte of '═'). Keeping comments ASCII-only makes
    the config safe for every tool/engine that reads it without an explicit encoding."""
    import unicodedata
    s = (s or "").replace("—", "-").replace("–", "-").replace("→", "->").replace("’", "'")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.strip() or "Section"


def _inject_section_comments(text, first_of, key_fn):
    out = []
    for line in text.splitlines():
        k = key_fn(line)
        if k is not None and k in first_of:
            if out:
                out.append("")
            out.append(f"# ========== {_ascii_title(first_of[k])} ==========")
        out.append(line)
    return "\n".join(out) + "\n"


def _yaml_key(line):
    import re as _re
    m = _re.match(r'^([A-Za-z0-9_]+):', line)
    return m.group(1) if m else None


def _toml_key(line):
    import re as _re
    m = _re.match(r'^\[+([A-Za-z0-9_]+)', line) or _re.match(r'^([A-Za-z0-9_]+)\s*=', line)
    return m.group(1) if m else None


# Redux OTF keys that the phase degradations can change during a run (read in
# RealESRGANModel.feed_data; same list as runner.py _PHASE_BASE_KEYS).
_PHASE_DEG_KEYS = (
    "thicklines_prob", "blur_prob", "resize_prob", "resize_range",
    "gaussian_noise_prob", "noise_range", "poisson_scale_range", "gray_noise_prob",
    "jpeg_prob", "jpeg_range",
    "blur_prob2", "resize_prob2", "resize_range2",
    "gaussian_noise_prob2", "noise_range2", "poisson_scale_range2", "gray_noise_prob2",
    "jpeg_prob2", "jpeg_range2",
)


# Criterion values accepted by each Redux loss (first = default). Checked against the
# engine source (2026-10-29).
_REDUX_LOSS_CRITERIA = {
    "perceptualloss": ("charbonnier", "l1", "fd", "pd", "fd+l1", "pd+l1"),
    "perceptualfp16loss": ("charbonnier", "l1", "fd", "pd", "fd+l1", "pd+l1"),
    "perceptualanimeloss": ("l1", "charbonnier"),
    "linedistillerloss": ("l1", "charbonnier"),
    "ldlloss": ("l1", "l2", "charbonnier", "huber"),
    "colorloss": ("l1", "l2", "charbonnier"),
    "lumaloss": ("l1", "l2", "charbonnier"),
    "gradientvarianceloss": ("charbonnier", "l1", "l2"),
    "aesoploss": ("charbonnier", "l1"),
    "sparkloss": ("fd", "charbonnier"),
}


def _gan_start_section(data, safe_num, safe_bool) -> dict:
    """Redux train keys of the delayed GAN start; empty when the GAN starts at 0."""
    if not safe_bool("use_gan", False):
        return {}
    start = safe_num("gan_start_iter", 0, int)
    auto = safe_bool("gan_auto_start", False)
    out = {}
    if start > 0:
        out["gan_start_iter"] = start
    if auto:
        out.update({
            "gan_auto_start": True,
            "gan_auto_patience": max(1, safe_num("gan_auto_patience", 3, int)),
            "gan_auto_min_delta": safe_num("gan_auto_min_delta", 0.05, float),
            "gan_auto_min_iter": max(0, safe_num("gan_auto_min_iter", 20000, int)),
        })
    if out and safe_bool("gan_restart_lr", False):
        out["gan_restart_lr"] = True
    return out


# neosr vgg_perceptual_loss criteria (anything else raises NotImplementedError at launch).
# Old configs / the Redux list carried "charbonnier" or "chc_loss": mapped to the nearest.
_NEOSR_PERCEP_CRIT = {"l1": "l1", "l2": "l2", "mse": "l2", "huber": "huber", "chc": "chc",
                      "chc_loss": "chc", "charbonnier": "l1"}


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
            # Phase losses: train.losses holds every loss of both profiles; split it back
            # into the PSNR profile (main fields) and the GAN profile (Losses panel).
            _lp2 = data.get("loss_phase2") if isinstance(data, dict) else None
            _lflat2 = None
            if isinstance(_lp2, dict) and _lp2 and isinstance(data.get("train"), dict):
                import copy as _cp

                def _profile(idx):
                    d = _cp.deepcopy(data)
                    keep = []
                    for _l in (d["train"].get("losses") or []):
                        ws = _lp2.get(str(_l.get("type"))) if isinstance(_l, dict) else None
                        if ws is not None:
                            if float(ws[idx]) == 0:
                                continue
                            _l["loss_weight"] = ws[idx]
                        keep.append(_l)
                    d["train"]["losses"] = keep
                    d.pop("loss_phase2", None)
                    return d
                try:
                    _lflat2 = self._flatten_config(_profile(1))
                    data = _profile(0)
                except Exception:
                    _lflat2 = None
            flat = self._flatten_config(data)
            flat["_detected_engine"] = detected
            if _lflat2 is not None:
                flat["_loss_phase2_flat"] = _lflat2
                flat["loss_phase2_ramp"] = str(data.get("loss_phase2_ramp", 5000))
            # Phase degradations: flatten the GAN profile too (PSNR profile overlaid
            # with deg_phase2), for the Dégradations tab.
            _p2 = data.get("deg_phase2") if isinstance(data, dict) else None
            if isinstance(_p2, dict) and _p2:
                _merged = dict(data)
                _merged.update(_p2)
                _merged.pop("deg_phase2", None)
                try:
                    flat["_deg_phase2_flat"] = self._flatten_config(_merged)
                except Exception:
                    pass
                flat["deg_phase2_ramp"] = str(data.get("deg_phase2_ramp", 15000))
            return True, flat
        except Exception as e:
            return False, f"Erreur de lecture : {e}"

    def _flatten_config(self, cfg):
        # --- FIX IMPORTANT: safe_num DOIT être défini ICI ---
        def safe_num(k, default, type_cast=float):
            val = cfg.get(k)
            if val is None: val = cfg.get("degradations", {}).get(k)
            # Redux writes some kernel/degradation params (e.g. final_sinc_prob) under
            # datasets.train instead of root/degradations — safe_num didn't know to look
            # there, so a value saved correctly for training was reloaded as the default
            # in the UI (2026-09-17 fix). The `degs`-merge below (lines ~713-725) already
            # handled this for a few keys read directly from `degs`, but safe_num itself
            # never saw it.
            if val is None: val = cfg.get("datasets", {}).get("train", {}).get(k)
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
                    flat[f"dd_{k}"] = v

        net_d2 = cfg.get("network_d2", {})
        if net_d2:
            flat["use_disc2"] = "true"
            flat["net_d2_type"] = net_d2.get("type", "unetdiscriminatorsn")
            for k, v in net_d2.items():
                if k not in ["type", "num_in_ch", "in_ch"]:
                    flat[f"dd2_{k}"] = v
        gan_opt2 = cfg.get("train", {}).get("gan_opt2")
        if gan_opt2:
            flat["gan_weight_d2"] = gan_opt2.get("loss_weight", 0.05)

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
        elif "pairedvideodataset" in raw_type or "singlevideodataset" in raw_type:
            # Video sequence datasets (must be checked BEFORE the generic "paired" check,
            # since "pairedvideodataset" also contains the substring "paired").
            flat["dataset_mode"] = "sequence"
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
        if flat["dataset_mode"] == "sequence":
            # PairedVideoDataset stores crop size as gt_size (not lq_size). Convert back
            # to the UI's "patch_size" (LR crop size) for round-trip display.
            try:
                _scale_for_ps = int(float(flat.get("scale", 4)))
            except Exception:
                _scale_for_ps = 4
            _gt_size_val = ds_train.get("gt_size", None)
            if isinstance(_gt_size_val, (int, float)) and _scale_for_ps > 0:
                flat["patch_size"] = int(_gt_size_val) // _scale_for_ps
            else:
                flat["patch_size"] = ds_train.get("patch_size", 64)
            flat["clip_size"] = ds_train.get("clip_size", 5)
        else:
            # Support both NeoSR (patch_size) and Redux (lq_size)
            flat["patch_size"] = ds_train.get("patch_size", ds_train.get("lq_size", 64))
        flat["num_worker_per_gpu"] = ds_train.get("num_worker_per_gpu", 4)
        flat["prefetch_mode"] = ds_train.get("prefetch_mode", "cuda")
        flat["use_hflip"] = str(ds_train.get("use_hflip", True)).lower()
        flat["use_rot"] = str(ds_train.get("use_rot", True)).lower()
        
        aug_map = {"mixup": "aug_mixup", "cutmix": "aug_cutmix", "resizemix": "aug_resizemix", "cutblur": "aug_cutblur"}
        # NeoSR format: augmentation/aug_prob inside datasets.train
        augs = ds_train.get("augmentation", [])
        probs = ds_train.get("aug_prob", [])
        idx = 0
        for a in augs:
            if a == "none": idx += 1; continue
            if a in aug_map:
                flat[aug_map[a]] = "true"
                if idx < len(probs): flat[f"prob_{aug_map[a]}"] = probs[idx]
            idx += 1
        # Redux format: use_moa/moa_augs/moa_probs under train: section (TrainOptions)
        _train_sect = cfg.get("train", {}) or {}
        _moa_augs  = _train_sect.get("moa_augs", cfg.get("moa_augs", []))   # fallback top-level for old configs
        _moa_probs = _train_sect.get("moa_probs", cfg.get("moa_probs", []))
        # Restore the augmentation checkboxes from moa_augs whether or not use_moa is set
        # (the user may have configured augs even if the master MoA toggle is off).
        if _moa_augs:
            flat["moa_augs"] = list(_moa_augs)
            # moa_probs are per-batch probabilities, shown as-is (see the MoA note in
            # _gen_redux; "none" is the remainder).
            for _i, _a in enumerate(_moa_augs):
                if _a == "none":
                    continue
                if _a in aug_map:
                    flat[aug_map[_a]] = "true"
                    _p = _moa_probs[_i] if _i < len(_moa_probs) else 0.15
                    flat[f"prob_{aug_map[_a]}"] = round(float(_p), 3)
                    # Config tab slider key (use_<aug>_prob): without it a config loaded
                    # without its .uss.json fell back to 0.15 per augmentation.
                    flat[f"use_{_a}_prob"] = round(float(_p), 3)
        # MoA debug (shared by both Redux train section and NeoSR)
        _moa_debug = _train_sect.get("moa_debug", cfg.get("moa_debug", False))
        flat["moa_debug"] = str(_moa_debug).lower()
        _moa_debug_limit = _train_sect.get("moa_debug_limit", cfg.get("moa_debug_limit", 100))
        flat["moa_debug_limit"] = int(_moa_debug_limit) if _moa_debug_limit else 100
        # OTF debug (Redux top-level) — was not flattened, so the UI checkbox never restored.
        flat["high_order_degradations_debug"] = str(cfg.get("high_order_degradations_debug", False)).lower()
        flat["high_order_degradations_debug_limit"] = int(cfg.get("high_order_degradations_debug_limit", 100) or 100)

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
        pmd = paths.get("pretrain_network_d")
        flat["pretrain_model_d"] = "" if pmd is None else str(pmd)
        pmd2 = paths.get("pretrain_network_d2")
        flat["pretrain_model_d2"] = "" if pmd2 is None else str(pmd2)

        train = cfg.get("train", {})
        logger = cfg.get("logger", {})
        
        iter_val = logger.get("total_iter") or train.get("total_iter") or train.get("n_iter") or 100000
        flat["total_iter"] = iter_val
        flat["warmup_iter"] = train.get("warmup_iter", -1)
        flat["adaptive_d"] = str(train.get("adaptive_d", False)).lower()
        flat["gan_start_iter"] = str(train.get("gan_start_iter", 0))
        flat["gan_auto_start"] = str(train.get("gan_auto_start", False)).lower()
        flat["gan_auto_patience"] = str(train.get("gan_auto_patience", 3))
        flat["gan_auto_min_delta"] = str(train.get("gan_auto_min_delta", 0.05))
        flat["gan_auto_min_iter"] = str(train.get("gan_auto_min_iter", 20000))
        # Key absent: checked (UI default) unless the config already delays the GAN,
        # where absence really means off (the engine default is False).
        flat["gan_restart_lr"] = str(train.get("gan_restart_lr", "gan_auto_start" not in train and "gan_start_iter" not in train)).lower()
        flat["grad_clip"] = str(train.get("grad_clip", False)).lower()
        flat["grad_clip_max_norm"] = str(train.get("grad_clip_max_norm", train.get("max_norm", 100)))
        # NeoSR uses "ema", Redux uses "ema_decay" — read both for round-trip fidelity
        flat["ema"] = str(train.get("ema", train.get("ema_decay", 0.999)))
        flat["sam"] = train.get("sam", "none")
        flat["sam_init"] = train.get("sam_init", -1)
        # NeoSR ECO schedule (train.eco = true + these keys)
        for _k, _d in (("eco_init", 15000), ("eco_iters", 80000), ("eco_schedule", "sigmoid")):
            if _k in train:
                flat[_k] = str(train[_k])
        # ECO: native dev nests it under train.eco.enabled; old custom used a bool train.eco.
        _eco = train.get("eco", False)
        if isinstance(_eco, dict):
            flat["eco_mode"] = str(_eco.get("enabled", False)).lower()
            flat["eco_end_ratio"] = str(_eco.get("end_ratio", 0.75))
            flat["eco_mode_kind"] = str(_eco.get("mode", "full"))
        else:
            flat["eco_mode"] = str(_eco).lower()
        # Teacher pretrain: native pretrain_network_g_teacher, fallback to old eco_pretrain_g.
        eco_pt = paths.get("pretrain_network_g_teacher", paths.get("eco_pretrain_g"))
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
        # NOTE: neosr's consistency_loss.__init__ signature uses "blur"/"cosim"/
        # "saturation"/"brightness" (not "use_blur"/"use_cosim"/"use_saturation"/
        # "use_brightness") -- read both key spellings for backward compat with
        # configs saved before this fix.
        _consistency_opt = train.get("consistency_opt", {})
        flat["consistency_blur"]       = str(_consistency_opt.get("blur",       _consistency_opt.get("use_blur",       True))).lower()
        flat["consistency_cosim"]      = str(_consistency_opt.get("cosim",      _consistency_opt.get("use_cosim",      True))).lower()
        flat["consistency_saturation"] = str(_consistency_opt.get("saturation", _consistency_opt.get("use_saturation", True))).lower()
        flat["consistency_brightness"] = str(_consistency_opt.get("brightness", _consistency_opt.get("use_brightness", True))).lower()

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
            flat["dyn_gan_weight"] = train["gan_opt"].get("loss_weight", 0.05)
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
                elif "perceptualfp16" in lt:
                    flat["loss_percep_fp16"] = "true"
                    flat["weight_loss_percep_fp16"] = lw
                    flat["percep_fp16_criterion"] = loss.get("criterion", "charbonnier")
                elif "perceptualanime" in lt:
                    flat["loss_percep_anime"] = "true"
                    flat["weight_loss_percep_anime"] = lw
                    flat["percep_anime_criterion"]   = loss.get("criterion", "l1")
                elif "perceptual" in lt:
                    flat["loss_percep"] = "true"
                    flat["weight_loss_percep"] = lw
                    flat["percep_criterion"] = loss.get("criterion", "charbonnier")
                    # Restore per-layer VGG weights from Redux losses list
                    _lw_redux = loss.get("layer_weights", {})
                    if _lw_redux:
                        _VGG_ALL_LOAD = ["conv1_2","conv2_2","conv3_2","conv3_4","conv4_2","conv4_4","conv5_2","conv5_4"]
                        for _vl in _VGG_ALL_LOAD:
                            # fd / pd configs name the same layers reluX_Y
                            flat[f"percep_vgg_{_vl}"] = _lw_redux.get(
                                _vl, _lw_redux.get(_vl.replace("conv", "relu", 1), 0.0))
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
                    flat["dyn_gan_weight"] = lw
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
                elif "linedistiller" in lt:
                    flat["loss_linedistiller"] = "true"
                    flat["weight_loss_linedistiller"] = lw
                    flat["linedistiller_criterion"] = loss.get("criterion", "l1")
                elif "aesop" in lt:
                    flat["loss_aesop"] = "true"
                    flat["weight_loss_aesop"] = lw
                    flat["aesop_criterion"] = loss.get("criterion", "charbonnier")
                    flat["aesop_ae_path"] = str(loss.get("pretrain_network_ae", "") or "")
                elif "bcewithlogitsdice" in lt or "bcedice" in lt:
                    flat["loss_bcedice"] = "true"
                    flat["weight_loss_bcedice"] = lw
                    flat["bcedice_weight_bce"]  = loss.get("weight_bce",  1.0)
                    flat["bcedice_weight_dice"] = loss.get("weight_dice", 1.0)
                elif "perceptualanimeloss" in lt or "percep_anime" in lt:
                    flat["loss_percep_anime"] = "true"
                    flat["weight_loss_percep_anime"] = lw
                    flat["percep_anime_criterion"]   = loss.get("criterion", "l1")
                elif "spark" in lt:
                    flat["loss_spark"] = "true"
                    flat["weight_loss_spark"] = lw
                    flat["spark_criterion"]   = loss.get("criterion", "fd")
                    flat["spark_path"]        = loss.get("path", "")
                    flat["spark_max_score"]   = loss.get("max_score", 50.0)
                elif "sobeledge" in lt or "sobel_edge" in lt:
                    # SobelEdgeLoss (traiNNer-redux native) — mapped to loss_edge widget
                    flat["loss_edge"] = "true"
                    flat["weight_loss_edge"] = lw

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
        flat["tile_val"] = flat["tile"]  # widget key ("Tile (Val)") differs from engine key "tile"
        flat["tile_pad"] = val.get("tile_pad", val.get("tile_overlap", 32))
        flat["save_img"] = str(val.get("save_img", True)).lower()
        flat["val_enabled"] = str(val.get("val_enabled", True)).lower()

        metrics = val.get("metrics", {})
        flat["metric_psnr"] = "true" if "psnr" in metrics else "false"
        flat["metric_ssim"] = "true" if "ssim" in metrics else "false"
        flat["metric_dists"] = "true" if "dists" in metrics else "false"
        flat["metric_topiq"] = "true" if "topiq" in metrics else "false"
        flat["metric_texture"] = "false" if str(cfg.get("usrs_texture_metrics", True)).lower() in ("false", "0", "no") else "true"

        degs = cfg.get("degradations", {})
        # Redux puts degradation params at top level
        if not degs and cfg.get("high_order_degradation"):
            degs = {k: v for k, v in cfg.items() if k.startswith(("resize_", "gaussian_", "noise_", "poisson_", "gray_", "blur_", "sinc_", "betag_", "betap_", "jpeg_", "final_", "second_", "queue_"))}
        # Custom degradations also sit at the top level in Redux: without this, their
        # ranges (posterize_bits_range, banding_levels_range…) fell back to the defaults
        # when a config was loaded without its .uss.json sidecar.
        try:
            from src.core.otf_custom_degradations import _ALL_CUSTOM_KEYS as _CK
            for _k in _CK:
                if _k in cfg and _k not in degs:
                    degs[_k] = cfg[_k]
        except Exception:
            pass
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
        for _sfx in ("", "2", "3"):
            for _rk in (f"resize_mode_list{_sfx}", f"resize_mode_prob{_sfx}"):
                _rv = cfg.get(_rk, degs.get(_rk))
                if isinstance(_rv, list) and _rv:
                    flat[_rk] = ",".join(str(x) for x in _rv)
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
        # Was not read back: a config loaded without its .uss.json got JPEG prob 2 = 1.0.
        flat["jpeg_prob2"] = degs.get("jpeg_prob2", 1.0)

        # Custom degradations (banding & posterize)
        flat["posterize_prob"] = safe_num("posterize_prob", 0.0, float)
        flat["posterize_bits_range"] = l2s(degs.get("posterize_bits_range", "3, 6"))
        flat["banding_prob"] = safe_num("banding_prob", 0.0, float)
        flat["banding_levels_range"] = l2s(degs.get("banding_levels_range", "16, 64"))
        flat["smooth_banding_prob"] = safe_num("smooth_banding_prob", 0.0, float)
        flat["smooth_banding_levels_range"] = l2s(degs.get("smooth_banding_levels_range", "48, 128"))
        flat["smooth_banding_dark_prob"] = safe_num("smooth_banding_dark_prob", 0.5, float)

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

        # Custom 3/4/5 keys (screentone, dithering, wtp_halo, color_level…) had no explicit
        # mapping above: they were dropped on load, so the next save from the UI lost them.
        try:
            from src.core.otf_custom_degradations import _ALL_CUSTOM_KEYS as _CK
            for _k in _CK:
                if _k in flat or _k not in degs:
                    continue
                _v = degs[_k]
                if _k.endswith("_prob"):
                    flat[_k] = safe_num(_k, 0.0, float)
                elif isinstance(_v, (list, tuple)):
                    flat[_k] = l2s(_v)
                else:
                    flat[_k] = _v
        except Exception:
            pass

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

    def _gen_redux(self, data, save_path, _return_cfg=False):
        """Génération complète pour TraiNNer-Redux (.yml).
        _return_cfg=True: return the config dict without writing (phase 2 profile)."""
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
            _RESIZE_MODES = ("bilinear", "bicubic", "nearest-exact", "lanczos", "area")
            def resize_modes(sfx):
                """(modes, probs) for resize stage sfx ("", "2", "3"). Kept from the loaded
                config when valid; default without nearest-exact (staircase / zipper LQ that
                real sources never have). Before this, 4 modes at 25 % were always written."""
                dm, dp = ["bilinear", "bicubic", "lanczos"], [0.3, 0.4, 0.3]
                m = data.get(f"resize_mode_list{sfx}")
                if isinstance(m, str):
                    m = [x.strip().strip("'\"") for x in m.strip("[]").split(",") if x.strip()]
                p = safe_list(f"resize_mode_prob{sfx}", None)
                if (isinstance(m, list) and m and all(x in _RESIZE_MODES for x in m)
                        and isinstance(p, list) and len(p) == len(m)):
                    try:
                        p = [float(x) for x in p]
                        if sum(p) > 0:
                            return list(m), p
                    except Exception:
                        pass
                return dm, dp

            scale = safe_num("scale", 4, int)
            total_iter = safe_num("total_iter", 100000, int)
            exp_name = data.get("name", "experiment").strip()
            arch = data.get("arch", "omnisr")
            scale = _clamp_scale_for_arch(arch, scale)
            lq_size = safe_num("patch_size", 64, int)
            bs = safe_num("batch_size", 4, int)
            _ds_mode = data.get("dataset_mode", "paired")
            is_otf = _ds_mode in ("otf", "realesrgandataset", "bicubic")
            is_sequence = _ds_mode == "sequence"

            config = {
                "name": exp_name,
                "scale": scale,
                "use_amp": safe_bool("use_amp", True),
                "amp_bf16": safe_bool("bfloat16", False),
                # Video archs work on rank-5 [B,T,C,H,W] tensors: channels_last is
                # unsupported there (traiNNer arch_info.ARCHS_WITHOUT_CHANNELS_LAST).
                "use_channels_last": safe_bool("channels_last", False) and not is_sequence,
                "fast_matmul": safe_bool("fast_matmul", False),
                "use_compile": safe_bool("compile", False),
                "compile_mode": "default",
                "num_gpu": (lambda v: int(v) if v.strip().isdigit() else "auto")(str(data.get("num_gpu", "auto"))),
                "deterministic": safe_bool("deterministic", False),
            }

            if is_otf:
                _is_bicubic_mode = _ds_mode == "bicubic"
                # Toujours True pour tout mode OTF (realesrgandataset) — SRModel ne peut pas
                # gérer les batches realesrgandataset (pas de clé 'lq'). RealESRGANModel
                # génère 'lq' depuis les kernels. Le mode "bicubic" est obtenu via des
                # params de dégradation minimaux (blur=0, noise=0), pas via ce flag.
                config["high_order_degradation"] = True
                # OTF debug: saves the degraded LQ to a debug folder each iter (lets the
                # user SEE that OTF + custom degradations are really applied).
                config["high_order_degradations_debug"] = safe_bool("high_order_degradations_debug", False)
                config["high_order_degradations_debug_limit"] = safe_num("high_order_degradations_debug_limit", 100, int)
                # LQ Unsharp Mask + ThickLines — exposed in the Dégradations tab (Avancé Redux).
                config["lq_usm"] = safe_bool("lq_usm", False)
                _usm_r = data.get("lq_usm_radius_range", "1,25")
                try:
                    _usm_parts = [int(round(float(x))) for x in str(_usm_r).replace(";", ",").split(",")[:2]]
                    config["lq_usm_radius_range"] = _usm_parts if len(_usm_parts) == 2 else [1, 25]
                except Exception:
                    config["lq_usm_radius_range"] = [1, 25]
                # First degradation
                config["thicklines_prob"] = safe_num("thicklines_prob", 0.0, float)
                config["blur_prob"] = safe_num("blur_prob", 0.0, float)
                config["resize_prob"] = safe_list("resize_prob", [0.2, 0.7, 0.1])
                config["resize_mode_list"], config["resize_mode_prob"] = resize_modes("")
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
                config["resize_mode_list2"], config["resize_mode_prob2"] = resize_modes("2")
                config["resize_range2"] = safe_list("resize_range2", [0.6, 1.2])
                config["gaussian_noise_prob2"] = safe_num("gaussian_noise_prob2", 0.0, float)
                config["noise_range2"] = safe_list("noise_range2", [1, 25])
                config["poisson_scale_range2"] = safe_list("poisson_scale_range2", [0.05, 2.5])
                config["gray_noise_prob2"] = safe_num("gray_noise_prob2", 0.0, float)
                config["jpeg_prob2"] = safe_num("jpeg_prob2", 1.0, float)
                config["jpeg_range2"] = safe_list("jpeg_range2", [30, 95])
                # Final resize
                config["resize_mode_list3"], config["resize_mode_prob3"] = resize_modes("3")
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
                sb_prob = safe_num("smooth_banding_prob", 0.0, float)
                if sb_prob > 0:
                    config["smooth_banding_prob"] = sb_prob
                    config["smooth_banding_levels_range"] = safe_list("smooth_banding_levels_range", [48, 128])
                    config["smooth_banding_dark_prob"] = safe_num("smooth_banding_dark_prob", 0.5, float)
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
                # Custom 3/4/5 degradations — were declared in _ALL_CUSTOM_KEYS and read by
                # the engine (otf_custom_degradations.py) but never written here, so they
                # were silently dropped from every generated config regardless of UI value
                # (2026-09-17 fix).
                st_prob = safe_num("screentone_prob", 0.0, float)
                if st_prob > 0:
                    config["screentone_prob"] = st_prob
                    config["screentone_dot_size"] = safe_list("screentone_dot_size", [7, 15])
                    config["screentone_angle"] = safe_list("screentone_angle", [0, 90])
                    config["screentone_dot_type"] = data.get("screentone_dot_type", "circle")
                    config["screentone_color_space"] = data.get("screentone_color_space", "rgb")
                dt_prob = safe_num("dithering_prob", 0.0, float)
                if dt_prob > 0:
                    config["dithering_prob"] = dt_prob
                    config["dithering_color_ch"] = safe_list("dithering_color_ch", [2, 8])
                    config["dithering_type"] = data.get("dithering_type", "floyd-steinberg")
                px_prob = safe_num("pixelate_prob", 0.0, float)
                if px_prob > 0:
                    config["pixelate_prob"] = px_prob
                    config["pixelate_size"] = safe_list("pixelate_size", [2, 16])
                sn_prob = safe_num("sin_prob", 0.0, float)
                if sn_prob > 0:
                    config["sin_prob"] = sn_prob
                    config["sin_shape"] = safe_list("sin_shape", [100, 600])
                    config["sin_alpha"] = safe_list("sin_alpha", [0.1, 0.4])
                    config["sin_bias"] = safe_list("sin_bias", [0.8, 1.2])
                    config["sin_orientation"] = data.get("sin_orientation", "random")
                ss_prob = safe_num("subsampling_prob", 0.0, float)
                if ss_prob > 0:
                    config["subsampling_prob"] = ss_prob
                    config["subsampling_format"] = data.get("subsampling_format", "4:2:0")
                    config["subsampling_yuv"] = data.get("subsampling_yuv", "601")
                dc_prob = safe_num("dct_block_prob", 0.0, float)
                if dc_prob > 0:
                    config["dct_block_prob"] = dc_prob
                    config["dct_block_quality_range"] = safe_list("dct_block_quality_range", [20, 80])
                    config["dct_block_size_range"] = safe_list("dct_block_size_range", [4, 4])
                    config["dct_block_chroma_scale_range"] = safe_list("dct_block_chroma_scale_range", [1.5, 2.5])
                    config["dct_block_yuv"] = data.get("dct_block_yuv", "601")
                cl_prob = safe_num("color_level_prob", 0.0, float)
                if cl_prob > 0:
                    config["color_level_prob"] = cl_prob
                    config["color_level_high"] = safe_list("color_level_high", [220, 255])
                    config["color_level_low"] = safe_list("color_level_low", [0, 35])
                    config["color_level_gamma"] = safe_list("color_level_gamma", [0.7, 1.5])
                wh_prob = safe_num("wtp_halo_prob", 0.0, float)
                if wh_prob > 0:
                    config["wtp_halo_prob"] = wh_prob
                    config["wtp_halo_strength"] = safe_list("wtp_halo_strength", [0.1, 0.5])
                    config["wtp_halo_radius"] = safe_list("wtp_halo_radius", [3, 12])
                sa_prob = safe_num("saturation_prob", 0.0, float)
                if sa_prob > 0:
                    config["saturation_prob"] = sa_prob
                    config["saturation_range"] = safe_list("saturation_range", [0.3, 1.8])
                sh_prob = safe_num("shift_prob", 0.0, float)
                if sh_prob > 0:
                    config["shift_prob"] = sh_prob
                    config["shift_range"] = safe_list("shift_range", [1, 8])
                    config["shift_axis"] = data.get("shift_axis", "random")
                db_prob = safe_num("disc_blur_prob", 0.0, float)
                if db_prob > 0:
                    config["disc_blur_prob"] = db_prob
                    config["disc_blur_radius_range"] = safe_list("disc_blur_radius_range", [2, 8])
                vg_prob = safe_num("vignette_prob", 0.0, float)
                if vg_prob > 0:
                    config["vignette_prob"] = vg_prob
                    config["vignette_strength_range"] = safe_list("vignette_strength_range", [0.2, 0.6])
                    config["vignette_radius_range"] = safe_list("vignette_radius_range", [0.45, 0.75])
                qd_prob = safe_num("quantize_depth_prob", 0.0, float)
                if qd_prob > 0:
                    config["quantize_depth_prob"] = qd_prob
                    config["quantize_depth_bits_range"] = safe_list("quantize_depth_bits_range", [4, 7])
                co_prob = safe_num("coupled_optical_prob", 0.0, float)
                if co_prob > 0:
                    config["coupled_optical_prob"] = co_prob
                cv_prob = safe_num("coupled_vintage_prob", 0.0, float)
                if cv_prob > 0:
                    config["coupled_vintage_prob"] = cv_prob

            # Datasets
            if is_otf:
                train_type = "realesrgandataset"
            elif is_sequence:
                train_type = "pairedvideodataset"
            else:
                train_type = "pairedimagedataset"
            gt_path = data.get("dataroot_gt", "datasets/train/dataset1/hr").replace("\\", "/")

            # MoA augmentations — Redux native format (top-level keys, NOT in datasets.train).
            # Format: use_moa + moa_augs list + moa_probs list (must sum to 1.0).
            # The UI sliders are per-batch probabilities (as their tooltip says): "none" gets
            # the remainder. Only when they sum above 1.0 are they scaled down to fit.
            # (Was: none fixed at 0.4 and the sliders used as relative weights, so 4 x 0.05
            # really trained with 4 x 0.15.) Defaults 4 x 0.15 -> none 0.4, unchanged.
            _moa_active = []  # (aug_name, raw_prob) for enabled augs
            for _akey, _aname, _apkey in [
                ("aug_mixup",     "mixup",     "prob_aug_mixup"),
                ("aug_cutmix",    "cutmix",    "prob_aug_cutmix"),
                ("aug_resizemix", "resizemix", "prob_aug_resizemix"),
                ("aug_cutblur",   "cutblur",   "prob_aug_cutblur"),
            ]:
                if safe_bool(_akey, False):
                    _moa_active.append((_aname, safe_num(_apkey, 0.15, float)))

            _moa_active = [(n, max(0.0, p)) for n, p in _moa_active if p > 0]
            if _moa_active:
                _raw_total = sum(p for _, p in _moa_active)
                _scale = 1.0 / _raw_total if _raw_total > 1.0 else 1.0
                _none_prob = max(0.0, 1.0 - _raw_total * _scale)
                _moa_augs_list  = ["none"] + [n for n, _ in _moa_active]
                _moa_probs_list = [_none_prob] + [p * _scale for _, p in _moa_active]
                # Normalize to exactly 1.0 (fix float rounding on first element)
                _diff = 1.0 - sum(_moa_probs_list)
                _moa_probs_list[0] = round(_moa_probs_list[0] + _diff, 4)
                _moa_probs_list = [round(v, 4) for v in _moa_probs_list]
            else:
                _moa_augs_list  = ["none", "mixup", "cutmix", "resizemix", "cutblur"]
                _moa_probs_list = [0.4, 0.15, 0.15, 0.15, 0.15]

            if is_sequence:
                # MoA batch augmentations (mixup/cutmix/resizemix/cutblur, traiNNer/ops/batchaug.py)
                # assume img_lq is (N,C,H,W). PairedVideoDataset's lq is (N,T,C,H,W) — these ops
                # would index the wrong dims (mixing frame/channel axes) and either crash on the
                # "same resolution" shape check or silently corrupt the clip. Force-disable
                # regardless of the UI checkbox state — not a user-facing toggle for video archs.
                _moa_augs_list  = ["none"]
                _moa_probs_list = [1.0]

            train_ds = {
                "name": "Train Dataset",
                "type": train_type,
                "dataroot_gt": [gt_path],
                "use_hflip": safe_bool("use_hflip", True),
                "use_rot": safe_bool("use_rot", True),
                "num_worker_per_gpu": safe_num("num_worker_per_gpu", 8, int),
                "batch_size_per_gpu": bs,
                "accum_iter": safe_num("accumulate", 1, int),
            }
            if is_sequence:
                # PairedVideoDataset expects gt_size (crop size on the GT/middle frame),
                # not lq_size. Convert from the UI's LR patch_size using scale.
                train_ds["gt_size"] = lq_size * scale
                train_ds["clip_size"] = safe_num("clip_size", 5, int)
                if arch == "motsr":
                    # MoTSR.forward() only accepts exactly 5 frames ([B,5,3,H,W]).
                    train_ds["clip_size"] = 5
            else:
                train_ds["lq_size"] = lq_size
            # NOTE: augmentation keys go to top-level config (MoA format), NOT here.
            # (NeoSR uses augmentation/aug_prob inside datasets.train — Redux uses use_moa top-level)
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
            val_ds = {
                "name": "Val Dataset",
                "type": "pairedvideodataset" if is_sequence else "pairedimagedataset",
                "dataroot_gt": [val_gt] if val_gt else ["datasets/val/dataset1/hr"],
                "dataroot_lq": [val_lq] if val_lq else ["datasets/val/dataset1/lr"],
            }
            # LMDB (Tools > LMDB Creator): the pinned traiNNer-redux cannot read it.
            # Its OTF dataset hands the LMDB keys to the image loader as file paths, and
            # the paired one builds its path list from the letters of "lq"/"gt" (keys
            # "l_path"/"g_path") and opens the folder lists as one path. NeoSR reads it.
            _is_lmdb = lambda x: str(x).rstrip("/\\").lower().endswith(".lmdb")
            _roots = (list(train_ds.get("dataroot_gt", [])) + list(train_ds.get("dataroot_lq", []))
                      + list(val_ds.get("dataroot_gt", [])) + list(val_ds.get("dataroot_lq", [])))
            if any(_is_lmdb(x) for x in _roots):
                return False, ("LMDB : traiNNer-redux ne sait pas lire les bases LMDB (lecture cassée dans "
                               "le moteur). Utilisez NeoSR (mode paired, GT + LQ en .lmdb) ou un dossier "
                               "d'images. / traiNNer-redux cannot read LMDB databases: use NeoSR or an image folder.")
            if is_sequence:
                # Validation also uses PairedVideoDataset (needs GT for periodic metrics).
                # No gt_size here: val runs on full images, no cropping (same as pairedimagedataset).
                val_ds["clip_size"] = 5 if arch == "motsr" else safe_num("clip_size", 5, int)
                # PairedVideoDataset builds clips per scene (file prefix before "_"): a
                # val folder whose longest scene is shorter than clip_size yields 0 clips
                # and traiNNer then dies in validation with "ZeroDivisionError".
                _vlq = val_ds["dataroot_lq"][0]
                # Only matters when validation runs (it blocked saving with validation off).
                if safe_bool("val_enabled", True) and os.path.isdir(_vlq):
                    _scenes = {}
                    for _f in os.listdir(_vlq):
                        if _f.lower().endswith((".png", ".jpg", ".jpeg")):
                            _scenes[_f.split("_")[0]] = _scenes.get(_f.split("_")[0], 0) + 1
                    _cs = val_ds["clip_size"]
                    if _scenes and max(_scenes.values()) < _cs:
                        return False, (
                            f"Validation séquence : la plus longue scène du dossier Val LQ fait "
                            f"{max(_scenes.values())} frames, il en faut au moins clip_size = {_cs} "
                            f"(frames nommées <scène>_<numéro>.png). Ajoutez des frames ou baissez clip_size.")
            config["datasets"] = {
                "train": train_ds,
                "val": val_ds,
            }

            # MoA (Mix of Augmentations) — NOT written.
            # This traiNNer-redux dev version does not include use_moa/moa_augs in ReduxOptions
            # (msgspec strict → ValidationError on unknown fields). Augmentation checkboxes in the
            # UI are saved in the .yml for round-trip fidelity but NOT sent to train.py.
            # use_hflip / use_rot remain in DatasetOptions and work correctly.

            # Network — dyn_ keys are arch params EXCEPT these UI-only ones:
            _DYN_NOT_NET = {"gan_weight"}
            # Abbreviated upsampler names — only for archs that use them.
            # gfisrv2, smosr, neosr-based archs expect the full name (e.g. "pixelshuffledirect").
            # spanplus, spanc, spanf use the short form ("ps", "dys", "conv").
            _UPSAMPLER_ALIASES = {
                "dysample": "dys", "dysample++": "dys",
                "pixelshuffle": "ps", "pixel_shuffle": "ps",
                "convolution": "conv",
            }
            _UPSAMPLER_ALIAS_ARCHS = {"spanplus", "spanc", "spanf", "span", "spanpp"}
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
                    # Toute la famille span* (spanplus_s/st/sts, spanf, spanc, spanpp…) attend
                    # les noms COURTS (dys/ps/conv) → on aliase. gfisrv2/smosr/neosr gardent le
                    # nom complet (non-span) donc exclus par le prefixe.
                    if field == "upsampler" and (arch_type.lower().startswith("span")
                                                 or arch_type.lower() in _UPSAMPLER_ALIAS_ARCHS):
                        v = _UPSAMPLER_ALIASES.get(str(v).lower().strip(), v)
                    # Listes/tuples Python natives (defaults d'arch : depths,
                    # num_heads…) : caster les éléments entiers en int. Sans ça
                    # un default [6.0, 6.0, …] passe verbatim et neosr fait
                    # torch.linspace(0, dpr, sum(depths)) avec steps=float →
                    # "linspace invalid combination" / "'float' cannot be
                    # interpreted as an integer". (l.991 ne traite que les str.)
                    if isinstance(v, (list, tuple)):
                        casted = []
                        for _x in v:
                            if isinstance(_x, bool):
                                casted.append(_x)
                            elif isinstance(_x, (int, float)):
                                casted.append(int(_x) if float(_x).is_integer() else float(_x))
                            elif isinstance(_x, str):
                                try:
                                    _fx = float(_x.strip())
                                    casted.append(int(_fx) if _fx.is_integer() else _fx)
                                except Exception:
                                    casted.append(_x)
                            else:
                                casted.append(_x)
                        net_g[field] = casted
                        continue
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
                    # Convertir "true"/"false" string → bool Python
                    # (ex: champ Rep de smosr vient d'un combobox string)
                    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
                        net_g[field] = (v.strip().lower() == "true")
                        continue
                    # A real bool (config loaded from YAML, no sidecar) stays a bool:
                    # float(True) made it 1.
                    if isinstance(v, bool):
                        net_g[field] = v
                        continue
                    try:
                        f = float(v)
                        net_g[field] = int(f) if f.is_integer() else f
                    except (ValueError, TypeError):
                        net_g[field] = v
            config["network_g"] = net_g

            # MambaIR/MambaIRv2: leur __init__ attend "upscale", pas "scale" — et
            # ce champ n'est pas exposé comme dyn_ param dans l'UI (descriptions.py),
            # donc jamais injecté par la boucle dyn_ ci-dessus. Sans ça, upscale
            # retombe sur le défaut de la classe (2) quel que soit le scale choisi.
            # (root cause du bug scale/upscale — précédemment corrigé seulement dans
            # _generate_neosr_toml, jamais ici, dans le générateur réellement utilisé.)
            if arch_type in ("mambair", "mambairv2"):
                config["network_g"]["upscale"] = scale

            # TFDAT v2 builds its fusion layer for exactly clip_size frames (fuse = dim *
            # clip_size): the network must match the dataset's clip, which is the one
            # place the user sets it.
            if arch_type == "tfdatv2" and isinstance(config.get("datasets", {}).get("train"), dict):
                config["network_g"]["clip_size"] = int(config["datasets"]["train"].get("clip_size", 5))

            # SpanC multi-scale: auto-set top-level scale = max(scale_list) si plusieurs échelles
            if arch_type == "spanc" and isinstance(net_g.get("scale_list"), (list, tuple)) and len(net_g["scale_list"]) > 1:
                config["scale"] = int(max(net_g["scale_list"]))

            use_gan = safe_bool("use_gan", False)
            if use_gan:
                config["network_d"] = {"type": data.get("net_d_type", "dunet")}
                # Per-discriminator params from the UI's "dd_" fields (config_tab.py's
                # _on_disc_change) — was entirely missing here (this is the LIVE
                # generator, _generate_neosr_toml below it is NOT what's used for
                # Redux configs — same dead-code trap as the scale/upscale bug
                # documented above at "ce champ n'est pas exposé comme dyn_ param").
                def _dd_convert(v):
                    if isinstance(v, bool):
                        return v
                    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
                        return v.strip().lower() == "true"
                    try:
                        f = float(v)
                        return int(f) if f.is_integer() else f
                    except (ValueError, TypeError):
                        return v
                for k, v in data.items():
                    if k.startswith("dd_") and v is not None and str(v).strip() != "":
                        config["network_d"][k.replace("dd_", "", 1)] = _dd_convert(v)
                if str(config["network_d"].get("type", "")).lower() == "dunet":
                    # traiNNer-redux DUnet(num_in_ch, num_feat): the UI field is "dim"
                    _nd = config["network_d"]
                    if "dim" in _nd:
                        _nd["num_feat"] = _nd.pop("dim")
                    _nd.pop("n_blocks", None)

            # 2nd discriminator (network_d2/gan_opt2/optim_d2) — generic 2-discriminator
            # GAN support (config_tab.py "Activer un 2e discriminateur"). Requires D1
            # (use_gan) to also be enabled; mirrors the network_d/dd_ pattern above with
            # a "dd2_" prefix.
            use_disc2 = use_gan and safe_bool("use_disc2", False)
            if use_disc2:
                config["network_d2"] = {"type": data.get("net_d2_type", "unetdiscriminatorsn")}
                for k, v in data.items():
                    if k.startswith("dd2_") and v is not None and str(v).strip() != "":
                        config["network_d2"][k.replace("dd2_", "", 1)] = _dd_convert(v)

            # Path
            config["path"] = {
                "param_key_g": None,
                "strict_load_g": True,
                "resume_state": None,
            }
            pretrain = data.get("pretrain_model", "")
            if pretrain:
                config["path"]["pretrain_network_g"] = pretrain.replace("\\", "/")
            pretrain_d = data.get("pretrain_model_d", "")
            if use_gan and pretrain_d:
                config["path"]["pretrain_network_d"] = pretrain_d.replace("\\", "/")
            # Second discriminator weights (engine patch: PathOptions.pretrain_network_d2).
            pretrain_d2 = data.get("pretrain_model_d2", "")
            if use_disc2 and pretrain_d2:
                config["path"]["pretrain_network_d2"] = pretrain_d2.replace("\\", "/")
            resume = data.get("resume_state", "")
            if resume:
                config["path"]["resume_state"] = resume.replace("\\", "/")
            # ECO (native in traiNNer-redux dev): teacher pretrain weights go to
            # path.pretrain_network_g_teacher (NOT the old custom eco_pretrain_g).
            eco_on = safe_bool("eco_mode", False)
            eco_pretrain = (data.get("eco_pretrain_path", "") or "").strip()
            if eco_on and eco_pretrain:
                config["path"]["pretrain_network_g_teacher"] = eco_pretrain.replace("\\", "/")

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
            sched_gamma = safe_num("gamma", 0.5, float)
            sched_t_max = safe_num("t_max", total_iter, int)
            sched_eta_min = safe_num("eta_min", 1e-7, float)
            if "cosineannealinglr" in sched_lower:
                sched_section = {"type": sched_type, "T_max": sched_t_max, "eta_min": sched_eta_min}
            elif "exponential" in sched_lower:
                sched_section = {"type": sched_type, "gamma": sched_gamma}
            elif "plateau" in sched_lower:
                sched_section = {"type": sched_type, "patience": 10, "factor": 0.5}
            elif "steplr" in sched_lower and "multi" not in sched_lower:
                sched_section = {"type": sched_type, "step_size": max(1000, total_iter // 10), "gamma": sched_gamma}
            else:  # MultiStepLR default
                sched_section = {"type": sched_type, "milestones": milestones, "gamma": sched_gamma}

            train_section = {
                "ema_decay": safe_num("ema", 0.999, float),
                "ema_power": 0.75,
                "grad_clip": safe_bool("grad_clip", False),
                **({"grad_clip_max_norm": safe_num("grad_clip_max_norm", 100, float)} if safe_bool("grad_clip", False) else {}),
                "optim_g": {
                    "type": data.get("optim_g", "AdamW"),
                    "lr": lr,
                    "weight_decay": 0,
                    "betas": [0.9, 0.99],
                },
                "scheduler": sched_section,
                "total_iter": total_iter,
                "warmup_iter": safe_num("warmup_iter", -1, int),
                "adaptive_d": safe_bool("adaptive_d", False),
                # Delayed GAN start (engine patches gan_start_iter / gan_auto_* /
                # gan_restart_lr). Written only when used.
                **_gan_start_section(data, safe_num, safe_bool),
                # MoA (Mixture of Augmentations) — in TrainOptions, NOT top-level.
                "use_moa": bool(_moa_active),
                "moa_augs": _moa_augs_list,
                "moa_probs": _moa_probs_list,
                "moa_debug": safe_bool("moa_debug", False),
                "moa_debug_limit": safe_num("moa_debug_limit", 100, int),
            }

            if use_gan:
                train_section["optim_d"] = {
                    "type": "AdamW",
                    "lr": safe_num("lr_d", lr, float),
                    "weight_decay": 0,
                    "betas": [0.9, 0.99],
                }

            if use_disc2:
                # Second discriminator's own optimizer -- reuses the same LR as D1
                # (lr_d) by default since the UI doesn't expose a separate lr_d2
                # field; can be edited by hand in the generated YAML if a different
                # LR is desired.
                train_section["optim_d2"] = {
                    "type": "AdamW",
                    "lr": safe_num("lr_d", lr, float),
                    "weight_decay": 0,
                    "betas": [0.9, 0.99],
                }
                # gan_opt2 is intentionally NOT appended to the `losses` list below
                # (unlike D1's ganloss): the engine (traiNNer/models/sr_model.py)
                # keeps it out of the generic losses loop to avoid a label collision
                # with l_g_gan (both use type "ganloss"), building it as its own
                # "l_g_gan2" entry instead. See engine_patches.py for the matching
                # engine-side patch. Same gan_type as D1 (single UI selector) but its
                # own loss_weight (gan_weight_d2), so relative D1/D2 weighting is
                # just the ratio of the two loss_weight sliders.
                train_section["gan_opt2"] = {
                    "type": "ganloss",
                    "gan_type": data.get("gan_type", "vanilla"),
                    "loss_weight": safe_num("gan_weight_d2", 0.05, float),
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
                _VGG_ALL_SAVE = ["conv1_2","conv2_2","conv3_2","conv3_4","conv4_2","conv4_4","conv5_2","conv5_4"]
                _vgg_lw_save = {_l: safe_num(f"percep_vgg_{_l}", 0.0, float) for _l in _VGG_ALL_SAVE}
                _vgg_lw_active = {k: v for k, v in _vgg_lw_save.items() if v != 0.0}
                # fd / pd work on the VGG relu layers: with conv layer names the engine had
                # no criterion for them (AssertionError at the first iteration). fd / pd ->
                # the same layers as reluX_Y; fd+l1 / pd+l1 -> conv (L1) + relu (fd/pd).
                # fd also needs exactly ONE relu layer per VGG stage, in order (its random
                # projections are sized per stage): the engine's own set for the fd side.
                _FD_RELU = {"relu1_2": 0.1, "relu2_2": 0.1, "relu3_4": 1.0, "relu4_4": 1.0, "relu5_4": 1.0}
                if _vgg_lw_active and crit == "pd":
                    _vgg_lw_active = {k.replace("conv", "relu", 1): v for k, v in _vgg_lw_active.items()}
                elif _vgg_lw_active and crit == "pd+l1":
                    _vgg_lw_active = {**_vgg_lw_active,
                                      **{k.replace("conv", "relu", 1): v for k, v in _vgg_lw_active.items()}}
                elif crit == "fd":
                    _vgg_lw_active = dict(_FD_RELU)
                elif crit == "fd+l1":
                    _vgg_lw_active = {**_vgg_lw_active, **_FD_RELU}
                _percep_entry = {"type": "perceptualloss", "criterion": crit,
                                 "loss_weight": safe_num("weight_loss_percep", 0.01, float)}
                if _vgg_lw_active:
                    _percep_entry["layer_weights"] = _vgg_lw_active
                losses.append(_percep_entry)
            if safe_bool("loss_ldl", False):
                losses.append({"type": "ldlloss", "loss_weight": safe_num("weight_loss_ldl", 1.0, float), "criterion": data.get("ldl_criterion", "l1")})
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
            # SobelEdgeLoss — native in traiNNer-redux dev (sobel_edge_loss.py).
            # Penalizes gradient magnitude diff (Sobel X+Y on luma). Sharpens edges + high-freq.
            # Reuses the loss_edge / weight_loss_edge widgets from the NeoSR EdgeLoss section.
            if safe_bool("loss_edge", False):
                losses.append({"type": "sobeledgeloss", "loss_weight": safe_num("weight_loss_edge", 0.05, float)})
            if safe_bool("loss_contextual", False):
                losses.append({"type": "contextualloss", "loss_weight": safe_num("weight_loss_contextual", 1.0, float), "distance_type": data.get("ctx_distance_type", "cosine"), "band_width": safe_num("ctx_band_width", 0.5, float)})
            # LineDistillerLoss — distils line-art / edge structure (Anime2Sketch-style
            # line extractor). Great for anime / line-art SR. (traiNNer-redux dev)
            if safe_bool("loss_linedistiller", False):
                losses.append({"type": "linedistillerloss",
                               "loss_weight": safe_num("weight_loss_linedistiller", 1.0, float),
                               "criterion": data.get("linedistiller_criterion", "l1")})
            # PerceptualFP16Loss — perceptual loss computed in FP16 (lower VRAM). (Redux dev)
            if safe_bool("loss_percep_fp16", False):
                losses.append({"type": "perceptualfp16loss",
                               "loss_weight": safe_num("weight_loss_percep_fp16", 1.0, float),
                               "criterion": data.get("percep_fp16_criterion", "charbonnier")})
            # AESOPLoss (AESOP-SR, CVPR 2025): distance after a frozen pretrained
            # autoencoder instead of raw pixels. The engine needs the AE checkpoint
            # (safetensors, loaded with load_file), its decoder arch (decoder_opt) and the
            # scale; the decoder is the trained arch (the paper's recipe).
            if safe_bool("loss_aesop", False):
                _ae = (data.get("aesop_ae_path") or "").strip().replace("\\", "/")
                if not _ae:
                    return False, ("AESOP : renseigner le checkpoint de l'autoencodeur (.safetensors) "
                                   "à côté de la loss, ou décocher AESOP. / AESOP: set the autoencoder "
                                   "checkpoint (.safetensors) next to the loss, or untick AESOP.")
                if not _ae.lower().endswith(".safetensors"):
                    return False, ("AESOP : le moteur ne charge que des .safetensors pour l'autoencodeur. / "
                                   "AESOP: the engine only loads a .safetensors autoencoder.")
                _dec = {k: v for k, v in (config.get("network_g") or {}).items() if k != "scale"}
                losses.append({"type": "aesoploss",
                               "loss_weight": safe_num("weight_loss_aesop", 1.0, float),
                               "criterion": data.get("aesop_criterion", "charbonnier"),
                               "decoder_opt": _dec,
                               "scale": int(config.get("scale", safe_num("scale", 4, int))),
                               "pretrain_network_ae": _ae})
            # BCEWithLogitsDiceLoss — BCE + Dice (segmentation-style). (Redux dev)
            if safe_bool("loss_bcedice", False):
                losses.append({"type": "bcewithlogitsdiceloss",
                               "loss_weight": safe_num("weight_loss_bcedice", 1.0, float),
                               "weight_bce": safe_num("bcedice_weight_bce", 1.0, float),
                               "weight_dice": safe_num("bcedice_weight_dice", 1.0, float)})
            if safe_bool("loss_percep_anime", False):
                # PerceptualAnimeLoss — ResNet50 backbone, APISR-ported, anime-specific perceptual loss
                from src.core import engine_paths as _ep
                _pa_available = os.path.exists(os.path.join(
                    _ep.redux_path(), "traiNNer", "losses", "perceptual_anime_loss.py"))
                if _pa_available:
                    losses.append({"type": "PerceptualAnimeLoss",
                                   "loss_weight": safe_num("weight_loss_percep_anime", 1.0, float),
                                   "criterion": data.get("percep_anime_criterion", "l1")})
                # else: absent from this traiNNer-redux version — silently skip
            if safe_bool("loss_spark", False):
                # Check if SparkLoss is available in this traiNNer-redux version.
                # It was removed from the official repo; skip silently if absent (avoids
                # "No object named 'sparkloss' found in 'loss' registry" crash).
                from src.core import engine_paths as _ep
                _spark_available = os.path.exists(os.path.join(
                    _ep.redux_path(), "traiNNer", "losses", "spark_loss.py"))
                if _spark_available:
                    _spark_criterion = data.get("spark_criterion", "fd")
                    # criterion="fd": forward_once() uses F.conv2d with patch_size=4×4 kernel on InceptionNeXt features.
                    # At depth ÷32 with pad=32: feature_map = (lq_size + 64) / 32 - 2.
                    # lq_size=96  → 3×3 feature map < 4×4 kernel → RuntimeError: kernel size > input size.
                    # lq_size=128 → 4×4 feature map = 4×4 kernel → OK.
                    # criterion="charbonnier": forward_once() never called → no size constraint.
                    if _spark_criterion == "fd" and lq_size < 128:
                        lq_size = 128
                        train_ds["lq_size"] = lq_size
                    _spark_entry = {"type": "SparkLoss", "loss_weight": safe_num("weight_loss_spark", 0.2, float), "criterion": _spark_criterion}
                    _spark_path = (data.get("spark_path") or "").strip()
                    if _spark_path:
                        _spark_entry["path"] = _spark_path
                    _spark_max_score = safe_num("spark_max_score", 50.0, float)
                    if _spark_max_score > 0:
                        _spark_entry["max_score"] = _spark_max_score
                    losses.append(_spark_entry)
                # else: SparkLoss absent from this traiNNer-redux version — silently skip
            if use_gan:
                _gan_entry = {"type": "ganloss", "gan_type": data.get("gan_type", "vanilla"),
                              "loss_weight": safe_num("dyn_gan_weight", 0.05, float)}
                # real_label_val/fake_label_val were exposed in the UI and written for
                # NeoSR but never for Redux — silently dropped (2026-09-17 fix).
                _real_lbl = safe_num("real_label_val", 1.0, float)
                _fake_lbl = safe_num("fake_label_val", 0.0, float)
                if _real_lbl != 1.0:
                    _gan_entry["real_label_val"] = _real_lbl
                if _fake_lbl != 0.0:
                    _gan_entry["fake_label_val"] = _fake_lbl
                losses.append(_gan_entry)
            if not losses:
                losses.append({"type": "charbonnierloss", "loss_weight": 1.0})
            train_section["losses"] = losses

            # ECO (native traiNNer-redux dev): nested eco struct + teacher network.
            # The teacher defaults to the same arch as the student (network_g).
            if eco_on:
                train_section["eco"] = {
                    "enabled": True,
                    "end_ratio": safe_num("eco_end_ratio", 0.75, float),
                    "mode": (data.get("eco_mode_kind") or "full").strip().lower()
                            if (data.get("eco_mode_kind") or "").strip().lower() in ("full", "hr_only")
                            else "full",
                }
                # network_g_teacher mirrors the student arch unless already provided.
                import copy as _copy
                config["network_g_teacher"] = _copy.deepcopy(config.get("network_g", {}))

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
            # USS-only (stripped at launch): add the USRS texture metrics to val.metrics
            config["usrs_texture_metrics"] = safe_bool("metric_texture", True)

            # Logger
            config["logger"] = {
                "print_freq": safe_num("print_freq", 100, int),
                "save_checkpoint_freq": safe_num("save_freq", 1000, int),
                "save_checkpoint_format": "safetensors",
                "use_tb_logger": safe_bool("use_tb_logger", True),
            }

            # Monitoring — stored in user YAML as a USS metadata section.
            # Stripped by tab_run._strip_custom_keys_from_yaml before sending to traiNNer-redux,
            # so msgspec never sees it. Reader restores it on config reload.
            config["monitoring"] = {
                "auto_tensorboard": safe_bool("auto_tensorboard", False),
                "auto_ngrok": safe_bool("auto_ngrok", False),
                "port": safe_num("port_tb", 6006, int),
            }

            # NOTE SpectralDiscriminator : patch_size doit diviser le crop (lq_size*scale)
            # sinon EinopsError dans spectrum_transform() au 1er iter. On ne corrige plus
            # patch_size silencieusement ici (l'utilisateur préfère garder sa valeur et
            # ajuster lq_size lui-même) — l'avertissement actif est dans config_tab.py
            # (_check_disc_pair_warning) et un filet de sécurité runtime (dernier
            # recours, avec avertissement console) reste dans spectral_disc_arch.py.

            # Criteria the Redux losses accept (anything else raises at build_loss). An
            # old .uss.json can still hold a value the UI used to offer (huber, l2…):
            # fall back to the loss default instead of crashing at launch.
            for _l in (config.get("train") or {}).get("losses") or []:
                if not isinstance(_l, dict) or "criterion" not in _l:
                    continue
                _ok = _REDUX_LOSS_CRITERIA.get(str(_l.get("type", "")).lower())
                if _ok and _l["criterion"] not in _ok:
                    _l["criterion"] = _ok[0]

            if _return_cfg:
                return config

            # Phase degradations: the Dégradations tab's GAN profile (UI keys) is turned
            # into engine keys by a second pass, and only the keys that differ from the
            # PSNR profile are kept (deg_phase2, USS-only, read by runner.py).
            _p2 = data.get("deg_phase2_data")
            if is_otf and isinstance(_p2, dict) and _p2:
                _d2 = dict(data)
                _d2.update(_p2)
                _d2.pop("deg_phase2_data", None)
                _cfg2 = self._gen_redux(_d2, save_path, _return_cfg=True)
                if isinstance(_cfg2, dict):
                    from src.core.otf_custom_degradations import _ALL_CUSTOM_KEYS as _CK
                    _keys = [k for k in _PHASE_DEG_KEYS if k in config or k in _cfg2]
                    _keys += [k for k in _CK if k in config or k in _cfg2]
                    _diff = {}
                    for _k in _keys:
                        _v1, _v2 = config.get(_k), _cfg2.get(_k)
                        if _v2 is None and _k.endswith("_prob"):
                            _v2 = 0.0  # custom degradation off in the GAN profile
                        if _v2 is not None and _v1 != _v2:
                            _diff[_k] = _v2
                    if _diff:
                        config["deg_phase2"] = _diff
                        config["deg_phase2_ramp"] = safe_num("deg_phase2_ramp", 15000, int)

            # Phase losses: the GAN loss profile (UI keys) goes through the same second
            # pass; every loss of either profile stays in train.losses (built at start,
            # no VRAM surprise at the GAN start) and loss_phase2 (USS-only, read by
            # runner.py) holds [PSNR weight, GAN weight] for the losses that change.
            _lp2 = data.get("loss_phase2_data")
            if isinstance(_lp2, dict) and _lp2:
                _dl = dict(data)
                _dl.update(_lp2)
                _dl.pop("loss_phase2_data", None)
                _dl.pop("deg_phase2_data", None)
                _cfgl = self._gen_redux(_dl, save_path, _return_cfg=True)
                if isinstance(_cfgl, dict):
                    def _wmap(c):
                        out = {}
                        for _l in (c.get("train") or {}).get("losses") or []:
                            if isinstance(_l, dict) and "gan" not in str(_l.get("type", "")).lower():
                                out[str(_l.get("type"))] = _l
                        return out
                    _m1, _m2 = _wmap(config), _wmap(_cfgl)
                    _lphase = {}
                    for _t in list(_m1) + [t for t in _m2 if t not in _m1]:
                        _w1 = float(_m1[_t].get("loss_weight", 0)) if _t in _m1 else 0.0
                        _w2 = float(_m2[_t].get("loss_weight", 0)) if _t in _m2 else 0.0
                        if _w1 == _w2:
                            continue
                        _lphase[_t] = [_w1, _w2]
                        if _t not in _m1:
                            # GAN-profile-only loss: built at start, weight 0 before the GAN
                            config["train"]["losses"].append(dict(_m2[_t]))
                    if _lphase:
                        config["loss_phase2"] = _lphase
                        config["loss_phase2_ramp"] = safe_num("loss_phase2_ramp", 5000, int)

            class _NullTilde(yaml.Dumper):
                pass
            _NullTilde.add_representer(
                type(None),
                lambda dumper, _: dumper.represent_scalar("tag:yaml.org,2002:null", "~")
            )

            def _dump(cfg):
                return yaml.dump(cfg, Dumper=_NullTilde, sort_keys=False,
                                 allow_unicode=True, default_flow_style=False)

            plain = _dump(config)
            # Sectioned (readable) variant — reorder top-level keys + comment headers.
            # Hard gate: it is used ONLY if it re-parses to the EXACT same dict.
            final_text = plain
            try:
                ordered, first_of = _build_section_index(config, _REDUX_SECTIONS)
                sectioned = _inject_section_comments(_dump(ordered), first_of, _yaml_key)
                if yaml.safe_load(sectioned) == config:
                    final_text = sectioned
            except Exception:
                final_text = plain
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(final_text)
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
                # NB: x.isdigit() est False sur " 6" (espace après split) → l'ancienne
                # version sortait [6, 6.0, 6.0, …] (1er sans espace=int, reste float) →
                # neosr torch.linspace(steps=float) / "'float' cannot be interpreted as
                # an integer". On strip + caste les entiers via is_integer().
                try:
                    return [int(float(x.strip())) if float(x.strip()).is_integer() else float(x.strip())
                            for x in val.strip("[]").split(",") if x.strip()]
                except Exception: return val
            if isinstance(val, float): return int(val) if val.is_integer() else val
            try:
                if "." in str(val): return float(val)
                return int(val)
            except Exception:
                return True if str(val).lower()=="true" else False if str(val).lower()=="false" else val

        scale = safe_num("scale", 4, int)
        scale = _clamp_scale_for_arch(data.get("arch", "omnisr"), scale)
        total_iter = safe_num("total_iter", 100000, int)

        # SCHEDULER
        sched_type = data.get("scheduler", "MultiStepLR")
        sched_config = {"type": sched_type}
        sched_lower = str(sched_type).lower()
        sched_gamma = safe_num("gamma", 0.5, float)
        sched_t_max = safe_num("t_max", total_iter, int)
        sched_eta_min = safe_num("eta_min", 1e-7, float)
        if "multistep" in sched_lower:
            ms = data.get("milestones", "75000, 112500")
            try:
                sched_config["milestones"] = [int(x) for x in str(ms).split(",")]
            except Exception:
                sched_config["milestones"] = [75000, 112500]
            sched_config["gamma"] = sched_gamma
        elif "cosineannealingrestarts" in sched_lower or "cosine" in sched_lower and "warm" in sched_lower:
            sched_config["periods"] = [total_iter // 4] * 4
            sched_config["restart_weights"] = [1, 0.5, 0.25, 0.125]
            sched_config["eta_min"] = sched_eta_min
        elif "cosine" in sched_lower:
            # CosineAnnealingLR / CosineAnnealing — uses T_max + eta_min, NOT milestones
            sched_config["T_max"] = sched_t_max
            sched_config["eta_min"] = sched_eta_min
        elif "step" in sched_lower:
            # StepLR
            sched_config["step_size"] = max(1000, total_iter // 10)
            sched_config["gamma"] = sched_gamma
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
        if safe_bool("metric_dists", False): val_metrics["dists"] = {"type": "calculate_dists", "better": "lower"}
        if safe_bool("metric_topiq", False): val_metrics["topiq"] = {"type": "calculate_topiq"}
        # NOTE: LPIPS/NIQE volontairement absents — ce fork neosr n'implémente PAS
        # calculate_lpips/calculate_niqe (vérifié dans neosr/neosr/metrics/calculate.py,
        # seuls psnr/ssim/dists/topiq existent). Les exposer générerait un TOML qui
        # plante au premier calcul de métrique de validation.

        # AUGMENTATIONS
        augs = ["none"]; probs = [0.4]
        if safe_bool("aug_mixup", False): augs.append("mixup"); probs.append(round(safe_num("prob_aug_mixup", 0.15, float), 2))
        if safe_bool("aug_cutmix", False): augs.append("cutmix"); probs.append(round(safe_num("prob_aug_cutmix", 0.15, float), 2))
        if safe_bool("aug_resizemix", False): augs.append("resizemix"); probs.append(round(safe_num("prob_aug_resizemix", 0.15, float), 2))
        if safe_bool("aug_cutblur", False): augs.append("cutblur"); probs.append(round(safe_num("prob_aug_cutblur", 0.15, float), 2))
        # neosr exige batch>1 pour les augmentations MoA (mixup/cutmix mélangent
        # des échantillons → impossible à batch=1) ; même `["none"]` déclenche
        # "Augmentations need batch >1 to work." → on désactive l'aug au batch=1.
        if safe_num("batch_size", 4, int) <= 1:
            augs, probs = [], []

        dataset_mode = data.get("dataset_mode", "otf")
        # NeoSR n'a pas de type "bicubic" — c'est un alias pour "otf" avec dégradations légères
        model_type = "otf" if dataset_mode in ("otf", "bicubic") else "image"

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
            # NeoSR stops with "bfloat16 option has no effect without use_amp": BF16 is
            # the AMP dtype, written only with AMP on
            "use_amp": safe_bool("use_amp", False),
            "bfloat16": safe_bool("bfloat16", False) and safe_bool("use_amp", False),
            "fast_matmul": safe_bool("fast_matmul", False),
            "compile": safe_bool("compile", False),
            # monitoring NOT included — traiNNer-redux msgspec rejects unknown fields
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
                "smooth_banding_prob": safe_num("smooth_banding_prob", 0.0, float),
                "smooth_banding_levels_range": safe_list("smooth_banding_levels_range", [48, 128]),
                "smooth_banding_dark_prob": safe_num("smooth_banding_dark_prob", 0.5, float),
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
                # Custom 3/4/5 degradations — see matching fix in _gen_redux (2026-09-17).
                "screentone_prob": safe_num("screentone_prob", 0.0, float),
                "screentone_dot_size": safe_list("screentone_dot_size", [7, 15]),
                "screentone_angle": safe_list("screentone_angle", [0, 90]),
                "screentone_dot_type": data.get("screentone_dot_type", "circle"),
                "screentone_color_space": data.get("screentone_color_space", "rgb"),
                "dithering_prob": safe_num("dithering_prob", 0.0, float),
                "dithering_color_ch": safe_list("dithering_color_ch", [2, 8]),
                "dithering_type": data.get("dithering_type", "floyd-steinberg"),
                "pixelate_prob": safe_num("pixelate_prob", 0.0, float),
                "pixelate_size": safe_list("pixelate_size", [2, 16]),
                "sin_prob": safe_num("sin_prob", 0.0, float),
                "sin_shape": safe_list("sin_shape", [100, 600]),
                "sin_alpha": safe_list("sin_alpha", [0.1, 0.4]),
                "sin_bias": safe_list("sin_bias", [0.8, 1.2]),
                "sin_orientation": data.get("sin_orientation", "random"),
                "subsampling_prob": safe_num("subsampling_prob", 0.0, float),
                "subsampling_format": data.get("subsampling_format", "4:2:0"),
                "subsampling_yuv": data.get("subsampling_yuv", "601"),
                "dct_block_prob": safe_num("dct_block_prob", 0.0, float),
                "dct_block_quality_range": safe_list("dct_block_quality_range", [20, 80]),
                "dct_block_size_range": safe_list("dct_block_size_range", [4, 4]),
                "dct_block_chroma_scale_range": safe_list("dct_block_chroma_scale_range", [1.5, 2.5]),
                "dct_block_yuv": data.get("dct_block_yuv", "601"),
                "color_level_prob": safe_num("color_level_prob", 0.0, float),
                "color_level_high": safe_list("color_level_high", [220, 255]),
                "color_level_low": safe_list("color_level_low", [0, 35]),
                "color_level_gamma": safe_list("color_level_gamma", [0.7, 1.5]),
                "wtp_halo_prob": safe_num("wtp_halo_prob", 0.0, float),
                "wtp_halo_strength": safe_list("wtp_halo_strength", [0.1, 0.5]),
                "wtp_halo_radius": safe_list("wtp_halo_radius", [3, 12]),
                "saturation_prob": safe_num("saturation_prob", 0.0, float),
                "saturation_range": safe_list("saturation_range", [0.3, 1.8]),
                "shift_prob": safe_num("shift_prob", 0.0, float),
                "shift_range": safe_list("shift_range", [1, 8]),
                "shift_axis": data.get("shift_axis", "random"),
                "disc_blur_prob": safe_num("disc_blur_prob", 0.0, float),
                "disc_blur_radius_range": safe_list("disc_blur_radius_range", [2, 8]),
                "vignette_prob": safe_num("vignette_prob", 0.0, float),
                "vignette_strength_range": safe_list("vignette_strength_range", [0.2, 0.6]),
                "vignette_radius_range": safe_list("vignette_radius_range", [0.45, 0.75]),
                "quantize_depth_prob": safe_num("quantize_depth_prob", 0.0, float),
                "quantize_depth_bits_range": safe_list("quantize_depth_bits_range", [4, 7]),
                "coupled_optical_prob": safe_num("coupled_optical_prob", 0.0, float),
                "coupled_vintage_prob": safe_num("coupled_vintage_prob", 0.0, float),
            },
            "datasets": {
                "train": {
                    "type": ("otf" if dataset_mode == "bicubic" else dataset_mode), "name": "TrainSet", "dataroot_gt": data.get("dataroot_gt", ""),
                    "num_worker_per_gpu": safe_num("num_worker_per_gpu", 4, int), "prefetch_mode": data.get("prefetch_mode", "cuda"),
                    "batch_size": safe_num("batch_size", 4, int), 
                    "accumulate": safe_num("accumulate", 1, int),
                    "patch_size": safe_num("patch_size", 64, int),
                    "use_shuffle": True, "use_hflip": safe_bool("use_hflip", True), "use_rot": safe_bool("use_rot", True),
                    "augmentation": augs, "aug_prob": probs,
                    "moa_debug": safe_bool("moa_debug", False),
                    "moa_debug_limit": safe_num("moa_debug_limit", 100, int),
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
                "adaptive_d": safe_bool("adaptive_d", False),
                "ema": safe_num("ema", 0.999, float),
                "grad_clip": safe_bool("grad_clip", False),
                **({"grad_clip_max_norm": safe_num("grad_clip_max_norm", 100, float)} if safe_bool("grad_clip", False) else {}),
                # "eco" removed — traiNNer-redux dropped this field in newer versions (msgspec strict)
                "match_lq_colors": safe_bool("match_lq_colors", False),
                "optim_g": {
                    "type": optim_type, "lr": safe_num("lr", 5e-5, float), 
                    "weight_decay": weight_decay_val, "betas": betas_vals
                },
                "scheduler": sched_config,
            },
            "logger": {"total_iter": total_iter, "print_freq": safe_num("print_freq", 100, int), "save_checkpoint_freq": safe_num("save_freq", 5000, int), "use_tb_logger": safe_bool("use_tb_logger", True)},
            "val": {"val_freq": safe_num("val_freq", 5000, int), "save_img": safe_bool("save_img", True), "pbar": True, "tile": safe_num("tile", 200, int), "tile_pad": safe_num("tile_pad", 32, int), "metrics": val_metrics},
            # USS metadata — not read by neosr itself, only by this app on reload.
            "monitoring": {
                "auto_tensorboard": safe_bool("auto_tensorboard", False),
                "auto_ngrok": safe_bool("auto_ngrok", False),
                "port": safe_num("port_tb", 6006, int),
            },
        }

        # Validation off: no [val] / datasets.val at all. NeoSR validates whenever a [val]
        # table exists (train.py: opt.get("val") is not None), so with "Validation" unticked
        # it still read the val folders, and crashed at launch if they did not exist.
        if not safe_bool("val_enabled", True):
            config.pop("val", None)
            config["datasets"].pop("val", None)

        # ECO (NeoSR): a plain train.eco flag. NeoSR blends towards the network's own
        # outputs (no separate teacher; it uses pretrain_network_g when set, else starts
        # after eco_init iterations). It was never written before, so ECO did nothing.
        if safe_bool("eco_mode", False) or safe_bool("use_eco", False):
            config["train"]["eco"] = True
            config["train"]["eco_init"] = safe_num("eco_init", 15000, int)
            config["train"]["eco_iters"] = safe_num("eco_iters", 80000, int)
            _sched = str(data.get("eco_schedule", "sigmoid")).strip().lower()
            config["train"]["eco_schedule"] = _sched if _sched in ("sigmoid", "linear") else "sigmoid"

        # Gestion du chemin d'expérience personnalisé
        if data.get("custom_exp_path"):
            config["path"]["results_root"] = data.get("custom_exp_path").replace("\\", "/")

        # Gestion du dossier LQ (seulement si Paired et chemin rempli)
        # NeoSR otf reject dataroot_lq — ne pas l'ajouter en mode otf ou bicubic
        _neosr_ds_type = config["datasets"]["train"].get("type", "")
        if data.get("dataroot_lq") and _neosr_ds_type not in ("otf",):
            config["datasets"]["train"]["dataroot_lq"] = data.get("dataroot_lq").replace("\\", "/")

        # Schedule-free follows the optimizer: NeoSR requires schedule_free with *_SF
        # optimizers ("The option 'schedule_free' must be in the config file", exit) and
        # Adam/AdamW/... reject it (unexpected keyword 'schedule_free'). The checkbox alone
        # could produce either crash.
        for _ok in ("optim_g", "optim_d"):
            _o = config["train"].get(_ok)
            if not isinstance(_o, dict):
                continue
            if str(_o.get("type", "")).lower().endswith("_sf"):
                _o["schedule_free"] = True
                if _ok == "optim_g":
                    _o["warmup_steps"] = safe_num("warmup_steps", -1, int)
            else:
                _o.pop("schedule_free", None)
                _o.pop("warmup_steps", None)

        # LOSSES DYNAMIC ADDITION
        if safe_bool("loss_pixel", True):
            _pix_t = data.get("pixel_criterion", "L1Loss")
            config["train"]["pixel_opt"] = {"type": _LOSS_TYPE_MAP.get(_pix_t, _pix_t), "loss_weight": safe_num("weight_loss_pixel", 1.0, float), "reduction": data.get("pixel_reduction", "mean")}

        if safe_bool("loss_wavelet", False):
            # NeoSR stops at start-up with "Wavelet-Guided requires GAN"
            if not safe_bool("use_gan", False):
                raise ValueError("Wavelet Guided (NeoSR) nécessite le GAN : activez le GAN "
                                 "ou décochez Wavelet Guided. / Wavelet Guided (NeoSR) requires the GAN.")
            config["train"]["wavelet_guided"] = True
            config["train"]["wavelet_init"] = safe_num("wavelet_init", 10000, int)
            config["train"]["wavelet_opt"] = {"type": "WaveletLoss", "loss_weight": safe_num("weight_loss_wavelet", 1.0, float)}

        if safe_bool("loss_fdl", False): config["train"]["fdl_opt"] = {"type": "fdl_loss", "loss_weight": safe_num("weight_loss_fdl", 1.0, float), "model": data.get("fdl_model", "vgg")}
        if safe_bool("loss_ff", False): config["train"]["ff_opt"] = {"type": "ff_loss", "loss_weight": safe_num("weight_loss_ff", 0.2, float), "alpha": safe_num("ff_alpha", 1.0, float)}
        if safe_bool("loss_ldl", False):
            # NeoSR ldl_loss only supports: l1, l2, huber, chc — not charbonnier
            _ldl_crit = data.get("ldl_criterion", "l1")
            if _ldl_crit not in ("l1", "l2", "huber", "chc"):
                _ldl_crit = "l1"
            config["train"]["ldl_opt"] = {"type": "ldl_loss", "loss_weight": safe_num("weight_loss_ldl", 1.0, float), "criterion": _ldl_crit, "ksize": safe_num("ldl_ksize", 7, int)}
        if safe_bool("loss_consistency", False):
            # neosr.losses.consistency_loss.__init__ signature is
            # (criterion, blur, cosim, saturation, brightness, loss_weight) --
            # must NOT be prefixed "use_*" or build_loss(**opt) raises
            # TypeError: consistency_loss.__init__() got an unexpected keyword
            # argument 'use_blur'.
            config["train"]["consistency_opt"] = {
                "type": "consistency_loss",
                "loss_weight": safe_num("weight_loss_consistency", 1.0, float),
                "blur": safe_bool("consistency_blur", True),
                "cosim": safe_bool("consistency_cosim", True),
                "saturation": safe_bool("consistency_saturation", True),
                "brightness": safe_bool("consistency_brightness", True),
            }
        # No edge loss in NeoSR (the engine never reads edge_opt): not written.

        if safe_bool("loss_mssim", False): config["train"]["mssim_opt"] = {"type": "mssim_loss", "loss_weight": safe_num("weight_loss_mssim", 1.0, float), "window_size": safe_num("mssim_window_size", 11, int), "sigma": safe_num("mssim_sigma", 1.5, float), "K1": safe_num("mssim_k1", 0.01, float), "K2": safe_num("mssim_k2", 0.03, float)}
        if safe_bool("loss_dists", False): config["train"]["dists_opt"] = {"type": "dists_loss", "loss_weight": safe_num("weight_loss_dists", 1.0, float)}
        if safe_bool("loss_msswd", False): config["train"]["msswd_opt"] = {"type": "msswd_loss", "loss_weight": safe_num("weight_loss_msswd", 1.0, float)}

        if safe_bool("loss_percep", False):
            _vgg_all = ["conv1_2","conv2_2","conv3_2","conv3_4","conv4_2","conv4_4","conv5_2","conv5_4"]
            _vgg_lw = {l: safe_num(f"percep_vgg_{l}", 0.0, float) for l in _vgg_all}
            _vgg_lw = {l: w for l, w in _vgg_lw.items() if w > 0}
            if not _vgg_lw:  # fallback: single-layer legacy
                _vgg_lw = {data.get("percep_layer", "conv5_4"): 1.0}
            config["train"]["perceptual_opt"] = {"type": "vgg_perceptual_loss", "loss_weight": safe_num("weight_loss_percep", 1.0, float), "criterion": _NEOSR_PERCEP_CRIT.get(str(data.get("percep_criterion", "huber")).lower(), "huber"), "layer_weights": _vgg_lw}

        # ARCH SPECIFIC
        arch = data.get("arch", "omnisr")
        if arch == "omnisr": config["network_g"]["upsampling"] = scale
        elif arch in ["span", "mambair", "mambairv2"]: config["network_g"]["upscale"] = scale
        elif arch in ["esrgan", "hat", "realplksr", "dat"]: config["network_g"]["scale"] = scale
        elif arch in _NEOSR_PRESET_NEED_SCALE: config["network_g"]["scale"] = scale
        # Presets neosr (hat_s/m/l, swinir_*, srformer_medium, aether_*) : self-contained,
        # ne PAS injecter les params dyn_ (collision embed_dim/img_size) ni canaux.
        if arch in _NEOSR_PRESET_ARCHS:
            config["network_g"].pop("num_in_ch", None)
            config["network_g"].pop("num_out_ch", None)
        else:
            for k, v in data.items():
                if k.startswith("dyn_") and k != "dyn_gan_weight": config["network_g"][k.replace("dyn_", "")] = _convert_val(v)
        # The scale is the project's, never an arch-field default: the dyn_ loop above wrote
        # e.g. omnisr upsampling = 4 (field default) in a 2x config -> 4x output, size
        # mismatch at the first loss (hat / dat / realplksr / omnisr).
        for _sk in ("upscale", "upsampling", "upscaling_factor"):
            if _sk in config["network_g"]:
                config["network_g"][_sk] = scale
        if "scale" in config["network_g"] and arch not in _NEOSR_PRESET_ARCHS:
            config["network_g"]["scale"] = scale

        # GAN
        if safe_bool("use_gan", False):
            config["train"]["gan_opt"] = {"type": "gan_loss", "gan_type": data.get("gan_type", "bce"), "loss_weight": safe_num("dyn_gan_weight", 0.05, float), "real_label_val": safe_num("real_label_val", 1.0, float), "fake_label_val": safe_num("fake_label_val", 0.0, float)}
            d_type = data.get("net_d_type", "unet"); d_config = {"type": d_type}
            if d_type in ["dunet", "metagan"]: d_config["in_ch"] = 3
            elif d_type != "ea2fpn": d_config["num_in_ch"] = 3
            config["network_d"] = d_config
            for k, v in data.items():
                if k.startswith("dd_"): config["network_d"][k.replace("dd_", "")] = _convert_val(v)
            config["train"]["optim_d"] = {"type": data.get("optim_g", "AdamW"), "lr": safe_num("lr_d", 5e-5, float), "weight_decay": 0, "betas": [0.9, 0.99]}
            if str(config["train"]["optim_d"]["type"]).lower().endswith("_sf"):
                config["train"]["optim_d"]["schedule_free"] = True  # NeoSR exits without it
            config["network_d"].pop("n_blocks", None) if d_type == "dunet" else None

        if not config["path"]["resume_state"]: del config["path"]["resume_state"]
        if not config["path"]["pretrain_network_g"]: del config["path"]["pretrain_network_g"]

        try:
            plain = toml.dumps(config)
            final_text = plain
            # Sectioned (readable) variant — gated on an exact re-parse to config.
            try:
                ordered, first_of = _build_section_index(config, _NEOSR_SECTIONS)
                sectioned = _inject_section_comments(toml.dumps(ordered), first_of, _toml_key)
                if tomllib.loads(sectioned) == tomllib.loads(plain):
                    final_text = sectioned
            except Exception:
                final_text = plain
            with open(path, "w", encoding="utf-8") as f:
                f.write(final_text)
            return True, f"Fichier NeoSR généré : {os.path.basename(path)}"
        except Exception as e:
            return False, f"Erreur écriture : {e}"