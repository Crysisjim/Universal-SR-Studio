"""
config_importer.py — Import community BasicSR-family YAML configs (Real-ESRGAN, SwinIR,
RCAN, HAT, BasicSR…) into the Universal SR Studio form.

The file is normalised (degradations found under datasets.train go to the root, where
the app's loader reads them), then read by the same loader as Configuration › Load
(ConfigHandler.load_config). That loader sets every loss toggle explicitly, so losses
absent from the file are switched off instead of staying on from the previous form
(B33), and it reads loss weights/criteria, scheduler milestones, batch size, network
parameters and paths. On top of it this module maps the community network name to an
architecture that really exists in the target engine, converts the crop size to the
app's LR patch size and reports everything it could not import.
"""
import os
import tempfile

import yaml

# community network_g.type (lower case) -> candidate app arch names, best first.
# The first candidate present in the engine's arch list wins.
_ARCH_CANDIDATES = {
    "rrdbnet": ["esrgan"],
    "srvggnetcompact": ["compact"],
    "swinir": ["swinir_m", "swinir_medium"],
    "hat": ["hat_m", "hat_s"],
    "rcan": ["rcan"],
    "omnisr": ["omnisr"],
    "dat": ["dat", "dat_s"],
    "srformer": ["srformer", "srformer_medium"],
    "realplksr": ["realplksr"],
    "plksr": ["plksr"],
    "span": ["span"],
    "atd": ["atd"],
    "drct": ["drct"],
    "upcunet": ["realcugan", "cugan"],
    "realcugan": ["realcugan", "cugan"],
    "grl": ["grl_b"],
}

_DEG_KEYS = [
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


def detect_config_format(config_path: str) -> str:
    """Config family, from network_g.type."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return "unknown"
    net_g = data.get("network_g", {})
    net_type = net_g.get("type", "").lower() if isinstance(net_g, dict) else ""
    for key, fmt in (("rrdb", "real_esrgan"), ("esrgan", "real_esrgan"), ("swinir", "swinir"),
                     ("rcan", "rcan"), ("hat", "hat"), ("omnisr", "omnisr")):
        if key in net_type:
            return fmt
    return "basicsr"


def _engine_archs(engine: str) -> set:
    try:
        from src.core.descriptions import get_arch_families
        return {a for lst in get_arch_families("redux" if "redux" in engine.lower() else "neosr").values()
                for a in lst}
    except Exception:
        return set()


def _map_arch(net_g: dict, engine: str):
    """(app arch or None, note)."""
    t = str(net_g.get("type", "")).strip()
    tl = t.lower()
    archs = _engine_archs(engine)
    cands = list(_ARCH_CANDIDATES.get(tl, []))
    # size variants
    emb = net_g.get("embed_dim")
    depths = net_g.get("depths") or []
    if tl == "swinir" and isinstance(emb, int):
        cands = (["swinir_s", "swinir_small"] if emb <= 60 else
                 ["swinir_l"] if emb >= 240 else ["swinir_m", "swinir_medium"]) + cands
    if tl == "hat" and isinstance(depths, list):
        cands = (["hat_l"] if len(depths) >= 12 else
                 ["hat_s"] if isinstance(emb, int) and emb <= 144 else ["hat_m"]) + cands
    if tl == "srvggnetcompact" and isinstance(net_g.get("num_conv"), int) and net_g["num_conv"] <= 8:
        cands = ["ultracompact"] + cands
    if tl == "rrdbnet" and isinstance(net_g.get("num_block"), int) and net_g["num_block"] < 23:
        cands = ["esrgan_lite"] + cands
    cands.append(tl)
    for c in cands:
        if c in archs:
            return c, ("" if c == tl else f"{t} → {c}")
    return None, f"architecture « {t} » sans équivalent dans ce moteur"


def import_yaml_config(config_path: str) -> dict:
    """Flat dict for ConfigTab._apply_flat_config. `_warnings` lists what could not be
    imported. Empty dict when the file cannot be read."""
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[Import] Erreur YAML: {e}")
        return {}
    if not isinstance(data, dict):
        return {}
    warnings = []
    datasets = data.get("datasets") or {}
    train_ds = datasets.get("train") or {}
    if not isinstance(train_ds, dict):
        train_ds = {}

    # B36: Real-ESRGAN keeps the first-order kernel settings under datasets.train and
    # the rest at the root; the app's loader reads the root. Root wins when both exist.
    norm = dict(data)
    for k in _DEG_KEYS:
        if k not in norm and k in train_ds:
            norm[k] = train_ds[k]

    # B35: the form holds one folder per path field
    def _first(v, label):
        if isinstance(v, (list, tuple)):
            if len(v) > 1:
                warnings.append(f"{label} : {len(v)} dossiers, seul le premier est importé ({v[0]})")
            return v[0] if v else ""
        return v
    for ds_key in ("train", "val"):
        ds = datasets.get(ds_key)
        if isinstance(ds, dict):
            ds = dict(ds)
            for pk in ("dataroot_gt", "dataroot_lq"):
                if pk in ds:
                    ds[pk] = _first(ds[pk], f"{ds_key}.{pk}")
            norm.setdefault("datasets", {})
            norm["datasets"] = dict(norm["datasets"]); norm["datasets"][ds_key] = ds

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as tf:
        yaml.safe_dump(norm, tf, sort_keys=False, allow_unicode=True)
        tmp = tf.name
    try:
        from src.core.config_handler import ConfigHandler
        ok, flat = ConfigHandler().load_config(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    if not ok or not isinstance(flat, dict):
        return {}
    engine = flat.get("_detected_engine") or "TraiNNer-Redux"

    # B32: an arch that exists in the engine, or none (reported) — never a made-up name
    net_g = data.get("network_g") or {}
    arch, note = _map_arch(net_g if isinstance(net_g, dict) else {}, engine)
    if arch:
        flat["arch"] = arch
        if note:
            warnings.append(f"Architecture : {note} (paramètres réseau à vérifier dans Configuration)")
    else:
        flat.pop("arch", None)
        warnings.append(f"Architecture : {note} — l'architecture actuelle du formulaire est conservée")

    # B34: crop size. lq_size / patch_size are LR; gt_size is HR (÷ scale)
    try:
        scale = int(float(flat.get("scale") or data.get("scale") or 4))
    except (TypeError, ValueError):
        scale = 4
    lq = train_ds.get("lq_size") or train_ds.get("patch_size")
    gt = train_ds.get("gt_size") or data.get("gt_size")
    if isinstance(lq, (int, float)):
        flat["patch_size"] = int(lq)
    elif isinstance(gt, (int, float)) and scale > 0:
        flat["patch_size"] = int(gt) // scale
    if "batch_size_per_gpu" in train_ds:
        flat["batch_size"] = train_ds["batch_size_per_gpu"]
    elif "batch_size" in train_ds:
        flat["batch_size"] = train_ds["batch_size"]

    sched = (data.get("train") or {}).get("scheduler") or {}
    if isinstance(sched, dict) and "gamma" in sched:
        flat["gamma"] = str(sched["gamma"])

    # paths of the original machine rarely exist here: say which ones to fix
    for k, label in (("dataroot_gt", "Train GT"), ("dataroot_lq", "Train LQ"), ("val_gt", "Val GT"),
                     ("val_lq", "Val LQ"), ("pretrain_model", "Pretrain G"), ("pretrain_model_d", "Pretrain D"),
                     ("pretrain_model_d2", "Pretrain D2")):
        v = str(flat.get(k) or "").strip()
        if v and not os.path.exists(v):
            warnings.append(f"{label} introuvable sur ce PC : {v} — à corriger dans Configuration")

    flat["_imported_from"] = os.path.basename(config_path)
    flat["_format"] = detect_config_format(config_path)
    flat["_warnings"] = warnings
    flat.setdefault("name", data.get("name", "imported_config"))
    return flat


def get_import_summary(config_path: str, result: dict = None) -> str:
    """Human-readable summary of what will be imported (pass `result` to avoid reading
    the file twice)."""
    if not os.path.exists(config_path):
        return f"Fichier non trouvé : {config_path}"
    if result is None:
        result = import_yaml_config(config_path)
    if not result:
        return "Erreur lors de l'import."
    pix = result.get("pixel_criterion") or "?"
    lines = [
        f"Format détecté : {result.get('_format', '?')}  →  moteur {result.get('_detected_engine', '?')}",
        f"Nom : {result.get('name', '?')}",
        f"Architecture : {result.get('arch', '(non importée)')}",
        f"Scale : x{result.get('scale', '?')}",
        f"Mode dataset : {result.get('dataset_mode', '?')}",
        f"Optimiseur : {result.get('optim_g', '?')} (lr={result.get('lr', '?')})",
        f"Scheduler : {result.get('scheduler', '?')} milestones={result.get('milestones', '-')} gamma={result.get('gamma', '-')}",
        f"Itérations : {result.get('total_iter', '?')}",
        f"Batch : {result.get('batch_size', '?')}",
        f"Patch (LR) : {result.get('patch_size', '?')}",
    ]
    losses = []
    for k, v in sorted(result.items()):
        if k.startswith("loss_") and str(v).lower() == "true":
            name = k[5:]
            w = result.get("weight_" + k)
            if name == "pixel":
                name = f"pixel ({pix})"
            elif name == "percep":
                name = f"perceptual ({result.get('percep_criterion', '?')})"
            losses.append(f"{name}={w}" if w is not None else name)
    if str(result.get("use_gan", "")).lower() == "true":
        losses.append(f"GAN {result.get('gan_type', '')}={result.get('dyn_gan_weight', '?')}")
    lines.append(f"Losses : {', '.join(losses) if losses else 'aucune'} (les autres sont désactivées)")
    if result.get("_warnings"):
        lines.append("")
        lines.append("⚠ Non importé / adapté :")
        lines += [f"  - {w}" for w in result["_warnings"]]
    return "\n".join(lines)
