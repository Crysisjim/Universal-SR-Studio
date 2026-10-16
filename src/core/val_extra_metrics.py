"""Validation texture metrics (USRS): one shared definition for the engine config,
the log parser, the dashboard cards and the Metrics window.

The values are computed by custom_engine/metrics/usrs_texture_metrics.py (injected into
traiNNer-redux) at each validation, on the luma of the output vs the validation GT.
"""

from __future__ import annotations



def _t(fr: str, en: str) -> str:
    try:
        from qt_ui.i18n import tr
        return tr(fr, en)
    except Exception:
        return fr

# key (= val.metrics name in the engine config), engine metric type, better, card label
# FR/EN, short graph label, color
METRICS = (
    ("usrs_tex_mae", "calculate_usrs_tex_mae", "lower", ("TEXTURE", "TEXTURE"), "Texture MAE", "#e67e22"),
    ("usrs_edge_mae", "calculate_usrs_edge_mae", "lower", ("CONTOURS", "EDGES"), "Edges MAE", "#f1c40f"),
    ("usrs_flat_mae", "calculate_usrs_flat_mae", "lower", ("GRAIN (APLATS)", "GRAIN (FLAT)"), "Flat MAE", "#95a5a6"),
    ("usrs_hf_corr", "calculate_usrs_hf_corr", "higher", ("DÉTAIL HF", "HF DETAIL"), "HF corr", "#2ecc71"),
    ("usrs_hf_ratio", "calculate_usrs_hf_ratio", "higher", ("NETTETÉ HF", "HF SHARPNESS"), "HF ratio", "#e84393"),
)
KEYS = tuple(m[0] for m in METRICS)

# Top-level config key (USS-only, stripped before the engine sees the config).
CONFIG_KEY = "usrs_texture_metrics"


def engine_metrics() -> dict:
    """val.metrics entries to add to the engine working copy."""
    return {k: {"type": t, "better": b} for k, t, b, *_ in METRICS}


def card_label(key: str) -> str:
    for k, _t_, _b, (fr, en), *_ in METRICS:
        if k == key:
            return _t(fr, en)
    return key


_SHORT = {
    "usrs_tex_mae": ("Texture", "Texture"),
    "usrs_edge_mae": ("Contours", "Edges"),
    "usrs_flat_mae": ("Grain (aplats)", "Grain (flat)"),
    "usrs_hf_corr": ("Détail HF", "HF detail"),
    "usrs_hf_ratio": ("Netteté HF", "HF sharpness"),
}


def short_label(key: str) -> str:
    fr, en = _SHORT.get(key, (key, key))
    return _t(fr, en)


def color(key: str) -> str:
    for k, *_rest in METRICS:
        if k == key:
            return _rest[-1]
    return "#888888"


def better(key: str) -> str:
    for k, _t_, b, *_ in METRICS:
        if k == key:
            return b
    return "higher"


def ratio_zone(v: float) -> tuple[str, str]:
    """(color, word) for an HF ratio value."""
    if v < 0.70:
        return "#3498db", _t("doux", "soft")
    if v <= 1.0:
        return "#2ecc71", _t("net", "sharp")
    return "#e74c3c", _t("sur-accentué", "oversharpened")


def tooltip(key: str) -> str:
    common = _t(
        "\n\nMesuré à chaque validation sur la luminance de la sortie, comparée à la GT de "
        "validation (unités 0-255). Coût : ~0,2 s par image 1080p sur le CPU.",
        "\n\nMeasured at each validation on the luma of the output, against the validation GT "
        "(0-255 units). Cost: ~0.2 s per 1080p image on the CPU.")
    tips = {
        "usrs_tex_mae": _t(
            "TEXTURE — erreur moyenne sur les zones texturées (cheveux, tissus, murs, décors "
            "peints), hors contours et hors aplats.\n"
            "Plus bas = micro-détails plus fidèles à la GT.\n"
            "Repère : monte quand le modèle lisse (perte de micro-texture) ou invente une "
            "texture qui n'est pas dans la GT (GAN trop fort).",
            "TEXTURE — mean error on textured areas (hair, fabric, walls, painted backgrounds), "
            "excluding edges and flat areas.\n"
            "Lower = micro-detail closer to the GT.\n"
            "Rises when the model smooths (micro-texture loss) or invents texture that is not "
            "in the GT (GAN too strong)."),
        "usrs_edge_mae": _t(
            "CONTOURS — erreur moyenne sur les 8 % de pixels aux contours les plus forts "
            "(lineart, bords nets).\n"
            "Plus bas = traits plus fidèles à la GT.\n"
            "Attention : si la GT de validation est douce (BD flou), un modèle qui affine les "
            "traits s'en éloigne et ce chiffre monte un peu — ce n'est pas forcément un défaut. "
            "Une forte hausse = halos, ringing, traits doublés ou déplacés.",
            "EDGES — mean error on the 8 % of pixels with the strongest edges (line art, sharp "
            "borders).\n"
            "Lower = lines closer to the GT.\n"
            "Note: if the validation GT is soft (blurry BD), a model that sharpens lines moves "
            "away from it and this rises a little — not necessarily a defect. A large rise = "
            "halos, ringing, doubled or shifted lines."),
        "usrs_flat_mae": _t(
            "GRAIN (APLATS) — erreur moyenne sur les zones plates (ciels, aplats de couleur, "
            "dégradés).\n"
            "Plus bas = aplats propres.\n"
            "Repère : une hausse en phase GAN = grain, bruit ou artefacts inventés dans les "
            "aplats (le discriminateur pousse trop) ; une hausse en phase PSNR = banding ou "
            "postérisation mal corrigés.",
            "GRAIN (FLAT) — mean error on flat areas (skies, flat color fills, gradients).\n"
            "Lower = clean flat areas.\n"
            "A rise during the GAN phase = invented grain, noise or artifacts in flat areas "
            "(the discriminator pushes too hard); a rise during the PSNR phase = banding or "
            "posterization not fixed."),
        "usrs_hf_corr": _t(
            "DÉTAIL HF — corrélation des hautes fréquences (fins détails) entre la sortie et la "
            "GT, sur les zones texturées. De -1 à 1.\n"
            "Plus haut = les détails produits sont au bon endroit et dans le bon sens (vrai "
            "détail plutôt que bruit).\n"
            "Repère : si la netteté HF monte mais que ce chiffre baisse, le modèle ajoute de la "
            "netteté « inventée ».",
            "HF DETAIL — correlation of the high frequencies (fine detail) between the output "
            "and the GT, on textured areas. From -1 to 1.\n"
            "Higher = the detail produced is in the right place and direction (real detail "
            "rather than noise).\n"
            "If HF sharpness rises while this drops, the model adds \"invented\" sharpness."),
        "usrs_hf_ratio": _t(
            "NETTETÉ HF — énergie des hautes fréquences de la sortie divisée par celle de la "
            "GT, sur les zones texturées.\n"
            "< 0,70 : plus doux que la GT (bleu) ; 0,70 à 1,0 : net (vert) ; > 1,0 : plus "
            "accentué que la GT (rouge, risque de sur-accentuation).\n"
            "Ce n'est pas « plus haut = mieux » : à lire avec le détail HF et le grain. Avec "
            "une GT de validation douce, ~0,8 est déjà une sortie nette.",
            "HF SHARPNESS — high-frequency energy of the output divided by the GT's, on "
            "textured areas.\n"
            "< 0.70: softer than the GT (blue); 0.70 to 1.0: sharp (green); > 1.0: crisper "
            "than the GT (red, oversharpening risk).\n"
            "Not \"higher = better\": read it with HF detail and grain. With a soft "
            "validation GT, ~0.8 is already a sharp output."),
    }
    return tips.get(key, "") + common
