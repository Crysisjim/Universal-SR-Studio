"""
i18n.py — lightweight FR/EN helper for the Qt port.

Use `tr(fr, en)` for ad-hoc bilingual strings (no translation keys needed):

    from qt_ui.i18n import tr
    label = QLabel(tr("Modèle", "Model"))

The active language is a process-global, loaded from settings["qt_lang"] at startup
and changed via the Réglages › Langue selector. Because most existing widgets are
still built with hard-coded French, switching language currently takes effect after a
restart; newly-wrapped strings switch live. Wrapping the remaining strings is done
tab by tab — every `tr(...)` added moves more of the UI under the language switch.
"""
from __future__ import annotations

_LANG = "fr"


def set_lang(code: str) -> None:
    global _LANG
    _LANG = "en" if str(code).lower().startswith("en") else "fr"


def get_lang() -> str:
    return _LANG


def init_from_settings(settings) -> None:
    try:
        set_lang(settings.get("qt_lang", "fr") or "fr")
    except Exception:
        pass


def tr(fr: str, en: str | None = None) -> str:
    """Return `fr` or `en` depending on the active language. If `en` is omitted,
    `fr` is returned for both (placeholder until an English string is provided)."""
    if _LANG == "en" and en is not None:
        return en
    return fr


def rich_tip(s: str) -> str:
    """Format a tooltip as clean, aerated rich text for beginners:
      • description first, then a blank line, then the ↑ / ↓ effect lines;
      • ↑ (increase) green, ↓ (decrease) red, ⚠ (warning) orange;
      • understands BOTH descriptions.py markers ([+]/[-]/[!]) and inline ↑/↓ hints.
    Returns the input unchanged if there's nothing to format."""
    if not s:
        return s
    import html as _html
    import re as _re
    UP, DN, WN = "\x01", "\x02", "\x03"  # sentinels (survive html escaping)
    t = s
    t = _re.sub(r'\[\+\]\s*', UP + " ", t)
    t = _re.sub(r'\[-\]\s*', DN + " ", t)
    t = _re.sub(r'\[!\]\s*', WN + " ", t)
    t = t.replace(" · ↓", "\n" + DN + " ").replace(" ↓", "\n" + DN + " ").replace("↓", DN)
    t = t.replace(" · ↑", "\n" + UP + " ").replace(" ↑", "\n" + UP + " ").replace("↑", UP)
    if UP not in t and DN not in t and WN not in t and "\n" not in t:
        return s
    # blank line before the FIRST effect marker → separates description from effects
    t = _re.sub(r'\n*([\x01\x02\x03])', lambda m: "\n\n" + m.group(1), t, count=1)
    t = _re.sub(r'\n{3,}', "\n\n", t)
    t = _html.escape(t).replace("\n", "<br>")
    t = t.replace(UP, '<span style="color:#2ecc71; font-weight:bold;">↑</span>')
    t = t.replace(DN, '<span style="color:#e74c3c; font-weight:bold;">↓</span>')
    t = t.replace(WN, '<span style="color:#e67e22; font-weight:bold;">⚠</span>')
    return t


# Generic fallbacks (FR, EN) for families of config keys that descriptions.py doesn't
# spell out individually (weights, criterions, VGG layers, scheduler params…).
_GENERIC_TIPS = {
    "weight_loss": ("Poids de cette perte dans la perte totale.\n[+] Plus haut : cette perte influence plus le résultat.\n[-] Plus bas : influence moindre. 0 = désactivée.",
                    "Weight of this loss in the total loss.\n[+] Higher: this loss influences the result more.\n[-] Lower: less influence. 0 = disabled."),
    "criterion": ("Fonction de distance utilisée.\nl1 = écart absolu (robuste, doux) · l2/mse = écart au carré (pénalise + les gros écarts) · huber/charbonnier = compromis.",
                  "Distance function used.\nl1 = absolute error (robust, soft) · l2/mse = squared error (penalizes large errors more) · huber/charbonnier = compromise."),
    "percep_vgg": ("Poids de cette couche VGG dans la perte perceptuelle.\n[+] Couches basses (conv1-2) : textures fines. [+] Couches hautes (conv4-5) : structure globale. 0 = couche ignorée.",
                   "Weight of this VGG layer in the perceptual loss.\n[+] Low layers (conv1-2): fine textures. [+] High layers (conv4-5): global structure. 0 = layer ignored."),
}
_KEY_TIPS = {
    "t_max": ("T_max (CosineAnnealing) : nombre d'itérations sur lequel le LR décroît en cosinus jusqu'à eta_min. Souvent = total_iter.",
              "T_max (CosineAnnealing): number of iterations over which the LR decays as a cosine down to eta_min. Often = total_iter."),
    "eta_min": ("LR minimal atteint en fin de cosinus.\n[-] Plus bas : finition plus fine. Typique : 1e-7.",
                "Minimum LR reached at the end of the cosine.\n[-] Lower: finer finishing. Typical: 1e-7."),
    "milestones": ("Itérations (MultiStepLR) où le LR est multiplié par gamma. Ex : 75000, 112500.",
                   "Iterations (MultiStepLR) where the LR is multiplied by gamma. E.g. 75000, 112500."),
    "gamma": ("Facteur de réduction du LR à chaque milestone (MultiStepLR). 0.5 = LR divisé par 2.",
              "LR reduction factor at each milestone (MultiStepLR). 0.5 = LR halved."),
    "sched_free": ("Optimiseur Schedule-Free : pas besoin de scheduler, le LR s'auto-régule. (NeoSR)",
                   "Schedule-Free optimizer: no scheduler needed, the LR self-regulates. (NeoSR)"),
    "use_eco": ("ECO (Empirical Centroid-oriented Optimization) : curriculum d'entraînement. Au début, "
                "l'entrée et la cible sont mélangées avec la sortie d'un modèle « teacher » (cible plus "
                "facile), puis le mélange décroît jusqu'aux vraies paires LQ/HR.\n"
                "traiNNer-redux : teacher = modèle déjà entraîné de même architecture ; le mélange atteint "
                "les vraies données à « Fin du mélange » × total_iter.\n"
                "NeoSR : le réseau sert de teacher à lui-même (pretrain G s'il existe).\n"
                "Ne réduit PAS la VRAM : le teacher en ajoute un peu (un forward de plus).",
                "ECO (Empirical Centroid-oriented Optimization): a training curriculum. At first the "
                "input and target are blended with a 'teacher' model's output (an easier target), then "
                "the blend decreases until the real LQ/HR pairs.\n"
                "traiNNer-redux: teacher = an already trained model of the same architecture; the "
                "blend reaches the real data at 'Blend end' × total_iter.\n"
                "NeoSR: the network is its own teacher (pretrain G when set).\n"
                "Does NOT reduce VRAM: the teacher adds some (one more forward)."),
    "eco_mode_kind": ("full : mélange l'entrée LQ ET la cible HR (conseillé pour un bicubic from scratch).\n"
                      "hr_only : garde la vraie entrée LQ, ne mélange que la cible (distillation progressive, "
                      "conseillé en real-SR + GAN). traiNNer-redux uniquement.",
                      "full: blends both the LQ input AND the HR target (advised for bicubic from scratch).\n"
                      "hr_only: keeps the real LQ input, blends the target only (scheduled distillation, "
                      "advised for real-SR + GAN). traiNNer-redux only."),
    "aesop_ae_path": ("AESOP (AESOP-SR, CVPR 2025) : compare l'image produite et la cible APRÈS un "
                      "autoencodeur figé, au lieu des pixels bruts. Rôle : remplacer la loss pixel en "
                      "phase GAN (fidélité sans tirer vers le flou).\n"
                      "Demande un autoencodeur .safetensors entraîné à la même échelle, dont le décodeur "
                      "est l'architecture entraînée ici (traiNNer-redux AEModel). Les poids officiels "
                      "sont 4x photo (RRDB).",
                      "AESOP (AESOP-SR, CVPR 2025): compares the output and the target AFTER a frozen "
                      "autoencoder instead of raw pixels. Role: replace the pixel loss in the GAN phase "
                      "(fidelity without pulling towards blur).\n"
                      "Needs a .safetensors autoencoder trained at the same scale, whose decoder is the "
                      "architecture trained here (traiNNer-redux AEModel). Official weights are 4x "
                      "photo (RRDB)."),
    "eco_init": ("NeoSR : itération où le mélange ECO commence quand aucun pretrain G n'est chargé "
                 "(le réseau doit d'abord savoir produire une image pour servir de teacher). Défaut 15000.",
                 "NeoSR: iteration where the ECO blend starts when no pretrain G is loaded (the network "
                 "must first produce a usable image to act as its own teacher). Default 15000."),
    "eco_iters": ("NeoSR : nombre d'itérations sur lesquelles le mélange décroît jusqu'aux vraies données. "
                  "Défaut 80000.",
                  "NeoSR: number of iterations over which the blend decreases to the real data. "
                  "Default 80000."),
    "eco_schedule": ("NeoSR : forme de la décroissance du mélange — sigmoid (défaut, lente au début) ou linear.",
                     "NeoSR: shape of the blend decrease — sigmoid (default, slow at first) or linear."),
    "eco_end_ratio": ("Fraction de total_iter où le mélange s'arrête (vraies données ensuite). "
                      "Défaut 0.75. traiNNer-redux uniquement.",
                      "Fraction of total_iter where the blend stops (real data from there on). "
                      "Default 0.75. traiNNer-redux only."),
    "high_order_degradations_debug": ("OTF Debug : sauve les LQ dégradées à la volée dans experiments/<nom>/debug pour VÉRIFIER les dégradations.",
                                      "OTF Debug: saves the on-the-fly degraded LQ to experiments/<name>/debug to VERIFY degradations."),
}


def tip_for(key: str) -> str:
    """Rich, language-aware tooltip for a config key. Pulls from descriptions.py
    (TOOLTIPS / TOOLTIPS_EN), then falls back to per-key and family generic tooltips."""
    en = _LANG == "en"
    try:
        from src.core import descriptions as _d
        src = _d.TOOLTIPS_EN if en else _d.TOOLTIPS
        raw = src.get(key) or _d.TOOLTIPS.get(key, "")
        if raw:
            return rich_tip(raw)
    except Exception:
        pass
    if key in _KEY_TIPS:
        return rich_tip(_KEY_TIPS[key][1 if en else 0])
    # weight_loss_X → specific: name the loss + point to its own tooltip.
    if key.startswith("weight_loss_"):
        ln = key[len("weight_loss_"):]
        nm = ln.replace("_", " ").title()
        if en:
            return rich_tip(f"Weight of the “{nm}” loss inside the combined total loss.\n"
                            f"[+] Higher: this loss matters more → the model optimizes it harder.\n"
                            f"[-] Lower / 0: less / not taken into account.\n"
                            f"See the “{nm}” loss tooltip for what it actually does.")
        return rich_tip(f"Poids (importance) de la perte « {nm} » dans la perte totale combinée.\n"
                        f"[+] Augmenter : cette perte pèse plus → le modèle l'optimise davantage.\n"
                        f"[-] Diminuer / 0 : moins / pas prise en compte.\n"
                        f"Voir l'infobulle de la perte « {nm} » pour son rôle exact.")
    for prefix, pair in _GENERIC_TIPS.items():
        if key.startswith(prefix):
            return rich_tip(pair[1 if en else 0])
    return ""
