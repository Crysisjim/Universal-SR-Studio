import customtkinter as ctk
from src.core.descriptions import ARCH_PROFILES, VRAM_FACTORS, DISC_VRAM_FACTORS


def _t(fr: str, en: str) -> str:
    try:
        from src.core.translations import get_translator
        tr = get_translator()
        if tr and getattr(tr, 'language', 'fr') == 'en':
            return en
    except Exception:
        pass
    return fr


class PerformanceBars(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(1, weight=1)

        self.labels = [
            _t("Netteté (Sharpness)", "Sharpness"), _t("Texture/Détail", "Texture/Detail"),
            _t("Fidélité (PSNR)", "Fidelity (PSNR)"),
            _t("Vitesse (Inference)", "Speed (Inference)"), _t("Légèreté VRAM", "VRAM Lightness"),
            _t("Efficacité Anime", "Anime Efficiency"), _t("Efficacité Réaliste", "Realism Efficiency")
        ]

        self.colors = ["#2ecc71", "#f1c40f", "#3498db", "#e67e22", "#9b59b6", "#ff7979", "#95a5a6"]
        self.bars = []

        for i, text in enumerate(self.labels):
            lbl = ctk.CTkLabel(self, text=text, font=("Roboto", 10, "bold"), anchor="w", width=120)
            lbl.grid(row=i, column=0, padx=5, pady=2)
            bar = ctk.CTkProgressBar(self, height=12, corner_radius=2)
            bar.grid(row=i, column=1, padx=5, pady=2, sticky="ew")
            bar.set(0)
            self.bars.append(bar)

        self.lbl_legend = ctk.CTkLabel(self, text=_t("Calcul dynamique basé sur tous les paramètres.", "Dynamic calculation based on all parameters."), font=("Arial", 9, "italic"), text_color="gray70")
        self.lbl_legend.grid(row=len(self.labels), column=0, columnspan=2, pady=(10, 5))

    def get_val(self, widgets, key, default=0, type_cast=float):
        if key not in widgets: return default
        try:
            val = widgets[key].get()
            if isinstance(val, str): val = val.lower()
            if val in ["true", "on", "1"]: return 1
            if val in ["false", "off", "0", ""]: return 0
            if "adam" in str(val) or "sgd" in str(val): return str(val)
            return type_cast(val)
        except Exception: return default

    def update_stats(self, arch, is_gan, disc_type, widgets):
        # Base Profiles
        base = list(ARCH_PROFILES.get(arch, [5, 5, 5, 5, 5, 5, 5]))
        
        bonus_qual = 0; malus_speed = 0; malus_vram = 0

        # 1. Scale Impact (Plus petit scale = Input LR plus gros pour patch fixe = VRAM++)
        scale = self.get_val(widgets, "scale", 4, int)
        if scale == 2: malus_vram += 1.5; malus_speed += 0.5
        elif scale == 3: malus_vram += 0.5
        elif scale >= 8: malus_speed += 2.0

        # 2. Window Size
        win = self.get_val(widgets, "dyn_window_size", 0)
        if win > 8: malus_vram += (win - 8) * 0.1; bonus_qual += 0.5

        # 3. Paramètres Système
        bs = self.get_val(widgets, "batch_size", 4)
        patch = self.get_val(widgets, "patch_size", 64)
        amp = self.get_val(widgets, "use_amp", 0)
        
        if bs > 4: malus_vram += (bs - 4) * 0.2
        if patch > 64: malus_vram += (patch - 64) * 0.05
        
        if amp: malus_vram -= 1.5; malus_speed -= 1.0

        # 4. Complexité Modèle
        feat = self.get_val(widgets, "dyn_num_feat", 64)
        if feat > 64: 
            complexity = (feat - 64) / 32
            malus_vram += complexity * 0.8
            malus_speed += complexity * 1.0
            bonus_qual += complexity * 0.4 

        # 5. GAN Type Impact
        if is_gan:
            base[0] += 2; base[1] += 2; base[2] -= 2
            if "patch" in disc_type: malus_vram += 0.2
            elif "unet" in disc_type: malus_vram += 0.5
            elif "meta" in disc_type: malus_vram += 1.2
            elif "ea2" in disc_type: malus_vram += 1.0
        
        # 6. Optimiseur
        opt = self.get_val(widgets, "optim_g", "AdamW", str)
        if "adan" in opt.lower(): malus_vram += 1.0; malus_speed += 0.5

        # 7. Dataset Mode
        if self.get_val(widgets, "dataset_mode", "paired", str) == "otf": malus_speed += 2.0

        # 8. Losses
        if self.get_val(widgets, "loss_fdl") and self.get_val(widgets, "fdl_model", "vgg", str) == "dinov2":
            malus_vram += 1.5; base[1] += 1
        if self.get_val(widgets, "loss_percep"): malus_vram += 0.5
        # Losses avec overhead VRAM mesuré en bench (Redux 2026-05-19)
        if self.get_val(widgets, "loss_mssim"): malus_vram += 0.2; malus_speed += 0.3
        if self.get_val(widgets, "loss_dists"): malus_vram += 0.4; malus_speed += 0.2
        if self.get_val(widgets, "loss_msswd"): malus_vram += 0.3; malus_speed += 0.4
        if self.get_val(widgets, "loss_ff"): malus_vram += 0.1
        if self.get_val(widgets, "loss_contextual"): malus_vram += 0.8; malus_speed += 1.0
        if self.get_val(widgets, "loss_wavelet"): malus_vram += 0.3; malus_speed += 0.3
        if self.get_val(widgets, "loss_ldl"): malus_vram += 0.1
        if self.get_val(widgets, "loss_consistency"): malus_vram += 0.1
        if self.get_val(widgets, "loss_edge"): malus_vram += 0.1

        final = [base[0] + bonus_qual, base[1] + bonus_qual, base[2] + (bonus_qual/2),
                 base[3] - malus_speed, base[4] - malus_vram, base[5], base[6]]

        for i, bar in enumerate(self.bars):
            val = max(1, min(10, final[i]))
            bar.set(val / 10.0)
            color = "#2ecc71" if val >= 7.5 else "#f1c40f" if val >= 4.5 else "#e74c3c"
            bar.configure(progress_color=color)