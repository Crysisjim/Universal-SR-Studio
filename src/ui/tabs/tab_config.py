import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import sys
import os
import threading
# Lazy imports for startup performance (PERF-03)
toml = None
yaml = None

def _ensure_toml():
    global toml
    if toml is None:
        import toml as _t
        toml = _t

def _ensure_yaml():
    global yaml
    if yaml is None:
        import yaml as _y
        yaml = _y
try: import tomllib
except ImportError: import toml as tomllib

from src.ui.components.tooltip import ToolTip
from src.core.ai_models_metadata import get_model_description, get_provider_default_model
from src.core.ai_cache import get_cached_response, store_response, test_api_connection, cache_stats, clear_cache
from src.core.config_templates import list_templates, get_template
from src.ui.components.performance_bars import PerformanceBars
from src.core.descriptions import (
    TOOLTIPS, get_tooltip, ARCH_FIELDS, REDUX_ARCH_FIELDS, DISC_FIELDS,
    OPTIMIZERS, VRAM_FACTORS, REDUX_VRAM_FACTORS, DISC_VRAM_FACTORS,
    ARCH_FAMILIES, NEOSR_ARCH_FAMILIES, REDUX_ARCH_FAMILIES, get_arch_families,
    NEOSR_OPTIMIZERS, REDUX_OPTIMIZERS,
    NEOSR_SCHEDULERS, REDUX_SCHEDULERS,
    NEOSR_SCALES, REDUX_SCALES,
    NEOSR_DISC_LIST, REDUX_DISC_LIST,
    NEOSR_GAN_TYPES, REDUX_GAN_TYPES,
    DISC_DISPLAY_NAMES, DISC_INTERNAL_NAMES,
)
from src.core.settings import SettingsManager
from src.core.compute_estimator import estimate_vram as _ce_estimate_vram


def _t(fr: str, en: str) -> str:
    """Pick FR or EN string based on active language."""
    try:
        from src.core.translations import get_translator
        tr = get_translator()
        if tr and getattr(tr, 'language', 'fr') == 'en':
            return en
    except Exception:
        pass
    return fr

# --- METRICS NEOSR ---
NEOSR_METRICS = [
    ("psnr", "PSNR"), 
    ("ssim", "SSIM"), 
    ("dists", "DISTS"), 
    ("topiq", "TOPIQ")
]

# Infobulles locales pour les nouvelles options
METRIC_DESCS = {
    "psnr": "Peak Signal-to-Noise Ratio.\nMesure la fidélité mathématique pure (pixel à pixel).\nStandard historique, mais ne voit pas le flou.",
    "ssim": "Structural Similarity.\nMesure la structure et la luminance.\nPlus proche de la perception humaine que le PSNR.",
    "dists": "Deep Image Structure and Texture Similarity.\nExcellent pour les textures ! Contrairement au PSNR, il préfère une image nette (avec du grain) à une image floue.",
    "topiq": "Transformer-based Image Quality.\nMétrique très récente (2023) basée sur l'IA pour juger la qualité sémantique."
}

# ConfigHandler shadow class removed (BUG-06) — real one imported from core


class _ScrollableOptionMenu(ctk.CTkOptionMenu):
    """CTkOptionMenu with mouse-wheel-scrollable dropdown (replaces inner frame with Listbox)."""

    _MAX_VISIBLE = 18
    _ROW_H = 20

    def __init__(self, master, values=None, command=None, **kw):
        self._opt_values = list(values or [])
        self._opt_command = command
        self._popup = None
        super().__init__(master, values=self._opt_values, command=self._on_select, **kw)

    def _on_select(self, value):
        if self._opt_command:
            self._opt_command(value)

    def _open_dropdown_menu(self):
        """Override CTkOptionMenu dropdown with a fast tk.Listbox popup."""
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy(); self._popup = None; return

        popup = tk.Toplevel(self)
        popup.wm_overrideredirect(True)
        popup.wm_attributes("-topmost", True)
        self._popup = popup

        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        w = self.winfo_width()
        n = len(self._opt_values)
        h = min(n * self._ROW_H + 4, self._MAX_VISIBLE * self._ROW_H + 4)
        popup.geometry(f"{w}x{h}+{x}+{y}")

        is_dark = ctk.get_appearance_mode() == "Dark"
        bg = "#2b2b2b" if is_dark else "#DBDBDB"
        fg = "#DCE4EE" if is_dark else "#1a1a1a"
        sel_bg = "#2fa572"
        sel_fg = "white"

        hover_bg = "#404040" if is_dark else "#c0c0c0"
        lb = tk.Listbox(popup, bg=bg, fg=fg, selectbackground=sel_bg,
                        selectforeground=sel_fg, borderwidth=0,
                        highlightthickness=1, highlightbackground=sel_bg,
                        relief="flat", font=("Arial", 12), activestyle="none")
        lb.pack(fill="both", expand=True)

        for v in self._opt_values:
            lb.insert("end", v)

        cur = self.get()
        if cur in self._opt_values:
            idx = self._opt_values.index(cur)
            lb.see(idx); lb.selection_set(idx)

        _hover_idx = [-1]

        def _on_motion(event):
            idx = lb.nearest(event.y)
            if idx == _hover_idx[0]:
                return
            if _hover_idx[0] >= 0:
                prev = _hover_idx[0]
                sel = lb.curselection()
                lb.itemconfig(prev, bg=sel_bg if prev in sel else bg,
                              fg="white" if prev in sel else fg)
            _hover_idx[0] = idx
            lb.itemconfig(idx, bg=hover_bg, fg=fg)

        def on_select(event):
            sel = lb.curselection()
            if sel:
                v = self._opt_values[sel[0]]
                self.set(v)
                if self._opt_command: self._opt_command(v)
                popup.destroy(); self._popup = None

        def _close_if_outside(event):
            try:
                fw = str(popup.focus_get() or "")
                if str(lb) not in fw and str(popup) not in fw:
                    popup.destroy(); self._popup = None
            except Exception:
                pass

        lb.bind("<Motion>", _on_motion)
        lb.bind("<ButtonRelease-1>", on_select)
        lb.bind("<Double-Button-1>", on_select)
        lb.bind("<Return>", on_select)
        lb.bind("<FocusOut>", lambda e: lb.after(150, _close_if_outside, e))
        popup.bind("<Escape>", lambda e: popup.destroy())
        # Clic hors de la popup → fermer
        self.winfo_toplevel().bind("<Button-1>",
            lambda e: popup.after(50, _close_if_outside, e), add="+")
        popup.after(20, lb.focus_set)

    def configure(self, **kw):
        if "values" in kw:
            self._opt_values = list(kw["values"])
        if "command" in kw:
            self._opt_command = kw.pop("command")
            kw["command"] = self._on_select
        super().configure(**kw)

    def cget(self, key):
        if key == "values": return self._opt_values
        return super().cget(key)


class ConfigTab(ctk.CTkFrame):
    def __init__(self, master, config_handler, **kwargs):
        super().__init__(master, **kwargs)
        self.config_handler = config_handler
        self.settings = SettingsManager()
        self.widgets = {} 
        self.aug_labels = {}
        self.dynamic_widgets_g = []
        self.dynamic_widgets_d = []
        self.gpu_name = "Inconnu"; self.total_vram_gb = 11.0
        self.detect_gpu_hardware()
        
        self.run_tab_ref = None 
        self._vram_timer = None  # PERF-03: debounce timer

        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)

        self.frame_nav = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.frame_nav.grid(row=0, column=0, sticky="nsew"); self.frame_nav.grid_rowconfigure(9, weight=1)
        ctk.CTkLabel(self.frame_nav, text="STUDIO CONFIG", font=("Roboto", 20, "bold")).grid(row=0, column=0, padx=20, pady=20)
        self.create_nav_btn(_t("Général", "General"), 1, "gen"); self.create_nav_btn(_t("Architecture", "Architecture"), 2, "net"); self.create_nav_btn(_t("Datasets", "Datasets"), 3, "data")
        self.create_nav_btn(_t("Entraînement", "Training"), 4, "train"); self.create_nav_btn(_t("Dégradations", "Degradations"), 5, "deg"); self.create_nav_btn(_t("Système / Avancé", "System / Advanced"), 6, "adv")
        self.create_nav_btn(_t("Vérification AI", "AI Check"), 7, "ai_check"); self.create_nav_btn(_t("Pipeline PSNR→GAN", "Pipeline PSNR→GAN"), 8, "pipeline")
        self.btn_load = ctk.CTkButton(self.frame_nav, text=_t("📂 Charger Config", "📂 Load Config"), fg_color="#444", command=self.load_action)
        self.btn_load.grid(row=10, column=0, padx=20, pady=(10, 5), sticky="ew")
        self.btn_save = ctk.CTkButton(self.frame_nav, text=_t("💾 Sauvegarder", "💾 Save"), fg_color="green", command=self.save_action)
        self.btn_save.grid(row=11, column=0, padx=20, pady=(5, 5), sticky="ew")
        self.btn_run = ctk.CTkButton(self.frame_nav, text=_t("🚀 Sauver & Tester", "🚀 Save & Test"), fg_color="#d35400", command=self.save_and_run)
        self.btn_run.grid(row=12, column=0, padx=20, pady=(5, 20), sticky="ew")

        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right_panel.grid_rowconfigure(1, weight=1); self.right_panel.grid_columnconfigure(0, weight=1)
        self.frame_vram = ctk.CTkFrame(self.right_panel, fg_color=("#E8E8E8", "#2B2B2B")); self.frame_vram.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        self.lbl_vram = ctk.CTkLabel(self.frame_vram, text=_t("VRAM : Calcul...", "VRAM : Computing..."), font=("Roboto", 12, "bold")); self.lbl_vram.pack(anchor="w", padx=15, pady=(10, 0))
        self.prog_vram = ctk.CTkProgressBar(self.frame_vram); self.prog_vram.pack(fill="x", padx=15, pady=(5, 10)); self.prog_vram.set(0)

        self.page_container = ctk.CTkFrame(self.right_panel, fg_color="transparent"); self.page_container.grid(row=1, column=0, sticky="nsew")
        self.frames = {
            "gen": self.create_page_general(), "net": self.create_page_network(), "data": self.create_page_datasets(),
            "train": self.create_page_train(), "deg": self.create_page_degradations(), "adv": self.create_page_advanced(),
            "ai_check": self.create_page_ai_check(), "pipeline": self.create_page_pipeline(),
        }
        self.show_frame("gen")
        # Init engine-dependent widgets (families, discriminators, GAN types, scales)
        self.on_engine_change("NeoSR")
        self.on_arch_change("omnisr"); self.on_disc_change("unet"); self.frame_gan_opts.pack_forget()

    # --- PAGES ---
    def create_page_general(self):
        f = ctk.CTkFrame(self.page_container, fg_color="transparent")

        self.add_header(f, _t("Paramètres Généraux", "General Settings"))
        self.row_entry(f, _t("Nom Expérience :", "Experiment Name:"), "name", "name")
        f_sub = ctk.CTkFrame(f, fg_color="transparent"); f_sub.pack(fill="x", pady=10)
        self.row_file_picker(f, _t("Dossier Expériences :", "Experiments Folder:"), "custom_exp_path", is_file=False, default=self.settings.get("custom_exp_path"))

        # MOTEUR AVEC LISTENER
        self.add_label_tip(f_sub, _t("Moteur :", "Engine:"), "engine")
        self.widgets["engine"] = ctk.CTkOptionMenu(f_sub, values=["NeoSR", "TraiNNer-Redux"], command=self.on_engine_change)
        self.widgets["engine"].pack(side="left", padx=10)

        self.add_label_tip(f_sub, _t("Scale :", "Scale:"), "scale"); self.widgets["scale"] = ctk.CTkOptionMenu(f_sub, values=NEOSR_SCALES, width=70); self.widgets["scale"].pack(side="left", padx=10); self.widgets["scale"].set("4")
        # --- SEED + CHECKBOX DÉTERMINISTE (liés) ---
        self.add_label_tip(f_sub, _t("Seed :", "Seed:"), "manual_seed")
        _seed_entry = ctk.CTkEntry(f_sub, width=60)
        _seed_entry.insert(0, "10")
        _seed_entry.pack(side="left")
        self.widgets["manual_seed"] = _seed_entry

        def _on_det_toggle():
            if self.widgets["deterministic"].get() == "true":
                _seed_entry.configure(state="normal", text_color=("gray10", "#DCE4EE"))
                if _seed_entry.get().strip() in ("0", ""):
                    _seed_entry.delete(0, "end")
                    _seed_entry.insert(0, "10")
            else:
                _seed_entry.delete(0, "end")
                _seed_entry.insert(0, "0")
                _seed_entry.configure(state="disabled", text_color="#666666")

        _det_chk = ctk.CTkCheckBox(f_sub, text=_t("Déterministe", "Deterministic"), onvalue="true", offvalue="false",
                                    command=_on_det_toggle)
        _det_chk.deselect()  # défaut : désactivé
        _det_chk.pack(side="left", padx=(8, 0))
        self.widgets["deterministic"] = _det_chk
        ToolTip(_det_chk, get_tooltip("deterministic"))

        # Engine banner — bottom, larger, centered
        self._banner_frame = ctk.CTkFrame(f, height=140, fg_color="transparent")
        self._banner_frame.pack(fill="x", pady=(30, 10), padx=10)
        self._banner_frame.pack_propagate(False)
        self._banner_label = ctk.CTkLabel(self._banner_frame, text="")
        self._banner_label.place(relx=0.5, rely=0.5, anchor="center")

        return f
    
    # --- LOGIQUE MOTEUR ---
    def _update_engine_banner(self, engine_name):
        """Load and display the engine logo banner (uniform 90px height, fills width)."""
        if not hasattr(self, '_banner_label'):
            return
        try:
            from PIL import Image
            is_redux = "Redux" in engine_name
            img_file = "traiNNer redux.png" if is_redux else "neosr.png"
            img_path = os.path.join("assets", img_file)
            if not os.path.exists(img_path):
                return
            pil_img = Image.open(img_path).convert("RGBA")
            # Scale to fit 140px height, capped at 700px width
            src_w, src_h = pil_img.size
            target_h = 140
            target_w = min(700, int(src_w * target_h / src_h))
            pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
            self._banner_ctk_img = ctk.CTkImage(
                light_image=pil_img, dark_image=pil_img,
                size=(target_w, target_h)
            )
            self._banner_label.configure(image=self._banner_ctk_img, text="")
        except Exception:
            pass

    def _disc_to_display(self, internal_list):
        """Convert internal disc names to friendly display names."""
        return [DISC_DISPLAY_NAMES.get(n, n) for n in internal_list]

    def _disc_from_display(self, display_name):
        """Convert display name back to internal name."""
        return DISC_INTERNAL_NAMES.get(display_name, display_name)

    def on_engine_change(self, choice):
        # 1. Mise à jour du ConfigHandler
        try: self.config_handler.set_engine(choice)
        except Exception: pass
        
        is_redux = "Redux" in choice
        
        # 2. Reset des familles par moteur
        families = get_arch_families("redux" if is_redux else "neosr")
        fam_list = [_t("TOUTES", "ALL")] + sorted(list(families.keys()))
        self.widgets["arch_family"].configure(values=fam_list)
        self.widgets["arch_family"].set(_t("TOUTES", "ALL"))
        self.filter_archs("ALL")

        # 3. Optimiseurs par moteur
        opt_list = REDUX_OPTIMIZERS if is_redux else NEOSR_OPTIMIZERS
        self.widgets["optim_g"].configure(values=opt_list)
        if self.widgets["optim_g"].get() not in opt_list:
            self.widgets["optim_g"].set(opt_list[0])

        # 4. Schedulers par moteur
        sched_list = REDUX_SCHEDULERS if is_redux else NEOSR_SCHEDULERS
        self.widgets["scheduler"].configure(values=sched_list)
        if self.widgets["scheduler"].get() not in sched_list:
            self.widgets["scheduler"].set(sched_list[0])

        # 5. Scales par moteur
        scale_list = REDUX_SCALES if is_redux else NEOSR_SCALES
        self.widgets["scale"].configure(values=scale_list)
        if self.widgets["scale"].get() not in scale_list:
            self.widgets["scale"].set("4")

        # 6. Discriminateurs par moteur (friendly names)
        disc_list = REDUX_DISC_LIST if is_redux else NEOSR_DISC_LIST
        display_list = self._disc_to_display(disc_list)
        self.widgets["net_d_type"].configure(values=display_list)
        current_internal = self._disc_from_display(self.widgets["net_d_type"].get())
        if current_internal not in disc_list:
            self.widgets["net_d_type"].set(display_list[0])
            self.on_disc_change(display_list[0])

        # 7. GAN loss types par moteur
        gan_types = REDUX_GAN_TYPES if is_redux else NEOSR_GAN_TYPES
        self.widgets["gan_type"].configure(values=gan_types)
        if self.widgets["gan_type"].get() not in gan_types:
            self.widgets["gan_type"].set(gan_types[0])

        # 8. Pixel criterion types par moteur
        pixel_types = ["charbonnierloss", "l1loss", "mseloss"] if is_redux else ["L1Loss", "MSELoss", "HuberLoss", "chc_loss"]
        if "pixel_criterion" in self.widgets:
            self.widgets["pixel_criterion"].configure(values=pixel_types)
            if self.widgets["pixel_criterion"].get() not in pixel_types:
                self.widgets["pixel_criterion"].set(pixel_types[0])

        # Perceptual criterion options par moteur
        # Redux: charbonnier (défaut), l1 — NeoSR: huber (défaut), l1, l2, charbonnier, chc
        percep_types = ["charbonnier", "l1", "l2", "huber"] if is_redux else ["huber", "l1", "l2", "charbonnier", "chc_loss"]
        if "percep_criterion" in self.widgets:
            self.widgets["percep_criterion"].configure(values=percep_types)
            if self.widgets["percep_criterion"].get() not in percep_types:
                self.widgets["percep_criterion"].set(percep_types[0])

        # 9. Griser les losses NeoSR-only en mode Redux
        _NEOSR_ONLY = [
            "loss_msswd", "weight_loss_msswd",
            "loss_consistency", "weight_loss_consistency",
            "consistency_blur", "consistency_cosim", "consistency_saturation", "consistency_brightness",
            "loss_edge", "weight_loss_edge", "edge_criterion", "edge_corner",
            "loss_ncc", "weight_loss_ncc",
            "loss_kl", "weight_loss_kl",
            "loss_wavelet", "weight_loss_wavelet", "wavelet_init",
            "loss_fdl", "weight_loss_fdl", "fdl_model",
            "mssim_window_size", "mssim_sigma", "mssim_k1", "mssim_k2",
        ]
        state = "disabled" if is_redux else "normal"
        for k in _NEOSR_ONLY:
            w = self.widgets.get(k)
            if w:
                try: w.configure(state=state)
                except Exception: pass

        # Losses disponibles uniquement en mode Redux
        _REDUX_ONLY = [
            "loss_hsluv", "weight_loss_hsluv", "hsluv_hue_weight", "hsluv_sat_weight", "hsluv_lum_weight",
            "loss_cosim", "weight_loss_cosim", "cosim_lambda",
            "loss_color", "weight_loss_color", "color_criterion",
            "loss_gv",    "weight_loss_gv",    "gv_patch_size",    "gv_criterion",
            "loss_luma",  "weight_loss_luma",  "luma_criterion",
            "loss_contextual", "weight_loss_contextual", "ctx_distance_type", "ctx_band_width",
        ]
        state_redux = "normal" if is_redux else "disabled"
        for k in _REDUX_ONLY:
            w = self.widgets.get(k)
            if w:
                try: w.configure(state=state_redux)
                except Exception: pass

        # Update engine banner
        self._update_engine_banner(choice)

    def create_page_network(self):
        f = ctk.CTkFrame(self.page_container, fg_color="transparent"); self.add_header(f, _t("Générateur (Modèle)", "Generator (Model)"))
        f_top = ctk.CTkFrame(f, fg_color="transparent"); f_top.pack(fill="x", pady=5)
        f_left = ctk.CTkFrame(f_top, fg_color="transparent"); f_left.pack(side="left", fill="y")
        
        # AJOUT MENU FAMILLE
        ctk.CTkLabel(f_left, text=_t("Famille :", "Family:")).pack(anchor="w")
        self.widgets["arch_family"] = ctk.CTkOptionMenu(f_left, values=[_t("TOUTES", "ALL")], command=self.filter_archs, width=200)
        self.widgets["arch_family"].pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(f_left, text=_t("Architecture :", "Architecture:")).pack(anchor="w")
        # Liste par défaut (NeoSR)
        self.widgets["arch"] = _ScrollableOptionMenu(f_left, values=sorted(list(ARCH_FIELDS.keys())), command=self.on_arch_change, width=200)
        self.widgets["arch"].pack(anchor="w", pady=(5,0))
        
        f_right = ctk.CTkFrame(f_top, fg_color=("#DEDEDE", "#222"), corner_radius=6); f_right.pack(side="right", padx=10, fill="both", expand=True)
        self.perf_bars = PerformanceBars(f_right, fg_color="transparent"); self.perf_bars.pack(padx=10, pady=5, fill="x", expand=True)
        self.frame_dynamic_g = ctk.CTkFrame(f, border_width=1, border_color="#3B8ED0"); self.frame_dynamic_g.pack(fill="x", pady=10, ipady=5)
        self.add_header(f, _t("Discriminateur (GAN)", "Discriminator (GAN)"))
        self.widgets["use_gan"] = ctk.CTkCheckBox(f, text=_t("Activer GAN", "Enable GAN"), onvalue="true", offvalue="false", command=self.toggle_gan_options); self.widgets["use_gan"].pack(anchor="w", pady=5); ToolTip(self.widgets["use_gan"], get_tooltip("use_gan", ""))
        self.frame_gan_opts = ctk.CTkFrame(f, fg_color="transparent")
        # ── Ligne principale : Type | Mode | Real | Fake | [bloc Loss GAN vertical] ──
        _f_row = ctk.CTkFrame(self.frame_gan_opts, fg_color="transparent"); _f_row.pack(fill="x", pady=(6, 2))
        self.add_label_tip(_f_row, _t("Type :", "Type:"), "net_d_type")
        self.widgets["net_d_type"] = ctk.CTkOptionMenu(_f_row, values=self._disc_to_display(NEOSR_DISC_LIST), command=self.on_disc_change, width=110)
        self.widgets["net_d_type"].pack(side="left", padx=(2, 12))
        self.add_label_tip(_f_row, _t("Mode :", "Mode:"), "gan_type")
        self.widgets["gan_type"] = ctk.CTkOptionMenu(_f_row, values=NEOSR_GAN_TYPES, width=70)
        self.widgets["gan_type"].pack(side="left", padx=(2, 12))
        self.add_label_tip(_f_row, _t("Real Label :", "Real Label:"), "real_label_val")
        self.widgets["real_label_val"] = ctk.CTkEntry(_f_row, width=50); self.widgets["real_label_val"].insert(0, "1.0"); self.widgets["real_label_val"].pack(side="left", padx=(2, 12))
        self.add_label_tip(_f_row, _t("Fake Label :", "Fake Label:"), "fake_label_val")
        self.widgets["fake_label_val"] = ctk.CTkEntry(_f_row, width=50); self.widgets["fake_label_val"].insert(0, "0.0"); self.widgets["fake_label_val"].pack(side="left", padx=(2, 16))
        # ── Bloc Loss Weight GAN : label+emoji+intensité sur ligne 1, slider+entry sur ligne 2 ──
        _gan_emoji_map = [
            (0.001, "🔵", "#5599ff"), (0.005, "🔵", "#6699ff"),
            (0.01,  "🟢", "#44cc88"), (0.05,  "🟢", "#33dd66"),
            (0.1,   "🟡", "#ffdd44"), (0.2,   "🟠", "#ffaa22"),
            (0.5,   "🔴", "#ff5533"), (1.0,   "💥", "#ff2222"),
        ]
        # Bloc Loss GAN : label + emoji | sous-cadre slider centré | entry
        _gan_blk = ctk.CTkFrame(_f_row, fg_color="transparent"); _gan_blk.pack(side="left", padx=(0, 8))
        _lbl_gan = ctk.CTkLabel(_gan_blk, text=_t("Loss Weight GAN :", "Loss Weight GAN:")); _lbl_gan.pack(side="left", padx=5)
        ToolTip(_lbl_gan, get_tooltip("gan_loss_weight", ""))
        self._gan_w_emoji = ctk.CTkLabel(_gan_blk, text="🟢", width=28, font=("Segoe UI Emoji", 13))
        self._gan_w_emoji.pack(side="left", padx=(4, 6))
        # Sous-cadre : "−intensité+" (width=218) centré sur slider, pady symétrique centre le slider
        _slider_sub = ctk.CTkFrame(_gan_blk, fg_color="transparent"); _slider_sub.pack(side="left")
        ctk.CTkLabel(_slider_sub, text=_t("−  intensité  +", "−  intensity  +"), text_color="#aaa",
                     font=("Arial", 9, "italic"), height=12, width=218).pack(anchor="center", pady=(0, 3))
        self._gan_w_sl = ctk.CTkSlider(_slider_sub, from_=0, to=1, number_of_steps=100, width=218)
        self._gan_w_sl.pack(pady=(0, 12))
        self.widgets["dyn_gan_weight"] = ctk.CTkEntry(_gan_blk, width=80)
        self.widgets["dyn_gan_weight"].insert(0, "0.05")
        self.widgets["dyn_gan_weight"].bind("<KeyRelease>", lambda e: self.refresh_ui_stats())
        self.widgets["dyn_gan_weight"].pack(side="left", padx=(6, 0))
        ToolTip(self.widgets["dyn_gan_weight"], _t("Poids de la loss GAN (adversarielle)\n\nValeurs guides :\n  0.001  quasi-PSNR  (GAN quasi absent)\n  0.01   GAN doux    (idéal pour démarrer)\n  0.05   défaut NeoSR (équilibre pixel/réalisme)\n  0.1    modéré      (textures plus vives)\n  0.2    agressif    (risque d'artefacts)\n  0.5    fort        (GAN dominant)\n  1.0    instable    (déconseillé)",
                                                    "GAN loss weight (adversarial)\n\nGuide values:\n  0.001  near-PSNR  (GAN almost absent)\n  0.01   soft GAN   (ideal to start)\n  0.05   NeoSR default (pixel/realism balance)\n  0.1    moderate   (more vivid textures)\n  0.2    aggressive (risk of artifacts)\n  0.5    strong     (GAN dominant)\n  1.0    unstable   (not recommended)"))
        self._init_log_slider(self._gan_w_sl, self.widgets["dyn_gan_weight"], self._gan_w_emoji,
                              lo=0.001, hi=1.0, emoji_map=_gan_emoji_map)
        self.frame_dynamic_d = ctk.CTkFrame(self.frame_gan_opts, border_width=1, border_color="#e74c3c"); self.frame_dynamic_d.pack(fill="x", pady=(15, 4), ipady=5)
        return f

    def create_page_datasets(self):
        f = ctk.CTkFrame(self.page_container, fg_color="transparent"); self.add_header(f, _t("Datasets & Chemins", "Datasets & Paths"))
        f_mode = ctk.CTkFrame(f, fg_color="transparent"); f_mode.pack(fill="x", pady=5)
        ctk.CTkLabel(f_mode, text=_t("Mode de Dataset :", "Dataset Mode:"), width=120, anchor="w").pack(side="left")
        self.widgets["dataset_mode"] = ctk.CTkOptionMenu(f_mode, values=["otf", "paired"], width=180); self.widgets["dataset_mode"].pack(side="left"); self.widgets["dataset_mode"].set("otf"); ToolTip(self.widgets["dataset_mode"], get_tooltip("dataset_mode"))
        self.row_file_picker(f, _t("Train HQ (GT) :", "Train HQ (GT):"), "dataroot_gt", default=self.settings.get("ds_train_gt"))
        self.row_file_picker(f, _t("Val HQ (GT) :", "Val HQ (GT):"), "val_gt", default=self.settings.get("ds_val_gt"), tip_key="val_freq")
        self.row_file_picker(f, _t("Val LQ (Optionnel) :", "Val LQ (Optional):"), "val_lq", default=self.settings.get("ds_val_lq"))
        self.row_file_picker(f, _t("Train LQ (Optionnel) :", "Train LQ (Optional):"), "dataroot_lq", default=self.settings.get("ds_train_lq"))

        row_sh = ctk.CTkFrame(f, fg_color="transparent"); row_sh.pack(fill="x", pady=2)
        self.widgets["use_shuffle"] = ctk.CTkCheckBox(row_sh, text=_t("Shuffle (Mélanger)", "Shuffle (Randomize)"), onvalue="true", offvalue="false"); self.widgets["use_shuffle"].pack(side="left", padx=5); self.widgets["use_shuffle"].select()
        ToolTip(self.widgets["use_shuffle"], _t("Mélanger les données à chaque epoch.\nIndispensable pour éviter que le modèle mémorise l'ordre des images.",
                                                  "Shuffle data at each epoch.\nEssential to prevent the model from memorizing the order of images."))
        self.add_header(f, _t("Augmentations (Train)", "Augmentations (Train)"))
        # Spatial augmentations (hflip, rot90)
        row_spatial = ctk.CTkFrame(f, fg_color="transparent"); row_spatial.pack(fill="x", pady=2)
        self.widgets["use_hflip"] = ctk.CTkCheckBox(row_spatial, text=_t("HFlip (Miroir H)", "HFlip (H Mirror)"), onvalue="true", offvalue="false"); self.widgets["use_hflip"].pack(side="left", padx=5); self.widgets["use_hflip"].select()
        ToolTip(self.widgets["use_hflip"], _t("Retournement horizontal aléatoire.\n        [+] Double la variété du dataset gratuitement (aucun coût VRAM).\n        [+] Toujours recommandé sauf si votre dataset a une orientation fixe (ex: texte).",
                                               "Random horizontal flip.\n        [+] Doubles dataset variety for free (no VRAM cost).\n        [+] Always recommended unless your dataset has a fixed orientation (e.g. text)."))
        self.widgets["use_rot"] = ctk.CTkCheckBox(row_spatial, text=_t("Rot90 (Rotations)", "Rot90 (Rotations)"), onvalue="true", offvalue="false"); self.widgets["use_rot"].pack(side="left", padx=15); self.widgets["use_rot"].select()
        ToolTip(self.widgets["use_rot"], _t("Rotations aléatoires de 90°, 180°, 270°.\n        [+] Quadruple la variété du dataset.\n        [+] Recommandé pour textures et photos naturelles.\n        [-] À désactiver si l'orientation compte (visages, texte).",
                                             "Random rotations of 90°, 180°, 270°.\n        [+] Quadruples dataset variety.\n        [+] Recommended for textures and natural photos.\n        [-] Disable if orientation matters (faces, text)."))
        # Mix augmentations (MoA)
        for aug, lbl, default_prob in [("mixup", "MixUp", 0.15), ("cutmix", "CutMix", 0.15), ("resizemix", "ResizeMix", 0.15), ("cutblur", "CutBlur", 0.15)]:
            row = ctk.CTkFrame(f, fg_color="transparent"); row.pack(fill="x", pady=2)
            chk = ctk.CTkCheckBox(row, text=lbl, width=90, onvalue="true", offvalue="false"); chk.pack(side="left")
            self.widgets[f"aug_{aug}"] = chk; ToolTip(chk, get_tooltip(f"aug_{aug}"))
            sl = ctk.CTkSlider(row, from_=0, to=1, number_of_steps=20, width=120); sl.pack(side="left", padx=10); sl.set(default_prob)
            self.widgets[f"prob_aug_{aug}"] = sl
            lbl_prob = ctk.CTkLabel(row, text=f"{default_prob:.2f}", width=40); lbl_prob.pack(side="left")
            self.aug_labels[f"prob_aug_{aug}"] = lbl_prob
            sl.configure(command=lambda v, l=lbl_prob: l.configure(text=f"{v:.2f}"))
        self.add_header(f, _t("Reprise / Pretrain", "Resume / Pretrain"))
        self.row_file_picker(f, _t("Resume State (.state) :", "Resume State (.state):"), "resume_state", is_file=True)
        self.row_file_picker(f, _t("Pretrain Model (.pth) :", "Pretrain Model (.pth):"), "pretrain_model", is_file=True)
        return f

    def _init_log_slider(self, slider, entry, emoji_lbl, lo=1e-6, hi=1e-3, emoji_map=None):
        """Wire up a logarithmic slider with color emoji intensity label.
        lo/hi: value range.  emoji_map: list of (value, emoji, color) tuples."""
        import math
        if emoji_map is None:
            emoji_map = [
                (1e-6, "🔵", "#5599ff"), (1e-5, "🔵", "#6699ff"),
                (5e-5, "🟢", "#33dd66"), (1e-4, "🟢", "#44cc88"),
                (2e-4, "🟡", "#ffdd44"), (5e-4, "🟠", "#ffaa22"),
                (8e-4, "🔴", "#ff5533"), (1e-3, "💥", "#ff2222"),
            ]
        _log_lo = math.log10(lo)
        _log_hi = math.log10(hi)
        def _to_sv(v):
            try: return max(0.0, min(1.0, (math.log10(float(v)) - _log_lo) / (_log_hi - _log_lo)))
            except Exception: return 0.5
        def _to_val(sv): return 10 ** (_log_lo + sv * (_log_hi - _log_lo))
        def _emoji_col(v):
            try:
                lv = math.log10(max(float(v), 1e-15))
                c = min(emoji_map, key=lambda x: abs(lv - math.log10(x[0])))
                return c[1], c[2]
            except Exception: return "🟢", "#88ff88"
        def _fmt(v): return f"{v:.2e}" if v < 0.01 else f"{v:.6f}"
        def slider_cb(sv):
            v = _to_val(sv); entry.delete(0, "end"); entry.insert(0, _fmt(v))
            em, col = _emoji_col(v); emoji_lbl.configure(text=em, text_color=col)
        def entry_cb(event=None):
            try:
                v = float(entry.get()); slider.set(_to_sv(v))
                em, col = _emoji_col(v); emoji_lbl.configure(text=em, text_color=col)
            except Exception: pass
        slider.configure(command=slider_cb)
        entry.bind("<Return>", entry_cb); entry.bind("<FocusOut>", entry_cb)
        try:
            v0 = float(entry.get()); slider.set(_to_sv(v0))
            em, col = _emoji_col(v0); emoji_lbl.configure(text=em, text_color=col)
        except Exception: slider.set(_to_sv(lo * 10))

    # Keep legacy alias used by any external callers
    def _init_lr_slider(self, slider, entry, emoji_lbl):
        self._init_log_slider(slider, entry, emoji_lbl)

    def create_page_train(self):
        f = ctk.CTkScrollableFrame(self.page_container, fg_color="transparent"); self.add_header(f, _t("Hyperparamètres", "Hyperparameters"))
        f_grid = ctk.CTkFrame(f, fg_color="transparent"); f_grid.pack(fill="x")
        self.add_param_grid(f_grid, "Batch Size", "4", "batch_size", 0, 0, "Nombre d'images par calcul GPU.\nAugmenter consomme plus de VRAM mais accélère le training.")
        self.add_param_grid(f_grid, "Accumulate", "1", "accumulate", 0, 1, "Multiplicateur de Batch Virtuel.\nPermet de simuler un gros batch (ex: 4x4=16) sans exploser la VRAM.")
        self.add_param_grid(f_grid, "Patch Size", "64", "patch_size", 0, 2, "Taille des carrés d'images découpés (ex: 64x64).\nPlus grand = meilleure cohérence globale, mais VRAM x4.")
        self.add_param_grid(f_grid, "Total Iter", "150000", "total_iter", 1, 0, "Nombre total de pas d'entraînement.\nPour un anime complet, 150k - 300k est recommandé.")
        self.add_param_grid(f_grid, "Warmup Iter", "-1", "warmup_iter", 1, 1, "warmup_iter")
        self.add_param_grid(f_grid, "Warmup Steps", "-1", "warmup_steps", 1, 2, "warmup_steps")
        # ---- Ligne: Optimiseur G + Scheduler + Sched. Free sur une seule ligne ----
        f_opt = ctk.CTkFrame(f, fg_color="transparent"); f_opt.pack(fill="x", pady=(5, 3))
        self.add_label_tip(f_opt, _t("Optimiseur G :", "Optimizer G:"), "optim_g")
        self.widgets["optim_g"] = ctk.CTkOptionMenu(f_opt, values=NEOSR_OPTIMIZERS, width=120, command=lambda x: (self._on_optimizer_change(x), self.refresh_ui_stats())); self.widgets["optim_g"].pack(side="left", padx=5)
        self.add_label_tip(f_opt, _t("Scheduler :", "Scheduler:"), "scheduler")
        self.widgets["scheduler"] = ctk.CTkOptionMenu(f_opt, values=NEOSR_SCHEDULERS, width=140); self.widgets["scheduler"].pack(side="left", padx=5)
        self.widgets["schedule_free"] = ctk.CTkCheckBox(f_opt, text=_t("Sched. Free", "Sched. Free"), width=90); self.widgets["schedule_free"].pack(side="left", padx=5); ToolTip(self.widgets["schedule_free"], get_tooltip("schedule_free"))

        # ---- LR Générateur | LR Discriminateur côte à côte ----
        _f_lr_cols = ctk.CTkFrame(f, fg_color="transparent"); _f_lr_cols.pack(fill="x", pady=(4, 4))

        # LR Générateur — label inline à gauche (comme GAN)
        _lr_g_fr = ctk.CTkFrame(_f_lr_cols, fg_color="transparent"); _lr_g_fr.pack(side="left", fill="x", expand=True, padx=(0, 6))
        _lbl_lr_g = ctk.CTkLabel(_lr_g_fr, text=_t("LR Générateur :", "LR Generator:")); _lbl_lr_g.pack(side="left", padx=(0, 5))
        ToolTip(_lbl_lr_g, get_tooltip("lr", ""))
        self._lr_g_emoji = ctk.CTkLabel(_lr_g_fr, text="🟢", width=30, font=("Segoe UI Emoji", 13))
        self._lr_g_emoji.pack(side="left", padx=(3, 0))
        _lr_g_sub = ctk.CTkFrame(_lr_g_fr, fg_color="transparent"); _lr_g_sub.pack(side="left", fill="x", expand=True, padx=3)
        ctk.CTkLabel(_lr_g_sub, text=_t("−  intensité  +", "−  intensity  +"), text_color="#aaa",
                     font=("Arial", 9, "italic"), height=12).pack(fill="x", pady=(0, 3))
        self._lr_g_sl = ctk.CTkSlider(_lr_g_sub, from_=0, to=1, number_of_steps=100)
        self._lr_g_sl.pack(fill="x", pady=(0, 12))
        self.widgets["_sl_lr"] = self._lr_g_sl
        self.widgets["lr"] = ctk.CTkEntry(_lr_g_fr, width=72); self.widgets["lr"].insert(0, "5e-5"); self.widgets["lr"].pack(side="left", padx=3)
        ToolTip(self.widgets["lr"], _t("LR Générateur — vitesse d'apprentissage\n\nValeurs guides :\n  1e-6  ultra lent   (micro-ajustement final)\n  1e-5  très lent    (fine-tuning existant)\n  5e-5  lent         (fine-tuning GAN)\n  1e-4  modéré       (stable, bon pour GAN)\n  2e-4  standard     (Adam/AdamW recommandé)\n  5e-4  rapide       (départ PSNR)\n  1e-3  très rapide  (instable)\n\nPSNR : 2e-4 à 5e-4\nGAN  : 5e-5 à 2e-4\nFine-tune : 1e-5 à 5e-5",
                                        "Generator LR — learning rate\n\nGuide values:\n  1e-6  ultra slow   (final micro-adjustment)\n  1e-5  very slow    (fine-tuning existing)\n  5e-5  slow         (GAN fine-tuning)\n  1e-4  moderate     (stable, good for GAN)\n  2e-4  standard     (Adam/AdamW recommended)\n  5e-4  fast         (PSNR start)\n  1e-3  very fast    (unstable)\n\nPSNR: 2e-4 to 5e-4\nGAN:  5e-5 to 2e-4\nFine-tune: 1e-5 to 5e-5"))
        self._init_log_slider(self._lr_g_sl, self.widgets["lr"], self._lr_g_emoji)

        # LR Discriminateur — label inline à gauche (comme GAN)
        _lr_d_fr = ctk.CTkFrame(_f_lr_cols, fg_color="transparent"); _lr_d_fr.pack(side="left", fill="x", expand=True, padx=(6, 0))
        _lbl_lr_d = ctk.CTkLabel(_lr_d_fr, text=_t("LR Discriminateur :", "LR Discriminator:")); _lbl_lr_d.pack(side="left", padx=(0, 5))
        ToolTip(_lbl_lr_d, get_tooltip("lr_d", ""))
        self._lr_d_emoji = ctk.CTkLabel(_lr_d_fr, text="🟢", width=30, font=("Segoe UI Emoji", 13))
        self._lr_d_emoji.pack(side="left", padx=(3, 0))
        _lr_d_sub = ctk.CTkFrame(_lr_d_fr, fg_color="transparent"); _lr_d_sub.pack(side="left", fill="x", expand=True, padx=3)
        ctk.CTkLabel(_lr_d_sub, text=_t("−  intensité  +", "−  intensity  +"), text_color="#aaa",
                     font=("Arial", 9, "italic"), height=12).pack(fill="x", pady=(0, 3))
        self._lr_d_sl = ctk.CTkSlider(_lr_d_sub, from_=0, to=1, number_of_steps=100)
        self._lr_d_sl.pack(fill="x", pady=(0, 12))
        self.widgets["_sl_lr_d"] = self._lr_d_sl
        self.widgets["lr_d"] = ctk.CTkEntry(_lr_d_fr, width=72); self.widgets["lr_d"].insert(0, "5e-5"); self.widgets["lr_d"].pack(side="left", padx=3)
        ToolTip(self.widgets["lr_d"], _t("LR Discriminateur — vitesse apprentissage du D\n\nValeurs guides :\n  1e-5  très lent    (D freine au max)\n  5e-5  conservateur (D stable)\n  1e-4  standard     (équilibre G/D)\n  2e-4  agressif     (D apprend vite)\n\nGénéralement = ou < LR du générateur.\nD domine (artefacts) : baisser ce LR\nG domine (flou) : augmenter ce LR",
                                          "Discriminator LR — D learning rate\n\nGuide values:\n  1e-5  very slow    (D brakes max)\n  5e-5  conservative (D stable)\n  1e-4  standard     (G/D balance)\n  2e-4  aggressive   (D learns fast)\n\nGenerally = or < Generator LR.\nD dominates (artifacts): lower this LR\nG dominates (blur): raise this LR"))
        self._init_log_slider(self._lr_d_sl, self.widgets["lr_d"], self._lr_d_emoji)
        
        # Paramètres Spécifiques
        self.add_header(f, _t("Paramètres Scheduler", "Scheduler Parameters"))
        f_sched = ctk.CTkFrame(f, fg_color="transparent"); f_sched.pack(fill="x")
        
        # MultiStep Params
        self.add_param_grid(f_sched, "Milestones (Multi)", "75000, 112500", "milestones", 0, 0, "Liste d'itérations où le Learning Rate diminue.\nEx: [75000, 112500] = Baisse à 50% et 75% du training.")
        self.add_param_grid(f_sched, "Gamma (Multi)", "0.5", "gamma", 0, 1, "Facteur de réduction du LR.\n0.5 = Diviser par 2 à chaque Milestone.")
        
        # Cosine Params (T_max et eta_min)
        self.add_param_grid(f_sched, "T_max (Cosine)", "300000", "t_max", 1, 0, "Durée d'un cycle complet en itérations.\nDoit généralement être égal à 'Total Iter'.")
        self.add_param_grid(f_sched, "Eta Min (Cosine)", "1e-7", "eta_min", 1, 1, "Learning Rate minimum atteint à la fin du cycle.\nÉvite que l'entraînement ne s'arrête totalement.")

        self.add_header(f, _t("Log & Validation", "Log & Validation"))
        f_val = ctk.CTkFrame(f, fg_color="transparent"); f_val.pack(fill="x", pady=10)
        self.add_param_grid(f_val, "Val Freq", "5000", "val_freq", 0, 0, "Fréquence de calcul des scores (PSNR/SSIM) sur le set de validation.")
        self.add_param_grid(f_val, "Save Freq", "5000", "save_freq", 0, 1, "Fréquence de sauvegarde du modèle (.pth).\nImportant en cas de crash.")
        self.add_param_grid(f_val, "Print Freq", "100", "print_freq", 0, 2, "Fréquence d'affichage des logs dans la console.\nEvite de spammer l'interface.") 
        
        self.add_param_grid(f_val, "Save Img", "true", "save_img", 1, 0, "Sauvegarder les images de validation générées pour vérifier la qualité visuelle.")
        self.add_param_grid(f_val, "Tile (Val)", "200", "tile", 1, 1, "Taille de découpe pour la validation (économise VRAM).\nSi ça crash en OOM pendant la validation, baissez cette valeur.")
        # --- AJOUT TILE PAD ---
        self.add_param_grid(f_val, "Tile Pad", "32", "tile_pad", 1, 2, "Padding (overlap) pour la validation.\nEssentiel pour RealPLKSR/SwinIR pour éviter les effets de grille (seams).\nValeurs : 32 ou 64.")
        
        # --- METRIQUES STRICTES NEOSR ---
        f_metrics = ctk.CTkFrame(f, fg_color="transparent"); f_metrics.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(f_metrics, text=_t("Métriques :", "Metrics:"), font=("Roboto", 10, "bold"), text_color=("gray30", "#AAA")).pack(anchor="w")
        for m, txt in NEOSR_METRICS:
            chk = ctk.CTkCheckBox(f_metrics, text=txt, onvalue="true", offvalue="false", width=80); chk.pack(side="left", padx=5)
            self.widgets[f"metric_{m}"] = chk; ToolTip(chk, METRIC_DESCS.get(m, get_tooltip(f"metric_{m}")))
            if m in ["psnr", "ssim"]: chk.select()
        return f

    def create_page_degradations(self):
        f = ctk.CTkFrame(self.page_container, fg_color="transparent")
        f.grid_columnconfigure(0, weight=1)

        # ─── Preset row ───
        f_pre = ctk.CTkFrame(f, fg_color="transparent"); f_pre.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        ctk.CTkLabel(f_pre, text="Preset :", font=("Roboto", 12, "bold")).pack(side="left", padx=10)
        self.deg_preset_menu = ctk.CTkOptionMenu(f_pre, values=["Light", "Medium", "Heavy", "Overkill"], command=self.apply_deg_preset)
        self.deg_preset_menu.pack(side="left")
        self.deg_preset_menu.set("Medium")
        ToolTip(self.deg_preset_menu, get_tooltip("deg_level"))
        # Checkbox: include custom degradations when applying preset
        self._chk_deg_custom = ctk.CTkCheckBox(f_pre, text="Deg. Custom", font=("Roboto", 11),
                                                command=self._on_deg_custom_toggle)
        self._chk_deg_custom.pack(side="left", padx=(12, 0))
        self._chk_deg_custom.select()
        ToolTip(self._chk_deg_custom, _t("Si coché : le preset applique aussi les dégradations custom.\nDécocher = remet à zéro toutes les dégradations custom immédiatement.",
                                          "If checked: the preset also applies custom degradations.\nUncheck = resets all custom degradations immediately."))
        # Checkbox: include VHS degradations
        self._chk_deg_vhs = ctk.CTkCheckBox(f_pre, text="VHS/Analog", font=("Roboto", 11),
                                             command=self._on_deg_vhs_toggle)
        self._chk_deg_vhs.pack(side="left", padx=(8, 0))
        ToolTip(self._chk_deg_vhs, _t("Si coché : le preset applique les dégradations VHS/analogiques.\nDécocher = remet VHS à zéro immédiatement.",
                                        "If checked: the preset applies VHS/analog degradations.\nUncheck = resets VHS to zero immediately."))
        self.lbl_deg_info = ctk.CTkLabel(f_pre, text="", text_color="gray"); self.lbl_deg_info.pack(side="left", padx=10)

        def add_control(parent, label, key, default, type="entry", slider_max=1.0, step=0.01, pady=2):
            row = ctk.CTkFrame(parent, fg_color="transparent"); row.pack(fill="x", pady=pady)
            chk = ctk.CTkCheckBox(row, text=label, width=20, font=("Arial", 11)); chk.pack(side="left", padx=(5, 0)); chk.select(); ToolTip(chk, get_tooltip(key, ""))

            if type == "range":
                raw = str(default).strip().strip("[]")
                parts = [p.strip() for p in raw.split(",")]
                min_val = parts[0] if parts else "0"
                max_val = parts[1] if len(parts) > 1 else "1"
                e_max = ctk.CTkEntry(row, width=48); e_max.insert(0, max_val); e_max.pack(side="right", padx=(0, 5))
                ctk.CTkLabel(row, text="→", width=10, text_color="#888").pack(side="right")
                e_min = ctk.CTkEntry(row, width=48); e_min.insert(0, min_val); e_min.pack(side="right", padx=(0, 2))

                class _RangeProxy:
                    _range_proxy = True
                    def get(self_): return f"{e_min.get()},{e_max.get()}"
                    def delete(self_, a, b): e_min.delete(0, "end"); e_max.delete(0, "end")
                    def insert(self_, pos, val):
                        raw2 = str(val).strip().strip("[]")
                        pts = [p.strip() for p in raw2.split(",")]
                        e_min.delete(0, "end"); e_min.insert(0, pts[0] if pts else "0")
                        e_max.delete(0, "end"); e_max.insert(0, pts[1] if len(pts) > 1 else "0")
                    def configure(self_, **kw): e_min.configure(**kw); e_max.configure(**kw)

                self.widgets[key] = _RangeProxy()
                def toggle():
                    st = "normal" if chk.get() else "disabled"
                    e_min.configure(state=st); e_max.configure(state=st)
                chk.configure(command=toggle)
                return

            e = ctk.CTkEntry(row, width=65); e.insert(0, str(default)); e.pack(side="right", padx=5); self.widgets[key] = e
            if type == "slider":
                sl = ctk.CTkSlider(row, from_=0, to=slider_max, number_of_steps=(slider_max/step if step else 100)); sl.pack(side="right", padx=5, fill="x", expand=True)
                try: sl.set(float(default))
                except Exception: sl.set(0)
                self.widgets[f"_sl_{key}"] = sl  # Store slider so load_action can update position
                def slider_cb(v): val = round(v, 2) if step < 1 else int(v); e.delete(0, "end"); e.insert(0, str(val))
                def entry_cb(event):
                    try: sl.set(float(e.get()))
                    except Exception: pass
                sl.configure(command=slider_cb); e.bind("<Return>", entry_cb); e.bind("<FocusOut>", entry_cb)
                def toggle(): st = "normal" if chk.get() else "disabled"; e.configure(state=st); sl.configure(state=st)
                chk.configure(command=toggle)
            else:
                def toggle(): e.configure(state="normal" if chk.get() else "disabled")
                chk.configure(command=toggle)

        # ─── Tab view: Stage 1 | Stage 2 | Custom ───
        tabview = ctk.CTkTabview(f, fg_color=("#E8E8E8", "#1e1e2e"), corner_radius=6)
        tabview.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)

        t1 = tabview.add("  Stage 1  ")
        t2 = tabview.add("  Stage 2  ")
        tc = tabview.add("  Custom   ")
        tc2 = tabview.add(" Custom 2  ")
        tc3 = tabview.add(" Custom 3  ")
        tc4 = tabview.add(" Custom 4  ")

        # --- Stage 1 : 2 colonnes côte à côte ---
        t1.columnconfigure(0, weight=1)
        t1.columnconfigure(1, weight=1)
        t1.rowconfigure(0, weight=1)
        s1_left  = ctk.CTkScrollableFrame(t1, fg_color="transparent")
        s1_left.grid(row=0, column=0, sticky="nsew", padx=(2, 1), pady=2)
        s1_right = ctk.CTkScrollableFrame(t1, fg_color="transparent")
        s1_right.grid(row=0, column=1, sticky="nsew", padx=(1, 2), pady=2)

        self.add_header(s1_left, "Stage 1 (Base)")
        add_control(s1_left, "Blur Prob",    "blur_prob",           0.5,           "slider", 1.0)
        add_control(s1_left, "Blur Sigma",   "blur_sigma",          "[0.2, 3.0]",  "range")
        add_control(s1_left, "Kernel Size",  "blur_kernel_size",    21,            "slider", 31, 2)
        add_control(s1_left, "Noise Prob",   "gaussian_noise_prob", 0.5,           "slider", 1.0)
        add_control(s1_left, "Noise Range",  "noise_range",         "[1, 30]",     "range")
        add_control(s1_left, "Resize Prob",  "resize_prob",         "[0.2, 0.7, 0.1]", "entry")
        add_control(s1_left, "Resize Range", "resize_range",        "[0.5, 1.5]",  "range")

        self.add_header(s1_right, "Compression")
        add_control(s1_right, "JPEG Prob",      "jpeg_prob",       1.0,         "slider", 1.0)
        add_control(s1_right, "JPEG Range",     "jpeg_range",      "[30, 95]",  "range")
        add_control(s1_right, "Gray Noise",     "gray_noise_prob", 0.4,         "slider", 1.0)
        add_control(s1_right, "Sinc (Ringing)", "final_sinc_prob", 0.8,         "slider", 1.0)

        # --- Stage 2 : 2 colonnes côte à côte ---
        t2.columnconfigure(0, weight=1)
        t2.columnconfigure(1, weight=1)
        t2.rowconfigure(0, weight=1)
        s2_left  = ctk.CTkScrollableFrame(t2, fg_color="transparent")
        s2_left.grid(row=0, column=0, sticky="nsew", padx=(2, 1), pady=2)
        s2_right = ctk.CTkScrollableFrame(t2, fg_color="transparent")
        s2_right.grid(row=0, column=1, sticky="nsew", padx=(1, 2), pady=2)

        self.add_header(s2_left, _t("Stage 2 — Blur / Bruit", "Stage 2 — Blur / Noise"))
        add_control(s2_left, "2nd Blur Prob",  "second_blur_prob",    0.5,          "slider", 1.0)
        add_control(s2_left, "Blur Sigma 2",   "blur_sigma2",         "[0.2, 1.5]", "range")
        add_control(s2_left, "Kernel Size 2",  "blur_kernel_size2",   21,           "slider", 31, 2)
        add_control(s2_left, "Noise Prob 2",   "gaussian_noise_prob2", 0.5,         "slider", 1.0)
        add_control(s2_left, "Noise Range 2",  "noise_range2",        "[1, 25]",    "range")
        add_control(s2_left, "Gray Noise 2",   "gray_noise_prob2",    0.0,          "slider", 1.0)

        self.add_header(s2_right, "Stage 2 — Compression")
        add_control(s2_right, "Resize Range 2", "resize_range2", "[0.3, 1.2]", "range")
        add_control(s2_right, "JPEG Range 2",   "jpeg_range2",   "[30, 95]",   "range")

        # --- Custom degradations : 2 colonnes côte à côte ---
        tc.columnconfigure(0, weight=1)
        tc.columnconfigure(1, weight=1)
        tc.rowconfigure(0, weight=1)
        sc_left  = ctk.CTkScrollableFrame(tc, fg_color="transparent")
        sc_left.grid(row=0, column=0, sticky="nsew", padx=(2, 1), pady=2)
        sc_right = ctk.CTkScrollableFrame(tc, fg_color="transparent")
        sc_right.grid(row=0, column=1, sticky="nsew", padx=(1, 2), pady=2)

        self.add_header(sc_left, _t("Quantification", "Quantization"))
        add_control(sc_left, "Posterize Prob",  "posterize_prob",        0.0,         "slider", 1.0, pady=1)
        add_control(sc_left, "Posterize Bits",  "posterize_bits_range",  "[3, 6]",    "range",  pady=1)
        add_control(sc_left, "Banding Prob",    "banding_prob",          0.0,         "slider", 1.0, pady=1)
        add_control(sc_left, "Banding Levels",  "banding_levels_range",  "[16, 64]",  "range",  pady=1)
        # VHS/Analog dans le bloc gauche pour éviter le scroll dans le bloc droit
        self.add_header(sc_left, _t("VHS / Analog", "VHS / Analog"))
        add_control(sc_left, "VHS Prob",        "vhs_prob",              0.0,          "slider", 1.0, pady=1)
        add_control(sc_left, "VHS Strength",    "vhs_strength_range",    "[0.1, 0.5]", "range",  pady=1)

        self.add_header(sc_right, _t("Optique / Analog", "Optical / Analog"))
        add_control(sc_right, "Chroma Sub Prob",   "chroma_prob",              0.0,           "slider", 1.0, pady=1)
        add_control(sc_right, "Chrom. Aber. Prob", "ca_prob",                  0.0,           "slider", 1.0, pady=1)
        add_control(sc_right, "CA Shift (px)",     "ca_shift_range",           "[1, 5]",      "range",  pady=1)
        add_control(sc_right, "Halation Prob",     "halation_prob",            0.0,           "slider", 1.0, pady=1)
        add_control(sc_right, "Halation Strength", "halation_strength_range",  "[0.05, 0.3]", "range",  pady=1)
        add_control(sc_right, "Salt&Pepper Prob",  "salt_pepper_prob",         0.0,           "slider", 1.0, pady=1)
        add_control(sc_right, "S&P Amount",        "salt_pepper_amount_range", "[0.001, 0.05]","range",  pady=1)

        # --- Custom 2 : Aliasing + Interlace + Film Grain + OverSharp + Scanlines ---
        tc2.columnconfigure(0, weight=1)
        tc2.columnconfigure(1, weight=1)
        tc2.rowconfigure(0, weight=1)
        sc2_left  = ctk.CTkScrollableFrame(tc2, fg_color="transparent")
        sc2_left.grid(row=0, column=0, sticky="nsew", padx=(2, 1), pady=2)
        sc2_right = ctk.CTkScrollableFrame(tc2, fg_color="transparent")
        sc2_right.grid(row=0, column=1, sticky="nsew", padx=(1, 2), pady=2)

        self.add_header(sc2_left, _t("Aliasing / Lignes", "Aliasing / Lines"))
        add_control(sc2_left, "Aliasing Prob",   "aliasing_prob",           0.0,           "slider", 1.0, pady=1)
        add_control(sc2_left, "Aliasing Scale",  "aliasing_scale_range",    "[0.5, 0.85]", "range",  pady=1)
        self.add_header(sc2_left, _t("Entrelacement — Weave", "Interlace — Weave"))
        add_control(sc2_left, "Weave Prob",      "interlace_weave_prob",            0.0,         "slider", 1.0, pady=1)
        add_control(sc2_left, "Weave Strength",  "interlace_weave_strength_range",  "[0.5, 1.0]","range",  pady=1)
        self.add_header(sc2_left, _t("Entrelacement — Flicker", "Interlace — Flicker"))
        add_control(sc2_left, "Flicker Prob",    "interlace_flicker_prob",            0.0,          "slider", 1.0, pady=1)
        add_control(sc2_left, "Flicker Strength","interlace_flicker_strength_range",  "[0.1, 0.4]", "range",  pady=1)
        self.add_header(sc2_left, _t("Entrelacement — Blend", "Interlace — Blend"))
        add_control(sc2_left, "Blend Prob",      "interlace_blend_prob",            0.0,         "slider", 1.0, pady=1)
        add_control(sc2_left, "Blend Strength",  "interlace_blend_strength_range",  "[0.3, 1.0]","range",  pady=1)

        self.add_header(sc2_right, _t("Grain Cinéma", "Film Grain"))
        add_control(sc2_right, "Film Grain Prob",     "film_grain_prob",           0.0,            "slider", 1.0, pady=1)
        add_control(sc2_right, "Grain Strength",      "film_grain_strength_range", "[0.03, 0.12]", "range",  pady=1)
        add_control(sc2_right, "Grain Size (px)",     "film_grain_size_range",     "[1, 2]",       "range",  pady=1)
        self.add_header(sc2_right, _t("Sur-Netteté (Halos)", "Oversharpening (Halos)"))
        add_control(sc2_right, "Oversharp Prob",     "oversharp_prob",            0.0,          "slider", 1.0, pady=1)
        add_control(sc2_right, "Oversharp Strength", "oversharp_strength_range",  "[0.5, 2.0]", "range",  pady=1)
        self.add_header(sc2_right, _t("Scanlines CRT", "CRT Scanlines"))
        add_control(sc2_right, "Scanlines Prob",     "scanlines_prob",            0.0,          "slider", 1.0, pady=1)
        add_control(sc2_right, "Scanlines Strength", "scanlines_strength_range",  "[0.2, 0.5]", "range",  pady=1)
        add_control(sc2_right, "Scanlines Spacing",  "scanlines_spacing_range",   "[2, 4]",     "range",  pady=1)

        # --- Custom 3 : Screentone + Dithering + Pixelate + Sinusoïdal + Subsampling ---
        tc3.columnconfigure(0, weight=1)
        tc3.columnconfigure(1, weight=1)
        tc3.rowconfigure(0, weight=1)
        sc3_left  = ctk.CTkScrollableFrame(tc3, fg_color="transparent")
        sc3_left.grid(row=0, column=0, sticky="nsew", padx=(2, 1), pady=2)
        sc3_right = ctk.CTkScrollableFrame(tc3, fg_color="transparent")
        sc3_right.grid(row=0, column=1, sticky="nsew", padx=(1, 2), pady=2)

        # Helper pour OptionMenu row (pas encore dans add_control)
        def _opt_row(parent, label, key, values, default=None, tip=""):
            r = ctk.CTkFrame(parent, fg_color="transparent"); r.pack(fill="x", pady=1)
            ctk.CTkLabel(r, text=label, font=("Arial", 11), width=150, anchor="w").pack(side="left", padx=(5, 0))
            om = ctk.CTkOptionMenu(r, values=values, width=130)
            if default:
                om.set(default)
            om.pack(side="right", padx=5)
            if tip:
                ToolTip(om, tip)
            self.widgets[key] = om

        self.add_header(sc3_left, _t("Screentone / Halftone", "Screentone / Halftone"))
        add_control(sc3_left, "Screentone Prob",  "screentone_prob",       0.0,       "slider", 1.0, pady=1)
        add_control(sc3_left, "Dot Size",         "screentone_dot_size",   "[7, 15]", "range",  pady=1)
        add_control(sc3_left, "Angle",            "screentone_angle",      "[0, 90]", "range",  pady=1)
        _opt_row(sc3_left, _t("Type de point", "Dot Type"), "screentone_dot_type",
                 ["circle", "ellipse", "square", "cross"], "circle",
                 _t("Forme des points du motif screentone.", "Shape of screentone pattern dots."))
        _opt_row(sc3_left, _t("Espace couleur", "Color space"), "screentone_color_space",
                 ["rgb", "cmyk", "hsv", "gray"], "rgb",
                 _t("Espace couleur dans lequel le motif est appliqué.",
                    "Color space in which the pattern is applied."))

        self.add_header(sc3_left, _t("Dithering", "Dithering"))
        add_control(sc3_left, "Dithering Prob",   "dithering_prob",        0.0,      "slider", 1.0, pady=1)
        add_control(sc3_left, _t("Canaux couleur", "Color channels"), "dithering_color_ch", "[2, 8]", "range", pady=1)
        _opt_row(sc3_left, _t("Algorithme", "Algorithm"), "dithering_type",
                 ["quantize", "floyd_steinberg", "jarvis_judice", "stucki", "atkinson", "ordered"],
                 "floyd_steinberg",
                 _t("Algorithme de tramage.\n"
                    "• floyd_steinberg : diffusion d'erreur classique\n"
                    "• ordered : tramage ordonné (Bayer)\n"
                    "• quantize : réduction couleur simple\n"
                    "• atkinson : tramage léger (Apple Mac originel)",
                    "Dithering algorithm.\n"
                    "• floyd_steinberg : classic error diffusion\n"
                    "• ordered : ordered dithering (Bayer)\n"
                    "• quantize : simple color reduction\n"
                    "• atkinson : light dithering (original Apple Mac)"))

        self.add_header(sc3_right, _t("Pixelisation", "Pixelation"))
        add_control(sc3_right, "Pixelate Prob",   "pixelate_prob",         0.0,         "slider", 1.0, pady=1)
        add_control(sc3_right, "Block Size",      "pixelate_size",         "[2, 16]",   "range",  pady=1)

        self.add_header(sc3_right, _t("Sinusoïdal (ondulations)", "Sinusoidal (ripples)"))
        add_control(sc3_right, "Sin Prob",        "sin_prob",              0.0,          "slider", 1.0, pady=1)
        add_control(sc3_right, _t("Fréquence", "Frequency"), "sin_shape",   "[100, 600]", "range", pady=1)
        add_control(sc3_right, _t("Amplitude", "Amplitude"), "sin_alpha",   "[0.1, 0.4]", "range", pady=1)
        add_control(sc3_right, _t("Biais", "Bias"),          "sin_bias",    "[0.8, 1.2]", "range", pady=1)
        _opt_row(sc3_right, _t("Orientation", "Orientation"), "sin_orientation",
                 [_t("aléatoire", "random"), "horizontal", "vertical"], _t("aléatoire", "random"),
                 _t("Direction des ondulations sinusoïdales. Fréquence haute = serrées.",
                    "Direction of sinusoidal ripples. High frequency = tight waves."))

        self.add_header(sc3_right, _t("Sous-Échantillonnage Chroma", "Chroma Subsampling"))
        add_control(sc3_right, "Subsamp Prob",    "subsampling_prob",      0.0,  "slider", 1.0, pady=1)
        _opt_row(sc3_right, _t("Format", "Format"), "subsampling_format",
                 ["4:4:4", "4:2:2", "4:2:0", "4:1:1"], "4:2:0",
                 _t("Format de sous-échantillonnage chroma.\n"
                    "• 4:4:4 : aucune réduction\n"
                    "• 4:2:2 : réduction horizontale ×2\n"
                    "• 4:2:0 : réduction ×2 H et V (JPEG standard)\n"
                    "• 4:1:1 : forte réduction horizontale",
                    "Chroma subsampling format.\n"
                    "• 4:4:4 : no reduction\n"
                    "• 4:2:2 : horizontal ×2 reduction\n"
                    "• 4:2:0 : ×2 H and V reduction (standard JPEG)\n"
                    "• 4:1:1 : heavy horizontal reduction"))
        _opt_row(sc3_right, _t("Standard YUV", "YUV Standard"), "subsampling_yuv",
                 ["601", "709", "2020"], "601",
                 _t("Matrice de conversion YUV.\n"
                    "601 = SD/vidéo ancienne, 709 = HD, 2020 = UHD.",
                    "YUV conversion matrix.\n"
                    "601 = SD/legacy video, 709 = HD, 2020 = UHD."))

        # --- Custom 4 : Niveaux couleur + Saturation + Shift + Halo (wtp) ---
        tc4.columnconfigure(0, weight=1)
        tc4.columnconfigure(1, weight=1)
        tc4.rowconfigure(0, weight=1)
        sc4_left  = ctk.CTkScrollableFrame(tc4, fg_color="transparent")
        sc4_left.grid(row=0, column=0, sticky="nsew", padx=(2, 1), pady=2)
        sc4_right = ctk.CTkScrollableFrame(tc4, fg_color="transparent")
        sc4_right.grid(row=0, column=1, sticky="nsew", padx=(1, 2), pady=2)

        self.add_header(sc4_left, _t("Niveaux de Couleur", "Color Levels"))
        add_control(sc4_left, "Color Level Prob", "color_level_prob",      0.0,           "slider", 1.0, pady=1)
        add_control(sc4_left, "Output High",      "color_level_high",      "[220, 255]",  "range",  pady=1)
        add_control(sc4_left, "Output Low",       "color_level_low",       "[0, 35]",     "range",  pady=1)
        add_control(sc4_left, "Gamma",            "color_level_gamma",     "[0.7, 1.5]",  "range",  pady=1)

        self.add_header(sc4_left, _t("Halo / Ringing (wtp)", "Halo / Ringing (wtp)"))
        add_control(sc4_left, "Halo Prob",        "wtp_halo_prob",         0.0,           "slider", 1.0, pady=1)
        add_control(sc4_left, "Halo Strength",    "wtp_halo_strength",     "[0.1, 0.5]",  "range",  pady=1)
        add_control(sc4_left, "Halo Radius",      "wtp_halo_radius",       "[3, 12]",     "range",  pady=1)

        self.add_header(sc4_right, _t("Saturation", "Saturation"))
        add_control(sc4_right, "Saturation Prob", "saturation_prob",       0.0,           "slider", 1.0, pady=1)
        add_control(sc4_right, "Saturation",      "saturation_range",      "[0.3, 1.8]",  "range",  pady=1)

        self.add_header(sc4_right, _t("Décalage Pixel (Shift)", "Pixel Shift"))
        add_control(sc4_right, "Shift Prob",      "shift_prob",            0.0,           "slider", 1.0, pady=1)
        add_control(sc4_right, "Amplitude (px)",  "shift_range",           "[1, 8]",      "range",  pady=1)
        _opt_row(sc4_right, _t("Axe", "Axis"), "shift_axis",
                 [_t("aléatoire", "random"), "horizontal", "vertical", _t("les deux", "both")],
                 _t("aléatoire", "random"),
                 _t("Axe de décalage pixel.\n"
                    "• horizontal : glitch latéral\n"
                    "• vertical : glitch vertical\n"
                    "• les deux : combiné",
                    "Pixel shift axis.\n"
                    "• horizontal : lateral glitch\n"
                    "• vertical : vertical glitch\n"
                    "• both : combined"))

        # ─── Live preview (toujours visible sous les onglets) ───
        f.grid_propagate(False)             # empêche f de grandir quand la preview charge une image
        f.grid_rowconfigure(0, weight=0)
        f.grid_rowconfigure(1, weight=1, minsize=245)  # tabview — adaptatif, min 245px
        f.grid_rowconfigure(2, weight=4)               # preview — 4x l'espace restant
        f_preview = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8)
        f_preview.grid(row=2, column=0, sticky="nsew", padx=5, pady=(6, 5))
        ctk.CTkLabel(f_preview, text=_t("🔍 Aperçu en direct des dégradations", "🔍 Live degradation preview"),
                     font=("Roboto", 13, "bold"), text_color="#9b59b6"
                     ).pack(anchor="w", padx=10, pady=(8, 5))
        ctk.CTkLabel(f_preview,
                     text=_t("Charge une image source, puis applique les paramètres ci-dessus pour voir le résultat. "
                              "Cliquez sur l'image pour zoomer (molette = zoom, glisser = pan).",
                              "Load a source image, then apply the parameters above to see the result. "
                              "Click the image to zoom (scroll wheel = zoom, drag = pan)."),
                     text_color=("gray30", "#AAA"), font=("Roboto", 10)
                     ).pack(anchor="w", padx=10, pady=(0, 5))

        prev_ctrl = ctk.CTkFrame(f_preview, fg_color="transparent")
        prev_ctrl.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(prev_ctrl, text=_t("Image :", "Image:"), width=60, anchor="w").pack(side="left")
        self.widgets["deg_preview_src"] = ctk.CTkEntry(prev_ctrl, width=380)
        self.widgets["deg_preview_src"].pack(side="left", padx=5)
        _last_img = self.settings.get("last_preview_image", "")
        if _last_img and os.path.isfile(_last_img):
            self.widgets["deg_preview_src"].insert(0, _last_img)
        ctk.CTkButton(prev_ctrl, text="...", width=30,
                      command=self._deg_browse_image).pack(side="left", padx=2)
        ctk.CTkLabel(prev_ctrl, text="  Scale :").pack(side="left", padx=(10, 2))
        self.widgets["deg_preview_scale"] = ctk.CTkOptionMenu(
            prev_ctrl, values=["1", "2", "3", "4", "8"], width=60
        )
        self.widgets["deg_preview_scale"].pack(side="left", padx=2)
        self.widgets["deg_preview_scale"].set("4")
        ctk.CTkButton(prev_ctrl, text=_t("🎲 Régénérer", "🎲 Regenerate"), fg_color="#9b59b6", width=120,
                      command=self._deg_refresh_preview).pack(side="left", padx=10)
        ctk.CTkButton(prev_ctrl, text=_t("🔍 Zoom", "🔍 Zoom"), fg_color="#16a085", width=80,
                      command=self._deg_zoom_preview).pack(side="left", padx=2)

        ctk.CTkLabel(prev_ctrl, text="  Var :").pack(side="left", padx=(12, 2))
        self.widgets["deg_otf_variants"] = ctk.CTkEntry(prev_ctrl, width=38)
        self.widgets["deg_otf_variants"].pack(side="left")
        self.widgets["deg_otf_variants"].insert(0, "1")
        ctk.CTkButton(prev_ctrl, text=_t("📁 Tester GT Val", "📁 Test GT Val"),
                      fg_color="#2471a3", width=150,
                      command=self._deg_generate_otf_batch).pack(side="left", padx=(8, 2))

        # Image display area (expands with frame)
        self.widgets["deg_preview_area"] = ctk.CTkFrame(f_preview, fg_color=("#EBEBEB", "#0d0d1a"))
        self.widgets["deg_preview_area"].pack(fill="both", expand=True, padx=10, pady=(5, 10))
        self.widgets["deg_preview_area"].pack_propagate(False)
        self._deg_preview_refs = {"hq": None, "lq": None, "hq_orig": None, "lq_orig": None}
        # Placeholder
        ctk.CTkLabel(self.widgets["deg_preview_area"],
                     text=_t("(Aucune image chargée — sélectionnez une image source)",
                              "(No image loaded — select a source image)"),
                     text_color="#666", font=("Roboto", 11)
                     ).pack(pady=80)

        return f

    def create_page_advanced(self):
        outer = ctk.CTkFrame(self.page_container, fg_color="transparent")
        # Scrollable container so content is visible even when window is not maximized
        f = ctk.CTkScrollableFrame(outer, fg_color="transparent")
        f.pack(fill="both", expand=True)
        self.add_header(f, _t("Optimisations Système", "System Optimizations"))
        f_checks = ctk.CTkFrame(f, fg_color="transparent"); f_checks.pack(fill="x", pady=5)
        f_gpu = ctk.CTkFrame(f, fg_color="transparent"); f_gpu.pack(fill="x", pady=2)
        ctk.CTkLabel(f_gpu, text="Num GPU :", width=80, anchor="w").pack(side="left")
        self.widgets["num_gpu"] = ctk.CTkEntry(f_gpu, width=50); self.widgets["num_gpu"].insert(0, "1"); self.widgets["num_gpu"].pack(side="left"); ToolTip(self.widgets["num_gpu"], get_tooltip("num_gpu"))
        for k, txt in [("use_amp", "Use AMP (FP16)"), ("bfloat16", "Use BF16 (RTX 30+)"), ("fast_matmul", "Fast MatMul (TF32)"), ("compile", "Torch Compile (Linux)"), ("grad_clip", "Gradient Clipping"), ("match_lq_colors", "Match Colors (LQ->GT)"), ("eco", "Mode ECO (Low Memory)")]:
            chk = ctk.CTkCheckBox(f_checks, text=txt, onvalue="true", offvalue="false", command=lambda: self.refresh_ui_stats()); chk.pack(anchor="w", pady=2); self.widgets[k] = chk; ToolTip(chk, get_tooltip(k))
        f_sam = ctk.CTkFrame(f, fg_color="transparent"); f_sam.pack(fill="x", pady=2)
        self.add_label_tip(f_sam, "Mode SAM :", "sam"); self.widgets["sam"] = ctk.CTkOptionMenu(f_sam, values=["none", "sam", "fsam"]); self.widgets["sam"].pack(side="left")
        self.add_label_tip(f_sam, "Init :", "sam_init"); self.widgets["sam_init"] = ctk.CTkEntry(f_sam, width=60); self.widgets["sam_init"].insert(0, "-1"); self.widgets["sam_init"].pack(side="left")
        # --- EMA ---
        f_ema = ctk.CTkFrame(f, fg_color="transparent"); f_ema.pack(fill="x", pady=2)
        ctk.CTkLabel(f_ema, text="EMA :", width=80, anchor="w").pack(side="left")
        self.widgets["ema"] = ctk.CTkEntry(f_ema, width=80); self.widgets["ema"].insert(0, "0.999"); self.widgets["ema"].pack(side="left", padx=5)
        ToolTip(self.widgets["ema"], _t("Momentum EMA (Exponential Moving Average).\n0.999 = standard, modèle plus stable.\n0 = désactivé.\nNeoSR : clé 'ema'  |  Redux : clé 'ema_decay'",
                                         "Momentum EMA (Exponential Moving Average).\n0.999 = standard, more stable model.\n0 = disabled.\nNeoSR: key 'ema'  |  Redux: key 'ema_decay'"))
        self.add_header(f, _t("Fonctions de Perte (Losses)", "Loss Functions (Losses)"))
        # ── Conteneur horizontal : colonne gauche (normal) | colonne droite (Redux-only) ──
        # pack anchor="w" → l'outer ne s'étend pas à pleine largeur, les deux blocs restent côte à côte
        # ── Conteneur pleine largeur fond sombre ──────────────────────────────────
        f_loss_adv_outer = ctk.CTkFrame(f, fg_color=("#EBEBEB", "#0d1117"), corner_radius=8)
        f_loss_adv_outer.pack(fill="x", pady=5)
        # Wrapper interne avec padding
        _f_inner = ctk.CTkFrame(f_loss_adv_outer, fg_color="transparent")
        _f_inner.pack(fill="x", padx=10, pady=8)
        # Gauche : losses normales (transparent sur fond sombre)
        f_loss_left = ctk.CTkFrame(_f_inner, fg_color="transparent")
        f_loss_left.pack(side="left", fill="y", padx=(0, 12))
        # Droite : Redux uniquement — largeur fixe 430px, fond légèrement plus clair
        f_loss_right = ctk.CTkFrame(_f_inner, fg_color=("#DEDEDE", "#111827"), corner_radius=6, width=430)
        f_loss_right.pack(side="left", fill="y")
        f_loss_right.pack_propagate(False)
        ctk.CTkLabel(f_loss_right, text=_t("⚡ Redux uniquement", "⚡ Redux only"), text_color="#3B8ED0",
                     font=("Arial", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        # alias : tout le code existant utilise f_loss_adv → redirige vers f_loss_left
        f_loss_adv = f_loss_left
        
        def add_weighted_loss(parent, key, label, default_w=1.0, has_reduction=False, has_type=None):
            row = ctk.CTkFrame(parent, fg_color="transparent"); row.pack(fill="x", pady=2)
            chk = ctk.CTkCheckBox(row, text=label, width=140, onvalue="true", offvalue="false"); chk.pack(side="left"); self.widgets[key] = chk; ToolTip(chk, get_tooltip(key))
            lbl_w = ctk.CTkLabel(row, text="W:", width=20); lbl_w.pack(side="left")
            e = ctk.CTkEntry(row, width=50); e.insert(0, str(default_w)); e.pack(side="left", padx=5); self.widgets[f"weight_{key}"] = e
            if has_reduction:
                lbl_r = ctk.CTkLabel(row, text="Mode:", width=40); lbl_r.pack(side="left")
                opt = ctk.CTkOptionMenu(row, values=["mean", "sum"], width=80); opt.pack(side="left"); opt.set("mean"); self.widgets["pixel_reduction"] = opt; ToolTip(opt, get_tooltip("pixel_reduction"))
            if has_type is not None:
                lbl_t = ctk.CTkLabel(row, text="Type:", width=40); lbl_t.pack(side="left")
                opt_t = ctk.CTkOptionMenu(row, values=has_type, width=130); opt_t.pack(side="left", padx=5); opt_t.set(has_type[0]); self.widgets["pixel_criterion"] = opt_t; ToolTip(opt_t, get_tooltip("pixel_criterion"))

        add_weighted_loss(f_loss_adv, "loss_pixel", "Pixel L1 / Charbonnier", 1.0, has_reduction=True, has_type=["L1Loss", "MSELoss", "HuberLoss", "chc_loss"]); self.widgets["loss_pixel"].select()
        f_perc = ctk.CTkFrame(f_loss_adv, fg_color="transparent"); f_perc.pack(fill="x", pady=2)
        chk_p = ctk.CTkCheckBox(f_perc, text="Perceptual (VGG)", width=140, onvalue="true", offvalue="false"); chk_p.pack(side="left"); self.widgets["loss_percep"] = chk_p; chk_p.select(); ToolTip(chk_p, get_tooltip("loss_percep"))
        lbl_w = ctk.CTkLabel(f_perc, text="W:", width=20); lbl_w.pack(side="left"); self.widgets["weight_loss_percep"] = ctk.CTkEntry(f_perc, width=50); self.widgets["weight_loss_percep"].insert(0, "1.0"); self.widgets["weight_loss_percep"].pack(side="left", padx=5)
        self.widgets["percep_criterion"] = ctk.CTkOptionMenu(f_perc, values=["huber", "l1", "l2", "charbonnier", "chc_loss"], width=110); self.widgets["percep_criterion"].pack(side="left", padx=5); self.widgets["percep_criterion"].set("huber"); ToolTip(self.widgets["percep_criterion"], get_tooltip("percep_criterion"))
        # VGG layers — CTkToplevel popup à droite du bouton
        _VGG_LAYERS = [("conv1_2","0.0"),("conv2_2","0.0"),("conv3_2","0.0"),("conv3_4","0.0"),
                       ("conv4_2","0.0"),("conv4_4","0.0"),("conv5_2","0.0"),("conv5_4","1.0")]
        _VGG_TIPS = {
            "conv1_2": "conv1_2 — Textures très fines, grain, arêtes sub-pixel.\nPremière couche : répond aux contrastes locaux de 3×3px.\nÉffet : sharpening micro-détails, peut amplifier le bruit.",
            "conv2_2": "conv2_2 — Motifs répétitifs, micro-textures (peau, tissu, écailles).\nDétecte les structures périodiques fines.\nÉffet : cohérence de texture à petite échelle.",
            "conv3_2": "conv3_2 — Textures moyennes (poils, herbe, feuilles, pavés).\nCapte les régions texturées ~8–16px.\nÉffet : textures organiques naturelles.",
            "conv3_4": "conv3_4 — Structures locales, contours durs, patterns de surface.\nCouche clé pour les objets à bords nets.\nÉffet : netteté des contours et relief des surfaces.",
            "conv4_2": "conv4_2 — Parties d'objets reconnaissables, forme des éléments.\nDébut de la sémantique basse : détecte des parties d'objets.\nÉffet : cohérence des formes et structures complexes.",
            "conv4_4": "conv4_4 — Style sémantique, tonalité générale, type de scène.\nIntègre texture + forme : style visuel global.\nÉffet : cohérence stylistique, rendu artistique.",
            "conv5_2": "conv5_2 — Contenu sémantique haut niveau, identité objet.\nRépond aux concepts visuels complets (visage, voiture…).\nÉffet : préserve l'identité et le contenu global.",
            "conv5_4": "conv5_4 — Sémantique complète, perception globale (défaut NeoSR).\nCouche la plus abstraite : active sur des objets entiers.\nÉffet : loss perceptuelle au niveau de la scène entière.\nValeur recommandée : 1.0 seule, ou combiner avec conv3_4.",
        }
        # Hidden CTkEntry widgets (parent=f_loss_adv, not packed) — used by config_handler via .get()/.insert()
        for _layer, _dw in _VGG_LAYERS:
            _hidden = ctk.CTkEntry(f_loss_adv, width=42)
            _hidden.insert(0, _dw)
            self.widgets[f"percep_vgg_{_layer}"] = _hidden
        # Legacy compat key (Redux + flat config loader) — also hidden
        _percep_layer_opt = ctk.CTkOptionMenu(f_loss_adv,
            values=["conv1_2","conv2_2","conv3_2","conv3_4","conv4_2","conv4_4","conv5_2","conv5_4"],
            width=90)
        _percep_layer_opt.set("conv5_4")
        self.widgets["percep_layer"] = _percep_layer_opt

        # Persistent popup (hidden until button clicked)
        _vgg_win = ctk.CTkToplevel(self)
        _vgg_win.withdraw()
        _vgg_win.wm_overrideredirect(True)
        _vgg_win.attributes("-topmost", True)
        _vgg_win.configure(fg_color=("#E8E8E8", "#1a1a2e"))
        # Header
        _vgg_hdr = ctk.CTkFrame(_vgg_win, fg_color="#0d1117", corner_radius=0)
        _vgg_hdr.pack(fill="x")
        ctk.CTkLabel(_vgg_hdr, text=_t("  Poids couches VGG  (0.0 = désactivée)",
                                        "  VGG layer weights  (0.0 = disabled)"),
                     font=("Arial", 10, "bold"), text_color=("gray30", "#ccc")).pack(side="left", padx=4, pady=4)
        ctk.CTkButton(_vgg_hdr, text="✕", width=24, height=20, fg_color="#333", hover_color="#555",
                      command=lambda: (_vgg_win.withdraw(), btn_vgg.configure(text=_t("▾ Couches VGG", "▾ VGG Layers")))
                      ).pack(side="right", padx=4, pady=4)
        # Grid 4-per-row with shared StringVar → updates hidden widget on change
        _vgg_popup_entries = {}
        for _row_start in range(0, len(_VGG_LAYERS), 4):
            _vgg_row = ctk.CTkFrame(_vgg_win, fg_color="transparent"); _vgg_row.pack(fill="x", padx=6, pady=2)
            for _layer, _dw in _VGG_LAYERS[_row_start:_row_start + 4]:
                _fc = ctk.CTkFrame(_vgg_row, fg_color="transparent"); _fc.pack(side="left", padx=8)
                ctk.CTkLabel(_fc, text=_layer, font=("Consolas", 10),
                             width=60, anchor="w", text_color="#aaa").pack(side="left")
                _sv = tk.StringVar(value=_dw)
                # Keep hidden widget in sync
                def _make_trace(_l, _sv=_sv):
                    def _cb(*_):
                        w = self.widgets.get(f"percep_vgg_{_l}")
                        if w:
                            try: w.delete(0, "end"); w.insert(0, _sv.get())
                            except Exception: pass
                    return _cb
                _sv.trace_add("write", _make_trace(_layer))
                _ep = ctk.CTkEntry(_fc, width=46, textvariable=_sv); _ep.pack(side="left")
                _vgg_popup_entries[_layer] = (_sv, _ep)
                ToolTip(_ep, _VGG_TIPS.get(_layer, _t(f"Poids couche {_layer}.\n0.0 = ignorée  |  1.0 = plein poids",
                                                         f"Layer {_layer} weight.\n0.0 = ignored  |  1.0 = full weight")))
        ctk.CTkLabel(_vgg_win,
                     text=_t("💡 conv5_4=1.0 (défaut)  ·  conv3+4 = style  ·  conv1+2 = détails",
                              "💡 conv5_4=1.0 (default)  ·  conv3+4 = style  ·  conv1+2 = details"),
                     text_color="#555", font=("Arial", 9)).pack(anchor="w", padx=8, pady=(2, 6))

        self._vgg_popup_entries = _vgg_popup_entries  # for config load sync

        def _toggle_vgg():
            if _vgg_win.winfo_viewable():
                _vgg_win.withdraw(); btn_vgg.configure(text=_t("▾ Couches VGG", "▾ VGG Layers"))
            else:
                # Sync popup entries from self.widgets (in case config was loaded)
                for _l, (_sv2, _ep2) in _vgg_popup_entries.items():
                    try: _sv2.set(self.widgets[f"percep_vgg_{_l}"].get())
                    except Exception: pass
                self.update_idletasks()
                bx = btn_vgg.winfo_rootx() + btn_vgg.winfo_width() + 6
                by = btn_vgg.winfo_rooty()
                _vgg_win.wm_geometry(f"+{bx}+{by}")
                _vgg_win.deiconify(); _vgg_win.lift()
                btn_vgg.configure(text=_t("▲ Couches VGG", "▲ VGG Layers"))

        btn_vgg = ctk.CTkButton(f_perc, text=_t("▾ Couches VGG", "▾ VGG Layers"), width=115, fg_color="#2c3e50",
                                command=_toggle_vgg); btn_vgg.pack(side="left", padx=5)
        ToolTip(btn_vgg, _t("Sélection individuelle des couches VGG et leur poids.\nPoids > 0 = couche active.\nconv5_4=1.0 par défaut.\nconv1_2/conv2_2 = détails fins  |  conv3_4/conv4_4 = style/texture  |  conv5_4 = sémantique",
                             "Individual VGG layer selection and weights.\nWeight > 0 = active layer.\nconv5_4=1.0 by default.\nconv1_2/conv2_2 = fine details  |  conv3_4/conv4_4 = style/texture  |  conv5_4 = semantics"))

        f_fdl = ctk.CTkFrame(f_loss_adv, fg_color="transparent"); f_fdl.pack(fill="x", pady=2)
        chk_fdl = ctk.CTkCheckBox(f_fdl, text="FDL (Freq)", width=140, onvalue="true", offvalue="false"); chk_fdl.pack(side="left"); self.widgets["loss_fdl"] = chk_fdl; ToolTip(chk_fdl, get_tooltip("loss_fdl"))
        lbl_w = ctk.CTkLabel(f_fdl, text="W:", width=20); lbl_w.pack(side="left"); self.widgets["weight_loss_fdl"] = ctk.CTkEntry(f_fdl, width=50); self.widgets["weight_loss_fdl"].insert(0, "1.0"); self.widgets["weight_loss_fdl"].pack(side="left", padx=5)
        self.widgets["fdl_model"] = ctk.CTkOptionMenu(f_fdl, values=["vgg", "dinov2", "resnet", "effnet"], width=90); self.widgets["fdl_model"].pack(side="left"); self.widgets["fdl_model"].set("vgg"); ToolTip(self.widgets["fdl_model"], get_tooltip("fdl_model"))

        # Helper : popup d'options pour une loss (CTkToplevel persistant, caché par défaut)
        def _loss_popup(title, fields):
            """
            fields : list of (key, label, default, "entry"|"option"|"check", options_or_None, tooltip)
            Crée une fenêtre popup et enregistre les widgets dans self.widgets[key].
            Retourne (win, toggle_fn) où toggle_fn(btn_widget) bascule la visibilité.
            """
            _win = ctk.CTkToplevel(self)
            _win.withdraw()
            _win.wm_overrideredirect(True)
            _win.attributes("-topmost", True)
            _win.configure(fg_color=("#E8E8E8", "#1a1a2e"))
            _btn_ref = [None]
            _hdr = ctk.CTkFrame(_win, fg_color="#0d1117", corner_radius=0); _hdr.pack(fill="x")
            ctk.CTkLabel(_hdr, text=f"  {title}", font=("Arial", 10, "bold"), text_color=("gray30", "#ccc")).pack(side="left", padx=4, pady=4)
            ctk.CTkButton(_hdr, text="✕", width=24, height=20, fg_color="#333", hover_color="#555",
                          command=lambda: (_win.withdraw(), _btn_ref[0] and _btn_ref[0].configure(text="⚙"))
                          ).pack(side="right", padx=4, pady=4)
            for key, lbl, default, wtype, opts, tip in fields:
                _row = ctk.CTkFrame(_win, fg_color="transparent"); _row.pack(fill="x", padx=8, pady=3)
                ctk.CTkLabel(_row, text=lbl + ":", width=130, anchor="w", text_color="#bbb").pack(side="left")
                if wtype == "entry":
                    _w = ctk.CTkEntry(_row, width=75); _w.insert(0, str(default)); _w.pack(side="left")
                elif wtype == "option":
                    _w = ctk.CTkOptionMenu(_row, values=opts, width=120); _w.set(str(default)); _w.pack(side="left")
                elif wtype == "check":
                    _w = ctk.CTkCheckBox(_row, text="", onvalue="true", offvalue="false")
                    if default: _w.select()
                    _w.pack(side="left")
                self.widgets[key] = _w
                if tip: ToolTip(_w, tip)
            ctk.CTkLabel(_win, text="", height=6).pack()
            def _toggle(btn_widget, _w=_win, _r=_btn_ref):
                _r[0] = btn_widget
                if _w.winfo_viewable():
                    _w.withdraw(); btn_widget.configure(text="⚙")
                else:
                    self.update_idletasks()
                    bx = btn_widget.winfo_rootx() + btn_widget.winfo_width() + 6
                    by = btn_widget.winfo_rooty()
                    _w.wm_geometry(f"+{bx}+{by}")
                    _w.deiconify(); _w.lift()
                    btn_widget.configure(text="▲")
            return _win, _toggle

        # --- MS-SSIM ---
        f_mssim = ctk.CTkFrame(f_loss_adv, fg_color="transparent"); f_mssim.pack(fill="x", pady=2)
        chk_mssim = ctk.CTkCheckBox(f_mssim, text="MS-SSIM", width=140, onvalue="true", offvalue="false"); chk_mssim.pack(side="left"); self.widgets["loss_mssim"] = chk_mssim; ToolTip(chk_mssim, get_tooltip("loss_mssim"))
        ctk.CTkLabel(f_mssim, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_mssim"] = ctk.CTkEntry(f_mssim, width=50); self.widgets["weight_loss_mssim"].insert(0, "1.0"); self.widgets["weight_loss_mssim"].pack(side="left", padx=5)
        _, _mssim_toggle = _loss_popup("MS-SSIM Options (NeoSR)", [
            ("mssim_window_size", "Window Size", "11",   "entry",  None, "Taille de la fenêtre Gaussienne.\nDéfaut : 11 (NeoSR)"),
            ("mssim_sigma",       "Sigma",       "1.5",  "entry",  None, "Écart-type Gaussien.\nDéfaut : 1.5 (NeoSR)"),
            ("mssim_k1",          "K1",          "0.01", "entry",  None, "Constante de stabilité K1.\nDéfaut : 0.01"),
            ("mssim_k2",          "K2",          "0.03", "entry",  None, "Constante de stabilité K2.\nDéfaut : 0.03"),
        ])
        _btn_mssim = ctk.CTkButton(f_mssim, text="⚙", width=30, fg_color="#2c3e50", command=lambda: _mssim_toggle(_btn_mssim)); _btn_mssim.pack(side="left", padx=5)
        ToolTip(_btn_mssim, _t("Paramètres avancés MS-SSIM (NeoSR).", "Advanced MS-SSIM parameters (NeoSR)."))

        # --- DISTS ---
        add_weighted_loss(f_loss_adv, "loss_dists", "DISTS", 1.0)

        # --- MS-SWD (NeoSR only) ---
        add_weighted_loss(f_loss_adv, "loss_msswd", "MS-SWD", 1.0)

        # --- Focal Freq ---
        f_ff = ctk.CTkFrame(f_loss_adv, fg_color="transparent"); f_ff.pack(fill="x", pady=2)
        chk_ff = ctk.CTkCheckBox(f_ff, text="Focal Freq", width=140, onvalue="true", offvalue="false"); chk_ff.pack(side="left"); self.widgets["loss_ff"] = chk_ff; ToolTip(chk_ff, get_tooltip("loss_ff"))
        ctk.CTkLabel(f_ff, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_ff"] = ctk.CTkEntry(f_ff, width=50); self.widgets["weight_loss_ff"].insert(0, "1.0"); self.widgets["weight_loss_ff"].pack(side="left", padx=5)
        ctk.CTkLabel(f_ff, text="α:", width=20).pack(side="left"); self.widgets["ff_alpha"] = ctk.CTkEntry(f_ff, width=50); self.widgets["ff_alpha"].insert(0, "1.0"); self.widgets["ff_alpha"].pack(side="left", padx=5)
        ToolTip(self.widgets["ff_alpha"], _t("Alpha — poids des fréquences hautes.\n1.0 = équilibré. >1 = accent hautes fréquences.\nDéfaut : 1.0",
                                              "Alpha — high frequency weight.\n1.0 = balanced. >1 = emphasis on high frequencies.\nDefault: 1.0"))

        # --- Consistency (NeoSR only) ---
        f_cst = ctk.CTkFrame(f_loss_adv, fg_color="transparent"); f_cst.pack(fill="x", pady=2)
        chk_cst = ctk.CTkCheckBox(f_cst, text="Consistency", width=140, onvalue="true", offvalue="false"); chk_cst.pack(side="left"); self.widgets["loss_consistency"] = chk_cst; ToolTip(chk_cst, get_tooltip("loss_consistency"))
        ctk.CTkLabel(f_cst, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_consistency"] = ctk.CTkEntry(f_cst, width=50); self.widgets["weight_loss_consistency"].insert(0, "1.0"); self.widgets["weight_loss_consistency"].pack(side="left", padx=5)
        _, _cst_toggle = _loss_popup("Consistency Options (NeoSR)", [
            ("consistency_blur",       "Blur",       True, "check", None,
             "Contrainte de cohérence par flou.\nApplique un filtre gaussien aux deux images avant comparaison.\nRéduit la sensibilité aux micro-décalages sub-pixel.\nRecommandé : actif."),
            ("consistency_cosim",      "CoSim",      True, "check", None,
             "Similarité Cosinus entre sortie et cible.\nMesure l'angle entre vecteurs de features (indépendant de la magnitude).\nRenforce la cohérence directionnelle des couleurs.\nRecommandé : actif."),
            ("consistency_saturation", "Saturation", True, "check", None,
             "Contrainte de saturation.\nPénalise les écarts de saturation entre sortie et cible (espace Oklab).\nÉvite les dérives de vivacité des couleurs.\nRecommandé : actif."),
            ("consistency_brightness", "Brightness", True, "check", None,
             "Contrainte de luminosité.\nPénalise les écarts de luminosité perçue (canal L* CIE).\nÉvite que l'image sortie soit plus sombre ou plus claire que la cible.\nRecommandé : actif."),
        ])
        _btn_cst = ctk.CTkButton(f_cst, text="⚙", width=30, fg_color="#2c3e50", command=lambda: _cst_toggle(_btn_cst)); _btn_cst.pack(side="left", padx=5)
        ToolTip(_btn_cst, _t("Activer/désactiver les contraintes de consistance (NeoSR).", "Enable/disable consistency constraints (NeoSR)."))

        # --- NCC (NeoSR only) ---
        add_weighted_loss(f_loss_adv, "loss_ncc", "NCC", 1.0)

        # --- KL Divergence (NeoSR only) ---
        add_weighted_loss(f_loss_adv, "loss_kl", "KL Divergence", 1.0)

        # --- LDL — criterion + ksize ---
        f_ldl = ctk.CTkFrame(f_loss_adv, fg_color="transparent"); f_ldl.pack(fill="x", pady=2)
        chk_ldl = ctk.CTkCheckBox(f_ldl, text="LDL", width=140, onvalue="true", offvalue="false"); chk_ldl.pack(side="left"); self.widgets["loss_ldl"] = chk_ldl; ToolTip(chk_ldl, get_tooltip("loss_ldl"))
        ctk.CTkLabel(f_ldl, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_ldl"] = ctk.CTkEntry(f_ldl, width=50); self.widgets["weight_loss_ldl"].insert(0, "1.0"); self.widgets["weight_loss_ldl"].pack(side="left", padx=5)
        self.widgets["ldl_criterion"] = ctk.CTkOptionMenu(f_ldl, values=["l1", "l2", "huber"], width=80); self.widgets["ldl_criterion"].pack(side="left", padx=5); self.widgets["ldl_criterion"].set("l1"); ToolTip(self.widgets["ldl_criterion"], _t("Critère LDL.\nl1 : Standard.\nl2 : Lissé.\nhuber : Hybride L1/L2.", "LDL criterion.\nl1: Standard.\nl2: Smoothed.\nhuber: L1/L2 hybrid."))
        ctk.CTkLabel(f_ldl, text="ksize:", width=45).pack(side="left"); self.widgets["ldl_ksize"] = ctk.CTkEntry(f_ldl, width=45); self.widgets["ldl_ksize"].insert(0, "7"); self.widgets["ldl_ksize"].pack(side="left", padx=5)
        ToolTip(self.widgets["ldl_ksize"], _t("Taille du kernel LDL (entier impair).\n7 = défaut.", "LDL kernel size (odd integer).\n7 = default."))

        # --- Edge Loss (NeoSR only) ---
        f_edge = ctk.CTkFrame(f_loss_adv, fg_color="transparent"); f_edge.pack(fill="x", pady=2)
        chk_edge = ctk.CTkCheckBox(f_edge, text="Edge Loss", width=140, onvalue="true", offvalue="false"); chk_edge.pack(side="left"); self.widgets["loss_edge"] = chk_edge; ToolTip(chk_edge, get_tooltip("loss_edge"))
        ctk.CTkLabel(f_edge, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_edge"] = ctk.CTkEntry(f_edge, width=50); self.widgets["weight_loss_edge"].insert(0, "0.05"); self.widgets["weight_loss_edge"].pack(side="left", padx=5)
        self.widgets["edge_criterion"] = ctk.CTkOptionMenu(f_edge, values=["l1", "l2", "huber", "chc"], width=80); self.widgets["edge_criterion"].pack(side="left", padx=5); self.widgets["edge_criterion"].set("l1"); ToolTip(self.widgets["edge_criterion"], _t("Critère Edge Loss.\nl1/l2/huber : Standards.\nchc : Clipped Huber+Cosine (NeoSR uniquement).", "Edge Loss criterion.\nl1/l2/huber: Standard.\nchc: Clipped Huber+Cosine (NeoSR only)."))
        self.widgets["edge_corner"] = ctk.CTkCheckBox(f_edge, text="Corner", width=70, onvalue="true", offvalue="false"); self.widgets["edge_corner"].pack(side="left", padx=5); ToolTip(self.widgets["edge_corner"], _t("Activer la détection de coins.\nRenforce les angles et intersections.", "Enable corner detection.\nStrengthens angles and intersections."))

        # --- Wavelet Guided (NeoSR only) — dans le bloc normal ---
        row_wav = ctk.CTkFrame(f_loss_adv, fg_color="transparent"); row_wav.pack(fill="x", pady=2)
        chk_wav = ctk.CTkCheckBox(row_wav, text="Wavelet Guided", onvalue="true", offvalue="false", width=140); chk_wav.pack(side="left"); self.widgets["loss_wavelet"] = chk_wav; ToolTip(chk_wav, get_tooltip("loss_wavelet"))
        lbl_w = ctk.CTkLabel(row_wav, text="W:", width=20); lbl_w.pack(side="left"); e_wav = ctk.CTkEntry(row_wav, width=40); e_wav.insert(0, "1.0"); e_wav.pack(side="left", padx=5); self.widgets["weight_loss_wavelet"] = e_wav
        lbl_i = ctk.CTkLabel(row_wav, text="Init:", width=30); lbl_i.pack(side="left"); e_init = ctk.CTkEntry(row_wav, width=50); e_init.insert(0, "10000"); e_init.pack(side="left", padx=5); self.widgets["wavelet_init"] = e_init

        # ── Redux-only losses (colonne droite : f_loss_right) ──

        # HSLuv
        f_hsluv = ctk.CTkFrame(f_loss_right, fg_color="transparent"); f_hsluv.pack(fill="x", pady=2, padx=6)
        chk_hsluv = ctk.CTkCheckBox(f_hsluv, text="HSLuv", width=120, onvalue="true", offvalue="false"); chk_hsluv.pack(side="left"); self.widgets["loss_hsluv"] = chk_hsluv; ToolTip(chk_hsluv, get_tooltip("loss_hsluv", _t("HSLuv Loss — espace couleur perceptuellement uniforme.", "HSLuv Loss — perceptually uniform color space.")))
        ctk.CTkLabel(f_hsluv, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_hsluv"] = ctk.CTkEntry(f_hsluv, width=42); self.widgets["weight_loss_hsluv"].insert(0, "1.0"); self.widgets["weight_loss_hsluv"].pack(side="left", padx=3)
        ctk.CTkLabel(f_hsluv, text="H:", width=16).pack(side="left"); self.widgets["hsluv_hue_weight"] = ctk.CTkEntry(f_hsluv, width=38); self.widgets["hsluv_hue_weight"].insert(0, "0.33"); self.widgets["hsluv_hue_weight"].pack(side="left", padx=2); ToolTip(self.widgets["hsluv_hue_weight"], _t("Poids Teinte (Hue).\nInfluence la correction de teinte.\nDéfaut : 0.33  |  Augmenter si la teinte dévie.", "Hue weight.\nInfluences hue correction.\nDefault: 0.33  |  Increase if hue drifts."))
        ctk.CTkLabel(f_hsluv, text="S:", width=16).pack(side="left"); self.widgets["hsluv_sat_weight"] = ctk.CTkEntry(f_hsluv, width=38); self.widgets["hsluv_sat_weight"].insert(0, "0.33"); self.widgets["hsluv_sat_weight"].pack(side="left", padx=2); ToolTip(self.widgets["hsluv_sat_weight"], _t("Poids Saturation.\nInfluence la vivacité des couleurs.\nDéfaut : 0.33  |  Augmenter si les couleurs semblent ternes.", "Saturation weight.\nInfluences color vividness.\nDefault: 0.33  |  Increase if colors look dull."))
        ctk.CTkLabel(f_hsluv, text="L:", width=16).pack(side="left"); self.widgets["hsluv_lum_weight"] = ctk.CTkEntry(f_hsluv, width=38); self.widgets["hsluv_lum_weight"].insert(0, "0.33"); self.widgets["hsluv_lum_weight"].pack(side="left", padx=2); ToolTip(self.widgets["hsluv_lum_weight"], _t("Poids Luminosité (Lightness).\nInfluence la correction de luminosité perçue.\nDéfaut : 0.33  |  Augmenter si la luminosité dévie.", "Lightness weight.\nInfluences perceived brightness correction.\nDefault: 0.33  |  Increase if brightness drifts."))

        # Cosim
        f_cosim = ctk.CTkFrame(f_loss_right, fg_color="transparent"); f_cosim.pack(fill="x", pady=2, padx=6)
        chk_cosim = ctk.CTkCheckBox(f_cosim, text="Cosim", width=120, onvalue="true", offvalue="false"); chk_cosim.pack(side="left"); self.widgets["loss_cosim"] = chk_cosim; ToolTip(chk_cosim, get_tooltip("loss_cosim", "Cosine Similarity Loss."))
        ctk.CTkLabel(f_cosim, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_cosim"] = ctk.CTkEntry(f_cosim, width=42); self.widgets["weight_loss_cosim"].insert(0, "1.0"); self.widgets["weight_loss_cosim"].pack(side="left", padx=3)
        ctk.CTkLabel(f_cosim, text="λ:", width=18).pack(side="left"); self.widgets["cosim_lambda"] = ctk.CTkEntry(f_cosim, width=42); self.widgets["cosim_lambda"].insert(0, "5"); self.widgets["cosim_lambda"].pack(side="left", padx=3); ToolTip(self.widgets["cosim_lambda"], _t("Lambda cosim.\nFacteur d'échelle de la pénalité angulaire.\nDéfaut : 5  |  Augmenter → correction couleur plus agressive.", "Lambda cosim.\nAngular penalty scale factor.\nDefault: 5  |  Increase → more aggressive color correction."))

        # Color
        f_color = ctk.CTkFrame(f_loss_right, fg_color="transparent"); f_color.pack(fill="x", pady=2, padx=6)
        chk_color = ctk.CTkCheckBox(f_color, text="Color", width=120, onvalue="true", offvalue="false"); chk_color.pack(side="left"); self.widgets["loss_color"] = chk_color; ToolTip(chk_color, get_tooltip("loss_color", _t("Color Loss — fidélité chromatique.", "Color Loss — chromatic fidelity.")))
        ctk.CTkLabel(f_color, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_color"] = ctk.CTkEntry(f_color, width=42); self.widgets["weight_loss_color"].insert(0, "1.0"); self.widgets["weight_loss_color"].pack(side="left", padx=3)
        self.widgets["color_criterion"] = ctk.CTkOptionMenu(f_color, values=["l1", "l2", "huber", "charbonnier"], width=105); self.widgets["color_criterion"].pack(side="left", padx=3); self.widgets["color_criterion"].set("l1"); ToolTip(self.widgets["color_criterion"], _t("Critère Color Loss.\nl1 : Erreur absolue — standard, recommandé.\nl2 : Quadratique — plus lissé.\nhuber : Hybride — robuste aux outliers.\ncharbonnier : L1 lissé — stable, recommandé Redux.", "Color Loss criterion.\nl1: Absolute error — standard, recommended.\nl2: Quadratic — smoother.\nhuber: Hybrid — robust to outliers.\ncharbonnier: Smooth L1 — stable, recommended Redux."))

        # Gradient Variance
        f_gv = ctk.CTkFrame(f_loss_right, fg_color="transparent"); f_gv.pack(fill="x", pady=2, padx=6)
        chk_gv = ctk.CTkCheckBox(f_gv, text="Grad Variance", width=120, onvalue="true", offvalue="false"); chk_gv.pack(side="left"); self.widgets["loss_gv"] = chk_gv; ToolTip(chk_gv, get_tooltip("loss_gv", "Gradient Variance Loss."))
        ctk.CTkLabel(f_gv, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_gv"] = ctk.CTkEntry(f_gv, width=42); self.widgets["weight_loss_gv"].insert(0, "1.0"); self.widgets["weight_loss_gv"].pack(side="left", padx=3)
        ctk.CTkLabel(f_gv, text="P:", width=16).pack(side="left"); self.widgets["gv_patch_size"] = ctk.CTkEntry(f_gv, width=38); self.widgets["gv_patch_size"].insert(0, "16"); self.widgets["gv_patch_size"].pack(side="left", padx=2); ToolTip(self.widgets["gv_patch_size"], _t("Patch size (pixels).\nTaille des patches pour le calcul de variance de gradient.\nDéfaut : 16  |  Augmenter → contexte plus large.", "Patch size (pixels).\nPatch size for gradient variance computation.\nDefault: 16  |  Increase → larger context."))
        self.widgets["gv_criterion"] = ctk.CTkOptionMenu(f_gv, values=["charbonnier", "l1", "l2", "huber"], width=105); self.widgets["gv_criterion"].pack(side="left", padx=3); self.widgets["gv_criterion"].set("charbonnier"); ToolTip(self.widgets["gv_criterion"], _t("Critère Gradient Variance.\ncharbonnier : Recommandé — L1 lissé, robuste.\nl1/l2/huber : alternatives standard.", "Gradient Variance criterion.\ncharbonnier: Recommended — smooth L1, robust.\nl1/l2/huber: standard alternatives."))

        # Luma
        f_luma = ctk.CTkFrame(f_loss_right, fg_color="transparent"); f_luma.pack(fill="x", pady=2, padx=6)
        chk_luma = ctk.CTkCheckBox(f_luma, text="Luma", width=120, onvalue="true", offvalue="false"); chk_luma.pack(side="left"); self.widgets["loss_luma"] = chk_luma; ToolTip(chk_luma, get_tooltip("loss_luma", "Luma Loss — luminance."))
        ctk.CTkLabel(f_luma, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_luma"] = ctk.CTkEntry(f_luma, width=42); self.widgets["weight_loss_luma"].insert(0, "1.0"); self.widgets["weight_loss_luma"].pack(side="left", padx=3)
        self.widgets["luma_criterion"] = ctk.CTkOptionMenu(f_luma, values=["l1", "l2", "huber", "charbonnier"], width=105); self.widgets["luma_criterion"].pack(side="left", padx=3); self.widgets["luma_criterion"].set("l1"); ToolTip(self.widgets["luma_criterion"], _t("Critère Luma Loss.\nl1 : Recommandé — erreur absolue sur la luminance.\ncharbonnier : Alternative lissée.", "Luma Loss criterion.\nl1: Recommended — absolute error on luminance.\ncharbonnier: Smooth alternative."))

        # Contextual
        f_ctx = ctk.CTkFrame(f_loss_right, fg_color="transparent"); f_ctx.pack(fill="x", pady=2, padx=6)
        chk_ctx = ctk.CTkCheckBox(f_ctx, text="Contextual", width=120, onvalue="true", offvalue="false"); chk_ctx.pack(side="left"); self.widgets["loss_contextual"] = chk_ctx; ToolTip(chk_ctx, get_tooltip("loss_contextual", "Contextual Loss."))
        ctk.CTkLabel(f_ctx, text="W:", width=20).pack(side="left"); self.widgets["weight_loss_contextual"] = ctk.CTkEntry(f_ctx, width=42); self.widgets["weight_loss_contextual"].insert(0, "1.0"); self.widgets["weight_loss_contextual"].pack(side="left", padx=3)
        self.widgets["ctx_distance_type"] = ctk.CTkOptionMenu(f_ctx, values=["cosine", "l2"], width=82); self.widgets["ctx_distance_type"].pack(side="left", padx=3); self.widgets["ctx_distance_type"].set("cosine"); ToolTip(self.widgets["ctx_distance_type"], _t("Métrique de distance entre patches VGG.\ncosine : Angle entre vecteurs — plus robuste aux changements d'échelle. Recommandé.\nl2 : Distance euclidienne — plus sensible à la magnitude.", "Distance metric between VGG patches.\ncosine: Angle between vectors — more robust to scale changes. Recommended.\nl2: Euclidean distance — more sensitive to magnitude."))
        ctk.CTkLabel(f_ctx, text="BW:", width=28).pack(side="left"); self.widgets["ctx_band_width"] = ctk.CTkEntry(f_ctx, width=42); self.widgets["ctx_band_width"].insert(0, "0.5"); self.widgets["ctx_band_width"].pack(side="left", padx=3); ToolTip(self.widgets["ctx_band_width"], _t("Bandwidth (largeur de bande contextuelle).\nContrôle la tolérance aux décalages spatiaux entre patches.\nDéfaut : 0.5  |  Plus haut → plus de tolérance, pénalité plus douce.", "Bandwidth (contextual bandwidth).\nControls tolerance to spatial shifts between patches.\nDefault: 0.5  |  Higher → more tolerance, softer penalty."))
        ctk.CTkLabel(f_loss_right, text="", height=4).pack()  # bottom padding

        self.add_header(f, _t("Monitoring", "Monitoring")); f_mon = ctk.CTkFrame(f, fg_color="transparent"); f_mon.pack(fill="x", pady=5)
        row_tb = ctk.CTkFrame(f_mon, fg_color="transparent"); row_tb.pack(fill="x", pady=2)
        chk_tb = ctk.CTkCheckBox(row_tb, text=_t("Serveur TB", "TB Server"), onvalue="true", offvalue="false", width=100); chk_tb.pack(side="left"); self.widgets["auto_tensorboard"] = chk_tb; ToolTip(chk_tb, get_tooltip("auto_tensorboard"))
        btn_restart = ctk.CTkButton(row_tb, text="♻️", width=30, fg_color="#e67e22", command=self.restart_tb); btn_restart.pack(side="left", padx=5); ToolTip(btn_restart, _t("Relancer TensorBoard", "Restart TensorBoard"))
        lbl_p = ctk.CTkLabel(row_tb, text="Port:", width=30); lbl_p.pack(side="left"); e_tb = ctk.CTkEntry(row_tb, width=60); e_tb.insert(0, "6006"); e_tb.pack(side="left", padx=5); self.widgets["port_tb"] = e_tb
        chk_log = ctk.CTkCheckBox(f_mon, text=_t("Générer Logs", "Generate Logs"), onvalue="true", offvalue="false"); chk_log.pack(anchor="w", pady=2); self.widgets["use_tb_logger"] = chk_log; chk_log.select()
        chk_ngr = ctk.CTkCheckBox(f_mon, text=_t("Tunnel Ngrok", "Ngrok Tunnel"), onvalue="true", offvalue="false"); chk_ngr.pack(anchor="w", pady=2); self.widgets["auto_ngrok"] = chk_ngr; ToolTip(chk_ngr, get_tooltip("auto_ngrok"))
        self.add_header(f, _t("Chargement & Perceptual", "Loading & Perceptual")); f_data = ctk.CTkFrame(f, fg_color="transparent"); f_data.pack(fill="x")
        self.add_label_tip(f_data, "Prefetch Mode :", "prefetch_mode"); self.widgets["prefetch_mode"] = ctk.CTkOptionMenu(f_data, values=["cuda", "cpu", "None"]); self.widgets["prefetch_mode"].pack(side="left", padx=10)
        self.add_label_tip(f_data, "Workers/GPU :", "num_worker"); self.widgets["num_worker_per_gpu"] = ctk.CTkEntry(f_data, width=50); self.widgets["num_worker_per_gpu"].insert(0, "4"); self.widgets["num_worker_per_gpu"].pack(side="left", padx=10)
        return outer

    # --- ACTIONS ---
    # ─── PAGE: VÉRIFICATION AI ─────────────────────────────────
    def create_page_ai_check(self):
        f = ctk.CTkScrollableFrame(self.page_container, fg_color="transparent")
        self.add_header(f, _t("Vérification de Configuration par IA", "AI Configuration Check"))
        ctk.CTkLabel(f, text=_t("Envoyez votre configuration à une IA pour analyse avant l'entraînement.",
                                 "Send your configuration to an AI for analysis before training."),
                     text_color=("gray30", "#AAA")).pack(anchor="w", padx=10, pady=(0, 10))

        # Templates section
        f_tpl = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8)
        f_tpl.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(f_tpl, text=_t("Templates de Configuration", "Configuration Templates"), font=("Roboto", 13, "bold"),
                     text_color="#9b59b6").pack(anchor="w", padx=10, pady=(8, 5))
        ctk.CTkLabel(f_tpl, text=_t("Charge une configuration pre-reglee pour des cas d'usage typiques.",
                                     "Load a pre-configured setup for typical use cases."),
                     text_color=("gray30", "#AAA"), font=("Roboto", 10)).pack(anchor="w", padx=10, pady=(0, 5))

        row_tpl = ctk.CTkFrame(f_tpl, fg_color="transparent")
        row_tpl.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(row_tpl, text=_t("Template :", "Template:"), width=120, anchor="w").pack(side="left")
        self.widgets["template_choice"] = ctk.CTkOptionMenu(
            row_tpl, values=[_t("(aucun)", "(none)")] + list_templates(), width=250
        )
        self.widgets["template_choice"].pack(side="left", padx=5)
        ToolTip(self.widgets["template_choice"],
                _t("Sélectionnez un template pour charger des hyperparamètres pré-réglés.\n"
                   "Templates disponibles :\n"
                   "  Anime 4x PSNR/GAN — anime/illustration\n"
                   "  Photo Réaliste 2x/4x — photos\n"
                   "  Pixel Art 4x — retro games\n"
                   "  Vidéo Compressée 4x — anime DVD/BluRay rip avec banding",
                   "Select a template to load pre-configured hyperparameters.\n"
                   "Available templates:\n"
                   "  Anime 4x PSNR/GAN — anime/illustration\n"
                   "  Realistic Photo 2x/4x — photos\n"
                   "  Pixel Art 4x — retro games\n"
                   "  Compressed Video 4x — anime DVD/BluRay rip with banding"))
        self.widgets["btn_load_template"] = ctk.CTkButton(
            row_tpl, text=_t("Charger Template", "Load Template"), fg_color="#9b59b6", width=150,
            command=self._load_template
        )
        self.widgets["btn_load_template"].pack(side="left", padx=5)

        # Model mapping per provider
        self._ai_models = {
            "OpenRouter (Gratuit)": ["meta-llama/llama-3.3-70b-instruct:free", "nvidia/nemotron-3-super-120b-a12b:free", "z-ai/glm-4.5-air:free", "google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free", "qwen/qwen3-coder:free", "openai/gpt-oss-120b:free"],
            "GitHub Models (Gratuit)": ["gpt-4o", "gpt-4o-mini", "Phi-4", "DeepSeek-R1", "Llama-3.3-70B-Instruct"],
            "Google (Gemini)": ["gemini-3.1-pro-preview", "gemini-3.1-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
            "Anthropic (Claude)": ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            "OpenAI (ChatGPT)": ["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "o3", "o3-mini"],
            "xAI (Grok)": ["grok-4.3", "grok-4.3-fast", "grok-4.3-mini", "grok-3", "grok-3-fast", "grok-3-mini", "grok-3-mini-fast"],
            "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
        }

        # API Selection
        f_api = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8); f_api.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(f_api, text=_t("Configuration API", "API Configuration"), font=("Roboto", 13, "bold"),
                     text_color="#3498db").pack(anchor="w", padx=10, pady=(8, 5))

        row1 = ctk.CTkFrame(f_api, fg_color="transparent"); row1.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row1, text=_t("Fournisseur AI :", "AI Provider:"), width=120, anchor="w").pack(side="left")
        self.widgets["ai_api"] = ctk.CTkOptionMenu(row1, values=list(self._ai_models.keys()), width=200,
                                                     command=self._on_ai_provider_change)
        self.widgets["ai_api"].pack(side="left", padx=5); self.widgets["ai_api"].set("OpenRouter (Gratuit)")

        row2 = ctk.CTkFrame(f_api, fg_color="transparent"); row2.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row2, text=_t("Clé API :", "API Key:"), width=120, anchor="w").pack(side="left")
        self.widgets["ai_api_key"] = ctk.CTkEntry(row2, width=350, show="*"); self.widgets["ai_api_key"].pack(side="left", padx=5)
        # Load saved key for current provider
        provider = "OpenRouter (Gratuit)"
        saved_key = self.settings.get(f"api_key_{provider}", self.settings.get("ai_api_key", ""))
        if saved_key: self.widgets["ai_api_key"].insert(0, saved_key)
        ctk.CTkLabel(row2, text=_t("(depuis Paramètres)", "(from Settings)"), text_color="#888", font=("Roboto", 9)).pack(side="left", padx=5)

        row3 = ctk.CTkFrame(f_api, fg_color="transparent"); row3.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row3, text=_t("Modèle :", "Model:"), width=120, anchor="w").pack(side="left")
        self.widgets["ai_model"] = ctk.CTkOptionMenu(row3, values=self._ai_models["OpenRouter (Gratuit)"], width=280,
                                                       command=lambda x: self._on_ai_model_change(x))
        self.widgets["ai_model"].pack(side="left", padx=5)
        self.widgets["ai_model"].set("deepseek/deepseek-r1:free")

        # Model description display
        self.widgets["ai_model_desc"] = ctk.CTkLabel(
            f_api, text="", text_color="#888", font=("Roboto", 10),
            justify="left", anchor="w", wraplength=600
        )
        self.widgets["ai_model_desc"].pack(anchor="w", padx=15, pady=(2, 8))
        # Initialize with current model description
        try:
            self.widgets["ai_model_desc"].configure(text=get_model_description("deepseek/deepseek-r1:free"))
        except Exception:
            pass

        # Test API connection button + cache controls
        row_test = ctk.CTkFrame(f_api, fg_color="transparent"); row_test.pack(fill="x", padx=10, pady=(5, 8))
        self.widgets["btn_test_api"] = ctk.CTkButton(
            row_test, text=_t("🔌 Tester la Connexion API", "🔌 Test API Connection"), fg_color="#16a085",
            width=200, command=self._test_api_connection
        )
        self.widgets["btn_test_api"].pack(side="left", padx=5)
        ToolTip(self.widgets["btn_test_api"],
                _t("Envoie un message minimal 'OK' pour valider que la clé API et le modèle fonctionnent.\n"
                   "Très rapide et très bon marché (quelques tokens).",
                   "Sends a minimal 'OK' message to validate that the API key and model work.\n"
                   "Very fast and very cheap (a few tokens)."))

        self.widgets["lbl_test_result"] = ctk.CTkLabel(row_test, text="", text_color="#888")
        self.widgets["lbl_test_result"].pack(side="left", padx=10)

        # Cache controls
        row_cache = ctk.CTkFrame(f_api, fg_color="transparent"); row_cache.pack(fill="x", padx=10, pady=(0, 8))
        self.widgets["ai_use_cache"] = ctk.CTkCheckBox(
            row_cache, text=_t("Utiliser le cache (éviter les appels redondants)",
                               "Use cache (avoid redundant API calls)")
        )
        self.widgets["ai_use_cache"].pack(side="left", padx=5)
        self.widgets["ai_use_cache"].select()
        ToolTip(self.widgets["ai_use_cache"],
                _t("Si la même config a déjà été analysée, réutiliser la réponse précédente\n"
                   "au lieu de re-payer un appel API. Cache valide 30 jours.",
                   "If the same config has already been analyzed, reuse the previous response\n"
                   "instead of paying for another API call. Cache valid 30 days."))

        self.widgets["btn_clear_cache"] = ctk.CTkButton(
            row_cache, text=_t("Vider Cache", "Clear Cache"), fg_color="#666", width=100,
            command=self._clear_ai_cache
        )
        self.widgets["btn_clear_cache"].pack(side="right", padx=5)

        ctk.CTkLabel(f_api, text="", height=5).pack()

        # Scope
        f_scope = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8); f_scope.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(f_scope, text=_t("L'IA vérifiera :", "The AI will check:"), font=("Roboto", 13, "bold"),
                     text_color="#e67e22").pack(anchor="w", padx=10, pady=(8, 5))
        for c in [_t("Cohérence architecture / scale / batch / patch",
                      "Architecture / scale / batch / patch consistency"),
                  _t("Compatibilité des losses avec le moteur",
                      "Loss compatibility with the engine"),
                  _t("Recommandations LR / scheduler / optimiseur",
                      "LR / scheduler / optimizer recommendations"),
                  _t("Estimation VRAM et risques de OOM",
                      "VRAM estimation and OOM risks"),
                  _t("Bonnes pratiques PSNR vs GAN",
                      "PSNR vs GAN best practices"),
                  _t("Erreurs courantes et pièges",
                      "Common mistakes and pitfalls")]:
            ctk.CTkLabel(f_scope, text=f"  {c}", text_color=("gray30", "#CCC"), anchor="w").pack(anchor="w", padx=15, pady=1)
        ctk.CTkLabel(f_scope, text="", height=5).pack()

        # Mode
        f_mode = ctk.CTkFrame(f, fg_color="transparent"); f_mode.pack(fill="x", padx=5, pady=5)
        self.widgets["ai_send_mode"] = ctk.CTkOptionMenu(f_mode, values=[_t("Texte résumé (léger)", "Summary text (light)"), _t("Config complète (détaillé)", "Full config (detailed)")], width=250)
        self.widgets["ai_send_mode"].pack(side="left", padx=5)
        self.widgets["ai_send_mode"].set(_t("Config complète (détaillé)", "Full config (detailed)"))

        # Context & Questions
        f_ctx = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8); f_ctx.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(f_ctx, text=_t("Contexte supplémentaire (optionnel)", "Additional context (optional)"), font=("Roboto", 13, "bold"),
                     text_color="#27ae60").pack(anchor="w", padx=10, pady=(8, 3))

        ctx_row1 = ctk.CTkFrame(f_ctx, fg_color="transparent"); ctx_row1.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(ctx_row1, text=_t("Type de dataset :", "Dataset type:"), width=140, anchor="w").pack(side="left")
        self.widgets["ai_ctx_dataset"] = ctk.CTkOptionMenu(ctx_row1, width=200,
            values=["(auto-detect)", "Anime / Illustration", "Photo realiste", "Texte / Document", "Medical / Scientifique", "Jeux video / Pixel art", "Mixte"])
        self.widgets["ai_ctx_dataset"].pack(side="left", padx=5)
        self.widgets["ai_ctx_dataset"].set("(auto-detect)")
        ToolTip(self.widgets["ai_ctx_dataset"], _t("Type de contenu dans votre dataset. Aide l'IA à mieux évaluer la config.",
                                                     "Content type in your dataset. Helps the AI better evaluate the config."))
        ctx_row2 = ctk.CTkFrame(f_ctx, fg_color="transparent"); ctx_row2.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(ctx_row2, text=_t("But du modèle :", "Model goal:"), width=140, anchor="w").pack(side="left")
        self.widgets["ai_ctx_goal"] = ctk.CTkOptionMenu(ctx_row2, width=200,
        values=["(non specifie)", "PSNR (fidelite pixel)", "GAN (qualite percue)", "PSNR puis GAN (pipeline)", "Denoise", "Deblur", "Compression artifact", "Debanding"])
        self.widgets["ai_ctx_goal"].pack(side="left", padx=5)
        self.widgets["ai_ctx_goal"].set("(non specifie)")
        ToolTip(self.widgets["ai_ctx_goal"], _t("Objectif de l'entraînement : fidélité pixel (PSNR) ou qualité perçue (GAN)",
                                                  "Training objective: pixel fidelity (PSNR) or perceived quality (GAN)"))

        ctx_row3 = ctk.CTkFrame(f_ctx, fg_color="transparent"); ctx_row3.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(ctx_row3, text=_t("Temps alloué :", "Time allocated:"), width=140, anchor="w").pack(side="left")
        self.widgets["ai_ctx_time"] = ctk.CTkOptionMenu(ctx_row3, width=200,
            values=["(non specifie)", "< 2 heures", "2-6 heures", "6-24 heures", "1-3 jours", "3-7 jours", "> 1 semaine"])
        self.widgets["ai_ctx_time"].pack(side="left", padx=5)
        self.widgets["ai_ctx_time"].set("(non specifie)")
        ToolTip(self.widgets["ai_ctx_time"], _t("Temps que vous pouvez allouer à l'entraînement. Influence les recommandations d'itérations.",
                                                  "Time you can allocate to training. Influences iteration recommendations."))

        ctx_row4 = ctk.CTkFrame(f_ctx, fg_color="transparent"); ctx_row4.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(ctx_row4, text=_t("Taille dataset :", "Dataset size:"), width=140, anchor="w").pack(side="left")
        self.widgets["ai_ctx_ds_size"] = ctk.CTkEntry(ctx_row4, width=160, placeholder_text="ex: 5000 images, 50Go...")
        self.widgets["ai_ctx_ds_size"].pack(side="left", padx=5)
        ctk.CTkButton(ctx_row4, text="Auto", width=55, height=24, fg_color="#2c6e49",
                      command=self._autodetect_dataset_size).pack(side="left", padx=2)
        ToolTip(self.widgets["ai_ctx_ds_size"], _t("Taille du dataset d'entraînement. Cliquez 'Auto' pour compter les images dans le dossier Train HQ.",
                                                     "Training dataset size. Click 'Auto' to count images in the Train HQ folder."))

        ctk.CTkLabel(f_ctx, text=_t("Questions / notes supplémentaires :", "Questions / additional notes:"), text_color=("gray30", "#AAA"),
                     font=("Roboto", 11)).pack(anchor="w", padx=10, pady=(5, 2))
        self.widgets["ai_ctx_notes"] = ctk.CTkTextbox(f_ctx, height=60, font=("Roboto", 11))
        self.widgets["ai_ctx_notes"].pack(fill="x", padx=10, pady=(0, 8))

        # Auto system info checkbox
        self.widgets["ai_include_sysinfo"] = ctk.CTkCheckBox(f_ctx, text=_t("Inclure infos système (CPU, GPU, RAM, stockage)",
                                                                               "Include system info (CPU, GPU, RAM, storage)"))
        self.widgets["ai_include_sysinfo"].pack(anchor="w", padx=10, pady=(0, 8))
        self.widgets["ai_include_sysinfo"].select()
        ToolTip(self.widgets["ai_include_sysinfo"], _t("Détecte automatiquement CPU, GPU, RAM et espace disque pour des recommandations adaptées à votre matériel.",
                                                         "Automatically detects CPU, GPU, RAM and disk space for recommendations tailored to your hardware."))

        # Launch button
        ctk.CTkButton(f, text=_t("Analyser ma Configuration", "Analyze my Configuration"), fg_color="#8e44ad", height=40,
                      font=("Roboto", 14, "bold"), command=self.check_config_ai).pack(fill="x", padx=20, pady=15)

        # Result area
        self.ai_result_box = ctk.CTkTextbox(f, height=300, wrap="word", state="disabled")
        self.ai_result_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        return f

    # ─── PAGE: PIPELINE PSNR → GAN ──────────────────────────
    def create_page_pipeline(self):
        f = ctk.CTkScrollableFrame(self.page_container, fg_color="transparent")
        self.add_header(f, _t("Pipeline Automatisé PSNR -> GAN", "Automated Pipeline PSNR -> GAN"))

        # Explanation
        f_info = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8); f_info.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(f_info, text=_t("Comment ça marche :", "How it works:"), font=("Roboto", 13, "bold"),
                     text_color="#3498db").pack(anchor="w", padx=10, pady=(8, 3))
        for txt in [
            _t("1. Configurez votre modèle dans les pages Général, Architecture, Datasets, etc.",
               "1. Configure your model in the General, Architecture, Datasets pages, etc."),
            _t("2. Ici, définissez uniquement les SURCHARGES par phase (iters, LR, losses).",
               "2. Here, define only the OVERRIDES per phase (iters, LR, losses)."),
            _t("3. Phase 1 (PSNR) entraîne sans GAN pour la fidélité structurelle.",
               "3. Phase 1 (PSNR) trains without GAN for structural fidelity."),
            _t("4. Phase 2 (GAN) charge le pretrain PSNR et active le discriminateur.",
               "4. Phase 2 (GAN) loads the PSNR pretrain and enables the discriminator."),
            _t("5. Vous pouvez aussi charger des fichiers .toml/.yml existants pour chaque phase.",
               "5. You can also load existing .toml/.yml files for each phase.")
        ]:
            ctk.CTkLabel(f_info, text=f"  {txt}", text_color=("gray30", "#AAA"), anchor="w", wraplength=700).pack(anchor="w", padx=15, pady=1)
        ctk.CTkLabel(f_info, text="", height=3).pack()

        # Phase 1: PSNR
        f_p1 = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a2e1a"), corner_radius=8); f_p1.pack(fill="x", padx=5, pady=5)
        h1 = ctk.CTkFrame(f_p1, fg_color="transparent"); h1.pack(fill="x", padx=10, pady=(8, 3))
        ctk.CTkLabel(h1, text=_t("Phase 1 -- PSNR (Fidélité)", "Phase 1 -- PSNR (Fidelity)"), font=("Roboto", 14, "bold"),
                     text_color="#2ecc71").pack(side="left")
        ctk.CTkButton(h1, text=_t("Charger .toml/.yml", "Load .toml/.yml"), fg_color="#444", width=130, height=25,
                      command=lambda: self._load_phase_config("psnr")).pack(side="right")

        for key, label, default, tip in [
            ("pipe_psnr_iter", "Iterations", "150000", "Iterations PSNR. Surcharge total_iter."),
            ("pipe_psnr_lr", "Learning Rate", "5e-4", 'Learning Rate phase PSNR\n\n  1e-3  = ⚡ Tres rapide — instable, seulement SOAP_SF\n  5e-4  = 🔥 Rapide — bon depart PSNR\n  3e-4  = ✅ Standard (recommande pour debut)\n  2e-4  = 🟢 Modere — stable, Adam/AdamW\n  1e-4  = 🐢 Lent — convergence sure\n  5e-5  = 🐌 Tres lent — fine-tuning PSNR\n  1e-5  = 🧊 Ultra lent — micro-ajustement\n  1e-6  = 🔬 Negligeable — derniere passe\n\nRegle : commencer a 3e-4 ou 5e-4, baisser si loss oscille.'),
            ("pipe_psnr_batch", "Batch Size", "", "Vide = utilise la valeur des pages principales."),
            ("pipe_psnr_patch", "Patch Size (GT)", "", "Vide = utilise la valeur des pages principales."),
        ]:
            row = ctk.CTkFrame(f_p1, fg_color="transparent"); row.pack(fill="x", padx=10, pady=2)
            lbl = ctk.CTkLabel(row, text=f"{label} :", width=140, anchor="w"); lbl.pack(side="left")
            e = ctk.CTkEntry(row, width=100, placeholder_text=_t("(depuis config)", "(from config)")); 
            if default: e.insert(0, default)
            e.pack(side="left", padx=5)
            self.widgets[key] = e; ToolTip(lbl, tip)

        f_p1_loss = ctk.CTkFrame(f_p1, fg_color="transparent"); f_p1_loss.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(f_p1_loss, text=_t("Losses :", "Losses:"), width=140, anchor="w").pack(side="left")
        self.widgets["pipe_psnr_pixel"] = ctk.CTkCheckBox(f_p1_loss, text="Pixel (L1)", width=85)
        self.widgets["pipe_psnr_pixel"].pack(side="left"); self.widgets["pipe_psnr_pixel"].select()
        self.widgets["pipe_psnr_percep"] = ctk.CTkCheckBox(f_p1_loss, text="Perceptual", width=85)
        self.widgets["pipe_psnr_percep"].pack(side="left", padx=5)
        self.widgets["pipe_psnr_mssim"] = ctk.CTkCheckBox(f_p1_loss, text="MS-SSIM", width=75)
        self.widgets["pipe_psnr_mssim"].pack(side="left", padx=5)
        self.widgets["pipe_psnr_ldl"] = ctk.CTkCheckBox(f_p1_loss, text="LDL", width=50)
        self.widgets["pipe_psnr_ldl"].pack(side="left", padx=5)
        self.widgets["pipe_psnr_ff"] = ctk.CTkCheckBox(f_p1_loss, text="Focal Freq", width=80)
        self.widgets["pipe_psnr_ff"].pack(side="left", padx=5)

        f_p1_opt = ctk.CTkFrame(f_p1, fg_color="transparent"); f_p1_opt.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(f_p1_opt, text="Optimizer :", width=140, anchor="w").pack(side="left")
        _dc = _t("(depuis config)", "(from config)")
        self.widgets["pipe_psnr_optim"] = ctk.CTkOptionMenu(f_p1_opt, values=[_dc, "adan", "adamw", "adam"], width=140)
        self.widgets["pipe_psnr_optim"].pack(side="left"); self.widgets["pipe_psnr_optim"].set(_dc)
        ctk.CTkLabel(f_p1_opt, text="Scheduler :", width=80).pack(side="left", padx=(15, 0))
        self.widgets["pipe_psnr_sched"] = ctk.CTkOptionMenu(f_p1_opt, values=[_dc, "CosineAnnealing", "MultiStepLR"], width=150)
        self.widgets["pipe_psnr_sched"].pack(side="left"); self.widgets["pipe_psnr_sched"].set(_dc)
        ctk.CTkLabel(f_p1, text="", height=3).pack()

        # Phase 2: GAN
        f_p2 = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#2e1a1a"), corner_radius=8); f_p2.pack(fill="x", padx=5, pady=5)
        h2 = ctk.CTkFrame(f_p2, fg_color="transparent"); h2.pack(fill="x", padx=10, pady=(8, 3))
        ctk.CTkLabel(h2, text=_t("Phase 2 -- GAN (Textures Réalistes)", "Phase 2 -- GAN (Realistic Textures)"), font=("Roboto", 14, "bold"),
                     text_color="#e74c3c").pack(side="left")
        ctk.CTkButton(h2, text=_t("Charger .toml/.yml", "Load .toml/.yml"), fg_color="#444", width=130, height=25,
                      command=lambda: self._load_phase_config("gan")).pack(side="right")

        for key, label, default, tip in [
            ("pipe_gan_iter", "Iterations", "100000", "Iterations GAN. 50K-200K typique."),
            ("pipe_gan_lr", "Learning Rate", "1e-4", 'Learning Rate phase GAN (plus bas que PSNR)\n\n  5e-4  = ⚡ Tres rapide — instable, deconseille\n  2e-4  = 🔥 Rapide — D risque de dominer\n  1e-4  = ✅ Standard GAN (recommande)\n  5e-5  = 🟢 Conservateur — equilibre G/D\n  2e-5  = 🐢 Lent — fine-tuning GAN\n  1e-5  = 🐌 Tres lent — affinage final\n  5e-6  = 🧊 Ultra lent — micro-corrections\n  1e-6  = 🔬 Negligeable — stabilisation finale\n\nRegle : 2x a 5x plus bas que la phase PSNR'),
            ("pipe_gan_batch", "Batch Size", "", "Vide = meme que Phase 1."),
            ("pipe_gan_weight", "Poids GAN", "0.1", "Poids du loss GAN. 0.05-0.3."),
            ("pipe_gan_percep_w", "Poids Perceptual", "1.0", "Poids perceptual en GAN."),
            ("pipe_gan_pixel_w", "Poids Pixel", "0.01", "Poids pixel en GAN (tres faible)."),
        ]:
            row = ctk.CTkFrame(f_p2, fg_color="transparent"); row.pack(fill="x", padx=10, pady=2)
            lbl = ctk.CTkLabel(row, text=f"{label} :", width=140, anchor="w"); lbl.pack(side="left")
            e = ctk.CTkEntry(row, width=100, placeholder_text=_t("(depuis config)", "(from config)"))
            if default: e.insert(0, default)
            e.pack(side="left", padx=5)
            self.widgets[key] = e; ToolTip(lbl, tip)

        f_p2_disc = ctk.CTkFrame(f_p2, fg_color="transparent"); f_p2_disc.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(f_p2_disc, text=_t("Discriminateur :", "Discriminator:"), width=140, anchor="w").pack(side="left")
        _dc2 = _t("(depuis config)", "(from config)")
        self.widgets["pipe_disc"] = ctk.CTkOptionMenu(f_p2_disc, values=[_dc2, "UNet", "DUNet", "PatchGAN", "MetaGAN", "EA2-FPN"], width=140)
        self.widgets["pipe_disc"].pack(side="left"); self.widgets["pipe_disc"].set(_dc2)
        ctk.CTkLabel(f_p2_disc, text="GAN Type :", width=80).pack(side="left", padx=(15, 0))
        self.widgets["pipe_gan_type"] = ctk.CTkOptionMenu(f_p2_disc, values=[_dc2, "bce", "mse", "huber"], width=120)
        self.widgets["pipe_gan_type"].pack(side="left"); self.widgets["pipe_gan_type"].set(_dc2)

        f_p2_extra = ctk.CTkFrame(f_p2, fg_color="transparent"); f_p2_extra.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(f_p2_extra, text=_t("Losses extra :", "Extra losses:"), width=140, anchor="w").pack(side="left")
        self.widgets["pipe_gan_pixel"] = ctk.CTkCheckBox(f_p2_extra, text="Pixel", width=55)
        self.widgets["pipe_gan_pixel"].pack(side="left"); self.widgets["pipe_gan_pixel"].select()
        self.widgets["pipe_gan_percep"] = ctk.CTkCheckBox(f_p2_extra, text="Perceptual", width=85)
        self.widgets["pipe_gan_percep"].pack(side="left"); self.widgets["pipe_gan_percep"].select()
        self.widgets["pipe_gan_ldl"] = ctk.CTkCheckBox(f_p2_extra, text="LDL", width=50)
        self.widgets["pipe_gan_ldl"].pack(side="left", padx=3)
        self.widgets["pipe_gan_dists"] = ctk.CTkCheckBox(f_p2_extra, text="DISTS", width=55)
        self.widgets["pipe_gan_dists"].pack(side="left", padx=3)
        self.widgets["pipe_gan_ff"] = ctk.CTkCheckBox(f_p2_extra, text="Focal Freq", width=80)
        self.widgets["pipe_gan_ff"].pack(side="left", padx=3)
        self.widgets["pipe_gan_mssim"] = ctk.CTkCheckBox(f_p2_extra, text="MS-SSIM", width=70)
        self.widgets["pipe_gan_mssim"].pack(side="left", padx=3)
        ctk.CTkLabel(f_p2, text="", height=3).pack()

        # Current config summary (auto-populated from main pages)
        f_sum = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8); f_sum.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(f_sum, text=_t("Config actuelle (depuis les pages principales) :", "Current config (from main pages):"), font=("Roboto", 12, "bold"),
                     text_color="#f39c12").pack(anchor="w", padx=10, pady=(8, 3))
        self.lbl_pipe_summary = ctk.CTkLabel(f_sum, text=_t("Cliquez 'Actualiser' pour voir le résumé.", "Click 'Refresh' to see the summary."), text_color="#888",
                                              justify="left", anchor="w", wraplength=700)
        self.lbl_pipe_summary.pack(anchor="w", padx=15, pady=3)
        ctk.CTkButton(f_sum, text=_t("Actualiser le résumé", "Refresh summary"), fg_color="#555", width=160, height=25,
                      command=self._refresh_pipeline_summary).pack(anchor="w", padx=10, pady=(3, 8))

        # Options
        f_adv = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8); f_adv.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(f_adv, text="Options", font=("Roboto", 13, "bold"),
                     text_color="#f39c12").pack(anchor="w", padx=10, pady=(8, 5))
        self.widgets["pipe_use_amp"] = ctk.CTkCheckBox(f_adv, text=_t("AMP (FP16) pour les deux phases", "AMP (FP16) for both phases"))
        self.widgets["pipe_use_amp"].pack(anchor="w", padx=15, pady=2); self.widgets["pipe_use_amp"].select()
        self.widgets["pipe_auto_pretrain"] = ctk.CTkCheckBox(f_adv, text=_t("Charger auto le meilleur pretrain PSNR pour la Phase 2",
                                                                              "Auto-load best PSNR pretrain for Phase 2"))
        self.widgets["pipe_auto_pretrain"].pack(anchor="w", padx=15, pady=2); self.widgets["pipe_auto_pretrain"].select()
        self.widgets["pipe_save_best"] = ctk.CTkCheckBox(f_adv, text=_t("Sauvegarder le meilleur modèle (validation PSNR)",
                                                                          "Save best model (PSNR validation)"))
        self.widgets["pipe_save_best"].pack(anchor="w", padx=15, pady=2); self.widgets["pipe_save_best"].select()
        ctk.CTkLabel(f_adv, text="", height=3).pack()

        # Buttons
        f_btn = ctk.CTkFrame(f, fg_color="transparent"); f_btn.pack(fill="x", padx=5, pady=10)
        ctk.CTkButton(f_btn, text=_t("Test Rapide (2K + 2K iters)", "Quick Test (2K + 2K iters)"), fg_color="#e67e22", height=40,
                      width=250, font=("Roboto", 13), command=self.pipeline_test).pack(side="left", padx=10)
        ctk.CTkButton(f_btn, text=_t("Lancer Pipeline Complet", "Launch Full Pipeline"), fg_color="#27ae60", height=40,
                      width=250, font=("Roboto", 13, "bold"), command=self.pipeline_full).pack(side="left", padx=10)

        return f

    def _load_phase_config(self, phase: str):
        """Load an existing config file to populate a pipeline phase."""
        p = filedialog.askopenfilename(
            title=f"Charger config {phase.upper()}",
            filetypes=[("Config files", "*.toml *.yml *.yaml"), ("All", "*.*")]
        )
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as cf:
                content = cf.read()
            data = {}
            if p.endswith(".toml"):
                # Try multiple TOML parsers - NeoSR uses non-standard arrays
                try:
                    import tomllib
                    data = tomllib.loads(content)
                except Exception:
                    try:
                        _ensure_toml(); data = toml.loads(content)
                    except Exception:
                        # Fallback: extract key values with regex
                        import re
                        for key in ["total_iter", "lr", "batch_size", "patch_size", "name"]:
                            m = re.search(rf'{key}\s*=\s*["\']?([^"\'\n,\]]+)', content)
                            if m:
                                data[key] = m.group(1).strip()
            else:
                _ensure_yaml(); data = yaml.safe_load(content) or {}

            # Flatten nested configs (train.total_iter, etc.)
            flat = {}
            flat.update(data)
            for section in ["train", "network_g", "dataset", "datasets"]:
                if section in data and isinstance(data[section], dict):
                    flat.update(data[section])
                    # Also check sub-dicts like datasets.train
                    for k, v in data[section].items():
                        if isinstance(v, dict):
                            flat.update(v)

            prefix = f"pipe_{phase}_"
            mapping = {"iter": "total_iter", "lr": "lr", "batch": "batch_size", "patch": "gt_size"}
            filled = 0
            for short, full in mapping.items():
                key = prefix + short
                val = str(flat.get(full, flat.get(full.replace("gt_size", "patch_size"), "")))
                if key in self.widgets and val and val != "":
                    self.widgets[key].delete(0, "end")
                    self.widgets[key].insert(0, val)
                    filled += 1
            messagebox.showinfo("Pipeline", _t(f"Config {phase.upper()} chargée depuis :\n{os.path.basename(p)}\n({filled} champs remplis)",
                                               f"Config {phase.upper()} loaded from:\n{os.path.basename(p)}\n({filled} fields filled)"))
        except Exception as e:
            messagebox.showerror(_t("Erreur", "Error"), _t(f"Impossible de charger : {e}", f"Unable to load: {e}"))

    def _refresh_pipeline_summary(self):
        """Refresh the pipeline summary from main config pages."""
        engine = self.widgets.get("engine", ctk.CTkEntry(self))
        arch = self.widgets.get("arch", ctk.CTkEntry(self))
        scale = self.widgets.get("scale", ctk.CTkEntry(self))
        batch = self.widgets.get("batch_size", ctk.CTkEntry(self))
        patch = self.widgets.get("patch_size", ctk.CTkEntry(self))
        optim = self.widgets.get("optim_g", ctk.CTkEntry(self))
        ds_gt = self.widgets.get("dataroot_gt", ctk.CTkEntry(self))

        try:
            summary = _t(
                f"Moteur: {engine.get()}  |  Architecture: {arch.get()}  |  Scale: {scale.get()}x\n"
                f"Batch: {batch.get()}  |  Patch GT: {patch.get()}  |  Optimizer: {optim.get()}\n"
                f"Dataset GT: {ds_gt.get()}",
                f"Engine: {engine.get()}  |  Architecture: {arch.get()}  |  Scale: {scale.get()}x\n"
                f"Batch: {batch.get()}  |  Patch GT: {patch.get()}  |  Optimizer: {optim.get()}\n"
                f"Dataset GT: {ds_gt.get()}"
            )
            self.lbl_pipe_summary.configure(text=summary, text_color=("gray30", "#CCC"))
        except Exception:
            self.lbl_pipe_summary.configure(text=_t("Erreur lecture config.", "Error reading config."), text_color="#e74c3c")


    # ─── CHECK CONFIG VIA AI ─────────────────────────────────
    def _on_ai_provider_change(self, choice):
        """Update model list and load saved API key when provider changes."""
        models = self._ai_models.get(choice, [])
        if models:
            self.widgets["ai_model"].configure(values=models)
            # Use recommended default for this provider
            default = get_provider_default_model(choice)
            chosen = default if default in models else models[0]
            self.widgets["ai_model"].set(chosen)
            # Update description
            self._on_ai_model_change(chosen)
        # Load saved key for this provider
        saved_key = self.settings.get(f"api_key_{choice}", "")
        self.widgets["ai_api_key"].delete(0, "end")
        if saved_key:
            self.widgets["ai_api_key"].insert(0, saved_key)

    def _on_ai_model_change(self, model_id):
        """Update the model description when the model selection changes."""
        try:
            desc = get_model_description(model_id)
            if "ai_model_desc" in self.widgets:
                self.widgets["ai_model_desc"].configure(text=desc)
        except Exception:
            pass

    def _test_api_connection(self):
        """Test the API connection with current credentials."""
        provider = self.widgets["ai_api"].get()
        api_key = self.widgets["ai_api_key"].get()
        model = self.widgets["ai_model"].get()

        if not api_key:
            self.widgets["lbl_test_result"].configure(
                text=_t("❌ Clé API vide", "❌ Empty API key"), text_color="#e74c3c"
            )
            return

        self.widgets["lbl_test_result"].configure(text=_t("⏳ Test en cours...", "⏳ Testing..."), text_color="#f39c12")
        self.widgets["btn_test_api"].configure(state="disabled")
        self.update_idletasks()

        # Run test in a thread to avoid blocking UI
        import threading
        def worker():
            try:
                ok, msg = test_api_connection(provider, api_key, model)
                color = "#2ecc71" if ok else "#e74c3c"
                icon = "✅" if ok else "❌"
                # Truncate long messages
                if len(msg) > 100:
                    msg = msg[:100] + "..."
                self.after(0, lambda: self.widgets["lbl_test_result"].configure(
                    text=f"{icon} {msg}", text_color=color
                ))
            except Exception as e:
                self.after(0, lambda: self.widgets["lbl_test_result"].configure(
                    text=f"❌ Erreur: {e}", text_color="#e74c3c"
                ))
            finally:
                self.after(0, lambda: self.widgets["btn_test_api"].configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _clear_ai_cache(self):
        """Clear the AI response cache."""
        from tkinter import messagebox
        stats = cache_stats()
        if stats["count"] == 0:
            messagebox.showinfo("Cache", _t("Le cache est déjà vide.", "The cache is already empty."))
            return
        if messagebox.askyesno(_t("Vider le cache", "Clear cache"),
                                _t(f"Supprimer {stats['count']} réponses cachées "
                                   f"({stats['size_kb']:.1f} KB) ?",
                                   f"Delete {stats['count']} cached responses "
                                   f"({stats['size_kb']:.1f} KB)?")):
            count = clear_cache()
            messagebox.showinfo("Cache", _t(f"{count} réponses supprimées.", f"{count} responses deleted."))

    def _load_template(self):
        """Load a configuration template into the UI."""
        from tkinter import messagebox
        choice = self.widgets["template_choice"].get()
        if not choice or choice in ("(aucun)", "(none)"):
            messagebox.showinfo("Template", _t("Sélectionnez un template.", "Select a template."))
            return

        tpl = get_template(choice)
        if not tpl:
            messagebox.showerror("Template", _t(f"Template '{choice}' non trouvé.", f"Template '{choice}' not found."))
            return

        # Confirm before overwriting
        if not messagebox.askyesno(_t("Charger Template", "Load Template"),
                                    _t(f"Charger '{choice}' ?\n\n"
                                       f"{tpl.get('_description', '')}\n\n"
                                       f"Les valeurs actuelles seront écrasées.",
                                       f"Load '{choice}'?\n\n"
                                       f"{tpl.get('_description', '')}\n\n"
                                       f"Current values will be overwritten.")):
            return

        # Apply each value to the widgets
        applied = 0
        skipped = []
        for key, val in tpl.items():
            if key.startswith("_"):
                continue  # Skip metadata
            if key not in self.widgets:
                skipped.append(key)
                continue
            try:
                w = self.widgets[key]
                if isinstance(w, ctk.CTkEntry):
                    w.delete(0, "end")
                    w.insert(0, str(val))
                    applied += 1
                elif isinstance(w, ctk.CTkOptionMenu):
                    w.set(str(val))
                    applied += 1
                elif isinstance(w, ctk.CTkCheckBox):
                    if val:
                        w.select()
                    else:
                        w.deselect()
                    applied += 1
                elif isinstance(w, ctk.CTkSlider):
                    w.set(float(val))
                    applied += 1
            except Exception:
                skipped.append(key)
        msg = _t(f"Template '{choice}' chargé.\n  Appliqué : {applied} valeurs",
                 f"Template '{choice}' loaded.\n  Applied: {applied} values")
        if skipped:
            msg += _t(f"\n  Ignoré : {len(skipped)} (clés inconnues)", f"\n  Skipped: {len(skipped)} (unknown keys)")
        messagebox.showinfo("Template", msg)

    def _collect_system_info(self) -> str:
        """Collect system hardware info for AI context."""
        parts = []
        import platform as _pf
        parts.append(f"OS: {_pf.system()} {_pf.release()}")
        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
            parts.append(f"RAM: {ram_gb}GB")
            # Check if SSD
            import shutil
            disk = shutil.disk_usage(os.getcwd())
            parts.append(f"Disque libre: {round(disk.free / (1024**3), 1)}GB")
        except ImportError:
            pass
        try:
            import subprocess as _sp
            r = _sp.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                for line in r.stdout.strip().split("\n"):
                    parts.append(f"GPU: {line.strip()}")
        except Exception:
            pass
        try:
            cpu_count = os.cpu_count()
            parts.append(f"CPU cores: {cpu_count}")
        except Exception:
            pass
        return ", ".join(parts) if parts else ""

    def _autodetect_dataset_size(self):
        """Count images in the Train HQ folder and fill the dataset size entry."""
        gt_widget = self.widgets.get("dataroot_gt")
        folder = gt_widget.get().strip() if gt_widget else ""
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Auto-detect", _t("Configurez d'abord un dossier Train HQ (GT) valide dans l'onglet Datasets.",
                                                   "First configure a valid Train HQ (GT) folder in the Datasets tab."))
            return
        img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        try:
            count = sum(
                1 for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in img_exts
            )
            size_entry = self.widgets.get("ai_ctx_ds_size")
            if size_entry:
                size_entry.delete(0, "end")
                size_entry.insert(0, f"{count} images")
        except Exception as e:
            messagebox.showerror("Auto-detect", _t(f"Erreur : {e}", f"Error: {e}"))

    def check_config_ai(self):
        """Send current config to AI API for review."""
        # Save API key
        api_key = self.widgets["ai_api_key"].get().strip()
        if not api_key:
            messagebox.showerror(_t("Erreur", "Error"), _t("Veuillez entrer votre clé API.\n\nVous pouvez sauvegarder vos clés dans\nParamètres > Clés API.",
                                                            "Please enter your API key.\n\nYou can save your keys in\nSettings > API Keys."))
            return
        # Save key for this provider
        api_choice = self.widgets["ai_api"].get()
        self.settings.set(f"api_key_{api_choice}", api_key)

        # Collect all config data
        data = {}
        for k, w in self.widgets.items():
            try: data[k] = w.get()
            except Exception: pass

        # Convert disc display name
        if "net_d_type" in data:
            data["net_d_type"] = self._disc_from_display(data["net_d_type"])

        engine = data.get("engine", "NeoSR")
        arch = data.get("arch", "omnisr")

        # Format config as readable text
        config_text = f"Moteur: {engine}\nArchitecture: {arch}\nScale: {data.get('scale', '4')}\n"
        config_text += f"Batch Size: {data.get('batch_size', '4')}\nPatch Size: {data.get('patch_size', '64')}\n"
        config_text += f"Total Iter: {data.get('total_iter', '150000')}\n"
        config_text += f"Optimizer: {data.get('optim_g', 'AdamW')}\nLR: {data.get('lr', '5e-5')}\n"
        config_text += f"Scheduler: {data.get('scheduler', 'MultiStepLR')}\n"
        config_text += f"AMP: {data.get('use_amp', 'false')}\nGAN: {data.get('use_gan', 'false')}\n"
        if str(data.get("use_gan", "false")).lower() == "true":
            config_text += f"Disc Type: {data.get('net_d_type', 'unet')}\n"
            config_text += f"GAN Weight: {data.get('dyn_gan_weight', '0.05')}\n"
            config_text += f"GAN Type: {data.get('gan_type', 'bce')}\n"

        # Active losses
        losses = []
        for key in data:
            if key.startswith("loss_") and str(data[key]).lower() == "true":
                w_key = f"weight_{key}"
                w_val = data.get(w_key, "1.0")
                losses.append(f"{key}={w_val}")
        if losses:
            config_text += f"Losses: {', '.join(losses)}\n"

        # Add context info
        ctx_parts = []
        ds_type = self.widgets.get("ai_ctx_dataset")
        if ds_type and ds_type.get() != "(auto-detect)":
            ctx_parts.append(f"Type de dataset: {ds_type.get()}")
        goal = self.widgets.get("ai_ctx_goal")
        if goal and goal.get() != "(non specifie)":
            ctx_parts.append(f"But: {goal.get()}")
        time_alloc = self.widgets.get("ai_ctx_time")
        if time_alloc and time_alloc.get() != "(non specifie)":
            ctx_parts.append(f"Temps alloue: {time_alloc.get()}")
        ds_size = self.widgets.get("ai_ctx_ds_size")
        if ds_size and ds_size.get().strip():
            ctx_parts.append(f"Taille dataset: {ds_size.get().strip()}")
        notes_w = self.widgets.get("ai_ctx_notes")
        if notes_w:
            notes = notes_w.get("1.0", "end").strip()
            if notes:
                ctx_parts.append(f"Notes utilisateur: {notes}")

        # System info
        sysinfo_w = self.widgets.get("ai_include_sysinfo")
        if sysinfo_w and sysinfo_w.get():
            sys_info = self._collect_system_info()
            if sys_info:
                ctx_parts.append(f"Systeme: {sys_info}")

        context_block = ""
        if ctx_parts:
            context_block = "\n\nContexte supplementaire:\n" + "\n".join(f"- {c}" for c in ctx_parts)

        prompt = (
            f"Tu es un expert en Super-Resolution et en entrainement de modeles IA. "
            f"Voici ma configuration {engine} avec le modele {arch}. "
            f"Analyse-la et donne-moi des recommandations concretes.{context_block}\n\n"
            f"```\n{config_text}```"
        )

        api_choice = self.widgets["ai_api"].get()
        model_id = self.widgets["ai_model"].get()

        # Check cache first
        use_cache = self.widgets.get("ai_use_cache")
        if use_cache and use_cache.get():
            cached = get_cached_response(api_choice, model_id, prompt)
            if cached:
                self._show_ai_result(f"[CACHE HIT — pas d'appel API]\n\n{cached}")
                return

        def worker():
            try:
                result = self._call_ai_api(api_choice, api_key, prompt)
                # Cache the response
                if use_cache and use_cache.get() and result and not result.startswith("Erreur"):
                    try:
                        store_response(api_choice, model_id, prompt, result)
                    except Exception:
                        pass
                self.after(0, lambda: self._show_ai_result(result))
            except Exception as e:
                self.after(0, lambda: self._show_ai_result(_t(f"Erreur : {e}", f"Error: {e}")))

        threading.Thread(target=worker, daemon=True).start()
        # Show loading in result box
        if hasattr(self, 'ai_result_box'):
            self.ai_result_box.configure(state="normal")
            self.ai_result_box.delete("1.0", "end")
            self.ai_result_box.insert("1.0", _t("Analyse en cours... Veuillez patienter.", "Analysis in progress... Please wait."))
            self.ai_result_box.configure(state="disabled")

    def _call_ai_api(self, api_choice: str, api_key: str, prompt: str) -> str:
        """Call the selected AI API with the config prompt."""
        import urllib.request
        import json as _json

        headers = {"Content-Type": "application/json", "User-Agent": "UniversalSRStudio/2.0"}
        model = self.widgets["ai_model"].get()

        if "OpenRouter" in api_choice:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
            headers["HTTP-Referer"] = "https://github.com/Universal-SR-Studio"
            body = _json.dumps({
                "model": model,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
        elif "GitHub" in api_choice:
            url = "https://models.github.ai/inference/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-GitHub-Api-Version"] = "2022-11-28"
            headers["Accept"] = "application/vnd.github+json"
            body = _json.dumps({
                "model": model,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
        elif "Anthropic" in api_choice:
            url = "https://api.anthropic.com/v1/messages"
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
            body = _json.dumps({
                "model": model,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
        elif "OpenAI" in api_choice or "ChatGPT" in api_choice:
            url = "https://api.openai.com/v1/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
            body = _json.dumps({
                "model": model,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
        elif "Google" in api_choice:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            body = _json.dumps({
                "contents": [{"parts": [{"text": prompt}]}]
            }).encode()
        elif "xAI" in api_choice:
            url = "https://api.x.ai/v1/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
            body = _json.dumps({
                "model": model,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
        elif "DeepSeek" in api_choice:
            url = "https://api.deepseek.com/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
            body = _json.dumps({
                "model": model,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
        else:
            return _t("API non supportée.", "API not supported.")

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = _json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            return f"Erreur HTTP {e.code}:\n{error_body[:500]}"

        # Extract text from different API response formats
        if "content" in result and isinstance(result["content"], list):
            # Anthropic
            return result["content"][0].get("text", str(result))
        elif "choices" in result:
            # OpenAI / xAI
            return result["choices"][0]["message"]["content"]
        elif "candidates" in result:
            # Google
            parts = result["candidates"][0].get("content", {}).get("parts", [])
            return parts[0].get("text", str(result)) if parts else str(result)
        return str(result)

    def _show_ai_result(self, text: str):
        """Show AI review result in the embedded textbox and a popup window."""
        # Update embedded textbox
        if hasattr(self, 'ai_result_box'):
            self.ai_result_box.configure(state="normal")
            self.ai_result_box.delete("1.0", "end")
            self.ai_result_box.insert("1.0", text)
            self.ai_result_box.configure(state="disabled")

        # Also show in popup for convenience
        win = ctk.CTkToplevel(self)
        win.title(_t("Analyse AI de la Configuration", "AI Configuration Analysis"))
        win.geometry("700x500")
        win.attributes("-topmost", True)  # Force foreground
        win.after(500, lambda: win.attributes("-topmost", False))  # Release after showing
        try:
            icon_path = os.path.join("assets", "icon.ico")
            if os.path.exists(icon_path): win.iconbitmap(icon_path)
        except Exception: pass

        ctk.CTkLabel(win, text=_t("Analyse de votre configuration", "Analysis of your configuration"), font=("Roboto", 16, "bold"),
                     text_color="#3498db").pack(pady=(15, 5))
        txt = ctk.CTkTextbox(win, wrap="word")
        txt.pack(fill="both", expand=True, padx=15, pady=10)
        txt.insert("1.0", text)
        txt.configure(state="disabled")

        f_btn = ctk.CTkFrame(win, fg_color="transparent"); f_btn.pack(pady=10)
        ctk.CTkButton(f_btn, text=_t("Copier", "Copy"), width=100,
                      command=lambda: (win.clipboard_clear(), win.clipboard_append(text))).pack(side="left", padx=5)
        ctk.CTkButton(f_btn, text=_t("Fermer", "Close"), width=100, fg_color="#666",
                      command=win.destroy).pack(side="left", padx=5)

    # ─── PIPELINE PSNR → GAN ──────────────────────────────────
    def pipeline_test(self):
        """Run a quick 2K+2K test of the PSNR→GAN pipeline."""
        self._launch_pipeline(test_mode=True)

    def pipeline_full(self):
        """Run the full PSNR→GAN pipeline."""
        psnr_iter = self.widgets["pipe_psnr_iter"].get()
        gan_iter = self.widgets["pipe_gan_iter"].get()
        confirm = messagebox.askyesno(
            "Pipeline PSNR → GAN",
            _t(f"Lancer le pipeline complet ?\n\n"
               f"Phase 1 (PSNR) : {psnr_iter} itérations\n"
               f"Phase 2 (GAN) : {gan_iter} itérations\n\n"
               f"L'entraînement basculera automatiquement de PSNR à GAN.",
               f"Launch the full pipeline?\n\n"
               f"Phase 1 (PSNR): {psnr_iter} iterations\n"
               f"Phase 2 (GAN): {gan_iter} iterations\n\n"
               f"Training will automatically switch from PSNR to GAN.")
        )
        if confirm:
            self._launch_pipeline(test_mode=False)

    def _launch_pipeline(self, test_mode: bool = False):
        """Generate and launch the 2-phase pipeline."""
        # Collect current config
        data = {}
        for k, w in self.widgets.items():
            try: data[k] = w.get()
            except Exception: pass

        if "net_d_type" in data:
            data["net_d_type"] = self._disc_from_display(data["net_d_type"])

        engine = data.get("engine", "NeoSR")
        if test_mode:
            psnr_iter, gan_iter = 2000, 2000
        else:
            try: psnr_iter = int(data.get("pipe_psnr_iter", "150000"))
            except ValueError: psnr_iter = 150000
            try: gan_iter = int(data.get("pipe_gan_iter", "100000"))
            except ValueError: gan_iter = 100000

        psnr_lr = data.get("pipe_psnr_lr", "5e-4")
        gan_lr = data.get("pipe_gan_lr", "1e-4")
        gan_weight = data.get("pipe_gan_weight", "0.1")
        use_percep = str(data.get("pipe_psnr_percep", "0")) in ("1", "true")

        # Phase 1: PSNR config (no GAN)
        data_p1 = dict(data)
        data_p1["total_iter"] = str(psnr_iter)
        data_p1["lr"] = psnr_lr
        data_p1["use_gan"] = "false"
        data_p1["loss_pixel"] = "true"
        if use_percep:
            data_p1["loss_percep"] = "true"
        prefix = "TEST_" if test_mode else ""
        data_p1["name"] = f"{prefix}{data.get('name', 'exp')}_PSNR"

        # Phase 2: GAN config (with GAN)
        data_p2 = dict(data)
        data_p2["total_iter"] = str(gan_iter)
        data_p2["lr"] = gan_lr
        data_p2["use_gan"] = "true"
        data_p2["dyn_gan_weight"] = gan_weight
        data_p2["loss_pixel"] = "true"
        data_p2["loss_percep"] = "true"
        data_p2["name"] = f"{prefix}{data.get('name', 'exp')}_GAN"
        # Will use the PSNR pretrain as starting point
        data_p2["_psnr_pretrain"] = True

        # Save to settings for RunTab to pick up
        import json as _json
        pipeline_data = {
            "phases": [
                {"name": "PSNR", "config_data": data_p1, "iters": psnr_iter},
                {"name": "GAN", "config_data": data_p2, "iters": gan_iter},
            ],
            "engine": engine,
            "test_mode": test_mode,
        }
        pipeline_path = os.path.join(os.path.expanduser("~"), "IA_Engine", "pipeline_queue.json")
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        with open(pipeline_path, "w") as pf:
            _json.dump(pipeline_data, pf, indent=2)

        # Generate Phase 1 config file
        base_dir = os.path.join(os.path.expanduser("~"), "IA_Engine", "Option Custom")
        sub = "trainner_redux" if "Redux" in engine else "neosr"
        target_dir = os.path.join(base_dir, sub)
        os.makedirs(target_dir, exist_ok=True)

        ext = ".yml" if "Redux" in engine else ".toml"
        p1_path = os.path.join(target_dir, f"{data_p1['name']}{ext}")
        ok, msg = self.config_handler.generate_config(data_p1, p1_path)

        if not ok:
            messagebox.showerror(_t("Erreur", "Error"), _t(f"Erreur génération Phase 1 :\n{msg}", f"Phase 1 generation error:\n{msg}"))
            return

        mode = "TEST " if test_mode else ""
        messagebox.showinfo(
            f"Pipeline {mode}PSNR → GAN",
            _t(f"Phase 1 (PSNR) : {psnr_iter} iters → {p1_path}\n\n"
               f"La Phase 2 (GAN) sera générée automatiquement\n"
               f"à la fin de la Phase 1 avec le pretrain PSNR.\n\n"
               f"Lancement de la Phase 1...",
               f"Phase 1 (PSNR): {psnr_iter} iters → {p1_path}\n\n"
               f"Phase 2 (GAN) will be generated automatically\n"
               f"at the end of Phase 1 using the PSNR pretrain.\n\n"
               f"Launching Phase 1...")
        )

        # Launch Phase 1 via RunTab
        if self.run_tab_ref:
            self.run_tab_ref.pipeline_mode = pipeline_data
            self.run_tab_ref.external_start(p1_path)
        else:
            messagebox.showerror(_t("Erreur", "Error"), _t("Lien vers l'onglet Entraînement non établi.", "Link to the Training tab not established."))

    def restart_tb(self):
        if sys.platform == "win32": subprocess.run(["taskkill", "/F", "/IM", "tensorboard.exe"], capture_output=True)
        port = self.widgets["port_tb"].get()
        py_path = self.settings.get("python_path", "python")
        # Dynamic logdir: neosr → experiments/tb_logger, redux → tb_logger
        engine_val = self.widgets.get("engine", None)
        engine_val = engine_val.get() if engine_val and hasattr(engine_val, "get") else ""
        if "Redux" in engine_val or "redux" in engine_val.lower():
            tb_logdir = os.path.join(self.settings.redux_path, "tb_logger")
        else:
            tb_logdir = os.path.join(self.settings.neosr_path, "experiments", "tb_logger")
        os.makedirs(tb_logdir, exist_ok=True)
        cmd = [py_path, "-m", "tensorboard.main", "--logdir", tb_logdir, f"--port={port}", "--bind_all"]
        try:
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            messagebox.showinfo("TensorBoard", _t(f"Serveur redémarré sur le port {port}.\nLogdir : {tb_logdir}",
                                                   f"Server restarted on port {port}.\nLogdir: {tb_logdir}"))
        except Exception as e: messagebox.showerror(_t("Erreur", "Error"), str(e))

    def toggle_gan_options(self):
        is_gan = str(self.widgets["use_gan"].get()).lower() == "true"
        if is_gan:
            # Pack after the use_gan checkbox for consistent positioning
            self.frame_gan_opts.pack(fill="x", pady=5, after=self.widgets["use_gan"])
            self.on_disc_change(self.widgets["net_d_type"].get())
        else:
            self.frame_gan_opts.pack_forget()
        self.update_idletasks()  # Force immediate layout update
        self.refresh_ui_stats()

    def build_dynamic_fields(self, fields, parent_frame, widget_list, prefix="dyn_"):
        for w in widget_list: w.destroy()
        widget_list.clear()
        
        # Nettoyage des widgets existants dans self.widgets
        keys_to_del = [k for k in self.widgets if k.startswith(prefix) and k != "dyn_gan_weight"]
        for k in keys_to_del: del self.widgets[k]

        for field in fields:
            container = ctk.CTkFrame(parent_frame, fg_color="transparent")
            container.pack(side="left", padx=10, pady=5)
            widget_list.append(container)
            
            lbl = ctk.CTkLabel(container, text=field["label"])
            lbl.pack()
            
            # Infobulle
            tip_key = field.get("tip_key", field["key"])
            if tip_key in TOOLTIPS: ToolTip(lbl, TOOLTIPS[tip_key])
            
            key_id = f"{prefix}{field['key']}"
            val = str(field["default"])
            
            def on_change(event=None): self.refresh_ui_stats()

            # Détection type de champ
            if field.get("type") == "combobox": # Choix multiples définis explicitement
                choices = field.get("choices", [val])
                m = ctk.CTkOptionMenu(container, values=choices, width=140, command=lambda v: on_change())
                m.set(val)
                m.pack()
                self.widgets[key_id] = m

            elif val.strip().startswith("["): # Listes (ex: [6,6,6,6])
                e = ctk.CTkEntry(container, width=120)
                e.insert(0, val)
                e.pack()
                self.widgets[key_id] = e
                e.bind("<KeyRelease>", on_change)

            elif val.lower() in ["true", "false"]: # Booléens
                m = ctk.CTkOptionMenu(container, values=["true", "false"], width=80, command=lambda v: on_change())
                m.set(val.lower())
                m.pack()
                self.widgets[key_id] = m

            elif ("Feat" in field["label"] or "Dim" in field["label"]) and val.isdigit(): # Sliders Intelligents
                self.add_stepped_slider(container, key_id, 8, 256, 8, int(val))

            else: # Standard Entry
                e = ctk.CTkEntry(container, width=80)
                e.insert(0, val)
                e.pack()
                self.widgets[key_id] = e
                e.bind("<KeyRelease>", on_change)


    def add_stepped_slider(self, parent, key_id, min_val, max_val, step, default, extra_cmd=None):
        val_lbl = ctk.CTkLabel(parent, text=str(default), font=("Consolas", 14, "bold"), text_color="#3B8ED0"); val_lbl.pack()
        slider = ctk.CTkSlider(parent, from_=min_val, to=max_val, number_of_steps=(max_val-min_val)/step); slider.set(default); slider.pack(pady=5)
        def update_val(value): 
            stepped_val = int(round(value / step) * step); val_lbl.configure(text=str(stepped_val)); slider.rounded_value = stepped_val
            self.refresh_ui_stats()
            if extra_cmd: extra_cmd(stepped_val)
        slider.configure(command=update_val); slider.rounded_value = default; self.widgets[key_id] = slider

    def _on_optimizer_change(self, optim_name):
        _SCHED_FREE_KEYS = ("milestones", "gamma", "t_max", "eta_min", "warmup_iter", "warmup_steps")
        is_sf = str(optim_name).endswith("_SF")
        sf_chk = self.widgets.get("schedule_free")
        if sf_chk:
            if is_sf: sf_chk.select()
            else: sf_chk.deselect()
        sched_menu = self.widgets.get("scheduler")
        if sched_menu:
            try: sched_menu.configure(state="disabled" if is_sf else "normal")
            except Exception: pass
        new_state = "disabled" if is_sf else "normal"
        for k in _SCHED_FREE_KEYS:
            w = self.widgets.get(k)
            if w:
                try: w.configure(state=new_state)
                except Exception: pass

    # --- FILTRAGE PAR FAMILLE (NOUVEAUTÉ) ---
    def filter_archs(self, fam):
        engine = self.widgets["engine"].get()
        is_redux = "Redux" in engine
        # Sélection du bon dictionnaire de champs et familles
        target_dict = REDUX_ARCH_FIELDS if is_redux else ARCH_FIELDS
        families = get_arch_families("redux" if is_redux else "neosr")

        if fam in ("ALL", "TOUTES"):
            vals = sorted(list(target_dict.keys()))
        else:
            family_members = families.get(fam, [])
            vals = sorted([x for x in family_members if x in target_dict])

        if not vals: vals = [_t("(Aucun)", "(None)")]

        self.widgets["arch"].configure(values=vals)
        self.widgets["arch"].set(vals[0])
        if vals[0] != "(Aucun)":
            self.on_arch_change(vals[0])

    def on_arch_change(self, arch):
        if arch == "(Aucun)": return
        engine = self.widgets["engine"].get()
        
        target_dict = REDUX_ARCH_FIELDS if "Redux" in engine else ARCH_FIELDS
        
        # Fallback si l'architecture n'est pas dans le dictionnaire (ex: custom)
        fields = target_dict.get(arch, target_dict.get("default", []))
        
        self.build_dynamic_fields(fields, self.frame_dynamic_g, self.dynamic_widgets_g, "dyn_")
        self.refresh_ui_stats()

    def on_disc_change(self, arch_display): 
        # Convert friendly display name to internal name for DISC_FIELDS lookup
        arch = self._disc_from_display(arch_display)
        # On récupère les champs depuis le dictionnaire DISC_FIELDS
        fields = DISC_FIELDS.get(arch, [])
        # On appelle la nouvelle version de build_dynamic_fields avec la liste directement
        self.build_dynamic_fields(fields, self.frame_dynamic_d, self.dynamic_widgets_d, "dynd_")
        self.refresh_ui_stats()
    
    def refresh_ui_stats(self):
        self.update_vram_estimate()
        disc_display = self.widgets.get("net_d_type", ctk.CTkEntry(self)).get()
        disc_internal = self._disc_from_display(disc_display)
        self.perf_bars.update_stats(self.widgets["arch"].get(), str(self.widgets["use_gan"].get()).lower() == "true", disc_internal, self.widgets)
    
    def _deg_browse_image(self):
        init_dir = None
        gt_entry = self.widgets.get("dataroot_gt")
        if gt_entry:
            gt_path = gt_entry.get().strip()
            if os.path.isdir(gt_path):
                init_dir = gt_path
        path = filedialog.askopenfilename(
            initialdir=init_dir,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")]
        )
        if not path:
            return
        self.widgets["deg_preview_src"].delete(0, "end")
        self.widgets["deg_preview_src"].insert(0, path)
        self.settings.set("last_preview_image", path)
        self._deg_refresh_preview()

    def _collect_current_degradations(self) -> dict:
        """Read current values of all degradation widgets into a flat config dict."""
        keys = [
            "blur_prob", "blur_sigma", "gaussian_noise_prob", "noise_range", "gray_noise_prob",
            "jpeg_prob", "jpeg_range", "jpeg_range2",
            "second_blur_prob", "blur_sigma2", "gaussian_noise_prob2", "noise_range2", "gray_noise_prob2",
            "final_sinc_prob",
            "posterize_prob", "posterize_bits_range", "banding_prob", "banding_levels_range",
            "chroma_prob",
            "ca_prob", "ca_shift_range",
            "halation_prob", "halation_strength_range",
            "salt_pepper_prob", "salt_pepper_amount_range",
            "vhs_prob", "vhs_strength_range",
            "aliasing_prob", "aliasing_scale_range",
            "interlace_weave_prob", "interlace_weave_strength_range",
            "interlace_flicker_prob", "interlace_flicker_strength_range",
            "interlace_blend_prob", "interlace_blend_strength_range",
            "film_grain_prob", "film_grain_strength_range", "film_grain_size_range",
            "oversharp_prob", "oversharp_strength_range",
            "scanlines_prob", "scanlines_strength_range", "scanlines_spacing_range",
        ]
        cfg = {}
        for k in keys:
            w = self.widgets.get(k)
            if w is None:
                continue
            try:
                val = w.get()
            except Exception:
                continue
            # Try numeric parse
            try:
                val_f = float(val)
                cfg[k] = val_f
                continue
            except (ValueError, TypeError):
                pass
            # Otherwise keep as string (e.g. "[0.2, 1.5]")
            cfg[k] = val
        return cfg

    def _deg_generate_otf_batch(self):
        """
        Apply current OTF degradations to every image in the GT Val folder
        and save the degraded LQ images to a timestamped test_otf subfolder.
        Opens the output folder when done.
        """
        import datetime
        from src.core.otf_preview import apply_otf_pipeline
        from PIL import Image

        # Resolve GT Val folder (dataset tab widget → settings fallback)
        gt_val = ""
        w = self.widgets.get("val_gt")
        if w:
            try:
                gt_val = w.get().strip()
            except Exception:
                pass
        if not gt_val:
            gt_val = self.settings.get("ds_val_gt", "").strip()

        if not gt_val or not os.path.isdir(gt_val):
            messagebox.showerror(
                "Tester OTF",
                _t("Dossier GT Val introuvable.\nDéfinissez-le dans l'onglet Datasets (champ 'Val GT').",
                   "GT Val folder not found.\nSet it in the Datasets tab (field 'Val GT').")
            )
            return

        try:
            scale = int(self.widgets["deg_preview_scale"].get())
        except Exception:
            scale = 4

        try:
            n_var = max(1, int(self.widgets["deg_otf_variants"].get()))
        except Exception:
            n_var = 1

        cfg = self._collect_current_degradations()

        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
        files = sorted([fn for fn in os.listdir(gt_val)
                        if os.path.splitext(fn)[1].lower() in exts])

        if not files:
            messagebox.showwarning("Tester OTF", _t("Aucune image trouvée dans le dossier GT Val.", "No images found in the GT Val folder."))
            return

        # Output folder: sibling of val/ to avoid polluting GT with extra images
        # (NeoSR scans GT recursively → subdirs cause image count mismatch)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        val_dir = os.path.dirname(gt_val)           # datasets/val/
        out_dir = os.path.join(os.path.dirname(val_dir), "test_otf_output", stamp)
        os.makedirs(out_dir, exist_ok=True)

        # Progress popup
        pop = ctk.CTkToplevel(self)
        pop.title(_t("Génération OTF en cours…", "OTF generation in progress…"))
        pop.geometry("380x120")
        pop.resizable(False, False)
        pop.grab_set()
        total = len(files) * n_var
        lbl_status = ctk.CTkLabel(pop, text=_t(f"Initialisation… 0 / {total}", f"Initializing… 0 / {total}"), font=("Arial", 11))
        lbl_status.pack(pady=(18, 8))
        bar = ctk.CTkProgressBar(pop, width=340)
        bar.pack(padx=20)
        bar.set(0)

        def _upd(done):
            pct = done / total
            bar.set(pct)
            lbl_status.configure(text=f"{done} / {total} images")

        def worker():
            done = 0
            try:
                for fname in files:
                    in_path = os.path.join(gt_val, fname)
                    base = os.path.splitext(fname)[0]
                    try:
                        img = Image.open(in_path).convert("RGB")
                    except Exception:
                        done += n_var
                        pop.after(0, lambda d=done: _upd(d))
                        continue
                    for v in range(n_var):
                        lq, _ = apply_otf_pipeline(img, cfg, scale=scale)
                        suffix = f"_v{v + 1}" if n_var > 1 else ""
                        lq.save(os.path.join(out_dir, f"{base}{suffix}_LQ.png"))
                        done += 1
                        pop.after(0, lambda d=done: _upd(d))
                # Done — close popup and open folder
                pop.after(0, pop.destroy)
                subprocess.Popen(["explorer", os.path.normpath(out_dir)])
            except Exception as e:
                # Capture str(e) immediately — Python 3.12+ deletes 'e' after the
                # except block exits, so a raw lambda would raise NameError.
                _err_msg = str(e)
                def _on_err(msg=_err_msg):
                    try:
                        pop.destroy()
                    except Exception:
                        pass
                    messagebox.showerror(_t("Erreur OTF", "OTF Error"), msg)
                pop.after(0, _on_err)

        threading.Thread(target=worker, daemon=True).start()

    def _deg_refresh_preview(self):
        """Apply current degradations to the loaded source image and display HQ/LQ side by side."""
        from src.core.otf_preview import apply_otf_pipeline
        from PIL import Image, ImageTk
        import tkinter as tk

        src = self.widgets["deg_preview_src"].get().strip()
        if not src or not os.path.isfile(src):
            messagebox.showwarning(_t("Aperçu", "Preview"), _t("Sélectionnez d'abord une image source.", "Select a source image first."))
            return

        try:
            scale = int(self.widgets["deg_preview_scale"].get())
        except Exception:
            scale = 4

        cfg = self._collect_current_degradations()

        try:
            hq_img = Image.open(src).convert("RGB")
            # Limit working size to keep things snappy in preview
            if max(hq_img.size) > 768:
                hq_img.thumbnail((768, 768), Image.LANCZOS)
            lq_img, log = apply_otf_pipeline(hq_img, cfg, scale=scale)
        except Exception as e:
            messagebox.showerror(_t("Aperçu", "Preview"), _t(f"Erreur : {e}", f"Error: {e}"))
            return

        # Save originals for zoom
        self._deg_preview_refs["hq_orig"] = hq_img
        self._deg_preview_refs["lq_orig"] = lq_img

        # Clear area
        for w in self.widgets["deg_preview_area"].winfo_children():
            w.destroy()

        # Display: target a thumbnail size that fits the available frame.
        # Read frame dimensions; fallback to safe defaults if not yet rendered.
        self.widgets["deg_preview_area"].update_idletasks()
        avail_w = self.widgets["deg_preview_area"].winfo_width() or 800
        avail_h = self.widgets["deg_preview_area"].winfo_height() or 280
        # Each image takes ~half of the width minus the arrow gap
        max_img_w = max(200, (avail_w - 100) // 2)
        # Leave room for labels (~80px) at top + log line at bottom (~30px)
        max_img_h = max(180, avail_h - 70)
        target_size = (max_img_w, max_img_h)

        hq_thumb = hq_img.copy()
        hq_thumb.thumbnail(target_size, Image.LANCZOS)
        # LQ is upscaled with NEAREST so degradation is visible at the same display size
        lq_disp = lq_img.copy().resize(hq_thumb.size, Image.NEAREST)

        row = ctk.CTkFrame(self.widgets["deg_preview_area"], fg_color="transparent")
        row.pack(pady=10)

        # HQ
        col_hq = ctk.CTkFrame(row, fg_color="transparent")
        col_hq.pack(side="left", padx=10)
        ctk.CTkLabel(col_hq, text=f"Source (HQ {hq_img.size[0]}×{hq_img.size[1]})",
                     text_color="#2ecc71", font=("Roboto", 11, "bold")).pack()
        hq_photo = ImageTk.PhotoImage(hq_thumb)
        self._deg_preview_refs["hq"] = hq_photo
        lbl_hq = tk.Label(col_hq, image=hq_photo, bg="#0d0d1a", cursor="hand2")
        lbl_hq.pack(pady=4)
        lbl_hq.bind("<Button-1>", lambda e: self._deg_zoom_preview(which="hq"))

        # Arrow
        ctk.CTkLabel(row, text="→", text_color="#9b59b6",
                     font=("Roboto", 28, "bold")).pack(side="left", padx=15)

        # LQ
        col_lq = ctk.CTkFrame(row, fg_color="transparent")
        col_lq.pack(side="left", padx=10)
        scale_lbl = f"x{scale}" if scale != 1 else _t("même taille", "same size")
        ctk.CTkLabel(col_lq, text=f"Résultat OTF (LQ {scale_lbl} — {lq_img.size[0]}×{lq_img.size[1]})",
                     text_color="#e67e22", font=("Roboto", 11, "bold")).pack()
        lq_photo = ImageTk.PhotoImage(lq_disp)
        self._deg_preview_refs["lq"] = lq_photo
        lbl_lq = tk.Label(col_lq, image=lq_photo, bg="#0d0d1a", cursor="hand2")
        lbl_lq.pack(pady=4)
        lbl_lq.bind("<Button-1>", lambda e: self._deg_zoom_preview(which="lq"))

        # Steps log
        log_text = " | ".join(log) if log else _t("(aucune dégradation appliquée — vérifiez les probabilités)",
                                                    "(no degradation applied — check the probabilities)")
        ctk.CTkLabel(self.widgets["deg_preview_area"], text=log_text,
                     text_color=("gray30", "#AAA"), font=("Roboto", 9), wraplength=900
                     ).pack(pady=(0, 8))

    def _deg_zoom_preview(self, which: str = "lq"):
        """Open a zoom window with mouse wheel zoom + drag-to-pan."""
        from PIL import Image, ImageTk
        import tkinter as tk

        ref = self._deg_preview_refs.get(f"{which}_orig")
        if ref is None:
            messagebox.showinfo(_t("Zoom", "Zoom"), _t("Aucune image chargée. Cliquez d'abord sur 'Régénérer'.",
                                                     "No image loaded. Click 'Regenerate' first."))
            return

        win = ctk.CTkToplevel(self)
        win.title(_t(f"Zoom — {'HQ source' if which == 'hq' else 'LQ après OTF'}",
                     f"Zoom — {'HQ source' if which == 'hq' else 'LQ after OTF'}"))
        win.geometry("1100x800")
        win.transient(self)

        w_orig, h_orig = ref.size

        # State: zoom factor (1.0 = fit-to-window), pan offset
        state = {
            "zoom": 1.0,
            "pan_x": 0.0,
            "pan_y": 0.0,
            "drag_anchor": None,
            "ref": ref,
        }

        info = ctk.CTkLabel(win, text="", text_color=("gray30", "#AAA"), font=("Consolas", 10))
        info.pack(pady=4)

        canvas = tk.Canvas(win, bg="#0d0d1a", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=10, pady=5)

        photo_holder = {"photo": None}

        def render():
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 10 or ch < 10:
                win.after(50, render)
                return
            # Compute fit-to-window base size
            ratio_fit = min(cw / w_orig, ch / h_orig)
            # Apply zoom on top of fit
            actual_ratio = ratio_fit * state["zoom"]
            disp_w = max(1, int(w_orig * actual_ratio))
            disp_h = max(1, int(h_orig * actual_ratio))
            # Resize source. Use NEAREST when zooming in (zoom > 1) to see pixel detail.
            method = Image.NEAREST if state["zoom"] >= 1.5 else Image.LANCZOS
            disp = ref.resize((disp_w, disp_h), method)
            photo = ImageTk.PhotoImage(disp)
            photo_holder["photo"] = photo
            canvas.delete("all")
            # Center + pan offset
            x = cw // 2 + state["pan_x"]
            y = ch // 2 + state["pan_y"]
            canvas.create_image(x, y, image=photo)
            zoom_pct = int(state["zoom"] * 100)
            info.configure(
                text=_t(f"Original: {w_orig}×{h_orig}  •  Affiché: {disp_w}×{disp_h}  •  Zoom: {zoom_pct}%  "
                        f"(molette = zoom, clic+glisser = pan, double-clic = reset)",
                        f"Original: {w_orig}×{h_orig}  •  Display: {disp_w}×{disp_h}  •  Zoom: {zoom_pct}%  "
                        f"(scroll = zoom, click+drag = pan, double-click = reset)")
            )

        def on_wheel(event):
            # On Windows, event.delta is ±120 per notch
            if event.delta > 0:
                state["zoom"] = min(state["zoom"] * 1.15, 16.0)
            else:
                state["zoom"] = max(state["zoom"] / 1.15, 0.1)
            render()

        def on_button_press(event):
            state["drag_anchor"] = (event.x, event.y, state["pan_x"], state["pan_y"])

        def on_drag(event):
            if state["drag_anchor"] is None:
                return
            ax, ay, px, py = state["drag_anchor"]
            state["pan_x"] = px + (event.x - ax)
            state["pan_y"] = py + (event.y - ay)
            render()

        def on_button_release(event):
            state["drag_anchor"] = None

        def on_double_click(event):
            state["zoom"] = 1.0
            state["pan_x"] = 0
            state["pan_y"] = 0
            render()

        canvas.bind("<MouseWheel>", on_wheel)        # Windows / macOS
        canvas.bind("<Button-4>", lambda e: (state.update(zoom=min(state["zoom"]*1.15, 16.0)), render()))   # Linux up
        canvas.bind("<Button-5>", lambda e: (state.update(zoom=max(state["zoom"]/1.15, 0.1)), render()))    # Linux down
        canvas.bind("<ButtonPress-1>", on_button_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_button_release)
        canvas.bind("<Double-Button-1>", on_double_click)
        canvas.bind("<Configure>", lambda e: render())

        # Initial render after window is ready
        win.after(100, render)

    def _set_deg_val(self, key, val):
        """Set a degradation widget (entry + slider) to val without triggering preset logic."""
        w  = self.widgets.get(key)
        sl = self.widgets.get(f"_sl_{key}")
        if w:
            try:
                w.delete(0, "end")
                w.insert(0, str(val))
            except Exception:
                pass
        if sl:
            try:
                sl.set(float(val))
            except Exception:
                pass

    def _on_deg_custom_toggle(self):
        """Called when '☑ Deg. Custom' checkbox changes state."""
        if not self._chk_deg_custom.get():
            for k in ["posterize_prob", "banding_prob", "chroma_prob",
                      "ca_prob", "halation_prob", "salt_pepper_prob", "vhs_prob",
                      "aliasing_prob", "interlace_weave_prob", "interlace_flicker_prob",
                      "interlace_blend_prob", "film_grain_prob", "oversharp_prob", "scanlines_prob"]:
                self._set_deg_val(k, 0.0)
            if hasattr(self, '_chk_deg_vhs'):
                self._chk_deg_vhs.deselect()
        else:
            if hasattr(self, 'deg_preset_menu'):
                self.apply_deg_preset(self.deg_preset_menu.get())

    def _on_deg_vhs_toggle(self):
        """Called when '☑ VHS/Analog' checkbox changes state."""
        if not self._chk_deg_vhs.get():
            self._set_deg_val("vhs_prob", 0.0)
        else:
            if hasattr(self, 'deg_preset_menu'):
                self.apply_deg_preset(self.deg_preset_menu.get())

    def apply_deg_preset(self, preset_name):
        _PRESETS = {
            "Light": {
                "blur_prob": 0.2,      "blur_sigma": "[0.2, 1.5]",
                "gaussian_noise_prob": 0.1, "noise_range": "[1, 10]",
                "gray_noise_prob": 0.1,
                "jpeg_prob": 0.5,      "jpeg_range": "[80, 99]",
                "second_blur_prob": 0.3, "blur_sigma2": "[0.2, 1.0]",
                "gaussian_noise_prob2": 0.1, "noise_range2": "[1, 8]",
                "gray_noise_prob2": 0.0, "jpeg_range2": "[75, 99]",
                "final_sinc_prob": 0.3,
                "posterize_prob": 0.0, "posterize_bits_range": "[4, 8]",
                "banding_prob": 0.0,   "banding_levels_range": "[32, 64]",
                "chroma_prob": 0.0,
                "ca_prob": 0.0, "ca_shift_range": "[1, 3]",
                "halation_prob": 0.0, "halation_strength_range": "[0.05, 0.15]",
                "salt_pepper_prob": 0.0, "salt_pepper_amount_range": "[0.001, 0.01]",
                "vhs_prob": 0.0, "vhs_strength_range": "[0.1, 0.3]",
                "aliasing_prob": 0.0, "aliasing_scale_range": "[0.7, 0.9]",
                "interlace_weave_prob": 0.0,   "interlace_weave_strength_range":   "[0.5, 1.0]",
                "interlace_flicker_prob": 0.0, "interlace_flicker_strength_range": "[0.1, 0.3]",
                "interlace_blend_prob": 0.0,   "interlace_blend_strength_range":   "[0.3, 0.7]",
                "film_grain_prob": 0.0, "film_grain_strength_range": "[0.03, 0.08]", "film_grain_size_range": "[1, 2]",
                "oversharp_prob": 0.0, "oversharp_strength_range": "[0.5, 1.5]",
                "scanlines_prob": 0.0, "scanlines_strength_range": "[0.2, 0.4]", "scanlines_spacing_range": "[2, 4]",
            },
            "Medium": {
                "blur_prob": 0.5,      "blur_sigma": "[0.2, 3.0]",
                "gaussian_noise_prob": 0.3, "noise_range": "[1, 20]",
                "gray_noise_prob": 0.2,
                "jpeg_prob": 0.8,      "jpeg_range": "[50, 95]",
                "second_blur_prob": 0.5, "blur_sigma2": "[0.2, 1.5]",
                "gaussian_noise_prob2": 0.3, "noise_range2": "[1, 15]",
                "gray_noise_prob2": 0.1, "jpeg_range2": "[50, 95]",
                "final_sinc_prob": 0.5,
                "posterize_prob": 0.0, "posterize_bits_range": "[4, 8]",
                "banding_prob": 0.0,   "banding_levels_range": "[16, 64]",
                "chroma_prob": 0.1,
                "ca_prob": 0.0, "ca_shift_range": "[1, 3]",
                "halation_prob": 0.0, "halation_strength_range": "[0.05, 0.2]",
                "salt_pepper_prob": 0.05, "salt_pepper_amount_range": "[0.001, 0.02]",
                "vhs_prob": 0.0, "vhs_strength_range": "[0.1, 0.3]",
                "aliasing_prob": 0.1, "aliasing_scale_range": "[0.65, 0.9]",
                "interlace_weave_prob": 0.0,   "interlace_weave_strength_range":   "[0.5, 1.0]",
                "interlace_flicker_prob": 0.0, "interlace_flicker_strength_range": "[0.1, 0.3]",
                "interlace_blend_prob": 0.0,   "interlace_blend_strength_range":   "[0.3, 0.7]",
                "film_grain_prob": 0.15, "film_grain_strength_range": "[0.03, 0.1]", "film_grain_size_range": "[1, 2]",
                "oversharp_prob": 0.1,  "oversharp_strength_range": "[0.5, 1.5]",
                "scanlines_prob": 0.0,  "scanlines_strength_range": "[0.2, 0.4]", "scanlines_spacing_range": "[2, 4]",
            },
            "Heavy": {
                "blur_prob": 0.7,      "blur_sigma": "[1.0, 4.0]",
                "gaussian_noise_prob": 0.5, "noise_range": "[10, 40]",
                "gray_noise_prob": 0.4,
                "jpeg_prob": 1.0,      "jpeg_range": "[30, 80]",
                "second_blur_prob": 0.8, "blur_sigma2": "[0.5, 2.5]",
                "gaussian_noise_prob2": 0.5, "noise_range2": "[5, 25]",
                "gray_noise_prob2": 0.3, "jpeg_range2": "[30, 80]",
                "final_sinc_prob": 0.7,
                "posterize_prob": 0.2, "posterize_bits_range": "[3, 8]",
                "banding_prob": 0.3,   "banding_levels_range": "[16, 64]",
                "chroma_prob": 0.3,
                "ca_prob": 0.2, "ca_shift_range": "[1, 5]",
                "halation_prob": 0.2, "halation_strength_range": "[0.1, 0.3]",
                "salt_pepper_prob": 0.1, "salt_pepper_amount_range": "[0.005, 0.03]",
                "vhs_prob": 0.1, "vhs_strength_range": "[0.2, 0.5]",
                "aliasing_prob": 0.2, "aliasing_scale_range": "[0.55, 0.85]",
                "interlace_weave_prob": 0.1,   "interlace_weave_strength_range":   "[0.5, 1.0]",
                "interlace_flicker_prob": 0.05, "interlace_flicker_strength_range": "[0.1, 0.35]",
                "interlace_blend_prob": 0.05,   "interlace_blend_strength_range":   "[0.3, 0.8]",
                "film_grain_prob": 0.3, "film_grain_strength_range": "[0.05, 0.15]", "film_grain_size_range": "[1, 3]",
                "oversharp_prob": 0.2, "oversharp_strength_range": "[0.8, 2.5]",
                "scanlines_prob": 0.05, "scanlines_strength_range": "[0.25, 0.5]", "scanlines_spacing_range": "[2, 4]",
            },
            "Overkill": {
                "blur_prob": 0.9,      "blur_sigma": "[2.0, 6.0]",
                "gaussian_noise_prob": 0.8, "noise_range": "[20, 60]",
                "gray_noise_prob": 0.8,
                "jpeg_prob": 1.0,      "jpeg_range": "[10, 60]",
                "second_blur_prob": 1.0, "blur_sigma2": "[1.0, 4.0]",
                "gaussian_noise_prob2": 0.8, "noise_range2": "[10, 40]",
                "gray_noise_prob2": 0.6, "jpeg_range2": "[10, 50]",
                "final_sinc_prob": 0.9,
                "posterize_prob": 0.5, "posterize_bits_range": "[2, 6]",
                "banding_prob": 0.6,   "banding_levels_range": "[8, 32]",
                "chroma_prob": 0.6,
                "ca_prob": 0.4, "ca_shift_range": "[2, 7]",
                "halation_prob": 0.4, "halation_strength_range": "[0.15, 0.4]",
                "salt_pepper_prob": 0.2, "salt_pepper_amount_range": "[0.01, 0.05]",
                "vhs_prob": 0.3, "vhs_strength_range": "[0.3, 0.7]",
                "aliasing_prob": 0.35, "aliasing_scale_range": "[0.5, 0.8]",
                "interlace_weave_prob": 0.2,   "interlace_weave_strength_range":   "[0.7, 1.0]",
                "interlace_flicker_prob": 0.15, "interlace_flicker_strength_range": "[0.2, 0.5]",
                "interlace_blend_prob": 0.15,   "interlace_blend_strength_range":   "[0.5, 1.0]",
                "film_grain_prob": 0.5, "film_grain_strength_range": "[0.06, 0.18]", "film_grain_size_range": "[1, 4]",
                "oversharp_prob": 0.35, "oversharp_strength_range": "[1.0, 3.0]",
                "scanlines_prob": 0.15, "scanlines_strength_range": "[0.3, 0.6]", "scanlines_spacing_range": "[2, 4]",
            },
        }
        _CUSTOM_KEYS = {
            "posterize_prob", "posterize_bits_range",
            "banding_prob", "banding_levels_range",
            "chroma_prob",
            "ca_prob", "ca_shift_range",
            "halation_prob", "halation_strength_range",
            "salt_pepper_prob", "salt_pepper_amount_range",
            "vhs_prob", "vhs_strength_range",
            "aliasing_prob", "aliasing_scale_range",
            "interlace_weave_prob", "interlace_weave_strength_range",
            "interlace_flicker_prob", "interlace_flicker_strength_range",
            "interlace_blend_prob", "interlace_blend_strength_range",
            "film_grain_prob", "film_grain_strength_range", "film_grain_size_range",
            "oversharp_prob", "oversharp_strength_range",
            "scanlines_prob", "scanlines_strength_range", "scanlines_spacing_range",
        }
        _VHS_KEYS = {"vhs_prob", "vhs_strength_range"}

        use_custom = not hasattr(self, '_chk_deg_custom') or bool(self._chk_deg_custom.get())
        use_vhs    = not hasattr(self, '_chk_deg_vhs')    or bool(self._chk_deg_vhs.get())

        p = _PRESETS.get(preset_name, {})
        for key, val in p.items():
            if key in _CUSTOM_KEYS and not use_custom:
                continue
            if key in _VHS_KEYS and not use_vhs:
                continue
            if key not in self.widgets:
                continue
            try:
                w = self.widgets[key]
                sl = self.widgets.get(f"_sl_{key}")
                if sl is not None:
                    try:
                        sl.set(float(val))
                    except Exception:
                        pass
                if isinstance(w, ctk.CTkEntry) or getattr(w, "_range_proxy", False):
                    w.delete(0, "end")
                    w.insert(0, str(val))
            except Exception:
                pass
        if hasattr(self, 'lbl_deg_info'):
            flags = []
            if not use_custom: flags.append("sans custom")
            if not use_vhs:    flags.append("sans VHS")
            suffix = f" ({', '.join(flags)})" if flags else ""
            self.lbl_deg_info.configure(text=f"Preset : {preset_name}{suffix}")

    def detect_gpu_hardware(self):
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
            res = subprocess.run(cmd, capture_output=True, text=True, creationflags=0x08000000 if sys.platform=='win32' else 0)
            if res.returncode == 0: p = res.stdout.strip().split(','); self.gpu_name=p[0].strip(); self.total_vram_gb=int(p[1].strip())/1024.0
        except Exception: pass
    
    def update_vram_estimate(self, *args):
        """Debounced VRAM estimation — waits 300ms after last keystroke."""
        if self._vram_timer is not None:
            self.after_cancel(self._vram_timer)
        self._vram_timer = self.after(300, self._do_vram_estimate)

    def _do_vram_estimate(self):
        """Actual VRAM calculation (called after debounce delay)."""
        self._vram_timer = None
        try:
            try: bs = int(self.widgets["batch_size"].get())
            except Exception: bs = 4
            try: patch = int(self.widgets["patch_size"].get())
            except Exception: patch = 64
            arch = self.widgets["arch"].get()
            use_amp = 0
            if "use_amp" in self.widgets:
                val = str(self.widgets["use_amp"].get()).lower()
                if val in ["1", "true", "on"]: use_amp = 1
            feat_factor = 64
            for k, w in self.widgets.items():
                if k.startswith("dyn_") and ("feat" in k or "dim" in k):
                    try: 
                        if hasattr(w, "rounded_value"): feat_factor = int(w.rounded_value)
                        else: feat_factor = int(float(w.get()))
                    except Exception: pass
            
            # --- FORMULE VRAM — utilise compute_estimator (source unique de vérité) ---
            use_amp_bool = use_amp == 1
            base_est = _ce_estimate_vram(arch, batch_size=bs, patch_size=patch, use_amp=use_amp_bool)

            # Overhead GAN (+30%)
            if "use_gan" in self.widgets and str(self.widgets["use_gan"].get()).lower() in ("true", "1", "on"):
                base_est *= 1.3
            # Overhead perceptuel / losses mesurés (bench Redux 2026-05-19)
            if "loss_fdl" in self.widgets and str(self.widgets["loss_fdl"].get()).lower() in ("true", "1", "on"):
                if "fdl_model" in self.widgets and self.widgets["fdl_model"].get() == "dinov2": base_est += 1.5
                else: base_est += 0.5
            if "loss_percep" in self.widgets and str(self.widgets["loss_percep"].get()).lower() in ("true", "1", "on"):
                base_est += 0.5
            if "loss_mssim" in self.widgets and str(self.widgets["loss_mssim"].get()).lower() in ("true", "1", "on"):
                base_est += 0.2
            if "loss_dists" in self.widgets and str(self.widgets["loss_dists"].get()).lower() in ("true", "1", "on"):
                base_est += 0.4
            if "loss_msswd" in self.widgets and str(self.widgets["loss_msswd"].get()).lower() in ("true", "1", "on"):
                base_est += 0.3
            if "loss_contextual" in self.widgets and str(self.widgets["loss_contextual"].get()).lower() in ("true", "1", "on"):
                base_est += 0.8
            if "loss_wavelet" in self.widgets and str(self.widgets["loss_wavelet"].get()).lower() in ("true", "1", "on"):
                base_est += 0.3

            if self.total_vram_gb > 0:
                pct = base_est / self.total_vram_gb
                self.prog_vram.set(min(pct, 1.0))
                color = "#2ecc71" if pct < 0.6 else "#e67e22" if pct < 0.9 else "#e74c3c"
                self.prog_vram.configure(progress_color=color)
                self.lbl_vram.configure(text=f"{_t('VRAM Estimée', 'Estimated VRAM')} : {base_est:.1f} GB / {self.total_vram_gb:.1f} GB ({int(pct*100)}%)")
            else:
                self.lbl_vram.configure(text=f"{_t('VRAM Estimée', 'Estimated VRAM')} : {base_est:.1f} GB")
        except Exception as e: pass

    # --- UI HELPERS ---
    def add_param_grid(self, parent, label, default, key, row, col, tip_text=None):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        lbl = ctk.CTkLabel(f, text=label, font=("Roboto", 10)); lbl.pack(anchor="w")
        
        # Gestion intelligente de l'infobulle (clé ou texte brut)
        if tip_text:
            text = get_tooltip(tip_text, tip_text) # Essaie de trouver la clé, sinon utilise le texte
            ToolTip(lbl, text)
            
        if str(default).lower() in ["true", "false"]: e = ctk.CTkOptionMenu(f, values=["true", "false"]); e.set(str(default).lower())
        else: e = ctk.CTkEntry(f); e.insert(0, default); e.bind("<KeyRelease>", lambda event: self.update_vram_estimate())
        e.pack(fill="x"); self.widgets[key] = e; parent.grid_columnconfigure(col, weight=1)

    def add_header(self, parent, text): ctk.CTkLabel(parent, text=text, font=("Roboto", 16, "bold"), text_color="#3B8ED0", anchor="w").pack(fill="x", pady=(15, 5))
    def add_label_tip(self, parent, text, tip_key=None):
        lbl = ctk.CTkLabel(parent, text=text); lbl.pack(side="left", padx=5); ToolTip(lbl, get_tooltip(tip_key, "")) if tip_key else None
    def row_entry(self, parent, label, key, tip_key=None):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill="x", pady=2)
        lbl = ctk.CTkLabel(f, text=label, width=150, anchor="w"); lbl.pack(side="left"); ToolTip(lbl, get_tooltip(tip_key, "")) if tip_key else None
        e = ctk.CTkEntry(f); e.bind("<KeyRelease>", lambda e: self.refresh_ui_stats()); e.pack(side="left", fill="x", expand=True); self.widgets[key] = e
    def row_file_picker(self, parent, label, key, is_file=False, default="", save_key=None, tip_key=None):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill="x", pady=2)
        lbl = ctk.CTkLabel(f, text=label, width=150, anchor="w"); lbl.pack(side="left"); ToolTip(lbl, get_tooltip(tip_key, "")) if tip_key else None
        e = ctk.CTkEntry(f); e.pack(side="left", fill="x", expand=True, padx=(0, 5)); 
        if default: e.insert(0, default)
        self.widgets[key] = e; ctk.CTkButton(f, text="...", width=30, command=lambda: self.browse(e, is_file, save_key)).pack(side="left")
    
    def create_nav_btn(self, text, row, name):
        ctk.CTkButton(self.frame_nav, text=text, fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=lambda: self.show_frame(name)).grid(row=row, column=0, sticky="ew", padx=20, pady=5)
    def show_frame(self, name):
        for f in self.frames.values(): f.pack_forget()
        self.frames[name].pack(fill="both", expand=True)
    def browse(self, e, f, k):
        if f:
            p = filedialog.askopenfilename()
        else:
            p = filedialog.askdirectory()
        if p:
            e.delete(0, "end")
            e.insert(0, p)
            if k:
                self.settings.set(k, p)
    
    def load_action(self):
        # Pour charger, on ouvre le dossier correspondant au moteur actuel
        curr_engine = self.widgets["engine"].get()
        init_dir = os.path.join(os.path.expanduser("~"), "IA_Engine", "Option Custom")
        if "Redux" in curr_engine: init_dir = os.path.join(init_dir, "trainner_redux")
        else: init_dir = os.path.join(init_dir, "neosr")
        
        if not os.path.exists(init_dir): os.makedirs(init_dir, exist_ok=True)

        p = filedialog.askopenfilename(filetypes=[("Config", "*.toml *.yml *.yaml")], initialdir=init_dir)
        if p:
            succ, data = self.config_handler.load_config(p)
            if succ:
                # --- AUTO-DETECT ENGINE from file type ---
                detected = data.pop("_detected_engine", None)
                if detected:
                    curr = self.widgets["engine"].get()
                    if detected != curr:
                        self.widgets["engine"].set(detected)
                        self.on_engine_change(detected)
                        self.update_idletasks()
                        try:
                            print(f"[Config] Moteur basculé automatiquement → {detected}")
                        except OSError:
                            pass

                for k in self.widgets:
                    if k.startswith("aug_"): self.widgets[k].deselect()
                if "arch" in data: self.widgets["arch"].set(data["arch"]); self.on_arch_change(data["arch"]); self.update_idletasks()
                if "use_gan" in data: 
                    is_gan = str(data["use_gan"]).lower() == "true"; 
                    if is_gan: self.widgets["use_gan"].select()
                    else: self.widgets["use_gan"].deselect()
                    self.toggle_gan_options()
                    if is_gan and "net_d_type" in data:
                        disc_internal = data["net_d_type"]
                        disc_display = DISC_DISPLAY_NAMES.get(disc_internal, disc_internal)
                        self.widgets["net_d_type"].set(disc_display)
                        self.on_disc_change(disc_display)
                        self.update_idletasks()
                for k, v in data.items():
                    if k in self.widgets:
                        try:
                            w = self.widgets[k]
                            if isinstance(w, ctk.CTkOptionMenu): val = str(v).lower() if str(v).lower() in ["true", "false"] else str(v); w.set(val)
                            elif isinstance(w, ctk.CTkCheckBox):
                                if str(v).lower() == "true": w.select()
                                else: w.deselect()
                            elif isinstance(w, ctk.CTkSlider):
                                w.set(float(v))
                                if hasattr(w, "rounded_value"): w._command(float(v))
                            else:
                                w.delete(0, "end"); w.insert(0, str(v))
                                # FocusOut triggers entry_cb which handles both log sliders
                                # (_to_sv conversion) and linear sliders (raw set)
                                try: w.event_generate("<FocusOut>")
                                except Exception: pass
                                # For linear sliders only: raw set (log sliders use entry_cb above)
                                _LOG_SLIDER_KEYS = ("lr", "lr_d", "dyn_gan_weight")
                                if k not in _LOG_SLIDER_KEYS:
                                    sl = self.widgets.get(f"_sl_{k}")
                                    if sl is not None:
                                        try: sl.set(float(v))
                                        except Exception: pass
                        except Exception: pass
                    # --- FIX LABEL UPDATE ON LOAD ---
                    if k in self.widgets and "prob_" in k and k in self.aug_labels:
                        try: self.aug_labels[k].configure(text=f"{float(v):.2f}")
                        except Exception: pass
                # Sync preview scale with project scale
                scale_val = str(data.get("scale", "4"))
                prev_scale_widget = self.widgets.get("deg_preview_scale")
                if prev_scale_widget and scale_val in ["1", "2", "3", "4", "8"]:
                    prev_scale_widget.set(scale_val)
                self.refresh_ui_stats()
                # Sync état seed entry ↔ checkbox déterministe après chargement
                _det_w = self.widgets.get("deterministic")
                _seed_w = self.widgets.get("manual_seed")
                if _det_w and _seed_w:
                    if _det_w.get() == "true":
                        _seed_w.configure(state="normal", text_color=("gray10", "#DCE4EE"))
                    else:
                        _seed_w.delete(0, "end")
                        _seed_w.insert(0, "0")
                        _seed_w.configure(state="disabled", text_color="#666666")
                messagebox.showinfo("OK", _t("Configuration chargée !", "Configuration loaded!"))
            else: messagebox.showerror(_t("Erreur", "Error"), data)

    def save_action(self):
        data = {}
        for k, w in self.widgets.items():
            try: data[k] = w.rounded_value if hasattr(w, "rounded_value") else w.get()
            except Exception: pass
        
        if "use_shuffle" in self.widgets: data["use_shuffle"] = bool(self.widgets["use_shuffle"].get())

        # Convert disc display name to internal name for config generation
        if "net_d_type" in data:
            data["net_d_type"] = self._disc_from_display(data["net_d_type"])
        
        # --- FIX SCHEDULER: MAPPING STRICT NEOSR ---
        sched_ui = self.widgets["scheduler"].get()
        if sched_ui == "CosineAnnealing":
            data["scheduler"] = "CosineAnnealing"
            data["scheduler_params"] = {"T_max": self.widgets["t_max"].get(), "eta_min": self.widgets["eta_min"].get()}
        elif sched_ui == "MultiStepLR":
            data["scheduler"] = "MultiStepLR"
            data["scheduler_params"] = {"milestones": self.widgets["milestones"].get(), "gamma": self.widgets["gamma"].get()}

        self.settings.set("ds_train_gt", data.get("dataroot_gt", "")); self.settings.set("ds_val_gt", data.get("val_gt", ""))
        
        # --- SAUVEGARDE INTELLIGENTE (Dossier Custom) ---
        curr_engine = self.widgets["engine"].get()
        base_custom = os.path.join(os.path.expanduser("~"), "IA_Engine", "Option Custom")
        target_dir = os.path.join(base_custom, "trainner_redux") if "Redux" in curr_engine else os.path.join(base_custom, "neosr")
        
        if not os.path.exists(target_dir): os.makedirs(target_dir, exist_ok=True)

        # Pré-remplir le nom
        init_file = data.get("name", "experiment").strip()
        if "Redux" in curr_engine: init_file += ".yml"
        else: init_file += ".toml"

        p = filedialog.asksaveasfilename(
            defaultextension=".yml" if "Redux" in curr_engine else ".toml", 
            initialdir=target_dir,
            initialfile=init_file
        )
        
        if p:
            succ, msg = self.config_handler.generate_config(data, p); messagebox.showinfo("Info", msg) if succ else messagebox.showerror(_t("Erreur", "Error"), msg)
            return p
        return None

    def save_and_run(self):
        toml_path = self.save_action()
        if toml_path:
            # Création préventive du dossier experiments pour éviter crash logger NeoSR
            try:
                exp_name = self.widgets["name"].get().strip()
                engine_path = os.path.join(os.path.expanduser("~"), "IA_Engine", "neosr", "experiments", exp_name)
                if not os.path.exists(engine_path):
                    os.makedirs(engine_path, exist_ok=True)
                    print(f"[FIX] Dossier créé manuellement : {engine_path}")
            except Exception as e:
                print(f"[WARN] Impossible de créer le dossier experiments : {e}")
            
            # Lancement via la référence RunTab
            if self.run_tab_ref:
                self.run_tab_ref.external_start(toml_path)
            else:
                messagebox.showerror(_t("Erreur", "Error"), _t("Lien vers l'onglet Entraînement non établi.", "Link to the Training tab is not established."))