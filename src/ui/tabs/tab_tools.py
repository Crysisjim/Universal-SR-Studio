import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
import os
import threading
import subprocess
import sys
import queue
import random
# Lazy imports for startup performance (PERF-01)
np = None
Image = ImageTk = ImageFilter = ImageDraw = None

def _ensure_pil():
    global Image, ImageTk, ImageFilter, ImageDraw
    if Image is None:
        from PIL import Image as _Img, ImageTk as _ITk, ImageFilter as _IF, ImageDraw as _ID
        Image, ImageTk, ImageFilter, ImageDraw = _Img, _ITk, _IF, _ID

def _ensure_numpy():
    global np
    if np is None:
        import numpy as _np
        np = _np

from src.ui.components.tooltip import ToolTip
from src.core.descriptions import TOOLTIPS
from src.core.settings import SettingsManager


class ToolsTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.settings = SettingsManager()
        self.proc_lmdb = None
        self.widgets    = {}
        self._gpu_panels = []   # liste des panneaux GPU actifs (upscale, bench, conv)

        # Thread-safe queue for UI updates
        self._ui_queue = queue.Queue()
        self._poll_ui_queue()

        # --- LAYOUT PRINCIPAL (SIDEBAR + CONTENT) ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. SIDEBAR (Navigation)
        self.frame_nav = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.frame_nav.grid(row=0, column=0, sticky="nsew")
        # Give weight to the row AFTER the last button so all buttons stay grouped at top
        # (row 13 is the first row after the 12 nav buttons)
        self.frame_nav.grid_rowconfigure(13, weight=1)

        ctk.CTkLabel(self.frame_nav, text="BOÎTE À OUTILS", font=("Roboto", 20, "bold")).grid(row=0, column=0, padx=20, pady=20)

        # ── Section 1: Visualisation ──
        self.create_nav_btn("📊 Comparateur", 1, "comp")
        self.create_nav_btn("🔎 Quick Upscale", 2, "upscale")
        # ── Section 2: Datasets ──
        self.create_nav_btn("⚡ Générateur LQ", 3, "gen")
        self.create_nav_btn("🔄 Convertisseur", 4, "conv")
        self.create_nav_btn("💾 Créateur LMDB", 5, "lmdb")
        self.create_nav_btn("🧐 Check Dataset", 6, "chk")
        # ── Section 3: Modèles ──
        self.create_nav_btn("📏 Métriques", 7, "met")
        self.create_nav_btn("ℹ Info Modèle", 8, "model_info")
        # ── Section 4: Suivi entraînements ──
        self.create_nav_btn("📜 Historique", 9, "history")
        self.create_nav_btn("♻ Resume Failed", 10, "resume")
        self.create_nav_btn("📦 Publier Modèle", 11, "export")
        # ── Section 5: Performance ──
        self.create_nav_btn("📈 Benchmark", 12, "bench")

        # 2. CONTENT AREA
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        self.frames = {
            "comp": self.create_page_comparator(),
            "upscale": self.create_page_upscale(),
            "gen": self.create_page_generator(),
            "conv": self.create_page_converter(),
            "lmdb": self.create_page_lmdb(),
            "met": self.create_page_metrics(),
            "chk": self.create_page_checker(),
            "history": self.create_page_history(),
            "resume": self.create_page_resume(),
            "model_info": self.create_page_model_info(),
            "export": self.create_page_export(),
            "bench": self.create_page_benchmark(),
        }
        self.show_frame("comp")

    # ─── Helpers ─────────────────────────────────────────────

    def create_nav_btn(self, text, row, name):
        ctk.CTkButton(
            self.frame_nav, text=text, fg_color="transparent",
            text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
            anchor="w", command=lambda _n=name: self.show_frame(_n)
        ).grid(row=row, column=0, sticky="ew", padx=20, pady=5)

    def show_frame(self, name):
        for f in self.frames.values():
            f.pack_forget()
        self.frames[name].pack(fill="both", expand=True)

    def _poll_ui_queue(self):
        try:
            while True:
                func, args, kwargs = self._ui_queue.get_nowait()
                func(*args, **kwargs)
        except queue.Empty:
            pass
        self.after(100, self._poll_ui_queue)

    def _ui_update(self, func, *args, **kwargs):
        self._ui_queue.put((func, args, kwargs))

    # ── GPU stats polling ────────────────────────────────────────────────────────

    def _create_gpu_panel(self, parent):
        """Panneau GPU — une ligne : 🖥 GPU | Load [bar] val | VRAM [bar] val | Temp [bar] val.

        Placé side='right' dans le header avec 1 cm de marge droite.
        Frame auto-dimensionnée au contenu (pas de pack_propagate(False) ni width fixe).
        Barres identiques : width=130, height=8.
        """
        gf = ctk.CTkFrame(parent, fg_color="#111827", corner_radius=8)
        panel: dict = {}

        row = ctk.CTkFrame(gf, fg_color="transparent")
        row.pack(padx=10, pady=6)

        ctk.CTkLabel(row, text="🖥 GPU", font=("Arial", 9, "bold"),
                     text_color="#3B8ED0").pack(side="left", padx=(0, 14))

        def _grp(lbl_text: str, bar_key: str, val_key: str,
                 bar_color: str, val_width: int = 32):
            g = ctk.CTkFrame(row, fg_color="transparent")
            g.pack(side="left", padx=8)
            ctk.CTkLabel(g, text=lbl_text, font=("Consolas", 9),
                         text_color="#94a3b8", width=28,
                         anchor="w").pack(side="left")
            bar = ctk.CTkProgressBar(g, height=8, width=130,
                                     progress_color=bar_color,
                                     fg_color="#1e293b")
            bar.set(0)
            bar.pack(side="left", padx=(3, 4))
            lbl = ctk.CTkLabel(g, text="—", font=("Consolas", 9),
                               width=val_width, anchor="w")
            lbl.pack(side="left")
            panel[bar_key] = bar
            panel[val_key]  = lbl

        _grp("Load", "load_bar", "load", "#3498db", val_width=30)
        _grp("VRAM", "vram_bar", "vram", "#9b59b6", val_width=56)
        _grp("Temp", "temp_bar", "temp", "#2ecc71", val_width=32)

        self._gpu_panels.append(panel)
        if not getattr(self, "_gpu_polling", False):
            self._gpu_polling = True
            self._gpu_do_poll()
        return gf

    def _gpu_do_poll(self):
        """Requête nvidia-smi dans thread daemon → met à jour tous les _gpu_panels.

        Reprogrammé toutes les 2 s via ui_queue (thread-safe).
        Arrêt automatique si nvidia-smi absent.
        """
        _cnow = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        def _query():
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--id=0",
                     "--query-gpu=temperature.gpu,utilization.gpu,"
                     "memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                    creationflags=_cnow)

                if res.returncode == 0:
                    parts = [p.strip() for p in res.stdout.strip().split(",")]
                    if len(parts) == 4:
                        temp_s, load_s, used_s, tot_s = parts

                        def _update():
                            try:
                                ti = int(temp_s); li = int(load_s)
                                ui = int(used_s); gi = int(tot_s)
                                tf = min(1.0, ti / 120.0)
                                lf = min(1.0, li / 100.0)
                                vf = min(1.0, ui / gi) if gi > 0 else 0.0
                                # Couleur temp
                                tc = ("#e74c3c" if ti > 85 else
                                      "#f39c12" if ti > 70 else "#2ecc71")
                                # Couleur load
                                lc = ("#e74c3c" if li > 80 else
                                      "#f39c12" if li > 60 else "#3498db")
                                # Couleur vram
                                vc = ("#e74c3c" if vf > 0.9 else
                                      "#f39c12" if vf > 0.7 else "#9b59b6")
                                vstr = f"{ui/1024:.1f}/{gi/1024:.1f}G"
                                for p in self._gpu_panels:
                                    p["temp_bar"].set(tf)
                                    p["temp_bar"].configure(progress_color=tc)
                                    p["temp"].configure(text=f"{ti}°C")
                                    p["load_bar"].set(lf)
                                    p["load_bar"].configure(progress_color=lc)
                                    p["load"].configure(text=f"{li}%")
                                    p["vram_bar"].set(vf)
                                    p["vram_bar"].configure(progress_color=vc)
                                    p["vram"].configure(text=vstr)
                            except Exception:
                                pass

                        self._ui_update(_update)
                else:
                    def _na():
                        for p in self._gpu_panels:
                            for k in ("temp", "load", "vram"):
                                p[k].configure(text="N/A")
                    self._ui_update(_na)

            except (FileNotFoundError, OSError):
                def _cpu():
                    for p in self._gpu_panels:
                        p["temp"].configure(text="—")
                        p["load"].configure(text="—")
                        p["vram"].configure(text="no GPU")
                self._ui_update(_cpu)
                self._gpu_polling = False
            except Exception:
                pass  # timeout transitoire — réessaie
            finally:
                if getattr(self, "_gpu_polling", False):
                    self._ui_update(self.after, 2000, self._gpu_do_poll)

        threading.Thread(target=_query, daemon=True).start()

    def _show_toast(self, title: str, msg: str = "", ok: bool = True,
                    duration_ms: int = 4500, notif_key: str = None):
        """Fire a native Windows 11 toast notification (if enabled in settings).
        Runs in a daemon thread so win11toast's blocking wait never freezes the UI.
        """
        if notif_key and not self.settings.get(notif_key, True):
            return

        def _fire():
            try:
                from src.core.toast_notifications import show_toast
                show_toast(title, msg, duration="short")
            except Exception:
                pass

        threading.Thread(target=_fire, daemon=True).start()

    def _play_sound(self, sound_name: str, setting_key: str = None):
        """Play a WAV from assets/, respecting the setting toggle."""
        if setting_key and not self.settings.get(setting_key, True):
            return
        import sys as _sys
        sound_path = os.path.join(
            os.path.dirname(os.path.abspath(_sys.argv[0])),
            "assets", f"{sound_name}.WAV")
        if not os.path.isfile(sound_path):
            sound_path = sound_path[:-4] + ".wav"  # try lowercase
        if not os.path.isfile(sound_path):
            return
        try:
            import winsound
            winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass

    def _ups_save_sash(self, paned):
        """Save PanedWindow sash Y position to settings."""
        try:
            h = paned.sash_coord(0)[1]
            if h > 20:
                self.settings.set("ups_log_sash_h", h)
        except Exception:
            pass

    def _ups_request_stop(self):
        """Request abort of current upscale batch."""
        if hasattr(self, "_ups_stop_flag"):
            self._ups_stop_flag.set()
        self.widgets["ups_stop_btn"].configure(state="disabled", text="Arrêt...")

    def add_header(self, parent, text, desc=""):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(f, text=text, font=("Roboto", 24, "bold"), text_color="#3B8ED0", anchor="w").pack(fill="x")
        if desc:
            ctk.CTkLabel(f, text=desc, font=("Arial", 12), text_color="gray", anchor="w").pack(fill="x")

    def add_path_row(self, parent, label, var_name, is_file=False):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", pady=5)
        ctk.CTkLabel(f, text=label, width=150, anchor="w").pack(side="left")
        e = ctk.CTkEntry(f)
        e.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets[var_name] = e
        cmd = (lambda _e=e, _f=is_file: self._browse_file(_e) if _f else self._browse_dir(_e))
        ctk.CTkButton(f, text="...", width=30, command=cmd).pack(side="left")

    def _browse_dir(self, e):
        d = filedialog.askdirectory()
        if d:
            e.delete(0, "end")
            e.insert(0, d)

    def _browse_file(self, e):
        d = filedialog.askopenfilename()
        if d:
            e.delete(0, "end")
            e.insert(0, d)

    # ==========================================
    # PAGE 1: COMPARATEUR (FIXED — both images load)
    # ==========================================
    def create_page_comparator(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Comparateur Visuel", "Glissez pour comparer LQ (gauche) et HQ (droite).")

        ctrl = ctk.CTkFrame(f)
        ctrl.pack(fill="x", pady=10, padx=10)
        ctk.CTkButton(ctrl, text="Ouvrir Image LQ (Avant)", command=lambda: self.load_comp_img("before")).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(ctrl, text="Ouvrir Image HQ (Après)", command=lambda: self.load_comp_img("after")).pack(side="left", padx=10, pady=10)

        # Auto-load checkbox
        self.comp_auto_load = ctk.CTkCheckBox(ctrl, text="Auto-charger dernière image traitée")
        self.comp_auto_load.pack(side="left", padx=15)

        self.canvas_frame = ctk.CTkFrame(f, fg_color="#101010")
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.cv = Canvas(self.canvas_frame, bg="#101010", highlightthickness=0, cursor="sb_h_double_arrow")
        self.cv.pack(fill="both", expand=True)

        self.cv.bind("<Motion>", self._on_comp_move)
        self.cv.bind("<B1-Motion>", self._on_comp_move)
        self.cv.bind("<Configure>", self._on_comp_resize)

        self.img_before = None
        self.img_after = None
        self.tk_before = None
        self.tk_after = None
        self.slider_x = 0
        self._comp_new_size = None
        return f

    def load_comp_img(self, type_img):
        _ensure_pil()
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            if type_img == "before":
                self.img_before = img
            else:
                self.img_after = img
            # Force redraw — invalidate cache because the underlying image changed
            self._comp_new_size = None
            self._update_comparator()
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image : {e}")

    def _update_comparator(self):
        _ensure_pil()
        """Redraw comparator with both images via PIL crop compositing."""
        ref_img = self.img_after or self.img_before
        if ref_img is None:
            return

        w_can = self.cv.winfo_width()
        h_can = self.cv.winfo_height()
        if w_can < 10 or h_can < 10:
            return

        img_w, img_h = ref_img.size
        ratio = min(w_can / img_w, h_can / img_h)
        new_size = (max(int(img_w * ratio), 1), max(int(img_h * ratio), 1))

        # PERF-06: Only resize if size actually changed
        if getattr(self, '_comp_new_size', None) == new_size:
            return
        self._comp_new_size = new_size

        # Resize both images (cached until next resize)
        if self.img_before:
            self._resized_before = self.img_before.resize(new_size, Image.LANCZOS)
        else:
            self._resized_before = Image.new("RGB", new_size, (40, 40, 40))

        if self.img_after:
            self._resized_after = self.img_after.resize(new_size, Image.LANCZOS)
        else:
            self._resized_after = Image.new("RGB", new_size, (40, 40, 40))

        self._draw_comp_slider(self.slider_x if self.slider_x > 0 else w_can // 2)

    def _on_comp_resize(self, event):
        self._comp_new_size = None  # Force recalculation on resize
        self._update_comparator()

    def _on_comp_move(self, event):
        if self.img_before is None and self.img_after is None:
            return
        self.slider_x = event.x
        self._draw_comp_slider(event.x)

    def _draw_comp_slider(self, x):
        """Composite left=LQ right=HQ with PIL crop, then draw on canvas."""
        self.cv.delete("all")
        if not hasattr(self, "_resized_before") or self._comp_new_size is None:
            return

        w, h = self._comp_new_size
        w_can = self.cv.winfo_width()
        h_can = self.cv.winfo_height()
        pos_x = (w_can - w) // 2
        pos_y = (h_can - h) // 2

        split = max(0, min(x - pos_x, w))

        composite = self._resized_after.copy()
        if split > 0:
            left_crop = self._resized_before.crop((0, 0, split, h))
            composite.paste(left_crop, (0, 0))

        # PERF-07: Explicitly delete old PhotoImage before creating new one
        if hasattr(self, '_tk_composite') and self._tk_composite:
            del self._tk_composite
        self._tk_composite = ImageTk.PhotoImage(composite)
        self.cv.create_image(pos_x, pos_y, image=self._tk_composite, anchor="nw")

        self.cv.create_line(x, pos_y, x, pos_y + h, fill="#e74c3c", width=2)
        self.cv.create_text(x - 5, pos_y + 5, text="LQ", anchor="ne", fill="#e74c3c", font=("Arial", 10, "bold"))
        self.cv.create_text(x + 5, pos_y + 5, text="HQ", anchor="nw", fill="#2ecc71", font=("Arial", 10, "bold"))

    # ── Quick Upscale helpers ──────────────────────────────────────
    def _ups_pick_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("Model Files", "*.safetensors *.pth *.onnx"), ("All Files", "*.*")])
        if path:
            self._ups_model_var.set(path)

    def _ups_pick_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif"),
                       ("All Files", "*.*")])
        if path:
            self._ups_input_var.set(path)

    def _ups_pick_input_folder(self):
        d = filedialog.askdirectory()
        if d:
            self._ups_input_var.set(d)

    def _ups_pick_output_folder(self):
        d = filedialog.askdirectory()
        if d:
            self._ups_output_var.set(d)

    def _ups_on_same_folder_toggle(self):
        if not hasattr(self, "_ups_same_folder"):
            return
        same = bool(self._ups_same_folder.get())
        self.settings.set("ups_same_folder", same)
        state = "disabled" if same else "normal"
        self._ups_output_entry.configure(state=state)
        self._ups_out_btn.configure(state=state)
        if same:
            self._ups_sync_output()

    def _ups_sync_output(self, *_):
        if not hasattr(self, "_ups_same_folder") or not self._ups_same_folder.get():
            return
        inp = self._ups_input_var.get().strip()
        if inp:
            folder = inp if os.path.isdir(inp) else os.path.dirname(inp)
            self._ups_output_var.set(folder)

    def _ups_on_format_change(self, fmt: str):
        """Enable/disable bit-depth and quality widgets based on selected format."""
        supports_16bit = fmt in ("PNG", "TIFF")
        supports_quality = fmt in ("JPEG", "WEBP")
        bd = self.widgets.get("ups_bitdepth")
        ql = self.widgets.get("ups_quality")
        if bd:
            bd.configure(state="normal" if supports_16bit else "disabled")
            if not supports_16bit:
                bd.set("8 bits")
        if ql:
            ql.configure(state="normal" if supports_quality else "disabled")

    def _ups_update_preview(self, in_path, out_path):
        """Load input and output images into the side-by-side preview."""
        _ensure_pil()
        # Cache paths so <Configure> resize can re-render
        self._ups_last_preview_paths = (in_path, out_path)
        refs = []
        for path, key in [(in_path, "ups_prev_in"), (out_path, "ups_prev_out")]:
            try:
                lbl = self.widgets[key]
                # Use actual rendered widget dimensions; fall back to 480x220
                w = lbl.winfo_width()
                h = lbl.winfo_height()
                if w < 10:
                    w = 480
                if h < 10:
                    h = 220
                img = Image.open(path).convert("RGB")
                img.thumbnail((w, h), Image.LANCZOS)
                ctk_img = ctk.CTkImage(
                    light_image=img, dark_image=img,
                    size=(img.width, img.height))
                refs.append(ctk_img)
                self._ui_update(
                    lbl.configure,
                    image=ctk_img, text="")
            except Exception:
                pass
        if refs:
            self._ups_preview_refs = refs  # keep alive

    def _ups_on_preview_resize(self, _event=None):
        """Re-render preview at new size when the pane is resized."""
        paths = getattr(self, "_ups_last_preview_paths", None)
        if paths:
            self._ups_update_preview(*paths)

    # ==========================================
    # PAGE 2: QUICK UPSCALE
    # ==========================================
    def create_page_upscale(self):
        from tkinter import StringVar
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")

        # ── En-tête + GPU (droite, retrait 1 cm bord droit) ────────────────────
        _top = ctk.CTkFrame(f, fg_color="transparent")
        _top.pack(fill="x", pady=(0, 15))
        # GPU à droite — 38 px ≈ 1 cm de marge droite
        self._create_gpu_panel(_top).pack(side="right", padx=(0, 0), pady=4)
        _hdr = ctk.CTkFrame(_top, fg_color="transparent")
        _hdr.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(_hdr, text="Quick Upscale", font=("Roboto", 24, "bold"),
                     text_color="#3B8ED0", anchor="w").pack(fill="x")
        ctk.CTkLabel(_hdr, text="Upscaler une image ou un dossier avec un modele entraine.",
                     font=("Arial", 12), text_color="gray", anchor="w").pack(fill="x")

        # --- Model ---
        mrow = ctk.CTkFrame(f, fg_color="transparent")
        mrow.pack(fill="x", pady=5)
        ctk.CTkLabel(mrow, text="Modele (.pth/.safetensors/.onnx) :", width=220, anchor="w").pack(side="left")
        self._ups_model_var = StringVar(value=self.settings.get("ups_last_model", ""))
        self._ups_model_entry = ctk.CTkEntry(mrow, textvariable=self._ups_model_var)
        self._ups_model_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets["ups_model"] = self._ups_model_entry
        ctk.CTkButton(mrow, text="...", width=30, command=self._ups_pick_model).pack(side="left")

        # --- Input ---
        irow = ctk.CTkFrame(f, fg_color="transparent")
        irow.pack(fill="x", pady=5)
        ctk.CTkLabel(irow, text="Source (image ou dossier) :", width=200, anchor="w").pack(side="left")
        self._ups_input_var = StringVar(value=self.settings.get("ups_last_input", ""))
        self._ups_input_entry = ctk.CTkEntry(irow, textvariable=self._ups_input_var)
        self._ups_input_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets["ups_input"] = self._ups_input_entry
        ctk.CTkButton(irow, text="Image", width=60, command=self._ups_pick_image).pack(side="left", padx=(0, 2))
        ctk.CTkButton(irow, text="Dossier", width=70, command=self._ups_pick_input_folder).pack(side="left")

        # --- Output ---
        orow = ctk.CTkFrame(f, fg_color="transparent")
        orow.pack(fill="x", pady=5)
        ctk.CTkLabel(orow, text="Dossier sortie :", width=200, anchor="w").pack(side="left")
        self._ups_output_var = StringVar(value=self.settings.get("ups_last_output", ""))
        self._ups_output_entry = ctk.CTkEntry(orow, textvariable=self._ups_output_var)
        self._ups_output_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets["ups_output"] = self._ups_output_entry
        self._ups_out_btn = ctk.CTkButton(orow, text="...", width=30, command=self._ups_pick_output_folder)
        self._ups_out_btn.pack(side="left")

        # --- Output options ---
        optrow = ctk.CTkFrame(f, fg_color="transparent")
        optrow.pack(fill="x", pady=(0, 5))
        self._ups_same_folder = ctk.CTkCheckBox(
            optrow, text="Meme dossier que la source",
            command=self._ups_on_same_folder_toggle)
        self._ups_same_folder.pack(side="left", padx=(200, 15))
        if self.settings.get("ups_same_folder", False):
            self._ups_same_folder.select()
        self._ups_subfolder = ctk.CTkCheckBox(
            optrow, text='Sous-dossier "upscaled/" (garde le nom original)')
        self._ups_subfolder.pack(side="left", padx=(0, 15))
        if self.settings.get("ups_subfolder", True):
            self._ups_subfolder.select()
        self._ups_modelname = ctk.CTkCheckBox(
            optrow, text="Nom du modele dans le fichier")
        self._ups_modelname.pack(side="left")
        if self.settings.get("ups_modelname", False):
            self._ups_modelname.select()

        # Sync input -> output when "same folder" is on
        self._ups_input_var.trace_add("write", self._ups_sync_output)
        self._ups_on_same_folder_toggle()  # apply initial state

        # Save paths to settings immediately on change (not just on run)
        def _save_model(*_):
            v = self._ups_model_var.get().strip()
            if v:
                self.settings.set("ups_last_model", v)
        def _save_input(*_):
            v = self._ups_input_var.get().strip()
            if v:
                self.settings.set("ups_last_input", v)
        def _save_output(*_):
            v = self._ups_output_var.get().strip()
            if v and not (hasattr(self, "_ups_same_folder") and self._ups_same_folder.get()):
                self.settings.set("ups_last_output", v)
        self._ups_model_var.trace_add("write", _save_model)
        self._ups_input_var.trace_add("write", _save_input)
        self._ups_output_var.trace_add("write", _save_output)

        # --- Scale / Tile / AMP ---
        opts = ctk.CTkFrame(f, fg_color="transparent")
        opts.pack(fill="x", pady=10)
        ctk.CTkLabel(opts, text="Scale:").pack(side="left")
        self.widgets["ups_scale"] = ctk.CTkOptionMenu(
            opts, values=["Auto", "1", "2", "3", "4", "8"], width=70)
        self.widgets["ups_scale"].pack(side="left", padx=5)
        self.widgets["ups_scale"].set(self.settings.get("ups_scale", "Auto"))
        ctk.CTkLabel(opts, text="Tile:").pack(side="left", padx=(15, 0))
        self.widgets["ups_tile"] = ctk.CTkOptionMenu(
            opts, values=["Auto", "128", "192", "256", "384", "512", "768", "1024", "0 (pas de tile)"], width=100)
        self.widgets["ups_tile"].pack(side="left", padx=5)
        self.widgets["ups_tile"].set(self.settings.get("ups_tile", "Auto"))
        ToolTip(self.widgets["ups_tile"], "Taille des tuiles. Auto = selon VRAM. 0 = pas de tiling.")
        ctk.CTkLabel(opts, text="Tile Pad:").pack(side="left", padx=(15, 0))
        self.widgets["ups_tile_pad"] = ctk.CTkOptionMenu(
            opts, values=["Auto", "8", "16", "32", "48", "64"], width=80)
        self.widgets["ups_tile_pad"].pack(side="left", padx=5)
        self.widgets["ups_tile_pad"].set(self.settings.get("ups_tile_pad", "Auto"))
        ToolTip(self.widgets["ups_tile_pad"], "Overlap entre les tuiles. Auto = tile/8.")
        self.widgets["ups_amp"] = ctk.CTkCheckBox(opts, text="AMP (FP16)")
        self.widgets["ups_amp"].pack(side="left", padx=15)
        if self.settings.get("ups_amp", True):
            self.widgets["ups_amp"].select()

        # --- Output format / bit depth / quality ---
        fmtrow = ctk.CTkFrame(f, fg_color="transparent")
        fmtrow.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(fmtrow, text="Format sortie :").pack(side="left")
        self.widgets["ups_format"] = ctk.CTkOptionMenu(
            fmtrow, values=["PNG", "JPEG", "WEBP", "TIFF", "BMP"],
            width=90, command=self._ups_on_format_change)
        self.widgets["ups_format"].pack(side="left", padx=(5, 20))
        self.widgets["ups_format"].set(self.settings.get("ups_format", "PNG"))
        ctk.CTkLabel(fmtrow, text="Bits/canal :").pack(side="left")
        self.widgets["ups_bitdepth"] = ctk.CTkOptionMenu(
            fmtrow, values=["8 bits", "16 bits"], width=90)
        self.widgets["ups_bitdepth"].pack(side="left", padx=(5, 20))
        self.widgets["ups_bitdepth"].set(self.settings.get("ups_bitdepth", "8 bits"))
        ToolTip(self.widgets["ups_bitdepth"], "16 bits uniquement pour PNG et TIFF.")
        ctk.CTkLabel(fmtrow, text="Qualité (JPEG/WEBP) :").pack(side="left")
        self.widgets["ups_quality"] = ctk.CTkOptionMenu(
            fmtrow, values=["70", "75", "80", "85", "90", "95", "100"], width=70)
        self.widgets["ups_quality"].pack(side="left", padx=5)
        self.widgets["ups_quality"].set(self.settings.get("ups_quality", "95"))
        ToolTip(self.widgets["ups_quality"], "Qualité de compression pour JPEG et WEBP.")
        # Initial state
        self._ups_on_format_change(self.widgets["ups_format"].get())

        # --- Run / Stop / progress / log ---
        run_row = ctk.CTkFrame(f, fg_color="transparent")
        run_row.pack(fill="x", pady=15)
        ctk.CTkButton(run_row, text="⚡ Lancer Upscale", fg_color="#2ecc71",
                      command=self.run_upscale).pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.widgets["ups_stop_btn"] = ctk.CTkButton(
            run_row, text="⏹ Stop", fg_color="#e74c3c", hover_color="#c0392b",
            width=110, state="disabled", command=self._ups_request_stop)
        self.widgets["ups_stop_btn"].pack(side="left")
        prog_row = ctk.CTkFrame(f, fg_color="transparent")
        prog_row.pack(fill="x")
        self.widgets["prog_ups"] = ctk.CTkProgressBar(prog_row)
        self.widgets["prog_ups"].pack(side="left", fill="x", expand=True)
        self.widgets["prog_ups"].set(0)
        self.widgets["prog_ups_pct"] = ctk.CTkLabel(prog_row, text="0%", width=42, anchor="e")
        self.widgets["prog_ups_pct"].pack(side="left", padx=(6, 0))
        # Timing row: elapsed / ETA / fps (batch only — hidden for single image)
        timing_row = ctk.CTkFrame(f, fg_color="transparent")
        timing_row.pack(fill="x", pady=(1, 0))
        self.widgets["ups_timing"] = ctk.CTkLabel(
            timing_row, text="", font=("Consolas", 11),
            text_color="#888", anchor="w")
        self.widgets["ups_timing"].pack(side="left", padx=2)
        # --- Resizable split: log (top) / preview (bottom) ---
        from tkinter import PanedWindow as _PanedWindow
        paned = _PanedWindow(f, orient="vertical", sashwidth=6, sashrelief="flat",
                             bg="#1a1a1a", bd=0, sashpad=1)
        paned.pack(fill="both", expand=True, pady=5)

        # Top pane: log textbox
        log_host = ctk.CTkFrame(paned, fg_color="transparent")
        _sash_h = int(self.settings.get("ups_log_sash_h", 150))
        paned.add(log_host, height=_sash_h, minsize=50)
        self.widgets["log_ups"] = ctk.CTkTextbox(log_host)
        self.widgets["log_ups"].pack(fill="both", expand=True)

        # Bottom pane: side-by-side preview
        prev_frame = ctk.CTkFrame(paned, fg_color="#111827", corner_radius=6)
        paned.add(prev_frame, minsize=80)
        for side, key, label in [("left", "ups_prev_in", "Source"),
                                  ("right", "ups_prev_out", "Resultat")]:
            col = ctk.CTkFrame(prev_frame, fg_color="transparent")
            col.pack(side=side, fill="both", expand=True, padx=4, pady=4)
            ctk.CTkLabel(col, text=label, font=("Arial", 10), text_color="gray").pack()
            lbl = ctk.CTkLabel(col, text="—", fg_color="#1e293b", corner_radius=4)
            lbl.pack(fill="both", expand=True)
            self.widgets[key] = lbl
        self._ups_preview_refs = []  # keep CTkImage refs alive
        self._ups_last_preview_paths = None
        # Re-render preview whenever the pane is resized (sash drag)
        prev_frame.bind("<Configure>", self._ups_on_preview_resize)
        # Save sash position when user finishes dragging
        paned.bind("<ButtonRelease-1>", lambda e, p=paned: self._ups_save_sash(p))

        return f

    def run_upscale(self):
        _ensure_pil()
        model = self._ups_model_var.get().strip()
        inp = self._ups_input_var.get().strip()
        same_folder = bool(self._ups_same_folder.get())
        use_subfolder = bool(self._ups_subfolder.get())

        base_out = (inp if os.path.isdir(inp) else os.path.dirname(inp)) \
            if same_folder else self._ups_output_var.get().strip()
        actual_out = os.path.join(base_out, "upscaled") if use_subfolder else base_out

        if not model:
            messagebox.showerror("Erreur", "Selectionnez un modele.")
            return
        if not inp:
            messagebox.showerror("Erreur", "Selectionnez une source.")
            return
        if not base_out:
            messagebox.showerror("Erreur", "Selectionnez un dossier de sortie.")
            return

        add_model_name = bool(self._ups_modelname.get())
        self.settings.set("ups_last_model", model)
        self.settings.set("ups_last_input", inp)
        if not same_folder:
            self.settings.set("ups_last_output", base_out)
        self.settings.set("ups_subfolder", use_subfolder)
        self.settings.set("ups_modelname", add_model_name)

        scale_str = self.widgets["ups_scale"].get()
        scale = 0 if scale_str == "Auto" else int(scale_str)
        tile_str = self.widgets["ups_tile"].get()
        if tile_str == "Auto":
            # Auto = tiling intelligent : taille basée sur la VRAM disponible
            # 0 désactive le tiling — ne jamais faire ça en Auto !
            try:
                import torch
                if torch.cuda.is_available():
                    free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                    tile = 512 if free_mb > 5000 else 256
                else:
                    tile = 256
            except Exception:
                tile = 256
        elif tile_str.startswith("0"):
            tile = 0
        else:
            try:
                tile = int(tile_str)
            except ValueError:
                tile = 256
        tile_pad_str = self.widgets["ups_tile_pad"].get()
        if tile_pad_str == "Auto":
            tile_pad = max(tile // 8, 8) if tile > 0 else 32
        else:
            try:
                tile_pad = int(tile_pad_str)
            except ValueError:
                tile_pad = 32
        use_amp = bool(self.widgets["ups_amp"].get())

        # Format / bit depth / quality
        out_format = self.widgets["ups_format"].get()          # "PNG", "JPEG", etc.
        bit_depth = 16 if self.widgets["ups_bitdepth"].get() == "16 bits" else 8
        try:
            quality = int(self.widgets["ups_quality"].get())
        except ValueError:
            quality = 95
        _EXT_MAP = {"PNG": ".png", "JPEG": ".jpg", "JPG": ".jpg",
                    "WEBP": ".webp", "TIFF": ".tif", "BMP": ".bmp"}
        out_ext = _EXT_MAP.get(out_format.upper(), ".png")

        # Persist Scale/Tile/Amp/Format
        self.settings.set("ups_scale", scale_str)
        self.settings.set("ups_tile", tile_str)
        self.settings.set("ups_tile_pad", tile_pad_str)
        self.settings.set("ups_amp", use_amp)
        self.settings.set("ups_format", out_format)
        self.settings.set("ups_bitdepth", self.widgets["ups_bitdepth"].get())
        self.settings.set("ups_quality", str(quality))

        self.widgets["log_ups"].delete("1.0", "end")
        self.widgets["prog_ups"].set(0)
        self.widgets["prog_ups_pct"].configure(text="0%")

        # Reset & enable stop button
        self._ups_stop_flag = threading.Event()
        self.widgets["ups_stop_btn"].configure(state="normal", text="⏹ Stop")

        def callback(msg):
            self._ui_update(self.widgets["log_ups"].insert, "end", msg + "\n")

        # Build output suffix: optional model name + safety "_UP" when no subfolder
        model_tag = f"_{os.path.splitext(os.path.basename(model))[0]}" if add_model_name else ""
        out_suffix = model_tag if use_subfolder else f"{model_tag}_UP"

        def set_progress(v):
            self._ui_update(self.widgets["prog_ups"].set, v)
            self._ui_update(self.widgets["prog_ups_pct"].configure, text=f"{int(v * 100)}%")

        def worker():
            try:
                from src.core.quick_upscale import upscale_image
                os.makedirs(actual_out, exist_ok=True)

                if os.path.isfile(inp):
                    self._ui_update(self.widgets["ups_timing"].configure, text="")
                    base_name = os.path.splitext(os.path.basename(inp))[0]
                    out_path = os.path.join(actual_out, f"{base_name}{out_suffix}{out_ext}")
                    ok, msg = upscale_image(model, inp, out_path, scale=scale,
                                            tile_size=tile, tile_pad=tile_pad,
                                            use_amp=use_amp, callback=callback,
                                            progress_callback=set_progress,
                                            out_format=out_format, bit_depth=bit_depth,
                                            quality=quality,
                                            stop_event=self._ups_stop_flag)
                    if ok:
                        set_progress(1.0)
                        callback(f"-> {out_path}")
                        self._ups_update_preview(inp, out_path)
                        self._ui_update(self._show_toast,
                                        "Upscale terminé",
                                        os.path.basename(out_path),
                                        True, 4500, "notif_win11_upscale")
                    else:
                        callback(f"ERREUR : {msg}")
                        self._play_sound("error", "sound_error_enabled")
                        self._ui_update(self._show_toast, "Upscale échoué", msg,
                                        False, 4500, "notif_win11_errors")

                elif os.path.isdir(inp):
                    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
                    files = sorted([fn for fn in os.listdir(inp)
                                    if os.path.splitext(fn)[1].lower() in exts])
                    total = len(files)
                    if total == 0:
                        callback("Aucune image trouvee dans le dossier.")
                        self._play_sound("warning", "sound_warning_enabled")
                        self._ui_update(self._show_toast,
                                        "Aucune image trouvée", inp,
                                        False, 4500, "notif_win11_errors")
                        return
                    success = 0
                    errors = []
                    import time as _time
                    _batch_start = _time.monotonic()
                    _img_times = []  # per-image durations for fps rolling average

                    def _update_timing(done, total_imgs, img_dur_s):
                        """Compute and push elapsed / ETA / fps to the timing label."""
                        _img_times.append(img_dur_s)
                        # Rolling average over last 5 images for smoother ETA
                        _recent = _img_times[-5:]
                        avg_s = sum(_recent) / len(_recent)
                        fps = 1.0 / avg_s if avg_s > 0 else 0.0
                        remaining = total_imgs - done
                        eta_s = avg_s * remaining
                        elapsed_s = _time.monotonic() - _batch_start
                        def _fmt(s):
                            s = int(s)
                            h, rem = divmod(s, 3600)
                            m, sec = divmod(rem, 60)
                            return (f"{h}h{m:02d}m{sec:02d}s" if h
                                    else f"{m}m{sec:02d}s" if m
                                    else f"{sec}s")
                        txt = (f"Écoulé : {_fmt(elapsed_s)}   "
                               f"ETA : {_fmt(eta_s)}   "
                               f"Vitesse : {fps:.2f} img/s   "
                               f"[{done}/{total_imgs}]")
                        self._ui_update(self.widgets["ups_timing"].configure, text=txt)

                    # Clear timing label at start
                    self._ui_update(self.widgets["ups_timing"].configure, text="")

                    for i, fname in enumerate(files):
                        if self._ups_stop_flag.is_set():
                            callback("Arrêt demandé par l'utilisateur.")
                            break
                        in_path = os.path.join(inp, fname)
                        base_name = os.path.splitext(fname)[0]
                        out_path = os.path.join(actual_out, f"{base_name}{out_suffix}{out_ext}")
                        callback(f"[{i + 1}/{total}] {fname}")
                        # Per-image progress: map 0-1 within the slice for this image
                        img_start = i / total
                        img_end = (i + 1) / total
                        def _sub_progress(v, s=img_start, e=img_end):
                            set_progress(s + v * (e - s))
                        _t0 = _time.monotonic()
                        ok, msg = upscale_image(model, in_path, out_path, scale=scale,
                                                tile_size=tile, tile_pad=tile_pad,
                                                use_amp=use_amp, callback=callback,
                                                progress_callback=_sub_progress,
                                                out_format=out_format, bit_depth=bit_depth,
                                                quality=quality,
                                                stop_event=self._ups_stop_flag)
                        _img_dur = _time.monotonic() - _t0
                        if ok:
                            success += 1
                            self._ups_update_preview(in_path, out_path)
                        else:
                            errors.append(f"{fname}: {msg}")
                            self._play_sound("warning", "sound_warning_enabled")
                        set_progress(img_end)
                        _update_timing(i + 1, total, _img_dur)

                    # Final elapsed
                    _total_elapsed = _time.monotonic() - _batch_start
                    _elapsed_h, _el_rem = divmod(int(_total_elapsed), 3600)
                    _elapsed_m, _elapsed_s2 = divmod(_el_rem, 60)
                    _el_str = (f"{_elapsed_h}h{_elapsed_m:02d}m{_elapsed_s2:02d}s" if _elapsed_h
                               else f"{_elapsed_m}m{_elapsed_s2:02d}s" if _elapsed_m
                               else f"{_elapsed_s2}s")
                    _fps_final = success / _total_elapsed if _total_elapsed > 0 else 0
                    self._ui_update(self.widgets["ups_timing"].configure,
                                    text=f"Terminé — {_el_str} total, {_fps_final:.2f} img/s moy.")

                    summary = f"Termine : {success}/{total} images traitees."
                    callback(summary)
                    for err in errors:
                        callback(f"  ERREUR : {err}")
                    toast_sub = f"{len(errors)} erreur(s)" if errors else ""
                    self._ui_update(self._show_toast,
                                    f"Batch : {success}/{total} OK",
                                    toast_sub,
                                    success == total, 4500, "notif_win11_batch")
                else:
                    callback(f"Chemin introuvable : {inp}")
                    self._play_sound("error", "sound_error_enabled")
                    self._ui_update(self._show_toast,
                                    "Chemin introuvable", inp,
                                    False, 4500, "notif_win11_errors")
            except ImportError:
                callback("Module quick_upscale non disponible (PyTorch requis)")
                self._play_sound("error", "sound_error_enabled")
                self._ui_update(self._show_toast,
                                "Module manquant", "PyTorch requis",
                                False, 4500, "notif_win11_errors")
            except Exception as e:
                _err = str(e)  # capture before Python 3.12+ deletes 'e'
                callback(f"Erreur : {_err}")
                self._play_sound("error", "sound_error_enabled")
                self._ui_update(self._show_toast, "Erreur inattendue", _err,
                                False, 4500, "notif_win11_errors")
            finally:
                # Always re-disable the stop button when worker exits
                self._ui_update(self.widgets["ups_stop_btn"].configure,
                                state="disabled", text="⏹ Stop")

        threading.Thread(target=worker, daemon=True).start()

    # ==========================================
    # PAGE 3: GÉNÉRATEUR LQ (enrichi)
    # ==========================================
    def create_page_generator(self):
        _ensure_pil()
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Générateur Dataset (LQ)", "Créez des données basse qualité avec dégradations optionnelles.")
        self.add_path_row(f, "Source (HQ) :", "gen_hq")
        self.add_path_row(f, "Destination (LQ) :", "gen_lq")

        # Scale + méthode de resize
        p = ctk.CTkFrame(f, fg_color="transparent")
        p.pack(fill="x", pady=5)
        ctk.CTkLabel(p, text="Scale :").pack(side="left")
        self.widgets["gen_scale"] = ctk.CTkOptionMenu(p, values=["2", "3", "4", "6", "8"], width=60)
        self.widgets["gen_scale"].pack(side="left", padx=5)
        self.widgets["gen_scale"].set("4")

        ctk.CTkLabel(p, text="Méthode :").pack(side="left", padx=(15, 0))
        self.widgets["gen_method"] = ctk.CTkOptionMenu(
            p, values=["BICUBIC", "BILINEAR", "LANCZOS", "NEAREST", "BOX"], width=100
        )
        self.widgets["gen_method"].pack(side="left", padx=5)
        self.widgets["gen_method"].set("BICUBIC")

        # Dégradations optionnelles
        self.add_header(f, "Dégradations (optionnel)")
        deg = ctk.CTkFrame(f, fg_color="transparent")
        deg.pack(fill="x", pady=5)

        self.widgets["gen_blur"] = ctk.CTkCheckBox(deg, text="Flou gaussien")
        self.widgets["gen_blur"].pack(side="left", padx=5)
        ctk.CTkLabel(deg, text="σ:").pack(side="left")
        self.widgets["gen_blur_sigma"] = ctk.CTkEntry(deg, width=40)
        self.widgets["gen_blur_sigma"].pack(side="left", padx=3)
        self.widgets["gen_blur_sigma"].insert(0, "1.0")

        self.widgets["gen_noise"] = ctk.CTkCheckBox(deg, text="Bruit gaussien")
        self.widgets["gen_noise"].pack(side="left", padx=(15, 5))
        ctk.CTkLabel(deg, text="σ:").pack(side="left")
        self.widgets["gen_noise_sigma"] = ctk.CTkEntry(deg, width=40)
        self.widgets["gen_noise_sigma"].pack(side="left", padx=3)
        self.widgets["gen_noise_sigma"].insert(0, "10")

        deg2 = ctk.CTkFrame(f, fg_color="transparent")
        deg2.pack(fill="x", pady=5)
        self.widgets["gen_jpeg"] = ctk.CTkCheckBox(deg2, text="Compression JPEG")
        self.widgets["gen_jpeg"].pack(side="left", padx=5)
        ctk.CTkLabel(deg2, text="Qualité:").pack(side="left")
        self.widgets["gen_jpeg_q"] = ctk.CTkEntry(deg2, width=40)
        self.widgets["gen_jpeg_q"].pack(side="left", padx=3)
        self.widgets["gen_jpeg_q"].insert(0, "50")

        self.widgets["gen_color_jitter"] = ctk.CTkCheckBox(deg2, text="Color Jitter (±)")
        self.widgets["gen_color_jitter"].pack(side="left", padx=(15, 5))

        # Banding & Posterize (advanced)
        deg3 = ctk.CTkFrame(f, fg_color="transparent")
        deg3.pack(fill="x", pady=5)

        self.widgets["gen_posterize"] = ctk.CTkCheckBox(deg3, text="Posterisation")
        self.widgets["gen_posterize"].pack(side="left", padx=5)
        ctk.CTkLabel(deg3, text="bits:").pack(side="left")
        self.widgets["gen_posterize_bits"] = ctk.CTkEntry(deg3, width=40)
        self.widgets["gen_posterize_bits"].pack(side="left", padx=3)
        self.widgets["gen_posterize_bits"].insert(0, "4")
        ToolTip(self.widgets["gen_posterize"],
            "Reduit la profondeur de bits par canal (1-8).\n"
            "8 = aucun effet, 6 = leger, 4 = visible, 2 = severe.\n"
            "Cree des paliers de couleur.")
        ToolTip(self.widgets["gen_posterize_bits"],
            "Bits par canal RGB :\n"
            "  8 = aucun effet (256 niveaux)\n"
            "  6 = leger (64 niveaux)\n"
            "  5 = visible (32 niveaux)\n"
            "  4 = marque (16 niveaux) — typique compression video\n"
            "  3 = severe (8 niveaux)\n"
            "  2 = extreme (4 niveaux)")

        self.widgets["gen_banding"] = ctk.CTkCheckBox(deg3, text="Banding")
        self.widgets["gen_banding"].pack(side="left", padx=(15, 5))
        ctk.CTkLabel(deg3, text="niveaux:").pack(side="left")
        self.widgets["gen_banding_levels"] = ctk.CTkEntry(deg3, width=40)
        self.widgets["gen_banding_levels"].pack(side="left", padx=3)
        self.widgets["gen_banding_levels"].insert(0, "32")
        ToolTip(self.widgets["gen_banding"],
            "Quantification couleur via dithering reduit.\n"
            "Cree des bandes visibles dans les degrades (ciels, ombres).\n"
            "Simule la compression video aggressive (H.264/HEVC bas bitrate).")
        ToolTip(self.widgets["gen_banding_levels"],
            "Nombre de niveaux totaux de couleur (palette) :\n"
            "  256 = peu visible\n"
            "  128 = leger\n"
            "   64 = visible\n"
            "   32 = marque (recommande)\n"
            "   16 = severe\n"
            "    8 = extreme")

        # Custom 2 degradations (compact — defaults from otf_preview)
        deg4 = ctk.CTkFrame(f, fg_color="transparent")
        deg4.pack(fill="x", pady=5)
        self.widgets["gen_aliasing"] = ctk.CTkCheckBox(deg4, text="Aliasing")
        self.widgets["gen_aliasing"].pack(side="left", padx=5)
        ToolTip(self.widgets["gen_aliasing"], "Nearest-neighbor downscale+upscale → artefacts escalier sur les bords diagonaux.")
        self.widgets["gen_interlace_weave"] = ctk.CTkCheckBox(deg4, text="Interlace weave")
        self.widgets["gen_interlace_weave"].pack(side="left", padx=(15, 5))
        ToolTip(self.widgets["gen_interlace_weave"], "Entrelacement weave : dents de peigne sur les bords (artefact VHS/DVD).")
        self.widgets["gen_interlace_flicker"] = ctk.CTkCheckBox(deg4, text="Flicker")
        self.widgets["gen_interlace_flicker"].pack(side="left", padx=(8, 5))
        ToolTip(self.widgets["gen_interlace_flicker"], "Flicker de champ : lignes paires/impaires à luminosité alternée (CRT).")
        self.widgets["gen_interlace_blend"] = ctk.CTkCheckBox(deg4, text="Field blend")
        self.widgets["gen_interlace_blend"].pack(side="left", padx=(8, 5))
        ToolTip(self.widgets["gen_interlace_blend"], "Ghosting entre champs : flou de mouvement par mélange de fields interlacés.")

        deg5 = ctk.CTkFrame(f, fg_color="transparent")
        deg5.pack(fill="x", pady=5)
        self.widgets["gen_film_grain"] = ctk.CTkCheckBox(deg5, text="Film Grain")
        self.widgets["gen_film_grain"].pack(side="left", padx=5)
        ToolTip(self.widgets["gen_film_grain"], "Grain cinéma luminance-dépendant (fort sur tons moyens, faible sur hautes lumières).")
        self.widgets["gen_oversharp"] = ctk.CTkCheckBox(deg5, text="Oversharpening")
        self.widgets["gen_oversharp"].pack(side="left", padx=(15, 5))
        ToolTip(self.widgets["gen_oversharp"], "Halos USM (sur-netteté), artefact typique des caméras consommateur / vidéo compressée.")
        self.widgets["gen_scanlines"] = ctk.CTkCheckBox(deg5, text="Scanlines CRT")
        self.widgets["gen_scanlines"].pack(side="left", padx=(15, 5))
        ToolTip(self.widgets["gen_scanlines"], "Lignes sombres CRT : assombrit une ligne sur 2-4 (retro games, émulateurs, captures TV).")

        ctk.CTkButton(f, text="Lancer Génération", fg_color="#E67E22", command=self.run_gen).pack(fill="x", pady=15)
        self.widgets["prog_gen"] = ctk.CTkProgressBar(f)
        self.widgets["prog_gen"].pack(fill="x")
        self.widgets["prog_gen"].set(0)
        self.widgets["lbl_gen"] = ctk.CTkLabel(f, text="En attente...")
        self.widgets["lbl_gen"].pack()
        return f

    def run_gen(self):
        hq = self.widgets["gen_hq"].get()
        lq = self.widgets["gen_lq"].get()
        if not hq or not lq:
            messagebox.showerror("Erreur", "Dossiers requis.")
            return
        scale = int(self.widgets["gen_scale"].get())
        method_name = self.widgets["gen_method"].get()
        method_map = {
            "BICUBIC": Image.BICUBIC, "BILINEAR": Image.BILINEAR,
            "LANCZOS": Image.LANCZOS, "NEAREST": Image.NEAREST, "BOX": Image.BOX,
        }
        method = method_map.get(method_name, Image.BICUBIC)

        opts = {
            "scale": scale, "method": method,
            "blur": bool(self.widgets["gen_blur"].get()),
            "blur_sigma": float(self.widgets["gen_blur_sigma"].get() or "1.0"),
            "noise": bool(self.widgets["gen_noise"].get()),
            "noise_sigma": float(self.widgets["gen_noise_sigma"].get() or "10"),
            "jpeg": bool(self.widgets["gen_jpeg"].get()),
            "jpeg_q": int(self.widgets["gen_jpeg_q"].get() or "50"),
            "color_jitter": bool(self.widgets["gen_color_jitter"].get()),
            "posterize": bool(self.widgets["gen_posterize"].get()),
            "posterize_bits": int(self.widgets["gen_posterize_bits"].get() or "4"),
            "banding": bool(self.widgets["gen_banding"].get()),
            "banding_levels": int(self.widgets["gen_banding_levels"].get() or "32"),
            "aliasing": bool(self.widgets["gen_aliasing"].get()),
            "interlace_weave": bool(self.widgets["gen_interlace_weave"].get()),
            "interlace_flicker": bool(self.widgets["gen_interlace_flicker"].get()),
            "interlace_blend": bool(self.widgets["gen_interlace_blend"].get()),
            "film_grain": bool(self.widgets["gen_film_grain"].get()),
            "oversharp": bool(self.widgets["gen_oversharp"].get()),
            "scanlines": bool(self.widgets["gen_scanlines"].get()),
        }
        threading.Thread(target=self._process_gen, args=(hq, lq, opts), daemon=True).start()

    def _process_gen(self, hq, lq, opts):
        import io
        try:
            if not os.path.exists(lq):
                os.makedirs(lq)
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            files = [x for x in os.listdir(hq) if os.path.splitext(x)[1].lower() in exts]
            total = len(files)
            if total == 0:
                self._ui_update(messagebox.showinfo, "Info", "Aucune image trouvée.")
                return
            for i, fname in enumerate(files):
                try:
                    img = Image.open(os.path.join(hq, fname)).convert("RGB")
                    w, h = img.size
                    new_w, new_h = w // opts["scale"], h // opts["scale"]
                    if new_w < 1 or new_h < 1:
                        continue
                    img = img.resize((new_w, new_h), opts["method"])

                    # Apply degradations
                    if opts["blur"]:
                        img = img.filter(ImageFilter.GaussianBlur(radius=opts["blur_sigma"]))

                    if opts["noise"]:
                        _ensure_numpy()
                        arr = np.array(img).astype(np.float32)
                        noise = np.random.normal(0, opts["noise_sigma"], arr.shape)
                        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                        img = Image.fromarray(arr)

                    if opts["color_jitter"]:
                        from PIL import ImageEnhance
                        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
                        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
                        img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))

                    if opts["jpeg"]:
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=opts["jpeg_q"])
                        buf.seek(0)
                        img = Image.open(buf).convert("RGB")

                    if opts.get("posterize"):
                        # Reduce bit-depth per channel — creates flat color regions
                        bits = max(1, min(8, opts["posterize_bits"]))
                        from PIL import ImageOps
                        img = ImageOps.posterize(img, bits)

                    if opts.get("banding"):
                        # Quantize to a small palette — creates banding in gradients
                        levels = max(2, min(256, opts["banding_levels"]))
                        # Use FASTOCTREE to keep speed; convert back to RGB
                        img = img.quantize(colors=levels, method=Image.Quantize.FASTOCTREE).convert("RGB")

                    # Custom 2 degradations — reuse otf_preview functions
                    try:
                        from src.core.otf_preview import (
                            apply_aliasing_pil, apply_interlace_weave_pil,
                            apply_interlace_flicker_pil, apply_interlace_blend_pil,
                            apply_film_grain_pil, apply_oversharpening_pil, apply_scanlines_pil,
                        )
                        if opts.get("aliasing"):
                            img = apply_aliasing_pil(img, (0.65, 0.85))
                        if opts.get("interlace_weave"):
                            img = apply_interlace_weave_pil(img, (0.6, 1.0))
                        if opts.get("interlace_flicker"):
                            img = apply_interlace_flicker_pil(img, (0.1, 0.35))
                        if opts.get("interlace_blend"):
                            img = apply_interlace_blend_pil(img, (0.3, 0.8))
                        if opts.get("film_grain"):
                            img = apply_film_grain_pil(img, (0.04, 0.12), (1, 2))
                        if opts.get("oversharp"):
                            img = apply_oversharpening_pil(img, (0.8, 2.0))
                        if opts.get("scanlines"):
                            img = apply_scanlines_pil(img, (2, 4), (0.25, 0.45))
                    except Exception:
                        pass

                    out_name = os.path.splitext(fname)[0] + ".png"
                    img.save(os.path.join(lq, out_name))
                except Exception:
                    pass

                prog = (i + 1) / total
                self._ui_update(self.widgets["prog_gen"].set, prog)
                self._ui_update(self.widgets["lbl_gen"].configure, text=f"{i+1}/{total}")

            self._ui_update(messagebox.showinfo, "OK", f"Terminé — {total} images traitées.")
        except Exception as e:
            self._ui_update(messagebox.showerror, "Erreur", str(e))

    # ==========================================
    # PAGE 4: CONVERTISSEUR (enrichi)
    # ==========================================
    def create_page_converter(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        # ── En-tête + GPU (droite, retrait 1 cm bord droit) ────────────────────
        _top = ctk.CTkFrame(f, fg_color="transparent")
        _top.pack(fill="x", pady=(0, 15))
        self._create_gpu_panel(_top).pack(side="right", padx=(0, 0), pady=4)
        _hdr = ctk.CTkFrame(_top, fg_color="transparent")
        _hdr.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(_hdr, text="Convertisseur de Modèle", font=("Roboto", 24, "bold"),
                     text_color="#3B8ED0", anchor="w").pack(fill="x")
        ctk.CTkLabel(_hdr, text="Convertir un modèle vers différents formats d'inférence.",
                     font=("Arial", 12), text_color="gray", anchor="w").pack(fill="x")
        self.add_path_row(f, "Modèle source (.pth) :", "conv_pth", is_file=True)
        self.add_path_row(f, "Dossier sortie :", "conv_output")

        # Format options
        self.add_header(f, "Formats de sortie")
        fmt = ctk.CTkFrame(f, fg_color="transparent")
        fmt.pack(fill="x", pady=5)

        self.widgets["chk_onnx"] = ctk.CTkCheckBox(fmt, text="ONNX")
        self.widgets["chk_onnx"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_onnx"], "Export ONNX — compatible avec ONNX Runtime, DirectML.\n[+] Portable, multiplateforme.\n[-] Performances moyennes vs TensorRT.")

        self.widgets["chk_fp16"] = ctk.CTkCheckBox(fmt, text="FP16 (demi-précision)")
        self.widgets["chk_fp16"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_fp16"], "Conversion en Float16.\n[+] Modèle 2x plus petit, inférence plus rapide.\n[-] Légère perte de précision (invisible en pratique).")

        self.widgets["chk_safetensors"] = ctk.CTkCheckBox(fmt, text="SafeTensors")
        self.widgets["chk_safetensors"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_safetensors"], "Conversion vers SafeTensors (Hugging Face).\n[+] Sécurisé (pas d'exécution de code), chargement rapide.\n[+] Standard pour partager des modèles.")

        fmt2 = ctk.CTkFrame(f, fg_color="transparent")
        fmt2.pack(fill="x", pady=5)

        self.widgets["chk_ncnn"] = ctk.CTkCheckBox(fmt2, text="NCNN")
        self.widgets["chk_ncnn"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_ncnn"], "Export NCNN (Tencent) — via ONNX.\n[+] Léger, optimisé pour mobile et CPU.\n[-] Nécessite onnx2ncnn installé séparément.")

        self.widgets["chk_tensorrt"] = ctk.CTkCheckBox(fmt2, text="TensorRT")
        self.widgets["chk_tensorrt"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_tensorrt"], "Export TensorRT (NVIDIA) — via ONNX.\n[+] Performances maximales sur GPU NVIDIA.\n[-] Spécifique à la carte GPU (pas portable).\n[-] Nécessite TensorRT SDK installé.")

        # Paramètres avancés
        self.add_header(f, "Paramètres")
        params = ctk.CTkFrame(f, fg_color="transparent")
        params.pack(fill="x", pady=5)

        ctk.CTkLabel(params, text="Architecture :").pack(side="left")
        self.widgets["conv_arch"] = ctk.CTkOptionMenu(
            params, values=["Auto-detect", "omnisr", "span", "realplksr", "compact", "esrgan", "hat", "dat", "swinir"], width=130
        )
        self.widgets["conv_arch"].pack(side="left", padx=5)
        self.widgets["conv_arch"].set("Auto-detect")

        ctk.CTkLabel(params, text="Scale :").pack(side="left", padx=(15, 0))
        self.widgets["conv_scale"] = ctk.CTkOptionMenu(params, values=["Auto", "1", "2", "3", "4", "8"], width=60)
        self.widgets["conv_scale"].pack(side="left", padx=5)
        self.widgets["conv_scale"].set("Auto")

        ctk.CTkButton(f, text="Convertir", fg_color="#3498db", command=self.run_conv).pack(fill="x", pady=15)
        self.widgets["log_conv"] = ctk.CTkTextbox(f, height=100)
        self.widgets["log_conv"].pack(fill="both", expand=True, pady=5)
        return f

    def run_conv(self):
        pth = self.widgets["conv_pth"].get()
        if not pth or not os.path.exists(pth):
            messagebox.showerror("Erreur", "Modèle introuvable.")
            return

        self.widgets["log_conv"].delete("1.0", "end")
        out_dir = self.widgets["conv_output"].get() or os.path.dirname(pth)
        os.makedirs(out_dir, exist_ok=True)

        do_onnx = bool(self.widgets["chk_onnx"].get())
        do_fp16 = bool(self.widgets["chk_fp16"].get())
        do_safe = bool(self.widgets["chk_safetensors"].get())
        do_ncnn = bool(self.widgets["chk_ncnn"].get())
        do_trt = bool(self.widgets["chk_tensorrt"].get())

        if not any([do_onnx, do_fp16, do_safe, do_ncnn, do_trt]):
            messagebox.showinfo("Info", "Sélectionnez au moins un format.")
            return

        def log(msg):
            self._ui_update(self.widgets["log_conv"].insert, "end", msg + "\n")

        def worker():
            try:
                import torch
                log(f"Chargement : {os.path.basename(pth)}")
                state = torch.load(pth, map_location="cpu", weights_only=False)

                # Extract weights
                for key in ("params_ema", "params_g", "params", "model", "state_dict"):
                    if key in state:
                        state = state[key]
                        break

                base = os.path.splitext(os.path.basename(pth))[0]

                if do_fp16:
                    log("→ Conversion FP16...")
                    fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
                    fp16_path = os.path.join(out_dir, f"{base}_fp16.pth")
                    torch.save(fp16_state, fp16_path)
                    log(f"  ✅ {fp16_path}")

                if do_safe:
                    log("→ Conversion SafeTensors...")
                    try:
                        from safetensors.torch import save_file
                        safe_path = os.path.join(out_dir, f"{base}.safetensors")
                        save_file(state, safe_path)
                        log(f"  ✅ {safe_path}")
                    except ImportError:
                        log("  ❌ safetensors non installé (pip install safetensors)")

                if do_onnx or do_ncnn or do_trt:
                    # Use neosr/redux converter if available
                    py_path = self.settings.get("python_path", "python")
                    cmd = [py_path, "-m", "neosr.utils.convert", "--input", pth]
                    if do_onnx or do_ncnn or do_trt:
                        cmd.append("--onnx")
                    log(f"→ Export ONNX via neosr.utils.convert...")
                    try:
                        creationflags = 0x08000000 if sys.platform == "win32" else 0
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, creationflags=creationflags)
                        if result.returncode == 0:
                            log("  ✅ ONNX exporté")
                        else:
                            log(f"  ⚠️ {result.stderr.strip()[:200]}")
                    except Exception as e:
                        log(f"  ❌ {e}")

                    if do_ncnn:
                        log("→ Pour NCNN : convertissez le fichier .onnx avec 'onnx2ncnn' (outil séparé)")
                        log("  → https://github.com/Tencent/ncnn/wiki/how-to-build")

                    if do_trt:
                        log("→ Pour TensorRT : utilisez 'trtexec --onnx=model.onnx --saveEngine=model.trt'")
                        log("  → Nécessite NVIDIA TensorRT SDK")

                log("✅ Conversion terminée.")
            except Exception as e:
                log(f"❌ Erreur : {e}")

        threading.Thread(target=worker, daemon=True).start()

    # ==========================================
    # PAGE 5: LMDB MAKER
    # ==========================================
    def create_page_lmdb(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Créateur LMDB", "Optimiser les datasets pour la vitesse de lecture.")
        self.add_path_row(f, "Source Images :", "lmdb_src")
        self.add_path_row(f, "Sortie (.lmdb) :", "lmdb_dst")

        btns = ctk.CTkFrame(f, fg_color="transparent")
        btns.pack(fill="x", pady=20)
        self.btn_lmdb = ctk.CTkButton(btns, text="START", fg_color="#2ecc71", command=self.run_lmdb)
        self.btn_lmdb.pack(side="left", fill="x", expand=True, padx=5)
        self.btn_lmdb_stop = ctk.CTkButton(btns, text="STOP", fg_color="#e74c3c", command=self.stop_lmdb, state="disabled")
        self.btn_lmdb_stop.pack(side="left", padx=5)

        self.widgets["prog_lmdb"] = ctk.CTkProgressBar(f)
        self.widgets["prog_lmdb"].pack(fill="x")
        self.widgets["prog_lmdb"].set(0)
        self.widgets["lbl_lmdb"] = ctk.CTkLabel(f, text="En attente...")
        self.widgets["lbl_lmdb"].pack()
        return f

    def run_lmdb(self):
        src = self.widgets["lmdb_src"].get()
        dst = self.widgets["lmdb_dst"].get()
        if not src or not dst:
            messagebox.showerror("Erreur", "Chemins requis.")
            return
        if not dst.endswith(".lmdb"):
            dst += ".lmdb"

        self.btn_lmdb.configure(state="disabled")
        self.btn_lmdb_stop.configure(state="normal")
        py_path = self.settings.get("python_path", "python")

        script_code = f"""import os, sys, cv2, lmdb; src=r"{src}"; dst=r"{dst}"
try:
    files=[f for f in os.listdir(src) if f.lower().endswith(('.png','.jpg'))]
    env=lmdb.open(dst, map_size=1099511627776)
    txn=env.begin(write=True); count=0
    for i,f in enumerate(files):
        img=cv2.imread(os.path.join(src,f), cv2.IMREAD_UNCHANGED); _,buf=cv2.imencode('.png',img)
        txn.put(os.path.splitext(f)[0].encode('ascii'), buf.tobytes()); count+=1
        if count%50==0: txn.commit(); txn=env.begin(write=True); print(f"PROGRESS:{{i}}/{{len(files)}}"); sys.stdout.flush()
    txn.commit(); env.close(); print("DONE")
except Exception as e: print(f"ERROR:{{e}}")
"""
        script_path = os.path.join(os.getcwd(), "temp_lmdb.py")
        with open(script_path, "w") as fw:
            fw.write(script_code)

        def worker():
            try:
                creationflags = 0x08000000 if sys.platform == "win32" else 0
                self.proc_lmdb = subprocess.Popen(
                    [py_path, script_path], stdout=subprocess.PIPE, text=True, creationflags=creationflags
                )
                for line in self.proc_lmdb.stdout:
                    if "PROGRESS" in line:
                        parts = line.strip().split(":")[1].split("/")
                        self._ui_update(self.widgets["prog_lmdb"].set, float(parts[0]) / float(parts[1]))
                        self._ui_update(self.widgets["lbl_lmdb"].configure, text=f"{parts[0]}/{parts[1]}")
                    elif "DONE" in line:
                        self._ui_update(messagebox.showinfo, "Succès", "LMDB Créé !")
            except Exception:
                pass
            finally:
                self._ui_update(self.btn_lmdb.configure, state="normal")
                self._ui_update(self.btn_lmdb_stop.configure, state="disabled")
                if os.path.exists(script_path):
                    os.remove(script_path)

        threading.Thread(target=worker, daemon=True).start()

    def stop_lmdb(self):
        if self.proc_lmdb:
            self.proc_lmdb.kill()

    # ==========================================
    # PAGE 6: METRICS
    # ==========================================
    def create_page_metrics(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Métriques", "Calcul PSNR/SSIM entre deux dossiers.")
        self.add_path_row(f, "Référence (GT) :", "met_ref")
        self.add_path_row(f, "Distorsion (Sortie) :", "met_dist")
        ctk.CTkButton(f, text="Calculer", fg_color="#9b59b6", command=self.run_met).pack(fill="x", pady=20)
        self.widgets["lbl_met"] = ctk.CTkLabel(f, text="Résultat : --", font=("Consolas", 12))
        self.widgets["lbl_met"].pack()
        return f

    def run_met(self):
        ref = self.widgets["met_ref"].get()
        dist = self.widgets["met_dist"].get()
        if not ref or not dist:
            return
        self.widgets["lbl_met"].configure(text="Calcul en cours...")

        def calc():
            try:
                files = os.listdir(ref)
                dist_files = set(os.listdir(dist))
                psnrs = []
                exts = {".png", ".jpg", ".jpeg"}
                for fname in files:
                    if fname in dist_files and os.path.splitext(fname)[1].lower() in exts:
                        _ensure_pil(); _ensure_numpy()
                        i1 = np.array(Image.open(os.path.join(ref, fname)).convert("RGB")).astype(float)
                        i2 = np.array(Image.open(os.path.join(dist, fname)).convert("RGB")).astype(float)
                        mse = np.mean((i1 - i2) ** 2)
                        psnrs.append(20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else 100)
                result = f"PSNR Moyen : {np.mean(psnrs):.2f} dB ({len(psnrs)} images)" if psnrs else "Aucune image commune trouvée"
                self._ui_update(self.widgets["lbl_met"].configure, text=result)
            except Exception as e:
                self._ui_update(self.widgets["lbl_met"].configure, text=f"Erreur: {e}")

        threading.Thread(target=calc, daemon=True).start()

    # ==========================================
    # PAGE 7: CHECKER
    # ==========================================
    def create_page_checker(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Vérificateur Dataset", "Trouver les images corrompues.")
        self.add_path_row(f, "Dossier :", "chk_src")
        ctk.CTkButton(f, text="Scanner", fg_color="#f39c12", command=self.run_chk).pack(fill="x", pady=10)
        self.widgets["log_chk"] = ctk.CTkTextbox(f)
        self.widgets["log_chk"].pack(fill="both", expand=True)
        return f

    def run_chk(self):
        src = self.widgets["chk_src"].get()
        if not src:
            return
        self.widgets["log_chk"].delete("1.0", "end")
        self.widgets["log_chk"].insert("end", f"Scanning {src}...\n")

        def worker():
            cnt = 0
            bad = 0
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            for r, _, fs in os.walk(src):
                for fname in fs:
                    if os.path.splitext(fname)[1].lower() in exts:
                        cnt += 1
                        try:
                            with Image.open(os.path.join(r, fname)) as im:
                                im.verify()
                        except Exception:
                            bad += 1
                            self._ui_update(self.widgets["log_chk"].insert, "end", f"[BAD] {fname}\n")
            self._ui_update(self.widgets["log_chk"].insert, "end", f"\nTerminé. Total: {cnt}, Corrompus: {bad}\n")

        threading.Thread(target=worker, daemon=True).start()

    # ==========================================
    # PAGE 8: HISTORIQUE TRAININGS
    # ==========================================
    def create_page_history(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Historique des Entrainements",
                        "Tous les trainings lances depuis cette app, avec leurs metriques.")

        # Control buttons
        ctrl = ctk.CTkFrame(f, fg_color="transparent")
        ctrl.pack(fill="x", pady=5)
        ctk.CTkButton(ctrl, text="🔄 Rafraichir", fg_color="#3498db", width=120,
                      command=self._refresh_history).pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text="📊 Stats par Architecture", fg_color="#9b59b6", width=200,
                      command=self._show_arch_stats).pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text="💾 Exporter Benchmark TXT", fg_color="#27ae60", width=200,
                      command=self._export_history_txt).pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text="🗑 Tout Supprimer", fg_color="#c0392b", width=160,
                      command=self._delete_all_history).pack(side="right", padx=5)

        # Scrollable list
        self.widgets["history_list"] = ctk.CTkScrollableFrame(f, fg_color="#1a1a2e", height=400)
        self.widgets["history_list"].pack(fill="both", expand=True, pady=10)

        self._refresh_history()
        return f

    def _refresh_history(self):
        from src.core.training_history import get_recent_trainings, format_duration, format_timestamp

        for w in self.widgets["history_list"].winfo_children():
            w.destroy()

        trainings = get_recent_trainings(limit=50)
        if not trainings:
            ctk.CTkLabel(self.widgets["history_list"],
                         text="Aucun training enregistre.\nLes trainings sont automatiquement\nenregistres lorsque vous lancez l'entrainement.",
                         text_color="#666", font=("Roboto", 12), justify="center").pack(pady=40)
            return

        # Header
        hdr = ctk.CTkFrame(self.widgets["history_list"], fg_color="#2B2B4B", corner_radius=4)
        hdr.pack(fill="x", pady=(0, 3))
        for txt, w in [("Nom", 190), ("Arch", 90), ("Iter", 80), ("PSNR", 70),
                       ("Vitesse", 80), ("Duree", 80), ("Date", 120), ("Status", 80), ("", 30)]:
            ctk.CTkLabel(hdr, text=txt, font=("Roboto", 9, "bold"),
                         text_color="#AAA", width=w, anchor="w").pack(side="left", padx=4)

        for t in trainings:
            row = ctk.CTkFrame(self.widgets["history_list"], fg_color="#2B2B3B", corner_radius=4)
            row.pack(fill="x", pady=1)
            status_colors = {"running": "#f39c12", "completed": "#2ecc71",
                             "interrupted": "#e74c3c", "failed": "#c0392b"}
            ctk.CTkLabel(row, text=t["name"][:24], width=190, font=("Roboto", 10, "bold"),
                         text_color="#3498db", anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=str(t.get("architecture", "?"))[:11], width=90,
                         text_color="#999", anchor="w").pack(side="left", padx=4)
            iter_str = f"{t.get('current_iter', 0)}/{t.get('total_iter', 0)}"
            ctk.CTkLabel(row, text=iter_str, width=80, text_color="#999",
                         anchor="w").pack(side="left", padx=4)
            psnr = t.get("best_psnr", 0)
            ctk.CTkLabel(row, text=f"{psnr:.2f}" if psnr else "—", width=70,
                         text_color="#2ecc71" if psnr else "#666", anchor="w").pack(side="left", padx=4)
            spd = t.get("avg_speed", 0)
            ctk.CTkLabel(row, text=f"{spd:.2f} it/s" if spd else "—", width=80,
                         text_color="#f39c12" if spd else "#666", anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=format_duration(t.get("duration_seconds", 0)),
                         width=80, text_color="#AAA", anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=format_timestamp(t.get("started_at", 0)),
                         width=120, text_color="#888", font=("Roboto", 9), anchor="w").pack(side="left", padx=4)
            status = t.get("status", "?")
            ctk.CTkLabel(row, text=status, width=80,
                         text_color=status_colors.get(status, "#888"), anchor="w").pack(side="left", padx=4)
            # Delete button per row
            tid = t["id"]
            ctk.CTkButton(row, text="✕", width=26, height=22, fg_color="#7f1c1c",
                          hover_color="#c0392b", font=("Roboto", 9),
                          command=lambda rid=tid: self._delete_single_history(rid)
                          ).pack(side="left", padx=2)

    def _delete_single_history(self, row_id: int):
        """Delete one history entry with confirmation."""
        from tkinter import messagebox
        from src.core.training_history import delete_training
        if messagebox.askyesno("Supprimer", f"Supprimer l'entrainement #{row_id} ?",
                               icon="warning"):
            delete_training(row_id)
            self._refresh_history()

    def _delete_all_history(self):
        """Delete all history entries with confirmation."""
        from tkinter import messagebox
        from src.core.training_history import delete_all_trainings
        if messagebox.askyesno("Tout Supprimer",
                               "Supprimer TOUT l'historique des entrainements ?\nCette action est irreversible.",
                               icon="warning"):
            delete_all_trainings()
            self._refresh_history()

    def _export_history_txt(self):
        """Export full history as a benchmark TXT file."""
        import tkinter.filedialog as fd
        from src.core.training_history import export_benchmark_txt
        from datetime import datetime
        default_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = fd.asksaveasfilename(
            title="Exporter Benchmark",
            defaultextension=".txt",
            filetypes=[("Fichier texte", "*.txt"), ("Tous", "*.*")],
            initialfile=default_name,
        )
        if not path:
            return
        try:
            content = export_benchmark_txt()
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            from tkinter import messagebox
            messagebox.showinfo("Export OK", f"Benchmark exporte :\n{path}")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Erreur Export", str(e))

    def _show_arch_stats(self):
        from src.core.training_history import get_stats_by_architecture, format_duration

        win = ctk.CTkToplevel(self)
        win.title("Stats par Architecture")
        win.geometry("700x400")
        ctk.CTkLabel(win, text="Performance par Architecture",
                     font=("Roboto", 16, "bold")).pack(pady=10)

        stats = get_stats_by_architecture()
        if not stats:
            ctk.CTkLabel(win, text="Pas assez de donnees.", text_color="#888").pack(pady=40)
            return

        for s in stats:
            row = ctk.CTkFrame(win, fg_color="#1a1a2e", corner_radius=6)
            row.pack(fill="x", padx=20, pady=3)
            ctk.CTkLabel(row, text=s["architecture"], font=("Roboto", 12, "bold"),
                         text_color="#3498db", width=150, anchor="w").pack(side="left", padx=10, pady=5)
            ctk.CTkLabel(row, text=f"Trainings: {s['count']}", width=100,
                         text_color="#AAA").pack(side="left", padx=5)
            ctk.CTkLabel(row, text=f"PSNR moy: {s['avg_psnr']:.2f}", width=130,
                         text_color="#2ecc71").pack(side="left", padx=5)
            ctk.CTkLabel(row, text=f"PSNR max: {s['max_psnr']:.2f}", width=130,
                         text_color="#27ae60").pack(side="left", padx=5)
            ctk.CTkLabel(row, text=f"Duree moy: {format_duration(int(s['avg_duration']))}",
                         text_color="#888").pack(side="left", padx=5)

    # ==========================================
    # PAGE 9: RESUME FAILED TRAININGS
    # ==========================================
    def create_page_resume(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Trainings Interrompus",
                        "Detecte les trainings qui se sont arretes avant la fin et propose de les reprendre.")

        ctk.CTkButton(f, text="🔍 Scanner experiments/", fg_color="#e67e22", width=200,
                      command=self._scan_resume).pack(pady=10)

        self.widgets["resume_list"] = ctk.CTkScrollableFrame(f, fg_color="#1a1a2e", height=400)
        self.widgets["resume_list"].pack(fill="both", expand=True, pady=10)

        ctk.CTkLabel(self.widgets["resume_list"],
                     text="Cliquez sur 'Scanner' pour rechercher les trainings interrompus.",
                     text_color="#888").pack(pady=40)
        return f

    def _scan_resume(self):
        from src.core.resume_failed import scan_interrupted_trainings, find_associated_config

        for w in self.widgets["resume_list"].winfo_children():
            w.destroy()

        ctk.CTkLabel(self.widgets["resume_list"], text="Scan en cours...",
                     text_color="#888").pack(pady=20)
        self.update_idletasks()

        def worker():
            interrupted = scan_interrupted_trainings()
            self._ui_update(self._show_resume_results, interrupted)

        threading.Thread(target=worker, daemon=True).start()

    def _show_resume_results(self, interrupted):
        from src.core.resume_failed import find_associated_config

        for w in self.widgets["resume_list"].winfo_children():
            w.destroy()

        if not interrupted:
            ctk.CTkLabel(self.widgets["resume_list"],
                         text="✅ Aucun training interrompu trouve.\n\nTous vos trainings se sont termines correctement.",
                         text_color="#2ecc71", font=("Roboto", 12)).pack(pady=40)
            return

        ctk.CTkLabel(self.widgets["resume_list"],
                     text=f"⚠ {len(interrupted)} training(s) interrompu(s) detecte(s) :",
                     text_color="#f39c12", font=("Roboto", 12, "bold")).pack(anchor="w", padx=10, pady=10)

        for item in interrupted:
            cfg = find_associated_config(item["path"])
            row = ctk.CTkFrame(self.widgets["resume_list"], fg_color="#2B2B3B", corner_radius=6)
            row.pack(fill="x", padx=5, pady=3)

            top = ctk.CTkFrame(row, fg_color="transparent")
            top.pack(fill="x", padx=10, pady=(8, 2))
            ctk.CTkLabel(top, text=item["name"], font=("Roboto", 12, "bold"),
                         text_color="#3498db").pack(side="left")
            ctk.CTkLabel(top, text=f" [{item['engine']}]", text_color="#666").pack(side="left")
            ctk.CTkLabel(top, text=f"Modifie : {item['mtime_str']}",
                         text_color="#888", font=("Roboto", 9)).pack(side="right")

            mid = ctk.CTkFrame(row, fg_color="transparent")
            mid.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(mid, text=f"Iter atteinte : {item['last_iter']}",
                         text_color="#AAA").pack(side="left")
            if item["state_file"]:
                ctk.CTkLabel(mid, text=f"  ✓ State file disponible",
                             text_color="#2ecc71").pack(side="left", padx=10)
            else:
                ctk.CTkLabel(mid, text=f"  ⚠ Pas de .state",
                             text_color="#e67e22").pack(side="left", padx=10)

            btns = ctk.CTkFrame(row, fg_color="transparent")
            btns.pack(fill="x", padx=10, pady=(2, 8))
            if cfg:
                ctk.CTkLabel(btns, text=f"Config : {os.path.basename(cfg)}",
                             text_color="#888", font=("Roboto", 9)).pack(side="left")
                ctk.CTkButton(btns, text="▶ Reprendre", fg_color="#27ae60", width=120,
                              command=lambda c=cfg: self._resume_training(c)).pack(side="right", padx=2)
            else:
                ctk.CTkLabel(btns, text="(Config non trouvee)",
                             text_color="#e74c3c", font=("Roboto", 9)).pack(side="left")
            ctk.CTkButton(btns, text="📁 Ouvrir dossier", fg_color="#666", width=120,
                          command=lambda p=item["path"]: self._open_folder(p)).pack(side="right", padx=2)

    def _resume_training(self, config_path):
        from tkinter import messagebox
        if messagebox.askyesno("Reprendre", f"Lancer l'entrainement avec la config :\n{os.path.basename(config_path)} ?"):
            try:
                app = self.winfo_toplevel()
                if hasattr(app, "train_tab") and app.train_tab:
                    app.train_tab.external_start(config_path)
                    # Switch to training tab
                    if hasattr(app, "tab_view"):
                        for tab_name in app.tab_view._tab_dict.keys():
                            if "ntrain" in tab_name.lower() or "Entra" in tab_name:
                                app.tab_view.set(tab_name)
                                break
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de lancer : {e}")

    def _open_folder(self, path):
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            print(f"Open folder error: {e}")

    # ==========================================
    # PAGE 10: APERCU DEGRADATIONS OTF
    # ==========================================
    def create_page_otf_preview(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Apercu Degradations OTF",
                        "Genere des images LQ d'exemple en appliquant les degradations OTF actuelles.")

        ctrl = ctk.CTkFrame(f, fg_color="transparent")
        ctrl.pack(fill="x", pady=5)

        ctk.CTkLabel(ctrl, text="Image source :").pack(side="left", padx=5)
        self.widgets["otf_src"] = ctk.CTkEntry(ctrl, width=400)
        self.widgets["otf_src"].pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text="...", width=30,
                      command=lambda: self._browse_file(self.widgets["otf_src"])).pack(side="left", padx=2)

        ctk.CTkButton(ctrl, text="Charger config TOML/YML", fg_color="#3498db",
                      command=self._otf_load_config).pack(side="left", padx=10)

        ctrl2 = ctk.CTkFrame(f, fg_color="transparent")
        ctrl2.pack(fill="x", pady=5)
        ctk.CTkLabel(ctrl2, text="Scale :").pack(side="left", padx=5)
        self.widgets["otf_scale"] = ctk.CTkOptionMenu(ctrl2, values=["1", "2", "3", "4", "8"], width=60)
        self.widgets["otf_scale"].pack(side="left", padx=5)
        self.widgets["otf_scale"].set("4")

        ctk.CTkLabel(ctrl2, text="Echantillons :").pack(side="left", padx=15)
        self.widgets["otf_n_samples"] = ctk.CTkOptionMenu(ctrl2, values=["1", "3", "4", "6"], width=60)
        self.widgets["otf_n_samples"].pack(side="left", padx=5)
        self.widgets["otf_n_samples"].set("4")

        ctk.CTkButton(ctrl2, text="🎲 Generer Apercu", fg_color="#9b59b6", width=200,
                      command=self._otf_generate_preview).pack(side="left", padx=15)

        self.widgets["otf_preview_area"] = ctk.CTkScrollableFrame(f, fg_color="#1a1a2e", height=500)
        self.widgets["otf_preview_area"].pack(fill="both", expand=True, pady=10)

        self._otf_config = {}
        ctk.CTkLabel(self.widgets["otf_preview_area"],
                     text="Charger une config + image, puis cliquer 'Generer Apercu'.",
                     text_color="#888").pack(pady=40)
        return f

    def _otf_load_config(self):
        from tkinter import messagebox
        path = filedialog.askopenfilename(filetypes=[("Config", "*.toml *.yml *.yaml")])
        if not path:
            return
        try:
            try:
                import tomllib
            except ImportError:
                import toml as tomllib
            if path.endswith(".toml"):
                with open(path, "rb") as fp:
                    cfg = tomllib.load(fp) if hasattr(tomllib, "load") else tomllib.loads(fp.read().decode())
            else:
                import yaml
                with open(path, "r", encoding="utf-8") as fp:
                    cfg = yaml.safe_load(fp) or {}
            self._otf_config = cfg
            messagebox.showinfo("Config", f"Config chargee :\n{os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _otf_generate_preview(self):
        from tkinter import messagebox
        from PIL import ImageTk
        from src.core.otf_preview import generate_preview_samples

        src = self.widgets["otf_src"].get()
        if not src or not os.path.exists(src):
            messagebox.showerror("Erreur", "Image source non trouvee.")
            return
        if not self._otf_config:
            messagebox.showerror("Erreur", "Chargez d'abord une config TOML/YML.")
            return

        scale = int(self.widgets["otf_scale"].get())
        n = int(self.widgets["otf_n_samples"].get())

        # Clear preview area
        for w in self.widgets["otf_preview_area"].winfo_children():
            w.destroy()

        try:
            samples = generate_preview_samples([src], self._otf_config, scale=scale,
                                                 n_samples_per_image=n)
        except Exception as e:
            ctk.CTkLabel(self.widgets["otf_preview_area"], text=f"Erreur : {e}",
                         text_color="#e74c3c").pack(pady=20)
            return

        if not samples:
            ctk.CTkLabel(self.widgets["otf_preview_area"],
                         text="Aucun echantillon genere.", text_color="#888").pack(pady=20)
            return

        self._otf_photo_refs = []  # Keep refs to avoid GC
        from PIL import Image

        for i, s in enumerate(samples):
            row = ctk.CTkFrame(self.widgets["otf_preview_area"], fg_color="#2B2B3B", corner_radius=6)
            row.pack(fill="x", padx=5, pady=5)
            ctk.CTkLabel(row, text=f"Echantillon {i+1}", font=("Roboto", 11, "bold"),
                         text_color="#9b59b6").pack(anchor="w", padx=10, pady=(5, 0))
            log_text = " | ".join(s["log"]) if s["log"] else "(aucune degradation appliquee)"
            ctk.CTkLabel(row, text=log_text, text_color="#AAA",
                         font=("Roboto", 9), wraplength=900).pack(anchor="w", padx=10)

            img_row = ctk.CTkFrame(row, fg_color="transparent")
            img_row.pack(fill="x", padx=10, pady=5)
            # HQ thumb
            hq_thumb = s["hq_image"].copy()
            hq_thumb.thumbnail((300, 300))
            hq_photo = ImageTk.PhotoImage(hq_thumb)
            self._otf_photo_refs.append(hq_photo)
            import tkinter as tk
            hq_lbl = tk.Label(img_row, image=hq_photo, bg="#2B2B3B")
            hq_lbl.pack(side="left", padx=5)
            ctk.CTkLabel(img_row, text="→", text_color="#9b59b6",
                         font=("Roboto", 16, "bold")).pack(side="left", padx=5)
            # LQ thumb (upscaled to match for display)
            lq_disp = s["lq_image"].copy().resize(hq_thumb.size, Image.NEAREST)
            lq_photo = ImageTk.PhotoImage(lq_disp)
            self._otf_photo_refs.append(lq_photo)
            lq_lbl = tk.Label(img_row, image=lq_photo, bg="#2B2B3B")
            lq_lbl.pack(side="left", padx=5)

    # ==========================================
    # PAGE 11: IMPORT REAL-ESRGAN/SwinIR CONFIG
    # ==========================================
    def create_page_import_config(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Import Config Real-ESRGAN / SwinIR / BasicSR",
                        "Importe une config YAML communautaire et la convertit au format USR Studio.")

        ctrl = ctk.CTkFrame(f, fg_color="transparent")
        ctrl.pack(fill="x", pady=5)
        ctk.CTkLabel(ctrl, text="Fichier YAML :").pack(side="left", padx=5)
        self.widgets["import_src"] = ctk.CTkEntry(ctrl, width=500)
        self.widgets["import_src"].pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text="...", width=30,
                      command=lambda: self._browse_file(self.widgets["import_src"])).pack(side="left", padx=2)
        ctk.CTkButton(ctrl, text="🔍 Analyser", fg_color="#3498db", width=120,
                      command=self._import_analyze).pack(side="left", padx=10)

        self.widgets["import_summary"] = ctk.CTkTextbox(f, height=300, font=("Consolas", 11))
        self.widgets["import_summary"].pack(fill="both", expand=True, pady=10)
        self.widgets["import_summary"].insert("1.0",
            "Selectionnez un fichier .yml/.yaml de config Real-ESRGAN, SwinIR, RCAN, HAT...\n"
            "Cliquez 'Analyser' pour voir le resume.\n\n"
            "Sources de configs :\n"
            "  - https://github.com/xinntao/Real-ESRGAN/tree/master/options\n"
            "  - https://github.com/JingyunLiang/SwinIR/tree/main/options\n"
            "  - https://github.com/XPixelGroup/BasicSR/tree/master/options\n")
        self.widgets["import_summary"].configure(state="disabled")

        return f

    def _import_analyze(self):
        from src.core.config_importer import get_import_summary, import_yaml_config
        from tkinter import messagebox

        path = self.widgets["import_src"].get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Erreur", "Fichier non trouve.")
            return

        summary = get_import_summary(path)
        self.widgets["import_summary"].configure(state="normal")
        self.widgets["import_summary"].delete("1.0", "end")
        self.widgets["import_summary"].insert("1.0",
            f"Resume de l'import :\n\n{summary}\n\n" +
            "─" * 50 + "\n\n" +
            "Vous pouvez ensuite copier les valeurs vers la Configuration manuellement,\n" +
            "ou les utiliser comme reference pour vos tests.")
        self.widgets["import_summary"].configure(state="disabled")

    # ==========================================
    # PAGE 12: INFO MODELE
    # ==========================================
    def create_page_model_info(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Inspection Modele",
                        "Inspecte un fichier .pth/.onnx/.safetensors pour identifier l'architecture, le scale, les modules.")

        ctrl = ctk.CTkFrame(f, fg_color="transparent")
        ctrl.pack(fill="x", pady=5)
        ctk.CTkLabel(ctrl, text="Modele :", width=70, anchor="w").pack(side="left", padx=5)
        self.widgets["mi_path"] = ctk.CTkEntry(ctrl, width=600)
        self.widgets["mi_path"].pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text="...", width=30,
                      command=lambda: self._browse_file(self.widgets["mi_path"])).pack(side="left", padx=2)
        ctk.CTkButton(ctrl, text="🔍 Inspecter", fg_color="#3498db", width=120,
                      command=self._inspect_model).pack(side="left", padx=10)

        # Convert to safetensors button
        conv = ctk.CTkFrame(f, fg_color="transparent")
        conv.pack(fill="x", pady=5)
        ctk.CTkButton(conv, text="💾 Convertir vers .safetensors",
                      fg_color="#16a085", command=self._convert_safetensors).pack(side="left", padx=5)
        ToolTip(conv, "Convertit un .pth en .safetensors (plus securise, charge plus vite)")

        self.widgets["mi_output"] = ctk.CTkTextbox(f, height=480, font=("Consolas", 11),
                                                     fg_color="#0d0d1a", text_color="#dcdcdc")
        self.widgets["mi_output"].pack(fill="both", expand=True, pady=10)
        self.widgets["mi_output"].insert(
            "1.0",
            "Sélectionnez un fichier modèle (.pth, .pt, .onnx, .safetensors)\n"
            "puis cliquez sur 'Inspecter' pour voir :\n"
            "  - Architecture détectée (RRDBNet, SwinIR, RCAN, OmniSR, HAT, SPAN, ...)\n"
            "  - Nombre de paramètres et estimation VRAM\n"
            "  - Top modules par taille\n"
            "  - Précision (FP32/FP16/etc)\n"
            "  - Scale inféré (x2, x3, x4...)\n"
            "  - Métadonnées du checkpoint\n"
        )
        return f

    def _inspect_model(self):
        from src.core.model_export import detect_model_format, format_model_info
        path = self.widgets["mi_path"].get()
        if not path or not os.path.exists(path):
            from tkinter import messagebox
            messagebox.showerror("Erreur", "Fichier non trouve.")
            return
        info = detect_model_format(path)
        text = format_model_info(info)
        self.widgets["mi_output"].delete("1.0", "end")
        self.widgets["mi_output"].insert("1.0", text)

    def _convert_safetensors(self):
        from src.core.model_export import convert_pth_to_safetensors
        from tkinter import messagebox
        path = self.widgets["mi_path"].get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Erreur", "Selectionnez d'abord un .pth.")
            return
        if not path.endswith(".pth") and not path.endswith(".pt"):
            messagebox.showwarning("Attention", "Format source recommande : .pth ou .pt")
            return
        out = filedialog.asksaveasfilename(
            defaultextension=".safetensors",
            initialfile=os.path.splitext(os.path.basename(path))[0] + ".safetensors",
            filetypes=[("SafeTensors", "*.safetensors")]
        )
        if not out:
            return
        ok, msg = convert_pth_to_safetensors(path, out)
        if ok:
            messagebox.showinfo("Conversion", msg)
        else:
            messagebox.showerror("Erreur", msg)

    # ==========================================
    # PAGE 13: GALERIE WEB + PATCH TENSORBOARD
    # ==========================================
    def create_page_gallery(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Galerie Validation & Patch TensorBoard",
                        "Visualisez les images de validation a distance OU activez l'affichage dans TensorBoard.")

        # ─── Section A: HTTP Gallery Server ───
        section_a = ctk.CTkFrame(f, fg_color="#1a1a2e", corner_radius=8)
        section_a.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(section_a, text="A. Serveur Galerie Web (sans toucher a NeoSR)",
                     font=("Roboto", 13, "bold"), text_color="#3498db"
                     ).pack(anchor="w", padx=10, pady=(8, 5))
        ctk.CTkLabel(section_a,
                     text="Lance un mini-serveur HTTP pointe sur un dossier d'images.\n"
                          "Compatible mobile, auto-refresh, zoom click. Optionnel : tunnel Ngrok pour acces distant.",
                     text_color="#AAA", font=("Roboto", 10), justify="left"
                     ).pack(anchor="w", padx=10, pady=(0, 10))

        # Directory picker
        dir_row = ctk.CTkFrame(section_a, fg_color="transparent")
        dir_row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(dir_row, text="Dossier :", width=80, anchor="w").pack(side="left")
        self.widgets["gal_dir"] = ctk.CTkEntry(dir_row, width=500)
        self.widgets["gal_dir"].pack(side="left", padx=5)
        # Try to autofill with last training's visualization dir
        try:
            home = os.path.expanduser("~")
            default_dir = os.path.join(home, "IA_Engine", "neosr", "experiments")
            if os.path.isdir(default_dir):
                # Find most recent experiment
                exps = [(d, os.path.getmtime(os.path.join(default_dir, d)))
                        for d in os.listdir(default_dir)
                        if os.path.isdir(os.path.join(default_dir, d)) and not d.startswith("_")]
                if exps:
                    latest = max(exps, key=lambda x: x[1])[0]
                    vis_dir = os.path.join(default_dir, latest, "visualization")
                    if os.path.isdir(vis_dir):
                        self.widgets["gal_dir"].insert(0, vis_dir)
        except Exception:
            pass
        ctk.CTkButton(dir_row, text="...", width=30,
                      command=lambda: self._browse_dir(self.widgets["gal_dir"])
                      ).pack(side="left", padx=2)

        # Options row
        opt_row = ctk.CTkFrame(section_a, fg_color="transparent")
        opt_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(opt_row, text="Port :", width=80, anchor="w").pack(side="left")
        self.widgets["gal_port"] = ctk.CTkEntry(opt_row, width=80)
        self.widgets["gal_port"].pack(side="left", padx=5)
        self.widgets["gal_port"].insert(0, "8765")
        self.widgets["gal_ngrok"] = ctk.CTkCheckBox(
            opt_row, text="Activer tunnel Ngrok (acces a distance)"
        )
        self.widgets["gal_ngrok"].pack(side="left", padx=20)
        ToolTip(self.widgets["gal_ngrok"],
                "Necessite ngrok installe et authentifie.\n"
                "Donnera une URL publique https://xxxx.ngrok-free.app\n"
                "accessible depuis n'importe quel appareil.")

        # Action buttons
        btn_row = ctk.CTkFrame(section_a, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=8)
        self.widgets["btn_gal_start"] = ctk.CTkButton(
            btn_row, text="▶ Demarrer Serveur", fg_color="#27ae60",
            width=180, command=self._gallery_start
        )
        self.widgets["btn_gal_start"].pack(side="left", padx=5)
        self.widgets["btn_gal_stop"] = ctk.CTkButton(
            btn_row, text="⏹ Arreter", fg_color="#e74c3c",
            width=120, command=self._gallery_stop, state="disabled"
        )
        self.widgets["btn_gal_stop"].pack(side="left", padx=5)
        self.widgets["btn_gal_open"] = ctk.CTkButton(
            btn_row, text="🌐 Ouvrir dans Navigateur", fg_color="#3498db",
            width=200, command=self._gallery_open, state="disabled"
        )
        self.widgets["btn_gal_open"].pack(side="left", padx=5)

        # Status display
        self.widgets["gal_status"] = ctk.CTkLabel(
            section_a, text="Etat : Arrete",
            text_color="#888", anchor="w", justify="left",
            font=("Consolas", 11)
        )
        self.widgets["gal_status"].pack(anchor="w", fill="x", padx=10, pady=(5, 10))

        # QR code area (only if URL is set)
        self.widgets["gal_qr_frame"] = ctk.CTkFrame(section_a, fg_color="transparent")
        self.widgets["gal_qr_frame"].pack(fill="x", padx=10, pady=5)

        # ─── Section B: NeoSR/Redux TB image patch ───
        section_b = ctk.CTkFrame(f, fg_color="#1a1a2e", corner_radius=8)
        section_b.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(section_b, text="B. Patch NeoSR/Redux pour images TensorBoard",
                     font=("Roboto", 13, "bold"), text_color="#9b59b6"
                     ).pack(anchor="w", padx=10, pady=(8, 5))
        ctk.CTkLabel(section_b,
                     text="Injecte un appel tb_logger.add_image() apres chaque imwrite() dans nondist_validation.\n"
                          "Maximum 4 images par validation pour eviter de gonfler les .tfevents.\n"
                          "Idempotent (detection via marqueur), reversible (backup .usr_bak cree).",
                     text_color="#AAA", font=("Roboto", 10), justify="left"
                     ).pack(anchor="w", padx=10, pady=(0, 10))

        # Engine selection
        eng_row = ctk.CTkFrame(section_b, fg_color="transparent")
        eng_row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(eng_row, text="Engine :", width=80, anchor="w").pack(side="left")
        self.widgets["tbp_engine"] = ctk.CTkOptionMenu(
            eng_row, values=["NeoSR", "traiNNer-Redux"], width=200,
            command=lambda x: self._tbp_refresh_status()
        )
        self.widgets["tbp_engine"].pack(side="left", padx=5)
        ctk.CTkButton(eng_row, text="🔄 Verifier statut", fg_color="#666",
                      width=140, command=self._tbp_refresh_status
                      ).pack(side="left", padx=10)

        # Status
        self.widgets["tbp_status"] = ctk.CTkLabel(
            section_b, text="(non verifie)", text_color="#888",
            anchor="w", justify="left", font=("Consolas", 10), wraplength=700
        )
        self.widgets["tbp_status"].pack(anchor="w", fill="x", padx=10, pady=5)

        # Action buttons
        tbp_btn_row = ctk.CTkFrame(section_b, fg_color="transparent")
        tbp_btn_row.pack(fill="x", padx=10, pady=8)
        ctk.CTkButton(tbp_btn_row, text="✅ Appliquer le Patch", fg_color="#27ae60",
                      width=180, command=self._tbp_apply
                      ).pack(side="left", padx=5)
        ctk.CTkButton(tbp_btn_row, text="❌ Retirer le Patch", fg_color="#e74c3c",
                      width=180, command=self._tbp_remove
                      ).pack(side="left", padx=5)

        # Initial status check
        self.after(200, self._tbp_refresh_status)
        return f

    # ─── Gallery server methods ───
    def _gallery_start(self):
        from tkinter import messagebox
        from src.core.gallery_server import get_server

        directory = self.widgets["gal_dir"].get().strip()
        port_str = self.widgets["gal_port"].get().strip()
        with_ngrok = self.widgets["gal_ngrok"].get()

        if not directory or not os.path.isdir(directory):
            messagebox.showerror("Erreur", "Selectionnez un dossier valide.")
            return

        try:
            port = int(port_str) if port_str else 0
        except ValueError:
            messagebox.showerror("Erreur", "Port invalide.")
            return

        srv = get_server()
        result = srv.start(directory, port=port, with_ngrok=bool(with_ngrok))

        if not result.get("ok"):
            messagebox.showerror("Erreur", result.get("error", "Echec inconnu"))
            return

        # Update UI
        self.widgets["btn_gal_start"].configure(state="disabled")
        self.widgets["btn_gal_stop"].configure(state="normal")
        self.widgets["btn_gal_open"].configure(state="normal")

        status_lines = [
            f"✅ Serveur actif",
            f"   Local : {result['local_url']}",
        ]
        if result.get("ngrok_url"):
            status_lines.append(f"   Public : {result['ngrok_url']}")
            self._gallery_show_qr(result["ngrok_url"])
        elif with_ngrok and result.get("ngrok_warning"):
            status_lines.append(f"   ⚠ {result['ngrok_warning']}")
        else:
            self._gallery_show_qr(result["local_url"])

        status_lines.append(f"   Dossier : {directory}")
        self.widgets["gal_status"].configure(
            text="\n".join(status_lines), text_color="#2ecc71"
        )

    def _gallery_stop(self):
        from src.core.gallery_server import get_server
        srv = get_server()
        srv.stop()

        # Clear QR area
        for w in self.widgets["gal_qr_frame"].winfo_children():
            w.destroy()

        self.widgets["btn_gal_start"].configure(state="normal")
        self.widgets["btn_gal_stop"].configure(state="disabled")
        self.widgets["btn_gal_open"].configure(state="disabled")
        self.widgets["gal_status"].configure(text="Etat : Arrete", text_color="#888")

    def _gallery_open(self):
        from src.core.gallery_server import get_server
        import webbrowser
        srv = get_server()
        st = srv.status()
        url = st.get("ngrok_url") or st.get("local_url")
        if url:
            webbrowser.open(url)

    def _gallery_show_qr(self, url):
        # Clear previous
        for w in self.widgets["gal_qr_frame"].winfo_children():
            w.destroy()

        try:
            from src.core.qr_code import generate_qr_image, is_qrcode_available
            if not is_qrcode_available():
                ctk.CTkLabel(
                    self.widgets["gal_qr_frame"],
                    text=f"💡 Installer 'qrcode' pour scanner avec votre tel : pip install qrcode[pil]\n"
                         f"URL : {url}",
                    text_color="#888", justify="left"
                ).pack(anchor="w")
                return
            qr_path = os.path.join(os.path.expanduser("~"), ".usr_studio_qr.png")
            if generate_qr_image(url, qr_path, box_size=6):
                from PIL import Image as _PImage, ImageTk
                img = _PImage.open(qr_path)
                img.thumbnail((180, 180))
                photo = ImageTk.PhotoImage(img)
                self._gal_qr_photo = photo  # Keep ref
                row = ctk.CTkFrame(self.widgets["gal_qr_frame"], fg_color="transparent")
                row.pack(fill="x", pady=5)
                import tkinter as tk
                tk.Label(row, image=photo, bg="#1a1a2e").pack(side="left", padx=5)
                ctk.CTkLabel(row,
                             text=f"📱 Scannez le QR code avec votre telephone\n\nURL : {url}",
                             text_color="#3498db", justify="left", font=("Roboto", 11)
                             ).pack(side="left", padx=15)
        except Exception as e:
            ctk.CTkLabel(self.widgets["gal_qr_frame"],
                         text=f"URL : {url}", text_color="#888"
                         ).pack(anchor="w")

    # ─── TB patch methods ───
    def _get_engine_root(self):
        eng = self.widgets["tbp_engine"].get()
        home = os.path.expanduser("~")
        if "Redux" in eng:
            return os.path.join(home, "IA_Engine", "traiNNer-redux")
        return os.path.join(home, "IA_Engine", "neosr")

    def _tbp_refresh_status(self):
        from src.core.tb_image_patch import get_patch_status
        root = self._get_engine_root()
        status = get_patch_status(root)
        if not status["found"]:
            text = (f"❌ Engine non trouve : {root}\n"
                    f"   (verifie que le dossier IA_Engine existe avec NeoSR/Redux installe)")
            color = "#e74c3c"
        elif status["patched"]:
            text = (f"✅ Patch deja applique\n"
                    f"   Fichier : {status['target_file']}\n"
                    f"   Backup .usr_bak : {'present' if status['backup_exists'] else 'absent'}")
            color = "#2ecc71"
        else:
            text = (f"⚪ Pas patche (pret a l'emploi)\n"
                    f"   Fichier cible : {status['target_file']}")
            color = "#f39c12"
        self.widgets["tbp_status"].configure(text=text, text_color=color)

    def _tbp_apply(self):
        from tkinter import messagebox
        from src.core.tb_image_patch import patch_engine
        root = self._get_engine_root()
        if not os.path.isdir(root):
            messagebox.showerror("Erreur", f"Engine introuvable : {root}")
            return
        if not messagebox.askyesno(
                "Appliquer Patch",
                f"Modifier les fichiers de {os.path.basename(root)} ?\n\n"
                f"Un backup .usr_bak sera cree.\n"
                f"Vous pourrez retirer le patch a tout moment via 'Retirer'."):
            return
        ok, msg, path = patch_engine(root)
        if ok:
            messagebox.showinfo("Patch", f"{msg}\n\nFichier modifie :\n{path}")
        else:
            messagebox.showerror("Erreur", msg)
        self._tbp_refresh_status()

    def _tbp_remove(self):
        from tkinter import messagebox
        from src.core.tb_image_patch import find_validation_file, unpatch_file
        root = self._get_engine_root()
        target = find_validation_file(root)
        if not target:
            messagebox.showerror("Erreur", "Fichier cible introuvable.")
            return
        if not messagebox.askyesno("Retirer Patch", f"Restaurer le fichier original ?\n{target}"):
            return
        ok, msg = unpatch_file(target)
        if ok:
            messagebox.showinfo("Patch", msg)
        else:
            messagebox.showerror("Erreur", msg)
        self._tbp_refresh_status()

    # ─── Publier Modèle ───────────────────────────────────────────────
    def create_page_export(self):
        from tkinter import StringVar
        # Instance state — initialised BEFORE any callback can fire
        self._exp_scan_data = []
        self._exp_selected  = {}
        self._exp_radio_var = StringVar(value="")

        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "📦 Publier Modèle",
                        "Préparer un package propre pour archiver ou partager votre modèle.")

        body = ctk.CTkFrame(f, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # ── LEFT COLUMN : sélecteur de modèle ──────────────────
        left = ctk.CTkFrame(body, fg_color="#1a1a2e", corner_radius=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)

        ctk.CTkLabel(left, text="📂 Sélectionner le Modèle",
                     font=("Roboto", 13, "bold"),
                     text_color="#3498db").pack(anchor="w", padx=10, pady=(8, 4))

        flt_row = ctk.CTkFrame(left, fg_color="transparent")
        flt_row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(flt_row, text="Filtre :", width=50, anchor="w").pack(side="left")
        self._exp_engine_var = StringVar(value="Tous")
        self.widgets["exp_engine_filter"] = ctk.CTkOptionMenu(
            flt_row,
            values=["Tous", "5 derniers", "10 derniers", "NeoSR", "Redux"],
            variable=self._exp_engine_var, width=140,
            command=lambda x: self._export_refresh_list()
        )
        self.widgets["exp_engine_filter"].pack(side="left", padx=5)
        ctk.CTkButton(flt_row, text="🔄 Scanner", width=100,
                      command=self._export_scan).pack(side="left", padx=3)

        self._exp_list_frame = ctk.CTkScrollableFrame(left, height=260, fg_color="#111")
        self._exp_list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        ctk.CTkLabel(self._exp_list_frame,
                     text="(cliquez 🔄 Scanner pour charger)",
                     text_color="#555").pack(anchor="w")

        man_row = ctk.CTkFrame(left, fg_color="transparent")
        man_row.pack(fill="x", padx=10, pady=(4, 8))
        ctk.CTkLabel(man_row, text="Ou parcourir :", width=90, anchor="w").pack(side="left")
        self.widgets["exp_manual_path"] = ctk.CTkEntry(
            man_row, placeholder_text="chemin/vers/modele.safetensors")
        self.widgets["exp_manual_path"].pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(man_row, text="...", width=30,
                      command=self._export_browse_model).pack(side="left")

        # ── RIGHT COLUMN : métadonnées + fiche ─────────────────
        right = ctk.CTkFrame(body, fg_color="#1a1a2e", corner_radius=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)

        ctk.CTkLabel(right, text="📝 Métadonnées & Fiche Technique",
                     font=("Roboto", 13, "bold"),
                     text_color="#3498db").pack(anchor="w", padx=10, pady=(8, 4))

        meta_f = ctk.CTkFrame(right, fg_color="transparent")
        meta_f.pack(fill="x", padx=10, pady=3)
        for lbl, key, ph in [
            ("Nom du modèle :", "exp_name",    "ex: Crysisjim SPANPlus Deband_HARD"),
            ("Version :",       "exp_version", "ex: 1.0"),
            ("Auteur :",        "exp_author",  "ex: Crysisjim"),
        ]:
            r = ctk.CTkFrame(meta_f, fg_color="transparent")
            r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=lbl, width=120, anchor="w").pack(side="left")
            e = ctk.CTkEntry(r, placeholder_text=ph)
            e.pack(side="left", fill="x", expand=True)
            self.widgets[key] = e

        self.widgets["exp_name"].bind(
            "<KeyRelease>", lambda e: self._export_update_preview_lbl())
        self.widgets["exp_version"].bind(
            "<KeyRelease>", lambda e: self._export_update_preview_lbl())

        ctk.CTkLabel(right, text="Notes personnelles :",
                     anchor="w").pack(anchor="w", padx=10, pady=(6, 0))
        self.widgets["exp_notes"] = ctk.CTkTextbox(right, height=60, fg_color="#111")
        self.widgets["exp_notes"].pack(fill="x", padx=10, pady=3)

        self.widgets["exp_config_info"] = ctk.CTkLabel(
            right, text="ℹ Aucun modèle sélectionné",
            text_color="#666", font=("Roboto", 10),
            justify="left", anchor="w", wraplength=380
        )
        self.widgets["exp_config_info"].pack(anchor="w", padx=10, pady=3)

        ctk.CTkLabel(right, text="📄 Fiche Technique du Modèle :",
                     anchor="w", font=("Roboto", 11, "bold")).pack(
            anchor="w", padx=10, pady=(6, 0))
        self.widgets["exp_fiche"] = ctk.CTkTextbox(right, height=200, fg_color="#111")
        self.widgets["exp_fiche"].pack(fill="x", padx=10, pady=3)

        # ── IA config row ──────────────────────────────────
        ia_row = ctk.CTkFrame(right, fg_color="#111827", corner_radius=6)
        ia_row.pack(fill="x", padx=10, pady=(4, 2))
        ctk.CTkLabel(ia_row, text="🤖 IA :", width=42, anchor="w",
                     font=("Roboto", 10)).pack(side="left", padx=(8, 2))
        # Provider names MUST match tab_config keys exactly (used to look up api_key_*)
        _ia_providers = ["OpenRouter (Gratuit)", "GitHub Models (Gratuit)",
                         "Google (Gemini)", "Anthropic (Claude)",
                         "OpenAI (ChatGPT)", "xAI (Grok)", "DeepSeek"]
        self._exp_ia_models = {
            "OpenRouter (Gratuit)":   ["meta-llama/llama-3.3-70b-instruct:free",
                                       "nvidia/nemotron-3-super-120b-a12b:free",
                                       "google/gemma-4-31b-it:free",
                                       "qwen/qwen3-coder:free"],
            "GitHub Models (Gratuit)": ["gpt-4o-mini", "gpt-4o", "DeepSeek-R1",
                                        "Phi-4", "Llama-3.3-70B-Instruct"],
            "Google (Gemini)":        ["gemini-2.5-flash", "gemini-2.5-flash-lite",
                                       "gemini-2.0-flash", "gemini-2.5-pro"],
            "Anthropic (Claude)":     ["claude-haiku-4-5-20251001",
                                       "claude-sonnet-4-6", "claude-opus-4-7"],
            "OpenAI (ChatGPT)":       ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini",
                                       "gpt-4.1", "o3-mini"],
            "xAI (Grok)":             ["grok-3-mini", "grok-3", "grok-3-fast",
                                       "grok-4.3-mini"],
            "DeepSeek":               ["deepseek-chat", "deepseek-reasoner"],
        }
        self.widgets["exp_ia_provider"] = ctk.CTkOptionMenu(
            ia_row, values=_ia_providers, width=180, font=("Roboto", 10),
            command=lambda x: self._exp_ia_provider_changed(x)
        )
        self.widgets["exp_ia_provider"].pack(side="left", padx=4, pady=4)
        self.widgets["exp_ia_provider"].set("OpenRouter (Gratuit)")
        first_models = self._exp_ia_models["OpenRouter (Gratuit)"]
        self.widgets["exp_ia_model"] = ctk.CTkOptionMenu(
            ia_row, values=first_models, width=230, font=("Roboto", 10))
        self.widgets["exp_ia_model"].pack(side="left", padx=4, fill="x", expand=True)
        self.widgets["exp_ia_model"].set(first_models[0])

        fiche_btns = ctk.CTkFrame(right, fg_color="transparent")
        fiche_btns.pack(fill="x", padx=10, pady=3)
        ctk.CTkButton(fiche_btns, text="📋 Générer (template)", width=165,
                      command=self._export_generate_template).pack(side="left", padx=(0, 6))
        ctk.CTkButton(fiche_btns, text="🤖 Générer avec IA", width=165,
                      fg_color="#8e44ad",
                      command=self._export_generate_ai).pack(side="left", padx=3)

        self.widgets["exp_dest_lbl"] = ctk.CTkLabel(
            right, text="📁 Destination : (sélectionnez un modèle)",
            text_color="#888", font=("Roboto", 10), anchor="w", wraplength=390
        )
        self.widgets["exp_dest_lbl"].pack(anchor="w", padx=10, pady=(4, 0))

        self.widgets["exp_export_btn"] = ctk.CTkButton(
            right, text="📦 Exporter le Package",
            fg_color="#16a085", height=38,
            font=("Roboto", 13, "bold"),
            command=self._export_do_export
        )
        self.widgets["exp_export_btn"].pack(fill="x", padx=10, pady=(6, 10))

        ToolTip(self.widgets["exp_export_btn"],
                "📦 Crée le dossier dans Final model/ avec :\n"
                "  • {Nom Version}.safetensors  (modèle renommé)\n"
                "  • Fiche Technique du Modèle.txt\n"
                "  • Option/  →  config d'entraînement\n"
                "  • resume/  →  fichier original conservé\n"
                "  • Trainning state/  →  fichier .state")
        return f

    def _export_scan(self):
        """Scan experiments folders for .safetensors models, sorted newest first."""
        import re as _re
        home = os.path.expanduser("~")
        bases = {
            "NeoSR": os.path.join(home, "IA_Engine", "neosr", "experiments"),
            "Redux":  os.path.join(home, "IA_Engine", "traiNNer-redux", "experiments"),
        }
        found = []
        for engine, exp_root in bases.items():
            if not os.path.isdir(exp_root):
                continue
            engine_root = os.path.join(home, "IA_Engine",
                                       "neosr" if engine == "NeoSR" else "traiNNer-redux")
            for exp_name in os.listdir(exp_root):
                exp_dir = os.path.join(exp_root, exp_name)
                models_dir = os.path.join(exp_dir, "models")
                if not os.path.isdir(models_dir):
                    continue
                # ── Config search (multi-strategy) ─────────────────
                config_path = None
                search_dirs = [
                    exp_dir,
                    os.path.join(engine_root, "options"),
                    os.path.join(engine_root, "options", "train"),
                    os.path.join(engine_root, "options", "test"),
                ]
                # 1) Exact name match
                for ext in (".yaml", ".yml", ".toml"):
                    for sd in search_dirs:
                        c = os.path.join(sd, exp_name + ext)
                        if os.path.isfile(c):
                            config_path = c
                            break
                    if config_path:
                        break
                # 2) Any yaml/toml directly inside exp_dir
                if not config_path:
                    for fn in os.listdir(exp_dir):
                        if fn.endswith((".yaml", ".yml", ".toml")):
                            config_path = os.path.join(exp_dir, fn)
                            break
                # 3) Fuzzy: any yaml in options/train whose name is substring of exp_name
                if not config_path:
                    for opt_dir in (os.path.join(engine_root, "options", "train"),
                                    os.path.join(engine_root, "options")):
                        if not os.path.isdir(opt_dir):
                            continue
                        for fn in os.listdir(opt_dir):
                            if not fn.endswith((".yaml", ".yml", ".toml")):
                                continue
                            stem = os.path.splitext(fn)[0]
                            if stem in exp_name or exp_name in stem:
                                config_path = os.path.join(opt_dir, fn)
                                break
                        if config_path:
                            break
                # ── State files ─────────────────────────────────────
                state_dir = os.path.join(exp_dir, "training_states")
                states = []
                if os.path.isdir(state_dir):
                    states = sorted(
                        f for f in os.listdir(state_dir) if f.endswith(".state"))
                # ── Models ──────────────────────────────────────────
                for mf in os.listdir(models_dir):
                    if not mf.endswith(".safetensors"):
                        continue
                    mpath = os.path.join(models_dir, mf)
                    mtime = os.path.getmtime(mpath) if os.path.isfile(mpath) else 0
                    m = _re.search(r"(\d+)", mf)
                    iter_m = m.group(1) if m else None
                    # Match state by iter number, then fallback to latest
                    state_file = None
                    if iter_m:
                        for s in states:
                            if iter_m in s:
                                state_file = os.path.join(state_dir, s)
                                break
                    if not state_file and states:
                        state_file = os.path.join(state_dir, states[-1])
                    found.append({
                        "engine":      engine,
                        "exp_name":    exp_name,
                        "model_path":  mpath,
                        "model_file":  mf,
                        "config_path": config_path,
                        "state_file":  state_file,
                        "iter":        iter_m or "?",
                        "mtime":       mtime,
                    })
        # Sort by modification time — most recent first
        found.sort(key=lambda x: x["mtime"], reverse=True)
        self._exp_scan_data = found
        self._export_refresh_list()

    def _export_refresh_list(self):
        """Rebuild radio-button list from scan data + filter."""
        import datetime as _dt
        for w in self._exp_list_frame.winfo_children():
            w.destroy()
        flt  = self._exp_engine_var.get()
        # Apply filter
        if flt in ("5 derniers", "10 derniers"):
            n    = int(flt.split()[0])
            data = self._exp_scan_data[:n]          # already sorted newest first
        elif flt in ("NeoSR", "Redux"):
            data = [d for d in self._exp_scan_data if d["engine"] == flt]
        else:
            data = list(self._exp_scan_data)
        if not data:
            ctk.CTkLabel(self._exp_list_frame,
                         text="Aucun modèle trouvé. Cliquez 🔄 Scanner.",
                         text_color="#555").pack(anchor="w")
            return
        for d in data:
            has_cfg   = bool(d.get("config_path"))
            has_state = bool(d.get("state_file"))
            cfg_icon  = "✅" if has_cfg   else "❌"
            sta_icon  = "✅" if has_state else "❌"
            # Format mtime
            try:
                ts = _dt.datetime.fromtimestamp(d["mtime"]).strftime("%d/%m/%y %H:%M")
            except Exception:
                ts = "?"
            size_kb = (os.path.getsize(d["model_path"]) // 1024
                       if os.path.isfile(d["model_path"]) else 0)

            card = ctk.CTkFrame(self._exp_list_frame,
                                fg_color="#1a1a2e", corner_radius=6)
            card.pack(fill="x", pady=3, padx=2)

            ctk.CTkRadioButton(
                card,
                text=f"{d['exp_name']}",
                variable=self._exp_radio_var,
                value=d["model_path"],
                command=lambda _d=d: self._export_on_select(_d),
                font=("Roboto", 11, "bold"),
            ).pack(anchor="w", padx=8, pady=(6, 1))

            ctk.CTkLabel(
                card,
                text=(f"  {d['engine']} · iter {d['iter']} · "
                      f"{size_kb:,} KB · 🕐 {ts}"),
                text_color="#aaa", font=("Roboto", 9)
            ).pack(anchor="w", padx=8)
            ctk.CTkLabel(
                card,
                text=f"  Config {cfg_icon}  État {sta_icon}  — {d['model_file']}",
                text_color="#666", font=("Roboto", 9)
            ).pack(anchor="w", padx=8, pady=(0, 5))

    def _export_on_select(self, d: dict):
        """Pre-fill metadata form when a model is selected via radio button."""
        self._exp_selected = d
        self.widgets["exp_name"].delete(0, "end")
        self.widgets["exp_name"].insert(0, d["exp_name"])
        # Parse config for info label
        if d.get("config_path") and os.path.isfile(d["config_path"]):
            info = self._export_parse_config(d["config_path"])
            txt = (f"Config : {os.path.basename(d['config_path'])} | "
                   f"Arch : {info.get('arch', '?')} | "
                   f"Scale : {info.get('scale', '?')}x | "
                   f"Losses : {info.get('losses_summary', '?')}")
        else:
            txt = "Config introuvable — utilisez le bouton '...' pour la retrouver"
        self.widgets["exp_config_info"].configure(text=txt)
        self._export_update_preview_lbl()

    def _export_browse_model(self):
        """Open file dialog to manually select a .safetensors model."""
        path = filedialog.askopenfilename(
            title="Sélectionner le modèle",
            filetypes=[("SafeTensors", "*.safetensors"), ("Tous", "*.*")]
        )
        if not path:
            return
        self.widgets["exp_manual_path"].delete(0, "end")
        self.widgets["exp_manual_path"].insert(0, path)
        self._exp_radio_var.set("")
        d = {
            "model_path":  path,
            "model_file":  os.path.basename(path),
            "exp_name":    os.path.basename(path).replace(".safetensors", ""),
            "config_path": None,
            "state_file":  None,
            "engine":      "?",
            "iter":        "?",
        }
        # Auto-locate config in parent experiment folder (multi-strategy)
        parent   = os.path.dirname(os.path.dirname(path))  # up from models/
        exp_name = os.path.basename(parent)
        eng_root = os.path.dirname(os.path.dirname(parent))  # up from experiments/
        search_dirs = [
            parent,
            os.path.join(eng_root, "options"),
            os.path.join(eng_root, "options", "train"),
        ]
        # 1) Exact name
        for ext in (".yaml", ".yml", ".toml"):
            for sd in search_dirs:
                c = os.path.join(sd, exp_name + ext)
                if os.path.isfile(c):
                    d["config_path"] = c
                    d["exp_name"]    = exp_name
                    break
            if d["config_path"]:
                break
        # 2) Any yaml inside parent exp dir
        if not d["config_path"]:
            for fn in os.listdir(parent):
                if fn.endswith((".yaml", ".yml", ".toml")):
                    d["config_path"] = os.path.join(parent, fn)
                    d["exp_name"]    = exp_name
                    break
        # 3) Fuzzy options/train match
        if not d["config_path"]:
            for opt_dir in (os.path.join(eng_root, "options", "train"),
                            os.path.join(eng_root, "options")):
                if not os.path.isdir(opt_dir):
                    continue
                for fn in os.listdir(opt_dir):
                    if not fn.endswith((".yaml", ".yml", ".toml")):
                        continue
                    stem = os.path.splitext(fn)[0]
                    if stem in exp_name or exp_name in stem:
                        d["config_path"] = os.path.join(opt_dir, fn)
                        d["exp_name"]    = exp_name
                        break
                if d["config_path"]:
                    break
        self._exp_selected = d
        self._export_on_select(d)

    def _export_update_preview_lbl(self):
        """Refresh the destination path preview label."""
        name = self.widgets["exp_name"].get().strip()
        ver  = self.widgets["exp_version"].get().strip()
        if name:
            folder = f"{name} {ver}".strip()
            dest = os.path.join(
                os.path.expanduser("~"), "IA_Engine", "Final model", folder)
            self.widgets["exp_dest_lbl"].configure(
                text=f"📁 Destination : {dest}")
        else:
            self.widgets["exp_dest_lbl"].configure(
                text="📁 Destination : (entrez un nom de modèle)")

    def _export_parse_config(self, path: str) -> dict:
        """Parse YAML/TOML training config — return key metadata as dict."""
        info = {}
        try:
            if path.endswith(".toml"):
                try:
                    import tomllib
                    with open(path, "rb") as fh:
                        cfg = tomllib.load(fh)
                except ImportError:
                    import tomli
                    with open(path, "rb") as fh:
                        cfg = tomli.load(fh)
            else:
                import yaml
                with open(path, "r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh)
            if not cfg:
                return info
            # Architecture
            net_g = cfg.get("network_g") or cfg.get("network") or {}
            if isinstance(net_g, dict):
                info["arch"]     = net_g.get("type", net_g.get("arch", "?"))
                info["num_feat"] = str(net_g.get("num_feat",
                                                  net_g.get("num_features", "?")))
            else:
                info["arch"] = str(net_g or "?")
            info["scale"] = str(cfg.get("scale", "?"))
            # GAN
            net_d = cfg.get("network_d")
            info["gan"]       = "Oui" if net_d else "Non"
            info["net_d_type"] = (str(net_d.get("type", "?"))
                                  if isinstance(net_d, dict) else "—")
            # Training
            train = cfg.get("train") or {}
            if isinstance(train, dict):
                opt_g = train.get("optim_g") or {}
                if isinstance(opt_g, dict):
                    info["lr"] = str(opt_g.get("lr", "?"))
                else:
                    info["lr"] = str(opt_g or "?")
                info["batch_size"] = str(train.get("batch_size", "?"))
                # Patch size from dataset
                _ds = cfg.get("datasets") or {}
                if isinstance(_ds, dict):
                    _tr_ds = _ds.get("train") or {}
                elif isinstance(_ds, list):
                    _tr_ds = next((d for d in _ds
                                   if isinstance(d, dict)
                                   and "train" in str(d.get("name", "")).lower()), {})
                else:
                    _tr_ds = {}
                info["patch_size"] = str(_tr_ds.get("gt_size",
                                                     _tr_ds.get("patch_size", "?")))
                # Summarise active losses
                losses = []
                for k, v in train.items():
                    if isinstance(v, dict) and ("loss" in k.lower()
                                                or "criterion" in k.lower()):
                        t = v.get("type", "")
                        if t:
                            losses.append(f"{k}({t})")
                    elif "weight" in k.lower() and isinstance(v, (int, float)) and v:
                        losses.append(k.replace("_weight", ""))
                info["losses_summary"] = ", ".join(losses[:5]) or "?"
        except Exception as ex:
            info["parse_error"] = str(ex)
        return info

    def _exp_ia_provider_changed(self, provider: str):
        """Update model combobox values when provider changes."""
        try:
            models = getattr(self, "_exp_ia_models", {}).get(provider, [""])
            w = self.widgets.get("exp_ia_model")
            if w and models:
                w.configure(values=models)
                w.set(models[0])
        except Exception:
            pass

    def _export_generate_template(self):
        """Build Fiche Technique in Discord-community format (no AI)."""
        import datetime
        _DESC_PH = "[À compléter — description libre du modèle, cas d'usage, observations…]"
        d       = self._exp_selected or {}
        name    = self.widgets["exp_name"].get().strip()    or d.get("exp_name", "Modèle")
        version = self.widgets["exp_version"].get().strip() or "1.0"
        author  = self.widgets["exp_author"].get().strip()  or "?"
        notes   = self.widgets["exp_notes"].get("1.0", "end").strip()
        cfg_path = d.get("config_path")
        info = (self._export_parse_config(cfg_path)
                if cfg_path and os.path.isfile(cfg_path) else {})
        iter_str = d.get("iter", "?")
        date_str = datetime.date.today().strftime("%d/%m/%Y")
        # Discord markdown bold format (standard "Enhance Everything!" community)
        fiche = (
            f"**{name} {version}**\n"
            "\n"
            f"**Scale:** {info.get('scale', '?')}\n"
            f"**Architecture:** {info.get('arch', '?')}\n"
            f"**Links:** [À compléter]\n"
            "\n"
            f"**Author:** {author}\n"
            f"**License:** CC BY 4.0\n"
            f"**Purpose:** [À compléter — ex: Deband, Restoration, Super-Resolution]\n"
            f"**Subject:** [À compléter — ex: Live action, Anime, Animation]\n"
            f"**Input Type:** Images\n"
            f"**Date:** {date_str}\n"
            "\n"
            f"**Dataset:** [À compléter]\n"
            f"**Dataset Size:** [À compléter]\n"
            f"**OTF (on the fly augmentations):** No\n"
            f"**Pretrained Model:** No\n"
            f"**Iterations:** {iter_str}\n"
            f"**Batch Size:** {info.get('batch_size', '?')}\n"
            f"**LQ Size:** {info.get('patch_size', '?')}\n"
            "\n"
            "**Description:**\n"
            f"{notes if notes else _DESC_PH}\n"
            "\n"
            "**Showcase:**\n"
            "[lien slow.pics ou image de comparaison]\n"
        )
        self.widgets["exp_fiche"].delete("1.0", "end")
        self.widgets["exp_fiche"].insert("1.0", fiche)

    def _export_generate_ai(self):
        """Generate Fiche Technique via selected AI provider (runs in thread)."""
        # Read provider + model from UI widgets
        provider = self.widgets["exp_ia_provider"].get()
        model    = self.widgets["exp_ia_model"].get().strip()
        if not model:
            messagebox.showerror("Modèle IA", "Sélectionnez un modèle.")
            return
        # Resolve API key — same key format as tab_config: api_key_{provider}
        api_key = self.settings.get(f"api_key_{provider}", "")
        if not api_key:
            messagebox.showerror(
                "Clé API manquante",
                f"Aucune clé sauvegardée pour :\n{provider}\n\n"
                "Pour sauvegarder :\n"
                "1. Onglet Configuration\n"
                "2. Section 'Vérification par IA'\n"
                "3. Sélectionnez le fournisseur + entrez la clé\n"
                "4. Cliquez 'Analyser avec IA' → la clé est sauvegardée."
            )
            return
        d       = self._exp_selected or {}
        name    = self.widgets["exp_name"].get().strip()    or d.get("exp_name", "Modèle")
        version = self.widgets["exp_version"].get().strip() or "1.0"
        author  = self.widgets["exp_author"].get().strip()  or "?"
        notes   = self.widgets["exp_notes"].get("1.0", "end").strip()
        cfg_path = d.get("config_path")
        info = (self._export_parse_config(cfg_path)
                if cfg_path and os.path.isfile(cfg_path) else {})
        iter_str = d.get("iter", "?")
        # Notes are passed verbatim as Description context — IA ne les corrige pas,
        # elle les utilise comme base pour rédiger la section Description.
        prompt = (
            "You are an expert in AI Super-Resolution models. "
            "Generate a model release card in the style of the 'Enhance Everything!' Discord community. "
            "Reply ONLY with the card text, no introduction, no markdown fences.\n\n"
            "Use this EXACT format (bold markdown, English):\n"
            f"**{name} {version}**\n\n"
            "Scale: [value]\n"
            "Architecture: [value]\n"
            "Links: [To fill]\n\n"
            "Author: [value]\n"
            "License: CC BY 4.0\n"
            "Purpose: [infer from name/notes]\n"
            "Subject: [infer from name/notes]\n"
            "Input Type: Images\n"
            "Date: [today]\n\n"
            "Dataset: [infer or 'Custom Dataset']\n"
            "Dataset Size: [unknown if not provided]\n"
            "OTF (on the fly augmentations): No\n"
            "Pretrained Model: No\n"
            "Iterations: [value]\n"
            "Batch Size: [value]\n"
            "LQ Size: [value]\n\n"
            "Description:\n"
            "[write 2-4 sentences based on purpose/notes]\n\n"
            "Showcase:\n"
            "[To fill]\n\n"
            "--- DATA ---\n"
            f"Model name : {name} v{version}\n"
            f"Author     : {author}\n"
            f"Architecture: {info.get('arch', 'unknown')}\n"
            f"Scale      : {info.get('scale', 'unknown')}x\n"
            f"Num feat   : {info.get('num_feat', 'unknown')}\n"
            f"GAN        : {info.get('gan', 'No')}\n"
            f"Discriminator: {info.get('net_d_type', 'none')}\n"
            f"Losses     : {info.get('losses_summary', 'unknown')}\n"
            f"LR         : {info.get('lr', 'unknown')}\n"
            f"Batch size : {info.get('batch_size', 'unknown')}\n"
            f"LQ size    : {info.get('patch_size', 'unknown')}\n"
            f"Iterations : {iter_str}\n"
            f"Personal notes (use as Description context): {notes or 'none'}\n"
        )
        self.widgets["exp_fiche"].delete("1.0", "end")
        self.widgets["exp_fiche"].insert("1.0", f"⏳ Génération IA en cours ({provider})…")

        def _set_fiche(text):
            self.widgets["exp_fiche"].delete("1.0", "end")
            self.widgets["exp_fiche"].insert("1.0", text)

        def worker():
            result = self._export_call_ai(provider, api_key, model, prompt)
            self.after(0, lambda: _set_fiche(result))

        threading.Thread(target=worker, daemon=True).start()

    def _export_call_ai(self, provider: str, api_key: str,
                        model: str, prompt: str) -> str:
        """Provider-agnostic AI call — returns response text."""
        import urllib.request
        import urllib.error
        import json as _json
        headers = {"Content-Type": "application/json",
                   "User-Agent":   "UniversalSRStudio/2.0"}
        if "OpenRouter" in provider:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
            headers["HTTP-Referer"]  = "https://github.com/Universal-SR-Studio"
        elif "GitHub" in provider:
            url = "https://models.github.ai/inference/chat/completions"
            headers["Authorization"]      = f"Bearer {api_key}"
            headers["X-GitHub-Api-Version"] = "2022-11-28"
            headers["Accept"] = "application/vnd.github+json"
        elif "Anthropic" in provider:
            url = "https://api.anthropic.com/v1/messages"
            headers["x-api-key"]         = api_key
            headers["anthropic-version"] = "2023-06-01"
        elif "OpenAI" in provider or "ChatGPT" in provider:
            url = "https://api.openai.com/v1/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
        elif "Google" in provider or "Gemini" in provider:
            url = (f"https://generativelanguage.googleapis.com/v1beta/"
                   f"models/{model}:generateContent?key={api_key}")
            body = _json.dumps(
                {"contents": [{"parts": [{"text": prompt}]}]}).encode()
            req = urllib.request.Request(url, data=body, headers=headers,
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    r = _json.loads(resp.read().decode())
                parts = (r.get("candidates", [{}])[0]
                         .get("content", {}).get("parts", []))
                return parts[0].get("text", str(r)) if parts else str(r)
            except urllib.error.HTTPError as e:
                return f"Erreur HTTP {e.code}: {e.read().decode()[:300]}"
            except Exception as ex:
                return f"Erreur: {ex}"
        elif "xAI" in provider or "Grok" in provider:
            url = "https://api.x.ai/v1/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
        elif "DeepSeek" in provider:
            url = "https://api.deepseek.com/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            return f"Fournisseur non supporté : {provider}"
        # Standard OpenAI-compatible body
        body = _json.dumps({
            "model": model, "max_tokens": 1500,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                r = _json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return f"Erreur HTTP {e.code}: {e.read().decode()[:300]}"
        except Exception as ex:
            return f"Erreur: {ex}"
        if "content" in r and isinstance(r["content"], list):
            return r["content"][0].get("text", str(r))
        if "choices" in r:
            return r["choices"][0]["message"]["content"]
        return str(r)

    def _export_do_export(self):
        """Create the model package in ~/IA_Engine/Final model/."""
        import shutil
        name    = self.widgets["exp_name"].get().strip()
        version = self.widgets["exp_version"].get().strip()
        if not name:
            messagebox.showerror("Erreur", "Entrez un nom de modèle.")
            return
        # Resolve model path (manual browse takes priority)
        d          = self._exp_selected or {}
        model_path = self.widgets["exp_manual_path"].get().strip() or d.get("model_path", "")
        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror("Erreur",
                                 "Aucun modèle sélectionné ou fichier introuvable.")
            return
        fiche_text = self.widgets["exp_fiche"].get("1.0", "end").strip()
        if not fiche_text:
            if not messagebox.askyesno(
                    "Fiche vide",
                    "La fiche technique est vide.\n\nContinuer quand même ?"):
                return
        folder_name = f"{name} {version}".strip()
        dest_root = os.path.join(
            os.path.expanduser("~"), "IA_Engine", "Final model", folder_name)
        if os.path.exists(dest_root):
            if not messagebox.askyesno(
                    "Dossier existant",
                    f"Le dossier existe déjà :\n{dest_root}\n\nÉcraser ?"):
                return
        try:
            os.makedirs(os.path.join(dest_root, "Option"),         exist_ok=True)
            os.makedirs(os.path.join(dest_root, "resume"),         exist_ok=True)
            os.makedirs(os.path.join(dest_root, "Trainning state"), exist_ok=True)
            # 1 — model renamed
            shutil.copy2(model_path,
                         os.path.join(dest_root, f"{folder_name}.safetensors"))
            # 2 — fiche technique
            if fiche_text:
                with open(os.path.join(dest_root, "Fiche Technique du Modèle.txt"),
                          "w", encoding="utf-8") as fh:
                    fh.write(fiche_text)
            # 3 — config to Option/
            cfg_path = d.get("config_path")
            if cfg_path and os.path.isfile(cfg_path):
                shutil.copy2(cfg_path,
                             os.path.join(dest_root, "Option",
                                          os.path.basename(cfg_path)))
            # 4 — original model to resume/
            shutil.copy2(model_path,
                         os.path.join(dest_root, "resume",
                                      os.path.basename(model_path)))
            # 5 — state file
            state_file = d.get("state_file")
            state_copied = False
            if state_file and os.path.isfile(state_file):
                shutil.copy2(state_file,
                             os.path.join(dest_root, "Trainning state",
                                          os.path.basename(state_file)))
                state_copied = True
            messagebox.showinfo(
                "✅ Export réussi !",
                f"Package créé dans :\n{dest_root}\n\n"
                f"• {folder_name}.safetensors\n"
                f"• Fiche Technique du Modèle.txt\n"
                f"• Option/{os.path.basename(cfg_path) if cfg_path and os.path.isfile(cfg_path) else '(vide)'}\n"
                f"• resume/{os.path.basename(model_path)}\n"
                f"• Trainning state/{os.path.basename(state_file) if state_copied else '(vide)'}"
            )
            try:
                os.startfile(dest_root)
            except Exception:
                pass
        except Exception as ex:
            messagebox.showerror("Erreur Export", f"Export échoué :\n{ex}")

    # ==========================================
    # PAGE 12: BENCHMARK (Arch / Feature)
    # ==========================================
    def create_page_benchmark(self):
        from pathlib import Path
        self._bench_proc   = None
        self._bench_thread = None

        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        # ── En-tête + GPU (droite, retrait 1 cm bord droit) ────────────────────
        _top = ctk.CTkFrame(f, fg_color="transparent")
        _top.pack(fill="x", pady=(0, 8))
        self._create_gpu_panel(_top).pack(side="right", padx=(0, 0), pady=4)
        _hdr = ctk.CTkFrame(_top, fg_color="transparent")
        _hdr.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(_hdr, text="📈 Benchmark", font=("Roboto", 24, "bold"),
                     text_color="#3B8ED0", anchor="w").pack(fill="x")
        ctk.CTkLabel(_hdr,
                     text="Mesure les it/s, VRAM et stabilité de chaque architecture ou feature.",
                     font=("Arial", 12), text_color="gray", anchor="w").pack(fill="x")

        # ── Top controls (scrollable) ──
        scroll = ctk.CTkScrollableFrame(f, fg_color="transparent", height=340)
        scroll.pack(fill="x", padx=0, pady=(0, 6))

        # Row helper
        def _row(parent, label, widget_fn, tip=""):
            r = ctk.CTkFrame(parent, fg_color="transparent")
            r.pack(fill="x", pady=3)
            ctk.CTkLabel(r, text=label, width=170, anchor="w").pack(side="left")
            w = widget_fn(r)
            if tip:
                ToolTip(w, tip)
            return w

        # ── Moteur ──
        r_engine = ctk.CTkFrame(scroll, fg_color="transparent")
        r_engine.pack(fill="x", pady=3)
        ctk.CTkLabel(r_engine, text="Moteur :", width=170, anchor="w").pack(side="left")
        self.widgets["bench_engine"] = ctk.CTkOptionMenu(
            r_engine, values=["Redux", "NeoSR", "Les deux"],
            width=130, command=self._bench_on_engine_change)
        self.widgets["bench_engine"].set("Redux")
        self.widgets["bench_engine"].pack(side="left")
        ToolTip(self.widgets["bench_engine"],
                "Redux  : traiNNer-redux (YAML, plus d'archs/features).\n"
                "NeoSR  : moteur NeoSR (TOML).\n"
                "Les deux : lance les deux moteurs successivement.\n\n"
                "Chaque moteur utilise son propre venv Python.")

        # ── Type de benchmark ──
        r_type = ctk.CTkFrame(scroll, fg_color="transparent")
        r_type.pack(fill="x", pady=3)
        ctk.CTkLabel(r_type, text="Type de benchmark :", width=170, anchor="w").pack(side="left")
        self.widgets["bench_type"] = ctk.CTkOptionMenu(
            r_type, values=["Architectures", "Features", "Arch + Features"],
            width=150, command=self._bench_on_type_change)
        self.widgets["bench_type"].set("Architectures")
        self.widgets["bench_type"].pack(side="left")
        ToolTip(self.widgets["bench_type"],
                "Architectures : mesure it/s et VRAM de chaque architecture.\n"
                "  → Idéal pour choisir la meilleure arch pour votre GPU.\n\n"
                "Features : teste l'impact de chaque loss/optimiseur/scheduler.\n"
                "  → Idéal pour optimiser votre config d'entraînement.\n\n"
                "Arch + Features : lance les deux suites successivement.\n"
                "  → Durée totale : ~6-10h selon votre GPU.")

        # ── Itérations ──
        self.widgets["bench_n_iter"] = _row(
            scroll, "Itérations / test :",
            lambda p: (lambda e: (e.insert(0, "2500"), e)[-1])(ctk.CTkEntry(p, width=90)),
            "Nombre d'itérations d'entraînement par test.\n"
            "Arch    → 2500 iters (recommandé, ~4 min/test)\n"
            "Feature → 500 iters  (recommandé, ~1 min/test)\n"
            "Plus bas = rapide mais moins stable.\nPlus haut = précis mais très long.")

        # ── Timeout ──
        self.widgets["bench_timeout"] = _row(
            scroll, "Timeout (s) / test :",
            lambda p: (lambda e: (e.insert(0, "3600"), e)[-1])(ctk.CTkEntry(p, width=90)),
            "Temps maximum autorisé par test avant kill forcé.\n"
            "Arch    → 3600s (1h)\nFeature → 900s  (15min)\n"
            "Si le test dépasse ce seuil, il est marqué 'timeout' et le benchmark continue.")

        # ── Modes précision (arch Redux seulement) ──
        f_modes = ctk.CTkFrame(scroll, fg_color="transparent")
        f_modes.pack(fill="x", pady=3)
        ctk.CTkLabel(f_modes, text="Modes précision :", width=170, anchor="w").pack(side="left")
        self.widgets["bench_mode_normal"] = ctk.CTkCheckBox(f_modes, text="normal", onvalue="normal", offvalue="")
        self.widgets["bench_mode_fp16"]   = ctk.CTkCheckBox(f_modes, text="fp16",   onvalue="fp16",   offvalue="")
        self.widgets["bench_mode_bf16"]   = ctk.CTkCheckBox(f_modes, text="bf16",   onvalue="bf16",   offvalue="")
        self.widgets["bench_mode_tf32"]   = ctk.CTkCheckBox(f_modes, text="tf32",   onvalue="tf32",   offvalue="")
        for chk in (self.widgets["bench_mode_normal"], self.widgets["bench_mode_bf16"]):
            chk.select()
        for chk in (self.widgets["bench_mode_normal"], self.widgets["bench_mode_fp16"],
                    self.widgets["bench_mode_bf16"],   self.widgets["bench_mode_tf32"]):
            chk.pack(side="left", padx=6)
        ToolTip(self.widgets["bench_mode_normal"],
                "Modes de précision à tester (Redux arch uniquement).\n\n"
                "normal : FP32 — baseline, toujours disponible.\n"
                "fp16   : AMP Float16 — +10-15% sur RTX 2000+.\n"
                "bf16   : AMP BFloat16 — +25-35% sur RTX 3000+ (Ampere).\n"
                "tf32   : fast_matmul TF32 — +5-10% sur RTX 3000+.\n\n"
                "Recommandé GTX 1080 Ti : normal seulement.\n"
                "Recommandé RTX 3070+ : normal + bf16.")
        self._bench_modes_frame = f_modes

        # ── Tests/archs ciblés ──
        f_tests = ctk.CTkFrame(scroll, fg_color="transparent")
        f_tests.pack(fill="x", pady=3)
        ctk.CTkLabel(f_tests, text="Tests ciblés :", width=170, anchor="w").pack(side="left")
        self.widgets["bench_tests"] = ctk.CTkEntry(
            f_tests, width=300,
            placeholder_text="vide = tous  |  ex: compact,span,hat_s")
        self.widgets["bench_tests"].pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f_tests, text="📋", width=30,
                      command=self._bench_list_tests).pack(side="left")
        ToolTip(self.widgets["bench_tests"],
                "Archs ou features à tester, séparés par virgule.\n"
                "Vide = teste TOUT (peut durer plusieurs heures).\n\n"
                "Exemples Arch :\n"
                "  compact, span, hat_s, ultracompact, artcnn_r8f64\n\n"
                "Exemples Feature :\n"
                "  baseline, loss_mse, loss_ssim, optim_adan, loss_perc_conv\n\n"
                "→ Cliquez 📋 pour lister tous les tests disponibles.")

        # Train GT (optional override)
        r_gt = ctk.CTkFrame(scroll, fg_color="transparent")
        r_gt.pack(fill="x", pady=3)
        ctk.CTkLabel(r_gt, text="Dataset train GT :", width=170, anchor="w").pack(side="left")
        self.widgets["bench_gt"] = ctk.CTkEntry(r_gt,
            placeholder_text=str(Path.home() / "IA_Engine/datasets/train/HR"))
        self.widgets["bench_gt"].pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(r_gt, text="...", width=30,
                      command=lambda: self._browse_dir(self.widgets["bench_gt"])
                      ).pack(side="left")

        # Output dir (optional)
        r_out = ctk.CTkFrame(scroll, fg_color="transparent")
        r_out.pack(fill="x", pady=3)
        ctk.CTkLabel(r_out, text="Dossier résultats :", width=170, anchor="w").pack(side="left")
        self.widgets["bench_outdir"] = ctk.CTkEntry(r_out,
            placeholder_text=str(Path.home() / "IA_Engine/benchmark_results"))
        self.widgets["bench_outdir"].pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(r_out, text="...", width=30,
                      command=lambda: self._browse_dir(self.widgets["bench_outdir"])
                      ).pack(side="left")

        # Options row
        f_opts = ctk.CTkFrame(scroll, fg_color="transparent")
        f_opts.pack(fill="x", pady=3)
        self.widgets["bench_reset"] = ctk.CTkCheckBox(
            f_opts, text="Reset état (repart de zéro)",
            onvalue="true", offvalue="false")
        self.widgets["bench_reset"].pack(side="left", padx=4)
        self.widgets["bench_no_upscale"] = ctk.CTkCheckBox(
            f_opts, text="Pas de test upscale",
            onvalue="true", offvalue="false")
        self.widgets["bench_no_upscale"].pack(side="left", padx=12)
        ToolTip(self.widgets["bench_reset"],
                "Ignore l'état sauvegardé et relance tous les tests depuis le début.")
        ToolTip(self.widgets["bench_no_upscale"],
                "Passe le test d'inférence rapide après chaque arch/feature.\n(Arch benchmark uniquement)")

        # ── Action buttons ──
        f_btns = ctk.CTkFrame(f, fg_color="transparent")
        f_btns.pack(fill="x", pady=6)
        self.widgets["bench_run_btn"] = ctk.CTkButton(
            f_btns, text="▶  Lancer le Benchmark",
            fg_color="#2e7d32", hover_color="#1b5e20",
            font=("Arial", 13, "bold"), command=self._bench_run)
        self.widgets["bench_run_btn"].pack(side="left", padx=(0, 8), ipady=4)

        self.widgets["bench_stop_btn"] = ctk.CTkButton(
            f_btns, text="⏹ Stop",
            fg_color="#b71c1c", hover_color="#7f0000",
            state="disabled", command=self._bench_stop)
        self.widgets["bench_stop_btn"].pack(side="left", padx=(0, 8), ipady=4)

        ctk.CTkButton(
            f_btns, text="📂 Ouvrir résultats",
            fg_color="#1565c0", hover_color="#0d47a1",
            command=self._bench_open_results).pack(side="left", ipady=4)

        ctk.CTkButton(
            f_btns, text="📋 Lister tests",
            fg_color="transparent", border_width=1,
            command=self._bench_list_tests).pack(side="left", padx=8, ipady=4)

        # Status label
        self.widgets["bench_status"] = ctk.CTkLabel(
            f, text="", text_color="#aaaaaa", font=("Consolas", 11), anchor="w")
        self.widgets["bench_status"].pack(fill="x", padx=2, pady=(0, 4))

        # ── Live log ──
        self.widgets["bench_log"] = ctk.CTkTextbox(
            f, font=("Consolas", 10), fg_color="#0a0a0a",
            text_color="#cccccc", state="disabled")
        self.widgets["bench_log"].pack(fill="both", expand=True)

        return f

    # ── Benchmark callbacks ──────────────────────────────────────────────────────

    def _bench_on_engine_change(self, val):
        """Ajuste les défauts selon le moteur choisi."""
        pass  # engines use same script interface; defaults already set

    def _bench_on_type_change(self, val):
        """Ajuste les défauts n_iter / timeout selon le type."""
        if val in ("Architectures", "Arch + Features"):
            # Arch phase domine → défauts arch
            self._bench_set_entry("bench_n_iter", "2500")
            self._bench_set_entry("bench_timeout", "3600")
        else:  # Features
            self._bench_set_entry("bench_n_iter", "500")
            self._bench_set_entry("bench_timeout", "900")

    def _bench_set_entry(self, key, val):
        w = self.widgets.get(key)
        if w:
            w.delete(0, "end")
            w.insert(0, val)

    def _bench_log_write(self, text):
        """Thread-safe log append."""
        def _do():
            log = self.widgets["bench_log"]
            log.configure(state="normal")
            log.insert("end", text + "\n")
            log.see("end")
            log.configure(state="disabled")
        self._ui_update(_do)

    def _bench_status_set(self, text):
        self._ui_update(self.widgets["bench_status"].configure, text=text)

    def _bench_resolve_script(self):
        """Return (python_exe, script_path) — benchmark_runner.py avec le Python courant (UI).
        Les backends trouvent leur propre venv Python via _find_venv_python().
        Utiliser le Python UI évite le faux positif 'REDUX_PYTHON == sys.executable'
        dans les backends (qui vérifient que leur venv Python ≠ Python courant).
        """
        from pathlib import Path
        base   = Path(__file__).parent.parent.parent / "core"
        script = base / "benchmark_runner.py"
        return sys.executable, str(script)

    def _bench_build_cmd(self, python, script, list_only=False):
        """Retourne une liste de commandes à exécuter séquentiellement.

        Gère 'Les deux' (redux + neosr) et 'Arch + Features'.
        Chaque élément de la liste est une commande complète [python, script, ...args].
        benchmark_runner.py accepte UNE combinaison par appel — on le lance N fois si besoin.
        """
        engine_sel = self.widgets["bench_engine"].get()  # "Redux" | "NeoSR" | "Les deux"
        type_sel   = self.widgets["bench_type"].get()    # "Architectures" | "Features" | "Arch + Features"

        # Expansion sélection moteur
        engines = ["redux", "neosr"] if engine_sel == "Les deux" else [engine_sel.lower()]

        # Expansion sélection type
        btypes  = ["arch", "feature"] if type_sel == "Arch + Features" else \
                  ["arch"]            if type_sel == "Architectures"    else \
                  ["feature"]

        if list_only:
            # Liste seulement pour la première combinaison
            return [[python, script,
                     "--engine", engines[0], "--type", btypes[0], "--list"]]

        n_iter  = self.widgets["bench_n_iter"].get().strip()  or "0"
        timeout = self.widgets["bench_timeout"].get().strip() or "0"
        tests   = self.widgets["bench_tests"].get().strip()
        gt      = self.widgets["bench_gt"].get().strip()
        outdir  = self.widgets["bench_outdir"].get().strip()
        reset   = self.widgets["bench_reset"].get() == "true"
        no_ups  = self.widgets["bench_no_upscale"].get() == "true"

        modes_str = ",".join(
            m for m in ("normal", "fp16", "bf16", "tf32")
            if self.widgets.get(f"bench_mode_{m}") and
               self.widgets[f"bench_mode_{m}"].get() == m
        )

        cmds = []
        for eng in engines:
            for btype in btypes:
                cmd = [python, script, "--engine", eng, "--type", btype]
                cmd += ["--n-iter", n_iter, "--timeout", timeout]
                if tests:   cmd += ["--tests",      tests]
                if gt:      cmd += ["--train-gt",   gt]
                if outdir:  cmd += ["--output-dir", outdir]
                if reset:   cmd.append("--reset")
                if no_ups:  cmd.append("--no-upscale")
                if btype == "arch" and modes_str:
                    cmd += ["--modes", modes_str]
                cmds.append(cmd)
        return cmds

    def _bench_run(self):
        if self._bench_proc and self._bench_proc.poll() is None:
            return  # already running

        python, script = self._bench_resolve_script()
        if not os.path.isfile(script):
            messagebox.showerror("Benchmark",
                f"Script introuvable :\n{script}\n\nVérifiez l'installation du moteur.")
            return

        cmds   = self._bench_build_cmd(python, script)
        engine = self.widgets["bench_engine"].get()
        btype  = self.widgets["bench_type"].get()

        # Reset log
        log = self.widgets["bench_log"]
        log.configure(state="normal"); log.delete("1.0", "end"); log.configure(state="disabled")
        self._bench_log_write(f"[Benchmark] {engine} — {btype}")
        if len(cmds) > 1:
            self._bench_log_write(f"[Benchmark] {len(cmds)} phase(s) à exécuter successivement.\n")

        self.widgets["bench_run_btn"].configure(state="disabled")
        self.widgets["bench_stop_btn"].configure(state="normal")
        self._bench_status_set("Benchmark en cours...")

        def _worker():
            try:
                for i, cmd in enumerate(cmds, 1):
                    if len(cmds) > 1:
                        self._bench_log_write(f"\n{'='*60}")
                        # Extrait engine/type de la commande pour l'affichage
                        try:
                            ei = cmd.index("--engine"); ti = cmd.index("--type")
                            phase_label = f"{cmd[ei+1].upper()} / {cmd[ti+1]}"
                        except ValueError:
                            phase_label = f"phase {i}"
                        self._bench_log_write(
                            f"[Benchmark] Phase {i}/{len(cmds)} : {phase_label}")
                        self._bench_log_write(f"{'='*60}")
                    self._bench_log_write(f"[Benchmark] Commande : {' '.join(cmd)}\n")

                    self._bench_proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        bufsize=1)
                    for line in self._bench_proc.stdout:
                        self._bench_log_write(line.rstrip())
                    self._bench_proc.wait()
                    rc = self._bench_proc.returncode

                    if rc != 0:
                        self._bench_status_set(f"Phase {i} terminée avec erreur (code {rc}).")
                        self._bench_log_write(
                            f"\n[Benchmark] ❌ Phase {i} terminée avec erreur (code {rc}). Arrêt.")
                        return  # stop on first error

                    self._bench_log_write(
                        f"\n[Benchmark] ✅ Phase {i}/{len(cmds)} terminée.")

                self._bench_status_set("Benchmark terminé avec succès.")
                self._bench_log_write(
                    "\n[Benchmark] ✅ Toutes les phases terminées — voir les résultats.")
            except Exception as ex:
                self._bench_log_write(f"[Benchmark] ERREUR : {ex}")
                self._bench_status_set(f"Erreur : {ex}")
            finally:
                self._ui_update(self.widgets["bench_run_btn"].configure, state="normal")
                self._ui_update(self.widgets["bench_stop_btn"].configure, state="disabled")

        self._bench_thread = threading.Thread(target=_worker, daemon=True)
        self._bench_thread.start()

    def _bench_stop(self):
        if self._bench_proc and self._bench_proc.poll() is None:
            try:
                import signal
                self._bench_proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                pass
            try:
                self._bench_proc.kill()
            except Exception:
                pass
            self._bench_log_write("[Benchmark] Arrêt demandé par l'utilisateur.")
            self._bench_status_set("Arrêté.")
        self.widgets["bench_stop_btn"].configure(state="disabled")
        self.widgets["bench_run_btn"].configure(state="normal")

    def _bench_open_results(self):
        from pathlib import Path
        outdir = self.widgets["bench_outdir"].get().strip()
        if not outdir:
            engine = self.widgets["bench_engine"].get()
            btype  = self.widgets["bench_type"].get()
            base   = Path.home() / "IA_Engine" / "benchmark_results"
            if engine == "Redux" and btype == "Features":
                outdir = str(base / "redux_feat")
            else:
                outdir = str(base)
        try:
            os.makedirs(outdir, exist_ok=True)
            os.startfile(outdir)
        except Exception as ex:
            messagebox.showerror("Ouvrir résultats", f"Impossible d'ouvrir :\n{outdir}\n\n{ex}")

    def _bench_list_tests(self):
        python, script = self._bench_resolve_script()
        if not os.path.isfile(script):
            messagebox.showerror("Benchmark", f"Script introuvable :\n{script}")
            return
        # list_only retourne une liste d'une seule commande (première combinaison)
        cmd = self._bench_build_cmd(python, script, list_only=True)[0]
        log = self.widgets["bench_log"]
        log.configure(state="normal"); log.delete("1.0", "end"); log.configure(state="disabled")
        self._bench_log_write("[Benchmark] Liste des tests disponibles :\n")

        def _worker():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace")
                for line in proc.stdout:
                    self._bench_log_write(line.rstrip())
                proc.wait()
            except Exception as ex:
                self._bench_log_write(f"Erreur : {ex}")

        threading.Thread(target=_worker, daemon=True).start()
