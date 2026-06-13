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
from src.core.descriptions import TOOLTIPS, get_tooltip
from src.core.settings import SettingsManager


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


def _natural_sort_key(s: str):
    """Tri naturel : 'frame_10.png' > 'frame_2.png' (compare les segments numériques en entier)."""
    import re
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _find_torch_python() -> str:
    """Return path to a Python executable that has torch available.

    Tries in order:
    1. Current process (works when running main.py in a torch env)
    2. TraiNNer-Redux venv
    3. NeoSR venv

    Returns '' if no usable Python with torch is found.
    """
    home = os.path.expanduser("~")
    from src.core import engine_paths as _ep
    candidates = [
        # v2.5.6: shared runtimes/.venv first, then legacy per-engine venvs.
        _ep.shared_venv_python(),
        os.path.join(home, "IA_Engine", "traiNNer-redux", ".venv", "Scripts", "python.exe"),
        os.path.join(home, "IA_Engine", "neosr", ".venv", "Scripts", "python.exe"),
        os.path.join(home, "IA_Engine", "traiNNer-redux", "venv", "Scripts", "python.exe"),
        os.path.join(home, "IA_Engine", "neosr", "venv", "Scripts", "python.exe"),
        os.path.join(home, "IA_Engine", "traiNNer-redux", ".venv", "bin", "python"),
        os.path.join(home, "IA_Engine", "neosr", ".venv", "bin", "python"),
    ]

    # Frozen exe: torch is NOT in-process. Return the first existing engine venv directly.
    # Do NOT run a verification subprocess from here — CWD = _internal/ (PyInstaller chdir)
    # makes `python -c "import torch"` pick up the exe's bundled numpy → ABI clash → false
    # negative. Real subprocess scripts (run with cwd=engine_dir) import torch fine.
    if getattr(sys, "frozen", False):
        for py in candidates:
            if os.path.exists(py):
                return py
        return ""

    # Dev mode: torch may be importable in-process
    try:
        import torch  # noqa: F401
        return sys.executable
    except ImportError:
        pass
    flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
    for py in candidates:
        if not os.path.exists(py):
            continue
        try:
            r = subprocess.run(
                [py, "-c", "import torch; print('ok')"],
                capture_output=True, text=True, timeout=40,
                creationflags=flags
            )
            if r.returncode == 0 and "ok" in r.stdout:
                return py
        except Exception:
            continue
    return ""


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
        # (row 14 is the first row after the 13 nav buttons — v2.5.8 added Post Processing)
        self.frame_nav.grid_rowconfigure(14, weight=1)

        ctk.CTkLabel(self.frame_nav, text=_t("BOÎTE À OUTILS", "TOOLBOX"), font=("Roboto", 20, "bold")).grid(row=0, column=0, padx=20, pady=20)

        # ── Section 1: Visualisation ──
        self.create_nav_btn("🔎 Quick Upscale", 1, "upscale")
        self.create_nav_btn(_t("📊 Comparateur", "📊 Comparator"), 2, "comp")
        # ── Section 2: Datasets ──
        self.create_nav_btn(_t("⚡ Générateur LQ", "⚡ LQ Generator"), 3, "gen")
        self.create_nav_btn(_t("🔄 Convertisseur", "🔄 Converter"), 4, "conv")
        self.create_nav_btn(_t("💾 Créateur LMDB", "💾 LMDB Creator"), 5, "lmdb")
        self.create_nav_btn(_t("🧐 Check Dataset", "🧐 Check Dataset"), 6, "chk")
        # ── Section 3: Modèles ──
        self.create_nav_btn(_t("📏 Métriques", "📏 Metrics"), 7, "met")
        self.create_nav_btn(_t("ℹ Info Modèle", "ℹ Model Info"), 8, "model_info")
        # ── Section 4: Suivi entraînements ──
        self.create_nav_btn(_t("📜 Historique", "📜 History"), 9, "history")
        self.create_nav_btn("♻ Resume Failed", 10, "resume")
        self.create_nav_btn(_t("📦 Publier Modèle", "📦 Publish Model"), 11, "export")
        # ── Section 5: Performance ──
        self.create_nav_btn("📈 Benchmark", 12, "bench")
        # ── Section 6: Post-processing standalone ──
        self.create_nav_btn(_t("⚗ Post Processing", "⚗ Post Processing"), 13, "postproc")

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
            "postproc": self.create_page_postproc(),
        }
        self.show_frame("upscale")

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
        gf = ctk.CTkFrame(parent, fg_color=("#E8E8E8", "#111827"), corner_radius=8)
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
                                     fg_color=("#D0D0D0", "#1e293b"))
            bar.set(0)
            bar.pack(side="left", padx=(3, 4))
            lbl = ctk.CTkLabel(g, text="—", font=("Consolas", 9),
                               width=val_width, anchor="w",
                               text_color=("gray20", "#CBD5E1"))
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
        Retry toutes les 10 s si nvidia-smi absent (ne s'arrête jamais).
        """
        _cnow = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        def _query():
            delay = 2000
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
                # nvidia-smi absent — affiche placeholder, réessaie lentement
                def _cpu():
                    for p in self._gpu_panels:
                        p["temp"].configure(text="—")
                        p["load"].configure(text="—")
                        p["vram"].configure(text="no GPU")
                self._ui_update(_cpu)
                delay = 10000  # retry every 10s when no nvidia-smi
            except Exception:
                pass  # timeout transitoire — réessaie normalement
            finally:
                if getattr(self, "_gpu_polling", False):
                    self._ui_update(self.after, delay, self._gpu_do_poll)

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
        sound_path = os.path.join(
            os.getcwd(),
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
        self.widgets["ups_stop_btn"].configure(state="disabled", text=_t("Arrêt...", "Stopping..."))

    def add_header(self, parent, text, desc=""):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(f, text=text, font=("Roboto", 24, "bold"), text_color="#3B8ED0", anchor="w").pack(fill="x")
        if desc:
            ctk.CTkLabel(f, text=desc, font=("Arial", 12), text_color="gray", anchor="w").pack(fill="x")

    def gen_slider_val(self, parent, key, default, min_v, max_v, step, tip=None):
        """Slider compact + entry pour une valeur d'intensité de dégradation.
        StringVar trace → slider se sync sur tout changement entry (user ou programmatique)."""
        _is_int = isinstance(step, int) or (step == int(step))
        n_steps = max(1, round((max_v - min_v) / step))
        sl = ctk.CTkSlider(parent, from_=min_v, to=max_v, number_of_steps=n_steps, width=85)
        sl.pack(side="left", padx=(2, 0))
        _fmt = (lambda v: str(int(round(v)))) if _is_int else (lambda v: f"{round(v, 3):.3g}")
        var = ctk.StringVar(value=_fmt(float(default)))
        e = ctk.CTkEntry(parent, width=42, textvariable=var)
        e.pack(side="left", padx=(2, 3))
        sl.set(float(default))
        def _sl_cb(v):
            var.set(_fmt(round(float(v) / step) * step))
        def _sync_slider(*_):
            try: sl.set(max(float(min_v), min(float(max_v), float(var.get()))))
            except (ValueError, TypeError): pass
        sl.configure(command=_sl_cb)
        var.trace_add("write", _sync_slider)
        if tip: ToolTip(e, tip)
        self.widgets[key] = e

    def add_path_row(self, parent, label, var_name, is_file=False, save_key=None, initialdir=None):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", pady=5)
        ctk.CTkLabel(f, text=label, width=150, anchor="w").pack(side="left")
        e = ctk.CTkEntry(f)
        e.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets[var_name] = e
        def _browse(_e=e, _f=is_file, _sk=save_key, _d=initialdir):
            _init = _d or (_e.get().strip() or None)
            if _init and not os.path.isdir(_init):
                _init = os.path.dirname(_init)
            if _f:
                d = filedialog.askopenfilename(initialdir=_init)
            else:
                d = filedialog.askdirectory(initialdir=_init)
            if d:
                _e.delete(0, "end"); _e.insert(0, d)
                if _sk: self.settings.set(_sk, d)
        ctk.CTkButton(f, text="...", width=30, command=_browse).pack(side="left")

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
        self.add_header(f, _t("Comparateur Visuel", "Visual Comparator"), _t("Glissez pour comparer LQ (gauche) et HQ (droite).", "Drag to compare LQ (left) and HQ (right)."))

        ctrl = ctk.CTkFrame(f)
        ctrl.pack(fill="x", pady=10, padx=10)
        ctk.CTkButton(ctrl, text=_t("Ouvrir Image LQ (Avant)", "Open LQ Image (Before)"), command=lambda: self.load_comp_img("before")).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(ctrl, text=_t("Ouvrir Image HQ (Après)", "Open HQ Image (After)"), command=lambda: self.load_comp_img("after")).pack(side="left", padx=10, pady=10)

        # Auto-load checkbox
        self.comp_auto_load = ctk.CTkCheckBox(ctrl, text=_t("Auto-charger dernière image traitée", "Auto-load last processed image"))
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
            messagebox.showerror(_t("Erreur", "Error"), f"{_t('Impossible d\'ouvrir l\'image : ', 'Cannot open image: ')}{e}")

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

    def _ups_on_colorfix_toggle(self):
        """Active/désactive les contrôles Color Fix selon la checkbox."""
        enabled = bool(self.widgets.get("ups_colorfix") and self.widgets["ups_colorfix"].get())
        state = "normal" if enabled else "disabled"
        for key in ("ups_colorfix_method", "ups_colorfix_settings_btn"):
            w = self.widgets.get(key)
            if w:
                try:
                    w.configure(state=state)
                except Exception:
                    pass

    def _ups_on_serialize_toggle(self):
        """Active/désactive la box de numéro de départ selon la checkbox sérialisation."""
        enabled = hasattr(self, "_ups_serialize") and bool(self._ups_serialize.get())
        state = "normal" if enabled else "disabled"
        if hasattr(self, "_ups_serialize_start"):
            try:
                self._ups_serialize_start.configure(state=state)
            except Exception:
                pass

    def _ups_on_dandere_toggle(self):
        """Active/désactive les contrôles Skip frames selon la checkbox."""
        enabled = "ups_dandere" in self.widgets and bool(self.widgets["ups_dandere"].get())
        state = "normal" if enabled else "disabled"
        for key in ("ups_dandere_threshold", "ups_dandere_block_size"):
            if key in self.widgets:
                try:
                    self.widgets[key].configure(state=state)
                except Exception:
                    pass

    def _ups_on_tempfix_toggle(self):
        """Met à jour l'info latence TF. Bouton ⚙ toujours accessible (contient aussi Undistort)."""
        enabled = "ups_tempfix" in self.widgets and bool(self.widgets["ups_tempfix"].get())
        if "ups_tempfix_info" in self.widgets:
            try:
                if enabled:
                    s = getattr(self, "_tf_settings", {})
                    w = int(s.get("window", 7))
                    lat = w // 2
                    prec = s.get("precision", "float32")
                    strength = float(s.get("strength", 0.5))
                    txt = _t(
                        f"win={w}  lat={lat}f  str={strength:.2f}  {prec}",
                        f"win={w}  lat={lat}f  str={strength:.2f}  {prec}")
                    self.widgets["ups_tempfix_info"].configure(text=txt)
                else:
                    self.widgets["ups_tempfix_info"].configure(text="")
            except Exception:
                pass

    def _ups_on_undistort_toggle(self):
        """Syncs UD checkbox → _tf_settings."""
        enabled = "ups_undistort" in self.widgets and bool(self.widgets["ups_undistort"].get())
        if not hasattr(self, "_tf_settings"):
            self._tf_settings = {}
        self._tf_settings["undistort_enabled"] = enabled
        self.settings.set("ups_undistort", enabled)

    def _ups_open_tf_settings(self):
        """Popup TF uniquement (séparé de UD — v2.5.9)."""
        popup = getattr(self, "_tf_popup", None)
        if popup and popup.winfo_exists():
            popup.lift()
            return

        s = getattr(self, "_tf_settings", {
            "strength": 0.5, "window": 7, "precision": "float32",
            "tempfix_mode": "classic"})

        popup = ctk.CTkToplevel(self)
        popup.title(_t("Réglages Temporal Fix", "Temporal Fix Settings"))
        popup.resizable(False, False)
        popup.grab_set()
        self._tf_popup = popup

        ctk.CTkLabel(popup, text="Temporal Fix — Post-processing",
                     font=("Roboto", 14, "bold"), text_color="#9B59B6").pack(
                     padx=20, pady=(15, 5))
        ctk.CTkLabel(popup,
                     text=_t(
                         "Réduction du scintillement SR sur séquences vidéo.\n"
                         "Blend adaptatif : zones statiques lissées, zones mobiles conservées.",
                         "SR flickering reduction on video sequences.\n"
                         "Adaptive blend: static regions smoothed, moving regions preserved."),
                     font=("Arial", 10), text_color="gray").pack(padx=20, pady=(0, 8))

        body = ctk.CTkFrame(popup, fg_color="transparent")
        body.pack(fill="x", padx=20, pady=4)

        ctk.CTkLabel(body, text=_t("Intensité (0 = désactivé, 1 = maximum) :",
                                   "Strength (0 = off, 1 = maximum):"),
                     anchor="w").pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "0.4-0.6 = recommandé pour anime SR. 0.8+ = effet fort (risque de flou).",
                         "0.4-0.6 = recommended for anime SR. 0.8+ = strong effect (blur risk)."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 6))
        str_row = ctk.CTkFrame(body, fg_color="transparent")
        str_row.pack(fill="x", pady=(0, 10))
        str_lbl = ctk.CTkLabel(str_row, text=f"{s.get('strength', 0.5):.2f}", width=38, anchor="w")
        str_sld = ctk.CTkSlider(str_row, from_=0.0, to=1.0, number_of_steps=20, width=220)
        str_sld.set(s.get("strength", 0.5))
        str_sld.pack(side="left")
        str_lbl.pack(side="left", padx=6)
        str_sld.configure(command=lambda v: str_lbl.configure(text=f"{float(v):.2f}"))

        ctk.CTkLabel(body, text=_t("Taille fenêtre (frames) :", "Window size (frames):"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        win_row = ctk.CTkFrame(body, fg_color="transparent")
        win_row.pack(fill="x", pady=(0, 6))
        win_var = ctk.StringVar(value=str(s.get("window", 7)))
        win_menu = ctk.CTkOptionMenu(win_row, values=["5", "7", "9"], variable=win_var, width=90)
        win_menu.pack(side="left")
        lat_lbl = ctk.CTkLabel(win_row,
                               text=_t(f"→ latence {int(s.get('window', 7)) // 2} frames",
                                       f"→ latency {int(s.get('window', 7)) // 2} frames"),
                               font=("Consolas", 10), text_color="gray60")
        lat_lbl.pack(side="left", padx=(10, 0))
        win_menu.configure(command=lambda v: lat_lbl.configure(
            text=_t(f"→ latence {int(v) // 2} frames", f"→ latency {int(v) // 2} frames")))

        ctk.CTkLabel(body, text=_t("Précision :", "Precision:"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "float16 = 2× moins de VRAM, qualité identique.\n"
                         "float32 = sûr sur toutes les cartes (défaut).",
                         "float16 = 2× less VRAM, same quality.\n"
                         "float32 = safe on all cards (default)."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 6))
        prec_var = ctk.StringVar(value=s.get("precision", "float32"))
        ctk.CTkOptionMenu(body, values=["float32", "float16"],
                          variable=prec_var, width=120).pack(anchor="w", pady=(0, 10))

        ctk.CTkFrame(body, height=1, fg_color="gray35").pack(fill="x", pady=(2, 8))
        ctk.CTkLabel(body, text=_t("Algorithme Temporal Fix :", "TemporalFix Algorithm:"),
                     font=("Roboto", 12, "bold"), anchor="w").pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "Variante : s1 léger / s2★ recommandé / s3 fort.\n"
                         "Backend : PyTorch CUDA | OnnxRuntime ⚡ (recommandé GPU) | TRT ⚡⚡ | CPU.",
                         "Variant: s1 light / s2★ recommended / s3 strong.\n"
                         "Backend: PyTorch CUDA | OnnxRuntime ⚡ (recommended GPU) | TRT ⚡⚡ | CPU."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 4))

        def _tf_parse_mode(m: str):
            if m == "classic":              return "classic", "pytorch"
            if m.startswith("model_"):      return m[6:], "pytorch"
            if m.startswith("ort_"):        return m[4:],  "ort"
            if m.startswith("trt_"):        return m[4:],  "trt"
            if m.startswith("cpu_"):        return m[4:],  "cpu"
            return "classic", "pytorch"

        def _tf_build_mode(variant: str, backend: str) -> str:
            if variant == "classic":
                return "classic"
            pfx = {"pytorch": "model_", "ort": "ort_", "trt": "trt_", "cpu": "cpu_"}.get(backend, "model_")
            return pfx + variant

        _tf_v_init, _tf_b_init = _tf_parse_mode(s.get("tempfix_mode", "classic"))
        _tf_variants = {"classic": _t("Classic (rapide / fallback)", "Classic (fast / fallback)"),
                        "s1": "s1  (~2 MB, léger)", "s2": "s2 ★ (~2 MB, recommandé)", "s3": "s3  (~2 MB, fort)"}
        _tf_variants_rev = {v: k for k, v in _tf_variants.items()}
        _tf_backends = {"pytorch": "PyTorch CUDA", "ort": "OnnxRuntime ⚡ (CUDA)",
                        "trt": "TRT + OnnxRuntime ⚡⚡", "cpu": "CPU (PyTorch)"}
        _tf_backends_rev = {v: k for k, v in _tf_backends.items()}

        tf_sel_row = ctk.CTkFrame(body, fg_color="transparent")
        tf_sel_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(tf_sel_row, text=_t("Variante:", "Variant:"), width=65, anchor="w").pack(side="left")
        tf_var_var = ctk.StringVar(value=_tf_variants.get(_tf_v_init, _tf_variants["classic"]))
        tf_var_menu = ctk.CTkOptionMenu(tf_sel_row, values=list(_tf_variants.values()),
                                        variable=tf_var_var, width=200)
        tf_var_menu.pack(side="left")

        tf_bck_row = ctk.CTkFrame(body, fg_color="transparent")
        tf_bck_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(tf_bck_row, text="Backend:", width=65, anchor="w").pack(side="left")
        tf_bck_var = ctk.StringVar(value=_tf_backends.get(_tf_b_init, _tf_backends["pytorch"]))
        tf_bck_menu = ctk.CTkOptionMenu(tf_bck_row, values=list(_tf_backends.values()),
                                        variable=tf_bck_var, width=200)
        tf_bck_menu.pack(side="left")

        tf_dl_row = ctk.CTkFrame(body, fg_color="transparent")
        tf_dl_row.pack(fill="x", pady=(0, 10))
        tf_dl_status = ctk.CTkLabel(tf_dl_row, text="", font=("Consolas", 10),
                                    text_color="gray60", anchor="w", width=200)
        tf_dl_status.pack(side="left")

        def _tf_current_model_key() -> str:
            variant = _tf_variants_rev.get(tf_var_var.get(), "classic")
            backend = _tf_backends_rev.get(tf_bck_var.get(), "pytorch")
            if variant == "classic":
                return ""
            if backend in ("ort", "trt"):
                return f"temporalfix_{variant}_onnx"
            return f"temporalfix_{variant}"

        def _tf_update_dl_status(*_):
            from src.core.model_manager import get_manager, MODELS
            variant = _tf_variants_rev.get(tf_var_var.get(), "classic")
            tf_bck_menu.configure(state="normal" if variant != "classic" else "disabled")
            model_key = _tf_current_model_key()
            if not model_key:
                tf_dl_status.configure(text=_t("Classic — aucun poids nécessaire.",
                                                "Classic — no weights needed."), text_color="gray60")
                tf_dl_btn.configure(state="disabled")
            else:
                mgr = get_manager()
                if mgr.is_ready(model_key):
                    info = MODELS[model_key]
                    tf_dl_status.configure(text=f"✓ {info['filename']} ({info['size_mb']:.1f} MB)",
                                           text_color="#2ecc71")
                    tf_dl_btn.configure(state="disabled", text=_t("Téléchargé ✓", "Downloaded ✓"))
                else:
                    info = MODELS[model_key]
                    tf_dl_status.configure(text=f"⬇ {info['filename']} ({info['size_mb']:.1f} MB)",
                                           text_color="#e67e22")
                    tf_dl_btn.configure(state="normal", text=_t("Télécharger", "Download"))

        def _tf_download():
            import threading as _thr
            from src.core.model_manager import get_manager
            model_key = _tf_current_model_key()
            if not model_key:
                return
            tf_dl_btn.configure(state="disabled", text=_t("Téléchargement…", "Downloading…"))
            tf_dl_status.configure(text="0 %", text_color="#3498db")
            def _run():
                try:
                    mgr = get_manager()
                    def _progress(dl, total):
                        pct = int(dl * 100 / total) if total else 0
                        popup.after(0, lambda: tf_dl_status.configure(
                            text=f"{pct}%  ({dl//1024}KB / {total//1024}KB)"))
                    mgr.download(model_key, progress_cb=_progress)
                    popup.after(0, _tf_update_dl_status)
                except Exception as e:
                    popup.after(0, lambda: (
                        tf_dl_status.configure(text=f"Erreur: {e}", text_color="#e74c3c"),
                        tf_dl_btn.configure(state="normal", text=_t("Réessayer", "Retry"))))
            _thr.Thread(target=_run, daemon=True).start()

        tf_dl_btn = ctk.CTkButton(tf_dl_row, text=_t("Télécharger", "Download"),
                                  width=110, height=26, command=_tf_download)
        tf_dl_btn.pack(side="right")
        tf_var_menu.configure(command=lambda _v: _tf_update_dl_status())
        tf_bck_menu.configure(command=lambda _v: _tf_update_dl_status())
        _tf_update_dl_status()

        btn_row = ctk.CTkFrame(popup, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 15))

        def _apply():
            self._tf_settings["strength"]     = round(float(str_sld.get()), 2)
            self._tf_settings["window"]       = int(win_var.get())
            self._tf_settings["precision"]    = prec_var.get()
            _tf_v = _tf_variants_rev.get(tf_var_var.get(), "classic")
            _tf_b = _tf_backends_rev.get(tf_bck_var.get(), "pytorch")
            self._tf_settings["tempfix_mode"] = _tf_build_mode(_tf_v, _tf_b)
            self.settings.set("ups_tempfix_strength",  str(self._tf_settings["strength"]))
            self.settings.set("ups_tempfix_window",    str(self._tf_settings["window"]))
            self.settings.set("ups_tempfix_precision", self._tf_settings["precision"])
            self.settings.set("ups_tempfix_mode",      self._tf_settings["tempfix_mode"])
            self._ups_on_tempfix_toggle()
            popup.destroy()

        ctk.CTkButton(btn_row, text=_t("Appliquer", "Apply"), fg_color="#2ecc71",
                      command=_apply).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(btn_row, text=_t("Annuler", "Cancel"), fg_color="#e74c3c",
                      command=popup.destroy).pack(side="left", fill="x", expand=True)

    def _ups_open_ud_settings(self):
        """Popup Undistort uniquement (séparé de TF — v2.5.9)."""
        popup = getattr(self, "_ud_popup", None)
        if popup and popup.winfo_exists():
            popup.lift()
            return

        s = getattr(self, "_tf_settings", {
            "undistort_strength": 0.35, "undistort_window": 5,
            "undistort_mode": "classic"})

        popup = ctk.CTkToplevel(self)
        popup.title(_t("Réglages Undistort", "Undistort Settings"))
        popup.resizable(False, False)
        popup.grab_set()
        self._ud_popup = popup

        ctk.CTkLabel(popup, text="Undistort — Correction de jitter HF",
                     font=("Roboto", 14, "bold"), text_color="#27AE60").pack(
                     padx=20, pady=(15, 5))
        ctk.CTkLabel(popup,
                     text=_t(
                         "Réduit le 'shimmer' sur les contours (jitter haute fréquence).\n"
                         "Médiane temporelle du composant HF (frame − flou gaussien).\n"
                         "Complémentaire à Temporal Fix. Fenêtre plus petite = moins de latence.",
                         "Reduces edge 'shimmer' (high-frequency temporal jitter).\n"
                         "Method: temporal median of HF component (frame − Gaussian blur).\n"
                         "Complementary to Temporal Fix. Smaller window = less latency."),
                     font=("Arial", 10), text_color="gray").pack(padx=20, pady=(0, 8))

        body = ctk.CTkFrame(popup, fg_color="transparent")
        body.pack(fill="x", padx=20, pady=4)

        ctk.CTkLabel(body, text=_t("Intensité Undistort :", "Undistort Strength:"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "0.3–0.5 = recommandé. Plus élevé = correction plus forte (risque de flou HF).",
                         "0.3–0.5 = recommended. Higher = stronger correction (HF blur risk)."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 4))
        undist_str_row = ctk.CTkFrame(body, fg_color="transparent")
        undist_str_row.pack(fill="x", pady=(0, 8))
        _ud_str_init = float(s.get("undistort_strength", 0.35))
        undist_str_lbl = ctk.CTkLabel(undist_str_row, text=f"{_ud_str_init:.2f}", width=38, anchor="w")
        undist_str_sld = ctk.CTkSlider(undist_str_row, from_=0.0, to=1.0, number_of_steps=20, width=220)
        undist_str_sld.set(_ud_str_init)
        undist_str_sld.pack(side="left")
        undist_str_lbl.pack(side="left", padx=6)
        undist_str_sld.configure(command=lambda v: undist_str_lbl.configure(text=f"{float(v):.2f}"))

        ctk.CTkLabel(body, text=_t("Fenêtre Undistort (frames) :", "Undistort Window (frames):"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        udist_win_var = ctk.StringVar(value=str(s.get("undistort_window", 5)))
        ctk.CTkOptionMenu(body, values=["3", "5", "7"], variable=udist_win_var, width=90).pack(
            anchor="w", pady=(0, 8))

        ctk.CTkLabel(body, text=_t("Algorithme Undistort :", "Undistort Algorithm:"),
                     anchor="w", font=("Roboto", 11, "bold")).pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "Classic : médiane HF, rapide, sans poids.\n"
                         "TMT : réseau xg416/pifroggi. Backend : ORT ⚡ recommandé GPU.",
                         "Classic: HF median, fast, no weights.\n"
                         "TMT: xg416/pifroggi network. Backend: ORT ⚡ recommended GPU."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 4))

        def _ud_parse_mode(m: str):
            if m == "classic":          return "classic", "pytorch"
            if m.startswith("model_"):  return m[6:], "pytorch"
            if m.startswith("ort_"):    return m[4:],  "ort"
            if m.startswith("trt_"):    return m[4:],  "trt"
            if m.startswith("cpu_"):    return m[4:],  "cpu"
            return "classic", "pytorch"

        def _ud_build_mode(variant: str, backend: str) -> str:
            if variant == "classic":
                return "classic"
            pfx = {"pytorch": "model_", "ort": "ort_", "trt": "trt_", "cpu": "cpu_"}.get(backend, "model_")
            return pfx + variant

        _ud_v_init, _ud_b_init = _ud_parse_mode(s.get("undistort_mode", "classic"))
        _ud_variants = {"classic": _t("Classic (rapide / fallback)", "Classic (fast / fallback)"),
                        "tmt": "TMT (~8 MB PTH / ~5 MB ONNX)"}
        _ud_variants_rev = {v: k for k, v in _ud_variants.items()}
        _ud_backends = {"pytorch": "PyTorch CUDA", "ort": "OnnxRuntime ⚡ (CUDA)",
                        "trt": "TRT + OnnxRuntime ⚡⚡", "cpu": "CPU (PyTorch)"}
        _ud_backends_rev = {v: k for k, v in _ud_backends.items()}

        ud_sel_row = ctk.CTkFrame(body, fg_color="transparent")
        ud_sel_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(ud_sel_row, text=_t("Variante:", "Variant:"), width=65, anchor="w").pack(side="left")
        ud_var_var = ctk.StringVar(value=_ud_variants.get(_ud_v_init, _ud_variants["classic"]))
        ud_var_menu = ctk.CTkOptionMenu(ud_sel_row, values=list(_ud_variants.values()),
                                        variable=ud_var_var, width=200)
        ud_var_menu.pack(side="left")

        ud_bck_row = ctk.CTkFrame(body, fg_color="transparent")
        ud_bck_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(ud_bck_row, text="Backend:", width=65, anchor="w").pack(side="left")
        ud_bck_var = ctk.StringVar(value=_ud_backends.get(_ud_b_init, _ud_backends["pytorch"]))
        ud_bck_menu = ctk.CTkOptionMenu(ud_bck_row, values=list(_ud_backends.values()),
                                        variable=ud_bck_var, width=200)
        ud_bck_menu.pack(side="left")

        ud_dl_row = ctk.CTkFrame(body, fg_color="transparent")
        ud_dl_row.pack(fill="x", pady=(0, 12))
        ud_dl_status = ctk.CTkLabel(ud_dl_row, text="", font=("Consolas", 10),
                                    text_color="gray60", anchor="w", width=200)
        ud_dl_status.pack(side="left")

        def _ud_current_model_key() -> str:
            variant = _ud_variants_rev.get(ud_var_var.get(), "classic")
            backend = _ud_backends_rev.get(ud_bck_var.get(), "pytorch")
            if variant == "classic":
                return ""
            if backend in ("ort", "trt"):
                return "undistort_tmt_onnx"
            return "undistort_tmt"

        def _ud_update_dl_status(*_):
            from src.core.model_manager import get_manager, MODELS
            variant = _ud_variants_rev.get(ud_var_var.get(), "classic")
            ud_bck_menu.configure(state="normal" if variant != "classic" else "disabled")
            model_key = _ud_current_model_key()
            if not model_key:
                ud_dl_status.configure(text=_t("Classic — aucun poids nécessaire.",
                                                "Classic — no weights needed."), text_color="gray60")
                ud_dl_btn.configure(state="disabled")
            else:
                mgr = get_manager()
                if mgr.is_ready(model_key):
                    info = MODELS[model_key]
                    ud_dl_status.configure(text=f"✓ {info['filename']} ({info['size_mb']:.1f} MB)",
                                           text_color="#2ecc71")
                    ud_dl_btn.configure(state="disabled", text=_t("Téléchargé ✓", "Downloaded ✓"))
                else:
                    info = MODELS[model_key]
                    ud_dl_status.configure(text=f"⬇ {info['filename']} ({info['size_mb']:.1f} MB)",
                                           text_color="#e67e22")
                    ud_dl_btn.configure(state="normal", text=_t("Télécharger", "Download"))

        def _ud_download():
            import threading as _thr
            from src.core.model_manager import get_manager
            model_key = _ud_current_model_key()
            if not model_key:
                return
            ud_dl_btn.configure(state="disabled", text=_t("Téléchargement…", "Downloading…"))
            ud_dl_status.configure(text="0 %", text_color="#3498db")
            def _run():
                try:
                    mgr = get_manager()
                    def _progress(dl, total):
                        pct = int(dl * 100 / total) if total else 0
                        popup.after(0, lambda: ud_dl_status.configure(
                            text=f"{pct}%  ({dl//1024}KB / {total//1024}KB)"))
                    mgr.download(model_key, progress_cb=_progress)
                    popup.after(0, _ud_update_dl_status)
                except Exception as e:
                    popup.after(0, lambda: (
                        ud_dl_status.configure(text=f"Erreur: {e}", text_color="#e74c3c"),
                        ud_dl_btn.configure(state="normal", text=_t("Réessayer", "Retry"))))
            _thr.Thread(target=_run, daemon=True).start()

        ud_dl_btn = ctk.CTkButton(ud_dl_row, text=_t("Télécharger", "Download"),
                                   width=110, height=26, command=_ud_download)
        ud_dl_btn.pack(side="right")
        ud_var_menu.configure(command=lambda _v: _ud_update_dl_status())
        ud_bck_menu.configure(command=lambda _v: _ud_update_dl_status())
        _ud_update_dl_status()

        btn_row = ctk.CTkFrame(popup, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 15))

        def _apply():
            self._tf_settings["undistort_strength"] = round(float(undist_str_sld.get()), 2)
            self._tf_settings["undistort_window"]   = int(udist_win_var.get())
            _ud_v = _ud_variants_rev.get(ud_var_var.get(), "classic")
            _ud_b = _ud_backends_rev.get(ud_bck_var.get(), "pytorch")
            self._tf_settings["undistort_mode"]     = _ud_build_mode(_ud_v, _ud_b)
            self.settings.set("ups_undistort_strength", str(self._tf_settings["undistort_strength"]))
            self.settings.set("ups_undistort_window",   str(self._tf_settings["undistort_window"]))
            self.settings.set("ups_undistort_mode",     self._tf_settings["undistort_mode"])
            popup.destroy()

        ctk.CTkButton(btn_row, text=_t("Appliquer", "Apply"), fg_color="#2ecc71",
                      command=_apply).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(btn_row, text=_t("Annuler", "Cancel"), fg_color="#e74c3c",
                      command=popup.destroy).pack(side="left", fill="x", expand=True)

    def _ups_open_tempfix_settings(self):
        """Ouvre la fenêtre de réglages Temporal Fix (popup modale)."""
        popup = getattr(self, "_tf_popup", None)
        if popup and popup.winfo_exists():
            popup.lift()
            return

        s = getattr(self, "_tf_settings", {
            "strength": 0.5, "window": 7, "precision": "float32",
            "tempfix_mode": "classic"})

        popup = ctk.CTkToplevel(self)
        popup.title(_t("Réglages Temporal Fix", "Temporal Fix Settings"))
        popup.resizable(False, False)
        popup.grab_set()
        self._tf_popup = popup

        ctk.CTkLabel(popup, text="Temporal Fix — Post-processing",
                     font=("Roboto", 14, "bold"), text_color="#9B59B6").pack(
                     padx=20, pady=(15, 5))

        ctk.CTkLabel(popup,
                     text=_t(
                         "Réduction du scintillement SR sur séquences vidéo.\n"
                         "Blend adaptatif : zones statiques lissées, zones mobiles conservées.",
                         "SR flickering reduction on video sequences.\n"
                         "Adaptive blend: static regions smoothed, moving regions preserved."),
                     font=("Arial", 10), text_color="gray").pack(padx=20, pady=(0, 8))

        body = ctk.CTkFrame(popup, fg_color="transparent")
        body.pack(fill="x", padx=20, pady=4)

        # ── Intensité ──
        ctk.CTkLabel(body, text=_t("Intensité (0 = désactivé, 1 = maximum) :",
                                   "Strength (0 = off, 1 = maximum):"),
                     anchor="w").pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "0.4-0.6 = recommandé pour anime SR. 0.8+ = effet fort (risque de flou).",
                         "0.4-0.6 = recommended for anime SR. 0.8+ = strong effect (blur risk)."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 6))
        str_row = ctk.CTkFrame(body, fg_color="transparent")
        str_row.pack(fill="x", pady=(0, 10))
        str_lbl = ctk.CTkLabel(str_row, text=f"{s['strength']:.2f}", width=38, anchor="w")
        str_sld = ctk.CTkSlider(str_row, from_=0.0, to=1.0, number_of_steps=20, width=220)
        str_sld.set(s["strength"])
        str_sld.pack(side="left")
        str_lbl.pack(side="left", padx=6)
        str_sld.configure(command=lambda v: str_lbl.configure(text=f"{float(v):.2f}"))

        # ── Fenêtre ──
        ctk.CTkLabel(body, text=_t("Taille fenêtre (frames) :", "Window size (frames):"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        win_row = ctk.CTkFrame(body, fg_color="transparent")
        win_row.pack(fill="x", pady=(0, 6))
        _win_opts = ["5", "7", "9"]
        win_var = ctk.StringVar(value=str(s.get("window", 7)))
        win_menu = ctk.CTkOptionMenu(win_row, values=_win_opts, variable=win_var, width=90)
        win_menu.pack(side="left")
        lat_lbl = ctk.CTkLabel(win_row,
                               text=_t(f"→ latence {int(s.get('window', 7)) // 2} frames",
                                       f"→ latency {int(s.get('window', 7)) // 2} frames"),
                               font=("Consolas", 10), text_color="gray60")
        lat_lbl.pack(side="left", padx=(10, 0))

        def _on_win_change(v):
            lat_lbl.configure(text=_t(f"→ latence {int(v) // 2} frames",
                                      f"→ latency {int(v) // 2} frames"))
        win_menu.configure(command=_on_win_change)

        # ── Précision ──
        ctk.CTkLabel(body, text=_t("Précision :", "Precision:"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "float16 = 2× moins de VRAM, qualité identique pour le blend.\n"
                         "float32 = sûr sur toutes les cartes (défaut).",
                         "float16 = 2× less VRAM, same quality for blending.\n"
                         "float32 = safe on all cards (default)."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 6))
        prec_var = ctk.StringVar(value=s.get("precision", "float32"))
        prec_menu = ctk.CTkOptionMenu(body, values=["float32", "float16"],
                                      variable=prec_var, width=120)
        prec_menu.pack(anchor="w", pady=(0, 10))

        # ── Mode algorithme TemporalFix ────────────────────────────────────────
        ctk.CTkFrame(body, height=1, fg_color="gray35").pack(fill="x", pady=(2, 8))
        ctk.CTkLabel(body, text=_t("Algorithme Temporal Fix :", "TemporalFix Algorithm:"),
                     font=("Roboto", 12, "bold"), anchor="w").pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "Variante : s1 léger / s2★ recommandé / s3 fort.\n"
                         "Backend : PyTorch CUDA | OnnxRuntime ⚡ (recommandé GPU) | TRT ⚡⚡ | CPU.",
                         "Variant: s1 light / s2★ recommended / s3 strong.\n"
                         "Backend: PyTorch CUDA | OnnxRuntime ⚡ (recommended GPU) | TRT ⚡⚡ | CPU."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 4))

        # ── Parse stored mode → (variant, backend) ────────────────────────────
        def _tf_parse_mode(m: str):
            if m == "classic":              return "classic", "pytorch"
            if m.startswith("model_"):      return m[6:], "pytorch"
            if m.startswith("ort_"):        return m[4:],  "ort"
            if m.startswith("trt_"):        return m[4:],  "trt"
            if m.startswith("cpu_"):        return m[4:],  "cpu"
            return "classic", "pytorch"

        def _tf_build_mode(variant: str, backend: str) -> str:
            if variant == "classic":        return "classic"
            pfx = {"pytorch": "model_", "ort": "ort_", "trt": "trt_", "cpu": "cpu_"}.get(backend, "model_")
            return pfx + variant

        _tf_v_init, _tf_b_init = _tf_parse_mode(s.get("tempfix_mode", "classic"))

        _tf_variants = {"classic": _t("Classic (rapide / fallback)", "Classic (fast / fallback)"),
                        "s1": "s1  (~2 MB, léger)", "s2": "s2 ★ (~2 MB, recommandé)", "s3": "s3  (~2 MB, fort)"}
        _tf_variants_rev = {v: k for k, v in _tf_variants.items()}
        _tf_backends = {"pytorch": "PyTorch CUDA", "ort": "OnnxRuntime ⚡ (CUDA)",
                        "trt": "TRT + OnnxRuntime ⚡⚡", "cpu": "CPU (PyTorch)"}
        _tf_backends_rev = {v: k for k, v in _tf_backends.items()}

        tf_sel_row = ctk.CTkFrame(body, fg_color="transparent")
        tf_sel_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(tf_sel_row, text=_t("Variante:", "Variant:"), width=65, anchor="w").pack(side="left")
        tf_var_var = ctk.StringVar(value=_tf_variants.get(_tf_v_init, _tf_variants["classic"]))
        tf_var_menu = ctk.CTkOptionMenu(tf_sel_row, values=list(_tf_variants.values()),
                                        variable=tf_var_var, width=200)
        tf_var_menu.pack(side="left")

        tf_bck_row = ctk.CTkFrame(body, fg_color="transparent")
        tf_bck_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(tf_bck_row, text="Backend:", width=65, anchor="w").pack(side="left")
        tf_bck_var = ctk.StringVar(value=_tf_backends.get(_tf_b_init, _tf_backends["pytorch"]))
        tf_bck_menu = ctk.CTkOptionMenu(tf_bck_row, values=list(_tf_backends.values()),
                                        variable=tf_bck_var, width=200)
        tf_bck_menu.pack(side="left")

        # Download row for TF models
        tf_dl_row = ctk.CTkFrame(body, fg_color="transparent")
        tf_dl_row.pack(fill="x", pady=(0, 10))
        tf_dl_status = ctk.CTkLabel(tf_dl_row, text="", font=("Consolas", 10),
                                    text_color="gray60", anchor="w", width=200)
        tf_dl_status.pack(side="left")

        def _tf_current_model_key() -> str:
            """Return model_manager key for current variant+backend selection, or ''."""
            variant = _tf_variants_rev.get(tf_var_var.get(), "classic")
            backend = _tf_backends_rev.get(tf_bck_var.get(), "pytorch")
            if variant == "classic":
                return ""
            if backend in ("ort", "trt"):
                return f"temporalfix_{variant}_onnx"
            return f"temporalfix_{variant}"

        def _tf_update_dl_status(*_):
            from src.core.model_manager import get_manager, MODELS
            variant = _tf_variants_rev.get(tf_var_var.get(), "classic")
            backend = _tf_backends_rev.get(tf_bck_var.get(), "pytorch")
            # Enable/disable backend menu
            tf_bck_menu.configure(state="normal" if variant != "classic" else "disabled")
            model_key = _tf_current_model_key()
            if not model_key:
                tf_dl_status.configure(text=_t("Classic — aucun poids nécessaire.",
                                                "Classic — no weights needed."), text_color="gray60")
                tf_dl_btn.configure(state="disabled")
            else:
                mgr = get_manager()
                if mgr.is_ready(model_key):
                    info = MODELS[model_key]
                    tf_dl_status.configure(
                        text=f"✓ {info['filename']} ({info['size_mb']:.1f} MB)",
                        text_color="#2ecc71")
                    tf_dl_btn.configure(state="disabled", text=_t("Téléchargé ✓", "Downloaded ✓"))
                else:
                    info = MODELS[model_key]
                    tf_dl_status.configure(
                        text=f"⬇ {info['filename']} ({info['size_mb']:.1f} MB)",
                        text_color="#e67e22")
                    tf_dl_btn.configure(state="normal", text=_t("Télécharger", "Download"))

        def _tf_download():
            import threading as _thr
            from src.core.model_manager import get_manager
            model_key = _tf_current_model_key()
            if not model_key:
                return
            tf_dl_btn.configure(state="disabled", text=_t("Téléchargement…", "Downloading…"))
            tf_dl_status.configure(text="0 %", text_color="#3498db")

            def _run():
                try:
                    mgr = get_manager()
                    def _progress(dl, total):
                        pct = int(dl * 100 / total) if total else 0
                        popup.after(0, lambda: tf_dl_status.configure(
                            text=f"{pct}%  ({dl//1024}KB / {total//1024}KB)"))
                    mgr.download(model_key, progress_cb=_progress)
                    popup.after(0, _tf_update_dl_status)
                except Exception as e:
                    popup.after(0, lambda: (
                        tf_dl_status.configure(text=f"Erreur: {e}", text_color="#e74c3c"),
                        tf_dl_btn.configure(state="normal", text=_t("Réessayer", "Retry"))))
            _thr.Thread(target=_run, daemon=True).start()

        tf_dl_btn = ctk.CTkButton(tf_dl_row, text=_t("Télécharger", "Download"),
                                  width=110, height=26, command=_tf_download)
        tf_dl_btn.pack(side="right")
        tf_var_menu.configure(command=lambda _v: _tf_update_dl_status())
        tf_bck_menu.configure(command=lambda _v: _tf_update_dl_status())
        _tf_update_dl_status()

        # ── Undistort ─────────────────────────────────────────────────────────
        ctk.CTkFrame(body, height=1, fg_color="gray35").pack(fill="x", pady=(6, 10))
        ctk.CTkLabel(body, text="Undistort — Correction de jitter HF",
                     font=("Roboto", 12, "bold"), text_color="#27AE60").pack(
                     anchor="w", pady=(0, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "Réduit le 'shimmer' sur les contours (jitter haute fréquence).\n"
                         "Calcul : médiane temporelle du composant HF (frame - flou gaussien).\n"
                         "Complémentaire à Temporal Fix. Fenêtre plus petite = moins de latence.",
                         "Reduces edge 'shimmer' (high-frequency temporal jitter).\n"
                         "Method: temporal median of HF component (frame - Gaussian blur).\n"
                         "Complementary to Temporal Fix. Smaller window = less latency."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 6))

        undist_enabled_var = ctk.BooleanVar(value=bool(s.get("undistort_enabled", False)))
        undist_chk = ctk.CTkCheckBox(
            body,
            text=_t("Activer Undistort", "Enable Undistort"),
            variable=undist_enabled_var)
        undist_chk.pack(anchor="w", pady=(0, 6))
        if s.get("undistort_enabled", False):
            undist_chk.select()
        ToolTip(undist_chk, _t(
            "Active la correction de jitter HF post-upscale.\n"
            "Fonctionne en parallèle du Temporal Fix.",
            "Enable HF jitter correction post-upscale.\n"
            "Works in parallel with Temporal Fix."))

        ctk.CTkLabel(body, text=_t("Intensité Undistort :", "Undistort Strength:"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "0.3–0.5 = recommandé. Plus élevé = correction plus forte (risque de flou HF).",
                         "0.3–0.5 = recommended. Higher = stronger correction (HF blur risk)."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 4))
        undist_str_row = ctk.CTkFrame(body, fg_color="transparent")
        undist_str_row.pack(fill="x", pady=(0, 8))
        _ud_str_init = float(s.get("undistort_strength", 0.35))
        undist_str_lbl = ctk.CTkLabel(undist_str_row, text=f"{_ud_str_init:.2f}", width=38, anchor="w")
        undist_str_sld = ctk.CTkSlider(undist_str_row, from_=0.0, to=1.0, number_of_steps=20, width=220)
        undist_str_sld.set(_ud_str_init)
        undist_str_sld.pack(side="left")
        undist_str_lbl.pack(side="left", padx=6)
        undist_str_sld.configure(command=lambda v: undist_str_lbl.configure(text=f"{float(v):.2f}"))

        ctk.CTkLabel(body, text=_t("Fenêtre Undistort (frames) :", "Undistort Window (frames):"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        _ud_win_opts = ["3", "5", "7"]
        udist_win_var = ctk.StringVar(value=str(s.get("undistort_window", 5)))
        ctk.CTkOptionMenu(body, values=_ud_win_opts, variable=udist_win_var, width=90).pack(
            anchor="w", pady=(0, 8))

        # ── Mode algorithme Undistort ──────────────────────────────────────────
        ctk.CTkLabel(body, text=_t("Algorithme Undistort :", "Undistort Algorithm:"),
                     anchor="w", font=("Roboto", 11, "bold")).pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "Classic : médiane HF, rapide, sans poids.\n"
                         "TMT : réseau xg416/pifroggi. Backend : ORT ⚡ recommandé GPU.",
                         "Classic: HF median, fast, no weights.\n"
                         "TMT: xg416/pifroggi network. Backend: ORT ⚡ recommended GPU."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 4))

        # ── Parse stored undistort mode → (variant, backend) ──────────────────
        def _ud_parse_mode(m: str):
            if m == "classic":          return "classic", "pytorch"
            if m.startswith("model_"):  return m[6:], "pytorch"
            if m.startswith("ort_"):    return m[4:],  "ort"
            if m.startswith("trt_"):    return m[4:],  "trt"
            if m.startswith("cpu_"):    return m[4:],  "cpu"
            return "classic", "pytorch"

        def _ud_build_mode(variant: str, backend: str) -> str:
            if variant == "classic":    return "classic"
            pfx = {"pytorch": "model_", "ort": "ort_", "trt": "trt_", "cpu": "cpu_"}.get(backend, "model_")
            return pfx + variant

        _ud_v_init, _ud_b_init = _ud_parse_mode(s.get("undistort_mode", "classic"))

        _ud_variants = {"classic": _t("Classic (rapide / fallback)", "Classic (fast / fallback)"),
                        "tmt": "TMT (~8 MB PTH / ~5 MB ONNX)"}
        _ud_variants_rev = {v: k for k, v in _ud_variants.items()}
        _ud_backends = {"pytorch": "PyTorch CUDA", "ort": "OnnxRuntime ⚡ (CUDA)",
                        "trt": "TRT + OnnxRuntime ⚡⚡", "cpu": "CPU (PyTorch)"}
        _ud_backends_rev = {v: k for k, v in _ud_backends.items()}

        ud_sel_row = ctk.CTkFrame(body, fg_color="transparent")
        ud_sel_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(ud_sel_row, text=_t("Variante:", "Variant:"), width=65, anchor="w").pack(side="left")
        ud_var_var = ctk.StringVar(value=_ud_variants.get(_ud_v_init, _ud_variants["classic"]))
        ud_var_menu = ctk.CTkOptionMenu(ud_sel_row, values=list(_ud_variants.values()),
                                        variable=ud_var_var, width=200)
        ud_var_menu.pack(side="left")

        ud_bck_row = ctk.CTkFrame(body, fg_color="transparent")
        ud_bck_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(ud_bck_row, text="Backend:", width=65, anchor="w").pack(side="left")
        ud_bck_var = ctk.StringVar(value=_ud_backends.get(_ud_b_init, _ud_backends["pytorch"]))
        ud_bck_menu = ctk.CTkOptionMenu(ud_bck_row, values=list(_ud_backends.values()),
                                        variable=ud_bck_var, width=200)
        ud_bck_menu.pack(side="left")

        ud_dl_row = ctk.CTkFrame(body, fg_color="transparent")
        ud_dl_row.pack(fill="x", pady=(0, 12))
        ud_dl_status = ctk.CTkLabel(ud_dl_row, text="", font=("Consolas", 10),
                                    text_color="gray60", anchor="w", width=200)
        ud_dl_status.pack(side="left")

        def _ud_current_model_key() -> str:
            variant = _ud_variants_rev.get(ud_var_var.get(), "classic")
            backend = _ud_backends_rev.get(ud_bck_var.get(), "pytorch")
            if variant == "classic":
                return ""
            if backend in ("ort", "trt"):
                return "undistort_tmt_onnx"
            return "undistort_tmt"

        def _ud_update_dl_status(*_):
            from src.core.model_manager import get_manager, MODELS
            variant = _ud_variants_rev.get(ud_var_var.get(), "classic")
            ud_bck_menu.configure(state="normal" if variant != "classic" else "disabled")
            model_key = _ud_current_model_key()
            if not model_key:
                ud_dl_status.configure(text=_t("Classic — aucun poids nécessaire.",
                                                "Classic — no weights needed."), text_color="gray60")
                ud_dl_btn.configure(state="disabled")
            else:
                mgr = get_manager()
                if mgr.is_ready(model_key):
                    info = MODELS[model_key]
                    ud_dl_status.configure(
                        text=f"✓ {info['filename']} ({info['size_mb']:.1f} MB)",
                        text_color="#2ecc71")
                    ud_dl_btn.configure(state="disabled", text=_t("Téléchargé ✓", "Downloaded ✓"))
                else:
                    info = MODELS[model_key]
                    ud_dl_status.configure(
                        text=f"⬇ {info['filename']} ({info['size_mb']:.1f} MB)",
                        text_color="#e67e22")
                    ud_dl_btn.configure(state="normal", text=_t("Télécharger", "Download"))

        def _ud_download():
            import threading as _thr
            from src.core.model_manager import get_manager
            model_key = _ud_current_model_key()
            if not model_key:
                return
            ud_dl_btn.configure(state="disabled", text=_t("Téléchargement…", "Downloading…"))
            ud_dl_status.configure(text="0 %", text_color="#3498db")

            def _run():
                try:
                    mgr = get_manager()
                    def _progress(dl, total):
                        pct = int(dl * 100 / total) if total else 0
                        popup.after(0, lambda: ud_dl_status.configure(
                            text=f"{pct}%  ({dl//1024}KB / {total//1024}KB)"))
                    mgr.download(model_key, progress_cb=_progress)
                    popup.after(0, _ud_update_dl_status)
                except Exception as e:
                    popup.after(0, lambda: (
                        ud_dl_status.configure(text=f"Erreur: {e}", text_color="#e74c3c"),
                        ud_dl_btn.configure(state="normal", text=_t("Réessayer", "Retry"))))
            _thr.Thread(target=_run, daemon=True).start()

        ud_dl_btn = ctk.CTkButton(ud_dl_row, text=_t("Télécharger", "Download"),
                                   width=110, height=26, command=_ud_download)
        ud_dl_btn.pack(side="right")
        ud_var_menu.configure(command=lambda _v: _ud_update_dl_status())
        ud_bck_menu.configure(command=lambda _v: _ud_update_dl_status())
        _ud_update_dl_status()

        # ── Boutons ──
        btn_row = ctk.CTkFrame(popup, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 15))

        def _apply():
            self._tf_settings["strength"]           = round(float(str_sld.get()), 2)
            self._tf_settings["window"]             = int(win_var.get())
            self._tf_settings["precision"]          = prec_var.get()
            _tf_v = _tf_variants_rev.get(tf_var_var.get(), "classic")
            _tf_b = _tf_backends_rev.get(tf_bck_var.get(), "pytorch")
            self._tf_settings["tempfix_mode"]       = _tf_build_mode(_tf_v, _tf_b)
            self._tf_settings["undistort_enabled"]  = bool(undist_enabled_var.get())
            self._tf_settings["undistort_strength"] = round(float(undist_str_sld.get()), 2)
            self._tf_settings["undistort_window"]   = int(udist_win_var.get())
            _ud_v = _ud_variants_rev.get(ud_var_var.get(), "classic")
            _ud_b = _ud_backends_rev.get(ud_bck_var.get(), "pytorch")
            self._tf_settings["undistort_mode"]     = _ud_build_mode(_ud_v, _ud_b)
            self.settings.set("ups_tempfix_strength",       str(self._tf_settings["strength"]))
            self.settings.set("ups_tempfix_window",         str(self._tf_settings["window"]))
            self.settings.set("ups_tempfix_precision",      self._tf_settings["precision"])
            self.settings.set("ups_tempfix_mode",           self._tf_settings["tempfix_mode"])
            self.settings.set("ups_undistort",              self._tf_settings["undistort_enabled"])
            self.settings.set("ups_undistort_strength",     str(self._tf_settings["undistort_strength"]))
            self.settings.set("ups_undistort_window",       str(self._tf_settings["undistort_window"]))
            self.settings.set("ups_undistort_mode",         self._tf_settings["undistort_mode"])
            self._ups_on_tempfix_toggle()  # rafraîchit info label
            popup.destroy()

        ctk.CTkButton(btn_row, text=_t("Appliquer", "Apply"), fg_color="#2ecc71",
                      command=_apply).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(btn_row, text=_t("Annuler", "Cancel"), fg_color="#e74c3c",
                      command=popup.destroy).pack(side="left", fill="x", expand=True)

    def _ups_on_colorfix_method_change(self, method: str):
        """Sauvegarde méthode et rafraîchit le popup si ouvert."""
        self.settings.set("ups_colorfix_method", method)
        popup = getattr(self, "_cf_popup", None)
        if popup and popup.winfo_exists():
            popup.destroy()
            self._ups_open_colorfix_settings()

    def _ups_open_colorfix_settings(self):
        """Ouvre la fenêtre de réglages Color Fix (mode wavelet ou average)."""
        popup = getattr(self, "_cf_popup", None)
        if popup and popup.winfo_exists():
            popup.lift()
            return

        method = self.widgets["ups_colorfix_method"].get()
        s = self._cf_settings

        popup = ctk.CTkToplevel(self)
        popup.title(_t("Réglages Color Fix", "Color Fix Settings"))
        popup.resizable(False, False)
        popup.grab_set()
        self._cf_popup = popup

        ctk.CTkLabel(popup, text=f"Color Fix — {method}",
                     font=("Roboto", 14, "bold"), text_color="#3B8ED0").pack(
                     padx=20, pady=(15, 5))

        body = ctk.CTkFrame(popup, fg_color="transparent")
        body.pack(fill="x", padx=20, pady=8)

        # ── Intensité (déplacé depuis la ligne principale) ─────────────────────
        ctk.CTkLabel(body, text=_t("Intensité (0 = désactivé, 1 = correction complète) :",
                                   "Intensity (0 = off, 1 = full correction):"),
                     anchor="w").pack(fill="x", pady=(0, 2))
        _cf_str_init = float(s.get("strength", 1.0))
        _cf_str_row = ctk.CTkFrame(body, fg_color="transparent")
        _cf_str_row.pack(fill="x", pady=(0, 8))
        cf_str_lbl = ctk.CTkLabel(_cf_str_row, text=f"{_cf_str_init:.1f}", width=28, anchor="w")
        cf_str_sld = ctk.CTkSlider(_cf_str_row, from_=0.0, to=1.0, number_of_steps=20, width=220)
        cf_str_sld.set(_cf_str_init)
        cf_str_sld.pack(side="left")
        cf_str_lbl.pack(side="left", padx=6)
        cf_str_sld.configure(command=lambda v: cf_str_lbl.configure(text=f"{float(v):.1f}"))
        ToolTip(cf_str_sld, _t(
            "Intensité de la correction couleur (0 = aucune, 1 = complète).",
            "Color correction intensity (0 = none, 1 = full)."))

        ctk.CTkFrame(body, height=1, fg_color="gray35").pack(fill="x", pady=(0, 8))

        if method == "wavelet":
            # ── WAVELET ──
            ctk.CTkLabel(body, text=_t(
                "Niveaux wavelet (1-10) :", "Wavelet levels (1-10):"),
                anchor="w").pack(fill="x", pady=(0, 2))
            ctk.CTkLabel(body, text=_t(
                "Rayon effectif = 2^N px. N=4 → 16 px (recommandé : 3-5).\n"
                "Plus élevé = correction globale. Plus bas = locale.",
                "Effective radius = 2^N px. N=4 → 16 px (recommended: 3-5).\n"
                "Higher = global correction. Lower = local."),
                font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 6))
            wrow = ctk.CTkFrame(body, fg_color="transparent")
            wrow.pack(fill="x", pady=(0, 10))
            wav_lbl = ctk.CTkLabel(wrow, text=str(s["wavelets"]), width=24, anchor="w")
            wav_sld = ctk.CTkSlider(wrow, from_=1, to=10, number_of_steps=9, width=220)
            wav_sld.set(s["wavelets"])
            wav_sld.pack(side="left")
            wav_lbl.pack(side="left", padx=6)
            wav_sld.configure(command=lambda v: wav_lbl.configure(text=str(int(v))))
            rad_sld = None
            fast_var = None
        else:
            # ── AVERAGE ──
            ctk.CTkLabel(body, text=_t(
                "Radius (5-100) :", "Radius (5-100):"),
                anchor="w").pack(fill="x", pady=(0, 2))
            ctk.CTkLabel(body, text=_t(
                "Rayon BoxBlur (fast=off) ou facteur de réduction (fast=on).\n"
                "32 = local, 96 = global. Recommandé : 20-50.",
                "BoxBlur radius (fast=off) or downscale factor (fast=on).\n"
                "32 = local, 96 = global. Recommended: 20-50."),
                font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 6))
            rrow = ctk.CTkFrame(body, fg_color="transparent")
            rrow.pack(fill="x", pady=(0, 6))
            rad_lbl = ctk.CTkLabel(rrow, text=str(s["radius"]), width=28, anchor="w")
            rad_sld = ctk.CTkSlider(rrow, from_=5, to=100, number_of_steps=19, width=220)
            rad_sld.set(s["radius"])
            rad_sld.pack(side="left")
            rad_lbl.pack(side="left", padx=6)
            rad_sld.configure(command=lambda v: rad_lbl.configure(text=str(int(v))))
            fast_var = ctk.BooleanVar(value=s["fast"])
            fast_chk = ctk.CTkCheckBox(body, text=_t(
                "Mode rapide (fast) — downscale+upscale, ~10× plus vite",
                "Fast mode — downscale+upscale, ~10× faster"),
                variable=fast_var)
            fast_chk.pack(anchor="w", pady=(0, 10))
            if s["fast"]:
                fast_chk.select()
            ToolTip(fast_chk, _t(
                "fast=True : divise/réupscale (chaiNNer-style). Très rapide, légèrement pixelisé pour radius>60.\n"
                "fast=False : BoxBlur PIL précis (~20 ms/1080p).",
                "fast=True: downscale+upscale (chaiNNer-style). Very fast, slightly blocky for radius>60.\n"
                "fast=False: precise PIL BoxBlur (~20 ms/1080p)."))
            wav_sld = None

        # ── Planes ──
        ctk.CTkLabel(body, text=_t("Canaux à corriger :", "Channels to correct:"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        plane_row = ctk.CTkFrame(body, fg_color="transparent")
        plane_row.pack(fill="x", pady=(0, 10))
        r_var = ctk.BooleanVar(value=0 in s["planes"])
        g_var = ctk.BooleanVar(value=1 in s["planes"])
        b_var = ctk.BooleanVar(value=2 in s["planes"])
        r_chk = ctk.CTkCheckBox(plane_row, text="R", variable=r_var, width=50)
        g_chk = ctk.CTkCheckBox(plane_row, text="G", variable=g_var, width=50)
        b_chk = ctk.CTkCheckBox(plane_row, text="B", variable=b_var, width=50)
        r_chk.pack(side="left", padx=(0, 10))
        g_chk.pack(side="left", padx=(0, 10))
        b_chk.pack(side="left")
        if 0 in s["planes"]: r_chk.select()
        if 1 in s["planes"]: g_chk.select()
        if 2 in s["planes"]: b_chk.select()
        ToolTip(r_chk, _t("Corriger le canal Rouge", "Correct Red channel"))
        ToolTip(g_chk, _t("Corriger le canal Vert", "Correct Green channel"))
        ToolTip(b_chk, _t("Corriger le canal Bleu", "Correct Blue channel"))

        # ── Accélération (device) ──
        ctk.CTkLabel(body, text=_t("Accélération :", "Acceleration:"),
                     anchor="w").pack(fill="x", pady=(6, 2))
        _dev_options = [
            _t("Auto (CUDA si dispo)", "Auto (CUDA if available)"),
            "CPU",
            "CUDA",
            _t("TRT (torch.compile, RTX recommandé)", "TRT (torch.compile, RTX recommended)"),
        ]
        _dev_map_ui  = {
            "auto": _dev_options[0],
            "cpu":  "CPU",
            "cuda": "CUDA",
            "trt":  _dev_options[3],
        }
        _dev_map_key = {v: k for k, v in _dev_map_ui.items()}
        dev_row = ctk.CTkFrame(body, fg_color="transparent")
        dev_row.pack(fill="x", pady=(0, 6))
        dev_var = ctk.StringVar(value=_dev_map_ui.get(s.get("device", "auto"), _dev_options[0]))
        dev_combo = ctk.CTkOptionMenu(dev_row, variable=dev_var, values=_dev_options, width=280)
        dev_combo.pack(side="left")
        ToolTip(dev_combo, _t(
            "Auto : CUDA si PyTorch+GPU dispo, sinon CPU.\n"
            "CPU  : toujours PIL BoxBlur (~20 ms/1080p).\n"
            "CUDA : GPU PyTorch F.conv2d (~10-30× vs CPU).\n"
            "TRT  : torch.compile max-autotune. ~3 s warmup au 1er appel (Triton),\n"
            "       puis mis en cache. Gain réel ~10-20% vs CUDA sur RTX 3070 Ti / 5080.",
            "Auto: CUDA if PyTorch+GPU available, else CPU.\n"
            "CPU: always PIL BoxBlur (~20 ms/1080p).\n"
            "CUDA: GPU PyTorch F.conv2d (~10-30× vs CPU).\n"
            "TRT: torch.compile max-autotune. ~3 s warmup on 1st call (Triton),\n"
            "     then cached. ~10-20% gain over CUDA on RTX 3070 Ti / 5080."))

        # ── Image de référence ────────────────────────────────────────────────
        import tkinter.filedialog as _fd
        ctk.CTkLabel(body, text=_t("Image de référence (optionnel) :",
                                   "Reference image (optional):"),
                     anchor="w").pack(fill="x", pady=(10, 2))
        ctk.CTkLabel(body, text=_t(
            "Source alternative pour extraire les couleurs (au lieu du LQ).\n"
            "Recommandé : PNG/TIFF 16-bit pour éviter le banding.",
            "Alternative source for color extraction (instead of LQ).\n"
            "Recommended: 16-bit PNG/TIFF to avoid banding."),
            font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 4))
        ref_row = ctk.CTkFrame(body, fg_color="transparent")
        ref_row.pack(fill="x", pady=(0, 6))
        ref_path_var = ctk.StringVar(value=s.get("ref", ""))
        ref_entry = ctk.CTkEntry(ref_row, textvariable=ref_path_var, placeholder_text=_t(
            "Aucune (utilise LQ par défaut)", "None (uses LQ by default)"), width=200)
        ref_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        def _browse_ref():
            p = _fd.askopenfilename(
                title=_t("Sélectionner image de référence", "Select reference image"),
                filetypes=[
                    (_t("Images", "Images"), "*.png *.tif *.tiff *.jpg *.jpeg *.bmp *.webp"),
                    ("PNG 16-bit", "*.png"),
                    ("TIFF", "*.tif *.tiff"),
                    (_t("Tous", "All"), "*.*"),
                ])
            if p:
                ref_path_var.set(p)

        ctk.CTkButton(ref_row, text="📂", width=36, command=_browse_ref).pack(side="left")

        def _clear_ref():
            ref_path_var.set("")
        ctk.CTkButton(ref_row, text="✕", width=30, fg_color="gray30",
                      command=_clear_ref).pack(side="left", padx=(4, 0))

        # ── Boutons ──
        btn_row = ctk.CTkFrame(popup, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 15))

        def _apply():
            planes = [c for c, v in ((0, r_var), (1, g_var), (2, b_var)) if v.get()]
            if not planes:
                planes = [0, 1, 2]
            # Intensité (slider in popup)
            self._cf_settings["strength"] = round(float(cf_str_sld.get()), 2)
            self.settings.set("ups_colorfix_strength", str(self._cf_settings["strength"]))
            if method == "wavelet" and wav_sld is not None:
                self._cf_settings["wavelets"] = int(wav_sld.get())
                self.settings.set("ups_colorfix_wavelets", str(self._cf_settings["wavelets"]))
            elif method == "average" and rad_sld is not None:
                self._cf_settings["radius"] = int(rad_sld.get())
                self._cf_settings["fast"]   = bool(fast_var.get()) if fast_var else False
                self.settings.set("ups_colorfix_radius", str(self._cf_settings["radius"]))
                self.settings.set("ups_colorfix_fast",   self._cf_settings["fast"])
            self._cf_settings["planes"] = planes
            self.settings.set("ups_colorfix_planes", planes)
            dev_key = _dev_map_key.get(dev_var.get(), "auto")
            self._cf_settings["device"] = dev_key
            self.settings.set("ups_colorfix_device", dev_key)
            ref_p = ref_path_var.get().strip()
            self._cf_settings["ref"] = ref_p
            self.settings.set("ups_colorfix_ref", ref_p)
            popup.destroy()

        ctk.CTkButton(btn_row, text=_t("Appliquer", "Apply"), fg_color="#2ecc71",
                      command=_apply).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(btn_row, text=_t("Annuler", "Cancel"), fg_color="#e74c3c",
                      command=popup.destroy).pack(side="left", fill="x", expand=True)

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
        _top.pack(fill="x", pady=(0, 4))
        # GPU à droite — 38 px ≈ 1 cm de marge droite
        self._create_gpu_panel(_top).pack(side="right", padx=(0, 0), pady=4)
        _hdr = ctk.CTkFrame(_top, fg_color="transparent")
        _hdr.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(_hdr, text="Quick Upscale", font=("Roboto", 24, "bold"),
                     text_color="#3B8ED0", anchor="w").pack(fill="x")
        ctk.CTkLabel(_hdr, text=_t("Upscaler une image ou un dossier avec un modele entraine.", "Upscale an image or folder with a trained model."),
                     font=("Arial", 12), text_color="gray", anchor="w").pack(fill="x")

        # --- Model ---
        mrow = ctk.CTkFrame(f, fg_color="transparent")
        mrow.pack(fill="x", pady=1)
        ctk.CTkLabel(mrow, text=_t("Modele (.pth/.safetensors/.onnx) :", "Model (.pth/.safetensors/.onnx):"), width=220, anchor="w").pack(side="left")
        self._ups_model_var = StringVar(value=self.settings.get("ups_last_model", ""))
        self._ups_model_entry = ctk.CTkEntry(mrow, textvariable=self._ups_model_var)
        self._ups_model_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets["ups_model"] = self._ups_model_entry
        ctk.CTkButton(mrow, text="...", width=30, command=self._ups_pick_model).pack(side="left")

        # --- Input ---
        irow = ctk.CTkFrame(f, fg_color="transparent")
        irow.pack(fill="x", pady=1)
        ctk.CTkLabel(irow, text=_t("Source (image ou dossier) :", "Source (image or folder):"), width=200, anchor="w").pack(side="left")
        self._ups_input_var = StringVar(value=self.settings.get("ups_last_input", ""))
        self._ups_input_entry = ctk.CTkEntry(irow, textvariable=self._ups_input_var)
        self._ups_input_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets["ups_input"] = self._ups_input_entry
        ctk.CTkButton(irow, text=_t("Image", "Image"), width=60, command=self._ups_pick_image).pack(side="left", padx=(0, 2))
        ctk.CTkButton(irow, text=_t("Dossier", "Folder"), width=70, command=self._ups_pick_input_folder).pack(side="left")

        # --- Output ---
        orow = ctk.CTkFrame(f, fg_color="transparent")
        orow.pack(fill="x", pady=1)
        ctk.CTkLabel(orow, text=_t("Dossier sortie :", "Output folder:"), width=200, anchor="w").pack(side="left")
        self._ups_output_var = StringVar(value=self.settings.get("ups_last_output", ""))
        self._ups_output_entry = ctk.CTkEntry(orow, textvariable=self._ups_output_var)
        self._ups_output_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.widgets["ups_output"] = self._ups_output_entry
        self._ups_out_btn = ctk.CTkButton(orow, text="...", width=30, command=self._ups_pick_output_folder)
        self._ups_out_btn.pack(side="left")

        # --- Output options ---
        optrow = ctk.CTkFrame(f, fg_color="transparent", height=34)
        optrow.pack_propagate(False)
        optrow.pack(fill="x", pady=(2, 0))
        _sp = ctk.CTkFrame(optrow, fg_color="transparent", width=1, height=1); _sp.pack_propagate(False); _sp.pack(side="left", fill="x", expand=True)
        self._ups_same_folder = ctk.CTkCheckBox(
            optrow, text=_t("Meme dossier que la source", "Same folder as source"),
            command=self._ups_on_same_folder_toggle)
        self._ups_same_folder.pack(side="left", padx=(0, 15))
        if self.settings.get("ups_same_folder", False):
            self._ups_same_folder.select()
        self._ups_subfolder = ctk.CTkCheckBox(
            optrow, text=_t('Sous-dossier "upscaled/" (garde le nom original)', '"upscaled/" subfolder (keeps original name)'))
        self._ups_subfolder.pack(side="left", padx=(0, 15))
        if self.settings.get("ups_subfolder", True):
            self._ups_subfolder.select()
        self._ups_modelname = ctk.CTkCheckBox(
            optrow, text=_t("Nom du modele dans le fichier", "Model name in filename"))
        self._ups_modelname.pack(side="left", padx=(0, 12))
        if self.settings.get("ups_modelname", False):
            self._ups_modelname.select()

        # --- Serialisation (même ligne, à droite de "Nom du modèle") ----------
        ctk.CTkFrame(optrow, width=1, fg_color="gray40").pack(
            side="left", fill="y", padx=(0, 10), pady=2)
        self._ups_serialize = ctk.CTkCheckBox(
            optrow,
            text=_t("Sérialisation", "Serialization"),
            command=self._ups_on_serialize_toggle)
        self._ups_serialize.pack(side="left", padx=(0, 6))
        ToolTip(self._ups_serialize, _t(
            "Nommage séquentiel des fichiers de sortie (00000.png, 00001.png…).\n"
            "Utile pour les épisodes à réassembler en vidéo.\n"
            "Le suffix modèle/_UP est omis — fichiers prêts pour FFmpeg.",
            "Sequential output file naming (00000.png, 00001.png…).\n"
            "Useful for episodes to be reassembled as video.\n"
            "Model suffix/_UP omitted — files ready for FFmpeg."))
        if self.settings.get("ups_serialize", False):
            self._ups_serialize.select()
        ctk.CTkLabel(optrow, text=_t("Début :", "Start:")).pack(side="left", padx=(0, 3))
        self._ups_serialize_start = ctk.CTkEntry(optrow, width=55, placeholder_text="0")
        self._ups_serialize_start.pack(side="left")
        _sv = str(self.settings.get("ups_serialize_start", "0"))
        self._ups_serialize_start.insert(0, _sv)
        self._ups_on_serialize_toggle()  # état initial

        # --- Format sortie (même ligne, à droite de sérialisation) --------------
        ctk.CTkFrame(optrow, width=1, fg_color="gray40").pack(
            side="left", fill="y", padx=(15, 10), pady=2)
        ctk.CTkLabel(optrow, text=_t("Format sortie :", "Output format:")).pack(side="left", padx=(0, 4))
        self.widgets["ups_format"] = ctk.CTkOptionMenu(
            optrow, values=["PNG", "JPEG", "WEBP", "TIFF", "BMP"],
            width=90, command=self._ups_on_format_change)
        self.widgets["ups_format"].pack(side="left", padx=(0, 8))
        self.widgets["ups_format"].set(self.settings.get("ups_format", "PNG"))
        self.widgets["ups_bitdepth"] = ctk.CTkOptionMenu(
            optrow, values=[_t("8 bits", "8 bits"), _t("16 bits", "16 bits")], width=90)
        self.widgets["ups_bitdepth"].pack(side="left", padx=(0, 8))
        self.widgets["ups_bitdepth"].set(self.settings.get("ups_bitdepth", _t("8 bits", "8 bits")))
        ToolTip(self.widgets["ups_bitdepth"], _t(
            "16 bits uniquement pour PNG et TIFF.", "16-bit only for PNG and TIFF."))
        ctk.CTkLabel(optrow, text=_t("Qualité (JPEG/WEBP) :", "Quality (JPEG/WEBP):")).pack(side="left", padx=(0, 4))
        self.widgets["ups_quality"] = ctk.CTkOptionMenu(
            optrow, values=["70", "75", "80", "85", "90", "95", "100"], width=70)
        self.widgets["ups_quality"].pack(side="left")
        self.widgets["ups_quality"].set(self.settings.get("ups_quality", "95"))
        ToolTip(self.widgets["ups_quality"], _t(
            "Qualité de compression pour JPEG et WEBP.", "Compression quality for JPEG and WEBP."))
        _sp = ctk.CTkFrame(optrow, fg_color="transparent", width=1, height=1); _sp.pack_propagate(False); _sp.pack(side="left", fill="x", expand=True)
        self._ups_on_format_change(self.settings.get("ups_format", "PNG"))

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
        opts = ctk.CTkFrame(f, fg_color="transparent", height=34)
        opts.pack_propagate(False)
        opts.pack(fill="x", pady=(2, 0))
        _sp2 = ctk.CTkFrame(opts, fg_color="transparent", width=1, height=1); _sp2.pack_propagate(False); _sp2.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(opts, text="Scale:").pack(side="left")
        self.widgets["ups_scale"] = ctk.CTkOptionMenu(
            opts, values=["Auto", "1", "2", "3", "4", "8"], width=70)
        self.widgets["ups_scale"].pack(side="left", padx=5)
        self.widgets["ups_scale"].set(self.settings.get("ups_scale", "Auto"))
        ctk.CTkLabel(opts, text="Tile:").pack(side="left", padx=(15, 0))
        self.widgets["ups_tile"] = ctk.CTkOptionMenu(
            opts, values=["Auto", "128", "192", "256", "384", "512", "768", "1024", _t("0 (pas de tile)", "0 (no tiling)")], width=100)
        self.widgets["ups_tile"].pack(side="left", padx=5)
        self.widgets["ups_tile"].set(self.settings.get("ups_tile", "Auto"))
        ToolTip(self.widgets["ups_tile"], _t("Taille des tuiles. Auto = selon VRAM. 0 = pas de tiling.", "Tile size. Auto = based on VRAM. 0 = no tiling."))
        ctk.CTkLabel(opts, text="Tile Pad:").pack(side="left", padx=(15, 0))
        self.widgets["ups_tile_pad"] = ctk.CTkOptionMenu(
            opts, values=["Auto", "8", "16", "32", "48", "64"], width=80)
        self.widgets["ups_tile_pad"].pack(side="left", padx=5)
        self.widgets["ups_tile_pad"].set(self.settings.get("ups_tile_pad", "Auto"))
        ToolTip(self.widgets["ups_tile_pad"], _t("Overlap entre les tuiles. Auto = tile/8.", "Overlap between tiles. Auto = tile/8."))
        self.widgets["ups_amp"] = ctk.CTkCheckBox(opts, text="AMP (FP16)")
        self.widgets["ups_amp"].pack(side="left", padx=15)
        if self.settings.get("ups_amp", True):
            self.widgets["ups_amp"].select()

        # ── Color Fix (même ligne, à droite de AMP) ──────────────────────────
        ctk.CTkFrame(opts, width=1, fg_color="gray40").pack(
            side="left", fill="y", padx=(12, 10), pady=4)

        self.widgets["ups_colorfix"] = ctk.CTkCheckBox(
            opts, text="Color Fix", width=90,
            command=self._ups_on_colorfix_toggle)
        self.widgets["ups_colorfix"].pack(side="left")
        if self.settings.get("ups_colorfix", False):
            self.widgets["ups_colorfix"].select()
        ToolTip(self.widgets["ups_colorfix"], _t(
            "Corrige le color drift post-upscale en transférant les basses fréquences couleur du LQ vers le SR.\n"
            "Inspiré de vs_colorfix (pifroggi). Aucune dépendance supplémentaire.",
            "Fix color drift after upscale by transferring low-frequency color from LQ to SR.\n"
            "Inspired by vs_colorfix (pifroggi). No extra dependencies."))

        ctk.CTkLabel(opts, text=_t("Méthode :", "Method:")).pack(side="left", padx=(8, 4))
        self.widgets["ups_colorfix_method"] = ctk.CTkOptionMenu(
            opts, values=["wavelet", "average", "lab", "hist", "colormap"], width=105,
            command=self._ups_on_colorfix_method_change)
        self.widgets["ups_colorfix_method"].pack(side="left")
        self.widgets["ups_colorfix_method"].set(self.settings.get("ups_colorfix_method", "wavelet"))
        ToolTip(self.widgets["ups_colorfix_method"], _t(
            "wavelet ★ RECOMMANDÉ : ATWT multi-niveaux, préserve détails SR, corrige teinte/saturation.\n"
            "average : correction globale R/G/B, ~10× plus rapide.\n"
            "lab (v2.0) : corrige chroma (a*/b*) en espace LAB, préserve luminance SR.\n"
            "hist (v2.0) : histogram matching complet par canal.\n"
            "colormap (v2.0) : LUT quantile 64 points — rapide, léger, correction globale.",
            "wavelet ★ RECOMMENDED: multi-level ATWT, preserves SR detail, corrects hue/saturation.\n"
            "average: global R/G/B correction, ~10× faster.\n"
            "lab (v2.0): correct chroma (a*/b*) in LAB space, preserves SR luminance.\n"
            "hist (v2.0): full per-channel histogram matching.\n"
            "colormap (v2.0): 64-point quantile LUT — fast, lightweight, global correction."))

        self.widgets["ups_colorfix_settings_btn"] = ctk.CTkButton(
            opts, text=_t("⚙ ColorFix…", "⚙ ColorFix…"), width=100,
            command=self._ups_open_colorfix_settings)
        self.widgets["ups_colorfix_settings_btn"].pack(side="left", padx=(6, 0))
        ToolTip(self.widgets["ups_colorfix_settings_btn"], _t(
            "Réglages Color Fix : intensité, méthode, wavelets/radius, canaux R/G/B, device.",
            "Color Fix settings: intensity, method, wavelets/radius, R/G/B channels, device."))

        # ── Temporal Fix (même ligne que Color Fix — v2.5.7) ──────────────────
        ctk.CTkFrame(opts, width=1, fg_color="gray40").pack(
            side="left", fill="y", padx=(10, 10), pady=4)

        self.widgets["ups_tempfix"] = ctk.CTkCheckBox(
            opts,
            text=_t("Temporal Fix", "Temporal Fix"),
            width=120,
            command=self._ups_on_tempfix_toggle)
        self.widgets["ups_tempfix"].pack(side="left")
        if self.settings.get("ups_tempfix", False):
            self.widgets["ups_tempfix"].select()
        ToolTip(self.widgets["ups_tempfix"], _t(
            "Réduction du scintillement temporel (vidéo / séquences d'images).\n"
            "Fenêtre glissante N frames → blend adaptatif selon le mouvement.\n"
            "Zones statiques = lissées, zones mobiles = conservées.\n"
            "Latence = fenêtre÷2 frames. Nécessite PyTorch.",
            "Temporal flickering reduction (video / image sequences).\n"
            "Sliding window N frames → motion-adaptive blend.\n"
            "Static regions = smoothed, moving regions = preserved.\n"
            "Latency = window÷2 frames. Requires PyTorch."))

        self.widgets["ups_tempfix_btn"] = ctk.CTkButton(
            opts,
            text=_t("⚙ TF…", "⚙ TF…"),
            width=80,
            command=self._ups_open_tf_settings)
        self.widgets["ups_tempfix_btn"].pack(side="left", padx=(6, 0))

        # ── Undistort (v2.5.9 : bouton séparé) ───────────────────────────────
        ctk.CTkFrame(opts, width=1, fg_color="gray40").pack(
            side="left", fill="y", padx=(10, 10), pady=4)

        self.widgets["ups_undistort"] = ctk.CTkCheckBox(
            opts,
            text="Undistort",
            width=100,
            command=self._ups_on_undistort_toggle)
        self.widgets["ups_undistort"].pack(side="left")
        if self.settings.get("ups_undistort", False):
            self.widgets["ups_undistort"].select()
        ToolTip(self.widgets["ups_undistort"], _t(
            "Correction du jitter haute fréquence (shimmer sur les contours).\n"
            "Médiane temporelle du composant HF. Complémentaire au Temporal Fix.",
            "High-frequency jitter correction (shimmer on edges).\n"
            "Temporal median of HF component. Complements Temporal Fix."))

        self.widgets["ups_undistort_btn"] = ctk.CTkButton(
            opts,
            text=_t("⚙ UD…", "⚙ UD…"),
            width=80,
            command=self._ups_open_ud_settings)
        self.widgets["ups_undistort_btn"].pack(side="left", padx=(6, 0))

        self.widgets["ups_tempfix_info"] = ctk.CTkLabel(
            opts, text="", font=("Consolas", 10), text_color="gray60", width=130, anchor="w")
        self.widgets["ups_tempfix_info"].pack(side="left", padx=(8, 0))

        # Trailing spacer (centrage)
        _sp_end = ctk.CTkFrame(opts, fg_color="transparent", width=1, height=1)
        _sp_end.pack_propagate(False)
        _sp_end.pack(side="left", fill="x", expand=True)

        # Initialise le dict de réglages internes (persisté entre sessions)
        try:
            _cf_planes = self.settings.get("ups_colorfix_planes", [0, 1, 2])
            if not isinstance(_cf_planes, list):
                _cf_planes = [0, 1, 2]
        except Exception:
            _cf_planes = [0, 1, 2]
        self._cf_settings = {
            "wavelets": int(self.settings.get("ups_colorfix_wavelets", 4)),
            "radius":   int(self.settings.get("ups_colorfix_radius",   32)),
            "fast":     bool(self.settings.get("ups_colorfix_fast",    False)),
            "planes":   _cf_planes,
            "device":   str(self.settings.get("ups_colorfix_device",   "auto")),
            "ref":      str(self.settings.get("ups_colorfix_ref",      "")),
            "strength": float(self.settings.get("ups_colorfix_strength", 1.0)),
        }
        self._tf_settings = {
            "strength":           float(self.settings.get("ups_tempfix_strength",    0.5)),
            "window":             int(self.settings.get("ups_tempfix_window",        7)),
            "precision":          str(self.settings.get("ups_tempfix_precision",     "float32")),
            "tempfix_mode":       str(self.settings.get("ups_tempfix_mode",          "classic")),
            "undistort_enabled":  bool(self.settings.get("ups_undistort",            False)),
            "undistort_strength": float(self.settings.get("ups_undistort_strength",  0.35)),
            "undistort_window":   int(self.settings.get("ups_undistort_window",      5)),
            "undistort_mode":     str(self.settings.get("ups_undistort_mode",        "classic")),
        }
        self._ups_on_colorfix_toggle()

        # ── v2.5.5 : Persistent batch + Dandere2x ────────────────────────────────
        v255_row = ctk.CTkFrame(f, height=34)
        v255_row.pack_propagate(False)
        v255_row.pack(fill="x", pady=(2, 0))

        # Left spacer — balances right spacer to center content
        _sp_left_v255 = ctk.CTkFrame(v255_row, fg_color="transparent", width=1, height=1)
        _sp_left_v255.pack_propagate(False)
        _sp_left_v255.pack(side="left", fill="x", expand=True)

        self.widgets["ups_persistent"] = ctk.CTkCheckBox(
            v255_row,
            text=_t("Subprocess persistant", "Persistent subprocess"),
            width=175)
        self.widgets["ups_persistent"].pack(side="left", padx=(8, 6))
        if self.settings.get("ups_persistent", False):
            self.widgets["ups_persistent"].select()
        ToolTip(self.widgets["ups_persistent"], _t(
            "Charge le modèle une seule fois en VRAM pour tout le batch.\n"
            "Économise 2-5s par image sur les archs subprocess (SPANPlus, SMoSR…).\n"
            "Gain typique : 16-25h sur 30 000 frames.",
            "Load model once in VRAM for the entire batch.\n"
            "Saves 2-5s/image for subprocess archs (SPANPlus, SMoSR…).\n"
            "Typical gain: 16-25h on 30 000 frames."))

        ctk.CTkFrame(v255_row, width=1, fg_color="gray40").pack(
            side="left", fill="y", padx=(8, 10), pady=4)

        self.widgets["ups_dandere"] = ctk.CTkCheckBox(
            v255_row,
            text=_t("Skip frames", "Skip frames"),
            width=105,
            command=self._ups_on_dandere_toggle)
        self.widgets["ups_dandere"].pack(side="left")
        if self.settings.get("ups_dandere", False):
            self.widgets["ups_dandere"].select()
        ToolTip(self.widgets["ups_dandere"], _t(
            "Skip frames : saute les frames identiques ou quasi-identiques à la précédente\n"
            "(selon l'intensité du seuil réglé).\n"
            "Compare par blocs (taille = champ Blocs) — plus le bloc est petit, plus la détection est fine.\n"
            "Si tous les blocs sont sous le seuil → frame sautée, SR précédent réutilisé.\n"
            "Économise du GPU sur les scènes statiques ou peu animées.",
            "Skip frames: skips frames that are identical or nearly identical to the previous one\n"
            "(depending on the threshold intensity).\n"
            "Compares block by block (size = Blocks field) — smaller blocks = finer detection.\n"
            "If all blocks are below threshold → frame skipped, previous SR reused.\n"
            "Saves GPU on static or low-motion scenes."))

        ctk.CTkLabel(v255_row, text=_t("Seuil :", "Threshold:")).pack(side="left", padx=(8, 4))
        self.widgets["ups_dandere_threshold"] = ctk.CTkEntry(
            v255_row, width=55, placeholder_text="0.010")
        self.widgets["ups_dandere_threshold"].pack(side="left")
        _dt = str(self.settings.get("ups_dandere_threshold", "0.010"))
        self.widgets["ups_dandere_threshold"].insert(0, _dt)
        ToolTip(self.widgets["ups_dandere_threshold"], _t(
            "Seuil de différence (MAE par pixel, 0-1).\n"
            "0.010 = défaut — skip frames statiques, conserve micro-mouvements.\n"
            "0.005 = très strict (skip seulement frames parfaitement identiques).\n"
            "0.030+ = permissif (attention aux faux skip sur scènes dynamiques).",
            "Difference threshold (MAE per pixel, 0-1).\n"
            "0.010 = default — skips static frames, keeps micro-movements.\n"
            "0.005 = very strict (skip only perfectly identical frames).\n"
            "0.030+ = permissive (risk of false skip on dynamic scenes)."))

        ctk.CTkLabel(v255_row, text=_t("Blocs :", "Blocks:")).pack(side="left", padx=(8, 4))
        self.widgets["ups_dandere_block_size"] = ctk.CTkEntry(
            v255_row, width=45, placeholder_text="14")
        self.widgets["ups_dandere_block_size"].pack(side="left")
        _dbs = str(self.settings.get("ups_dandere_block_size", "14"))
        self.widgets["ups_dandere_block_size"].insert(0, _dbs)
        ToolTip(self.widgets["ups_dandere_block_size"], _t(
            "Taille des blocs en espace LQ (pixels).\n"
            "16 = standard (bon équilibre précision/vitesse).\n"
            "8 = plus précis, plus lent. 32 = plus rapide, moins précis.",
            "Block size in LQ space (pixels).\n"
            "16 = standard (good precision/speed balance).\n"
            "8 = more precise, slower. 32 = faster, less precise."))

        _sp_v255 = ctk.CTkFrame(v255_row, fg_color="transparent", width=1, height=1)
        _sp_v255.pack_propagate(False)
        _sp_v255.pack(side="left", fill="x", expand=True)

        self._ups_on_dandere_toggle()  # état initial

        self._ups_on_tempfix_toggle()  # état initial (widgets maintenant dans opts)

        # --- Run / Stop / progress / log ---
        run_row = ctk.CTkFrame(f, fg_color="transparent")
        run_row.pack(fill="x", pady=(12, 3))
        ctk.CTkButton(run_row, text=_t("⚡ Lancer Upscale", "⚡ Run Upscale"), fg_color="#2ecc71",
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
            text_color="#9B59B6", anchor="w")
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
        prev_frame = ctk.CTkFrame(paned, fg_color=("#E8E8E8", "#111827"), corner_radius=6)
        paned.add(prev_frame, minsize=80)
        for side, key, label in [("left", "ups_prev_in", _t("Source", "Source")),
                                  ("right", "ups_prev_out", _t("Resultat", "Result"))]:
            col = ctk.CTkFrame(prev_frame, fg_color="transparent")
            col.pack(side=side, fill="both", expand=True, padx=4, pady=4)
            ctk.CTkLabel(col, text=label, font=("Arial", 10), text_color="gray").pack()
            lbl = ctk.CTkLabel(col, text="—", fg_color=("#D0D0D0", "#1e293b"), corner_radius=4)
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
            messagebox.showerror(_t("Erreur", "Error"), _t("Selectionnez un modele.", "Select a model."))
            return
        if not inp:
            messagebox.showerror(_t("Erreur", "Error"), _t("Selectionnez une source.", "Select a source."))
            return
        if not base_out:
            messagebox.showerror(_t("Erreur", "Error"), _t("Selectionnez un dossier de sortie.", "Select an output folder."))
            return

        add_model_name = bool(self._ups_modelname.get())
        use_serialize  = bool(self._ups_serialize.get())
        _ser_raw = self._ups_serialize_start.get().strip()
        serialize_start = int(_ser_raw) if _ser_raw.isdigit() else 0
        self.settings.set("ups_last_model", model)
        self.settings.set("ups_last_input", inp)
        if not same_folder:
            self.settings.set("ups_last_output", base_out)
        self.settings.set("ups_subfolder", use_subfolder)
        self.settings.set("ups_modelname", add_model_name)
        self.settings.set("ups_serialize", use_serialize)
        self.settings.set("ups_serialize_start", str(serialize_start))

        scale_str = self.widgets["ups_scale"].get()
        scale = 0 if scale_str == "Auto" else int(scale_str)

        # ── Multi-scale model detection (Auto mode only) ──────────────────────
        # If Scale=Auto and model has MetaIGConv (SpanPP / SpanC family → multiple
        # trained scales in a single checkpoint), ask the user which scale to use
        # instead of silently defaulting to max.
        if scale == 0 and model and os.path.isfile(model):
            try:
                # Read MetaIGConv scale list to show multi-scale dialog.
                # In frozen mode: torch not importable in-process (CWD=_internal, numpy ABI clash).
                # Fix: torch-free safetensors read for .safetensors; subprocess for .pth.
                _ms_raw = None  # will hold list of int scale values if MetaIGConv found
                if model.endswith(".safetensors"):
                    # Torch-free: read safetensors header + binary blob
                    try:
                        import json as _json_st, struct as _struct_st
                        with open(model, "rb") as _f_st:
                            _n_st = _struct_st.unpack("<Q", _f_st.read(8))[0]
                            _hdr = _json_st.loads(_f_st.read(_n_st).decode("utf-8", "replace"))
                        if "MetaIGConv" in _hdr:
                            _meta = _hdr["MetaIGConv"]
                            _dtype, _shape = _meta.get("dtype", ""), _meta.get("shape", [])
                            _offs = _meta.get("data_offsets", [0, 0])
                            _nbytes = _offs[1] - _offs[0]
                            _elem = _nbytes // max(1, abs(_offs[1] - _offs[0]) // max(1, sum(_shape) if _shape else 1))
                            # Read the raw i64 or i32 integers
                            with open(model, "rb") as _f2:
                                _f2.seek(8 + _n_st + _offs[0])
                                _raw = _f2.read(_nbytes)
                            _fmt = "<" + ("q" if "I64" in _dtype.upper() else "i") * (_nbytes // (8 if "I64" in _dtype.upper() else 4))
                            _ms_raw = list(_struct_st.unpack(_fmt, _raw))
                    except Exception:
                        pass
                else:
                    # .pth file: run a quick subprocess with venv python to detect
                    _py_venv = _find_torch_python()
                    if _py_venv:
                        import subprocess as _sp_ms
                        _ms_script = (
                            "import torch, sys\n"
                            f"ck = torch.load({repr(model)}, map_location='cpu', weights_only=False)\n"
                            "sd = None\n"
                            "for k in ('params_ema','params_g','params','model','state_dict'):\n"
                            "    if k in ck: sd = ck[k]; break\n"
                            "if sd is None and any(x.endswith('.weight') for x in ck): sd = ck\n"
                            "if sd and 'MetaIGConv' in sd:\n"
                            "    print(','.join(str(int(v.item())) for v in sd['MetaIGConv']))\n"
                        )
                        _r_ms = _sp_ms.run(
                            [_py_venv, "-c", _ms_script],
                            capture_output=True, text=True, timeout=30,
                            creationflags=0x08000000 if sys.platform == "win32" else 0,
                        )
                        if _r_ms.returncode == 0 and _r_ms.stdout.strip():
                            try:
                                _ms_raw = [int(x) for x in _r_ms.stdout.strip().split(",")]
                            except Exception:
                                pass

                if _ms_raw is not None:
                    _ms = sorted(set(int(v) for v in _ms_raw))
                if _ms_raw is not None and len(_ms) > 1:
                        import tkinter as _tk
                        _result = [max(_ms)]   # default = max if dialog dismissed
                        _dlg = ctk.CTkToplevel(self)
                        _dlg.title(_t("Scale multi-échelle", "Multi-scale model"))
                        _dlg.resizable(False, False)
                        _dlg.transient(self.winfo_toplevel())
                        _dlg.grab_set()
                        # Center on main window
                        self.update_idletasks()
                        _pw = self.winfo_toplevel().winfo_width()
                        _ph = self.winfo_toplevel().winfo_height()
                        _px = self.winfo_toplevel().winfo_rootx()
                        _py = self.winfo_toplevel().winfo_rooty()
                        _dh = 130 + len(_ms) * 38
                        _dlg.geometry(f"300x{_dh}+{_px + _pw//2 - 150}+{_py + _ph//2 - _dh//2}")
                        ctk.CTkLabel(
                            _dlg,
                            text=_t(
                                f"Modèle multi-échelle détecté\nScales disponibles : {' / '.join(str(s)+'×' for s in _ms)}\n\nChoisissez le scale de sortie :",
                                f"Multi-scale model detected\nAvailable scales: {' / '.join(str(s)+'×' for s in _ms)}\n\nChoose output scale:"
                            ),
                            font=ctk.CTkFont(size=12), justify="center",
                        ).pack(pady=(14, 6), padx=14)
                        _var = _tk.IntVar(value=max(_ms))
                        for _s in _ms:
                            ctk.CTkRadioButton(
                                _dlg,
                                text=f"{_s}×",
                                variable=_var, value=_s,
                                font=ctk.CTkFont(size=13),
                            ).pack(anchor="w", padx=60, pady=2)
                        def _confirm_scale():
                            _result[0] = _var.get()
                            _dlg.destroy()
                        ctk.CTkButton(_dlg, text=_t("Confirmer", "Confirm"),
                                      command=_confirm_scale, width=130).pack(pady=12)
                        _dlg.wait_window()
                        scale = _result[0]
            except Exception:
                pass   # silently fall through — Auto continues with max scale

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

        # Color Fix
        _cf_enabled        = bool(self.widgets["ups_colorfix"].get())
        color_fix          = self.widgets["ups_colorfix_method"].get() if _cf_enabled else "none"
        _cfs               = getattr(self, "_cf_settings", {})
        color_fix_strength = float(_cfs.get("strength", 1.0))
        color_fix_wavelets = int(_cfs.get("wavelets", 4))
        color_fix_radius   = int(_cfs.get("radius",   32))
        color_fix_fast     = bool(_cfs.get("fast",    False))
        color_fix_planes   = _cfs.get("planes", [0, 1, 2])
        color_fix_device   = str(_cfs.get("device",  "auto"))
        color_fix_ref      = str(_cfs.get("ref",     ""))
        self.settings.set("ups_colorfix",         _cf_enabled)
        self.settings.set("ups_colorfix_method",  color_fix)
        self.settings.set("ups_colorfix_strength", str(color_fix_strength))

        # Persistent batch + Skip frames (v2.5.5)
        use_persistent = bool(self.widgets.get("ups_persistent", type("_D", (), {"get": lambda s: False})()).get())
        use_dandere    = bool(self.widgets.get("ups_dandere",    type("_D", (), {"get": lambda s: False})()).get())
        dandere_full_skip = use_dandere  # toujours mode Skip frames
        _dt_entry = self.widgets.get("ups_dandere_threshold", None)
        try:
            dandere_threshold = float(_dt_entry.get().strip() if _dt_entry else "0.010")
        except (ValueError, AttributeError):
            dandere_threshold = 0.010
        _dbs_entry = self.widgets.get("ups_dandere_block_size", None)
        try:
            dandere_block_size = max(4, int(_dbs_entry.get().strip() if _dbs_entry else "14"))
        except (ValueError, AttributeError):
            dandere_block_size = 14
        self.settings.set("ups_persistent",         use_persistent)
        self.settings.set("ups_dandere",            use_dandere)
        self.settings.set("ups_dandere_threshold",  str(dandere_threshold))
        self.settings.set("ups_dandere_block_size", str(dandere_block_size))

        # Temporal Fix + Undistort (v2.5.7)
        use_tempfix = bool(self.widgets.get("ups_tempfix", type("_D", (), {"get": lambda s: False})()).get())
        _tfs = getattr(self, "_tf_settings", {})
        tempfix_strength  = float(_tfs.get("strength",  0.5))
        tempfix_window    = int(_tfs.get("window",    7))
        tempfix_precision = str(_tfs.get("precision", "float32"))
        tempfix_mode         = str(_tfs.get("tempfix_mode",      "classic"))
        use_undistort        = bool(_tfs.get("undistort_enabled",  False))
        undistort_strength   = float(_tfs.get("undistort_strength", 0.35))
        undistort_window     = int(_tfs.get("undistort_window",     5))
        undistort_mode       = str(_tfs.get("undistort_mode",      "classic"))
        self.settings.set("ups_tempfix",           use_tempfix)
        self.settings.set("ups_undistort",         use_undistort)

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

        # Guard: prevent launching a second upscale while one is running
        _existing = getattr(self, "_ups_thread", None)
        if _existing is not None and _existing.is_alive():
            messagebox.showwarning(
                _t("Upscale en cours", "Upscale in progress"),
                _t("Un upscale est déjà en cours. Cliquez sur Stop puis attendez l'arrêt.",
                   "An upscale is already running. Click Stop and wait for it to finish.")
            )
            return

        # Reset & enable stop button
        self._ups_stop_flag = threading.Event()
        self.widgets["ups_stop_btn"].configure(state="normal", text="⏹ Stop")

        def callback(msg):
            def _ins():
                self.widgets["log_ups"].insert("end", msg + "\n")
                self.widgets["log_ups"].see("end")
            self._ui_update(_ins)

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
                                            stop_event=self._ups_stop_flag,
                                            color_fix=color_fix,
                                            color_fix_wavelets=color_fix_wavelets,
                                            color_fix_radius=color_fix_radius,
                                            color_fix_fast=color_fix_fast,
                                            color_fix_strength=color_fix_strength,
                                            color_fix_planes=color_fix_planes,
                                            color_fix_device=color_fix_device,
                                            color_fix_ref=color_fix_ref)
                    if ok:
                        set_progress(1.0)
                        callback(f"-> {out_path}")
                        self._ups_update_preview(inp, out_path)
                        self._ui_update(self._show_toast,
                                        _t("Upscale terminé", "Upscale complete"),
                                        os.path.basename(out_path),
                                        True, 4500, "notif_win11_upscale")
                    else:
                        callback(f"{_t('ERREUR : ', 'ERROR: ')}{msg}")
                        self._play_sound("error", "sound_error_enabled")
                        self._ui_update(self._show_toast, _t("Upscale échoué", "Upscale failed"), msg,
                                        False, 4500, "notif_win11_errors")

                elif os.path.isdir(inp):
                    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
                    files = sorted(
                        [fn for fn in os.listdir(inp)
                         if os.path.splitext(fn)[1].lower() in exts],
                        key=_natural_sort_key,
                    )
                    total = len(files)
                    # Pré-calcul du zero-padding pour la sérialisation
                    _pad_width = max(5, len(str(serialize_start + total - 1))) if use_serialize and total > 0 else 5
                    if total == 0:
                        callback(_t("Aucune image trouvee dans le dossier.", "No images found in folder."))
                        self._play_sound("warning", "sound_warning_enabled")
                        self._ui_update(self._show_toast,
                                        _t("Aucune image trouvée", "No images found"), inp,
                                        False, 4500, "notif_win11_errors")
                        return
                    success = 0
                    errors = []
                    import time as _time
                    _batch_start = _time.monotonic()
                    _img_times = []  # per-image durations for fps rolling average

                    # Stats dandere (dict mutable pour closure)
                    _dandere_stats = {"skipped": 0, "blocks_copied": 0}

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
                        # Dandere stats suffix (skip applies to both modes now)
                        _dsuffix = ""
                        if use_dandere and _dandere_stats["skipped"] > 0:
                            _dsuffix = f"   {_dandere_stats['skipped']} skip↩"
                        txt = (f"{_t('Écoulé', 'Elapsed')} : {_fmt(elapsed_s)}   "
                               f"ETA : {_fmt(eta_s)}   "
                               f"{_t('Vitesse', 'Speed')} : {fps:.2f} img/s{_dsuffix}   "
                               f"[{done}/{total_imgs}]")
                        self._ui_update(self.widgets["ups_timing"].configure, text=txt)

                    # Clear timing label at start
                    self._ui_update(self.widgets["ups_timing"].configure, text="")

                    # ── v2.5.5: Persistent batch session ─────────────────────────
                    from src.core.quick_upscale import (
                        PersistentBatchSession, _TRAINNER_VENV_PY,
                        dandere_should_skip,
                        _pil_to_float, _float_to_pil,
                    )

                    _session = None
                    if use_persistent:
                        _session = PersistentBatchSession(
                            venv_py=_TRAINNER_VENV_PY,
                            model_path=model,
                            tile_size=tile,
                            tile_pad=tile_pad,
                            use_amp=use_amp,
                            log=callback,
                            scale_hint=scale,  # user-selected scale (0=auto)
                        )
                        if not _session.start():
                            callback(_t("[Persistent] Démarrage échoué — mode normal utilisé",
                                       "[Persistent] Start failed — fallback to normal mode"))
                            _session = None

                    # ── v2.5.5: Skip frames state ─────────────────────────────────
                    _prev_lq_arr   = None
                    _prev_sr_arr   = None
                    _prev_in_path  = None   # path of previous LQ input
                    _prev_out_path = None   # path of previous SR output

                    # ── v2.5.7: Post-proc subprocess (TF + UD) ───────────────────
                    # Worker runs with venv Python → correct torch + CUDA.
                    # Frames passed as file paths on disk (same pattern as PersistentBatch).
                    from src.core.post_proc_session import PostProcSession
                    _pp = PostProcSession(venv_py=_TRAINNER_VENV_PY, log=callback)
                    _pp_started = _pp.start()
                    _pp_tf_latency = 0
                    _pp_ud_latency = 0
                    _tf_active = False  # TF effectively enabled in worker
                    _ud_active = False  # UD effectively enabled in worker

                    if use_tempfix and _pp_started:
                        try:
                            _pp_tf_latency = _pp.init_tf(
                                mode=tempfix_mode, strength=tempfix_strength,
                                window=tempfix_window, precision=tempfix_precision,
                            )
                            _tf_active = True
                            _tf_mode_lbl = (f"mode={tempfix_mode}" if tempfix_mode == "classic"
                                            else f"mode={tempfix_mode} (neural)")
                            callback(_t(
                                f"[TemporalFix] Activé — {_tf_mode_lbl} fenêtre={tempfix_window} "
                                f"str={tempfix_strength:.2f} {tempfix_precision} "
                                f"(latence {_pp_tf_latency} frames)",
                                f"[TemporalFix] Enabled — {_tf_mode_lbl} window={tempfix_window} "
                                f"str={tempfix_strength:.2f} {tempfix_precision} "
                                f"(latency {_pp_tf_latency} frames)"))
                        except Exception as _tfe0:
                            callback(f"[TemporalFix] Init échoué ({_tfe0}) — désactivé")

                    if use_undistort and _pp_started:
                        try:
                            _pp_ud_latency = _pp.init_ud(
                                mode=undistort_mode, strength=undistort_strength,
                                window=undistort_window, precision=tempfix_precision,
                            )
                            _ud_active = True
                            _ud_mode_lbl = (f"mode={undistort_mode}" if undistort_mode == "classic"
                                            else f"mode={undistort_mode} (neural)")
                            callback(_t(
                                f"[Undistort] Activé — {_ud_mode_lbl} fenêtre={undistort_window} "
                                f"str={undistort_strength:.2f} (latence {_pp_ud_latency} frames)",
                                f"[Undistort] Enabled — {_ud_mode_lbl} window={undistort_window} "
                                f"str={undistort_strength:.2f} (latency {_pp_ud_latency} frames)"))
                        except Exception as _ude0:
                            callback(f"[Undistort] Init échoué ({_ude0}) — désactivé")

                    try:
                        for i, fname in enumerate(files):
                            if self._ups_stop_flag.is_set():
                                callback(_t("Arrêt demandé par l'utilisateur.", "Stop requested by user."))
                                break
                            in_path = os.path.join(inp, fname)
                            base_name = os.path.splitext(fname)[0]
                            if use_serialize:
                                out_path = os.path.join(actual_out,
                                                         f"{serialize_start + i:0{_pad_width}d}{out_ext}")
                            else:
                                out_path = os.path.join(actual_out, f"{base_name}{out_suffix}{out_ext}")
                            callback(f"[{i + 1}/{total}] {fname}")

                            # ── Skip frames: load current LQ ─────────────────────
                            _curr_lq_arr = None
                            if use_dandere:
                                try:
                                    from PIL import Image as _PILImg
                                    _curr_lq_arr = _pil_to_float(_PILImg.open(in_path))
                                except Exception:
                                    pass

                            # ── Skip frames: full-frame skip ──────────────────────
                            # Block-level check: ANY block exceeding threshold
                            # → don't skip (catches head turns, local motion).
                            _skipped_this = False
                            if (use_dandere and dandere_full_skip and
                                    _prev_lq_arr is not None and _curr_lq_arr is not None and
                                    _prev_sr_arr is not None):
                                _safe_to_skip = dandere_should_skip(
                                    _prev_lq_arr, _curr_lq_arr,
                                    block_size=dandere_block_size,
                                    threshold=dandere_threshold,
                                )
                                if _safe_to_skip:
                                    callback(f"  [Skip frames] tous blocs < {dandere_threshold:.4f}")
                                    try:
                                        _float_to_pil(_prev_sr_arr).save(out_path)
                                    except Exception:
                                        pass
                                    _dandere_stats["skipped"] += 1
                                    _skipped_this = True
                                    success += 1
                                    _prev_lq_arr = _curr_lq_arr

                            if _skipped_this:
                                set_progress((i + 1) / total)
                                _update_timing(i + 1, total, 0.05)
                                continue

                            # ── Normal upscale ────────────────────────────────────
                            img_start = i / total
                            img_end = (i + 1) / total
                            def _sub_progress(v, s=img_start, e=img_end):
                                set_progress(s + v * (e - s))

                            _t0 = _time.monotonic()
                            ok, msg = False, ""   # initialisation sécurisée

                            # ── SR normal (persistent ou one-shot) ───────────────
                            if _session is not None and _session.is_ready:
                                ok, msg, _ = _session.infer(in_path, out_path)
                            else:
                                ok, msg = upscale_image(model, in_path, out_path, scale=scale,
                                                        tile_size=tile, tile_pad=tile_pad,
                                                        use_amp=use_amp, callback=callback,
                                                        progress_callback=_sub_progress,
                                                        out_format=out_format, bit_depth=bit_depth,
                                                        quality=quality,
                                                        stop_event=self._ups_stop_flag,
                                                        color_fix=color_fix,
                                                        color_fix_wavelets=color_fix_wavelets,
                                                        color_fix_radius=color_fix_radius,
                                                        color_fix_fast=color_fix_fast,
                                                        color_fix_strength=color_fix_strength,
                                                        color_fix_planes=color_fix_planes,
                                                        color_fix_device=color_fix_device,
                                                        color_fix_ref=color_fix_ref)
                            _img_dur = _time.monotonic() - _t0

                            if ok:
                                success += 1
                                self._ups_update_preview(in_path, out_path)

                                # ── Skip frames: mise à jour du cache prev ─────────
                                if use_dandere:
                                    _prev_in_path  = in_path
                                    _prev_out_path = out_path
                                    try:
                                        from PIL import Image as _PILImg
                                        _curr_sr_arr = _pil_to_float(_PILImg.open(out_path))
                                        _prev_sr_arr = _curr_sr_arr
                                    except Exception as _de:
                                        callback(f"  [Skip frames] Erreur cache : {_de}")

                                # ── Undistort + TemporalFix (subprocess worker) ────────
                                if _ud_active:
                                    try:
                                        _pp.push_ud(out_path)
                                    except Exception as _ude2:
                                        callback(f"  [Undistort] Erreur frame {i}: {_ude2}")
                                        _ud_active = False

                                if _tf_active:
                                    try:
                                        _pp.push_tf(out_path)
                                    except Exception as _tfe2:
                                        callback(f"  [TemporalFix] Erreur frame {i}: {_tfe2}")
                                        _tf_active = False
                            else:
                                errors.append(f"{fname}: {msg}")
                                self._play_sound("warning", "sound_warning_enabled")

                            if use_dandere:
                                _prev_lq_arr = _curr_lq_arr

                            set_progress(img_end)
                            _update_timing(i + 1, total, _img_dur)
                    finally:
                        if _session is not None:
                            _session.stop()
                        # ── Post-proc flush + stop (subprocess worker) ─────────────
                        if _pp_started:
                            if _ud_active and _pp_ud_latency > 0:
                                callback(_t("[Undistort] Flush des dernières frames…",
                                            "[Undistort] Flushing remaining frames…"))
                                try:
                                    _pp.flush_ud()
                                except Exception as _udf:
                                    callback(f"[Undistort] Erreur flush : {_udf}")
                            if _tf_active and _pp_tf_latency > 0:
                                callback(_t("[TemporalFix] Flush des dernières frames…",
                                            "[TemporalFix] Flushing remaining frames…"))
                                try:
                                    _pp.flush_tf()
                                except Exception as _tff:
                                    callback(f"[TemporalFix] Erreur flush : {_tff}")
                            _pp.stop()

                    # Skip frames — résumé final dans le log
                    if use_dandere:
                        _skip_n = _dandere_stats["skipped"]
                        _pct_sk = _skip_n / total * 100 if total > 0 else 0
                        callback(f"[Skip frames] {_skip_n}/{total} frames ({_pct_sk:.0f}% GPU économisé)")

                    # Final elapsed
                    _total_elapsed = _time.monotonic() - _batch_start
                    _elapsed_h, _el_rem = divmod(int(_total_elapsed), 3600)
                    _elapsed_m, _elapsed_s2 = divmod(_el_rem, 60)
                    _el_str = (f"{_elapsed_h}h{_elapsed_m:02d}m{_elapsed_s2:02d}s" if _elapsed_h
                               else f"{_elapsed_m}m{_elapsed_s2:02d}s" if _elapsed_m
                               else f"{_elapsed_s2}s")
                    _fps_final = success / _total_elapsed if _total_elapsed > 0 else 0
                    _final_dsuffix = ""
                    if use_dandere and _dandere_stats["skipped"] > 0:
                        _final_dsuffix = f"   {_dandere_stats['skipped']} skip"
                    self._ui_update(self.widgets["ups_timing"].configure,
                                    text=f"{_t('Terminé', 'Done')} — {_el_str} total, {_fps_final:.2f} img/s {_t('moy.', 'avg.')}{_final_dsuffix}")

                    summary = f"{_t('Termine', 'Done')} : {success}/{total} {_t('images traitees.', 'images processed.')}"
                    callback(summary)
                    for err in errors:
                        callback(f"  {_t('ERREUR : ', 'ERROR: ')}{err}")
                    toast_sub = f"{len(errors)} {_t('erreur(s)', 'error(s)')}" if errors else ""
                    self._ui_update(self._show_toast,
                                    f"Batch : {success}/{total} OK",
                                    toast_sub,
                                    success == total, 4500, "notif_win11_batch")
                else:
                    callback(f"{_t('Chemin introuvable : ', 'Path not found: ')}{inp}")
                    self._play_sound("error", "sound_error_enabled")
                    self._ui_update(self._show_toast,
                                    _t("Chemin introuvable", "Path not found"), inp,
                                    False, 4500, "notif_win11_errors")
            except ImportError:
                callback(_t("Module quick_upscale non disponible (PyTorch requis)", "Module quick_upscale not available (PyTorch required)"))
                self._play_sound("error", "sound_error_enabled")
                self._ui_update(self._show_toast,
                                _t("Module manquant", "Missing module"), _t("PyTorch requis", "PyTorch required"),
                                False, 4500, "notif_win11_errors")
            except Exception as e:
                _err = str(e)  # capture before Python 3.12+ deletes 'e'
                callback(f"{_t('Erreur', 'Error')} : {_err}")
                self._play_sound("error", "sound_error_enabled")
                self._ui_update(self._show_toast, _t("Erreur inattendue", "Unexpected error"), _err,
                                False, 4500, "notif_win11_errors")
            finally:
                # Always re-disable the stop button when worker exits
                self._ui_update(self.widgets["ups_stop_btn"].configure,
                                state="disabled", text="⏹ Stop")

        self._ups_thread = threading.Thread(target=worker, daemon=True)
        self._ups_thread.start()

    # ==========================================
    # PAGE 3: GÉNÉRATEUR LQ (enrichi)
    # ==========================================
    def create_page_generator(self):
        _ensure_pil()
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(
            f,
            _t("Générateur Dataset LQ", "LQ Dataset Generator"),
            _t("Générez des données basse qualité avec toutes les dégradations connues.",
               "Generate low-quality data with all known degradation types."),
        )

        # ── Chemins HQ / LQ ───────────────────────────────────────────────────
        self.add_path_row(f, _t("Source (HQ) :", "Source (HQ):"), "gen_hq", save_key="gen_hq")
        self.add_path_row(f, _t("Destination (LQ) :", "Destination (LQ):"), "gen_lq", save_key="gen_lq")
        for _k in ("gen_hq", "gen_lq"):
            _v = self.settings.get(_k, "")
            if _v:
                self.widgets[_k].insert(0, _v)

        # ── Scale + méthode ────────────────────────────────────────────────────
        p = ctk.CTkFrame(f, fg_color="transparent")
        p.pack(fill="x", pady=(5, 2))
        ctk.CTkLabel(p, text="Scale :").pack(side="left")
        self.widgets["gen_scale"] = ctk.CTkOptionMenu(p, values=["1", "2", "3", "4", "6", "8"], width=60)
        self.widgets["gen_scale"].pack(side="left", padx=5)
        self.widgets["gen_scale"].set("4")
        ctk.CTkLabel(p, text=_t("Méthode :", "Method:")).pack(side="left", padx=(15, 0))
        self.widgets["gen_method"] = ctk.CTkOptionMenu(
            p, values=["BICUBIC", "BILINEAR", "LANCZOS", "NEAREST", "BOX"], width=100
        )
        self.widgets["gen_method"].pack(side="left", padx=5)
        self.widgets["gen_method"].set("BICUBIC")

        # ── Helper: checkbox + slider row ─────────────────────────────────────
        def _row(parent, chk_key, label, tip, params):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", pady=2)
            chk = ctk.CTkCheckBox(row, text=label, width=145)
            chk.pack(side="left", padx=5)
            if tip:
                ToolTip(chk, tip)
            self.widgets[chk_key] = chk
            for item in params:
                lbl_t, key, default, item_tip = item[0], item[1], item[2], item[3]
                ctk.CTkLabel(row, text=lbl_t, width=85, anchor="e").pack(side="left", padx=(6, 0))
                if len(item) == 7:
                    self.gen_slider_val(row, key, default, item[4], item[5], item[6], item_tip)
                else:
                    e = ctk.CTkEntry(row, width=46)
                    e.insert(0, str(default))
                    e.pack(side="left", padx=(2, 0))
                    self.widgets[key] = e
                    if item_tip:
                        ToolTip(e, item_tip)
            return row

        # ── Onglets dégradations ───────────────────────────────────────────────
        tabs = ctk.CTkTabview(f, height=340)
        tabs.pack(fill="x", pady=5)

        tb = tabs.add(_t("🔧 Basique", "🔧 Basic"))
        tc = tabs.add(_t("🎨 Couleur", "🎨 Colour"))
        tv = tabs.add(_t("📺 Vidéo", "📺 Video"))
        ta = tabs.add(_t("⚙ Avancé", "⚙ Advanced"))
        tp = tabs.add(_t("🎲 Pipeline", "🎲 Pipeline"))

        # ── Tab Basique ────────────────────────────────────────────────────────
        sb = ctk.CTkScrollableFrame(tb, fg_color="transparent", height=280)
        sb.pack(fill="both", expand=True)

        _row(sb, "gen_blur", _t("Flou gaussien", "Gaussian Blur"), None,
             [("σ:", "gen_blur_sigma", 1.0, _t("σ 0.1-8.0", "σ 0.1-8.0"), 0.1, 8.0, 0.1)])
        _row(sb, "gen_noise", _t("Bruit gaussien", "Gaussian Noise"), None,
             [("σ:", "gen_noise_sigma", 10, _t("σ 1-50", "σ 1-50"), 1, 50, 1)])
        _row(sb, "gen_jpeg", "Compression JPEG", None,
             [(_t("Qualité :", "Qualité :"), "gen_jpeg_q", 50, _t("Qualité 1-99", "Quality 1-99"), 1, 99, 1)])
        _row(sb, "gen_color_jitter", "Color Jitter (±10%)",
             _t("Variation aléatoire luminosité/contraste/couleur ±10%.", "Random brightness/contrast/colour ±10%."), [])
        _row(sb, "gen_sharpen", _t("Netteté", "Sharpening"),
             _t("Légère netteté PIL avant dégradation.", "Mild PIL sharpening before degradation."),
             [(_t("Intensité :", "Intensité :"), "gen_sharpen_amt", 1.5, _t("1.0=neutre, 2.0=fort", "1.0=neutral, 2.0=strong"), 1.0, 3.0, 0.1)])

        # ── Tab Couleur — 2 colonnes ───────────────────────────────────────────
        _cc = ctk.CTkFrame(tc, fg_color="transparent")
        _cc.pack(fill="both", expand=True, padx=4, pady=4)
        _cc.columnconfigure(0, weight=1)
        _cc.columnconfigure(1, weight=1)
        sc_l = ctk.CTkFrame(_cc, fg_color="transparent")
        sc_l.grid(row=0, column=0, sticky="new", padx=(0, 8))
        ctk.CTkFrame(_cc, width=1, fg_color="gray30").grid(row=0, column=1, sticky="ns", padx=(0, 8))
        sc_r = ctk.CTkFrame(_cc, fg_color="transparent")
        sc_r.grid(row=0, column=2, sticky="new")
        _cc.columnconfigure(2, weight=1)

        _row(sc_l, "gen_posterize", _t("Postérisation", "Posterization"),
             _t("Réduit profondeur bits/canal (2-8).", "Reduces bit depth per channel (2-8)."),
             [("bits:", "gen_posterize_bits", 4, _t("8=nul, 4=visible, 2=extrême", "8=none, 4=visible, 2=extreme"), 2, 8, 1)])
        _row(sc_l, "gen_banding", "Banding",
             _t("Quantification couleur → bandes dans les dégradés.", "Colour quantization → bands in gradients."),
             [(_t("Niveaux :", "Niveaux :"), "gen_banding_levels", 32, _t("8-256 niveaux palette", "8-256 palette levels"), 8, 256, 8)])
        _row(sc_l, "gen_ca", _t("Aberration chrom.", "Chromatic Aberr."),
             _t("Décale canaux R/B horizontalement (frange de couleur).", "Shifts R/B channels horizontally (colour fringing)."),
             [(_t("Pixels :", "Pixels :"), "gen_ca_shift", 2, _t("1-10 px", "1-10 px"), 1, 10, 1)])
        _row(sc_l, "gen_disc_blur", _t("Flou disque (bokeh)", "Disc Blur (bokeh)"),
             _t("Simule le flou de défocalisation (bokeh). Lent sur grandes images.", "Simulates defocus bokeh. Slow on large images."),
             [(_t("Rayon :", "Rayon :"), "gen_disc_blur_r", 4.0, _t("Rayon 1-12", "Radius 1-12"), 1.0, 12.0, 0.5)])
        _row(sc_l, "gen_vignette", "Vignette",
             _t("Assombrit les bords (objectif).", "Darkens edges (lens effect)."),
             [(_t("Force :", "Force :"), "gen_vignette_str", 0.4, _t("Force 0.1-1.0", "Strength 0.1-1.0"), 0.1, 1.0, 0.05)])

        _row(sc_r, "gen_halo", _t("Halo (wtp_halo)", "Halo (wtp_halo)"),
             _t("Renforcement des bords haute fréquence (ringing).", "High-frequency edge ringing."),
             [(_t("Force :", "Force :"), "gen_halo_str", 0.3, _t("Force 0.1-1.0", "Strength 0.1-1.0"), 0.1, 1.0, 0.05),
              (_t("Rayon :", "Rayon :"), "gen_halo_r", 6, _t("Rayon 2-15", "Radius 2-15"), 2, 15, 1)])
        _row(sc_r, "gen_saturation", "Saturation",
             _t("Ajuste la saturation globale (désaturation ou hypersaturation).", "Global saturation adjustment."),
             [(_t("Facteur :", "Facteur :"), "gen_sat_factor", 0.8, _t("0=gris, 1=neutre, 2=hyper", "0=grey, 1=neutral, 2=hyper"), 0.0, 2.0, 0.05)])
        _row(sc_r, "gen_color_levels", _t("Niveaux couleur", "Colour Levels"),
             _t("Compresse la plage dynamique (écrase hautes/basses lumières).", "Compresses dynamic range."),
             [(_t("Seuil haut :", "Seuil haut :"), "gen_col_high", 230, _t("Seuil haut 200-255", "High threshold 200-255"), 200, 255, 5),
              (_t("Seuil bas :", "Seuil bas :"), "gen_col_low", 10, _t("Seuil bas 0-40", "Low threshold 0-40"), 0, 40, 2)])
        _row(sc_r, "gen_quantize_depth", _t("Quantize depth", "Quantize depth"),
             _t("Quantification uniforme à N bits (4-8).", "Uniform N-bit quantization (4-8)."),
             [("bits:", "gen_qdepth_bits", 6, _t("4-8 bits", "4-8 bits"), 4, 8, 1)])

        # ── Tab Vidéo — 2 colonnes ─────────────────────────────────────────────
        _cv = ctk.CTkFrame(tv, fg_color="transparent")
        _cv.pack(fill="both", expand=True, padx=4, pady=4)
        _cv.columnconfigure(0, weight=1)
        _cv.columnconfigure(2, weight=1)
        sv_l = ctk.CTkFrame(_cv, fg_color="transparent")
        sv_l.grid(row=0, column=0, sticky="new", padx=(0, 8))
        ctk.CTkFrame(_cv, width=1, fg_color="gray30").grid(row=0, column=1, sticky="ns", padx=(0, 8))
        sv_r = ctk.CTkFrame(_cv, fg_color="transparent")
        sv_r.grid(row=0, column=2, sticky="new")

        _row(sv_l, "gen_chroma_sub", _t("Sous-éch. chroma", "Chroma Subsampling"),
             _t("Simule 4:2:0 (MPEG/H.264) : réduit la résolution couleur 2×.", "Simulates 4:2:0 (MPEG/H.264): halves colour resolution."), [])
        _row(sv_l, "gen_aliasing", "Aliasing",
             _t("Nearest-neighbor ↓↑ → artefacts escalier sur diagonales.", "Nearest-neighbour ↓↑ → staircase on diagonals."),
             [(_t("Force :", "Force :"), "gen_aliasing_str", 0.75, _t("0.5-1.0", "0.5-1.0"), 0.5, 1.0, 0.05)])
        _row(sv_l, "gen_interlace_weave", "Interlace weave",
             _t("Dents de peigne sur les bords (artefact VHS/DVD).", "Comb teeth on edges (VHS/DVD artifact)."),
             [(_t("Force :", "Force :"), "gen_weave_str", 0.8, _t("0.1-1.0", "0.1-1.0"), 0.1, 1.0, 0.05)])
        _row(sv_l, "gen_interlace_flicker", "Flicker",
             _t("Luminosité alternée lignes paires/impaires (CRT).", "Alternating brightness on even/odd lines (CRT)."),
             [(_t("Amplitude :", "Amplitude :"), "gen_flicker_amp", 0.22, _t("0.01-0.5", "0.01-0.5"), 0.01, 0.5, 0.01)])
        _row(sv_l, "gen_interlace_blend", "Field blend",
             _t("Ghosting entre champs entrelacés (flou de mouvement).", "Ghosting between interlaced fields (motion blur)."),
             [(_t("Mélange :", "Mélange :"), "gen_blend_mix", 0.55, _t("0.1-1.0", "0.1-1.0"), 0.1, 1.0, 0.05)])
        _row(sv_l, "gen_scanlines", "Scanlines CRT",
             _t("Assombrit une ligne sur 2-8 (CRT, émulateurs).", "Darkens every 2-8 lines (CRT, emulators)."),
             [(_t("Période :", "Période :"), "gen_scanlines_period", 3, _t("2-8", "2-8"), 2, 8, 1),
              (_t("Noirceur :", "Noirceur :"), "gen_scanlines_dark", 0.35, _t("0.1-0.6", "0.1-0.6"), 0.1, 0.6, 0.05)])

        _row(sv_r, "gen_vhs", "VHS / Analog",
             _t("Saignée chroma + dropout de lignes (artefact VHS).", "Chroma bleeding + scanline dropout (VHS artifact)."),
             [(_t("Force :", "Force :"), "gen_vhs_str", 0.3, _t("0.1-1.0", "0.1-1.0"), 0.1, 1.0, 0.05)])
        _row(sv_r, "gen_screentone", "Screentone",
             _t("Trame halftone (manga, presse imprimée).", "Halftone dot screen (manga, print)."),
             [(_t("Taille :", "Taille :"), "gen_screen_sz", 6, _t("Cellule 3-20 px", "Cell 3-20 px"), 3, 20, 1)])
        _row(sv_r, "gen_dithering", _t("Dithering", "Dithering"),
             _t("Floyd-Steinberg avec palette réduite.", "Floyd-Steinberg with reduced palette."),
             [(_t("Couleurs :", "Couleurs :"), "gen_dither_col", 4, _t("4-64 couleurs", "4-64 colours"), 4, 64, 4)])
        _row(sv_r, "gen_sinusoidal", _t("Sinusoïdal", "Sinusoidal"),
             _t("Distorsion sinusoïdale horizontale (interférence).", "Horizontal sinusoidal distortion (interference)."),
             [(_t("Période :", "Période :"), "gen_sin_period", 300, _t("50-600", "50-600"), 50, 600, 10),
              (_t("Amplitude :", "Amplitude :"), "gen_sin_alpha", 0.2, _t("0.05-0.5", "0.05-0.5"), 0.05, 0.5, 0.05)])
        _row(sv_r, "gen_pixel_shift", _t("Pixel Shift", "Pixel Shift"),
             _t("Décalage global horizontal ou vertical (glitch).", "Global horizontal or vertical pixel shift (glitch)."),
             [(_t("Pixels :", "Pixels :"), "gen_pxsh_amt", 3, _t("1-20 px", "1-20 px"), 1, 20, 1)])

        # ── Tab Avancé ─────────────────────────────────────────────────────────
        sa = ctk.CTkScrollableFrame(ta, fg_color="transparent", height=280)
        sa.pack(fill="both", expand=True)

        _row(sa, "gen_film_grain", "Film Grain",
             _t("Grain luminance-dépendant (fort sur midtones).", "Luminance-dependent grain (strong on midtones)."),
             [("σ:", "gen_grain_sigma", 0.08, _t("0.01-0.3", "0.01-0.3"), 0.01, 0.3, 0.01),
              (_t("Taille :", "Taille :"), "gen_grain_size", 1, _t("1-3 px", "1-3 px"), 1, 3, 1)])
        _row(sa, "gen_oversharp", "Oversharpening",
             _t("Halos USM (caméras consommateur, vidéo compressée).", "USM halos (consumer cameras, compressed video)."),
             [(_t("Force :", "Force :"), "gen_oversharp_amt", 1.4, _t("0.5-3.0", "0.5-3.0"), 0.5, 3.0, 0.1)])
        _row(sa, "gen_motion_blur", _t("Motion Blur", "Motion Blur"),
             _t("Flou de mouvement horizontal (caméra en déplacement).", "Horizontal motion blur (camera movement)."),
             [(_t("Pixels :", "Pixels :"), "gen_mb_px", 7, _t("3-31 px (impair)", "3-31 px (odd)"), 3, 31, 2)])
        _row(sa, "gen_codec", _t("Artefacts codec H.264", "H.264 Codec Artifacts"),
             _t("Simule les artefacts de bloc MPEG via double-JPEG.", "Simulates MPEG block artifacts via double-JPEG."),
             [(_t("JPEG 1 :", "JPEG 1 :"), "gen_codec_q1", 30, _t("JPEG pass 1 (1-70)", "JPEG pass 1 (1-70)"), 1, 70, 1),
              (_t("JPEG 2 :", "JPEG 2 :"), "gen_codec_q2", 60, _t("JPEG pass 2 (1-99)", "JPEG pass 2 (1-99)"), 1, 99, 1)])
        _row(sa, "gen_salt_pepper", _t("Sel & Poivre", "Salt & Pepper"),
             _t("Pixels noirs/blancs aléatoires (capteur défectueux).", "Random black/white pixels (faulty sensor)."),
             [(_t("Quantité :", "Quantité :"), "gen_sp_amt", 0.01, _t("0.001-0.1", "0.001-0.1"), 0.001, 0.1, 0.001)])
        _row(sa, "gen_halation", _t("Halation (film)", "Halation (film)"),
             _t("Saignée lumineuse dans les hautes lumières (film argentique).", "Light bloom from highlights (analog film)."),
             [(_t("Force :", "Force :"), "gen_hal_str", 0.2, _t("0.05-0.8", "0.05-0.8"), 0.05, 0.8, 0.05)])
        _row(sa, "gen_autocrop", _t("Auto-crop patches", "Auto-crop patches"),
             _t("Crop aléatoire avant dégradation (simule wtp_dataset_destroyer).", "Random crop before degradation (mimics wtp_dataset_destroyer)."),
             [(_t("Taille :", "Taille :"), "gen_crop_sz", 256, _t("Taille patch 64-512", "Patch size 64-512"), 64, 512, 32)])

        # ── Tab Pipeline ───────────────────────────────────────────────────────
        spipe = ctk.CTkScrollableFrame(tp, fg_color="transparent", height=280)
        spipe.pack(fill="both", expand=True)

        ctk.CTkLabel(spipe, text=_t("Répétitions du pipeline", "Pipeline repetitions"),
                     font=("Roboto", 12, "bold"), anchor="w").pack(fill="x", padx=8, pady=(8, 2))
        pr = ctk.CTkFrame(spipe, fg_color="transparent")
        pr.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(pr, text=_t("Passes :", "Passes:")).pack(side="left")
        self.widgets["gen_passes"] = ctk.CTkOptionMenu(pr, values=["1", "2", "3", "4", "5"], width=60)
        self.widgets["gen_passes"].pack(side="left", padx=5)
        self.widgets["gen_passes"].set("1")
        ToolTip(self.widgets["gen_passes"],
                _t("Applique le pipeline complet N fois sur chaque image.\n"
                   "Chaque passe dégrade davantage (accumulation réaliste).",
                   "Applies the full pipeline N times per image.\n"
                   "Each pass degrades further (realistic accumulation)."))

        ctk.CTkLabel(spipe, text=_t("Probabilité par dégradation", "Per-degradation probability"),
                     font=("Roboto", 12, "bold"), anchor="w").pack(fill="x", padx=8, pady=(12, 2))
        ctk.CTkLabel(spipe,
                     text=_t("Probabilité globale d'appliquer chaque dégradation activée (0=jamais, 1=toujours).\n"
                              "Utile pour créer de la variété : même config → résultats différents.",
                              "Global probability of applying each enabled degradation (0=never, 1=always).\n"
                              "Useful for variety: same config → different results."),
                     text_color="#888", font=("Roboto", 10), justify="left", anchor="w",
                     wraplength=580).pack(fill="x", padx=12, pady=(0, 4))
        pp = ctk.CTkFrame(spipe, fg_color="transparent")
        pp.pack(fill="x", padx=8)
        ctk.CTkLabel(pp, text=_t("Probabilité :", "Probability:")).pack(side="left")
        self.gen_slider_val(pp, "gen_prob", 1.0, 0.0, 1.0, 0.05,
                            _t("1.0 = déterministe, 0.5 = 50% chance par dégradation",
                               "1.0 = deterministic, 0.5 = 50% chance per degradation"))

        ctk.CTkLabel(spipe, text=_t("Mode probabiliste", "Probabilistic mode"),
                     font=("Roboto", 12, "bold"), anchor="w").pack(fill="x", padx=8, pady=(12, 2))
        pm = ctk.CTkFrame(spipe, fg_color="transparent")
        pm.pack(fill="x", padx=8)
        self.widgets["gen_prob_mode"] = ctk.CTkOptionMenu(
            pm,
            values=[
                _t("Indépendant (par dégradation)", "Independent (per degradation)"),
                _t("Par image (1 tirage)", "Per image (1 draw)"),
            ],
            width=280,
        )
        self.widgets["gen_prob_mode"].pack(side="left", padx=5)
        ToolTip(self.widgets["gen_prob_mode"],
                _t("Indépendant : chaque dégradation tirée séparément.\n"
                   "Par image : 1 tirage décide si TOUTES les dégradations s'appliquent.",
                   "Independent: each degradation drawn separately.\n"
                   "Per image: 1 draw decides if ALL degradations apply."))

        # ── Bouton Run + progress ──────────────────────────────────────────────
        ctk.CTkButton(
            f, text=_t("▶  Lancer la Génération", "▶  Run Generation"),
            fg_color="#E67E22", font=("Roboto", 14, "bold"), height=38,
            command=self.run_gen,
        ).pack(fill="x", pady=(8, 4))
        self.widgets["prog_gen"] = ctk.CTkProgressBar(f)
        self.widgets["prog_gen"].pack(fill="x")
        self.widgets["prog_gen"].set(0)
        self.widgets["lbl_gen"] = ctk.CTkLabel(f, text=_t("En attente…", "Waiting…"))
        self.widgets["lbl_gen"].pack()

        # ── Aperçu Avant / Après ───────────────────────────────────────────────
        self.add_header(f, _t("Aperçu Avant / Après", "Before / After Preview"))
        prev_frame = ctk.CTkFrame(f, fg_color="transparent")
        prev_frame.pack(fill="x", pady=4)

        btn_load_prev = ctk.CTkButton(
            prev_frame,
            text=_t("📂 Charger image de test", "📂 Load test image"),
            fg_color="#2980b9", width=180,
            command=self._gen_load_preview,
        )
        btn_load_prev.pack(side="left", padx=5)
        btn_apply_prev = ctk.CTkButton(
            prev_frame,
            text=_t("⚡ Appliquer dégradations", "⚡ Apply degradations"),
            fg_color="#16a085", width=180,
            command=self._gen_apply_preview,
        )
        btn_apply_prev.pack(side="left", padx=5)
        self.widgets["lbl_gen_prev_status"] = ctk.CTkLabel(prev_frame, text="", text_color="#888")
        self.widgets["lbl_gen_prev_status"].pack(side="left", padx=5)

        # ── Contrôle hauteur canvas ────────────────────────────────────────────
        h_ctrl = ctk.CTkFrame(f, fg_color="transparent")
        h_ctrl.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(h_ctrl, text=_t("Hauteur aperçu :", "Preview height:"), anchor="w").pack(side="left", padx=(5, 4))
        _prev_h_lbl = ctk.CTkLabel(h_ctrl, text="280 px", width=55, anchor="w")
        _prev_h_lbl.pack(side="left")
        _prev_h_slider = ctk.CTkSlider(h_ctrl, from_=120, to=700, width=220, number_of_steps=58)
        _prev_h_slider.set(280)
        _prev_h_slider.pack(side="left", padx=6)

        import tkinter as _tk

        panels = ctk.CTkFrame(f, fg_color="transparent")
        panels.pack(fill="x", pady=4)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)

        lbl_before = ctk.CTkLabel(panels, text=_t("Avant (HQ original)", "Before (HQ original)"),
                                  font=("Roboto", 11, "bold"), text_color="#3498db")
        lbl_before.grid(row=0, column=0, pady=(0, 2))
        lbl_after = ctk.CTkLabel(panels, text=_t("Après (LQ dégradé)", "After (LQ degraded)"),
                                 font=("Roboto", 11, "bold"), text_color="#e67e22")
        lbl_after.grid(row=0, column=1, pady=(0, 2))

        canvas_before = _tk.Canvas(panels, bg="#1e1e2e", highlightthickness=1,
                                   highlightbackground="#3498db", height=280)
        canvas_before.grid(row=1, column=0, padx=4, sticky="nsew")
        canvas_before._zoom = 1.0
        canvas_before._pil = None
        canvas_before._photo = None

        canvas_after = _tk.Canvas(panels, bg="#1e1e2e", highlightthickness=1,
                                  highlightbackground="#e67e22", height=280)
        canvas_after.grid(row=1, column=1, padx=4, sticky="nsew")
        canvas_after._zoom = 1.0
        canvas_after._pil = None
        canvas_after._photo = None

        def _on_prev_height(v):
            h = int(float(v))
            _prev_h_lbl.configure(text=f"{h} px")
            canvas_before.configure(height=h)
            canvas_after.configure(height=h)
            _canvas_draw(canvas_before)
            _canvas_draw(canvas_after)

        _prev_h_slider.configure(command=_on_prev_height)

        def _canvas_draw(cv):
            if cv._pil is None:
                return
            cw = max(50, cv.winfo_width())
            ch = max(50, cv.winfo_height())
            auto = min(cw / cv._pil.width, ch / cv._pil.height)
            scale = auto * cv._zoom
            nw = max(1, int(cv._pil.width * scale))
            nh = max(1, int(cv._pil.height * scale))
            from PIL import ImageTk as _ITK
            resample = Image.LANCZOS if scale < 1 else Image.NEAREST
            thumb = cv._pil.resize((nw, nh), resample)
            photo = _ITK.PhotoImage(thumb)
            cv._photo = photo
            cv.delete("all")
            cv.create_image(cw // 2, ch // 2, anchor="center", image=photo)

        def _bind_zoom(cv):
            def _wheel(e):
                cv._zoom = max(0.05, min(20.0, cv._zoom * (1.15 if e.delta > 0 else 1 / 1.15)))
                _canvas_draw(cv)
            cv.bind("<MouseWheel>", _wheel)
            cv.bind("<Configure>", lambda e, c=cv: _canvas_draw(c))

        _bind_zoom(canvas_before)
        _bind_zoom(canvas_after)
        self._canvas_draw = _canvas_draw
        self.widgets["gen_canvas_before"] = canvas_before
        self.widgets["gen_canvas_after"] = canvas_after

        # Placeholder text initial
        def _canvas_placeholder(cv, msg):
            cv.delete("all")
            cv.create_text(cv.winfo_width() // 2 or 190, cv.winfo_height() // 2 or 140,
                           text=msg, fill="#555", font=("Roboto", 11))
        canvas_before.after(100, lambda: _canvas_placeholder(
            canvas_before, _t("Aucune image", "No image")))
        canvas_after.after(100, lambda: _canvas_placeholder(
            canvas_after, _t("Appuyer sur ⚡", "Press ⚡")))

        self._gen_preview_img = None  # PIL Image original pour preview

        return f

    # ── Preview helpers ────────────────────────────────────────────────────────

    def _gen_load_preview(self):
        from tkinter import filedialog as _fd
        path = _fd.askopenfilename(
            title=_t("Charger image test", "Load test image"),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Tous", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self._gen_preview_img = img
            self._gen_show_preview(self.widgets["gen_canvas_before"], img)
            self.widgets["lbl_gen_prev_status"].configure(
                text=f"{os.path.basename(path)} — {img.width}×{img.height}", text_color="#2ecc71"
            )
            # Clear after panel
            cv = self.widgets["gen_canvas_after"]
            cv._pil = None
            cv.delete("all")
            cv.create_text(cv.winfo_width() // 2 or 190, cv.winfo_height() // 2 or 140,
                           text=_t("Appuyer sur ⚡", "Press ⚡"), fill="#555", font=("Roboto", 11))
        except Exception as e:
            self.widgets["lbl_gen_prev_status"].configure(text=str(e), text_color="#e74c3c")

    def _gen_apply_preview(self):
        if self._gen_preview_img is None:
            self.widgets["lbl_gen_prev_status"].configure(
                text=_t("Charger une image d'abord.", "Load an image first."), text_color="#e67e22"
            )
            return
        self.widgets["lbl_gen_prev_status"].configure(
            text=_t("Traitement…", "Processing…"), text_color="#3498db"
        )

        def _run():
            try:
                opts = self._collect_gen_opts()
                img = self._gen_preview_img.copy()
                # Scale down for preview
                scale = opts["scale"]
                new_w = max(1, img.width // scale)
                new_h = max(1, img.height // scale)
                img = img.resize((new_w, new_h), opts["method"])
                img = self._apply_gen_degradations(img, opts)
                def _show():
                    self._gen_show_preview(self.widgets["gen_canvas_after"], img)
                    self.widgets["lbl_gen_prev_status"].configure(
                        text=_t("Aperçu mis à jour.", "Preview updated."), text_color="#2ecc71"
                    )
                self.after(0, _show)
            except Exception as e:
                def _err(e=e):
                    self.widgets["lbl_gen_prev_status"].configure(text=str(e), text_color="#e74c3c")
                self.after(0, _err)

        threading.Thread(target=_run, daemon=True).start()

    def _gen_show_preview(self, canvas_widget, pil_img):
        canvas_widget._pil = pil_img
        canvas_widget._zoom = 1.0
        self._canvas_draw(canvas_widget)

    # ── Build opts dict from widgets ──────────────────────────────────────────

    def _collect_gen_opts(self):
        W = self.widgets
        method_map = {
            "BICUBIC": Image.BICUBIC, "BILINEAR": Image.BILINEAR,
            "LANCZOS": Image.LANCZOS, "NEAREST": Image.NEAREST, "BOX": Image.BOX,
        }

        def _fv(key, default):
            try:
                return float(W[key].get() or default)
            except Exception:
                return float(default)

        def _iv(key, default):
            try:
                return int(float(W[key].get() or default))
            except Exception:
                return int(default)

        def _bv(key):
            try:
                return bool(W[key].get())
            except Exception:
                return False

        return {
            "scale": _iv("gen_scale", 4),
            "method": method_map.get(W["gen_method"].get(), Image.BICUBIC),
            # Basique
            "blur": _bv("gen_blur"), "blur_sigma": _fv("gen_blur_sigma", 1.0),
            "noise": _bv("gen_noise"), "noise_sigma": _fv("gen_noise_sigma", 10),
            "jpeg": _bv("gen_jpeg"), "jpeg_q": _iv("gen_jpeg_q", 50),
            "color_jitter": _bv("gen_color_jitter"),
            "sharpen": _bv("gen_sharpen"), "sharpen_amt": _fv("gen_sharpen_amt", 1.5),
            # Couleur
            "posterize": _bv("gen_posterize"), "posterize_bits": _iv("gen_posterize_bits", 4),
            "banding": _bv("gen_banding"), "banding_levels": _iv("gen_banding_levels", 32),
            "ca": _bv("gen_ca"), "ca_shift": _iv("gen_ca_shift", 2),
            "disc_blur": _bv("gen_disc_blur"), "disc_blur_r": _fv("gen_disc_blur_r", 4.0),
            "vignette": _bv("gen_vignette"), "vignette_str": _fv("gen_vignette_str", 0.4),
            "halo": _bv("gen_halo"), "halo_str": _fv("gen_halo_str", 0.3), "halo_r": _iv("gen_halo_r", 6),
            "saturation": _bv("gen_saturation"), "sat_factor": _fv("gen_sat_factor", 0.8),
            "color_levels": _bv("gen_color_levels"),
            "col_high": _fv("gen_col_high", 230), "col_low": _fv("gen_col_low", 10),
            "quantize_depth": _bv("gen_quantize_depth"), "qdepth_bits": _iv("gen_qdepth_bits", 6),
            # Vidéo
            "chroma_sub": _bv("gen_chroma_sub"),
            "aliasing": _bv("gen_aliasing"), "aliasing_str": _fv("gen_aliasing_str", 0.75),
            "interlace_weave": _bv("gen_interlace_weave"), "weave_str": _fv("gen_weave_str", 0.8),
            "interlace_flicker": _bv("gen_interlace_flicker"), "flicker_amp": _fv("gen_flicker_amp", 0.22),
            "interlace_blend": _bv("gen_interlace_blend"), "blend_mix": _fv("gen_blend_mix", 0.55),
            "scanlines": _bv("gen_scanlines"),
            "scanlines_period": _iv("gen_scanlines_period", 3), "scanlines_dark": _fv("gen_scanlines_dark", 0.35),
            "vhs": _bv("gen_vhs"), "vhs_str": _fv("gen_vhs_str", 0.3),
            "screentone": _bv("gen_screentone"), "screen_sz": _iv("gen_screen_sz", 6),
            "dithering": _bv("gen_dithering"), "dither_col": _iv("gen_dither_col", 4),
            "sinusoidal": _bv("gen_sinusoidal"),
            "sin_period": _fv("gen_sin_period", 300), "sin_alpha": _fv("gen_sin_alpha", 0.2),
            "pixel_shift": _bv("gen_pixel_shift"), "pxsh_amt": _iv("gen_pxsh_amt", 3),
            # Avancé
            "film_grain": _bv("gen_film_grain"),
            "grain_sigma": _fv("gen_grain_sigma", 0.08), "grain_size": _iv("gen_grain_size", 1),
            "oversharp": _bv("gen_oversharp"), "oversharp_amt": _fv("gen_oversharp_amt", 1.4),
            "motion_blur": _bv("gen_motion_blur"), "mb_px": _iv("gen_mb_px", 7),
            "codec": _bv("gen_codec"), "codec_q1": _iv("gen_codec_q1", 30), "codec_q2": _iv("gen_codec_q2", 60),
            "salt_pepper": _bv("gen_salt_pepper"), "sp_amt": _fv("gen_sp_amt", 0.01),
            "halation": _bv("gen_halation"), "hal_str": _fv("gen_hal_str", 0.2),
            "autocrop": _bv("gen_autocrop"), "crop_sz": _iv("gen_crop_sz", 256),
            # Pipeline
            "passes": _iv("gen_passes", 1),
            "prob": _fv("gen_prob", 1.0),
            "prob_per_image": "Par image" in self.widgets["gen_prob_mode"].get(),
        }

    def run_gen(self):
        hq = self.widgets["gen_hq"].get()
        lq = self.widgets["gen_lq"].get()
        if not hq or not lq:
            messagebox.showerror(_t("Erreur", "Error"), _t("Dossiers requis.", "Folders required."))
            return
        self.settings.set("gen_hq", hq)
        self.settings.set("gen_lq", lq)
        opts = self._collect_gen_opts()
        threading.Thread(target=self._process_gen, args=(hq, lq, opts), daemon=True).start()

    # ── Apply all degradations to one PIL image ────────────────────────────────

    def _apply_gen_degradations(self, img, opts):
        import io as _io
        from PIL import ImageFilter as _IF, ImageEnhance as _IE, ImageOps as _IO

        prob = opts.get("prob", 1.0)
        prob_per_image = opts.get("prob_per_image", False)

        # Determine if this image is "activated" in per-image mode
        if prob_per_image:
            apply_all = (random.random() < prob)
        else:
            apply_all = None  # per-degradation mode

        def _should(flag):
            if not flag:
                return False
            if prob_per_image:
                return apply_all
            return random.random() < prob

        # ── Basique ────────────────────────────────────────────────────────────
        if _should(opts.get("sharpen")):
            img = img.filter(_IF.SHARPEN)

        if _should(opts.get("blur")):
            img = img.filter(_IF.GaussianBlur(radius=opts["blur_sigma"]))

        if _should(opts.get("noise")):
            _ensure_numpy()
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, opts["noise_sigma"], arr.shape)
            img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

        if _should(opts.get("color_jitter")):
            img = _IE.Brightness(img).enhance(random.uniform(0.9, 1.1))
            img = _IE.Contrast(img).enhance(random.uniform(0.9, 1.1))
            img = _IE.Color(img).enhance(random.uniform(0.9, 1.1))

        if _should(opts.get("jpeg")):
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=opts["jpeg_q"])
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # ── Couleur ────────────────────────────────────────────────────────────
        if _should(opts.get("posterize")):
            img = _IO.posterize(img, max(1, min(8, opts["posterize_bits"])))

        if _should(opts.get("banding")):
            levels = max(2, min(256, opts["banding_levels"]))
            img = img.quantize(colors=levels, method=Image.Quantize.FASTOCTREE).convert("RGB")

        if _should(opts.get("saturation")):
            img = _IE.Color(img).enhance(opts["sat_factor"])

        try:
            from src.core.otf_preview import (
                apply_chromatic_aberration, apply_disc_blur_pil, apply_vignette_pil,
                apply_wtp_halo, apply_color_levels, apply_quantize_depth_pil,
                apply_vhs, apply_aliasing_pil, apply_interlace_weave_pil,
                apply_interlace_flicker_pil, apply_interlace_blend_pil,
                apply_scanlines_pil, apply_screentone, apply_dithering,
                apply_sinusoidal, apply_subsampling_wtp, apply_pixel_shift,
                apply_film_grain_pil, apply_oversharpening_pil,
                apply_salt_pepper, apply_halation,
            )
            _otf_ok = True
        except Exception:
            _otf_ok = False

        if _otf_ok:
            if _should(opts.get("ca")):
                img = apply_chromatic_aberration(img, shift_range=(opts["ca_shift"], opts["ca_shift"]))
            if _should(opts.get("disc_blur")):
                img = apply_disc_blur_pil(img, radius_range=(opts["disc_blur_r"], opts["disc_blur_r"]))
            if _should(opts.get("vignette")):
                img = apply_vignette_pil(img, strength_range=(opts["vignette_str"], opts["vignette_str"]))
            if _should(opts.get("halo")):
                img = apply_wtp_halo(img, strength_range=(opts["halo_str"], opts["halo_str"]),
                                     radius_range=(opts["halo_r"], opts["halo_r"]))
            if _should(opts.get("color_levels")):
                img = apply_color_levels(img,
                                         high_range=(opts["col_high"], opts["col_high"]),
                                         low_range=(opts["col_low"], opts["col_low"]))
            if _should(opts.get("quantize_depth")):
                img = apply_quantize_depth_pil(img, bits_range=(opts["qdepth_bits"], opts["qdepth_bits"]))
            # Vidéo
            if _should(opts.get("chroma_sub")):
                img = apply_subsampling_wtp(img, subsampling_format="4:2:0")
            if _should(opts.get("aliasing")):
                img = apply_aliasing_pil(img, scale_range=(opts["aliasing_str"], opts["aliasing_str"]))
            if _should(opts.get("interlace_weave")):
                img = apply_interlace_weave_pil(img, strength_range=(opts["weave_str"], opts["weave_str"]))
            if _should(opts.get("interlace_flicker")):
                img = apply_interlace_flicker_pil(img, strength_range=(opts["flicker_amp"], opts["flicker_amp"]))
            if _should(opts.get("interlace_blend")):
                img = apply_interlace_blend_pil(img, strength_range=(opts["blend_mix"], opts["blend_mix"]))
            if _should(opts.get("scanlines")):
                img = apply_scanlines_pil(img,
                                          spacing_range=(opts["scanlines_period"], opts["scanlines_period"]),
                                          strength_range=(opts["scanlines_dark"], opts["scanlines_dark"]))
            if _should(opts.get("vhs")):
                img = apply_vhs(img, strength_range=(opts["vhs_str"], opts["vhs_str"]))
            if _should(opts.get("screentone")):
                img = apply_screentone(img, dot_size=opts["screen_sz"])
            if _should(opts.get("dithering")):
                img = apply_dithering(img, n_colors=opts["dither_col"])
            if _should(opts.get("sinusoidal")):
                img = apply_sinusoidal(img,
                                       shape_range=(opts["sin_period"], opts["sin_period"]),
                                       alpha_range=(opts["sin_alpha"], opts["sin_alpha"]))
            if _should(opts.get("pixel_shift")):
                img = apply_pixel_shift(img, shift_range=(opts["pxsh_amt"], opts["pxsh_amt"]))
            # Avancé
            if _should(opts.get("film_grain")):
                img = apply_film_grain_pil(img,
                                           strength_range=(opts["grain_sigma"], opts["grain_sigma"]),
                                           size_range=(opts["grain_size"], opts["grain_size"]))
            if _should(opts.get("oversharp")):
                img = apply_oversharpening_pil(img,
                                               strength_range=(opts["oversharp_amt"], opts["oversharp_amt"]))
            if _should(opts.get("salt_pepper")):
                img = apply_salt_pepper(img, amount_range=(opts["sp_amt"], opts["sp_amt"]))
            if _should(opts.get("halation")):
                img = apply_halation(img, strength_range=(opts["hal_str"], opts["hal_str"]))

        # Motion blur — PIL kernel
        if _should(opts.get("motion_blur")):
            ksize = max(3, opts["mb_px"] | 1)  # ensure odd
            kernel_data = [0] * (ksize * ksize)
            mid = ksize // 2
            for x in range(ksize):
                kernel_data[mid * ksize + x] = 1
            from PIL import ImageFilter as _IF2
            img = img.filter(_IF2.Kernel(size=(ksize, ksize), kernel=kernel_data, scale=ksize, offset=0))

        # H.264 codec simulation: double JPEG pass
        if _should(opts.get("codec")):
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=opts["codec_q1"])
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            buf2 = _io.BytesIO()
            img.save(buf2, format="JPEG", quality=opts["codec_q2"])
            buf2.seek(0)
            img = Image.open(buf2).convert("RGB")

        return img

    def _process_gen(self, hq, lq, opts):
        try:
            os.makedirs(lq, exist_ok=True)
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            files = [x for x in os.listdir(hq) if os.path.splitext(x)[1].lower() in exts]
            total = len(files)
            if total == 0:
                self._ui_update(messagebox.showinfo, _t("Info", "Info"),
                                _t("Aucune image trouvée.", "No images found."))
                return

            passes = max(1, opts.get("passes", 1))

            _live_every = max(1, total // 20)  # update canvas ~20 times max
            for i, fname in enumerate(files):
                try:
                    src_path = os.path.join(hq, fname)
                    img = Image.open(src_path).convert("RGB")
                    _hq_snap = img.copy()  # original HQ avant crop/resize/dégradation

                    # Auto-crop patch
                    if opts.get("autocrop"):
                        csz = opts["crop_sz"]
                        if img.width >= csz and img.height >= csz:
                            x0 = random.randint(0, img.width - csz)
                            y0 = random.randint(0, img.height - csz)
                            img = img.crop((x0, y0, x0 + csz, y0 + csz))

                    scale = opts["scale"]
                    new_w = max(1, img.width // scale)
                    new_h = max(1, img.height // scale)
                    img = img.resize((new_w, new_h), opts["method"])

                    for _ in range(passes):
                        img = self._apply_gen_degradations(img, opts)

                    if i % _live_every == 0:
                        _lq_snap = img.copy()
                        self._ui_update(self._gen_show_preview,
                                        self.widgets["gen_canvas_before"], _hq_snap)
                        self._ui_update(self._gen_show_preview,
                                        self.widgets["gen_canvas_after"], _lq_snap)

                    out_name = os.path.splitext(fname)[0] + ".png"
                    img.save(os.path.join(lq, out_name))
                except Exception:
                    pass

                prog = (i + 1) / total
                self._ui_update(self.widgets["prog_gen"].set, prog)
                self._ui_update(self.widgets["lbl_gen"].configure, text=f"{i+1}/{total}")

            self._ui_update(
                messagebox.showinfo, "OK",
                f"{_t('Terminé', 'Done')} — {total} {_t('images traitées.', 'images processed.')}",
            )
        except Exception as e:
            self._ui_update(messagebox.showerror, _t("Erreur", "Error"), str(e))

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
        ctk.CTkLabel(_hdr, text=_t("Convertisseur de Modèle", "Model Converter"), font=("Roboto", 24, "bold"),
                     text_color="#3B8ED0", anchor="w").pack(fill="x")
        ctk.CTkLabel(_hdr, text=_t("Convertir un modèle vers différents formats d'inférence.", "Convert a model to different inference formats."),
                     font=("Arial", 12), text_color="gray", anchor="w").pack(fill="x")
        self.add_path_row(f, _t("Modèle source (.pth/.safetensors) :", "Source model (.pth/.safetensors):"), "conv_pth", is_file=True)
        self.add_path_row(f, _t("Dossier sortie :", "Output folder:"), "conv_output")

        # Format options
        self.add_header(f, _t("Formats de sortie", "Output Formats"))
        fmt = ctk.CTkFrame(f, fg_color="transparent")
        fmt.pack(fill="x", pady=5)

        self.widgets["chk_onnx"] = ctk.CTkCheckBox(fmt, text="ONNX")
        self.widgets["chk_onnx"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_onnx"], _t("Export ONNX — compatible avec ONNX Runtime, DirectML.\n[+] Portable, multiplateforme.\n[-] Performances moyennes vs TensorRT.", "ONNX export — compatible with ONNX Runtime, DirectML.\n[+] Portable, cross-platform.\n[-] Average performance vs TensorRT."))

        self.widgets["chk_fp16"] = ctk.CTkCheckBox(fmt, text=_t("FP16 (demi-précision)", "FP16 (half-precision)"))
        self.widgets["chk_fp16"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_fp16"], _t("Conversion en Float16.\n[+] Modèle 2x plus petit, inférence plus rapide.\n[-] Légère perte de précision (invisible en pratique).", "Convert to Float16.\n[+] 2x smaller model, faster inference.\n[-] Slight precision loss (invisible in practice)."))

        self.widgets["chk_bf16"] = ctk.CTkCheckBox(fmt, text=_t("BF16 (bfloat16)", "BF16 (bfloat16)"))
        self.widgets["chk_bf16"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_bf16"], _t("Conversion en BFloat16 (Brain Float16).\n[+] Même plage dynamique que FP32, moins de risque NaN.\n[+] Idéal RTX 3000+ (Ampere) et A100.\n[-] Non supporté sur GPU < Ampere.", "Convert to BFloat16 (Brain Float16).\n[+] Same dynamic range as FP32, lower NaN risk.\n[+] Ideal for RTX 3000+ (Ampere) and A100.\n[-] Not supported on GPUs older than Ampere."))

        self.widgets["chk_safetensors"] = ctk.CTkCheckBox(fmt, text="SafeTensors")
        self.widgets["chk_safetensors"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_safetensors"], _t("Conversion vers SafeTensors (Hugging Face).\n[+] Sécurisé (pas d'exécution de code), chargement rapide.\n[+] Standard pour partager des modèles.", "Convert to SafeTensors (Hugging Face).\n[+] Secure (no code execution), fast loading.\n[+] Standard for sharing models."))

        fmt2 = ctk.CTkFrame(f, fg_color="transparent")
        fmt2.pack(fill="x", pady=5)

        self.widgets["chk_pth"] = ctk.CTkCheckBox(fmt2, text=_t("PTH (poids seuls)", "PTH (weights only)"))
        self.widgets["chk_pth"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_pth"], _t("Sauvegarde les poids seuls en .pth (sans métadonnées d'entraînement).\n[+] Fichier plus léger, compatible PyTorch standard.\n[+] Utiliser pour créer un modèle propre depuis un .state ou checkpoint.", "Save weights-only .pth (no training metadata).\n[+] Lighter file, standard PyTorch compatible.\n[+] Use to create a clean model from a .state or checkpoint."))

        self.widgets["chk_ncnn"] = ctk.CTkCheckBox(fmt2, text="NCNN")
        self.widgets["chk_ncnn"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_ncnn"], _t("Export NCNN (Tencent) — via ONNX.\n[+] Léger, optimisé pour mobile et CPU.\n[-] Nécessite onnx2ncnn installé séparément.", "NCNN export (Tencent) — via ONNX.\n[+] Lightweight, optimized for mobile and CPU.\n[-] Requires onnx2ncnn installed separately."))

        self.widgets["chk_tensorrt"] = ctk.CTkCheckBox(fmt2, text="TensorRT")
        self.widgets["chk_tensorrt"].pack(side="left", padx=10)
        ToolTip(self.widgets["chk_tensorrt"], _t("Export TensorRT (NVIDIA) — via ONNX.\n[+] Performances maximales sur GPU NVIDIA.\n[-] Spécifique à la carte GPU (pas portable).\n[-] Nécessite TensorRT SDK installé.", "TensorRT export (NVIDIA) — via ONNX.\n[+] Maximum performance on NVIDIA GPU.\n[-] GPU-specific (not portable).\n[-] Requires TensorRT SDK installed."))

        # Paramètres avancés
        self.add_header(f, _t("Paramètres", "Parameters"))
        params = ctk.CTkFrame(f, fg_color="transparent")
        params.pack(fill="x", pady=5)

        ctk.CTkLabel(params, text=_t("Architecture :", "Architecture:")).pack(side="left")
        self.widgets["conv_arch"] = ctk.CTkOptionMenu(
            params, values=["Auto-detect", "omnisr", "span", "realplksr", "compact", "esrgan", "hat", "dat", "swinir"], width=130
        )
        self.widgets["conv_arch"].pack(side="left", padx=5)
        self.widgets["conv_arch"].set("Auto-detect")

        ctk.CTkLabel(params, text="Scale :").pack(side="left", padx=(15, 0))
        self.widgets["conv_scale"] = ctk.CTkOptionMenu(params, values=["Auto", "1", "2", "3", "4", "8"], width=60)
        self.widgets["conv_scale"].pack(side="left", padx=5)
        self.widgets["conv_scale"].set("Auto")

        params2 = ctk.CTkFrame(f, fg_color="transparent")
        params2.pack(fill="x", pady=5)
        ctk.CTkLabel(params2, text=_t("Opset ONNX :", "ONNX Opset:")).pack(side="left")
        self.widgets["conv_opset"] = ctk.CTkOptionMenu(params2, values=["17", "14", "13", "11", "9"], width=60)
        self.widgets["conv_opset"].pack(side="left", padx=5)
        self.widgets["conv_opset"].set("17")
        ToolTip(self.widgets["conv_opset"], _t("Version opset ONNX (17 = recommandé, max compatibilité TensorRT 8+).\nBaisser si erreur d'export vers outils anciens.", "ONNX opset version (17 = recommended, max compatibility with TensorRT 8+).\nLower if export fails with older tools."))

        ctk.CTkLabel(params2, text="TF32 :").pack(side="left", padx=(15, 0))
        self.widgets["conv_tf32"] = ctk.CTkCheckBox(params2, text=_t("Activer (matmul rapide RTX 3000+)", "Enable (fast matmul RTX 3000+)"))
        self.widgets["conv_tf32"].pack(side="left", padx=5)
        ToolTip(self.widgets["conv_tf32"], _t("Active torch.backends.cuda.matmul.allow_tf32 pendant la conversion.\n[+] Accélère le traitement sur RTX 3000+ (Ampere) de 5-10%.\n[ℹ] Option CUDA, ne change pas le format du modèle.", "Enables torch.backends.cuda.matmul.allow_tf32 during conversion.\n[+] Speeds up processing on RTX 3000+ (Ampere) by 5-10%.\n[ℹ] CUDA option only — does not change model format."))

        ctk.CTkButton(f, text=_t("Convertir", "Convert"), fg_color="#3498db", command=self.run_conv).pack(fill="x", pady=15)
        self.widgets["log_conv"] = ctk.CTkTextbox(f, height=120)
        self.widgets["log_conv"].pack(fill="both", expand=True, pady=5)
        return f

    def run_conv(self):
        pth = self.widgets["conv_pth"].get()
        if not pth or not os.path.exists(pth):
            messagebox.showerror(_t("Erreur", "Error"), _t("Modèle introuvable.", "Model not found."))
            return

        self.widgets["log_conv"].delete("1.0", "end")
        out_dir = self.widgets["conv_output"].get() or os.path.dirname(pth)
        os.makedirs(out_dir, exist_ok=True)

        do_onnx = bool(self.widgets["chk_onnx"].get())
        do_fp16 = bool(self.widgets["chk_fp16"].get())
        do_bf16 = bool(self.widgets["chk_bf16"].get())
        do_safe = bool(self.widgets["chk_safetensors"].get())
        do_pth  = bool(self.widgets["chk_pth"].get())
        do_ncnn = bool(self.widgets["chk_ncnn"].get())
        do_trt  = bool(self.widgets["chk_tensorrt"].get())
        do_tf32 = bool(self.widgets["conv_tf32"].get())
        opset   = int(self.widgets["conv_opset"].get())

        if not any([do_onnx, do_fp16, do_bf16, do_safe, do_pth, do_ncnn, do_trt]):
            messagebox.showinfo(_t("Info", "Info"), _t("Sélectionnez au moins un format.", "Select at least one format."))
            return

        def log(msg):
            self._ui_update(self.widgets["log_conv"].insert, "end", msg + "\n")

        # ── Portable / no-torch fallback: find engine Python ───────────────
        _torch_py = _find_torch_python()
        if not _torch_py:
            messagebox.showerror(
                _t("PyTorch introuvable", "PyTorch not found"),
                _t(
                    "PyTorch n'est pas disponible.\n\n"
                    "Installez-le dans NeoSR ou TraiNNer-Redux\n"
                    "(Réglages → Système & Dépendances).",
                    "PyTorch is not available.\n\n"
                    "Install it in NeoSR or TraiNNer-Redux\n"
                    "(Settings → System & Dependencies)."
                )
            )
            return
        _use_subprocess = _torch_py != sys.executable

        def worker():
            try:
                if _use_subprocess:
                    import json as _json, tempfile as _tmp
                    # Generate a self-contained conversion script and run it
                    # with the engine Python that has torch available.
                    script = f"""
import sys, os, json
import torch

pth      = {repr(pth)}
out_dir  = {repr(out_dir)}
do_pth   = {do_pth}
do_fp16  = {do_fp16}
do_bf16  = {do_bf16}
do_safe  = {do_safe}
do_onnx  = {do_onnx}
do_ncnn  = {do_ncnn}
do_trt   = {do_trt}
do_tf32  = {do_tf32}
opset    = {opset}

def log(m): print(m, flush=True)

if do_tf32:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log("→ TF32 enabled (fast matmul)")
    except Exception: pass

log(f"Loading: {{os.path.basename(pth)}}")
pth_lower = pth.lower()
if pth_lower.endswith(".safetensors"):
    try:
        from safetensors.torch import load_file as _stload
        state = _stload(pth, device="cpu")
    except ImportError:
        log("❌ safetensors not installed (pip install safetensors)"); sys.exit(1)
else:
    state = torch.load(pth, map_location="cpu", weights_only=False)
    for key in ("params_ema","params_g","params","model","state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]; break

if not isinstance(state, dict):
    log("❌ Unrecognized format — state_dict not found."); sys.exit(1)

base = os.path.splitext(os.path.basename(pth))[0]

if do_pth:
    log("→ Saving PTH (weights only)...")
    torch.save(state, os.path.join(out_dir, f"{{base}}_clean.pth"))
    log(f"  ✅ {{os.path.join(out_dir, base+'_clean.pth')}}")

if do_fp16:
    log("→ FP16 conversion...")
    fp16 = {{k: v.half() if v.is_floating_point() else v for k, v in state.items()}}
    torch.save(fp16, os.path.join(out_dir, f"{{base}}_fp16.pth"))
    log(f"  ✅ {{os.path.join(out_dir, base+'_fp16.pth')}}")

if do_bf16:
    log("→ BF16 conversion...")
    try:
        bf16 = {{k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()}}
        torch.save(bf16, os.path.join(out_dir, f"{{base}}_bf16.pth"))
        log(f"  ✅ {{os.path.join(out_dir, base+'_bf16.pth')}}")
    except Exception as e: log(f"  ❌ BF16: {{e}}")

if do_safe:
    log("→ SafeTensors conversion...")
    try:
        from safetensors.torch import save_file
        safe = {{k: v.contiguous().float() if v.is_floating_point() else v for k, v in state.items()}}
        save_file(safe, os.path.join(out_dir, f"{{base}}.safetensors"))
        log(f"  ✅ {{os.path.join(out_dir, base+'.safetensors')}}")
    except ImportError: log("  ❌ safetensors not installed (pip install safetensors)")

if do_onnx or do_ncnn or do_trt:
    log(f"→ ONNX export via neosr.utils.convert (opset {{opset}})...")
    try:
        import subprocess as _sp
        r = _sp.run([sys.executable, "-m", "neosr.utils.convert", "--input", pth,
                     "--onnx", "--opset", str(opset)],
                    capture_output=True, text=True, timeout=120)
        log("  ✅ ONNX exported" if r.returncode == 0 else f"  ⚠️ {{r.stderr.strip()[:300]}}")
    except Exception as e: log(f"  ❌ {{e}}")
    if do_ncnn: log("→ For NCNN: convert .onnx with 'onnx2ncnn' (separate tool)")
    if do_trt:  log("→ For TensorRT: use 'trtexec --onnx=model.onnx --saveEngine=model.trt'")

log("✅ Conversion complete.")
"""
                    with _tmp.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                                 encoding='utf-8') as _f:
                        _f.write(script)
                        _tmp_path = _f.name
                    try:
                        flags = 0x08000000 if sys.platform == "win32" else 0
                        proc = subprocess.Popen(
                            [_torch_py, _tmp_path],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, creationflags=flags
                        )
                        for line in proc.stdout:
                            log(line.rstrip())
                        proc.wait()
                    finally:
                        try: os.remove(_tmp_path)
                        except Exception: pass
                    return

                import torch
                if do_tf32:
                    try:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        log(_t("→ TF32 activé (matmul rapide)", "→ TF32 enabled (fast matmul)"))
                    except Exception:
                        pass

                log(f"{_t('Chargement', 'Loading')} : {os.path.basename(pth)}")

                # Smart loader — support both .pth and .safetensors
                pth_lower = pth.lower()
                if pth_lower.endswith(".safetensors"):
                    try:
                        from safetensors.torch import load_file as _stload
                        state = _stload(pth, device="cpu")
                    except ImportError:
                        log(f"  ❌ {_t('safetensors non installé (pip install safetensors)', 'safetensors not installed (pip install safetensors)')}")
                        return
                else:
                    state = torch.load(pth, map_location="cpu", weights_only=False)
                    # Extract weights from checkpoint dict
                    for key in ("params_ema", "params_g", "params", "model", "state_dict"):
                        if isinstance(state, dict) and key in state:
                            state = state[key]
                            break

                if not isinstance(state, dict):
                    log(f"  ❌ {_t('Format non reconnu — state_dict introuvable.', 'Unrecognized format — state_dict not found.')}")
                    return

                base = os.path.splitext(os.path.basename(pth))[0]

                if do_pth:
                    log(f"→ {_t('Sauvegarde PTH (poids seuls)...', 'Saving PTH (weights only)...')}")
                    pth_path = os.path.join(out_dir, f"{base}_clean.pth")
                    torch.save(state, pth_path)
                    log(f"  ✅ {pth_path}")

                if do_fp16:
                    log(f"→ {_t('Conversion FP16...', 'FP16 conversion...')}")
                    fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
                    fp16_path = os.path.join(out_dir, f"{base}_fp16.pth")
                    torch.save(fp16_state, fp16_path)
                    log(f"  ✅ {fp16_path}")

                if do_bf16:
                    log(f"→ {_t('Conversion BF16...', 'BF16 conversion...')}")
                    try:
                        bf16_state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                                      for k, v in state.items()}
                        bf16_path = os.path.join(out_dir, f"{base}_bf16.pth")
                        torch.save(bf16_state, bf16_path)
                        log(f"  ✅ {bf16_path}")
                    except Exception as e:
                        log(f"  ❌ BF16: {e}")

                if do_safe:
                    log(f"→ {_t('Conversion SafeTensors...', 'SafeTensors conversion...')}")
                    try:
                        from safetensors.torch import save_file
                        safe_path = os.path.join(out_dir, f"{base}.safetensors")
                        # safetensors requires contiguous float tensors
                        safe_state = {k: v.contiguous().float() if v.is_floating_point() else v
                                      for k, v in state.items()}
                        save_file(safe_state, safe_path)
                        log(f"  ✅ {safe_path}")
                    except ImportError:
                        log(f"  ❌ {_t('safetensors non installé (pip install safetensors)', 'safetensors not installed (pip install safetensors)')}")

                if do_onnx or do_ncnn or do_trt:
                    py_path = self.settings.get("python_path", "python")
                    cmd = [py_path, "-m", "neosr.utils.convert", "--input", pth,
                           "--onnx", "--opset", str(opset)]
                    log(f"→ {_t('Export ONNX via neosr.utils.convert (opset', 'ONNX export via neosr.utils.convert (opset')} {opset})...")
                    try:
                        creationflags = 0x08000000 if sys.platform == "win32" else 0
                        result = subprocess.run(cmd, capture_output=True, text=True,
                                                timeout=120, creationflags=creationflags)
                        if result.returncode == 0:
                            log(f"  ✅ {_t('ONNX exporté', 'ONNX exported')}")
                        else:
                            log(f"  ⚠️ {result.stderr.strip()[:300]}")
                    except Exception as e:
                        log(f"  ❌ {e}")

                    if do_ncnn:
                        log(_t("→ Pour NCNN : convertissez le .onnx avec 'onnx2ncnn' (outil séparé)", "→ For NCNN: convert .onnx with 'onnx2ncnn' (separate tool)"))
                        log("  → https://github.com/Tencent/ncnn/wiki/how-to-build")

                    if do_trt:
                        log(_t("→ Pour TensorRT : utilisez 'trtexec --onnx=model.onnx --saveEngine=model.trt'", "→ For TensorRT: use 'trtexec --onnx=model.onnx --saveEngine=model.trt'"))
                        log(f"  → {_t('Nécessite NVIDIA TensorRT SDK installé', 'Requires NVIDIA TensorRT SDK installed')}")

                log(f"✅ {_t('Conversion terminée.', 'Conversion complete.')}")
            except Exception as e:
                log(f"❌ {_t('Erreur', 'Error')} : {e}")

        threading.Thread(target=worker, daemon=True).start()

    # ==========================================
    # PAGE 5: LMDB MAKER
    # ==========================================
    def create_page_lmdb(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, _t("Créateur LMDB", "LMDB Creator"), _t("Optimiser les datasets pour la vitesse de lecture.", "Optimize datasets for read speed."))
        self.add_path_row(f, _t("Source Images :", "Source Images:"), "lmdb_src")
        self.add_path_row(f, _t("Sortie (.lmdb) :", "Output (.lmdb):"), "lmdb_dst")

        btns = ctk.CTkFrame(f, fg_color="transparent")
        btns.pack(fill="x", pady=20)
        self.btn_lmdb = ctk.CTkButton(btns, text="START", fg_color="#2ecc71", command=self.run_lmdb)
        self.btn_lmdb.pack(side="left", fill="x", expand=True, padx=5)
        self.btn_lmdb_stop = ctk.CTkButton(btns, text="STOP", fg_color="#e74c3c", command=self.stop_lmdb, state="disabled")
        self.btn_lmdb_stop.pack(side="left", padx=5)

        self.widgets["prog_lmdb"] = ctk.CTkProgressBar(f)
        self.widgets["prog_lmdb"].pack(fill="x")
        self.widgets["prog_lmdb"].set(0)
        self.widgets["lbl_lmdb"] = ctk.CTkLabel(f, text=_t("En attente...", "Waiting..."))
        self.widgets["lbl_lmdb"].pack()
        return f

    def run_lmdb(self):
        src = self.widgets["lmdb_src"].get()
        dst = self.widgets["lmdb_dst"].get()
        if not src or not dst:
            messagebox.showerror(_t("Erreur", "Error"), _t("Chemins requis.", "Paths required."))
            return
        if not dst.endswith(".lmdb"):
            dst += ".lmdb"

        self.btn_lmdb.configure(state="disabled")
        self.btn_lmdb_stop.configure(state="normal")
        # LMDB script needs cv2 + lmdb → use an engine venv python (frozen exe has neither
        # in a usable form). Fall back to settings/PATH only if no venv found.
        py_path = _find_torch_python() or self.settings.get("python_path", "") or "python"
        if py_path == sys.executable:
            # Frozen exe itself — won't have cv2/lmdb importable; prefer a venv
            from src.core import engine_paths as _ep
            _vp = _ep.any_engine_python()
            if _vp:
                py_path = _vp

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
                        self._ui_update(messagebox.showinfo, _t("Succès", "Success"), _t("LMDB Créé !", "LMDB Created!"))
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
        self.add_header(f, _t("Métriques", "Metrics"), _t("Calcul PSNR/SSIM entre deux dossiers.", "Compute PSNR/SSIM between two folders."))
        self.add_path_row(f, _t("Référence (GT) :", "Reference (GT):"), "met_ref")
        self.add_path_row(f, _t("Distorsion (Sortie) :", "Distortion (Output):"), "met_dist")
        ctk.CTkButton(f, text=_t("Calculer", "Compute"), fg_color="#9b59b6", command=self.run_met).pack(fill="x", pady=20)
        self.widgets["lbl_met"] = ctk.CTkLabel(f, text=_t("Résultat : --", "Result: --"), font=("Consolas", 12))
        self.widgets["lbl_met"].pack()
        return f

    def run_met(self):
        ref = self.widgets["met_ref"].get()
        dist = self.widgets["met_dist"].get()
        if not ref or not dist:
            return
        self.widgets["lbl_met"].configure(text=_t("Calcul en cours...", "Computing..."))

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
                result = f"{_t('PSNR Moyen', 'Average PSNR')} : {np.mean(psnrs):.2f} dB ({len(psnrs)} {_t('images', 'images')})" if psnrs else _t("Aucune image commune trouvée", "No common images found")
                self._ui_update(self.widgets["lbl_met"].configure, text=result)
            except Exception as e:
                self._ui_update(self.widgets["lbl_met"].configure, text=f"{_t('Erreur', 'Error')}: {e}")

        threading.Thread(target=calc, daemon=True).start()

    # ==========================================
    # PAGE 7: CHECKER
    # ==========================================
    def create_page_checker(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, _t("Vérificateur Dataset", "Dataset Checker"), _t("Trouver les images corrompues.", "Find corrupted images."))
        self.add_path_row(f, _t("Dossier :", "Folder:"), "chk_src")
        ctk.CTkButton(f, text=_t("Scanner", "Scan"), fg_color="#f39c12", command=self.run_chk).pack(fill="x", pady=10)
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
            self._ui_update(self.widgets["log_chk"].insert, "end", f"\n{_t('Terminé', 'Done')}. {_t('Total', 'Total')}: {cnt}, {_t('Corrompus', 'Corrupted')}: {bad}\n")

        threading.Thread(target=worker, daemon=True).start()

    # ==========================================
    # PAGE 8: HISTORIQUE TRAININGS
    # ==========================================
    def create_page_history(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, _t("Historique des Entrainements", "Training History"),
                        _t("Tous les trainings lances depuis cette app, avec leurs metriques.", "All trainings launched from this app, with their metrics."))

        # Control buttons
        ctrl = ctk.CTkFrame(f, fg_color="transparent")
        ctrl.pack(fill="x", pady=5)
        ctk.CTkButton(ctrl, text=_t("🔄 Rafraichir", "🔄 Refresh"), fg_color="#3498db", width=120,
                      command=self._refresh_history).pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text=_t("📊 Stats par Architecture", "📊 Stats by Architecture"), fg_color="#9b59b6", width=200,
                      command=self._show_arch_stats).pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text=_t("💾 Exporter Benchmark TXT", "💾 Export Benchmark TXT"), fg_color="#27ae60", width=200,
                      command=self._export_history_txt).pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text=_t("🗑 Tout Supprimer", "🗑 Delete All"), fg_color="#c0392b", width=160,
                      command=self._delete_all_history).pack(side="right", padx=5)

        # Scrollable list
        self.widgets["history_list"] = ctk.CTkScrollableFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), height=400)
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
                         text=_t("Aucun training enregistre.\nLes trainings sont automatiquement\nenregistres lorsque vous lancez l'entrainement.",
                                 "No training recorded.\nTrainings are automatically\nrecorded when you start training."),
                         text_color="#666", font=("Roboto", 12), justify="center").pack(pady=40)
            return

        # Header
        hdr = ctk.CTkFrame(self.widgets["history_list"], fg_color=("#D8D8D8", "#2B2B4B"), corner_radius=4)
        hdr.pack(fill="x", pady=(0, 3))
        for txt, w in [(_t("Nom", "Name"), 190), ("Arch", 90), ("Iter", 80), ("PSNR", 70),
                       (_t("Vitesse", "Speed"), 80), (_t("Duree", "Duration"), 80), (_t("Date", "Date"), 120), ("Status", 80), ("", 30)]:
            ctk.CTkLabel(hdr, text=txt, font=("Roboto", 9, "bold"),
                         text_color=("gray30", "#AAA"), width=w, anchor="w").pack(side="left", padx=4)

        for t in trainings:
            row = ctk.CTkFrame(self.widgets["history_list"], fg_color=("#E0E0E0", "#2B2B3B"), corner_radius=4)
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
                         width=80, text_color=("gray30", "#AAA"), anchor="w").pack(side="left", padx=4)
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
        if messagebox.askyesno(_t("Supprimer", "Delete"), _t(f"Supprimer l'entrainement #{row_id} ?", f"Delete training #{row_id}?"),
                               icon="warning"):
            delete_training(row_id)
            self._refresh_history()

    def _delete_all_history(self):
        """Delete all history entries with confirmation."""
        from tkinter import messagebox
        from src.core.training_history import delete_all_trainings
        if messagebox.askyesno(_t("Tout Supprimer", "Delete All"),
                               _t("Supprimer TOUT l'historique des entrainements ?\nCette action est irreversible.",
                                  "Delete ALL training history?\nThis action is irreversible."),
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
            title=_t("Exporter Benchmark", "Export Benchmark"),
            defaultextension=".txt",
            filetypes=[(_t("Fichier texte", "Text file"), "*.txt"), (_t("Tous", "All"), "*.*")],
            initialfile=default_name,
        )
        if not path:
            return
        try:
            content = export_benchmark_txt()
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            from tkinter import messagebox
            messagebox.showinfo(_t("Export OK", "Export OK"), f"{_t('Benchmark exporté', 'Benchmark exported')} :\n{path}")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror(_t("Erreur Export", "Export Error"), str(e))

    def _show_arch_stats(self):
        from src.core.training_history import get_stats_by_architecture, format_duration

        win = ctk.CTkToplevel(self)
        win.title(_t("Stats par Architecture", "Stats by Architecture"))
        win.geometry("700x400")
        ctk.CTkLabel(win, text=_t("Performance par Architecture", "Performance by Architecture"),
                     font=("Roboto", 16, "bold")).pack(pady=10)

        stats = get_stats_by_architecture()
        if not stats:
            ctk.CTkLabel(win, text=_t("Pas assez de donnees.", "Not enough data."), text_color="#888").pack(pady=40)
            return

        for s in stats:
            row = ctk.CTkFrame(win, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=6)
            row.pack(fill="x", padx=20, pady=3)
            ctk.CTkLabel(row, text=s["architecture"], font=("Roboto", 12, "bold"),
                         text_color="#3498db", width=150, anchor="w").pack(side="left", padx=10, pady=5)
            ctk.CTkLabel(row, text=f"{_t('Trainings', 'Trainings')}: {s['count']}", width=100,
                         text_color=("gray30", "#AAA")).pack(side="left", padx=5)
            ctk.CTkLabel(row, text=f"{_t('PSNR moy', 'Avg PSNR')}: {s['avg_psnr']:.2f}", width=130,
                         text_color="#2ecc71").pack(side="left", padx=5)
            ctk.CTkLabel(row, text=f"{_t('PSNR max', 'Max PSNR')}: {s['max_psnr']:.2f}", width=130,
                         text_color="#27ae60").pack(side="left", padx=5)
            ctk.CTkLabel(row, text=f"{_t('Duree moy', 'Avg duration')}: {format_duration(int(s['avg_duration']))}",
                         text_color="#888").pack(side="left", padx=5)

    # ==========================================
    # PAGE 9: RESUME FAILED TRAININGS
    # ==========================================
    def create_page_resume(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, _t("Trainings Interrompus", "Interrupted Trainings"),
                        _t("Detecte les trainings qui se sont arretes avant la fin et propose de les reprendre.", "Detects trainings that stopped before completion and offers to resume them."))

        ctk.CTkButton(f, text=_t("🔍 Scanner experiments/", "🔍 Scan experiments/"), fg_color="#e67e22", width=200,
                      command=self._scan_resume).pack(pady=10)

        self.widgets["resume_list"] = ctk.CTkScrollableFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), height=400)
        self.widgets["resume_list"].pack(fill="both", expand=True, pady=10)

        ctk.CTkLabel(self.widgets["resume_list"],
                     text=_t("Cliquez sur 'Scanner' pour rechercher les trainings interrompus.",
                             "Click 'Scan' to search for interrupted trainings."),
                     text_color="#888").pack(pady=40)
        return f

    def _scan_resume(self):
        from src.core.resume_failed import scan_interrupted_trainings, find_associated_config

        for w in self.widgets["resume_list"].winfo_children():
            w.destroy()

        ctk.CTkLabel(self.widgets["resume_list"], text=_t("Scan en cours...", "Scanning..."),
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
                         text=_t("✅ Aucun training interrompu trouve.\n\nTous vos trainings se sont termines correctement.",
                                 "✅ No interrupted training found.\n\nAll your trainings completed successfully."),
                         text_color="#2ecc71", font=("Roboto", 12)).pack(pady=40)
            return

        ctk.CTkLabel(self.widgets["resume_list"],
                     text=f"⚠ {len(interrupted)} {_t('training(s) interrompu(s) detecte(s) :', 'interrupted training(s) detected:')}",
                     text_color="#f39c12", font=("Roboto", 12, "bold")).pack(anchor="w", padx=10, pady=10)

        for item in interrupted:
            cfg = find_associated_config(item["path"])
            row = ctk.CTkFrame(self.widgets["resume_list"], fg_color=("#E0E0E0", "#2B2B3B"), corner_radius=6)
            row.pack(fill="x", padx=5, pady=3)

            top = ctk.CTkFrame(row, fg_color="transparent")
            top.pack(fill="x", padx=10, pady=(8, 2))
            ctk.CTkLabel(top, text=item["name"], font=("Roboto", 12, "bold"),
                         text_color="#3498db").pack(side="left")
            ctk.CTkLabel(top, text=f" [{item['engine']}]", text_color="#666").pack(side="left")
            ctk.CTkLabel(top, text=f"{_t('Modifié', 'Modified')} : {item['mtime_str']}",
                         text_color="#888", font=("Roboto", 9)).pack(side="right")

            mid = ctk.CTkFrame(row, fg_color="transparent")
            mid.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(mid, text=f"{_t('Iter atteinte', 'Iter reached')} : {item['last_iter']}",
                         text_color=("gray30", "#AAA")).pack(side="left")
            if item["state_file"]:
                ctk.CTkLabel(mid, text=f"  ✓ {_t('State file disponible', 'State file available')}",
                             text_color="#2ecc71").pack(side="left", padx=10)
            else:
                ctk.CTkLabel(mid, text=f"  ⚠ {_t('Pas de .state', 'No .state file')}",
                             text_color="#e67e22").pack(side="left", padx=10)

            btns = ctk.CTkFrame(row, fg_color="transparent")
            btns.pack(fill="x", padx=10, pady=(2, 8))
            if cfg:
                ctk.CTkLabel(btns, text=f"{_t('Config', 'Config')} : {os.path.basename(cfg)}",
                             text_color="#888", font=("Roboto", 9)).pack(side="left")
                ctk.CTkButton(btns, text=_t("▶ Reprendre", "▶ Resume"), fg_color="#27ae60", width=120,
                              command=lambda c=cfg: self._resume_training(c)).pack(side="right", padx=2)
            else:
                ctk.CTkLabel(btns, text=_t("(Config non trouvee)", "(Config not found)"),
                             text_color="#e74c3c", font=("Roboto", 9)).pack(side="left")
            ctk.CTkButton(btns, text=_t("📁 Ouvrir dossier", "📁 Open folder"), fg_color="#666", width=120,
                          command=lambda p=item["path"]: self._open_folder(p)).pack(side="right", padx=2)

    def _resume_training(self, config_path):
        from tkinter import messagebox
        if messagebox.askyesno(_t("Reprendre", "Resume"), _t(f"Lancer l'entrainement avec la config :\n{os.path.basename(config_path)} ?", f"Start training with config:\n{os.path.basename(config_path)}?")):
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
                messagebox.showerror(_t("Erreur", "Error"), f"{_t('Impossible de lancer', 'Cannot start')} : {e}")

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

        ctk.CTkLabel(ctrl, text=_t("Image source :", "Source image:")).pack(side="left", padx=5)
        self.widgets["otf_src"] = ctk.CTkEntry(ctrl, width=400)
        self.widgets["otf_src"].pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text="...", width=30,
                      command=lambda: self._browse_file(self.widgets["otf_src"])).pack(side="left", padx=2)

        ctk.CTkButton(ctrl, text=_t("Charger config TOML/YML", "Load TOML/YML config"), fg_color="#3498db",
                      command=self._otf_load_config).pack(side="left", padx=10)

        ctrl2 = ctk.CTkFrame(f, fg_color="transparent")
        ctrl2.pack(fill="x", pady=5)
        ctk.CTkLabel(ctrl2, text="Scale :").pack(side="left", padx=5)
        self.widgets["otf_scale"] = ctk.CTkOptionMenu(ctrl2, values=["1", "2", "3", "4", "8"], width=60)
        self.widgets["otf_scale"].pack(side="left", padx=5)
        self.widgets["otf_scale"].set("4")

        ctk.CTkLabel(ctrl2, text=_t("Echantillons :", "Samples:")).pack(side="left", padx=15)
        self.widgets["otf_n_samples"] = ctk.CTkOptionMenu(ctrl2, values=["1", "3", "4", "6"], width=60)
        self.widgets["otf_n_samples"].pack(side="left", padx=5)
        self.widgets["otf_n_samples"].set("4")

        ctk.CTkButton(ctrl2, text=_t("🎲 Generer Apercu", "🎲 Generate Preview"), fg_color="#9b59b6", width=200,
                      command=self._otf_generate_preview).pack(side="left", padx=15)

        self.widgets["otf_preview_area"] = ctk.CTkScrollableFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), height=500)
        self.widgets["otf_preview_area"].pack(fill="both", expand=True, pady=10)

        self._otf_config = {}
        ctk.CTkLabel(self.widgets["otf_preview_area"],
                     text=_t("Charger une config + image, puis cliquer 'Generer Apercu'.",
                             "Load a config + image, then click 'Generate Preview'."),
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
            messagebox.showinfo("Config", f"{_t('Config chargée', 'Config loaded')} :\n{os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror(_t("Erreur", "Error"), str(e))

    def _otf_generate_preview(self):
        from tkinter import messagebox
        from PIL import ImageTk
        from src.core.otf_preview import generate_preview_samples

        src = self.widgets["otf_src"].get()
        if not src or not os.path.exists(src):
            messagebox.showerror(_t("Erreur", "Error"), _t("Image source non trouvee.", "Source image not found."))
            return
        if not self._otf_config:
            messagebox.showerror(_t("Erreur", "Error"), _t("Chargez d'abord une config TOML/YML.", "Load a TOML/YML config first."))
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
            ctk.CTkLabel(self.widgets["otf_preview_area"], text=f"{_t('Erreur', 'Error')} : {e}",
                         text_color="#e74c3c").pack(pady=20)
            return

        if not samples:
            ctk.CTkLabel(self.widgets["otf_preview_area"],
                         text=_t("Aucun echantillon genere.", "No sample generated."), text_color="#888").pack(pady=20)
            return

        self._otf_photo_refs = []  # Keep refs to avoid GC
        from PIL import Image

        for i, s in enumerate(samples):
            row = ctk.CTkFrame(self.widgets["otf_preview_area"], fg_color=("#E0E0E0", "#2B2B3B"), corner_radius=6)
            row.pack(fill="x", padx=5, pady=5)
            ctk.CTkLabel(row, text=f"{_t('Echantillon', 'Sample')} {i+1}", font=("Roboto", 11, "bold"),
                         text_color="#9b59b6").pack(anchor="w", padx=10, pady=(5, 0))
            log_text = " | ".join(s["log"]) if s["log"] else _t("(aucune degradation appliquee)", "(no degradation applied)")
            ctk.CTkLabel(row, text=log_text, text_color=("gray30", "#AAA"),
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
        ctk.CTkLabel(ctrl, text=_t("Fichier YAML :", "YAML file:")).pack(side="left", padx=5)
        self.widgets["import_src"] = ctk.CTkEntry(ctrl, width=500)
        self.widgets["import_src"].pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text="...", width=30,
                      command=lambda: self._browse_file(self.widgets["import_src"])).pack(side="left", padx=2)
        ctk.CTkButton(ctrl, text=_t("🔍 Analyser", "🔍 Analyze"), fg_color="#3498db", width=120,
                      command=self._import_analyze).pack(side="left", padx=10)

        self.widgets["import_summary"] = ctk.CTkTextbox(f, height=300, font=("Consolas", 11))
        self.widgets["import_summary"].pack(fill="both", expand=True, pady=10)
        self.widgets["import_summary"].insert("1.0",
            _t("Selectionnez un fichier .yml/.yaml de config Real-ESRGAN, SwinIR, RCAN, HAT...\n"
               "Cliquez 'Analyser' pour voir le resume.\n\n"
               "Sources de configs :\n"
               "  - https://github.com/xinntao/Real-ESRGAN/tree/master/options\n"
               "  - https://github.com/JingyunLiang/SwinIR/tree/main/options\n"
               "  - https://github.com/XPixelGroup/BasicSR/tree/master/options\n",
               "Select a .yml/.yaml config file from Real-ESRGAN, SwinIR, RCAN, HAT...\n"
               "Click 'Analyze' to see the summary.\n\n"
               "Config sources:\n"
               "  - https://github.com/xinntao/Real-ESRGAN/tree/master/options\n"
               "  - https://github.com/JingyunLiang/SwinIR/tree/main/options\n"
               "  - https://github.com/XPixelGroup/BasicSR/tree/master/options\n"))
        self.widgets["import_summary"].configure(state="disabled")

        return f

    def _import_analyze(self):
        from src.core.config_importer import get_import_summary, import_yaml_config
        from tkinter import messagebox

        path = self.widgets["import_src"].get()
        if not path or not os.path.exists(path):
            messagebox.showerror(_t("Erreur", "Error"), _t("Fichier non trouve.", "File not found."))
            return

        summary = get_import_summary(path)
        self.widgets["import_summary"].configure(state="normal")
        self.widgets["import_summary"].delete("1.0", "end")
        self.widgets["import_summary"].insert("1.0",
            _t("Résumé de l'import", "Import summary") + f" :\n\n{summary}\n\n" +
            "─" * 50 + "\n\n" +
            _t("Vous pouvez ensuite copier les valeurs vers la Configuration manuellement,\n"
               "ou les utiliser comme reference pour vos tests.",
               "You can then copy the values to the Configuration manually,\n"
               "or use them as a reference for your tests."))
        self.widgets["import_summary"].configure(state="disabled")

    # ==========================================
    # PAGE 12: INFO MODELE
    # ==========================================
    def create_page_model_info(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, _t("Inspection Modele", "Model Inspection"),
                        _t("Inspecte un fichier .pth/.onnx/.safetensors pour identifier l'architecture, le scale, les modules.",
                           "Inspect a .pth/.onnx/.safetensors file to identify architecture, scale, and modules."))

        ctrl = ctk.CTkFrame(f, fg_color="transparent")
        ctrl.pack(fill="x", pady=5)
        ctk.CTkLabel(ctrl, text=_t("Modele :", "Model:"), width=70, anchor="w").pack(side="left", padx=5)
        self.widgets["mi_path"] = ctk.CTkEntry(ctrl, width=600)
        self.widgets["mi_path"].pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text="...", width=30,
                      command=lambda: self._browse_file(self.widgets["mi_path"])).pack(side="left", padx=2)
        ctk.CTkButton(ctrl, text=_t("🔍 Inspecter", "🔍 Inspect"), fg_color="#3498db", width=120,
                      command=self._inspect_model).pack(side="left", padx=10)

        # Convert to safetensors button
        conv = ctk.CTkFrame(f, fg_color="transparent")
        conv.pack(fill="x", pady=5)
        ctk.CTkButton(conv, text=_t("💾 Convertir vers .safetensors", "💾 Convert to .safetensors"),
                      fg_color="#16a085", command=self._convert_safetensors).pack(side="left", padx=5)
        ToolTip(conv, _t("Convertit un .pth en .safetensors (plus securise, charge plus vite)", "Converts a .pth to .safetensors (more secure, faster loading)"))

        self.widgets["mi_output"] = ctk.CTkTextbox(f, height=480, font=("Consolas", 11),
                                                     fg_color=("#F5F5F5", "#0d0d1a"), text_color=("gray10", "#dcdcdc"))
        self.widgets["mi_output"].pack(fill="both", expand=True, pady=10)
        self.widgets["mi_output"].insert(
            "1.0",
            _t("Sélectionnez un fichier modèle (.pth, .pt, .onnx, .safetensors)\n"
               "puis cliquez sur 'Inspecter' pour voir :\n"
               "  - Architecture détectée (RRDBNet, SwinIR, RCAN, OmniSR, HAT, SPAN, ...)\n"
               "  - Nombre de paramètres et estimation VRAM\n"
               "  - Top modules par taille\n"
               "  - Précision (FP32/FP16/etc)\n"
               "  - Scale inféré (x2, x3, x4...)\n"
               "  - Métadonnées du checkpoint\n",
               "Select a model file (.pth, .pt, .onnx, .safetensors)\n"
               "then click 'Inspect' to see:\n"
               "  - Detected architecture (RRDBNet, SwinIR, RCAN, OmniSR, HAT, SPAN, ...)\n"
               "  - Parameter count and VRAM estimate\n"
               "  - Top modules by size\n"
               "  - Precision (FP32/FP16/etc)\n"
               "  - Inferred scale (x2, x3, x4...)\n"
               "  - Checkpoint metadata\n")
        )
        return f

    def _inspect_model(self):
        path = self.widgets["mi_path"].get()
        if not path or not os.path.exists(path):
            from tkinter import messagebox
            messagebox.showerror(_t("Erreur", "Error"), _t("Fichier non trouve.", "File not found."))
            return
        # Frozen exe lacks torch/safetensors → run model_export CLI in an engine venv
        if getattr(sys, "frozen", False) and not path.lower().endswith(".onnx"):
            py = _find_torch_python()
            _me = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "model_export.py")
            if py and py != sys.executable and os.path.isfile(_me):
                try:
                    r = subprocess.run([py, _me, path], capture_output=True, text=True,
                                       timeout=60, creationflags=0x08000000 if sys.platform == "win32" else 0)
                    text = r.stdout.strip() or (r.stderr.strip()[:500] if r.stderr else "Aucune sortie.")
                    self.widgets["mi_output"].delete("1.0", "end")
                    self.widgets["mi_output"].insert("1.0", text)
                    return
                except Exception as e:
                    self.widgets["mi_output"].delete("1.0", "end")
                    self.widgets["mi_output"].insert("1.0", f"Erreur inspection venv : {e}")
                    return
        from src.core.model_export import detect_model_format, format_model_info
        info = detect_model_format(path)
        text = format_model_info(info)
        self.widgets["mi_output"].delete("1.0", "end")
        self.widgets["mi_output"].insert("1.0", text)

    def _convert_safetensors(self):
        from src.core.model_export import convert_pth_to_safetensors
        from tkinter import messagebox
        path = self.widgets["mi_path"].get()
        if not path or not os.path.exists(path):
            messagebox.showerror(_t("Erreur", "Error"), _t("Selectionnez d'abord un .pth.", "Select a .pth file first."))
            return
        if not path.endswith(".pth") and not path.endswith(".pt"):
            messagebox.showwarning(_t("Attention", "Warning"), _t("Format source recommande : .pth ou .pt", "Recommended source format: .pth or .pt"))
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
            messagebox.showinfo(_t("Conversion", "Conversion"), msg)
        else:
            messagebox.showerror(_t("Erreur", "Error"), msg)

    # ==========================================
    # PAGE 13: GALERIE WEB + PATCH TENSORBOARD
    # ==========================================
    def create_page_gallery(self):
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.add_header(f, "Galerie Validation & Patch TensorBoard",
                        "Visualisez les images de validation a distance OU activez l'affichage dans TensorBoard.")

        # ─── Section A: HTTP Gallery Server ───
        section_a = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8)
        section_a.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(section_a, text=_t("A. Serveur Galerie Web (sans toucher a NeoSR)", "A. Web Gallery Server (without touching NeoSR)"),
                     font=("Roboto", 13, "bold"), text_color="#3498db"
                     ).pack(anchor="w", padx=10, pady=(8, 5))
        ctk.CTkLabel(section_a,
                     text=_t("Lance un mini-serveur HTTP pointe sur un dossier d'images.\n"
                             "Compatible mobile, auto-refresh, zoom click. Optionnel : tunnel Ngrok pour acces distant.",
                             "Launches a mini HTTP server pointing to an image folder.\n"
                             "Mobile-compatible, auto-refresh, zoom click. Optional: Ngrok tunnel for remote access."),
                     text_color=("gray30", "#AAA"), font=("Roboto", 10), justify="left"
                     ).pack(anchor="w", padx=10, pady=(0, 10))

        # Directory picker
        dir_row = ctk.CTkFrame(section_a, fg_color="transparent")
        dir_row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(dir_row, text=_t("Dossier :", "Folder:"), width=80, anchor="w").pack(side="left")
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
        ctk.CTkLabel(opt_row, text=_t("Port :", "Port:"), width=80, anchor="w").pack(side="left")
        self.widgets["gal_port"] = ctk.CTkEntry(opt_row, width=80)
        self.widgets["gal_port"].pack(side="left", padx=5)
        self.widgets["gal_port"].insert(0, "8765")
        self.widgets["gal_ngrok"] = ctk.CTkCheckBox(
            opt_row, text=_t("Activer tunnel Ngrok (acces a distance)", "Enable Ngrok tunnel (remote access)")
        )
        self.widgets["gal_ngrok"].pack(side="left", padx=20)
        ToolTip(self.widgets["gal_ngrok"],
                _t("Necessite ngrok installe et authentifie.\n"
                   "Donnera une URL publique https://xxxx.ngrok-free.app\n"
                   "accessible depuis n'importe quel appareil.",
                   "Requires ngrok installed and authenticated.\n"
                   "Will provide a public URL https://xxxx.ngrok-free.app\n"
                   "accessible from any device."))

        # Action buttons
        btn_row = ctk.CTkFrame(section_a, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=8)
        self.widgets["btn_gal_start"] = ctk.CTkButton(
            btn_row, text=_t("▶ Demarrer Serveur", "▶ Start Server"), fg_color="#27ae60",
            width=180, command=self._gallery_start
        )
        self.widgets["btn_gal_start"].pack(side="left", padx=5)
        self.widgets["btn_gal_stop"] = ctk.CTkButton(
            btn_row, text=_t("⏹ Arreter", "⏹ Stop"), fg_color="#e74c3c",
            width=120, command=self._gallery_stop, state="disabled"
        )
        self.widgets["btn_gal_stop"].pack(side="left", padx=5)
        self.widgets["btn_gal_open"] = ctk.CTkButton(
            btn_row, text=_t("🌐 Ouvrir dans Navigateur", "🌐 Open in Browser"), fg_color="#3498db",
            width=200, command=self._gallery_open, state="disabled"
        )
        self.widgets["btn_gal_open"].pack(side="left", padx=5)

        # Status display
        self.widgets["gal_status"] = ctk.CTkLabel(
            section_a, text=_t("Etat : Arrete", "Status: Stopped"),
            text_color="#888", anchor="w", justify="left",
            font=("Consolas", 11)
        )
        self.widgets["gal_status"].pack(anchor="w", fill="x", padx=10, pady=(5, 10))

        # QR code area (only if URL is set)
        self.widgets["gal_qr_frame"] = ctk.CTkFrame(section_a, fg_color="transparent")
        self.widgets["gal_qr_frame"].pack(fill="x", padx=10, pady=5)

        # ─── Section B: NeoSR/Redux TB image patch ───
        section_b = ctk.CTkFrame(f, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8)
        section_b.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(section_b, text=_t("B. Patch NeoSR/Redux pour images TensorBoard", "B. NeoSR/Redux Patch for TensorBoard images"),
                     font=("Roboto", 13, "bold"), text_color="#9b59b6"
                     ).pack(anchor="w", padx=10, pady=(8, 5))
        ctk.CTkLabel(section_b,
                     text=_t("Injecte un appel tb_logger.add_image() apres chaque imwrite() dans nondist_validation.\n"
                             "Maximum 4 images par validation pour eviter de gonfler les .tfevents.\n"
                             "Idempotent (detection via marqueur), reversible (backup .usr_bak cree).",
                             "Injects a tb_logger.add_image() call after each imwrite() in nondist_validation.\n"
                             "Maximum 4 images per validation to avoid bloating .tfevents.\n"
                             "Idempotent (marker-based detection), reversible (backup .usr_bak created)."),
                     text_color=("gray30", "#AAA"), font=("Roboto", 10), justify="left"
                     ).pack(anchor="w", padx=10, pady=(0, 10))

        # Engine selection
        eng_row = ctk.CTkFrame(section_b, fg_color="transparent")
        eng_row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(eng_row, text=_t("Engine :", "Engine:"), width=80, anchor="w").pack(side="left")
        self.widgets["tbp_engine"] = ctk.CTkOptionMenu(
            eng_row, values=["NeoSR", "traiNNer-Redux"], width=200,
            command=lambda x: self._tbp_refresh_status()
        )
        self.widgets["tbp_engine"].pack(side="left", padx=5)
        ctk.CTkButton(eng_row, text=_t("🔄 Verifier statut", "🔄 Check status"), fg_color="#666",
                      width=140, command=self._tbp_refresh_status
                      ).pack(side="left", padx=10)

        # Status
        self.widgets["tbp_status"] = ctk.CTkLabel(
            section_b, text=_t("(non verifie)", "(not checked)"), text_color="#888",
            anchor="w", justify="left", font=("Consolas", 10), wraplength=700
        )
        self.widgets["tbp_status"].pack(anchor="w", fill="x", padx=10, pady=5)

        # Action buttons
        tbp_btn_row = ctk.CTkFrame(section_b, fg_color="transparent")
        tbp_btn_row.pack(fill="x", padx=10, pady=8)
        ctk.CTkButton(tbp_btn_row, text=_t("✅ Appliquer le Patch", "✅ Apply Patch"), fg_color="#27ae60",
                      width=180, command=self._tbp_apply
                      ).pack(side="left", padx=5)
        ctk.CTkButton(tbp_btn_row, text=_t("❌ Retirer le Patch", "❌ Remove Patch"), fg_color="#e74c3c",
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
            messagebox.showerror(_t("Erreur", "Error"), _t("Selectionnez un dossier valide.", "Select a valid folder."))
            return

        try:
            port = int(port_str) if port_str else 0
        except ValueError:
            messagebox.showerror(_t("Erreur", "Error"), _t("Port invalide.", "Invalid port."))
            return

        srv = get_server()
        result = srv.start(directory, port=port, with_ngrok=bool(with_ngrok))

        if not result.get("ok"):
            messagebox.showerror(_t("Erreur", "Error"), result.get("error", _t("Echec inconnu", "Unknown failure")))
            return

        # Update UI
        self.widgets["btn_gal_start"].configure(state="disabled")
        self.widgets["btn_gal_stop"].configure(state="normal")
        self.widgets["btn_gal_open"].configure(state="normal")

        status_lines = [
            f"✅ {_t('Serveur actif', 'Server active')}",
            f"   {_t('Local', 'Local')} : {result['local_url']}",
        ]
        if result.get("ngrok_url"):
            status_lines.append(f"   {_t('Public', 'Public')} : {result['ngrok_url']}")
            self._gallery_show_qr(result["ngrok_url"])
        elif with_ngrok and result.get("ngrok_warning"):
            status_lines.append(f"   ⚠ {result['ngrok_warning']}")
        else:
            self._gallery_show_qr(result["local_url"])

        status_lines.append(f"   {_t('Dossier', 'Folder')} : {directory}")
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
        self.widgets["gal_status"].configure(text=_t("Etat : Arrete", "Status: Stopped"), text_color="#888")

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
                    text=_t(f"💡 Installer 'qrcode' pour scanner avec votre tel : pip install qrcode[pil]\n"
                            f"URL : {url}",
                            f"💡 Install 'qrcode' to scan with your phone: pip install qrcode[pil]\n"
                            f"URL: {url}"),
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
                             text=_t(f"📱 Scannez le QR code avec votre telephone\n\nURL : {url}",
                                     f"📱 Scan the QR code with your phone\n\nURL: {url}"),
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
            text = (_t(f"❌ Engine non trouve : {root}\n"
                       f"   (verifie que le dossier IA_Engine existe avec NeoSR/Redux installe)",
                       f"❌ Engine not found: {root}\n"
                       f"   (check that the IA_Engine folder exists with NeoSR/Redux installed)"))
            color = "#e74c3c"
        elif status["patched"]:
            text = (_t(f"✅ Patch deja applique\n"
                       f"   Fichier : {status['target_file']}\n"
                       f"   Backup .usr_bak : {'present' if status['backup_exists'] else 'absent'}",
                       f"✅ Patch already applied\n"
                       f"   File: {status['target_file']}\n"
                       f"   Backup .usr_bak: {'present' if status['backup_exists'] else 'absent'}"))
            color = "#2ecc71"
        else:
            text = (_t(f"⚪ Pas patche (pret a l'emploi)\n"
                       f"   Fichier cible : {status['target_file']}",
                       f"⚪ Not patched (ready to use)\n"
                       f"   Target file: {status['target_file']}"))
            color = "#f39c12"
        self.widgets["tbp_status"].configure(text=text, text_color=color)

    def _tbp_apply(self):
        from tkinter import messagebox
        from src.core.tb_image_patch import patch_engine
        root = self._get_engine_root()
        if not os.path.isdir(root):
            messagebox.showerror(_t("Erreur", "Error"), _t(f"Engine introuvable : {root}", f"Engine not found: {root}"))
            return
        if not messagebox.askyesno(
                _t("Appliquer Patch", "Apply Patch"),
                _t(f"Modifier les fichiers de {os.path.basename(root)} ?\n\n"
                   f"Un backup .usr_bak sera cree.\n"
                   f"Vous pourrez retirer le patch a tout moment via 'Retirer'.",
                   f"Modify files of {os.path.basename(root)} ?\n\n"
                   f"A .usr_bak backup will be created.\n"
                   f"You can remove the patch at any time via 'Remove'.")):
            return
        ok, msg, path = patch_engine(root)
        if ok:
            messagebox.showinfo("Patch", _t(f"{msg}\n\nFichier modifie :\n{path}", f"{msg}\n\nModified file:\n{path}"))
        else:
            messagebox.showerror(_t("Erreur", "Error"), msg)
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
        self.add_header(f, _t("📦 Publier Modèle", "📦 Publish Model"),
                        _t("Préparer un package propre pour archiver ou partager votre modèle.", "Prepare a clean package to archive or share your model."))

        body = ctk.CTkFrame(f, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # ── LEFT COLUMN : sélecteur de modèle ──────────────────
        left = ctk.CTkFrame(body, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)

        ctk.CTkLabel(left, text=_t("📂 Sélectionner le Modèle", "📂 Select Model"),
                     font=("Roboto", 13, "bold"),
                     text_color="#3498db").pack(anchor="w", padx=10, pady=(8, 4))

        flt_row = ctk.CTkFrame(left, fg_color="transparent")
        flt_row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(flt_row, text=_t("Filtre :", "Filter:"), width=50, anchor="w").pack(side="left")
        self._exp_engine_var = StringVar(value=_t("Tous", "All"))
        self.widgets["exp_engine_filter"] = ctk.CTkOptionMenu(
            flt_row,
            values=[_t("Tous", "All"), _t("5 derniers", "Last 5"), _t("10 derniers", "Last 10"), "NeoSR", "Redux"],
            variable=self._exp_engine_var, width=140,
            command=lambda x: self._export_refresh_list()
        )
        self.widgets["exp_engine_filter"].pack(side="left", padx=5)
        ctk.CTkButton(flt_row, text=_t("🔄 Scanner", "🔄 Scan"), width=100,
                      command=self._export_scan).pack(side="left", padx=3)

        self._exp_list_frame = ctk.CTkScrollableFrame(left, height=260, fg_color=("#EBEBEB", "#111"))
        self._exp_list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        ctk.CTkLabel(self._exp_list_frame,
                     text=_t("(cliquez 🔄 Scanner pour charger)", "(click 🔄 Scan to load)"),
                     text_color="#555").pack(anchor="w")

        man_row = ctk.CTkFrame(left, fg_color="transparent")
        man_row.pack(fill="x", padx=10, pady=(4, 8))
        ctk.CTkLabel(man_row, text=_t("Ou parcourir :", "Or browse:"), width=90, anchor="w").pack(side="left")
        self.widgets["exp_manual_path"] = ctk.CTkEntry(
            man_row, placeholder_text="chemin/vers/modele.safetensors")
        self.widgets["exp_manual_path"].pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(man_row, text="...", width=30,
                      command=self._export_browse_model).pack(side="left")

        # ── RIGHT COLUMN : métadonnées + fiche ─────────────────
        right = ctk.CTkFrame(body, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)

        ctk.CTkLabel(right, text=_t("📝 Métadonnées & Fiche Technique", "📝 Metadata & Technical Sheet"),
                     font=("Roboto", 13, "bold"),
                     text_color="#3498db").pack(anchor="w", padx=10, pady=(8, 4))

        meta_f = ctk.CTkFrame(right, fg_color="transparent")
        meta_f.pack(fill="x", padx=10, pady=3)
        for lbl, key, ph in [
            (_t("Nom du modèle :", "Model name:"), "exp_name",    "ex: Crysisjim SPANPlus Deband_HARD"),
            (_t("Version :",       "Version:"),    "exp_version", "ex: 1.0"),
            (_t("Auteur :",        "Author:"),     "exp_author",  "ex: Crysisjim"),
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

        ctk.CTkLabel(right, text=_t("Notes personnelles :", "Personal notes:"),
                     anchor="w").pack(anchor="w", padx=10, pady=(6, 0))
        self.widgets["exp_notes"] = ctk.CTkTextbox(right, height=60, fg_color=("#EBEBEB", "#111"))
        self.widgets["exp_notes"].pack(fill="x", padx=10, pady=3)

        self.widgets["exp_config_info"] = ctk.CTkLabel(
            right, text=_t("ℹ Aucun modèle sélectionné", "ℹ No model selected"),
            text_color="#666", font=("Roboto", 10),
            justify="left", anchor="w", wraplength=380
        )
        self.widgets["exp_config_info"].pack(anchor="w", padx=10, pady=3)

        ctk.CTkLabel(right, text=_t("📄 Fiche Technique du Modèle :", "📄 Model Technical Sheet:"),
                     anchor="w", font=("Roboto", 11, "bold")).pack(
            anchor="w", padx=10, pady=(6, 0))
        self.widgets["exp_fiche"] = ctk.CTkTextbox(right, height=200, fg_color=("#EBEBEB", "#111"))
        self.widgets["exp_fiche"].pack(fill="x", padx=10, pady=3)

        # ── IA config row ──────────────────────────────────
        ia_row = ctk.CTkFrame(right, fg_color=("#E8E8E8", "#111827"), corner_radius=6)
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
            ia_row, values=first_models, width=180, font=("Roboto", 10))
        self.widgets["exp_ia_model"].pack(side="left", padx=4)
        self.widgets["exp_ia_model"].set(first_models[0])

        fiche_btns = ctk.CTkFrame(right, fg_color="transparent")
        fiche_btns.pack(fill="x", padx=10, pady=3)
        ctk.CTkButton(fiche_btns, text=_t("📋 Générer (template)", "📋 Generate (template)"), width=165,
                      command=self._export_generate_template).pack(side="left", padx=(0, 6))
        ctk.CTkButton(fiche_btns, text=_t("🤖 Générer avec IA", "🤖 Generate with AI"), width=165,
                      fg_color="#8e44ad",
                      command=self._export_generate_ai).pack(side="left", padx=3)

        self.widgets["exp_dest_lbl"] = ctk.CTkLabel(
            right, text=_t("📁 Destination : (sélectionnez un modèle)", "📁 Destination: (select a model)"),
            text_color="#888", font=("Roboto", 10), anchor="w", wraplength=390
        )
        self.widgets["exp_dest_lbl"].pack(anchor="w", padx=10, pady=(4, 0))

        self.widgets["exp_export_btn"] = ctk.CTkButton(
            right, text=_t("📦 Exporter le Package", "📦 Export Package"),
            fg_color="#16a085", height=38,
            font=("Roboto", 13, "bold"),
            command=self._export_do_export
        )
        self.widgets["exp_export_btn"].pack(fill="x", padx=10, pady=(6, 10))

        ToolTip(self.widgets["exp_export_btn"],
                _t("📦 Crée le dossier dans Final model/ avec :\n"
                   "  • {Nom Version}.safetensors  (modèle renommé)\n"
                   "  • Fiche Technique du Modèle.txt\n"
                   "  • Option/  →  config d'entraînement\n"
                   "  • resume/  →  fichier original conservé\n"
                   "  • Trainning state/  →  fichier .state",
                   "📦 Creates the folder in Final model/ with:\n"
                   "  • {Name Version}.safetensors  (renamed model)\n"
                   "  • Fiche Technique du Modèle.txt\n"
                   "  • Option/  →  training config\n"
                   "  • resume/  →  original file kept\n"
                   "  • Trainning state/  →  .state file"))
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
        # Apply filter — check both FR and EN values
        if flt in (_t("5 derniers", "Last 5"), "5 derniers", "Last 5"):
            data = self._exp_scan_data[:5]
        elif flt in (_t("10 derniers", "Last 10"), "10 derniers", "Last 10"):
            data = self._exp_scan_data[:10]
        elif flt in ("NeoSR", "Redux"):
            data = [d for d in self._exp_scan_data if d["engine"] == flt]
        else:
            data = list(self._exp_scan_data)
        if not data:
            ctk.CTkLabel(self._exp_list_frame,
                         text=_t("Aucun modèle trouvé. Cliquez 🔄 Scanner.", "No model found. Click 🔄 Scan."),
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
                                fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=6)
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
                text=f"  Config {cfg_icon}  {_t('État', 'State')} {sta_icon}  — {d['model_file']}",
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
            txt = _t("Config introuvable — utilisez le bouton '...' pour la retrouver",
                     "Config not found — use the '...' button to locate it")
        self.widgets["exp_config_info"].configure(text=txt)
        self._export_update_preview_lbl()

    def _export_browse_model(self):
        """Open file dialog to manually select a .safetensors model."""
        path = filedialog.askopenfilename(
            title=_t("Sélectionner le modèle", "Select model"),
            filetypes=[("SafeTensors", "*.safetensors"), (_t("Tous", "All"), "*.*")]
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
                text=f"📁 {_t('Destination', 'Destination')} : {dest}")
        else:
            self.widgets["exp_dest_lbl"].configure(
                text=_t("📁 Destination : (entrez un nom de modèle)", "📁 Destination: (enter a model name)"))

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
            info["gan"]       = _t("Oui", "Yes") if net_d else _t("Non", "No")
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
            messagebox.showerror(_t("Modèle IA", "AI Model"), _t("Sélectionnez un modèle.", "Select a model."))
            return
        # Resolve API key — same key format as tab_config: api_key_{provider}
        api_key = self.settings.get(f"api_key_{provider}", "")
        if not api_key:
            messagebox.showerror(
                _t("Clé API manquante", "Missing API Key"),
                _t(f"Aucune clé sauvegardée pour :\n{provider}\n\n"
                   "Pour sauvegarder :\n"
                   "1. Onglet Configuration\n"
                   "2. Section 'Vérification par IA'\n"
                   "3. Sélectionnez le fournisseur + entrez la clé\n"
                   "4. Cliquez 'Analyser avec IA' → la clé est sauvegardée.",
                   f"No saved key for:\n{provider}\n\n"
                   "To save:\n"
                   "1. Configuration tab\n"
                   "2. 'AI Verification' section\n"
                   "3. Select provider + enter key\n"
                   "4. Click 'Analyze with AI' → key is saved.")
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
        self.widgets["exp_fiche"].insert("1.0", f"⏳ {_t('Génération IA en cours', 'AI generation in progress')} ({provider})…")

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
                return f"{_t('Erreur HTTP', 'HTTP Error')} {e.code}: {e.read().decode()[:300]}"
            except Exception as ex:
                return f"{_t('Erreur', 'Error')}: {ex}"
        elif "xAI" in provider or "Grok" in provider:
            url = "https://api.x.ai/v1/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
        elif "DeepSeek" in provider:
            url = "https://api.deepseek.com/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            return f"{_t('Fournisseur non supporté', 'Unsupported provider')} : {provider}"
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
            return f"{_t('Erreur HTTP', 'HTTP Error')} {e.code}: {e.read().decode()[:300]}"
        except Exception as ex:
            return f"{_t('Erreur', 'Error')}: {ex}"
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
            messagebox.showerror(_t("Erreur", "Error"), _t("Entrez un nom de modèle.", "Enter a model name."))
            return
        # Resolve model path (manual browse takes priority)
        d          = self._exp_selected or {}
        model_path = self.widgets["exp_manual_path"].get().strip() or d.get("model_path", "")
        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror(_t("Erreur", "Error"),
                                 _t("Aucun modèle sélectionné ou fichier introuvable.", "No model selected or file not found."))
            return
        fiche_text = self.widgets["exp_fiche"].get("1.0", "end").strip()
        if not fiche_text:
            if not messagebox.askyesno(
                    _t("Fiche vide", "Empty sheet"),
                    _t("La fiche technique est vide.\n\nContinuer quand même ?",
                       "The technical sheet is empty.\n\nContinue anyway?")):
                return
        folder_name = f"{name} {version}".strip()
        dest_root = os.path.join(
            os.path.expanduser("~"), "IA_Engine", "Final model", folder_name)
        if os.path.exists(dest_root):
            if not messagebox.askyesno(
                    _t("Dossier existant", "Folder exists"),
                    _t(f"Le dossier existe déjà :\n{dest_root}\n\nÉcraser ?",
                       f"The folder already exists:\n{dest_root}\n\nOverwrite?")):
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
                _t("✅ Export réussi !", "✅ Export successful!"),
                f"{_t('Package créé dans', 'Package created in')} :\n{dest_root}\n\n"
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
            messagebox.showerror(_t("Erreur Export", "Export Error"), f"{_t('Export échoué', 'Export failed')} :\n{ex}")

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
                     text=_t("Mesure les it/s, VRAM et stabilité de chaque architecture ou feature.",
                             "Measures it/s, VRAM and stability for each architecture or feature."),
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
            w.pack(side="left")   # fix: widget was created but never placed
            if tip:
                ToolTip(w, tip)
            return w

        # ── Moteur ──
        r_engine = ctk.CTkFrame(scroll, fg_color="transparent")
        r_engine.pack(fill="x", pady=3)
        ctk.CTkLabel(r_engine, text=_t("Moteur :", "Engine:"), width=170, anchor="w").pack(side="left")
        self.widgets["bench_engine"] = ctk.CTkOptionMenu(
            r_engine, values=["Redux", "NeoSR", _t("Les deux", "Both")],
            width=130, command=self._bench_on_engine_change)
        self.widgets["bench_engine"].set("Redux")
        self.widgets["bench_engine"].pack(side="left")
        ToolTip(self.widgets["bench_engine"],
                _t("Redux  : traiNNer-redux (YAML, plus d'archs/features).\n"
                   "NeoSR  : moteur NeoSR (TOML).\n"
                   "Les deux : lance les deux moteurs successivement.\n\n"
                   "Chaque moteur utilise son propre venv Python.",
                   "Redux  : traiNNer-redux (YAML, more archs/features).\n"
                   "NeoSR  : NeoSR engine (TOML).\n"
                   "Both   : runs both engines sequentially.\n\n"
                   "Each engine uses its own Python venv."))

        # ── Type de benchmark ──
        r_type = ctk.CTkFrame(scroll, fg_color="transparent")
        r_type.pack(fill="x", pady=3)
        ctk.CTkLabel(r_type, text=_t("Type de benchmark :", "Benchmark type:"), width=170, anchor="w").pack(side="left")
        self.widgets["bench_type"] = ctk.CTkOptionMenu(
            r_type, values=[_t("Architectures", "Architectures"), _t("Features", "Features"), _t("Arch + Features", "Arch + Features")],
            width=150, command=self._bench_on_type_change)
        self.widgets["bench_type"].set("Architectures")
        self.widgets["bench_type"].pack(side="left")
        ToolTip(self.widgets["bench_type"],
                _t("Architectures : mesure it/s et VRAM de chaque architecture.\n"
                   "  → Idéal pour choisir la meilleure arch pour votre GPU.\n\n"
                   "Features : teste l'impact de chaque loss/optimiseur/scheduler.\n"
                   "  → Idéal pour optimiser votre config d'entraînement.\n\n"
                   "Arch + Features : lance les deux suites successivement.\n"
                   "  → Durée totale : ~6-10h selon votre GPU.",
                   "Architectures: measures it/s and VRAM per architecture.\n"
                   "  → Ideal to pick the best arch for your GPU.\n\n"
                   "Features: tests the impact of each loss/optimizer/scheduler.\n"
                   "  → Ideal to optimize your training config.\n\n"
                   "Arch + Features: runs both suites sequentially.\n"
                   "  → Total duration: ~6-10h depending on your GPU."))

        # ── Itérations ──
        self.widgets["bench_n_iter"] = _row(
            scroll, _t("Itérations / test :", "Iterations / test:"),
            lambda p: (lambda e: (e.insert(0, "2500"), e)[-1])(ctk.CTkEntry(p, width=90)),
            _t("Nombre d'itérations d'entraînement par test.\n"
               "Arch    → 2500 iters (recommandé, ~4 min/test)\n"
               "Feature → 500 iters  (recommandé, ~1 min/test)\n"
               "Plus bas = rapide mais moins stable.\nPlus haut = précis mais très long.",
               "Number of training iterations per test.\n"
               "Arch    → 2500 iters (recommended, ~4 min/test)\n"
               "Feature → 500 iters  (recommended, ~1 min/test)\n"
               "Lower = faster but less stable.\nHigher = accurate but very long."))

        # ── Timeout ──
        self.widgets["bench_timeout"] = _row(
            scroll, _t("Timeout (s) / test :", "Timeout (s) / test:"),
            lambda p: (lambda e: (e.insert(0, "3600"), e)[-1])(ctk.CTkEntry(p, width=90)),
            _t("Temps maximum autorisé par test avant kill forcé.\n"
               "Arch    → 3600s (1h)\nFeature → 900s  (15min)\n"
               "Si le test dépasse ce seuil, il est marqué 'timeout' et le benchmark continue.",
               "Maximum time allowed per test before forced kill.\n"
               "Arch    → 3600s (1h)\nFeature → 900s  (15min)\n"
               "If the test exceeds this threshold, it is marked 'timeout' and the benchmark continues."))

        # ── Modes précision (arch Redux seulement) ──
        f_modes = ctk.CTkFrame(scroll, fg_color="transparent")
        f_modes.pack(fill="x", pady=3)
        ctk.CTkLabel(f_modes, text=_t("Modes précision :", "Precision modes:"), width=170, anchor="w").pack(side="left")
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
                _t("Modes de précision à tester (Redux arch uniquement).\n\n"
                   "normal : FP32 — baseline, toujours disponible.\n"
                   "fp16   : AMP Float16 — +10-15% sur RTX 2000+.\n"
                   "bf16   : AMP BFloat16 — +25-35% sur RTX 3000+ (Ampere).\n"
                   "tf32   : fast_matmul TF32 — +5-10% sur RTX 3000+.\n\n"
                   "Recommandé GTX 1080 Ti : normal seulement.\n"
                   "Recommandé RTX 3070+ : normal + bf16.",
                   "Precision modes to test (Redux arch only).\n\n"
                   "normal : FP32 — baseline, always available.\n"
                   "fp16   : AMP Float16 — +10-15% on RTX 2000+.\n"
                   "bf16   : AMP BFloat16 — +25-35% on RTX 3000+ (Ampere).\n"
                   "tf32   : fast_matmul TF32 — +5-10% on RTX 3000+.\n\n"
                   "Recommended GTX 1080 Ti: normal only.\n"
                   "Recommended RTX 3070+: normal + bf16."))
        self._bench_modes_frame = f_modes

        # ── Tests/archs ciblés ──
        f_tests = ctk.CTkFrame(scroll, fg_color="transparent")
        f_tests.pack(fill="x", pady=3)
        ctk.CTkLabel(f_tests, text=_t("Tests ciblés :", "Targeted tests:"), width=170, anchor="w").pack(side="left")
        self.widgets["bench_tests"] = ctk.CTkEntry(
            f_tests, width=300,
            placeholder_text="vide = tous  |  ex: compact,span,hat_s")
        self.widgets["bench_tests"].pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f_tests, text="📋", width=30,
                      command=self._bench_list_tests).pack(side="left")
        ToolTip(self.widgets["bench_tests"],
                _t("Archs ou features à tester, séparés par virgule.\n"
                   "Vide = teste TOUT (peut durer plusieurs heures).\n\n"
                   "Exemples Arch :\n"
                   "  compact, span, hat_s, ultracompact, artcnn_r8f64\n\n"
                   "Exemples Feature :\n"
                   "  baseline, loss_mse, loss_ssim, optim_adan, loss_perc_conv\n\n"
                   "→ Cliquez 📋 pour lister tous les tests disponibles.",
                   "Archs or features to test, comma-separated.\n"
                   "Empty = test ALL (may take several hours).\n\n"
                   "Arch examples:\n"
                   "  compact, span, hat_s, ultracompact, artcnn_r8f64\n\n"
                   "Feature examples:\n"
                   "  baseline, loss_mse, loss_ssim, optim_adan, loss_perc_conv\n\n"
                   "→ Click 📋 to list all available tests."))

        # Train GT (optional override)
        r_gt = ctk.CTkFrame(scroll, fg_color="transparent")
        r_gt.pack(fill="x", pady=3)
        ctk.CTkLabel(r_gt, text=_t("Dataset train GT :", "Train dataset GT:"), width=170, anchor="w").pack(side="left")
        self.widgets["bench_gt"] = ctk.CTkEntry(r_gt,
            placeholder_text=str(Path.home() / "IA_Engine/datasets/train/HR"))
        self.widgets["bench_gt"].pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(r_gt, text="...", width=30,
                      command=lambda: self._browse_dir(self.widgets["bench_gt"])
                      ).pack(side="left")

        # Output dir (optional)
        r_out = ctk.CTkFrame(scroll, fg_color="transparent")
        r_out.pack(fill="x", pady=3)
        ctk.CTkLabel(r_out, text=_t("Dossier résultats :", "Results folder:"), width=170, anchor="w").pack(side="left")
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
            f_opts, text=_t("Reset état (repart de zéro)", "Reset state (start from scratch)"),
            onvalue="true", offvalue="false")
        self.widgets["bench_reset"].pack(side="left", padx=4)
        self.widgets["bench_no_upscale"] = ctk.CTkCheckBox(
            f_opts, text=_t("Pas de test upscale", "No upscale test"),
            onvalue="true", offvalue="false")
        self.widgets["bench_no_upscale"].pack(side="left", padx=12)
        ToolTip(self.widgets["bench_reset"],
                _t("Ignore l'état sauvegardé et relance tous les tests depuis le début.",
                   "Ignores saved state and reruns all tests from the beginning."))
        ToolTip(self.widgets["bench_no_upscale"],
                _t("Passe le test d'inférence rapide après chaque arch/feature.\n(Arch benchmark uniquement)",
                   "Skips the quick inference test after each arch/feature.\n(Arch benchmark only)"))

        # ── Action buttons ──
        f_btns = ctk.CTkFrame(f, fg_color="transparent")
        f_btns.pack(fill="x", pady=6)
        self.widgets["bench_run_btn"] = ctk.CTkButton(
            f_btns, text=_t("▶  Lancer le Benchmark", "▶  Run Benchmark"),
            fg_color="#2e7d32", hover_color="#1b5e20",
            font=("Arial", 13, "bold"), command=self._bench_run)
        self.widgets["bench_run_btn"].pack(side="left", padx=(0, 8), ipady=4)

        self.widgets["bench_stop_btn"] = ctk.CTkButton(
            f_btns, text="⏹ Stop",
            fg_color="#b71c1c", hover_color="#7f0000",
            state="disabled", command=self._bench_stop)
        self.widgets["bench_stop_btn"].pack(side="left", padx=(0, 8), ipady=4)

        ctk.CTkButton(
            f_btns, text=_t("📂 Ouvrir résultats", "📂 Open results"),
            fg_color="#1565c0", hover_color="#0d47a1",
            command=self._bench_open_results).pack(side="left", ipady=4)

        ctk.CTkButton(
            f_btns, text=_t("📋 Lister tests", "📋 List tests"),
            fg_color="transparent", border_width=1,
            command=self._bench_list_tests).pack(side="left", padx=8, ipady=4)

        # ── Poids pré-entraînés requis (SparK, ECO) ──────────────────────────
        f_weights = ctk.CTkFrame(f, fg_color="transparent")
        f_weights.pack(fill="x", pady=(2, 4))
        ctk.CTkLabel(f_weights,
                     text=_t("🔽 Poids requis (SparK / ECO) :", "🔽 Required weights (SparK / ECO):"),
                     font=("Arial", 11, "bold"), text_color="#aaaaaa", anchor="w").pack(side="left", padx=(2, 8))
        ctk.CTkButton(
            f_weights,
            text="⬇  SparK epoch290.pth",
            fg_color="#5c3d91", hover_color="#3d2460", width=200,
            font=("Arial", 11),
            command=self._bench_download_spark
        ).pack(side="left", padx=4)
        ToolTip(
            f_weights.winfo_children()[-1],
            _t("Télécharge le modèle InceptionNext pré-entraîné requis par SparK Perceptual Loss.\n"
               "Fichier : epoch290.pth (~200 MB)\n"
               "Source  : github.com/umzi2/SparK_Perceptual/releases\n"
               "Dossier : ~/IA_Engine/weights/spark/",
               "Downloads the pre-trained InceptionNext model required by SparK Perceptual Loss.\n"
               "File    : epoch290.pth (~200 MB)\n"
               "Source  : github.com/umzi2/SparK_Perceptual/releases\n"
               "Folder  : ~/IA_Engine/weights/spark/")
        )
        self.widgets["bench_spark_status"] = ctk.CTkLabel(
            f_weights, text="", text_color="#27ae60", font=("Consolas", 10))
        self.widgets["bench_spark_status"].pack(side="left", padx=8)

        # Status label
        self.widgets["bench_status"] = ctk.CTkLabel(
            f, text="", text_color="#aaaaaa", font=("Consolas", 11), anchor="w")
        self.widgets["bench_status"].pack(fill="x", padx=2, pady=(0, 4))

        # ── Live log ──
        self.widgets["bench_log"] = ctk.CTkTextbox(
            f, font=("Consolas", 10), fg_color=("#F5F5F5", "#0a0a0a"),
            text_color=("gray10", "#cccccc"), state="disabled")
        self.widgets["bench_log"].pack(fill="both", expand=True)

        return f

    # ── Benchmark callbacks ──────────────────────────────────────────────────────

    def _bench_download_spark(self):
        """Télécharge epoch290.pth (SparK InceptionNext pretrained ~200 MB) dans ~/IA_Engine/weights/spark/."""
        import threading
        import urllib.request
        import urllib.error
        from pathlib import Path

        # Essayer d'abord GitHub Releases, fallback HuggingFace
        _SPARK_URLS = [
            "https://github.com/umzi2/SparK_Perceptual/releases/download/model/epoch290.pth",
            "https://huggingface.co/umzi2/SparK_Perceptual/resolve/main/epoch290.pth",
        ]
        dest_dir  = Path.home() / "IA_Engine" / "weights" / "spark"
        dest_file = dest_dir / "epoch290.pth"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Vérification préalable
        if dest_file.exists() and dest_file.stat().st_size > 50_000_000:
            self._ui_update(self.widgets["bench_spark_status"].configure,
                            text=_t("✔ Déjà téléchargé", "✔ Already downloaded"),
                            text_color="#27ae60")
            return

        self._ui_update(self.widgets["bench_spark_status"].configure,
                        text=_t("⏳ Téléchargement…", "⏳ Downloading…"),
                        text_color="#f39c12")

        def _download():
            for url in _SPARK_URLS:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Universal-SR-Studio/1.0"})
                    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_file, "wb") as f:
                        total = int(resp.headers.get("Content-Length", 0))
                        downloaded = 0
                        while chunk := resp.read(65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = int(100 * downloaded / total)
                                self._ui_update(self.widgets["bench_spark_status"].configure,
                                                text=f"⏳ {pct}%", text_color="#f39c12")
                    if dest_file.stat().st_size > 50_000_000:
                        self._ui_update(self.widgets["bench_spark_status"].configure,
                                        text=_t(f"✔ Téléchargé → {dest_file}", f"✔ Downloaded → {dest_file}"),
                                        text_color="#27ae60")
                        return
                except Exception as e:
                    continue  # essayer prochain URL
            # Tous les URLs ont échoué
            self._ui_update(self.widgets["bench_spark_status"].configure,
                            text=_t("✗ Échec (voir console)", "✗ Download failed (see console)"),
                            text_color="#e74c3c")

        threading.Thread(target=_download, daemon=True).start()

    def _bench_on_engine_change(self, val):
        """Ajuste les défauts selon le moteur choisi."""
        pass  # engines use same script interface; defaults already set

    def _bench_on_type_change(self, val):
        """Ajuste les défauts n_iter / timeout selon le type."""
        if val in (_t("Architectures", "Architectures"), "Architectures",
                   _t("Arch + Features", "Arch + Features"), "Arch + Features"):
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
        # In a frozen build sys.executable is the .exe (no torch) → re-launches the app.
        # Use an engine venv python that actually has torch for benchmark_runner.py.
        py = sys.executable
        if getattr(sys, "frozen", False):
            py = _find_torch_python() or py
        return py, str(script)

    def _bench_build_cmd(self, python, script, list_only=False):
        """Retourne une liste de commandes à exécuter séquentiellement.

        Gère 'Les deux' (redux + neosr) et 'Arch + Features'.
        Chaque élément de la liste est une commande complète [python, script, ...args].
        benchmark_runner.py accepte UNE combinaison par appel — on le lance N fois si besoin.
        """
        engine_sel = self.widgets["bench_engine"].get()  # "Redux" | "NeoSR" | "Les deux"
        type_sel   = self.widgets["bench_type"].get()    # "Architectures" | "Features" | "Arch + Features"

        # Expansion sélection moteur
        engines = ["redux", "neosr"] if engine_sel in (_t("Les deux", "Both"), "Les deux", "Both") else [engine_sel.lower()]

        # Expansion sélection type
        btypes  = ["arch", "feature"] if type_sel in (_t("Arch + Features", "Arch + Features"), "Arch + Features") else \
                  ["arch"]            if type_sel in (_t("Architectures", "Architectures"), "Architectures")    else \
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
                _t(f"Script introuvable :\n{script}\n\nVérifiez l'installation du moteur.",
                   f"Script not found:\n{script}\n\nCheck the engine installation."))
            return

        cmds   = self._bench_build_cmd(python, script)
        engine = self.widgets["bench_engine"].get()
        btype  = self.widgets["bench_type"].get()

        # Reset log
        log = self.widgets["bench_log"]
        log.configure(state="normal"); log.delete("1.0", "end"); log.configure(state="disabled")
        self._bench_log_write(f"[Benchmark] {engine} — {btype}")
        if len(cmds) > 1:
            self._bench_log_write(f"[Benchmark] {len(cmds)} {_t('phase(s) à exécuter successivement.', 'phase(s) to run sequentially.')}\n")

        self.widgets["bench_run_btn"].configure(state="disabled")
        self.widgets["bench_stop_btn"].configure(state="normal")
        self._bench_status_set(_t("Benchmark en cours...", "Benchmark running..."))

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
                            phase_label = f"{_t('phase', 'phase')} {i}"
                        self._bench_log_write(
                            f"[Benchmark] {_t('Phase', 'Phase')} {i}/{len(cmds)} : {phase_label}")
                        self._bench_log_write(f"{'='*60}")
                    self._bench_log_write(f"[Benchmark] {_t('Commande', 'Command')} : {' '.join(cmd)}\n")

                    self._bench_proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        bufsize=1)
                    for line in self._bench_proc.stdout:
                        self._bench_log_write(line.rstrip())
                    self._bench_proc.wait()
                    rc = self._bench_proc.returncode

                    if rc != 0:
                        self._bench_status_set(_t(f"Phase {i} terminée avec erreur (code {rc}).", f"Phase {i} finished with error (code {rc})."))
                        self._bench_log_write(
                            f"\n[Benchmark] ❌ {_t(f'Phase {i} terminée avec erreur (code {rc}). Arrêt.', f'Phase {i} finished with error (code {rc}). Stopping.')}")
                        return  # stop on first error

                    self._bench_log_write(
                        f"\n[Benchmark] ✅ {_t(f'Phase {i}/{len(cmds)} terminée.', f'Phase {i}/{len(cmds)} done.')}")

                self._bench_status_set(_t("Benchmark terminé avec succès.", "Benchmark completed successfully."))
                self._bench_log_write(
                    f"\n[Benchmark] ✅ {_t('Toutes les phases terminées — voir les résultats.', 'All phases completed — see results.')}")
            except Exception as ex:
                self._bench_log_write(f"[Benchmark] {_t('ERREUR', 'ERROR')} : {ex}")
                self._bench_status_set(f"{_t('Erreur', 'Error')} : {ex}")
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
            self._bench_log_write(f"[Benchmark] {_t('Arrêt demandé par l\'utilisateur.', 'Stop requested by user.')}")
            self._bench_status_set(_t("Arrêté.", "Stopped."))
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
            messagebox.showerror(_t("Ouvrir résultats", "Open results"), _t("Impossible d'ouvrir", "Cannot open") + f" :\n{outdir}\n\n{ex}")

    def _bench_list_tests(self):
        python, script = self._bench_resolve_script()
        if not os.path.isfile(script):
            messagebox.showerror("Benchmark", _t(f"Script introuvable :\n{script}", f"Script not found:\n{script}"))
            return
        # list_only retourne une liste d'une seule commande (première combinaison)
        cmd = self._bench_build_cmd(python, script, list_only=True)[0]
        log = self.widgets["bench_log"]
        log.configure(state="normal"); log.delete("1.0", "end"); log.configure(state="disabled")
        self._bench_log_write(f"[Benchmark] {_t('Liste des tests disponibles :', 'Available tests list:')}\n")

        def _worker():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace")
                for line in proc.stdout:
                    self._bench_log_write(line.rstrip())
                proc.wait()
            except Exception as ex:
                self._bench_log_write(f"{_t('Erreur', 'Error')} : {ex}")

        threading.Thread(target=_worker, daemon=True).start()

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 14: POST PROCESSING  (v2.5.8)
    # Chaîne de post-traitement standalone : TF, UD, couleur, resize, netteté.
    # ═══════════════════════════════════════════════════════════════════════════

    def create_page_postproc(self):
        from tkinter import StringVar
        f = ctk.CTkFrame(self.right_panel, fg_color="transparent")

        # ── En-tête + GPU ────────────────────────────────────────────────────
        _top = ctk.CTkFrame(f, fg_color="transparent")
        _top.pack(fill="x", pady=(0, 4))
        self._create_gpu_panel(_top).pack(side="right", pady=4)
        _hdr = ctk.CTkFrame(_top, fg_color="transparent")
        _hdr.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(_hdr, text=_t("⚗ Post Processing", "⚗ Post Processing"),
                     font=("Roboto", 24, "bold"), text_color="#3B8ED0", anchor="w").pack(fill="x")
        ctk.CTkLabel(_hdr, text=_t(
            "Chaîne de post-traitement : Temporal Fix, Undistort, couleur, netteté, redimensionnement.",
            "Post-processing chain: Temporal Fix, Undistort, color, sharpening, resize."),
            font=("Arial", 12), text_color="gray", anchor="w").pack(fill="x")

        # ── Source ──────────────────────────────────────────────────────────
        irow = ctk.CTkFrame(f, fg_color="transparent")
        irow.pack(fill="x", pady=1)
        ctk.CTkLabel(irow, text=_t("Source (image ou dossier) :", "Source (image or folder):"),
                     width=220, anchor="w").pack(side="left")
        self._pp_input_var = StringVar(value=self.settings.get("pp_last_input", ""))
        self._pp_input_entry = ctk.CTkEntry(irow, textvariable=self._pp_input_var)
        self._pp_input_entry.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(irow, text=_t("Image", "Image"), width=60,
                      command=self._pp_pick_image).pack(side="left", padx=(0, 2))
        ctk.CTkButton(irow, text=_t("Dossier", "Folder"), width=70,
                      command=self._pp_pick_folder).pack(side="left")

        # ── Dossier sortie ───────────────────────────────────────────────────
        orow = ctk.CTkFrame(f, fg_color="transparent")
        orow.pack(fill="x", pady=1)
        ctk.CTkLabel(orow, text=_t("Dossier sortie :", "Output folder:"),
                     width=220, anchor="w").pack(side="left")
        self._pp_output_var = StringVar(value=self.settings.get("pp_last_output", ""))
        self._pp_output_entry = ctk.CTkEntry(orow, textvariable=self._pp_output_var)
        self._pp_output_entry.pack(side="left", fill="x", expand=True, padx=5)
        self._pp_out_btn = ctk.CTkButton(orow, text="...", width=30,
                                          command=self._pp_pick_output_folder)
        self._pp_out_btn.pack(side="left")

        # ── Options sortie ───────────────────────────────────────────────────
        optrow = ctk.CTkFrame(f, fg_color="transparent", height=34)
        optrow.pack_propagate(False)
        optrow.pack(fill="x", pady=(2, 0))
        ctk.CTkFrame(optrow, fg_color="transparent", width=1, height=1).pack(
            side="left", fill="x", expand=True)

        self._pp_same_folder = ctk.CTkCheckBox(
            optrow, text=_t("Même dossier que source", "Same folder as source"),
            command=self._pp_on_same_folder_toggle)
        self._pp_same_folder.pack(side="left", padx=(0, 15))
        if self.settings.get("pp_same_folder", False):
            self._pp_same_folder.select()

        self._pp_subfolder = ctk.CTkCheckBox(
            optrow, text=_t('Sous-dossier "postproc/"', '"postproc/" subfolder'))
        self._pp_subfolder.pack(side="left", padx=(0, 15))
        if self.settings.get("pp_subfolder", True):
            self._pp_subfolder.select()

        ctk.CTkFrame(optrow, width=1, fg_color="gray40").pack(side="left", fill="y", padx=(0, 10), pady=2)

        self._pp_serialize = ctk.CTkCheckBox(
            optrow, text=_t("Sérialisation", "Serialization"),
            command=self._pp_on_serialize_toggle)
        self._pp_serialize.pack(side="left", padx=(0, 6))
        if self.settings.get("pp_serialize", False):
            self._pp_serialize.select()
        ToolTip(self._pp_serialize, _t(
            "Nommage séquentiel : 00000.png, 00001.png…\nUtile pour réassembler en vidéo.",
            "Sequential naming: 00000.png, 00001.png…\nUseful for video reassembly."))

        ctk.CTkLabel(optrow, text=_t("Début :", "Start:")).pack(side="left", padx=(0, 3))
        self._pp_serialize_start = ctk.CTkEntry(optrow, width=55, placeholder_text="0")
        self._pp_serialize_start.pack(side="left")
        self._pp_serialize_start.insert(0, str(self.settings.get("pp_serialize_start", "0")))
        self._pp_on_serialize_toggle()

        self._pp_input_var.trace_add("write", lambda *_: self._pp_sync_output())
        self._pp_on_same_folder_toggle()

        # ── Chaîne de traitement ─────────────────────────────────────────────
        ctk.CTkLabel(f, text=_t("Chaîne de traitement (ordre d'application) :",
                                "Processing chain (application order):"),
                     font=("Arial", 12, "bold"), anchor="w").pack(fill="x", pady=(10, 2))

        chain_host = ctk.CTkScrollableFrame(f, fg_color=("#EBEBEB", "#1a2535"),
                                             corner_radius=8, height=210)
        chain_host.pack(fill="x", pady=(0, 6))

        self._pp_settings = {
            "tf_enabled":        bool(self.settings.get("pp_tf_enabled",        False)),
            "tf_mode":           str(self.settings.get("pp_tf_mode",            "classic")),
            "tf_strength":       float(self.settings.get("pp_tf_strength",      0.5)),
            "tf_window":         int(self.settings.get("pp_tf_window",          7)),
            "tf_precision":      str(self.settings.get("pp_tf_precision",       "float32")),
            "ud_enabled":        bool(self.settings.get("pp_ud_enabled",        False)),
            "ud_mode":           str(self.settings.get("pp_ud_mode",            "classic")),
            "ud_strength":       float(self.settings.get("pp_ud_strength",      0.35)),
            "ud_window":         int(self.settings.get("pp_ud_window",          5)),
            "ud_precision":      str(self.settings.get("pp_ud_precision",       "float32")),
            "color_enabled":     bool(self.settings.get("pp_color_enabled",     False)),
            "brightness":        float(self.settings.get("pp_brightness",       1.0)),
            "contrast":          float(self.settings.get("pp_contrast",         1.0)),
            "saturation":        float(self.settings.get("pp_saturation",       1.0)),
            "gamma":             float(self.settings.get("pp_gamma",            1.0)),
            "resize_enabled":    bool(self.settings.get("pp_resize_enabled",    False)),
            "resize_mode":       str(self.settings.get("pp_resize_mode",        "percent")),
            "resize_scale":      int(self.settings.get("pp_resize_scale",       100)),
            "resize_multiplier": float(self.settings.get("pp_resize_multiplier", 1.0)),
            "resize_width":      int(self.settings.get("pp_resize_width",        0)),
            "resize_height":     int(self.settings.get("pp_resize_height",       0)),
            "resize_aspect":     bool(self.settings.get("pp_resize_aspect",      True)),
            "resize_method":     str(self.settings.get("pp_resize_method",      "LANCZOS")),
            "sharpen_enabled":   bool(self.settings.get("pp_sharpen_enabled",   False)),
            "sharpen_strength":  float(self.settings.get("pp_sharpen_strength", 1.0)),
            "sharpen_radius":    float(self.settings.get("pp_sharpen_radius",   1.5)),
            "sharpen_threshold": int(self.settings.get("pp_sharpen_threshold",  3)),
        }

        _tools = [
            ("tf",      _t("🎞  Temporal Fix",        "🎞  Temporal Fix"),
                        _t("Réduit le scintillement temporel. Nécessite torch + séquence d'images.",
                           "Reduces temporal flickering. Requires torch + image sequence.")),
            ("ud",      _t("🔬  Undistort",            "🔬  Undistort"),
                        _t("Corrige les artefacts HF temporels (distorsion). Nécessite torch + séquence.",
                           "Fixes temporal HF artifacts (distortion). Requires torch + sequence.")),
            ("color",   _t("🎨  Correction couleur",   "🎨  Color correction"),
                        _t("Luminosité, contraste, saturation, gamma.",
                           "Brightness, contrast, saturation, gamma.")),
            ("resize",  _t("📐  Redimensionnement",    "📐  Resize"),
                        _t("Mise à l'échelle par pourcentage. Algorithme configurable (Lanczos, Bicubic…).",
                           "Scale by percentage. Configurable algorithm (Lanczos, Bicubic…).")),
            ("sharpen", _t("✨  Netteté (UnsharpMask)", "✨  Sharpen (UnsharpMask)"),
                        _t("Améliore la netteté. Force, rayon, seuil.",
                           "Improves sharpness. Strength, radius, threshold.")),
        ]

        self._pp_chain_checkboxes = {}
        for key, label, tip in _tools:
            row = ctk.CTkFrame(chain_host, fg_color="transparent")
            row.pack(fill="x", padx=6, pady=3)
            cb = ctk.CTkCheckBox(row, text="", width=24,
                                  command=lambda k=key: self._pp_on_tool_toggle(k))
            cb.pack(side="left", padx=(0, 4))
            if self._pp_settings.get(f"{key}_enabled"):
                cb.select()
            self._pp_chain_checkboxes[key] = cb
            lbl = ctk.CTkLabel(row, text=label, anchor="w", font=("Arial", 13))
            lbl.pack(side="left", fill="x", expand=True)
            ToolTip(lbl, tip)
            ctk.CTkButton(row, text="⚙", width=34,
                           command=lambda k=key: self._pp_open_settings(k)).pack(side="right")

        # ── Run / Stop / Progress ────────────────────────────────────────────
        run_row = ctk.CTkFrame(f, fg_color="transparent")
        run_row.pack(fill="x", pady=(10, 3))
        self.widgets["pp_run_btn"] = ctk.CTkButton(
            run_row, text=_t("▶ Lancer Post Processing", "▶ Run Post Processing"),
            fg_color="#2ecc71", command=self._pp_run)
        self.widgets["pp_run_btn"].pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.widgets["pp_stop_btn"] = ctk.CTkButton(
            run_row, text="⏹ Stop", fg_color="#e74c3c", hover_color="#c0392b",
            width=110, state="disabled", command=self._pp_request_stop)
        self.widgets["pp_stop_btn"].pack(side="left")

        prog_row = ctk.CTkFrame(f, fg_color="transparent")
        prog_row.pack(fill="x")
        self.widgets["pp_prog"] = ctk.CTkProgressBar(prog_row)
        self.widgets["pp_prog"].pack(side="left", fill="x", expand=True)
        self.widgets["pp_prog"].set(0)
        self.widgets["pp_prog_pct"] = ctk.CTkLabel(prog_row, text="0%", width=42, anchor="e")
        self.widgets["pp_prog_pct"].pack(side="left", padx=(6, 0))

        # ── Log + Preview (PanedWindow) ──────────────────────────────────────
        from tkinter import PanedWindow as _PanedWindow
        paned = _PanedWindow(f, orient="vertical", sashwidth=6, sashrelief="flat",
                             bg="#1a1a1a", bd=0, sashpad=1)
        paned.pack(fill="both", expand=True, pady=5)

        log_host = ctk.CTkFrame(paned, fg_color="transparent")
        _sash_h = int(self.settings.get("pp_log_sash_h", 150))
        paned.add(log_host, height=_sash_h, minsize=50)
        self.widgets["log_pp"] = ctk.CTkTextbox(log_host)
        self.widgets["log_pp"].pack(fill="both", expand=True)

        prev_frame = ctk.CTkFrame(paned, fg_color=("#E8E8E8", "#111827"), corner_radius=6)
        paned.add(prev_frame, minsize=80)
        for side, key, label in [("left",  "pp_prev_in",  _t("Avant", "Before")),
                                  ("right", "pp_prev_out", _t("Après", "After"))]:
            col = ctk.CTkFrame(prev_frame, fg_color="transparent")
            col.pack(side=side, fill="both", expand=True, padx=4, pady=4)
            ctk.CTkLabel(col, text=label, font=("Arial", 10), text_color="gray").pack()
            lbl = ctk.CTkLabel(col, text="—", fg_color=("#D0D0D0", "#1e293b"), corner_radius=4)
            lbl.pack(fill="both", expand=True)
            self.widgets[key] = lbl
        self._pp_preview_refs = []
        self._pp_last_preview_paths = None
        prev_frame.bind("<Configure>", self._pp_on_preview_resize)
        paned.bind("<ButtonRelease-1>", lambda e, p=paned: self._pp_save_sash(p))

        return f

    # ── Post Processing : helpers I/O ─────────────────────────────────────────

    def _pp_pick_image(self):
        path = filedialog.askopenfilename(
            title=_t("Sélectionner une image", "Select image"),
            filetypes=[(_t("Images", "Images"),
                        "*.png *.jpg *.jpeg *.webp *.tiff *.tif *.bmp"),
                       (_t("Tous les fichiers", "All files"), "*.*")])
        if path:
            self._pp_input_var.set(path)

    def _pp_pick_folder(self):
        path = filedialog.askdirectory(
            title=_t("Sélectionner un dossier source", "Select source folder"))
        if path:
            self._pp_input_var.set(path)

    def _pp_pick_output_folder(self):
        path = filedialog.askdirectory(
            title=_t("Sélectionner le dossier de sortie", "Select output folder"))
        if path:
            self._pp_output_var.set(path)

    def _pp_sync_output(self):
        if not (hasattr(self, "_pp_same_folder") and self._pp_same_folder.get()):
            return
        inp = self._pp_input_var.get().strip()
        base = inp if os.path.isdir(inp) else os.path.dirname(inp)
        self._pp_output_var.set(base)

    def _pp_on_same_folder_toggle(self):
        sf = hasattr(self, "_pp_same_folder") and bool(self._pp_same_folder.get())
        if hasattr(self, "_pp_out_btn"):
            self._pp_out_btn.configure(state="disabled" if sf else "normal")
        if hasattr(self, "_pp_output_entry"):
            self._pp_output_entry.configure(state="disabled" if sf else "normal")
        if sf:
            self._pp_sync_output()

    def _pp_on_serialize_toggle(self):
        on = hasattr(self, "_pp_serialize") and bool(self._pp_serialize.get())
        if hasattr(self, "_pp_serialize_start"):
            self._pp_serialize_start.configure(state="normal" if on else "disabled")

    def _pp_on_tool_toggle(self, key: str):
        enabled = bool(self._pp_chain_checkboxes[key].get())
        self._pp_settings[f"{key}_enabled"] = enabled
        self.settings.set(f"pp_{key}_enabled", enabled)

    def _pp_on_preview_resize(self, _event=None):
        paths = getattr(self, "_pp_last_preview_paths", None)
        if paths:
            self._pp_update_preview(*paths)

    def _pp_save_sash(self, paned):
        try:
            h = paned.sash_coord(0)[1]
            if h > 20:
                self.settings.set("pp_log_sash_h", h)
        except Exception:
            pass

    # ── Post Processing : settings popups ────────────────────────────────────

    def _pp_open_settings(self, key: str):
        dispatch = {
            "tf":      self._pp_settings_tf,
            "ud":      self._pp_settings_ud,
            "color":   self._pp_settings_color,
            "resize":  self._pp_settings_resize,
            "sharpen": self._pp_settings_sharpen,
        }
        fn = dispatch.get(key)
        if fn:
            fn()

    def _pp_settings_tf(self):
        """Popup Temporal Fix — Post-process (style identique à Quick Upscale v2.5.9)."""
        popup = getattr(self, "_pp_tf_popup", None)
        if popup and popup.winfo_exists():
            popup.lift()
            return

        s = self._pp_settings
        popup = ctk.CTkToplevel(self)
        popup.title(_t("PP — Réglages Temporal Fix", "PP — Temporal Fix Settings"))
        popup.resizable(False, False)
        popup.grab_set()
        self._pp_tf_popup = popup

        ctk.CTkLabel(popup, text="Temporal Fix — Post-processing",
                     font=("Roboto", 14, "bold"), text_color="#9B59B6").pack(
                     padx=20, pady=(15, 5))
        ctk.CTkLabel(popup,
                     text=_t(
                         "Réduction du scintillement SR sur séquences vidéo.\n"
                         "Blend adaptatif : zones statiques lissées, zones mobiles conservées.",
                         "SR flickering reduction on video sequences.\n"
                         "Adaptive blend: static regions smoothed, moving regions preserved."),
                     font=("Arial", 10), text_color="gray").pack(padx=20, pady=(0, 8))

        body = ctk.CTkFrame(popup, fg_color="transparent")
        body.pack(fill="x", padx=20, pady=4)

        ctk.CTkLabel(body, text=_t("Intensité (0 = désactivé, 1 = maximum) :",
                                   "Strength (0 = off, 1 = maximum):"),
                     anchor="w").pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "0.4-0.6 = recommandé pour anime SR. 0.8+ = effet fort (risque de flou).",
                         "0.4-0.6 = recommended for anime SR. 0.8+ = strong effect (blur risk)."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 6))
        str_row = ctk.CTkFrame(body, fg_color="transparent")
        str_row.pack(fill="x", pady=(0, 10))
        str_lbl = ctk.CTkLabel(str_row, text=f"{s.get('tf_strength', 0.5):.2f}", width=38, anchor="w")
        str_sld = ctk.CTkSlider(str_row, from_=0.0, to=1.0, number_of_steps=20, width=220)
        str_sld.set(s.get("tf_strength", 0.5))
        str_sld.pack(side="left")
        str_lbl.pack(side="left", padx=6)
        str_sld.configure(command=lambda v: str_lbl.configure(text=f"{float(v):.2f}"))

        ctk.CTkLabel(body, text=_t("Taille fenêtre (frames) :", "Window size (frames):"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        win_row = ctk.CTkFrame(body, fg_color="transparent")
        win_row.pack(fill="x", pady=(0, 6))
        win_var = ctk.StringVar(value=str(s.get("tf_window", 7)))
        win_menu = ctk.CTkOptionMenu(win_row, values=["5", "7", "9"], variable=win_var, width=90)
        win_menu.pack(side="left")
        lat_lbl = ctk.CTkLabel(win_row,
                               text=_t(f"→ latence {int(s.get('tf_window', 7)) // 2} frames",
                                       f"→ latency {int(s.get('tf_window', 7)) // 2} frames"),
                               font=("Consolas", 10), text_color="gray60")
        lat_lbl.pack(side="left", padx=(10, 0))
        win_menu.configure(command=lambda v: lat_lbl.configure(
            text=_t(f"→ latence {int(v) // 2} frames", f"→ latency {int(v) // 2} frames")))

        ctk.CTkLabel(body, text=_t("Précision :", "Precision:"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "float16 = 2× moins de VRAM, qualité identique.\n"
                         "float32 = sûr sur toutes les cartes (défaut).",
                         "float16 = 2× less VRAM, same quality.\n"
                         "float32 = safe on all cards (default)."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 6))
        prec_var = ctk.StringVar(value=s.get("tf_precision", "float32"))
        ctk.CTkOptionMenu(body, values=["float32", "float16"],
                          variable=prec_var, width=120).pack(anchor="w", pady=(0, 10))

        ctk.CTkFrame(body, height=1, fg_color="gray35").pack(fill="x", pady=(2, 8))
        ctk.CTkLabel(body, text=_t("Algorithme Temporal Fix :", "TemporalFix Algorithm:"),
                     font=("Roboto", 12, "bold"), anchor="w").pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "Variante : s1 léger / s2★ recommandé / s3 fort.\n"
                         "Backend : PyTorch CUDA | OnnxRuntime ⚡ (recommandé GPU) | TRT ⚡⚡ | CPU.",
                         "Variant: s1 light / s2★ recommended / s3 strong.\n"
                         "Backend: PyTorch CUDA | OnnxRuntime ⚡ (recommended GPU) | TRT ⚡⚡ | CPU."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 4))

        def _tf_parse_mode(m: str):
            if m == "classic":              return "classic", "pytorch"
            if m.startswith("model_"):      return m[6:], "pytorch"
            if m.startswith("ort_"):        return m[4:],  "ort"
            if m.startswith("trt_"):        return m[4:],  "trt"
            if m.startswith("cpu_"):        return m[4:],  "cpu"
            return "classic", "pytorch"

        def _tf_build_mode(variant: str, backend: str) -> str:
            if variant == "classic":
                return "classic"
            pfx = {"pytorch": "model_", "ort": "ort_", "trt": "trt_", "cpu": "cpu_"}.get(backend, "model_")
            return pfx + variant

        _tf_v_init, _tf_b_init = _tf_parse_mode(s.get("tf_mode", "classic"))
        _tf_variants = {"classic": _t("Classic (rapide / fallback)", "Classic (fast / fallback)"),
                        "s1": "s1  (~2 MB, léger)", "s2": "s2 ★ (~2 MB, recommandé)", "s3": "s3  (~2 MB, fort)"}
        _tf_variants_rev = {v: k for k, v in _tf_variants.items()}
        _tf_backends = {"pytorch": "PyTorch CUDA", "ort": "OnnxRuntime ⚡ (CUDA)",
                        "trt": "TRT + OnnxRuntime ⚡⚡", "cpu": "CPU (PyTorch)"}
        _tf_backends_rev = {v: k for k, v in _tf_backends.items()}

        tf_sel_row = ctk.CTkFrame(body, fg_color="transparent")
        tf_sel_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(tf_sel_row, text=_t("Variante:", "Variant:"), width=65, anchor="w").pack(side="left")
        tf_var_var = ctk.StringVar(value=_tf_variants.get(_tf_v_init, _tf_variants["classic"]))
        tf_var_menu = ctk.CTkOptionMenu(tf_sel_row, values=list(_tf_variants.values()),
                                        variable=tf_var_var, width=200)
        tf_var_menu.pack(side="left")

        tf_bck_row = ctk.CTkFrame(body, fg_color="transparent")
        tf_bck_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(tf_bck_row, text="Backend:", width=65, anchor="w").pack(side="left")
        tf_bck_var = ctk.StringVar(value=_tf_backends.get(_tf_b_init, _tf_backends["pytorch"]))
        tf_bck_menu = ctk.CTkOptionMenu(tf_bck_row, values=list(_tf_backends.values()),
                                        variable=tf_bck_var, width=200)
        tf_bck_menu.pack(side="left")

        tf_dl_row = ctk.CTkFrame(body, fg_color="transparent")
        tf_dl_row.pack(fill="x", pady=(0, 10))
        tf_dl_status = ctk.CTkLabel(tf_dl_row, text="", font=("Consolas", 10),
                                    text_color="gray60", anchor="w", width=200)
        tf_dl_status.pack(side="left")

        def _tf_current_model_key() -> str:
            variant = _tf_variants_rev.get(tf_var_var.get(), "classic")
            backend = _tf_backends_rev.get(tf_bck_var.get(), "pytorch")
            if variant == "classic":
                return ""
            if backend in ("ort", "trt"):
                return f"temporalfix_{variant}_onnx"
            return f"temporalfix_{variant}"

        def _tf_update_dl_status(*_):
            from src.core.model_manager import get_manager, MODELS
            variant = _tf_variants_rev.get(tf_var_var.get(), "classic")
            tf_bck_menu.configure(state="normal" if variant != "classic" else "disabled")
            model_key = _tf_current_model_key()
            if not model_key:
                tf_dl_status.configure(text=_t("Classic — aucun poids nécessaire.",
                                                "Classic — no weights needed."), text_color="gray60")
                tf_dl_btn.configure(state="disabled")
            else:
                mgr = get_manager()
                if mgr.is_ready(model_key):
                    info = MODELS[model_key]
                    tf_dl_status.configure(text=f"✓ {info['filename']} ({info['size_mb']:.1f} MB)",
                                           text_color="#2ecc71")
                    tf_dl_btn.configure(state="disabled", text=_t("Téléchargé ✓", "Downloaded ✓"))
                else:
                    info = MODELS[model_key]
                    tf_dl_status.configure(text=f"⬇ {info['filename']} ({info['size_mb']:.1f} MB)",
                                           text_color="#e67e22")
                    tf_dl_btn.configure(state="normal", text=_t("Télécharger", "Download"))

        def _tf_download():
            import threading as _thr
            from src.core.model_manager import get_manager
            model_key = _tf_current_model_key()
            if not model_key:
                return
            tf_dl_btn.configure(state="disabled", text=_t("Téléchargement…", "Downloading…"))
            tf_dl_status.configure(text="0 %", text_color="#3498db")
            def _run():
                try:
                    mgr = get_manager()
                    def _progress(dl, total):
                        pct = int(dl * 100 / total) if total else 0
                        popup.after(0, lambda: tf_dl_status.configure(
                            text=f"{pct}%  ({dl//1024}KB / {total//1024}KB)"))
                    mgr.download(model_key, progress_cb=_progress)
                    popup.after(0, _tf_update_dl_status)
                except Exception as e:
                    popup.after(0, lambda: (
                        tf_dl_status.configure(text=f"Erreur: {e}", text_color="#e74c3c"),
                        tf_dl_btn.configure(state="normal", text=_t("Réessayer", "Retry"))))
            _thr.Thread(target=_run, daemon=True).start()

        tf_dl_btn = ctk.CTkButton(tf_dl_row, text=_t("Télécharger", "Download"),
                                  width=110, height=26, command=_tf_download)
        tf_dl_btn.pack(side="right")
        tf_var_menu.configure(command=lambda _v: _tf_update_dl_status())
        tf_bck_menu.configure(command=lambda _v: _tf_update_dl_status())
        _tf_update_dl_status()

        btn_row = ctk.CTkFrame(popup, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 15))

        def _apply():
            _tf_v = _tf_variants_rev.get(tf_var_var.get(), "classic")
            _tf_b = _tf_backends_rev.get(tf_bck_var.get(), "pytorch")
            self._pp_settings["tf_mode"]      = _tf_build_mode(_tf_v, _tf_b)
            self._pp_settings["tf_strength"]  = round(float(str_sld.get()), 2)
            self._pp_settings["tf_window"]    = int(win_var.get())
            self._pp_settings["tf_precision"] = prec_var.get()
            self.settings.set("pp_tf_mode",      self._pp_settings["tf_mode"])
            self.settings.set("pp_tf_strength",  self._pp_settings["tf_strength"])
            self.settings.set("pp_tf_window",    self._pp_settings["tf_window"])
            self.settings.set("pp_tf_precision", self._pp_settings["tf_precision"])
            popup.destroy()

        ctk.CTkButton(btn_row, text=_t("Appliquer", "Apply"), fg_color="#2ecc71",
                      command=_apply).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(btn_row, text=_t("Annuler", "Cancel"), fg_color="#e74c3c",
                      command=popup.destroy).pack(side="left", fill="x", expand=True)

    def _pp_settings_ud(self):
        """Popup Undistort — Post-process (style identique à Quick Upscale v2.5.9)."""
        popup = getattr(self, "_pp_ud_popup", None)
        if popup and popup.winfo_exists():
            popup.lift()
            return

        s = self._pp_settings
        popup = ctk.CTkToplevel(self)
        popup.title(_t("PP — Réglages Undistort", "PP — Undistort Settings"))
        popup.resizable(False, False)
        popup.grab_set()
        self._pp_ud_popup = popup

        ctk.CTkLabel(popup, text="Undistort — Correction de jitter HF",
                     font=("Roboto", 14, "bold"), text_color="#27AE60").pack(
                     padx=20, pady=(15, 5))
        ctk.CTkLabel(popup,
                     text=_t(
                         "Réduit le 'shimmer' sur les contours (jitter haute fréquence).\n"
                         "Médiane temporelle du composant HF (frame − flou gaussien).\n"
                         "Complémentaire à Temporal Fix. Fenêtre plus petite = moins de latence.",
                         "Reduces edge 'shimmer' (high-frequency temporal jitter).\n"
                         "Method: temporal median of HF component (frame − Gaussian blur).\n"
                         "Complementary to Temporal Fix. Smaller window = less latency."),
                     font=("Arial", 10), text_color="gray").pack(padx=20, pady=(0, 8))

        body = ctk.CTkFrame(popup, fg_color="transparent")
        body.pack(fill="x", padx=20, pady=4)

        ctk.CTkLabel(body, text=_t("Intensité Undistort :", "Undistort Strength:"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "0.3–0.5 = recommandé. Plus élevé = correction plus forte (risque de flou HF).",
                         "0.3–0.5 = recommended. Higher = stronger correction (HF blur risk)."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 4))
        undist_str_row = ctk.CTkFrame(body, fg_color="transparent")
        undist_str_row.pack(fill="x", pady=(0, 8))
        _ud_str_init = float(s.get("ud_strength", 0.35))
        undist_str_lbl = ctk.CTkLabel(undist_str_row, text=f"{_ud_str_init:.2f}", width=38, anchor="w")
        undist_str_sld = ctk.CTkSlider(undist_str_row, from_=0.0, to=1.0, number_of_steps=20, width=220)
        undist_str_sld.set(_ud_str_init)
        undist_str_sld.pack(side="left")
        undist_str_lbl.pack(side="left", padx=6)
        undist_str_sld.configure(command=lambda v: undist_str_lbl.configure(text=f"{float(v):.2f}"))

        ctk.CTkLabel(body, text=_t("Fenêtre Undistort (frames) :", "Undistort Window (frames):"),
                     anchor="w").pack(fill="x", pady=(4, 2))
        udist_win_var = ctk.StringVar(value=str(s.get("ud_window", 5)))
        ctk.CTkOptionMenu(body, values=["3", "5", "7"], variable=udist_win_var, width=90).pack(
            anchor="w", pady=(0, 8))

        ctk.CTkLabel(body, text=_t("Algorithme Undistort :", "Undistort Algorithm:"),
                     anchor="w", font=("Roboto", 11, "bold")).pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(body,
                     text=_t(
                         "Classic : médiane HF, rapide, sans poids.\n"
                         "TMT : réseau xg416/pifroggi. Backend : ORT ⚡ recommandé GPU.",
                         "Classic: HF median, fast, no weights.\n"
                         "TMT: xg416/pifroggi network. Backend: ORT ⚡ recommended GPU."),
                     font=("Arial", 10), text_color="gray", anchor="w").pack(fill="x", pady=(0, 4))

        def _ud_parse_mode(m: str):
            if m == "classic":          return "classic", "pytorch"
            if m.startswith("model_"):  return "tmt", "pytorch"
            if m.startswith("ort_"):    return "tmt",  "ort"
            if m.startswith("trt_"):    return "tmt",  "trt"
            return "classic", "pytorch"

        def _ud_build_mode(variant: str, backend: str) -> str:
            if variant == "classic":
                return "classic"
            pfx = {"pytorch": "model_", "ort": "ort_", "trt": "trt_"}.get(backend, "model_")
            return pfx + "tmt"

        _ud_v_init, _ud_b_init = _ud_parse_mode(s.get("ud_mode", "classic"))
        _ud_variants = {"classic": _t("Classic (rapide / fallback)", "Classic (fast / fallback)"),
                        "tmt": "TMT (~8 MB PTH / ~5 MB ONNX)"}
        _ud_variants_rev = {v: k for k, v in _ud_variants.items()}
        _ud_backends = {"pytorch": "PyTorch CUDA", "ort": "OnnxRuntime ⚡ (CUDA)",
                        "trt": "TRT + OnnxRuntime ⚡⚡"}
        _ud_backends_rev = {v: k for k, v in _ud_backends.items()}

        ud_sel_row = ctk.CTkFrame(body, fg_color="transparent")
        ud_sel_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(ud_sel_row, text=_t("Variante:", "Variant:"), width=65, anchor="w").pack(side="left")
        ud_var_var = ctk.StringVar(value=_ud_variants.get(_ud_v_init, _ud_variants["classic"]))
        ud_var_menu = ctk.CTkOptionMenu(ud_sel_row, values=list(_ud_variants.values()),
                                        variable=ud_var_var, width=200)
        ud_var_menu.pack(side="left")

        ud_bck_row = ctk.CTkFrame(body, fg_color="transparent")
        ud_bck_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(ud_bck_row, text="Backend:", width=65, anchor="w").pack(side="left")
        ud_bck_var = ctk.StringVar(value=_ud_backends.get(_ud_b_init, _ud_backends["pytorch"]))
        ud_bck_menu = ctk.CTkOptionMenu(ud_bck_row, values=list(_ud_backends.values()),
                                        variable=ud_bck_var, width=200)
        ud_bck_menu.pack(side="left")

        ud_dl_row = ctk.CTkFrame(body, fg_color="transparent")
        ud_dl_row.pack(fill="x", pady=(0, 12))
        ud_dl_status = ctk.CTkLabel(ud_dl_row, text="", font=("Consolas", 10),
                                    text_color="gray60", anchor="w", width=200)
        ud_dl_status.pack(side="left")

        def _ud_current_model_key() -> str:
            variant = _ud_variants_rev.get(ud_var_var.get(), "classic")
            backend = _ud_backends_rev.get(ud_bck_var.get(), "pytorch")
            if variant == "classic":
                return ""
            if backend in ("ort", "trt"):
                return "undistort_tmt_onnx"
            return "undistort_tmt"

        def _ud_update_dl_status(*_):
            from src.core.model_manager import get_manager, MODELS
            variant = _ud_variants_rev.get(ud_var_var.get(), "classic")
            ud_bck_menu.configure(state="normal" if variant != "classic" else "disabled")
            model_key = _ud_current_model_key()
            if not model_key:
                ud_dl_status.configure(text=_t("Classic — aucun poids nécessaire.",
                                                "Classic — no weights needed."), text_color="gray60")
                ud_dl_btn.configure(state="disabled")
            else:
                mgr = get_manager()
                if mgr.is_ready(model_key):
                    info = MODELS[model_key]
                    ud_dl_status.configure(text=f"✓ {info['filename']} ({info['size_mb']:.1f} MB)",
                                           text_color="#2ecc71")
                    ud_dl_btn.configure(state="disabled", text=_t("Téléchargé ✓", "Downloaded ✓"))
                else:
                    info = MODELS[model_key]
                    ud_dl_status.configure(text=f"⬇ {info['filename']} ({info['size_mb']:.1f} MB)",
                                           text_color="#e67e22")
                    ud_dl_btn.configure(state="normal", text=_t("Télécharger", "Download"))

        def _ud_download():
            import threading as _thr
            from src.core.model_manager import get_manager
            model_key = _ud_current_model_key()
            if not model_key:
                return
            ud_dl_btn.configure(state="disabled", text=_t("Téléchargement…", "Downloading…"))
            ud_dl_status.configure(text="0 %", text_color="#3498db")
            def _run():
                try:
                    mgr = get_manager()
                    def _progress(dl, total):
                        pct = int(dl * 100 / total) if total else 0
                        popup.after(0, lambda: ud_dl_status.configure(
                            text=f"{pct}%  ({dl//1024}KB / {total//1024}KB)"))
                    mgr.download(model_key, progress_cb=_progress)
                    popup.after(0, _ud_update_dl_status)
                except Exception as e:
                    popup.after(0, lambda: (
                        ud_dl_status.configure(text=f"Erreur: {e}", text_color="#e74c3c"),
                        ud_dl_btn.configure(state="normal", text=_t("Réessayer", "Retry"))))
            _thr.Thread(target=_run, daemon=True).start()

        ud_dl_btn = ctk.CTkButton(ud_dl_row, text=_t("Télécharger", "Download"),
                                  width=110, height=26, command=_ud_download)
        ud_dl_btn.pack(side="right")
        ud_var_menu.configure(command=lambda _v: _ud_update_dl_status())
        ud_bck_menu.configure(command=lambda _v: _ud_update_dl_status())
        _ud_update_dl_status()

        btn_row = ctk.CTkFrame(popup, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 15))

        def _apply():
            _ud_v = _ud_variants_rev.get(ud_var_var.get(), "classic")
            _ud_b = _ud_backends_rev.get(ud_bck_var.get(), "pytorch")
            self._pp_settings["ud_mode"]     = _ud_build_mode(_ud_v, _ud_b)
            self._pp_settings["ud_strength"] = round(float(undist_str_sld.get()), 2)
            self._pp_settings["ud_window"]   = int(udist_win_var.get())
            self.settings.set("pp_ud_mode",     self._pp_settings["ud_mode"])
            self.settings.set("pp_ud_strength", self._pp_settings["ud_strength"])
            self.settings.set("pp_ud_window",   self._pp_settings["ud_window"])
            popup.destroy()

        ctk.CTkButton(btn_row, text=_t("Appliquer", "Apply"), fg_color="#2ecc71",
                      command=_apply).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(btn_row, text=_t("Annuler", "Cancel"), fg_color="#e74c3c",
                      command=popup.destroy).pack(side="left", fill="x", expand=True)

    def _pp_settings_color(self):
        from tkinter import DoubleVar, Canvas as _Canvas
        import os as _os
        _ensure_pil()
        dlg = ctk.CTkToplevel(self)
        dlg.title(_t("Réglages — Correction couleur", "Settings — Color correction"))
        dlg.resizable(True, True)
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()
        dlg.geometry("480x640")

        _bri_var = DoubleVar(value=self._pp_settings.get("brightness", 1.0))
        _con_var = DoubleVar(value=self._pp_settings.get("contrast",   1.0))
        _sat_var = DoubleVar(value=self._pp_settings.get("saturation", 1.0))
        _gam_var = DoubleVar(value=self._pp_settings.get("gamma",      1.0))

        # ── Preview helpers ───────────────────────────────────────────────────
        _zoom    = [1.0]
        _tk_ref  = [None]
        _sched   = [False]
        _orig    = [None]

        def _load_src():
            prev = getattr(self, "_pp_last_preview_paths", None)
            if prev and isinstance(prev, tuple) and len(prev) >= 1:
                try:
                    return Image.open(prev[0]).convert("RGB")
                except Exception:
                    pass
            inp = self._pp_input_var.get().strip()
            _IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}
            if _os.path.isfile(inp) and _os.path.splitext(inp)[1].lower() in _IMG_EXT:
                try:
                    return Image.open(inp).convert("RGB")
                except Exception:
                    pass
            if _os.path.isdir(inp):
                for fn in sorted(_os.listdir(inp)):
                    if _os.path.splitext(fn)[1].lower() in _IMG_EXT:
                        try:
                            return Image.open(_os.path.join(inp, fn)).convert("RGB")
                        except Exception:
                            pass
            return None

        _orig[0] = _load_src()

        # ── Sliders ───────────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(dlg, fg_color="transparent")
        ctrl.pack(fill="x", padx=0)

        def _slider_row(parent, label_text, var, from_, to):
            lbl = ctk.CTkLabel(parent, text=f"{label_text} : {var.get():.2f}", anchor="w")
            lbl.pack(fill="x", padx=20, pady=(10, 0))
            def _upd(v, l=lbl, lt=label_text):
                l.configure(text=f"{lt} : {float(v):.2f}")
                _schedule_preview()
            ctk.CTkSlider(parent, from_=from_, to=to, variable=var,
                          command=_upd).pack(fill="x", padx=20)

        _slider_row(ctrl, _t("Luminosité", "Brightness"), _bri_var, 0.3, 3.0)
        _slider_row(ctrl, _t("Contraste",  "Contrast"),   _con_var, 0.3, 3.0)
        _slider_row(ctrl, _t("Saturation", "Saturation"), _sat_var, 0.0, 2.0)
        _slider_row(ctrl, _t("Gamma",      "Gamma"),      _gam_var, 0.2, 4.0)

        ctk.CTkLabel(ctrl, text=_t(
            "1.0 = pas de changement pour chaque paramètre.",
            "1.0 = no change for each parameter."),
            font=("Arial", 11), text_color="gray").pack(padx=20, pady=(6, 0), anchor="w")

        # ── Live preview ──────────────────────────────────────────────────────
        ctk.CTkFrame(dlg, height=1, fg_color="gray35").pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(dlg,
                     text=_t("Aperçu (molette = zoom)", "Preview (scroll = zoom)"),
                     font=("Arial", 10), text_color="gray").pack(anchor="w", padx=12)

        prev_frame = ctk.CTkFrame(dlg, fg_color="#181818", corner_radius=6)
        prev_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        canvas = _Canvas(prev_frame, bg="#181818", highlightthickness=0, cursor="crosshair")
        canvas.pack(fill="both", expand=True, padx=2, pady=2)

        def _render():
            _sched[0] = False
            from PIL import ImageTk
            src = _orig[0]
            if src is None:
                canvas.delete("all")
                cw = canvas.winfo_width() or 400
                ch = canvas.winfo_height() or 200
                canvas.create_text(cw // 2, ch // 2,
                                   text=_t("Aucune image source", "No source image"),
                                   fill="#555", font=("Arial", 11))
                return
            try:
                ps = {
                    "color_enabled":   True,
                    "brightness":      float(_bri_var.get()),
                    "contrast":        float(_con_var.get()),
                    "saturation":      float(_sat_var.get()),
                    "gamma":           float(_gam_var.get()),
                    "resize_enabled":  False,
                    "sharpen_enabled": False,
                }
                img = self._pp_apply_pil_chain(src.copy(), ps)
                z = _zoom[0]
                cw = max(1, canvas.winfo_width() or 400)
                ch = max(1, canvas.winfo_height() or 220)
                dw = max(1, round(img.width * z))
                dh = max(1, round(img.height * z))
                if z == 1.0:
                    ratio = min(cw / img.width, ch / img.height, 1.0)
                    dw = max(1, round(img.width * ratio))
                    dh = max(1, round(img.height * ratio))
                resamp = Image.Resampling.LANCZOS if z <= 1.0 else Image.Resampling.NEAREST
                disp = img.resize((dw, dh), resamp)
                x0 = max(0, (dw - cw) // 2)
                y0 = max(0, (dh - ch) // 2)
                disp = disp.crop((x0, y0, x0 + min(cw, dw), y0 + min(ch, dh)))
                tk_img = ImageTk.PhotoImage(disp)
                _tk_ref[0] = tk_img
                canvas.delete("all")
                canvas.create_image(0, 0, anchor="nw", image=tk_img)
            except Exception as e:
                canvas.delete("all")
                canvas.create_text(8, 8, anchor="nw", text=str(e),
                                   fill="#e74c3c", font=("Arial", 9))

        def _schedule_preview(*_):
            if not _sched[0]:
                _sched[0] = True
                dlg.after(80, _render)

        def _on_wheel(event):
            if event.delta > 0:
                _zoom[0] = min(8.0, _zoom[0] * 1.2)
            else:
                _zoom[0] = max(0.05, _zoom[0] / 1.2)
            _render()

        canvas.bind("<MouseWheel>", _on_wheel)
        dlg.after(120, _render)

        # ── Confirm ───────────────────────────────────────────────────────────
        def _confirm():
            for k, v in [("brightness", _bri_var), ("contrast", _con_var),
                         ("saturation", _sat_var), ("gamma", _gam_var)]:
                val = round(float(v.get()), 3)
                self._pp_settings[k] = val
                self.settings.set(f"pp_{k}", val)
            dlg.destroy()

        ctk.CTkButton(dlg, text=_t("Confirmer", "Confirm"), command=_confirm).pack(pady=8)
        dlg.wait_window()

    def _pp_settings_resize(self):
        from tkinter import IntVar, StringVar as _SV, DoubleVar, BooleanVar
        dlg = ctk.CTkToplevel(self)
        dlg.title(_t("Réglages — Redimensionnement", "Settings — Resize"))
        dlg.resizable(False, False)
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()
        dlg.geometry("420x360")

        _mode_var   = _SV(value=self._pp_settings.get("resize_mode", "percent"))
        _scale_var  = IntVar(value=self._pp_settings.get("resize_scale", 100))
        _mult_var   = _SV(value=str(self._pp_settings.get("resize_multiplier", 1.0)))
        _w_var      = _SV(value=str(self._pp_settings.get("resize_width", 0) or ""))
        _h_var      = _SV(value=str(self._pp_settings.get("resize_height", 0) or ""))
        _asp_var    = BooleanVar(value=self._pp_settings.get("resize_aspect", True))
        _method_var = _SV(value=self._pp_settings.get("resize_method", "LANCZOS"))

        # ── Mode selector ─────────────────────────────────────────────────────
        mode_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        mode_frame.pack(fill="x", padx=20, pady=(14, 6))
        ctk.CTkLabel(mode_frame, text=_t("Mode :", "Mode:"), width=60, anchor="w").pack(side="left")
        for _lbl, _val in [("% Scale", "percent"), ("× Mult", "multiplier"), ("px Pixels", "pixels")]:
            ctk.CTkRadioButton(mode_frame, text=_lbl, variable=_mode_var, value=_val,
                               command=lambda: _show_mode()).pack(side="left", padx=(0, 12))

        # ── Content area (swapped per mode) ───────────────────────────────────
        content = ctk.CTkFrame(dlg, fg_color="transparent", height=140)
        content.pack(fill="x", padx=20)
        content.pack_propagate(False)

        _active_widgets = [None]

        def _clear():
            for w in content.winfo_children():
                w.destroy()

        def _show_mode():
            _clear()
            mode = _mode_var.get()
            if mode == "percent":
                _scale_lbl = ctk.CTkLabel(content,
                    text=f"Scale : {_scale_var.get()} %", anchor="w")
                _scale_lbl.pack(fill="x", pady=(6, 0))
                def _upd(v): _scale_lbl.configure(text=f"Scale : {int(float(v))} %")
                ctk.CTkSlider(content, from_=10, to=400, variable=_scale_var,
                              command=_upd, number_of_steps=390).pack(fill="x")
                ctk.CTkLabel(content, text=_t(
                    "10 %–400 %. 100 % = pas de redimensionnement.",
                    "10 %–400 %. 100 % = no resize."),
                    font=("Arial", 10), text_color="gray").pack(anchor="w", pady=(4, 0))

            elif mode == "multiplier":
                ctk.CTkLabel(content, text=_t("Multiplicateur :", "Multiplier:"), anchor="w").pack(
                    fill="x", pady=(6, 2))
                _mults = ["0.25", "0.5", "1.0", "2.0", "3.0", "4.0", "8.0"]
                ctk.CTkOptionMenu(content, variable=_mult_var, values=_mults, width=120).pack(anchor="w")
                ctk.CTkLabel(content, text=_t(
                    "2× = doubler la résolution. 0.5× = réduire de moitié.",
                    "2× = double resolution. 0.5× = halve."),
                    font=("Arial", 10), text_color="gray").pack(anchor="w", pady=(4, 0))

            elif mode == "pixels":
                px_row = ctk.CTkFrame(content, fg_color="transparent")
                px_row.pack(fill="x", pady=(6, 2))
                ctk.CTkLabel(px_row, text="W:", width=22, anchor="w").pack(side="left")
                ctk.CTkEntry(px_row, textvariable=_w_var, width=80,
                             placeholder_text="1920").pack(side="left", padx=(0, 10))
                ctk.CTkLabel(px_row, text="H:", width=22, anchor="w").pack(side="left")
                ctk.CTkEntry(px_row, textvariable=_h_var, width=80,
                             placeholder_text="1080").pack(side="left")
                ctk.CTkCheckBox(content, text=_t("Garder ratio", "Keep aspect"),
                                variable=_asp_var).pack(anchor="w", pady=(4, 0))
                ctk.CTkLabel(content, text=_t(
                    "Laisser W ou H à 0 pour calculer automatiquement.",
                    "Leave W or H at 0 to auto-calculate."),
                    font=("Arial", 10), text_color="gray").pack(anchor="w", pady=(2, 0))

        _show_mode()

        # ── Algorithm ─────────────────────────────────────────────────────────
        ctk.CTkFrame(dlg, height=1, fg_color="gray35").pack(fill="x", padx=10, pady=(8, 6))
        alg_row = ctk.CTkFrame(dlg, fg_color="transparent")
        alg_row.pack(fill="x", padx=20)
        ctk.CTkLabel(alg_row, text=_t("Algo :", "Algo:"), width=50, anchor="w").pack(side="left")
        ctk.CTkOptionMenu(alg_row, variable=_method_var,
                          values=["LANCZOS", "BICUBIC", "BILINEAR", "NEAREST"],
                          width=140).pack(side="left")
        ctk.CTkLabel(alg_row, text=_t("★ LANCZOS = best quality", "★ LANCZOS = best quality"),
                     font=("Arial", 10), text_color="gray").pack(side="left", padx=(8, 0))

        # ── Confirm ───────────────────────────────────────────────────────────
        def _confirm():
            mode = _mode_var.get()
            self._pp_settings["resize_mode"]   = mode
            self._pp_settings["resize_method"] = _method_var.get()
            self.settings.set("pp_resize_mode",   mode)
            self.settings.set("pp_resize_method", _method_var.get())
            if mode == "percent":
                self._pp_settings["resize_scale"] = int(_scale_var.get())
                self.settings.set("pp_resize_scale", int(_scale_var.get()))
            elif mode == "multiplier":
                try:
                    m = float(_mult_var.get())
                except ValueError:
                    m = 1.0
                self._pp_settings["resize_multiplier"] = m
                self.settings.set("pp_resize_multiplier", m)
            elif mode == "pixels":
                try:
                    w = int(_w_var.get()) if _w_var.get().strip().isdigit() else 0
                except Exception:
                    w = 0
                try:
                    h = int(_h_var.get()) if _h_var.get().strip().isdigit() else 0
                except Exception:
                    h = 0
                self._pp_settings["resize_width"]  = w
                self._pp_settings["resize_height"] = h
                self._pp_settings["resize_aspect"] = bool(_asp_var.get())
                self.settings.set("pp_resize_width",  w)
                self.settings.set("pp_resize_height", h)
                self.settings.set("pp_resize_aspect", bool(_asp_var.get()))
            dlg.destroy()

        ctk.CTkButton(dlg, text=_t("Confirmer", "Confirm"), command=_confirm).pack(pady=12)
        dlg.wait_window()

    def _pp_settings_sharpen(self):
        from tkinter import DoubleVar, IntVar, Canvas as _Canvas
        import os as _os
        _ensure_pil()
        dlg = ctk.CTkToplevel(self)
        dlg.title(_t("Réglages — Netteté", "Settings — Sharpen"))
        dlg.resizable(True, True)
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()
        dlg.geometry("480x580")

        _str_var = DoubleVar(value=self._pp_settings.get("sharpen_strength",  1.0))
        _rad_var = DoubleVar(value=self._pp_settings.get("sharpen_radius",    1.5))
        _thr_var = IntVar(value=self._pp_settings.get("sharpen_threshold", 3))

        # ── Preview helpers ───────────────────────────────────────────────────
        _zoom   = [1.0]
        _tk_ref = [None]
        _sched  = [False]
        _orig   = [None]

        def _load_src():
            prev = getattr(self, "_pp_last_preview_paths", None)
            if prev and isinstance(prev, tuple) and len(prev) >= 1:
                try:
                    return Image.open(prev[0]).convert("RGB")
                except Exception:
                    pass
            inp = self._pp_input_var.get().strip()
            _IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}
            if _os.path.isfile(inp) and _os.path.splitext(inp)[1].lower() in _IMG_EXT:
                try:
                    return Image.open(inp).convert("RGB")
                except Exception:
                    pass
            if _os.path.isdir(inp):
                for fn in sorted(_os.listdir(inp)):
                    if _os.path.splitext(fn)[1].lower() in _IMG_EXT:
                        try:
                            return Image.open(_os.path.join(inp, fn)).convert("RGB")
                        except Exception:
                            pass
            return None

        _orig[0] = _load_src()

        # ── Sliders ───────────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(dlg, fg_color="transparent")
        ctrl.pack(fill="x", padx=0)

        def _slider_row(parent, label_text, var, from_, to, fmt=".2f"):
            lbl = ctk.CTkLabel(parent, text=f"{label_text} : {var.get():{fmt}}", anchor="w")
            lbl.pack(fill="x", padx=20, pady=(10, 0))
            def _upd(v, l=lbl, lt=label_text, f=fmt):
                l.configure(text=f"{lt} : {float(v):{f}}")
                _schedule_preview()
            ctk.CTkSlider(parent, from_=from_, to=to, variable=var,
                          command=_upd).pack(fill="x", padx=20)

        _slider_row(ctrl, _t("Force (Percent)", "Strength (Percent)"), _str_var, 0.0, 5.0)
        _slider_row(ctrl, _t("Rayon",           "Radius"),             _rad_var, 0.3, 6.0)
        _slider_row(ctrl, _t("Seuil (Threshold)", "Threshold"),        _thr_var, 0,  15, ".0f")

        ctk.CTkLabel(ctrl, text=_t(
            "Force = multiplicateur du percent UnsharpMask (x100).\n"
            "Seuil = différence minimum pour appliquer l'effet.",
            "Strength = multiplier for UnsharpMask percent (x100).\n"
            "Threshold = minimum difference to apply effect."),
            font=("Arial", 10), text_color="gray").pack(padx=20, pady=(8, 0), anchor="w")

        # ── Live preview ──────────────────────────────────────────────────────
        ctk.CTkFrame(dlg, height=1, fg_color="gray35").pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(dlg,
                     text=_t("Aperçu (molette = zoom)", "Preview (scroll = zoom)"),
                     font=("Arial", 10), text_color="gray").pack(anchor="w", padx=12)

        prev_frame = ctk.CTkFrame(dlg, fg_color="#181818", corner_radius=6)
        prev_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        canvas = _Canvas(prev_frame, bg="#181818", highlightthickness=0, cursor="crosshair")
        canvas.pack(fill="both", expand=True, padx=2, pady=2)

        def _render():
            _sched[0] = False
            from PIL import ImageTk
            src = _orig[0]
            if src is None:
                canvas.delete("all")
                cw = canvas.winfo_width() or 400
                ch = canvas.winfo_height() or 200
                canvas.create_text(cw // 2, ch // 2,
                                   text=_t("Aucune image source", "No source image"),
                                   fill="#555", font=("Arial", 11))
                return
            try:
                ps = {
                    "color_enabled":    False,
                    "resize_enabled":   False,
                    "sharpen_enabled":  True,
                    "sharpen_strength": float(_str_var.get()),
                    "sharpen_radius":   float(_rad_var.get()),
                    "sharpen_threshold": int(_thr_var.get()),
                }
                img = self._pp_apply_pil_chain(src.copy(), ps)
                z = _zoom[0]
                cw = max(1, canvas.winfo_width() or 400)
                ch = max(1, canvas.winfo_height() or 220)
                dw = max(1, round(img.width * z))
                dh = max(1, round(img.height * z))
                if z == 1.0:
                    ratio = min(cw / img.width, ch / img.height, 1.0)
                    dw = max(1, round(img.width * ratio))
                    dh = max(1, round(img.height * ratio))
                resamp = Image.Resampling.LANCZOS if z <= 1.0 else Image.Resampling.NEAREST
                disp = img.resize((dw, dh), resamp)
                x0 = max(0, (dw - cw) // 2)
                y0 = max(0, (dh - ch) // 2)
                disp = disp.crop((x0, y0, x0 + min(cw, dw), y0 + min(ch, dh)))
                tk_img = ImageTk.PhotoImage(disp)
                _tk_ref[0] = tk_img
                canvas.delete("all")
                canvas.create_image(0, 0, anchor="nw", image=tk_img)
            except Exception as e:
                canvas.delete("all")
                canvas.create_text(8, 8, anchor="nw", text=str(e),
                                   fill="#e74c3c", font=("Arial", 9))

        def _schedule_preview(*_):
            if not _sched[0]:
                _sched[0] = True
                dlg.after(80, _render)

        def _on_wheel(event):
            if event.delta > 0:
                _zoom[0] = min(8.0, _zoom[0] * 1.2)
            else:
                _zoom[0] = max(0.05, _zoom[0] / 1.2)
            _render()

        canvas.bind("<MouseWheel>", _on_wheel)
        dlg.after(120, _render)

        # ── Confirm ───────────────────────────────────────────────────────────
        def _confirm():
            self._pp_settings["sharpen_strength"]  = round(float(_str_var.get()), 3)
            self._pp_settings["sharpen_radius"]    = round(float(_rad_var.get()), 2)
            self._pp_settings["sharpen_threshold"] = int(_thr_var.get())
            self.settings.set("pp_sharpen_strength",  round(float(_str_var.get()), 3))
            self.settings.set("pp_sharpen_radius",    round(float(_rad_var.get()), 2))
            self.settings.set("pp_sharpen_threshold", int(_thr_var.get()))
            dlg.destroy()

        ctk.CTkButton(dlg, text=_t("Confirmer", "Confirm"), command=_confirm).pack(pady=8)
        dlg.wait_window()

    # ── Post Processing : run / stop ─────────────────────────────────────────

    def _pp_run(self):
        _ensure_pil()
        inp = self._pp_input_var.get().strip()
        if not inp:
            messagebox.showerror(_t("Erreur", "Error"),
                                 _t("Sélectionnez une source.", "Select a source."))
            return

        same_folder  = bool(self._pp_same_folder.get())
        use_subfolder = bool(self._pp_subfolder.get())
        base_out = (inp if os.path.isdir(inp) else os.path.dirname(inp)) if same_folder \
                   else self._pp_output_var.get().strip()
        if not base_out:
            messagebox.showerror(_t("Erreur", "Error"),
                                 _t("Sélectionnez un dossier de sortie.", "Select an output folder."))
            return
        output_dir = os.path.join(base_out, "postproc") if use_subfolder else base_out

        _IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}
        if os.path.isfile(inp):
            ext_low = os.path.splitext(inp)[1].lower()
            input_paths = [inp] if ext_low in _IMG_EXT else []
        elif os.path.isdir(inp):
            input_paths = sorted(
                [os.path.join(inp, fn) for fn in os.listdir(inp)
                 if os.path.splitext(fn)[1].lower() in _IMG_EXT],
                key=lambda p: _natural_sort_key(os.path.basename(p))
            )
        else:
            messagebox.showerror(_t("Erreur", "Error"), _t("Source introuvable.", "Source not found."))
            return

        if not input_paths:
            messagebox.showerror(_t("Erreur", "Error"),
                                 _t("Aucune image dans la source.", "No images in source."))
            return

        _any_enabled = any(self._pp_settings.get(f"{k}_enabled", False)
                           for k in ("tf", "ud", "color", "resize", "sharpen"))
        if not _any_enabled:
            messagebox.showwarning(_t("Avertissement", "Warning"),
                                   _t("Activez au moins un outil de la chaîne.",
                                      "Enable at least one tool in the chain."))
            return

        use_serialize = bool(self._pp_serialize.get())
        _ser_raw = self._pp_serialize_start.get().strip()
        serialize_start = int(_ser_raw) if _ser_raw.isdigit() else 0

        self.settings.set("pp_last_input", inp)
        if not same_folder:
            self.settings.set("pp_last_output", base_out)
        self.settings.set("pp_same_folder", same_folder)
        self.settings.set("pp_subfolder", use_subfolder)
        self.settings.set("pp_serialize", use_serialize)
        self.settings.set("pp_serialize_start", str(serialize_start))

        _existing = getattr(self, "_pp_thread", None)
        if _existing is not None and _existing.is_alive():
            messagebox.showwarning(
                _t("En cours", "In progress"),
                _t("Un traitement est déjà en cours.", "Processing is already running."))
            return

        self.widgets["log_pp"].delete("1.0", "end")
        self.widgets["pp_prog"].set(0)
        self.widgets["pp_prog_pct"].configure(text="0%")
        self.widgets["pp_run_btn"].configure(
            state="disabled", text=_t("⏳ En cours…", "⏳ Running…"))
        self.widgets["pp_stop_btn"].configure(state="normal")

        self._pp_stop_flag = threading.Event()

        def _cb(msg: str):
            self._ui_update(self._pp_log_append, msg)

        import copy as _copy
        settings_snap = _copy.deepcopy(self._pp_settings)

        self._pp_thread = threading.Thread(
            target=self._pp_run_batch,
            args=(input_paths, output_dir, use_serialize, serialize_start, settings_snap, _cb),
            daemon=True
        )
        self._pp_thread.start()

    def _pp_request_stop(self):
        if hasattr(self, "_pp_stop_flag"):
            self._pp_stop_flag.set()
        if "pp_stop_btn" in self.widgets:
            self.widgets["pp_stop_btn"].configure(
                state="disabled", text=_t("⏳ Arrêt…", "⏳ Stopping…"))

    def _pp_log_append(self, msg: str):
        if "log_pp" in self.widgets:
            self.widgets["log_pp"].insert("end", msg + "\n")
            self.widgets["log_pp"].see("end")

    # ── Post Processing : worker thread ──────────────────────────────────────

    def _pp_run_batch(self, input_paths, output_dir, use_serialize,
                       serialize_start, settings, callback):
        import time as _time
        _ensure_pil()
        _ensure_numpy()

        tf_en  = bool(settings.get("tf_enabled",      False))
        ud_en  = bool(settings.get("ud_enabled",      False))
        col_en = bool(settings.get("color_enabled",   False))
        rsz_en = bool(settings.get("resize_enabled",  False))
        shp_en = bool(settings.get("sharpen_enabled", False))

        use_neural = tf_en or ud_en
        use_pil    = col_en or rsz_en or shp_en

        total = len(input_paths)
        _pad  = max(5, len(str(total + serialize_start)))

        out_paths = []
        for i, p in enumerate(input_paths):
            fname = os.path.basename(p)
            _, ext = os.path.splitext(fname)
            ext = ext or ".png"
            if use_serialize:
                out_paths.append(os.path.join(output_dir,
                                              f"{serialize_start + i:0{_pad}d}{ext}"))
            else:
                out_paths.append(os.path.join(output_dir, fname))

        os.makedirs(output_dir, exist_ok=True)

        def _set_prog(v):
            v = min(1.0, max(0.0, v))
            self._ui_update(self.widgets["pp_prog"].set, v)
            self._ui_update(self.widgets["pp_prog_pct"].configure,
                            text=f"{int(v * 100)}%")

        # ── Init PostProcSession ──────────────────────────────────────────────
        _pp = None
        _pp_started = False
        _tf_active  = False
        _ud_active  = False
        _pp_tf_lat  = 0
        _pp_ud_lat  = 0

        if use_neural:
            _venv_py = _find_torch_python()
            if not _venv_py:
                callback(_t("[AVERT.] Python torch introuvable — TF/UD désactivés.",
                            "[WARN] Python with torch not found — TF/UD disabled."))
                tf_en = ud_en = use_neural = False
            else:
                from src.core.post_proc_session import PostProcSession
                _pp = PostProcSession(venv_py=_venv_py, log=callback)
                _pp_started = _pp.start()

                if ud_en and _pp_started:
                    try:
                        _pp_ud_lat = _pp.init_ud(
                            mode=settings.get("ud_mode",      "classic"),
                            strength=float(settings.get("ud_strength", 0.35)),
                            window=int(settings.get("ud_window",       5)),
                            precision=settings.get("ud_precision",     "float32"),
                        )
                        _ud_active = True
                        callback(_t(
                            f"[Undistort] Activé — mode={settings.get('ud_mode')} "
                            f"str={settings.get('ud_strength'):.2f} latence={_pp_ud_lat}",
                            f"[Undistort] Enabled — mode={settings.get('ud_mode')} "
                            f"str={settings.get('ud_strength'):.2f} latency={_pp_ud_lat}"))
                    except Exception as e:
                        callback(f"[Undistort] Init échoué : {e}")

                if tf_en and _pp_started:
                    try:
                        _pp_tf_lat = _pp.init_tf(
                            mode=settings.get("tf_mode",      "classic"),
                            strength=float(settings.get("tf_strength", 0.5)),
                            window=int(settings.get("tf_window",       7)),
                            precision=settings.get("tf_precision",     "float32"),
                        )
                        _tf_active = True
                        callback(_t(
                            f"[TemporalFix] Activé — mode={settings.get('tf_mode')} "
                            f"str={settings.get('tf_strength'):.2f} latence={_pp_tf_lat}",
                            f"[TemporalFix] Enabled — mode={settings.get('tf_mode')} "
                            f"str={settings.get('tf_strength'):.2f} latency={_pp_tf_lat}"))
                    except Exception as e:
                        callback(f"[TemporalFix] Init échoué : {e}")

        errors = []
        success = 0
        _batch_start = _time.monotonic()

        # ── Phase 1 : copie + push TF/UD ────────────────────────────────────
        _ph1_weight = 0.6 if use_neural else (0.85 if use_pil else 1.0)
        callback(_t(f"Phase 1/{'3' if (use_neural and use_pil) else ('2' if (use_neural or use_pil) else '1')} — copie ({total} images)…",
                    f"Phase 1/{'3' if (use_neural and use_pil) else ('2' if (use_neural or use_pil) else '1')} — copy ({total} images)…"))

        for i, (in_path, out_path) in enumerate(zip(input_paths, out_paths)):
            if self._pp_stop_flag.is_set():
                callback(_t("Arrêt demandé.", "Stop requested.")); break
            fname = os.path.basename(in_path)
            callback(f"[{i+1}/{total}] {fname}")
            try:
                img = Image.open(in_path).convert("RGB")
                img.save(out_path)
                if _ud_active:
                    try:
                        _pp.push_ud(out_path)
                    except Exception as e:
                        callback(f"  [Undistort] {e}")
                if _tf_active:
                    try:
                        _pp.push_tf(out_path)
                    except Exception as e:
                        callback(f"  [TemporalFix] {e}")
                success += 1
            except Exception as e:
                errors.append(f"{fname}: {e}")
            _set_prog((i + 1) / total * _ph1_weight)

        # ── Phase 2 : flush TF/UD ────────────────────────────────────────────
        try:
            if _pp_started:
                if _ud_active and _pp_ud_lat > 0:
                    callback(_t("[Undistort] Flush des dernières frames…",
                                "[Undistort] Flushing remaining frames…"))
                    try:
                        _pp.flush_ud()
                    except Exception as e:
                        callback(f"[Undistort] Erreur flush : {e}")
                if _tf_active and _pp_tf_lat > 0:
                    callback(_t("[TemporalFix] Flush des dernières frames…",
                                "[TemporalFix] Flushing remaining frames…"))
                    try:
                        _pp.flush_tf()
                    except Exception as e:
                        callback(f"[TemporalFix] Erreur flush : {e}")
        finally:
            if _pp_started and _pp:
                _pp.stop()

        if use_neural:
            _set_prog(0.8)

        # ── Phase 3 : chaîne PIL ─────────────────────────────────────────────
        if use_pil and not self._pp_stop_flag.is_set():
            _pil_tools = ([_t("couleur","color")] if col_en else []) + \
                         ([_t("resize","resize")]  if rsz_en else []) + \
                         ([_t("netteté","sharpen")] if shp_en else [])
            callback(_t(f"Phase PIL — {', '.join(_pil_tools)}…",
                        f"PIL phase — {', '.join(_pil_tools)}…"))
            _pil_start  = 0.8 if use_neural else 0.0
            _pil_weight = 0.2 if use_neural else _ph1_weight

            for i, (in_path, out_path) in enumerate(zip(input_paths, out_paths)):
                if self._pp_stop_flag.is_set(): break
                if not os.path.isfile(out_path): continue
                try:
                    pil_img = Image.open(out_path).convert("RGB")
                    pil_img = self._pp_apply_pil_chain(pil_img, settings)
                    pil_img.save(out_path)
                    self._pp_update_preview(in_path, out_path)
                except Exception as e:
                    callback(f"  [PIL] {os.path.basename(out_path)}: {e}")
                _set_prog(_pil_start + (i + 1) / total * _pil_weight)
        elif input_paths and out_paths and os.path.isfile(out_paths[-1]):
            self._pp_update_preview(input_paths[-1], out_paths[-1])

        _set_prog(1.0)
        elapsed = _time.monotonic() - _batch_start
        callback(_t(f"Terminé en {elapsed:.1f}s — {success}/{total} images",
                    f"Done in {elapsed:.1f}s — {success}/{total} images") +
                 (f" | {len(errors)} erreur(s)" if errors else ""))
        for err in errors[:10]:
            callback(f"  ERR: {err}")

        self._ui_update(self.widgets["pp_run_btn"].configure, state="normal",
                        text=_t("▶ Lancer Post Processing", "▶ Run Post Processing"))
        self._ui_update(self.widgets["pp_stop_btn"].configure, state="disabled")

    # ── Post Processing : chaîne PIL ─────────────────────────────────────────

    @staticmethod
    def _pp_apply_pil_chain(pil_img, settings: dict):
        """Apply PIL color/resize/sharpen chain. Returns modified PIL Image."""
        _ensure_pil()
        _ensure_numpy()
        from PIL import ImageEnhance, ImageFilter

        # 1. Color correction
        if settings.get("color_enabled"):
            b = float(settings.get("brightness", 1.0))
            c = float(settings.get("contrast",   1.0))
            s = float(settings.get("saturation", 1.0))
            g = float(settings.get("gamma",      1.0))
            if abs(b - 1.0) > 1e-3:
                pil_img = ImageEnhance.Brightness(pil_img).enhance(b)
            if abs(c - 1.0) > 1e-3:
                pil_img = ImageEnhance.Contrast(pil_img).enhance(c)
            if abs(s - 1.0) > 1e-3:
                pil_img = ImageEnhance.Color(pil_img).enhance(s)
            if abs(g - 1.0) > 1e-3:
                arr = np.array(pil_img, dtype=np.float32) / 255.0
                arr = np.clip(np.power(arr, 1.0 / g), 0.0, 1.0)
                pil_img = Image.fromarray((arr * 255.0).astype(np.uint8))

        # 2. Resize
        if settings.get("resize_enabled"):
            _RESAMPLE_MAP = {
                "LANCZOS":  Image.Resampling.LANCZOS,
                "BICUBIC":  Image.Resampling.BICUBIC,
                "BILINEAR": Image.Resampling.BILINEAR,
                "NEAREST":  Image.Resampling.NEAREST,
            }
            method = _RESAMPLE_MAP.get(settings.get("resize_method", "LANCZOS"), Image.Resampling.LANCZOS)
            w, h = pil_img.size
            nw, nh = w, h
            resize_mode = settings.get("resize_mode", "percent")
            if resize_mode == "percent":
                scale_pct = int(settings.get("resize_scale", 100))
                nw = max(1, round(w * scale_pct / 100))
                nh = max(1, round(h * scale_pct / 100))
            elif resize_mode == "multiplier":
                mult = float(settings.get("resize_multiplier", 1.0))
                nw = max(1, round(w * mult))
                nh = max(1, round(h * mult))
            elif resize_mode == "pixels":
                tw = int(settings.get("resize_width", 0))
                th = int(settings.get("resize_height", 0))
                if tw > 0 and th > 0:
                    if settings.get("resize_aspect", True):
                        ratio = min(tw / w, th / h)
                        nw = max(1, round(w * ratio))
                        nh = max(1, round(h * ratio))
                    else:
                        nw, nh = tw, th
                elif tw > 0:
                    nw = tw
                    nh = max(1, round(h * tw / w))
                elif th > 0:
                    nh = th
                    nw = max(1, round(w * th / h))
            if (nw, nh) != (w, h):
                pil_img = pil_img.resize((nw, nh), method)

        # 3. Sharpen (UnsharpMask)
        if settings.get("sharpen_enabled"):
            strength  = float(settings.get("sharpen_strength",  1.0))
            radius    = float(settings.get("sharpen_radius",    1.5))
            threshold = int(settings.get("sharpen_threshold",   3))
            percent   = max(0, int(strength * 100))
            pil_img = pil_img.filter(
                ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

        return pil_img

    # ── Post Processing : preview ─────────────────────────────────────────────

    def _pp_update_preview(self, in_path: str, out_path: str):
        _ensure_pil()
        self._pp_last_preview_paths = (in_path, out_path)
        refs = []
        for path, key in [(in_path, "pp_prev_in"), (out_path, "pp_prev_out")]:
            try:
                lbl = self.widgets[key]
                w = lbl.winfo_width()
                h = lbl.winfo_height()
                if w < 10: w = 480
                if h < 10: h = 220
                img = Image.open(path).convert("RGB")
                img.thumbnail((w, h), Image.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                       size=(img.width, img.height))
                refs.append(ctk_img)
                self._ui_update(lbl.configure, image=ctk_img, text="")
            except Exception:
                pass
        if refs:
            self._pp_preview_refs = refs
