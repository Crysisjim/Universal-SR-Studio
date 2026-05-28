import customtkinter as ctk
from tkinter import filedialog, messagebox
import sys
import os
import re
import time
import datetime
import threading
import subprocess
import platform
import shutil
import yaml
import ctypes
import queue  # <--- IMPORT CRITIQUE POUR LA STABILITÉ

# Stubs for legacy lazy-import calls — yaml is already imported above.
def _ensure_yaml():
    pass
def _ensure_toml():
    pass

# winreg imported conditionally inside functions that need it

try:
    import tomllib
except ImportError:
    import toml as tomllib

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def _t(fr: str, en: str) -> str:
    """Return FR or EN string based on active language."""
    try:
        from src.core.translations import get_translator
        tr = get_translator()
        if tr and getattr(tr, 'language', 'fr') == 'en':
            return en
    except Exception:
        pass
    return fr

# ─── Windows Taskbar Progress (pure ctypes COM VTable, no comtypes) ────────────────
# Adapted from proven working implementation (PyAudioCodingTools)

class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8)
    ]

TBPF_NOPROGRESS = 0
TBPF_INDETERMINATE = 1
TBPF_NORMAL = 2
TBPF_ERROR = 4
TBPF_PAUSED = 8

_CLSID_TaskbarList = "{56FDF344-FD6D-11d0-958A-006097C9A090}"
_IID_ITaskbarList3 = "{EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF}"

class _TaskbarController:
    """Windows taskbar progress controller using pure ctypes COM VTable calls."""
    def __init__(self):
        self.ptr = None
        self.hwnd = 0
        if sys.platform != "win32":
            return
        try:
            ctypes.windll.ole32.CoInitialize(0)
            clsid = _GUID()
            iid = _GUID()
            ctypes.windll.ole32.CLSIDFromString(ctypes.c_wchar_p(_CLSID_TaskbarList), ctypes.byref(clsid))
            ctypes.windll.ole32.CLSIDFromString(ctypes.c_wchar_p(_IID_ITaskbarList3), ctypes.byref(iid))
            self.ptr = ctypes.c_void_p()
            ctypes.windll.ole32.CoCreateInstance(
                ctypes.byref(clsid), 0, 1,  # CLSCTX_INPROC_SERVER
                ctypes.byref(iid), ctypes.byref(self.ptr)
            )
        except Exception:
            self.ptr = None

    def set_hwnd(self, hwnd):
        """Set the window handle. Call with GetParent(winfo_id()) or winfo_id()."""
        self.hwnd = hwnd

    def _call_method(self, method_index, *args):
        """Call a COM method by VTable index."""
        if not self.ptr or not self.hwnd:
            return
        try:
            vtable = ctypes.cast(self.ptr, ctypes.POINTER(ctypes.c_void_p)).contents
            func_ptr = ctypes.cast(
                vtable.value + (method_index * ctypes.sizeof(ctypes.c_void_p)),
                ctypes.POINTER(ctypes.c_void_p)
            ).contents
            arg_types = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_uint64 for _ in args]
            functype = ctypes.WINFUNCTYPE(ctypes.c_long, *arg_types)
            func = functype(func_ptr.value)
            func(self.ptr, ctypes.c_void_p(self.hwnd), *args)
        except Exception:
            pass

    def set_progress(self, current, total):
        """SetProgressValue — VTable index 9."""
        self._call_method(9, int(current), int(total))

    def set_state(self, state_flag):
        """SetProgressState — VTable index 10."""
        self._call_method(10, state_flag)

_taskbar_ctrl = None

def _init_taskbar_controller(root_widget):
    """Initialize the taskbar controller with the app's root window."""
    global _taskbar_ctrl
    if sys.platform != "win32" or _taskbar_ctrl is not None:
        return
    try:
        _taskbar_ctrl = _TaskbarController()
        hwnd = ctypes.windll.user32.GetParent(root_widget.winfo_id())
        if not hwnd:
            hwnd = root_widget.winfo_id()
        _taskbar_ctrl.set_hwnd(hwnd)
    except Exception:
        _taskbar_ctrl = None

def set_taskbar_progress(hwnd: int, progress: float, state: str = "normal"):
    """
    Set Windows taskbar progress bar.
    progress: 0.0 to 1.0
    state: 'normal'(green), 'paused'(yellow), 'error'(red), 'indeterminate', 'none'
    """
    global _taskbar_ctrl
    if not _taskbar_ctrl:
        return
    try:
        state_map = {"none": TBPF_NOPROGRESS, "indeterminate": TBPF_INDETERMINATE,
                     "normal": TBPF_NORMAL, "error": TBPF_ERROR, "paused": TBPF_PAUSED}
        flag = state_map.get(state, TBPF_NORMAL)
        _taskbar_ctrl.set_state(flag)
        if state not in ("none", "indeterminate"):
            _taskbar_ctrl.set_progress(int(progress * 1000), 1000)
    except Exception:
        pass

# PERF-02: Use nvidia-ml-py for GPU polling instead of nvidia-smi subprocess
NVML_AVAILABLE = False
_nvml_handle = None
_nvml = None  # Module reference

try:
    # Modern package (no deprecation warning)
    import nvidia_smi as _nvml
    _nvml.nvmlInit()
    _nvml_handle = _nvml.nvmlDeviceGetHandleByIndex(0)
    NVML_AVAILABLE = True
except Exception:
    try:
        # Deprecated pynvml — suppress FutureWarning at import time
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
            warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
            import pynvml as _nvml  # noqa: F811
        _nvml.nvmlInit()
        _nvml_handle = _nvml.nvmlDeviceGetHandleByIndex(0)
        NVML_AVAILABLE = True
    except Exception:
        pass

from src.core.runner import TrainingRunner
from src.core.settings import SettingsManager
from src.ui.components.tooltip import ToolTip

class RunTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.runner = TrainingRunner()
        self.settings = SettingsManager()
        self.entries_dict = {} 
        
        # --- SYSTEME DE FILE D'ATTENTE (ANTI-FREEZE) ---
        self.log_queue = queue.Queue()
        self.check_log_queue()
        
        self.start_time = None
        self.timer_running = False
        self.target_iters = 100000
        self.pipeline_mode = None  # Set by ConfigTab for PSNR→GAN pipeline 
        self.current_iter = 0
        self.accumulate = 1 
        
        self.best_psnr = 0.0
        self.best_psnr_iter = 0
        self.best_ssim = 0.0
        self.best_ssim_iter = 0
        
        self.cpu_model_name = self.get_cpu_name()
        
        self.user_home = os.path.expanduser("~")
        self.base_engine_path = os.path.join(self.user_home, "IA_Engine")

        # --- GRID LAYOUT ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1) 

        # 1. TOP ZONE (3 COLONNES : GPU | FICHIERS | CPU)
        self.frame_top = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_top.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.frame_top.grid_columnconfigure(1, weight=1)

        # > COLONNE GAUCHE : GPU
        self.frame_gpu = ctk.CTkFrame(self.frame_top, fg_color=("#E8E8E8", "#181818"), border_width=2, border_color=("#AAAAAA", "#333"), width=220)
        self.frame_gpu.grid(row=0, column=0, sticky="ns", padx=(0, 5))
        self.frame_gpu.grid_propagate(False) 
        self.setup_gpu_monitor()
        ToolTip(self.frame_gpu, _t("Moniteur GPU.\nSurveillez la VRAM pour éviter les crashs (OOM).", "GPU Monitor.\nWatch VRAM to avoid crashes (OOM)."))

        # > COLONNE MILIEU : FICHIERS (Auto-Config)
        self.frame_files = ctk.CTkFrame(self.frame_top, fg_color=("#E8E8E8", "#2B2B2B"))
        self.frame_files.grid(row=0, column=1, sticky="nsew", padx=5)
        self.setup_file_controls()

        # > COLONNE DROITE : CPU
        self.frame_cpu = ctk.CTkFrame(self.frame_top, fg_color=("#E8E8E8", "#181818"), border_width=2, border_color=("#AAAAAA", "#333"), width=220)
        self.frame_cpu.grid(row=0, column=2, sticky="ns", padx=(5, 0))
        self.frame_cpu.grid_propagate(False)
        self.setup_cpu_monitor()
        ToolTip(self.frame_cpu, _t("Moniteur Système.\nCharge CPU et RAM.", "System Monitor.\nCPU and RAM load."))

        # 2. DASHBOARD (STATS)
        self.frame_stats = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_stats.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.setup_dashboard()

        # 3. INFO BAR
        self.frame_info = ctk.CTkFrame(self, fg_color=("#DEDEDE", "#222"), height=40)
        self.frame_info.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.setup_info_panel()

        # 4. CONSOLE
        self.frame_console = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_console.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 5))
        
        self.f_cons_head = ctk.CTkFrame(self.frame_console, fg_color="transparent", height=20)
        self.f_cons_head.pack(fill="x")
        self.lbl_progress = ctk.CTkLabel(self.f_cons_head, text=_t("Progression: 0%", "Progress: 0%"), font=("Roboto", 10))
        self.lbl_progress.pack(side="left", padx=5)
        
        self.btn_clear = ctk.CTkButton(self.f_cons_head, text="♻", width=30, height=20, fg_color="#444", command=self.clear_logs)
        self.btn_clear.pack(side="right", padx=5)
        ToolTip(self.btn_clear, _t("Effacer les logs de la console", "Clear console logs"))
        _EMOJI_FONT = ("Segoe UI Emoji", 11, "bold")
        self.btn_preview_val = ctk.CTkButton(self.f_cons_head, text="🔍 Validation",
                                              width=90, height=26, fg_color="#8e44ad",
                                              font=_EMOJI_FONT,
                                              command=self.show_validation_preview)
        self.btn_preview_val.pack(side="right", padx=5)
        ToolTip(self.btn_preview_val, _t("Afficher la derniere image de validation (LQ vs SR) avec comparaison", "Show last validation image (LQ vs SR) with comparison"))

        # Metrics graph button
        self.btn_metrics = ctk.CTkButton(self.f_cons_head, text=_t("📊 Métriques", "📊 Metrics"),
                                          width=90, height=26, fg_color="#2980b9",
                                          font=_EMOJI_FONT,
                                          command=self._show_metrics_graph)
        self.btn_metrics.pack(side="right", padx=5)
        ToolTip(self.btn_metrics, _t("Afficher les courbes de Loss et PSNR en temps reel", "Show Loss and PSNR curves in real time"))

        # Unified servers button (Gallery + TensorBoard)
        self.btn_servers = ctk.CTkButton(self.f_cons_head, text=_t("📡 Serveurs", "📡 Servers"),
                                          width=90, height=26, fg_color="#16a085",
                                          font=_EMOJI_FONT,
                                          command=self._show_servers_panel)
        self.btn_servers.pack(side="right", padx=5)
        ToolTip(self.btn_servers, _t("Galerie images + TensorBoard : QR codes, liens et contrôles", "Image gallery + TensorBoard: QR codes, links and controls"))

        # LR Schedule button
        self.btn_lr_sched = ctk.CTkButton(self.f_cons_head, text="📉 LR",
                                            width=50, height=26, fg_color="#9b59b6",
                                            font=_EMOJI_FONT,
                                            command=self._show_lr_schedule)
        self.btn_lr_sched.pack(side="right", padx=2)
        ToolTip(self.btn_lr_sched, _t("Visualiser l'evolution du Learning Rate\nsur toute la duree de l'entrainement", "Visualize the Learning Rate schedule\nover the full training duration"))

        # Auto-resume + options
        f_opts = ctk.CTkFrame(self.frame_console, fg_color="transparent", height=25)
        f_opts.pack(fill="x", pady=(0, 2))
        import tkinter as tk
        self._auto_resume = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(f_opts, text=_t("Auto-resume (reprendre si crash)", "Auto-resume (restart on crash)"), variable=self._auto_resume,
                        height=20, font=("Roboto", 11)).pack(side="left", padx=5)
        ToolTip(f_opts.winfo_children()[-1], _t("Si active, detecte le dernier checkpoint et reprend automatiquement l'entrainement", "If enabled, detects the last checkpoint and automatically resumes training"))

        self.progress_bar = ctk.CTkProgressBar(self.frame_console, height=10, progress_color="#2ecc71")
        self.progress_bar.pack(fill="x", pady=(0, 2))
        self.progress_bar.set(0)

        self.textbox_logs = ctk.CTkTextbox(self.frame_console, font=("Consolas", 10), activate_scrollbars=True, wrap="word")
        self.textbox_logs.pack(fill="both", expand=True)
        self.textbox_logs.configure(state="disabled")
        self.setup_log_tags()


        # 5. BOUTONS
        self.frame_btns = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_btns.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        
        self.btn_start = ctk.CTkButton(self.frame_btns, text=_t("▶  DÉMARRER", "▶  START"), fg_color="green", height=40, font=("Roboto", 14, "bold"), command=self.on_start)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_stop = ctk.CTkButton(self.frame_btns, text="⏹  STOP & SAVE", fg_color="#D35B58", height=40, font=("Roboto", 14, "bold"), state="disabled", command=self.on_stop)
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        self.chk_shutdown = ctk.CTkCheckBox(self.frame_btns, text=_t("Éteindre PC à la fin", "Shutdown PC when done"), font=("Arial", 11), fg_color="#e74c3c", hover_color="#c0392b")
        self.chk_shutdown.pack(side="right", padx=10)

        # (Live preview removed — OTF images not accessible during training)

        # Init Windows taskbar progress
        try:
            _init_taskbar_controller(self.winfo_toplevel())
        except Exception:
            pass

        self.after(500, self.auto_load_settings)
        self.poll_gpu_stats()
        self.poll_cpu_stats()

    # --- QUEUE MONITOR (LE FIX EST ICI) ---
    def check_log_queue(self):
        """Vérifie la file d'attente pour mettre à jour l'UI dans le thread principal"""
        try:
            while True:
                # Récupère tout ce qui est en attente sans bloquer
                text = self.log_queue.get_nowait()
                self._update_log_ui(text)
        except queue.Empty:
            pass
        # Rappel dans 100ms
        self.after(100, self.check_log_queue)

    def append_log(self, text):
        """Appelé par le thread d'entraînement (Runner) -> On met juste dans la queue"""
        self.log_queue.put(text)

    def _reset_redux_buf(self):
        """Reset the Redux multi-line buffer state (call at training start/stop/clear)."""
        self._redux_buf = ""
        self._redux_buf_active = False
        self._redux_buf_kind = ""
        self._redux_buf_lines = 0
        self._redux_buf_time = ""
        self._redux_buf_level = ""
        self._redux_draining = False  # True after iter flush — discard tail continuation lines
        # Validation tracking (reset between validation runs)
        self._val_images_seen = []
        self._val_total = 0
        self._val_in_progress = False
        # NeoSR validation buffer
        self._neosr_val_active = False
        self._neosr_val_name = ""
        self._neosr_val_time = ""
        self._neosr_val_psnr = None
        self._neosr_val_ssim = None
        self._neosr_val_img_count = 0

    # Verbose Redux init blocks to suppress entirely (big YAML dumps, system info repeats)
    _REDUX_SUPPRESS = (
        "Diff with default config",
        "Training statistics for",
    )
    # Verbose init records to suppress (loss/optimizer/scheduler creation details)
    _REDUX_SUPPRESS_PREFIX = (
        "Loss ",
        "Scheduler ",
    )

    def _normalize_redux_line(self, line: str):
        """Convert noisy Redux/rich-logging output into clean NeoSR-style lines."""
        if not hasattr(self, "_redux_buf"):
            self._reset_redux_buf()

        # Suppress raw tqdm validation progress lines; track image names for summary box
        # Format 1: "Test luminouswitches018: 71%  ---- 5/7 [0:00:05 < 0:00:02, 1 images/s]"
        if re.match(r'^Test\s+\S+:\s+\d+%', line):
            m_t = re.match(r'^Test\s+(\S+):\s+\d+%.*?(\d+)/(\d+)', line)
            if m_t:
                img_name = m_t.group(1)
                self._val_total = int(m_t.group(3))
                if img_name not in self._val_images_seen:
                    self._val_images_seen.append(img_name)
            return []
        # Format 2: plain validation tqdm bars "5/7 [0:00:05<0:00:02, 1.00 image/s]" or "images/s"
        if re.search(r'\d+/\d+\s+\[.*?image', line, re.IGNORECASE):
            m_total = re.search(r'\d+/(\d+)', line)
            if m_total:
                self._val_total = max(getattr(self, '_val_total', 0), int(m_total.group(1)))
            return []

        # --- NeoSR timestamp format: "DD-MM-YYYY HH:MM AM|PM | INFO: content" ---
        neosr_m = re.match(r"^(\d{2}-\d{2}-\d{4})\s+(\d{1,2}:\d{2})\s+(AM|PM)\s*\|\s*INFO:\s*(.*)$", line, re.IGNORECASE)
        if neosr_m:
            time_str = f"{neosr_m.group(2)} {neosr_m.group(3).upper()}"
            content = re.sub(r'[\x1b\033]\[[0-9;]*m', '', neosr_m.group(4)).strip()
            output = []
            C = "\x1b[36m"; G = "\x1b[32m"; Y = "\x1b[33m"; X = "\x1b[0m"
            # Flush pending NeoSR validation if we're starting a new timestamped line
            if not hasattr(self, '_neosr_val_active'):
                self._neosr_val_active = False
            if self._neosr_val_active:
                box = self._flush_neosr_val(self._neosr_val_time)
                if box:
                    output.append(box)
                self._neosr_val_active = False
            if not content:
                return output
            # Iter line
            if re.search(r"\[\s*epoch:\s*\d+", content):
                output.append(self._format_redux_record(time_str, content))
                return output
            # Validation start
            m_nval = re.match(r"Validation\s+(.+)", content)
            if m_nval:
                self._neosr_val_active = True
                self._neosr_val_time = time_str
                self._neosr_val_name = m_nval.group(1).strip()
                self._neosr_val_psnr = None
                self._neosr_val_ssim = None
                return output
            # Capture val image count from startup info
            m_nval_count = re.search(r"Number of val images/folders:\s*(\d+)", content)
            if m_nval_count:
                self._neosr_val_img_count = int(m_nval_count.group(1))

            # Colored status lines
            if "Saving models and training states" in content:
                output.append(f"{C}[{time_str}]{X} {G}INFO: {content}{X}")
                return output
            if "Resuming training from epoch" in content or "Start training from epoch" in content:
                output.append(f"{C}[{time_str}]{X} {G}INFO: {content}{X}")
                return output
            if "Interrupted" in content or "interrupted" in content:
                output.append(f"{C}[{time_str}]{X} {Y}INFO: {content}{X}")
                return output
            output.append(f"{C}[{time_str}]{X} {C}INFO:{X} {content}")
            return output

        # NeoSR validation continuation lines (psnr/ssim, no timestamp)
        if getattr(self, '_neosr_val_active', False):
            stripped = re.sub(r'[\x1b\033]\[[0-9;]*m', '', line)
            m_psnr = re.search(r'psnr:\s*([0-9.]+)', stripped, re.IGNORECASE)
            m_ssim = re.search(r'ssim:\s*([0-9.]+)', stripped, re.IGNORECASE)
            if m_psnr:
                self._neosr_val_psnr = m_psnr.group(1)
            if m_ssim:
                self._neosr_val_ssim = m_ssim.group(1)
            if self._neosr_val_psnr is not None and self._neosr_val_ssim is not None:
                box = self._flush_neosr_val(self._neosr_val_time)
                self._neosr_val_active = False
                return [box] if box else []
            return []

        prefix_re = re.compile(r"^\[(\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})\]\s+(\w+)\s+(.*)$")
        src_re = re.compile(r"\s+\w+\.py:\d+\s*$")

        m = prefix_re.match(line)
        if m:
            output_lines = []
            self._redux_draining = False  # New record — stop draining previous iter tail
            if self._redux_buf_active and self._redux_buf:
                flushed = self._flush_redux_buf()
                if flushed:
                    output_lines.append(flushed)

            time_str = m.group(2)
            level = m.group(3)
            content = src_re.sub("", m.group(4)).rstrip()

            # Suppress tqdm speed-only lines ("image/s image/s image/s..." spam from rich progress)
            if re.match(r'^(?:(?:images?|it|s)\/s\s*)+$', content.strip(), re.IGNORECASE):
                return output_lines  # flush any pending buffer but discard this line

            is_iter = bool(re.search(r"\[\s*epoch:\s*\d+", content)) or "[ iter:" in content or "[iter:" in content
            is_suppress = (
                any(content.startswith(p) for p in self._REDUX_SUPPRESS)
                or any(content.startswith(p) for p in self._REDUX_SUPPRESS_PREFIX)
            )

            # Immediate flush for validation result lines — don't buffer, render now
            # Otherwise the box only appears when the NEXT iter line arrives (minutes later)
            if not is_iter and not is_suppress and re.match(
                    r'Validation\s+.+?(?:#\s*)?psnr\s*[:=]\s*[0-9.]+', content, re.IGNORECASE):
                self._redux_buf_time = time_str
                self._redux_buf_level = level
                self._redux_buf = content
                self._redux_buf_lines = 1
                self._redux_buf_kind = "info"
                box = self._flush_redux_buf()
                if box:
                    output_lines.append(box)
                return output_lines

            self._redux_buf_time = time_str
            self._redux_buf_level = level
            self._redux_buf = content
            self._redux_buf_active = True
            self._redux_buf_lines = 1
            self._redux_buf_kind = "iter" if is_iter else ("suppress" if is_suppress else "info")
            return output_lines

        if self._redux_buf_active:
            # Discard continuation lines for suppressed records
            if self._redux_buf_kind == "suppress":
                return []
            src_re = re.compile(r"\s+\w+\.py:\d+\s*$")
            stripped = src_re.sub("", line.strip()).rstrip()
            if not stripped:
                return []
            self._redux_buf_lines += 1
            self._redux_buf += " " + stripped
            if self._redux_buf_kind == "iter":
                has_complete = bool(re.search(r"scale_g:\s*[0-9.eE+\-]+", self._redux_buf))
                if has_complete or self._redux_buf_lines > 14:
                    flushed = self._flush_redux_buf()
                    self._redux_draining = True  # absorb tail continuation lines until next timestamp
                    return [flushed] if flushed else []
                return []
            else:
                # Partial flush at 25 lines, then re-activate to avoid orphan lines
                if self._redux_buf_lines > 25:
                    saved_time = self._redux_buf_time
                    saved_level = self._redux_buf_level
                    flushed = self._flush_redux_buf()
                    self._redux_buf_active = True
                    self._redux_buf_kind = "info"
                    self._redux_buf_time = saved_time
                    self._redux_buf_level = saved_level
                    return [flushed] if flushed else []
                return []

        # Not in active buffer — drain tail lines of a just-flushed iter record
        if getattr(self, '_redux_draining', False):
            if not re.match(r"^\[[\d/]+\s+[\d:]+\]", line):
                return []  # No timestamp = still a tail continuation — discard silently

        return [line]

    def _flush_redux_buf(self) -> str:
        """Flush the current redux buffer to a single formatted string. Returns '' to suppress."""
        C = "\x1b[36m"; G = "\x1b[32m"; Y = "\x1b[33m"; X = "\x1b[0m"
        kind = self._redux_buf_kind
        time_str = self._redux_buf_time
        level = self._redux_buf_level
        raw = self._redux_buf

        self._redux_buf = ""
        self._redux_buf_active = False
        self._redux_buf_kind = ""
        self._redux_buf_lines = 0

        if kind == "suppress":
            return ""

        if kind == "iter":
            return self._format_redux_record(time_str, raw)

        # info record — collapse whitespace, shorten paths
        content = re.sub(r"\s+", " ", raw).strip()
        # Suppress tqdm speed-only fragments (e.g. "image/s image/s ..." from rich/tqdm)
        if re.match(r'^(?:(?:images?|it|s)\/s\s*)+$', content, re.IGNORECASE):
            return ""
        content = re.sub(r'[A-Za-z]:[\\\/](?:[^\\\/\n,]+[\\\/]){2,}([^\\\/\n,]+)', r'...\\\1', content)

        # Detect "Saving N validation images" — marks start of validation phase, extracts total
        m_saving = re.search(r'Saving\s+(\d+)\s+validation\s+images', content, re.IGNORECASE)
        if m_saving:
            self._val_in_progress = True
            self._val_total = int(m_saving.group(1))
            # Return a clean progress indicator; tqdm bars merged into this buffer are suppressed
            return f"{C}[{time_str}]{X} 🔍 Traitement de {self._val_total} images de validation..."

        # Suppress validation progress lines that came through with Redux prefix; track image names
        # e.g. "Test val_dataset: 100%  7/7 [0:00:00 < 0:00:00, 1 images/s]"
        if re.match(r'Test\s+\S+:\s+\d+%', content):
            m_t = re.match(r'Test\s+(\S+):\s+\d+%.*?(\d+)/(\d+)', content)
            if m_t:
                self._val_total = int(m_t.group(3))
            return ""
        # Suppress plain validation tqdm bars during validation phase
        if getattr(self, '_val_in_progress', False) and re.search(r'\d+/\d+\s+\[.*?image', content, re.IGNORECASE):
            m_total = re.search(r'\d+/(\d+)', content)
            if m_total:
                self._val_total = max(getattr(self, '_val_total', 0), int(m_total.group(1)))
            return ""

        # Format final validation result as a beautiful summary box.
        # traiNNer-redux format: "Validation val_xxx # psnr = X @ N iter # ssim = Y @ N iter"
        # NeoSR format:          "Validation val_xxx: psnr: X ssim: Y best: X @ N iter"
        m_val = re.match(r'Validation\s+(.+?)\s*(?:#\s*)?psnr\s*[:=]\s*([0-9.]+)', content, re.IGNORECASE)
        if m_val:
            val_name = m_val.group(1).strip().rstrip(":").strip()
            psnr_v = m_val.group(2)
            ssim_m = (re.search(r'#\s*ssim\s*[:=]\s*([0-9.]+)', content, re.IGNORECASE) or
                      re.search(r'\bssim\s*[:=]\s*([0-9.]+)', content, re.IGNORECASE))
            ssim_v = ssim_m.group(1) if ssim_m else None

            # Update PSNR card directly (bypass parse_metrics — ensures card always updates)
            try:
                pf = float(psnr_v)
                if 0 < pf < 100 and pf > self.best_psnr:
                    self.best_psnr = pf
                    self.best_psnr_iter = self.current_iter
                    self.card_psnr.configure(
                        text=f"{pf:.4f} dB\n(@ {self.best_psnr_iter})", text_color="#2ecc71")
            except Exception:
                pass

            # Update SSIM card directly (bypass parse_metrics — ensures card always updates)
            if ssim_v:
                try:
                    sf = float(ssim_v)
                    if 0 < sf <= 1.0 and sf > self.best_ssim:
                        self.best_ssim = sf
                        self.best_ssim_iter = self.current_iter
                        self.card_ssim.configure(
                            text=f"{sf:.4f}\n(@ {self.best_ssim_iter})", text_color="#1abc9c")
                except Exception:
                    pass

            images = list(getattr(self, '_val_images_seen', []))
            # Use the LARGER of tqdm-captured total and startup count ("Number of val images: N")
            real_total = max(
                getattr(self, '_val_total', 0),
                getattr(self, '_neosr_val_img_count', 0)
            )
            total = real_total or len(images)
            # Tqdm lines use \r overwrites → only some image names reach us as complete lines.
            # Always supplement from the val dataset folder so all images appear in the box.
            # This also fixes Redux where _val_total may be 0 (tqdm unit is "it", not "image").
            _ds_dir = (getattr(self, '_val_lq_dataset_dir', '')
                       or getattr(self, '_val_dataset_dir', ''))
            if _ds_dir and os.path.isdir(_ds_dir):
                _img_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
                try:
                    _all_stems = sorted(
                        (os.path.splitext(fn)[0] for fn in os.listdir(_ds_dir)
                         if os.path.splitext(fn)[1].lower() in _img_exts),
                        key=str.lower
                    )
                    # Update total if filesystem has more images than tqdm reported
                    if len(_all_stems) > total:
                        total = len(_all_stems)
                    _seen_set = set(images)
                    for _stem in _all_stems:
                        if _stem not in _seen_set:
                            images.append(_stem)
                            _seen_set.add(_stem)
                except Exception:
                    pass
            box = self._render_validation_box(val_name, psnr_v, ssim_v, images, total, time_str)

            # Reset validation tracking for next run
            self._val_images_seen = []
            self._val_total = 0
            self._val_in_progress = False

            # Auto-reload validation preview window if open and checkbox enabled
            if (getattr(self, '_val_win', None) and self._val_win.winfo_exists()
                    and getattr(self, '_val_auto_reload_var', None)
                    and self._val_auto_reload_var.get()):
                self.after(1500, self._auto_reload_val_preview)

            return box

        # Smart compression: traiNNer-redux greeting + system info → compact
        if "\U0001f680" in content and "traiNNer-redux" in content:
            gpu = re.search(r"Name:\s*(NVIDIA\s+GeForce\s+[^\d]*\d+[^\s]*(?:\s+Ti)?)", content)
            vram = re.search(r"Total VRAM:\s*([0-9.]+\s*GB)", content)
            pytorch = re.search(r"PyTorch:\s*([0-9.+a-z]+)", content)
            parts = ["\U0001f680 traiNNer-redux: good luck! \U0001f680"]
            if gpu:   parts.append(f"GPU: {gpu.group(1).strip()}")
            if vram:  parts.append(f"VRAM: {vram.group(1)}")
            if pytorch: parts.append(f"PyTorch: {pytorch.group(1)}")
            content = " | ".join(parts)

        # Smart compression: dataset build → keep first sentence + image count
        m_ds = re.match(r"(Dataset \w+ - \w+ Dataset is built\.)\s*Number of train images:\s*([\d,]+)", content)
        if m_ds:
            content = f"{m_ds.group(1)} ({m_ds.group(2)} images)"

        # Green highlight for resume line
        if "Resuming training from epoch" in content:
            return f"{C}[{time_str}]{X} {G}{level}: {content}{X}"

        # Green highlight for checkpoint save line
        if "Saving models and training states" in content:
            return f"{C}[{time_str}]{X} {G}{level}: {content}{X}"

        # Orange highlight for user interrupt line
        if "User interrupted" in content or "Preparing to save state" in content:
            return f"{C}[{time_str}]{X} {Y}{level}: {content}{X}"

        return f"{C}[{time_str}]{X} {C}{level}:{X} {content}"

    def _render_validation_box(self, val_name: str, psnr_v: str, ssim_v,
                                images: list, total: int, time_str: str) -> str:
        """Render a formatted validation summary box (similar to OTF degradation summary)."""
        C = "\x1b[36m"  # cyan  (borders, header)
        M = "\x1b[35m"  # magenta/purple (image names)
        G = "\x1b[32m"  # green  (checkmark, count)
        Y = "\x1b[33m"  # yellow (PSNR)
        B = "\x1b[34m"  # blue   (SSIM)
        X = "\x1b[0m"
        # --- Multi-column layout ---
        MAX_ROWS = 7  # images per column before wrapping to a new column
        n_imgs = len(images)
        n_cols = max(1, (n_imgs + MAX_ROWS - 1) // MAX_ROWS) if n_imgs else 1

        # Max name length (capped at 24) drives cell width
        max_nm_len = min(max((len(nm) for nm in images), default=10), 24)
        # cell = "  ✓  " (5) + name (max_nm_len) + "  " (2 trailing spaces)
        cell_w = max_nm_len + 7

        # Total inner width: at least 62 to fit header + metrics, or n_cols * cell_w
        header_vis = f"  VALIDATION — {val_name}  [{time_str}]"
        W = max(62, n_cols * cell_w, len(header_vis) + 2)

        def hline(left="╠", right="╣"):
            return f"{C}{left}{'═' * W}{right}{X}"

        def row(vis: str, col: str = None) -> str:
            """One box row. vis = plain text for width calc, col = ANSI-colored version."""
            if col is None:
                col = vis
            pad = max(0, W - len(vis))
            return f"{C}║{X}{col}{' ' * pad}{C}║{X}"

        # Header
        header_col = f"  {C}VALIDATION{X} — {C}{val_name}{X}  [{time_str}]"
        lines = [hline("╔", "╗"), row(header_vis, header_col), hline()]

        # Distribute images into columns (column-major order: fill col 0 then col 1 …)
        _cols = [images[i * MAX_ROWS:(i + 1) * MAX_ROWS] for i in range(n_cols)]
        n_rows = max((len(c) for c in _cols), default=0)

        for r_idx in range(n_rows):
            row_vis = ""
            row_col = ""
            for c_idx, col_items in enumerate(_cols):
                if r_idx < len(col_items):
                    nm = col_items[r_idx]
                    if len(nm) > max_nm_len:
                        nm = nm[:max_nm_len - 1] + "…"
                    row_vis += f"  ✓  {nm:<{max_nm_len}}  "
                    row_col += f"  {G}✓{X}  {M}{nm:<{max_nm_len}}{X}  "
                else:
                    row_vis += " " * cell_w
                    row_col += " " * cell_w
            lines.append(row(row_vis, row_col))

        lines.append(hline())

        # Metrics
        psnr_vis = f"  Best PSNR : {psnr_v} dB"
        psnr_col = f"  Best PSNR : {Y}{psnr_v} dB{X}"
        lines.append(row(psnr_vis, psnr_col))

        if ssim_v:
            ssim_vis = f"  Best SSIM : {ssim_v}"
            ssim_col = f"  Best SSIM : {B}{ssim_v}{X}"
            lines.append(row(ssim_vis, ssim_col))

        if total > 0 or n_imgs > 0:
            shown = n_imgs if n_imgs else total
            cnt_total = total if total > 0 else n_imgs
            count_vis = f"  Images traitées : {shown} / {cnt_total}"
            count_col = f"  Images traitées : {G}{shown} / {cnt_total}{X}"
            lines.append(row(count_vis, count_col))

        lines.append(hline("╚", "╝"))
        return "\n".join(lines)

    def _flush_neosr_val(self, time_str: str) -> str:
        """Render a NeoSR validation summary box from buffered psnr/ssim."""
        name = getattr(self, '_neosr_val_name', 'ValSet')
        psnr_v = getattr(self, '_neosr_val_psnr', None)
        ssim_v = getattr(self, '_neosr_val_ssim', None)
        if psnr_v is None:
            return ""
        try:
            pf = float(psnr_v)
            if pf > 0 and pf > self.best_psnr:
                self.best_psnr = pf
                self.best_psnr_iter = self.current_iter
                self.card_psnr.configure(text=f"{pf:.4f} dB\n(@ {self.best_psnr_iter})", text_color="#2ecc71")
        except Exception:
            pass
        if ssim_v:
            try:
                sf = float(ssim_v)
                if sf > 0 and sf > self.best_ssim:
                    self.best_ssim = sf
                    self.best_ssim_iter = self.current_iter
                    self.card_ssim.configure(text=f"{sf:.4f}\n(@ {self.best_ssim_iter})", text_color="#1abc9c")
            except Exception:
                pass
        img_count = getattr(self, '_neosr_val_img_count', 0)
        # Supplement image names from validation dataset folder
        images = []
        _ds_dir = (getattr(self, '_val_lq_dataset_dir', '')
                   or getattr(self, '_val_dataset_dir', ''))
        if _ds_dir and os.path.isdir(_ds_dir):
            _img_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
            try:
                images = sorted(
                    (os.path.splitext(fn)[0] for fn in os.listdir(_ds_dir)
                     if os.path.splitext(fn)[1].lower() in _img_exts),
                    key=str.lower
                )
            except Exception:
                pass
        total = max(img_count, len(images))
        return self._render_validation_box(name, psnr_v, ssim_v, images, total, time_str)

    def _format_redux_record(self, time_str: str, raw: str) -> str:
        """Take a (potentially wrapped+joined) Redux iter record and format like NeoSR's."""
        C  = "\x1b[36m"   # cyan (timestamp, VRAM)
        R  = "\x1b[31m"   # red (G-losses)
        P  = "\x1b[35m"   # purple (D-losses)
        G  = "\x1b[32m"   # green — matches iter card color
        Y  = "\x1b[33m"   # yellow — matches speed card color
        O  = "\x1b[38;5;208m"  # orange (epoch number)
        BR = "\x1b[38;5;130m"  # brown (learning rate)
        PK = "\x1b[95m"   # bright-magenta / pink (grad norms)
        X  = "\x1b[0m"

        ep = re.search(r"epoch:\s*(\d+)", raw)
        it = re.search(r"iter:\s*([\d,]+)", raw)
        lr = re.search(r"lr:\s*\(?\s*([0-9.eE+\-]+)\s*\)?", raw)
        perf = re.search(r"(?:perf|performance):\s*([0-9.]+)\s*it/s", raw)
        eta = re.search(r"eta:\s*([^\]]+?)\s*\]", raw)
        vram = re.search(r"VRAM:\s*([0-9.]+)\s*GB", raw)
        lg_total = re.search(r"l_g_total:\s*([0-9.eE+\-]+)", raw)
        # Capture all individual loss G components dynamically
        lg_components = re.findall(r"l_g_(\w+):\s*([0-9.eE+\-]+)", raw)
        # Discriminator losses (Redux GAN)
        ld_total = re.search(r"l_d_total:\s*([0-9.eE+\-]+)", raw)
        ld_real  = re.search(r"l_d_real:\s*([0-9.eE+\-]+)", raw)
        ld_fake  = re.search(r"l_d_fake:\s*([0-9.eE+\-]+)", raw)
        gn_g = re.search(r"grad_norm_g:\s*([0-9.eE+\-]+)", raw)
        gn_d = re.search(r"grad_norm_d:\s*([0-9.eE+\-]+)", raw)

        parts = [f"{C}[{time_str}]{X} {C}INFO:{X}"]
        if ep and it:
            iter_clean = it.group(1).rstrip(",")
            parts.append(f"[ epoch: {O}{ep.group(1):>3}{X} ] [ iter: {G}{iter_clean:>7}{X} ]")
        if perf:
            parts.append(f"[ perf: {Y}{perf.group(1)} it/s{X} ]")
        if lr:
            try:
                parts.append(f"[ lr: {BR}{float(lr.group(1)):.2e}{X} ]")
            except ValueError:
                parts.append(f"[ lr: {BR}{lr.group(1)}{X} ]")
        if eta:
            eta_str = eta.group(1).strip()
            parts.append(f"[ eta: {eta_str} ]")
        # Split G-losses: key losses on line 1, detailed perceptual on line 2
        # Key: total, gan, charbonnier, ldl1, ff  (short + most diagnostic)
        # Detail: perceptual_convN, and anything else with a long name
        _KEY_NAMES = {"total", "gan", "charbonnier", "ldl1", "ff", "focal", "real"}

        def _abbrev_g(name: str) -> str:
            _MAP = {"charbonnier": "charb", "ldl1": "ldl1", "ff": "ff",
                    "gan": "gan", "real": "real", "focal": "focal", "total": "total"}
            if name in _MAP:
                return _MAP[name]
            m = re.match(r"perceptual(?:_conv)?_?(\w+)", name)
            if m:
                return f"pc_{m.group(1)}"
            return name[:8] if len(name) > 8 else name

        loss_g_key = []
        loss_g_detail = []
        if lg_total:
            loss_g_key.append(f"total: {R}{lg_total.group(1)}{X}")
        for comp_name, comp_val in lg_components:
            if comp_name == "total":
                continue
            abbr = _abbrev_g(comp_name)
            piece = f"{abbr}: {R}{comp_val}{X}"
            if comp_name in _KEY_NAMES:
                loss_g_key.append(piece)
            else:
                loss_g_detail.append(piece)
        if loss_g_key:
            parts.append("[ " + " | ".join(loss_g_key) + " ]")

        # Line 2: detailed perceptual + D-losses + VRAM + grad norms
        line2_parts = []
        if loss_g_detail:
            line2_parts.append("[ " + " | ".join(loss_g_detail) + " ]")
        # Discriminator losses block (purple — matches LOSS D card color)
        loss_d_pieces = []
        if ld_total:
            loss_d_pieces.append(f"d_total: {P}{ld_total.group(1)}{X}")
        else:
            if ld_real:
                loss_d_pieces.append(f"d_real: {P}{ld_real.group(1)}{X}")
            if ld_fake:
                loss_d_pieces.append(f"d_fake: {P}{ld_fake.group(1)}{X}")
        if loss_d_pieces:
            line2_parts.append("[ " + " | ".join(loss_d_pieces) + " ]")
        if vram:
            line2_parts.append(f"[ VRAM: {C}{vram.group(1)} GB{X} ]")
        gn_pieces = []
        if gn_g:
            gn_pieces.append(f"|g|: {PK}{gn_g.group(1)}{X}")
        if gn_d:
            gn_pieces.append(f"|d|: {PK}{gn_d.group(1)}{X}")
        if gn_pieces:
            line2_parts.append("[ " + " | ".join(gn_pieces) + " ]")

        if line2_parts:
            return " ".join(parts) + " " + " ".join(line2_parts)
        return " ".join(parts)

    def _update_log_ui(self, text):
        """Mise à jour réelle de l'interface (Safe dans le MainThread)"""
        # Normalize Redux/rich-logging output line by line
        cleaned_chunks = []
        for raw_line in text.splitlines(keepends=True):
            # Strip any trailing newline for processing, re-add after
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                cleaned_chunks.append(raw_line)
                continue
            normalized = self._normalize_redux_line(line)
            for nline in normalized:
                if nline:
                    cleaned_chunks.append(nline + "\n")
        text_to_display = "".join(cleaned_chunks)

        self.textbox_logs.configure(state="normal")
        parts = re.split(r'(\x1b\[[0-9;]*m)', text_to_display)
        tags = []
        for p in parts:
            if p.startswith('\x1b['):
                code = p[2:-1]  # strip \x1b[ and trailing m
                if '38;5;208' in code:   tags = ["orange"]
                elif '38;5;130' in code: tags = ["brown"]
                elif code == '95':       tags = ["pink"]
                elif '32' in code:       tags = ["green"]
                elif '31' in code:       tags = ["red"]
                elif '33' in code:       tags = ["yellow"]
                elif '36' in code:       tags = ["cyan"]
                elif '35' in code:       tags = ["purple"]
                elif code == '0':        tags = []
            elif p:
                self.textbox_logs.insert("end", p, tuple(tags))
        
        self.textbox_logs.see("end")
        self.textbox_logs.configure(state="disabled")
        # Parsing des métriques (PSNR, Iteration, etc.) — pass the ORIGINAL text
        # so the parser can still see raw Redux multi-line records
        self.parse_metrics(text)

    # --- AUTO-PATCH ---
    def _load_config_any(self, config_path: str) -> dict:
        """Load a TOML or YAML config file, dispatching by extension."""
        ext = os.path.splitext(config_path)[1].lower()
        if ext in (".yml", ".yaml"):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        # Default to TOML
        try:
            import tomllib as _tl
        except ImportError:
            import toml as _tl
        with open(config_path, "rb") as f:
            if hasattr(_tl, "load"):
                return _tl.load(f)
            return _tl.loads(f.read().decode("utf-8"))

    def _show_training_estimation(self, config_path):
        """Parse config and show training duration estimation."""
        try:
            from src.core.compute_estimator import estimate_training_time, format_estimation_message
            cfg = self._load_config_any(config_path)

            # Extract relevant params
            arch = "omnisr_net"
            if "network_g" in cfg:
                ng = cfg["network_g"]
                if isinstance(ng, dict):
                    arch = ng.get("type", arch)
                elif isinstance(ng, str):
                    arch = ng

            scale = cfg.get("scale", 4)

            # Find batch + patch in datasets.train (NeoSR) or directly
            train_ds = cfg.get("datasets", {}).get("train", {})
            # Redux: batch_size_per_gpu / lq_size; NeoSR (TOML): batch_size / patch_size
            batch = (train_ds.get("batch_size_per_gpu")
                     or train_ds.get("batch_size")
                     or cfg.get("batch_size_per_gpu")
                     or cfg.get("batch_size", 4))
            # Patch size: NeoSR uses patch_size (LR), Redux uses lq_size (LR-side already)
            patch = (train_ds.get("lq_size")
                     or train_ds.get("patch_size")
                     or train_ds.get("gt_size")
                     or cfg.get("patch_size")
                     or cfg.get("gt_size", 64))
            # Accumulate: NeoSR -> accumulate, Redux -> accum_iter
            accumulate = (train_ds.get("accumulate")
                          or train_ds.get("accum_iter")
                          or cfg.get("accumulate", 1))
            total_iter = cfg.get("total_iter") or cfg.get("train", {}).get("total_iter", 100000)

            # If patch is HR-side (gt_size >= 128 typically), convert to LR-side
            if patch >= 128 and scale > 1:
                patch = patch // scale

            # AMP settings: NeoSR uses use_amp + bfloat16; Redux uses use_amp + amp_bf16
            _use_amp = bool(cfg.get("use_amp", False))
            _amp_bf16 = bool(
                cfg.get("amp_bf16", False)       # traiNNer-redux
                or cfg.get("bfloat16", False)    # neosr
            )

            est = estimate_training_time(
                architecture=arch,
                batch_size=batch,
                patch_size=patch,
                scale=scale,
                accumulate=accumulate,
                total_iter=total_iter,
                use_amp=_use_amp,
                amp_bf16=_amp_bf16,
            )

            # Count training images and estimate epochs
            dataset_line = ""
            try:
                gt_dirs = train_ds.get("dataroot_gt", [])
                if isinstance(gt_dirs, str): gt_dirs = [gt_dirs]
                _img_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                n_images = sum(
                    1 for d in gt_dirs if d and os.path.isdir(d)
                    for f in os.scandir(d)
                    if f.is_file() and os.path.splitext(f.name)[1].lower() in _img_ext
                )
                if n_images > 0:
                    eff_batch = int(batch) * max(1, int(accumulate))
                    total_epochs = max(1, round(total_iter * eff_batch / n_images))
                    dataset_line = f"{_t('Dataset train', 'Training dataset')} : {n_images} {_t('images', 'images')} → ~{total_epochs} {_t('epochs', 'epochs')}\n"
            except Exception:
                pass

            self.append_log("\n" + "=" * 50 + "\n")
            self.append_log(f"[{_t('Estimation', 'Estimation')}] {_t('Duree prevue de l entrainement', 'Estimated training duration')} :\n")
            if dataset_line:
                self.append_log(dataset_line)
            self.append_log(format_estimation_message(est) + "\n")
            self.append_log("=" * 50 + "\n\n")
        except Exception as e:
            self.append_log(f"[Estimation] Impossible : {e}\n")

    def apply_runtime_patch(self, script_path):
        if not os.path.exists(script_path): return
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            for line in lines:
                if "if sys.version_info < (3, 12)" in line:
                    new_lines.append(f"# PATCHED BY STUDIO: {line}")
                    modified = True
                elif "raise ValueError(msg)" in line:
                    new_lines.append(f"# PATCHED BY STUDIO: {line}")
                    modified = True
                else:
                    new_lines.append(line)
            
            if modified:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                self.append_log("[INFO] Patch appliqué : Vérification Python 3.12 désactivée.\n")
        except Exception as e:
            self.append_log(f"[WARN] Impossible d'appliquer le patch : {e}\n")

    # --- ACTIONS ---
    def external_start(self, config_path):
        if config_path:
            self.entries_dict["config_path"].delete(0, "end"); self.entries_dict["config_path"].insert(0, config_path)
            self.detect_engine_and_setup(config_path)
        self.on_start()

    def _check_auto_resume(self, config_path):
        """Check for existing checkpoints and offer to resume.

        For Redux: must set path.resume_state to the latest .state file.
        For NeoSR: same thing — checks training_states/*.state.
        """
        if not self._auto_resume.get():
            return config_path
        try:
            data = self._load_config_any(config_path)
            exp_name = data.get("name", "")
            if not exp_name:
                return config_path

            # Search for checkpoints — only in the current engine's experiments folder.
            # Filtering by extension prevents cross-engine contamination (e.g. a Redux
            # .state being detected for a NeoSR run sharing the same experiment name).
            home = os.path.expanduser("~")
            _cfg_ext = os.path.splitext(config_path)[1].lower()
            if _cfg_ext == ".toml":
                search_dirs = [os.path.join(home, "IA_Engine", "neosr", "experiments", exp_name)]
            elif _cfg_ext in (".yml", ".yaml"):
                search_dirs = [os.path.join(home, "IA_Engine", "traiNNer-redux", "experiments", exp_name)]
            else:
                search_dirs = [
                    os.path.join(home, "IA_Engine", "neosr", "experiments", exp_name),
                    os.path.join(home, "IA_Engine", "traiNNer-redux", "experiments", exp_name),
                ]
            for d in search_dirs:
                if not os.path.isdir(d):
                    continue
                # Look for .state files in training_states/ subfolder (Redux + NeoSR convention)
                state_dir = os.path.join(d, "training_states")
                if os.path.isdir(state_dir):
                    states = sorted(
                        [f for f in os.listdir(state_dir) if f.endswith(".state")],
                        key=lambda n: self._extract_iter_from_name(n),
                        reverse=True
                    )
                    if states:
                        latest_state = os.path.join(state_dir, states[0]).replace("\\", "/")
                        self.append_log(f"[Auto-Resume] {_t('State detecte', 'State found')}: {latest_state}\n")
                        # Write resume_state into the config
                        self._set_resume_state_in_config(config_path, latest_state)
                        self.append_log(f"[Auto-Resume] {_t('Config modifiee pour reprendre depuis ce state.', 'Config updated to resume from this state.')}\n")
                        return config_path

                # Fallback: check models dir for .pth or .safetensors
                models_dir = os.path.join(d, "models")
                if os.path.isdir(models_dir):
                    ckpts = sorted(
                        [f for f in os.listdir(models_dir)
                         if f.endswith(".pth") or f.endswith(".safetensors")],
                        key=lambda n: self._extract_iter_from_name(n),
                        reverse=True
                    )
                    if ckpts:
                        self.append_log(f"[Auto-Resume] {_t('Checkpoint detecte', 'Checkpoint found')}: {ckpts[0]} ({_t('pas de .state', 'no .state file')})\n")
                        self.append_log(f"[Auto-Resume] {_t('Aucun .state — l engine reprendra via le nom d experience.', 'No .state — engine will resume via experiment name.')}\n")
                        return config_path
        except Exception as e:
            self.append_log(f"[Auto-Resume] {_t('Erreur', 'Error')}: {e}\n")
        return config_path

    def _extract_iter_from_name(self, filename: str) -> int:
        """Extract the iteration number from a filename like 'net_g_10000.pth' or '5000.state'."""
        m = re.search(r"(\d+)", os.path.splitext(filename)[0])
        return int(m.group(1)) if m else 0

    def _set_resume_state_in_config(self, config_path: str, state_path: str):
        """Write path.resume_state = <state_path> into the config file."""
        ext = os.path.splitext(config_path)[1].lower()
        try:
            if ext in (".yml", ".yaml"):
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if "path" not in data or not isinstance(data["path"], dict):
                    data["path"] = {}
                data["path"]["resume_state"] = state_path
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            elif ext == ".toml":
                # NeoSR uses [path] section. We rewrite the whole config.
                try:
                    import tomllib as _tl
                except ImportError:
                    import toml as _tl
                with open(config_path, "rb") as f:
                    if hasattr(_tl, "load"):
                        data = _tl.load(f)
                    else:
                        data = _tl.loads(f.read().decode("utf-8"))
                if "path" not in data or not isinstance(data["path"], dict):
                    data["path"] = {}
                data["path"]["resume_state"] = state_path
                # Write back using tomli_w or toml
                try:
                    import tomli_w
                    with open(config_path, "wb") as f:
                        tomli_w.dump(data, f)
                except ImportError:
                    try:
                        import toml as _toml_w
                        with open(config_path, "w", encoding="utf-8") as f:
                            _toml_w.dump(data, f)
                    except ImportError:
                        self.append_log(_t("[Auto-Resume] Avertissement: ni tomli_w ni toml dispo, TOML non modifie. Le resume ne fonctionnera pas.\n",
                                           "[Auto-Resume] Warning: neither tomli_w nor toml available, TOML not modified. Resume will not work.\n"))
        except Exception as e:
            self.append_log(f"[Auto-Resume] {_t('Echec ecriture resume_state', 'Failed to write resume_state')}: {e}\n")

    def _pre_archive_exp_folder(self, config_path: str, script_path: str):
        """Pre-rename the existing experiment folder before starting any servers or training.

        traiNNer-redux does the same rename itself, but fails when an external process
        (old TensorBoard, Windows Explorer, etc.) holds a handle on the folder.
        By doing it here — before launching TB/gallery/training — we own the lock window.

        Only renames if:
        - config has resume_state == None / null (fresh start, not a resume)
        - the experiments/{name} folder exists under the detected engine root
        """
        import time as _time
        try:
            data = self._load_config_any(config_path)
        except Exception:
            return

        exp_name = data.get("name", "")
        if not exp_name:
            return

        # Determine if this is a resume — if resume_state is set, traiNNer won't rename
        path_section = data.get("path", {}) or {}
        resume_state = path_section.get("resume_state")
        if resume_state and str(resume_state).strip() not in ("", "~", "null", "None"):
            return  # it's a resume — traiNNer won't archive, we don't need to either

        # Locate engine root from script path
        engine_root = os.path.dirname(script_path) if script_path and os.path.isfile(script_path) else None
        if not engine_root:
            # Guess from script entry
            se = self.entries_dict.get("script_path")
            sp = se.get().strip() if se else ""
            engine_root = os.path.dirname(sp) if sp else None

        if not engine_root:
            return

        exp_folder = os.path.join(engine_root, "experiments", exp_name)
        if not os.path.isdir(exp_folder):
            return  # nothing to rename

        # Build the same archive name traiNNer would use
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = exp_folder + f"_archived_{ts}"

        self.append_log(f"[Pre-Archive] Dossier existant détecté: {exp_folder}\n")
        self.append_log(f"[Pre-Archive] Archivage → {os.path.basename(archived)}\n")

        # Try rename up to 4 times with increasing delays
        for attempt in range(4):
            try:
                os.rename(exp_folder, archived)
                self.append_log("[Pre-Archive] OK — traiNNer créera un nouveau dossier propre.\n")
                return
            except PermissionError as e:
                if attempt < 3:
                    wait_s = 0.5 * (attempt + 1)
                    self.append_log(
                        f"[Pre-Archive] Tentative {attempt+1}/4 échouée (fichier verrouillé). "
                        f"Nouvelle tentative dans {wait_s:.1f}s...\n"
                    )
                    _time.sleep(wait_s)
                else:
                    # All retries exhausted — warn the user
                    from tkinter import messagebox
                    messagebox.showwarning(
                        _t("Dossier verrouillé", "Folder locked"),
                        _t(f"Impossible de renommer le dossier existant :\n{exp_folder}\n\n"
                           "Erreur : " + str(e) + "\n\n"
                           "Solutions :\n"
                           "  • Fermez l'Explorateur Windows s'il affiche ce dossier\n"
                           "  • Attendez quelques secondes que TensorBoard libère les fichiers\n"
                           "  • Ou renommez le dossier manuellement avant de relancer\n\n"
                           "L'entraînement va quand même démarrer — traiNNer retentira le renommage.",
                           f"Cannot rename existing folder:\n{exp_folder}\n\n"
                           "Error: " + str(e) + "\n\n"
                           "Solutions:\n"
                           "  • Close Windows Explorer if it is showing this folder\n"
                           "  • Wait a few seconds for TensorBoard to release file handles\n"
                           "  • Or rename the folder manually before relaunching\n\n"
                           "Training will start anyway — traiNNer will retry the rename.")
                    )
            except OSError as e:
                self.append_log(f"[Pre-Archive] Erreur inattendue: {e}\n")
                return

    def _record_training_start(self, config_path, script_path):
        """Record training start in the history database."""
        try:
            from src.core.training_history import record_training_start
            cfg = self._load_config_any(config_path)

            arch = "unknown"
            if "network_g" in cfg:
                ng = cfg["network_g"]
                if isinstance(ng, dict):
                    arch = ng.get("type", arch)
                elif isinstance(ng, str):
                    arch = ng

            scale = cfg.get("scale", 4)
            total_iter = cfg.get("total_iter") or cfg.get("train", {}).get("total_iter", 0)
            train_ds = cfg.get("datasets", {}).get("train", {})
            batch = (train_ds.get("batch_size_per_gpu")
                     or train_ds.get("batch_size")
                     or cfg.get("batch_size_per_gpu")
                     or cfg.get("batch_size", 0))
            patch = (train_ds.get("lq_size")
                     or train_ds.get("patch_size")
                     or train_ds.get("gt_size")
                     or cfg.get("patch_size")
                     or cfg.get("gt_size", 0))
            ds_path = train_ds.get("dataroot_gt", "")
            if isinstance(ds_path, list):
                ds_path = ds_path[0] if ds_path else ""

            train_section = cfg.get("train", {})
            optim_g = train_section.get("optim_g", {})
            lr_str = ""
            optim_str = ""
            if isinstance(optim_g, dict):
                lr_str = str(optim_g.get("lr", ""))
                optim_str = optim_g.get("type", "")

            engine = "NeoSR" if "neosr" in script_path.lower() else "Redux"
            name = cfg.get("name", os.path.splitext(os.path.basename(config_path))[0])

            # Detect GPU name for benchmark
            gpu_name = ""
            try:
                from src.core.compute_estimator import detect_gpu
                gpu_name, _ = detect_gpu()
            except Exception:
                pass

            # GAN mode : détecte si discriminateur actif
            gan_mode = 0
            _has_netd = bool(cfg.get("network_d") or cfg.get("net_d"))
            _train_sec = cfg.get("train", {})
            _losses_cfg = _train_sec.get("losses", {})
            if isinstance(_losses_cfg, dict) and any("gan" in k.lower() for k in _losses_cfg):
                gan_mode = 1
            elif _has_netd:
                gan_mode = 1

            # Active losses (noms, virgule-séparée)
            losses_list = []
            if isinstance(_losses_cfg, dict):
                for _lk, _lv in _losses_cfg.items():
                    if isinstance(_lv, dict) and _lv.get("weight", 0):
                        losses_list.append(_lk)
                    elif _lv:
                        losses_list.append(_lk)
            elif isinstance(_losses_cfg, list):
                losses_list = [str(l) for l in _losses_cfg]
            losses_str = ",".join(losses_list) if losses_list else ""

            # Upsampler (network_g.upsampler ou similar)
            _ng = cfg.get("network_g", {})
            upsampler_str = ""
            if isinstance(_ng, dict):
                upsampler_str = _ng.get("upsampler", "") or _ng.get("upsample_mode", "") or ""

            self._history_row_id = record_training_start(
                name=name, engine=engine, architecture=arch, scale=scale,
                total_iter=total_iter, config_path=config_path,
                dataset_path=str(ds_path), batch_size=int(batch) if batch else 0,
                patch_size=int(patch) if patch else 0,
                lr=lr_str, optimizer=optim_str, gpu_name=gpu_name,
                gan_mode=gan_mode, losses=losses_str, upsampler=upsampler_str
            )
            self._speed_samples = []   # Reset speed accumulator for this run
            self._vram_peak_mb = 0     # Reset VRAM peak tracker
            self._power_samples = []   # Reset power accumulator
            # Persist visualization path for gallery auto-path in settings
            try:
                from src.core.settings import SettingsManager as _SM
                _sm = _SM()
                engine_root = os.path.dirname(script_path)
                viz_path = os.path.join(engine_root, "experiments", name, "visualization")
                _sm.set("gallery_auto_dir", viz_path)
            except Exception:
                pass
            self.append_log(f"[History] {_t('Training enregistre', 'Training recorded')} (ID #{self._history_row_id})\n")
        except Exception as e:
            self._history_row_id = 0
            self.append_log(f"[History] Erreur : {e}\n")

    def _log_otf_summary(self, config_path: str):
        """Log a formatted box showing all active OTF degradations."""
        try:
            cfg = self._load_config_any(config_path)
        except Exception:
            return

        ds_train = cfg.get("datasets", {}).get("train", {})

        # blur_prob2 is the Redux engine key for second blur probability
        _key_aliases = {"second_blur_prob": "blur_prob2"}

        def get_val(key, default=0.0):
            if key in cfg:
                return cfg[key]
            if key in cfg.get("degradations", {}):
                return cfg["degradations"][key]
            # Redux OTF top-level format stores kernel params in datasets.train
            v = ds_train.get(key)
            if v is not None:
                return v
            # Check engine-side alias (e.g. blur_prob2 for second_blur_prob)
            alias = _key_aliases.get(key)
            if alias and alias in cfg:
                return cfg[alias]
            # NeoSR TOML: first blur has no probability gate — always-on when
            # degradations section is present with blur_kernel_size configured.
            if key == "blur_prob" and cfg.get("degradations", {}).get("blur_kernel_size") is not None:
                return 1.0
            return default

        DEG_SLOTS = [
            ("blur_prob",                  _t("Flou 1", "Blur 1")),
            ("gaussian_noise_prob",        _t("Bruit Gaussien 1", "Gaussian Noise 1")),
            ("gray_noise_prob",            _t("Bruit Gris 1", "Gray Noise 1")),
            ("jpeg_prob",                  "JPEG 1"),
            ("second_blur_prob",           _t("Flou 2", "Blur 2")),
            ("gaussian_noise_prob2",       _t("Bruit Gaussien 2", "Gaussian Noise 2")),
            ("gray_noise_prob2",           _t("Bruit Gris 2", "Gray Noise 2")),
            ("final_sinc_prob",            _t("Sinc final", "Final Sinc")),
            ("posterize_prob",             "Posterize (custom)"),
            ("banding_prob",               "Banding (custom)"),
            ("aliasing_prob",              "Aliasing (custom)"),
            ("interlace_weave_prob",       "Interlace Weave (custom)"),
            ("interlace_flicker_prob",     "Interlace Flicker (custom)"),
            ("interlace_blend_prob",       "Interlace Blend (custom)"),
            ("film_grain_prob",            "Film Grain (custom)"),
            ("oversharp_prob",             "Oversharp (custom)"),
            ("scanlines_prob",             "Scanlines (custom)"),
        ]

        active, inactive = [], []
        for key, label in DEG_SLOTS:
            try:
                prob = float(get_val(key, 0.0))
            except (ValueError, TypeError):
                prob = 0.0
            if prob > 0.0:
                active.append(f"  ✓ {label} (p={prob:.2f})")
            else:
                inactive.append(f"  - {label}")

        total = len(DEG_SLOTS)
        n_active = len(active)
        title = f"  {_t('Degradations OTF actives', 'Active OTF Degradations')} : {n_active}/{total}  "
        width = max(len(title), max((len(s) for s in active + inactive), default=0)) + 2
        border = "═" * width

        lines = [
            f"╔{border}╗",
            f"║{title.ljust(width)}║",
            f"╠{border}╣",
        ]
        for s in active:
            lines.append(f"║{s.ljust(width)}║")
        if inactive:
            lines.append(f"║{'  ' + _t('(inactives)', '(inactive)'):{width}}║")
            for s in inactive:
                lines.append(f"║{s.ljust(width)}║")
        lines.append(f"╚{border}╝")

        self.append_log("\n".join(lines) + "\n")

    def _apply_custom_degradation_patches(self, config_path, script_path):
        """Read banding/posterize from config and patch the engine accordingly.

        For traiNNer-redux: msgspec strict validation rejects unknown keys, so we
        must STRIP custom degradation keys from the YAML and write a sanitized
        version. The runtime patch reads them from a side-channel JSON file.
        """
        try:
            cfg = self._load_config_any(config_path)
        except Exception:
            return

        # Look for banding/posterize at top level (NeoSR) or under degradations (Redux)
        def get_val(key, default=0.0):
            if key in cfg:
                return cfg[key]
            return cfg.get("degradations", {}).get(key, default)

        post_prob  = float(get_val("posterize_prob", 0.0))
        band_prob  = float(get_val("banding_prob", 0.0))
        chroma_prob = float(get_val("chroma_prob", 0.0))
        ca_prob    = float(get_val("ca_prob", 0.0))
        hal_prob   = float(get_val("halation_prob", 0.0))
        sp_prob    = float(get_val("salt_pepper_prob", 0.0))
        vhs_prob   = float(get_val("vhs_prob", 0.0))
        alias_prob  = float(get_val("aliasing_prob", 0.0))
        iw_prob     = float(get_val("interlace_weave_prob", 0.0))
        if_prob     = float(get_val("interlace_flicker_prob", 0.0))
        ib_prob     = float(get_val("interlace_blend_prob", 0.0))
        fg_prob     = float(get_val("film_grain_prob", 0.0))
        os_prob     = float(get_val("oversharp_prob", 0.0))
        sl_prob     = float(get_val("scanlines_prob", 0.0))

        has_post     = post_prob > 0.0
        has_band     = band_prob > 0.0
        has_chroma   = chroma_prob > 0.0
        has_ca       = ca_prob > 0.0
        has_hal      = hal_prob > 0.0
        has_sp       = sp_prob > 0.0
        has_vhs      = vhs_prob > 0.0
        has_alias    = alias_prob > 0.0
        has_interlace = any([iw_prob > 0.0, if_prob > 0.0, ib_prob > 0.0])
        has_fg       = fg_prob > 0.0
        has_os       = os_prob > 0.0
        has_sl       = sl_prob > 0.0

        # Engine dir = parent of script (train.py)
        engine_dir = os.path.dirname(script_path)

        # Detect if this is traiNNer-redux (uses msgspec strict YAML)
        is_redux = (
            "trainner-redux" in script_path.lower()
            or "trainner_redux" in script_path.lower()
            or os.path.exists(os.path.join(engine_dir, "traiNNer", "utils", "config.py"))
        )

        # Always strip USR Studio metadata sections from Redux YAMLs (msgspec rejects unknown fields)
        if is_redux and config_path.lower().endswith((".yml", ".yaml")):
            try:
                self._strip_custom_keys_from_yaml(config_path, ["monitoring"])
            except Exception:
                pass

        if not any([has_post, has_band, has_chroma, has_ca, has_hal, has_sp, has_vhs,
                    has_alias, has_interlace, has_fg, has_os, has_sl]):
            return  # Nothing more to patch

        # Collect custom deg params for the side-channel file
        custom_params = {
            "posterize_prob": post_prob,
            "posterize_bits_range": list(get_val("posterize_bits_range", [3, 6])),
            "banding_prob": band_prob,
            "banding_levels_range": list(get_val("banding_levels_range", [16, 64])),
            "chroma_prob": chroma_prob,
            "ca_prob": ca_prob,
            "ca_shift_range": list(get_val("ca_shift_range", [1, 5])),
            "halation_prob": hal_prob,
            "halation_strength_range": list(get_val("halation_strength_range", [0.05, 0.3])),
            "salt_pepper_prob": sp_prob,
            "salt_pepper_amount_range": list(get_val("salt_pepper_amount_range", [0.001, 0.05])),
            "vhs_prob": vhs_prob,
            "vhs_strength_range": list(get_val("vhs_strength_range", [0.1, 0.5])),
            "aliasing_prob": alias_prob,
            "aliasing_scale_range": list(get_val("aliasing_scale_range", [0.5, 0.85])),
            "interlace_weave_prob": iw_prob,
            "interlace_weave_strength_range": list(get_val("interlace_weave_strength_range", [0.5, 1.0])),
            "interlace_flicker_prob": if_prob,
            "interlace_flicker_strength_range": list(get_val("interlace_flicker_strength_range", [0.1, 0.4])),
            "interlace_blend_prob": ib_prob,
            "interlace_blend_strength_range": list(get_val("interlace_blend_strength_range", [0.3, 1.0])),
            "film_grain_prob": fg_prob,
            "film_grain_strength_range": list(get_val("film_grain_strength_range", [0.03, 0.12])),
            "film_grain_size_range": list(get_val("film_grain_size_range", [1, 2])),
            "oversharp_prob": os_prob,
            "oversharp_strength_range": list(get_val("oversharp_strength_range", [0.5, 2.0])),
            "scanlines_prob": sl_prob,
            "scanlines_strength_range": list(get_val("scanlines_strength_range", [0.2, 0.5])),
            "scanlines_spacing_range": list(get_val("scanlines_spacing_range", [2, 4])),
        }

        # If Redux: strip custom deg keys from the YAML before launch (msgspec rejects unknown fields)
        if is_redux and config_path.lower().endswith((".yml", ".yaml")):
            try:
                self._strip_custom_keys_from_yaml(config_path, [
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
                ])
                self.append_log(f"[Custom Deg] {_t('Cles custom retirees du YAML pour msgspec compatibility.', 'Custom keys removed from YAML for msgspec compatibility.')}\n")
            except Exception as e:
                self.append_log(f"[Custom Deg] Avertissement strip YAML: {e}\n")

        # Write side-channel file that the runtime patch reads
        try:
            import json
            sidecar = os.path.join(engine_dir, "_usr_studio_custom_deg.json")
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(custom_params, f, indent=2)
            self.append_log(f"[Custom Deg] {_t('Sidecar ecrit', 'Sidecar written')}: {sidecar}\n")
        except Exception as e:
            self.append_log(f"[Custom Deg] Avertissement sidecar: {e}\n")

        try:
            from src.core.otf_custom_degradations import install_patches
            ok = install_patches(engine_dir, has_post, has_band,
                                 has_chroma=has_chroma, has_ca=has_ca,
                                 has_halation=has_hal, has_salt_pepper=has_sp,
                                 has_vhs=has_vhs, has_aliasing=has_alias,
                                 has_interlace=has_interlace, has_film_grain=has_fg,
                                 has_oversharp=has_os, has_scanlines=has_sl)
            if ok:
                msg = []
                if has_post:   msg.append(f"posterize p={post_prob:.2f}")
                if has_band:   msg.append(f"banding p={band_prob:.2f}")
                if has_chroma: msg.append(f"chroma p={chroma_prob:.2f}")
                if has_ca:     msg.append(f"CA p={ca_prob:.2f}")
                if has_hal:    msg.append(f"halation p={hal_prob:.2f}")
                if has_sp:     msg.append(f"salt&pepper p={sp_prob:.2f}")
                if has_vhs:    msg.append(f"VHS p={vhs_prob:.2f}")
                if has_alias:     msg.append(f"aliasing p={alias_prob:.2f}")
                if iw_prob > 0:   msg.append(f"interlace-weave p={iw_prob:.2f}")
                if if_prob > 0:   msg.append(f"interlace-flicker p={if_prob:.2f}")
                if ib_prob > 0:   msg.append(f"interlace-blend p={ib_prob:.2f}")
                if has_fg:        msg.append(f"film-grain p={fg_prob:.2f}")
                if has_os:        msg.append(f"oversharp p={os_prob:.2f}")
                if has_sl:        msg.append(f"scanlines p={sl_prob:.2f}")
                self.append_log(f"[Custom Deg] Patch applique: {', '.join(msg)}\n")
        except Exception as e:
            self.append_log(f"[Custom Deg] Erreur: {e}\n")

    def _patch_monitoring_section(self, config_path: str):
        """Override auto_tensorboard and auto_ngrok in working config with Studio settings.

        Studio checkboxes are the master control. Values in the config file are overridden
        so runner.py (which reads the config directly) honours the Studio UI choices.
        """
        from src.core.settings import SettingsManager as _SM
        _sm = _SM()
        tb_auto = bool(_sm.get("tb_auto_start_with_training", False))
        ngrok_auto = bool(_sm.get("ngrok_auto_from_config", False))

        ext = os.path.splitext(config_path)[1].lower()
        try:
            if ext in (".yml", ".yaml"):
                import yaml
                with open(config_path, "r", encoding="utf-8") as _f:
                    data = yaml.safe_load(_f) or {}
                mon = data.get("monitoring")
                if isinstance(mon, dict):
                    if "auto_tensorboard" in mon:
                        mon["auto_tensorboard"] = tb_auto
                    if "auto_ngrok" in mon:
                        mon["auto_ngrok"] = ngrok_auto
                    with open(config_path, "w", encoding="utf-8") as _f:
                        yaml.safe_dump(data, _f, default_flow_style=False,
                                       sort_keys=False, allow_unicode=True)

            elif ext == ".toml":
                try:
                    import tomllib as _tl
                except ImportError:
                    try:
                        import tomli as _tl
                    except ImportError:
                        return  # no TOML reader available — skip silently
                with open(config_path, "rb") as _f:
                    data = _tl.load(_f)
                mon = data.get("monitoring")
                if isinstance(mon, dict):
                    if "auto_tensorboard" in mon:
                        mon["auto_tensorboard"] = tb_auto
                    if "auto_ngrok" in mon:
                        mon["auto_ngrok"] = ngrok_auto
                    try:
                        import tomli_w as _tw
                        with open(config_path, "wb") as _f:
                            _tw.dump(data, _f)
                    except ImportError:
                        try:
                            import toml as _tw2
                            with open(config_path, "w", encoding="utf-8") as _f:
                                _tw2.dump(data, _f)
                        except ImportError:
                            pass  # no TOML writer — leave config as-is
        except Exception as _e:
            self.append_log(f"[Monitoring] Impossible de patcher: {_e}\n")

    def _sanity_check_scheduler(self, config_path: str):
        """Fix incompatible scheduler params before launch.

        Common problem: a config has scheduler.type='CosineAnnealingLR' but still
        contains 'milestones' / 'gamma' from a previous MultiStepLR setting.
        PyTorch will raise TypeError: __init__() got an unexpected keyword argument 'milestones'.

        We rewrite the config in place (and back up to .usr_orig if not already done).
        Idempotent: a clean config triggers no rewrite.
        """
        ext = os.path.splitext(config_path)[1].lower()
        is_yaml = ext in (".yml", ".yaml")
        is_toml = ext == ".toml"
        if not (is_yaml or is_toml):
            return

        # Load
        try:
            data = self._load_config_any(config_path)
        except Exception as e:
            self.append_log(f"[Sanity] Lecture impossible: {e}\n")
            return

        # NeoSR (TOML) and Redux (YAML) both nest scheduler under [train.scheduler]
        train_section = data.get("train")
        if not isinstance(train_section, dict):
            return
        sched = train_section.get("scheduler")
        if not isinstance(sched, dict):
            return

        sched_type = str(sched.get("type", "")).lower()
        if not sched_type:
            return

        # Define allowed keys per scheduler family
        allowed = {"type"}
        if "multistep" in sched_type:
            allowed |= {"milestones", "gamma", "last_epoch", "verbose"}
        elif "cosineannealingwarmrestarts" in sched_type or ("cosine" in sched_type and "warm" in sched_type):
            allowed |= {"T_0", "T_mult", "eta_min", "last_epoch", "verbose",
                        "periods", "restart_weights"}
        elif "cosine" in sched_type:
            allowed |= {"T_max", "eta_min", "last_epoch", "verbose"}
        elif sched_type.endswith("step") or sched_type == "steplr":
            allowed |= {"step_size", "gamma", "last_epoch", "verbose"}
        elif "linear" in sched_type:
            allowed |= {"start_factor", "end_factor", "total_iters", "last_epoch", "verbose"}
        elif "exponential" in sched_type:
            allowed |= {"gamma", "last_epoch", "verbose"}
        elif "cyclic" in sched_type:
            allowed |= {"base_lr", "max_lr", "step_size_up", "step_size_down",
                        "mode", "gamma", "last_epoch", "verbose"}
        else:
            # Unknown scheduler — don't touch it (could be custom)
            return

        # Identify offending keys
        bad_keys = [k for k in sched.keys() if k not in allowed]
        if not bad_keys:
            return  # Nothing to fix

        self.append_log(
            f"[Sanity] Scheduler '{sched.get('type')}' a des cles incompatibles: "
            f"{bad_keys}. Nettoyage automatique...\n"
        )

        # Backup original ONCE
        backup_path = config_path + ".usr_orig"
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(config_path, backup_path)

        # Strip bad keys
        for k in bad_keys:
            del sched[k]

        # Add sensible defaults if missing for cosine variants
        if "cosine" in sched_type and "warm" not in sched_type and "T_max" not in sched:
            total_iter = train_section.get("total_iter") or data.get("total_iter") or 100000
            sched["T_max"] = int(total_iter)
        if "cosine" in sched_type and "eta_min" not in sched:
            sched["eta_min"] = 1e-7

        # Write back
        try:
            if is_yaml:
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            else:
                # Use toml lib for write (tomllib only reads in stdlib)
                try:
                    import tomli_w
                    with open(config_path, "wb") as f:
                        tomli_w.dump(data, f)
                except ImportError:
                    try:
                        import toml as _toml_w
                        with open(config_path, "w", encoding="utf-8") as f:
                            _toml_w.dump(data, f)
                    except ImportError:
                        self.append_log("[Sanity] Avertissement: ni tomli_w ni toml dispo, "
                                        "fichier TOML non reecrit.\n")
                        return
            self.append_log(f"[Sanity] Config nettoyee. Backup: {backup_path}\n")
        except Exception as e:
            self.append_log(f"[Sanity] Erreur ecriture: {e}\n")

    def _strip_custom_keys_from_yaml(self, yaml_path: str, keys_to_strip: list):
        """Remove specific keys from a YAML file (in place) so msgspec doesn't reject them.

        Saves a .usr_orig backup the first time so the user doesn't lose their settings.
        Idempotent: re-running on an already-stripped file is a no-op.
        """
        _ensure_yaml()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Backup original ONCE
        backup_path = yaml_path + ".usr_orig"
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(yaml_path, backup_path)

        modified = False
        for k in keys_to_strip:
            if k in data:
                del data[k]
                modified = True
            # Also check nested under 'degradations' (rare in Redux)
            if "degradations" in data and isinstance(data["degradations"], dict):
                if k in data["degradations"]:
                    del data["degradations"][k]
                    modified = True

        if modified:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def on_start(self):
        py = self.entries_dict["python_path"].get()
        sc = self.entries_dict["script_path"].get()
        cf = self.entries_dict["config_path"].get()

        if not py or not sc or not cf:
            self.append_log("[ERREUR] Chemins manquants. Veuillez charger un fichier config.\n")
            return

        # Compute training duration estimation
        try:
            self._show_training_estimation(cf)
        except Exception as e:
            self.append_log(f"[Estimation] {e}\n")

        self.apply_runtime_patch(sc)
        self.update_config_preview(cf)
        
        self.btn_start.configure(state="disabled", text=_t("En cours...", "Running..."))
        self.btn_stop.configure(state="normal")
        self.chk_shutdown.configure(state="normal")
        self.progress_bar.set(0)
        self.best_psnr = 0.0; self.best_ssim = 0.0
        self.card_psnr.configure(text="-- dB"); self.card_ssim.configure(text="--")
        
        self.start_time = datetime.datetime.now(); self.timer_running = True; self.update_timer()
        
        self.settings.set("python_path", py)
        self.settings.set("script_path", sc)
        self.settings.set("config_path", cf)

        # Create a working copy of the config for training.
        # All patches (monitoring strip, banding strip, scheduler fix, auto-resume)
        # are applied to the working copy — the original file on disk is NEVER modified,
        # so reloading after training start always shows the user's original values.
        import shutil as _shutil
        working_cf = cf
        self._training_tmp_config = None
        if cf.lower().endswith((".yml", ".yaml")):
            _base, _ext = os.path.splitext(cf)
            working_cf = _base + "._training_tmp" + _ext
            _shutil.copy2(cf, working_cf)
            self._training_tmp_config = working_cf
        elif cf.lower().endswith(".toml"):
            _base = os.path.splitext(cf)[0]
            working_cf = _base + "._training_tmp.toml"
            _shutil.copy2(cf, working_cf)
            self._training_tmp_config = working_cf

        # Log OTF degradation summary (reads original, unpatched config)
        try:
            self._log_otf_summary(cf)
        except Exception as e:
            self.append_log(f"[OTF] Avertissement summary: {e}\n")

        # Apply custom OTF degradation patches (banding/posterize) on working copy
        try:
            self._apply_custom_degradation_patches(working_cf, sc)
        except Exception as e:
            self.append_log(f"[Custom Deg] Avertissement: {e}\n")

        # Sanity-check the scheduler section on working copy
        try:
            self._sanity_check_scheduler(working_cf)
        except Exception as e:
            self.append_log(f"[Sanity] Avertissement scheduler: {e}\n")

        # Auto-resume: detect latest .state file and write resume_state into working copy
        try:
            working_cf = self._check_auto_resume(working_cf)
        except Exception as e:
            self.append_log(f"[Auto-Resume] Avertissement: {e}\n")

        # Pre-archive existing experiment folder — do this BEFORE starting gallery/TB/training
        # so no file locks exist yet.  traiNNer would do the same rename, but fails when
        # something else holds a handle (old TB, Explorer, etc.).
        try:
            self._pre_archive_exp_folder(working_cf, sc)
        except Exception as e:
            self.append_log(f"[Pre-Archive] Avertissement: {e}\n")

        # Record training in history database (use original path for display)
        try:
            self._record_training_start(cf, sc)
        except Exception as e:
            self.append_log(f"[History] Avertissement: {e}\n")

        # Patch monitoring flags in working copy — Studio settings are master control.
        # This prevents auto_tensorboard/auto_ngrok in the config file from bypassing
        # Studio checkboxes and launching unwanted processes.
        try:
            self._patch_monitoring_section(working_cf)
        except Exception as _e:
            self.append_log(f"[Monitoring] Avertissement patch: {_e}\n")

        # Auto-start gallery server if configured
        try:
            from src.core.settings import SettingsManager as _SM
            _sm = _SM()
            if _sm.get("gallery_auto_start_with_training", False):
                from src.core.gallery_server import get_server
                _srv = get_server()
                gal_dir = _sm.get("gallery_auto_dir", "")
                # Do NOT makedirs here — creating the dir would recreate an experiment folder
                # that was just pre-archived, causing PermissionError in the training engine.
                # Only start gallery if the directory already exists.
                if gal_dir and os.path.isdir(gal_dir):
                    _port = int(_sm.get("gallery_port", 8765))
                    _ngrok = bool(_sm.get("gallery_ngrok", True))
                    if not (_srv and _srv.httpd):
                        _srv.start(gal_dir, port=_port, with_ngrok=_ngrok)
                        self.append_log("[Serveurs] Galerie auto-démarrée.\n")
                elif gal_dir:
                    self.append_log("[Serveurs] Galerie : dossier introuvable, auto-start ignoré "
                                    "(il sera créé par l'entraînement).\n")
        except Exception as _e:
            self.append_log(f"[Serveurs] Avertissement auto-start: {_e}\n")

        # Auto-launch TensorBoard via Studio (browser-open aware).
        # Note: runner.py also reads auto_tensorboard from config — _patch_monitoring_section
        # keeps both in sync so TB is not launched twice.
        try:
            from src.core.settings import SettingsManager as _SM2
            _sm2 = _SM2()
            if _sm2.get("tb_auto_start_with_training", False):
                self.after(3000, self._open_tensorboard)  # slight delay so process starts first
                self.append_log("[Serveurs] TensorBoard auto-démarrage planifié...\n")
        except Exception as _e:
            self.append_log(f"[Serveurs] Avertissement auto-start TB: {_e}\n")

        # Reset Redux log buffer so fragments from previous run don't bleed in
        self._reset_redux_buf()
        # Store engine path + val dataset dir for live preview and validation box
        try:
            _cf_ext = os.path.splitext(cf)[1].lower()
            _eng = "neosr" if _cf_ext == ".toml" else "traiNNer-redux"
            self._current_engine_path = os.path.join(self.base_engine_path, _eng)
            cfg_data = self._load_config_any(cf)
            self._current_exp_name = cfg_data.get("name", "") if cfg_data else ""
            # Extract val dataset folder so validation box can supplement image names from FS
            self._val_dataset_dir = ""
            if cfg_data:
                _ds_list = cfg_data.get("datasets", [])

                def _extract_dir(ds_dict):
                    """Extract first valid directory from a dataset entry.
                    Redux YAML: dataroot_gt / dataroot_lq can be a list → take first element."""
                    for _key in ("dataroot_gt", "gt_path", "dataroot_lq", "lq_path"):
                        _v = ds_dict.get(_key, "")
                        if isinstance(_v, list):
                            _v = _v[0] if _v else ""
                        _v = str(_v).strip()
                        if _v:
                            return _v
                    return ""

                if isinstance(_ds_list, list):
                    for _ds in _ds_list:
                        if str(_ds.get("name", "")).lower().startswith("val"):
                            _d = _extract_dir(_ds)
                            if _d:
                                self._val_dataset_dir = _d
                                break
                elif isinstance(_ds_list, dict):
                    # NeoSR flat TOML — datasets is a dict of sub-tables
                    for _k, _ds in _ds_list.items():
                        if isinstance(_ds, dict) and str(_ds.get("name", _k)).lower().startswith("val"):
                            _d = _extract_dir(_ds)
                            if _d:
                                self._val_dataset_dir = _d
                                break
        except Exception:
            pass

        # For NeoSR: pre-create the experiment directory before training starts.
        # neosr tries to open experiments/<name>/train_*.log before the engine
        # creates the folder, causing FileNotFoundError on first run.
        if cf.lower().endswith(".toml"):
            try:
                _exp_name_mk = getattr(self, '_current_exp_name', '') or ''
                _eng_path_mk = getattr(self, '_current_engine_path', '') or ''
                if _exp_name_mk and _eng_path_mk:
                    os.makedirs(os.path.join(_eng_path_mk, "experiments", _exp_name_mk), exist_ok=True)
            except Exception:
                pass

        self.runner.start_training(py, sc, working_cf, self.append_log, self.on_finished)

    # --- LOGIQUE DETECTION ---
    def detect_engine_and_setup(self, config_path):
        if not config_path: return
        engine = None
        ext = os.path.splitext(config_path)[1].lower()

        if ext == ".toml": engine = "neosr"
        elif ext in [".yml", ".yaml"]: engine = "traiNNer-redux"
        
        if not engine: return 

        engine_root = os.path.join(self.base_engine_path, engine)
        if sys.platform == "win32":
            py_path = os.path.join(engine_root, ".venv", "Scripts", "python.exe")
        else:
            py_path = os.path.join(engine_root, ".venv", "bin", "python")
        script_path = os.path.join(engine_root, "train.py")

        if os.path.exists(py_path):
            self.entries_dict["python_path"].delete(0, "end")
            self.entries_dict["python_path"].insert(0, py_path)
        
        if os.path.exists(script_path):
            self.entries_dict["script_path"].delete(0, "end")
            self.entries_dict["script_path"].insert(0, script_path)

        self.update_config_preview(config_path, engine)

        # Persist visualization path for gallery auto-path (triggered on config load)
        try:
            cfg = self._load_config_any(config_path)
            exp_name = cfg.get("name", "")
            if exp_name and engine_root:
                viz_path = os.path.join(engine_root, "experiments", exp_name, "visualization")
                self.settings.set("gallery_auto_dir", viz_path)
        except Exception:
            pass

    # --- FICHIERS ---
    def setup_file_controls(self):
        self.frame_files.grid_columnconfigure(1, weight=1)
        self.frame_files.grid_rowconfigure(0, weight=1)
        self.frame_files.grid_rowconfigure(1, weight=1)
        self.frame_files.grid_rowconfigure(2, weight=1)
        
        self.row_file(0, "Python", "python_path", _t("Chemin vers l'exécutable python.exe (Auto-détecté).", "Path to python.exe executable (Auto-detected)."))
        self.row_file(1, "Script", "script_path", _t("Le script train.py (Auto-détecté).", "The train.py script (Auto-detected)."))
        self.row_file(2, "Config", "config_path", _t("Le fichier .toml (NeoSR) ou .yml (Redux).", "The .toml (NeoSR) or .yml (Redux) config file."))
        
        self.btn_scan = ctk.CTkButton(self.frame_files, text="Load", width=50, fg_color="#444", font=("Arial", 9, "bold"), 
                                      command=lambda: self.browse(self.entries_dict["config_path"], "config_path"))
        self.btn_scan.grid(row=0, column=3, rowspan=3, padx=5, sticky="ns")

    def row_file(self, row, txt, key, tip):
        ctk.CTkLabel(self.frame_files, text=txt, width=50, anchor="e", font=("Arial", 10)).grid(row=row, column=0, padx=5)
        e = ctk.CTkEntry(self.frame_files, height=28, font=("Consolas", 11))
        e.grid(row=row, column=1, sticky="ew", padx=2, pady=1)
        self.entries_dict[key] = e
        ctk.CTkButton(self.frame_files, text="..", width=20, height=28, command=lambda: self.browse(e, key)).grid(row=row, column=2, padx=2, pady=1)
        ToolTip(e, tip)

    def browse(self, entry, key):
        ft = [("Config", "*.toml *.yml *.yaml")] if "config" in key else [("Exe", "*.exe")] if "python" in key else [("Py", "*.py")]
        # initialdir : pour les configs, ouvrir dans IA_Engine/Option Custom
        initialdir = None
        if "config" in key:
            _candidate = os.path.join(os.path.expanduser("~"), "IA_Engine", "Option Custom")
            if os.path.isdir(_candidate):
                initialdir = _candidate
        p = filedialog.askopenfilename(filetypes=ft, initialdir=initialdir)
        if p:
            entry.delete(0, "end"); entry.insert(0, p); self.settings.set(key, p)
            if key == "config_path": self.detect_engine_and_setup(p)

    def update_config_preview(self, path, engine_hint=None):
        if not path or not os.path.exists(path): return
        if not engine_hint:
            if path.endswith(".toml"): engine_hint = "neosr"
            elif path.endswith(".yml") or path.endswith(".yaml"): engine_hint = "traiNNer-redux"

        try:
            arch, scale, bs, ps, acc = "?", "?", "?", "?", "?"
            if engine_hint == "neosr":
                with open(path, "rb") as f: data = tomllib.load(f)
                arch = data.get("network_g", {}).get("type", "?")
                scale = data.get("scale", "?")
                bs = data.get("datasets", {}).get("train", {}).get("batch_size", "?")
                ps = data.get("datasets", {}).get("train", {}).get("patch_size", "?")
                acc = data.get("datasets", {}).get("train", {}).get("accumulate", 1)
            elif engine_hint == "traiNNer-redux":
                with open(path, "r", encoding="utf-8") as f: data = yaml.safe_load(f)
                arch = data.get("network_g", {}).get("type", "?")
                scale = data.get("scale", "?")
                train_ds = data.get("datasets", {}).get("train", {})
                bs = train_ds.get("batch_size_per_gpu", "?")
                ps = train_ds.get("gt_size", train_ds.get("lq_size", "?"))
                acc = train_ds.get("accum_iter", "?")

            self.lbl_inf_arch.configure(text=str(arch).upper())
            self.lbl_inf_scale.configure(text=f"x{scale}")
            self.lbl_inf_batch.configure(text=str(bs))
            self.lbl_inf_patch.configure(text=str(ps))
            self.lbl_inf_acc.configure(text=str(acc))
            self.load_iters_from_config_data(data)
        except Exception: pass

    # --- MONITEURS ---
    def get_load_color(self, percent):
        if percent < 30: return "#3498db" 
        if percent < 60: return "#2ecc71" 
        if percent < 85: return "#e67e22" 
        return "#e74c3c" 

    def _mk_bar_row(self, parent, left_text, color):
        """Compact inline row: [label 52px] [====bar====] [value 62px]. Returns (lbl, bar, val)."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=2, pady=2)
        row.columnconfigure(1, weight=1)
        lbl = ctk.CTkLabel(row, text=left_text, font=("Consolas", 10), width=52, anchor="w")
        lbl.grid(row=0, column=0, sticky="w")
        bar = ctk.CTkProgressBar(row, height=6, progress_color=color, corner_radius=2)
        bar.grid(row=0, column=1, sticky="ew", padx=3)
        bar.set(0)
        val = ctk.CTkLabel(row, text="--", font=("Consolas", 10), width=62, anchor="e")
        val.grid(row=0, column=2, sticky="e")
        return lbl, bar, val

    def setup_gpu_monitor(self):
        inner = ctk.CTkFrame(self.frame_gpu, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=12, pady=(6, 4))
        ctk.CTkLabel(inner, text="GPU", font=("Roboto", 12, "bold"), text_color=("gray30", "#AAA")).pack(pady=(0, 2))
        self.lbl_gpu_name = ctk.CTkLabel(inner, text="...", font=("Arial", 11, "bold"),
                                          text_color="#2ecc71", wraplength=170)
        self.lbl_gpu_name.pack(pady=(0, 5))
        _, self.bar_vram,    self.lbl_vram_txt    = self._mk_bar_row(inner, "VRAM",    "#3498db")
        _, self.bar_util,    self.lbl_util_txt    = self._mk_bar_row(inner, "Load",    "#e67e22")
        _, self.bar_memctrl, self.lbl_memctrl_txt = self._mk_bar_row(inner, "MemCtrl", "#9b59b6")
        _, self.bar_pcie,    self.lbl_pcie_txt    = self._mk_bar_row(inner, "PCIe",    "#1abc9c")
        _, self.bar_gpuwatt, self.lbl_gpuwatt_txt = self._mk_bar_row(inner, "Power",   "#e74c3c")
        self.lbl_temp = ctk.CTkLabel(inner, text="-- °C", font=("Roboto", 16, "bold"))
        self.lbl_temp.pack(side="bottom", pady=4)

    def setup_cpu_monitor(self):
        inner = ctk.CTkFrame(self.frame_cpu, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=12, pady=(6, 4))
        ctk.CTkLabel(inner, text=_t("SYSTÈME", "SYSTEM"), font=("Roboto", 12, "bold"), text_color=("gray30", "#AAA")).pack(pady=(0, 2))
        col_cpu = "#9b59b6"
        nm = self.cpu_model_name.upper()
        if "AMD" in nm or "RYZEN" in nm: col_cpu = "#e67e22"
        elif "INTEL" in nm or "CORE" in nm: col_cpu = "#3498db"
        self.lbl_cpu_name = ctk.CTkLabel(inner, text=self.cpu_model_name, font=("Arial", 11, "bold"),
                                          text_color=col_cpu, wraplength=170)
        self.lbl_cpu_name.pack(pady=(0, 5))
        _, self.bar_cpu,     self.lbl_cpu_txt     = self._mk_bar_row(inner, "CPU",    "#2ecc71")
        _, self.bar_ram,     self.lbl_ram_txt     = self._mk_bar_row(inner, "RAM",    "#3498db")
        _, self.bar_disk,    self.lbl_disk_txt    = self._mk_bar_row(inner, "DISK",   "#95a5a6")
        _, self.bar_freq,    self.lbl_freq_txt    = self._mk_bar_row(inner, "Freq",   "#f39c12")
        _, self.bar_cpuwatt, self.lbl_cpuwatt_txt = self._mk_bar_row(inner, "Power",  "#c0392b")
        self.lbl_cpu_temp = ctk.CTkLabel(inner, text="-- °C", font=("Roboto", 16, "bold"))
        self.lbl_cpu_temp.pack(side="bottom", pady=4)

    def get_cpu_name(self):
        try:
            if sys.platform == "win32":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                return name.strip()
            elif sys.platform == "darwin": return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            else:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line: return line.split(":")[1].strip()
        except Exception: return platform.machine()

    def get_aida64_temp(self):
        if sys.platform != "win32":
            return ""
        try:
            import winreg
            aida_path = r"Software\FinalWire\AIDA64\SensorValues"
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, aida_path)
            i = 0
            while True:
                try:
                    val_name, val_data, _ = winreg.EnumValue(key, i)
                    if val_name in ["Value.TCPU", "Value.TCPUPKG", "Value.Tcore1"]: return f"{val_data} °C"
                    i += 1
                except OSError: break
        except Exception: return ""
        return ""

    def get_cpu_temp_fallback(self) -> str:
        """CPU temp fallback: OpenHardwareMonitor WMI → MSAcpi ThermalZone (wmic)."""
        if sys.platform != "win32":
            return ""
        # 1. OpenHardwareMonitor / LibreHardwareMonitor WMI (if running)
        try:
            import wmi  # type: ignore
            w = wmi.WMI(namespace="root/OpenHardwareMonitor")
            for sensor in w.Sensor():
                if getattr(sensor, "SensorType", "") == "Temperature" and "CPU" in getattr(sensor, "Name", ""):
                    return f"{sensor.Value:.0f} °C"
        except Exception:
            pass
        # 2. MSAcpi ThermalZone via wmic subprocess (no extra deps)
        try:
            res = subprocess.run(
                ["wmic", r"/namespace:\\root\WMI", "PATH",
                 "MSAcpi_ThermalZoneTemperature", "get", "CurrentTemperature", "/value"],
                capture_output=True, text=True, timeout=2,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            for line in res.stdout.splitlines():
                if "CurrentTemperature=" in line:
                    raw = int(line.split("=")[1].strip())
                    celsius = (raw / 10.0) - 273.15
                    if 0 < celsius < 120:
                        return f"{celsius:.0f} °C"
        except Exception:
            pass
        return ""

    def get_cpu_temp(self) -> str:
        """Get CPU temp: AIDA64 → OpenHardwareMonitor WMI → MSAcpi wmic."""
        return self.get_aida64_temp() or self.get_cpu_temp_fallback()

    def poll_gpu_stats(self):
        try:
            if NVML_AVAILABLE:
                # Fast path: pynvml direct (no subprocess)
                name = _nvml.nvmlDeviceGetName(_nvml_handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                mem = _nvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
                util = _nvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
                temp = _nvml.nvmlDeviceGetTemperature(_nvml_handle, 0)  # NVML_TEMPERATURE_GPU=0
                
                used_mb = mem.used // (1024 * 1024)
                total_mb = mem.total // (1024 * 1024)
                vram_pct = (used_mb / total_mb) * 100 if total_mb > 0 else 0
                gpu_util = util.gpu
                
                mem_ctrl = util.memory  # Memory controller utilization %

                # PCIe throughput (KB/s over ~20ms window) — may not be available on all drivers
                pcie_tx = pcie_rx = 0
                try:
                    pcie_tx = _nvml.nvmlDeviceGetPcieThroughput(_nvml_handle, 0)  # TX
                    pcie_rx = _nvml.nvmlDeviceGetPcieThroughput(_nvml_handle, 1)  # RX
                except Exception:
                    pass
                pcie_total_kbs = pcie_tx + pcie_rx
                # Dynamic max: starts at 2500 MB/s, auto-expands if real traffic exceeds it
                if not hasattr(self, '_pcie_max_kbs'):
                    self._pcie_max_kbs = 2500 * 1024  # 2500 MB/s in KB/s
                if pcie_total_kbs > self._pcie_max_kbs:
                    self._pcie_max_kbs = pcie_total_kbs
                PCIE_MAX_KBS = self._pcie_max_kbs
                pcie_pct = min(100.0, (pcie_total_kbs / PCIE_MAX_KBS) * 100) if PCIE_MAX_KBS > 0 else 0

                # GPU watts
                gpu_watts = gpu_watts_lim = 0
                try:
                    gpu_watts = _nvml.nvmlDeviceGetPowerUsage(_nvml_handle) // 1000
                    gpu_watts_lim = _nvml.nvmlDeviceGetEnforcedPowerLimit(_nvml_handle) // 1000
                except Exception:
                    pass

                self.lbl_gpu_name.configure(text=str(name))
                vram_suffix = ""
                vram_color = "white"
                if total_mb <= 8192:
                    if vram_pct > 95:
                        vram_suffix = "⚠"; vram_color = "#e74c3c"
                    elif vram_pct > 85:
                        vram_suffix = "⚠"; vram_color = "#e67e22"
                self.lbl_vram_txt.configure(
                    text=f"{used_mb/1024:.1f}/{total_mb//1024:.0f}GB{vram_suffix}",
                    text_color=vram_color)
                self.bar_vram.set(used_mb / max(total_mb, 1))
                self.bar_vram.configure(progress_color=self.get_load_color(vram_pct))
                # Tracking pic VRAM + conso moyenne pour le benchmark historique
                if getattr(self, "_history_row_id", 0) > 0:
                    if used_mb > getattr(self, "_vram_peak_mb", 0):
                        self._vram_peak_mb = used_mb
                self.lbl_util_txt.configure(text=f"{gpu_util}%")
                self.bar_util.set(gpu_util / 100.0)
                self.bar_util.configure(progress_color=self.get_load_color(gpu_util))
                self.lbl_memctrl_txt.configure(text=f"{mem_ctrl}%")
                self.bar_memctrl.set(mem_ctrl / 100.0)
                self.bar_memctrl.configure(progress_color=self.get_load_color(mem_ctrl))
                if pcie_total_kbs > 0:
                    self.lbl_pcie_txt.configure(text=f"{pcie_total_kbs // 1024}MB/s")
                else:
                    self.lbl_pcie_txt.configure(text="--")
                self.bar_pcie.set(pcie_pct / 100.0)
                self.bar_pcie.configure(progress_color=self.get_load_color(pcie_pct))
                if gpu_watts > 0:
                    watt_pct = (gpu_watts / max(gpu_watts_lim, 1)) * 100
                    watt_lim_str = f"/{gpu_watts_lim}W" if gpu_watts_lim > 0 else "W"
                    self.lbl_gpuwatt_txt.configure(text=f"{gpu_watts}{watt_lim_str}")
                    self.bar_gpuwatt.set(min(1.0, gpu_watts / max(gpu_watts_lim, 1)))
                    self.bar_gpuwatt.configure(progress_color=self.get_load_color(watt_pct))
                    # Accumule la conso élec pour avg dans le benchmark
                    if getattr(self, "_history_row_id", 0) > 0:
                        if not hasattr(self, "_power_samples"):
                            self._power_samples = []
                        self._power_samples.append(gpu_watts)
                self.lbl_temp.configure(
                    text=f"{temp} °C",
                    text_color="#e74c3c" if temp > 80 else "white"
                )
            else:
                # Fallback: nvidia-smi subprocess (slower)
                cmd = ["nvidia-smi",
                       "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,utilization.memory",
                       "--format=csv,noheader,nounits"]
                creationflags = 0x08000000 if sys.platform == 'win32' else 0
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, creationflags=creationflags)
                if res.returncode == 0:
                    d = res.stdout.strip().split(',')
                    if len(d) >= 5:
                        self.lbl_gpu_name.configure(text=d[0].strip())
                        u, t = int(d[2].strip()), int(d[3].strip())
                        vram_pct = (u / t) * 100
                        vsuf = ""; vcol = "white"
                        if t <= 8192:
                            if vram_pct > 95: vsuf = " ⚠"; vcol = "#e74c3c"
                            elif vram_pct > 85: vsuf = " ⚠"; vcol = "#e67e22"
                        self.lbl_vram_txt.configure(text=f"{u/1024:.1f}/{t//1024:.0f}GB{vsuf}", text_color=vcol)
                        self.bar_vram.set(u / t)
                        self.bar_vram.configure(progress_color=self.get_load_color(vram_pct))
                        util = int(d[1].strip())
                        self.lbl_util_txt.configure(text=f"{util}%")
                        self.bar_util.set(util / 100.0)
                        self.bar_util.configure(progress_color=self.get_load_color(util))
                        mem_ctrl = int(d[5].strip()) if len(d) >= 6 else 0
                        self.lbl_memctrl_txt.configure(text=f"{mem_ctrl}%")
                        self.bar_memctrl.set(mem_ctrl / 100.0)
                        self.bar_memctrl.configure(progress_color=self.get_load_color(mem_ctrl))
                        self.lbl_pcie_txt.configure(text="--")
                        self.bar_pcie.set(0)
                        # GPU watts via nvidia-smi power.draw (separate query, optional)
                        try:
                            res_w = subprocess.run(
                                ["nvidia-smi", "--query-gpu=power.draw,power.limit", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=3,
                                creationflags=0x08000000 if sys.platform == 'win32' else 0)
                            if res_w.returncode == 0:
                                dw = res_w.stdout.strip().split(',')
                                if len(dw) >= 2:
                                    gw = float(dw[0].strip())
                                    gl = float(dw[1].strip()) if dw[1].strip().replace('.','').isdigit() else 0
                                    self.lbl_gpuwatt_txt.configure(text=f"{gw:.0f}/{gl:.0f}W" if gl > 0 else f"{gw:.0f}W")
                                    self.bar_gpuwatt.set(min(1.0, gw / max(gl, 1)))
                                    self.bar_gpuwatt.configure(progress_color=self.get_load_color((gw/max(gl,1))*100))
                        except Exception:
                            pass
                        tmp = int(d[4].strip())
                        self.lbl_temp.configure(text=f"{tmp} °C", text_color="#e74c3c" if tmp > 80 else ("gray10", "white"))
        except Exception:
            pass
        self.after(2000, self.poll_gpu_stats)

    def get_aida64_cpu_watts(self):
        """Read CPU package power from AIDA64 registry (W)."""
        if sys.platform != "win32":
            return None
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\FinalWire\AIDA64\SensorValues")
            i = 0
            while True:
                try:
                    name, data, _ = winreg.EnumValue(key, i)
                    if name in ("Value.PCPUPKG", "Value.PCPU", "Value.PCPUIA"):
                        return float(data)
                    i += 1
                except OSError:
                    break
        except Exception:
            pass
        return None

    def poll_cpu_stats(self):
        if PSUTIL_AVAILABLE:
            try:
                cpu = psutil.cpu_percent()
                self.lbl_cpu_txt.configure(text=f"{cpu:.1f}%")
                self.bar_cpu.set(cpu / 100.0)
                self.bar_cpu.configure(progress_color=self.get_load_color(cpu))

                ram = psutil.virtual_memory()
                self.lbl_ram_txt.configure(text=f"{ram.percent:.1f}%")
                self.bar_ram.set(ram.percent / 100.0)
                self.bar_ram.configure(progress_color=self.get_load_color(ram.percent))

                total, used, _ = shutil.disk_usage(".")
                disk_pct = (used / total) * 100
                self.lbl_disk_txt.configure(text=f"{disk_pct:.1f}%")
                self.bar_disk.set(disk_pct / 100.0)
                self.bar_disk.configure(progress_color=self.get_load_color(disk_pct))

                # CPU frequency — chaîne: AIDA64 CLK → NtPowerInformation → psutil
                try:
                    cur_mhz = None
                    if sys.platform == "win32":
                        # 1. AIDA64 SensorValues — clé directe Value.SCPUCLK (MHz)
                        #    Évite de confondre avec débit réseau (SNIC*) ou mémoire GPU qui tombent
                        #    aussi dans la plage 2500-7500.
                        try:
                            import winreg
                            _ak = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                                 r"Software\FinalWire\AIDA64\SensorValues")
                            # Lecture directe de la clé CPU Clock
                            try:
                                _clk_val, _ = winreg.QueryValueEx(_ak, "Value.SCPUCLK")
                                _v = float(_clk_val)
                                if 500.0 < _v < 8000.0:   # sanity check (MHz)
                                    cur_mhz = int(_v)
                            except (FileNotFoundError, OSError, ValueError, TypeError):
                                pass
                            # Fallback : scan par plage si SCPUCLK absent, cap 6000 MHz
                            if cur_mhz is None:
                                _i = 0
                                _clk_candidates = []
                                while True:
                                    try:
                                        _nm, _dat, _ = winreg.EnumValue(_ak, _i)
                                        try:
                                            _v = float(_dat)
                                            # Cap à 6000 MHz — aucun CPU consumer ≥ 6 GHz
                                            # Exclut SNIC* (débit réseau), SGPU* (VRAM MB)
                                            if 2500.0 < _v < 6000.0:
                                                _clk_candidates.append(_v)
                                        except (ValueError, TypeError):
                                            pass
                                        _i += 1
                                    except OSError:
                                        break
                                if _clk_candidates:
                                    cur_mhz = int(max(_clk_candidates))
                            winreg.CloseKey(_ak)
                        except Exception:
                            pass
                        # 2. NtPowerInformation (PState freq — précis sans OC)
                        if cur_mhz is None:
                            class _PPI(ctypes.Structure):
                                _fields_ = [
                                    ("Number",          ctypes.c_ulong),
                                    ("MaxMhz",          ctypes.c_ulong),
                                    ("CurrentMhz",      ctypes.c_ulong),
                                    ("MhzLimit",        ctypes.c_ulong),
                                    ("MaxIdleState",    ctypes.c_ulong),
                                    ("CurrentIdleState", ctypes.c_ulong),
                                ]
                            _n = os.cpu_count() or 1
                            _buf = (_PPI * _n)()
                            if ctypes.windll.ntdll.NtPowerInformation(11, None, 0, _buf, ctypes.sizeof(_buf)) == 0:
                                cur_mhz = max(_buf[_j].CurrentMhz for _j in range(_n))
                    if cur_mhz is None:
                        _f = psutil.cpu_freq()
                        if _f: cur_mhz = _f.current
                    if cur_mhz:
                        cur_ghz = float(cur_mhz) / 1000.0
                        if not hasattr(self, '_cpu_freq_max_ghz'):
                            _fmax = psutil.cpu_freq()
                            _base = (_fmax.max / 1000.0) if (_fmax and _fmax.max) else 5.5
                            # psutil.max often reports base clock — add headroom for boost
                            self._cpu_freq_max_ghz = max(_base * 1.4, 4.5)
                        if cur_ghz > self._cpu_freq_max_ghz:
                            self._cpu_freq_max_ghz = cur_ghz * 1.05
                        self.lbl_freq_txt.configure(text=f"{cur_ghz:.2f}GHz")
                        self.bar_freq.set(min(1.0, cur_ghz / max(self._cpu_freq_max_ghz, 0.1)))
                        self.bar_freq.configure(progress_color="#f39c12")
                except Exception:
                    pass

                # CPU watts (AIDA64 registry)
                try:
                    cpu_w = self.get_aida64_cpu_watts()
                    if cpu_w is not None:
                        CPU_WATT_MAX = 200.0
                        self.lbl_cpuwatt_txt.configure(text=f"{cpu_w:.0f}W")
                        self.bar_cpuwatt.set(min(1.0, cpu_w / CPU_WATT_MAX))
                        self.bar_cpuwatt.configure(progress_color=self.get_load_color((cpu_w / CPU_WATT_MAX) * 100))
                except Exception:
                    pass

                t_str = self.get_cpu_temp()
                self.lbl_cpu_temp.configure(text=t_str)
            except Exception:
                pass
        self.after(2000, self.poll_cpu_stats)

    def setup_dashboard(self):
        for i in range(5): self.frame_stats.grid_columnconfigure(i, weight=1)
        self.card_timer = self.create_card(0, 0, _t("DURÉE", "DURATION"), "00:00:00", "white",
            tip=_t("⏱ Durée totale de la session d'entraînement en cours depuis le démarrage.",
                   "⏱ Total duration of the current training session since start."))
        self.card_iter = self.create_card(0, 1, _t("ITÉRATION", "ITERATION"), "0 / --", "#2ecc71",
            tip=_t("🔄 Itérations d'optimisation (mises à jour des poids).\n"
                   "Ligne 1 : étapes optimizer actuelles / total configuré.\n"
                   "Ligne 2 (accumulation active) : GPU×N — passes physiques réelles\n"
                   "  sur le GPU (forward + backward), soit optimizer × accumulate.\n\n"
                   "Exemple : 60 000 iters config + accumulate=2\n"
                   "  → 60 000 mises à jour des poids\n"
                   "  → 120 000 passes GPU réelles",
                   "🔄 Optimization iterations (weight updates).\n"
                   "Line 1: current optimizer steps / total configured.\n"
                   "Line 2 (accumulation active): GPU×N — real physical passes\n"
                   "  on the GPU (forward + backward), i.e. optimizer × accumulate.\n\n"
                   "Example: 60,000 config iters + accumulate=2\n"
                   "  → 60,000 weight updates\n"
                   "  → 120,000 real GPU passes"))
        self.card_speed = self.create_card(0, 2, _t("VITESSE", "SPEED"), "-- it/s", "#f1c40f",
            tip=_t("⚡ Vitesse de traitement en itérations par seconde.\n"
                   "Valeur instantanée du dernier log. Dépend du batch size, scale et GPU.",
                   "⚡ Processing speed in iterations per second.\n"
                   "Instantaneous value from last log. Depends on batch size, scale and GPU."))
        self.card_eta = self.create_card(0, 3, _t("FIN ESTIMÉE", "ETA"), "--:--:--", "white",
            tip=_t("🏁 Heure de fin estimée basée sur la vitesse actuelle et les itérations restantes.\n"
                   "Se recalcule à chaque itération loggée.",
                   "🏁 Estimated end time based on current speed and remaining iterations.\n"
                   "Recalculated at each logged iteration."))
        self.card_epoch = self.create_card(0, 4, "EPOCH", "0", "#e67e22",
            tip=_t("📦 Epoch actuelle — un epoch = passage complet sur tout le dataset d'entraînement.\n"
                   "Dépend du nombre d'images et du batch size.",
                   "📦 Current epoch — one epoch = full pass over the entire training dataset.\n"
                   "Depends on the number of images and batch size."))
        self.card_loss_g = self.create_card(1, 0, "LOSS G", "--", "#e74c3c",
            tip=_t("📉 Perte totale du Générateur (G).\n"
                   "Cumul : pixel loss + perceptual loss + GAN loss + autres composantes.\n"
                   "Plus basse = meilleure qualité SR. Valeur normale : 0.001–0.1",
                   "📉 Total Generator (G) loss.\n"
                   "Sum: pixel loss + perceptual loss + GAN loss + other components.\n"
                   "Lower = better SR quality. Normal range: 0.001–0.1"))
        self.card_loss_d = self.create_card(1, 1, "LOSS D", "--", "#9b59b6",
            tip=_t("⚖️ Perte du Discriminateur (D) — uniquement en mode GAN.\n"
                   "Idéalement ~0.5–0.7 (equilibre G/D).\n"
                   "Trop basse → D trop puissant → G a du mal à le duper.\n"
                   "Trop haute → G domine → artefacts possibles.",
                   "⚖️ Discriminator (D) loss — GAN mode only.\n"
                   "Ideally ~0.5–0.7 (G/D balance).\n"
                   "Too low → D too strong → G struggles to fool it.\n"
                   "Too high → G dominates → possible artifacts."))
        self.card_lr = self.create_card(1, 2, "LR", "--", "white",
            tip=_t("🎚 Learning Rate actuel du Générateur.\n"
                   "Diminue selon le scheduler choisi (MultiStepLR, CosineAnnealing, etc.).\n"
                   "Un LR trop élevé → instabilité. Trop bas → convergence lente.",
                   "🎚 Current Generator Learning Rate.\n"
                   "Decreases according to the chosen scheduler (MultiStepLR, CosineAnnealing, etc.).\n"
                   "Too high → instability. Too low → slow convergence."))
        self.card_psnr = self.create_card(1, 3, "BEST PSNR", "-- dB", "#3498db",
            tip=_t("📡 Meilleur PSNR (Peak Signal-to-Noise Ratio) obtenu en validation.\n"
                   "> 28 dB = bon  |  > 32 dB = excellent  |  > 38 dB = quasi-parfait.\n"
                   "Affiché avec l'itération où ce score a été atteint.",
                   "📡 Best PSNR (Peak Signal-to-Noise Ratio) achieved in validation.\n"
                   "> 28 dB = good  |  > 32 dB = excellent  |  > 38 dB = near-perfect.\n"
                   "Shown with the iteration where this score was reached."))
        self.card_ssim = self.create_card(1, 4, "BEST SSIM", "--", "#1abc9c",
            tip=_t("🔬 Meilleur SSIM (Structural Similarity Index) obtenu en validation.\n"
                   "De 0 à 1 — proche de 1.0 = excellente similarité structurelle.\n"
                   "> 0.90 = bon  |  > 0.95 = excellent.\n"
                   "Actualisé après chaque run de validation.",
                   "🔬 Best SSIM (Structural Similarity Index) achieved in validation.\n"
                   "From 0 to 1 — close to 1.0 = excellent structural similarity.\n"
                   "> 0.90 = good  |  > 0.95 = excellent.\n"
                   "Updated after each validation run."))

    def setup_info_panel(self):
        for i in range(5): self.frame_info.grid_columnconfigure(i, weight=1)
        def mk_info(col, label, desc):
            f = ctk.CTkFrame(self.frame_info, fg_color="transparent")
            f.grid(row=0, column=col, sticky="ew")
            ctk.CTkLabel(f, text=f"{label} : ", font=("Arial", 11), text_color="gray").pack(side="left", padx=(10, 0))
            v = ctk.CTkLabel(f, text="--", font=("Roboto", 12, "bold"), text_color="#3B8ED0")
            v.pack(side="left")
            ToolTip(f, desc); ToolTip(v, desc)
            return v
        
        self.lbl_inf_arch = mk_info(0, "Architecture", _t("Le modèle de réseau neuronal.", "The neural network model."))
        self.lbl_inf_scale = mk_info(1, "Scale", _t("Facteur d'agrandissement.", "Upscaling factor."))
        self.lbl_inf_batch = mk_info(2, "Batch Size", _t("Images par passe GPU.", "Images per GPU pass."))
        self.lbl_inf_patch = mk_info(3, "Patch/LQ Size", _t("Taille de travail.", "Working patch size."))
        self.lbl_inf_acc = mk_info(4, "Accumulate", "Gradient accumulation.")

    def create_card(self, r, c, title, val, color, tip=""):
        f = ctk.CTkFrame(self.frame_stats, fg_color=("#DEDEDE", "#222"), corner_radius=4)
        f.grid(row=r, column=c, padx=3, pady=3, sticky="ew")
        ctk.CTkLabel(f, text=title, font=("Arial", 8, "bold"), text_color="gray").pack(pady=(2,0))
        l = ctk.CTkLabel(f, text=val, font=("Roboto", 12, "bold"), text_color=color)
        l.pack(pady=(0,2))
        if tip: ToolTip(f, tip); ToolTip(l, tip)
        return l

    def on_stop(self): self.runner.stop_training(self.append_log)
    def _play_finish_sound(self):
        """Play completion sound if enabled, with volume control."""
        try:
            enabled = self.settings.get("sound_enabled", True)
            if not enabled:
                return
            volume = int(self.settings.get("sound_volume", 70))
            sound_path = os.path.join(os.getcwd(), "assets", "success.wav")
            if not os.path.exists(sound_path):
                return
            self._play_wav_with_volume(sound_path, volume)
        except Exception:
            pass

    def _play_wav_with_volume(self, wav_path, volume_pct):
        """Play a WAV file at a given volume percentage (0-100)."""
        try:
            if sys.platform == "win32":
                # Set system wave volume via winmm then play
                import ctypes
                winmm = ctypes.windll.winmm
                # waveOutSetVolume: 0x0000=mute, 0xFFFF=max, both channels
                vol = int(volume_pct / 100 * 0xFFFF)
                vol_dword = vol | (vol << 16)  # Left and right channels
                winmm.waveOutSetVolume(0, vol_dword)
                import winsound
                winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                # Linux: use aplay with amixer for volume
                subprocess.Popen(["aplay", "-q", wav_path], stderr=subprocess.DEVNULL)
        except Exception:
            # Fallback: just play without volume control
            try:
                import winsound
                winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass

    def _show_metrics_graph(self):
        """Show real-time metrics graph in a popup window with auto-refresh."""
        import tkinter as tk
        if getattr(self, '_metrics_win', None) and self._metrics_win.winfo_exists():
            self._metrics_win.focus(); return
        win = ctk.CTkToplevel(self)
        self._metrics_win = win
        win.title(_t("Métriques d'Entraînement", "Training Metrics"))
        win.geometry("820x700")
        win.attributes("-topmost", True)
        win.after(500, lambda: win.attributes("-topmost", False))

        # ── Barre d'outils ──────────────────────────────────────────────────
        toolbar = ctk.CTkFrame(win, fg_color="#0d1117", height=32, corner_radius=0)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)
        lbl_status = ctk.CTkLabel(toolbar, text="", font=("Roboto", 10), text_color="#888")
        lbl_status.pack(side="left", padx=10)
        btn_refresh = ctk.CTkButton(toolbar, text=_t("⟳ Rafraîchir", "⟳ Refresh"), width=100, height=24,
                                    fg_color="#1e3a5f", hover_color="#2a5298", corner_radius=4)
        btn_refresh.pack(side="right", padx=6, pady=4)

        # ── Canvas ──────────────────────────────────────────────────────────
        canvas = tk.Canvas(win, bg="#1a1a2e", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        # ── Parsing ─────────────────────────────────────────────────────────
        def parse_metrics():
            _losses, _psnrs, _loss_d = [], [], []
            _log = self.textbox_logs.get("1.0", "end")
            for _line in _log.split("\n"):
                m = re.search(r'l_g_total[:\s]+([0-9.e+\-]+)', _line)
                if not m:
                    # NeoSR : [ total: 1.1018e+00 | ff: ... ]  (pipe avant le ])
                    # Redux  : [ total: 1.3654e+00 ] (bracket direct)
                    m = re.search(r'\[\s*total[:\s]+([0-9.e+\-]+)', _line)
                if m:
                    try: _losses.append(float(m.group(1)))
                    except Exception: pass
                # Loss D — d_total si NeoSR/legacy, sinon (l_d_real+l_d_fake)/2 pour traiNNer-redux
                md = re.search(r'd_total[:\s]+([0-9.e+\-]+)', _line)
                if md:
                    try: _loss_d.append(float(md.group(1)))
                    except Exception: pass
                else:
                    md_r = re.search(r'(?:l_)?d_real[:\s]+([0-9.e+\-]+)', _line)
                    md_f = re.search(r'(?:l_)?d_fake[:\s]+([0-9.e+\-]+)', _line)
                    if md_r and md_f:
                        try: _loss_d.append((float(md_r.group(1)) + float(md_f.group(1))) / 2)
                        except Exception: pass
                # PSNR — NeoSR: "Best PSNR : 24.6367........ dB"  → [0-9]+\.[0-9]+ évite de capturer les points trailers
                # Redux: "psnr = 32.88" ou "psnr: 32.88"
                mp = re.search(r'psnr\s*[=:]\s*([0-9]+\.[0-9]+)', _line, re.IGNORECASE)
                if mp:
                    try: _psnrs.append(float(mp.group(1)))
                    except Exception: pass
            _pf = re.search(r'print[_\s]freq\s*[=:]\s*(\d+)', _log, re.IGNORECASE)
            _freq = int(_pf.group(1)) if _pf else 100
            # Inférer val_freq depuis les données réelles (ratio loss pts / psnr pts)
            # plutôt que de parser le log (val_freq souvent absent du log traiNNer-redux)
            if _losses and _psnrs and len(_psnrs) > 1:
                _estimated_iter = len(_losses) * _freq
                _val_freq = max(1, _estimated_iter // len(_psnrs))
            else:
                _vf = re.search(r'val[_\s]freq\s*[=:]\s*(\d+)', _log, re.IGNORECASE)
                _val_freq = int(_vf.group(1)) if _vf else 5000
            # Détecter l'iter de départ (resume ou début) pour l'axe X
            # Cherche le premier [ iter: N,NNN ] dans les lignes de log
            _start_iter = 0
            _first_iter_m = re.search(r'\[\s*iter:\s*([\d,]+)\s*\]', _log)
            if _first_iter_m:
                try: _start_iter = int(_first_iter_m.group(1).replace(",", "")) - _freq
                except Exception: pass
            return _losses, _psnrs, _loss_d, _freq, _val_freq, max(0, _start_iter)

        # ── Draw ─────────────────────────────────────────────────────────────
        def draw_graph(data, color, label, y_offset, height, print_freq, start_iter=0):
            if not data or len(data) < 2:
                return
            min_v, max_v = min(data), max(data)
            if max_v == min_v:
                max_v = min_v + 1
            w = canvas.winfo_width() - 92   # +12 left margin vs old -80
            h = height - 60                  # +10 bottom margin vs old -50
            x_start, y_start = 74, y_offset + 22   # left 74 (was 60), top +22 (was +20)
            # Axes
            canvas.create_line(x_start, y_start, x_start, y_start + h, fill="#444")
            canvas.create_line(x_start, y_start + h, x_start + w, y_start + h, fill="#444")
            # Label
            canvas.create_text(x_start + w // 2, y_offset + 11, text=label, fill=color, font=("Roboto", 11, "bold"))
            # Y — Min/Max
            canvas.create_text(x_start - 6, y_start, text=f"{max_v:.4f}", fill="#888", anchor="e", font=("Roboto", 8))
            canvas.create_text(x_start - 6, y_start + h, text=f"{min_v:.4f}", fill="#888", anchor="e", font=("Roboto", 8))
            # X — graduation ~6 ticks (offset par start_iter si resume)
            n = len(data)
            tick_step = max(1, n // 6)
            for ti in range(0, n, tick_step):
                tx = x_start + (ti / max(n - 1, 1)) * w
                iter_val = start_iter + ti * print_freq
                canvas.create_line(tx, y_start + h, tx, y_start + h + 4, fill="#555")
                canvas.create_text(tx, y_start + h + 12, text=f"{iter_val:,}",
                                   fill="#666", font=("Roboto", 7), anchor="n")
            tx_last = x_start + w
            iter_last = start_iter + (n - 1) * print_freq
            canvas.create_line(tx_last, y_start + h, tx_last, y_start + h + 4, fill="#555")
            canvas.create_text(tx_last, y_start + h + 12, text=f"{iter_last:,}",
                               fill="#666", font=("Roboto", 7), anchor="n")
            canvas.create_text(x_start + w // 2, y_start + h + 26, text=_t("itérations", "iterations"),
                               fill="#555", font=("Roboto", 8))
            # Curve
            points = []
            for i, v in enumerate(data):
                x = x_start + (i / max(n - 1, 1)) * w
                y = y_start + h - ((v - min_v) / (max_v - min_v)) * h
                points.extend([x, y])
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2, smooth=True)

        # ── Refresh ───────────────────────────────────────────────────────────
        _state = {"losses": [], "psnrs": [], "loss_d": [], "freq": 100, "val_freq": 5000, "start_iter": 0}

        def redraw(event=None):
            canvas.delete("all")
            losses = _state["losses"]
            psnrs  = _state["psnrs"]
            loss_d = _state["loss_d"]
            pf     = _state["freq"]
            vf     = _state["val_freq"]
            si     = _state["start_iter"]
            ch = canvas.winfo_height()
            if not losses and not psnrs:
                canvas.create_text(canvas.winfo_width() // 2, ch // 2,
                                   text=_t("Aucune métrique — en attente de l'entraînement...", "No metrics yet — waiting for training..."),
                                   fill="#555", font=("Roboto", 12))
                return
            has_g = bool(losses)
            has_d = bool(loss_d)
            has_p = bool(psnrs)
            n_panels = sum([has_g, has_d, has_p])
            ph = ch // n_panels if n_panels > 0 else ch
            offset = 0
            if has_g:
                draw_graph(losses, "#e74c3c", f"Loss G ({len(losses)} pts)", offset, ph, pf, si)
                offset += ph
            if has_d:
                draw_graph(loss_d, "#9b59b6", f"Loss D ({len(loss_d)} pts)", offset, ph, pf, si)
                offset += ph
            if has_p:
                draw_graph(psnrs, "#2ecc71", f"PSNR ({len(psnrs)} pts)", offset, ph, vf, si)

        def refresh_data(manual=False):
            if not win.winfo_exists():
                return
            losses, psnrs, loss_d, freq, val_freq, start_iter = parse_metrics()
            _state["losses"] = losses
            _state["psnrs"] = psnrs
            _state["loss_d"] = loss_d
            _state["freq"] = freq
            _state["val_freq"] = val_freq
            _state["start_iter"] = start_iter
            lbl_status.configure(text=f"Loss G: {len(losses)} pts  |  Loss D: {len(loss_d)} pts  |  PSNR: {len(psnrs)} pts  |  auto-refresh 5s")
            redraw()
            # Reschedule auto-refresh
            if win.winfo_exists():
                win.after(5000, refresh_data)

        btn_refresh.configure(command=lambda: refresh_data(manual=True))
        canvas.bind("<Configure>", redraw)
        # Démarrer le premier refresh immédiatement
        win.after(100, refresh_data)

    def _open_tensorboard(self):
        """Open tensorboard in browser using the correct venv Python and logdir.

        Supports both engine layouts:
          - traiNNer-redux : {engine_root}/tb_logger
          - NeoSR          : {engine_root}/experiments/{exp_name}/tb_logger

        When both exist (centralized view), uses --logdir_spec to show both in one TB instance.
        Browser auto-open is gated on the tb_auto_open_browser setting (default True).
        """
        import yaml as _yaml

        # --- 1. Resolve Python executable ---
        py_entry = self.entries_dict.get("python_path")
        py_exec = py_entry.get().strip() if py_entry else ""
        if not py_exec or not os.path.isfile(py_exec):
            py_exec = sys.executable

        # --- 2. Resolve engine root from script path ---
        script_entry = self.entries_dict.get("script_path")
        config_entry = self.entries_dict.get("config_path")
        engine_root = None
        if script_entry:
            sp = script_entry.get().strip()
            if sp and os.path.isfile(sp):
                engine_root = os.path.dirname(sp)

        # --- 3. Build labelled logdir map {label: path} for each engine layout ---
        logdir_map = {}   # {label: abs_path}

        def _probe_engine(root, label):
            """Check both TB layouts under root, add whichever exists."""
            if not root or not os.path.isdir(root):
                return
            # traiNNer-redux style: {root}/tb_logger
            direct = os.path.join(root, "tb_logger")
            if os.path.isdir(direct):
                logdir_map[label] = direct
                return
            # NeoSR style: {root}/experiments/{exp_name}/tb_logger
            # First try to read exp name from config
            exp_name = None
            if config_entry:
                cfg_path = config_entry.get().strip()
                if cfg_path and os.path.isfile(cfg_path):
                    try:
                        with open(cfg_path, "r", encoding="utf-8") as _f:
                            _cfg = _yaml.safe_load(_f) or {}
                        exp_name = _cfg.get("name", "")
                    except Exception:
                        pass
            if exp_name:
                nested = os.path.join(root, "experiments", exp_name, "tb_logger")
                if os.path.isdir(nested):
                    logdir_map[label] = nested
                    return
            # Fallback: scan experiments/ for any tb_logger subfolders
            exp_base = os.path.join(root, "experiments")
            if os.path.isdir(exp_base):
                for ename in sorted(os.listdir(exp_base)):
                    candidate = os.path.join(exp_base, ename, "tb_logger")
                    if os.path.isdir(candidate):
                        logdir_map[label] = candidate
                        return

        # Probe active engine root first
        if engine_root:
            _probe_engine(engine_root, "ActiveEngine")

        # Always probe both known IA_Engine locations for centralized view
        _ia = os.path.join(os.path.expanduser("~"), "IA_Engine")
        _probe_engine(os.path.join(_ia, "traiNNer-redux"), "traiNNer")
        _probe_engine(os.path.join(_ia, "neosr"), "NeoSR")

        if not logdir_map:
            from tkinter import messagebox
            messagebox.showinfo("TensorBoard",
                _t("Aucun dossier tb_logger trouvé.\n"
                   "Lancez un entrainement d'abord,\n"
                   "ou vérifiez le chemin du moteur.",
                   "No tb_logger folder found.\n"
                   "Start a training run first,\n"
                   "or check the engine path."))
            return

        # Remove duplicate paths (ActiveEngine may duplicate one of the others)
        seen_paths = {}
        deduped = {}
        for label, path in logdir_map.items():
            norm = os.path.normcase(os.path.normpath(path))
            if norm not in seen_paths:
                seen_paths[norm] = label
                deduped[label] = path
        logdir_map = deduped

        # --- 4. Build --logdir or --logdir_spec argument ---
        if len(logdir_map) == 1:
            tb_flag = "--logdir"
            tb_val = next(iter(logdir_map.values()))
            log_desc = tb_val
        else:
            tb_flag = "--logdir_spec"
            tb_val = ",".join(f"{lbl}:{path}" for lbl, path in logdir_map.items())
            log_desc = " | ".join(f"{lbl}:{path}" for lbl, path in logdir_map.items())

        self.append_log(f"[TensorBoard] Logdir(s) : {log_desc}\n")

        # --- 5. Resolve tb_launcher.py ---
        _this_dir = os.path.dirname(os.path.abspath(__file__))  # src/ui/tabs
        _launcher = os.path.normpath(os.path.join(_this_dir, "..", "..", "core", "tb_launcher.py"))
        if not os.path.isfile(_launcher):
            _launcher = os.path.normpath(os.path.join(os.getcwd(), "src", "core", "tb_launcher.py"))
        if not os.path.isfile(_launcher):
            self.append_log(f"[TensorBoard] Erreur : tb_launcher.py introuvable ({_launcher}).\n")
            return

        # --- 6. Kill any stale TB on port 6006, then launch fresh ---
        # Kill app-managed TB process reference
        _old_tb = getattr(self, '_tb_proc', None)
        if _old_tb and _old_tb.poll() is None:
            _old_tb.kill()
        self._tb_proc = None
        # Kill any external process still holding port 6006
        try:
            from src.core.runner import kill_process_on_port
            kill_process_on_port(6006)
        except Exception:
            pass
        import time as _t; _t.sleep(1.0)  # wait for OS to release port after kill

        import tempfile
        _err_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
        _err_path = _err_file.name
        _err_file.close()

        try:
            proc = subprocess.Popen(
                [py_exec, _launcher, tb_flag, tb_val, "--port", "6006"],
                stdout=subprocess.DEVNULL,
                stderr=open(_err_path, "w"),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self._tb_proc = proc  # store for future clean restarts
            self.append_log(f"[TensorBoard] Lance (PID {proc.pid})\n")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("TensorBoard", _t(f"Erreur lancement:\n{e}\n\nvenv Python:\n{py_exec}",
                                                    f"Launch error:\n{e}\n\nvenv Python:\n{py_exec}"))
            return

        # --- 7. Poll for port + conditionally open browser ---
        from src.core.settings import SettingsManager as _SM
        _auto_open_browser = _SM().get("tb_auto_open_browser", True)

        def _check_and_open():
            import socket, webbrowser as _wb
            for _ in range(24):  # up to 12s
                try:
                    with socket.create_connection(("127.0.0.1", 6006), timeout=0.5):
                        if _auto_open_browser:
                            _wb.open("http://localhost:6006")
                        else:
                            self.append_log("[TensorBoard] Prêt sur http://localhost:6006 "
                                            "(ouverture auto désactivée)\n")
                        return
                except OSError:
                    import time; time.sleep(0.5)
            # Timeout — read stderr
            try:
                with open(_err_path, "r") as _f:
                    err_txt = _f.read().strip()[-600:]
            except Exception:
                err_txt = "(impossible de lire les logs)"
            self.append_log(f"[TensorBoard] Echec démarrage. Erreur:\n{err_txt}\n")

        import threading
        threading.Thread(target=_check_and_open, daemon=True).start()

    def _show_servers_panel(self):
        """Unified popup: Gallery images server + TensorBoard data server."""
        import tkinter as tk, tempfile, os, threading, socket
        if getattr(self, '_servers_win', None) and self._servers_win.winfo_exists():
            self._servers_win.focus(); return

        # --- Resolve Gallery URL ---
        gal_url = ""
        try:
            from src.core.gallery_server import get_server
            srv = get_server()
            if srv and srv.httpd:
                gal_url = srv.ngrok_url or f"http://localhost:{srv.port}"
        except Exception:
            pass
        if not gal_url:
            gal_url = "http://localhost:8765"

        # --- Resolve TensorBoard URL ---
        tb_url = "http://localhost:6006"
        tb_alive = False
        try:
            with socket.create_connection(("127.0.0.1", 6006), timeout=0.3):
                tb_alive = True
        except OSError:
            pass

        win = ctk.CTkToplevel(self)
        self._servers_win = win
        win.title(_t("📡 Serveurs — Galerie & TensorBoard", "📡 Servers — Gallery & TensorBoard"))
        win.geometry("460x700")
        win.minsize(400, 500)
        win.resizable(True, True)
        win.grab_set()

        # Scrollable inner frame so both QR codes are always accessible
        scroll = ctk.CTkScrollableFrame(win, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        def _make_qr_widget(parent, url, label_text):
            """Build a QR block: label + image (or fallback text) + open button."""
            ctk.CTkLabel(parent, text=label_text, font=("Roboto", 10, "bold"),
                         text_color="#3498db", wraplength=400).pack(pady=(8, 2))
            ctk.CTkLabel(parent, text=url, font=("Roboto", 8),
                         text_color="#aaa", wraplength=400).pack(pady=(0, 4))
            qr_shown = False
            try:
                from src.core.qr_code import generate_qr_image, is_qrcode_available
                if is_qrcode_available():
                    tmp = tempfile.mktemp(suffix=".png")
                    if generate_qr_image(url, tmp, box_size=7):
                        from PIL import Image as _PImg, ImageTk as _ITk
                        img = _PImg.open(tmp).resize((210, 210), _PImg.LANCZOS)
                        photo = _ITk.PhotoImage(img)
                        lbl = tk.Label(parent, image=photo, bg="#1a1a2e")
                        lbl.image = photo
                        lbl.pack(pady=4)
                        try: os.remove(tmp)
                        except Exception: pass
                        qr_shown = True
            except Exception:
                pass
            if not qr_shown:
                ctk.CTkLabel(parent, text=url, font=("Roboto", 10, "bold"),
                             text_color="#2ecc71", wraplength=400).pack(pady=4)
            ctk.CTkButton(parent, text=_t("🌐 Ouvrir dans le navigateur", "🌐 Open in browser"), width=220, height=28,
                          command=lambda u=url: __import__("webbrowser").open(u)).pack(pady=(4, 2))

        # === Section A: Gallery ===
        fra = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=8)
        fra.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(fra, text=_t("🖼 Galerie Images", "🖼 Image Gallery"), font=("Roboto", 13, "bold"),
                     text_color="#2ecc71").pack(pady=(10, 0))
        _make_qr_widget(fra, gal_url, gal_url)

        def _restart_gallery():
            try:
                from src.core.gallery_server import get_server
                from src.core.settings import SettingsManager
                sm = SettingsManager()
                srv = get_server()
                gal_dir = sm.get("gallery_auto_dir", "")
                port = int(sm.get("gallery_port", 8765))
                with_ngrok = bool(sm.get("gallery_ngrok", True))
                if srv:
                    srv.stop()
                if gal_dir and os.path.isdir(gal_dir):
                    srv.start(gal_dir, port=port, with_ngrok=with_ngrok)
                self.append_log(_t("[Serveurs] Galerie images redémarrée.\n", "[Servers] Image gallery restarted.\n"))
                win.destroy()
                self.after(500, self._show_servers_panel)
            except Exception as e:
                self.append_log(f"[Serveurs] Erreur galerie: {e}\n")

        ctk.CTkButton(fra, text=_t("🔄 Redémarrer Galerie", "🔄 Restart Gallery"), width=190, height=26,
                      fg_color="#27ae60", command=_restart_gallery).pack(pady=(4, 10))

        # === Section B: TensorBoard ===
        frb = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=8)
        frb.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(frb, text="📈 TensorBoard Data", font=("Roboto", 13, "bold"),
                     text_color="#e67e22").pack(pady=(10, 0))
        if tb_alive:
            _make_qr_widget(frb, tb_url, tb_url)
        else:
            ctk.CTkLabel(frb, text=_t("TensorBoard non démarré — cliquez Lancer", "TensorBoard not started — click Launch"),
                         text_color="#888", font=("Roboto", 9)).pack(pady=6)

        def _launch_tb():
            win.destroy()
            self._open_tensorboard()

        ctk.CTkButton(frb, text=_t("▶ Lancer / Relancer TensorBoard", "▶ Launch / Restart TensorBoard"), width=230, height=28,
                      fg_color="#e67e22", command=_launch_tb).pack(pady=(4, 10))

        ctk.CTkButton(scroll, text=_t("Fermer", "Close"), width=120, fg_color="#444",
                      command=win.destroy).pack(pady=8)

    def on_finished(self):
        self.timer_running = False; self.btn_start.configure(state="normal", text=_t("▶  DÉMARRER", "▶  START")); self.btn_stop.configure(state="disabled")
        # Flush any buffered Redux record that never got a trailing prefix line
        if getattr(self, "_redux_buf_active", False) and self._redux_buf:
            self.append_log(self._flush_redux_buf() + "\n")
        self._reset_redux_buf()
        self.append_log("\n[INFO] Processus terminé.\n")
        # Auto-stop gallery if configured
        try:
            from src.core.settings import SettingsManager as _SM
            _sm = _SM()
            if _sm.get("gallery_auto_stop_with_training", False):
                from src.core.gallery_server import get_server
                _srv = get_server()
                if _srv and _srv.httpd:
                    _srv.stop()
                    self.append_log(_t("[Serveurs] Galerie auto-arrêtée.\n", "[Servers] Gallery auto-stopped.\n"))
        except Exception:
            pass
        self._play_finish_sound()

        # Clean up working temp config created for this training run
        tmp = getattr(self, "_training_tmp_config", None)
        if tmp:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            self._training_tmp_config = None

        # Update training history
        try:
            from src.core.training_history import record_training_end, update_training_progress
            row_id = getattr(self, "_history_row_id", 0)
            if row_id > 0:
                # Compute avg speed from accumulated samples
                samples = getattr(self, "_speed_samples", [])
                avg_spd = (sum(samples) / len(samples)) if samples else None

                # Save current_iter before finishing
                if self.current_iter > 0:
                    update_training_progress(row_id, current_iter=self.current_iter,
                                             avg_speed=avg_spd)

                # Determine status from the log
                log_text = self.textbox_logs.get("1.0", "end")[-2000:]
                if "End of training" in log_text or "training finished" in log_text.lower():
                    status = "completed"
                elif "[ERREUR]" in log_text or "Traceback" in log_text or "Error" in log_text:
                    status = "failed"
                else:
                    status = "interrupted"
                vram_pk = getattr(self, "_vram_peak_mb", 0)
                _pwr_s = getattr(self, "_power_samples", [])
                pwr_avg = sum(_pwr_s) / len(_pwr_s) if _pwr_s else 0.0
                record_training_end(row_id, status=status, avg_speed=avg_spd,
                                    vram_peak_mb=vram_pk, power_avg_w=pwr_avg)
                self._vram_peak_mb = 0
                self._power_samples = []
        except Exception:
            pass

        # Show Windows 11 toast notification
        try:
            from src.core.toast_notifications import show_training_complete_toast
            duration_str = ""
            if hasattr(self, "start_time") and self.start_time:
                elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
                hours = int(elapsed // 3600)
                mins = int((elapsed % 3600) // 60)
                duration_str = f"{hours}h {mins}min"
            model_name = "Training"
            try:
                cf = self.entries_dict["config_path"].get()
                if cf:
                    model_name = os.path.splitext(os.path.basename(cf))[0]
            except Exception:
                pass
            # Resolve icon reliably from project root (not CWD which can vary)
            # Prefer .png over .ico — win11toast handles PNG better for appLogoOverride
            _proj_root = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
            )
            icon_path = None
            for _icon in ("assets/icon.png", "assets/icon.ico"):
                _p = os.path.join(_proj_root, _icon)
                if os.path.exists(_p):
                    icon_path = _p
                    break
            show_training_complete_toast(
                model_name=model_name,
                duration_str=duration_str or "?",
                best_psnr=getattr(self, "best_psnr", None),
                icon_path=icon_path,
            )
        except Exception:
            pass

        # Reset taskbar progress
        try:
            hwnd = self.winfo_toplevel().winfo_id()
            set_taskbar_progress(hwnd, 0, "none")
        except Exception:
            pass

        # ─── PIPELINE: Auto-launch next phase ───
        if self.pipeline_mode and "phases" in self.pipeline_mode:
            phases = self.pipeline_mode["phases"]
            # Find which phase just completed
            current_idx = self.pipeline_mode.get("_current_phase", 0)
            next_idx = current_idx + 1

            if next_idx < len(phases):
                next_phase = phases[next_idx]
                self.append_log(f"\n{'='*60}\n")
                self.append_log(f"[PIPELINE] Phase {current_idx+1} ({phases[current_idx]['name']}) terminée !\n")
                self.append_log(f"[PIPELINE] Lancement Phase {next_idx+1} ({next_phase['name']})...\n")
                self.append_log(f"{'='*60}\n\n")

                self.pipeline_mode["_current_phase"] = next_idx

                # Generate Phase 2 config with pretrain from Phase 1
                try:
                    self._launch_pipeline_phase(next_phase)
                    return  # Don't show shutdown dialog
                except Exception as e:
                    self.append_log(f"[PIPELINE] Erreur lancement phase {next_idx+1}: {e}\n")
            else:
                self.append_log(f"\n{'='*60}\n")
                self.append_log(f"[PIPELINE] Toutes les phases sont terminées !\n")
                self.append_log(f"{'='*60}\n")
                self.pipeline_mode = None

        if self.chk_shutdown.get():
            confirm = messagebox.askyesno(
                _t("Arrêt automatique", "Auto shutdown"),
                _t("Le training est terminé.\n\nÉteindre le PC dans 60 secondes ?\n\n(Cliquez 'Non' pour annuler)",
                   "Training is complete.\n\nShut down PC in 60 seconds?\n\n(Click 'No' to cancel)"),
                icon="warning"
            )
            if confirm:
                self.append_log(_t("\n[ATTENTION] ARRÊT AUTOMATIQUE DANS 60 SECONDES...\n",
                                   "\n[WARNING] AUTO SHUTDOWN IN 60 SECONDS...\n"))
                if sys.platform == "win32":
                    os.system("shutdown /s /t 60")
                else:
                    os.system("shutdown -h +1")  # Linux/Mac
            else:
                self.append_log(_t("[INFO] Arrêt automatique annulé par l'utilisateur.\n",
                                   "[INFO] Auto shutdown cancelled by user.\n"))

    def _launch_pipeline_phase(self, phase: dict):
        """Generate config for a pipeline phase and launch it."""
        data = phase.get("config_data", {})
        engine = self.pipeline_mode.get("engine", "NeoSR")

        # If this is the GAN phase, find the PSNR pretrain
        if data.get("_psnr_pretrain"):
            prev_name = self.pipeline_mode["phases"][0]["config_data"].get("name", "exp_PSNR")
            exp_dir = os.path.join(os.path.expanduser("~"), "IA_Engine", engine.lower().replace("trainner-", ""), "experiments", prev_name, "models")
            # Try to find the latest .pth in the experiments folder
            pretrain = ""
            if os.path.exists(exp_dir):
                pths = sorted([f for f in os.listdir(exp_dir) if f.endswith((".pth", ".safetensors"))], reverse=True)
                if pths:
                    pretrain = os.path.join(exp_dir, pths[0])
                    self.append_log(f"[PIPELINE] Pretrain trouvé : {pretrain}\n")
            if pretrain:
                data["pretrain_g"] = pretrain
            del data["_psnr_pretrain"]

        # Generate config
        from src.core.config_handler import ConfigHandler
        handler = ConfigHandler()
        handler.set_engine(engine)

        base_dir = os.path.join(os.path.expanduser("~"), "IA_Engine", "Option Custom")
        sub = "trainner_redux" if "Redux" in engine else "neosr"
        target_dir = os.path.join(base_dir, sub)
        os.makedirs(target_dir, exist_ok=True)

        ext = ".yml" if "Redux" in engine else ".toml"
        config_path = os.path.join(target_dir, f"{data.get('name', 'phase')}{ext}")
        ok, msg = handler.generate_config(data, config_path)

        if ok:
            self.after(1000, lambda: self.external_start(config_path))
        else:
            self.append_log(f"[PIPELINE] Erreur config : {msg}\n")

    def setup_log_tags(self):
        try:
            tb = self.textbox_logs._textbox
            tb.tag_config("green",  foreground="#2ecc71")
            tb.tag_config("red",    foreground="#e74c3c")
            tb.tag_config("yellow", foreground="#f1c40f")
            tb.tag_config("cyan",   foreground="#3498db")
            tb.tag_config("purple", foreground="#9b59b6")
            tb.tag_config("orange", foreground="#e67e22")  # epoch / iter
            tb.tag_config("brown",  foreground="#c0874e")  # learning rate
            tb.tag_config("pink",   foreground="#ff69b4")  # grad norms
        except Exception: pass

    def clear_logs(self):
        self.textbox_logs.configure(state="normal"); self.textbox_logs.delete("0.0", "end"); self.textbox_logs.configure(state="disabled")
        self._reset_redux_buf()

    def show_validation_preview(self):
        """Show the latest validation SR image alongside its LQ input."""
        if getattr(self, '_val_win', None) and self._val_win.winfo_exists():
            self._val_win.focus(); return
        try:
            from PIL import Image, ImageTk
        except ImportError:
            messagebox.showerror(_t("Erreur", "Error"), _t("Pillow (PIL) requis.", "Pillow (PIL) is required."))
            return

        # Find experiment folder from config path
        config_path = self.entries_dict.get("config_path")
        if config_path:
            config_path = config_path.get().strip()

        exp_name = ""
        if config_path and os.path.isfile(config_path):
            try:
                with open(config_path, "r") as cf:
                    content = cf.read()
                if config_path.endswith(".toml"):
                    try:
                        import tomllib
                        data = tomllib.loads(content)
                    except Exception:
                        m = re.search(r'name\s*=\s*["\'](.*?)["\']', content)
                        data = {"name": m.group(1)} if m else {}
                else:
                    data = yaml.safe_load(content) or {}
                exp_name = data.get("name", "")
                # Try to extract val_lq path from config — used to match LQ images
                try:
                    _datasets = data.get("datasets", {})
                    # Redux YAML: datasets is a LIST [{name: "val", dataroot_lq: ...}, ...]
                    # NeoSR TOML: datasets is a DICT {val: {dataroot_lq: ...}}
                    if isinstance(_datasets, list):
                        _val_ds = next(
                            (d for d in _datasets
                             if isinstance(d, dict) and str(d.get("name", "")).lower().startswith("val")),
                            {}
                        )
                    elif isinstance(_datasets, dict):
                        _val_ds = _datasets.get("val", _datasets.get("val_1", {}))
                    else:
                        _val_ds = {}
                    _val_lq_dir = _val_ds.get("dataroot_lq", _val_ds.get("lq_path", ""))
                    if not _val_lq_dir:
                        # NeoSR flat TOML: search for val.dataroot_lq
                        _val_lq_dir = data.get("dataroot_lq", "")
                    # Redux YAML: dataroot_lq est une LIST ['path'] → extraire le premier élément
                    if isinstance(_val_lq_dir, list):
                        _val_lq_dir = _val_lq_dir[0] if _val_lq_dir else ""
                    self._val_lq_dataset_dir = str(_val_lq_dir).strip() if _val_lq_dir else ""
                except Exception:
                    self._val_lq_dataset_dir = ""
            except Exception:
                pass

        # Search for validation images
        base_paths = []
        for engine_dir in ["neosr", "traiNNer-redux"]:
            exp_root = os.path.join(os.path.expanduser("~"), "IA_Engine", engine_dir, "experiments")
            if not os.path.isdir(exp_root):
                continue
            dirs = sorted(os.listdir(exp_root), key=lambda d: os.path.getmtime(os.path.join(exp_root, d)) if os.path.isdir(os.path.join(exp_root, d)) else 0, reverse=True)
            for d in dirs:
                if exp_name and exp_name in d:
                    base_paths.insert(0, os.path.join(exp_root, d, "visualization"))
                else:
                    base_paths.append(os.path.join(exp_root, d, "visualization"))

        sr_image = lq_image = None
        found_exp = ""
        for viz_dir in base_paths:
            if not os.path.isdir(viz_dir):
                continue
            subdirs = [viz_dir]
            for sd in sorted(os.listdir(viz_dir), reverse=True):
                full = os.path.join(viz_dir, sd)
                if os.path.isdir(full):
                    subdirs.insert(0, full)

            def _iter_key(fname):
                """Extrait le numéro d'itération pour tri numérique correct."""
                _m = re.search(r'_(\d+)\.\w+$', fname)
                return int(_m.group(1)) if _m else 0

            for sd in subdirs:
                imgs = sorted(
                    [f for f in os.listdir(sd) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
                    key=_iter_key, reverse=True)  # tri numérique, pas lexicographique
                for img_name in imgs:
                    img_path = os.path.join(sd, img_name)
                    name_lower = img_name.lower()
                    if sr_image is None and ("_sr" in name_lower or (not "_lq" in name_lower and not "_gt" in name_lower and not "input" in name_lower)):
                        sr_image = img_path
                    if lq_image is None and ("_lq" in name_lower or "input" in name_lower):
                        lq_image = img_path
                if sr_image:
                    found_exp = os.path.basename(os.path.dirname(viz_dir))
                    break
            if sr_image:
                break

        if not sr_image:
            messagebox.showinfo(_t("Validation", "Validation"),
                                _t("Aucune image de validation trouvee.\nVerifiez experiments/.../visualization/",
                                   "No validation image found.\nCheck experiments/.../visualization/"))
            return

        # Collect ALL validation images for the selector
        # NeoSR naming: {name}_{iter}.png (SR), {name}_lq.png (LQ)
        all_val_images = {}  # base_name -> {"sr": path, "lq": path}
        for viz_dir in base_paths:
            if not os.path.isdir(viz_dir):
                continue
            all_files = []
            for root_d, dirs, files in os.walk(viz_dir):
                for fname in files:
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        all_files.append((fname, os.path.join(root_d, fname)))

            # First pass: find all LQ files to identify base names
            # NeoSR: {name}_lq.png / {name}_input.png
            # Redux: {name}_lr.png  (LQ Low-Resolution input)
            lq_names = set()
            for fname, fpath in all_files:
                fl = fname.lower()
                stem_raw = fname.rsplit('.', 1)[0]
                stem_lo = stem_raw.lower()
                if "_lq" in fl or "_input" in fl:
                    # Base name is everything before _lq / _input
                    base = re.sub(r'_lq\..*$', '', fname, flags=re.IGNORECASE)
                    base = re.sub(r'_input\..*$', '', base, flags=re.IGNORECASE)
                    lq_names.add(base)
                    if base not in all_val_images:
                        all_val_images[base] = {"sr": None, "lq": None}
                    all_val_images[base]["lq"] = fpath
                elif stem_lo.endswith('_lr'):
                    # Redux naming convention: {name}_lr.png is the LQ input
                    base = stem_raw[:-3]  # strip '_lr'
                    lq_names.add(base)
                    if base not in all_val_images:
                        all_val_images[base] = {"sr": None, "lq": None}
                    all_val_images[base]["lq"] = fpath

            # Second pass: match SR files (name_ITER.png) to their base
            for fname, fpath in all_files:
                fl = fname.lower()
                stem = fname.rsplit('.', 1)[0]
                if "_lq" in fl or "_input" in fl or stem.lower().endswith('_lr'):
                    continue
                # Try to match: base_DIGITS pattern
                m = re.match(r'^(.+?)_(\d+)$', stem)
                if m:
                    base = m.group(1)
                    iter_num = int(m.group(2))
                    if base in all_val_images:
                        # Keep the latest iteration — compare NUMÉRIQUEMENT (pas lexicographique)
                        # Bug: "5000" > "10000" lexicographiquement car '5' > '1'
                        existing = all_val_images[base].get("sr", "") or ""
                        ex_m = re.search(r'_(\d+)\.\w+$', os.path.basename(existing))
                        existing_iter = int(ex_m.group(1)) if ex_m else -1
                        if not existing or iter_num > existing_iter:
                            all_val_images[base]["sr"] = fpath
                    else:
                        all_val_images[base] = {"sr": fpath, "lq": None}
                else:
                    # No iteration suffix — use stem as-is
                    if stem not in all_val_images:
                        all_val_images[stem] = {"sr": fpath, "lq": None}
                    elif all_val_images[stem]["sr"] is None:
                        all_val_images[stem]["sr"] = fpath

            if all_val_images:
                break

        # Fallback: match LQ from val_lq dataset folder (config val.dataroot_lq)
        # NeoSR typically doesn't save LQ to visualization — LQ lives in the dataset folder
        _lq_ds_dir = getattr(self, '_val_lq_dataset_dir', '')
        if _lq_ds_dir and os.path.isdir(_lq_ds_dir):
            _lq_exts = {".png", ".jpg", ".jpeg", ".webp"}
            _lq_by_stem = {}  # stem_lower → full_path
            try:
                for _lqf in os.listdir(_lq_ds_dir):
                    _stem, _ext = os.path.splitext(_lqf)
                    if _ext.lower() in _lq_exts:
                        _lq_by_stem[_stem.lower()] = os.path.join(_lq_ds_dir, _lqf)
            except Exception:
                pass
            for _base in list(all_val_images.keys()):
                if all_val_images[_base].get("lq"):
                    continue  # already found
                # Try exact match first, then prefix match (base may have extra suffix)
                _bl = _base.lower()
                if _bl in _lq_by_stem:
                    all_val_images[_base]["lq"] = _lq_by_stem[_bl]
                else:
                    # Try: stem starts with base (e.g. base="tsurune084", file="tsurune084_01.png")
                    for _stem_l, _fp in _lq_by_stem.items():
                        if _stem_l.startswith(_bl) or _bl.startswith(_stem_l):
                            all_val_images[_base]["lq"] = _fp
                            break

        # Only show entries that have an SR image — skip LQ-only entries (e.g. Redux _lr files
        # that weren't matched to any SR because training hasn't produced them yet)
        val_names = sorted(
            k for k, v in all_val_images.items() if v.get("sr")
        ) if all_val_images else ["default"]
        if not val_names:
            val_names = sorted(all_val_images.keys()) or ["default"]

        # Rafraîchir sr_image / lq_image depuis all_val_images qui inclut le dataset LQ folder
        # Fix : à l'ouverture, pil_lq était None même si all_val_images l'avait trouvé
        _init_name = next(
            (n for n in val_names if all_val_images.get(n, {}).get("sr")),
            val_names[0] if val_names else None
        )
        if _init_name and _init_name in all_val_images:
            _entry = all_val_images[_init_name]
            if _entry.get("sr"):
                sr_image = _entry["sr"]
            if _entry.get("lq"):
                lq_image = _entry["lq"]

        # Create preview window
        win = ctk.CTkToplevel(self)
        win.title(_t("Prévisualisation Validation", "Validation Preview"))
        win.geometry("1200x700")
        win.attributes("-topmost", True)
        win.after(500, lambda: win.attributes("-topmost", False))

        # Header with model info + image selector
        header = ctk.CTkFrame(win, fg_color="transparent", height=30)
        header.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(header, text=f"{_t('Modele', 'Model')}: {found_exp}", font=("Roboto", 12, "bold"),
                     text_color="#3498db").pack(side="left")

        # Image selector dropdown
        if len(val_names) > 1:
            ctk.CTkLabel(header, text=f"  {_t('Image', 'Image')}:", text_color=("gray30", "#AAA")).pack(side="left", padx=(15, 3))
            self._val_all_images = all_val_images
            self._val_selector = ctk.CTkOptionMenu(header, values=val_names, width=180,
                command=lambda name: self._on_val_image_select(name, win))
            self._val_selector.pack(side="left", padx=3)
            # Set to first name that has sr
            for n in val_names:
                if all_val_images.get(n, {}).get("sr"):
                    self._val_selector.set(n)
                    break

        # Mode toggle + auto-reload
        import tkinter as tk
        self._val_compare_mode = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(header, text=_t("Comparaison par balayage", "Swipe comparison"), variable=self._val_compare_mode,
                        command=lambda: self._toggle_val_mode(win)).pack(side="right", padx=10)
        self._val_auto_reload_var = tk.BooleanVar(value=False)
        _ar_cb = ctk.CTkCheckBox(header, text=_t("Auto-reload", "Auto-reload"), variable=self._val_auto_reload_var,
                                  font=("Arial", 13))
        _ar_cb.pack(side="right", padx=(0, 4))
        from src.ui.components.tooltip import ToolTip as _TT
        _TT(_ar_cb, _t("Recharge automatiquement la dernière image\nà chaque fin de validation.", "Automatically reloads the latest image\nafter each validation pass."))

        # Load images
        pil_sr = Image.open(sr_image)
        pil_lq = Image.open(lq_image) if lq_image and os.path.exists(lq_image) else None

        # Store for resize
        self._val_pil_sr = pil_sr
        self._val_pil_lq = pil_lq
        self._val_win = win

        # Side by side frame (default mode)
        self._val_side_frame = ctk.CTkFrame(win, fg_color="transparent")
        self._val_side_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Use a canvas for each image so they scale with window
        import tkinter as tk
        self._val_canvas_lq = tk.Canvas(self._val_side_frame, bg="#2B2B2B", highlightthickness=0)
        self._val_canvas_sr = tk.Canvas(self._val_side_frame, bg="#2B2B2B", highlightthickness=0)

        # Labels and canvas layout — stored refs for dynamic re-grid on image switch
        self._val_lbl_lq = ctk.CTkLabel(self._val_side_frame, text=_t("LQ (Entrée)", "LQ (Input)"),
                                        font=("Roboto", 11, "bold"), text_color="#e67e22")
        self._val_lbl_sr = ctk.CTkLabel(self._val_side_frame, text=_t("SR (Sortie)", "SR (Output)"),
                                        font=("Roboto", 11, "bold"), text_color="#2ecc71")

        def _apply_side_grid(has_lq: bool):
            """Re-arrange label/canvas grid depending on whether LQ exists."""
            self._val_lbl_lq.grid_forget()
            self._val_lbl_sr.grid_forget()
            self._val_canvas_lq.grid_forget()
            self._val_canvas_sr.grid_forget()
            if has_lq:
                self._val_lbl_lq.grid(row=0, column=0)
                self._val_canvas_lq.grid(row=1, column=0, sticky="nsew", padx=3)
                self._val_lbl_sr.grid(row=0, column=1)
                self._val_canvas_sr.grid(row=1, column=1, sticky="nsew", padx=3)
                self._val_side_frame.grid_columnconfigure(0, weight=1)
                self._val_side_frame.grid_columnconfigure(1, weight=1)
            else:
                self._val_lbl_sr.grid(row=0, column=0, columnspan=2)
                self._val_canvas_sr.grid(row=1, column=0, sticky="nsew", padx=3, columnspan=2)
                self._val_side_frame.grid_columnconfigure(0, weight=1)
                self._val_side_frame.grid_columnconfigure(1, weight=0)

        self._apply_side_grid = _apply_side_grid
        _apply_side_grid(pil_lq is not None)
        self._val_side_frame.grid_rowconfigure(1, weight=1)

        # Slider compare frame (hidden initially)
        self._val_slider_frame = ctk.CTkFrame(win, fg_color="transparent")
        self._val_slider_canvas = tk.Canvas(self._val_slider_frame, bg="#2B2B2B", highlightthickness=0)
        self._val_slider_canvas.pack(fill="both", expand=True)
        self._val_slider_x = 0

        self._val_slider_pending = False

        def _do_slider_draw():
            self._val_slider_pending = False
            self._draw_val_slider()

        def _on_slider_move(event):
            self._val_slider_x = event.x
            # Throttle: max ~30fps (33ms) to avoid lag during fast mouse sweeps
            if not self._val_slider_pending:
                self._val_slider_pending = True
                win.after(33, _do_slider_draw)

        self._val_slider_canvas.bind("<Motion>", _on_slider_move)

        # Refs to prevent GC
        self._val_tk_refs = {}
        self._val_zoom = 1.0

        # Per-key cache: {key: (nw, nh, tk_img)} — skip resize if size unchanged
        self._val_side_cache = {}

        def _resize_side(event=None):
            zoom = self._val_zoom
            for canvas, pil_img, key in [
                    (self._val_canvas_lq, self._val_pil_lq, "lq"),
                    (self._val_canvas_sr, self._val_pil_sr, "sr")]:
                if pil_img is None:
                    canvas.delete("all")
                    continue
                w = canvas.winfo_width()
                h = canvas.winfo_height()
                if w < 10 or h < 10:
                    continue
                iw, ih = pil_img.size
                ratio = min(w / iw, h / ih) * zoom
                # Cap output size to 2× canvas — no point rendering beyond screen pixels
                nw = min(max(int(iw * ratio), 1), w * 2)
                nh = min(max(int(ih * ratio), 1), h * 2)
                cached = self._val_side_cache.get(key)
                if cached and cached[0] == nw and cached[1] == nh:
                    tk_img = cached[2]
                else:
                    resized = pil_img.resize((nw, nh), Image.BILINEAR)
                    tk_img = ImageTk.PhotoImage(resized)
                    self._val_side_cache[key] = (nw, nh, tk_img)
                self._val_tk_refs[key] = tk_img
                canvas.delete("all")
                canvas.create_image(w // 2, h // 2, image=tk_img, anchor="center")

        self._val_zoom_after_id = None
        def _on_zoom(event):
            if event.delta > 0 or event.num == 4:
                self._val_zoom = min(self._val_zoom * 1.15, 3.0)
            else:
                self._val_zoom = max(self._val_zoom / 1.15, 0.2)
            # Invalidate caches on zoom change
            self._val_side_cache.clear()
            self._val_slider_cache_size = None
            # Throttle redraws to ~30fps
            if self._val_zoom_after_id:
                win.after_cancel(self._val_zoom_after_id)
            if hasattr(self, '_val_compare_mode') and self._val_compare_mode.get():
                self._val_zoom_after_id = win.after(33, self._draw_val_slider)
            else:
                self._val_zoom_after_id = win.after(33, _resize_side)

        self._val_resize_pending = False
        def _resize_side_debounced(event=None):
            if not self._val_resize_pending:
                self._val_resize_pending = True
                win.after(60, lambda: [setattr(self, '_val_resize_pending', False), _resize_side()])

        self._val_canvas_lq.bind("<Configure>", _resize_side_debounced)
        self._val_canvas_sr.bind("<Configure>", _resize_side_debounced)
        # Zoom bindings (Windows + Linux)
        for c in [self._val_canvas_lq, self._val_canvas_sr, self._val_slider_canvas]:
            c.bind("<MouseWheel>", _on_zoom)
            c.bind("<Button-4>", _on_zoom)
            c.bind("<Button-5>", _on_zoom)
        self._val_resize_side = _resize_side

        def _resize_slider(event=None):
            self._draw_val_slider()
        self._val_slider_canvas.bind("<Configure>", _resize_slider)

        # Force initial draw after window layout completes (canvas size may be 0 at bind time)
        win.after(120, _resize_side)

        # Info bar — stocker la référence pour before= stable dans _toggle_val_mode
        info = f"SR: {os.path.basename(sr_image)}"
        if lq_image:
            info += f"  |  LQ: {os.path.basename(lq_image)}"
        self._val_info_label = ctk.CTkLabel(win, text=info, text_color="#888", font=("Roboto", 9))
        self._val_info_label.pack(pady=(0, 2))
        ctk.CTkButton(win, text=_t("Fermer", "Close"), width=100, fg_color="#666", command=win.destroy).pack(pady=(0, 8))

    def _draw_val_slider(self):
        """Draw slider comparison in validation preview."""
        from PIL import Image, ImageTk
        canvas = self._val_slider_canvas
        pil_sr = self._val_pil_sr
        pil_lq = self._val_pil_lq
        if not pil_sr:
            return
        if not pil_lq:
            # Pas de LQ : afficher SR seul sans split
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 10 or h < 10:
                return
            iw, ih = pil_sr.size
            zoom = getattr(self, '_val_zoom', 1.0)
            ratio = min(w / iw, h / ih) * zoom
            nw, nh = max(int(iw * ratio), 1), max(int(ih * ratio), 1)
            img = pil_sr.resize((nw, nh), Image.LANCZOS)
            self._val_tk_slider = ImageTk.PhotoImage(img)
            canvas.delete("all")
            canvas.create_image(w // 2, h // 2, image=self._val_tk_slider, anchor="center")
            canvas.create_text(w // 2, 10, text=_t("SR uniquement (pas d'image LQ)", "SR only (no LQ image)"), fill="#888",
                               font=("Roboto", 9), anchor="n")
            return
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10:
            return
        iw, ih = pil_sr.size
        zoom = getattr(self, '_val_zoom', 1.0)
        ratio = min(w / iw, h / ih) * zoom
        nw = min(max(int(iw * ratio), 1), w * 2)
        nh = min(max(int(ih * ratio), 1), h * 2)
        # Cache resized images — only re-resize when zoom/window changes, not on every mouse move
        cache_key = (nw, nh)
        if getattr(self, '_val_slider_cache_size', None) != cache_key:
            self._val_slider_cache_sr = pil_sr.resize((nw, nh), Image.BILINEAR)
            self._val_slider_cache_lq = pil_lq.resize((nw, nh), Image.BILINEAR)
            self._val_slider_cache_size = cache_key
        resized_sr = self._val_slider_cache_sr
        resized_lq = self._val_slider_cache_lq
        pos_x = (w - nw) // 2
        pos_y = (h - nh) // 2
        split = max(0, min(self._val_slider_x - pos_x, nw))
        composite = resized_sr.copy()
        if split > 0:
            left = resized_lq.crop((0, 0, split, nh))
            composite.paste(left, (0, 0))
        if hasattr(self, "_val_tk_slider"):
            del self._val_tk_slider
        self._val_tk_slider = ImageTk.PhotoImage(composite)
        canvas.delete("all")
        canvas.create_image(pos_x, pos_y, image=self._val_tk_slider, anchor="nw")
        canvas.create_line(self._val_slider_x, pos_y, self._val_slider_x, pos_y + nh, fill="#e74c3c", width=2)
        canvas.create_text(pos_x + 5, pos_y + 5, text="LQ", fill="#e67e22", anchor="nw", font=("Roboto", 10, "bold"))
        canvas.create_text(pos_x + nw - 5, pos_y + 5, text="SR", fill="#2ecc71", anchor="ne", font=("Roboto", 10, "bold"))

    def _toggle_val_mode(self, win):
        """Toggle between side-by-side and slider comparison."""
        # Utiliser la référence stockée plutôt que win.winfo_children()[-2] (fragile)
        anchor = getattr(self, "_val_info_label", None)
        if self._val_compare_mode.get():
            self._val_side_frame.pack_forget()
            if anchor:
                self._val_slider_frame.pack(fill="both", expand=True, padx=10, pady=5, before=anchor)
            else:
                self._val_slider_frame.pack(fill="both", expand=True, padx=10, pady=5)
            w = self._val_slider_canvas.winfo_width()
            self._val_slider_x = w // 2 if w > 10 else 400
            win.after(150, self._draw_val_slider)
        else:
            self._val_slider_frame.pack_forget()
            if anchor:
                self._val_side_frame.pack(fill="both", expand=True, padx=10, pady=5, before=anchor)
            else:
                self._val_side_frame.pack(fill="both", expand=True, padx=10, pady=5)
            win.after(100, self._val_resize_side)

    def _on_val_image_select(self, name, win):
        """Load a specific validation image pair by name."""
        try:
            from PIL import Image, ImageTk
            pair = self._val_all_images.get(name, {})
            sr_path = pair.get("sr")
            lq_path = pair.get("lq")
            if sr_path and os.path.exists(sr_path):
                self._val_pil_sr = Image.open(sr_path)
            else:
                return  # No SR image found
            if lq_path and os.path.exists(lq_path):
                self._val_pil_lq = Image.open(lq_path)
            else:
                self._val_pil_lq = None
            self._val_zoom = 1.0
            self._val_side_cache = {}
            self._val_slider_cache_size = None

            # Re-grid side view if LQ availability changed
            if hasattr(self, '_apply_side_grid'):
                self._apply_side_grid(self._val_pil_lq is not None)

            # Update info bar
            info = f"SR: {os.path.basename(sr_path)}"
            if lq_path and os.path.exists(lq_path):
                info += f"  |  LQ: {os.path.basename(lq_path)}"
            else:
                info += _t("  (pas d'image LQ)", "  (no LQ image)")
            if hasattr(self, '_val_info_label'):
                try: self._val_info_label.configure(text=info)
                except Exception: pass

            # Redraw both modes
            if hasattr(self, '_val_compare_mode') and self._val_compare_mode.get():
                self._val_slider_x = self._val_slider_canvas.winfo_width() // 2
                self._draw_val_slider()
            elif hasattr(self, '_val_resize_side'):
                win.after(50, self._val_resize_side)
        except Exception as e:
            self.append_log(f"[Validation] Erreur chargement {name}: {e}")

    def _auto_reload_val_preview(self):
        """Rescan visualization folder and reload the latest SR image in the preview window."""
        try:
            if not (getattr(self, '_val_win', None) and self._val_win.winfo_exists()):
                return
            # Rescan all visualization folders to find newest SR image
            config_path = getattr(self, '_current_config_path', '') or ''
            exp_name = ""
            if config_path and os.path.isfile(config_path):
                try:
                    import yaml
                    if config_path.endswith((".yml", ".yaml")):
                        with open(config_path, "r", encoding="utf-8") as f:
                            _d = yaml.safe_load(f) or {}
                    else:
                        import tomllib
                        with open(config_path, "rb") as f:
                            _d = tomllib.load(f)
                    exp_name = _d.get("name", "")
                except Exception:
                    pass

            # Find latest SR image across all experiments
            best_mtime = -1
            best_base = None
            best_viz_dir = None
            for engine_dir in ["neosr", "traiNNer-redux"]:
                exp_root = os.path.join(os.path.expanduser("~"), "IA_Engine", engine_dir, "experiments")
                if not os.path.isdir(exp_root):
                    continue
                for d in os.listdir(exp_root):
                    if exp_name and exp_name not in d:
                        continue
                    viz = os.path.join(exp_root, d, "visualization")
                    if not os.path.isdir(viz):
                        continue
                    for root_d, _, files in os.walk(viz):
                        for fn in files:
                            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                                fp = os.path.join(root_d, fn)
                                stem = fn.rsplit('.', 1)[0]
                                if stem.lower().endswith('_lr') or '_lq' in fn.lower():
                                    continue
                                mt = os.path.getmtime(fp)
                                if mt > best_mtime:
                                    best_mtime = mt
                                    best_base = re.sub(r'_\d+$', '', stem)
                                    best_viz_dir = viz

            if best_base and best_base in getattr(self, '_val_all_images', {}):
                if hasattr(self, '_val_selector'):
                    try: self._val_selector.set(best_base)
                    except Exception: pass
                win = self._val_win
                self._on_val_image_select(best_base, win)
        except Exception:
            pass

    # ───────────────────────────────────────────────────────────────────────────

    def _show_lr_schedule(self):
        """Show visual LR schedule curve."""
        import tkinter as tk
        if getattr(self, '_lr_win', None) and self._lr_win.winfo_exists():
            self._lr_win.focus(); return
        win = ctk.CTkToplevel(self)
        self._lr_win = win
        win.title(_t("Scheduler LR - Visualisation", "LR Scheduler - Visualization"))
        win.geometry("700x400")
        win.attributes("-topmost", True)
        win.after(500, lambda: win.attributes("-topmost", False))

        canvas = tk.Canvas(win, bg="#1a1a2e", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Read config
        try:
            lr_start = float(self.card_lr.cget("text").strip() or "2e-4")
        except Exception:
            lr_start = 2e-4
        total_iter = max(self.target_iters, 1000)

        import math
        # Simulate MultiStepLR (default) with milestones at 50%, 75%, 90%
        milestones = [int(total_iter * 0.5), int(total_iter * 0.75), int(total_iter * 0.9)]
        gamma = 0.5

        def get_lr(step):
            lr = lr_start
            for m in milestones:
                if step >= m:
                    lr *= gamma
            return lr

        # Also compute CosineAnnealing for comparison
        def get_lr_cosine(step):
            return lr_start * 0.5 * (1 + math.cos(math.pi * step / total_iter))

        n_points = 200

        def draw(event=None):
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 50 or h < 50:
                return
            canvas.delete("all")
            margin_l, margin_r, margin_t, margin_b = 70, 20, 30, 40
            gw = w - margin_l - margin_r
            gh = h - margin_t - margin_b

            # Title
            canvas.create_text(w // 2, 12, text=_t("Évolution du Learning Rate", "Learning Rate Schedule"), fill="#3498db", font=("Roboto", 12, "bold"))

            # Axes
            canvas.create_line(margin_l, margin_t, margin_l, h - margin_b, fill="#555")
            canvas.create_line(margin_l, h - margin_b, w - margin_r, h - margin_b, fill="#555")

            # Y axis labels
            canvas.create_text(margin_l - 5, margin_t, text=f"{lr_start:.1e}", fill="#AAA", anchor="e", font=("Roboto", 8))
            canvas.create_text(margin_l - 5, h - margin_b, text="0", fill="#AAA", anchor="e", font=("Roboto", 8))

            # X axis labels
            canvas.create_text(margin_l, h - margin_b + 12, text="0", fill="#AAA", font=("Roboto", 8))
            canvas.create_text(w - margin_r, h - margin_b + 12, text=f"{total_iter//1000}K", fill="#AAA", font=("Roboto", 8))
            canvas.create_text(w // 2, h - margin_b + 25, text=_t("Itérations", "Iterations"), fill="#888", font=("Roboto", 9))

            # Plot MultiStepLR
            pts_step = []
            pts_cos = []
            for i in range(n_points + 1):
                step = int(i / n_points * total_iter)
                x = margin_l + (i / n_points) * gw
                # MultiStep
                lr_v = get_lr(step)
                y = margin_t + gh - (lr_v / lr_start) * gh
                pts_step.extend([x, y])
                # Cosine
                lr_c = get_lr_cosine(step)
                y2 = margin_t + gh - (lr_c / lr_start) * gh
                pts_cos.extend([x, y2])

            if len(pts_step) >= 4:
                canvas.create_line(*pts_step, fill="#2ecc71", width=2)
            if len(pts_cos) >= 4:
                canvas.create_line(*pts_cos, fill="#e67e22", width=2, dash=(5, 3))

            # Legend
            canvas.create_line(w - 180, margin_t + 5, w - 160, margin_t + 5, fill="#2ecc71", width=2)
            canvas.create_text(w - 155, margin_t + 5, text="MultiStepLR", fill="#2ecc71", anchor="w", font=("Roboto", 9))
            canvas.create_line(w - 180, margin_t + 20, w - 160, margin_t + 20, fill="#e67e22", width=2, dash=(5, 3))
            canvas.create_text(w - 155, margin_t + 20, text="CosineAnnealing", fill="#e67e22", anchor="w", font=("Roboto", 9))

            # Milestone markers
            for m in milestones:
                mx = margin_l + (m / total_iter) * gw
                canvas.create_line(mx, margin_t, mx, h - margin_b, fill="#444", dash=(2, 4))
                canvas.create_text(mx, h - margin_b + 12, text=f"{m//1000}K", fill="#666", font=("Roboto", 7))

        canvas.bind("<Configure>", draw)
        ctk.CTkButton(win, text=_t("Fermer", "Close"), width=100, fg_color="#666", command=win.destroy).pack(pady=5)

    def parse_metrics(self, line):
        # Redux logs via rich are wrapped onto MANY lines. The actual record looks like:
        #   [05/08/26 16:44:32] INFO    [epoch:    0, iter:     700,    logger.py:158
        #                                lr:(7.000e-05)] [performance: 0.727
        #                                it/s] [eta: 2 days, 7:44:32] [peak
        #                                VRAM: 8.72 GB] l_g_charbonnier:
        #                                7.4075e-02 l_g_total: 3.7037e-02
        #                                grad_norm_g: 4.1103e-01 scale_g:
        #                                5.0000e-01
        # We buffer from "[epoch" until we see scale_g / l_g_total / next iter start.
        if not hasattr(self, "_metric_buf"):
            self._metric_buf = ""
            self._metric_buf_active = False
            self._metric_buf_lines = 0

        # Detect start of an iter record (NeoSR or Redux).
        # NeoSR: "[ epoch:   7 ] [ iter:  5,200 ]"
        # Redux: "[epoch:    0, iter:     700,"  (possibly preceded by "[time] INFO")
        is_iter_start = bool(re.search(r"\[\s*epoch:\s*\d+", line))
        if is_iter_start:
            # Flush previous buffer if any
            if self._metric_buf_active and self._metric_buf:
                self._do_parse_metrics(self._metric_buf)
            self._metric_buf = line
            self._metric_buf_active = True
            self._metric_buf_lines = 1
            # If this single line already contains the closing markers (NeoSR is single-line),
            # parse and flush immediately.
            if ("l_g_total" in line or "l_g_charbonnier" in line) and "performance:" in line:
                self._do_parse_metrics(self._metric_buf)
                self._metric_buf = ""
                self._metric_buf_active = False
            return

        # Continuation of a wrapped Redux record
        if self._metric_buf_active:
            stripped = line.strip()
            self._metric_buf_lines += 1

            # Accept the line as continuation if it looks like a fragment of metrics.
            # Tokens that appear in Redux wrapped output:
            looks_like_metrics = any(tok in stripped for tok in (
                "it/s", "eta:", "VRAM:", "l_g_", "l_d_", "lr:(",
                "scale_g:", "grad_norm", "peak"
            ))
            # Also accept pure number fragments like "7.4075e-02" or "5.0000e-01"
            looks_like_number = bool(re.match(r"^[0-9.eE+\-,\s]+$", stripped)) and stripped != ""

            if looks_like_metrics or looks_like_number:
                self._metric_buf += " " + stripped
                # Flush as soon as we have all the data we care about.
                # IMPORTANT: must wait for the VALUE after the key, not just the key — Redux wraps
                # like "l_g_charbonnier:" on one line and "7.4075e-02 l_g_total: 3.7037e-02" on the next.
                has_iter = bool(re.search(r"iter:\s*[\d,]+", self._metric_buf))
                has_perf = bool(re.search(r"(?:perf|performance):\s*[0-9.]+\s*it/s", self._metric_buf))
                has_loss = bool(re.search(
                    r"(?:l_g_total|l_g_charbonnier|l_g_pix):\s*[0-9.eE+\-]+", self._metric_buf
                ))
                if has_iter and has_perf and has_loss:
                    self._do_parse_metrics(self._metric_buf)
                    self._metric_buf = ""
                    self._metric_buf_active = False
                    self._metric_buf_lines = 0
                # Safety: flush if we've buffered too many lines (broken record)
                elif self._metric_buf_lines > 12:
                    self._do_parse_metrics(self._metric_buf)
                    self._metric_buf = ""
                    self._metric_buf_active = False
                    self._metric_buf_lines = 0
                return
            else:
                # Not a continuation — flush what we have, then process the new line standalone
                self._do_parse_metrics(self._metric_buf)
                self._metric_buf = ""
                self._metric_buf_active = False
                self._metric_buf_lines = 0

        # Standalone line — parse for psnr/ssim or anything else
        self._do_parse_metrics(line)

    def _do_parse_metrics(self, line):
        """Actual metric extraction from a (possibly multi-line) record string."""
        if "iter:" in line:
            it = re.search(r"iter:\s*([\d,]+)", line)
            if it:
                self.current_iter = int(it.group(1).replace(",", ""))
                if self.target_iters > 0:
                    pct = self.current_iter / self.target_iters
                    self.progress_bar.set(pct); self.lbl_progress.configure(text=f"{pct*100:.1f}%")
                    # Update Windows taskbar progress
                    try:
                        hwnd = self.winfo_toplevel().winfo_id()
                        set_taskbar_progress(hwnd, pct, "normal")
                    except Exception:
                        pass
                    if self.accumulate > 1:
                        gpu_iter = self.current_iter * self.accumulate
                        gpu_target = self.target_iters * self.accumulate
                        display_txt = (f"{self.current_iter:,} / {self.target_iters:,}"
                                       f"\nGPU×{self.accumulate}: {gpu_iter:,} / {gpu_target:,}")
                    else:
                        display_txt = f"{self.current_iter:,} / {self.target_iters}"
                    self.card_iter.configure(text=display_txt)

            spd = re.search(r"(?:perf|performance):\s*([0-9.]+)\s*it/s", line)
            if spd:
                s = float(spd.group(1)); self.card_speed.configure(text=f"{s:.2f} it/s")
                if s > 0:
                    # Accumulate speed samples for avg_speed in history
                    if not hasattr(self, "_speed_samples"):
                        self._speed_samples = []
                    self._speed_samples.append(s)
                if s > 0 and self.target_iters > 0:
                    rem = (self.target_iters - self.current_iter) / s
                    _rd, _rr = divmod(int(rem), 86400)
                    _rh, _rr = divmod(_rr, 3600)
                    _rm, _rs = divmod(_rr, 60)
                    self.card_eta.configure(text=f"{_rd:3d}{_t('j', 'd')} {_rh:02d}:{_rm:02d}:{_rs:02d}")

            # Redux uses l_g_charbonnier / l_g_pix; NeoSR uses l_g_total
            lg = (re.search(r"l_g_total:\s*([0-9.e+\-]+)", line)
                  or re.search(r"l_g_pix:\s*([0-9.e+\-]+)", line)
                  or re.search(r"l_g_charbonnier:\s*([0-9.e+\-]+)", line))
            if lg: self.card_loss_g.configure(text=f"{float(lg.group(1)):.4f}")

            ep = re.search(r"epoch:\s*(\d+)", line)
            # Redux: lr:(1.000e-04)  /  NeoSR: lr: 5.00e-04
            lr = re.search(r"lr:\s*\(?\s*([0-9.eE+\-]+)\s*\)?", line)
            if ep: self.card_epoch.configure(text=ep.group(1))
            if lr:
                try: self.card_lr.configure(text=f"{float(lr.group(1)):.1e}")
                except ValueError: pass

        # Discriminator losses — run unconditionally so Redux GAN records on separate
        # log lines (without "iter:") still update the card.
        ld = re.search(r"l_d_total:\s*([0-9.e+\-]+)", line)
        if ld:
            try: self.card_loss_d.configure(text=f"{float(ld.group(1)):.4f}")
            except Exception: pass
        else:
            ld_r = re.search(r"l_d_real:\s*([0-9.e+\-]+)", line)
            ld_f = re.search(r"l_d_fake:\s*([0-9.e+\-]+)", line)
            if ld_r and ld_f:
                try:
                    self.card_loss_d.configure(
                        text=f"{(float(ld_r.group(1))+float(ld_f.group(1)))/2:.4f}")
                except Exception: pass
            elif ld_r or ld_f:
                try:
                    _v = ld_r or ld_f
                    self.card_loss_d.configure(text=f"{float(_v.group(1)):.4f}")
                except Exception: pass

        if "psnr" in line.lower():
            m_p = re.search(r"(?:psnr|PSNR)[:=\s]+\s*([0-9.]+)", line, re.IGNORECASE)
            if m_p:
                v = float(m_p.group(1))
                if v > self.best_psnr and v < 100:
                    self.best_psnr = v; self.best_psnr_iter = self.current_iter
                    self.card_psnr.configure(text=f"{v:.4f} dB\n(@ {self.best_psnr_iter})", text_color="#2ecc71")
                    try:
                        from src.core.training_history import update_training_progress
                        row_id = getattr(self, "_history_row_id", 0)
                        if row_id > 0:
                            update_training_progress(row_id, current_iter=self.current_iter,
                                                       best_psnr=v, best_iter=self.current_iter)
                    except Exception:
                        pass

        # Only update SSIM from validation result lines (which always contain both psnr + ssim).
        # Training loss lines contain l_g_ssim but NOT psnr — so this guard filters them out.
        # Also: real SSIM is 0-1; l_g_ssim values like 1.5738e-02 would match as 1.5738 (wrong).
        if "ssim" in line.lower() and "psnr" in line.lower():
            m_s = re.search(r"\bssim[:=\s]+([0-9.]+)", line, re.IGNORECASE)
            if m_s:
                v = float(m_s.group(1))
                if 0 < v <= 1.0 and v > self.best_ssim:
                    self.best_ssim = v; self.best_ssim_iter = self.current_iter
                    self.card_ssim.configure(text=f"{v:.4f}\n(@ {self.best_ssim_iter})", text_color="#1abc9c")
                    try:
                        from src.core.training_history import update_training_progress
                        row_id = getattr(self, "_history_row_id", 0)
                        if row_id > 0:
                            update_training_progress(row_id, best_ssim=v)
                    except Exception:
                        pass

    def load_iters_from_config_data(self, data):
        try:
            v = data.get("total_iter") or data.get("train", {}).get("total_iter") or 100000
            self.target_iters = int(v)
            # accumulate can live at root, train, or datasets.train under various keys
            acc = (
                data.get("accumulate_grad_batches") or
                data.get("train", {}).get("accumulate_grad_batches") or
                data.get("train", {}).get("accumulate") or
                data.get("datasets", {}).get("train", {}).get("accumulate") or
                data.get("datasets", {}).get("train", {}).get("accum_iter") or  # traiNNer-redux key
                data.get("train", {}).get("accum_iter") or
                data.get("accum_iter") or
                data.get("accumulate") or 1
            )
            self.accumulate = max(1, int(acc) if acc else 1)
            self.card_iter.configure(text=f"0 / {self.target_iters}")
        except Exception: pass

    def update_timer(self):
        if self.timer_running:
            self.card_timer.configure(text=str(datetime.datetime.now() - self.start_time).split('.')[0])
            self.after(1000, self.update_timer)

    def auto_load_settings(self):
        try:
            self.entries_dict["python_path"].insert(0, self.settings.get("python_path", ""))
            self.entries_dict["script_path"].insert(0, self.settings.get("script_path", ""))
            p = self.settings.get("config_path", "")
            self.entries_dict["config_path"].insert(0, p)
            if p: self.detect_engine_and_setup(p)
        except Exception: pass