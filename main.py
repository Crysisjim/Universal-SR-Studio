import sys
import os
import traceback
import datetime

# Force UTF-8 stdout/stderr — évite crash cp1252 sur Windows (emoji dans print)
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ─── PyInstaller bundle path resolution ───────────────────────────────────────
# PyInstaller 6+ places bundled files in _internal/ (= sys._MEIPASS).
# The exe itself sits one level up (os.path.dirname(sys.executable)).
#
#   dist/Universal_SR_Studio/
#   ├── Universal_SR_Studio.exe   ← sys.executable
#   └── _internal/                ← sys._MEIPASS
#       └── assets/               ← bundled assets
#
# _BASE_DIR  → exe directory (crash_log, user_settings, .first_launch_done)
# _ASSETS_DIR → _MEIPASS when frozen, else script dir (assets/, themes/, …)
if getattr(sys, 'frozen', False):
    _BASE_DIR   = os.path.dirname(sys.executable)
    _ASSETS_DIR = sys._MEIPASS          # _internal/ contains assets/
    os.chdir(_ASSETS_DIR)               # makes os.getcwd()/assets/… work in src/app.py
else:
    _BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    _ASSETS_DIR = _BASE_DIR

# Crash Logger — always written next to the exe / script
CRASH_LOG = os.path.join(_BASE_DIR, "crash_log.txt")

def log_crash(exc_type, exc_value, exc_tb):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    entry = f"\n{'='*60}\n[CRASH] {timestamp}\n{'='*60}\n{tb_text}\n"
    try:
        with open(CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass
    print(entry, file=sys.stderr)
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Universal SR Studio - Crash",
            f"Une erreur fatale s'est produite.\n\n"
            f"{exc_type.__name__}: {exc_value}\n\n"
            f"Details dans:\n{CRASH_LOG}")
        root.destroy()
    except Exception:
        pass

sys.excepthook = log_crash

def check_first_launch():
    missing = []
    warnings = []
    for pkg, name in [("customtkinter", "CustomTkinter"), ("PIL", "Pillow"),
                       ("yaml", "PyYAML"), ("toml", "toml")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(name)
    for pkg, name in [("numpy", "NumPy"), ("psutil", "psutil"),
                       ("win11toast", "win11toast (notifications Windows 11)"),
                       ("safetensors", "safetensors (export modeles)"),
                       ("comtypes", "comtypes (taskbar progress)"),
                       ("qrcode", "qrcode (QR codes pour galerie distante)")]:
        try:
            __import__(pkg)
        except ImportError:
            warnings.append(f"{name} manquant (optionnel)")
    engine_paths = [
        os.path.join(os.path.expanduser("~"), "IA_Engine", "neosr"),
        os.path.join(os.path.expanduser("~"), "IA_Engine", "traiNNer-redux"),
    ]
    if not any(os.path.isdir(p) for p in engine_paths):
        warnings.append("Aucun moteur IA détecté (NeoSR / TraiNNer-Redux)\nNo AI engine detected (NeoSR / TraiNNer-Redux)")
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if r.returncode != 0:
            warnings.append("nvidia-smi non disponible (pas de GPU NVIDIA?)")
    except Exception:
        warnings.append("nvidia-smi non trouve")
    if missing:
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk(); root.withdraw()
            messagebox.showerror("Dépendances manquantes / Missing dependencies",
                f"Packages requis absents / Required packages missing:\n\n"
                f"{chr(10).join('  - ' + m for m in missing)}\n\n"
                f"Installez avec / Install with:\n  pip install {' '.join(m.lower() for m in missing)}")
            root.destroy()
        except Exception:
            print(f"ERREUR: Packages manquants: {', '.join(missing)}")
        return False
    if warnings:
        marker = os.path.join(_BASE_DIR, ".first_launch_done")
        if not os.path.exists(marker):
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk(); root.withdraw()
                messagebox.showwarning("Premier lancement / First launch",
                    f"Bienvenue dans Universal SR Studio !\n"
                    f"Welcome to Universal SR Studio!\n\n"
                    f"Avertissements / Warnings:\n\n"
                    f"{chr(10).join('  - ' + w for w in warnings)}\n\n"
                    f"Ces éléments sont optionnels mais recommandés.\n"
                    f"These elements are optional but recommended.")
                root.destroy()
                with open(marker, "w") as f:
                    f.write(datetime.datetime.now().isoformat())
            except Exception:
                pass
    return True

def _ask_language_first_launch():
    """Show FR/EN picker on very first launch (before settings exist)."""
    try:
        from src.core.settings import SettingsManager
        s = SettingsManager()
        if s.get("language", ""):
            return  # Already chosen
        import tkinter as tk
        result = {'lang': 'fr'}
        root = tk.Tk()
        root.title("Universal SR Studio")
        root.geometry("360x190")
        root.resizable(False, False)
        root.configure(bg="#1a1a2e")
        root.update_idletasks()
        x = (root.winfo_screenwidth()  - 360) // 2
        y = (root.winfo_screenheight() - 190) // 2
        root.geometry(f"360x190+{x}+{y}")
        tk.Label(root, text="Universal SR Studio",
                 font=("Arial", 14, "bold"), fg="white", bg="#1a1a2e").pack(pady=(22, 4))
        tk.Label(root, text="Choose your language / Choisissez votre langue",
                 font=("Arial", 10), fg="#aaaaaa", bg="#1a1a2e").pack()
        frame = tk.Frame(root, bg="#1a1a2e")
        frame.pack(pady=22)
        def pick(lang):
            result['lang'] = lang
            root.destroy()
        tk.Button(frame, text="🇫🇷  Français",  command=lambda: pick('fr'),
                  width=13, height=2, bg="#2d6a4f", fg="white",
                  font=("Arial", 11, "bold"), relief="flat", cursor="hand2").pack(side="left", padx=12)
        tk.Button(frame, text="🇬🇧  English",   command=lambda: pick('en'),
                  width=13, height=2, bg="#1d3557", fg="white",
                  font=("Arial", 11, "bold"), relief="flat", cursor="hand2").pack(side="left", padx=12)
        root.protocol("WM_DELETE_WINDOW", lambda: pick('fr'))
        root.mainloop()
        s.set("language", result['lang'])
    except Exception:
        pass


if __name__ == "__main__":
    check_first_launch()
    _ask_language_first_launch()

    # ─── Windows: Set AppUserModelID for proper taskbar icon ───
    if sys.platform == "win32":
        try:
            import ctypes
            app_id = "Crysisjim.UniversalSRStudio.1.0"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass

    # Per-monitor DPI awareness — prevents blurry/oversized UI on high-DPI displays
    # Must be called before any window/CTk initialisation
    if sys.platform == "win32":
        try:
            import ctypes as _dpi
            _dpi.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE_V2
        except Exception:
            try:
                _dpi.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    import customtkinter as ctk
    from src.app import App
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("green")
    app = App()

    # Set window icon
    try:
        icon_path = os.path.join(_ASSETS_DIR, "assets", "icon.ico")
        if os.path.exists(icon_path):
            app.iconbitmap(icon_path)
            # Also set for alt-tab and taskbar
            if sys.platform == "win32":
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                icon_flags = 0x00000080  # LR_LOADFROMFILE
                hicon = ctypes.windll.user32.LoadImageW(0, icon_path, 1, 0, 0, icon_flags | 0x00000010)
                if hicon:
                    ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, hicon)  # WM_SETICON ICON_SMALL
                    ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, hicon)  # WM_SETICON ICON_BIG
    except Exception:
        pass

    # Register cleanup on window close (stops gallery server / ngrok if running)
    try:
        app.protocol("WM_DELETE_WINDOW", app._cleanup_on_exit)
    except Exception:
        pass

    app.mainloop()
    # Fallback: if mainloop exits without WM_DELETE_WINDOW (e.g. Alt+F4 bypass)
    sys.exit(0)
