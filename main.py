import sys
import os
import traceback
import datetime

# ─── PyInstaller bundle path resolution ───────────────────────────────────────
# When frozen (exe), __file__ points inside _MEIPASS (temp extraction).
# _BASE_DIR is always the directory containing the exe or the script.
if getattr(sys, 'frozen', False):
    _BASE_DIR = os.path.dirname(sys.executable)
    # Make os.getcwd() reliable for asset loading in src/app.py (theme paths etc.)
    os.chdir(_BASE_DIR)
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        warnings.append("Aucun moteur IA detecte (NeoSR / TraiNNer-Redux)")
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
            messagebox.showerror("Dependances manquantes",
                f"Packages requis absents:\n\n"
                f"{chr(10).join('  - ' + m for m in missing)}\n\n"
                f"Installez avec:\n  pip install {' '.join(m.lower() for m in missing)}")
            root.destroy()
        except Exception:
            print(f"ERREUR: Packages manquants: {', '.join(missing)}")
        return False
    if warnings:
        marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".first_launch_done")
        if not os.path.exists(marker):
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk(); root.withdraw()
                messagebox.showwarning("Premier lancement",
                    f"Bienvenue dans Universal SR Studio !\n\n"
                    f"Avertissements:\n\n"
                    f"{chr(10).join('  - ' + w for w in warnings)}\n\n"
                    f"Ces elements sont optionnels mais recommandes.")
                root.destroy()
                with open(marker, "w") as f:
                    f.write(datetime.datetime.now().isoformat())
            except Exception:
                pass
    return True

if __name__ == "__main__":
    check_first_launch()

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
    ctk.set_default_color_theme("blue")
    app = App()

    # Set window icon
    try:
        icon_path = os.path.join(_BASE_DIR, "assets", "icon.ico")
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
