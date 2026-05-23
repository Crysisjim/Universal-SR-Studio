import customtkinter as ctk
from tkinter import messagebox, filedialog
import os
import webbrowser
import subprocess
import threading
import shutil
import sys
import stat
import time
import glob
from src.core.settings import SettingsManager
from src.core.compute_estimator import detect_gpu_name, get_pytorch_recommendation

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

# --- FENÊTRE DE PROGRESSION (CONSOLE LIVE) ---
class ConsolePopup(ctk.CTkToplevel):
    def __init__(self, master, title, command, cwd, on_close_callback=None):
        super().__init__(master)
        self.title(title)
        self.attributes("-topmost", True)
        self.on_close_callback = on_close_callback
        
        w, h = 850, 600
        try:
            root_x = master.winfo_rootx(); root_y = master.winfo_rooty()
            root_w = master.winfo_width(); root_h = master.winfo_height()
            x = root_x + (root_w // 2) - (w // 2)
            y = root_y + (root_h // 2) - (h // 2)
        except Exception:
            ws = self.winfo_screenwidth(); hs = self.winfo_screenheight()
            x = (ws // 2) - (w // 2); y = (hs // 2) - (h // 2)
            
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(0, weight=1)
        
        self.textbox = ctk.CTkTextbox(self, font=("Consolas", 10), text_color="#ecf0f1", fg_color="#101010")
        self.textbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.textbox.insert("0.0", f"--- SEQUENCE D'OPERATION ---\n> Cible : {cwd}\n> Action : {title}\n\n")
        
        self.btn_close = ctk.CTkButton(self, text=_t("En cours...", "Running..."), state="disabled", command=self.close_and_refresh, fg_color="#555")
        self.btn_close.grid(row=1, column=0, pady=10)

        self.command = command
        self.cwd = cwd
        
        self.after(100, lambda: self.focus_force())
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        try:
            if not os.path.exists(self.cwd): os.makedirs(self.cwd, exist_ok=True)
            
            # Utilisation de shell=True pour Windows pour gérer les chemins complexes
            process = subprocess.Popen(
                self.command, 
                cwd=self.cwd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                shell=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            for line in process.stdout:
                self.textbox.insert("end", line)
                self.textbox.see("end")
            
            process.wait()
            
            if process.returncode == 0:
                self.textbox.insert("end", _t("\n\n[SUCCES] Operation terminee.\n", "\n\n[SUCCESS] Operation completed.\n"))
                self.btn_close.configure(state="normal", text=_t("Fermer & Actualiser", "Close & Refresh"), fg_color="green")
            else:
                self.textbox.insert("end", _t(f"\n\n[ERREUR] Une etape a echoue (Code {process.returncode}).\n", f"\n\n[ERROR] A step failed (Code {process.returncode}).\n"))
                self.btn_close.configure(state="normal", text=_t("Fermer", "Close"), fg_color="#e74c3c")

        except Exception as e:
            self.textbox.insert("end", _t(f"\n\n[CRASH CRITIQUE] {str(e)}\n", f"\n\n[CRITICAL CRASH] {str(e)}\n"))
            self.btn_close.configure(state="normal", text=_t("Fermer (Crash)", "Close (Crash)"), fg_color="#e74c3c")

    def close_and_refresh(self):
        self.destroy()
        if self.on_close_callback:
            self.on_close_callback()

# --- ONGLET SETTINGS ---
class SettingsTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.settings = SettingsManager()
        
        self.user_home = os.path.expanduser("~")
        self.base_engine_path = os.path.join(self.user_home, "IA_Engine")
        self.runtimes_path = os.path.join(self.base_engine_path, "runtimes")
        self.neosr_path = os.path.join(self.base_engine_path, "neosr")
        self.redux_path = os.path.join(self.base_engine_path, "traiNNer-redux")
        
        self.assets_themes = os.path.join(os.getcwd(), "assets", "themes")

        self.repos = {
            "NeoSR": "https://github.com/muslll/neosr",
            "TraiNNer-Redux": "https://github.com/the-database/traiNNer-redux"
        }

        self.system_tools = ["git", "ffmpeg", "nvidia-smi"]
        self.python_modules = [
            "torch", "torchvision", "numpy", "cv2", "yaml", 
            "scipy", "pywt", "tqdm", "rich", "tensorboard", "PIL",
            "ema_pytorch" 
        ]
        
        self.status_labels = {}
        self._gpu_advisory_frame = None
        self._gpu_advisory_body  = None

        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(0, weight=1)
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.tab_engines = self.tab_view.add("Moteurs IA")
        self.tab_sys = self.tab_view.add("Système & Dépendances")
        self.tab_lang = self.tab_view.add("Langue & Notif.")
        self.tab_apikeys = self.tab_view.add("Cles API")
        self.tab_gallery = self.tab_view.add("Galerie & Patch TB")
        self.tab_about = self.tab_view.add("A Propos")

        self.setup_engines_tab()
        self.setup_system_tab()
        self.setup_workspace_folders()
        try:
            self.setup_language_tab()
            self.setup_notifications_tab()
        except Exception as e:
            ctk.CTkLabel(self.tab_lang, text=f"Erreur: {e}", text_color="#e74c3c").pack(pady=20)

        try:
            self.setup_apikeys_tab()
        except Exception as e:
            ctk.CTkLabel(self.tab_apikeys, text=f"Erreur: {e}", text_color="#e74c3c").pack(pady=20)
        try:
            self.setup_gallery_tab()
        except Exception as e:
            ctk.CTkLabel(self.tab_gallery, text=f"Erreur: {e}", text_color="#e74c3c").pack(pady=20)
        try:
            self.setup_about_tab()
        except Exception as e:
            ctk.CTkLabel(self.tab_about, text=f"Erreur: {e}", text_color="#e74c3c").pack(pady=20)

    def setup_workspace_folders(self):
        folders = [
            os.path.join(self.base_engine_path, "experiments"),
            os.path.join(self.base_engine_path, "datasets", "train", "HR"),
            os.path.join(self.base_engine_path, "datasets", "train", "LQ"),
            os.path.join(self.base_engine_path, "datasets", "val", "GT"),
            os.path.join(self.base_engine_path, "datasets", "val", "LQ"),
        ]
        for f in folders:
            try: os.makedirs(f, exist_ok=True)
            except Exception: pass

    # --- THEMES & CONFIG ---
    def get_available_themes(self):
        themes = ["blue", "green", "dark-blue"]
        if os.path.exists(self.assets_themes):
            for f in glob.glob(os.path.join(self.assets_themes, "*.json")):
                themes.append(os.path.splitext(os.path.basename(f))[0])
        return themes

    def change_appearance(self, mode):
        ctk.set_appearance_mode(mode)
        self.settings.set("appearance_mode", mode)

    def change_color(self, color_name):
        self.settings.set("theme_color", color_name)
        messagebox.showinfo("Changement de Thème", f"Le thème '{color_name}' a été sauvegardé.\n\nVeuillez redémarrer l'application.")

    def save_aida(self):
        val = "true" if self.chk_aida.get() else "false"
        self.settings.set("use_aida64", val)

    # --- SUPPRESSION SECURISEE ---
    def check_user_data_protection(self, path):
        if not os.path.exists(path): return True
        protected = ["datasets", "experiments", "datsheet", "models", "weights"]
        found = [i for i in os.listdir(path) if i.lower() in protected]
        if found:
            messagebox.showwarning("PROTECTION DONNEES", f"Suppression BLOQUÉE : Dossiers sensibles détectés ({found}).")
            return False
        return True

    def on_rm_error(self, func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        try: func(path)
        except Exception: pass

    def force_delete_folder(self, path):
        if not self.check_user_data_protection(path): return False
        if not os.path.exists(path): return True
        for i in range(3): 
            try:
                shutil.rmtree(path, onerror=self.on_rm_error)
                time.sleep(0.5)
                if not os.path.exists(path): return True
            except Exception: time.sleep(1)
        return not os.path.exists(path)

    # --- PYTHON PORTABLE ---
    def get_portable_python_path(self):
        py_path = os.path.join(self.runtimes_path, "python-3.11.9", "python.exe")
        if os.path.exists(py_path): return py_path
        return None

    def install_portable_python(self):
        target_dir = os.path.join(self.runtimes_path, "python-3.11.9")
        if not os.path.exists(self.base_engine_path): os.makedirs(self.base_engine_path, exist_ok=True)
        installer_script_path = os.path.join(self.base_engine_path, "install_runtime.py")
        
        script_content = f"""
import urllib.request, zipfile, os, sys, time, ssl
try: ssl._create_default_https_context = ssl._create_unverified_context
except Exception: pass
def log(msg): print(f'>> {{msg}}', flush=True)
target_dir = r'{target_dir}'
runtimes_path = r'{self.runtimes_path}'
zip_path = os.path.join(runtimes_path, "python311.zip")
if os.path.exists(target_dir):
    import shutil
    try: shutil.rmtree(target_dir, ignore_errors=True)
    except Exception: pass
os.makedirs(runtimes_path, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)
url = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip'
log('Download Python 3.11...')
try: urllib.request.urlretrieve(url, zip_path)
except Exception: sys.exit(1)
log('Extraction...')
with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(target_dir)
os.remove(zip_path)
pth = os.path.join(target_dir, "python311._pth")
if os.path.exists(pth):
    with open(pth, 'r') as f: c = f.read()
    c = c.replace('#import site', 'import site')
    with open(pth, 'w') as f: f.write(c)
gp = os.path.join(target_dir, "get-pip.py")
try: urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', gp)
except Exception: pass
py = os.path.join(target_dir, "python.exe")
log('Install PIP + Virtualenv...')
os.system(f'{{py}} {{gp}}')
os.system(f'{{py}} -m pip install virtualenv --no-warn-script-location')
log('[OK] TERMINE !')
"""
        try:
            with open(installer_script_path, "w", encoding="utf-8") as f: f.write(script_content)
        except Exception: return messagebox.showerror("Erreur", "Impossible d'ecrire le script.")
        cmd = [sys.executable, installer_script_path]
        self.launch_console("Installation Runtime", cmd, self.base_engine_path)

    # --- MISE A JOUR DES DEPENDANCES (VERSION FINALE) ---
    def install_missing_deps_only(self, engine_name, cwd, py_venv):
        """
        Installe msgspec et les autres dépendances manquantes via un script piloté.
        """
        # Liste complète mise à jour pour Redux
        pkgs = ["numpy<2.0.0", "opencv-python", "scipy", "pyyaml", "toml", "tqdm", 
                "tensorboard", "rich", "PyWavelets", "pillow", "einops", "lmdb", "msgspec"]
        
        if "Redux" in engine_name:
            pkgs.extend(["albumentations", "ema-pytorch"])
            
        pkgs_str = " ".join(pkgs)
        script_path = os.path.join(cwd, "update_deps_temp.py")
        
        script_content = f"""
import subprocess
import sys
import os

py_exe = r'{py_venv}'
pkgs = {pkgs}

print(f">> Utilisation de l'environnement : {{py_exe}}")
print(f">> Installation des paquets requis...")

# Commande python -m pip install
cmd = [py_exe, "-m", "pip", "install"] + pkgs + ["--no-warn-script-location"]

try:
    # On force la mise à jour de pip d'abord
    subprocess.check_call([py_exe, "-m", "pip", "install", "--upgrade", "pip"])
    # Installation des dépendances
    subprocess.check_call(cmd)
    print(">> [SUCCES] Toutes les dépendances sont installées.")
except Exception as e:
    print(f">> [ERREUR] {{e}}")
    sys.exit(1)
"""
        try:
            with open(script_path, "w", encoding="utf-8") as f: f.write(script_content)
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec création script: {e}")
            return

        cmd = [sys.executable, script_path]
        self.launch_console(f"Mise à jour (Deps + msgspec)", cmd, cwd)


    # --- INSTALLATION COMPLETE (SCRIPT) ---
    def launch_engine_install_script(self, git_url, target_path, engine_name, portable_py):
        if not os.path.exists(self.base_engine_path): os.makedirs(self.base_engine_path, exist_ok=True)
        installer_path = os.path.join(self.base_engine_path, f"install_{engine_name}.py")

        # Determine recommended PyTorch version for this GPU
        try:
            _rec = get_pytorch_recommendation()
        except Exception:
            _rec = {"torch_version": "2.7.0", "torchvision_version": "0.22.0",
                    "cuda_tag": "cu126", "whl_url": "https://download.pytorch.org/whl/cu126",
                    "install_pkgs": ["torch==2.7.0", "torchvision==0.22.0"]}
        _torch_pkg  = _rec["install_pkgs"][0]
        _tv_pkg     = _rec["install_pkgs"][1]
        _whl_url    = _rec["whl_url"]
        _cuda_label = f"{_rec['torch_version']} CUDA {_rec['cuda_tag']}"

        reqs = "numpy<2.0.0\nopencv-python\nscipy\npyyaml\ntoml\ntqdm\ntensorboard\nrich\nPyWavelets\npillow\neinops\nlmdb\n"
        if "Redux" in engine_name:
            reqs += "albumentations\nema-pytorch\n"

        script_content = f"""
import os, sys, subprocess
def log(msg): print(f'>> {{msg}}', flush=True)
TARGET = r'{target_path}'
GIT_URL = '{git_url}'
PORTABLE_PY = r'{portable_py}'
REQ_FILE = os.path.join(TARGET, "requirements.txt")
VENV_DIR = os.path.join(TARGET, ".venv")
if not os.path.exists(os.path.join(TARGET, ".git")):
    log("Git Clone...")
    subprocess.call(['git', 'clone', GIT_URL, '.'], cwd=TARGET, shell=True)
else:
    log("Git Pull...")
    subprocess.call(['git', 'pull'], cwd=TARGET, shell=True)
if not os.path.exists(REQ_FILE):
    with open(REQ_FILE, "w", encoding="utf-8") as f: f.write('''{reqs}''')
if not os.path.exists(VENV_DIR):
    log("Creation VENV...")
    subprocess.call([PORTABLE_PY, "-m", "virtualenv", VENV_DIR], cwd=TARGET)
pip = os.path.join(VENV_DIR, "Scripts", "pip.exe") if sys.platform == "win32" else os.path.join(VENV_DIR, "bin", "pip")
log("Update PIP...")
subprocess.call([pip, "install", "--upgrade", "pip", "wheel"], cwd=TARGET)
log("Install Torch {_cuda_label}...")
subprocess.call([pip, "uninstall", "-y", "torch", "torchvision"], cwd=TARGET)
subprocess.call([pip, "install", "{_torch_pkg}", "{_tv_pkg}", "--index-url", "{_whl_url}", "--no-cache-dir"], cwd=TARGET)
log("Install Deps...")
subprocess.call([pip, "install", "-r", "requirements.txt"], cwd=TARGET)
subprocess.call([pip, "install", "tensorboard", "rich"], cwd=TARGET)
log("[OK] TERMINE !")
"""
        try:
            with open(installer_path, "w", encoding="utf-8") as f: f.write(script_content)
        except Exception: return messagebox.showerror("Erreur", "Script error.")
        cmd = [sys.executable, installer_path]
        if not os.path.exists(target_path): os.makedirs(target_path)
        self.launch_console(f"Install {engine_name}", cmd, target_path)

    # --- UI ---
    def setup_engines_tab(self):
        for w in self.tab_engines.winfo_children(): w.destroy()
        self.tab_engines.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.tab_engines, text=_t("CENTRE D'INSTALLATION DES MOTEURS", "ENGINE INSTALLATION CENTER"), font=("Roboto", 20, "bold"), text_color="#3B8ED0").pack(pady=15)
        self.create_engine_block(self.tab_engines, "NeoSR", self.repos["NeoSR"], self.neosr_path).pack(fill="x", pady=10, padx=15)
        self.create_engine_block(self.tab_engines, "TraiNNer-Redux", self.repos["TraiNNer-Redux"], self.redux_path).pack(fill="x", pady=10, padx=15)

    def get_folder_size(self, path):
        total = 0
        try:
            for d, _, f in os.walk(path):
                for i in f: 
                    fp = os.path.join(d, i)
                    if not os.path.islink(fp): total += os.path.getsize(fp)
            return f"{total / (1024*1024):.1f} MB"
        except Exception: return "0 MB"

    # ── GPU & PyTorch advisory ────────────────────────────────────────────
    def _load_gpu_advisory(self):
        """Background thread: detect GPU, build recommendation, read installed versions."""
        try:
            rec = get_pytorch_recommendation()
        except Exception as e:
            rec = {"gpu_name": "Erreur", "gpu_gen": "?", "has_tensor_cores": False,
                   "torch_version": "?", "cuda_tag": "?", "min_torch_version": "0",
                   "note": str(e), "features": [], "limitations": [],
                   "install_pkgs": [], "whl_url": "", "upgrade_reason": ""}

        # Read installed torch version for each engine venv
        venv_versions = {}
        for eng_name, eng_path in [("NeoSR", self.neosr_path), ("TraiNNer-Redux", self.redux_path)]:
            venv_py = os.path.join(eng_path, ".venv", "Scripts", "python.exe")
            venv_versions[eng_name] = self._get_venv_torch_version(venv_py)

        rec["_venv_versions"] = venv_versions
        self.after(0, lambda: self._render_gpu_advisory(rec))

    @staticmethod
    def _get_venv_torch_version(py_venv: str) -> str:
        """Return installed torch version string (e.g. '2.6.0+cu124') or '' if absent."""
        if not os.path.exists(py_venv):
            return ""
        try:
            r = subprocess.run(
                [py_venv, "-c", "import torch; print(torch.__version__)"],
                capture_output=True, text=True, timeout=10,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _parse_version(ver_str: str) -> tuple:
        """Parse '2.6.0+cu124' → (2, 6, 0). Returns (0,0,0) on failure."""
        try:
            base = ver_str.split("+")[0]
            parts = base.split(".")
            return tuple(int(x) for x in parts[:3])
        except Exception:
            return (0, 0, 0)

    def _render_gpu_advisory(self, rec: dict):
        """Render GPU + PyTorch recommendation inside _gpu_advisory_body."""
        for w in self._gpu_advisory_body.winfo_children():
            w.destroy()

        has_tc = rec.get("has_tensor_cores", False)
        gpu_col = "#2ecc71" if has_tc else "#f39c12"
        tc_tag = "✅ Tensor Cores" if has_tc else "⚠️ Sans Tensor Cores"

        # Row 1 : GPU name + generation
        r1 = ctk.CTkFrame(self._gpu_advisory_body, fg_color="transparent")
        r1.pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(r1, text=f"🖥️  {rec['gpu_name']}", font=("Consolas", 11, "bold")).pack(side="left")
        ctk.CTkLabel(r1, text=f"  {tc_tag}", text_color=gpu_col, font=("Arial", 10)).pack(side="left", padx=6)
        ctk.CTkLabel(r1, text=rec.get("gpu_gen", ""), text_color="#aaa", font=("Arial", 10)).pack(side="right")

        # Row 2 : Recommended version
        r2 = ctk.CTkFrame(self._gpu_advisory_body, fg_color="transparent")
        r2.pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(r2,
                     text=f"PyTorch recommandé : {rec['torch_version']} + CUDA {rec['cuda_tag']}",
                     font=("Consolas", 11), text_color="#ddd").pack(side="left")

        # Row 3 : Features (green) + Limitations (orange)
        r3 = ctk.CTkFrame(self._gpu_advisory_body, fg_color="transparent")
        r3.pack(fill="x", pady=(4, 0))
        feat_str = "  ·  ".join(rec.get("features", []))
        if feat_str:
            ctk.CTkLabel(r3, text=f"✅ {feat_str}", text_color="#2ecc71",
                         font=("Arial", 10), wraplength=600, justify="left").pack(anchor="w")
        for lim in rec.get("limitations", []):
            ctk.CTkLabel(self._gpu_advisory_body, text=f"⚠️  {lim}",
                         text_color="#f39c12", font=("Arial", 10)).pack(anchor="w", pady=1)

        # Row 4 : Per-engine version status + conditional install buttons
        pkgs       = rec.get("install_pkgs", [])
        whl        = rec.get("whl_url", "")
        rec_ver    = rec.get("torch_version", "?")
        min_ver    = rec.get("min_torch_version", "2.3.0")
        upg_reason = rec.get("upgrade_reason", "")
        venv_vers  = rec.get("_venv_versions", {})

        engines = [
            ("NeoSR",           self.neosr_path,  "#1a5276"),
            ("TraiNNer-Redux",  self.redux_path,  "#1a3a4f"),
        ]

        any_engine_shown = False
        for eng_name, eng_path, btn_color in engines:
            venv_py = os.path.join(eng_path, ".venv", "Scripts", "python.exe")
            if not os.path.exists(venv_py):
                continue
            any_engine_shown = True
            installed = venv_vers.get(eng_name, "")

            r_eng = ctk.CTkFrame(self._gpu_advisory_body, fg_color="transparent")
            r_eng.pack(fill="x", pady=(4, 0))

            if not installed:
                # PyTorch absent → propose install
                ctk.CTkButton(
                    r_eng, text=f"📦 Installer PyTorch {rec_ver} → {eng_name}",
                    fg_color=btn_color, height=28, font=("Arial", 11),
                    command=lambda p=venv_py, ep=eng_path, pk=pkgs, w=whl, en=eng_name:
                        self._install_pytorch_for_engine(p, ep, pk, w, en)
                ).pack(side="left", padx=(0, 8))
                ctk.CTkLabel(r_eng, text="(PyTorch absent)", text_color="#e74c3c",
                             font=("Arial", 10)).pack(side="left")
            else:
                inst_tuple = self._parse_version(installed)
                min_tuple  = self._parse_version(min_ver)
                rec_tuple  = self._parse_version(rec_ver)

                if inst_tuple < min_tuple:
                    # Version trop ancienne → upgrade nécessaire
                    ctk.CTkButton(
                        r_eng, text=f"⬆ Mettre a jour PyTorch {installed} → {rec_ver} ({eng_name})",
                        fg_color="#c0392b", height=28, font=("Arial", 11),
                        command=lambda p=venv_py, ep=eng_path, pk=pkgs, w=whl, en=eng_name:
                            self._install_pytorch_for_engine(p, ep, pk, w, en)
                    ).pack(side="left", padx=(0, 8))
                elif inst_tuple < rec_tuple:
                    # Version OK mais pas la dernière recommandée → bouton discret
                    ctk.CTkLabel(r_eng,
                                 text=f"✅ {eng_name} : PyTorch {installed}",
                                 text_color="#2ecc71", font=("Consolas", 10)).pack(side="left")
                    ctk.CTkButton(
                        r_eng, text=f"⬆ {rec_ver} dispo",
                        fg_color="#2c3e50", height=22, width=90, font=("Arial", 10),
                        command=lambda p=venv_py, ep=eng_path, pk=pkgs, w=whl, en=eng_name:
                            self._install_pytorch_for_engine(p, ep, pk, w, en)
                    ).pack(side="left", padx=(6, 0))
                    if upg_reason:
                        ctk.CTkLabel(r_eng, text=f"({upg_reason})",
                                     text_color="#888", font=("Arial", 9)).pack(side="left", padx=4)
                else:
                    # Version >= recommandée → tout est bon
                    ctk.CTkLabel(r_eng,
                                 text=f"✅ {eng_name} : PyTorch {installed} — OK",
                                 text_color="#2ecc71", font=("Consolas", 10)).pack(side="left")

        if not any_engine_shown:
            r4_empty = ctk.CTkFrame(self._gpu_advisory_body, fg_color="transparent")
            r4_empty.pack(fill="x", pady=(4, 0))
            ctk.CTkLabel(r4_empty,
                         text="(Aucun moteur installe — installe d'abord NeoSR ou TraiNNer-Redux)",
                         text_color="#888", font=("Arial", 10)).pack(anchor="w")

    def _install_pytorch_for_engine(self, py_venv: str, cwd: str, pkgs: list, whl_url: str, engine_name: str):
        """Generate and run a pip install script for the recommended PyTorch in given venv."""
        import tempfile
        script_lines = [
            "import subprocess, sys",
            f"py = r'{py_venv}'",
            f"pkgs = {pkgs!r}",
            f"whl  = {whl_url!r}",
            "print('>> Desinstallation torch/torchvision existants...')",
            "subprocess.call([py, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'])",
            "print('>> Installation PyTorch recommande...')",
            "cmd = [py, '-m', 'pip', 'install'] + pkgs + ['--index-url', whl, '--no-cache-dir']",
            "r = subprocess.call(cmd)",
            "print('>> [OK] Termine !' if r == 0 else f'>> [ERREUR] code={r}')",
        ]
        script = "\n".join(script_lines)
        tmp = os.path.join(cwd, "_install_pytorch_rec.py")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(script)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'écrire le script : {e}")
            return
        self.launch_console(f"PyTorch → {engine_name}", [sys.executable, tmp], cwd)

    def create_engine_block(self, parent, title, repo_url, install_path):
        f = ctk.CTkFrame(parent, border_width=1, border_color="#333")
        head = ctk.CTkFrame(f, fg_color="transparent"); head.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(head, text=title, font=("Arial", 16, "bold")).pack(side="left")
        
        is_inst = os.path.exists(os.path.join(install_path, "train.py"))
        st = "✅ INSTALLÉ" if is_inst else "❌ ABSENT"
        col = "#2ecc71" if is_inst else "#e74c3c"
        ctk.CTkLabel(head, text=st, text_color=col, font=("Arial", 12, "bold")).pack(side="right")
        
        if is_inst:
            ctk.CTkLabel(f, text=f"Taille : {self.get_folder_size(install_path)} | Chemin : {install_path}", font=("Consolas", 10), text_color="gray", anchor="w").pack(fill="x", padx=15, pady=(0, 10))

        btn = ctk.CTkFrame(f, fg_color="transparent"); btn.pack(fill="x", padx=15, pady=(0, 15))
        if not is_inst:
            ctk.CTkButton(btn, text="⬇ INSTALLATION (via Python Portable)", fg_color="#e67e22", command=lambda: self.check_and_launch_install(repo_url, install_path, title)).pack(side="left", fill="x", expand=True)
        else:
            ctk.CTkButton(btn, text="🔄 PULL", fg_color="#2980b9", width=100, command=lambda: self.launch_console(f"Update {title}", "git pull", install_path)).pack(side="left", padx=(0,5))
            ctk.CTkButton(btn, text="⚠️ REINSTALL", fg_color="#c0392b", width=100, command=lambda: self.wipe_and_full_install(repo_url, install_path, title)).pack(side="left", padx=(0,5))
            ctk.CTkButton(btn, text="📂", fg_color="#444", width=40, command=lambda: os.startfile(install_path)).pack(side="right")
        return f

    def check_and_launch_install(self, url, path, engine_name, wipe=False):
        if wipe:
            if not self.force_delete_folder(path): return
            os.makedirs(path, exist_ok=True)
        elif os.path.exists(path) and os.listdir(path) and not os.path.exists(os.path.join(path, ".git")):
             if messagebox.askyesno("Corrompu", "Dossier invalide. Réinstaller ?"): self.wipe_and_full_install(url, path, engine_name); return

        py = self.get_portable_python_path()
        if not py:
            if messagebox.askyesno("Requis", "Python 3.11 Portable requis. Télécharger ?"): self.install_portable_python()
            return
        self.launch_engine_install_script(url, path, engine_name, py)

    def wipe_and_full_install(self, url, path, engine_name):
        if messagebox.askyesno("Confirm", "Tout supprimer ?"): self.check_and_launch_install(url, path, engine_name, wipe=True)

    def launch_console(self, title, command, cwd):
        if not shutil.which("git"): return messagebox.showerror("Err", "Git not found")
        ConsolePopup(self.winfo_toplevel(), title, command, cwd, on_close_callback=self.refresh_ui)

    def refresh_ui(self):
        self.setup_engines_tab(); self.setup_system_tab()

    def setup_system_tab(self):
        for w in self.tab_sys.winfo_children(): w.destroy()
        self.tab_sys.grid_columnconfigure(0, weight=1); self.tab_sys.grid_columnconfigure(1, weight=1)

        f_top = ctk.CTkFrame(self.tab_sys); f_top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        ctk.CTkLabel(f_top, text=_t("Configuration Système & Runtime", "System & Runtime Configuration"), font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        row_1 = ctk.CTkFrame(f_top, fg_color="transparent"); row_1.pack(fill="x", padx=10, pady=5)
        
        # CHARGEMENT DES PREFERENCES
        curr_mode = self.settings.get("appearance_mode", "System")
        curr_theme = self.settings.get("theme_color", "blue")
        use_aida = self.settings.get("use_aida64") == "true"

        # THEMES PERSONNALISÉS
        theme_list = self.get_available_themes()

        ctk.CTkLabel(row_1, text=_t("Mode :", "Mode:")).pack(side="left")
        opt_mode = ctk.CTkOptionMenu(row_1, values=["System", "Dark", "Light"], width=100, command=self.change_appearance)
        opt_mode.set(curr_mode)
        opt_mode.pack(side="left", padx=5)

        ctk.CTkLabel(row_1, text=_t("Thème :", "Theme:")).pack(side="left")
        opt_col = ctk.CTkOptionMenu(row_1, values=theme_list, width=150, command=self.change_color)
        opt_col.set(curr_theme)
        opt_col.pack(side="left", padx=5)

        self.chk_aida = ctk.CTkCheckBox(row_1, text=_t("Activer AIDA64", "Enable AIDA64"), command=self.save_aida)
        if use_aida: self.chk_aida.select()
        else: self.chk_aida.deselect()
        self.chk_aida.pack(side="right", padx=10)

        row_py = ctk.CTkFrame(f_top, fg_color="transparent", height=30); row_py.pack(fill="x", padx=10, pady=(5, 10))
        py = self.get_portable_python_path()
        st, col = ("✅ Python 3.11 Portable (Prêt)", "#2ecc71") if py else ("❌ Python Portable manquant", "#e74c3c")
        ctk.CTkLabel(row_py, text=st, text_color=col, font=("Consolas", 12, "bold")).pack(side="left")
        if not py: ctk.CTkButton(row_py, text="📥 Télécharger", height=24, fg_color="#8e44ad", command=self.install_portable_python).pack(side="right")

        # ── GPU & PyTorch advisory ──────────────────────────────────────────
        self._gpu_advisory_frame = ctk.CTkFrame(f_top, fg_color="#1a1a2e", corner_radius=8)
        self._gpu_advisory_frame.pack(fill="x", padx=10, pady=(0, 12))
        ctk.CTkLabel(self._gpu_advisory_frame, text="GPU & PyTorch",
                     font=("Roboto", 13, "bold"), text_color="#3B8ED0").pack(anchor="w", padx=12, pady=(8, 2))
        self._gpu_advisory_body = ctk.CTkFrame(self._gpu_advisory_frame, fg_color="transparent")
        self._gpu_advisory_body.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(self._gpu_advisory_body, text=_t("Détection GPU…", "Detecting GPU…"), text_color="gray").pack(anchor="w")
        threading.Thread(target=self._load_gpu_advisory, daemon=True).start()

        self.create_dashboard_card(self.tab_sys, "NeoSR", self.neosr_path, 0)
        self.create_dashboard_card(self.tab_sys, "TraiNNer-Redux", self.redux_path, 1)

    def create_dashboard_card(self, parent, name, path, col):
        f = ctk.CTkFrame(parent, border_width=2, border_color="#333")
        f.grid(row=1, column=col, sticky="nsew", padx=10, pady=10)
        
        head = ctk.CTkFrame(f, fg_color="transparent"); head.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(head, text=name, font=("Roboto", 16, "bold"), text_color="#3B8ED0").pack(side="left")
        
        f_st = ctk.CTkFrame(f, fg_color="transparent"); f_st.pack(fill="both", expand=True, padx=10)
        f_act = ctk.CTkFrame(f, fg_color="transparent", height=50); f_act.pack(fill="x", padx=10, pady=10)

        venv = os.path.join(path, ".venv")
        py_venv = os.path.join(venv, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(venv, "bin", "python")
        
        if not os.path.exists(os.path.join(path, "train.py")):
            ctk.CTkLabel(f_st, text="⚠️ FICHIERS MANQUANTS", font=("Arial", 14, "bold"), text_color="orange").pack(pady=20)
            ctk.CTkButton(f_act, text="⬇ INSTALLER", fg_color="#e67e22", command=lambda: self.check_and_launch_install(self.repos[name], path, name)).pack(fill="x")
            self.draw_full_grid(f_st, name, py_venv)
            return

        if not os.path.exists(py_venv):
            ctk.CTkLabel(f_st, text="❌ VENV ABSENT", font=("Arial", 14, "bold"), text_color="#e74c3c").pack(pady=10)
            ctk.CTkButton(f_act, text="🛠️ CRÉER VENV", fg_color="#8e44ad", height=40, command=lambda: self.check_and_launch_install(self.repos[name], path, name)).pack(fill="x")
            self.draw_full_grid(f_st, name, py_venv)
            return

        self.draw_full_grid(f_st, name, py_venv) 
        lbl = ctk.CTkLabel(f_act, text="Scan...", text_color="gray"); lbl.pack()
        threading.Thread(target=lambda: self.check_modules_thread(name, py_venv, path, f_act, lbl), daemon=True).start()

    def draw_full_grid(self, parent, engine_name, py_venv):
        for w in parent.winfo_children(): w.destroy()
        grid = ctk.CTkFrame(parent, fg_color="transparent"); grid.pack(fill="both", expand=True, pady=10)
        self.status_labels[engine_name] = {}
        full = self.system_tools + self.python_modules
        for i, mod in enumerate(full):
            r, c = i // 2, i % 2
            f = ctk.CTkFrame(grid, fg_color="#222", corner_radius=6)
            f.grid(row=r, column=c, sticky="ew", padx=5, pady=5)
            grid.grid_columnconfigure(c, weight=1)
            ctk.CTkLabel(f, text=mod.upper(), font=("Consolas", 11, "bold")).pack(side="left", padx=10)
            txt, col = ("...", "gray")
            if mod in self.system_tools:
                ok = shutil.which(mod) is not None
                txt, col = ("✅ OK", "#2ecc71") if ok else ("❌", "#e74c3c")
            l = ctk.CTkLabel(f, text=txt, text_color=col, font=("Arial", 10))
            l.pack(side="right", padx=10)
            self.status_labels[engine_name][mod] = l

    # --- VERIFICATION ROBUSTE ---
    def check_modules_thread(self, name, py_venv, cwd, action_frame, load_lbl):
        missing = 0
        pkg_map = {"cv2": "opencv-python", "yaml": "PyYAML", "PIL": "Pillow", "pywt": "PyWavelets", "rich": "rich", "ema_pytorch": "ema-pytorch"}
        checker_path = os.path.join(cwd, "_check_modules.py")

        # PERF-04: Batch all checks into ONE subprocess call (was N separate calls)
        imp_map = {}
        for mod in self.python_modules:
            imp = "cv2" if mod == "cv2" else ("pywt" if mod == "pywt" else ("PIL" if mod == "pillow" else mod))
            pkg = pkg_map.get(imp, imp)
            imp_map[mod] = (imp, pkg)

        check_code = "import sys, json\nresults = {}\n"
        for mod, (imp, pkg) in imp_map.items():
            check_code += f"""
try:
    import {imp}
    v = None
    try: v = {imp}.__version__
    except Exception: pass
    if not v:
        try:
            from importlib.metadata import version
            v = version(\'{pkg}\')
        except Exception: pass
    results[\'{mod}\'] = v if v else \'OK\'
except ImportError:
    results[\'{mod}\'] = \'MISSING\'
"""
        check_code += "\nprint(json.dumps(results))\n"

        try:
            with open(checker_path, "w", encoding="utf-8") as f:
                f.write(check_code)
            res = subprocess.run([py_venv, checker_path], capture_output=True, text=True, timeout=30,
                                creationflags=0x08000000 if sys.platform == 'win32' else 0)
            if res.returncode == 0 and res.stdout.strip():
                import json
                results = json.loads(res.stdout.strip())
                for mod, status in results.items():
                    if status == "MISSING":
                        missing += 1
                        self.after(0, lambda n=name, m=mod: self.status_labels[n][m].configure(text="X", text_color="#e74c3c"))
                    else:
                        self.after(0, lambda n=name, m=mod, v=status: self.status_labels[n][m].configure(text=f"v{v}", text_color="#2ecc71"))
            else:
                for mod in self.python_modules:
                    missing += 1
                    self.after(0, lambda n=name, m=mod: self.status_labels[n][m].configure(text="?", text_color="#f39c12"))
        except Exception:
            for mod in self.python_modules:
                missing += 1
                self.after(0, lambda n=name, m=mod: self.status_labels[n][m].configure(text="?", text_color="#f39c12"))

        if os.path.exists(checker_path):
            try: os.remove(checker_path)
            except Exception: pass

        self.after(0, lambda: self.update_action_buttons(action_frame, load_lbl, missing, cwd, py_venv, name))

    def update_action_buttons(self, frame, lbl, missing_count, cwd, py_venv, name):
        lbl.destroy()
        if missing_count > 0:
            # SI FICHIERS PRESENTS MAIS MODULES MANQUANTS -> BOUTON ORANGE "UPDATE"
            ctk.CTkButton(frame, text=f"📦 INSTALLER MANQUANTS ({missing_count})", fg_color="#e67e22", height=35,
                          command=lambda: self.install_missing_deps_only(name, cwd, py_venv)).pack(fill="x")
        else:
            ctk.CTkLabel(frame, text="✨ TOUT EST PRÊT", text_color="#2ecc71", font=("Arial", 12, "bold")).pack(pady=5)
            # Bouton de secours au cas où
            ctk.CTkButton(frame, text="Forcer Réinstallation", fg_color="#444", height=24,
                          command=lambda: self.check_and_launch_install(self.repos[name], cwd, name)).pack(fill="x", pady=(5,0))
    # ─── LANGUAGE TAB ────────────────────────────────────────
    def setup_notifications_tab(self):
        f = self.tab_lang
        ctk.CTkLabel(f, text=_t("Notifications & Son", "Notifications & Sound"), font=("Roboto", 18, "bold")).pack(pady=(20, 5))
        ctk.CTkLabel(f, text=_t("Configure les alertes sonores et visuelles.", "Configure sound and visual alerts."), text_color="#AAA").pack(pady=(0, 10))

        # ── Windows 11 notifications section ──────────────────────────
        import tkinter as tk
        win_frame = ctk.CTkFrame(f, fg_color="#1a1a2e", corner_radius=8)
        win_frame.pack(fill="x", padx=20, pady=(0, 10))
        ctk.CTkLabel(win_frame, text=_t("Notifications Windows 11", "Windows 11 Notifications"),
                     font=("Roboto", 14, "bold"), text_color="#3498db").pack(
                     anchor="w", padx=15, pady=(10, 2))
        ctk.CTkLabel(win_frame, text=_t("Afficher des notifications systeme pour :", "Show system notifications for:"),
                     text_color="#AAA", font=("Arial", 11)).pack(anchor="w", padx=15, pady=(0, 8))

        _notif_items = [
            ("notif_win11_upscale",  _t("Upscale image termine", "Image upscale finished")),
            ("notif_win11_batch",    _t("Batch upscale termine", "Batch upscale finished")),
            ("notif_win11_errors",   _t("Erreurs & crashes", "Errors & crashes")),
            ("notif_win11_training", _t("Fin d'entrainement", "Training finished")),
        ]
        self._notif_vars = {}
        for key, label in _notif_items:
            var = tk.BooleanVar(value=self.settings.get(key, True))
            self._notif_vars[key] = var
            chk = ctk.CTkCheckBox(
                win_frame, text=label, variable=var,
                command=lambda k=key, v=var: self.settings.set(k, v.get()))
            chk.pack(anchor="w", padx=30, pady=3)

        def _test_win11():
            try:
                from src.core.toast_notifications import show_toast
                ok = show_toast("Universal SR Studio",
                                "Notification de test - tout fonctionne !")
                if not ok:
                    messagebox.showinfo("Notification",
                        "Aucun backend disponible.\n"
                        "Installez win11toast :  pip install win11toast")
            except Exception as e:
                messagebox.showerror("Notification", str(e))

        ctk.CTkButton(win_frame, text=_t("Tester une notification", "Test notification"),
                      fg_color="#2980b9", hover_color="#1a6fa0", width=200,
                      command=_test_win11).pack(anchor="w", padx=15, pady=(8, 12))

        # Sound section
        snd_frame = ctk.CTkFrame(f, fg_color="#1a1a2e", corner_radius=8)
        snd_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(snd_frame, text=_t("Son de fin d'entrainement", "Training completion sound"), font=("Roboto", 14, "bold"),
                     text_color="#3498db").pack(anchor="w", padx=15, pady=(10, 5))

        # Checkbox
        import tkinter as tk
        self._sound_enabled = tk.BooleanVar(value=self.settings.get("sound_enabled", True))
        chk = ctk.CTkCheckBox(snd_frame, text=_t("Jouer un son quand un entrainement se termine", "Play a sound when training finishes"),
                               variable=self._sound_enabled,
                               command=lambda: self.settings.set("sound_enabled", self._sound_enabled.get()))
        chk.pack(anchor="w", padx=15, pady=5)

        # Volume slider
        vol_frame = ctk.CTkFrame(snd_frame, fg_color="transparent")
        vol_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(vol_frame, text=_t("Volume :", "Volume:"), width=80, anchor="w").pack(side="left")
        self._volume_val = tk.IntVar(value=int(self.settings.get("sound_volume", 70)))
        self._vol_label = ctk.CTkLabel(vol_frame, text=f"{self._volume_val.get()}%", width=40)
        self._vol_label.pack(side="right", padx=5)
        self._vol_slider = ctk.CTkSlider(vol_frame, from_=0, to=100, variable=self._volume_val,
                                          command=self._on_volume_change)
        self._vol_slider.pack(side="left", fill="x", expand=True, padx=10)

        # Test button
        ctk.CTkButton(snd_frame, text=_t("🔊 Tester le son", "🔊 Test sound"), fg_color="#27ae60", width=150,
                      command=self._test_sound).pack(anchor="w", padx=15, pady=(5, 15))

        # Sound file info
        import sys
        sound_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "assets", "success.wav")
        if os.path.exists(sound_path):
            size_kb = os.path.getsize(sound_path) // 1024
            ctk.CTkLabel(f, text=f"Fichier son: assets/success.wav ({size_kb} Ko)",
                         text_color="#666", font=("Roboto", 10)).pack(pady=5)
        else:
            ctk.CTkLabel(f, text="⚠ Fichier assets/success.wav non trouve",
                         text_color="#e74c3c", font=("Roboto", 10)).pack(pady=5)

        # ── Sons d'événements (error / warning / about) ───────────────
        evt_frame = ctk.CTkFrame(f, fg_color="#1a1a2e", corner_radius=8)
        evt_frame.pack(fill="x", padx=20, pady=(5, 15))
        ctk.CTkLabel(evt_frame, text="Sons d'evenements",
                     font=("Roboto", 14, "bold"), text_color="#3498db").pack(
                     anchor="w", padx=15, pady=(10, 4))

        _evt_sounds = [
            ("sound_error_enabled",   "error",   "Son d'erreur grave  (crash, erreur critique)"),
            ("sound_warning_enabled", "warning", "Son d'avertissement (erreur non fatale dans un batch)"),
            ("sound_about_enabled",   "about",   "Son de la section A Propos"),
        ]
        self._evt_sound_vars = {}
        for key, snd, label in _evt_sounds:
            var = tk.BooleanVar(value=self.settings.get(key, True))
            self._evt_sound_vars[key] = var
            row = ctk.CTkFrame(evt_frame, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=3)
            chk = ctk.CTkCheckBox(row, text=label, variable=var,
                                   command=lambda k=key, v=var: self.settings.set(k, v.get()))
            chk.pack(side="left", fill="x", expand=True)
            ctk.CTkButton(row, text="🔊 Test", width=70,
                          command=lambda s=snd: self._test_event_sound(s)).pack(side="right")

        ctk.CTkLabel(evt_frame,
                     text="Fichiers : assets/error.WAV  •  warning.WAV  •  about.WAV",
                     text_color="#555", font=("Roboto", 9)).pack(anchor="w", padx=15, pady=(4, 10))

    def _test_event_sound(self, sound_name: str):
        """Play one of the event WAV files for testing."""
        import sys
        snd_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                                "assets", f"{sound_name}.WAV")
        if not os.path.isfile(snd_path):
            messagebox.showwarning("Son", f"Fichier assets/{sound_name}.WAV introuvable")
            return
        try:
            import winsound
            winsound.PlaySound(snd_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            messagebox.showerror("Son", f"Erreur lecture: {e}")

    def _play_about_sound(self):
        """Play about.WAV when the A Propos tab becomes visible."""
        if not self.settings.get("sound_about_enabled", True):
            return
        import sys
        snd_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                                "assets", "about.WAV")
        if not os.path.isfile(snd_path):
            return
        try:
            import winsound
            winsound.PlaySound(snd_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass

    def _on_volume_change(self, val):
        v = int(float(val))
        self._vol_label.configure(text=f"{v}%")
        self.settings.set("sound_volume", v)

    def _test_sound(self):
        import sys
        sound_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "assets", "success.wav")
        if not os.path.exists(sound_path):
            messagebox.showwarning("Son", "Fichier assets/success.wav introuvable")
            return
        volume = self._volume_val.get()
        try:
            if sys.platform == "win32":
                import ctypes
                winmm = ctypes.windll.winmm
                vol = int(volume / 100 * 0xFFFF)
                winmm.waveOutSetVolume(0, vol | (vol << 16))
                import winsound
                winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                import subprocess
                subprocess.Popen(["aplay", "-q", sound_path], stderr=subprocess.DEVNULL)
        except Exception as e:
            messagebox.showerror("Son", f"Erreur lecture: {e}")

    def setup_apikeys_tab(self):
        f = self.tab_apikeys
        ctk.CTkLabel(f, text=_t("Cles API pour la Verification AI", "API Keys for AI Review"), font=("Roboto", 18, "bold")).pack(pady=(20, 5))
        ctk.CTkLabel(f, text=_t("Entrez vos cles API ici. Elles seront sauvegardees localement\net chargees automatiquement dans Configuration > Verification AI.",
                                "Enter your API keys here. They are saved locally\nand auto-loaded in Configuration > AI Review."),
                     text_color="#AAA").pack(pady=(0, 15))

        self._api_key_entries = {}
        providers = [
            ("OpenRouter (Gratuit)", "https://openrouter.ai/settings/keys"),
            ("GitHub Models (Gratuit)", "https://github.com/settings/tokens"),
            ("Google (Gemini)", "https://aistudio.google.com/apikey"),
            ("Anthropic (Claude)", "https://console.anthropic.com/settings/keys"),
            ("OpenAI (ChatGPT)", "https://platform.openai.com/api-keys"),
            ("xAI (Grok)", "https://console.x.ai"),
            ("DeepSeek", "https://platform.deepseek.com/api_keys"),
        ]
        for provider, url in providers:
            frame = ctk.CTkFrame(f, fg_color="#1a1a2e", corner_radius=8)
            frame.pack(fill="x", padx=20, pady=5)

            row = ctk.CTkFrame(frame, fg_color="transparent"); row.pack(fill="x", padx=10, pady=8)
            ctk.CTkLabel(row, text=provider, font=("Roboto", 12, "bold"), width=180, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, width=350, show="*", placeholder_text="sk-... / AIza... / xai-...")
            entry.pack(side="left", padx=5)
            self._api_key_entries[provider] = entry

            # Load saved key
            saved = self.settings.get(f"api_key_{provider}", "")
            if saved:
                entry.insert(0, saved)

            url_label = ctk.CTkLabel(row, text=url, text_color="#3498db", font=("Roboto", 9),
                         cursor="hand2")
            url_label.pack(side="left", padx=10)
            url_label.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

        # Save button
        def save_all_keys():
            for prov, entry in self._api_key_entries.items():
                key = entry.get().strip()
                if key:
                    self.settings.set(f"api_key_{prov}", key)
                else:
                    # Clear the key if empty
                    self.settings.set(f"api_key_{prov}", "")
            from tkinter import messagebox
            messagebox.showinfo(_t("Cles API", "API Keys"), _t("Cles API sauvegardees avec succes !", "API keys saved successfully!"))

        ctk.CTkButton(f, text=_t("Sauvegarder les cles API", "Save API keys"), fg_color="#27ae60", height=35,
                      font=("Roboto", 13, "bold"), command=save_all_keys).pack(pady=15)

        ctk.CTkLabel(f, text=_t("Les cles sont stockees localement dans user_settings.json.\nElles ne sont jamais envoyees ailleurs qu'au fournisseur AI choisi.",
                                "Keys are stored locally in user_settings.json.\nThey are never sent anywhere except to the chosen AI provider."),
                     text_color="#666", font=("Roboto", 9)).pack(pady=(0, 10))

    def setup_language_tab(self):
        f = self.tab_lang
        ctk.CTkLabel(f, text="Langue / Language", font=("Roboto", 18, "bold")).pack(pady=(20, 10))
        ctk.CTkLabel(f, text=_t("Choisissez la langue du programme.\nLes options, infobulles et descriptions seront traduites.",
                                "Choose the interface language.\nOptions, tooltips and descriptions will be translated."),
                     text_color="#AAA").pack(pady=(0, 15))

        lang_frame = ctk.CTkFrame(f, fg_color="transparent")
        lang_frame.pack(pady=10)

        ctk.CTkLabel(lang_frame, text="Langue :").pack(side="left", padx=(0, 10))
        current_lang = self.settings.get("language", "fr")
        self.lang_var = ctk.StringVar(value=current_lang)
        lang_menu = ctk.CTkOptionMenu(
            lang_frame, values=["Français", "English"],
            variable=self.lang_var, command=self._on_lang_change, width=150
        )
        lang_menu.pack(side="left")
        lang_menu.set("Français" if current_lang == "fr" else "English")

        self.lbl_lang_info = ctk.CTkLabel(f, text="", text_color="#e67e22")
        self.lbl_lang_info.pack(pady=10)

    def _on_lang_change(self, choice):
        lang_code = "fr" if choice == "Français" else "en"
        self.settings.set("language", lang_code)

        try:
            from src.core.translations import set_language
            set_language(lang_code)
            self.lbl_lang_info.configure(
                text=f"✅ Langue changée : {choice}\n⚠️ Redémarrez l'application pour appliquer complètement."
            )
        except ImportError:
            self.lbl_lang_info.configure(
                text=f"✅ Langue enregistrée : {choice}\n⚠️ Redémarrez l'application pour appliquer."
            )


    # ─── ABOUT TAB ───────────────────────────────────────────
    def setup_gallery_tab(self):
        """Galerie HTTP server + NeoSR/Redux TensorBoard image patch."""
        # Lazy imports
        try:
            from src.core.gallery_server import get_server
            from src.core.tb_image_patch import find_validation_file, patch_file, unpatch_file, is_patched, get_patch_status
            from src.core.qr_code import is_qrcode_available, generate_qr_image
        except Exception as e:
            ctk.CTkLabel(self.tab_gallery, text=f"Erreur import : {e}",
                         text_color="#e74c3c").pack(pady=20)
            return

        # Use a scrollable frame because the content can be tall
        scroll = ctk.CTkScrollableFrame(self.tab_gallery, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(scroll, text=_t("Galerie HTTP & Patch TensorBoard", "HTTP Gallery & TensorBoard Patch"),
                     font=("Roboto", 16, "bold"), text_color="#3498db"
                     ).pack(anchor="w", padx=10, pady=(5, 10))

        # ─── Section A: HTTP Gallery ───
        sec_a = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=8)
        sec_a.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(sec_a, text=_t("A. Serveur Galerie Web (sans modifier NeoSR)", "A. Web Gallery Server (no NeoSR changes needed)"),
                     font=("Roboto", 13, "bold"), text_color="#3498db"
                     ).pack(anchor="w", padx=10, pady=(8, 5))
        ctk.CTkLabel(sec_a,
                     text=_t("Lance un mini-serveur HTTP sur un dossier d'images. Compatible mobile, "
                             "auto-refresh, zoom au clic. Tunnel Ngrok optionnel pour accès distant.",
                             "Starts a mini HTTP server on an image folder. Mobile-friendly, "
                             "auto-refresh, click-to-zoom. Optional Ngrok tunnel for remote access."),
                     text_color="#AAA", font=("Roboto", 10), justify="left", wraplength=900
                     ).pack(anchor="w", padx=10, pady=(0, 10))

        # Directory picker — auto-fill from settings, then fall back to last experiment's visualization
        dir_row = ctk.CTkFrame(sec_a, fg_color="transparent"); dir_row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(dir_row, text=_t("Dossier :", "Folder:"), width=80, anchor="w").pack(side="left")
        self._gal_dir_entry = ctk.CTkEntry(dir_row, width=600)
        self._gal_dir_entry.pack(side="left", padx=5)
        _saved_gal_dir = self.settings.get("gallery_auto_dir", "")
        if _saved_gal_dir:
            self._gal_dir_entry.insert(0, _saved_gal_dir)

        def _refresh_gal_dir_from_settings(event=None):
            """Sync entry with latest settings value (updated by RunTab when config loads)."""
            current = self._gal_dir_entry.get().strip()
            fresh = self.settings.get("gallery_auto_dir", "")
            if fresh and fresh != current:
                self._gal_dir_entry.delete(0, "end")
                self._gal_dir_entry.insert(0, fresh)

        self._gal_dir_entry.bind("<FocusIn>", _refresh_gal_dir_from_settings)
        # Also refresh when the parent frame becomes visible (tab switch)
        try:
            sec_a.bind("<Visibility>", _refresh_gal_dir_from_settings)
        except Exception:
            pass

        if not _saved_gal_dir:
            try:
                default_root = self.neosr_path
                exp_root = os.path.join(default_root, "experiments")
                if os.path.isdir(exp_root):
                    exps = [(d, os.path.getmtime(os.path.join(exp_root, d)))
                            for d in os.listdir(exp_root)
                            if os.path.isdir(os.path.join(exp_root, d)) and not d.startswith("_")]
                    if exps:
                        latest = max(exps, key=lambda x: x[1])[0]
                        vis = os.path.join(exp_root, latest, "visualization")
                        if os.path.isdir(vis):
                            self._gal_dir_entry.insert(0, vis)
            except Exception:
                pass

        def _browse_gal_dir():
            d = filedialog.askdirectory()
            if d:
                self._gal_dir_entry.delete(0, "end")
                self._gal_dir_entry.insert(0, d)
                self.settings.set("gallery_auto_dir", d)

        ctk.CTkButton(dir_row, text="...", width=30,
                      command=_browse_gal_dir).pack(side="left", padx=2)

        # Options
        opt_row = ctk.CTkFrame(sec_a, fg_color="transparent"); opt_row.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(opt_row, text="Port :", width=80, anchor="w").pack(side="left")
        self._gal_port_entry = ctk.CTkEntry(opt_row, width=80)
        self._gal_port_entry.pack(side="left", padx=5)
        self._gal_port_entry.insert(0, str(self.settings.get("gallery_port", "8765")))
        self._gal_ngrok_var = ctk.CTkCheckBox(
            opt_row, text="Tunnel Ngrok (accès distant)",
            command=lambda: self.settings.set("gallery_ngrok", bool(self._gal_ngrok_var.get()))
        )
        if self.settings.get("gallery_ngrok", False):
            self._gal_ngrok_var.select()
        self._gal_ngrok_var.pack(side="left", padx=20)

        # Auto-start / auto-stop checkboxes
        auto_row = ctk.CTkFrame(sec_a, fg_color="transparent"); auto_row.pack(fill="x", padx=10, pady=2)
        self._gal_auto_start_var = ctk.CTkCheckBox(
            auto_row,
            text="Auto-démarrer la galerie au lancement de l'entrainement",
            command=lambda: self.settings.set("gallery_auto_start_with_training",
                                              bool(self._gal_auto_start_var.get()))
        )
        if self.settings.get("gallery_auto_start_with_training", False):
            self._gal_auto_start_var.select()
        self._gal_auto_start_var.pack(side="left", padx=5)

        auto_row2 = ctk.CTkFrame(sec_a, fg_color="transparent"); auto_row2.pack(fill="x", padx=10, pady=2)
        self._gal_auto_stop_var = ctk.CTkCheckBox(
            auto_row2,
            text="Auto-arrêter la galerie à l'arrêt de l'entrainement",
            command=lambda: self.settings.set("gallery_auto_stop_with_training",
                                              bool(self._gal_auto_stop_var.get()))
        )
        if self.settings.get("gallery_auto_stop_with_training", False):
            self._gal_auto_stop_var.select()
        self._gal_auto_stop_var.pack(side="left", padx=5)

        auto_row3 = ctk.CTkFrame(sec_a, fg_color="transparent"); auto_row3.pack(fill="x", padx=10, pady=2)
        self._tb_auto_start_var = ctk.CTkCheckBox(
            auto_row3,
            text="Auto-démarrer TensorBoard au lancement de l'entrainement (démarre le serveur TB)",
            command=lambda: self.settings.set("tb_auto_start_with_training",
                                              bool(self._tb_auto_start_var.get()))
        )
        if self.settings.get("tb_auto_start_with_training", False):
            self._tb_auto_start_var.select()
        self._tb_auto_start_var.pack(side="left", padx=5)

        auto_row4 = ctk.CTkFrame(sec_a, fg_color="transparent"); auto_row4.pack(fill="x", padx=10, pady=2)
        # Persist the default value the FIRST time so new installs start with a known state
        if "tb_auto_open_browser" not in self.settings.data:
            self.settings.set("tb_auto_open_browser", True)
        self._tb_auto_browser_var = ctk.CTkCheckBox(
            auto_row4,
            text="Auto-ouvrir le navigateur sur http://localhost:6006 quand TB est prêt",
            command=lambda: self.settings.set("tb_auto_open_browser",
                                              bool(self._tb_auto_browser_var.get()))
        )
        if self.settings.get("tb_auto_open_browser", True):
            self._tb_auto_browser_var.select()
        self._tb_auto_browser_var.pack(side="left", padx=5)

        auto_row5 = ctk.CTkFrame(sec_a, fg_color="transparent"); auto_row5.pack(fill="x", padx=10, pady=2)
        if "ngrok_auto_from_config" not in self.settings.data:
            self.settings.set("ngrok_auto_from_config", False)
        self._ngrok_auto_var = ctk.CTkCheckBox(
            auto_row5,
            text="Autoriser le lancement automatique de Ngrok depuis la config (monitoring.auto_ngrok)",
            command=lambda: self.settings.set("ngrok_auto_from_config",
                                              bool(self._ngrok_auto_var.get()))
        )
        if self.settings.get("ngrok_auto_from_config", False):
            self._ngrok_auto_var.select()
        self._ngrok_auto_var.pack(side="left", padx=5)

        # Buttons
        btn_row = ctk.CTkFrame(sec_a, fg_color="transparent"); btn_row.pack(fill="x", padx=10, pady=8)
        self._gal_btn_start = ctk.CTkButton(btn_row, text="▶ Démarrer", fg_color="#27ae60",
                                             width=140, command=self._gal_start_clicked)
        self._gal_btn_start.pack(side="left", padx=5)
        self._gal_btn_stop = ctk.CTkButton(btn_row, text="⏹ Arrêter", fg_color="#e74c3c",
                                            width=120, command=self._gal_stop_clicked, state="disabled")
        self._gal_btn_stop.pack(side="left", padx=5)
        self._gal_btn_open = ctk.CTkButton(btn_row, text="🌐 Ouvrir", fg_color="#3498db",
                                            width=120, command=self._gal_open_clicked, state="disabled")
        self._gal_btn_open.pack(side="left", padx=5)

        # Status + QR area
        self._gal_status_label = ctk.CTkLabel(sec_a, text="État : Arrêté",
                                                text_color="#888", anchor="w", justify="left",
                                                font=("Consolas", 11))
        self._gal_status_label.pack(anchor="w", fill="x", padx=10, pady=(5, 5))
        self._gal_qr_frame = ctk.CTkFrame(sec_a, fg_color="transparent")
        self._gal_qr_frame.pack(fill="x", padx=10, pady=(0, 10))

        if not is_qrcode_available():
            ctk.CTkLabel(self._gal_qr_frame,
                         text="💡 Installer 'qrcode' pour afficher un QR code scannable au démarrage : pip install qrcode[pil]",
                         text_color="#666", font=("Roboto", 9)).pack(anchor="w")

        # ─── Section B: NeoSR/Redux TB image patch ───
        sec_b = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=8)
        sec_b.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(sec_b, text="B. Patch NeoSR/Redux pour images TensorBoard",
                     font=("Roboto", 13, "bold"), text_color="#9b59b6"
                     ).pack(anchor="w", padx=10, pady=(8, 5))
        ctk.CTkLabel(sec_b,
                     text="Modifie le CODE SOURCE de NeoSR ou traiNNer-Redux pour qu'il envoie\n"
                          "les images de validation dans TensorBoard (en plus de les sauvegarder sur disque).\n"
                          "→ Pointez sur la RACINE de l'engine (ex: C:/Users/.../IA_Engine/neosr),\n"
                          "  pas un dossier d'expérimentation/logs.\n"
                          "Idempotent et réversible (backup .usr_bak créé). Max 4 images par validation.",
                     text_color="#AAA", font=("Roboto", 10), justify="left", wraplength=900
                     ).pack(anchor="w", padx=10, pady=(0, 10))

        # Engine path picker
        path_row = ctk.CTkFrame(sec_b, fg_color="transparent"); path_row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(path_row, text="Engine :", width=80, anchor="w").pack(side="left")
        self._tbp_path_entry = ctk.CTkEntry(path_row, width=550)
        self._tbp_path_entry.pack(side="left", padx=5)
        # Default to NeoSR path from settings
        self._tbp_path_entry.insert(0, self.neosr_path)

        def _browse_engine():
            d = filedialog.askdirectory()
            if d:
                self._tbp_path_entry.delete(0, "end")
                self._tbp_path_entry.insert(0, d)

        ctk.CTkButton(path_row, text="...", width=30, command=_browse_engine
                      ).pack(side="left", padx=2)

        # Quick selectors
        quick_row = ctk.CTkFrame(sec_b, fg_color="transparent"); quick_row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(quick_row, text="Raccourcis :", anchor="w").pack(side="left", padx=(80, 5))

        def _set_neosr():
            self._tbp_path_entry.delete(0, "end"); self._tbp_path_entry.insert(0, self.neosr_path)
            self._tbp_refresh()

        def _set_redux():
            self._tbp_path_entry.delete(0, "end"); self._tbp_path_entry.insert(0, self.redux_path)
            self._tbp_refresh()

        ctk.CTkButton(quick_row, text="NeoSR", fg_color="#666", width=100,
                      command=_set_neosr).pack(side="left", padx=2)
        ctk.CTkButton(quick_row, text="traiNNer-Redux", fg_color="#666", width=140,
                      command=_set_redux).pack(side="left", padx=2)
        ctk.CTkButton(quick_row, text="🔄 Vérifier statut", fg_color="#3498db",
                      width=140, command=self._tbp_refresh).pack(side="left", padx=10)

        # Status
        self._tbp_status_label = ctk.CTkLabel(sec_b, text="(non vérifié)",
                                                text_color="#888", anchor="w", justify="left",
                                                font=("Consolas", 10), wraplength=900)
        self._tbp_status_label.pack(anchor="w", fill="x", padx=10, pady=5)

        # Action buttons
        tbp_btns = ctk.CTkFrame(sec_b, fg_color="transparent"); tbp_btns.pack(fill="x", padx=10, pady=8)
        ctk.CTkButton(tbp_btns, text="✅ Appliquer le Patch", fg_color="#27ae60",
                      width=180, command=self._tbp_apply_clicked).pack(side="left", padx=5)
        ctk.CTkButton(tbp_btns, text="❌ Retirer le Patch", fg_color="#e74c3c",
                      width=180, command=self._tbp_remove_clicked).pack(side="left", padx=5)

        # Initial check
        self.after(200, self._tbp_refresh)

    # ── Gallery actions ──
    def _gal_start_clicked(self):
        from src.core.gallery_server import get_server
        from src.core.qr_code import generate_qr_image, is_qrcode_available
        from PIL import Image as _PImage, ImageTk
        import tkinter as tk

        directory = self._gal_dir_entry.get().strip()
        port_str = self._gal_port_entry.get().strip()
        with_ngrok = self._gal_ngrok_var.get()

        # Persist current values to settings
        if directory:
            self.settings.set("gallery_auto_dir", directory)
        if port_str:
            self.settings.set("gallery_port", port_str)
        self.settings.set("gallery_ngrok", bool(with_ngrok))

        if not directory or not os.path.isdir(directory):
            messagebox.showerror("Erreur", "Sélectionnez un dossier valide.")
            return
        try:
            port = int(port_str) if port_str else 0
        except ValueError:
            messagebox.showerror("Erreur", "Port invalide.")
            return

        srv = get_server()
        result = srv.start(directory, port=port, with_ngrok=bool(with_ngrok))
        if not result.get("ok"):
            messagebox.showerror("Erreur", result.get("error", "Échec inconnu"))
            return

        self._gal_btn_start.configure(state="disabled")
        self._gal_btn_stop.configure(state="normal")
        self._gal_btn_open.configure(state="normal")

        url_for_qr = result.get("ngrok_url") or result.get("local_url")
        lines = [f"✅ Serveur actif", f"   Local : {result['local_url']}"]
        if result.get("ngrok_url"):
            lines.append(f"   Public : {result['ngrok_url']}")
        elif with_ngrok and result.get("ngrok_warning"):
            lines.append(f"   ⚠ {result['ngrok_warning']}")
        lines.append(f"   Dossier : {directory}")
        self._gal_status_label.configure(text="\n".join(lines), text_color="#2ecc71")

        # QR code
        for w in self._gal_qr_frame.winfo_children():
            w.destroy()
        if is_qrcode_available():
            qr_path = os.path.join(os.path.expanduser("~"), ".usr_studio_qr.png")
            if generate_qr_image(url_for_qr, qr_path, box_size=6):
                try:
                    img = _PImage.open(qr_path); img.thumbnail((180, 180))
                    photo = ImageTk.PhotoImage(img)
                    self._gal_qr_photo_ref = photo
                    qr_row = ctk.CTkFrame(self._gal_qr_frame, fg_color="transparent")
                    qr_row.pack(fill="x", pady=5)
                    tk.Label(qr_row, image=photo, bg="#1a1a2e").pack(side="left", padx=5)
                    ctk.CTkLabel(qr_row,
                                 text=f"📱 Scannez avec votre téléphone\n\nURL : {url_for_qr}",
                                 text_color="#3498db", justify="left", font=("Roboto", 11)
                                 ).pack(side="left", padx=15)
                except Exception as e:
                    ctk.CTkLabel(self._gal_qr_frame, text=f"QR error: {e}",
                                 text_color="#666").pack()
        else:
            ctk.CTkLabel(self._gal_qr_frame,
                         text=f"💡 Pour afficher un QR scannable : pip install qrcode[pil]\n   URL : {url_for_qr}",
                         text_color="#888", justify="left").pack(anchor="w")

    def _gal_stop_clicked(self):
        from src.core.gallery_server import get_server
        get_server().stop()
        for w in self._gal_qr_frame.winfo_children():
            w.destroy()
        self._gal_btn_start.configure(state="normal")
        self._gal_btn_stop.configure(state="disabled")
        self._gal_btn_open.configure(state="disabled")
        self._gal_status_label.configure(text="État : Arrêté", text_color="#888")

    def _gal_open_clicked(self):
        import webbrowser
        from src.core.gallery_server import get_server
        srv = get_server()
        if not srv.is_running():
            messagebox.showwarning("Galerie", "Le serveur n'est pas actif.")
            return
        st = srv.status()
        # Prefer local URL for the in-app open (more reliable, no ngrok cold start)
        url = st.get("local_url")
        if not url and st.get("ngrok_url"):
            url = st["ngrok_url"]
        if url:
            webbrowser.open(url)
        else:
            messagebox.showerror("Galerie", "URL non disponible — relancez le serveur.")

    # ── TB patch actions ──
    def _tbp_refresh(self):
        from src.core.tb_image_patch import get_patch_status, find_validation_file
        root = self._tbp_path_entry.get().strip().replace("\\", "/")
        if not root:
            self._tbp_status_label.configure(text="(chemin vide)", text_color="#888")
            return

        # Heuristic: if user pointed inside experiments/, walk up to find engine root
        # Common mistakes:
        #   .../neosr/experiments/Deband/visualization
        #   .../neosr/experiments/tb_logger/Deband  ← Jimmy's case
        #   .../neosr/experiments/<run>/tb_logger
        suggested_root = None
        path_lower = root.lower()
        if "/experiments/" in path_lower or "\\experiments\\" in path_lower:
            # Walk back up: take everything before /experiments/
            idx = path_lower.find("/experiments/")
            if idx > 0:
                candidate = root[:idx]
                if os.path.isdir(candidate):
                    suggested_root = candidate

        if suggested_root and suggested_root != root:
            self._tbp_status_label.configure(
                text=f"⚠ Vous avez pointé un dossier d'expérimentation, pas la racine de l'engine.\n"
                     f"   Le patch modifie le code source du moteur (NeoSR/Redux), pas les logs.\n"
                     f"   → Suggestion : utilisez « {suggested_root} »\n"
                     f"   Cliquez sur le bouton 'NeoSR' (raccourci) ou modifiez le chemin manuellement.",
                text_color="#f39c12"
            )
            return

        if not os.path.isdir(root):
            self._tbp_status_label.configure(
                text=f"❌ Dossier introuvable : {root}\n"
                     f"   → Vérifiez votre installation NeoSR/Redux.",
                text_color="#e74c3c"
            )
            return

        status = get_patch_status(root)
        if not status["found"]:
            # Show what we tried
            tried = [
                "neosr/models/default.py",
                "neosr/models/sr_model.py",
                "traiNNer/models/sr_model.py",
                "traiNNer/models/default.py",
            ]
            # Check if engine looks installed at all
            sub_check = []
            for sub in ["neosr", "traiNNer", "basicsr"]:
                if os.path.isdir(os.path.join(root, sub)):
                    sub_check.append(f"{sub}/ ✓")
                else:
                    sub_check.append(f"{sub}/ ✗")

            self._tbp_status_label.configure(
                text=f"❌ Aucun fichier de validation trouvé dans :\n   {root}\n\n"
                     f"   Sous-dossiers détectés : {', '.join(sub_check)}\n"
                     f"   Fichiers cherchés : {', '.join(tried)}\n"
                     f"   Recherche : tout fichier model*.py contenant 'nondist_validation' + 'imwrite'.\n\n"
                     f"   💡 Le patch modifie le code source de l'engine pour logger les images de validation\n"
                     f"      dans TensorBoard. Pointez sur la racine de NeoSR ou traiNNer-Redux.",
                text_color="#e74c3c"
            )
        elif status["patched"]:
            self._tbp_status_label.configure(
                text=f"✅ Patch déjà appliqué\n"
                     f"   Fichier : {status['target_file']}\n"
                     f"   Backup .usr_bak : {'présent' if status['backup_exists'] else 'absent'}",
                text_color="#2ecc71"
            )
        else:
            self._tbp_status_label.configure(
                text=f"⚪ Pas patché (prêt à l'emploi)\n"
                     f"   Fichier cible : {status['target_file']}\n"
                     f"   ➜ Cliquez sur '✅ Appliquer le Patch' pour activer les images TensorBoard.",
                text_color="#f39c12"
            )

    def _tbp_apply_clicked(self):
        from src.core.tb_image_patch import patch_engine
        root = self._tbp_path_entry.get().strip()
        if not os.path.isdir(root):
            messagebox.showerror("Erreur", f"Dossier introuvable : {root}")
            return
        if not messagebox.askyesno("Patch",
                                    f"Modifier les fichiers dans :\n{root}\n\n"
                                    f"Un backup .usr_bak sera créé. Réversible."):
            return
        ok, msg, path = patch_engine(root)
        if ok:
            messagebox.showinfo("Patch", f"{msg}\n\nFichier : {path}")
        else:
            messagebox.showerror("Erreur", msg)
        self._tbp_refresh()

    def _tbp_remove_clicked(self):
        from src.core.tb_image_patch import find_validation_file, unpatch_file
        root = self._tbp_path_entry.get().strip()
        target = find_validation_file(root)
        if not target:
            messagebox.showerror("Erreur", "Fichier cible introuvable.")
            return
        if not messagebox.askyesno("Retirer", f"Restaurer le fichier original ?\n{target}"):
            return
        ok, msg = unpatch_file(target)
        if ok:
            messagebox.showinfo("Patch", msg)
        else:
            messagebox.showerror("Erreur", msg)
        self._tbp_refresh()

    def setup_about_tab(self):
        import tkinter as tk
        f = self.tab_about
        # Play about.WAV whenever this tab becomes visible
        f.bind("<Map>", lambda e: self._play_about_sound())

        # Use a tk.Canvas as the base - this guarantees bg image works
        # We need to get the internal tkinter widget from CTkFrame
        inner = f
        # CTkFrame wraps a tk widget - access it for raw tk ops
        try:
            inner_tk = f.winfo_children()[0] if f.winfo_children() else f
        except Exception:
            inner_tk = f

        canvas = tk.Canvas(f, highlightthickness=0, bg="#1a1a2e")
        canvas.pack(fill="both", expand=True)

        self._about_pil_bg = None
        self._about_tk_bg = None

        try:
            from PIL import Image, ImageEnhance, ImageTk
            import sys
            # Try multiple paths for the bg image
            search_paths = [
                os.path.join(os.getcwd(), "assets"),
                os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "assets"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets"),
            ]
            for base in search_paths:
                for ext in ("about_bg.jpg", "about_bg.png"):
                    bg_path = os.path.join(base, ext)
                    if os.path.exists(bg_path):
                        pil_img = Image.open(bg_path).convert("RGB")
                        self._about_pil_bg = ImageEnhance.Brightness(pil_img).enhance(0.3)
                        break
                if self._about_pil_bg:
                    break
        except Exception:
            pass

        def _on_canvas_resize(event=None):
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 10 or h < 10:
                return
            # Draw background
            if self._about_pil_bg:
                try:
                    from PIL import Image, ImageTk as ITk
                    resized = self._about_pil_bg.resize((w, h), Image.LANCZOS)
                    self._about_tk_bg = ITk.PhotoImage(resized)
                    canvas.delete("bg")
                    canvas.create_image(0, 0, image=self._about_tk_bg, anchor="nw", tags="bg")
                    canvas.tag_lower("bg")
                except Exception:
                    pass
            # Resize content window
            canvas.itemconfig("content", width=w)

        canvas.bind("<Configure>", _on_canvas_resize)

        # Content frame on canvas - must be a tk.Frame (not CTk) for true transparency
        content_tk = tk.Frame(canvas, bg="#1a1a2e")
        canvas.create_window(0, 0, window=content_tk, anchor="nw", tags="content")

        # Build content using tk widgets for bg compatibility
        try:
            from PIL import Image, ImageTk as ITk
            # Icon
            for base_p in search_paths if 'search_paths' in dir() else [os.path.join(os.getcwd(), "assets")]:
                for ext in ("icon.png", "icon.ico"):
                    icon_path = os.path.join(base_p, ext)
                    if os.path.exists(icon_path):
                        icon_pil = Image.open(icon_path).resize((80, 80), Image.LANCZOS)
                        self._about_icon_ref = ITk.PhotoImage(icon_pil)
                        tk.Label(content_tk, image=self._about_icon_ref, bg="#1a1a2e").pack(pady=(20, 5))
                        break
                if hasattr(self, '_about_icon_ref'):
                    break
        except Exception:
            pass

        tk.Label(content_tk, text="Universal SR Studio", font=("Roboto", 24, "bold"),
                 fg="#3498db", bg="#1a1a2e").pack(pady=(5, 2))
        tk.Label(content_tk, text="v2.2.0 -- Super-Resolution Training Suite",
                 fg="#AAAAAA", bg="#1a1a2e", font=("Roboto", 12)).pack()

        sep = tk.Frame(content_tk, height=2, bg="#3498db")
        sep.pack(fill="x", pady=15, padx=60)

        info_text = (
            "Application de configuration et d'entrainement de modeles\n"
            "de Super-Resolution (NeoSR & TraiNNer-Redux).\n\n"
            "Fonctionnalites : wizard guide, configuration avancee,\n"
            "monitoring GPU temps reel, outils de conversion et comparaison."
        )
        tk.Label(content_tk, text=info_text, fg="#CCCCCC", bg="#1a1a2e",
                 justify="center", font=("Roboto", 11)).pack(pady=5)

        # Credits
        credits_frame = tk.Frame(content_tk, bg="#111128", bd=1, relief="solid",
                                  highlightbackground="#333355", highlightthickness=1)
        credits_frame.pack(fill="x", pady=10, padx=40)

        tk.Label(credits_frame, text="Auteurs & Credits", font=("Roboto", 14, "bold"),
                 fg="#e74c3c", bg="#111128").pack(pady=(10, 5))

        for name, role in [
            ("CrysisJim", "Concept, Design, Direction, Testing"),
            ("Gemini 3.1 Pro", "Google -- Base du code, Architecture initiale"),
            ("Grok 4.3", "xAI -- Base du code, Contributions initiales"),
            ("Claude Opus 4.7", "Anthropic -- Architecture, Code principal, Refactoring"),
        ]:
            row = tk.Frame(credits_frame, bg="#111128")
            row.pack(fill="x", padx=15, pady=2)
            tk.Label(row, text=name, font=("Roboto", 12, "bold"),
                     fg="#FFFFFF", bg="#111128", anchor="w", width=18).pack(side="left")
            tk.Label(row, text=role, fg="#999999", bg="#111128", anchor="w").pack(side="left", padx=10)
        tk.Label(credits_frame, text="", bg="#111128", height=0).pack(pady=(0, 5))

        links_frame = tk.Frame(content_tk, bg="#1a1a2e")
        links_frame.pack(pady=10)
        tk.Label(links_frame, text="Moteurs SR :", fg="#888888", bg="#1a1a2e").pack(side="left")
        lbl_neo = tk.Label(links_frame, text="NeoSR", fg="#3498db", bg="#1a1a2e",
                 cursor="hand2", font=("Roboto", 12, "underline"))
        lbl_neo.pack(side="left", padx=10)
        lbl_neo.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/neosr-project/neosr"))
        lbl_trx = tk.Label(links_frame, text="TraiNNer-Redux", fg="#e67e22", bg="#1a1a2e",
                 cursor="hand2", font=("Roboto", 12, "underline"))
        lbl_trx.pack(side="left", padx=10)
        lbl_trx.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/the-database/traiNNer-redux"))

        tk.Label(content_tk, text="2025-2026 -- Licence MIT",
                 fg="#666666", bg="#1a1a2e", font=("Roboto", 10)).pack(pady=(15, 5))

