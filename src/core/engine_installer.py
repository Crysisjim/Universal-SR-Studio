"""Installeur de moteur UI-agnostique (port du flux CTk vers un module partagé).

But : un nouveau user (EXE portable) doit pouvoir, depuis le GUI, installer
d'un coup python portable + venv partagé + dépendances + PyTorch CUDA assorti
au GPU + le moteur choisi — exactement comme le faisait la version CustomTkinter.

Ce module NE LANCE RIEN lui-même : chaque fonction écrit le script d'install
piloté (.py) et renvoie la commande `(programme, [args], cwd)` à exécuter dans
le runner console de l'UI (CTk `launch_console`, QT `_ConsoleDialog`). Ainsi CTk
et QT partagent la MÊME logique d'install.

Logique identique au CTk historique (src/ui/tabs/tab_settings.py) :
  - venv UNIQUE partagé `runtimes/.venv` (les deux moteurs s'y installent).
  - 1er moteur installé → tout (deps + torch CUDA + clone). 2e moteur → si le
    venv a déjà le bon torch CUDA, on skip le re-download (juste clone + deps).
  - torch CUDA réinstallé EN DERNIER (les deps tirent parfois un torch CPU).
  - version torch + wheel CUDA choisies par `get_pytorch_recommendation()`
    (cu128 Blackwell / cu126 Maxwell→Hopper / cu118 ancien) → le triton lazy
    s'aligne ensuite dessus (runner._ensure_triton_if_needed).
"""
from __future__ import annotations

import os
import shutil

from src.core import engine_paths as ep
from src.core.engine_constraints import CONSTRAINTS as _CONSTRAINTS

# Liste de dépendances = UNION traiNNer-redux (dev) + neosr. numpy>=2 requis par
# les DEUX moteurs depuis le refactor venv partagé. torch/torchvision installés
# à part (build CUDA). opencv-python (full) couvre les symboles de neosr.
_REQS = (
    "numpy>=2\nopencv-python\nscipy\npyyaml\ntoml\ntqdm\ntensorboard\n"
    "rich\nPyWavelets\npillow\neinops\nlmdb\nema-pytorch\nsafetensors\n"
    "spandrel\nspandrel-extra-arches\npyvips\npyvips-binary\nrequests\n"
    "timm\nmsgspec\nantialiased_cnns\npytorch-optimizer\nonnxruntime-gpu<1.27\nonnx\n"
    "albumentations\nimageio-ffmpeg\n"
)

# Fallback si get_pytorch_recommendation échoue (pas de GPU détecté p.ex.).
# onnxruntime-gpu >= 1.27 is built for CUDA 13 (PyPI default since 1.27): it neither
# matches PyTorch's CUDA 12 runtime nor supports Pascal GPUs, so ONNX inference silently
# fell back to the CPU. 1.26.x is the last CUDA-12 build.
ORT_GPU_SPEC = "onnxruntime-gpu<1.27"

_TORCH_FALLBACK = {
    "install_pkgs": ["torch==2.7.0", "torchvision==0.22.0"],
    "whl_url": "https://download.pytorch.org/whl/cu126",
    "torch_version": "2.7.0", "cuda_tag": "cu126",
}


def _torch_reco() -> dict:
    try:
        from src.core.compute_estimator import get_pytorch_recommendation
        rec = get_pytorch_recommendation()
        # garde-fous : il faut au moins install_pkgs[0,1] + whl_url
        if rec and rec.get("install_pkgs") and rec.get("whl_url") and len(rec["install_pkgs"]) >= 2:
            return rec
    except Exception:
        pass
    return dict(_TORCH_FALLBACK)


def normalize_engine(engine_name: str) -> str:
    """→ 'TraiNNer-Redux' ou 'NeoSR'."""
    return "TraiNNer-Redux" if "redux" in (engine_name or "").lower() else "NeoSR"


def engine_repo(engine_name: str) -> tuple[str, str]:
    """(git_url, branch) pour le moteur."""
    if normalize_engine(engine_name) == "TraiNNer-Redux":
        return ep.REDUX_GIT_URL, ep.REDUX_GIT_BRANCH
    return ep.NEOSR_GIT_URL, ep.NEOSR_GIT_BRANCH


def engine_target(engine_name: str) -> str:
    return ep.redux_path() if normalize_engine(engine_name) == "TraiNNer-Redux" else ep.neosr_path()


# ── États (pour piloter l'UI) ────────────────────────────────────────────────

def is_portable_python_installed() -> bool:
    pp = ep.portable_python()
    return bool(pp and os.path.exists(pp))


def is_venv_ready() -> bool:
    return os.path.exists(ep.shared_venv_python())


def is_engine_installed(engine_name: str) -> bool:
    return os.path.exists(os.path.join(engine_target(engine_name), ".git"))


def ensure_workspace_folders() -> None:
    """Crée l'arborescence de travail sous base_engine_path (comme le CTk) :
    datasets, Option Custom, experiments, models, weights, datsheet, Final model.
    Idempotent. Appelé au démarrage + après une install moteur."""
    base = ep.base_engine_path()
    folders = [
        os.path.join(base, "datasets", "train", "HR"),
        os.path.join(base, "datasets", "train", "LQ"),
        os.path.join(base, "datasets", "val", "GT"),
        os.path.join(base, "datasets", "val", "LQ"),
        os.path.join(base, "experiments"),
        os.path.join(base, "Option Custom"),
        os.path.join(base, "Final model"),
        os.path.join(base, "models"),
        os.path.join(base, "weights"),
    ]
    for f in folders:
        try:
            os.makedirs(f, exist_ok=True)
        except Exception:
            pass


def bundled_python_dir() -> str | None:
    """Si l'app gelée embarque un python portable (bundle A), renvoie son dossier.
    PyInstaller onedir : les datas sont dans sys._MEIPASS (sous _internal/)."""
    import sys
    bases = []
    mp = getattr(sys, "_MEIPASS", None)
    if mp:
        bases.append(mp)
    bases.append(os.path.dirname(os.path.abspath(sys.executable)))
    for b in bases:
        cand = os.path.join(b, "runtimes", ep.PORTABLE_PY_DIRNAME)
        if os.path.exists(os.path.join(cand, "python.exe")):
            return cand
    return None


def bundled_mamba_wheel() -> str | None:
    """mamba_ssm wheel built for Windows by tools/mamba_build (selective-scan CUDA kernel
    used by MambaIR / MambaIRv2; without it they fall back to a ~100x slower pure-PyTorch
    scan). Shipped in vendor_wheels/ (exe: _internal/vendor_wheels, dev: repo root)."""
    import sys
    bases = [getattr(sys, "_MEIPASS", None), os.path.dirname(os.path.abspath(sys.executable)),
             os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]
    for b in bases:
        d = os.path.join(b, "vendor_wheels") if b else ""
        if d and os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith("mamba_ssm-") and f.endswith(".whl"):
                    return os.path.join(d, f)
    return None


def _mamba_install_code(vpy_var: str, indent: str = "") -> str:
    """Script fragment: install the bundled mamba_ssm wheel into the venv. Built against
    torch 2.7.0+cu126 / CPython 3.12 — skipped (with a log line) for any other torch.
    Never fatal: MambaIR still runs on its pure-PyTorch scan without it."""
    import textwrap
    whl = bundled_mamba_wheel() or ""
    return textwrap.indent(f"""
_MAMBA_WHL = r'{whl}'
if _MAMBA_WHL and os.path.exists(_MAMBA_WHL):
    try:
        _tv = subprocess.run([{vpy_var}, "-c", "import torch, sys; print(torch.__version__, sys.version_info[:2] == (3, 12))"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        _tv = ""
    if _tv.startswith("2.7.0+cu126") and _tv.endswith("True"):
        print(">> Install mamba_ssm (kernel CUDA MambaIR, build Windows embarque)...", flush=True)
        if subprocess.call([{vpy_var}, "-m", "pip", "install", "--no-deps", _MAMBA_WHL]) != 0:
            print(">> [WARN] mamba_ssm non installe : MambaIR utilisera le scan pur PyTorch (lent).", flush=True)
    else:
        print(">> mamba_ssm embarque prevu pour torch 2.7.0+cu126 (venv : " + (_tv or "?") + ") : ignore, MambaIR utilisera le scan pur PyTorch.", flush=True)
""", indent)


def ensure_portable_python_bundled(log=lambda s: None) -> bool:
    """Bundle A : copie le python portable embarqué → base_engine_path/runtimes
    s'il n'est pas déjà installé. Renvoie True si un python portable est dispo
    après coup (déjà installé OU copié). False si rien à copier (mode dev → download)."""
    if is_portable_python_installed():
        return True
    src = bundled_python_dir()
    if not src:
        return False
    dst = os.path.join(ep.runtimes_path(), ep.PORTABLE_PY_DIRNAME)
    try:
        os.makedirs(ep.runtimes_path(), exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        log(f">> Python portable embarqué copié -> {dst}\n")
    except Exception as e:
        log(f">> Copie du python embarqué échouée : {e}\n")
    return is_portable_python_installed()


def system_python() -> str | None:
    """Un python système pour amorcer l'install du python portable (le 1er pas).
    Cherche `py` puis `python` dans le PATH. None si rien (→ message UI)."""
    for cand in ("py", "python", "python3"):
        p = shutil.which(cand)
        if p:
            return p
    return None


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ── 1) Python portable ───────────────────────────────────────────────────────

def build_portable_python_script() -> tuple[list[str], str] | None:
    """Écrit install_runtime.py (télécharge python embeddable + pip + virtualenv).
    Renvoie (cmd, cwd) ou None si aucun python système pour amorcer."""
    sys_py = system_python()
    if not sys_py:
        return None
    base = ep.base_engine_path()
    runtimes = ep.runtimes_path()
    target_dir = os.path.join(runtimes, ep.PORTABLE_PY_DIRNAME)
    script_path = os.path.join(base, "install_runtime.py")
    content = f"""
import urllib.request, zipfile, os, sys, ssl
try: ssl._create_default_https_context = ssl._create_unverified_context
except Exception: pass
def log(msg): print(f'>> {{msg}}', flush=True)
target_dir = r'{target_dir}'
runtimes_path = r'{runtimes}'
zip_path = os.path.join(runtimes_path, "python312.zip")
if os.path.exists(target_dir):
    import shutil
    try: shutil.rmtree(target_dir, ignore_errors=True)
    except Exception: pass
os.makedirs(runtimes_path, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)
url = 'https://www.python.org/ftp/python/{ep.PYTHON_PATCH}/python-{ep.PYTHON_PATCH}-embed-amd64.zip'
log('Download Python {ep.PYTHON_PATCH}...')
try: urllib.request.urlretrieve(url, zip_path)
except Exception: sys.exit(1)
log('Extraction...')
with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(target_dir)
os.remove(zip_path)
pth = os.path.join(target_dir, "python312._pth")
if os.path.exists(pth):
    with open(pth, 'r') as f: c = f.read()
    c = c.replace('#import site', 'import site')
    with open(pth, 'w') as f: f.write(c)
gp = os.path.join(target_dir, "get-pip.py")
try: urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', gp)
except Exception: pass
py = os.path.join(target_dir, "python.exe")
log('Install PIP + Virtualenv...')
# list form (not os.system): user folders with spaces (e.g. "Jean Dupont") must work
import subprocess as _sp
_rc1 = _sp.call([py, gp])
_rc2 = _sp.call([py, "-m", "pip", "install", "virtualenv", "--no-warn-script-location"])
if _rc1 != 0 or _rc2 != 0:
    log('[ERREUR] pip / virtualenv non installe (codes ' + str(_rc1) + ', ' + str(_rc2) + ')')
    import sys as _sys; _sys.exit(1)
log('[OK] TERMINE !')
try: import os as _os; _os.remove(__file__)
except Exception: pass
"""
    _write(script_path, content)
    return [sys_py, script_path], base


# ── 2) Install moteur complet (venv + deps + torch CUDA + clone) ─────────────

def build_engine_install_script(engine_name: str) -> tuple[list[str], str] | None:
    """Écrit install_<engine>.py (clone + venv partagé + deps + torch CUDA,
    skip torch si déjà bon). Renvoie (cmd, cwd) ou None si python portable absent."""
    portable_py = ep.portable_python()
    if not (portable_py and os.path.exists(portable_py)):
        return None
    eng = normalize_engine(engine_name)
    git_url, git_branch = engine_repo(eng)
    git_pin = ep.REDUX_GIT_PIN if eng == "TraiNNer-Redux" else ep.NEOSR_GIT_PIN
    target_path = engine_target(eng)
    base = ep.base_engine_path()
    shared_venv = ep.shared_venv_dir()
    rec = _torch_reco()
    torch_pkg = rec["install_pkgs"][0]
    tv_pkg = rec["install_pkgs"][1]
    whl_url = rec["whl_url"]
    cuda_tag = rec.get("cuda_tag", "cu126")
    want_ver = torch_pkg.split("==")[1] if "==" in torch_pkg else ""
    cuda_label = f"{rec.get('torch_version', '?')} CUDA {cuda_tag}"
    script_path = os.path.join(base, f"install_{eng}.py")
    content = f"""
import os, sys, subprocess
def log(msg): print(f'>> {{msg}}', flush=True)
TARGET = r'{target_path}'
GIT_URL = '{git_url}'
GIT_BRANCH = '{git_branch}'
GIT_PIN = '{git_pin}'
PORTABLE_PY = r'{portable_py}'
REQ_FILE = os.path.join(TARGET, "requirements.txt")
# Exact versions of the tested release (engine_constraints.py): "-c" on every deps install.
CONSTRAINTS_FILE = os.path.join(TARGET, "usrs_constraints.txt")
VENV_DIR = r'{shared_venv}'
os.makedirs(os.path.dirname(VENV_DIR), exist_ok=True)
os.makedirs(TARGET, exist_ok=True)
def _force_rmtree(path):
    # Plain shutil.rmtree(ignore_errors=True) silently leaves files behind on
    # Windows when git has marked its .git/objects/** read-only — the directory
    # LOOKS gone (no exception raised) but isn't, so the clone below still hits
    # "destination path '.' already exists and is not an empty directory."
    # Clearing the read-only bit before each removal actually deletes it.
    import shutil, stat
    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass
    shutil.rmtree(path, onerror=_onerror)

import shutil as _sh
_git_head = os.path.join(TARGET, ".git", "HEAD")
_has_git = _sh.which("git") is not None
# User data the engine folder may hold (trained models, datasets…): never wiped.
_PROTECTED = ("datasets", "experiments", "datsheet", "models", "weights", "tb_logger", "results")
_has_user_data = any(e.lower() in _PROTECTED for e in os.listdir(TARGET))

def _download_source():
    # No git (most Windows users) or a folder that must not be emptied: fetch the
    # source archive of the exact commit (or branch head) and extract it over TARGET.
    # Files are overwritten, nothing else is deleted.
    import urllib.request, zipfile, tempfile, ssl
    try: ssl._create_default_https_context = ssl._create_unverified_context
    except Exception: pass
    _repo = GIT_URL.rstrip("/").removesuffix(".git").split("github.com/")[1]
    _ref = GIT_PIN or ("refs/heads/" + GIT_BRANCH)
    _url = "https://codeload.github.com/" + _repo + "/zip/" + _ref
    log("Telechargement du code source (" + (GIT_PIN[:8] if GIT_PIN else GIT_BRANCH) + ") sans git...")
    _zp = os.path.join(tempfile.gettempdir(), "usrs_engine_src.zip")
    for _att in range(1, 4):
        try:
            urllib.request.urlretrieve(_url, _zp)
            break
        except Exception as _e:
            log("  tentative " + str(_att) + "/3 echouee : " + str(_e))
    else:
        return False
    with zipfile.ZipFile(_zp) as _z:
        _top = _z.namelist()[0].split("/")[0] + "/"
        for _m in _z.infolist():
            _rel = _m.filename[len(_top):]
            if not _rel:
                continue
            _dst = os.path.join(TARGET, *_rel.split("/"))
            if _m.is_dir():
                os.makedirs(_dst, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(_dst), exist_ok=True)
                with _z.open(_m) as _src, open(_dst, "wb") as _out:
                    _out.write(_src.read())
    os.remove(_zp)
    return os.path.exists(os.path.join(TARGET, "train.py"))

_clone_ok = True
_used_git = False
if os.path.exists(_git_head) and not _has_git:
    log("Depot git present mais git introuvable : mise a jour par telechargement.")
    _clone_ok = _download_source()
elif not os.path.exists(_git_head) and (not _has_git or _has_user_data):
    if _has_user_data:
        log("Donnees utilisateur detectees dans le dossier moteur : rien n'est supprime.")
    _clone_ok = _download_source()
elif not os.path.exists(_git_head):
    _used_git = True
    # A ".git" folder existing isn't proof of a working clone — a previous
    # interrupted/failed clone can leave an empty or half-written ".git" with
    # no HEAD, which used to pass this check, fall into "git pull", and fail
    # with "fatal: not a git repository" forever (silently, deps still
    # installed into the shared venv, so the engine LOOKED set up while its
    # source was never actually fetched) (2026-09-17 fix). "git clone url ."
    # also refuses outright into a non-empty directory (even with no .git at
    # all), and TARGET always has at least the requirements.txt this same
    # script writes below on a first partial run — so the directory must be
    # emptied, not just its .git, before retrying the clone.
    log(f"Git Clone (branch {{GIT_BRANCH}})...")
    for _entry in os.listdir(TARGET):
        _p = os.path.join(TARGET, _entry)
        try:
            _force_rmtree(_p) if os.path.isdir(_p) else os.remove(_p)
        except Exception as _e:
            log(f"[WARN] Nettoyage {{_entry}} : {{_e}}")
    _rc = subprocess.call(['git', 'clone', '-b', GIT_BRANCH, GIT_URL, '.'], cwd=TARGET, shell=True)
    _clone_ok = (_rc == 0) and os.path.exists(_git_head)
else:
    _used_git = True
    log(f"Git Pull (branch {{GIT_BRANCH}})...")
    subprocess.call(['git', 'checkout', GIT_BRANCH], cwd=TARGET, shell=True)
    _rc = subprocess.call(['git', 'pull', 'origin', GIT_BRANCH], cwd=TARGET, shell=True)
    _clone_ok = (_rc == 0)
if _clone_ok and GIT_PIN and _used_git:
    # Tested commit of the branch (see engine_paths.REDUX_GIT_PIN).
    log(f"Git checkout {{GIT_PIN[:8]}} (version testée)...")
    _clone_ok = subprocess.call(['git', 'checkout', GIT_PIN], cwd=TARGET, shell=True) == 0
if not _clone_ok:
    # Never seen before this fix: a failed clone/pull used to fall straight
    # through into pip installs and still print "[OK] TERMINE !" at the end,
    # because only torch's presence was checked, never the engine source
    # itself — so the UI showed a fully green install after a run that fetched
    # no source at all (2026-09-17 fix, found from a real failed NeoSR install
    # log: "fatal: not a git repository", exit code 0, "[OK] TERMINE !").
    log("[ERREUR] Le clone/pull Git a échoué — le code source du moteur n'a pas été récupéré.")
    log("         Verifie ta connexion internet (ou proxy/antivirus), puis relance l installation.")
    sys.exit(1)
if not os.path.exists(REQ_FILE):
    with open(REQ_FILE, "w", encoding="utf-8") as f: f.write('''{_REQS}''')
with open(CONSTRAINTS_FILE, "w", encoding="utf-8") as f: f.write({_CONSTRAINTS!r})
VPY = os.path.join(VENV_DIR, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(VENV_DIR, "bin", "python")
if not os.path.exists(VENV_DIR):
    log("Creation VENV partage (runtimes/.venv)...")
    # --app-data : dossier de cache dédié dans runtimes/ → évite l'avertissement
    # "could not migrate app data ... virtualenv" (migration du cache global).
    APPDATA = os.path.join(os.path.dirname(VENV_DIR), "_virtualenv_appdata")
    os.makedirs(APPDATA, exist_ok=True)
    # VIRTUALENV_OVERRIDE_APP_DATA : empeche virtualenv de tenter la migration du
    # cache GLOBAL (~/AppData/Local/pypa/virtualenv) -> supprime l'avertissement
    # "could not migrate app data ...". --app-data en plus pour etre sur.
    _venv_env = dict(os.environ); _venv_env["VIRTUALENV_OVERRIDE_APP_DATA"] = APPDATA
    subprocess.call([PORTABLE_PY, "-m", "virtualenv", "--app-data", APPDATA, VENV_DIR], cwd=TARGET, env=_venv_env)
log("Update PIP...")
# python -m pip (pas pip.exe : Windows refuse que pip se mette a jour lui-meme).
subprocess.call([VPY, "-m", "pip", "install", "--upgrade", "pip", "wheel"], cwd=TARGET)

def _torch_ver():
    try:
        _r = subprocess.run([VPY, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True, timeout=20)
        return _r.stdout.strip() if _r.returncode == 0 else ""
    except Exception:
        return ""

_want_ver = "{want_ver}"
_cuda_tag = "{cuda_tag}"

def _torch_is_ok():
    _it = _torch_ver()
    return bool(_want_ver) and _cuda_tag in _it and _want_ver in _it

def _install_torch_cuda(reason=""):
    # Telechargement ~2.8 GB : sujet aux coupures reseau (SSL bad record mac sur
    # gros fichiers, souvent AV/proxy). On RETENTE plusieurs fois. On ne desinstalle
    # PAS avant (pip remplace) -> en cas d'echec, l'ancien torch reste en place.
    if reason: log(reason)
    for _att in range(1, 6):
        log("Install Torch {cuda_label} (~2.8 GB) - tentative " + str(_att) + "/5...")
        _rc = subprocess.call([VPY, "-m", "pip", "install", "{torch_pkg}", "{tv_pkg}",
                               "--index-url", "{whl_url}", "--no-cache-dir",
                               "--retries", "5", "--timeout", "60"], cwd=TARGET)
        if _rc == 0 and _torch_is_ok():
            log("Torch " + _torch_ver() + " installe OK.")
            return True
        log("Echec/coupure reseau pendant le telechargement torch - nouvelle tentative...")
    log("[ERREUR] Torch CUDA n'a pas pu etre installe apres 5 tentatives (reseau).")
    log("         Verifie ta connexion / antivirus / proxy, puis relance via le bouton")
    log("         'Installer / MAJ PyTorch (GPU)'. L'entrainement ne marchera pas sans torch.")
    return False

# Torch CUDA EN PREMIER : evite que les deps tirent un torch recent (~123 MB) jete ensuite.
if _torch_is_ok():
    log("Torch " + _torch_ver() + " CUDA deja installe - skip re-download.")
else:
    _install_torch_cuda("Install Torch EN PREMIER (evite le double download)...")

log("Install Deps (torch CUDA deja present -> pas de re-telechargement)...")
# F25: every step's return code counts; a partial install no longer exits 0
_fails = []
if subprocess.call([VPY, "-m", "pip", "install", "-r", "requirements.txt", "-c", CONSTRAINTS_FILE], cwd=TARGET) != 0:
    _fails.append("requirements.txt")
if subprocess.call([VPY, "-m", "pip", "install", "tensorboard", "rich", "spandrel", "spandrel-extra-arches", "safetensors", "pyvips", "pyvips-binary", "requests", "timm", "antialiased_cnns", "pytorch-optimizer", "msgspec", "onnxruntime-gpu<1.27", "onnx", "imageio-ffmpeg", "-c", CONSTRAINTS_FILE], cwd=TARGET) != 0:
    _fails.append("deps app")

# Securite : si une dep a quand meme remplace torch hors cu126, on reasserte (avec retry).
if not _torch_is_ok():
    _install_torch_cuda("Une dep a modifie torch -> reassertion CUDA...")

log("Migration pynvml -> nvidia-ml-py (supprime FutureWarning PyTorch)...")
subprocess.call([VPY, "-m", "pip", "uninstall", "pynvml", "-y"], cwd=TARGET)
subprocess.call([VPY, "-m", "pip", "install", "nvidia-ml-py", "-c", CONSTRAINTS_FILE], cwd=TARGET)
if not _torch_is_ok():
    _fails.append("torch CUDA")
{_mamba_install_code("VPY")}
try: import os as _os; _os.remove(__file__)
except Exception: pass
if _fails:
    log("[ERREUR] Installation incomplete : " + ", ".join(_fails) + " - relance l'installation (ou 'Installer/MAJ PyTorch (GPU)' pour torch).")
    sys.exit(1)
log("[OK] TERMINE !")
"""
    _write(script_path, content)
    # The console runs the script with the engine folder as working directory: on a
    # fresh install it does not exist yet, and the process would not even start
    # (WinError 267 / QProcess "working directory does not exist").
    os.makedirs(target_path, exist_ok=True)
    return [portable_py, script_path], target_path


# ── 3) Réinstaller / compléter les dépendances (venv déjà là) ────────────────

def build_deps_only_script() -> tuple[list[str], str] | None:
    """Écrit update_deps_temp.py (deps manquantes + réassertion torch CUDA).
    Renvoie (cmd, cwd) ou None si venv absent."""
    venv_py = ep.shared_venv_python()
    if not os.path.exists(venv_py):
        return None
    base = ep.base_engine_path()
    rec = _torch_reco()
    torch_pkg, tv_pkg, whl_url = rec["install_pkgs"][0], rec["install_pkgs"][1], rec["whl_url"]
    pkgs = ["numpy>=2", "opencv-python", "scipy", "pyyaml", "toml", "tqdm",
            "tensorboard", "rich", "PyWavelets", "pillow", "einops", "lmdb", "msgspec",
            "ema-pytorch", "torchvision", "safetensors", "spandrel", "spandrel-extra-arches",
            "pyvips", "pyvips-binary", "requests", "timm", "antialiased_cnns", "pytorch-optimizer",
            "onnxruntime-gpu<1.27", "onnx", "albumentations", "imageio-ffmpeg"]
    script_path = os.path.join(base, "update_deps_temp.py")
    content = f"""
import subprocess, sys, os
py_exe = r'{venv_py}'
pkgs = {pkgs!r}
print(">> Environnement :", py_exe, flush=True)
_cf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usrs_constraints.txt")
with open(_cf, "w", encoding="utf-8") as f: f.write({_CONSTRAINTS!r})
cmd = [py_exe, "-m", "pip", "install"] + pkgs + ["--no-warn-script-location", "-c", _cf]
try:
    subprocess.check_call([py_exe, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call(cmd)
    _has_cuda = False
    try:
        _r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=8)
        _has_cuda = _r.returncode == 0
    except Exception:
        pass
    if _has_cuda:
        print(">> Reassertion PyTorch CUDA (override torch CPU eventuel)...", flush=True)
        for _att in range(1, 6):
            print(">> Torch tentative " + str(_att) + "/5...", flush=True)
            _rc = subprocess.call([py_exe, "-m", "pip", "install", "{torch_pkg}", "{tv_pkg}",
                                   "--index-url", "{whl_url}", "--no-cache-dir",
                                   "--retries", "5", "--timeout", "60"])
            if _rc == 0:
                break
            print(">> Coupure reseau - nouvelle tentative...", flush=True)
{_mamba_install_code("py_exe", "    ")}
    print(">> [SUCCES] Toutes les dependances sont installees.", flush=True)
except Exception as e:
    print(">> [ERREUR]", e, flush=True); sys.exit(1)
try: import os as _os; _os.remove(__file__)
except Exception: pass
"""
    _write(script_path, content)
    return [venv_py, script_path], base


# ── 4) (Ré)installer UNIQUEMENT le PyTorch CUDA assorti au GPU ────────────────

def build_torch_install_script() -> tuple[list[str], str] | None:
    """Écrit install_torch_temp.py (uninstall torch + install build CUDA GPU-matched).
    Pour le bouton « Installer / MAJ PyTorch (GPU) ». None si venv absent."""
    venv_py = ep.shared_venv_python()
    if not os.path.exists(venv_py):
        return None
    base = ep.base_engine_path()
    rec = _torch_reco()
    torch_pkg, tv_pkg, whl_url = rec["install_pkgs"][0], rec["install_pkgs"][1], rec["whl_url"]
    label = f"{rec.get('torch_version', '?')} {rec.get('cuda_tag', '?')}"
    script_path = os.path.join(base, "install_torch_temp.py")
    want_ver = torch_pkg.split("==")[1] if "==" in torch_pkg else ""
    cuda_tag = rec.get("cuda_tag", "cu126")
    content = f"""
import subprocess, sys
py_exe = r'{venv_py}'
_want_ver = "{want_ver}"
_cuda_tag = "{cuda_tag}"
def _torch_ok():
    try:
        _r = subprocess.run([py_exe, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True, timeout=20)
        _it = _r.stdout.strip() if _r.returncode == 0 else ""
        return bool(_want_ver) and _cuda_tag in _it and _want_ver in _it
    except Exception:
        return False
print(">> (Re)installation PyTorch {label} GPU-matched...", flush=True)
# Pas de uninstall prealable (pip remplace) + RETRY reseau (download ~2.8 GB,
# sujet aux coupures SSL sur gros fichiers via AV/proxy).
_ok = False
for _att in range(1, 6):
    print(">> Tentative " + str(_att) + "/5...", flush=True)
    _rc = subprocess.call([py_exe, "-m", "pip", "install", "{torch_pkg}", "{tv_pkg}",
                           "--index-url", "{whl_url}", "--no-cache-dir",
                           "--retries", "5", "--timeout", "60"])
    if _rc == 0 and _torch_ok():
        _ok = True; break
    print(">> Coupure reseau pendant le telechargement - nouvelle tentative...", flush=True)
if _ok:
    print(">> [SUCCES] PyTorch CUDA installe.", flush=True)
else:
    print(">> [ERREUR] Echec apres 5 tentatives (reseau/AV/proxy). Reessaie plus tard.", flush=True)
    sys.exit(1)
try: import os as _os; _os.remove(__file__)
except Exception: pass
"""
    _write(script_path, content)
    return [venv_py, script_path], base
