# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — Universal SR Studio (PORT QT / PySide6) — build de TEST
# Build : pyinstaller universal_sr_studio_qt.spec
# Sortie: dist/Universal_SR_Studio_QT/Universal_SR_Studio_QT.exe
#
# --onedir. Bundle A : embarque le python 3.12.9 portable (runtimes/), copié au
# 1er install vers ~/USRS/runtimes/ par engine_installer.
# torch (~2.8 GB) PAS embarqué → téléchargé au 1er install.

import os
import sys
from PyInstaller.utils.hooks import collect_submodules

# collect_submodules('qt_ui') must see the project folder. When the build is
# started via pyinstaller.exe (not "python -m PyInstaller") it is not on
# sys.path: qt_ui is then "not a package", every tab is silently left out and
# the exe crashes at start ("No module named 'qt_ui.tabs'").
_HERE = os.path.dirname(os.path.abspath(SPEC))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Python 3.12.9 portable à embarquer (bundle A) — cherche dans ~/USRS.
_BUNDLED_PY = ''
for _cand in (os.path.join(os.path.expanduser('~'), _root, 'runtimes', 'python-3.12.9')
              for _root in ('USRS', 'USRS-EXPERIMENTAL')):
    if os.path.isfile(os.path.join(_cand, 'python.exe')):
        _BUNDLED_PY = _cand
        break

_QT_UI_MODULES = collect_submodules('qt_ui')
assert 'qt_ui.tabs.config_tab' in _QT_UI_MODULES, 'qt_ui not collected: tabs would be missing from the exe'

block_cipher = None

_datas = [('assets', 'assets')]
# Everything the engine venv runs or imports must stay a LOOSE .py on disk (the venv
# can't read the exe's PYZ): runners, workers, arch_infer, quick_upscale (video
# pipeline), guided_color_fix, model_convert, engine_paths… Listing the folder instead
# of naming files: a new runner/helper can no longer be forgotten in the build.
for _f in sorted(os.listdir(os.path.join(_HERE, 'src', 'core'))):
    if _f.endswith('.py'):
        _datas.append((os.path.join('src', 'core', _f), 'src/core/'))
# Whole folders copied into the engines (custom archs/losses/utils) or loaded loose.
# File by file, never __pycache__: stale bytecode (e.g. of an arch held back from public
# builds) must not ship.
for _d in ('custom_archs', 'custom_engine', 'custom_neosr_archs', 'custom_utils', 'models'):
    for _root, _dirs, _files in os.walk(os.path.join(_HERE, 'src', 'core', _d)):
        _dirs[:] = [x for x in _dirs if x != '__pycache__']
        _rel = os.path.relpath(_root, _HERE).replace(os.sep, '/')
        for _f in _files:
            if not _f.endswith(('.pyc', '.pyo')):
                _datas.append((os.path.join(_rel, _f), _rel))
# Prebuilt Windows wheels installed into the engine venv (mamba_ssm CUDA kernel for
# MambaIR, built by tools/mamba_build) — vendor_wheels/ is not in git (size).
_VW = os.path.join(_HERE, 'vendor_wheels')
if os.path.isdir(_VW):
    for _f in sorted(os.listdir(_VW)):
        if _f.endswith('.whl'):
            _datas.append((os.path.join('vendor_wheels', _f), 'vendor_wheels'))
# Bundle A : python portable (si présent au moment du build).
# File by file, WITHOUT __pycache__/*.pyc and Scripts/: those are generated on the build
# machine and embed its absolute paths (e.g. C:\Users\<dev>\USRS\runtimes\...) — the
# Scripts/*.exe launchers would even point at the dev's python. The app only runs
# "python -m pip / -m virtualenv", and .pyc files are rebuilt on the user's machine.
if os.path.isdir(_BUNDLED_PY):
    for _root, _dirs, _files in os.walk(_BUNDLED_PY):
        _dirs[:] = [x for x in _dirs if x != '__pycache__'
                    and not (_root == _BUNDLED_PY and x == 'Scripts')]
        _rel = os.path.relpath(_root, _BUNDLED_PY).replace(os.sep, '/')
        _dst = 'runtimes/python-3.12.9' + ('' if _rel == '.' else '/' + _rel)
        for _f in _files:
            if not _f.endswith(('.pyc', '.pyo')):
                _datas.append((os.path.join(_root, _f), _dst))

a = Analysis(
    ['main_qt.py'],
    pathex=['.'],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        # UI QT (découverte dynamique des tabs/pages)
        *_QT_UI_MODULES,
        # Backend partagé côté GUI (les runners venv restent loose, pas ici)
        'src.core.settings',
        'src.core.config_handler',
        'src.core.config_importer',
        'src.core.config_templates',
        'src.core.runner',
        'src.core.engine_installer',
        'src.core.engine_paths',
        'src.core.otf_preview',
        'src.core.otf_custom_degradations',
        'src.core.compute_estimator',
        'src.core.training_history',
        'src.core.descriptions',
        'src.core.translations',
        'src.core.ai_cache',
        'src.core.ai_models_metadata',
        'src.core.dataset_tools',
        'src.core.quick_upscale',
        'src.core.lq_generator',
        'pepedpid',  # DPID downscale of the LQ generator (GUI side)
        'src.core.arch_infer',
        'src.core.gallery_server',
        'src.core.qr_code',
        'src.core.toast_notifications',
        'src.core.line_filters',
        'src.core.vs_aa',
        # Paquets externes
        'appdirs', 'pkg_resources', 'setuptools',
        'yaml', 'toml', 'tomllib', 'psutil', 'psutil._pswindows',
        'pynvml', 'win11toast', 'safetensors', 'safetensors.torch',
        'comtypes', 'comtypes.client', 'comtypes.server',
        'qrcode', 'qrcode.image.pil', 'lmdb',
        'PIL', 'PIL.Image', 'PIL.ImageFilter', 'PIL.ImageEnhance', 'PIL.ImageOps',
        'PIL.ImageDraw', 'PIL.ImageFont', 'PIL.ImageCms', 'PIL.ImageQt',
        'PIL.PngImagePlugin', 'PIL.JpegImagePlugin', 'PIL.BmpImagePlugin', 'PIL.WebPImagePlugin',
        'numpy', 'numpy.core', 'numpy.lib',
        'scipy', 'scipy.ndimage', 'scipy.ndimage._filters', 'scipy.ndimage._interpolation',
        'ctypes', 'ctypes.wintypes',
        'http.server', 'socketserver', 'urllib.parse', 'urllib.request',
        'hashlib', 'queue', 'threading', 'subprocess', 'webbrowser',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hooks/pyi_rth_compat312.py'],
    excludes=[
        # Moteurs d'entraînement — trop lourds, installés à part (venv)
        'torch', 'torchvision', 'torchaudio', 'tensorflow', 'jax', 'triton',
        'onnxruntime', 'onnxruntime.capi', 'pyarrow',
        # OpenCV: the app process uses pyvips / Pillow (src/core/image_io.py); cv2 lives
        # in the engine venv (runners, workers)
        'cv2',
        # Paquets lourds inutiles
        'matplotlib', 'sklearn', 'pandas', 'jupyter', 'IPython', 'notebook',
        'pytest', 'mypy', 'black', 'pylint', 'flake8',
        # CTk (port QT — plus de customtkinter ni tkinter)
        'customtkinter', 'tkinter', 'tkinter.ttk', 'tkinter.messagebox', 'tkinter.filedialog',
        'src.ui',
        # Modules PySide6 NON utilisés (app = QtCore/QtGui/QtWidgets seulement)
        'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick',
        'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtQuickWidgets', 'PySide6.QtQuick3D',
        'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets', 'PySide6.QtSpatialAudio',
        'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.Qt3DInput', 'PySide6.Qt3DAnimation',
        'PySide6.Qt3DExtras', 'PySide6.Qt3DLogic',
        'PySide6.QtCharts', 'PySide6.QtDataVisualization', 'PySide6.QtGraphs',
        'PySide6.QtSql', 'PySide6.QtTest', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets',
        'PySide6.QtDesigner', 'PySide6.QtUiTools', 'PySide6.QtBluetooth',
        'PySide6.QtPositioning', 'PySide6.QtSensors', 'PySide6.QtSerialPort',
        'PySide6.QtWebSockets', 'PySide6.QtWebChannel', 'PySide6.QtNfc',
        'PySide6.QtHelp', 'PySide6.QtRemoteObjects', 'PySide6.QtScxml', 'PySide6.QtStateMachine',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Universal_SR_Studio_QT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets\\icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Universal_SR_Studio_QT',
)
