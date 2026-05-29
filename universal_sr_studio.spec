# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — Universal SR Studio v2.5.5
# Build: pyinstaller universal_sr_studio.spec
# Output: dist/Universal_SR_Studio/Universal_SR_Studio.exe
#
# --onedir mode: fast startup, assets alongside exe.
# Zip dist/Universal_SR_Studio/ for distribution.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ── Collect external package data ─────────────────────────────────────────────
ctk_datas   = collect_data_files('customtkinter')   # CTk built-in themes & assets

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=ctk_datas + [
        ('assets', 'assets'),                                    # App assets: icons, sounds, themes
        ('src/core/persistent_upscale_worker.py', 'src/core/'), # v2.5.5: persistent batch worker
        ('src/core/universal_runner.py', 'src/core/'),          # v2.5.6: standalone subprocess runner (RCAN 1x fix) — run by external venv, must be a loose .py
    ],
    hiddenimports=[
        # ── App modules (dynamic try/except imports) ──────────────────────────
        'src.app',
        'src.core.settings',
        'src.core.config_handler',
        'src.core.config_importer',
        'src.core.config_templates',
        'src.core.runner',
        'src.core.neosr_runner',
        'src.core.neosr_general_runner',
        'src.core.universal_runner',
        'src.core.redux_inference_runner',
        'src.core.spanplus_runner',
        'src.core.otf_preview',
        'src.core.otf_custom_degradations',
        'src.core.compute_estimator',
        'src.core.training_history',
        'src.core.descriptions',
        'src.core.translations',
        'src.core.wizard_advanced',
        'src.core.ai_cache',
        'src.core.ai_models_metadata',
        'src.core.dataset_tools',
        'src.core.model_export',
        'src.core.quick_upscale',
        'src.core.validation_rotation',
        'src.core.resume_failed',
        'src.core.gallery_server',
        'src.core.qr_code',
        'src.core.toast_notifications',
        'src.core.tb_image_patch',
        'src.core.tb_launcher',
        'src.core.distributed_client',
        'src.core.benchmark_runner',
        'src.core.arch_benchmark',
        'src.core.feature_benchmark',
        'src.core.redux_arch_benchmark',
        'src.core.redux_feature_benchmark',
        'src.ui.components.tooltip',
        'src.ui.components.performance_bars',
        'src.ui.tabs.tab_wizard',
        'src.ui.tabs.tab_config',
        'src.ui.tabs.tab_run',
        'src.ui.tabs.tab_tools',
        'src.ui.tabs.tab_settings',
        'src.ui.tabs.tab_queue',
        'src.ui.tabs.tab_distributed',
        # ── External packages ─────────────────────────────────────────────────
        'appdirs',
        'pkg_resources',
        'setuptools',
        'yaml',
        'toml',
        'tomllib',              # Python 3.11+ built-in
        'psutil',
        'psutil._pswindows',
        'pynvml',
        'win11toast',
        'safetensors',
        'safetensors.torch',
        'comtypes',
        'comtypes.client',
        'comtypes.server',
        'qrcode',
        'qrcode.image.pil',
        'lmdb',
        # ── Pillow ────────────────────────────────────────────────────────────
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL.ImageFilter',
        'PIL.ImageEnhance',
        'PIL.ImageOps',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        'PIL.ImageCms',         # v2.5.5: color fix LAB method
        'PIL.PngImagePlugin',
        'PIL.JpegImagePlugin',
        'PIL.BmpImagePlugin',
        'PIL.WebPImagePlugin',
        # ── NumPy ─────────────────────────────────────────────────────────────
        'numpy',
        'numpy.core',
        'numpy.lib',
        # ── Tkinter ───────────────────────────────────────────────────────────
        'tkinter',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'tkinter.ttk',
        # ── ctypes (Windows) ──────────────────────────────────────────────────
        'ctypes',
        'ctypes.wintypes',
        # ── stdlib (sometimes missed by analysis) ─────────────────────────────
        'http.server',
        'socketserver',
        'urllib.parse',
        'urllib.request',
        'hashlib',
        'queue',
        'threading',
        'subprocess',
        'webbrowser',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hooks/pyi_rth_compat312.py'],
    excludes=[
        # Training engines — too large, users install separately
        'torch',
        'torchvision',
        'torchaudio',
        'tensorflow',
        'jax',
        'triton',
        # ONNX Runtime — 300 MB GPU DLL, not used by the UI
        'onnxruntime',
        'onnxruntime.capi',
        # Apache Arrow / PyArrow — pulled in by onnxruntime
        'pyarrow',
        # Unused heavy packages
        'matplotlib',
        'scipy',
        'sklearn',
        'pandas',
        'jupyter',
        'IPython',
        'notebook',
        'pytest',
        'mypy',
        'black',
        'pylint',
        'flake8',
        # win11toast internal not needed as separate module
        'win11toast.utils',
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
    name='Universal_SR_Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # No console window in release
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
    name='Universal_SR_Studio',
)
