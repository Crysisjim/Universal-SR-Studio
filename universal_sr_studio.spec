# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — Universal SR Studio v2.5.7
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
        ('src/core/custom_archs', 'src/core/custom_archs'),     # v2.5.5: custom arch .py injected into engine (gfisrv2, smosr, spanpp, figsr — absent from official repos)
        ('src/core/custom_engine', 'src/core/custom_engine'),   # v2.5.6: custom TRAINING files injected into engine (spark_loss.py + inceptionnext_arch.py — SparkLoss backbone)
        ('src/core/custom_neosr_archs', 'src/core/custom_neosr_archs'),  # v2.5.6: aethernet arch for neosr training
        ('src/core/onnx_runner.py', 'src/core/'),               # v2.5.6: ONNX inference subprocess (onnxruntime in venv, not bundled in exe)
        # v2.5.6: ALL subprocess scripts run by the engine venv must be loose .py on disk
        ('src/core/neosr_runner.py', 'src/core/'),              # ESC inference (neosr venv)
        ('src/core/persistent_neosr_worker.py', 'src/core/'), # v2.5.9: ESC persistent worker (model loaded once)
        ('src/core/neosr_general_runner.py', 'src/core/'),      # ninasr/lmlt/eimn/drct (neosr venv)
        ('src/core/spanplus_runner.py', 'src/core/'),           # legacy SPANPlus runner
        ('src/core/redux_inference_runner.py', 'src/core/'),    # redux inference helper
        ('src/core/benchmark_runner.py', 'src/core/'),          # benchmark orchestrator
        ('src/core/arch_benchmark.py', 'src/core/'),            # neosr arch benchmark
        ('src/core/feature_benchmark.py', 'src/core/'),         # neosr feature benchmark
        ('src/core/redux_arch_benchmark.py', 'src/core/'),      # redux arch benchmark
        ('src/core/redux_feature_benchmark.py', 'src/core/'),   # redux feature benchmark
        ('src/core/model_export.py', 'src/core/'),              # model info/convert CLI (torch/safetensors in venv)
        ('src/core/tb_launcher.py', 'src/core/'),               # v2.5.6: TensorBoard subprocess launcher (venv python, must be loose .py on disk)
        ('src/core/post_proc_worker.py', 'src/core/'),          # v2.5.7: TemporalFix+Undistort subprocess worker (venv python, CUDA sm_61 fix)
        ('src/core/post_proc_session.py', 'src/core/'),         # v2.5.7: PostProcSession client for post_proc_worker
        # v2.5.7: deps de post_proc_worker — doivent être loose .py (importés par le venv Python externe)
        ('src/__init__.py', 'src/'),
        ('src/core/__init__.py', 'src/core/'),
        ('src/core/temporal_fix.py', 'src/core/'),
        ('src/core/undistort.py', 'src/core/'),
        ('src/core/model_manager.py', 'src/core/'),
        ('src/core/ort_session.py', 'src/core/'),
        ('src/core/models/__init__.py', 'src/core/models/'),
        ('src/core/models/temporalfix_arch.py', 'src/core/models/'),
        ('src/core/models/unet3d_tmt.py', 'src/core/models/'),
        ('src/core/models/tmt_inference.py', 'src/core/models/'),
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
        'src.core.post_proc_session',
        'src.core.post_proc_worker',
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
