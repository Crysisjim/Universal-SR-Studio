"""
main_qt.py — entry point for the PySide6 Qt rewrite preview (v2.6.0).

Run:  python main_qt.py
This is a parallel, non-destructive shell next to the CustomTkinter app (main.py).
It reuses the existing GUI-agnostic backend in src/core/.
"""
import os
import sys
import warnings

# Silence the deprecated-`pynvml` FutureWarning emitted by torch.cuda's
# `import pynvml` when the legacy `pynvml` shim is installed alongside
# `nvidia-ml-py`. We read NVML via nvidia-ml-py ourselves; this is torch's
# own import, not ours, so we filter it before torch is ever loaded.
warnings.filterwarnings("ignore", message=r".*pynvml package is deprecated.*")
# The GUI's Python may import a torch built for newer GPUs (sm_75+); on an older
# card (e.g. GTX 1080 Ti = sm_61) torch prints CUDA-capability UserWarnings. Training
# runs in the engine venv (separate, compatible torch), so these are cosmetic here.
for _msg in (r".*cuda capability.*", r".*not compatible with the current PyTorch.*",
             r".*Please install PyTorch with.*", r".*CUDA capabilit.*"):
    warnings.filterwarnings("ignore", message=_msg)

# Make `src` and `qt_ui` importable when run from the project root.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtCore import qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from qt_ui.main_window import MainWindow


def _qt_message_filter(mode, ctx, msg):
    # Hide the harmless "QFont::setPointSize: Point size <= 0 (-1)" spam emitted when a
    # stylesheet sets font-family without font-size (Qt falls back to the default size).
    if "Point size <= 0" in msg or "setPixelSize" in msg and "<= 0" in msg:
        return
    sys.stderr.write(msg + "\n")


qInstallMessageHandler(_qt_message_filter)


def _install_tooltip_delay(app, wake_ms=1100):
    """Make tooltips appear after a longer hover so they don't flash instantly."""
    from PySide6.QtWidgets import QProxyStyle, QStyle

    class _SlowTipStyle(QProxyStyle):
        def styleHint(self, hint, option=None, widget=None, returnData=None):
            if hint == QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
                return wake_ms
            return super().styleHint(hint, option, widget, returnData)

    try:
        app.setStyle(_SlowTipStyle(app.style()))
    except Exception:
        pass


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Universal SR Studio (Qt)")
    _install_tooltip_delay(app)

    # App / window / dialog icon (all top-level windows inherit the app icon).
    from PySide6.QtGui import QIcon
    for _ic in ("icon.ico", "icon.png"):
        _p = os.path.join(_ROOT, "assets", _ic)
        if os.path.isfile(_p):
            app.setWindowIcon(QIcon(_p))
            break

    # Reuse the existing settings store for persistence (accent, mode, …).
    try:
        from src.core.settings import SettingsManager
        settings = SettingsManager()
    except Exception:
        class _Mem:
            _d = {}
            def get(self, k, d=""): return self._d.get(k, d)
            def set(self, k, v): self._d[k] = v
        settings = _Mem()

    # One-time migration: the old CTk app stored API keys as `api_key_{Full Label}`
    # (e.g. "api_key_OpenRouter (Gratuit)"); the Qt app uses slugs (`api_key_openrouter`).
    # Without this, keys saved under the CTk build are silently invisible in Qt.
    try:
        _LEGACY_API_KEY_LABELS = {
            "api_key_openrouter": "OpenRouter (Gratuit)",
            "api_key_github": "GitHub Models (Gratuit)",
            "api_key_google": "Google (Gemini)",
            "api_key_anthropic": "Anthropic (Claude)",
            "api_key_openai": "OpenAI (ChatGPT)",
            "api_key_xai": "xAI (Grok)",
            "api_key_deepseek": "DeepSeek",
            "api_key_nvidia": "NVIDIA NIM",
        }
        for _slug, _label in _LEGACY_API_KEY_LABELS.items():
            if not settings.get(_slug, ""):
                _legacy = settings.get(f"api_key_{_label}", "")
                if _legacy:
                    settings.set(_slug, _legacy)
    except Exception:
        pass

    # Premier lancement : demander la langue de l'UI (FR/EN) une seule fois.
    try:
        if not settings.get("qt_lang", ""):
            from PySide6.QtWidgets import QMessageBox
            box = QMessageBox()
            box.setWindowTitle("Langue / Language")
            box.setText("Choisissez la langue de l'interface :\nChoose the interface language:")
            box.addButton("Français", QMessageBox.AcceptRole)
            en = box.addButton("English", QMessageBox.AcceptRole)
            box.exec()
            settings.set("qt_lang", "en" if box.clickedButton() is en else "fr")
    except Exception:
        pass

    try:
        from qt_ui.i18n import init_from_settings
        init_from_settings(settings)
    except Exception:
        pass

    # Bundle A : copie le python 3.12.9 embarqué → base_engine_path/runtimes dès
    # le démarrage, pour que le statut « Python portable » soit Prêt d'emblée et
    # que l'install moteur saute le téléchargement du python.
    try:
        from src.core import engine_installer
        engine_installer.ensure_portable_python_bundled()
        engine_installer.ensure_workspace_folders()  # datasets/Option Custom/experiments/… (comme CTk)
    except Exception:
        pass

    try:
        from src.core import vs_aa
        vs_aa.init_from_settings(settings)  # honor user-set VapourSynth path (portable-friendly)
    except Exception:
        pass

    win = MainWindow(settings, app)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
