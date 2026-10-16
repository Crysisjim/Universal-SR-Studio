"""
notify.py — shared Win11 toast + sound helpers for the Qt app.

Mirrors the CustomTkinter app's `_show_toast` / `_play_sound`:
  • toasts fire via src.core.toast_notifications.show_toast in a daemon thread
    (win11toast blocks), gated by a per-event settings toggle (default ON);
  • sounds play a WAV from assets/, gated by a settings toggle.

Notification setting keys match the legacy app so existing prefs carry over:
  notif_win11_upscale · notif_win11_batch · notif_win11_errors · notif_win11_training
Sound setting keys: sound_on (training end) · snd_error · snd_warning · snd_about
(the Settings checkboxes); the volume is "sound_vol" (0-100).
"""
from __future__ import annotations

import os
import threading


def notify(settings, title: str, message: str = "", notif_key: str | None = None,
           audio: str = "default") -> None:
    """Fire a Win11 toast unless its per-event setting is disabled."""
    if notif_key is not None and not _enabled(settings, notif_key):
        return

    def _fire():
        try:
            from src.core.toast_notifications import show_toast
            show_toast(title, message, duration="short", audio=audio)
        except Exception:
            pass

    threading.Thread(target=_fire, daemon=True).start()


# old keys some callers used -> the Settings checkbox that really controls the sound
_KEY_ALIASES = {"sound_error_enabled": "snd_error", "sound_warning_enabled": "snd_warning"}


def _volume_copy(path: str, vol: int) -> str:
    """WAV scaled to `vol` % (winsound has no volume control: the Settings slider did
    nothing). 16-bit PCM only; other formats play at full volume. Cached in %TEMP%."""
    if vol >= 100:
        return path
    try:
        import tempfile
        import wave
        import array
        out = os.path.join(tempfile.gettempdir(), f"uss_snd_{os.path.basename(path)}_{vol}.wav")
        if os.path.isfile(out) and os.path.getmtime(out) >= os.path.getmtime(path):
            return out
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                return path
            params, frames = w.getparams(), w.readframes(w.getnframes())
        a = array.array("h", frames)
        f = max(0.0, min(1.0, vol / 100.0))
        a = array.array("h", (int(x * f) for x in a))
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(a.tobytes())
        return out
    except Exception:
        return path


def play_sound(settings, sound_name: str, setting_key: str | None = None, force: bool = False) -> None:
    """Play assets/<sound_name>.WAV asynchronously at the Settings volume, respecting its
    checkbox (force=True: the Settings "Test" buttons)."""
    if not force and setting_key is not None:
        if not _enabled(settings, _KEY_ALIASES.get(setting_key, setting_key)):
            return
    from qt_ui.theme import asset
    path = asset(f"{sound_name}.WAV")
    if not os.path.isfile(path):
        path = path[:-4] + ".wav"
    if not os.path.isfile(path):
        return
    try:
        vol = int(float(settings.get("sound_vol", 100) or 100))
    except (TypeError, ValueError):
        vol = 100
    if vol <= 0:
        return
    try:
        import winsound
        winsound.PlaySound(_volume_copy(path, vol), winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass


def _enabled(settings, key) -> bool:
    v = settings.get(key, True)
    if isinstance(v, bool):
        return v
    return str(v).lower() not in ("0", "false", "no", "off")
