import json
import os

# Always resolve relative to project root (2 levels up from src/core/),
# regardless of working directory at runtime.
_SETTINGS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS_FILE = os.path.join(_SETTINGS_DIR, "user_settings.json")


def to_bool(value) -> bool:
    """Centralized boolean conversion (ARCH-03 fix).
    Handles: True/False, "true"/"false", "1"/"0", "on"/"off", etc.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "on", "yes", "oui")
    return False


class SettingsManager:
    """Persistent JSON settings with mtime-based caching (PERF-01 fix)."""

    def __init__(self):
        self.data: dict = {}
        self._last_mtime: float = 0
        self.load()

    def load(self):
        """Load settings from disk."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                self._last_mtime = os.path.getmtime(SETTINGS_FILE)
                # BUG-12: Clean up polluted "null" key
                if "null" in self.data:
                    del self.data["null"]
            except (json.JSONDecodeError, OSError) as e:
                print(f"[Settings] Erreur chargement: {e}")
                self.data = {}
        else:
            self.data = {}

    def _reload_if_changed(self):
        """Only reload from disk if the file was modified externally (PERF-01)."""
        try:
            if os.path.exists(SETTINGS_FILE):
                current_mtime = os.path.getmtime(SETTINGS_FILE)
                if current_mtime != self._last_mtime:
                    self.load()
        except OSError:
            pass

    def save(self):
        """Save settings to disk."""
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            self._last_mtime = os.path.getmtime(SETTINGS_FILE)
        except OSError as e:
            print(f"[Settings] Erreur sauvegarde: {e}")

    def get(self, key: str, default=""):
        """Get a setting value. Uses mtime cache instead of reloading every time."""
        self._reload_if_changed()
        return self.data.get(key, default)

    def set(self, key, value):
        """Set a setting value. Rejects None keys (BUG-12 fix)."""
        if key is None:
            print(f"[Settings] WARN: Tentative de set(None, {value!r}) ignorée")
            return
        self._reload_if_changed()
        self.data[str(key)] = value
        self.save()

    def delete(self, key: str):
        """Delete a setting key."""
        self._reload_if_changed()
        if key in self.data:
            del self.data[key]
            self.save()
