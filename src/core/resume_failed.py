"""
resume_failed.py — Scan ~/USRS/.../experiments/ for interrupted trainings.

A training is "interrupted" if its log doesn't contain "End of training"
and there's a recent .state file we could resume from.
"""
import os
import re
import glob
from datetime import datetime


END_MARKERS = ["End of training", "End of training.", "training finished"]


def _read_log_tail(log_path: str, n_lines: int = 50) -> str:
    """Read the last n_lines of a log file."""
    if not os.path.exists(log_path):
        return ""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except Exception:
        return ""


def _find_latest_state(exp_dir: str) -> str:
    """Find the latest .state file in an experiment dir's training_states/."""
    states_dir = os.path.join(exp_dir, "training_states")
    if not os.path.isdir(states_dir):
        return ""
    # Sort numerically by iter number, not lexicographically
    def state_iter(path):
        m = re.search(r"(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0
    states = glob.glob(os.path.join(states_dir, "*.state"))
    states.sort(key=state_iter)
    return states[-1] if states else ""


def _find_latest_iter(exp_dir: str) -> int:
    """Find the highest iteration number from checkpoint files in models/.
    Supports both .pth (NeoSR) and .safetensors (Redux).
    """
    models_dir = os.path.join(exp_dir, "models")
    if not os.path.isdir(models_dir):
        return 0
    max_iter = 0
    for f in os.listdir(models_dir):
        if not (f.endswith(".pth") or f.endswith(".safetensors")):
            continue
        # Extract trailing iteration number
        stem = os.path.splitext(f)[0]
        m = re.search(r"(\d+)$", stem)
        if m:
            try:
                max_iter = max(max_iter, int(m.group(1)))
            except Exception:
                pass
    return max_iter


def _is_training_complete(exp_dir: str) -> bool:
    """Check if the training reached its end (via the log)."""
    log_pattern = os.path.join(exp_dir, "*.log")
    logs = glob.glob(log_pattern)
    if not logs:
        # No log file — could be very fresh or aborted
        return False
    log_path = max(logs, key=os.path.getmtime)
    tail = _read_log_tail(log_path, n_lines=20)
    return any(marker in tail for marker in END_MARKERS)


def scan_interrupted_trainings(engine_paths: list = None) -> list:
    """
    Scan engine experiment directories for interrupted trainings.

    Args:
        engine_paths: List of paths to scan (default: NeoSR + traiNNer-Redux)

    Returns:
        List of dicts with: name, path, last_iter, state_file, log_tail, mtime
        Sorted by mtime descending (most recent first).
    """
    if engine_paths is None:
        home = os.path.expanduser("~")
        from src.core.engine_paths import base_engine_path as _ebp
        _base = _ebp()
        engine_paths = [
            os.path.join(_base, "neosr", "experiments"),
            os.path.join(_base, "traiNNer-redux", "experiments"),
        ]

    interrupted = []

    for engine_exp_root in engine_paths:
        if not os.path.isdir(engine_exp_root):
            continue
        for entry in os.listdir(engine_exp_root):
            exp_dir = os.path.join(engine_exp_root, entry)
            if not os.path.isdir(exp_dir):
                continue
            # Skip "_archived_" directories — they're already archived
            if "_archived_" in entry:
                continue
            # Check if training is complete
            if _is_training_complete(exp_dir):
                continue
            # Get info
            state_file = _find_latest_state(exp_dir)
            last_iter = _find_latest_iter(exp_dir)
            if last_iter < 100 and not state_file:
                # Probably a freshly created dir that crashed at startup
                continue
            try:
                mtime = os.path.getmtime(exp_dir)
            except Exception:
                mtime = 0
            log_pattern = os.path.join(exp_dir, "*.log")
            logs = glob.glob(log_pattern)
            log_tail = ""
            running = False
            if logs:
                log_path = max(logs, key=os.path.getmtime)
                log_tail = _read_log_tail(log_path, n_lines=10)
                try:
                    import time as _t
                    running = (_t.time() - os.path.getmtime(log_path)) < 120
                except OSError:
                    pass
            interrupted.append({
                "name": entry,
                "path": exp_dir,
                "engine": "NeoSR" if "neosr" in engine_exp_root else "Redux",
                "last_iter": last_iter,
                "state_file": state_file,
                "log_tail": log_tail,
                "running": running,
                "mtime": mtime,
                "mtime_str": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else "?",
            })

    interrupted.sort(key=lambda x: x["mtime"], reverse=True)
    return interrupted


def find_associated_config(exp_dir: str) -> str:
    """
    Try to find the .toml/.yml config that was used for a given experiment.
    Looks in standard locations. Tolerant to folder name variants
    (trainner_redux, traiNNer-redux, etc.) and case differences.
    """
    name = os.path.basename(exp_dir)
    name_lower = name.lower()
    home = os.path.expanduser("~")
    from src.core.engine_paths import base_engine_path as _ebp
    _base = _ebp()
    base_roots = [
        os.path.join(_base, "Option Custom"),
        _base,
    ]
    search_dirs = []
    for root in base_roots:
        if not os.path.isdir(root):
            continue
        # Direct subdirs
        for sub in os.listdir(root):
            full = os.path.join(root, sub)
            if os.path.isdir(full):
                search_dirs.append(full)
                # One level deeper too (e.g. neosr/options)
                try:
                    for sub2 in os.listdir(full):
                        full2 = os.path.join(full, sub2)
                        if os.path.isdir(full2):
                            search_dirs.append(full2)
                except Exception:
                    pass

    # The engines copy the options file into the experiment folder: look there first.
    search_dirs = [exp_dir] + search_dirs
    exact, partial = [], []
    for sd in search_dirs:
        if not os.path.isdir(sd):
            continue
        try:
            for f in os.listdir(sd):
                if not f.endswith((".toml", ".yml", ".yaml")):
                    continue
                stem = os.path.splitext(f)[0].lower()
                p = os.path.join(sd, f)
                if sd == exp_dir or name_lower == stem:
                    exact.append(p)
                elif name_lower in stem or stem in name_lower:
                    partial.append(p)
        except Exception:
            pass
    # B51: a partial file-name match is accepted only when the config's own `name` is
    # this experiment — the first partial match used to be another experiment's config.
    for p in exact + partial:
        if _config_name(p) == name_lower:
            return p
    for p in exact:
        if _config_name(p) in ("", None):
            return p
    return ""


def _config_name(path):
    """`name` field of a .yml/.toml config, lower case; "" when absent; None if unreadable."""
    try:
        if path.endswith(".toml"):
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with open(path, "rb") as f:
                d = tomllib.load(f)
        else:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
        return str((d or {}).get("name", "")).strip().lower()
    except Exception:
        return None
