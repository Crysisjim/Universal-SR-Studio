"""
post_proc_worker.py — Subprocess post-processing worker (TemporalFix + Undistort).

Runs with the venv Python (torch 2.7+cu126 — correct CUDA support for sm_61).
Protocol: JSON-lines on stdin/stdout, same pattern as persistent_upscale_worker.

Commands:
  {"cmd": "init_tf", "mode": "ort_s2", "strength": 0.5, "window": 7, "precision": "float32"}
  → {"status": "ready", "latency": 3, "window": 7}

  {"cmd": "push_tf", "path": "/path/to/frame.png"}
  → {"status": "pending"} | {"status": "written", "path": "/path/to/frame.png"}

  {"cmd": "flush_tf"}
  → {"status": "flushed", "count": N}

  Same pattern for "init_ud", "push_ud", "flush_ud".

  {"cmd": "quit"} → exit

v2.5.7 — Universal SR Studio
"""
import sys
import os
import json
import traceback
from collections import deque

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Project root: <root>/src/core/post_proc_worker.py → up 2 levels
# IMPORTANT: use append, NOT insert(0). _internal/ contains numpy/PIL compiled for
# Python314 (PyInstaller host). insert(0) would shadow venv packages → DLL version
# conflict → "Module use of python314.dll conflicts with this version of Python".
# With append, venv site-packages are searched first (torch/numpy/PIL from Python312),
# then _internal/ for src.core.* loose .py files.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)


def _send(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _read_frame(path: str):
    from PIL import Image
    import numpy as np
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _write_frame(arr, path: str) -> None:
    from PIL import Image
    import numpy as np
    Image.fromarray((arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


# ── TemporalFix ───────────────────────────────────────────────────────────────
_tf_proc = None
_tf_pending: deque = deque()


def _init_tf(cmd: dict) -> None:
    global _tf_proc, _tf_pending
    try:
        from src.core.temporal_fix import TemporalFixProcessor
        _tf_proc = TemporalFixProcessor(
            mode=cmd.get("mode", "classic"),
            strength=float(cmd.get("strength", 0.5)),
            window=int(cmd.get("window", 7)),
            precision=cmd.get("precision", "float32"),
            device=cmd.get("device", "auto"),
        )
        _tf_pending = deque()
        _send({"status": "ready", "latency": _tf_proc.latency, "window": _tf_proc.window})
    except Exception as e:
        _send({"status": "error", "msg": f"init_tf: {e}\n{traceback.format_exc()}"})


def _push_tf(cmd: dict) -> None:
    global _tf_pending
    if _tf_proc is None:
        _send({"status": "error", "msg": "TF not initialized"})
        return
    path = cmd["path"]
    try:
        frame_np = _read_frame(path)
        _tf_pending.append(path)
        result = _tf_proc.push(frame_np)
        if result is not None:
            write_path = _tf_pending.popleft()
            _write_frame(result, write_path)
            _send({"status": "written", "path": write_path})
        else:
            _send({"status": "pending"})
    except Exception as e:
        _send({"status": "error", "msg": f"push_tf: {e}\n{traceback.format_exc()}"})


def _flush_tf(_cmd: dict) -> None:
    global _tf_pending
    if _tf_proc is None:
        _send({"status": "flushed", "count": 0})
        return
    count = 0
    try:
        for result in _tf_proc.flush():
            if not _tf_pending:
                break
            _write_frame(result, _tf_pending.popleft())
            count += 1
    except Exception as e:
        _send({"status": "error", "msg": f"flush_tf: {e}"})
        return
    _send({"status": "flushed", "count": count})


# ── Undistort ─────────────────────────────────────────────────────────────────
_ud_proc = None
_ud_pending: deque = deque()


def _init_ud(cmd: dict) -> None:
    global _ud_proc, _ud_pending
    try:
        from src.core.undistort import UndistortProcessor
        _ud_proc = UndistortProcessor(
            mode=cmd.get("mode", "classic"),
            strength=float(cmd.get("strength", 0.35)),
            window=int(cmd.get("window", 5)),
            precision=cmd.get("precision", "float32"),
            device=cmd.get("device", "auto"),
            tmt_window=int(cmd.get("tmt_window", 10)),
            tmt_patch=int(cmd.get("tmt_patch", 256)),
        )
        _ud_pending = deque()
        latency = getattr(_ud_proc, "latency", int(cmd.get("window", 5)) // 2)
        _send({"status": "ready", "latency": latency})
    except Exception as e:
        _send({"status": "error", "msg": f"init_ud: {e}\n{traceback.format_exc()}"})


def _push_ud(cmd: dict) -> None:
    global _ud_pending
    if _ud_proc is None:
        _send({"status": "error", "msg": "UD not initialized"})
        return
    path = cmd["path"]
    try:
        frame_np = _read_frame(path)
        _ud_pending.append(path)
        result = _ud_proc.push(frame_np)
        if result is not None:
            write_path = _ud_pending.popleft()
            _write_frame(result, write_path)
            _send({"status": "written", "path": write_path})
        else:
            _send({"status": "pending"})
    except Exception as e:
        _send({"status": "error", "msg": f"push_ud: {e}\n{traceback.format_exc()}"})


def _flush_ud(_cmd: dict) -> None:
    global _ud_pending
    if _ud_proc is None:
        _send({"status": "flushed", "count": 0})
        return
    count = 0
    try:
        for result in _ud_proc.flush():
            if not _ud_pending:
                break
            _write_frame(result, _ud_pending.popleft())
            count += 1
    except Exception as e:
        _send({"status": "error", "msg": f"flush_ud: {e}"})
        return
    _send({"status": "flushed", "count": count})


# ── Main loop ─────────────────────────────────────────────────────────────────
_HANDLERS = {
    "init_tf":  _init_tf,
    "push_tf":  _push_tf,
    "flush_tf": _flush_tf,
    "init_ud":  _init_ud,
    "push_ud":  _push_ud,
    "flush_ud": _flush_ud,
}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            _send({"status": "error", "msg": f"JSON parse: {e}"})
            continue

        op = cmd.get("cmd", "")
        if op == "quit":
            _send({"status": "bye"})
            break

        handler = _HANDLERS.get(op)
        if handler is None:
            _send({"status": "error", "msg": f"Unknown command: {op}"})
            continue

        handler(cmd)


if __name__ == "__main__":
    main()
