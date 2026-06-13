"""
post_proc_session.py — Client for post_proc_worker subprocess.

Launches the worker with the venv Python (correct torch/CUDA), communicates
via JSON-lines over stdin/stdout — same pattern as PersistentBatchSession.

Usage:
    from src.core.post_proc_session import PostProcSession
    pp = PostProcSession(venv_py, log=callback)
    pp.start()
    pp.init_tf(mode="ort_s2", strength=0.5, window=7, precision="float32")
    pp.init_ud(mode="classic", strength=0.35, window=5, precision="float32")
    # per-frame:
    pp.push_ud(out_path)
    pp.push_tf(out_path)
    # end:
    pp.flush_ud()
    pp.flush_tf()
    pp.stop()

v2.5.7 — Universal SR Studio
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Callable, Optional

_WORKER = os.path.join(os.path.dirname(__file__), "post_proc_worker.py")


class PostProcSession:
    """
    Persistent subprocess session for TemporalFix + Undistort.
    Worker runs with venv Python → correct torch + CUDA.
    Frames passed as file paths on disk (read/write by worker).
    """

    def __init__(self, venv_py: str, log: Optional[Callable] = None) -> None:
        self._venv_py = venv_py
        self._log = log or (lambda m: None)
        self._proc: Optional[subprocess.Popen] = None
        self._tf_ok = False
        self._ud_ok = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if not os.path.isfile(self._venv_py):
            self._log(f"[PostProc] venv Python introuvable : {self._venv_py}")
            return False
        if not os.path.isfile(_WORKER):
            self._log(f"[PostProc] Worker introuvable : {_WORKER}")
            return False
        _env = os.environ.copy()
        _env["PYTHONIOENCODING"] = "utf-8"
        _env["PYTHONUTF8"] = "1"
        try:
            self._proc = subprocess.Popen(
                [self._venv_py, _WORKER],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                env=_env,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            return True
        except Exception as e:
            self._log(f"[PostProc] Échec lancement : {e}")
            return False

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._send({"cmd": "quit"})
        except Exception:
            pass
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        self._proc = None
        self._tf_ok = False
        self._ud_ok = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send(self, obj: dict) -> dict:
        assert self._proc is not None
        self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        resp = self._proc.stdout.readline()
        if not resp:
            raise RuntimeError("Post-proc worker closed stdout unexpectedly")
        return json.loads(resp.strip())

    # ── TemporalFix ───────────────────────────────────────────────────────────

    def init_tf(self, mode: str = "classic", strength: float = 0.5,
                window: int = 7, precision: str = "float32",
                device: str = "auto") -> int:
        """Init TF in worker. Returns latency (frames). Raises on failure."""
        resp = self._send({"cmd": "init_tf", "mode": mode, "strength": strength,
                           "window": window, "precision": precision, "device": device})
        if resp.get("status") == "ready":
            self._tf_ok = True
            return int(resp.get("latency", window // 2))
        raise RuntimeError(f"init_tf: {resp.get('msg', resp)}")

    def push_tf(self, path: str) -> bool:
        """Push frame to TF. Worker reads + writes result to same path. Returns True if written."""
        if not self._tf_ok:
            return False
        resp = self._send({"cmd": "push_tf", "path": path})
        if resp.get("status") == "error":
            raise RuntimeError(resp.get("msg", "push_tf error"))
        return resp.get("status") == "written"

    def flush_tf(self) -> int:
        """Flush TF pending frames. Returns count written."""
        if not self._tf_ok:
            return 0
        resp = self._send({"cmd": "flush_tf"})
        if resp.get("status") == "error":
            raise RuntimeError(resp.get("msg", "flush_tf error"))
        return int(resp.get("count", 0))

    # ── Undistort ─────────────────────────────────────────────────────────────

    def init_ud(self, mode: str = "classic", strength: float = 0.35,
                window: int = 5, precision: str = "float32", device: str = "auto",
                tmt_window: int = 10, tmt_patch: int = 256) -> int:
        """Init UD in worker. Returns latency (frames). Raises on failure."""
        resp = self._send({"cmd": "init_ud", "mode": mode, "strength": strength,
                           "window": window, "precision": precision, "device": device,
                           "tmt_window": tmt_window, "tmt_patch": tmt_patch})
        if resp.get("status") == "ready":
            self._ud_ok = True
            return int(resp.get("latency", window // 2))
        raise RuntimeError(f"init_ud: {resp.get('msg', resp)}")

    def push_ud(self, path: str) -> bool:
        """Push frame to UD. Worker reads + writes result. Returns True if written."""
        if not self._ud_ok:
            return False
        resp = self._send({"cmd": "push_ud", "path": path})
        if resp.get("status") == "error":
            raise RuntimeError(resp.get("msg", "push_ud error"))
        return resp.get("status") == "written"

    def flush_ud(self) -> int:
        """Flush UD pending frames. Returns count written."""
        if not self._ud_ok:
            return 0
        resp = self._send({"cmd": "flush_ud"})
        if resp.get("status") == "error":
            raise RuntimeError(resp.get("msg", "flush_ud error"))
        return int(resp.get("count", 0))
