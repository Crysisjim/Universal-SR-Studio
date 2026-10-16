# -*- coding: utf-8 -*-
"""
engine_paths.py — Single source of truth for all USRS engine paths.

v2.5.6 refactor: one SHARED virtual-env (``runtimes/.venv``) replaces the two
per-engine ``.venv`` folders (``neosr/.venv`` + ``traiNNer-redux/.venv``).

Compatibility is preserved: every resolver prefers the shared venv but falls
back to the legacy per-engine ``.venv`` so existing installs keep working until
the user re-runs the installer. NOTHING is deleted.

Layout (target)::

    ~/USRS/
        runtimes/
            python-3.12.x/        # portable CPython 3.12 (neosr requires 3.12)
            .venv/                # ONE shared venv for BOTH engines
        neosr/                    # git clone (no .venv after refactor)
        traiNNer-redux/           # git clone (dev branch)
        experiments/
        datasets/

NeoSR pins ``python>=3.12,<3.13`` and ``numpy>=2.2.4``; traiNNer-redux (dev)
pins ``python>=3.11`` and ``numpy>=2`` — so a single Python 3.12 + numpy 2.x
venv satisfies both.
"""
from __future__ import annotations

import os
import sys

# ── Python runtime bundled in the portable build ──────────────────────────────
# neosr requires CPython 3.12 strictly (>=3.12,<3.13). traiNNer-redux dev works
# on 3.11+ so 3.12 satisfies both. Keep the *minor* fixed so the embeddable URL
# and the on-disk folder name stay in sync with install_portable_python().
PYTHON_MINOR = "3.12"
PYTHON_PATCH = "3.12.9"
PORTABLE_PY_DIRNAME = f"python-{PYTHON_PATCH}"

# Legacy portable folder(s) we may still find on older installs.
_LEGACY_PY_DIRNAMES = ("python-3.11.9",)

_WIN = sys.platform == "win32"
_PY_EXE = "python.exe" if _WIN else "python"
_SCRIPTS = "Scripts" if _WIN else "bin"


# ── Base locations ────────────────────────────────────────────────────────────
def user_home() -> str:
    return os.path.expanduser("~")


def base_engine_path() -> str:
    # "USRS" is the only engine folder the app uses/generates now (the old
    # "IA_Engine" name confused English-speaking users). Nothing is ever
    # deleted or migrated automatically — existing "IA_Engine" installs are
    # simply left untouched on disk, unused by the app going forward.
    #
    # === Dev isolation ===
    # Public builds use ~/USRS. On the developer's machine the stable copy's live
    # training lives in ~/USRS and must never be touched by dev builds, so a dev
    # root ~/USRS-EXPERIMENTAL is used instead — but ONLY when it carries the
    # marker file below (created by hand on that machine, never shipped). Every
    # runner and tool resolves the engine root through this function.
    # USRS_EXPERIMENTAL_ROOT (env var) overrides both.
    override = os.environ.get("USRS_EXPERIMENTAL_ROOT")
    if override:
        return override
    dev_root = os.path.join(user_home(), "USRS-EXPERIMENTAL")
    if os.path.isfile(os.path.join(dev_root, DEV_ISOLATION_MARKER)):
        return dev_root
    return os.path.join(user_home(), "USRS")


#: File whose presence in ~/USRS-EXPERIMENTAL routes the app to that dev root.
DEV_ISOLATION_MARKER = ".usrs_dev_isolation"


def runtimes_path() -> str:
    return os.path.join(base_engine_path(), "runtimes")


def neosr_path() -> str:
    return os.path.join(base_engine_path(), "neosr")


def redux_path() -> str:
    return os.path.join(base_engine_path(), "traiNNer-redux")


# ── Portable Python ───────────────────────────────────────────────────────────
def portable_python() -> str | None:
    """Return path to the bundled portable CPython, or None if absent.

    Prefers the 3.12 runtime; falls back to any legacy 3.11 folder so an
    in-progress migration doesn't lose the bootstrap interpreter.
    """
    rt = runtimes_path()
    for name in (PORTABLE_PY_DIRNAME, *_LEGACY_PY_DIRNAMES):
        p = os.path.join(rt, name, _PY_EXE)
        if os.path.exists(p):
            return p
    return None


# ── Shared venv ───────────────────────────────────────────────────────────────
def shared_venv_dir() -> str:
    return os.path.join(runtimes_path(), ".venv")


def shared_venv_python() -> str:
    return os.path.join(shared_venv_dir(), _SCRIPTS, _PY_EXE)


def legacy_engine_python(engine_root: str) -> str:
    """Per-engine venv python (pre-2.5.6 layout)."""
    return os.path.join(engine_root, ".venv", _SCRIPTS, _PY_EXE)


def resolve_engine_python(engine_root: str | None = None) -> str:
    """Best python.exe to run an engine subprocess.

    Order: shared venv → per-engine legacy venv → portable python → "".
    Returns the *first path that exists*; if none exist, returns the shared
    venv path (so callers can show a meaningful "install the engine" hint).
    """
    shared = shared_venv_python()
    if os.path.exists(shared):
        return shared
    if engine_root:
        legacy = legacy_engine_python(engine_root)
        if os.path.exists(legacy):
            return legacy
    pp = portable_python()
    if pp:
        return pp
    return shared


def any_engine_python() -> str:
    """Resolve a usable engine python without knowing which engine.

    Tries shared venv, then traiNNer-redux legacy, then neosr legacy.
    Returns "" if nothing is found.
    """
    candidates = [
        shared_venv_python(),
        legacy_engine_python(redux_path()),
        legacy_engine_python(neosr_path()),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


# ── traiNNer-redux engine branch ──────────────────────────────────────────────
# v2.6: track *master*. Since mid-2026 master is ahead of dev: it merged the dev work we
# relied on (ECO, spanf3, srformerv2) plus HEART/NERVE archs, >8bpc training and
# OOM/checkpoint fixes. The only dev-only commit is the Python 3.14 / newer-torch
# migration, which our Python 3.12 + torch 2.7 runtime must not pick up.
REDUX_GIT_URL = "https://github.com/the-database/traiNNer-redux"
REDUX_GIT_BRANCH = "master"
# Exact master commit this release is tested against (2026-10-03, HEART/NERVE merge).
# master moved on 2026-10-09 to a Python 3.14 / torch>=2.14 migration that our
# Python 3.12 runtime must not pick up: installs/updates check this commit out.
# Set to "" to follow the branch head again.
REDUX_GIT_PIN = "3db1bf5b5ac6a79e95ed92e9611d26b9a2397666"
NEOSR_GIT_PIN = ""
NEOSR_GIT_URL = "https://github.com/neosr-project/neosr"  # moved from muslll/neosr
NEOSR_GIT_BRANCH = "master"
