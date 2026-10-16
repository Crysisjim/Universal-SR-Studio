"""
benchmark_runner.py — Point d'entrée unifié pour tous les benchmarks SR.

Fusionne les 4 scripts spécialisés en un seul outil cohérent.
Sélectionne automatiquement le bon moteur et le bon type de test.

Usage:
    python benchmark_runner.py [options]

    --engine neosr|redux       Moteur d'entraînement (défaut: redux)
    --type   arch|feature      Type de benchmark (défaut: arch)
    --n-iter 2500              Itérations par test
    --timeout 3600             Timeout en secondes par test
    --tests compact,span       Tests ciblés (vide = tous)
    --modes normal,fp16,bf16   Modes précision (arch only — redux)
    --train-gt PATH            Dataset GT d'entraînement
    --output-dir PATH          Dossier de résultats
    --reset                    Repart de zéro (ignore état existant)
    --no-upscale               Passe le test d'inférence finale
    --list                     Liste les tests disponibles et quitte

Backends:
    NeoSR  + arch    → arch_benchmark.py
    NeoSR  + feature → feature_benchmark.py
    Redux  + arch    → redux_arch_benchmark.py
    Redux  + feature → redux_feature_benchmark.py
"""
from __future__ import annotations

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Encodage UTF-8 (consoles cp1252 sur Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_THIS_DIR = Path(__file__).parent

# ── Correspondance engine × type → script ──────────────────────────────────────
_BACKENDS: dict[tuple[str, str], str] = {
    ("neosr",  "arch"):    str(_THIS_DIR / "arch_benchmark.py"),
    ("neosr",  "feature"): str(_THIS_DIR / "feature_benchmark.py"),
    ("redux",  "arch"):    str(_THIS_DIR / "redux_arch_benchmark.py"),
    ("redux",  "feature"): str(_THIS_DIR / "redux_feature_benchmark.py"),
}

# ── Chemins Python par moteur ──────────────────────────────────────────────────
def _find_python(engine: str) -> str:
    """Trouve python.exe dans le venv du moteur. Retourne sys.executable si absent."""
    try:
        from src.core.engine_paths import base_engine_path as _ebp
    except ImportError:  # venv runner: engine_paths.py ships loose next to this file
        from engine_paths import base_engine_path as _ebp
    _root = Path(_ebp())
    base = _root / ("traiNNer-redux" if engine == "redux" else "neosr")

    candidates = [
        _root / "runtimes" / ".venv" / "Scripts" / "python.exe",  # v2.5.6 shared
        _root / "runtimes" / ".venv" / "bin" / "python",          # Linux shared
        base / ".venv" / "Scripts" / "python.exe",   # Windows .venv
        base / "venv"  / "Scripts" / "python.exe",   # Windows venv
        base / ".venv" / "bin"     / "python",        # Linux .venv
        base / "venv"  / "bin"     / "python",        # Linux venv
        base / ".venv" / "bin"     / "python3",
        base / "venv"  / "bin"     / "python3",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


# ── Valeurs par défaut selon le type ──────────────────────────────────────────
_DEFAULTS: dict[str, dict] = {
    "arch":    {"n_iter": 2500, "timeout": 3600},
    "feature": {"n_iter": 500,  "timeout": 900},
}

# ── Flags réellement supportés par chaque backend ─────────────────────────────
# _build_backend_args used to append --tests/--reset/--list unconditionally to all
# 4 backends, assuming they share one CLI surface. They don't — confirmed by
# grepping each script's own add_argument() calls. Sending an unsupported flag
# makes argparse exit(2) ("unrecognized arguments"), which is exactly what 3 of
# the 8 Moteur×Type×Action combinations did in practice (2026-09-17 fix):
#   - (neosr, arch): no --tests/--reset/--list at all (uses --archs instead)
#   - (neosr, feature): --tests/--list yes, but no --reset
#   - (redux, arch) and (redux, feature): all three supported
_SUPPORTS_TESTS = {("neosr", "arch"): False, ("neosr", "feature"): True,
                    ("redux", "arch"): True,  ("redux", "feature"): True}
_SUPPORTS_RESET = {("neosr", "arch"): False, ("neosr", "feature"): False,
                    ("redux", "arch"): True,  ("redux", "feature"): True}
_SUPPORTS_LIST  = {("neosr", "arch"): True,  ("neosr", "feature"): True,
                    ("redux", "arch"): True,  ("redux", "feature"): True}


def _build_backend_args(args: argparse.Namespace) -> list[str]:
    """Traduit les args unifiés vers les args du script backend."""
    btype   = args.type
    engine  = args.engine
    key     = (engine, btype)
    backend = _BACKENDS[key]
    python  = _find_python(engine)

    cmd = [python, backend]

    # Args communs à tous les backends
    cmd += ["--n-iter",  str(args.n_iter)]
    cmd += ["--timeout", str(args.timeout)]
    if args.tests:
        if _SUPPORTS_TESTS[key]:
            cmd += ["--tests", args.tests]
        elif key == ("neosr", "arch"):
            # arch_benchmark.py (neosr) names the same thing --archs instead of --tests.
            cmd += ["--archs", args.tests]
    if args.train_gt:
        cmd += ["--train-gt", args.train_gt]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
    if args.reset and _SUPPORTS_RESET[key]:
        cmd.append("--reset")
    if args.no_upscale and key != ("redux", "feature"):
        # B56: redux_feature_benchmark has no upscale test at all (no --no-upscale)
        cmd.append("--no-upscale")
    if getattr(args, "chain_features", False) and key == ("redux", "arch"):
        cmd += ["--chain-features", "--feature-n-iter", str(args.n_iter)]
    # B53: validation folders and test image
    if args.val_gt:
        cmd += ["--val-gt", args.val_gt]
    if args.val_lq:
        cmd += ["--val-lq", args.val_lq]
    if args.test_image and key != ("redux", "feature"):
        cmd += ["--test-img" if key == ("redux", "arch") else "--test-image", args.test_image]
    if args.list:
        if _SUPPORTS_LIST[key]:
            cmd.append("--list")
        else:
            # (neosr, arch) has no --list of its own — arch_benchmark.py --archs=""
            # with no training GT configured just prints nothing useful, so tell the
            # user honestly instead of letting argparse crash with exit(2).
            print("[benchmark_runner] --list non supporté par arch_benchmark.py (NeoSR + "
                  "Architectures) — ce backend n'a pas d'option de listing dédiée.",
                  flush=True)
            return []

    # Args spécifiques au type / moteur
    if btype == "arch":
        if args.modes:
            # B54: both arch backends take the exact list (NeoSR one used to get
            # --laptop = all four modes, or nothing and an unanswerable input())
            cmd += ["--modes", args.modes]

    return cmd


def _print_header(engine: str, btype: str, backend: str, python: str) -> None:
    print("=" * 80, flush=True)
    print(f"  Universal SR Studio — Benchmark Unifié", flush=True)
    print(f"  Moteur  : {engine.upper()}", flush=True)
    print(f"  Type    : {btype}", flush=True)
    print(f"  Script  : {Path(backend).name}", flush=True)
    print(f"  Python  : {python}", flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark unifié NeoSR / traiNNer-Redux (arch + feature)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--engine",     choices=["neosr", "redux"], default="redux",
                    help="Moteur d'entraînement (défaut: redux)")
    ap.add_argument("--type",       choices=["arch", "feature"], default="arch",
                    help="Type de benchmark : arch (architectures) ou feature (losses/optim/…)")
    ap.add_argument("--n-iter",     type=int,  default=0,
                    help="Itérations par test (0 = défaut selon le type)")
    ap.add_argument("--timeout",    type=int,  default=0,
                    help="Timeout par test en secondes (0 = défaut selon le type)")
    ap.add_argument("--tests",      type=str,  default="",
                    help="Tests ciblés, séparés par virgule (vide = tous)")
    ap.add_argument("--modes",      type=str,  default="normal",
                    help="Modes précision séparés par virgule : normal,fp16,bf16,tf32\n"
                         "(Redux arch uniquement)")
    ap.add_argument("--train-gt",   type=str,  default="",
                    help="Chemin vers le dataset HQ d'entraînement")
    ap.add_argument("--output-dir", type=str,  default="",
                    help="Dossier pour les résultats (créé si absent)")
    ap.add_argument("--reset",      action="store_true",
                    help="Ignore l'état sauvegardé et repart de zéro")
    ap.add_argument("--no-upscale", action="store_true",
                    help="Passe le test d'inférence après chaque arch/feature")
    ap.add_argument("--chain-features", action="store_true",
                    help="redux arch: then the feature benchmark on the fastest arch")
    ap.add_argument("--val-gt",     type=str,  default="")
    ap.add_argument("--val-lq",     type=str,  default="")
    ap.add_argument("--test-image", type=str,  default="")
    ap.add_argument("--list",       action="store_true",
                    help="Liste les tests disponibles et quitte")
    args = ap.parse_args()

    # Appliquer les défauts selon le type si non fournis
    if args.n_iter  == 0: args.n_iter  = _DEFAULTS[args.type]["n_iter"]
    if args.timeout == 0: args.timeout = _DEFAULTS[args.type]["timeout"]

    engine  = args.engine
    btype   = args.type
    key     = (engine, btype)

    if key not in _BACKENDS:
        print(f"[ERROR] Combinaison inconnue : engine={engine} type={btype}", flush=True)
        sys.exit(1)

    backend = _BACKENDS[key]
    python  = _find_python(engine)

    if not os.path.isfile(backend):
        print(f"[ERROR] Script backend introuvable : {backend}", flush=True)
        print(f"        Vérifiez que Universal SR Studio DEV est complet.", flush=True)
        sys.exit(1)

    if python == sys.executable and not Path(python).samefile(sys.executable):
        # Uniquement si le venv n'existe vraiment pas (pas le cas quand on tourne déjà dedans)
        print(f"[WARN]  Venv {engine} introuvable — utilisation du Python courant.", flush=True)
        print(f"        Résultats peuvent différer de l'environnement de training réel.", flush=True)

    if not args.list:
        _print_header(engine, btype, backend, python)

    cmd = _build_backend_args(args)
    if not cmd:
        sys.exit(1)

    print(f"[CMD] {' '.join(cmd)}\n", flush=True)

    # Exécute le backend en passant stdin/stdout/stderr au terminal courant
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=None,   # hérite du terminal → affichage live direct
            stderr=None,
            stdin=None,
        )
        proc.wait()
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("\n[INFO] Interruption utilisateur (Ctrl+C).", flush=True)
        try:
            proc.kill()
        except Exception:
            pass
        sys.exit(1)
    except Exception as ex:
        print(f"[ERROR] Impossible de lancer le backend : {ex}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
