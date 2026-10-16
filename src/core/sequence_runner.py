"""
sequence_runner.py — runs Quick Upscale's video/sequence pipeline
(_sequence_video_upscale_folder: tscunet, temporalspan(v2), tfdat, MoTSR) inside the
engine venv. The portable exe has no torch in-process, so before this runner those
archs always failed there with "PyTorch non installé".

Usage:  <engine venv python> sequence_runner.py <args.json>
Prints the pipeline's log lines as they happen; the last line is "RESULT <json>".
Stop: the caller creates args["stop_file"]; it is checked before every frame.
"""
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # folder that contains src/
# APPEND, never prepend: in the portable build ROOT is PyInstaller's _internal/,
# which holds the exe's own Python 3.14 extension modules (_ctypes, numpy, PIL…).
# Ahead of the venv's stdlib they shadow it ("Module use of python314.dll
# conflicts with this version of Python") and torch fails to import.
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.append(p)


class _StopFile:
    def __init__(self, path):
        self.path = path

    def is_set(self):
        return bool(self.path) and os.path.exists(self.path)


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        a = json.load(f)
    from src.core import engine_paths as ep
    redux = ep.redux_path()
    if os.path.isdir(redux) and redux not in sys.path:
        sys.path.insert(0, redux)
    from src.core.quick_upscale import _sequence_video_upscale_folder

    def log(msg):
        print(str(msg).replace("\n", " "), flush=True)

    ok, total, errors = _sequence_video_upscale_folder(
        a["model_path"], a["input_folder"], a["output_folder"], a["detected_arch"],
        int(a.get("scale", 0)), log,
        progress_callback=None, stop_event=_StopFile(a.get("stop_file")),
        color_fix=a.get("color_fix", "none"),
        color_fix_wavelets=int(a.get("color_fix_wavelets", 4)),
        color_fix_radius=int(a.get("color_fix_radius", 32)),
        color_fix_fast=bool(a.get("color_fix_fast", False)),
        color_fix_strength=float(a.get("color_fix_strength", 1.0)),
        color_fix_device=a.get("color_fix_device", "auto"),
        out_bit_depth=int(a.get("out_bit_depth", 8)),
    )
    print("RESULT " + json.dumps({"success": ok, "total": total, "errors": errors}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"ERREUR sequence_runner : {e}", flush=True)
        print("RESULT " + json.dumps({"success": 0, "total": 0,
                                       "errors": [f"{e}\n{traceback.format_exc()[-600:]}"]}),
              flush=True)
        sys.exit(1)
