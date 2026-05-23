"""
tb_launcher.py — TensorBoard launcher with duplicate-plugin deduplication.

Fixes: ValueError: Duplicate plugins for name projector
Cause: tensorflow + tensorboard both register a 'projector' plugin.

Usage: python tb_launcher.py --logdir <path> --port <port>

Executed with the training venv Python (not the Studio Python).
"""
import sys


def _patch_duplicate_plugins():
    """Monkey-patch TensorBoardWSGI.__init__ to silently deduplicate plugins."""
    try:
        from tensorboard.backend import application

        _orig = application.TensorBoardWSGI.__init__

        def _dedup_init(self, plugins, *args, **kwargs):
            seen = set()
            unique = []
            for p in plugins:
                name = getattr(p, "plugin_name", None)
                if name is None or name not in seen:
                    unique.append(p)
                    if name:
                        seen.add(name)
            _orig(self, unique, *args, **kwargs)

        application.TensorBoardWSGI.__init__ = _dedup_init
    except Exception as e:
        print(f"[tb_launcher] Warning: could not patch duplicate plugins: {e}", flush=True)


if __name__ == "__main__":
    _patch_duplicate_plugins()
    from tensorboard.main import run_main
    run_main()
