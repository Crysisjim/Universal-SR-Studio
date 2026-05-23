"""
validation_rotation.py — Rotate validation image subsets across runs.

Instead of always validating on the same 7 images, pick a different subset
of N from a larger pool of 50-100+ validation images. More representative
of true generalization.
"""
import os
import random
import json
import time


def get_state_path(val_pool_dir: str) -> str:
    """Path to the rotation state file for a given validation pool."""
    base = os.path.basename(val_pool_dir.rstrip("/\\"))
    return os.path.join(val_pool_dir, f".rotation_state_{base}.json")


def load_state(val_pool_dir: str) -> dict:
    """Load rotation state."""
    state_path = get_state_path(val_pool_dir)
    if not os.path.exists(state_path):
        return {"current_subset": [], "last_rotation": 0, "history": []}
    try:
        with open(state_path, "r") as f:
            return json.load(f)
    except Exception:
        return {"current_subset": [], "last_rotation": 0, "history": []}


def save_state(val_pool_dir: str, state: dict):
    """Save rotation state."""
    try:
        with open(get_state_path(val_pool_dir), "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def list_validation_images(val_pool_dir: str) -> list:
    """List all validation images in the pool directory."""
    if not os.path.isdir(val_pool_dir):
        return []
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    return sorted([
        f for f in os.listdir(val_pool_dir)
        if f.lower().endswith(exts) and not f.startswith(".")
    ])


def select_subset(val_pool_dir: str, subset_size: int = 7,
                  strategy: str = "rotation", seed: int = None) -> list:
    """
    Select a subset of validation images.

    Args:
        val_pool_dir: Pool of validation images to choose from
        subset_size: Number of images to select
        strategy: "rotation" (different each call), "random", "fixed" (always same)
        seed: For reproducibility with strategy="random"

    Returns:
        List of full paths to the selected images.
    """
    all_images = list_validation_images(val_pool_dir)
    if not all_images:
        return []

    if subset_size >= len(all_images):
        return [os.path.join(val_pool_dir, f) for f in all_images]

    if strategy == "fixed":
        # Always pick the first N (sorted)
        chosen = all_images[:subset_size]
    elif strategy == "random":
        if seed is not None:
            random.seed(seed)
        chosen = random.sample(all_images, subset_size)
    else:  # rotation
        state = load_state(val_pool_dir)
        history = state.get("history", [])
        # Build a "freshness score" — prefer images not seen recently
        recent_seen = set()
        for h in history[-3:]:  # Last 3 rotations
            recent_seen.update(h.get("subset", []))
        candidates = [img for img in all_images if img not in recent_seen]
        if len(candidates) < subset_size:
            # Not enough fresh images — fallback to random
            candidates = all_images
        chosen = random.sample(candidates, subset_size)
        # Update state
        history.append({"ts": int(time.time()), "subset": chosen})
        if len(history) > 10:
            history = history[-10:]
        state["history"] = history
        state["current_subset"] = chosen
        state["last_rotation"] = int(time.time())
        save_state(val_pool_dir, state)

    return [os.path.join(val_pool_dir, f) for f in chosen]


def setup_validation_dir(source_pool_dir: str, target_val_dir: str,
                          subset_size: int = 7, strategy: str = "rotation") -> int:
    """
    Copy a rotated subset from source_pool to target_val (used by the engine).

    Returns the number of files copied.
    """
    import shutil
    if not os.path.isdir(source_pool_dir):
        return 0
    os.makedirs(target_val_dir, exist_ok=True)

    # Clear existing target (only image files, keep .state etc)
    for f in os.listdir(target_val_dir):
        fp = os.path.join(target_val_dir, f)
        if os.path.isfile(fp) and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            try:
                os.remove(fp)
            except Exception:
                pass

    # Select and copy
    chosen = select_subset(source_pool_dir, subset_size, strategy)
    count = 0
    for src in chosen:
        try:
            shutil.copy2(src, os.path.join(target_val_dir, os.path.basename(src)))
            count += 1
        except Exception:
            pass
    return count
