"""
tb_image_patch.py — Patch NeoSR/traiNNer-Redux to log validation images to TensorBoard.

Both engines write validation images to disk via cv2.imwrite() during
nondist_validation(). We inject a tb_logger.add_image() call right after.

Key design choices:
- IDEMPOTENT: Detected via marker comment "# PATCHED_BY_USR_STUDIO_TBIMG"
- REVERSIBLE: Can be unpatched cleanly
- SAFE: Subsamples max 4 images per validation to avoid log file bloat
- BACKUP: Saves original as .bak before first patch
"""
import os
import re
import shutil


PATCH_MARKER = "# PATCHED_BY_USR_STUDIO_TBIMG"
MAX_IMAGES_PER_VAL = 4  # Cap to avoid GB-sized .tfevents files


# Code injected after the cv2.imwrite call in nondist_validation
INJECTION_CODE = f'''
                    {PATCH_MARKER}_START
                    # Auto-injected by Universal SR Studio: log validation images to TensorBoard
                    try:
                        if hasattr(self, "tb_logger") and self.tb_logger is not None:
                            _img_idx = getattr(self, "_usr_img_count", 0)
                            if _img_idx < {MAX_IMAGES_PER_VAL}:
                                import numpy as _np
                                _arr = sr_img
                                if _arr.ndim == 3 and _arr.shape[2] == 3:
                                    _arr = _arr[:, :, ::-1]  # BGR -> RGB
                                _arr = _arr.astype("float32") / 255.0
                                _arr = _arr.transpose(2, 0, 1)  # HWC -> CHW
                                self.tb_logger.add_image(
                                    f"Validation/{{img_name}}", _arr, current_iter, dataformats="CHW"
                                )
                                self._usr_img_count = _img_idx + 1
                    except Exception as _e:
                        pass
                    {PATCH_MARKER}_END
'''


def find_validation_file(engine_root: str) -> str:
    """Find the model file containing nondist_validation in NeoSR/Redux."""
    if not os.path.isdir(engine_root):
        return ""
    candidates = [
        os.path.join(engine_root, "neosr", "models", "image.py"),      # NeoSR actual validation file
        os.path.join(engine_root, "neosr", "models", "default.py"),
        os.path.join(engine_root, "neosr", "models", "sr_model.py"),
        os.path.join(engine_root, "neosr", "models", "base_model.py"),
        os.path.join(engine_root, "traiNNer", "models", "sr_model.py"),
        os.path.join(engine_root, "traiNNer", "models", "default.py"),
        os.path.join(engine_root, "basicsr", "models", "sr_model.py"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    content = f.read()
                if "nondist_validation" in content and "imwrite" in content:
                    return c
            except Exception:
                pass
    # Fallback: walk the tree
    for root, dirs, files in os.walk(engine_root):
        # Skip caches and experiments
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "experiments", "datasets", ".git")]
        for f in files:
            if f.endswith(".py") and "model" in f:
                fp = os.path.join(root, f)
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        content = fh.read()
                    if "nondist_validation" in content and "imwrite" in content:
                        return fp
                except Exception:
                    pass
    return ""


def is_patched(file_path: str) -> bool:
    """Check if a file is already patched."""
    if not os.path.isfile(file_path):
        return False
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return PATCH_MARKER in f.read()
    except Exception:
        return False


def patch_file(file_path: str) -> tuple:
    """
    Inject TB image logging code into the validation file.

    Returns (success: bool, message: str).
    """
    if not os.path.isfile(file_path):
        return (False, f"Fichier non trouve: {file_path}")

    if is_patched(file_path):
        return (True, "Deja patche (idempotent).")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return (False, f"Erreur lecture: {e}")

    # Backup
    backup_path = file_path + ".usr_bak"
    if not os.path.exists(backup_path):
        try:
            shutil.copy2(file_path, backup_path)
        except Exception as e:
            return (False, f"Erreur backup: {e}")

    # Find the imwrite line(s) inside nondist_validation
    # Match: imwrite(sr_img, save_img_path) or cv2.imwrite(save_img_path, sr_img)
    # We want to inject AFTER the imwrite call, in the same block.
    # The function uses save_img_path as the target.

    # Reset _usr_img_count at the start of nondist_validation
    # We detect the def line and inject a counter init right after it.
    # Tricky part: the next line inside the function has its own indentation;
    # we must match the first inner line and inject BEFORE it (with same indent).
    nondist_pattern = re.compile(
        r"(def\s+nondist_validation\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*\n)"
        r"(\s+)"  # capture the inner indent of the first body line
        r"(?=\S)",  # next char must be non-whitespace (start of statement)
        re.MULTILINE
    )

    def _init_replacer(m):
        indent = m.group(2)
        return (
            m.group(1)
            + f"{indent}{PATCH_MARKER}_INIT\n"
            + f"{indent}self._usr_img_count = 0  # USR Studio: cap TB image logging\n"
            + indent
        )

    new_content = nondist_pattern.sub(_init_replacer, content, count=1)

    # Find the imwrite call containing sr_img and inject TB logging after it.
    # Use [^\n]* instead of [^)]* to handle nested parens like:
    #   NeoSR:  imwrite(sr_img, str(save_img_path))
    #   Redux:  imwrite(cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR), save_img_path)
    imwrite_pattern = re.compile(
        r"^(\s+)((?:cv2\.)?imwrite\s*\([^\n]*sr_img[^\n]*\))\s*$",
        re.MULTILINE
    )

    injected = False
    match = imwrite_pattern.search(new_content)
    if match:
        indent = match.group(1)
        call = match.group(2)
        # Redux passes sr_img through cv2.cvtColor(...RGB2BGR) — sr_img is already RGB, no flip needed.
        # NeoSR passes sr_img directly — sr_img is BGR, must flip to RGB for TensorBoard.
        sr_img_is_rgb = "cv2.cvtColor" in call and "RGB2BGR" in call
        flip_lines = (
            "" if sr_img_is_rgb else
            f"{indent}            if _arr.ndim == 3 and _arr.shape[2] == 3:\n"
            f"{indent}                _arr = _arr[:, :, ::-1]  # BGR -> RGB\n"
        )
        replacement = (
            f"{indent}{call}\n"
            f"{indent}{PATCH_MARKER}_START\n"
            f"{indent}# Auto-injected: TensorBoard image logging\n"
            f"{indent}try:\n"
            f"{indent}    if hasattr(self, 'tb_logger') and self.tb_logger is not None:\n"
            f"{indent}        _i = getattr(self, '_usr_img_count', 0)\n"
            f"{indent}        if _i < {MAX_IMAGES_PER_VAL}:\n"
            f"{indent}            import numpy as _np\n"
            f"{indent}            _arr = sr_img\n"
            + flip_lines +
            f"{indent}            _arr = (_arr.astype('float32') / 255.0).transpose(2, 0, 1)\n"
            f"{indent}            self.tb_logger.add_image(\n"
            f"{indent}                f'Validation/{{img_name}}', _arr, current_iter, dataformats='CHW'\n"
            f"{indent}            )\n"
            f"{indent}            self._usr_img_count = _i + 1\n"
            f"{indent}except Exception:\n"
            f"{indent}    pass\n"
            f"{indent}{PATCH_MARKER}_END"
        )
        new_content = new_content[:match.start()] + replacement + new_content[match.end():]
        injected = True

    if not injected:
        return (False, "imwrite(sr_img, ...) introuvable. Le code source a peut-etre change.")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return (False, f"Erreur ecriture: {e}")

    return (True, f"Patch applique. Backup: {backup_path}")


def unpatch_file(file_path: str) -> tuple:
    """Remove the patch by restoring the .usr_bak backup."""
    backup_path = file_path + ".usr_bak"
    if not os.path.isfile(backup_path):
        # Try in-place removal of the patch markers
        if not is_patched(file_path):
            return (True, "Pas patche, rien a faire.")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Remove init block
            content = re.sub(
                rf"\s*{re.escape(PATCH_MARKER)}_INIT.*?{re.escape(PATCH_MARKER)}_INIT\b.*?\n",
                "",
                content,
                flags=re.DOTALL,
            )
            # Remove main injection block
            content = re.sub(
                rf"\s*{re.escape(PATCH_MARKER)}_START.*?{re.escape(PATCH_MARKER)}_END\s*",
                "\n",
                content,
                flags=re.DOTALL,
            )
            # Remove _usr_img_count line
            content = re.sub(
                r"\s+self\._usr_img_count\s*=\s*0.*\n",
                "\n",
                content,
            )
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return (True, "Patch retire (sans backup).")
        except Exception as e:
            return (False, f"Erreur retrait: {e}")

    try:
        shutil.copy2(backup_path, file_path)
        return (True, "Patch retire (backup restaure).")
    except Exception as e:
        return (False, f"Erreur restauration: {e}")


def patch_engine(engine_root: str) -> tuple:
    """
    Convenience wrapper: find the validation file in an engine root and patch it.

    Returns (success: bool, message: str, file_path: str)
    """
    target = find_validation_file(engine_root)
    if not target:
        return (False, "Fichier de validation introuvable dans cet engine.", "")
    ok, msg = patch_file(target)
    return (ok, msg, target)


def get_patch_status(engine_root: str) -> dict:
    """Get patch status for an engine."""
    target = find_validation_file(engine_root)
    return {
        "engine_root": engine_root,
        "target_file": target,
        "found": bool(target),
        "patched": is_patched(target) if target else False,
        "backup_exists": os.path.isfile(target + ".usr_bak") if target else False,
    }
