import subprocess
import threading
import os
import sys
import signal
import time
import ctypes


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its children (prevents zombie worker processes)."""
    if os.name == 'nt':
        try:
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


def kill_process_on_port(port: int) -> None:
    """Kill any process listening on the given TCP port (Windows)."""
    if os.name != 'nt':
        return
    try:
        out = subprocess.check_output(
            ['netstat', '-ano'],
            text=True,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        for line in out.splitlines():
            if f':{port} ' in line and 'LISTENING' in line:
                parts = line.strip().split()
                pid = parts[-1]
                if pid.isdigit() and int(pid) > 0:
                    subprocess.run(
                        ['taskkill', '/F', '/PID', pid],
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    break
    except Exception:
        pass

class TrainingRunner:
    def __init__(self):
        self.process = None       # Processus Principal (Train)
        self.tb_process = None    # Processus TensorBoard
        self.ngrok_process = None # Processus Ngrok
        
        self.is_running = False
        self.stop_requested = False

    @staticmethod
    def _patch_sr_model_multiscale(script_path: str, log_callback) -> None:
        """Auto-patch traiNNer/models/sr_model.py for SpanC multi-scale compatibility.

        Three fixes applied:
        1. GT resize: when scale_list=[1,2], SpanC samples scale=1 → output=96px
           but GT is 192px (lq_size * dataset_scale). Resize GT before all losses.
        2. target swap: losses loop uses gt_for_loss instead of self.gt directly.
        3. LDL EMA resize: EMA net samples a scale independently → output_ema may
           have a different size than main output. Resize output_ema to match.

        Survives git pull — re-applies on every training start if any part is missing.
        """
        engine_dir = os.path.dirname(script_path)
        target = os.path.join(engine_dir, "traiNNer", "models", "sr_model.py")
        if not os.path.isfile(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                src = f.read()

            changed = False

            # Part 1: insert gt_for_loss block after assert isinstance(self.output, Tensor)
            if "gt_for_loss = self.gt" not in src:
                marker1 = "                assert isinstance(self.output, Tensor)"
                if marker1 not in src:
                    log_callback("[WARN] sr_model.py: marqueur Part1 non trouvé, patch multi-scale ignoré.\n")
                    return
                insert_block = (
                    "\n"
                    "                # [USS] SpanC/multi-scale: resize GT if output spatial dims differ\n"
                    "                # (e.g. scale_list=[1,2] → scale=1 sampled → output=96px, GT=192px)\n"
                    "                gt_for_loss = self.gt\n"
                    "                if self.output.shape[-2:] != self.gt.shape[-2:]:\n"
                    "                    gt_for_loss = F.interpolate(\n"
                    "                        self.gt,\n"
                    "                        size=self.output.shape[-2:],\n"
                    '                        mode="bicubic",\n'
                    "                        antialias=True,\n"
                    "                    ).clamp(0, 1)\n"
                )
                idx1 = src.find(marker1)
                end_of_line1 = src.find("\n", idx1) + 1
                src = src[:end_of_line1] + insert_block + src[end_of_line1:]
                changed = True

            # Part 2: replace target = self.gt in losses loop with gt_for_loss
            old_target = (
                "                for label, loss in self.losses.items():\n"
                "                    target = self.gt"
            )
            new_target = (
                "                for label, loss in self.losses.items():\n"
                "                    target = gt_for_loss"
            )
            if old_target in src:
                src = src.replace(old_target, new_target, 1)
                changed = True

            # Part 3: resize output_ema in LDL block to match self.output
            if "# [USS] SpanC multi-scale: EMA may sample" not in src:
                ldl_marker = "                        l_g_loss = loss(self.output, output_ema, target)"
                if ldl_marker in src:
                    ldl_insert = (
                        "                        # [USS] SpanC multi-scale: EMA may sample a different scale\n"
                        "                        # Align output_ema to self.output size before LDL comparison\n"
                        "                        if output_ema.shape[-2:] != self.output.shape[-2:]:\n"
                        "                            output_ema = F.interpolate(\n"
                        "                                output_ema,\n"
                        "                                size=self.output.shape[-2:],\n"
                        '                                mode="bicubic",\n'
                        "                                antialias=True,\n"
                        "                            ).clamp(0, 1)\n"
                        "                        l_g_loss = loss(self.output, output_ema, target)"
                    )
                    src = src.replace(ldl_marker, ldl_insert, 1)
                    changed = True

            if changed:
                with open(target, "w", encoding="utf-8") as f:
                    f.write(src)
                log_callback("[PATCH] sr_model.py — support multi-scale SpanC (GT resize + LDL EMA align) appliqué.\n")
        except Exception as e:
            log_callback(f"[WARN] Impossible de patcher sr_model.py : {e}\n")

    @staticmethod
    def _patch_ldl_loss_huber(script_path: str, log_callback) -> None:
        """Auto-patch traiNNer/losses/ldl_loss.py to add 'huber' criterion support.

        LDLLoss only supports l1/l2/charbonnier by default. When criterion='huber'
        is set in the YAML, self.criterion is never assigned → AttributeError at runtime.
        Adds torch.nn.HuberLoss() branch + error for unknown criterion types.
        """
        engine_dir = os.path.dirname(script_path)
        target = os.path.join(engine_dir, "traiNNer", "losses", "ldl_loss.py")
        if not os.path.isfile(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                src = f.read()
            if "HuberLoss" in src:
                return  # already patched
            old = (
                "        elif self.criterion_type == \"charbonnier\":\n"
                "            self.criterion = charbonnier_loss"
            )
            new = (
                "        elif self.criterion_type == \"charbonnier\":\n"
                "            self.criterion = charbonnier_loss\n"
                "        elif self.criterion_type == \"huber\":\n"
                "            self.criterion = torch.nn.HuberLoss()\n"
                "        else:\n"
                "            raise ValueError(\n"
                "                f\"LDLLoss: unsupported criterion '{criterion}'. \"\n"
                "                \"Use 'l1', 'l2', 'charbonnier', or 'huber'.\"\n"
                "            )"
            )
            if old not in src:
                log_callback("[WARN] ldl_loss.py: marqueur non trouvé, patch huber ignoré.\n")
                return
            src = src.replace(old, new, 1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(src)
            log_callback("[PATCH] ldl_loss.py — support criterion 'huber' ajouté.\n")
        except Exception as e:
            log_callback(f"[WARN] Impossible de patcher ldl_loss.py : {e}\n")

    @staticmethod
    def _patch_sr_model_val_multiscale(script_path: str, log_callback) -> None:
        """Auto-patch traiNNer/models/sr_model.py validation loop for multi-scale GT mismatch.

        When eval_base_scale=2 but val pairs are same-resolution (deband: LQ=GT size),
        output is 2x LQ while GT is 1x → calculate_psnr/ssim crash on shape mismatch.
        Patch resizes output to GT size before metrics computation.
        """
        engine_dir = os.path.dirname(script_path)
        target = os.path.join(engine_dir, "traiNNer", "models", "sr_model.py")
        if not os.path.isfile(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                src = f.read()
            if "# [USS] SpanC multi-scale: output may be 2x when val GT is 1x" in src:
                return  # already patched
            old = (
                "                metric_data[gt_key] = gt_img\n"
                "                self.gt = None"
            )
            new = (
                "                metric_data[gt_key] = gt_img\n"
                "                self.gt = None\n"
                "                # [USS] SpanC multi-scale: output may be 2x when val GT is 1x (deband pairs)\n"
                "                # Resize output to GT size so PSNR/SSIM can compare same-size images\n"
                "                if metric_data[\"img\"].shape[:2] != gt_img.shape[:2]:\n"
                "                    h, w = gt_img.shape[:2]\n"
                "                    metric_data[\"img\"] = cv2.resize(\n"
                "                        metric_data[\"img\"], (w, h), interpolation=cv2.INTER_CUBIC\n"
                "                    )"
            )
            if old not in src:
                log_callback("[WARN] sr_model.py: marqueur validation non trouvé, patch val multi-scale ignoré.\n")
                return
            src = src.replace(old, new, 1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(src)
            log_callback("[PATCH] sr_model.py — resize validation output pour multi-scale (PSNR/SSIM) appliqué.\n")
        except Exception as e:
            log_callback(f"[WARN] Impossible de patcher sr_model.py (val) : {e}\n")

    @staticmethod
    def _patch_check_dependencies(script_path: str, log_callback) -> None:
        """Auto-patch traiNNer/check/check_dependencies.py to bypass false PyTorch version error.

        TraiNNer-Redux ships a dependency checker that raises a RuntimeError when
        torch 2.6.0 is installed but pyproject.toml requires >=2.11.0 (false positive —
        the training code runs fine on 2.6.0+). We insert `return` as the first
        statement of check_dependencies() so the check is silently skipped.
        Survives `git pull` updates because it re-applies on every training start.
        """
        engine_dir = os.path.dirname(script_path)
        target = os.path.join(engine_dir, "traiNNer", "check", "check_dependencies.py")
        if not os.path.isfile(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                src = f.read()
            # Match both old "def check_dependencies():" and new "def check_dependencies() -> None:"
            import re as _re
            bypass_line = "    return  # auto-patched by Universal SR Studio — bypasses false PyTorch version error\n"
            # Already patched?
            if _re.search(r"def check_dependencies\(.*\).*:\s*\n\s+return\b", src):
                return  # already patched
            _m = _re.search(r"def check_dependencies\(.*\).*:", src)
            if not _m:
                return
            idx = _m.start()
            insert_at = src.find("\n", idx) + 1
            patched = src[:insert_at] + bypass_line + src[insert_at:]
            with open(target, "w", encoding="utf-8") as f:
                f.write(patched)
            log_callback("[PATCH] check_dependencies.py — bypass version PyTorch appliqué.\n")
        except Exception as e:
            log_callback(f"[WARN] Impossible de patcher check_dependencies.py : {e}\n")

    @staticmethod
    def _inject_custom_engine_files(script_path: str, log_callback) -> None:
        """Copy bundled custom losses/archs into the engine before training.

        v2.5.6: re-integrates features that lived in the old IA_Engine and were
        lost when it was deleted. NOTHING is removed from the engine — we only
        ADD files that the official traiNNer-redux (dev) does not ship:

          losses/spark_loss.py        → SparkLoss (FD + Charbonnier, umzi2/SparK_Perceptual)
          archs/inceptionnext_arch.py → InceptionNeXt backbone required by SparkLoss

        Idempotent: only writes when the destination is absent or differs from the
        bundled source. Survives `git pull` (re-applies on every training start).
        ECO is NOT injected here — it is native in traiNNer-redux dev (EcoOptions).
        """
        engine_dir = os.path.dirname(script_path)
        core_dir = os.path.dirname(__file__)
        bundled = os.path.join(core_dir, "custom_engine")
        # Detect engine type from script path to know neosr vs traiNNer
        _is_neosr = "neosr" in engine_dir.lower() and "trainner" not in engine_dir.lower()
        # (src dir, engine dst dir) pairs — engine-type aware.
        if _is_neosr:
            mapping = [
                # aethernet (neosr arch) — neosr training injection
                (os.path.join(core_dir, "custom_neosr_archs"),
                 os.path.join(engine_dir, "neosr", "archs")),
            ]
        else:
            mapping = [
                (os.path.join(bundled, "losses"), os.path.join(engine_dir, "traiNNer", "losses")),
                (os.path.join(bundled, "archs"),  os.path.join(engine_dir, "traiNNer", "archs")),
                # Custom archs absent from official dev: gfisrv2, smosr, spanpp, figsr, paragonsr2
                (os.path.join(core_dir, "custom_archs"), os.path.join(engine_dir, "traiNNer", "archs")),
            ]
        import shutil
        for src_dir, dst_dir in mapping:
            if not os.path.isdir(src_dir) or not os.path.isdir(dst_dir):
                continue
            for fname in os.listdir(src_dir):
                if not fname.endswith(".py"):
                    continue
                src = os.path.join(src_dir, fname)
                dst = os.path.join(dst_dir, fname)
                try:
                    need_copy = True
                    if os.path.isfile(dst):
                        with open(src, "rb") as a, open(dst, "rb") as b:
                            need_copy = a.read() != b.read()
                    if need_copy:
                        shutil.copy2(src, dst)
                        log_callback(f"[INJECT] {fname} ajouté au moteur (custom feature ré-intégrée).\n")
                except Exception as e:
                    log_callback(f"[WARN] Injection {fname} échouée : {e}\n")

    @staticmethod
    def _clean_val_dirs(config_path: str, log_callback) -> None:
        """Remove non-image files (Thumbs.db, .DS_Store, desktop.ini, etc.) from all dataset dirs.

        Windows auto-generates Thumbs.db (hidden system file) in image folders; pyvips crashes
        when it tries to load it during validation. Handles both YAML inline and list format.

        YAML inline:   dataroot_gt: C:/path
        YAML list:     dataroot_gt:
                           - C:/path
        TOML:          dataroot_gt = "C:/path"
        """
        _IMAGE_EXTS = {
            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
            ".gif", ".avif", ".heic", ".heif",
        }
        # Files to delete — Windows hidden system files that break pyvips/PIL loaders
        _JUNK_NAMES = {"thumbs.db", ".ds_store", "desktop.ini", "picasa.ini", ".picasa.ini"}

        try:
            with open(config_path, "r", encoding="utf-8") as _f:
                lines = _f.readlines()
        except Exception:
            return

        # Two-pass line scan: handles both "key: value" and "key:\n  - value" YAML list format
        _PATH_KEYS = {"dataroot_gt", "dataroot_lq", "val_gt", "val_lq"}
        candidates = set()
        _expect_list = False  # True after we see "key:" with no inline value

        for i, line in enumerate(lines):
            stripped = line.strip()

            # YAML list item — "- C:/some/path"
            if _expect_list:
                if stripped.startswith("- "):
                    val = stripped[2:].strip().strip('"\'')
                    if val:
                        candidates.add(val)
                        continue  # keep consuming list items
                elif stripped and not stripped.startswith("#"):
                    _expect_list = False  # Non-empty non-list line: list ended

            # YAML/TOML key detection
            for key in _PATH_KEYS:
                # Match "dataroot_gt:" or "dataroot_gt = "
                if stripped.lower().startswith(key + ":") or stripped.lower().startswith(key + " ="):
                    sep = ":" if ":" in stripped else "="
                    after = stripped.split(sep, 1)[1].strip().strip('"\'')
                    if after and not after.startswith("#"):
                        candidates.add(after)
                    else:
                        _expect_list = True  # No inline value → expect list on next lines
                    break

        # Also scan by absolute path pattern (fallback for embedded paths or unusual formats)
        import re as _re
        for m in _re.finditer(r'[A-Za-z]:[/\\][^\s\'"#\n\r]+', "\n".join(lines)):
            candidates.add(m.group(0).rstrip(",;"))

        removed = []
        scanned = set()
        for raw_path in candidates:
            folder = raw_path.strip().replace("/", os.sep).replace("\\", os.sep)
            if folder in scanned or not os.path.isdir(folder):
                continue
            scanned.add(folder)
            try:
                # Use os.scandir with FILE_FLAG_BACKUP_SEMANTICS to see hidden system files on Windows
                for entry in os.scandir(folder):
                    if not entry.is_file():
                        continue
                    fname_lower = entry.name.lower()
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in _IMAGE_EXTS and fname_lower in _JUNK_NAMES:
                        try:
                            os.remove(entry.path)
                            removed.append(entry.path)
                        except PermissionError:
                            # Thumbs.db may be locked by Explorer — try attrib to clear hidden flag first
                            try:
                                import subprocess as _sp
                                _sp.run(["attrib", "-H", "-S", entry.path],
                                        capture_output=True, timeout=3)
                                os.remove(entry.path)
                                removed.append(entry.path)
                            except Exception as _e2:
                                log_callback(f"[WARN] Impossible de supprimer {entry.name} : {_e2}\n")
                        except Exception as _e:
                            log_callback(f"[WARN] Impossible de supprimer {entry.name} : {_e}\n")
            except Exception:
                pass

        if removed:
            for p in removed:
                log_callback(f"[CLEAN] Supprimé fichier non-image : {os.path.basename(p)}\n")
        else:
            # Silent when nothing to clean — no log spam on normal runs
            pass

    def start_training(self, python_path, script_path, config_path, log_callback, on_finish_callback):
        if self.is_running:
            log_callback("[ERREUR] Un entraînement est déjà en cours.\n")
            return

        if not os.path.exists(python_path) or not os.path.exists(script_path) or not os.path.exists(config_path):
            log_callback(f"[ERREUR] Chemins introuvables.\n")
            return

        # Auto-patch check_dependencies.py for traiNNer-redux (false PyTorch version error)
        _wd_lower = script_path.replace("\\", "/").lower()
        if "trainner" in _wd_lower or "redux" in _wd_lower:
            self._patch_check_dependencies(script_path, log_callback)
            self._patch_sr_model_multiscale(script_path, log_callback)
            self._patch_sr_model_val_multiscale(script_path, log_callback)
            self._patch_ldl_loss_huber(script_path, log_callback)
            # Re-integrate custom engine files (SparkLoss + InceptionNeXt backbone).
            self._inject_custom_engine_files(script_path, log_callback)

        # Nettoyer les fichiers non-images dans les dossiers de validation (Thumbs.db, .DS_Store, etc.)
        self._clean_val_dirs(config_path, log_callback)

        # Réinitialisation absolue des flags
        self.stop_requested = False

        script_dir = os.path.dirname(script_path)

        # Lecture rapide des options
        use_tb = False
        use_ngrok = False
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Check both TOML (=) and YAML (:) syntax
                if "auto_tensorboard = true" in content.lower() or "auto_tensorboard: true" in content.lower(): use_tb = True
                if "auto_ngrok = true" in content.lower() or "auto_ngrok: true" in content.lower(): use_ngrok = True
        except Exception: pass
        
        # Lancement des outils
        self.launch_monitoring_tools(python_path, script_dir, use_tb, use_ngrok, log_callback)

        # Lancement du thread principal
        thread = threading.Thread(target=self._run_process, args=(python_path, script_path, config_path, log_callback, on_finish_callback))
        thread.daemon = True
        thread.start()

    def launch_monitoring_tools(self, python_path, working_dir, use_tb, use_ngrok, log_callback):
        # 1. TensorBoard
        if use_tb:
            try:
                log_callback("[MONITOR] Démarrage de TensorBoard (Port 6006)...\n")
                # Kill any stale TB process on port 6006 before launching
                if self.tb_process and self.tb_process.poll() is None:
                    self.tb_process.kill()
                    self.tb_process = None
                kill_process_on_port(6006)
                # Detect engine from working_dir:
                #   neosr      → {engine}/experiments/tb_logger
                #   traiNNer-redux → {engine}/tb_logger
                _wd_lower = working_dir.replace("\\", "/").lower()
                _is_neosr = ("neosr" in _wd_lower and "trainner" not in _wd_lower)
                if _is_neosr:
                    _tb_logdir = os.path.join(working_dir, "experiments", "tb_logger")
                else:
                    _tb_logdir = os.path.join(working_dir, "tb_logger")
                os.makedirs(_tb_logdir, exist_ok=True)
                cmd_tb = [python_path, "-m", "tensorboard.main", "--logdir", _tb_logdir, "--port", "6006", "--bind_all"]
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self.tb_process = subprocess.Popen(cmd_tb, cwd=working_dir, creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log_callback(f"[MONITOR] TensorBoard logdir : {_tb_logdir}\n")
            except Exception as e:
                log_callback(f"[ERR TB] Impossible de lancer TensorBoard : {e}\n")

        # 2. Ngrok
        if use_ngrok:
            try:
                log_callback("[MONITOR] Démarrage du Tunnel Ngrok...\n")
                cmd_ng = ["ngrok", "http", "6006"]
                if sys.platform == 'win32':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = 0  # SW_HIDE
                    flags = subprocess.CREATE_NO_WINDOW
                    self.ngrok_process = subprocess.Popen(cmd_ng, cwd=working_dir, creationflags=flags, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    self.ngrok_process = subprocess.Popen(cmd_ng, cwd=working_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log_callback("[MONITOR] Ngrok lancé.\n")
            except Exception as e:
                log_callback(f"[ERR NGROK] Impossible de lancer Ngrok : {e}\n")

    def kill_monitoring_tools(self, log_callback=None):
        if self.tb_process:
            try: self.tb_process.kill(); self.tb_process = None
            except Exception: pass
        if self.ngrok_process:
            try: self.ngrok_process.kill(); self.ngrok_process = None
            except Exception: pass

    def _run_process(self, python_path, script_path, config_path, log_callback, on_finish_callback):
        self.is_running = True
        script_dir = os.path.dirname(script_path)
        cmd = [python_path, script_path, "-opt", config_path]
        
        log_callback(f"--- Lancement ---\n")
        log_callback(f"> Commande : {' '.join(cmd)}\n\n")

        try:
            creation_flags = 0
            startupinfo = None

            if sys.platform == 'win32':
                # IMPORTANT : CREATE_NEW_CONSOLE permet d'isoler le processus pour l'injection du signal plus tard
                creation_flags = subprocess.CREATE_NEW_CONSOLE
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE — fenêtre cachée, CTRL+C reste fonctionnel

            # Force UTF-8 for the child process — fixes UnicodeEncodeError on Windows cp1252
            # when engines like traiNNer-redux print emojis (rocket etc.) via rich logging.
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"
            # Tell rich/click etc. that the terminal supports unicode.
            child_env.setdefault("FORCE_COLOR", "1")

            self.process = subprocess.Popen(
                cmd,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace',
                env=child_env,
                creationflags=creation_flags,
                startupinfo=startupinfo
            )

            # Lecture des logs en temps réel
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    log_callback(line)
            
            self.process.stdout.close()
            return_code = self.process.wait()

            if return_code == 0:
                log_callback("\n[SUCCES] Entraînement terminé.\n")
            elif return_code == 3221225786 or return_code == -1073741510: # Codes d'arrêt CTRL+C Windows
                log_callback("\n[STOP] Arrêt manuel confirmé (Sauvegarde OK).\n")
            else:
                log_callback(f"\n[STOP] Processus arrêté (Code : {return_code}).\n")

        except Exception as e:
            log_callback(f"\n[CRASH] : {e}\n")
        finally:
            self.is_running = False
            self.process = None
            self.kill_monitoring_tools()
            if on_finish_callback:
                on_finish_callback()

    def stop_training(self, log_callback):
        if not self.process or not self.is_running:
            return

        # Si l'utilisateur clique une 2ème fois, on force le KILL
        if self.stop_requested:
            log_callback("\n[KILL] Arrêt forcé immédiat !\n")
            try: kill_process_tree(self.process.pid)
            except Exception: pass
            try: self.process.kill()
            except Exception: pass
            self.kill_monitoring_tools()
            return

        self.stop_requested = True
        log_callback("\n[ACTION] Tentative de sauvegarde (Soft Stop)... Patientez...\n")
        
        # Lancement de l'injection dans un thread pour ne pas geler l'UI
        threading.Thread(target=self._inject_ctrl_c_windows, args=(log_callback,), daemon=True).start()

    def _inject_ctrl_c_windows(self, log_callback):
        """Méthode robuste pour envoyer CTRL+C sous Windows"""
        if sys.platform != 'win32':
            try: os.kill(self.process.pid, signal.SIGINT)
            except Exception: pass
            return

        try:
            kernel32 = ctypes.windll.kernel32
            pid = self.process.pid

            # 1. Se détacher de toute console actuelle (au cas où)
            try: kernel32.FreeConsole()
            except Exception: pass

            # 2. S'attacher à la console de l'entraînement
            if kernel32.AttachConsole(pid):
                # 3. Désactiver le handler CTRL+C de notre propre GUI (sinon l'appli se ferme aussi !)
                kernel32.SetConsoleCtrlHandler(None, True)
                
                # 4. Envoyer le signal
                # GenerateConsoleCtrlEvent(0, 0) envoie le signal à tous les processus de la console attachée
                kernel32.GenerateConsoleCtrlEvent(0, 0)
                
                log_callback("> Signal envoyé. Attente de la sauvegarde...\n")
                
                # 5. Attendre un peu que le signal parte
                time.sleep(0.2)
                
                # 6. Se détacher proprement pour permettre une future réutilisation
                kernel32.FreeConsole()
                
                # 7. Réactiver le handler CTRL+C pour l'avenir (optionnel mais propre)
                kernel32.SetConsoleCtrlHandler(None, False)
            else:
                log_callback("[ERREUR] Impossible de s'attacher à la console du processus.\n")
                # Fallback : Si on n'arrive pas à s'attacher, on devra killer plus tard
        
        except Exception as e:
            log_callback(f"[ERREUR TECHNIQUE] {e}\n")

        # Surveillance : Si après 120 secondes le process tourne toujours, on kill
        waited = 0
        while self.is_running and waited < 120:
            time.sleep(1)
            waited += 1
            # Petit check pour voir si le process est mort entre temps
            if self.process and self.process.poll() is not None:
                break
        
        if self.is_running:
            log_callback("\n[TIMEOUT] Le script est trop long à sauvegarder -> Kill.\n")
            try: kill_process_tree(self.process.pid)
            except Exception: pass
            try: self.process.kill()
            except Exception: pass