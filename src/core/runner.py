import subprocess
import threading
import os
import sys
import signal
import time
import ctypes


def _t(fr: str, en: str) -> str:
    # UI QT (PySide6) : la langue vit dans qt_ui.i18n. Same pattern as compute_estimator._t().
    try:
        from qt_ui.i18n import get_lang
        if get_lang() == "en":
            return en
    except Exception:
        pass
    return fr


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
                    log_callback(_t("[WARN] sr_model.py: marqueur Part1 non trouvé, patch multi-scale ignoré.\n", "[WARN] sr_model.py: Part1 marker not found, multi-scale patch skipped.\n"))
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
                log_callback(_t("[PATCH] sr_model.py — support multi-scale SpanC (GT resize + LDL EMA align) appliqué.\n", "[PATCH] sr_model.py — multi-scale SpanC support (GT resize + LDL EMA align) applied.\n"))
        except Exception as e:
            log_callback(_t(f"[WARN] Impossible de patcher sr_model.py : {e}\n", f"[WARN] Could not patch sr_model.py: {e}\n"))

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
                log_callback(_t("[WARN] ldl_loss.py: marqueur non trouvé, patch huber ignoré.\n", "[WARN] ldl_loss.py: marker not found, huber patch skipped.\n"))
                return
            src = src.replace(old, new, 1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(src)
            log_callback(_t("[PATCH] ldl_loss.py — support criterion 'huber' ajouté.\n", "[PATCH] ldl_loss.py — 'huber' criterion support added.\n"))
        except Exception as e:
            log_callback(_t(f"[WARN] Impossible de patcher ldl_loss.py : {e}\n", f"[WARN] Could not patch ldl_loss.py: {e}\n"))

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
                log_callback(_t("[WARN] sr_model.py: marqueur validation non trouvé, patch val multi-scale ignoré.\n", "[WARN] sr_model.py: validation marker not found, val multi-scale patch skipped.\n"))
                return
            src = src.replace(old, new, 1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(src)
            log_callback(_t("[PATCH] sr_model.py — resize validation output pour multi-scale (PSNR/SSIM) appliqué.\n", "[PATCH] sr_model.py — resize validation output for multi-scale (PSNR/SSIM) applied.\n"))
        except Exception as e:
            log_callback(_t(f"[WARN] Impossible de patcher sr_model.py (val) : {e}\n", f"[WARN] Could not patch sr_model.py (val): {e}\n"))

    # Archs qui exigent triton (Flex attention) — triton est absent par défaut sous Windows.
    # archs whose default needs triton: ESCRealM ("Flex" attention), ParagonSR2 Ultimate
    # (compile_blocks=True -> torch.compile; GPUTooOldForTriton on Pascal)
    _TRITON_ARCHS = {"escrealm", "escrealm_xl", "paragonsr2_ultimate", "paragonsr2_ultimate_v2"}

    def _ensure_triton_if_needed(self, config_path, python_path, log_callback, run_config=None):
        """Lazy-install : si l'arch du config requiert triton (escrealm…) et qu'il est absent
        du venv, l'installe AVANT l'entraînement (wheel triton-windows matchée à la version
        torch). N'installe rien pour les 95% d'archs qui n'en ont pas besoin. Échec gracieux.

        ESCRealM : triton ne sert qu'à son attention "Flex" (défaut du moteur). Si triton
        est inutilisable (GPU antérieur à Turing, ex. Pascal / GTX 10xx, ou install ratée),
        la copie de travail passe en attn_type "SDPA" : mêmes poids, même calcul, aucune
        dépendance (testé : sortie identique à 3e-6 près, VRAM ÷ 1,8 vs "Naive")."""
        try:
            # NeoSR configs are TOML, not YAML — yaml.safe_load() on a TOML file threw
            # "expected '<document start>', but found '['" on every single NeoSR run
            # (harmless — caught below — but silently skipped the triton auto-install
            # check for 100% of NeoSR trainings, not just the 95% of archs that don't
            # need it) (2026-09-17 fix).
            ext = os.path.splitext(config_path)[1].lower()
            if ext == ".toml":
                # tomllib, not the third-party `toml` package — see _runner_auto_resume
                # for why (`toml`'s strict homogeneous-array reader chokes on ordinary
                # degradation ranges like [0.2, 3,]).
                import tomllib
                with open(config_path, "rb") as f:
                    cfg = tomllib.load(f) or {}
            else:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            arch = str((cfg.get("network_g") or {}).get("type", "")).lower()
            if arch not in self._TRITON_ARCHS:
                return
            if not (python_path and os.path.isfile(python_path)):
                return
            _ng = cfg.get("network_g") or {}
            if arch.startswith("escrealm"):
                if str(_ng.get("attn_type", "flex")).lower() != "flex":
                    return  # SDPA / Naive chosen explicitly: no triton needed
            elif str(_ng.get("compile_blocks", True)).lower() in ("false", "0"):
                return  # compile already off
            # triton (triton-windows) needs a Turing+ GPU (compute capability >= 7.5)
            _cc = subprocess.run([python_path, "-c",
                                  "import torch;c=torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0);print(c[0]*10+c[1])"],
                                 capture_output=True, text=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                _cc = int((_cc.stdout or "0").strip().splitlines()[-1])
            except Exception:
                _cc = 0
            if _cc < 75:
                self._escrealm_attn_fallback(run_config, arch, log_callback,
                                             _t(f"GPU sans support triton (compute {_cc / 10:.1f} < 7.5)",
                                                f"GPU without triton support (compute {_cc / 10:.1f} < 7.5)"))
                return
            # triton déjà présent dans le venv ?
            chk = subprocess.run([python_path, "-c", "import triton"],
                                 capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if chk.returncode == 0:
                return
            log_callback(_t(f"[DEP] L'architecture « {arch} » requiert triton (absent) — installation auto…\n", f"[DEP] Architecture “{arch}” requires triton (missing) — auto-installing…\n"))
            # contrainte de version selon torch (triton-windows suit les versions de triton)
            constraint = "triton-windows"
            try:
                tv = subprocess.run([python_path, "-c", "import torch;print(torch.__version__)"],
                                    capture_output=True, text=True,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                ver = (tv.stdout or "").strip().split("+")[0]
                mm = ".".join(ver.split(".")[:2])
                constraint = {"2.6": "triton-windows<3.3", "2.7": "triton-windows<3.4",
                              "2.8": "triton-windows<3.5"}.get(mm, "triton-windows")
            except Exception:
                pass
            r = subprocess.run([python_path, "-m", "pip", "install", constraint],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if r.returncode == 0:
                log_callback(_t("[DEP] ✅ triton installé — l'entraînement peut démarrer.\n", "[DEP] ✅ triton installed — training can start.\n"))
            else:
                tail = (r.stderr or r.stdout or "")[-300:]
                log_callback(_t(f"[DEP] ❌ Échec install triton.\n{tail}\n",
                                f"[DEP] ❌ triton install failed.\n{tail}\n"))
                self._escrealm_attn_fallback(run_config, arch, log_callback,
                                             _t("triton non installable", "triton could not be installed"))
        except Exception as e:
            log_callback(_t(f"[DEP] Vérification triton impossible : {e}\n", f"[DEP] Could not check triton: {e}\n"))

    @staticmethod
    def _escrealm_attn_fallback(run_config, arch, log_callback, reason) -> None:
        """Working copy only (the user's file is never touched): a triton-free setting
        with the same weights — ESCRealM attn_type "SDPA", ParagonSR2 Ultimate
        compile_blocks false (torch.compile only speeds the blocks up)."""
        if not run_config or not os.path.isfile(run_config):
            return
        try:
            import yaml
            with open(run_config, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if arch.startswith("escrealm"):
                data.setdefault("network_g", {})["attn_type"] = "SDPA"
                fr, en = ("attention « SDPA » au lieu de « Flex »", "\"SDPA\" attention instead of \"Flex\"")
            else:
                data.setdefault("network_g", {})["compile_blocks"] = False
                fr, en = ("blocs non compilés (torch.compile désactivé)", "blocks not compiled (torch.compile off)")
            with open(run_config, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            log_callback(_t(
                f"[DEP] {arch} : {reason} → {fr} (mêmes poids, pas de triton).\n",
                f"[DEP] {arch}: {reason} → {en} (same weights, no triton).\n"))
        except Exception as e:
            log_callback(_t(f"[DEP] Bascule SDPA impossible : {e}\n", f"[DEP] Could not switch to SDPA: {e}\n"))

    @staticmethod
    def _patch_rcan_inplace(script_path: str, log_callback) -> None:
        """rcan_arch.py uses in-place ops (`x *= self.rgb_range`, `res += x`, ReLU(inplace=True))
        that break autograd → 'a variable needed for gradient computation has been modified by
        an inplace operation' during backward. Rewrite them out-of-place (identical maths).
        Re-applied each training start (survives git pull)."""
        engine_dir = os.path.dirname(script_path)
        target = os.path.join(engine_dir, "traiNNer", "archs", "rcan_arch.py")
        if not os.path.isfile(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                src = f.read()
            orig = src
            src = src.replace("x *= self.rgb_range", "x = x * self.rgb_range")
            src = src.replace("res += x", "res = res + x")
            src = src.replace("inplace=True", "inplace=False")
            if src != orig:
                with open(target, "w", encoding="utf-8") as f:
                    f.write(src)
                log_callback(_t("[PATCH] rcan_arch.py — ops in-place → out-of-place (fix autograd).\n", "[PATCH] rcan_arch.py — in-place → out-of-place ops (autograd fix).\n"))
        except Exception as e:
            log_callback(_t(f"[WARN] Impossible de patcher rcan_arch.py : {e}\n", f"[WARN] Could not patch rcan_arch.py: {e}\n"))

    @staticmethod
    def _patch_copy_opt_file_encoding(script_path: str, log_callback) -> None:
        """traiNNer/utils/options.py copy_opt_file() reopens the config with `open(.., "r+")`
        (OS default encoding = cp1252 on Windows) → UnicodeDecodeError on any non-ASCII byte
        in the config (e.g. our section-header comments, or an accented experiment name).
        Force utf-8 so training never crashes on a non-ASCII config. Re-applied each start."""
        engine_dir = os.path.dirname(script_path)
        target = os.path.join(engine_dir, "traiNNer", "utils", "options.py")
        if not os.path.isfile(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                src = f.read()
            old = 'open(filename, "r+")'
            new = 'open(filename, "r+", encoding="utf-8", errors="replace")'
            if old not in src or new in src:
                return  # not found or already patched
            src = src.replace(old, new)
            with open(target, "w", encoding="utf-8") as f:
                f.write(src)
            log_callback(_t("[PATCH] options.py copy_opt_file — encodage utf-8 forcé (anti-crash cp1252).\n", "[PATCH] options.py copy_opt_file — utf-8 encoding forced (anti-crash cp1252).\n"))
        except Exception as e:
            log_callback(_t(f"[WARN] Impossible de patcher options.py : {e}\n", f"[WARN] Could not patch options.py: {e}\n"))

    @staticmethod
    def _patch_train_sigbreak(script_path: str, log_callback) -> None:
        """train.py only registers signal.signal(signal.SIGINT, ...) to trigger a
        graceful save-and-stop — SIGINT fires on CTRL_C_EVENT only. Our "soft stop"
        button sends CTRL_BREAK_EVENT (the only console event Windows lets a parent
        target at a *specific* child process — CTRL_C_EVENT can only be broadcast to
        every process sharing the caller's console, which isn't an option here since
        the child runs isolated in its own CREATE_NEW_PROCESS_GROUP). CTRL_BREAK_EVENT
        never reaches the SIGINT handler, so the sentinel/CTRL_BREAK soft-stop always
        timed out and fell back to a hard KILL. Fix: also handle signal.SIGBREAK
        (Windows-only, raised by CTRL_BREAK_EVENT) with the same graceful handler.
        Survives git pull — re-applies on every training start if missing.

        NOTE (kept, still useful as a fast-path): in real testing this alone was NOT
        enough — Windows console-control-event delivery to a CREATE_NO_WINDOW child
        proved unreliable (2+ full iterations still ran after the event was sent).
        See _patch_train_stop_sentinel below for the mechanism that is actually
        deterministic: it doesn't depend on the OS delivering any signal at all."""
        target = script_path
        if not os.path.isfile(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                src = f.read()
            marker = "signal.signal(signal.SIGINT, handle_keyboard_interrupt)"
            if marker not in src or "signal.SIGBREAK" in src:
                return  # not found or already patched
            insert = (
                marker + "\n"
                '    if hasattr(signal, "SIGBREAK"):  # [USS] Windows CTRL_BREAK_EVENT → same graceful stop\n'
                "        signal.signal(signal.SIGBREAK, handle_keyboard_interrupt)"
            )
            src = src.replace(marker, insert, 1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(src)
            log_callback(_t(
                "[PATCH] train.py — SIGBREAK relié à l'arrêt propre (fix Soft Stop qui ne sauvegardait jamais).\n",
                "[PATCH] train.py — SIGBREAK wired to the graceful stop (fixes Soft Stop never saving).\n"))
        except Exception as e:
            log_callback(_t(f"[WARN] Impossible de patcher train.py (SIGBREAK) : {e}\n", f"[WARN] Could not patch train.py (SIGBREAK): {e}\n"))

    @staticmethod
    def _patch_train_stop_sentinel(script_path: str, log_callback) -> None:
        """The real, deterministic fix for Soft Stop: make the training loop poll the
        USS_STOP_FILE sentinel (already written by _inject_ctrl_c_windows, already
        exported as an env var — the plumbing existed, nothing ever read it) instead
        of relying on a Windows console-control-event reaching the child process,
        which testing showed is NOT reliable for a CREATE_NO_WINDOW child (SIGBREAK
        patch above did not stop the loop within 2+ full iterations in practice).
        Checking a file's existence once per iteration is cheap and always works.
        Survives git pull — re-applies on every training start if missing."""
        target = script_path
        if not os.path.isfile(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                src = f.read()
            if "_uss_stop_file" in src:
                return  # already patched
            marker_a = "    interrupt_received = False"
            marker_b = (
                "            while train_data is not None:\n"
                "                data_timer.record()"
            )
            if marker_a not in src or marker_b not in src:
                log_callback(_t(
                    "[WARN] train.py: marqueurs sentinel non trouvés, patch Soft Stop ignoré.\n",
                    "[WARN] train.py: sentinel markers not found, Soft Stop patch skipped.\n"))
                return
            src = src.replace(
                marker_a,
                marker_a + "\n"
                "    # [USS] Deterministic stop: polled once per iteration below, independent\n"
                "    # of whether a Windows console-control-event actually reaches this process.\n"
                '    _uss_stop_file = os.environ.get("USS_STOP_FILE", "")',
                1,
            )
            src = src.replace(
                marker_b,
                "            while train_data is not None:\n"
                "                if _uss_stop_file and os.path.exists(_uss_stop_file):\n"
                "                    interrupt_received = True\n"
                "                data_timer.record()",
                1,
            )
            with open(target, "w", encoding="utf-8") as f:
                f.write(src)
            log_callback(_t(
                "[PATCH] train.py — poll du fichier stop à chaque itération (Soft Stop fiable, indépendant des signaux Windows).\n",
                "[PATCH] train.py — polls the stop file every iteration (reliable Soft Stop, independent of Windows signals).\n"))
        except Exception as e:
            log_callback(_t(f"[WARN] Impossible de patcher train.py (stop sentinel) : {e}\n", f"[WARN] Could not patch train.py (stop sentinel): {e}\n"))

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
            log_callback(_t("[PATCH] check_dependencies.py — bypass version PyTorch appliqué.\n", "[PATCH] check_dependencies.py — PyTorch version bypass applied.\n"))
        except Exception as e:
            log_callback(_t(f"[WARN] Impossible de patcher check_dependencies.py : {e}\n", f"[WARN] Could not patch check_dependencies.py: {e}\n"))

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
                # USRS texture metrics (tex/edge/flat MAE, HF corr/ratio) for validation
                (os.path.join(bundled, "metrics"), os.path.join(engine_dir, "traiNNer", "metrics")),
                # Custom archs absent from official dev: gfisrv2, smosr, spanpp, figsr, paragonsr2,
                # spectral_disc, nextsrgan_disc, projected_disc (GAN discriminators)
                (os.path.join(core_dir, "custom_archs"), os.path.join(engine_dir, "traiNNer", "archs")),
                # Held back from public builds until the author agrees (WireSR) — only
                # exists on the developer's machine.
                (os.path.join(core_dir, "custom_archs_pending"), os.path.join(engine_dir, "traiNNer", "archs")),
                # traiNNer/utils/ dependencies needed by some custom archs above (currently just
                # spectrum.py, used by spectral_disc_arch.py's spectrum_transform).
                (os.path.join(core_dir, "custom_utils"), os.path.join(engine_dir, "traiNNer", "utils")),
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
                        log_callback(_t(f"[INJECT] {fname} ajouté au moteur (custom feature ré-intégrée).\n", f"[INJECT] {fname} added to the engine (custom feature re-integrated).\n"))
                except Exception as e:
                    log_callback(_t(f"[WARN] Injection {fname} échouée : {e}\n", f"[WARN] {fname} injection failed: {e}\n"))

        # Small targeted patches to upstream engine files (bugs in files the official
        # engine ships, not whole custom files) — distinct from the whole-file
        # injection above. Self-heals across engine re-clones. See engine_patches.py.
        try:
            from src.core.engine_patches import apply_engine_patches
            apply_engine_patches(engine_dir, "neosr" if _is_neosr else "redux", log_callback)
        except Exception as e:
            log_callback(_t(f"[WARN] Application des patchs moteur échouée : {e}\n", f"[WARN] Engine patch application failed: {e}\n"))

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
                                log_callback(_t(f"[WARN] Impossible de supprimer {entry.name} : {_e2}\n", f"[WARN] Could not delete {entry.name}: {_e2}\n"))
                        except Exception as _e:
                            log_callback(_t(f"[WARN] Impossible de supprimer {entry.name} : {_e}\n", f"[WARN] Could not delete {entry.name}: {_e}\n"))
            except Exception:
                pass

        if removed:
            for p in removed:
                log_callback(_t(f"[CLEAN] Supprimé fichier non-image : {os.path.basename(p)}\n", f"[CLEAN] Removed non-image file: {os.path.basename(p)}\n"))
        else:
            # Silent when nothing to clean — no log spam on normal runs
            pass

    # USS-only top-level sections that traiNNer-redux's strict msgspec parser rejects.
    _USS_ONLY_KEYS = {
        "monitoring", "augmentation", "aug_prob",
        "use_moa", "moa_augs", "moa_probs", "moa_debug", "moa_debug_limit",
        "deg_phase2", "deg_phase2_ramp", "loss_phase2", "loss_phase2_ramp",
        "usrs_texture_metrics",
    }

    # Redux OTF options read in RealESRGANModel.feed_data, so they can change during
    # the run (phase degradations). Kernel shapes, sinc and final sinc are drawn in the
    # dataset workers: they stay on the PSNR profile.
    _PHASE_BASE_KEYS = (
        "thicklines_prob", "blur_prob", "resize_prob", "resize_range",
        "gaussian_noise_prob", "noise_range", "poisson_scale_range", "gray_noise_prob",
        "jpeg_prob", "jpeg_range",
        "blur_prob2", "resize_prob2", "resize_range2",
        "gaussian_noise_prob2", "noise_range2", "poisson_scale_range2", "gray_noise_prob2",
        "jpeg_prob2", "jpeg_range2",
    )

    def _prepare_phase_degradations(self, config_path, script_path, log_callback):
        """Phase degradations (redux): write _usr_studio_deg_phase.json from the config's
        deg_phase2 / deg_phase2_ramp (USS-only keys). The engine patch blends the PSNR
        profile into the GAN profile after the GAN start. No deg_phase2 = sidecar removed."""
        engine_dir = os.path.dirname(script_path)
        sidecar = os.path.join(engine_dir, "_usr_studio_deg_phase.json")
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        p2 = cfg.get("deg_phase2")
        if not isinstance(p2, dict) or not p2:
            if os.path.exists(sidecar):
                try:
                    os.remove(sidecar)
                except Exception:
                    pass
            return
        try:
            from src.core.otf_custom_degradations import (
                _ALL_CUSTOM_KEYS, install_patches_redux)
        except Exception:
            _ALL_CUSTOM_KEYS, install_patches_redux = [], None
        ramp = int(cfg.get("deg_phase2_ramp", 0) or 0)
        base, custom, ignored = {}, {}, []
        for k, v2 in p2.items():
            if k in self._PHASE_BASE_KEYS:
                v1 = cfg.get(k, v2)
                if v1 != v2:
                    base[k] = [v1, v2]
            elif k in _ALL_CUSTOM_KEYS:
                # A custom degradation off in the PSNR profile starts from prob 0.
                v1 = cfg.get(k, 0.0 if k.endswith("_prob") else v2)
                if v1 != v2:
                    custom[k] = [v1, v2]
            else:
                ignored.append(k)
        try:
            import json
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump({"ramp": ramp, "base": base, "custom": custom}, f, indent=2)
        except Exception as e:
            log_callback(_t(f"[Deg Phase] Écriture sidecar impossible : {e}\n",
                            f"[Deg Phase] Could not write sidecar: {e}\n"))
            return
        # A custom degradation used only in the GAN profile still needs the custom hook
        # and its sidecar (PSNR-profile values, i.e. prob 0).
        if custom and install_patches_redux is not None:
            sc = os.path.join(engine_dir, "_usr_studio_custom_deg.json")
            try:
                import json
                cur = {}
                if os.path.exists(sc):
                    with open(sc, "r", encoding="utf-8") as f:
                        cur = json.load(f) or {}
                for k, (v1, _v2) in custom.items():
                    cur.setdefault(k, v1)
                with open(sc, "w", encoding="utf-8") as f:
                    json.dump(cur, f, indent=2)
                install_patches_redux(engine_dir, True)
            except Exception as e:
                log_callback(_t(f"[Deg Phase] Hook custom non installé : {e}\n",
                                f"[Deg Phase] Custom hook not installed: {e}\n"))
        # The custom-deg (un)install may restore realesrgan_model.py from its backup:
        # re-apply the engine patches (idempotent) so the phase blend is present.
        try:
            from src.core.engine_patches import apply_engine_patches
            apply_engine_patches(engine_dir, "redux", log_callback)
        except Exception:
            pass
        ok = False
        try:
            with open(os.path.join(engine_dir, "traiNNer", "models", "realesrgan_model.py"),
                      "r", encoding="utf-8") as f:
                ok = "_uss_deg_phase" in f.read()
        except Exception:
            pass
        changed = sorted(base) + sorted(custom)
        lines = [_t(f"Transition : {ramp} it après le démarrage du GAN",
                    f"Transition: {ramp} it after the GAN start")]
        for k in changed:
            v1, v2 = (base.get(k) or custom.get(k))
            lines.append(f"{k}: {v1} → {v2}")
        if ignored:
            lines.append(_t(f"Ignorés (restent profil PSNR) : {', '.join(sorted(ignored))}",
                            f"Ignored (stay on the PSNR profile): {', '.join(sorted(ignored))}"))
        head = (_t("[Deg Phase] ✅ Dégradations par phase PSNR → GAN actives",
                   "[Deg Phase] ✅ Phase degradations PSNR → GAN active") if ok else
                _t("[Deg Phase] ⚠ Patch moteur absent : les dégradations resteront celles du profil PSNR",
                   "[Deg Phase] ⚠ Engine patch missing: degradations will stay on the PSNR profile"))
        log_callback(head + "\n" + "".join(f"[Deg Phase]    {l}\n" for l in lines))

    def _prepare_phase_losses(self, config_path, script_path, log_callback):
        """Phase losses (redux): write _usr_studio_loss_phase.json from the config's
        loss_phase2 ({loss type: [PSNR weight, GAN weight]}) / loss_phase2_ramp (USS-only
        keys). The engine patch blends the weights after the GAN start and skips a loss
        whose weight is 0. No loss_phase2 = sidecar removed."""
        engine_dir = os.path.dirname(script_path)
        sidecar = os.path.join(engine_dir, "_usr_studio_loss_phase.json")
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        p2 = cfg.get("loss_phase2")
        if not isinstance(p2, dict) or not p2:
            if os.path.exists(sidecar):
                try:
                    os.remove(sidecar)
                except Exception:
                    pass
            return
        ramp = int(cfg.get("loss_phase2_ramp", 0) or 0)
        weights = {}
        for t, ws in p2.items():
            try:
                w1, w2 = float(ws[0]), float(ws[1])
            except Exception:
                continue
            weights[str(t)] = [w1, w2]
        try:
            import json
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump({"ramp": ramp, "weights": weights}, f, indent=2)
        except Exception as e:
            log_callback(_t(f"[Loss Phase] Écriture sidecar impossible : {e}\n",
                            f"[Loss Phase] Could not write sidecar: {e}\n"))
            return
        ok = False
        try:
            with open(os.path.join(engine_dir, "traiNNer", "models", "sr_model.py"),
                      "r", encoding="utf-8") as f:
                ok = "_uss_phase_losses" in f.read()
        except Exception:
            pass
        lines = [_t(f"Transition : {ramp} it après le démarrage du GAN",
                    f"Transition: {ramp} it after the GAN start")]
        for t in sorted(weights):
            w1, w2 = weights[t]
            lines.append(f"{t}: {w1:g} → {w2:g}")
        head = (_t("[Loss Phase] ✅ Losses par phase PSNR → GAN actives",
                   "[Loss Phase] ✅ Phase losses PSNR → GAN active") if ok else
                _t("[Loss Phase] ⚠ Patch moteur absent : les losses resteront celles du profil PSNR",
                   "[Loss Phase] ⚠ Engine patch missing: losses will stay on the PSNR profile"))
        log_callback(head + "\n" + "".join(f"[Loss Phase]    {l}\n" for l in lines))

    def _make_working_config(self, config_path, log_callback, extra_strip_keys=None):
        """Return a stripped working copy of a redux YAML / neosr TOML config (USS-only
        sections + any extra_strip_keys removed) so the engine's strict parser doesn't
        reject unknown fields. The original file is never modified. Line-based strip
        preserves the rest of the file (formatting + comments) exactly. Returns the
        original path if nothing to strip."""
        ext = os.path.splitext(config_path)[1].lower()
        if ext not in (".yml", ".yaml", ".toml"):
            return config_path
        strip_keys = set(self._USS_ONLY_KEYS) | set(extra_strip_keys or ())
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            out = []
            skip = False
            if ext == ".toml":
                # TOML tables ([monitoring]) run until the next [section] header.
                for ln in lines:
                    stripped = ln.strip()
                    if stripped.startswith("[") and "]" in stripped:
                        header = stripped.strip("[]").split(".", 1)[0].strip()
                        skip = header in strip_keys
                    if not skip:
                        out.append(ln)
            else:
                for ln in lines:
                    # A top-level key starts at column 0 (no indent), isn't a comment, has ':'
                    if ln[:1] not in (" ", "\t", "#", "\n", "\r", "") and ":" in ln:
                        key = ln.split(":", 1)[0].strip()
                        skip = key in strip_keys
                    if not skip:
                        out.append(ln)
            # Always run on a working copy so later patches (resume_state, scheduler)
            # never touch the user's original file on disk.
            base, e = os.path.splitext(config_path)
            tmp = base + "._uss_tmp" + e
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(out)
            self._tmp_config = tmp
            if len(out) != len(lines):
                log_callback(_t(
                    "[CONFIG] Section(s) USS (monitoring…) retirée(s) pour le moteur "
                    "(copie de travail, fichier original intact).\n",
                    "[CONFIG] USS section(s) (monitoring…) stripped for the engine "
                    "(working copy, original file untouched).\n"))
            return tmp
        except Exception as e:
            log_callback(_t(f"[CONFIG] Strip impossible, lancement sur l'original : {e}\n", f"[CONFIG] Strip failed, launching on the original: {e}\n"))
            return config_path

    def _runner_texture_metrics(self, config_path, run_config, script_path, log_callback):
        """Add the USRS texture metrics (tex/edge/flat MAE, HF corr/ratio) to val.metrics
        of the engine working copy, unless the config sets usrs_texture_metrics: false.
        Only when validation metrics are on and the metric module is in the engine.
        Returns the config path to launch (a working copy is created if needed: the
        original file is never modified)."""
        try:
            import yaml
            from src.core import val_extra_metrics as vem
            with open(config_path, "r", encoding="utf-8") as f:
                orig = yaml.safe_load(f) or {}
            if str(orig.get(vem.CONFIG_KEY, True)).lower() in ("false", "0", "no"):
                return run_config
            with open(run_config, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            val = data.get("val")
            if not isinstance(val, dict) or not val.get("val_enabled", True) \
                    or not val.get("metrics_enabled", True) or not isinstance(val.get("metrics"), dict):
                return run_config
            mod = os.path.join(os.path.dirname(os.path.abspath(script_path)),
                               "traiNNer", "metrics", "usrs_texture_metrics.py")
            if not os.path.isfile(mod):
                return run_config
            missing = {k: v for k, v in vem.engine_metrics().items() if k not in val["metrics"]}
            if not missing:
                return run_config
            val["metrics"].update(missing)
            if os.path.abspath(run_config) == os.path.abspath(config_path):
                base, ext = os.path.splitext(config_path)
                run_config = base + "._uss_tmp" + ext
                self._tmp_config = run_config
            with open(run_config, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            log_callback(_t("[Métriques] Texture / contours / grain / détail HF / netteté HF ajoutées à la validation.\n",
                            "[Metrics] Texture / edges / grain / HF detail / HF sharpness added to the validation.\n"))
        except Exception as e:
            log_callback(_t(f"[Métriques] Métriques texture non ajoutées : {e}\n",
                            f"[Metrics] Texture metrics not added: {e}\n"))
        return run_config

    @staticmethod
    def _attention_dims_error(cfg_path):
        """Attention archs split the channel dim across heads: dim % heads must be 0.
        The dynamic dim slider has a step of 1 (to keep each arch's exact default), so a
        value like MambaIRv2 embed_dim 30 with num_heads 4 used to crash the engine
        ~1 min after launch ("shape '[48, 256, 3, 4, 7]' is invalid"). Returns a message
        (FR, EN) or None. Reads YAML (Redux) or TOML (NeoSR); any read error -> None."""
        try:
            if str(cfg_path).lower().endswith(".toml"):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    txt = f.read()
                try:
                    import tomllib
                    data = tomllib.loads(txt)
                except ImportError:
                    import toml  # type: ignore
                    data = toml.loads(txt)
            else:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            net = data.get("network_g") or {}
            if not isinstance(net, dict):
                return None
            dim_key = next((k for k in ("embed_dim", "dim", "embed_dims") if k in net), None)
            head_key = next((k for k in ("num_heads", "heads") if k in net), None)
            if not dim_key or not head_key:
                return None
            dims, heads = net[dim_key], net[head_key]
            heads = list(heads) if isinstance(heads, (list, tuple)) else [heads]
            dims = list(dims) if isinstance(dims, (list, tuple)) else [dims] * len(heads)
            for d, h in zip(dims, heads):
                d, h = int(d), int(h)
                if h > 0 and d % h:
                    lo, hi = d - d % h, d - d % h + h
                    fr_v = f"{lo} ou {hi}" if lo > 0 else str(hi)
                    en_v = f"{lo} or {hi}" if lo > 0 else str(hi)
                    arch = net.get("type", "?")
                    return (
                        f"[ERREUR] {arch} : {dim_key} = {d} n'est pas divisible par "
                        f"{head_key} = {h} (l'attention répartit les canaux entre les têtes).\n"
                        f"[ERREUR] Lancement annulé. Onglet Configuration → Architecture : mettez "
                        f"{dim_key} à {fr_v} (multiple de {h}), puis sauvegardez.\n",
                        f"[ERROR] {arch}: {dim_key} = {d} is not divisible by "
                        f"{head_key} = {h} (attention splits the channels across the heads).\n"
                        f"[ERROR] Launch cancelled. Configuration tab → Architecture: set "
                        f"{dim_key} to {en_v} (a multiple of {h}), then save.\n")
        except Exception:
            return None
        return None

    def _runner_check_val_scale(self, cfg_path, log_callback):
        """Warn (never block) when the paired validation LQ is not GT / scale."""
        try:
            import yaml
            from src.core.dataset_tools import val_scale_mismatch
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            val = (data.get("datasets") or {}).get("val") or {}
            if not isinstance(val, dict) or not val.get("dataroot_lq"):
                return
            scale = int(data.get("scale", 1) or 1)
            bad = val_scale_mismatch(val.get("dataroot_gt"), val.get("dataroot_lq"), scale)
            if not bad:
                return
            name, gt, lq = bad[0]
            log_callback(_t(
                f"[WARN] Validation : les LQ ne sont pas les GT réduites ×{scale} "
                f"(ex. {name} : GT {gt[0]}×{gt[1]}, LQ {lq[0]}×{lq[1]}).\n"
                f"[WARN] La sortie est remise à la taille de la GT avant le calcul : PSNR/SSIM "
                f"et démarrage auto du GAN ne mesurent plus l'upscale ×{scale}.\n"
                f"[WARN] Générez les LQ avec Outils → Générateur LQ (réduction ×{scale}).\n",
                f"[WARN] Validation: the LQ images are not the GT downscaled ×{scale} "
                f"(e.g. {name}: GT {gt[0]}×{gt[1]}, LQ {lq[0]}×{lq[1]}).\n"
                f"[WARN] The output is resized to the GT size before scoring: PSNR/SSIM "
                f"and the automatic GAN start no longer measure the ×{scale} upscale.\n"
                f"[WARN] Generate the LQ with Tools → LQ Generator (×{scale} downscale).\n"))
        except Exception:
            pass

    def _runner_sanity_scheduler(self, cfg_path, log_callback):
        """Strip scheduler keys incompatible with the chosen scheduler type
        (e.g. leftover 'milestones' on a CosineAnnealingLR). Writes to the working copy."""
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            train = data.get("train")
            if not isinstance(train, dict):
                return
            sched = train.get("scheduler")
            if not isinstance(sched, dict):
                return
            st = str(sched.get("type", "")).lower()
            if not st:
                return
            allowed = {"type"}
            if "multistep" in st:
                allowed |= {"milestones", "gamma", "last_epoch", "verbose"}
            elif "cosineannealingwarmrestarts" in st or ("cosine" in st and "warm" in st):
                allowed |= {"T_0", "T_mult", "eta_min", "last_epoch", "verbose", "periods", "restart_weights"}
            elif "cosine" in st:
                allowed |= {"T_max", "eta_min", "last_epoch", "verbose"}
            elif st.endswith("step") or st == "steplr":
                allowed |= {"step_size", "gamma", "last_epoch", "verbose"}
            elif "linear" in st:
                allowed |= {"start_factor", "end_factor", "total_iters", "last_epoch", "verbose"}
            elif "exponential" in st:
                allowed |= {"gamma", "last_epoch", "verbose"}
            elif "cyclic" in st:
                allowed |= {"base_lr", "max_lr", "step_size_up", "step_size_down", "mode", "gamma", "last_epoch", "verbose"}
            else:
                return  # unknown/custom scheduler — leave it
            bad = [k for k in sched if k not in allowed]
            if not bad:
                return
            for k in bad:
                del sched[k]
            if "cosine" in st and "warm" not in st and "T_max" not in sched:
                sched["T_max"] = int(train.get("total_iter") or data.get("total_iter") or 100000)
            if "cosine" in st and "eta_min" not in sched:
                sched["eta_min"] = 1e-7
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            log_callback(_t(f"[Sanity] Scheduler '{sched.get('type')}' nettoyé : {bad}\n", f"[Sanity] Scheduler '{sched.get('type')}' cleaned: {bad}\n"))
        except Exception as e:
            log_callback(_t(f"[Sanity] Avertissement : {e}\n", f"[Sanity] Warning: {e}\n"))

    def _runner_auto_resume(self, cfg_path, script_path, log_callback):
        """Detect the latest .state in experiments/<name>/training_states and write
        path.resume_state into the working copy so a crashed run resumes.
        Format-aware (.yml → Redux, .toml → NeoSR) — this used to hardcode
        yaml.safe_load/dump, so calling it on a NeoSR .toml either raised (silently
        caught as a warning) or, worse, could corrupt the file; it was never actually
        reachable for NeoSR anyway since the old caller gated it behind a Redux-only
        check (2026-09-17 fix, see the caller)."""
        try:
            import re as _re
            _is_toml = cfg_path.lower().endswith(".toml")
            if _is_toml:
                # tomllib (stdlib, read-only) — NOT the third-party `toml` package here:
                # `toml` enforces TOML-pre-1.0's homogeneous-array rule and raises "Not a
                # homogeneous array" on totally ordinary configs like
                # `blur_sigma = [0.2, 3,]` (float min, int max — every OTF degradation
                # block has several of these), which silently killed auto-resume on any
                # real NeoSR config, not just an edge case (2026-09-17 fix). `toml.dump`
                # for writing below is unaffected — only its reader is overly strict.
                import tomllib
                with open(cfg_path, "rb") as f:
                    data = tomllib.load(f)
            else:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            name = data.get("name", "")
            if not name:
                return
            engine_root = os.path.dirname(script_path)
            state_dir = os.path.join(engine_root, "experiments", name, "training_states")
            if not os.path.isdir(state_dir):
                return
            def _it(n):
                m = _re.search(r"(\d+)", os.path.splitext(n)[0])
                return int(m.group(1)) if m else 0
            states = sorted([f for f in os.listdir(state_dir) if f.endswith(".state")], key=_it, reverse=True)
            if not states:
                return
            # A save interrupted by a full disk leaves a truncated .state (and/or missing
            # resume models): resuming from it crashes with "failed finding central
            # directory" on every relaunch. Take the newest state that is readable AND
            # has its resume models next to it; report the skipped ones.
            exp_dir = os.path.dirname(state_dir)
            needs_d = bool(data.get("network_d"))
            latest, skipped = None, []
            for st in states:
                reason = self._resume_state_problem(os.path.join(state_dir, st), exp_dir,
                                                    _it(st), needs_d, _is_toml)
                if reason is None:
                    latest = os.path.join(state_dir, st).replace("\\", "/")
                    break
                skipped.append(f"{st} ({reason})")
            for s in skipped:
                log_callback(_t(f"[Auto-Resume] Ignoré : {s}\n", f"[Auto-Resume] Skipped: {s}\n"))
            if latest is None:
                log_callback(_t("[Auto-Resume] Aucun état de reprise complet : démarrage sans reprise.\n",
                                "[Auto-Resume] No complete resume state: starting without resume.\n"))
                return
            if not isinstance(data.get("path"), dict):
                data["path"] = {}
            data["path"]["resume_state"] = latest
            if _is_toml:
                import toml
                with open(cfg_path, "w", encoding="utf-8") as f:
                    toml.dump(data, f)
            else:
                import yaml
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            log_callback(_t(f"[Auto-Resume] Reprise depuis : {latest}\n", f"[Auto-Resume] Resuming from: {latest}\n"))
        except Exception as e:
            log_callback(_t(f"[Auto-Resume] Avertissement : {e}\n", f"[Auto-Resume] Warning: {e}\n"))

    @staticmethod
    def _resume_state_problem(state_path, exp_dir, it, needs_d, is_toml):
        """None if the state can be resumed from, else a short reason."""
        import zipfile
        try:
            if os.path.getsize(state_path) == 0:
                return _t("fichier vide", "empty file")
            if not zipfile.is_zipfile(state_path):
                return _t("fichier tronqué / illisible", "truncated / unreadable file")
            with zipfile.ZipFile(state_path) as z:
                if z.testzip() is not None:
                    return _t("archive corrompue", "corrupted archive")
        except Exception as e:
            return str(e)
        if is_toml:
            return None  # NeoSR: model naming differs, the state check is enough
        rm = os.path.join(exp_dir, "models", "resume_models")
        if not os.path.isdir(rm):
            return None  # older layout: let the engine decide
        def _has(prefix):
            return any(os.path.isfile(os.path.join(rm, f"{prefix}_{it}{ext}")) and
                       os.path.getsize(os.path.join(rm, f"{prefix}_{it}{ext}")) > 0
                       for ext in (".safetensors", ".pth"))
        if not _has("net_g"):
            return _t(f"net_g_{it} manquant", f"net_g_{it} missing")
        if needs_d and not _has("net_d"):
            return _t(f"net_d_{it} manquant", f"net_d_{it} missing")
        return None

    def _runner_pre_archive(self, cfg_path, script_path, log_callback):
        """Pre-rename an existing experiments/<name> folder on a fresh start (no resume),
        owning the lock window so the engine doesn't fail on a folder Explorer/TB holds."""
        try:
            import yaml, datetime as _dt
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            name = data.get("name", "")
            if not name:
                return
            rs = (data.get("path", {}) or {}).get("resume_state")
            if rs and str(rs).strip() not in ("", "~", "null", "None"):
                return  # resuming — engine keeps the folder
            engine_root = os.path.dirname(script_path)
            exp = os.path.join(engine_root, "experiments", name)
            if not os.path.isdir(exp):
                return
            archived = exp + "_archived_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            for attempt in range(4):
                try:
                    os.rename(exp, archived)
                    log_callback(_t(f"[Pre-Archive] Dossier existant archivé → {os.path.basename(archived)}\n", f"[Pre-Archive] Existing folder archived → {os.path.basename(archived)}\n"))
                    return
                except PermissionError:
                    if attempt < 3:
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        log_callback(_t("[Pre-Archive] Dossier verrouillé — l'engine retentera le renommage.\n", "[Pre-Archive] Folder locked — the engine will retry the rename.\n"))
                except OSError as e:
                    log_callback(f"[Pre-Archive] {e}\n")
                    return
        except Exception as e:
            log_callback(_t(f"[Pre-Archive] Avertissement : {e}\n", f"[Pre-Archive] Warning: {e}\n"))

    def _prepare_custom_degradations(self, config_path, script_path, log_callback):
        """Faithfully replicate the old UI's custom-degradation pipeline, but sourced
        from the config YAML (which config_handler writes the custom keys into):
          1. read every USS custom degradation key present (top-level or under degradations:)
          2. write the side-channel JSON the engine patch reads
          3. install the OTF hook into the engine (install_patches)
        Returns the list of custom keys present (so the working copy can strip them —
        msgspec rejects them). Returns [] if no custom degradation is active."""
        try:
            import yaml
            from src.core.otf_custom_degradations import (
                _ALL_CUSTOM_KEYS, install_patches, install_patches_redux, REDUX_HOOK_MARKER)
        except Exception as e:
            log_callback(_t(f"[Custom Deg] Module indisponible : {e}\n", f"[Custom Deg] Module unavailable: {e}\n"))
            return []
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            log_callback(_t(f"[Custom Deg] Lecture config impossible : {e}\n", f"[Custom Deg] Could not read config: {e}\n"))
            return []

        deg = cfg.get("degradations", {}) if isinstance(cfg.get("degradations"), dict) else {}

        def gv(key):
            if key in cfg:
                return cfg[key]
            return deg.get(key)

        present = [k for k in _ALL_CUSTOM_KEYS if (k in cfg or k in deg)]
        if not present:
            # No custom degradations this run → remove any stale sidecar/hook from a
            # previous run so it never silently keeps applying custom degradations.
            try:
                from src.core.otf_custom_degradations import uninstall_patches_redux, uninstall_patches
                ed = os.path.dirname(script_path)
                sc = os.path.join(ed, "_usr_studio_custom_deg.json")
                if os.path.exists(sc):
                    os.remove(sc)
                if os.path.isdir(os.path.join(ed, "traiNNer", "models")):
                    uninstall_patches_redux(ed)
                else:
                    uninstall_patches(ed)
            except Exception:
                pass
            return []

        custom_params = {k: gv(k) for k in present}

        def _p(key):
            try:
                return float(custom_params.get(key, 0) or 0)
            except Exception:
                return 0.0

        has_posterize = _p("posterize_prob") > 0
        has_banding = _p("banding_prob") > 0
        has_chroma = _p("chroma_prob") > 0
        has_ca = _p("ca_prob") > 0
        has_halation = _p("halation_prob") > 0
        has_salt_pepper = _p("salt_pepper_prob") > 0
        has_vhs = _p("vhs_prob") > 0
        has_aliasing = _p("aliasing_prob") > 0
        has_interlace = any(_p(k) > 0 for k in
                            ("interlace_weave_prob", "interlace_flicker_prob", "interlace_blend_prob"))
        has_film_grain = _p("film_grain_prob") > 0
        has_oversharp = _p("oversharp_prob") > 0
        has_scanlines = _p("scanlines_prob") > 0
        has_extra = any(_p(k) > 0 for k in
                        ("screentone_prob", "dithering_prob", "pixelate_prob", "sin_prob",
                         "subsampling_prob", "color_level_prob", "wtp_halo_prob", "saturation_prob",
                         "shift_prob", "disc_blur_prob", "vignette_prob", "quantize_depth_prob",
                         "smooth_banding_prob",
                         "coupled_optical_prob", "coupled_vintage_prob"))

        engine_dir = os.path.dirname(script_path)
        # Side-channel JSON read by the injected OTF hook
        try:
            import json
            sidecar = os.path.join(engine_dir, "_usr_studio_custom_deg.json")
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(custom_params, f, indent=2)
        except Exception as e:
            log_callback(_t(f"[Custom Deg] Écriture sidecar impossible : {e}\n", f"[Custom Deg] Could not write sidecar: {e}\n"))

        active = [k.replace("_prob", "") for k in present if k.endswith("_prob") and _p(k) > 0]
        _is_redux = "trainner" in engine_dir.lower() or "redux" in engine_dir.lower() \
            or os.path.isdir(os.path.join(engine_dir, "traiNNer", "models"))
        hook_ok = False
        try:
            from src.core.otf_custom_degradations import NEOSR_HOOK_MARKER
            if _is_redux:
                # Redux: OTF degradation runs in RealESRGANModel.feed_data (GPU) → hook there.
                install_patches_redux(engine_dir, bool(active))
                target = os.path.join(engine_dir, "traiNNer", "models", "realesrgan_model.py")
                if os.path.isfile(target):
                    with open(target, "r", encoding="utf-8") as _f:
                        hook_ok = REDUX_HOOK_MARKER in _f.read()
            else:
                # NeoSR: OTF degradation runs in the dataset feed_data → hook there.
                install_patches(engine_dir, has_posterize, has_banding,
                                has_chroma=has_chroma, has_ca=has_ca,
                                has_halation=has_halation, has_salt_pepper=has_salt_pepper,
                                has_vhs=has_vhs, has_aliasing=has_aliasing,
                                has_interlace=has_interlace, has_film_grain=has_film_grain,
                                has_oversharp=has_oversharp, has_scanlines=has_scanlines,
                                has_extra=has_extra)
                data_dir = os.path.join(engine_dir, "data")
                for cand in ("otf_dataset.py", "realesrgan_dataset.py", "otf.py"):
                    fp = os.path.join(data_dir, cand)
                    if os.path.isfile(fp):
                        with open(fp, "r", encoding="utf-8") as _f:
                            hook_ok = NEOSR_HOOK_MARKER in _f.read()
                        break
        except Exception as e:
            log_callback(_t(f"[Custom Deg] Patch a échoué : {e}\n", f"[Custom Deg] Patch failed: {e}\n"))

        if active and hook_ok:
            where = "RealESRGANModel.feed_data (GPU)" if _is_redux else "dataset.feed_data"
            log_callback(_t(
                f"[Custom Deg] ✅ Hook OTF custom VÉRIFIÉ dans {where} — "
                f"appliqué à la volée à chaque image : {', '.join(active)}\n",
                f"[Custom Deg] ✅ Custom OTF hook VERIFIED in {where} — "
                f"applied on the fly to every image: {', '.join(active)}\n"))
            log_callback(_t(f"[Custom Deg]    Sidecar : {sidecar}\n", f"[Custom Deg]    Sidecar: {sidecar}\n"))
        elif active:
            log_callback(_t(
                f"[Custom Deg] ⚠ ATTENTION : dégradations custom détectées ({', '.join(active)}) "
                "mais le hook OTF n'a PAS pu être installé/vérifié → elles NE seront PAS appliquées. "
                "Préviens-moi pour corriger.\n",
                f"[Custom Deg] ⚠ WARNING: custom degradations detected ({', '.join(active)}) "
                "but the OTF hook could NOT be installed/verified → they will NOT be applied. "
                "Let me know so this can be fixed.\n"))

        return present

    def start_training(self, python_path, script_path, config_path, log_callback, on_finish_callback, auto_resume=False):
        if self.is_running:
            log_callback(_t("[ERREUR] Un entraînement est déjà en cours.\n", "[ERROR] A training run is already in progress.\n"))
            return

        if not (python_path and os.path.exists(python_path)) or not (script_path and os.path.exists(script_path)) \
                or not (config_path and os.path.exists(config_path)):
            missing = [n for n, p in (("Python", python_path), ("train.py", script_path), ("config", config_path))
                       if not (p and os.path.exists(p))]
            log_callback(_t(f"[ERREUR] Chemins introuvables : {', '.join(missing)}.\n",
                            f"[ERROR] Paths not found: {', '.join(missing)}.\n"))
            # report the failure: the Queue waits for this callback to start the next item
            try:
                on_finish_callback(1)
            except Exception:
                pass
            return

        _dim_err = self._attention_dims_error(config_path)
        if _dim_err:
            log_callback(_t(*_dim_err))
            try:
                on_finish_callback(1)
            except Exception:
                pass
            return

        # Auto-patch check_dependencies.py for traiNNer-redux (false PyTorch version error)
        _wd_lower = script_path.replace("\\", "/").lower()
        if "trainner" in _wd_lower or "redux" in _wd_lower:
            self._patch_check_dependencies(script_path, log_callback)
            self._patch_copy_opt_file_encoding(script_path, log_callback)
            self._patch_rcan_inplace(script_path, log_callback)
            self._patch_sr_model_multiscale(script_path, log_callback)
            self._patch_sr_model_val_multiscale(script_path, log_callback)
            self._patch_ldl_loss_huber(script_path, log_callback)
            self._patch_train_sigbreak(script_path, log_callback)
            self._patch_train_stop_sentinel(script_path, log_callback)
        # Re-integrate custom engine files. Engine-aware (cf. _inject_custom_engine_files) :
        # redux = SparkLoss + InceptionNeXt + archs custom (gfisrv2/smosr/spanpp/…) ;
        # neosr = AetherNet (aether_*). DOIT tourner aussi pour neosr, sinon les archs
        # aether_* ne sont jamais copiées → KeyError "aether_* not in arch registry".
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
        # Settings › Gallery automations (set by the Run tab before the launch)
        _mp = getattr(self, "monitor_prefs", None) or {}
        if _mp.get("auto_tb"):
            use_tb = True
        if _mp.get("ngrok_allowed") is False:
            if use_ngrok:
                log_callback(_t("[MONITOR] Ngrok demandé par la config mais désactivé dans Réglages › Galerie.\n",
                                "[MONITOR] Ngrok asked by the config but disabled in Settings › Gallery.\n"))
            use_ngrok = False
        
        # Lancement des outils
        self.launch_monitoring_tools(python_path, script_dir, use_tb, use_ngrok, log_callback, config_path=config_path)

        # traiNNer-redux uses strict msgspec parsing and rejects unknown top-level fields
        # (monitoring, augmentation, …). The original config keeps these USS-only sections;
        # we run on a stripped working copy so the original file on disk is never modified.
        run_config = config_path
        self._tmp_config = None
        if "trainner" in _wd_lower or "redux" in _wd_lower:
            # Wire custom degradations (sidecar + engine patch) and learn which custom
            # keys to also strip from the working copy (msgspec rejects them).
            custom_keys = self._prepare_custom_degradations(config_path, script_path, log_callback)
            self._prepare_phase_degradations(config_path, script_path, log_callback)
            self._prepare_phase_losses(config_path, script_path, log_callback)
            run_config = self._make_working_config(config_path, log_callback, extra_strip_keys=custom_keys)
            # Pre-launch config hygiene on the working copy (original never touched):
            self._runner_sanity_scheduler(run_config, log_callback)
            self._runner_check_val_scale(run_config, log_callback)
            run_config = self._runner_texture_metrics(config_path, run_config, script_path, log_callback)
            if auto_resume:
                self._runner_auto_resume(run_config, script_path, log_callback)
            self._runner_pre_archive(run_config, script_path, log_callback)
        else:
            # NeoSR path — custom-degradation injection is redux-specific, but the
            # USS-only [monitoring] table (added 2026-09-17 for TB port persistence)
            # still needs stripping before the engine sees the TOML, same reasoning as
            # the Redux msgspec strip above (unverified whether neosr's own parser is
            # equally strict, so strip defensively rather than risk a crash).
            run_config = self._make_working_config(config_path, log_callback)
            # Auto-resume itself has no msgspec/Redux dependency once it's format-aware.
            # It used to live only inside the Redux branch above, so the "Auto-resume"
            # checkbox silently did nothing on NeoSR (no error, no log line, just never
            # checked for a .state to resume from) (2026-09-17 fix).
            if auto_resume:
                self._runner_auto_resume(run_config, script_path, log_callback)

        # Lazy-install triton si l'arch le requiert (escrealm…) et qu'il manque
        self._ensure_triton_if_needed(config_path, python_path, log_callback, run_config)

        # Lancement du thread principal
        thread = threading.Thread(target=self._run_process, args=(python_path, script_path, run_config, log_callback, on_finish_callback))
        thread.daemon = True
        thread.start()

    def launch_monitoring_tools(self, python_path, working_dir, use_tb, use_ngrok, log_callback, config_path=None):
        # 1. TensorBoard
        if use_tb:
            try:
                log_callback(_t("[MONITOR] Démarrage de TensorBoard (Port 6006)...\n", "[MONITOR] Starting TensorBoard (Port 6006)...\n"))
                # Kill any stale TB process on port 6006 before launching
                if self.tb_process and self.tb_process.poll() is None:
                    self.tb_process.kill()
                    self.tb_process = None
                kill_process_on_port(6006)
                # Detect engine from working_dir:
                #   neosr      → {engine}/experiments/{exp_name}/tb_logger
                #   traiNNer-redux → {engine}/tb_logger
                # (was missing the {exp_name} level for NeoSR — computed
                # {engine}/experiments/tb_logger, an empty folder TensorBoard would
                # never see real data in, since NeoSR actually writes under
                # experiments/{exp_name}/tb_logger. Same path ServersDialog._logdir_map
                # already uses and was verified correct against. 2026-09-17 fix.)
                _wd_lower = working_dir.replace("\\", "/").lower()
                _is_neosr = ("neosr" in _wd_lower and "trainner" not in _wd_lower)
                if _is_neosr:
                    _exp_name = None
                    if config_path and os.path.isfile(config_path):
                        try:
                            with open(config_path, "r", encoding="utf-8") as _f:
                                _content = _f.read()
                            import re as _re_tb
                            _m = _re_tb.search(r'^\s*name\s*=\s*"([^"]+)"', _content, _re_tb.MULTILINE)
                            if not _m:
                                _m = _re_tb.search(r'^\s*name\s*:\s*(.+)$', _content, _re_tb.MULTILINE)
                            if _m:
                                _exp_name = _m.group(1).strip().strip('"').strip("'")
                        except Exception:
                            pass
                    if _exp_name:
                        # neosr/train.py: Path(root) / "experiments" / "tb_logger" / name
                        # (experiments/<name>/tb_logger was read before: always empty)
                        _tb_logdir = os.path.join(working_dir, "experiments", "tb_logger", _exp_name)
                    else:
                        # Name not found (shouldn't normally happen) — fall back to the
                        # old (wrong-but-non-crashing) path rather than guessing further.
                        _tb_logdir = os.path.join(working_dir, "experiments", "tb_logger")
                else:
                    _tb_logdir = os.path.join(working_dir, "tb_logger")
                os.makedirs(_tb_logdir, exist_ok=True)
                cmd_tb = [python_path, "-m", "tensorboard.main", "--logdir", _tb_logdir, "--port", "6006", "--bind_all"]
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self.tb_process = subprocess.Popen(cmd_tb, cwd=working_dir, creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if (getattr(self, "monitor_prefs", None) or {}).get("open_browser"):
                    def _open_tb_when_ready():
                        import urllib.request, webbrowser
                        for _ in range(60):  # up to ~60 s for TensorBoard to answer
                            try:
                                urllib.request.urlopen("http://localhost:6006", timeout=1)
                                webbrowser.open("http://localhost:6006"); return
                            except Exception:
                                time.sleep(1)
                    threading.Thread(target=_open_tb_when_ready, daemon=True).start()
                log_callback(_t(f"[MONITOR] TensorBoard logdir : {_tb_logdir}\n", f"[MONITOR] TensorBoard logdir: {_tb_logdir}\n"))
            except Exception as e:
                log_callback(_t(f"[ERR TB] Impossible de lancer TensorBoard : {e}\n", f"[ERR TB] Could not launch TensorBoard: {e}\n"))

        # 2. Ngrok
        if use_ngrok:
            try:
                log_callback(_t("[MONITOR] Démarrage du Tunnel Ngrok...\n", "[MONITOR] Starting Ngrok Tunnel...\n"))
                cmd_ng = ["ngrok", "http", "6006"]
                if sys.platform == 'win32':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = 0  # SW_HIDE
                    flags = subprocess.CREATE_NO_WINDOW
                    self.ngrok_process = subprocess.Popen(cmd_ng, cwd=working_dir, creationflags=flags, startupinfo=startupinfo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    self.ngrok_process = subprocess.Popen(cmd_ng, cwd=working_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log_callback(_t("[MONITOR] Ngrok lancé.\n", "[MONITOR] Ngrok launched.\n"))
            except Exception as e:
                log_callback(_t(f"[ERR NGROK] Impossible de lancer Ngrok : {e}\n", f"[ERR NGROK] Could not launch Ngrok: {e}\n"))

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
        # Code retour transmis à on_finish_callback en fin de run (0=succès).
        # 1 par défaut : si une exception coupe le lancement avant que le
        # process ne se termine normalement, on ne doit jamais rapporter un succès.
        return_code = 1
        
        log_callback(_t(f"--- Lancement ---\n", f"--- Launching ---\n"))
        log_callback(_t(f"> Commande : {' '.join(cmd)}\n\n", f"> Command: {' '.join(cmd)}\n\n"))

        try:
            creation_flags = 0
            startupinfo = None

            if sys.platform == 'win32':
                # CREATE_NEW_PROCESS_GROUP : isole le processus pour CTRL_BREAK_EVENT sans conflit stdout=PIPE
                # (CREATE_NEW_CONSOLE + stdout=PIPE → WinError 6 sur certains états Windows)
                # CREATE_NO_WINDOW : empêche Windows Terminal d'ouvrir un onglet console (fenêtre noire)
                # Note: CREATE_NO_WINDOW seul suffit, mais CREATE_NEW_PROCESS_GROUP est requis pour
                # CTRL_BREAK_EVENT delivery. Les deux sont compatibles avec stdout=PIPE.
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE (double protection)

            # Force UTF-8 for the child process — fixes UnicodeEncodeError on Windows cp1252
            # when engines like traiNNer-redux print emojis (rocket etc.) via rich logging.
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"
            child_env["PYTHONWARNINGS"] = "ignore::FutureWarning:pynvml"
            # Tell rich/click etc. that the terminal supports unicode.
            child_env.setdefault("FORCE_COLOR", "1")
            # Sentinel file pour arrêt propre — traiNNer poll ce fichier à chaque iter.
            # Plus fiable que CTRL_BREAK_EVENT (non-déterministe selon charge GPU/DataLoader).
            import tempfile
            self._stop_sentinel = os.path.join(
                # per runner, not per app: the Run tab and the Queue can train at the same
                # time, and a shared file stopped both / was wiped by the other's launch
                tempfile.gettempdir(), f"uss_stop_{os.getpid()}_{id(self):x}.sentinel"
            )
            # Nettoyage d'un éventuel sentinel résiduel d'une session précédente
            try:
                if os.path.exists(self._stop_sentinel):
                    os.remove(self._stop_sentinel)
            except Exception:
                pass
            child_env["USS_STOP_FILE"] = self._stop_sentinel

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
            self.save_failed = False
            _low_disk_warned = False
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    log_callback(line)
                    if ("Still cannot save" in line or "No space left on device" in line
                            or "os error 112" in line):
                        self.save_failed = True
                    if not _low_disk_warned and "Free space:" in line:
                        try:
                            _gb = float(line.split("Free space:")[1].split("GB")[0].strip())
                        except (ValueError, IndexError):
                            _gb = None
                        if _gb is not None and _gb < 5.0:
                            _low_disk_warned = True
                            log_callback(_t(
                                f"\n[ATTENTION] Plus que {_gb:.1f} Go libres : les prochaines sauvegardes "
                                f"risquent d'échouer. Libérez de la place (anciens .state / modèles).\n",
                                f"\n[WARNING] Only {_gb:.1f} GB free: the next saves may fail. "
                                f"Free some space (old .state files / models).\n"))
            
            self.process.stdout.close()
            return_code = self.process.wait()

            if return_code == 0 and getattr(self, "save_failed", False):
                log_callback(_t(
                    "\n[ATTENTION] Le processus s'est terminé, mais des sauvegardes ont échoué "
                    "(disque plein ?). La reprise repartira du dernier état complet.\n",
                    "\n[WARNING] The process ended, but some saves failed (disk full?). "
                    "Resume will restart from the last complete state.\n"))
            elif return_code == 0:
                log_callback(_t("\n[SUCCES] Entraînement terminé.\n", "\n[SUCCESS] Training finished.\n"))
            elif return_code in (3221225786, -1073741510, 3221225787, -1073741509): # CTRL+C / CTRL+BREAK Windows
                log_callback(_t("\n[STOP] Arrêt manuel confirmé (Sauvegarde OK).\n", "\n[STOP] Manual stop confirmed (save OK).\n"))
            else:
                log_callback(_t(f"\n[STOP] Processus arrêté (Code : {return_code}).\n", f"\n[STOP] Process stopped (Code: {return_code}).\n"))

        except Exception as e:
            log_callback(_t(f"\n[CRASH] : {e}\n", f"\n[CRASH]: {e}\n"))
        finally:
            self.is_running = False
            self.process = None
            self.kill_monitoring_tools()
            # Nettoyage du sentinel file si encore présent
            try:
                sf = getattr(self, '_stop_sentinel', None)
                if sf and os.path.exists(sf):
                    os.remove(sf)
            except Exception:
                pass
            # Nettoyage de la copie de travail (config stripée)
            try:
                tmp = getattr(self, '_tmp_config', None)
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
                self._tmp_config = None
            except Exception:
                pass
            if on_finish_callback:
                on_finish_callback(return_code)

    def stop_training(self, log_callback):
        if not self.process or not self.is_running:
            return

        # Si l'utilisateur clique une 2ème fois, on force le KILL
        if self.stop_requested:
            log_callback(_t("\n[KILL] Arrêt forcé immédiat !\n", "\n[KILL] Immediate forced stop!\n"))
            try: kill_process_tree(self.process.pid)
            except Exception: pass
            try: self.process.kill()
            except Exception: pass
            self.kill_monitoring_tools()
            return

        self.stop_requested = True
        log_callback(_t("\n[ACTION] Tentative de sauvegarde (Soft Stop)... Patientez...\n", "\n[ACTION] Attempting to save (Soft Stop)... Please wait...\n"))
        
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

            # Méthode principale : sentinel file → traiNNer poll à chaque iter (fiable).
            # CTRL_BREAK_EVENT : hint pour accélérer l'arrêt, non-obligatoire.
            sentinel_written = False
            stop_file = getattr(self, '_stop_sentinel', None)
            if stop_file:
                try:
                    import pathlib
                    pathlib.Path(stop_file).touch()
                    sentinel_written = True
                    log_callback(_t("> Fichier stop créé. Attente de la sauvegarde...\n", "> Stop file created. Waiting for save...\n"))
                except Exception as e_f:
                    log_callback(_t(f"[WARN] Sentinel file échoué ({e_f}) — fallback CTRL_BREAK.\n", f"[WARN] Sentinel file failed ({e_f}) — falling back to CTRL_BREAK.\n"))

            # Tenter aussi CTRL_BREAK_EVENT pour accélérer (peut ne pas fonctionner selon contexte)
            kernel32.SetConsoleCtrlHandler(None, True)
            kernel32.GenerateConsoleCtrlEvent(1, pid)   # 1 = CTRL_BREAK_EVENT (best-effort)
            kernel32.SetConsoleCtrlHandler(None, False)

            if not sentinel_written:
                log_callback(_t("> Signal CTRL_BREAK envoyé. Attente de la sauvegarde...\n", "> CTRL_BREAK signal sent. Waiting for save...\n"))

        except Exception as e:
            log_callback(_t(f"[ERREUR TECHNIQUE] {e}\n", f"[TECHNICAL ERROR] {e}\n"))

        # Surveillance : Si après 120 secondes le process tourne toujours, on kill
        waited = 0
        while self.is_running and waited < 120:
            time.sleep(1)
            waited += 1
            # Petit check pour voir si le process est mort entre temps
            if self.process and self.process.poll() is not None:
                break
        
        if self.is_running:
            log_callback(_t("\n[TIMEOUT] Le script est trop long à sauvegarder -> Kill.\n", "\n[TIMEOUT] Script is taking too long to save -> Kill.\n"))
            try: kill_process_tree(self.process.pid)
            except Exception: pass
            try: self.process.kill()
            except Exception: pass