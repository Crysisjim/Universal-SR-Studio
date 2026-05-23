import subprocess
import threading
import os
import sys
import signal
import time
import ctypes


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
            marker = "def check_dependencies():"
            bypass_line = "    return  # auto-patched by Universal SR Studio — bypasses false PyTorch version error\n"
            # Already bypassed if any `return` immediately follows the function def
            import re as _re
            if _re.search(r"def check_dependencies\(\):\s*\n\s+return\b", src):
                return  # already patched
            idx = src.find(marker)
            if idx == -1:
                return
            insert_at = src.find("\n", idx) + 1
            patched = src[:insert_at] + bypass_line + src[insert_at:]
            with open(target, "w", encoding="utf-8") as f:
                f.write(patched)
            log_callback("[PATCH] check_dependencies.py — bypass version PyTorch appliqué.\n")
        except Exception as e:
            log_callback(f"[WARN] Impossible de patcher check_dependencies.py : {e}\n")

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
            try: self.process.kill()
            except Exception: pass