"""
queue_tab.py — File d'attente (PySide6). A simple training queue: add config files,
reorder / remove them, then run them sequentially via the SAME TrainingRunner pipeline
used by the Entraînement tab (patches/custom injection, gallery/TensorBoard auto-start,
Historique logging, notification-on-finish) — not a bare subprocess. Mirrors the CTk
tab_queue intent, which drove the queue through app.train_tab.external_start(...).
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QPlainTextEdit, QFileDialog, QProgressBar,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor

from qt_ui.i18n import tr



def _load_any_config(path):
    """YAML (traiNNer-redux) or TOML (NeoSR) config -> dict."""
    if str(path).lower().endswith(".toml"):
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f) or {}
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

class QueueTab(QWidget):
    _log_sig = Signal(str)
    _fin_sig = Signal(int)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.runner = None   # src.core.runner.TrainingRunner instance, one per queued item
        self._hist_id = None
        self._running_idx = -1
        self._user_stopped = False
        self._via_rt = False        # current item runs in the Training tab
        self._rt_connected = False
        self._log_sig.connect(self._on_runner_log)
        self._fin_sig.connect(self._on_finished)

        root = QVBoxLayout(self); root.setContentsMargins(16, 16, 16, 16); root.setSpacing(8)
        t = QLabel(tr("File d'attente", "Queue")); t.setObjectName("title"); root.addWidget(t)
        root.addWidget(QLabel(tr("Enchaîne plusieurs configs d'entraînement, l'une après l'autre.", "Chain several training configs, one after another.")))

        ctr = QHBoxLayout(); root.addLayout(ctr)
        self._edit_btns = []
        for txt, slot, obj in ((tr("➕ Ajouter Config", "➕ Add Config"), self._add, ""), (tr("📁 Dossier", "📁 Folder"), self._add_folder, "ghost"),
                               (tr("🗑 Retirer", "🗑 Remove"), self._remove, "ghost"), ("⬆", self._up, "ghost"),
                               ("⬇", self._down, "ghost"), (tr("♻ Vider", "♻ Clear"), self._clear, "ghost")):
            b = QPushButton(txt)
            if obj:
                b.setObjectName(obj)
            if txt in ("⬆", "⬇"):
                b.setFixedWidth(40)
            b.clicked.connect(slot); ctr.addWidget(b); self._edit_btns.append(b)
        ctr.addStretch(1)

        self.list = QListWidget(); root.addWidget(self.list, 1)

        run = QHBoxLayout(); root.addLayout(run)
        self.btn_start = QPushButton(tr("▶  Lancer la queue", "▶  Run queue")); self.btn_start.clicked.connect(self._start)
        self.btn_start.setStyleSheet('background:#27ae60; color:#fff; border:none; border-radius:8px; padding:10px; font-weight:700;')
        run.addWidget(self.btn_start, 1)
        self.btn_stop = QPushButton(tr("⏹  Stop", "⏹  Stop")); self.btn_stop.setObjectName("warn"); self.btn_stop.setEnabled(False)
        from qt_ui.widgets import same_height; same_height(self.btn_start, self.btn_stop)
        self.btn_stop.clicked.connect(self._stop); run.addWidget(self.btn_stop)

        self.prog = QProgressBar(); self.prog.setRange(0, 100); root.addWidget(self.prog)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFont(QFont("Consolas", 10))
        self.log.setMaximumBlockCount(6000); root.addWidget(self.log, 1)

    # ── list management ──
    def _add(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Configs", "", tr("Config (*.yml *.yaml *.toml);;Tous (*.*)", "Config (*.yml *.yaml *.toml);;All (*.*)"))
        for p in paths:
            self.list.addItem(QListWidgetItem(p))

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr("Dossier de configs", "Configs folder"))
        if folder:
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith((".yml", ".yaml", ".toml")):
                    self.list.addItem(QListWidgetItem(os.path.join(folder, f)))

    def _remove(self):
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))

    def _clear(self):
        self.list.clear()

    def _move(self, delta):
        r = self.list.currentRow()
        if r < 0:
            return
        nr = r + delta
        if 0 <= nr < self.list.count():
            it = self.list.takeItem(r); self.list.insertItem(nr, it); self.list.setCurrentRow(nr)

    def _up(self):
        self._move(-1)

    def _down(self):
        self._move(1)

    # ── sequential run ──
    def _configs(self):
        return [self.list.item(i).text() for i in range(self.list.count())]

    def _start(self):
        if self.runner is not None or self.list.count() == 0:
            return
        self._running_idx = -1
        self._user_stopped = False
        # frozen copy + locked editing: removing / moving items during the run shifted the
        # index the queue walks with, so a config could be skipped or run twice
        self._run_list = self._configs()
        self._lock_edit(True)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self._next()

    def _lock_edit(self, locked):
        for b in getattr(self, "_edit_btns", []):
            b.setEnabled(not locked)

    def _next(self):
        self._running_idx += 1
        configs = getattr(self, "_run_list", None) or self._configs()
        if self._running_idx >= len(configs):
            self._append(tr("\n== File d'attente terminée ==\n", "\n== Queue finished ==\n"))
            self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self.runner = None
            self._lock_edit(False)
            self.prog.setValue(100)
            return
        cfg = configs[self._running_idx]
        self._highlight(self._running_idx)
        self.prog.setValue(int(self._running_idx / max(len(configs), 1) * 100))
        py, script = self._build_command(cfg)
        if not py or not script:
            self._append(tr("❌ Config invalide, on saute : ", "❌ Invalid config, skipping: ") + f"{cfg}\n"); self._next(); return
        self._append(f"\n>> [{self._running_idx + 1}/{len(configs)}] {os.path.basename(cfg)}\n")
        # Preferred path: drive the Training tab (full dashboard, formatted log, badges,
        # Historique, notifications) — the CTk queue did the same via external_start().
        rt = self._run_tab()
        if rt is not None:
            if rt.is_busy():
                self._append(tr("❌ Un entraînement tourne déjà dans l'onglet Entraînement — file arrêtée.\n",
                                "❌ A training is already running in the Training tab — queue stopped.\n"))
                self._user_stopped = True
                self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self._lock_edit(False)
                return
            if not self._rt_connected:
                rt.run_finished.connect(self._on_rt_finished)
                self._rt_connected = True
            self._via_rt = True
            last = self._running_idx >= len(configs) - 1
            if rt.external_start(cfg, suppress_shutdown=not last):
                self._append(tr("   → suivi en direct dans l'onglet Entraînement.\n",
                                "   → live progress in the Training tab.\n"))
                return
            self._via_rt = False
            self._append(tr("❌ Lancement impossible, on saute : ", "❌ Could not start, skipping: ") + f"{cfg}\n")
            self._next(); return
        self._record_history_start(cfg, script)
        from src.core.runner import TrainingRunner
        self.runner = TrainingRunner()
        log_cb = lambda s: self._log_sig.emit(str(s))
        fin_cb = lambda *a: self._fin_sig.emit(int(a[0]) if a and isinstance(a[0], int) else 0)
        try:
            self.runner.start_training(py, script, cfg, log_cb, fin_cb)
        except Exception as e:
            self._append(f"\n❌ {e}\n"); self.runner = None; self._next()

    def _build_command(self, cfg):
        """Pick the venv python + the train.py matching the config's engine
        (.yml/.yaml → Redux, .toml → NeoSR) — same rule as the Entraînement tab."""
        py = self.settings.get("python_path", "") or ""
        try:
            from src.core import engine_paths as ep
            venv_py = ep.shared_venv_python()
            if venv_py and os.path.isfile(venv_py):
                py = venv_py
        except Exception:
            pass
        if not py:
            try:
                from src.core import engine_paths as ep
                py = ep.shared_venv_python()
            except Exception:
                py = "python"
        if not cfg or not os.path.isfile(cfg):
            return None, None
        is_redux = os.path.splitext(cfg)[1].lower() in (".yml", ".yaml")
        try:
            from src.core import engine_paths as ep
            order = [ep.redux_path()] if is_redux else [ep.neosr_path()]  # never the other engine
            for root in order:
                ts = os.path.join(root, "train.py")
                if os.path.isfile(ts):
                    return py, ts
        except Exception:
            pass
        return None, None

    def _record_history_start(self, config_path, script_path):
        """Log this queued run in the Historique DB (parity with the Entraînement tab)."""
        self._hist_id = None
        try:
            from src.core.training_history import record_training_start
            cfg = _load_any_config(config_path)
            ng = cfg.get("network_g", {})
            arch = ng.get("type", "unknown") if isinstance(ng, dict) else (ng if isinstance(ng, str) else "unknown")
            train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
            self._hist_id = record_training_start(
                name=cfg.get("name", os.path.splitext(os.path.basename(config_path))[0]),
                engine="NeoSR" if "neosr" in script_path.lower() else "Redux",
                architecture=str(arch), scale=int(cfg.get("scale", 4) or 4),
                total_iter=int(float(train.get("total_iter") or cfg.get("total_iter")
                                     or (cfg.get("logger") or {}).get("total_iter") or 0)),  # NeoSR: [logger]
                config_path=config_path)
        except Exception:
            pass

    def _highlight(self, idx):
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setForeground(Qt.green if i == idx else Qt.white if i > idx else Qt.gray)

    def _on_runner_log(self, text):
        self._append(text if text.endswith("\n") else text + "\n")

    def _on_finished(self, code):
        try:
            from src.core.training_history import record_training_end
            status = "interrupted" if getattr(self, "_user_stopped", False) else ("completed" if code == 0 else "failed")
            record_training_end(self._hist_id or 0, status=status)
        except Exception:
            pass
        self._append(tr(f"\n== Config terminée (code {code}) ==\n", f"\n== Config done (code {code}) ==\n"))
        self.runner = None
        if not getattr(self, "_user_stopped", False):
            self._next()

    def _run_tab(self):
        try:
            from qt_ui.tabs.run_tab import RunTab
            w = self.window()
            return w.findChild(RunTab) if w is not None else None
        except Exception:
            return None

    def _on_rt_finished(self, code, user_stopped):
        """Training-tab run finished (only acted on while this queue drives it)."""
        if not self._via_rt:
            return
        self._via_rt = False
        self._append(tr(f"== Config terminée (code {code}) ==\n", f"== Config done (code {code}) ==\n"))
        if user_stopped or self._user_stopped:
            self._user_stopped = True
            self._append(tr("== Queue arrêtée ==\n", "== Queue stopped ==\n"))
            self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self._lock_edit(False)
            return
        self._next()

    def _stop(self):
        self._user_stopped = True
        self.btn_stop.setEnabled(False)
        if self._via_rt:
            rt = self._run_tab()
            if rt is not None and rt.is_busy():
                rt._stop()   # soft stop + save, same as STOP & SAVE in the Training tab
                self._append(tr("\n== Arrêt demandé (sauvegarde en cours) ==\n", "\n== Stop requested (saving) ==\n"))
                return
        if self.runner is not None:
            try:
                self.runner.stop_training(lambda s: self._log_sig.emit(str(s)))
            except Exception as e:
                self._append(f"\n❌ stop : {e}\n")
        self._append(tr("\n== Queue arrêtée ==\n", "\n== Queue stopped ==\n"))
        self.btn_start.setEnabled(True)
        self._lock_edit(False)

    def _append(self, text):
        sb = self.log.verticalScrollBar(); at_bottom = sb.value() >= sb.maximum() - 4
        self.log.moveCursor(QTextCursor.End); self.log.insertPlainText(text)
        if at_bottom:
            self.log.moveCursor(QTextCursor.End)
