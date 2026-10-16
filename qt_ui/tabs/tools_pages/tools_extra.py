"""
tools_extra.py — extra Tools pages (PySide6):
  · HistoryPage    — list training experiments (NeoSR / Redux)
  · LMDBPage       — build an LMDB from an image folder
  · ResumePage     — find experiments with .state checkpoints to resume
  · BenchmarkPage  — drive benchmark_runner.py (arch/feature × NeoSR/Redux)

(GenLQPage and PublishPage used to also live here but were dead code — the real,
routed pages are qt_ui/tabs/tools_pages/gen_lq.py and publish.py respectively.
Removed 2026-09-17 after an audit confirmed zero references anywhere in the repo;
this docstring itself had gone stale, still listing them as if live.)

All heavy work runs off the UI thread (QThread).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QPlainTextEdit, QProgressBar, QFileDialog, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QDialog, QMessageBox,
    QCheckBox,
)
from PySide6.QtCore import QThread, Signal, Qt, QProcess
from PySide6.QtGui import QFont, QColor

from qt_ui.i18n import tr

_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _row(parent, grid, r, label, folder=False, file=False, flt="Tous (*.*)"):
    grid.addWidget(QLabel(label), r, 0)
    e = QLineEdit(); grid.addWidget(e, r, 1)
    b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedWidth(40)

    def pick():
        if folder:
            p = QFileDialog.getExistingDirectory(parent, tr("Dossier", "Folder"))
        else:
            p, _ = QFileDialog.getOpenFileName(parent, label, "", flt)
        if p:
            e.setText(p)
    b.clicked.connect(pick); grid.addWidget(b, r, 2)
    return e


# ── Historique ─────────────────────────────────────────────────────────────────
_HIST_STATUS_COLORS = {"running": "#f39c12", "completed": "#2ecc71",
                       "interrupted": "#e74c3c", "failed": "#c0392b"}


class HistoryPage(QWidget):
    """Faithful to the v2.5.8 history: full metrics table backed by the
    training_history DB + Refresh / Arch stats / Export benchmark / Delete all."""
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        v = QVBoxLayout(self); v.setContentsMargins(20, 16, 20, 16); v.setSpacing(8)
        t = QLabel(tr("Historique des Entraînements", "Training History")); t.setObjectName("title"); v.addWidget(t)
        sub = QLabel(tr("Tous les trainings lancés depuis cette app, avec leurs métriques.", "All trainings launched from this app, with their metrics."))
        sub.setStyleSheet("color:#888;"); v.addWidget(sub)

        row = QHBoxLayout(); v.addLayout(row)
        def _b(text, color, slot):
            b = QPushButton(text)
            b.setStyleSheet(f'background:{color}; color:#fff; border:none; border-radius:6px; padding:7px 14px; '
                            f'font-weight:600; font-family:"Segoe UI Emoji"; font-size:13px;')
            b.clicked.connect(slot); return b
        row.addWidget(_b(tr("🔄 Rafraîchir", "🔄 Refresh"), "#3498db", self._refresh))
        row.addWidget(_b(tr("📊 Stats par Architecture", "📊 Stats per Architecture"), "#9b59b6", self._arch_stats))
        row.addWidget(_b(tr("💾 Exporter Benchmark TXT", "💾 Export Benchmark TXT"), "#27ae60", self._export_txt))
        row.addStretch(1)
        row.addWidget(_b(tr("🗑 Supprimer la sélection", "🗑 Delete selection"), "#e67e22", self._delete_selected))
        row.addWidget(_b(tr("🗑 Tout Supprimer", "🗑 Delete All"), "#c0392b", self._delete_all))

        self._row_checks = []  # (QCheckBox, training_id)
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(["☑", tr("Nom", "Name"), "Arch", "Iter", "PSNR", tr("Vitesse", "Speed"), tr("Durée", "Duration"), tr("Date", "Date"), "Status", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        # Slightly blue panel (matches the v2.5.8 history list).
        self.table.setStyleSheet(
            "QTableWidget{ background:#1a1a2e; alternate-background-color:#20203a; gridline-color:#2a2a44; }"
            "QHeaderView::section{ background:#2b2b4b; color:#aab; border:none; padding:4px; }"
            "QTableWidget::item:selected{ background:#33335a; }")
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed); self.table.setColumnWidth(0, 34)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in range(2, 9):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(9, QHeaderView.Fixed); self.table.setColumnWidth(9, 40)
        v.addWidget(self.table, 1)
        self._refresh()

    def _refresh(self):
        try:
            from src.core.training_history import get_recent_trainings, format_duration, format_timestamp
        except Exception as e:
            self.table.setRowCount(1); self.table.setItem(0, 0, QTableWidgetItem(f"Backend indisponible : {e}")); return
        trainings = get_recent_trainings(limit=200)
        self._row_checks = []
        self.table.setRowCount(len(trainings))
        for r, t in enumerate(trainings):
            psnr = t.get("best_psnr", 0); spd = t.get("avg_speed", 0); status = t.get("status", "?")

            def _cell(col, text, color=None):
                it = QTableWidgetItem(str(text))
                if color:
                    it.setForeground(QColor(color))
                self.table.setItem(r, col, it)

            # selection checkbox (centered in col 0)
            cw = QWidget(); cl = QHBoxLayout(cw); cl.setContentsMargins(0, 0, 0, 0); cl.setAlignment(Qt.AlignCenter)
            chk = QCheckBox(); cl.addWidget(chk)
            self.table.setCellWidget(r, 0, cw); self._row_checks.append((chk, t["id"]))

            _cell(1, (t.get("name", "") or "")[:48], "#3498db")
            _cell(2, str(t.get("architecture", "?")), "#999")
            _cell(3, f"{t.get('current_iter', 0)}/{t.get('total_iter', 0)}", "#999")
            _cell(4, f"{psnr:.2f}" if psnr else "—", "#2ecc71" if psnr else "#666")
            _cell(5, f"{spd:.2f} it/s" if spd else "—", "#f39c12" if spd else "#666")
            _cell(6, format_duration(t.get("duration_seconds", 0)), "#aaa")
            _cell(7, format_timestamp(t.get("started_at", 0)), "#888")
            _cell(8, status, _HIST_STATUS_COLORS.get(status, "#888"))
            tip = (f"Moteur: {t.get('engine','?')} · Scale: {t.get('scale','?')} · "
                   f"Batch: {t.get('batch_size','?')} · Patch: {t.get('patch_size','?')}\n"
                   f"Optim: {t.get('optimizer','?')} · LR: {t.get('lr','?')} · GAN: {t.get('gan_mode','?')}\n"
                   f"SSIM: {t.get('best_ssim','—')} · VRAM peak: {t.get('vram_peak_mb','?')}MB · "
                   f"Power: {t.get('power_avg_w','?')}W · GPU: {t.get('gpu_name','?')}\n"
                   f"Losses: {t.get('losses','—')}")
            for c in range(1, 9):
                if self.table.item(r, c):
                    self.table.item(r, c).setToolTip(tip)
            btn = QPushButton("X"); btn.setFixedSize(26, 22); btn.setToolTip(tr("Supprimer cet entraînement", "Delete this training"))
            btn.setStyleSheet('QPushButton{background:#c0392b; color:#fff; border:none; border-radius:4px; padding:0; margin:0; font-weight:800; font-size:13px;}')
            btn.clicked.connect(lambda _=False, rid=t["id"]: self._delete_one(rid))
            bw = QWidget(); bl = QHBoxLayout(bw); bl.setContentsMargins(0, 0, 0, 0); bl.setAlignment(Qt.AlignCenter); bl.addWidget(btn)
            self.table.setCellWidget(r, 9, bw)

    def _delete_one(self, rid):
        if QMessageBox.question(self, tr("Supprimer", "Delete"), tr(f"Supprimer l'entraînement #{rid} ?", f"Delete training #{rid}?")) == QMessageBox.Yes:
            try:
                from src.core.training_history import delete_training
                delete_training(rid); self._refresh()
            except Exception as e:
                QMessageBox.critical(self, tr("Erreur", "Error"), str(e))

    def _delete_selected(self):
        ids = [rid for chk, rid in getattr(self, "_row_checks", []) if chk.isChecked()]
        if not ids:
            QMessageBox.information(self, tr("Suppression", "Deletion"), tr("Aucun entraînement coché.", "No training checked.")); return
        if QMessageBox.question(self, tr("Supprimer la sélection", "Delete selection"),
                                tr(f"Supprimer {len(ids)} entraînement(s) coché(s) ?\nCette action est irréversible.",
                                   f"Delete {len(ids)} checked training(s)?\nThis action is irreversible.")) == QMessageBox.Yes:
            try:
                from src.core.training_history import delete_training
                for rid in ids:
                    delete_training(rid)
                self._refresh()
            except Exception as e:
                QMessageBox.critical(self, tr("Erreur", "Error"), str(e))

    def _delete_all(self):
        if QMessageBox.question(self, tr("Tout Supprimer", "Delete All"),
                                tr("Supprimer TOUT l'historique des entraînements ?\nCette action est irréversible.",
                                   "Delete the ENTIRE training history?\nThis action is irreversible.")) == QMessageBox.Yes:
            try:
                from src.core.training_history import delete_all_trainings
                delete_all_trainings(); self._refresh()
            except Exception as e:
                QMessageBox.critical(self, tr("Erreur", "Error"), str(e))

    def _export_txt(self):
        from datetime import datetime
        default = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(self, tr("Exporter Benchmark", "Export Benchmark"), default, tr("Fichier texte (*.txt);;Tous (*.*)", "Text file (*.txt);;All (*.*)"))
        if not path:
            return
        try:
            from src.core.training_history import export_benchmark_txt
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(export_benchmark_txt())
            QMessageBox.information(self, tr("Export OK", "Export OK"), tr(f"Benchmark exporté :\n{path}", f"Benchmark exported:\n{path}"))
        except Exception as e:
            QMessageBox.critical(self, tr("Erreur Export", "Export error"), str(e))

    def _arch_stats(self):
        try:
            from src.core.training_history import get_stats_by_architecture, format_duration
            stats = get_stats_by_architecture()
        except Exception as e:
            QMessageBox.critical(self, "Stats", str(e)); return
        dlg = QDialog(self); dlg.setWindowTitle(tr("Stats par Architecture", "Stats per Architecture")); dlg.resize(700, 400)
        lv = QVBoxLayout(dlg)
        h = QLabel(tr("Performance par Architecture", "Performance per Architecture")); h.setAlignment(Qt.AlignCenter)
        h.setStyleSheet("font-size:16px; font-weight:800; padding:8px;"); lv.addWidget(h)
        if not stats:
            lv.addWidget(QLabel(tr("Pas assez de données.", "Not enough data."))); dlg.exec(); return
        for s in stats:
            box = QWidget(); box.setStyleSheet("background:#1a1a2e; border-radius:6px;")
            hb = QHBoxLayout(box); hb.setContentsMargins(12, 8, 12, 8)
            a = QLabel(str(s["architecture"])); a.setFixedWidth(160); a.setStyleSheet("color:#3498db; font-weight:700; background:transparent;")
            hb.addWidget(a)
            hb.addWidget(self._stat_lbl(f"Trainings: {s['count']}", "#aaa", 110))
            hb.addWidget(self._stat_lbl(f"PSNR moy: {s['avg_psnr']:.2f}", "#2ecc71", 140))
            hb.addWidget(self._stat_lbl(f"PSNR max: {s['max_psnr']:.2f}", "#27ae60", 140))
            hb.addWidget(self._stat_lbl(tr(f"Durée moy: {format_duration(int(s['avg_duration']))}", f"Avg duration: {format_duration(int(s['avg_duration']))}"), "#888", 160))
            hb.addStretch(1)
            lv.addWidget(box)
        lv.addStretch(1)
        dlg.exec()

    @staticmethod
    def _stat_lbl(text, color, w):
        lb = QLabel(text); lb.setFixedWidth(w); lb.setStyleSheet(f"color:{color}; background:transparent;")
        return lb


# ── Créateur LMDB ──────────────────────────────────────────────────────────────
_LMDB_SCRIPT = '''import os, sys, cv2, lmdb
src=r"{src}"; dst=r"{dst}"
try:
    files=sorted(f for f in os.listdir(src) if f.lower().endswith((".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff")))
    # B48: the engines' LMDB readers encode keys as ASCII and key = file stem, so a
    # non-ASCII name cannot be read back and two files with the same stem collide.
    seen={{}}; keep=[]; skipped=[]
    for f in files:
        k=os.path.splitext(f)[0]
        if not k.isascii():
            skipped.append(f); print(f"SKIP:{{f}}:nom non ASCII (illisible par les lecteurs LMDB des moteurs)"); continue
        if k in seen:
            skipped.append(f); print(f"SKIP:{{f}}:même nom que {{seen[k]}} (clé LMDB en double)"); continue
        seen[k]=f; keep.append(f)
    # map size from the data (PNG level 1 ~ 3x the source files), doubled on MapFullError.
    # A fixed 1 TB map made Windows size data.mdb at 1 TB, which the engines' readers
    # then failed to open in their workers.
    _map=sum(os.path.getsize(os.path.join(src,f)) for f in keep)*3+(256<<20)
    env=lmdb.open(dst, map_size=_map)
    txn=env.begin(write=True); count=0; meta=[]; _pending=[]
    for i,f in enumerate(keep):
        try:
            img=cv2.imread(os.path.join(src,f), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("image illisible/corrompue (cv2.imread a retourné None)")
            ok,buf=cv2.imencode(".png",img,[cv2.IMWRITE_PNG_COMPRESSION,1])
            if not ok:
                raise ValueError("échec encodage PNG")
            k=os.path.splitext(f)[0]
            while True:
                try:
                    txn.put(k.encode("ascii"), buf.tobytes()); break
                except lmdb.MapFullError:
                    txn.abort(); _map*=2; env.set_mapsize(_map); txn=env.begin(write=True)
                    for _k,_b in _pending: txn.put(_k,_b)
            _pending.append((k.encode("ascii"), buf.tobytes())); count+=1
            h,w=img.shape[:2]; c=1 if img.ndim==2 else img.shape[2]
            meta.append(f"{{k}}.png ({{h}},{{w}},{{c}}) 1")
        except Exception as fe:
            skipped.append(f)
            print(f"SKIP:{{f}}:{{fe}}"); sys.stdout.flush()
            continue
        if count%50==0:
            txn.commit(); txn=env.begin(write=True); _pending=[]; print(f"PROGRESS:{{i}}/{{len(keep)}}"); sys.stdout.flush()
    txn.commit(); env.close()
    # B47: meta_info.txt is required by the Redux and NeoSR LMDB readers
    with open(os.path.join(dst,"meta_info.txt"),"w",encoding="utf-8") as mf:
        mf.write("\\n".join(meta)+("\\n" if meta else ""))
    if skipped:
        print(f"SKIPPED_TOTAL:{{len(skipped)}}:{{','.join(skipped)}}")
    print(f"DONE:{{count}}")
except Exception as e:
    print(f"ERROR:{{e}}")
'''


class LMDBPage(QWidget):
    # Sidebar-badge hooks (qt_ui/tabs/tools_tab.py) — see QuickUpscalePage.
    toolProgress = Signal(int)
    toolFinished = Signal(bool)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings; self.proc = None; self._tmp = None
        v = QVBoxLayout(self); v.setContentsMargins(20, 20, 20, 20); v.setSpacing(8)
        t = QLabel(tr("Créateur LMDB", "LMDB Creator")); t.setObjectName("title"); v.addWidget(t)
        v.addWidget(QLabel(tr("Optimise un dataset pour la vitesse de lecture (LMDB). Lu par NeoSR (mode paired, GT et LQ en .lmdb) ;\n"
                              "traiNNer-redux ne sait pas lire les bases LMDB.",
                              "Optimizes a dataset for read speed (LMDB). Read by NeoSR (paired mode, GT and LQ as .lmdb);\n"
                              "traiNNer-redux cannot read LMDB databases.")))
        g = QGridLayout(); v.addLayout(g)
        self.src = _row(self, g, 0, tr("Source images :", "Source images:"), folder=True)
        self.src.setToolTip(tr("Dossier d'images à empaqueter dans la base LMDB.", "Image folder to package into the LMDB database."))
        self.dst = _row(self, g, 1, tr("Sortie (.lmdb) :", "Output (.lmdb):"), folder=True)
        self.dst.setToolTip(tr("Chemin de la base LMDB à créer (l'extension .lmdb est ajoutée si absente).", "Path of the LMDB database to create (.lmdb extension added if missing)."))
        r = QHBoxLayout(); v.addLayout(r)
        self.btn = QPushButton("▶ START"); self.btn.clicked.connect(self._start)
        self.btn.setToolTip(tr("Crée une base LMDB à partir des images : lecture du dataset bien plus rapide\n"
                            "pendant l'entraînement (utile pour les gros datasets / disques lents).", "Creates an LMDB database from the images: much faster dataset reads\nduring training (useful for large datasets / slow disks)."))
        self.btn.setStyleSheet('background:#27ae60;color:#fff;border-radius:6px;padding:6px;'); r.addWidget(self.btn, 1)
        self.btn_stop = QPushButton("⏹ STOP"); self.btn_stop.setObjectName("warn"); self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip(tr("Interrompt la création de la base LMDB.", "Aborts the LMDB database creation."))
        self.btn_stop.clicked.connect(self._stop); r.addWidget(self.btn_stop)
        from qt_ui.widgets import same_height; same_height(self.btn, self.btn_stop)
        self.prog = QProgressBar(); v.addWidget(self.prog)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFont(QFont("Consolas", 10)); v.addWidget(self.log, 1)

    def _venv_py(self):
        try:
            from src.core import engine_paths as ep
            for fn in ("any_engine_python", "shared_venv_python"):
                if hasattr(ep, fn):
                    p = getattr(ep, fn)()
                    if p and os.path.isfile(p):
                        return p
        except Exception:
            pass
        return self.settings.get("python_path", "python") or "python"

    def _start(self):
        if self.proc is not None:
            return
        src, dst = self.src.text().strip(), self.dst.text().strip()
        if not os.path.isdir(src):
            self.log.setPlainText("❌ Source invalide."); return
        if not dst:
            self.log.setPlainText("❌ Sortie requise."); return
        if not dst.endswith(".lmdb"):
            dst += ".lmdb"; self.dst.setText(dst)
        import tempfile
        self._tmp = os.path.join(tempfile.gettempdir(), f"_usrs_lmdb_{os.getpid()}.py")
        with open(self._tmp, "w", encoding="utf-8") as f:
            f.write(_LMDB_SCRIPT.format(src=src, dst=dst))
        self.log.clear(); self.prog.setValue(0)
        self.btn.setEnabled(False); self.btn_stop.setEnabled(True)
        self._had_error = False; self._done_ok = False; self._stopped = False
        self.toolProgress.emit(0)
        self.proc = QProcess(self); self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._out)
        self.proc.finished.connect(self._done)
        # B49: a process that cannot start never emits finished
        self.proc.errorOccurred.connect(self._proc_error)
        self.proc.start(self._venv_py(), [self._tmp])

    def _out(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            if line.startswith("PROGRESS:"):
                try:
                    a, b = line.split(":")[1].split("/")
                    pct = int(int(a) / max(int(b), 1) * 100)
                    self.prog.setValue(pct); self.toolProgress.emit(pct)
                except Exception:
                    pass
            elif line.startswith("SKIP:"):
                # One bad file no longer aborts the whole batch (2026-09-17 fix) —
                # surface each skip in the log as it happens, named, so the user
                # knows exactly which file(s) to look at.
                _rest = line[len("SKIP:"):]
                _fname, _, _reason = _rest.partition(":")
                self.log.appendPlainText(f"⚠ {_fname} : {_reason or tr('ignoré (erreur)', 'skipped (error)')}")
            elif line.startswith("SKIPPED_TOTAL:"):
                _rest = line[len("SKIPPED_TOTAL:"):]
                _n, _, _names = _rest.partition(":")
                self.log.appendPlainText(tr(
                    f"⚠ {_n} image(s) ignorée(s) au total : {_names}",
                    f"⚠ {_n} image(s) skipped in total: {_names}"))
            elif line.startswith("DONE:"):
                self.log.appendPlainText(tr(f"✅ LMDB créé ({line.split(':')[1]} images) + meta_info.txt.",
                                            f"✅ LMDB created ({line.split(':')[1]} images) + meta_info.txt.")); self.prog.setValue(100)
                self._done_ok = True
            elif line.startswith("ERROR:"):
                self.log.appendPlainText("❌ " + line[6:])
                self._had_error = True
            else:
                self.log.appendPlainText(line)

    def _proc_error(self, err):
        from PySide6.QtCore import QProcess as _QP
        if err == _QP.ProcessError.FailedToStart:
            self.log.appendPlainText(tr(f"❌ Impossible de lancer Python ({self._venv_py()}).",
                                        f"❌ Cannot start Python ({self._venv_py()})."))
            self._had_error = True
            self._done()

    def _done(self, code=None, *_):
        if self.proc is None and code is None:
            return
        self.btn.setEnabled(True); self.btn_stop.setEnabled(False); self.proc = None
        # B49: success = the script reached DONE, no ERROR, exit code 0, not stopped
        ok = self._done_ok and not self._had_error and not self._stopped and (code in (None, 0))
        if self._stopped:
            self.log.appendPlainText(tr("⏹ Arrêté — base LMDB incomplète.", "⏹ Stopped — LMDB incomplete."))
        elif not ok and not self._had_error:
            self.log.appendPlainText(tr(f"❌ Le script s'est arrêté sans terminer (code {code}).",
                                        f"❌ The script stopped before finishing (code {code})."))
        self.toolFinished.emit(ok)
        try:
            if self._tmp and os.path.isfile(self._tmp):
                os.remove(self._tmp)
        except Exception:
            pass

    def _stop(self):
        if self.proc is not None:
            self._stopped = True
            self.proc.kill()


# ── Resume Failed ──────────────────────────────────────────────────────────────
class _ResumeScanWorker(QThread):
    done = Signal(list)

    def run(self):
        try:
            from src.core.resume_failed import scan_interrupted_trainings
            self.done.emit(list(scan_interrupted_trainings()))
        except Exception as e:
            # B50: a failed scan is not "everything finished"
            self.done.emit([{"_error": str(e)}])


class ResumePage(QWidget):
    """Faithful to v2.5.8 'Trainings Interrompus' : scan + cards (name [engine],
    iter reached, .state availability, config) with Resume / Open-folder."""
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings; self._worker = None
        from PySide6.QtWidgets import QScrollArea, QFrame
        v = QVBoxLayout(self); v.setContentsMargins(20, 16, 20, 16); v.setSpacing(8)
        t = QLabel(tr("Trainings Interrompus", "Interrupted Trainings")); t.setObjectName("title"); v.addWidget(t)
        sub = QLabel(tr("Détecte les entraînements arrêtés avant la fin et propose de les reprendre.", "Detects trainings stopped before the end and offers to resume them."))
        sub.setStyleSheet("color:#888;"); v.addWidget(sub)
        btn = QPushButton(tr("🔍 Scanner experiments/", "🔍 Scan experiments/"))
        btn.setStyleSheet('background:#e67e22; color:#fff; border:none; border-radius:6px; padding:8px 16px; '
                          'font-weight:600; font-family:"Segoe UI Emoji"; font-size:13px;')
        btn.clicked.connect(self._scan)
        v.addWidget(btn, alignment=Qt.AlignCenter)

        self._scroll = QScrollArea(); self._scroll.setWidgetResizable(True); self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet("QScrollArea{background:#1a1a2e; border-radius:6px;}")
        self._host = QWidget(); self._host.setStyleSheet("background:transparent;")
        self._hv = QVBoxLayout(self._host); self._hv.setContentsMargins(10, 10, 10, 10); self._hv.setSpacing(6)
        self._hv.addStretch(1)
        self._scroll.setWidget(self._host); v.addWidget(self._scroll, 1)
        self._info(tr("Cliquez sur « Scanner » pour rechercher les entraînements interrompus.", "Click “Scan” to look for interrupted trainings."), "#888")
        self._scan()

    def _clear(self):
        while self._hv.count():
            it = self._hv.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

    def _info(self, text, color):
        self._clear()
        lb = QLabel(text); lb.setStyleSheet(f"color:{color}; background:transparent;"); lb.setWordWrap(True)
        self._hv.addWidget(lb); self._hv.addStretch(1)

    def _scan(self):
        self._info(tr("Scan en cours…", "Scanning…"), "#888")
        self._worker = _ResumeScanWorker(); self._worker.done.connect(self._show); self._worker.start()

    def _show(self, interrupted):
        self._worker = None
        self._clear()
        if interrupted and "_error" in interrupted[0]:
            self._info(tr(f"❌ Scan impossible : {interrupted[0]['_error']}", f"❌ Scan failed: {interrupted[0]['_error']}"), "#e74c3c")
            return
        if not interrupted:
            self._info(tr("✅ Aucun entraînement interrompu trouvé.\n\nTous vos entraînements se sont terminés correctement.",
                       "✅ No interrupted training found.\n\nAll your trainings finished correctly."), "#2ecc71")
            return
        head = QLabel(tr(f"⚠ {len(interrupted)} entraînement(s) interrompu(s) détecté(s) :", f"⚠ {len(interrupted)} interrupted training(s) detected:"))
        head.setStyleSheet("color:#f39c12; font-weight:700; background:transparent;")
        self._hv.addWidget(head)
        try:
            from src.core.resume_failed import find_associated_config
        except Exception:
            find_associated_config = lambda p: None
        for item in interrupted:
            cfg = find_associated_config(item["path"])
            self._hv.addWidget(self._card(item, cfg))
        self._hv.addStretch(1)

    def _card(self, item, cfg):
        from PySide6.QtWidgets import QFrame
        card = QFrame(); card.setObjectName("rcard")  # bg/border from theme QSS
        cv = QVBoxLayout(card); cv.setContentsMargins(12, 8, 12, 8); cv.setSpacing(3)
        top = QHBoxLayout()
        nm = QLabel(f'<b style="color:#3498db;">{item["name"]}</b> <span style="color:#666;">[{item.get("engine","?")}]</span>')
        nm.setStyleSheet("background:transparent;"); top.addWidget(nm); top.addStretch(1)
        md = QLabel(tr(f"Modifié : {item.get('mtime_str','?')}", f"Modified: {item.get('mtime_str','?')}")); md.setStyleSheet("color:#888; background:transparent;")
        top.addWidget(md); cv.addLayout(top)
        mid = QHBoxLayout()
        it_lbl = QLabel(tr(f"Iter atteinte : {item.get('last_iter','?')}", f"Iteration reached: {item.get('last_iter','?')}")); it_lbl.setStyleSheet("color:#888; background:transparent;")
        mid.addWidget(it_lbl)
        if item.get("running"):
            st = QLabel(tr("⏳ Journal modifié il y a moins de 2 min — probablement en cours",
                           "⏳ Log written less than 2 min ago — probably still running"))
            st.setStyleSheet("color:#3498db; background:transparent;")
        elif item.get("state_file"):
            st = QLabel(tr("✓ State file disponible", "✓ State file available")); st.setStyleSheet("color:#2ecc71; background:transparent;")
        else:
            st = QLabel(tr("⚠ Pas de .state", "⚠ No .state")); st.setStyleSheet("color:#e67e22; background:transparent;")
        mid.addWidget(st); mid.addStretch(1); cv.addLayout(mid)
        btns = QHBoxLayout()
        if cfg:
            cl = QLabel(f"Config : {os.path.basename(cfg)}"); cl.setStyleSheet("color:#888; background:transparent;")
            btns.addWidget(cl)
        else:
            cl = QLabel(tr("(Config non trouvée)", "(Config not found)")); cl.setStyleSheet("color:#e74c3c; background:transparent;")
            btns.addWidget(cl)
        btns.addStretch(1)
        of = QPushButton(tr("📁 Ouvrir dossier", "📁 Open folder")); of.setStyleSheet('background:#666; color:#fff; border:none; border-radius:5px; padding:5px 10px; font-family:"Segoe UI Emoji"; font-size:12px;')
        of.clicked.connect(lambda _=False, p=item["path"]: self._open(p)); btns.addWidget(of)
        if cfg and not item.get("running"):
            rb = QPushButton(tr("▶ Reprendre", "▶ Resume")); rb.setStyleSheet('background:#27ae60; color:#fff; border:none; border-radius:5px; padding:5px 12px; font-weight:600; font-family:"Segoe UI Emoji"; font-size:12px;')
            rb.clicked.connect(lambda _=False, c=cfg, it=item: self._resume(c, it)); btns.addWidget(rb)
        cv.addLayout(btns)
        return card

    def _open(self, path):
        try:
            os.startfile(path)
        except Exception as e:
            QMessageBox.warning(self, tr("Ouvrir", "Open"), str(e))

    def _resume(self, cfg, item=None):
        st = (item or {}).get("state_file")
        txt = tr(f"Charger la config dans l'onglet Entraînement et activer l'Auto-resume ?\n{os.path.basename(cfg)}",
                 f"Load the config into the Training tab and enable Auto-resume?\n{os.path.basename(cfg)}")
        if st:
            txt += tr(f"\n\nReprise depuis : {os.path.basename(st)}", f"\n\nResumes from: {os.path.basename(st)}")
        if QMessageBox.question(self, tr("Reprendre", "Resume"), txt) != QMessageBox.Yes:
            return
        try:
            from qt_ui.tabs.run_tab import RunTab
            mw = self.window()
            rt = mw.findChild(RunTab)
            if rt and "config_path" in getattr(rt, "entries", {}):
                rt.entries["config_path"].setText(cfg)
                # B52: resuming = Auto-resume on (the runner picks the latest .state of
                # this experiment); without it the card's Resume restarted from scratch
                if hasattr(rt, "chk_resume"):
                    rt.chk_resume.setChecked(True)
                # switch the main nav to the page hosting the RunTab
                stack = getattr(mw, "stack", None)
                if stack is not None:
                    for i in range(stack.count()):
                        pg = stack.widget(i)
                        if pg is rt or pg.findChild(RunTab) is rt:
                            mw._select(i); break
            else:
                from PySide6.QtWidgets import QApplication
                QApplication.clipboard().setText(cfg)
                QMessageBox.information(self, tr("Reprendre", "Resume"), tr("Onglet Entraînement introuvable — chemin de config copié dans le presse-papiers.", "Training tab not found — config path copied to clipboard."))
        except Exception as e:
            QMessageBox.critical(self, tr("Reprendre", "Resume"), str(e))


# ── Benchmark ──────────────────────────────────────────────────────────────────
# Drives src/core/benchmark_runner.py as a subprocess (see BenchmarkPage._run below) —
# the real CLI tool with per-engine venv resolution, arch/feature backends and --list,
# not an in-process shortcut.

class BenchmarkPage(QWidget):
    # Sidebar-badge hooks (qt_ui/tabs/tools_tab.py) — see QuickUpscalePage.
    # Only phase-level progress is available (no per-iteration %), so
    # toolProgress here means "phase i/n done", not fine-grained training %.
    toolProgress = Signal(int)
    toolFinished = Signal(bool)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        import threading
        self.settings = settings; self._stop = None; self._threading = threading
        self._cmd_proc = None; self._list_proc = None; self._cmds = []; self._cmd_idx = 0
        from PySide6.QtWidgets import QCheckBox, QScrollArea, QFrame
        root = QVBoxLayout(self); root.setContentsMargins(16, 12, 16, 12); root.setSpacing(6)
        hdr = QHBoxLayout(); root.addLayout(hdr)
        lc = QVBoxLayout(); lc.setSpacing(0)
        t = QLabel("📈 Benchmark"); t.setObjectName("title"); lc.addWidget(t)
        sub = QLabel(tr("Mesure les it/s, VRAM et stabilité de chaque architecture.", "Measures it/s, VRAM and stability of each architecture."))
        sub.setStyleSheet("color:#888;"); lc.addWidget(sub)
        hdr.addLayout(lc); hdr.addStretch(1)
        try:
            from qt_ui.tabs.tools_pages.quick_upscale import _GpuMiniPanel
            self.gpu_panel = _GpuMiniPanel(); hdr.addWidget(self.gpu_panel, 0, Qt.AlignTop | Qt.AlignRight)
        except Exception:
            self.gpu_panel = None

        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame)
        inner = QWidget(); sc.setWidget(inner); root.addWidget(sc, 1)
        v = QVBoxLayout(inner); v.setSpacing(5)

        def _lab(text, w):
            r = QHBoxLayout(); lb = QLabel(text); lb.setFixedWidth(170); r.addWidget(lb); r.addWidget(w)
            r.addStretch(1); v.addLayout(r); return w

        self.engine = QComboBox(); self.engine.addItems(["Redux", "NeoSR", tr("Les deux", "Both")]); self.engine.setFixedWidth(140)
        self.engine.setToolTip(tr("Moteur d'entraînement sur lequel mesurer les architectures.", "Training engine on which to benchmark the architectures."))
        _lab("Moteur :", self.engine)
        self.btype = QComboBox(); self.btype.addItems(["Architectures", "Features", "Arch + Features"]); self.btype.setFixedWidth(160)
        self.btype.setToolTip(tr("Architectures : compare des réseaux différents.\n"
                              "Features : compare des options (AMP, compile…) sur une même arch.\n"
                              "Arch + Features : combine les deux.", "Architectures: compares different networks.\nFeatures: compares options (AMP, compile…) on the same arch.\nArch + Features: combines both."))
        _lab("Type de benchmark :", self.btype)
        self.niter = QLineEdit(str(settings.get("bench_n_iter", "2500"))); self.niter.setFixedWidth(90)
        self.niter.setToolTip(tr("Nombre d'itérations d'entraînement par test. Plus élevé = mesure de\n"
                              "vitesse plus fiable, mais benchmark plus long. 2000–3000 suffit en général.", "Number of training iterations per test. Higher = more reliable\nspeed measurement, but longer benchmark. 2000–3000 is usually enough."))
        _lab(tr("Itérations / test :", "Iterations / test:"), self.niter)
        self.tout = QLineEdit(str(settings.get("bench_timeout", "3600"))); self.tout.setFixedWidth(90)
        self.tout.setToolTip(tr("Temps maximum (secondes) accordé à chaque test avant abandon.\n"
                             "Évite qu'une architecture trop lourde bloque tout le benchmark.", "Maximum time (seconds) allowed for each test before giving up.\nPrevents a too-heavy architecture from blocking the whole benchmark."))
        _lab("Timeout (s) / test :", self.tout)

        mrow = QHBoxLayout(); ml = QLabel(tr("Modes précision :", "Precision modes:")); ml.setFixedWidth(170); mrow.addWidget(ml)
        self.m_normal = QCheckBox("normal"); self.m_normal.setChecked(True)
        self.m_fp16 = QCheckBox("fp16"); self.m_bf16 = QCheckBox("bf16"); self.m_tf32 = QCheckBox("tf32")
        self.m_normal.setToolTip(tr("Précision FP32 standard (référence).", "Standard FP32 precision (reference)."))
        self.m_fp16.setToolTip(tr("Demi-précision FP16 : plus rapide + moins de VRAM sur GPU récents.", "FP16 half precision: faster + less VRAM on recent GPUs."))
        self.m_bf16.setToolTip(tr("bfloat16 : comme FP16 mais plus stable numériquement (GPU récents).", "bfloat16: like FP16 but numerically more stable (recent GPUs)."))
        self.m_tf32.setToolTip(tr("TF32 : matmul accéléré sur RTX 3000+ (léger gain, qualité préservée).", "TF32: accelerated matmul on RTX 3000+ (slight gain, quality preserved)."))
        for c in (self.m_normal, self.m_fp16, self.m_bf16, self.m_tf32):
            mrow.addWidget(c)
        mrow.addStretch(1); v.addLayout(mrow)

        self.archs = QLineEdit(); self.archs.setPlaceholderText(tr("vide = tous  |  ex: compact,span,hat_s", "empty = all  |  e.g. compact,span,hat_s"))
        self.archs.setToolTip(tr("Liste d'architectures à tester, séparées par des virgules.\n"
                              "Laisse vide pour tester TOUTES les architectures disponibles.", "List of architectures to test, comma-separated.\nLeave empty to test ALL available architectures."))
        _lab(tr("Tests ciblés :", "Targeted tests:"), self.archs)
        _sc = QLabel(tr("fixée par chaque test du backend", "set by each backend test")); _sc.setStyleSheet("color:#888;")
        _lab("Scale :", _sc)
        self.train = self._prow(v, "Dataset train GT :", settings.get("bench_train_gt", ""), folder=True)
        self.vgt = self._prow(v, "Val GT (optionnel) :", "", folder=True)
        self.vlq = self._prow(v, "Val LQ (optionnel) :", "", folder=True)
        self.test = self._prow(v, "Image test (optionnel) :", "", folder=False)
        self.test.setToolTip(tr("Image upscalée à la fin de chaque test (sauf Redux + Features, qui n'en fait pas).",
                                "Image upscaled at the end of each test (except Redux + Features, which has no upscale test)."))
        self.out = self._prow(v, tr("Dossier résultats :", "Results folder:"), settings.get("bench_out", os.path.join(__import__("src.core.engine_paths", fromlist=["base_engine_path"]).base_engine_path(), "benchmark_results")), folder=True)

        crow = QHBoxLayout()
        self.reset_state = QCheckBox(tr("Reset état (repart de zéro)", "Reset state (start over)"))
        self.reset_state.setToolTip(tr("Ignore tout checkpoint existant et relance chaque test depuis l'itération 0.", "Ignores any existing checkpoint and restarts each test from iteration 0."))
        crow.addWidget(self.reset_state)
        self.no_upscale = QCheckBox(tr("Pas de test upscale", "No upscale test"))
        self.no_upscale.setToolTip(tr("Mesure uniquement la vitesse d'entraînement, sans générer d'image upscalée\n"
                                   "de contrôle à la fin de chaque test (benchmark plus rapide).", "Measures only training speed, without generating a control upscaled\nimage at the end of each test (faster benchmark)."))
        crow.addWidget(self.no_upscale); crow.addStretch(1)
        v.addLayout(crow)
        self.engine.currentIndexChanged.connect(self._sync_support)
        self.btype.currentIndexChanged.connect(self._sync_support)
        self._sync_support()
        v.addStretch(1)

        br = QHBoxLayout(); root.addLayout(br)
        self.btn = QPushButton(tr("▶ Lancer le Benchmark", "▶ Run Benchmark"))
        self.btn.setToolTip(tr("Entraîne brièvement chaque architecture sélectionnée et mesure\n"
                            "vitesse (it/s), VRAM et stabilité. Compare-les pour choisir la plus adaptée à ton GPU.", "Briefly trains each selected architecture and measures\nspeed (it/s), VRAM and stability. Compare them to pick the best fit for your GPU."))
        self.btn.setStyleSheet(
            'QPushButton{background:#27ae60; color:#fff; border:none; border-radius:6px; padding:8px 14px; font-weight:700; font-family:"Segoe UI Emoji";}'
            'QPushButton:hover{background:#2ecc71;}'
            'QPushButton:pressed{background:#1e8449; padding-top:10px;}'
            'QPushButton:disabled{background:#3a4a40; color:#8aa;}')
        self.btn.clicked.connect(self._run); br.addWidget(self.btn)
        self.btn_stop = QPushButton("⏹ Stop"); self.btn_stop.setEnabled(False); self.btn_stop.setFixedWidth(90)
        self.btn_stop.setToolTip(tr("Arrête le benchmark immédiatement : le test en cours et ses entraînements sont interrompus.",
                                    "Stops the benchmark now: the running test and its trainings are interrupted."))
        self.btn_stop.setStyleSheet(
            'QPushButton{background:#e74c3c; color:#fff; border:none; border-radius:6px; padding:8px; font-weight:700; font-family:"Segoe UI Emoji";}'
            'QPushButton:hover{background:#ef6657;}'
            'QPushButton:pressed{background:#b83a2e; padding-top:10px;}'
            'QPushButton:disabled{background:#4a3a3a; color:#a88;}')
        self.btn_stop.clicked.connect(self._request_stop); br.addWidget(self.btn_stop)
        blist = QPushButton(tr("📋 Lister tests", "📋 List tests")); blist.setObjectName("ghost")
        blist.setToolTip(tr("Liste les tests disponibles pour le moteur/type choisi, sans lancer de benchmark.",
                            "Lists the available tests for the chosen engine/type, without running a benchmark."))
        blist.clicked.connect(self._list_tests); br.addWidget(blist)
        bopen = QPushButton(tr("📂 Ouvrir résultats", "📂 Open results")); bopen.setObjectName("ghost")
        bopen.setToolTip(tr("Ouvre le dossier des résultats du benchmark dans l'explorateur.", "Opens the benchmark results folder in the explorer."))
        bopen.clicked.connect(lambda: os.startfile(self.out.text()) if os.path.isdir(self.out.text()) else None); br.addWidget(bopen)
        br.addStretch(1)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFont(QFont("Consolas", 10)); root.addWidget(self.log, 1)

    def _sync_support(self, *_):
        e = self.engine.currentText(); t = self.btype.currentText()
        redux = e == "Redux" or e in ("Les deux", "Both")
        arch = t != "Features"
        feature = t != "Architectures"
        # B55: only the Redux backends know --reset
        self.reset_state.setEnabled(redux)
        self.reset_state.setToolTip(tr("Ignore tout checkpoint existant et relance chaque test depuis l'itération 0."
                                       + ("" if e == "Redux" else " (Redux uniquement)"),
                                       "Ignores any existing checkpoint and restarts each test from iteration 0."
                                       + ("" if e == "Redux" else " (Redux only)")))
        # B56: Redux + Features has no upscale test
        only_rf = e == "Redux" and t == "Features"
        self.no_upscale.setEnabled(not only_rf)
        # B54: the precision modes drive the Architectures backends only
        for c in (self.m_normal, self.m_fp16, self.m_bf16, self.m_tf32):
            c.setEnabled(arch)

    def _prow(self, layout, label, value, folder=True):
        r = QHBoxLayout(); lb = QLabel(label); lb.setFixedWidth(170); r.addWidget(lb)
        e = QLineEdit(value or ""); r.addWidget(e, 1)
        b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedWidth(40)

        def pick():
            p = QFileDialog.getExistingDirectory(self, label) if folder else QFileDialog.getOpenFileName(self, label, "", "Images (*.png *.jpg *.jpeg);;Tous (*.*)")[0]
            if p:
                e.setText(p)
        b.clicked.connect(pick); r.addWidget(b); layout.addLayout(r)
        return e

    # ── real backend: drives src/core/benchmark_runner.py as a subprocess (one call
    # per engine×type combination, run sequentially), exactly like the CTk app did.
    # The in-process arch_benchmark.run_benchmark_arch() shortcut this used to fall
    # back to ignored the Moteur/Type selectors entirely — this is the real deal. ──
    def _resolve_script(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        return sys.executable, str(root / "src" / "core" / "benchmark_runner.py")

    def _build_cmds(self, python, script, list_only=False):
        engine_sel = self.engine.currentText()       # "Redux" | "NeoSR" | "Les deux"/"Both"
        type_sel = self.btype.currentText()           # "Architectures" | "Features" | "Arch + Features"
        engines = ["redux", "neosr"] if engine_sel in ("Les deux", "Both") else [engine_sel.lower()]
        btypes = (["arch", "feature"] if "+" in type_sel else
                  ["feature"] if type_sel.lower().startswith(("feature", "features")) else ["arch"])
        if list_only:
            # B57: every selected engine x type, not only the first one
            return [[python, script, "--engine", e, "--type", b, "--list"] for e in engines for b in btypes]
        n_iter = self.niter.text().strip() or "0"
        timeout = self.tout.text().strip() or "0"
        tests = self.archs.text().strip()
        gt = self.train.text().strip()
        outdir = self.out.text().strip()
        modes_str = ",".join(m for m, c in (("normal", self.m_normal), ("fp16", self.m_fp16),
                                             ("bf16", self.m_bf16), ("tf32", self.m_tf32)) if c.isChecked())
        cmds = []
        for eng in engines:
            # Redux "Arch + Features": the arch backend chains the feature benchmark on
            # its fastest arch (a separate feature phase ran them twice before)
            _bts = ["arch"] if (eng == "redux" and btypes == ["arch", "feature"]) else btypes
            for bt in _bts:
                cmd = [python, script, "--engine", eng, "--type", bt, "--n-iter", n_iter, "--timeout", timeout]
                if eng == "redux" and bt == "arch" and _bts != btypes:
                    cmd.append("--chain-features")
                if tests:
                    cmd += ["--tests", tests]
                if gt:
                    cmd += ["--train-gt", gt]
                if outdir:
                    cmd += ["--output-dir", outdir]
                for flag, w in (("--val-gt", self.vgt), ("--val-lq", self.vlq), ("--test-image", self.test)):
                    if w.text().strip():
                        cmd += [flag, w.text().strip()]
                if self.reset_state.isChecked():
                    cmd.append("--reset")
                if self.no_upscale.isChecked():
                    cmd.append("--no-upscale")
                if bt == "arch" and modes_str:
                    cmd += ["--modes", modes_str]
                cmds.append(cmd)
        return cmds

    def _list_tests(self):
        python, script = self._resolve_script()
        if not os.path.isfile(script):
            self.log.setPlainText(tr(f"❌ Script introuvable :\n{script}", f"❌ Script not found:\n{script}")); return
        self._list_cmds = self._build_cmds(python, script, list_only=True)
        self.log.clear()
        self.log.appendPlainText(tr("[Benchmark] Liste des tests disponibles :\n", "[Benchmark] Available tests list:\n"))
        self._list_next()

    def _list_next(self, *_):
        if not getattr(self, "_list_cmds", None):
            self._list_proc = None; return
        cmd = self._list_cmds.pop(0)
        self.log.appendPlainText(f"── {cmd[cmd.index('--engine') + 1]} / {cmd[cmd.index('--type') + 1]} ──")
        proc = QProcess(self); proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda p=proc: self.log.appendPlainText(bytes(p.readAllStandardOutput()).decode("utf-8", "replace")))
        proc.finished.connect(self._list_next)
        proc.errorOccurred.connect(lambda err, c=cmd: self.log.appendPlainText(f"❌ {c[0]} : {err}"))
        proc.start(cmd[0], cmd[1:])
        self._list_proc = proc  # keep a ref so it isn't GC'd mid-run

    def _request_stop(self):
        if self._stop:
            self._stop.set()
        if self._cmd_proc is not None:
            pid = self._cmd_proc.processId()
            if pid and sys.platform == "win32":
                # runner -> backend -> train.py: kill() alone left the children running
                import subprocess as _sp
                _sp.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                        creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0))
            else:
                self._cmd_proc.kill()
        self.btn_stop.setEnabled(False); self.btn_stop.setText(tr("Arrêt…", "Stopping…"))

    def _run(self):
        if self._cmd_proc is not None:
            return
        python, script = self._resolve_script()
        if not os.path.isfile(script):
            self.log.setPlainText(tr(f"❌ Script introuvable :\n{script}\n\nVérifie l'installation du moteur.",
                                      f"❌ Script not found:\n{script}\n\nCheck the engine installation."))
            return
        out = self.out.text().strip() or os.path.join(__import__("src.core.engine_paths", fromlist=["base_engine_path"]).base_engine_path(), "benchmark_results")
        self.out.setText(out)
        self.settings.set("bench_n_iter", self.niter.text().strip()); self.settings.set("bench_timeout", self.tout.text().strip())
        self.settings.set("bench_train_gt", self.train.text().strip()); self.settings.set("bench_out", out)
        self._cmds = self._build_cmds(python, script)
        self._cmd_idx = 0
        self._stop = self._threading.Event()
        self.log.clear()
        self.log.appendPlainText(f"[Benchmark] {self.engine.currentText()} — {self.btype.currentText()}")
        if len(self._cmds) > 1:
            self.log.appendPlainText(tr(f"[Benchmark] {len(self._cmds)} phase(s) à exécuter successivement.\n",
                                         f"[Benchmark] {len(self._cmds)} phase(s) to run sequentially.\n"))
        self.btn.setEnabled(False); self.btn.setText("Benchmark…")
        self.btn_stop.setEnabled(True); self.btn_stop.setText("⏹ Stop")
        self._bench_ok = True
        self.toolProgress.emit(0)
        self._run_next_cmd()

    def _run_next_cmd(self):
        if self._stop is not None and self._stop.is_set():
            self._finish(); return
        if self._cmd_idx >= len(self._cmds):
            self.log.appendPlainText(tr("\n[Benchmark] ✅ Toutes les phases terminées — voir les résultats.",
                                         "\n[Benchmark] ✅ All phases completed — see results."))
            self._finish(); return
        cmd = self._cmds[self._cmd_idx]
        n = len(self._cmds)
        self.toolProgress.emit(int(self._cmd_idx / max(n, 1) * 100))
        if n > 1:
            self.log.appendPlainText("\n" + "=" * 60)
            try:
                ei = cmd.index("--engine"); ti = cmd.index("--type")
                phase = f"{cmd[ei + 1].upper()} / {cmd[ti + 1]}"
            except ValueError:
                phase = tr(f"phase {self._cmd_idx + 1}", f"phase {self._cmd_idx + 1}")
            self.log.appendPlainText(tr(f"[Benchmark] Phase {self._cmd_idx + 1}/{n} : {phase}",
                                         f"[Benchmark] Phase {self._cmd_idx + 1}/{n}: {phase}"))
            self.log.appendPlainText("=" * 60)
        self.log.appendPlainText(tr(f"[Benchmark] Commande : {' '.join(cmd)}\n", f"[Benchmark] Command: {' '.join(cmd)}\n"))
        self._cmd_proc = QProcess(self); self._cmd_proc.setProcessChannelMode(QProcess.MergedChannels)
        self._cmd_proc.readyReadStandardOutput.connect(self._on_cmd_output)
        self._cmd_proc.finished.connect(self._on_cmd_finished)
        self._cmd_proc.errorOccurred.connect(self._on_cmd_error)
        self._cmd_proc.start(cmd[0], cmd[1:])

    def _on_cmd_error(self, err):
        if err == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText(tr(f"❌ Impossible de lancer : {self._cmds[self._cmd_idx][0]}",
                                        f"❌ Cannot start: {self._cmds[self._cmd_idx][0]}"))
            self._cmd_proc = None; self._bench_ok = False
            self._finish()

    def _on_cmd_output(self):
        if self._cmd_proc is not None:
            self.log.appendPlainText(bytes(self._cmd_proc.readAllStandardOutput()).decode("utf-8", "replace").rstrip("\n"))

    def _on_cmd_finished(self, code, _status):
        self._cmd_proc = None
        i = self._cmd_idx + 1
        if self._stop is not None and self._stop.is_set():
            self.log.appendPlainText(tr("\n[Benchmark] ⏹ Arrêté par l'utilisateur.", "\n[Benchmark] ⏹ Stopped by the user."))
            self._bench_ok = False
            self._finish(); return
        if code != 0:
            self.log.appendPlainText(tr(f"\n[Benchmark] ❌ Phase {i} terminée avec erreur (code {code}). Arrêt.",
                                         f"\n[Benchmark] ❌ Phase {i} finished with error (code {code}). Stopping."))
            self._bench_ok = False
            self._finish(); return
        self.log.appendPlainText(tr(f"\n[Benchmark] ✅ Phase {i}/{len(self._cmds)} terminée.",
                                     f"\n[Benchmark] ✅ Phase {i}/{len(self._cmds)} done."))
        self._cmd_idx += 1
        self._run_next_cmd()

    def _finish(self):
        self.btn.setEnabled(True); self.btn.setText(tr("▶ Lancer le Benchmark", "▶ Run Benchmark"))
        self.btn_stop.setEnabled(False); self.btn_stop.setText("⏹ Stop")
        self._stop = None; self._cmd_proc = None
        self.toolFinished.emit(getattr(self, "_bench_ok", True))
