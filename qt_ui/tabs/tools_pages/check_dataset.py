"""
check_dataset.py — Tools › Check Dataset (PySide6). Folder picker + integrity check
via src.core.dataset_tools.DatasetTools.check_integrity (off-thread, streamed log).
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QPlainTextEdit, QFileDialog,
)
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont

from qt_ui.i18n import tr


class _CheckWorker(QThread):
    line = Signal(str)
    done = Signal()

    def __init__(self, folder, lq_folder=None):
        super().__init__()
        self.folder = folder
        self.lq_folder = lq_folder or None

    def run(self):
        try:
            from src.core.dataset_tools import DatasetTools
            dt = DatasetTools()
            dt.check_integrity(self.folder, callback=self.line.emit, lq_folder_path=self.lq_folder)
        except Exception as e:
            self.line.emit(f"❌ {e}")
        self.done.emit()


class CheckDatasetPage(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.worker = None
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        t = QLabel("Check Dataset"); t.setObjectName("title")
        root.addWidget(t)
        root.addWidget(QLabel(tr("Vérifie l'intégrité des images d'un dossier (corrompues, illisibles…).", "Checks the integrity of a folder's images (corrupted, unreadable…).")))

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Dossier HR :", "HR Folder:")))
        self.edit = QLineEdit(); row.addWidget(self.edit, 1)
        self.edit.setToolTip(tr("Dossier d'images à vérifier (HR, train ou val).\n"
                             "L'analyse parcourt aussi les sous-dossiers.", "Image folder to check (HR, train or val).\nThe analysis also scans subfolders."))
        b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedWidth(40)
        b.setToolTip(tr("Parcourir pour choisir le dossier à vérifier.", "Browse to choose the folder to check."))
        b.clicked.connect(self._pick); row.addWidget(b)
        root.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel(tr("Dossier LQ (optionnel) :", "LQ Folder (optional):")))
        self.edit_lq = QLineEdit(); row2.addWidget(self.edit_lq, 1)
        self.edit_lq.setToolTip(tr(
            "Optionnel — si renseigné, vérifie aussi que chaque image HR a bien\n"
            "une contrepartie LQ du même nom, et que les résolutions HR/LQ ont\n"
            "un ratio entier cohérent sur les deux axes.",
            "Optional — if set, also checks that every HR image has a matching\n"
            "LQ counterpart with the same name, and that HR/LQ resolutions have\n"
            "a consistent integer ratio on both axes."))
        b2 = QPushButton("…"); b2.setObjectName("ghost"); b2.setFixedWidth(40)
        b2.setToolTip(tr("Parcourir pour choisir le dossier LQ (optionnel).", "Browse to choose the LQ folder (optional)."))
        b2.clicked.connect(self._pick_lq); row2.addWidget(b2)
        root.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn = QPushButton(tr("Vérifier", "Check")); self.btn.clicked.connect(self._run); row3.addWidget(self.btn)
        self.btn.setToolTip(tr("Ouvre chaque image et signale les fichiers corrompus, illisibles,\n"
                            "tronqués ou au mauvais format. Si un dossier LQ est renseigné, vérifie\n"
                            "aussi les paires manquantes et la cohérence des résolutions.\n"
                            "Aucune image n'est modifiée ni supprimée.",
                            "Opens each image and reports corrupted, unreadable,\ntruncated or wrong-format files. If an LQ folder is set, also checks\n"
                            "for missing pairs and resolution consistency.\nNo image is modified or deleted."))
        row3.addStretch(1)
        root.addLayout(row3)

        self.out = QPlainTextEdit(); self.out.setReadOnly(True)
        self.out.setFont(QFont("Consolas", 10)); self.out.setMaximumBlockCount(5000)
        root.addWidget(self.out, 1)

    def _pick(self):
        p = QFileDialog.getExistingDirectory(self, tr("Dossier dataset", "Dataset folder"))
        if p:
            self.edit.setText(p)

    def _pick_lq(self):
        p = QFileDialog.getExistingDirectory(self, tr("Dossier LQ (optionnel)", "LQ folder (optional)"))
        if p:
            self.edit_lq.setText(p)

    def _run(self):
        folder = self.edit.text().strip()
        lq_folder = self.edit_lq.text().strip()
        if not folder or not os.path.isdir(folder):
            self.out.setPlainText(tr("❌ Dossier introuvable.", "❌ Folder not found."))
            return
        if lq_folder and not os.path.isdir(lq_folder):
            self.out.setPlainText(tr("❌ Dossier LQ introuvable.", "❌ LQ folder not found."))
            return
        self.out.clear()
        self.btn.setEnabled(False); self.btn.setText(tr("Vérification…", "Checking…"))
        self.worker = _CheckWorker(folder, lq_folder)
        self.worker.line.connect(self.out.appendPlainText)
        self.worker.done.connect(self._done)
        self.worker.start()

    def _done(self):
        self.btn.setEnabled(True); self.btn.setText(tr("Vérifier", "Check"))
        self.worker = None
