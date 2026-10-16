"""
import_config.py — Tools › Importer Config (PySide6). Imports a community BasicSR-style
YAML config (Real-ESRGAN, SwinIR, RCAN, HAT...) and converts it to the app's flat format
via src.core.config_importer, then pushes it into the Configuration tab. This page had
no Qt equivalent at all — src.core.config_importer was entirely orphaned in the port.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QPlainTextEdit, QFileDialog, QMessageBox,
)
from PySide6.QtGui import QFont

from qt_ui.i18n import tr


class ImportConfigPage(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._flat = None

        v = QVBoxLayout(self); v.setContentsMargins(20, 20, 20, 20); v.setSpacing(8)
        t = QLabel(tr("📥 Importer Config", "📥 Import Config")); t.setObjectName("title"); v.addWidget(t)
        v.addWidget(QLabel(tr(
            "Importe une config communautaire (Real-ESRGAN, SwinIR, RCAN, HAT, BasicSR générique)\n"
            "et la convertit vers le format de l'app.",
            "Imports a community config (Real-ESRGAN, SwinIR, RCAN, HAT, generic BasicSR)\n"
            "and converts it to the app's format.")))

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Fichier :", "File:")))
        self.path = QLineEdit(); row.addWidget(self.path, 1)
        # B31: a new path invalidates the previous analysis
        self.path.textChanged.connect(self._invalidate)
        b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedWidth(40)
        b.clicked.connect(self._browse); row.addWidget(b)
        v.addLayout(row)

        self.btn_scan = QPushButton(tr("🔍 Analyser", "🔍 Analyze"))
        self.btn_scan.setStyleSheet('background:#3498db; color:#fff; border:none; border-radius:6px; padding:6px 14px; font-weight:600;')
        self.btn_scan.clicked.connect(self._scan)
        v.addWidget(self.btn_scan)

        self.summary = QPlainTextEdit(); self.summary.setReadOnly(True)
        self.summary.setFont(QFont("Consolas", 10)); self.summary.setMinimumHeight(220)
        v.addWidget(self.summary, 1)

        self.btn_import = QPushButton(tr("✅ Importer dans Configuration", "✅ Import into Configuration"))
        self.btn_import.setStyleSheet('background:#27ae60; color:#fff; border:none; border-radius:8px; padding:9px; font-weight:700;')
        self.btn_import.setEnabled(False)
        self.btn_import.clicked.connect(self._import)
        v.addWidget(self.btn_import)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Config à importer", "Config to import"), "",
                                           tr("Config YAML (*.yml *.yaml);;Tous (*.*)", "YAML Config (*.yml *.yaml);;All (*.*)"))
        if p:
            self.path.setText(p); self._scan()

    def _invalidate(self, *_):
        if self._flat is not None:
            self._flat = None; self.btn_import.setEnabled(False)
            self.summary.setPlainText(tr("Chemin modifié — relancez l'analyse.", "Path changed — run the analysis again."))

    def _scan(self):
        path = self.path.text().strip()
        if not path or not os.path.isfile(path):
            self.summary.setPlainText(tr("❌ Sélectionne d'abord un fichier .yml valide.", "❌ Select a valid .yml file first."))
            self.btn_import.setEnabled(False); self._flat = None
            return
        try:
            from src.core.config_importer import import_yaml_config, get_import_summary
            self._flat = import_yaml_config(path)
            self.summary.setPlainText(get_import_summary(path, self._flat))
            self.btn_import.setEnabled(bool(self._flat))
        except Exception as e:
            self.summary.setPlainText(tr(f"❌ Erreur : {e}", f"❌ Error: {e}"))
            self.btn_import.setEnabled(False); self._flat = None

    def _import(self):
        if not self._flat:
            return
        try:
            from qt_ui.tabs.config_tab import ConfigTab
            mw = self.window()
            ct = mw.findChild(ConfigTab)
            if ct is None:
                QMessageBox.information(self, "Import", tr(
                    "Onglet Configuration introuvable — impossible d'appliquer directement.",
                    "Configuration tab not found — cannot apply directly."))
                return
            flat = dict(self._flat)
            applied = ct._apply_flat_config(flat)
            # B32: a combo that refused the value keeps its old one — say so
            notes = list(self._flat.get("_warnings") or [])
            arch = self._flat.get("arch")
            if arch and ct.arch_combo.currentText() != arch:
                notes.append(tr(f"Architecture « {arch} » refusée par le formulaire (actuelle : {ct.arch_combo.currentText()})",
                                f"Architecture '{arch}' refused by the form (current: {ct.arch_combo.currentText()})"))
            # switch the main nav to the Configuration page so the user sees the result
            stack = getattr(mw, "stack", None)
            if stack is not None:
                for i in range(stack.count()):
                    pg = stack.widget(i)
                    if pg is ct or pg.findChild(ConfigTab) is ct:
                        mw._select(i); break
            msg = tr(f"Config importée : {applied} champs appliqués dans Configuration.",
                     f"Config imported: {applied} fields applied in Configuration.")
            if notes:
                msg += "\n\n" + tr("À vérifier :", "To check:") + "\n" + "\n".join(f"- {n}" for n in notes)
            QMessageBox.information(self, "Import", msg)
        except Exception as e:
            QMessageBox.critical(self, tr("Erreur", "Error"), str(e))
