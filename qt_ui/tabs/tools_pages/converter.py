"""
converter.py — Tools › Convertisseur (PySide6 port, faithful to tab_tools.create_page_converter).

Layout mirrors the CustomTkinter page: header, source/output paths, output-format
checkboxes (2 rows), parameters (arch / scale / opset / TF32), Convert button, log.
Conversion runs off-thread via src.core.model_convert.convert and streams to the log.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QPlainTextEdit, QFileDialog,
)
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont

from qt_ui.i18n import tr


class _ConvertWorker(QThread):
    line = Signal(str)
    done = Signal(bool)

    def __init__(self, pth, out_dir, opts):
        super().__init__()
        self.pth, self.out_dir, self.opts = pth, out_dir, opts

    def run(self):
        try:
            from src.core.model_convert import convert
            ok = convert(self.pth, self.out_dir, self.opts, log=self.line.emit)
        except Exception as e:
            self.line.emit(f"❌ {e}")
            ok = False
        self.done.emit(ok)


class ConverterPage(QWidget):
    # Sidebar-badge hooks (qt_ui/tabs/tools_tab.py) — see QuickUpscalePage.
    # No per-item progress here (single conversion), so toolBusy (indeterminate
    # spinner) rather than toolProgress.
    toolBusy = Signal()
    toolFinished = Signal(bool)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.worker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)

        title = QLabel(tr("Convertisseur de Modèle", "Model Converter")); title.setObjectName("title")
        root.addWidget(title)
        sub = QLabel(tr("Convertir un modèle vers différents formats d'inférence.", "Convert a model to various inference formats."))
        sub.setObjectName("subtitle")
        root.addWidget(sub)

        # Paths
        self.pth_edit = self._path_row(root, tr("Modèle source (.pth/.safetensors) :", "Source model (.pth/.safetensors):"), is_file=True)
        self.pth_edit.setToolTip(tr("Modèle d'entrée à convertir (.pth, .safetensors, .pt).", "Input model to convert (.pth, .safetensors, .pt)."))
        self.out_edit = self._path_row(root, tr("Dossier sortie :", "Output folder:"), is_file=False)
        self.out_edit.setToolTip(tr("Dossier où écrire les fichiers convertis.\n"
                                    "Si vide, ils sont écrits à côté du modèle source.",
                                    "Folder to write the converted files.\n"
                                    "If empty, they are written next to the source model."))

        # Output formats (transparent panel — no dark box behind the labels)
        box = QGroupBox(tr("Formats de sortie", "Output formats")); box.setStyleSheet("QGroupBox{background:transparent;}")
        gl = QGridLayout(box); gl.setHorizontalSpacing(40)
        self.chk = {}
        formats = [
            ("onnx", "ONNX"), ("fp16", tr("FP16 (demi-précision)", "FP16 (half precision)")), ("bf16", "BF16 (bfloat16)"),
            ("safetensors", "SafeTensors"),
            ("pth", tr("PTH (poids seuls)", "PTH (weights only)")), ("ncnn", "NCNN"), ("trt", "TensorRT"),
        ]
        _fmt_tips = {
            "onnx": tr("Format universel d'inférence (chaî­nes CPU/GPU, DirectML, ONNX Runtime).\n"
                       "Idéal pour utiliser le modèle hors PyTorch (chaiNNer, vidéo, etc.).",
                       "Universal inference format (CPU/GPU chains, DirectML, ONNX Runtime).\n"
                       "Ideal to use the model outside PyTorch (chaiNNer, video, etc.)."),
            "fp16": tr("Poids en demi-précision (16 bits) : ~2× plus léger, plus rapide sur GPU récent.\n"
                       "Qualité quasi identique. À éviter si ton GPU ne gère pas bien le FP16.",
                       "Half-precision weights (16-bit): ~2× lighter, faster on recent GPUs.\n"
                       "Near-identical quality. Avoid if your GPU handles FP16 poorly."),
            "bf16": tr("bfloat16 : même taille que FP16 mais meilleure plage dynamique.\n"
                       "Supporté surtout sur GPU récents (RTX 30xx+, A100…).",
                       "bfloat16: same size as FP16 but better dynamic range.\n"
                       "Mostly supported on recent GPUs (RTX 30xx+, A100…)."),
            "safetensors": tr("Format de poids sûr et rapide à charger (pas de code exécuté au chargement,\n"
                              "contrairement au .pth/pickle). Recommandé pour le partage.",
                              "Safe, fast-loading weight format (no code executed on load,\n"
                              "unlike .pth/pickle). Recommended for sharing."),
            "pth": tr("Poids seuls au format PyTorch (state_dict). Format d'origine le plus courant.",
                      "Weights only in PyTorch format (state_dict). Most common original format."),
            "ncnn": tr("⚠ Ne produit PAS de fichier .param/.bin directement — génère un ONNX\n"
                       "intermédiaire puis affiche l'instruction pour le convertir manuellement\n"
                       "avec l'outil externe onnx2ncnn (non inclus). Format Tencent NCNN,\n"
                       "utilisé par Real-ESRGAN-ncnn, waifu2x-ncnn…",
                       "⚠ Does NOT produce a .param/.bin file directly — generates an intermediate\n"
                       "ONNX file then prints the instruction to convert it manually with the\n"
                       "external onnx2ncnn tool (not included). Tencent NCNN format,\n"
                       "used by Real-ESRGAN-ncnn, waifu2x-ncnn…"),
            "trt": tr("⚠ Ne produit PAS de fichier .engine directement — génère un ONNX\n"
                      "intermédiaire puis affiche l'instruction pour le convertir manuellement\n"
                      "avec trtexec (SDK TensorRT requis, non inclus). Inférence NVIDIA\n"
                      "ultra-rapide, spécifique à un GPU + une version TensorRT donnés.",
                      "⚠ Does NOT produce a .engine file directly — generates an intermediate\n"
                      "ONNX file then prints the instruction to convert it manually with\n"
                      "trtexec (TensorRT SDK required, not included). Ultra-fast NVIDIA\n"
                      "inference, specific to a given GPU + TensorRT version."),
        }
        for i, (key, label) in enumerate(formats):
            cb = QCheckBox(label)
            if key in _fmt_tips:
                cb.setToolTip(_fmt_tips[key])
            self.chk[key] = cb
            gl.addWidget(cb, i // 4, i % 4)
        # FP16 and BF16 are a precision for the other formats, and exclusive (FP16 used
        # to win silently when both were ticked)
        self.chk["fp16"].toggled.connect(lambda on: on and self.chk["bf16"].setChecked(False))
        self.chk["bf16"].toggled.connect(lambda on: on and self.chk["fp16"].setChecked(False))
        root.addWidget(box)

        # Parameters (transparent panel; combos sit right next to their labels)
        pbox = QGroupBox(tr("Paramètres", "Settings")); pbox.setStyleSheet("QGroupBox{background:transparent;}")
        pv = QVBoxLayout(pbox)
        # B03/B04: the old "Architecture" and "Scale" combos were never read — the backend
        # always rebuilds the model from its weights (spandrel, then the traiNNer archs).
        _auto = QLabel(tr("Architecture et échelle : détectées depuis les poids (affichées dans le journal).",
                          "Architecture and scale: detected from the weights (shown in the log)."))
        _auto.setStyleSheet("color:#888;"); pv.addWidget(_auto)
        r2 = QHBoxLayout(); pv.addLayout(r2)
        r2.addWidget(QLabel(tr("Opset ONNX :", "ONNX Opset:")))
        self.opset = QComboBox(); self.opset.setFixedWidth(90); self.opset.addItems(["17", "14", "13", "11", "9"])
        self.opset.setToolTip(tr("Version du jeu d'opérateurs ONNX. 17 = récent (recommandé).\n"
                                 "Baisse-la (14, 11…) si ton moteur d'inférence est ancien et refuse le modèle.",
                                 "ONNX operator set version. 17 = recent (recommended).\n"
                                 "Lower it (14, 11…) if your inference engine is old and rejects the model."))
        r2.addWidget(self.opset); r2.addSpacing(24)
        r2.addWidget(QLabel(tr("Taille fixe ONNX :", "Fixed ONNX size:")))
        self.onnx_size = QLineEdit(); self.onnx_size.setFixedWidth(100)
        self.onnx_size.setPlaceholderText(tr("dynamique", "dynamic"))
        self.onnx_size.setToolTip(tr(
            "Vide = ONNX à taille d'entrée dynamique.\n"
            "Ex. « 960x540 » = graphe figé à cette résolution d'entrée exacte.\n"
            "Obligatoire pour les archs à attention à fenêtres (FDAT, SRFormer, SwinIR, HAT…) :\n"
            "leur export dynamique ne fonctionne qu'en 64x64. Un ONNX à taille fixe est aussi\n"
            "le plus rapide pour vs-mlrt / TensorRT.",
            "Empty = dynamic-size ONNX.\n"
            "E.g. “960x540” = graph frozen at that exact input resolution.\n"
            "Required for window-attention archs (FDAT, SRFormer, SwinIR, HAT…):\n"
            "their dynamic export only works at 64x64. A fixed-size ONNX is also\n"
            "the fastest for vs-mlrt / TensorRT."))
        r2.addWidget(self.onnx_size); r2.addSpacing(24)
        self.tf32 = QCheckBox(tr("TF32 (matmul rapide RTX 3000+)", "TF32 (fast matmul RTX 3000+)"))
        self.tf32.setToolTip(tr("Active les calculs matriciels TF32 (plus rapides) pendant la conversion.\n"
                                "Sans effet sur les GPU antérieurs aux RTX 3000.",
                                "Enables TF32 matrix math (faster) during conversion.\n"
                                "No effect on GPUs older than RTX 3000."))
        r2.addWidget(self.tf32); r2.addStretch(1)
        root.addWidget(pbox)

        self.btn = QPushButton(tr("Convertir", "Convert"))
        self.btn.clicked.connect(self._convert)
        self.btn.setToolTip(tr("Lance la conversion vers tous les formats cochés.\n"
                               "Les fichiers sont écrits dans le dossier de sortie ; le modèle source est conservé.",
                               "Runs the conversion to all checked formats.\n"
                               "Files are written to the output folder; the source model is kept."))
        root.addWidget(self.btn)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 10))
        self.log.setMaximumBlockCount(2000)
        root.addWidget(self.log, 1)

    def _path_row(self, parent_layout, label, is_file):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edit = QLineEdit()
        row.addWidget(edit, 1)
        b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedWidth(40)
        b.clicked.connect(lambda _=False: self._browse(edit, is_file))
        row.addWidget(b)
        parent_layout.addLayout(row)
        return edit

    def _browse(self, edit, is_file):
        if is_file:
            p, _ = QFileDialog.getOpenFileName(self, tr("Modèle", "Model"), "",
                                               tr("Modèles (*.pth *.safetensors *.pt);;Tous (*.*)", "Models (*.pth *.safetensors *.pt);;All (*.*)"))
        else:
            p = QFileDialog.getExistingDirectory(self, tr("Dossier sortie", "Output folder"))
        if p:
            edit.setText(p)

    def _convert(self):
        pth = self.pth_edit.text().strip()
        out_dir = self.out_edit.text().strip() or os.path.dirname(pth)
        if not pth or not os.path.isfile(pth):
            self.log.appendPlainText("❌ " + tr("Modèle source introuvable.", "Source model not found."))
            return
        opts = {k: cb.isChecked() for k, cb in self.chk.items()}
        if not any(opts.values()):
            self.log.appendPlainText("⚠️ " + tr("Sélectionne au moins un format.", "Select at least one format."))
            return
        opts["opset"] = int(self.opset.currentText())
        opts["onnx_size"] = self.onnx_size.text().strip()
        opts["tf32"] = self.tf32.isChecked()
        self.log.clear()
        self.btn.setEnabled(False); self.btn.setText("Conversion…")
        self.toolBusy.emit()
        self.worker = _ConvertWorker(pth, out_dir, opts)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.done.connect(self._done)
        self.worker.start()

    def _done(self, ok):
        self.btn.setEnabled(True); self.btn.setText(tr("Convertir", "Convert"))
        self.worker = None
        self.toolFinished.emit(bool(ok))
