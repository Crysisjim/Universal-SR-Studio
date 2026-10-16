"""
model_info.py — Tools › Info Modèle (PySide6). Inspects a model file and shows its
format / params / inferred scale via the existing backend
src.core.model_export.detect_model_format.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QPlainTextEdit, QFileDialog,
)
from PySide6.QtGui import QFont

from qt_ui.i18n import tr


class ModelInfoPage(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        t = QLabel(tr("Info Modèle", "Model Info")); t.setObjectName("title")
        root.addWidget(t)
        root.addWidget(QLabel(tr("Inspecte un modèle (format, paramètres, scale détecté).", "Inspects a model (format, parameters, detected scale).")))

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Modèle :", "Model:")))
        self.edit = QLineEdit(); row.addWidget(self.edit, 1)
        self.edit.setToolTip(tr("Fichier modèle à inspecter (.pth, .safetensors, .pt, .onnx).", "Model file to inspect (.pth, .safetensors, .pt, .onnx)."))
        b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedWidth(40)
        b.setToolTip(tr("Parcourir pour choisir le fichier modèle.", "Browse to choose the model file."))
        b.clicked.connect(self._pick); row.addWidget(b)
        ana = QPushButton(tr("Analyser", "Analyze")); ana.clicked.connect(self._analyze); row.addWidget(ana)
        ana.setToolTip(tr("Lit le modèle et affiche : format, taille, nombre de paramètres,\n"
                       "types de données, architecture détectée (via spandrel), scale,\n"
                       "canaux entrée/sortie et support FP16/BF16. Lecture seule, ne modifie rien.", "Reads the model and shows: format, size, parameter count,\ndata types, detected architecture (via spandrel), scale,\ninput/output channels and FP16/BF16 support. Read-only, modifies nothing."))
        self.conv_btn = QPushButton(tr("→ SafeTensors", "→ SafeTensors")); self.conv_btn.setObjectName("ghost")
        self.conv_btn.setToolTip(tr("Convertit rapidement ce modèle en .safetensors (à côté du fichier source).",
                                    "Quickly converts this model to .safetensors (next to the source file)."))
        self.conv_btn.clicked.connect(self._quick_convert); row.addWidget(self.conv_btn)
        self.edit.textChanged.connect(self._sync_conv)
        root.addLayout(row)

        self.out = QPlainTextEdit(); self.out.setReadOnly(True)
        self.out.setFont(QFont("Consolas", 10))
        root.addWidget(self.out, 1)

    def _sync_conv(self, *_):
        # B39: the quick conversion reads PyTorch checkpoints; an ONNX file has no state dict
        onnx = self.edit.text().strip().lower().endswith(".onnx")
        self.conv_btn.setEnabled(not onnx)
        self.conv_btn.setToolTip(
            tr("Indisponible pour un .onnx (pas de state_dict PyTorch à convertir).",
               "Unavailable for a .onnx (no PyTorch state_dict to convert).") if onnx else
            tr("Convertit rapidement ce modèle en .safetensors (à côté du fichier source).",
               "Quickly converts this model to .safetensors (next to the source file)."))

    def _pick(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Modèle", "Model"), "",
                                           tr("Modèles (*.pth *.safetensors *.pt *.onnx);;Tous (*.*)", "Models (*.pth *.safetensors *.pt *.onnx);;All (*.*)"))
        if p:
            self.edit.setText(p); self._analyze()

    def _analyze(self):
        path = self.edit.text().strip()
        if not path or not os.path.isfile(path):
            self.out.setPlainText(tr("❌ Fichier introuvable.", "❌ File not found."))
            return
        self.out.setPlainText(tr("Analyse en cours…", "Analyzing…"))
        from PySide6.QtCore import QThread, Signal

        class _W(QThread):
            done = Signal(str)
            def run(self_):
                self_.done.emit(self._collect(path))
        self._w = _W(); self._w.done.connect(self.out.setPlainText); self._w.start()

    def _quick_convert(self):
        path = self.edit.text().strip()
        if not path or not os.path.isfile(path):
            self.out.setPlainText(tr("❌ Fichier introuvable.", "❌ File not found.")); return
        self.conv_btn.setEnabled(False)
        self.out.setPlainText(tr("Conversion en SafeTensors…", "Converting to SafeTensors…"))
        from PySide6.QtCore import QThread, Signal

        class _CW(QThread):
            line = Signal(str); done = Signal(bool)

            def run(self_):
                try:
                    from src.core.model_convert import convert
                    out_dir = os.path.dirname(path)
                    ok = convert(path, out_dir, {"safetensors": True}, log=self_.line.emit)
                    self_.done.emit(bool(ok))
                except Exception as e:
                    self_.line.emit(f"❌ {e}"); self_.done.emit(False)

        def _finish(ok):
            self.conv_btn.setEnabled(True)
            self.out.appendPlainText(tr("\n✅ Terminé." if ok else "\n❌ Échec.",
                                        "\n✅ Done." if ok else "\n❌ Failed."))

        self._cw = _CW()
        self._cw.line.connect(self.out.appendPlainText)
        self._cw.done.connect(_finish)
        self._cw.start()

    def _collect(self, path):
        lines = []

        def sec(title):
            lines.append(""); lines.append(f"── {title} " + "─" * max(0, 40 - len(title)))

        sec(tr("Fichier", "File"))
        lines.append(f"{tr('nom', 'name'):16}: {os.path.basename(path)}")
        lines.append(f"{'format':16}: {os.path.splitext(path)[1]}")
        lines.append(f"{tr('taille', 'size'):16}: {os.path.getsize(path) / (1024*1024):.2f} MB")

        try:
            from src.core.model_convert import _load_state
            state = _load_state(path)
            if isinstance(state, dict) and state:
                import collections
                total = 0; dtypes = collections.Counter()
                for v in state.values():
                    try:
                        total += int(v.numel()); dtypes[str(v.dtype).replace("torch.", "")] += 1
                    except Exception:
                        pass
                sec(tr("Poids", "Weights"))
                lines.append(f"{tr('tenseurs', 'tensors'):16}: {len(state)}")
                lines.append(f"{tr('valeurs', 'values'):16}: {total:,} ({total/1e6:.2f} M) "
                             + tr("— tous les tenseurs enregistrés, buffers inclus",
                                  "— every saved tensor, buffers included"))
                lines.append(f"{'dtypes':16}: " + ", ".join(f"{d}×{c}" for d, c in dtypes.most_common()))
                try:
                    from qt_ui.tabs.tools_pages.quick_upscale import _detect_multiscale
                    ms = _detect_multiscale(path)
                    if ms and len(ms) > 1:
                        lines.append(f"{tr('multi-échelle', 'multi-scale'):16}: {' / '.join(str(s)+'×' for s in ms)}")
                except Exception:
                    pass
        except Exception as e:
            lines.append(tr(f"(lecture state_dict impossible : {e})", f"(could not read state_dict: {e})"))

        try:
            from src.core import arch_infer
            _ai, _kw = arch_infer.describe(path)
        except Exception:
            _ai, _kw = None, None
        if _ai:
            sec(tr("Architecture (traiNNer-redux)", "Architecture (traiNNer-redux)"))
            lines.append(f"{tr('architecture', 'architecture'):16}: {_ai.upper()}")
            lines.append(f"{'scale':16}: {_kw.get('scale')}×")
            for k, v in _kw.items():
                if k != "scale":
                    lines.append(f"{k:16}: {v}")
            if _ai == "heart":
                lines.append(tr(f"{'window_size':16}: 32 (non stocké dans le fichier — défaut supposé)",
                                f"{'window_size':16}: 32 (not stored in the file — default assumed)"))
            return "\n".join(lines).lstrip()
        try:
            from spandrel import ModelLoader
            try:
                import spandrel_extra_arches
                spandrel_extra_arches.install()
            except Exception:
                pass
            d = ModelLoader().load_from_file(path)
            sec(tr("Architecture (spandrel)", "Architecture (spandrel)"))
            lines.append(f"{tr('architecture', 'architecture'):16}: {getattr(getattr(d, 'architecture', None), 'name', '?')}")
            lines.append(f"{'scale':16}: {getattr(d, 'scale', '?')}×")
            try:
                _np = sum(p.numel() for p in d.model.parameters())
                lines.append(f"{tr('paramètres', 'parameters'):16}: {_np:,} ({_np/1e6:.2f} M)")
            except Exception:
                pass
            lines.append(f"{tr('canaux entrée', 'input channels'):16}: {getattr(d, 'input_channels', '?')}")
            lines.append(f"{tr('canaux sortie', 'output channels'):16}: {getattr(d, 'output_channels', '?')}")
            if getattr(d, "purpose", None):
                lines.append(f"{tr('usage', 'purpose'):16}: {d.purpose}")
            if getattr(d, "tags", None):
                lines.append(f"{'tags':16}: {', '.join(map(str, d.tags))}")
            lines.append(f"{tr('FP16 supporté', 'FP16 supported'):16}: {getattr(d, 'supports_half', '?')}")
            lines.append(f"{tr('BF16 supporté', 'BF16 supported'):16}: {getattr(d, 'supports_bfloat16', '?')}")
            if getattr(d, "size_requirements", None) is not None:
                lines.append(f"{tr('taille requise', 'size requirement'):16}: {d.size_requirements}")
        except Exception as e:
            sec(tr("Architecture", "Architecture"))
            _why = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            lines.append(tr(f"(spandrel n'a pas reconnu l'architecture : {_why})", f"(spandrel did not recognize the architecture: {_why})"))
            try:
                from src.core.model_export import detect_model_format
                for k, v in detect_model_format(path).items():
                    sv = str(v)
                    lines.append(f"{k:16}: {sv[:160] + ' …' if len(sv) > 160 else sv}")
            except Exception:
                pass

        return "\n".join(lines).lstrip()
