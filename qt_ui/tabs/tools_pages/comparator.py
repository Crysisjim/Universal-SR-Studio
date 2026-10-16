"""
comparator.py — Tools › Comparateur (PySide6). A/B image slider comparison.
Loads two images (avant/après), composites left=A right=B at the divider position.
Pure PIL + Qt. Faithful intent of tab_tools.create_page_comparator.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QSlider,
    QFileDialog, QSizePolicy, QCheckBox,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

from qt_ui.i18n import tr


def _pil_to_pixmap(im):
    im = im.convert("RGB")
    qim = QImage(im.tobytes("raw", "RGB"), im.width, im.height, im.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qim.copy())


class ComparatorPage(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._a = None
        self._b = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        t = QLabel(tr("Comparateur", "Comparator")); t.setObjectName("title")
        root.addWidget(t)
        root.addWidget(QLabel(tr("Glisse le curseur : gauche = Avant, droite = Après.", "Drag the slider: left = Before, right = After.")))

        self._a_edit = self._row(root, tr("Image Avant (A) :", "Before Image (A):"), self._load_a,
                                 tr("Image de gauche (ex. l'original basse résolution).", "Left image (e.g. the low-resolution original)."))
        self._b_edit = self._row(root, tr("Image Après (B) :", "After Image (B):"), self._load_b,
                                 tr("Image de droite (ex. le résultat super-résolu).\n"
                                 "Si les tailles diffèrent, la plus petite est agrandie (bicubique) à la taille de l'autre.",
                                 "Right image (e.g. the super-resolved result).\n"
                                 "If the sizes differ, the smaller one is enlarged (bicubic) to the other's size."))

        self.auto_load = QCheckBox(tr("Auto-charger dernière image traitée", "Auto-load last processed image"))
        self.auto_load.setToolTip(tr(
            "Charge la dernière paire avant/après produite par Quick Upscale, puis la recharge "
            "à chaque retour sur cette page si une nouvelle paire a été produite.",
            "Loads the last before/after pair produced by Quick Upscale, and reloads it each "
            "time you come back to this page if a new pair was produced."))
        self.auto_load.stateChanged.connect(self._on_auto_load_toggled)
        root.addWidget(self.auto_load)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100); self.slider.setValue(50)
        self.slider.valueChanged.connect(self._render)
        self.slider.setToolTip(tr("Position du séparateur : à gauche on voit l'image Avant (A),\n"
                               "à droite l'image Après (B). Glisse pour comparer les détails.", "Separator position: on the left you see the Before image (A),\non the right the After image (B). Drag to compare details."))
        root.addWidget(self.slider)

        self.view = QLabel(tr("Charge deux images.", "Load two images."))
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setStyleSheet("QLabel{background:#0b0d12;border-radius:10px;}")
        root.addWidget(self.view, 1)

    def _row(self, parent, label, slot, tip=""):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        e = QLineEdit(); row.addWidget(e, 1)
        # B01: a path typed or pasted in the field is loaded too
        e.editingFinished.connect(lambda e=e: self._load_typed(e))
        if tip:
            e.setToolTip(tip)
        b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedWidth(40)
        b.setToolTip(tr("Parcourir pour choisir l'image.", "Browse to choose the image."))
        b.clicked.connect(slot)
        row.addWidget(b)
        parent.addLayout(row)
        return e

    def _pick(self):
        p, _ = QFileDialog.getOpenFileName(self, "Image", "",
                                           "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        return p

    def _open(self, p):
        from PIL import Image
        try:
            return Image.open(p).convert("RGB")
        except Exception as e:
            self.view.setText(tr(f"Impossible d'ouvrir {os.path.basename(p)} : {e}",
                                 f"Cannot open {os.path.basename(p)}: {e}"))
            return None

    def _load_typed(self, edit):
        p = edit.text().strip().strip('"')
        if not p or not os.path.isfile(p):
            return
        im = self._open(p)
        if im is None:
            return
        if edit is self._a_edit:
            self._a = im
        else:
            self._b = im
        self._render()

    def _on_auto_load_toggled(self, *_):
        if not self.auto_load.isChecked():
            return
        before = self.settings.get("last_upscale_before", "") or ""
        after = self.settings.get("last_upscale_after", "") or ""
        if (before, after) == getattr(self, "_auto_pair", None):
            return
        self._auto_pair = (before, after)
        if before and os.path.isfile(before):
            self._a = self._open(before); self._a_edit.setText(before)
        if after and os.path.isfile(after):
            self._b = self._open(after); self._b_edit.setText(after)
        self._render()

    def showEvent(self, ev):
        # B02: the newest Quick Upscale pair is picked up when the page is shown again
        super().showEvent(ev)
        self._on_auto_load_toggled()

    def _load_a(self):
        p = self._pick()
        if p:
            from PIL import Image
            self._a = Image.open(p).convert("RGB"); self._a_edit.setText(p); self._render()

    def _load_b(self):
        p = self._pick()
        if p:
            from PIL import Image
            self._b = Image.open(p).convert("RGB"); self._b_edit.setText(p); self._render()

    def _render(self):
        if self._a is None and self._b is None:
            return
        from PIL import Image
        a = self._a or self._b
        b = self._b or self._a
        # align sizes: the smaller image is enlarged to the larger one. Shrinking the SR
        # result to the LQ size hid exactly the details being compared.
        if b.size != a.size:
            if a.size[0] * a.size[1] >= b.size[0] * b.size[1]:
                b = b.resize(a.size, Image.BICUBIC)
            else:
                a = a.resize(b.size, Image.BICUBIC)
        w, h = a.size
        x = int(w * self.slider.value() / 100)
        comp = a.copy()
        if x < w:
            comp.paste(b.crop((x, 0, w, h)), (x, 0))
        pm = _pil_to_pixmap(comp)
        self.view.setPixmap(pm.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._render()
