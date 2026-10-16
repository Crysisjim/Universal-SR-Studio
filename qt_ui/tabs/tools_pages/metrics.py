"""
metrics.py — Tools › Métriques (PySide6). PSNR / SSIM between a GT folder and an SR
folder (matching filenames). Self-contained (numpy + PIL; SSIM uses scipy if present).
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

_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _psnr(a, b):
    """PSNR in dB; None for identical images (infinite — left out of the average instead
    of counting an arbitrary 99 dB)."""
    import numpy as np
    mse = float(np.mean((a - b) ** 2))
    return None if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def _box(x, r=3):
    """Mean over a (2r+1)^2 window, reflect-padded (integral image: no SciPy needed —
    SSIM used to vanish silently when SciPy was missing)."""
    import numpy as np
    k = 2 * r + 1
    xp = np.pad(x, r, mode="reflect")
    c = np.cumsum(np.cumsum(xp, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _ssim(a, b):
    import numpy as np
    a = a.mean(axis=2); b = b.mean(axis=2)
    mu_a, mu_b = _box(a), _box(b)
    va = _box(a * a) - mu_a ** 2
    vb = _box(b * b) - mu_b ** 2
    vab = _box(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(np.clip(s, 0, 1).mean())


def _load01(path):
    """RGB float in [0, 1], keeping 16-bit PNG/TIFF precision."""
    import numpy as np
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, np.float32) / 65535.0
        return np.repeat(arr[..., None], 3, axis=2)
    from src.core.image_io import read_float01
    return read_float01(path)


def _match_sr(gt_files, sr_dir):
    """GT name -> SR path: same file name, else same stem with any extension, else the
    single SR file starting with "<stem>_" (Quick Upscale adds a suffix: name_UP.png)."""
    sr = [f for f in os.listdir(sr_dir) if os.path.splitext(f)[1].lower() in _EXT]
    by_stem = {}
    for f in sr:
        by_stem.setdefault(os.path.splitext(f)[0], []).append(f)
    out = {}
    for g in gt_files:
        stem = os.path.splitext(g)[0]
        if g in sr:
            out[g] = g
        elif len(by_stem.get(stem, [])) == 1:
            out[g] = by_stem[stem][0]
        else:
            cand = [f for f in sr if os.path.splitext(f)[0].startswith(stem + "_")]
            if len(cand) == 1:
                out[g] = cand[0]
    return out


class _Worker(QThread):
    line = Signal(str)
    done = Signal()

    def __init__(self, gt, sr):
        super().__init__(); self.gt, self.sr = gt, sr

    def run(self):
        try:
            self._run()
        except Exception as e:  # always give the button back
            self.line.emit(f"❌ {e}")
        finally:
            self.done.emit()

    def _run(self):
        import numpy as np
        from PIL import Image
        files = [f for f in sorted(os.listdir(self.gt))
                 if os.path.splitext(f)[1].lower() in _EXT]
        if not files:
            self.line.emit("❌ " + tr("Aucune image dans le dossier GT.", "No image in the GT folder.")); return
        pairs = _match_sr(files, self.sr)
        psnrs, ssims, same = [], [], 0
        for f in files:
            if f not in pairs:
                self.line.emit(tr(f"… {f} : pas d'équivalent SR", f"… {f}: no matching SR")); continue
            sp = os.path.join(self.sr, pairs[f])
            try:
                a = _load01(os.path.join(self.gt, f))
                b = _load01(sp)
                if a.shape != b.shape:
                    self.line.emit(tr(f"   {f} : tailles différentes ({b.shape[1]}x{b.shape[0]} → "
                                      f"{a.shape[1]}x{a.shape[0]}), SR redimensionné",
                                      f"   {f}: sizes differ ({b.shape[1]}x{b.shape[0]} → "
                                      f"{a.shape[1]}x{a.shape[0]}), SR resized"))
                    b = np.asarray(Image.fromarray((np.clip(b, 0, 1) * 255).astype("uint8")).resize(
                        (a.shape[1], a.shape[0]), Image.BICUBIC), np.float32) / 255.0
                p = _psnr(a, b); s = _ssim(a, b)
                ssims.append(s)
                if p is None:
                    same += 1
                    self.line.emit(f"{f:40} PSNR   ∞ ({tr('identiques', 'identical')})  SSIM {s:.4f}")
                else:
                    psnrs.append(p)
                    self.line.emit(f"{f:40} PSNR {p:6.2f} dB  SSIM {s:.4f}")
            except Exception as e:
                self.line.emit(f"❌ {f} : {e}")
        if psnrs or same:
            self.line.emit("─" * 50)
            avg = f"PSNR {sum(psnrs) / len(psnrs):.2f} dB" if psnrs else "PSNR ∞"
            extra = tr(f"  ({same} paire(s) identique(s) hors moyenne PSNR)",
                       f"  ({same} identical pair(s) left out of the PSNR mean)") if same and psnrs else ""
            self.line.emit(tr("MOYENNE  ", "AVERAGE  ") + avg +
                           (f"  SSIM {sum(ssims) / len(ssims):.4f}" if ssims else "") + extra)


class MetricsPage(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.worker = None
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        t = QLabel(tr("Métriques (PSNR / SSIM)", "Metrics (PSNR / SSIM)")); t.setObjectName("title")
        root.addWidget(t)
        root.addWidget(QLabel(tr("Compare un dossier GT (référence) et un dossier SR (résultat).", "Compares a GT folder (reference) and an SR folder (result).")))
        self.gt = self._row(root, tr("Dossier GT :", "GT Folder:"),
                            tr("Images de référence (vérité terrain, haute qualité).\n"
                            "Chaque résultat SR est comparé à l'image GT du même nom de fichier.",
                            "Reference images (ground truth, high quality).\n"
                            "Each SR result is compared to the GT image of the same file name."))
        self.sr = self._row(root, tr("Dossier SR :", "SR Folder:"),
                            tr("Images produites par le modèle (super-résolues).\n"
                            "Même nom que le GT (extension libre, suffixe Quick Upscale « _UP » accepté).",
                            "Images produced by the model (super-resolved).\n"
                            "Same name as the GT (any extension, Quick Upscale \"_UP\" suffix accepted)."))
        self.btn = QPushButton(tr("Calculer", "Calculate")); self.btn.clicked.connect(self._run)
        self.btn.setToolTip(tr("Calcule PSNR et SSIM image par image, puis la moyenne.\n"
                            "PSNR (dB) : plus haut = plus proche de la référence (≈30+ bon, 40+ excellent).\n"
                            "SSIM (0→1) : similarité de structure perçue, plus proche de 1 = mieux.\n"
                            "Si les tailles diffèrent, le SR est redimensionné à la taille du GT.", "Computes PSNR and SSIM image by image, then the average.\nPSNR (dB): higher = closer to the reference (≈30+ good, 40+ excellent).\nSSIM (0→1): perceived structural similarity, closer to 1 = better.\nIf sizes differ, the SR is resized to the GT size."))
        root.addWidget(self.btn)
        self.out = QPlainTextEdit(); self.out.setReadOnly(True)
        self.out.setFont(QFont("Consolas", 10)); self.out.setMaximumBlockCount(10000)
        root.addWidget(self.out, 1)

    def _row(self, parent, label, tip=""):
        row = QHBoxLayout(); row.addWidget(QLabel(label))
        e = QLineEdit(); row.addWidget(e, 1)
        if tip:
            e.setToolTip(tip)
        b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedWidth(40)
        b.setToolTip(tr("Parcourir pour choisir le dossier.", "Browse to choose the folder."))
        b.clicked.connect(lambda _=False, ed=e: ed.setText(
            QFileDialog.getExistingDirectory(self, tr("Dossier", "Folder")) or ed.text()))
        row.addWidget(b); parent.addLayout(row)
        return e

    def _run(self):
        gt, sr = self.gt.text().strip(), self.sr.text().strip()
        if not os.path.isdir(gt) or not os.path.isdir(sr):
            self.out.setPlainText(tr("❌ Dossiers GT/SR invalides.", "❌ Invalid GT/SR folders.")); return
        self.out.clear(); self.btn.setEnabled(False); self.btn.setText(tr("Calcul…", "Computing…"))
        self.worker = _Worker(gt, sr)
        self.worker.line.connect(self.out.appendPlainText)
        self.worker.done.connect(self._done)
        self.worker.start()

    def _done(self):
        self.btn.setEnabled(True); self.btn.setText(tr("Calculer", "Calculate")); self.worker = None
