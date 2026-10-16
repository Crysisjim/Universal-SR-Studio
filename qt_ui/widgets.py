"""
widgets.py — small reusable PySide6 building blocks for the Qt rewrite.
Keeps tab code declarative and consistent (cards, labelled sliders, sections).
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, 
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGroupBox,
)
from PySide6.QtCore import Qt, Signal, QObject, QEvent


class _ComboClickFilter(QObject):
    """Open an editable QComboBox's popup when its line-edit is clicked (so the whole
    control is clickable, like a non-editable combo) while still allowing typing."""
    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.MouseButtonPress:
            combo = obj.parent()
            if combo is not None and hasattr(combo, "showPopup"):
                combo.showPopup()
                return True
        return False


def combo_open_on_click(combo) -> None:
    """Make an *editable* combobox drop down on a click anywhere on it (not only the arrow)."""
    le = combo.lineEdit()
    if le is None:
        return
    flt = _ComboClickFilter(combo)   # parented to the combo → lives as long as it
    combo._click_filter = flt        # keep a ref so it isn't garbage-collected
    le.installEventFilter(flt)


class _SameHeight(QObject):
    """Gives a row of widgets the height of its tallest one when the row is first shown
    (the stylesheet padding is only known once the widgets are polished in the window)."""

    def __init__(self, widgets):
        super().__init__(widgets[0])
        self._w = list(widgets); self._done = False
        for w in self._w:
            w.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Show and not self._done:
            self._done = True
            h = max(w.sizeHint().height() for w in self._w)
            for w in self._w:
                w.setMinimumHeight(h)
        return False


def same_height(*widgets) -> None:
    """Same height for buttons of one row (see _SameHeight)."""
    _SameHeight(widgets)


def card(title: str | None = None) -> QGroupBox:
    """A titled card/group container."""
    box = QGroupBox(title or "")
    return box


def hsep() -> QFrame:
    f = QFrame()
    f.setObjectName("sep")
    f.setFrameShape(QFrame.HLine)
    return f



def provider_label(key: str) -> str:
    """Display name of an AI provider key. The keys ("OpenRouter (Gratuit)"…) are stored in
    the settings and used as dict keys everywhere, so only the shown text is translated."""
    from qt_ui.i18n import tr
    return str(key).replace("(Gratuit)", tr("(Gratuit)", "(Free)"))


class KeyedCombo(QComboBox):
    """Combo whose items show a translated label but whose currentText() / setCurrentText()
    and textChanged signal work with the internal key (item data)."""
    keyChanged = Signal(str)

    def __init__(self, keys, label=provider_label, parent=None):
        super().__init__(parent)
        for k in keys:
            self.addItem(label(k), k)
        self.currentIndexChanged.connect(lambda _i: self.keyChanged.emit(self.currentText()))

    def currentText(self) -> str:  # noqa: N802 (Qt naming)
        d = self.currentData()
        return str(d) if d is not None else super().currentText()

    def setCurrentText(self, key: str) -> None:  # noqa: N802
        i = self.findData(key)
        if i < 0:
            i = self.findText(key)
        if i >= 0:
            self.setCurrentIndex(i)

class LabeledSlider(QWidget):
    """A slider with a 'Label : value' caption above it, a tooltip, and a
    valueChangedF(float) signal. Works in float space via an integer slider."""
    valueChangedF = Signal(float)

    def __init__(self, label: str, lo: float, hi: float, value: float,
                 step: float = 0.01, fmt: str = "{:.2f}", hint: str = "",
                 parent=None):
        super().__init__(parent)
        self._label = label
        self._lo, self._hi, self._step, self._fmt = lo, hi, step, fmt
        self._steps = max(1, round((hi - lo) / step))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(2)
        self._cap = QLabel()
        lay.addWidget(self._cap)
        self._sld = QSlider(Qt.Horizontal)
        self._sld.setMinimum(0)
        self._sld.setMaximum(self._steps)
        self._sld.setValue(self._to_int(value))
        self._sld.valueChanged.connect(self._on_change)
        lay.addWidget(self._sld)

        if hint:
            try:
                from qt_ui.i18n import rich_tip
                hint = rich_tip(hint)
            except Exception:
                pass
            self.setToolTip(hint)
            self._cap.setToolTip(hint)
            self._sld.setToolTip(hint)
        self._update_caption(value)

    def _to_int(self, v: float) -> int:
        return max(0, min(self._steps, round((v - self._lo) / self._step)))

    def _to_float(self, i: int) -> float:
        return self._lo + i * self._step

    def _on_change(self, i: int):
        v = self._to_float(i)
        self._update_caption(v)
        self.valueChangedF.emit(v)

    def _update_caption(self, v: float):
        self._cap.setText(f"{self._label} : {self._fmt.format(v)}")

    def value(self) -> float:
        return self._to_float(self._sld.value())

    def setValue(self, v: float):
        self._sld.setValue(self._to_int(v))
        self._update_caption(v)
