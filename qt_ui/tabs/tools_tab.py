"""
tools_tab.py — Tools tab (PySide6 port) — mirrors the CustomTkinter "Outils" tab:
a left sidebar nav + a swapping page area (QStackedWidget). Same page list + order
as tab_tools.py. Post Processing is the real ported page; the rest are placeholders
to port one by one.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QStackedWidget, QFrame,
)
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QFont

from qt_ui.i18n import tr

# (label, key) — identical order to tab_tools.create_nav_btn calls.
_PAGES = [
    ("🔎 Quick Upscale", "upscale"),
    (tr("📊 Comparateur", "📊 Comparator"), "comp"),
    (tr("⚡ Générateur LQ", "⚡ LQ Generator"), "gen"),
    (tr("🎞 Extraction Frames", "🎞 Frame Extraction"), "frames"),
    (tr("🔄 Convertisseur de modèle", "🔄 Model Converter"), "conv"),
    (tr("💾 Créateur LMDB", "💾 LMDB Creator"), "lmdb"),
    (tr("🧐 Check Dataset", "🧐 Check Dataset"), "chk"),
    (tr("📏 Métriques", "📏 Metrics"), "met"),
    (tr("ℹ Info Modèle", "ℹ Model Info"), "model_info"),
    (tr("📜 Historique", "📜 History"), "history"),
    (tr("♻ Reprendre Entraînement", "♻ Resume Training"), "resume"),
    (tr("📦 Publier Modèle", "📦 Publish Model"), "export"),
    ("📈 Benchmark", "bench"),
    (tr("⚗ Post Processing", "⚗ Post Processing"), "postproc"),
    (tr("📥 Importer Config", "📥 Import Config"), "import_cfg"),
]


class _SpinnerBadge(QWidget):
    """Sidebar-nav "running tool" badge: a small animated rotating arc next to a
    tool's nav entry while it has a background job running (percentage shown in
    the center when the page reports one via toolProgress; just a bare spinner
    when it only reports toolBusy/toolFinished, e.g. a single-shot conversion
    with no per-item progress). Turns into a static filled circle (green=done,
    red=failed) once finished, and stays that way — even if the user is already
    on that page when it finishes — until the page is actually (re-)opened,
    which acknowledges and clears it. Fully hidden while idle.
    """
    _SIZE = 22

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self._state = "idle"   # idle | running | done_ok | done_fail
        self._percent = None   # int 0-99, or None while indeterminate (toolBusy)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(45)
        self._timer.timeout.connect(self._tick)
        self.setVisible(False)
        self.setToolTip("")

    def _tick(self):
        self._angle = (self._angle + 18) % 360
        self.update()

    def set_running(self, percent=None):
        self._state = "running"
        self._percent = None if percent is None else max(0, min(99, int(percent)))
        self.setVisible(True)
        if not self._timer.isActive():
            self._timer.start()
        self.update()

    def set_done(self, ok: bool):
        self._state = "done_ok" if ok else "done_fail"
        self._timer.stop()
        self.setVisible(True)
        self.update()

    def clear(self):
        self._state = "idle"
        self._percent = None
        self._timer.stop()
        self.setVisible(False)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self._SIZE - 5
        rect = QRectF(2.5, 2.5, r, r)
        if self._state == "running":
            p.setPen(QPen(QColor(255, 255, 255, 45), 2.4))
            p.drawEllipse(rect)
            p.setPen(QPen(QColor("#3b82f6"), 2.4, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(rect, int(-self._angle * 16), int(100 * 16))
            if self._percent is not None:
                p.setPen(QColor("white"))
                f = QFont(); f.setPixelSize(8); f.setBold(True); p.setFont(f)
                p.drawText(self.rect(), Qt.AlignCenter, str(self._percent))
        elif self._state in ("done_ok", "done_fail"):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#22c55e") if self._state == "done_ok" else QColor("#ef4444"))
            p.drawEllipse(rect)
            p.setPen(QColor("white"))
            f = QFont(); f.setPixelSize(10); f.setBold(True); p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, "✓" if self._state == "done_ok" else "!")
        p.end()


class ToolsTab(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── left sidebar nav ──
        nav = QFrame()
        nav.setObjectName("card")
        nav.setFixedWidth(210)
        nl = QVBoxLayout(nav)
        nl.setContentsMargins(8, 12, 8, 12)
        nl.setSpacing(4)

        self.stack = QStackedWidget()
        self._btns = []
        self._badges = {}       # key -> _SpinnerBadge
        self._badge_done = {}   # key -> True once finished, until the page is opened

        for label, key in _PAGES:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(4)
            b = QPushButton(label)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self._show(k))
            rl.addWidget(b, 1)
            badge = _SpinnerBadge()
            rl.addWidget(badge, 0)
            nl.addWidget(row)
            self._btns.append((key, b))
            self._badges[key] = badge
            page = self._page(label, key)
            self._wire_badge(key, page)
            self.stack.addWidget(page)
        nl.addStretch(1)

        root.addWidget(nav)
        root.addWidget(self.stack, 1)

        # #nav buttons are styled by the global theme QSS (theme-aware text + accent highlight).

        # Open on the last-used page (default Quick Upscale), not always Post Processing.
        last = self.settings.get("tools_last_page", "upscale")
        if not any(k == last for k, _ in self._btns):
            last = "upscale"
        self._show(last)

    def _page(self, label, key) -> QWidget:
        if key == "postproc":
            try:
                from qt_ui.tabs.post_process import PostProcessTab
                return PostProcessTab(self.settings)
            except Exception as e:
                return self._placeholder(label, f"(post-process indisponible : {e})")
        _real = {
            "conv": ("qt_ui.tabs.tools_pages.converter", "ConverterPage"),
            "comp": ("qt_ui.tabs.tools_pages.comparator", "ComparatorPage"),
            "model_info": ("qt_ui.tabs.tools_pages.model_info", "ModelInfoPage"),
            "chk": ("qt_ui.tabs.tools_pages.check_dataset", "CheckDatasetPage"),
            "met": ("qt_ui.tabs.tools_pages.metrics", "MetricsPage"),
            "upscale": ("qt_ui.tabs.tools_pages.quick_upscale", "QuickUpscalePage"),
            "gen": ("qt_ui.tabs.tools_pages.gen_lq", "GenLQPage"),
            "frames": ("qt_ui.tabs.tools_pages.frame_extractor", "FrameExtractorPage"),
            "export": ("qt_ui.tabs.tools_pages.publish", "PublishPage"),
            "history": ("qt_ui.tabs.tools_pages.tools_extra", "HistoryPage"),
            "resume": ("qt_ui.tabs.tools_pages.tools_extra", "ResumePage"),
            "lmdb": ("qt_ui.tabs.tools_pages.tools_extra", "LMDBPage"),
            "bench": ("qt_ui.tabs.tools_pages.tools_extra", "BenchmarkPage"),
            "import_cfg": ("qt_ui.tabs.tools_pages.import_config", "ImportConfigPage"),
        }
        if key in _real:
            mod_name, cls = _real[key]
            try:
                mod = __import__(mod_name, fromlist=[cls])
                return getattr(mod, cls)(self.settings)
            except Exception as e:
                return self._placeholder(label, f"(indisponible : {e})")
        return self._placeholder(label, tr(
            "Page à porter depuis la version CustomTkinter.\nLe backend (src/core/) est déjà réutilisable.",
            "Page to be ported from the CustomTkinter version.\nThe backend (src/core/) is already reusable."))

    def _placeholder(self, label, msg) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(24, 24, 24, 24)
        t = QLabel(label); t.setObjectName("title")
        v.addWidget(t)
        s = QLabel(msg); s.setObjectName("subtitle")
        v.addWidget(s)
        v.addStretch(1)
        return w

    def _show(self, key):
        for i, (k, b) in enumerate(self._btns):
            b.setChecked(k == key)
            if k == key:
                self.stack.setCurrentIndex(i)
        try:
            self.settings.set("tools_last_page", key)
        except Exception:
            pass
        # Opening a page acknowledges its finished-run badge (green/red) — a
        # badge for a job still IN PROGRESS is left alone (it's not "done" yet).
        if self._badge_done.get(key):
            self._badge_done[key] = False
            b = self._badges.get(key)
            if b:
                b.clear()

    def _wire_badge(self, key, page):
        """Hook a tool page's optional toolProgress/toolBusy/toolFinished signals
        to the sidebar badge for that page's nav entry. Pages that don't define
        these signals are simply skipped — this is opt-in per page, not a
        requirement. Current pages wired: Quick Upscale, LQ Generator, Post
        Processing, Model Converter, LMDB Creator, Benchmark — the ones with
        genuinely long-running background jobs; instant/lightweight pages
        (Model Info, History, Check Dataset…) don't need it."""
        badge = self._badges.get(key)
        if badge is None:
            return
        if hasattr(page, "toolProgress"):
            page.toolProgress.connect(lambda p, k=key: self._set_badge_progress(k, p))
        if hasattr(page, "toolBusy"):
            page.toolBusy.connect(lambda k=key: self._set_badge_busy(k))
        if hasattr(page, "toolFinished"):
            page.toolFinished.connect(lambda ok, k=key: self._set_badge_done(k, ok))

    def _set_badge_progress(self, key, percent):
        badge = self._badges.get(key)
        if badge is None:
            return
        self._badge_done[key] = False
        badge.set_running(percent)

    def _set_badge_busy(self, key):
        badge = self._badges.get(key)
        if badge is None:
            return
        self._badge_done[key] = False
        badge.set_running(None)

    def _set_badge_done(self, key, ok):
        badge = self._badges.get(key)
        if badge is None:
            return
        # Stays shown even if the user is currently on that page when the run
        # finishes — per spec, it only clears when the page is (re-)opened.
        self._badge_done[key] = True
        badge.set_done(bool(ok))
