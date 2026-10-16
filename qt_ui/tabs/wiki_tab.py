"""
wiki_tab.py — Wiki shortcut tab (parity with the CustomTkinter "📖 Wiki" tab).
Shows the project wiki link and opens it in the default browser on demand.
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices

from qt_ui.i18n import tr

_WIKI_URL = "https://github.com/Crysisjim/Universal-SR-Studio/wiki"


class WikiTab(QWidget):
    def __init__(self, settings, **kwargs):
        super().__init__()
        self.settings = settings
        v = QVBoxLayout(self)
        v.setContentsMargins(40, 40, 40, 40)
        v.setSpacing(14)
        v.addStretch(1)

        title = QLabel("📖 " + tr("Wiki & Documentation", "Wiki & Documentation"))
        title.setObjectName("title"); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-family:"Segoe UI Emoji"; font-size:20px; font-weight:700;')
        v.addWidget(title)

        desc = QLabel(tr("Guides, tutoriels et documentation complète du projet sont sur le wiki GitHub.",
                         "Guides, tutorials and full project documentation live on the GitHub wiki."))
        desc.setAlignment(Qt.AlignCenter); desc.setWordWrap(True); desc.setStyleSheet("color:#888;")
        v.addWidget(desc)

        link = QLabel(f'<a href="{_WIKI_URL}" style="color:#3498db;">{_WIKI_URL}</a>')
        link.setAlignment(Qt.AlignCenter); link.setOpenExternalLinks(True)
        link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        v.addWidget(link)

        btn = QPushButton("🌐 " + tr("Ouvrir le Wiki", "Open the Wiki"))
        btn.setFixedWidth(220); btn.setFixedHeight(38)
        btn.setStyleSheet('background:#2ecc71; color:#fff; border:none; border-radius:8px;'
                          ' font-weight:700; font-family:"Segoe UI Emoji";')
        btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(_WIKI_URL)))
        bw = QWidget(); from PySide6.QtWidgets import QHBoxLayout
        bh = QHBoxLayout(bw); bh.setContentsMargins(0, 0, 0, 0)
        bh.addStretch(1); bh.addWidget(btn); bh.addStretch(1)
        v.addWidget(bw)

        v.addStretch(2)
