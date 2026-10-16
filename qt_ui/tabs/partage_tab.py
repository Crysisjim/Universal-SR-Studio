"""
partage_tab.py — Partage / Distributed (PySide6). Basic master view: manage slave
nodes (add/remove, persisted in settings), auto-discover on the LAN and benchmark
them, via src.core.distributed_client (discover_slaves / SlaveClient). Network calls
run off the UI thread.
"""
from __future__ import annotations

import socket
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QListWidget,
    QListWidgetItem, QPlainTextEdit, QGroupBox,
)
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont

from qt_ui.i18n import tr


def _local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class _DiscoverWorker(QThread):
    found = Signal(list); error = Signal(str)

    def run(self):
        try:
            from src.core.distributed_client import discover_slaves
            self.found.emit(list(discover_slaves() or []))
        except Exception as e:
            self.error.emit(str(e))


class _PingWorker(QThread):
    """Ping each slave host and report latency/OK-fail — CTk's "Test Sync" button
    had no equivalent at all in the Qt port."""
    line = Signal(str); done = Signal()

    def __init__(self, hosts):
        super().__init__(); self.hosts = hosts

    def run(self):
        try:
            from src.core.distributed_client import SlaveClient
        except Exception as e:
            self.line.emit(f"❌ {e}"); self.done.emit(); return
        for h in self.hosts:
            host, _, port = h.partition(":")
            t0 = time.monotonic()
            try:
                c = SlaveClient(host, int(port or 5001))
                res = c.ping()
                dt_ms = (time.monotonic() - t0) * 1000
                if res.get("ok"):
                    extra = " ".join(f"{k}={res[k]}" for k in ("name", "hostname", "platform", "version") if res.get(k))
                    self.line.emit(f"🟢 {h} — {dt_ms:.0f} ms  {extra}")
                else:
                    self.line.emit(f"🔴 {h} — {dt_ms:.0f} ms — {res.get('error', 'ping failed')}")
            except Exception as e:
                dt_ms = (time.monotonic() - t0) * 1000
                self.line.emit(f"🔴 {h} — {dt_ms:.0f} ms — {e}")
        self.done.emit()


class _BenchWorker(QThread):
    line = Signal(str); done = Signal()

    def __init__(self, hosts):
        super().__init__(); self.hosts = hosts

    def run(self):
        try:
            from src.core.distributed_client import SlaveClient
        except Exception as e:
            self.line.emit(f"❌ {e}"); self.done.emit(); return
        for h in self.hosts:
            try:
                host, _, port = h.partition(":")
                c = SlaveClient(host, int(port or 5001))
                res = c.benchmark(seconds=5)
                # F48: the client returns {"ok": false, ...} on failure
                ok = not (isinstance(res, dict) and res.get("ok") is False)
                self.line.emit(f"{'✅' if ok else '❌'} {h} : {res}")
            except Exception as e:
                self.line.emit(f"❌ {h} : {e}")
        self.done.emit()


class PartageTab(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.worker = None
        root = QVBoxLayout(self); root.setContentsMargins(16, 16, 16, 16); root.setSpacing(10)
        t = QLabel(tr("🌐 Partage — Nœuds du réseau","🌐 Sharing — Network nodes")); t.setObjectName("title"); root.addWidget(t)
        # F48: this page manages, discovers, pings and benchmarks nodes; it does not
        # launch a distributed training
        root.addWidget(QLabel(tr("Gère les machines « slave » du réseau local : ajout, découverte, ping et benchmark.\n"
                                 "Le lancement d'un entraînement réparti n'est pas encore disponible ici.",
                                 "Manages the local-network \"slave\" machines: add, discover, ping and benchmark.\n"
                                 "Launching a distributed training is not available here yet.")))

        mb = QGroupBox(tr("Master (cette machine)","Master (this machine)")); mh = QHBoxLayout(mb)
        mh.addWidget(QLabel(tr("IP locale :","Local IP:")))
        ip = QLabel(_local_ip()); ip.setStyleSheet("color:#2ecc71; font-weight:700;"); mh.addWidget(ip); mh.addStretch(1)
        root.addWidget(mb)

        sb = QGroupBox(tr("Nœuds esclaves","Slave nodes")); sv = QVBoxLayout(sb)
        add = QHBoxLayout(); add.addWidget(QLabel(tr("Hôte:port :", "Host:port:")))
        self.host = QLineEdit(); self.host.setPlaceholderText("192.168.1.20:5001"); add.addWidget(self.host, 1)
        b = QPushButton(tr("➕ Ajouter","➕ Add")); b.clicked.connect(self._add); add.addWidget(b)
        disc = QPushButton(tr("🔎 Découvrir","🔎 Discover")); disc.setObjectName("ghost"); disc.clicked.connect(self._discover); add.addWidget(disc)
        ping = QPushButton(tr("🏓 Tester Sync", "🏓 Test Sync")); ping.setObjectName("ghost"); ping.clicked.connect(self._ping)
        ping.setToolTip(tr("Ping chaque slave de la liste et affiche la latence + statut (OK/échec).", "Pings each slave in the list and shows latency + status (OK/fail)."))
        add.addWidget(ping)
        rem = QPushButton(tr("🗑 Retirer","🗑 Remove")); rem.setObjectName("ghost"); rem.clicked.connect(self._remove); add.addWidget(rem)
        sv.addLayout(add)
        self.list = QListWidget(); sv.addWidget(self.list)
        for h in (self.settings.get("dist_slaves", "") or "").split(",") if self.settings.get("dist_slaves") else []:
            if h.strip():
                self.list.addItem(QListWidgetItem(h.strip()))
        bench = QPushButton(tr("⚡ Benchmark des slaves","⚡ Benchmark slaves")); bench.clicked.connect(self._bench)
        bench.setStyleSheet('background:#2980b9; color:#fff; border:none; border-radius:8px; padding:8px; font-weight:700;')
        sv.addWidget(bench)
        root.addWidget(sb)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFont(QFont("Consolas", 10))
        root.addWidget(self.log, 1)
        root.addWidget(QLabel(tr("Note : nécessite le serveur slave lancé sur chaque machine (port 5001).","Note: requires the slave server running on each machine (port 5001).")))

    def _hosts(self):
        return [self.list.item(i).text() for i in range(self.list.count())]

    def _save(self):
        self.settings.set("dist_slaves", ",".join(self._hosts()))

    def _add(self):
        h = self.host.text().strip()
        if h:
            self.list.addItem(QListWidgetItem(h)); self.host.clear(); self._save()

    def _remove(self):
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))
        self._save()

    def _discover(self):
        self.log.appendPlainText(tr("🔎 Découverte sur le réseau local…","🔎 Discovering on local network…"))
        self.worker = _DiscoverWorker()
        self.worker.found.connect(self._on_found)
        self.worker.error.connect(lambda e: self.log.appendPlainText(f"❌ {e}"))
        self.worker.start()

    def _on_found(self, slaves):
        if not slaves:
            self.log.appendPlainText(tr("… aucun slave trouvé.","… no slave found.")); return
        existing = set(self._hosts())
        for s in slaves:
            if isinstance(s, str):
                h = s
            else:
                h = f"{s.get('ip', s.get('host', '?'))}:{s.get('port', 5001)}"
            if h not in existing:
                self.list.addItem(QListWidgetItem(h))
                # discover_slaves() returns hostname/platform/version per slave — CTk's
                # discovery table showed these; the Qt port collapsed them to a bare
                # "host:port" string. Surface them in the log instead of silently discarding.
                if isinstance(s, dict):
                    extra = ", ".join(f"{k}={s[k]}" for k in ("hostname", "platform", "version") if s.get(k) and s.get(k) != "?")
                    if extra:
                        self.log.appendPlainText(f"   {h} — {extra}")
        self._save(); self.log.appendPlainText(tr(f"✅ {len(slaves)} slave(s) trouvé(s).", f"✅ {len(slaves)} slave(s) found."))

    def _ping(self):
        hosts = self._hosts()
        if not hosts:
            self.log.appendPlainText(tr("❌ Aucun slave.", "❌ No slave.")); return
        self.log.appendPlainText(tr("🏓 Test de synchronisation…", "🏓 Sync test…"))
        self.worker = _PingWorker(hosts)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.done.connect(lambda: self.log.appendPlainText(tr("— fin test —", "— test done —")))
        self.worker.start()

    def _bench(self):
        hosts = self._hosts()
        if not hosts:
            self.log.appendPlainText(tr("❌ Aucun slave.","❌ No slave.")); return
        self.log.appendPlainText("⚡ Benchmark…")
        self.worker = _BenchWorker(hosts)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.done.connect(lambda: self.log.appendPlainText(tr("— fin benchmark —","— benchmark done —")))
        self.worker.start()
