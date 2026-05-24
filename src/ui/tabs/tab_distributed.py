"""
tab_distributed.py — Distributed Training UI
Discover slaves, benchmark, and manage multi-GPU distributed training.
"""
import customtkinter as ctk
from tkinter import messagebox
import threading
import os
from src.ui.components.tooltip import ToolTip


def _t(fr: str, en: str) -> str:
    """Pick FR or EN string based on active language."""
    try:
        from src.core.translations import get_translator
        tr = get_translator()
        if tr and getattr(tr, 'language', 'fr') == 'en':
            return en
    except Exception:
        pass
    return fr


class DistributedTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._slaves = []
        self._build_ui()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(15, 5))

        ctk.CTkLabel(header, text=_t("Entrainement Partage", "Distributed Training"),
                     font=("Roboto", 20, "bold"), text_color="#9b59b6").pack(side="left")

        # Info
        info = ctk.CTkFrame(self, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8)
        info.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(info, text=_t(
            "L'entrainement partage permet de repartir le travail sur plusieurs machines du reseau.\n"
            "Chaque machine 'slave' execute neo_sr_slave.py et attend les instructions du master.\n\n"
            "Etapes :\n"
            "  1. Lancez neo_sr_slave.py sur chaque machine du reseau (meme sous-reseau)\n"
            "  2. Cliquez 'Rechercher' ci-dessous pour decouvrir les slaves disponibles\n"
            "  3. Lancez un benchmark pour evaluer les performances GPU de chaque slave\n"
            "  4. Utilisez 'Lancer Entrainement' pour partager le travail",
            "Distributed training splits the workload across multiple network machines.\n"
            "Each 'slave' machine runs neo_sr_slave.py and waits for instructions from the master.\n\n"
            "Steps:\n"
            "  1. Launch neo_sr_slave.py on each machine (same subnet)\n"
            "  2. Click 'Discover' below to find available slaves\n"
            "  3. Run a benchmark to evaluate each slave's GPU performance\n"
            "  4. Use 'Launch Distributed Training' to share the workload"
        ), text_color=("gray30", "#AAA"), font=("Roboto", 11), justify="left", anchor="w",
            wraplength=800).pack(padx=15, pady=10, fill="x")

        # Controls
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.pack(fill="x", padx=15, pady=5)

        self.btn_discover = ctk.CTkButton(ctrl, text=_t("🔍 Rechercher Slaves", "🔍 Discover Slaves"),
                                           fg_color="#8e44ad", width=180, height=35,
                                           command=self._discover_slaves)
        self.btn_discover.pack(side="left", padx=5)
        ToolTip(self.btn_discover, _t("Scanner le reseau local pour trouver les machines slaves executant neo_sr_slave.py", "Scan the local network to find slave machines running neo_sr_slave.py"))

        self.btn_benchmark = ctk.CTkButton(ctrl, text="⚡ Benchmark",
                                            fg_color="#2980b9", width=130, height=35,
                                            command=self._run_benchmark,
                                            state="disabled")
        self.btn_benchmark.pack(side="left", padx=5)
        ToolTip(self.btn_benchmark, _t("Lancer un benchmark GPU sur chaque slave pour evaluer les performances", "Run a GPU benchmark on each slave to evaluate performance"))

        self.btn_sync = ctk.CTkButton(ctrl, text="🔄 Test Sync",
                                         fg_color="#e67e22", width=120, height=35,
                                         command=self._sync_test,
                                         state="disabled")
        self.btn_sync.pack(side="left", padx=5)
        ToolTip(self.btn_sync, _t("Tester la communication reseau entre le master et chaque slave (ping + latence)", "Test network communication between master and each slave (ping + latency)"))

        self.btn_launch = ctk.CTkButton(ctrl, text=_t("🚀 Lancer Entrainement Partage", "🚀 Launch Distributed Training"),
                                         fg_color="#27ae60", width=250, height=35,
                                         command=self._launch_distributed,
                                         state="disabled")
        self.btn_launch.pack(side="left", padx=5)
        ToolTip(self.btn_launch, _t("Lancer l'entrainement distribue via PyTorch DDP sur toutes les machines connectees", "Launch distributed training via PyTorch DDP on all connected machines"))

        self.lbl_status = ctk.CTkLabel(ctrl, text=_t("0 slave(s) detecte(s)", "0 slave(s) detected"),
                                        text_color="#888", font=("Roboto", 12))
        self.lbl_status.pack(side="right", padx=15)

        # Slaves list
        self.slaves_frame = ctk.CTkScrollableFrame(self, fg_color=("#E8E8E8", "#1a1a2e"),
                                                     corner_radius=8, height=300)
        self.slaves_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # Initial message
        self._show_empty_message()

        # Log
        self.log_text = ctk.CTkTextbox(self, height=120, font=("Consolas", 10))
        self.log_text.pack(fill="x", padx=15, pady=(0, 10))
        self.log_text.insert("1.0", "[Partage] En attente...\n")
        self.log_text.configure(state="disabled")

    def _show_empty_message(self):
        for w in self.slaves_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.slaves_frame,
                     text=_t("Aucun slave detecte.\n\nLancez neo_sr_slave.py sur les machines du reseau,\npuis cliquez 'Rechercher Slaves'.",
                             "No slave detected.\n\nRun neo_sr_slave.py on network machines,\nthen click 'Discover Slaves'."),
                     text_color="#666", font=("Roboto", 12), justify="center").pack(pady=40)

    def _log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _discover_slaves(self):
        self.lbl_status.configure(text="Recherche en cours...", text_color="#f39c12")
        self.btn_discover.configure(state="disabled")
        self._log("[Recherche] Scan du reseau local (port 5001)...")

        def worker():
            try:
                from src.core.distributed_client import discover_slaves
                slaves = discover_slaves(port=5001, timeout=2.0)
                self.after(0, lambda: self._on_discovery_done(slaves))
            except Exception as e:
                self.after(0, lambda: self._on_discovery_done([], str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_discovery_done(self, slaves, error=None):
        self.btn_discover.configure(state="normal")
        self._slaves = slaves

        if error:
            self.lbl_status.configure(text=f"Erreur: {error}", text_color="#e74c3c")
            self._log(f"[Erreur] {error}")
            return

        count = len(slaves)
        self.lbl_status.configure(
            text=f"{count} slave(s) detecte(s)",
            text_color="#2ecc71" if count > 0 else "#888"
        )
        self._log(f"[Resultat] {count} slave(s) trouve(s)")

        if count > 0:
            self.btn_benchmark.configure(state="normal")
            self.btn_sync.configure(state="normal")
            self.btn_launch.configure(state="normal")

        # Update UI
        for w in self.slaves_frame.winfo_children():
            w.destroy()

        if not slaves:
            self._show_empty_message()
            return

        # Header row
        hdr = ctk.CTkFrame(self.slaves_frame, fg_color=("#D8D8D8", "#2B2B4B"), corner_radius=5)
        hdr.pack(fill="x", pady=(0, 5))
        for col, w in [("Hostname", 140), ("GPU", 220), ("VRAM", 80),
                       ("IP", 130), ("Status", 80), ("Perf", 80)]:
            ctk.CTkLabel(hdr, text=col, font=("Roboto", 10, "bold"),
                         text_color=("gray30", "#AAA"), width=w, anchor="w").pack(side="left", padx=5)

        for s in slaves:
            row = ctk.CTkFrame(self.slaves_frame, fg_color=("#E0E0E0", "#2B2B2B"), corner_radius=5)
            row.pack(fill="x", pady=2)
            hostname = s.get("hostname", "?")
            gpu = s.get("gpu_name", "GPU?")
            vram = f"{s.get('gpu_vram_gb', '?')}GB"
            ip = s.get("ip", "?")
            status = s.get("status", "idle")
            status_color = "#2ecc71" if status == "idle" else "#e67e22"

            ctk.CTkLabel(row, text=hostname, font=("Roboto", 11, "bold"),
                         text_color="#3498db", width=140, anchor="w").pack(side="left", padx=5)
            ctk.CTkLabel(row, text=gpu, text_color=("gray20", "#CCC"), width=220, anchor="w").pack(side="left", padx=5)
            ctk.CTkLabel(row, text=vram, text_color=("gray20", "#CCC"), width=80, anchor="w").pack(side="left", padx=5)
            ctk.CTkLabel(row, text=ip, text_color=("gray40", "#888"), width=130, anchor="w").pack(side="left", padx=5)
            ctk.CTkLabel(row, text=status, text_color=status_color, width=80, anchor="w").pack(side="left", padx=5)
            ctk.CTkLabel(row, text="--", text_color="#888", width=80, anchor="w").pack(side="left", padx=5)

            self._log(f"  → {hostname} ({gpu}, {vram}) @ {ip} [{status}]")

    def _run_benchmark(self):
        if not self._slaves:
            messagebox.showinfo(_t("Benchmark", "Benchmark"), _t("Aucun slave disponible.", "No slave available."))
            return
        self._log("[Benchmark] Lancement du benchmark sur tous les slaves...")
        self.btn_benchmark.configure(state="disabled", text="⏳ Benchmark...")

        def worker():
            try:
                from src.core.distributed_client import SlaveClient
                client = SlaveClient()
                client.slaves = self._slaves
                results = client.benchmark_all()
                self.after(0, lambda: self._on_benchmark_done(results))
            except Exception as e:
                self.after(0, lambda: self._on_benchmark_err(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_benchmark_done(self, results):
        self.btn_benchmark.configure(state="normal", text="⚡ Benchmark")
        self._log(f"[Benchmark] Termine: {len(results)} resultats")
        for r in results:
            self._log(f"  → {r.get('hostname', '?')}: {r.get('score', '?')} points")

    def _on_benchmark_err(self, error):
        self.btn_benchmark.configure(state="normal", text="⚡ Benchmark")
        self._log(f"[Benchmark] Erreur: {error}")

    def _sync_test(self):
        """Run a quick sync test between master and slaves."""
        if not self._slaves:
            messagebox.showinfo(_t("Test Sync", "Sync Test"), _t("Aucun slave disponible.", "No slave available."))
            return
        self._log("[Sync Test] Verification de la synchronisation...")
        self.btn_sync.configure(state="disabled", text="Test en cours...")

        def worker():
            results = []
            import time
            for s in self._slaves:
                try:
                    from src.core.distributed_client import SlaveClient
                    client = SlaveClient()
                    start = time.time()
                    # Ping test
                    ok = client.ping(s.get("ip", ""), s.get("port", 5001))
                    latency = round((time.time() - start) * 1000, 1)
                    results.append({"host": s.get("hostname","?"), "ok": ok, "latency": latency})
                except Exception as e:
                    results.append({"host": s.get("hostname","?"), "ok": False, "latency": 0, "error": str(e)})
            self.after(0, lambda: self._on_sync_done(results))

        threading.Thread(target=worker, daemon=True).start()

    def _on_sync_done(self, results):
        self.btn_sync.configure(state="normal", text="🔄 Test Sync")
        all_ok = all(r["ok"] for r in results)
        for r in results:
            status = "OK" if r["ok"] else "ECHEC"
            self._log(f"  → {r['host']}: {status} (latence: {r['latency']}ms)")
        if all_ok:
            self._log("[Sync Test] Tous les slaves repondent correctement !")
            messagebox.showinfo(_t("Test Sync", "Sync Test"), _t(f"Synchronisation OK\n{len(results)} slave(s) pret(s)", f"Synchronization OK\n{len(results)} slave(s) ready"))
        else:
            fails = sum(1 for r in results if not r["ok"])
            self._log(f"[Sync Test] {fails} slave(s) en echec")
            messagebox.showwarning(_t("Test Sync", "Sync Test"), _t(f"{fails}/{len(results)} slave(s) ne repondent pas", f"{fails}/{len(results)} slave(s) not responding"))

    def _launch_distributed(self):
        if not self._slaves:
            messagebox.showinfo(_t("Partage", "Distributed Training"), _t("Aucun slave disponible.", "No slave available."))
            return
        self._log("[Partage] Fonctionnalite en cours de developpement...")
        messagebox.showinfo(_t("Partage", "Distributed Training"),
                            _t(f"{len(self._slaves)} slave(s) pret(s).\n\n"
                               "Configuration a utiliser :\n"
                               "- Le fichier .toml/.yml genere par le Configurateur\n"
                               "  ou le fichier genere par le Pipeline PSNR->GAN\n\n"
                               "L'entrainement sera lance via PyTorch DDP\n"
                               "sur toutes les machines connectees.\n\n"
                               "Conseil : Faites un Test Sync avant de lancer\n"
                               "pour verifier la communication entre les machines.",
                               f"{len(self._slaves)} slave(s) ready.\n\n"
                               "Configuration to use:\n"
                               "- The .toml/.yml file generated by the Configurator\n"
                               "  or the file generated by the PSNR->GAN Pipeline\n\n"
                               "Training will be launched via PyTorch DDP\n"
                               "on all connected machines.\n\n"
                               "Tip: Run a Sync Test before launching\n"
                               "to verify communication between machines."))
