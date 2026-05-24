"""
tab_queue.py — Training Queue
Chain multiple training configurations automatically.
"""
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import threading
import time
from src.ui.components.tooltip import ToolTip
from src.core.settings import SettingsManager


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


class QueueTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.settings = SettingsManager()
        self._queue = []  # List of {"path": str, "status": "pending"|"running"|"done"|"error", "name": str}
        self._running = False
        self._current_idx = -1
        self._build_ui()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(10, 5))
        ctk.CTkLabel(header, text=_t("File d'Attente d'Entrainements", "Training Queue"),
                     font=("Roboto", 20, "bold"), text_color="#e67e22").pack(side="left")

        # Info
        info = ctk.CTkFrame(self, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8)
        info.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(info, text=_t(
            "Enchainez plusieurs configurations d'entrainement automatiquement.\n"
            "Ajoutez des fichiers .toml ou .yml, ordonnez-les, puis lancez la queue.\n"
            "Chaque config sera executee sequentiellement — la suivante demarre quand la precedente finit.",
            "Chain multiple training configurations automatically.\n"
            "Add .toml or .yml files, reorder them, then launch the queue.\n"
            "Each config runs sequentially — the next one starts when the previous finishes."
        ), text_color=("gray30", "#AAA"), font=("Roboto", 11), justify="left", wraplength=800).pack(padx=15, pady=10)

        # Controls
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.pack(fill="x", padx=15, pady=5)

        self.btn_add = ctk.CTkButton(ctrl, text=_t("➕ Ajouter Config", "➕ Add Config"), fg_color="#27ae60",
                                      width=150, height=35, command=self._add_config)
        self.btn_add.pack(side="left", padx=3)
        ToolTip(self.btn_add, _t("Ajouter un fichier de configuration (.toml, .yml) a la queue", "Add a configuration file (.toml, .yml) to the queue"))

        self.btn_add_folder = ctk.CTkButton(ctrl, text=_t("📁 Dossier", "📁 Folder"), fg_color="#2980b9",
                                             width=100, height=35, command=self._add_folder)
        self.btn_add_folder.pack(side="left", padx=3)
        ToolTip(self.btn_add_folder, _t("Ajouter tous les fichiers config d'un dossier", "Add all config files from a folder"))

        self.btn_remove = ctk.CTkButton(ctrl, text=_t("🗑 Retirer", "🗑 Remove"), fg_color="#e74c3c",
                                         width=90, height=35, command=self._remove_selected)
        self.btn_remove.pack(side="left", padx=3)
        ToolTip(self.btn_remove, _t("Retirer l'element selectionne de la queue", "Remove the selected item from the queue"))

        self.btn_move_up = ctk.CTkButton(ctrl, text="⬆", fg_color="#666", width=35, height=35,
                                          command=lambda: self._move(-1))
        self.btn_move_up.pack(side="left", padx=2)
        ToolTip(self.btn_move_up, _t("Monter dans la queue", "Move up in the queue"))

        self.btn_move_down = ctk.CTkButton(ctrl, text="⬇", fg_color="#666", width=35, height=35,
                                            command=lambda: self._move(1))
        self.btn_move_down.pack(side="left", padx=2)
        ToolTip(self.btn_move_down, _t("Descendre dans la queue", "Move down in the queue"))

        self.btn_clear = ctk.CTkButton(ctrl, text=_t("♻ Vider", "♻ Clear"), fg_color="#555", width=80, height=35,
                                        command=self._clear_queue)
        self.btn_clear.pack(side="left", padx=3)

        self.lbl_count = ctk.CTkLabel(ctrl, text="0 config(s)", text_color="#888")
        self.lbl_count.pack(side="right", padx=10)

        # Queue list
        self.queue_frame = ctk.CTkScrollableFrame(self, fg_color=("#E8E8E8", "#1a1a2e"), corner_radius=8, height=250)
        self.queue_frame.pack(fill="both", expand=True, padx=15, pady=5)
        self._show_empty()

        # Start/Stop
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=5)

        self.btn_start = ctk.CTkButton(btn_frame, text=_t("▶  LANCER LA QUEUE", "▶  LAUNCH QUEUE"), fg_color="#27ae60",
                                        height=45, font=("Roboto", 14, "bold"), command=self._start_queue)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ToolTip(self.btn_start, _t("Lancer tous les entrainements de la queue sequentiellement", "Run all training sessions in the queue sequentially"))

        self.btn_stop = ctk.CTkButton(btn_frame, text="⏹  STOP QUEUE", fg_color="#D35B58",
                                       height=45, font=("Roboto", 14, "bold"), state="disabled",
                                       command=self._stop_queue)
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Log
        self.log = ctk.CTkTextbox(self, height=100, font=("Consolas", 10))
        self.log.pack(fill="x", padx=15, pady=(0, 10))
        self.log.insert("1.0", _t("[Queue] En attente...\n", "[Queue] Waiting...\n"))
        self.log.configure(state="disabled")

        # Selected index
        self._selected_idx = -1

    def _show_empty(self):
        for w in self.queue_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.queue_frame, text=_t("Aucune configuration dans la queue.\n\nCliquez '+ Ajouter Config' pour commencer.",
                                                "No configuration in the queue.\n\nClick '+ Add Config' to get started."),
                     text_color="#666", font=("Roboto", 12), justify="center").pack(pady=40)

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _refresh_list(self):
        for w in self.queue_frame.winfo_children():
            w.destroy()

        if not self._queue:
            self._show_empty()
            self.lbl_count.configure(text="0 config(s)")
            return

        self.lbl_count.configure(text=f"{len(self._queue)} config(s)")

        # Header
        hdr = ctk.CTkFrame(self.queue_frame, fg_color=("#D8D8D8", "#2B2B4B"), corner_radius=4, height=25)
        hdr.pack(fill="x", pady=(0, 3))
        for txt, w in [("#", 30), (_t("Nom", "Name"), 200), (_t("Fichier", "File"), 350), (_t("Statut", "Status"), 80)]:
            ctk.CTkLabel(hdr, text=txt, font=("Roboto", 9, "bold"), text_color=("gray30", "#AAA"),
                         width=w, anchor="w").pack(side="left", padx=4)

        for i, item in enumerate(self._queue):
            status = item["status"]
            colors = {"pending": "#888", "running": "#f39c12", "done": "#2ecc71", "error": "#e74c3c"}
            status_icons = {"pending": "⏳", "running": "🔄", "done": "✅", "error": "❌"}
            bg = "#2B2B3B" if i != self._selected_idx else "#3B3B5B"

            row = ctk.CTkFrame(self.queue_frame, fg_color=bg, corner_radius=4, height=30)
            row.pack(fill="x", pady=1)
            row.bind("<Button-1>", lambda e, idx=i: self._select(idx))

            ctk.CTkLabel(row, text=str(i + 1), width=30, text_color=("gray30", "#AAA"), anchor="w").pack(side="left", padx=4)
            name_lbl = ctk.CTkLabel(row, text=item["name"], width=200, font=("Roboto", 11, "bold"),
                                     text_color="#3498db", anchor="w")
            name_lbl.pack(side="left", padx=4)
            name_lbl.bind("<Button-1>", lambda e, idx=i: self._select(idx))

            ctk.CTkLabel(row, text=os.path.basename(item["path"]), width=350, text_color="#999",
                         anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=f"{status_icons.get(status, '?')} {status}",
                         text_color=colors.get(status, "#888"), width=80, anchor="w").pack(side="left", padx=4)

    def _select(self, idx):
        self._selected_idx = idx
        self._refresh_list()

    def _add_config(self):
        paths = filedialog.askopenfilenames(
            title=_t("Ajouter des configurations", "Add configurations"),
            filetypes=[("Config files", "*.toml *.yml *.yaml"), ("All", "*.*")]
        )
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            # Try to extract experiment name from file
            try:
                with open(p, 'r') as f:
                    for line in f:
                        if 'name' in line and '=' in line:
                            name = line.split('=')[1].strip().strip('"').strip("'")
                            break
                        elif 'name:' in line:
                            name = line.split(':')[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass
            self._queue.append({"path": p, "status": "pending", "name": name})
            self._log(f"[+] Ajouté: {name}")
        self._refresh_list()

    def _add_folder(self):
        folder = filedialog.askdirectory(title=_t("Dossier de configurations", "Configuration folder"))
        if not folder:
            return
        count = 0
        for f in sorted(os.listdir(folder)):
            if f.endswith((".toml", ".yml", ".yaml")):
                p = os.path.join(folder, f)
                name = os.path.splitext(f)[0]
                self._queue.append({"path": p, "status": "pending", "name": name})
                count += 1
        self._log(f"[+] {count} config(s) ajoutee(s) depuis {os.path.basename(folder)}")
        self._refresh_list()

    def _remove_selected(self):
        if 0 <= self._selected_idx < len(self._queue):
            removed = self._queue.pop(self._selected_idx)
            self._log(f"[-] Retire: {removed['name']}")
            self._selected_idx = min(self._selected_idx, len(self._queue) - 1)
            self._refresh_list()

    def _move(self, direction):
        idx = self._selected_idx
        new_idx = idx + direction
        if 0 <= idx < len(self._queue) and 0 <= new_idx < len(self._queue):
            self._queue[idx], self._queue[new_idx] = self._queue[new_idx], self._queue[idx]
            self._selected_idx = new_idx
            self._refresh_list()

    def _clear_queue(self):
        self._queue.clear()
        self._selected_idx = -1
        self._refresh_list()
        self._log("[Queue] Videe")

    def _start_queue(self):
        pending = [q for q in self._queue if q["status"] == "pending"]
        if not pending:
            messagebox.showinfo("Queue", _t("Aucune configuration en attente.", "No pending configurations."))
            return
        self._running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._log(f"[Queue] Lancement de {len(pending)} entrainement(s)")

        def run_queue():
            for i, item in enumerate(self._queue):
                if not self._running:
                    self.after(0, lambda: self._log("[Queue] Arretee par l'utilisateur"))
                    break
                if item["status"] != "pending":
                    continue
                self._current_idx = i
                item["status"] = "running"
                self.after(0, self._refresh_list)
                self.after(0, lambda n=item["name"]: self._log(f"[Queue] Demarrage: {n}"))

                # Launch training via the Run tab
                try:
                    app = self.winfo_toplevel()
                    if hasattr(app, 'train_tab') and app.train_tab:
                        # Use after() to run on main thread
                        import queue as _q
                        done_event = threading.Event()
                        original_on_finished = app.train_tab.on_finished

                        def patched_on_finished():
                            original_on_finished()
                            done_event.set()

                        self.after(0, lambda p=item["path"]: self._launch_one(app.train_tab, p, patched_on_finished))
                        # Wait for completion (with timeout of 7 days)
                        done_event.wait(timeout=7 * 24 * 3600)
                        # Restore
                        app.train_tab.on_finished = original_on_finished

                        if done_event.is_set():
                            item["status"] = "done"
                            self.after(0, lambda n=item["name"]: self._log(f"[Queue] Termine: {n}"))
                        else:
                            item["status"] = "error"
                            self.after(0, lambda n=item["name"]: self._log(f"[Queue] Timeout: {n}"))
                    else:
                        item["status"] = "error"
                        self.after(0, lambda: self._log("[Queue] Erreur: Onglet Entrainement non disponible"))
                except Exception as e:
                    item["status"] = "error"
                    self.after(0, lambda err=str(e): self._log(f"[Queue] Erreur: {err}"))

                self.after(0, self._refresh_list)

            self._running = False
            self._current_idx = -1
            self.after(0, lambda: self.btn_start.configure(state="normal"))
            self.after(0, lambda: self.btn_stop.configure(state="disabled"))
            self.after(0, lambda: self._log("[Queue] Toutes les taches terminees"))

        threading.Thread(target=run_queue, daemon=True).start()

    def _launch_one(self, train_tab, config_path, on_finished_cb):
        """Launch a single training on the main thread."""
        train_tab.on_finished = on_finished_cb
        train_tab.external_start(config_path)

    def _stop_queue(self):
        self._running = False
        self._log("[Queue] Arret demande...")
        # Also stop current training
        app = self.winfo_toplevel()
        if hasattr(app, 'train_tab') and app.train_tab:
            app.train_tab.on_stop()
