"""
tab_wizard.py — Onglet Wizard pour Universal SR Studio.
Interface pas-à-pas avec détection GPU, support NeoSR + Redux, traductions.
"""
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os

try:
    import tomli_w
except ImportError:
    tomli_w = None

try:
    import toml as toml_write
except ImportError:
    toml_write = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    from src.core.wizard_advanced import (
        WizardEngine, QuestionType, GPUDetector,
        NEOSR_ARCHITECTURES, REDUX_ARCHITECTURES,
        NEOSR_DISCRIMINATORS, REDUX_DISCRIMINATORS,
        _safe_int, _safe_float,
    )
except ImportError:
    WizardEngine = None

try:
    from src.core.translations import t
except ImportError:
    def t(key, **kw): return key


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


def _write_toml(data: dict, filepath: str):
    """Write a dict as TOML file, using available library."""
    if tomli_w:
        with open(filepath, "wb") as f:
            tomli_w.dump(data, f)
        return
    if toml_write:
        with open(filepath, "w", encoding="utf-8") as f:
            toml_write.dump(data, f)
        return
    # Fallback: manual basic TOML writer
    _write_toml_basic(data, filepath)


def _write_toml_basic(data: dict, filepath: str, _indent=""):
    """Basic TOML writer for simple nested dicts."""
    lines = []
    tables = []
    for k, v in data.items():
        if isinstance(v, dict):
            tables.append((k, v))
        elif isinstance(v, list):
            items = ", ".join(_toml_val(x) for x in v)
            lines.append(f"{k} = [ {items} ]")
        else:
            lines.append(f"{k} = {_toml_val(v)}")

    for section_name, section_data in tables:
        lines.append("")
        lines.append(f"[{section_name}]")
        for k2, v2 in section_data.items():
            if isinstance(v2, dict):
                lines.append("")
                lines.append(f"[{section_name}.{k2}]")
                for k3, v3 in v2.items():
                    if isinstance(v3, dict):
                        lines.append(f"")
                        lines.append(f"[{section_name}.{k2}.{k3}]")
                        for k4, v4 in v3.items():
                            lines.append(f"{k4} = {_toml_val(v4)}")
                    elif isinstance(v3, list):
                        items = ", ".join(_toml_val(x) for x in v3)
                        lines.append(f"{k3} = [ {items} ]")
                    else:
                        lines.append(f"{k3} = {_toml_val(v3)}")
            elif isinstance(v2, list):
                items = ", ".join(_toml_val(x) for x in v2)
                lines.append(f"{k2} = [ {items} ]")
            else:
                lines.append(f"{k2} = {_toml_val(v2)}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _toml_val(v):
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, str): return f'"{v}"'
    if isinstance(v, float):
        if v < 0.001 and v > 0: return f"{v:.1e}"
        return str(v)
    if v is None: return '""'
    return str(v)


class WizardTab(ctk.CTkScrollableFrame):
    """Interface du wizard avec navigation par étapes."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        if WizardEngine is None:
            ctk.CTkLabel(self, text="❌ Module wizard_advanced.py introuvable.",
                         font=("Arial", 14)).pack(pady=20)
            return

        self.wizard = WizardEngine()
        self.current_step = 0
        self.answer_widget = None

        self._create_header()
        self._create_content_area()
        self._create_navigation()
        self._show_step(0)

    # ─── Header ──────────────────────────────────────────────

    def _create_header(self):
        header = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=10)
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header, text=_t("😊 Assistant de Configuration", "😊 Configuration Assistant"),
                     font=("Arial", 20, "bold"),
                     text_color="#e0e0ff").pack(pady=(10, 5))

        # GPU Info bar
        gpu = self.wizard.gpu_info
        if gpu:
            suitable = gpu.is_suitable_for_training()
            amp_ok = gpu.supports_amp
            gpu_text = f"🖥️ {gpu.name} — {gpu.total_vram_gb:.0f} GB VRAM"
            if not amp_ok:
                gpu_text += " — ⚠️ AMP non supporté (sm_61)"
            color = "#2ecc71" if (suitable and amp_ok) else "#e67e22" if suitable else "#e74c3c"
        else:
            gpu_text = _t("⚠️ Aucun GPU NVIDIA détecté — training impossible", "⚠️ No NVIDIA GPU detected — training impossible")
            color = "#e74c3c"

        ctk.CTkLabel(header, text=gpu_text,
                     font=("Arial", 12), text_color=color).pack(pady=(0, 5))

        # Progress
        self.lbl_progress = ctk.CTkLabel(header, text="Étape 1",
                                         font=("Arial", 11), text_color="#888")
        self.lbl_progress.pack()
        self.progress_bar = ctk.CTkProgressBar(header, width=400, height=8,
                                                progress_color="#3498db")
        self.progress_bar.pack(pady=(5, 10))
        self.progress_bar.set(0)

    # ─── Content ─────────────────────────────────────────────

    def _create_content_area(self):
        self.content_frame = ctk.CTkFrame(self, fg_color="#16213e", corner_radius=10)
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.lbl_question = ctk.CTkLabel(
            self.content_frame, text="", font=("Arial", 16, "bold"),
            text_color="#ecf0f1", wraplength=550
        )
        self.lbl_question.pack(pady=(20, 5))

        self.lbl_help = ctk.CTkLabel(
            self.content_frame, text="", font=("Arial", 11),
            text_color="#95a5a6", wraplength=550, justify="left"
        )
        self.lbl_help.pack(pady=(0, 10), padx=20)

        self.answer_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.answer_frame.pack(fill="x", padx=40, pady=10)

        self.lbl_recommendation = ctk.CTkLabel(
            self.content_frame, text="", font=("Arial", 11, "italic"),
            text_color="#f39c12", wraplength=500, justify="left"
        )
        self.lbl_recommendation.pack(pady=(10, 5), padx=20)

        self.lbl_estimate = ctk.CTkLabel(
            self.content_frame, text="", font=("Arial", 11), text_color="#95a5a6"
        )
        self.lbl_estimate.pack(pady=(0, 15))

    # ─── Navigation ──────────────────────────────────────────

    def _create_navigation(self):
        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.pack(fill="x", padx=10, pady=10)

        self.btn_prev = ctk.CTkButton(
            nav, text=_t("◀  Précédent", "◀  Previous"), width=120,
            fg_color="#34495e", hover_color="#2c3e50",
            command=self._prev_step
        )
        self.btn_prev.pack(side="left", padx=5)

        self.btn_next = ctk.CTkButton(
            nav, text=_t("Suivant  ▶", "Next  ▶"), width=120,
            fg_color="#2980b9", hover_color="#2471a3",
            command=self._next_step
        )
        self.btn_next.pack(side="right", padx=5)

        self.btn_generate = ctk.CTkButton(
            nav, text=_t("✨ Générer la Config", "✨ Generate Config"), width=180,
            fg_color="#27ae60", hover_color="#219a52",
            command=self._generate_config
        )

    # ─── Step Display ────────────────────────────────────────

    def _show_step(self, step_index: int):
        self.current_step = step_index
        question = self.wizard.get_question(step_index)
        if not question:
            return

        visible = self.wizard.get_visible_questions()
        total = len(visible)

        progress = (step_index + 1) / max(total, 1)
        self.progress_bar.set(progress)
        self.lbl_progress.configure(text=f"Étape {step_index + 1}/{total}")

        self.lbl_question.configure(text=question.text)
        self.lbl_help.configure(text=question.help_text)

        # Dynamic recommendation based on current answers
        rec = self.wizard.get_recommendation(question.id)
        self.lbl_recommendation.configure(text=rec)

        # Update dynamic options before creating widget
        if question.id == "arch":
            question.options = self.wizard.get_arch_options()
        elif question.id == "discriminator_type":
            question.options = self.wizard.get_discriminator_options()
        elif question.id == "scale":
            question.options = self.wizard.get_scale_options()
        elif question.id == "optimizer":
            question.options = self.wizard.get_optimizer_options()
        elif question.id == "scheduler":
            question.options = self.wizard.get_scheduler_options()

        self._create_answer_widget(question)

        # Navigation
        self.btn_prev.configure(state="normal" if step_index > 0 else "disabled")
        is_last = step_index >= total - 1
        if is_last:
            self.btn_next.pack_forget()
            self.btn_generate.pack(side="right", padx=5)
        else:
            self.btn_generate.pack_forget()
            self.btn_next.pack(side="right", padx=5)

        self._update_estimate()

    def _create_answer_widget(self, question):
        for w in self.answer_frame.winfo_children():
            w.destroy()

        existing = self.wizard.get_answer(question.id, question.default)

        if question.type == QuestionType.INFO:
            # Info panel — no input, just display
            self.answer_widget = None
            return

        if question.type == QuestionType.CHOICE:
            var = ctk.StringVar(value=str(existing) if existing else
                                (question.options[0] if question.options else ""))
            if len(question.options) <= 5:
                for opt in question.options:
                    rb = ctk.CTkRadioButton(
                        self.answer_frame, text=opt, variable=var, value=opt,
                        font=("Arial", 13), text_color="#bdc3c7"
                    )
                    rb.pack(anchor="w", pady=3)
            else:
                menu = ctk.CTkOptionMenu(
                    self.answer_frame, values=question.options, variable=var, width=280
                )
                menu.pack(pady=5)
            self.answer_widget = var

        elif question.type == QuestionType.BOOL:
            var = ctk.BooleanVar(value=bool(existing))
            frame = ctk.CTkFrame(self.answer_frame, fg_color="transparent")
            frame.pack(pady=5)
            ctk.CTkRadioButton(
                frame, text="✅ Oui", variable=var, value=True,
                font=("Arial", 13), text_color="#2ecc71"
            ).pack(side="left", padx=20)
            ctk.CTkRadioButton(
                frame, text="❌ Non", variable=var, value=False,
                font=("Arial", 13), text_color="#e74c3c"
            ).pack(side="left", padx=20)
            self.answer_widget = var

        elif question.type == QuestionType.PATH:
            frame = ctk.CTkFrame(self.answer_frame, fg_color="transparent")
            frame.pack(fill="x", pady=5)
            entry = ctk.CTkEntry(frame, width=350, placeholder_text="Sélectionnez un dossier...")
            entry.pack(side="left", padx=(0, 5))
            if existing and existing != question.default:
                entry.insert(0, str(existing))
            btn = ctk.CTkButton(frame, text="📁", width=40,
                                command=lambda: self._browse_path(entry))
            btn.pack(side="left")
            self.answer_widget = entry

        elif question.type == QuestionType.NUMBER:
            if question.options:
                var = ctk.StringVar(value=str(existing) if existing else question.options[0])
                seg = ctk.CTkSegmentedButton(
                    self.answer_frame, values=question.options, variable=var,
                    font=("Arial", 12)
                )
                seg.pack(pady=5)
                ctk.CTkLabel(self.answer_frame, text="ou valeur personnalisée :",
                             font=("Arial", 10), text_color="#7f8c8d").pack()
                entry = ctk.CTkEntry(self.answer_frame, width=100, placeholder_text="Manuel")
                entry.pack(pady=3)
                self.answer_widget = (var, entry)
            else:
                entry = ctk.CTkEntry(self.answer_frame, width=150)
                if existing:
                    entry.insert(0, str(existing))
                entry.pack(pady=5)
                self.answer_widget = entry

        elif question.type == QuestionType.TEXT:
            entry = ctk.CTkEntry(self.answer_frame, width=300,
                                 placeholder_text=str(question.default) or "")
            if existing:
                entry.insert(0, str(existing))
            entry.pack(pady=5)
            self.answer_widget = entry

    def _browse_path(self, entry):
        path = filedialog.askdirectory(title="Sélectionner un dossier")
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _get_current_answer(self):
        w = self.answer_widget
        if w is None:
            return "ok"  # INFO type
        if isinstance(w, (ctk.StringVar, ctk.BooleanVar)):
            return w.get()
        elif isinstance(w, ctk.CTkEntry):
            return w.get()
        elif isinstance(w, tuple):
            var, entry = w
            manual = entry.get().strip()
            return manual if manual else var.get()
        return None

    # ─── Navigation Actions ──────────────────────────────────

    def _next_step(self):
        if self._validate_and_save():
            self._show_step(self.current_step + 1)

    def _prev_step(self):
        if self.current_step > 0:
            self._save_answer()
            self._show_step(self.current_step - 1)

    def _validate_and_save(self) -> bool:
        question = self.wizard.get_question(self.current_step)
        if not question:
            return True
        answer = self._get_current_answer()
        if not question.validate(answer):
            messagebox.showwarning(
                "Validation",
                f"La valeur '{answer}' n'est pas valide.\n\n{question.help_text}"
            )
            return False
        self.wizard.set_answer(question.id, answer)
        return True

    def _save_answer(self):
        question = self.wizard.get_question(self.current_step)
        if question:
            answer = self._get_current_answer()
            if answer is not None:
                self.wizard.set_answer(question.id, answer)

    def _update_estimate(self):
        try:
            est = self.wizard.estimate_training_time()
            if est["estimated_hours"] > 0:
                self.lbl_estimate.configure(
                    text=f"⏱️ Estimation : {est['readable']} ({est['estimated_speed']} it/s)"
                )
            else:
                self.lbl_estimate.configure(text="")
        except Exception:
            self.lbl_estimate.configure(text="")

    # ─── Config Generation ───────────────────────────────────

    def _generate_config(self):
        if not self._validate_and_save():
            return

        config = self.wizard.generate_config()
        ext = self.wizard.get_config_extension()
        engine = str(self.wizard.get_answer("engine", "NeoSR"))
        exp_name = config.get("name", "config")

        file_path = filedialog.asksaveasfilename(
            title="Sauvegarder la configuration",
            defaultextension=ext,
            filetypes=[
                ("TOML (NeoSR)", "*.toml") if ext == ".toml" else ("YAML (Redux)", "*.yml"),
                ("Tous", "*.*"),
            ],
            initialfile=f"{exp_name}{ext}"
        )

        if not file_path:
            return

        try:
            if ext == ".toml":
                _write_toml(config, file_path)
            else:
                if yaml is None:
                    messagebox.showerror("Erreur", "Module PyYAML non installé.\npip install pyyaml")
                    return
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False,
                              allow_unicode=True, sort_keys=False)

            arch = config.get("network_g", {}).get("type", "?")
            amp_status = "✅ AMP ON" if config.get("use_amp", False) else "❌ AMP OFF"
            is_gan = "network_d" in config
            gan_str = "✅ GAN" if is_gan else "PSNR only"

            messagebox.showinfo(
                "Configuration générée !",
                f"✅ Sauvegardée : {os.path.basename(file_path)}\n\n"
                f"🔧 Moteur : {engine}\n"
                f"🏗️ Architecture : {arch}\n"
                f"📊 Batch : {config.get('datasets', {}).get('train', {}).get('batch_size', config.get('datasets', {}).get('train', {}).get('batch_size_per_gpu', '?'))}\n"
                f"⚡ {amp_status}\n"
                f"🎯 {gan_str}\n\n"
                f"Chargez ce fichier dans l'onglet Configuration pour le modifier."
            )
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder :\n{e}")
