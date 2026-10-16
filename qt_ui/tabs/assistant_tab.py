"""
assistant_tab.py — Assistant / Wizard (PySide6). Drives src.core.wizard_advanced.
WizardEngine: renders its adaptive questions, shows per-question recommendations
(GPU-aware), then generates a ready config (NeoSR .toml / Redux .yml) the user can save.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QCheckBox,
    QPushButton, QFrame, QScrollArea, QPlainTextEdit, QFileDialog, QMessageBox,
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from qt_ui.i18n import tr



# Display labels for the wizard's internal option keys (the keys stay the stored values).
# They were shown raw before: "qualite" / "vitesse" / "equilibre" / "mixte" in both languages.
_OPTION_LABELS = {
    "qualite": ("Qualité", "Quality"), "vitesse": ("Vitesse", "Speed"),
    "equilibre": ("Équilibre", "Balance"), "anime": ("Anime", "Anime"),
    "photo": ("Photo", "Photo"), "mixte": ("Mixte (anime + photo)", "Mixed (anime + photo)"),
    "otf": ("OTF (dégradations à la volée)", "OTF (on-the-fly degradations)"),
    "paired": ("Paired (paires LQ/HR)", "Paired (LQ/HR pairs)"),
}


def _option_label(key) -> str:
    fr_en = _OPTION_LABELS.get(str(key))
    return tr(*fr_en) if fr_en else str(key)


def _choice_value(cb) -> str:
    d = cb.currentData()
    return str(d) if d is not None else cb.currentText()


class _ReadOnlySettings:
    """Settings view for the hidden helper Config tab: reads the user's settings, never
    writes them (the helper must not change the real Config tab's saved state)."""

    def __init__(self, real):
        self._real = real

    def get(self, key, default=None):
        return self._real.get(key, default)

    def set(self, *_a, **_k):
        pass

    def save(self, *_a, **_k):
        pass

    def __getattr__(self, name):
        return getattr(self._real, name)

class AssistantTab(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.engine = None
        self.widgets = {}

        root = QVBoxLayout(self); root.setContentsMargins(16, 16, 16, 16); root.setSpacing(8)
        t = QLabel(tr("🧙 Assistant — Wizard de configuration", "🧙 Assistant — Configuration Wizard")); t.setObjectName("title"); root.addWidget(t)
        root.addWidget(QLabel(tr(
            "Réponds aux questions, l'assistant génère une config prête à entraîner "
            "(recommandations adaptées à ton GPU). ↓ Fais défiler pour voir toutes les questions.",
            "Answer the questions, the assistant generates a ready-to-train config "
            "(recommendations adapted to your GPU). ↓ Scroll to see all questions.")))

        try:
            from src.core.wizard_advanced import WizardEngine, QuestionType
            self.engine = WizardEngine(); self.QT = QuestionType
        except Exception as e:
            root.addWidget(QLabel(tr(f"(Wizard indisponible : {e})", f"(Wizard unavailable: {e})"))); root.addStretch(1); return

        # GPU suitability banner (CTk showed this prominently at the top, colored by
        # VRAM/AMP support) — the Qt port previously never read self.engine.gpu_info at all.
        self.gpu_banner = QLabel(""); self.gpu_banner.setWordWrap(True)
        self.gpu_banner.setStyleSheet("padding:8px; border-radius:6px; font-weight:600;")
        root.addWidget(self.gpu_banner)
        self._refresh_gpu_banner()

        # Live training-time estimate (CTk's "⏱️ Estimation" line), recomputed whenever
        # iterations/batch/patch/GAN/AMP change.
        self.eta_lbl = QLabel(""); self.eta_lbl.setStyleSheet("color:#9b59b6; font-weight:600;")
        root.addWidget(self.eta_lbl)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget(); self.qv = QVBoxLayout(inner); self.qv.setSpacing(8)
        # every question gets a card; visibility follows the answers (the cards used to be
        # built once from the initial answers: Paired never showed its LQ folder, GAN never
        # showed the discriminator choice)
        self._cards, self._recs = {}, {}
        for q in self.engine.questions:
            card = self._question_card(q)
            self._cards[q.id] = card
            self.qv.addWidget(card)
        self.qv.addStretch(1)
        # any answer change (not only the combos) refreshes visibility and drops a config
        # generated from older answers
        for _qid, (_kind, _w) in self.widgets.items():
            if _kind in ("number", "text"):
                _w.textChanged.connect(self._answers_changed)
            elif _kind == "bool":
                _w.stateChanged.connect(self._refresh_dynamic)
        self._refresh_visibility()
        scroll.setWidget(inner); root.addWidget(scroll, 1)
        self._wire_eta_updates()
        self._update_eta()

        bot = QHBoxLayout(); root.addLayout(bot)
        gen = QPushButton(tr("✨ Générer la config", "✨ Generate config")); gen.clicked.connect(self._generate)
        gen.setToolTip(tr(
            "Construit une configuration d'entraînement complète et prête à l'emploi\n"
            "à partir de tes réponses (architecture, dégradations, losses, optimiseur…).\n"
            "Un aperçu s'affiche dessous ; rien n'est encore écrit sur le disque.",
            "Builds a complete, ready-to-use training configuration\n"
            "from your answers (architecture, degradations, losses, optimizer…).\n"
            "A preview shows below; nothing is written to disk yet."))
        gen.setStyleSheet('background:#27ae60; color:#fff; border:none; border-radius:8px; padding:8px 16px; font-weight:700;')
        bot.addWidget(gen)
        self.save_btn = QPushButton(tr("💾 Enregistrer…", "💾 Save…")); self.save_btn.setObjectName("ghost"); self.save_btn.setEnabled(False)
        self.save_btn.setToolTip(tr(
            "Enregistre la config générée dans un fichier (.yml Redux ou .toml NeoSR)\n"
            "que tu pourras ensuite charger dans l'onglet Configuration.",
            "Saves the generated config to a file (.yml Redux or .toml NeoSR)\n"
            "that you can then load in the Configuration tab."))
        self.save_btn.clicked.connect(self._save); bot.addWidget(self.save_btn)
        bot.addStretch(1)
        self.out = QPlainTextEdit(); self.out.setReadOnly(True); self.out.setFont(QFont("Consolas", 10))
        self.out.setMaximumHeight(220); root.addWidget(self.out)
        self._cfg = None
        self._cfg_text, self._cfg_ext = None, ".yml"

    def _question_card(self, q) -> QFrame:
        QT = self.QT
        card = QFrame(); card.setObjectName("qcard")
        card.setStyleSheet("QFrame#qcard{border:1px solid #3a3a4a; border-radius:8px;}")
        v = QVBoxLayout(card); v.setContentsMargins(12, 8, 12, 8); v.setSpacing(3)
        head = QLabel(q.text); head.setStyleSheet("font-weight:700;"); head.setWordWrap(True); v.addWidget(head)
        # recommendation / help
        rec = ""
        try:
            rec = self.engine.get_recommendation(q.id) or q.recommendation or q.help_text
        except Exception:
            rec = q.recommendation or q.help_text
        r = QLabel(rec or ""); r.setStyleSheet("color:#3b8ed0;"); r.setWordWrap(True); v.addWidget(r)
        r.setVisible(bool(rec))
        self._recs_pending = getattr(self, "_recs_pending", {})
        self._recs_pending[q.id] = (r, q)

        # tooltip = full guidance for this question (recommendation + help), so it's
        # reachable by hovering the control itself, not only the blue help line.
        _tip = "\n\n".join(s for s in (rec, getattr(q, "help_text", "")) if s and s != rec).strip()

        def _set_tip(w):
            if _tip:
                w.setToolTip(_tip)

        if q.type == QT.INFO:
            self.widgets[q.id] = ("info", None)
        elif q.type in (QT.CHOICE, QT.SCALE):
            cb = QComboBox()
            opts = [str(o) for o in (q.options or [])]
            if not opts:
                opts = self._dynamic_options(q.id)
            if not opts and q.type == QT.SCALE:
                opts = ["1", "2", "3", "4", "8"]
            for o in opts:
                cb.addItem(_option_label(o), o)  # shown label, internal key as data
            if q.default is not None and str(q.default) in opts:
                cb.setCurrentIndex(opts.index(str(q.default)))
            cb.currentIndexChanged.connect(self._refresh_dynamic)
            _set_tip(cb); v.addWidget(cb); self.widgets[q.id] = ("choice", cb)
        elif q.type == QT.BOOL:
            c = QCheckBox(tr("Activer", "Enable")); c.setChecked(bool(q.default)); _set_tip(c); v.addWidget(c)
            self.widgets[q.id] = ("bool", c)
        elif q.type == QT.PATH:
            row = QHBoxLayout(); e = QLineEdit(str(q.default or "")); _set_tip(e); row.addWidget(e, 1)
            b = QPushButton("…"); b.setObjectName("ghost"); b.setFixedWidth(40)
            b.setToolTip(tr("Parcourir pour choisir un dossier.", "Browse to choose a folder."))
            b.clicked.connect(lambda _=False, ed=e: ed.setText(QFileDialog.getExistingDirectory(self, tr("Dossier", "Folder")) or ed.text()))
            row.addWidget(b); v.addLayout(row); self.widgets[q.id] = ("text", e)
        else:  # NUMBER / TEXT
            e = QLineEdit(str(q.default if q.default is not None else "")); _set_tip(e); v.addWidget(e)
            self.widgets[q.id] = ("number" if q.type == QT.NUMBER else "text", e)
        return card

    # ── the Config tab is the single source of truth: option lists + config generation ──
    def _helper(self):
        """A hidden Config tab (read-only settings: it never writes the user's preferences).
        Its combos give the exact lists the app supports, and its generator writes the
        config — the wizard's own generator had drifted (NeoSR model_type "paired",
        degradations in a section NeoSR does not read, unsupported optimizers /
        schedulers, unregistered archs)."""
        if getattr(self, "_helper_tab", None) is None:
            from qt_ui.tabs.config_tab import ConfigTab
            self._helper_tab = ConfigTab(_ReadOnlySettings(self.settings))
            self._helper_tab.hide()
        return self._helper_tab

    def _helper_engine(self):
        tab = self._helper()
        eng = str(self.engine.get_answer("engine", "NeoSR"))
        want = "TraiNNer-Redux" if "redux" in eng.lower() else "NeoSR"
        if tab.engine_combo.currentText() != want:
            tab.engine_combo.setCurrentText(want)
        return tab

    @staticmethod
    def _combo_items(cb):
        return [cb.itemText(i) for i in range(cb.count())
                if cb.itemText(i) and not cb.itemText(i).startswith(("─", "—", "--"))]

    _DYNAMIC = {
        "scale": "get_scale_options", "arch": "get_arch_options",
        "discriminator_type": "get_discriminator_options",
        "optimizer": "get_optimizer_options", "scheduler": "get_scheduler_options",
    }

    def _dynamic_options(self, qid):
        if qid not in self._DYNAMIC:
            return []
        try:
            tab = self._helper_engine()
            cb = {"scale": tab.fields.get("scale"), "arch": tab.arch_combo,
                  "discriminator_type": getattr(tab, "disc_combo", None),
                  "optimizer": getattr(tab, "optim_combo", None),
                  "scheduler": getattr(tab, "sched_combo", None)}.get(qid)
            if cb is not None:
                return self._combo_items(cb)
        except Exception:
            pass
        try:
            return [str(o) for o in getattr(self.engine, self._DYNAMIC[qid])()]
        except Exception:
            return []

    def _answers_changed(self, *_):
        if getattr(self, "_refreshing", False) or self.engine is None:
            return
        self._collect()
        self._refresh_visibility()

    def _refresh_visibility(self):
        """Show the cards of the questions that apply to the current answers, refresh the
        recommendations (they were computed once, at construction) and drop any config
        generated from older answers (Save could write it in the new engine's format)."""
        if self.engine is None:
            return
        visible = {q.id for q in self.engine.get_visible_questions()}
        for qid, card in getattr(self, "_cards", {}).items():
            card.setVisible(qid in visible)
        for qid, (lbl, q) in getattr(self, "_recs_pending", {}).items():
            try:
                rec = self.engine.get_recommendation(q.id) or q.recommendation or q.help_text
            except Exception:
                rec = q.recommendation or q.help_text
            lbl.setText(rec or ""); lbl.setVisible(bool(rec))
        if getattr(self, "_cfg_text", None) is not None:
            self._cfg_text = None
            if hasattr(self, "save_btn"):
                self.save_btn.setEnabled(False)
            if hasattr(self, "out"):
                self.out.setPlainText(tr("(réponses modifiées — régénère la config)",
                                         "(answers changed — generate the config again)"))

    def _refresh_dynamic(self, *_):
        """Re-evaluate engine-dependent option lists when an earlier answer changes."""
        if getattr(self, "_refreshing", False) or self.engine is None:
            return
        self._refreshing = True
        try:
            self._collect()  # push current answers into the engine
            for qid, meth in self._DYNAMIC.items():
                entry = self.widgets.get(qid)
                if not entry or entry[0] != "choice":
                    continue
                cb = entry[1]; cur = _choice_value(cb)
                opts = self._dynamic_options(qid)
                if opts and opts != [cb.itemData(i) for i in range(cb.count())]:
                    cb.blockSignals(True); cb.clear()
                    for o in opts:
                        cb.addItem(_option_label(o), o)
                    cb.setCurrentIndex(opts.index(cur) if cur in opts else 0)
                    cb.blockSignals(False)
            self._collect()
            self._refresh_visibility()
        finally:
            self._refreshing = False

    def _refresh_gpu_banner(self):
        gpu = getattr(self.engine, "gpu_info", None)
        if gpu is None:
            self.gpu_banner.setText(tr("⚠ Aucun GPU détecté — l'entraînement sera très lent (CPU).",
                                        "⚠ No GPU detected — training will be very slow (CPU)."))
            self.gpu_banner.setStyleSheet("padding:8px; border-radius:6px; font-weight:600; background:#5a1e1e; color:#ff8a8a;")
            return
        suitable = gpu.is_suitable_for_training()
        if not suitable:
            bg, fg, icon = "#5a1e1e", "#ff8a8a", "🔴"
        elif not gpu.supports_amp:
            bg, fg, icon = "#5a4a1e", "#ffcf7a", "🟠"
        else:
            bg, fg, icon = "#1e5a2e", "#8affa0", "🟢"
        txt = f"{icon} {gpu.name} — {gpu.total_vram_gb:.1f} GB VRAM"
        if not suitable:
            txt += tr(" — VRAM insuffisante (<4 GB), l'entraînement risque d'échouer (OOM).",
                       " — Insufficient VRAM (<4 GB), training will likely fail (OOM).")
        elif not gpu.supports_amp:
            txt += tr(" — AMP (FP16) non supporté, l'entraînement sera plus lent.",
                       " — AMP (FP16) not supported, training will be slower.")
        self.gpu_banner.setText(txt)
        self.gpu_banner.setStyleSheet(f"padding:8px; border-radius:6px; font-weight:600; background:{bg}; color:{fg};")

    _ETA_QIDS = ("iterations", "batch_size", "patch_size", "use_gan", "use_amp")

    def _wire_eta_updates(self):
        for qid in self._ETA_QIDS:
            entry = self.widgets.get(qid)
            if not entry:
                continue
            kind, w = entry
            if kind == "choice":
                w.currentIndexChanged.connect(self._update_eta)
            elif kind == "bool":
                w.stateChanged.connect(self._update_eta)
            elif kind in ("number", "text"):
                w.textChanged.connect(self._update_eta)

    def _update_eta(self, *_):
        if self.engine is None:
            return
        try:
            self._collect()
            est = self.engine.estimate_training_time()
            self.eta_lbl.setText(tr(
                f"⏱️ Estimation : {est['readable']}  (~{est['estimated_speed']} it/s, {est['iterations']:,} itérations)",
                f"⏱️ Estimate: {est['readable']}  (~{est['estimated_speed']} it/s, {est['iterations']:,} iterations)"))
        except Exception:
            self.eta_lbl.setText("")

    def _collect(self):
        for qid, (kind, w) in self.widgets.items():
            if kind == "info" or w is None:
                continue
            if kind == "choice":
                val = _choice_value(w)
            elif kind == "bool":
                val = w.isChecked()
            elif kind == "number":
                txt = w.text().strip()
                try:
                    val = int(txt) if txt and "." not in txt else float(txt)
                except ValueError:
                    val = txt
            else:
                val = w.text().strip()
            self.engine.set_answer(qid, val)

    def _generate(self):
        if self.engine is None:
            return
        self._collect()
        invalid = []
        for q in self.engine.get_visible_questions():
            try:
                if not q.validate(self.engine.get_answer(q.id)):
                    invalid.append(q.text)
            except Exception:
                pass
        if invalid:
            QMessageBox.warning(self, tr("Réponses invalides", "Invalid answers"),
                                tr("Corrige ces réponses avant de générer la config :\n\n",
                                   "Fix these answers before generating the config:\n\n")
                                + "\n".join(f"• {t}" for t in invalid))
            return
        try:
            text, ext, cfg, warn = self._build_config()
        except Exception as e:
            self._cfg_text = None
            self.out.setPlainText(f"❌ {e}"); self.save_btn.setEnabled(False)
            return
        self._cfg_text, self._cfg_ext, self._cfg = text, ext, cfg
        head = ""
        try:  # the real estimator (same as the Config / Run tabs) on the generated config
            from src.core.compute_estimator import estimate_training_time, run_params_from_config
            est = estimate_training_time(**run_params_from_config(cfg))
            hrs = est.get("hours") or est.get("estimated_hours")
            if hrs:
                head = tr(f"# ⏱ Estimation : ~{hrs:.0f} h", f"# ⏱ Estimate: ~{hrs:.0f} h") + "\n"
        except Exception:
            pass
        self.out.setPlainText(head + (f"# ⚠ {warn}\n" if warn else "") + text)
        self.save_btn.setEnabled(True)

    def _build_config(self):
        """Answers -> the Config tab's widgets -> ConfigHandler (the same file a click on
        Save in the Configuration tab would write). Returns (text, ext, cfg dict, warning)."""
        import tempfile
        from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit
        from src.core.config_handler import ConfigHandler
        tab = self._helper_engine()
        A = self.engine.get_answer
        warn = []

        def put(key, value):
            w = tab.fields.get(key) if key not in ("arch",) else tab.arch_combo
            if w is None or value is None:
                return
            if isinstance(w, QCheckBox):
                w.setChecked(bool(value) if not isinstance(value, str) else value.lower() in ("1", "true", "yes", "on"))
            elif isinstance(w, QComboBox):
                if w.findText(str(value)) >= 0:
                    w.setCurrentText(str(value))
                else:
                    warn.append(f"{key}={value}")
            elif isinstance(w, QLineEdit):
                w.setText(str(value))
            elif hasattr(w, "setValue"):
                try:
                    w.setValue(float(value))
                except (TypeError, ValueError):
                    pass
        put("arch", A("arch"))
        put("scale", A("scale"))
        mode = str(A("dataset_mode", "otf")).lower()
        put("dataset_mode", "paired" if mode == "paired" else "otf")
        put("dataroot_gt", A("dataset_gt"))
        if mode == "paired":
            put("dataroot_lq", A("dataset_lq"))
        put("batch_size", A("batch_size"))
        put("patch_size", A("patch_size"))
        put("total_iter", A("iterations"))
        put("use_gan", A("use_gan"))
        if A("use_gan") and A("discriminator_type"):
            put("net_d_type", A("discriminator_type"))
        put("use_amp", A("use_amp"))
        put("lr", A("learning_rate"))
        put("optim_g", A("optimizer"))
        put("scheduler", A("scheduler"))
        put("name", A("experiment_name"))
        d = tab._data()
        eng = tab.engine_combo.currentText()
        ext = ".yml" if "Redux" in eng else ".toml"
        path = os.path.join(tempfile.mkdtemp(prefix="uss_wizard_"), "config" + ext)
        ch = ConfigHandler(); ch.set_engine(eng)
        ok, msg = ch.generate_config(d, path)
        if not ok:
            raise RuntimeError(msg)
        text = open(path, encoding="utf-8").read()
        if ext == ".yml":
            import yaml
            cfg = yaml.safe_load(text) or {}
        else:
            import tomllib
            cfg = tomllib.loads(text)
        w = (tr("valeur(s) refusée(s) par l'onglet Config : ", "value(s) refused by the Config tab: ")
             + ", ".join(warn)) if warn else ""
        return text, ext, cfg, w

    @staticmethod
    def _format(cfg, ext):
        try:
            if ext == ".yml":
                import yaml
                return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
            import json
            return json.dumps(cfg, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return str(cfg)

    def _save(self):
        if not getattr(self, "_cfg_text", None):
            return
        ext = self._cfg_ext
        path, _ = QFileDialog.getSaveFileName(self, tr("Enregistrer la config", "Save the config"), f"config{ext}",
                                              tr(f"Config (*{ext});;Tous (*.*)", f"Config (*{ext});;All (*.*)"))
        if not path:
            return
        if not path.lower().endswith(ext if ext == ".toml" else (".yml", ".yaml")):
            path += ext
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._cfg_text)
            QMessageBox.information(self, "Assistant", tr(f"Config enregistrée :\n{path}", f"Config saved:\n{path}"))
        except Exception as e:
            QMessageBox.critical(self, tr("Erreur", "Error"), str(e))
