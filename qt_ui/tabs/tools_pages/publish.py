"""
publish.py — Tools › Publier Modèle (PySide6) — faithful port of the v2.5.8
create_page_export: left model selector (filter + scan + radio cards + browse),
right metadata + technical sheet (template / AI generation) + package export to
~/USRS/Final model/. AI generation reuses qt_ui.tabs.run_dialogs.ai_request.
"""
from __future__ import annotations

import datetime
import os
import re
import shutil

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton,
    QPlainTextEdit, QFileDialog, QFrame, QScrollArea, QRadioButton, QButtonGroup,
    QMessageBox, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from qt_ui.i18n import tr

# provider display → settings key (matches the rest of the app)
_PROVIDER_KEY = {
    "OpenRouter (Gratuit)": "api_key_openrouter",
    "Google (Gemini)": "api_key_google", "Anthropic (Claude)": "api_key_anthropic",
    "OpenAI (ChatGPT)": "api_key_openai", "xAI (Grok)": "api_key_xai", "DeepSeek": "api_key_deepseek",
}
try:
    from src.core.ai_providers import AI_MODELS as _AIM
    _IA_MODELS = {k: v[1] for k, v in _AIM.items()}
    _PROVIDER_KEY = {k: v[0] for k, v in _AIM.items()}
except Exception:
    _IA_MODELS = {"OpenRouter (Gratuit)": ["openrouter/free"]}


def _parse_config(path: str) -> dict:
    info = {}
    try:
        if path.endswith(".toml"):
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with open(path, "rb") as fh:
                cfg = tomllib.load(fh)
        else:
            import yaml
            with open(path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
        if not cfg:
            return info
        net_g = cfg.get("network_g") or cfg.get("network") or {}
        if isinstance(net_g, dict):
            info["arch"] = net_g.get("type", net_g.get("arch", "?"))
            info["num_feat"] = str(net_g.get("num_feat", net_g.get("num_features", "?")))
        else:
            info["arch"] = str(net_g or "?")
        info["scale"] = str(cfg.get("scale", "?"))
        net_d = cfg.get("network_d")
        info["gan"] = "Oui" if net_d else "Non"
        info["net_d_type"] = str(net_d.get("type", "?")) if isinstance(net_d, dict) else "—"
        train = cfg.get("train") or {}
        if isinstance(train, dict):
            opt_g = train.get("optim_g") or {}
            info["lr"] = str(opt_g.get("lr", "?")) if isinstance(opt_g, dict) else str(opt_g or "?")
            _ds = cfg.get("datasets") or {}
            _tr = _ds.get("train") if isinstance(_ds, dict) else {}
            _tr = _tr or {}
            # Real batch-size key lives under datasets.train, as "batch_size" (NeoSR/our
            # own generator) or "batch_size_per_gpu" (Redux/BasicSR/community configs) —
            # never at train.batch_size (that key doesn't exist in either engine's schema).
            # Same fallback chain as config_handler.py's loader (line ~352-353) so this
            # fiche technique matches what the app itself would show (2026-09-17 fix).
            info["batch_size"] = str(_tr.get("batch_size", _tr.get("batch_size_per_gpu", train.get("batch_size", "?"))))
            # LR crop: Redux lq_size / NeoSR patch_size are LR; BasicSR gt_size is HR
            try:
                _sc = int(cfg.get("scale") or 0)
            except (TypeError, ValueError):
                _sc = 0
            if _tr.get("lq_size") or _tr.get("patch_size"):
                info["patch_size"] = str(_tr.get("lq_size") or _tr.get("patch_size"))
            elif _tr.get("gt_size") and _sc:
                info["patch_size"] = str(int(_tr["gt_size"]) // _sc)
            else:
                info["patch_size"] = "?"
            _ttype = str(_tr.get("type", "")).lower()
            info["otf"] = "Yes" if (cfg.get("high_order_degradation") or "otf" in _ttype
                                    or "realesrgan" in _ttype) else "No"
            _pt = (cfg.get("path") or {}).get("pretrain_network_g")
            info["pretrained"] = os.path.basename(str(_pt)) if _pt else "No"
            losses = []
            # BasicSR/community configs (and our own generated ones) name loss entries
            # "pixel_opt"/"perceptual_opt"/"gan_opt"/"ldl_opt"/"color_opt" — none contain
            # "loss" or "criterion" as a substring, so the old check never matched them.
            _known_loss_opts = ("pixel_opt", "perceptual_opt", "gan_opt", "ldl_opt",
                                 "color_opt", "style_opt", "ssim_opt", "consistency_opt")
            for k, v in train.items():
                _kl = k.lower()
                if isinstance(v, dict) and (k in _known_loss_opts or "loss" in _kl
                                             or "criterion" in _kl or _kl.endswith("_opt")):
                    if v.get("type"):
                        losses.append(f"{k}({v['type']})")
                    elif "loss_weight" in v or "weight" in v:
                        losses.append(k)
                elif "weight" in _kl and isinstance(v, (int, float)) and v:
                    losses.append(k.replace("_weight", ""))
            for _l in (train.get("losses") or []):
                if isinstance(_l, dict) and _l.get("type"):
                    losses.append(f"{_l['type']}({_l.get('loss_weight', 1)})")
            info["losses_summary"] = ", ".join(losses[:6]) or "?"
    except Exception as ex:
        info["parse_error"] = str(ex)
    return info


class _AIWorker(QThread):
    done = Signal(str)

    def __init__(self, provider, model, key, prompt):
        super().__init__(); self.provider, self.model, self.key, self.prompt = provider, model, key, prompt

    def run(self):
        try:
            from qt_ui.tabs.run_dialogs import ai_request
            self.done.emit(ai_request(self.provider, self.model, self.key, self.prompt))
        except Exception as e:
            self.done.emit(f"❌ {e}")


class PublishPage(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._scan_data = []
        self._selected = {}
        self._aiw = None

        root = QVBoxLayout(self); root.setContentsMargins(16, 12, 16, 12); root.setSpacing(6)
        t = QLabel(tr("📦 Publier Modèle", "📦 Publish Model")); t.setObjectName("title"); root.addWidget(t)
        sub = QLabel(tr("Préparer un package propre pour archiver ou partager votre modèle.", "Prepare a clean package to archive or share your model."))
        sub.setStyleSheet("color:#888;"); root.addWidget(sub)

        body = QHBoxLayout(); root.addLayout(body, 1)
        body.addWidget(self._build_left(), 1)
        body.addWidget(self._build_right(), 1)

    # ── LEFT : selector ──
    def _build_left(self):
        box = QFrame(); box.setObjectName("pcard")  # bg/border from theme QSS
        v = QVBoxLayout(box); v.setContentsMargins(10, 8, 10, 8); v.setSpacing(5)
        h = QLabel(tr("📂 Sélectionner le Modèle", "📂 Select Model")); h.setStyleSheet('color:#3498db; font-weight:700; background:transparent; font-family:"Segoe UI Emoji";')
        v.addWidget(h)
        fr = QHBoxLayout(); fl = QLabel(tr("Filtre :", "Filter:")); fl.setStyleSheet("background:transparent;"); fr.addWidget(fl)
        self.flt = QComboBox(); self.flt.addItems([tr("Tous", "All"), tr("5 derniers", "Last 5"), tr("10 derniers", "Last 10"), "NeoSR", "Redux"]); self.flt.setFixedWidth(140); self.flt.setFixedHeight(30)
        self.flt.setToolTip(tr("Filtre la liste des modèles trouvés (par récence ou par moteur).", "Filters the list of found models (by recency or engine)."))
        self.flt.currentIndexChanged.connect(self._refresh_list); fr.addWidget(self.flt)
        bscan = QPushButton(tr("🔄 Scanner", "🔄 Scan")); bscan.setFixedHeight(30); bscan.setStyleSheet('font-family:"Segoe UI Emoji"; padding:0 12px;'); bscan.clicked.connect(self._scan)
        bscan.setToolTip(tr("Parcourt les dossiers d'entraînement (NeoSR + Redux) et liste tous les\n"
                         "checkpoints .safetensors trouvés, avec leur config et leur état associés.", "Scans the training folders (NeoSR + Redux) and lists all\n.safetensors checkpoints found, with their associated config and state."))
        fr.addWidget(bscan); fr.addStretch(1); v.addLayout(fr)

        self._list_scroll = QScrollArea(); self._list_scroll.setWidgetResizable(True); self._list_scroll.setFrameShape(QFrame.NoFrame)
        self._list_scroll.setStyleSheet("QScrollArea{background:#111; border-radius:6px;}")
        self._list_host = QWidget(); self._list_host.setStyleSheet("background:transparent;")
        self._lv = QVBoxLayout(self._list_host); self._lv.setContentsMargins(6, 6, 6, 6); self._lv.setSpacing(4); self._lv.addStretch(1)
        self._list_scroll.setWidget(self._list_host); v.addWidget(self._list_scroll, 1)
        self._grp = QButtonGroup(self)
        self._list_info(tr("(cliquez 🔄 Scanner pour charger)", "(click 🔄 Scan to load)"))

        mr = QHBoxLayout(); ml = QLabel(tr("Ou parcourir :", "Or browse:")); ml.setStyleSheet("background:transparent;"); mr.addWidget(ml)
        self.manual = QLineEdit(); self.manual.setPlaceholderText(tr("chemin/vers/modele.safetensors", "path/to/model.safetensors")); mr.addWidget(self.manual, 1)
        mb = QPushButton("…"); mb.setObjectName("ghost"); mb.setFixedWidth(40); mb.clicked.connect(self._browse); mr.addWidget(mb)
        cb = QPushButton(tr("📄 Config…", "📄 Config…")); cb.setObjectName("ghost"); cb.clicked.connect(self._pick_config)
        cb.setToolTip(tr("Choisir la config d'entraînement de ce modèle (.yml/.toml).",
                         "Choose this model's training config (.yml/.toml)."))
        mr.addWidget(cb)
        self.manual.editingFinished.connect(lambda: self._from_manual(self.manual.text().strip()))
        v.addLayout(mr)
        return box

    # ── RIGHT : metadata + sheet ──
    def _build_right(self):
        box = QFrame(); box.setObjectName("pcard2")  # bg/border from theme QSS
        v = QVBoxLayout(box); v.setContentsMargins(10, 8, 10, 8); v.setSpacing(5)
        h = QLabel(tr("📝 Métadonnées & Fiche Technique", "📝 Metadata & Datasheet")); h.setStyleSheet('color:#3498db; font-weight:700; background:transparent; font-family:"Segoe UI Emoji";')
        v.addWidget(h)
        self.f_name = self._field(v, tr("Nom du modèle :", "Model Name:"), "ex: Crysisjim SPANPlus Deband_HARD")
        self.f_name.setToolTip(tr("Nom du modèle publié. Sert aussi à nommer le dossier du package final.", "Name of the published model. Also used to name the final package folder."))
        self.f_ver = self._field(v, tr("Version :", "Version:"), "ex: 1.0")
        self.f_ver.setToolTip(tr("Numéro de version (ex: 1.0). Ajouté au nom du dossier du package.", "Version number (e.g. 1.0). Added to the package folder name."))
        self.f_author = self._field(v, tr("Auteur :", "Author:"), "ex: Crysisjim")
        self.f_author.setToolTip(tr("Ton pseudo/nom, inscrit dans la fiche technique.", "Your handle/name, written into the datasheet."))
        self.f_name.textChanged.connect(self._update_dest); self.f_ver.textChanged.connect(self._update_dest)

        ln = QLabel(tr("Notes personnelles :", "Personal Notes:")); ln.setStyleSheet("background:transparent;"); v.addWidget(ln)
        self.notes = QPlainTextEdit(); self.notes.setFixedHeight(56); v.addWidget(self.notes)
        self.info_lbl = QLabel(tr("ℹ Aucun modèle sélectionné", "ℹ No model selected")); self.info_lbl.setStyleSheet("color:#888; background:transparent;")
        self.info_lbl.setWordWrap(True); v.addWidget(self.info_lbl)

        lf = QLabel(tr("📄 Fiche Technique du Modèle :", "📄 Model Datasheet:")); lf.setStyleSheet('background:transparent; font-weight:700; font-family:"Segoe UI Emoji";')
        v.addWidget(lf)
        self.fiche = QPlainTextEdit(); self.fiche.setMinimumHeight(180); self.fiche.setFont(QFont("Consolas", 10)); v.addWidget(self.fiche, 1)

        ia = QHBoxLayout(); il = QLabel(tr("🤖 IA :", "🤖 AI:")); il.setStyleSheet('background:transparent; font-family:"Segoe UI Emoji";'); ia.addWidget(il)
        from qt_ui.widgets import KeyedCombo
        self.ia_provider = KeyedCombo(list(_IA_MODELS.keys())); self.ia_provider.setFixedWidth(190); self.ia_provider.setFixedHeight(30)
        self.ia_provider.keyChanged.connect(self._on_provider); ia.addWidget(self.ia_provider)
        self.ia_model = QComboBox(); self.ia_model.setEditable(True); self.ia_model.setFixedWidth(200); self.ia_model.setFixedHeight(30); ia.addWidget(self.ia_model)
        from qt_ui.widgets import combo_open_on_click
        combo_open_on_click(self.ia_model)  # click anywhere opens the list
        ia.addStretch(1); v.addLayout(ia)
        self._on_provider(self.ia_provider.currentText())

        gr = QHBoxLayout()
        bt = QPushButton(tr("📋 Générer (template)", "📋 Generate (template)")); bt.setFixedHeight(32)
        bt.setToolTip(tr("Pré-remplit la fiche technique avec les infos lues dans la config\n"
                      "(arch, scale, losses, batch…). Les champs [À compléter] restent à remplir à la main.", "Pre-fills the datasheet with info read from the config\n(arch, scale, losses, batch…). The [To complete] fields stay to be filled by hand."))
        bt.setStyleSheet('background:#3a3f4b; color:#fff; border:none; border-radius:6px; padding:0 14px; font-family:"Segoe UI Emoji";')
        bt.clicked.connect(self._gen_template); gr.addWidget(bt)
        ba = QPushButton(tr("🤖 Générer avec IA", "🤖 Generate with AI")); ba.setFixedHeight(32)
        ba.setToolTip(tr("Demande à l'IA de rédiger une fiche de release complète (style communauté\n"
                      "'Enhance Everything!') à partir des infos du modèle. Nécessite une clé API configurée.", "Asks the AI to write a complete release sheet (community style\n'Enhance Everything!') from the model info. Requires a configured API key."))
        ba.setStyleSheet('background:#8e44ad; color:#fff; border:none; border-radius:6px; padding:0 14px; font-family:"Segoe UI Emoji";')
        ba.clicked.connect(self._gen_ai); gr.addWidget(ba); gr.addStretch(1); v.addLayout(gr)

        self.dest_lbl = QLabel(tr("📁 Destination : (sélectionnez un modèle)", "📁 Destination: (select a model)")); self.dest_lbl.setStyleSheet("color:#888; background:transparent;")
        self.dest_lbl.setWordWrap(True); v.addWidget(self.dest_lbl)
        self.export_btn = QPushButton(tr("📦 Exporter le Package", "📦 Export Package"))
        self.export_btn.setToolTip(tr("Crée un dossier propre dans ~/USRS/Final model/ contenant :\n"
                                   "le modèle .safetensors, la fiche technique, la config (Option/),\n"
                                   "le checkpoint de reprise (resume/) et l'état d'entraînement.\n"
                                   "Prêt à archiver ou partager. Le modèle d'origine n'est pas déplacé.", "Creates a clean folder in <engine>/Final model/ containing:\nthe .safetensors model, the datasheet, the config (Option/),\nthe resume checkpoint (resume/) and the training state.\nReady to archive or share. The original model is not moved."))
        self.export_btn.setStyleSheet(
            'QPushButton{background:#16a085; color:#fff; border:none; border-radius:8px; padding:9px; font-weight:700; font-family:"Segoe UI Emoji";}'
            'QPushButton:hover{background:#1abc9c;}'
            'QPushButton:pressed{background:#128a72; padding-top:11px;}')
        self.export_btn.clicked.connect(self._export); v.addWidget(self.export_btn)
        return box

    def _field(self, layout, label, ph):
        r = QHBoxLayout(); lb = QLabel(label); lb.setFixedWidth(120); lb.setStyleSheet("background:transparent;"); r.addWidget(lb)
        e = QLineEdit(); e.setPlaceholderText(ph); r.addWidget(e, 1); layout.addLayout(r)
        return e

    # ── selector logic ──
    def _list_clear(self):
        while self._lv.count():
            it = self._lv.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    def _list_info(self, text):
        self._list_clear()
        lb = QLabel(text); lb.setStyleSheet("color:#555; background:transparent;"); self._lv.addWidget(lb); self._lv.addStretch(1)

    def _scan(self):
        from src.core import engine_paths as _ep
        bases = {"NeoSR": os.path.join(_ep.neosr_path(), "experiments"),
                 "Redux": os.path.join(_ep.redux_path(), "experiments")}
        found = []
        for engine, exp_root in bases.items():
            if not os.path.isdir(exp_root):
                continue
            engine_root = os.path.dirname(exp_root)
            for exp_name in os.listdir(exp_root):
                exp_dir = os.path.join(exp_root, exp_name)
                models_dir = os.path.join(exp_dir, "models")
                if not os.path.isdir(models_dir):
                    continue
                config_path = self._find_config(exp_dir, exp_name, engine_root)
                state_dir = os.path.join(exp_dir, "training_states")
                states = sorted(f for f in os.listdir(state_dir) if f.endswith(".state")) if os.path.isdir(state_dir) else []
                for mf in os.listdir(models_dir):
                    # NeoSR saves .pth, Redux .safetensors; discriminators are not exported
                    if not mf.endswith((".safetensors", ".pth")) or mf.startswith("net_d"):
                        continue
                    mpath = os.path.join(models_dir, mf)
                    m = re.search(r"_(\d+)\.(?:safetensors|pth)$", mf); iter_m = m.group(1) if m else None
                    state_file = None
                    if iter_m:
                        # exact iteration: "100" must not pick "1000.state"
                        state_file = next((os.path.join(state_dir, s) for s in states
                                           if re.fullmatch(rf"0*{int(iter_m)}\.state", s)), None)
                    elif "latest" in mf and states:
                        # "latest" = the newest state, by iteration number
                        _n = [(int(x.group(1)), st) for st in states if (x := re.fullmatch(r"(\d+)\.state", st))]
                        if _n:
                            iter_m = str(max(_n)[0]) + " (latest)"
                            state_file = os.path.join(state_dir, max(_n)[1])
                    found.append({"engine": engine, "exp_name": exp_name, "model_path": mpath, "model_file": mf,
                                  "config_path": config_path, "state_file": state_file, "iter": iter_m or "?",
                                  "mtime": os.path.getmtime(mpath) if os.path.isfile(mpath) else 0})
        found.sort(key=lambda x: x["mtime"], reverse=True)
        self._scan_data = found
        self._refresh_list()

    @staticmethod
    def _find_config(exp_dir, exp_name, engine_root):
        dirs = [exp_dir, os.path.join(engine_root, "options"), os.path.join(engine_root, "options", "train")]
        for ext in (".yaml", ".yml", ".toml"):
            for sd in dirs:
                c = os.path.join(sd, exp_name + ext)
                if os.path.isfile(c):
                    return c
        try:
            for fn in os.listdir(exp_dir):
                if fn.endswith((".yaml", ".yml", ".toml")):
                    return os.path.join(exp_dir, fn)
        except Exception:
            pass
        for od in (os.path.join(engine_root, "options", "train"), os.path.join(engine_root, "options")):
            if os.path.isdir(od):
                for fn in os.listdir(od):
                    if fn.endswith((".yaml", ".yml", ".toml")):
                        stem = os.path.splitext(fn)[0]
                        if stem in exp_name or exp_name in stem:
                            return os.path.join(od, fn)
        return None

    def _refresh_list(self, *_):
        flt = self.flt.currentText()
        data = self._scan_data
        if flt == tr("5 derniers", "Last 5"):
            data = data[:5]
        elif flt == tr("10 derniers", "Last 10"):
            data = data[:10]
        elif flt in ("NeoSR", "Redux"):
            data = [d for d in data if d["engine"] == flt]
        if not data:
            self._list_info(tr("Aucun modèle trouvé. Cliquez 🔄 Scanner.", "No model found. Click 🔄 Scan."))
            return
        self._list_clear()
        self._grp = QButtonGroup(self)
        for d in data:
            card = QFrame(); card.setStyleSheet("QFrame{background:#23233a; border-radius:6px;}")
            cv = QVBoxLayout(card); cv.setContentsMargins(8, 5, 8, 5); cv.setSpacing(1)
            rb = QRadioButton(d["exp_name"]); rb.setStyleSheet("font-weight:700; background:transparent;")
            rb.toggled.connect(lambda on, dd=d: self._on_card(dd) if on else None)
            self._grp.addButton(rb); cv.addWidget(rb)
            try:
                ts = datetime.datetime.fromtimestamp(d["mtime"]).strftime("%d/%m/%y %H:%M")
            except Exception:
                ts = "?"
            kb = os.path.getsize(d["model_path"]) // 1024 if os.path.isfile(d["model_path"]) else 0
            l1 = QLabel(f"  {d['engine']} · iter {d['iter']} · {kb:,} KB · 🕐 {ts}")
            l1.setStyleSheet("color:#888; background:transparent;"); cv.addWidget(l1)
            l2 = QLabel(f"  {tr('Config', 'Config')} {'✅' if d.get('config_path') else '❌'}  {tr('État', 'State')} {'✅' if d.get('state_file') else '❌'}  — {d['model_file']}")
            l2.setStyleSheet("color:#666; background:transparent;"); cv.addWidget(l2)
            self._lv.addWidget(card)
        self._lv.addStretch(1)

    def _on_card(self, d):
        self.manual.blockSignals(True); self.manual.clear(); self.manual.blockSignals(False)
        self._on_select(d)

    def _on_select(self, d):
        self._selected = d
        self.f_name.setText(d["exp_name"])
        if d.get("config_path") and os.path.isfile(d["config_path"]):
            info = _parse_config(d["config_path"])
            self.info_lbl.setText(f"Config : {os.path.basename(d['config_path'])} | Arch : {info.get('arch','?')} | "
                                  f"Scale : {info.get('scale','?')}x | " + tr("Losses", "Losses") + f" : {info.get('losses_summary','?')}")
        else:
            self.info_lbl.setText(tr("Config introuvable — utilisez « 📄 Config… » pour la retrouver.",
                                     "Config not found — use “📄 Config…” to locate it."))
        self._update_dest()

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Sélectionner le modèle", "Select the model"), "",
                                           tr("Modèles (*.safetensors *.pth);;Tous (*.*)", "Models (*.safetensors *.pth);;All (*.*)"))
        if not p:
            return
        self.manual.setText(p)
        self._from_manual(p)

    def _pick_config(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("Config d'entraînement", "Training config"), "",
                                           tr("Config (*.yml *.yaml *.toml);;Tous (*.*)", "Config (*.yml *.yaml *.toml);;All (*.*)"))
        if p:
            d = dict(self._selected or {})
            d["config_path"] = p
            self._on_select(d)

    def _from_manual(self, p):
        if not p or not os.path.isfile(p):
            return
        if (self._selected or {}).get("model_path") == p:
            return
        exp_dir = os.path.dirname(os.path.dirname(p)); exp_name = os.path.basename(exp_dir)
        engine_root = os.path.dirname(os.path.dirname(exp_dir))
        d = {"model_path": p, "model_file": os.path.basename(p), "exp_name": exp_name or os.path.splitext(os.path.basename(p))[0],
             "config_path": self._find_config(exp_dir, exp_name, engine_root), "state_file": None, "engine": "?", "iter": "?"}
        self._on_select(d)

    # ── metadata / fiche ──
    def _on_provider(self, name):
        self.ia_model.clear(); self.ia_model.addItems(_IA_MODELS.get(name, []))

    def _update_dest(self, *_):
        name = self.f_name.text().strip(); ver = self.f_ver.text().strip()
        if name:
            from src.core import engine_paths as _ep
            dest = os.path.join(_ep.base_engine_path(), "Final model", f"{name} {ver}".strip())
            self.dest_lbl.setText(f"📁 Destination : {dest}")
        else:
            self.dest_lbl.setText(tr("📁 Destination : (entrez un nom de modèle)", "📁 Destination: (enter a model name)"))

    def _meta(self):
        d = self._selected or {}
        name = self.f_name.text().strip() or d.get("exp_name", tr("Modèle", "Model"))
        ver = self.f_ver.text().strip() or "1.0"
        author = self.f_author.text().strip() or "?"
        notes = self.notes.toPlainText().strip()
        cfg = d.get("config_path")
        info = _parse_config(cfg) if cfg and os.path.isfile(cfg) else {}
        return name, ver, author, notes, info, d.get("iter", "?")

    def _gen_template(self):
        name, ver, author, notes, info, it = self._meta()
        date_str = datetime.date.today().strftime("%d/%m/%Y")
        ph = tr("[À compléter — description libre du modèle, cas d'usage, observations…]", "[To complete — free-form model description, use cases, observations…]")
        fiche = (f"**{name} {ver}**\n\n"
                 f"**Scale:** {info.get('scale','?')}\n**Architecture:** {info.get('arch','?')}\n**Links:** [{tr('À compléter', 'To complete')}]\n\n"
                 f"**Author:** {author}\n**License:** CC BY 4.0\n"
                 f"**Purpose:** [{tr('À compléter', 'To complete')} — ex: Deband, Restoration, Super-Resolution]\n"
                 f"**Subject:** [{tr('À compléter', 'To complete')} — ex: Live action, Anime, Animation]\n"
                 f"**Input Type:** Images\n**Date:** {date_str}\n\n"
                 f"**Dataset:** [{tr('À compléter', 'To complete')}]\n**Dataset Size:** [{tr('À compléter', 'To complete')}]\n"
                 f"**OTF (on the fly augmentations):** {info.get('otf', '?')}\n**Pretrained Model:** {info.get('pretrained', '?')}\n"
                 f"**Iterations:** {it}\n**Batch Size:** {info.get('batch_size','?')}\n**LQ Size:** {info.get('patch_size','?')}\n\n"
                 f"**Description:**\n{notes if notes else ph}\n\n"
                 f"**Showcase:**\n[lien slow.pics ou image de comparaison]\n")
        self.fiche.setPlainText(fiche)

    def _gen_ai(self):
        provider = self.ia_provider.currentText(); model = self.ia_model.currentText().strip()
        if not model:
            QMessageBox.warning(self, tr("Modèle IA", "AI Model"), tr("Sélectionnez un modèle.", "Select a model.")); return
        key = self.settings.get(_PROVIDER_KEY.get(provider, ""), "")
        if not key:
            QMessageBox.critical(self, tr("Clé API manquante", "Missing API key"),
                                 tr(f"Aucune clé sauvegardée pour {provider}.\nConfigure-la dans Réglages › Apparence & API "
                                 "ou Configuration › Vérification IA.",
                                 f"No key saved for {provider}.\nConfigure it in Settings › Appearance & API "
                                 "or Configuration › AI Verification."))
            return
        name, ver, author, notes, info, it = self._meta()
        prompt = (
            "You are an expert in AI Super-Resolution models. Generate a model release card in the style of the "
            "'Enhance Everything!' Discord community. Reply ONLY with the card text, no intro, no markdown fences.\n\n"
            f"Use this EXACT bold-markdown English format:\n**{name} {ver}**\n\nScale: [value]\nArchitecture: [value]\n"
            "Links: [To fill]\n\nAuthor: [value]\nLicense: CC BY 4.0\nPurpose: [infer]\nSubject: [infer]\nInput Type: Images\n"
            "Date: [today]\n\nDataset: [infer or 'Custom Dataset']\nDataset Size: [unknown if not provided]\n"
            "OTF (on the fly augmentations): [value]\nPretrained Model: [value]\nIterations: [value]\nBatch Size: [value]\nLQ Size: [value]\n\n"
            "Description:\n[2-4 sentences from purpose/notes]\n\nShowcase:\n[To fill]\n\n--- DATA ---\n"
            f"Model: {name} v{ver}\nAuthor: {author}\nArchitecture: {info.get('arch','unknown')}\nScale: {info.get('scale','unknown')}x\n"
            f"Num feat: {info.get('num_feat','unknown')}\nGAN: {info.get('gan','No')}\nDiscriminator: {info.get('net_d_type','none')}\n"
            f"Losses: {info.get('losses_summary','unknown')}\nLR: {info.get('lr','unknown')}\nBatch: {info.get('batch_size','unknown')}\n"
            f"LQ size: {info.get('patch_size','unknown')}\nOTF: {info.get('otf','unknown')}\n"
            f"Pretrained model: {info.get('pretrained','unknown')}\nIterations: {it}\nNotes (use as Description context): {notes or 'none'}\n")
        self.fiche.setPlainText(tr(f"⏳ Génération IA en cours ({provider})…", f"⏳ AI generation in progress ({provider})…"))
        self._aiw = _AIWorker(provider, model, key, prompt)
        self._aiw.done.connect(lambda txt: (self.fiche.setPlainText(txt), setattr(self, "_aiw", None)))
        self._aiw.start()

    # ── export ──
    def _export(self):
        name = self.f_name.text().strip(); ver = self.f_ver.text().strip()
        if not name:
            QMessageBox.critical(self, tr("Erreur", "Error"), tr("Entrez un nom de modèle.", "Enter a model name.")); return
        d = self._selected or {}
        if self.manual.text().strip():
            self._from_manual(self.manual.text().strip()); d = self._selected or {}
        model_path = d.get("model_path", "")
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.critical(self, tr("Erreur", "Error"), tr("Aucun modèle sélectionné ou fichier introuvable.", "No model selected or file not found.")); return
        fiche_text = self.fiche.toPlainText().strip()
        if not fiche_text and QMessageBox.question(self, tr("Fiche vide", "Empty sheet"), tr("La fiche technique est vide. Continuer ?", "The technical sheet is empty. Continue?")) != QMessageBox.Yes:
            return
        folder = f"{name} {ver}".strip()
        if re.search(r'[<>:"/\\|?*\x00-\x1f]', folder) or folder.endswith((".", " ")) or \
                folder.upper().split(".")[0] in ("CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                                 *(f"LPT{i}" for i in range(1, 10))):
            QMessageBox.critical(self, tr("Erreur", "Error"),
                                 tr("Nom ou version invalide pour un dossier Windows (\\ / : * ? \" < > | interdits).",
                                    "Invalid name or version for a Windows folder (\\ / : * ? \" < > | not allowed)."))
            return
        from src.core import engine_paths as _ep
        final_root = os.path.abspath(os.path.join(_ep.base_engine_path(), "Final model"))
        dest = os.path.abspath(os.path.join(final_root, folder))
        if os.path.dirname(dest) != final_root:
            QMessageBox.critical(self, tr("Erreur", "Error"), tr("Destination hors de « Final model ».", "Destination outside “Final model”."))
            return
        if os.path.exists(dest) and QMessageBox.question(self, tr("Dossier existant", "Existing folder"), tr(f"{dest}\n\nÉcraser ?", f"{dest}\n\nOverwrite?")) != QMessageBox.Yes:
            return
        try:
            for sub in ("Option", "resume", "Trainning state"):
                os.makedirs(os.path.join(dest, sub), exist_ok=True)
            # the published file keeps its real format: a .pth is not renamed .safetensors
            ext = os.path.splitext(model_path)[1].lower() or ".safetensors"
            shutil.copy2(model_path, os.path.join(dest, f"{folder}{ext}"))
            if fiche_text:
                with open(os.path.join(dest, tr("Fiche Technique du Modèle.txt", "Model Datasheet.txt")), "w", encoding="utf-8") as fh:
                    fh.write(fiche_text)
            cfg = d.get("config_path")
            if cfg and os.path.isfile(cfg):
                shutil.copy2(cfg, os.path.join(dest, "Option", os.path.basename(cfg)))
            shutil.copy2(model_path, os.path.join(dest, "resume", os.path.basename(model_path)))
            # resume needs the discriminator of the same iteration too (GAN runs)
            mdir = os.path.dirname(model_path)
            it = re.search(r"_(\d+|latest)\.(?:safetensors|pth)$", os.path.basename(model_path))
            if it:
                for sub in (mdir, os.path.join(mdir, "resume_models")):
                    if os.path.isdir(sub):
                        for fn in os.listdir(sub):
                            if (fn.startswith("net_d") or sub.endswith("resume_models")) and \
                                    re.search(rf"_{it.group(1)}\.(?:safetensors|pth)$", fn):
                                shutil.copy2(os.path.join(sub, fn), os.path.join(dest, "resume", fn))
            sf = d.get("state_file")
            if sf and os.path.isfile(sf):
                shutil.copy2(sf, os.path.join(dest, "Trainning state", os.path.basename(sf)))
            QMessageBox.information(self, tr("✅ Export réussi", "✅ Export successful"), tr(f"Package créé :\n{dest}", f"Package created:\n{dest}"))
            try:
                from qt_ui.notify import notify
                notify(self.settings, tr("Modèle publié", "Model published"), folder, "notif_win11_batch")
            except Exception:
                pass
            try:
                os.startfile(dest)
            except Exception:
                pass
        except Exception as ex:
            QMessageBox.critical(self, tr("Erreur Export", "Export error"), str(ex))
