<div align="center">
  <img src="assets/icon.png" alt="Universal SR Studio" width="140"/>
  <h1>Universal SR Studio</h1>
  <p>Graphical interface for training super-resolution AI models<br>with <strong>NeoSR</strong> and <strong>traiNNer-Redux</strong> engines.</p>

  <a href="https://github.com/Crysisjim/Universal-SR-Studio/releases"><img src="https://img.shields.io/badge/Version-2.5.0-blue" alt="Version"/></a>
  <a href="https://github.com/Crysisjim/Universal-SR-Studio/wiki"><img src="https://img.shields.io/badge/📖_Wiki-Documentation-informational" alt="Wiki"/></a>
  <img src="https://img.shields.io/badge/Platform-Windows-lightgrey" alt="Platform"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20required-brightgreen" alt="GPU"/>

  <br/><br/>

   🇬🇧 [English](#english) | 🇫🇷 [Français](#français)
</div>

---

<a name="english"></a>
## 🇬🇧 English

A graphical interface for training and managing super-resolution AI models with **NeoSR** and **traiNNer-Redux** engines.

[![📖 Wiki — Full Documentation](https://img.shields.io/badge/📖_Wiki-Full_Documentation-blue?style=for-the-badge)](https://github.com/Crysisjim/Universal-SR-Studio/wiki)

### What's new in v2.5.0

- **Quick Upscale — batch serialization** — sequential output numbering (`00000.png … 24999.png`) for direct video reassembly, with configurable start index
- **Quick Upscale — natural sort** — files processed in correct order (`frame_1, frame_2, …, frame_10`) instead of lexicographic
- **Quick Upscale — Color Fix ATWT** — adaptive wavelet color correction popup with CPU/CUDA selection
- **Quick Upscale — CUDA fallback** — SPANPlus, SMoSR, GFISRv2, SpanC, SpanF now run via traiNNer venv subprocess, fixing `cudaErrorNoKernelImageForDevice` on Pascal GPUs (GTX 1080/Ti)
- **Quick Upscale — UI layout** — tighter spacing matching v2.2, centered controls, proper button separation
- **Benchmark — ECO Training Mode** — tests `eco_training` and `eco_personal` now fully functional; fixed `FileExistsError` in traiNNer `make_exp_dirs` for `eco_pretrain_g` paths
- **Benchmark — personal model tests** — `bicubic_personal` + `eco_personal` test upscale pipeline with your own model after training
- **traiNNer-Redux as default engine** — replaces NeoSR as the default when no preference is set
- **AI Assistant — Gemini 3.1 Pro** — updated provider references

### Features

- **Configuration wizard** — visual TOML/YAML config editor for NeoSR and traiNNer-Redux
- **Training monitor** — real-time loss curves, PSNR/SSIM, TensorBoard integration, live GPU stats
- **OTF preview** — on-the-fly degradation pipeline preview (blur, noise, JPEG, compression, screentone, dithering, …)
- **Benchmark suite** — automated architecture and feature benchmarks with resume support (Sprint 20: SpanF, SpanC, GFISRv2, SMoSR, ECO mode, 30+ features)
- **Model tools** — quick upscale (batch + serialization + color fix), model export (safetensors), model packaging
- **Dataset tools** — tile splitter, validation rotation, LMDB converter
- **Distributed training** — multi-machine training coordination
- **Training queue** — schedule multiple training sessions back-to-back
- **20+ themes** — customizable UI themes
- **Bilingual UI** — French / English interface

### Quick Start — Portable (recommended)

1. Download `Universal_SR_Studio_v2.5.0_portable.zip` from [Releases](https://github.com/Crysisjim/Universal-SR-Studio/releases)
2. Extract anywhere
3. Run `Universal_SR_Studio.exe`
4. On first launch, choose your language (FR/EN), then go to **⚙️ Settings** → the built-in installer handles everything else

> No Python installation required. The portable version is fully self-contained.

### Prerequisites

- **Windows 10/11**
- **NVIDIA GPU** with CUDA support (8 GB+ VRAM recommended)
- **Internet connection** for the first setup (engine download)

That's it. Universal SR Studio handles the rest automatically via the **⚙️ Settings** tab.

### Automatic Setup (via Settings tab)

| Step | What it does |
|------|-------------|
| **GPU detection** | Detects your GPU and recommends the correct PyTorch + CUDA version |
| **Engine install** | Downloads and installs NeoSR and/or traiNNer-Redux from their official repositories |
| **Virtual environment** | Creates an isolated `.venv` for each engine |
| **PyTorch** | Installs the correct CUDA-compatible version automatically |
| **Dependencies** | Installs all engine-specific packages |

Just open the **⚙️ Settings** tab, choose which engine(s) to install, and click — the console window shows live progress.

### Expected folder structure

```
~/IA_Engine/
├── traiNNer-redux/        (installed via Settings)
│   └── .venv/
├── neosr/                 (installed via Settings)
│   └── .venv/
├── datasets/
│   ├── train/HR/          (your training images)
│   └── val/
│       ├── GT/
│       └── LQ/
└── Option Custom/         (custom degradation presets)
```

### Source installation (developers)

```bash
git clone https://github.com/Crysisjim/Universal-SR-Studio.git
cd Universal-SR-Studio
pip install -r requirements.txt
python main.py
```

Then use the **⚙️ Settings** tab to install the training engines.

### Tabs overview

| Tab | Description |
|-----|-------------|
| 😊 Assistant | Guided setup wizard for beginners |
| 📝 Configuration | TOML/YAML config editor with live OTF preview |
| 🚀 Training | Start/stop training, live curves, TensorBoard |
| 🔧 Tools | Benchmark, quick upscale, model export, dataset tools |
| 📋 Queue | Schedule multiple training sessions |
| ⚙️ Settings | Engine installer, paths, language, appearance, API keys |
| 🌐 Distributed | Multi-machine training (experimental) |
| 📖 Wiki | Opens the GitHub wiki documentation in your browser |

### Benchmarks (CLI)

```bash
# Architecture benchmark (traiNNer-Redux)
python src/core/benchmark_runner.py --engine redux --type arch

# Feature benchmark (NeoSR)
python src/core/benchmark_runner.py --engine neosr --type feature

# List available tests
python src/core/benchmark_runner.py --list
```

### Roadmap — v2.5.5

| Feature | Description |
|---------|-------------|
| **Persistent batch subprocess** | Keep the model loaded in VRAM across all frames — no reload per image. Eliminates ~17-25h overhead on 30k-frame batches. |
| **Temporal SR training** | Enter a video as GT reference, extract frame sequences, train TSPAN/TSPANv2 with sliding window input `[B, N, C, H, W]`. Full temporal consistency pipeline. |
| **Temporal SR inference** | Sliding window N-frame inference with TSPAN/TSPANv2 and frame reassembly. |
| **NVIDIA NIM provider** | `build.nvidia.com` as a new AI assistant provider — OpenAI-compatible API, free model credits (Llama, Mistral, Phi…). |
| **VOSR** | Third inference engine (CVPR 2026, diffusion-based SR). |

### Contributing

Pull requests welcome. For major changes, open an issue first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Open a pull request

### License

[MIT](LICENSE) — free to use, modify, and distribute.

---

<a name="français"></a>
## 🇫🇷 Français

Interface graphique pour l'entraînement et la gestion de modèles d'IA super-résolution avec les moteurs **NeoSR** et **traiNNer-Redux**.

[![📖 Wiki — Documentation complète](https://img.shields.io/badge/📖_Wiki-Documentation_complète-blue?style=for-the-badge)](https://github.com/Crysisjim/Universal-SR-Studio/wiki)

### Nouveautés v2.5.0

- **Quick Upscale — sérialisation sortie** — numérotation séquentielle (`00000.png … 24999.png`) pour réassemblage vidéo direct, index de départ configurable
- **Quick Upscale — tri naturel** — fichiers traités dans le bon ordre (`frame_1, frame_2, …, frame_10`) au lieu de l'ordre lexicographique
- **Quick Upscale — Color Fix ATWT** — correction couleur par ondelettes adaptative, popup avec sélection CPU/CUDA
- **Quick Upscale — fallback CUDA** — SPANPlus, SMoSR, GFISRv2, SpanC, SpanF via subprocess venv traiNNer, résout `cudaErrorNoKernelImageForDevice` sur GPU Pascal (GTX 1080/Ti)
- **Quick Upscale — UI** — layout compact identique à v2.2, contrôles centrés, espacement bouton corrigé
- **Benchmark — ECO Training Mode** — tests `eco_training` et `eco_personal` entièrement fonctionnels ; bug `FileExistsError` traiNNer `make_exp_dirs` corrigé
- **Benchmark — tests modèle perso** — `bicubic_personal` + `eco_personal` testent le pipeline upscale avec votre propre modèle
- **traiNNer-Redux par défaut** — remplace NeoSR comme moteur par défaut
- **Assistant IA — Gemini 3.1 Pro** — références providers mises à jour

### Fonctionnalités

- **Assistant de configuration** — éditeur visuel TOML/YAML pour NeoSR et traiNNer-Redux
- **Moniteur d'entraînement** — courbes de perte en temps réel, PSNR/SSIM, intégration TensorBoard, stats GPU live
- **Aperçu OTF** — prévisualisation du pipeline de dégradation à la volée (flou, bruit, JPEG, compression, screentone, dithering, …)
- **Suite de benchmarks** — benchmarks automatisés d'architectures et de features avec reprise (Sprint 20 : SpanF, SpanC, GFISRv2, SMoSR, ECO mode, 30+ features)
- **Outils modèles** — upscale rapide (batch + sérialisation + color fix), export modèle (safetensors), packaging
- **Outils dataset** — découpeur de tuiles, rotation de validation, convertisseur LMDB
- **Entraînement distribué** — coordination multi-machines
- **File d'entraînements** — planifier plusieurs sessions à la suite
- **20+ thèmes** — thèmes UI personnalisables
- **Interface bilingue** — Français / Anglais

### Démarrage rapide — Portable (recommandé)

1. Télécharger `Universal_SR_Studio_v2.5.0_portable.zip` depuis les [Releases](https://github.com/Crysisjim/Universal-SR-Studio/releases)
2. Extraire n'importe où
3. Lancer `Universal_SR_Studio.exe`
4. Au premier lancement, choisir la langue (FR/EN), puis aller dans **⚙️ Paramètres** → l'installeur intégré gère le reste

> Aucune installation Python requise. La version portable est entièrement autonome.

### Prérequis

- **Windows 10/11**
- **GPU NVIDIA** avec support CUDA (8 Go+ VRAM recommandé)
- **Connexion internet** pour le premier setup (téléchargement des moteurs)

C'est tout. Universal SR Studio gère le reste automatiquement via l'onglet **⚙️ Paramètres**.

### Installation automatique (via l'onglet Paramètres)

| Étape | Action |
|-------|--------|
| **Détection GPU** | Détecte le GPU et recommande la bonne version PyTorch + CUDA |
| **Installation moteur** | Télécharge et installe NeoSR et/ou traiNNer-Redux depuis leurs dépôts officiels |
| **Environnement virtuel** | Crée un `.venv` isolé pour chaque moteur |
| **PyTorch** | Installe la version compatible CUDA automatiquement |
| **Dépendances** | Installe tous les packages spécifiques au moteur |

Ouvrir l'onglet **⚙️ Paramètres**, choisir le(s) moteur(s) à installer, et cliquer — la console affiche la progression en direct.

### Structure de dossiers attendue

```
~/IA_Engine/
├── traiNNer-redux/        (installé via Paramètres)
│   └── .venv/
├── neosr/                 (installé via Paramètres)
│   └── .venv/
├── datasets/
│   ├── train/HR/          (vos images d'entraînement)
│   └── val/
│       ├── GT/
│       └── LQ/
└── Option Custom/         (presets de dégradation personnalisés)
```

### Installation source (développeurs)

```bash
git clone https://github.com/Crysisjim/Universal-SR-Studio.git
cd Universal-SR-Studio
pip install -r requirements.txt
python main.py
```

Puis utiliser l'onglet **⚙️ Paramètres** pour installer les moteurs d'entraînement.

### Aperçu des onglets

| Onglet | Description |
|--------|-------------|
| 😊 Assistant | Wizard de configuration guidée pour débutants |
| 📝 Configuration | Éditeur TOML/YAML avec prévisualisation OTF live |
| 🚀 Entraînement | Démarrer/arrêter, courbes live, TensorBoard |
| 🔧 Outils | Benchmark, upscale rapide, export modèle, outils dataset |
| 📋 File d'attente | Planifier plusieurs sessions d'entraînement |
| ⚙️ Paramètres | Installeur moteurs, chemins, langue, apparence, clés API |
| 🌐 Distribué | Entraînement multi-machines (expérimental) |
| 📖 Wiki | Ouvre la documentation wiki GitHub dans le navigateur |

### Roadmap — v2.5.5

| Feature | Description |
|---------|-------------|
| **Subprocess persistant (batch)** | Modèle chargé en VRAM sur toute la durée du batch — plus de rechargement par image. Élimine 17-25h de surcoût sur 30k frames. |
| **Entraînement Temporal SR** | Vidéo GT en entrée → extraction séquences frames → entraînement TSPAN/TSPANv2 avec fenêtre glissante `[B, N, C, H, W]`. Pipeline temporel complet. |
| **Inférence Temporal SR** | Inférence N frames en fenêtre glissante avec TSPAN/TSPANv2 + réassemblage. |
| **Provider NVIDIA NIM** | `build.nvidia.com` comme nouveau provider IA — API compatible OpenAI, crédits gratuits (Llama, Mistral, Phi…). |
| **VOSR** | Troisième moteur d'inférence (CVPR 2026, SR diffusion). |

### Contribuer

Les pull requests sont les bienvenues. Pour les changements majeurs, ouvrir une issue d'abord.

1. Forker le dépôt
2. Créer une branche (`git checkout -b feature/ma-feature`)
3. Commiter les changements
4. Ouvrir une pull request

### Licence

[MIT](LICENSE) — libre d'utilisation, de modification et de distribution.
