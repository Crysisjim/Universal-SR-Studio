<div align="center">
  <img src="assets/icon.png" alt="Universal SR Studio" width="140"/>
  <h1>Universal SR Studio</h1>
  <p>Graphical interface for training super-resolution AI models<br>with <strong>NeoSR</strong> and <strong>traiNNer-Redux</strong> engines.</p>

  <a href="https://github.com/Crysisjim/Universal-SR-Studio/releases"><img src="https://img.shields.io/badge/Version-2.5.7-blue" alt="Version"/></a>
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

### What's new in v2.5.7

- **New "⚗ Post Processing" tab** — post-process an image or a folder *after* upscaling, independently of the upscale step. Ordered chain: **Temporal Fix → Undistort → Color correction → Resize → Sharpen (UnsharpMask)**. Each stage has an enable toggle and a ⚙ settings popup, with a Before/After preview, a live log and a progress bar.
- **Temporal Fix** — reduces temporal flicker between frames. Modes: *Classic* (motion-aware blend, no weights) or neural models *S1 / S2 / S3* (pifroggi) with on-demand weight download. Backends: PyTorch CUDA, OnnxRuntime (CUDA), TensorRT + OnnxRuntime, CPU.
- **Undistort** — removes temporal high-frequency artifacts (shimmering / jittering edges on SR output). Modes: *Classic* (temporal HF median, no weights) or neural *TMT* (xg416), with the same backend choices and on-demand weights.
- **LQ Dataset Generator — full redesign** — the degradation tool is now organized into 5 tabs (**🔧 Basic · 🎨 Colour · 📺 Video · ⚙ Advanced · 🎲 Pipeline**) with ~30 degradations, each with a checkbox + live slider: blur, noise, JPEG/H.264 codec, posterization, banding, chromatic aberration, disc blur, vignette, halo, saturation, quantize depth, chroma subsampling, aliasing, interlace (weave / flicker / field blend), CRT scanlines, VHS/analog, screentone, dithering, sinusoidal distortion, pixel shift, film grain, oversharpening, motion blur, salt & pepper, halation, auto-crop patches. The new **🎲 Pipeline** tab adds a **probabilistic pipeline** (wtp_dataset_destroyer-style): **N passes (1–5)** and a **per-degradation probability** so the same config yields varied results across a dataset.
- **AI Training Analysis button (🤖)** — in the **Training** tab console header: sends the training log + config to a chosen AI provider to analyze your run and suggest adjustments.
- **SparkLoss `Clamp (max_score)` field** — now exposed in the Configuration losses block; the losses block was widened and aerated (full labels, no abbreviations) for the GAN Phase 2 workflow.
- **Important fixes:**
  - **Mixed-resolution batches** — a folder mixing e.g. 1080p and 480p frames no longer crashes (`cannot reshape array of size …`) and no longer triggers a VRAM explosion + permanent slowdown (cuDNN autotune storm now disabled for inference; allocator defragmented on resolution change).
  - **Pascal GPUs (GTX 1080 / 1080 Ti, sm_61)** — Temporal Fix / Undistort now run inside the engine venv subprocess → fixes `no kernel image for device`.
  - **ONNX Temporal Fix** — fixed rank error (5D vs 4D input) on the OnnxRuntime backend.
  - **ORT performance** — the inference session is now cached once (no reload between frames).
  - GPU stats reliability (pynvml), zombie process on close, double-launch guard.

### What's new in v2.5.6

- **Shared venv** — single `runtimes/.venv` replaces dual neosr+traiNNer venvs; Python 3.12.9, numpy ≥2, traiNNer-redux `dev` branch; saves ~3 GB disk
- **New architectures** — custom-bundled and auto-injected: **ParagonSR** (nano/tiny/xs/s/m/l/xl/anime), **ParagonSR2** (photo/pro/realtime/stream/ultimate/ultimate_v2), **AetherNet** (NeoSR, mobile→extreme), FIGSR, GFISRv2, SMoSR, SPANpp; native in traiNNer-redux dev: srformer/v2, fdat, tfdat, spanf, spanplus, moesr/mosr/mosrv2, temporal_span_v2
- **GAN Phase 2 training** — `RealESRGANModel` + `UNetDiscriminatorSN` pipeline; Adaptive D checkbox (auto-pauses discriminator to prevent collapse); SparkLoss FD + PerceptualAnimeLoss in Redux config; GAN weight persistence fix
- **Custom degradations** — Custom 3+4 groups now correctly applied to training (were wired in UI but missing from sidecar writer); 3 new effects (disc_blur, vignette, quantize_depth); 2 new coupled clusters; severity preset button; 68 total keys (was 29)
- **RCAN 1× fix** — bypass spandrel bug (wrong n_feats for 1× models); construction from state_dict directly
- **Crash fixes** — Thumbs.db auto-clean before validation; TensorBoard path in frozen exe; UnicodeEncodeError on Windows stdout; zombie process on close; double-launch guard
- **VRAM estimation** — calibrated for all ParagonSR, ParagonSR2 and AetherNet variants against real `torch.cuda.memory_reserved()` measurements

### Features

**Training**
- **Configuration wizard** — visual TOML/YAML config editor for NeoSR and traiNNer-Redux; live VRAM estimation per architecture/patch size
- **GAN Phase 2** — `RealESRGANModel` + `UNetDiscriminatorSN`; Adaptive D (auto-pause to prevent collapse); SparkLoss + PerceptualAnimeLoss
- **Training monitor** — real-time loss curves, PSNR/SSIM, TensorBoard integration, live GPU stats (pynvml)
- **AI Training Analysis (🤖)** — send log + config to an AI provider for run analysis and tuning suggestions
- **Training queue** — schedule multiple sessions back-to-back
- **Distributed training** — multi-machine coordination

**Upscaling**
- **Quick Upscale** — persistent batch subprocess (model stays in VRAM), sequential numbering for video reassembly, skip duplicate frames (MAE), Color Fix ATWT, CUDA fallback for Pascal GPUs
- **30+ architectures** — ParagonSR (8 variants), ParagonSR2 (6 variants), AetherNet, FIGSR, GFISRv2, SMoSR, SPANpp, SRFormer/v2, FDAT, TFDAT, SpanF, SpanC, SpanPlus, MoESR/MoSR/MoSRv2, Temporal SPAN v2, and more

**Post Processing (⚗)**
- Ordered chain: **Temporal Fix → Undistort → Color correction → Resize → Sharpen → Deband → Line Darken → Line Thinning → Edge Cleanup**
- Neural Temporal Fix (S1/S2/S3, pifroggi) and Undistort (TMT, xg416) with PyTorch / OnnxRuntime / TensorRT backends
- CAS (AMD FidelityFX) and UnsharpMask sharpen modes
- Line-art modules: Deband (neo_f3kdb-style), Line Darken (Hysteria-style), Line Thinning (aWarpSharp2-style), Edge Cleanup (havsfunc-style)
- Each stage: enable toggle + ⚙ settings popup, Before/After preview, live log

**Dataset & Degradation**
- **LQ Dataset Generator** — ~30 degradations in 5 tabs (Basic · Color · Video · Advanced · Pipeline): blur, noise, JPEG/H.264, posterize, banding, chroma, aliasing, interlace, CRT, VHS, screentone, dithering, disc blur, vignette, pixel shift, film grain, motion blur, halation, auto-crop patches…
- **Probabilistic pipeline** (wtp_dataset_destroyer-style): N passes (1–5) + per-degradation probability for varied dataset results
- **OTF preview** — live on-the-fly degradation preview in the Configuration tab
- **Dataset tools** — tile splitter, validation rotation, LMDB converter

**AI Assistant**
- Multi-provider support: OpenRouter (Nemotron, Claude, GPT-4o, Gemini…), local models
- Send training logs for AI-assisted run analysis

**Other**
- **Benchmark suite** — 30+ automated architecture/feature benchmarks with resume (SpanF, SpanC, GFISRv2, SMoSR, ECO mode, personal model tests)
- **Model tools** — export (safetensors), packaging
- **20+ themes** — customizable UI
- **Bilingual UI** — French / English

### Quick Start — Portable (recommended)

1. Download `Universal_SR_Studio_v2.5.7_portable.zip` from [Releases](https://github.com/Crysisjim/Universal-SR-Studio/releases)
2. Extract anywhere
3. Run `Universal_SR_Studio.exe`
4. On first launch, choose your language (FR/EN), then go to **⚙️ Settings** → the built-in installer handles everything else

> No Python installation required. The portable version is fully self-contained.

### Prerequisites

- **Windows 10/11**
- **NVIDIA GPU** with CUDA support (8 GB+ VRAM recommended)
- **Internet connection** for the first setup (engine download)
- **~11 GB free disk space** for a full install (both engines + PyTorch CUDA + dependencies)

That's it. Universal SR Studio handles the rest automatically via the **⚙️ Settings** tab.

### Automatic Setup (via Settings tab)

| Step | What it does |
|------|-------------|
| **GPU detection** | Detects your GPU and recommends the correct PyTorch + CUDA version |
| **Engine install** | Downloads and installs NeoSR and/or traiNNer-Redux from their official repositories |
| **Virtual environment** | Creates a single shared `runtimes/.venv` (Python 3.12.9 + PyTorch CUDA) for all engines |
| **PyTorch** | Installs the correct CUDA-compatible version automatically |
| **Dependencies** | Installs all engine-specific packages |

Just open the **⚙️ Settings** tab, choose which engine(s) to install, and click — the console window shows live progress.

### Expected folder structure

```
~/IA_Engine/
├── runtimes/
│   ├── traiNNer-redux/        (installed via Settings)
│   ├── neosr/                 (installed via Settings)
│   └── .venv/                 (shared Python 3.12.9 + PyTorch CUDA environment)
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
| 🔧 Tools | Benchmark, quick upscale, model export, dataset tools, ⚗ Post Processing (Temporal Fix / Undistort / color / resize / sharpen / deband / line-art modules) |
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

### Roadmap — v2.6.0

| Feature | Description |
|---------|-------------|
| **Temporal SR training** | Enter a video as GT reference, extract frame sequences, train TSPAN/TSPANv2 with sliding window input `[B, N, C, H, W]`. Full temporal consistency pipeline. |
| **Temporal SR inference** | Sliding window N-frame inference with TSPAN/TSPANv2 and frame reassembly. |
| **AA post-processing** | NNEDI3/EEDI3-ONNX antialiasing module in the Post Processing chain (insaneAA-quality). |

> ⏸ **VOSR / OSEDiff** (diffusion-based SR, CVPR 2026) — integration paused indefinitely. Interest is limited due to very high VRAM requirements, making them impractical for most consumer GPUs.

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

### Nouveautés v2.5.7

- **Nouvel onglet "⚗ Post Processing"** — post-traiter une image ou un dossier *après* l'upscale, indépendamment. Chaîne ordonnée : **Temporal Fix → Undistort → Correction couleur → Redimensionnement → Netteté (UnsharpMask)**. Chaque étape a une case enable + un popup ⚙ de réglages, avec preview Avant/Après, log live et barre de progression.
- **Temporal Fix** — réduit le scintillement temporel entre frames. Modes : *Classic* (blend motion-aware, sans poids) ou modèles neuraux *S1 / S2 / S3* (pifroggi) avec téléchargement des poids à la demande. Backends : PyTorch CUDA, OnnxRuntime (CUDA), TensorRT + OnnxRuntime, CPU.
- **Undistort** — supprime les artefacts haute-fréquence temporels (bords qui scintillent/tremblent sur la sortie SR). Modes : *Classic* (médiane HF temporelle, sans poids) ou neural *TMT* (xg416), mêmes choix de backend + poids à la demande.
- **Générateur de dataset LQ — refonte complète** — l'outil de dégradation est maintenant organisé en 5 onglets (**🔧 Basique · 🎨 Couleur · 📺 Vidéo · ⚙ Avancé · 🎲 Pipeline**) avec ~30 dégradations, chacune avec case + slider live : flou, bruit, codec JPEG/H.264, postérisation, banding, aberration chromatique, flou disque, vignette, halo, saturation, quantize depth, sous-échantillonnage chroma, aliasing, entrelacement (weave / flicker / field blend), scanlines CRT, VHS/analog, screentone, dithering, distorsion sinusoïdale, pixel shift, grain de film, oversharpening, motion blur, sel & poivre, halation, auto-crop patches. Le nouvel onglet **🎲 Pipeline** ajoute un **pipeline probabiliste** (style wtp_dataset_destroyer) : **N passes (1–5)** et une **probabilité par dégradation** → une même config produit des résultats variés sur tout un dataset.
- **Bouton Analyse IA (🤖)** — dans l'en-tête de la console de l'onglet **Entraînement** : envoie le log + la config à un provider IA pour analyser l'entraînement et suggérer des ajustements.
- **Champ `Clamp (max_score)` SparkLoss** — maintenant exposé dans le bloc losses de Configuration ; bloc losses élargi et aéré (labels complets, plus d'abréviations) pour le workflow GAN Phase 2.
- **Corrections importantes :**
  - **Batchs multi-résolution** — un dossier mélangeant par ex. du 1080p et du 480p ne plante plus (`cannot reshape array of size …`) et ne déclenche plus d'explosion VRAM + ralentissement permanent (tempête d'autotune cuDNN désactivée en inférence ; allocateur défragmenté au changement de résolution).
  - **GPU Pascal (GTX 1080 / 1080 Ti, sm_61)** — Temporal Fix / Undistort tournent désormais dans le subprocess venv du moteur → corrige `no kernel image for device`.
  - **ONNX Temporal Fix** — erreur de rang corrigée (entrée 5D vs 4D) sur le backend OnnxRuntime.
  - **Performance ORT** — la session d'inférence est mise en cache une seule fois (plus de rechargement entre frames).
  - Fiabilité stats GPU (pynvml), processus zombie à la fermeture, garde anti-double-lancement.

### Nouveautés v2.5.6

- **Venv partagé** — un seul `runtimes/.venv` remplace les deux venvs neosr+traiNNer ; Python 3.12.9, numpy ≥2, branche `dev` traiNNer-redux ; économise ~3 Go
- **Nouvelles architectures** — custom bundlées et auto-injectées : **ParagonSR** (nano/tiny/xs/s/m/l/xl/anime), **ParagonSR2** (photo/pro/realtime/stream/ultimate/ultimate_v2), **AetherNet** (NeoSR, mobile→extreme), FIGSR, GFISRv2, SMoSR, SPANpp ; natives dans traiNNer-redux dev : srformer/v2, fdat, tfdat, spanf, spanplus, moesr/mosr/mosrv2, temporal_span_v2
- **Entraînement GAN Phase 2** — pipeline `RealESRGANModel` + `UNetDiscriminatorSN` ; checkbox Adaptive D (met le discriminateur en pause pour éviter l'effondrement) ; SparkLoss FD + PerceptualAnimeLoss dans la config Redux ; fix persistance du poids GAN
- **Dégradations custom** — groupes Custom 3+4 maintenant correctement appliqués à l'entraînement (câblés dans l'UI mais absents du sidecar) ; 3 nouveaux effets (disc_blur, vignette, quantize_depth) ; 2 clusters couplés ; bouton preset de sévérité ; 68 clés au total (était 29)
- **Fix RCAN 1×** — contournement du bug spandrel (n_feats incorrect pour les modèles 1×) ; construction directe depuis le state_dict
- **Fixes crashs** — nettoyage automatique Thumbs.db avant validation ; chemin TensorBoard dans l'exe portable ; UnicodeEncodeError sur stdout Windows ; processus zombie à la fermeture ; garde anti-double-lancement
- **Estimation VRAM** — calibrée pour tous les variants ParagonSR, ParagonSR2 et AetherNet sur des mesures `torch.cuda.memory_reserved()` réelles

### Fonctionnalités

**Entraînement**
- **Assistant de configuration** — éditeur visuel TOML/YAML pour NeoSR et traiNNer-Redux ; estimation VRAM live selon l'architecture et la patch size
- **GAN Phase 2** — `RealESRGANModel` + `UNetDiscriminatorSN` ; Adaptive D (pause auto du discriminateur) ; SparkLoss + PerceptualAnimeLoss
- **Moniteur d'entraînement** — courbes de perte en temps réel, PSNR/SSIM, TensorBoard, stats GPU live (pynvml)
- **Analyse IA (🤖)** — envoyer le log + la config à un provider IA pour analyser l'entraînement et suggérer des réglages
- **File d'entraînements** — planifier plusieurs sessions à la suite
- **Entraînement distribué** — coordination multi-machines

**Upscale**
- **Quick Upscale** — subprocess batch persistant (modèle reste en VRAM), numérotation séquentielle pour réassemblage vidéo, skip frames dupliquées (MAE), Color Fix ATWT, fallback CUDA pour GPU Pascal
- **30+ architectures** — ParagonSR (8 variants), ParagonSR2 (6 variants), AetherNet, FIGSR, GFISRv2, SMoSR, SPANpp, SRFormer/v2, FDAT, TFDAT, SpanF, SpanC, SpanPlus, MoESR/MoSR/MoSRv2, Temporal SPAN v2, et plus

**Post Processing (⚗)**
- Chaîne ordonnée : **Temporal Fix → Undistort → Correction couleur → Redimensionnement → Netteté → Deband → Renforcement lignes → Amincissement lignes → Edge Cleanup**
- Temporal Fix neural (S1/S2/S3, pifroggi) et Undistort (TMT, xg416) avec backends PyTorch / OnnxRuntime / TensorRT
- Modes sharpen CAS (AMD FidelityFX) et UnsharpMask
- Modules line-art : Deband (style neo_f3kdb), Renforcement lignes (style Hysteria), Amincissement lignes (style aWarpSharp2), Edge Cleanup (style havsfunc)
- Chaque étape : toggle enable + popup ⚙ de réglages, preview Avant/Après, log live

**Dataset & Dégradation**
- **Générateur LQ** — ~30 dégradations en 5 onglets (Basique · Couleur · Vidéo · Avancé · Pipeline) : flou, bruit, JPEG/H.264, postérisation, banding, chroma, aliasing, entrelacement, CRT, VHS, screentone, dithering, flou disque, vignette, pixel shift, grain, motion blur, halation, auto-crop…
- **Pipeline probabiliste** (style wtp_dataset_destroyer) : N passes (1–5) + probabilité par dégradation → résultats variés sur tout un dataset
- **Aperçu OTF** — prévisualisation live dans l'onglet Configuration
- **Outils dataset** — découpeur de tuiles, rotation de validation, convertisseur LMDB

**Assistant IA**
- Multi-providers : OpenRouter (Nemotron, Claude, GPT-4o, Gemini…), modèles locaux
- Analyse des logs d'entraînement assistée par IA

**Autre**
- **Suite de benchmarks** — 30+ benchmarks architectures/features automatisés avec reprise (SpanF, SpanC, GFISRv2, SMoSR, ECO mode, tests modèle perso)
- **Outils modèles** — export (safetensors), packaging
- **20+ thèmes** — UI personnalisable
- **Interface bilingue** — Français / Anglais

### Démarrage rapide — Portable (recommandé)

1. Télécharger `Universal_SR_Studio_v2.5.7_portable.zip` depuis les [Releases](https://github.com/Crysisjim/Universal-SR-Studio/releases)
2. Extraire n'importe où
3. Lancer `Universal_SR_Studio.exe`
4. Au premier lancement, choisir la langue (FR/EN), puis aller dans **⚙️ Paramètres** → l'installeur intégré gère le reste

> Aucune installation Python requise. La version portable est entièrement autonome.

### Prérequis

- **Windows 10/11**
- **GPU NVIDIA** avec support CUDA (8 Go+ VRAM recommandé)
- **Connexion internet** pour le premier setup (téléchargement des moteurs)
- **~11 Go d'espace disque libre** pour une installation complète (les deux moteurs + PyTorch CUDA + dépendances)

C'est tout. Universal SR Studio gère le reste automatiquement via l'onglet **⚙️ Paramètres**.

### Installation automatique (via l'onglet Paramètres)

| Étape | Action |
|-------|--------|
| **Détection GPU** | Détecte le GPU et recommande la bonne version PyTorch + CUDA |
| **Installation moteur** | Télécharge et installe NeoSR et/ou traiNNer-Redux depuis leurs dépôts officiels |
| **Environnement virtuel** | Crée un seul `runtimes/.venv` partagé (Python 3.12.9 + PyTorch CUDA) pour tous les moteurs |
| **PyTorch** | Installe la version compatible CUDA automatiquement |
| **Dépendances** | Installe tous les packages spécifiques au moteur |

Ouvrir l'onglet **⚙️ Paramètres**, choisir le(s) moteur(s) à installer, et cliquer — la console affiche la progression en direct.

### Structure de dossiers attendue

```
~/IA_Engine/
├── runtimes/
│   ├── traiNNer-redux/        (installé via Paramètres)
│   ├── neosr/                 (installé via Paramètres)
│   └── .venv/                 (environnement Python 3.12.9 + PyTorch CUDA partagé)
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
| 🔧 Outils | Benchmark, upscale rapide, export modèle, outils dataset, ⚗ Post Processing (Temporal Fix / Undistort / couleur / resize / netteté / deband / modules line-art) |
| 📋 File d'attente | Planifier plusieurs sessions d'entraînement |
| ⚙️ Paramètres | Installeur moteurs, chemins, langue, apparence, clés API |
| 🌐 Distribué | Entraînement multi-machines (expérimental) |
| 📖 Wiki | Ouvre la documentation wiki GitHub dans le navigateur |

### Roadmap — v2.6.0

| Feature | Description |
|---------|-------------|
| **Entraînement Temporal SR** | Vidéo GT en entrée → extraction séquences frames → entraînement TSPAN/TSPANv2 avec fenêtre glissante `[B, N, C, H, W]`. Pipeline temporel complet. |
| **Inférence Temporal SR** | Inférence N frames en fenêtre glissante avec TSPAN/TSPANv2 + réassemblage. |
| **AA post-processing** | Module antialiasing NNEDI3/EEDI3-ONNX dans la chaîne Post Processing (qualité insaneAA). |

> ⏸ **VOSR / OSEDiff** (SR par diffusion, CVPR 2026) — intégration en pause indéfinie. Intérêt limité en raison d'une consommation VRAM très élevée, peu pratique sur la majorité des GPU grand public.

### Contribuer

Les pull requests sont les bienvenues. Pour les changements majeurs, ouvrir une issue d'abord.

1. Forker le dépôt
2. Créer une branche (`git checkout -b feature/ma-feature`)
3. Commiter les changements
4. Ouvrir une pull request

### Licence

[MIT](LICENSE) — libre d'utilisation, de modification et de distribution.
