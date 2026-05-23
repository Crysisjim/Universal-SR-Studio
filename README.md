# Universal SR Studio

A graphical interface for training and managing super-resolution AI models with **NeoSR** and **traiNNer-Redux** engines.

![Version](https://img.shields.io/badge/Version-2.4-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20required-brightgreen)

---

## Features

- **Configuration wizard** — visual TOML/YAML config editor for NeoSR and traiNNer-Redux
- **Training monitor** — real-time loss curves, PSNR/SSIM, TensorBoard integration, live GPU stats
- **OTF preview** — on-the-fly degradation pipeline preview (blur, noise, JPEG, compression, screentone, dithering, …)
- **Benchmark suite** — automated architecture and feature benchmarks with resume support
- **Model tools** — quick upscale, model export (safetensors), model packaging
- **Dataset tools** — tile splitter, validation rotation, LMDB converter
- **Distributed training** — multi-machine training coordination
- **Training queue** — schedule multiple training sessions back-to-back
- **20+ themes** — customizable UI themes

---

## Quick Start — Portable (recommended)

1. Download `Universal_SR_Studio_v2.4.0_portable.zip` from [Releases](https://github.com/Crysisjim/Universal-SR-Studio/releases)
2. Extract anywhere
3. Run `Universal_SR_Studio.exe`
4. On first launch, go to **⚙️ Settings** → the built-in installer handles everything else

> No Python installation required. The portable version is fully self-contained.

---

## Prerequisites

- **Windows 10/11**
- **NVIDIA GPU** with CUDA support (8 GB+ VRAM recommended)
- **Internet connection** for the first setup (engine download)

That's it. Universal SR Studio handles the rest automatically via the **⚙️ Settings** tab.

---

## Automatic Setup (via Settings tab)

Universal SR Studio includes a fully automated installer:

| Step | What it does |
|------|-------------|
| **GPU detection** | Detects your GPU and recommends the correct PyTorch + CUDA version |
| **Engine install** | Downloads and installs NeoSR and/or traiNNer-Redux from their official repositories |
| **Virtual environment** | Creates an isolated `.venv` for each engine |
| **PyTorch** | Installs the correct CUDA-compatible version automatically |
| **Dependencies** | Installs all engine-specific packages |

Just open the **⚙️ Settings** tab, choose which engine(s) to install, and click — the console window shows live progress.

---

## Expected folder structure

After installation, engines are placed at:

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

---

## Source installation (developers)

```bash
git clone https://github.com/Crysisjim/Universal-SR-Studio.git
cd Universal-SR-Studio
pip install -r requirements.txt
python main.py
```

Then use the **⚙️ Settings** tab to install the training engines.

---

## Project structure

```
Universal_SR_Studio/
├── main.py                    Entry point
├── requirements.txt
├── assets/                    Icons, sounds, UI themes
│   └── themes/                20+ color themes
├── scripts/
│   └── install_deps.bat       Engine dependency helper (optional)
└── src/
    ├── app.py                 Application root (tab layout)
    ├── core/                  Backend: config, training, OTF, tools
    └── ui/
        ├── components/        Shared UI widgets
        └── tabs/              Tab panels (config, run, tools, settings…)
```

---

## Tabs overview

| Tab | Description |
|-----|-------------|
| 😊 Assistant | Guided setup wizard for beginners |
| 📝 Configuration | TOML/YAML config editor with live OTF preview |
| 🚀 Training | Start/stop training, live curves, TensorBoard |
| 🔧 Tools | Benchmark, quick upscale, model export, dataset tools |
| 📋 Queue | Schedule multiple training sessions |
| ⚙️ Settings | Engine installer, paths, language, appearance, API keys |
| 🌐 Distributed | Multi-machine training (experimental) |

---

## Benchmarks (CLI)

```bash
# Architecture benchmark (traiNNer-Redux)
python src/core/benchmark_runner.py --engine redux --type arch

# Feature benchmark (NeoSR)
python src/core/benchmark_runner.py --engine neosr --type feature

# List available tests
python src/core/benchmark_runner.py --list
```

---

## Contributing

Pull requests welcome. For major changes, open an issue first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Open a pull request

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.
