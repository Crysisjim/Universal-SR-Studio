# Universal SR Studio

A graphical interface for training and managing super-resolution AI models with **NeoSR** and **traiNNer-Redux** engines.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
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

## Prerequisites

### Required
- **Python 3.11+** — [python.org](https://www.python.org)
- **NVIDIA GPU** with CUDA support (8 GB+ VRAM recommended)
- At least one training engine installed in `~/IA_Engine/`:
  - [traiNNer-Redux](https://github.com/the-database/traiNNer-redux) → `~/IA_Engine/traiNNer-redux/`
  - [NeoSR](https://github.com/muslll/neosr) → `~/IA_Engine/neosr/`

### Python packages
```bash
pip install -r requirements.txt
```

> **PyTorch** is NOT in `requirements.txt`. Install it separately from [pytorch.org](https://pytorch.org/get-started/locally/) matching your CUDA version **before** installing the training engines.

### Engine dependencies
Run the helper script to install engine-specific packages:
```bash
scripts\install_deps.bat
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Crysisjim/Universal_SR_Studio.git
cd Universal_SR_Studio

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install engine-specific dependencies (optional but recommended)
scripts\install_deps.bat

# 4. Launch
python main.py
```

---

## Expected folder structure

Universal SR Studio expects the training engines to be installed at:

```
~/IA_Engine/
├── traiNNer-redux/        (traiNNer-Redux engine)
│   └── .venv/             (engine virtual environment)
├── neosr/                 (NeoSR engine)
│   └── .venv/
├── datasets/
│   ├── train/HR/          (training high-resolution images)
│   └── val/
│       ├── GT/            (validation ground truth)
│       └── LQ/            (validation low-quality)
└── Option Custom/         (custom degradation presets)
```

---

## Project structure

```
Universal_SR_Studio/
├── main.py                    Entry point
├── requirements.txt
├── assets/                    Icons, sounds, UI themes
│   └── themes/                20+ color themes
├── scripts/
│   └── install_deps.bat       Engine dependency installer
└── src/
    ├── app.py                 Application root (tab layout)
    ├── core/                  Backend: config, training, OTF, tools
    └── ui/
        ├── components/        Shared UI widgets (tooltip, bars)
        └── tabs/              Tab panels (config, run, tools, settings…)
```

---

## Usage

### Launch
```bash
python main.py
```

### Tabs overview
| Tab | Description |
|-----|-------------|
| 😊 Assistant | Guided setup wizard for beginners |
| 📝 Configuration | TOML/YAML config editor with live OTF preview |
| 🚀 Training | Start/stop training, live curves, TensorBoard |
| 🔧 Tools | Benchmark, quick upscale, model export, dataset tools |
| 📋 Queue | Schedule multiple training sessions |
| ⚙️ Settings | Engine paths, language, appearance, API keys |
| 🌐 Distributed | Multi-machine training (experimental) |

### Benchmarks
Run standalone from the command line:
```bash
# Architecture benchmark (traiNNer-Redux)
python src/core/benchmark_runner.py --engine redux --type arch

# Feature benchmark (NeoSR)
python src/core/benchmark_runner.py --engine neosr --type feature

# List available tests
python src/core/benchmark_runner.py --list
```

---

## Configuration

On first launch, go to **⚙️ Settings** and set:
- Path to your NeoSR/traiNNer-Redux installation
- Training dataset paths
- (Optional) API keys for AI assistant features

Settings are saved in `user_settings.json` at the project root (excluded from git).

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
