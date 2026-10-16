# mamba_ssm for Windows (MambaIR / MambaIRv2 CUDA kernel)

MambaIR and MambaIRv2 call `selective_scan_fn`, the compiled CUDA kernel of
[state-spaces/mamba](https://github.com/state-spaces/mamba). Upstream only builds on
Linux. Without the kernel, both archs fall back to a pure-PyTorch scan that is roughly
100× slower and needs far more VRAM (the arch prints a warning in the log when that
happens).

The portable build ships a prebuilt wheel in `vendor_wheels/`. The engine installer and
"reinstall dependencies" install it into the engine venv when that venv runs
torch 2.7.0+cu126 on CPython 3.12. This folder rebuilds that wheel.

## Requirements

- Visual Studio 2022 with the C++ workload (`cl.exe`).
- CUDA Toolkit 12.6 (`CUDA_HOME` in `build_mamba.bat`).
- Python headers in the engine venv. The engine's portable Python has none, and torch's
  build reads them from `sys.base_prefix`, which is `runtimes\.venv\Scripts` here. Copy
  `tools\include\` → `.venv\Scripts\Include\` and `tools\libs\` → `.venv\Scripts\libs\`
  from the nuget `python` package of the same version (3.12.9). Extract with Python's
  `zipfile`: the package stores backslash paths, and `unzip` flattens the `cpython\` and
  `internal\` subfolders.
- `pip install ninja` in the venv (parallel build).

## Build

```bat
git clone --depth 1 --branch v2.2.2 https://github.com/state-spaces/mamba.git mamba_src
python patch_mamba_windows.py mamba_src
build_mamba.bat
```

`build_mamba.bat clean` empties `mamba_src\build` first. Do that after any `setup.py`
change: pip reuses the stale objects otherwise. The wheel lands in `wheels\`. Copy it to
`vendor_wheels\` at the repo root before building the exe.

## What the patch changes (see `patch_mamba_windows.py`)

1. `setup.py`: `-D_USE_MATH_DEFINES` (MSVC has no `M_LOG2E`) and real GPU targets.
   Upstream hard-codes its `-gencode` list, ignores `TORCH_CUDA_ARCH_LIST` and has no
   sm_61. Targets: sm_61 (GTX 10), 75 (RTX 20), 86 (RTX 30), 89 (RTX 40), 90, and
   compute_89 PTX so newer GPUs can JIT it. Native RTX 50 (sm_120) needs CUDA 12.8+.
2. Kernel headers: the `#ifndef USE_ROCM` blocks inside the `BOOL_SWITCH` lambdas are
   rejected by the MSVC preprocessor. Only the CUDA branch is kept.
3. Launchers rewritten without lambdas. MSVC, as the nvcc host compiler, fails on the
   constexpr template arguments built in the nested lambdas (C2975 `kNRows_` /
   `kIsEvenLen_`, with or without `/Zc:preprocessor`). Variable B/C is required and
   checked at run time; even/odd length, `z` and `delta_softplus` stay dispatched.
4. `mamba_ssm/__init__.py`: the language-model modules (triton, transformers) are
   imported only if available.

## History

First built in August 2026 (session 135). Lost when the experimental engine folder was
deleted on 2026-09-03. Rebuilt on 2026-10-31 with these scripts, kept in git this time.
