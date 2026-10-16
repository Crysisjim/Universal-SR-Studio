# Third-party notices — Universal SR Studio

Universal SR Studio itself is MIT-licensed (see [LICENSE](LICENSE)). It bundles or
downloads third-party code and models that keep their own licenses, listed below.
Licenses were checked against each upstream repository (October 2026).

## Engines (installed by the app from their official repositories, not bundled)

| Project | License |
|---|---|
| [traiNNer-redux](https://github.com/the-database/traiNNer-redux) | Apache-2.0 |
| [neosr](https://github.com/neosr-project/neosr) | see upstream |
| [spandrel](https://github.com/chaiNNer-org/spandrel) | MIT |
| PyTorch and the Python packages installed into the engine environment | their own licenses |

## Architectures bundled in `src/core/custom_archs`, `custom_engine`, `custom_neosr_archs`, `custom_utils`

| File | Origin | License |
|---|---|---|
| `tfdatv2_arch.py` | TFDAT v2 — Kim2091 ([TFDAT](https://github.com/Kim2091/TFDAT), branch `tfdat-v2` of [traiNNer-redux-1](https://github.com/Kim2091/traiNNer-redux-1)), based on [FDAT](https://github.com/stinkybread/fdat) (MIT) | Apache-2.0 — `licenses/TFDAT-Apache-2.0.txt` |
| `motsr_arch.py` | [MoTSR](https://github.com/XYETHER/MoTSR) — xyether, built on [MoSRv2](https://github.com/umzi2/MoSRV2) (umzi2) | MIT — `licenses/MoTSR-MIT.txt`, `MoTSR-NOTICE.txt`, `MoSRv2-MIT.txt` |
| `mambair_arch.py`, `mambairv2_arch.py` | [MambaIR / MambaIRv2](https://github.com/csguoh/MambaIR) — csguoh | Apache-2.0 |
| `paragonsr_arch.py`, `paragonsr2_arch.py`, `resampler.py` | [ParagonSR](https://github.com/Phhofm/ParagonSR) — Phhofm | MIT |
| `aether_arch.py`, `aether_core.py` | [AetherNet](https://github.com/Phhofm/aethernet) — Phhofm | MIT |
| `gfisrv2_arch.py` | [GFISRv2](https://github.com/umzi2/gfisrv2) — umzi2 | MIT |
| `spanpp_arch.py` | [SPANpp](https://github.com/umzi2/spanpp) — umzi2 | MIT |
| `smosr_arch.py` | [SMoSR](https://github.com/umzi2/SMoSR) — umzi2 | MIT |
| `spark_loss.py` | [SparK_Perceptual](https://github.com/umzi2/SparK_Perceptual) — umzi2 | MIT |
| DySample upsampler (in GFISRv2 / SMoSR) | [dysample](https://github.com/tiny-smart/dysample) | MIT |
| `inceptionnext_arch.py` | [InceptionNeXt](https://github.com/sail-sg/inceptionnext) — Sea AI Lab | Apache-2.0 |
| `spectral_disc_arch.py`, `custom_utils/spectrum.py` | [DualFormer](https://github.com/Luciennnnnnn/DualFormer) spectral discriminator | MIT (stated in the upstream README) |
| `projected_disc_arch.py` | [Projected GAN](https://github.com/autonomousvision/projected-gan) | MIT |
| `figsr_arch.py` | [FIGSR](https://github.com/enhancr/figsr) — enhancr | MIT — `licenses/FIGSR-MIT.txt` |
| `nextsrgan_disc_arch.py` | PyTorch port of [NeXtSRGAN](https://github.com/PomKlementieff/NeXtSRGAN)'s discriminator (paper: Klementieff, Kim & Han, The Visual Computer 2024) | upstream repository has no license file and is unmaintained; credited here |
| `wiresr_arch.py` | WireSR — CF2lter ("1x_Wired"; source from the author's Google Drive) | Included **with the author's permission** (granted 2026-10-26; no license file published upstream) — `licenses/WireSR-PERMISSION.txt` (the 1x_Wired weights stay CC BY-NC-SA 4.0, not bundled) |
| `fdat_rd_arch.py` | [FDAT-RD](https://github.com/stinkybread/FDAT-RD) — stinkybread, based on FDAT (MIT) | **supposed MIT** (the repository has no LICENSE file yet) — `licenses/FDAT-RD-MIT.txt` |

## Other bundled code

| File | Origin | License |
|---|---|---|
| `src/core/aa_nnedi3_native.py` | port of [znedi3](https://github.com/sekrit-twc/znedi3) (sekrit-twc) | **GPL-2.0** (derived work) |
| `src/core/models/temporalfix_arch.py` | [vs_temporalfix](https://github.com/pifroggi/vs_temporalfix) (pifroggi), RIFE flow ([ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE), MIT) | Apache-2.0 |
| `src/core/models/unet3d_tmt.py`, `tmt_inference.py` | [TMT](https://github.com/xg416/TMT) via [vs_undistort](https://github.com/pifroggi/vs_undistort) | upstream has no license file |
| `src/core/guided_color_fix.py` (PACC V4 network) | Bendel (labx), [2x_aniCC](https://huggingface.co/labx/2x_aniCC) chaiNNer node | CC BY-NC-SA 4.0 (non-commercial) |

## Models downloaded on first use (never bundled)

| Model | Source | License |
|---|---|---|
| aniCC (Guided Color Fix) | [labx/2x_aniCC](https://huggingface.co/labx/2x_aniCC) — Bendel | CC BY-NC-SA 4.0 |
| `nnedi3_weights.bin` | [vapoursynth-nnedi3](https://github.com/dubhater/vapoursynth-nnedi3) | GPL-2.0 |
| Temporal Fix / Undistort weights | pifroggi releases | see upstream |

Models you load yourself (e.g. Bendel's 2x_bndl_animefilm, 1x_Wired, OpenModelDB
models) are not distributed with the app and keep their authors' licenses.
