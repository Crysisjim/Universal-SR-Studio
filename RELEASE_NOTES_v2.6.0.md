# Universal SR Studio v2.6.0

**Install** — Windows 10/11 + NVIDIA GPU. Download `Universal_SR_Studio_v2.6.0_portable.zip`, extract it anywhere and run `Universal_SR_Studio_QT\Universal_SR_Studio_QT.exe`. On first launch open **⚙️ Settings** to install the training engine(s): the app installs its Python, PyTorch CUDA and every dependency itself.

**Installation** — Windows 10/11 + GPU NVIDIA. Télécharger `Universal_SR_Studio_v2.6.0_portable.zip`, extraire n'importe où et lancer `Universal_SR_Studio_QT\Universal_SR_Studio_QT.exe`. Au premier lancement, ouvrir **⚙️ Paramètres** pour installer le(s) moteur(s) : l'app installe elle-même son Python, PyTorch CUDA et toutes les dépendances.

Licenses / licences : app MIT ; bundled architectures and downloaded models keep their own licenses — see `THIRD_PARTY_NOTICES.md`.

---

<a name="v260--english"></a>
## v2.6.0 — English

*Changes since v2.5.7 (June 2026).* v2.6.0 is the largest update so far: the whole
interface was rewritten, ~45 architectures were added or made usable everywhere, video
super-resolution became a complete pipeline, and the app was validated against
thousands of generated training configurations and real training runs.

### 1. A new interface (Qt / PySide6)
- **Complete rewrite** of the CustomTkinter app in Qt: 7 tabs, 13 tool pages, 7
  Configuration sub-pages, 5 Settings sub-tabs and 5 Training windows. Every feature of
  v2.5.7 was ported, then audited function by function against the old app (23 gaps
  fixed).
- **Full English interface** (~1,200 strings translated), language switch in Settings,
  FR/EN tooltips on every parameter — including per-architecture parameters, which
  showed almost no tooltips before (84 of them had their own text that was never
  displayed).
- Dark / light / auto theme, **interface scale 90–150 %**, Windows taskbar progress
  (green running, yellow stopped, red crashed), sounds and Windows notifications.
- **Training tab progress bar**: thicker, with the percentage and the iteration
  (`21% · 52 500 / 250 000`) written in the middle. For a two-phase run (delayed or
  automatic GAN start), a badge shows the current phase — **PSNR** in green with the GAN
  auto start counter (`GAN auto 2/3 (≥ 85 000)`), **GAN** in mauve with its start
  iteration — and the bar takes the phase colour.
- **Loss badges**: each loss that relies on a pretrained network shows what it was
  trained on — **Anime** (Percep. Anime, LineDistiller) or **Photo** (VGG Perceptual,
  FP16, DISTS, Contextual, FDL, SparK). Losses without a badge use no learned network.
- **Engine folder renamed** `~/IA_Engine` → `~/USRS`. Existing data is never moved or
  deleted.
- Reinstalling an engine is now **blocked if its folder holds your datasets, experiments
  or models**. Before, it deleted the folder without checking.

### 2. Architectures
**New, trainable and usable everywhere** (Quick Upscale, Model Converter, Model Info):
- **TFDAT v2** (Kim2091). Runs Bendel's **2x_bndl_animefilm_v3** (9-frame clips, output
  identical to the official ONNX); earlier bndl models (v1 / v1.5 DAT2 & FDAT,
  tfdat_v2) work as well.
- **MoTSR** (xyether) — recurrent 2x video SR, scene-cut aware, joint / partial /
  bootstrap training stages.
- **MambaIR / MambaIRv2** (csguoh, ECCV 2024 / CVPR 2025). Their CUDA kernel
  (`mamba_ssm`, Linux-only upstream) is built for Windows and **shipped with the app**: the
  engine install installs it (GTX 10 → RTX 40, torch 2.7.0+cu126). GTX 1080 Ti, x2,
  batch 2, LQ 64: ~1.9 it/s at 3.6 GB (MambaIR) and ~1.8 it/s at 4.4 GB (MambaIRv2).
  Without the kernel they fall back to a pure-PyTorch scan (~100× slower, far more VRAM)
  and the log says so.
- **HEART** and **NERVE** (Phhofm).
- **DocRes** (document restoration, 1x) and its discriminator.
- Now exposed for training: **TFDAT**, **NinaSR**, **HAT-iLN / DAT-iLN**.
- **WireSR** (CF2lter, "1x_Wired") — anti-aliasing / thin-line restoration.
- **FDAT-RD** (stinkybread) — FDAT evolution for anime (rectangular 10×30 windows + learned
  token dictionary, static TensorRT export): small / medium / large presets, gradient
  checkpointing option (training VRAM ÷ 7). Fixed a dark line on the bottom/right edge
  of upscales whose size is a multiple of 60 (720×480, 1920×1080): the arch now always
  keeps a small mirrored margin, as in training. Weights and output size are unchanged.
  Models trained with an LQ patch that is not a multiple of 60 (2x unshuffle, else 30)
  keep this line in other tools (chaiNNer, ONNX): the training check now warns and
  suggests 60 or 120, and the tooltip explains it.

**Quick Upscale coverage**, in normal and persistent-subprocess modes:
- **traiNNer-redux**: ArtCNN, LKFMixer, RTMoSR, MoSRv2, Sebica, LMLT, MAN, EIMN, ELAN,
  GateRV3, HiT-SIR/SNG, CascadedGaze, EMT, ESCRealM, FlexNet, SRFormer/v2.
- **NeoSR**: PlainUSR, ASID, GRFormer, DCT, HiT-SRF, HMA, SRFormer-medium, HASN, MSDAN.
  AetherNet (7 presets) was 100 % broken before and works now.
- Every one was tested by rebuilding from real weights with a strict load.
- **Full inference sweep** (October 2026): for every architecture the app can train,
  the network was built by its engine from the config the app generates, at 2x and 4x, then
  run through Quick Upscale in single-image mode and in folder mode with the persistent
  subprocess. Result: **185 / 185** models run at 2x and at 4x (119 / 185 before these fixes), and the persistent folder mode works for all 179 image archs (video archs use the sequence pipeline). Only ESCRealM was not tested: it needs `triton`. The full table is in the README ("Supported architectures"). The fixes it led to:
  - **Scale**: the scale detected from the weights used to be sent to the runners with a
    default of 4, which overrode their own (exact) detection. Detection itself was
    rewritten: multi-stage pixel-shuffle upsamplers multiply (4x HAT / SwinIR / DAT read
    as 2x before), and MoSRv2 / RTMoSR `unshuffle_mod` models, DocRes (1x) and SpanPP are
    read correctly. Wrong detections on 370 test models: 74 → 2 (ParagonSR v1 and MSDAN,
    whose scale is not stored in the weights: the scale chosen in the interface is used).
  - **Folder mode with "persistent subprocess"** crashed on the first image for every
    traiNNer-redux subprocess architecture (ArtCNN, ELAN, MAN, ParagonSR, FDAT-RD,
    HEART…) — a return-value mismatch. Fixed, with a per-image fallback when the worker
    fails on one image.
  - The Quick Upscale tab always started the traiNNer-redux worker: NeoSR-only archs (ESC,
    GRFormer, CATANet, DRCT, LMLT…) failed to start it and silently ran one subprocess per
    image. The tab now picks the NeoSR worker for them.
  - A `print()` inside an architecture constructor (DocRes) broke the worker protocol;
    the workers' stderr was never read either, which could stall a long batch. Both fixed.
  - Rebuilt from the weights instead of the defaults: **SPANPlus** S / ST / STS (the
    persistent worker even loaded them non-strictly, giving a wrong image without error),
    **DRCT-L / XL**, **LMLT tiny / large**, **MetaFlexNet**, **DAT-iLN** (all presets), and
    the NeoSR versions of **CATANet, KRGN, MAN, MoSRv2** (they were sent to the
    traiNNer-redux runner, which does not have them), and **SCUNet-aaf6aa** (spandrel read it
    as plain SCUNet).
  - SRFormer / DRCT models saved with another attention-mask size load again; GRFormer
    no longer fails on recent PyTorch (engine patch); ParagonSR v1 / v2 use the right
    scale; TSCUNet and MambaIR are no longer mistaken for other archs.

**GAN discriminators**:
- New discriminators: **Spectral** (DualFormer, ICCV 2023), **NeXtSRGAN** and
  **Projected GAN** (frozen EfficientNet backbone).
- **Second discriminator** (any pair, weighted): each discriminator gets its own
  Adaptive-D pause. A visible warning flags redundant pairs.
- **Delayed GAN start** (traiNNer-redux): the generator first trains on its other losses
  only, then the GAN loss and the discriminator(s) start. Recommended when training from
  scratch. Two modes:
  - at a fixed iteration ("GAN start (iter)");
  - **Auto**: when the validation PSNR stops improving (patience, minimum gain, minimum
    iteration; "GAN start" then acts as the latest start).
  The "Restart LR schedule" option sets the LR back to its initial value at the GAN start,
  and the milestones count from there, like a separate second training run. The GAN
  start state is saved, so a resumed training keeps it.
- The discriminator's VRAM is now counted in the VRAM estimate. It never was before,
  not even for the original UNet/PatchGAN.

### 3. Video / temporal super-resolution
- **Training**:
  - New "sequence" dataset mode (`PairedVideoDataset`, `clip_size`), locked
    automatically for video archs.
  - Batch augmentations that would corrupt clips are disabled in that mode.
  - Train/Val paths are remembered separately for video and image modes.
  - A clear error is shown when the validation clips are shorter than `clip_size`.
- **Quick Upscale** takes a frame folder **or a video file** for video models
  (TSCUNet, TemporalSPAN v1/v2, TFDAT v1/v2, MoTSR), and also for **any image model**
  (frames processed one by one). Optional reassembly back to video.
- **Encoding options** for reassembly:
  - container MP4/MKV/WebM;
  - x264/x265 or NVENC H.264/HEVC/AV1, each tested on your own GPU;
  - CRF or bitrate, speed presets.
- **True 16-bit** sequence pipeline (pyvips load, real PNG16 output).
- **LQ Generator video mode**: extracts frames, degrades them and builds a contiguous
  validation block (end / start / random).
- **ONNX export of video models**:
  - a `1x(T·3)xHxW` graph (vs-mlrt / VapourKit layout);
  - MoTSR's seed + recurrent graphs.
- Video models now **run in the portable build**. Before, they failed there with
  "PyTorch not installed".

### 4. Training reliability
- **Validated on real training runs**, which revealed several serious bugs:
  - The wrong Python was used, so training crashed at start.
  - The PSNR graph stayed empty.
  - **Soft Stop** didn't save. It now stops cleanly in ~10 s with a checkpoint.
  - The dashboard kept the previous run's numbers.
  - **Validation tiling** was always ignored.
- **Automated test harness**: ~2,500 generated configurations (archs × engines ×
  losses × optimizers × GAN). Real bugs fixed:
  - NeoSR list parameters were parsed as floats.
  - batch = 1 with augmentations failed.
  - Preset archs (hat_s/m/l, swinir_medium…) received duplicate parameters.
  - AetherNet was missing `scale`, and was never injected into NeoSR.
  - Some discriminator names were wrong.
  - Archs and optimizers absent from the selected engine were still offered.
  - Fixed-scale archs (CascadedGaze, RealCUGAN, MSDAN) and minimum patch sizes were
    not enforced.
- **MambaIR**: the config generator never set `upscale`, so models silently trained
  at 2x instead of 1x.
- Also fixed: tf32 / `fast_matmul`, `ReduceLROnPlateau`, Schedule-Free on NeoSR,
  `torch.compile` (greyed out on Windows), consistency loss `use_blur`, AESOP and ECO
  safeguards.
- **Self-healing engine patches**: fixes to upstream engine files (25+) are re-applied
  automatically after an engine update or re-clone.
- **NeoSR broken after a Quick Upscale (fixed)**: before each upscale, the app copied its
  traiNNer-redux custom archs (FDAT-RD, MoTSR, TFDAT v2, MambaIR, WireSR, the new
  discriminators…) into the NeoSR engine too. Several import traiNNer, and NeoSR loads
  every arch file of its folder: no NeoSR arch could be built any more, in training or
  in inference (0/49 in the test, 49/49 after the fix). They now go to traiNNer-redux
  only, and the copies already in NeoSR are disabled (renamed `*.usrs_disabled`,
  nothing deleted).
- **VRAM / speed estimator** calibrated on real measurements: MambaIR/v2, MoTSR,
  discriminators, and real training runs for HEART, NERVE, WireSR and FDAT-RD
  small/medium/large. HEART, NERVE and WireSR used isolated-benchmark ratios before,
  which underestimated their real training speed 8× to 30×. Gradient checkpointing VRAM
  saving corrected (×0.66 in a real run, not ×0.25). FDAT-RD VRAM no longer grows like
  a power of the patch size: most of it is fixed (weights, loss networks, OTF), so patch
  120 now reads 8.6 GB with the UNet-SN discriminator, as measured (11.3 GB before).
- **One estimate for the Configuration and Training tabs**, now counting the losses
  and the discriminators. Each loss network and each discriminator was benchmarked on
  the GPU (GTX 1080 Ti, FDAT-RD medium): its VRAM and its time per iteration are added
  to the estimate, scaled with the batch and the HR patch. Examples: LineDistiller
  +0.7 GB, DISTS +0.65 GB, A-DISTS +3.3 GB and ×2.6 slower, MS-SSIM-L1 −28 % speed,
  LDL −29 % speed (it runs the EMA generator), UNet-SN +0.5 GB and −15 %. Checked
  against the real FDAT-RD runs: VRAM within 3 % (8.12 / 8.61 / 9.41 GB), speed within
  1 % without GAN and within 5 % with it. The Training tab used to ignore losses,
  discriminators and checkpointing, and estimated an LQ patch of 128 at 2x as 64.
- FDAT-RD pads its input to a multiple of 60 (2x unshuffle): LQ 64, 96 and 120 cost the
  same, LQ 128 runs at 180 (×3 slower, +3.5 GB). The estimate now follows this.
- FDAT-RD **without pixel-unshuffle** (`unshuffle_mod: false`, the engine default) is
  now estimated: the body runs at full LQ resolution (multiple of 30, LQ 64 -> 90),
  several times slower and heavier. The estimate always assumed unshuffle on: medium at
  LQ 64 with 2 discriminators showed 1.65 it/s / 5.0 GB for a real ~0.6 it/s / 8.7 GB
  (the log keeps the full-resolution validation peak). Now 0.62 it/s / 8.73 GB;
  generator within 2-5 % of the bench at LQ 64 and 96.
- The estimate also reads the loss criterion: VGG Perceptual / FP16 with **fd** or
  **fd+l1** costs +5.1 to +5.4 GB and runs 35-45 % slower than with l1/charbonnier. The
  other criteria (l1, l2, charbonnier, huber, pd) of every loss measured within the
  noise. DUNet, DocRes-Disc and the NeoSR discriminators (UNet, PatchGAN, MetaGAN,
  EA2-FPN) are measured too.
- Dataset modes measured in real 400-iteration trainings (FDAT-RD medium, same
  config): paired 1.195 it/s / 9.24 GB, OTF 1.162 / 9.34, bicubic 1.12 / 9.34 — OTF costs
  about 3 %, now in the estimate. MoTSR (sequence mode) trained at 10.6 it/s / 0.8 GB:
  its old value, never measured, predicted 0.36 it/s (30× too slow). Estimate vs these
  runs: within 3 % speed and 4 % VRAM.
- Sequence mode: saving no longer fails on a validation folder with too few frames when
  validation is off.
- **Loss criteria the engine refuses are no longer offered** (they crashed the training
  at launch): Percep. Anime and LineDistiller l2, Gradient Variance / Color / Luma huber.
  Percep. Anime now offers charbonnier. On traiNNer-redux, VGG Perceptual offered l2 and
  huber, which were silently turned into l1: it now offers charbonnier, l1, fd, pd,
  fd+l1 and pd+l1 (Percep. FP16 too). A config or `.uss.json` that still holds a refused
  value gets the loss default instead of crashing.
- **Edge Loss is traiNNer-redux only**: NeoSR has no edge loss and never read the
  `edge_opt` the app wrote, so the option did nothing there. Its criterion / corner
  fields, which no engine reads, are removed.
- EA2-FPN (NeoSR discriminator) needs an HR patch that is a multiple of 32 (LQ 120 in 2x
  = HR 240 crashes); its tooltip says so.
- **44 settings got the tooltip they were missing**: every loss criterion (with the
  values each engine accepts), SparK path, Gradient Variance patch, LDL window, Focal
  Frequency alpha, Contextual distance / band width, HSLuv / BCE-Dice / Consistency /
  MS-SSIM sub-settings, GAN weight, 2nd discriminator type and weight, channels_last,
  workers, print frequency, validation tiles, debug limits.
- The VRAM line of the Configuration tab also shows the estimated training speed. The
  capability bars use the same estimate: "VRAM Lightness" follows the VRAM gauge,
  "Speed" is the inference speed of the architecture only (training options no longer
  lower it), the GAN weight and the sharpness/perceptual losses now move Sharpness,
  Texture and Fidelity, and 2x no longer counts as heavier than 4x.
- **Resume after a full disk**: a save interrupted by a full disk left a truncated
  `.state`, and every relaunch crashed on it. Auto-resume now picks the newest
  *complete* state (readable file + its resume models) and logs the skipped ones. A
  warning appears when free space drops under 5 GB, and the end banner no longer says
  "saved" when saves failed.
- **Validation set check**: a warning (on save and in the training log) when the
  validation LQ images are not the GT divided by the scale (e.g. a 1x set used for a 2x
  model). Such a set was accepted silently and made PSNR/SSIM meaningless.
- **MixUp / CutMix / ResizeMix / CutBlur** sliders are now real per-batch
  probabilities, as their tooltip says. They were relative weights before (4 × 0.05
  really trained with 4 × 0.15). Default values give the same result as before.

### 5. Degradations & LQ Generator
- **Live OTF preview rewritten** to match the training pipeline exactly: sinc/jinc
  kernels, intermediate resizes, final order, JPEG 2. It used to show something
  different from what was trained.
- **7 hidden degradation settings** are now exposed: JPEG prob 2, sinc probabilities,
  generalized/plateau kernel shapes. The discriminator pretrain field is also exposed.
- **Custom GPU degradations** fixed:
  - Dithering and chroma subsampling now really run in training.
  - Screentone angle, shape and colour space are now applied.
  - A chromatic-aberration/VHS bug silently **cancelled every custom degradation** on
    ~10 % of iterations.
- **DPID** downscaling (pepedpid) and a new **4×4 DCT block** degradation (Custom 5).
- **Phase degradations** (traiNNer-redux, with a GAN): a second OTF degradation profile
  for the GAN phase. "Separate GAN profile" + "Edited profile: PSNR phase / GAN phase" in
  the Degradations tab. Before the GAN start, the PSNR profile is used; after it, the
  engine moves to the GAN profile with a linear ramp over "Transition" iterations
  (15,000 by default, 0 = immediate switch). Typical use: light degradations while the
  network learns the line structure, then realistic ones (blur, JPEG, banding) for the
  GAN. Blur 1/2, resize, noise, JPEG, thicklines and every custom degradation change
  during the run; kernel shapes (sigma, size, sinc) and the final sinc stay on the PSNR
  profile. The ramp position comes from the saved GAN start, so a resumed run continues
  where it was. The training log shows both profiles at launch and the progress
  ("GAN profile 50%").
- **Phase losses** (traiNNer-redux, with a GAN): the same idea for the loss weights.
  "Separate GAN profile" + "Edited profile: PSNR phase / GAN phase" + "Transition (it)"
  at the top of the Losses panel. A loss unchecked in a profile has weight 0 in that
  phase and is not computed. Typical use: structural losses only (Charbonnier, color,
  LDL…) during the PSNR phase, perceptual ones (Perceptual Anime, LineDistiller, DISTS…)
  added at the GAN start, or simply other weights. The weights ramp linearly after the
  GAN start (5,000 iterations by default, 0 = immediate switch) and a resumed run
  continues where it was. Every loss of both profiles is loaded from the start: the VRAM
  shown by the estimator is the full run's, with no out-of-memory surprise when the GAN
  starts.
- **Training tab**: next to the phase badge, an orange **Degradations** badge and a cyan
  **Losses** badge (profile 1 (PSNR) → `1 → 2 · N %` → profile 2 (GAN)), shown only when
  the matching option is on.
- The Degradations and Losses badges now follow the GAN profile to 100 %: the engine
  console wraps long log lines and inserts the source file name (`sr_model.py:737`) in
  the middle of the message, which hid the ramp lines from the badges.
- A `train.py` remembered from another engine folder (`~/USRS` vs the active engine) is
  re-aligned to the active engine before launch, so a run no longer starts on an older
  engine (missing archs such as NERVE: `No object named 'nerve'`) and no longer writes
  its experiment into the wrong folder. Custom script paths are left alone.
- **Training no longer dies on one bad gradient** (traiNNer-redux): the engine only
  checked the loss for NaN/Inf. Without AMP, a single infinite gradient (finite loss) went
  through gradient clipping into the optimizer and turned every weight, EMA included,
  into NaN, then stopped the run 10 iterations later. Generator and both discriminator
  updates are now skipped on a non-finite gradient (warning in the log); the run stops
  only after 10 bad generator gradients in a row.
- **Closing the app during a training run now asks first**: *Save and quit* (clean stop,
  model and state saved, then the app closes), *Keep running* (the window closes, training
  goes on in the background) or *Cancel*. Closing used to leave the training process
  running unseen, with no way to stop it from the app.
- The MixUp / CutMix / ResizeMix / CutBlur tooltips get their description back after
  switching from a video arch to an image arch (they kept showing "disabled in sequence
  mode").
- Loading a YAML config without its `.uss.json` file no longer resets the custom
  degradation ranges (posterize bits, banding levels, aliasing scale…) to their
  defaults, and no longer turns the generator/discriminator `true`/`false` options into
  `1`/`0`.
- Fixed: Motion Blur crash, inert Sharpen intensity, a crash in the live degradation
  preview.
- **Banding fixed**: the level formula kept only 3–4 levels per channel (`levels**(1/3)`),
  so "banding 32" gave heavy posterization instead of soft bands. The setting now means
  real levels per channel. Quantize depth and posterize now round to the nearest level
  (they truncated, which darkened the image).
- **New Smooth banding** degradation (Custom tab): bands only on flat areas (smooth
  gradients, skies, shading), lines and textures untouched. Option to work in linear
  light so dark tones band like on a real encode. Also in the live preview and the phase
  degradations.
- **Resize modes are kept from the YAML**: the app always wrote 4 resize modes at 25 %
  each, `nearest-exact` included, whatever the config said. Nearest at 25 % gives jagged
  "zipper" lines. The modes and probabilities from the config are now kept; the default
  is bilinear / bicubic / lanczos (0.3 / 0.4 / 0.3). `poisson_scale_range` (no widget)
  is kept too.
- **`true`/`false` architecture options were reset on load**: a YAML `false` (for
  example `unshuffle_mod: false`, `use_checkpoint: true`) did not match the combobox
  item, so the default stayed and was saved back. Every boolean architecture option is
  now loaded correctly.
- **LineDistiller patch check**: LineDistiller (traiNNer-redux) crashes at the first
  iteration when Patch Size × scale is not a multiple of 16 (for example LQ 60 at 2x:
  `The size of tensor a (15) must match the size of tensor b (14)`). Saving the config
  now warns and suggests the nearest valid sizes, whether LineDistiller is on in the PSNR
  or in the GAN profile. The FDAT-RD alignment warning no longer suggests a size that
  LineDistiller rejects.
- **Custom degradations kept on reload**: screentone, dithering, halo (wtp), color levels,
  saturation, pixel shift and the other "Custom 3/4/5" degradations were not read back
  when a .yml was opened in the Config tab. Saving it again silently turned them off.
- **Training time estimate: delayed GAN and Real-CUGAN fast**: with a delayed GAN start
  (gan_start_iter, or gan_auto_start), the estimate now adds a PSNR phase without the
  discriminators and the GAN-only losses, then the GAN phase. It used to count the GAN for
  the whole run. Real-CUGAN `fast` (pixel-unshuffle input) now counts a quarter of the
  generator work. Real-CUGAN fast 2x run: 2 d 20 h announced, now 1 d 4 h (measured
  4.1 it/s PSNR phase, 1.32 it/s GAN phase; estimate 3.9 / 1.1 it/s).
- **New archs: WSR and DUAL (Rift73)** — training (traiNNer-Redux) and inference
  (Quick Upscale, sequence runner, ONNX converter):
  - **WSR**: wide residual trunk + LR/4 context stream, Halo attention on `wsr` and
    `wsr_compact`. Presets `wsr_mobile`, `wsr_superultracompact`, `wsr_ultracompact`,
    `wsr_compact`, `wsr`. Convolutional, simple TensorRT export.
  - **DUAL**: dual-context transformer (16 px windows + 32/64 neighbourhood attention).
    Presets `dual_light` to `dual_xl`. Heavy: its fast paths need BF16 + Triton, absent
    on Pascal (GTX 10xx); on a 1080 Ti `dual_light` trains ~9x slower than NERVE.
  - Scales 1x / 2x / 4x. The preset and the scale are read back from the weights.
  - Licences: WSR MIT (+ Apache-2.0 for its SPAN-derived Conv3XC), DUAL MIT.
  - Speed / VRAM estimates calibrated on real 150-iteration runs (GTX 1080 Ti) of all 11 presets.
- **Fix: SPANF3 models now load in Quick Upscale** (normal and persistent subprocess).
  spandrel does not know SPANF3 (the arch of AnimeJaNai V3.1), so a model trained with
  `spanf3` failed with "architecture non reconnue". Width, depth, scale and
  `unshuffle_mod` are now read from the weights.
- **Fix: the second discriminator's pretrained weights are kept** (traiNNer-Redux).
  A config with `pretrain_network_d2` (fine-tune that reuses both discriminators) lost
  this path when it was saved again from the Config tab, so D2 restarted from random
  weights. New "Pretrain D2" field in Resume / Pretrain, read from and written to the
  config (written only when the second discriminator is enabled).

### 6. Tools
- **Frame Extraction**:
  - frame N, every X s or all frames;
  - PNG/TIFF 8/16-bit, JPEG, BMP;
  - preview, Stop, ETA, notification.
- **Guided Color Fix (aniCC, Bendel)** — learned colour correction for non-uniform
  shifts, downloaded on first use.
- **Model Converter**:
  - works in the portable build;
  - fixed-size ONNX for window-attention archs;
  - automatic check of "dynamic" ONNX files;
  - video-model export.
- **Native NNEDI3 anti-aliasing** (no VapourSynth needed) in Post Processing.
  Undistort got its own precision setting.
- New **Import Config** page: imports a Real-ESRGAN / SwinIR / RCAN / HAT / BasicSR
  config.
- Validation viewer: auto-reload and before/after slider. LQ and SR are shown at the
  same size (LQ enlarged with nearest neighbour, true pixels); wheel zooms under the
  cursor and holding the middle button pans both images together.
- **PSNR → GAN pipeline page**: phase 2 now starts from phase 1's final model
  automatically (Redux) and both phases can be queued. New mode: chain two full option
  files (a `.chained` copy of phase 2 is written, your file is untouched).
- Comparator: auto-loads the last upscale.
- Model Info: "→ SafeTensors" button. `.safetensors` files now show their architecture.
- Quick Upscale:
  - auto tile size based on free VRAM;
  - timing in single-image mode.
- Queue: uses the real training pipeline (monitoring, history).
- Assistant: GPU suitability banner, live ETA, answer validation.
- **AI training analysis**: provider list refreshed (GitHub Models shut down upstream),
  NVIDIA NIM / OpenRouter / OpenAI / Gemini / xAI / DeepSeek / Anthropic models updated
  and live-checked.

### 7. Portable build & installation
- The portable zip is self-contained:
  - its own Python;
  - engines, PyTorch CUDA and all dependencies installed from Settings;
  - optional model files downloaded on first use;
  - **Git is no longer required**: without it, the engine source is downloaded as an
    archive of the tested commit;
  - **ffmpeg is no longer required**: a portable ffmpeg is installed with the engine.
- PyTorch is chosen from the GPU: **torch 2.7.0 + CUDA 12.6** from GTX 10 (Pascal) to
  RTX 40, **CUDA 12.8** for RTX 50 / Blackwell; an outdated NVIDIA driver is flagged
  before installing.
- Verified with a fresh install on an empty machine profile without Git or ffmpeg:
  install, training, video upscale, conversion.
- traiNNer-redux follows `master`, **pinned to a tested commit**. NeoSR is still
  supported; its upstream has not changed since mid-2025.
- **Python packages pinned to the tested versions**: every dependency install uses a
  constraints file generated from the environment this release was validated with, so a
  new install gets exactly the tested packages instead of whatever PyPI serves that day.
  PyTorch and its CUDA libraries still come from the wheel matching your GPU.
- The bundled Python no longer ships build-machine caches and launchers (`__pycache__`,
  `Scripts/`); the app only runs `python -m pip` / `python -m virtualenv`.
- Fixed:
  - the NeoSR installer reported a broken clone as a success;
  - frame extraction with ffmpeg ≥ 7;
  - SRFormer and other spandrel "extra" archs were refused in the portable build;
  - several engine-side files were missing from the build.
  - **ONNX models ran on the CPU**: the latest onnxruntime-gpu (1.27+) is built for CUDA 13,
    which matches neither PyTorch's CUDA 12 nor Pascal GPUs. Pinned to the last CUDA-12
    build (1.26), ONNX inference is back on the GPU; unreadable onnxruntime messages
    removed from the log. Existing installs: Settings → reinstall dependencies.

### 8. Full functional audit (every function, every setting)
Three static audits of the whole app (Quick Upscale, tools, settings / post-processing /
publishing: 147 findings) were checked one by one and fixed, then tested on real data
and real trainings. The main fixes:
- **Training**:
  - **x3 with OTF degradations crashed on both engines.** The crop size was never a
    multiple of 3 (traiNNer-redux gt_size + 32, NeoSR a fixed 512). An engine patch,
    applied at every launch, rounds it to the scale. After it: 7/7 Redux and 4/4 NeoSR
    archs train at x3.
  - NeoSR refused BF16 without AMP and Wavelet Guided without GAN. BF16 is now greyed
    out until AMP is ticked; Wavelet Guided without GAN is refused with a clear message.
  - NeoSR ESRGAN is x1/x2/x4 only: x3/x8 are now flagged before the launch.
  - The VRAM and speed estimates are recalibrated on real trainings: VRAM on 162 archs
    (it was 2.8× too low on average), speed on 159 archs measured over 150 iterations
    (it was off by ×10 in the median; now within ±10 % for 145 of 149). The VRAM line
    warns when the GPU memory will be exceeded.
- **Quick Upscale**:
  - A folder of frames with a video model now goes to the sequence pipeline (it was
    refused). A `.pth` video model is detected in the portable build.
  - 16-bit output and JPEG/WEBP quality are respected everywhere: persistent mode, every
    inference runner, after Temporal Fix / Undistort, skip-frames. Tile, tile pad and AMP
    reach the NeoSR runners.
  - **Temporal Fix / Undistort wrote frames shifted by half a window** (the first result,
    centred on frame 4, replaced frame 1). Both now write frame k with frame k's
    correction. Strength 0 is a true pass-through, and Undistort feeds Temporal Fix
    instead of both reading the raw frame.
  - Undistort TMT: the download links were dead (files renamed upstream); the new ONNX
    model is supported; frame sizes that are not multiples of 16 work. All neural modes
    (Temporal Fix PTH / CPU / ONNX, Undistort TMT PTH / CPU / ONNX) tested on real
    frames.
  - Color Fix: the reference image reaches every path and keeps 16-bit; LAB respects the
    selected channels; a missing reference or a guided failure is reported, not
    silently ignored.
  - Output safety: an output can no longer overwrite a source image; `a.jpg` + `a.png`
    no longer write the same file; Stop works during extraction; no reassembly after a
    Stop; the video is rebuilt from this run's frames only.
  - An explicit scale the model cannot produce is announced before the run (detection
    checked on 51 models). Options that do not apply (single image, sequence pipeline)
    are greyed out with the reason.
- **Tools**:
  - **Import Config** rewritten on the full config loader: losses absent from the file
    are switched off, weights and criteria, scheduler, batch, LR patch and network
    parameters are imported, and the architecture is mapped to a name the engine has. A
    Real-ESRGAN x4plus config was imported and trained.
  - **Publish**: NeoSR `.pth` checkpoints are listed; the training state matches the
    exact iteration; `resume/` gets the real G and D; a `.pth` is no longer renamed
    `.safetensors`; name/version are checked; the datasheet reads OTF, pretrain and the
    LR patch from the config.
  - **LMDB Creator** writes the `meta_info.txt` both engines require, reports non-ASCII
    and duplicate names, and sizes the database from the data: a fixed 1 TB map made
    Windows create a 1 TB `data.mdb` that NeoSR could not open. Trained for real on NeoSR
    (paired, GT + LQ `.lmdb`). The pinned traiNNer-redux cannot read LMDB (its OTF and
    paired loaders are broken for it): the configuration says so instead of crashing.
  - **Resume**: the right config is chosen by its content, running trainings are
    detected, and Resume turns on Auto-resume.
  - **Benchmark**: exact precision modes (no hidden terminal prompt), Val/test image
    passed, Stop ends the whole process tree, every engine/type can list its tests.
  - **LQ Generator**: real hue jitter, a real disc kernel and error-diffusion dithering.
    Overlapping HQ/LQ/validation folders and frames left by an earlier extraction are
    refused. Train keeps at least one clip.
  - **Comparator**: typed paths load; the smaller image is enlarged (the SR result used
    to be shrunk).
- **Post Processing**:
  - Real 16-bit output; Temporal Fix / Undistort settings kept and resettable.
  - Edge Cleanup really removes isolated hot and cold pixels (the pixel itself was
    outside the edge mask).
  - The end status reflects skipped filters.
- **Settings**:
  - The installers fail loudly when dependencies or PyTorch are missing.
  - Updating traiNNer-redux returns to the tested commit.
  - "Reinstall all deps" really reinstalls.
  - The gallery opens its real URL.

### Known limitations
- EEDI3 anti-aliasing still needs VapourSynth (NNEDI3 is native).
- MoTSR is 2x only. MambaIR cannot be exported to ONNX.
- Second discriminator: traiNNer-redux only.
- MambaIR's fast kernel needs torch 2.7.0+cu126 (the default engine install). With
  another torch (RTX 50 / cu128, old drivers / cu118) MambaIR uses the pure-PyTorch scan:
  on an 11 GB card that means LQ 32 / batch 1 at ~0.01 it/s. When a run needs more VRAM than the card has,
  Windows can spill it to shared memory: the run then crawls instead of stopping (seen
  with GRL-B and Swin2SR-L 4x at batch 2).
- LMDB datasets: NeoSR only.
- RTX 50 / RTX PRO 6000 (Blackwell): training uses native PyTorch kernels (cu128).
  ONNX inference (onnxruntime-gpu 1.26, CUDA 12.8) relies on the driver compiling GPU code
  for the card on first use; not yet verified on Blackwell hardware. The TensorRT backend
  is only used when TensorRT (10.8+ for Blackwell) is installed on the system.

---

<a name="v260--français"></a>
## v2.6.0 — Français

*Changements depuis la v2.5.7 (juin 2026).* La v2.6.0 est la plus grosse mise à jour à
ce jour :
- l'interface a été entièrement réécrite ;
- ~45 architectures ont été ajoutées ou rendues utilisables partout ;
- la super-résolution vidéo est devenue un pipeline complet ;
- l'app a été validée sur des milliers de configurations générées et de vrais
  entraînements.

### 1. Nouvelle interface (Qt / PySide6)
- **Réécriture complète** de l'app CustomTkinter en Qt : 7 onglets, 13 pages d'outils,
  7 sous-pages de Configuration, 5 sous-onglets de Paramètres et 5 fenêtres
  d'Entraînement. Toutes les fonctions de la v2.5.7 ont été portées, puis auditées une
  par une contre l'ancienne app (23 écarts corrigés).
- **Interface anglaise complète** (~1 200 textes traduits), choix de la langue dans
  Paramètres, infobulles FR/EN sur tous les paramètres. Cela inclut les paramètres par
  architecture, qui n'avaient presque aucune infobulle avant (84 avaient un texte dédié
  qui ne s'affichait jamais).
- Thème sombre / clair / auto, **échelle de l'interface 90–150 %**, progression dans la
  barre des tâches Windows (vert en cours, jaune arrêté, rouge planté), sons et
  notifications Windows.
- **Barre de progression de l'onglet Entraînement** : plus épaisse, avec le pourcentage
  et l'itération (`21% · 52 500 / 250 000`) écrits au milieu. Pour un entraînement en 2
  phases (GAN différé ou démarrage automatique), un badge indique la phase en cours —
  **PSNR** en vert avec le compteur du démarrage auto du GAN (`GAN auto 2/3 (≥ 85 000)`),
  **GAN** en mauve avec son itération de départ — et la barre prend la couleur de la
  phase.
- **Badges sur les losses** : chaque loss qui s'appuie sur un réseau pré-entraîné indique
  sur quoi il a été entraîné — **Anime** (Percep. Anime, LineDistiller) ou **Photo**
  (Perceptual VGG, FP16, DISTS, Contextual, FDL, SparK). Sans badge : aucun réseau appris.
- **Dossier moteur renommé** `~/IA_Engine` → `~/USRS`. Les données existantes ne sont
  jamais déplacées ni supprimées.
- Réinstaller un moteur est maintenant **bloqué si son dossier contient tes datasets,
  expériences ou modèles**. Avant, il supprimait le dossier sans vérifier.

### 2. Architectures
**Nouvelles, entraînables et utilisables partout** (Quick Upscale, Convertisseur, Info
Modèle) :
- **TFDAT v2** (Kim2091). Fait tourner **2x_bndl_animefilm_v3** de Bendel (clips de
  9 frames, sortie identique à l'ONNX officiel) ; les modèles bndl précédents (v1 /
  v1.5 DAT2 et FDAT, tfdat_v2) fonctionnent aussi.
- **MoTSR** (xyether) — SR vidéo 2x récurrente, coupes de scène détectées, étapes
  d'entraînement joint / partial / bootstrap.
- **MambaIR / MambaIRv2** (csguoh, ECCV 2024 / CVPR 2025). Leur kernel CUDA
  (`mamba_ssm`, Linux uniquement à l'origine) est compilé pour Windows et **livré avec
  l'app** : l'installation du moteur l'installe (GTX 10 → RTX 40, torch 2.7.0+cu126).
  GTX 1080 Ti, x2, batch 2, LQ 64 : ~1,9 it/s pour 3,6 Go (MambaIR) et ~1,8 it/s pour
  4,4 Go (MambaIRv2). Sans le kernel, repli sur un scan pur PyTorch (~100× plus lent,
  beaucoup plus de VRAM), signalé dans le log.
- **HEART** et **NERVE** (Phhofm).
- **DocRes** (restauration de documents, 1x) et son discriminateur.
- Désormais proposées à l'entraînement : **TFDAT**, **NinaSR**, **HAT-iLN / DAT-iLN**.
- **WireSR** (CF2lter, « 1x_Wired ») — anti-crénelage / restauration des traits fins.
- **FDAT-RD** (stinkybread) — évolution de FDAT pour l'anime (fenêtres rectangulaires 10×30
  + dictionnaire de tokens appris, export TensorRT statique) : presets small / medium /
  large, option gradient checkpointing (VRAM d'entraînement ÷ 7). Correction d'une ligne
  sombre sur le bord bas/droit des upscales dont la taille est un multiple de 60 (720×480,
  1920×1080) : l'archi garde toujours une petite marge miroir, comme à l'entraînement.
  Poids et taille de sortie inchangés. Un modèle entraîné avec un patch LQ non multiple
  de 60 (2x unshuffle, sinon 30) garde cette ligne dans d'autres outils (chaiNNer, ONNX) :
  la vérification avant training avertit et propose 60 ou 120, l'info-bulle l'explique.

**Couverture Quick Upscale**, en mode normal et en subprocess persistant :
- **traiNNer-redux** : ArtCNN, LKFMixer, RTMoSR, MoSRv2, Sebica, LMLT, MAN, EIMN, ELAN,
  GateRV3, HiT-SIR/SNG, CascadedGaze, EMT, ESCRealM, FlexNet, SRFormer/v2.
- **NeoSR** : PlainUSR, ASID, GRFormer, DCT, HiT-SRF, HMA, SRFormer-medium, HASN, MSDAN.
  AetherNet (7 presets) était cassé à 100 % et fonctionne maintenant.
- Chacune a été testée en la reconstruisant depuis de vrais poids, avec chargement
  strict.
- **Balayage complet de l'inférence** (octobre 2026) : pour chaque architecture entraînable
  par l'app, le réseau a été construit par son moteur depuis la config générée par l'app, en
  2x et 4x, puis passé dans Quick Upscale en image seule et en dossier avec le subprocess
  persistant. Résultat : **185 / 185** modèles fonctionnent en 2x et en 4x (119 / 185 avant ces corrections), et le mode dossier persistant marche pour les 179 archs image (les archs vidéo passent par le pipeline séquence). Seul ESCRealM n'a pas été testé : il demande `triton`. Le tableau complet est dans le README (« Architectures supportées »). Corrections qui
  en découlent :
  - **Échelle** : l'échelle détectée dans les poids était transmise aux runners avec 4 par
    défaut, ce qui écrasait leur propre détection (exacte). La détection a aussi été
    réécrite : les upsamplers pixel-shuffle en plusieurs étages se multiplient (HAT /
    SwinIR / DAT 4x lus 2x avant), et les modèles MoSRv2 / RTMoSR `unshuffle_mod`, DocRes
    (1x) et SpanPP sont lus correctement. Détections fausses sur 370 modèles de test :
    74 → 2 (ParagonSR v1 et MSDAN, dont l'échelle n'est pas dans les poids : l'échelle
    choisie dans l'interface est utilisée).
  - **Mode dossier avec « subprocess persistant »** : plantage dès la première image sur
    toutes les archs subprocess traiNNer-redux (ArtCNN, ELAN, MAN, ParagonSR, FDAT-RD,
    HEART…), à cause d'une valeur de retour mal dépaquetée. Corrigé, avec repli image par
    image si le worker échoue sur une image.
  - L'onglet Quick Upscale lançait toujours le worker traiNNer-redux : les archs propres à
    NeoSR (ESC, GRFormer, CATANet, DRCT, LMLT…) n'y démarraient pas et passaient sans le
    dire à un subprocess par image. L'onglet choisit maintenant le worker NeoSR pour elles.
  - Un `print()` dans le constructeur d'une arch (DocRes) cassait le protocole du worker ;
    le stderr des workers n'était jamais lu non plus, ce qui pouvait bloquer un long lot.
    Les deux sont corrigés.
  - Reconstruites depuis les poids au lieu des valeurs par défaut : **SPANPlus** S / ST /
    STS (le worker persistant les chargeait même en non strict, donnant une image fausse
    sans erreur), **DRCT-L / XL**, **LMLT tiny / large**, **MetaFlexNet**, **DAT-iLN** (tous
    les presets), et les versions NeoSR de **CATANet, KRGN, MAN, MoSRv2** (elles partaient
    vers le runner traiNNer-redux, qui ne les a pas), et **SCUNet-aaf6aa** (spandrel le lisait
    comme un SCUNet simple).
  - Les modèles SRFormer / DRCT enregistrés avec une autre taille de masque d'attention se
    chargent de nouveau ; GRFormer ne plante plus sur un PyTorch récent (patch moteur) ;
    ParagonSR v1 / v2 utilisent la bonne échelle ; TSCUNet et MambaIR ne sont plus pris
    pour d'autres archs.

**Discriminateurs GAN** :
- Nouveaux discriminateurs : **Spectral** (DualFormer, ICCV 2023), **NeXtSRGAN** et
  **Projected GAN** (backbone EfficientNet gelé).
- **Second discriminateur** (n'importe quelle paire, pondérée) : chaque discriminateur
  a sa propre pause Adaptive-D. Un avertissement visible signale les paires redondantes.
- **Démarrage différé du GAN** (traiNNer-redux) : le générateur apprend d'abord avec ses
  autres losses seulement, puis la loss GAN et le ou les discriminateurs démarrent.
  Conseillé pour un modèle entraîné de zéro. Deux modes :
  - à une itération fixe (« Début GAN (iter) ») ;
  - **Auto** : quand le PSNR de validation ne monte plus (patience, gain minimum,
    itération minimum ; « Début GAN » sert alors de démarrage au plus tard).
  L'option « Relancer le planning LR » remet le LR à sa valeur de départ au démarrage du
  GAN, et les milestones comptent à partir de là, comme un 2e training séparé. L'état du
  démarrage est sauvegardé : une reprise de training le conserve.
- La VRAM du discriminateur est maintenant comptée dans l'estimation. Elle ne l'était
  jamais avant, même pour les UNet/PatchGAN d'origine.

### 3. Super-résolution vidéo / temporelle
- **Entraînement** :
  - nouveau mode dataset « sequence » (`PairedVideoDataset`, `clip_size`), verrouillé
    automatiquement pour les archs vidéo ;
  - les augmentations batch qui corrompraient les clips sont désactivées dans ce mode ;
  - les chemins Train/Val sont mémorisés séparément en mode vidéo et en mode image ;
  - une erreur claire s'affiche quand les clips de validation sont plus courts que
    `clip_size`.
- **Quick Upscale** accepte un dossier de frames **ou un fichier vidéo** pour les
  modèles vidéo (TSCUNet, TemporalSPAN v1/v2, TFDAT v1/v2, MoTSR), et aussi pour
  **n'importe quel modèle image** (frames traitées une à une). Réassemblage en vidéo en
  option.
- **Options d'encodage** du réassemblage :
  - conteneur MP4/MKV/WebM ;
  - x264/x265 ou NVENC H.264/HEVC/AV1, chacun testé sur ton propre GPU ;
  - CRF ou débit, presets de vitesse.
- Pipeline séquence en **vrai 16 bits** (chargement pyvips, vraie sortie PNG16).
- **Générateur LQ, mode vidéo** : extrait les frames, les dégrade et crée un bloc de
  validation contigu (fin / début / aléatoire).
- **Export ONNX des modèles vidéo** :
  - un graphe `1x(T·3)xHxW` (format vs-mlrt / VapourKit) ;
  - les graphes seed + récurrent de MoTSR.
- Les modèles vidéo **tournent maintenant dans la version portable**. Avant, ils y
  échouaient avec « PyTorch non installé ».

### 4. Fiabilité de l'entraînement
- **Validé sur de vrais entraînements**, ce qui a révélé plusieurs bugs sérieux :
  - le mauvais Python était utilisé, donc le training plantait au démarrage ;
  - le graphe PSNR restait vide ;
  - **Soft Stop** ne sauvegardait pas. Il s'arrête maintenant proprement en ~10 s avec
    un checkpoint ;
  - le tableau de bord gardait les chiffres du run précédent ;
  - le **tuilage de validation** était toujours ignoré.
- **Harnais de tests automatique** : ~2 500 configurations générées (archs × moteurs ×
  pertes × optimiseurs × GAN). Vrais bugs corrigés :
  - les paramètres en liste de NeoSR étaient lus en float ;
  - batch = 1 avec augmentations échouait ;
  - les archs « preset » (hat_s/m/l, swinir_medium…) recevaient des paramètres en
    double ;
  - il manquait `scale` à AetherNet, qui n'était jamais injecté dans NeoSR ;
  - certains noms de discriminateurs étaient faux ;
  - des archs et optimiseurs absents du moteur choisi étaient quand même proposés ;
  - les archs à scale imposé (CascadedGaze, RealCUGAN, MSDAN) et les tailles de patch
    minimales n'étaient pas contrôlées.
- **MambaIR** : le générateur de config n'écrivait jamais `upscale`, donc les modèles
  s'entraînaient en 2x au lieu de 1x sans le dire.
- Aussi corrigés : tf32 / `fast_matmul`, `ReduceLROnPlateau`, Schedule-Free sous NeoSR,
  `torch.compile` (grisé sous Windows), `use_blur` de la perte consistency, garde-fous
  AESOP et ECO.
- **Patchs moteur auto-réparants** : les corrections des fichiers du moteur amont (25+)
  sont réappliquées automatiquement après une mise à jour ou un re-clone du moteur.
- **NeoSR cassé après un Quick Upscale (corrigé)** : avant chaque upscale, l'app copiait
  aussi ses archis custom traiNNer-redux (FDAT-RD, MoTSR, TFDAT v2, MambaIR, WireSR, les
  nouveaux discriminateurs…) dans le moteur NeoSR. Plusieurs importent traiNNer, et NeoSR
  charge tous les fichiers d'archi de son dossier : plus aucune archi NeoSR ne se
  construisait, ni en training ni en inférence (0/49 au test, 49/49 après correction).
  Elles ne vont plus que dans traiNNer-redux, et les copies déjà présentes dans NeoSR
  sont désactivées (renommées `*.usrs_disabled`, rien n'est supprimé).
- **Estimateur VRAM / vitesse** calibré sur de vraies mesures : MambaIR/v2, MoTSR,
  discriminateurs, et de vrais trainings pour HEART, NERVE, WireSR et FDAT-RD
  small/medium/large. HEART, NERVE et WireSR utilisaient des ratios de benchmark isolé,
  qui sous-estimaient leur vraie vitesse d'entraînement de 8× à 30×. Gain VRAM du
  gradient checkpointing corrigé (×0,66 en training réel, pas ×0,25). La VRAM de FDAT-RD
  ne croît plus comme une puissance de la taille de patch : l'essentiel est fixe (poids,
  réseaux de loss, OTF), donc le patch 120 affiche 8,6 Go avec le discriminateur UNet-SN,
  comme mesuré (11,3 Go avant).
- **Une seule estimation pour les onglets Configuration et Entraînement**, qui compte
  maintenant les losses et les discriminateurs. Chaque réseau de loss et chaque
  discriminateur a été mesuré sur GPU (GTX 1080 Ti, FDAT-RD medium) : sa VRAM et son
  temps par itération s'ajoutent à l'estimation, mis à l'échelle du batch et du patch
  HR. Exemples : LineDistiller +0,7 Go, DISTS +0,65 Go, A-DISTS +3,3 Go et ×2,6 plus
  lent, MS-SSIM-L1 −28 % de vitesse, LDL −29 % (il fait tourner le générateur EMA),
  UNet-SN +0,5 Go et −15 %. Vérifié sur les vrais trainings FDAT-RD : VRAM à 3 % près
  (8,12 / 8,61 / 9,41 Go), vitesse à 1 % près sans GAN et à 5 % près avec. L'onglet
  Entraînement ignorait les losses, les discriminateurs et le checkpointing, et estimait
  un patch LQ de 128 en 2x comme un patch 64.
- FDAT-RD complète son entrée au multiple de 60 (unshuffle 2x) : LQ 64, 96 et 120
  coûtent pareil, LQ 128 tourne en 180 (×3 plus lent, +3,5 Go). L'estimation en tient
  compte.
- FDAT-RD **sans pixel-unshuffle** (`unshuffle_mod: false`, le défaut du moteur) est
  maintenant estimé : le corps tourne à la pleine résolution LQ (multiple de 30, LQ 64
  -> 90), plusieurs fois plus lent et plus lourd. L'estimation supposait toujours
  l'unshuffle actif : medium en LQ 64 avec 2 discriminateurs affichait 1,65 it/s / 5,0 Go
  pour ~0,6 it/s / 8,7 Go réels (le log garde le pic de la validation en pleine
  résolution). Maintenant 0,62 it/s / 8,73 Go ; générateur à 2-5 % du bench en LQ 64 et 96.
- L'estimation lit aussi le critère des losses : VGG Perceptual / FP16 en **fd** ou
  **fd+l1** coûte +5,1 à +5,4 Go et tourne 35 à 45 % moins vite qu'en l1/charbonnier.
  Les autres critères (l1, l2, charbonnier, huber, pd) de toutes les losses mesurent dans
  le bruit. DUNet, DocRes-Disc et les discriminateurs NeoSR (UNet, PatchGAN, MetaGAN,
  EA2-FPN) sont mesurés aussi.
- Modes de dataset mesurés en vrais trainings de 400 itérations (FDAT-RD medium, même
  config) : paired 1,195 it/s / 9,24 Go, OTF 1,162 / 9,34, bicubic 1,12 / 9,34 — l'OTF
  coûte environ 3 %, maintenant dans l'estimation. MoTSR (mode séquence) a tourné à
  10,6 it/s / 0,8 Go : son ancienne valeur, jamais mesurée, prévoyait 0,36 it/s (30× trop
  lent). Estimation face à ces runs : à 3 % près en vitesse et 4 % en VRAM.
- Mode séquence : la sauvegarde n'échoue plus sur un dossier de validation trop court
  quand la validation est désactivée.
- **Les critères de loss refusés par le moteur ne sont plus proposés** (ils faisaient
  planter le training au lancement) : l2 pour Percep. Anime et LineDistiller, huber pour
  Gradient Variance / Color / Luma. Percep. Anime propose maintenant charbonnier. Sous
  traiNNer-redux, VGG Perceptual proposait l2 et huber, transformés en l1 sans le dire :
  il propose maintenant charbonnier, l1, fd, pd, fd+l1 et pd+l1 (Percep. FP16 aussi). Une
  config ou un `.uss.json` qui contient encore une valeur refusée reçoit le défaut de la
  loss au lieu de planter.
- **Edge Loss réservée à traiNNer-redux** : NeoSR n'a pas de loss Edge et ne lisait jamais
  le `edge_opt` écrit par l'app, l'option n'y faisait rien. Ses champs critère / corner,
  lus par aucun moteur, sont retirés.
- EA2-FPN (discriminateur NeoSR) demande un patch HR multiple de 32 (LQ 120 en 2x = HR 240
  plante) ; son infobulle le dit.
- **44 réglages reçoivent l'infobulle qui leur manquait** : tous les critères de loss
  (avec les valeurs acceptées par chaque moteur), chemin SparK, patch Gradient Variance,
  fenêtre LDL, alpha Focal Frequency, distance / largeur de bande Contextual,
  sous-réglages HSLuv / BCE-Dice / Consistency / MS-SSIM, poids GAN, type et poids du 2e
  discriminateur, channels_last, workers, fréquence du log, tuiles de validation, limites
  de debug.
- La ligne VRAM de l'onglet Configuration affiche aussi la vitesse d'entraînement
  estimée. Les barres de capacités utilisent la même estimation : « Légèreté VRAM » suit
  la jauge VRAM, « Vitesse » est la vitesse d'inférence de l'architecture seule (les
  options d'entraînement ne la baissent plus), le poids GAN et les losses de netteté /
  perceptuelles font maintenant bouger Netteté, Texture et Fidélité, et le 2x ne compte
  plus comme plus lourd que le 4x.
- **Reprise après disque plein** : une sauvegarde interrompue par un disque plein
  laissait un `.state` tronqué, et chaque relance plantait dessus. L'auto-reprise prend
  maintenant le dernier état *complet* (fichier lisible + ses modèles de reprise) et
  signale les états ignorés. Un avertissement apparaît sous 5 Go libres, et le bandeau
  de fin ne dit plus « sauvegardé » quand des sauvegardes ont échoué.
- **Contrôle du set de validation** : avertissement (à la sauvegarde et dans le log)
  quand les LQ de validation ne sont pas les GT divisées par le scale (ex. set 1x utilisé
  pour un modèle 2x). Un tel set était accepté sans rien dire et faussait PSNR/SSIM.
- Les curseurs **MixUp / CutMix / ResizeMix / CutBlur** sont maintenant de vraies
  probabilités par batch, comme l'annonce leur infobulle. Avant, c'étaient des poids
  relatifs (4 × 0,05 entraînait en réalité avec 4 × 0,15). Les valeurs par défaut
  donnent le même résultat qu'avant.

### 5. Dégradations et Générateur LQ
- **Aperçu OTF live réécrit** pour coller exactement au pipeline d'entraînement :
  noyaux sinc/jinc, redimensionnements intermédiaires, ordre final, JPEG 2. Avant, il
  montrait autre chose que ce qui était entraîné.
- **7 réglages de dégradation cachés** sont maintenant exposés : JPEG prob 2,
  probabilités sinc, formes de noyaux generalized/plateau. Le champ pretrain du
  discriminateur est aussi exposé.
- **Dégradations custom GPU** corrigées :
  - dithering et sous-échantillonnage chroma tournent vraiment en entraînement ;
  - angle, forme et espace couleur du screentone sont maintenant appliqués ;
  - un bug d'aberration chromatique/VHS **annulait en silence toutes les dégradations
    custom** sur ~10 % des itérations.
- Downscale **DPID** (pepedpid) et nouvelle dégradation **blocs DCT 4×4** (Custom 5).
- **Dégradations par phase** (traiNNer-redux, avec un GAN) : un 2e profil de
  dégradations OTF pour la phase GAN. « Profil GAN différent » + « Profil édité : Phase
  PSNR / Phase GAN » dans l'onglet Dégradations. Avant le démarrage du GAN, le profil PSNR
  s'applique ; ensuite, le moteur passe au profil GAN par une rampe linéaire sur
  « Transition » itérations (15 000 par défaut, 0 = bascule immédiate). Usage type :
  dégradations légères pendant que le réseau apprend la structure des traits, puis
  réalistes (flou, JPEG, banding) pour le GAN. Changent pendant le run : blur 1/2,
  resize, bruit, JPEG, thicklines et toutes les dégradations custom ; les formes de
  noyaux (sigma, taille, sinc) et le final sinc restent sur le profil PSNR. La position
  dans la rampe vient du démarrage GAN sauvegardé : une reprise continue au même point.
  Le log affiche les deux profils au lancement et la progression (« GAN profile 50% »).
- **Losses par phase** (traiNNer-redux, avec un GAN) : le même principe pour les poids des
  losses. « Profil GAN différent » + « Profil édité : Phase PSNR / Phase GAN » +
  « Transition (it) » en haut du panneau Losses. Une loss décochée dans un profil a un
  poids 0 dans cette phase et n'est pas calculée. Usage type : losses structurelles seules
  (Charbonnier, couleur, LDL…) en phase PSNR, puis losses perceptuelles (Percep. Anime,
  LineDistiller, DISTS…) ajoutées au démarrage du GAN, ou simplement d'autres poids. Les
  poids suivent une rampe linéaire après le démarrage du GAN (5 000 itérations par défaut,
  0 = bascule immédiate), et une reprise continue au même point. Toutes les losses des
  deux profils sont chargées dès le départ : la VRAM affichée par l'estimateur est celle du
  run complet, sans mauvaise surprise de mémoire au démarrage du GAN.
- **Onglet Entraînement** : à côté du badge de phase, un badge orange **Dégradations** et un
  badge cyan **Losses** (profil 1 (PSNR) → `1 → 2 · N %` → profil 2 (GAN)), affichés
  seulement quand l'option correspondante est active.
- Les badges Dégradations et Losses suivent maintenant le profil GAN jusqu'à 100 % : la
  console du moteur coupe les longues lignes et insère le nom du fichier source
  (`sr_model.py:737`) au milieu du message, ce qui cachait les lignes de rampe aux badges.
- Un `train.py` mémorisé depuis un autre dossier moteur (`~/USRS` au lieu du moteur actif)
  est ré-aligné sur le moteur actif avant le lancement : un run ne démarre plus sur un
  moteur plus ancien (archs manquantes comme NERVE : `No object named 'nerve'`) et
  n'écrit plus son expérience dans le mauvais dossier. Les chemins personnalisés ne sont
  pas touchés.
- **Un seul gradient aberrant ne tue plus l'entraînement** (traiNNer-redux) : le moteur ne
  vérifiait que la loss. Sans AMP, un gradient infini (loss finie) passait le clipping
  puis l'optimiseur et mettait tous les poids, EMA comprise, à NaN, avant l'arrêt du run
  10 itérations plus tard. Les mises à jour du générateur et des deux discriminateurs sont
  maintenant sautées sur gradient non fini (avertissement dans le log) ; le run ne
  s'arrête qu'après 10 gradients générateur aberrants d'affilée.
- **Fermer l'application pendant un entraînement demande maintenant confirmation** :
  *Sauvegarder et quitter* (arrêt propre, modèle et état sauvés, puis fermeture),
  *Laisser tourner* (la fenêtre se ferme, l'entraînement continue en arrière-plan) ou
  *Annuler*. Avant, fermer laissait le processus d'entraînement tourner sans fenêtre,
  impossible à arrêter depuis l'application.
- Les infobulles MixUp / CutMix / ResizeMix / CutBlur retrouvent leur description après
  le passage d'une arch vidéo à une arch image (elles restaient sur « désactivé en mode
  séquence »).
- Charger une config YAML sans son fichier `.uss.json` ne remet plus à leurs valeurs par
  défaut les plages des dégradations custom (bits posterize, niveaux banding, échelle
  aliasing…), et ne transforme plus les options `true`/`false` du générateur et du
  discriminateur en `1`/`0`.
- Corrigés : plantage du Motion Blur, intensité du Sharpen sans effet, plantage de
  l'aperçu live des dégradations.
- **Banding corrigé** : la formule ne gardait que 3 à 4 niveaux par canal (`levels**(1/3)`),
  « banding 32 » donnait une forte postérisation au lieu de bandes douces. Le réglage
  correspond maintenant à de vrais niveaux par canal. Quantize depth et posterize
  arrondissent au niveau le plus proche (ils tronquaient, ce qui assombrissait l'image).
- **Nouvelle dégradation Smooth banding** (onglet Custom) : bandes uniquement sur les
  aplats (dégradés, ciels, ombrages), lignes et textures intactes. Option lumière
  linéaire pour que les tons sombres bandent comme sur un vrai encodage. Aussi dans
  l'aperçu live et les dégradations par phase.
- **Modes de resize gardés depuis le YAML** : l'app écrivait toujours 4 modes de resize à
  25 % chacun, `nearest-exact` compris, quel que soit la config. Nearest à 25 % donne
  des lignes en escalier (effet tirette). Les modes et probabilités de la config sont
  maintenant gardés ; par défaut bilinear / bicubic / lanczos (0.3 / 0.4 / 0.3).
  `poisson_scale_range` (sans widget) est aussi gardé.
- **Options `true`/`false` d'architecture remises par défaut au chargement** : un `false`
  YAML (par exemple `unshuffle_mod: false`, `use_checkpoint: true`) ne correspondait à
  aucun élément de la liste, la valeur par défaut restait et était resauvée. Toutes les
  options booléennes d'architecture sont maintenant chargées correctement.
- **Contrôle du patch pour LineDistiller** : LineDistiller (traiNNer-redux) plante dès la
  première itération quand Patch Size × scale n'est pas un multiple de 16 (par exemple
  LQ 60 en 2x : `The size of tensor a (15) must match the size of tensor b (14)`). La
  sauvegarde de la config avertit maintenant et propose les tailles valides les plus
  proches, que LineDistiller soit actif dans le profil PSNR ou dans le profil GAN.
  L'avertissement d'alignement FDAT-RD ne propose plus une taille refusée par LineDistiller.
- **Dégradations custom conservées au rechargement** : screentone, dithering, halo (wtp),
  niveaux de couleur, saturation, décalage de pixels et les autres dégradations « Custom
  3/4/5 » n'étaient pas relues à l'ouverture d'un .yml dans l'onglet Config. Le
  réenregistrer les désactivait sans prévenir.
- **Estimation de durée : GAN différé et Real-CUGAN fast** : avec un démarrage GAN différé
  (gan_start_iter, ou gan_auto_start), l'estimation additionne maintenant une phase PSNR
  sans discriminateurs ni pertes propres au GAN, puis la phase GAN. Elle comptait le GAN
  sur tout l'entraînement. Real-CUGAN `fast` (entrée pixel-unshuffle) compte maintenant un
  quart du travail du générateur. Run Real-CUGAN fast 2x : 2 j 20 h annoncés, maintenant
  1 j 4 h (mesuré 4,1 it/s en phase PSNR, 1,32 it/s en phase GAN ; estimé 3,9 / 1,1 it/s).
- **Nouvelles archs : WSR et DUAL (Rift73)** — entraînement (traiNNer-Redux) et inférence
  (Quick Upscale, runner séquence, convertisseur ONNX) :
  - **WSR** : tronc résiduel large + flux de contexte LR/4, attention Halo sur `wsr` et
    `wsr_compact`. Presets `wsr_mobile`, `wsr_superultracompact`, `wsr_ultracompact`,
    `wsr_compact`, `wsr`. Convolutif, export TensorRT simple.
  - **DUAL** : transformer double contexte (fenêtres de 16 px + attention de voisinage
    32/64). Presets `dual_light` à `dual_xl`. Lourd : ses chemins rapides demandent BF16 +
    Triton, absents sur Pascal (GTX 10xx) ; sur une 1080 Ti, `dual_light` s'entraîne
    ~9 fois plus lentement que NERVE.
  - Échelles 1x / 2x / 4x. Le preset et l'échelle sont relus depuis les poids.
  - Licences : WSR MIT (+ Apache-2.0 pour son Conv3XC issu de SPAN), DUAL MIT.
  - Estimations de vitesse / VRAM calées sur de vrais entraînements de 150 itérations (GTX 1080 Ti) des 11 presets.
- **Correctif : les modèles SPANF3 se chargent dans Quick Upscale** (subprocess normal et
  persistant). spandrel ne connaît pas SPANF3 (l'arch d'AnimeJaNai V3.1) : un modèle
  entraîné en `spanf3` échouait avec « architecture non reconnue ». Largeur, profondeur,
  échelle et `unshuffle_mod` sont maintenant relus depuis les poids.
- **Correctif : les poids pré-entraînés du 2e discriminateur sont conservés**
  (traiNNer-Redux). Une config avec `pretrain_network_d2` (fine-tune qui reprend les deux
  discriminateurs) perdait ce chemin quand on la ré-enregistrait depuis l'onglet
  Configuration : D2 repartait de poids aléatoires. Nouveau champ « Pretrain D2 » dans
  Reprise / Pretrain, relu et écrit dans la config (écrit seulement si le 2e
  discriminateur est activé).

### 6. Outils
- **Extraction de frames** :
  - frame N, toutes les X s ou toutes les frames ;
  - PNG/TIFF 8/16 bits, JPEG, BMP ;
  - aperçu, Stop, ETA, notification.
- **Color Fix guidé (aniCC, Bendel)** — correction de couleur apprise pour les dérives
  non uniformes, téléchargée au premier usage.
- **Convertisseur de modèle** :
  - fonctionne dans la version portable ;
  - ONNX à taille fixe pour les archs à fenêtres d'attention ;
  - vérification automatique des ONNX « dynamiques » ;
  - export des modèles vidéo.
- **Anti-crénelage NNEDI3 natif** (sans VapourSynth) dans Post Processing. Undistort a
  maintenant son propre réglage de précision.
- Nouvelle page **Importer une config** : importe une config Real-ESRGAN / SwinIR /
  RCAN / HAT / BasicSR.
- Visionneuse de validation : rechargement auto et curseur avant/après. LQ et SR sont
  affichées à la même taille (LQ agrandie au plus proche voisin, vrais pixels) ; la
  molette zoome sous le curseur et le clic molette maintenu déplace les deux images
  ensemble.
- **Page Pipeline PSNR → GAN** : la phase 2 part automatiquement du modèle final de la
  phase 1 (Redux) et les deux phases peuvent être mises en file. Nouveau mode : enchaîner
  deux fichiers d'options complets (une copie `.chained` de la phase 2 est écrite, ton
  fichier n'est pas modifié).
- Comparateur : charge automatiquement le dernier upscale.
- Info Modèle : bouton « → SafeTensors ». Les fichiers `.safetensors` affichent
  maintenant leur architecture.
- Quick Upscale :
  - taille de tuile automatique selon la VRAM libre ;
  - durée affichée en mode image unique.
- File d'attente : passe par le vrai pipeline d'entraînement (suivi, historique).
- Assistant : bandeau d'aptitude du GPU, ETA en direct, validation des réponses.
- **Analyse IA de l'entraînement** : liste des providers rafraîchie (GitHub Models a
  fermé en amont), modèles NVIDIA NIM / OpenRouter / OpenAI / Gemini / xAI / DeepSeek /
  Anthropic mis à jour et vérifiés en direct.

### 7. Version portable et installation
- Le zip portable est autonome :
  - il embarque son propre Python ;
  - les moteurs, PyTorch CUDA et toutes les dépendances s'installent depuis Paramètres ;
  - les fichiers de modèles optionnels se téléchargent au premier usage ;
  - **Git n'est plus nécessaire** : sans lui, le code du moteur est téléchargé en
    archive, au commit testé ;
  - **ffmpeg n'est plus nécessaire** : un ffmpeg portable s'installe avec le moteur.
- PyTorch est choisi selon le GPU : **torch 2.7.0 + CUDA 12.6** des GTX 10 (Pascal) aux
  RTX 40, **CUDA 12.8** pour les RTX 50 / Blackwell ; un driver NVIDIA trop ancien est
  signalé avant l'installation.
- Vérifié par une installation neuve sur un profil vierge, sans Git ni ffmpeg :
  installation, entraînement, upscale vidéo, conversion.
- traiNNer-redux suit `master`, **épinglé sur un commit testé**. NeoSR reste supporté ;
  son dépôt amont n'a pas bougé depuis mi-2025.
- **Paquets Python figés sur les versions testées** : chaque installation de dépendances
  utilise un fichier de contraintes généré depuis l'environnement validé pour cette
  version ; une installation neuve reçoit exactement les paquets testés, pas ceux que PyPI
  sert ce jour-là. PyTorch et ses bibliothèques CUDA viennent toujours du build adapté au GPU.
- Le Python embarqué n'emporte plus les caches et lanceurs de la machine de build
  (`__pycache__`, `Scripts/`) ; l'app n'utilise que `python -m pip` / `python -m virtualenv`.
- Corrigés :
  - l'installeur NeoSR annonçait réussi un clone cassé ;
  - l'extraction de frames avec ffmpeg ≥ 7 ;
  - SRFormer et les autres archs « extra » de spandrel étaient refusées dans la version
    portable ;
  - plusieurs fichiers côté moteur manquaient dans le build.
  - **les modèles ONNX tournaient sur le CPU** : le dernier onnxruntime-gpu (1.27+) est
    compilé pour CUDA 13, incompatible avec le CUDA 12 de PyTorch et les GPU Pascal.
    Épinglé sur la dernière version CUDA 12 (1.26), l'inférence ONNX repasse sur le GPU ;
    les messages illisibles d'onnxruntime ne polluent plus le log. Installations
    existantes : Paramètres → réinstaller les dépendances.

### 8. Audit fonctionnel complet (chaque fonction, chaque réglage)
Trois audits statiques de toute l'app (Quick Upscale, outils, réglages / post-traitement
/ publication : 147 constats) ont été vérifiés un par un, corrigés, puis testés sur des
données et des entraînements réels. Principales corrections :
- **Entraînement** :
  - **Le x3 avec dégradations OTF plantait sur les deux moteurs.** La taille de découpe
    n'était jamais multiple de 3 (traiNNer-redux gt_size + 32, NeoSR 512 fixe). Un
    patch moteur, appliqué à chaque lancement, l'arrondit au scale. Après correction :
    7/7 archs Redux et 4/4 NeoSR entraînent en x3.
  - NeoSR refusait BF16 sans AMP et Wavelet Guided sans GAN. BF16 est maintenant grisé
    tant que l'AMP n'est pas cochée ; Wavelet Guided sans GAN est refusé avec un message
    clair.
  - ESRGAN sous NeoSR ne fait que x1/x2/x4 : x3/x8 sont signalés avant le lancement.
  - Les estimations VRAM et vitesse sont recalibrées sur des entraînements réels : VRAM sur
    162 archs (elle était en moyenne 2,8× trop basse), vitesse sur 159 archs mesurées sur
    150 itérations (écart médian ×10 ; maintenant à ±10 % pour 145 sur 149). La ligne VRAM
    prévient quand la mémoire du GPU sera dépassée.
- **Quick Upscale** :
  - Un dossier de frames avec un modèle vidéo passe maintenant par le pipeline séquence
    (il était refusé). Un modèle vidéo `.pth` est reconnu dans la version portable.
  - La sortie 16 bits et la qualité JPEG/WEBP sont respectées partout : mode persistant,
    tous les runners d'inférence, après Temporal Fix / Undistort, skip frames. Tile, tile
    pad et AMP arrivent aux runners NeoSR.
  - **Temporal Fix / Undistort écrivaient les frames décalées d'une demi-fenêtre** (le
    1er résultat, centré sur la frame 4, remplaçait la frame 1). La frame k reçoit
    maintenant sa propre correction. Intensité 0 = vraie identité, et Undistort alimente
    Temporal Fix au lieu que les deux lisent la frame brute.
  - Undistort TMT : liens de téléchargement morts (fichiers renommés à la source) ; le
    nouveau modèle ONNX est pris en charge ; les tailles non multiples de 16 marchent.
    Tous les modes neuronaux (Temporal Fix PTH / CPU / ONNX, Undistort TMT PTH / CPU /
    ONNX) sont testés sur de vraies frames.
  - Color Fix : l'image de référence arrive à tous les chemins et garde ses 16 bits ; LAB
    respecte les canaux choisis ; une référence absente ou un échec guided est signalé
    au lieu d'être ignoré.
  - Sécurité des sorties : une sortie ne peut plus écraser une image source ;
    `a.jpg` + `a.png` n'écrivent plus le même fichier ; Stop agit pendant l'extraction ;
    pas de réassemblage après un Stop ; la vidéo est reconstruite avec les seules frames
    de ce traitement.
  - Un scale explicite que le modèle ne peut pas produire est annoncé avant le lancement
    (détection vérifiée sur 51 modèles). Les options sans effet (image seule, pipeline
    séquence) sont grisées avec la raison.
- **Outils** :
  - **Importer une config** réécrit sur le chargeur complet : les losses absentes du
    fichier sont désactivées ; poids, critères, scheduler, batch, patch LR et
    paramètres réseau sont importés ; l'architecture est convertie vers un nom que le
    moteur possède. Une config Real-ESRGAN x4plus a été importée puis entraînée.
  - **Publier** : les checkpoints `.pth` de NeoSR sont listés ; l'état d'entraînement
    correspond à l'itération exacte ; `resume/` reçoit les vrais G et D ; un `.pth` n'est
    plus renommé `.safetensors` ; nom et version sont vérifiés ; la fiche lit l'OTF, le
    pretrain et le patch LR dans la config.
  - **Créateur LMDB** : écrit le `meta_info.txt` exigé par les deux moteurs, signale les
    noms non ASCII et en double, et dimensionne la base d'après les données : la carte
    fixe de 1 To faisait créer à Windows un `data.mdb` de 1 To que NeoSR ne pouvait pas
    ouvrir. Entraînement réel validé sur NeoSR (paired, GT + LQ en `.lmdb`). Le
    traiNNer-redux épinglé ne sait pas lire le LMDB (ses chargeurs OTF et paired sont
    cassés pour ce format) : la configuration le dit au lieu de planter.
  - **Reprendre** : la bonne config est choisie d'après son contenu, les entraînements en
    cours sont détectés, et Reprendre active l'Auto-resume.
  - **Benchmark** : modes de précision exacts (plus de question cachée dans un
    terminal) ; Val et image test transmises ; Stop arrête tout l'arbre de processus ;
    chaque moteur / type peut lister ses tests.
  - **Générateur LQ** : vraie variation de teinte, vrai noyau disque et tramage par
    diffusion d'erreur. Les dossiers HQ / LQ / validation qui se chevauchent et les
    frames laissées par une extraction précédente sont refusés. Le Train garde au moins
    un clip.
  - **Comparateur** : les chemins saisis se chargent ; la plus petite image est agrandie
    (le résultat SR était réduit).
- **Post Processing** :
  - Vraie sortie 16 bits ; réglages Temporal Fix / Undistort conservés et
    réinitialisables.
  - Edge Cleanup supprime vraiment les pixels isolés clairs et sombres (le pixel
    lui-même était hors du masque des bords).
  - Le statut final signale les filtres non appliqués.
- **Réglages** :
  - Les installeurs échouent clairement s'il manque des dépendances ou PyTorch.
  - Mettre à jour traiNNer-redux revient au commit testé.
  - « Réinstaller toutes les deps » réinstalle vraiment.
  - La galerie ouvre sa vraie URL.

### Limites connues
- L'anti-crénelage EEDI3 demande encore VapourSynth (NNEDI3 est natif).
- MoTSR est 2x uniquement. MambaIR ne s'exporte pas en ONNX.
- Second discriminateur : traiNNer-redux uniquement.
- Le kernel rapide de MambaIR demande torch 2.7.0+cu126 (l'installation moteur par
  défaut). Avec un autre torch (RTX 50 / cu128, vieux pilotes / cu118), MambaIR utilise le
  scan pur PyTorch : sur une carte de 11 Go, LQ 32 / batch 1 à ~0,01 it/s. Quand un entraînement demande plus de VRAM que
  la carte, Windows peut déborder en mémoire partagée : il rampe alors au lieu de s'arrêter
  (vu avec GRL-B et Swin2SR-L 4x en batch 2).
- Datasets LMDB : NeoSR uniquement.
- RTX 50 / RTX PRO 6000 (Blackwell) : l'entraînement utilise les kernels PyTorch natifs
  (cu128). L'inférence ONNX (onnxruntime-gpu 1.26, CUDA 12.8) compte sur le pilote pour
  compiler le code GPU de la carte au premier usage ; pas encore vérifié sur une carte
  Blackwell. Le backend TensorRT n'est utilisé que si TensorRT (10.8+ pour Blackwell) est
  installé sur le système.
