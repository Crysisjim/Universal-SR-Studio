"""Small, targeted patches to UPSTREAM traiNNer-redux/neosr engine files.

Distinct from custom_archs/custom_engine/custom_neosr_archs (whole NEW files the
official engine doesn't ship — pure additions, safe to just copy in). These are
real bugs found in files the official engine DOES ship — a plain `git pull` or a
fresh re-clone of the engine silently reverts them, because there's nothing in the
official repo tracking our fix. Without this module, each fix only lives in
whoever's session memory found it, which is exactly how we lost track of them once
already (see Changelog Dev, 2026-09-03 incident: USRS-EXPERIMENTAL deleted by a
runaway agent, fixes had to be reapplied from memory on re-clone).

Registered patches are applied automatically and idempotently, right alongside the
custom-file injection, at every training start (runner.py::_inject_custom_engine_files)
and before every Quick Upscale run (quick_upscale.py::ensure_custom_archs) — so they
self-heal across engine re-clones/updates instead of silently disappearing.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

# Each patch: engine ("redux" or "neosr"), path (relative to that engine's own root
# folder — traiNNer-redux/ or neosr/ — NOT the traiNNer/ or neosr/ subfolder), the
# exact old/new text (matched literally, once), and a short description for logs.
ENGINE_PATCHES: list[dict] = [
    {
        "engine": "neosr",
        "path": "neosr/archs/krgn_arch.py",
        "old": (
            '        std = torch.Tensor(rgb_std).to("cuda")\n'
            '        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1).to(\n'
            '            "cuda"\n'
            '        )\n'
            '        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean).to("cuda")\n'
        ),
        "new": (
            '        std = torch.tensor(rgb_std, dtype=torch.float32)\n'
            '        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)\n'
            '        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean, dtype=torch.float32)\n'
        ),
        "description": (
            'KRGN MeanShift hardcoded .to("cuda") on tensors while torch.eye(3) is '
            "never moved -> RuntimeError: two devices, cuda:0 and cpu, at every "
            "construction (even on a CUDA machine). Let the module's own .to(device) "
            "call (done by the training/inference framework) handle placement."
        ),
    },
    {
        "engine": "redux",
        "path": "train.py",
        "old": (
            '    if opt.fast_matmul:\n'
            '        torch.set_float32_matmul_precision("high")\n'
            '        torch.backends.cuda.matmul.fp32_precision = "tf32"\n'
        ),
        "new": (
            '    if opt.fast_matmul:\n'
            '        torch.set_float32_matmul_precision("high")\n'
            '        # PyTorch >=2.7 exposes the new fp32_precision API; older versions only\n'
            '        # have the deprecated allow_tf32 flags. Try the new API first and fall\n'
            '        # back so this doesn\'t crash with AttributeError: Unknown attribute\n'
            '        # fp32_precision on PyTorch versions that don\'t implement it yet.\n'
            '        try:\n'
            '            torch.backends.cuda.matmul.fp32_precision = "tf32"\n'
            '            torch.backends.cudnn.conv.fp32_precision = "tf32"\n'
            '        except AttributeError:\n'
            '            torch.backends.cuda.matmul.allow_tf32 = True\n'
            '            torch.backends.cudnn.allow_tf32 = True\n'
        ),
        "description": (
            "train.py hardcoded the new-only torch.backends.cuda.matmul.fp32_precision "
            "API -> AttributeError: Unknown attribute fp32_precision on PyTorch versions "
            "that don't implement it yet. Try new API, fall back to allow_tf32."
        ),
    },
    {
        "engine": "redux",
        "path": "traiNNer/models/base_model.py",
        "old": (
            '        for i, scheduler in enumerate(self.schedulers):\n'
            '            if not self.optimizers_skipped[i]:\n'
            '                scheduler.step()\n'
        ),
        "new": (
            '        for i, scheduler in enumerate(self.schedulers):\n'
            '            if not self.optimizers_skipped[i]:\n'
            '                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):\n'
            '                    # ReduceLROnPlateau.step() requires a metric argument, unlike\n'
            '                    # every other LRScheduler subclass which takes none. Feed it\n'
            '                    # the running total training loss (falls back to a no-op-safe\n'
            '                    # value if unavailable this early) so plateau scheduling\n'
            '                    # doesn\'t crash with:\n'
            '                    # TypeError: step() missing 1 required positional argument: \'metrics\'\n'
            '                    metric = self.log_dict.get("l_g_total")\n'
            '                    if metric is None:\n'
            '                        metric = self.log_dict.get("l_total")\n'
            '                    if metric is not None:\n'
            '                        scheduler.step(metric)\n'
            '                else:\n'
            '                    scheduler.step()\n'
        ),
        "description": (
            "base_model.py::update_learning_rate called scheduler.step() generically for "
            "all scheduler types -> TypeError: step() missing 1 required positional "
            "argument: 'metrics' when using REDUCELRONPLATEAU. Branch and pass the "
            "running training loss as the metric for ReduceLROnPlateau."
        ),
    },
    {
        "engine": "neosr",
        "path": "neosr/models/base.py",
        "old": (
            '        elif optim_type in {"SOAP_SF", "soap_sf"}:\n'
            '            optimizer = soap_sf(params, lr, **kwargs)\n'
            '        else:\n'
            '            logger = get_root_logger()\n'
            '            msg = f"{tc.red}Optimizer {optim_type} is not supported yet.{tc.end}"\n'
            '            logger.error(msg)\n'
            '            sys.exit(1)\n'
            '\n'
            '        return cast("Optimizer", optimizer)\n'
        ),
        "new": (
            '        elif optim_type in {"SOAP_SF", "soap_sf"}:\n'
            '            optimizer = soap_sf(params, lr, **kwargs)\n'
            '        else:\n'
            '            # Accept "standalone" schedule-free optimizer names too, i.e. the\n'
            '            # naming convention used by the `schedulefree` PyPI package itself\n'
            '            # (AdamWScheduleFree, AdanScheduleFree, SOAPScheduleFree, ...)\n'
            '            # rather than only our own `_SF`-suffixed names. Normalize by\n'
            '            # stripping separators/case so "AdamWScheduleFree",\n'
            '            # "adamw_schedule_free" and "AdamW_SF" all resolve to the same\n'
            '            # bundled implementation instead of hitting the\n'
            '            # "not supported yet" -> sys.exit(1) fallback below.\n'
            '            normalized = optim_type.lower().replace("_", "").replace("-", "")\n'
            '            sf_alias_map = {\n'
            '                "adamwschedulefree": adamw_sf,\n'
            '                "adamschedulefree": adamw_sf,\n'
            '                "adanschedulefree": adan_sf,\n'
            '                "soapschedulefree": soap_sf,\n'
            '            }\n'
            '            if normalized.endswith("schedulefree") and normalized in sf_alias_map:\n'
            '                optimizer = sf_alias_map[normalized](params, lr, **kwargs)\n'
            '            else:\n'
            '                logger = get_root_logger()\n'
            '                msg = f"{tc.red}Optimizer {optim_type} is not supported yet.{tc.end}"\n'
            '                logger.error(msg)\n'
            '                sys.exit(1)\n'
            '\n'
            '        return cast("Optimizer", optimizer)\n'
        ),
        "description": (
            "base.py::get_optimizer only mapped our own _SF-suffixed names -> "
            "standalone schedule_free optimizer names (AdamWScheduleFree, etc., the "
            "`schedulefree` package's own naming convention) hit the unsupported-optimizer "
            "sys.exit(1) fallback. Normalize and alias them onto the bundled _SF impls."
        ),
    },
    {
        "engine": "redux",
        "path": "traiNNer/losses/ldl_loss.py",
        "old": (
            '        elif self.criterion_type == "charbonnier":\n'
            '            self.criterion = charbonnier_loss\n'
        ),
        "new": (
            '        elif self.criterion_type == "charbonnier":\n'
            '            self.criterion = charbonnier_loss\n'
            '        elif self.criterion_type == "huber":\n'
            '            self.criterion = torch.nn.HuberLoss()\n'
            '        else:\n'
            '            raise ValueError(\n'
            '                f"LDLLoss: unsupported criterion \'{criterion}\'. "\n'
            '                "Use \'l1\', \'l2\', \'charbonnier\', or \'huber\'."\n'
            '            )\n'
        ),
        "description": (
            "LDLLoss only supports l1/l2/charbonnier criteria upstream -> "
            "self.criterion is never assigned when criterion='huber' (a criterion "
            "the app UI offers for LDL loss) -> AttributeError at forward() time. "
            "Add a HuberLoss branch (migrated from the older ad-hoc "
            "runner.py::_patch_ldl_loss_huber patcher into this registry)."
        ),
    },
    # --- 2-discriminator support (network_d2/gan_opt2/optim_d2) ---
    {
        "engine": "redux",
        "path": 'traiNNer/utils/redux_options.py',
        "old": '    pretrain_network_d: Annotated[\n        str | None,\n        Meta(\n            description="Path to the pretrain model for the discriminator. `pth` and `safetensors` formats are supported."\n        ),\n    ] = None\n    param_key_d: str | None = None\n    strict_load_d: Annotated[\n        bool,\n        Meta(\n            description="Whether to load the pretrain model for the discriminator in strict mode. It should be enabled in most cases."\n        ),\n    ] = True\n',
        "new": '    pretrain_network_d: Annotated[\n        str | None,\n        Meta(\n            description="Path to the pretrain model for the discriminator. `pth` and `safetensors` formats are supported."\n        ),\n    ] = None\n    param_key_d: str | None = None\n    strict_load_d: Annotated[\n        bool,\n        Meta(\n            description="Whether to load the pretrain model for the discriminator in strict mode. It should be enabled in most cases."\n        ),\n    ] = True\n    pretrain_network_d2: Annotated[\n        str | None,\n        Meta(\n            description="Path to the pretrain model for the optional second discriminator (network_d2). `pth` and `safetensors` formats are supported."\n        ),\n    ] = None\n    param_key_d2: str | None = None\n    strict_load_d2: Annotated[\n        bool,\n        Meta(\n            description="Whether to load the pretrain model for the second discriminator in strict mode. It should be enabled in most cases."\n        ),\n    ] = True\n',
        "description": (
            "2-discriminator support: PathOptions needs pretrain_network_d2/param_key_d2/strict_load_d2 for the optional second discriminator's pretrained-weight loading."
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/utils/redux_options.py',
        "old": '    adaptive_d_ema_decay: float = 0.999\n    adaptive_d_threshold: float = 1.02\n',
        "new": '    adaptive_d_ema_decay: float = 0.999\n    adaptive_d_threshold: float = 1.02\n    # Design note (2-discriminator support): when network_d2 is also defined,\n    # adaptive_d applies independently to each discriminator (its own EMA of\n    # l_g_gan / l_g_gan2 and its own pause decision). This is a more correct\n    # design in theory -- it lets D1 pause while D2 keeps training if only D1\n    # is dominating -- but it has NOT been empirically validated against the\n    # simpler "pause both together" alternative; no A/B training run was done\n    # as part of this change. Treat it as a reasonable default, not a proven\n    # improvement.\n',
        "description": (
            '2-discriminator support: document the independent-per-discriminator adaptive_d design choice (not empirically validated vs a shared-pause alternative).'
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/utils/redux_options.py',
        "old": '    optim_d: Annotated[\n        dict[str, Any] | None,\n        Meta(description="The optimizer to use for the discriminator model."),\n    ] = None\n',
        "new": '    optim_d: Annotated[\n        dict[str, Any] | None,\n        Meta(description="The optimizer to use for the discriminator model."),\n    ] = None\n    optim_d2: Annotated[\n        dict[str, Any] | None,\n        Meta(\n            description="The optimizer to use for the optional second discriminator (network_d2). Can use a different type/lr than optim_d."\n        ),\n    ] = None\n',
        "description": (
            "2-discriminator support: TrainOptions.optim_d2, the second discriminator's own optimizer config (can differ from optim_d, e.g. different LR)."
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/utils/redux_options.py',
        "old": '    gan_opt: dict[str, Any] | None = None\n',
        "new": '    gan_opt: dict[str, Any] | None = None\n    gan_opt2: Annotated[\n        dict[str, Any] | None,\n        Meta(\n            description="GAN loss options for the optional second discriminator (network_d2). Has its own loss_weight, applied to l_g_gan2 (the generator loss vs net_d2), independent of gan_opt\'s weight for the first discriminator. Only used when network_d2 is defined."\n        ),\n    ] = None\n',
        "description": (
            "2-discriminator support: TrainOptions.gan_opt2, the second discriminator's GAN loss (own loss_weight -> relative D1/D2 weighting is just the ratio of gan_opt.loss_weight / gan_opt2.loss_weight, no separate combination-weight field needed)."
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/utils/redux_options.py',
        "old": '    network_d: Annotated[\n        dict[str, Any] | None,\n        Meta(description="The options for the discriminator model."),\n    ] = None\n',
        "new": '    network_d: Annotated[\n        dict[str, Any] | None,\n        Meta(description="The options for the discriminator model."),\n    ] = None\n    network_d2: Annotated[\n        dict[str, Any] | None,\n        Meta(\n            description="The options for an optional second discriminator, trained simultaneously alongside network_d with its own loss (gan_opt2) and optimizer (optim_d2). Requires network_d to also be defined. Useful for combining discriminators from complementary domains (e.g. a spatial discriminator + a spectral discriminator, as recommended by the DualFormer paper) rather than using a single discriminator."\n        ),\n    ] = None\n',
        "description": (
            "2-discriminator support: ReduxOptions.network_d2, the optional second discriminator's architecture config."
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/models/base_model.py',
        "old": '        self.l_g_gan_ema = torch.tensor(0.0, device=self.device)\n        self.adaptive_d = False\n',
        "new": '        self.l_g_gan_ema = torch.tensor(0.0, device=self.device)\n        self.l_g_gan_ema2 = torch.tensor(0.0, device=self.device)\n        self.adaptive_d = False\n',
        "description": (
            "2-discriminator support: BaseModel needs a default l_g_gan_ema2 (second discriminator's own adaptive-D EMA) alongside l_g_gan_ema."
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/models/base_model.py',
        "old": '        self.net_d = None\n        self.net_ae = None\n',
        "new": '        self.net_d = None\n        self.net_d2 = None\n        self.net_ae = None\n',
        "description": (
            '2-discriminator support: BaseModel needs a default net_d2 = None alongside net_d.'
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/models/sr_model.py',
        "old": '            self.net_d = None\n            if self.has_gan:\n                if self.opt.train.optim_d is None:\n                    raise ValueError(\n                        "GAN loss requires discriminator optimizer (optim_d). Define optim_d or disable GAN loss."\n                    )\n                if self.opt.network_d is None:\n                    raise ValueError(\n                        "GAN loss requires discriminator network (network_d). Define network_d or disable GAN loss."\n                    )\n                else:\n                    self.net_d = build_network(self.opt.network_d)\n                    # load pretrained models\n                    if self.opt.path.pretrain_network_d is not None:\n                        self.load_network(\n                            self.net_d,\n                            self.opt.path.pretrain_network_d,\n                            self.opt.path.strict_load_d,\n                            self.opt.path.param_key_d,\n                        )\n                    self.net_d = self.model_to_device(self.net_d)\n\n            self.losses = {}\n\n            self.ema_decay = 0\n            self.net_g_ema: EMA | None = None\n\n            self.optimizer_g: Optimizer | None = None\n            self.optimizer_d: Optimizer | None = None\n\n            self.init_training_settings()\n',
        "new": '            self.net_d = None\n            if self.has_gan:\n                if self.opt.train.optim_d is None:\n                    raise ValueError(\n                        "GAN loss requires discriminator optimizer (optim_d). Define optim_d or disable GAN loss."\n                    )\n                if self.opt.network_d is None:\n                    raise ValueError(\n                        "GAN loss requires discriminator network (network_d). Define network_d or disable GAN loss."\n                    )\n                else:\n                    self.net_d = build_network(self.opt.network_d)\n                    # load pretrained models\n                    if self.opt.path.pretrain_network_d is not None:\n                        self.load_network(\n                            self.net_d,\n                            self.opt.path.pretrain_network_d,\n                            self.opt.path.strict_load_d,\n                            self.opt.path.param_key_d,\n                        )\n                    self.net_d = self.model_to_device(self.net_d)\n\n            # Optional second discriminator (network_d2). Generic 2-discriminator\n            # support: any pair among the registered discriminators can be trained\n            # simultaneously, each with its own loss (gan_opt2) and optimizer\n            # (optim_d2), combined into the generator loss as a weighted sum (each\n            # discriminator keeps its own gan_opt(.2).loss_weight, so the "relative\n            # weight" between D1 and D2 is just the ratio of those two existing\n            # fields -- no separate combination-weight field was needed).\n            self.has_gan2 = False\n            gan_opt2 = self.opt.train.gan_opt2\n            if gan_opt2 and gan_opt2.get("loss_weight", 0) != 0:\n                self.has_gan2 = True\n\n            self.net_d2 = None\n            if self.has_gan2:\n                if self.opt.network_d is None:\n                    raise ValueError(\n                        "network_d2 requires network_d to also be defined (network_d2 is a *second*, additional discriminator)."\n                    )\n                if self.opt.train.optim_d2 is None:\n                    raise ValueError(\n                        "GAN loss for the second discriminator requires optim_d2. Define optim_d2 or disable gan_opt2."\n                    )\n                if self.opt.network_d2 is None:\n                    raise ValueError(\n                        "gan_opt2 requires discriminator network (network_d2). Define network_d2 or disable gan_opt2."\n                    )\n                else:\n                    self.net_d2 = build_network(self.opt.network_d2)\n                    if self.opt.path.pretrain_network_d2 is not None:\n                        self.load_network(\n                            self.net_d2,\n                            self.opt.path.pretrain_network_d2,\n                            self.opt.path.strict_load_d2,\n                            self.opt.path.param_key_d2,\n                        )\n                    self.net_d2 = self.model_to_device(self.net_d2)\n\n            self.losses = {}\n\n            self.ema_decay = 0\n            self.net_g_ema: EMA | None = None\n\n            self.optimizer_g: Optimizer | None = None\n            self.optimizer_d: Optimizer | None = None\n            self.optimizer_d2: Optimizer | None = None\n\n            # Independent adaptive-D state for the second discriminator (see\n            # adaptive_d design note in redux_options.py::TrainOptions). D1\'s\n            # self.l_g_gan_ema lives in BaseModel.__init__.\n            self.l_g_gan_ema2 = torch.tensor(0.0, device=self.device)\n            self._d_optim_idx: int | None = None\n            self._d2_optim_idx: int | None = None\n\n            self.init_training_settings()\n',
        "description": (
            '2-discriminator support: construct optional net_d2 (mirrors net_d construction/pretrain-load/model_to_device), track its own optimizer slot and adaptive-D EMA state.'
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/models/sr_model.py',
        "old": '        self.net_g.train()\n        if self.net_d is not None:\n            self.net_d.train()\n\n        train_opt = self.opt.train\n        assert train_opt is not None\n\n        logger = get_root_logger()\n\n        enable_gradscaler = self.use_amp and not self.opt.amp_bf16\n\n        self.scaler_g = GradScaler(enabled=enable_gradscaler, device="cuda")\n        self.scaler_d = GradScaler(enabled=enable_gradscaler, device="cuda")\n',
        "new": '        self.net_g.train()\n        if self.net_d is not None:\n            self.net_d.train()\n        if self.net_d2 is not None:\n            self.net_d2.train()\n\n        train_opt = self.opt.train\n        assert train_opt is not None\n\n        logger = get_root_logger()\n\n        enable_gradscaler = self.use_amp and not self.opt.amp_bf16\n\n        self.scaler_g = GradScaler(enabled=enable_gradscaler, device="cuda")\n        self.scaler_d = GradScaler(enabled=enable_gradscaler, device="cuda")\n        self.scaler_d2 = GradScaler(enabled=enable_gradscaler, device="cuda")\n',
        "description": (
            "2-discriminator support: put net_d2 in train() mode and give it its own AMP GradScaler (scaler_d2), independent from net_d's scaler_d."
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/models/sr_model.py',
        "old": '        assert self.losses, "At least one loss must be defined."\n\n        if not self.has_gan:\n            # warn that discriminator network / optimizer won\'t be used if enabled\n            if self.opt.network_d is not None:\n                logger.warning(\n                    "Discriminator network (network_d) is defined but GAN loss is disabled. Discriminator network will have no effect."\n                )\n\n            if train_opt.optim_d is not None:\n                logger.warning(\n                    "Discriminator optimizer (optim_d) is defined but GAN loss is disabled. Discriminator optimizer will have no effect."\n                )\n',
        "new": '        # gan_opt2 is deliberately NOT part of train_opt.losses (unlike gan_opt):\n        # both discriminators use loss type "ganloss", which would collide under\n        # the same "l_g_gan" label if both went through the generic losses list.\n        # Build it directly here instead, under its own "l_g_gan2" label.\n        if self.has_gan2:\n            assert train_opt.gan_opt2 is not None\n            self.losses["l_g_gan2"] = build_loss(train_opt.gan_opt2).to(\n                self.device,\n                memory_format=self.memory_format,\n                non_blocking=True,\n            )  # pyright: ignore[reportCallIssue]\n\n        assert self.losses, "At least one loss must be defined."\n\n        if not self.has_gan:\n            # warn that discriminator network / optimizer won\'t be used if enabled\n            if self.opt.network_d is not None:\n                logger.warning(\n                    "Discriminator network (network_d) is defined but GAN loss is disabled. Discriminator network will have no effect."\n                )\n\n            if train_opt.optim_d is not None:\n                logger.warning(\n                    "Discriminator optimizer (optim_d) is defined but GAN loss is disabled. Discriminator optimizer will have no effect."\n                )\n\n        if not self.has_gan2:\n            if self.opt.network_d2 is not None:\n                logger.warning(\n                    "Second discriminator network (network_d2) is defined but gan_opt2 is disabled/missing. network_d2 will have no effect."\n                )\n            if train_opt.optim_d2 is not None:\n                logger.warning(\n                    "Second discriminator optimizer (optim_d2) is defined but gan_opt2 is disabled/missing. optim_d2 will have no effect."\n                )\n',
        "description": (
            '2-discriminator support: build the l_g_gan2 loss (own dict entry, kept out of the generic train_opt.losses loop to avoid label collision with l_g_gan) and warn on an unused network_d2/optim_d2, mirroring the existing network_d/optim_d warnings.'
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/models/sr_model.py',
        "old": '        # optimizer d\n        if self.net_d is not None:\n            assert train_opt.optim_d is not None\n            self.optimizer_d = self.get_optimizer(\n                self.net_d.parameters(), train_opt.optim_d\n            )\n            self.optimizers.append(self.optimizer_d)\n            self.optimizers_skipped.append(False)\n            self.optimizers_schedule_free.append(\n                "SCHEDULEFREE" in train_opt.optim_d["type"].upper()\n            )\n',
        "new": '        # optimizer d\n        if self.net_d is not None:\n            assert train_opt.optim_d is not None\n            self.optimizer_d = self.get_optimizer(\n                self.net_d.parameters(), train_opt.optim_d\n            )\n            self._d_optim_idx = len(self.optimizers)\n            self.optimizers.append(self.optimizer_d)\n            self.optimizers_skipped.append(False)\n            self.optimizers_schedule_free.append(\n                "SCHEDULEFREE" in train_opt.optim_d["type"].upper()\n            )\n\n        # optimizer d2 (optional second discriminator)\n        if self.net_d2 is not None:\n            assert train_opt.optim_d2 is not None\n            self.optimizer_d2 = self.get_optimizer(\n                self.net_d2.parameters(), train_opt.optim_d2\n            )\n            self._d2_optim_idx = len(self.optimizers)\n            self.optimizers.append(self.optimizer_d2)\n            self.optimizers_skipped.append(False)\n            self.optimizers_schedule_free.append(\n                "SCHEDULEFREE" in train_opt.optim_d2["type"].upper()\n            )\n',
        "description": (
            "2-discriminator support: track the exact list index each discriminator's optimizer lands at (self._d_optim_idx / self._d2_optim_idx) instead of relying on hardcoded [1]/[-1] indices, which broke once a 3rd optimizer (D2) could be appended after D1."
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/models/sr_model.py',
        "old": '        skip_d_update = False\n\n        # optimize net_d\n        if self.net_d is not None:\n            for p in self.net_d.parameters():\n                p.requires_grad = False\n',
        "new": '        skip_d_update = False\n        skip_d2_update = False\n\n        # optimize net_d\n        if self.net_d is not None:\n            for p in self.net_d.parameters():\n                p.requires_grad = False\n        if self.net_d2 is not None:\n            for p in self.net_d2.parameters():\n                p.requires_grad = False\n',
        "description": (
            "2-discriminator support: freeze net_d2's params during the generator forward pass, same as net_d."
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/models/sr_model.py',
        "old": '                    if label == "l_g_gan":\n                        assert self.net_d is not None\n                        fake_g_pred = self.net_d(self.output)\n                        l_g_loss = loss(fake_g_pred, True, is_disc=False)\n\n                        if self.adaptive_d:\n                            l_g_gan_ema = (\n                                self.adaptive_d_ema_decay * self.l_g_gan_ema\n                                + (1 - self.adaptive_d_ema_decay) * l_g_loss.detach()\n                            )\n\n                            if (\n                                l_g_gan_ema\n                                > self.l_g_gan_ema * self.adaptive_d_threshold\n                            ):\n                                skip_d_update = True\n                                self.optimizers_skipped[1] = True\n                                # print(current_iter, "skip_d_update")\n\n                            self.l_g_gan_ema = l_g_gan_ema\n\n                    elif label == "l_g_ldl":\n',
        "new": '                    if label == "l_g_gan":\n                        assert self.net_d is not None\n                        fake_g_pred = self.net_d(self.output)\n                        l_g_loss = loss(fake_g_pred, True, is_disc=False)\n\n                        if self.adaptive_d:\n                            l_g_gan_ema = (\n                                self.adaptive_d_ema_decay * self.l_g_gan_ema\n                                + (1 - self.adaptive_d_ema_decay) * l_g_loss.detach()\n                            )\n\n                            if (\n                                l_g_gan_ema\n                                > self.l_g_gan_ema * self.adaptive_d_threshold\n                            ):\n                                skip_d_update = True\n                                assert self._d_optim_idx is not None\n                                self.optimizers_skipped[self._d_optim_idx] = True\n                                # print(current_iter, "skip_d_update")\n\n                            self.l_g_gan_ema = l_g_gan_ema\n\n                    elif label == "l_g_gan2":\n                        # Second discriminator\'s generator-facing loss. adaptive_d\n                        # (when enabled) is independent per-discriminator here: D1\n                        # and D2 each keep their own EMA/pause decision, so one\n                        # dominating discriminator doesn\'t pause the other. See\n                        # design note on TrainOptions.adaptive_d in redux_options.py.\n                        assert self.net_d2 is not None\n                        fake_g_pred2 = self.net_d2(self.output)\n                        l_g_loss = loss(fake_g_pred2, True, is_disc=False)\n\n                        if self.adaptive_d:\n                            l_g_gan_ema2 = (\n                                self.adaptive_d_ema_decay * self.l_g_gan_ema2\n                                + (1 - self.adaptive_d_ema_decay) * l_g_loss.detach()\n                            )\n\n                            if (\n                                l_g_gan_ema2\n                                > self.l_g_gan_ema2 * self.adaptive_d_threshold\n                            ):\n                                skip_d2_update = True\n                                assert self._d2_optim_idx is not None\n                                self.optimizers_skipped[self._d2_optim_idx] = True\n\n                            self.l_g_gan_ema2 = l_g_gan_ema2\n\n                    elif label == "l_g_ldl":\n',
        "description": (
            "2-discriminator support: generator loss vs net_d2 (l_g_gan2), with its own independent adaptive-D EMA/pause tracking. Also switches D1's adaptive-D pause flag from the hardcoded optimizers_skipped[1] to the tracked self._d_optim_idx."
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/models/sr_model.py',
        "old": '                if self.grad_clip:\n                    clip_grad_norm_(self.net_d.parameters(), self.grad_clip_max_norm)\n                scale_before = self.scaler_d.get_scale()\n                self.scaler_d.step(self.optimizer_d)\n                self.scaler_d.update()\n                scale_after = self.scaler_d.get_scale()\n                loss_dict["scale_d"] = scale_after\n                self.optimizers_skipped[-1] = scale_after < scale_before\n                self.optimizer_d.zero_grad()\n\n        for key, value in loss_dict.items():\n',
        "new": '                if self.grad_clip:\n                    clip_grad_norm_(self.net_d.parameters(), self.grad_clip_max_norm)\n                scale_before = self.scaler_d.get_scale()\n                self.scaler_d.step(self.optimizer_d)\n                self.scaler_d.update()\n                scale_after = self.scaler_d.get_scale()\n                loss_dict["scale_d"] = scale_after\n                assert self._d_optim_idx is not None\n                self.optimizers_skipped[self._d_optim_idx] = scale_after < scale_before\n                self.optimizer_d.zero_grad()\n\n        cri_gan2 = self.losses.get("l_g_gan2")\n\n        if (\n            self.net_d2 is not None\n            and cri_gan2 is not None\n            and self.optimizer_d2 is not None\n            and not skip_d2_update\n        ):\n            # optimize net_d2 (second discriminator, independent from net_d)\n            for p in self.net_d2.parameters():\n                p.requires_grad = True\n\n            with torch.autocast(\n                device_type=self.device.type,\n                dtype=self.amp_dtype,\n                enabled=self.use_amp,\n            ):\n                real_d2_pred = self.net_d2(\n                    self._real_hr if self._real_hr is not None else self.gt\n                )\n                l_d2_real = cri_gan2(real_d2_pred, True, is_disc=True)\n                loss_dict["l_d2_real"] = l_d2_real\n                loss_dict["out_d2_real"] = torch.mean(real_d2_pred.detach())\n\n                fake_d2_pred = self.net_d2(self.output.detach())\n                l_d2_fake = cri_gan2(fake_d2_pred, False, is_disc=True)\n                loss_dict["l_d2_fake"] = l_d2_fake\n                loss_dict["out_d2_fake"] = torch.mean(fake_d2_pred.detach())\n\n            self.scaler_d2.scale(\n                (l_d2_real + l_d2_fake) / self.accum_iters\n            ).backward()\n\n            if apply_gradient:\n                self.scaler_d2.unscale_(self.optimizer_d2)\n                grad_norm_d2 = torch.linalg.vector_norm(\n                    torch.stack(\n                        [\n                            torch.linalg.vector_norm(p.grad, 2)\n                            for p in self.net_d2.parameters()\n                            if p.grad is not None\n                        ]\n                    )\n                ).detach()\n\n                loss_dict["grad_norm_d2"] = grad_norm_d2\n\n                if self.grad_clip:\n                    clip_grad_norm_(self.net_d2.parameters(), self.grad_clip_max_norm)\n                scale_before2 = self.scaler_d2.get_scale()\n                self.scaler_d2.step(self.optimizer_d2)\n                self.scaler_d2.update()\n                scale_after2 = self.scaler_d2.get_scale()\n                loss_dict["scale_d2"] = scale_after2\n                assert self._d2_optim_idx is not None\n                self.optimizers_skipped[self._d2_optim_idx] = (\n                    scale_after2 < scale_before2\n                )\n                self.optimizer_d2.zero_grad()\n\n        for key, value in loss_dict.items():\n',
        "applied_marker": 'cri_gan2 = self.losses.get("l_g_gan2")',
        "description": (
            "2-discriminator support: train net_d2 on its own real/fake loss (l_d2_real/l_d2_fake) with its own AMP scaler and optimizer, independent of net_d's update (and independently skippable via skip_d2_update). Also fixes a pre-existing bug where optimizers_skipped[-1] silently pointed at whichever optimizer was appended last -- harmless with only G+D1, but would have written D2's skip flag onto D1's slot once a 3rd optimizer existed. Now uses the tracked self._d_optim_idx."
        ),
    },
    {
        "engine": "redux",
        "path": 'traiNNer/models/sr_model.py',
        "old": '        if self.net_d is not None:\n            self.save_network(\n                self.net_d, "net_d", self.opt.path.resume_models, current_iter, "params"\n            )\n\n        self.save_training_state(epoch, current_iter)\n',
        "new": '        if self.net_d is not None:\n            self.save_network(\n                self.net_d, "net_d", self.opt.path.resume_models, current_iter, "params"\n            )\n\n        if self.net_d2 is not None:\n            self.save_network(\n                self.net_d2,\n                "net_d2",\n                self.opt.path.resume_models,\n                current_iter,\n                "params",\n            )\n\n        self.save_training_state(epoch, current_iter)\n',
        "description": (
            "2-discriminator support: persist net_d2's weights on checkpoint save, mirroring net_d."
        ),
    },
    {
        # Verbatim port of the generic hook added with TFDAT v2 (Kim2091/traiNNer-redux-1,
        # branch tfdat-v2, commit f0e10ca): generators may expose get_aux_losses(iter)
        # — TFDAT v2 uses it for its annealed optical-flow supervision. Archs without
        # the method are unaffected.
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": (
            "                        loss_dict[label] = weighted_l_g_loss\n"
            "\n"
            "                if not l_g_total.isfinite():\n"
        ),
        "new": (
            "                        loss_dict[label] = weighted_l_g_loss\n"
            "\n"
            "                # Auxiliary losses exposed by the generator itself (e.g. TFDAT\n"
            "                # optical flow supervision). The arch computes and weights them\n"
            "                # during its forward pass; anything it returns here is already\n"
            "                # weighted and annealed.\n"
            "                get_aux_losses = getattr(\n"
            "                    self.get_bare_model(self.net_g), \"get_aux_losses\", None\n"
            "                )\n"
            "                if callable(get_aux_losses):\n"
            "                    for aux_label, aux_loss in get_aux_losses(current_iter).items():\n"
            "                        l_g_total += aux_loss / self.accum_iters\n"
            "                        loss_dict[aux_label] = aux_loss\n"
            "\n"
            "                if not l_g_total.isfinite():\n"
        ),
        "description": (
            "Generator auxiliary losses (get_aux_losses) — TFDAT v2 annealed flow supervision."
        ),
    },
    # --- Delayed GAN start: fixed iteration (gan_start_iter) or automatic on a
    # validation PSNR plateau (gan_auto_*), optional LR schedule restart ---
    {
        "engine": "redux",
        "path": "traiNNer/utils/redux_options.py",
        "old": (
            '    # as part of this change. Treat it as a reasonable default, not a proven\n'
            '    # improvement.\n'
        ),
        "new": (
            '    # as part of this change. Treat it as a reasonable default, not a proven\n'
            '    # improvement.\n'
            '    gan_start_iter: Annotated[\n'
            '        int,\n'
            '        Meta(\n'
            '            description="Iteration at which the GAN starts. Before it, the generator trains on its other losses only: no GAN loss (l_g_gan / l_g_gan2) and no discriminator update. 0 = GAN active from the start. With gan_auto_start, this is the latest start (used when no PSNR plateau is detected before it)."\n'
            '        ),\n'
            '    ] = 0\n'
            '    gan_auto_start: Annotated[\n'
            '        bool,\n'
            '        Meta(\n'
            '            description="Start the GAN automatically when the validation PSNR stops improving. Needs validation with a PSNR metric; without it, the GAN starts at gan_start_iter (or gan_auto_min_iter)."\n'
            '        ),\n'
            '    ] = False\n'
            '    gan_auto_patience: Annotated[\n'
            '        int,\n'
            '        Meta(\n'
            '            description="gan_auto_start: number of validations without a PSNR gain larger than gan_auto_min_delta before the GAN starts."\n'
            '        ),\n'
            '    ] = 3\n'
            '    gan_auto_min_delta: Annotated[\n'
            '        float,\n'
            '        Meta(\n'
            '            description="gan_auto_start: smallest PSNR gain (dB) counted as an improvement."\n'
            '        ),\n'
            '    ] = 0.05\n'
            '    gan_auto_min_iter: Annotated[\n'
            '        int,\n'
            '        Meta(description="gan_auto_start: the GAN never starts before this iteration."),\n'
            '    ] = 20000\n'
            '    gan_restart_lr: Annotated[\n'
            '        bool,\n'
            '        Meta(\n'
            '            description="When a delayed GAN starts, restart every LR scheduler from the initial LR: scheduler milestones then count from the GAN start, like a separate GAN training run."\n'
            '        ),\n'
            '    ] = False\n'
        ),
        "description": 'delayed GAN: TrainOptions fields gan_start_iter, gan_auto_*, gan_restart_lr.',
    },
    {
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": (
            '        self.adaptive_d_threshold = train_opt.adaptive_d_threshold\n'
        ),
        "new": (
            '        self.adaptive_d_threshold = train_opt.adaptive_d_threshold\n'
            '        self.gan_start_iter = train_opt.gan_start_iter\n'
            '        self.gan_auto_start = train_opt.gan_auto_start\n'
            '        self.gan_auto_patience = train_opt.gan_auto_patience\n'
            '        self.gan_auto_min_delta = train_opt.gan_auto_min_delta\n'
            '        self.gan_auto_min_iter = train_opt.gan_auto_min_iter\n'
            '        self.gan_restart_lr = train_opt.gan_restart_lr\n'
            '        # Iteration at which the GAN started (None = not yet) and the PSNR plateau\n'
            '        # tracker of gan_auto_start. Saved in the training state for resume.\n'
            '        self.gan_started_iter: int | None = None\n'
            '        self.gan_auto_best: float | None = None\n'
            '        self.gan_auto_bad = 0\n'
            '        if self.net_d is not None and self.gan_auto_start:\n'
            '            val_opt = self.opt.val\n'
            '            if val_opt is None or not val_opt.val_enabled or not val_opt.metrics_enabled:\n'
            '                self.gan_auto_start = False\n'
            '                if self.gan_start_iter <= 0:\n'
            '                    self.gan_start_iter = self.gan_auto_min_iter\n'
            '                logger.warning(\n'
            '                    "GAN auto start needs validation with metrics: GAN starts at iteration %d instead.",\n'
            '                    self.gan_start_iter,\n'
            '                )\n'
            '            else:\n'
            '                logger.info(\n'
            '                    "GAN auto start: when the validation PSNR gains less than %.3f dB over %d validation(s), not before iteration %d%s.",\n'
            '                    self.gan_auto_min_delta,\n'
            '                    self.gan_auto_patience,\n'
            '                    self.gan_auto_min_iter,\n'
            '                    f", at the latest at iteration {self.gan_start_iter}"\n'
            '                    if self.gan_start_iter > 0\n'
            '                    else "",\n'
            '                )\n'
            '        if self.net_d is not None and not self.gan_auto_start and self.gan_start_iter > 0:\n'
            '            logger.info(\n'
            '                "GAN delayed: GAN losses and discriminator updates start at iteration %d.",\n'
            '                self.gan_start_iter,\n'
            '            )\n'
        ),
        "description": 'delayed GAN: read the options in init_training_settings and log the mode.',
    },
    {
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": (
            '    def optimize_parameters(\n'
            '        self, current_iter: int, current_accum_iter: int, apply_gradient: bool\n'
            '    ) -> None:\n'
        ),
        "new": (
            '    def _gan_active(self, current_iter: int) -> bool:\n'
            '        """Delayed GAN start (gan_start_iter / gan_auto_start)."""\n'
            '        if self.gan_started_iter is None:\n'
            '            if not self.gan_auto_start and self.gan_start_iter <= 0:\n'
            '                return True  # standard behaviour: GAN from the start\n'
            '            if self.gan_start_iter > 0 and current_iter >= self.gan_start_iter:\n'
            '                self._start_gan(\n'
            '                    current_iter,\n'
            '                    "latest start reached, no PSNR plateau detected"\n'
            '                    if self.gan_auto_start\n'
            '                    else "gan_start_iter",\n'
            '                )\n'
            '        return self.gan_started_iter is not None and current_iter >= self.gan_started_iter\n'
            '\n'
            '    def _start_gan(self, start_iter: int, reason: str) -> None:\n'
            '        self.gan_started_iter = start_iter\n'
            '        logger = get_root_logger()\n'
            '        logger.info("GAN starts at iteration %d (%s).", start_iter, reason)\n'
            '        if self.gan_restart_lr:\n'
            '            for scheduler in self.schedulers:\n'
            '                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):\n'
            '                    continue\n'
            '                for group in scheduler.optimizer.param_groups:\n'
            '                    group["lr"] = group.get("initial_lr", group["lr"])\n'
            '                scheduler.last_epoch = 0\n'
            '                scheduler._last_lr = [  # pyright: ignore[reportAttributeAccessIssue]\n'
            '                    group["lr"] for group in scheduler.optimizer.param_groups\n'
            '                ]\n'
            '            logger.info(\n'
            '                "LR schedule restarted: scheduler milestones now count from the GAN start."\n'
            '            )\n'
            '\n'
            '    def _gan_auto_check(self, current_iter: int) -> None:\n'
            '        """gan_auto_start: PSNR plateau check, called after each validation."""\n'
            '        if (\n'
            '            not self.gan_auto_start\n'
            '            or self.gan_started_iter is not None\n'
            '            or self.net_d is None\n'
            '        ):\n'
            '            return\n'
            '        logger = get_root_logger()\n'
            '        psnr_key = next((k for k in self.metric_results if "psnr" in k.lower()), None)\n'
            '        if psnr_key is None:\n'
            '            self.gan_auto_start = False\n'
            '            if self.gan_start_iter <= 0:\n'
            '                self.gan_start_iter = max(self.gan_auto_min_iter, current_iter + 1)\n'
            '            logger.warning(\n'
            '                "GAN auto start: no PSNR metric in validation, GAN starts at iteration %d instead.",\n'
            '                self.gan_start_iter,\n'
            '            )\n'
            '            return\n'
            '        psnr = float(self.metric_results[psnr_key])\n'
            '        if self.gan_auto_best is None or psnr > self.gan_auto_best + self.gan_auto_min_delta:\n'
            '            self.gan_auto_best = psnr\n'
            '            self.gan_auto_bad = 0\n'
            '        else:\n'
            '            self.gan_auto_bad += 1\n'
            '        logger.info(\n'
            '            "GAN auto start: PSNR %.4f dB (reference %.4f dB), %d/%d validation(s) without gain.",\n'
            '            psnr,\n'
            '            self.gan_auto_best,\n'
            '            self.gan_auto_bad,\n'
            '            self.gan_auto_patience,\n'
            '        )\n'
            '        if (\n'
            '            self.gan_auto_bad >= self.gan_auto_patience\n'
            '            and current_iter >= self.gan_auto_min_iter\n'
            '        ):\n'
            '            self._start_gan(\n'
            '                current_iter + 1, f"PSNR plateau at {self.gan_auto_best:.4f} dB"\n'
            '            )\n'
            '\n'
            '    def optimize_parameters(\n'
            '        self, current_iter: int, current_accum_iter: int, apply_gradient: bool\n'
            '    ) -> None:\n'
        ),
        "description": 'delayed GAN: _gan_active / _start_gan (optional LR restart) / _gan_auto_check (PSNR plateau).',
    },
    {
        # Runs after the 2-discriminator patch above (overrides skip_d/skip_d2_update);
        # anchored outside that patch's text so both stay recognisable.
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": (
            '        n_samples = self.gt.shape[0]\n'
            '        self.loss_samples += n_samples\n'
        ),
        "new": (
            '        # Before the GAN start: no GAN loss for G, no D update. The D optimizers\n'
            '        # are flagged as skipped so their LR scheduler does not step either (same\n'
            '        # behaviour as an adaptive_d pause).\n'
            '        gan_active = self._gan_active(current_iter)\n'
            '        skip_d_update = not gan_active\n'
            '        skip_d2_update = not gan_active\n'
            '        if not gan_active:\n'
            '            for idx in (self._d_optim_idx, self._d2_optim_idx):\n'
            '                if idx is not None:\n'
            '                    self.optimizers_skipped[idx] = True\n'
            '\n'
            '        n_samples = self.gt.shape[0]\n'
            '        self.loss_samples += n_samples\n'
        ),
        "description": 'delayed GAN: skip both discriminator updates before the GAN start.',
    },
    {
        # Anchored below the `target = ...` line: runner.py's SpanC multi-scale
        # patcher rewrites that line (self.gt -> gt_for_loss), so it must stay
        # right after the `for` line.
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": (
            '                    if loss.loss_weight < 0:\n'
        ),
        "new": (
            '                    if not gan_active and label in ("l_g_gan", "l_g_gan2"):\n'
            '                        continue\n'
            '\n'
            '                    if loss.loss_weight < 0:\n'
        ),
        "description": 'delayed GAN: leave out the generator GAN losses before the GAN start.',
    },
    {
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": (
            '            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)\n'
        ),
        "new": (
            '            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)\n'
            '            self._gan_auto_check(current_iter)\n'
        ),
        "description": 'delayed GAN: PSNR plateau check after each validation (gan_auto_start).',
    },
    {
        "engine": "redux",
        "path": "traiNNer/models/base_model.py",
        "old": (
            '            if self.net_g_ema is not None:\n'
            '                state["ema_step"] = self.net_g_ema.step\n'
        ),
        "new": (
            '            if getattr(self, "gan_started_iter", None) is not None or getattr(\n'
            '                self, "gan_auto_best", None\n'
            '            ) is not None:\n'
            '                state["gan_start"] = {  # pyright: ignore[reportGeneralTypeIssues]\n'
            '                    "started_iter": self.gan_started_iter,  # pyright: ignore[reportAttributeAccessIssue]\n'
            '                    "auto_best": self.gan_auto_best,  # pyright: ignore[reportAttributeAccessIssue]\n'
            '                    "auto_bad": self.gan_auto_bad,  # pyright: ignore[reportAttributeAccessIssue]\n'
            '                }\n'
            '            if self.net_g_ema is not None:\n'
            '                state["ema_step"] = self.net_g_ema.step\n'
        ),
        "description": 'delayed GAN: save the GAN start state in the training state.',
    },
    {
        "engine": "redux",
        "path": "traiNNer/models/base_model.py",
        "old": (
            '        current_iter = resume_state["iter"]\n'
        ),
        "new": (
            '        current_iter = resume_state["iter"]\n'
            '        gan_state = resume_state.get("gan_start")\n'
            '        if gan_state is not None and hasattr(self, "gan_started_iter"):\n'
            '            self.gan_started_iter = gan_state["started_iter"]\n'
            '            self.gan_auto_best = gan_state["auto_best"]\n'
            '            self.gan_auto_bad = gan_state["auto_bad"]\n'
            '        elif 0 < getattr(self, "gan_start_iter", 0) <= current_iter:\n'
            '            # State saved without the GAN start record: the GAN is already running,\n'
            '            # so mark it started (no LR restart).\n'
            '            self.gan_started_iter = self.gan_start_iter\n'
        ),
        "description": 'delayed GAN: restore the GAN start state on resume.',
    },
    {
        # GRFormer: torch.clamp(cuda tensor, max=<cpu tensor>) fails on GPU ("Expected
        # all tensors to be on the same device ... cuda:0 and cpu"). A float bound is the
        # same value on any device.
        "engine": "neosr",
        "path": "neosr/archs/grformer_arch.py",
        "old": (
            '            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))\n'
        ),
        "new": (
            '            self.logit_scale, max=4.605170185988092  # log(1 / 0.01)\n'
        ),
        "description": "GRFormer: clamp bound as a float (CPU tensor bound failed on GPU).",
    },
    # --- Phase degradations (PSNR profile -> GAN profile, linear ramp after the
    # GAN start). Inert without _usr_studio_deg_phase.json (written by runner.py).
    {
        "engine": "redux",
        "path": "traiNNer/models/realesrgan_model.py",
        "old": (
            '    @torch.no_grad()\n'
            '    def feed_data(self, data: DataFeed) -> None:\n'
        ),
        "new": (
            '    def _uss_deg_phase(self) -> None:\n'
            '        """USRS phase degradations: blend the OTF options from the PSNR profile to\n'
            '        the GAN profile (_usr_studio_deg_phase.json, written by the studio at launch).\n'
            '        Before the GAN start: PSNR profile. After it: linear ramp over `ramp` iterations."""\n'
            '        if not hasattr(self, "_uss_phase"):\n'
            '            self._uss_phase = None\n'
            '            self._uss_phase_alpha = 0.0\n'
            '            self._uss_phase_step = -1\n'
            '            try:\n'
            '                import json as _uss_j\n'
            '\n'
            '                _uss_fp = osp.join(\n'
            '                    osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),\n'
            '                    "_usr_studio_deg_phase.json",\n'
            '                )\n'
            '                if osp.exists(_uss_fp):\n'
            '                    with open(_uss_fp, encoding="utf-8") as f:\n'
            '                        self._uss_phase = _uss_j.load(f)\n'
            '            except Exception:\n'
            '                self._uss_phase = None\n'
            '        phase = self._uss_phase\n'
            '        if not phase:\n'
            '            return\n'
            '        it = getattr(self, "_uss_iter", 0)\n'
            '        start = getattr(self, "gan_started_iter", None)\n'
            '        if start is None and getattr(self, "net_d", None) is not None:\n'
            '            delayed = getattr(self, "gan_auto_start", False) or getattr(\n'
            '                self, "gan_start_iter", 0\n'
            '            ) > 0\n'
            '            if not delayed:\n'
            '                start = 0  # GAN active from the start\n'
            '        ramp = max(0, int(phase.get("ramp", 0)))\n'
            '        if start is None or it < start:\n'
            '            alpha = 0.0\n'
            '        elif ramp == 0:\n'
            '            alpha = 1.0\n'
            '        else:\n'
            '            alpha = min(1.0, (it - start) / ramp)\n'
            '        self._uss_phase_alpha = alpha\n'
            '        for key, (v1, v2) in phase.get("base", {}).items():\n'
            '            setattr(self.opt, key, self._uss_lerp(v1, v2, alpha))\n'
            '        step = int(alpha * 10)\n'
            '        if step != self._uss_phase_step:\n'
            '            self._uss_phase_step = step\n'
            '            get_root_logger().info(\n'
            '                "OTF degradations: %s (GAN profile %d%%).",\n'
            '                "PSNR profile" if alpha <= 0 else "GAN profile" if alpha >= 1 else "transition",\n'
            '                round(alpha * 100),\n'
            '            )\n'
            '\n'
            '    def _uss_phase_custom(self, opt: dict) -> dict:\n'
            '        """Phase blend for the custom degradations (sidecar dict of the custom hook)."""\n'
            '        phase = getattr(self, "_uss_phase", None)\n'
            '        if not phase or not phase.get("custom"):\n'
            '            return opt\n'
            '        alpha = getattr(self, "_uss_phase_alpha", 0.0)\n'
            '        opt = dict(opt)\n'
            '        for key, (v1, v2) in phase["custom"].items():\n'
            '            opt[key] = self._uss_lerp(v1, v2, alpha)\n'
            '        return opt\n'
            '\n'
            '    @staticmethod\n'
            '    def _uss_lerp(v1, v2, alpha):\n'
            '        if alpha <= 0:\n'
            '            return v1\n'
            '        if alpha >= 1:\n'
            '            return v2\n'
            '        if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)) and len(v1) == len(v2):\n'
            '            return [RealESRGANModel._uss_lerp(a, b, alpha) for a, b in zip(v1, v2)]\n'
            '        num = (int, float)\n'
            '        if isinstance(v1, num) and isinstance(v2, num) and not isinstance(v1, bool) and not isinstance(v2, bool):\n'
            '            v = v1 + (v2 - v1) * alpha\n'
            '            return int(round(v)) if isinstance(v1, int) and isinstance(v2, int) else v\n'
            '        return v2 if alpha >= 0.5 else v1\n'
            '\n'
            '    @torch.no_grad()\n'
            '    def feed_data(self, data: DataFeed) -> None:\n'
        ),
        "description": 'phase degradations: _uss_deg_phase / _uss_phase_custom / _uss_lerp on RealESRGANModel.',
    },
    {
        "engine": "redux",
        "path": "traiNNer/models/realesrgan_model.py",
        "old": (
            '            ori_h, ori_w = self.gt.size()[2:4]\n'
        ),
        "new": (
            '            ori_h, ori_w = self.gt.size()[2:4]\n'
            '            self._uss_deg_phase()\n'
        ),
        "description": 'phase degradations: blend the OTF options at each feed_data.',
    },
    {
        # Anchored on an upstream comment line: inserting inside another patch's
        # "new" text would make that patch re-apply at every launch.
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": (
            '        # assert self.optimizer_g is not None\n'
        ),
        "new": (
            '        # assert self.optimizer_g is not None\n'
            '        self._uss_iter = current_iter  # read by the phase degradations (feed_data)\n'
        ),
        "description": 'phase degradations: record the current iteration for feed_data.',
    },
    # --- Phase losses (PSNR weights -> GAN weights, linear ramp after the GAN start).
    # Inert without _usr_studio_loss_phase.json (written by runner.py). Every loss of
    # either profile is in the config (so it is built at start, no VRAM surprise at the
    # GAN start); the sidecar holds [PSNR weight, GAN weight] per loss type, and a loss
    # whose current weight is 0 is not computed.
    {
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": (
            '    def _apply_eco(self) -> None:\n'
        ),
        "new": (
            '    def _uss_phase_losses(self, current_iter: int):\n'
            '        """USRS phase losses: (label, loss) pairs to compute at this iteration, with\n'
            '        loss_weight blended from the PSNR profile to the GAN profile."""\n'
            '        if not hasattr(self, "_uss_lphase"):\n'
            '            self._uss_lphase = {}\n'
            '            self._uss_lphase_step = -1\n'
            '            try:\n'
            '                import json as _uss_j\n'
            '\n'
            '                _uss_fp = osp.join(\n'
            '                    osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),\n'
            '                    "_usr_studio_loss_phase.json",\n'
            '                )\n'
            '                if osp.exists(_uss_fp):\n'
            '                    with open(_uss_fp, encoding="utf-8") as f:\n'
            '                        _uss_d = _uss_j.load(f) or {}\n'
            '                    self._uss_lphase_ramp = max(0, int(_uss_d.get("ramp", 0)))\n'
            '                    for _t, (_w1, _w2) in (_uss_d.get("weights") or {}).items():\n'
            '                        self._uss_lphase[loss_type_to_label(_t)] = (float(_w1), float(_w2))\n'
            '            except Exception:\n'
            '                self._uss_lphase = {}\n'
            '        if not self._uss_lphase:\n'
            '            return list(self.losses.items())\n'
            '        start = getattr(self, "gan_started_iter", None)\n'
            '        if start is None and getattr(self, "net_d", None) is not None:\n'
            '            delayed = getattr(self, "gan_auto_start", False) or getattr(\n'
            '                self, "gan_start_iter", 0\n'
            '            ) > 0\n'
            '            if not delayed:\n'
            '                start = 0  # GAN active from the start\n'
            '        ramp = self._uss_lphase_ramp\n'
            '        if start is None or current_iter < start:\n'
            '            alpha = 0.0\n'
            '        elif ramp == 0:\n'
            '            alpha = 1.0\n'
            '        else:\n'
            '            alpha = min(1.0, (current_iter - start) / ramp)\n'
            '        out = []\n'
            '        for label, loss in self.losses.items():\n'
            '            ws = self._uss_lphase.get(label)\n'
            '            if ws is not None:\n'
            '                loss.loss_weight = ws[0] + (ws[1] - ws[0]) * alpha\n'
            '                if loss.loss_weight == 0:\n'
            '                    continue  # off in this phase: not computed\n'
            '            out.append((label, loss))\n'
            '        step = int(alpha * 10)\n'
            '        if step != self._uss_lphase_step:\n'
            '            self._uss_lphase_step = step\n'
            '            get_root_logger().info(\n'
            '                "Losses: %s (GAN profile %d%%).",\n'
            '                "PSNR profile" if alpha <= 0 else "GAN profile" if alpha >= 1 else "transition",\n'
            '                round(alpha * 100),\n'
            '            )\n'
            '        return out\n'
            '\n'
            '    def _apply_eco(self) -> None:\n'
        ),
        "description": 'phase losses: _uss_phase_losses on SRModel (blended loss weights).',
    },
    {
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": (
            '                for label, loss in self.losses.items():\n'
            '                    target = '
        ),
        "new": (
            '                for label, loss in self._uss_phase_losses(current_iter):\n'
            '                    target = '
        ),
        "description": 'phase losses: iterate over the losses of the current phase.',
    },
    # Non-finite gradient guard. The engine only checks the LOSS for NaN/Inf: without
    # AMP the GradScaler is a no-op, so one Inf gradient (finite loss) went through
    # clip_grad_norm_ (Inf/Inf = NaN) into optimizer.step() and turned every weight,
    # EMA included, into NaN (Nerve v2 at 85.7k, just after the GAN start). Skip the
    # step instead; stop after 10 consecutive bad generator gradients.
    {
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": "                    self.scaler_g.step(self.optimizer_g)\n",
        "new": (
            "                    if torch.isfinite(grad_norm_g):  # [USS] non-finite grad guard\n"
            "                        self.scaler_g.step(self.optimizer_g)\n"
            "                        self._uss_gnan = 0\n"
            "                    else:\n"
            "                        self._uss_gnan = getattr(self, '_uss_gnan', 0) + 1\n"
            "                        get_root_logger().warning(\n"
            "                            '[USS] Non-finite generator gradient (count: %d), update skipped',\n"
            "                            self._uss_gnan,\n"
            "                        )\n"
            "                        if self._uss_gnan > 10:\n"
            "                            raise RuntimeError(\n"
            "                                'Training failed: generator gradients stay NaN/Inf. '\n"
            "                                'Lower the learning rate (or the GAN / perceptual weights).'\n"
            "                            )\n"
        ),
        "description": "non-finite gradient guard: generator step skipped on NaN/Inf gradients.",
    },
    {
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": "                self.scaler_d.step(self.optimizer_d)\n",
        "new": (
            "                if torch.isfinite(grad_norm_d):  # [USS] non-finite grad guard\n"
            "                    self.scaler_d.step(self.optimizer_d)\n"
            "                else:\n"
            "                    get_root_logger().warning(\n"
            "                        '[USS] Non-finite discriminator gradient, update skipped'\n"
            "                    )\n"
        ),
        "description": "non-finite gradient guard: discriminator step skipped on NaN/Inf gradients.",
    },
    {
        "engine": "redux",
        "path": "traiNNer/models/sr_model.py",
        "old": "                self.scaler_d2.step(self.optimizer_d2)\n",
        "new": (
            "                if torch.isfinite(grad_norm_d2):  # [USS] non-finite grad guard\n"
            "                    self.scaler_d2.step(self.optimizer_d2)\n"
            "                else:\n"
            "                    get_root_logger().warning(\n"
            "                        '[USS] Non-finite discriminator 2 gradient, update skipped'\n"
            "                    )\n"
        ),
        "description": "non-finite gradient guard: discriminator 2 step skipped on NaN/Inf gradients.",
    },
    # MambaIR/MambaIRv2 EMA-unwrap fix (traiNNer-redux sr_model.py::infer_tiled /
    # infer_tiled_context_crop) intentionally NOT registered yet: the arch-name
    # dispatch branch it patches is itself part of a custom MambaIR integration that
    # isn't currently bundled in custom_archs/ (mambair_arch.py/mambairv2_arch.py
    # aren't shipped there) — nothing to patch in a fresh engine clone right now.
    # TODO: register once the MambaIR custom-arch bundle (including whatever adds the
    # MambaIR/MambaIRv2 branch to infer_tiled) is formalized. Original fix documented
    # in Changelog Dev for manual reapplication meanwhile: unwrap
    # `getattr(net, "ema_model", net)` before checking
    # `actual_net.__class__.__name__ in ("MambaIR", "MambaIRv2")`.
    {
        # ESCRealM (SDPA / Naive attention): the relative-position index table was built
        # with a Python triple loop, ~20 s per model build. Same tensor, vectorized.
        "engine": "redux",
        "path": "traiNNer/archs/escreal_arch.py",
        "old": '        idxs_window = []\n        for head in range(heads):\n            for h in range(window_size**2):\n                for w in range(window_size**2):\n                    q_h = h // window_size\n                    q_w = h % window_size\n                    k_h = w // window_size\n                    k_w = w % window_size\n                    rel_h = k_h - q_h + window_size - 1\n                    rel_w = k_w - q_w + window_size - 1\n                    rel_idx = rel_h * (2 * window_size - 1) + rel_w\n                    idxs_window.append((head, rel_idx))\n        idxs = torch.tensor(idxs_window, dtype=torch.long, requires_grad=False)\n        return idxs\n',
        "new": '        # USRS: vectorized (identical tensor). The Python triple loop took ~2 s per\n        # block at window 32 (4 heads x 32^4 entries), ~20 s per ESCRealM build.\n        n = window_size * window_size\n        pos = torch.arange(n)\n        q_h, q_w = (pos // window_size)[:, None], (pos % window_size)[:, None]\n        k_h, k_w = (pos // window_size)[None, :], (pos % window_size)[None, :]\n        rel = ((k_h - q_h + window_size - 1) * (2 * window_size - 1)\n               + (k_w - q_w + window_size - 1)).reshape(-1)\n        head = torch.arange(heads).repeat_interleave(n * n)\n        return torch.stack([head, rel.repeat(heads)], dim=1).to(torch.long)\n',
        "description": "ESCRealM: vectorized relative-position index table (20 s -> 0.2 s per build).",
    },
    {
        # NeoSR Soft Stop: its train loop only saves on KeyboardInterrupt (Ctrl+C). The app
        # stops a run with the USS_STOP_FILE sentinel + CTRL_BREAK, which NeoSR handled
        # neither: after 120 s the app killed it WITHOUT saving. Poll the sentinel once per
        # iteration and reuse the engine's own "Interrupted, saving latest models" path.
        "engine": "neosr",
        "path": "train.py",
        "old": (
            "                iter_timer.start()\n"
            "                train_data = prefetcher.next()  # type: ignore[reportPossiblyUnboundVariable]\n"
        ),
        "new": (
            "                iter_timer.start()\n"
            "                # [USS] Soft Stop: the app writes this file to ask for a save + stop\n"
            "                if __import__('os').environ.get('USS_STOP_FILE') and "
            "osp.exists(__import__('os').environ['USS_STOP_FILE']):\n"
            "                    raise KeyboardInterrupt\n"
            "                train_data = prefetcher.next()  # type: ignore[reportPossiblyUnboundVariable]\n"
        ),
        "description": "NeoSR Soft Stop: poll the app's stop file each iteration (it was killed unsaved).",
    },
    {
        # KRGN MeanShift, 2nd step: torch.Tensor(...) (legacy constructor) ignores the default
        # device NeoSR builds the network under, torch.eye() does not -> cuda:0 / cpu mix at
        # construction. torch.tensor(...) follows the default device like eye().
        "engine": "neosr",
        "path": "neosr/archs/krgn_arch.py",
        "old": (
            '        std = torch.Tensor(rgb_std)\n'
            '        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)\n'
            '        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)\n'
        ),
        "new": (
            '        std = torch.tensor(rgb_std, dtype=torch.float32)\n'
            '        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)\n'
            '        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean, dtype=torch.float32)\n'
        ),
        "description": "KRGN MeanShift: torch.tensor follows NeoSR's CUDA default device (CPU/CUDA mix).",
    },
    {
        # OTF crop size gt_size + 32 is never a multiple of 3 (gt_size = 3 x LQ): every
        # x3 OTF run stopped on "Scale mismatches. GT (224, 224) is not 3x ... (74, 74)".
        "engine": "redux",
        "path": "traiNNer/data/realesrgan_dataset.py",
        "old": "        crop_pad_size = self.opt.gt_size + 32\n",
        "new": (
            "        crop_pad_size = self.opt.gt_size + 32\n"
            "        # [USS] multiple of the scale (x3: 224 -> 225), else the LQ/GT check fails\n"
            "        _uss_sc = int(getattr(self.opt, 'scale', 1) or 1)\n"
            "        crop_pad_size = -(-crop_pad_size // _uss_sc) * _uss_sc\n"
        ),
        "description": "Redux OTF: crop/pad size rounded to a multiple of the scale (x3 crashed).",
    },
    {
        # NeoSR hard-codes a 512 crop: 512 / 3 is not an integer, every x3 OTF run stopped
        # on "Scale mismatches. GT (512, 512) is not 3x of LQ (170, 170)".
        "engine": "neosr",
        "path": "neosr/data/otf_dataset.py",
        "old": "        crop_pad_size = 512\n",
        "new": (
            "        crop_pad_size = 512\n"
            "        # [USS] multiple of the scale (x3: 512 -> 513), else the LQ/GT check fails\n"
            "        _uss_sc = int(self.opt.get('scale', 1) or 1)\n"
            "        crop_pad_size = -(-crop_pad_size // _uss_sc) * _uss_sc\n"
        ),
        "description": "NeoSR OTF: 512 crop rounded to a multiple of the scale (x3 crashed).",
    },
    {
        # USRS texture metrics (custom_engine/metrics/usrs_texture_metrics.py, injected
        # next to the official metrics): traiNNer/metrics/__init__.py imports each metric
        # module explicitly, so the new file must be imported there to register its
        # calculate_usrs_* functions. Guarded: a missing file never breaks the engine.
        "engine": "redux",
        "path": "traiNNer/metrics/__init__.py",
        "old": "from traiNNer.metrics.topiq import calculate_topiq\n",
        "new": (
            "from traiNNer.metrics.topiq import calculate_topiq\n"
            "\n"
            "try:  # [USS] texture / edge / flat MAE + HF corr / ratio validation metrics\n"
            "    from traiNNer.metrics import usrs_texture_metrics  # noqa: F401\n"
            "except ImportError:\n"
            "    pass\n"
        ),
        "description": "USRS texture metrics: register calculate_usrs_* in the metric registry.",
    },
    {
        # Found 2026-11-14 (FT2 resume at 11244): check_resume() only redirected G, G EMA
        # and D to the resume checkpoints, so a resumed 2-discriminator run reloaded D2 from
        # pretrain_network_d2 (the fine-tune start point). Lived only in the engine copy
        # until the 2026-11-26 fresh-clone audit.
        "engine": "redux",
        "path": "traiNNer/utils/misc.py",
        "old": (
            '                        f"Unable to resume, pretrain_network_d not found at path: {basepath}.{model_extensions[0]}"\n'
            "                    )\n"
            "\n"
            '        if opt.path.param_key_g == "params_ema":\n'
        ),
        "new": (
            '                        f"Unable to resume, pretrain_network_d not found at path: {basepath}.{model_extensions[0]}"\n'
            "                    )\n"
            "\n"
            "        # USRS patch: set second discriminator path (network_d2) on resume too.\n"
            "        # Without this, a resume reloaded net_d2 from pretrain_network_d2 (the\n"
            "        # fine-tune start point) instead of the resume_iter checkpoint.\n"
            "        if opt.network_d2 is not None and (\n"
            "            opt.path.ignore_resume_networks is None\n"
            '            or "network_d2" not in opt.path.ignore_resume_networks\n'
            "        ):\n"
            "            model_exists = False\n"
            '            basepath = ""\n'
            "            for ext in model_extensions:\n"
            "                for model_dir in model_dirs:\n"
            '                    basepath = osp.join(model_dir, f"net_d2_{resume_iter}")\n'
            '                    if osp.exists(f"{basepath}.{ext}"):\n'
            '                        opt.path.pretrain_network_d2 = f"{basepath}.{ext}"\n'
            "                        model_exists = True\n"
            "                        print(\n"
            '                            f"Set pretrain_network_d2 to {opt.path.pretrain_network_d2}",\n'
            "                        )\n"
            "            if not model_exists:\n"
            "                raise FileNotFoundError(\n"
            '                    f"Unable to resume, pretrain_network_d2 not found at path: {basepath}.{model_extensions[0]}"\n'
            "                )\n"
            "\n"
            '        if opt.path.param_key_g == "params_ema":\n'
        ),
        "description": "Resume: also point pretrain_network_d2 at net_d2_<resume_iter>.",
    },
]


def apply_engine_patches(
    engine_root: str, engine: str, log: Optional[Callable[[str], None]] = None
) -> None:
    """Apply every registered patch for `engine` ("redux" or "neosr") whose target
    file exists under `engine_root` (the traiNNer-redux/ or neosr/ folder itself).

    Idempotent: a patch already applied (new text present) is skipped silently.
    Never raises: if neither the old nor the new text is found (upstream file
    changed shape since the patch was written), logs a warning and moves on —
    a stale patch must never block training or inference.
    """
    for patch in ENGINE_PATCHES:
        if patch["engine"] != engine:
            continue
        target = os.path.join(engine_root, *patch["path"].split("/"))
        if not os.path.isfile(target):
            continue
        try:
            with open(target, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            if log:
                log(f"[PATCH] Lecture échouée {patch['path']} : {e}")
            continue
        if patch["new"] in content:
            continue  # already applied
        # A later patch may rewrite part of this one's new text (e.g. the gradient guard
        # inside the 2-discriminator block): a marker then proves it was applied.
        if patch.get("applied_marker") and patch["applied_marker"] in content:
            continue
        if patch["old"] not in content:
            if log:
                log(
                    f"[PATCH] {patch['path']} : motif attendu introuvable "
                    "(fichier upstream modifié ?) — patch ignoré, à vérifier manuellement."
                )
            continue
        try:
            content = content.replace(patch["old"], patch["new"], 1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            if log:
                log(f"[PATCH] {patch['path']} corrigé — {patch['description'][:70]}...")
        except Exception as e:
            if log:
                log(f"[PATCH] Échec application {patch['path']} : {e}")


def apply_all_known_engines(base_dir: str, log: Optional[Callable[[str], None]] = None) -> None:
    """Convenience wrapper: applies redux patches under <base_dir>/traiNNer-redux and
    neosr patches under <base_dir>/neosr, if those folders exist."""
    apply_engine_patches(os.path.join(base_dir, "traiNNer-redux"), "redux", log)
    apply_engine_patches(os.path.join(base_dir, "neosr"), "neosr", log)
