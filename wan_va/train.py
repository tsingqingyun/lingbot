# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import csv
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import get_config
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from .dataset import MultiLatentLeRobotDataset
from .dataset.lerobot_latent_dataset import get_robocasa_binarize_thresholds
import gc


def _git_metadata(repo_root: Path) -> dict:
    def _run_git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return None
        text = out.stdout.strip()
        return text or None

    return {
        "git_commit": _run_git("rev-parse", "HEAD"),
        "git_branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_short": _run_git("status", "--short"),
    }


def _jsonable_config_subset(config) -> dict:
    keys = [
        "config_name",
        "dataset_path",
        "task_names",
        "empty_emb_path",
        "save_root",
        "resume_from",
        "wan22_pretrained_model_name_or_path",
        "wan22_base_pretrained_model_name_or_path",
        "env_type",
        "frame_chunk_size",
        "action_per_frame",
        "action_dim",
        "used_action_channel_ids",
        "action_norm_method",
        "num_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
    ]
    out = {}
    for key in keys:
        value = getattr(config, key, None)
        if isinstance(value, Path):
            value = str(value)
        out[key] = value
    return out

def _pad_tensor_along_dim(x: torch.Tensor, target_len: int, dim: int, pad_value=0):
    cur_len = x.shape[dim]
    if cur_len == target_len:
        return x

    if cur_len > target_len:
        slices = [slice(None)] * x.ndim
        slices[dim] = slice(0, target_len)
        return x[tuple(slices)]

    pad_shape = list(x.shape)
    pad_shape[dim] = target_len - cur_len
    pad_tensor = torch.full(
        pad_shape,
        fill_value=pad_value,
        dtype=x.dtype,
        device=x.device,
    )
    return torch.cat([x, pad_tensor], dim=dim)


def _collate_pad_batch(batch):
    """
    处理变长 batch：
    - latents:      [C, F, H, W]   -> 按 F 维 pad
    - actions:      [C, F, N, 1]   -> 按 F 维 pad
    - actions_mask: [C, F, N, 1]   -> 按 F 维 pad，pad=False
    - text_emb:     [L, D]         -> 按 L 维 pad，并生成 text_mask
    """
    if len(batch) == 0:
        return {}

    # Fast path for bs=1: avoid variable-length padding logic entirely.
    # This keeps masks/training targets on the exact original timeline.
    if len(batch) == 1:
        sample = batch[0]
        out = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                out[k] = v.unsqueeze(0)
            else:
                out[k] = [v]

        if 'latents' in sample and 'latents_mask' not in out:
            f = int(sample['latents'].shape[1])
            out['latents_mask'] = torch.ones(
                (1, 1, f, 1, 1),
                dtype=torch.bool,
                device=sample['latents'].device,
            )
        if 'text_emb' in sample and 'text_mask' not in out:
            l = int(sample['text_emb'].shape[0])
            out['text_mask'] = torch.ones(
                (1, l),
                dtype=torch.bool,
                device=sample['text_emb'].device,
            )
        return out

    out = {}
    keys = batch[0].keys()

    max_latent_f = 0
    max_action_f = 0
    max_text_len = 0

    # 先统计这一批里的最大长度
    for sample in batch:
        if 'latents' in sample:
            max_latent_f = max(max_latent_f, int(sample['latents'].shape[1]))
        if 'actions' in sample:
            max_action_f = max(max_action_f, int(sample['actions'].shape[1]))
        if 'actions_mask' in sample:
            max_action_f = max(max_action_f, int(sample['actions_mask'].shape[1]))
        if 'text_emb' in sample:
            # text_emb: [L, D]
            max_text_len = max(max_text_len, int(sample['text_emb'].shape[0]))

    for k in keys:
        vals = [sample[k] for sample in batch]
        v0 = vals[0]

        if torch.is_tensor(v0):
            if k == 'latents':
                padded_vals = []
                latent_masks = []

                for v in vals:
                    cur_f = int(v.shape[1])

                    # latents_mask: [1, F, 1, 1]
                    m = torch.ones(
                        (1, cur_f, 1, 1),
                        dtype=torch.bool,
                        device=v.device,
                    )
                    m = _pad_tensor_along_dim(m, max_latent_f, dim=1, pad_value=False)

                    v = _pad_tensor_along_dim(v, max_latent_f, dim=1, pad_value=0)
                    padded_vals.append(v)
                    latent_masks.append(m)

                out[k] = torch.stack(padded_vals, dim=0)
                out['latents_mask'] = torch.stack(latent_masks, dim=0)

            elif k == 'actions':
                vals = [_pad_tensor_along_dim(v, max_action_f, dim=1, pad_value=0) for v in vals]
                out[k] = torch.stack(vals, dim=0)

            elif k == 'actions_mask':
                vals = [_pad_tensor_along_dim(v, max_action_f, dim=1, pad_value=False) for v in vals]
                out[k] = torch.stack(vals, dim=0)

            elif k == 'text_emb':
                padded_vals = []
                text_masks = []

                for v in vals:
                    cur_len = int(v.shape[0])

                    # text_mask: [L]
                    m = torch.ones(
                        (cur_len,),
                        dtype=torch.bool,
                        device=v.device,
                    )
                    m = _pad_tensor_along_dim(m, max_text_len, dim=0, pad_value=False)

                    v = _pad_tensor_along_dim(v, max_text_len, dim=0, pad_value=0)
                    padded_vals.append(v)
                    text_masks.append(m)

                out[k] = torch.stack(padded_vals, dim=0)         # [B, L, D]
                out['text_mask'] = torch.stack(text_masks, dim=0) # [B, L]

            else:
                same_shape = all(v.shape == v0.shape for v in vals)
                if not same_shape:
                    shapes = [tuple(v.shape) for v in vals]
                    raise RuntimeError(f"collate key={k} has inconsistent shapes: {shapes}")
                out[k] = torch.stack(vals, dim=0)

        else:
            out[k] = vals

    return out


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            api_key = os.environ.get("WANDB_API_KEY")
            if not api_key:
                logger.warning(
                    "enable_wandb=True but WANDB_API_KEY is not set; disable wandb logging for this run."
                )
                config.enable_wandb = False
            else:
                base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
                init_timeout = float(
                    os.environ.get(
                        "WANDB_INIT_TIMEOUT",
                        getattr(config, "wandb_init_timeout", 180),
                    )
                )
                try:
                    wandb.login(host=base_url, key=api_key)
                    self.wandb = wandb
                    # 403 upsertBucket: entity 必须与 key 有写权限的 team/user 一致。
                    # 优先级: config.wandb_entity -> WANDB_ENTITY -> WANDB_TEAM_NAME；皆空则不传 entity，用 key 默认 workspace。
                    entity = (
                        getattr(config, "wandb_entity", None)
                        or os.environ.get("WANDB_ENTITY")
                        or os.environ.get("WANDB_TEAM_NAME")
                    )
                    project = (
                        getattr(config, "wandb_project", None)
                        or os.environ.get("WANDB_PROJECT", "va_robotwin")
                    )
                    run_name = (
                        getattr(config, "wandb_run_name", None)
                        or os.environ.get("WANDB_RUN_NAME", "train")
                    )
                    mode = os.environ.get("WANDB_MODE", "online")
                    init_kw = dict(
                        project=project,
                        config=config,
                        mode=mode,
                        name=run_name,
                        settings=wandb.Settings(init_timeout=init_timeout),
                    )
                    if entity:
                        init_kw["entity"] = entity
                    self.wandb.init(**init_kw)
                    logger.info(
                        "WandB enabled: project=%r, entity=%s, mode=%r, init_timeout=%.1fs",
                        project,
                        entity if entity else "<default from API key>",
                        mode,
                        init_timeout,
                    )
                except Exception as e:
                    logger.warning(
                        "WandB init failed (%r). Disable wandb logging for this run.",
                        e,
                    )
                    config.enable_wandb = False
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.use_deepspeed = bool(getattr(config, "use_deepspeed", False))
        self.gradient_accumulation_steps = int(
            getattr(config, "gradient_accumulation_steps", 1)
        )
        self.enable_binary_action_aux = bool(
            getattr(config, "enable_binary_action_aux", False)
        )
        self.binary_action_aux_channels = [
            int(x) for x in getattr(config, "binary_action_aux_channels", [14, 29])
        ]
        self.binary_action_aux_weight = float(
            getattr(config, "binary_action_aux_weight", 0.0)
        )
        self.binary_action_aux_pos_weight = float(
            getattr(config, "binary_action_aux_pos_weight", 1.0)
        )
        self.binary_action_aux_focal_gamma = float(
            getattr(config, "binary_action_aux_focal_gamma", 2.0)
        )
        self.binary_action_aux_loss_type = str(
            getattr(config, "binary_action_aux_loss_type", "bce")
        ).lower()
        default_binary_thresholds = get_robocasa_binarize_thresholds()
        self.binary_action_aux_threshold = float(
            getattr(
                config,
                "binary_action_aux_threshold",
                default_binary_thresholds.get("gripper", 0.0),
            )
        )
        self.binary_action_aux_logit_scale = float(
            getattr(config, "binary_action_aux_logit_scale", 8.0)
        )
        self._last_binary_action_aux_loss = torch.tensor(0.0, device=self.device)

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device="cpu",
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(
            self.transformer,
            inner_checkpoint_min_layer=int(
                getattr(config, "ac_inner_checkpoint_min_layer", 10)
            ),
            checkpoint_attn2=bool(
                getattr(config, "ac_checkpoint_attn2", True)
            ),
        )

        logger.info("Setting up distributed training wrapper...")
        if self.use_deepspeed:
            try:
                import deepspeed
            except ImportError as e:
                raise RuntimeError(
                    "DeepSpeed is enabled but not installed. "
                    "Please run `pip install deepspeed`."
                ) from e
            if not torch.cuda.is_available():
                raise RuntimeError("DeepSpeed training requires CUDA.")

            ds_config_path = getattr(config, "deepspeed_config", None)
            if not ds_config_path:
                raise ValueError(
                    "DeepSpeed is enabled but `deepspeed_config` is not set."
                )
            ds_config_path = os.path.abspath(str(ds_config_path))
            if not os.path.exists(ds_config_path):
                raise FileNotFoundError(
                    f"DeepSpeed config file not found: {ds_config_path}"
                )
            with open(ds_config_path, "r") as f:
                ds_config = json.load(f)
            ds_config.setdefault("train_micro_batch_size_per_gpu", int(config.batch_size))
            ds_config.setdefault(
                "gradient_accumulation_steps",
                int(getattr(config, "gradient_accumulation_steps", 1)),
            )
            if int(ds_config["train_micro_batch_size_per_gpu"]) != int(config.batch_size):
                logger.warning(
                    "DeepSpeed micro batch size (%s) != config.batch_size (%s). "
                    "DataLoader uses config.batch_size.",
                    ds_config["train_micro_batch_size_per_gpu"],
                    config.batch_size,
                )
            optimizer_offload_device = (
                ds_config.get("zero_optimization", {})
                .get("offload_optimizer", {})
                .get("device", "")
            )
            use_fused_adamw = str(optimizer_offload_device).lower() != "cpu"
            self.transformer.to(device=self.device, dtype=self.dtype)
            self.transformer.train()
            self.transformer.requires_grad_(True)
            self.optimizer = torch.optim.AdamW(
                [p for p in self.transformer.parameters() if p.requires_grad],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=1e-8,
                weight_decay=config.weight_decay,
                fused=use_fused_adamw,
                foreach=False,
            )
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: warmup_constant_lambda(
                    step, warmup_steps=config.warmup_steps
                ),
            )
            self.transformer, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=self.transformer,
                model_parameters=[p for p in self.transformer.parameters() if p.requires_grad],
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                config=ds_config,
                dist_init_required=False,
            )
            ds_accum = getattr(self.transformer, "gradient_accumulation_steps", None)
            if ds_accum is not None:
                self.gradient_accumulation_steps = int(
                    ds_accum() if callable(ds_accum) else ds_accum
                )
            if config.rank == 0:
                logger.info(f"DeepSpeed enabled, config: {ds_config_path}")
                logger.info(
                    "DeepSpeed grad accumulation steps: %s",
                    self.gradient_accumulation_steps,
                )
        else:
            shard_fn = partial(shard_model, param_dtype=self.dtype)
            self.transformer = _configure_model(
                model=self.transformer,
                shard_fn=shard_fn,
                param_dtype=self.dtype,
                device=self.device,
                eval_mode=False,
            )
            self.transformer.train()
            self.transformer.requires_grad_(True)
            self.optimizer = torch.optim.AdamW(
                [p for p in self.transformer.parameters() if p.requires_grad],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=1e-8,
                weight_decay=config.weight_decay,
                fused=True,
                foreach=False,
            )
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset_full = MultiLatentLeRobotDataset(config=config)

        self.validation_split_ratio = float(getattr(config, "validation_split_ratio", 0.01))
        self.validation_interval = int(
            getattr(config, "validation_interval", getattr(config, "save_interval", 0))
        )
        self.validation_num_batches = max(
            1, int(getattr(config, "validation_num_batches", 8))
        )
        self.validation_split_seed = int(getattr(config, "validation_split_seed", 42))
        self.val_loader = None

        train_dataset = train_dataset_full
        val_dataset = None
        if (
            self.validation_interval > 0
            and self.validation_split_ratio > 0.0
            and len(train_dataset_full) > 1
        ):
            total_size = len(train_dataset_full)
            val_size = max(1, int(total_size * self.validation_split_ratio))
            train_size = total_size - val_size
            if train_size > 0:
                split_generator = torch.Generator().manual_seed(self.validation_split_seed)
                train_dataset, val_dataset = random_split(
                    train_dataset_full,
                    [train_size, val_size],
                    generator=split_generator,
                )
            else:
                if self.config.rank == 0:
                    logger.warning(
                        "Validation split disabled: train set would be empty (dataset size=%d, ratio=%.4f)",
                        total_size,
                        self.validation_split_ratio,
                    )

        if self.config.rank == 0:
            if val_dataset is not None:
                logger.info(
                    "Validation enabled: split_ratio=%.4f, train_size=%d, val_size=%d, interval=%d, max_batches=%d",
                    self.validation_split_ratio,
                    len(train_dataset),
                    len(val_dataset),
                    self.validation_interval,
                    self.validation_num_batches,
                )
            else:
                logger.info("Validation disabled (no validation split).")

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.load_worker,
            sampler=train_sampler,
            collate_fn=_collate_pad_batch,
        )

        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=False,
                seed=self.validation_split_seed,
            ) if config.world_size > 1 else None
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.load_worker,
                sampler=val_sampler,
                collate_fn=_collate_pad_batch,
            )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._cuda_mem_log_path = getattr(config, 'cuda_mem_log_path', None)
        self._cuda_mem_log_every = int(getattr(config, 'cuda_mem_log_every', 1))
        self._cuda_mem_log_step_count = 0
        self._cuda_mem_log_fp = None
        self._cuda_mem_log_writer = None
        if self._cuda_mem_log_path and torch.cuda.is_available():
            raw = str(self._cuda_mem_log_path)
            if '{rank}' in raw:
                log_path = Path(raw.replace('{rank}', str(config.local_rank)))
            else:
                p = Path(raw)
                log_path = p.parent / f'{p.stem}_rank{config.local_rank}{p.suffix}'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not log_path.exists() or log_path.stat().st_size == 0
            self._cuda_mem_log_fp = log_path.open('a', newline='')
            self._cuda_mem_log_writer = csv.writer(self._cuda_mem_log_fp)
            if new_file:
                self._cuda_mem_log_writer.writerow([
                    'ts_unix',
                    'optimizer_step',
                    'local_rank',
                    'batch_size',
                    'grad_accum_steps',
                    'allocated_mib',
                    'reserved_mib',
                    'max_allocated_mib',
                    'max_reserved_mib',
                ])
                self._cuda_mem_log_fp.flush()
            if config.rank == 0:
                logger.info(f"CUDA memory stats CSV (per rank): pattern {raw}")

        self.train_loader_iter = None

        self._use_amp = bool(getattr(config, "use_amp", True)) and torch.cuda.is_available()
        self._amp_dtype = self.dtype if self.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        self._use_fp16_scaler = self._use_amp and self.dtype == torch.float16
        self._grad_scaler = GradScaler("cuda", enabled=self._use_fp16_scaler)
        if hasattr(config, 'resume_from') and config.resume_from:
            self._load_training_state(config.resume_from)
    def report_cuda_mem(self, tag=""):
        if not torch.cuda.is_available():
            return
        device = self.device
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        max_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

        if self.config.rank == 0:
            logger.info(
                f"[{tag}] "
                f"allocated={alloc:.2f} MiB (active tensors), "
                f"reserved={reserved:.2f} MiB (allocator pool), "
                f"max_allocated={max_alloc:.2f} MiB (peak tensors since last reset), "
                f"max_reserved={max_reserved:.2f} MiB (peak pool since last reset)"
            )
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    def _append_cuda_mem_csv(self, optimizer_step):
        """Write one row; call after synchronize(), before empty_cache."""
        if self._cuda_mem_log_writer is None:
            return
        dev = self.device
        alloc = torch.cuda.memory_allocated(dev) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(dev) / (1024 ** 2)
        max_a = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        max_r = torch.cuda.max_memory_reserved(dev) / (1024 ** 2)
        self._cuda_mem_log_writer.writerow([
            f'{time.time():.6f}',
            optimizer_step,
            self.config.local_rank,
            self.config.batch_size,
            self.gradient_accumulation_steps,
            f'{alloc:.4f}',
            f'{reserved:.4f}',
            f'{max_a:.4f}',
            f'{max_r:.4f}',
        ])
        self._cuda_mem_log_fp.flush()

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        latent_dict = self._add_noise(
            latent=batch_dict['latents'],
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=0.5,
        )

        action_dict = self._add_noise(
            latent=batch_dict['actions'],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict['actions_mask'],
            action_mode=True,
            noisy_cond_prob=0.0,
        )

        # Match inference contract: the action branch does not get a separate
        # clean-action stream. The first action frame acts as an empty prefix
        # rather than a denoising target.
        action_dict['noisy_latents'][:, :, 0:1] = 0
        action_dict['latent'][:, :, 0:1] = 0
        action_dict['targets'][:, :, 0:1] = 0
        action_dict['timesteps'][:, 0:1] = 0

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask'].clone()
        action_dict['actions_mask'][:, :, 0:1] = 0

        if 'latents_mask' in batch_dict:
            latent_dict['latents_mask'] = batch_dict['latents_mask']

        if 'text_mask' in batch_dict:
            latent_dict['text_mask'] = batch_dict['text_mask']
            action_dict['text_mask'] = batch_dict['text_mask']

        # window_size 上界影响 attention 激活峰值；默认与原先 randint(4,65) 一致（最大 64）
        w_max_inc = int(getattr(self.config, "train_window_size_max", 64))
        w_max_inc = max(4, min(64, w_max_inc))
        w_hi_excl = w_max_inc + 1  # randint high exclusive → 取值 <= w_max_inc
        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, w_hi_excl, (1,)).item(),
        }
        return input_dict
    # def _prepare_input_dict(self, batch_dict):
    #     """Prepare input dict following infer code pattern from wan_va_server.py."""
    #     # Generate grid_id following infer code (no batch dimension yet)
    #     # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
    #     latent_dict = self._add_noise(
    #         latent=batch_dict['latents'], 
    #         train_scheduler=self.train_scheduler_latent, 
    #         action_mask=None, 
    #         action_mode=False,
    #         noisy_cond_prob=0.5)
        
    #     action_dict = self._add_noise(
    #         latent=batch_dict['actions'], 
    #         train_scheduler=self.train_scheduler_action, 
    #         action_mask=batch_dict['actions_mask'], 
    #         action_mode=True,
    #         noisy_cond_prob=0.0)

    #     latent_dict['text_emb'] = batch_dict['text_emb']
    #     action_dict['text_emb'] = batch_dict['text_emb']
    #     action_dict['actions_mask'] = batch_dict['actions_mask']

    #     latent_dict['text_emb'] = batch_dict['text_emb']
    #     action_dict['text_emb'] = batch_dict['text_emb']
    #     action_dict['actions_mask'] = batch_dict['actions_mask']

    #     if 'latents_mask' in batch_dict:
    #         latent_dict['latents_mask'] = batch_dict['latents_mask']

    #     if 'text_mask' in batch_dict:
    #         latent_dict['text_mask'] = batch_dict['text_mask']
    #         action_dict['text_mask'] = batch_dict['text_mask']
            
    #     input_dict = {
    #         'latent_dict': latent_dict,
    #         'action_dict': action_dict,
    #         'chunk_size': torch.randint(1, 5, (1,)).item(),
    #         'window_size': torch.randint(4, 65, (1,)).item(),
    #     }
    #     return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def _compute_action_reconstruction(self, action_pred, action_dict):
        action_timesteps = action_dict['timesteps']
        timestep_grid = self.train_scheduler_action.timesteps.to(
            device=action_timesteps.device,
            dtype=action_timesteps.dtype,
        )
        timestep_ids = torch.argmin(
            (timestep_grid[:, None, None] - action_timesteps[None]).abs(),
            dim=0,
        )
        sigma_grid = self.train_scheduler_action.sigmas.to(
            device=action_timesteps.device,
            dtype=action_pred.dtype,
        )
        sigma = sigma_grid[timestep_ids]

        action_recon = action_dict['noisy_latents'].float() - (
            sigma[:, None, :, None, None] * action_pred.float()
        )
        return action_recon

    def _compute_binary_action_aux_loss(self, action_pred, action_dict):
        if (not self.enable_binary_action_aux) or self.binary_action_aux_weight <= 0:
            return action_pred.new_tensor(0.0)

        action_recon = self._compute_action_reconstruction(action_pred, action_dict)
        action_target = action_dict['latent'].float().detach()
        action_mask = action_dict['actions_mask'].float()

        valid_channels = []
        for channel_id in self.binary_action_aux_channels:
            if 0 <= channel_id < action_recon.shape[1]:
                valid_channels.append(channel_id)
        if len(valid_channels) == 0:
            return action_pred.new_tensor(0.0)

        recon_sel = action_recon[:, valid_channels]
        target_sel = action_target[:, valid_channels]
        mask_sel = action_mask[:, valid_channels]
        target_bin = (target_sel > self.binary_action_aux_threshold).float()
        logits = (recon_sel - self.binary_action_aux_threshold) * max(
            self.binary_action_aux_logit_scale,
            1e-6,
        )
        prob = torch.sigmoid(logits)
        pos_w = max(self.binary_action_aux_pos_weight, 1e-6)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target_bin,
            pos_weight=torch.as_tensor(
                pos_w,
                device=logits.device,
                dtype=logits.dtype,
            ),
            reduction='none',
        )
        if self.binary_action_aux_loss_type == "focal":
            pt = torch.where(target_bin > 0.5, prob, 1.0 - prob)
            bce = bce * torch.pow(
                torch.clamp(1.0 - pt, min=0.0),
                self.binary_action_aux_focal_gamma,
            )

        bce = bce * mask_sel
        denom = mask_sel.sum()
        if denom <= 0:
            return action_pred.new_tensor(0.0)
        return bce.sum() / (denom + 1e-6)

    def _compute_action_generation_mse(self, action_pred, action_dict):
        action_recon = self._compute_action_reconstruction(action_pred, action_dict)
        action_target = action_dict['latent'].float().detach()
        action_mask = action_dict['actions_mask'].float()

        action_generation_mse = (action_recon - action_target).pow(2)
        action_generation_mse = action_generation_mse * action_mask

        action_generation_mse = action_generation_mse.permute(0, 2, 3, 4, 1)
        action_mask = action_mask.permute(0, 2, 3, 4, 1)
        action_generation_mse = action_generation_mse.flatten(0, 1).flatten(1)
        action_mask = action_mask.flatten(0, 1).flatten(1)

        action_generation_mse_per_frame = action_generation_mse.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        return (action_generation_mse_per_frame / (action_mask_per_frame + 1e-6)).mean()

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None

        was_training = self.transformer.training
        self.transformer.eval()

        latent_losses = []
        action_losses = []
        action_generation_mses = []

        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= self.validation_num_batches:
                break
            batch = self.convert_input_format(batch)
            input_dict = self._prepare_input_dict(batch)

            with autocast(
                device_type="cuda",
                dtype=self._amp_dtype,
                enabled=self._use_amp,
            ):
                output = self.transformer(input_dict, train_mode=True)
                latent_loss, action_loss, action_generation_mse = self.compute_loss(
                    input_dict,
                    output,
                    scale_by_grad_accum=False,
                )

            latent_losses.append(latent_loss.detach())
            action_losses.append(action_loss.detach())
            action_generation_mses.append(action_generation_mse.detach())

        if was_training:
            self.transformer.train()

        if not latent_losses:
            return None

        latent_loss_mean = dist_mean(torch.stack(latent_losses).mean()).detach().cpu().item()
        action_loss_mean = dist_mean(torch.stack(action_losses).mean()).detach().cpu().item()
        action_generation_mse_mean = dist_mean(
            torch.stack(action_generation_mses).mean()
        ).detach().cpu().item()
        return {
            'latent_loss': latent_loss_mean,
            'action_loss': action_loss_mean,
            'action_generation_mse': action_generation_mse_mean,
            'total_loss': latent_loss_mean + action_loss_mean,
        }

    def compute_loss(self,
        input_dict,
        pred,
        scale_by_grad_accum=True,
    ):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        
        
        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict['latent_dict']['targets'].float().detach(),
            reduction='none'
        )  # [B, C, F, H, W]

        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]

        # 正确处理 latent mask：
        # latents_mask 原始通常是 [B, 1, F, 1, 1]
        # 需要 expand 成和 latent_loss 一样的形状，才能正确统计有效元素个数
        if 'latents_mask' in input_dict['latent_dict']:
            latent_mask = input_dict['latent_dict']['latents_mask'].float()
            latent_mask = latent_mask.expand_as(latent_loss)   # [B, C, F, H, W]
        else:
            latent_mask = torch.ones_like(latent_loss)

        # 先把无效位置清零
        latent_loss = latent_loss * latent_mask

        # 变成按帧统计
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)   # [B, F, H, W, C]
        latent_mask = latent_mask.permute(0, 2, 3, 4, 1)   # [B, F, H, W, C]

        latent_loss = latent_loss.flatten(0, 1).flatten(1) # [B*F, H*W*C]
        latent_mask = latent_mask.flatten(0, 1).flatten(1) # [B*F, H*W*C]

        latent_loss_per_frame = latent_loss.sum(dim=1)      # [B*F]
        latent_mask_per_frame = latent_mask.sum(dim=1)      # [B*F]

        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()
        # # # Frame-wise video loss calculation
        # latent_loss = F.mse_loss(
        #     latent_pred.float(),
        #     input_dict['latent_dict']['targets'].float().detach(),
        #     reduction='none'
        # )
        # latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]

        # # 如果 batch 里有 latents_mask，就把 pad 出来的帧屏蔽掉
        # if 'latents_mask' in input_dict['latent_dict']:
        #     latent_loss = latent_loss * input_dict['latent_dict']['latents_mask'].float()

        # latent_loss = latent_loss.permute(0, 2, 3, 4, 1)   # (B,C,F,H,W) -> (B,F,H,W,C)
        # latent_loss = latent_loss.flatten(0, 1).flatten(1) # (B*F, H*W*C)

        # if 'latents_mask' in input_dict['latent_dict']:
        #     latent_mask = input_dict['latent_dict']['latents_mask'].float().permute(0, 2, 3, 4, 1)
        #     latent_mask = latent_mask.flatten(0, 1).flatten(1)
        #     latent_mask_per_frame = latent_mask.sum(dim=1)
        # else:
        #     latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)

        # latent_loss_per_frame = latent_loss.sum(dim=1)
        # latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()
        # latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        # latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        # latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        # latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # # Sum per frame and compute mask per frame
        # latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        # latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        # latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()
        binary_action_aux_loss = self._compute_binary_action_aux_loss(
            action_pred,
            input_dict['action_dict'],
        )
        self._last_binary_action_aux_loss = binary_action_aux_loss.detach()
        action_loss = action_loss + (self.binary_action_aux_weight * binary_action_aux_loss)

        with torch.no_grad():
            action_generation_mse = self._compute_action_generation_mse(
                action_pred,
                input_dict['action_dict'],
            )

        loss_scale = self.gradient_accumulation_steps if scale_by_grad_accum else 1
        return (
            latent_loss / loss_scale,
            action_loss / loss_scale,
            action_generation_mse / loss_scale,
        )
    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0

        if not self.use_deepspeed:
            if not should_sync:
                self.transformer.set_requires_gradient_sync(False)
            else:
                self.transformer.set_requires_gradient_sync(True)

        if self._cuda_mem_log_writer is not None and batch_idx == 0:
            torch.cuda.reset_peak_memory_stats(self.device)

        if self.step < 3 and batch_idx == 0:
            self.report_cuda_mem("before forward")

        with autocast(
            device_type="cuda",
            dtype=self._amp_dtype,
            enabled=self._use_amp,
        ):
            output = self.transformer(input_dict, train_mode=True)
            latent_loss, action_loss, action_generation_mse = self.compute_loss(input_dict, output)
            loss = latent_loss + action_loss

        if self.step < 3 and batch_idx == 0:
            self.report_cuda_mem("after forward")

        if self.use_deepspeed:
            should_sync = self.transformer.is_gradient_accumulation_boundary()
            self.transformer.backward(loss)
        elif self._use_fp16_scaler:
            self._grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.step < 3 and batch_idx == 0:
            self.report_cuda_mem("after backward")

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'action_generation_mse': action_generation_mse.detach(),
            'binary_action_aux_loss': self._last_binary_action_aux_loss.detach(),
        }

        if self.use_deepspeed:
            self.transformer.step()
            if should_sync:
                total_norm = self.transformer.get_global_grad_norm()
                if total_norm is None:
                    total_norm = torch.tensor(float("nan"), device=self.device)
                elif not isinstance(total_norm, torch.Tensor):
                    total_norm = torch.tensor(float(total_norm), device=self.device)
                losses['total_norm'] = total_norm.detach()
                losses['should_log'] = True
            else:
                losses['should_log'] = False
        elif should_sync:
            if self._use_fp16_scaler:
                self._grad_scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            if self._use_fp16_scaler:
                self._grad_scaler.step(self.optimizer)
                self._grad_scaler.update()
            else:
                self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.step < 3 and batch_idx == 0:
                self.report_cuda_mem("after optimizer.step")

            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses
    # def _train_step(self, batch, batch_idx):
    #     """Train a single batch, returns losses for logging."""
    #     batch = self.convert_input_format(batch)
    #     input_dict = self._prepare_input_dict(batch)

    #     should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0

    #     if not should_sync:
    #         self.transformer.set_requires_gradient_sync(False)
    #     else:
    #         self.transformer.set_requires_gradient_sync(True)

    #     if self._cuda_mem_log_writer is not None and batch_idx == 0:
    #         torch.cuda.reset_peak_memory_stats(self.device)

    #     # 只在前几步打印，避免刷屏
    #     if self.step < 3 and batch_idx == 0:
    #         self.report_cuda_mem("before forward")

    #     output = self.transformer(input_dict, train_mode=True)

    #     if self.step < 3 and batch_idx == 0:
    #         self.report_cuda_mem("after forward")

    #     latent_loss, action_loss = self.compute_loss(input_dict, output)
    #     loss = latent_loss + action_loss

    #     loss.backward()

    #     if self.step < 3 and batch_idx == 0:
    #         self.report_cuda_mem("after backward")

    #     losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}

    #     # Only update weights after accumulating gradients
    #     if should_sync:
    #         total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
    #         self.optimizer.step()
    #         self.lr_scheduler.step()
    #         self.optimizer.zero_grad(set_to_none=True)

    #         if self.step < 3 and batch_idx == 0:
    #             self.report_cuda_mem("after optimizer.step")

    #         losses['total_norm'] = total_norm
    #         losses['should_log'] = True
    #     else:
    #         losses['should_log'] = False

    #     return losses
    # def _train_step(self, batch, batch_idx):
    #     """Train a single batch, returns losses for logging."""
    #     batch = self.convert_input_format(batch)
    #     input_dict = self._prepare_input_dict(batch)
        
    #     should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
    #     if not should_sync:
    #         self.transformer.set_requires_gradient_sync(False)
    #     else:
    #         self.transformer.set_requires_gradient_sync(True)

    #     output = self.transformer(input_dict, train_mode=True)
    #     latent_loss, action_loss = self.compute_loss(input_dict, output)
    #     loss = latent_loss + action_loss

    #     loss.backward()

    #     losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}
        
    #     # Only update weights after accumulating gradients
    #     if should_sync:
    #         total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
    #         self.optimizer.step()
    #         self.lr_scheduler.step()
    #         self.optimizer.zero_grad()
            
    #         losses['total_norm'] = total_norm
    #         losses['should_log'] = True
    #     else:
    #         losses['should_log'] = False

    #     return losses

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            optim_state = None
            if self.use_deepspeed:
                transformer_module = (
                    self.transformer.module
                    if hasattr(self.transformer, "module")
                    else self.transformer
                )
                state_dict = {
                    k: v.detach().cpu()
                    for k, v in transformer_module.state_dict().items()
                }
            else:
                state_dict = get_model_state_dict(
                    self.transformer,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
                optim_state = get_optimizer_state_dict(
                    self.transformer,
                    self.optimizer,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                provenance_path = checkpoint_dir / "provenance.json"
                provenance = {
                    "step": self.step,
                    "saved_at_unix": time.time(),
                    "hostname": os.uname().nodename,
                    "config": _jsonable_config_subset(self.config),
                    **_git_metadata(Path(__file__).resolve().parent.parent),
                }
                with open(provenance_path, "w", encoding="utf-8") as f:
                    json.dump(provenance, f, ensure_ascii=False, indent=2)

                # Save optimizer state for resumable training (FSDP path).
                if optim_state is not None:
                    training_state_path = checkpoint_dir / "training_state.pt"
                    logger.info(f"Saving training state to {training_state_path}")
                    torch.save({
                        'step': self.step,
                        'optimizer_state_dict': optim_state,
                        'lr_scheduler_state_dict': (
                            self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                        ),
                        'grad_scaler_state_dict': (
                            self._grad_scaler.state_dict() if self._use_fp16_scaler else None
                        ),
                        'config': vars(self.config),
                    }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        if self.use_deepspeed:
            if self.config.rank == 0:
                logger.warning("DeepSpeed path currently skips optimizer state restore from training_state.pt.")
            return

        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)
        self.step = int(training_state.get('step', 0))

        # Restore optimizer state when possible. If model/optimizer param names
        # changed across code versions, continue from model weights + step only.
        optim_state = training_state.get('optimizer_state_dict')
        if optim_state is None:
            if self.config.rank == 0:
                logger.warning(
                    "Optimizer state not found in training_state.pt; "
                    "resuming with freshly initialized optimizer at step %d.",
                    self.step,
                )
        else:
            try:
                set_optimizer_state_dict(
                    self.transformer,
                    self.optimizer,
                    optim_state_dict=optim_state,
                    options=StateDictOptions(full_state_dict=True, strict=False),
                )
            except Exception as e:
                if self.config.rank == 0:
                    logger.warning(
                        "Skipping optimizer state restore due to mismatch: %r. "
                        "Continue from model weights at step %d with a fresh optimizer.",
                        e,
                        self.step,
                    )

        scheduler_state = training_state.get('lr_scheduler_state_dict')
        if self.lr_scheduler is not None:
            if scheduler_state is not None:
                try:
                    self.lr_scheduler.load_state_dict(scheduler_state)
                except Exception as e:
                    if self.config.rank == 0:
                        logger.warning(
                            "Skipping lr_scheduler state restore due to mismatch: %r. "
                            "Will align scheduler to global step %d.",
                            e,
                            self.step,
                        )
                    try:
                        self.lr_scheduler.step(self.step)
                    except Exception:
                        for _ in range(self.step):
                            self.lr_scheduler.step()
            else:
                # Backward compatibility for old checkpoints: align scheduler to restored step.
                try:
                    self.lr_scheduler.step(self.step)
                except Exception:
                    for _ in range(self.step):
                        self.lr_scheduler.step()

        grad_scaler_state = training_state.get('grad_scaler_state_dict')
        if self._use_fp16_scaler and grad_scaler_state is not None:
            try:
                self._grad_scaler.load_state_dict(grad_scaler_state)
            except Exception as e:
                if self.config.rank == 0:
                    logger.warning(
                        "Skipping GradScaler state restore due to mismatch: %r. "
                        "Continue with fresh scaler.",
                        e,
                    )

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        if self.use_deepspeed:
            self.transformer.zero_grad()
        else:
            self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_action_generation_mses = []
        accumulated_binary_action_aux_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            accumulated_action_generation_mses.append(losses['action_generation_mse'])
            accumulated_binary_action_aux_losses.append(losses['binary_action_aux_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "get_last_lr"):
                    lr = self.lr_scheduler.get_last_lr()[0]
                else:
                    lr = self.optimizer.param_groups[0]["lr"]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                action_generation_mse_show = dist_mean(torch.stack(accumulated_action_generation_mses).sum()).detach().cpu().item()
                binary_action_aux_loss_show = dist_mean(torch.stack(accumulated_binary_action_aux_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_action_generation_mse_show = dist_max(torch.stack(accumulated_action_generation_mses).sum()).detach().cpu().item()
                max_binary_action_aux_loss_show = dist_max(torch.stack(accumulated_binary_action_aux_losses).sum()).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_action_generation_mses = []
                accumulated_binary_action_aux_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self._cuda_mem_log_writer is not None:
                    self._cuda_mem_log_step_count += 1
                    if self._cuda_mem_log_step_count % self._cuda_mem_log_every == 0:
                        self._append_cuda_mem_csv(self.step + 1)
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += 1
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'action_gen_mse': f'{action_generation_mse_show:.4f}',
                        'binary_aux': f'{binary_action_aux_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_avg_action_generation_mse': action_generation_mse_show,
                            'loss_metrics/global_avg_binary_action_aux_loss': binary_action_aux_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'loss_metrics/global_max_action_generation_mse': max_action_generation_mse_show,
                            'loss_metrics/global_max_binary_action_aux_loss': max_binary_action_aux_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }, step=self.step)
                
                self.step += 1

                if (
                    self.val_loader is not None
                    and self.validation_interval > 0
                    and self.step % self.validation_interval == 0
                ):
                    if self.config.rank == 0:
                        logger.info(f"Running validation at step {self.step}...")
                    val_metrics = self.validate()
                    if val_metrics is not None and self.config.rank == 0:
                        logger.info(
                            "Validation step %d | video_loss=%.6f | action_loss=%.6f | action_gen_mse=%.6f | total_loss=%.6f",
                            self.step,
                            val_metrics['latent_loss'],
                            val_metrics['action_loss'],
                            val_metrics['action_generation_mse'],
                            val_metrics['total_loss'],
                        )
                        if self.config.enable_wandb:
                            self.wandb.log({
                                'val_metrics/video_loss': val_metrics['latent_loss'],
                                'val_metrics/action_loss': val_metrics['action_loss'],
                                'val_metrics/action_generation_mse': val_metrics['action_generation_mse'],
                                'val_metrics/total_loss': val_metrics['total_loss'],
                            }, step=self.step)
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

        progress_bar.close()
        if self._cuda_mem_log_fp is not None:
            self._cuda_mem_log_fp.close()
            self._cuda_mem_log_fp = None
            self._cuda_mem_log_writer = None
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = get_config(args.config_name)
    config.config_name = args.config_name
    config.use_deepspeed = bool(args.use_deepspeed)
    if args.deepspeed_config is not None:
        config.deepspeed_config = args.deepspeed_config
    elif config.use_deepspeed and not getattr(config, "deepspeed_config", None):
        config.deepspeed_config = str(
            Path(__file__).resolve().parent / "configs" / "deepspeed" / "zero2_offload.json"
        )

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        old_save_root = str(getattr(config, "save_root", ""))
        new_save_root = str(args.save_root)
        config.save_root = new_save_root
        # Keep CUDA memory CSV under the same run root when --save-root overrides config.
        cuda_mem_log_path = getattr(config, "cuda_mem_log_path", None)
        if cuda_mem_log_path:
            raw_log_path = str(cuda_mem_log_path)
            if old_save_root and raw_log_path.startswith(old_save_root):
                config.cuda_mem_log_path = new_save_root + raw_log_path[len(old_save_root):]

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Enable DeepSpeed training path (disable FSDP path).",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="DeepSpeed JSON config path. Used when --use-deepspeed is set.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
