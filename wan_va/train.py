# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import csv
import os
import sys
import time
from pathlib import Path
#import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
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

from dataset import MultiLatentLeRobotDataset
import gc

import gc


def _pad_tensor_along_dim(x: torch.Tensor, target_len: int, dim: int, pad_value=0):
    """
    沿指定维度补到 target_len。
    x: torch.Tensor
    dim: 要补的维度
    """
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
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                # dir=log_dir,
                config=config,
                mode="online",
                name='test_lln'
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

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
            torch_device='cpu',
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer
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
        train_dataset = MultiLatentLeRobotDataset(config=config)
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

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
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

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        if 'latents_mask' in batch_dict:
            latent_dict['latents_mask'] = batch_dict['latents_mask']

        if 'text_mask' in batch_dict:
            latent_dict['text_mask'] = batch_dict['text_mask']
            action_dict['text_mask'] = batch_dict['text_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
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

    def compute_loss(self,
        input_dict,
        pred
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

        # # Frame-wise video loss calculation
        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict['latent_dict']['targets'].float().detach(),
            reduction='none'
        )
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]

        # 如果 batch 里有 latents_mask，就把 pad 出来的帧屏蔽掉
        if 'latents_mask' in input_dict['latent_dict']:
            latent_loss = latent_loss * input_dict['latent_dict']['latents_mask'].float()

        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)   # (B,C,F,H,W) -> (B,F,H,W,C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1) # (B*F, H*W*C)

        if 'latents_mask' in input_dict['latent_dict']:
            latent_mask = input_dict['latent_dict']['latents_mask'].float().permute(0, 2, 3, 4, 1)
            latent_mask = latent_mask.flatten(0, 1).flatten(1)
            latent_mask_per_frame = latent_mask.sum(dim=1)
        else:
            latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)

        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()
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

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps
    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0

        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        if self._cuda_mem_log_writer is not None and batch_idx == 0:
            torch.cuda.reset_peak_memory_stats(self.device)

        if self.step < 3 and batch_idx == 0:
            self.report_cuda_mem("before forward")

        output = self.transformer(input_dict, train_mode=True)

        if self.step < 3 and batch_idx == 0:
            self.report_cuda_mem("after forward")

        latent_loss, action_loss = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss

        loss.backward()

        if self.step < 3 and batch_idx == 0:
            self.report_cuda_mem("after backward")

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
        }

        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
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
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

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

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

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

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

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

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
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
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }, step=self.step)
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        if self._cuda_mem_log_fp is not None:
            self._cuda_mem_log_fp.close()
            self._cuda_mem_log_fp = None
            self._cuda_mem_log_writer = None
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = get_config(args.config_name)

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

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

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()