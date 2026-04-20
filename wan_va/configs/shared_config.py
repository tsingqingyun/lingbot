# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

# wan_va_server.run(): "server" (RPC/HTTP service) or "i2va" (single-process demo).
va_shared_cfg.infer_mode = 'server'

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './train_out_new_04200104'

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = True

# If set (str path), append per-step CUDA memory stats as CSV. Use "{rank}" for local_rank.
va_shared_cfg.cuda_mem_log_path = None
# Log every N optimizer steps (1 = each step).
va_shared_cfg.cuda_mem_log_every = 1

# torch.amp.autocast around forward; GradScaler is used only when param_dtype is float16.
va_shared_cfg.use_amp = True

# When inner activation checkpointing applies, also checkpoint cross-attn (attn2).
va_shared_cfg.ac_checkpoint_attn2 = False