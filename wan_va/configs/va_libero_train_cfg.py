# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_libero_cfg import va_libero_cfg


va_libero_train_cfg = EasyDict(__name__='Config: VA libero train')
va_libero_train_cfg.update(va_libero_cfg)

va_libero_train_cfg.resume_from = os.environ.get(
    "LINGBOT_LIBERO_RESUME_FROM",
    None,
)
va_libero_train_cfg.dataset_path = os.environ.get(
    "LINGBOT_LIBERO_DATASET_PATH",
    "/path/to/libero_long_lerobot",
)
va_libero_train_cfg.empty_emb_path = os.path.join(
    va_libero_train_cfg.dataset_path, "empty_emb.pt"
)
va_libero_train_cfg.enable_wandb = os.environ.get(
    "LINGBOT_ENABLE_WANDB", "1"
).strip() not in ("0", "false", "False")
va_libero_train_cfg.load_worker = 16
va_libero_train_cfg.save_interval = 200
va_libero_train_cfg.gc_interval = 50
va_libero_train_cfg.cfg_prob = 0.1

# Training parameters
va_libero_train_cfg.learning_rate = 1e-5
va_libero_train_cfg.beta1 = 0.9
va_libero_train_cfg.beta2 = 0.95
va_libero_train_cfg.weight_decay = 1e-1
va_libero_train_cfg.warmup_steps = 10
va_libero_train_cfg.batch_size = 1
va_libero_train_cfg.gradient_accumulation_steps = 10
va_libero_train_cfg.num_steps = 5000

va_libero_train_cfg.cuda_mem_log_path = os.path.join(
    va_libero_train_cfg.save_root, 'cuda_mem_logs', 'mem_rank{rank}.csv'
)
