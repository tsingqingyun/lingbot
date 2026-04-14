# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robocasa_cfg import va_robocasa_cfg
import os

va_robocasa_train_cfg = EasyDict(__name__='Config: VA robocasa train')
va_robocasa_train_cfg.update(va_robocasa_cfg)

# va_robotwin_train_cfg.resume_from = '/robby/share/Robotics/lilin1/code/Wan_VA_Release/train_out/checkpoints/checkpoint_step_10'

va_robocasa_train_cfg.dataset_path = '/cephfs/shared/xcxhx/robocasa_datasets_composite'
va_robocasa_train_cfg.task_names = [
    'StoreLeftoversInBowl',
    'LoadDishwasher',
    'KettleBoiling',
    'SearingMeat',
]
va_robocasa_train_cfg.empty_emb_path = os.path.join(va_robocasa_train_cfg.dataset_path, 'empty_emb.pt')
va_robocasa_train_cfg.enable_wandb = False
va_robocasa_train_cfg.load_worker = 16
va_robocasa_train_cfg.save_interval = 1000
va_robocasa_train_cfg.gc_interval = 50
va_robocasa_train_cfg.cfg_prob = 0.1

# Training parameters
va_robocasa_train_cfg.learning_rate = 1e-5
va_robocasa_train_cfg.beta1 = 0.9
va_robocasa_train_cfg.beta2 = 0.95
va_robocasa_train_cfg.weight_decay = 0.1
va_robocasa_train_cfg.warmup_steps = 10
va_robocasa_train_cfg.batch_size = 1
va_robocasa_train_cfg.gradient_accumulation_steps = 6
va_robocasa_train_cfg.num_steps = 50000

# 显存 CSV：allocated/reserved/max_*（MiB），每步在 optimizer 更新后、empty_cache 前写入；{rank}=本进程 local_rank
va_robocasa_train_cfg.cuda_mem_log_path = os.path.join(
    va_robocasa_train_cfg.save_root, 'cuda_mem_logs', 'mem_rank{rank}.csv'
)
