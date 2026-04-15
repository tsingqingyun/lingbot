# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robocasa_cfg import va_robocasa_cfg
import os

va_robocasa_train_cfg = EasyDict(__name__='Config: VA robocasa train')
va_robocasa_train_cfg.update(va_robocasa_cfg)

# va_robotwin_train_cfg.resume_from = '/robby/share/Robotics/lilin1/code/Wan_VA_Release/train_out/checkpoints/checkpoint_step_10'

va_robocasa_train_cfg.dataset_path = '/root/lingbot_va/robocasa_datasets_composite'
va_robocasa_train_cfg.task_names = [
    'StoreLeftoversInBowl',
   'LoadDishwasher',
   'KettleBoiling',
    'SearingMeat',
]
va_robocasa_train_cfg.empty_emb_path = os.path.join(va_robocasa_train_cfg.dataset_path, 'empty_emb.pt')
va_robocasa_train_cfg.enable_wandb = True
# 若 403：在网页端确认 team slug，与下面一致；或注释掉三行仅用环境变量 / 仅用 key 默认 entity。
# va_robocasa_train_cfg.wandb_entity = "your-team-or-username"
# va_robocasa_train_cfg.wandb_project = "lingbot-va"
# va_robocasa_train_cfg.wandb_run_name = "robocasa_train"
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
# 峰值显存主要由 batch_size 决定；grad_accum 几乎不改变单次 forward 峰值。
# 每卡每 optimizer step 样本数 = batch_size * gradient_accumulation_steps（再 × world_size 为整节点）。
# 此前 bs=2 易 OOM；用 bs=1、grad_accum=8 与「每卡 2×4=8 个样本/step」对齐且峰值更低。
va_robocasa_train_cfg.batch_size = 1
va_robocasa_train_cfg.gradient_accumulation_steps = 8
# 训练时 window_size 原逻辑为 randint(4,65)→最大 64，偶发极大序列会顶满显存；此处限制上界（含）。
va_robocasa_train_cfg.train_window_size_max = 40
# 0 = 每层对 attn1/ffn 额外做 AC（更省显存、更慢）；10 = 仅后 20 层（默认与其它任务一致）
va_robocasa_train_cfg.ac_inner_checkpoint_min_layer = 0
va_robocasa_train_cfg.num_steps = 50000

# 显存 CSV：allocated/reserved/max_*（MiB），每步在 optimizer 更新后、empty_cache 前写入；{rank}=本进程 local_rank
va_robocasa_train_cfg.cuda_mem_log_path = os.path.join(
    va_robocasa_train_cfg.save_root, 'cuda_mem_logs', 'mem_rank{rank}.csv'
)
