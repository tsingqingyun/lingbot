# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robocasa_cfg import va_robocasa_cfg
import os

va_robocasa_train_cfg = EasyDict(__name__='Config: VA robocasa train')
va_robocasa_train_cfg.update(va_robocasa_cfg)

# Resume training from a previous checkpoint directory.
# Example:
#   export LINGBOT_ROBOCASA_RESUME_FROM=/path/to/checkpoints/checkpoint_step_1000
# Directory should contain `transformer/`; `training_state.pt` is optional.
va_robocasa_train_cfg.resume_from = os.environ.get(
    "LINGBOT_ROBOCASA_RESUME_FROM",
    getattr(va_robocasa_cfg, "resume_from", None),
)

va_robocasa_train_cfg.dataset_path = os.environ.get(
    "LINGBOT_ROBOCASA_DATASET_PATH",
    '/root/lingbot_va/robocasa_datasets_composite',
)
va_robocasa_train_cfg.task_names = [
    'StoreLeftoversInBowl',
   'LoadDishwasher',
   'KettleBoiling',
    'SearingMeat',
]
va_robocasa_train_cfg.empty_emb_path = os.path.join(va_robocasa_train_cfg.dataset_path, 'empty_emb.pt')
va_robocasa_train_cfg.enable_wandb = os.environ.get("LINGBOT_ENABLE_WANDB", "1").strip() not in ("0", "false", "False")
# 若 403：在网页端确认 team slug，与下面一致；或注释掉三行仅用环境变量 / 仅用 key 默认 entity。
# va_robocasa_train_cfg.wandb_entity = "your-team-or-username"
# va_robocasa_train_cfg.wandb_project = "lingbot-va"
# va_robocasa_train_cfg.wandb_run_name = "robocasa_train"
va_robocasa_train_cfg.load_worker = 16
va_robocasa_train_cfg.save_interval = 100
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
# Binary auxiliary supervision for discrete action channels:
# 14 = gripper, 29 = control_mode (in LingBot 30D space).
va_robocasa_train_cfg.enable_binary_action_aux = True
va_robocasa_train_cfg.binary_action_aux_channels = [14, 29]
va_robocasa_train_cfg.binary_action_aux_weight = 0.25
va_robocasa_train_cfg.binary_action_aux_loss_type = "focal"  # "bce" | "focal"
va_robocasa_train_cfg.binary_action_aux_focal_gamma = 2.0
va_robocasa_train_cfg.binary_action_aux_pos_weight = 1.5
# 0 = 每层对 attn1/ffn 额外做 AC（更省显存、更慢）；10 = 仅后 20 层（默认与其它任务一致）
va_robocasa_train_cfg.ac_inner_checkpoint_min_layer = 0
va_robocasa_train_cfg.num_steps = 50000

# 显存 CSV：allocated/reserved/max_*（MiB），每步在 optimizer 更新后、empty_cache 前写入；{rank}=本进程 local_rank
va_robocasa_train_cfg.cuda_mem_log_path = os.path.join(
    va_robocasa_train_cfg.save_root, 'cuda_mem_logs', 'mem_rank{rank}.csv'
)
