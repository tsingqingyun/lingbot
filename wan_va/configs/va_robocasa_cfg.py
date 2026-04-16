# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robocasa_cfg = EasyDict(__name__='Config: VA robocasa')
va_robocasa_cfg.update(va_shared_cfg)

va_robocasa_cfg.wan22_pretrained_model_name_or_path = "/cephfs/shared/xcx/lingbot-va/train_out_1/checkpoints/checkpoint_step_9000"
    
va_robocasa_cfg.attn_window = 72
va_robocasa_cfg.frame_chunk_size = 2
va_robocasa_cfg.env_type = 'robocasa_tshape'

va_robocasa_cfg.height = 256
va_robocasa_cfg.width = 256
va_robocasa_cfg.action_dim = 30
va_robocasa_cfg.action_per_frame = 16
va_robocasa_cfg.obs_cam_keys = [
    'observation.images.robot0_agentview_left', 'observation.images.robot0_agentview_right',
    'observation.images.robot0_eye_in_hand'
]
va_robocasa_cfg.guidance_scale = 5
va_robocasa_cfg.action_guidance_scale = 1

va_robocasa_cfg.num_inference_steps = 25
va_robocasa_cfg.video_exec_step = -1
va_robocasa_cfg.action_num_inference_steps = 50

va_robocasa_cfg.snr_shift = 5.0
va_robocasa_cfg.action_snr_shift = 1.0

va_robocasa_cfg.sample_by_frames = True
va_robocasa_cfg.sample_unit_frames = 8

va_robocasa_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_robocasa_cfg.used_action_channel_ids)
] * va_robocasa_cfg.action_dim
for i, j in enumerate(va_robocasa_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robocasa_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robocasa_cfg.action_norm_method = 'quantiles'
va_robocasa_cfg.norm_stat = {
    "q01": [
    -0.5142857432365417,
    -0.8642857074737549,
    -0.5199999809265137,
    -0.22984884679317474,
    -0.4794255495071411,
    -0.4207354784011841,
    0.7701511383056641,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    -1.0,
    -0.39142856001853943,
    -0.41999998688697815,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.39714285731315613,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0
    ],
    "q99": [
    1.0,
    1.0,
    0.4657142758369446,
    0.22984884679317474,
    0.4794255495071411,
    0.4207354784011841,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.3028571307659149,
    0.322857141494751,
    1.0,
    1.0,
    1.0,
    1.0,
    0.43714284896850586,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
}
