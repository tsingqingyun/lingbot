# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg


va_libero_cfg = EasyDict(__name__='Config: VA libero')
va_libero_cfg.update(va_shared_cfg)

# Trained checkpoint root (must contain subdirectory transformer/).
va_libero_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_LIBERO_MODEL_PATH",
    os.environ.get("LINGBOT_ROBOCASA_MODEL_PATH", "/path/to/pretrained/model"),
)
# Full Wan tree root (vae/, tokenizer/, text_encoder/).
va_libero_cfg.wan22_base_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_WAN_BASE_PATH",
    va_libero_cfg.wan22_pretrained_model_name_or_path,
)

va_libero_cfg.attn_window = 30
va_libero_cfg.frame_chunk_size = 4
va_libero_cfg.inference_attn_mode = "torch"
va_libero_cfg.env_type = 'none'

va_libero_cfg.height = 128
va_libero_cfg.width = 128
va_libero_cfg.action_dim = 30
va_libero_cfg.action_per_frame = 4
va_libero_cfg.obs_cam_keys = [
    'observation.images.agentview_rgb',
    'observation.images.eye_in_hand_rgb',
]
va_libero_cfg.guidance_scale = 5
va_libero_cfg.action_guidance_scale = 1

va_libero_cfg.num_inference_steps = 20
va_libero_cfg.video_exec_step = -1
va_libero_cfg.action_num_inference_steps = 50

va_libero_cfg.snr_shift = 5.0
va_libero_cfg.action_snr_shift = 1.0

# LIBERO policy action typically uses 7 dims:
# [eef(6), gripper(1)] -> mapped to LingBot 30D channels [0:6] + [28].
va_libero_cfg.used_action_channel_ids = list(range(0, 6)) + [28]
inverse_used_action_channel_ids = [
    len(va_libero_cfg.used_action_channel_ids)
] * va_libero_cfg.action_dim
for i, j in enumerate(va_libero_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_libero_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_libero_cfg.action_norm_method = 'quantiles'
va_libero_cfg.norm_stat = {
    "q01": [
        -0.6589285731315613,
        -0.84375,
        -0.9375,
        -0.12107142806053162,
        -0.15964286029338837,
        -0.26571428775787354,
    ] + [0.0] * 22 + [-1.0, 0.0],
    "q99": [
        0.8999999761581421,
        0.8544642925262451,
        0.9375,
        0.17142857611179352,
        0.1842857152223587,
        0.34392857551574707,
    ] + [0.0] * 22 + [1.0, 0.0],
}
