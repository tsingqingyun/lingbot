# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robocasa_cfg = EasyDict(__name__='Config: VA robocasa')
va_robocasa_cfg.update(va_shared_cfg)

# Trained checkpoint root (must contain subdirectory transformer/).
va_robocasa_cfg.wan22_pretrained_model_name_or_path = (
    os.environ.get(
        "LINGBOT_ROBOCASA_MODEL_PATH",
        #"/cephfs/shared/xcxhx/lingbot-va",
        #"/cephfs/shared/xcx/lingbot-va/train_out/checkpoint_step_500",
        "/cephfs/shared/xcx/lingbot-va/train_out_0420/checkpoint_step_100",
    )
)
# Optional: checkpoint directory for resume/finetune.
# Expected layout:
#   <resume_from>/transformer/
#   <resume_from>/training_state.pt (optional, for optimizer/step restore)
va_robocasa_cfg.resume_from = os.environ.get(
    "LINGBOT_ROBOCASA_RESUME_FROM",
    None,
)
# Full Wan tree: vae/, tokenizer/, text_encoder/.
va_robocasa_cfg.wan22_base_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_WAN_BASE_PATH", "/cephfs/shared/xcxhx/lingbot-va"
)
    
va_robocasa_cfg.attn_window = 72
va_robocasa_cfg.frame_chunk_size = 2
# Websocket inference: use "torch" unless inference forward wires FlexAttn init_mask.
va_robocasa_cfg.inference_attn_mode = "torch"
va_robocasa_cfg.env_type = 'robocasa_tshape'

va_robocasa_cfg.height = 256
va_robocasa_cfg.width = 256
va_robocasa_cfg.action_dim = 30
va_robocasa_cfg.action_per_frame = 8
va_robocasa_cfg.obs_cam_keys = [
    'observation.images.robot0_agentview_left', 'observation.images.robot0_agentview_right',
    'observation.images.robot0_eye_in_hand'
]
va_robocasa_cfg.guidance_scale = 5
va_robocasa_cfg.action_guidance_scale = 1

va_robocasa_cfg.num_inference_steps = 40
va_robocasa_cfg.video_exec_step = -1
va_robocasa_cfg.action_num_inference_steps = 50

va_robocasa_cfg.snr_shift = 5.0
va_robocasa_cfg.action_snr_shift = 1.0

va_robocasa_cfg.sample_by_frames = True
va_robocasa_cfg.sample_unit_frames = 8

# Keep this list aligned with wan_va.dataset.lerobot_latent_dataset.robocasa_to_lingbot mask.
va_robocasa_cfg.used_action_channel_ids = [0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 22, 29]
inverse_used_action_channel_ids = [
    len(va_robocasa_cfg.used_action_channel_ids)
] * va_robocasa_cfg.action_dim
for i, j in enumerate(va_robocasa_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robocasa_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robocasa_cfg.action_norm_method = 'quantiles'
va_robocasa_cfg.norm_stat = {
    "q01": [
    -1.0,
    -1.0,
    -1.0,
    -0.19411402940750122,
    -0.208142052590847,
    -0.19713866859674453,
    0.9403075903654099,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    -0.5142857432365417,
    -0.8642857074737549,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.5199999809265137,
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
    1.0,
    0.1506267488002777,
    0.16056651771068553,
    0.21542802453041077,
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
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.4657142758369446,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
}
