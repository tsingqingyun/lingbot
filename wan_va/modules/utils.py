# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    attn_mode=None,
):
    """Load Wan transformer. If ``attn_mode`` is set, it overrides the checkpoint
    (e.g. ``torch`` for websocket inference: ``flex`` requires ``FlexAttnFunc.init_mask``
    which is only invoked on the training forward path today).
    """
    kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=None,
    )
    if attn_mode is not None:
        kwargs["attn_mode"] = attn_mode
    model = WanTransformer3DModel.from_pretrained(transformer_path, **kwargs)
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv
        self.min_temporal_frames = 1

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count
        for m in self.encoder.modules():
            if not isinstance(m, torch.nn.Conv3d):
                continue
            kernel_size = m.kernel_size
            if isinstance(kernel_size, tuple) and len(kernel_size) == 3:
                self.min_temporal_frames = max(self.min_temporal_frames,
                                               int(kernel_size[0]))

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        original_frames = x_chunk.shape[2]
        if original_frames < self.min_temporal_frames:
            pad_frames = self.min_temporal_frames - original_frames
            head = x_chunk[:, :, :1].repeat(1, 1, pad_frames, 1, 1)
            x_chunk = torch.cat([head, x_chunk], dim=2)
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        if original_frames < self.min_temporal_frames:
            # Drop prefix outputs introduced by temporal head-padding.
            enc = enc[:, :, -original_frames:]
        return enc
