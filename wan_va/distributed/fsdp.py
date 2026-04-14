# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def apply_ac(model):
    """Apply selective activation checkpointing to transformer blocks.

    Layers 0–9: whole-block checkpoint only (faster).
    Layers 10+: extra checkpointing on self-attn and FFN (lower activation memory).
    """
    for layer_id, transformer_block in enumerate(model.blocks):
        if layer_id >= 10:
            if hasattr(transformer_block, "attn1"):
                transformer_block.attn1 = ptd_checkpoint_wrapper(
                    transformer_block.attn1,
                    preserve_rng_state=False,
                )
            if hasattr(transformer_block, "ffn"):
                transformer_block.ffn = ptd_checkpoint_wrapper(
                    transformer_block.ffn,
                    preserve_rng_state=False,
                )
        wrapped = ptd_checkpoint_wrapper(
            transformer_block,
            preserve_rng_state=False,
        )
        model.blocks[layer_id] = wrapped


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    for block in model.blocks:
        fully_shard(block.attn1, **fsdp_config)
        fully_shard(block.attn2, **fsdp_config)
        fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
