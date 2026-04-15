#!/usr/bin/bash

set -x

umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29531"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # robotwin_train, robocasa_train
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"wan_va/configs/deepspeed/zero2_offload.json"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-lingbot_va}"

num_gpu=${NGPU}
master_port=${MASTER_PORT}
config_name=${CONFIG_NAME}
ds_config=${DEEPSPEED_CONFIG}

export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" deepspeed \
    --num_gpus=${num_gpu} \
    --master_port ${master_port} \
    -m wan_va.train \
    --config-name ${config_name} \
    --use-deepspeed \
    --deepspeed-config ${ds_config} \
    $overrides
