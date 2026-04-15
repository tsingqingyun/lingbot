#!/usr/bin/bash
#
# 与 run_va_posttrain.sh 对齐：相同 torch.distributed.run 启动方式 + 相同 CONFIG_NAME/NGPU 环境变量习惯。
# 在仓库根目录 lingbot-va/ 下执行：
#   bash script/run_va_check.sh
#   CONFIG_NAME=robocasa_train SAVE_DIR=./debug_rc bash script/run_va_check.sh
# 额外参数会原样传给 check.py，例如：
#   bash script/run_va_check.sh --phases config,data --no_model
#

set -x

umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29531"}
LOG_RANK=${LOG_RANK:-"0"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"}
SAVE_DIR=${SAVE_DIR:-"./debug_check"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --local-ranks-filter=${LOG_RANK} \
    --master_port ${MASTER_PORT} \
    --tee 3 \
    check.py \
    --config-name "${CONFIG_NAME}" \
    --save_dir "${SAVE_DIR}" \
    ${overrides}
