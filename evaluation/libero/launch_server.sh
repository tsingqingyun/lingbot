#!/bin/bash

START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
LINGBOT_LIBERO_MODEL_PATH=${LINGBOT_LIBERO_MODEL_PATH:-""}
LINGBOT_WAN_BASE_PATH=${LINGBOT_WAN_BASE_PATH:-""}

save_root=${SAVE_ROOT:-"visualization/"}
mkdir -p "${save_root}"

if [ -n "${LINGBOT_LIBERO_MODEL_PATH}" ]; then
  export LINGBOT_LIBERO_MODEL_PATH
fi
if [ -n "${LINGBOT_WAN_BASE_PATH}" ]; then
  export LINGBOT_WAN_BASE_PATH
fi

python3 -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "${MASTER_PORT}" \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port "${START_PORT}" \
    --save_root "${save_root}"
