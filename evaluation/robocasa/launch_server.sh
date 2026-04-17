START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
LINGBOT_ROBOCASA_MODEL_PATH=${LINGBOT_ROBOCASA_MODEL_PATH:-""}
LINGBOT_WAN_BASE_PATH=${LINGBOT_WAN_BASE_PATH:-""}

save_root='visualization/'
mkdir -p $save_root

# Optional:
#   export LINGBOT_ROBOCASA_MODEL_PATH=/path/to/checkpoint_step_xxx
#   export LINGBOT_WAN_BASE_PATH=/path/to/wan22_root
if [ -n "${LINGBOT_ROBOCASA_MODEL_PATH}" ]; then
  export LINGBOT_ROBOCASA_MODEL_PATH
fi
if [ -n "${LINGBOT_WAN_BASE_PATH}" ]; then
  export LINGBOT_WAN_BASE_PATH
fi

python3 -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robocasa \
    --port $START_PORT \
    --save_root $save_root


