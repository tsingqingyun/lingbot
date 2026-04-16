#!/bin/bash

save_root=${1:-"./results_robocasa"}
env_id=${2:-"robocasa/PickPlaceCounterToCabinet"}
n_episodes=${3:-1}
dataset_base_path=${4:-${ROBOCASA_DATASET_BASE_PATH:-""}}
PORT=${PORT:-29056}
seed=${SEED:-0}
DISABLE_ENV_CHECKER=${DISABLE_ENV_CHECKER:-1}

extra_dataset_args=()
if [ -n "${dataset_base_path}" ]; then
  extra_dataset_args+=(--dataset_base_path "${dataset_base_path}")
fi
if [ "${DISABLE_ENV_CHECKER}" = "1" ]; then
  extra_dataset_args+=(--disable_env_checker)
else
  extra_dataset_args+=(--enable_env_checker)
fi

python -m evaluation.robocasa.eval_policy_client_openpi \
  --save_root "${save_root}" \
  --env_id "${env_id}" \
  --n_episodes "${n_episodes}" \
  --seed "${seed}" \
  --port "${PORT}" \
  "${extra_dataset_args[@]}" \
  --video_guidance_scale 5 \
  --action_guidance_scale 1
