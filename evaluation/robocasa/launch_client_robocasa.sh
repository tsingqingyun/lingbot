#!/bin/bash

save_root=${1:-"./results_robocasa"}
env_id=${2:-"robocasa/PickPlaceCounterToCabinet"}
n_episodes=${3:-1}
PORT=${PORT:-29056}
seed=${SEED:-0}

python -m evaluation.robocasa.eval_policy_client_openpi \
  --save_root "${save_root}" \
  --env_id "${env_id}" \
  --n_episodes "${n_episodes}" \
  --seed "${seed}" \
  --port "${PORT}" \
  --video_guidance_scale 5 \
  --action_guidance_scale 1
