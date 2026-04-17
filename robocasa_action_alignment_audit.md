# RoboCasa Action Alignment Audit Log

## Scope
- Objective: strict alignment of action dimensions and semantics between training and inference.
- Repos referenced: lingbot, robocasa, robosuite.
- Constraint from user: keep configured data paths unchanged.

## Iteration 1 - Static contract audit (2026-04-17)

### Checked files
- wan_va/dataset/lerobot_latent_dataset.py
- wan_va/configs/va_robocasa_cfg.py
- wan_va/configs/va_robocasa_train_cfg.py
- wan_va/wan_va_server.py
- evaluation/robocasa/eval_policy_client_openpi.py
- robocasa/robocasa/wrappers/gym_wrapper.py
- robocasa/robocasa/utils/env_utils.py
- robosuite/robosuite/controllers/config/robots/default_pandaomron.json

### Findings
- Mapping index contract is consistent for active channels:
  - active 30D indices: [0,1,2,3,4,5,6,14,15,16,17,22,29]
  - RoboCasa base mapping in 30D: x->15, y->16, yaw->22, torso->17.
- Wrapper action split semantics are consistent:
  - action.base_motion[0:3] -> robot0_base
  - action.base_motion[3:4] -> robot0_torso
  - action.control_mode threshold uses 0.5 in wrapper.
- Important semantic mismatch found in mapping code before fix:
  - lingbot_to_robocasa used threshold >0 for gripper and control_mode,
    while wrapper semantics use 0.5 boundary.

## Iteration 2 - Runtime evidence from saved rollout actions (2026-04-17)

### Data source
- visualization/real/robocasa/KettleBoiling_20260416_234302/actions_*.pt

### Method
- Replay server saved tensors through the same postprocess and mapping chain:
  - normalized 30D tensor -> denorm by q01/q99 -> used channels -> 30D restore -> lingbot_to_robocasa.
- Reconstruct executed action sequence with first-chunk start_idx rule matching eval client.

### Key observations
- Executed step count: 2512
- mode change rate (binary): 0.0271
- Base channels are nearly constant in this run:
  - base_x std ~ 0
  - base_y std ~ 0
  - base_yaw std ~ 0
  - torso std ~ 0
- Control channel (30D ch29 after denorm) is mostly negative and often out of nominal range:
  - mean -5.2302, std 2.8593, min -22.3750, max 5.6563
  - ratio in (0, 0.5] is non-zero; this is sensitive to threshold boundary choice.

### Cross-run quick scan
- Multiple recent experiment folders show near-zero base variance in saved action tensors.
- mode flip ratio typically ~0.03 to ~0.08 depending on run.

## Iteration 3 - Fixes applied (2026-04-17)

### Fix A: Align binary threshold semantics to wrapper
- File: wan_va/dataset/lerobot_latent_dataset.py
- Change:
  - gripper threshold changed from >0 to >0.5
  - control_mode threshold changed from >0 to >0.5
- Rationale:
  - Keep binary conversion boundary consistent with RoboCasa wrapper (<0.5 vs >=0.5).

### Fix B: Enforce action domain validity before env.step and log diagnostics
- File: evaluation/robocasa/eval_policy_client_openpi.py
- Change:
  - Added sanitize_robocasa_action12:
    - NaN/Inf handling
    - clip all 12 action dims to [-1, 1]
  - Added summarize_episode_action_stats:
    - mode_change_rate
    - mode_positive_ratio
    - base mean/std
    - base abs diff mean/p95
    - clipped_action_steps and ratio
  - Metrics now store action_stats per episode.
- Rationale:
  - Wrapper action space is [-1, 1] for each action key.
  - This prevents semantic out-of-range actions from entering env.step silently.
  - Provides per-episode audit evidence for continuous verification.

### Fix C: Repair alignment self-test logic
- File: check.py
- Change:
  - Roundtrip self-test now validates:
    - continuous dims with strict reconstruction error
    - binary dims (gripper/gate) with 0.5-threshold expectation
- Rationale:
  - Avoid false failures from treating discretized outputs as continuous.

## Validation after fixes

### Compile validation
- py_compile succeeded for:
  - wan_va/dataset/lerobot_latent_dataset.py
  - evaluation/robocasa/eval_policy_client_openpi.py
  - check.py

### Behavioral spot-check
- Threshold test result after fix:
  - gate_out for [-0.2, 0.2, 0.5, 0.6] -> [-1, -1, -1, 1]
  - gripper_out for [-0.2, 0.2, 0.5, 0.6] -> [-1, -1, -1, 1]
- check.py mapping self-test now returns ok=True.

## Remaining risk and next verification
- Remaining risk:
  - Base channels still appear nearly constant in saved model outputs across multiple runs,
    suggesting possible model-side underfitting/collapse on base dimensions rather than index mismatch.
- Next verification to run after next rollout:
  1. Read metrics/res.json and compare action_stats.mode_change_rate and clipped_action_ratio.
  2. Confirm base_absdiff_mean is no longer near all-zero in successful episodes.
  3. If base remains frozen, inspect training data distribution for base dims in the active dataset location and compare with checkpoint provenance.

## Iteration 4 - tmux server/client runtime fix and regression (2026-04-17)

### Runtime error reproduced
- Server log showed websocket request handling crash during client infer:
  - `ValueError: Default process group has not been initialized`
  - stack: `wan_va/utils/sever_utils.py` -> `distributed_infer` -> `dist.get_rank()`

### Root cause
- Single-GPU server path (`nproc_per_node=1`) can run without initialized torch.distributed default process group,
  but `distributed_infer` unconditionally called distributed APIs.

### Fix applied
- File: `wan_va/utils/sever_utils.py`
- Added `_dist_ready()` guard.
- `distributed_infer` now:
  - directly calls `model.infer(obs)` when process group is not initialized,
  - bypasses broadcast path when `world_size <= 1`.
- `worker_loop` and `run_async_server_mode` now guard distributed-only operations.

### tmux regression procedure
- Server session: `lingbot_server_test`
  - command: `START_PORT=29156 MASTER_PORT=29161 bash evaluation/robocasa/launch_server.sh`
- Client session: `lingbot_client_test`
  - command: `python -m evaluation.robocasa.eval_policy_client_openpi --env_id robocasa/PickPlaceCounterToCabinet --split pretrain --n_episodes 1 --max_steps 5 --port 29156 --save_root ./results_robocasa_tmux_test`
  - env: `ROBOCASA_DATASET_BASE_PATH=/cephfs/shared/xcxhx/robocasa_datasets_composite`

### Regression outcome
- Client finished and wrote metrics:
  - `results_robocasa_tmux_test/metrics/res.json`
  - `steps=5`, no runtime exception.
- Server no longer emitted `Default process group has not been initialized`.
- Observed warning only (non-fatal): numpy non-writable tensor conversion warning in KV cache path.
