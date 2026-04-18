#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import imageio
import numpy as np
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
from wan_va.configs import get_config
from wan_va.dataset.lerobot_latent_dataset import lingbot_to_robocasa, robocasa_to_lingbot


DEFAULT_OBS_KEY_CANDIDATES = {
    "observation.images.robot0_agentview_left": (
        "observation.images.robot0_agentview_left",
        "video.robot0_agentview_left",
        "robot0_agentview_left_image",
        "agentview_left_image",
    ),
    "observation.images.robot0_agentview_right": (
        "observation.images.robot0_agentview_right",
        "video.robot0_agentview_right",
        "robot0_agentview_right_image",
        "agentview_right_image",
    ),
    "observation.images.robot0_eye_in_hand": (
        "observation.images.robot0_eye_in_hand",
        "video.robot0_eye_in_hand",
        "robot0_eye_in_hand_image",
        "robot0_wrist_image",
        "eye_in_hand_image",
    ),
}

def _infer_used_action_channel_ids() -> List[int]:
    # Keep inference-side unpacking strictly aligned with dataset mapping logic.
    _, mask_30 = robocasa_to_lingbot(np.zeros((1, 12), dtype=np.float32))
    if hasattr(mask_30, "detach"):
        mask_30 = mask_30.detach().cpu().numpy()
    mask_30 = np.asarray(mask_30, dtype=bool)
    channel_ids = np.flatnonzero(mask_30[0]).astype(np.int64).tolist()
    if len(channel_ids) == 0:
        raise RuntimeError("Failed to infer used RoboCasa action channels from robocasa_to_lingbot.")
    return channel_ids


USED_ACTION_CHANNEL_IDS = _infer_used_action_channel_ids()
CONTROL_MODE_COLLAPSE_EPS = 1e-6
CONTROL_MODE_COLLAPSE_STREAK_THRESHOLD = 4
CONTROL_MODE_REPLAN_COOLDOWN_STEPS = 32
ROBOCASA_ACTION_PER_FRAME = int(get_config("robocasa").action_per_frame)


def _find_obs_value(obs: Dict, candidates: Iterable[str]):
    for key in candidates:
        if key in obs:
            return obs[key]
    return None


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            vmax = float(np.max(arr)) if arr.size > 0 else 1.0
            if vmax <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def format_obs_for_lingbot(obs: Dict) -> Dict:
    out = {}
    for target_key, candidates in DEFAULT_OBS_KEY_CANDIDATES.items():
        value = _find_obs_value(obs, candidates)
        if value is None:
            available = sorted(list(obs.keys()))
            raise KeyError(
                f"Missing camera key for {target_key}. candidates={list(candidates)}, "
                f"available(first 40/{len(available)})={available[:40]}"
            )
        out[target_key] = _to_hwc_uint8(value)
    return out


def used_channels_to_action30(action_used: np.ndarray) -> np.ndarray:
    action_used = np.asarray(action_used, dtype=np.float32)
    if action_used.shape[-1] != len(USED_ACTION_CHANNEL_IDS):
        raise ValueError(
            f"Expected used action dim={len(USED_ACTION_CHANNEL_IDS)}, got shape={action_used.shape}"
        )
    action_30 = np.zeros((action_used.shape[0], 30), dtype=np.float32)
    action_30[:, USED_ACTION_CHANNEL_IDS] = action_used
    return action_30


def robocasa_action12_to_gym_dict(action_12: np.ndarray) -> Dict[str, np.ndarray]:
    """RoboCasa 12D flat action -> ``spaces.Dict`` keys for ``RoboCasaGymEnv.step``."""
    a, _ = sanitize_robocasa_action12(action_12)
    if a.shape[0] != 12:
        raise ValueError(f"Expected RoboCasa action dim 12, got shape {getattr(action_12, 'shape', None)}")
    # Same layout as ``wan_va.dataset.lerobot_latent_dataset.robocasa_to_lingbot``.
    return {
        "action.end_effector_position": a[0:3].copy(),
        "action.end_effector_rotation": a[3:6].copy(),
        "action.gripper_close": a[6:7].copy(),
        "action.base_motion": a[7:11].copy(),
        "action.control_mode": a[11:12].copy(),
    }


def sanitize_robocasa_action12(action_12: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Clamp to RoboCasa gym action bounds and drop NaN/Inf before env.step."""
    raw = np.asarray(action_12, dtype=np.float32).reshape(-1)
    if raw.shape[0] != 12:
        raise ValueError(f"Expected RoboCasa action dim 12, got shape {getattr(action_12, 'shape', None)}")
    finite = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = np.clip(finite, -1.0, 1.0).astype(np.float32, copy=False)
    changed = not np.allclose(raw, clipped, atol=1e-6, rtol=0.0, equal_nan=True)
    return clipped, changed


def summarize_episode_action_stats(executed_actions: List[np.ndarray], clipped_steps: int) -> Dict[str, Any]:
    if len(executed_actions) == 0:
        return {
            "executed_action_steps": 0,
            "clipped_action_steps": int(clipped_steps),
            "clipped_action_ratio": 0.0,
        }

    arr = np.asarray(executed_actions, dtype=np.float32).reshape(-1, 12)
    base = arr[:, 7:11]
    mode = arr[:, 11]

    base_absdiff = np.abs(np.diff(base, axis=0)) if arr.shape[0] > 1 else np.zeros((0, 4), dtype=np.float32)
    mode_change_rate = float(np.mean(mode[1:] != mode[:-1])) if mode.shape[0] > 1 else 0.0

    stats: Dict[str, Any] = {
        "executed_action_steps": int(arr.shape[0]),
        "clipped_action_steps": int(clipped_steps),
        "clipped_action_ratio": float(clipped_steps / max(1, arr.shape[0])),
        "mode_change_rate": mode_change_rate,
        "mode_positive_ratio": float(np.mean(mode > 0.5)),
        "base_mean": base.mean(axis=0).astype(np.float64).tolist(),
        "base_std": base.std(axis=0).astype(np.float64).tolist(),
    }
    if base_absdiff.shape[0] > 0:
        stats["base_absdiff_mean"] = base_absdiff.mean(axis=0).astype(np.float64).tolist()
        stats["base_absdiff_p95"] = np.quantile(base_absdiff, 0.95, axis=0).astype(np.float64).tolist()
    else:
        stats["base_absdiff_mean"] = [0.0, 0.0, 0.0, 0.0]
        stats["base_absdiff_p95"] = [0.0, 0.0, 0.0, 0.0]
    return stats


def summarize_chunk_prediction(
    pred: np.ndarray,
    action_30_batch: np.ndarray,
    action_12_seq: np.ndarray,
    step_count_before_chunk: int,
    first_chunk: bool,
) -> Dict[str, Any]:
    action_30_seq = action_30_batch.reshape(pred.shape[1], pred.shape[2], 30)
    gate_raw = action_30_seq[:, :, 29]
    gripper_raw = action_30_seq[:, :, 14]
    base = action_12_seq[:, :, 7:11].reshape(-1, 4)
    control_mode = action_12_seq[:, :, 11].reshape(-1)
    gripper_exec = action_12_seq[:, :, 6].reshape(-1)
    start_idx = 1 if first_chunk else 0
    executable_steps = max(0, pred.shape[2] * max(0, pred.shape[1] - start_idx))

    base_absdiff = (
        np.abs(np.diff(base, axis=0)) if base.shape[0] > 1 else np.zeros((0, 4), dtype=np.float32)
    )
    chunk_stats: Dict[str, Any] = {
        "step_start": int(step_count_before_chunk),
        "step_end_exclusive": int(step_count_before_chunk + executable_steps),
        "pred_frames": int(pred.shape[1]),
        "actions_per_frame": int(pred.shape[2]),
        "first_chunk": bool(first_chunk),
        "control_mode_raw_mean": float(gate_raw.mean()),
        "control_mode_raw_min": float(gate_raw.min()),
        "control_mode_raw_max": float(gate_raw.max()),
        "control_mode_raw_positive_ratio": float(np.mean(gate_raw > 0.5)),
        "control_mode_exec_positive_ratio": float(np.mean(control_mode > 0.5)),
        "gripper_raw_positive_ratio": float(np.mean(gripper_raw > 0.5)),
        "gripper_exec_positive_ratio": float(np.mean(gripper_exec > 0.5)),
        "base_mean": base.mean(axis=0).astype(np.float64).tolist(),
        "base_std": base.std(axis=0).astype(np.float64).tolist(),
    }
    if base_absdiff.shape[0] > 0:
        chunk_stats["base_absdiff_mean"] = base_absdiff.mean(axis=0).astype(np.float64).tolist()
        chunk_stats["base_absdiff_p95"] = np.quantile(base_absdiff, 0.95, axis=0).astype(np.float64).tolist()
    else:
        chunk_stats["base_absdiff_mean"] = [0.0, 0.0, 0.0, 0.0]
        chunk_stats["base_absdiff_p95"] = [0.0, 0.0, 0.0, 0.0]
    return chunk_stats


def infer_success(info: Dict, terminated: bool) -> bool:
    for key in ("success", "task_success", "is_success", "episode_success"):
        if key in info:
            return bool(info[key])
    if "metrics" in info and isinstance(info["metrics"], dict):
        for key in ("success", "task_success"):
            if key in info["metrics"]:
                return bool(info["metrics"][key])
    return bool(terminated)


def create_env(env_id: str, split: str, seed: Optional[int], render_mode: Optional[str]):
    import gymnasium as gym
    import robocasa  # noqa: F401  # import required to register robocasa/* env IDs with gymnasium

    kwargs = {"split": split}
    if seed is not None:
        kwargs["seed"] = seed
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, **kwargs)


def _normalize_dataset_base_path(dataset_base_path: str) -> str:
    """
    RoboCasa registry paths already include "v1.0/...".
    If user provides ".../v1.0", convert to its parent to avoid duplicated "v1.0/v1.0".
    """
    p = Path(dataset_base_path).expanduser().resolve()
    if p.name == "v1.0":
        return str(p.parent)
    return str(p)


def configure_robocasa_dataset_path(dataset_base_path: Optional[str]) -> Optional[str]:
    """
    Priority:
      1) explicit --dataset_base_path
      2) env ROBOCASA_DATASET_BASE_PATH
      3) keep RoboCasa default behavior
    """
    candidate = dataset_base_path or os.environ.get("ROBOCASA_DATASET_BASE_PATH")
    if not candidate:
        return None

    normalized = _normalize_dataset_base_path(candidate)
    import robocasa.macros as macros

    macros.DATASET_BASE_PATH = normalized
    return normalized


def _prompt_from_value(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _prompt_from_value(value.item())
        if value.size == 0:
            return None
        if value.dtype.kind in ("U", "S", "O"):
            return _prompt_from_value(value.reshape(-1)[0])
        return None
    return None


def _prompt_from_reset_obs(reset_obs: Dict) -> Optional[str]:
    if not isinstance(reset_obs, dict):
        return None
    for key in (
        "annotation.human.task_description",
        "language",
        "task",
    ):
        text = _prompt_from_value(reset_obs.get(key))
        if text:
            return text
    return None


def _prompt_from_env_meta(env) -> Optional[str]:
    for candidate in (getattr(env, "unwrapped", None), env):
        if candidate is None:
            continue
        getter = getattr(candidate, "get_ep_meta", None)
        if not callable(getter):
            continue
        try:
            ep_meta = getter()
        except Exception:
            continue
        if isinstance(ep_meta, dict):
            text = _prompt_from_value(ep_meta.get("lang"))
            if text:
                return text
    return None


def resolve_episode_prompt(explicit_prompt: str, env, reset_obs: Dict, env_id: str) -> Tuple[str, str]:
    text = _prompt_from_value(explicit_prompt)
    if text:
        return text, "args.prompt"

    text = _prompt_from_reset_obs(reset_obs)
    if text:
        return text, "reset_obs.annotation.human.task_description"

    text = _prompt_from_env_meta(env)
    if text:
        return text, "env.get_ep_meta().lang"

    text = _prompt_from_value(getattr(getattr(env, "unwrapped", env), "instruction", None))
    if text:
        return text, "env.unwrapped.instruction"

    return env_id, "env_id_fallback"


def run_episode(
    env,
    model: WebsocketClientPolicy,
    explicit_prompt: str,
    env_id: str,
    max_steps: int,
    video_guidance_scale: float,
    action_guidance_scale: float,
) -> Tuple[bool, int, List[np.ndarray], str, Dict[str, Any]]:
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, reset_info = reset_out
    else:
        obs, reset_info = reset_out, {}

    episode_prompt, prompt_source = resolve_episode_prompt(
        explicit_prompt=explicit_prompt,
        env=env,
        reset_obs=obs,
        env_id=env_id,
    )
    if prompt_source == "env_id_fallback":
        print(
            "[Warn] Could not find language prompt in args/reset_obs/ep_meta/instruction; "
            f"fallback prompt uses env_id='{env_id}'."
        )

    frame = format_obs_for_lingbot(obs)
    frames_for_video = [frame["observation.images.robot0_agentview_left"]]

    model.infer(dict(reset=True, prompt=episode_prompt))

    done = False
    success = False
    step_count = 0
    first = True
    executed_actions: List[np.ndarray] = []
    clipped_action_steps = 0
    chunk_diagnostics: List[Dict[str, Any]] = []
    mode_collapse_streak = 0
    replan_count = 0
    last_replan_step = -CONTROL_MODE_REPLAN_COOLDOWN_STEPS

    while (not done) and (step_count < max_steps):
        server_obs = frame if first else next_frame
        infer_obs = server_obs
        if first:
            # The server-side streaming VAE can enter a cached temporal branch where
            # kernel_size=3 effectively requires >=2 fresh frames in this request.
            infer_obs = [copy.deepcopy(server_obs) for _ in range(2)]
        ret = model.infer(
            dict(
                obs=infer_obs,
                prompt=episode_prompt,
                video_guidance_scale=video_guidance_scale,
                action_guidance_scale=action_guidance_scale,
            )
        )
        pred = np.asarray(ret["action"], dtype=np.float32)
        if pred.ndim != 3:
            raise ValueError(
                f"Expected action shape [C,F,H] where C=used action channels, "
                f"F=predicted frame chunks, H=actions per frame, and C={len(USED_ACTION_CHANNEL_IDS)}; "
                f"got {pred.shape}"
            )
        if pred.shape[2] != ROBOCASA_ACTION_PER_FRAME:
            raise ValueError(
                "RoboCasa action contract mismatch: "
                f"server returned action_per_frame={pred.shape[2]}, "
                f"but config expects {ROBOCASA_ACTION_PER_FRAME}."
            )

        action_used_batch = pred.transpose(1, 2, 0).reshape(-1, pred.shape[0])
        action_30_batch = used_channels_to_action30(action_used_batch)
        action_12_batch = lingbot_to_robocasa(action_30_batch)
        action_12_seq = action_12_batch.reshape(pred.shape[1], pred.shape[2], 12)
        chunk_stats = summarize_chunk_prediction(
            pred=pred,
            action_30_batch=action_30_batch,
            action_12_seq=action_12_seq,
            step_count_before_chunk=step_count,
            first_chunk=first,
        )
        chunk_diagnostics.append(chunk_stats)
        mode_ratio = chunk_stats["control_mode_exec_positive_ratio"]
        if (
            mode_ratio <= CONTROL_MODE_COLLAPSE_EPS
            or mode_ratio >= 1.0 - CONTROL_MODE_COLLAPSE_EPS
        ):
            mode_collapse_streak += 1
        else:
            mode_collapse_streak = 0
        if mode_collapse_streak >= CONTROL_MODE_COLLAPSE_STREAK_THRESHOLD:
            print(
                "[Warn] control_mode collapse persists for "
                f"{mode_collapse_streak} chunks: "
                f"step_start={chunk_stats['step_start']} "
                f"mode_ratio={mode_ratio:.4f} "
                f"gate_raw_mean={chunk_stats['control_mode_raw_mean']:.4f} "
                f"base_absdiff_mean={chunk_stats['base_absdiff_mean']}"
            )

        key_frame_list = []
        start_idx = 1 if first else 0
        for i in range(start_idx, action_12_seq.shape[0]):
            stepped = False
            for j in range(action_12_seq.shape[1]):
                stepped = True
                raw_action_12 = action_12_seq[i, j]
                action_12, clipped = sanitize_robocasa_action12(raw_action_12)
                clipped_action_steps += int(clipped)
                executed_actions.append(action_12.copy())
                step_out = env.step(robocasa_action12_to_gym_dict(action_12))
                if len(step_out) == 5:
                    obs, _, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                    success = infer_success(info, bool(terminated))
                else:
                    obs, _, done, info = step_out
                    done = bool(done)
                    success = infer_success(info, done)

                step_count += 1
                frame = format_obs_for_lingbot(obs)
                frames_for_video.append(frame["observation.images.robot0_agentview_left"])

                if done or step_count >= max_steps:
                    break
            # One lingbot-format obs per chunk frame (after j sub-steps for index i).
            if stepped:
                key_frame_list.append(frame)

            if done or step_count >= max_steps:
                break

        should_replan = (
            (not done)
            and mode_collapse_streak >= CONTROL_MODE_COLLAPSE_STREAK_THRESHOLD
            and (step_count - last_replan_step) >= CONTROL_MODE_REPLAN_COOLDOWN_STEPS
        )
        if should_replan:
            print(
                "[Info] Triggering replanning after persistent control_mode collapse: "
                f"step={step_count} "
                f"streak={mode_collapse_streak} "
                f"gate_raw_mean={chunk_stats['control_mode_raw_mean']:.4f}"
            )
            model.infer(dict(reset=True, prompt=episode_prompt))
            replan_count += 1
            last_replan_step = step_count
            mode_collapse_streak = 0
            next_frame = frame
            first = True
            continue

        if key_frame_list:
            # Warm VAE caches can hit multiple temporal downsample stages.
            # In practice this path needs >=4 frames for robust conv3d(k=3) validity.
            min_kv_cache_frames = max(4, pred.shape[1])
            cache_frames = list(key_frame_list)
            if len(cache_frames) < min_kv_cache_frames:
                padding_count = min_kv_cache_frames - len(cache_frames)
                cache_frames = [copy.deepcopy(server_obs) for _ in range(padding_count)] + cache_frames

            model.infer(
                dict(
                    obs=cache_frames,
                    compute_kv_cache=True,
                    imagine=False,
                    state=np.asarray(pred, dtype=np.float32),
                )
            )
            next_frame = cache_frames[-1]
        else:
            next_frame = frame
        first = False

    action_stats = summarize_episode_action_stats(executed_actions, clipped_action_steps)
    action_stats["chunk_diagnostics"] = chunk_diagnostics
    action_stats["replan_count"] = int(replan_count)
    action_stats["control_mode_collapse_streak_threshold"] = int(
        CONTROL_MODE_COLLAPSE_STREAK_THRESHOLD
    )
    action_stats["mode_collapse_warning"] = bool(
        any(
            diag["control_mode_exec_positive_ratio"] <= CONTROL_MODE_COLLAPSE_EPS
            or diag["control_mode_exec_positive_ratio"] >= 1.0 - CONTROL_MODE_COLLAPSE_EPS
            for diag in chunk_diagnostics
        )
    )
    return success, step_count, frames_for_video, episode_prompt, action_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", type=str, default="pretrain")
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--save_root", type=str, default="results_robocasa")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        default=None,
        help=(
            "RoboCasa dataset base path. Can be either '<base>' (contains v1.0/) "
            "or '<base>/v1.0'. If omitted, tries ROBOCASA_DATASET_BASE_PATH."
        ),
    )
    parser.add_argument("--video_guidance_scale", type=float, default=5.0)
    parser.add_argument("--action_guidance_scale", type=float, default=1.0)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--render_mode", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    resolved_dataset_base_path = configure_robocasa_dataset_path(args.dataset_base_path)
    if resolved_dataset_base_path is not None:
        print(f"[Info] RoboCasa DATASET_BASE_PATH set to: {resolved_dataset_base_path}")

    save_root = Path(args.save_root)
    metrics_dir = save_root / "metrics"
    video_dir = save_root / "visualization"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    model = WebsocketClientPolicy(port=args.port)

    episode_metrics = []
    succ = 0

    for ep in range(args.n_episodes):
        env_seed = args.seed + ep
        env = create_env(args.env_id, split=args.split, seed=env_seed, render_mode=args.render_mode)
        try:
            ok, steps, frames, episode_prompt, action_stats = run_episode(
                env=env,
                model=model,
                explicit_prompt=args.prompt,
                env_id=args.env_id,
                max_steps=args.max_steps,
                video_guidance_scale=args.video_guidance_scale,
                action_guidance_scale=args.action_guidance_scale,
            )
        finally:
            env.close()

        succ += int(ok)
        record = {
            "episode_id": ep,
            "seed": env_seed,
            "success": bool(ok),
            "steps": int(steps),
            "env_id": args.env_id,
            "prompt": episode_prompt,
            "action_stats": action_stats,
        }
        episode_metrics.append(record)

        if args.save_video and len(frames) > 0:
            status = "success" if ok else "fail"
            video_path = video_dir / f"ep_{ep:03d}_seed_{env_seed}_{status}.mp4"
            imageio.mimsave(video_path, frames, fps=15)

        print(
            f"[Episode {ep + 1}/{args.n_episodes}] "
            f"success={ok} steps={steps} "
            f"mode_change_rate={action_stats.get('mode_change_rate', 0.0):.4f} "
            f"clip_ratio={action_stats.get('clipped_action_ratio', 0.0):.4f} "
            f"running_sr={succ / (ep + 1):.4f}"
        )

    out = {
        "env_id": args.env_id,
        "split": args.split,
        "n_episodes": args.n_episodes,
        "success_num": succ,
        "success_rate": (succ / args.n_episodes) if args.n_episodes > 0 else 0.0,
        "episodes": episode_metrics,
    }
    if resolved_dataset_base_path is not None:
        out["dataset_base_path"] = resolved_dataset_base_path
    out_file = metrics_dir / "res.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {out_file}")


if __name__ == "__main__":
    main()
