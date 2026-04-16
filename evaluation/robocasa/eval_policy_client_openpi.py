#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio
import numpy as np

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
from wan_va.dataset.lerobot_latent_dataset import lingbot_to_robocasa


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

USED_ACTION_CHANNEL_IDS = list(range(0, 7)) + [28] + list(range(7, 14)) + [29]


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
    a = np.asarray(action_12, dtype=np.float32).reshape(-1)
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


def infer_success(info: Dict, terminated: bool) -> bool:
    for key in ("success", "task_success", "is_success", "episode_success"):
        if key in info:
            return bool(info[key])
    if "metrics" in info and isinstance(info["metrics"], dict):
        for key in ("success", "task_success"):
            if key in info["metrics"]:
                return bool(info["metrics"][key])
    return bool(terminated)


def create_env(
    env_id: str,
    split: str,
    seed: Optional[int],
    render_mode: Optional[str],
    disable_env_checker: bool = True,
):
    import gymnasium as gym
    import robocasa  # noqa: F401  # import required to register robocasa/* env IDs with gymnasium

    kwargs = {"split": split}
    if seed is not None:
        kwargs["seed"] = seed
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, disable_env_checker=disable_env_checker, **kwargs)


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


def run_episode(
    env,
    model: WebsocketClientPolicy,
    prompt: str,
    max_steps: int,
    video_guidance_scale: float,
    action_guidance_scale: float,
) -> Tuple[bool, int, List[np.ndarray]]:
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, reset_info = reset_out
    else:
        obs, reset_info = reset_out, {}

    frame = format_obs_for_lingbot(obs)
    frames_for_video = [frame["observation.images.robot0_agentview_left"]]

    model.infer(dict(reset=True, prompt=prompt))

    done = False
    success = False
    step_count = 0
    first = True

    while (not done) and (step_count < max_steps):
        server_obs = frame if first else next_frame
        ret = model.infer(
            dict(
                obs=server_obs,
                prompt=prompt,
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

        action_used_batch = pred.transpose(1, 2, 0).reshape(-1, pred.shape[0])
        action_30_batch = used_channels_to_action30(action_used_batch)
        action_12_batch = lingbot_to_robocasa(action_30_batch)
        action_12_seq = action_12_batch.reshape(pred.shape[1], pred.shape[2], 12)

        key_frame_list = []
        start_idx = 1 if first else 0
        for i in range(start_idx, action_12_seq.shape[0]):
            stepped = False
            for j in range(action_12_seq.shape[1]):
                stepped = True
                action_12 = action_12_seq[i, j]
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

        if key_frame_list:
            min_kv_cache_frames = pred.shape[1]
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

    return success, step_count, frames_for_video


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
    parser.add_argument(
        "--disable_env_checker",
        dest="disable_env_checker",
        action="store_true",
        default=True,
        help="Disable Gymnasium passive env checker (default: enabled for compatibility).",
    )
    parser.add_argument(
        "--enable_env_checker",
        dest="disable_env_checker",
        action="store_false",
        help="Enable Gymnasium passive env checker for strict env-space validation.",
    )
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
        env = create_env(
            args.env_id,
            split=args.split,
            seed=env_seed,
            render_mode=args.render_mode,
            disable_env_checker=args.disable_env_checker,
        )
        try:
            if args.prompt:
                episode_prompt = args.prompt
            else:
                episode_prompt = getattr(env.unwrapped, "instruction", None)
                if not episode_prompt:
                    print(
                        "[Warn] env.unwrapped.instruction is missing; "
                        f"fallback prompt uses env_id='{args.env_id}'."
                    )
                    episode_prompt = args.env_id
            ok, steps, frames = run_episode(
                env=env,
                model=model,
                prompt=episode_prompt,
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
        }
        episode_metrics.append(record)

        if args.save_video and len(frames) > 0:
            status = "success" if ok else "fail"
            video_path = video_dir / f"ep_{ep:03d}_seed_{env_seed}_{status}.mp4"
            imageio.mimsave(video_path, frames, fps=15)

        print(
            f"[Episode {ep + 1}/{args.n_episodes}] "
            f"success={ok} steps={steps} "
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
