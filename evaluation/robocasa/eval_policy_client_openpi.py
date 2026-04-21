#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import imageio
import numpy as np
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
from wan_va.configs import get_config
from wan_va.dataset.lerobot_latent_dataset import (
    get_robocasa_binarize_thresholds,
    lingbot_to_robocasa,
    robocasa_to_lingbot,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
except ModuleNotFoundError:
    plt = None
    FigureCanvas = None


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
BIN_THRESHOLDS = get_robocasa_binarize_thresholds()
ROBOCASA_ACTION_PER_FRAME = int(get_config("robocasa").action_per_frame)
CONTROL_MODE_COLLAPSE_EPS = 1e-6
CONTROL_MODE_COLLAPSE_STREAK_THRESHOLD = 4
CONTROL_MODE_REPLAN_COOLDOWN_STEPS = 32


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


def add_title_bar(img: np.ndarray, text: str, font_scale: float = 0.8, thickness: int = 2) -> np.ndarray:
    """Add a black title bar with centered text above an image."""
    h, w, _ = img.shape
    del h
    bar_height = 40
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = max(0, (w - text_w) // 2)
    text_y = (bar_height + text_h) // 2 - 5
    cv2.putText(
        title_bar,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return np.vstack([title_bar, img])


def visualize_action_step(action_history: List[np.ndarray], step_idx: int, window: int = 64) -> np.ndarray:
    """Render a compact diagnostic chart for RoboCasa 12D actions."""
    if plt is None or FigureCanvas is None:
        return visualize_action_step_fallback(action_history, step_idx, window=window)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=100, sharex=True)
    ax1, ax2, ax3, ax4 = axes.reshape(-1)

    start = max(0, step_idx - window)
    end = step_idx + 1
    history_subset = np.asarray(action_history[start:end], dtype=np.float32)
    x_axis = range(start, start + len(history_subset))

    if history_subset.size > 0 and history_subset.shape[1] >= 12:
        ax1.plot(x_axis, history_subset[:, 0], label="base_x", color="r", linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 1], label="base_y", color="g", linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 2], label="base_yaw", color="b", linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 3], label="torso_z", color="orange", linewidth=1.5)
        ax1.set_ylabel("Base")
        ax1.legend(loc="upper right", fontsize="x-small", ncol=4)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Step {step_idx}: Base / Torso")

        ax2.plot(x_axis, history_subset[:, 5], label="eef_x", color="r", linewidth=1.5)
        ax2.plot(x_axis, history_subset[:, 6], label="eef_y", color="g", linewidth=1.5)
        ax2.plot(x_axis, history_subset[:, 7], label="eef_z", color="b", linewidth=1.5)
        ax2.set_ylabel("EEF Pos")
        ax2.legend(loc="upper right", fontsize="x-small", ncol=3)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("End Effector Position")

        ax3.plot(x_axis, history_subset[:, 8], label="rot_x", color="c", linewidth=1.5)
        ax3.plot(x_axis, history_subset[:, 9], label="rot_y", color="m", linewidth=1.5)
        ax3.plot(x_axis, history_subset[:, 10], label="rot_z", color="y", linewidth=1.5)
        ax3.set_ylabel("Axis-angle")
        ax3.legend(loc="upper right", fontsize="x-small", ncol=3)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("End Effector Rotation")

        ax4.plot(x_axis, history_subset[:, 4], label="control_mode", color="purple", linewidth=1.5)
        ax4.plot(x_axis, history_subset[:, 11], label="gripper", color="orange", linewidth=1.5)
        ax4.set_ylabel("Binary")
        ax4.legend(loc="upper right", fontsize="x-small", ncol=2)
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Control Mode / Gripper")

    ax1.set_xlim(max(0, step_idx - window), max(window, step_idx))
    ax3.set_xlabel("Step")
    ax4.set_xlabel("Step")

    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return _to_hwc_uint8(img)


def visualize_action_step_fallback(
    action_history: List[np.ndarray],
    step_idx: int,
    window: int = 64,
) -> np.ndarray:
    """Fallback diagnostic panel when matplotlib is unavailable."""
    panel_h, panel_w = 420, 1280
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    if not action_history:
        cv2.putText(
            panel,
            "Action history unavailable",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return panel

    start = max(0, step_idx - window)
    end = min(len(action_history), step_idx + 1)
    history_subset = np.asarray(action_history[start:end], dtype=np.float32)
    latest = history_subset[-1]
    mean = history_subset.mean(axis=0)
    std = history_subset.std(axis=0)

    cv2.putText(
        panel,
        f"RoboCasa Action Summary  steps[{start}:{end}) current={min(step_idx, len(action_history) - 1)}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    lines = [
        f"base_motion  cur={np.array2string(latest[0:4], precision=3, suppress_small=True)}",
        f"             avg={np.array2string(mean[0:4], precision=3, suppress_small=True)}  std={np.array2string(std[0:4], precision=3, suppress_small=True)}",
        f"control_mode cur={latest[4]: .3f}  avg={mean[4]: .3f}  std={std[4]: .3f}",
        f"eef_pos      cur={np.array2string(latest[5:8], precision=3, suppress_small=True)}",
        f"             avg={np.array2string(mean[5:8], precision=3, suppress_small=True)}  std={np.array2string(std[5:8], precision=3, suppress_small=True)}",
        f"eef_rot      cur={np.array2string(latest[8:11], precision=3, suppress_small=True)}",
        f"             avg={np.array2string(mean[8:11], precision=3, suppress_small=True)}  std={np.array2string(std[8:11], precision=3, suppress_small=True)}",
        f"gripper      cur={latest[11]: .3f}  avg={mean[11]: .3f}  std={std[11]: .3f}",
    ]

    y = 95
    for line in lines:
        cv2.putText(
            panel,
            line,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        y += 42

    def draw_bar(label: str, value: float, row_y: int, color: Tuple[int, int, int]) -> None:
        x0, x1 = 260, 1180
        mid = (x0 + x1) // 2
        cv2.putText(
            panel,
            label,
            (40, row_y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.line(panel, (x0, row_y), (x1, row_y), (90, 90, 90), 2)
        cv2.line(panel, (mid, row_y - 12), (mid, row_y + 12), (140, 140, 140), 2)
        clipped = max(-1.0, min(1.0, float(value)))
        x_val = int(mid + clipped * (x1 - x0) / 2.0)
        cv2.circle(panel, (x_val, row_y), 8, color, -1)

    draw_bar("control_mode", latest[4], 315, (180, 120, 255))
    draw_bar("gripper", latest[11], 355, (0, 190, 255))
    draw_bar("base_yaw", latest[2], 395, (255, 200, 0))
    return panel


def save_comparison_video(
    real_obs_list: List[Dict[str, np.ndarray]],
    action_history: List[np.ndarray],
    save_path: Path,
    imagined_video: Optional[List[np.ndarray]] = None,
    fps: int = 15,
) -> None:
    """Save a RobotWin-style stitched diagnostic video for RoboCasa."""
    if not real_obs_list:
        return

    final_frames: List[np.ndarray] = []
    imagined_frames: Optional[np.ndarray] = None
    imagined_count = 0
    if imagined_video:
        expanded_imagined: List[np.ndarray] = []
        for chunk_idx, chunk in enumerate(imagined_video):
            chunk_arr = np.asarray(chunk)
            if chunk_arr.ndim == 0 or chunk_arr.shape[0] == 0:
                continue
            start_idx = 0
            if chunk_idx == 0:
                # The first returned frame is the conditioning observation rather than
                # a newly executed future frame, so align it with the initial real obs.
                expanded_imagined.append(chunk_arr[0])
                start_idx = 1
            for frame_idx in range(start_idx, int(chunk_arr.shape[0])):
                expanded_imagined.extend([chunk_arr[frame_idx]] * ROBOCASA_ACTION_PER_FRAME)
        if expanded_imagined:
            imagined_frames = np.asarray(expanded_imagined[: len(real_obs_list)])
            imagined_count = int(imagined_frames.shape[0])

    def resize_h(img: np.ndarray, height: int) -> np.ndarray:
        img = _to_hwc_uint8(img)
        if img.shape[0] != height:
            width = int(img.shape[1] * height / img.shape[0])
            img = cv2.resize(img, (width, height))
        return np.ascontiguousarray(img)

    def fit_to_canvas(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        img = _to_hwc_uint8(img)
        src_h, src_w = img.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)

        scale = min(target_width / src_w, target_height / src_h)
        resize_w = max(1, int(round(src_w * scale)))
        resize_h_ = max(1, int(round(src_h * scale)))
        resized = cv2.resize(img, (resize_w, resize_h_))

        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y0 = (target_height - resize_h_) // 2
        x0 = (target_width - resize_w) // 2
        canvas[y0:y0 + resize_h_, x0:x0 + resize_w] = resized
        return canvas

    for i, obs in enumerate(real_obs_list):
        cam_left = obs["observation.images.robot0_agentview_left"]
        cam_right = obs["observation.images.robot0_agentview_right"]
        cam_wrist = obs["observation.images.robot0_eye_in_hand"]
        base_h = cam_left.shape[0]

        row_real = np.hstack(
            [
                resize_h(cam_left, base_h),
                resize_h(cam_right, base_h),
                resize_h(cam_wrist, base_h),
            ]
        )
        row_real = add_title_bar(
            np.ascontiguousarray(row_real),
            "Real Observation (Agentview Left / Agentview Right / Eye In Hand)",
        )

        target_width = row_real.shape[1]
        imagined_body_height = 300
        if imagined_frames is not None and i < imagined_count:
            row_imagined = fit_to_canvas(imagined_frames[i], target_width, imagined_body_height)
        else:
            row_imagined = np.zeros((imagined_body_height, target_width, 3), dtype=np.uint8)
            text = "Imagined video not returned"
            cv2.putText(
                row_imagined,
                text,
                (max(20, target_width // 2 - 180), imagined_body_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (100, 100, 100),
                2,
                cv2.LINE_AA,
            )
        row_imagined = add_title_bar(np.ascontiguousarray(row_imagined), "Imagined Video Stream")

        action_frame = visualize_action_step(action_history, min(i, max(0, len(action_history) - 1)))
        action_frame = cv2.resize(action_frame, (row_real.shape[1], action_frame.shape[0]))
        action_frame = add_title_bar(action_frame, "Executed RoboCasa Action History")

        full_frame = np.vstack([row_real, row_imagined, action_frame])
        final_frames.append(np.ascontiguousarray(full_frame))

    imageio.mimsave(save_path, final_frames, fps=fps)


def save_imagined_video(imagined_video: List[np.ndarray], save_path: Path, fps: int = 15) -> None:
    """Optionally dump the imagined video stream as its own mp4."""
    if not imagined_video:
        return
    frames = np.concatenate(imagined_video, axis=0)
    frames = [_to_hwc_uint8(frame) for frame in frames]
    imageio.mimsave(save_path, frames, fps=fps)


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
        "action.base_motion": a[0:4].copy(),
        "action.control_mode": a[4:5].copy(),
        "action.end_effector_position": a[5:8].copy(),
        "action.end_effector_rotation": a[8:11].copy(),
        "action.gripper_close": a[11:12].copy(),
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
    base = arr[:, 0:4]
    mode = arr[:, 4]
    gripper = arr[:, 11]

    base_absdiff = np.abs(np.diff(base, axis=0)) if arr.shape[0] > 1 else np.zeros((0, 4), dtype=np.float32)
    mode_change_rate = float(np.mean(mode[1:] != mode[:-1])) if mode.shape[0] > 1 else 0.0

    stats: Dict[str, Any] = {
        "executed_action_steps": int(arr.shape[0]),
        "clipped_action_steps": int(clipped_steps),
        "clipped_action_ratio": float(clipped_steps / max(1, arr.shape[0])),
        "mode_change_rate": mode_change_rate,
        "mode_positive_ratio": float(np.mean(mode > 0.0)),
        "gripper_positive_ratio": float(np.mean(gripper > 0.0)),
        "binarize_thresholds": BIN_THRESHOLDS,
        "configured_action_per_frame": int(ROBOCASA_ACTION_PER_FRAME),
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


def executed_action12_seq_to_lingbot_state(action_12_seq: np.ndarray) -> np.ndarray:
    """Convert executed RoboCasa actions into the server's expected LingBot action state layout."""
    action_12_seq = np.asarray(action_12_seq, dtype=np.float32)
    if action_12_seq.ndim != 3 or action_12_seq.shape[-1] != 12:
        raise ValueError(
            f"Expected executed action shape [F,H,12], got {action_12_seq.shape}"
        )
    action_30_seq, _ = robocasa_to_lingbot(action_12_seq.reshape(-1, 12))
    action_30_seq = np.asarray(action_30_seq, dtype=np.float32).reshape(
        action_12_seq.shape[0], action_12_seq.shape[1], 30
    )
    return np.transpose(action_30_seq[:, :, USED_ACTION_CHANNEL_IDS], (2, 0, 1))


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
    start_idx = 1 if first_chunk else 0
    executable_steps = max(0, pred.shape[2] * max(0, pred.shape[1] - start_idx))
    exec_action_12 = action_12_seq[start_idx:]
    if exec_action_12.size > 0:
        base = exec_action_12[:, :, 0:4].reshape(-1, 4)
        control_mode = exec_action_12[:, :, 4].reshape(-1)
        gripper_exec = exec_action_12[:, :, 11].reshape(-1)
    else:
        base = np.zeros((0, 4), dtype=np.float32)
        control_mode = np.zeros((0,), dtype=np.float32)
        gripper_exec = np.zeros((0,), dtype=np.float32)

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
        "control_mode_raw_positive_ratio": float(
            np.mean(gate_raw > BIN_THRESHOLDS["control_mode"])
        ),
        "control_mode_exec_positive_ratio": float(np.mean(control_mode > 0.5)) if control_mode.size > 0 else 0.0,
        "gripper_raw_positive_ratio": float(
            np.mean(gripper_raw > BIN_THRESHOLDS["gripper"])
        ),
        "gripper_exec_positive_ratio": float(np.mean(gripper_exec > 0.5)) if gripper_exec.size > 0 else 0.0,
        "base_mean": base.mean(axis=0).astype(np.float64).tolist() if base.size > 0 else [0.0, 0.0, 0.0, 0.0],
        "base_std": base.std(axis=0).astype(np.float64).tolist() if base.size > 0 else [0.0, 0.0, 0.0, 0.0],
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


def seed_screen_check(
    model: WebsocketClientPolicy,
    explicit_prompt: str,
    env_id: str,
    split: str,
    seed: int,
    render_mode: Optional[str],
    warmup_steps: int,
    video_guidance_scale: float,
    action_guidance_scale: float,
    enable_control_mode_collapse_guard: bool,
) -> Tuple[bool, str]:
    """
    Filter obviously bad seeds before the real rollout.

    This screening now performs a short real model rollout against a temporary env:
    reset -> obs/prompt resolution -> websocket inference -> action execution.
    It is meant to catch broken seeds / env contracts / server rollout failures
    before the real evaluation episode starts.
    """
    env = None
    try:
        env = create_env(env_id, split=split, seed=seed, render_mode=render_mode)
        _, steps, _, episode_prompt, _, _, _ = run_episode(
            env=env,
            model=model,
            explicit_prompt=explicit_prompt,
            env_id=env_id,
            max_steps=max(0, warmup_steps),
            video_guidance_scale=video_guidance_scale,
            action_guidance_scale=action_guidance_scale,
            collect_video_obs=False,
            collect_imagined_video=False,
            enable_control_mode_collapse_guard=enable_control_mode_collapse_guard,
        )
        if steps < max(0, warmup_steps):
            return (
                False,
                f"model warmup terminated early at step {steps}/{max(0, warmup_steps)} "
                f"prompt={episode_prompt!r}",
            )
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()


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
    collect_video_obs: bool = True,
    collect_imagined_video: bool = True,
    enable_control_mode_collapse_guard: bool = True,
) -> Tuple[
    bool,
    int,
    List[Dict[str, np.ndarray]],
    str,
    Dict[str, Any],
    List[np.ndarray],
    List[np.ndarray],
]:
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
    video_obs_list = [copy.deepcopy(frame)] if collect_video_obs else []

    model.infer(
        dict(
            reset=True,
            prompt=episode_prompt,
            video_guidance_scale=video_guidance_scale,
            action_guidance_scale=action_guidance_scale,
        )
    )

    done = False
    success = False
    step_count = 0
    first = True
    executed_actions: List[np.ndarray] = []
    imagined_video_chunks: List[np.ndarray] = []
    clipped_action_steps = 0
    chunk_diagnostics: List[Dict[str, Any]] = []
    mode_collapse_streak = 0
    replan_count = 0
    last_replan_step = -CONTROL_MODE_REPLAN_COOLDOWN_STEPS
    has_video = False

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
                return_video=collect_imagined_video,
            )
        )
        if "video" in ret and ret["video"] is not None:
            has_video = True
            if collect_imagined_video:
                imagined_video_chunks.append(np.asarray(ret["video"]))
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
        if enable_control_mode_collapse_guard:
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
        executed_chunk_frames: List[np.ndarray] = []
        start_idx = 1 if first else 0
        for i in range(start_idx, action_12_seq.shape[0]):
            stepped = False
            executed_frame_actions: List[np.ndarray] = []
            for j in range(action_12_seq.shape[1]):
                stepped = True
                raw_action_12 = action_12_seq[i, j]
                action_12, clipped = sanitize_robocasa_action12(raw_action_12)
                clipped_action_steps += int(clipped)
                executed_actions.append(action_12.copy())
                executed_frame_actions.append(action_12.copy())
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
                if collect_video_obs:
                    video_obs_list.append(copy.deepcopy(frame))

                if done or step_count >= max_steps:
                    break
            # One lingbot-format obs per chunk frame (after j sub-steps for index i).
            if stepped:
                key_frame_list.append(frame)
                executed_chunk_frames.append(np.asarray(executed_frame_actions, dtype=np.float32))

            if done or step_count >= max_steps:
                break

        should_replan = (
            enable_control_mode_collapse_guard
            and (not done)
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

        if key_frame_list and (not done) and (step_count < max_steps):
            cache_state = executed_action12_seq_to_lingbot_state(
                np.asarray(executed_chunk_frames, dtype=np.float32)
            )
            model.infer(
                dict(
                    obs=key_frame_list,
                    compute_kv_cache=True,
                    imagine=False,
                    state=cache_state,
                )
            )
            next_frame = key_frame_list[-1]
        else:
            next_frame = frame
        first = False

    action_stats = summarize_episode_action_stats(executed_actions, clipped_action_steps)
    action_stats["chunk_diagnostics"] = chunk_diagnostics
    action_stats["replan_count"] = int(replan_count)
    action_stats["has_video"] = bool(has_video)
    action_stats["imagined_video_chunk_count"] = int(len(imagined_video_chunks))
    action_stats["imagined_video_frame_count"] = int(
        sum(int(np.asarray(chunk).shape[0]) for chunk in imagined_video_chunks)
    )
    action_stats["control_mode_collapse_streak_threshold"] = int(
        CONTROL_MODE_COLLAPSE_STREAK_THRESHOLD
    )
    action_stats["control_mode_collapse_guard_enabled"] = bool(enable_control_mode_collapse_guard)
    action_stats["mode_collapse_warning"] = bool(
        enable_control_mode_collapse_guard
        and any(
            diag["control_mode_exec_positive_ratio"] <= CONTROL_MODE_COLLAPSE_EPS
            or diag["control_mode_exec_positive_ratio"] >= 1.0 - CONTROL_MODE_COLLAPSE_EPS
            for diag in chunk_diagnostics
        )
    )
    return (
        success,
        step_count,
        video_obs_list,
        episode_prompt,
        action_stats,
        executed_actions,
        imagined_video_chunks,
    )


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
    parser.add_argument("--dump_imagined_video", action="store_true")
    seed_screen_group = parser.add_mutually_exclusive_group()
    seed_screen_group.add_argument(
        "--enable_seed_screen",
        dest="enable_seed_screen",
        action="store_true",
        help="Enable seed screening before each real rollout (default).",
    )
    seed_screen_group.add_argument(
        "--disable_seed_screen",
        dest="enable_seed_screen",
        action="store_false",
        help="Disable seed screening before each real rollout.",
    )
    parser.set_defaults(enable_seed_screen=True)
    parser.add_argument(
        "--disable_control_mode_collapse_guard",
        action="store_true",
        help="Disable control_mode collapse detection and automatic replanning.",
    )
    parser.add_argument("--seed_screen_warmup_steps", type=int, default=24)
    parser.add_argument("--max_seed_tries", type=int, default=0)
    parser.add_argument("--render_mode", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    enable_control_mode_collapse_guard = bool(not args.disable_control_mode_collapse_guard)
    enable_seed_screen = bool(args.enable_seed_screen)
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
    screened_out: List[Dict[str, Any]] = []
    ep = 0
    seed_cursor = int(args.seed)
    max_seed_tries = (
        int(args.max_seed_tries)
        if args.max_seed_tries > 0
        else max(args.n_episodes * 5, args.n_episodes + 16)
    )
    attempted_seeds = 0

    while ep < args.n_episodes and attempted_seeds < max_seed_tries:
        env_seed = seed_cursor
        seed_cursor += 1
        attempted_seeds += 1
        if enable_seed_screen:
            passed, reason = seed_screen_check(
                model=model,
                explicit_prompt=args.prompt,
                env_id=args.env_id,
                split=args.split,
                seed=env_seed,
                render_mode=args.render_mode,
                warmup_steps=args.seed_screen_warmup_steps,
                video_guidance_scale=args.video_guidance_scale,
                action_guidance_scale=args.action_guidance_scale,
                enable_control_mode_collapse_guard=enable_control_mode_collapse_guard,
            )
            if not passed:
                screened_out.append({"seed": env_seed, "reason": reason})
                print(f"[SeedScreen] skip seed={env_seed} reason={reason}")
                continue

        env = create_env(args.env_id, split=args.split, seed=env_seed, render_mode=args.render_mode)
        try:
            ok, steps, video_obs_list, episode_prompt, action_stats, executed_actions, imagined_video_chunks = run_episode(
                env=env,
                model=model,
                explicit_prompt=args.prompt,
                env_id=args.env_id,
                max_steps=args.max_steps,
                video_guidance_scale=args.video_guidance_scale,
                action_guidance_scale=args.action_guidance_scale,
                enable_control_mode_collapse_guard=enable_control_mode_collapse_guard,
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

        if args.save_video and len(video_obs_list) > 0:
            status = "success" if ok else "fail"
            video_path = video_dir / f"ep_{ep:03d}_seed_{env_seed}_{status}.mp4"
            save_comparison_video(
                real_obs_list=video_obs_list,
                action_history=executed_actions,
                save_path=video_path,
                imagined_video=imagined_video_chunks,
                fps=15,
            )
            if args.dump_imagined_video and imagined_video_chunks:
                imagined_path = video_dir / f"ep_{ep:03d}_seed_{env_seed}_{status}_imagined.mp4"
                save_imagined_video(imagined_video_chunks, imagined_path, fps=15)

        print(
            f"[Episode {ep + 1}/{args.n_episodes}] "
            f"success={ok} steps={steps} "
            f"mode_change_rate={action_stats.get('mode_change_rate', 0.0):.4f} "
            f"gripper_positive_ratio={action_stats.get('gripper_positive_ratio', 0.0):.4f} "
            f"has_video={action_stats.get('has_video', False)} "
            f"clip_ratio={action_stats.get('clipped_action_ratio', 0.0):.4f} "
            f"running_sr={succ / (ep + 1):.4f}"
        )
        ep += 1

    out = {
        "env_id": args.env_id,
        "split": args.split,
        "n_episodes": args.n_episodes,
        "success_num": succ,
        "success_rate": (succ / args.n_episodes) if args.n_episodes > 0 else 0.0,
        "episodes": episode_metrics,
        "seed_screening": {
            "enabled": bool(enable_seed_screen),
            "warmup_steps": int(args.seed_screen_warmup_steps),
            "attempted_seeds": int(attempted_seeds),
            "max_seed_tries": int(max_seed_tries),
            "screened_out": screened_out,
        },
        "control_mode_collapse_guard_enabled": bool(enable_control_mode_collapse_guard),
    }
    if ep < args.n_episodes:
        out["warning"] = (
            f"Only completed {ep}/{args.n_episodes} episodes before reaching "
            f"max_seed_tries={max_seed_tries}."
        )
    if resolved_dataset_base_path is not None:
        out["dataset_base_path"] = resolved_dataset_base_path
    out_file = metrics_dir / "res.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {out_file}")


if __name__ == "__main__":
    main()
