import argparse
import json
import os
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy


def save_video(
    real_obs_list,
    save_path,
    fps=15,
    video_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
):
    if not real_obs_list:
        print("No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"Video saved to: {save_path}")


def _write_json(data, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def construct_single_env(env_args):
    from libero.libero.envs import OffScreenRenderEnv

    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {
        "observation.images.agentview_rgb": agentview,
        "observation.images.eye_in_hand_rgb": eye_in_hand,
    }


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.0] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(model, libero_benchmark, task_idx, out_dir, episode_idx):
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: task id must be < {num_tasks}"

    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
        "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    if cur_env is None:
        return False

    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])
    model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    done = False
    first = True

    while cur_env.env.timestep < 800:
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = ret["action"]

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                observes, done = env_one_step(cur_env, ee_action)
                if done:
                    break
                if (j + 1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    key_frame_list.append(observes)
            if done:
                break

        first = False
        if done:
            break

        model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    out_file = (
        Path(out_dir)
        / libero_benchmark
        / f"{task_idx}_{prompt.replace(' ', '_')}"
        / f"{episode_idx}_{done}.mp4"
    )
    out_file.parent.mkdir(exist_ok=True, parents=True)
    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
    )
    cur_env.close()
    return bool(done)


def run(libero_benchmark, port, out_dir, test_num, task_range=None):
    from libero.libero import benchmark

    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f"task_range must be [start, end), got {task_range}"
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"Use benchmark: {libero_benchmark}, num_tasks: {num_tasks}")
    model = WebsocketClientPolicy(port=port)

    for task_idx in progress_bar:
        succ_num = 0.0
        for episode_idx in tqdm(range(test_num), total=test_num):
            res_i = run_one(model, libero_benchmark, task_idx, out_dir, episode_idx)
            succ_num += float(res_i)
            succ_rate = succ_num / float(episode_idx + 1)
            print(f"Success rate: {succ_rate:.4f}, success num: {succ_num}, total num: {episode_idx + 1}")

            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            _write_json(
                {
                    "succ_num": succ_num,
                    "total_num": float(episode_idx + 1),
                    "succ_rate": succ_rate,
                },
                out_file,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=29056,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finished LIBERO evaluation.")


if __name__ == "__main__":
    main()
