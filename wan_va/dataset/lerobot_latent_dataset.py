# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import bisect
import os
import packaging.version
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME
try:
    from robosuite.utils.transform_utils import axisangle2quat as _rs_axisangle2quat
    from robosuite.utils.transform_utils import quat2axisangle as _rs_quat2axisangle
except Exception:
    _rs_axisangle2quat = None
    _rs_quat2axisangle = None


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _is_torch(x) -> bool:
    return torch.is_tensor(x)


def _to_float_array(x):
    """
    保持输入类型（torch / numpy），并尽量转为 float32 以避免后续归一化/三角函数精度和类型问题。
    """
    if _is_torch(x):
        if not x.is_floating_point():
            x = x.float()
        return x
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
    return x


def euler_xyz_to_quat_xyzw(euler_xyz):
    """
    Euler(roll,pitch,yaw) -> Quaternion(x,y,z,w)

    - **欧拉角约定**：这里采用与 `scipy.spatial.transform.Rotation.from_euler('xyz', ...)` 一致的
      外旋/内旋等细节以 SciPy 为准（等价于常见的 roll(x)-pitch(y)-yaw(z) 组合）。
    - **四元数顺序**：严格使用 (x, y, z, w)。

    输入 shape: (..., 3)
    输出 shape: (..., 4)
    """
    euler_xyz = _to_float_array(euler_xyz)
    if _is_torch(euler_xyz):
        roll = euler_xyz[..., 0]
        pitch = euler_xyz[..., 1]
        yaw = euler_xyz[..., 2]

        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        # 对应 SciPy 'xyz' 的常用组合（x-roll, y-pitch, z-yaw）
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return torch.stack([qx, qy, qz, qw], dim=-1)

    quat = R.from_euler("xyz", euler_xyz, degrees=False).as_quat()  # (x,y,z,w)
    return quat.astype(np.float32)


def quat_xyzw_to_euler_xyz(quat_xyzw):
    """
    Quaternion(x,y,z,w) -> Euler(roll,pitch,yaw)  (xyz 顺序)

    输入 shape: (..., 4)
    输出 shape: (..., 3)
    """
    quat_xyzw = _to_float_array(quat_xyzw)
    if _is_torch(quat_xyzw):
        qx, qy, qz, qw = [quat_xyzw[..., i] for i in range(4)]

        # 防止数值漂移导致的非单位四元数
        norm = torch.sqrt(qx * qx + qy * qy + qz * qz + qw * qw).clamp_min(1e-8)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        # 经典 yaw-pitch-roll 反解（与上面的组合配套）
        t0 = 2.0 * (qw * qx + qy * qz)
        t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(t0, t1)

        t2 = 2.0 * (qw * qy - qz * qx)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)

        t3 = 2.0 * (qw * qz + qx * qy)
        t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(t3, t4)

        return torch.stack([roll, pitch, yaw], dim=-1)

    euler = R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
    return euler.astype(np.float32)


def _axisangle2quat_compat(vec):
    if _rs_axisangle2quat is not None:
        return _rs_axisangle2quat(vec)
    # scipy fallback: rotation vector <-> quaternion are mathematically equivalent.
    return R.from_rotvec(np.asarray(vec, dtype=np.float64)).as_quat()


def _quat2axisangle_compat(quat):
    if _rs_quat2axisangle is not None:
        return _rs_quat2axisangle(quat)
    # scipy fallback: returns rotvec (axis-angle vector).
    return R.from_quat(np.asarray(quat, dtype=np.float64)).as_rotvec()


def axisangle_to_quat_xyzw(axisangle):
    """
    Axis-angle(rotvec) -> Quaternion(x,y,z,w)

    输入 shape: (..., 3)
    输出 shape: (..., 4)
    """
    axisangle = _to_float_array(axisangle)
    if _is_torch(axisangle):
        device = axisangle.device
        dtype = axisangle.dtype
        axisangle_np = axisangle.detach().cpu().numpy().reshape(-1, 3)
        quat_np = np.stack([_axisangle2quat_compat(v) for v in axisangle_np], axis=0).astype(np.float32)
        quat = torch.from_numpy(quat_np).to(device=device, dtype=dtype)
        return quat.reshape(*axisangle.shape[:-1], 4)

    axisangle_np = np.asarray(axisangle, dtype=np.float32).reshape(-1, 3)
    quat_np = np.stack([_axisangle2quat_compat(v) for v in axisangle_np], axis=0).astype(np.float32)
    return quat_np.reshape(*axisangle.shape[:-1], 4)


def quat_xyzw_to_axisangle(quat_xyzw):
    """
    Quaternion(x,y,z,w) -> Axis-angle(rotvec)

    输入 shape: (..., 4)
    输出 shape: (..., 3)
    """
    quat_xyzw = _to_float_array(quat_xyzw)
    if _is_torch(quat_xyzw):
        device = quat_xyzw.device
        dtype = quat_xyzw.dtype
        quat_np = quat_xyzw.detach().cpu().numpy().reshape(-1, 4)
        axisangle_np = np.stack([_quat2axisangle_compat(v) for v in quat_np], axis=0).astype(np.float32)
        axisangle = torch.from_numpy(axisangle_np).to(device=device, dtype=dtype)
        return axisangle.reshape(*quat_xyzw.shape[:-1], 3)

    quat_np = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1, 4)
    axisangle_np = np.stack([_quat2axisangle_compat(v) for v in quat_np], axis=0).astype(np.float32)
    return axisangle_np.reshape(*quat_xyzw.shape[:-1], 3)


def robocasa_to_lingbot(action_12):
    """
    RoboCasa 12D -> LingBot-VA 30D

    RoboCasa(12):
      0:3   EEF XYZ
      3:6   EEF 旋转 Axis-angle(rotvec)
      6     Gripper
      7:9   Base XY
      9     Base Yaw
      10    Torso Z
      11    Gate（模式）

    LingBot(30): 右臂(0:15) + 左臂(15:30)，每臂 15:
      [XYZ(3), Quat(4), J1..J7(7), Gripper(1)]
    """
    action_12 = _to_float_array(action_12)
    last_dim = action_12.shape[-1]
    assert last_dim == 12, f"Expected (...,12) RoboCasa action, got {action_12.shape}"

    if _is_torch(action_12):
        out = torch.zeros(*action_12.shape[:-1], 30, dtype=action_12.dtype, device=action_12.device)
        mask = torch.zeros(*action_12.shape[:-1], 30, dtype=torch.bool, device=action_12.device)
    else:
        out = np.zeros((*action_12.shape[:-1], 30), dtype=np.float32)
        mask = np.zeros((*action_12.shape[:-1], 30), dtype=bool)

    # --- 主臂（右臂 0~14） ---
    out[..., 0:3] = action_12[..., 0:3]
    out[..., 3:7] = axisangle_to_quat_xyzw(action_12[..., 3:6])
    out[..., 14] = action_12[..., 6]

    mask[..., 0:3] = True
    mask[..., 3:7] = True
    mask[..., 14] = True

    # --- 副臂（左臂 15~29），用空槽位塞底盘/躯干 ---
    out[..., 15:17] = action_12[..., 7:9]   # base XY
    out[..., 17] = action_12[..., 10]      # torso Z
    out[..., 22] = action_12[..., 9]       # base yaw -> 副臂 J1
    out[..., 29] = action_12[..., 11]       # gate -> 副臂 gripper

    mask[..., 15:17] = True
    mask[..., 17] = True
    mask[..., 22] = True
    mask[..., 29] = True

    return out, mask


def lingbot_to_robocasa(action_30):
    """
    LingBot-VA 30D -> RoboCasa 12D

        语义约定（与 RoboCasa wrapper 一致）：
        - gripper_close / control_mode 采用 0.5 阈值二值化，再映射为 {-1, +1}
            （wrapper 内部判断同样使用 <0.5 / >=0.5）。
    """
    action_30 = _to_float_array(action_30)
    last_dim = action_30.shape[-1]
    assert last_dim == 30, f"Expected (...,30) LingBot action, got {action_30.shape}"

    if _is_torch(action_30):
        out = torch.zeros(*action_30.shape[:-1], 12, dtype=action_30.dtype, device=action_30.device)
    else:
        out = np.zeros((*action_30.shape[:-1], 12), dtype=np.float32)

    # 主臂：EEF XYZ + Quat->Axis-angle + Gripper
    out[..., 0:3] = action_30[..., 0:3]
    out[..., 3:6] = quat_xyzw_to_axisangle(action_30[..., 3:7])
    gripper = action_30[..., 14]
    if _is_torch(action_30):
        out[..., 6] = torch.where(gripper > 0.5, torch.ones_like(gripper), -torch.ones_like(gripper))
    else:
        out[..., 6] = np.where(gripper > 0.5, 1.0, -1.0).astype(np.float32)

    # 底盘/躯干：从副臂被劫持的槽位取回
    out[..., 7:9] = action_30[..., 15:17]
    out[..., 10] = action_30[..., 17]
    out[..., 9] = action_30[..., 22]

    gate = action_30[..., 29]
    if _is_torch(action_30):
        out[..., 11] = torch.where(gate > 0.5, torch.ones_like(gate), -torch.ones_like(gate))
    else:
        out[..., 11] = np.where(gate > 0.5, 1.0, -1.0).astype(np.float32)

    return out


def normalize_action_30(action_30, q01, q99):
    """
    归一化占位符（提醒接口）：在送入 LingBot 前，通常需要对 30D 每一维做分位数归一化。
    当前仓库实际归一化逻辑在 `_action_post_process` 内实现。
    """
    if torch.is_tensor(action_30):
        q01_t = torch.as_tensor(q01, dtype=action_30.dtype, device=action_30.device)
        q99_t = torch.as_tensor(q99, dtype=action_30.dtype, device=action_30.device)
        return (action_30 - q01_t) / (q99_t - q01_t + 1e-6) * 2.0 - 1.0
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    return (action_30 - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def denormalize_action_30(action_30_norm, q01, q99):
    """
    反归一化占位符（提醒接口）：将 [-1,1] 空间还原回原始 30D 动作尺度。
    """
    if torch.is_tensor(action_30_norm):
        q01_t = torch.as_tensor(q01, dtype=action_30_norm.dtype, device=action_30_norm.device)
        q99_t = torch.as_tensor(q99, dtype=action_30_norm.dtype, device=action_30_norm.device)
        return (action_30_norm + 1.0) * 0.5 * (q99_t - q01_t) + q01_t
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    return (action_30_norm + 1.0) * 0.5 * (q99 - q01) + q01

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )
def construct_lerobot_multi_processor(config, num_init_worker=8):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )

    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]

    # 只保留指定任务
    if hasattr(config, 'task_names') and config.task_names is not None:
        selected = set(config.task_names)
        def get_task_name(repo_path):
            p = Path(repo_path)
            # .../TaskName/Date/lerobot
            return p.parents[1].name

        repo_list = [v for v in repo_list if get_task_name(v) in selected]

    print("Selected repos:")
    for v in repo_list:
        print("  ", v)
    print("Num selected repos:", len(repo_list))

    if not repo_list:
        return []

    # 子进程数不超过 repo 数，避免 fork 一堆空闲 worker（4 个 repo 时 Pool(8) 无意义）
    num_init_worker = min(int(num_init_worker), len(repo_list))

    with Pool(num_init_worker) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)
    print("\nLoaded datasets:")
    for repo, dset in zip(repo_list, datasets_out_lst):
        print(f"{os.path.basename(repo)}: len={len(dset)}")

    print("Total dataset num:", sum(len(d) for d in datasets_out_lst))
    return datasets_out_lst
# def construct_lerobot_multi_processor(config, 
#                                       num_init_worker=8,
#                                       ):
#     datasets_out_lst = []
#     construct_func = partial(
#         construct_lerobot,
#         config=config,
#     )
#     repo_list = recursive_find_file(config.dataset_path, 'info.json')
#     repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
#     with Pool(num_init_worker) as pool:
#         datasets_out_lst = pool.map(construct_func, repo_list)
                
#     return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)
class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    """
    按帧数比例采样：
    - 每个 meta/clip 的采样概率 ~ 它的帧数
    - 通过把每个 meta 映射成 sample_units 个“虚拟样本”实现
    - DataLoader 仍然可以继续用 shuffle=True，不需要改训练主循环
    """
    def __init__(
        self,
        config,
        num_init_worker=None,
    ):
        self.config = config

        if num_init_worker is None:
            ncpu = os.cpu_count() or 8
            num_init_worker = int(
                getattr(config, "dataset_init_workers", min(32, max(4, ncpu)))
            )

        self._datasets = construct_lerobot_multi_processor(
            config,
            num_init_worker,
        )

        # 一个虚拟样本代表多少帧；默认 1 = 严格按帧数
        self.sample_unit_frames = max(1, int(getattr(config, "sample_unit_frames", 1)))

        # 展平后的索引：
        # self._meta_lut[i] = (dset_id, local_idx)
        # self._sample_prefix[i] ~ 第 i 个 meta 对应的累计 sample_units
        self._meta_lut = []
        self._sample_prefix = self._build_weighted_prefix()

        total_clips = len(self._meta_lut)
        total_virtual = self.__len__()
        print(f"[Frame-proportional sampling] sample_unit_frames={self.sample_unit_frames}")
        print(f"[Frame-proportional sampling] total_clips={total_clips}, total_virtual_samples={total_virtual}")

    def _build_weighted_prefix(self):
        acc = [0]
        self._meta_lut = []

        for dset_id, dset in enumerate(self._datasets):
            for local_idx, meta in enumerate(dset.new_metas):
                num_frames = int(meta.get("num_frames", meta["end_frame"] - meta["start_frame"]))
                num_frames = max(1, num_frames)

                sample_units = int(meta.get(
                    "sample_units",
                    max(1, (num_frames + self.sample_unit_frames - 1) // self.sample_unit_frames)
                ))

                self._meta_lut.append((dset_id, local_idx))
                acc.append(acc[-1] + sample_units)

        return acc

    def __len__(self):
        return self._sample_prefix[-1] if self._sample_prefix else 0

    def __getitem__(self, idx) -> dict:
        assert idx < len(self), f"idx={idx}, len={len(self)}"

        # 把“虚拟样本 idx”映射回某个真实 meta
        meta_pos = bisect.bisect_right(self._sample_prefix, idx) - 1
        dset_id, local_idx = self._meta_lut[meta_pos]
        return self._datasets[dset_id][local_idx]
# class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         config,
#         num_init_worker=None,
#     ):
#         if num_init_worker is None:
#             ncpu = os.cpu_count() or 8
#             # 默认并行度与机器核数挂钩，避免无脑 128（仍可由 config.dataset_init_workers 覆盖）
#             num_init_worker = int(
#                 getattr(config, "dataset_init_workers", min(32, max(4, ncpu)))
#             )
#         self._datasets = construct_lerobot_multi_processor(
#             config,
#             num_init_worker,
#         )
#         self._acc_prefix = self._build_acc_prefix()

#     def __len__(
#         self,
#     ):
#         return self._acc_prefix[-1] if self._acc_prefix else 0

#     def _build_acc_prefix(self):
#         acc = [0]
#         for dset in self._datasets:
#             acc.append(acc[-1] + len(dset))
#         return acc

#     def __getitem__(self, idx) -> dict:
#         assert idx < len(self)
#         dset_id = bisect.bisect_right(self._acc_prefix, idx) - 1
#         local_idx = idx - self._acc_prefix[dset_id]
#         return self._datasets[dset_id][local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        # 按目录一次性列出 .pth，避免 parse_meta 里对 CephFS 做数百万次 exists（极慢）
        if _env_truthy("WAN_VA_SKIP_LATENT_EXISTS_CHECK"):
            self._latent_file_index = {}
        else:
            self._latent_file_index = self._build_latent_file_index()
        self.parse_meta()

    def _build_latent_file_index(self):
        """(chunk 目录名, 相机 key) -> 该目录下 .pth 文件名集合。"""
        index = {}
        base = Path(self.latent_path)
        if not base.is_dir():
            return index
        for chunk_dir in base.glob("chunk-*"):
            if not chunk_dir.is_dir():
                continue
            chunk_name = chunk_dir.name
            for cam in self.used_video_keys:
                cam_dir = chunk_dir / cam
                if not cam_dir.is_dir():
                    continue
                try:
                    names = {
                        p.name
                        for p in cam_dir.iterdir()
                        if p.is_file() and p.suffix == ".pth"
                    }
                except OSError:
                    names = set()
                index[(chunk_name, cam)] = names
        return index
    def parse_meta(self):
        out = []
        episodes_iter = self.meta.episodes.items()
        if _env_truthy("WAN_VA_PARSE_META_PROGRESS"):
            repo_tag = Path(self.repo_id).name
            episodes_iter = tqdm(
                episodes_iter,
                total=len(self.meta.episodes),
                desc=f"parse_meta {repo_tag}",
                leave=False,
            )

        # 一个“采样单位”对应多少帧；默认 1，表示严格按帧数比例
        # 如果你觉得 epoch 太长，可以在 config 里设成 4 / 8 / 16
        self.sample_unit_frames = max(1, int(getattr(self.config, "sample_unit_frames", 1)))

        for key, value in episodes_iter:
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]

            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    # end_frame 在你这里是切片右边界，用 end-start 作为帧数更稳
                    num_frames = int(cur_meta["end_frame"] - cur_meta["start_frame"])
                    num_frames = max(1, num_frames)

                    # 采样权重单位数：严格按帧数比例，或者按 sample_unit_frames 做缩放
                    sample_units = max(
                        1,
                        (num_frames + self.sample_unit_frames - 1) // self.sample_unit_frames
                    )

                    cur_meta["num_frames"] = num_frames
                    cur_meta["sample_units"] = sample_units

                    out.append(cur_meta)

        self.new_metas = out
    # def parse_meta(self):
    #     out = []
    #     episodes_iter = self.meta.episodes.items()
    #     if _env_truthy("WAN_VA_PARSE_META_PROGRESS"):
    #         repo_tag = Path(self.repo_id).name
    #         episodes_iter = tqdm(
    #             episodes_iter,
    #             total=len(self.meta.episodes),
    #             desc=f"parse_meta {repo_tag}",
    #             leave=False,
    #         )
    #     for key, value in episodes_iter:
    #         episode_index = value["episode_index"]
    #         tasks = value["tasks"]
    #         action_config = value["action_config"]
    #         for acfg in action_config:
    #             cur_meta = {
    #                 "episode_index": episode_index,
    #                 "tasks": tasks,
    #             }
    #             cur_meta.update(acfg)

    #             check_statu = self._check_meta(
    #                 cur_meta["start_frame"],
    #                 cur_meta["end_frame"],
    #                 cur_meta["episode_index"],
    #             )

    #             if check_statu:
    #                 out.append(cur_meta)
    #     self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        if _env_truthy("WAN_VA_SKIP_LATENT_EXISTS_CHECK"):
            return True
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        chunk_name = f"chunk-{episode_chunk:03d}"
        fname = f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        for key in self.used_video_keys:
            names = self._latent_file_index.get((chunk_name, key))
            if not names or fname not in names:
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    '''def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()'''
    def _get_unorm_robocasa_action30(self, local_start_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]

        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()

        assert action.shape[1] == 12, f"Expected 12-dim action, got {action.shape}"

        action = np.pad(
            action,
            pad_width=((frame_stride * 4, 0), (0, 0)),
            mode='constant',
            constant_values=0
        )

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4
        action = action[:required_action_num]

        assert action.shape[0] == required_action_num

        action_30, action_mask_30 = robocasa_to_lingbot(action)
        return action_30, action_mask_30
    def get_stats_item(self, idx):
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]

        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)
        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]

        start_frame_global = self._get_global_idx(episode_index, start_frame)
        end_frame_global = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame_global, end_frame_global)
        ori_data_dict.update(hf_data_frames)

        raw_action_12 = ori_data_dict["action"]
        if torch.is_tensor(raw_action_12):
            raw_action_12 = raw_action_12.detach().cpu().numpy()

        action_30, action_mask_30 = self._get_unorm_robocasa_action30(
            local_start_frame=local_start_frame,
            latent_frame_ids=latent_frame_ids,
            action=raw_action_12,
        )

        return {
            "raw_action_12": raw_action_12,
            "latent_frame_ids": latent_frame_ids.detach().cpu().numpy() if torch.is_tensor(latent_frame_ids) else np.asarray(latent_frame_ids),
            "action_30": action_30,
            "action_mask_30": action_mask_30,
        }
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]

        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()

        if self.config.env_type == 'robotwin_tshape':
            ## 原来的逻辑保留
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate(
                [left_action, action[:, 7:8], right_action, action[:, 15:16]],
                axis=1
            )

            action = np.pad(
                action,
                pad_width=((frame_stride * 4, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )

            latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
            required_action_num = latent_frame_num * frame_stride * 4
            action = action[:required_action_num]
            action_mask = np.ones_like(action, dtype='bool')

            assert action.shape[0] == required_action_num

            action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
            action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

            action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
            action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]

            action_aligned = (action_aligned - self.q01) / (self.q99 - self.q01 + 1e-6) * 2. - 1.
            action_aligned = np.clip(action_aligned, -1.5, 1.5)

            action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
            action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
            action_aligned *= action_mask_aligned

            return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

        elif self.config.env_type == 'robocasa_tshape':
            # RoboCasa 原始动作约定为 [T, 12]
            assert action.shape[1] == 12, f"Expected 12-dim action, got {action.shape}"

            action = np.pad(
                action,
                pad_width=((frame_stride * 4, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )

            latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
            required_action_num = latent_frame_num * frame_stride * 4
            action = action[:required_action_num]

            assert action.shape[0] == required_action_num

            # 12D -> 30D 严格映射（含 Euler->Quat），并生成对应 mask
            action_30, action_mask_30 = robocasa_to_lingbot(action)

            # 归一化，只对前 12 维有意义；后面反正 mask 会乘掉
            action_30 = (action_30 - self.q01) / (self.q99 - self.q01 + 1e-6) * 2. - 1.
            action_30 *= action_mask_30

            action_30 = rearrange(action_30, "(f n) c -> c f n 1", f=latent_frame_num)
            action_mask_30 = rearrange(action_mask_30, "(f n) c -> c f n 1", f=latent_frame_num)

            return torch.from_numpy(action_30).float(), torch.from_numpy(action_mask_30).bool()

        else:
            raise NotImplementedError(f"Unsupported env_type: {self.config.env_type}")
    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(
            local_start_frame,
            local_end_frame,
            latent_frame_ids,
            ori_data_dict['action'],
        )

        # [F, H, W, C] -> [C, F, H, W]
        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2).contiguous()

        if torch.is_tensor(out_dict['text_emb']):
            out_dict['text_emb'] = out_dict['text_emb'].contiguous()

        if torch.is_tensor(out_dict['actions']):
            out_dict['actions'] = out_dict['actions'].contiguous()

        if torch.is_tensor(out_dict['actions_mask']):
            out_dict['actions_mask'] = out_dict['actions_mask'].contiguous()

        return out_dict
        # idx = idx % len(self.new_metas)
        # cur_meta = self.new_metas[idx]
        # episode_index = cur_meta["episode_index"]
        # start_frame = cur_meta["start_frame"]
        # end_frame = cur_meta["end_frame"]
        # local_start_frame = start_frame
        # local_end_frame = end_frame

        # ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        # latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        # start_frame = self._get_global_idx(episode_index, start_frame)
        # end_frame = self._get_global_idx(episode_index, end_frame)

        # hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        # ori_data_dict.update(hf_data_frames)
        # out_dict = self._cat_video_latents(ori_data_dict)

        # out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        # out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        
        # 方法一：新增 raw_prompt 用于检查脚本提取文本提示词
        # out_dict['raw_prompt'] = cur_meta.get("tasks", "No prompt found")
        #out_dict['task_name'] = getattr(self, "repo_id", "unknown_task")
        
        # 方法一：尝试从原数据集中提取第一帧原始图像给脚本检查
        # 注意：为了不影响正常的高优前向性能，加上异常处理，只取单个相机的第一帧
        # try:
        #     if self.used_video_keys:
        #         cam_key = self.used_video_keys[0] # 例如 "observation.images.agent0_rgb"
        #         # 读取 HF 基础数据集中对应那一帧！
        #         raw_item = self.hf_dataset[int(start_frame)]
        #         if cam_key in raw_item:
        #             img = raw_item[cam_key]
        #             # 如果是 PIL 图片或者 numpy 数组，转换为 Torch 张量以适应 check_com 的接收要求
        #             if not torch.is_tensor(img):
        #                 import torchvision.transforms.functional as TF
        #                 img = TF.to_tensor(img)
        #             out_dict['raw_image'] = img.clone().detach() * 255.0 # 放大到0-255以便后续 dump 为 uint8
        # except Exception as e:
        #     pass

        #return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import get_config
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        get_config('demo_train')
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])

    # --- RoboCasa 12D <-> LingBot 30D 映射自测 ---
    print("\n[robocasa<->lingbot mapping quick test]")
    # 构造一个可读的单样本（单位：弧度）
    robo = np.zeros((1, 12), dtype=np.float32)
    robo[0, 0:3] = np.array([0.1, -0.2, 0.3], dtype=np.float32)          # eef xyz
    robo[0, 3:6] = np.array([0.2, -0.1, 0.05], dtype=np.float32)         # euler xyz
    robo[0, 6] = 0.7                                                     # gripper
    robo[0, 7:9] = np.array([1.5, -2.0], dtype=np.float32)               # base xy
    robo[0, 9] = 0.42                                                    # torso z
    robo[0, 10] = -0.3                                                   # base yaw
    robo[0, 11] = 0.2                                                    # gate (soft, will be thresholded on inverse)

    ling, ling_mask = robocasa_to_lingbot(robo)
    robo_rec = lingbot_to_robocasa(ling)

    print("robo shape:", robo.shape, "ling shape:", ling.shape, "mask shape:", ling_mask.shape)
    print("EEF xyz (robo -> ling):", robo[0, 0:3], "=>", ling[0, 0:3])
    print("Gripper (robo -> ling):", robo[0, 6], "=>", ling[0, 14])
    print("Base XY (robo -> ling):", robo[0, 7:9], "=>", ling[0, 15:17])
    print("TorsoZ (robo -> ling):", robo[0, 10], "=>", ling[0, 17])
    print("Yaw (robo -> ling):", robo[0, 9], "=>", ling[0, 22])
    print("Gate (robo -> ling):", robo[0, 11], "=>", ling[0, 29])
    print("Gate thresholded (ling -> robo):", ling[0, 29], "=>", robo_rec[0, 11], "(must be +/-1)")
    
