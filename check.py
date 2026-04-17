#!/usr/bin/env python3
# check.py — 三阶段流水线体检：静态配置 / 数据与白盒校验 / 前向与可选 smoke 训练步
#
# 与训练对齐（推荐，与 script/run_va_posttrain.sh 相同 torchrun + config-name）：
#   bash script/run_va_check.sh
#   # 或
#   torchrun --nproc_per_node=8 --master_port 29501 check.py --config-name robocasa_train --save_dir ./debug_check
#
# 单机单卡（也会 init_process_group world_size=1，与 torchrun --nproc_per_node=1 一致）：
#   python check.py --config-name robocasa_train --save_dir ./debug_check
#
# 仍支持从 .py 路径加载：
#   python check.py --config wan_va/configs/va_robocasa_train_cfg.py --save_dir ./debug_check

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_dist_env_defaults() -> None:
    """与 torchrun 未注入时的单机默认一致，便于直接 python check.py 也能走 FSDP 分支。"""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def read_dist_env() -> Tuple[int, int, int]:
    _ensure_dist_env_defaults()
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])


def to_cpu_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu()
    return x


def tensor_stats(x: torch.Tensor) -> dict:
    x = x.detach()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "min": float(x.min().item()) if x.numel() > 0 else None,
        "max": float(x.max().item()) if x.numel() > 0 else None,
        "mean": float(x.float().mean().item()) if x.numel() > 0 else None,
        "std": float(x.float().std().item()) if x.numel() > 1 else 0.0,
        "nan_count": int(torch.isnan(x).sum().item()) if x.is_floating_point() else 0,
        "inf_count": int(torch.isinf(x).sum().item()) if x.is_floating_point() else 0,
    }


class RunningStats:
    """累计多个 batch 的全局 min/max/mean/std（逐元素）。"""

    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min: Optional[float] = None
        self.max: Optional[float] = None

    def update(self, x: torch.Tensor):
        x = x.detach().float()
        if x.numel() == 0:
            return
        self.count += x.numel()
        self.sum += x.sum().item()
        self.sumsq += (x * x).sum().item()
        mn = x.min().item()
        mx = x.max().item()
        self.min = mn if self.min is None else min(self.min, mn)
        self.max = mx if self.max is None else max(self.max, mx)

    def summary(self) -> dict:
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        std = math.sqrt(var)
        return {"count": self.count, "mean": mean, "std": std, "min": self.min, "max": self.max}


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def dump_json(obj, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def log_line(path: str | Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def config_to_jsonable(obj: Any, depth: int = 0) -> Any:
    if depth > 20:
        return "<max_depth>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [config_to_jsonable(x, depth + 1) for x in obj]
    if isinstance(obj, dict) or hasattr(obj, "keys"):
        out = {}
        try:
            for k in obj.keys():
                try:
                    v = obj[k]
                except Exception:
                    v = getattr(obj, str(k), None)
                out[str(k)] = config_to_jsonable(v, depth + 1)
        except Exception:
            pass
        return out
    if isinstance(obj, (np.ndarray,)):
        return {"__numpy__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    return str(type(obj).__name__)


# ---------------------------------------------------------------------------
# 阶段一：静态配置审查
# ---------------------------------------------------------------------------

def _path_exists(p: Optional[str], label: str, issues: List[str]) -> bool:
    if p is None:
        issues.append(f"{label}: 未设置")
        return False
    if not isinstance(p, (str, os.PathLike)):
        issues.append(f"{label}: 非路径类型 {type(p)}")
        return False
    ok = os.path.exists(p)
    if not ok:
        issues.append(f"{label}: 路径不存在 -> {p}")
    return ok


def static_config_audit(cfg: Any) -> dict:
    """
    纯静态核对：路径、维度、与训练相关的关键超参是否自洽。
    """
    issues: List[str] = []
    notes: List[str] = []

    # 路径
    _path_exists(getattr(cfg, "dataset_path", None), "dataset_path", issues)
    _path_exists(getattr(cfg, "empty_emb_path", None), "empty_emb_path", issues)
    pretrained = getattr(cfg, "wan22_pretrained_model_name_or_path", None)
    if _path_exists(pretrained, "wan22_pretrained_model_name_or_path", issues) and pretrained:
        tf_dir = os.path.join(pretrained, "transformer")
        if not os.path.isdir(tf_dir):
            issues.append(f"transformer 权重目录不存在: {tf_dir}")
        else:
            st_files = list(Path(tf_dir).glob("*.safetensors"))
            if not st_files:
                issues.append(f"{tf_dir} 下未找到 .safetensors")
    save_root = getattr(cfg, "save_root", None)
    if save_root and not os.path.exists(save_root):
        notes.append(f"save_root 尚不存在（训练时会创建）: {save_root}")

    # 图像 / patch
    patch = getattr(cfg, "patch_size", None)
    if patch is not None:
        if not (isinstance(patch, (list, tuple)) and len(patch) == 3):
            issues.append(f"patch_size 应为长度为 3 的序列，当前: {patch}")
    for name in ("height", "width"):
        v = getattr(cfg, name, None)
        if v is not None and (not isinstance(v, int) or v <= 0):
            issues.append(f"{name} 应为正整数，当前: {v}")

    action_dim = getattr(cfg, "action_dim", None)
    if action_dim is not None:
        if not isinstance(action_dim, int) or action_dim <= 0:
            issues.append(f"action_dim 应为正整数，当前: {action_dim}")

    norm_stat = getattr(cfg, "norm_stat", None)
    if norm_stat is not None:
        q01 = norm_stat.get("q01") if isinstance(norm_stat, dict) else None
        q99 = norm_stat.get("q99") if isinstance(norm_stat, dict) else None
        if q01 is None or q99 is None:
            issues.append("norm_stat 缺少 q01 或 q99")
        else:
            if len(q01) != len(q99):
                issues.append(f"norm_stat q01 长度 {len(q01)} != q99 长度 {len(q99)}")
            if action_dim is not None and len(q01) != action_dim:
                issues.append(
                    f"norm_stat 长度 {len(q01)} 与 action_dim={action_dim} 不一致（RoboCasa/LingBot 应为 30）"
                )
            bad = [(i, a, b) for i, (a, b) in enumerate(zip(q01, q99)) if float(a) >= float(b)]
            if bad:
                issues.append(f"norm_stat 存在 q01>=q99 的维度（前 5 个）: {bad[:5]}")

    used = getattr(cfg, "used_action_channel_ids", None)
    inv = getattr(cfg, "inverse_used_action_channel_ids", None)
    if used is not None and inv is not None:
        ad = action_dim or 30
        if len(inv) != ad:
            issues.append(f"inverse_used_action_channel_ids 长度 {len(inv)} != action_dim {ad}")
        if max(used, default=-1) >= ad:
            issues.append(f"used_action_channel_ids 存在下标 >= action_dim ({ad})")

    obs_keys = getattr(cfg, "obs_cam_keys", None)
    if obs_keys is not None and len(obs_keys) == 0:
        issues.append("obs_cam_keys 为空")

    env_type = getattr(cfg, "env_type", None)
    if env_type == "robocasa_tshape" and action_dim not in (None, 30):
        issues.append(f"robocasa_tshape 通常 action_dim=30，当前 {action_dim}")

    mapping_test = None
    if env_type == "robocasa_tshape":
        mapping_test = _robocasa_lingbot_roundtrip_selftest()
        if not mapping_test.get("ok"):
            issues.append(f"RoboCasa<->LingBot 映射自测失败: {mapping_test.get('error', mapping_test)}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "notes": notes,
        "config_snapshot": config_to_jsonable(cfg),
        "robocasa_lingbot_mapping_selftest": mapping_test,
    }


def _robocasa_lingbot_roundtrip_selftest() -> dict:
    """静态代码级自测：12D 与 30D 互转（与 dataset 实现一致）。"""
    try:
        from wan_va.dataset.lerobot_latent_dataset import lingbot_to_robocasa, robocasa_to_lingbot
    except Exception as e:
        return {"ok": False, "error": f"import failed: {e}"}
    robo = np.zeros((1, 12), dtype=np.float32)
    robo[0, 0:3] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    robo[0, 3:6] = np.array([0.2, -0.1, 0.05], dtype=np.float32)
    robo[0, 6] = 0.7
    robo[0, 7:9] = np.array([1.5, -2.0], dtype=np.float32)
    robo[0, 9] = 0.42
    robo[0, 10] = -0.3
    robo[0, 11] = 0.2
    ling, ling_mask = robocasa_to_lingbot(robo)
    robo_rec = lingbot_to_robocasa(ling)
    # 连续维应近似可逆；离散维（gripper/gate）按阈值二值化后再校验。
    continuous_ids = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10], dtype=np.int64)
    err_cont = float(np.abs(robo[0, continuous_ids] - robo_rec[0, continuous_ids]).max())

    expected_gripper = 1.0 if float(robo[0, 6]) > 0.5 else -1.0
    expected_gate = 1.0 if float(robo[0, 11]) > 0.5 else -1.0
    gripper_ok = float(robo_rec[0, 6]) == expected_gripper
    gate_ok = float(robo_rec[0, 11]) == expected_gate

    mask_ct = int(np.asarray(ling_mask[0]).sum())
    ok = err_cont < 1e-5 and gripper_ok and gate_ok and mask_ct == 13
    return {
        "ok": ok,
        "max_abs_err_continuous_dims": err_cont,
        "gripper_recovered": float(robo_rec[0, 6]),
        "gripper_expected": expected_gripper,
        "gripper_thresholded_ok": gripper_ok,
        "gate_recovered": float(robo_rec[0, 11]),
        "gate_expected": expected_gate,
        "gate_thresholded_ok": gate_ok,
        "lingbot_mask_true_count": mask_ct,
    }


# ---------------------------------------------------------------------------
# 数据字段与动作语义
# ---------------------------------------------------------------------------

def try_decode_prompt(batch: dict) -> dict:
    candidates = [
        "prompt", "prompts", "text", "texts", "caption", "captions",
        "instruction", "instructions",
    ]
    out = {}
    for k in candidates:
        if k in batch:
            v = batch[k]
            if isinstance(v, str):
                out[k] = [v]
            elif isinstance(v, (list, tuple)):
                out[k] = list(v[:8])
            else:
                out[k] = str(type(v))
    return out


def collect_basic_info(batch: dict) -> dict:
    info = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            info[k] = {}
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    info[k][kk] = {
                        "shape": list(vv.shape),
                        "dtype": str(vv.dtype),
                        "device": str(vv.device),
                    }
                else:
                    info[k][kk] = str(type(vv))
        elif isinstance(v, torch.Tensor):
            info[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "device": str(v.device),
            }
        else:
            info[k] = str(type(v))
    return info


def check_tensor(name: str, x: torch.Tensor, problems: list, summary: Optional[dict] = None) -> dict:
    st = tensor_stats(x)
    if summary is not None:
        summary[name] = st
    if x.is_floating_point():
        if st["nan_count"] > 0:
            problems.append(f"{name}: 含 NaN，数量={st['nan_count']}")
        if st["inf_count"] > 0:
            problems.append(f"{name}: 含 Inf，数量={st['inf_count']}")
        if st["std"] is not None and st["std"] == 0.0 and x.numel() > 1:
            problems.append(f"{name}: 标准差为 0，可能全常数")
    return st


def update_running_stats(stats_dict: dict, prefix: str, x: torch.Tensor):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        if prefix not in stats_dict:
            stats_dict[prefix] = RunningStats()
        stats_dict[prefix].update(x)


def recursive_collect_stats(obj, prefix: str, stats_dict: dict, batch_summary: dict, problems: list):
    if isinstance(obj, dict):
        for k, v in obj.items():
            recursive_collect_stats(
                v, f"{prefix}.{k}" if prefix else k, stats_dict, batch_summary, problems
            )
    elif isinstance(obj, torch.Tensor):
        check_tensor(prefix, obj, problems, batch_summary)
        update_running_stats(stats_dict, prefix, obj)


def inspect_action_space(batch: dict, report: dict):
    """扩散 action 张量结构（noisy_latents 等）— 在 raw batch 上通常不存在，占位兼容。"""
    out: dict = {"exists": False}
    if "action_dict" not in batch:
        report["action_check"] = out
        return
    out["exists"] = True
    ad = batch["action_dict"]
    for k in ["noisy_latents", "latent", "timesteps", "cond_timesteps", "grid_id"]:
        if k in ad and isinstance(ad[k], torch.Tensor):
            out[k] = tensor_stats(ad[k])
    if "noisy_latents" in ad and isinstance(ad["noisy_latents"], torch.Tensor):
        shape = list(ad["noisy_latents"].shape)
        out["noisy_latents_rank"] = len(shape)
        out["noisy_latents_shape"] = shape
        if len(shape) not in (3, 5):
            out["warning"] = f"action_dict.noisy_latents 维度为 {shape}，不是常见 3D/5D 结构"
    report["action_check"] = out


def inspect_text_length(batch: dict, report: dict):
    out = {}
    if "latent_dict" in batch and isinstance(batch["latent_dict"], dict):
        ld = batch["latent_dict"]
        if "text_emb" in ld and isinstance(ld["text_emb"], torch.Tensor):
            shape = list(ld["text_emb"].shape)
            out["text_emb_shape"] = shape
            if len(shape) >= 2:
                out["text_len"] = shape[1]
                if shape[1] != 512:
                    out["warning"] = (
                        f"text_emb 第二维是 {shape[1]}，不是 512；若 flex mask 写死 512 会报错。"
                    )
    report["text_check"] = out


def inspect_raw_text_emb(batch: dict, report: dict):
    """Dataset 直出 batch 内的 text_emb 形状（送进模型前的条件向量）。"""
    out = {}
    if "text_emb" in batch and isinstance(batch["text_emb"], torch.Tensor):
        t = batch["text_emb"]
        out["text_emb_shape"] = list(t.shape)
        if t.dim() >= 2:
            out["text_seq_len"] = t.shape[1]
    report["raw_text_check"] = out


def inspect_embodied_actions_raw(batch: dict, cfg: Any, report: dict):
    """
    具身：核对 Dataset 输出的 actions / actions_mask 维度与 RoboCasa 30D 语义槽位。
    与 lerobot_latent_dataset.robocasa_to_lingbot 注释一致。
    """
    issues: List[str] = []
    warnings: List[str] = []
    out: Dict[str, Any] = {
        "env_type": getattr(cfg, "env_type", None),
        "issues": issues,
        "warnings": warnings,
    }
    adim = getattr(cfg, "action_dim", 30)

    if "actions" not in batch or "actions_mask" not in batch:
        issues.append("batch 缺少 actions 或 actions_mask")
        out["ok"] = False
        report["embodied_action_check"] = out
        return

    act = batch["actions"]
    mask = batch["actions_mask"]
    if not isinstance(act, torch.Tensor) or not isinstance(mask, torch.Tensor):
        issues.append("actions/actions_mask 非 Tensor")
        out["ok"] = False
        report["embodied_action_check"] = out
        return

    out["actions_shape"] = list(act.shape)
    out["mask_shape"] = list(mask.shape)
    if act.shape != mask.shape:
        issues.append(f"actions 与 actions_mask shape 不一致: {act.shape} vs {mask.shape}")

    if act.dim() == 5:
        b, c, f, n, one = act.shape
        out["layout"] = "B,C,F,N,1"
        if c != adim:
            issues.append(f"通道维 C={c} 与 config.action_dim={adim} 不一致")
        if one != 1:
            issues.append(f"末维应为 1，当前 {one}")
    else:
        warnings.append(f"未预期的 actions 维度 {act.dim()}，期望 5 (B,C,F,N,1)")

    # 掩码为 True 的位置应在归一化后仍有界；全 False 通道应接近 0
    if act.dim() == 5 and mask.dtype == torch.bool:
        m = mask.bool()
        if m.any():
            vals = act[m]
            out["masked_action_min"] = float(vals.min().item())
            out["masked_action_max"] = float(vals.max().item())
        dead = ~m
        if dead.any():
            dead_vals = act[dead].float().abs()
            out["unmasked_abs_max"] = float(dead_vals.max().item())
            if out["unmasked_abs_max"] > 1e-3:
                warnings.append(
                    f"mask=False 处 |action| 最大 {out['unmasked_abs_max']:.6f}，期望接近 0"
                )

    if getattr(cfg, "env_type", None) == "robocasa_tshape":
        # 与 robocasa_to_lingbot 中 mask=True 的索引一致（0-based）
        expected_active = {0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 22, 29}
        out["expected_robocasa_mapped_channel_indices"] = sorted(expected_active)
        if act.dim() == 5:
            # 任取 batch0：哪些通道在任意时空位置曾被激活
            ch_any = mask[0, :, :, :, 0].any(dim=(1, 2)).nonzero(as_tuple=True)[0].tolist()
            out["sample_channels_with_any_mask"] = ch_any
            missing = sorted(expected_active - set(ch_any))
            extra = sorted(set(ch_any) - expected_active)
            if missing:
                warnings.append(
                    f"本 batch 样本 0 中从未 True 的期望活跃通道（若多 batch 均缺失则需排查映射）: {missing}"
                )
            if extra:
                warnings.append(f"本 batch 样本 0 中出现非标准活跃通道: {extra}")

    out["ok"] = len(issues) == 0
    report["embodied_action_check"] = out


def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def norm_range_audit(
    stats_summary: dict,
    problems: List[str],
    latent_key: str = "batch.latents",
    action_key: str = "batch.actions",
    action_margin: float = 0.05,
    latent_lo: float = -12.0,
    latent_hi: float = 12.0,
):
    """
    动作经 Dataset 归一化后目标区间约 [-1,1]；VAE latent 不做 [-1,1] 假设，仅做宽区间异常提示。
    stats_summary: 键 -> RunningStats.summary() 的字典。
    """
    if action_key in stats_summary:
        s = stats_summary[action_key]
        if s.get("min") is not None and s.get("max") is not None:
            lo, hi = -1.0 - action_margin, 1.0 + action_margin
            if s["min"] < lo or s["max"] > hi:
                problems.append(
                    f"{action_key} 全局范围 [{s['min']:.4f}, {s['max']:.4f}] 超出预期约 [{lo}, {hi}]，"
                    "请检查分位数 norm_stat 或异常值。"
                )
    if latent_key in stats_summary:
        s = stats_summary[latent_key]
        if s.get("min") is not None and s.get("max") is not None:
            if s["min"] < latent_lo or s["max"] > latent_hi:
                problems.append(
                    f"{latent_key} 范围 [{s['min']:.4f}, {s['max']:.4f}] 超出宽阈值 [{latent_lo}, {latent_hi}]，"
                    "请确认 latent 未损坏或缩放一致。"
                )


# ---------------------------------------------------------------------------
# 可视化 / 反查（在能力范围内）
# ---------------------------------------------------------------------------

def save_debug_tensors(
    batch: dict,
    save_dir: Path,
    batch_idx: int = 0,
    max_text_tokens_print: int = 4,
):
    """
    - text_emb: 保存统计与前几向量范数（无条件生成 tokenizer 时无法还原自然语言）。
    - latents: 取首帧、多通道拼成灰度网格 PNG，用于肉眼检查是否全常数/花屏。
    """
    vis_dir = save_dir / "visual_debug"
    ensure_dir(vis_dir)

    if "text_emb" in batch and isinstance(batch["text_emb"], torch.Tensor):
        te = batch["text_emb"].detach().float().cpu()
        meta = {
            "shape": list(te.shape),
            "min": float(te.min().item()),
            "max": float(te.max().item()),
            "mean": float(te.mean().item()),
            "std": float(te.std().item()),
        }
        if te.dim() >= 2:
            norms = te[0, :max_text_tokens_print].norm(dim=-1).tolist()
            meta["first_sequence_token_l2_norms"] = norms
        dump_json(meta, vis_dir / f"batch{batch_idx}_text_emb_meta.json")
        torch.save(te[:1], vis_dir / f"batch{batch_idx}_text_emb_slice.pt")

    if "latents" in batch and isinstance(batch["latents"], torch.Tensor):
        lat = batch["latents"].detach().float().cpu()
        # 期望 [B, C, F, H, W]
        if lat.dim() == 5:
            frame0 = lat[0, :, 0, :, :]
            c = min(16, frame0.shape[0])
            grid = frame0[:c]
            # [C,H,W] -> 简单归一化到 0-255
            g = grid.numpy()
            g = (g - g.min()) / (g.max() - g.min() + 1e-8)
            side = int(math.ceil(math.sqrt(c)))
            pad = side * side - c
            if pad > 0:
                g = np.pad(g, ((0, pad), (0, 0), (0, 0)), mode="constant", constant_values=1.0)
            tiles = g.reshape(side, side, g.shape[1], g.shape[2])
            img = np.zeros((side * g.shape[1], side * g.shape[2]), dtype=np.float32)
            for i in range(side):
                for j in range(side):
                    idx = i * side + j
                    if idx < g.shape[0]:
                        img[i * g.shape[1] : (i + 1) * g.shape[1], j * g.shape[2] : (j + 1) * g.shape[2]] = g[
                            idx
                        ]
            img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
            try:
                from PIL import Image

                Image.fromarray(img_u8, mode="L").save(vis_dir / f"batch{batch_idx}_latent_frame0_grid.png")
            except Exception as e:
                dump_json({"error": str(e)}, vis_dir / f"batch{batch_idx}_latent_png_error.json")


# ---------------------------------------------------------------------------
# 加载配置与 Trainer
# ---------------------------------------------------------------------------

def load_cfg_from_path(config_path: str) -> Tuple[Any, str]:
    """从配置文件路径加载 EasyDict 配置对象。"""
    rel = os.path.relpath(os.path.abspath(config_path), PROJECT_ROOT)
    if rel.startswith(".."):
        spec = importlib.util.spec_from_file_location("user_train_cfg", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载配置: {config_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        module_name = Path(rel).as_posix().replace("/", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        mod = importlib.import_module(module_name)

    stem = Path(config_path).stem
    cfg = getattr(mod, stem, None)
    if cfg is None:
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if isinstance(obj, dict) and "dataset_path" in obj:
                cfg = obj
                stem = name
                break
    if cfg is None:
        raise ValueError(f"在 {config_path} 中未找到与文件同名或含 dataset_path 的配置对象")
    return cfg, stem


def load_train_cfg(args: argparse.Namespace) -> Tuple[Any, dict]:
    """与 wan_va.train.run 相同：get_config(config_name) 或 .py 路径；并应用 --save-root。"""
    from easydict import EasyDict

    if args.config_name is not None:
        from wan_va.configs import get_config

        cfg = copy.deepcopy(get_config(args.config_name))
        meta = {"source": "config_name", "name": args.config_name}
    else:
        cfg, stem = load_cfg_from_path(args.config)
        cfg = copy.deepcopy(cfg)
        meta = {"source": "config_path", "path": args.config, "module_stem": stem}

    if args.save_root is not None:
        if not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)
        cfg.save_root = args.save_root

    return cfg, meta


def build_trainer_for_check(cfg: Any):
    """cfg.rank / local_rank / world_size 须已由调用方按训练入口设置（含 torchrun 环境）。"""
    from wan_va.train import Trainer

    return Trainer(cfg)


def forward_sanity_check(trainer, batch: dict, device: str) -> dict:
    """
    与正式训练一致：convert_input_format -> _prepare_input_dict -> transformer(..., train_mode=True)。
    """
    result: dict = {"ok": False}
    trainer.transformer.eval()
    try:
        batch_dev = trainer.convert_input_format({k: v for k, v in batch.items()})
        input_dict = trainer._prepare_input_dict(batch_dev)
        with torch.no_grad():
            t0 = time.time()
            out = trainer.transformer(input_dict, train_mode=True)
            result["time_sec"] = time.time() - t0
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error"] = str(e)
        return result

    result["ok"] = True
    if isinstance(out, tuple):
        result["num_outputs"] = len(out)
        result["outputs"] = []
        for i, item in enumerate(out):
            if isinstance(item, torch.Tensor):
                result["outputs"].append(
                    {
                        "idx": i,
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                        "min": float(item.min().item()),
                        "max": float(item.max().item()),
                        "mean": float(item.float().mean().item()),
                        "std": float(item.float().std().item()),
                    }
                )
            else:
                result["outputs"].append({"idx": i, "type": str(type(item))})
    elif isinstance(out, torch.Tensor):
        result["num_outputs"] = 1
        result["outputs"] = [
            {
                "shape": list(out.shape),
                "dtype": str(out.dtype),
                "min": float(out.min().item()),
                "max": float(out.max().item()),
                "mean": float(out.float().mean().item()),
                "std": float(out.float().std().item()),
            }
        ]
    else:
        result["output_type"] = str(type(out))
    return result


def smoke_train_step(trainer, batch: dict) -> dict:
    """单步：loss + backward + clip + step（与 _train_step 一致）。"""
    out: dict = {"ok": False}
    try:
        trainer.transformer.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        batch_dev = trainer.convert_input_format({k: v for k, v in batch.items()})
        losses = trainer._train_step(batch_dev, batch_idx=0)
        out["ok"] = True
        out["latent_loss"] = to_cpu_scalar(losses["latent_loss"])
        out["action_loss"] = to_cpu_scalar(losses["action_loss"])
        if "total_norm" in losses:
            out["grad_norm"] = to_cpu_scalar(losses["total_norm"])
    except Exception as e:
        out["error_type"] = type(e).__name__
        out["error"] = str(e)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="LingBot-VA 三阶段体检（与 torchrun + wan_va.train 配置/分布式对齐）",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="与 wan_va.train --config-name 相同，如 robotwin_train / robocasa_train",
    )
    src.add_argument(
        "--config",
        type=str,
        default=None,
        help="从训练配置 .py 路径加载（旧方式）",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="覆盖 config.save_root（与 wan_va.train --save-root 一致）",
    )
    parser.add_argument("--save_dir", type=str, default="./debug_check")
    parser.add_argument("--max_batches", type=int, default=100, help="DataLoader 空跑批次数")
    parser.add_argument("--forward_batch_idx", type=int, default=0, help="用于前向 / smoke 的 batch 下标")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--phases",
        type=str,
        default="config,data,forward",
        help="逗号分隔：config / data / forward / smoke（smoke 会反传+优化器步）",
    )
    parser.add_argument(
        "--no_model",
        action="store_true",
        help="不加载模型，仅跑 config + data（forward/smoke 会被跳过）",
    )
    parser.add_argument(
        "--smoke_train",
        action="store_true",
        help="等价于在 phases 中加入 smoke（完整一步训练）",
    )
    parser.add_argument(
        "--check_batch_size",
        type=int,
        default=None,
        help="覆盖 config.batch_size（用于极小显存试跑）",
    )
    parser.add_argument(
        "--check_load_worker",
        type=int,
        default=None,
        help="覆盖 config.load_worker（例如 0 排除多进程读盘干扰）",
    )
    args = parser.parse_args()

    rank, local_rank, world_size = read_dist_env()
    phases = {p.strip().lower() for p in args.phases.split(",") if p.strip()}
    if args.smoke_train:
        phases.add("smoke")

    save_dir = Path(args.save_dir)
    if rank == 0:
        ensure_dir(save_dir)

    report: dict = {
        "resolved_config": None,
        "max_batches": args.max_batches,
        "device": args.device,
        "phases": sorted(phases),
        "rank": rank,
        "world_size": world_size,
        "batches": [],
        "problems": [],
        "dataloader_timing": {},
    }

    cfg, cfg_meta = load_train_cfg(args)
    if rank == 0:
        report["resolved_config"] = cfg_meta

    if rank == 0 and "config" in phases:
        report["static_config_audit"] = static_config_audit(cfg)
        if not report["static_config_audit"]["ok"]:
            report["problems"].extend(
                [f"[static_config] {x}" for x in report["static_config_audit"]["issues"]]
            )
        print("[Phase1] 静态配置审查:", "通过" if report["static_config_audit"]["ok"] else "有问题，见 static_config_audit.issues")

    overrides = {}
    if args.check_batch_size is not None:
        overrides["batch_size"] = args.check_batch_size
    if args.check_load_worker is not None:
        overrides["load_worker"] = args.check_load_worker
    if rank == 0:
        for k, v in overrides.items():
            print(f"[INFO] 运行时将覆盖 config.{k} = {v}")

    need_model = bool(phases & {"forward", "smoke"}) and not args.no_model
    need_data = bool(phases & {"data", "forward", "smoke"})

    from easydict import EasyDict

    work_cfg = copy.deepcopy(cfg)
    if not isinstance(work_cfg, EasyDict):
        work_cfg = EasyDict(work_cfg)
    for k, v in overrides.items():
        work_cfg[k] = v

    dist_inited = False
    if need_model:
        from wan_va.distributed.util import init_distributed

        init_distributed(world_size, local_rank, rank)
        dist_inited = True
        work_cfg.rank = rank
        work_cfg.local_rank = local_rank
        work_cfg.world_size = world_size
    elif need_data and world_size > 1:
        work_cfg.rank = rank
        work_cfg.local_rank = local_rank
        work_cfg.world_size = world_size

    trainer = None
    dataloader = None

    if need_model:
        if rank == 0:
            print(
                f"[INFO] 加载 Trainer（模型+DataLoader）world_size={world_size} local_rank={local_rank} ..."
            )
        trainer = build_trainer_for_check(work_cfg)
        dataloader = trainer.train_loader
    elif need_data:
        if rank == 0:
            print("[INFO] 仅构建 DataLoader（不加载 Transformer）...")
        from wan_va.dataset import MultiLatentLeRobotDataset
        from torch.utils.data import DataLoader, DistributedSampler

        ds = MultiLatentLeRobotDataset(config=work_cfg)
        if world_size > 1:
            sampler = DistributedSampler(
                ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=42,
            )
            dataloader = DataLoader(
                ds,
                batch_size=work_cfg.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=getattr(work_cfg, "load_worker", 0),
            )
        else:
            dataloader = DataLoader(
                ds,
                batch_size=work_cfg.batch_size,
                shuffle=True,
                num_workers=getattr(work_cfg, "load_worker", 0),
            )

    running_stats: dict = {}
    batch_times: List[float] = []
    forward_batch = None

    if need_data and dataloader is not None:
        if rank == 0:
            print(f"[INFO] 空跑 DataLoader 前 {args.max_batches} 个 batch (world_size={world_size})...")
        t_scan0 = time.time()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.max_batches:
                break
            if rank == 0:
                t0 = time.time()
                batch_report = {
                    "batch_idx": batch_idx,
                    "basic_info": collect_basic_info(batch),
                    "prompt_info": try_decode_prompt(batch),
                    "tensor_summaries": {},
                    "problems": [],
                }
                recursive_collect_stats(
                    batch, "batch", running_stats, batch_report["tensor_summaries"], batch_report["problems"]
                )
                inspect_action_space(batch, batch_report)
                inspect_text_length(batch, batch_report)
                inspect_raw_text_emb(batch, batch_report)
                inspect_embodied_actions_raw(batch, work_cfg, batch_report)
                for msg in batch_report.get("embodied_action_check", {}).get("issues", []):
                    batch_report["problems"].append(f"embodied: {msg}")

            if batch_idx == args.forward_batch_idx:
                forward_batch = {
                    k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
                }
                if rank == 0:
                    save_debug_tensors(batch, save_dir, batch_idx=batch_idx)

            if rank == 0:
                log_line(save_dir / "batches.jsonl", batch_report)
                report["batches"].append(
                    {
                        "batch_idx": batch_idx,
                        "num_problems": len(batch_report["problems"]),
                        "text_check": batch_report.get("text_check", {}),
                        "raw_text_check": batch_report.get("raw_text_check", {}),
                        "embodied_action_check": batch_report.get("embodied_action_check", {}),
                        "action_check_brief": {
                            "exists": batch_report.get("action_check", {}).get("exists", False),
                            "noisy_latents_shape": batch_report.get("action_check", {}).get(
                                "noisy_latents_shape", None
                            ),
                        },
                    }
                )
                if batch_report["problems"]:
                    report["problems"].extend(
                        [f"batch {batch_idx}: {x}" for x in batch_report["problems"]]
                    )

                batch_times.append(time.time() - t0)
                if batch_idx % 10 == 0:
                    print(f"[INFO] scanned batch {batch_idx}")

        if rank == 0:
            report["dataloader_timing"] = {
                "total_wall_sec": time.time() - t_scan0,
                "per_batch_mean_sec": float(np.mean(batch_times)) if batch_times else 0.0,
                "per_batch_p50_sec": float(np.percentile(batch_times, 50)) if batch_times else 0.0,
                "per_batch_p95_sec": float(np.percentile(batch_times, 95)) if batch_times else 0.0,
                "per_batch_max_sec": float(max(batch_times)) if batch_times else 0.0,
                "num_batches": len(batch_times),
            }

            stats_summary = {k: v.summary() for k, v in running_stats.items()}
            report["running_stats"] = stats_summary
            norm_range_audit(stats_summary, report["problems"])

            text_lens = set()
            for brief in report["batches"]:
                rtc = brief.get("raw_text_check", {})
                if "text_seq_len" in rtc:
                    text_lens.add(rtc["text_seq_len"])
            if text_lens:
                report["diagnosis_text_seq_lens"] = sorted(text_lens)

    if dist_inited:
        dist.barrier()

    if "forward" in phases and not args.no_model:
        if trainer is None or forward_batch is None:
            if rank == 0:
                report["forward_check"] = {"ok": False, "error": "缺少模型或未迭代到 forward_batch_idx"}
        else:
            if rank == 0:
                print("[INFO] 前向连通性检查（与训练相同的 input_dict）...")
            fc = forward_sanity_check(trainer, forward_batch, args.device)
            if rank == 0:
                report["forward_check"] = fc
                if fc.get("ok"):
                    print("[INFO] 前向通过")
                else:
                    print("[ERROR] 前向失败:", fc.get("error"))
                    report["problems"].append(f"forward: {fc.get('error')}")

    if dist_inited:
        dist.barrier()

    if "smoke" in phases and not args.no_model:
        if trainer is None or forward_batch is None:
            if rank == 0:
                report["smoke_train"] = {"ok": False, "error": "缺少模型或未迭代到 forward_batch_idx"}
        else:
            if rank == 0:
                print("[INFO] Smoke：单步 backward + optimizer（不看收敛）...")
            st = smoke_train_step(trainer, forward_batch)
            if rank == 0:
                report["smoke_train"] = st
                if st.get("ok"):
                    print("[INFO] Smoke 通过:", st)
                else:
                    print("[ERROR] Smoke 失败:", st.get("error"))
                    report["problems"].append(f"smoke: {st.get('error')}")

    if dist_inited:
        dist.barrier()

    if rank == 0:
        dump_json(report, save_dir / "summary.json")

        print("\n========== SUMMARY ==========")
        print(
            json.dumps(
                {
                    "num_batches_scanned": len(report["batches"]),
                    "num_problems": len(report["problems"]),
                    "dataloader_timing": report.get("dataloader_timing", {}),
                    "forward_check": report.get("forward_check"),
                    "smoke_train": report.get("smoke_train"),
                    "save_dir": str(save_dir),
                    "world_size": world_size,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print("详细 batch 日志:", save_dir / "batches.jsonl")
        print("总报告:", save_dir / "summary.json")
        if (save_dir / "visual_debug").exists():
            print("可视化反查目录:", save_dir / "visual_debug")

    if dist_inited:
        dist.destroy_process_group()


if __name__ == "__main__":
    from wan_va.utils import init_logger

    init_logger()
    main()
