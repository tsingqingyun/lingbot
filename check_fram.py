#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wan_va.configs import get_config
from wan_va.dataset import MultiLatentLeRobotDataset
from wan_va.modules.model import FlexAttnFunc

def to_python(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return [to_python(v) for v in x]
    if isinstance(x, dict):
        return {k: to_python(v) for k, v in x.items()}
    return x


def tensor_stats(x: torch.Tensor):
    x = x.detach().float().cpu()
    finite_mask = torch.isfinite(x)
    finite_x = x[finite_mask]

    out = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "numel": int(x.numel()),
        "nan_count": int(torch.isnan(x).sum().item()),
        "inf_count": int(torch.isinf(x).sum().item()),
    }

    if finite_x.numel() == 0:
        out.update({
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_mean": None,
            "abs_max": None,
        })
        return out

    out.update({
        "min": float(finite_x.min().item()),
        "max": float(finite_x.max().item()),
        "mean": float(finite_x.mean().item()),
        "std": float(finite_x.std(unbiased=False).item()) if finite_x.numel() > 1 else 0.0,
        "abs_mean": float(finite_x.abs().mean().item()),
        "abs_max": float(finite_x.abs().max().item()),
    })
    return out


def masked_tensor_stats(x: torch.Tensor, mask: torch.Tensor):
    x = x.detach().float().cpu()
    mask = mask.detach().float().cpu()

    valid = mask > 0.5
    out = {
        "x_shape": list(x.shape),
        "mask_shape": list(mask.shape),
        "valid_count": int(valid.sum().item()),
        "valid_ratio": float(valid.float().mean().item()),
    }

    if valid.sum() == 0:
        out.update({
            "min": None, "max": None, "mean": None, "std": None,
            "abs_gt_1_ratio": None,
            "abs_gt_1_5_ratio": None,
            "abs_gt_3_ratio": None,
        })
        return out

    vals = x[valid]
    finite = torch.isfinite(vals)
    vals = vals[finite]

    if vals.numel() == 0:
        out.update({
            "min": None, "max": None, "mean": None, "std": None,
            "abs_gt_1_ratio": None,
            "abs_gt_1_5_ratio": None,
            "abs_gt_3_ratio": None,
        })
        return out

    abs_vals = vals.abs()
    out.update({
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
        "abs_gt_1_ratio": float((abs_vals > 1).float().mean().item()),
        "abs_gt_1_5_ratio": float((abs_vals > 1.5).float().mean().item()),
        "abs_gt_3_ratio": float((abs_vals > 3).float().mean().item()),
    })
    return out


def per_channel_masked_stats(x: torch.Tensor, mask: torch.Tensor, max_channels=64):
    """
    约定输入形状优先是 [B, C, F, H, W]。
    如果不是，也尽量在第 2 维当作 channel 做统计。
    """
    x = x.detach().float().cpu()
    mask = mask.detach().float().cpu()

    if x.ndim < 2:
        return {"error": "tensor ndim < 2, cannot do channel stats"}

    cdim = 1
    C = x.shape[cdim]
    if C > max_channels:
        return {"warning": f"channel too many ({C}), skip per-channel stats"}

    stats = []
    for c in range(C):
        xc = x.select(cdim, c)
        mc = mask.select(cdim, c) if mask.shape == x.shape else None

        if mc is None:
            vals = xc.reshape(-1)
        else:
            valid = mc > 0.5
            vals = xc[valid]

        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            stats.append({
                "channel": c,
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            })
        else:
            stats.append({
                "channel": c,
                "count": int(vals.numel()),
                "min": float(vals.min().item()),
                "max": float(vals.max().item()),
                "mean": float(vals.mean().item()),
                "std": float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
            })
    return stats


def summarize_strings(obj, max_items=3, max_len=160):
    out = []
    if isinstance(obj, str):
        out.append(obj[:max_len])
    elif isinstance(obj, (list, tuple)):
        for v in obj[:max_items]:
            if isinstance(v, str):
                out.append(v[:max_len])
            elif isinstance(v, (list, tuple)):
                for vv in v[:max_items]:
                    if isinstance(vv, str):
                        out.append(vv[:max_len])
    return out[:max_items]


def find_prompt_keys(batch):
    candidates = []
    for k in batch.keys():
        lk = k.lower()
        if any(t in lk for t in [
            "prompt", "instruction", "lang", "language", "text", "caption", "task", "desc"
        ]):
            candidates.append(k)
    return candidates


def find_image_keys(batch):
    candidates = []
    for k in batch.keys():
        lk = k.lower()
        if any(t in lk for t in [
            "image", "images", "pixel", "rgb", "frame", "obs"
        ]):
            candidates.append(k)
    return candidates


def aggregate_numeric(stats_list, key):
    vals = [s[key] for s in stats_list if isinstance(s, dict) and s.get(key) is not None]
    if not vals:
        return None
    return {
        "min_over_batches": float(min(vals)),
        "max_over_batches": float(max(vals)),
        "mean_over_batches": float(sum(vals) / len(vals)),
    }


def summarize_dataset_sampling(dataset):
    """
    汇总数据集采样模式信息：
    - 是否按帧数比例采样
    - sample_unit_frames
    - 每个 task 的 clip/frame 统计
    """
    sample_by_frames = bool(getattr(dataset, "sample_by_frames", True))
    sample_unit_frames = int(getattr(dataset, "sample_unit_frames", 1))
    expected_sampling_basis = "frame_ratio" if sample_by_frames else "clip_ratio"

    task_stats = {}
    total_clips = 0
    total_frames = 0

    datasets = getattr(dataset, "_datasets", [])
    for dset in datasets:
        repo_id = getattr(dset, "repo_id", "unknown_task")
        metas = getattr(dset, "new_metas", [])

        clip_count = len(metas)
        frame_count = 0

        for meta in metas:
            num_frames = int(meta.get("num_frames", meta["end_frame"] - meta["start_frame"]))
            num_frames = max(1, num_frames)
            frame_count += num_frames

        task_stats[repo_id] = {
            "clip_count": clip_count,
            "frame_count": frame_count,
        }
        total_clips += clip_count
        total_frames += frame_count

    for task_name, st in task_stats.items():
        st["clip_ratio"] = float(st["clip_count"] / total_clips) if total_clips > 0 else None
        st["frame_ratio"] = float(st["frame_count"] / total_frames) if total_frames > 0 else None

    return {
        "sample_by_frames": sample_by_frames,
        "sample_unit_frames": sample_unit_frames,
        "expected_sampling_basis": expected_sampling_basis,
        "total_clips": total_clips,
        "total_frames": total_frames,
        "task_stats": task_stats,
    }

def debug_cross_attention_mask(batch, config):
    """
    直接复用新版 model.py 里的 cross-attention mask 构造逻辑做检查。
    这里只做 shape / valid count / mask 语义检查，不跑整个模型 forward。
    """
    out = {
        "ok": False,
        "reason": None,
    }

    required_keys = ["latents", "text_emb"]
    for k in required_keys:
        if k not in batch:
            out["reason"] = f"missing required key: {k}"
            return out

    if "text_mask" not in batch:
        out["reason"] = "no text_mask in batch"
        return out

    latents = batch["latents"]
    text_emb = batch["text_emb"]
    text_mask = batch["text_mask"]

    if not isinstance(latents, torch.Tensor):
        out["reason"] = "latents is not tensor"
        return out
    if not isinstance(text_emb, torch.Tensor):
        out["reason"] = "text_emb is not tensor"
        return out
    if not isinstance(text_mask, torch.Tensor):
        out["reason"] = "text_mask is not tensor"
        return out

    if latents.ndim != 5:
        out["reason"] = f"latents ndim expected 5, got {latents.ndim}"
        return out

    device = latents.device
    B, C, F, H, W = latents.shape

    patch_size = getattr(config, "patch_size", [1, 2, 2])
    chunk_size = getattr(config, "chunk_size", 4)
    window_size = getattr(config, "window_size", 16)

    dummy_action_shape = (B, 30, F, 1, 1)

    latent_tokens_per_sample = (F // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2])
    action_tokens_per_sample = F * 1 * 1

    total_length = B * (
        latent_tokens_per_sample +
        latent_tokens_per_sample +
        action_tokens_per_sample +
        action_tokens_per_sample
    )
    padded_length = (128 - total_length % 128) % 128
    q_len = total_length + padded_length

    try:
        FlexAttnFunc.init_mask(
            latent_shape=latents.shape,
            action_shape=dummy_action_shape,
            padded_length=padded_length,
            chunk_size=chunk_size,
            window_size=window_size,
            patch_size=patch_size,
            text_len=text_emb.shape[1] if text_emb.ndim >= 2 else 0,
            device=device,
        )
    except Exception as e:
        out["reason"] = f"init_mask failed: {type(e).__name__}: {e}"
        return out

    try:
        cross_attention_mask = FlexAttnFunc.build_flattened_cross_attn_additive_mask(
            seq_ids=FlexAttnFunc.seq_ids.to(device),
            text_mask=text_mask.to(device),
            q_len=q_len,
            dtype=torch.bfloat16,
        )
    except Exception as e:
        out["reason"] = f"build_flattened_cross_attn_additive_mask failed: {type(e).__name__}: {e}"
        return out

    valid_zero = (cross_attention_mask == 0)
    finite_mask = torch.isfinite(cross_attention_mask)

    row0_valid = int(valid_zero[0, 0, 0].sum().item()) if q_len > 0 else None
    lastrow_valid = int(valid_zero[0, 0, -1].sum().item()) if q_len > 0 else None

    out.update({
        "ok": True,
        "latent_shape": list(latents.shape),
        "text_emb_shape": list(text_emb.shape),
        "text_mask_shape": list(text_mask.shape),
        "text_mask_dtype": str(text_mask.dtype),
        "text_mask_unique": to_python(torch.unique(text_mask)),
        "text_mask_first_row_64": to_python(text_mask[0, :64]) if text_mask.ndim >= 2 else None,

        "patch_size": patch_size,
        "chunk_size": chunk_size,
        "window_size": window_size,

        "latent_tokens_per_sample": latent_tokens_per_sample,
        "action_tokens_per_sample": action_tokens_per_sample,
        "total_length_before_pad": total_length,
        "padded_length": padded_length,
        "q_len": q_len,

        "seq_ids_shape": list(FlexAttnFunc.seq_ids.shape) if FlexAttnFunc.seq_ids is not None else None,
        "seq_ids_unique_head": to_python(torch.unique(FlexAttnFunc.seq_ids[: min(256, FlexAttnFunc.seq_ids.numel())])) if FlexAttnFunc.seq_ids is not None else None,

        "cross_mask_shape": list(cross_attention_mask.shape),
        "cross_mask_finite_ratio": float(finite_mask.float().mean().item()),
        "cross_mask_zero_ratio": float(valid_zero.float().mean().item()),
        "cross_mask_row0_valid_count": row0_valid,
        "cross_mask_lastrow_valid_count": lastrow_valid,
    })

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="robocasa_train")
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="./debug_dataloader")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = get_config(args.config_name)
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1

    # 调试时一般把 worker 调低，避免卡死时难定位
    config.load_worker = args.num_workers
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    dataset = MultiLatentLeRobotDataset(config=config)
    sampling_summary = summarize_dataset_sampling(dataset)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.load_worker,
    )

    print("=" * 80)
    print(f"[INFO] config_name = {args.config_name}")
    print(f"[INFO] dataset_path = {getattr(config, 'dataset_path', None)}")
    print(f"[INFO] batch_size   = {config.batch_size}")
    print(f"[INFO] num_workers  = {config.load_worker}")
    print(f"[INFO] dataset_len  = {len(dataset)}")
    print(f"[INFO] sample_by_frames = {sampling_summary['sample_by_frames']}")
    print(f"[INFO] sample_unit_frames = {sampling_summary['sample_unit_frames']}")
    print(f"[INFO] expected_sampling_basis = {sampling_summary['expected_sampling_basis']}")
    print("=" * 80)

    batch_reports = []
    prompt_seen = []
    first_batch_keys = None

    action_global_stats = []
    latent_global_stats = []
    image_global_stats = []
    text_emb_global_stats = []

    observed_task_sample_count = defaultdict(int)
    observed_task_latent_frames = defaultdict(int)

    for batch_idx, batch in enumerate(tqdm(loader, total=min(args.num_batches, len(loader)))):
        if batch_idx >= args.num_batches:
            break

        report = {
            "batch_idx": batch_idx,
            "keys": list(batch.keys()) if isinstance(batch, dict) else str(type(batch)),
        }

        if not isinstance(batch, dict):
            report["error"] = f"batch is not dict, got {type(batch)}"
            batch_reports.append(report)
            continue

        if first_batch_keys is None:
            first_batch_keys = list(batch.keys())
            print("[INFO] first batch keys:")
            for k in first_batch_keys:
                v = batch[k]
                if isinstance(v, torch.Tensor):
                    print(f"  - {k}: tensor shape={list(v.shape)} dtype={v.dtype}")
                else:
                    print(f"  - {k}: type={type(v)}")

        # 1) 通用张量统计
        tensor_info = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                tensor_info[k] = tensor_stats(v)
        report["tensor_stats"] = tensor_info

        # 2) prompt 检查
        prompt_keys = find_prompt_keys(batch)
        prompt_info = {
            "candidate_prompt_keys": prompt_keys,
            "samples": {},
            "note": None,
        }

        for k in prompt_keys:
            v = batch[k]
            if isinstance(v, torch.Tensor):
                # text_emb 这种情况
                prompt_info["samples"][k] = {
                    "type": "tensor",
                    "stats": tensor_stats(v),
                }
            else:
                samples = summarize_strings(v)
                if samples:
                    prompt_info["samples"][k] = {
                        "type": str(type(v)),
                        "samples": samples,
                    }
                    prompt_seen.extend(samples)

        if not prompt_keys:
            prompt_info["note"] = "batch 中没有明显的原始 prompt 字段；当前更可能只有 text_emb。"

        report["prompt_check"] = prompt_info

        # 3) 图像/latent 检查
        image_keys = find_image_keys(batch)
        image_info = {
            "candidate_image_keys": image_keys,
            "items": {}
        }

        for k in image_keys:
            v = batch[k]
            if isinstance(v, torch.Tensor):
                image_info["items"][k] = tensor_stats(v)

        # 单独把 latents 拿出来强调
        if "latents" in batch and isinstance(batch["latents"], torch.Tensor):
            latent_stat = tensor_stats(batch["latents"])
            report["latent_check"] = latent_stat
            latent_global_stats.append(latent_stat)
        else:
            report["latent_check"] = {"note": "no 'latents' key"}

        # 如果真有原始图像字段，也单独累计
        for k in image_keys:
            if isinstance(batch[k], torch.Tensor):
                image_global_stats.append(tensor_stats(batch[k]))

        report["image_check"] = image_info

        # 4) text_emb 检查
        if "text_emb" in batch and isinstance(batch["text_emb"], torch.Tensor):
            text_stat = tensor_stats(batch["text_emb"])
            report["text_emb_check"] = text_stat
            text_emb_global_stats.append(text_stat)
        else:
            report["text_emb_check"] = {"note": "no 'text_emb' key"}

        # 4.1) 新版 cross-attention mask 检查
        cross_mask_debug = debug_cross_attention_mask(batch, config)
        report["cross_attention_mask_check"] = cross_mask_debug

        if batch_idx == 0:
            print("\n[DEBUG] cross_attention_mask_check:")
            print(json.dumps(to_python(cross_mask_debug), ensure_ascii=False, indent=2))

        # 5) action + mask 归一化检查
        if "actions" in batch and isinstance(batch["actions"], torch.Tensor):
            if "actions_mask" in batch and isinstance(batch["actions_mask"], torch.Tensor):
                action_stat = masked_tensor_stats(batch["actions"], batch["actions_mask"])
                channel_stat = per_channel_masked_stats(batch["actions"], batch["actions_mask"])
                report["action_check"] = {
                    "masked_stats": action_stat,
                    "per_channel_stats": channel_stat,
                }
                action_global_stats.append(action_stat)
            else:
                action_stat = tensor_stats(batch["actions"])
                report["action_check"] = {
                    "stats": action_stat,
                    "note": "no actions_mask found"
                }
                action_global_stats.append(action_stat)
        else:
            report["action_check"] = {"note": "no 'actions' key"}

        # 6) 采样模式实际观测统计
        task_name_list = batch.get("task_name", [])
        latents = batch.get("latents", None)

        batch_frame_len = None
        if isinstance(latents, torch.Tensor) and latents.ndim >= 3:
            # 约定 latents 形状通常是 [B, C, F, H, W]
            batch_frame_len = int(latents.shape[2])

        if isinstance(task_name_list, (list, tuple)) and len(task_name_list) > 0:
            for t_name in task_name_list:
                if not isinstance(t_name, str):
                    continue
                observed_task_sample_count[t_name] += 1
                if batch_frame_len is not None:
                    observed_task_latent_frames[t_name] += batch_frame_len

        report["sampling_check"] = {
            "dataset_sample_by_frames": sampling_summary["sample_by_frames"],
            "dataset_sample_unit_frames": sampling_summary["sample_unit_frames"],
            "expected_sampling_basis": sampling_summary["expected_sampling_basis"],
            "batch_task_names": list(task_name_list) if isinstance(task_name_list, (list, tuple)) else str(type(task_name_list)),
            "batch_latent_frame_len": batch_frame_len,
        }

        batch_reports.append(report)

    # 理论分布
    expected_task_ratios = {}
    for task_name, st in sampling_summary["task_stats"].items():
        key = sampling_summary["expected_sampling_basis"]
        expected_task_ratios[task_name] = st.get(key, None)

    # 实际分布：按样本数
    total_obs_samples = sum(observed_task_sample_count.values())
    observed_sample_ratios = {}
    if total_obs_samples > 0:
        observed_sample_ratios = {
            k: float(v / total_obs_samples)
            for k, v in observed_task_sample_count.items()
        }

    # 实际分布：按 latent 帧
    total_obs_frames = sum(observed_task_latent_frames.values())
    observed_frame_ratios = {}
    if total_obs_frames > 0:
        observed_frame_ratios = {
            k: float(v / total_obs_frames)
            for k, v in observed_task_latent_frames.items()
        }

    summary = {
        "config_name": args.config_name,
        "dataset_path": getattr(config, "dataset_path", None),
        "num_batches_checked": len(batch_reports),
        "first_batch_keys": first_batch_keys,
        "prompt_preview": prompt_seen[:10],

        "sampling_check": {
            "sample_by_frames": sampling_summary["sample_by_frames"],
            "sample_unit_frames": sampling_summary["sample_unit_frames"],
            "expected_sampling_basis": sampling_summary["expected_sampling_basis"],
            "total_clips": sampling_summary["total_clips"],
            "total_frames": sampling_summary["total_frames"],
            "task_stats": sampling_summary["task_stats"],
            "expected_task_sampling_ratios": expected_task_ratios,
            "observed_task_sample_count": dict(observed_task_sample_count),
            "observed_task_sample_ratios": observed_sample_ratios,
            "observed_task_latent_frames": dict(observed_task_latent_frames),
            "observed_task_frame_ratios": observed_frame_ratios,
        },

        "global_summary": {
            "latents": {
                "min": aggregate_numeric(latent_global_stats, "min"),
                "max": aggregate_numeric(latent_global_stats, "max"),
                "mean": aggregate_numeric(latent_global_stats, "mean"),
                "std": aggregate_numeric(latent_global_stats, "std"),
                "abs_max": aggregate_numeric(latent_global_stats, "abs_max"),
            },
            "text_emb": {
                "min": aggregate_numeric(text_emb_global_stats, "min"),
                "max": aggregate_numeric(text_emb_global_stats, "max"),
                "mean": aggregate_numeric(text_emb_global_stats, "mean"),
                "std": aggregate_numeric(text_emb_global_stats, "std"),
                "abs_max": aggregate_numeric(text_emb_global_stats, "abs_max"),
            },
            "actions": {
                "min": aggregate_numeric(action_global_stats, "min"),
                "max": aggregate_numeric(action_global_stats, "max"),
                "mean": aggregate_numeric(action_global_stats, "mean"),
                "std": aggregate_numeric(action_global_stats, "std"),
                "abs_gt_1_ratio": aggregate_numeric(action_global_stats, "abs_gt_1_ratio"),
                "abs_gt_1_5_ratio": aggregate_numeric(action_global_stats, "abs_gt_1_5_ratio"),
                "abs_gt_3_ratio": aggregate_numeric(action_global_stats, "abs_gt_3_ratio"),
                "valid_ratio": aggregate_numeric(action_global_stats, "valid_ratio"),
            },
            "images_or_obs": {
                "min": aggregate_numeric(image_global_stats, "min"),
                "max": aggregate_numeric(image_global_stats, "max"),
                "mean": aggregate_numeric(image_global_stats, "mean"),
                "std": aggregate_numeric(image_global_stats, "std"),
                "abs_max": aggregate_numeric(image_global_stats, "abs_max"),
            }
        }
    }

    jsonl_path = save_dir / "batch_reports.jsonl"
    summary_path = save_dir / "summary.json"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in batch_reports:
            f.write(json.dumps(to_python(r), ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("[DONE] checked batches:", len(batch_reports))
    print("[DONE] saved batch reports to:", jsonl_path)
    print("[DONE] saved summary to:", summary_path)
    print("=" * 80)

    print("\n[SUMMARY]")
    print(json.dumps(to_python(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()