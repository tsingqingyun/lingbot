#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LingBot-VA 全面检测脚本
1) 输入核对：完整保存 prompt、图像样本和关键张量切片
2) 小规模链路验证：可选 Trainer 前向 + 可选 1 步 smoke train
3) 归一化检查：统计超出 [norm_low, norm_high] 的异常值数量与比例
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wan_va.configs import get_config
from wan_va.dataset import MultiLatentLeRobotDataset


PROMPT_HINTS = [
    "prompt", "instruction", "text", "caption", "task", "desc", "lang", "language"
]
IMAGE_HINTS = [
    "image", "images", "pixel", "rgb", "frame", "obs", "camera"
]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        if obj.numel() <= 16:
            return obj.detach().cpu().tolist()
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    return str(obj)


def dump_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def append_jsonl(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=_json_default) + "\n")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    t = x.detach().float().cpu()
    finite = torch.isfinite(t)
    finite_vals = t[finite]
    out = {
        "shape": list(t.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "numel": int(t.numel()),
        "nan_count": int(torch.isnan(t).sum().item()),
        "inf_count": int(torch.isinf(t).sum().item()),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "abs_max": None,
    }
    if finite_vals.numel() > 0:
        out["min"] = float(finite_vals.min().item())
        out["max"] = float(finite_vals.max().item())
        out["mean"] = float(finite_vals.mean().item())
        out["std"] = float(finite_vals.std(unbiased=False).item()) if finite_vals.numel() > 1 else 0.0
        out["abs_max"] = float(finite_vals.abs().max().item())
    return out


def infer_batch_size(batch: Dict[str, Any]) -> int:
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.ndim > 0:
            return int(v.shape[0])
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return len(v)
    return 1


def find_keys_by_hints(batch: Dict[str, Any], hints: List[str]) -> List[str]:
    out = []
    for k in batch.keys():
        lk = k.lower()
        if any(h in lk for h in hints):
            out.append(k)
    return out


def flatten_strings(obj: Any, out: List[str]):
    if isinstance(obj, str):
        out.append(obj)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            flatten_strings(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            flatten_strings(v, out)
        return


def extract_sample_texts(obj: Any, sample_idx: int, batch_size: int) -> List[str]:
    if isinstance(obj, (list, tuple)) and len(obj) == batch_size:
        out: List[str] = []
        flatten_strings(obj[sample_idx], out)
        return out
    out: List[str] = []
    flatten_strings(obj, out)
    return out


def tensor_to_visual_image(x: torch.Tensor) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    arr = x.detach().float().cpu().numpy()
    meta = {
        "input_shape": list(arr.shape),
        "input_dtype": str(x.dtype),
    }

    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[-1] in (1, 3, 4):
            pass
        else:
            arr = arr[0]

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim not in (2, 3):
        meta["error"] = "cannot convert to 2D/3D image"
        return None, meta

    finite = np.isfinite(arr)
    if not finite.any():
        img = np.zeros_like(arr, dtype=np.uint8)
        meta["note"] = "all values are non-finite"
        return img, meta

    vals = arr[finite]
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())

    if hi <= lo:
        img = np.zeros_like(arr, dtype=np.uint8)
        meta["note"] = "degenerate value range"
        meta["vis_min"] = lo
        meta["vis_max"] = hi
        return img, meta

    clipped = np.clip(arr, lo, hi)
    norm = (clipped - lo) / (hi - lo + 1e-8)
    img = (norm * 255.0).astype(np.uint8)
    meta["vis_min"] = lo
    meta["vis_max"] = hi
    return img, meta


def save_png_or_npy(img: np.ndarray, path_stem: Path) -> str:
    try:
        from PIL import Image

        if img.ndim == 2:
            Image.fromarray(img, mode="L").save(str(path_stem) + ".png")
            return "png"
        if img.ndim == 3 and img.shape[-1] == 3:
            Image.fromarray(img, mode="RGB").save(str(path_stem) + ".png")
            return "png"
        if img.ndim == 3 and img.shape[-1] == 4:
            Image.fromarray(img, mode="RGBA").save(str(path_stem) + ".png")
            return "png"

        if img.ndim == 3:
            one = img[..., 0]
        else:
            one = img
        Image.fromarray(one, mode="L").save(str(path_stem) + ".png")
        return "png"
    except Exception:
        np.save(str(path_stem) + ".npy", img)
        return "npy"


def dump_input_snapshot(
    batch: Dict[str, Any],
    batch_idx: int,
    save_dir: Path,
    samples_per_batch: int = 2,
    max_image_keys: int = 8,
):
    batch_dir = save_dir / "input_dump" / f"batch_{batch_idx:04d}"
    raw_dir = batch_dir / "raw_tensors"
    img_dir = batch_dir / "images"
    batch_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    batch_size = infer_batch_size(batch)

    prompt_keys = find_keys_by_hints(batch, PROMPT_HINTS)
    prompt_dump: Dict[str, Dict[str, List[str]]] = {}
    for k in prompt_keys:
        v = batch[k]
        per_sample: Dict[str, List[str]] = {}
        for sid in range(min(batch_size, samples_per_batch)):
            txts = extract_sample_texts(v, sid, batch_size)
            if txts:
                per_sample[str(sid)] = txts
        if per_sample:
            prompt_dump[k] = per_sample
    dump_json(batch_dir / "prompt_full.json", prompt_dump)

    tensor_meta = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            tensor_meta[k] = tensor_stats(v)
    dump_json(batch_dir / "tensor_meta.json", tensor_meta)

    image_keys = find_keys_by_hints(batch, IMAGE_HINTS)[:max_image_keys]
    image_manifest = []
    for k in image_keys:
        v = batch.get(k, None)
        if not isinstance(v, torch.Tensor):
            continue

        for sid in range(min(batch_size, samples_per_batch)):
            if v.ndim > 0 and v.shape[0] == batch_size:
                sample = v[sid]
            else:
                sample = v

            raw_path = raw_dir / f"{k}_s{sid:02d}.pt"
            torch.save(sample.detach().cpu(), raw_path)

            img, meta = tensor_to_visual_image(sample)
            rec = {
                "key": k,
                "sample_idx": sid,
                "raw_path": str(raw_path),
                "meta": meta,
                "saved_as": None,
                "vis_path": None,
            }
            if img is not None:
                stem = img_dir / f"{k}_s{sid:02d}"
                fmt = save_png_or_npy(img, stem)
                rec["saved_as"] = fmt
                rec["vis_path"] = str(stem) + (".png" if fmt == "png" else ".npy")
            image_manifest.append(rec)

    dump_json(batch_dir / "image_manifest.json", image_manifest)


class NormOutlierTracker:
    def __init__(self, low: float, high: float):
        self.low = float(low)
        self.high = float(high)
        self.stats = defaultdict(lambda: {
            "selected_total": 0,
            "finite_total": 0,
            "nonfinite_total": 0,
            "outlier_total": 0,
            "below_total": 0,
            "above_total": 0,
            "global_min": None,
            "global_max": None,
            "batch_outlier_ratios": [],
        })

    def update(self, key: str, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        t = x.detach().float().cpu()

        if mask is not None:
            m = mask.detach().float().cpu()
            try:
                active = m > 0.5
                if active.shape != t.shape:
                    active = active.expand_as(t)
            except Exception:
                active = torch.ones_like(t, dtype=torch.bool)
        else:
            active = torch.ones_like(t, dtype=torch.bool)

        vals = t[active]
        selected_total = int(vals.numel())
        finite_mask = torch.isfinite(vals)
        finite_vals = vals[finite_mask]
        finite_total = int(finite_vals.numel())
        nonfinite_total = selected_total - finite_total

        below_total = 0
        above_total = 0
        outlier_total = 0
        cur_min = None
        cur_max = None

        if finite_total > 0:
            below = finite_vals < self.low
            above = finite_vals > self.high
            below_total = int(below.sum().item())
            above_total = int(above.sum().item())
            outlier_total = below_total + above_total
            cur_min = float(finite_vals.min().item())
            cur_max = float(finite_vals.max().item())

        st = self.stats[key]
        st["selected_total"] += selected_total
        st["finite_total"] += finite_total
        st["nonfinite_total"] += nonfinite_total
        st["outlier_total"] += outlier_total
        st["below_total"] += below_total
        st["above_total"] += above_total
        if cur_min is not None:
            st["global_min"] = cur_min if st["global_min"] is None else min(st["global_min"], cur_min)
            st["global_max"] = cur_max if st["global_max"] is None else max(st["global_max"], cur_max)

        ratio = None
        if finite_total > 0:
            ratio = float(outlier_total / finite_total)
        st["batch_outlier_ratios"].append(ratio)

        return {
            "selected_total": selected_total,
            "finite_total": finite_total,
            "nonfinite_total": nonfinite_total,
            "outlier_total": outlier_total,
            "outlier_ratio": ratio,
            "min": cur_min,
            "max": cur_max,
        }

    def summary(self) -> Dict[str, Any]:
        out = {}
        for k, st in self.stats.items():
            ratios = [r for r in st["batch_outlier_ratios"] if r is not None]
            ratio_mean = float(np.mean(ratios)) if ratios else None
            ratio_p95 = float(np.percentile(ratios, 95)) if ratios else None
            ratio_max = float(np.max(ratios)) if ratios else None
            global_ratio = (
                float(st["outlier_total"] / st["finite_total"]) if st["finite_total"] > 0 else None
            )
            out[k] = {
                "range_expected": [self.low, self.high],
                "selected_total": st["selected_total"],
                "finite_total": st["finite_total"],
                "nonfinite_total": st["nonfinite_total"],
                "outlier_total": st["outlier_total"],
                "below_total": st["below_total"],
                "above_total": st["above_total"],
                "global_outlier_ratio": global_ratio,
                "batch_outlier_ratio_mean": ratio_mean,
                "batch_outlier_ratio_p95": ratio_p95,
                "batch_outlier_ratio_max": ratio_max,
                "global_min": st["global_min"],
                "global_max": st["global_max"],
            }
        return out


def ensure_dist_defaults():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def read_dist_env() -> Tuple[int, int, int]:
    ensure_dist_defaults()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return rank, local_rank, world_size


def summarize_model_output(out: Any) -> Dict[str, Any]:
    if isinstance(out, torch.Tensor):
        return {"type": "tensor", "stats": tensor_stats(out)}
    if isinstance(out, tuple):
        items = []
        for i, x in enumerate(out):
            if isinstance(x, torch.Tensor):
                items.append({"idx": i, "type": "tensor", "stats": tensor_stats(x)})
            else:
                items.append({"idx": i, "type": str(type(x))})
        return {"type": "tuple", "len": len(out), "items": items}
    return {"type": str(type(out))}


def run_small_scale_sanity(cfg: Any, args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "forward_steps": args.forward_steps,
        "smoke_train_step": args.smoke_train_step,
        "steps": [],
    }

    dist_started_here = False
    try:
        from wan_va.distributed.util import init_distributed
        from wan_va.train import Trainer

        rank, local_rank, world_size = read_dist_env()
        if not dist.is_initialized():
            init_distributed(world_size, local_rank, rank)
            dist_started_here = True

        cfg2 = copy.deepcopy(cfg)
        cfg2.rank = rank
        cfg2.local_rank = local_rank
        cfg2.world_size = world_size
        cfg2.enable_wandb = False
        if args.batch_size is not None:
            cfg2.batch_size = args.batch_size
        cfg2.load_worker = args.num_workers

        trainer = Trainer(cfg2)
        loader = trainer.train_loader
        it = iter(loader)

        for step in range(args.forward_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            t0 = time.time()
            with torch.no_grad():
                batch_dev = trainer.convert_input_format({k: v for k, v in batch.items()})
                input_dict = trainer._prepare_input_dict(batch_dev)
                out = trainer.transformer(input_dict, train_mode=True)
            elapsed = time.time() - t0

            result["steps"].append({
                "step": step,
                "time_sec": elapsed,
                "output": summarize_model_output(out),
            })

        if args.smoke_train_step:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            trainer.transformer.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            smoke = trainer._train_step(batch, batch_idx=0)
            smoke_out = {"ok": True}
            for k, v in smoke.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        smoke_out[k] = float(v.item())
                    else:
                        smoke_out[k] = str(v.shape)
                else:
                    smoke_out[k] = v
            result["smoke_result"] = smoke_out

        result["ok"] = True
        return result

    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        return result

    finally:
        if dist_started_here and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def main():
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser("LingBot-VA comprehensive checker")
    parser.add_argument("--config-name", type=str, default="robocasa_train")
    parser.add_argument("--save-dir", type=str, default="./debug_comprehensive_check")

    parser.add_argument("--num-batches", type=int, default=100, help="用于数据与归一化统计的 batch 数")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dump-batches", type=int, default=3, help="前多少个 batch 做输入全量落盘")
    parser.add_argument("--dump-samples-per-batch", type=int, default=2)
    parser.add_argument("--norm-keys", type=str, default="actions", help="逗号分隔，如 actions,latents")
    parser.add_argument("--norm-low", type=float, default=-1.0)
    parser.add_argument("--norm-high", type=float, default=1.0)

    parser.add_argument("--run-forward", action="store_true", help="是否做小规模前向链路测试")
    parser.add_argument("--forward-steps", type=int, default=1)
    parser.add_argument("--smoke-train-step", action="store_true", help="是否附加 1 步 train smoke")

    args = parser.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = get_config(args.config_name)
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.enable_wandb = False
    config.load_worker = args.num_workers
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    print("=" * 100)
    print("[INFO] config_name:", args.config_name)
    print("[INFO] dataset_path:", getattr(config, "dataset_path", None))
    print("[INFO] batch_size:", getattr(config, "batch_size", None))
    print("[INFO] load_worker:", getattr(config, "load_worker", None))
    print("[INFO] num_batches:", args.num_batches)
    print("[INFO] normalization range:", [args.norm_low, args.norm_high])
    print("=" * 100)

    dataset = MultiLatentLeRobotDataset(config=config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.load_worker,
    )

    norm_keys = [k.strip() for k in args.norm_keys.split(",") if k.strip()]
    tracker = NormOutlierTracker(args.norm_low, args.norm_high)

    jsonl_path = save_dir / "batch_reports.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    batch_reports = []
    task_distribution = {}
    t_scan0 = time.time()

    total_to_scan = min(args.num_batches, len(loader))
    for batch_idx, batch in enumerate(tqdm(loader, total=total_to_scan, desc="Scanning Dataloader")):
        if batch_idx >= args.num_batches:
            break

        t0 = time.time()
        report: Dict[str, Any] = {
            "batch_idx": batch_idx,
            "error": None,
            "keys": [],
            "prompt_preview": {},
            "image_candidate_keys": [],
            "norm_brief": {},
            "tensor_stats": {},
        }

        if not isinstance(batch, dict):
            report["error"] = f"batch is not dict: {type(batch)}"
            append_jsonl(jsonl_path, report)
            batch_reports.append(report)
            continue

        batch_size = infer_batch_size(batch)
        report["keys"] = list(batch.keys())

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                report["tensor_stats"][k] = tensor_stats(v)

        prompt_keys = find_keys_by_hints(batch, PROMPT_HINTS)
        for k in prompt_keys:
            txts = extract_sample_texts(batch[k], sample_idx=0, batch_size=batch_size)
            if txts:
                report["prompt_preview"][k] = txts[:2]

        report["image_candidate_keys"] = find_keys_by_hints(batch, IMAGE_HINTS)

        for nk in norm_keys:
            v = batch.get(nk, None)
            if isinstance(v, torch.Tensor):
                mask = None
                if nk == "actions":
                    m = batch.get("actions_mask", None)
                    if isinstance(m, torch.Tensor):
                        mask = m
                report["norm_brief"][nk] = tracker.update(nk, v, mask=mask)
            else:
                report["norm_brief"][nk] = {"note": "missing or not tensor"}

        task_name_list = batch.get("task_name", [])
        if len(task_name_list) > 0:
            for t_name in task_name_list:
                if isinstance(t_name, str):
                    task_distribution[t_name] = task_distribution.get(t_name, 0) + 1

        if batch_idx < args.dump_batches:
            dump_input_snapshot(
                batch=batch,
                batch_idx=batch_idx,
                save_dir=save_dir,
                samples_per_batch=args.dump_samples_per_batch,
            )

        report["scan_time_sec"] = time.time() - t0
        append_jsonl(jsonl_path, report)
        batch_reports.append(report)

    scan_time = time.time() - t_scan0
    norm_summary = tracker.summary()

    diagnosis = []
    for nk, st in norm_summary.items():
        ratio = st.get("global_outlier_ratio", None)
        if ratio is not None:
            diagnosis.append(
                f"{nk}: outlier_total={st['outlier_total']}, finite_total={st['finite_total']}, "
                f"global_outlier_ratio={ratio:.6f}, range=[{args.norm_low}, {args.norm_high}]"
            )

    if task_distribution:
        total_samples = sum(task_distribution.values())
        task_ratios = {k: f"{(v/total_samples)*100:.2f}%" for k, v in task_distribution.items()}
        diagnosis.append(f"dynamic_sampling_ratios: {task_ratios}")

    summary: Dict[str, Any] = {
        "config_name": args.config_name,
        "dataset_path": getattr(config, "dataset_path", None),
        "num_batches_checked": len(batch_reports),
        "scan_total_time_sec": scan_time,
        "avg_batch_time_sec": float(scan_time / max(len(batch_reports), 1)),
        "task_distribution": task_distribution,
        "norm_keys": norm_keys,
        "norm_range": [args.norm_low, args.norm_high],
        "normalization_summary": norm_summary,
        "diagnosis": diagnosis,
        "input_dump_dir": str(save_dir / "input_dump"),
        "batch_report_jsonl": str(jsonl_path),
        "small_scale_sanity": None,
    }

    if args.run_forward:
        print("[INFO] running small-scale sanity forward ...")
        summary["small_scale_sanity"] = run_small_scale_sanity(config, args)

    summary_path = save_dir / "summary.json"
    dump_json(summary_path, summary)

    print("\n" + "=" * 100)
    print("[DONE] checked batches:", len(batch_reports))
    print("[DONE] batch reports:", jsonl_path)
    print("[DONE] summary:", summary_path)
    if args.run_forward:
        ok = summary["small_scale_sanity"].get("ok", False)
        print("[DONE] forward sanity:", "PASS" if ok else "FAIL")
        if not ok:
            print("[ERROR]", summary["small_scale_sanity"].get("error"))
    print("=" * 100)

    print("\n[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

