# tools/check_pipeline.py
# 用法示例：
# python tools/check_pipeline.py \
#   --config your_train_config.py \
#   --max_batches 100 \
#   --forward_batch_idx 0 \
#   --save_dir ./debug_check

import os
import json
import math
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch


def to_cpu_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu()
    return x


def tensor_stats(x: torch.Tensor):
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
    """
    用于累计前 max_batches 个 batch 的统计
    """
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = None
        self.max = None

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

    def summary(self):
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        std = math.sqrt(var)
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "min": self.min,
            "max": self.max,
        }


def safe_shape(x):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        return [len(x)]
    return str(type(x))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def log_line(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def try_decode_prompt(batch):
    """
    尽量从 batch 中拿到原始 prompt 或可读文本。
    这里做了宽松兼容，你可以按自己的 dataset 再加字段。
    """
    candidates = [
        "prompt", "prompts", "text", "texts", "caption", "captions",
        "instruction", "instructions"
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


def collect_basic_info(batch):
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


def check_tensor(name, x: torch.Tensor, problems: list, summary: dict = None):
    st = tensor_stats(x)
    if summary is not None:
        summary[name] = st

    if x.is_floating_point():
        if st["nan_count"] > 0:
            problems.append(f"{name}: 含 NaN，数量={st['nan_count']}")
        if st["inf_count"] > 0:
            problems.append(f"{name}: 含 Inf，数量={st['inf_count']}")
        if st["std"] is not None and st["std"] == 0.0:
            problems.append(f"{name}: 标准差为 0，可能全常数")
    return st


def update_running_stats(stats_dict, prefix, x):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        key = prefix
        if key not in stats_dict:
            stats_dict[key] = RunningStats()
        stats_dict[key].update(x)


def recursive_collect_stats(obj, prefix, stats_dict, batch_summary, problems):
    if isinstance(obj, dict):
        for k, v in obj.items():
            recursive_collect_stats(v, f"{prefix}.{k}" if prefix else k, stats_dict, batch_summary, problems)
    elif isinstance(obj, torch.Tensor):
        check_tensor(prefix, obj, problems, batch_summary)
        update_running_stats(stats_dict, prefix, obj)


def inspect_action_space(batch, report):
    """
    检查 action_dict 里 noisy_latents / latent 的维度和取值
    """
    out = {}
    if "action_dict" not in batch:
        out["exists"] = False
        report["action_check"] = out
        return

    out["exists"] = True
    ad = batch["action_dict"]
    for k in ["noisy_latents", "latent", "timesteps", "cond_timesteps", "grid_id"]:
        if k in ad and isinstance(ad[k], torch.Tensor):
            out[k] = tensor_stats(ad[k])

    # 检查 action 维度是否像 [B, C, F, H, W] 或 [B, L, C]
    if "noisy_latents" in ad and isinstance(ad["noisy_latents"], torch.Tensor):
        shape = list(ad["noisy_latents"].shape)
        out["noisy_latents_rank"] = len(shape)
        out["noisy_latents_shape"] = shape
        if len(shape) not in [3, 5]:
            out["warning"] = f"action_dict.noisy_latents 维度为 {shape}，不是常见 3D/5D 结构"

    report["action_check"] = out


def inspect_text_length(batch, report):
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
                        f"text_emb 第二维是 {shape[1]}，不是 512。"
                        "如果你在 flex cross-attn mask 里写死 512，会直接报错。"
                    )
    report["text_check"] = out


def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    else:
        return batch


def maybe_get_dataloader_and_model_from_config(config_path):
    """
    你需要按自己的项目稍微改这里。
    默认思路是：
    1) import 你的训练配置
    2) 拿到 train dataloader
    3) 拿到 model
    """
    import importlib
    import sys

    # 检查并安装必要的依赖
    try:
        import torch
    except ImportError:
        import subprocess
        print("[INFO] Installing torch and related packages...")
        subprocess.check_call(['python', '-m', 'pip', 'install', 'torch==2.9.0', 'torchvision==0.24.0', 'torchaudio==2.9.0', 'diffusers==0.36.0', 'transformers==4.55.2', 'accelerate', 'einops', 'easydict', 'flash_attn', 'numpy==1.26.4', 'tqdm', 'imageio[ffmpeg]', 'websockets', 'msgpack', 'opencv-python', 'matplotlib', 'ftfy', 'safetensors', 'Pillow', 'lerobot==0.3.3', 'scipy', 'wandb'])
        import torch

    # 添加项目根目录到 sys.path 以支持相对导入
    project_root = '/cephfs/shared/xcx/lingbot-va'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 从 config_path 提取模块名
    if config_path.startswith(project_root):
        rel_path = config_path[len(project_root) + 1:]  # e.g., 'wan_va/configs/va_robocasa_train_cfg.py'
        module_name = rel_path.replace('/', '.').replace('.py', '')
        user_cfg = importlib.import_module(module_name)
    else:
        # Fallback to old method if path doesn't match
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_cfg", config_path)
        user_cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_cfg)

    # ===== 你需要根据自己的项目改这里 =====
    # 下面是“约定式接口”，如果没有就自己替换：
    # - build_model()
    # - build_train_dataloader()
    if not hasattr(user_cfg, "build_model"):
        # 如果没有 build_model，手动构建模型
        # 例如，从 wan_va.train 导入 Trainer 类，然后实例化
        from wan_va.train import Trainer
        from configs import get_config
        config = get_config('robocasa_train')
        # 添加分布式相关的属性，因为 Trainer 需要这些
        config.rank = 0
        config.local_rank = 0
        config.world_size = 1
        trainer = Trainer(config)
        model = trainer.transformer  # 假设模型是 transformer
    else:
        model = user_cfg.build_model()

    if not hasattr(user_cfg, "build_train_dataloader"):
        # 如果没有 build_train_dataloader，手动构建 dataloader
        # 重用上面的 trainer，如果已经创建
        if 'trainer' not in locals():
            from wan_va.train import Trainer
            from configs import get_config
            config = get_config('robocasa_train')
            config.rank = 0
            config.local_rank = 0
            config.world_size = 1
            trainer = Trainer(config)
        dataloader = trainer.train_loader  # 假设 dataloader 是 train_loader
    else:
        dataloader = user_cfg.build_train_dataloader()

    return model, dataloader


def forward_sanity_check(model, batch, device):
    """
    真正跑一个小 forward。
    默认适配你贴过的调用方式：model(input_dict, train_mode=True)
    """
    model = model.to(device)
    model.eval()

    batch = move_batch_to_device(batch, device)

    result = {"ok": False}
    with torch.no_grad():
        start = time.time()
        out = model(batch, train_mode=True)
        elapsed = time.time() - start

    result["ok"] = True
    result["time_sec"] = elapsed

    if isinstance(out, tuple):
        result["num_outputs"] = len(out)
        result["outputs"] = []
        for i, item in enumerate(out):
            if isinstance(item, torch.Tensor):
                result["outputs"].append({
                    "idx": i,
                    "shape": list(item.shape),
                    "dtype": str(item.dtype),
                    "min": float(item.min().item()),
                    "max": float(item.max().item()),
                    "mean": float(item.float().mean().item()),
                    "std": float(item.float().std().item()),
                })
            else:
                result["outputs"].append({
                    "idx": i,
                    "type": str(type(item))
                })
    elif isinstance(out, torch.Tensor):
        result["num_outputs"] = 1
        result["outputs"] = [{
            "shape": list(out.shape),
            "dtype": str(out.dtype),
            "min": float(out.min().item()),
            "max": float(out.max().item()),
            "mean": float(out.float().mean().item()),
            "std": float(out.float().std().item()),
        }]
    else:
        result["output_type"] = str(type(out))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="你的配置文件路径，里面至少能构建 model 和 train dataloader")
    parser.add_argument("--save_dir", type=str, default="./debug_check")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--forward_batch_idx", type=int, default=0, help="用第几个 batch 做前向测试")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ensure_dir(args.save_dir)
    report = {
        "config": args.config,
        "max_batches": args.max_batches,
        "device": args.device,
        "batches": [],
        "problems": [],
    }

    print(f"[INFO] loading model and dataloader from: {args.config}")
    model, dataloader = maybe_get_dataloader_and_model_from_config(args.config)

    running_stats = {}
    first_forward_batch = None

    print(f"[INFO] start scanning first {args.max_batches} batches ...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break

        batch_report = {
            "batch_idx": batch_idx,
            "basic_info": collect_basic_info(batch),
            "prompt_info": try_decode_prompt(batch),
            "tensor_summaries": {},
            "problems": [],
        }

        recursive_collect_stats(
            batch,
            prefix="batch",
            stats_dict=running_stats,
            batch_summary=batch_report["tensor_summaries"],
            problems=batch_report["problems"],
        )

        inspect_action_space(batch, batch_report)
        inspect_text_length(batch, batch_report)

        # 抽一个 batch 用来 forward
        if batch_idx == args.forward_batch_idx:
            first_forward_batch = batch

        # 只把重要信息落盘
        log_line(os.path.join(args.save_dir, "batches.jsonl"), batch_report)
        report["batches"].append({
            "batch_idx": batch_idx,
            "num_problems": len(batch_report["problems"]),
            "text_check": batch_report.get("text_check", {}),
            "action_check_brief": {
                "exists": batch_report.get("action_check", {}).get("exists", False),
                "noisy_latents_shape": batch_report.get("action_check", {}).get("noisy_latents_shape", None),
            }
        })

        if batch_report["problems"]:
            report["problems"].extend([f"batch {batch_idx}: {x}" for x in batch_report["problems"]])

        if batch_idx % 10 == 0:
            print(f"[INFO] scanned batch {batch_idx}")

    # 汇总前 100 个 batch 的统计
    stats_summary = {}
    for k, v in running_stats.items():
        stats_summary[k] = v.summary()
    report["running_stats"] = stats_summary

    # 做一次前向测试
    if first_forward_batch is None:
        report["forward_check"] = {
            "ok": False,
            "error": "没有拿到用于前向测试的 batch"
        }
    else:
        print("[INFO] running one forward sanity check ...")
        try:
            forward_result = forward_sanity_check(model, first_forward_batch, args.device)
            report["forward_check"] = forward_result
            print("[INFO] forward sanity check passed")
        except Exception as e:
            report["forward_check"] = {
                "ok": False,
                "error_type": type(e).__name__,
                "error": str(e),
            }
            print("[ERROR] forward sanity check failed")
            print(type(e).__name__, str(e))

    # 给出一些自动判断
    diagnosis = []

    # 查 text_len
    text_lens = set()
    for brief in report["batches"]:
        tc = brief.get("text_check", {})
        if "text_len" in tc:
            text_lens.add(tc["text_len"])
    if len(text_lens) > 0:
        diagnosis.append(f"扫描到的 text_len 集合: {sorted(list(text_lens))}")
        if any(x != 512 for x in text_lens):
            diagnosis.append("存在 text_len != 512 的 batch；如果 flex cross-attn mask 写死 512，会报 block_mask mismatch。")

    # 看 action/noisy_latents
    key = "batch.action_dict.noisy_latents"
    if key in stats_summary:
        s = stats_summary[key]
        diagnosis.append(
            f"action_dict.noisy_latents 统计: mean={s['mean']:.6f}, std={s['std']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}"
        )

    key = "batch.latent_dict.noisy_latents"
    if key in stats_summary:
        s = stats_summary[key]
        diagnosis.append(
            f"latent_dict.noisy_latents 统计: mean={s['mean']:.6f}, std={s['std']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}"
        )

    report["diagnosis"] = diagnosis

    dump_json(report, os.path.join(args.save_dir, "summary.json"))

    print("\n========== SUMMARY ==========")
    print(json.dumps({
        "num_batches_scanned": len(report["batches"]),
        "num_problems": len(report["problems"]),
        "forward_check": report["forward_check"],
        "diagnosis": diagnosis,
        "save_dir": args.save_dir,
    }, ensure_ascii=False, indent=2))
    print("详细 batch 日志已保存到:", os.path.join(args.save_dir, "batches.jsonl"))
    print("总报告已保存到:", os.path.join(args.save_dir, "summary.json"))


if __name__ == "__main__":
    main()