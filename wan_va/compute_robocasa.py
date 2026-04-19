#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import get_config
from dataset import MultiLatentLeRobotDataset


def main():
    config = get_config("robocasa_train")
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1

    # 统计时别开太多并行，先稳一点
    config.load_worker = 0
    config.dataset_init_workers = 4

    dset = MultiLatentLeRobotDataset(config)

    values_per_channel = [[] for _ in range(30)]

    # MultiLatentLeRobotDataset 已改为按帧加权的虚拟样本长度（_sample_prefix），
    # 不再有 _acc_prefix。统计 norm 应对每个 repo 的每个 clip 各扫一次（与训练样本一一对应），
    # 不要按 len(dset) 虚拟索引重复统计同一条 meta。
    total_clips = sum(len(sub) for sub in dset._datasets)
    pbar = tqdm(total=total_clips, desc="compute 30d stats")
    for sub_dset in dset._datasets:
        for local_idx in range(len(sub_dset)):
            item = sub_dset.get_stats_item(local_idx)
            pbar.update(1)

            action_30 = item["action_30"]  # [N, 30]
            mask_30 = item["action_mask_30"]  # [N, 30]

            for c in range(30):
                valid = mask_30[:, c].astype(bool)
                vals = action_30[valid, c]
                if vals.size > 0:
                    values_per_channel[c].append(vals.astype(np.float64))

    pbar.close()

    mean = []
    std = []
    q01 = []
    q99 = []
    count = []

    for c in range(30):
        if len(values_per_channel[c]) == 0:
            vals = np.array([], dtype=np.float64)
        else:
            vals = np.concatenate(values_per_channel[c], axis=0)

        if vals.size == 0:
            mean.append(0.0)
            std.append(1.0)
            q01.append(0.0)
            q99.append(1.0)
            count.append(0)
        else:
            mean.append(float(vals.mean()))
            std.append(float(vals.std() + 1e-6))
            q01.append(float(np.quantile(vals, 0.01)))
            q99.append(float(np.quantile(vals, 0.99)))
            count.append(int(vals.size))

        print(
            f"channel {c:02d}: count={count[-1]}, "
            f"mean={mean[-1]:.6f}, std={std[-1]:.6f}, "
            f"q01={q01[-1]:.6f}, q99={q99[-1]:.6f}"
        )

    out = {
        "mean": mean,
        "std": std,
        "q01": q01,
        "q99": q99,
        "count": count,
    }

    save_path = "/root/lingbot_va/lingbot-va/wan_va/robocasa_30d_norm_stats_new.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()