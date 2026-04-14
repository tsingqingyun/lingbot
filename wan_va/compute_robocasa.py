#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import bisect
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

    for global_idx in tqdm(range(len(dset)), desc="compute 30d stats"):
        dset_id = bisect.bisect_right(dset._acc_prefix, global_idx) - 1
        local_idx = global_idx - dset._acc_prefix[dset_id]

        sub_dset = dset._datasets[dset_id]
        item = sub_dset.get_stats_item(local_idx)

        action_30 = item["action_30"]          # [N, 30]
        mask_30 = item["action_mask_30"]       # [N, 30]

        for c in range(30):
            valid = mask_30[:, c].astype(bool)
            vals = action_30[valid, c]
            if vals.size > 0:
                values_per_channel[c].append(vals.astype(np.float64))

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

    save_path = "robocasa_30d_norm_stats.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()