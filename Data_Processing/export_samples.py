# 把数据集样本导出成 png 图（rgb / depth / mask / label），方便人工可视化检查。


import os
from pathlib import Path

import torch
import numpy as np
import cv2

from src.dataloader import COMP0248KeyframeDataset, DataConfig

CSV_PATH = r"C:\Users\zimoc\Desktop\COMP0248_CW1\Data_Processing\meta\index_keyframes.csv"


def to_uint8_img(chw):
    """chw: torch tensor [C,H,W], float/uint8"""
    x = chw.detach().cpu().float().numpy()
    x = np.transpose(x, (1, 2, 0))  # HWC

    # robust min-max to [0,255]
    x_min = np.percentile(x, 1)
    x_max = np.percentile(x, 99)
    x = (x - x_min) / max(1e-6, (x_max - x_min))
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)



def save_samples(split="train", out_dir="debug_samples_train", n_samples=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(csv_path=CSV_PATH, split=split, use_depth=True, keep_original_size=True)
    ds = COMP0248KeyframeDataset(cfg)

    n = len(ds) if (n_samples is None) else min(n_samples, len(ds))
    print(f"Exporting {n} samples from split='{split}' to: {out_dir}")

    for i in range(n):
        x, mask, y = ds[i]

        # x: [4,H,W] -> rgb:[3,H,W], depth:[1,H,W]
        rgb = x[:3]
        depth = x[3:4]

        rgb_u8 = to_uint8_img(rgb)              # HWC 3-ch
        depth_u8 = to_uint8_img(depth)[:, :, 0] # HW single-ch

        m = mask.detach().cpu().numpy()
        if m.ndim == 3:  # [1,H,W]
            m = m[0]
        # mask to 0/255
        m_u8 = (m > 0.5).astype(np.uint8) * 255

        # save
        cv2.imwrite(str(out_dir / f"{i:05d}_rgb.png"), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{i:05d}_depth.png"), depth_u8)
        cv2.imwrite(str(out_dir / f"{i:05d}_mask.png"), m_u8)

        with open(out_dir / f"{i:05d}_label.txt", "w", encoding="utf-8") as f:
            f.write(str(int(y)))

    print("Done.")



if __name__ == "__main__":
    BASE_OUT = Path(r"D:\0248_data_check")
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    save_samples(split="train", out_dir=BASE_OUT / "debug_samples_train", n_samples=None)
    save_samples(split="val", out_dir=BASE_OUT / "debug_samples_val", n_samples=None)