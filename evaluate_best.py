# 随机抽一些样本做推理，把 rgb/gtmask/predmask/overlay 全部导出，方便你肉眼检查模型效果和错分。

import random
from pathlib import Path
import torch
import numpy as np
import cv2

from src.dataloader import COMP0248KeyframeDataset, DataConfig
from src.model import UNetMultiTask

CSV_PATH = r"C:\Users\zimoc\Desktop\COMP0248_CW1\Data_Processing\meta\index_keyframes.csv"
CKPT_PATH = r"weights\best_val_dice.pt"
OUT_DIR = Path(r"D:\0248_data_check\pred_vis")   # 你也可以改


def chw_to_rgb_u8(x3):
    """x3: torch [3,H,W] -> uint8 HWC RGB, robust normalize"""
    a = x3.detach().cpu().float().numpy()
    a = np.transpose(a, (1, 2, 0))
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = (a - lo) / max(1e-6, hi - lo)
    a = np.clip(a, 0, 1)
    return (a * 255).astype(np.uint8)


def mask_to_u8(m):
    """m: torch [1,H,W] or [H,W] -> uint8 HW (0/255)"""
    m = m.detach().cpu().float().numpy()
    if m.ndim == 3:
        m = m[0]
    return ((m > 0.5).astype(np.uint8) * 255)


def overlay(rgb_u8, mask_u8):
    """rgb_u8: HWC RGB, mask_u8: HW 0/255"""
    vis = rgb_u8.copy()
    # 用绿色叠加（不指定颜色也行，但cv2需要一个颜色…这里固定一个）
    color = np.zeros_like(vis)
    color[:, :, 1] = 255
    alpha = 0.35
    m = (mask_u8 > 0)[..., None]
    vis[m] = (vis[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)
    return vis


def main(n_samples=30, split="val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # dataset
    cfg = DataConfig(csv_path=CSV_PATH, split=split, use_depth=True, keep_original_size=True)
    ds = COMP0248KeyframeDataset(cfg)

    # model
    model = UNetMultiTask(in_channels=4, num_classes=10, base_ch=32).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(n_samples, len(ds))]

    for k, i in enumerate(idxs):
        x, gt_mask, y_gt = ds[i]
        x_b = x.unsqueeze(0).to(device)

        with torch.no_grad():
            seg_logits, cls_logits = model(x_b)
            pred_mask = (seg_logits.sigmoid() > 0.5).float().cpu()[0]
            y_pred = cls_logits.argmax(dim=1).item()

        rgb_u8 = chw_to_rgb_u8(x[:3])
        gt_u8 = mask_to_u8(gt_mask)
        pred_u8 = mask_to_u8(pred_mask)

        out_base = OUT_DIR / f"{k:03d}_idx{i:05d}_gt{int(y_gt)}_pred{y_pred}"
        cv2.imwrite(str(out_base) + "_rgb.png", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_base) + "_gtmask.png", gt_u8)
        cv2.imwrite(str(out_base) + "_predmask.png", pred_u8)

        ov = overlay(rgb_u8, pred_u8)
        cv2.imwrite(str(out_base) + "_overlay.png", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

        with open(str(out_base) + "_info.txt", "w", encoding="utf-8") as f:
            f.write(f"dataset_index: {i}\n")
            f.write(f"gt_class: {int(y_gt)}\n")
            f.write(f"pred_class: {y_pred}\n")

    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main(n_samples=50, split="val")