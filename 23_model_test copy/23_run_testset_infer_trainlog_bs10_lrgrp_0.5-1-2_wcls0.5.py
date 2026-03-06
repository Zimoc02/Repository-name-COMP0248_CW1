# 加载训练好的 UNetMultiTask，在 Test 集上做完整推理 + 可视化 + 生成预测CSV + 计算混淆矩阵。
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # 指向 .../COMP0248_CW1
sys.path.insert(0, str(PROJECT_ROOT))
# -------- sampling / saving switches --------
FRACTION = 0.01   # 只跑 10% 的 test
SEED = 42         # 固定随机种子，方便复现

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2

from src.dataloader import COMP0248KeyframeDataset, DataConfig
from src.model import UNetMultiTask


# -----------------------------
# Paths (edit if needed)
# -----------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../COMP0248_CW1


CKPT_PATH = PROJECT_ROOT / "weights" / "best_cls_bs10_lrgrp_0.5-1-2_wcls0.5.pt"

TEST_CSV = Path(r"D:\0248_data_check\test_build\COMP0248_Test_data_23\test_index_23_depthpng_for_loader.csv")
OUT_DIR = Path(r"D:\0248_data_check\pred_vis_test\COMP0248_Test_data_23")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LOG_CSV = PROJECT_ROOT / "train_data" / "trainlog_bs10_lrgrp_0.5-1-2_wcls0.5.csv"

# -----------------------------
# Utils (same as yours)
# -----------------------------
def chw_to_rgb_u8(x3: torch.Tensor) -> np.ndarray:
    """x3: torch [3,H,W] -> uint8 HWC RGB, robust normalize"""
    a = x3.detach().cpu().float().numpy()
    a = np.transpose(a, (1, 2, 0))
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = (a - lo) / max(1e-6, hi - lo)
    a = np.clip(a, 0, 1)
    return (a * 255).astype(np.uint8)


def mask_to_u8(m: torch.Tensor) -> np.ndarray:
    """m: torch [1,H,W] or [H,W] -> uint8 HW (0/255)"""
    m = m.detach().cpu().float().numpy()
    if m.ndim == 3:
        m = m[0]
    return ((m > 0.5).astype(np.uint8) * 255)


def overlay(rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """rgb_u8: HWC RGB, mask_u8: HW 0/255"""
    vis = rgb_u8.copy()
    alpha = 0.35

    # HW boolean mask
    m = (mask_u8 > 0)

    # green color overlay
    green = np.zeros_like(vis)
    green[:, :, 1] = 255

    # 用 HW mask 去选“像素”，而不是 HW1 去索引 HWC
    vis[m] = (vis[m] * (1 - alpha) + green[m] * alpha).astype(np.uint8)
    return vis


# -----------------------------
# CSV sanity: show & try map common columns
# -----------------------------
def _pick_col(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fuzzy contains
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None


def sanity_check_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    
    cols = df.columns.tolist()

    rgb_col = _pick_col(cols, ["rgb_path", "rgb", "image_path", "img_path"])
    depth_col = _pick_col(cols, ["depth_path", "depth_raw", "depth", "depth_png"])
    mask_col = _pick_col(cols, ["mask_path", "mask", "annotation", "seg_path", "gt_mask"])
    y_col = _pick_col(cols, ["y", "label", "class_id", "cls"])
    split_col = _pick_col(cols, ["split", "set"])

    if any(x is None for x in [rgb_col, depth_col, mask_col, y_col, split_col]):
        print("\n[CSV columns found]")
        print(cols)
        raise RuntimeError(
            "Cannot auto-map CSV columns. "
            "Please make sure your test CSV contains columns for rgb/depth/mask/y/split "
            "(e.g., rgb_path, depth_path, mask_path, y, split)."
        )

    print("[CSV mapped columns]")
    print(" rgb:", rgb_col)
    print(" depth:", depth_col)
    print(" mask:", mask_col)
    print(" y:", y_col)
    print(" split:", split_col)

    # Quick stats
    print("[CSV rows]:", len(df))
    print("[split counts]:")
    print(df[split_col].value_counts())

    return True


def plot_train_curves(
    train_csv: Path,
    out_dir: Path,
    prefix: str = "train",
):
    """
    读取训练日志 CSV，生成三张图：
      - {prefix}_loss.png
      - {prefix}_val_dice.png
      - {prefix}_val_cls_acc.png
    """
    if not train_csv.exists():
        print(f"[plot] Skip: train log not found: {train_csv}")
        return

    df = pd.read_csv(train_csv)

    # 1) epoch 转数字，丢掉坏行
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)

    # 2) 按“epoch 变小”检测 run 切换：run_id = 0,1,2,...
    df["run_id"] = (df["epoch"].diff().fillna(0) < 0).cumsum()

    # 3) 只保留最后一次 run（run_id 最大的那段）
    last_run = df["run_id"].max()
    df = df[df["run_id"] == last_run].copy()

    # 4) 如果同一 epoch 有重复（多次写入），只保留最后一条（可选但很建议）
    df = df.sort_values(["epoch"]).drop_duplicates(subset=["epoch"], keep="last")

    x = df["epoch"].to_numpy()
    if len(df) == 0:
        print(f"[plot] Skip: empty train log: {train_csv}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plt_out_dir = out_dir / "plot_trainlog_bs10_lrgrp_0.5-1-2_wcls0.5"
    plt_out_dir.mkdir(parents=True, exist_ok=True)

   

    # 1) loss
    plt.figure()
    plt.plot(x, df["train_loss"].to_numpy())
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.tight_layout()
    p = plt_out_dir / f"{prefix}_loss.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print("[plot] Saved:", p)

    # 2) val_dice
    plt.figure()
    plt.plot(x, df["val_dice"].to_numpy())
    plt.xlabel("epoch")
    plt.ylabel("val_dice")
    plt.tight_layout()
    p = plt_out_dir / f"{prefix}_val_dice.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print("[plot] Saved:", p)

    # 3) val_cls_acc
    plt.figure()
    plt.plot(x, df["val_cls_acc"].to_numpy())
    plt.xlabel("epoch")
    plt.ylabel("val_cls_acc")
    plt.tight_layout()
    p = plt_out_dir / f"{prefix}_val_cls_acc.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print("[plot] Saved:", p)

# -----------------------------
# Main inference
# -----------------------------
def run_infer(n_samples=50, split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    sanity_check_csv(TEST_CSV)

    # Dataset
    cfg = DataConfig(
        csv_path=str(TEST_CSV),
        split=split,
        use_depth=True,
        keep_original_size=True,
    )
    ds = COMP0248KeyframeDataset(cfg)
    print("len(ds):", len(ds))

    # Model
    model = UNetMultiTask(in_channels=4, num_classes=10, base_ch=32).to(device)
    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Loaded ckpt:", CKPT_PATH)

    
    # choose sample indices
    # idxs = list(range(len(ds)))
    # random.shuffle(idxs)
    # idxs = idxs[: min(n_samples, len(ds))]

    #idxs = list(range(len(ds)))   # 全量 1950，不抽样
    idxs = list(range(len(ds)))
    rng = np.random.default_rng(SEED)
    n_use = max(1, int(len(idxs) * FRACTION))
    idxs = rng.choice(idxs, size=n_use, replace=False).tolist()
    print(f"Using {n_use}/{len(ds)} samples ({FRACTION*100:.1f}%)")
    
    # also dump a prediction csv for all samples? (optional quick)
    all_pred_rows = []

    for k, i in enumerate(idxs):
        x, gt_mask, y_gt = ds[i]
        x_b = x.unsqueeze(0).to(device)

        with torch.no_grad():
            seg_logits, cls_logits = model(x_b)
            pred_mask = (seg_logits.sigmoid() > 0.5).float().cpu()[0]
            y_pred = cls_logits.argmax(dim=1).item()

        # save vis
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
            f.write(f"csv: {TEST_CSV}\n")

        all_pred_rows.append({"dataset_index": i, "gt": int(y_gt), "pred": int(y_pred)})
    # ===== metrics: accuracy + confusion matrix =====
    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows=gt, cols=pred
    correct = 0

    for r in all_pred_rows:
        gt = int(r["gt"])
        pred = int(r["pred"])
        cm[gt, pred] += 1
        correct += (gt == pred)

    acc = correct / max(1, len(all_pred_rows))
    print(f"Test accuracy: {acc:.4f} ({correct}/{len(all_pred_rows)})")
    print("Confusion matrix (rows=gt, cols=pred):")
    print(cm)

    # also save confusion matrix as csv
    cm_path = OUT_DIR / "confusion_matrix.csv"
    
    pd.DataFrame(cm).to_csv(cm_path, index=False)
    print("Saved confusion matrix csv:", cm_path)

    # save sample preds csv
    # pred_csv = OUT_DIR / "sample_preds.csv"
    pred_csv = OUT_DIR / "preds_all.csv"
    pd.DataFrame(all_pred_rows).to_csv(pred_csv, index=False)
    print("Saved visualizations to:", OUT_DIR)
    print("Saved sample preds csv:", pred_csv)


if __name__ == "__main__":
    plot_train_curves(TRAIN_LOG_CSV, OUT_DIR, prefix="model_1")
    run_infer(n_samples=50, split="test")

