# 主训练脚本（MaskPool版 + tqdm显示 + CSV记录 + 安全命名保存 best/last/best_cls）
import random
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
from pathlib import Path
from tqdm.auto import tqdm
import csv
from datetime import datetime

from torch.utils.data import DataLoader

from src.dataloader import COMP0248KeyframeDataset, DataConfig
from src.model_maskpool import UNetMultiTask_MaskPool
from src.losses import MultiTaskLoss
from src.metrics import dice_from_logits, iou_from_logits, bbox_metrics_from_masks


CSV_PATH = r"C:\Users\zimoc\Desktop\COMP0248_CW1\Data_Processing\meta\index_keyframes.csv"

# ====== 实验配置（用于命名 + 记录） ======
BATCH_SIZE = 10
EPOCHS = 100
BASE_LR = 1e-3
W_SEG = 1.0
W_DICE = 1.0
W_CLS = 0.5

BOTTLENECK = "down4"
USE_GT_FOR_CLS_TRAIN = True

# 安全 run id：每次运行都不重名，避免覆盖
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

EXP_TAG = (
    f"maskpool_bn{BOTTLENECK}_gtcls{int(USE_GT_FOR_CLS_TRAIN)}_"
    f"bs{BATCH_SIZE}_lr{BASE_LR:g}_wcls{W_CLS:g}_{RUN_ID}"
)


def run_one_epoch(model, loader, loss_fn, optimizer, device, epoch: int, epochs: int):
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Train {epoch}/{epochs}", leave=False, dynamic_ncols=True)
    for x, mask, y in pbar:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        seg_logits, cls_logits = model(x, mask_gt=mask)
        loss, _parts = loss_fn(seg_logits, cls_logits, mask, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(total_loss/max(1,n)):.4f}")

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, epoch: int, epochs: int):
    model.eval()

    dice_sum = 0.0
    iou_sum = 0.0
    bbox_iou_sum = 0.0
    bbox_acc05_sum = 0.0

    correct = 0
    total = 0  # 样本数

    pbar = tqdm(loader, desc=f"Val   {epoch}/{epochs}", leave=False, dynamic_ncols=True)
    for x, mask, y in pbar:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        seg_logits, cls_logits = model(x, mask_gt=mask)

        dice = dice_from_logits(seg_logits, mask).item()
        iou = iou_from_logits(seg_logits, mask).item()
        b_iou, b_acc05 = bbox_metrics_from_masks(seg_logits, mask, thr=0.5)

        pred = cls_logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

        bs = x.size(0)
        dice_sum += dice * bs
        iou_sum += iou * bs
        bbox_iou_sum += b_iou * bs
        bbox_acc05_sum += b_acc05 * bs

        n = max(1, total)
        pbar.set_postfix(
            dice=f"{dice_sum/n:.4f}",
            iou=f"{iou_sum/n:.4f}",
            b_iou=f"{bbox_iou_sum/n:.4f}",
            cls=f"{correct/n:.4f}",
        )

    n = max(1, total)
    return {
        "val_dice": dice_sum / n,
        "val_iou": iou_sum / n,
        "val_bbox_iou": bbox_iou_sum / n,
        "val_bbox_acc05": bbox_acc05_sum / n,
        "val_cls_acc": correct / n,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # data
    train_cfg = DataConfig(csv_path=CSV_PATH, split="train", use_depth=True, keep_original_size=True)
    val_cfg = DataConfig(csv_path=CSV_PATH, split="val", use_depth=True, keep_original_size=True)

    train_ds = COMP0248KeyframeDataset(train_cfg)
    val_ds = COMP0248KeyframeDataset(val_cfg)

    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))
    print("total samples:", len(train_ds) + len(val_ds))


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # model / loss / optim
    model = UNetMultiTask_MaskPool(
        in_channels=4,
        num_classes=10,
        base_ch=32,
        bottleneck_layer_name=BOTTLENECK,
        use_gt_for_cls_when_training=USE_GT_FOR_CLS_TRAIN,
    ).to(device)

    loss_fn = MultiTaskLoss(w_seg=W_SEG, w_dice=W_DICE, w_cls=W_CLS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    print("LR groups:", [pg["lr"] for pg in optimizer.param_groups])

    # outputs (安全命名，不覆盖)
    weights_dir = Path("weights1")
    weights_dir.mkdir(exist_ok=True)
    best_path = weights_dir / f"best_dice_{EXP_TAG}.pt"
    last_path = weights_dir / f"last_{EXP_TAG}.pt"
    best_cls_path = weights_dir / f"best_cls_{EXP_TAG}.pt"

    train_data_dir = Path("train_data")
    train_data_dir.mkdir(exist_ok=True)
    log_path = train_data_dir / f"trainlog_{EXP_TAG}.csv"

    # CSV header
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch",
                "train_loss",
                "val_dice",
                "val_iou",
                "val_bbox_iou",
                "val_bbox_acc05",
                "val_cls_acc",
                "lr_enc", "lr_seg", "lr_cls",
            ])

    best_dice = -1.0
    best_cls = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, EPOCHS)
        metrics = evaluate(model, val_loader, device, epoch, EPOCHS)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_dice {metrics['val_dice']:.4f} | "
            f"val_iou {metrics['val_iou']:.4f} | "
            f"bbox_iou {metrics['val_bbox_iou']:.4f} | "
            f"acc@0.5 {metrics['val_bbox_acc05']:.4f} | "
            f"cls_acc {metrics['val_cls_acc']:.4f}"
        )

        # 单一 lr，为了兼容你之前 CSV 格式
        lr_enc = optimizer.param_groups[0]["lr"]
        lr_seg = optimizer.param_groups[0]["lr"]
        lr_cls = optimizer.param_groups[0]["lr"]

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                float(train_loss),
                float(metrics["val_dice"]),
                float(metrics["val_iou"]),
                float(metrics["val_bbox_iou"]),
                float(metrics["val_bbox_acc05"]),
                float(metrics["val_cls_acc"]),
                float(lr_enc),
                float(lr_seg),
                float(lr_cls),
            ])

        # save last
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
                "exp_tag": EXP_TAG,
            },
            last_path,
        )

        # save best by val Dice
        if metrics["val_dice"] > best_dice:
            best_dice = metrics["val_dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": metrics,
                    "exp_tag": EXP_TAG,
                },
                best_path,
            )
            print(f"  -> saved BEST (val_dice={best_dice:.4f}) to {best_path}")

        # save best by val CLS
        if metrics["val_cls_acc"] > best_cls:
            best_cls = metrics["val_cls_acc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": metrics,
                    "exp_tag": EXP_TAG,
                },
                best_cls_path,
            )
            print(f"  -> saved BEST CLS (val_cls={best_cls:.4f}) to {best_cls_path}")

    print("Done. Best val_dice:", best_dice, "| Best val_cls:", best_cls)
    print("Log saved to:", log_path)
  
if __name__ == "__main__":
    main()