# 训练脚本的“成功配置版”（把 batch_size=10、epoch=100，用来稳定跑出更好结果）。
from tqdm.auto import tqdm
from datetime import datetime

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataloader import COMP0248KeyframeDataset, DataConfig
from src.model import UNetMultiTask
from src.losses import MultiTaskLoss
from src.metrics import dice_from_logits, iou_from_logits, bbox_metrics_from_masks


CSV_PATH = r"C:\Users\zimoc\Desktop\COMP0248_CW1\Data_Processing\meta\index_keyframes.csv"


def run_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        seg_logits, cls_logits = model(x)
        loss, _parts = loss_fn(seg_logits, cls_logits, mask, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    dice_sum = 0.0
    iou_sum = 0.0
    bbox_iou_sum = 0.0
    bbox_acc05_sum = 0.0

    correct = 0
    total = 0

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        seg_logits, cls_logits = model(x)

        # seg metrics
        dice = dice_from_logits(seg_logits, mask).item()
        iou = iou_from_logits(seg_logits, mask).item()

        # bbox metrics from masks
        b_iou, b_acc05 = bbox_metrics_from_masks(seg_logits, mask, thr=0.5)

        # cls metrics
        pred = cls_logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

        bs = x.size(0)
        dice_sum += dice * bs
        iou_sum += iou * bs
        bbox_iou_sum += b_iou * bs
        bbox_acc05_sum += b_acc05 * bs

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

    train_loader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=0, pin_memory=True)

    # model / loss / optim
    model = UNetMultiTask(in_channels=4, num_classes=10, base_ch=32).to(device)
    loss_fn = MultiTaskLoss(w_seg=1.0, w_dice=1.0, w_cls=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # outputs
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    best_path = weights_dir / "best_val_dice.pt"
    last_path = weights_dir / "last.pt"

    best_dice = -1.0
    epochs = 100  # 先跑 10 个 epoch 试水；后面你再加

    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer, device)
        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_dice {metrics['val_dice']:.4f} | "
            f"val_iou {metrics['val_iou']:.4f} | "
            f"bbox_iou {metrics['val_bbox_iou']:.4f} | "
            f"acc@0.5 {metrics['val_bbox_acc05']:.4f} | "
            f"cls_acc {metrics['val_cls_acc']:.4f}"
        )

        # save last
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
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
                },
                best_path,
            )
            print(f"  -> saved BEST (val_dice={best_dice:.4f}) to {best_path}")

    print("Done. Best val_dice:", best_dice)


if __name__ == "__main__":
    main()