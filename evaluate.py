# evaluate.py：在验证集上完整评估 best checkpoint，并额外保存一些可视化结果到 results/vis。


from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataloader import COMP0248KeyframeDataset, DataConfig
from src.model import UNetMultiTask
from src.metrics import dice_from_logits, iou_from_logits, bbox_metrics_from_masks, bbox_from_mask


CSV_PATH = r"C:\Users\zimoc\Desktop\COMP0248_CW1\Data_Processing\meta\index_keyframes.csv"
BEST_CKPT = Path("weights/best_val_dice.pt")


def overlay_mask_on_rgb(rgb: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.5):
    """
    rgb: HxWx3 uint8
    mask_bin: HxW uint8 {0,1}
    """
    overlay = rgb.copy()
    color = np.zeros_like(rgb)
    color[..., 1] = 255  # green
    overlay[mask_bin > 0] = (alpha * color[mask_bin > 0] + (1 - alpha) * overlay[mask_bin > 0]).astype(np.uint8)
    return overlay


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # data
    val_cfg = DataConfig(csv_path=CSV_PATH, split="val", use_depth=True, keep_original_size=True)
    val_ds = COMP0248KeyframeDataset(val_cfg)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)

    # model
    model = UNetMultiTask(in_channels=4, num_classes=10, base_ch=32).to(device)

    ckpt = torch.load(BEST_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("loaded:", BEST_CKPT, "epoch:", ckpt.get("epoch", "?"), "metrics:", ckpt.get("metrics", {}))

    dice_sum = 0.0
    iou_sum = 0.0
    bbox_iou_sum = 0.0
    bbox_acc05_sum = 0.0
    correct = 0
    total = 0

    # save a few visualizations
    out_dir = Path("results/vis")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    max_save = 10

    for x, mask, y in val_loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            seg_logits, cls_logits = model(x)

        # metrics
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

        # visualization (use RGB from input tensor)
        if saved < max_save:
            # x is normalized; recover approx RGB for visualization:
            # take first 3 channels, denormalize ImageNet (matches dataloader default)
            rgb = x[:, :3].detach().cpu().numpy()  # [B,3,H,W]
            rgb = np.transpose(rgb, (0, 2, 3, 1))  # [B,H,W,3]

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            rgb = (rgb * std + mean)
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

            prob = torch.sigmoid(seg_logits).detach().cpu().numpy()  # [B,1,H,W]
            pred_mask = (prob[:, 0] > 0.5).astype(np.uint8)
            gt_mask = (mask.detach().cpu().numpy()[:, 0] > 0.5).astype(np.uint8)

            for i in range(bs):
                if saved >= max_save:
                    break

                vis = overlay_mask_on_rgb(rgb[i], pred_mask[i], alpha=0.5)

                # draw bbox from predicted mask + gt bbox
                pm = torch.from_numpy(pred_mask[i])
                gm = torch.from_numpy(gt_mask[i])
                box_p = bbox_from_mask(pm)
                box_g = bbox_from_mask(gm)

                if box_g is not None:
                    x1, y1, x2, y2 = box_g
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)  # yellow GT
                if box_p is not None:
                    x1, y1, x2, y2 = box_p
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)  # magenta Pred

                # put text
                cv2.putText(vis, f"gt={int(y[i])} pred={int(pred[i])}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                out_path = out_dir / f"val_{saved:03d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                saved += 1

    n = max(1, total)
    print("VAL results:")
    print("  dice:", dice_sum / n)
    print("  iou:", iou_sum / n)
    print("  bbox_iou:", bbox_iou_sum / n)
    print("  acc@0.5:", bbox_acc05_sum / n)
    print("  cls_acc:", correct / n)
    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()