# Test inference for MaskPool model
# Outputs:
# - test Dice / IoU / BBox IoU / Detection Acc@0.5 / Classification Acc
# - confusion_matrix.csv / confusion_matrix.png
# - preds_all.csv
# - rgb / predmask / overlay visualizations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# =========================================================
# Find project root FIRST, then import src
# =========================================================
THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = None
for p in [THIS_FILE.parent] + list(THIS_FILE.parents):
    if (p / "src").is_dir():
        PROJECT_ROOT = p
        break

if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot find project root containing 'src' from: {THIS_FILE}")

sys.path.insert(0, str(PROJECT_ROOT))
print("[debug] PROJECT_ROOT =", PROJECT_ROOT)

from src.dataloader import COMP0248KeyframeDataset, DataConfig
from src.model_maskpool import UNetMultiTask_MaskPool
from src.metrics import dice_from_logits, iou_from_logits, bbox_metrics_from_masks


# =========================================================
# Sampling / saving switches
# =========================================================
FRACTION = 1.0   # 1.0 = full test set
SEED = 42

# =========================================================
# Paths
# =========================================================
CKPT_PATH = Path(
    r"C:\Users\zimoc\Desktop\COMP0248_CW1\weights\best_cls_maskpool_bndown4_gtcls1_bs10_lr0.001_wcls0.5_20260305_060640.pt"
)

TEST_CSV = Path(
    r"D:\0248_data_check\test_build\COMP0248_Test_data_23\test_index_23_depthpng_for_loader.csv"
)

OUT_DIR = Path(
    r"D:\0248_data_check\pred_vis_test\COMP0248_Test_data_23\maskpool"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Utils
# =========================================================
def chw_to_rgb_u8(x3: torch.Tensor) -> np.ndarray:
    """x3: [3,H,W] torch -> HWC uint8 RGB"""
    a = x3.detach().cpu().float().numpy()
    a = np.transpose(a, (1, 2, 0))
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = (a - lo) / max(1e-6, hi - lo)
    a = np.clip(a, 0, 1)
    return (a * 255).astype(np.uint8)


def mask_to_u8(m: torch.Tensor) -> np.ndarray:
    """m: [1,H,W] or [H,W] torch -> HW uint8 (0/255)"""
    m = m.detach().cpu().float().numpy()
    if m.ndim == 3:
        m = m[0]
    return ((m > 0.5).astype(np.uint8) * 255)


def overlay(rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Overlay predicted mask on RGB image"""
    vis = rgb_u8.copy()
    alpha = 0.35
    m = (mask_u8 > 0)

    green = np.zeros_like(vis)
    green[:, :, 1] = 255

    vis[m] = (vis[m] * (1 - alpha) + green[m] * alpha).astype(np.uint8)
    return vis


# =========================================================
# CSV sanity
# =========================================================
def _pick_col(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
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
            "Please ensure test CSV contains rgb/depth/mask/y/split columns."
        )

    print("[CSV mapped columns]")
    print(" rgb:", rgb_col)
    print(" depth:", depth_col)
    print(" mask:", mask_col)
    print(" y:", y_col)
    print(" split:", split_col)

    print("[CSV rows]:", len(df))
    print("[split counts]:")
    print(df[split_col].value_counts())

    return True


# =========================================================
# Confusion matrix plot
# =========================================================
def save_confusion_matrix_png(cm: np.ndarray, out_path: Path, class_names=None, title="Confusion Matrix"):
    C = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(C)]

    fig_w = max(6, 0.6 * C)
    fig_h = max(5, 0.55 * C)

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(C)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    for i in range(C):
        for j in range(C):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[cm] Saved PNG:", out_path)


# =========================================================
# Main inference
# =========================================================
def run_infer(split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    sanity_check_csv(TEST_CSV)

    cfg = DataConfig(
        csv_path=str(TEST_CSV),
        split=split,
        use_depth=True,
        keep_original_size=True,
    )
    ds = COMP0248KeyframeDataset(cfg)
    print("len(ds):", len(ds))

    # Model
    model = UNetMultiTask_MaskPool(
        in_channels=4,
        num_classes=10,
        base_ch=32,
        bottleneck_layer_name="down4",
        use_gt_for_cls_when_training=True,
    ).to(device)

    ckpt = torch.load(str(CKPT_PATH), map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        print("[ckpt] format: dict with model_state")
    else:
        state_dict = ckpt
        print("[ckpt] format: raw state_dict")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Loaded ckpt:", CKPT_PATH)

    idxs = list(range(len(ds)))
    rng = np.random.default_rng(SEED)
    n_use = max(1, int(len(idxs) * FRACTION))
    idxs = rng.choice(idxs, size=n_use, replace=False).tolist()
    print(f"Using {n_use}/{len(ds)} samples ({FRACTION*100:.1f}%)")

    all_pred_rows = []

    # accumulators for test metrics
    dice_sum = 0.0
    iou_sum = 0.0
    bbox_iou_sum = 0.0
    bbox_acc05_sum = 0.0
    correct = 0
    total = 0

    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows=gt, cols=pred

    pbar = tqdm(enumerate(idxs), total=len(idxs), desc=f"Infer [{split}]", dynamic_ncols=True)

    for k, i in pbar:
        x, gt_mask, y_gt = ds[i]
        x_b = x.unsqueeze(0).to(device)
        gt_mask_b = gt_mask.unsqueeze(0).to(device)

        with torch.no_grad():
            # some MaskPool models accept mask_gt in forward, some don't
            try:
                outputs = model(x_b, mask_gt=gt_mask_b)
            except TypeError:
                outputs = model(x_b)

            if isinstance(outputs, (tuple, list)):
                seg_logits, cls_logits = outputs[0], outputs[1]
            elif isinstance(outputs, dict):
                seg_logits, cls_logits = outputs["seg"], outputs["cls"]
            else:
                raise RuntimeError(f"Unexpected model output type: {type(outputs)}")

            pred_mask = (seg_logits.sigmoid() > 0.5).float().cpu()[0]
            y_pred = cls_logits.argmax(dim=1).item()

            # compute test metrics
            dice = dice_from_logits(seg_logits, gt_mask_b).item()
            iou = iou_from_logits(seg_logits, gt_mask_b).item()
            b_iou, b_acc05 = bbox_metrics_from_masks(seg_logits, gt_mask_b, thr=0.5)

        # accumulate metrics
        dice_sum += dice
        iou_sum += iou
        bbox_iou_sum += b_iou
        bbox_acc05_sum += b_acc05
        correct += int(y_pred == int(y_gt))
        total += 1

        # confusion matrix
        cm[int(y_gt), int(y_pred)] += 1

        # save visualization
        rgb_u8 = chw_to_rgb_u8(x[:3])
        pred_u8 = mask_to_u8(pred_mask)

        out_base = OUT_DIR / f"{k:04d}_idx{i:05d}_gt{int(y_gt)}_pred{y_pred}"
        cv2.imwrite(str(out_base) + "_rgb.png", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_base) + "_predmask.png", pred_u8)
        ov = overlay(rgb_u8, pred_u8)
        cv2.imwrite(str(out_base) + "_overlay.png", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

        all_pred_rows.append({
            "dataset_index": i,
            "gt": int(y_gt),
            "pred": int(y_pred),
            "dice": float(dice),
            "iou": float(iou),
            "bbox_iou": float(b_iou),
            "bbox_acc05": float(b_acc05),
        })

        if (k + 1) % 20 == 0 or (k + 1) == len(idxs):
            n = max(1, total)
            pbar.set_postfix(
                cls=f"{correct/n:.3f}",
                dice=f"{dice_sum/n:.3f}",
                iou=f"{iou_sum/n:.3f}",
                b_iou=f"{bbox_iou_sum/n:.3f}",
            )

    # final test metrics
    n = max(1, total)
    test_dice = dice_sum / n
    test_iou = iou_sum / n
    test_bbox_iou = bbox_iou_sum / n
    test_bbox_acc05 = bbox_acc05_sum / n
    test_cls_acc = correct / n

    print("\n===== Test Metrics =====")
    print(f"Dice: {test_dice:.4f}")
    print(f"IoU: {test_iou:.4f}")
    print(f"BBox IoU: {test_bbox_iou:.4f}")
    print(f"Detection Acc@0.5: {test_bbox_acc05:.4f}")
    print(f"Classification Acc: {test_cls_acc:.4f} ({correct}/{total})")

    print("\nConfusion matrix (rows=gt, cols=pred):")
    print(cm)

    # save summary metrics csv
    summary_csv = OUT_DIR / "test_metrics_summary.csv"
    pd.DataFrame([{
        "dice": test_dice,
        "iou": test_iou,
        "bbox_iou": test_bbox_iou,
        "bbox_acc05": test_bbox_acc05,
        "cls_acc": test_cls_acc,
        "num_samples": total,
    }]).to_csv(summary_csv, index=False)
    print("Saved test metrics summary:", summary_csv)

    # save confusion matrix
    cm_csv_path = OUT_DIR / "confusion_matrix.csv"
    pd.DataFrame(cm).to_csv(cm_csv_path, index=False)
    print("Saved confusion matrix csv:", cm_csv_path)

    cm_png_path = OUT_DIR / "confusion_matrix.png"
    save_confusion_matrix_png(cm, cm_png_path, class_names=[f"C{i}" for i in range(num_classes)])

    # save per-sample predictions
    pred_csv = OUT_DIR / "preds_all.csv"
    pd.DataFrame(all_pred_rows).to_csv(pred_csv, index=False)
    print("Saved preds csv:", pred_csv)

    print("Saved visualizations to:", OUT_DIR)


if __name__ == "__main__":
    run_infer(split="test")