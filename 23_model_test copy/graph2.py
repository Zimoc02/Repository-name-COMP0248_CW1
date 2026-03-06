import pandas as pd
import matplotlib.pyplot as plt

# ===== CSV路径 =====
maskpool = r"C:\Users\zimoc\Desktop\COMP0248_CW1\train_data\trainlog_maskpool_bndown4_gtcls1_bs10_lr0.001_wcls0.5_20260305_060640.csv"
baseline = r"C:\Users\zimoc\Desktop\COMP0248_CW1\train_data\trainlog_try_base_line_bs10_lr0.001_wcls0.5.csv"
lrgroup  = r"C:\Users\zimoc\Desktop\COMP0248_CW1\train_data\trainlog_bs10_lrgrp_0.5-1-2_wcls0.5.csv"

df_base = pd.read_csv(baseline)
df_mask = pd.read_csv(maskpool)
df_lr   = pd.read_csv(lrgroup)

# 如果某些CSV有重复epoch（你 baseline 之前有）
def dedup_epoch(df):
    if "epoch" in df.columns:
        return df.groupby("epoch", as_index=False).mean(numeric_only=True)
    return df

df_base = dedup_epoch(df_base)
df_mask = dedup_epoch(df_mask)
df_lr   = dedup_epoch(df_lr)

# ====== 只画 0~79 epoch ======
MAX_EPOCH = 79
def clamp_epoch(df, max_epoch=79, min_epoch=0):
    return df[df["epoch"].between(min_epoch, max_epoch)]

df_base = clamp_epoch(df_base, MAX_EPOCH, 0)
df_mask = clamp_epoch(df_mask, MAX_EPOCH, 0)
df_lr   = clamp_epoch(df_lr,   MAX_EPOCH, 0)

# =========================
# 1 Train Loss
# =========================
plt.figure()
plt.plot(df_base["epoch"], df_base["train_loss"], label="Baseline")
plt.plot(df_mask["epoch"], df_mask["train_loss"], label="MaskPool")
plt.plot(df_lr["epoch"],   df_lr["train_loss"],   label="LR-Groups (0.5-1-2)")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss (Epoch 0-79)")
plt.legend()
plt.grid()
plt.savefig("curve_train_loss.png", dpi=300)
plt.show()

# =========================
# 2 Dice
# =========================
plt.figure()
plt.plot(df_base["epoch"], df_base["val_dice"], label="Baseline Dice")
plt.plot(df_mask["epoch"], df_mask["val_dice"], label="MaskPool Dice")
plt.plot(df_lr["epoch"],   df_lr["val_dice"],   label="LR-Groups Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Validation Dice (Epoch 0-79)")
plt.legend()
plt.grid()
plt.savefig("curve_dice.png", dpi=300)
plt.show()

# =========================
# 3 Classification
# =========================
plt.figure()
plt.plot(df_base["epoch"], df_base["val_cls_acc"], label="Baseline")
plt.plot(df_mask["epoch"], df_mask["val_cls_acc"], label="MaskPool")
plt.plot(df_lr["epoch"],   df_lr["val_cls_acc"],   label="LR-Groups (0.5-1-2)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Classification Accuracy (Epoch 0-79)")
plt.legend()
plt.grid()
plt.savefig("curve_cls.png", dpi=300)
plt.show()

# =========================
# 4 Bounding box IoU
# =========================
plt.figure()
plt.plot(df_base["epoch"], df_base["val_bbox_iou"], label="Baseline")
plt.plot(df_mask["epoch"], df_mask["val_bbox_iou"], label="MaskPool")
plt.plot(df_lr["epoch"],   df_lr["val_bbox_iou"],   label="LR-Groups (0.5-1-2)")
plt.xlabel("Epoch")
plt.ylabel("BBox IoU")
plt.title("Bounding Box IoU (Epoch 0-79)")
plt.legend()
plt.grid()
plt.savefig("curve_bbox.png", dpi=300)
plt.show()