import pandas as pd
import matplotlib.pyplot as plt

# ===== CSV路径 =====

maskpool = r"C:\Users\zimoc\Desktop\COMP0248_CW1\train_data\trainlog_maskpool_bndown4_gtcls1_bs10_lr0.001_wcls0.5_20260305_060640.csv"
baseline = r"C:\Users\zimoc\Desktop\COMP0248_CW1\train_data\trainlog_try_base_line_bs10_lr0.001_wcls0.5.csv"

df_base = pd.read_csv(baseline)
df_mask = pd.read_csv(maskpool)

# 如果 baseline 有重复epoch（你这个有）
df_base = df_base.groupby("epoch").mean().reset_index()

# =========================
# 1 Train Loss
# =========================
plt.figure()

plt.plot(df_base["epoch"], df_base["train_loss"], label="Baseline")
plt.plot(df_mask["epoch"], df_mask["train_loss"], label="MaskPool")

plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss")
plt.legend()
plt.grid()

plt.savefig("curve_train_loss.png", dpi=300)
plt.show()


# =========================
# 2 Dice / IoU
# =========================
plt.figure()

plt.plot(df_base["epoch"], df_base["val_dice"], label="Baseline Dice")
plt.plot(df_mask["epoch"], df_mask["val_dice"], label="MaskPool Dice")

plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Validation Dice")
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

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Classification Accuracy")
plt.legend()
plt.grid()

plt.savefig("curve_cls.png", dpi=300)
plt.show()


# =========================
# 4 Bounding box
# =========================
plt.figure()

plt.plot(df_base["epoch"], df_base["val_bbox_iou"], label="Baseline")
plt.plot(df_mask["epoch"], df_mask["val_bbox_iou"], label="MaskPool")

plt.xlabel("Epoch")
plt.ylabel("BBox IoU")
plt.title("Bounding Box IoU")
plt.legend()
plt.grid()

plt.savefig("curve_bbox.png", dpi=300)
plt.show()