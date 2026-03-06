from pathlib import Path
import pandas as pd
import shutil

# 你的文件路径
PRED_CSV = Path(r"D:\0248_data_check\pred_vis_test\preds_all.csv")
TEST_CSV = Path(r"D:\0248_data_check\test_build\test_index_for_loader_depthpng.csv")
OUT_DIR  = Path(r"D:\0248_data_check\pred_vis_test\misclassified")

# 可选：把类别 id 映射成名字（按你根目录顺序）
ID2NAME = {
    0: "G01_call",
    1: "G02_dislike",
    2: "G03_like",
    3: "G04_ok",
    4: "G05_one",
    5: "G06_palm",
    6: "G07_peace",
    7: "G08_rock",
    8: "G09_stop",
    9: "G10_three",
}

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(PRED_CSV)  # columns: dataset_index, gt, pred
    test_df = pd.read_csv(TEST_CSV)  # columns: rgb_path, depth_path, mask_path, class_id, ...

    # 用 dataset_index 对齐到 test_df 的行号
    # 注意：我们之前的 dataset_index 就是 ds[i] 的 i，对应 test_df.iloc[i]
    wrong = pred_df[pred_df["gt"] != pred_df["pred"]].copy()
    print("total wrong:", len(wrong), " / ", len(pred_df))

    # 每个错样本复制 rgb/depth/mask（你也可以只要 rgb）
    for _, r in wrong.iterrows():
        i = int(r["dataset_index"])
        gt = int(r["gt"])
        pr = int(r["pred"])

        row = test_df.iloc[i]
        rgb  = Path(row["rgb_path"])
        depth = Path(row["depth_path"])
        mask = Path(row["mask_path"])

        gt_name = ID2NAME.get(gt, str(gt))
        pr_name = ID2NAME.get(pr, str(pr))

        # 分目录：gt->pred
        sub = OUT_DIR / f"gt_{gt:02d}_{gt_name}__pred_{pr:02d}_{pr_name}"
        sub.mkdir(parents=True, exist_ok=True)

        # 统一命名，保留原 frame 信息
        base = f"idx{i:05d}"

        # 复制文件
        if rgb.exists():   safe_copy(rgb,   sub / f"{base}_rgb{rgb.suffix}")
        if depth.exists(): safe_copy(depth, sub / f"{base}_depth{depth.suffix}")
        if mask.exists():  safe_copy(mask,  sub / f"{base}_mask{mask.suffix}")

    print("saved to:", OUT_DIR)

if __name__ == "__main__":
    main()