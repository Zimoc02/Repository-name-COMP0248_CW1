# 根据 preds_all.csv，把所有分类错误样本单独拷贝出来，并生成更直观的统计报表
from pathlib import Path
import pandas as pd
import shutil

# ========= 你的文件路径 =========
PRED_CSV = Path(r"D:\0248_data_check\pred_vis_test\COMP0248_Test_data_23\preds_all.csv")
TEST_CSV = Path(r"D:\0248_data_check\test_build\COMP0248_Test_data_23\test_index_23_depthpng.csv")
OUT_DIR  = Path(r"D:\0248_data_check\pred_vis_test\COMP0248_Test_data_23\misclassified")

# ========= 可选：类别名 =========
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

COPY_RGB = True
COPY_DEPTH = True
COPY_MASK = True

TOPK_PAIRS = 15  # 打印 Top-K 误分类对


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def name(cid: int) -> str:
    return ID2NAME.get(int(cid), str(int(cid)))


def build_confusion(df: pd.DataFrame, n_cls: int) -> pd.DataFrame:
    cm = pd.crosstab(df["gt"], df["pred"], dropna=False)
    # 补齐缺失类
    idx = list(range(n_cls))
    cm = cm.reindex(index=idx, columns=idx, fill_value=0)
    cm.index = [f"{i:02d}_{name(i)}" for i in cm.index]
    cm.columns = [f"{i:02d}_{name(i)}" for i in cm.columns]
    return cm


def per_class_report(cm: pd.DataFrame) -> pd.DataFrame:
    # cm: rows=gt, cols=pred
    diag = pd.Series([cm.iloc[i, i] for i in range(len(cm))], index=cm.index)
    support = cm.sum(axis=1)  # gt count
    pred_cnt = cm.sum(axis=0) # predicted count

    recall = diag / support.replace(0, pd.NA)
    precision = diag / pred_cnt.replace(0, pd.NA)

    out = pd.DataFrame({
        "support(gt_count)": support,
        "pred_count": pred_cnt,
        "tp": diag,
        "recall": recall,
        "precision": precision,
        "fn": (support - diag),
        "fp": (pred_cnt - diag),
    })
    out = out.sort_values("support(gt_count)", ascending=False)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(PRED_CSV)   # columns: dataset_index, gt, pred
    test_df = pd.read_csv(TEST_CSV)   # columns: rgb_path, depth_path, mask_path, class_id, ...
    assert {"dataset_index", "gt", "pred"}.issubset(pred_df.columns), f"pred_df columns: {pred_df.columns}"

    # ====== 基础统计 ======
    pred_df["gt"] = pred_df["gt"].astype(int)
    pred_df["pred"] = pred_df["pred"].astype(int)
    pred_df["is_correct"] = (pred_df["gt"] == pred_df["pred"])

    total = len(pred_df)
    correct = int(pred_df["is_correct"].sum())
    wrong = total - correct
    acc = correct / total if total else 0.0

    n_cls = max(max(ID2NAME.keys(), default=0) + 1, int(pred_df["gt"].max()) + 1, int(pred_df["pred"].max()) + 1)

    print("\n" + "=" * 70)
    print(f"[Overall] total={total} | correct={correct} | wrong={wrong} | acc={acc:.4f}")
    print("=" * 70)

    # ====== 混淆矩阵 + 每类指标 ======
    cm = build_confusion(pred_df, n_cls)
    cls_report = per_class_report(cm)

    print("\n[Per-class summary] (sorted by support)")
    # 只打印关键列，别太长
    show_cols = ["support(gt_count)", "tp", "recall", "precision", "fn", "fp"]
    print(cls_report[show_cols].to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "nan"))

    # ====== 误分类对 TopK ======
    wrong_df = pred_df[~pred_df["is_correct"]].copy()
    if len(wrong_df) > 0:
        pair_cnt = (
            wrong_df.groupby(["gt", "pred"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print("\n[Top misclassification pairs] (gt -> pred)")
        for _, r in pair_cnt.head(TOPK_PAIRS).iterrows():
            gt, pr, c = int(r["gt"]), int(r["pred"]), int(r["count"])
            print(f"  {gt:02d}_{name(gt)}  ->  {pr:02d}_{name(pr)}   : {c}")

    # ====== 导出统计CSV（更直观）======
    stats_dir = OUT_DIR / "_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    cm.to_csv(stats_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    cls_report.to_csv(stats_dir / "per_class_report.csv", encoding="utf-8-sig")
    wrong_df.to_csv(stats_dir / "misclassified_rows.csv", index=False, encoding="utf-8-sig")
    print(f"\n[Saved stats] -> {stats_dir}")

    # ====== 拷贝错分样本 ======
    print("\n[Copy misclassified samples] ...")
    for _, r in wrong_df.iterrows():
        i = int(r["dataset_index"])
        gt = int(r["gt"])
        pr = int(r["pred"])

        row = test_df.iloc[i]
        rgb = Path(row["rgb_path"]) if "rgb_path" in row else None
        depth = Path(row["depth_path"]) if "depth_path" in row else None
        mask = Path(row["mask_path"]) if "mask_path" in row else None

        gt_name = name(gt)
        pr_name = name(pr)

        # 分目录：gt -> pred
        sub = OUT_DIR / f"gt_{gt:02d}_{gt_name}__pred_{pr:02d}_{pr_name}"
        sub.mkdir(parents=True, exist_ok=True)

        # 统一命名（保留 idx）
        base = f"idx{i:05d}__gt_{gt:02d}__pred_{pr:02d}"

        if COPY_RGB and rgb and rgb.exists():
            safe_copy(rgb, sub / f"{base}_rgb{rgb.suffix}")
        if COPY_DEPTH and depth and depth.exists():
            safe_copy(depth, sub / f"{base}_depth{depth.suffix}")
        if COPY_MASK and mask and mask.exists():
            safe_copy(mask, sub / f"{base}_mask{mask.suffix}")

    print(f"\nDone. Misclassified saved to: {OUT_DIR}")
    print("Tip: open _stats/per_class_report.csv and confusion_matrix.csv for analysis.")


if __name__ == "__main__":
    main()