# 扫描完整手势数据集，生成训练用的 index_keyframes.csv（核心元数据构建脚本）。

import os, json, random
from pathlib import Path
import pandas as pd

FULL_DATA = Path(r"C:\Users\zimoc\Desktop\COMP0248_CW1\COMP0248_set_dataset\data_edded\full_data")

OUT_DIR = Path(__file__).resolve().parent / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "G01_call","G02_dislike","G03_like","G04_ok","G05_one",
    "G06_palm","G07_peace","G08_rock","G09_stop","G10_three",
]

VAL_RATIO = 0.2
SEED = 42

def main():
    
    students = sorted([p for p in FULL_DATA.iterdir() if p.is_dir()])
    rng = random.Random(SEED)
    rng.shuffle(students)

    n_val = max(1, int(round(len(students) * VAL_RATIO)))
    val_students = set([p.name for p in students[:n_val]])

    rows = []

    for student_dir in students:
        student_name = student_dir.name
        split = "val" if student_name in val_students else "train"

        for gesture in CLASSES:
            gdir = student_dir / gesture
            if not gdir.is_dir():
                continue
            class_id = CLASSES.index(gesture)

            for clip_dir in sorted(gdir.glob("clip*")):
                ann_dir = clip_dir / "annotation"
                rgb_dir = clip_dir / "rgb"
                depth_dir = clip_dir / "depth"

                if not ann_dir.is_dir():
                    continue

                # ✅ 只要有 mask 的帧
                for mask_path in sorted(ann_dir.glob("frame_*.png")):
                    rgb_path = rgb_dir / mask_path.name
                    depth_path = depth_dir / mask_path.name

                    if not rgb_path.is_file():
                        continue
                    if not depth_path.is_file():
                        continue

                    rows.append({
                        "split": split,
                        "class_id": class_id,
                        "gesture": gesture,
                        "student_folder": student_name,
                        "clip": clip_dir.name,
                        "frame": mask_path.name,
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_path),
                        "mask_path": str(mask_path),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "index_keyframes.csv", index=False, encoding="utf-8")

    (OUT_DIR / "split.json").write_text(json.dumps({
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "val_students": sorted(list(val_students)),
    }, indent=2), encoding="utf-8")

    print("Saved:", OUT_DIR / "index_keyframes.csv")
    print("Rows:", len(df))

if __name__ == "__main__":
    main()