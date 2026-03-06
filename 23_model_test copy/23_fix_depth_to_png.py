# 把 CSV 里的 depth_raw.npy 替换为 depth.png

import pandas as pd
from pathlib import Path

# === 改成你新数据集的子目录（不覆盖旧的）===
IN_CSV  = Path(r"D:\0248_data_check\test_build\COMP0248_Test_data_23\test_index_23.csv")
OUT_CSV = Path(r"D:\0248_data_check\test_build\COMP0248_Test_data_23\test_index_23_depthpng.csv")

df = pd.read_csv(IN_CSV)

swap = 0
keep_npy = 0
rows = []

for _, row in df.iterrows():
    dpath = Path(str(row["depth_path"]))

    parts = list(dpath.parts)
    if "depth_raw" in parts:
        parts[parts.index("depth_raw")] = "depth"

    png = Path(*parts).with_suffix(".png")

    if png.exists():
        row["depth_path"] = str(png)
        swap += 1
    else:
        # 找不到 png：保留原来的 npy（不丢行）
        keep_npy += 1

    rows.append(row)

df2 = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df2.to_csv(OUT_CSV, index=False)

print("saved:", OUT_CSV)
print("rows_in:", len(df), "rows_out:", len(df2))
print("swapped_to_png:", swap, "kept_npy:", keep_npy)