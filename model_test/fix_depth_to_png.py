import pandas as pd
from pathlib import Path

IN_CSV  = Path(r"D:\0248_data_check\test_build\test_index_for_loader.csv")
OUT_CSV = Path(r"D:\0248_data_check\test_build\test_index_for_loader_depthpng.csv")

df = pd.read_csv(IN_CSV)

swap = 0
drop = 0
rows = []

for _, row in df.iterrows():
    dpath = Path(row["depth_path"])

    # try: ...\depth_raw\frame_014.npy -> ...\depth\frame_014.png
    png = Path(str(dpath).replace("\\depth_raw\\", "\\depth\\")).with_suffix(".png")

    if png.exists():
        row["depth_path"] = str(png)
        swap += 1
        rows.append(row)
    else:
        # 如果某些样本没有 depth png，就先丢掉，避免 dataloader 报错
        drop += 1

df2 = pd.DataFrame(rows)
df2.to_csv(OUT_CSV, index=False)

print("saved:", OUT_CSV)
print("rows_in:", len(df), "rows_out:", len(df2))
print("swapped:", swap, "dropped:", drop)