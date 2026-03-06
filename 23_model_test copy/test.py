import pandas as pd
from pathlib import Path

IN_CSV  = Path(r"D:\0248_data_check\test_build\COMP0248_Test_data_23\test_index_23_depthpng.csv")
OUT_CSV = Path(r"D:\0248_data_check\test_build\COMP0248_Test_data_23\test_index_23_depthpng_for_loader.csv")

df = pd.read_csv(IN_CSV)

# y -> class_id
if "class_id" not in df.columns:
    if "y" in df.columns:
        df = df.rename(columns={"y": "class_id"})
    else:
        raise RuntimeError(f"Neither 'y' nor 'class_id' found in {IN_CSV}")

df.to_csv(OUT_CSV, index=False)
print("saved:", OUT_CSV)
print("columns:", df.columns.tolist())
print("rows:", len(df))