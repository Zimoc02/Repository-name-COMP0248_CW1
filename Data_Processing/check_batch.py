# 检查 DataLoader 输出的一个 batch 的 shape 和数据类型是否正确。

from torch.utils.data import DataLoader
from src.dataloader import COMP0248KeyframeDataset, DataConfig

CSV_PATH = r"C:\Users\zimoc\Desktop\COMP0248_CW1\Data_Processing\meta\index_keyframes.csv"

def main():
    cfg = DataConfig(
        csv_path=CSV_PATH,
        split="train",
        use_depth=True,
        keep_original_size=True,   # ✅ 原尺寸 640x480
        target_size=None,
    )
    ds = COMP0248KeyframeDataset(cfg)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    x, m, y = next(iter(dl))
    print("x:", x.shape, x.dtype)   # expected: [B, 4, 480, 640]
    print("m:", m.shape, m.dtype)   # expected: [B, 1, 480, 640]
    print("y:", y.shape, y.dtype, "min/max:", int(y.min()), int(y.max()))

if __name__ == "__main__":
    main()