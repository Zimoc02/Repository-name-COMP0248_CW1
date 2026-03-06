# 只做一次前向传播（不训练），检查模型输出 tensor 的 shape 是否正确。

import torch
from torch.utils.data import DataLoader

from src.dataloader import COMP0248KeyframeDataset, DataConfig
from src.model import UNetMultiTask

CSV_PATH = r"C:\Users\zimoc\Desktop\COMP0248_CW1\Data_Processing\meta\index_keyframes.csv"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # dataset / loader
    cfg = DataConfig(
        csv_path=CSV_PATH,
        split="train",
        use_depth=True,
        keep_original_size=True,   # 原尺寸 480x640
        target_size=None,
    )
    ds = COMP0248KeyframeDataset(cfg)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    x, m, y = next(iter(dl))
    x = x.to(device)
    m = m.to(device)
    y = y.to(device)

    # model
    model = UNetMultiTask(in_channels=4, num_classes=10, base_ch=32).to(device)
    model.eval()

    with torch.no_grad():
        seg_logits, cls_logits = model(x)

    print("input x:", x.shape)
    print("gt mask:", m.shape, "gt y:", y.shape)
    print("seg_logits:", seg_logits.shape)
    print("cls_logits:", cls_logits.shape)

if __name__ == "__main__":
    main()