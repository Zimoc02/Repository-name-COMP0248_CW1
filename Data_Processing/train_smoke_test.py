# 跑一个最小训练流程（5 step），验证数据加载 → 模型 → loss → 反向传播是否正常。


import torch
from torch.utils.data import DataLoader

from src.dataloader import COMP0248KeyframeDataset, DataConfig


from src.model import UNetMultiTask
from src.losses import MultiTaskLoss

CSV_PATH = r"C:\Users\zimoc\Desktop\COMP0248_CW1\Data_Processing\meta\index_keyframes.csv"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cfg = DataConfig(
        csv_path=CSV_PATH,
        split="train",
        use_depth=True,
        keep_original_size=True,
        target_size=None,
    )
    ds = COMP0248KeyframeDataset(cfg)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    model = UNetMultiTask(in_channels=4, num_classes=10, base_ch=32).to(device)
    loss_fn = MultiTaskLoss(w_seg=1.0, w_dice=1.0, w_cls=0.5).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for step, (x, mask, y) in enumerate(dl):
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)

        seg_logits, cls_logits = model(x)
        loss, parts = loss_fn(seg_logits, cls_logits, mask, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        print(f"step {step} | loss {loss.item():.4f} | "
              f"bce {parts['bce'].item():.4f} dice {parts['dice'].item():.4f} cls {parts['cls'].item():.4f}")

        if step >= 4:  # 跑 5 个 step 就够验证链路
            break

if __name__ == "__main__":
    main()