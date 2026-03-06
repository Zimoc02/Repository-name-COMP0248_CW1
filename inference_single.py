# 对单张 RGB+Depth 图片做一次推理，输出预测类别 + mask overlay 图。
from pathlib import Path
import cv2
import numpy as np
import torch

from src.model import UNetMultiTask


# ====== 修改成你的路径 ======
#RGB_PATH = r"D:\0248_data_check\test_rgb.png"
#DEPTH_PATH = r"D:\0248_data_check\test_depth.png"
CKPT_PATH = r"weights/best_val_dice.pt"
#OUT_PATH = r"D:\0248_data_check\single_result.png"

#RGB_PATH = r"D:\0248_data_check\debug_samples_val\00039_rgb.png"
#DEPTH_PATH = r"D:\0248_data_check\debug_samples_val\00039_depth.png"
RGB_PATH = r"D:\0248_data_check\single_item_test\item\00000_rgb.png"
DEPTH_PATH = r"D:\0248_data_check\single_item_test\item\00000_depth.png"
OUT_PATH = r"D:\0248_data_check\single_item_test\result\00000_result.png"

RGB_PATH = r"D:\0248_data_check\single_item_test\item\00001_rgb.png"
DEPTH_PATH = r"D:\0248_data_check\single_item_test\item\00001_depth.png"
OUT_PATH = r"D:\0248_data_check\single_item_test\result\00001_result.png"
# =============================


def preprocess(rgb_path, depth_path):
    """
    读取 RGB + Depth，并转换成模型输入 tensor [1,4,H,W]
    """
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    if rgb is None or depth is None:
        raise ValueError("无法读取图像路径")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    rgb = rgb.astype(np.float32) / 255.0
    depth = depth.astype(np.float32) / 255.0

    # ImageNet normalize（必须和训练一致）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std

    # 拼成4通道
    depth = depth[..., None]
    x = np.concatenate([rgb, depth], axis=2)

    # HWC → CHW
    x = np.transpose(x, (2, 0, 1))

    x = torch.from_numpy(x).float().unsqueeze(0)
    return x


def overlay_mask(rgb, mask, alpha=0.5):
    vis = rgb.copy()
    color = np.zeros_like(rgb)
    color[..., 1] = 255
    vis[mask > 0] = (
        alpha * color[mask > 0] + (1 - alpha) * vis[mask > 0]
    ).astype(np.uint8)
    return vis


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1️⃣ 加载模型
    model = UNetMultiTask(in_channels=4, num_classes=10, base_ch=32).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Loaded model from:", CKPT_PATH)

    # 2️⃣ 预处理
    x = preprocess(RGB_PATH, DEPTH_PATH).to(device)

    # 3️⃣ 推理
    with torch.no_grad():
        seg_logits, cls_logits = model(x)

        prob = torch.sigmoid(seg_logits)[0, 0]
        pred_mask = (prob > 0.5).cpu().numpy().astype(np.uint8)

        pred_class = cls_logits.argmax(dim=1).item()

    print("Predicted class:", pred_class)

    # 4️⃣ 可视化
    rgb = cv2.imread(RGB_PATH)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    vis = overlay_mask(rgb, pred_mask)

    cv2.putText(
        vis,
        f"class: {pred_class}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(OUT_PATH, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("Saved result to:", OUT_PATH)


if __name__ == "__main__":
    main()