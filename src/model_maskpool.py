import torch
import torch.nn as nn
import torch.nn.functional as F

# 原模型，不改它
from src.model import UNetMultiTask


class MaskGuidedPooling(nn.Module):
    """
    Mask-guided pooling:
    fb:   [B, C, hb, wb]
    mask: [B,1,H,W] or [B,H,W] or [H,W] (GT 0/1 or 0/255, or prob in [0,1])
    out:  [B, C]

    v = sum(F * M) / (sum(M) + eps)
    若 mask 太小/为空 -> 回退为 GAP，避免数值问题/预测崩溃
    """

    def __init__(self, eps: float = 1e-6, empty_thr: float = 1.0):
        super().__init__()
        self.eps = eps
        self.empty_thr = empty_thr

    @staticmethod
    def _ensure_b1hw(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:          # [H,W]
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:        # [B,H,W]
            mask = mask.unsqueeze(1)
        elif mask.dim() == 4:        # [B,1,H,W]
            pass
        else:
            raise ValueError(f"Unsupported mask dim: {mask.dim()}")
        return mask

    def forward(self, fb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, hb, wb = fb.shape

        # 1) 形状统一 + device/dtype 对齐
        mask = self._ensure_b1hw(mask).to(device=fb.device, dtype=fb.dtype)

        # 2) 兜底：0/255 -> 0/1
        if mask.max() > 1.0:
            mask = mask / 255.0

        # 3) resize 到 bottleneck 尺度
        # 简单策略：取值多 -> soft mask -> bilinear；接近二值 -> nearest
        flat = mask.detach().flatten()
        sample = flat[:: max(flat.numel() // 2048, 1)]
        uniq = torch.unique(sample)

        if uniq.numel() > 4:
            mask_small = F.interpolate(mask, size=(hb, wb), mode="bilinear", align_corners=False)
        else:
            mask_small = F.interpolate(mask, size=(hb, wb), mode="nearest")

        # 4) masked average pooling
        weighted = fb * mask_small            # [B,C,hb,wb]
        num = weighted.sum(dim=(2, 3))        # [B,C]
        den = mask_small.sum(dim=(2, 3))      # [B,1]
        den = den.clamp_min(self.eps)

        pooled = num / den                    # broadcast -> [B,C]

        # 5) 空 mask 回退
        den_scalar = den.squeeze(1)           # [B]
        is_empty = den_scalar < self.empty_thr
        if is_empty.any():
            gap = fb.mean(dim=(2, 3))         # [B,C]
            pooled[is_empty] = gap[is_empty]

        return pooled


class MLPHead(nn.Module):
    """Linear -> ReLU -> Dropout -> Linear"""

    def __init__(self, in_dim: int, num_classes: int = 10, hidden: int = 256, p_drop: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class UNetMultiTask_MaskPool(nn.Module):
    """
    Wrapper：不改 UNetMultiTask。
    - 用 hook 抓 down4 输出作为 fb（512通道 bottleneck）
    - 用 mask-guided pooling 得到 cls_feat
    - 用我们自己的 cls_head 输出分类
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 10,
        base_ch: int = 32,
        cls_hidden: int = 256,
        cls_drop: float = 0.3,
        use_gt_for_cls_when_training: bool = True,
        empty_thr: float = 1.0,
        bottleneck_layer_name: str = "down4",
    ):
        super().__init__()

        # 原模型（保持不改）
        self.base = UNetMultiTask(in_channels=in_channels, num_classes=num_classes, base_ch=base_ch)

        # pooling + 新分类头（注意：这里我们直接写死 in_dim=512，因为你 down4 输出就是 512）
        self.mask_pool = MaskGuidedPooling(eps=1e-6, empty_thr=empty_thr)
        self.cls_head = MLPHead(in_dim=512, num_classes=num_classes, hidden=cls_hidden, p_drop=cls_drop)

        self.use_gt_for_cls_when_training = use_gt_for_cls_when_training

        # hook 缓存
        self._fb = None
        self._register_bottleneck_hook(bottleneck_layer_name)

    def _register_bottleneck_hook(self, layer_name: str):
        module = self.base
        for part in layer_name.split("."):
            if not hasattr(module, part):
                raise AttributeError(f"UNetMultiTask has no submodule '{part}' in path '{layer_name}'")
            module = getattr(module, part)

        def hook_fn(_module, _inp, out):
            self._fb = out

        module.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor, mask_gt: torch.Tensor = None):
        self._fb = None

        out = self.base(x)

        # base 返回 (seg_logits, cls_logits)，我们只需要 seg_logits
        if isinstance(out, (tuple, list)):
            seg_logits = out[0]
        elif isinstance(out, dict):
            seg_logits = out["seg_logits"]
        else:
            raise ValueError(f"Unexpected base output type: {type(out)}")

        if self._fb is None:
            raise RuntimeError(
                "Hook did not capture bottleneck feature. "
                "Check bottleneck_layer_name (should be 'down4')."
            )
        fb = self._fb  # [B,512,hb,wb]

        # mask 选择：训练优先用 GT，推理/验证用 pred mask（soft）
        if self.training and self.use_gt_for_cls_when_training and (mask_gt is not None):
            mask_for_cls = mask_gt
        else:
            mask_for_cls = torch.sigmoid(seg_logits)

        cls_feat = self.mask_pool(fb, mask_for_cls)  # [B,512]
        cls_logits = self.cls_head(cls_feat)         # [B,10]

        return seg_logits, cls_logits