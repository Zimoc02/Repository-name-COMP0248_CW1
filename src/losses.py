import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits:  [B,1,H,W]
    targets: [B,1,H,W] in {0,1}
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, w_seg: float = 1.0, w_dice: float = 1.0, w_cls: float = 0.5):
        super().__init__()
        self.w_seg = w_seg
        self.w_dice = w_dice
        self.w_cls = w_cls
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, seg_logits, cls_logits, mask_gt, y_gt):
        l_bce = self.bce(seg_logits, mask_gt)
        l_dice = dice_loss_from_logits(seg_logits, mask_gt)
        l_cls = self.ce(cls_logits, y_gt)

        total = self.w_seg * l_bce + self.w_dice * l_dice + self.w_cls * l_cls
        return total, {"bce": l_bce.detach(), "dice": l_dice.detach(), "cls": l_cls.detach()}