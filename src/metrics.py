import torch


@torch.no_grad()
def dice_from_logits(seg_logits: torch.Tensor, mask_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    seg_logits: [B,1,H,W]
    mask_gt:    [B,1,H,W] in {0,1}
    returns mean dice over batch (scalar tensor)
    """
    probs = torch.sigmoid(seg_logits)
    probs = probs.view(probs.size(0), -1)
    gt = mask_gt.view(mask_gt.size(0), -1)

    inter = (probs * gt).sum(dim=1)
    denom = probs.sum(dim=1) + gt.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()


@torch.no_grad()
def iou_from_logits(seg_logits: torch.Tensor, mask_gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Binary IoU (foreground)
    """
    pred = (torch.sigmoid(seg_logits) > thr).float()
    pred = pred.view(pred.size(0), -1)
    gt = mask_gt.view(mask_gt.size(0), -1)

    inter = (pred * gt).sum(dim=1)
    union = pred.sum(dim=1) + gt.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


@torch.no_grad()
def bbox_from_mask(mask_bin: torch.Tensor):
    """
    mask_bin: [H,W] tensor (0/1), on any device
    return (xmin, ymin, xmax, ymax) as Python ints, or None if empty
    """
    ys, xs = torch.where(mask_bin > 0)
    if xs.numel() == 0:
        return None
    xmin = int(xs.min().item())
    xmax = int(xs.max().item())
    ymin = int(ys.min().item())
    ymax = int(ys.max().item())
    return xmin, ymin, xmax, ymax


@torch.no_grad()
def bbox_iou(box_a, box_b, eps: float = 1e-6) -> float:
    """
    box: (xmin, ymin, xmax, ymax) inclusive coords
    """
    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter

    return float(inter / (union + eps))


@torch.no_grad()
def bbox_metrics_from_masks(seg_logits: torch.Tensor, mask_gt: torch.Tensor, thr: float = 0.5):
    """
    Compute mean bbox IoU and acc@0.5 over batch.
    BBox derived from (pred mask) and (gt mask).
    """
    probs = torch.sigmoid(seg_logits)
    pred_bin = (probs > thr).float()  # [B,1,H,W]
    B = pred_bin.size(0)

    ious = []
    acc05 = 0
    for i in range(B):
        pb = pred_bin[i, 0]
        gb = (mask_gt[i, 0] > 0.5).float()

        box_p = bbox_from_mask(pb)
        box_g = bbox_from_mask(gb)
        iou = bbox_iou(box_p, box_g)
        ious.append(iou)
        if iou >= 0.5:
            acc05 += 1

    mean_iou = sum(ious) / max(1, len(ious))
    acc05 = acc05 / max(1, B)
    return mean_iou, acc05