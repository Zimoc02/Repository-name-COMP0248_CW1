import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """
    Upsample + concat skip + DoubleConv
    Uses bilinear upsample (lighter than transposed conv).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetMultiTask(nn.Module):
    """
    Input:  [B, 4, H, W]  (RGB-D)
    Output:
      seg_logits: [B, 1, H, W]
      cls_logits: [B, 10]
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 10, base_ch: int = 32, dropout_p: float = 0.3):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, base_ch)            # H,W
        self.down1 = Down(base_ch, base_ch * 2)               # H/2
        self.down2 = Down(base_ch * 2, base_ch * 4)           # H/4
        self.down3 = Down(base_ch * 4, base_ch * 8)           # H/8
        self.down4 = Down(base_ch * 8, base_ch * 16)          # H/16 (bottleneck)

        # Decoder (segmentation)
        self.up3 = Up(base_ch * 16, base_ch * 8, base_ch * 8)   # back to H/8
        self.up2 = Up(base_ch * 8,  base_ch * 4, base_ch * 4)   # H/4
        self.up1 = Up(base_ch * 4,  base_ch * 2, base_ch * 2)   # H/2
        self.up0 = Up(base_ch * 2,  base_ch,     base_ch)       # H

        self.seg_head = nn.Conv2d(base_ch, 1, kernel_size=1)

        # Classification head from bottleneck feature
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # [B, C, 1, 1]
            nn.Flatten(),                         # [B, C]
            nn.Linear(base_ch * 16, base_ch * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(base_ch * 8, num_classes),
        )

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)     # [B, base, H, W]
        x1 = self.down1(x0)  # [B, 2b, H/2, W/2]
        x2 = self.down2(x1)  # [B, 4b, H/4, W/4]
        x3 = self.down3(x2)  # [B, 8b, H/8, W/8]
        xb = self.down4(x3)  # [B,16b, H/16,W/16]

        # Heads
        cls_logits = self.cls_head(xb)

        # Decoder
        d3 = self.up3(xb, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        d0 = self.up0(d1, x0)

        seg_logits = self.seg_head(d0)

        return seg_logits, cls_logits