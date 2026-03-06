import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    csv_path: str
    split: str = "train"          # "train" or "val"
    use_depth: bool = True        # RGB-D (True) or RGB only (False)
    keep_original_size: bool = True
    target_size: Optional[Tuple[int, int]] = None  # (W, H) if you want resize
    rgb_normalize: str = "imagenet"  # "imagenet" or "none"
    depth_normalize: str = "minmax"  # "minmax" or "none"
    threshold_mask: int = 0          # mask > threshold => 1


class COMP0248KeyframeDataset(Dataset):
    """
    Reads rows from index_keyframes.csv.
    Returns:
      x: FloatTensor [C, H, W] where C=4 if use_depth else 3
      mask: FloatTensor [1, H, W] values in {0,1}
      y: LongTensor scalar class_id in [0..9]
    """
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        df = pd.read_csv(cfg.csv_path)
        df = df[df["split"] == cfg.split].reset_index(drop=True)

        # basic sanity: ensure key columns exist
        need_cols = ["rgb_path", "depth_path", "mask_path", "class_id"]
        for c in need_cols:
            if c not in df.columns:
                raise ValueError(f"CSV missing column: {c}")

        self.df = df

        if not cfg.keep_original_size:
            if cfg.target_size is None:
                raise ValueError("target_size must be provided when keep_original_size=False")

        # ImageNet normalization (common baseline)
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _read_rgb(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"RGB not found or unreadable: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb  # uint8 HxWx3

    def _read_depth(self, path: str) -> np.ndarray:
        # Depth PNG can be 8-bit or 16-bit depending on your export.
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(f"Depth not found or unreadable: {path}")
        if d.ndim == 3:
            # sometimes depth png might be saved as 3-channel; take one channel
            d = d[:, :, 0]
        return d  # uint8/uint16 HxW

    def _read_mask(self, path: str) -> np.ndarray:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {path}")
        return m  # uint8 HxW

    def _resize_if_needed(self, rgb, depth, mask):
        if self.cfg.keep_original_size:
            return rgb, depth, mask

        W, H = self.cfg.target_size  # (W,H)
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        if depth is not None:
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        # mask must be nearest
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return rgb, depth, mask

    def _normalize_rgb(self, rgb_f: np.ndarray) -> np.ndarray:
        # rgb_f: float32 in [0,1]
        if self.cfg.rgb_normalize == "imagenet":
            return (rgb_f - self.imagenet_mean) / self.imagenet_std
        elif self.cfg.rgb_normalize == "none":
            return rgb_f
        else:
            raise ValueError("rgb_normalize must be 'imagenet' or 'none'")

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        # depth: uint8/uint16 HxW
        d = depth.astype(np.float32)

        if self.cfg.depth_normalize == "minmax":
            # per-image min-max; robust enough for baseline
            mn, mx = float(np.min(d)), float(np.max(d))
            if mx - mn < 1e-6:
                return np.zeros_like(d, dtype=np.float32)
            return (d - mn) / (mx - mn)
        elif self.cfg.depth_normalize == "none":
            # scale to [0,1] based on dtype range if possible
            if depth.dtype == np.uint8:
                return d / 255.0
            if depth.dtype == np.uint16:
                return d / 65535.0
            return d
        else:
            raise ValueError("depth_normalize must be 'minmax' or 'none'")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rgb_path = str(row["rgb_path"])
        depth_path = str(row["depth_path"])
        mask_path = str(row["mask_path"])
        y = int(row["class_id"])

        rgb = self._read_rgb(rgb_path)
        depth = self._read_depth(depth_path) if self.cfg.use_depth else None
        mask = self._read_mask(mask_path)

        rgb, depth, mask = self._resize_if_needed(rgb, depth, mask)

        # mask -> {0,1}
        mask_bin = (mask > self.cfg.threshold_mask).astype(np.float32)  # HxW

        # rgb -> float [0,1] then normalize
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_n = self._normalize_rgb(rgb_f)  # HxWx3

        if self.cfg.use_depth:
            depth_n = self._normalize_depth(depth)  # HxW float [0,1]
            depth_n = depth_n[:, :, None]           # HxWx1
            x = np.concatenate([rgb_n, depth_n], axis=2)  # HxWx4
        else:
            x = rgb_n  # HxWx3

        # to torch: [C,H,W]
        x_t = torch.from_numpy(x).permute(2, 0, 1).contiguous().float()
        mask_t = torch.from_numpy(mask_bin[None, :, :]).contiguous().float()
        y_t = torch.tensor(y, dtype=torch.long)

        return x_t, mask_t, y_t