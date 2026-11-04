"""Medical Image & Mask Dataset Loader with CLAHE Preprocessing & Torchvision."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

EXTS = {".png", ".jpg", ".jpeg"}


def enhance_radiograph_clahe(gray_img: np.ndarray) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for pleural lung boundaries."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def transform_image_and_mask(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resizes, normalizes, and converts image & mask to PyTorch tensors."""
    img_resized = cv2.resize(img_rgb, (size, size))
    mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    # Convert image to (C, H, W) normalized tensor
    img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float() / 255.0
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor)

    # Convert binary mask to (1, H, W) float tensor
    mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)
    return img_tensor, mask_tensor


class SegDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, size: int = 256, use_clahe: bool = True) -> None:
        masks = {p.stem: p for p in Path(mask_dir).iterdir() if p.suffix.lower() in EXTS}
        self.pairs = [
            (p, masks[p.stem]) for p in sorted(Path(img_dir).iterdir())
            if p.suffix.lower() in EXTS and p.stem in masks
        ]
        self.size = size
        self.use_clahe = use_clahe

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.pairs[i]
        gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if self.use_clahe:
            gray = enhance_radiograph_clahe(gray)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        mask = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0).astype("uint8")
        return transform_image_and_mask(img, mask, size=self.size)



def generate_synthetic_chest_xray(has_lesion: bool = True, size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic chest radiograph with optional lung lesion mask."""
    # Dark radiographic background with lung rib cage oval shapes
    img = np.full((size, size), 30, dtype=np.uint8)
    # Left and right lung fields (dark air regions)
    cv2.ellipse(img, (size // 3, size // 2), (size // 6, size // 3), 0, 0, 360, 90, -1)
    cv2.ellipse(img, (2 * size // 3, size // 2), (size // 6, size // 3), 0, 0, 360, 90, -1)
    # Add subtle rib shadows
    for y in range(size // 4, 3 * size // 4, 25):
        cv2.line(img, (size // 5, y), (4 * size // 5, y + 10), 120, 3)

    # Gaussian blur for radiographic smoothness
    img = cv2.GaussianBlur(img, (15, 15), 0)
    
    mask = np.zeros((size, size), dtype=np.uint8)
    if has_lesion:
        # Infiltrate / pneumonia opacity in right lower lung
        center = (2 * size // 3 + 10, size // 2 + 25)
        cv2.circle(img, center, size // 10, 180, -1)
        cv2.circle(mask, center, size // 10, 1, -1)

    return img, mask
