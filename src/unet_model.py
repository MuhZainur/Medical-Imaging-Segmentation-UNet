"""Pure PyTorch Light U-Net Architecture and Dice/IoU Evaluation Metrics.
Designed with zero third-party dependencies beyond native PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2 block."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Classic U-Net with contracting path (encoder), bottleneck, and expanding path (decoder)
    with skip connections for pixel-level medical segmentation.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = self.up1(x4)
        u1 = torch.cat([x3, u1], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([x2, u2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([x1, u3], dim=1)
        u3 = self.conv_up3(u3)

        logits = self.outc(u3)
        return logits


def build_model(pretrained: bool = False) -> torch.nn.Module:
    """Builds native U-Net."""
    return UNet(in_channels=3, num_classes=1)


def dice_iou(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    """Calculate mean Dice coefficient and IoU per batch."""
    p = (logits.sigmoid() > 0.5).float()
    inter = (p * mask).sum((1, 2, 3))
    total = p.sum((1, 2, 3)) + mask.sum((1, 2, 3))
    dice = (2 * inter + eps) / (total + eps)
    iou = (inter + eps) / (total - inter + eps)
    return dice.mean().item(), iou.mean().item()
