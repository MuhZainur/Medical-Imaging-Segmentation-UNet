"""python train.py --images data/images --masks data/masks --epochs 10"""
import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import SegDataset
from src.unet_model import build_model, dice_iou

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--masks", required=True)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--out", default="unet.pt")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using compute device: {dev}")

full_ds = SegDataset(a.images, a.masks)
idx = torch.randperm(len(full_ds), generator=torch.Generator().manual_seed(0)).tolist()
cut = max(1, int(0.8 * len(idx)))
train_dl = DataLoader(Subset(full_ds, idx[:cut]), a.batch, shuffle=True)
val_dl = DataLoader(Subset(full_ds, idx[cut:]), max(1, a.batch))

model = build_model().to(dev)
bce = nn.BCEWithLogitsLoss()

def dice_loss_fn(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.sigmoid()
    inter = (probs * target).sum((1, 2, 3))
    union = probs.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()

opt = torch.optim.Adam(model.parameters(), 1e-3)

best = 0.0
for ep in range(a.epochs):
    model.train()
    total_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss = dice_loss_fn(out, y) + bce(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        m = [dice_iou(model(x.to(dev)), y.to(dev)) for x, y in val_dl]
    dice, iou = (sum(v[i] for v in m) / len(m) for i in (0, 1))
    print(f"Epoch {ep + 1:02d}/{a.epochs:02d} | Loss: {total_loss/len(train_dl):.4f} | Val Dice: {dice:.4f} | IoU: {iou:.4f}")
    if dice > best or ep == 0:
        best = dice
        torch.save(model.state_dict(), a.out)

print(f"[SUCCESS] Best Val Dice: {best:.4f}, saved checkpoint to {a.out}")
