#!/usr/bin/env python3
"""Train a small UNet to segment {background, lane_line, stop_line} on the
bird's-eye images, and export to ONNX for TensorRT on the Jetson.

Run on your LAB GPU MACHINE (needs torch + CUDA):
  pip install torch torchvision opencv-python numpy
  python3 train_unet.py --data labeled --epochs 80 --w 320 --h 192

Dataset = labeled/images + labeled/masks, filtered by labeled/keep.txt (from
review.py). Strong lighting/blur augmentation is the whole point — it's what
makes the model robust where the threshold wasn't.

Outputs: runs/unet_best.pt and runs/lane_unet.onnx (NCHW, input 1x3xHxW float
0..1, output 1x3xHxW logits -> argmax = class).
"""
import argparse
import glob
import os
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NUM_CLASSES = 3   # 0 bg, 1 lane_line, 2 stop_line


# ----------------------------- model -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small UNet (transposed-conv upsampling -> TensorRT 8.2 friendly)."""
    def __init__(self, n=NUM_CLASSES, b=24):
        super().__init__()
        self.d1 = DoubleConv(3, b)
        self.d2 = DoubleConv(b, b * 2)
        self.d3 = DoubleConv(b * 2, b * 4)
        self.d4 = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.c3 = DoubleConv(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.c2 = DoubleConv(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.c1 = DoubleConv(b * 2, b)
        self.out = nn.Conv2d(b, n, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))
        y = self.c3(torch.cat([self.u3(x4), x3], 1))
        y = self.c2(torch.cat([self.u2(y), x2], 1))
        y = self.c1(torch.cat([self.u1(y), x1], 1))
        return self.out(y)


# ----------------------------- data ------------------------------
class SegDS(Dataset):
    def __init__(self, items, w, h, train):
        self.items, self.w, self.h, self.train = items, w, h, train

    def __len__(self):
        return len(self.items)

    def _aug(self, img, m):
        if random.random() < 0.5:                              # h-flip
            img, m = img[:, ::-1], m[:, ::-1]
        # lighting: brightness + contrast + gamma (the key augmentation)
        a = random.uniform(0.6, 1.5); bca = random.uniform(-40, 40)
        img = np.clip(img.astype(np.float32) * a + bca, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            g = random.uniform(0.6, 1.6)
            img = np.clip(((img / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
        if random.random() < 0.3:                              # blur (motion)
            k = random.choice([3, 5]); img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < 0.3:                              # noise
            img = np.clip(img.astype(np.int16) +
                          np.random.randint(-12, 12, img.shape, np.int16), 0, 255).astype(np.uint8)
        if random.random() < 0.5:                              # small affine
            H, W = img.shape[:2]
            ang = random.uniform(-7, 7); sc = random.uniform(0.9, 1.1)
            tx, ty = random.uniform(-0.05, 0.05) * W, random.uniform(-0.05, 0.05) * H
            M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc); M[0, 2] += tx; M[1, 2] += ty
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
            m = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST)
        return img, m

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img = cv2.imread(ip)
        m = cv2.imread(mp, 0)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if self.train:
            img, m = self._aug(img, m)
        img = np.ascontiguousarray(img[:, :, ::-1])            # BGR->RGB
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        y = torch.from_numpy(m.astype(np.int64).copy())
        return x, y


def load_items(data):
    keepf = os.path.join(data, "keep.txt")
    names = (open(keepf).read().split() if os.path.exists(keepf)
             else [os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(data, "images", "*.png"))])
    items = []
    for n in names:
        ip = os.path.join(data, "images", n + ".png")
        mp = os.path.join(data, "masks", n + ".png")
        if os.path.exists(ip) and os.path.exists(mp):
            items.append((ip, mp))
    random.Random(0).shuffle(items)
    return items


# ----------------------------- loss/metric -----------------------
def dice_loss(logits, y):
    p = F.softmax(logits, 1)
    yh = F.one_hot(y, NUM_CLASSES).permute(0, 3, 1, 2).float()
    inter = (p * yh).sum((0, 2, 3)); union = (p + yh).sum((0, 2, 3))
    return (1 - (2 * inter + 1) / (union + 1)).mean()


@torch.no_grad()
def iou(logits, y):
    pred = logits.argmax(1)
    out = []
    for c in range(NUM_CLASSES):
        i = ((pred == c) & (y == c)).sum().item()
        u = ((pred == c) | (y == c)).sum().item()
        out.append(i / u if u else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="labeled")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--w", type=int, default=320)
    ap.add_argument("--h", type=int, default=192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, "| input %dx%d" % (args.w, args.h))

    items = load_items(args.data)
    nval = max(1, len(items) // 10)
    tr, va = items[nval:], items[:nval]
    print("train", len(tr), "val", len(va))
    dl_tr = DataLoader(SegDS(tr, args.w, args.h, True), args.batch, shuffle=True,
                       num_workers=4, drop_last=True)
    dl_va = DataLoader(SegDS(va, args.w, args.h, False), args.batch, num_workers=2)

    net = UNet(b=args.base).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    # bg is ~90% of pixels -> weight lane/stop up
    wt = torch.tensor([0.5, 3.0, 5.0], device=dev)
    best = -1
    for ep in range(args.epochs):
        net.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            logit = net(x)
            loss = F.cross_entropy(logit, y, weight=wt) + dice_loss(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval(); ious = []
        with torch.no_grad():
            for x, y in dl_va:
                ious.append(iou(net(x.to(dev)), y.to(dev)))
        miou = np.nanmean(ious, 0)
        score = np.nanmean(miou[1:])               # mean of lane+stop IoU
        print("ep %2d  loss %.3f  IoU bg/lane/stop %.2f/%.2f/%.2f  score %.3f"
              % (ep, loss.item(), miou[0], miou[1], miou[2], score), flush=True)
        if score > best:
            best = score
            torch.save(net.state_dict(), os.path.join(args.out, "unet_best.pt"))

    # ---- ONNX export (best) ----
    net.load_state_dict(torch.load(os.path.join(args.out, "unet_best.pt")))
    net.eval()
    dummy = torch.randn(1, 3, args.h, args.w, device=dev)
    onnx = os.path.join(args.out, "lane_unet.onnx")
    torch.onnx.export(net, dummy, onnx, opset_version=12,
                      input_names=["input"], output_names=["logits"],
                      do_constant_folding=True)
    print("BEST lane+stop IoU %.3f  ->  %s  (input 1x3x%dx%d)" % (best, onnx, args.h, args.w))


if __name__ == "__main__":
    main()
