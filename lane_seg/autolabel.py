#!/usr/bin/env python3
"""Auto-label raw frames into 3-class bird's-eye segmentation masks.

For each raw frame:
  1. crop the top (drop the horizon/room) and warp to the metric bird's-eye
     (same homography/geometry the C++ node uses, so the model is consistent).
  2. lane_line  = lighting-robust dark-line mask: CLAHE on the LAB L-channel +
     adaptive threshold + morphology. (Class 1)
  3. stop_line  = horizontal bands (rows with many dark pixels spread across the
     width = a perpendicular stop-line, even when dashed). (Class 2)
Outputs, per frame <name>:
  images/<name>.png    bird's-eye RGB (the model INPUT)
  masks/<name>.png     uint8 mask, pixel value in {0,1,2} (the model LABEL)
  overlays/<name>.png  RGB overlay for human review (green=lane, red=stop)

Classes: 0 background, 1 lane_line, 2 stop_line.

Usage:
  python3 autolabel.py --frames raw_frames --out labeled [--limit N] [--overlays-only]
"""
import argparse
import glob
import os

import numpy as np
import cv2

# bird's-eye geometry — MUST match lane_pilot config (the C++ inference warp)
X_MIN, X_MAX, Y_HALF, MPP = 0.01, 0.70, 0.60, 0.0025
RAW_TOP, RAW_BOT, RAW_L, RAW_R = 0.36, 1.0, 0.0, 1.0


class Geom:
    def __init__(self, x_min, x_max, y_half, mpp):
        self.x_min, self.x_max, self.y_half, self.mpp = x_min, x_max, y_half, mpp
        self.W = round(2 * y_half / mpp)
        self.H = round((x_max - x_min) / mpp)

    def warp_matrix(self, H):
        dx = self.x_max - self.x_min
        cY = (self.W - 1) / (2 * self.y_half)
        cX = (self.H - 1) / dx
        A = np.array([[0, -cY, (self.W - 1) / 2.0],
                      [-cX, 0, self.x_max * cX], [0, 0, 1]], float)
        return A @ H


def crop_raw(raw):
    h, w = raw.shape[:2]
    t, b = int(RAW_TOP * h), int(RAW_BOT * h)
    l, r = int(RAW_L * w), int(RAW_R * w)
    out = np.zeros_like(raw)
    out[t:b, l:r] = raw[t:b, l:r]
    return out


def _row_runs_keep(binrow, min_px, max_px):
    """True where a horizontal dark run has width in [min_px, max_px] (line-like)."""
    out = np.zeros_like(binrow, dtype=bool)
    n = len(binrow)
    c = 0
    while c < n:
        if not binrow[c]:
            c += 1
            continue
        c0 = c
        while c < n and binrow[c]:
            c += 1
        if min_px <= (c - c0) <= max_px:
            out[c0:c] = True
    return out


def auto_mask(bird, thr=108, open_px=5, close_px=3, min_px=1, max_px=26,
              stop_frac=0.45, stop_min_rows=4):
    """Clean lane mask via the tuned threshold + morph + per-row line-width filter
    (rejects wide floor blobs and thin noise); stop-line = dense contiguous rows."""
    gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
    nodata = (bird.sum(axis=2) == 0)               # warp border (no source)
    m = ((gray <= thr) & (~nodata)).astype(np.uint8) * 255
    if open_px >= 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    if close_px >= 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))

    H, W = m.shape
    lane = np.zeros((H, W), bool)
    mb = m > 0
    for r in range(H):
        if mb[r].any():
            lane[r] = _row_runs_keep(mb[r], min_px, max_px)
    mask = lane.astype(np.uint8)                   # 1 = lane line

    valid_w = np.maximum((~nodata).sum(axis=1), 1)
    stoprow = (lane.sum(axis=1) / valid_w) > stop_frac
    band = np.zeros(H, bool)
    i = 0
    while i < H:
        if stoprow[i]:
            j = i
            while j < H and stoprow[j]:
                j += 1
            if j - i >= stop_min_rows:
                band[i:j] = True
            i = j
        else:
            i += 1
    mask[band[:, None] & lane] = 2                 # 2 = stop_line (override lane)
    return mask, nodata


def overlay(bird, mask):
    viz = bird.copy()
    viz[mask == 1] = (0, 255, 0)                    # lane green
    viz[mask == 2] = (0, 0, 255)                    # stop red
    return cv2.addWeighted(bird, 0.5, viz, 0.5, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="raw_frames")
    ap.add_argument("--homography", default="ground_homography.yaml")
    ap.add_argument("--out", default="labeled")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--overlays-only", action="store_true",
                    help="only write overlays (for tuning the labeler)")
    args = ap.parse_args()

    fs = cv2.FileStorage(args.homography, cv2.FILE_STORAGE_READ)
    Hmat = fs.getNode("H_img2ground").mat(); fs.release()
    g = Geom(X_MIN, X_MAX, Y_HALF, MPP)
    Wmap = g.warp_matrix(Hmat)

    for sub in ("images", "masks", "overlays"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    if args.limit:
        files = files[:: max(1, len(files) // args.limit)][:args.limit]
    print("processing %d frames -> %s (bird %dx%d)" % (len(files), args.out, g.W, g.H))

    n_stop = 0
    for i, f in enumerate(files):
        name = os.path.splitext(os.path.basename(f))[0]
        raw = cv2.imread(f)
        if raw is None:
            continue
        bird = cv2.warpPerspective(crop_raw(raw), Wmap, (g.W, g.H), flags=cv2.INTER_LINEAR)
        mask, _ = auto_mask(bird)
        cv2.imwrite(os.path.join(args.out, "overlays", name + ".png"), overlay(bird, mask))
        if not args.overlays_only:
            cv2.imwrite(os.path.join(args.out, "images", name + ".png"), bird)
            cv2.imwrite(os.path.join(args.out, "masks", name + ".png"), mask)
        if (mask == 2).any():
            n_stop += 1
        if (i + 1) % 100 == 0:
            print("  %d/%d" % (i + 1, len(files)), flush=True)
    print("DONE. %d frames, %d had a stop-line." % (len(files), n_stop))


if __name__ == "__main__":
    main()
