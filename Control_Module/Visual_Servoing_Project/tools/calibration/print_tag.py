#!/usr/bin/env python3
"""
Generate a printable PNG of a tag36h11 AprilTag at a chosen physical size.

OpenCV's ArUco module ships dictionaries for the AprilTag families
(DICT_APRILTAG_36h11), so we use it as the generator — no extra deps.

Print the resulting PNG at 100 % scale (no fit-to-page). Measure the
black border after printing to confirm the side length matches what you
passed to --size; that side length must be passed as `tag_size` to the
detector.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--id", type=int, default=0,
                   help="Tag id within the tag36h11 family (default: 0).")
    p.add_argument("--size", type=float, default=0.133,
                   help="Side length in meters, black border to black border (default: 0.133).")
    p.add_argument("--dpi", type=int, default=300,
                   help="Print DPI (default: 300).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG (default: tag36h11_<id>_<size>m.png).")
    args = p.parse_args()

    px = int(round(args.size * 1000.0 / 25.4 * args.dpi))
    quiet = px // 4  # AprilTag spec recommends a white quiet zone >= 1 module

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, args.id, px)

    canvas = np.full((px + 2 * quiet, px + 2 * quiet), 255, dtype=np.uint8)
    canvas[quiet:quiet + px, quiet:quiet + px] = tag

    out = args.out or Path(f"tag36h11_{args.id}_{args.size:.2f}m.png")
    cv2.imwrite(str(out), canvas)

    inches = args.size / 0.0254
    print(f"Wrote {out}")
    print(f"  tag:    tag36h11 id {args.id}")
    print(f"  size:   {args.size*1000:.0f} mm ({inches:.2f} in) at {args.dpi} DPI")
    print(f"  pixels: {px} (+ {quiet}px quiet zone each side)")
    print("Print at 100% scale and measure the black border to verify.")


if __name__ == "__main__":
    main()
