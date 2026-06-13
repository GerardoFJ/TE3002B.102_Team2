#!/usr/bin/env python3
"""Record camera frames for the lane-segmentation dataset.

Subscribes to the compressed camera topic and saves a frame every `--interval`
seconds (deduped against near-identical frames) into `--out`. Run it while you
PUSH the robot slowly around the whole track so you capture variety:
  - straights AND curves (especially the tight ones)
  - every intersection / stop-line, from a few approach angles
  - different lighting (move the robot to brighter/darker spots, turn lights
    on/off if you can) and a few reflective/shiny spots
  - a few off-center / rotated views (so the model sees the line at the edge)

Aim for ~600-1500 frames total. More variety > more frames.

Run on the Jetson (camera is local) with the env sourced:
  python3 record_frames.py --out ~/lane_dataset --interval 0.3 --max 1500
Then pull the folder to your computer for labeling + training.
"""
import argparse
import os
import time

import numpy as np
import cv2
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/lane_dataset"))
    ap.add_argument("--topic", default="/camera/image_rect/compressed")
    ap.add_argument("--interval", type=float, default=0.3, help="seconds between saves")
    ap.add_argument("--max", type=int, default=1500, help="stop after this many frames")
    ap.add_argument("--dedup", type=float, default=6.0,
                    help="skip a frame if mean abs diff vs last saved < this (0=off)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # continue numbering if the folder already has frames
    existing = [f for f in os.listdir(args.out) if f.startswith("f") and f.endswith(".jpg")]
    start_idx = (max(int(f[1:6]) for f in existing) + 1) if existing else 0

    rclpy.init()
    node = rclpy.create_node("record_frames")
    state = {"img": None, "saved": start_idx, "last_save": 0.0, "last_small": None}

    def cb(m):
        img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        now = time.time()
        if now - state["last_save"] < args.interval:
            return
        small = cv2.resize(img, (80, 45))
        if args.dedup > 0 and state["last_small"] is not None:
            if np.mean(np.abs(small.astype(int) - state["last_small"].astype(int))) < args.dedup:
                return                                   # too similar, skip
        path = os.path.join(args.out, "f%05d.jpg" % state["saved"])
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        state["saved"] += 1
        state["last_save"] = now
        state["last_small"] = small
        if state["saved"] % 25 == 0:
            print("saved %d frames -> %s" % (state["saved"], args.out), flush=True)

    node.create_subscription(CompressedImage, args.topic, cb, qos_profile_sensor_data)
    print("recording from %s every %.2fs to %s (Ctrl+C to stop, max %d)"
          % (args.topic, args.interval, args.out, args.max))
    try:
        while rclpy.ok() and state["saved"] < args.max:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    print("DONE: %d frames in %s" % (state["saved"], args.out))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
