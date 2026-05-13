#!/usr/bin/env python3
"""
Print every AprilTag 36h11 id seen in a ZED compressed image stream.
Quick debugging helper — point it at whichever camera you want.
"""

from __future__ import annotations

import argparse
import time

import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-topic",
                   default="/zed2/zed_node/rgb/color/rect/image",
                   help="Base image topic; '/compressed' is appended.")
    p.add_argument("--duration", type=float, default=10.0,
                   help="How many seconds to listen before giving up.")
    args = p.parse_args()

    rclpy.init()
    node = Node("list_tag_ids")
    bridge = CvBridge()
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        cv2.aruco.DetectorParameters())
    seen: set[int] = set()

    def on_image(msg: CompressedImage) -> None:
        img = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            return
        new = {int(i) for i in ids.flatten().tolist()} - seen
        if new:
            seen.update(new)
            print(f"+ new ids: {sorted(new)}    "
                  f"all so far: {sorted(seen)}", flush=True)

    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                     history=HistoryPolicy.KEEP_LAST, depth=5)
    node.create_subscription(CompressedImage,
                             args.image_topic + "/compressed",
                             on_image, qos)

    print(f"Listening on {args.image_topic}/compressed for {args.duration:.0f}s ...",
          flush=True)
    end = time.time() + args.duration
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f"\nfinal ids seen: {sorted(seen) if seen else '(none)'}", flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
