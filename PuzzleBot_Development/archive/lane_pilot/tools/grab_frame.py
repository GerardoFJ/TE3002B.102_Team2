#!/usr/bin/env python3
"""Grab one frame from a CompressedImage topic and save it. Headless util.

Usage: grab_frame.py <out.jpg> [topic]
Default topic: /camera/image_rect/compressed
"""
import sys
import time

import numpy as np
import cv2
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


def grab_one(topic, timeout_s=10.0):
    """Spin briefly and return the first decoded BGR frame on `topic`."""
    node = rclpy.create_node("grab_frame")
    got = {}

    def cb(m):
        if "img" not in got:
            got["img"] = cv2.imdecode(np.frombuffer(m.data, np.uint8),
                                      cv2.IMREAD_COLOR)

    node.create_subscription(CompressedImage, topic, cb, qos_profile_sensor_data)
    t0 = time.time()
    while "img" not in got and time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    return got.get("img")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/grab.jpg"
    topic = sys.argv[2] if len(sys.argv) > 2 else "/camera/image_rect/compressed"
    rclpy.init()
    img = grab_one(topic)
    rclpy.shutdown()
    if img is None:
        print("NO FRAME on", topic)
        sys.exit(1)
    cv2.imwrite(out, img)
    print("saved", out, img.shape)


if __name__ == "__main__":
    main()
