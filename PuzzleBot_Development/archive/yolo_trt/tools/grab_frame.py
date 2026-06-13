#!/usr/bin/env python3
"""Grab one frame from a CompressedImage topic and save it. Headless test util.

Usage: python3 grab_frame.py <out.jpg> [topic]
"""
import sys
import time

import numpy as np
import cv2
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/grab.jpg"
    topic = sys.argv[2] if len(sys.argv) > 2 else "/yolo/image/compressed"
    rclpy.init()
    node = rclpy.create_node("grab_frame")
    got = {}

    def cb(m):
        if "img" not in got:
            got["img"] = cv2.imdecode(np.frombuffer(m.data, np.uint8),
                                      cv2.IMREAD_COLOR)

    node.create_subscription(CompressedImage, topic, cb, qos_profile_sensor_data)
    t0 = time.time()
    while "img" not in got and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.2)
    if "img" in got:
        cv2.imwrite(out, got["img"])
        print("saved", out, got["img"].shape)
    else:
        print("NO FRAME on", topic)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
