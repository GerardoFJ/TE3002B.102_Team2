#!/usr/bin/env python3
"""Sanity-check a calibrated ground homography by overlaying a metric grid.

Loads H_img2ground from a YAML written by calibrate_ground_homography.py, grabs
a live frame (or reads --image), and draws ground-plane grid lines (constant X =
forward distance, constant Y = lateral) back onto the camera image. If the
calibration is good the lines hug real features on the floor (line spacing,
tile edges) at the right distances.

Usage:
  ros2 run lane_pilot check_homography.py --homography ground_homography.yaml
  ros2 run lane_pilot check_homography.py --homography H.yaml --image frame.jpg
"""
import argparse
import sys

import numpy as np
import cv2


def grab_one(topic, timeout_s=10.0):
    import time
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = rclpy.create_node("checkH_grab")
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
    rclpy.shutdown()
    return got.get("img")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--homography", default="ground_homography.yaml")
    ap.add_argument("--image", default=None)
    ap.add_argument("--topic", default="/camera/image_rect/compressed")
    ap.add_argument("--out", default="/tmp/check_homography.png")
    ap.add_argument("--x-max", type=float, default=0.7)
    ap.add_argument("--y-max", type=float, default=0.3)
    args = ap.parse_args()

    fs = cv2.FileStorage(args.homography, cv2.FILE_STORAGE_READ)
    H = fs.getNode("H_img2ground").mat()
    fs.release()
    if H is None:
        print("could not read H_img2ground from", args.homography)
        sys.exit(1)
    Hi = np.linalg.inv(H)

    if args.image:
        img = cv2.imread(args.image)
    else:
        print("grabbing a frame from", args.topic, "...")
        img = grab_one(args.topic)
    if img is None:
        print("no input image")
        sys.exit(1)

    def g2i(X, Y):
        p = cv2.perspectiveTransform(
            np.array([[[X, Y]]], dtype=np.float64), Hi).reshape(2)
        return int(round(p[0])), int(round(p[1]))

    # constant-X lines (every 0.1 m forward)
    xs = np.arange(0.1, args.x_max + 1e-6, 0.1)
    for X in xs:
        pts = [g2i(X, Y) for Y in np.linspace(-args.y_max, args.y_max, 40)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "%.1fm" % X, g2i(X, args.y_max),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)

    # constant-Y lines (every 0.1 m lateral); Y=0 highlighted
    ys = np.arange(-args.y_max, args.y_max + 1e-6, 0.1)
    for Y in ys:
        pts = [g2i(X, Y) for X in np.linspace(0.1, args.x_max, 40)]
        is_center = abs(Y) < 1e-6
        color = (0, 255, 255) if is_center else (200, 120, 0)
        thick = 2 if is_center else 1
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, color, thick, cv2.LINE_AA)

    cv2.imwrite(args.out, img)
    print("saved", args.out, "  (green=constant forward distance, "
          "yellow=Y=0 centerline)")


if __name__ == "__main__":
    main()
