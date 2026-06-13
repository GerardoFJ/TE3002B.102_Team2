#!/usr/bin/env python3
"""Calibrate the camera->ground homography from a checkerboard on the floor.

Lay a printed checkerboard FLAT on the ground in front of the robot, roughly
centered on the robot's forward axis and with its grid axes aligned to the robot
(one grid axis pointing forward, the other across). Then run this tool: it
detects the inner corners, assigns each a real (X, Y) position in base_link
meters, and fits the homography that lane_ipm_node uses for inverse-perspective
mapping.

Frame (REP-103 base_link): X forward from the wheel-axle origin, Y +left.

Required measurements:
  --pattern  COLSxROWS  inner corners: COLS across (lateral), ROWS forward
  --square   meters     checkerboard square size
  --near-x   meters     distance from base_link origin to the NEAREST row of
                        inner corners (the row closest to the robot)

Input frame: either --image <file>, or grabbed live from --topic.

Example:
  ros2 run lane_pilot calibrate_ground_homography.py \
      --pattern 9x6 --square 0.025 --near-x 0.12 \
      --out ground_homography.yaml --debug-out /tmp/calib_debug.png
"""
import argparse
import sys

import numpy as np
import cv2


def grab_one(topic, timeout_s=10.0):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = rclpy.create_node("calib_grab")
    got = {}

    def cb(m):
        if "img" not in got:
            got["img"] = cv2.imdecode(np.frombuffer(m.data, np.uint8),
                                      cv2.IMREAD_COLOR)

    import time
    node.create_subscription(CompressedImage, topic, cb, qos_profile_sensor_data)
    t0 = time.time()
    while "img" not in got and time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    return got.get("img")


def _try_sb(gray, w, h):
    """findChessboardCornersSB — better, but some OpenCV builds THROW (flann
    assertion) instead of returning False, so guard it."""
    if not hasattr(cv2, "findChessboardCornersSB"):
        return None
    flags = (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY) \
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE") else 0
    try:
        ok, corners = cv2.findChessboardCornersSB(gray, (w, h), flags=flags)
    except cv2.error:
        return None
    return corners.reshape(h, w, 2) if ok else None


def _try_classic(gray, w, h):
    try:
        ok, corners = cv2.findChessboardCorners(
            gray, (w, h),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    except cv2.error:
        return None
    if not ok:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (7, 7), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return corners.reshape(h, w, 2)


def find_corners(gray, a, b):
    """Detect the board trying patternSize (a,b) then (b,a) (the board may be
    rotated 90 deg in the image), SB first then the classic detector. Returns
    (grid[h][w][2], h, w) or None; h = corners along image-vertical, w along
    image-horizontal."""
    cands = [(a, b), (b, a)] if a != b else [(a, b)]
    for finder in (_try_sb, _try_classic):
        for (w, h) in cands:
            g = finder(gray, w, h)
            if g is not None:
                return g, h, w
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", required=True,
                    help="inner corners COLSxROWS, e.g. 9x6 (COLS across, ROWS forward)")
    ap.add_argument("--square", type=float, required=True, help="square size (m)")
    ap.add_argument("--near-x", type=float, required=True,
                    help="distance base_link origin -> nearest corner row (m)")
    ap.add_argument("--image", default=None, help="input image file (else grab live)")
    ap.add_argument("--topic", default="/camera/image_rect/compressed")
    ap.add_argument("--out", default="ground_homography.yaml")
    ap.add_argument("--debug-out", default="/tmp/calib_debug.png")
    args = ap.parse_args()

    pat_a, pat_b = (int(x) for x in args.pattern.lower().split("x"))

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print("could not read", args.image)
            sys.exit(1)
    else:
        print("grabbing a frame from", args.topic, "...")
        img = grab_one(args.topic)
        if img is None:
            print("NO FRAME on", args.topic)
            sys.exit(1)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = find_corners(gray, pat_a, pat_b)
    if found is None:
        print("checkerboard %dx%d NOT found. Check --pattern, lighting, and that "
              "the whole board is visible." % (pat_a, pat_b))
        cv2.imwrite(args.debug_out, img)
        sys.exit(2)
    grid, rows, cols = found   # grid[r][c] -> (u, v); rows along image-y, cols along image-x

    # --- auto-orient so r=0 is NEAR (robot side) and c=0 is LEFT (+Y) ---
    # Forward = away from robot = higher in image = smaller v.
    if grid[0, :, 1].mean() < grid[rows - 1, :, 1].mean():
        grid = grid[::-1, :, :]          # flip so row 0 has the largest v (nearest)
    # Left of robot (+Y) = left in image = smaller u.
    if grid[:, 0, 0].mean() > grid[:, cols - 1, 0].mean():
        grid = grid[:, ::-1, :]          # flip so col 0 is leftmost

    # --- assign ground coordinates (base_link meters) ---
    img_pts = []
    gnd_pts = []
    for r in range(rows):
        for c in range(cols):
            X = args.near_x + r * args.square
            Y = ((cols - 1) / 2.0 - c) * args.square
            img_pts.append(grid[r, c])
            gnd_pts.append([X, Y])
    img_pts = np.array(img_pts, dtype=np.float64)
    gnd_pts = np.array(gnd_pts, dtype=np.float64)

    H, _ = cv2.findHomography(img_pts, gnd_pts, method=0)
    if H is None:
        print("findHomography failed")
        sys.exit(3)

    # --- reprojection error (image -> ground), reported in mm ---
    proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - gnd_pts, axis=1)
    rms_mm = float(np.sqrt((err ** 2).mean()) * 1000.0)
    max_mm = float(err.max() * 1000.0)

    fs = cv2.FileStorage(args.out, cv2.FILE_STORAGE_WRITE)
    fs.write("H_img2ground", H)
    fs.write("image_width", w)
    fs.write("image_height", h)
    fs.write("pattern_cols", cols)
    fs.write("pattern_rows", rows)
    fs.write("square_size", args.square)
    fs.write("near_x", args.near_x)
    fs.write("reproj_rms_mm", rms_mm)
    fs.release()

    # --- debug overlay: corners colored near(green)->far(red), Y=0 axis ---
    dbg = img.copy()
    for r in range(rows):
        for c in range(cols):
            u, v = grid[r, c]
            f = r / max(1, rows - 1)
            color = (0, int(255 * (1 - f)), int(255 * f))
            cv2.circle(dbg, (int(u), int(v)), 4, color, -1)
    # draw the ground axes (X forward along Y=0, and the near row) back into image
    Hi = np.linalg.inv(H)
    def g2i(X, Y):
        p = cv2.perspectiveTransform(
            np.array([[[X, Y]]], dtype=np.float64), Hi).reshape(2)
        return int(p[0]), int(p[1])
    far = args.near_x + (rows - 1) * args.square
    cv2.line(dbg, g2i(args.near_x, 0.0), g2i(far, 0.0), (255, 255, 0), 2)
    cv2.putText(dbg, "Y=0 (forward)", g2i(far, 0.0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(args.debug_out, dbg)

    print("OK  pattern %dx%d  square %.3fm  near_x %.3fm" %
          (cols, rows, args.square, args.near_x))
    print("    reprojection RMS = %.1f mm   max = %.1f mm" % (rms_mm, max_mm))
    print("    saved homography -> %s" % args.out)
    print("    saved debug overlay -> %s   (green=near, red=far; check the "
          "Y=0 line points straight ahead)" % args.debug_out)
    if rms_mm > 15.0:
        print("    WARNING: RMS > 15 mm. Re-check --near-x, --square, board "
              "flatness, and that the board axes are aligned with the robot.")


if __name__ == "__main__":
    main()
