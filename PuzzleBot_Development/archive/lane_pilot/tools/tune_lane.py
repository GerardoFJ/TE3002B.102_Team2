#!/usr/bin/env python3
"""Interactive trackbar tuner for the lane_pilot middle-line detector.

Warps the camera image to the metric bird's-eye with the calibrated homography
and runs the SAME detection pipeline as lane_detector.cpp, with every parameter
on a slider so you can tune it visually and live. The tuned values transfer 1:1
to the node (config/lane_pilot.yaml or `ros2 param set /lane_ipm_node ...`).

Needs a DISPLAY (X11). Run in your ROS 2 docker:

  # live from the robot's camera (same ROS graph / RMW as the robot):
  python3 tune_lane.py --homography ground_homography.yaml

  # or tune on a saved frame (no ROS needed):
  python3 tune_lane.py --homography ground_homography.yaml --image frame.jpg

Copy ground_homography.yaml off the robot first, e.g.:
  scp puzzlebot@192.168.0.180:/home/puzzlebot/ground_homography.yaml .

Windows: 'bird' = bird's-eye + overlay (sliders here), 'mask' = binary mask.
Keys:  p = print current params (paste into config / param set)
       s = save current bird overlay to /tmp/tune_bird.png
       m = toggle mask window
       q / ESC = quit
"""
import argparse
import sys

import numpy as np
import cv2


# ----- bird's-eye geometry (MUST match the node's bird_* params) -----
class Geom:
    def __init__(self, x_min, x_max, y_half, mpp):
        self.x_min, self.x_max, self.y_half, self.mpp = x_min, x_max, y_half, mpp
        self.W = round(2 * y_half / mpp)
        self.H = round((x_max - x_min) / mpp)

    def warp_matrix(self, H_img2ground):
        dx = self.x_max - self.x_min
        cY = (self.W - 1) / (2 * self.y_half)
        cX = (self.H - 1) / dx
        A = np.array([[0, -cY, (self.W - 1) / 2.0],
                      [-cX, 0, self.x_max * cX],
                      [0, 0, 1]], float)
        return A @ H_img2ground

    def bird_to_ground(self, col, row):
        X = self.x_max - row * (self.x_max - self.x_min) / (self.H - 1)
        Y = self.y_half - col * (2 * self.y_half) / (self.W - 1)
        return X, Y

    def ground_to_bird(self, X, Y):
        col = (self.y_half - Y) * (self.W - 1) / (2 * self.y_half)
        row = (self.x_max - X) * (self.H - 1) / (self.x_max - self.x_min)
        return col, row


def row_clusters(rowmask, min_px, max_px):
    out = []
    c, n = 0, len(rowmask)
    while c < n:
        if rowmask[c] == 0:
            c += 1
            continue
        c0 = c
        while c < n and rowmask[c] != 0:
            c += 1
        w = c - c0
        if min_px <= w <= max_px:
            out.append((c0 + c - 1) / 2.0)
    return out


def detect(bird, g, P):
    """Replicates lane_detector.cpp exactly. Returns (selected, clusters_by_row,
    mask, roi, conf, valid)."""
    gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
    if P["blur"] >= 3:
        k = P["blur"] | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    # EXACTLY matches lane_detector.cpp: invert=true -> dark line (gray<=thr) with
    # the no-data warp border excluded; invert=false -> bright (gray>thr).
    if P["invert"]:
        mask = (gray <= P["thr"]).astype(np.uint8) * 255
        mask[gray == 0] = 0                                    # exclude no-data
    else:
        mask = (gray > P["thr"]).astype(np.uint8) * 255
    # Optional morphology to clean noise (matches lane_detector.cpp open/close).
    if P.get("open_px", 0) >= 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["open_px"], P["open_px"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if P.get("close_px", 0) >= 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["close_px"], P["close_px"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    W, H = g.W, g.H
    mpp = g.mpp
    max_jump_px = (P["max_jump_cm"] / 100.0) / mpp

    # ROI bounds
    _, r_a = g.ground_to_bird(P["roi_xmax"] / 100.0, 0.0)
    _, r_b = g.ground_to_bird(P["roi_xmin"] / 100.0, 0.0)
    roi_row_top = int(np.clip(np.floor(min(r_a, r_b)), 0, H - 1))
    roi_row_bot = int(np.clip(np.ceil(max(r_a, r_b)), 0, H - 1))
    col_left, _ = g.ground_to_bird(0.0, P["roi_yhalf"] / 100.0)
    col_right, _ = g.ground_to_bird(0.0, -P["roi_yhalf"] / 100.0)
    if col_left > col_right:
        col_left, col_right = col_right, col_left

    selected, all_clusters = [], []
    cur = -1.0
    rows_with = 0
    for r in range(H - 1, -1, -max(1, P["row_step"])):
        if r < roi_row_top or r > roi_row_bot:
            continue
        cl = [c for c in row_clusters(mask[r], P["min_px"], P["max_px"])
              if col_left <= c <= col_right]
        if not cl:
            continue
        rows_with += 1
        for c in cl:
            all_clusters.append((c, r))
        if cur < 0:
            pick = min(cl, key=lambda c: abs(c - (W - 1) / 2.0))
            if abs(pick - (W - 1) / 2.0) > max_jump_px:
                continue
        else:
            pick = min(cl, key=lambda c: abs(c - cur))
            if abs(pick - cur) > max_jump_px:
                continue
        cur = pick
        selected.append((pick, r))

    conf = (len(selected) / rows_with) if rows_with else 0.0
    valid = len(selected) >= P["min_pts"]
    roi = (roi_row_top, roi_row_bot, col_left, col_right)
    return selected, all_clusters, mask, roi, conf, valid


def grab_loop(topic):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    rclpy.init()
    node = rclpy.create_node("tune_lane")
    state = {}
    node.create_subscription(
        CompressedImage, topic,
        lambda m: state.__setitem__(
            "img", cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)),
        qos_profile_sensor_data)
    return rclpy, node, state


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--homography", required=True)
    ap.add_argument("--topic", default="/camera/image_rect/compressed")
    ap.add_argument("--image", default=None, help="tune on a static frame instead of live")
    ap.add_argument("--x-min", type=float, default=0.05)
    ap.add_argument("--x-max", type=float, default=0.70)
    ap.add_argument("--y-half", type=float, default=0.35)
    ap.add_argument("--mpp", type=float, default=0.0025)
    ap.add_argument("--scale", type=float, default=2.5, help="display upscale")
    args = ap.parse_args()

    fs = cv2.FileStorage(args.homography, cv2.FILE_STORAGE_READ)
    Hmat = fs.getNode("H_img2ground").mat()
    fs.release()
    if Hmat is None:
        print("could not read H_img2ground from", args.homography)
        sys.exit(1)

    g = Geom(args.x_min, args.x_max, args.y_half, args.mpp)
    Wmap = g.warp_matrix(Hmat)

    static_img = cv2.imread(args.image) if args.image else None
    rclpy = node = state = None
    if static_img is None:
        rclpy, node, state = grab_loop(args.topic)

    win = "bird"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # (name, default, max)
    bars = [
        ("thr", 110, 255), ("invert", 1, 1), ("blur", 3, 9),
        ("min_px", 6, 40), ("max_px", 60, 150), ("row_step", 2, 6),
        ("max_jump_cm", 6, 20), ("min_pts", 6, 30),
        ("roi_xmin", 5, 40), ("roi_xmax", 55, 70), ("roi_yhalf", 25, 35),
    ]
    for n, d, m in bars:
        cv2.createTrackbar(n, win, d, m, lambda x: None)

    show_mask = True
    print("tuning... move sliders; 'p' prints params, 'q' quits.")
    while True:
        if static_img is not None:
            raw = static_img.copy()
        else:
            rclpy.spin_once(node, timeout_sec=0.02)
            if "img" not in state:
                continue
            raw = state["img"]
        if raw is None:
            continue

        P = {n: cv2.getTrackbarPos(n, win) for n, _, _ in bars}
        P["invert"] = bool(P["invert"])
        bird = cv2.warpPerspective(raw, Wmap, (g.W, g.H), flags=cv2.INTER_LINEAR)
        selected, clusters, mask, roi, conf, valid = detect(bird, g, P)
        rt, rb, cl, cr = roi

        viz = bird.copy()
        cv2.line(viz, (g.W // 2, 0), (g.W // 2, g.H - 1), (120, 120, 120), 1)
        cv2.rectangle(viz, (int(cl), int(rt)), (int(cr), int(rb)), (0, 140, 255), 1)
        for (c, r) in clusters:
            cv2.circle(viz, (int(c), r), 1, (0, 165, 255), -1)
        for (c, r) in selected:
            cv2.circle(viz, (int(c), r), 1, (0, 255, 0), -1)
        txt = "%s n=%d conf=%.2f" % ("VALID" if valid else "weak", len(selected), conf)
        cv2.putText(viz, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(viz, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        disp = cv2.resize(viz, None, fx=args.scale, fy=args.scale,
                          interpolation=cv2.INTER_NEAREST)
        cv2.imshow(win, disp)
        if show_mask:
            cv2.imshow("mask", cv2.resize(mask, None, fx=args.scale, fy=args.scale,
                                          interpolation=cv2.INTER_NEAREST))

        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord('m'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("mask")
        if k == ord('s'):
            cv2.imwrite("/tmp/tune_bird.png", disp)
            print("saved /tmp/tune_bird.png")
        if k == ord('p'):
            print("\n# --- paste into lane_pilot.yaml (lane_ipm_node.ros__parameters) ---")
            print("    black_threshold: %d" % P["thr"])
            print("    invert_threshold: %s" % ("true" if P["invert"] else "false"))
            print("    blur_ksize: %d" % P["blur"])
            print("    min_cluster_px: %d" % P["min_px"])
            print("    max_cluster_px: %d" % P["max_px"])
            print("    row_step: %d" % P["row_step"])
            print("    max_jump_m: %.2f" % (P["max_jump_cm"] / 100.0))
            print("    min_points: %d" % P["min_pts"])
            print("    roi_x_min: %.2f" % (P["roi_xmin"] / 100.0))
            print("    roi_x_max: %.2f" % (P["roi_xmax"] / 100.0))
            print("    roi_y_half: %.2f" % (P["roi_yhalf"] / 100.0))
            print("# --- or live: ros2 param set /lane_ipm_node black_threshold %d  (etc.) ---\n"
                  % P["thr"])

    cv2.destroyAllWindows()
    if node is not None:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
