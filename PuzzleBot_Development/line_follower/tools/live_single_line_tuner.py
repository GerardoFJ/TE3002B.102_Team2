#!/usr/bin/env python3
"""LIVE single-black-line tuner.

Same detection pipeline and trackbars as single_line_tuner.py, but the frames
come from the robot's camera topic in real time instead of a bag/MP4. Point
the camera at the track and drag sliders until the line is detected cleanly.

Pipeline (mirrors follow_single_line.cpp EXACTLY — keep them in sync):
    BGR -> downscale -> ROI crop -> grayscale -> gaussian blur
        -> (optional median) -> THRESH_BINARY_INV (black_threshold)
        -> MORPH_OPEN (small) -> MORPH_CLOSE (tall-thin)
        -> connectedComponents -> pick the single largest valid blob
        -> centroid x -> error vs ROI center

Run on your computer, in the ros2 docker (talks to the robot over DDS):
  docker run --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    --user $(id -u):$(id -g) \
    -v /home/gerardo/Dockers/Ros2/Workspace:/workspace ros2-generic:latest \
    bash -lc 'source /opt/ros/humble/setup.bash && \
      export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
      python3 /workspace/TE3002B.102_Team2/PuzzleBot_Development/line_follower/tools/live_single_line_tuner.py'

Keys:
  f             freeze / unfreeze the current frame (tune on a still image)
  s             print params + save tuned_single_params.yaml next to this script
  q / ESC       quit (also saves)
"""
import argparse
import os

import numpy as np
import cv2

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

# Defaults = the user's current calibration (config/follow_single_line.yaml).
DEFAULTS = {
    "detect_scale": 0.600,
    "roi_top_frac": 0.700,
    "roi_bottom_frac": 1.000,
    "roi_left_frac": 0.300,
    "roi_right_frac": 0.630,
    "gaussian_ksize": 3,
    "median_ksize": 0,
    "black_threshold": 170,
    "open_kernel": 8,
    "close_kernel_h": 11,
    "min_line_area": 450,
    "max_line_width_frac": 0.80,
    "min_line_height_frac": 0.72,
}

WIN = "live tuner"
PARAM_ORDER = list(DEFAULTS.keys())


def odd_at_least(n, lo):
    n = max(lo, n)
    return n if n % 2 == 1 else n + 1


def make_trackbars():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 800)
    d = DEFAULTS
    # fractions are stored as percent on the bars
    cv2.createTrackbar("detect_scale %", WIN, int(d["detect_scale"] * 100), 100, lambda v: None)
    cv2.createTrackbar("roi_top %", WIN, int(d["roi_top_frac"] * 100), 99, lambda v: None)
    cv2.createTrackbar("roi_bottom %", WIN, int(d["roi_bottom_frac"] * 100), 100, lambda v: None)
    cv2.createTrackbar("roi_left %", WIN, int(d["roi_left_frac"] * 100), 99, lambda v: None)
    cv2.createTrackbar("roi_right %", WIN, int(d["roi_right_frac"] * 100), 100, lambda v: None)
    cv2.createTrackbar("gaussian_ksize", WIN, d["gaussian_ksize"], 15, lambda v: None)
    cv2.createTrackbar("median_ksize", WIN, d["median_ksize"], 15, lambda v: None)
    cv2.createTrackbar("black_threshold", WIN, d["black_threshold"], 255, lambda v: None)
    cv2.createTrackbar("open_kernel", WIN, d["open_kernel"], 25, lambda v: None)
    cv2.createTrackbar("close_kernel_h", WIN, d["close_kernel_h"], 40, lambda v: None)
    cv2.createTrackbar("min_line_area", WIN, d["min_line_area"], 3000, lambda v: None)
    cv2.createTrackbar("max_line_w %", WIN, int(d["max_line_width_frac"] * 100), 100, lambda v: None)
    cv2.createTrackbar("min_line_h %", WIN, int(d["min_line_height_frac"] * 100), 100, lambda v: None)


def read_params():
    g = lambda n: cv2.getTrackbarPos(n, WIN)
    return {
        "detect_scale": max(10, g("detect_scale %")) / 100.0,
        "roi_top_frac": g("roi_top %") / 100.0,
        "roi_bottom_frac": max(g("roi_top %") + 1, g("roi_bottom %")) / 100.0,
        "roi_left_frac": g("roi_left %") / 100.0,
        "roi_right_frac": max(g("roi_left %") + 1, g("roi_right %")) / 100.0,
        "gaussian_ksize": g("gaussian_ksize"),
        "median_ksize": g("median_ksize"),
        "black_threshold": g("black_threshold"),
        "open_kernel": max(1, g("open_kernel")),
        "close_kernel_h": max(1, g("close_kernel_h")),
        "min_line_area": g("min_line_area"),
        "max_line_width_frac": g("max_line_w %") / 100.0,
        "min_line_height_frac": g("min_line_h %") / 100.0,
    }


def detect(bgr, p):
    """Returns (viz, mask_bgr, err or None). Mirrors follow_single_line.cpp."""
    if p["detect_scale"] < 0.999:
        bgr = cv2.resize(bgr, None, fx=p["detect_scale"], fy=p["detect_scale"],
                         interpolation=cv2.INTER_AREA)
    H, W = bgr.shape[:2]
    x0 = int(round(p["roi_left_frac"] * W))
    x1 = max(x0 + 1, int(round(p["roi_right_frac"] * W)))
    y0 = int(round(p["roi_top_frac"] * H))
    y1 = max(y0 + 1, int(round(p["roi_bottom_frac"] * H)))
    x0, x1 = np.clip([x0, x1], 0, W).tolist()
    y0, y1 = np.clip([y0, y1], 0, H).tolist()
    roi = bgr[y0:y1, x0:x1]
    roi_w, roi_h = x1 - x0, y1 - y0
    roi_center_x = (x0 + x1) / 2.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gk = odd_at_least(p["gaussian_ksize"], 1)
    if gk >= 3:
        gray = cv2.GaussianBlur(gray, (gk, gk), 0)
    mk = odd_at_least(p["median_ksize"], 1)
    if mk >= 3:
        gray = cv2.medianBlur(gray, mk)

    _, mask = cv2.threshold(gray, p["black_threshold"], 255, cv2.THRESH_BINARY_INV)
    ok_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (p["open_kernel"], p["open_kernel"]))
    ck_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, p["close_kernel_h"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ok_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck_k)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    max_w = int(round(p["max_line_width_frac"] * roi_w))
    min_h = int(round(p["min_line_height_frac"] * roi_h))
    best = None     # (area, cx, cy, bx, by, bw, bh)
    rejected = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        ok = area >= p["min_line_area"] and bw <= max_w and bh >= min_h
        if ok and (best is None or area > best[0]):
            best = (area, cents[i][0], cents[i][1], bx, by, bw, bh)
        elif not ok:
            rejected.append((bx, by, bw, bh, area, bw > max_w, bh < min_h))

    viz = bgr.copy()
    cv2.rectangle(viz, (x0, y0), (x1 - 1, y1 - 1), (50, 200, 50), 1)
    cv2.line(viz, (int(roi_center_x), y0), (int(roi_center_x), y1 - 1),
             (120, 120, 120), 1)
    # rejected blobs in red (with the reason), accepted in orange
    for bx, by, bw, bh, area, too_wide, too_short in rejected:
        cv2.rectangle(viz, (bx + x0, by + y0), (bx + x0 + bw, by + y0 + bh),
                      (0, 0, 255), 1)
        reason = "wide" if too_wide else ("short" if too_short else "small")
        cv2.putText(viz, "%s a%d" % (reason, area), (bx + x0, by + y0 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    err = None
    if best is not None:
        area, cx, cy, bx, by, bw, bh = best
        cx_full = cx + x0
        err = (cx_full - roi_center_x) / max(1.0, roi_w / 2.0)
        cv2.rectangle(viz, (bx + x0, by + y0), (bx + x0 + bw, by + y0 + bh),
                      (0, 200, 255), 2)
        cv2.circle(viz, (int(cx_full), int(cy) + y0), 5, (0, 200, 255), -1)
        cv2.line(viz, (int(cx_full), y0), (int(cx_full), y1 - 1), (0, 0, 255), 2)

    txt = "err=%+.2f" % err if err is not None else "NO LINE"
    cv2.putText(viz, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(viz, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    mask_bgr = np.zeros_like(bgr)
    mask_bgr[y0:y1, x0:x1] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return viz, mask_bgr, err


# Trackbar display order (top to bottom in the window) -> param key.
BAR_ORDER = [
    ("detect_scale %", "detect_scale"),
    ("roi_top %", "roi_top_frac"),
    ("roi_bottom %", "roi_bottom_frac"),
    ("roi_left %", "roi_left_frac"),
    ("roi_right %", "roi_right_frac"),
    ("gaussian_ksize", "gaussian_ksize"),
    ("median_ksize", "median_ksize"),
    ("black_threshold", "black_threshold"),
    ("open_kernel", "open_kernel"),
    ("close_kernel_h", "close_kernel_h"),
    ("min_line_area", "min_line_area"),
    ("max_line_w %", "max_line_width_frac"),
    ("min_line_h %", "min_line_height_frac"),
]


def params_panel(p, height):
    """Side panel listing slider names + current values (same order as the
    trackbars, numbered top->bottom) — the docker cv2 GUI doesn't render
    trackbar labels, so we draw them ourselves."""
    panel = np.full((max(height, 24 * (len(BAR_ORDER) + 2)), 330, 3), 30,
                    np.uint8)
    cv2.putText(panel, "sliders (top to bottom):", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for i, (bar, key) in enumerate(BAR_ORDER):
        v = p[key]
        s = ("%.2f" % v) if isinstance(v, float) else str(v)
        cv2.putText(panel, "%2d. %-16s %s" % (i + 1, key, s), (10, 48 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.putText(panel, "f=freeze  s=save  q=quit",
                (10, 48 + 24 * len(BAR_ORDER) + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return panel


def dump(p, path):
    print("\n# ---- detection (tuned live) ----")
    lines = []
    for k in PARAM_ORDER:
        v = p[k]
        s = ("%.3f" % v) if isinstance(v, float) else str(v)
        lines.append("%s: %s" % (k, s))
        print("    %s: %s" % (k, s))
    with open(path, "w") as f:
        f.write("# follow_single_line detection params (live-tuned)\n")
        f.write("\n".join(lines) + "\n")
    print("# saved -> %s" % path)
    print("# paste into config/follow_single_line.yaml and restart the node\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/camera/image_rect/compressed")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("live_single_line_tuner")
    state = {"bgr": None, "frozen": None}

    def cb(msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            state["bgr"] = img

    node.create_subscription(CompressedImage, args.topic, cb,
                             qos_profile_sensor_data)
    make_trackbars()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "tuned_single_params.yaml")
    print("waiting for %s ...  (f=freeze  s=save  q=quit)" % args.topic)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            frame = state["frozen"] if state["frozen"] is not None else state["bgr"]
            if frame is None:
                if cv2.waitKey(50) in (ord('q'), 27):
                    break
                continue
            p = read_params()
            viz, mask, _ = detect(frame, p)
            both = np.hstack([viz, mask])
            panel = params_panel(p, both.shape[0])
            if panel.shape[0] > both.shape[0]:        # pad image to panel height
                pad = np.zeros(
                    (panel.shape[0] - both.shape[0], both.shape[1], 3), np.uint8)
                both = np.vstack([both, pad])
            both = np.hstack([both, panel])
            if state["frozen"] is not None:
                cv2.putText(both, "FROZEN (f to resume)", (10, both.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                            cv2.LINE_AA)
            cv2.imshow(WIN, both)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                dump(p, out)
                break
            elif k == ord('s'):
                dump(p, out)
            elif k == ord('f'):
                state["frozen"] = None if state["frozen"] is not None \
                    else (state["bgr"].copy() if state["bgr"] is not None else None)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
