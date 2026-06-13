#!/usr/bin/env python3
"""Full debug dashboard for the lane_pilot stack — run on your computer.

Subscribes to the live camera + odometry and replicates the ENTIRE pipeline in
Python (warp -> threshold mask -> middle-line detection -> odom memory ->
polynomial centerline -> regulated pure pursuit), drawing everything in one
window so you can see exactly why a frame fails. Every detection AND control
parameter is on a slider; values transfer 1:1 to the C++ node.

READ-ONLY: it never publishes /cmd_vel, so the robot does NOT move.

Dashboard panels:
  raw camera        |  bird's-eye + detection (ROI box, clusters, middle line)
  threshold mask    |  planned trajectory (memory, fitted centerline, lookahead)

Run in your ROS 2 docker (needs a display + Cyclone DDS to see the robot):
  xhost +local:docker
  docker run --rm -it --network host -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v <WORKSPACE>:/ws -v ~/ground_homography.yaml:/root/ground_homography.yaml:ro \
    ros2-generic:latest bash -lc '
      source /opt/ros/humble/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
      python3 /ws/lane_pilot/tools/lane_debug_ui.py --homography /root/ground_homography.yaml'

Keys: p = print all params   s = save dashboard to /tmp/lane_debug.png   q = quit
"""
import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_lane as T            # Geom, row_clusters, detect (the EXACT detector)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------- control (port of lane_pilot_node.cpp) ---------------
class Pilot:
    def __init__(self):
        self.mem = deque()       # (ox, oy, t) in odom frame

    def add(self, pts_base, pose, t, mem_time, mem_max):
        x, y, th = pose
        ct, st = np.cos(th), np.sin(th)
        for (X, Y) in pts_base:
            self.mem.append((x + X * ct - Y * st, y + X * st + Y * ct, t))
        while self.mem and self.mem[0][2] < t - mem_time:
            self.mem.popleft()
        while len(self.mem) > mem_max:
            self.mem.popleft()

    def control(self, pose, C):
        x, y, th = pose
        ct, st = np.cos(th), np.sin(th)
        xs, ys = [], []
        for (ox, oy, _) in self.mem:
            dx, dy = ox - x, oy - y
            X = dx * ct + dy * st
            Y = -dx * st + dy * ct
            if X < C["x_keep_min"] or X > C["x_keep_max"] or abs(Y) > C["y_keep"]:
                continue
            xs.append(X)
            ys.append(Y)
        out = {"xs": xs, "ys": ys, "have_path": False, "coeffs": None,
               "max_x": 0.0, "v": 0.0, "omega": 0.0, "ld": 0.0, "kappa": 0.0,
               "xL": 0.0, "yL": 0.0, "status": "NO-PATH"}
        if not xs:
            return out
        max_x, min_x = max(xs), min(xs)
        span = max_x - min_x
        if len(xs) < C["min_pts_fit"] or span < C["min_span"] or max_x <= 0.02:
            return out

        deg = C["max_degree"]
        if len(xs) < deg + 2:
            deg = len(xs) - 1
        if span < 0.10:
            deg = min(deg, 1)
        elif span < 0.25:
            deg = min(deg, 2)
        deg = int(np.clip(deg, 1, 3))
        coeffs = np.polyfit(xs, ys, deg)            # highest-order first
        d1 = np.polyder(coeffs)
        d2 = np.polyder(coeffs, 2)
        xe = float(np.clip(C["curv_eval_x"], 0.0, max_x))
        dp = np.polyval(d1, xe)
        ddp = np.polyval(d2, xe)
        kappa = abs(ddp) / (1.0 + dp * dp) ** 1.5

        v = C["v_max"] * (1.0 - C["slow_gain"] * min(1.0, kappa / max(1e-6, C["kref"])))
        if max_x < C["short_path_x"]:
            v *= np.clip(max_x / max(1e-6, C["short_path_x"]), 0.3, 1.0)
        v = float(np.clip(v, C["v_min"], C["v_max"]))

        ld = float(np.clip(C["ld_base"] + C["ld_k"] * v, C["ld_min"], C["ld_max"]))
        ld = min(ld, max(C["ld_min"], max_x))

        xL = yL = chord = 0.0
        xq = 0.0
        while xq <= max_x + 1e-6:
            yq = np.polyval(coeffs, xq)
            chord = np.hypot(xq, yq)
            xL, yL = xq, yq
            if chord >= ld:
                break
            xq += 0.01
        omega = 0.0
        if chord > 1e-3:
            omega = float(np.clip(v * 2.0 * yL / (chord * chord),
                                  -C["max_ang"], C["max_ang"]))
        out.update(have_path=True, coeffs=coeffs, max_x=max_x, v=v, omega=omega,
                   ld=ld, kappa=kappa, xL=xL, yL=yL, status="TRACK")
        return out


# ----------------------- rendering -------------------------------------------
def label(img, text, color=(255, 255, 255)):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bar = np.full((22, img.shape[1], 3), 45, np.uint8)
    cv2.putText(bar, text, (5, 16), FONT, 0.5, color, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit_h(img, h):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), h),
                      interpolation=cv2.INTER_NEAREST)


def pad_w(img, w):
    if img.shape[1] >= w:
        return img
    return cv2.copyMakeBorder(img, 0, 0, 0, w - img.shape[1],
                              cv2.BORDER_CONSTANT, value=(20, 20, 20))


def render_bird(bird, g, sel, clusters, roi, conf, valid):
    viz = bird.copy()
    rt, rb, cl, cr = roi
    cv2.line(viz, (g.W // 2, 0), (g.W // 2, g.H - 1), (120, 120, 120), 1)
    cv2.rectangle(viz, (int(cl), int(rt)), (int(cr), int(rb)), (0, 140, 255), 1)
    for (c, r) in clusters:
        cv2.circle(viz, (int(c), r), 1, (0, 165, 255), -1)
    for (c, r) in sel:
        cv2.circle(viz, (int(c), r), 1, (0, 255, 0), -1)
    t = "%s n=%d conf=%.2f" % ("VALID" if valid else "weak", len(sel), conf)
    cv2.putText(viz, t, (6, 16), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(viz, t, (6, 16), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return viz


def render_traj(res, perc_pts):
    X_FWD, X_BACK, Y_MAX, ppm = 0.80, -0.25, 0.45, 400.0
    Wd, Hd = int(2 * Y_MAX * ppm), int((X_FWD - X_BACK) * ppm)
    img = np.full((Hd, Wd, 3), 35, np.uint8)

    def px(X, Y):
        return (int((Y_MAX - Y) * ppm), int((X_FWD - X) * ppm))

    xm = X_BACK
    while xm <= X_FWD + 1e-6:
        r = px(xm, 0.0)[1]
        cv2.line(img, (0, r), (Wd - 1, r), (60, 60, 60), 1)
        cv2.putText(img, "%.1f" % xm, (2, r - 2), FONT, 0.3, (120, 120, 120), 1, cv2.LINE_AA)
        xm += 0.1
    cv2.line(img, px(X_FWD, 0), px(X_BACK, 0), (90, 90, 90), 1)
    cv2.line(img, px(0.21, -Y_MAX), px(0.21, Y_MAX), (40, 40, 110), 1)  # blind-spot edge

    for X, Y in zip(res["xs"], res["ys"]):
        cv2.circle(img, px(X, Y), 1, (150, 150, 150), -1)         # memory gray
    for (X, Y) in perc_pts:
        cv2.circle(img, px(X, Y), 2, (0, 220, 0), -1)             # perceived green
    if res["coeffs"] is not None:
        prev = None
        xq = 0.0
        while xq <= res["max_x"] + 1e-6:
            p = px(xq, float(np.polyval(res["coeffs"], xq)))
            if prev is not None:
                cv2.line(img, prev, p, (255, 220, 0), 2)          # centerline cyan
            prev = p
            xq += 0.02
        cv2.line(img, px(0, 0), px(res["xL"], res["yL"]), (200, 0, 200), 1)
        cv2.circle(img, px(res["xL"], res["yL"]), 6, (255, 0, 255), 2)  # lookahead
    o = px(0, 0)
    cv2.polylines(img, [np.array([[o[0], o[1] - 10], [o[0] - 7, o[1] + 7],
                                  [o[0] + 7, o[1] + 7]])], True, (0, 165, 255), 2)
    hud = "%s v=%.2f w=%+.2f Ld=%.2f k=%.1f n=%d" % (
        res["status"], res["v"], res["omega"], res["ld"], res["kappa"], len(res["xs"]))
    cv2.putText(img, hud, (6, 16), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, hud, (6, 16), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ----------------------- sliders ---------------------------------------------
DET_BARS = [  # (name, default, max)  -- defaults = current tuned config
    ("thr", 125, 255), ("invert", 0, 1), ("blur", 4, 9), ("min_px", 2, 40),
    ("max_px", 16, 150), ("row_step", 2, 6), ("max_jump_cm", 6, 20),
    ("min_pts", 7, 30), ("roi_xmin", 17, 40), ("roi_xmax", 30, 70), ("roi_yhalf", 16, 35),
]
CTL_BARS = [
    ("v_max_x100", 12, 30), ("ld_base_cm", 30, 50), ("ld_min_cm", 25, 45),
    ("ld_max_cm", 45, 60), ("kref_x10", 40, 80), ("slow_x100", 70, 100),
    ("mem_time_x10", 30, 50), ("max_ang_x10", 18, 30), ("max_degree", 2, 3),
    ("curv_eval_cm", 30, 50),
]


def read_params(win):
    g = lambda n: cv2.getTrackbarPos(n, win)
    P = {"thr": g("thr"), "invert": bool(g("invert")), "blur": g("blur"),
         "min_px": g("min_px"), "max_px": g("max_px"), "row_step": g("row_step"),
         "max_jump_cm": g("max_jump_cm"), "min_pts": g("min_pts"),
         "roi_xmin": g("roi_xmin"), "roi_xmax": g("roi_xmax"), "roi_yhalf": g("roi_yhalf")}
    C = {"v_max": g("v_max_x100") / 100.0, "v_min": 0.03,
         "ld_base": g("ld_base_cm") / 100.0, "ld_k": 0.2,
         "ld_min": g("ld_min_cm") / 100.0, "ld_max": g("ld_max_cm") / 100.0,
         "kref": g("kref_x10") / 10.0, "slow_gain": g("slow_x100") / 100.0,
         "mem_time": g("mem_time_x10") / 10.0, "max_ang": g("max_ang_x10") / 10.0,
         "max_degree": g("max_degree"), "curv_eval_x": g("curv_eval_cm") / 100.0,
         "x_keep_min": -0.20, "x_keep_max": 0.80, "y_keep": 0.40,
         "min_pts_fit": 5, "min_span": 0.05, "short_path_x": 0.30, "mem_max": 1500}
    return P, C


def print_params(P, C):
    print("\n# ---- lane_ipm_node (detection) ----")
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
    print("# ---- lane_pilot_node (control) ----")
    print("    v_max: %.2f" % C["v_max"])
    print("    lookahead_base: %.2f" % C["ld_base"])
    print("    lookahead_min: %.2f" % C["ld_min"])
    print("    lookahead_max: %.2f" % C["ld_max"])
    print("    curv_kappa_ref: %.1f" % C["kref"])
    print("    curv_slow_gain: %.2f" % C["slow_gain"])
    print("    curv_eval_x: %.2f" % C["curv_eval_x"])
    print("    mem_time: %.1f" % C["mem_time"])
    print("    max_angular: %.1f" % C["max_ang"])
    print("    max_degree: %d\n" % C["max_degree"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--homography", required=True)
    ap.add_argument("--topic", default="/camera/image_rect/compressed")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--image", default=None, help="static frame (no ROS/memory)")
    ap.add_argument("--x-min", type=float, default=0.05)
    ap.add_argument("--x-max", type=float, default=0.70)
    ap.add_argument("--y-half", type=float, default=0.35)
    ap.add_argument("--mpp", type=float, default=0.0025)
    args = ap.parse_args()

    fs = cv2.FileStorage(args.homography, cv2.FILE_STORAGE_READ)
    Hmat = fs.getNode("H_img2ground").mat()
    fs.release()
    if Hmat is None:
        print("could not read H_img2ground from", args.homography)
        sys.exit(1)
    g = T.Geom(args.x_min, args.x_max, args.y_half, args.mpp)
    Wmap = g.warp_matrix(Hmat)

    static = cv2.imread(args.image) if args.image else None
    rclpy = node = None
    odom = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    state = {}
    if static is None:
        import rclpy as _rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        from nav_msgs.msg import Odometry
        rclpy = _rclpy
        rclpy.init()
        node = rclpy.create_node("lane_debug_ui")
        node.create_subscription(
            CompressedImage, args.topic,
            lambda m: state.__setitem__(
                "img", cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)),
            qos_profile_sensor_data)

        def on_odom(m):
            q = m.pose.pose.orientation
            odom["x"] = m.pose.pose.position.x
            odom["y"] = m.pose.pose.position.y
            odom["yaw"] = np.arctan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)
        node.create_subscription(Odometry, args.odom_topic, on_odom, 10)

    win = "controls"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 360, 700)
    for n, d, m in DET_BARS + CTL_BARS:
        cv2.createTrackbar(n, win, d, m, lambda x: None)
    cv2.namedWindow("lane_debug", cv2.WINDOW_NORMAL)

    pilot = Pilot()
    print("debug UI running. p=print params, s=save, q=quit.")
    while True:
        if static is not None:
            raw = static.copy()
        else:
            rclpy.spin_once(node, timeout_sec=0.02)
            raw = state.get("img")
        if raw is None:
            continue

        P, C = read_params(win)
        bird = cv2.warpPerspective(raw, Wmap, (g.W, g.H), flags=cv2.INTER_LINEAR)
        sel, clusters, mask, roi, conf, valid = T.detect(bird, g, P)
        perc = [g.bird_to_ground(c, r) for (c, r) in sel]

        pose = (odom["x"], odom["y"], odom["yaw"])
        if static is None and valid:
            pilot.add(perc, pose, time.time(), C["mem_time"], C["mem_max"])
        res = pilot.control(pose, C) if static is None else \
            _static_control(perc, C)

        bird_viz = render_bird(bird, g, sel, clusters, roi, conf, valid)
        traj = render_traj(res, perc)
        top = pad_w(cv2.hconcat([fit_h(label(raw, "raw camera"), 300),
                                 fit_h(label(bird_viz, "bird + detection"), 300)]), 1)
        bot = cv2.hconcat([fit_h(label(mask, "threshold mask"), 300),
                           fit_h(label(traj, "planned trajectory"), 300)])
        wmax = max(top.shape[1], bot.shape[1])
        dash = cv2.vconcat([pad_w(top, wmax), pad_w(bot, wmax)])
        cv2.imshow("lane_debug", dash)

        k = cv2.waitKey(15) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord('p'):
            print_params(P, C)
        if k == ord('s'):
            cv2.imwrite("/tmp/lane_debug.png", dash)
            print("saved /tmp/lane_debug.png")

    cv2.destroyAllWindows()
    if node is not None:
        node.destroy_node()
        rclpy.shutdown()


def _static_control(perc, C):
    """No odom/memory: fit + pursuit on the current perception only."""
    p = Pilot()
    p.mem = deque((X, Y, 0.0) for (X, Y) in perc)
    return p.control((0.0, 0.0, 0.0), C)


if __name__ == "__main__":
    main()
