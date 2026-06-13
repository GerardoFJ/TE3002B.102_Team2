#!/usr/bin/env python3
"""PyQt5 debug dashboard for the lane_pilot stack — run on your computer.

A proper GUI version of lane_debug_ui.py: clearly LABELLED sliders (name + live
value + unit) grouped into Detection / Control, the four image panels (raw,
bird+detection, threshold mask, planned trajectory), and a numeric readout.
Replicates the entire pipeline in Python (warp -> mask -> detection -> odom
memory -> centerline fit -> pure pursuit). READ-ONLY: never publishes /cmd_vel.

Run in your ROS 2 docker (PyQt5 is already in ros2-generic):
  xhost +local:docker
  docker run --rm -it --network host -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v <WORKSPACE>:/ws -v ~/ground_homography.yaml:/root/ground_homography.yaml:ro \
    ros2-generic:latest bash -lc '
      source /opt/ros/humble/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
      python3 /ws/lane_pilot/tools/lane_debug_qt.py --homography /root/ground_homography.yaml'
"""
import argparse
import os
import sys
import time

import numpy as np
import cv2

# opencv-python ships its own Qt and hijacks QT_QPA_PLATFORM_PLUGIN_PATH, which
# clashes with PyQt5; combined with QT_DEBUG_PLUGINS=1 in the env this makes Qt
# verbosely scan /usr/bin and spam "... not a plugin". Drop those so PyQt5 uses
# its own Qt plugins quietly. Must run BEFORE the QApplication is created.
for _v in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_DEBUG_PLUGINS"):
    os.environ.pop(_v, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_lane as T          # Geom, detect (exact detector)
import lane_debug_ui as D      # Pilot, render_bird, render_traj, label/fit_h/pad_w, print_params

from PyQt5 import QtCore, QtGui, QtWidgets


# (key, label, lo, hi, default, scale, unit)
RAWCROP = [
    ("raw_top", "raw cut top", 0, 80, 0, 1, "%"),
    ("raw_bottom", "raw cut bottom", 20, 100, 100, 1, "%"),
    ("raw_left", "raw cut left", 0, 50, 0, 1, "%"),
    ("raw_right", "raw cut right", 50, 100, 100, 1, "%"),
]
GEOM = [
    ("gx_min", "ground x_min", 0, 30, 5, 0.01, "m"),
    ("gx_max", "ground x_max (depth)", 30, 150, 70, 0.01, "m"),
    ("gy_half", "ground y_half (width)", 15, 60, 35, 0.01, "m"),
    ("gmpp", "mpp", 15, 50, 25, 0.1, "mm"),
]
DET = [
    ("thr", "black_threshold", 0, 255, 130, 1, ""),
    ("invert", "invert (1=dark line)", 0, 1, 1, 1, ""),
    ("blur", "blur_ksize", 0, 9, 4, 1, ""),
    ("open_px", "morph_open (noise-)", 0, 9, 0, 1, "px"),
    ("close_px", "morph_close (gap+)", 0, 9, 0, 1, "px"),
    ("min_px", "min_cluster_px", 0, 40, 2, 1, "px"),
    ("max_px", "max_cluster_px", 5, 150, 16, 1, "px"),
    ("row_step", "row_step", 1, 6, 2, 1, ""),
    ("max_jump_cm", "max_jump", 1, 20, 6, 0.01, "m"),
    ("min_pts", "min_points", 1, 30, 7, 1, ""),
    ("roi_xmin", "roi_x_min", 0, 40, 17, 0.01, "m"),
    ("roi_xmax", "roi_x_max", 20, 70, 30, 0.01, "m"),
    ("roi_yhalf", "roi_y_half", 5, 35, 16, 0.01, "m"),
]
CTL = [
    ("v_max", "v_max", 1, 30, 12, 0.01, "m/s"),
    ("ld_base", "lookahead_base", 10, 50, 30, 0.01, "m"),
    ("ld_min", "lookahead_min", 10, 45, 25, 0.01, "m"),
    ("ld_max", "lookahead_max", 20, 60, 45, 0.01, "m"),
    ("kref", "curv_kappa_ref", 10, 80, 40, 0.1, "1/m"),
    ("slow", "curv_slow_gain", 0, 100, 70, 0.01, ""),
    ("curv_eval", "curv_eval_x", 5, 50, 30, 0.01, "m"),
    ("mem_time", "mem_time", 5, 50, 30, 0.1, "s"),
    ("max_ang", "max_angular", 5, 30, 18, 0.1, "rad/s"),
    ("max_degree", "max_degree", 1, 3, 2, 1, ""),
]


class SliderRow(QtWidgets.QWidget):
    def __init__(self, spec):
        super().__init__()
        self.key, lbl, lo, hi, dflt, self.scale, unit = spec
        self.unit = unit
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(2, 1, 2, 1)
        name = QtWidgets.QLabel(lbl)
        name.setFixedWidth(135)
        name.setStyleSheet("color:#ddd;")
        self.s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.s.setMinimum(lo)
        self.s.setMaximum(hi)
        self.s.setValue(dflt)
        self.val = QtWidgets.QLabel()
        self.val.setFixedWidth(80)
        self.val.setStyleSheet("color:#7cf; font-family:monospace;")
        self.s.valueChanged.connect(self._upd)
        h.addWidget(name)
        h.addWidget(self.s)
        h.addWidget(self.val)
        self._upd()

    def _upd(self):
        if self.scale == 1:
            self.val.setText("%d %s" % (self.s.value(), self.unit))
        else:
            self.val.setText("%.2f %s" % (self.s.value() * self.scale, self.unit))

    def raw(self):
        return self.s.value()

    def scaled(self):
        return self.s.value() * self.scale


def to_pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


class DebugWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("lane_pilot debug")
        fs = cv2.FileStorage(args.homography, cv2.FILE_STORAGE_READ)
        Hmat = fs.getNode("H_img2ground").mat()
        fs.release()
        if Hmat is None:
            raise SystemExit("could not read H_img2ground from " + args.homography)
        self.Hmat = Hmat
        self.pilot = D.Pilot()
        self.static = cv2.imread(args.image) if args.image else None
        self.odom = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self.state = {}
        self.node = self.rclpy = None
        if self.static is None:
            self._ros(args)

        # ---- layout ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        self.img_label = QtWidgets.QLabel()
        self.img_label.setAlignment(QtCore.Qt.AlignTop)
        scroll_img = QtWidgets.QScrollArea()
        scroll_img.setWidget(self.img_label)
        scroll_img.setWidgetResizable(True)
        scroll_img.setMinimumWidth(840)
        root.addWidget(scroll_img, 3)

        side = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(side)
        self.rows = {}

        def section(title, specs):
            head = QtWidgets.QLabel(title)
            head.setStyleSheet("color:#fff; font-weight:bold; margin-top:6px;")
            sv.addWidget(head)
            for sp in specs:
                row = SliderRow(sp)
                self.rows[sp[0]] = row
                sv.addWidget(row)

        section("RAW CROP (image space)", RAWCROP)
        section("GROUND STRIP (view)", GEOM)
        section("DETECTION", DET)
        section("CONTROL", CTL)
        for k, v in (("gx_min", args.x_min), ("gx_max", args.x_max),
                     ("gy_half", args.y_half)):
            self.rows[k].s.setValue(int(round(v * 100)))
        self.rows["gmpp"].s.setValue(int(round(args.mpp * 10000)))

        self.readout = QtWidgets.QLabel()
        self.readout.setStyleSheet("color:#9f9; font-family:monospace; font-size:11px;")
        sv.addWidget(self.readout)
        btns = QtWidgets.QHBoxLayout()
        b_print = QtWidgets.QPushButton("Print params")
        b_print.clicked.connect(self._print)
        b_save = QtWidgets.QPushButton("Save PNG")
        b_save.clicked.connect(self._save)
        b_clear = QtWidgets.QPushButton("Clear memory")
        b_clear.clicked.connect(lambda: self.pilot.mem.clear())
        btns.addWidget(b_print)
        btns.addWidget(b_save)
        btns.addWidget(b_clear)
        sv.addLayout(btns)
        sv.addStretch(1)

        side_scroll = QtWidgets.QScrollArea()
        side_scroll.setWidget(side)
        side_scroll.setWidgetResizable(True)
        side_scroll.setFixedWidth(420)
        root.addWidget(side_scroll, 1)
        self.setStyleSheet("background:#222;")
        self._dash = None

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    def _ros(self, args):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        from nav_msgs.msg import Odometry
        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node("lane_debug_qt")
        self.node.create_subscription(
            CompressedImage, args.topic,
            lambda m: self.state.__setitem__(
                "img", cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)),
            qos_profile_sensor_data)

        def on_odom(m):
            q = m.pose.pose.orientation
            self.odom["x"] = m.pose.pose.position.x
            self.odom["y"] = m.pose.pose.position.y
            self.odom["yaw"] = float(np.arctan2(2 * q.w * q.z, 1 - 2 * q.z * q.z))
        self.node.create_subscription(Odometry, args.odom_topic, on_odom, 10)

    def _P(self):
        r = self.rows
        return dict(
            thr=r["thr"].raw(), invert=bool(r["invert"].raw()), blur=r["blur"].raw(),
            open_px=r["open_px"].raw(), close_px=r["close_px"].raw(),
            min_px=r["min_px"].raw(), max_px=r["max_px"].raw(), row_step=r["row_step"].raw(),
            max_jump_cm=r["max_jump_cm"].raw(), min_pts=r["min_pts"].raw(),
            roi_xmin=r["roi_xmin"].raw(), roi_xmax=r["roi_xmax"].raw(),
            roi_yhalf=r["roi_yhalf"].raw())

    def _geom(self):
        r = self.rows
        x_min = r["gx_min"].scaled()
        x_max = max(x_min + 0.10, r["gx_max"].scaled())
        return T.Geom(x_min, x_max, r["gy_half"].scaled(), r["gmpp"].scaled() / 1000.0)

    def _C(self):
        r = self.rows
        return dict(
            v_max=r["v_max"].scaled(), v_min=0.03, ld_base=r["ld_base"].scaled(), ld_k=0.2,
            ld_min=r["ld_min"].scaled(), ld_max=r["ld_max"].scaled(), kref=r["kref"].scaled(),
            slow_gain=r["slow"].scaled(), mem_time=r["mem_time"].scaled(),
            max_ang=r["max_ang"].scaled(), max_degree=r["max_degree"].raw(),
            curv_eval_x=r["curv_eval"].scaled(), x_keep_min=-0.20, x_keep_max=0.80,
            y_keep=0.40, min_pts_fit=5, min_span=0.05, short_path_x=0.30, mem_max=1500)

    def tick(self):
        if self.static is not None:
            raw = self.static.copy()
        else:
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
            raw = self.state.get("img")
        if raw is None:
            return
        P, C = self._P(), self._C()
        g = self._geom()
        Wmap = g.warp_matrix(self.Hmat)
        # Raw image-space crop: black out everything outside the rectangle BEFORE
        # warping (keeps full-res coords so the homography still maps correctly).
        h0, w0 = raw.shape[:2]
        rt = int(self.rows["raw_top"].raw() / 100.0 * h0)
        rb = int(self.rows["raw_bottom"].raw() / 100.0 * h0)
        rl = int(self.rows["raw_left"].raw() / 100.0 * w0)
        rr = int(self.rows["raw_right"].raw() / 100.0 * w0)
        cropped = np.zeros_like(raw)
        cropped[rt:rb, rl:rr] = raw[rt:rb, rl:rr]
        bird = cv2.warpPerspective(cropped, Wmap, (g.W, g.H), flags=cv2.INTER_LINEAR)
        sel, clusters, mask, roi, conf, valid = T.detect(bird, g, P)
        perc = [g.bird_to_ground(c, r) for (c, r) in sel]
        pose = (self.odom["x"], self.odom["y"], self.odom["yaw"])
        if self.static is None and valid:
            self.pilot.add(perc, pose, time.time(), C["mem_time"], C["mem_max"])
        res = self.pilot.control(pose, C) if self.static is None else D._static_control(perc, C)

        raw_disp = raw.copy()
        cv2.rectangle(raw_disp, (rl, rt), (rr, rb), (0, 255, 255), 2)  # crop rect
        bird_viz = D.render_bird(bird, g, sel, clusters, roi, conf, valid)
        traj = D.render_traj(res, perc)
        top = D.pad_w(cv2.hconcat([D.fit_h(D.label(raw_disp, "raw camera"), 320),
                                   D.fit_h(D.label(bird_viz, "bird + detection"), 320)]), 1)
        bot = cv2.hconcat([D.fit_h(D.label(mask, "threshold mask"), 320),
                           D.fit_h(D.label(traj, "planned trajectory"), 320)])
        w = max(top.shape[1], bot.shape[1])
        self._dash = cv2.vconcat([D.pad_w(top, w), D.pad_w(bot, w)])
        self.img_label.setPixmap(to_pixmap(self._dash))
        self.img_label.resize(self._dash.shape[1], self._dash.shape[0])

        self.readout.setText(
            "status : %s\nv      : %.3f m/s\nomega  : %+.3f rad/s\nkappa  : %.2f 1/m\n"
            "lookah.: %.2f m\nmem pts: %d\nperc   : n=%d conf=%.2f %s\nmax_x  : %.2f m\n"
            "odom   : x=%.2f y=%.2f yaw=%.0f deg" % (
                res["status"], res["v"], res["omega"], res["kappa"], res["ld"],
                len(res["xs"]), len(sel), conf, "VALID" if valid else "weak",
                res["max_x"], self.odom["x"], self.odom["y"], np.degrees(self.odom["yaw"])))

    def _print(self):
        P, C, g = self._P(), self._C(), self._geom()
        print("\n# ---- lane_ipm_node (detection) ----")
        print("    bird_x_min: %.2f" % g.x_min)
        print("    bird_x_max: %.2f" % g.x_max)
        print("    bird_y_half: %.2f" % g.y_half)
        print("    bird_mpp: %.4f" % g.mpp)
        print("    raw_top_frac: %.2f" % (self.rows["raw_top"].raw() / 100.0))
        print("    raw_bottom_frac: %.2f" % (self.rows["raw_bottom"].raw() / 100.0))
        print("    raw_left_frac: %.2f" % (self.rows["raw_left"].raw() / 100.0))
        print("    raw_right_frac: %.2f" % (self.rows["raw_right"].raw() / 100.0))
        print("    black_threshold: %d" % P["thr"])
        print("    invert_threshold: %s" % ("true" if P["invert"] else "false"))
        print("    blur_ksize: %d" % P["blur"])
        print("    open_px: %d" % P["open_px"])
        print("    close_px: %d" % P["close_px"])
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

    def _save(self):
        if self._dash is not None:
            cv2.imwrite("/tmp/lane_debug.png", self._dash)
            print("saved /tmp/lane_debug.png")

    def closeEvent(self, e):
        if self.node is not None:
            self.node.destroy_node()
            self.rclpy.shutdown()
        e.accept()


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

    app = QtWidgets.QApplication(sys.argv)
    win = DebugWindow(args)
    win.resize(1320, 760)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
