#!/usr/bin/env python3
"""
Pixel-space alignment from one AprilTag (source, on the bottle) to
another AprilTag (target, in the workspace) — moving in ONE base axis
only.

Same plumbing as align_top_tag.py: MoveIt Servo via TwistStamped on
/servo_node/delta_twist_cmds, no xArm mode switching, no IK. The
previous step centred the bottle on its saved alignment position;
this step moves the bottle FROM there ONTO tag id 1 (default) by
commanding velocity in only one base axis. The other is held at zero.

Tag size does NOT matter for pixel centroid alignment — we use the
average of the 4 corners. The 54 mm size of tag id 1 only matters
for pose-based work (which this script does not do).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import CompressedImage, CameraInfo


class AlignDropTarget(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("align_drop_target")
        self.args = args
        self.bridge = CvBridge()

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11)
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary, cv2.aruco.DetectorParameters())

        self.cx: float | None = None
        self.cy: float | None = None
        self.image_w: int | None = None
        self.image_h: int | None = None

        self._lock = threading.Lock()
        # Pixel centroids and detection timestamps for the two tags.
        self._src_pix: tuple[float, float] | None = None  # bottle (--tag-id)
        self._tgt_pix: tuple[float, float] | None = None  # drop point (--target-tag-id)
        self._src_t: float = 0.0
        self._tgt_t: float = 0.0

        self._in_tol_count = 0
        self._tick_n = 0
        self._t0 = time.time()
        self.done = threading.Event()
        self.fail = False

        qos_image = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST, depth=5)
        qos_info = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(CompressedImage,
                                 args.image_topic + "/compressed",
                                 self.on_image, qos_image)
        self.create_subscription(CameraInfo, args.camera_info_topic,
                                 self.on_info, qos_info)
        self.pub = self.create_publisher(TwistStamped, args.twist_topic, 10)
        self.timer = self.create_timer(1.0 / float(args.rate), self.tick)

        self.get_logger().info(
            f"align_drop_target ready  source_tag={args.tag_id}  "
            f"target_tag={args.target_tag_id}  control_axis={args.control_axis}  "
            f"image={args.image_topic}/compressed  rate={args.rate:.0f}Hz  "
            f"dry_run={args.dry_run}")
        self.get_logger().info(
            f"map: axis_x={args.axis_x}  invert_x={args.invert_x}  "
            f"invert_y={args.invert_y}  gain={args.gain}  "
            f"max_speed={args.max_speed}  tol_px={args.tol_px}")

    # ---------- subscribers ----------

    def on_info(self, msg: CameraInfo) -> None:
        if self.cx is not None:
            return
        K = np.array(msg.k).reshape(3, 3)
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.image_w = int(msg.width)
        self.image_h = int(msg.height)
        self.get_logger().info(
            f"intrinsics locked: cx={self.cx:.1f}  cy={self.cy:.1f}  "
            f"image={self.image_w}x{self.image_h}")

    def on_image(self, msg: CompressedImage) -> None:
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return
        ids_flat = ids.flatten().tolist()
        now = time.time()
        with self._lock:
            if self.args.tag_id in ids_flat:
                i = ids_flat.index(self.args.tag_id)
                pts = corners[i][0]
                self._src_pix = (float(pts[:, 0].mean()),
                                 float(pts[:, 1].mean()))
                self._src_t = now
            if self.args.target_tag_id in ids_flat:
                i = ids_flat.index(self.args.target_tag_id)
                pts = corners[i][0]
                self._tgt_pix = (float(pts[:, 0].mean()),
                                 float(pts[:, 1].mean()))
                self._tgt_t = now

    # ---------- control ----------

    def pixel_to_base_vel(self, du: float, dv: float) -> tuple[float, float]:
        """Same axis-mapping logic as align_top_tag.py."""
        a = self.args
        g = float(a.gain)
        if a.axis_x == "u":
            vx_raw, vy_raw = -g * du, -g * dv
        else:  # axis_x == "v"
            vx_raw, vy_raw = -g * dv, -g * du
        if a.invert_x:
            vx_raw = -vx_raw
        if a.invert_y:
            vy_raw = -vy_raw
        vx = float(np.clip(vx_raw, -a.max_speed, a.max_speed))
        vy = float(np.clip(vy_raw, -a.max_speed, a.max_speed))
        return vx, vy

    def tick(self) -> None:
        if self.cx is None:
            self.publish_twist(0.0, 0.0)
            return

        self._tick_n += 1
        with self._lock:
            src = self._src_pix
            tgt = self._tgt_pix
            src_age = (time.time() - self._src_t) if src else float("inf")
            tgt_age = (time.time() - self._tgt_t) if tgt else float("inf")

        now = time.time()

        # Startup bail if either tag is still unseen.
        if src is None or tgt is None:
            self.publish_twist(0.0, 0.0)
            if (now - self._t0) > self.args.startup_timeout:
                missing = []
                if src is None:
                    missing.append(f"source id {self.args.tag_id}")
                if tgt is None:
                    missing.append(f"target id {self.args.target_tag_id}")
                self.get_logger().error(
                    f"never saw tag(s): {', '.join(missing)} — aborting.")
                self.fail = True
                self.done.set()
            return

        # The target is stationary, so we tolerate a stale target reading.
        # The source moves with the arm, so a stale source means we should
        # halt.
        if src_age > self.args.lost_giveup:
            self.publish_twist(0.0, 0.0)
            self.get_logger().error(
                f"source tag {self.args.tag_id} lost for {src_age:.1f}s "
                f"— aborting.")
            self.fail = True
            self.done.set()
            return
        if src_age > self.args.lost_timeout:
            self.publish_twist(0.0, 0.0)
            return

        u_s, v_s = src
        u_t, v_t = tgt
        du = u_s - u_t   # source minus target ⇒ same sign convention
        dv = v_s - v_t   # as align_top_tag.py (drive du,dv → 0)

        vx, vy = self.pixel_to_base_vel(du, dv)

        # Apply the one-axis lock.
        axis = self.args.control_axis
        if axis == "x":
            vy = 0.0
        elif axis == "y":
            vx = 0.0
        # axis == "both" → leave both active

        # Done condition uses only the unlocked axes.
        if axis == "x":
            # base-X is driven by the image axis selected as 'axis_x'.
            err_unlocked = dv if self.args.axis_x == "v" else du
            in_tol = abs(err_unlocked) < self.args.tol_px
        elif axis == "y":
            err_unlocked = du if self.args.axis_x == "v" else dv
            in_tol = abs(err_unlocked) < self.args.tol_px
        else:
            in_tol = (abs(du) < self.args.tol_px
                      and abs(dv) < self.args.tol_px)

        if in_tol:
            self._in_tol_count += 1
        else:
            self._in_tol_count = 0
        if self._in_tol_count >= self.args.tol_frames:
            self.publish_twist(0.0, 0.0)
            self.get_logger().info(
                f"ALIGNED  du={du:+.1f}  dv={dv:+.1f}  "
                f"({self._in_tol_count} frames < {self.args.tol_px}px on "
                f"axis={axis}). Done.")
            self.done.set()
            return

        if in_tol:
            vx = vy = 0.0
        self.publish_twist(vx, vy)

        log_every = max(1, int(self.args.rate / 2))
        if self._tick_n % log_every == 1:
            dry = " [DRY]" if self.args.dry_run else ""
            self.get_logger().info(
                f"src=({u_s:.0f},{v_s:.0f})  tgt=({u_t:.0f},{v_t:.0f})  "
                f"du={du:+7.1f}  dv={dv:+7.1f}  "
                f"vx={vx:+.3f}  vy={vy:+.3f}  in_tol={self._in_tol_count}{dry}")

    def publish_twist(self, vx: float, vy: float) -> None:
        if self.args.dry_run:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.base_frame
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = 0.0
        self.pub.publish(msg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-topic",
                   default="/zed2/zed_node/rgb/color/rect/image",
                   help="Base image topic; '/compressed' is appended.")
    p.add_argument("--camera-info-topic",
                   default="/zed2/zed_node/rgb/color/rect/camera_info")
    p.add_argument("--tag-id", type=int, default=13,
                   help="Source tag id (on the bottle, moves with arm).")
    p.add_argument("--target-tag-id", type=int, default=1,
                   help="Target tag id (drop point in workspace).")
    p.add_argument("--control-axis", choices=("x", "y", "both"), default="y",
                   help="Which base axis to drive. 'x' drives base-X and "
                        "locks base-Y. 'y' drives base-Y and locks base-X. "
                        "'both' = no lock. Default 'y' — flip if your two "
                        "tags are offset along base-X instead.")
    p.add_argument("--rate", type=float, default=20.0,
                   help="TwistStamped publish rate in Hz.")
    p.add_argument("--gain", type=float, default=0.001,
                   help="m/s per pixel of error.")
    p.add_argument("--max-speed", type=float, default=0.1,
                   help="Hard cap on |v| in m/s.")
    p.add_argument("--tol-px", type=float, default=25.0,
                   help="Per-axis pixel tolerance on the UNLOCKED axis only.")
    p.add_argument("--tol-frames", type=int, default=5,
                   help="Consecutive ticks inside tolerance to declare done.")
    p.add_argument("--axis-x", choices=("u", "v"), default="v",
                   help="Image axis that drives base-X (same convention "
                        "as align_top_tag.py — keep 'v' unless you remount "
                        "the camera).")
    p.add_argument("--invert-x", action="store_true",
                   help="Flip sign of the base-X command.")
    p.add_argument("--invert-y", action="store_true",
                   help="Flip sign of the base-Y command.")
    p.add_argument("--base-frame", default="link_base")
    p.add_argument("--twist-topic", default="/servo_node/delta_twist_cmds")
    p.add_argument("--lost-timeout", type=float, default=0.4)
    p.add_argument("--lost-giveup", type=float, default=5.0)
    p.add_argument("--startup-timeout", type=float, default=15.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Compute everything but do NOT publish twist.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = AlignDropTarget(args)
    try:
        while rclpy.ok() and not node.done.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            for _ in range(2):
                node.publish_twist(0.0, 0.0)
                time.sleep(0.05)
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if (node.done.is_set() and not node.fail) else 1


if __name__ == "__main__":
    sys.exit(main())
