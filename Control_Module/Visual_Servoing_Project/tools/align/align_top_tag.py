#!/usr/bin/env python3
"""
Align the xArm6 so AprilTag id 13 (mounted on the bottle's lid) sits at
the image center of the *second* ZED.

Uses MoveIt Servo: publishes a base-frame TwistStamped on
/servo_node/delta_twist_cmds with vx/vy only — Servo handles the
joint-space math. No xArm mode switching, no IK.

Servo MUST already be running (e.g. via bringup.launch.py). The
incoming-command timeout for Servo is 0.2 s, so we publish at >= 5 Hz
even when there is no fresh detection (zero twist when stale).

CLI flags let you pick which image axis drives which base axis and the
sign — calibrate once with --dry-run, then run live.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import CompressedImage, CameraInfo


class AlignTopTag(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("align_top_tag")
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
        self._last_pix: tuple[float, float] | None = None
        self._last_pix_t: float = 0.0

        self._in_tol_count = 0
        self._tick_n = 0
        self._t0 = time.time()
        self.done = threading.Event()
        self.fail = False

        # Alignment target in pixels. If a target file exists and we are
        # not in save mode, load it now; otherwise we'll fall back to the
        # image center once camera_info arrives.
        self._target_u: float | None = None
        self._target_v: float | None = None
        self._target_source: str = "image center (no target file)"
        if not args.save_target and args.target and args.target.exists():
            data = yaml.safe_load(args.target.read_text())
            self._target_u = float(data["u"])
            self._target_v = float(data["v"])
            self._target_source = (
                f"file {args.target.name} (tag {data.get('tag_id', '?')}, "
                f"saved u={self._target_u:.1f} v={self._target_v:.1f})")

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

        mode = "SAVE-TARGET" if args.save_target else "ALIGN"
        self.get_logger().info(
            f"align_top_tag ready  mode={mode}  "
            f"image={args.image_topic}/compressed  "
            f"info={args.camera_info_topic}  tag_id={args.tag_id}  "
            f"rate={args.rate:.0f}Hz  dry_run={args.dry_run}")
        self.get_logger().info(f"target: {self._target_source}")
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
            f"camera intrinsics locked: cx={self.cx:.1f}  cy={self.cy:.1f}  "
            f"image={self.image_w}x{self.image_h}")

    def on_image(self, msg: CompressedImage) -> None:
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return
        ids_flat = ids.flatten().tolist()
        if self.args.tag_id not in ids_flat:
            return
        i = ids_flat.index(self.args.tag_id)
        pts = corners[i][0]  # (4, 2)
        u = float(pts[:, 0].mean())
        v = float(pts[:, 1].mean())
        with self._lock:
            self._last_pix = (u, v)
            self._last_pix_t = time.time()

    # ---------- control ----------

    def pixel_to_base_vel(self, du: float, dv: float) -> tuple[float, float]:
        """Map (du, dv) pixel error to (vx_base, vy_base) m/s.

        Sign convention: we want du and dv driven toward zero, so the raw
        command is -gain * error. The image-axis-to-base-axis mapping
        plus invert flags are user-configurable since the second ZED is
        not in the robot TF tree.
        """
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
            # Camera info hasn't arrived yet.
            self.publish_twist(0.0, 0.0)
            return

        self._tick_n += 1
        with self._lock:
            pix = self._last_pix
            t_pix = self._last_pix_t

        now = time.time()
        age = (now - t_pix) if pix is not None else float("inf")

        # Never seen the tag? Bail after startup_timeout.
        if pix is None:
            self.publish_twist(0.0, 0.0)
            if (now - self._t0) > self.args.startup_timeout:
                self.get_logger().error(
                    f"tag id {self.args.tag_id} never seen after "
                    f"{self.args.startup_timeout:.1f}s — aborting.")
                self.fail = True
                self.done.set()
            return

        # ── save-target mode: dump (u, v) and exit on first sighting ─────
        if self.args.save_target:
            u, v = pix
            data = {
                "tag_id": int(self.args.tag_id),
                "u": float(u),
                "v": float(v),
                "image_w": int(self.image_w) if self.image_w else None,
                "image_h": int(self.image_h) if self.image_h else None,
            }
            self.args.target.write_text(yaml.safe_dump(data, sort_keys=False))
            self.get_logger().info(
                f"saved target u={u:.1f}  v={v:.1f}  (tag {self.args.tag_id}) "
                f"-> {self.args.target}")
            self.done.set()
            return

        # Tag was seen but is now stale.
        if age > self.args.lost_giveup:
            self.publish_twist(0.0, 0.0)
            self.get_logger().error(
                f"tag id {self.args.tag_id} lost for {age:.1f}s — aborting.")
            self.fail = True
            self.done.set()
            return
        if age > self.args.lost_timeout:
            # Brief drop-out — coast at zero.
            self.publish_twist(0.0, 0.0)
            return

        # Target falls back to the image center if no target file was loaded.
        tu = self._target_u if self._target_u is not None else self.cx
        tv = self._target_v if self._target_v is not None else self.cy

        u, v = pix
        du = u - tu
        dv = v - tv
        err_px = float(np.hypot(du, dv))

        in_tol = abs(du) < self.args.tol_px and abs(dv) < self.args.tol_px
        if in_tol:
            self._in_tol_count += 1
        else:
            self._in_tol_count = 0

        if self._in_tol_count >= self.args.tol_frames:
            self.publish_twist(0.0, 0.0)
            self.get_logger().info(
                f"ALIGNED  du={du:+.1f}px  dv={dv:+.1f}px  "
                f"({self._in_tol_count} frames < {self.args.tol_px}px). Done.")
            self.done.set()
            return

        vx, vy = self.pixel_to_base_vel(du, dv)
        if in_tol:
            vx = vy = 0.0
        self.publish_twist(vx, vy)

        # ~2 Hz log so we can watch convergence.
        log_every = max(1, int(self.args.rate / 2))
        if self._tick_n % log_every == 1:
            tag = " [DRY]" if self.args.dry_run else ""
            self.get_logger().info(
                f"du={du:+7.1f}  dv={dv:+7.1f}  err={err_px:6.1f}px  "
                f"vx={vx:+.3f}  vy={vy:+.3f}  in_tol={self._in_tol_count}{tag}")

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
                   default="/zed2/zed_node/rgb/color/rect/camera_info",
                   help="CameraInfo topic for the second ZED.")
    p.add_argument("--tag-id", type=int, default=13,
                   help="AprilTag 36h11 id to track (default 13, bottle lid).")
    p.add_argument("--rate", type=float, default=20.0,
                   help="TwistStamped publish rate in Hz. Must stay > 5 Hz "
                        "(Servo incoming_command_timeout is 0.2 s).")
    p.add_argument("--gain", type=float, default=0.0005,
                   help="m/s per pixel of error. 0.0005 * 100 px = 0.05 m/s, "
                        "clamped by --max-speed.")
    p.add_argument("--max-speed", type=float, default=0.03,
                   help="Hard cap on |vx|, |vy| in m/s.")
    p.add_argument("--tol-px", type=float, default=15.0,
                   help="Per-axis pixel tolerance to declare 'aligned'.")
    p.add_argument("--tol-frames", type=int, default=5,
                   help="Consecutive ticks inside tolerance to declare done.")
    p.add_argument("--axis-x", choices=("u", "v"), default="v",
                   help="Which image axis drives base-X. 'v' (image rows) "
                        "by default — typical for an overhead-mounted camera "
                        "whose optical X is parallel to the robot's base Y.")
    p.add_argument("--invert-x", action="store_true",
                   help="Flip sign of the base-X command.")
    p.add_argument("--invert-y", action="store_true",
                   help="Flip sign of the base-Y command.")
    p.add_argument("--base-frame", default="link_base",
                   help="frame_id for the published TwistStamped — must "
                        "match Servo's robot_link_command_frame.")
    p.add_argument("--twist-topic", default="/servo_node/delta_twist_cmds")
    p.add_argument("--lost-timeout", type=float, default=0.4,
                   help="Send zero twist if no fresh pixel reading for this long.")
    p.add_argument("--lost-giveup", type=float, default=5.0,
                   help="Abort with non-zero exit if tag lost for this long.")
    p.add_argument("--startup-timeout", type=float, default=15.0,
                   help="Abort if tag never seen after this many seconds.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute everything but do NOT publish twist — use "
                        "for direction/axis calibration.")
    p.add_argument("--target", type=Path,
                   default=Path("align_target.yaml"),
                   help="YAML file with the saved tag pixel target "
                        "(u, v). If the file exists, the controller "
                        "aligns to that pixel instead of the image "
                        "center. Default: align_target.yaml in cwd.")
    p.add_argument("--save-target", action="store_true",
                   help="Capture the current tag pixel position and "
                        "write it to --target, then exit. Use this to "
                        "snapshot the arm's current alignment.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = AlignTopTag(args)
    try:
        while rclpy.ok() and not node.done.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Always send a couple of zero-twists on the way out so Servo halts
        # cleanly rather than coasting on the last command.
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
