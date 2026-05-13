#!/usr/bin/env python3
"""
Two-stage pixel-size approach.

Phase 1 (FAR): drive base-X using tag 2's apparent pixel size as
feedback. Phase ends when tag 2's size reaches its saved target
(== "id-2 finish point") AND tag 1 has become reliably visible.

Phase 2 (NEAR): drive base-X using tag 1's size AND base-Z toward the
saved Z position. Phase 2 is "closer and upper". Z motion is:
  - FEEDBACK mode (preferred): if the YAML has near_pixel_v, vz is
    proportional to (target_v - current_v), so the arm stops at the
    saved Z position. Done when both size AND v errors are in tolerance.
  - OPEN-LOOP fallback: if near_pixel_v is missing, vz is a constant
    --vz-up lift speed until size hits target. Less precise — Z stops
    only when X does.

Targets come from ./approach_target.yaml. Capture them with two
explicit save commands:
  1. Pose arm at the id-2 finish point (intermediate). Run with
     --save-far  → writes far_pixel_size.
  2. Pose arm at the id-1 finish point (final engaged pose).
     Run with --save-near → writes near_pixel_size AND near_pixel_v
     (the 'v' enables feedback Z control automatically).
(--save-target is also accepted and saves whatever tags are visible
right now — merges with existing file content.)

Twist plumbing: publishes TwistStamped on /servo_node/delta_twist_cmds
in --base-frame. Servo must already be running. Flip --invert-x or
--invert-z if approach makes the tag smaller / Z direction is wrong.
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


def tag_pixel_size(corners: np.ndarray) -> float:
    """Mean side length (px) of a 4-corner AprilTag detection."""
    sides = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
             for i in range(4)]
    return float(np.mean(sides))


class AlignApproach(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("align_approach_tag")
        self.args = args
        self.bridge = CvBridge()

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11)
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary, cv2.aruco.DetectorParameters())

        self.image_w: int | None = None
        self.image_h: int | None = None
        self._info_seen = False

        self._lock = threading.Lock()
        self._last_size_far: float | None = None
        self._last_size_near: float | None = None
        self._last_pix_near: tuple[float, float] | None = None
        self._last_t_far: float = 0.0
        self._last_t_near: float = 0.0

        # Saved targets (run mode only).
        self._target_far: float | None = None
        self._target_near: float | None = None
        self._target_near_v: float | None = None
        self._target_near_u: float | None = None
        self._target_source: str = "(not loaded)"
        a_save = args.save_target or args.save_far or args.save_near
        if not a_save and args.target and args.target.exists():
            data = yaml.safe_load(args.target.read_text()) or {}
            tf = data.get("far_pixel_size")
            tn = data.get("near_pixel_size")
            tnv = data.get("near_pixel_v")
            tnu = data.get("near_pixel_u")
            self._target_far = float(tf) if tf is not None else None
            self._target_near = float(tn) if tn is not None else None
            self._target_near_v = float(tnv) if tnv is not None else None
            self._target_near_u = float(tnu) if tnu is not None else None
            zmode = "feedback (v)" if self._target_near_v is not None else "const +vz-up"
            self._target_source = (
                f"{args.target.name}  far={self._target_far}  "
                f"near={self._target_near}  near_v={self._target_near_v}  "
                f"z_mode={zmode}")

        # State machine. `_near_eligible` becomes True once the FAR phase
        # is complete (target reached OR far lost while near is visible);
        # from that point on we prefer NEAR. `_active` is the tag whose
        # feedback we are currently using — it can fall back to FAR if
        # near temporarily drops out.
        self._near_eligible: bool = False
        self._active: str = "far"     # "far" | "near"
        self._near_streak: int = 0
        self._in_tol_count: int = 0
        self._tick_n: int = 0
        self._t0: float = time.time()
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

        if args.save_far:
            mode = "SAVE-FAR"
        elif args.save_near:
            mode = "SAVE-NEAR"
        elif args.save_target:
            mode = "SAVE-TARGET"
        else:
            mode = "APPROACH"
        self.get_logger().info(
            f"align_approach_tag ready  mode={mode}  "
            f"far_id={args.far_tag_id}  near_id={args.near_tag_id}  "
            f"rate={args.rate:.0f}Hz  dry_run={args.dry_run}")
        self.get_logger().info(f"target: {self._target_source}")
        self.get_logger().info(
            f"control: gain={args.gain}  max_speed={args.max_speed}  "
            f"tol_px={args.tol_px}  invert_x={args.invert_x}  "
            f"switch_frames={args.switch_frames}  "
            f"max_duration={args.max_duration}s")

        # In RUN mode (no save flag, not dry-run), require both targets.
        in_save_mode = (args.save_target or args.save_far or args.save_near)
        if not in_save_mode and not args.dry_run:
            if self._target_far is None:
                self.get_logger().error(
                    f"missing 'far_pixel_size' in {args.target} — capture "
                    f"with --save-far while tag {args.far_tag_id} is "
                    f"visible.")
                self.fail = True
                self.done.set()
            if self._target_near is None and not self.fail:
                self.get_logger().error(
                    f"missing 'near_pixel_size' in {args.target} — capture "
                    f"with --save-near while tag {args.near_tag_id} is "
                    f"visible.")
                self.fail = True
                self.done.set()

    # ---------- subscribers ----------

    def on_info(self, msg: CameraInfo) -> None:
        if self._info_seen:
            return
        self.image_w = int(msg.width)
        self.image_h = int(msg.height)
        self._info_seen = True
        self.get_logger().info(
            f"camera_info: image={self.image_w}x{self.image_h}")

    def on_image(self, msg: CompressedImage) -> None:
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return
        ids_flat = ids.flatten().tolist()
        now = time.time()
        with self._lock:
            if self.args.far_tag_id in ids_flat:
                i = ids_flat.index(self.args.far_tag_id)
                self._last_size_far = tag_pixel_size(corners[i][0])
                self._last_t_far = now
            if self.args.near_tag_id in ids_flat:
                i = ids_flat.index(self.args.near_tag_id)
                pts = corners[i][0]
                self._last_size_near = tag_pixel_size(pts)
                self._last_pix_near = (float(pts[:, 0].mean()),
                                       float(pts[:, 1].mean()))
                self._last_t_near = now

    # ---------- control ----------

    def tick(self) -> None:
        self._tick_n += 1
        with self._lock:
            sf = self._last_size_far
            sn = self._last_size_near
            tf = self._last_t_far
            tn = self._last_t_near
        now = time.time()

        # ── SAVE modes ─────────────────────────────────────────────────
        if self.args.save_target or self.args.save_far or self.args.save_near:
            age_far = (now - tf) if sf is not None else float("inf")
            age_near = (now - tn) if sn is not None else float("inf")
            fresh_far_size = sf if (sf is not None and age_far < 0.5) else None
            fresh_near_size = sn if (sn is not None and age_near < 0.5) else None

            want_far = self.args.save_target or self.args.save_far
            want_near = self.args.save_target or self.args.save_near

            ready_far = (fresh_far_size is not None) if want_far else True
            ready_near = (fresh_near_size is not None) if want_near else True
            # For --save-target we don't require BOTH — any single tag is OK.
            if self.args.save_target:
                ready = (fresh_far_size is not None
                         or fresh_near_size is not None)
            else:
                ready = ready_far and ready_near

            if ready:
                existing: dict = {}
                if self.args.target.exists():
                    loaded = yaml.safe_load(self.args.target.read_text())
                    if isinstance(loaded, dict):
                        existing = loaded
                if want_far and fresh_far_size is not None:
                    existing["far_tag_id"] = int(self.args.far_tag_id)
                    existing["far_pixel_size"] = float(fresh_far_size)
                if want_near and fresh_near_size is not None:
                    existing["near_tag_id"] = int(self.args.near_tag_id)
                    existing["near_pixel_size"] = float(fresh_near_size)
                    pix = self._last_pix_near
                    if pix is not None:
                        existing["near_pixel_u"] = float(pix[0])
                        existing["near_pixel_v"] = float(pix[1])
                if self.image_w:
                    existing["image_w"] = int(self.image_w)
                if self.image_h:
                    existing["image_h"] = int(self.image_h)
                self.args.target.write_text(
                    yaml.safe_dump(existing, sort_keys=False))
                tag = ("--save-far" if self.args.save_far
                       else "--save-near" if self.args.save_near
                       else "--save-target")
                self.get_logger().info(
                    f"{tag} -> {self.args.target}  "
                    f"far={existing.get('far_pixel_size')}  "
                    f"near={existing.get('near_pixel_size')}")
                self.done.set()
                return
            if (now - self._t0) > self.args.startup_timeout:
                missing = []
                if want_far and fresh_far_size is None:
                    missing.append(f"FAR tag {self.args.far_tag_id}")
                if want_near and fresh_near_size is None:
                    missing.append(f"NEAR tag {self.args.near_tag_id}")
                self.get_logger().error(
                    f"never saw: {', '.join(missing)} after "
                    f"{self.args.startup_timeout:.1f}s — aborting save.")
                self.fail = True
                self.done.set()
            return

        # ── RUN mode: state machine + control ──────────────────────────

        age_far = (now - tf) if sf is not None else float("inf")
        age_near = (now - tn) if sn is not None else float("inf")
        fresh_far = sf is not None and age_far < self.args.lost_timeout
        fresh_near = sn is not None and age_near < self.args.lost_timeout

        # Track near-tag visibility streak (used for transition gating).
        if fresh_near:
            self._near_streak += 1
        else:
            self._near_streak = 0

        # FAR target check: tag 2 size is within tolerance of its saved
        # target. Only meaningful if target_far loaded AND we have a
        # fresh size reading.
        far_in_tol = False
        if self._target_far is not None and fresh_far and sf is not None:
            far_err = float(self._target_far) - float(sf)
            far_in_tol = abs(far_err) < self.args.far_tol_px

        # Become "near-eligible" once either:
        #   (a) FAR has reached its target AND near has been steadily
        #       visible for switch_frames frames, or
        #   (b) FAR has gone stale AND near is steadily visible — the
        #       far tag dropped out, so jump forward into the near phase.
        # Once eligible we stay eligible (no falling back to phase 1).
        if not self._near_eligible:
            if far_in_tol and self._near_streak >= self.args.switch_frames:
                self._near_eligible = True
                self.get_logger().info(
                    f"FAR target reached (within "
                    f"{self.args.far_tol_px:.0f}px). Entering NEAR phase.")
            elif (not fresh_far
                  and self._near_streak >= self.args.switch_frames):
                self._near_eligible = True
                self.get_logger().info(
                    f"FAR tag {self.args.far_tag_id} not visible, near is "
                    f"— entering NEAR phase early.")

        # Pick which tag drives feedback THIS tick. Preference order:
        # near (if eligible + fresh) → far (fallback while waiting for
        # near) → whichever is currently fresh → nothing.
        if self._near_eligible and fresh_near:
            new_active: str | None = "near"
        elif self._near_eligible and fresh_far:
            new_active = "far"   # temporary fallback
        elif fresh_far:
            new_active = "far"
        elif fresh_near:
            new_active = "near"
        else:
            new_active = None

        if new_active is not None and new_active != self._active:
            self.get_logger().info(
                f"active tag: {self._active} → {new_active}  "
                f"(near_eligible={self._near_eligible}, "
                f"fresh_far={fresh_far}, fresh_near={fresh_near})")
            self._active = new_active
            self._in_tol_count = 0

        # Global timeout — only enforced when --max-duration > 0.
        if (self.args.max_duration > 0
                and (now - self._t0) > self.args.max_duration):
            self.publish_twist(0.0, 0.0)
            self.get_logger().error(
                f"max_duration {self.args.max_duration:.1f}s exceeded "
                f"— aborting.")
            self.fail = True
            self.done.set()
            return

        # No fresh tag this tick. Coast. Abort only when BOTH tags have
        # been gone past lost_giveup (or neither has ever been seen).
        if new_active is None:
            self.publish_twist(0.0, 0.0)
            if (sf is None and sn is None
                    and (now - self._t0) > self.args.startup_timeout):
                self.get_logger().error(
                    f"neither far ({self.args.far_tag_id}) nor near "
                    f"({self.args.near_tag_id}) seen after "
                    f"{self.args.startup_timeout:.1f}s — aborting.")
                self.fail = True
                self.done.set()
            elif (sf is not None and sn is not None
                  and min(age_far, age_near) > self.args.lost_giveup):
                self.get_logger().error(
                    f"both tags lost (far {age_far:.1f}s, near "
                    f"{age_near:.1f}s) > lost_giveup="
                    f"{self.args.lost_giveup:.1f}s — aborting.")
                self.fail = True
                self.done.set()
            return

        # Active tag size + target.
        if self._active == "far":
            size = sf
            target = self._target_far
            tag_id = self.args.far_tag_id
            age = age_far
        else:
            size = sn
            target = self._target_near
            tag_id = self.args.near_tag_id
            age = age_near

        # Should not happen given new_active != None, but keep a guard.
        if size is None:
            self.publish_twist(0.0, 0.0)
            return

        # By construction, age <= lost_timeout (new_active was picked
        # from freshness). Nothing more to do for staleness here.

        # Dry-run with no target loaded → just print readings.
        if target is None and self.args.dry_run:
            log_every = max(1, int(self.args.rate / 2))
            if self._tick_n % log_every == 1:
                self.get_logger().info(
                    f"[DRY no-target]  active={self._active}  "
                    f"far={sf}  near={sn}  near_streak={self._near_streak}")
            return

        target_f = float(target)            # narrows for the type checker
        err = target_f - float(size)        # +ve when we're too far away
        vx_raw = float(self.args.gain) * err
        if self.args.invert_x:
            vx_raw = -vx_raw
        # Phase-specific cap on vx: phase 2 (NEAR) typically runs slower
        # so Z motion can keep up and the arm arrives at the saved point
        # rather than overshooting in X.
        vx_cap = (self.args.near_max_speed
                  if self._active == "near"
                  else self.args.max_speed)
        vx = float(np.clip(vx_raw, -vx_cap, vx_cap))

        # Z command:
        #   feedback mode  → vz proportional to (target_v - current_v_of_near)
        #   open-loop mode → constant vz_up
        # In FAR phase vz is always zero.
        vz_cmd = 0.0
        v_err: float | None = None
        z_mode = "off"
        if self._active == "near":
            if self._target_near_v is not None:
                with self._lock:
                    pix = self._last_pix_near
                if pix is not None:
                    v_cur = pix[1]
                    v_err = float(self._target_near_v) - v_cur
                    vz_raw = float(self.args.gain_z) * v_err
                    if self.args.invert_z:
                        vz_raw = -vz_raw
                    vz_cmd = float(np.clip(vz_raw,
                                           -self.args.max_speed_z,
                                           self.args.max_speed_z))
                    z_mode = "fb"
            else:
                vz_cmd = float(self.args.vz_up)
                if self.args.invert_z:
                    vz_cmd = -vz_cmd
                z_mode = "open"

        # Done: only fires once near-eligible AND active=near AND both
        # axis errors are within tolerance. A temporary fallback to far
        # while in the near phase will NOT trigger done.
        in_tol_x = abs(err) < self.args.tol_px
        in_tol_z = (
            True
            if v_err is None
            else abs(v_err) < self.args.tol_v_px
        )
        near_phase = self._near_eligible and self._active == "near"
        if near_phase and in_tol_x and in_tol_z:
            self._in_tol_count += 1
        else:
            self._in_tol_count = 0

        if near_phase and self._in_tol_count >= self.args.tol_frames:
            self.publish_twist(0.0, 0.0)
            verr_s = f"  v_err={v_err:+.1f}px" if v_err is not None else ""
            self.get_logger().info(
                f"APPROACHED (near)  size={size:.1f}px  "
                f"target={target_f:.1f}px  err={err:+.1f}px{verr_s} "
                f"({self._in_tol_count} frames). Done.")
            self.done.set()
            return

        if near_phase and in_tol_x and in_tol_z:
            vx = 0.0
            vz_cmd = 0.0
        elif near_phase and in_tol_x:
            # X is done but Z (v) still off — stop vx but keep lifting.
            vx = 0.0
        self.publish_twist(vx, vz_cmd)

        log_every = max(1, int(self.args.rate / 2))
        if self._tick_n % log_every == 1:
            dry = " [DRY]" if self.args.dry_run else ""
            verr_s = (f"  v_err={v_err:+5.1f}"
                      if v_err is not None else "")
            self.get_logger().info(
                f"[{self._active}/z={z_mode}]  size={size:6.1f}px  "
                f"target={target_f:6.1f}px  err={err:+6.1f}px{verr_s}  "
                f"vx={vx:+.3f}  vz={vz_cmd:+.3f}  "
                f"near_streak={self._near_streak}{dry}")

    def publish_twist(self, vx: float, vz: float = 0.0) -> None:
        if self.args.dry_run:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.base_frame
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = float(vz)
        self.pub.publish(msg)

    # Backward-compatible alias used in shutdown / no-target paths.
    def publish_vx(self, vx: float) -> None:
        self.publish_twist(vx, 0.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-topic",
                   default="/zed2/zed_node/rgb/color/rect/image")
    p.add_argument("--camera-info-topic",
                   default="/zed2/zed_node/rgb/color/rect/camera_info")
    p.add_argument("--far-tag-id", type=int, default=2,
                   help="Tag id used until the near tag is reliably "
                        "visible (default 2).")
    p.add_argument("--near-tag-id", type=int, default=1,
                   help="Tag id used for the final precision phase "
                        "(default 1).")
    p.add_argument("--switch-frames", type=int, default=5,
                   help="Consecutive frames the near tag must be seen "
                        "before switching from FAR → NEAR. Default 5.")
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--gain", type=float, default=0.002,
                   help="m/s per pixel of size error.")
    p.add_argument("--max-speed", type=float, default=0.05,
                   help="Hard cap on |vx| during phase 1 (FAR). In m/s.")
    p.add_argument("--near-max-speed", type=float, default=0.05,
                   help="Hard cap on |vx| during phase 2 (NEAR). Keep "
                        "close to --max-speed-z so X and Z arrive at the "
                        "saved target together. In m/s.")
    p.add_argument("--tol-px", type=float, default=3.0,
                   help="Pixel tolerance on apparent-size error (near).")
    p.add_argument("--tol-frames", type=int, default=5)
    p.add_argument("--invert-x", action="store_true",
                   help="Flip vx if approach makes the tag smaller.")
    p.add_argument("--base-frame", default="link_base")
    p.add_argument("--twist-topic", default="/servo_node/delta_twist_cmds")
    p.add_argument("--lost-timeout", type=float, default=0.4)
    p.add_argument("--lost-giveup", type=float, default=5.0)
    p.add_argument("--startup-timeout", type=float, default=15.0)
    p.add_argument("--max-duration", type=float, default=0.0,
                   help="Hard cap on total approach time, in seconds. "
                        "0 = no timeout (default).")
    p.add_argument("--target", type=Path,
                   default=Path("approach_target.yaml"),
                   help="YAML with saved tag pixel sizes (far + near).")
    p.add_argument("--save-target", action="store_true",
                   help="Capture currently-visible tags' pixel sizes "
                        "and write to --target, then exit. Partial saves "
                        "are merged.")
    p.add_argument("--save-far", action="store_true",
                   help="Capture ONLY the FAR tag's pixel size at the "
                        "current pose and merge into --target. Use this "
                        "from the id-2 finish point.")
    p.add_argument("--save-near", action="store_true",
                   help="Capture ONLY the NEAR tag's pixel size at the "
                        "current pose and merge into --target. Use this "
                        "from the id-1 finish point.")
    p.add_argument("--far-tol-px", type=float, default=5.0,
                   help="Pixel tolerance on the FAR tag's size error "
                        "for the phase-1 → phase-2 transition.")
    p.add_argument("--vz-up", type=float, default=0.02,
                   help="Open-loop base-Z velocity (m/s) commanded in "
                        "phase 2 IF the YAML does NOT have a saved "
                        "near_pixel_v (i.e. no feedback target). 0 "
                        "disables lift. Default 0.02 m/s.")
    p.add_argument("--gain-z", type=float, default=0.001,
                   help="m/s per pixel of v-error for feedback Z control "
                        "in phase 2. Used when near_pixel_v is in the YAML.")
    p.add_argument("--max-speed-z", type=float, default=0.05,
                   help="Hard cap on |vz| (m/s) in feedback mode.")
    p.add_argument("--tol-v-px", type=float, default=10.0,
                   help="Pixel tolerance on v-error for feedback Z "
                        "control's done condition.")
    p.add_argument("--invert-z", action="store_true",
                   help="Flip the sign of vz (whether open-loop or "
                        "feedback) if 'upper' is the opposite direction.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute everything but do NOT publish twist.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = AlignApproach(args)
    try:
        while rclpy.ok() and not node.done.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            for _ in range(2):
                node.publish_vx(0.0)
                time.sleep(0.05)
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if (node.done.is_set() and not node.fail) else 1


if __name__ == "__main__":
    sys.exit(main())
