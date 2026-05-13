#!/usr/bin/env python3
"""
Interactive hand-eye calibration sample collector.

Workflow:
  1. Start apriltag_zed.launch.py so /apriltag/detections is publishing.
  2. Put the arm in freedrive / manual mode so you can pose it by hand
     (or move it via RViz / MoveIt — anything that updates TF).
  3. Run this script. Each press of <Enter> captures one (T_base_gripper,
     T_cam_target) pair. Press 'q' then <Enter> to stop and write the JSON.

Aim for 12–20 samples spanning translations and rotations of the arm.
Vary orientation more than position — pure-translation samples don't
constrain the rotation part of the AX=XB solve.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from apriltag_msgs.msg import AprilTagDetectionArray
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException


import math


# Minimum motion between consecutive captures. Below these, the sample is
# almost certainly a duplicate / stale-tf artefact and gets rejected.
MIN_TRANSLATION_M = 0.010   # 1 cm
MIN_ROTATION_DEG = 3.0


def transform_to_dict(t):
    return {
        "xyz": [t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z],
        "quat_xyzw": [t.transform.rotation.x,
                      t.transform.rotation.y,
                      t.transform.rotation.z,
                      t.transform.rotation.w],
        "frame_id": t.header.frame_id,
        "child_frame_id": t.child_frame_id,
        "stamp_sec": t.header.stamp.sec,
        "stamp_nsec": t.header.stamp.nanosec,
    }


def pose_delta(prev: dict, curr: dict) -> tuple[float, float]:
    """Translation distance (m) and rotation angle (deg) between two poses."""
    dx = curr["xyz"][0] - prev["xyz"][0]
    dy = curr["xyz"][1] - prev["xyz"][1]
    dz = curr["xyz"][2] - prev["xyz"][2]
    t_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    # Quaternion dot -> angle. dot > 1 happens with float jitter; clamp.
    q1, q2 = prev["quat_xyzw"], curr["quat_xyzw"]
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = max(-1.0, min(1.0, dot))
    r_deg = math.degrees(2.0 * math.acos(dot))
    return t_dist, r_deg


class Collector(Node):
    def __init__(self, base_frame: str, gripper_frame: str,
                 camera_frame: str, tag_frame: str):
        super().__init__("handeye_collector")
        self.base_frame = base_frame
        self.gripper_frame = gripper_frame
        self.camera_frame = camera_frame
        self.tag_frame = tag_frame

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.last_detection_ids = []
        self.create_subscription(AprilTagDetectionArray, "/apriltag/detections",
                                 self._on_detections, qos)

    def _on_detections(self, msg: AprilTagDetectionArray):
        self.last_detection_ids = [d.id for d in msg.detections]

    def capture(self):
        # Use latest available time for both lookups; small skew is fine
        # since the arm is held still during a capture.
        now = rclpy.time.Time()
        timeout = Duration(seconds=1.0)

        try:
            t_base_grip = self.tf_buffer.lookup_transform(
                self.base_frame, self.gripper_frame, now, timeout)
        except (LookupException, ExtrapolationException) as e:
            return None, f"TF {self.base_frame} <- {self.gripper_frame} unavailable: {e}"

        try:
            t_cam_tag = self.tf_buffer.lookup_transform(
                self.camera_frame, self.tag_frame, now, timeout)
        except (LookupException, ExtrapolationException) as e:
            return None, f"TF {self.camera_frame} <- {self.tag_frame} unavailable: {e}"

        return (transform_to_dict(t_base_grip), transform_to_dict(t_cam_tag)), None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("handeye_samples.json"),
                   help="Output JSON path (default: handeye_samples.json)")
    p.add_argument("--base-frame", default="link_base",
                   help="Robot base frame (default: link_base — xarm convention).")
    p.add_argument("--gripper-frame", default="gripper",
                   help="Frame rigidly attached to the arm tip; the joint we'll edit "
                        "is gripper -> zed. Default: gripper.")
    p.add_argument("--camera-frame", default="zed_left_camera_optical_frame",
                   help="Optical frame the tag pose is expressed in (must match the "
                        "camera_info.frame_id used by the detector).")
    p.add_argument("--tag-frame", default="tag36h11_0",
                   help="Frame name apriltag_ros assigns to the tag (matches "
                        "apriltag_zed.launch.py 'tag.frames' entry).")
    p.add_argument("--tag-size", type=float, default=0.133,
                   help="Physical tag size in meters — recorded in the JSON for traceability.")
    p.add_argument("--zed-link", default="zed",
                   help="The link that the gripper.xacro ZED joint produces as child; "
                        "the chain from here to --camera-frame is fixed by the ZED URDF "
                        "and is captured once so the solver can back out the joint origin.")
    args = p.parse_args()

    rclpy.init()
    node = Collector(args.base_frame, args.gripper_frame,
                     args.camera_frame, args.tag_frame)

    # Briefly spin so TF and detection topic warm up.
    print("[*] Waiting for TF and AprilTag detections...", file=sys.stderr)
    deadline = node.get_clock().now() + Duration(seconds=5.0)
    while rclpy.ok() and node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.tf_buffer.can_transform(
                args.camera_frame, args.tag_frame, rclpy.time.Time()):
            break

    # One-time snapshot of the invariant chain zed -> camera_optical_frame.
    # The solver uses this to back out the new gripper -> zed joint origin.
    try:
        t_zed_cam = node.tf_buffer.lookup_transform(
            args.zed_link, args.camera_frame, rclpy.time.Time(),
            Duration(seconds=2.0))
        zed_to_cam = transform_to_dict(t_zed_cam)
    except (LookupException, ExtrapolationException) as e:
        print(f"[!] Could not look up {args.zed_link} -> {args.camera_frame}: {e}",
              file=sys.stderr)
        print("    The solver will still produce T_gripper_camera but cannot",
              file=sys.stderr)
        print("    derive the gripper.xacro joint origin without this transform.",
              file=sys.stderr)
        zed_to_cam = None

    samples = []
    try:
        while rclpy.ok():
            # Keep TF buffer fresh between prompts.
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.05)

            seen = node.last_detection_ids
            prompt = (f"[{len(samples):2d}] tag seen: {seen!r:>20s}   "
                      "<Enter>=capture  q=quit > ")
            line = input(prompt).strip().lower()
            if line == "q":
                break

            # Re-spin once more so we capture the freshest tf.
            for _ in range(3):
                rclpy.spin_once(node, timeout_sec=0.05)

            result, err = node.capture()
            if err:
                print(f"    skipped: {err}")
                continue
            t_base_grip, t_cam_tag = result

            # Reject duplicates / stale-tf captures. The hand-eye solver
            # gets bad data when consecutive samples don't both move; in
            # particular when the gripper TF is stale (same as previous)
            # but the tag was re-detected, the AX=XB system is fed a
            # contradiction.
            if samples:
                bg_dt, bg_dr = pose_delta(samples[-1]["T_base_gripper"], t_base_grip)
                ct_dt, ct_dr = pose_delta(samples[-1]["T_cam_target"], t_cam_tag)
                if bg_dt < MIN_TRANSLATION_M and bg_dr < MIN_ROTATION_DEG:
                    print(f"    rejected: gripper barely moved since last "
                          f"capture (Δt={bg_dt*1000:.1f} mm, Δr={bg_dr:.1f}°). "
                          "Pose the arm noticeably differently before "
                          "capturing.")
                    continue
                if ct_dt < MIN_TRANSLATION_M and ct_dr < MIN_ROTATION_DEG:
                    print(f"    rejected: tag pose in camera barely moved "
                          f"(Δt={ct_dt*1000:.1f} mm, Δr={ct_dr:.1f}°). The "
                          "detector may not have produced a fresh detection; "
                          "wait a beat and try again.")
                    continue
                print(f"    captured. Δgripper={bg_dt*1000:.1f} mm / {bg_dr:.1f}°, "
                      f"Δtag-in-cam={ct_dt*1000:.1f} mm / {ct_dr:.1f}°")
            else:
                print(f"    captured (first sample). gripper xyz = "
                      f"{[round(v, 4) for v in t_base_grip['xyz']]}")

            samples.append({
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "T_base_gripper": t_base_grip,
                "T_cam_target": t_cam_tag,
            })
    except (EOFError, KeyboardInterrupt):
        print()

    out = {
        "tag_family": "36h11",
        "tag_id": 0,
        "tag_size_m": args.tag_size,
        "frames": {
            "base": args.base_frame,
            "gripper": args.gripper_frame,
            "camera_optical": args.camera_frame,
            "tag": args.tag_frame,
            "zed_link": args.zed_link,
        },
        "zed_to_camera_optical": zed_to_cam,
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[*] Wrote {len(samples)} samples to {args.out}", file=sys.stderr)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
