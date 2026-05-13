#!/usr/bin/env python3
"""
Capture the current T_gripper_tag36h11_0 as the visual-servoing goal pose.

Run this with the arm physically posed at the grasp position you want to
reproduce and the AprilTag visible to the ZED. The resulting YAML is
loaded by mpc_vs_pick.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("goal_pose.yaml"),
                   help="Output YAML path (default: goal_pose.yaml).")
    p.add_argument("--gripper-frame", default="gripper",
                   help="Frame to express the tag pose in (default: gripper).")
    p.add_argument("--tag-frame", default="tag36h11_0",
                   help="Tag frame published by apriltag_ros (default: tag36h11_0).")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="Seconds to wait for the tag to appear (default: 15).")
    args = p.parse_args()

    rclpy.init()
    node = Node("goal_recorder")
    buf = Buffer()
    TransformListener(buf, node)

    print(f"[*] Waiting up to {args.timeout:.0f}s for "
          f"{args.gripper_frame} -> {args.tag_frame}...", file=sys.stderr)
    deadline = node.get_clock().now() + Duration(seconds=args.timeout)
    while rclpy.ok() and node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if buf.can_transform(args.gripper_frame, args.tag_frame, rclpy.time.Time()):
            break

    try:
        t = buf.lookup_transform(args.gripper_frame, args.tag_frame,
                                 rclpy.time.Time(), Duration(seconds=2.0))
    except TransformException as e:
        sys.exit(f"\n[!] Could not look up {args.gripper_frame} -> {args.tag_frame}: {e}\n"
                 "    Is the AprilTag detector running and seeing the tag?\n")

    xyz = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
    quat = [t.transform.rotation.x, t.transform.rotation.y,
            t.transform.rotation.z, t.transform.rotation.w]

    data = {
        "gripper_frame": args.gripper_frame,
        "tag_frame": args.tag_frame,
        "xyz": xyz,
        "quat_xyzw": quat,
    }
    args.out.write_text(yaml.safe_dump(data, sort_keys=False))

    print(f"\n[*] Wrote {args.out}")
    print(f"    {args.gripper_frame} -> {args.tag_frame}:")
    print(f"    xyz  (m):   [{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}]")
    print(f"    quat xyzw:  [{quat[0]:+.4f}, {quat[1]:+.4f}, "
          f"{quat[2]:+.4f}, {quat[3]:+.4f}]")
    print(f"    distance (gripper -> tag): "
          f"{(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)**0.5*1000:.1f} mm")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
