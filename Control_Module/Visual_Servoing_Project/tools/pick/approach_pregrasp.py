#!/usr/bin/env python3
"""
Coarse approach: use MoveIt to bring the gripper to a pre-grasp pose
that's close enough to the demonstrated goal for visual servoing to
take over.

Pipeline:
  1. Read goal_pose.yaml  (T_gripper_tag at the moment of grasp closure)
  2. Look up T_base_tag from tf2  (where the tag is in base coords NOW)
  3. Compute T_base_gripper_target = T_base_tag · (T_gripper_tag_goal)⁻¹
  4. Apply --standoff: back off along the gripper approach axis so the
     arm stops a few cm short. Visual servoing finishes the last bit.
  5. Send a MoveIt plan-and-execute to that pose via pymoveit2.

Run BEFORE mpc_vs_pick.py. The xArm driver, MoveIt, ZED, tag_detector
and Servo should all already be up.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from pymoveit2 import MoveIt2
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException


# ---------- pose math helpers (same conventions as everywhere else) ----------

def quat_to_R(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R: np.ndarray) -> list[float]:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return [(R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
                0.25 * s]
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return [0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[2, 1] - R[1, 2]) / s]
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return [(R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
                (R[0, 2] - R[2, 0]) / s]
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return [(R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
            (R[1, 0] - R[0, 1]) / s]


def to_T(xyz: list[float], quat_xyzw: list[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(quat_xyzw)
    T[:3, 3] = xyz
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--goal", type=Path, default=Path("goal_pose.yaml"))
    p.add_argument("--base-frame", default="link_base",
                   help="The robot base frame (MoveIt planning frame).")
    p.add_argument("--gripper-frame", default="gripper",
                   help="The link the goal is expressed in.")
    p.add_argument("--ee-frame", default="link_eef",
                   help="MoveIt end-effector frame for the planning group.")
    p.add_argument("--tag-frame", default="tag36h11_0")
    p.add_argument("--standoff", type=float, default=0.05,
                   help="Distance (m) to stop short of the demonstrated goal, "
                        "measured along the gripper's local Z (camera-forward) "
                        "axis. Default 5 cm — leaves room for MPC to finish.")
    p.add_argument("--planning-group", default="xarm6")
    p.add_argument("--velocity-scale", type=float, default=0.15,
                   help="MoveIt velocity scaling (0-1). Default 0.15 — slow.")
    p.add_argument("--accel-scale", type=float, default=0.15)
    p.add_argument("--planning-tries", type=int, default=5,
                   help="If the first plan attempt fails, retry up to this many times.")
    args = p.parse_args()

    goal_data = yaml.safe_load(args.goal.open())
    if goal_data["gripper_frame"] != args.gripper_frame:
        print(f"[warn] goal_pose.yaml stored gripper_frame={goal_data['gripper_frame']!r} "
              f"but --gripper-frame is {args.gripper_frame!r}", file=sys.stderr)
    T_gripper_tag_goal = to_T(goal_data["xyz"], goal_data["quat_xyzw"])

    rclpy.init()
    node = Node("approach_pregrasp")
    buf = Buffer()
    TransformListener(buf, node)

    # Start a spinner so tf buffer fills and pymoveit2 action clients work.
    callback_group = ReentrantCallbackGroup()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Wait for tag tf.
    print(f"[*] Waiting for {args.base_frame} -> {args.tag_frame}...")
    deadline = node.get_clock().now() + Duration(seconds=10.0)
    while rclpy.ok() and node.get_clock().now() < deadline:
        if buf.can_transform(args.base_frame, args.tag_frame, rclpy.time.Time()):
            break
        rclpy.spin_once(node, timeout_sec=0.1)

    try:
        tf_bt = buf.lookup_transform(args.base_frame, args.tag_frame,
                                     rclpy.time.Time(), Duration(seconds=2.0))
    except (LookupException, ExtrapolationException) as e:
        sys.exit(f"[!] Cannot look up {args.base_frame} -> {args.tag_frame}: {e}")

    T_base_tag = to_T(
        [tf_bt.transform.translation.x, tf_bt.transform.translation.y, tf_bt.transform.translation.z],
        [tf_bt.transform.rotation.x, tf_bt.transform.rotation.y,
         tf_bt.transform.rotation.z, tf_bt.transform.rotation.w])

    # T_gripper_tag = T_gripper_base * T_base_tag
    # => at the goal: T_base_gripper_target = T_base_tag * inv(T_gripper_tag_goal)
    T_base_gripper_target = T_base_tag @ inv_T(T_gripper_tag_goal)

    # Stand-off: back off along the gripper's local -Z (i.e., away from the tag).
    if args.standoff > 0:
        offset_local = np.array([0.0, 0.0, -float(args.standoff), 1.0])
        offset_in_base = T_base_gripper_target @ offset_local
        T_base_gripper_target[:3, 3] = offset_in_base[:3]

    pos = T_base_gripper_target[:3, 3].tolist()
    quat = R_to_quat(T_base_gripper_target[:3, :3])

    print(f"[*] Tag in base:           xyz={T_base_tag[:3,3].round(4).tolist()}")
    print(f"[*] Target gripper in base (standoff={args.standoff*1000:.0f} mm):")
    print(f"      xyz:        {[round(v,4) for v in pos]}")
    print(f"      quat xyzw:  {[round(v,4) for v in quat]}")

    # pymoveit2 wiring.
    joint_names = [f"joint{i}" for i in range(1, 7)]
    moveit2 = MoveIt2(
        node=node,
        joint_names=joint_names,
        base_link_name=args.base_frame,
        end_effector_name=args.ee_frame,
        group_name=args.planning_group,
        callback_group=callback_group,
    )
    moveit2.max_velocity = float(args.velocity_scale)
    moveit2.max_acceleration = float(args.accel_scale)

    # Plan + execute. pymoveit2 wraps the action; we retry on planning failure.
    print(f"[*] Planning to pre-grasp pose (group={args.planning_group}, "
          f"velocity_scale={args.velocity_scale})...")
    success = False
    for attempt in range(args.planning_tries):
        moveit2.move_to_pose(
            position=pos,
            quat_xyzw=quat,
            tolerance_position=0.01,
            tolerance_orientation=0.05,
            cartesian=False,
        )
        ok = moveit2.wait_until_executed()
        if ok:
            success = True
            break
        print(f"    attempt {attempt+1}/{args.planning_tries} failed; retrying...")

    if not success:
        print("[!] All planning attempts failed. Things to try:",
              file=sys.stderr)
        print("    - Reposition the arm so the tag is visible and not far behind it.",
              file=sys.stderr)
        print("    - Lower --standoff to bring the target closer.",
              file=sys.stderr)
        print("    - Loosen tolerances by editing this script.",
              file=sys.stderr)
        sys.exit(1)

    print("[*] Pre-grasp pose reached. Now run mpc_vs_pick.py to align.")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
