#!/usr/bin/env python3
"""
Full pick pipeline:

    Anywhere → PREGRASP → MPC → LIFT_UP → DROP → ALIGN_TOP →
        ALIGN_APPROACH → OPEN_TWIST → done

Steps in detail:
  1. Plan + execute a MoveIt joint goal to PREGRASP_JOINTS (regardless of
     where the arm starts). Joint goals always have valid IK, so the
     OMPL/IK headaches that bite Cartesian goals are avoided.
  2. Spawn mpc_vs_pick.py as a subprocess. It drives the gripper into
     alignment with the bottle, fires the gripper-close service via
     /xarm/set_tgpio_digital, and exits cleanly.
  3. LIFT_UP — joint goal that clears the table after the close.
  4. DROP — joint goal that brings the bottle over the workspace where
     the second ZED can see its lid (carrying the bottle).
  5. ALIGN_TOP — spawn align_top_tag.py as a subprocess. It detects
     AprilTag id 13 on the bottle lid in the second ZED's image and
     publishes Cartesian twist (vx/vy) on /servo_node/delta_twist_cmds
     until the tag sits at the saved pixel target (./align_target.yaml).
  6. ALIGN_APPROACH — spawn align_approach_tag.py. Two-stage 1-D
     approach in base-X using AprilTag pixel sizes:
       - starts tracking tag id 2 (FAR — visible from far away),
       - switches to tag id 1 (NEAR — precision goal) as soon as
         id 1 is reliably visible, and stays with it.
     Targets for both tags come from ./approach_target.yaml (capture
     with `align_approach_tag.py --save-target` from the engaged
     final pose; partial saves are merged).
  7. OPEN_TWIST — two-part opener motion:
       (a) MoveIt joint goal that rotates joint 6 by --twist-deg
           (default +30°) relative to the current pose.
       (b) Servo-driven Cartesian back-off in base-X by --back-off-m
           (default 5 cm) so the bottle clears the opener.
     Override with --open-twist J1..J6 to use an absolute joint pose
     instead of the relative twist. Flip the back-off sign with
     --back-off-invert if needed.

Requires bringup.launch.py already running so that move_group, Servo,
and tag_detector are up.

The two hardcoded joint configurations are radians, 6 values for
xArm6 joints 1..6. Edit the constants below for your setup, or
override on the CLI.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from pymoveit2 import MoveIt2
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, Empty


class JointStateCache:
    """Caches the latest /joint_states message so we can read it on demand."""

    def __init__(self, node: Node, joint_names: list[str]):
        self.joint_names = joint_names
        self._latest: JointState | None = None
        node.create_subscription(JointState, "/joint_states",
                                 self._cb, 10)

    def _cb(self, msg: JointState) -> None:
        self._latest = msg

    def wait(self, timeout: float = 3.0) -> list[float] | None:
        end = time.time() + timeout
        while time.time() < end:
            msg = self._latest
            if msg is not None:
                idx = {n: i for i, n in enumerate(msg.name)}
                if all(jn in idx for jn in self.joint_names):
                    return [float(msg.position[idx[jn]])
                            for jn in self.joint_names]
            time.sleep(0.05)
        return None


# Joint angles in radians for joints 1..6.
# These are placeholders — update to whatever your demo uses.
# Tip: read /joint_states with the arm in your desired pose to get the values.
PREGRASP_JOINTS = [-1.7802, -0.5061, -0.3316, +3.0194, +0.6981, +0.8727]
# LIFT_UP runs right after the gripper closes. Pick a pose that's just
# above PREGRASP — basically lift the arm straight up by 5–10 cm — so we
# clear the table before swinging to DROP.
LIFT_UP_JOINTS  = [-1.7802, -0.7000, -0.3316, +3.0194, +0.6981, +0.8727]
DROP_JOINTS     = [-1.5708, -0.4363, -0.5952, -2.8623, +0.5585, -0.5061]
# OPEN_TWIST is the "after the cap is in the opener" pose: arm pulled
# back from the opener slightly AND joint 6 rotated ~30° to pop the cap
# off. The step reads the current /joint_states at runtime and rotates
# j6 by --twist-deg (default -30°); if you also need a backwards motion,
# update this constant from /joint_states and pass it via --open-twist.
# Placeholder = engaged pose (DROP) with j6 -= 30° (≈0.524 rad).
OPEN_TWIST_JOINTS = [-1.5708, -0.4363, -0.5952, -2.8623, +0.5585, -1.0297]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--goal", type=Path, default=Path("goal_pose.yaml"),
                   help="Goal YAML for mpc_vs_pick.py.")
    p.add_argument("--mpc-script", type=Path,
                   default=Path("/workspace/src/visual_servoing/mpc_vs_pick.py"),
                   help="Path to the MPC controller script.")
    p.add_argument("--mpc-args", default="--v-max 0.15 --w-max 0.15",
                   help="Extra args forwarded to mpc_vs_pick.py "
                        "(default 0.15 m/s linear, 0.15 rad/s angular).")
    p.add_argument("--align-script", type=Path,
                   default=Path("/workspace/src/visual_servoing/align_top_tag.py"),
                   help="Path to align_top_tag.py (step 5).")
    p.add_argument("--align-args",
                   default="--gain 0.001 --max-speed 0.2 --tol-px 25 "
                           "--invert-x --invert-y",
                   help="Extra args forwarded to align_top_tag.py. The "
                        "default --invert-x --invert-y matches the current "
                        "second-ZED mounting; remove them if you remount it. "
                        "align_top_tag.py auto-loads ./align_target.yaml if "
                        "present — capture one with "
                        "`align_top_tag.py --save-target`.")
    p.add_argument("--approach-script", type=Path,
                   default=Path("/workspace/src/visual_servoing/align_approach_tag.py"),
                   help="Path to align_approach_tag.py (step 6).")
    p.add_argument("--approach-args",
                   default="--gain 0.02 --max-speed 0.3 --tol-px 5 "
                           "--near-max-speed 0.05 --gain-z 0.002 "
                           "--max-speed-z 0.05 --tol-v-px 10",
                   help="Extra args forwarded to align_approach_tag.py. "
                        "Phase 1 (FAR) runs at --max-speed; phase 2 (NEAR) "
                        "is capped at --near-max-speed so X and Z arrive "
                        "at the saved target together. Add --invert-x / "
                        "--invert-z if a sign is wrong. Requires "
                        "./approach_target.yaml with far_pixel_size + "
                        "near_pixel_size + near_pixel_v — capture via "
                        "`--save-far` and `--save-near`.")
    p.add_argument("--pregrasp", nargs=6, type=float, default=PREGRASP_JOINTS,
                   metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                   help="6 joint angles (rad) for the pre-grasp pose.")
    p.add_argument("--liftup", nargs=6, type=float, default=LIFT_UP_JOINTS,
                   metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                   help="6 joint angles (rad) for the LIFT-UP pose used "
                        "immediately after the gripper closes (clears the "
                        "table before swinging to DROP).")
    p.add_argument("--drop", nargs=6, type=float, default=DROP_JOINTS,
                   metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                   help="6 joint angles (rad) for the final drop pose.")
    p.add_argument("--planning-group", default="xarm6")
    p.add_argument("--base-link", default="link_base")
    p.add_argument("--ee-link", default="link_eef")
    p.add_argument("--velocity-scale", type=float, default=0.2,
                   help="MoveIt scaling (0–1). 0.2 = slow & safe.")
    p.add_argument("--accel-scale", type=float, default=0.2)
    p.add_argument("--skip-pregrasp", action="store_true",
                   help="Skip step 1 — useful if the arm is already at the "
                        "pregrasp pose and you just want to run MPC + drop.")
    p.add_argument("--skip-liftup", action="store_true",
                   help="Skip the lift-up step (go straight to DROP).")
    p.add_argument("--skip-drop", action="store_true",
                   help="Skip the final drop — pipeline ends after the lift-up "
                        "(or after gripper close if --skip-liftup is also set).")
    p.add_argument("--skip-align", action="store_true",
                   help="Skip the ALIGN_TOP step (no second-ZED image centering).")
    p.add_argument("--skip-approach", action="store_true",
                   help="Skip the ALIGN_APPROACH step (no depth approach).")
    p.add_argument("--open-twist", nargs=6, type=float, default=None,
                   metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                   help="If set, OPEN_TWIST plans to these 6 absolute "
                        "joint angles (rad). Otherwise the step reads "
                        "the current joints and rotates j6 by "
                        "--twist-deg.")
    p.add_argument("--twist-deg", type=float, default=+30.0,
                   help="Degrees to rotate joint 6 in OPEN_TWIST (default "
                        "+30°). Only used when --open-twist is not given. "
                        "Flip the sign if your setup needs the wrist to "
                        "twist the other way.")
    p.add_argument("--back-off-m", type=float, default=0.05,
                   help="Cartesian back-off distance (m) in base-X "
                        "after the wrist twist, to pull the bottle out "
                        "of the opener. 0 disables. Default 0.05 m.")
    p.add_argument("--back-off-speed", type=float, default=0.03,
                   help="Speed of the back-off motion in m/s. Default 0.03.")
    p.add_argument("--back-off-invert", action="store_true",
                   help="Flip the back-off direction (use if your "
                        "approach uses --invert-x — back-off then needs "
                        "+X instead of -X).")
    p.add_argument("--skip-open-twist", action="store_true",
                   help="Skip the OPEN_TWIST step (no opener motion).")
    return p.parse_args()


def move_to(moveit2: MoveIt2, joints: list[float], label: str,
            tries: int = 3) -> bool:
    for attempt in range(1, tries + 1):
        suffix = f" (attempt {attempt}/{tries})" if tries > 1 else ""
        print(f"[*] {label}: planning to joints "
              f"[{', '.join(f'{j:+.3f}' for j in joints)}]{suffix} ...",
              flush=True)
        moveit2.move_to_configuration(joints)
        ok = moveit2.wait_until_executed()
        if ok:
            print(f"[*] {label}: reached.", flush=True)
            return True
        if attempt < tries:
            print(f"[!] {label}: attempt {attempt} failed, retrying after 1 s.",
                  file=sys.stderr, flush=True)
            time.sleep(1.0)
    print(f"[!] {label}: failed after {tries} attempts.", file=sys.stderr,
          flush=True)
    return False


def call_empty(node: Node, service_name: str, timeout: float = 3.0) -> bool:
    """Call a std_srvs/Empty service. Used for /clear_octomap."""
    cli = node.create_client(Empty, service_name)
    if not cli.wait_for_service(timeout_sec=timeout):
        print(f"[!] {service_name} not available after {timeout}s — skipping.",
              file=sys.stderr, flush=True)
        return False
    future = cli.call_async(Empty.Request())
    end = time.time() + timeout
    while not future.done() and time.time() < end:
        time.sleep(0.05)
    if not future.done():
        print(f"[!] {service_name} call timed out.", file=sys.stderr, flush=True)
        return False
    print(f"[*] {service_name}: returned.", flush=True)
    return True


def call_trigger(node: Node, service_name: str, timeout: float = 3.0) -> bool:
    """Call a std_srvs/Trigger service synchronously. Returns success."""
    cli = node.create_client(Trigger, service_name)
    if not cli.wait_for_service(timeout_sec=timeout):
        print(f"[!] {service_name} not available after {timeout}s — skipping.",
              file=sys.stderr)
        return False
    future = cli.call_async(Trigger.Request())
    # We're running a MultiThreadedExecutor in another thread, so .result()
    # will block on the future correctly.
    end = time.time() + timeout
    while not future.done() and time.time() < end:
        time.sleep(0.05)
    if not future.done():
        print(f"[!] {service_name} call timed out.", file=sys.stderr)
        return False
    res = future.result()
    print(f"[*] {service_name}: success={res.success} message={res.message!r}")
    return bool(res.success)


def main() -> int:
    args = parse_args()

    rclpy.init()
    node = Node("pick_pipeline")
    cb_group = ReentrantCallbackGroup()
    moveit2 = MoveIt2(
        node=node,
        joint_names=[f"joint{i}" for i in range(1, 7)],
        base_link_name=args.base_link,
        end_effector_name=args.ee_link,
        group_name=args.planning_group,
        callback_group=cb_group,
    )
    moveit2.max_velocity = float(args.velocity_scale)
    moveit2.max_acceleration = float(args.accel_scale)

    joint_cache = JointStateCache(
        node, [f"joint{i}" for i in range(1, 7)])

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # ── Step 1: pregrasp ─────────────────────────────────────────────
        if not args.skip_pregrasp:
            print("[*] Clearing octomap before planning...", flush=True)
            call_empty(node, "/clear_octomap")
            if not move_to(moveit2, args.pregrasp, "Step 1/7  PREGRASP"):
                return 1
        else:
            print("[*] Step 1/7  PREGRASP: skipped", flush=True)

        # ── Step 2: visual servoing MPC (as subprocess) ──────────────────
        print("[*] Step 2/7  MPC: starting visual servoing...", flush=True)
        mpc_cmd = ["python3", str(args.mpc_script),
                   "--goal", str(args.goal)] + args.mpc_args.split()
        print("    cmd:", " ".join(mpc_cmd), flush=True)
        res = subprocess.run(mpc_cmd)
        print(f"[*] Step 2/7  MPC: subprocess exited (returncode={res.returncode}).",
              flush=True)
        if res.returncode != 0:
            print(f"[!] MPC exited with non-zero code {res.returncode}",
                  file=sys.stderr, flush=True)
            return res.returncode

        # ── Steps 3 + 4 share Servo-stop / restart, so do both inside ─────
        need_moveit = not (args.skip_liftup and args.skip_drop)
        if need_moveit:
            # MoveIt Servo and the trajectory controller share the same
            # /xarm6_traj_controller. Servo keeps the controller "claimed"
            # even when no twist is coming in, so a follow-up MoveIt plan
            # silently fails to execute. Stop Servo first, do all the
            # MoveIt steps, then restart Servo at the end.
            print("[*] Stopping Servo so MoveIt can drive the controller...",
                  flush=True)
            call_trigger(node, "/servo_node/stop_servo")
            settle = 2.0
            print(f"[*] Waiting {settle}s for Servo to release the controller...",
                  flush=True)
            time.sleep(settle)

            # Step 3: LIFT_UP (clear the table before swinging to drop).
            if not args.skip_liftup:
                print("[*] Clearing octomap before LIFT_UP plan...", flush=True)
                call_empty(node, "/clear_octomap")
                if not move_to(moveit2, args.liftup, "Step 3/7  LIFT_UP"):
                    call_trigger(node, "/servo_node/start_servo")
                    return 1
            else:
                print("[*] Step 3/7  LIFT_UP: skipped", flush=True)

            # Step 4: DROP.
            if not args.skip_drop:
                print("[*] Clearing octomap before DROP plan...", flush=True)
                call_empty(node, "/clear_octomap")
                if not move_to(moveit2, args.drop, "Step 4/7  DROP"):
                    call_trigger(node, "/servo_node/start_servo")
                    return 1
            else:
                print("[*] Step 4/7  DROP: skipped", flush=True)

            print("[*] Restarting Servo so it can accept the ALIGN_TOP twist.",
                  flush=True)
            call_trigger(node, "/servo_node/start_servo")
        else:
            print("[*] LIFT_UP and DROP both skipped — Servo still up from "
                  "bringup, proceeding to ALIGN_TOP.", flush=True)

        # ── Step 5: ALIGN_TOP (second-ZED image centering) ───────────────
        if not args.skip_align:
            # Small settle so Servo has actually re-claimed the controller
            # before we start streaming twist into it.
            time.sleep(1.0)
            print("[*] Step 5/7  ALIGN_TOP: starting image centering...",
                  flush=True)
            align_cmd = (["python3", str(args.align_script)]
                         + args.align_args.split())
            print("    cmd:", " ".join(align_cmd), flush=True)
            res = subprocess.run(align_cmd)
            print(f"[*] Step 5/7  ALIGN_TOP: subprocess exited "
                  f"(returncode={res.returncode}).", flush=True)
            if res.returncode != 0:
                print(f"[!] ALIGN_TOP exited with non-zero code {res.returncode}",
                      file=sys.stderr, flush=True)
                return res.returncode
        else:
            print("[*] Step 5/7  ALIGN_TOP: skipped", flush=True)

        # ── Step 6: ALIGN_APPROACH (base-X drive to target pixel size) ───
        if not args.skip_approach:
            print("[*] Step 6/7  ALIGN_APPROACH: approaching to saved tag size...",
                  flush=True)
            approach_cmd = (["python3", str(args.approach_script)]
                            + args.approach_args.split())
            print("    cmd:", " ".join(approach_cmd), flush=True)
            res = subprocess.run(approach_cmd)
            print(f"[*] Step 6/7  ALIGN_APPROACH: subprocess exited "
                  f"(returncode={res.returncode}).", flush=True)
            if res.returncode != 0:
                print(f"[!] ALIGN_APPROACH exited with non-zero code "
                      f"{res.returncode}", file=sys.stderr, flush=True)
                return res.returncode
        else:
            print("[*] Step 6/7  ALIGN_APPROACH: skipped", flush=True)

        # ── Step 7: OPEN_TWIST (rotate wrist, MoveIt joint plan) ─────────
        if not args.skip_open_twist:
            # Servo was running for ALIGN_APPROACH; MoveIt and Servo
            # share the trajectory controller, so stop Servo first.
            print("[*] Stopping Servo so MoveIt can drive the controller...",
                  flush=True)
            call_trigger(node, "/servo_node/stop_servo")
            settle = 2.0
            print(f"[*] Waiting {settle}s for Servo to release the controller...",
                  flush=True)
            time.sleep(settle)

            if args.open_twist is not None:
                target = list(args.open_twist)
                src = "args.open_twist (absolute)"
            else:
                current = joint_cache.wait(timeout=3.0)
                if current is None:
                    print("[!] OPEN_TWIST: couldn't read /joint_states.",
                          file=sys.stderr, flush=True)
                    return 1
                twist_rad = math.radians(float(args.twist_deg))
                target = list(current[:5]) + [current[5] + twist_rad]
                src = (f"current_joints + j6{'+' if twist_rad >= 0 else ''}"
                       f"{math.degrees(twist_rad):.1f}°")

            print(f"[*] Step 7/7  OPEN_TWIST: target = "
                  f"[{', '.join(f'{j:+.3f}' for j in target)}]  ({src})",
                  flush=True)
            print("[*] Clearing octomap before OPEN_TWIST plan...", flush=True)
            call_empty(node, "/clear_octomap")
            if not move_to(moveit2, target, "Step 7/7  OPEN_TWIST (twist)"):
                return 1

            # Back off via Servo (Cartesian -X) so the bottle clears the
            # opener after the wrist twist.
            if args.back_off_m > 0.0:
                speed = float(args.back_off_speed)
                back_vx = -speed if not args.back_off_invert else +speed
                duration = float(args.back_off_m) / max(speed, 1e-3)
                print(f"[*] Step 7/7  OPEN_TWIST (back-off): "
                      f"{args.back_off_m*100:.1f}cm at {speed:.2f}m/s "
                      f"(vx={back_vx:+.3f}) for {duration:.2f}s",
                      flush=True)
                print("[*] Restarting Servo for the back-off motion...",
                      flush=True)
                call_trigger(node, "/servo_node/start_servo")
                time.sleep(0.5)
                pub = node.create_publisher(
                    TwistStamped, "/servo_node/delta_twist_cmds", 10)
                rate_hz = 20.0
                end = time.time() + duration
                while time.time() < end:
                    msg = TwistStamped()
                    msg.header.stamp = node.get_clock().now().to_msg()
                    msg.header.frame_id = args.base_link
                    msg.twist.linear.x = float(back_vx)
                    pub.publish(msg)
                    time.sleep(1.0 / rate_hz)
                # Send a couple of zero twists so Servo halts cleanly.
                for _ in range(3):
                    z = TwistStamped()
                    z.header.stamp = node.get_clock().now().to_msg()
                    z.header.frame_id = args.base_link
                    pub.publish(z)
                    time.sleep(0.05)
                print("[*] Back-off complete; stopping Servo.", flush=True)
                call_trigger(node, "/servo_node/stop_servo")
        else:
            print("[*] Step 7/7  OPEN_TWIST: skipped", flush=True)

        print("\n[*] Pick pipeline complete.")
        return 0
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
