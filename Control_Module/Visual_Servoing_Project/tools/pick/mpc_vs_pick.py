#!/usr/bin/env python3
"""
MPC-based visual servoing pick (eye-in-hand AprilTag).

Reads the goal T_gripper_tag (from record_goal_pose.py), gets the live
tag-in-gripper pose from tf2, runs a small box-constrained MPC each
cycle, and publishes a Cartesian twist to /servo_node/delta_twist_cmds
so MoveIt Servo drives the arm.

Model (linearized about the current state, in the gripper frame):
  state e   = (Δp, Δr)            6-vector, tag-in-gripper minus goal
  control u = (v, ω)              gripper twist in gripper frame
  dynamics  e_{k+1} = e_k + dt · M · u_k
              M = [[-I_3,  [p_current]_×],
                   [ 0,         -I_3   ]]
  cost      Σ_{k=1..N} e_k' Q e_k + Σ_{k=0..N-1} u_k' R u_k + e_N' Q_f e_N
  bounds    |v|_i ≤ v_max, |ω|_i ≤ ω_max

When ‖Δp‖ < t_threshold AND ‖Δr‖ < r_threshold, prints "CLOSE NOW" and
stops moving. Gripper closure is manual (user-triggered) for this first
cut — see --help.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize, Bounds

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from tf2_ros import Buffer, TransformListener, TransformException

from xarm_msgs.srv import SetDigitalIO, MoveVelocity, SetInt16


# ---------- math helpers ----------

def quat_to_R(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> 3-vector r = axis * angle (rotation vector)."""
    trace = float(np.clip((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2.0, -1.0, 1.0))
    theta = np.arccos(trace)
    if theta < 1e-7:
        return np.zeros(3)
    if np.pi - theta < 1e-3:
        # Near 180°: extract axis from the diagonal-dominant column of (R + I)/2
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diagonal(M)))
        axis = M[:, i] / np.sqrt(max(M[i, i], 1e-9))
        return theta * axis
    factor = theta / (2.0 * np.sin(theta))
    return factor * np.array([R[2, 1] - R[1, 2],
                              R[0, 2] - R[2, 0],
                              R[1, 0] - R[0, 1]])


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


# ---------- MPC ----------

class MPC:
    """
    Box-constrained linear MPC, dense formulation.

    Decision variable z = [u_0; u_1; ...; u_{N-1}]  (size 6N).
    Forward simulation matrices built once per call (since M depends on
    the current state). At horizon N=8 and 20 Hz this stays well under
    one millisecond per solve on the Orin.
    """

    def __init__(self, N: int, dt: float, Q: np.ndarray, R: np.ndarray,
                 Q_f: np.ndarray, v_max: float, w_max: float):
        self.N = N
        self.dt = dt
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self._v_max = float(v_max)
        self._w_max = float(w_max)
        self._set_bounds(self._v_max, self._w_max)
        self.warm = np.zeros(6 * N)

    def _set_bounds(self, v_max: float, w_max: float) -> None:
        bnd_one = [v_max] * 3 + [w_max] * 3
        self.lb = np.array([-b for b in bnd_one] * self.N)
        self.ub = np.array(bnd_one * self.N)

    def set_runtime_caps(self, v_max: float, w_max: float) -> None:
        """Adjust the per-axis velocity bounds for the next solve."""
        self._set_bounds(v_max, w_max)

    def solve(self, e0: np.ndarray, M: np.ndarray) -> np.ndarray:
        N, dt = self.N, self.dt
        # Build "rollout" propagators so that e_k = e0 + dt*M * (sum_{j<k} u_j).
        # Stack the cumulative sum operator as a (6N x 6N) lower-triangular
        # block matrix of dt*M; then the trajectory of e_k for k=1..N is
        # E = (kron(L, dt*M)) z + (1_N ⊗ e0)
        L = np.tril(np.ones((N, N)))                # N x N
        big = np.kron(L, dt * M)                    # 6N x 6N (e_1..e_N stacked)
        e0_stack = np.tile(e0, N)                   # 6N

        # Quadratic cost on u-side:  z' diag(R,...,R) z
        Rblk = np.kron(np.eye(N), self.R)
        # Quadratic cost on x-side:  E' Qblk E  with Qblk = diag(Q,...,Q, Q_f)
        Qblk = np.kron(np.eye(N), self.Q)
        Qblk[-6:, -6:] = self.Q_f                   # terminal cost

        # J(z) = z' H z + 2 g' z + const
        H = Rblk + big.T @ Qblk @ big
        g = big.T @ Qblk @ e0_stack                 # linear term (1/2 factor folded)
        # Symmetrize to be safe.
        H = 0.5 * (H + H.T)

        def cost(z):
            return float(z @ H @ z + 2.0 * g @ z)

        def grad(z):
            return 2.0 * (H @ z + g)

        result = minimize(
            cost, self.warm, jac=grad, method="L-BFGS-B",
            bounds=Bounds(self.lb, self.ub),
            options={"maxiter": 50, "ftol": 1e-8, "gtol": 1e-6},
        )
        # Warm-start next call by shifting the solution one step.
        self.warm = np.concatenate([result.x[6:], np.zeros(6)])
        return result.x[:6].copy()


# ---------- XArm direct-velocity backend ----------

class XArmVelocityBackend:
    """Stream Cartesian velocity directly to the xArm via mode 5.

    Skips MoveIt Servo entirely. The xArm's onboard motion controller has
    its own smoothing (acceleration ramps) and is what UFactory designed
    for streaming velocity commands. Tradeoff: no MoveIt collision /
    joint-limit safety — the controller will accept anything we send up
    to firmware limits. Use low velocity caps when testing.
    """

    def __init__(self, node):
        self.node = node
        self.log = node.get_logger()
        self.mode_cli = node.create_client(SetInt16, "/xarm/set_mode")
        self.state_cli = node.create_client(SetInt16, "/xarm/set_state")
        self.cart_vel_cli = node.create_client(MoveVelocity,
                                               "/xarm/vc_set_cartesian_velocity")
        self._enabled = False

    def _call_int(self, cli, value: int, timeout: float = 3.0) -> bool:
        req = SetInt16.Request()
        req.data = int(value)
        if not cli.wait_for_service(timeout_sec=timeout):
            self.log.error(f"{cli.srv_name} not available")
            return False
        future = cli.call_async(req)
        # Spin the node ourselves while we wait — enable()/disable() can be
        # called from main() before the main spin loop starts, so we can't
        # rely on background spinning to deliver the response.
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        if not future.done():
            self.log.error(f"{cli.srv_name} timed out")
            return False
        res = future.result()
        self.log.info(f"{cli.srv_name}({value}) -> ret={res.ret} msg={res.message!r}")
        return int(res.ret) == 0

    def enable(self) -> bool:
        """Mode 5 = Cartesian online (velocity) mode; state 0 = ready."""
        if not self._call_int(self.mode_cli, 5):
            return False
        if not self._call_int(self.state_cli, 0):
            return False
        self._enabled = True
        self.log.info("XArm Cartesian velocity mode (mode 5) enabled.")
        return True

    def disable(self) -> None:
        """Restore mode 0 (position mode) so MoveIt can drive the arm again."""
        if not self._enabled:
            return
        self.stop()
        self._call_int(self.mode_cli, 0)
        self._call_int(self.state_cli, 0)
        self._enabled = False
        self.log.info("XArm position mode restored.")

    def send_velocity_base(self, v: np.ndarray, duration: float = 0.2) -> None:
        """Send a 6-DOF Cartesian velocity in BASE frame (mm/s, rad/s).

        xArm SDK expects linear velocity in mm/s (NOT m/s).
        """
        req = MoveVelocity.Request()
        req.speeds = [
            float(v[0]) * 1000.0,  # mm/s
            float(v[1]) * 1000.0,
            float(v[2]) * 1000.0,
            float(v[3]),           # rad/s
            float(v[4]),
            float(v[5]),
        ]
        req.is_sync = False
        req.is_tool_coord = False
        req.duration = float(duration)
        # Fire-and-forget; do not block the MPC tick on the round-trip.
        self.cart_vel_cli.call_async(req)

    def stop(self) -> None:
        self.send_velocity_base(np.zeros(6), duration=0.0)


# ---------- ROS node ----------

class MpcVSController(Node):
    def __init__(self, goal: dict, args: argparse.Namespace):
        super().__init__("mpc_vs_pick")
        self.args = args
        self.gripper_frame = goal["gripper_frame"]
        self.tag_frame = goal["tag_frame"]
        self.p_goal = np.array(goal["xyz"])
        self.R_goal = quat_to_R(goal["quat_xyzw"])

        self.twist_frame = args.twist_frame or self.gripper_frame

        self.buf = Buffer()
        TransformListener(self.buf, self)

        self.backend_name = str(args.backend)
        self.base_frame = str(args.base_frame)
        self.pub = self.create_publisher(TwistStamped, args.twist_topic, 10)

        if self.backend_name == "xarm":
            self.xarm = XArmVelocityBackend(self)
        else:
            self.xarm = None

        self.timer = self.create_timer(1.0 / args.rate, self._tick)
        self.dry_run = bool(args.dry_run)

        # Auto-close wiring
        self.auto_close = bool(args.auto_close) and not self.dry_run
        self.gripper_ionum = int(args.gripper_ionum)
        self.gripper_value = int(args.gripper_value)
        self.close_settle_time = float(args.close_settle_time)
        self.gripper_cli = self.create_client(SetDigitalIO, args.gripper_service)
        self.close_requested = False
        self.close_done_time: float | None = None  # wall-clock seconds, set after close fires

        Q = np.diag([args.q_pos] * 3 + [args.q_rot] * 3)
        R = np.diag([args.r_vel] * 3 + [args.r_omega] * 3)
        Q_f = Q * args.terminal_weight
        self.mpc = MPC(N=args.horizon, dt=args.mpc_dt, Q=Q, R=R, Q_f=Q_f,
                       v_max=args.v_max, w_max=args.w_max)

        # Output smoothing (EMA on commanded twist).
        self.smooth_alpha = float(args.smooth_alpha)
        self.u_filtered = np.zeros(6)

        # Auto-close on tag loss when we were already close (handles the
        # case where the gripper grabs the bottle and the tag rotates out
        # of view due to friction).
        self.lost_close_enabled = bool(args.lost_close_enabled)
        self.lost_close_t_mm = float(args.lost_close_t_mm)
        self.lost_close_r_deg = float(args.lost_close_r_deg)
        self.lost_seconds_to_trigger = float(args.lost_seconds_to_trigger)
        self.last_good_err_t_mm: float = float("inf")
        self.last_good_err_r_deg: float = float("inf")

        self.aligned = False
        self.miss_count = 0
        # Set this event to make main() break out of its spin loop and exit cleanly.
        self.shutdown_event = threading.Event()
        self.get_logger().info(
            f"goal: gripper -> tag xyz={self.p_goal.round(4).tolist()} "
            f"R[:row0]={self.R_goal[0].round(3).tolist()}")
        self.get_logger().info(
            f"publishing TwistStamped on {args.twist_topic} in frame "
            f"'{self.twist_frame}' @ {args.rate:.0f} Hz")

    def _publish_zero(self) -> None:
        if self.dry_run:
            return
        if self.backend_name == "xarm" and self.xarm is not None:
            self.xarm.stop()
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.twist_frame
        self.pub.publish(msg)

    def _publish_twist(self, u_gripper: np.ndarray) -> None:
        """Send commanded twist to whichever backend is configured.
        u_gripper is in gripper frame: [vx, vy, vz, wx, wy, wz]."""
        if self.dry_run:
            return
        if self.backend_name == "xarm" and self.xarm is not None:
            # xArm wants the velocity in BASE frame. Transform via tf.
            try:
                tf_bg = self.buf.lookup_transform(self.base_frame,
                                                  self.gripper_frame,
                                                  rclpy.time.Time(),
                                                  Duration(seconds=0.1))
            except TransformException as e:
                self.get_logger().warn(
                    f"No {self.base_frame}->{self.gripper_frame} for "
                    f"twist xform: {type(e).__name__}", throttle_duration_sec=2.0)
                return
            R_bg = quat_to_R([tf_bg.transform.rotation.x,
                              tf_bg.transform.rotation.y,
                              tf_bg.transform.rotation.z,
                              tf_bg.transform.rotation.w])
            v_base = R_bg @ u_gripper[:3]
            w_base = R_bg @ u_gripper[3:]
            u_base = np.concatenate([v_base, w_base])
            self.xarm.send_velocity_base(u_base, duration=0.2)
            return
        # Default: publish to MoveIt Servo.
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.twist_frame
        msg.twist.linear.x = float(u_gripper[0])
        msg.twist.linear.y = float(u_gripper[1])
        msg.twist.linear.z = float(u_gripper[2])
        msg.twist.angular.x = float(u_gripper[3])
        msg.twist.angular.y = float(u_gripper[4])
        msg.twist.angular.z = float(u_gripper[5])
        self.pub.publish(msg)

    def _maybe_close_gripper_and_exit(self) -> None:
        """Once aligned, fire the gripper close service exactly once,
        then wait settle_time seconds and shut down. Dry-run mode logs
        but never fires."""
        import time
        if self.dry_run:
            if not self.close_requested:
                self.get_logger().info(
                    f"[dry-run] would close gripper (TO {self.gripper_ionum} "
                    f"-> {self.gripper_value}) here.")
                self.close_requested = True
            return
        if not self.auto_close:
            if not self.close_requested:
                self.get_logger().info(">>> CLOSE NOW <<< (auto-close disabled — "
                                       "trigger the gripper manually)")
                self.close_requested = True
            return

        if not self.close_requested:
            if not self.gripper_cli.service_is_ready():
                self.get_logger().warn(
                    f"Gripper service {self.gripper_cli.srv_name!r} not ready yet — "
                    "waiting...", throttle_duration_sec=1.0)
                return
            req = SetDigitalIO.Request()
            req.ionum = self.gripper_ionum
            req.value = self.gripper_value
            self.get_logger().info(
                f"Calling {self.gripper_cli.srv_name} "
                f"(ionum={req.ionum}, value={req.value})")
            future = self.gripper_cli.call_async(req)
            future.add_done_callback(self._on_close_response)
            self.close_requested = True
            return

        if self.close_done_time is not None:
            elapsed = time.monotonic() - self.close_done_time
            if elapsed >= self.close_settle_time:
                self.get_logger().info(
                    f"Gripper closed and settled ({self.close_settle_time:.1f} s). "
                    "Shutting down.")
                self.timer.cancel()
                # Signal the main loop to exit. Don't call rclpy.shutdown()
                # from inside this callback — that path leaves spin() in a
                # state where destroy_node can hang, freezing the process.
                self.shutdown_event.set()

    def _on_close_response(self, future) -> None:
        import time
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Gripper close service raised: {e}")
            self.close_done_time = time.monotonic()
            return
        if res is None:
            self.get_logger().error("Gripper close service returned None.")
        elif int(res.ret) != 0:
            self.get_logger().warn(
                f"Gripper close returned ret={res.ret} message={res.message!r} — "
                "the driver may need to be in a specific mode/state.")
        else:
            self.get_logger().info(f"Gripper close ack: ret={res.ret} {res.message!r}")
        self.close_done_time = time.monotonic()


    def _tick(self) -> None:
        # Once we've fired the gripper close, the tag is rigid-bodied to
        # the arm — moving the arm doesn't reduce the tag-in-gripper error.
        # Stay in the shutdown sequence; never resume MPC.
        if self.close_requested:
            self._publish_zero()
            self._maybe_close_gripper_and_exit()
            return

        try:
            tf = self.buf.lookup_transform(self.gripper_frame, self.tag_frame,
                                           rclpy.time.Time(),
                                           Duration(seconds=0.2))
        except TransformException as e:
            # Covers Lookup / Extrapolation / Connectivity / InvalidArgument —
            # all mean the same thing for us: no fresh tag pose, halt motion.
            self.miss_count += 1
            lost_seconds = self.miss_count / self.args.rate

            # "Bottle is grabbed, tag rotated out of view" auto-close:
            # if we were close to the goal the moment the tag disappeared,
            # treat tag-loss as proof of contact and fire the close.
            if (self.lost_close_enabled
                    and lost_seconds >= self.lost_seconds_to_trigger
                    and self.last_good_err_t_mm < self.lost_close_t_mm
                    and self.last_good_err_r_deg < self.lost_close_r_deg
                    and not self.close_requested):
                self.get_logger().info("")
                self.get_logger().info("=" * 44)
                self.get_logger().info(
                    f"  TAG LOST while close (Δt={self.last_good_err_t_mm:.1f} mm, "
                    f"Δr={self.last_good_err_r_deg:.2f}°)")
                self.get_logger().info(
                    "  Treating as bottle-in-gripper → closing.")
                self.get_logger().info("=" * 44)
                self.aligned = True
                self._publish_zero()
                self._maybe_close_gripper_and_exit()
                return

            if self.miss_count > int(self.args.rate * 1.0):
                self.get_logger().warn(
                    f"Tag not seen for {lost_seconds:.1f} s "
                    f"({type(e).__name__}) — stopping motion.",
                    throttle_duration_sec=2.0)
                self._publish_zero()
            return
        self.miss_count = 0

        p = np.array([tf.transform.translation.x,
                      tf.transform.translation.y,
                      tf.transform.translation.z])
        R = quat_to_R([tf.transform.rotation.x, tf.transform.rotation.y,
                       tf.transform.rotation.z, tf.transform.rotation.w])

        dp = p - self.p_goal
        # Rotation error: R_err takes goal-orientation into current-orientation.
        # Driving R_err -> I drives R -> R_goal.
        dr = R_to_axis_angle(R @ self.R_goal.T)

        err_t = float(np.linalg.norm(dp))
        err_r = float(np.linalg.norm(dr))
        err_t_mm = err_t * 1000.0
        err_r_deg = np.degrees(err_r)
        # Cache the most recent good measurement for the tag-lost auto-close.
        self.last_good_err_t_mm = err_t_mm
        self.last_good_err_r_deg = err_r_deg

        if (err_t_mm < self.args.t_threshold_mm
                and err_r_deg < self.args.r_threshold_deg):
            if not self.aligned:
                self.get_logger().info("")
                self.get_logger().info("=" * 44)
                self.get_logger().info(
                    f"  ALIGNED  Δt={err_t_mm:.1f} mm   Δr={err_r_deg:.2f}°")
                self.get_logger().info("=" * 44)
            self.aligned = True
            self._publish_zero()
            self._maybe_close_gripper_and_exit()
            return

        if self.aligned:
            self.get_logger().info(
                f"drifted out of tolerance (Δt={err_t_mm:.1f} mm "
                f"Δr={err_r_deg:.2f}°) — resuming servoing")
        self.aligned = False

        # Build linearized interaction matrix at current p.
        I3 = np.eye(3)
        M = np.zeros((6, 6))
        M[:3, :3] = -I3
        M[:3, 3:] = skew(p)
        M[3:, 3:] = -I3

        # Soft brake near the goal: linearly scale velocity caps with the
        # current error. Full caps when |Δt| ≥ brake_t and |Δr| ≥ brake_r,
        # but at zero error the caps shrink to v_max * floor_frac (no
        # absolute zero so the controller can still creep into the threshold).
        scale_t = min(1.0, err_t / self.args.brake_t_m)
        scale_r = min(1.0, err_r / np.radians(self.args.brake_r_deg))
        scale = max(self.args.brake_floor, max(scale_t, scale_r))
        self.mpc.set_runtime_caps(self.args.v_max * scale,
                                  self.args.w_max * scale)

        e0 = np.concatenate([dp, dr])
        u_raw = self.mpc.solve(e0, M)

        # Safety: refuse to publish any NaN/Inf — that's how we got an arm
        # in a NaN joint state earlier. Zero-twist on bad data.
        if not np.all(np.isfinite(u_raw)):
            self.get_logger().error(
                f"MPC produced non-finite twist {u_raw.tolist()} — publishing zero",
                throttle_duration_sec=1.0)
            self._publish_zero()
            return

        # Exponential moving average on the commanded twist.
        # smooth_alpha = 1 → no filtering (raw MPC output)
        # smooth_alpha = 0 → frozen (no response). Default 0.3 = mostly smooth.
        self.u_filtered = (self.smooth_alpha * u_raw
                           + (1.0 - self.smooth_alpha) * self.u_filtered)
        u = self.u_filtered

        self._publish_twist(u)

        self.get_logger().info(
            f"Δt={err_t_mm:6.1f} mm  Δr={err_r_deg:5.2f}°   "
            f"v=[{u[0]:+.3f},{u[1]:+.3f},{u[2]:+.3f}]  "
            f"ω=[{u[3]:+.3f},{u[4]:+.3f},{u[5]:+.3f}]",
            throttle_duration_sec=0.4)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--goal", type=Path, default=Path("goal_pose.yaml"),
                   help="Path to the YAML produced by record_goal_pose.py.")
    p.add_argument("--rate", type=float, default=15.0,
                   help="Control rate (Hz). Default 15 — matches the ZED tag "
                        "publish rate so every tick has a fresh detection.")
    p.add_argument("--twist-topic", default="/servo_node/delta_twist_cmds",
                   help="Topic MoveIt Servo subscribes to.")
    p.add_argument("--twist-frame", default=None,
                   help="frame_id stamped on the TwistStamped (defaults to "
                        "the gripper frame from the goal yaml).")
    p.add_argument("--backend", choices=("servo", "xarm"), default="servo",
                   help="Output backend. 'servo' (default) publishes TwistStamped "
                        "to MoveIt Servo. 'xarm' calls /xarm/vc_set_cartesian_velocity "
                        "directly — usually smoother for streaming velocities, "
                        "but skips Servo's IK/limit/singularity safety. The MPC "
                        "must run before any subsequent MoveIt motion since "
                        "mode 5 needs to be reset to mode 0 — that's handled "
                        "on exit.")
    p.add_argument("--base-frame", default="link_base",
                   help="Robot base frame (only used by --backend xarm to "
                        "transform the MPC's gripper-frame twist into base "
                        "frame for the xArm service).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and log commanded velocities but do NOT publish — "
                        "the arm will not move. Use this for the first run to "
                        "validate MPC outputs are sensible before wiring up Servo.")

    g_mpc = p.add_argument_group("mpc")
    g_mpc.add_argument("--horizon", type=int, default=8,
                       help="MPC horizon length (default 8).")
    g_mpc.add_argument("--mpc-dt", type=float, default=0.05,
                       help="MPC integration step (s). Default 0.05.")
    g_mpc.add_argument("--q-pos", type=float, default=400.0,
                       help="State cost weight on translation error.")
    g_mpc.add_argument("--q-rot", type=float, default=40.0,
                       help="State cost weight on rotation error.")
    g_mpc.add_argument("--r-vel", type=float, default=1.0,
                       help="Control cost weight on linear velocity.")
    g_mpc.add_argument("--r-omega", type=float, default=1.0,
                       help="Control cost weight on angular velocity.")
    g_mpc.add_argument("--terminal-weight", type=float, default=10.0,
                       help="Multiplier on Q for the terminal step.")

    g_lim = p.add_argument_group("limits")
    g_lim.add_argument("--v-max", type=float, default=0.05,
                       help="Per-axis linear velocity cap (m/s). Start small!")
    g_lim.add_argument("--w-max", type=float, default=0.5,
                       help="Per-axis angular velocity cap (rad/s).")
    g_lim.add_argument("--brake-t-m", type=float, default=0.08,
                       help="Translation error (m) below which v_max scales "
                            "linearly toward --brake-floor of its value. "
                            "Default 0.08 m — controller slows down inside 8 cm.")
    g_lim.add_argument("--brake-r-deg", type=float, default=20.0,
                       help="Rotation error (deg) below which w_max scales "
                            "linearly toward --brake-floor of its value. "
                            "Default 20°.")
    g_lim.add_argument("--brake-floor", type=float, default=0.15,
                       help="Minimum fraction of v_max/w_max at zero error "
                            "(default 0.15 = 15%% of cap). Keeps the "
                            "controller from stalling out short of the goal.")

    g_done = p.add_argument_group("done conditions")
    g_done.add_argument("--t-threshold-mm", type=float, default=15.0,
                       help="Translation error to declare 'aligned' (default 15 mm). "
                            "Larger = closes the gripper earlier; useful when the "
                            "fingers are already around the object and you don't need "
                            "perfect tag alignment.")
    g_done.add_argument("--r-threshold-deg", type=float, default=30.0,
                       help="Rotation error to declare 'aligned' (default 30°). "
                            "Loose because the bottle just has to be between the "
                            "fingers — tag orientation is irrelevant for the close.")

    g_smooth = p.add_argument_group("smoothing")
    g_smooth.add_argument("--smooth-alpha", type=float, default=0.2,
                          help="EMA factor on commanded twist (default 0.2). "
                               "1.0 = no filtering, 0.0 = frozen. Lower = smoother "
                               "motion but more lag tracking the goal.")

    g_lost = p.add_argument_group("auto-close on tag loss")
    g_lost.add_argument("--lost-close-enabled", dest="lost_close_enabled",
                        action="store_true", default=True,
                        help="If the tag is lost while the gripper was close to the "
                             "goal, close the gripper anyway. Handles the case where "
                             "the gripper grabs the bottle and friction rotates the "
                             "tag out of view (default on).")
    g_lost.add_argument("--no-lost-close", dest="lost_close_enabled",
                        action="store_false",
                        help="Disable tag-lost-close behavior.")
    g_lost.add_argument("--lost-close-t-mm", type=float, default=30.0,
                        help="Last-known translation error (mm) must be under this "
                             "for tag-lost-close to fire. Default 30 mm.")
    g_lost.add_argument("--lost-close-r-deg", type=float, default=20.0,
                        help="Last-known rotation error (deg) must be under this. "
                             "Default 20°.")
    g_lost.add_argument("--lost-seconds-to-trigger", type=float, default=0.7,
                        help="Tag must be lost for at least this many seconds "
                             "before tag-lost-close fires. Default 0.7 s.")

    g_close = p.add_argument_group("auto-close")
    g_close.add_argument("--auto-close", dest="auto_close", action="store_true",
                         default=True,
                         help="When aligned, automatically toggle the gripper output "
                              "and shut down (default: on). Use --no-auto-close to disable.")
    g_close.add_argument("--no-auto-close", dest="auto_close", action="store_false",
                         help="Disable auto-close — controller just prints CLOSE NOW.")
    g_close.add_argument("--gripper-service", default="/xarm/set_tgpio_digital",
                         help="xarm_msgs/SetDigitalIO service to fire when aligned.")
    g_close.add_argument("--gripper-ionum", type=int, default=1,
                         help="Tool digital output to toggle (TO1 closes your gripper).")
    g_close.add_argument("--gripper-value", type=int, default=1,
                         help="Output value to set (1 = High, 0 = Low).")
    g_close.add_argument("--close-settle-time", type=float, default=1.5,
                         help="Seconds to wait after firing the close before shutdown.")

    args = p.parse_args()
    goal = yaml.safe_load(args.goal.open())

    rclpy.init()
    node = MpcVSController(goal, args)
    # If using xarm backend, enable Cartesian velocity mode before the
    # control loop starts. Must be restored to position mode on exit so
    # MoveIt can drive the arm afterwards.
    if node.backend_name == "xarm" and node.xarm is not None:
        if not node.xarm.enable():
            print("Failed to enable XArm Cartesian velocity mode — aborting.",
                  file=sys.stderr)
            rclpy.try_shutdown()
            return
    try:
        while rclpy.ok() and not node.shutdown_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        try:
            node._publish_zero()
        except Exception:
            pass
    finally:
        try:
            node._publish_zero()
        except Exception:
            pass
        # Always restore xArm to position mode so MoveIt is usable afterwards.
        if node.backend_name == "xarm" and node.xarm is not None:
            try:
                node.xarm.disable()
            except Exception as e:
                print(f"xArm.disable() failed: {e}", file=sys.stderr)
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
