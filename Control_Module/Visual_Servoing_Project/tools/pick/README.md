# Visual servoing pick with MPC

Drives the gripper to a demonstrated goal pose relative to the AprilTag
on a bottle, then asks you to close the gripper.

```
record_goal_pose.py   →  goal_pose.yaml          (one-shot, with arm at demo pose)
servo_pick.launch.py  →  brings up servo_node    (listens on /servo_node/delta_twist_cmds)
mpc_vs_pick.py        →  publishes TwistStamped to /servo_node/delta_twist_cmds at ~20 Hz
                          (computed by a 6-DOF linear MPC)
```

## Terminals overview

Five terminals total inside the manipulation container. A–C are the
"infrastructure" (already running for calibration); D–E are new.

| # | What runs | When to start |
|---|---|---|
| A | `ros2 launch arm_pkg frida_moveit_config.launch.py`  *(MoveIt + xArm driver)* | already running |
| B | your ZED launch | already running |
| C | `ros2 launch /workspace/src/visual_servoing/apriltag_zed.launch.py` | already running |
| D | `ros2 launch /workspace/src/visual_servoing/servo_pick.launch.py` | NEW — section 3 |
| E | `python3 /workspace/src/visual_servoing/mpc_vs_pick.py …` | NEW — section 4 |

## 1. Capture the goal pose

Pose the arm physically at the position where you want it to be **at the
moment of grasp closure**, with the tag visible. Then, from inside the
container:

```bash
mkdir -p ~/handeye_workdir && cd ~/handeye_workdir
python3 /workspace/src/visual_servoing/record_goal_pose.py
```

This snapshots `T_gripper → tag36h11_0` and writes `goal_pose.yaml`.
Check the distance it prints — it should match roughly the gripper-to-bottle
distance you measured by eye.

## 2. Dry-run the MPC (no motion)

The MPC controller can be run **without publishing twists** so you can
sanity-check the commanded velocities before any motion happens:

```bash
cd ~/handeye_workdir
python3 /workspace/src/visual_servoing/mpc_vs_pick.py --goal goal_pose.yaml --dry-run
```

You'll see lines every ~0.4 s like:
```
Δt= 134.5 mm  Δr= 12.43°   v=[+0.025,-0.018,+0.041]  ω=[+0.083,-0.054,+0.012]
```

**Manually move the arm in freedrive while it's running** and watch:
- Moving the gripper *toward* the goal pose → `Δt` should shrink.
- Moving it *away* → `Δt` should grow.
- The commanded `v`/`ω` should point in a direction that would reduce error.

If signs are obviously wrong, paste me one or two log lines + description of
how you moved the arm and I'll fix the frame conventions.

## 3. Start MoveIt Servo

Terminal D:
```bash
ros2 launch /workspace/src/visual_servoing/servo_pick.launch.py
```

You should see:
```
[servo_node_main-1] Loading robot model 'FRIDA'...
[servo_node_main-1] Listening to joint states on topic '/joint_states'
[servo_node_main-1] Publishing maintained planning scene on '/servo_node/publish_planning_scene'
```

Warnings about `zed_camera_center is not known to URDF` are cosmetic — the
SRDF mentions a link that the current URDF doesn't expose. Ignore.

**Servo starts paused.** Enable it (one-time, until next stop):
```bash
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger {}
```

Verify it's accepting commands:
```bash
ros2 topic info /servo_node/delta_twist_cmds
# expect: Subscription count: 1
```

## 4. Run the MPC controller (with motion)

Terminal E:
```bash
cd ~/handeye_workdir
python3 /workspace/src/visual_servoing/mpc_vs_pick.py --goal goal_pose.yaml \
    --v-max 0.03 --w-max 0.3        # START SMALL: 3 cm/s, 17°/s
```

Watch the error shrink. The arm should move slowly toward the goal pose.

When you see:
```
============================================
  ALIGNED  Δt=4.2 mm   Δr=1.87°
  >>> CLOSE NOW <<<
============================================
```

…close the gripper from yet another terminal (or just by hand if you're
debugging alignment in isolation, since you picked the "manual close"
option):
```bash
# Example — adjust to your gripper service interface:
ros2 service call /xarm/set_gripper_position xarm_msgs/srv/GripperMove \
    "{pos: 0.0}"
```

Then Ctrl+C the MPC (it sends a final zero twist on exit).

## Stopping safely

Anything to stop motion right now:
- Ctrl+C the MPC controller → publishes a zero twist.
- `ros2 service call /servo_node/stop_servo std_srvs/srv/Trigger {}` →
  Servo halts and ignores all future twist commands until `start_servo`
  is called again.

## Tuning knobs

| Flag | What it does | When to change |
|---|---|---|
| `--v-max` | Max linear velocity per axis (m/s) | Raise once you trust the controller; 0.05–0.10 m/s is typical |
| `--w-max` | Max angular velocity per axis (rad/s) | Raise after `--v-max` is comfortable |
| `--q-pos` / `--q-rot` | MPC state cost (translation / rotation error weight) | Raise to track tighter; lower to be lazier |
| `--r-vel` / `--r-omega` | MPC control cost (smoothness) | Raise to get smoother (slower) motion |
| `--horizon` | MPC look-ahead steps | More = smarter (but slower per tick) |
| `--mpc-dt` | MPC integration step (s) | Match to ~1/`rate` for predictive accuracy |
| `--t-threshold-mm` / `--r-threshold-deg` | "Aligned" tolerance | Tighter = more precision required before CLOSE NOW |

## Safety notes

- **Always run with `--dry-run` first** after any change to the goal,
  calibration, or controller tuning.
- **Start with low velocity caps** (`--v-max 0.03`). The arm can overshoot
  if the MPC is mis-tuned or if visual jitter is high.
- The controller **halts (zero twist)** if the tag is lost for more than
  1 second.
- Ctrl+C always publishes a zero twist on exit.
- `/servo_node/stop_servo` is your kill switch — keep that command pasted
  in a terminal ready to hit Enter.

## What this MPC actually solves

For curiosity / report-writing:

- **State** `e ∈ ℝ⁶` = (Δp, Δr): translation error + axis-angle rotation
  error of the tag-in-gripper pose relative to the demonstrated goal,
  both expressed in the gripper frame.
- **Control** `u ∈ ℝ⁶` = (v, ω): commanded gripper twist in the gripper
  frame.
- **Linearized dynamics** (one step):
  ```
  e_{k+1} = e_k + dt · M · u_k         with  M = [[ -I₃ ,  [p]ₓ ],
                                                 [  0  ,  -I₃ ]]
  ```
  where `[p]ₓ` is the skew of the current tag-in-gripper translation
  (re-linearized each control cycle, so the "linear" model still tracks
  large rotations correctly tick-by-tick).
- **Cost**:
  ```
  Σ_{k=1..N}  e_k' Q e_k  +  Σ_{k=0..N-1}  u_k' R u_k  +  e_N' Q_f e_N
  ```
  with `Q_f = terminal_weight · Q` (default ×10) so the controller weighs
  ending up at the goal more than transient error.
- **Constraints**: box bounds on each component of `u_k` (`v_max`, `ω_max`).
- **Solver**: L-BFGS-B over the stacked decision variable
  `z = [u_0; u_1; …; u_{N-1}] ∈ ℝ^{6N}`. Warm-started from the previous
  cycle's solution shifted by one step (standard MPC trick).
- **Receding horizon**: apply only `u_0` per cycle, re-observe, re-solve.
