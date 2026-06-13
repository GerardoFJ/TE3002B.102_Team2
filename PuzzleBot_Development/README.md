# PuzzleBot — Autonomous Line-Following Navigation

ROS 2 (Humble) workspace for the **PuzzleBot** autonomous navigation challenge
(Reto TE3002B, Team 2). The robot follows a black line on a printed "mini-city"
track while a YOLOv8 network reads traffic signs and lights, and an odometry-based
state machine resolves intersections (turn left/right, go straight, stop).

Runs on an **NVIDIA Jetson Nano** on board; the embedded motor control runs on a
micro-ROS microcontroller. All percepcion and decision logic is ROS 2 nodes.

---

## How it works (data flow)

```
                 ┌───────────────────────┐
   CSI camera ──▶│ puzzlebot_ros          │  /camera/image_rect(/compressed)
   1280x720@30   │  (camera_rectify)      │────────────┬───────────────┐
                 └───────────────────────┘             │               │
                                                        ▼               ▼
                                       ┌────────────────────┐  ┌────────────────────┐
                                       │ yolo_detector      │  │ follow_single_line │
                                       │  YOLOv8, 9 classes │  │  line follower +   │
                                       │ /yolo/detections ──┼─▶│  intersection FSM  │
                                       └────────────────────┘  └─────────┬──────────┘
                                       ┌────────────────────┐            │ /cmd_vel
                                       │ traffic_light      │            ▼
                                       │  HSV light gate    │  ┌────────────────────┐
                                       │ /traffic_light/go ─┼─▶│ cmd_vel_to_wheels  │
                                       └────────────────────┘  │ inv. kinematics +  │
                                                     /odom  ◀───┤ wheel odometry     │
                                                                └─────────┬──────────┘
                                                       /VelocitySet L,R   │  ▲ /VelocityEnc L,R
                                                                          ▼  │
                                                                ┌────────────────────┐
                                                                │ MCU (micro-ROS)    │
                                                                │ per-wheel PID      │
                                                                └────────────────────┘
```

When the line disappears at a junction, `follow_single_line` waits ~1.5 s, reads
the freshest YOLO detection, and executes an **odometry maneuver** (drive forward,
turn 90° toward the arrow, exit straight) or **holds** for a red/yellow light.

---

## Main pipeline packages

| Package | Lang | Role |
|---|---|---|
| **`puzzlebot_ros`** | Python | CSI camera capture + rectify + JPEG (`camera_rectify`); bringup & micro-ROS agent launch files. |
| **`cmd_vel_to_wheels`** | C++ | `/cmd_vel` → wheel setpoints (inverse kinematics) and encoder velocities → `/odom` (wheel odometry / pose). |
| **`line_follower`** | C++ | `follow_single_line`: threshold-on-black line follower, P/PD control, intersection state machine, YOLO-guided decisions. **Core node.** Includes `config/` params and `tools/` live tuners. |
| **`yolo_detector`** | Python | YOLOv8 inference; publishes `/yolo/detections` (`name:conf:area:cx`) and an annotated image. |
| **`traffic_light`** | C++ | HSV color traffic-light detector; publishes the `/traffic_light/go` gate (backup to the YOLO light classes). |
| **`micro_ros_setup`**, **`uros`** | — | micro-ROS tooling for the embedded motor-control firmware/agent. |

---

## Build

From the colcon workspace root (one level up):

```bash
colcon build --symlink-install
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

> The `archive/` folder carries a `COLCON_IGNORE` file, so its packages are
> skipped by the build.

## Run the demo (on the Jetson)

```bash
# 1. micro-ROS agent  (VelocitySet/Enc L,R bridge to the MCU)
ros2 launch puzzlebot_ros micro_ros_agent.launch.py

# 2. odometry + inverse kinematics  (/cmd_vel, /odom)
ros2 run cmd_vel_to_wheels cmd_vel_to_wheels --ros-args -p publish_tf:=true

# 3. camera  (/camera/image_rect, /camera/image_rect/compressed)
ros2 launch puzzlebot_ros camera_jetson.launch.py

# 4. YOLO sign/light detection  (/yolo/detections)
ros2 run yolo_detector yolo_node --ros-args -p model_path:=<path-to>/best.pt

# 5. line follower + intersection FSM  (the brain)
ros2 run line_follower follow_single_line --ros-args \
  --params-file line_follower/config/follow_single_line.yaml
```

Optional: `ros2 run traffic_light detect` for the HSV light gate.

All control/decision parameters can be tuned live, e.g.
`ros2 param set /follow_single_line kp 0.8`.

### Offline ROI / threshold tuning (from a laptop)

`line_follower/tools/live_single_line_tuner.py` subscribes to the compressed
camera topic and mirrors the C++ detector with on-image sliders — drag until the
line is detected cleanly, press `s` to save the params.

---

## `archive/`

Experimental, alternative, and superseded work, **not part of the deployed
pipeline** (kept for reference, ignored by colcon):

| Folder | What it is |
|---|---|
| `lane_pilot` | Alternative metric lane follower (IPM projection + pure pursuit). |
| `hough_features` | Hough-segment feature extractor (Phase 1 of a learned-steering idea). |
| `ml_steering` | XGBoost steering predictor (experimental). |
| `steering_safety` | Smoothing / clamping / light-gating for `ml_steering`. |
| `yolo_trt` | TensorRT (`.engine`) YOLO node — in progress, not used in the run. |
| `square_controller` | Early waypoint/square path controller (superseded by the FSM). |
| `puzzlebot_camera`, `camera_info_publisher` | Older camera nodes (superseded by `puzzlebot_ros/camera_rectify`; only the `aruco_jetson` demo still references them). |
| `ros_deep_learning` | Third-party jetson-inference nodes. |
