# ZED hand-eye calibration (eye-in-hand)

Solves for the **gripper → ZED** transform so the URDF reflects how the
ZED is actually bolted on. Output goes into
`robot_description/frida_description/urdf/Gripper/Custom/gripper.xacro`,
specifically the `<joint name="ZED">` origin (parent=`gripper`, child=`zed`).

The pipeline is three scripts run on the orin inside the manipulation
container:

```
collect_samples.py  →  handeye_samples.json
solve_handeye.py    →  handeye_samples_result.json   (cv2.calibrateHandEye)
apply_to_urdf.py    →  edits gripper.xacro + backup
```

All scripts live on the orin at `/home/orin/dev/nav/visual_servoing/`,
which appears in the manipulation container at `/workspace/src/visual_servoing/`.

## Prerequisites

1. **apriltag_ros** installed in the container (one-off, already done):
   ```bash
   sudo apt-get install -y ros-humble-apriltag-ros
   ```
   Note: This is installed live in the running container. To persist
   across rebuilds, add the same line to
   `/home/orin/dev/nav/docker/manipulation/Dockerfile.l4t`.

2. **A printed AprilTag**, family `tag36h11`, id `0`, square side length
   **133 mm** (matches the `TAG_SIZE_M` in `apriltag_zed.launch.py`
   and the default `--tag-size` in `collect_samples.py`).

3. ZED + `frida_moveit_config` already running.

## Procedure

All commands run on `orin@192.168.31.10`. Open four terminals (or tmux
panes) inside the manipulation container — every command in this README
assumes the container's ROS env is sourced.

### 1. Start MoveIt + xArm

Terminal A:
```bash
ros2 launch arm_pkg frida_moveit_config.launch.py
```

### 2. Start the ZED node

Terminal B — whichever ZED launch you normally use, so
`/zed/zed_node/left/image_rect_color` and `camera_info` are publishing.

### 3. Start the AprilTag detector

Terminal C:
```bash
ros2 launch /workspace/src/visual_servoing/apriltag_zed.launch.py
```

Quick smoke test:
```bash
ros2 topic echo /apriltag/detections --once
```
You should see `id: 0` when the camera is pointed at the tag. The tag's
frame `tag36h11_0` will also appear under `zed_left_camera_optical_frame`
in `ros2 run tf2_tools view_frames`.

### 4. Put the arm in freedrive

```bash
ros2 service call /xarm/set_mode  xarm_msgs/srv/SetInt "{data: 2}"
ros2 service call /xarm/set_state xarm_msgs/srv/SetInt "{data: 0}"
```
Mode 2 = manual / freedrive. If service names differ on your driver,
pose via RViz interactive markers instead — the collector doesn't care
how the arm moves.

### 5. Collect samples

Terminal D:
```bash
mkdir -p ~/handeye_workdir && cd ~/handeye_workdir
python3 /workspace/src/visual_servoing/collect_samples.py
```

For each capture:
1. Pose the arm so the tag is fully visible and reasonably close
   (0.3 – 0.8 m typical for ZED2).
2. Press **Enter**.
3. Move to a *different* orientation — at least 15–30° away — and capture
   again. Pure translations don't constrain the rotation part.
4. Press **q + Enter** when you have 12–20 samples. Vary roll, pitch,
   yaw, and translation across the workspace.

Output: `handeye_samples.json` in the current directory. Contains the
samples plus a one-time snapshot of `zed → zed_left_camera_optical_frame`
(used by the solver to back out the joint origin).

### 6. Solve

```bash
python3 /workspace/src/visual_servoing/solve_handeye.py handeye_samples.json
```

You'll see a table with five hand-eye methods (TSAI, PARK, HORAUD,
ANDREFF, DANIILIDIS). If they agree within a few mm / few degrees,
trust the result. If they disagree wildly, your samples are too uniform
in orientation — go back to step 5 and add more rotational variety.

The script prints the proposed `<origin xyz="…" rpy="…"/>` line and
writes `handeye_samples_result.json`.

**Residual RMS** is also printed: how tightly the tag's position in the
base frame agrees across all samples after applying the solution. Under
10 mm is solid; over 30 mm means recollect.

### 7. Apply to URDF

From the orin host (NOT inside the container, since this script edits
the bind-mounted source file):
```bash
exit       # leave the container if you're in one
cd /home/orin/dev/nav/visual_servoing
python3 apply_to_urdf.py ~/handeye_workdir/handeye_samples_result.json --dry-run
# inspect the before/after diff. If it looks right:
python3 apply_to_urdf.py ~/handeye_workdir/handeye_samples_result.json
```
This edits `gripper.xacro` at
`/home/orin/dev/nav/robot_description/frida_description/urdf/Gripper/Custom/gripper.xacro`
and writes a timestamped backup next to it.

### 8. Rebuild frida_description and relaunch

Back inside the container:
```bash
cd /workspace
colcon build --packages-select frida_description --symlink-install
source install/setup.bash
# stop the running moveit, then:
ros2 launch arm_pkg frida_moveit_config.launch.py
```

Verify:
```bash
ros2 run tf2_ros tf2_echo gripper zed_left_camera_optical_frame
```
The translation should match the rigid offset you measured by hand, and
in RViz the ZED point cloud should overlay the real-world geometry.

## Notes & gotchas

- **TF lookup uses `link_base` as the base frame** (xArm convention). If
  your URDF's root link is named differently, pass `--base-frame` to
  `collect_samples.py`.
- The `gripper_frame` is the *link we want to express the camera offset
  relative to*. `gripper.xacro` defines that link literally as `gripper`,
  so the default is correct unless you renamed it.
- `apriltag_ros` v3.x publishes detections both in
  `apriltag_msgs/AprilTagDetectionArray` and as `tf` transforms. The
  collector uses tf2 — it's the canonical pose representation and
  matches what hand-eye math expects.
- If the detector sees multiple tags or you change tag id, update
  `TAG_ID`, `TAG_SIZE_M`, and `TAG_FRAME` in `apriltag_zed.launch.py`
  and pass `--tag-frame` to the collector.
- The solver assumes a **fixed** tag (the tag does not move during
  collection). If you re-tape the tag between captures, throw out the
  samples and restart.
