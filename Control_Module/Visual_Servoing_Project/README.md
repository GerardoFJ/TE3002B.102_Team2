# Visual Servoing Project

Visual servoing on an xArm6 with a ZED2 mounted on a custom gripper.
The robot runs on a Jetson Orin AGX (`orin@192.168.31.10`) using the
RoBorregos `home2` manipulation stack (ROS 2 Humble, MoveIt, Cyclone DDS).

## Layout

```
tools/
  calibration/       # eye-in-hand calibration: gripper -> ZED
                     #   collect_samples.py, solve_handeye.py,
                     #   apply_to_urdf.py, print_tag.py
  perception/        # AprilTag detection / debug helpers
                     #   tag_detector.py (compressed, pose-stable),
                     #   list_tag_ids.py
  pick/              # MPC visual-servo grasp (eye-in-hand)
                     #   record_goal_pose.py, mpc_vs_pick.py,
                     #   approach_pregrasp.py
  align/             # 2D / 1D pixel-space alignments (second ZED)
                     #   align_top_tag.py (centroid → saved target),
                     #   align_drop_target.py (centroid → other tag),
                     #   align_approach_tag.py (FAR→NEAR pixel-size)
  launch/            # ROS 2 launch files
                     #   apriltag_zed.launch.py, servo_pick.launch.py,
                     #   bringup.launch.py
  pick_pipeline.py   # 7-stage orchestrator: PREGRASP → MPC → LIFT_UP
                     # → DROP → ALIGN_TOP → ALIGN_APPROACH → OPEN_TWIST
```

On the orin everything is deployed flat at
`/home/orin/dev/nav/visual_servoing/` (mounted into the container as
`/workspace/src/visual_servoing/`); scripts reference each other by that
container path. The subfolders here are only for source organization.

## Status

| Stage | State |
|---|---|
| Manipulation stack launches (MoveIt + xArm) | ✅ done (Cyclone DDS override fix) |
| ZED publishing camera streams + tf | ✅ done (`/zed/zed_node/...`) |
| AprilTag detector (apriltag_ros 3.3.0) | ✅ installed in container, launch ready |
| Gripper → ZED hand-eye calibration | ✅ done — see `tools/calibration/README.md` |
| MPC visual-servo grasp on a bottle | ✅ done — see `tools/pick/README.md` |
| Drop + top-of-bottle alignment (second ZED) | ✅ done — `tools/align/align_top_tag.py` |
| AprilTag-pixel-size approach to opener | ✅ done — `tools/align/align_approach_tag.py` |
| Full pipeline (7 stages, end-to-end) | ✅ done — `tools/pick_pipeline.py` |

## Quick start (after calibration)

On the orin:
```bash
ssh orin@192.168.31.10
cd /home/orin/dev/nav
./run.sh manipulation
# inside the container:
ros2 launch arm_pkg frida_moveit_config.launch.py
```

For the hand-eye calibration procedure see `tools/calibration/README.md`.
For the visual-servoing grasp pipeline see `tools/pick/README.md`.
To run the full 7-stage pipeline, see the docstring of
`tools/pick_pipeline.py`:

```bash
python3 /workspace/src/visual_servoing/pick_pipeline.py --goal goal_pose.yaml
```

Prerequisites: `bringup.launch.py` running and the three saved YAMLs in
the cwd (`goal_pose.yaml`, `align_target.yaml`, `approach_target.yaml`).

## Useful topics

| Topic | Purpose |
|---|---|
| `/zed/zed_node/left/image_rect_color` + `/zed/zed_node/left/camera_info` | Tag/object detection inputs |
| `/zed/zed_node/depth/depth_registered` | Per-pixel depth aligned to the left camera |
| `/zed/zed_node/point_cloud/cloud_registered` | Filtered point cloud |
| `/apriltag/detections` (when detector running) | Tag id list + corners |
| `tf: zed_left_camera_optical_frame -> tag36h11_0` | Tag pose for grasp planning |
| `/move_group` (action) | MoveIt motion planning + execution |
| `/xarm6_traj_controller/joint_trajectory` | Direct trajectory commands |
| `/xarm/joint_states`, `/joint_states` | Arm feedback |
