"""
One-shot launch for the visual servoing infrastructure. Brings up:
  1. frida_moveit_config  (MoveIt + xArm driver, limited:=false)
  2. tag_detector.py      (our robust AprilTag pose publisher, compressed)
  3. servo_pick.launch.py (MoveIt Servo, limited:=false), unpaused
After everything is up, calls /servo_node/start_servo so Servo accepts twist
commands immediately.

NOT included on purpose:
  - The ZED launch (start that separately the way you already do).
  - The MPC controller (you start mpc_vs_pick.py or pick_pipeline.py
    by hand whenever you want a pick to happen).

Timings: moveit_config takes ~6 s to come up; servo_node ~2 s on top of
that; start_servo is then called after a safety pad. Defaults work on
the Orin.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


# Inside the container these resolve to /workspace/src/visual_servoing/...
TOOLS_DIR = "/workspace/src/visual_servoing"


def generate_launch_description():
    tag_size = LaunchConfiguration("tag_size")
    side = LaunchConfiguration("side")
    # Both delays are absolute (measured from launch start).
    moveit_delay = LaunchConfiguration("moveit_settle")
    start_servo_delay = LaunchConfiguration("start_servo_delay")

    return LaunchDescription([
        DeclareLaunchArgument("tag_size", default_value="0.024",
                              description="Physical AprilTag side length in meters."),
        DeclareLaunchArgument("side", default_value="right",
                              description="ZED camera to use: 'left' or 'right'."),
        DeclareLaunchArgument("moveit_settle", default_value="8.0",
                              description="Seconds (from launch start) before "
                                          "Servo is launched."),
        DeclareLaunchArgument("start_servo_delay", default_value="12.0",
                              description="Seconds (from launch start) before "
                                          "/servo_node/start_servo is called."),

        # 1. MoveIt + xArm driver with the full ±2π joint range.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare("arm_pkg"),
                                      "launch", "frida_moveit_config.launch.py"])
            ),
            launch_arguments={"limited": "false"}.items(),
        ),

        # 2. AprilTag detector (immediate — it just waits for camera_info anyway).
        ExecuteProcess(
            cmd=[
                "python3", f"{TOOLS_DIR}/tag_detector.py",
                "--tag-size", tag_size,
                "--image-topic", ["/zed/zed_node/", side, "/image_rect_color"],
                "--camera-info-topic", ["/zed/zed_node/", side, "/camera_info"],
            ],
            output="screen",
            name="tag_detector",
        ),

        # 3. MoveIt Servo, delayed so move_group is ready first.
        TimerAction(
            period=moveit_delay,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        f"{TOOLS_DIR}/servo_pick.launch.py"
                    ),
                    launch_arguments={"limited": "false"}.items(),
                ),
            ],
        ),

        # 4. Unpause Servo once the service exists.
        TimerAction(
            period=start_servo_delay,
            actions=[
                ExecuteProcess(
                    cmd=["ros2", "service", "call",
                         "/servo_node/start_servo",
                         "std_srvs/srv/Trigger", "{}"],
                    output="screen",
                    name="servo_start_call",
                ),
            ],
        ),
    ])
