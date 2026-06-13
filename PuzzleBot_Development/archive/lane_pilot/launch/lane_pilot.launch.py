"""Launch the metric lane follower: perception (IPM) + control (pure pursuit).

The camera node and cmd_vel_to_wheels (with publish_tf optional — this stack
uses /odom directly, so TF is NOT required) are expected to be running already.

Pass the calibrated homography produced by calibrate_ground_homography.py:
  ros2 launch lane_pilot lane_pilot.launch.py \
      homography_file:=/home/puzzlebot/ground_homography.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("lane_pilot")
    params = os.path.join(share, "config", "lane_pilot.yaml")

    homography_arg = DeclareLaunchArgument(
        "homography_file",
        default_value=os.path.expanduser("~/ground_homography.yaml"),
        description="Path to the ground homography YAML from "
                    "calibrate_ground_homography.py",
    )
    debug_arg = DeclareLaunchArgument(
        "debug", default_value="true",
        description="Publish the bird's-eye debug overlay image",
    )
    homography_file = LaunchConfiguration("homography_file")
    debug = LaunchConfiguration("debug")

    ipm_node = Node(
        package="lane_pilot",
        executable="lane_ipm_node",
        name="lane_ipm_node",
        output="screen",
        parameters=[
            params,
            {"homography_file": homography_file,
             "publish_debug_image": debug},
        ],
    )

    pilot_node = Node(
        package="lane_pilot",
        executable="lane_pilot_node",
        name="lane_pilot_node",
        output="screen",
        parameters=[params],
    )

    return LaunchDescription([homography_arg, debug_arg, ipm_node, pilot_node])
