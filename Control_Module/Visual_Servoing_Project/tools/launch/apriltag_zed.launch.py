"""
Start apriltag_ros's apriltag_node against one of the ZED camera streams.

Uses **compressed** image_transport because the raw 1280x720x3 stream at
15 Hz overwhelms Cyclone DDS on the Orin and silently drops messages
(QoS RELIABLE publisher + bursty 40 MB/s = subscriber starvation).

Defaults: ZED right camera (the side currently looking at the bottle),
tag36h11 id 0, side length 0.133 m. To switch sides:

    ros2 launch apriltag_zed.launch.py side:=left
    ros2 launch apriltag_zed.launch.py side:=right
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


TAG_ID = 0
TAG_SIZE_M = 0.133
TAG_FRAME = "tag36h11_0"


def launch_setup(context, *args, **kwargs):
    side = LaunchConfiguration("side").perform(context)
    if side not in ("left", "right"):
        raise RuntimeError(f"side must be 'left' or 'right', got {side!r}")

    return [
        Node(
            package="apriltag_ros",
            executable="apriltag_node",
            name="apriltag",
            output="screen",
            parameters=[{
                "image_transport": "compressed",
                "family": "36h11",
                "size": TAG_SIZE_M,
                "max_hamming": 0,
                "z_up": True,
                "tag.ids": [TAG_ID],
                "tag.frames": [TAG_FRAME],
                "tag.sizes": [TAG_SIZE_M],
            }],
            remappings=[
                # With image_transport=compressed the node subscribes to
                # <image_rect>/compressed automatically — supply the base
                # topic name here, not the /compressed suffix.
                ("image_rect", f"/zed/zed_node/{side}/image_rect_color"),
                ("camera_info", f"/zed/zed_node/{side}/camera_info"),
                ("detections", "/apriltag/detections"),
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("side", default_value="right",
                              description="Which ZED camera to use: 'left' or 'right'."),
        OpaqueFunction(function=launch_setup),
    ])
