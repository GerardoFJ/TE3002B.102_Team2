#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    puzzlebot_ros_share = get_package_share_directory('puzzlebot_ros')

    micro_ros_agent = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(puzzlebot_ros_share, 'launch', 'micro_ros_agent.launch.py')
        )
    )

    cmd_vel_to_wheels = Node(
        package='cmd_vel_to_wheels',
        executable='cmd_vel_to_wheels',
        name='cmd_vel_to_wheels',
        parameters=[{'publish_tf': True}],
        output='screen',
    )

    return LaunchDescription([
        micro_ros_agent,
        cmd_vel_to_wheels,
    ])
