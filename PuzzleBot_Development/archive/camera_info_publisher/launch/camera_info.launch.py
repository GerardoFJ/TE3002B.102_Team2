#!/usr/bin/env python3

import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='camera_info_publisher',
            executable='camera_info_publisher',
            name='camera_info_publisher',
            parameters=[
                {"camera_calibration_file": "file:///home/puzzlebot/.ros/jetson_cam.yaml"},
                {"frame_id": "camera"}
            ],
            output='screen')
  ])

