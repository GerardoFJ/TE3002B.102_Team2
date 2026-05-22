#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    camera = Node(
        package='ros_deep_learning',
        executable='video_source',
        name='video_source',
        parameters=[
            {'resource': 'csi://0'},
            {'width': 1280},
            {'height': 720},
            {'codec': 'unknown'},
            {'loop': 0},
            {'latency': 2000},
        ],
        output='screen'
    )

    camera_info = Node(
        package='camera_info_publisher',
        executable='camera_info_publisher',
        name='camera_info_publisher',
        parameters=[
            {'camera_calibration_file': 'file:///home/puzzlebot/.ros/jetson_cam.yaml'},
            {'frame_id': 'camera'},
        ],
        output='screen'
    )

    # Subscribes to /video_source/raw and republishes as /video_source/compressed
    image_republisher = Node(
        package='image_transport',
        executable='republish',
        name='image_republisher',
        arguments=['raw', 'compressed'],
        remappings=[
            ('in', '/video_source/raw'),
            ('out/compressed', '/video_source/compressed'),
        ],
        output='screen'
    )

    return LaunchDescription([
        camera,
        camera_info,
        image_republisher,
    ])
