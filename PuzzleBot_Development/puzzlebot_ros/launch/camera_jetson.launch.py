#!/usr/bin/env python3
"""Camera pipeline launch for the Puzzlebot Jetson.

A single node (``camera_rectify``) owns the whole pipeline:

    nvarguscamerasrc (CSI) -> nvvidconv (NVMM->BGR)
        -> cv2.remap (rectify)
        -> publish /camera/image_rect (Image)
        -> publish /camera/image_rect/compressed (JPEG)

The previous launch fanned out across two nodes (ros_deep_learning's
``video_source`` + ``camera_info_publisher``) and an extra DDS hop, which
capped the achievable rate at ~15 fps even with a 30 fps sensor mode.
Both nodes are gone here. The JPEG topic is intended for Wi-Fi streaming.

The launch also forces the RMW to Cyclone DDS and points the runtime at a
config file with bumped socket / WHC buffer sizes for large image
messages. The Cyclone install must be sourced ahead of ``ros2 launch``
(``~/cyclone_ws/install/setup.bash``); the launch will still start under
Fast DDS if the library is missing.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


# Calibration from calibration_output/calibration.txt (1280x720)
CAMERA_MATRIX = [
    809.551247, 0.0,        669.692191,
    0.0,        805.624908, 308.610156,
    0.0,        0.0,        1.0,
]
DISTORTION = [-0.352616, 0.184246, 0.002804, 0.000020, -0.061219]

# White-balance gains [B, G, R] from white_balance.yaml (patch-pick method
# against /camera/image_rect/compressed).
WB_GAINS_BGR = [1.0161808729171753, 1.188890814781189, 0.8512064218521118]


def generate_launch_description():
    pkg_share = get_package_share_directory('puzzlebot_ros')
    cyclone_cfg = os.path.join(pkg_share, 'config', 'cyclonedds.xml')

    set_rmw = SetEnvironmentVariable(
        name='RMW_IMPLEMENTATION', value='rmw_cyclonedds_cpp')
    # CYCLONEDDS_URI accepts file:// URIs. If the file is missing, Cyclone
    # falls back to its built-in defaults and just logs a warning.
    set_cyclone_uri = SetEnvironmentVariable(
        name='CYCLONEDDS_URI', value=f'file://{cyclone_cfg}')

    camera = Node(
        package='puzzlebot_ros',
        executable='camera_rectify',
        name='camera_rectify',
        parameters=[{
            'sensor_id': 0,
            'image_width': 1280,
            'image_height': 720,
            'fps': 30,
            'flip_method': 0,
            'output_ns': 'camera',
            'frame_id': 'camera',
            'camera_matrix': CAMERA_MATRIX,
            'distortion': DISTORTION,
            'alpha': 0.0,            # 0 = crop black borders, 1 = full FOV
            'jpeg_quality': 80,      # good quality / bandwidth tradeoff
            'publish_uncompressed': True,
            'wb_gains_bgr': WB_GAINS_BGR,
            'wb_enable': True,
        }],
        output='screen',
    )

    return LaunchDescription([set_rmw, set_cyclone_uri, camera])
