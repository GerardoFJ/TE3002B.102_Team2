"""Setup del paquete ROS2 mini_reto_s2 (ament_python)."""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'mini_reto_s2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['tests', 'tests.*']),
    data_files=[
        # ament index marker
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='daniel-wlg',
    maintainer_email='daniel-wlg@local',
    description=('Mini reto Semana 2 TE3002B - almacen robotico colaborativo '
                 '(Husky + ANYmal + 3 PuzzleBots).'),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'husky_pusher_node = mini_reto_s2.nodes.husky_pusher_node:main',
            'anymal_gait_node = mini_reto_s2.nodes.anymal_gait_node:main',
            'puzzlebot_arm_node = mini_reto_s2.nodes.puzzlebot_arm_node:main',
            'coordinator_node = mini_reto_s2.nodes.coordinator_node:main',
        ],
    },
)
