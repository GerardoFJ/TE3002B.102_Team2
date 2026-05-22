from setuptools import find_packages, setup

package_name = 'cmd_vel_to_wheels'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='puzzlebot',
    maintainer_email='puzzlebot@todo.todo',
    description='Convert /cmd_vel Twist to per-wheel angular velocities (rad/s) for the Puzzlebot.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cmd_vel_to_wheels = cmd_vel_to_wheels.cmd_vel_to_wheels_node:main',
        ],
    },
)
