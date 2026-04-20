"""Launch alternativo: lanza los 4 wrappers individualmente.

Util para inspeccionar cada modulo aislado en el ROS graph (con
``ros2 node list`` veras 6 nodos: husky_pusher_node, anymal_gait_node,
y 3 puzzlebot_arm_node con roles A/B/C). NO ejecutan en estricto orden
secuencial; cada uno reproduce su fase en cuanto arranca.

Uso tipico:
    ros2 launch mini_reto_s2 modules.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    husky = Node(
        package='mini_reto_s2',
        executable='husky_pusher_node',
        name='husky_pusher_node',
        output='screen',
        parameters=[{'terrain': 'grass', 'dt': 0.05}],
    )
    anymal = Node(
        package='mini_reto_s2',
        executable='anymal_gait_node',
        name='anymal_gait_node',
        output='screen',
        parameters=[{'target_x': 11.0, 'target_y': 3.6,
                     'dt': 0.005, 'publish_decim': 10}],
    )
    pb_c = Node(
        package='mini_reto_s2',
        executable='puzzlebot_arm_node',
        name='pb_c',
        output='screen',
        parameters=[{'name': 'pb_c', 'role_box': 'C', 'stack_layer': 0}],
    )
    pb_b = Node(
        package='mini_reto_s2',
        executable='puzzlebot_arm_node',
        name='pb_b',
        output='screen',
        parameters=[{'name': 'pb_b', 'role_box': 'B', 'stack_layer': 1}],
    )
    pb_a = Node(
        package='mini_reto_s2',
        executable='puzzlebot_arm_node',
        name='pb_a',
        output='screen',
        parameters=[{'name': 'pb_a', 'role_box': 'A', 'stack_layer': 2}],
    )
    return LaunchDescription([husky, anymal, pb_c, pb_b, pb_a])
