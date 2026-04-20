"""Launch del mini reto Semana 2.

Por defecto lanza UN solo coordinator_node que reproduce las 3 fases
secuencialmente publicando todos los topicos relevantes (Husky, ANYmal,
3 PuzzleBots, fases, cajas). Es el modo "demo": un solo proceso, todo
el ROS graph cubierto.

Si quieres ver los nodos individuales (uno por modulo), usa el launch
``modules.launch.py``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    terrain_arg = DeclareLaunchArgument(
        'husky_terrain', default_value='grass',
        description="Terreno del Husky (grass/mud/asphalt)")
    dt_arg = DeclareLaunchArgument(
        'dt', default_value='0.05',
        description="Periodo del replay [s]")
    loop_arg = DeclareLaunchArgument(
        'loop', default_value='false',
        description="Si true, reinicia el replay al terminar")

    coordinator = Node(
        package='mini_reto_s2',
        executable='coordinator_node',
        name='coordinator_node',
        output='screen',
        parameters=[{
            'husky_terrain': LaunchConfiguration('husky_terrain'),
            'dt': LaunchConfiguration('dt'),
            'loop': LaunchConfiguration('loop'),
        }],
    )

    return LaunchDescription([
        terrain_arg, dt_arg, loop_arg,
        coordinator,
    ])
