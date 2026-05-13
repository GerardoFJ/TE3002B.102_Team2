#!/usr/bin/env python3
"""
Start MoveIt Servo for the xArm6 + Frida gripper, configured to ingest
TwistStamped commands on /servo_node/delta_twist_cmds and emit
JointTrajectory on /xarm6_traj_controller/joint_trajectory.

Designed to run ALONGSIDE an already-running `frida_moveit_config.launch.py`
— it adds only the servo_node (and re-uses the same URDF/SRDF/kinematics so
TF and collision checks line up with what move_group already sees).

Launch args mirror the most important ones from frida_moveit_config so the
config matches; defaults are the same.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from arm_pkg.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file


def launch_setup(context, *args, **kwargs):
    # Same defaults as frida_moveit_config.launch.py.
    robot_ip = LaunchConfiguration("robot_ip").perform(context)
    dof_str = LaunchConfiguration("dof").perform(context)
    dof = int(dof_str)
    robot_type = LaunchConfiguration("robot_type").perform(context)
    prefix = LaunchConfiguration("prefix").perform(context)
    hw_ns = LaunchConfiguration("hw_ns").perform(context)
    attach_to = LaunchConfiguration("attach_to").perform(context)
    limited = LaunchConfiguration("limited").perform(context).lower() in ("true", "1")

    # Mirror frida_moveit_config's choice for real hardware.
    ros2_control_plugin = "uf_robot_hardware/UFRobotSystemHardware"
    controllers_name = "controllers"

    # Temporary ros2_control params file (same helper frida uses).
    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(
            get_package_share_directory("xarm_controller"),
            "config",
            f"{robot_type}{dof}_controllers.yaml",
        ),
        prefix=prefix,
        add_gripper=False,
        add_bio_gripper=False,
        ros_namespace="",
        robot_type=robot_type,
    )

    moveit_config = MoveItConfigsBuilder(
        context=context,
        controllers_name=controllers_name,
        robot_ip=robot_ip,
        report_type="normal",
        baud_checkset=True,
        default_gripper_baud=2000000,
        dof=dof,
        robot_type=robot_type,
        prefix=prefix,
        hw_ns=hw_ns,
        limited=limited,
        effort_control=False,
        velocity_control=False,
        attach_to=attach_to,
        attach_xyz='"0 0 0"',
        attach_rpy='"0 0 0"',
        mesh_suffix="stl",
        kinematics_suffix="",
        ros2_control_plugin=ros2_control_plugin,
        ros2_control_params=ros2_control_params,
        add_gripper=False,
        add_vacuum_gripper=False,
        add_bio_gripper=False,
        add_realsense_d435i=False,
        add_d435i_links=True,
        add_other_geometry=False,
    ).to_moveit_configs()

    # MoveIt Servo parameters. Mirrors the UFactory xarm_moveit_servo
    # config but inlined so we don't depend on that package being built in
    # the workspace.
    servo_params = {
        "use_gazebo": False,
        # Incoming command type & scaling
        "command_in_type": "speed_units",   # m/s and rad/s
        "scale": {"linear": 0.4, "rotational": 0.8, "joint": 0.5},
        # Outgoing command publishing
        "publish_period": 0.034,
        "low_latency_mode": False,
        "command_out_type": "trajectory_msgs/JointTrajectory",
        "command_out_topic": f"/{robot_type}{dof}_traj_controller/joint_trajectory",
        "publish_joint_positions": True,
        "publish_joint_velocities": False,
        "publish_joint_accelerations": False,
        # Filtering — higher = smoother output, more lag. 2.0 is the UFactory
        # default and is responsive but jumpy; 6.0–8.0 is much smoother and
        # is what you want when the MPC commands have step-like changes at
        # the controller's input rate.
        "low_pass_filter_coeff": 6.0,
        # MoveIt config refs
        "move_group_name": f"{robot_type}{dof}",
        "planning_frame": "link_base",
        "ee_frame_name": "link_eef",
        "robot_link_command_frame": "link_base",
        # Stopping behavior
        "incoming_command_timeout": 0.2,
        "num_outgoing_halt_msgs_to_publish": 4,
        # Singularity / joint-limit handling
        "lower_singularity_threshold": 17.0,
        "hard_stop_singularity_threshold": 30.0,
        "joint_limit_margin": 0.1,
        # Topic names (relative -> become /servo_node/...)
        "cartesian_command_in_topic": "~/delta_twist_cmds",
        "joint_command_in_topic": "~/delta_joint_cmds",
        "joint_topic": "/joint_states",
        "status_topic": "~/status",
        # Collision checking
        "check_collisions": True,
        "collision_check_rate": 10.0,
        "collision_check_type": "threshold_distance",
        "self_collision_proximity_threshold": 0.01,
        "scene_collision_proximity_threshold": 0.02,
        "collision_distance_safety_factor": 1000.0,
        "min_allowable_collision_distance": 0.01,
    }

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            {"moveit_servo": servo_params},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
    )

    return [servo_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="192.168.31.180"),
        DeclareLaunchArgument("dof", default_value="6"),
        DeclareLaunchArgument("robot_type", default_value="xarm"),
        DeclareLaunchArgument("prefix", default_value=""),
        DeclareLaunchArgument("hw_ns", default_value="xarm"),
        DeclareLaunchArgument("attach_to", default_value="xarm_base"),
        DeclareLaunchArgument("limited", default_value="false",
                              description="Use the limited (±π) joint range for j4/j6 "
                                          "instead of the hardware-supported ±2π. The xArm6 "
                                          "URDF defaults this to true but the physical arm "
                                          "supports the wider range — keep this 'false' "
                                          "unless your MoveIt motion planning needs the cap."),
        OpaqueFunction(function=launch_setup),
    ])
