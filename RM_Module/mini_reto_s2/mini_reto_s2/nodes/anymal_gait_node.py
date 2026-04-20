"""Wrapper rclpy del trote del ANYmal (fase 2 del mini reto).

Precomputa la simulacion de ``simulate_anymal_to_target`` (trote +
navigator) y la republica paso a paso. Publica pose de la base, joint
states de las 12 articulaciones, det(J) por pata y la velocidad
comandada.

Topicos publicados
------------------
/anymal/odom         nav_msgs/Odometry          Pose 2D base + cmd vel
/anymal/cmd_vel      geometry_msgs/Twist        v_forward, omega_yaw
/anymal/joint_states sensor_msgs/JointState     12 articulaciones
/anymal/det_J        std_msgs/Float32MultiArray det por pata [LF, RF, LH, RH]
/anymal/status       std_msgs/String            'walking' / 'arrived'

Parametros
----------
target_x  float Destino X (default 11.0)
target_y  float Destino Y (default 3.6)
dt        float Periodo del timer y de integracion (default 0.005)
publish_decim int  Cada cuantos ticks publica al ROS graph (default 10)
loop      bool  Reinicia el replay al terminar (default False)
"""

import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from mini_reto_s2.anymal_gait import (
    ANYmalTrotGait, ANYmalNavigator, simulate_anymal_to_target,
)
from mini_reto_s2.robots_base import ANYmal


# Las 12 articulaciones del ANYmal en el orden del log:
#   q[0:3]   = LF (HAA, HFE, KFE)
#   q[3:6]   = RF
#   q[6:9]   = LH
#   q[9:12]  = RH
JOINT_NAMES = [
    'LF_HAA', 'LF_HFE', 'LF_KFE',
    'RF_HAA', 'RF_HFE', 'RF_KFE',
    'LH_HAA', 'LH_HFE', 'LH_KFE',
    'RH_HAA', 'RH_HFE', 'RH_KFE',
]
LEG_ORDER = ('LF', 'RF', 'LH', 'RH')


def _yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class AnymalGaitNode(Node):

    def __init__(self):
        super().__init__('anymal_gait_node')

        self.declare_parameter('target_x', 11.0)
        self.declare_parameter('target_y', 3.6)
        self.declare_parameter('dt', 0.005)
        self.declare_parameter('publish_decim', 10)
        self.declare_parameter('loop', False)

        target_x = float(self.get_parameter('target_x').value)
        target_y = float(self.get_parameter('target_y').value)
        self.dt = float(self.get_parameter('dt').value)
        self.decim = int(self.get_parameter('publish_decim').value)
        self.loop = bool(self.get_parameter('loop').value)

        # Modelo + gait + nav (logica pura)
        self.anymal = ANYmal()
        self.gait = ANYmalTrotGait(self.anymal)
        self.nav = ANYmalNavigator()

        self.get_logger().info(
            f"Precomputando trote a target=({target_x:.2f}, {target_y:.2f})...")
        self.log = simulate_anymal_to_target(
            self.anymal, target_xy=(target_x, target_y),
            gait=self.gait, navigator=self.nav, dt=self.dt)
        self.get_logger().info(
            f"  pasos={len(self.log['t'])}, exito={self.log['success']}, "
            f"err_final={self.log['final_error']:.3f} m, "
            f"violaciones det(J)={self.log['violations']}")

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'anymal/odom', 10)
        self.cmd_pub = self.create_publisher(Twist, 'anymal/cmd_vel', 10)
        self.joint_pub = self.create_publisher(
            JointState, 'anymal/joint_states', 10)
        self.detj_pub = self.create_publisher(
            Float32MultiArray, 'anymal/det_J', 10)
        self.status_pub = self.create_publisher(String, 'anymal/status', 10)

        # Replay (publicamos cada `decim` pasos para no inundar)
        self.idx = 0
        self.timer = self.create_timer(self.dt * self.decim, self._tick)
        self._arrived_announced = False

    # ------------------------------------------------------------------
    def _tick(self):
        if self.idx >= len(self.log['t']):
            if not self._arrived_announced:
                self.status_pub.publish(String(data='arrived'))
                self._arrived_announced = True
            if self.loop:
                self.idx = 0
                self._arrived_announced = False
            else:
                return

        i = self.idx
        log = self.log
        now = self.get_clock().now().to_msg()

        # Odom
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'anymal/base'
        odom.pose.pose.position.x = float(log['base_x'][i])
        odom.pose.pose.position.y = float(log['base_y'][i])
        qx, qy, qz, qw = _yaw_to_quat(float(log['base_yaw'][i]))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(log['v_cmd'][i])
        odom.twist.twist.angular.z = float(log['omega_cmd'][i])
        self.odom_pub.publish(odom)

        # cmd
        cmd = Twist()
        cmd.linear.x = float(log['v_cmd'][i])
        cmd.angular.z = float(log['omega_cmd'][i])
        self.cmd_pub.publish(cmd)

        # joint states
        js = JointState()
        js.header.stamp = now
        js.name = JOINT_NAMES
        q_full = log['q'][i]            # array de 12
        js.position = [float(v) for v in q_full]
        self.joint_pub.publish(js)

        # det_J
        det = Float32MultiArray()
        det.data = [float(log['det_J'][leg][i]) for leg in LEG_ORDER]
        self.detj_pub.publish(det)

        # status
        self.status_pub.publish(String(data='walking'))

        self.idx += self.decim


def main(args=None):
    rclpy.init(args=args)
    node = AnymalGaitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
