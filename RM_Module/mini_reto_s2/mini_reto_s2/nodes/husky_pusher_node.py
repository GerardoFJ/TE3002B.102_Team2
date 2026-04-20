"""Wrapper rclpy del HuskyPusher (fase 1 del mini reto).

Precomputa la trayectoria completa de despeje del corredor con
``HuskyPusher.clear_corridor`` y la republica paso a paso en un
timer. Tambien publica un scan inicial del LiDAR 2D simulado.

Topicos publicados
------------------
/husky/odom        nav_msgs/Odometry           Pose 2D + velocidades reales
/husky/cmd_vel     geometry_msgs/Twist         Velocidad comandada por el ctrl
/husky/scan        sensor_msgs/LaserScan       Scan inicial (una vez)
/husky/status      std_msgs/String             Caja que esta empujando ('B1'..)
/boxes/big         std_msgs/Float32MultiArray  [x1,y1,x2,y2,x3,y3] cada tick

Parametros
----------
terrain    str   Terreno del Husky ('grass' por defecto)
dt         float Paso de simulacion / periodo del timer (default 0.05 s)
frame_id   str   Frame del odom (default 'map')
loop       bool  Si True, reinicia el replay al terminar (default False)
"""

import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from mini_reto_s2.husky_pusher import (
    HuskyPusher, CorridorWorld, Lidar2D,
)
from mini_reto_s2.robots_base import HuskyA200


def _yaw_to_quat(yaw):
    """Quaternion (x, y, z, w) que representa una rotacion solo en yaw."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class HuskyPusherNode(Node):

    def __init__(self):
        super().__init__('husky_pusher_node')

        # Parametros
        self.declare_parameter('terrain', 'grass')
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('loop', False)

        terrain = self.get_parameter('terrain').value
        self.dt = float(self.get_parameter('dt').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.loop = bool(self.get_parameter('loop').value)

        # Modelo + planner (logica pura)
        self.husky = HuskyA200()
        self.husky.set_terrain(terrain)
        self.husky.reset(x=0.0, y=2.0, theta=0.0)
        self.world = CorridorWorld()
        self.lidar = Lidar2D()
        self.pusher = HuskyPusher(self.husky, self.world, lidar=self.lidar)

        # Precomputar la trayectoria entera. La fisica vive aqui; el
        # nodo solo publica.
        self.get_logger().info(
            f"Precomputando despeje del corredor (terreno={terrain})...")
        self.log = self.pusher.clear_corridor(dt=self.dt)
        self.get_logger().info(
            f"  pasos={len(self.log['t'])}, exito={self.log['success']}")

        # Scan inicial (antes de moverse) para publicar en /husky/scan
        self.husky.reset(x=0.0, y=2.0, theta=0.0)
        self._initial_scan_angles, self._initial_scan_ranges = \
            self.lidar.scan(self.husky.get_pose(), self.world.boxes)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'husky/odom', 10)
        self.cmd_pub = self.create_publisher(Twist, 'husky/cmd_vel', 10)
        self.scan_pub = self.create_publisher(LaserScan, 'husky/scan', 10)
        self.status_pub = self.create_publisher(String, 'husky/status', 10)
        self.boxes_pub = self.create_publisher(
            Float32MultiArray, 'boxes/big', 10)

        # Replay
        self.idx = 0
        self.timer = self.create_timer(self.dt, self._tick)
        self.scan_published = False

    # ------------------------------------------------------------------
    def _publish_scan(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'husky/laser'
        # angles del lidar son en marco mundo en mi codigo; los
        # convierto a marco husky para que LaserScan sea coherente.
        # Tomo el theta inicial (0) -> son los mismos.
        n = len(self._initial_scan_ranges)
        if n >= 2:
            msg.angle_min = float(self._initial_scan_angles[0])
            msg.angle_max = float(self._initial_scan_angles[-1])
            msg.angle_increment = float(
                (msg.angle_max - msg.angle_min) / (n - 1))
        msg.range_min = 0.0
        msg.range_max = float(self.lidar.max_range)
        msg.ranges = [float(r) for r in self._initial_scan_ranges]
        self.scan_pub.publish(msg)

    # ------------------------------------------------------------------
    def _tick(self):
        if not self.scan_published:
            self._publish_scan()
            self.scan_published = True

        if self.idx >= len(self.log['t']):
            if self.loop:
                self.idx = 0
            else:
                return

        i = self.idx
        log = self.log
        now = self.get_clock().now().to_msg()

        # Odometry
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = 'husky/base_link'
        odom.pose.pose.position.x = float(log['x'][i])
        odom.pose.pose.position.y = float(log['y'][i])
        qx, qy, qz, qw = _yaw_to_quat(float(log['theta'][i]))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(log['v_real'][i])
        odom.twist.twist.angular.z = float(log['omega_real'][i])
        self.odom_pub.publish(odom)

        # cmd_vel
        cmd = Twist()
        cmd.linear.x = float(log['v_cmd'][i])
        cmd.angular.z = float(log['omega_cmd'][i])
        self.cmd_pub.publish(cmd)

        # status (caja activa)
        if i < len(log['phase']):
            self.status_pub.publish(String(data=log['phase'][i]))

        # cajas grandes
        big = Float32MultiArray()
        flat = []
        for name in ('B1', 'B2', 'B3'):
            x, y = log['boxes'][name][i]
            flat.extend([float(x), float(y)])
        big.data = flat
        self.boxes_pub.publish(big)

        self.idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = HuskyPusherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
