import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class CmdVelToWheels(Node):

    def __init__(self):
        super().__init__('cmd_vel_to_wheels')

        self.declare_parameter('wheel_radius', 0.05225)
        self.declare_parameter('wheel_separation', 0.17387)
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.wheel_separation = self.get_parameter('wheel_separation').get_parameter_value().double_value
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        self.left_pub = self.create_publisher(Float32, '/VelocitySetL', 10)
        self.right_pub = self.create_publisher(Float32, '/VelocitySetR', 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 50)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.wL = 0.0
        self.wR = 0.0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = None

        self.create_subscription(Float32, '/VelocityEncL', self.enc_left_cb, qos_profile_sensor_data)
        self.create_subscription(Float32, '/VelocityEncR', self.enc_right_cb, qos_profile_sensor_data)

        self.get_logger().info(
            f'cmd_vel_to_wheels ready (r={self.wheel_radius} m, '
            f'L={self.wheel_separation} m, publish_tf={self.publish_tf})'
        )

    def cmd_vel_cb(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z

        v_left = (v - w * self.wheel_separation / 2.0) / self.wheel_radius
        v_right = (v + w * self.wheel_separation / 2.0) / self.wheel_radius

        left_msg = Float32()
        left_msg.data = float(v_left)
        right_msg = Float32()
        right_msg.data = float(v_right)

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

    def enc_left_cb(self, msg: Float32):
        self.wL = float(msg.data)
        self._update_odom()

    def enc_right_cb(self, msg: Float32):
        self.wR = float(msg.data)
        self._update_odom()

    def _update_odom(self):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.last_time = now

        v = self.wheel_radius * (self.wR + self.wL) / 2.0
        w = self.wheel_radius * (self.wR - self.wL) / self.wheel_separation

        self.theta = math.atan2(
            math.sin(self.theta + w * dt),
            math.cos(self.theta + w * dt),
        )
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt

        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)
        stamp = now.to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToWheels()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
