import math
import sys
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, String


class PID:
    def __init__(self, kp, ki, kd, out_min, out_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, error, now):
        if self.prev_time is None:
            self.prev_time = now
            self.prev_error = error
            return self._clamp(self.kp * error)
        dt = now - self.prev_time
        if dt <= 0.0:
            return self._clamp(self.kp * error + self.ki * self.integral)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        clamped = self._clamp(out)
        if clamped != out and self.ki != 0.0:
            self.integral -= error * dt
        self.prev_error = error
        self.prev_time = now
        return clamped

    def _clamp(self, x):
        return max(self.out_min, min(self.out_max, x))


def yaw_from_quat(qz, qw):
    return 2.0 * math.atan2(qz, qw)


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class WaypointController(Node):
    STATE_WAITING = 'waiting'
    STATE_NAVIGATE = 'navigate'
    STATE_PAUSED = 'paused'
    STATE_DONE = 'done'

    def __init__(self):
        super().__init__('square_controller')

        self.declare_parameter('waypoints_x', [0.0, 2.0, 2.0, 0.0])
        self.declare_parameter('waypoints_y', [2.0, 2.0, 0.0, 0.0])
        self.declare_parameter('max_linear_vel', 0.1)
        self.declare_parameter('max_angular_vel', 0.19)
        self.declare_parameter('position_tolerance', 0.03)
        self.declare_parameter('rotate_first_threshold', 0.7854)  # ~pi/4
        self.declare_parameter('allow_reverse', True)
        self.declare_parameter('reverse_max_distance', 0.20)
        self.declare_parameter('reverse_hysteresis', 0.15)
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('settle_ticks', 5)
        self.declare_parameter('kp_lin', 1.2)
        self.declare_parameter('ki_lin', 0.0)
        self.declare_parameter('kd_lin', 0.05)
        self.declare_parameter('kp_ang', 2.0)
        self.declare_parameter('ki_ang', 0.0)
        self.declare_parameter('kd_ang', 0.05)
        self.declare_parameter('wait_for_key', True)

        wx = list(self.get_parameter('waypoints_x').get_parameter_value().double_array_value)
        wy = list(self.get_parameter('waypoints_y').get_parameter_value().double_array_value)
        if len(wx) == 0 or len(wx) != len(wy):
            raise ValueError(
                f'waypoints_x and waypoints_y must be non-empty and same length '
                f'(got {len(wx)} and {len(wy)})'
            )
        self.waypoints = list(zip(wx, wy))

        self.max_v = float(self.get_parameter('max_linear_vel').value)
        self.max_w = float(self.get_parameter('max_angular_vel').value)
        self.pos_tol = float(self.get_parameter('position_tolerance').value)
        self.rotate_first_threshold = float(self.get_parameter('rotate_first_threshold').value)
        self.allow_reverse = bool(self.get_parameter('allow_reverse').value)
        self.reverse_max_distance = float(self.get_parameter('reverse_max_distance').value)
        self.reverse_hysteresis = float(self.get_parameter('reverse_hysteresis').value)
        self.settle_ticks = int(self.get_parameter('settle_ticks').value)
        rate = float(self.get_parameter('control_rate').value)

        self.pid_lin = PID(
            float(self.get_parameter('kp_lin').value),
            float(self.get_parameter('ki_lin').value),
            float(self.get_parameter('kd_lin').value),
            -self.max_v, self.max_v,
        )
        self.pid_ang = PID(
            float(self.get_parameter('kp_ang').value),
            float(self.get_parameter('ki_ang').value),
            float(self.get_parameter('kd_ang').value),
            -self.max_w, self.max_w,
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.dist_err_pub = self.create_publisher(Float32, '/controller/distance_error', 10)
        self.heading_err_pub = self.create_publisher(Float32, '/controller/heading_error', 10)
        self.target_x_pub = self.create_publisher(Float32, '/controller/target_x', 10)
        self.target_y_pub = self.create_publisher(Float32, '/controller/target_y', 10)
        self.current_x_pub = self.create_publisher(Float32, '/controller/current_x', 10)
        self.current_y_pub = self.create_publisher(Float32, '/controller/current_y', 10)
        self.state_pub = self.create_publisher(String, '/controller/state', 10)
        self.traffic_light_echo_pub = self.create_publisher(
            Bool, '/controller/traffic_light_go', 10)

        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(
            Bool, '/traffic_light/go', self.traffic_light_cb, 10)

        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.idx = 0
        self.settled_count = 0

        self.wait_for_key = bool(self.get_parameter('wait_for_key').value)
        if self.wait_for_key:
            self.state = self.STATE_WAITING
            self._key_thread = threading.Thread(
                target=self._wait_for_key_thread, daemon=True)
            self._key_thread.start()
            self.get_logger().info('press <Enter> to start waypoint following...')
        else:
            self.state = self.STATE_NAVIGATE

        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f'waypoint controller ready: {len(self.waypoints)} waypoints, '
            f'max_v={self.max_v}, max_w={self.max_w}, '
            f'pos_tol={self.pos_tol}, allow_reverse={self.allow_reverse}'
        )
        for i, (gx, gy) in enumerate(self.waypoints):
            self.get_logger().info(f'  wp{i}: ({gx:.3f}, {gy:.3f})')

    def _wait_for_key_thread(self):
        try:
            sys.stdin.readline()
        except Exception:
            return
        if self.state == self.STATE_WAITING:
            self.state = self.STATE_PAUSED
            self.get_logger().info(
                'key pressed; waiting for first /traffic_light/go message...')

    def traffic_light_cb(self, msg: Bool):
        go = bool(msg.data)
        self.traffic_light_echo_pub.publish(Bool(data=go))
        if self.state == self.STATE_DONE:
            return
        if go:
            if self.state == self.STATE_PAUSED:
                self.state = self.STATE_NAVIGATE
                self.get_logger().info(f'traffic_light=GO; resuming at wp{self.idx}')
        else:
            if self.state == self.STATE_NAVIGATE:
                self.state = self.STATE_PAUSED
                self._publish_cmd(0.0, 0.0)
                self.pid_lin.reset()
                self.pid_ang.reset()
                self.settled_count = 0
                self.get_logger().info(f'traffic_light=STOP; pausing at wp{self.idx}')

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quat(q.z, q.w)
        self.have_odom = True

    def _publish_cmd(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _publish_debug(self, gx, gy, dist_err, heading_err):
        self.target_x_pub.publish(Float32(data=float(gx)))
        self.target_y_pub.publish(Float32(data=float(gy)))
        self.current_x_pub.publish(Float32(data=float(self.x)))
        self.current_y_pub.publish(Float32(data=float(self.y)))
        self.dist_err_pub.publish(Float32(data=float(dist_err)))
        self.heading_err_pub.publish(Float32(data=float(heading_err)))
        self.state_pub.publish(String(data=f'{self.state}|wp{self.idx}'))

    def control_loop(self):
        if not self.have_odom:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.state == self.STATE_DONE:
            self._publish_cmd(0.0, 0.0)
            return

        if self.state == self.STATE_WAITING:
            self._publish_cmd(0.0, 0.0)
            return

        if self.state == self.STATE_PAUSED:
            self._publish_cmd(0.0, 0.0)
            gx, gy = self.waypoints[self.idx]
            dx = gx - self.x
            dy = gy - self.y
            self._publish_debug(
                gx, gy, math.hypot(dx, dy),
                normalize_angle(math.atan2(dy, dx) - self.yaw))
            return

        gx, gy = self.waypoints[self.idx]
        dx = gx - self.x
        dy = gy - self.y
        distance = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        heading_err = normalize_angle(angle_to_goal - self.yaw)

        forward = True
        if (self.allow_reverse
                and distance < self.reverse_max_distance
                and abs(heading_err) > math.pi / 2.0 + self.reverse_hysteresis):
            heading_err = normalize_angle(heading_err + math.pi)
            forward = False

        if distance < self.pos_tol:
            self.settled_count += 1
        else:
            self.settled_count = 0

        if self.settled_count >= self.settle_ticks:
            self._publish_cmd(0.0, 0.0)
            self._publish_debug(gx, gy, distance, heading_err)
            self.get_logger().info(
                f'wp{self.idx} reached at ({self.x:.3f}, {self.y:.3f}); '
                f'target=({gx:.3f}, {gy:.3f}), err={distance:.3f} m'
            )
            self.idx += 1
            self.settled_count = 0
            self.pid_lin.reset()
            self.pid_ang.reset()
            if self.idx >= len(self.waypoints):
                self.state = self.STATE_DONE
                self.get_logger().info('all waypoints reached')
            return

        w_cmd = self.pid_ang.compute(heading_err, now)
        w_cmd = max(-self.max_w, min(self.max_w, w_cmd))

        if abs(heading_err) > self.rotate_first_threshold:
            v_cmd = 0.0
            self.pid_lin.reset()
        else:
            v_mag = self.pid_lin.compute(distance, now)
            v_mag = max(0.0, min(self.max_v, v_mag))
            v_mag *= max(0.0, math.cos(heading_err))
            v_cmd = v_mag if forward else -v_mag

        self._publish_cmd(v_cmd, w_cmd)
        self._publish_debug(gx, gy, distance, heading_err)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish_cmd(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
