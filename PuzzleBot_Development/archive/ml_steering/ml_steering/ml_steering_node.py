#!/usr/bin/env python3
"""ML steering predictor.

Subscribes to /hough_features (std_msgs/Float32MultiArray of 14 floats
from hough_features package), runs an XGBoost regressor, publishes
/navigation/omega (std_msgs/Float64) — a raw, unfiltered angular-rate
prediction in rad/s. Smoothing/clamping/traffic-light gating happens
downstream in safety_controller_node, NOT here.

The trained model lives at share/ml_steering/models/xgb_steering.json and
is loaded once at startup. Use `model_path` parameter to override.
"""

import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64

from ml_steering.tiny_xgb import TinyXGBPredictor


N_FEATURES = 14


class MlSteeringNode(Node):

    def __init__(self):
        super().__init__('ml_steering_node')

        default_model = ''
        try:
            share = get_package_share_directory('ml_steering')
            default_model = os.path.join(share, 'models', 'xgb_steering.json')
        except Exception:
            pass

        self.declare_parameter('model_path', default_model)
        self.declare_parameter('input_topic', '/hough_features')
        self.declare_parameter('output_topic', '/navigation/omega')

        model_path = self.get_parameter('model_path').value
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(
                f'XGBoost model not found at {model_path!r}. '
                'Pass -p model_path:=… or rebuild ml_steering after putting '
                'the trained model under models/.'
            )

        # Use the pure-numpy tree walker — Jetson-friendly, no xgboost
        # runtime dependency needed. Output matches xgboost.Booster.predict
        # to float32 rounding (verified).
        self.predictor = TinyXGBPredictor(model_path)
        self.get_logger().info(
            f'Loaded XGBoost model ({self.predictor.n_trees} trees) '
            f'from {model_path}'
        )

        self.pub = self.create_publisher(
            Float64, self.get_parameter('output_topic').value, 10)
        self.sub = self.create_subscription(
            Float32MultiArray,
            self.get_parameter('input_topic').value,
            self.on_features, 10)

        self._frames = 0
        self._last_log = self.get_clock().now()
        self.create_timer(5.0, self._log_stats)

    def on_features(self, msg: Float32MultiArray):
        if len(msg.data) != N_FEATURES:
            self.get_logger().warn(
                f'Expected {N_FEATURES} features, got {len(msg.data)} — skipping',
                throttle_duration_sec=2.0,
            )
            return
        x = np.asarray(msg.data, dtype=np.float32)
        pred = float(self.predictor.predict_one(x))
        out = Float64()
        out.data = pred
        self.pub.publish(out)
        self._frames += 1

    def _log_stats(self):
        now = self.get_clock().now()
        dt = (now - self._last_log).nanoseconds * 1e-9
        if dt > 0 and self._frames > 0:
            self.get_logger().info(
                f'predicted {self._frames} omegas in {dt:.1f}s '
                f'({self._frames / dt:.1f} Hz)'
            )
        self._frames = 0
        self._last_log = now


def main(args=None):
    rclpy.init(args=args)
    node = MlSteeringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
