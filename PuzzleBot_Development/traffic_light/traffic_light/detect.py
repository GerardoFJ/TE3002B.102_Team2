import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from traffic_light.TraficLightDetection import TrafficLightDetection


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """Convert a sensor_msgs/Image to a BGR numpy array without cv_bridge."""
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('bgr8', 'rgb8'):
        img = raw.reshape((msg.height, msg.width, 3))
        if msg.encoding == 'rgb8':
            img = img[:, :, ::-1]  # RGB → BGR
    elif msg.encoding == 'mono8':
        img = raw.reshape((msg.height, msg.width))
    else:
        # fallback: treat as bgr8
        img = raw.reshape((msg.height, msg.width, -1))
    return np.ascontiguousarray(img)


def bgr_to_imgmsg(bgr: np.ndarray, stamp=None,
                  frame_id: str = 'traffic_light_debug') -> Image:
    """Pack a BGR numpy array into a sensor_msgs/Image (no cv_bridge)."""
    bgr = np.ascontiguousarray(bgr)
    msg = Image()
    msg.height = int(bgr.shape[0])
    msg.width = int(bgr.shape[1])
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = int(bgr.shape[1] * 3)
    msg.data = bgr.tobytes()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    return msg


class TrafficLightNode(Node):
    # States that allow the controller to move. red and yellow stop it.
    # 'none' (no light detected) also allows movement so detection dropouts
    # don't bring the robot to a halt.
    GO_STATES = {'green', 'none'}

    GO_TOPIC = '/traffic_light/go'
    STATE_TOPIC = '/traffic_light/state'
    DEBUG_IMAGE_TOPIC = '/traffic_light/debug_image'

    def __init__(self):
        super().__init__('traffic_light_detector')
        self.declare_parameter('debug', False)
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('go_publish_hz', 5.0)
        debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.confirm_frames = max(
            1, int(self.get_parameter('confirm_frames').value)
        )
        go_hz = max(0.1, float(self.get_parameter('go_publish_hz').value))
        self.detector = TrafficLightDetection(debug=debug)

        self.subscription = self.create_subscription(
            Image,
            '/video_source/raw',
            self._image_callback,
            10,
        )
        self.state_pub = self.create_publisher(String, self.STATE_TOPIC, 10)
        self.go_pub = self.create_publisher(Bool, self.GO_TOPIC, 10)
        # Live per-frame debug visualization — NOT gated by the confirmation
        # filter; publishes whatever the detector saw this frame.
        self.debug_image_pub = self.create_publisher(
            Image, self.DEBUG_IMAGE_TOPIC, 5
        )

        # Debounce: only update the published "go" once the same state has
        # repeated for confirm_frames in a row.
        self._candidate_state = None
        self._candidate_count = 0
        self._confirmed_state = None
        # Default to True ("go") so the receiver gets a sane heartbeat from
        # the very first tick, before any detection has happened.
        self._current_go = True

        # Steady heartbeat on /traffic_light/go — the receiver gets True or
        # False at go_publish_hz regardless of camera framerate or whether
        # a new confirmation has fired.
        self.go_timer = self.create_timer(1.0 / go_hz, self._publish_go)

        self.get_logger().info(
            f'Traffic light detector started '
            f'(confirm_frames={self.confirm_frames}, '
            f'go_publish_hz={go_hz:g}).'
        )

    def _image_callback(self, msg: Image):
        frame = imgmsg_to_bgr(msg)
        raw_state = self.detector.detect_state(frame)

        # Always publish the raw per-frame state for downstream debugging.
        self.state_pub.publish(String(data=raw_state))

        # Live debug visualization — every frame, no debounce.
        dbg = self.detector.last_debug_frame
        if dbg is not None:
            self.debug_image_pub.publish(
                bgr_to_imgmsg(dbg, stamp=msg.header.stamp)
            )

        confirmed = self._debounce(raw_state)
        if confirmed is not None:
            self._on_confirmed_state(confirmed)

    def _debounce(self, raw_state: str):
        """
        Return the new *confirmed* state once raw_state has repeated for
        `confirm_frames` consecutive frames and differs from the previous
        confirmed state. Otherwise return None.
        """
        if raw_state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = raw_state
            self._candidate_count = 1

        if (self._candidate_count >= self.confirm_frames
                and raw_state != self._confirmed_state):
            self._confirmed_state = raw_state
            return raw_state
        return None

    def _on_confirmed_state(self, state: str):
        """Update the latched go value and log the transition.

        Publishing is left to the timer (`_publish_go`) so the receiver
        always sees a steady stream at `go_publish_hz`.
        """
        go = state in self.GO_STATES
        self._current_go = go
        self.get_logger().info(f'detected: {state}  -> go={go}')

    def _publish_go(self):
        """Timer callback — publish the latest go value on every tick."""
        self.go_pub.publish(Bool(data=self._current_go))


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
