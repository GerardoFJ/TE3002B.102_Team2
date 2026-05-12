import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray


def create_apriltag_detector(dictionary=cv2.aruco.DICT_APRILTAG_36H11):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)


def detect_apriltags(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    detections = []

    if ids is None:
        return detections

    for i, corner in enumerate(corners):
        pts = corner[0]
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())

        detections.append({
            "id":       int(ids[i][0]),
            "bbox":     (x, y, w, h),
            "centroid": (cx, cy),
        })

    return detections


class AprilTagDetectorNode(Node):

    def __init__(self):
        super().__init__("apriltag_detector")

        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("reliability", "reliable")

        camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        reliability  = self.get_parameter("reliability").get_parameter_value().string_value

        reliability_policy = (
            ReliabilityPolicy.BEST_EFFORT
            if reliability == "best_effort"
            else ReliabilityPolicy.RELIABLE
        )
        qos = QoSProfile(
            reliability=reliability_policy,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub_image = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            qos,
        )

        self.pub_detections = self.create_publisher(
            AprilTagDetectionArray,
            "/apriltag/detections",
            10,
        )

        self.bridge   = CvBridge()
        self.detector = create_apriltag_detector()

        self.get_logger().info(f"Escuchando en '{camera_topic}' [{reliability}]")

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections = detect_apriltags(frame, self.detector)

        array_msg = AprilTagDetectionArray()
        array_msg.header = Header()
        array_msg.header.stamp = self.get_clock().now().to_msg()
        array_msg.header.frame_id = msg.header.frame_id

        for d in detections:
            cx, cy   = d["centroid"]
            x, y, w, h = d["bbox"]

            detection = AprilTagDetection()
            detection.id     = d["id"]
            detection.cx     = float(cx)
            detection.cy     = float(cy)
            detection.bbox_x = x
            detection.bbox_y = y
            detection.bbox_w  = w
            detection.bbox_h  = h

            array_msg.detections.append(detection)

            self.get_logger().info(
                f"Tag {d['id']} | centroid: ({cx},{cy}) | bbox: ({x},{y},{w},{h})"
            )

        self.pub_detections.publish(array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
