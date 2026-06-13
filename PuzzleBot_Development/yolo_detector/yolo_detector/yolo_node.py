#!/usr/bin/env python3
"""YOLO detection node.

Subscribes to a CompressedImage camera topic, runs an Ultralytics YOLO model
on each frame, and republishes a *new* CompressedImage with the detection
boxes / labels drawn on top.

Design notes (kept consistent with the C++ nodes in this workspace):
  * No cv_bridge. Compressed JPEG is decoded with cv2.imdecode and the
    annotated frame is re-encoded with cv2.imencode, exactly like
    traffic_light/src/detect.cpp and line_follower/src/follow_line.cpp.
  * SensorData QoS, keep_last(1): naturally drops stale frames so inference
    never builds a backlog when the model is slower than the camera FPS.
  * The whole pipeline (decode -> infer -> draw -> encode -> publish) runs in
    the subscription callback. With a depth-1 sensor-data queue, frames that
    arrive while inference is running are simply dropped.
"""

import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# numpy>=2 (the training machine) pickles arrays under numpy._core, which was
# renamed from numpy.core; alias the old modules so torch can unpickle best.pt
# on this numpy 1.23 install (Python 3.8 cannot run numpy>=1.26).
import importlib as _il
import sys as _sys
for _new in ("numpy._core", "numpy._core.multiarray", "numpy._core.umath",
             "numpy._core.numeric", "numpy._core._multiarray_umath"):
    try:
        _sys.modules.setdefault(_new, _il.import_module(_new.replace("numpy._core", "numpy.core")))
    except ImportError:
        pass

from ultralytics import YOLO


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        # ---- parameters -----------------------------------------------------
        self.input_topic = self.declare_parameter(
            "input_topic", "/camera/image_rect/compressed"
        ).value
        self.output_topic = self.declare_parameter(
            "output_topic", "/yolo/image/compressed"
        ).value
        self.model_path = self.declare_parameter("model_path", "yolo26n.pt").value
        self.conf = float(self.declare_parameter("conf", 0.25).value)
        self.iou = float(self.declare_parameter("iou", 0.45).value)
        self.imgsz = int(self.declare_parameter("imgsz", 640).value)
        # device: '' lets Ultralytics auto-pick (CUDA if available), or set
        # 'cpu' / '0' explicitly.
        self.device = self.declare_parameter("device", "").value
        self.half = bool(self.declare_parameter("half", False).value)
        self.jpeg_quality = int(self.declare_parameter("jpeg_quality", 70).value)
        # Optional class filter, e.g. [0, 2, 7]. Empty -> all classes.
        self.classes = list(self.declare_parameter("classes", []).value)

        # ---- model ----------------------------------------------------------
        self.get_logger().info("loading YOLO model '%s' ..." % self.model_path)
        self.model = YOLO(self.model_path)
        # Warm up once so the first real frame isn't slowed by lazy init.
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self._infer(dummy)
            self.get_logger().info("model warm-up done")
        except Exception as e:  # noqa: BLE001 - warm-up failure is non-fatal
            self.get_logger().warn("warm-up failed (continuing): %s" % e)

        # ---- ROS I/O --------------------------------------------------------
        self.pub = self.create_publisher(
            CompressedImage, self.output_topic, qos_profile_sensor_data
        )
        # Machine-readable detections: comma list of "name:conf" entries
        # (empty string when nothing is detected). Consumed by
        # line_follower/follow_single_line for lost-line decisions.
        self.det_topic = self.declare_parameter(
            "detections_topic", "/yolo/detections"
        ).value
        self.det_pub = self.create_publisher(String, self.det_topic, 10)
        self.sub = self.create_subscription(
            CompressedImage,
            self.input_topic,
            self.on_image,
            qos_profile_sensor_data,
        )

        # ---- stats ----------------------------------------------------------
        self._n = 0
        self._sum_ms = 0.0
        self.create_timer(5.0, self._log_stats)

        self.get_logger().info(
            "yolo_detector started: in=%s out=%s imgsz=%d conf=%.2f device='%s'"
            % (
                self.input_topic,
                self.output_topic,
                self.imgsz,
                self.conf,
                self.device if self.device else "auto",
            )
        )

    # -------------------------------------------------------------------------
    def _infer(self, bgr):
        kwargs = dict(
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            half=self.half,
        )
        if self.device != "":
            kwargs["device"] = self.device
        if self.classes:
            kwargs["classes"] = self.classes
        return self.model.predict(bgr, **kwargs)

    def on_image(self, msg: CompressedImage):
        t0 = time.monotonic()

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn("imdecode failed (bad JPEG frame)")
            return

        results = self._infer(bgr)

        det = String()
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            names = results[0].names
            items = []
            for b in boxes:
                # area = bbox fraction of the image (resolution-independent),
                # used downstream to reject far-away signs/lights.
                # cx = normalized bbox center x (0=left, 1=right), used to
                # prefer the rightmost of two similar-size signals.
                xywhn = b.xywhn[0]
                area = float(xywhn[2]) * float(xywhn[3])
                cx = float(xywhn[0])
                items.append(
                    "%s:%.2f:%.4f:%.3f"
                    % (names[int(b.cls)], float(b.conf), area, cx)
                )
            det.data = ",".join(items)
        self.det_pub.publish(det)

        # results[0].plot() returns a BGR ndarray with boxes + labels drawn.
        annotated = results[0].plot()

        ok, enc = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            self.get_logger().warn("imencode failed")
            return

        out = CompressedImage()
        out.header = msg.header  # preserve stamp + frame_id
        out.format = "jpeg"
        out.data = enc.tobytes()
        self.pub.publish(out)

        self._n += 1
        self._sum_ms += (time.monotonic() - t0) * 1000.0

    def _log_stats(self):
        if self._n == 0:
            self.get_logger().info("no frames in last 5s (is the camera up?)")
            return
        fps = self._n / 5.0
        mean_ms = self._sum_ms / self._n
        self.get_logger().info(
            "yolo: %.1f fps, %.1f ms/frame (%d frames)" % (fps, mean_ms, self._n)
        )
        self._n = 0
        self._sum_ms = 0.0


def main(args=None):
    try:
        rclpy.init(args=args)
        node = YoloNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
    except (KeyboardInterrupt, ExternalShutdownException):
        # Clean exit on Ctrl-C / SIGTERM (e.g. `timeout`, launch shutdown).
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
