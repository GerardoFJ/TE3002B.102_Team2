#!/usr/bin/env python3
import atexit
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

WEIGHTS_DEFAULT = Path(__file__).parent / "./runs/detect/señales_trafico/weights/best.pt"
TOPIC          = "/Image"
CONF           = 0.74
BLUR_THRESHOLD = 10.0
MIN_BOX        = 30
SAVE_VIDEO     = False      # True para guardar el video de las detecciones
VIDEO_PATH     = "detection_output.mp4"
VIDEO_FPS      = 20

COLORS = [
    (255,  56,  56),
    ( 56, 255,  56),
    ( 56,  56, 255),
    (255, 200,  56),
    (200,  56, 255),
]


def is_blurry(frame: np.ndarray, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

#FUNCION PARA CONVERTIR TOPICO DE IMG A CV2 PORQUE CVBRIDGE NOS CAUSO PROBLEMAS CON LIBRERIAS DE YOLO
def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    dtype = np.uint8
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels if channels > 1 else 1)
    if msg.encoding == "rgb8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "rgba8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif msg.encoding == "bgra8":
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif msg.encoding == "mono8":
        arr = cv2.cvtColor(arr.squeeze(), cv2.COLOR_GRAY2BGR)
    return arr


class YoloDetectorNode(Node):
    def __init__(self, topic: str, weights: str, conf: float, blur_threshold: float, min_box: int):
        super().__init__("yolo_detector")
        self.model = YOLO(weights)
        self.conf = conf
        self.blur_threshold = blur_threshold
        self.min_box = min_box
        self.writer = None
        self.get_logger().info(f"Loaded weights: {weights}")
        self.get_logger().info(f"Subscribing to: {topic}")
        if SAVE_VIDEO:
            self.get_logger().info(f"Recording video to: {VIDEO_PATH}")

        self.sub = self.create_subscription(Image, topic, self._cb, 10)

    def _cb(self, msg: Image):
        try:
            frame = imgmsg_to_bgr(msg)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        if is_blurry(frame, self.blur_threshold):
            return

        results = self.model.predict(frame, conf=self.conf, imgsz=640, verbose=False)
        annotated = self._draw(frame, results[0])
        cv2.imshow("YOLO Detections", annotated)
        cv2.waitKey(1)

        if SAVE_VIDEO:
            if self.writer is None:
                h, w = annotated.shape[:2]
                self.writer = cv2.VideoWriter(
                    VIDEO_PATH,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    VIDEO_FPS,
                    (w, h),
                )
            self.writer.write(annotated)

    def _draw(self, img, result):
        out = img.copy()
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            if (x2 - x1) < self.min_box or (y2 - y1) < self.min_box:
                continue
            cls = int(box.cls)
            label = f"{self.model.names[cls]} {float(box.conf):.2f}"
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out


def main():
    rclpy.init()
    node = YoloDetectorNode(TOPIC, str(WEIGHTS_DEFAULT), CONF, BLUR_THRESHOLD, MIN_BOX)

    def _cleanup():
        if node.writer is not None:
            node.writer.release()
            print(f"Video saved to: {VIDEO_PATH}")
        cv2.destroyAllWindows()

    atexit.register(_cleanup)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
