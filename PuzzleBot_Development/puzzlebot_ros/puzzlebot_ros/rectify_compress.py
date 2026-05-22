#!/usr/bin/env python3
"""Capture CSI camera frames, rectify them, and publish raw + JPEG topics.

Opens the Argus CSI camera directly through an OpenCV GStreamer pipeline so
the frame never travels through a ROS pub/sub between capture and
rectification. That hop was previously capping us at ~15 fps; this node now
hits the native 30 fps of the 1280x720 sensor mode.

Published topics
----------------
``/camera/image_rect``            sensor_msgs/Image       (bgr8)
``/camera/image_rect/compressed`` sensor_msgs/CompressedImage (jpeg)

``cv_bridge`` is intentionally not used (its Python bindings are broken on
this Jetson) — Image messages are built manually with array.array, which
also takes the rclpy fast path for the ``data`` field. Without that fast
path, ``publish()`` of a 1280x720 image takes 1.3 s (rclpy iterates each
byte through an ``isinstance`` assertion).
"""

import array
import threading

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage


# Calibration from calibration_output/calibration.txt (1280x720, RMS 0.536 px)
DEFAULT_K = [
    809.551247, 0.0,        669.692191,
    0.0,        805.624908, 308.610156,
    0.0,        0.0,        1.0,
]
DEFAULT_D = [-0.352616, 0.184246, 0.002804, 0.000020, -0.061219]

# White-balance gains [B, G, R] from white_balance.yaml (method: patch-pick,
# computed against /camera/image_rect/compressed).
DEFAULT_WB_GAINS_BGR = [1.0161808729171753, 1.188890814781189, 0.8512064218521118]


def build_gst_pipeline(sensor_id: int, width: int, height: int, fps: int,
                       flip_method: int = 0) -> str:
    """nvarguscamerasrc -> NVMM NV12 -> nvvidconv (GPU) -> BGR appsink.

    drop=true and max-buffers=1 keep the latency at one frame even if the
    Python-side consumer falls slightly behind.
    """
    return (
        f'nvarguscamerasrc sensor-id={sensor_id} ! '
        f'video/x-raw(memory:NVMM), width={width}, height={height}, '
        f'framerate={fps}/1, format=NV12 ! '
        f'nvvidconv flip-method={flip_method} ! '
        f'video/x-raw, width={width}, height={height}, format=BGRx ! '
        f'videoconvert ! video/x-raw, format=BGR ! '
        f'appsink drop=true sync=false max-buffers=1'
    )


def ndarray_to_image(arr: np.ndarray, stamp, frame_id: str,
                     encoding: str) -> Image:
    """Wrap a numpy array as an Image message taking the rclpy fast path."""
    out = Image()
    out.header.stamp = stamp
    out.header.frame_id = frame_id
    out.height = arr.shape[0]
    out.width = arr.shape[1]
    out.encoding = encoding
    out.is_bigendian = 0
    out.step = arr.strides[0]
    out.data = array.array('B', arr.tobytes())
    return out


class CameraRectify(Node):
    def __init__(self):
        super().__init__('camera_rectify')

        # Parameters
        self.declare_parameter('sensor_id', 0)
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('fps', 30)
        self.declare_parameter('flip_method', 0)
        self.declare_parameter('output_ns', 'camera')
        self.declare_parameter('frame_id', 'camera')
        self.declare_parameter('camera_matrix', DEFAULT_K)
        self.declare_parameter('distortion', DEFAULT_D)
        self.declare_parameter('alpha', 0.0)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('publish_uncompressed', True)
        self.declare_parameter('wb_gains_bgr', DEFAULT_WB_GAINS_BGR)
        self.declare_parameter('wb_enable', True)

        sensor_id = int(self.get_parameter('sensor_id').value)
        w = int(self.get_parameter('image_width').value)
        h = int(self.get_parameter('image_height').value)
        fps = int(self.get_parameter('fps').value)
        flip_method = int(self.get_parameter('flip_method').value)
        ns = self.get_parameter('output_ns').value.rstrip('/')
        self._frame_id = self.get_parameter('frame_id').value
        K = np.array(self.get_parameter('camera_matrix').value,
                     dtype=np.float64).reshape(3, 3)
        D = np.array(self.get_parameter('distortion').value, dtype=np.float64)
        alpha = float(self.get_parameter('alpha').value)
        self._jpeg_q = int(self.get_parameter('jpeg_quality').value)
        publish_raw = bool(
            self.get_parameter('publish_uncompressed').value)

        # Use all four Jetson Nano cores in OpenCV (remap + JPEG encode).
        cv2.setNumThreads(4)

        # Precompute rectification maps. CV_16SC2 is ~3x faster than CV_32FC1
        # in cv2.remap.
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (w, h), cv2.CV_16SC2)
        self.get_logger().info(
            f'Precomputed rectify maps for {w}x{h}, alpha={alpha}')

        # Precompute a 3-channel BGR white-balance LUT. Applied with a
        # single cv2.LUT call per frame (~3 ms at 1280x720), which is
        # cheaper than per-channel multiply + clip + cast in numpy.
        gains = list(self.get_parameter('wb_gains_bgr').value)
        wb_enable = bool(self.get_parameter('wb_enable').value)
        if (wb_enable and len(gains) == 3
                and not all(abs(g - 1.0) < 1e-3 for g in gains)):
            lut = np.empty((1, 256, 3), dtype=np.uint8)
            ramp = np.arange(256, dtype=np.float32)
            for i, g in enumerate(gains):
                lut[0, :, i] = np.clip(ramp * float(g), 0, 255).astype(np.uint8)
            self._wb_lut = lut
            self.get_logger().info(
                f'WB enabled, gains BGR = '
                f'[{gains[0]:.4f}, {gains[1]:.4f}, {gains[2]:.4f}]')
        else:
            self._wb_lut = None
            self.get_logger().info('WB disabled (identity gains)')

        # Open the camera through the GStreamer pipeline.
        pipeline = build_gst_pipeline(sensor_id, w, h, fps, flip_method)
        self.get_logger().info(f'GStreamer pipeline:\n  {pipeline}')
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError('Failed to open CSI camera via GStreamer')

        # Pub QoS: sensor-data style (BE/depth=1).
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub_rect = (
            self.create_publisher(Image, f'/{ns}/image_rect', pub_qos)
            if publish_raw else None
        )
        self._pub_comp = self.create_publisher(
            CompressedImage, f'/{ns}/image_rect/compressed', pub_qos)

        # Pipeline: one capture+rectify thread, plus two independent
        # consumer threads (raw Image, JPEG). OpenCV's remap/imencode and
        # rclpy publish all release the GIL, so the consumers run on
        # separate cores while capture stays uninterrupted.
        #
        # The capture thread writes the latest rectified frame into a
        # single shared slot under a Condition; consumers wait for "new
        # frame", grab the reference, and publish. If a consumer is slower
        # than the capture rate the slot is simply overwritten — newest
        # frame wins, no queue buildup.
        # Initialize *all* shared state BEFORE starting any thread. The
        # capture thread races to `self._cap_frames += 1` on its first
        # iteration; touching that attribute before it exists raises an
        # AttributeError and silently kills the thread.
        self._stop = threading.Event()
        self._slot_cv = threading.Condition()
        self._slot = None        # (stamp, rect_bgr ndarray)
        self._rect_seq = 0
        self._raw_seq = -1
        self._jpeg_seq = -1
        self._cap_frames = 0
        self._raw_frames = 0
        self._jpeg_frames = 0
        self._t_last = self.get_clock().now()

        self._cap_thread = threading.Thread(target=self._capture_loop,
                                            daemon=True, name='capture')
        self._raw_thread = None
        if self._pub_rect is not None:
            self._raw_thread = threading.Thread(target=self._raw_pub_loop,
                                                daemon=True, name='raw_pub')
        self._jpeg_thread = threading.Thread(target=self._jpeg_pub_loop,
                                             daemon=True, name='jpeg_pub')
        self._cap_thread.start()
        if self._raw_thread is not None:
            self._raw_thread.start()
        self._jpeg_thread.start()

        self.create_timer(5.0, self._log_stats)

    def _capture_loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                self.get_logger().warn('cv2.VideoCapture.read() failed',
                                       throttle_duration_sec=2.0)
                continue
            stamp = self.get_clock().now().to_msg()
            rect = cv2.remap(frame, self._map1, self._map2,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
            if self._wb_lut is not None:
                rect = cv2.LUT(rect, self._wb_lut)
            with self._slot_cv:
                self._slot = (stamp, rect)
                self._rect_seq += 1
                self._slot_cv.notify_all()
            self._cap_frames += 1

    def _wait_for_frame(self, last_seq):
        with self._slot_cv:
            while (not self._stop.is_set()
                   and self._rect_seq == last_seq):
                self._slot_cv.wait(timeout=1.0)
            if self._stop.is_set():
                return None, None
            return self._rect_seq, self._slot

    def _raw_pub_loop(self):
        while not self._stop.is_set():
            seq, slot = self._wait_for_frame(self._raw_seq)
            if slot is None:
                continue
            self._raw_seq = seq
            stamp, rect = slot
            self._pub_rect.publish(
                ndarray_to_image(rect, stamp, self._frame_id, 'bgr8'))
            self._raw_frames += 1

    def _jpeg_pub_loop(self):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_q]
        while not self._stop.is_set():
            seq, slot = self._wait_for_frame(self._jpeg_seq)
            if slot is None:
                continue
            self._jpeg_seq = seq
            stamp, rect = slot
            ok, enc = cv2.imencode('.jpg', rect, params)
            if not ok:
                continue
            comp = CompressedImage()
            comp.header.stamp = stamp
            comp.header.frame_id = self._frame_id
            comp.format = 'jpeg'
            comp.data = array.array('B', enc.tobytes())
            self._pub_comp.publish(comp)
            self._jpeg_frames += 1

    def _log_stats(self):
        now = self.get_clock().now()
        dt = (now - self._t_last).nanoseconds * 1e-9
        if dt > 0:
            self.get_logger().info(
                f'cap={self._cap_frames/dt:.1f} '
                f'raw={self._raw_frames/dt:.1f} '
                f'jpeg={self._jpeg_frames/dt:.1f} fps '
                f'(jpeg q={self._jpeg_q})')
        self._cap_frames = self._raw_frames = self._jpeg_frames = 0
        self._t_last = now

    def shutdown(self):
        self._stop.set()
        with self._slot_cv:
            self._slot_cv.notify_all()
        for t in (self._cap_thread, self._raw_thread, self._jpeg_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()


def main():
    rclpy.init()
    node = CameraRectify()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
