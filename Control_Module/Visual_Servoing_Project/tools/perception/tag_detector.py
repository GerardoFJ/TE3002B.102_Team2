#!/usr/bin/env python3
"""
Robust AprilTag detector + pose publisher.

Avoids `apriltag_ros 3.3`'s pose-ambiguity flipping by:
  1. Detecting the tag with OpenCV's ArUco DICT_APRILTAG_36h11.
  2. Running cv2.solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE) which returns
     **both** ambiguous pose solutions for a planar square target.
  3. Publishing the one with the lower reprojection error.

Subscribes (compressed image transport, ZED-friendly QoS) to
  <image_topic>/compressed   sensor_msgs/CompressedImage
  <camera_info_topic>        sensor_msgs/CameraInfo

Publishes the chosen tag pose via tf as a child of
  camera_info.header.frame_id  (e.g. zed_right_camera_optical_frame)
with child_frame_id = --tag-frame (default tag36h11_0).

Run this **instead of** apriltag_ros's apriltag_node — they both want to
publish the same tag frame, so don't run them together.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage, CameraInfo
from tf2_ros import TransformBroadcaster


def R_to_quat(R: np.ndarray) -> list[float]:
    """Rotation matrix → quaternion (x, y, z, w). Numerically robust."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return [(R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
                0.25 * s]
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return [0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[2, 1] - R[1, 2]) / s]
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return [(R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
                (R[0, 2] - R[2, 0]) / s]
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return [(R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
            (R[1, 0] - R[0, 1]) / s]


class TagDetector(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("tag_detector")
        self.tag_size: float = float(args.tag_size)
        self.tag_id: int = int(args.tag_id)
        self.tag_frame: str = str(args.tag_frame)
        self.prefer: str = str(args.prefer)
        self.ambiguity_eps: float = float(args.ambiguity_eps)
        self.bridge = CvBridge()
        self.K: np.ndarray | None = None
        self.D: np.ndarray | None = None
        self.cam_frame: str | None = None
        self.n_pub = 0

        # Tag corners in tag's own frame (z out of the tag plane).
        # Order MUST match cv2.aruco.detectMarkers output: TL, TR, BR, BL.
        s = self.tag_size / 2.0
        self.tag_corners_3d = np.array([
            [-s, +s, 0.0],
            [+s, +s, 0.0],
            [+s, -s, 0.0],
            [-s, -s, 0.0],
        ], dtype=np.float32)

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.aruco_params)

        self.br = TransformBroadcaster(self)

        # ZED publishes images RELIABLE / VOLATILE / KEEP_LAST(10).
        qos_image = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST, depth=5)
        qos_info = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(CompressedImage,
                                 args.image_topic + "/compressed",
                                 self.on_image, qos_image)
        self.create_subscription(CameraInfo, args.camera_info_topic,
                                 self.on_info, qos_info)

        self.get_logger().info(
            f"detector ready: image={args.image_topic}/compressed  "
            f"info={args.camera_info_topic}  tag={self.tag_frame} "
            f"(id={self.tag_id}, size={self.tag_size*1000:.0f} mm)")

    def on_info(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k).reshape(3, 3)
        self.D = np.array(msg.d)
        self.cam_frame = msg.header.frame_id

    def on_image(self, msg: CompressedImage) -> None:
        if self.K is None or self.cam_frame is None:
            return

        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return

        ids_flat = ids.flatten().tolist()
        if self.tag_id not in ids_flat:
            return

        i = ids_flat.index(self.tag_id)
        img_pts = corners[i][0].astype(np.float32)  # (4, 2): TL, TR, BR, BL

        # solvePnPGeneric with IPPE_SQUARE returns BOTH ambiguous pose
        # solutions for a planar square. For tags near-frontal to the
        # camera the reprojection errors are nearly equal and the
        # "far/flipped" solution often wins by 0.01 px — that's what
        # caused apriltag_ros to flip between solutions for us.
        #
        # We instead pick by smallest tvec.z when both solutions have
        # comparable reprojection error (within --ambiguity-eps px of
        # each other). When one solution is clearly better (errors
        # diverge), trust reprojection. Override with --prefer=far if you
        # actually have a faraway tag and a closer false solution.
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            self.tag_corners_3d, img_pts, self.K, self.D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if n < 1:
            return

        errs_flat = np.asarray(errs).flatten()
        if n == 1:
            best = 0
            zs = np.array([float(np.asarray(tvecs[0]).flatten()[2])])
        else:
            zs = np.array([float(np.asarray(tvecs[k]).flatten()[2]) for k in range(n)])
            best_reproj = int(np.argmin(errs_flat))
            # Are the two solutions effectively tied on reprojection error?
            if abs(errs_flat[0] - errs_flat[1]) < self.ambiguity_eps:
                # Tie-break by depth: nearer or farther, per CLI.
                best = int(np.argmin(zs)) if self.prefer == "near" else int(np.argmax(zs))
            else:
                best = best_reproj

        rvec = rvecs[best].flatten()
        tvec = tvecs[best].flatten()

        R, _ = cv2.Rodrigues(rvec)
        q = R_to_quat(R)

        tfs = TransformStamped()
        tfs.header.stamp = msg.header.stamp
        tfs.header.frame_id = self.cam_frame
        tfs.child_frame_id = self.tag_frame
        tfs.transform.translation.x = float(tvec[0])
        tfs.transform.translation.y = float(tvec[1])
        tfs.transform.translation.z = float(tvec[2])
        tfs.transform.rotation.x = float(q[0])
        tfs.transform.rotation.y = float(q[1])
        tfs.transform.rotation.z = float(q[2])
        tfs.transform.rotation.w = float(q[3])
        self.br.sendTransform(tfs)

        self.n_pub += 1
        if self.n_pub % 30 == 1:
            # Log every ~1.5 s at 20 Hz to confirm we're alive without spamming.
            dist = float(np.linalg.norm(tvec))
            if n >= 2:
                other_z = zs[1 - best]
                other_err = errs_flat[1 - best]
                self.get_logger().info(
                    f"tag id {self.tag_id}: chose z={zs[best]*100:.1f} cm "
                    f"(err {errs_flat[best]:.3f} px) over z={other_z*100:.1f} cm "
                    f"(err {other_err:.3f} px), prefer='{self.prefer}'")
            else:
                self.get_logger().info(
                    f"tag id {self.tag_id}: dist={dist*100:.1f} cm "
                    f"(only one IPPE solution returned)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-topic", default="/zed/zed_node/right/image_rect_color",
                   help="Base image topic; '/compressed' is appended.")
    p.add_argument("--camera-info-topic", default="/zed/zed_node/right/camera_info")
    p.add_argument("--tag-size", type=float, default=0.133,
                   help="Tag side length (m), black border to black border.")
    p.add_argument("--tag-id", type=int, default=0)
    p.add_argument("--tag-frame", default="tag36h11_0",
                   help="Frame name to publish the tag pose under.")
    p.add_argument("--prefer", choices=("near", "far"), default="near",
                   help="When IPPE returns two pose candidates with similar "
                        "reprojection error, pick the one closer to ('near') "
                        "or farther from ('far') the camera. Default near, "
                        "which is correct for our 0.4 m setup.")
    p.add_argument("--ambiguity-eps", type=float, default=0.5,
                   help="If |err_0 - err_1| < this (px), treat the two pose "
                        "solutions as tied and pick by --prefer. Otherwise "
                        "use the lower-error one. Default 0.5 px.")
    args = p.parse_args()

    rclpy.init()
    node = TagDetector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
