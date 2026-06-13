#!/usr/bin/env python3
"""Live viewer of the REAL running lane_pilot nodes — what the robot actually
detects and plans (NOT a re-computation like lane_debug_qt.py).

Subscribes to the debug image topics the nodes publish and shows them together:
  raw camera | detection (bird's-eye, from lane_ipm_node)
             | planning (trajectory + lookahead + v/omega, from lane_pilot_node)

Run the follow first (e.g. `ros2 launch lane_pilot lane_pilot.launch.py
homography_file:=...`), then run this viewer on your computer:

  xhost +local:docker
  docker run --rm -it --network host -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix -v <WS>:/ws ros2-generic:latest bash -lc '
      source /opt/ros/humble/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
      python3 /ws/lane_pilot/tools/lane_view.py'

The debug images are LAZY (published only while something subscribes), so they
appear as soon as this viewer connects. If a panel stays black, that node isn't
running or isn't publishing that topic.
"""
import os
import sys
import time

import numpy as np
import cv2

# opencv-python sets QT_QPA_PLATFORM_PLUGIN_PATH to its bundled Qt ON IMPORT,
# which clashes with PyQt5 ("could not load xcb in cv2/qt/plugins"). Drop it
# AFTER importing cv2 (and before PyQt5 / QApplication) so PyQt5 uses its own Qt.
for _v in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_DEBUG_PLUGINS"):
    os.environ.pop(_v, None)

from PyQt5 import QtCore, QtGui, QtWidgets

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

TOPICS = [
    ("/camera/image_rect/compressed", "raw camera"),
    ("/lane/ipm_debug/compressed", "detection (bird's-eye)"),
    ("/lane/pilot_debug/compressed", "planning (trajectory)"),
]


def to_pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


class Viewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("lane_pilot LIVE (real robot output)")
        rclpy.init()
        self.node = rclpy.create_node("lane_view")
        self.frames = {}
        self.last = {}
        for topic, _ in TOPICS:
            self.node.create_subscription(
                CompressedImage, topic, self._mk(topic), qos_profile_sensor_data)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        row = QtWidgets.QHBoxLayout(central)
        self.labels = {}
        self.titles = {}
        for topic, name in TOPICS:
            col = QtWidgets.QVBoxLayout()
            t = QtWidgets.QLabel(name)
            t.setStyleSheet("color:#fff; font-weight:bold;")
            img = QtWidgets.QLabel("waiting...")
            img.setStyleSheet("color:#888; background:#111;")
            img.setMinimumSize(360, 260)
            img.setAlignment(QtCore.Qt.AlignCenter)
            self.labels[topic] = img
            self.titles[topic] = t
            col.addWidget(t)
            col.addWidget(img, 1)
            row.addLayout(col, 1)
        self.setStyleSheet("background:#222;")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    def _mk(self, topic):
        def cb(m):
            img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.frames[topic] = img
                self.last[topic] = time.time()
        return cb

    def tick(self):
        rclpy.spin_once(self.node, timeout_sec=0.0)
        now = time.time()
        for topic, name in TOPICS:
            if topic in self.frames:
                age = now - self.last.get(topic, 0)
                self.titles[topic].setText("%s  (%s)" %
                    (name, "live" if age < 1.0 else "stale %.0fs" % age))
                self.labels[topic].setPixmap(
                    to_pixmap(self.frames[topic]).scaled(
                        self.labels[topic].size(), QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.FastTransformation))
            else:
                self.titles[topic].setText("%s  (no data — is the node running?)" % name)

    def closeEvent(self, e):
        self.node.destroy_node()
        rclpy.shutdown()
        e.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    v = Viewer()
    v.resize(1280, 520)
    v.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
