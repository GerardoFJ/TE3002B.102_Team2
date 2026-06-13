#!/usr/bin/env python3
"""Review/fix the auto-labeled segmentation dataset (run on your computer).

For each bird's-eye frame it shows the image with the mask overlaid (green=lane,
red=stop_line). You:
  - DISCARD frames where the lane label is wrong (off-track, threshold failed).
  - MARK the real stop-lines by dragging a horizontal band over them (the lane
    pixels inside the band become stop_line). The unreliable auto stop-lines are
    cleared on load, so you start from lane-only and add the true ones.

Writes:
  labeled/keep.txt        one kept frame name per line (the training set)
  labeled/masks/<n>.png   updated with your stop-line edits

Run (same docker pattern as the other GUIs):
  python3 lane_seg/review.py --labeled lane_seg/labeled

Keys:  Right/Space = next   Left = prev   d = toggle discard   c = clear stop
       drag vertically with the mouse = mark a stop-line band
       s = save now    q = save + quit
Progress is saved on quit (and every 50 frames).
"""
import os
import sys

import numpy as np
import cv2

for _v in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_DEBUG_PLUGINS"):
    os.environ.pop(_v, None)
from PyQt5 import QtCore, QtGui, QtWidgets   # noqa: E402

import argparse


class Review(QtWidgets.QMainWindow):
    def __init__(self, labeled):
        super().__init__()
        self.labeled = labeled
        self.imgdir = os.path.join(labeled, "images")
        self.maskdir = os.path.join(labeled, "masks")
        self.names = sorted(os.path.splitext(f)[0] for f in os.listdir(self.imgdir)
                            if f.endswith(".png"))
        self.i = 0
        self.discard = set()
        self.drag0 = None
        self.scale = 2

        keepf = os.path.join(labeled, "keep.txt")
        if os.path.exists(keepf):
            kept = set(open(keepf).read().split())
            self.discard = set(self.names) - kept

        self.setWindowTitle("lane_seg review")
        self.lbl = QtWidgets.QLabel()
        self.lbl.setAlignment(QtCore.Qt.AlignTop)
        self.lbl.setMouseTracking(True)
        self.lbl.mousePressEvent = self._press
        self.lbl.mouseReleaseEvent = self._release
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("color:#fff; font-family:monospace;")
        c = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(c)
        v.addWidget(self.status)
        v.addWidget(self.lbl, 1)
        self.setCentralWidget(c)
        self.setStyleSheet("background:#222;")
        self.show_frame()

    # ---- data ----
    def _img(self):
        return cv2.imread(os.path.join(self.imgdir, self.names[self.i] + ".png"))

    def _mask(self):
        return cv2.imread(os.path.join(self.maskdir, self.names[self.i] + ".png"), 0)

    def show_frame(self):
        img = self._img()
        mask = self._mask()
        if not hasattr(self, "_loaded") or self._loaded != self.i:
            mask[mask == 2] = 1          # clear unreliable auto stop on first view
            self._curmask = mask
            self._loaded = self.i
        mask = self._curmask
        viz = img.copy()
        viz[mask == 1] = (0, 255, 0)
        viz[mask == 2] = (0, 0, 255)
        viz = cv2.addWeighted(img, 0.45, viz, 0.55, 0)
        if self.names[self.i] in self.discard:
            cv2.rectangle(viz, (0, 0), (viz.shape[1] - 1, viz.shape[0] - 1), (0, 0, 255), 4)
        big = cv2.resize(viz, None, fx=self.scale, fy=self.scale,
                         interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.lbl.setPixmap(QtGui.QPixmap.fromImage(
            QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()))
        kept = len(self.names) - len(self.discard)
        st = "DISCARDED" if self.names[self.i] in self.discard else "keep"
        self.status.setText("[%d/%d] %s  %s   stop-px=%d   (kept %d)" % (
            self.i + 1, len(self.names), self.names[self.i], st,
            int((self._curmask == 2).sum()), kept))

    # ---- stop-line drag ----
    def _press(self, e):
        self.drag0 = int(e.pos().y() / self.scale)

    def _release(self, e):
        if self.drag0 is None:
            return
        r0, r1 = sorted((self.drag0, int(e.pos().y() / self.scale)))
        r0 = max(0, r0); r1 = min(self._curmask.shape[0], r1 + 1)
        if r1 - r0 >= 1:
            band = self._curmask[r0:r1]
            band[band == 1] = 2          # lane pixels in band -> stop
            self._save_mask()
        self.drag0 = None
        self.show_frame()

    def _save_mask(self):
        cv2.imwrite(os.path.join(self.maskdir, self.names[self.i] + ".png"), self._curmask)

    def save(self):
        kept = [n for n in self.names if n not in self.discard]
        with open(os.path.join(self.labeled, "keep.txt"), "w") as f:
            f.write("\n".join(kept) + "\n")
        print("saved keep.txt (%d kept / %d)" % (len(kept), len(self.names)))

    def keyPressEvent(self, e):
        k = e.key()
        if k in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Space):
            self.i = min(len(self.names) - 1, self.i + 1)
        elif k == QtCore.Qt.Key_Left:
            self.i = max(0, self.i - 1)
        elif k == QtCore.Qt.Key_D:
            n = self.names[self.i]
            self.discard.discard(n) if n in self.discard else self.discard.add(n)
        elif k == QtCore.Qt.Key_C:
            self._curmask[self._curmask == 2] = 1
            self._save_mask()
        elif k == QtCore.Qt.Key_S:
            self.save()
        elif k == QtCore.Qt.Key_Q:
            self.save()
            self.close()
            return
        if self.i % 50 == 0:
            self.save()
        self.show_frame()

    def closeEvent(self, e):
        self.save()
        e.accept()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="lane_seg/labeled")
    args = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    w = Review(args.labeled)
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
