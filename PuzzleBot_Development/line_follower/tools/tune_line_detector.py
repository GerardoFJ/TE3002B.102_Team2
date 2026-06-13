#!/usr/bin/env python3
"""Interactive offline tuner for the line_follower black-line detector.

Replays a rosbag (or an MP4) frame by frame, runs the same detection
pipeline the runtime C++ node uses, and lets you tune every parameter
with OpenCV trackbars in real time. When the detection looks right on a
representative range of frames, press 's' to dump the parameters in a
form you can paste into your `ros2 run line_follower follow_line
--ros-args -p ...` invocation.

The detector in this script is a 1:1 Python mirror of the C++ code in
src/follow_line.cpp. If the C++ behavior diverges, mirror it here too —
otherwise the tuning won't transfer.

Keys:
  space         play / pause
  n / l         next frame
  p / h         previous frame
  r             rewind to frame 0
  s             print + save tuned params to tuned_params.yaml
  q             quit

Usage:
  # Inside the dev container (rosbag2_py needs ROS sourced):
  python3 tune_line_detector.py path/to/rosbag_dir

  # Or against an extracted mp4 (no ROS needed):
  python3 tune_line_detector.py path/to/video.mp4
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

class FrameSource:
    """Frame index → decoded BGR, on demand (so we don't OOM on long bags)."""
    def __init__(self):
        self._cache_idx = -1
        self._cache_bgr = None

    def __len__(self):
        raise NotImplementedError

    def _decode(self, idx):
        raise NotImplementedError

    def get(self, idx):
        if idx == self._cache_idx and self._cache_bgr is not None:
            return self._cache_bgr
        bgr = self._decode(idx)
        self._cache_idx = idx
        self._cache_bgr = bgr
        return bgr


class BagSource(FrameSource):
    def __init__(self, bag_path,
                 image_topic='/camera/image_rect/compressed'):
        super().__init__()
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CompressedImage
        self._deserialize_message = deserialize_message
        self._CompressedImage = CompressedImage

        print(f'Indexing bag: {bag_path}')
        so = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        co = rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                         output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(so, co)
        # Store raw serialized message blobs — small (~40 KB JPEG inside CDR)
        # and lazy-decoded only when a frame is requested. 6 000 frames ≈
        # 240 MB resident, vs. 17 GB if we decoded everything upfront.
        self._blobs = []
        while reader.has_next():
            topic, raw, _ = reader.read_next()
            if topic != image_topic:
                continue
            self._blobs.append(raw)
            if len(self._blobs) % 500 == 0:
                print(f'  {len(self._blobs)} frames indexed')
        print(f'Indexed {len(self._blobs)} frames from bag')

    def __len__(self):
        return len(self._blobs)

    def _decode(self, idx):
        msg = self._deserialize_message(self._blobs[idx], self._CompressedImage)
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class VideoSource(FrameSource):
    def __init__(self, path):
        super().__init__()
        print(f'Opening video: {path}')
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f'cv2.VideoCapture failed on {path}')
        self._n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Video has {self._n} frames')

    def __len__(self):
        return self._n

    def _decode(self, idx):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        return frame if ok else None


# ---------------------------------------------------------------------------
# Detector — mirrors src/follow_line.cpp
# ---------------------------------------------------------------------------

def detect(bgr, p):
    """Run the detector. p is a dict of parameters in human units (floats)."""
    scale = max(0.1, min(1.0, p['detect_scale']))
    if scale < 0.999:
        scaled = cv2.resize(bgr, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
    else:
        scaled = bgr.copy()
    sh, sw = scaled.shape[:2]

    x0 = max(0, min(sw - 2, int(round(p['roi_left_frac'] * sw))))
    x1 = max(x0 + 1, min(sw, int(round(p['roi_right_frac'] * sw))))
    y0 = max(0, min(sh - 2, int(round(p['roi_top_frac'] * sh))))
    y1 = max(y0 + 1, min(sh, int(round(p['roi_bottom_frac'] * sh))))
    roi = scaled[y0:y1, x0:x1]
    roi_h = y1 - y0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if p['black_threshold'] > 0:
        _, mask = cv2.threshold(
            blurred, p['black_threshold'], 255, cv2.THRESH_BINARY_INV)
    else:
        bs = max(3, int(p['adaptive_block']))
        if bs % 2 == 0:
            bs += 1
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, bs, p['adaptive_c'])

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, max(1, int(p['close_kernel_h']))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)

    max_width = p['max_line_width_frac'] * sw
    min_height = p['min_line_height_frac'] * roi_h
    cands = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p['min_line_area']:
            continue
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w > max_width or h < min_height:
            continue
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        cands.append(dict(
            cx=float(cents[i, 0]) + x0,
            cy=float(cents[i, 1]),
            area=area, w=w, h=h, bbox=(bx + x0, by + y0, w, h),
        ))

    # Top-N by area.
    cands.sort(key=lambda c: c['area'], reverse=True)
    cands = cands[: int(p['top_n_components'])]
    # Sort survivors left-to-right.
    cands.sort(key=lambda c: c['cx'])

    image_center = sw / 2.0
    target_x = None
    if len(cands) >= 3:
        target_x = cands[len(cands) // 2]['cx']
    elif cands:
        target_x = min(cands, key=lambda c: abs(c['cx'] - image_center))['cx']

    # --- viz ----
    # Left panel: overlay on scaled BGR with bboxes + target.
    overlay = scaled.copy()
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (50, 200, 50), 1)
    cv2.line(overlay, (int(image_center), 0),
             (int(image_center), sh - 1), (120, 120, 120), 1)
    for c in cands:
        bx, by, bw, bh = c['bbox']
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh),
                      (0, 200, 255), 1)
        cv2.circle(overlay, (int(c['cx']), int(c['cy']) + y0),
                   4, (0, 200, 255), -1)
    if target_x is not None:
        tx = int(target_x)
        cv2.line(overlay, (tx, y0), (tx, y1 - 1), (0, 0, 255), 2)
        cv2.circle(overlay, (tx, (y0 + y1) // 2), 8, (0, 0, 255), 2)

    # Right panel: binary mask, painted into a same-size canvas.
    mask_canvas = np.zeros_like(scaled)
    mask_canvas[y0:y1, x0:x1] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    side = np.hstack([overlay, mask_canvas])
    err = "-" if target_x is None else f"{(target_x - image_center) / image_center:+.2f}"
    hud = (f"n={len(cands)}  err={err}  "
           f"black={p['black_threshold']}  top_n={p['top_n_components']}")
    cv2.putText(side, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(side, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return side, cands, target_x


# ---------------------------------------------------------------------------
# Param save
# ---------------------------------------------------------------------------

def save_params(p, out_path='tuned_params.yaml'):
    yaml = (
        "# line_follower tuned parameters\n"
        "# Apply at runtime:\n"
        "#   ros2 run line_follower follow_line --ros-args \\\n"
        + "".join(
            f"#     -p {k}:={_yaml_val(v)} \\\n"
            for k, v in _ros_param_entries(p)
        )
        + "\n"
        + "\n".join(f"{k}: {_yaml_val(v)}" for k, v in _ros_param_entries(p))
        + "\n"
    )
    print()
    print("=" * 60)
    print("TUNED PARAMETERS")
    print("=" * 60)
    print(yaml)
    print("=" * 60)
    with open(out_path, 'w') as f:
        f.write(yaml)
    print(f"saved to {os.path.abspath(out_path)}")


def _ros_param_entries(p):
    return [
        ('detect_scale', p['detect_scale']),
        ('black_threshold', int(p['black_threshold'])),
        ('roi_top_frac', p['roi_top_frac']),
        ('roi_bottom_frac', p['roi_bottom_frac']),
        ('roi_left_frac', p['roi_left_frac']),
        ('roi_right_frac', p['roi_right_frac']),
        ('min_line_area', int(p['min_line_area'])),
        ('max_line_width_frac', p['max_line_width_frac']),
        ('min_line_height_frac', p['min_line_height_frac']),
        ('close_kernel_h', int(p['close_kernel_h'])),
        ('top_n_components', int(p['top_n_components'])),
    ]


def _yaml_val(v):
    if isinstance(v, int):
        return str(v)
    return f"{v:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', help='rosbag directory or .mp4 path')
    ap.add_argument('--image-topic', default='/camera/image_rect/compressed')
    args = ap.parse_args()

    if not os.path.exists(args.source):
        sys.exit(f"Not found: {args.source}")

    if args.source.endswith('.mp4') or args.source.endswith('.mkv'):
        frames = VideoSource(args.source)
    else:
        frames = BagSource(args.source, args.image_topic)
    if len(frames) == 0:
        sys.exit("No frames in source.")

    win = 'line_follower tuner | left: overlay | right: mask'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def nop(_): pass

    # All trackbars are integers; we map to floats on read.
    cv2.createTrackbar('frame',         win, 0, len(frames) - 1, nop)
    cv2.createTrackbar('detect_scale%', win, 50,  100, nop)
    cv2.createTrackbar('black_thr',     win, 70,  255, nop)
    cv2.createTrackbar('roi_top%',      win, 55,   99, nop)
    cv2.createTrackbar('roi_bot%',      win, 95,  100, nop)
    cv2.createTrackbar('roi_left%',     win, 0,    50, nop)
    cv2.createTrackbar('roi_right%',    win, 100, 100, nop)
    cv2.createTrackbar('min_area',      win, 50, 2000, nop)
    cv2.createTrackbar('max_w%',        win, 30,  100, nop)
    cv2.createTrackbar('min_h%',        win, 30,  100, nop)
    cv2.createTrackbar('close_h',       win, 11,   60, nop)
    cv2.createTrackbar('top_n',         win, 5,    15, nop)

    playing = False
    last_play_t = 0.0
    play_fps = 10.0

    while True:
        idx = cv2.getTrackbarPos('frame', win)
        idx = max(0, min(idx, len(frames) - 1))

        params = dict(
            detect_scale         = cv2.getTrackbarPos('detect_scale%', win) / 100.0,
            black_threshold      = cv2.getTrackbarPos('black_thr', win),
            adaptive_block       = 41,    # only used if black_threshold = 0
            adaptive_c           = 15.0,
            roi_top_frac         = cv2.getTrackbarPos('roi_top%', win) / 100.0,
            roi_bottom_frac      = cv2.getTrackbarPos('roi_bot%', win) / 100.0,
            roi_left_frac        = cv2.getTrackbarPos('roi_left%', win) / 100.0,
            roi_right_frac       = cv2.getTrackbarPos('roi_right%', win) / 100.0,
            min_line_area        = cv2.getTrackbarPos('min_area', win),
            max_line_width_frac  = cv2.getTrackbarPos('max_w%', win) / 100.0,
            min_line_height_frac = cv2.getTrackbarPos('min_h%', win) / 100.0,
            close_kernel_h       = cv2.getTrackbarPos('close_h', win),
            top_n_components     = max(1, cv2.getTrackbarPos('top_n', win)),
        )
        frame = frames.get(idx)
        if frame is None:
            continue
        viz, _, _ = detect(frame, params)
        cv2.imshow(win, viz)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            playing = not playing
            last_play_t = time.time()
        elif key in (ord('n'), ord('l'), 83, 46):
            cv2.setTrackbarPos('frame', win, min(idx + 1, len(frames) - 1))
        elif key in (ord('p'), ord('h'), 81, 44):
            cv2.setTrackbarPos('frame', win, max(idx - 1, 0))
        elif key == ord('r'):
            cv2.setTrackbarPos('frame', win, 0)
        elif key == ord('s'):
            save_params(params)

        if playing and time.time() - last_play_t > 1.0 / play_fps:
            last_play_t = time.time()
            nxt = idx + 1
            if nxt >= len(frames):
                nxt = 0
            cv2.setTrackbarPos('frame', win, nxt)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
