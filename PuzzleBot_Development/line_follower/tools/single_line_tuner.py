#!/usr/bin/env python3
"""Single-black-line offline tuner.

Replays a rosbag (or an MP4) frame by frame, runs a simplified single-line
detection pipeline, and lets you tune every parameter with OpenCV
trackbars in real time. The runtime C++ node `follow_single_line` mirrors
this pipeline exactly — when the detection looks right on a representative
range of frames, press 's' to dump the parameters ready to feed to
`ros2 run line_follower follow_single_line --ros-args -p ...`.

Pipeline (mirrors follow_single_line.cpp):
    BGR → downscale → ROI crop → grayscale → gaussian blur
        → (optional median) → THRESH_BINARY_INV (black_threshold)
        → MORPH_OPEN (small) → MORPH_CLOSE (tall-thin)
        → connectedComponents → pick the single largest valid blob
        → centroid x → error vs ROI center → P/PD ω

Keys:
  space         play / pause
  n / l         next frame
  p / h         previous frame
  r             rewind to frame 0
  s             print + save tuned params to tuned_single_params.yaml
  q             quit

Usage:
  # Inside the dev container (rosbag2_py needs ROS sourced):
  python3 single_line_tuner.py path/to/rosbag_dir

  # Or against an extracted mp4 (no ROS needed):
  python3 single_line_tuner.py path/to/video.mp4
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame loading — lazy decode so long bags don't blow out RAM.
# ---------------------------------------------------------------------------

class FrameSource:
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
# Detector — mirrors src/follow_single_line.cpp
# ---------------------------------------------------------------------------

def _odd(n, lo=1):
    n = max(lo, int(n))
    return n if n % 2 == 1 else n + 1


def detect(bgr, p):
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
    roi_w = x1 - x0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Gaussian first (smooths) then optional median (kills salt-and-pepper).
    gk = _odd(p['gaussian_ksize'], lo=1)
    if gk >= 3:
        gray = cv2.GaussianBlur(gray, (gk, gk), 0)
    mk = _odd(p['median_ksize'], lo=1)
    if mk >= 3:
        gray = cv2.medianBlur(gray, mk)

    _, mask = cv2.threshold(
        gray, p['black_threshold'], 255, cv2.THRESH_BINARY_INV)

    ok = max(1, int(p['open_kernel']))
    ck_h = max(1, int(p['close_kernel_h']))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, ck_h))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)

    max_width = p['max_line_width_frac'] * roi_w
    min_height = p['min_line_height_frac'] * roi_h
    min_area = p['min_line_area']

    # Walk every component. Tag each as kept/rejected and remember WHY,
    # so the overlay can show what filter killed a blob you can clearly
    # see in the mask. Reasons are evaluated independently (not short-
    # circuited) so the HUD can show e.g. "area_ok width_BAD height_ok".
    cands = []
    n_reject_area = 0
    n_reject_width = 0
    n_reject_height = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])

        bad_area = area < min_area
        bad_width = w > max_width
        bad_height = h < min_height
        if bad_area: n_reject_area += 1
        if bad_width: n_reject_width += 1
        if bad_height: n_reject_height += 1
        cands.append(dict(
            cx=float(cents[i, 0]) + x0,
            cy=float(cents[i, 1]),
            area=area, w=w, h=h,
            bbox=(bx + x0, by + y0, w, h),
            bad_area=bad_area,
            bad_width=bad_width,
            bad_height=bad_height,
            ok=(not bad_area and not bad_width and not bad_height),
        ))

    best = None
    for c in cands:
        if not c['ok']:
            continue
        if best is None or c['area'] > best['area']:
            best = c

    target_x = best['cx'] if best is not None else None
    roi_center_x = (x0 + x1) / 2.0

    # --- viz ---------------------------------------------------------------
    overlay = scaled.copy()
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (50, 200, 50), 1)
    cv2.line(overlay, (int(roi_center_x), y0),
             (int(roi_center_x), y1 - 1), (120, 120, 120), 1)

    # Draw every candidate. Rejected blobs are painted in a magenta-ish
    # box and annotated with the failing filter, so when you can SEE a
    # curve in the mask but the red target line is missing you can tell
    # at a glance whether it's area/width/height that's killing it.
    for c in cands:
        bx, by, bw, bh = c['bbox']
        if c['ok']:
            color = (0, 200, 255)         # yellow — viable
            thickness = 1
        elif c['bad_width']:
            color = (255, 80, 80)         # blue-ish — too wide (curve case)
            thickness = 2
        elif c['bad_height']:
            color = (80, 80, 255)         # red-ish — too short
            thickness = 2
        elif c['bad_area']:
            color = (120, 120, 120)       # gray — too small / noise
            thickness = 1
        else:
            color = (200, 0, 200); thickness = 1  # shouldn't happen
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, thickness)
        # Tiny tag on rejected blobs explaining why.
        if not c['ok'] and not c['bad_area']:
            tag = []
            if c['bad_width']:  tag.append(f"W{c['w']}>{int(max_width)}")
            if c['bad_height']: tag.append(f"H{c['h']}<{int(min_height)}")
            label = " ".join(tag)
            cv2.putText(overlay, label, (bx, max(10, by - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, label, (bx, max(10, by - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        color, 1, cv2.LINE_AA)

    if best is not None:
        bx, by, bw, bh = best['bbox']
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh),
                      (0, 255, 0), 2)        # green = winner
        cv2.circle(overlay, (int(best['cx']), int(best['cy']) + y0),
                   5, (0, 255, 0), -1)
    if target_x is not None:
        tx = int(target_x)
        cv2.line(overlay, (tx, y0), (tx, y1 - 1), (0, 0, 255), 2)
        cv2.circle(overlay, (tx, (y0 + y1) // 2), 8, (0, 0, 255), 2)

    mask_canvas = np.zeros_like(scaled)
    mask_canvas[y0:y1, x0:x1] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    side = np.hstack([overlay, mask_canvas])

    if target_x is None:
        err_str = "-"
    else:
        # Normalized to roi half-width so it matches what the C++ node does.
        half = max(1.0, (x1 - x0) / 2.0)
        err_str = f"{(target_x - roi_center_x) / half:+.2f}"
    n_kept = sum(1 for c in cands if c['ok'])
    hud1 = (f"kept={n_kept} rej[a={n_reject_area} w={n_reject_width} "
            f"h={n_reject_height}]  err={err_str}")
    hud2 = (f"black={p['black_threshold']}  area>={min_area}  "
            f"w<={int(max_width)}  h>={int(min_height)}  "
            f"gk={gk} mk={mk} ok={ok} ck={ck_h}")
    for i, line in enumerate((hud1, hud2)):
        y = 24 + i * 22
        cv2.putText(side, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(side, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return side, best, target_x


# ---------------------------------------------------------------------------
# Param save
# ---------------------------------------------------------------------------

def _yaml_val(v):
    if isinstance(v, int):
        return str(v)
    return f"{v:.3f}"


def _ros_param_entries(p):
    return [
        ('detect_scale', p['detect_scale']),
        ('roi_top_frac', p['roi_top_frac']),
        ('roi_bottom_frac', p['roi_bottom_frac']),
        ('roi_left_frac', p['roi_left_frac']),
        ('roi_right_frac', p['roi_right_frac']),
        ('gaussian_ksize', int(p['gaussian_ksize'])),
        ('median_ksize', int(p['median_ksize'])),
        ('black_threshold', int(p['black_threshold'])),
        ('open_kernel', int(p['open_kernel'])),
        ('close_kernel_h', int(p['close_kernel_h'])),
        ('min_line_area', int(p['min_line_area'])),
        ('max_line_width_frac', p['max_line_width_frac']),
        ('min_line_height_frac', p['min_line_height_frac']),
    ]


def save_params(p, out_path='tuned_single_params.yaml'):
    entries = _ros_param_entries(p)
    yaml = (
        "# follow_single_line tuned parameters\n"
        "# Apply at runtime:\n"
        "#   ros2 run line_follower follow_single_line --ros-args \\\n"
        + "".join(f"#     -p {k}:={_yaml_val(v)} \\\n" for k, v in entries)
        + "\n"
        + "\n".join(f"{k}: {_yaml_val(v)}" for k, v in entries)
        + "\n"
    )
    print()
    print("=" * 60)
    print("TUNED PARAMETERS (single line)")
    print("=" * 60)
    print(yaml)
    print("=" * 60)
    with open(out_path, 'w') as f:
        f.write(yaml)
    print(f"saved to {os.path.abspath(out_path)}")


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

    win = 'single-line tuner | left: overlay | right: mask'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def nop(_): pass

    cv2.createTrackbar('frame',         win, 0, len(frames) - 1, nop)
    cv2.createTrackbar('detect_scale%', win, 50,  100, nop)
    cv2.createTrackbar('roi_top%',      win, 55,   99, nop)
    cv2.createTrackbar('roi_bot%',      win, 95,  100, nop)
    cv2.createTrackbar('roi_left%',     win, 0,   100, nop)
    cv2.createTrackbar('roi_right%',    win, 100, 100, nop)
    cv2.createTrackbar('gauss_k',       win, 5,    21, nop)
    cv2.createTrackbar('median_k',      win, 0,    21, nop)
    cv2.createTrackbar('black_thr',     win, 80,  255, nop)
    cv2.createTrackbar('open_k',        win, 3,    15, nop)
    cv2.createTrackbar('close_h',       win, 11,   60, nop)
    cv2.createTrackbar('min_area',      win, 200, 5000, nop)
    cv2.createTrackbar('max_w%',        win, 60,  100, nop)
    cv2.createTrackbar('min_h%',        win, 20,  100, nop)

    playing = False
    last_play_t = 0.0
    play_fps = 10.0

    while True:
        idx = cv2.getTrackbarPos('frame', win)
        idx = max(0, min(idx, len(frames) - 1))

        params = dict(
            detect_scale         = cv2.getTrackbarPos('detect_scale%', win) / 100.0,
            roi_top_frac         = cv2.getTrackbarPos('roi_top%', win) / 100.0,
            roi_bottom_frac      = cv2.getTrackbarPos('roi_bot%', win) / 100.0,
            roi_left_frac        = cv2.getTrackbarPos('roi_left%', win) / 100.0,
            roi_right_frac       = cv2.getTrackbarPos('roi_right%', win) / 100.0,
            gaussian_ksize       = cv2.getTrackbarPos('gauss_k', win),
            median_ksize         = cv2.getTrackbarPos('median_k', win),
            black_threshold      = cv2.getTrackbarPos('black_thr', win),
            open_kernel          = max(1, cv2.getTrackbarPos('open_k', win)),
            close_kernel_h       = max(1, cv2.getTrackbarPos('close_h', win)),
            min_line_area        = cv2.getTrackbarPos('min_area', win),
            max_line_width_frac  = cv2.getTrackbarPos('max_w%', win) / 100.0,
            min_line_height_frac = cv2.getTrackbarPos('min_h%', win) / 100.0,
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
