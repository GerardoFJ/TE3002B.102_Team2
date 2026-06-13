#!/usr/bin/env python3
"""LIVE tuner for the adaptive black-line pipeline on real camera frames.

Runs the divide-normalization + adaptive-threshold + contour pipeline on
/camera/image_rect/compressed with EVERY stage parameter on a trackbar,
so you can calibrate while moving the robot / lighting around.

Extra robustness filters (not in the original line_pipeline.py):
  * max_gray      — blob median gray (original image) must be BELOW this.
                    Real tape is near-black; shadow edges / mat seams are
                    mid-gray and get rejected.
  * dark_vs_bg %  — blob median must be < pct% of the median of a ring
                    around it (locally darker than its surroundings).
                    100 = disabled.

Accepted lines are green (fitted curve magenta), rejected blobs red with
the reject reason. Trackbar labels don't render in the docker GUI, so the
numbered parameter panel is drawn onto the image itself.

GUI mode (on your computer, in the ros2 docker, talks to the robot over DDS):
  docker run --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    --user $(id -u):$(id -g) --net=host \
    -v /home/gerardo/Dockers/Ros2/Workspace:/workspace ros2-generic:latest \
    bash -lc 'source /opt/ros/humble/setup.bash && \
      export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
      python3 /workspace/TE3002B.102_Team2/PuzzleBot_Development/line_follower/tools/live_pipeline_test.py'

Headless mode (no X; processes N frames with DEFAULTS, prints stats):
  ... live_pipeline_test.py --headless 60 --save-dir /tmp/pipeline_snaps

Keys (GUI):
  f         freeze / unfreeze the current frame
  d         toggle stage-montage view (gray/normalized/binary/clean)
  s         print params + save tuned_pipeline_params.yaml next to this script
  q / ESC   quit (also saves)
"""
import argparse
import os
import time

import numpy as np
import cv2

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

DEFAULTS = {
    "scale": 0.35,          # downscale before everything (speed + makes
                            # tape width fit inside block_size)
    "roi_top_frac": 0.40,   # crop the room/benches above the track
    "roi_bottom_frac": 1.00,
    "roi_left_frac": 0.00,
    "roi_right_frac": 1.00,
    "median_ksize": 5,
    "bg_sigma": 31,         # background blur for divide-normalization
    "block_size": 41,       # adaptive threshold neighborhood; must stay
                            # bigger than the tape width AFTER downscale
    "thresh_C": 12,         # how much darker than neighborhood = "line"
    "kernel": 3,            # morphology kernel size
    "open_iter": 1,
    "close_iter": 2,
    "min_area_pm": 2,       # blob area, per-mille of ROI area
                            # (2 also rejects the thin jigsaw seams)
    "thinness_x10": 40,     # length/width * 10  (40 -> 4.0)
    "max_gray": 90,         # NEW: blob median gray must be below this.
                            # Track tape reads 69-80, jigsaw seams ~101,
                            # floor 112+ -> 90 splits them cleanly.
    "dark_vs_bg_pct": 65,   # NEW: blob median < pct% of surrounding ring
                            # median. 100 = disabled.
}
PARAM_ORDER = list(DEFAULTS.keys())
WIN = "pipeline tuner"


def odd_at_least(n, lo):
    n = max(lo, int(n))
    return n if n % 2 == 1 else n + 1


def _resample(pts, m):
    """Resample a polyline to m points uniformly spaced by arc length."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] <= 0:
        return None
    t /= t[-1]
    ts = np.linspace(0.0, 1.0, m)
    return np.stack([np.interp(ts, t, pts[:, 0]),
                     np.interp(ts, t, pts[:, 1])], axis=1)


def stroke_centerline(cnt, width, n_out=60):
    """Midline of a thin stroke of ANY shape (works through 90-deg bends).

    A non-functional fit like x=f(y) cuts corners on strokes that bend past
    horizontal. Instead, work on the contour itself:
      1. find the stroke's two END CAPS = contour regions where the walking
         direction reverses (~180 deg turn within ~ one stroke width)
      2. split the closed contour at the caps into its two long sides
      3. resample both sides to the same point count and average them
         pairwise -> true centerline, then smooth with a parametric cubic
         x(t), y(t) over arc length.
    Returns (n_out, 2) float32 points in cnt coords, or None if degenerate.
    """
    c = cnt[:, 0, :].astype(np.float32)
    n = len(c)
    if n < 24:
        return None
    s = int(np.clip(round(width), 3, n // 6))
    fwd = np.roll(c, -s, axis=0) - c          # direction ahead
    bwd = c - np.roll(c, s, axis=0)           # direction behind
    norm = (np.linalg.norm(fwd, axis=1) * np.linalg.norm(bwd, axis=1)) + 1e-6
    turn = (fwd * bwd).sum(axis=1) / norm     # cos(direction change)

    # the two strongest reversal points = the two end caps
    i1 = int(np.argmin(turn))
    # mask out a neighborhood of i1 (half the contour can't be the same cap)
    guard = n // 4
    idx = (np.arange(n) - i1) % n
    masked = np.where((idx > guard) & (idx < n - guard), turn, np.inf)
    i2 = int(np.argmin(masked))
    if turn[i1] > 0.3 or turn[i2] > 0.3:
        return None                            # no clear caps (blob-ish)
    a, b = min(i1, i2), max(i1, i2)
    side1 = c[a:b + 1]
    side2 = np.concatenate([c[b:], c[:a + 1]])[::-1]  # same direction
    if len(side1) < 4 or len(side2) < 4:
        return None
    m = 80
    r1, r2 = _resample(side1, m), _resample(side2, m)
    if r1 is None or r2 is None:
        return None
    mids = (r1 + r2) / 2.0

    # light moving-average smoothing; a global polynomial would cut the
    # corner on sharp bends, the raw midpoints follow it exactly
    k = 7
    pad = np.concatenate([mids[:1].repeat(k // 2, 0), mids,
                          mids[-1:].repeat(k // 2, 0)])
    kern = np.ones(k) / k
    sm = np.stack([np.convolve(pad[:, 0], kern, mode="valid"),
                   np.convolve(pad[:, 1], kern, mode="valid")], axis=1)
    out = _resample(sm, n_out)
    return None if out is None else out.astype(np.float32)


# ----------------------------------------------------------------------
# pipeline (parameterized version of line_pipeline.detect_black_lines)
# ----------------------------------------------------------------------

def detect_black_lines(bgr_full, p, debug=False):
    """Returns (lines, rejected, stages, roi_box).

    lines    : accepted [{contour, center, width, coeffs, vertical}] in
               downscaled-frame coords
    rejected : [(contour, reason)] in downscaled-frame coords
    roi_box  : (x0, y0, x1, y1) in downscaled-frame coords
    """
    if p["scale"] < 0.999:
        bgr = cv2.resize(bgr_full, None, fx=p["scale"], fy=p["scale"],
                         interpolation=cv2.INTER_AREA)
    else:
        bgr = bgr_full
    H, W = bgr.shape[:2]
    x0 = int(round(p["roi_left_frac"] * W))
    x1 = max(x0 + 8, int(round(p["roi_right_frac"] * W)))
    y0 = int(round(p["roi_top_frac"] * H))
    y1 = max(y0 + 8, int(round(p["roi_bottom_frac"] * H)))
    x0, x1 = np.clip([x0, x1], 0, W).tolist()
    y0, y1 = np.clip([y0, y1], 0, H).tolist()
    roi = bgr[y0:y1, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mk = p["median_ksize"]
    denoised = cv2.medianBlur(gray, odd_at_least(mk, 3)) if mk > 0 else gray

    # illumination normalization: divide by heavily blurred copy
    background = cv2.GaussianBlur(denoised, (0, 0),
                                  sigmaX=max(3, p["bg_sigma"]))
    normalized = cv2.divide(denoised, background, scale=255)

    binary = cv2.adaptiveThreshold(
        normalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=odd_at_least(p["block_size"], 3),
        C=max(1, p["thresh_C"]),
    )

    k = max(1, p["kernel"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    clean = binary
    if p["open_iter"] > 0:
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel,
                                 iterations=p["open_iter"])
    if p["close_iter"] > 0:
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel,
                                 iterations=p["close_iter"])

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    roi_area = gray.shape[0] * gray.shape[1]
    min_area = roi_area * p["min_area_pm"] / 1000.0
    min_thinness = p["thinness_x10"] / 10.0

    def to_frame(cnt):
        return cnt + np.array([x0, y0])

    lines, rejected = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(min_area, 8):
            continue  # specks: not even drawn, too many of them
        perimeter = cv2.arcLength(cnt, closed=True)
        length = perimeter / 2.0
        width = area / max(length, 1e-3)
        thinness = length / max(width, 1e-3)
        if thinness < min_thinness:
            rejected.append((to_frame(cnt), "fat %.1f" % thinness))
            continue

        # blob pixels + surrounding ring, on the ORIGINAL gray (pre-norm)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        m = 9
        ex0, ey0 = max(0, bx - m), max(0, by - m)
        ex1 = min(gray.shape[1], bx + bw + m)
        ey1 = min(gray.shape[0], by + bh + m)
        sub = np.zeros((ey1 - ey0, ex1 - ex0), np.uint8)
        cv2.drawContours(sub, [cnt], -1, 255, -1, offset=(-ex0, -ey0))
        blob_px = denoised[ey0:ey1, ex0:ex1][sub > 0]
        if blob_px.size < 8:
            continue
        blob_med = float(np.median(blob_px))
        if blob_med > p["max_gray"]:
            rejected.append((to_frame(cnt), "bright %d" % int(blob_med)))
            continue
        if p["dark_vs_bg_pct"] < 100:
            ring = cv2.dilate(sub, np.ones((7, 7), np.uint8)) & ~sub
            ring_px = denoised[ey0:ey1, ex0:ex1][ring > 0]
            if ring_px.size >= 8:
                ring_med = float(np.median(ring_px))
                if blob_med > ring_med * p["dark_vs_bg_pct"] / 100.0:
                    rejected.append(
                        (to_frame(cnt),
                         "notdark %d/%d" % (int(blob_med), int(ring_med))))
                    continue

        fit_pts = stroke_centerline(cnt, width)
        if fit_pts is not None:
            fit_pts = fit_pts + np.array([x0, y0], np.float32)

        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx, cy = M["m10"] / M["m00"] + x0, M["m01"] / M["m00"] + y0
        lines.append({
            "contour": to_frame(cnt),
            "center": (cx, cy),
            "width": width,
            "gray": blob_med,
            "fit_pts": fit_pts,
        })

    lines.sort(key=lambda l: l["center"][0])
    stages = {}
    if debug:
        stages = {"gray": denoised, "normalized": normalized,
                  "binary": binary, "clean": clean}
    return lines, rejected, stages, (x0, y0, x1, y1, bgr)


def draw_result(bgr_small, lines, rejected, roi_box, ms=None, extra=""):
    out = bgr_small.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = roi_box
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1)
    for cnt, reason in rejected:
        cv2.drawContours(out, [cnt], -1, (0, 0, 255), 1)
        bx, by = cnt[:, 0, 0].min(), cnt[:, 0, 1].min()
        cv2.putText(out, reason, (bx, max(12, by - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    for i, line in enumerate(lines):
        cv2.drawContours(out, [line["contour"]], -1, (0, 255, 0), 2)
        if line["fit_pts"] is not None:
            pts = line["fit_pts"].astype(np.int32)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)
                      & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
            if len(pts) > 1:
                cv2.polylines(out, [pts], False, (255, 0, 255), 2)
        cx, cy = map(int, line["center"])
        cv2.putText(out, "L%d w%.0f g%d" % (i + 1, line["width"],
                                            int(line["gray"])),
                    (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1)
    status = "lines: %d  rej: %d" % (len(lines), len(rejected))
    if ms is not None:
        status += "  %.0f ms" % ms
    if extra:
        status += "  " + extra
    color = (0, 200, 0) if len(lines) == 3 else (0, 0, 255)
    cv2.putText(out, status, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return out


def stage_montage(stages, det, cell_w=480):
    cells = []
    for label, img in list(stages.items()) + [("detection", det)]:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        s = cell_w / img.shape[1]
        img = cv2.resize(img, (cell_w, int(img.shape[0] * s)))
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 22), (40, 40, 40), -1)
        cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cells.append(img)
    h = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT) for c in cells]
    rows = [np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1],
                               cv2.BORDER_CONSTANT) for r in rows]
    return np.vstack(rows)


# ----------------------------------------------------------------------
# trackbars (labels drawn on the image: docker GUI hides bar names)
# ----------------------------------------------------------------------

def make_trackbars():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1500, 950)
    d = DEFAULTS
    nop = lambda v: None
    cv2.createTrackbar("1 scale %", WIN, int(d["scale"] * 100), 100, nop)
    cv2.createTrackbar("2 roi_top %", WIN, int(d["roi_top_frac"] * 100), 99, nop)
    cv2.createTrackbar("3 roi_bot %", WIN, int(d["roi_bottom_frac"] * 100), 100, nop)
    cv2.createTrackbar("4 roi_left %", WIN, int(d["roi_left_frac"] * 100), 99, nop)
    cv2.createTrackbar("5 roi_right %", WIN, int(d["roi_right_frac"] * 100), 100, nop)
    cv2.createTrackbar("6 median_k", WIN, d["median_ksize"], 15, nop)
    cv2.createTrackbar("7 bg_sigma", WIN, d["bg_sigma"], 80, nop)
    cv2.createTrackbar("8 block_size", WIN, d["block_size"], 151, nop)
    cv2.createTrackbar("9 thresh_C", WIN, d["thresh_C"], 40, nop)
    cv2.createTrackbar("10 kernel", WIN, d["kernel"], 9, nop)
    cv2.createTrackbar("11 open_it", WIN, d["open_iter"], 4, nop)
    cv2.createTrackbar("12 close_it", WIN, d["close_iter"], 4, nop)
    cv2.createTrackbar("13 min_area", WIN, d["min_area_pm"], 20, nop)
    cv2.createTrackbar("14 thin_x10", WIN, d["thinness_x10"], 150, nop)
    cv2.createTrackbar("15 max_gray", WIN, d["max_gray"], 255, nop)
    cv2.createTrackbar("16 darkbg %", WIN, d["dark_vs_bg_pct"], 100, nop)


def read_params():
    g = lambda n: cv2.getTrackbarPos(n, WIN)
    return {
        "scale": max(10, g("1 scale %")) / 100.0,
        "roi_top_frac": g("2 roi_top %") / 100.0,
        "roi_bottom_frac": max(g("2 roi_top %") + 1, g("3 roi_bot %")) / 100.0,
        "roi_left_frac": g("4 roi_left %") / 100.0,
        "roi_right_frac": max(g("4 roi_left %") + 1, g("5 roi_right %")) / 100.0,
        "median_ksize": g("6 median_k"),
        "bg_sigma": max(3, g("7 bg_sigma")),
        "block_size": max(3, g("8 block_size")),
        "thresh_C": max(1, g("9 thresh_C")),
        "kernel": max(1, g("10 kernel")),
        "open_iter": g("11 open_it"),
        "close_iter": g("12 close_it"),
        "min_area_pm": g("13 min_area"),
        "thinness_x10": max(5, g("14 thin_x10")),
        "max_gray": g("15 max_gray"),
        "dark_vs_bg_pct": max(10, g("16 darkbg %")),
    }


PANEL_LABELS = [
    ("1 scale %", "scale", "%.2f"),
    ("2 roi_top %", "roi_top_frac", "%.2f"),
    ("3 roi_bot %", "roi_bottom_frac", "%.2f"),
    ("4 roi_left %", "roi_left_frac", "%.2f"),
    ("5 roi_right %", "roi_right_frac", "%.2f"),
    ("6 median_k", "median_ksize", "%d"),
    ("7 bg_sigma", "bg_sigma", "%d"),
    ("8 block_size", "block_size", "%d"),
    ("9 thresh_C", "thresh_C", "%d"),
    ("10 kernel", "kernel", "%d"),
    ("11 open_it", "open_iter", "%d"),
    ("12 close_it", "close_iter", "%d"),
    ("13 min_area (permille)", "min_area_pm", "%d"),
    ("14 thinness x10", "thinness_x10", "%d"),
    ("15 max_gray (abs dark)", "max_gray", "%d"),
    ("16 dark_vs_bg % (100=off)", "dark_vs_bg_pct", "%d"),
]


def params_panel(height, p):
    panel = np.full((height, 340, 3), 28, np.uint8)
    cv2.putText(panel, "PARAMS (sliders above)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    y = 52
    for label, key, fmt in PANEL_LABELS:
        cv2.putText(panel, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(panel, fmt % p[key], (255, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y += 24
    y += 8
    for txt in ("f freeze   d stages", "s save     q quit",
                "green=line magenta=fit", "red=rejected (reason)",
                "block_size must be >", "  tape width after scale"):
        cv2.putText(panel, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        y += 20
    return panel


def save_params(p, path):
    with open(path, "w") as f:
        f.write("# tuned adaptive-pipeline params (live_pipeline_test.py)\n")
        for k in PARAM_ORDER:
            f.write("%s: %s\n" % (k, p[k]))
    print("saved", path)
    for k in PARAM_ORDER:
        print("  %s: %s" % (k, p[k]))


# ----------------------------------------------------------------------
# ROS plumbing
# ----------------------------------------------------------------------

class FrameGrabber:
    def __init__(self, node, topic):
        self.frame = None
        self.stamp = 0.0
        self.n_rx = 0
        node.create_subscription(CompressedImage, topic, self._cb,
                                 qos_profile_sensor_data)

    def _cb(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        self.frame = bgr
        self.stamp = time.monotonic()
        self.n_rx += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", default="/camera/image_rect/compressed")
    ap.add_argument("--scale", type=float, default=None,
                    help="override DEFAULTS scale (headless mode)")
    ap.add_argument("--headless", type=int, metavar="N", default=0,
                    help="no GUI: process N frames with DEFAULTS, save snaps")
    ap.add_argument("--save-dir", default="/tmp/pipeline_snaps")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("live_pipeline_test")
    grab = FrameGrabber(node, args.topic)
    os.makedirs(args.save_dir, exist_ok=True)
    print("waiting for frames on %s ..." % args.topic)

    if args.headless:
        p = dict(DEFAULTS)
        if args.scale is not None:
            p["scale"] = args.scale
        counts, times = [], []
        last_seen = None
        t_start = time.monotonic()
        saved = 0
        small_shape = None
        while len(counts) < args.headless:
            rclpy.spin_once(node, timeout_sec=0.5)
            if grab.frame is None or grab.stamp == last_seen:
                if time.monotonic() - t_start > 20 and grab.n_rx == 0:
                    print("NO FRAMES after 20 s — is the camera up "
                          "and DDS reachable?")
                    rclpy.shutdown()
                    return
                continue
            last_seen = grab.stamp
            t0 = time.perf_counter()
            lines, rejected, stages, (x0, y0, x1, y1, small) = \
                detect_black_lines(grab.frame, p, debug=True)
            ms = (time.perf_counter() - t0) * 1000.0
            counts.append(len(lines))
            times.append(ms)
            small_shape = small.shape
            if len(counts) == 1 or len(counts) % 20 == 0:
                det = draw_result(small, lines, rejected, (x0, y0, x1, y1), ms)
                mont = stage_montage(stages, det)
                path = os.path.join(args.save_dir,
                                    "frame%03d_n%d.png" % (len(counts),
                                                           len(lines)))
                cv2.imwrite(path, mont)
                saved += 1
            if len(counts) % 10 == 0:
                print("  %d/%d frames, last count=%d (rej %d), %.0f ms"
                      % (len(counts), args.headless, len(lines),
                         len(rejected), ms))
        hist = {}
        for c in counts:
            hist[c] = hist.get(c, 0) + 1
        print("\nframes: %d  (processed %dx%d, scale %.2f)"
              % (len(counts), small_shape[1], small_shape[0], p["scale"]))
        print("line-count histogram: %s"
              % ", ".join("%d lines x%d" % (k, v)
                          for k, v in sorted(hist.items())))
        print("pipeline time: mean %.0f ms  max %.0f ms  (~%.1f FPS)"
              % (np.mean(times), np.max(times), 1000.0 / np.mean(times)))
        print("snapshots (%d) saved to %s" % (saved, args.save_dir))
        rclpy.shutdown()
        return

    # ---- GUI mode -------------------------------------------------------
    make_trackbars()
    frozen = None
    show_stages = False
    snap_i = 0
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "tuned_pipeline_params.yaml")
    p = dict(DEFAULTS)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.03)
        frame = frozen if frozen is not None else grab.frame
        if frame is None:
            blank = np.zeros((300, 900, 3), np.uint8)
            cv2.putText(blank, "waiting for %s ..." % args.topic, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(WIN, blank)
            if (cv2.waitKey(50) & 0xFF) in (ord("q"), 27):
                break
            continue
        p = read_params()
        t0 = time.perf_counter()
        lines, rejected, stages, (x0, y0, x1, y1, small) = \
            detect_black_lines(frame, p, debug=True)
        ms = (time.perf_counter() - t0) * 1000.0
        det = draw_result(small, lines, rejected, (x0, y0, x1, y1), ms,
                          extra="[FROZEN]" if frozen is not None else "")
        view = stage_montage(stages, det) if show_stages else det
        # upscale small views so they're comfortable to look at
        if view.shape[1] < 900:
            s = 900.0 / view.shape[1]
            view = cv2.resize(view, None, fx=s, fy=s,
                              interpolation=cv2.INTER_NEAREST)
        panel = params_panel(view.shape[0], p)
        cv2.imshow(WIN, np.hstack([view, panel]))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            save_params(p, yaml_path)
            break
        elif k == ord("f"):
            frozen = None if frozen is not None else frame.copy()
        elif k == ord("d"):
            show_stages = not show_stages
        elif k == ord("s"):
            save_params(p, yaml_path)
            path = os.path.join(args.save_dir, "snap%03d.png" % snap_i)
            cv2.imwrite(path, stage_montage(stages, det))
            print("saved", path)
            snap_i += 1
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
