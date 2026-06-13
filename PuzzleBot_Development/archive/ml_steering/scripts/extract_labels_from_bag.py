#!/usr/bin/env python3
"""Offline label generator (vision oracle) for ml_steering training.

Reads a rosbag containing /camera/image_rect/compressed and (optionally)
/hough_features, runs a *trustworthy-on-clean-frames* geometric
estimator on each image to produce an `omega_true` label, and writes a
CSV of (14 features) + (omega_true) for confident frames only.

The oracle is intentionally strict — it skips frames where it can't
confidently identify the central line (curves, partial occlusions,
weird lighting). The XGBoost model then learns to generalize to the
hard frames the oracle refused to label.

Output CSV columns:
  n_lines, mean_angle, std_angle, mean_abs_angle,
  mean_length, max_length, left_count, right_count, balance_lr,
  center_bottom, center_error, vanishing_x, vanishing_error,
  confidence,
  omega_true, oracle_confidence, frame_ts_ns

Usage:
  python3 extract_labels_from_bag.py <bag_path> [--output dataset.csv] \\
                                     [--detect-scale 0.5] \\
                                     [--min-confidence 0.30] \\
                                     [--max-samples 0]
"""

import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# rosbag2_py is an apt package (ros-humble-rosbag2-py). Make sure ROS env
# is sourced before running this script.
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray


FEATURE_COLS = [
    "n_lines", "mean_angle", "std_angle", "mean_abs_angle",
    "mean_length", "max_length", "left_count", "right_count", "balance_lr",
    "center_bottom", "center_error", "vanishing_x", "vanishing_error",
    "confidence",
]


# ---------------------------------------------------------------------------
# Hough detector + feature extractor — direct Python mirror of the C++
# code in hough_features/src/hough_detector.cpp so the labels are built
# from the SAME features the runtime node will see at inference time.
# (If you change the C++ detector, mirror it here.)
# ---------------------------------------------------------------------------

CANNY1, CANNY2 = 60, 160
HOUGH_THRESHOLD = 35
MIN_LINE_LENGTH = 25
MAX_LINE_GAP = 15
ROI_TOP_FRAC, ROI_BOT_FRAC = 0.40, 1.00
ROI_LEFT_FRAC, ROI_RIGHT_FRAC = 0.00, 1.00


def collapse_to_orientation(a: float) -> float:
    if a > math.pi / 2:
        return a - math.pi
    if a < -math.pi / 2:
        return a + math.pi
    return a


def hough_detect(bgr: np.ndarray, detect_scale: float):
    """Returns (segments, processed_width, processed_height, roi_offset_xy).

    Mirrors hough_features::HoughDetector::detect.
    """
    if detect_scale < 0.999:
        scaled = cv2.resize(bgr, None, fx=detect_scale, fy=detect_scale,
                            interpolation=cv2.INTER_AREA)
    else:
        scaled = bgr
    h, w = scaled.shape[:2]
    x0 = int(round(ROI_LEFT_FRAC * w))
    x1 = int(round(ROI_RIGHT_FRAC * w))
    y0 = int(round(ROI_TOP_FRAC * h))
    y1 = int(round(ROI_BOT_FRAC * h))
    roi = scaled[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )
    if lines is None:
        segs = np.empty((0, 4), dtype=np.float32)
    else:
        segs = lines.reshape(-1, 4).astype(np.float32)
    return segs, x1 - x0, y1 - y0, (x0, y0)


def line_x_at_y(x1, y1, x2, y2, y_query):
    dy = y2 - y1
    if abs(dy) < 1e-6:
        return float("nan")
    t = (y_query - y1) / dy
    return x1 + t * (x2 - x1)


def estimate_center_bottom(segments: np.ndarray, w: int, h: int) -> float:
    if len(segments) == 0:
        return w / 2.0
    bottom_y = h - 1
    xs = []
    for s in segments:
        xb = line_x_at_y(s[0], s[1], s[2], s[3], bottom_y)
        if math.isfinite(xb) and -0.5 * w <= xb <= 1.5 * w:
            xs.append(xb)
    if len(xs) < 2:
        return w / 2.0
    xs = np.asarray(xs)
    half = w / 2.0
    left = xs[xs < half]
    right = xs[xs >= half]
    if left.size > 0 and right.size > 0:
        return float((np.median(left) + np.median(right)) / 2.0)
    return float(np.median(xs))


def estimate_vanishing_x(segments: np.ndarray, w: int, h: int,
                         max_pairs: int = 80) -> float:
    if len(segments) < 2:
        return w / 2.0
    ix_acc = []
    count = 0
    n = len(segments)
    for i in range(n):
        if count >= max_pairs:
            break
        x1, y1, x2, y2 = segments[i]
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1
        for j in range(i + 1, n):
            if count >= max_pairs:
                break
            x3, y3, x4, y4 = segments[j]
            a2 = y4 - y3
            b2 = x3 - x4
            c2 = a2 * x3 + b2 * y3
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                continue
            ix = (b2 * c1 - b1 * c2) / det
            iy = (a1 * c2 - a2 * c1) / det
            if -w <= ix <= 2 * w and -h <= iy <= h:
                ix_acc.append(ix)
                count += 1
    if not ix_acc:
        return w / 2.0
    return float(np.median(ix_acc))


def extract_features(segments: np.ndarray, w: int, h: int) -> dict:
    half = w / 2.0
    if len(segments) == 0:
        return {
            "n_lines": 0.0, "mean_angle": 0.0, "std_angle": 0.0,
            "mean_abs_angle": 0.0, "mean_length": 0.0, "max_length": 0.0,
            "left_count": 0.0, "right_count": 0.0, "balance_lr": 0.0,
            "center_bottom": half, "center_error": 0.0,
            "vanishing_x": half, "vanishing_error": 0.0, "confidence": 0.0,
        }
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    lengths = np.sqrt(dx * dx + dy * dy)
    raw_angles = np.arctan2(dy, dx)
    angles = np.array([collapse_to_orientation(a) for a in raw_angles])
    abs_angles = np.abs(angles)
    midxs = (segments[:, 0] + segments[:, 2]) / 2.0

    mean_angle = float(np.mean(angles))
    std_angle = float(np.std(angles)) if len(angles) > 1 else 0.0
    mean_abs_angle = float(np.mean(abs_angles))
    mean_length = float(np.mean(lengths))
    max_length = float(np.max(lengths))
    left_count = int(np.sum(midxs < half))
    right_count = int(np.sum(midxs >= half))
    balance_lr = (right_count - left_count) / max(1, len(segments))
    center_bottom = estimate_center_bottom(segments, w, h)
    center_error = (center_bottom - half) / half
    vanishing_x = estimate_vanishing_x(segments, w, h)
    vanishing_error = (vanishing_x - half) / half
    confidence = min(1.0, len(segments) / 12.0) * float(
        np.clip(mean_length / 80.0, 0.0, 1.0))
    return {
        "n_lines": float(len(segments)),
        "mean_angle": mean_angle, "std_angle": std_angle,
        "mean_abs_angle": mean_abs_angle, "mean_length": mean_length,
        "max_length": max_length,
        "left_count": float(left_count), "right_count": float(right_count),
        "balance_lr": float(balance_lr),
        "center_bottom": float(center_bottom),
        "center_error": float(center_error),
        "vanishing_x": float(vanishing_x),
        "vanishing_error": float(vanishing_error),
        "confidence": float(confidence),
    }


# ---------------------------------------------------------------------------
# Vision oracle: cluster near-vertical segments into "lines", pick the
# central one, derive omega from its tilt + lateral offset. Outputs
# (omega, confidence). Caller filters on confidence.
# ---------------------------------------------------------------------------

K_TILT = 1.5     # rad / rad — how much omega per rad of line tilt
K_LATERAL = 0.5  # rad/s per normalized-lateral-error unit
OMEGA_CLAMP = 1.5
TILT_MAX = math.pi / 4  # accept lines tilted up to 45° from vertical


def oracle_omega(segments: np.ndarray, w: int, h: int) -> Tuple[float, float]:
    """Returns (omega, confidence in [0, 1]).

    Algorithm:
      1. Keep only long, near-vertical segments.
      2. Greedy-cluster by mid-x (segments within 8% of width = same line).
      3. Pick top-3 clusters by total length, then within those pick the
         one whose mean-x is closest to image center — that's the center
         line.
      4. omega = -K_TILT * tilt(center_line) - K_LATERAL * lateral_error
      5. Confidence rises with cluster count and segment count.
    """
    if len(segments) < 2:
        return 0.0, 0.0

    cands = []  # (mid_x, length, signed_tilt_from_vertical)
    for s in segments:
        x1, y1, x2, y2 = s
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < MIN_LINE_LENGTH:
            continue
        # Direction-agnostic: orient bottom-down so we can compute a signed
        # "lean" from vertical. If dy < 0, flip endpoints.
        if dy < 0:
            dx = -dx
            dy = -dy
        if dy <= 1e-6:
            continue
        tilt = math.atan2(dx, dy)  # 0 = vertical, + = leans right
        if abs(tilt) > TILT_MAX:
            continue
        mid_x = (x1 + x2) / 2.0
        cands.append((mid_x, length, tilt))

    if len(cands) < 2:
        return 0.0, 0.0

    cands.sort(key=lambda c: c[0])

    # Greedy cluster by mid-x.
    bucket = w * 0.08
    clusters: List[List[Tuple[float, float, float]]] = [[cands[0]]]
    for c in cands[1:]:
        if c[0] - clusters[-1][-1][0] < bucket:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    if len(clusters) < 2:
        return 0.0, 0.0

    # Aggregate each cluster: total length, length-weighted mean tilt and mid_x.
    info = []
    for cl in clusters:
        total_len = sum(c[1] for c in cl)
        if total_len <= 0:
            continue
        mean_x = sum(c[0] * c[1] for c in cl) / total_len
        mean_tilt = sum(c[2] * c[1] for c in cl) / total_len
        info.append((mean_x, mean_tilt, total_len, len(cl)))

    # Pick top 3 clusters by total length.
    info.sort(key=lambda c: c[2], reverse=True)
    top = info[:3]
    if len(top) < 2:
        return 0.0, 0.0

    image_center = w / 2.0
    top.sort(key=lambda c: abs(c[0] - image_center))
    center_line = top[0]
    mean_x, mean_tilt, total_len, n_segs = center_line

    lateral_error = (mean_x - image_center) / image_center
    omega = -K_TILT * mean_tilt - K_LATERAL * lateral_error
    omega = max(-OMEGA_CLAMP, min(OMEGA_CLAMP, omega))

    # Confidence:
    #   - segments in the chosen central cluster (more = better)
    #   - bonus if we saw 3+ distinct clusters (full corridor visible)
    seg_conf = min(1.0, n_segs / 4.0)
    visibility = 1.0 if len(clusters) >= 3 else 0.55
    confidence = seg_conf * visibility
    return omega, confidence


# ---------------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------------

def open_bag(path: str):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="Path to the rosbag2 directory")
    ap.add_argument(
        "--output", default="dataset.csv",
        help="Output CSV path (default: dataset.csv next to the bag)",
    )
    ap.add_argument(
        "--detect-scale", type=float, default=0.5,
        help="Image downscale before Hough — MUST match the runtime "
             "hough_feature_node detect_scale parameter.",
    )
    ap.add_argument(
        "--min-confidence", type=float, default=0.30,
        help="Drop samples whose oracle confidence is below this.",
    )
    ap.add_argument(
        "--frame-skip", type=int, default=1,
        help="Process every Nth image (default 1 = all).",
    )
    ap.add_argument(
        "--max-samples", type=int, default=0,
        help="Cap on samples written (0 = unlimited).",
    )
    ap.add_argument(
        "--image-topic", default="/camera/image_rect/compressed",
    )
    args = ap.parse_args()

    if not os.path.exists(args.bag):
        sys.exit(f"Bag path not found: {args.bag}")

    print(f"Opening bag: {args.bag}")
    reader = open_bag(args.bag)
    topic_types = reader.get_all_topics_and_types()
    print(f"  topics: {[(t.name, t.type) for t in topic_types]}")

    rows = []
    n_images_seen = 0
    n_kept = 0
    n_low_conf = 0
    n_no_lines = 0

    type_by_name = {t.name: t.type for t in topic_types}
    if args.image_topic not in type_by_name:
        sys.exit(f"Bag has no topic {args.image_topic}")

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if topic != args.image_topic:
            continue
        n_images_seen += 1
        if (n_images_seen - 1) % args.frame_skip != 0:
            continue

        msg = deserialize_message(raw, CompressedImage)
        np_buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        segs, pw, ph, _ = hough_detect(bgr, args.detect_scale)
        if len(segs) == 0:
            n_no_lines += 1
            continue

        features = extract_features(segs, pw, ph)
        omega, conf = oracle_omega(segs, pw, ph)
        if conf < args.min_confidence:
            n_low_conf += 1
            continue

        row = dict(features)
        row["omega_true"] = omega
        row["oracle_confidence"] = conf
        row["frame_ts_ns"] = t_ns
        rows.append(row)
        n_kept += 1

        if n_kept % 200 == 0:
            print(
                f"  {n_kept:5d} kept / {n_images_seen:5d} seen "
                f"(skip={n_no_lines} no-lines, {n_low_conf} low-conf)"
            )
        if args.max_samples and n_kept >= args.max_samples:
            print("  hit --max-samples cap, stopping")
            break

    print()
    print(f"images seen   : {n_images_seen}")
    print(f"kept          : {n_kept}")
    print(f"  no-lines    : {n_no_lines}")
    print(f"  low-conf    : {n_low_conf}")
    if n_kept == 0:
        sys.exit("No samples kept — relax --min-confidence or check camera framing.")

    df = pd.DataFrame(rows)
    df = df[FEATURE_COLS + ["omega_true", "oracle_confidence", "frame_ts_ns"]]
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} samples to {args.output}")
    print(f"omega_true: range=[{df.omega_true.min():.3f}, {df.omega_true.max():.3f}] "
          f"mean={df.omega_true.mean():+.3f} std={df.omega_true.std():.3f}")
    print(f"oracle_confidence: mean={df.oracle_confidence.mean():.3f}")
    # Quick label distribution histogram
    bins = np.linspace(df.omega_true.min(), df.omega_true.max(), 9)
    counts, edges = np.histogram(df.omega_true, bins=bins)
    print("\nomega_true distribution:")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(c / max(1, len(df)) * 40)
        pct = 100 * c / len(df)
        print(f"  [{lo:+.3f}, {hi:+.3f}]  {bar:40s}  {c:4d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
