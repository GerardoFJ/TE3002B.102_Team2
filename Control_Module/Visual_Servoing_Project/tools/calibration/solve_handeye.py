#!/usr/bin/env python3
"""
Solve eye-in-hand calibration from samples produced by collect_samples.py.

Inputs (JSON):
  - 12+ samples of (T_base_gripper, T_cam_target)
  - One-time T_zed_optical (the chain from the gripper.xacro "zed" link
    down to the optical frame in which the tag is detected)

Output:
  - Best-fit T_gripper_camera (xyz, rpy) — what the arm tip really sees
  - Required gripper.xacro joint origin: xyz, rpy for parent=gripper, child=zed
    so that the chain reproduces T_gripper_camera
  - Cross-method comparison and a residual-style sanity check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# OpenCV's calibrateHandEye supports these algorithms; we run all five so the
# user can spot disagreement that signals a bad sample set.
METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def quat_xyzw_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def R_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    # ZYX (roll about X, pitch about Y, yaw about Z) — URDF convention.
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def to_T(xyz: list[float], quat_xyzw: list[float]) -> np.ndarray:
    R = quat_xyzw_to_R(*quat_xyzw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def split(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return T[:3, :3], T[:3, 3].reshape(3, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples", type=Path, help="Path to handeye_samples.json")
    p.add_argument("--method", default="auto", choices=["auto", *METHODS],
                   help="Which method's result to write to JSON (default: auto = "
                        "pick the method with the lowest residual RMS). PARK and "
                        "DANIILIDIS are usually the most robust on small datasets.")
    p.add_argument("--no-trim", action="store_true",
                   help="Disable automatic outlier trimming (keep all samples).")
    p.add_argument("--trim-target-mm", type=float, default=15.0,
                   help="Stop trimming once RMS drops below this (default: 15 mm).")
    p.add_argument("--min-keep", type=int, default=6,
                   help="Don't trim below this sample count (default: 6).")
    args = p.parse_args()

    data = json.loads(args.samples.read_text())
    all_samples = data["samples"]
    if len(all_samples) < 5:
        raise SystemExit(f"Need at least 5 samples; have {len(all_samples)}.")

    print(f"\nLoaded {len(all_samples)} samples from {args.samples}\n")

    def fit(idx: list[int]) -> dict[str, tuple[np.ndarray, float]]:
        R_g2b = [to_T(all_samples[i]["T_base_gripper"]["xyz"],
                      all_samples[i]["T_base_gripper"]["quat_xyzw"])[:3, :3] for i in idx]
        t_g2b = [to_T(all_samples[i]["T_base_gripper"]["xyz"],
                      all_samples[i]["T_base_gripper"]["quat_xyzw"])[:3, 3].reshape(3, 1) for i in idx]
        R_t2c = [to_T(all_samples[i]["T_cam_target"]["xyz"],
                      all_samples[i]["T_cam_target"]["quat_xyzw"])[:3, :3] for i in idx]
        t_t2c = [to_T(all_samples[i]["T_cam_target"]["xyz"],
                      all_samples[i]["T_cam_target"]["quat_xyzw"])[:3, 3].reshape(3, 1) for i in idx]
        out: dict[str, tuple[np.ndarray, float]] = {}
        for name, method in METHODS.items():
            Rcg, tcg = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
            T_gc = np.eye(4)
            T_gc[:3, :3] = Rcg
            T_gc[:3, 3] = tcg.flatten()
            tbase = np.array([
                (to_T(all_samples[i]["T_base_gripper"]["xyz"],
                      all_samples[i]["T_base_gripper"]["quat_xyzw"]) @ T_gc @
                 to_T(all_samples[i]["T_cam_target"]["xyz"],
                      all_samples[i]["T_cam_target"]["quat_xyzw"]))[:3, 3]
                for i in idx])
            rms = float(np.sqrt(np.mean(np.sum((tbase - tbase.mean(0)) ** 2, axis=1))))
            out[name] = (T_gc, rms)
        return out

    def print_table(results: dict[str, tuple[np.ndarray, float]]) -> None:
        print(f"{'method':<12}  {'xyz (m)':<36}  {'rpy (deg)':<32}  {'RMS (mm)':>9}")
        print("-" * 96)
        for name, (T_gc, rms) in results.items():
            xyz = T_gc[:3, 3]
            rpy = np.degrees(R_to_rpy(T_gc[:3, :3]))
            print(f"{name:<12}  "
                  f"[{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}]      "
                  f"[{rpy[0]:+7.2f}, {rpy[1]:+7.2f}, {rpy[2]:+7.2f}]    "
                  f"{rms*1000:>8.1f}")

    # Initial fit on all samples — shows the methods comparison even if we
    # end up trimming.
    print("All samples:")
    results = fit(list(range(len(all_samples))))
    print_table(results)

    # Iteratively drop the single highest-residual sample (per the best
    # method on each iteration) until RMS drops below the target OR we hit
    # the minimum. This is robust to a handful of bad captures (bumped tag,
    # stale tf, motion not finished) without manual inspection.
    kept = list(range(len(all_samples)))
    dropped: list[int] = []
    if not args.no_trim:
        while True:
            chosen_name = min(results, key=lambda k: results[k][1])
            T_gc, rms = results[chosen_name]
            if rms * 1000 < args.trim_target_mm:
                break
            if len(kept) <= args.min_keep:
                break
            # Find worst sample under the chosen method.
            tbase = np.array([
                (to_T(all_samples[i]["T_base_gripper"]["xyz"],
                      all_samples[i]["T_base_gripper"]["quat_xyzw"]) @ T_gc @
                 to_T(all_samples[i]["T_cam_target"]["xyz"],
                      all_samples[i]["T_cam_target"]["quat_xyzw"]))[:3, 3]
                for i in kept])
            errs = np.linalg.norm(tbase - tbase.mean(0), axis=1) * 1000
            worst_local = int(np.argmax(errs))
            worst_global = kept[worst_local]
            dropped.append(worst_global)
            kept.pop(worst_local)
            results = fit(kept)
            chosen_name = min(results, key=lambda k: results[k][1])
            print(f"  ... dropped sample {worst_global} (residual {errs[worst_local]:.1f} mm) "
                  f"-> now {len(kept)} samples, best={chosen_name} "
                  f"RMS={results[chosen_name][1]*1000:.1f} mm")

        if dropped:
            print(f"\nAfter trimming ({len(kept)}/{len(all_samples)} samples kept, "
                  f"dropped {dropped}):")
            print_table(results)

    if args.method == "auto":
        chosen = min(results, key=lambda k: results[k][1])
        print(f"\n[auto] Lowest-residual method: {chosen} "
              f"({results[chosen][1]*1000:.1f} mm)")
    else:
        chosen = args.method

    T_gc, rms = results[chosen]
    # Recompute mean for printing.
    tbase = np.array([
        (to_T(all_samples[i]["T_base_gripper"]["xyz"],
              all_samples[i]["T_base_gripper"]["quat_xyzw"]) @ T_gc @
         to_T(all_samples[i]["T_cam_target"]["xyz"],
              all_samples[i]["T_cam_target"]["quat_xyzw"]))[:3, 3]
        for i in kept])
    mean = tbase.mean(0)
    print()
    print(f"Residual check (method={chosen}, n={len(kept)} samples):")
    print(f"  Tag position in base, mean = "
          f"[{mean[0]:+.4f}, {mean[1]:+.4f}, {mean[2]:+.4f}] m")
    print(f"  RMS spread across samples   = {rms*1000:.2f} mm")
    print("  (<10 mm is solid; >30 mm usually means poor sample diversity, "
          "stale tf in some captures, or a bad detection.)")
    samples = [all_samples[i] for i in kept]  # for the JSON write at the bottom

    # Back-derive the gripper.xacro joint origin (gripper -> zed).
    # Chain: T_gripper_cam = T_gripper_zed * T_zed_cam
    # => T_gripper_zed = T_gripper_cam * inv(T_zed_cam)
    zed_to_cam = data.get("zed_to_camera_optical")
    if zed_to_cam is None:
        print("\n[!] No zed_to_camera_optical in samples JSON — cannot compute "
              "gripper.xacro joint origin.\n"
              "    The T_gripper_cam values above are still useful as a "
              "static_transform_publisher.")
        return

    T_zc = to_T(zed_to_cam["xyz"], zed_to_cam["quat_xyzw"])
    T_gz = T_gc @ inv_T(T_zc)
    xyz = T_gz[:3, 3]
    rpy = R_to_rpy(T_gz[:3, :3])

    print()
    print("=" * 84)
    print("gripper.xacro joint origin (parent=gripper, child=zed):")
    print(f'  <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" '
          f'rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>')
    print("=" * 84)

    out = args.samples.with_name(args.samples.stem + "_result.json")
    out.write_text(json.dumps({
        "method": chosen,
        "T_gripper_camera_optical": {
            "xyz": T_gc[:3, 3].tolist(),
            "R": T_gc[:3, :3].tolist(),
        },
        "gripper_xacro_joint_origin": {
            "xyz": [float(v) for v in xyz],
            "rpy": [float(v) for v in rpy],
        },
        "residual_rms_mm": rms * 1000,
        "n_samples_used": len(samples),
        "n_samples_total": len(all_samples),
        "dropped_samples": dropped,
        "all_methods_rms_mm": {k: v[1] * 1000 for k, v in results.items()},
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
