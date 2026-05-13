#!/usr/bin/env python3
"""
Write the hand-eye-solved joint origin into frida_description's
gripper.xacro on the orin, with a timestamped backup.

Targets the gripper.xacro joint named "ZED" (parent=gripper, child=zed).
Replaces only its `xyz` and `rpy` attributes; the rest of the file is
untouched.

Usage:
  ./apply_to_urdf.py handeye_samples_result.json \
      --remote orin@192.168.31.10 \
      --remote-path /home/orin/home2/robot_description/frida_description/urdf/Gripper/Custom/gripper.xacro

By default --remote and --remote-path point at the orin paths discovered
in the project memory. Pass --dry-run to preview the change.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

# Run on the orin itself — no SSH. Path is the gripper.xacro that the
# container's /workspace/src bind mount points at (= what colcon builds).
DEFAULT_REMOTE = ""
DEFAULT_REMOTE_PATH = (
    "/home/orin/dev/nav/robot_description/frida_description/urdf/Gripper/Custom/gripper.xacro"
)

# Match a multi-line xacro joint block of the form
#   <joint name="ZED" ...>
#     <origin xyz="..." rpy="..."/>
#     ... other children ...
#   </joint>
# We only swap the xyz and rpy on the origin element inside.
ORIGIN_RE = re.compile(
    r'(<origin\s+)xyz="[^"]*"\s+rpy="[^"]*"\s*(/?>)'
)
RMS_REFUSE_MM = 30.0   # >30 mm = calibration is bad; refuse without --force
ZED_JOINT_RE = re.compile(
    r'(<joint\s+name="ZED"[^>]*>.*?</joint>)',
    flags=re.DOTALL,
)


def patch_xacro(text: str, xyz: list[float], rpy: list[float]) -> str:
    def patch_joint(m: re.Match) -> str:
        block = m.group(1)
        new_attrs = (
            f'xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" '
            f'rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"'
        )
        new_block, n = ORIGIN_RE.subn(r"\g<1>" + new_attrs + r"\2", block, count=1)
        if n != 1:
            raise RuntimeError(
                "Could not find <origin xyz=... rpy=.../> inside the ZED joint."
            )
        return new_block

    new_text, n = ZED_JOINT_RE.subn(patch_joint, text, count=1)
    if n != 1:
        raise RuntimeError('Could not find <joint name="ZED" ...> ... </joint> block.')
    return new_text


def run(cmd: list[str], check: bool = True) -> str:
    res = subprocess.run(cmd, check=check, capture_output=True, text=True)
    if res.stderr:
        print(res.stderr, end="", file=sys.stderr)
    return res.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result", type=Path,
                   help="Path to handeye_samples_result.json from solve_handeye.py.")
    p.add_argument("--remote", default=DEFAULT_REMOTE,
                   help=f"SSH target (default: {DEFAULT_REMOTE}). Empty string means "
                        "the gripper.xacro is local — operate on --remote-path directly.")
    p.add_argument("--remote-path", default=DEFAULT_REMOTE_PATH,
                   help=f"Path to gripper.xacro on the target (default: {DEFAULT_REMOTE_PATH}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the patched file diff but do not write it.")
    p.add_argument("--force", action="store_true",
                   help=f"Apply even if residual RMS > {RMS_REFUSE_MM:.0f} mm. Use only if you know "
                        "you're overriding a bad-looking calibration on purpose.")
    args = p.parse_args()

    res = json.loads(args.result.read_text())
    if "gripper_xacro_joint_origin" not in res:
        if "samples" in res:
            sys.exit(
                f"\n[!] {args.result} looks like a samples file (it has a "
                "'samples' key but no 'gripper_xacro_joint_origin').\n"
                "    You need to run solve_handeye.py first:\n\n"
                f"      python3 solve_handeye.py {args.result}\n\n"
                "    That produces *_result.json — pass THAT to apply_to_urdf.py.\n"
            )
        sys.exit(
            f"\n[!] {args.result} does not contain 'gripper_xacro_joint_origin'. "
            "Did you pass the right file?\n"
            "    Expected: the *_result.json output of solve_handeye.py.\n"
        )
    origin = res["gripper_xacro_joint_origin"]
    xyz, rpy = origin["xyz"], origin["rpy"]
    rms_mm = res.get("residual_rms_mm")

    print(f"[*] New joint origin: xyz={xyz} rpy={rpy}", file=sys.stderr)
    if isinstance(rms_mm, (int, float)):
        print(f"[*] Residual RMS:     {rms_mm:.2f} mm", file=sys.stderr)
    print(f"[*] Target:           {args.remote or '(local)'}:{args.remote_path}",
          file=sys.stderr)

    if (isinstance(rms_mm, (int, float)) and rms_mm > RMS_REFUSE_MM
            and not args.force and not args.dry_run):
        print(f"\n[!] Refusing to apply: RMS {rms_mm:.1f} mm exceeds the "
              f"{RMS_REFUSE_MM:.0f} mm sanity threshold.",
              file=sys.stderr)
        print("    Re-collect with more orientation variety, or pass --force "
              "to override.", file=sys.stderr)
        sys.exit(2)

    if args.remote:
        original = run(["ssh", args.remote, f"cat {shlex.quote(args.remote_path)}"])
    else:
        original = Path(args.remote_path).read_text()

    patched = patch_xacro(original, xyz, rpy)

    # Show a small diff focused on the change.
    print("\n--- before / after (joint origin only) ---", file=sys.stderr)
    for line in original.splitlines():
        if "xyz=" in line and "rpy=" in line and "ZED" not in line:
            # gripper.xacro layout: the origin line follows the <joint name="ZED" ...>
            # opener; the most reliable diff hint is the literal xyz/rpy line.
            pass
    # Easier: just show the matched origin neighbourhoods.
    for label, blob in (("BEFORE", original), ("AFTER ", patched)):
        m = ZED_JOINT_RE.search(blob)
        if m:
            print(f"  {label}:")
            for ln in m.group(1).splitlines():
                if "origin" in ln:
                    print(f"    {ln.strip()}")
                    break

    if args.dry_run:
        print("\n[dry-run] not writing.", file=sys.stderr)
        return

    if args.remote:
        with tempfile.NamedTemporaryFile("w", suffix=".xacro", delete=False) as f:
            f.write(patched)
            tmp = f.name
        backup_cmd = (
            f"cp {shlex.quote(args.remote_path)} "
            f"{shlex.quote(args.remote_path)}.bak.$(date +%Y%m%d-%H%M%S)"
        )
        run(["ssh", args.remote, backup_cmd])
        run(["scp", "-q", tmp, f"{args.remote}:{args.remote_path}"])
        Path(tmp).unlink(missing_ok=True)
    else:
        target = Path(args.remote_path)
        target.with_suffix(target.suffix + ".bak").write_text(original)
        target.write_text(patched)

    print(f"[*] Updated gripper.xacro. Rebuild frida_description and relaunch:",
          file=sys.stderr)
    print(f"      docker exec -it home2-manipulation bash -lc \\\n"
          f"        'cd /workspace && colcon build --packages-select frida_description "
          f"--symlink-install'",
          file=sys.stderr)


if __name__ == "__main__":
    main()
