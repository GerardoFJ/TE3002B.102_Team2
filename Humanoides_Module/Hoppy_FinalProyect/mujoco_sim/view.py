#!/usr/bin/env python3
"""Interactive viewer for the HOPPY MuJoCo model.

    python view.py            # passive viewer, no control (drag joints with ctrl+click)
    python view.py --drop     # let it fall under gravity and settle on the foot

Requires a display (you have DISPLAY=:1). Run from this folder so meshdir resolves.
"""
import argparse, time
import mujoco
import mujoco.viewer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene.xml")
    ap.add_argument("--drop", action="store_true", help="step physics (gravity) live")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as v:
        # nice default camera framing the robot
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 130, -15, 2.2
        v.cam.lookat[:] = (-0.2, 0.0, 0.15)
        t0 = time.time()
        while v.is_running():
            if args.drop:
                mujoco.mj_step(m, d)
            else:
                mujoco.mj_forward(m, d)   # kinematics only; joints stay where you drag them
            v.sync()
            # real-time pacing
            dt = m.opt.timestep if args.drop else 0.01
            time.sleep(max(0, dt - (time.time() - t0) % dt))

if __name__ == "__main__":
    main()
