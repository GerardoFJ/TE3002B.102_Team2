#!/usr/bin/env python3
"""Run the HOPPY hopping controller and log everything (rubrica Fase 4 + 5).

    python run_hop.py                  # headless, prints hop stats, saves hop_log.npz
    python run_hop.py --plots          # + Fase 5 analysis figures (PNG)
    python run_hop.py --video out.mp4  # + offscreen video of the hop
    python run_hop.py --live           # interactive viewer (needs a display)

Tunables live in HopConfig (hoppy_brain.py); override the common ones via flags.
"""
import os, sys, argparse
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoppy_brain import HoppyController, HopConfig

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "scene.xml")


def simulate(cfg, duration, q_init=None, record_video=None, fps=60, model_mod=None):
    m = mujoco.MjModel.from_xml_path(SCENE)
    m.opt.timestep = cfg.dt
    if model_mod is not None:        # ablation hook (Fase 2.2): mutate the model
        model_mod(m)                 # in place to disable one physical effect
    d = mujoco.MjData(m)
    if q_init is not None:
        d.qpos[:] = q_init
    ctl = HoppyController(m, cfg)

    nstep = int(duration / cfg.dt)
    log = {k: [] for k in ("t", "qpos", "qvel_true", "qvel_est", "tau",
                            "foot_pos", "foot_vel_est", "fN", "touch",
                            "F_cmd", "state", "V", "i", "Fz_des", "hip_pos",
                            "foot_delta", "foot_contact")}
    touch_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "foot_touch")
    transitions = []

    frames, ren = [], None
    if record_video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        ren = mujoco.Renderer(m, 720, 1280)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
        # FIXED shot centered on the tower so the boom is seen lapping around it
        cam.lookat[:] = (0.0, 0.0, 0.12)
        cam.azimuth, cam.elevation, cam.distance = 90, -22, 2.3
        vid_every = max(1, int(1.0 / (fps * cfg.dt)))
        leg_mid_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Link3")

    for k in range(nstep):
        t = k * cfg.dt
        tau = ctl.apply(d, t)
        L = ctl.last
        if L.get("transition"):
            transitions.append((t, L["transition"], d.xpos[ctl.hip_bid][2]))
        # foot velocity estimate (J * qd_est) for Cartesian analysis
        J = ctl.foot_jacobian_hk(d)
        vfoot = J @ np.array([L["v_est"][ctl.hip_col], L["v_est"][ctl.knee_col]])

        log["t"].append(t)
        log["qpos"].append(d.qpos.copy())
        log["qvel_true"].append(d.qvel.copy())
        log["qvel_est"].append(L["v_est"].copy())
        log["tau"].append(tau.copy())
        log["foot_pos"].append(L["p_foot"].copy())
        log["foot_vel_est"].append(vfoot.copy())
        log["fN"].append(L["fN"])
        log["touch"].append(float(d.sensordata[touch_sid]))
        log["F_cmd"].append(np.atleast_1d(L["F_cmd"]).copy())
        log["state"].append(0 if L["state"] == "FLIGHT" else 1)
        log["V"].append(L["V"].copy())
        log["i"].append(L["i"].copy())
        log["Fz_des"].append(L["Fz_des"])
        log["hip_pos"].append(L["p_hip"].copy())
        log["foot_delta"].append(L["foot_delta"])
        log["foot_contact"].append(float(L["foot_contact"]))

        mujoco.mj_step(m, d)

        if record_video and (k % vid_every == 0):
            ren.update_scene(d, cam); frames.append(ren.render())  # fixed cam

    if ren is not None:
        import imageio
        imageio.mimsave(record_video, frames, fps=fps)
        ren.close()
        print(f"  video -> {record_video}  ({len(frames)} frames)")

    for k in log:
        log[k] = np.array(log[k])
    return m, log, transitions, ctl


def hop_stats(log, transitions, apex_min=0.03, dur_min=0.05):
    """Real hops only: flight phases that clear apex_min and last > dur_min.
    Brief liftoffs during the thrust transient are filtered out."""
    state = log["state"]; t = log["t"]
    foot_z = log["foot_pos"][:, 2]
    flights, i = [], 0
    n = len(state)
    while i < n:
        if state[i] == 0:  # FLIGHT
            j = i
            while j < n and state[j] == 0:
                j += 1
            seg = slice(i, j)
            apex = foot_z[seg].max()
            dur = t[min(j, n - 1)] - t[i]
            if apex > apex_min and dur > dur_min:
                flights.append((t[i], t[min(j, n - 1)], apex))
            i = j
        else:
            i += 1
    n_td = len(flights)   # one touchdown ends each real hop
    return flights, n_td


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--f_peak", type=float)
    ap.add_argument("--t_push", type=float)
    ap.add_argument("--kp_cart", type=float)
    ap.add_argument("--kv_fwd", type=float)
    ap.add_argument("--k_place", type=float)
    ap.add_argument("--k_yaw", type=float)
    ap.add_argument("--v_des", type=float)
    ap.add_argument("--video", type=str)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "hop_log.npz"))
    args = ap.parse_args()

    cfg = HopConfig()
    for k in ("f_peak", "t_push", "kp_cart", "kv_fwd", "k_place", "k_yaw", "v_des"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))

    if args.live:
        return run_live(cfg)

    m, log, transitions, ctl = simulate(cfg, args.duration, record_video=args.video)
    flights, n_td = hop_stats(log, transitions)

    print(f"\n=== HOP STATS  (f_peak={cfg.f_peak} N, t_push={cfg.t_push}s, "
          f"kp_cart={cfg.kp_cart}) ===")
    print(f"  duration {args.duration}s | touchdowns: {n_td} | flight phases: {len(flights)}")
    stance_frac = np.mean(log["state"] == 1)
    print(f"  stance fraction: {stance_frac*100:.0f}%  | peak |tau|: "
          f"{np.abs(log['tau']).max():.2f} N*m | peak GRF: {log['fN'].max():.1f} N")
    for i, (ta, tb, apex) in enumerate(flights[:8]):
        print(f"   flight {i+1}: t=[{ta:.2f},{tb:.2f}]s  foot apex z={apex:.3f} m")
    yaw_travel = log["qpos"][-1, 0] - log["qpos"][0, 0]
    print(f"  yaw travel: {np.degrees(yaw_travel):+.1f} deg")

    np.savez(args.out, **log)
    print(f"  log -> {args.out}")

    if args.plots:
        import plots
        plots.make_all(args.out)


def _add_geom(scn, gtype, size, pos, rgba, label=""):
    """Append one decorative geom (with optional text label) to a user scene."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, float))
    g.label = label
    scn.ngeom += 1


def _draw_sensor_overlay(scn, d, ctl, t):
    """Draw HOPPY's foot-sensor data DIRECTLY in the MuJoCo scene: a contact
    indicator at the foot, a delta 'thermometer' above the tower, and text."""
    L = ctl.last
    scn.ngeom = 0
    pf = d.site_xpos[ctl.foot_sid]
    contact = L["foot_contact"]; delta = L["foot_delta"]; stroke = 0.0254
    SP, BOX = mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_BOX
    cg = [0.10, 0.90, 0.25, 1] if contact else [0.90, 0.15, 0.10, 1]
    # 1) contact indicator: glowing ball at the foot, green=closed / red=open
    _add_geom(scn, SP, [0.03, 0, 0], pf, cg)
    # 2) delta 'thermometer' above the tower: white full-scale frame + filled bar
    z0, H = 0.30, 0.30                                   # base height, full height
    _add_geom(scn, BOX, [0.012, 0.012, H / 2], [0, 0, z0 + H / 2], [1, 1, 1, 0.12])
    h = max(1e-4, min(delta / stroke, 1.0) * H)
    _add_geom(scn, BOX, [0.02, 0.02, h / 2], [0, 0, z0 + h / 2], [0.2, 0.55, 1.0, 0.9])
    _add_geom(scn, SP, [0.001, 0, 0], [0, 0, z0 + H + 0.04],
              [1, 1, 1, 1], f"sensor pie  delta={delta*1000:4.1f}mm")
    # 3) floating text: state / contact / force / lap
    yaw = np.degrees(d.qpos[ctl.qadr["yaw"]])
    _add_geom(scn, SP, [0.001, 0, 0], [0, 0, z0 + H + 0.13], cg,
              f"{L['state']}   contacto: {'CERRADO' if contact else 'abierto'}")
    _add_geom(scn, SP, [0.001, 0, 0], [0, 0, z0 + H + 0.20], [1, 1, 1, 1],
              f"F_N={L['fN']:5.0f}N   vuelta={yaw:+5.0f} deg")


def run_live(cfg):
    import mujoco.viewer, time
    m = mujoco.MjModel.from_xml_path(SCENE); m.opt.timestep = cfg.dt
    d = mujoco.MjData(m); ctl = HoppyController(m, cfg)
    with mujoco.viewer.launch_passive(m, d) as v:
        # fixed shot centered on the tower so the boom is seen lapping around it
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 90, -20, 2.6
        v.cam.lookat[:] = (0.0, 0.0, 0.12)
        t = 0.0
        while v.is_running():
            ctl.apply(d, t); mujoco.mj_step(m, d); t += cfg.dt
            _draw_sensor_overlay(v.user_scn, d, ctl, t)   # sensor data IN MuJoCo
            v.sync(); time.sleep(cfg.dt)


if __name__ == "__main__":
    main()
