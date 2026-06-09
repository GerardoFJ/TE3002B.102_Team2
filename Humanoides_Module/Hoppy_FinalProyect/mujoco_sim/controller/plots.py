#!/usr/bin/env python3
"""Fase 5.2 analysis figures from a hop_log.npz produced by run_hop.py.

    python plots.py [hop_log.npz]

Generates:
  hoppy_signals.png   -- joint & Cartesian positions/velocities, contact force,
                         control torques, and FLIGHT/STANCE timeline.
  hoppy_encoder.png   -- encoder-emulated vs true joint velocity (Fase 5.1).
  hoppy_limitcycle.png-- foot height vs vertical velocity phase portrait.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JN = ["yaw", "pitch", "hip", "knee"]
HIP, KNEE = 2, 3


def _shade_stance(ax, t, state):
    """Light shading over STANCE (state==1) intervals."""
    s = state.astype(int)
    edges = np.diff(np.concatenate([[0], s, [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    for a, b in zip(starts, ends):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color="0.85", lw=0, zorder=0)


def make_all(npz="hop_log.npz"):
    d = np.load(npz)
    t = d["t"]; state = d["state"]
    qpos = d["qpos"]; qvt = d["qvel_true"]; qve = d["qvel_est"]
    tau = d["tau"]; fpos = d["foot_pos"]; fvel = d["foot_vel_est"]
    fN = d["fN"]; touch = d["touch"]
    outdir = os.path.dirname(os.path.abspath(npz))

    # ---- touchdown / lift-off events (Fase 3.2) ----
    s = state.astype(int)
    edges = np.diff(np.concatenate([[0], s]))
    td = t[np.where(edges == 1)[0]]   # FLIGHT->STANCE
    lo = t[np.where(edges == -1)[0]]  # STANCE->FLIGHT

    # =================== Figure 1: signals ===================
    fig, ax = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    fig.suptitle("HOPPY — análisis de señales (Fase 5.2)", fontsize=14, weight="bold")

    # joint positions
    a = ax[0, 0]
    for i, n in enumerate(JN):
        a.plot(t, qpos[:, i], label=n)
    _shade_stance(a, t, state); a.set_ylabel("q [rad]"); a.legend(ncol=4, fontsize=8)
    a.set_title("Posiciones articulares  (sombreado = STANCE)")

    # joint velocities (estimated) hip/knee
    a = ax[0, 1]
    a.plot(t, qve[:, HIP], label="hip est"); a.plot(t, qve[:, KNEE], label="knee est")
    _shade_stance(a, t, state); a.set_ylabel("dq/dt [rad/s]"); a.legend(fontsize=8)
    a.set_title("Velocidades articulares estimadas (encoder)")

    # foot cartesian position
    a = ax[1, 0]
    for i, n in enumerate("xyz"):
        a.plot(t, fpos[:, i], label="foot " + n)
    _shade_stance(a, t, state); a.axhline(0.012, ls=":", c="r", lw=0.8)
    a.set_ylabel("p_foot [m]"); a.legend(ncol=3, fontsize=8)
    a.set_title("Posición cartesiana del pie")

    # foot cartesian velocity (estimated)
    a = ax[1, 1]
    for i, n in enumerate("xyz"):
        a.plot(t, fvel[:, i], label="v" + n)
    _shade_stance(a, t, state); a.set_ylabel("v_foot [m/s]"); a.legend(ncol=3, fontsize=8)
    a.set_title("Velocidad cartesiana del pie (estimada)")

    # contact force
    a = ax[2, 0]
    a.plot(t, fN, label="GRF normal (contacto)", c="C3")
    a.plot(t, touch, label="sensor touch", c="C0", alpha=0.6)
    for x in td: a.axvline(x, c="g", ls="--", lw=0.7)
    for x in lo: a.axvline(x, c="m", ls=":", lw=0.7)
    _shade_stance(a, t, state); a.set_ylabel("F_N [N]"); a.legend(fontsize=8)
    a.set_title("Fuerza de contacto  (— verde=touchdown, : magenta=lift-off)")

    # control torques
    a = ax[2, 1]
    a.plot(t, tau[:, 0], label="τ hip"); a.plot(t, tau[:, 1], label="τ knee")
    a.axhline(3.73, ls=":", c="k", lw=0.8); a.axhline(-3.73, ls=":", c="k", lw=0.8)
    _shade_stance(a, t, state); a.set_ylabel("τ [N·m]"); a.legend(fontsize=8)
    a.set_title("Torques de control (: saturación ±3.73 N·m)")

    # FSM timeline
    a = ax[3, 0]
    a.plot(t, state, drawstyle="steps-post", c="k")
    a.set_yticks([0, 1]); a.set_yticklabels(["FLIGHT", "STANCE"])
    a.set_ylabel("estado FSM"); a.set_xlabel("t [s]"); a.set_title("Máquina de estados híbrida")

    # foot height detail
    a = ax[3, 1]
    a.plot(t, fpos[:, 2], c="C2"); _shade_stance(a, t, state)
    a.axhline(0.012, ls=":", c="r", lw=0.8)
    a.set_ylabel("foot z [m]"); a.set_xlabel("t [s]"); a.set_title("Altura del pie (saltos)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = os.path.join(outdir, "hoppy_signals.png"); fig.savefig(p1, dpi=110); plt.close(fig)

    # =================== Figure 2: encoder emulation ===================
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Fase 5.1 — Emulación de encoder: velocidad estimada vs real",
                 fontsize=13, weight="bold")
    for k, j, name in [(0, HIP, "hip"), (1, KNEE, "knee")]:
        a = ax[k]
        a.plot(t, qvt[:, j], c="0.6", lw=2, label="qvel real (MuJoCo)")
        a.plot(t, qve[:, j], c="C1", lw=1, label="estimada (Δqpos cuantizado + LP)")
        _shade_stance(a, t, state)
        err = np.sqrt(np.mean((qvt[:, j] - qve[:, j]) ** 2))
        a.set_ylabel(f"{name}  dq/dt [rad/s]"); a.legend(fontsize=9, loc="upper right")
        a.set_title(f"{name}: RMSE estimación = {err:.3f} rad/s")
    ax[1].set_xlabel("t [s]")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = os.path.join(outdir, "hoppy_encoder.png"); fig.savefig(p2, dpi=110); plt.close(fig)

    # =================== Figure 3: limit cycle ===================
    fig, a = plt.subplots(figsize=(6.5, 6))
    # use second half (steady state)
    h = len(t) // 2
    a.plot(fpos[h:, 2], fvel[h:, 2], lw=0.8, c="C0")
    a.set_xlabel("altura del pie  z [m]"); a.set_ylabel("velocidad vertical del pie [m/s]")
    a.set_title("Ciclo límite del salto (régimen permanente)")
    a.grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(outdir, "hoppy_limitcycle.png"); fig.savefig(p3, dpi=110); plt.close(fig)

    # =================== Figure 4: companion-style 6-panel resumen ===========
    V = d["V"]; Fz_des = d["Fz_des"]; hipz = d["hip_pos"][:, 2]
    dt = t[1] - t[0]
    hipz_vel = np.gradient(hipz, dt)
    # count apexes (hip height local maxima above a threshold) for the title
    apex_h = hipz.max()
    n_hops = len(td)
    fig, ax = plt.subplots(3, 2, figsize=(13, 9.5))
    fig.suptitle("HOPPY — resultados del salto (resumen)", fontsize=14, weight="bold")

    # 1) altura del cuerpo (cadera) — salto sostenido
    a = ax[0, 0]
    a.plot(t, hipz, "b", lw=0.9); _shade_stance(a, t, state); a.grid(alpha=0.3)
    a.set_title(f"Altura de la cadera ({n_hops} saltos, máx={apex_h:.3f} m)")
    a.set_xlabel("t [s]"); a.set_ylabel("z [m]")

    # 2) fuerza deseada (medio-seno) vs fuerza real de contacto
    a = ax[0, 1]
    a.plot(t, Fz_des, "r", lw=0.9, label="Fz deseada (medio-seno)")
    a.plot(t, fN, "k", lw=0.8, label="Fz real (contacto)")
    a.grid(alpha=0.3); a.legend(fontsize=8); a.set_title("Fuerza de reacción del pie")
    a.set_xlabel("t [s]"); a.set_ylabel("N")
    if t[-1] > 4: a.set_xlim(2, 4)

    # 3) torques articulares con saturación
    a = ax[1, 0]
    a.plot(t, tau[:, 0], "b", lw=0.8, label="cadera")
    a.plot(t, tau[:, 1], "r", lw=0.8, label="rodilla")
    a.axhline(3.73, c="k", ls="--", lw=0.7); a.axhline(-3.73, c="k", ls="--", lw=0.7)
    a.grid(alpha=0.3); a.legend(fontsize=8); a.set_title("Pares articulares (±3.73 N·m)")
    a.set_xlabel("t [s]"); a.set_ylabel("N·m")
    if t[-1] > 4: a.set_xlim(2, 4)

    # 4) voltaje del motor con límite ±12 V (Fase actuadores)
    a = ax[1, 1]
    a.plot(t, V[:, 0], "b", lw=0.6, label="V cadera")
    a.plot(t, V[:, 1], "r", lw=0.6, label="V rodilla")
    a.axhline(12, c="k", ls="--", lw=0.8); a.axhline(-12, c="k", ls="--", lw=0.8)
    a.grid(alpha=0.3); a.legend(fontsize=8); a.set_title("Voltaje del motor (límite ±12 V)")
    a.set_xlabel("t [s]"); a.set_ylabel("V")
    if t[-1] > 4: a.set_xlim(2, 4)

    # 5) velocidad de cadera: real vs estimada por encoder
    a = ax[2, 0]
    a.plot(t, qvt[:, HIP], "c", lw=0.7, label="real (qvel)")
    a.plot(t, qve[:, HIP], "b", lw=0.9, label="estimada (encoder)")
    a.grid(alpha=0.3); a.legend(fontsize=8)
    a.set_title("Velocidad de cadera: real vs derivada filtrada")
    a.set_xlabel("t [s]"); a.set_ylabel("rad/s")
    if t[-1] > 3: a.set_xlim(2, 3)

    # 6) ciclo límite (retrato de fase del cuerpo)
    a = ax[2, 1]
    h = len(t) // 3
    a.plot(hipz[h:], hipz_vel[h:], "b", lw=0.5)
    a.grid(alpha=0.3); a.set_title("Ciclo límite (retrato de fase de la cadera)")
    a.set_xlabel("z [m]"); a.set_ylabel("dz/dt [m/s]")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p4 = os.path.join(outdir, "hoppy_resultados.png"); fig.savefig(p4, dpi=120); plt.close(fig)

    # =================== Figure 5: HOPPY foot contact sensor =================
    # The Linear_Disp_Sensor_404R1KL1.0: displacement (analog pot) -> binary
    # switch -> FSM. Shows the sensor chain that drives touchdown/lift-off.
    if "foot_delta" in d.files:
        delta_mm = d["foot_delta"] * 1000.0
        contact = d["foot_contact"]
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("HOPPY — sensor de contacto del pie (Linear_Disp_Sensor 404R1KL1.0)",
                     fontsize=13, weight="bold")
        a = ax[0]
        a.plot(t, delta_mm, c="C0", lw=0.9)
        a.axhline(25.4, ls=":", c="k", lw=0.8, label="fin de carrera (1 in = 25.4 mm)")
        _shade_stance(a, t, state); a.set_ylabel("δ [mm]"); a.legend(fontsize=8)
        a.set_title("Desplazamiento lineal del pie (potenciómetro, cuantizado por ADC)")
        a = ax[1]
        a.plot(t, fN, c="C3", lw=0.9, label="fuerza de contacto F_N")
        a.set_ylabel("F_N [N]"); a.legend(fontsize=8, loc="upper right")
        a.set_title("Fuerza de reacción (de la que el potenciómetro deriva δ = F_N / k_pie)")
        _shade_stance(a, t, state)
        a = ax[2]
        a.plot(t, contact, drawstyle="steps-post", c="C2", lw=1.3, label="switch binario del pie")
        a.plot(t, state, drawstyle="steps-post", c="k", lw=0.8, ls="--", label="estado FSM (1=STANCE)")
        for x in td: a.axvline(x, c="g", ls="--", lw=0.6)
        for x in lo: a.axvline(x, c="m", ls=":", lw=0.6)
        a.set_yticks([0, 1]); a.set_yticklabels(["abierto/FLIGHT", "cerrado/STANCE"])
        a.set_ylabel("contacto"); a.set_xlabel("t [s]"); a.legend(fontsize=8, loc="upper right")
        a.set_title("Switch de contacto (con histéresis) → transiciones de la FSM")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p5 = os.path.join(outdir, "hoppy_foot_sensor.png"); fig.savefig(p5, dpi=110); plt.close(fig)
        out = (p1, p2, p3, p4, p5)
    else:
        out = (p1, p2, p3, p4)

    print("figures ->")
    for p in out:
        print("  ", p)
    return out


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "hop_log.npz")
    make_all(npz)
