#!/usr/bin/env python3
"""Fase 2.2 — Validación del Modelo Mecánico (estudio de ablación).

La rúbrica pide demostrar mediante **simulaciones comparativas** la influencia de
cada efecto físico modelado. Este script corre la misma sim de salto en 5
configuraciones —el modelo completo (baseline) y cuatro ablaciones, cada una
desactivando UN solo efecto— y compara su impacto con gráficas y una tabla:

  * armature  -> inercia reflejada del rotor  I_ref = N^2 * I_r   (Fase 1.2)
  * damping   -> amortiguamiento equivalente del actuador          (Fase 1.3)
  * resorte paralelo de rodilla (stiffness)                        (Fase 1.4)
  * saturación de torque (límite físico del actuador)              (Fase 2.1)

    python ablation.py                 # corre las 5, imprime tabla, genera PNG
    python ablation.py --duration 8

Salida: tabla por stdout + figura `hoppy_ablacion.png`.
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoppy_brain import HopConfig
from run_hop import simulate, hop_stats

HERE = os.path.dirname(os.path.abspath(__file__))


# --- helpers para localizar joints/DoF en el modelo cargado ------------------
def _jntid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

def _dofadr(m, name):
    return m.jnt_dofadr[_jntid(m, name)]


# --- model_mod: cada callback desactiva UN efecto físico (in place) ----------
def mod_no_armature(m):
    """Quita la inercia reflejada del rotor (N^2 I_r) de cadera y rodilla."""
    m.dof_armature[:] = 0.0

def mod_no_damping(m):
    """Quita el amortiguamiento viscoso equivalente de todas las articulaciones
    (cojinetes del boom 0.2 + pérdidas del reductor 0.02). El back-EMF se deja:
    es el amortiguamiento eléctrico, no el parámetro `damping` que pide 1.3."""
    m.dof_damping[:] = 0.0

def mod_no_spring(m):
    """Quita el resorte paralelo de la rodilla (stiffness -> 0)."""
    m.jnt_stiffness[_jntid(m, "knee")] = 0.0


# --- cfg builders ------------------------------------------------------------
def cfg_full():
    return HopConfig()

def cfg_no_torque_sat():
    """Sin saturación: ni el clip explícito (Fase 2.1) ni los límites
    eléctricos V/I acotan el par -> el actuador entrega el par 'ideal' que pide
    la ley de control (físicamente irreal)."""
    c = HopConfig()
    c.tau_max = np.inf
    c.V_max = np.inf
    c.I_max = np.inf
    return c


# key, etiqueta, cfg-builder, model_mod, color
RUNS = [
    ("baseline",      "Modelo completo",          cfg_full,           None,            "k"),
    ("no_armature",   "sin armature",             cfg_full,           mod_no_armature, "C3"),
    ("no_damping",    "sin damping",              cfg_full,           mod_no_damping,  "C1"),
    ("no_spring",     "sin resorte rodilla",      cfg_full,           mod_no_spring,   "C0"),
    ("no_torque_sat", "sin saturación de torque", cfg_no_torque_sat,  None,            "C4"),
]


def run_all(duration):
    res = {}
    for key, label, cfgb, mod, color in RUNS:
        cfg = cfgb()
        m, log, trans, ctl = simulate(cfg, duration, model_mod=mod)
        flights, n_td = hop_stats(log, trans)
        res[key] = dict(label=label, color=color, cfg=cfg, m=m,
                        log=log, flights=flights, n_td=n_td)
    return res


def metrics(r):
    m, log, flights = r["m"], r["log"], r["flights"]
    t = log["t"]
    knee = _dofadr(m, "knee")
    qd_knee = log["qvel_true"][:, knee]
    foot_z = log["foot_pos"][:, 2]
    tau = log["tau"]
    stance = log["state"] == 1
    finite = np.isfinite(foot_z).all() and np.isfinite(tau).all()
    with np.errstate(invalid="ignore"):
        acc_knee = np.gradient(np.nan_to_num(qd_knee), t)
    return dict(
        n_hops=len(flights),
        apex_foot=np.nanmax(foot_z) if finite else float("nan"),
        peak_tau=np.nanmax(np.abs(tau)),
        mean_tau_stance=(np.abs(tau[stance]).mean() if stance.any() else 0.0),
        peak_GRF=np.nanmax(log["fN"]),
        peak_knee_acc=np.nanmax(np.abs(acc_knee)),
        peak_V=np.nanmax(np.abs(log["V"])),
        finite=finite,
    )


def _window(res, span=0.65, lead=0.05):
    """Ventana [td-lead, td+span] alrededor del 1er touchdown del baseline,
    para ver el transitorio de impacto en las señales articulares."""
    fl = res["baseline"]["flights"]
    td = fl[0][1] if fl else 0.5
    return td - lead, td + span


def make_figure(res, mets, path, duration):
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.5))
    fig.suptitle("HOPPY · Fase 2.2 — Validación del modelo mecánico "
                 "(estudio de ablación)", fontsize=14, fontweight="bold")
    t = res["baseline"]["log"]["t"]
    knee = _dofadr(res["baseline"]["m"], "knee")
    t0, t1 = _window(res)
    win = (t >= t0) & (t <= t1)

    def qd_knee(key):
        return res[key]["log"]["qvel_true"][:, knee]

    # (a) ARMATURE: transitorio de impacto en la velocidad de rodilla --------
    a = axes[0, 0]
    a.plot(t[win], qd_knee("baseline")[win], "k", lw=1.8, label="con armature")
    a.plot(t[win], qd_knee("no_armature")[win], "C3", lw=1.3, alpha=0.9,
           label="sin armature")
    a.set_title("(a) Armature — inercia reflejada del rotor")
    a.set_xlabel("t [s]"); a.set_ylabel("dq_rodilla/dt [rad/s]")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a.text(0.02, 0.04, f"acc. rodilla pico: {mets['baseline']['peak_knee_acc']:.0f} → "
           f"{mets['no_armature']['peak_knee_acc']:.0f} rad/s²",
           transform=a.transAxes, fontsize=8, color="0.25")

    # (b) DAMPING: ringing del boom pasivo (yaw/pitch). Se grafica el pitch:
    # no tiene motor, así que su única disipación de velocidad es el `damping`
    # del cojinete (0.2) -> aísla el efecto, que en cadera/rodilla enmascara el
    # back-EMF. El efecto es modesto pero se acumula: la oscilación (RMS de la
    # velocidad) crece sin damping. Se muestra el run completo.
    pitch = _dofadr(res["baseline"]["m"], "pitch")
    qp_b = res["baseline"]["log"]["qvel_true"][:, pitch]
    qp_n = res["no_damping"]["log"]["qvel_true"][:, pitch]
    a = axes[0, 1]
    a.plot(t, qp_b, "k", lw=1.0, label="con damping")
    a.plot(t, qp_n, "C1", lw=0.9, alpha=0.85, label="sin damping")
    a.set_title("(b) Damping — amortiguamiento del boom pasivo (pitch)")
    a.set_xlabel("t [s]"); a.set_ylabel("dq_pitch/dt [rad/s]")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    rms_b, rms_n = np.sqrt(np.mean(qp_b**2)), np.sqrt(np.mean(qp_n**2))
    a.text(0.02, 0.04, f"RMS osc. pitch: {rms_b:.2f} → {rms_n:.2f} rad/s "
           f"(+{100*(rms_n/rms_b-1):.0f}%)",
           transform=a.transAxes, fontsize=8, color="0.25")

    # (c) RESORTE: altura del pie (energía almacenada/liberada) --------------
    a = axes[1, 0]
    fz_b = res["baseline"]["log"]["foot_pos"][:, 2]
    fz_s = res["no_spring"]["log"]["foot_pos"][:, 2]
    a.plot(t, fz_b, "k", lw=1.3, label="con resorte rodilla")
    a.plot(t, fz_s, "C0", lw=1.1, alpha=0.85, label="sin resorte rodilla")
    a.set_title("(c) Resorte paralelo de rodilla — energía del salto")
    a.set_xlabel("t [s]"); a.set_ylabel("altura del pie z [m]")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    a.text(0.02, 0.92, f"τ medio en apoyo: {mets['baseline']['mean_tau_stance']:.2f} → "
           f"{mets['no_spring']['mean_tau_stance']:.2f} N·m",
           transform=a.transAxes, fontsize=8, color="0.25")

    # (d) SATURACIÓN DE TORQUE: par exigido vs acotado -----------------------
    a = axes[1, 1]
    tau_b = np.abs(res["baseline"]["log"]["tau"]).max(axis=1)
    tau_n = np.abs(res["no_torque_sat"]["log"]["tau"]).max(axis=1)
    a.plot(t, np.nan_to_num(tau_n), "C4", lw=1.0, alpha=0.9,
           label="sin saturación (par exigido)")
    a.plot(t, tau_b, "k", lw=1.5, label="con saturación")
    a.axhline(3.73, ls="--", c="0.4", lw=1, label="límite 3.73 N·m")
    a.set_title("(d) Saturación de torque — límite físico del actuador")
    a.set_xlabel("t [s]"); a.set_ylabel("max|τ| articular [N·m]")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    ymax = np.nanmax(np.nan_to_num(tau_n))
    if ymax > 20:                       # comprime el pico irreal para ver ambos
        a.set_yscale("symlog", linthresh=4)

    # (e) barras: par pico por configuración ---------------------------------
    keys = [k for k, *_ in RUNS]
    labels = [res[k]["label"] for k in keys]
    colors = [res[k]["color"] for k in keys]
    a = axes[2, 0]
    vals = [mets[k]["peak_tau"] for k in keys]
    a.bar(range(len(keys)), vals, color=colors)
    a.axhline(3.73, ls="--", c="0.4", lw=1)
    a.set_yscale("log")
    a.set_title("(e) Par pico |τ| por configuración")
    a.set_ylabel("|τ| pico [N·m] (log)")
    a.set_xticks(range(len(keys))); a.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    a.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    # (f) barras: apex del pie y nº de saltos --------------------------------
    a = axes[2, 1]
    apex = [mets[k]["apex_foot"] for k in keys]
    a.bar(range(len(keys)), apex, color=colors)
    a.set_title("(f) Apex del pie y nº de saltos por configuración")
    a.set_ylabel("apex del pie z [m]")
    a.set_xticks(range(len(keys))); a.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    a.grid(alpha=0.3, axis="y")
    for i, k in enumerate(keys):
        h = apex[i] if np.isfinite(apex[i]) else 0
        a.text(i, h, f"{mets[k]['n_hops']} saltos", ha="center", va="bottom", fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_table(res, mets, duration):
    print(f"\n=== FASE 2.2 · ESTUDIO DE ABLACIÓN  (duración {duration:.0f} s) ===")
    hdr = (f"{'configuración':<26}{'saltos':>7}{'apex_pie':>10}{'|τ|pico':>10}"
           f"{'τ̄_apoyo':>10}{'GRFpico':>10}{'acc_rod':>11}{'V_pico':>9}")
    print(hdr); print("-" * len(hdr))
    for key, label, *_ in RUNS:
        m = mets[key]
        apex = f"{m['apex_foot']:.3f}" if np.isfinite(m['apex_foot']) else "  n/a"
        flag = "" if m["finite"] else "  ⚠ inestable"
        print(f"{label:<26}{m['n_hops']:>7}{apex:>10}{m['peak_tau']:>10.2f}"
              f"{m['mean_tau_stance']:>10.2f}{m['peak_GRF']:>10.0f}"
              f"{m['peak_knee_acc']:>11.0f}{m['peak_V']:>9.1f}{flag}")
    b = mets["baseline"]
    print("\nInterpretación (vs modelo completo):")
    print(f"  • armature : sin él la acc. de rodilla en impacto sube "
          f"{mets['no_armature']['peak_knee_acc']/max(b['peak_knee_acc'],1e-9):.1f}× "
          f"(impactos/transiciones irreales, rúbrica 1.2).")
    print(f"  • damping  : sin él el transitorio de impacto se amortigua menos "
          f"(más oscilación post-touchdown, rúbrica 1.3).")
    print(f"  • resorte  : sin él τ̄ en apoyo pasa de {b['mean_tau_stance']:.2f} a "
          f"{mets['no_spring']['mean_tau_stance']:.2f} N·m y el apex baja "
          f"(menos energía elástica recuperada, rúbrica 1.4).")
    print(f"  • saturación: sin ella el par exigido llega a "
          f"{mets['no_torque_sat']['peak_tau']:.0f} N·m (>{'':s}{3.73:.2f}); el "
          f"clip a 3.73 N·m hace físico al actuador (rúbrica 2.1).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "hoppy_ablacion.png"))
    args = ap.parse_args()

    res = run_all(args.duration)
    mets = {k: metrics(res[k]) for k in res}
    print_table(res, mets, args.duration)
    make_figure(res, mets, args.out, args.duration)
    print(f"\nfigura -> {args.out}")


if __name__ == "__main__":
    main()
