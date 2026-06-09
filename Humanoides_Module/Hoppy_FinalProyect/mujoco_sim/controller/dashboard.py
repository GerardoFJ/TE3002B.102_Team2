#!/usr/bin/env python3
"""HOPPY — Dashboard interactivo en tiempo real (render 3D + telemetría).

Tema oscuro, con tarjetas KPI, medidores de utilización, barra de tiempo real y
BOTONES para cambiar de vista y controlar la sim.

  Banda superior (KPI): frecuencia de salto, apex, duty factor, velocidad de
                        avance, potencia pico, costo de transporte.
  Tarjeta de estado   : badge FSM + lecturas + 3 medidores (torque/voltaje/
                        corriente, % del límite físico) con foco de saturación.
  Vistas (botones)    : Telemetría · Encoder (5.1) · Sensor pie · Fase/Energía.
  Controles (botones) : Pausa/Reanudar · Reset · Modo Avance/En sitio.
  Pie de página       : factor de tiempo real (sim/wall), FPS, indicador EN VIVO.

Rendimiento: el modo en vivo usa BLITTING (redibuja solo los artistas dinámicos
sobre un fondo cacheado) -> ~3x FPS vs un redraw completo de matplotlib. Para
ello los ejes usan límites FIJOS y el eje X es tiempo RELATIVO (ahora = 0). El
modo --record usa redraw completo (necesita el frame entero para el video). La
física es independiente del dibujo: nada de esto altera la simulación.

    python dashboard.py                 # ventana en vivo (tiempo real, TkAgg)
    python dashboard.py --record dash.mp4 --duration 10   # graba a mp4 (headless)
    python dashboard.py --page energia                    # abrir en una vista
"""
import os, sys, time, argparse
from collections import deque
import numpy as np
import mujoco

os.environ.setdefault("MUJOCO_GL", "egl")     # render offscreen -> array
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoppy_brain import HoppyController, HopConfig

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "scene.xml")
G = 9.81

P = dict(
    bg="#0d1117", panel="#161b22", card="#1c2333", edge="#30363d", grid="#222a35",
    fg="#e6edf3", muted="#8b949e", accent="#58a6ff",
    foot="#2ec4ff", hip="#7d8590", grf="#ff6b6b", delta="#20e3b2",
    tau_h="#ffa94d", tau_k="#69db7c", v_h="#ffa94d", v_k="#69db7c",
    true="#7d8590", est="#58a6ff", pmech="#2ec4ff", pelec="#ffa94d",
    stance="#2f9e44", flight="#1f6feb", limit="#6e7681", warn="#e3b341",
)


def _style_axis(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=P["fg"], pad=6)
    ax.set_facecolor(P["panel"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(P["edge"])
    ax.grid(True, color=P["grid"], lw=0.7, alpha=0.9)
    ax.tick_params(length=0, colors=P["muted"], labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel)


def _legend(ax, **kw):
    lg = ax.legend(facecolor=P["card"], edgecolor=P["edge"], framealpha=0.92,
                   fontsize=7.5, labelcolor=P["fg"], **kw)
    lg.get_frame().set_linewidth(0.8)
    return lg


def _util_color(frac):
    return P["stance"] if frac < 0.8 else (P["warn"] if frac < 1.0 else P["grf"])


class Gauge:
    def __init__(self, ax, label, blit):
        from matplotlib.patches import Wedge
        self.ax = ax; self.blit = blit
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.42, 1.18)
        ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.30, fc=P["panel"], ec=P["edge"], lw=1.0, zorder=1))
        self.fill = None
        self.val = ax.text(0, 0.30, "0%", ha="center", va="center", fontsize=11,
                           fontweight="bold", color=P["fg"], zorder=4)
        self.val.set_animated(blit)
        ax.text(0, -0.30, label, ha="center", va="center", fontsize=8, color=P["muted"], fontweight="bold")

    def set(self, frac):
        from matplotlib.patches import Wedge
        if self.fill is not None:
            self.fill.remove()
        f = min(max(frac, 0.0), 1.0); col = _util_color(frac)
        self.fill = Wedge((0, 0), 1.0, 180 - 180 * f, 180, width=0.30, fc=col, ec="none", zorder=2)
        self.fill.set_animated(self.blit); self.ax.add_patch(self.fill)
        self.val.set_text(f"{frac*100:.0f}%"); self.val.set_color(col if frac >= 1.0 else P["fg"])

    def draw(self):
        self.ax.draw_artist(self.fill); self.ax.draw_artist(self.val)


class Telemetry:
    KEYS = ("t", "foot_z", "hip_z", "fN", "delta", "tau_h", "tau_k", "V_h", "V_k",
            "stance", "vtr_h", "vte_h", "vtr_k", "vte_k", "contact",
            "foot_vz", "p_mech", "p_elec", "v_fwd")

    def __init__(self, window_s, sample_dt):
        n = max(8, int(window_s / sample_dt))
        self.buf = {k: deque(maxlen=n) for k in self.KEYS}

    def push(self, t, L, ctl, d):
        b = self.buf; hc, kc = ctl.hip_col, ctl.knee_col
        b["t"].append(t)
        b["foot_z"].append(L["p_foot"][2]); b["hip_z"].append(L["p_hip"][2])
        b["fN"].append(L["fN"]); b["delta"].append(L["foot_delta"] * 1000.0)
        b["tau_h"].append(L["tau"][0]); b["tau_k"].append(L["tau"][1])
        b["V_h"].append(L["V"][0]); b["V_k"].append(L["V"][1])
        b["stance"].append(1.0 if L["state"] == "STANCE" else 0.0)
        b["vte_h"].append(L["v_est"][hc]); b["vte_k"].append(L["v_est"][kc])
        vtr = np.array([d.qvel[hc], d.qvel[kc]])
        b["vtr_h"].append(vtr[0]); b["vtr_k"].append(vtr[1])
        b["contact"].append(1.0 if L["foot_contact"] else 0.0)
        J = ctl.foot_jacobian_hk(d)
        vfoot = J @ np.array([L["v_est"][hc], L["v_est"][kc]])
        b["foot_vz"].append(float(vfoot[2]))
        b["p_mech"].append(float(L["tau"][0] * vtr[0] + L["tau"][1] * vtr[1]))
        b["p_elec"].append(float(L["V"][0] * L["i"][0] + L["V"][1] * L["i"][1]))
        Rb = max(np.linalg.norm(L["p_foot"][:2]), 1e-3)
        b["v_fwd"].append(float(d.qvel[ctl.vadr["yaw"]] * Rb))

    def clear(self):
        for q in self.buf.values():
            q.clear()

    def arr(self, k):
        return np.asarray(self.buf[k])


def _overlay_contact(scene, d, ctl):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    rgba = [0.18, 0.80, 0.27, 1] if ctl.last["foot_contact"] else [1.0, 0.42, 0.42, 1]
    pf = d.site_xpos[ctl.foot_sid]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.03, 0, 0.]),
                        np.asarray(pf, float), np.eye(3).flatten(), np.asarray(rgba, float))
    scene.ngeom += 1


class Dashboard:
    KPI = [("Frec. salto", "Hz"), ("Apex pie", "cm"), ("Duty apoyo", "%"),
           ("Avance", "m/s"), ("Pot. pico", "W"), ("Costo transp.", "")]

    def __init__(self, fig, window_s, m, cfg, m_total, blit):
        import matplotlib.patches as mpatches
        from matplotlib.collections import PolyCollection
        from matplotlib.widgets import Button
        self.fig, self.window_s, self.m, self.cfg, self.m_total = fig, window_s, m, cfg, m_total
        self.blit = blit; self.bg = None
        yj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
        self.yaw_dof = m.jnt_dofadr[yj]
        self.page = "telem"; self.paused = False; self._reset = False; self.mode = "avance"; self._stop = False

        def A(art):                       # mark dynamic (animated under blit)
            art.set_animated(blit); return art

        gs = fig.add_gridspec(4, 4, hspace=0.72, wspace=0.42, left=0.035, right=0.965, top=0.80, bottom=0.155)
        fig.text(0.035, 0.957, "HOPPY", fontsize=21, fontweight="bold", color=P["accent"])
        fig.text(0.112, 0.961, "· Dashboard en tiempo real", fontsize=12.5, color=P["muted"])
        fig.text(0.965, 0.961, "MuJoCo 3.9  ·  Fase 1–5", fontsize=9, color=P["muted"], ha="right")

        # KPI band
        self.ax_kpi = fig.add_axes([0.035, 0.835, 0.93, 0.085]); self.ax_kpi.axis("off")
        self.kpi_val = []
        n = len(self.KPI); gap = 0.011; w = (1 - (n + 1) * gap) / n
        for i, (lab, unit) in enumerate(self.KPI):
            x0 = gap + i * (w + gap)
            self.ax_kpi.add_patch(mpatches.FancyBboxPatch((x0, 0.06), w, 0.88, transform=self.ax_kpi.transAxes,
                boxstyle="round,pad=0,rounding_size=0.04", fc=P["card"], ec=P["edge"], lw=1.0))
            self.ax_kpi.text(x0 + 0.022, 0.74, lab.upper(), transform=self.ax_kpi.transAxes,
                             fontsize=8, color=P["muted"], fontweight="bold", va="center")
            v = A(self.ax_kpi.text(x0 + 0.022, 0.33, "—", transform=self.ax_kpi.transAxes,
                                   fontsize=15, color=P["fg"], fontweight="bold", va="center"))
            self.ax_kpi.text(x0 + w - 0.018, 0.30, unit, transform=self.ax_kpi.transAxes,
                             fontsize=8.5, color=P["muted"], va="center", ha="right")
            self.kpi_val.append(v)

        # 3D render
        self.ax_im = fig.add_subplot(gs[0:3, 0:2])
        self.ax_im.set_xticks([]); self.ax_im.set_yticks([])
        for s in self.ax_im.spines.values():
            s.set_color(P["edge"])
        self.ax_im.set_title("Vista 3D", loc="left", fontsize=10.5, fontweight="bold", color=P["fg"], pad=6)
        self.im = A(self.ax_im.imshow(np.zeros((560, 680, 3), np.uint8), aspect="auto", interpolation="antialiased"))

        # status card + gauges
        self.ax_txt = fig.add_subplot(gs[3, 0:2]); self.ax_txt.axis("off")
        self.ax_txt.add_patch(mpatches.FancyBboxPatch((0.004, 0.06), 0.992, 0.88, transform=self.ax_txt.transAxes,
            boxstyle="round,pad=0,rounding_size=0.05", fc=P["card"], ec=P["edge"], lw=1.0, zorder=1))
        self.badge = A(mpatches.FancyBboxPatch((0.018, 0.24), 0.125, 0.52, transform=self.ax_txt.transAxes,
            boxstyle="round,pad=0,rounding_size=0.12", fc=P["flight"], ec="none", zorder=2))
        self.ax_txt.add_patch(self.badge)
        self.badge_txt = A(self.ax_txt.text(0.0805, 0.50, "FLIGHT", transform=self.ax_txt.transAxes,
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0d1117", zorder=3))
        self.m1 = A(self.ax_txt.text(0.155, 0.64, "", transform=self.ax_txt.transAxes,
            va="center", ha="left", fontsize=9.5, family="monospace", color=P["fg"]))
        self.m2 = A(self.ax_txt.text(0.155, 0.36, "", transform=self.ax_txt.transAxes,
            va="center", ha="left", fontsize=9.5, family="monospace", color=P["fg"]))
        p = self.ax_txt.get_position()
        self.gauges = []
        for i, lab in enumerate(("TORQUE", "VOLTAJE", "CORRIENTE")):
            axg = fig.add_axes([p.x0 + p.width * (0.605 + 0.132 * i), p.y0 + p.height * 0.12,
                                p.width * 0.115, p.height * 0.78])
            self.gauges.append(Gauge(axg, lab, blit))
        self.sat_lamp = A(self.ax_txt.text(0.605, 0.95, "", transform=self.ax_txt.transAxes,
            fontsize=8, fontweight="bold", color=P["grf"], va="center", ha="left", zorder=3))

        # ---- pages ----
        self.ax_h = fig.add_subplot(gs[0, 2:4]); self.ax_f = fig.add_subplot(gs[1, 2:4])
        self.ax_t = fig.add_subplot(gs[2, 2:4]); self.ax_v = fig.add_subplot(gs[3, 2:4])
        self.ax_d = self.ax_f.twinx()
        self.l_footz = A(self.ax_h.plot([], [], color=P["foot"], lw=1.7, label="pie z")[0])
        self.l_hipz = A(self.ax_h.plot([], [], color=P["hip"], lw=1.1, label="cadera z")[0])
        self.d_footz = A(self.ax_h.plot([], [], "o", ms=5, color=P["foot"])[0])
        _style_axis(self.ax_h, "Altura del salto", "z [m]"); self.ax_h.set_ylim(-0.02, 0.30); _legend(self.ax_h, loc="upper right", ncol=2)
        self.l_fN = A(self.ax_f.plot([], [], color=P["grf"], lw=1.5, label="GRF F_N")[0])
        self.d_fN = A(self.ax_f.plot([], [], "o", ms=5, color=P["grf"])[0])
        self.l_del = A(self.ax_d.plot([], [], color=P["delta"], lw=1.2, alpha=0.9, label="δ pie")[0])
        self.ax_f.axhline(6.0, ls=":", c=P["limit"], lw=0.9)
        _style_axis(self.ax_f, "Contacto · GRF + sensor de pie", "F_N [N]"); self.ax_f.set_ylim(0, 650)
        self.ax_d.set_ylabel("δ [mm]", color=P["delta"]); self.ax_d.set_ylim(0, 26)
        self.ax_d.tick_params(colors=P["delta"], length=0, labelsize=8); self.ax_d.spines["right"].set_color(P["delta"])
        for s in ("top", "left"):
            self.ax_d.spines[s].set_visible(False)
        self.ax_d.grid(False)
        self.l_th = A(self.ax_t.plot([], [], color=P["tau_h"], lw=1.4, label="τ cadera")[0])
        self.l_tk = A(self.ax_t.plot([], [], color=P["tau_k"], lw=1.4, label="τ rodilla")[0])
        self.ax_t.axhspan(3.73, 4.3, color=P["grf"], alpha=0.06); self.ax_t.axhspan(-4.3, -3.73, color=P["grf"], alpha=0.06)
        for s in (3.73, -3.73):
            self.ax_t.axhline(s, ls="--", c=P["limit"], lw=0.9)
        _style_axis(self.ax_t, "Torques · saturación ±3.73 N·m", "τ [N·m]"); self.ax_t.set_ylim(-4.3, 4.3); _legend(self.ax_t, loc="upper right", ncol=2)
        self.l_vh = A(self.ax_v.plot([], [], color=P["v_h"], lw=1.4, label="V cadera")[0])
        self.l_vk = A(self.ax_v.plot([], [], color=P["v_k"], lw=1.4, label="V rodilla")[0])
        for s in (12, -12):
            self.ax_v.axhline(s, ls="--", c=P["limit"], lw=0.9)
        _style_axis(self.ax_v, "Voltaje del motor · límite ±12 V", "V [V]"); self.ax_v.set_ylim(-13.5, 13.5)
        self.ax_v.set_xlabel("t − ahora [s]"); _legend(self.ax_v, loc="upper right", ncol=2)

        self.ax_e1 = fig.add_subplot(gs[0, 2:4]); self.ax_e2 = fig.add_subplot(gs[1, 2:4]); self.ax_e3 = fig.add_subplot(gs[2:4, 2:4])
        self.l_e1t = A(self.ax_e1.plot([], [], color=P["true"], lw=2.2, alpha=0.7, label="real (qvel)")[0])
        self.l_e1e = A(self.ax_e1.plot([], [], color=P["est"], lw=1.3, label="encoder (est)")[0])
        _style_axis(self.ax_e1, "Encoder · velocidad de cadera (Fase 5.1)", "dq/dt [rad/s]"); self.ax_e1.set_ylim(-25, 25); _legend(self.ax_e1, loc="upper right", ncol=2)
        self.l_e2t = A(self.ax_e2.plot([], [], color=P["true"], lw=2.2, alpha=0.7, label="real (qvel)")[0])
        self.l_e2e = A(self.ax_e2.plot([], [], color=P["est"], lw=1.3, label="encoder (est)")[0])
        _style_axis(self.ax_e2, "Encoder · velocidad de rodilla", "dq/dt [rad/s]"); self.ax_e2.set_ylim(-25, 25); _legend(self.ax_e2, loc="upper right", ncol=2)
        self.l_e3h = A(self.ax_e3.plot([], [], color=P["tau_h"], lw=1.2, label="error cadera")[0])
        self.l_e3k = A(self.ax_e3.plot([], [], color=P["tau_k"], lw=1.2, label="error rodilla")[0])
        self.ax_e3.axhline(0, ls="-", c=P["edge"], lw=0.8)
        _style_axis(self.ax_e3, "Error de estimación (encoder − real)", "error [rad/s]"); self.ax_e3.set_ylim(-6, 6)
        self.ax_e3.set_xlabel("t − ahora [s]"); _legend(self.ax_e3, loc="upper right", ncol=2)
        self.rmse_txt = A(self.ax_e3.text(0.015, 0.06, "", transform=self.ax_e3.transAxes, fontsize=9, family="monospace", color=P["muted"], va="bottom"))

        self.ax_s1 = fig.add_subplot(gs[0, 2:4]); self.ax_s2 = fig.add_subplot(gs[1, 2:4]); self.ax_s3 = fig.add_subplot(gs[2:4, 2:4])
        self.l_s_del = A(self.ax_s1.plot([], [], color=P["delta"], lw=1.6, label="δ medido")[0])
        self.ax_s1.axhline(25.4, ls="--", c=P["limit"], lw=0.9, label="stroke 25.4 mm")
        _style_axis(self.ax_s1, "Sensor lineal del pie · desplazamiento δ", "δ [mm]"); self.ax_s1.set_ylim(0, 26); _legend(self.ax_s1, loc="upper right", ncol=2)
        self.l_s_fN = A(self.ax_s2.plot([], [], color=P["grf"], lw=1.5, label="F_N")[0])
        self.ax_s2.axhline(6.0, ls="--", c=P["stance"], lw=1.0, label="touchdown 6 N")
        self.ax_s2.axhline(2.0, ls=":", c=P["warn"], lw=1.0, label="lift-off 2 N")
        _style_axis(self.ax_s2, "Fuerza normal + umbrales (histéresis)", "F_N [N]"); self.ax_s2.set_ylim(0, 650); _legend(self.ax_s2, loc="upper right", ncol=3)
        self.l_s_sw = A(self.ax_s3.step([], [], where="post", color=P["delta"], lw=1.8, label="switch contacto")[0])
        self.l_s_st = A(self.ax_s3.step([], [], where="post", color=P["accent"], lw=1.2, alpha=0.8, label="FSM STANCE")[0])
        _style_axis(self.ax_s3, "Switch de contacto → estado FSM", "on / off"); self.ax_s3.set_ylim(-0.15, 1.25); self.ax_s3.set_yticks([0, 1])
        self.ax_s3.set_xlabel("t − ahora [s]"); _legend(self.ax_s3, loc="upper right", ncol=2)

        self.ax_lc = fig.add_subplot(gs[0:2, 2:4]); self.ax_pw = fig.add_subplot(gs[2:4, 2:4])
        self.l_lc = A(self.ax_lc.plot([], [], color=P["foot"], lw=0.9, alpha=0.85)[0])
        self.d_lc = A(self.ax_lc.plot([], [], "o", ms=6, color=P["grf"])[0])
        _style_axis(self.ax_lc, "Ciclo límite del pie  (z vs ż)", "ż pie [m/s]"); self.ax_lc.set_xlabel("z pie [m]")
        self.ax_lc.set_xlim(-0.02, 0.30); self.ax_lc.set_ylim(-3, 3)
        self.l_pm = A(self.ax_pw.plot([], [], color=P["pmech"], lw=1.4, label="P mecánica  τ·ω")[0])
        self.l_pe = A(self.ax_pw.plot([], [], color=P["pelec"], lw=1.4, label="P eléctrica  V·i")[0])
        self.ax_pw.axhline(0, ls="-", c=P["edge"], lw=0.8)
        _style_axis(self.ax_pw, "Potencia + eficiencia", "P [W]"); self.ax_pw.set_ylim(-500, 500); self.ax_pw.set_xlabel("t − ahora [s]")
        _legend(self.ax_pw, loc="upper right", ncol=2)
        self.eff_txt = A(self.ax_pw.text(0.015, 0.06, "", transform=self.ax_pw.transAxes, fontsize=9, family="monospace", color=P["muted"], va="bottom"))

        for ax in (self.ax_h, self.ax_f, self.ax_t, self.ax_v, self.ax_e1, self.ax_e2,
                   self.ax_e3, self.ax_s1, self.ax_s2, self.ax_s3, self.ax_pw):
            ax.set_xlim(-window_s, 0)

        # dynamic line artists drawn per page (axis -> [stance first, then these])
        self.page_lines = {
            "telem":   [(self.ax_h, [self.l_footz, self.l_hipz, self.d_footz]),
                        (self.ax_f, [self.l_fN, self.d_fN]), (self.ax_d, [self.l_del]),
                        (self.ax_t, [self.l_th, self.l_tk]), (self.ax_v, [self.l_vh, self.l_vk])],
            "encoder": [(self.ax_e1, [self.l_e1t, self.l_e1e]), (self.ax_e2, [self.l_e2t, self.l_e2e]),
                        (self.ax_e3, [self.l_e3h, self.l_e3k, self.rmse_txt])],
            "sensor":  [(self.ax_s1, [self.l_s_del]), (self.ax_s2, [self.l_s_fN]),
                        (self.ax_s3, [self.l_s_sw, self.l_s_st])],
            "energia": [(self.ax_lc, [self.l_lc, self.d_lc]), (self.ax_pw, [self.l_pm, self.l_pe, self.eff_txt])],
        }
        self.pages = {
            "telem":   [self.ax_h, self.ax_f, self.ax_d, self.ax_t, self.ax_v],
            "encoder": [self.ax_e1, self.ax_e2, self.ax_e3],
            "sensor":  [self.ax_s1, self.ax_s2, self.ax_s3],
            "energia": [self.ax_lc, self.ax_pw],
        }
        self.time_axes = {
            "telem":   [self.ax_h, self.ax_f, self.ax_t, self.ax_v],
            "encoder": [self.ax_e1, self.ax_e2, self.ax_e3],
            "sensor":  [self.ax_s1, self.ax_s2, self.ax_s3],
            "energia": [self.ax_pw],
        }
        # one stance-shade collection per time axis (updated in place)
        self.stance = {}
        for axes in self.time_axes.values():
            for ax in axes:
                pc = PolyCollection([], facecolors=P["stance"], alpha=0.13, linewidths=0, zorder=0.5)
                pc.set_animated(blit); ax.add_collection(pc); self.stance[ax] = pc

        # buttons
        def _mk(rect, label, cb):
            from matplotlib.widgets import Button
            axb = self.fig.add_axes(rect)
            b = Button(axb, label, color=P["card"], hovercolor=P["edge"])
            b.label.set_color(P["fg"]); b.label.set_fontsize(9.0)
            for s in axb.spines.values():
                s.set_color(P["edge"])
            b.on_clicked(cb); return b
        y, h = 0.045, 0.052
        self._view_btns = {
            "telem":   _mk([0.035, y, 0.083, h], "Telemetría", lambda e: self.select_page("telem")),
            "encoder": _mk([0.123, y, 0.083, h], "Encoder", lambda e: self.select_page("encoder")),
            "sensor":  _mk([0.211, y, 0.083, h], "Sensor pie", lambda e: self.select_page("sensor")),
            "energia": _mk([0.299, y, 0.083, h], "Fase/Energía", lambda e: self.select_page("energia")),
        }
        self.b_pause = _mk([0.560, y, 0.088, h], "Pausa", self._cb_pause)
        self.b_reset = _mk([0.653, y, 0.075, h], "Reset", self._cb_reset)
        self.b_mode = _mk([0.733, y, 0.150, h], "Modo: Avance", self._cb_mode)
        self.b_exit = _mk([0.888, y, 0.077, h], "Salir", self._cb_exit)
        self.b_exit.color = P["grf"]; self.b_exit.ax.set_facecolor(P["grf"])
        self.b_exit.label.set_color("#0d1117"); self.b_exit.label.set_fontweight("bold")
        self._lbl_v = fig.text(0.035, 0.108, "VISTAS", fontsize=8, color=P["muted"], fontweight="bold")
        self._lbl_c = fig.text(0.560, 0.108, "CONTROLES", fontsize=8, color=P["muted"], fontweight="bold")

        self.foot_l = A(fig.text(0.035, 0.013, "● EN VIVO", fontsize=9, color=P["stance"], fontweight="bold", va="center"))
        self.foot_r = A(fig.text(0.965, 0.013, "", fontsize=9, color=P["muted"], ha="right", va="center", family="monospace"))

        # ---- click-to-zoom: click a plot to enlarge it; ✕ to restore ----
        self.ax_lines = {}
        for plines in self.page_lines.values():
            for ax, arts in plines:
                self.ax_lines[ax] = arts
        self.zoomable = {pg: list(self.time_axes[pg]) for pg in self.time_axes}
        self.zoomable["energia"] = [self.ax_lc, self.ax_pw]
        self.zoom = None; self._zsave = {}
        axc = fig.add_axes([0.952, 0.905, 0.028, 0.045])
        self.btn_close = Button(axc, "✕", color=P["grf"], hovercolor="#ff8a8a")
        self.btn_close.label.set_color("#0d1117"); self.btn_close.label.set_fontsize(13); self.btn_close.label.set_fontweight("bold")
        axc.set_visible(False)
        self.btn_close.on_clicked(lambda e: self._exit_zoom())
        fig.canvas.mpl_connect("button_press_event", self._on_click)

        if blit:
            fig.canvas.mpl_connect("resize_event", lambda e: setattr(self, "bg", None))
        self.select_page("telem")

    # --- callbacks ---
    def select_page(self, name):
        self.page = name
        for g, axes in self.pages.items():
            for ax in axes:
                ax.set_visible(g == name)
        for k, b in self._view_btns.items():
            active = (k == name)
            b.color = P["accent"] if active else P["card"]
            b.ax.set_facecolor(P["accent"] if active else P["card"])
            b.label.set_color("#0d1117" if active else P["fg"]); b.label.set_fontweight("bold" if active else "normal")
        self.bg = None                     # static layout changed -> recache

    def _cb_pause(self, _):
        self.paused = not self.paused
        self.b_pause.label.set_text("Reanudar" if self.paused else "Pausa")
        col = P["warn"] if self.paused else P["card"]
        self.b_pause.color = col; self.b_pause.ax.set_facecolor(col)
        self.b_pause.label.set_color("#0d1117" if self.paused else P["fg"]); self.bg = None

    def _cb_reset(self, _):
        self._reset = True

    def _cb_exit(self, _):
        self._stop = True

    def _cb_mode(self, _):
        self.mode = "sitio" if self.mode == "avance" else "avance"
        if self.mode == "sitio":
            self.cfg.v_des = 0.0; self.m.dof_damping[self.yaw_dof] = 18.0; self.b_mode.label.set_text("Modo: En sitio")
        else:
            self.cfg.v_des = 2.0; self.m.dof_damping[self.yaw_dof] = 0.2; self.b_mode.label.set_text("Modo: Avance")
        self.bg = None

    def consume_reset(self):
        r = self._reset; self._reset = False
        return r

    # --- click-to-zoom one plot to (near) full size, ✕ to restore ---
    def _chrome(self):
        a = [self.ax_kpi, self.ax_im, self.ax_txt, self.foot_l, self.foot_r, self._lbl_v, self._lbl_c]
        a += [g.ax for g in self.gauges]
        a += [b.ax for b in self._view_btns.values()] + [self.b_pause.ax, self.b_reset.ax, self.b_mode.ax, self.b_exit.ax]
        return a

    def _relayout(self):
        if self.blit:
            self.bg = None
        else:
            self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is None or event.inaxes is self.btn_close.ax or self.zoom is not None:
            return
        ax = self.ax_f if event.inaxes is self.ax_d else event.inaxes
        if ax in self.zoomable.get(self.page, []):
            self._enter_zoom(ax)

    def _enter_zoom(self, ax):
        BIG = (0.055, 0.085, 0.89, 0.805)
        self.zoom = ax
        # store original geometry as a value tuple, BEFORE resizing. The δ axis
        # (ax_d) is a twin of ax_f and follows its position AUTOMATICALLY, so we
        # only move/restore ax_f — touching ax_d too makes the twin-sync fight
        # and leaves the plot stuck enlarged.
        self._zsave = {ax: ax.get_position().bounds}
        ax.set_position(BIG)
        self.ax_d.set_visible(ax is self.ax_f)      # δ axis only for the GRF panel
        for a in self._chrome():
            a.set_visible(False)
        for other in self.zoomable[self.page]:
            other.set_visible(other is ax)
        ax.set_visible(True); self.btn_close.ax.set_visible(True)
        self._relayout()

    def _exit_zoom(self):
        if self.zoom is None:
            return
        for a, pos in self._zsave.items():
            a.set_position(pos)
        self._zsave = {}; self.zoom = None
        self.btn_close.ax.set_visible(False)
        for a in self._chrome():
            a.set_visible(True)
        self.select_page(self.page)        # restores this page's plot visibility
        self._relayout()

    def _update_stance(self, tr, st, ax):
        pc = self.stance[ax]
        y0, y1 = ax.get_ylim(); verts = []
        if len(tr) >= 2:
            edges = np.flatnonzero(np.diff(np.r_[0, st, 0]))
            for a, b in zip(edges[0::2], edges[1::2]):
                ta = tr[max(a - 1, 0)]; tb = tr[min(b, len(tr) - 1)]
                verts.append([(ta, y0), (ta, y1), (tb, y1), (tb, y0)])
        pc.set_verts(verts)

    def update(self, frame_rgb, tel, hud):
        """Set artist data only (no drawing). Drawing is blit_draw / full draw."""
        self.im.set_data(frame_rgb)
        t = tel.arr("t"); t1 = t[-1] if len(t) else 0.0; tr = t - t1
        if hud["paused"]:
            self.badge.set_facecolor(P["warn"]); self.badge_txt.set_text("PAUSA")
        else:
            stance = hud["state"] == "STANCE"
            self.badge.set_facecolor(P["stance"] if stance else P["flight"]); self.badge_txt.set_text("STANCE" if stance else "FLIGHT")
        self.m1.set_text(hud["m1"]); self.m2.set_text(hud["m2"])
        self.gauges[0].set(hud["g_tau"]); self.gauges[1].set(hud["g_v"]); self.gauges[2].set(hud["g_i"])
        self.sat_lamp.set_text("● SAT" if hud["g_tau"] >= 0.99 else "")
        self._update_kpis(tel, hud)
        self.foot_l.set_text("● GRABANDO" if hud["rec"] else "● EN VIVO")
        self.foot_l.set_color(P["grf"] if hud["rec"] else P["stance"])
        self.foot_r.set_text(f"RT {hud['rt']:.2f}x   ·   {hud['fps']:.0f} FPS   ·   solver RK4/Newton  dt=1ms")

        st = tel.arr("stance")
        for ax in self.time_axes[self.page]:
            self._update_stance(tr, st, ax)
        pg = self.page
        if pg == "telem":
            self.l_footz.set_data(tr, tel.arr("foot_z")); self.l_hipz.set_data(tr, tel.arr("hip_z"))
            self.l_fN.set_data(tr, tel.arr("fN")); self.l_del.set_data(tr, tel.arr("delta"))
            self.l_th.set_data(tr, tel.arr("tau_h")); self.l_tk.set_data(tr, tel.arr("tau_k"))
            self.l_vh.set_data(tr, tel.arr("V_h")); self.l_vk.set_data(tr, tel.arr("V_k"))
            if len(t):
                self.d_footz.set_data([0], [tel.arr("foot_z")[-1]]); self.d_fN.set_data([0], [tel.arr("fN")[-1]])
        elif pg == "encoder":
            self.l_e1t.set_data(tr, tel.arr("vtr_h")); self.l_e1e.set_data(tr, tel.arr("vte_h"))
            self.l_e2t.set_data(tr, tel.arr("vtr_k")); self.l_e2e.set_data(tr, tel.arr("vte_k"))
            if len(t):
                eh = tel.arr("vte_h") - tel.arr("vtr_h"); ek = tel.arr("vte_k") - tel.arr("vtr_k")
                self.l_e3h.set_data(tr, eh); self.l_e3k.set_data(tr, ek)
                self.rmse_txt.set_text(f"RMSE cadera {np.sqrt(np.mean(eh**2)):.3f}   rodilla {np.sqrt(np.mean(ek**2)):.3f}  rad/s")
        elif pg == "sensor":
            self.l_s_del.set_data(tr, tel.arr("delta")); self.l_s_fN.set_data(tr, tel.arr("fN"))
            self.l_s_sw.set_data(tr, tel.arr("contact")); self.l_s_st.set_data(tr, tel.arr("stance"))
        elif pg == "energia":
            self.l_lc.set_data(tel.arr("foot_z"), tel.arr("foot_vz"))
            self.l_pm.set_data(tr, tel.arr("p_mech")); self.l_pe.set_data(tr, tel.arr("p_elec"))
            if len(t):
                self.d_lc.set_data([tel.arr("foot_z")[-1]], [tel.arr("foot_vz")[-1]])
                pm = tel.arr("p_mech"); pe = tel.arr("p_elec")
                ein = float(np.mean(np.abs(pe))); eout = float(np.mean(np.clip(pm, 0, None)))
                eff = 100.0 * eout / ein if ein > 1e-6 else 0.0
                self.eff_txt.set_text(f"P̄_mec {np.mean(pm):+.1f} W   P̄_elec {ein:.1f} W   eficiencia ≈ {eff:.0f}%")

    def _update_kpis(self, tel, hud):
        t = tel.arr("t")
        if len(t) < 2:
            return
        vfwd = float(np.mean(np.abs(tel.arr("v_fwd")[-40:])))
        pe = float(np.mean(np.abs(tel.arr("p_elec"))))
        cot = pe / (self.m_total * G * vfwd) if vfwd > 0.05 else None
        vals = [f"{hud['hop_freq']:.2f}", f"{tel.arr('foot_z').max()*100:.0f}", f"{tel.arr('stance').mean()*100:.0f}",
                f"{vfwd:.2f}", f"{np.abs(tel.arr('p_mech')).max():.0f}", (f"{cot:.1f}" if cot is not None else "—")]
        for tx, v in zip(self.kpi_val, vals):
            tx.set_text(v)

    def capture_bg(self):
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def blit_draw(self):
        c = self.fig.canvas
        if self.bg is None:
            self.capture_bg()
        c.restore_region(self.bg)
        if self.zoom is not None:                  # zoomed: only the big plot
            ax = self.zoom
            if ax in self.stance:
                ax.draw_artist(self.stance[ax])
            for a in self.ax_lines.get(ax, []):
                ax.draw_artist(a)
            if ax is self.ax_f:
                for a in self.ax_lines.get(self.ax_d, []):
                    self.ax_d.draw_artist(a)
            c.blit(self.fig.bbox)
            return
        self.ax_im.draw_artist(self.im)
        for a in (self.badge, self.badge_txt, self.m1, self.m2, self.sat_lamp):
            self.ax_txt.draw_artist(a)
        for gg in self.gauges:
            gg.draw()
        for v in self.kpi_val:
            self.ax_kpi.draw_artist(v)
        self.fig.draw_artist(self.foot_l); self.fig.draw_artist(self.foot_r)
        for ax in self.time_axes[self.page]:
            ax.draw_artist(self.stance[ax])
        for ax, arts in self.page_lines[self.page]:
            for a in arts:
                ax.draw_artist(a)
        c.blit(self.fig.bbox)


def build_cam(m):
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = (0.0, 0.0, 0.12)
    cam.azimuth, cam.elevation, cam.distance = 90, -20, 2.7
    return cam


def run(cfg, live=True, duration=None, record=None, fps=30, window_s=5.0, page="telem", use_blit=True):
    import matplotlib
    matplotlib.use("TkAgg" if live else "Agg", force=True)
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "figure.facecolor": P["bg"], "savefig.facecolor": P["bg"], "axes.facecolor": P["panel"],
        "text.color": P["fg"], "axes.labelcolor": P["muted"], "font.size": 9, "font.family": "sans-serif",
    })

    m = mujoco.MjModel.from_xml_path(SCENE); m.opt.timestep = cfg.dt
    d = mujoco.MjData(m); ctl = HoppyController(m, cfg)
    ren = mujoco.Renderer(m, 560, 680); cam = build_cam(m)
    scn_opt = mujoco.MjvOption(); m_total = float(m.body_mass[1:].sum())

    sample_every = max(1, int(round(0.004 / cfg.dt)))
    tel = Telemetry(window_s, cfg.dt * sample_every)
    n_sub = max(1, int(round((1.0 / fps) / cfg.dt)))

    blit = live and use_blit
    fig = plt.figure(figsize=(15.5, 9.0))
    dash = Dashboard(fig, window_s, m, cfg, m_total, blit=blit)
    if page != "telem":
        dash.select_page(page)
    writer = None
    if record:
        import imageio
        writer = imageio.get_writer(record, fps=fps, macro_block_size=None)
    if live:
        try:
            fig.canvas.manager.set_window_title("HOPPY · Dashboard")
        except Exception:
            pass
        # show + let the GUI realize/draw the window BEFORE capturing the blit
        # background (capturing too early grabs an unready canvas -> blit blank).
        plt.show(block=False); plt.pause(0.3)

    t = 0.0; step = 0; n_hops = 0; peak_tau = 0.0
    td_times = deque(maxlen=6); fps_ema = float(fps); last_wall = time.perf_counter()
    t_end = duration if duration else (1e9 if live else 8.0)
    try:
        while t < t_end:
            if live and (dash._stop or not plt.fignum_exists(fig.number)):
                break
            if dash.consume_reset():
                mujoco.mj_resetData(m, d); ctl = HoppyController(m, cfg)
                tel.clear(); t = 0.0; step = 0; n_hops = 0; peak_tau = 0.0; td_times.clear()
            wall0 = time.perf_counter()
            if not dash.paused:
                for _ in range(n_sub):
                    tau = ctl.apply(d, t); mujoco.mj_step(m, d); t += cfg.dt; step += 1
                    L = ctl.last
                    if L.get("transition") == ("FLIGHT", "STANCE"):
                        n_hops += 1; td_times.append(t)
                    peak_tau = max(peak_tau, float(np.abs(tau).max()))
                    if step % sample_every == 0:
                        tel.push(t, L, ctl, d)

            ren.update_scene(d, cam, scn_opt); _overlay_contact(ren.scene, d, ctl)
            frame = ren.render()
            L = ctl.last
            yaw_deg = np.degrees(d.qpos[ctl.qadr["yaw"]])
            hop_freq = (1.0 / np.median(np.diff(td_times))) if len(td_times) >= 2 else 0.0
            now = time.perf_counter(); fdt = max(now - last_wall, 1e-4); last_wall = now
            fps_ema = 0.85 * fps_ema + 0.15 * (1.0 / fdt)
            rt = (n_sub * cfg.dt) / fdt if not dash.paused else 0.0
            tag = "PAUSA" if dash.paused else L["state"]
            hud = dict(
                state=L["state"], paused=dash.paused, hop_freq=hop_freq, rec=bool(record), fps=fps_ema, rt=rt,
                g_tau=float(np.abs(L["tau"]).max()) / cfg.tau_max, g_v=float(np.abs(L["V"]).max()) / cfg.V_max,
                g_i=float(np.abs(L["i"]).max()) / cfg.I_max,
                m1=(f"t {t:4.1f}s   F_N {L['fN']:5.1f} N   δ {L['foot_delta']*1000:4.1f} mm"),
                m2=(f"saltos {n_hops:2d}   |τ|max {peak_tau:.2f}   yaw {yaw_deg:+.0f}°"))
            dash.update(frame, tel, hud)

            if writer is not None:
                fig.canvas.draw()
                rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                hh, ww = rgb.shape[:2]
                writer.append_data(np.ascontiguousarray(rgb[:hh - hh % 2, :ww - ww % 2]))
            if live:
                if blit:
                    dash.blit_draw()
                else:
                    fig.canvas.draw_idle()
                fig.canvas.flush_events()
                dt_wall = time.perf_counter() - wall0
                if dt_wall < n_sub * cfg.dt:
                    time.sleep(n_sub * cfg.dt - dt_wall)
    finally:
        ren.close()
        if writer is not None:
            writer.close(); print(f"  dashboard -> {record}")
        if live:
            plt.ioff()
            try:
                plt.close(fig)
            except Exception:
                pass
            if dash._stop:
                print("  simulación terminada (botón Salir)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=str, help="grabar a mp4 (modo headless)")
    ap.add_argument("--duration", type=float, help="duración [s] (record)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--window", type=float, default=5.0, help="ventana rodante [s]")
    ap.add_argument("--page", choices=("telem", "encoder", "sensor", "energia"), default="telem")
    ap.add_argument("--no-blit", action="store_true", help="dibujo completo (sin blitting)")
    ap.add_argument("--v_des", type=float)
    ap.add_argument("--f_peak", type=float)
    args = ap.parse_args()

    cfg = HopConfig()
    if args.v_des is not None: cfg.v_des = args.v_des
    if args.f_peak is not None: cfg.f_peak = args.f_peak

    run(cfg, live=not args.record, duration=args.duration, record=args.record,
        fps=args.fps, window_s=args.window, page=args.page, use_blit=not args.no_blit)


if __name__ == "__main__":
    main()
