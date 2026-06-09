#!/usr/bin/env python3
"""HOPPY hybrid hopping controller  (rubrica Fase 4 + Fase 5).

Two cooperating objects:

  EncoderEstimator  -- Fase 5.1: emulates real encoders. Quantizes joint
                       position, then estimates joint velocity by filtered
                       numerical differentiation. The main controller NEVER
                       reads data.qvel; it uses these estimates.

  HoppyController   -- Fase 4: FLIGHT/STANCE state machine with
                       * FLIGHT: Cartesian PD on the foot via transposed
                                 contact Jacobian   tau = -J^T(Kp(p-pd)) - Kd q.
                       * STANCE: desired ground-reaction-force control
                                 tau =  J^T F_foot   (F_foot pushes DOWN on the
                                 ground -> body is thrust UP; sign verified in
                                 sim). A half-sine vertical force profile injects
                                 the hop energy.
                       The two laws are blended over t_blend at each transition
                       (smooth transitions, rubrica Fase 4).

The leg swings in the boom's sagittal plane (model axes parallel to pitch), so
HOPPY hops in place; the horizontal-force / Raibert terms are kept for Fase 4.4
velocity control and perturbation rejection.

Actuated joints: hip, knee. Gantry yaw/pitch are passive. Only the hip/knee
columns of the foot Jacobian are used. The desired torque is realized through a
VOLTAGE actuator model (back-EMF + 12 V / 30 A limits) and then clipped at
tau_max, which together implement the physical torque saturation (Fase 2.1).
"""
from dataclasses import dataclass, field
import numpy as np
import mujoco


# -----------------------------------------------------------------------------
# Fase 5.1 : encoder emulation + filtered velocity estimation
# -----------------------------------------------------------------------------
class EncoderEstimator:
    """Quantize qpos like a real encoder and estimate qvel by filtered finite
    differences (first-order low-pass).  No access to the true data.qvel."""

    # goBILDA 5202 (26.9:1): 751.8 PPR at output * 4 (quadrature) = 3012.8 cpr
    def __init__(self, n, dt, resolution_rad=2 * np.pi / 3012.8, cutoff_hz=80.0):
        self.dt = dt
        self.res = resolution_rad                    # joint-side encoder step
        # first-order low-pass coefficient: y += a*(x-y), a = 1-exp(-2*pi*fc*dt)
        self.a = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt)
        self.q_prev = None
        self.v_filt = np.zeros(n)

    def quantize(self, q):
        return np.round(q / self.res) * self.res

    def update(self, q_true):
        """Return (q_measured, v_estimated)."""
        q = self.quantize(q_true)
        if self.q_prev is None:
            self.q_prev = q
        v_raw = (q - self.q_prev) / self.dt          # backward difference
        self.v_filt += self.a * (v_raw - self.v_filt)  # discrete low-pass
        self.q_prev = q
        return q, self.v_filt.copy()


# -----------------------------------------------------------------------------
# HOPPY foot contact sensor : the Linear_Disp_Sensor_404R1KL1.0 on the shank
# -----------------------------------------------------------------------------
class FootContactSensor:
    """Emulates HOPPY's foot sensor (CAD: Linear_Disp_Sensor_404R1KL1.0, a 1-inch
    / 25.4 mm linear potentiometer on the shank). The physical sensor reads the
    foot's compliance displacement delta as it loads the ground; HOPPY's micro-
    controller thresholds it into the BINARY foot-contact signal it uses for
    touchdown / lift-off (HOPPY Technical Guideline, sec. VIII).

    Our foot is rigid, so delta is modeled from the measured contact force via a
    foot stiffness  delta = F_N / k_foot  (saturating at the 25.4 mm stroke),
    quantized like the pot's ADC. The binary switch adds hysteresis (on at
    f_on, off at f_off) exactly like a Schmitt-triggered contact switch."""

    def __init__(self, f_on, f_off, k_foot=20000.0, stroke=0.0254, adc_bits=10,
                 debounce_steps=6):
        self.k_foot = k_foot                  # N/m, modeled foot compliance
        self.stroke = stroke                  # m, sensor full range (1 inch)
        self.res = stroke / (2 ** adc_bits)   # m, pot ADC quantization step
        self.f_on = f_on                      # N, contact-switch ON threshold
        self.f_off = f_off                    # N, contact-switch OFF threshold
        self.debounce = debounce_steps        # samples a new state must persist
        self.contact = False                  # latched, debounced binary state
        self._pending = 0

    def read(self, f_normal):
        """Return (delta_quantized [m], contact [bool])."""
        f = max(0.0, f_normal)
        delta = min(f / self.k_foot, self.stroke)        # analog displacement
        delta_q = round(delta / self.res) * self.res     # ADC quantization
        # hysteresis (Schmitt) candidate, then debounce against impact ringing
        if not self.contact and f > self.f_on:
            cand = True
        elif self.contact and f < self.f_off:
            cand = False
        else:
            cand = self.contact
        if cand != self.contact:
            self._pending += 1
            if self._pending >= self.debounce:
                self.contact = cand
                self._pending = 0
        else:
            self._pending = 0
        return delta_q, self.contact


# -----------------------------------------------------------------------------
# Controller configuration (all physically-meaningful, tunable parameters)
# -----------------------------------------------------------------------------
@dataclass
class HopConfig:
    # -- loop / actuator --
    dt: float = 0.001                 # 1 kHz (Fase 4.1)
    tau_max: float = 3.73             # explicit torque-saturation clip (Fase 2.1)
    gravity_comp: bool = True         # add leg gravity feedforward

    # -- motor electrical model (HOPPY get_params.m): the controller emits a
    #    VOLTAGE, the motor realizes current with back-EMF, both saturated.
    #    This makes back-EMF + the V/I limits explicit (and IS the physical
    #    torque saturation, refined into a torque-speed curve). --
    use_motor_model: bool = True
    Rw: float = 1.3                   # ohm, coil resistance
    kT: float = 0.0135                # N*m/A, motor torque constant
    kv: float = 0.0186                # V*s/rad, back-EMF constant
    V_max: float = 12.0               # V, supply limit
    I_max: float = 30.0               # A, driver current limit
    N_gear: np.ndarray = field(default_factory=lambda: np.array([26.9, 28.8]))  # hip, knee

    # -- contact detection (Fase 3.2) with hysteresis --
    f_touchdown: float = 6.0          # N, normal force to declare STANCE
    f_liftoff: float = 2.0            # N, normal force below which -> FLIGHT

    # -- smooth FLIGHT<->STANCE transition (rubrica Fase 4: transiciones suaves) --
    t_blend: float = 0.012            # s, linear blend of the two control laws

    # -- FLIGHT: Cartesian PD on the foot, expressed in the BOOM (Link2) frame --
    # so the target is invariant to the passive yaw/pitch coupling.
    kp_cart: float = 1200.0           # N/m, foot position stiffness
    kd_joint: float = 1.0             # N*m*s/rad, joint-space damping
    # desired foot position (foot - hip, in Link2 frame) = a BENT "crouch" on
    # the FAITHFUL URDF leg (hip~0.70, knee~0.35), |Jz|~0.27 for good vertical
    # power transmission. The leg lands here and the stance thrust extends it.
    r_des_L2: np.ndarray = field(default_factory=lambda: np.array([-0.037, -0.139, 0.243]))

    # -- STANCE: desired GRF (half-sine vertical thrust + speed term) --
    # f_peak commands above the per-config force ceiling so the leg saturates
    # at tau_max (3.73 N*m); the impulse is then set by t_push.
    f_peak: float = 120.0            # N, peak DOWNWARD foot force (thrust up);
                                     # higher hop, kept stable by the yaw brake
    t_push: float = 0.24             # s, thrust pulse duration
    v_des: float = 2.0              # m/s desired forward (yaw-tangent) speed.
                                     # >0 drives HOPPY forward around the boom
                                     # (continuous yaw); the achievable cruise is
                                     # ~0.45 m/s (~1 lap / 9 s). Set 0 to idle.
    k_place: float = 0.06            # m per (m/s): Raibert flight foot placement
    place_max: float = 0.05          # m, cap on the Raibert foot shift
    kv_fwd: float = -20.0           # N per (m/s): stance tangent speed trim
    f_tan_max: float = 20.0          # N, CAP on stance tangent force (forward
                                     # drive); bounded so it can't starve thrust
    # -- yaw-angle hold: with the re-planarized leg (axes parallel to pitch),
    #    the leg swings in the boom's sagittal plane and no longer pumps yaw, so
    #    the robot hops in place with k_yaw = 0. The outer loop is kept (it folds
    #    a heading error into the speed reference, bounded by f_tan_max) as a
    #    safety net for perturbations; raise it only if a residual drift appears.
    k_yaw: float = 0.0              # (m/s) per rad of yaw error


# -----------------------------------------------------------------------------
# Fase 4 : hybrid FLIGHT / STANCE controller
# -----------------------------------------------------------------------------
class HoppyController:
    FLIGHT, STANCE = "FLIGHT", "STANCE"

    def __init__(self, model, cfg: HopConfig = None):
        self.m = model
        self.cfg = cfg or HopConfig()
        g = lambda t, n: mujoco.mj_name2id(model, t, n)
        self.jid = {n: g(mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("yaw", "pitch", "hip", "knee")}
        self.vadr = {n: model.jnt_dofadr[self.jid[n]] for n in self.jid}
        self.qadr = {n: model.jnt_qposadr[self.jid[n]] for n in self.jid}
        self.hip_col = self.vadr["hip"]
        self.knee_col = self.vadr["knee"]
        self.foot_sid = g(mujoco.mjtObj.mjOBJ_SITE, "foot")
        self.foot_gid = g(mujoco.mjtObj.mjOBJ_GEOM, "foot")
        self.hip_bid = g(mujoco.mjtObj.mjOBJ_BODY, "Link3")   # hip anchor = Link3 origin
        self.boom_bid = g(mujoco.mjtObj.mjOBJ_BODY, "Link2")  # boom frame for r_des
        self.act = {n: g(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("hip", "knee")}

        self.enc = EncoderEstimator(model.nv, self.cfg.dt)
        # HOPPY foot contact sensor (linear-pot -> binary switch). Its switch
        # thresholds ARE the FSM's force thresholds, so behaviour is unchanged.
        self.foot_sensor = FootContactSensor(self.cfg.f_touchdown, self.cfg.f_liftoff)
        self.state = self.FLIGHT
        self.t_touchdown = None
        self.t_liftoff = None
        self.t_trans = -1.0           # time of last state change (for blending)
        self.yaw_ref = None           # heading to hold (captured on first call)
        self._Jp = np.zeros((3, model.nv))
        self._Jr = np.zeros((3, model.nv))

        # for logging / introspection
        self.last = {}

    # --- contact / GRF sensing (Fase 3.2) ---
    def foot_normal_force(self, data):
        f6 = np.zeros(6)
        tot = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            if self.foot_gid in (con.geom1, con.geom2):
                mujoco.mj_contactForce(self.m, data, c, f6)
                tot += f6[0]          # contact-frame normal (x)
        return tot

    def foot_jacobian_hk(self, data):
        """3x2 foot-position Jacobian w.r.t. (hip, knee)."""
        mujoco.mj_jacSite(self.m, data, self._Jp, self._Jr, self.foot_sid)
        return self._Jp[:, [self.hip_col, self.knee_col]]

    def _grav_comp_hk(self, data):
        """Gravity feedforward on hip/knee from the bias force."""
        if not self.cfg.gravity_comp:
            return np.zeros(2)
        return np.array([data.qfrc_bias[self.hip_col], data.qfrc_bias[self.knee_col]])

    # --- the two control laws, computed every step so they can be blended ---
    def _tau_flight(self, J, p_foot, p_hip, R_L2, qd_hk, tangent, v_err):
        """Cartesian PD on the foot in the BOOM frame (Fase 4.3):
        tau = -J^T (Kp (p - p_des)) - Kd qd. Raibert placement shifts the
        target along the forward tangent to regulate hop speed (Fase 4.4)."""
        cfg = self.cfg
        p_des = p_hip + R_L2 @ cfg.r_des_L2              # crouch posture (world)
        shift = np.clip(cfg.k_place * v_err, -cfg.place_max, cfg.place_max)
        p_des = p_des + shift * tangent
        F = cfg.kp_cart * (p_foot - p_des)
        self._F_flight = -F
        return -J.T @ F - cfg.kd_joint * qd_hk

    def _tau_stance(self, J, t, tangent, v_err):
        """Desired-GRF control (Fase 4.4): half-sine vertical thrust (foot
        pushes DOWN -> body thrust UP) plus a tangent term that trims forward
        speed.  tau = J^T F_foot."""
        cfg = self.cfg
        s = 0.0 if self.t_touchdown is None else np.clip(
            (t - self.t_touchdown) / cfg.t_push, 0.0, 1.0)
        Fz_down = -cfg.f_peak * np.sin(np.pi * s)        # negative z = downward
        F_tan = np.clip(-cfg.kv_fwd * v_err, -cfg.f_tan_max, cfg.f_tan_max)
        F_foot = np.array([0.0, 0.0, Fz_down]) + F_tan * tangent
        self._F_stance = F_foot
        self._Fz_des = -Fz_down                          # upward thrust magnitude
        return J.T @ F_foot

    def _motor_model(self, tau_cmd, qd_est, qd_true):
        """Electrical actuator (HOPPY get_params.m). The CONTROLLER turns the
        desired joint torque into a coil voltage using its ESTIMATED velocity
        (Fase 5: never the true qvel); the physical motor then realizes the
        current from its TRUE back-EMF, with both V and I saturated:
            V = (R/(kT N)) tau + kv N w_est,         |V| <= V_max
            i = (V - kv N w_true) / R,               |i| <= I_max
            tau = kT N i.
        Returns (V, i, tau_actual)."""
        cfg = self.cfg
        if not cfg.use_motor_model:
            return np.zeros(2), np.zeros(2), tau_cmd
        N = cfg.N_gear
        V = (cfg.Rw / (cfg.kT * N)) * tau_cmd + cfg.kv * N * qd_est
        V = np.clip(V, -cfg.V_max, cfg.V_max)            # 12 V supply limit
        i = np.clip((V - cfg.kv * N * qd_true) / cfg.Rw, -cfg.I_max, cfg.I_max)  # 30 A
        return V, i, cfg.kT * N * i

    def control(self, data, t):
        cfg = self.cfg
        # ---- Fase 5: encoder-emulated measurements (no true qvel) ----
        q_meas, v_est = self.enc.update(data.qpos.copy())
        qd_hk = np.array([v_est[self.hip_col], v_est[self.knee_col]])
        # true hip/knee rates feed ONLY the physical motor's back-EMF (the plant,
        # not the controller) -> the controller itself stays qvel-free (Fase 5).
        qd_hk_true = np.array([data.qvel[self.hip_col], data.qvel[self.knee_col]])

        if self.yaw_ref is None:
            self.yaw_ref = data.qpos[self.qadr["yaw"]]

        # ---- Fase 3.2: contact detection -> FSM (committed stance) ----
        # The foot sensor (linear pot) turns the contact force into the BINARY
        # touch switch HOPPY uses (with hysteresis). FLIGHT->STANCE on touchdown;
        # STANCE->FLIGHT only AFTER the thrust pulse and the foot has left.
        fN = self.foot_normal_force(data)
        foot_delta, foot_contact = self.foot_sensor.read(fN)
        prev_state = self.state
        if self.state == self.FLIGHT and foot_contact:
            self.state = self.STANCE
            self.t_touchdown = t
            self.t_trans = t
        elif self.state == self.STANCE:
            thrust_done = (t - self.t_touchdown) > cfg.t_push
            if thrust_done and not foot_contact:
                self.state = self.FLIGHT
                self.t_liftoff = t
                self.t_trans = t

        J = self.foot_jacobian_hk(data)                 # 3x2
        p_foot = data.site_xpos[self.foot_sid].copy()
        p_hip = data.xpos[self.hip_bid].copy()
        R_L2 = data.xmat[self.boom_bid].reshape(3, 3)   # boom orientation

        # Forward (yaw-tangent) kinematics, shared by both states.
        yaw = data.qpos[self.qadr["yaw"]]
        yaw_rate = v_est[self.vadr["yaw"]]
        Rb = max(np.linalg.norm(p_foot[:2]), 1e-3)
        tangent = np.array([-p_foot[1], p_foot[0], 0.0]) / Rb
        v_fwd = yaw_rate * Rb
        # outer yaw-hold loop folds heading error into the speed reference, so
        # accumulated yaw drift is actively driven back to yaw_ref (hop in place).
        v_des_eff = cfg.v_des - cfg.k_yaw * (yaw - self.yaw_ref)
        v_err = v_fwd - v_des_eff

        # both laws every step; blend over t_blend at each transition (smooth)
        tau_flight = self._tau_flight(J, p_foot, p_hip, R_L2, qd_hk, tangent, v_err)
        tau_stance = self._tau_stance(J, t, tangent, v_err)
        if cfg.t_blend > 0:
            ramp = np.clip((t - self.t_trans) / cfg.t_blend, 0.0, 1.0)
        else:
            ramp = 1.0
        w = ramp if self.state == self.STANCE else (1.0 - ramp)  # 1 = stance law
        tau = (1.0 - w) * tau_flight + w * tau_stance
        F_cmd = self._F_stance if w >= 0.5 else self._F_flight

        tau = tau + self._grav_comp_hk(data)
        # ---- actuator: voltage/current limits then explicit torque clip ----
        V, i_coil, tau = self._motor_model(tau, qd_hk, qd_hk_true)
        tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)     # Fase 2.1 saturation

        self.last = dict(state=self.state, fN=fN, q_meas=q_meas, v_est=v_est,
                         p_foot=p_foot, p_hip=p_hip, tau=tau.copy(), F_cmd=F_cmd,
                         V=V.copy(), i=i_coil.copy(), Fz_des=self._Fz_des, blend=w,
                         foot_delta=foot_delta, foot_contact=foot_contact,
                         transition=(prev_state, self.state) if prev_state != self.state else None)
        return tau

    def apply(self, data, t):
        tau = self.control(data, t)
        data.ctrl[self.act["hip"]] = tau[0]
        data.ctrl[self.act["knee"]] = tau[1]
        return tau
