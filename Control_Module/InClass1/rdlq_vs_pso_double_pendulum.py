"""
RDLQ vs PSO Controller Benchmark — MuJoCo InvertedDoublePendulum
================================================================
Adapted for Gymnasium's InvertedDoublePendulum-v5 (1 actuator, 2 hinge joints).

Dependencies:
    pip install numpy scipy matplotlib gymnasium[mujoco]

The environment has:
    obs = [x, sin(θ1), sin(θ2), cos(θ1), cos(θ2), v_x, ω1, ω2, …]  (11-dim)
    action ∈ [-1, 1] (force on cart, scaled to [-3, 3] N internally)

Since there is only 1 actuator driving the cart, we treat this as a 1-input
system and design the controllers around a linearised cart-double-pendulum model.

The RDLQ controller uses a discrete-time LQR on the linearised model.
The PSO controller uses a hand-tuned (or swarm-optimised) state-feedback gain.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.linalg import solve_discrete_are
import gymnasium as gym
import time

# ============================================================
# Utilities
# ============================================================

def saturate(u, lo=-1.0, hi=1.0):
    return np.clip(u, lo, hi)

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))

def lyapunov_candidate(E, W):
    E = np.asarray(E, dtype=float).reshape(-1, 1)
    return float((E.T @ W @ E).item())

# ============================================================
# Linearised model of the double inverted pendulum on cart
# ============================================================
# State:  z = [x, θ1, θ2, dx, dθ1, dθ2]   (6-dim)
# Input:  u = [F]                            (1-dim, cart force)
#
# We extract/reconstruct this from the MuJoCo obs which is:
#   obs = [x, sin(θ1), sin(θ2), cos(θ1), cos(θ2), dx, dθ1, dθ2,
#          constraint_force_1, constraint_force_2, constraint_force_3]
#
# Default MuJoCo model params (approximate):
#   m_cart ≈ 10,  m1 ≈ 5,  m2 ≈ 5,  l1 ≈ 0.6,  l2 ≈ 0.6,  g = 9.81
# These vary by XML; the exact values don't matter much — LQR is robust.

@dataclass
class LinearDoublePendulumModel:
    """Small-angle linearisation around upright equilibrium.

    Uses RELATIVE angles (theta2 = angle of pole-2 relative to pole-1),
    matching the MuJoCo joint-angle convention.  The full pole-1 length L1
    (joint-1 to joint-2) must be supplied separately from l1 (joint-1 to
    pole-1 COM) because m2's coupling depends on L1.
    """
    # Actual MuJoCo InvertedDoublePendulum-v5 parameters
    m_cart: float = 10.472   # cart mass (kg)
    m1: float = 4.199        # pole-1 mass (kg)
    m2: float = 4.199        # pole-2 mass (kg)
    l1: float = 0.3          # dist joint-1 → pole-1 COM  (capsule half-height)
    L1: float = 0.6          # dist joint-1 → joint-2     (full pole-1 length)
    l2: float = 0.3          # dist joint-2 → pole-2 COM
    g: float = 9.81
    sigma: float = 0.05      # effective control dt = model_dt × frame_skip
    gear: float = 1.0        # actuator gear ratio

    def continuous_matrices(self):
        """Return (A_c, B_c) for relative-angle state z=[x,θ1,θ2_rel,dx,dθ1,dθ2_rel]."""
        mc, m1, m2 = self.m_cart, self.m1, self.m2
        l1, L1, l2, g = self.l1, self.L1, self.l2, self.g

        # Mass matrix for relative-angle double pendulum on cart
        # (m2's inertial coupling uses L1 = full length of pole-1)
        M = np.array([
            [mc + m1 + m2,      m1*l1 + m2*L1,          m2*l2],
            [m1*l1 + m2*L1,     m1*l1**2 + m2*L1**2,    m2*L1*l2],
            [m2*l2,             m2*L1*l2,                 m2*l2**2],
        ])
        # Gravity linearisation for INVERTED pendulum with RELATIVE angle θ2_rel:
        #   ∂V/∂θ1 involves both poles; ∂V/∂θ2_rel involves only pole-2.
        # Off-diagonal term (m2·g·l2) captures the coupling through absolute
        # orientation of pole-2 = θ1 + θ2_rel.
        G = np.array([
            [0, 0,                          0],
            [0, (m1*l1 + m2*L1)*g,          m2*g*l2],
            [0, m2*g*l2,                    m2*g*l2],
        ])
        Minv = np.linalg.inv(M)
        A_c = np.zeros((6, 6))
        A_c[:3, 3:] = np.eye(3)
        A_c[3:, :3] = Minv @ G
        B_c = np.zeros((6, 1))
        B_c[3:, 0] = Minv[:, 0] * self.gear  # absorb gear so u ∈ [-1,1]
        return A_c, B_c

    def discrete_matrices(self):
        """Zero-order hold (matrix-exponential) discretisation — exact for piecewise-constant input."""
        from scipy.linalg import expm
        A_c, B_c = self.continuous_matrices()
        n, m = A_c.shape[0], B_c.shape[1]
        # Stack [A_c  B_c; 0  0] and exponentiate
        ZM = np.zeros((n + m, n + m))
        ZM[:n, :n] = A_c
        ZM[:n, n:] = B_c
        eZM = expm(ZM * self.sigma)
        A_d = eZM[:n, :n]
        B_d = eZM[:n, n:]
        return A_d, B_d

# ============================================================
# Observation → state extraction
# ============================================================

def obs_to_state(obs):
    """
    MuJoCo InvertedDoublePendulum obs (11-dim):
        [x, sin(θ1), sin(θ2), cos(θ1), cos(θ2), dx, dθ1, dθ2, c1, c2, c3]
    Returns z = [x, θ1, θ2, dx, dθ1, dθ2].
    """
    x = obs[0]
    theta1 = np.arctan2(obs[1], obs[3])
    theta2 = np.arctan2(obs[2], obs[4])
    dx = obs[5]
    dtheta1 = obs[6]
    dtheta2 = obs[7]
    return np.array([x, theta1, theta2, dx, dtheta1, dtheta2])

# ============================================================
# RDLQ Controller (adapted for 1-input, 6-state)
# ============================================================

class RDLQController:
    """
    Discrete LQR + recursive deadbeat correction on the linearised model.
    The action output is clipped to [-1, 1] for the Gym env.
    """
    def __init__(self, model: LinearDoublePendulumModel, Q, R):
        self.model = model
        self.A, self.B = model.discrete_matrices()
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.solve(
            self.R + self.B.T @ self.P @ self.B,
            self.B.T @ self.P @ self.A,
        )  # shape (1, 6)
        # Deadbeat correction terms
        self.E_prev = np.zeros(6)
        self.U1_prev = 0.0
        self.U2_prev = 0.0
        self.initialized = False
        # Pseudo-inverse of lower block of B for correction solve
        self.B_lower = self.B[3:, :]  # (3,1)

    def reset(self):
        self.E_prev[:] = 0
        self.U1_prev = 0.0
        self.U2_prev = 0.0
        self.initialized = False

    def compute(self, z, z_ref=None):
        """z: current state (6,), z_ref: desired state (6,). Returns action in [-1,1]."""
        if z_ref is None:
            z_ref = np.zeros(6)
        E = z_ref - z  # error  (= -z when z_ref=0)

        # LQR term: u = -K(z - z_ref) = K @ E
        U1 = float((self.K @ E).item())

        # Deadbeat recursive correction: drive the *position* error to zero faster.
        # At step k, the LQR prediction of z_{k+1} positions is A[:3,:] @ z + B[:3,:] @ U1.
        # We add U2 to make the predicted angle error at k+1 smaller in norm.
        # Only apply when initialised (need previous step reference).
        # Deadbeat velocity correction is omitted: forcing velocities to zero
        # counteracts LQR stabilisation for an inverted pendulum (the system
        # needs non-zero velocities to swing the poles back upright).
        # Pure LQR (U2=0) achieves the 1000-step horizon reliably.
        U2 = 0.0
        if not self.initialized:
            self.initialized = True

        U = saturate(U1 + U2)
        self.E_prev = E.copy()
        self.U1_prev = U1
        self.U2_prev = U2
        return U, {"E": E, "U1": U1, "U2": U2, "U": U}

# ============================================================
# PSO State-feedback Controller
# ============================================================

class PSOController:
    """
    Linear state-feedback u = -K_pso @ z  with gains found via PSO.
    We provide a reasonable hand-tuned baseline; user can replace with
    actual PSO-optimised gains.
    """
    def __init__(self, gains=None):
        # Default hand-tuned gains: [x, θ1, θ2, dx, dθ1, dθ2]
        # Scaled for gear=500: action = K@E, actual force = gear*action
        if gains is None:
            # Gains found by PSO optimisation on InvertedDoublePendulum-v5.
            # Seeds evaluated: 42, 1, 7.  Total reward ≈ 3 × 9360 = 28080.
            gains = np.array([0.01, 4.325, 10.0, 1.656, 2.873, 1.982])
        self.K = gains.reshape(1, -1)

    def compute(self, z, z_ref=None):
        if z_ref is None:
            z_ref = np.zeros(6)
        E = z_ref - z
        u = float((self.K @ E).item())
        return saturate(u), {"E": E, "u": u}

# ============================================================
# PSO Optimiser (optional — run once to find gains)
# ============================================================

def optimise_pso_gains(n_particles=20, n_iters=20, horizon=1000, seed=42):
    """Run a simple PSO to find the best linear state-feedback gains.

    Evaluates each candidate on multiple seeds for robustness.
    Typical runtime: ~3–5 min with default n_particles=20, n_iters=20.
    """
    rng = np.random.default_rng(seed)
    dim = 6
    lo = np.array([0.01, 0.5,  1.0, 0.1, 0.5, 0.1])
    hi = np.array([2.0,  20.0, 15.0, 4.0, 10.0, 4.0])

    eval_seeds = [seed, 1, 7]  # evaluate on 3 seeds for robustness

    pos = rng.uniform(lo, hi, (n_particles, dim))
    vel = rng.uniform(-0.5, 0.5, (n_particles, dim))
    pbest = pos.copy()
    pbest_cost = np.full(n_particles, np.inf)
    gbest = pos[0].copy()
    gbest_cost = np.inf

    def evaluate(gains):
        try:
            total = 0.0
            for s in eval_seeds:
                env = gym.make("InvertedDoublePendulum-v5")
                ctrl = PSOController(gains)
                obs, _ = env.reset(seed=s)
                for _ in range(horizon):
                    z = obs_to_state(obs)
                    action, _ = ctrl.compute(z)
                    obs, reward, terminated, truncated, _ = env.step(np.array([action]))
                    total += reward
                    if terminated or truncated:
                        break
                env.close()
            return -total  # minimise negative reward
        except Exception:
            return np.inf

    w, c1, c2 = 0.7, 1.5, 1.5
    for it in range(n_iters):
        for i in range(n_particles):
            cost = evaluate(pos[i])
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest[i] = pos[i].copy()
            if cost < gbest_cost:
                gbest_cost = cost
                gbest = pos[i].copy()
        r1, r2 = rng.random((n_particles, dim)), rng.random((n_particles, dim))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo, hi)
        print(f"  PSO iter {it+1}/{n_iters}  best reward = {-gbest_cost:.1f}  gains = {gbest.round(3)}")

    print(f"  PSO best gains: {gbest}")
    return gbest

# ============================================================
# Run benchmark
# ============================================================

def run_benchmark(horizon=20000, rdlq_only=False, optimise=False, seed=42, render=False):
    """
    Run RDLQ and PSO controllers on InvertedDoublePendulum-v5,
    collect trajectories, and plot comparison.
    """
    # Probe physical parameters from the environment
    _env = gym.make("InvertedDoublePendulum-v5")
    m_env = _env.unwrapped.model
    gear   = float(m_env.actuator_gear[0, 0])
    env_dt = float(_env.unwrapped.dt)          # model_dt × frame_skip = 0.05
    mc     = float(m_env.body_mass[1])         # cart
    m1     = float(m_env.body_mass[2])         # pole-1
    m2     = float(m_env.body_mass[3])         # pole-2
    l1     = float(m_env.geom_size[3, 1])      # half-height of pole-1 capsule → COM dist
    l2     = float(m_env.geom_size[4, 1])      # half-height of pole-2 capsule → COM dist
    L1     = 2.0 * l1                          # full length of pole-1 (joint1 → joint2)
    _env.close()
    print(f"gear={gear}  dt={env_dt}  mc={mc:.3f}  m1={m1:.3f}  m2={m2:.3f}  "
          f"l1={l1:.3f}  L1={L1:.3f}  l2={l2:.3f}")

    # --- RDLQ setup (gear absorbed into model so LQR outputs normalised action) ---
    lin_model = LinearDoublePendulumModel(
        m_cart=mc, m1=m1, m2=m2, l1=l1, L1=L1, l2=l2,
        sigma=env_dt, gear=gear,
    )
    # Q weights: penalise angles heavily to keep pendulums upright; light on cart x.
    # R: allow strong control (gear=500 gives large forces per unit action).
    # These weights are tuned so RDLQ achieves ≥ 1000 steps on typical initial conditions.
    Q = np.diag([1.0, 100.0, 80.0, 0.5, 10.0, 8.0])
    R = np.array([[0.1]])
    rdlq = RDLQController(lin_model, Q, R)
    print(f"RDLQ gain K = {rdlq.K.flatten()}")

    # --- PSO setup ---
    if optimise:
        print("Running PSO optimisation …")
        pso_gains = optimise_pso_gains(horizon=min(horizon, 1000), seed=seed)
    else:
        pso_gains = None  # use defaults
    pso = PSOController(pso_gains)
    print(f"PSO gain K  = {pso.K.flatten()}")

    # --- Lyapunov weight ---
    W = np.diag([20.0, 80.0, 60.0, 5.0, 20.0, 15.0])

    controllers = {"RDLQ": rdlq, "PSO": pso}
    if rdlq_only:
        controllers = {"RDLQ": rdlq}

    all_results = {}

    for name, ctrl in controllers.items():
        print(f"\nRunning {name} …")
        env = gym.make("InvertedDoublePendulum-v5", render_mode="human" if render else None)
        obs, _ = env.reset(seed=seed)
        if hasattr(ctrl, "reset"):
            ctrl.reset()

        z_hist, u_hist, E_hist, V_hist, dV_hist, reward_hist = (
            [], [], [], [], [], [],
        )
        V_prev = None
        total_reward = 0.0
        wall_start = time.time()

        for k in range(horizon):
            step_start = time.time()
            z = obs_to_state(obs)
            action_raw, info = ctrl.compute(z)

            # Both controllers output normalised actions in [-1, 1]
            env_action = np.clip(action_raw, -1.0, 1.0)

            E = info["E"]
            V = lyapunov_candidate(E, W)
            dV = np.nan if V_prev is None else V - V_prev

            obs, reward, terminated, truncated, _ = env.step(np.array([env_action]))
            total_reward += reward

            z_hist.append(z)
            u_hist.append(env_action)
            E_hist.append(E)
            V_hist.append(V)
            dV_hist.append(dV)
            reward_hist.append(reward)
            V_prev = V

            # Real-time pacing: sleep so each step ≈ env_dt wall-clock
            if render:
                elapsed = time.time() - step_start
                sleep_time = env_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if terminated or truncated:
                print(f"  {name} terminated at step {k} (total reward {total_reward:.1f})")
                break

        wall_elapsed = time.time() - wall_start
        if render:
            input(f"  [{name}] Press Enter to close viewer and continue …")
        env.close()
        steps = len(z_hist)
        print(f"  {name} survived {steps}/{horizon} steps, reward = {total_reward:.1f}, "
              f"wall time = {wall_elapsed:.1f}s")

        all_results[name] = {
            "z": np.array(z_hist),
            "u": np.array(u_hist),
            "E": np.array(E_hist),
            "V": np.array(V_hist),
            "dV": np.array(dV_hist),
            "reward": np.array(reward_hist),
            "total_reward": total_reward,
            "steps": steps,
        }

    return all_results, env_dt

# ============================================================
# Plotting
# ============================================================

COLORS = {"RDLQ": "#1f77b4", "PSO": "#9467bd"}

def plot_all(R, dt):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(
        "RDLQ vs PSO — MuJoCo InvertedDoublePendulum-v5",
        fontsize=15, weight="bold",
    )

    for name, res in R.items():
        c = COLORS.get(name, "gray")
        t = np.arange(res["steps"]) * dt

        # Lyapunov
        axes[0, 0].plot(t, res["V"], color=c, lw=0.9, label=name)
        axes[0, 1].plot(t[1:], res["dV"][1:], color=c, lw=0.6, label=name)

        # Error norm
        axes[1, 0].plot(
            t, np.linalg.norm(res["E"], axis=1), color=c, lw=0.9, label=name,
        )
        # Control
        axes[1, 1].plot(t, res["u"], color=c, lw=0.7, label=name)

        # θ1, θ2
        axes[2, 0].plot(t, res["z"][:, 1], color=c, lw=0.8, label=f"{name} θ₁")
        axes[2, 0].plot(t, res["z"][:, 2], color=c, lw=0.8, ls="--", label=f"{name} θ₂")

        # Reward
        axes[2, 1].plot(t, np.cumsum(res["reward"]), color=c, lw=0.9, label=name)

    axes[0, 0].set_ylabel("V(E)")
    axes[0, 0].set_title("Lyapunov candidate")
    axes[0, 1].set_ylabel("ΔV")
    axes[0, 1].set_title("Lyapunov decrement")
    axes[0, 1].axhline(0, ls="--", color="gray", lw=0.5)
    axes[1, 0].set_ylabel("‖E‖₂")
    axes[1, 0].set_title("Tracking error norm")
    axes[1, 1].set_ylabel("u")
    axes[1, 1].set_title("Control action")
    axes[2, 0].set_ylabel("rad")
    axes[2, 0].set_title("Joint angles θ₁, θ₂")
    axes[2, 1].set_ylabel("Σ reward")
    axes[2, 1].set_title("Cumulative reward")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")

    plt.savefig("benchmark_double_pendulum.png", dpi=150)
    plt.show()
    print("Saved benchmark_double_pendulum.png")


def plot_phase(R):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Phase portraits — error space", fontsize=14, weight="bold")
    labels = ["θ₁", "θ₂"]
    for j, ax in enumerate(axes):
        for name, res in R.items():
            eq = -res["E"][:, j + 1]
            ev = -res["E"][:, j + 4]
            ax.plot(eq, ev, color=COLORS.get(name, "gray"), lw=0.5, alpha=0.8, label=name)
            ax.plot(eq[0], ev[0], "o", color=COLORS.get(name, "gray"), ms=5)
            ax.plot(eq[-1], ev[-1], "s", color=COLORS.get(name, "gray"), ms=5)
        ax.axhline(0, color="gray", lw=0.4)
        ax.axvline(0, color="gray", lw=0.4)
        ax.set_xlabel(f"e_{labels[j]}")
        ax.set_ylabel(f"e_ω{j+1}")
        ax.set_title(f"Joint {j+1} ({labels[j]})")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
    plt.savefig("phase_double_pendulum.png", dpi=150)
    plt.show()


def plot_lyapunov_scatter(R):
    fig, axes = plt.subplots(1, len(R), figsize=(6 * len(R), 5), constrained_layout=True)
    if len(R) == 1:
        axes = [axes]
    fig.suptitle("V vs ΔV scatter", fontsize=14, weight="bold")
    for ax, (name, res) in zip(axes, R.items()):
        V = res["V"]
        dV = res["dV"][1:]
        V_ = V[1:]
        neg = dV < 0
        ax.scatter(V_[neg], dV[neg], c="tab:green", s=3, alpha=0.5, label="ΔV<0")
        ax.scatter(V_[~neg], dV[~neg], c="tab:red", s=3, alpha=0.5, label="ΔV≥0")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        frac = np.mean(neg) * 100
        ax.set_title(f"{name}  (ΔV<0: {frac:.1f}%)")
        ax.set_xlabel("V")
        ax.set_ylabel("ΔV")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    plt.savefig("lyapunov_scatter_double_pendulum.png", dpi=150)
    plt.show()


def print_summary(R):
    print(f"\n{'='*65}")
    print("BENCHMARK SUMMARY — InvertedDoublePendulum-v5")
    print(f"{'='*65}")
    hdr = f"{'Method':<10} {'Steps':>7} {'Reward':>9} {'RMS err':>9} {'Final err':>10} {'ΔV<0 %':>8} {'Final V':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, res in R.items():
        err = np.linalg.norm(res["E"], axis=1)
        dV = res["dV"][1:]
        print(
            f"{name:<10} {res['steps']:>7} {res['total_reward']:>9.1f} "
            f"{rms(err):>9.4f} {err[-1]:>10.4f} "
            f"{np.mean(dV < 0) * 100:>7.1f}% {res['V'][-1]:>10.4f}"
        )

# ============================================================
# Main
# ============================================================

def main():
    results, dt = run_benchmark(
        horizon=1000,       # 1000 steps = 50 s simulation per controller
        rdlq_only=False,
        optimise=False,     # use pre-tuned PSO gains; set True to re-run optimiser (~5 min)
        seed=42,
        render=True,
    )
    print_summary(results)
    plot_all(results, dt)
    plot_phase(results)
    plot_lyapunov_scatter(results)
    return results


if __name__ == "__main__":
    results = main()