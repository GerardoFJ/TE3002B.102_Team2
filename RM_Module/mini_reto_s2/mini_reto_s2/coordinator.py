"""Coordinador del mini reto: maquina de estados de las 3 fases.

Sigue el consejo del profesor: la maquina de estados debe ser dumb,
solo llamar funciones de los modulos especializados. Toda la logica
"inteligente" (planificacion, FK, IK, control) vive en sus respectivos
modulos.

Fases
-----
PHASE 1 - HUSKY_CLEAR
    El Husky empuja las 3 cajas grandes fuera del corredor 6x2 m.
    Delegado a HuskyPusher.clear_corridor().

PHASE 2 - ANYMAL_TRANSPORT
    El ANYmal camina en trote desde la zona de inicio hasta el punto
    p_destino = (11.0, 3.6) cargando 3 PuzzleBots sobre el dorso.
    Delegado a simulate_anymal_to_target().

PHASE 3 - PUZZLEBOT_STACK
    Los 3 PuzzleBots, cada uno con un mini brazo de 3 DoF, recogen una
    caja pequena (A, B o C) y la apilan en la pila destino. Orden
    obligatorio C abajo, B en medio, A arriba. Coordinacion via
    time-slotting (solo un PuzzleBot activo a la vez).
"""

import math

import numpy as np

from .robots_base import HuskyA200, ANYmal, PuzzleBot
from .puzzlebot_arm import PuzzleBotArm
from .husky_pusher import HuskyPusher, CorridorWorld
from .anymal_gait import simulate_anymal_to_target


# ===========================================================================
#  Modelos del entorno de la fase 3
# ===========================================================================
class SmallBox:
    """Caja pequena (4 cm de lado) para los mini brazos."""

    def __init__(self, name, xy, z=0.05, side=0.02):
        self.name = name
        self.xy = np.array(xy, dtype=float)
        self.z = float(z)             # Altura del centro [m]
        self.side = float(side)       # Half-side [m]
        self.placed = False

    @property
    def position(self):
        return np.array([self.xy[0], self.xy[1], self.z])


class WorkZone:
    """Zona de trabajo: 3 cajas pequenas + un punto de apilado."""

    def __init__(self,
                 box_a=(12.00, 3.45),
                 box_b=(12.30, 3.45),
                 box_c=(12.60, 3.45),
                 stack_xy=(12.30, 3.85),
                 table_z=0.05):
        self.boxes = {
            'A': SmallBox('A', box_a, z=table_z),
            'B': SmallBox('B', box_b, z=table_z),
            'C': SmallBox('C', box_c, z=table_z),
        }
        self.stack_xy = np.array(stack_xy, dtype=float)
        self.table_z = float(table_z)
        # Numero de cajas ya apiladas (0, 1, 2, 3)
        self.stack_count = 0
        # Altura entre centros de cajas adyacentes en la pila
        self.layer_height = 2.0 * self.boxes['A'].side + 0.005


# ===========================================================================
#  Unidad PuzzleBot + brazo
# ===========================================================================
class PuzzleBotUnit:
    """Una unidad fisica del reto: PuzzleBot + brazo + caja asignada.

    El brazo esta montado en el centro del PuzzleBot a la altura de la
    base (altura del shoulder = ``arm.l1`` sobre el suelo del bot).
    """

    def __init__(self, name, role_box, start_xy=(11.0, 3.6),
                 base_kwargs=None, arm_kwargs=None):
        self.name = name
        self.role_box = role_box                # 'A', 'B', 'C'
        self.base = PuzzleBot(**(base_kwargs or {}))
        self.arm = PuzzleBotArm(**(arm_kwargs or {}))
        self.base.reset(start_xy[0], start_xy[1], 0.0)
        self.arm.reset()


# ===========================================================================
#  Helper: drive a PuzzleBot to a 2D point
# ===========================================================================
def drive_puzzlebot_to(pb, target_xy, dt=0.05, eps=0.05,
                       v_max=0.4, omega_max=1.5,
                       angle_threshold_deg=12.0,
                       max_steps=600):
    """Lleva un PuzzleBot a un punto (x, y) con go-to-pose. Devuelve el log."""
    log = {'t': [], 'x': [], 'y': [], 'theta': [],
           'v_cmd': [], 'omega_cmd': []}
    angle_thresh = math.radians(angle_threshold_deg)
    for i in range(max_steps):
        x, y, th = pb.get_pose()
        dx = target_xy[0] - x
        dy = target_xy[1] - y
        dist = math.hypot(dx, dy)
        if dist < eps:
            break
        target_yaw = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(target_yaw - th),
                             math.cos(target_yaw - th))
        if abs(yaw_err) > angle_thresh:
            v_cmd = 0.0
            omega_cmd = float(np.clip(2.0 * yaw_err, -omega_max, omega_max))
        else:
            v_cmd = float(np.clip(0.6 * dist, 0.0, v_max))
            v_cmd *= max(0.0, math.cos(yaw_err))
            omega_cmd = float(np.clip(2.0 * yaw_err, -omega_max, omega_max))

        wR, wL = pb.inverse_kinematics(v_cmd, omega_cmd)
        v_real, omega_real = pb.forward_kinematics(wR, wL)
        pb.update_pose(v_real, omega_real, dt)

        log['t'].append(i * dt)
        log['x'].append(pb.x)
        log['y'].append(pb.y)
        log['theta'].append(pb.theta)
        log['v_cmd'].append(v_cmd)
        log['omega_cmd'].append(omega_cmd)
    return log


# ===========================================================================
#  Coordinador
# ===========================================================================
class MissionCoordinator:
    """FSM que orquesta las 3 fases del mini reto.

    Uso minimo:

        coord = MissionCoordinator()
        log = coord.run()

    El log es un dict con tres entradas (``phase1``, ``phase2``,
    ``phase3``) mas un resumen ``success`` por fase.
    """

    PHASES = ('HUSKY_CLEAR', 'ANYMAL_TRANSPORT', 'PUZZLEBOT_STACK', 'DONE')
    STACK_ORDER = ('C', 'B', 'A')      # Apilar de abajo hacia arriba
    ANYMAL_TARGET = (11.0, 3.6)

    def __init__(self,
                 husky_terrain="grass",
                 work_zone=None):
        # --- Fase 1 ---
        self.husky = HuskyA200()
        self.husky.set_terrain(husky_terrain)
        self.husky.reset(x=0.0, y=2.0, theta=0.0)
        self.world = CorridorWorld()
        self.pusher = HuskyPusher(self.husky, self.world)

        # --- Fase 2 ---
        self.anymal = ANYmal()

        # --- Fase 3 ---
        self.work_zone = work_zone or WorkZone()
        # 3 PuzzleBots colocados en la zona de trabajo (los baja el ANYmal)
        self.units = [
            PuzzleBotUnit('PB_C', 'C', start_xy=(11.20, 3.30)),
            PuzzleBotUnit('PB_B', 'B', start_xy=(11.20, 3.60)),
            PuzzleBotUnit('PB_A', 'A', start_xy=(11.20, 3.90)),
        ]

        # Estado de la FSM
        self.phase = 'HUSKY_CLEAR'
        self.log = {
            'phase1': None,
            'phase2': None,
            'phase3': None,
            'success': {},
        }

    # ------------------------------------------------------------------
    def run(self):
        """Ejecuta las 3 fases en secuencia y retorna el log completo."""
        self.run_phase1()
        self.run_phase2()
        self.run_phase3()
        return self.log

    # ------------------------------------------------------------------
    def run_phase1(self):
        """Husky despeja el corredor."""
        assert self.phase == 'HUSKY_CLEAR'
        log = self.pusher.clear_corridor(dt=0.05)
        self.log['phase1'] = log
        self.log['success']['phase1'] = bool(log['success'])
        self.phase = 'ANYMAL_TRANSPORT'
        return log

    def run_phase2(self):
        """ANYmal trota hasta el punto de entrega."""
        assert self.phase == 'ANYMAL_TRANSPORT'
        log = simulate_anymal_to_target(
            self.anymal,
            target_xy=self.ANYMAL_TARGET,
            T_max=60.0,
            dt=0.01,
        )
        self.log['phase2'] = log
        self.log['success']['phase2'] = bool(log['success'])
        self.phase = 'PUZZLEBOT_STACK'
        return log

    def run_phase3(self):
        """3 PuzzleBots apilan A-B-C en orden C, B, A (time-slotting)."""
        assert self.phase == 'PUZZLEBOT_STACK'
        units_log = []
        for box_name in self.STACK_ORDER:
            unit = next(u for u in self.units if u.role_box == box_name)
            units_log.append(self._pick_and_place(unit, box_name))

        all_placed = all(b.placed for b in self.work_zone.boxes.values())
        ph3 = {
            'order': list(self.STACK_ORDER),
            'units': units_log,
            'success': all_placed,
            'final_box_positions': {
                name: {'xy': b.xy.tolist(), 'z': b.z, 'placed': b.placed}
                for name, b in self.work_zone.boxes.items()
            },
        }
        self.log['phase3'] = ph3
        self.log['success']['phase3'] = all_placed
        self.phase = 'DONE'
        return ph3

    # ------------------------------------------------------------------
    def _pick_and_place(self, unit, box_name):
        """Sub-rutina pick & place para una unidad y una caja.

        Pasos:
            1. Manejar el PuzzleBot a un punto de approach junto a la caja.
            2. Brazo: trayectoria cartesiana hasta agarrar la caja
               (calcula tau = J^T f para el grip).
            3. Manejar el PuzzleBot al punto de apilado.
            4. Brazo: colocar la caja a la altura correspondiente del stack.
            5. Marcar la caja como colocada y avanzar el contador.
        """
        box = self.work_zone.boxes[box_name]

        # --- 1. Drive to box ---
        # El PuzzleBot se posiciona ~0.12 m al oeste de la caja, encarando +x
        approach_pick = np.array([box.xy[0] - 0.12, box.xy[1]])
        drive_log_pick = drive_puzzlebot_to(unit.base, approach_pick)

        # --- 2. Brazo: grasp ---
        # Posicion de la caja en marco brazo: x = 0.12, y = 0, z = box.z
        # (el brazo esta montado en el centro del bot, base a altura 0)
        box_in_arm_frame = np.array([0.10, 0.0, box.z])
        grasp_result = unit.arm.grasp_box(box_in_arm_frame,
                                          grip_force=2.0,
                                          n_steps=15)

        # --- 3. Drive to stack ---
        approach_place = np.array([self.work_zone.stack_xy[0] - 0.12,
                                   self.work_zone.stack_xy[1]])
        drive_log_place = drive_puzzlebot_to(unit.base, approach_place)

        # --- 4. Brazo: place ---
        target_z = (self.work_zone.table_z
                    + self.work_zone.stack_count * self.work_zone.layer_height)
        place_pos = np.array([0.10, 0.0, target_z])
        # Reseteamos el arm para que la trayectoria parta de su pose
        # actual (post-grasp) hasta el punto de coloca.
        place_result = unit.arm.grasp_box(place_pos,
                                          grip_force=2.0,
                                          n_steps=15)

        # --- 5. Actualizar la caja y la pila ---
        box.xy = self.work_zone.stack_xy.copy()
        box.z = target_z
        box.placed = True
        self.work_zone.stack_count += 1

        return {
            'unit_name': unit.name,
            'box': box_name,
            'drive_pick': drive_log_pick,
            'drive_place': drive_log_place,
            'grasp': grasp_result,
            'place': place_result,
            'tau_grasp': grasp_result['tau_grip'].tolist(),
            'tau_place': place_result['tau_grip'].tolist(),
            'singular_grasp': bool(grasp_result['singular']),
            'singular_place': bool(place_result['singular']),
        }


# ===========================================================================
#  Demo rapida
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" MISSION COORDINATOR - mini reto Semana 2")
    print("=" * 60)
    coord = MissionCoordinator()
    log = coord.run()

    print("\n--- Resumen por fase ---")
    for k, v in log['success'].items():
        ok = "OK " if v else "FAIL"
        print(f"  [{ok}] {k}")

    p1 = log['phase1']
    print(f"\nFase 1 (Husky)   : pasos={len(p1['t'])}, "
          f"v_cmd_prom={np.mean(p1['v_cmd']):+.3f} m/s")

    p2 = log['phase2']
    print(f"Fase 2 (ANYmal)  : error final={p2['final_error']:.3f} m, "
          f"violaciones det(J)={sum(p2['violations'].values())}")

    p3 = log['phase3']
    print(f"Fase 3 (PB stack): orden={p3['order']}, "
          f"todas colocadas={p3['success']}")
    for u in p3['units']:
        print(f"   {u['unit_name']} -> caja {u['box']}: "
              f"tau_grasp norm={np.linalg.norm(u['tau_grasp']):.4f}")
    print(f"\nFase final: {coord.phase}")
