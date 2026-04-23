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
    p_destino cargando 3 PuzzleBots sobre el dorso.
    Delegado a simulate_anymal_to_target().

PHASE 2.5 - XARM_TRANSFER
    Un xArm 6 fijo junto a la mesa baja los 3 PuzzleBots del dorso del
    ANYmal y los deposita sobre una mesa. Con eso termina la mision.
"""

import math

import numpy as np

from .robots_base import HuskyA200, ANYmal, PuzzleBot
from .husky_pusher import HuskyPusher, CorridorWorld
from .anymal_gait import simulate_anymal_to_target
from .xarm import XArm


# ===========================================================================
#  Unidad PuzzleBot (sin brazo; solo pasajero del ANYmal)
# ===========================================================================
class PuzzleBotUnit:
    """Un PuzzleBot pasajero del mini reto.

    Solo necesita su base movil; el xArm se encarga de agarrarlo. La
    clase existe para conservar un nombre humano (PB_A/B/C) y la pose
    final en la mesa.
    """

    def __init__(self, name, role, start_xy=(11.0, 3.6),
                 base_kwargs=None):
        self.name = name
        self.role = role                        # 'A', 'B', 'C'
        self.base = PuzzleBot(**(base_kwargs or {}))
        self.base.reset(start_xy[0], start_xy[1], 0.0)


# ===========================================================================
#  Coordinador
# ===========================================================================
class MissionCoordinator:
    """FSM que orquesta las 3 fases del mini reto.

    Uso minimo:

        coord = MissionCoordinator()
        log = coord.run()

    El log es un dict con entradas ``phase1``, ``phase2``, ``phase2_5``
    mas un resumen ``success`` por fase.
    """

    PHASES = ('HUSKY_CLEAR', 'ANYMAL_TRANSPORT', 'XARM_TRANSFER', 'DONE')
    # Orden en que el xArm transfiere los PBs a la mesa
    TRANSFER_ORDER = ('C', 'B', 'A')
    # Destino del ANYmal: justo al sur del xArm, para que su reach de
    # ~0.73 m cubra todo el dorso.
    ANYMAL_TARGET = (11.5, 3.0)

    # Posiciones de las 3 plazas del dorso del ANYmal donde viajan los
    # PuzzleBots. Offsets en marco del ANYmal (frente/atras, lateral, alto).
    # Espaciado reducido a 0.22 m para que los 3 entren en el reach del xArm.
    PB_ON_ANYMAL_OFFSETS = {
        'C': (-0.22, 0.0, 0.30),
        'B': ( 0.00, 0.0, 0.30),
        'A': ( 0.22, 0.0, 0.30),
    }
    # Posiciones donde el xArm deposita cada PuzzleBot (en el piso, al
    # norte del xArm, cerca de la mesa). Coinciden con el ``start_xy``
    # con el que la fase 3 espera a los bots.
    PB_TABLE_DROP = {
        'C': (11.30, 3.65),
        'B': (11.50, 3.65),
        'A': (11.70, 3.65),
    }
    # Base del xArm 6 (fija, anclada al piso), entre el ANYmal y la mesa.
    XARM_BASE_XY = (11.50, 3.35)
    XARM_BASE_YAW = math.pi / 2.0

    def __init__(self, husky_terrain="grass"):
        # --- Fase 1 ---
        self.husky = HuskyA200()
        self.husky.set_terrain(husky_terrain)
        self.husky.reset(x=0.0, y=2.0, theta=0.0)
        self.world = CorridorWorld()
        self.pusher = HuskyPusher(self.husky, self.world)

        # --- Fase 2 ---
        self.anymal = ANYmal()

        # --- Fase 2.5: xArm 6 fijo junto a la mesa ---
        self.xarm = XArm(base_xy=self.XARM_BASE_XY,
                         base_yaw=self.XARM_BASE_YAW)

        # 3 PuzzleBots pasajeros: empiezan sobre el dorso del ANYmal y
        # el xArm los baja a la mesa al final de la fase 2.5.
        self.units = [
            PuzzleBotUnit('PB_C', 'C', start_xy=self.PB_TABLE_DROP['C']),
            PuzzleBotUnit('PB_B', 'B', start_xy=self.PB_TABLE_DROP['B']),
            PuzzleBotUnit('PB_A', 'A', start_xy=self.PB_TABLE_DROP['A']),
        ]

        # Estado de la FSM
        self.phase = 'HUSKY_CLEAR'
        self.log = {
            'phase1': None,
            'phase2': None,
            'phase2_5': None,
            'success': {},
        }

    # ------------------------------------------------------------------
    def run(self):
        """Ejecuta las 3 fases en secuencia y retorna el log completo."""
        self.run_phase1()
        self.run_phase2()
        self.run_phase2_5()
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
        self.phase = 'XARM_TRANSFER'
        return log

    def run_phase2_5(self):
        """El xArm 6 baja los 3 PuzzleBots del dorso del ANYmal a la mesa.

        El ANYmal ya llego a ``ANYMAL_TARGET``; los 3 PuzzleBots estan en
        su dorso con offsets ``PB_ON_ANYMAL_OFFSETS``. El xArm hace 3
        ciclos pick & place (orden C, B, A) llevando cada bot a su
        ``PB_TABLE_DROP``.
        """
        assert self.phase == 'XARM_TRANSFER'
        # Pose final de la base del ANYmal
        p2 = self.log['phase2']
        ax, ay, ayaw = (p2['base_x'][-1], p2['base_y'][-1], p2['base_yaw'][-1])

        units_log = []
        all_ok = True
        for role in self.TRANSFER_ORDER:
            off = self.PB_ON_ANYMAL_OFFSETS[role]
            # Pose del PuzzleBot sobre el ANYmal en marco mundo
            cs, sn = math.cos(ayaw), math.sin(ayaw)
            pick_x = ax + cs * off[0] - sn * off[1]
            pick_y = ay + sn * off[0] + cs * off[1]
            pick_z = off[2]
            p_pick = np.array([pick_x, pick_y, pick_z])

            # Destino en la mesa
            drop = self.PB_TABLE_DROP[role]
            # Altura de "apoyo" en la mesa (el bot tiene ~0.10 m de alto)
            p_place = np.array([drop[0], drop[1], 0.10])

            # Trayectoria cartesiana del TCP y trayectoria articular via IK
            cart_path = self.xarm.pick_place_cartesian_path(
                p_pick, p_place, approach_height=0.15, n_seg=12)
            q_path = self.xarm.joint_path_from_cartesian(cart_path)

            # Indices notables del path (ver pick_place_cartesian_path):
            # 0=home, n_seg=above_pick, 2n=pick, 3n=above_pick,
            # 4n=above_place, 5n=place, 6n=above_place, 7n=home
            n = 12
            idx_grab = 2 * n     # PB queda enganchado al TCP
            idx_release = 5 * n  # PB queda en la mesa

            # En la sim: el PB sigue al TCP entre [idx_grab, idx_release].
            units_log.append({
                'role': role,
                'p_pick': p_pick.tolist(),
                'p_place': p_place.tolist(),
                'cart_path': cart_path,
                'q_path': q_path,
                'idx_grab': idx_grab,
                'idx_release': idx_release,
            })
            # Dejamos el xArm en home para el siguiente ciclo
            self.xarm.reset()

        self.log['phase2_5'] = {
            'units': units_log,
            'base_xy': tuple(self.XARM_BASE_XY),
            'base_yaw': self.XARM_BASE_YAW,
            'anymal_final': (ax, ay, ayaw),
            'success': all_ok,
        }
        self.log['success']['phase2_5'] = all_ok
        self.phase = 'DONE'
        return self.log['phase2_5']


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

    p25 = log['phase2_5']
    print(f"Fase 2.5 (xArm)  : PBs transferidos={len(p25['units'])}, "
          f"exito={p25['success']}")
    for u in p25['units']:
        print(f"   PB_{u['role']} pick={u['p_pick']} -> place={u['p_place']}")
    print(f"\nFase final: {coord.phase}")
