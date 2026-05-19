"""Coordinador del mini reto — pipeline guiado por visión (sin waypoints).

Pipeline completo:

  FASE 1 — HUSKY_CLEAR
    1. Cámara global renderiza escena → detecta grid, robots y obstáculos.
    2. Cámara planifica path del ANYmal (A* 4-vecinos sobre lo detectado).
    3. Cámara detecta qué obstáculos bloquean ese path.
    4. Para cada obstáculo bloqueante:
         a. Cámara planifica ruta del Husky hacia la celda de approach.
         b. Husky navega SOBRE el grid celda a celda (cámara decide en
            cada intersección — sin lista de waypoints).
         c. Husky empuja el obstáculo fuera del corredor.
    5. Husky regresa a su celda inicial sobre el grid.
    6. Cámara valida que el path del ANYmal está despejado.

  FASE 2 — ANYMAL_TRANSPORT
    1. Cámara valida path limpio.
    2. ANYmal navega SOBRE el grid celda a celda (cámara decide en cada
       intersección — sin lista de waypoints).

  FASE 2.5 — XARM_TRANSFER  (sin cambios)
    xArm baja los 3 PuzzleBots del dorso del ANYmal a la mesa.
"""

import math

import numpy as np

from .robots_base import HuskyA200, ANYmal, PuzzleBot
from .husky_pusher import HuskyPusher, CorridorWorld
from .xarm import XArm
from .aerial_camera import AerialCamera, HUSKY_RGB, ANYMAL_RGB
from .line_follower import transit_to_waypoints, LINE_RGB
from .navigation_grid import NavigationGrid
from .anymal_gait import simulate_anymal_grid_vision, ANYmalTrotGait, ANYmalNavigator


# ===========================================================================
#  Unidad PuzzleBot (pasajero)
# ===========================================================================
class PuzzleBotUnit:
    def __init__(self, name, role, start_xy=(11.0, 3.6), base_kwargs=None):
        self.name = name
        self.role = role
        self.base = PuzzleBot(**(base_kwargs or {}))
        self.base.reset(start_xy[0], start_xy[1], 0.0)


# ===========================================================================
#  Coordinador
# ===========================================================================
class MissionCoordinator:
    """FSM + pipeline de visión que orquesta las 3 fases del mini reto.

    Uso:
        coord = MissionCoordinator()
        log   = coord.run()
    """

    PHASES        = ('HUSKY_CLEAR', 'ANYMAL_TRANSPORT', 'XARM_TRANSFER', 'DONE')
    TRANSFER_ORDER = ('C', 'B', 'A')
    ANYMAL_TARGET  = (12.0, 3.0)   # centro de celda (3,12) del grid

    PB_ON_ANYMAL_OFFSETS = {
        'C': (-0.22, 0.0, 0.30),
        'B': ( 0.00, 0.0, 0.30),
        'A': ( 0.22, 0.0, 0.30),
    }
    PB_TABLE_DROP = {
        'C': (11.50, 3.70),
        'B': (12.00, 3.70),
        'A': (12.50, 3.70),
    }
    XARM_BASE_XY  = (12.00, 3.5)  # entre ANYmal (y=3) y mesa (y=4), dentro del alcance
    XARM_BASE_YAW = math.pi / 2.0
    HUSKY_HOME_XY = (0.0,   2.0)

    def __init__(self, husky_terrain="grass"):
        self.husky = HuskyA200()
        self.husky.set_terrain(husky_terrain)
        self.husky.reset(self.HUSKY_HOME_XY[0], self.HUSKY_HOME_XY[1], 0.0)

        # Cámara global cenital y grid (necesarios antes de crear el mundo)
        self._grid = NavigationGrid()
        self._cam  = AerialCamera()

        # Calcular trayectoria limpia del ANYmal (sin obstáculos) y colocar
        # las cajas sobre ella para que el Husky tenga que despejarlas.
        _clean = self._grid.astar((0.0, 0.0), self.ANYMAL_TARGET, set())
        _interior = _clean[2:-2]   # excluir celdas de inicio y fin
        _n = len(_interior)
        if _n >= 3:
            _box_cells = [
                _interior[_n // 4],
                _interior[_n // 2],
                _interior[3 * _n // 4],
            ]
        else:
            _box_cells = _interior[:min(3, _n)]
        _box_positions = [self._grid.cell_center(r, c) for r, c in _box_cells]

        self.world  = CorridorWorld(box_positions=_box_positions)
        self.pusher = HuskyPusher(self.husky, self.world)
        self.anymal = ANYmal()
        self.xarm   = XArm(base_xy=self.XARM_BASE_XY, base_yaw=self.XARM_BASE_YAW)

        self.units = [
            PuzzleBotUnit('PB_C', 'C', start_xy=self.PB_TABLE_DROP['C']),
            PuzzleBotUnit('PB_B', 'B', start_xy=self.PB_TABLE_DROP['B']),
            PuzzleBotUnit('PB_A', 'A', start_xy=self.PB_TABLE_DROP['A']),
        ]

        # Paths actuales (calculados por la cámara, guardados para sim.py)
        self._paths_ordered = {}
        self._obstacle_cells = set()
        self._blocked_cells  = set()

        self.phase = 'HUSKY_CLEAR'
        self.log = {
            'phase1': None, 'phase2': None, 'phase2_5': None,
            'success': {},
            'push_order': [],
        }

    # ------------------------------------------------------------------
    #  Punto de entrada único
    # ------------------------------------------------------------------
    def run(self):
        self.run_phase1()
        self.run_phase2()
        self.run_phase2_5()
        return self.log

    # ==================================================================
    #  FASE 1 — Husky despeja el corredor
    # ==================================================================
    def run_phase1(self):
        assert self.phase == 'HUSKY_CLEAR'

        grid = self._grid
        cam  = self._cam
        dt   = 0.05

        # ── 1. Cámara detecta la escena ────────────────────────────────
        img = cam.render(grid, self.world.boxes,
                         {'husky': (self.husky.x, self.husky.y, self.husky.theta)})
        scene = cam.analyze_scene(img, grid)
        obs_cells = scene['obstacle_cells']
        self._obstacle_cells = obs_cells

        # ── 2. Planificar path del ANYmal (A* 4-vecinos) ───────────────
        anymal_path = cam.plan_path(
            grid, (0.0, 0.0), self.ANYMAL_TARGET, obs_cells)
        anymal_path_set = set(anymal_path)
        self._paths_ordered['anymal'] = anymal_path

        # ── 3. Detectar qué obstáculos bloquean el path del ANYmal ─────
        img = cam.render(grid, self.world.boxes,
                         {'husky': (self.husky.x, self.husky.y, self.husky.theta)})
        blocked = cam.detect_path_blockages(img, anymal_path_set, grid)
        self._blocked_cells = blocked

        # Cajas bloqueantes ordenadas oeste→este
        name_to_box = {b.name: b for b in self.world.boxes}
        blocking_boxes = [
            b for b in sorted(self.world.boxes_in_corridor(), key=lambda b: b.x)
            if grid.world_to_cell(b.x, b.y) in blocked
               or any(grid.world_to_cell(b.x, b.y) == cell
                      for cell in anymal_path_set)
        ]
        # Si no hay cajas bloqueantes en el path detectado, usar todas las del corredor
        if not blocking_boxes:
            blocking_boxes = sorted(self.world.boxes_in_corridor(), key=lambda b: b.x)

        push_order = [b.name for b in blocking_boxes]
        self.log['push_order'] = push_order

        # ── 4. Log de fase 1 ───────────────────────────────────────────
        log = self.pusher._empty_log(self.world.boxes)
        husky_full_path = []   # path acumulado de todo el recorrido del Husky

        def _extend_path(full, segment):
            """Concatena segment a full sin duplicar la celda de unión."""
            if not segment:
                return
            start = 1 if full and full[-1] == segment[0] else 0
            full.extend(segment[start:])

        for box in blocking_boxes:
            # ── 4a. Cámara planifica ruta Husky → approach de la caja ─
            approach_xy = (box.x, box.y + self.pusher.contact_offset
                           + self.pusher.approach_buffer)
            other_obs = {grid.world_to_cell(b.x, b.y)
                         for b in self.world.boxes if b is not box}

            img = cam.render(grid, self.world.boxes,
                             {'husky': (self.husky.x, self.husky.y, self.husky.theta)})
            husky_path = cam.plan_path(
                grid,
                (self.husky.x, self.husky.y),
                approach_xy,
                other_obs)
            _extend_path(husky_full_path, husky_path)
            self._paths_ordered['husky'] = list(husky_full_path)

            # ── 4b. Husky navega SOBRE el grid (cámara decide en cada celda)
            self._follow_grid_vision(
                self.husky, husky_path, grid, cam,
                HUSKY_RGB, LINE_RGB['husky'], dt, log)

            # ── 4c. Husky empuja la caja ───────────────────────────────
            self.pusher.push_box(box, dt, log)

            # ── Actualizar rutas guardadas tras el empuje ──────────────
            img = cam.render(grid, self.world.boxes,
                             {'husky': (self.husky.x, self.husky.y, self.husky.theta)})
            obs_cells = cam.detect_obstacles(img, grid)
            self._obstacle_cells = obs_cells

        # ── 5. Husky regresa al home sobre el grid ─────────────────────
        img = cam.render(grid, self.world.boxes,
                         {'husky': (self.husky.x, self.husky.y, self.husky.theta)})
        home_path = cam.plan_path(
            grid,
            (self.husky.x, self.husky.y),
            self.HUSKY_HOME_XY,
            cam.detect_obstacles(img, grid))
        _extend_path(husky_full_path, home_path)
        self._paths_ordered['husky'] = list(husky_full_path)
        self._follow_grid_vision(
            self.husky, home_path, grid, cam,
            HUSKY_RGB, LINE_RGB['husky'], dt, log)

        # ── 6. Cámara valida que el path del ANYmal está despejado ─────
        img = cam.render(grid, self.world.boxes,
                         {'husky': (self.husky.x, self.husky.y, self.husky.theta)})

        # Recalcular path final (con obstáculos ya desplazados) para sim.py
        final_obs = cam.detect_obstacles(img, grid)
        final_anymal = cam.plan_path(grid, (0.0, 0.0), self.ANYMAL_TARGET, final_obs)
        self._paths_ordered['anymal'] = final_anymal
        self._obstacle_cells = final_obs
        # Verificar bloqueos sobre el path FINAL (no el original)
        self._blocked_cells = cam.detect_path_blockages(img, set(final_anymal), grid)

        log['success'] = self.world.all_clear()
        self.log['phase1'] = log
        self.log['success']['phase1'] = bool(log['success'])
        self.phase = 'ANYMAL_TRANSPORT'
        return log

    # ==================================================================
    #  FASE 2 — ANYmal transporta los PuzzleBots
    # ==================================================================
    def run_phase2(self):
        assert self.phase == 'ANYMAL_TRANSPORT'

        grid = self._grid
        cam  = self._cam

        # ── 1. Cámara valida que el path está despejado ────────────────
        img = cam.render(grid, self.world.boxes,
                         {'anymal': (0.0, 0.0, 0.0)})
        obs = cam.detect_obstacles(img, grid)
        anymal_path = cam.plan_path(grid, (0.0, 0.0), self.ANYMAL_TARGET, obs)
        self._paths_ordered['anymal'] = anymal_path

        blocked = cam.detect_path_blockages(img, set(anymal_path), grid)
        if blocked:
            raise RuntimeError(
                f"Fase 2: path del ANYmal aun bloqueado en celdas {blocked}")

        # ── 2. ANYmal navega SOBRE el grid guiado por la cámara ────────
        log = simulate_anymal_grid_vision(
            self.anymal,
            goal_xy=self.ANYMAL_TARGET,
            aerial_cam=cam,
            grid=grid,
            T_max=120.0,
            dt=0.005,
            start_xy=(0.0, 0.0),
            boxes=self.world.boxes,
        )

        self.log['phase2'] = log
        self.log['success']['phase2'] = bool(log['success'])
        self.phase = 'XARM_TRANSFER'
        return log

    # ==================================================================
    #  FASE 2.5 — xArm transfiere los PuzzleBots
    # ==================================================================
    def run_phase2_5(self):
        assert self.phase == 'XARM_TRANSFER'

        p2 = self.log['phase2']
        ax, ay, ayaw = (p2['base_x'][-1], p2['base_y'][-1], p2['base_yaw'][-1])

        units_log = []
        for role in self.TRANSFER_ORDER:
            off = self.PB_ON_ANYMAL_OFFSETS[role]
            cs, sn = math.cos(ayaw), math.sin(ayaw)
            pick_x = ax + cs * off[0] - sn * off[1]
            pick_y = ay + sn * off[0] + cs * off[1]
            p_pick  = np.array([pick_x, pick_y, off[2]])
            drop    = self.PB_TABLE_DROP[role]
            p_place = np.array([drop[0], drop[1], 0.10])

            cart_path = self.xarm.pick_place_cartesian_path(
                p_pick, p_place, approach_height=0.15, n_seg=12)
            q_path = self.xarm.joint_path_from_cartesian(cart_path)
            n = 12
            units_log.append({
                'role': role,
                'p_pick': p_pick.tolist(),
                'p_place': p_place.tolist(),
                'cart_path': cart_path,
                'q_path': q_path,
                'idx_grab': 2 * n,
                'idx_release': 5 * n,
            })
            self.xarm.reset()

        result = {
            'units': units_log,
            'base_xy': tuple(self.XARM_BASE_XY),
            'base_yaw': self.XARM_BASE_YAW,
            'anymal_final': (ax, ay, ayaw),
            'success': True,
        }
        self.log['phase2_5'] = result
        self.log['success']['phase2_5'] = True
        self.phase = 'DONE'
        return result

    # ==================================================================
    #  Helper: navegación sobre el grid guiada por cámara
    # ==================================================================
    def _follow_grid_vision(self, robot, initial_path, grid, cam,
                            robot_rgb, line_rgb, dt, log):
        """Navega el robot SOBRE el grid celda a celda sin waypoints explícitos.

        En cada celda:
          1. La cámara global renderiza la escena y detecta la posición
             actual del robot y los obstáculos.
          2. ``cam.next_step`` corre A* sobre lo detectado y devuelve
             la dirección (dr, dc) para el siguiente paso.
          3. El robot sigue la línea hasta el centro de esa celda usando
             su cámara individual (``transit_to_waypoints``).
          4. Se repite hasta llegar a la celda destino.

        Parameters
        ----------
        initial_path : list of (row, col) — solo se usa la celda destino
        """
        if not initial_path:
            return

        goal_cell = initial_path[-1]
        MAX_CELLS = 60   # tope de seguridad

        for _ in range(MAX_CELLS):
            x, y, theta = robot.get_pose()
            cur_cell = grid.world_to_cell(x, y)
            if cur_cell == goal_cell:
                break

            # Cámara global: detecta escena actual y decide dirección
            img = cam.render(
                grid, self.world.boxes,
                {'husky': (self.husky.x, self.husky.y, self.husky.theta)})
            dr, dc = cam.next_step(img, robot_rgb, goal_cell, grid)

            if (dr, dc) == (0, 0):
                break   # destino alcanzado o sin camino

            # Celda siguiente y su centro en coordenadas mundo
            next_cell = (cur_cell[0] + dr, cur_cell[1] + dc)
            next_xy   = grid.cell_center(*next_cell)

            # Cámara individual: sigue la línea hasta el centro de la celda
            transit_to_waypoints(
                robot, [(x, y), next_xy], line_rgb, dt, log,
                world_boxes=self.world.boxes,
                eps=0.35, max_steps=800)
