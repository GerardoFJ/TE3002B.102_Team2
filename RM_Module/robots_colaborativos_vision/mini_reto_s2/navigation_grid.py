"""Sistema de navegacion por cuadricula 1×1 m con A* para el mini reto.

NavigationGrid divide el escenario en celdas de 1 m, resolucion tipica
de un complejo industrial. Sobre esta cuadricula se planean los caminos
de los 3 robots mediante A* (8 vecinos). El resultado se usa para:

  - Dibujar lineas de color pintadas en el suelo (Husky=amarillo,
    ANYmal=cian, xArm=magenta).
  - Proveer waypoints ordenados para que cada robot siga el path celda
    por celda usando su camara de seguimiento de lineas.
  - Determinar el orden optimo de empuje del Husky (cajas que bloquean
    la trayectoria directa del ANYmal se empujan primero).

Las lineas pintadas en el suelo siguen los centros de las celdas del
camino A*. Los robots las detectan con una camara de abordo y el pipeline
CV del modulo ``line_follower``.
"""

import heapq
import math

import numpy as np


# ---------------------------------------------------------------------------
# Colores de visualizacion
# ---------------------------------------------------------------------------
CELL_COLORS = {
    'husky':  ('#FFFF00', 0.80),   # amarillo vivo
    'anymal': ('#00FFFF', 0.80),   # cian vivo
    'xarm':   ('#FF00FF', 0.60),   # magenta
}
BLOCKED_CELL_COLOR = ('#FF3333', 0.55)

LINE_WIDTH = 12   # grosor de la linea pintada en el suelo (pts matplotlib)


# ===========================================================================
class NavigationGrid:
    """Cuadricula 2D uniforme 1×1 m con A* de 8 vecinos.

    Parameters
    ----------
    cell : float
        Lado de cada celda cuadrada [m]. Por defecto 1.0 (escala industrial).
    x_min, x_max, y_min, y_max : float
        Limites del escenario [m].
    """

    def __init__(self, cell=1.0,
                 x_min=-0.5, x_max=13.5,
                 y_min=-0.5, y_max=5.5):
        self.cell = float(cell)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.n_cols = int(math.ceil((x_max - x_min) / cell))
        self.n_rows = int(math.ceil((y_max - y_min) / cell))

    # ------------------------------------------------------------------
    def world_to_cell(self, x, y):
        """(x, y) mundo → (row, col) celda."""
        col = int((float(x) - self.x_min) / self.cell)
        row = int((float(y) - self.y_min) / self.cell)
        col = max(0, min(self.n_cols - 1, col))
        row = max(0, min(self.n_rows - 1, row))
        return row, col

    def cell_center(self, row, col):
        """Centro en coordenadas mundo de la celda (row, col)."""
        x = self.x_min + (col + 0.5) * self.cell
        y = self.y_min + (row + 0.5) * self.cell
        return x, y

    # ------------------------------------------------------------------
    def boxes_to_cells(self, boxes, padding=0.0):
        """Conjunto de (row, col) solapados con la AABB de cada caja."""
        cells = set()
        for box in boxes:
            half = box.side + float(padding)
            xlo, xhi = box.x - half, box.x + half
            ylo, yhi = box.y - half, box.y + half
            r0, c0 = self.world_to_cell(xlo, ylo)
            r1, c1 = self.world_to_cell(xhi, yhi)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    cells.add((r, c))
        return cells

    # ------------------------------------------------------------------
    def astar(self, start_xy, goal_xy, blocked_cells=None):
        """A* 8-vecinos. Devuelve lista ORDENADA de (row, col) o []."""
        blocked = blocked_cells or set()
        sr, sc = self.world_to_cell(*start_xy)
        gr, gc = self.world_to_cell(*goal_xy)

        if (sr, sc) == (gr, gc):
            return [(sr, sc)]

        def _h(r, c):
            dr, dc = abs(r - gr), abs(c - gc)
            return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)

        open_set = []
        heapq.heappush(open_set, (_h(sr, sc), 0.0, sr, sc))
        g_cost = {(sr, sc): 0.0}
        came_from = {}

        dirs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]

        while open_set:
            f, g, r, c = heapq.heappop(open_set)
            if (r, c) == (gr, gc):
                path, cur = [], (r, c)
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append((sr, sc))
                path.reverse()
                return path
            if g > g_cost.get((r, c), float('inf')) + 1e-9:
                continue
            for dr, dc, cost in dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
                    continue
                if (nr, nc) in blocked:
                    continue
                ng = g + cost
                if ng < g_cost.get((nr, nc), float('inf')):
                    g_cost[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_set, (ng + _h(nr, nc), ng, nr, nc))
        return []

    # ------------------------------------------------------------------
    def cells_to_world_polyline(self, cells_ordered):
        """Convierte lista ordenada de (row, col) a lista de (x, y) mundo."""
        return [self.cell_center(r, c) for r, c in cells_ordered]

    # ------------------------------------------------------------------
    def draw(self, ax, paths_ordered=None,
             obstacle_cells=None, blocked_cells=None):
        """Dibuja la cuadricula y las lineas de path en el suelo.

        Render layers (de abajo a arriba):
          zorder 0.2   lineas de cuadricula (gris tenue — marcas del suelo)
          zorder 0.5   lineas de path de cada robot (color vivo, grosor LINE_WIDTH)
          zorder 0.6   celda bloqueada (overlay rojo semitransparente)

        Parameters
        ----------
        ax : matplotlib Axes
        paths_ordered : dict robot -> list of (row, col)  — en orden A*
        obstacle_cells : set of (row, col)  — no se dibuja, solo informativo
        blocked_cells  : set of (row, col)  — se superpone en rojo
        """
        import matplotlib.patches as mpatches

        paths_ordered = paths_ordered or {}
        blocked_cells = blocked_cells or set()

        # 1. Grid lines (marcas del suelo)
        kw = dict(color='#6b8cba', linewidth=1.1, alpha=0.55, zorder=0.2)
        for i in range(self.n_cols + 1):
            ax.axvline(self.x_min + i * self.cell, **kw)
        for j in range(self.n_rows + 1):
            ax.axhline(self.y_min + j * self.cell, **kw)

        # 2. Lineas de path (polylines a traves de los centros de las celdas)
        for robot, cells in paths_ordered.items():
            if not cells:
                continue
            color, alpha = CELL_COLORS.get(robot, ('#AAAAAA', 0.7))
            pts = self.cells_to_world_polyline(cells)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            # Linea principal gruesa
            ax.plot(xs, ys, '-', color=color, linewidth=LINE_WIDTH,
                    alpha=alpha, solid_capstyle='round', solid_joinstyle='round',
                    zorder=0.5)
            # Borde blanco fino para mejor contraste sobre el fondo
            ax.plot(xs, ys, '-', color='white', linewidth=LINE_WIDTH + 2,
                    alpha=alpha * 0.25, solid_capstyle='round',
                    solid_joinstyle='round', zorder=0.45)

        # 3. Celdas bloqueadas (overlay rojo)
        bc_color, bc_alpha = BLOCKED_CELL_COLOR
        for r, c in blocked_cells:
            x = self.x_min + c * self.cell
            y = self.y_min + r * self.cell
            ax.add_patch(mpatches.Rectangle(
                (x, y), self.cell, self.cell,
                facecolor=bc_color, edgecolor='#cc0000',
                linewidth=1.2, alpha=bc_alpha, zorder=0.6))


# ===========================================================================
#  Planificacion global de mision
# ===========================================================================
def compute_mission_paths(coord):
    """Planea en cuadricula 1m los caminos de los 3 robots.

    Resultado:
    ----------
    dict:
        'grid'                : NavigationGrid (1m celdas)
        'paths_ordered'       : dict robot -> list of (row, col) en orden A*
        'paths'               : dict robot -> set of (row, col)
        'obstacle_cells'      : set of (row, col)
        'blocked_cells'       : obstaculos ∩ paths
        'push_order'          : list de nombres de caja en orden de empuje
        'husky_transit_wps'   : list-of-lists de (x,y) para cada transit del Husky
        'anymal_waypoints'    : list de (x,y) para el ANYmal (cell centers + target)
    """
    grid = NavigationGrid()

    boxes = coord.world.boxes
    obs_cells = grid.boxes_to_cells(boxes, padding=0.05)

    # -----------------------------------------------------------------------
    # ANYmal: A* con obstaculos → waypoints mundo
    # -----------------------------------------------------------------------
    anymal_start = (0.0, 0.0)
    anymal_goal  = tuple(coord.ANYMAL_TARGET)

    anymal_path = grid.astar(anymal_start, anymal_goal, obs_cells)
    direct_path_set = set(grid.astar(anymal_start, anymal_goal, set()))

    # Clasificar cajas: bloqueantes de la ruta directa
    blocking_names, non_blocking_names = [], []
    for box in sorted(boxes, key=lambda b: b.x):
        if grid.boxes_to_cells([box]) & direct_path_set:
            blocking_names.append(box.name)
        else:
            non_blocking_names.append(box.name)
    push_order = blocking_names + non_blocking_names

    # Waypoints mundo del ANYmal: centros de celdas + destino exacto
    anymal_waypoints = grid.cells_to_world_polyline(anymal_path)
    if anymal_waypoints and anymal_goal not in anymal_waypoints:
        anymal_waypoints.append(anymal_goal)

    # -----------------------------------------------------------------------
    # Husky: camino completo + transits separados por caja
    # -----------------------------------------------------------------------
    approach_y = 3.20
    name_to_box = {b.name: b for b in boxes}

    husky_path_ordered = []
    husky_transit_wps  = []   # un elemento por caja: waypoints para el transit
    husky_cur_world    = (0.0, 2.0)

    for bname in push_order:
        bx = name_to_box[bname].x
        approach_world = (bx, approach_y)

        # A* transit: current pos → approach col
        transit_cells = grid.astar(husky_cur_world, approach_world, set())

        # Waypoints mundo del transit (cell centers)
        transit_world_wps = grid.cells_to_world_polyline(transit_cells)
        # Append exact approach position as last waypoint
        if not transit_world_wps or transit_world_wps[-1] != approach_world:
            transit_world_wps.append(approach_world)
        husky_transit_wps.append(transit_world_wps)

        # Extender el camino global del Husky
        if husky_path_ordered and transit_cells:
            transit_cells = transit_cells[1:]   # evitar duplicar el nodo de union
        husky_path_ordered.extend(transit_cells)

        # Despues del push el Husky queda en el retreat = approach
        husky_cur_world = approach_world

    # Camino de regreso al inicio (solo visual, no navegado)
    ret_cells = grid.astar(husky_cur_world, (0.0, 2.0), set())
    if ret_cells:
        husky_path_ordered.extend(ret_cells[1:])

    # -----------------------------------------------------------------------
    # xArm: celdas dentro del workspace (disco de reach)
    # -----------------------------------------------------------------------
    bx_arm, by_arm = coord.XARM_BASE_XY
    reach = coord.xarm.L2 + coord.xarm.L3
    xarm_cells = []
    for r in range(grid.n_rows):
        for c in range(grid.n_cols):
            cx, cy = grid.cell_center(r, c)
            if math.hypot(cx - bx_arm, cy - by_arm) <= reach + grid.cell * 0.5:
                xarm_cells.append((r, c))
    # Ordenar por distancia al centro (para dibujar como arco)
    xarm_cells.sort(key=lambda rc: math.hypot(
        grid.cell_center(rc[0], rc[1])[0] - bx_arm,
        grid.cell_center(rc[0], rc[1])[1] - by_arm))

    # -----------------------------------------------------------------------
    # Celdas bloqueadas
    # -----------------------------------------------------------------------
    all_path_cells = set(husky_path_ordered) | set(anymal_path) | set(xarm_cells)
    obs_plain = grid.boxes_to_cells(boxes)
    blocked_cells = obs_plain & all_path_cells

    return {
        'grid':           grid,
        'paths_ordered': {
            'husky':  husky_path_ordered,
            'anymal': anymal_path,
            'xarm':   xarm_cells,
        },
        'paths': {
            'husky':  set(husky_path_ordered),
            'anymal': set(anymal_path),
            'xarm':   set(xarm_cells),
        },
        'obstacle_cells':    obs_cells,
        'blocked_cells':     blocked_cells,
        'push_order':        push_order,
        'husky_transit_wps': husky_transit_wps,
        'anymal_waypoints':  anymal_waypoints,
    }
