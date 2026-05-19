"""Cámara global cenital con pipeline de visión computacional.

Renderiza la escena desde arriba como imagen numpy y aplica CV para:

  1. Detectar la estructura del grid (líneas horizontales y verticales).
  2. Detectar la posición de los robots (blobs de color).
  3. Detectar obstáculos (cajas naranjas) y mapearlos a celdas del grid.
  4. Planificar un path SOBRE el grid (A* 4-vecinos) usando solo lo detectado.
  5. Proveer, celda a celda, la siguiente dirección de movimiento — sin lista
     de waypoints precomputados; cada decisión se toma sobre la imagen actual.

Flujo del pipeline (en coordinator.py):

    cam  = AerialCamera()
    img  = cam.render(grid, boxes, robots)          # renderizado sintético
    obs  = cam.detect_obstacles(img, grid)          # CV: naranjas → celdas
    path = cam.plan_path(grid, start, goal, obs)    # A* 4-vecinos
    ok   = cam.is_path_clear(img, path, grid)       # validación visual
    # En bucle de navegación sin waypoints:
    dr, dc = cam.next_step(img, HUSKY_RGB, goal_cell, grid)
"""

import math
import numpy as np

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ---------------------------------------------------------------------------
#  Colores RGB de cada elemento de la escena
# ---------------------------------------------------------------------------
HUSKY_RGB   = np.array([ 31, 119, 180], dtype=np.uint8)   # azul
ANYMAL_RGB  = np.array([148, 103, 189], dtype=np.uint8)    # morado
BOX_RGB     = np.array([255, 127,  14], dtype=np.uint8)    # naranja
GRID_RGB    = np.array([153, 153, 153], dtype=np.uint8)    # gris
PATH_HUSKY  = np.array([255, 255,   0], dtype=np.uint8)    # amarillo
PATH_ANYMAL = np.array([  0, 255, 255], dtype=np.uint8)    # cian
PATH_COLORS = {'husky': PATH_HUSKY, 'anymal': PATH_ANYMAL}

# 4 direcciones cardinales (dr, dc):  dr>0 = norte (y crece),  dc>0 = este (x crece)
NORTH = ( 1,  0)
SOUTH = (-1,  0)
EAST  = ( 0,  1)
WEST  = ( 0, -1)
DIRS4 = [NORTH, SOUTH, EAST, WEST]
DIR_NAME = {NORTH: 'N', SOUTH: 'S', EAST: 'E', WEST: 'W'}


# ===========================================================================
class AerialCamera:
    """Cámara cenital global: renderiza, detecta y planifica sobre el grid.

    Parameters
    ----------
    xlim : (float, float)   rango x del mundo [m]
    ylim : (float, float)   rango y del mundo [m]
    ppm  : int              píxeles por metro (resolución de la imagen)
    """

    _COLOR_TOL       = 40    # tolerancia detección color (dist. euclídea RGB)
    _OBS_THRESHOLD   = 0.12  # fracción mínima de la celda para contar obstáculo
    _ROBOT_THRESHOLD = 0.04  # ídem para robot

    def __init__(self, xlim=(-1.0, 14.0), ylim=(-1.0, 5.5), ppm=40):
        self.xlim  = xlim
        self.ylim  = ylim
        self.ppm   = int(ppm)
        self.img_w = int((xlim[1] - xlim[0]) * ppm)
        self.img_h = int((ylim[1] - ylim[0]) * ppm)

    # ------------------------------------------------------------------
    #  Utilidades de coordenadas
    # ------------------------------------------------------------------
    def world_to_px(self, x, y):
        """Coordenadas mundo [m] → píxel (px, py) en la imagen."""
        px = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.img_w
        py = (self.ylim[1] - y) / (self.ylim[1] - self.ylim[0]) * self.img_h
        return (int(max(0, min(self.img_w - 1, px))),
                int(max(0, min(self.img_h - 1, py))))

    def px_to_world(self, px, py):
        """Píxel (px, py) → coordenadas mundo (x, y) [m]."""
        x = self.xlim[0] + px / self.img_w * (self.xlim[1] - self.xlim[0])
        y = self.ylim[1] - py / self.img_h * (self.ylim[1] - self.ylim[0])
        return float(x), float(y)

    # ------------------------------------------------------------------
    #  Renderizado sintético de la escena
    # ------------------------------------------------------------------
    def render(self, grid, boxes, robots_pose, paths_ordered=None):
        """Renderiza la escena actual a una imagen RGB (H×W×3 uint8).

        La cámara NO accede a posiciones directamente en el pipeline de
        detección; solo las usa aquí para generar la imagen sintética
        que luego analiza mediante CV.

        Parameters
        ----------
        grid          : NavigationGrid
        boxes         : list of Box
        robots_pose   : dict  {'husky': (x,y,th), 'anymal': (x,y,th), ...}
        paths_ordered : dict  robot → list of (row,col)  [opcional]

        Returns
        -------
        img : ndarray uint8 (H, W, 3)  imagen RGB
        """
        img = np.full((self.img_h, self.img_w, 3), 22, dtype=np.uint8)
        self._draw_grid(img, grid)
        if paths_ordered:
            self._draw_paths(img, grid, paths_ordered)
        self._draw_boxes(img, boxes)
        self._draw_robots(img, robots_pose)
        return img

    # ---- helpers de dibujo -------------------------------------------
    def _draw_grid(self, img, grid):
        c = tuple(int(v) for v in GRID_RGB)
        for col in range(grid.n_cols + 1):
            x_w = grid.x_min + col * grid.cell
            p1 = self.world_to_px(x_w, grid.y_min)
            p2 = self.world_to_px(x_w, grid.y_max)
            if _HAS_CV2:
                _cv2.line(img, p1, p2, c, 1)
            else:
                self._line_np(img, *p1, *p2, c)
        for row in range(grid.n_rows + 1):
            y_w = grid.y_min + row * grid.cell
            p1 = self.world_to_px(grid.x_min, y_w)
            p2 = self.world_to_px(grid.x_max, y_w)
            if _HAS_CV2:
                _cv2.line(img, p1, p2, c, 1)
            else:
                self._line_np(img, *p1, *p2, c)

    def _draw_paths(self, img, grid, paths_ordered):
        colors = {'husky':  PATH_HUSKY,  'anymal': PATH_ANYMAL}
        lw = max(2, self.ppm // 12)
        for robot, cells in paths_ordered.items():
            if not cells or robot not in colors:
                continue
            c = tuple(int(v) for v in colors[robot])
            pts = [self.world_to_px(*grid.cell_center(r, c)) for r, c in cells]
            if _HAS_CV2:
                for i in range(len(pts) - 1):
                    _cv2.line(img, pts[i], pts[i + 1], c, lw)

    def _draw_boxes(self, img, boxes):
        c = tuple(int(v) for v in BOX_RGB)
        for box in boxes:
            hs = box.side
            corners = [
                self.world_to_px(box.x - hs, box.y - hs),
                self.world_to_px(box.x + hs, box.y - hs),
                self.world_to_px(box.x + hs, box.y + hs),
                self.world_to_px(box.x - hs, box.y + hs),
            ]
            if _HAS_CV2:
                pts = np.array(corners, np.int32).reshape(-1, 1, 2)
                _cv2.fillPoly(img, [pts], c)
                _cv2.polylines(img, [pts], True, (0, 0, 0), 1)

    def _draw_robots(self, img, robots_pose):
        cfg = {
            'husky':  (HUSKY_RGB,  0.495, 0.335),
            'anymal': (ANYMAL_RGB, 0.475, 0.275),
        }
        for name, pose in (robots_pose or {}).items():
            if pose is None or name not in cfg:
                continue
            rx, ry, rth = pose
            rgb, hl, hw = cfg[name]
            c = tuple(int(v) for v in rgb)
            cs, sn = math.cos(rth), math.sin(rth)
            corners = []
            for lx, ly in [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]:
                wx = rx + cs * lx - sn * ly
                wy = ry + sn * lx + cs * ly
                corners.append(self.world_to_px(wx, wy))
            if _HAS_CV2:
                pts = np.array(corners, np.int32).reshape(-1, 1, 2)
                _cv2.fillPoly(img, [pts], c)
                _cv2.polylines(img, [pts], True, (255, 255, 255), 1)

    @staticmethod
    def _line_np(img, x1, y1, x2, y2, color):
        """Línea de Bresenham (fallback sin cv2)."""
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        err = dx - dy
        h, w = img.shape[:2]
        x, y = int(x1), int(y1)
        for _ in range(max(dx, dy) + 2):
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = color
            if x == int(x2) and y == int(y2):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x += sx
            if e2 < dx:
                err += dx; y += sy

    # ------------------------------------------------------------------
    #  Detección por visión computacional
    # ------------------------------------------------------------------
    def _color_mask(self, img, rgb_target):
        """Máscara binaria uint8 (255/0) de los píxeles cercanos a rgb_target."""
        diff = img.astype(np.int32) - np.asarray(rgb_target, dtype=np.int32)
        return ((diff ** 2).sum(axis=2) < self._COLOR_TOL ** 2).astype(np.uint8) * 255

    def _cell_coverage(self, mask, grid, r, c, margin=0.38):
        """Fracción del área central de la celda (r,c) cubierta por mask."""
        cx, cy = grid.cell_center(r, c)
        m = margin * grid.cell
        # world_to_px flips y; py_top < py_bot en imagen
        px1, py_bot = self.world_to_px(cx - m, cy - m)
        px2, py_top = self.world_to_px(cx + m, cy + m)
        px1, px2 = sorted([px1, px2]); py1, py2 = sorted([py_top, py_bot])
        px1 = max(0, px1); px2 = min(self.img_w, px2 + 1)
        py1 = max(0, py1); py2 = min(self.img_h, py2 + 1)
        if px2 <= px1 or py2 <= py1:
            return 0.0
        region = mask[py1:py2, px1:px2]
        return float(region.sum()) / max(255 * region.size, 1)

    def detect_obstacles(self, img, grid):
        """Detecta celdas con obstáculos mediante segmentación de color naranja.

        Returns
        -------
        set of (row, col)
        """
        mask = self._color_mask(img, BOX_RGB)
        return {(r, c)
                for r in range(grid.n_rows)
                for c in range(grid.n_cols)
                if self._cell_coverage(mask, grid, r, c) > self._OBS_THRESHOLD}

    def detect_robot_cell(self, img, robot_rgb, grid):
        """Detecta en qué celda está el robot por centroide del blob de color.

        Returns (row, col) o None si no se detecta.
        """
        mask = self._color_mask(img, np.asarray(robot_rgb, np.uint8))
        if mask.sum() == 0:
            return None
        ys, xs = np.where(mask > 0)
        x_w, y_w = self.px_to_world(float(xs.mean()), float(ys.mean()))
        return grid.world_to_cell(x_w, y_w)

    # ------------------------------------------------------------------
    #  Planificación sobre el grid detectado (sin waypoints explícitos)
    # ------------------------------------------------------------------
    def plan_path(self, grid, start_xy, goal_xy, obstacle_cells=None):
        """A* 4-vecinos sobre el grid detectado por la cámara.

        Cada llamada es independiente: no mantiene estado de paths previos.
        """
        return grid.astar(start_xy, goal_xy, obstacle_cells or set())

    def is_path_clear(self, img, path_cells, grid):
        """Valida visualmente que ninguna celda del path está bloqueada."""
        obs = self.detect_obstacles(img, grid)
        return not any(cell in obs for cell in path_cells)

    def detect_path_blockages(self, img, path_cells, grid):
        """Retorna el subconjunto de path_cells actualmente bloqueadas."""
        obs = self.detect_obstacles(img, grid)
        return {cell for cell in path_cells if cell in obs}

    # ------------------------------------------------------------------
    #  Decisión de siguiente paso — núcleo del pipeline sin waypoints
    # ------------------------------------------------------------------
    def next_step(self, img, robot_rgb, goal_cell, grid):
        """Siguiente dirección (dr, dc) decidida exclusivamente por la cámara.

        Pipeline:
          1. Detecta la celda actual del robot en la imagen.
          2. Detecta celdas obstruidas en la imagen.
          3. Corre A* 4-vecinos con esa información.
          4. Retorna (dr, dc) del primer paso del camino.

        NO recibe lista de waypoints — cada llamada re-analiza la imagen.

        Returns
        -------
        (dr, dc) : primer paso hacia el objetivo
                   (0, 0)  si ya está en la celda destino o no hay camino
        """
        current = self.detect_robot_cell(
            img, np.asarray(robot_rgb, np.uint8), grid)
        if current is None or current == goal_cell:
            return (0, 0)

        obs = self.detect_obstacles(img, grid)
        obs.discard(current)   # la celda donde está el robot no es obstáculo

        path = grid.astar(grid.cell_center(*current),
                          grid.cell_center(*goal_cell), obs)
        if len(path) < 2:
            return (0, 0)
        return (path[1][0] - path[0][0], path[1][1] - path[0][1])

    # ------------------------------------------------------------------
    #  Análisis completo de la escena
    # ------------------------------------------------------------------
    def analyze_scene(self, img, grid):
        """Resumen estructurado del estado actual detectado por la cámara.

        Returns
        -------
        dict con 'obstacle_cells', 'husky_cell', 'anymal_cell'
        """
        return {
            'obstacle_cells': self.detect_obstacles(img, grid),
            'husky_cell':     self.detect_robot_cell(img, HUSKY_RGB,  grid),
            'anymal_cell':    self.detect_robot_cell(img, ANYMAL_RGB, grid),
        }
