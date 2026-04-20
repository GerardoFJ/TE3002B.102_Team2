"""Husky A200 empuja cajas grandes fuera de un corredor 6x2 m.

Contiene:

    - Box, CorridorWorld : modelos del entorno (cajas AABB + corredor).
    - Lidar2D            : sensor 2D simulado (raycaster contra AABBs).
    - HuskyPusher        : controlador alto nivel que planea y ejecuta el
                           empuje secuencial de las 3 cajas usando control
                           skid-steer con compensacion de deslizamiento.

Cubre los requisitos de la Fase 1 del mini reto:
    * Localizacion de las 3 cajas via LiDAR 2D simulado.
    * Planificacion de trayectoria para empujar cada caja.
    * Skid-steer con compensacion de slip (factor s del terreno).
    * Reporte en tiempo real de (v, omega) comandados y medidos.
    * Condicion de exito: las 3 cajas fuera del rectangulo del corredor.
"""

import math

import numpy as np

from .robots_base import HuskyA200


# ===========================================================================
#  Mundo del reto
# ===========================================================================
class Box:
    """Caja rigida 2D con masa, modelada como AABB cuadrada.

    Parameters
    ----------
    x, y : float
        Centro de la caja en el marco mundo [m].
    side : float
        Half-side de la caja [m] (la caja ocupa [x-side, x+side] x ...).
    mass : float
        Masa de la caja [kg], usada solo a efectos de reporte.
    name : str, opcional
        Etiqueta legible (B1, B2, B3).
    """

    def __init__(self, x, y, side=0.30, mass=15.0, name="B"):
        self.x = float(x)
        self.y = float(y)
        self.side = float(side)
        self.mass = float(mass)
        self.name = name

    @property
    def position(self):
        return np.array([self.x, self.y])

    def aabb(self):
        """Retorna (xmin, xmax, ymin, ymax)."""
        return (self.x - self.side, self.x + self.side,
                self.y - self.side, self.y + self.side)


class CorridorWorld:
    """Geometria del escenario almacen del mini reto.

    El corredor es un rectangulo de 6x2 m por donde luego pasara el
    ANYmal. Empezamos con 3 cajas pesadas dentro de el. La condicion
    de exito de la fase 1 es que las 3 queden fuera del rectangulo.
    """

    def __init__(self,
                 corridor=(3.0, 9.0, 1.0, 3.0),
                 box_positions=None):
        # corridor = (xmin, xmax, ymin, ymax)
        self.corridor = {
            'xmin': float(corridor[0]),
            'xmax': float(corridor[1]),
            'ymin': float(corridor[2]),
            'ymax': float(corridor[3]),
        }
        if box_positions is None:
            box_positions = [(4.5, 2.0), (6.0, 2.0), (7.5, 2.0)]
        self.boxes = [Box(x, y, name=f"B{i+1}")
                      for i, (x, y) in enumerate(box_positions)]

    # ------------------------------------------------------------------
    def box_in_corridor(self, box):
        """True si el centro de la caja esta dentro del rectangulo."""
        c = self.corridor
        return (c['xmin'] <= box.x <= c['xmax'] and
                c['ymin'] <= box.y <= c['ymax'])

    def boxes_in_corridor(self):
        return [b for b in self.boxes if self.box_in_corridor(b)]

    def all_clear(self):
        """True si las 3 cajas estan fuera del corredor."""
        return all(not self.box_in_corridor(b) for b in self.boxes)


# ===========================================================================
#  LiDAR 2D simulado
# ===========================================================================
class Lidar2D:
    """LiDAR 2D barato: rayos uniformes en un FOV centrado en el frente.

    Parameters
    ----------
    n_beams : int
        Numero de rayos.
    max_range : float
        Alcance maximo del sensor [m].
    fov_deg : float
        Campo de vision total [grados].
    """

    def __init__(self, n_beams=181, max_range=8.0, fov_deg=270.0):
        self.n_beams = int(n_beams)
        self.max_range = float(max_range)
        self.fov_rad = math.radians(fov_deg)

    # ------------------------------------------------------------------
    def scan(self, pose, boxes):
        """Devuelve (angles, ranges) en el marco del sensor.

        ``angles`` esta en el marco mundo (ya rotados por el yaw del
        husky); ``ranges`` es la distancia al primer obstaculo o
        max_range si no hay impacto.
        """
        x0, y0, theta = pose
        half = self.fov_rad / 2.0
        local_angles = np.linspace(-half, half, self.n_beams)
        world_angles = local_angles + theta
        ranges = np.full(self.n_beams, self.max_range)
        for i, ang in enumerate(world_angles):
            dx = math.cos(ang)
            dy = math.sin(ang)
            best = self.max_range
            for box in boxes:
                t = self._ray_aabb(x0, y0, dx, dy, box, best)
                if t is not None and 0.0 < t < best:
                    best = t
            ranges[i] = best
        return world_angles, ranges

    @staticmethod
    def _ray_aabb(x0, y0, dx, dy, box, t_far):
        """Interseccion rayo-AABB. Retorna distancia o None."""
        xmin, xmax, ymin, ymax = box.aabb()
        # Slabs en x
        if abs(dx) < 1e-12:
            if x0 < xmin or x0 > xmax:
                return None
            tx_lo, tx_hi = -math.inf, math.inf
        else:
            tx1 = (xmin - x0) / dx
            tx2 = (xmax - x0) / dx
            tx_lo, tx_hi = min(tx1, tx2), max(tx1, tx2)
        # Slabs en y
        if abs(dy) < 1e-12:
            if y0 < ymin or y0 > ymax:
                return None
            ty_lo, ty_hi = -math.inf, math.inf
        else:
            ty1 = (ymin - y0) / dy
            ty2 = (ymax - y0) / dy
            ty_lo, ty_hi = min(ty1, ty2), max(ty1, ty2)

        t_lo = max(tx_lo, ty_lo)
        t_hi = min(tx_hi, ty_hi)
        if t_hi < 0 or t_lo > t_hi or t_lo > t_far:
            return None
        return max(t_lo, 0.0)


# ===========================================================================
#  Detector de cajas a partir de scans
# ===========================================================================
def detect_boxes_from_scan(angles, ranges, max_range,
                           cluster_gap=0.30):
    """Agrupa puntos cercanos del scan y devuelve un centro por cluster.

    Algoritmo: convierte (angle, range) a (x, y) en marco mundo, descarta
    puntos en max_range, recorre el barrido angular y separa clusters
    cuando dos rayos consecutivos saltan mas de ``cluster_gap`` metros.
    Cada cluster reporta su centroide.

    Esto basta para que el Husky "vea" las 3 cajas grandes en el
    corredor. No es robusto para escenas complejas, pero es lo que pide
    el mini reto.
    """
    points = []
    valid = ranges < max_range - 1e-3
    for i, ok in enumerate(valid):
        if not ok:
            points.append(None)
            continue
        r = ranges[i]
        a = angles[i]
        points.append((r * math.cos(a), r * math.sin(a)))

    clusters = []
    current = []
    last_pt = None
    for p in points:
        if p is None:
            if current:
                clusters.append(current)
                current = []
            last_pt = None
            continue
        if last_pt is None:
            current = [p]
        else:
            d = math.hypot(p[0] - last_pt[0], p[1] - last_pt[1])
            if d > cluster_gap:
                clusters.append(current)
                current = [p]
            else:
                current.append(p)
        last_pt = p
    if current:
        clusters.append(current)

    centroids = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        xs = np.array([p[0] for p in cluster])
        ys = np.array([p[1] for p in cluster])
        centroids.append((float(xs.mean()), float(ys.mean())))
    return centroids


# ===========================================================================
#  Controlador go-to-pose con compensacion de slip
# ===========================================================================
class HuskyGoToPoseController:
    """Lleva al Husky a una pose 2D usando un esquema 'turn then drive'.

    Si el error angular es grande, gira en el sitio. Si el error angular
    es pequeno, avanza con velocidad proporcional al error de distancia.
    Las velocidades de cuerpo (v, omega) se traducen a velocidades de
    rueda con compensacion de slip por terreno.
    """

    def __init__(self, husky,
                 k_v=0.7, k_omega=2.0,
                 v_max=0.6, omega_max=1.0,
                 angle_threshold_deg=10.0):
        self.husky = husky
        self.k_v = float(k_v)
        self.k_omega = float(k_omega)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.angle_threshold = math.radians(angle_threshold_deg)

    def step_to(self, target_xy, dt):
        """Calcula y aplica un paso de control hacia ``target_xy``.

        Returns
        -------
        dict con claves 'v_cmd', 'omega_cmd', 'v_real', 'omega_real',
        'dist', 'yaw_err'.
        """
        x, y, th = self.husky.get_pose()
        dx = target_xy[0] - x
        dy = target_xy[1] - y
        dist = math.hypot(dx, dy)

        target_yaw = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(target_yaw - th),
                             math.cos(target_yaw - th))

        if abs(yaw_err) > self.angle_threshold:
            # Girar en el sitio
            v_cmd = 0.0
            omega_cmd = float(np.clip(self.k_omega * yaw_err,
                                      -self.omega_max, self.omega_max))
        else:
            # Avanzar; reduce v si todavia hay error angular residual
            v_cmd = float(np.clip(self.k_v * dist, 0.0, self.v_max))
            v_cmd *= max(0.0, math.cos(yaw_err))
            omega_cmd = float(np.clip(self.k_omega * yaw_err,
                                      -self.omega_max, self.omega_max))

        # Compensacion de slip + actuacion
        wR1, wR2, wL1, wL2 = self.husky.inverse_kinematics(
            v_cmd, omega_cmd, compensate_slip=True)
        v_real, omega_real = self.husky.forward_kinematics(wR1, wR2, wL1, wL2)
        self.husky.update_pose(v_real, omega_real, dt)

        return {
            'v_cmd': v_cmd, 'omega_cmd': omega_cmd,
            'v_real': v_real, 'omega_real': omega_real,
            'dist': dist, 'yaw_err': yaw_err,
        }


# ===========================================================================
#  HuskyPusher de alto nivel
# ===========================================================================
class HuskyPusher:
    """Empuja secuencialmente todas las cajas del corredor.

    El plan por caja es siempre el mismo:
        1. APPROACH: ir a un punto al norte de la caja con holgura.
        2. ALIGN  : (incluido en go-to-pose) girar para mirar al sur.
        3. PUSH   : avanzar hacia el sur arrastrando la caja consigo.
        4. RETREAT: regresar al norte para no estorbar el corredor.

    Por simplicidad la "fisica del contacto" es ideal: durante PUSH la
    caja se desplaza en lockstep con el centro del Husky (con un offset
    fijo).

    Parameters
    ----------
    husky : HuskyA200
    world : CorridorWorld
    lidar : Lidar2D, opcional
    push_distance : float
        Holgura al sur del corredor a la que se deja la caja [m].
    """

    def __init__(self, husky, world, lidar=None,
                 push_distance=0.6,
                 contact_offset=0.80,
                 approach_buffer=0.40):
        self.husky = husky
        self.world = world
        self.lidar = lidar or Lidar2D()
        self.controller = HuskyGoToPoseController(husky)
        self.push_distance = float(push_distance)
        # Distancia centro_husky - centro_caja en el momento del contacto.
        # Debe ser >= half_husky + half_box (~0.5 + 0.3 = 0.8 por defecto).
        self.contact_offset = float(contact_offset)
        # Holgura adicional entre el approach y el contacto.
        self.approach_buffer = float(approach_buffer)

    # ------------------------------------------------------------------
    def scan(self):
        """Hace un scan completo y devuelve angulos, rangos y centroides
        detectados en marco mundo.
        """
        angles, ranges = self.lidar.scan(self.husky.get_pose(),
                                         self.world.boxes)
        local_centroids = detect_boxes_from_scan(angles, ranges,
                                                 self.lidar.max_range)
        # Pasar de marco husky a marco mundo
        x0, y0, _ = self.husky.get_pose()
        world_centroids = [(x0 + cx, y0 + cy) for cx, cy in local_centroids]
        return angles, ranges, world_centroids

    # ------------------------------------------------------------------
    def plan_push(self, box):
        """Genera waypoints (approach, contact, push_end, retreat).

        Las cuatro etapas son:
            approach -> punto al norte del contacto, con holgura
            contact  -> husky justo en contacto con la caja (sin moverla)
            push_end -> husky al final del empuje (caja al sur del corredor)
            retreat  -> husky regresa al approach para no estorbar
        """
        c = self.world.corridor
        # Y final que queremos para el centro de la caja (al sur del corredor)
        target_box_y = c['ymin'] - self.push_distance - box.side
        # Posiciones del husky correspondientes
        contact_husky_y = box.y + self.contact_offset
        approach_husky_y = contact_husky_y + self.approach_buffer
        push_end_husky_y = target_box_y + self.contact_offset

        approach = np.array([box.x, approach_husky_y])
        contact = np.array([box.x, contact_husky_y])
        push_end = np.array([box.x, push_end_husky_y])
        retreat = approach.copy()
        return approach, contact, push_end, retreat

    # ------------------------------------------------------------------
    def _run_segment(self, target_xy, dt, log,
                     contact_box=None, max_steps=2000, eps=0.08):
        """Conduce el husky hasta ``target_xy``, opcionalmente arrastrando
        ``contact_box``."""
        offset = None
        if contact_box is not None:
            x, y, _ = self.husky.get_pose()
            offset = np.array([contact_box.x - x, contact_box.y - y])

        for _ in range(max_steps):
            x, y, _ = self.husky.get_pose()
            if math.hypot(target_xy[0] - x, target_xy[1] - y) < eps:
                break
            info = self.controller.step_to(target_xy, dt)
            # Log
            log['t'].append(log['t'][-1] + dt if log['t'] else 0.0)
            log['x'].append(self.husky.x)
            log['y'].append(self.husky.y)
            log['theta'].append(self.husky.theta)
            log['v_cmd'].append(info['v_cmd'])
            log['omega_cmd'].append(info['omega_cmd'])
            log['v_real'].append(info['v_real'])
            log['omega_real'].append(info['omega_real'])
            for b in self.world.boxes:
                log['boxes'][b.name].append((b.x, b.y))
            # Mover la caja en contacto
            if contact_box is not None:
                contact_box.x = self.husky.x + offset[0]
                contact_box.y = self.husky.y + offset[1]

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_log(boxes):
        return {
            't': [], 'x': [], 'y': [], 'theta': [],
            'v_cmd': [], 'omega_cmd': [],
            'v_real': [], 'omega_real': [],
            'boxes': {b.name: [] for b in boxes},
            'phase': [],
        }

    # ------------------------------------------------------------------
    def push_box(self, box, dt=0.05, log=None):
        """Ejecuta el plan completo para una caja. Acumula en ``log``.

        El segmento PUSH calcula su offset husky->caja al inicio, cuando
        el husky ya esta en CONTACT (a contact_offset al norte de la caja),
        de modo que el offset queda exactamente (0, -contact_offset) y la
        caja sigue al husky pegada por el lado norte.
        """
        if log is None:
            log = self._empty_log(self.world.boxes)
        approach, contact, push_end, retreat = self.plan_push(box)

        n0 = len(log['t'])
        self._run_segment(approach, dt, log, contact_box=None)
        self._run_segment(contact, dt, log, contact_box=None)
        self._run_segment(push_end, dt, log, contact_box=box)
        self._run_segment(retreat, dt, log, contact_box=None)
        log['phase'].extend([box.name] * (len(log['t']) - n0))
        return log

    # ------------------------------------------------------------------
    def clear_corridor(self, dt=0.05):
        """Empuja todas las cajas que esten dentro del corredor."""
        log = self._empty_log(self.world.boxes)
        # Procesa de oeste a este (orden natural de avance del husky)
        boxes_to_push = sorted(self.world.boxes_in_corridor(),
                               key=lambda b: b.x)
        for box in boxes_to_push:
            self.push_box(box, dt, log)
        log['success'] = self.world.all_clear()
        return log


# ===========================================================================
#  Demo rapida
# ===========================================================================
if __name__ == "__main__":
    print("Demo: Husky despeja un corredor con 3 cajas")
    husky = HuskyA200()
    husky.set_terrain("grass")             # con slip 0.85
    husky.reset(x=0.0, y=2.0, theta=0.0)
    world = CorridorWorld()
    pusher = HuskyPusher(husky, world)

    print("  Cajas iniciales en el corredor:",
          [b.name for b in world.boxes_in_corridor()])
    print("  Terreno              :", husky.terrain)

    log = pusher.clear_corridor(dt=0.05)
    print(f"  Pasos integrados     : {len(log['t'])}")
    print(f"  Exito (corredor libre): {log['success']}")
    for b in world.boxes:
        print(f"    {b.name}: ({b.x:.2f}, {b.y:.2f}) "
              f"-> {'fuera' if not world.box_in_corridor(b) else 'dentro'}")
    if log['v_cmd']:
        v_cmd = np.array(log['v_cmd'])
        v_real = np.array(log['v_real'])
        print(f"  v_cmd promedio       : {v_cmd.mean():+.3f} m/s")
        print(f"  v_real promedio      : {v_real.mean():+.3f} m/s")
