"""xArm 6 simplificado para el mini reto.

Modelo del brazo de 6 DoF de UFactory (xArm 6), con dimensiones extraidas
del URDF oficial de ``xarm_description`` (repo xArm-Developer/xarm_ros2,
rama humble). Los offsets del YAML de cinematica default son:

    joint1: z = 0.267                         (base -> hombro)
    joint3: (x=0.0535, y=-0.2845)             (brazo superior)
    joint4: (x=0.0775, y=0.3425)              (antebrazo)
    joint6: (x=0.076,  y=0.097)               (muneca + efector)

Para la sim 2D nos alcanza con un modelo planar equivalente:

    L2 = sqrt(0.0535^2 + 0.2845^2) ~ 0.289 m  (longitud efectiva brazo)
    L3 = sqrt(0.0775^2 + 0.3425^2) + 0.097 ~ 0.449 m (antebrazo + muneca)
    h0 = 0.267 m                               (altura del hombro)

El reach horizontal maximo es ~L2 + L3 = 0.738 m, consistente con los
700 mm del xArm 6 real.

Controles que expone este modulo
--------------------------------
- ``forward_kinematics(q)`` con q = (q1, q2, q3, q4) -> (x, y, z) del TCP
  en el marco base del brazo. Se usa un sub-modelo de 4 DoF funcional:
    q1 : yaw de la base (azimut)
    q2 : pitch del hombro
    q3 : pitch del codo
    q4 : pitch de la muneca (orientar el gripper)
  (los joints 5 y 6 del xArm real son para orientacion fina del efector
  y no se modulan en la sim; quedan en home.)

- ``inverse_kinematics(p, wrist_pitch=-pi/2)`` devuelve q1..q4 para llegar
  a ``p=(x,y,z)`` con el efector apuntando hacia abajo (pitch fijo).

- ``pick_place_cartesian_path(p_start, p_pick, p_place)`` genera una
  trayectoria tipo *waypoint* (approach -> pick -> lift -> transit ->
  approach_place -> place -> lift) en coordenadas cartesianas, util para
  la sim 2D.

- ``draw_links_2d(q)`` devuelve la polilinea proyectada en el plano xy
  (top-view) de base -> hombro -> codo -> muneca -> TCP, lista para que
  ``sim.py`` la dibuje como un brazo que se ve desde arriba.
"""

import math

import numpy as np


class XArm:
    """Brazo xArm 6 simplificado (4 DoF efectivos) para el mini reto.

    Parameters
    ----------
    base_xy : tuple(float, float)
        Posicion de la base en el marco mundo [m].
    base_yaw : float
        Orientacion del eje +x del brazo respecto al mundo [rad].
    h0 : float
        Altura del hombro sobre la base [m] (default 0.267 = xArm 6).
    L2 : float
        Longitud efectiva del brazo superior [m] (default 0.289 = xArm 6).
    L3 : float
        Longitud efectiva del antebrazo + muneca [m] (default 0.449).
    """

    def __init__(self,
                 base_xy=(11.50, 3.35),
                 base_yaw=math.pi / 2,
                 h0=0.267,
                 L2=0.289,
                 L3=0.449):
        self.base_xy = np.array(base_xy, dtype=float)
        self.base_yaw = float(base_yaw)
        self.h0 = float(h0)
        self.L2 = float(L2)
        self.L3 = float(L3)

        # Limites articulares aproximados del xArm 6 (rad).
        # Valores tipicos del URDF: j1 +-6.28, j2 [-2.06, 2.09],
        # j3 [-0.19, 3.93] (aprox). Aqui simplificado.
        self.q_min = np.array([-math.pi,      -2.06, -math.pi,  -math.pi])
        self.q_max = np.array([ math.pi,       2.09,  math.pi,   math.pi])

        # Home pose: brazo recogido, apuntando hacia adelante-abajo.
        self.q_home = np.array([0.0, 0.4, -1.8, -0.3])
        self.q = self.q_home.copy()

    # ------------------------------------------------------------------
    #  Utilidades de marco mundo <-> marco base del brazo
    # ------------------------------------------------------------------
    def world_from_arm(self, p_arm):
        """Convierte un punto del marco del brazo al marco mundo.

        El marco del brazo tiene su origen en la base, con +x rotado por
        ``base_yaw`` alrededor del eje z del mundo.
        """
        p_arm = np.asarray(p_arm, dtype=float)
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]])
        origin = np.array([self.base_xy[0], self.base_xy[1], 0.0])
        return R @ p_arm + origin

    def arm_from_world(self, p_world):
        """Inversa de :meth:`world_from_arm`."""
        p_world = np.asarray(p_world, dtype=float)
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        Rt = np.array([[ c,  s, 0.0],
                       [-s,  c, 0.0],
                       [0.0, 0.0, 1.0]])
        origin = np.array([self.base_xy[0], self.base_xy[1], 0.0])
        return Rt @ (p_world - origin)

    # ------------------------------------------------------------------
    #  Cinematica directa (marco base del brazo)
    # ------------------------------------------------------------------
    def forward_kinematics(self, q=None):
        """FK: q=(q1,q2,q3,q4) -> (x,y,z) del TCP en marco base del brazo.

        q1 : yaw de la base
        q2 : pitch del hombro (q2=0 -> brazo vertical; q2>0 -> hacia +x)
        q3 : pitch del codo (q3=0 -> brazo extendido)
        q4 : pitch de la muneca (ajusta la orientacion del gripper, solo
             afecta la posicion del TCP si modelamos longitud extra;
             aqui suponemos TCP en el centro de la muneca, asi que q4
             no afecta la posicion).
        """
        if q is not None:
            self.q = np.asarray(q, dtype=float).copy()
        q1, q2, q3, _q4 = self.q

        # Plano vertical que contiene al brazo (rotado q1 alrededor de z).
        # En ese plano, el "reach horizontal" desde el hombro es:
        #   r = L2*sin(q2) + L3*sin(q2+q3)
        #   h_rel = L2*cos(q2) + L3*cos(q2+q3)   (altura por encima del hombro)
        # q2 = 0 -> brazo apuntando hacia arriba; crecer q2 empuja hacia +x.
        r = self.L2 * math.sin(q2) + self.L3 * math.sin(q2 + q3)
        h_rel = self.L2 * math.cos(q2) + self.L3 * math.cos(q2 + q3)

        x = r * math.cos(q1)
        y = r * math.sin(q1)
        z = self.h0 + h_rel
        return np.array([x, y, z])

    # ------------------------------------------------------------------
    #  Cinematica inversa
    # ------------------------------------------------------------------
    def inverse_kinematics(self, p_des_arm, elbow_up=True):
        """IK para un punto en el marco base del brazo.

        Resuelve (q1, q2, q3) via IK planar 2-link en el plano vertical
        del brazo y deja q4 = 0 (el TCP se asume en el centro de la
        muneca; para el sim basta). El gripper "apunta hacia abajo" por
        convencion, lo cual equivale a q4 = -(q2+q3).

        Parameters
        ----------
        p_des_arm : array (3,)
            Punto objetivo (x, y, z) en el marco base del brazo [m].
        elbow_up : bool
            True -> codo hacia arriba (configuracion natural).

        Returns
        -------
        np.ndarray shape (4,)
            q1..q4.

        Raises
        ------
        ValueError
            Si el punto esta fuera del workspace.
        """
        x, y, z = p_des_arm

        # q1: azimut
        q1 = math.atan2(y, x)

        # Pasamos al plano del brazo (r, h_rel)
        r = math.hypot(x, y)
        h_rel = z - self.h0
        d_sq = r * r + h_rel * h_rel
        d = math.sqrt(d_sq)

        d_max = self.L2 + self.L3
        d_min = abs(self.L2 - self.L3)
        if d > d_max + 1e-6 or d < d_min - 1e-6:
            raise ValueError(
                f"XArm IK: punto fuera de workspace "
                f"(d={d:.3f}, rango=[{d_min:.3f}, {d_max:.3f}] m)"
            )

        # Ley de cosenos para q3 (angulo interno codo).
        # FK: r = L2 sin q2 + L3 sin(q2+q3), h_rel = L2 cos q2 + L3 cos(q2+q3).
        # => d^2 = L2^2 + L3^2 + 2*L2*L3*cos(q3).
        cos_q3 = (d_sq - self.L2**2 - self.L3**2) / (2.0 * self.L2 * self.L3)
        cos_q3 = max(-1.0, min(1.0, cos_q3))
        q3 = -math.acos(cos_q3) if elbow_up else math.acos(cos_q3)

        # q2: angulo entre el vector (r, h_rel) y la vertical +z del hombro,
        # menos el offset que introduce el codo.
        alpha = math.atan2(r, h_rel)
        beta = math.atan2(self.L3 * math.sin(q3),
                          self.L2 + self.L3 * math.cos(q3))
        q2 = alpha - beta

        # q4 para que el gripper apunte hacia abajo (eje -z mundo).
        q4 = -(q2 + q3)

        q = np.array([q1, q2, q3, q4])
        return q

    def ik_world(self, p_des_world, elbow_up=True):
        """Conveniencia: IK con target expresado en el marco mundo."""
        return self.inverse_kinematics(
            self.arm_from_world(p_des_world), elbow_up=elbow_up)

    # ------------------------------------------------------------------
    #  Trayectorias
    # ------------------------------------------------------------------
    def pick_place_cartesian_path(self, p_pick_world, p_place_world,
                                  approach_height=0.18, n_seg=12):
        """Genera la trayectoria cartesiana del TCP en marco mundo.

        Secuencia de waypoints:

            home_tcp -> p_pick + h_up -> p_pick -> p_pick + h_up ->
            p_place + h_up -> p_place -> p_place + h_up -> home_tcp

        Cada tramo se subdivide en ``n_seg`` puntos para tener animacion
        fluida. Devuelve un ``np.ndarray`` shape (N, 3).
        """
        h = float(approach_height)
        p_pick = np.asarray(p_pick_world, dtype=float)
        p_place = np.asarray(p_place_world, dtype=float)

        # Home en marco mundo: el TCP en la posicion home del brazo.
        home_arm = self.forward_kinematics(self.q_home)
        home_w = self.world_from_arm(home_arm)

        above_pick = p_pick + np.array([0.0, 0.0, h])
        above_place = p_place + np.array([0.0, 0.0, h])

        waypoints = [home_w, above_pick, p_pick, above_pick,
                     above_place, p_place, above_place, home_w]
        path = []
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            for s in np.linspace(0.0, 1.0, n_seg, endpoint=False):
                path.append((1 - s) * a + s * b)
        path.append(waypoints[-1])
        return np.array(path)

    def joint_path_from_cartesian(self, cart_path_world, elbow_up=True):
        """Convierte una trayectoria cartesiana (mundo) a trayectoria
        articular via IK punto a punto. Filas fuera de workspace se
        dejan con el ultimo q valido.
        """
        qs = np.zeros((len(cart_path_world), 4))
        q_last = self.q_home.copy()
        for i, p_w in enumerate(cart_path_world):
            try:
                q_last = self.ik_world(p_w, elbow_up=elbow_up)
            except ValueError:
                pass  # mantiene q_last
            qs[i] = q_last
        return qs

    # ------------------------------------------------------------------
    #  Render helpers
    # ------------------------------------------------------------------
    def link_points_world(self, q=None):
        """Devuelve las 4 articulaciones + TCP en el mundo (array (5,3)).

        Puntos: base, hombro, codo, muneca, TCP. Utiles para dibujar el
        brazo tanto en top view (xy) como en side view (xz).
        """
        if q is None:
            q = self.q
        q1, q2, q3, _q4 = q

        # En el marco base del brazo
        p_base = np.array([0.0, 0.0, 0.0])
        p_shoulder = np.array([0.0, 0.0, self.h0])

        # Direccion del plano del brazo: (cos q1, sin q1, 0)
        dir_h = np.array([math.cos(q1), math.sin(q1), 0.0])
        dir_z = np.array([0.0, 0.0, 1.0])

        # Codo: el segmento del hombro al codo tiene longitud L2; q2 es
        # el pitch desde +z hacia dir_h.
        s2, c2 = math.sin(q2), math.cos(q2)
        elbow_offset = self.L2 * (s2 * dir_h + c2 * dir_z)
        p_elbow = p_shoulder + elbow_offset

        # Muneca: el antebrazo tiene longitud L3; pitch total q2+q3.
        s23, c23 = math.sin(q2 + q3), math.cos(q2 + q3)
        wrist_offset = self.L3 * (s23 * dir_h + c23 * dir_z)
        p_wrist = p_elbow + wrist_offset

        # TCP coincide con la muneca en el modelo simplificado
        p_tcp = p_wrist.copy()

        pts_arm = np.stack([p_base, p_shoulder, p_elbow, p_wrist, p_tcp])
        # A marco mundo
        return np.array([self.world_from_arm(p) for p in pts_arm])

    # ------------------------------------------------------------------
    def reset(self, q=None):
        if q is None:
            q = self.q_home
        self.q = np.asarray(q, dtype=float).copy()
