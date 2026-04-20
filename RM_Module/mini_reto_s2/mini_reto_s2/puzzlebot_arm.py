"""Mini brazo de 3 DoF montado sobre un PuzzleBot.

Geometria
---------
El brazo tiene una base rotacional (q1) que gira alrededor del eje vertical
(z) y dos articulaciones (q2, q3) que actuan en el plano vertical que
contiene al brazo. Los eslabones son:

    l1 : altura fija del hombro sobre la base [m]
    l2 : longitud del brazo superior (humero)  [m]
    l3 : longitud del antebrazo                [m]

Convenciones de angulos
-----------------------
    q1 = 0  -> el brazo apunta hacia +x del marco base
    q2 = 0  -> el brazo superior es horizontal (paralelo al suelo)
    q3 = 0  -> antebrazo alineado con el brazo superior
    q2 > 0  -> el brazo superior apunta hacia arriba
    q3 < 0  -> "codo arriba" (elbow up); el antebrazo cae hacia el suelo

Cinematica directa
------------------
    r = l2*cos(q2) + l3*cos(q2+q3)         (reach horizontal)
    x = r*cos(q1)
    y = r*sin(q1)
    z = l1 + l2*sin(q2) + l3*sin(q2+q3)

El Jacobiano analitico se deriva en :meth:`PuzzleBotArm.jacobian`.
"""

import numpy as np


class PuzzleBotArm:
    """Mini brazo planar de 3 DoF montado sobre un PuzzleBot.

    Parameters
    ----------
    l1, l2, l3 : float
        Longitudes de los eslabones [m] (altura del hombro, brazo, antebrazo).
    q_min, q_max : np.ndarray, opcional
        Limites articulares en rad. Por defecto rangos amplios pero no
        infinitos para que la IK quede acotada.
    """

    def __init__(self, l1=0.10, l2=0.08, l3=0.06,
                 q_min=None, q_max=None):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.l3 = float(l3)

        # Limites articulares por defecto: base 360, hombro y codo +-150 deg
        self.q_min = np.array([-np.pi,        -2.6, -2.6]) if q_min is None \
            else np.asarray(q_min, dtype=float)
        self.q_max = np.array([ np.pi,         2.6,  2.6]) if q_max is None \
            else np.asarray(q_max, dtype=float)

        # Estado articular y de torque (para logging/control)
        self.q = np.zeros(3)
        self.tau = np.zeros(3)

    # ------------------------------------------------------------------
    #  Cinematica directa e inversa
    # ------------------------------------------------------------------
    def forward_kinematics(self, q=None):
        """FK analitica: q=(q1,q2,q3) -> p=(x,y,z) del efector final.

        Si ``q`` no es None, ademas actualiza ``self.q``.
        """
        if q is not None:
            self.q = np.asarray(q, dtype=float).copy()
        q1, q2, q3 = self.q
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)

        r = self.l2 * c2 + self.l3 * c23           # reach horizontal
        x = r * c1
        y = r * s1
        z = self.l1 + self.l2 * s2 + self.l3 * s23
        return np.array([x, y, z])

    def inverse_kinematics(self, p_des, elbow_up=True):
        """IK geometrica cerrada: p_des -> (q1, q2, q3).

        Parameters
        ----------
        p_des : array-like (3,)
            Posicion deseada del efector en el marco base [m].
        elbow_up : bool
            Si True, escoge la rama "codo arriba" (q3 <= 0). Es la
            configuracion natural para alcanzar cajas sobre una mesa.

        Returns
        -------
        np.ndarray shape (3,)
            Angulos articulares [q1, q2, q3] en rad.

        Raises
        ------
        ValueError
            Si la posicion solicitada cae fuera del workspace alcanzable.
        """
        x, y, z = p_des

        # --- Paso 1: q1 (rotacion base) ---
        q1 = float(np.arctan2(y, x))

        # --- Paso 2: pasar al plano del brazo (r, h) ---
        r = float(np.hypot(x, y))               # reach horizontal
        h = float(z - self.l1)                  # altura sobre el hombro
        d_sq = r * r + h * h                    # distancia hombro->wrist al cuadrado
        d = np.sqrt(d_sq)

        # Chequeo de alcanzabilidad
        d_max = self.l2 + self.l3
        d_min = abs(self.l2 - self.l3)
        if d > d_max + 1e-6 or d < d_min - 1e-6:
            raise ValueError(
                f"Posicion fuera del workspace: d={d:.4f} m, "
                f"rango alcanzable=[{d_min:.4f}, {d_max:.4f}] m")

        # --- Paso 3: q3 (codo) por ley de cosenos ---
        D = (d_sq - self.l2**2 - self.l3**2) / (2.0 * self.l2 * self.l3)
        D = float(np.clip(D, -1.0, 1.0))
        q3 = -np.arccos(D) if elbow_up else np.arccos(D)

        # --- Paso 4: q2 (hombro) ---
        alpha = np.arctan2(h, r)
        beta = np.arctan2(self.l3 * np.sin(q3),
                          self.l2 + self.l3 * np.cos(q3))
        q2 = float(alpha - beta)

        return np.array([q1, q2, q3])

    # ------------------------------------------------------------------
    #  Jacobiano y control de fuerza
    # ------------------------------------------------------------------
    def jacobian(self, q=None):
        """Jacobiano analitico 3x3: J[i,j] = d p_i / d q_j.

        Filas: (x, y, z) del efector.
        Columnas: (q1, q2, q3).
        """
        if q is None:
            q = self.q
        q1, q2, q3 = q
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)

        r = self.l2 * c2 + self.l3 * c23                    # reach horizontal
        dr_dq2 = -(self.l2 * s2 + self.l3 * s23)
        dr_dq3 = -self.l3 * s23

        J = np.zeros((3, 3))
        # Fila x = r*c1
        J[0, 0] = -r * s1
        J[0, 1] = dr_dq2 * c1
        J[0, 2] = dr_dq3 * c1
        # Fila y = r*s1
        J[1, 0] = r * c1
        J[1, 1] = dr_dq2 * s1
        J[1, 2] = dr_dq3 * s1
        # Fila z = l1 + l2*s2 + l3*s23
        J[2, 0] = 0.0
        J[2, 1] = self.l2 * c2 + self.l3 * c23
        J[2, 2] = self.l3 * c23
        return J

    def is_singular(self, q=None, tol=1e-4):
        """True si |det(J)| < tol (cerca de una singularidad).

        Las dos singularidades de este brazo son:
            - Brazo extendido (q3 = 0): l2 y l3 colineales -> rank drop.
            - Pasaje por el eje base (r = 0): cualquier q1 cumple FK.
        """
        return abs(np.linalg.det(self.jacobian(q))) < tol

    def force_to_torque(self, f_tip, q=None):
        """Mapea fuerza en el efector a torques articulares: tau = J^T * f.

        Parameters
        ----------
        f_tip : array-like (3,)
            Fuerza aplicada en el efector, en el marco base [N].

        Returns
        -------
        np.ndarray shape (3,)
            Torque articular requerido (si no hay friccion ni dinamica).
        """
        J = self.jacobian(q)
        tau = J.T @ np.asarray(f_tip, dtype=float)
        self.tau = tau
        return tau

    # ------------------------------------------------------------------
    #  Trajectory helpers
    # ------------------------------------------------------------------
    def cartesian_path(self, p_start, p_end, n_steps=20, elbow_up=True):
        """Genera (n_steps+1) puntos (x,y,z) por interpolacion lineal.

        Util para alimentar luego un IK paso a paso. No resuelve IK aqui
        porque cada llamada de IK puede fallar y conviene que el caller
        maneje los errores por punto.
        """
        p_start = np.asarray(p_start, dtype=float)
        p_end = np.asarray(p_end, dtype=float)
        ts = np.linspace(0.0, 1.0, n_steps + 1)
        return np.array([(1 - t) * p_start + t * p_end for t in ts])

    def cartesian_to_joint_path(self, p_start, p_end, n_steps=20,
                                elbow_up=True):
        """Trayectoria cartesiana -> trayectoria articular via IK punto a punto.

        Returns
        -------
        np.ndarray shape (n_steps+1, 3)
            Angulos articulares en cada punto del path.
        """
        cart = self.cartesian_path(p_start, p_end, n_steps=n_steps)
        qs = np.zeros_like(cart)
        for i, p in enumerate(cart):
            qs[i] = self.inverse_kinematics(p, elbow_up=elbow_up)
        return qs

    # ------------------------------------------------------------------
    #  Acciones de alto nivel
    # ------------------------------------------------------------------
    def grasp_box(self, box_pos, grip_force=5.0, n_steps=20):
        """Mueve el efector hasta ``box_pos`` y devuelve los datos del grasp.

        El movimiento es interpolacion lineal cartesiana desde la pose
        actual hasta ``box_pos``, resuelta con IK paso a paso. Al final,
        se calcula el torque articular necesario para aplicar una fuerza
        vertical de magnitud ``grip_force`` (hacia abajo, eje -z) usando
        tau = J^T * f.

        Parameters
        ----------
        box_pos : array-like (3,)
            Posicion del centro de la caja en el marco base del brazo [m].
        grip_force : float
            Magnitud de la fuerza vertical de contacto [N].
        n_steps : int
            Numero de subdivisiones del segmento cartesiano.

        Returns
        -------
        dict con claves:
            'q_path' : (n+1, 3) trayectoria articular
            'p_path' : (n+1, 3) trayectoria cartesiana
            'tau_grip' : torque articular para mantener grip_force
            'singular' : bool, True si la pose final esta cerca de singularidad
        """
        p_start = self.forward_kinematics()
        q_path = self.cartesian_to_joint_path(p_start, box_pos,
                                              n_steps=n_steps)

        # Aplicar la pose final y calcular torques de grip
        self.q = q_path[-1].copy()
        f_tip = np.array([0.0, 0.0, -float(grip_force)])
        tau_grip = self.force_to_torque(f_tip)

        return {
            'q_path': q_path,
            'p_path': self.cartesian_path(p_start, box_pos, n_steps=n_steps),
            'tau_grip': tau_grip,
            'singular': self.is_singular(),
        }

    # ------------------------------------------------------------------
    #  Utilidades
    # ------------------------------------------------------------------
    def reset(self, q=None):
        """Resetea el estado articular (default: home pose)."""
        if q is None:
            q = np.array([0.0, 0.5, -1.0])      # postura "lista" para alcanzar
        self.q = np.asarray(q, dtype=float).copy()
        self.tau = np.zeros(3)

    def in_joint_limits(self, q=None, tol=1e-6):
        """True si todos los q estan dentro de [q_min, q_max]."""
        if q is None:
            q = self.q
        return bool(np.all(q >= self.q_min - tol) and
                    np.all(q <= self.q_max + tol))

    def workspace_radius(self):
        """Radio maximo alcanzable desde el hombro (l2 + l3)."""
        return self.l2 + self.l3
