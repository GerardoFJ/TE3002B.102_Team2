"""Clases base de los tres robots del curso TE3002B.

Este archivo recoge, sin matplotlib ni demos, las clases del codigo compartido
por el profesor para que puedan ser reutilizadas por los modulos del mini reto:

    - PuzzleBot   : robot diferencial de 2 ruedas
    - HuskyA200   : skid-steer de 4 ruedas con factor de deslizamiento
    - ANYmalLeg   : una pata de 3 DoF (HAA, HFE, KFE)
    - ANYmal      : cuadrupedo completo con 4 patas

Cada clase implementa cinematica directa (FK), inversa (IK) e integracion de
pose o estado articular. La logica de planeacion, control y visualizacion vive
en otros modulos del paquete.
"""

import numpy as np


# ---------------------------------------------------------------------------
#  PuzzleBot - robot diferencial de 2 ruedas
# ---------------------------------------------------------------------------
class PuzzleBot:
    """Robot diferencial de 2 ruedas (PuzzleBot Manchester Robotics).

    Modelo cinematico:
        v     = r/2 * (wR + wL)
        omega = r/L * (wR - wL)
    """

    def __init__(self, r=0.05, L=0.19):
        # Parametros fisicos calibrados con el robot real
        self.r = r              # Radio de la rueda [m]
        self.L = L              # Distancia entre ruedas [m]

        # Estado interno: pose en marco mundo
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0        # [-pi, pi]

        # Limites fisicos (proteccion)
        self.v_max = 0.8        # [m/s]
        self.omega_max = 3.0    # [rad/s]

    def forward_kinematics(self, wR, wL):
        """Cinematica directa: (wR, wL) [rad/s] -> (v, omega) del cuerpo."""
        v = self.r / 2.0 * (wR + wL)
        omega = self.r / self.L * (wR - wL)
        return v, omega

    def inverse_kinematics(self, v, omega):
        """Cinematica inversa: (v, omega) -> (wR, wL).

        Satura los comandos a los limites fisicos antes de resolver.
        """
        v = float(np.clip(v, -self.v_max, self.v_max))
        omega = float(np.clip(omega, -self.omega_max, self.omega_max))
        wR = (2.0 * v + omega * self.L) / (2.0 * self.r)
        wL = (2.0 * v - omega * self.L) / (2.0 * self.r)
        return wR, wL

    def update_pose(self, v, omega, dt):
        """Integra la pose por Euler explicito."""
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        # Normalizar a [-pi, pi]
        self.theta = float(np.arctan2(np.sin(self.theta), np.cos(self.theta)))

    def get_pose(self):
        """Retorna la pose actual como tupla (x, y, theta)."""
        return (self.x, self.y, self.theta)

    def reset(self, x=0.0, y=0.0, theta=0.0):
        """Resetea la pose del robot a un valor dado."""
        self.x, self.y, self.theta = float(x), float(y), float(theta)


# ---------------------------------------------------------------------------
#  Husky A200 - skid-steer de 4 ruedas
# ---------------------------------------------------------------------------
class HuskyA200:
    """Husky A200 de Clearpath Robotics - skid-steer de 4 ruedas.

    El skid-steer NO tiene mecanismo de direccion: el giro se logra
    variando la velocidad entre los dos lados (como un tanque). Esto
    implica deslizamiento lateral durante el giro, modelado aqui con un
    factor de slip dependiente del terreno.
    """

    # Factores de deslizamiento por terreno (afectan v lineal del cuerpo)
    SLIP_FACTORS = {
        "asphalt": 1.00,
        "grass":   0.85,
        "gravel":  0.78,
        "sand":    0.65,
        "mud":     0.50,
    }

    def __init__(self, r=0.1651, B=0.555, mass=50.0):
        # Parametros geometricos
        self.r = r              # Radio de rueda [m]
        self.B = B              # Distancia efectiva entre lados [m] (calibrada)

        # Parametros inerciales
        self.mass = mass        # [kg]
        self.payload_max = 75.0 # [kg]

        # Estado
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.v, self.omega = 0.0, 0.0

        # Terreno actual
        self.terrain = "asphalt"

    def forward_kinematics(self, wR1, wR2, wL1, wL2):
        """Cinematica directa skid-steer 4W -> (v, omega) del cuerpo.

        Promediamos las dos ruedas de cada lado y aplicamos el factor de
        slip del terreno a la componente lineal. La velocidad angular NO
        depende del slip en este modelo simplificado.
        """
        avg_R = (wR1 + wR2) / 2.0
        avg_L = (wL1 + wL2) / 2.0
        slip = self.SLIP_FACTORS.get(self.terrain, 0.8)
        v = self.r / 2.0 * (avg_R + avg_L) * slip
        omega = self.r / self.B * (avg_R - avg_L)
        return v, omega

    def inverse_kinematics(self, v, omega, compensate_slip=False):
        """Cinematica inversa: (v, omega) -> (wR1, wR2, wL1, wL2).

        Asume que las ruedas de cada lado giran a la misma velocidad. Si
        ``compensate_slip`` es True, divide v entre el factor de slip del
        terreno actual para que la velocidad real coincida con v.
        """
        if compensate_slip:
            slip = self.SLIP_FACTORS.get(self.terrain, 0.8)
            v = v / max(slip, 1e-3)
        wR = (2.0 * v + omega * self.B) / (2.0 * self.r)
        wL = (2.0 * v - omega * self.B) / (2.0 * self.r)
        return wR, wR, wL, wL

    def update_pose(self, v, omega, dt):
        """Integra la pose con el metodo del punto medio (mas preciso)."""
        theta_mid = self.theta + omega * dt / 2.0
        self.x += v * np.cos(theta_mid) * dt
        self.y += v * np.sin(theta_mid) * dt
        self.theta += omega * dt
        self.theta = float(np.arctan2(np.sin(self.theta), np.cos(self.theta)))
        self.v, self.omega = v, omega

    def set_terrain(self, terrain_name):
        """Cambia el terreno actual (afecta el slip)."""
        self.terrain = terrain_name

    def get_pose(self):
        """Retorna la pose actual como tupla (x, y, theta)."""
        return (self.x, self.y, self.theta)

    def reset(self, x=0.0, y=0.0, theta=0.0):
        """Resetea pose y velocidades."""
        self.x, self.y, self.theta = float(x), float(y), float(theta)
        self.v, self.omega = 0.0, 0.0


# ---------------------------------------------------------------------------
#  ANYmal - cuadrupedo de 12 DoF
# ---------------------------------------------------------------------------
class ANYmalLeg:
    """Una pata del ANYmal con 3 articulaciones (HAA, HFE, KFE).

    Convenciones del marco de referencia (origen en el hombro):
        eje X : hacia adelante
        eje Y : lateral (positivo hacia afuera del cuerpo)
        eje Z : hacia arriba
        side  : +1 para patas izquierdas (LF, LH), -1 para derechas (RF, RH)
    """

    def __init__(self, name, l0=0.0585, l1=0.35, l2=0.33, side=1):
        self.name = name
        # Longitudes de eslabones (valores tipicos ANYmal)
        self.l0 = l0            # Brazo HAA [m]
        self.l1 = l1            # Muslo (thigh) [m]
        self.l2 = l2            # Espinilla (shank) [m]
        self.side = side
        # Estado articular [HAA, HFE, KFE]
        self.q = np.zeros(3)
        # Limites articulares
        self.q_min = np.array([-0.72, -9.42, -2.69])
        self.q_max = np.array([ 0.49,  9.42, -0.03])

    def forward_kinematics(self, q=None):
        """FK analitica: q=(q1,q2,q3) [rad] -> p=(x,y,z) del pie [m]."""
        if q is not None:
            self.q = np.asarray(q, dtype=float)
        q1, q2, q3 = self.q
        x = self.l1 * np.sin(q2) + self.l2 * np.sin(q2 + q3)
        y = self.side * self.l0 * np.cos(q1)
        z = -self.l1 * np.cos(q2) - self.l2 * np.cos(q2 + q3)
        return np.array([x, y, z])

    def inverse_kinematics(self, p_des):
        """IK geometrica cerrada (configuracion 'rodilla atras', q3 < 0)."""
        x, y, z = p_des
        # Paso 1: q1 (HAA) - proyeccion en el plano YZ
        r_yz_sq = y**2 + z**2 - self.l0**2
        r_yz = np.sqrt(max(r_yz_sq, 1e-9))
        q1 = np.arctan2(y, -z) - np.arctan2(self.side * self.l0, r_yz)
        # Paso 2: q3 (KFE) - ley de cosenos
        r_sq = x**2 + z**2
        D = (r_sq - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2)
        D = float(np.clip(D, -1.0, 1.0))
        q3 = -np.arccos(D)
        # Paso 3: q2 (HFE)
        alpha = np.arctan2(x, -z)
        beta = np.arctan2(self.l2 * np.sin(-q3),
                          self.l1 + self.l2 * np.cos(q3))
        q2 = alpha - beta
        return np.array([q1, q2, q3])

    def jacobian(self, q=None):
        """Jacobiano analitico 3x3: dp_pie / dq."""
        if q is None:
            q = self.q
        q1, q2, q3 = q
        J = np.zeros((3, 3))
        J[0, 1] = self.l1 * np.cos(q2) + self.l2 * np.cos(q2 + q3)
        J[0, 2] = self.l2 * np.cos(q2 + q3)
        J[1, 0] = -self.side * self.l0 * np.sin(q1)
        J[2, 1] = self.l1 * np.sin(q2) + self.l2 * np.sin(q2 + q3)
        J[2, 2] = self.l2 * np.sin(q2 + q3)
        return J

    def is_singular(self, q=None, tol=1e-3):
        """True si |det(J)| < tol (cerca de singularidad)."""
        return abs(np.linalg.det(self.jacobian(q))) < tol


class ANYmal:
    """Cuadrupedo ANYmal completo: 4 patas, 12 DoF + base flotante."""

    LEG_NAMES = ['LF', 'RF', 'LH', 'RH']

    def __init__(self):
        self.legs = {
            'LF': ANYmalLeg('LF', side=+1),
            'RF': ANYmalLeg('RF', side=-1),
            'LH': ANYmalLeg('LH', side=+1),
            'RH': ANYmalLeg('RH', side=-1),
        }
        # Estado de la base
        self.base_pos = np.array([0.0, 0.0, 0.45])
        self.base_yaw = 0.0
        # Parametros dinamicos
        self.mass = 30.0
        self.payload = 0.0
        self.n_legs = 4
        self.n_dof_legs = 12

    def get_all_joint_angles(self):
        """Retorna un vector de 12 elementos con todos los q de las 4 patas."""
        return np.concatenate([self.legs[name].q for name in self.LEG_NAMES])

    def set_all_joint_angles(self, q12):
        """Asigna los 12 angulos articulares a las 4 patas."""
        q12 = np.asarray(q12)
        assert q12.shape == (12,), f"Se esperan 12 angulos, llegaron {q12.shape}"
        for i, name in enumerate(self.LEG_NAMES):
            self.legs[name].q = q12[3*i: 3*(i+1)].copy()

    def get_all_foot_positions(self):
        """Dict name -> posicion 3D del pie en su marco hombro."""
        return {name: self.legs[name].forward_kinematics()
                for name in self.LEG_NAMES}

    def reset_base(self, x=0.0, y=0.0, yaw=0.0, z=0.45):
        """Resetea la pose de la base flotante."""
        self.base_pos = np.array([float(x), float(y), float(z)])
        self.base_yaw = float(yaw)
