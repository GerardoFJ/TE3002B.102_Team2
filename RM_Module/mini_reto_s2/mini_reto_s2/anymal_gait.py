"""Generador de marcha trote para el ANYmal y navegacion punto-a-punto.

Contiene tres piezas:

    - ANYmalTrotGait : convierte (v_forward, t) en posiciones cartesianas
                       deseadas del pie por pata, resuelve IK y monitorea
                       det(J) en cada paso.

    - ANYmalNavigator : controlador proporcional sencillo que dado un
                        target (x, y) en el marco mundo produce comandos
                        (v_forward, omega_yaw) para la base del cuadrupedo.

    - simulate_anymal_to_target : ejecuta el lazo cerrado completo,
                                  integra la base flotante e instancia
                                  el log para visualizacion y reportes.

Diseno alineado con los tips del profesor:
    * Interpolacion lineal cartesiana en el plano de cada pata.
    * Workspace del ANYmal limitado para no acercarse a q3 -> 0.
    * Monitor de det(J) por pata y conteo de violaciones.

Notas sobre el modelo simplificado del ANYmal
---------------------------------------------
El modelo de :class:`ANYmalLeg` es el del codigo compartido por el
profesor:

    x = l1*sin(q2) + l2*sin(q2+q3)
    y = side * l0 * cos(q1)
    z = -l1*cos(q2) - l2*cos(q2+q3)

Tiene dos consecuencias importantes que impactan al gait:

    1. ``y`` solo depende de ``q1``, asi que el determinante del
       Jacobiano vale ``det(J) = side*l0*l1*l2*sin(q1)*sin(q3)``
       (justo lo que pide demostrar la Pregunta 5 del examen). El
       Jacobiano es singular en ``q1=0`` Y en ``q3=0``. Para alejarnos
       de la primera singularidad escogemos ``home_q`` con
       ``q1 != 0``.

    2. La IK cerrada que trae el shared code asume una cadena mas
       general y NO es exactamente inversa de esta FK simplificada
       (FK->IK->FK no cierra cuando q1 != 0). Por eso este modulo no
       usa ``ANYmalLeg.inverse_kinematics`` para el trote. En su lugar
       resolvemos un IK planar 2-link en el plano (x, z) para
       (q2, q3), dejando ``q1`` fijo en el valor nominal. Esto es
       consistente porque el trote solo modula (x, z) del pie, no su
       coordenada lateral y.
"""

import numpy as np

from .robots_base import ANYmal


# ---------------------------------------------------------------------------
#  IK planar 2-link auxiliar
# ---------------------------------------------------------------------------
def _planar_2link_ik(x, z, l1, l2, elbow_back=True):
    """IK planar para una pata (q2, q3) en el plano (x, z) del hombro.

    Resuelve el sistema:
        x = l1*sin(q2) + l2*sin(q2+q3)
        z = -l1*cos(q2) - l2*cos(q2+q3)

    Parameters
    ----------
    x, z : float
        Coordenadas deseadas del pie en el marco hombro [m].
        Convencion: ``z`` es negativa porque el pie cuelga.
    l1, l2 : float
        Longitudes del muslo y la espinilla [m].
    elbow_back : bool
        Si True, escoge la rama "rodilla atras" (q3 < 0), que es la
        configuracion natural del ANYmal.

    Returns
    -------
    (q2, q3) : tuple of float
        Angulos articulares en rad.

    Raises
    ------
    ValueError
        Si (x, z) cae fuera del workspace alcanzable.
    """
    d_sq = x * x + z * z
    d = np.sqrt(d_sq)
    d_max = l1 + l2
    d_min = abs(l1 - l2)
    if d > d_max + 1e-6 or d < d_min - 1e-6:
        raise ValueError(
            f"Pie fuera de workspace: d={d:.4f} m, "
            f"rango=[{d_min:.4f}, {d_max:.4f}] m")

    cos_q3 = (d_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_q3 = float(np.clip(cos_q3, -1.0, 1.0))
    q3 = -np.arccos(cos_q3) if elbow_back else np.arccos(cos_q3)

    # alpha = angulo del vector (x, -z) desde la vertical hacia adelante.
    # Como z es negativo, -z > 0 es la coordenada "hacia abajo".
    alpha = np.arctan2(x, -z)
    beta = np.arctan2(l2 * np.sin(q3), l1 + l2 * np.cos(q3))
    q2 = float(alpha - beta)
    return q2, float(q3)


# ---------------------------------------------------------------------------
#  Generador de marcha trote
# ---------------------------------------------------------------------------
class ANYmalTrotGait:
    """Genera trayectorias de pie para una marcha trote del ANYmal.

    En trote, las patas diagonales opuestas estan en fase:
        LF + RH  (par diagonal 1)
        RF + LH  (par diagonal 2)
    El par 2 va a media-fase del par 1.

    Cada pata alterna entre dos fases:
        Stance (apoyo)  : pie sobre el suelo, retrocede a -v_forward
        Swing  (vuelo)  : pie levantado en arco, avanza a la nueva huella

    Parameters
    ----------
    anymal : ANYmal
        Instancia del cuadrupedo cuyas patas se controlan.
    period : float
        Periodo de un ciclo de marcha [s]. Tipico: 0.5-0.8.
    step_length : float
        Distancia horizontal entre el inicio y fin del stance [m].
    step_height : float
        Altura maxima del pie durante el swing [m].
    duty_factor : float
        Fraccion del periodo en stance. 0.5 para trote puro.
    home_q : array (3,)
        Angulos articulares "nominales" usados para definir la posicion
        de reposo del pie en el marco del hombro.
    """

    PHASE_OFFSETS = {
        'LF': 0.0,
        'RH': 0.0,
        'RF': 0.5,
        'LH': 0.5,
    }

    def __init__(self, anymal,
                 period=0.6,
                 step_length=0.18,
                 step_height=0.07,
                 duty_factor=0.5,
                 home_q=(0.25, 0.7, -1.4)):
        self.anymal = anymal
        self.period = float(period)
        self.step_length = float(step_length)
        self.step_height = float(step_height)
        self.duty_factor = float(duty_factor)
        # NOTA: q1 != 0 para alejarnos de la singularidad del modelo
        # simplificado. Ver el docstring del modulo.
        self.home_q = np.asarray(home_q, dtype=float)

        # Posicion nominal del pie en el marco hombro de cada pata.
        # Usamos la FK del shared code (que si es consistente, lo
        # inconsistente es la IK).
        self.nominal_feet = {
            name: leg.forward_kinematics(self.home_q)
            for name, leg in self.anymal.legs.items()
        }

    # ------------------------------------------------------------------
    def foot_target(self, leg_name, t, v_forward):
        """Posicion deseada del pie de ``leg_name`` en su marco hombro.

        Parameters
        ----------
        leg_name : str
            'LF', 'RF', 'LH' o 'RH'.
        t : float
            Tiempo absoluto desde el inicio de la marcha [s].
        v_forward : float
            Velocidad de avance del cuerpo [m/s]. Si es 0 la marcha
            queda "marcando el paso" en el sitio.

        Returns
        -------
        np.ndarray shape (3,)
            Posicion deseada del pie (x, y, z) en marco hombro.
        """
        offset = self.PHASE_OFFSETS[leg_name]
        phi = ((t / self.period) + offset) % 1.0
        nominal = self.nominal_feet[leg_name].copy()

        # Escalar el step_length al regimen de velocidad: a v=0 no hay paso
        # (el pie solo se levanta), a v=v_nom cubre step_length completo.
        v_ref = self.step_length / self.period      # velocidad "natural"
        v_ratio = float(np.clip(v_forward / max(v_ref, 1e-6), 0.0, 1.5))
        eff_step = self.step_length * v_ratio

        if phi < self.duty_factor:
            # ---------------- Stance (apoyo) ----------------
            # x recorre +eff_step/2 -> -eff_step/2 linealmente
            s = phi / self.duty_factor                       # [0, 1]
            dx = eff_step / 2.0 - eff_step * s
            dz = 0.0
        else:
            # ---------------- Swing (vuelo) -----------------
            # x recorre -eff_step/2 -> +eff_step/2; z sube en arco senoidal
            s = (phi - self.duty_factor) / (1.0 - self.duty_factor)
            dx = -eff_step / 2.0 + eff_step * s
            dz = self.step_height * np.sin(np.pi * s)

        return np.array([nominal[0] + dx,
                         nominal[1],
                         nominal[2] + dz])

    # ------------------------------------------------------------------
    def step(self, t, v_forward):
        """Calcula angulos articulares deseados para las 4 patas en t.

        Usa el IK planar 2-link en el plano (x, z) del hombro y mantiene
        ``q1`` fijo en su valor nominal (no se modula la coordenada y
        durante el trote). Tambien evalua el Jacobiano completo y el
        determinante por pata para monitorear cercania a singularidades.

        Returns
        -------
        joint_targets : dict[str, np.ndarray]
            Mapa nombre_pata -> q (3,) con los angulos articulares.
        det_jacs : dict[str, float]
            Mapa nombre_pata -> det(J) evaluado en la nueva q.
        foot_targets : dict[str, np.ndarray]
            Mapa nombre_pata -> p (3,) con la posicion deseada del pie.
        """
        joint_targets = {}
        det_jacs = {}
        foot_targets = {}
        q1_nominal = float(self.home_q[0])

        for name, leg in self.anymal.legs.items():
            p_des = self.foot_target(name, t, v_forward)
            try:
                q2, q3 = _planar_2link_ik(p_des[0], p_des[2],
                                          leg.l1, leg.l2,
                                          elbow_back=True)
                q = np.array([q1_nominal, q2, q3])
            except ValueError:
                # Si el target cae fuera del workspace, conserva la pose previa.
                q = leg.q.copy()
            det_J = float(np.linalg.det(leg.jacobian(q)))
            joint_targets[name] = q
            det_jacs[name] = det_J
            foot_targets[name] = p_des

        return joint_targets, det_jacs, foot_targets

    # ------------------------------------------------------------------
    def is_phase_swing(self, leg_name, t):
        """True si la pata esta en la fase de vuelo en el tiempo t."""
        offset = self.PHASE_OFFSETS[leg_name]
        phi = ((t / self.period) + offset) % 1.0
        return phi >= self.duty_factor


# ---------------------------------------------------------------------------
#  Navegador punto-a-punto
# ---------------------------------------------------------------------------
class ANYmalNavigator:
    """Controlador proporcional simple cuerpo-a-target.

    Convierte la diferencia (target - base_pos) en un comando de cuerpo
    (v_forward, omega_yaw) que se le pasa al gait y al integrador de la
    base flotante.

    El controlador frena automaticamente cerca del target y reduce la
    velocidad de avance cuando el robot esta mal orientado (asi gira
    primero y luego camina, en lugar de espiralear).
    """

    def __init__(self, k_v=0.7, k_yaw=2.0, v_max=0.45, omega_max=0.8,
                 stop_radius=0.10):
        self.k_v = float(k_v)
        self.k_yaw = float(k_yaw)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.stop_radius = float(stop_radius)

    def compute_command(self, base_xy, base_yaw, target_xy):
        """Retorna (v_forward, omega_yaw, dist_to_target)."""
        dx = float(target_xy[0] - base_xy[0])
        dy = float(target_xy[1] - base_xy[1])
        dist = float(np.hypot(dx, dy))

        if dist < self.stop_radius:
            return 0.0, 0.0, dist

        target_yaw = np.arctan2(dy, dx)
        yaw_err = float(np.arctan2(np.sin(target_yaw - base_yaw),
                                   np.cos(target_yaw - base_yaw)))

        # Avanzar solo si miramos hacia el destino. cos(yaw_err) >= 0
        # asegura que no se camine hacia atras durante el giro.
        v_forward = float(np.clip(self.k_v * dist, 0.0, self.v_max))
        v_forward *= max(0.0, np.cos(yaw_err))

        omega_yaw = float(np.clip(self.k_yaw * yaw_err,
                                  -self.omega_max, self.omega_max))
        return v_forward, omega_yaw, dist


# ---------------------------------------------------------------------------
#  Simulacion completa: gait + navegador + integracion de base
# ---------------------------------------------------------------------------
def simulate_anymal_to_target(anymal, target_xy,
                              gait=None, navigator=None,
                              T_max=40.0, dt=0.005,
                              start_xy=(0.0, 0.0), start_yaw=0.0,
                              singular_tol=1e-3):
    """Lazo cerrado: el ANYmal trota hasta ``target_xy``.

    Integra la base flotante con Euler explicito a partir del comando
    (v_forward, omega_yaw) producido por ``navigator`` y empuja las
    referencias articulares calculadas por ``gait`` a las 4 patas.

    Parameters
    ----------
    anymal : ANYmal
        Robot a simular. Su pose se modifica in-place.
    target_xy : array-like (2,)
        Destino en el plano del piso [m].
    gait : ANYmalTrotGait, opcional
        Si None se crea con parametros por defecto.
    navigator : ANYmalNavigator, opcional
        Si None se crea con parametros por defecto.
    T_max : float
        Tiempo maximo de simulacion [s] (por seguridad).
    dt : float
        Paso de integracion [s] (~200 Hz por defecto).
    start_xy, start_yaw : pose inicial de la base.
    singular_tol : float
        Si |det(J)| < singular_tol en cualquier pata, cuenta como violacion.

    Returns
    -------
    log : dict
        Trayectorias de tiempo, base, comandos, q por pata, det(J) por pata,
        posiciones de pie, conteo de violaciones de singularidad y flag de
        exito (True si la base llego dentro de stop_radius).
    """
    if gait is None:
        gait = ANYmalTrotGait(anymal)
    if navigator is None:
        navigator = ANYmalNavigator()

    # Reset de la base
    anymal.reset_base(start_xy[0], start_xy[1], start_yaw)
    anymal.set_all_joint_angles(np.tile(gait.home_q, 4))

    target_xy = np.asarray(target_xy, dtype=float)
    n_max = int(T_max / dt)

    log = {
        't':         np.zeros(n_max),
        'base_x':    np.zeros(n_max),
        'base_y':    np.zeros(n_max),
        'base_yaw':  np.zeros(n_max),
        'v_cmd':     np.zeros(n_max),
        'omega_cmd': np.zeros(n_max),
        'dist':      np.zeros(n_max),
        'q':         np.zeros((n_max, 12)),
        'det_J':     {name: np.zeros(n_max) for name in ANYmal.LEG_NAMES},
        'foot':      {name: np.zeros((n_max, 3)) for name in ANYmal.LEG_NAMES},
    }
    violations = {name: 0 for name in ANYmal.LEG_NAMES}
    success = False
    n_used = n_max

    for i in range(n_max):
        t = i * dt
        base_xy = anymal.base_pos[:2]
        v_cmd, omega_cmd, dist = navigator.compute_command(
            base_xy, anymal.base_yaw, target_xy)

        # Generar referencias del gait y aplicarlas a las patas
        q_targets, det_jacs, foot_targets = gait.step(t, v_cmd)
        for name, q in q_targets.items():
            anymal.legs[name].q = q
            if abs(det_jacs[name]) < singular_tol:
                violations[name] += 1

        # Integrar la base flotante
        anymal.base_pos[0] += v_cmd * np.cos(anymal.base_yaw) * dt
        anymal.base_pos[1] += v_cmd * np.sin(anymal.base_yaw) * dt
        anymal.base_yaw = float(np.arctan2(
            np.sin(anymal.base_yaw + omega_cmd * dt),
            np.cos(anymal.base_yaw + omega_cmd * dt)))

        # Logging
        log['t'][i] = t
        log['base_x'][i] = anymal.base_pos[0]
        log['base_y'][i] = anymal.base_pos[1]
        log['base_yaw'][i] = anymal.base_yaw
        log['v_cmd'][i] = v_cmd
        log['omega_cmd'][i] = omega_cmd
        log['dist'][i] = dist
        log['q'][i, :] = anymal.get_all_joint_angles()
        for name in ANYmal.LEG_NAMES:
            log['det_J'][name][i] = det_jacs[name]
            log['foot'][name][i, :] = foot_targets[name]

        if dist < navigator.stop_radius:
            success = True
            n_used = i + 1
            break

    # Trim al numero real de pasos
    for k in ('t', 'base_x', 'base_y', 'base_yaw',
              'v_cmd', 'omega_cmd', 'dist'):
        log[k] = log[k][:n_used]
    log['q'] = log['q'][:n_used]
    for name in ANYmal.LEG_NAMES:
        log['det_J'][name] = log['det_J'][name][:n_used]
        log['foot'][name] = log['foot'][name][:n_used]

    log['success'] = success
    log['violations'] = violations
    log['final_pose'] = (anymal.base_pos[0], anymal.base_pos[1], anymal.base_yaw)
    log['final_error'] = float(np.hypot(anymal.base_pos[0] - target_xy[0],
                                        anymal.base_pos[1] - target_xy[1]))
    return log


# ---------------------------------------------------------------------------
#  Demo rapida (no requiere matplotlib)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Demo rapida: ANYmal trota hacia (11.0, 3.6)")
    bot = ANYmal()
    log = simulate_anymal_to_target(bot, target_xy=(11.0, 3.6),
                                    T_max=60.0, dt=0.01)
    print(f"  exito           : {log['success']}")
    print(f"  pose final      : x={log['final_pose'][0]:.3f} m, "
          f"y={log['final_pose'][1]:.3f} m, "
          f"yaw={np.degrees(log['final_pose'][2]):.1f} deg")
    print(f"  error final     : {log['final_error']:.4f} m")
    print(f"  pasos integrados: {len(log['t'])}")
    print(f"  violaciones det(J) por pata: {log['violations']}")
    min_dets = {n: float(np.min(np.abs(log['det_J'][n])))
                for n in ANYmal.LEG_NAMES}
    print(f"  min |det(J)| por pata: {min_dets}")
