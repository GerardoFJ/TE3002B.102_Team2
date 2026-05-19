"""Seguimiento de lineas con camara sintetica a bordo del robot.

Adapta el pipeline de vision computacional de BOOST/BOOST_corregido.ipynb
para la cuadricula de navegacion del mini reto.

Flujo (por paso de control):
  1. ``synthesize_view`` — genera una imagen RGB sintetica de lo que
     el robot ve mirando al suelo adelante: la linea coloreada del path.
  2. ``detect_center_error`` — umbralado de color → mascara → Hough
     probabilistico (cv2.HoughLinesP) → posicion del centro de la linea.
  3. ``line_follow_step`` — controlador geometrico basado en BOOST:
        omega = -k_center * center_error - k_heading * heading_error
     (el mismo ``geometric_baseline`` del BOOST, con heading adicional).

Intersecciones:
  El robot llega al centro de una celda (waypoint). El coordinador ya
  pre-computo la trayectoria A* y provee el siguiente waypoint. El robot
  gira hacia el y continua. El cambio de segmento (wp_from → wp_to) se
  refleja instantaneamente en la imagen sintetica: la linea "dobla" en
  la interseccion.

Dependencia de OpenCV:
  Si cv2 no esta instalado se cae al centroide por color (equivalente
  al geometric_baseline de BOOST sin la etapa Hough).
"""

import math

import numpy as np

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ---------------------------------------------------------------------------
# Colores RGB de las lineas en el suelo (deben coincidir con navigation_grid)
# ---------------------------------------------------------------------------
LINE_RGB = {
    'husky':  (255, 255,   0),   # amarillo
    'anymal': (  0, 255, 255),   # cian
    'xarm':   (255,   0, 255),   # magenta
}

# Tamaño de la imagen de camara sintetica
CAM_W, CAM_H = 80, 40

# Campo visual de la camara
HALF_WIDTH_M = 1.5     # semiancho del campo de vision [m]
CAM_RANGE_M  = 2.5     # alcance frontal del campo de vision [m]


# ===========================================================================
#  Camara sintetica
# ===========================================================================
def synthesize_view(robot_x, robot_y, robot_theta,
                    wp_from, wp_to, line_rgb,
                    cam_w=CAM_W, cam_h=CAM_H,
                    half_width=HALF_WIDTH_M):
    """Genera la imagen sintetica de la camara de abordo.

    Simula una camara montada en el robot que mira hacia el suelo adelante.
    El path se pinta como una banda coloreada sobre el piso gris.

    La posicion horizontal de la banda refleja el error lateral del robot
    respecto a la linea del path; la inclinacion de la banda refleja el
    error de encabezado (heading error).

    Parameters
    ----------
    robot_x, robot_y, robot_theta : pose del robot [m, m, rad]
    wp_from, wp_to : (x, y) extremos del segmento de linea actual [m]
    line_rgb : (R, G, B) color de la linea

    Returns
    -------
    img : ndarray uint8 (cam_h, cam_w, 3)  imagen RGB
    """
    img = np.full((cam_h, cam_w, 3), 55, dtype=np.uint8)   # piso gris oscuro

    # Marcas horizontales del piso (simulan juntas de baldosas)
    step = max(1, cam_h // 5)
    for row in range(0, cam_h, step):
        img[row] = np.clip(img[row].astype(np.int32) + 18, 0, 255)

    # Geometria del segmento
    seg = np.array(wp_to, dtype=float) - np.array(wp_from, dtype=float)
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-6:
        return img
    seg_dir = seg / seg_len

    # Marco del robot
    fwd   = np.array([math.cos(robot_theta), math.sin(robot_theta)])
    robot_pos = np.array([robot_x, robot_y])
    to_wp_from = np.array(wp_from, dtype=float) - robot_pos

    # Error lateral: + = robot a la izquierda de la linea
    #   (se calcula como proyeccion transversal del vector robot→linea)
    fwd_proj = float(np.dot(to_wp_from, seg_dir))
    foot_on_line = np.array(wp_from) + fwd_proj * seg_dir
    lateral_vec = foot_on_line - robot_pos
    # Signed: positive when line is to the robot's left (line appears right in img)
    right = np.array([math.sin(robot_theta), -math.cos(robot_theta)])
    lateral_err = float(np.dot(lateral_vec, right))

    # Error de encabezado: angulo entre heading del robot y direccion de la linea
    cos_h = float(np.dot(fwd, seg_dir))
    sin_h = float(np.cross(fwd, seg_dir))
    heading_err = math.atan2(sin_h, cos_h)   # [-pi, pi]

    # Centro de la banda en imagen (pixeles)
    center_px = cam_w / 2 + (lateral_err / half_width) * (cam_w / 2)
    center_px = float(np.clip(center_px, 0, cam_w))

    line_w_px = max(6, cam_w // 7)   # ancho de la banda
    max_slant = int(heading_err * cam_w * 0.35)   # desplazamiento tope por inclinacion

    # Dibujar la banda fila a fila (perspectiva simplificada)
    for row in range(cam_h):
        # row=0 (lejos) tiene el maximo slant; row=cam_h-1 (cerca) = sin slant
        t = row / max(cam_h - 1, 1)
        slant = int(max_slant * (1.0 - t))
        lo = max(0, int(center_px - line_w_px // 2) + slant)
        hi = min(cam_w, int(center_px + line_w_px // 2) + slant)
        if lo < hi:
            img[row, lo:hi] = line_rgb

    return img


# ===========================================================================
#  Deteccion de linea (pipeline BOOST adaptado)
# ===========================================================================
def _centroid_detection(img, line_rgb, threshold=70):
    """Centroide de color — fallback sin cv2."""
    r, g, b = line_rgb
    dist_sq = (
        (img[:, :, 0].astype(np.int32) - r) ** 2 +
        (img[:, :, 1].astype(np.int32) - g) ** 2 +
        (img[:, :, 2].astype(np.int32) - b) ** 2
    )
    mask = dist_sq < threshold ** 2
    if not mask.any():
        return float(img.shape[1]) / 2, 0.0
    xs = np.where(mask)[1]
    return float(xs.mean()), float(mask.sum()) / mask.size


def detect_center_error(img, line_rgb, use_hough=True, threshold=70):
    """Detecta la linea coloreada en la imagen sintetica.

    Pipeline (identico al BOOST):
      1. Mascara de color (umbralado RGB).
      2. Canny sobre la mascara → bordes de la banda.
      3. HoughLinesP → segmentos de linea en los bordes.
      4. Media de posiciones x de los segmentos a media altura → center_x.
      5. center_error = (center_x - W/2) / (W/2)   en [-1, 1].

    Fallback a centroide si Hough no encuentra lineas o cv2 no esta.

    Returns
    -------
    center_error : float [-1, 1]   (0 = centrado, + = linea a la derecha)
    confidence   : float [0, 1]
    center_x_px  : float           (x detectada en pixeles)
    """
    cam_w = img.shape[1]
    cam_h = img.shape[0]

    # --- Mascara de color ---
    r, g, b = line_rgb
    dist_sq = (
        (img[:, :, 0].astype(np.int32) - r) ** 2 +
        (img[:, :, 1].astype(np.int32) - g) ** 2 +
        (img[:, :, 2].astype(np.int32) - b) ** 2
    )
    mask = dist_sq < threshold ** 2

    if not mask.any():
        return 0.0, 0.0, float(cam_w) / 2

    # Centroide de respaldo
    cx_centroid, conf_centroid = _centroid_detection(img, line_rgb, threshold)

    if not (use_hough and _HAS_CV2):
        err = (cx_centroid - cam_w / 2) / (cam_w / 2)
        return float(np.clip(err, -1, 1)), conf_centroid, cx_centroid

    # --- Hough probabilistico sobre los bordes de la mascara (BOOST) ---
    bw = mask.astype(np.uint8) * 255
    edges = _cv2.Canny(bw, 50, 150)
    lines = _cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=5,
        minLineLength=cam_h // 4, maxLineGap=cam_h // 3)

    if lines is None:
        err = (cx_centroid - cam_w / 2) / (cam_w / 2)
        return float(np.clip(err, -1, 1)), conf_centroid, cx_centroid

    # x de cada segmento a media altura
    mid_y = cam_h / 2
    xs_h = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if abs(y2 - y1) < 1e-6:
            continue
        t = (mid_y - y1) / (y2 - y1)
        xm = x1 + t * (x2 - x1)
        if 0 <= xm <= cam_w:
            xs_h.append(xm)

    if not xs_h:
        err = (cx_centroid - cam_w / 2) / (cam_w / 2)
        return float(np.clip(err, -1, 1)), conf_centroid, cx_centroid

    center_x = float(np.mean(xs_h))
    confidence = min(1.0, len(xs_h) / 4.0)
    err = (center_x - cam_w / 2) / (cam_w / 2)
    return float(np.clip(err, -1, 1)), confidence, center_x


# ===========================================================================
#  Controlador de seguimiento
# ===========================================================================
def line_follow_step(robot, wp_from, wp_to, line_rgb, dt,
                     k_center=2.0, k_heading=1.5,
                     v_cmd=0.45, omega_max=1.5,
                     use_hough=True):
    """Un paso de control de seguimiento de linea con camara sintetica.

    Geometric baseline de BOOST:
        omega = -k_center * center_error - k_heading * heading_error

    Parameters
    ----------
    robot    : robot con metodos get_pose(), inverse_kinematics(), update_pose()
    wp_from  : (x, y) extremo anterior del segmento de path
    wp_to    : (x, y) proximo waypoint
    line_rgb : (R, G, B) color de la linea
    dt       : paso de integracion [s]

    Returns
    -------
    dict: v_cmd, omega_cmd, v_real, omega_real, center_error, confidence, cam_img
    """
    x, y, theta = robot.get_pose()

    # Imagen sintetica de la camara
    cam_img = synthesize_view(x, y, theta, wp_from, wp_to, line_rgb)

    # Deteccion de linea (BOOST pipeline)
    center_err, conf, _ = detect_center_error(cam_img, line_rgb, use_hough)

    # Error de encabezado (geometrico, mas robusto que vision para angulos grandes)
    dx, dy = wp_to[0] - x, wp_to[1] - y
    target_yaw = math.atan2(dy, dx)
    heading_err = math.atan2(math.sin(target_yaw - theta),
                             math.cos(target_yaw - theta))

    # Omega: steer toward line (camera) + steer toward waypoint direction (geometric).
    # heading_err > 0 means target is to the LEFT → need positive omega (CCW).
    omega = -k_center * center_err + k_heading * heading_err
    omega = float(np.clip(omega, -omega_max, omega_max))

    # Velocidad forward: cero cuando error > 90°, maxima cuando alineado.
    cos_h = math.cos(heading_err)
    v = v_cmd * max(0.05, cos_h) if cos_h > 0 else v_cmd * 0.05

    # Actuacion segun tipo de robot
    try:
        # HuskyA200 (4-ruedas con compensacion de slip)
        wR, wR2, wL, wL2 = robot.inverse_kinematics(v, omega, compensate_slip=True)
        v_real, omega_real = robot.forward_kinematics(wR, wR2, wL, wL2)
    except (TypeError, ValueError):
        try:
            # PuzzleBot (2-ruedas)
            wR, wL = robot.inverse_kinematics(v, omega)
            v_real, omega_real = robot.forward_kinematics(wR, wL)
        except Exception:
            v_real, omega_real = v, omega

    robot.update_pose(v_real, omega_real, dt)

    return {
        'v_cmd':        v,
        'omega_cmd':    omega,
        'v_real':       v_real,
        'omega_real':   omega_real,
        'center_error': center_err,
        'confidence':   conf,
        'cam_img':      cam_img,
    }


# ===========================================================================
#  Transit completo (secuencia de waypoints)
# ===========================================================================
def transit_to_waypoints(robot, waypoints, line_rgb, dt, log,
                          world_boxes=None, eps=0.45,
                          max_steps=2000, k_center=2.0, k_heading=1.5,
                          v_cmd=0.45, use_hough=True):
    """Navega el robot a traves de una lista de waypoints usando la camara.

    En cada waypoint (interseccion de cuadricula), el robot recibe
    automaticamente el siguiente (la decision del coordinador ya esta
    codificada en la lista de waypoints A*).

    Parameters
    ----------
    robot      : HuskyA200 u otro robot con get_pose() etc.
    waypoints  : list of (x, y) — centros de celdas del path A*
    line_rgb   : (R, G, B) color de la linea a seguir
    dt         : paso [s]
    log        : dict de log con claves t, x, y, theta, v_cmd, omega_cmd,
                 v_real, omega_real, boxes, phase
    world_boxes: list de Box (para mantener log['boxes'] consistente)
    eps        : radio de llegada a cada waypoint [m]
    """
    if len(waypoints) < 2:
        return

    t_cur = log['t'][-1] if log['t'] else 0.0
    box_names = list(log.get('boxes', {}).keys())
    prev_wp = waypoints[0]

    for next_wp in waypoints[1:]:
        for _ in range(max_steps):
            x, y, _ = robot.get_pose()
            if math.hypot(next_wp[0] - x, next_wp[1] - y) < eps:
                break

            step = line_follow_step(
                robot, prev_wp, next_wp, line_rgb, dt,
                k_center=k_center, k_heading=k_heading,
                v_cmd=v_cmd, use_hough=use_hough)

            t_cur += dt
            log['t'].append(t_cur)
            log['x'].append(robot.x)
            log['y'].append(robot.y)
            log['theta'].append(robot.theta)
            log['v_cmd'].append(step['v_cmd'])
            log['omega_cmd'].append(step['omega_cmd'])
            log['v_real'].append(step['v_real'])
            log['omega_real'].append(step['omega_real'])

            # Cajas no se mueven durante el transit
            if world_boxes is not None:
                for bname in box_names:
                    b = next((b for b in world_boxes if b.name == bname), None)
                    if b:
                        log['boxes'][bname].append((b.x, b.y))
            else:
                for bname in box_names:
                    prev_xy = log['boxes'][bname][-1] if log['boxes'][bname] else (0.0, 0.0)
                    log['boxes'][bname].append(prev_xy)

            if 'phase' in log:
                log['phase'].append('transit')

        prev_wp = next_wp
