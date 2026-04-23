"""Simulador 2D animado del mini reto (matplotlib).

Reproduce visualmente las 3 fases del MissionCoordinator en una sola
ventana matplotlib animada con `FuncAnimation`. No reimplementa fisica:
solo dibuja los logs ya generados por el coordinador.

Uso minimo:

    from mini_reto_s2.sim import MissionVisualizer
    vis = MissionVisualizer()
    vis.run_and_show()                 # corre el coord y abre la ventana
    # o, si solo quieres guardar a mp4/gif:
    vis.run_and_save('mission.mp4', fps=30)

Tambien se puede ejecutar como script:

    python3 -m mini_reto_s2.sim
    python3 -m mini_reto_s2.sim --save mission.mp4
    python3 -m mini_reto_s2.sim --no-show

Disenado para ser **dumb**: la logica del reto vive en los modulos
especializados (husky_pusher, anymal_gait, puzzlebot_arm, coordinator).
Aqui solo hay matplotlib + interpolacion sobre los logs.
"""

import argparse
import math

import numpy as np

from .coordinator import MissionCoordinator


def _import_matplotlib():
    """Carga matplotlib bajo demanda.

    El frame-building no necesita matplotlib (es solo numpy + python),
    asi que diferimos el import para que el modulo y los tests del
    builder funcionen incluso en entornos donde matplotlib no esta
    instalado / esta roto (ej: numpy 2.x con mpl compilado para 1.x).
    """
    import matplotlib                          # noqa: F401
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation
    return matplotlib, plt, mpatches, FuncAnimation


# ===========================================================================
#  Helpers de dibujo
# ===========================================================================
def _draw_husky(ax, mpatches, x, y, theta,
                length=0.99, width=0.67, color='#1f77b4'):
    """Dibuja un Husky A200 como rectangulo orientado con flecha de yaw."""
    cs, sn = math.cos(theta), math.sin(theta)
    hl, hw = length / 2.0, width / 2.0
    corners_local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    R = np.array([[cs, -sn], [sn, cs]])
    corners = (R @ corners_local.T).T + np.array([x, y])
    poly = mpatches.Polygon(corners, closed=True, facecolor=color,
                            edgecolor='black', linewidth=1.0, alpha=0.85)
    ax.add_patch(poly)
    ax.arrow(x, y, 0.35 * cs, 0.35 * sn,
             head_width=0.10, head_length=0.10, fc='yellow', ec='black',
             length_includes_head=True, linewidth=0.6)


def _draw_anymal(ax, mpatches, x, y, yaw,
                 length=0.95, width=0.55, color='#9467bd'):
    """Dibuja el ANYmal como rectangulo orientado (vista cenital)."""
    cs, sn = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    corners_local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    R = np.array([[cs, -sn], [sn, cs]])
    corners = (R @ corners_local.T).T + np.array([x, y])
    poly = mpatches.Polygon(corners, closed=True, facecolor=color,
                            edgecolor='black', linewidth=1.0, alpha=0.85)
    ax.add_patch(poly)
    ax.plot(x + 0.45 * cs, y + 0.45 * sn, 'o',
            color='yellow', markersize=4, markeredgecolor='black')


def _draw_puzzlebot(ax, mpatches, x, y, theta, side=0.20, color='#2ca02c'):
    """Dibuja un PuzzleBot como cuadrado pequeno orientado."""
    cs, sn = math.cos(theta), math.sin(theta)
    h = side / 2.0
    corners_local = np.array([[h, h], [h, -h], [-h, -h], [-h, h]])
    R = np.array([[cs, -sn], [sn, cs]])
    corners = (R @ corners_local.T).T + np.array([x, y])
    poly = mpatches.Polygon(corners, closed=True, facecolor=color,
                            edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.add_patch(poly)
    ax.plot(x + 0.12 * cs, y + 0.12 * sn, '.',
            color='yellow', markersize=3)


def _draw_xarm(ax, mpatches, link_points_world, base_yaw,
               base_radius=0.12, base_color='#333333'):
    """Dibuja el xArm 6 en top-view: base + polilinea de eslabones.

    ``link_points_world`` es el resultado de ``XArm.link_points_world()``:
    5 puntos (base, hombro, codo, muneca, tcp) en el marco mundo.
    Para el top-view proyectamos (x, y) y pintamos grosor proporcional
    a la altura z para dar sensacion de elevacion.
    """
    pts = np.asarray(link_points_world)
    xb, yb = pts[0, 0], pts[0, 1]

    # Base (caja cuadrada fija anclada al piso)
    base_side = 0.22
    cs, sn = math.cos(base_yaw), math.sin(base_yaw)
    corners_local = np.array([
        [ base_side / 2,  base_side / 2],
        [ base_side / 2, -base_side / 2],
        [-base_side / 2, -base_side / 2],
        [-base_side / 2,  base_side / 2],
    ])
    R = np.array([[cs, -sn], [sn, cs]])
    corners = (R @ corners_local.T).T + np.array([xb, yb])
    base_poly = mpatches.Polygon(corners, closed=True, facecolor=base_color,
                                 edgecolor='black', linewidth=1.0, alpha=0.9)
    ax.add_patch(base_poly)

    # Eslabones como polilinea, proyeccion top-view (x, y)
    xs = pts[:, 0]
    ys = pts[:, 1]
    # Eslabon 1 (base->hombro) es vertical puro, casi no aporta visual;
    # lo marcamos como punto. Dibujamos brazo y antebrazo:
    ax.plot(xs[1:4], ys[1:4], '-', color='#888888',
            linewidth=4, alpha=0.9, solid_capstyle='round')
    # Efector (TCP) como circulo rojo
    ax.plot(pts[4, 0], pts[4, 1], 'o', color='#d62728',
            markersize=6, markeredgecolor='black', zorder=5)
    # Articulaciones (hombro, codo, muneca) como puntos negros
    for i in (1, 2, 3):
        ax.plot(pts[i, 0], pts[i, 1], 'o', color='black',
                markersize=3.5, zorder=5)


def _draw_box(ax, mpatches, cx, cy, side, color='#ff7f0e', label=None):
    """Dibuja una caja AABB centrada en (cx, cy) con half-side ``side``."""
    rect = mpatches.Rectangle((cx - side, cy - side), 2 * side, 2 * side,
                              facecolor=color, edgecolor='black',
                              linewidth=0.8, alpha=0.85)
    ax.add_patch(rect)
    if label:
        ax.text(cx, cy, label, ha='center', va='center', fontsize=7,
                fontweight='bold')


# ===========================================================================
#  Visualizador
# ===========================================================================
class MissionVisualizer:
    """Anima los logs de un MissionCoordinator en una sola ventana 2D.

    Parameters
    ----------
    coord : MissionCoordinator, opcional
        Si es None, se crea uno por defecto.
    stride : int
        Submuestreo del log (1 = todos los frames). Para fluidez tipica
        se usan ~10-20 (las fases generan miles de pasos).
    """

    def __init__(self, coord=None, stride=15):
        self.coord = coord or MissionCoordinator()
        self.stride = int(stride)
        self.log = None
        # Marco mundo: cubre el corredor + zona de trabajo de fase 3
        self.xlim = (-1.0, 14.0)
        self.ylim = (-1.0, 5.5)

    # ------------------------------------------------------------------
    def run(self):
        """Ejecuta el coordinador y guarda el log."""
        self.log = self.coord.run()
        return self.log

    # ------------------------------------------------------------------
    def _build_frames(self):
        """Convierte los 3 logs en una unica lista de frames homogeneos.

        Cada frame es un dict con la pose de cada robot y la posicion
        de cada caja en ese instante. Asi el animator solo tiene que
        leer y dibujar.
        """
        if self.log is None:
            self.run()

        frames = []
        # ----- Fase 1 -----
        # El ANYmal aun no se mueve y carga los 3 PuzzleBots en su dorso.
        anymal_start = (0.0, 0.0, 0.0)
        pb_on_back = self._puzzlebots_on_anymal(*anymal_start)
        p1 = self.log['phase1']
        n1 = len(p1['t'])
        for i in range(0, n1, self.stride):
            frames.append({
                'phase': 1,
                'husky': (p1['x'][i], p1['y'][i], p1['theta'][i]),
                'anymal': anymal_start,
                'puzzlebots': pb_on_back,
                'big_boxes': {name: p1['boxes'][name][i]
                              for name in p1['boxes']},
            })

        # Posicion final de fase 1 (para arrancar fase 2)
        last_husky = (p1['x'][-1], p1['y'][-1], p1['theta'][-1])
        last_big_boxes = {name: p1['boxes'][name][-1] for name in p1['boxes']}

        # ----- Fase 2 -----
        p2 = self.log['phase2']
        n2 = len(p2['t'])
        for i in range(0, n2, self.stride):
            frames.append({
                'phase': 2,
                'husky': last_husky,
                'anymal': (p2['base_x'][i], p2['base_y'][i], p2['base_yaw'][i]),
                'puzzlebots': self._puzzlebots_on_anymal(
                    p2['base_x'][i], p2['base_y'][i], p2['base_yaw'][i]),
                'big_boxes': last_big_boxes,
            })

        last_anymal = (p2['base_x'][-1], p2['base_y'][-1], p2['base_yaw'][-1])

        # ----- Fase 2.5: xArm transfiere los PBs del ANYmal a la mesa -----
        p25 = self.log['phase2_5']
        pb_world = self._puzzlebots_on_anymal(*last_anymal)
        role_to_idx = {'C': 0, 'B': 1, 'A': 2}

        from .xarm import XArm  # import local para no tocar el top-level
        xarm_vis = XArm(base_xy=p25['base_xy'], base_yaw=p25['base_yaw'])

        for unit in p25['units']:
            role = unit['role']
            qs = unit['q_path']
            idx_grab = unit['idx_grab']
            idx_release = unit['idx_release']

            for i in range(0, len(qs), max(1, self.stride // 2)):
                xarm_vis.q = qs[i].copy()
                links = xarm_vis.link_points_world()
                tcp_xy = (links[4, 0], links[4, 1])

                # El PB "activo" sigue al TCP mientras este agarrado
                if idx_grab <= i < idx_release:
                    pb_world[role_to_idx[role]] = (tcp_xy[0], tcp_xy[1], 0.0)
                elif i >= idx_release:
                    drop = self.coord.PB_TABLE_DROP[role]
                    pb_world[role_to_idx[role]] = (drop[0], drop[1], 0.0)

                frames.append({
                    'phase': 2.5,
                    'husky': last_husky,
                    'anymal': last_anymal,
                    'puzzlebots': list(pb_world),
                    'xarm_links': links.copy(),
                    'xarm_base_yaw': p25['base_yaw'],
                    'big_boxes': last_big_boxes,
                    'active': role,
                })

            # Estado final del PB depositado
            drop = self.coord.PB_TABLE_DROP[role]
            pb_world[role_to_idx[role]] = (drop[0], drop[1], 0.0)

        # Frames finales "post-misión": PBs en la mesa, xArm en home.
        home_links = self.coord.xarm.link_points_world(
            self.coord.xarm.q_home)
        for _ in range(15):
            frames.append({
                'phase': 2.5,
                'husky': last_husky,
                'anymal': last_anymal,
                'puzzlebots': list(pb_world),
                'xarm_links': home_links.copy(),
                'xarm_base_yaw': self.coord.XARM_BASE_YAW,
                'big_boxes': last_big_boxes,
                'active': None,
            })

        return frames

    # ------------------------------------------------------------------
    def _puzzlebots_on_anymal(self, ax_, ay_, ayaw):
        """Devuelve poses de los 3 PB cuando van encima del ANYmal.

        Los offsets 2D se toman del coordinator (``PB_ON_ANYMAL_OFFSETS``)
        para que coincidan con lo que espera el xArm en la fase 2.5.
        Orden de salida: [C, B, A].
        """
        cs, sn = math.cos(ayaw), math.sin(ayaw)
        out = []
        for role in ('C', 'B', 'A'):
            off = self.coord.PB_ON_ANYMAL_OFFSETS[role]
            lx, ly = off[0], off[1]
            wx = ax_ + cs * lx - sn * ly
            wy = ay_ + sn * lx + cs * ly
            out.append((wx, wy, ayaw))
        return out

    # ------------------------------------------------------------------
    def _draw_static(self, ax, mpatches):
        """Dibuja el corredor y la mesa donde descansaran los PuzzleBots."""
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        # Corredor 6x2 (fase 1)
        c = self.coord.world.corridor
        corr = mpatches.Rectangle(
            (c['xmin'], c['ymin']),
            c['xmax'] - c['xmin'], c['ymax'] - c['ymin'],
            facecolor='#dddddd', edgecolor='black',
            linewidth=1.2, hatch='//', alpha=0.4,
        )
        ax.add_patch(corr)
        ax.text((c['xmin'] + c['xmax']) / 2, c['ymax'] + 0.15,
                'Corredor 6x2', ha='center', fontsize=8)

        # Mesa: rectangulo que cubre los 3 drops del xArm
        drops = list(self.coord.PB_TABLE_DROP.values())
        xs = [d[0] for d in drops]
        ys = [d[1] for d in drops]
        pad = 0.18
        x0 = min(xs) - pad
        x1 = max(xs) + pad
        y0 = min(ys) - pad
        y1 = max(ys) + pad
        table = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor='#bcd9b3', edgecolor='black',
                                   linewidth=0.8, alpha=0.5)
        ax.add_patch(table)
        ax.text((x0 + x1) / 2, y1 + 0.08, 'mesa', ha='center', fontsize=8)

        # Marca del target del ANYmal
        ax.plot(*self.coord.ANYMAL_TARGET, 'r*', markersize=10)
        ax.text(self.coord.ANYMAL_TARGET[0] + 0.10,
                self.coord.ANYMAL_TARGET[1],
                'p_destino', fontsize=7, color='red')

        # Etiqueta de la base del xArm
        xb, yb = self.coord.XARM_BASE_XY
        ax.text(xb, yb - 0.20, 'xArm 6', ha='center', fontsize=8)

    # ------------------------------------------------------------------
    def _draw_frame(self, frame, ax, mpatches, title_text):
        """Limpia y redibuja una snapshot."""
        ax.clear()
        self._draw_static(ax, mpatches)

        # Cajas grandes del corredor
        big_side = 0.30
        for name, (bx, by) in frame['big_boxes'].items():
            _draw_box(ax, mpatches, bx, by, big_side,
                      color='#ff7f0e', label=name)

        # Husky
        hx, hy, hth = frame['husky']
        _draw_husky(ax, mpatches, hx, hy, hth)

        # ANYmal
        ax_, ay_, ath = frame['anymal']
        _draw_anymal(ax, mpatches, ax_, ay_, ath)

        # xArm 6 (siempre visible; en fase 2.5 usa links animados, en el
        # resto se dibuja en pose home usando el xarm del coordinator).
        if 'xarm_links' in frame:
            _draw_xarm(ax, mpatches, frame['xarm_links'],
                       frame.get('xarm_base_yaw', self.coord.XARM_BASE_YAW))
        else:
            home_links = self.coord.xarm.link_points_world(
                self.coord.xarm.q_home)
            _draw_xarm(ax, mpatches, home_links, self.coord.XARM_BASE_YAW)

        # PuzzleBots
        for (px, py, pth) in frame['puzzlebots']:
            _draw_puzzlebot(ax, mpatches, px, py, pth)

        ax.set_title(title_text, fontsize=10)

    # ------------------------------------------------------------------
    def animate(self, fig=None, ax=None, interval=30):
        """Devuelve (fig, FuncAnimation) listos para mostrar o guardar."""
        if self.log is None:
            self.run()
        frames = self._build_frames()
        matplotlib, plt, mpatches, FuncAnimation = _import_matplotlib()

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(11, 5.5))
        self._draw_static(ax, mpatches)

        phase_titles = {
            1:   "Fase 1: Husky despeja el corredor",
            2:   "Fase 2: ANYmal trota al destino con 3 PuzzleBots",
            2.5: "Fase 2.5: xArm 6 baja los PuzzleBots a la mesa",
        }

        def update(idx):
            frame = frames[idx]
            title = (f"Frame {idx+1}/{len(frames)} - "
                     f"{phase_titles[frame['phase']]}")
            self._draw_frame(frame, ax, mpatches, title)
            return ax.patches

        anim = FuncAnimation(fig, update, frames=len(frames),
                             interval=interval, blit=False, repeat=False)
        return fig, anim

    # ------------------------------------------------------------------
    def run_and_show(self, interval=30):
        """Atajo: corre la mision y abre la ventana matplotlib."""
        _, plt, _, _ = _import_matplotlib()
        fig, anim = self.animate(interval=interval)
        plt.tight_layout()
        plt.show()
        return anim

    def run_and_save(self, filename, fps=30, dpi=120):
        """Atajo: corre la mision y guarda el video/gif a disco.

        Soporta cualquier extension que matplotlib reconozca (.mp4
        requiere ffmpeg, .gif usa Pillow). Devuelve el FuncAnimation.
        """
        matplotlib, plt, _, _ = _import_matplotlib()
        matplotlib.use('Agg', force=True)
        fig, anim = self.animate(interval=1000 // fps)
        anim.save(filename, fps=fps, dpi=dpi)
        plt.close(fig)
        return anim


# ===========================================================================
#  CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Simulador 2D animado del mini reto Semana 2")
    parser.add_argument('--save', type=str, default=None,
                        help="Guarda la animacion a este archivo (mp4/gif)")
    parser.add_argument('--no-show', action='store_true',
                        help="No abre la ventana (util para CI/headless)")
    parser.add_argument('--stride', type=int, default=15,
                        help="Submuestreo de los logs (default: 15)")
    parser.add_argument('--fps', type=int, default=30,
                        help="FPS al guardar (default: 30)")
    args = parser.parse_args()

    vis = MissionVisualizer(stride=args.stride)
    if args.save:
        print(f"Generando animacion -> {args.save} ...")
        vis.run_and_save(args.save, fps=args.fps)
        print("Listo.")
    elif args.no_show:
        # Solo correr el coordinador y construir frames (sin mostrar nada)
        vis.run()
        frames = vis._build_frames()
        print(f"Mision ejecutada. Frames generados: {len(frames)}")
        for k, v in vis.log['success'].items():
            print(f"  [{k}] success={v}")
    else:
        vis.run_and_show()


if __name__ == "__main__":
    main()
