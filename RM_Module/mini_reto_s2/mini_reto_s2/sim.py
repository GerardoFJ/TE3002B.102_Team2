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
        # El ANYmal aun no se mueve y carga los 3 PuzzleBots en su dorso
        # (segun el reto). Posicion inicial: (0, 0, 0).
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
                'small_boxes': self._initial_small_boxes(),
                'stack_count': 0,
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
                'small_boxes': self._initial_small_boxes(),
                'stack_count': 0,
            })

        last_anymal = (p2['base_x'][-1], p2['base_y'][-1], p2['base_yaw'][-1])

        # ----- Fase 3 -----
        # Cada unidad tiene drive_pick + drive_place. Estado: 3 pb iniciales
        # en (11.20, 3.30/3.60/3.90) + small boxes en su WorkZone.
        p3 = self.log['phase3']
        order = p3['order']                      # ['C', 'B', 'A']
        wz = self.coord.work_zone

        # Estado inicial mutable para fase 3
        pb_states = {  # role_box -> (x, y, theta)
            'C': (11.20, 3.30, 0.0),
            'B': (11.20, 3.60, 0.0),
            'A': (11.20, 3.90, 0.0),
        }
        small_boxes_state = self._initial_small_boxes()
        stack_count_state = 0

        for unit_log, role in zip(p3['units'], order):
            # Driving pick
            dp = unit_log['drive_pick']
            for i in range(0, len(dp['t']), self.stride):
                pb_states[role] = (dp['x'][i], dp['y'][i], dp['theta'][i])
                frames.append({
                    'phase': 3,
                    'husky': last_husky,
                    'anymal': last_anymal,
                    'puzzlebots': self._pb_state_to_list(pb_states),
                    'big_boxes': last_big_boxes,
                    'small_boxes': dict(small_boxes_state),
                    'stack_count': stack_count_state,
                    'active': role,
                })
            # "Pick" instantaneo: la caja deja la mesa y queda con el bot
            # (la dibujamos junto al PuzzleBot)
            small_boxes_state[role] = (None, 'carried_by:' + role)

            # Driving place
            dp2 = unit_log['drive_place']
            for i in range(0, len(dp2['t']), self.stride):
                pb_states[role] = (dp2['x'][i], dp2['y'][i], dp2['theta'][i])
                frames.append({
                    'phase': 3,
                    'husky': last_husky,
                    'anymal': last_anymal,
                    'puzzlebots': self._pb_state_to_list(pb_states),
                    'big_boxes': last_big_boxes,
                    'small_boxes': dict(small_boxes_state),
                    'stack_count': stack_count_state,
                    'active': role,
                })
            # "Place" instantaneo: la caja queda en la pila
            stack_z = stack_count_state    # solo para etiqueta
            small_boxes_state[role] = (
                (wz.stack_xy[0], wz.stack_xy[1]),
                f'stack:{stack_count_state}',
            )
            stack_count_state += 1
            # Algunos frames "post-place" para que el ojo registre la pila
            for _ in range(8):
                frames.append({
                    'phase': 3,
                    'husky': last_husky,
                    'anymal': last_anymal,
                    'puzzlebots': self._pb_state_to_list(pb_states),
                    'big_boxes': last_big_boxes,
                    'small_boxes': dict(small_boxes_state),
                    'stack_count': stack_count_state,
                    'active': None,
                })

        return frames

    # ------------------------------------------------------------------
    def _initial_small_boxes(self):
        wz = self.coord.work_zone
        out = {}
        for name, box in wz.boxes.items():
            out[name] = ((float(box.xy[0]), float(box.xy[1])), 'on_table')
        return out

    def _pb_state_to_list(self, pb_states):
        # PB_C, PB_B, PB_A en ese orden
        return [pb_states['C'], pb_states['B'], pb_states['A']]

    def _puzzlebots_on_anymal(self, ax_, ay_, ayaw):
        """Devuelve poses de los 3 PB cuando van encima del ANYmal."""
        cs, sn = math.cos(ayaw), math.sin(ayaw)
        # 3 puntos a lo largo del eje longitudinal del ANYmal
        offsets_local = [(-0.30, 0.0), (0.0, 0.0), (0.30, 0.0)]
        out = []
        for lx, ly in offsets_local:
            wx = ax_ + cs * lx - sn * ly
            wy = ay_ + sn * lx + cs * ly
            out.append((wx, wy, ayaw))
        return out

    # ------------------------------------------------------------------
    def _draw_static(self, ax, mpatches):
        """Dibuja el corredor, la zona de trabajo y la pila destino."""
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

        # Zona de trabajo de fase 3 (mesa)
        wz = self.coord.work_zone
        xs = [b.xy[0] for b in wz.boxes.values()]
        ys = [b.xy[1] for b in wz.boxes.values()]
        x0 = min(xs) - 0.10
        x1 = max(xs) + 0.10
        y0 = min(ys) - 0.10
        y1 = max(ys) + 0.10
        table = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor='#bcd9b3', edgecolor='black',
                                   linewidth=0.8, alpha=0.5)
        ax.add_patch(table)
        ax.text((x0 + x1) / 2, y0 - 0.10, 'mesa cajas',
                ha='center', fontsize=7)

        # Marca de la pila destino
        ax.plot(wz.stack_xy[0], wz.stack_xy[1], 'kx', markersize=8)
        ax.text(wz.stack_xy[0] + 0.05, wz.stack_xy[1] + 0.05,
                'pila', fontsize=7)

        # Marca del target del ANYmal
        ax.plot(*self.coord.ANYMAL_TARGET, 'r*', markersize=10)
        ax.text(self.coord.ANYMAL_TARGET[0] + 0.10,
                self.coord.ANYMAL_TARGET[1],
                'p_destino', fontsize=7, color='red')

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

        # PuzzleBots
        for (px, py, pth) in frame['puzzlebots']:
            _draw_puzzlebot(ax, mpatches, px, py, pth)

        # Cajas pequenas
        small_side = 0.04   # 4 cm visibles, > el side fisico, para legibilidad
        for name, (pos, status) in frame['small_boxes'].items():
            if status == 'on_table':
                cx, cy = pos
                _draw_box(ax, mpatches, cx, cy, small_side,
                          color='#e377c2', label=name)
            elif status.startswith('carried_by'):
                # Dibujar pegada al PB activo (si existe)
                role = status.split(':', 1)[1]
                idx = {'C': 0, 'B': 1, 'A': 2}[role]
                px, py, _ = frame['puzzlebots'][idx]
                _draw_box(ax, mpatches, px + 0.10, py, small_side,
                          color='#e377c2', label=name)
            elif status.startswith('stack'):
                cx, cy = pos
                # Pequeno offset vertical visual segun la capa
                layer = int(status.split(':', 1)[1])
                _draw_box(ax, mpatches, cx, cy + 0.04 * layer, small_side,
                          color='#e377c2', label=name)

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
            1: "Fase 1: Husky despeja el corredor",
            2: "Fase 2: ANYmal trota al destino con 3 PuzzleBots",
            3: "Fase 3: PuzzleBots apilan cajas (orden C, B, A)",
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
