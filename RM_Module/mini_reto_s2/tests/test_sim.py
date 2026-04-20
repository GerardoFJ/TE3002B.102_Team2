"""Tests del modulo sim (sin abrir matplotlib).

Solo testeamos la logica pura: el frame builder que toma los logs del
coordinador y produce una lista de snapshots homogeneas. Los helpers
de dibujo (_draw_*) requieren matplotlib y no se cubren aqui.
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mini_reto_s2.sim import MissionVisualizer   # noqa: E402


class TestFrameBuilder(unittest.TestCase):
    """Test del builder de frames (no requiere matplotlib)."""

    @classmethod
    def setUpClass(cls):
        # Una sola corrida para todo el test case (varios miles de pasos)
        cls.vis = MissionVisualizer(stride=20)
        cls.vis.run()
        cls.frames = cls.vis._build_frames()

    def test_frames_non_empty(self):
        self.assertGreater(len(self.frames), 0)

    def test_all_three_phases_present(self):
        phases = {f['phase'] for f in self.frames}
        self.assertEqual(phases, {1, 2, 3})

    def test_frame_schema(self):
        """Cada frame contiene las claves esperadas."""
        required = {'phase', 'husky', 'anymal', 'puzzlebots',
                    'big_boxes', 'small_boxes', 'stack_count'}
        for f in self.frames[::25]:
            with self.subTest(idx=self.frames.index(f)):
                self.assertTrue(required.issubset(f.keys()))
                self.assertEqual(len(f['puzzlebots']), 3)
                self.assertEqual(len(f['big_boxes']), 3)
                self.assertEqual(set(f['small_boxes'].keys()),
                                 {'A', 'B', 'C'})

    def test_phase_ordering(self):
        """Las fases aparecen en orden 1 -> 2 -> 3 (sin retroceso)."""
        last = 0
        for f in self.frames:
            self.assertGreaterEqual(f['phase'], last)
            last = f['phase']

    def test_phase1_husky_moves_anymal_static(self):
        """En fase 1 el husky se mueve y el anymal NO."""
        phase1 = [f for f in self.frames if f['phase'] == 1]
        anymal_x = {f['anymal'][0] for f in phase1}
        anymal_y = {f['anymal'][1] for f in phase1}
        self.assertEqual(len(anymal_x), 1)   # constante
        self.assertEqual(len(anymal_y), 1)
        # Husky se mueve
        husky_xs = [f['husky'][0] for f in phase1]
        self.assertGreater(max(husky_xs) - min(husky_xs), 1.0)

    def test_phase2_anymal_reaches_target(self):
        """Al final de fase 2 el ANYmal esta cerca de p_destino."""
        phase2 = [f for f in self.frames if f['phase'] == 2]
        self.assertGreater(len(phase2), 0)
        last = phase2[-1]
        ax_, ay_, _ = last['anymal']
        tx, ty = self.vis.coord.ANYMAL_TARGET
        self.assertLess(np.hypot(ax_ - tx, ay_ - ty), 0.30)

    def test_phase2_puzzlebots_track_anymal(self):
        """En fase 2 los 3 PuzzleBots se mueven con el ANYmal."""
        phase2 = [f for f in self.frames if f['phase'] == 2]
        # Cada pb deberia tener al menos 2 posiciones distintas
        for idx in range(3):
            xs = {round(f['puzzlebots'][idx][0], 4) for f in phase2}
            with self.subTest(pb=idx):
                self.assertGreater(len(xs), 1)

    def test_phase3_stacks_grow(self):
        """stack_count crece monotonamente en fase 3 hasta llegar a 3."""
        phase3 = [f for f in self.frames if f['phase'] == 3]
        self.assertGreater(len(phase3), 0)
        counts = [f['stack_count'] for f in phase3]
        # Monotono no decreciente
        for a, b in zip(counts, counts[1:]):
            self.assertLessEqual(a, b)
        self.assertEqual(max(counts), 3)

    def test_phase3_final_boxes_at_stack(self):
        """En el ultimo frame las 3 cajas pequenas estan en la pila."""
        last = self.frames[-1]
        wz = self.vis.coord.work_zone
        for name, (pos, status) in last['small_boxes'].items():
            with self.subTest(box=name):
                self.assertTrue(status.startswith('stack'))
                np.testing.assert_allclose(pos, wz.stack_xy, atol=1e-9)


class TestStrideRespected(unittest.TestCase):
    """El parametro stride debe afectar el numero de frames generados."""

    def test_stride_reduces_frames(self):
        v1 = MissionVisualizer(stride=5)
        v1.run()
        n1 = len(v1._build_frames())

        v2 = MissionVisualizer(stride=30)
        v2.run()
        n2 = len(v2._build_frames())

        self.assertGreater(n1, n2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
