"""Tests del modulo anymal_gait.

Verifica:
    - El IK planar 2-link es inverso exacto de la FK de la pata.
    - El gait es periodico con el periodo configurado.
    - Las patas diagonales (LF/RH y RF/LH) estan en fase entre si.
    - El simulador llega al destino con error < 0.15 m y sin violaciones
      del determinante del Jacobiano.

Ejecutar:
    python3 -m unittest tests.test_anymal_gait -v
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mini_reto_s2.anymal_gait import (   # noqa: E402
    ANYmalTrotGait,
    ANYmalNavigator,
    simulate_anymal_to_target,
    _planar_2link_ik,
)
from mini_reto_s2.robots_base import ANYmal   # noqa: E402


class TestPlanarIK(unittest.TestCase):
    """El IK planar 2-link debe ser inverso exacto de la FK simplificada."""

    def setUp(self):
        self.l1 = 0.35
        self.l2 = 0.33

    def _fk_planar(self, q2, q3):
        x = self.l1 * np.sin(q2) + self.l2 * np.sin(q2 + q3)
        z = -self.l1 * np.cos(q2) - self.l2 * np.cos(q2 + q3)
        return x, z

    def test_round_trip(self):
        """FK -> IK -> FK reproduce el punto."""
        configs = [
            (0.7, -1.4),
            (0.5, -1.0),
            (1.0, -2.0),
            (0.2, -0.6),
        ]
        for q2, q3 in configs:
            x, z = self._fk_planar(q2, q3)
            q2_back, q3_back = _planar_2link_ik(x, z, self.l1, self.l2)
            x_back, z_back = self._fk_planar(q2_back, q3_back)
            with self.subTest(q2=q2, q3=q3):
                self.assertAlmostEqual(x, x_back, places=8)
                self.assertAlmostEqual(z, z_back, places=8)

    def test_unreachable_raises(self):
        with self.assertRaises(ValueError):
            _planar_2link_ik(2.0, -0.1, self.l1, self.l2)


class TestTrotGait(unittest.TestCase):
    """Propiedades estructurales del generador de trote."""

    def setUp(self):
        self.bot = ANYmal()
        self.gait = ANYmalTrotGait(self.bot)

    def test_phase_offsets(self):
        """LF y RH en fase, RF y LH en fase opuesta."""
        self.assertEqual(self.gait.PHASE_OFFSETS['LF'],
                         self.gait.PHASE_OFFSETS['RH'])
        self.assertEqual(self.gait.PHASE_OFFSETS['RF'],
                         self.gait.PHASE_OFFSETS['LH'])
        self.assertNotEqual(self.gait.PHASE_OFFSETS['LF'],
                            self.gait.PHASE_OFFSETS['RF'])

    def test_periodicity(self):
        """foot_target debe ser periodico con periodo = self.period."""
        for name in ANYmal.LEG_NAMES:
            p0 = self.gait.foot_target(name, 0.0, v_forward=0.3)
            p1 = self.gait.foot_target(name, self.gait.period, v_forward=0.3)
            with self.subTest(leg=name):
                np.testing.assert_allclose(p0, p1, atol=1e-12)

    def test_diagonals_in_phase(self):
        """Pares diagonales estan en mismo punto de fase en cualquier t."""
        for t in (0.05, 0.13, 0.27, 0.41):
            p_lf = self.gait.foot_target('LF', t, v_forward=0.3)
            p_rh = self.gait.foot_target('RH', t, v_forward=0.3)
            with self.subTest(t=t):
                # Solo comparamos x y z (lateral y depende del side)
                self.assertAlmostEqual(p_lf[0], p_rh[0], places=12)
                self.assertAlmostEqual(p_lf[2], p_rh[2], places=12)

    def test_swing_lifts_foot(self):
        """Durante el swing el pie debe estar mas alto que en stance."""
        # En t=0 la pata LF empieza en stance (offset=0).
        z_stance = self.gait.foot_target('LF', 0.0, v_forward=0.3)[2]
        # A medio periodo entra en swing y deberia estar mas arriba
        # cerca del pico (3/4 del periodo aprox).
        t_mid_swing = self.gait.period * 0.75
        z_swing = self.gait.foot_target('LF', t_mid_swing,
                                        v_forward=0.3)[2]
        self.assertGreater(z_swing, z_stance)


class TestSimulationToTarget(unittest.TestCase):
    """El lazo cerrado debe llegar al destino con error pequeno."""

    def test_reaches_phase2_target(self):
        bot = ANYmal()
        log = simulate_anymal_to_target(
            bot, target_xy=(11.0, 3.6), T_max=60.0, dt=0.01)
        # Condicion de exito de la fase 2 del reto
        self.assertTrue(log['success'])
        self.assertLess(log['final_error'], 0.15)
        # Cero violaciones de singularidad en la marcha
        for name, n in log['violations'].items():
            with self.subTest(leg=name):
                self.assertEqual(n, 0)

    def test_navigator_stops_at_target(self):
        nav = ANYmalNavigator()
        # Ya en el target -> comando cero
        v, w, d = nav.compute_command((1.0, 1.0), 0.0, (1.0, 1.0))
        self.assertEqual(v, 0.0)
        self.assertEqual(w, 0.0)
        # Lejos -> avanza
        v, w, d = nav.compute_command((0.0, 0.0), 0.0, (5.0, 0.0))
        self.assertGreater(v, 0.0)
        self.assertAlmostEqual(w, 0.0, places=8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
