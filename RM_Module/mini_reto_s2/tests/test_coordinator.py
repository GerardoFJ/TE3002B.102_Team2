"""Tests del MissionCoordinator (las 3 fases del mini reto)."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mini_reto_s2.coordinator import (   # noqa: E402
    MissionCoordinator,
    WorkZone,
    SmallBox,
    PuzzleBotUnit,
    drive_puzzlebot_to,
)
from mini_reto_s2.robots_base import PuzzleBot   # noqa: E402


class TestWorkZone(unittest.TestCase):

    def test_default_three_boxes(self):
        wz = WorkZone()
        self.assertEqual(set(wz.boxes.keys()), {'A', 'B', 'C'})
        for b in wz.boxes.values():
            self.assertFalse(b.placed)
        self.assertEqual(wz.stack_count, 0)

    def test_layer_height_positive(self):
        wz = WorkZone()
        self.assertGreater(wz.layer_height, 0.0)


class TestDrivePuzzleBotTo(unittest.TestCase):

    def test_reaches_target(self):
        pb = PuzzleBot()
        pb.reset(0.0, 0.0, 0.0)
        log = drive_puzzlebot_to(pb, (1.5, 0.5), max_steps=600)
        x, y, _ = pb.get_pose()
        dist = np.hypot(1.5 - x, 0.5 - y)
        self.assertLess(dist, 0.10)
        self.assertGreater(len(log['t']), 0)


class TestPuzzleBotUnit(unittest.TestCase):

    def test_unit_initial_state(self):
        u = PuzzleBotUnit('PB_X', 'A', start_xy=(2.0, 1.0))
        x, y, _ = u.base.get_pose()
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertEqual(u.role_box, 'A')


class TestMissionCoordinator(unittest.TestCase):
    """Test end-to-end del coordinador completo."""

    @classmethod
    def setUpClass(cls):
        # La fase 1 + 2 + 3 toma varios miles de pasos -> ejecutar 1 sola vez
        cls.coord = MissionCoordinator()
        cls.log = cls.coord.run()

    def test_fsm_reaches_done(self):
        self.assertEqual(self.coord.phase, 'DONE')

    def test_phase1_success(self):
        self.assertTrue(self.log['success']['phase1'])

    def test_phase2_success(self):
        self.assertTrue(self.log['success']['phase2'])
        self.assertLess(self.log['phase2']['final_error'], 0.15)

    def test_phase3_all_boxes_placed(self):
        self.assertTrue(self.log['success']['phase3'])
        ph3 = self.log['phase3']
        for name, info in ph3['final_box_positions'].items():
            with self.subTest(box=name):
                self.assertTrue(info['placed'])

    def test_phase3_stack_order(self):
        """Las cajas deben quedar apiladas C abajo, B en medio, A arriba."""
        ph3 = self.log['phase3']
        z_a = ph3['final_box_positions']['A']['z']
        z_b = ph3['final_box_positions']['B']['z']
        z_c = ph3['final_box_positions']['C']['z']
        self.assertLess(z_c, z_b)
        self.assertLess(z_b, z_a)

    def test_phase3_stack_xy_aligned(self):
        """Las 3 cajas deben quedar en el mismo (x, y) (la pila)."""
        ph3 = self.log['phase3']
        target_xy = self.coord.work_zone.stack_xy
        for name, info in ph3['final_box_positions'].items():
            with self.subTest(box=name):
                np.testing.assert_allclose(info['xy'], target_xy, atol=1e-9)

    def test_phase3_no_singularities(self):
        """Ningun grasp/place del brazo debe ser singular."""
        for u in self.log['phase3']['units']:
            with self.subTest(unit=u['unit_name']):
                self.assertFalse(u['singular_grasp'])
                self.assertFalse(u['singular_place'])

    def test_phase3_torques_nonzero(self):
        """tau = J^T f debe ser no nulo (hay fuerza de agarre aplicada)."""
        for u in self.log['phase3']['units']:
            with self.subTest(unit=u['unit_name']):
                self.assertGreater(np.linalg.norm(u['tau_grasp']), 0.0)


class TestPhaseAssertions(unittest.TestCase):
    """La FSM debe rechazar llamadas fuera de orden."""

    def test_run_phase2_before_phase1_raises(self):
        coord = MissionCoordinator()
        with self.assertRaises(AssertionError):
            coord.run_phase2()

    def test_run_phase3_before_phase2_raises(self):
        coord = MissionCoordinator()
        coord.run_phase1()
        with self.assertRaises(AssertionError):
            coord.run_phase3()


if __name__ == '__main__':
    unittest.main(verbosity=2)
