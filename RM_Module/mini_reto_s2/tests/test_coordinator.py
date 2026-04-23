"""Tests del MissionCoordinator (las 3 fases del mini reto)."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mini_reto_s2.coordinator import (   # noqa: E402
    MissionCoordinator,
    PuzzleBotUnit,
)


class TestPuzzleBotUnit(unittest.TestCase):

    def test_unit_initial_state(self):
        u = PuzzleBotUnit('PB_X', 'A', start_xy=(2.0, 1.0))
        x, y, _ = u.base.get_pose()
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertEqual(u.role, 'A')


class TestMissionCoordinator(unittest.TestCase):
    """Test end-to-end del coordinador completo."""

    @classmethod
    def setUpClass(cls):
        cls.coord = MissionCoordinator()
        cls.log = cls.coord.run()

    def test_fsm_reaches_done(self):
        self.assertEqual(self.coord.phase, 'DONE')

    def test_phase1_success(self):
        self.assertTrue(self.log['success']['phase1'])

    def test_phase2_success(self):
        self.assertTrue(self.log['success']['phase2'])
        self.assertLess(self.log['phase2']['final_error'], 0.15)

    def test_phase2_5_success(self):
        self.assertTrue(self.log['success']['phase2_5'])
        self.assertEqual(len(self.log['phase2_5']['units']), 3)


class TestPhaseAssertions(unittest.TestCase):
    """La FSM debe rechazar llamadas fuera de orden."""

    def test_run_phase2_before_phase1_raises(self):
        coord = MissionCoordinator()
        with self.assertRaises(AssertionError):
            coord.run_phase2()

    def test_run_phase2_5_before_phase2_raises(self):
        coord = MissionCoordinator()
        coord.run_phase1()
        with self.assertRaises(AssertionError):
            coord.run_phase2_5()


if __name__ == '__main__':
    unittest.main(verbosity=2)
