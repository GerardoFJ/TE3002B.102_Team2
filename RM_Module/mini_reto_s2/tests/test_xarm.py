"""Tests del xArm 6 simplificado y la fase 2.5 del coordinador."""

import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mini_reto_s2.xarm import XArm                 # noqa: E402
from mini_reto_s2.coordinator import MissionCoordinator  # noqa: E402


class TestXArmKinematics(unittest.TestCase):

    def setUp(self):
        self.arm = XArm()

    def test_home_pose_reachable(self):
        p = self.arm.forward_kinematics(self.arm.q_home)
        self.assertEqual(p.shape, (3,))
        # Home cae dentro del semi-espacio z > 0
        self.assertGreater(p[2], 0.0)

    def test_fk_ik_roundtrip_mesa(self):
        """IK(FK(q)) ~ q para un punto alcanzable en el marco mundo."""
        # Drop de PB 'B' con la geometria default del coordinator
        p_world = np.array([11.50, 3.65, 0.10])
        q = self.arm.ik_world(p_world)
        # Re-FK y comparar
        p_rec_arm = self.arm.forward_kinematics(q)
        p_rec_world = self.arm.world_from_arm(p_rec_arm)
        np.testing.assert_allclose(p_rec_world, p_world, atol=1e-4)

    def test_ik_out_of_workspace_raises(self):
        p_far = np.array([50.0, 50.0, 0.0])
        with self.assertRaises(ValueError):
            self.arm.ik_world(p_far)

    def test_link_points_chain_lengths(self):
        """Distancias hombro->codo y codo->muneca respetan L2, L3."""
        pts = self.arm.link_points_world(self.arm.q_home)
        # pts = base, shoulder, elbow, wrist, tcp
        d_upper = np.linalg.norm(pts[2] - pts[1])
        d_fore = np.linalg.norm(pts[3] - pts[2])
        self.assertAlmostEqual(d_upper, self.arm.L2, places=5)
        self.assertAlmostEqual(d_fore, self.arm.L3, places=5)

    def test_pick_place_path_endpoints(self):
        """La trayectoria empieza y termina en home, pasa por pick y place."""
        p_pick = np.array([11.0, 3.60, 0.30])
        p_place = np.array([11.20, 3.60, 0.10])
        path = self.arm.pick_place_cartesian_path(
            p_pick, p_place, approach_height=0.15, n_seg=10)

        home_arm = self.arm.forward_kinematics(self.arm.q_home)
        home_w = self.arm.world_from_arm(home_arm)
        np.testing.assert_allclose(path[0], home_w, atol=1e-9)
        np.testing.assert_allclose(path[-1], home_w, atol=1e-9)
        # Debe pasar cerca de p_pick y p_place en algun punto
        d_pick_min = min(np.linalg.norm(p - p_pick) for p in path)
        d_place_min = min(np.linalg.norm(p - p_place) for p in path)
        self.assertLess(d_pick_min, 1e-6)
        self.assertLess(d_place_min, 1e-6)


class TestPhase2_5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.coord = MissionCoordinator()
        cls.coord.run()

    def test_phase2_5_in_log(self):
        self.assertIn('phase2_5', self.coord.log)
        self.assertIsNotNone(self.coord.log['phase2_5'])
        self.assertTrue(self.coord.log['success']['phase2_5'])

    def test_phase2_5_has_three_units(self):
        p25 = self.coord.log['phase2_5']
        self.assertEqual(len(p25['units']), 3)
        roles = [u['role'] for u in p25['units']]
        self.assertEqual(roles, list(MissionCoordinator.TRANSFER_ORDER))

    def test_pb_drop_positions_reachable(self):
        """Todos los drops del xArm estan dentro de su reach."""
        for role, drop in MissionCoordinator.PB_TABLE_DROP.items():
            with self.subTest(role=role):
                # IK no debe tirar ValueError -> drop alcanzable
                import numpy as np
                self.coord.xarm.ik_world(
                    np.array([drop[0], drop[1], 0.10]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
