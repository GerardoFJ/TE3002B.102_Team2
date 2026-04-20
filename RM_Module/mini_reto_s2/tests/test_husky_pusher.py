"""Tests del modulo husky_pusher."""

import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mini_reto_s2.husky_pusher import (   # noqa: E402
    Box,
    CorridorWorld,
    Lidar2D,
    detect_boxes_from_scan,
    HuskyPusher,
)
from mini_reto_s2.robots_base import HuskyA200   # noqa: E402


class TestCorridorWorld(unittest.TestCase):

    def test_default_boxes_inside(self):
        world = CorridorWorld()
        self.assertEqual(len(world.boxes_in_corridor()), 3)
        self.assertFalse(world.all_clear())

    def test_box_outside_marked_as_clear(self):
        world = CorridorWorld()
        for b in world.boxes:
            b.y = -1.0
        self.assertEqual(len(world.boxes_in_corridor()), 0)
        self.assertTrue(world.all_clear())


class TestLidar(unittest.TestCase):

    def test_hits_single_box_in_front(self):
        lidar = Lidar2D(n_beams=11, max_range=10.0, fov_deg=60)
        boxes = [Box(2.0, 0.0, side=0.3)]
        angles, ranges = lidar.scan((0.0, 0.0, 0.0), boxes)
        # El rayo central (frente) deberia chocar con la cara mas cercana
        center = ranges[len(ranges) // 2]
        self.assertAlmostEqual(center, 2.0 - 0.3, places=6)

    def test_max_range_when_no_box(self):
        lidar = Lidar2D(n_beams=5, max_range=4.0, fov_deg=30)
        angles, ranges = lidar.scan((0.0, 0.0, 0.0), [])
        np.testing.assert_array_equal(ranges, np.full(5, 4.0))

    def test_box_behind_robot_not_hit_with_narrow_fov(self):
        lidar = Lidar2D(n_beams=5, max_range=4.0, fov_deg=30)
        boxes = [Box(-2.0, 0.0, side=0.3)]
        angles, ranges = lidar.scan((0.0, 0.0, 0.0), boxes)
        # Con FOV de 30 grados al frente, no se ve nada atras
        np.testing.assert_array_equal(ranges, np.full(5, 4.0))


class TestDetector(unittest.TestCase):

    def test_detects_three_boxes(self):
        # Lidar 360 deg desde un punto NO colineal con las 3 cajas
        # (colineales se ocluyen entre si).
        lidar = Lidar2D(n_beams=361, max_range=15.0, fov_deg=359)
        world = CorridorWorld()
        angles, ranges = lidar.scan((5.0, 5.0, 0.0), world.boxes)
        centroids = detect_boxes_from_scan(angles, ranges, lidar.max_range)
        self.assertGreaterEqual(len(centroids), 3)

    def test_collinear_boxes_only_first_visible(self):
        # Si miramos las 3 cajas en linea recta, solo se ve la mas cercana
        # (las otras quedan ocluidas).
        lidar = Lidar2D(n_beams=181, max_range=15.0, fov_deg=120)
        world = CorridorWorld()
        angles, ranges = lidar.scan((0.0, 2.0, 0.0), world.boxes)
        centroids = detect_boxes_from_scan(angles, ranges, lidar.max_range)
        self.assertEqual(len(centroids), 1)


class TestPushIntegration(unittest.TestCase):
    """Test end-to-end de la fase 1 del reto."""

    def test_clear_corridor_on_grass(self):
        husky = HuskyA200()
        husky.set_terrain("grass")
        husky.reset(x=0.0, y=2.0, theta=0.0)
        world = CorridorWorld()
        pusher = HuskyPusher(husky, world)
        log = pusher.clear_corridor(dt=0.05)
        self.assertTrue(log['success'])
        # Las 3 cajas fuera del corredor
        for b in world.boxes:
            self.assertFalse(world.box_in_corridor(b),
                             f"{b.name} sigue dentro: y={b.y:.2f}")
        # v_cmd y v_real consistentes (slip compensado)
        v_cmd = np.array(log['v_cmd'])
        v_real = np.array(log['v_real'])
        np.testing.assert_allclose(v_cmd, v_real, atol=1e-9)

    def test_inverse_kinematics_compensates_slip(self):
        """Sin compensacion, v_real < v_cmd en grass; con compensacion, igual."""
        husky = HuskyA200()
        husky.set_terrain("grass")     # slip = 0.85
        # Sin compensacion
        wR1, wR2, wL1, wL2 = husky.inverse_kinematics(1.0, 0.0,
                                                      compensate_slip=False)
        v_real, _ = husky.forward_kinematics(wR1, wR2, wL1, wL2)
        self.assertAlmostEqual(v_real, 0.85, places=6)
        # Con compensacion
        wR1, wR2, wL1, wL2 = husky.inverse_kinematics(1.0, 0.0,
                                                      compensate_slip=True)
        v_real, _ = husky.forward_kinematics(wR1, wR2, wL1, wL2)
        self.assertAlmostEqual(v_real, 1.0, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
