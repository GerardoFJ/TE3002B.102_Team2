"""Tests unitarios del mini brazo PuzzleBotArm.

Cubre los requisitos de la rubrica del mini reto:
    - FK correcta en poses canonicas
    - IK round-trip (IK -> FK reproduce el punto)
    - Jacobiano analitico coincide con diferencias finitas
    - Deteccion de singularidad cuando el brazo esta extendido
    - tau = J^T f produce torques consistentes

Ejecutar:
    cd robotics_ws/mini_reto_s2
    python -m unittest tests.test_puzzlebot_arm -v
"""

import os
import sys
import unittest

import numpy as np

# Permitir ejecutar como `python -m unittest` desde la raiz del paquete
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mini_reto_s2.puzzlebot_arm import PuzzleBotArm  # noqa: E402


class TestForwardKinematics(unittest.TestCase):
    """FK debe devolver el punto correcto en poses canonicas."""

    def setUp(self):
        self.arm = PuzzleBotArm(l1=0.10, l2=0.08, l3=0.06)

    def test_home_pose_extended(self):
        """q=(0,0,0): brazo extendido sobre +x al nivel del hombro."""
        p = self.arm.forward_kinematics(np.zeros(3))
        np.testing.assert_allclose(p, [0.14, 0.0, 0.10], atol=1e-9)

    def test_base_rotation_90deg(self):
        """q1=pi/2: el brazo extendido apunta sobre +y."""
        p = self.arm.forward_kinematics([np.pi / 2, 0.0, 0.0])
        np.testing.assert_allclose(p, [0.0, 0.14, 0.10], atol=1e-9)

    def test_shoulder_up(self):
        """q2=pi/2: brazo apuntando recto hacia arriba."""
        p = self.arm.forward_kinematics([0.0, np.pi / 2, 0.0])
        np.testing.assert_allclose(p, [0.0, 0.0, 0.10 + 0.14], atol=1e-9)

    def test_elbow_folded(self):
        """q3=-pi: antebrazo regresa sobre el brazo superior."""
        p = self.arm.forward_kinematics([0.0, 0.0, -np.pi])
        # reach = l2 - l3 = 0.02
        np.testing.assert_allclose(p, [0.02, 0.0, 0.10], atol=1e-9)


class TestInverseKinematics(unittest.TestCase):
    """IK debe ser inverso exacto de FK en puntos alcanzables."""

    def setUp(self):
        self.arm = PuzzleBotArm(l1=0.10, l2=0.08, l3=0.06)

    def _round_trip(self, p_des, elbow_up=True):
        q = self.arm.inverse_kinematics(p_des, elbow_up=elbow_up)
        p_back = self.arm.forward_kinematics(q)
        return q, p_back

    def test_round_trip_canonical_points(self):
        """Cada punto reproduce p tras IK->FK."""
        targets = [
            [0.10, 0.00, 0.12],
            [0.07, 0.05, 0.08],
            [0.05, -0.05, 0.13],
            [0.00, 0.10, 0.10],
            [-0.08, 0.04, 0.11],
        ]
        for p in targets:
            with self.subTest(p=p):
                q, p_back = self._round_trip(p)
                np.testing.assert_allclose(p_back, p, atol=1e-8)

    def test_round_trip_elbow_up_vs_down(self):
        """Las dos ramas (codo arriba/abajo) producen el mismo punto."""
        p_des = [0.09, 0.03, 0.07]
        _, p_up = self._round_trip(p_des, elbow_up=True)
        _, p_down = self._round_trip(p_des, elbow_up=False)
        np.testing.assert_allclose(p_up, p_des, atol=1e-8)
        np.testing.assert_allclose(p_down, p_des, atol=1e-8)

    def test_unreachable_raises(self):
        """Pedir un punto fuera del workspace lanza ValueError."""
        with self.assertRaises(ValueError):
            self.arm.inverse_kinematics([1.0, 0.0, 0.10])


class TestJacobian(unittest.TestCase):
    """El Jacobiano analitico debe coincidir con diferencias finitas."""

    def setUp(self):
        self.arm = PuzzleBotArm(l1=0.10, l2=0.08, l3=0.06)

    def _numerical_jacobian(self, q, eps=1e-6):
        """Jacobiano numerico por diferencias centradas."""
        J = np.zeros((3, 3))
        for j in range(3):
            dq = np.zeros(3)
            dq[j] = eps
            p_plus = self.arm.forward_kinematics(q + dq)
            p_minus = self.arm.forward_kinematics(q - dq)
            J[:, j] = (p_plus - p_minus) / (2.0 * eps)
        return J

    def test_jacobian_vs_numerical(self):
        """Compara J analitico vs numerico en varias configuraciones."""
        configs = [
            [0.0, 0.5, -1.0],
            [0.3, -0.2, -0.8],
            [-0.4, 0.7, -1.2],
            [1.2, 0.0, -0.3],
        ]
        for q in configs:
            with self.subTest(q=q):
                J_analytic = self.arm.jacobian(q)
                J_num = self._numerical_jacobian(np.asarray(q))
                np.testing.assert_allclose(J_analytic, J_num, atol=1e-6)

    def test_extended_arm_is_singular(self):
        """Brazo extendido (q3=0) reduce el rango y el det se anula."""
        q = np.array([0.0, 0.0, 0.0])
        J = self.arm.jacobian(q)
        self.assertAlmostEqual(np.linalg.det(J), 0.0, places=8)
        self.assertTrue(self.arm.is_singular(q))


class TestForceControl(unittest.TestCase):
    """Verifica el mapeo tau = J^T f."""

    def setUp(self):
        self.arm = PuzzleBotArm(l1=0.10, l2=0.08, l3=0.06)
        self.arm.reset(np.array([0.0, 0.5, -1.0]))

    def test_pure_vertical_force_zero_torque_on_base(self):
        """Una fuerza vertical pura no genera torque en la base (q1).

        Razon: la columna 1 del Jacobiano (d p / d q1) tiene componente z = 0,
        asi que (J^T f)[0] = 0 cuando f = (0,0,fz).
        """
        f = np.array([0.0, 0.0, -10.0])
        tau = self.arm.force_to_torque(f)
        self.assertAlmostEqual(tau[0], 0.0, places=12)

    def test_lateral_force_produces_base_torque(self):
        """Una fuerza con componente lateral si genera torque en q1."""
        f = np.array([0.0, 5.0, 0.0])
        # Ponemos el brazo apuntando sobre +x: la fuerza en y produce
        # torque positivo en la base.
        self.arm.reset(np.array([0.0, 0.5, -1.0]))
        tau = self.arm.force_to_torque(f)
        self.assertGreater(abs(tau[0]), 1e-6)

    def test_grasp_box_returns_consistent_data(self):
        """grasp_box devuelve trayectoria consistente y sin singularidad."""
        self.arm.reset(np.array([0.0, 0.5, -1.0]))
        result = self.arm.grasp_box(np.array([0.10, 0.0, 0.05]),
                                    grip_force=5.0, n_steps=10)
        self.assertEqual(result['q_path'].shape, (11, 3))
        self.assertEqual(result['p_path'].shape, (11, 3))
        self.assertEqual(result['tau_grip'].shape, (3,))
        self.assertFalse(result['singular'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
