"""Wrapper rclpy de un PuzzleBot + mini brazo (fase 3 del mini reto).

Cada instancia de este nodo representa UNA unidad PuzzleBot que ejecuta
un ciclo completo pick & place sobre una caja pequena (A, B o C).
La logica vive en ``coordinator.PuzzleBotUnit`` + ``puzzlebot_arm.PuzzleBotArm``.

Para correr 3 PuzzleBots se lanzan 3 instancias con el parametro
``role_box`` distinto (ver ``launch/mission.launch.py``).

Topicos publicados (todos prefijados con el parametro ``name``)
---------------------------------------------------------------
<name>/odom         nav_msgs/Odometry          Pose 2D base
<name>/cmd_vel      geometry_msgs/Twist        v, omega del controlador
<name>/joint_states sensor_msgs/JointState     3 articulaciones del brazo
<name>/torque       std_msgs/Float32MultiArray tau = J^T f del grip
<name>/status       std_msgs/String            Fase actual del ciclo

Parametros
----------
name        str   Prefijo del namespace y nombre del nodo (ej 'pb_c')
role_box    str   Caja a recoger ('A', 'B' o 'C')
start_x     float Posicion inicial X del bot (default depende de role_box)
start_y     float Posicion inicial Y del bot
target_x    float X de la pila destino (default 12.30)
target_y    float Y de la pila destino (default 3.85)
stack_layer int   Capa en la que va a colocar la caja (0 = abajo)
dt          float Periodo del timer / paso de simulacion (default 0.05)
loop        bool  Reinicia el replay al terminar (default False)
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from mini_reto_s2.coordinator import PuzzleBotUnit, drive_puzzlebot_to


ARM_JOINT_NAMES = ['arm_q1', 'arm_q2', 'arm_q3']

# Posiciones de inicio por defecto (mismas que MissionCoordinator)
DEFAULT_START = {
    'A': (11.20, 3.90),
    'B': (11.20, 3.60),
    'C': (11.20, 3.30),
}

# Z de cada capa cuando se apila C, B, A (table_z=0.05, layer~0.045)
TABLE_Z = 0.05
LAYER_HEIGHT = 0.045


def _yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _build_pick_place_frames(unit, box_xy, box_z, stack_xy, target_z):
    """Precomputa todos los frames de un ciclo pick & place.

    Cada frame es un tuple (base_pose, q_arm, status, tau).
    Reutiliza el helper drive_puzzlebot_to y arm.grasp_box; toda la
    fisica vive en esos modulos. Aqui solo unificamos los logs en una
    secuencia plana para el replay del timer ROS2.
    """
    frames = []

    # 1) drive to pick (~0.12 m al oeste de la caja)
    approach_pick = np.array([box_xy[0] - 0.12, box_xy[1]])
    drive_log = drive_puzzlebot_to(unit.base, approach_pick)
    q_home = unit.arm.q.copy()
    for i in range(len(drive_log['t'])):
        frames.append({
            'base': (drive_log['x'][i], drive_log['y'][i],
                     drive_log['theta'][i]),
            'v_cmd': drive_log['v_cmd'][i],
            'omega_cmd': drive_log['omega_cmd'][i],
            'q_arm': q_home.copy(),
            'tau': np.zeros(3),
            'status': 'driving_to_pick',
        })

    # 2) grasp (caja en marco brazo: 0.10 al frente)
    box_in_arm_frame = np.array([0.10, 0.0, box_z])
    grasp = unit.arm.grasp_box(box_in_arm_frame, grip_force=2.0, n_steps=15)
    bx, by, bth = unit.base.get_pose()
    for q in grasp['q_path']:
        frames.append({
            'base': (bx, by, bth),
            'v_cmd': 0.0,
            'omega_cmd': 0.0,
            'q_arm': np.array(q),
            'tau': np.array(grasp['tau_grip']),
            'status': 'grasping',
        })
    q_carry = np.array(grasp['q_path'][-1])

    # 3) drive to place
    approach_place = np.array([stack_xy[0] - 0.12, stack_xy[1]])
    drive_log_place = drive_puzzlebot_to(unit.base, approach_place)
    for i in range(len(drive_log_place['t'])):
        frames.append({
            'base': (drive_log_place['x'][i], drive_log_place['y'][i],
                     drive_log_place['theta'][i]),
            'v_cmd': drive_log_place['v_cmd'][i],
            'omega_cmd': drive_log_place['omega_cmd'][i],
            'q_arm': q_carry.copy(),
            'tau': np.array(grasp['tau_grip']),
            'status': 'driving_to_place',
        })

    # 4) place
    place_pos = np.array([0.10, 0.0, target_z])
    place = unit.arm.grasp_box(place_pos, grip_force=2.0, n_steps=15)
    bx, by, bth = unit.base.get_pose()
    for q in place['q_path']:
        frames.append({
            'base': (bx, by, bth),
            'v_cmd': 0.0,
            'omega_cmd': 0.0,
            'q_arm': np.array(q),
            'tau': np.array(place['tau_grip']),
            'status': 'placing',
        })

    # 5) marcador final
    bx, by, bth = unit.base.get_pose()
    q_done = np.array(place['q_path'][-1])
    frames.append({
        'base': (bx, by, bth),
        'v_cmd': 0.0, 'omega_cmd': 0.0,
        'q_arm': q_done,
        'tau': np.zeros(3),
        'status': 'done',
    })
    return frames


class PuzzleBotArmNode(Node):

    def __init__(self):
        super().__init__('puzzlebot_arm_node')

        self.declare_parameter('name', 'pb_a')
        self.declare_parameter('role_box', 'A')
        self.declare_parameter('start_x', 0.0)   # 0.0 = usar default
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('target_x', 12.30)
        self.declare_parameter('target_y', 3.85)
        self.declare_parameter('stack_layer', 0)
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('loop', False)

        self.name = self.get_parameter('name').value
        role = str(self.get_parameter('role_box').value).upper()
        if role not in ('A', 'B', 'C'):
            raise ValueError(f"role_box debe ser A/B/C, se recibio {role}")
        sx = float(self.get_parameter('start_x').value)
        sy = float(self.get_parameter('start_y').value)
        if sx == 0.0 and sy == 0.0:
            sx, sy = DEFAULT_START[role]
        target_xy = (
            float(self.get_parameter('target_x').value),
            float(self.get_parameter('target_y').value),
        )
        layer = int(self.get_parameter('stack_layer').value)
        target_z = TABLE_Z + layer * LAYER_HEIGHT
        self.dt = float(self.get_parameter('dt').value)
        self.loop = bool(self.get_parameter('loop').value)

        # Pose inicial de la caja en marco mundo (la traemos de la WorkZone
        # por defecto del coordinador para no duplicar constantes).
        from mini_reto_s2.coordinator import WorkZone
        wz = WorkZone()
        box = wz.boxes[role]
        box_xy = (float(box.xy[0]), float(box.xy[1]))
        box_z = float(box.z)

        # Unidad fisica + brazo
        self.unit = PuzzleBotUnit(self.name, role, start_xy=(sx, sy))

        self.get_logger().info(
            f"[{self.name}] Precomputando pick&place caja={role} "
            f"start=({sx:.2f},{sy:.2f}) target_layer={layer} "
            f"target_z={target_z:.3f}")
        self.frames = _build_pick_place_frames(
            self.unit, box_xy, box_z, target_xy, target_z)
        self.get_logger().info(
            f"[{self.name}] frames precomputados: {len(self.frames)}")

        # Publishers (namespaced por self.name)
        ns = self.name
        self.odom_pub = self.create_publisher(
            Odometry, f'{ns}/odom', 10)
        self.cmd_pub = self.create_publisher(
            Twist, f'{ns}/cmd_vel', 10)
        self.joint_pub = self.create_publisher(
            JointState, f'{ns}/joint_states', 10)
        self.torque_pub = self.create_publisher(
            Float32MultiArray, f'{ns}/torque', 10)
        self.status_pub = self.create_publisher(
            String, f'{ns}/status', 10)

        self.idx = 0
        self.timer = self.create_timer(self.dt, self._tick)

    # ------------------------------------------------------------------
    def _tick(self):
        if self.idx >= len(self.frames):
            if self.loop:
                self.idx = 0
            else:
                return
        frame = self.frames[self.idx]
        now = self.get_clock().now().to_msg()

        bx, by, bth = frame['base']

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'map'
        odom.child_frame_id = f'{self.name}/base'
        odom.pose.pose.position.x = float(bx)
        odom.pose.pose.position.y = float(by)
        qx, qy, qz, qw = _yaw_to_quat(float(bth))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(frame['v_cmd'])
        odom.twist.twist.angular.z = float(frame['omega_cmd'])
        self.odom_pub.publish(odom)

        twist = Twist()
        twist.linear.x = float(frame['v_cmd'])
        twist.angular.z = float(frame['omega_cmd'])
        self.cmd_pub.publish(twist)

        js = JointState()
        js.header.stamp = now
        js.name = ARM_JOINT_NAMES
        js.position = [float(v) for v in frame['q_arm']]
        self.joint_pub.publish(js)

        tau_msg = Float32MultiArray()
        tau_msg.data = [float(v) for v in frame['tau']]
        self.torque_pub.publish(tau_msg)

        self.status_pub.publish(String(data=frame['status']))

        self.idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = PuzzleBotArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
