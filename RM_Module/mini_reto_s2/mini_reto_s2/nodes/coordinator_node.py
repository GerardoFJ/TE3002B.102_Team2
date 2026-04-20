"""Wrapper rclpy del MissionCoordinator (las 3 fases del mini reto).

Ejecuta el coordinador completo al arranque y republica el merge de los
3 logs paso a paso. Este nodo es el "demo principal": con un solo
``ros2 run mini_reto_s2 coordinator_node`` se ven las 3 fases en el ROS
graph (no requiere lanzar los demas nodos).

Topicos publicados
------------------
/mission/phase    std_msgs/String  Fase actual: HUSKY_CLEAR/ANYMAL_TRANSPORT/
                                   PUZZLEBOT_STACK/DONE
/husky/odom       nav_msgs/Odometry             Pose del Husky (fase 1)
/husky/cmd_vel    geometry_msgs/Twist           cmd del Husky
/husky/status     std_msgs/String               Caja activa
/anymal/odom      nav_msgs/Odometry             Pose del ANYmal (fase 2)
/anymal/joint_states sensor_msgs/JointState     12 articulaciones
/anymal/det_J     std_msgs/Float32MultiArray    det por pata
/<pb>/odom        nav_msgs/Odometry             Pose de cada PuzzleBot (fase 3)
/<pb>/joint_states sensor_msgs/JointState       3 articulaciones del brazo
/boxes/big        std_msgs/Float32MultiArray    [x,y]*3 cajas grandes
/boxes/small      std_msgs/Float32MultiArray    [x,y,z]*3 cajas pequenas

Parametros
----------
husky_terrain  str   Terreno (default 'grass')
dt             float Periodo del timer (default 0.05)
publish_decim  int   Decimacion solo para fase 2 (default 5)
loop           bool  Reinicia el replay al terminar (default False)
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from mini_reto_s2.coordinator import MissionCoordinator


ANYMAL_JOINT_NAMES = [
    'LF_HAA', 'LF_HFE', 'LF_KFE',
    'RF_HAA', 'RF_HFE', 'RF_KFE',
    'LH_HAA', 'LH_HFE', 'LH_KFE',
    'RH_HAA', 'RH_HFE', 'RH_KFE',
]
LEG_ORDER = ('LF', 'RF', 'LH', 'RH')
ARM_JOINT_NAMES = ['arm_q1', 'arm_q2', 'arm_q3']
PB_TOPIC_NAMES = {'C': 'pb_c', 'B': 'pb_b', 'A': 'pb_a'}


def _yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


# ===========================================================================
#  Builder de frames unificados
# ===========================================================================
def build_mission_frames(coord, decim_phase2=5):
    """Convierte el log de un MissionCoordinator en una lista plana
    de frames listos para publicar.

    Cada frame es un dict con la fase y solo los campos relevantes
    para esa fase. El nodo decide que topic publicar segun frame['phase'].
    """
    frames = []

    # ----- Fase 1 -----
    p1 = coord.log['phase1']
    n1 = len(p1['t'])
    for i in range(n1):
        frames.append({
            'phase': 'HUSKY_CLEAR',
            'husky_pose': (p1['x'][i], p1['y'][i], p1['theta'][i]),
            'husky_v': p1['v_real'][i],
            'husky_w': p1['omega_real'][i],
            'husky_status': p1['phase'][i] if i < len(p1['phase']) else '',
            'big_boxes': [(p1['boxes'][k][i][0], p1['boxes'][k][i][1])
                          for k in ('B1', 'B2', 'B3')],
        })
    last_husky = (p1['x'][-1], p1['y'][-1], p1['theta'][-1])
    last_big = [(p1['boxes'][k][-1][0], p1['boxes'][k][-1][1])
                for k in ('B1', 'B2', 'B3')]

    # ----- Fase 2 -----
    p2 = coord.log['phase2']
    n2 = len(p2['t'])
    for i in range(0, n2, decim_phase2):
        frames.append({
            'phase': 'ANYMAL_TRANSPORT',
            'husky_pose': last_husky,
            'big_boxes': last_big,
            'anymal_pose': (p2['base_x'][i], p2['base_y'][i], p2['base_yaw'][i]),
            'anymal_v': p2['v_cmd'][i],
            'anymal_w': p2['omega_cmd'][i],
            'anymal_q': list(p2['q'][i]),
            'anymal_det': [p2['det_J'][leg][i] for leg in LEG_ORDER],
        })
    last_anymal = (p2['base_x'][-1], p2['base_y'][-1], p2['base_yaw'][-1])

    # ----- Fase 3 -----
    p3 = coord.log['phase3']
    wz = coord.work_zone
    pb_states = {
        'C': (11.20, 3.30, 0.0),
        'B': (11.20, 3.60, 0.0),
        'A': (11.20, 3.90, 0.0),
    }
    pb_arms = {'C': [0.0, 0.5, -1.0],
               'B': [0.0, 0.5, -1.0],
               'A': [0.0, 0.5, -1.0]}
    small_boxes = {  # name -> (x, y, z) en mundo
        n: (float(b.xy[0]), float(b.xy[1]), float(b.z))
        for n, b in wz.boxes.items()
    }

    for unit_log, role in zip(p3['units'], p3['order']):
        # drive_pick: solo el PB activo se mueve
        dp = unit_log['drive_pick']
        for i in range(len(dp['t'])):
            pb_states[role] = (dp['x'][i], dp['y'][i], dp['theta'][i])
            frames.append({
                'phase': 'PUZZLEBOT_STACK',
                'husky_pose': last_husky,
                'big_boxes': last_big,
                'anymal_pose': last_anymal,
                'pb_states': dict(pb_states),
                'pb_arms': dict(pb_arms),
                'pb_active': role,
                'small_boxes': dict(small_boxes),
                'pb_status': 'driving_to_pick',
            })

        # grasp: brazo se mueve, base estatica
        gr = unit_log['grasp']
        for q in gr['q_path']:
            pb_arms[role] = list(q)
            frames.append({
                'phase': 'PUZZLEBOT_STACK',
                'husky_pose': last_husky,
                'big_boxes': last_big,
                'anymal_pose': last_anymal,
                'pb_states': dict(pb_states),
                'pb_arms': dict(pb_arms),
                'pb_active': role,
                'small_boxes': dict(small_boxes),
                'pb_status': 'grasping',
            })

        # drive_place
        dp2 = unit_log['drive_place']
        for i in range(len(dp2['t'])):
            pb_states[role] = (dp2['x'][i], dp2['y'][i], dp2['theta'][i])
            # La caja viaja con el bot
            small_boxes[role] = (dp2['x'][i] + 0.10, dp2['y'][i],
                                 small_boxes[role][2])
            frames.append({
                'phase': 'PUZZLEBOT_STACK',
                'husky_pose': last_husky,
                'big_boxes': last_big,
                'anymal_pose': last_anymal,
                'pb_states': dict(pb_states),
                'pb_arms': dict(pb_arms),
                'pb_active': role,
                'small_boxes': dict(small_boxes),
                'pb_status': 'driving_to_place',
            })

        # place
        pl = unit_log['place']
        for q in pl['q_path']:
            pb_arms[role] = list(q)
            frames.append({
                'phase': 'PUZZLEBOT_STACK',
                'husky_pose': last_husky,
                'big_boxes': last_big,
                'anymal_pose': last_anymal,
                'pb_states': dict(pb_states),
                'pb_arms': dict(pb_arms),
                'pb_active': role,
                'small_boxes': dict(small_boxes),
                'pb_status': 'placing',
            })

        # caja queda en la pila
        info = p3['final_box_positions'][role]
        small_boxes[role] = (info['xy'][0], info['xy'][1], info['z'])

    # ----- DONE -----
    frames.append({
        'phase': 'DONE',
        'husky_pose': last_husky,
        'big_boxes': last_big,
        'anymal_pose': last_anymal,
        'pb_states': dict(pb_states),
        'pb_arms': dict(pb_arms),
        'small_boxes': dict(small_boxes),
        'pb_status': 'done',
    })
    return frames


# ===========================================================================
#  Nodo
# ===========================================================================
class CoordinatorNode(Node):

    def __init__(self):
        super().__init__('coordinator_node')

        self.declare_parameter('husky_terrain', 'grass')
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('publish_decim', 5)
        self.declare_parameter('loop', False)

        terrain = self.get_parameter('husky_terrain').value
        self.dt = float(self.get_parameter('dt').value)
        self.decim2 = int(self.get_parameter('publish_decim').value)
        self.loop = bool(self.get_parameter('loop').value)

        self.coord = MissionCoordinator(husky_terrain=terrain)
        self.get_logger().info("Ejecutando MissionCoordinator (las 3 fases)...")
        self.coord.run()
        self.get_logger().info(
            f"  fase final={self.coord.phase}, success={self.coord.log['success']}")

        self.frames = build_mission_frames(self.coord, decim_phase2=self.decim2)
        self.get_logger().info(
            f"  frames generados: {len(self.frames)}")

        # Publishers
        self.phase_pub = self.create_publisher(String, 'mission/phase', 10)
        self.husky_odom = self.create_publisher(Odometry, 'husky/odom', 10)
        self.husky_cmd = self.create_publisher(Twist, 'husky/cmd_vel', 10)
        self.husky_status = self.create_publisher(String, 'husky/status', 10)
        self.anymal_odom = self.create_publisher(Odometry, 'anymal/odom', 10)
        self.anymal_joint = self.create_publisher(
            JointState, 'anymal/joint_states', 10)
        self.anymal_det = self.create_publisher(
            Float32MultiArray, 'anymal/det_J', 10)
        self.pb_odom = {
            r: self.create_publisher(Odometry, f'{PB_TOPIC_NAMES[r]}/odom', 10)
            for r in ('A', 'B', 'C')
        }
        self.pb_joint = {
            r: self.create_publisher(
                JointState, f'{PB_TOPIC_NAMES[r]}/joint_states', 10)
            for r in ('A', 'B', 'C')
        }
        self.pb_status = {
            r: self.create_publisher(
                String, f'{PB_TOPIC_NAMES[r]}/status', 10)
            for r in ('A', 'B', 'C')
        }
        self.big_boxes_pub = self.create_publisher(
            Float32MultiArray, 'boxes/big', 10)
        self.small_boxes_pub = self.create_publisher(
            Float32MultiArray, 'boxes/small', 10)

        self.idx = 0
        self.timer = self.create_timer(self.dt, self._tick)

    # ------------------------------------------------------------------
    def _publish_husky(self, frame):
        x, y, th = frame['husky_pose']
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'husky/base_link'
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        qx, qy, qz, qw = _yaw_to_quat(float(th))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(frame.get('husky_v', 0.0))
        odom.twist.twist.angular.z = float(frame.get('husky_w', 0.0))
        self.husky_odom.publish(odom)
        if 'husky_v' in frame:
            t = Twist()
            t.linear.x = float(frame['husky_v'])
            t.angular.z = float(frame['husky_w'])
            self.husky_cmd.publish(t)
        if frame.get('husky_status'):
            self.husky_status.publish(String(data=frame['husky_status']))

    def _publish_anymal(self, frame):
        if 'anymal_pose' not in frame:
            return
        x, y, th = frame['anymal_pose']
        now = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'anymal/base'
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        qx, qy, qz, qw = _yaw_to_quat(float(th))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(frame.get('anymal_v', 0.0))
        odom.twist.twist.angular.z = float(frame.get('anymal_w', 0.0))
        self.anymal_odom.publish(odom)

        if 'anymal_q' in frame:
            js = JointState()
            js.header.stamp = now
            js.name = ANYMAL_JOINT_NAMES
            js.position = [float(v) for v in frame['anymal_q']]
            self.anymal_joint.publish(js)
        if 'anymal_det' in frame:
            d = Float32MultiArray()
            d.data = [float(v) for v in frame['anymal_det']]
            self.anymal_det.publish(d)

    def _publish_pbs(self, frame):
        if 'pb_states' not in frame:
            return
        now = self.get_clock().now().to_msg()
        for role, pose in frame['pb_states'].items():
            x, y, th = pose
            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = 'map'
            odom.child_frame_id = f'{PB_TOPIC_NAMES[role]}/base'
            odom.pose.pose.position.x = float(x)
            odom.pose.pose.position.y = float(y)
            qx, qy, qz, qw = _yaw_to_quat(float(th))
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw
            self.pb_odom[role].publish(odom)

            q_arm = frame['pb_arms'].get(role, [0.0, 0.5, -1.0])
            js = JointState()
            js.header.stamp = now
            js.name = ARM_JOINT_NAMES
            js.position = [float(v) for v in q_arm]
            self.pb_joint[role].publish(js)

            if frame.get('pb_active') == role:
                self.pb_status[role].publish(
                    String(data=frame.get('pb_status', '')))

    def _publish_boxes(self, frame):
        if 'big_boxes' in frame:
            m = Float32MultiArray()
            flat = []
            for x, y in frame['big_boxes']:
                flat.extend([float(x), float(y)])
            m.data = flat
            self.big_boxes_pub.publish(m)
        if 'small_boxes' in frame:
            m = Float32MultiArray()
            flat = []
            for name in ('A', 'B', 'C'):
                x, y, z = frame['small_boxes'][name]
                flat.extend([float(x), float(y), float(z)])
            m.data = flat
            self.small_boxes_pub.publish(m)

    # ------------------------------------------------------------------
    def _tick(self):
        if self.idx >= len(self.frames):
            if self.loop:
                self.idx = 0
            else:
                return
        frame = self.frames[self.idx]
        self.phase_pub.publish(String(data=frame['phase']))
        self._publish_husky(frame)
        self._publish_anymal(frame)
        self._publish_pbs(frame)
        self._publish_boxes(frame)
        self.idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = CoordinatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
