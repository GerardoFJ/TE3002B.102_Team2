# Mini Reto Semana 2 — Almacén robótico colaborativo

TE3002B · Robots Móviles · Semana 2 · Parte II

Tres robots cooperan en un almacén:

1. **Husky A200** despeja un corredor empujando 3 cajas pesadas.
2. **ANYmal** camina en trote por el corredor cargando 3 PuzzleBots sobre su dorso.
3. **3 PuzzleBots** con mini brazos de 3 DoF apilan 3 cajas en el orden C–B–A.

## Estructura

```
mini_reto_s2/
├── package.xml                    # Metadata ROS2 (ament_python)
├── setup.py                       # Entry points de los 4 nodos
├── setup.cfg
├── resource/mini_reto_s2          # Marker ament
├── mini_reto_s2/                  # Paquete Python (lógica pura)
│   ├── robots_base.py             # PuzzleBot, HuskyA200, ANYmal, ANYmalLeg
│   ├── puzzlebot_arm.py           # Mini brazo 3 DoF (FK / IK / Jacobiano / τ=Jᵀf)
│   ├── husky_pusher.py            # Planner local + skid-steer + LiDAR 2D
│   ├── anymal_gait.py             # Generador de trote + navigator
│   ├── coordinator.py             # FSM de las 3 fases
│   ├── sim.py                     # Simulador 2D matplotlib (frame builder + render)
│   └── nodes/                     # Wrappers ROS2 (rclpy)
│       ├── husky_pusher_node.py
│       ├── anymal_gait_node.py
│       ├── puzzlebot_arm_node.py
│       └── coordinator_node.py
├── launch/
│   ├── mission.launch.py          # Demo: lanza solo coordinator_node
│   └── modules.launch.py          # Lanza los 4 wrappers individuales
├── tests/                         # 53 tests unitarios
│   ├── test_puzzlebot_arm.py
│   ├── test_anymal_gait.py
│   ├── test_husky_pusher.py
│   ├── test_coordinator.py
│   └── test_sim.py
├── requirements.txt
└── README.md
```

## Cómo correr los tests (sin ROS2)

```bash
cd robotics_ws/mini_reto_s2
python3 -m unittest discover -s tests -v
```

Los 53 tests cubren FK/IK/Jacobiano del brazo, periodicidad del trote,
LiDAR + push del Husky, las 3 fases del coordinator end-to-end y el
frame builder del simulador.

## Cómo correr la simulación 2D matplotlib

```bash
cd robotics_ws/mini_reto_s2
python3 -m mini_reto_s2.sim                  # ventana animada
python3 -m mini_reto_s2.sim --save mission.mp4 --fps 30
python3 -m mini_reto_s2.sim --no-show        # smoke test headless
```

## Cómo construir y correr en ROS2 Humble (Docker)

El paquete está estructurado como `ament_python`. Dentro del contenedor
con ROS2 Humble:

```bash
# Asumiendo que tu workspace de colcon es ~/ros2_ws y este repo está
# montado/copiado en src/mini_reto_s2:
cd ~/ros2_ws
colcon build --packages-select mini_reto_s2 --symlink-install
source install/setup.bash
```

### Demo principal (un solo nodo, todas las fases)

```bash
ros2 launch mini_reto_s2 mission.launch.py
# Argumentos opcionales:
ros2 launch mini_reto_s2 mission.launch.py husky_terrain:=grass dt:=0.05 loop:=true
```

Este lanza `coordinator_node`, que ejecuta `MissionCoordinator.run()` al
arrancar (precomputa los logs de las 3 fases) y luego republica todo el
estado en un timer. Tópicos publicados:

| Tópico                  | Tipo                          | Descripción                       |
|-------------------------|-------------------------------|-----------------------------------|
| `/mission/phase`        | `std_msgs/String`             | Fase actual de la FSM             |
| `/husky/odom`           | `nav_msgs/Odometry`           | Pose 2D del Husky                 |
| `/husky/cmd_vel`        | `geometry_msgs/Twist`         | Velocidad comandada al Husky      |
| `/husky/status`         | `std_msgs/String`             | Caja activa (B1/B2/B3)            |
| `/anymal/odom`          | `nav_msgs/Odometry`           | Pose 2D base del ANYmal           |
| `/anymal/joint_states`  | `sensor_msgs/JointState`      | 12 articulaciones del ANYmal      |
| `/anymal/det_J`         | `std_msgs/Float32MultiArray`  | det(J) por pata [LF, RF, LH, RH]  |
| `/pb_a/odom`            | `nav_msgs/Odometry`           | Pose 2D del PuzzleBot A           |
| `/pb_a/joint_states`    | `sensor_msgs/JointState`      | 3 articulaciones del brazo de A   |
| `/pb_a/status`          | `std_msgs/String`             | Sub-fase del PuzzleBot A          |
| `/pb_b/...`, `/pb_c/...`| (idem)                        | PuzzleBots B y C                  |
| `/boxes/big`            | `std_msgs/Float32MultiArray`  | [x,y]·3 cajas grandes             |
| `/boxes/small`          | `std_msgs/Float32MultiArray`  | [x,y,z]·3 cajas pequeñas          |

Inspección rápida:

```bash
ros2 topic echo /mission/phase
ros2 topic hz /husky/odom
rqt_graph
```

### Demo modular (4 nodos individuales)

```bash
ros2 launch mini_reto_s2 modules.launch.py
```

Lanza:
- `husky_pusher_node` → reproduce solo la fase 1
- `anymal_gait_node` → reproduce solo la fase 2
- 3× `puzzlebot_arm_node` → cada uno hace pick&place de su caja (A/B/C)

Útil para inspeccionar cada wrapper aislado en `ros2 node list`.

### Correr un solo nodo

```bash
ros2 run mini_reto_s2 husky_pusher_node
ros2 run mini_reto_s2 anymal_gait_node --ros-args -p target_x:=11.0 -p target_y:=3.6
ros2 run mini_reto_s2 puzzlebot_arm_node --ros-args -p name:=pb_c -p role_box:=C -p stack_layer:=0
ros2 run mini_reto_s2 coordinator_node
```

## Estado actual

| Módulo                   | Estado    |
|--------------------------|-----------|
| `robots_base.py`         | ✅ listo  |
| `puzzlebot_arm.py`       | ✅ listo (12 tests OK) |
| `husky_pusher.py`        | ✅ listo (9 tests OK) |
| `anymal_gait.py`         | ✅ listo (8 tests OK) |
| `coordinator.py`         | ✅ listo (14 tests OK) |
| `sim.py`                 | ✅ listo (10 tests OK) |
| Nodos ROS2               | ✅ listos (4 wrappers) |
| Launch files             | ✅ listos (mission + modules) |

Total: **53/53 tests pasando**.

## Notas de diseño

- **Lógica pura primero**: las clases de `mini_reto_s2/*.py` no dependen
  de ROS2 ni de matplotlib (excepto `sim.py`, con import perezoso).
  Esto permite testear cada módulo aislado, como sugiere el profesor.
- **Coordinator dumb**: la máquina de estados solo invoca métodos; toda
  la inteligencia vive en los módulos especializados.
- **Wrappers ROS2 = replay de logs**: cada nodo precomputa la simulación
  al arrancar y luego republica el log paso a paso en un timer. Esto
  mantiene el código de los nodos trivial (sin lazos de control en
  callbacks) y desacoplado de la física.
- **Singularidades del ANYmal**: el modelo simplificado de `ANYmalLeg`
  tiene `det(J) ∝ sin(q1)`, así que el trote arranca con `q1 = 0.25`
  para evitar la singularidad geométrica del modelo.
- **Compensación de slip del Husky**: `inverse_kinematics(v, ω,
  compensate_slip=True)` divide `v` por el factor del terreno antes de
  calcular las velocidades de rueda, así `v_real == v_cmd`.

## Tips del profe que estamos siguiendo

- Empezar por simulación, no por hardware. ✅
- Testear cada módulo aislado antes de integrar. ✅ (53 tests)
- Interpolación lineal cartesiana para motion planning (no splines). ✅
- Limitar workspace del ANYmal para evitar `q3 → 0` (singularidad). ✅
- Time-slotting (turnos) para coordinar los 3 PuzzleBots. ✅
