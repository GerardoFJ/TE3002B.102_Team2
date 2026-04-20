"""Nodos ROS2 (rclpy) que envuelven la logica pura del mini reto.

Cada nodo es un wrapper delgado: instancia la clase correspondiente,
ejecuta su simulacion una sola vez al arranque para precomputar el log,
y luego republica el log paso a paso en un `create_timer` (replay).

Este patron mantiene la 'logica pura' del paquete (`mini_reto_s2.*`)
intacta y desacoplada de ROS2.

Modulos:
    - husky_pusher_node : reproduce la fase 1 (Husky empuja cajas).
    - anymal_gait_node  : reproduce la fase 2 (ANYmal trote a destino).
    - puzzlebot_arm_node: reproduce el pick&place de un PuzzleBot.
    - coordinator_node  : reproduce las 3 fases en secuencia y publica
                          /mission/phase como String.
"""
