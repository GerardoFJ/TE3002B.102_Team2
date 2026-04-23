"""Mini Reto Semana 2 - Almacen robotico colaborativo.

Paquete con la logica pura (matplotlib + numpy) del reto:
    - robots_base : clases base PuzzleBot, HuskyA200, ANYmal
    - husky_pusher : planner local + controlador del Husky para empujar cajas
    - anymal_gait : generador de marcha trote con FK/IK por pata
    - xarm : xArm 6 simplificado (cinematica del URDF oficial)
    - coordinator : maquina de estados que orquesta las 3 fases
    - sim : simulador 2D animado con matplotlib
"""
