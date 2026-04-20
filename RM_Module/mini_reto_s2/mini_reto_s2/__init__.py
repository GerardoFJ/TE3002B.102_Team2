"""Mini Reto Semana 2 - Almacen robotico colaborativo.

Paquete con la logica pura (matplotlib + numpy) del reto:
    - robots_base : clases base PuzzleBot, HuskyA200, ANYmal
    - puzzlebot_arm : mini brazo 3 DoF montado sobre PuzzleBot
    - husky_pusher : planner local + controlador del Husky para empujar cajas
    - anymal_gait : generador de marcha trote con FK/IK por pata
    - coordinator : maquina de estados que orquesta las tres fases
    - sim : simulador 2D animado con matplotlib
"""
