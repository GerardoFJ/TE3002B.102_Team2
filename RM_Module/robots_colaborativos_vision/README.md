# Almacen Robotico Colaborativo con Navegacion por Vision

Simulacion 2D de tres robots colaborando en un almacen industrial.
La logica de navegacion esta guiada exclusivamente por una camara cenital sintetica; no se utilizan waypoints precomputados ni sensores LiDAR.

## Arquitectura del sistema

El sistema esta compuesto por tres robots y un brazo manipulador:

* **Husky A200** (skid steer de 4 ruedas): despeja el corredor empujando cajas fuera de la trayectoria del ANYmal.
* **ANYmal** (cuadrupedo de 12 DoF): transporta tres PuzzleBots sobre su dorso desde el origen hasta el destino.
* **xArm 6** (brazo de 6 DoF): recoge los PuzzleBots del dorso del ANYmal y los deposita en la mesa.
* **PuzzleBot** (robot diferencial de 2 ruedas): pasajero transportado por el ANYmal.

## Pipeline de vision

La camara cenital renderiza la escena completa a una imagen RGB y aplica procesamiento de imagen para:

1. Detectar la cuadricula de navegacion (lineas grises de 1 m x 1 m).
2. Localizar cada robot por su color caracteristico (blob detection).
3. Detectar obstaculos (cajas naranjas) y mapearlos a celdas de la cuadricula.
4. Planificar un camino mediante A* de 4 vecinos sobre las celdas detectadas.
5. Decidir, en cada celda, la siguiente direccion de movimiento sin lista de waypoints.

## Fases de la mision

**Fase 1 (HUSKY CLEAR)**
La camara calcula la trayectoria que necesita el ANYmal. Los obstaculos se generan dinamicamente sobre esa trayectoria para que el Husky tenga que despejarla. El Husky navega celda por celda guiado por la camara y empuja cada caja fuera del corredor. Al terminar regresa a su posicion de origen.

**Fase 2 (ANYMAL TRANSPORT)**
La camara valida que el corredor esta libre y planifica la ruta del ANYmal. El ANYmal trota al destino (12 m, 3 m) con los tres PuzzleBots en el dorso.

**Fase 2.5 (XARM TRANSFER)**
El xArm 6 recoge cada PuzzleBot del dorso del ANYmal con una trayectoria cartesiana approach > pick > lift > place y los deposita en la mesa.

## Estructura del proyecto

```
robots_colaborativos_vision/
    mini_reto_s2/
        robots_base.py                   Cinematica de PuzzleBot, Husky A200 y ANYmal
        husky_pusher.py                  Planner y controlador de empuje del Husky
        anymal_gait.py                   Generador de marcha trote con FK/IK por pata
        xarm.py                          xArm 6 simplificado (4 DoF efectivos)
        navigation_grid.py               Cuadricula 1 m x 1 m con A* de 4 vecinos
        aerial_camera.py                 Camara cenital: renderizado y vision computacional
        line_follower.py                 Seguidor de linea por camara individual del robot
        coordinator.py                   Maquina de estados que orquesta las 3 fases
        sim.py                           Visualizador 2D animado con matplotlib
    avance_robots_colaborativos.ipynb    Notebook con analisis y demostracion por fases
    demo.py                              Punto de entrada principal
    requirements.txt                     Dependencias Python
```

## Requisitos

```
numpy>=1.24
matplotlib>=3.7
opencv-python>=4.8
```

Instalar con:

```
pip install -r requirements.txt
```

## Como ejecutar

Animacion interactiva:

```
python3 demo.py
```

Guardar como video (requiere ffmpeg):

```
python3 demo.py --save mision.mp4
```

Guardar como gif animado (requiere Pillow):

```
python3 demo.py --save mision.gif
```

Solo ejecutar la logica sin ventana grafica:

```
python3 demo.py --no-show
```

Controlar la velocidad de la animacion con `--stride` (mayor = mas rapido):

```
python3 demo.py --stride 10
```
