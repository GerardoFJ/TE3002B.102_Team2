"""Demo del almacen robotico colaborativo con navegacion por vision.

Ejecutar:
    python3 demo.py                       # animacion interactiva
    python3 demo.py --save mision.mp4     # guarda video (requiere ffmpeg)
    python3 demo.py --save mision.gif     # guarda gif  (requiere Pillow)
    python3 demo.py --no-show             # solo ejecuta la logica (sin ventana)
    python3 demo.py --stride 10           # animacion mas rapida
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mini_reto_s2.sim import main

if __name__ == "__main__":
    main()
