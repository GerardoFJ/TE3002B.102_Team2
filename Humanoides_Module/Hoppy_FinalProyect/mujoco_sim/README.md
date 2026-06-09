# HOPPY en MuJoCo — Simulación y control de salto

Monópodo HOPPY sobre gantry (boom planarizante) simulado en MuJoCo 3.9.
Modelo derivado de `HOPPY-E0-final.urdf` (export SolidWorks) + control híbrido
FLIGHT/STANCE para salto. Cubre las 5 fases de la rúbrica.

## Arquitectura mecánica

```
base_link (torre, fija al mundo)
 └─ yaw   (joint1, PASIVO)   gantry: barrido del boom alrededor de la torre
     └─ pitch (joint2, PASIVO) gantry: altura de salto
         ├─ contrapeso (2.4 kg, extremo +x del boom)
         └─ hip (joint3, ACTIVO)
             └─ knee (joint4, ACTIVO, + resorte paralelo)
                 └─ pie (esfera de contacto puntual)
```
El boom solo tiene yaw+pitch → **no hay DOF de volcadura**, la estabilidad
lateral está garantizada por construcción.

**Pierna fiel al URDF → locomoción de avance.** La geometría de la pierna es la
**exportación fiel de SolidWorks/URDF** (frames y mallas de Link3/Link4 sin
tocar). Los ejes hip/knee quedan ~perpendiculares al pitch → la pierna oscila en
el plano **tangencial**-vertical, así que al empujar mueve el yaw pasivo y
**HOPPY salta hacia adelante dando vueltas alrededor de la torre** (su locomoción
real). El yaw se hizo **continuo** (sin límite de rango; el URDF traía ±180° que
lo frenaba a media vuelta) y `v_des` fija la velocidad de crucero (~1 vuelta cada
~9 s, Fase 4.4). Para saltar **en el sitio**, subir el `damping` del joint yaw a
~18 (freno) — está documentado en el XML.

## Estructura

| Archivo | Contenido |
|---|---|
| `scene.xml` | Mundo: piso, luces, opciones de solver (RK4, Newton, dt=1e-3, iters=50, tol=1e-8) |
| `hoppy.xml` | Robot: árbol cinemático, armature/damping/resorte, actuadores, sensores |
| `view.py` | Visor interactivo (`python view.py [--drop]`) |
| `controller/hoppy_brain.py` | `EncoderEstimator` + `FootContactSensor` + `HoppyController` (FSM + leyes de control) |
| `controller/run_hop.py` | Bucle 1 kHz, logging, video, stats |
| `controller/plots.py` | Figuras de análisis (Fase 5.2) |
| `controller/ablation.py` | Estudio comparativo de ablación (Fase 2.2): baseline vs sin armature/damping/resorte/saturación |
| `controller/dashboard.py` | Dashboard interactivo en tiempo real: render 3D + tarjetas KPI + medidores (torque/voltaje/corriente) + páginas por botones (Telemetría / Encoder 5.1 / Sensor / Fase·Energía) + controles (Pausa·Reset·Modo·**Salir**) + barra RT/FPS. **Click en cualquier gráfica para agrandarla a pantalla completa; ✕ para cerrar.** El botón **Salir** termina la sim y cierra el programa. |
| `tools/process_meshes.py` | Decimado reproducible de STLs (límite 200k caras de MuJoCo) |

## Uso

```bash
# (desde mujoco_sim/, con el venv del proyecto activo)
python view.py --drop                       # ver el modelo caer y asentarse

cd controller
python run_hop.py --duration 6 --plots      # simula, imprime stats, genera PNGs
python run_hop.py --video hop_demo.mp4       # graba el salto (offscreen EGL)
python run_hop.py --live                     # visor en vivo (necesita display)
python ablation.py --duration 6             # Fase 2.2: ablación comparativa -> hoppy_ablacion.png
python dashboard.py                          # dashboard interactivo en vivo (botones: vistas + Pausa/Reset/Modo)
python dashboard.py --record dash.mp4 --duration 10   # graba el dashboard a mp4 (headless)
python dashboard.py --record enc.mp4 --page encoder   # vista concreta (telem|encoder|sensor|energia)
# En vivo usa BLITTING (redibuja solo lo dinámico sobre fondo cacheado) -> ~3x FPS,
# corre a tiempo real (RT≈1.0). Ejes con límites fijos y X en tiempo relativo (ahora=0).
python dashboard.py --no-blit                # respaldo: dibujo completo (~13 FPS) si el blit falla

# overrides de tuning: --f_peak --t_push --kp_cart --kv_fwd --k_place --k_yaw --v_des
```

## Mapeo con la rúbrica

| Fase | Dónde |
|---|---|
| **1.1** Gantry 4 DoF + contrapeso | `hoppy.xml` (yaw/pitch pasivos, hip/knee activos, body `counterweight`) |
| **1.2** Armature N²·Iᵣ | `hoppy.xml` joints hip/knee `armature=...` |
| **1.3** Damping | `hoppy.xml` clases `gantry`/`leg` `damping` |
| **1.4** Resorte paralelo rodilla | `hoppy.xml` knee `stiffness`+`springref` |
| **2.1** Saturación de torque | modelo de actuador por voltaje (back-EMF + límites 12 V / 30 A) en `_motor_model` + `np.clip(±3.73)` + `forcerange` |
| **2.2** Validación del modelo mecánico | `ablation.py` → simulaciones comparativas (baseline vs sin armature/damping/resorte/saturación) + tabla de métricas + `hoppy_ablacion.png` |
| **3.1** Contacto duro | `solref`/`solimp`/`friction` en pie y piso; solver en `scene.xml` |
| **3.2** Touchdown/lift-off | `HoppyController.foot_normal_force` + transiciones FSM con histéresis |
| **4.1** Bucle 1 kHz | `run_hop.py` (`dt=0.001`) |
| **4.2** FSM FLIGHT/STANCE | `HoppyController.control` |
| **4.3** Cartesiano Jacobiano (vuelo) | `tau = -J^T(Kp(p-pd)) - Kd q̇` en marco boom |
| **4.4** GRF (apoyo) + velocidad | `tau = J^T F_GRF` (medio-seno) + término tangente/Raibert |
| **4.x** Transiciones suaves | blending lineal de las dos leyes durante `t_blend` (12 ms) |
| **5.1** Emulación de sensores | `EncoderEstimator` (encoders: cuantización + filtro, NO usa qvel) + `FootContactSensor` (sensor lineal del pie → switch binario) |
| **5.2** Gráficas y análisis | `plots.py` → `hoppy_signals/encoder/limitcycle/resultados/foot_sensor.png` |

## Parámetros del motor (goBILDA 5202, modelo HOPPY de referencia)

HOPPY usa el mismo motor en hip y rodilla pero con **distinta reducción**. Los
valores autoritativos vienen de `Simulator_MATLAB/fcns/get_params.m` del proyecto
HOPPY original (que la rúbrica 1.2 cita: NH=26.9, NK=28.8):

| Spec (get_params.m) | Valor | Parámetro del modelo |
|---|---|---|
| Reducción hip / knee | NH = 26.9 / NK = **28.8** | `N_gear = [26.9, 28.8]` |
| Inercia rotor | Iᵣ = 7e-6 kg·m² | `armature = N²·Iᵣ` → 0.005065 / 0.005806 |
| Resistencia bobina | R = 1.3 Ω | modelo de voltaje |
| Constante de par | kₜ = 0.0135 N·m/A | `tau = kₜ·N·i` |
| Constante de fcem | kᵥ = 0.0186 V·s/rad | back-EMF `kᵥ·N·ω` |
| Límites | V ≤ 12 V, I ≤ 30 A | saturación → par ≈ 3.4 N·m |
| back-EMF damping | kᵥkₜN²/R = 0.140 / 0.160 | **en el modelo de voltaje** (no en `damping`) |
| Encoder 751.8 PPR ×4 | 3012.8 cpr | resolución = 2.086e-3 rad |

El `damping` del joint (0.02) representa solo la **fricción viscosa del reductor**
y `frictionloss=0.10` la **fricción de Coulomb**; el back-EMF se modela explícito
en `_motor_model` (tau→V→i→tau) para no contarlo doble. El par de saturación
físico lo fijan los límites 12 V / 30 A; un `clip` adicional a ±3.73 N·m cierra la
Fase 2.1. (La datasheet goBILDA en `Assets/` da ~3.73 N·m de par de calado,
consistente con el modelo.)

Contrapeso = 2.4 kg (~70% del balance estático 3.40 kg) → ajústalo a tu build.

## Sensores de HOPPY (guía técnica §VIII)

Se emula el set real de sensores del kit:

| Sensor real | En el modelo | Dato |
|---|---|---|
| 2 encoders incrementales (cadera, rodilla) | `EncoderEstimator` (cuantiza qpos + derivada filtrada) | posición y velocidad articular estimadas |
| Sensado de corriente del motor | modelo de voltaje (`_motor_model`) | corriente `i` por bobina (límite 30 A) |
| **Sensor lineal del pie** (`Linear_Disp_Sensor_404R1KL1.0`, potenciómetro de 1 in en la espinilla) | `FootContactSensor` | desplazamiento `δ` (mm, cuantizado por ADC) + **switch binario de contacto** (histéresis + antirrebote) que dispara touchdown/lift-off |

El sensor del pie se ve **dibujado dentro de la ventana de MuJoCo** con
`python run_hop.py --live`: una **esfera en el pie** verde (contacto cerrado) /
roja (abierto), un **"termómetro" del desplazamiento δ** sobre la torre, y texto
flotante con estado FSM, F_N y la vuelta (vía `viewer.user_scn`). También en la
gráfica `hoppy_foot_sensor.png` (δ → fuerza → switch → FSM).

## Notas / pendientes

- **Salto en sitio**: con la pierna re-planarizada (ejes hip/knee ∥ pitch) el
  yaw ya **no deriva** (≈ −4° de asentamiento inicial, sin acumular). El
  controlador conserva el término horizontal/Raibert para la Fase 4.4, pero al
  ser la pierna planar su autoridad tangencial es baja: HOPPY salta en el sitio,
  igual que el modelo de referencia. (La geometría original, con la pata
  oscilando tangencialmente, sí "viajaba" alrededor del boom — eso era el giro.)
- **Pico de GRF en touchdown** (~600 N): impacto de contacto rígido con el
  empuje alto (f_peak=75). Suavizable bajando `f_peak` o afinando `solref` del
  pie (Fase 3) si se requiere un impacto menor.
- **Avance (vueltas al boom)**: yaw continuo + `v_des=2.0` → ~1.3 vueltas en
  12 s (≈1 vuelta/9 s, ~0.45 m/s), saltando a la vez. La velocidad de avance la
  limita la inercia del boom+contrapeso (la pierna no puede empujarlo más rápido
  por salto); `v_des` mayor satura en ese máximo. Para frenar/saltar en sitio,
  subir el damping del joint yaw.
- Postura de apoyo (hip≈0.70, knee≈0.35): |Jz|≈0.27 para buena transmisión
  vertical (vs 0.08 de la pata recta, casi singular). Resorte de rodilla
  stiffness=30, springref=+0.30.
- Tuning del salto: `f_peak=120 N`, `t_push=0.24 s` → ~12 saltos en 12 s, apex
  de pie ≈0.18 m, |τ|≤3.73 N·m, V tope 12 V, I<30 A. El video usa cámara fija
  centrada en la torre para ver el avance; `--live` también.
