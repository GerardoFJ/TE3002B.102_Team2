# TE3002B.102 — Team 2

Coursework and project repository for **TE3002B.102 (Robotics)**, Team 2.
Each top-level directory is a self-contained module with its own
README and (when applicable) its own ROS workspace, Python package,
or Jupyter notebooks.

## Modules

```
.
├── Control_Module/         classical / fuzzy / fault-tolerant control
│   ├── InClass1/           RDLQ vs PSO on a double pendulum
│   ├── InClass2/           quadruped optimal control (upstream submodule)
│   ├── InClass3/           quadruped optimal control (forked submodule)
│   ├── InClass4/           AUV fault-tolerant control (forked submodule)
│   ├── Visual_Servoing/    home2 manipulation stack (xArm6 + ZED2)
│   └── Visual_Servoing_Project/  visual-servoing project on the xArm6
├── Vision_Module/          OpenCV / image-processing exercises
├── ML_Module/              ML reto + supporting documents
├── RM_Module/              ROS 2 / mini-reto integration
└── dataset_generation/     synthetic YOLO dataset for traffic signs
```

---

## Control_Module

Each `InClassN` folder is one in-class activity. The two `Visual_Servoing*`
folders host the visual-servoing exercise on the lab arm.

### `InClass1/` — RDLQ vs PSO on a double pendulum
Single-file Python comparison of a Recurrent Deep LQ controller against
Particle Swarm Optimization tuning for a double-pendulum stabilization
task. Entry point: `rdlq_vs_pso_double_pendulum.py`. Assets and cached
plots live in `Assets/`.

### `InClass2/` — Quadruped optimal control (upstream)
Submodule pointing at the original `nezih-niegu/quadruped-optimal-control`
repository. Read-only reference for the in-class exercise.

### `InClass3/` — Quadruped optimal control (fork)
Submodule pointing at the Team 2 fork of the same project, with the
in-class changes applied. See the submodule's own README for the
controller design.

### `InClass4/` — AUV fault-tolerant control (fork)
Submodule pointing at the Team 2 fork of `auv_ftc_ws`, an
implementation of the Zhang et al. (Sensors 24, 3029) T-S fuzzy +
weighted pseudo-inverse + active-set QP fault-tolerant controller
for an autonomous underwater vehicle in Gazebo Classic. The fork's
README has a detailed change log (added SMC robustifier, fixed
controller / B matrix alignment, NaN guards, auto-recovery, etc.),
a long-test results table, and a known-issues section pointing at
follow-on work.

Pinned commit: see `git submodule status Control_Module/InClass4`.

### `Visual_Servoing/`
Stripped-down checkout of RoBorregos `home2` manipulation stack
(ROS 2 Humble + MoveIt + Cyclone DDS), the platform we ran the
visual-servoing exercise on. The Jetson Orin running this is at
`orin@192.168.31.10`.

### `Visual_Servoing_Project/`
Project deliverable for the visual-servoing exercise: end-effector
pose servoing on an xArm6 from a ZED2 mounted on a custom gripper,
using an AprilTag fiducial as the target. See its own `README.md`
for the bring-up sequence and `report/` for the writeup.

---

## Vision_Module

Stand-alone OpenCV / image-processing exercises (course activities 2.x).
`act_2_04/actividad_2_04.py` is the script for activity 2.04; root-level
`opencv_test.py` is a sanity check that OpenCV + the test camera
pipeline work. `requirements.txt` pins the Python dependencies.

---

## ML_Module

ML reto delivery.

- `ML_Reto/` — `arrow_classifier.py` trains a small classifier on
  arrow images from `Data_Set/` and dumps confusion matrices /
  curves into `Results/`.
- `Documents/` — supporting writeups and references.

---

## RM_Module

ROS 2 integration mini-reto for semester 2 (`mini_reto_s2/`), packaged
as a colcon-buildable Python package with launch files, tests and
resources. `notebooks/` contains the supporting Jupyter analyses.

---

## dataset_generation

End-to-end synthetic YOLO dataset pipeline for traffic-sign detection:
composites segmented sign crops onto random backgrounds, trains
YOLOv8, and runs inference for the ROS 2 stack. See
`dataset_generation/README.md` for the class table, the augmentation
pipeline and the trained-model paths. Output videos and the trained
weights live under `runs/`.

---

## Working with the submodules

All three submodules live under `Control_Module/`. After a fresh
clone:

```bash
git clone --recurse-submodules https://github.com/GerardoFJ/TE3002B.102_Team2.git
```

Or for an existing clone:

```bash
git submodule update --init --recursive
```

To pull the latest commits in every submodule:

```bash
git submodule update --remote --merge
```
