"""Prepare HOPPY STLs for MuJoCo with COMPONENT-AWARE decimation.

MuJoCo's STL decoder caps a mesh at 200k faces; SolidWorks exported Link2/Link3
well above. Naive whole-mesh quadric decimation shrinks and distorts the visible
structure (actuator box, PVC boom tube). But those meshes are bloated almost
entirely by over-tessellated small hardware: a 5 mm bolt carries ~10k faces, and
there are ~100 of them, while the structural parts total only ~110k faces.

So we SPLIT each heavy mesh into connected components and:
  * keep big structural parts (bbox diagonal > BIG_MM) at FULL resolution
    -> the box and tube are untouched (no shrink, no faceting), and
  * crush each small bolt/pin to a few hundred faces (a sub-mm change invisible
    at that scale),
then recombine. Result stays well under 200k with the visible geometry intact.

Coordinates preserved -> MJCF mesh placement matches the URDF link frames.
Link4 (115k) and the small links are already under the cap -> copied as-is.
"""
import os, shutil, trimesh, fast_simplification
import numpy as np

SRC = "/home/daniel-wlg/ros2/workspace/TE3002B.102_Team2/Humanoides_Module/Hoppy_FinalProyect/HOPPY-E0-final/meshes"
OUT = "/home/daniel-wlg/ros2/workspace/TE3002B.102_Team2/Humanoides_Module/Hoppy_FinalProyect/mujoco_sim/meshes"
os.makedirs(OUT, exist_ok=True)

SPLIT = {"Link2", "Link3"}     # heavy meshes that need component-aware treatment
COPY = {"base_link", "Link1", "Link4"}
BIG_MM = 25.0                  # bbox diagonal above which a part is kept full-res
BOLT_CAP = 300                 # max faces for a small hardware component


def process_split(name):
    m = trimesh.load(os.path.join(SRC, name + ".STL"), process=True)
    nf0 = len(m.faces); b0 = m.bounds
    comps = m.split(only_watertight=False)
    kept, n_big, n_small = [], 0, 0
    for c in comps:
        diag = float(np.linalg.norm(c.bounds[1] - c.bounds[0]))
        if diag > BIG_MM / 1000.0:
            kept.append(c); n_big += 1                      # full res
        elif len(c.faces) > BOLT_CAP:
            red = 1.0 - BOLT_CAP / len(c.faces)
            v, f = fast_simplification.simplify(c.vertices, c.faces, target_reduction=red)
            kept.append(trimesh.Trimesh(vertices=v, faces=f, process=False)); n_small += 1
        else:
            kept.append(c); n_small += 1
    out = trimesh.util.concatenate(kept)
    out.export(os.path.join(OUT, name + ".STL"))
    b1 = out.bounds
    dext = (np.abs(b1[0] - b0[0]) + np.abs(b1[1] - b0[1])) * 1000.0
    ok = "OK" if dext.max() < 1.0 else f"extent dmax={dext.max():.1f}mm"
    print(f"{name:10s} {nf0:>7d} -> {len(out.faces):>7d} faces  "
          f"(big full-res:{n_big}, small crushed:{n_small})  "
          f"x[{b1[0,0]:+.3f},{b1[1,0]:+.3f}]  {ok}")


for name in COPY:
    shutil.copyfile(os.path.join(SRC, name + ".STL"), os.path.join(OUT, name + ".STL"))
    nf = len(trimesh.load(os.path.join(SRC, name + ".STL"), process=False).faces)
    print(f"{name:10s} {nf:>7d} faces  copied (already < 200k)")

for name in SPLIT:
    process_split(name)

print("\nDone ->", OUT)
