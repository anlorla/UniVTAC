"""[PATCH-D] Generate the dense gelpad robot USD for option B (denser gelpad FEM mesh).

Re-tetrahedralizes the gsmini gelpad at a finer edge_length_r (fTetWild) and writes the tet into
a copy of the gripper USD (both gelpad prims). Default elr=0.05 -> ~615 nodal / 2276 tets, vs the
shipped 169 / 424. This raises the real spatial resolution of the tactile force field (option B),
at the cost of a slower UIPC contact solve.

Needs trimesh + wildmeshing (the UniVTAC conda env) and pxr (USD). pxr is not on the default path;
provide it via the Isaac Sim usd libs, e.g.:
  # binary Isaac Sim (e.g. node .56):
  EXT=$(ls -d ~/yifan/isaacsim/extscache/omni.usd.libs-*/ | head -1)
  PYTHONPATH=$EXT LD_LIBRARY_PATH=$EXT/bin:$EXT/pxr/.libs \
    python scripts/asset_tools/make_dense_gelpad.py [edge_length_r]
  # pip Isaac Sim (e.g. node .150): EXT=.../site-packages/isaacsim/extscache/omni.usd.libs-*

Run from the repo root. Output: .../GelSight_Mini/Gripper/uipc_gelpads_dense_wrist.usd
(enabled at runtime by setting `dense_gelpad: true` in the task config).
"""
import sys
import shutil
from pathlib import Path

import numpy as np
import trimesh
import wildmeshing as wm
from pxr import Usd, Sdf, Vt

ROOT = Path(__file__).resolve().parents[2]
GRIP = ROOT / ("third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka"
               "/GelSight_Mini/Gripper")
STL = GRIP / "textures/gelpad.stl"
SRC = GRIP / "uipc_gelpads_high_res_wrist.usd"
DST = GRIP / "uipc_gelpads_dense_wrist.usd"
PRIMS = ["/panda/gelpad_left/mesh", "/panda/gelpad_right/mesh"]

elr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05  # smaller = denser (0.08 ~= shipped mesh)

# --- tetrahedralize the gelpad surface ---
m = trimesh.load(str(STL), process=False)
m = m.to_geometry() if isinstance(m, trimesh.Scene) else m
pts = np.asarray(m.vertices, float)
tri = np.asarray(m.faces, np.uint32).reshape(-1, 3)
t = wm.Tetrahedralizer(stop_quality=10, max_its=40, epsilon=0.001,
                       edge_length_r=elr, skip_simplify=False, coarsen=True)
t.set_log_level(6)
t.set_mesh(pts, tri)
t.tetrahedralize()
tp, ti, _ = t.get_tet_mesh(correct_surface_orientation=True, manifold_surface=True, use_input_for_wn=False)
sp, si = t.get_tracked_surfaces()
tp = np.asarray(tp, np.float32)
ti = np.asarray(ti).reshape(-1).astype(np.uint32)
sp = np.asarray(sp[0], np.float32)
si = np.asarray(si).reshape(-1).astype(np.uint32)
print(f"baked elr={elr}: nodal={len(tp)} tets={len(ti)//4} surface_verts={len(sp)}")

# --- write the tet into a copy of the gripper USD (both gelpad prims) ---
shutil.copy(str(SRC), str(DST))
s = Usd.Stage.Open(str(DST))
old = np.array(s.GetPrimAtPath(PRIMS[0]).GetAttribute("tet_points").Get())
assert np.allclose(old.min(0), tp.min(0), atol=2e-3) and np.allclose(old.max(0), tp.max(0), atol=2e-3), (
    f"frame/scale mismatch: old [{old.min(0)}..{old.max(0)}] vs new [{tp.min(0)}..{tp.max(0)}]")
for path in PRIMS:
    mp = s.GetPrimAtPath(path)
    mp.CreateAttribute("tet_points", Sdf.ValueTypeNames.Float3Array).Set(Vt.Vec3fArray.FromNumpy(tp.reshape(-1, 3)))
    mp.CreateAttribute("tet_indices", Sdf.ValueTypeNames.UIntArray).Set(Vt.UIntArray.FromNumpy(ti))
    mp.CreateAttribute("tet_surf_points", Sdf.ValueTypeNames.Float3Array).Set(Vt.Vec3fArray.FromNumpy(sp.reshape(-1, 3)))
    mp.CreateAttribute("tet_surf_indices", Sdf.ValueTypeNames.UIntArray).Set(Vt.UIntArray.FromNumpy(si))
s.Save()
print(f"wrote {DST}")
