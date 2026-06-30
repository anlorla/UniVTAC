"""Preprocess the raw Blender bowl OBJ into a sim-ready mesh (meters, Z-up, centered),
mirroring the CUP convention: origin at the body centre, opening toward local +Z.
  - Blender export is Y-up; rotate +90deg about X so vertical (old +Y, the open top) -> +Z.
  - Uniform-scale so the rim outer diameter ~= TARGET_DIAM (m).
  - Recentre so z in [-half, +half] and radial centre at (0,0).
Usage: python scripts/asset_tools/prep_bowl_obj.py <in.obj> <out.obj> [target_diam_m]
"""
import sys, numpy as np, trimesh

src, dst = sys.argv[1], sys.argv[2]
TARGET_DIAM = float(sys.argv[3]) if len(sys.argv) > 3 else 0.130   # rim outer diameter (m)

m = trimesh.load(src, process=False)
if isinstance(m, trimesh.Scene):
    m = m.to_geometry()
V = np.asarray(m.vertices, np.float64)

# raw OBJ: Y is up (open top at y_max), x/z are radial
# rotate +90deg about X: (x,y,z) -> (x, -z, y)  [proper rotation, preserves winding]
V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)

# uniform scale: rim outer radius (max radial extent) -> TARGET_DIAM/2
r = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
scale = (TARGET_DIAM / 2.0) / r.max()
V *= scale

# recentre: radial centre to origin, vertical centred (z in [-half, +half])
V[:, 0] -= 0.5 * (V[:, 0].min() + V[:, 0].max())
V[:, 1] -= 0.5 * (V[:, 1].min() + V[:, 1].max())
V[:, 2] -= 0.5 * (V[:, 2].min() + V[:, 2].max())

m.vertices = V
m.export(dst)

r = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
half = 0.5 * (V[:, 2].max() - V[:, 2].min())
rim = np.abs(V[:, 2] - V[:, 2].max()) < 0.05 * (2 * half)
rim_r = r[rim]
print(f"[prep_bowl] scale={scale:.6f}  outer_diam={2*r.max()*1000:.1f}mm  height={2*half*1000:.1f}mm")
print(f"[prep_bowl] rim wall ~{(rim_r.max()-rim_r.min())*1000:.1f}mm  half={half:.4f}  z in [{V[:,2].min():.4f},{V[:,2].max():.4f}]")
print(f"[prep_bowl] watertight={m.is_watertight}  verts={len(V)} faces={len(m.faces)}  -> {dst}")
