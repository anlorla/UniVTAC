"""Generate a simple spur gear mesh with a center bore.

Usage:
  python scripts/asset_tools/build_gear.py [out.obj]
"""
import os
import sys
import numpy as np
import trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gear_holder_assets/GEAR.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

# mm
TEETH = 12
ROOT_R = 22.5
TIP_R = 27.0
BORE_R = 9.0
THICKNESS = 16.0
SECTIONS = TEETH * 2

theta = np.linspace(0.0, 2.0 * np.pi, SECTIONS, endpoint=False)
outer_r = np.where(np.arange(SECTIONS) % 2 == 0, TIP_R, ROOT_R)
inner_r = np.full(SECTIONS, BORE_R)
z_top = THICKNESS * 0.5
z_bot = -THICKNESS * 0.5

outer_top = np.column_stack([outer_r * np.cos(theta), outer_r * np.sin(theta), np.full(SECTIONS, z_top)])
outer_bot = np.column_stack([outer_r * np.cos(theta), outer_r * np.sin(theta), np.full(SECTIONS, z_bot)])
inner_top = np.column_stack([inner_r * np.cos(theta), inner_r * np.sin(theta), np.full(SECTIONS, z_top)])
inner_bot = np.column_stack([inner_r * np.cos(theta), inner_r * np.sin(theta), np.full(SECTIONS, z_bot)])

vertices = np.vstack([outer_top, outer_bot, inner_top, inner_bot])
ot = 0
ob = SECTIONS
it = SECTIONS * 2
ib = SECTIONS * 3

faces = []
for i in range(SECTIONS):
    j = (i + 1) % SECTIONS
    # top annulus
    faces.append([ot + i, ot + j, it + j])
    faces.append([ot + i, it + j, it + i])
    # bottom annulus
    faces.append([ob + j, ob + i, ib + i])
    faces.append([ob + j, ib + i, ib + j])
    # outer wall
    faces.append([ot + j, ot + i, ob + i])
    faces.append([ot + j, ob + i, ob + j])
    # inner bore wall
    faces.append([it + i, it + j, ib + j])
    faces.append([it + i, ib + j, ib + i])

gear = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
gear.merge_vertices()
gear.update_faces(gear.unique_faces())
gear.update_faces(gear.nondegenerate_faces())
gear.apply_scale(0.001)
gear.export(out)

print(
    f"saved {out} teeth={TEETH} outer_d(mm)={TIP_R * 2:.1f} "
    f"bore_d(mm)={BORE_R * 2:.1f} thickness(mm)={THICKNESS:.1f} "
    f"watertight={gear.is_watertight}"
)
