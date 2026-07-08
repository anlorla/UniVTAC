"""Generate a steeper-sided bowl OBJ for grasping/unstacking.

The old bowl had a very shallow flare near the rim, which made the local contact
surface too horizontal for stable gripper closure. This profile keeps the same
overall envelope (approx. Ø130 mm, H39.4 mm) but makes the upper wall more
vertical and the rim region easier to pinch.

Usage:
  python scripts/asset_tools/build_bowl.py [out.obj]
"""
import os
import sys
import numpy as np
import trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bowl_assets/BOWL.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

# ---- geometry in mm ----
H = 58.0            # taller wall (was 39.4) -> more vertical grasp band, deeper grip room
R_OUT = 65.0
WALL = 4.5
BASE_THICK = 15.0
SECTIONS = 160

ZB = -H / 2.0
ZT = H / 2.0
R_IN = R_OUT - WALL

# Cross-section polygon in (r, z), traced around the ceramic material.
# Tall STRAIGHT-WALLED design: the wall is perfectly VERTICAL at r=R_OUT over the
# upper region so the gripper pinches a flat vertical surface (stable no-weld
# grasp); the wall-to-base transition keeps a smooth CURVED fillet (not a sharp
# corner) so a second bowl still seats/nests and the base is not a hard edge.
R_FOOT = 40.0        # outer radius at the base of the fillet
Z_KNEE = ZB + 15.0   # height where the outer fillet reaches full radius -> vertical above
# Outer bottom fillet: smooth arc from the flat foot (R_FOOT, ZB) out to (R_OUT, Z_KNEE).
outer_fillet = np.array([
    [R_FOOT,       ZB],
    [R_FOOT + 8.0, ZB + 2.0],
    [R_FOOT + 15.0, ZB + 5.0],
    [R_OUT - 4.0,  ZB + 9.0],
    [R_OUT - 1.0,  ZB + 12.0],
    [R_OUT,        Z_KNEE],
], dtype=np.float64)
# Inner bottom fillet: mirror, offset in by WALL, closing to the cavity floor.
inner_fillet = np.array([
    [R_IN,             Z_KNEE + 2.0],
    [R_IN - 2.0,       ZB + 12.0],
    [R_IN - 8.0,       ZB + BASE_THICK + 4.0],
    [R_FOOT - WALL - 2.0, ZB + BASE_THICK],
    [0.0,              ZB + BASE_THICK],   # inner bottom closes the cavity
], dtype=np.float64)
profile = np.vstack([
    [[0.0, ZB]],          # bottom center outer
    outer_fillet,         # curved foot -> full radius
    [[R_OUT, ZT]],        # >>> VERTICAL outer wall (grasp band) <<<
    [[R_IN, ZT]],         # inner rim lip
    inner_fillet,         # vertical inner wall -> curved -> cavity floor
]).astype(np.float64)

bowl = trimesh.creation.revolve(profile, sections=SECTIONS)
bowl.apply_translation(-bowl.bounds.mean(axis=0))
bowl.apply_scale(0.001)  # mm -> m
bowl.export(out)

ext = np.round(bowl.extents * 1000, 2)
print(
    f"saved {out} extents(mm)={ext} verts={len(bowl.vertices)} "
    f"faces={len(bowl.faces)} watertight={bowl.is_watertight}"
)
