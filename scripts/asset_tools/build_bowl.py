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
H = 39.4
R_OUT = 65.0
WALL = 4.5
BASE_THICK = 14.0
SECTIONS = 160

ZB = -H / 2.0
ZT = H / 2.0
R_IN = R_OUT - WALL

# Cross-section polygon in (r, z), traced around the ceramic material.
# Raise the inner bottom / foot stand so a second bowl bottoms out earlier and
# cannot sink as deep when nested. That creates a larger geometry-defined gap.
profile = np.array([
    [0.0,  ZB],          # bottom center outer
    [28.0, ZB],          # wider/taller outer foot stand
    [36.0, ZB + 4.0],
    [43.0, ZB + 10.8],
    [50.0, ZB + 18.5],
    [57.0, ZB + 26.0],
    [61.0, ZB + 31.8],
    [63.0, ZB + 35.1],
    [64.0, ZB + 37.5],
    [R_OUT, ZT],         # outer rim
    [R_IN,  ZT],         # inner rim lip
    [60.5, ZB + 36.9],
    [57.2, ZB + 33.0],
    [51.8, ZB + 25.2],
    [46.2, ZB + 19.0],
    [41.0, ZB + 16.2],
    [35.0, ZB + BASE_THICK],
    [0.0,  ZB + BASE_THICK],   # inner bottom closes the cavity
], dtype=np.float64)

bowl = trimesh.creation.revolve(profile, sections=SECTIONS)
bowl.apply_translation(-bowl.bounds.mean(axis=0))
bowl.apply_scale(0.001)  # mm -> m
bowl.export(out)

ext = np.round(bowl.extents * 1000, 2)
print(
    f"saved {out} extents(mm)={ext} verts={len(bowl.vertices)} "
    f"faces={len(bowl.faces)} watertight={bowl.is_watertight}"
)
