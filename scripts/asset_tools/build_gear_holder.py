"""Generate a gear holder: round plate with three vertical pegs.

Usage:
  python scripts/asset_tools/build_gear_holder.py [out.obj]
"""
import os
import sys
import numpy as np
import trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gear_holder_assets/GEAR_HOLDER.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

# mm
PLATE_R = 66.0
PLATE_H = 8.0
PEG_R = 5.2
PEG_H = 28.0
PEG_LAYOUT_R = 32.0
SECTIONS = 16
PEG_SECTIONS = 8

plate = trimesh.creation.cylinder(radius=PLATE_R, height=PLATE_H, sections=SECTIONS)
plate.apply_translation([0.0, 0.0, PLATE_H * 0.5])

parts = [plate]
peg_angles = np.deg2rad([90.0, 210.0, 330.0])
for a in peg_angles:
    peg = trimesh.creation.cylinder(radius=PEG_R, height=PEG_H, sections=PEG_SECTIONS)
    peg.apply_translation([
        PEG_LAYOUT_R * np.cos(a),
        PEG_LAYOUT_R * np.sin(a),
        PLATE_H + PEG_H * 0.5 - 0.5,
    ])
    parts.append(peg)

holder = trimesh.boolean.union(parts)
holder.apply_translation(-holder.bounds.mean(axis=0))
holder.apply_scale(0.001)
holder.export(out)

print(
    f"saved {out} plate_d(mm)={PLATE_R * 2:.1f} peg_r(mm)={PEG_R:.1f} "
    f"peg_layout_r(mm)={PEG_LAYOUT_R:.1f} watertight={holder.is_watertight}"
)
