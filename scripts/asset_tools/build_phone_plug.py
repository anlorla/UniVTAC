"""Generate a simple phone charging plug mesh.

Shape:
- a compact cable-head body
- a smaller symmetric charging tongue protruding along -Z

Usage:
  python scripts/asset_tools/build_phone_plug.py [out.obj]
"""
import os
import sys
import numpy as np
import trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phone_socket_assets/PHONE_PLUG.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

# mm
BODY = [14.0, 8.0, 22.0]
STRAIN = [10.0, 6.0, 8.0]
TONGUE = [8.8, 2.8, 6.0]

body = trimesh.creation.box(extents=BODY)
body.apply_translation([0.0, 0.0, BODY[2] * 0.5])

strain = trimesh.creation.box(extents=STRAIN)
strain.apply_translation([0.0, 0.0, BODY[2] + STRAIN[2] * 0.5 - 1.5])

tongue = trimesh.creation.box(extents=TONGUE)
tongue.apply_translation([0.0, 0.0, -TONGUE[2] * 0.5 + 1.0])

plug = trimesh.boolean.union([body, strain, tongue])
plug.apply_translation(-plug.bounds.mean(axis=0))
plug.apply_scale(0.001)
plug.export(out)

print(
    f"saved {out} extents(mm)={np.round(plug.extents * 1000, 2)} "
    f"watertight={plug.is_watertight}"
)
