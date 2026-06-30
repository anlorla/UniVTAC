"""Generate a simple two-pin charger plug mesh.

Shape:
- compact charger head body
- two cylindrical pins protruding along -Z

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
BODY = [18.0, 10.0, 18.0]
STRAIN = [12.0, 8.0, 8.0]
PIN_R = 1.1
PIN_L = 8.0
PIN_DX = 4.0

body = trimesh.creation.box(extents=BODY)
body.apply_translation([0.0, 0.0, BODY[2] * 0.5])

strain = trimesh.creation.box(extents=STRAIN)
strain.apply_translation([0.0, 0.0, BODY[2] + STRAIN[2] * 0.5 - 1.0])

pin_l = trimesh.creation.cylinder(radius=PIN_R, height=PIN_L, sections=32)
pin_l.apply_translation([-PIN_DX, 0.0, -PIN_L * 0.5 + 0.6])

pin_r = trimesh.creation.cylinder(radius=PIN_R, height=PIN_L, sections=32)
pin_r.apply_translation([PIN_DX, 0.0, -PIN_L * 0.5 + 0.6])

plug = trimesh.boolean.union([body, strain, pin_l, pin_r])
plug.apply_translation(-plug.bounds.mean(axis=0))
plug.apply_scale(0.001)
plug.export(out)

print(
    f"saved {out} extents(mm)={np.round(plug.extents * 1000, 2)} "
    f"watertight={plug.is_watertight}"
)
