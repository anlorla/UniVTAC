"""Generate a socket base with a vertical insertion hole for the phone plug.

Usage:
  python scripts/asset_tools/build_phone_socket_base.py [out.obj]
"""
import os
import sys
import trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phone_socket_assets/PHONE_SOCKET_BASE.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

# Plug tongue section is 8.8 x 2.8 mm.
# Keep generous clearance so the unplug/replug task is motion-planning dominated,
# not contact-jam dominated.
CLR = 1.3
HOLE_W = 8.8 + 2 * CLR
HOLE_T = 2.8 + 2 * CLR
BASE = [28.0, 18.0, 12.0]
HOLE_DEPTH = 7.0
LEAD = 1.0

outer = trimesh.creation.box(extents=BASE)
outer.apply_translation([0.0, 0.0, BASE[2] * 0.5])

main = trimesh.creation.box(extents=[HOLE_W, HOLE_T, HOLE_DEPTH * 2.0])
main.apply_translation([0.0, 0.0, BASE[2]])

lead = trimesh.creation.box(extents=[HOLE_W + 2 * LEAD, HOLE_T + 2 * LEAD, 3.0])
lead.apply_translation([0.0, 0.0, BASE[2]])

slot = trimesh.boolean.difference([outer, main, lead])
slot.apply_scale(0.001)
slot.export(out)

print(
    f"saved {out} hole(mm)={HOLE_W:.2f}x{HOLE_T:.2f} depth={HOLE_DEPTH:.2f} "
    f"watertight={slot.is_watertight}"
)
