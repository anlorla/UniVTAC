"""Generate a socket base with two vertical pin holes.

Usage:
  python scripts/asset_tools/build_phone_socket_base.py [out.obj]
"""
import os
import sys
import trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phone_socket_assets/PHONE_SOCKET_BASE.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

BASE = [30.0, 20.0, 12.0]
PIN_R = 1.1
CLR = 0.85
PIN_DX = 4.0
HOLE_DEPTH = 11.5
LEAD_R = 1.3
LEAD_H = 1.5


def make_lead_in(radius_top, radius_bottom, height, sections=32):
    top_overhang = 0.2
    slope_depth = height + top_overhang
    cone_height = slope_depth * radius_top / (radius_top - radius_bottom)
    lead = trimesh.creation.cone(radius=radius_top, height=cone_height, sections=sections)
    lead.apply_scale([1.0, 1.0, -1.0])
    lead.apply_translation([0.0, 0.0, top_overhang])
    return lead


outer = trimesh.creation.box(extents=BASE)
outer.apply_translation([0.0, 0.0, BASE[2] * 0.5])

hole_r = PIN_R + CLR
lead_r = hole_r + LEAD_R

main_l = trimesh.creation.cylinder(radius=hole_r, height=HOLE_DEPTH * 2.0, sections=32)
main_l.apply_translation([-PIN_DX, 0.0, BASE[2]])
main_r = trimesh.creation.cylinder(radius=hole_r, height=HOLE_DEPTH * 2.0, sections=32)
main_r.apply_translation([PIN_DX, 0.0, BASE[2]])

lead_l = make_lead_in(radius_top=lead_r, radius_bottom=hole_r, height=LEAD_H, sections=32)
lead_l.apply_translation([-PIN_DX, 0.0, BASE[2]])
lead_r_mesh = make_lead_in(radius_top=lead_r, radius_bottom=hole_r, height=LEAD_H, sections=32)
lead_r_mesh.apply_translation([PIN_DX, 0.0, BASE[2]])

cut_l = trimesh.boolean.union([main_l, lead_l])
cut_r = trimesh.boolean.union([main_r, lead_r_mesh])
slot = trimesh.boolean.difference([outer, cut_l, cut_r])
slot.merge_vertices()
slot.update_faces(slot.unique_faces())
slot.update_faces(slot.nondegenerate_faces())
slot.apply_scale(0.001)
slot.export(out)

print(
    f"saved {out} hole_r(mm)={hole_r:.2f} pin_dx(mm)={PIN_DX:.2f} depth(mm)={HOLE_DEPTH:.2f} "
    f"watertight={slot.is_watertight}"
)
