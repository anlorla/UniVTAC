"""生成"螺丝钉"(图钉状) peg 网格: 圆柱杆 + 顶端平盘(头)。
杆沿 Z, 头在 +Z 端; 居中到原点。
用法: python scripts/asset_tools/build_screw.py [out.obj]   (默认 /tmp/screw_assets/SCREW.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/screw_assets/SCREW.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

SHAFT_D, SHAFT_H = 20.0, 70.0     # 杆 直径 / 高 (mm)
HEAD_W,  HEAD_H  = 44.0, 8.0      # 头(方盘)边长 / 厚 (mm) —— 方形便于夹爪平面夹持、不移位

shaft = trimesh.creation.cylinder(radius=SHAFT_D / 2, height=SHAFT_H, sections=64)
shaft.apply_translation([0, 0, SHAFT_H / 2])                       # 杆底在 z=0
head = trimesh.creation.box(extents=[HEAD_W, HEAD_W, HEAD_H])      # 方形头盘
head.apply_translation([0, 0, SHAFT_H + HEAD_H / 2 - 1.5])         # 盘压在杆顶(嵌 1.5mm)

screw = trimesh.boolean.union([shaft, head])      # 水密单体
screw.apply_translation(-screw.bounds.mean(axis=0))   # 居中, 头端在 +Z
screw.apply_scale(0.001)                              # mm -> m
screw.export(out)
print(f"saved {out}  extents(mm)={np.round(screw.extents*1000,2)}  watertight={screw.is_watertight}")
