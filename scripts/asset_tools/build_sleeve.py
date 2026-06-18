"""生成"套子": 外方内圆的套筒(外壁方形便于夹爪平面夹持、不滑; 内圆通孔套在螺丝杆上)。
孔轴沿 Z, 居中。
用法: python scripts/asset_tools/build_sleeve.py [out.obj]   (默认 /tmp/screw_assets/SLEEVE.obj)
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/screw_assets/SLEEVE.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

SHAFT_D = 20.0                 # 配套杆径 (mm), 与 build_screw.py 一致
CLR = 4.0                      # 单边间隙 (mm): 放宽到 4mm, 容下横向插入 ~2-3mm 对准误差
INNER_D = SHAFT_D + 2 * CLR    # 内孔直径 28 (圆)
OUTER_W = 36.0                 # 外壁方形边长 (mm) —— 平面便于夹爪夹持
H = 44.0                       # 套子长 (通孔, 上下都通)

outer = trimesh.creation.box(extents=[OUTER_W, OUTER_W, H])         # 方形外壁
inner = trimesh.creation.cylinder(radius=INNER_D / 2, height=H * 1.1, sections=64)  # 圆形通孔(略长挖通)
sleeve = trimesh.boolean.difference([outer, inner])   # 外方内圆

sleeve.apply_translation(-sleeve.bounds.mean(axis=0))  # 居中
sleeve.apply_scale(0.001)                              # mm -> m
sleeve.export(out)
print(f"saved {out}  outer(方){OUTER_W}mm / inner(圆){INNER_D}mm  H={H}  watertight={sleeve.is_watertight}")
