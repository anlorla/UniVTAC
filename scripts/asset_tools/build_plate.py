"""生成一个浅圆盘(餐盘/托盘), 用于"将红杯中的球倒入绿杯"任务。
   两只杯子立在盘内, 盘的低矮边沿可兜住倒球时溅出的球。
   截面回转成实体: 薄盘底 + 低边沿(向内留一圈井, 杯子坐在井里更稳)。盘轴沿 Z, 居中到原点。
   尺寸: 外径 Ø200mm, 边沿高 8mm, 井内径 ~Ø176mm —— 两只 Ø76mm 杯子可并排放下。
用法: python scripts/asset_tools/build_plate.py [out.obj]   (默认 /tmp/plate_assets/PLATE.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/plate_assets/PLATE.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

# ---- 几何 (mm) ----
R_OUT   = 100.0   # 盘外半径 (Ø200)
RIM_H   = 8.0     # 边沿总高
FLOOR   = 3.0     # 盘底(井面)厚度
WALL    = 4.0     # 边沿壁厚 (井内半径 = R_OUT - WALL = 96)
WELL_R  = R_OUT - 12.0  # 井底外半径(边沿内侧斜面到井底, 88mm)
SECTIONS = 128    # 回转细分

r_rim_in = R_OUT - WALL   # 边沿内缘半径 (96)

# 半截面闭合多边形 (r, z): 外底 -> 外壁(边沿)上升 -> 边沿顶 -> 内缘下到井面 -> 井底中心
profile = np.array([
    [0.0,      0.0],      # 外底中心
    [R_OUT,    0.0],      # 外底边
    [R_OUT,    RIM_H],    # 边沿外缘顶
    [r_rim_in, RIM_H],    # 边沿内缘顶
    [WELL_R,   FLOOR],    # 内壁斜面降到井面
    [0.0,      FLOOR],    # 井面中心
])

plate = trimesh.creation.revolve(profile, sections=SECTIONS)   # 绕 Z 回转成实体盘
plate.apply_translation(-plate.bounds.mean(axis=0))            # 居中
plate.apply_scale(0.001)                                       # mm -> m
plate.export(out)
print(f"saved {out}  extents(mm)={np.round(plate.extents*1000,2)}  "
      f"verts={len(plate.vertices)} faces={len(plate.faces)} watertight={plate.is_watertight}")
