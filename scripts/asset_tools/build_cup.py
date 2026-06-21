"""生成"纸杯"网格: 上宽下窄的截锥(frustum), 薄壁、闭底、开口朝上, 顶端带一圈外翻卷边(lip)。
   卷边让两个杯子套叠时卡住留缝 -> 不易分离(对应"分离套叠纸杯"任务)。
   杯轴沿 Z, 杯口在 +Z 端; 居中到原点。
用法: python scripts/asset_tools/build_cup.py [out.obj]   (默认 /tmp/cup_assets/CUP.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cup_assets/CUP.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

# ---- 几何 (mm) ----
R_TOP   = 38.0    # 杯口外半径 (Ø76)
R_BASE  = 27.0    # 杯底外半径 (Ø54) —— 上宽下窄, 锥度便于套叠
H       = 92.0    # 杯高
WALL    = 1.6     # 壁厚 (薄壁纸杯)
BOTTOM  = 2.5     # 底厚
LIP_OUT = 2.2     # 卷边外凸量
LIP_H   = 4.0     # 卷边高度(从杯口往下)
SECTIONS = 96     # 回转细分

# 杯口处内半径(壁顶内缘)
r_top_in  = R_TOP - WALL
# 杯底处内半径(内壁底)
r_base_in = R_BASE - WALL

# 半截面闭合多边形 (r, z): 沿实体壁外轮廓向上 -> 卷边 -> 内轮廓向下 -> 闭底
profile = np.array([
    [0.0,        0.0],          # 外底中心
    [R_BASE,     0.0],          # 外底边
    [R_TOP,      H - LIP_H],    # 外壁升到卷边下沿
    [R_TOP + LIP_OUT, H - LIP_H],  # 卷边外凸
    [R_TOP + LIP_OUT, H],       # 卷边顶外缘
    [r_top_in,   H],            # 杯口内缘
    [r_base_in,  BOTTOM],       # 内壁降到内底
    [0.0,        BOTTOM],       # 内底中心
])

cup = trimesh.creation.revolve(profile, sections=SECTIONS)   # 绕 Z 回转成实体杯
cup.apply_translation(-cup.bounds.mean(axis=0))              # 居中
cup.apply_scale(0.001)                                       # mm -> m
cup.export(out)
print(f"saved {out}  extents(mm)={np.round(cup.extents*1000,2)}  "
      f"verts={len(cup.vertices)} faces={len(cup.faces)} watertight={cup.is_watertight}")
