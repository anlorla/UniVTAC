"""生成一个【更大、壁更矮的碗】形托盘(本质就是浅口宽碗), 用于双臂叠盘任务(dual_plate_place_stack)。
   与 build_bowl.py 同一套设计: 外壁上半段【完全竖直】(给夹爪一个竖直平面 -> no-weld 也夹得住),
   壁到底用平滑【圆弧】过渡(不是硬转角), 第二只盘可套叠。比碗更宽(Ø190)、壁更矮(H30)。
   注意: 只喂给 dual_plate_place_stack (烤成 PLATE_STACK.usd); 不要覆盖 assets/objects/PLATE.usd
   —— 那个浅盘还被 pour_ball / dual_bowl_unstack / dual_cup_handover_place 共用。
用法: python scripts/asset_tools/build_plate.py [out.obj]   (默认 /tmp/plate_assets/PLATE.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/plate_assets/PLATE.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

# ---- 几何 (mm) ----
H = 40.0            # 壁高(比碗 58mm 矮, 但比纯浅盘高, 给夹取留出空间)
R_OUT = 95.0        # 外半径(Ø190, 比碗 Ø130 更大)
WALL = 8.0          # 加厚沿壁(4.5->8mm): 宽薄盘中段壁太软会被夹弯, 加厚变刚才钳得住
BASE_THICK = 8.0
SECTIONS = 160

ZB = -H / 2.0
ZT = H / 2.0
R_IN = R_OUT - WALL

# 矮壁竖直设计: 上半段外壁 r=R_OUT 完全竖直(夹取带); 壁到底圆弧过渡, 圈足更窄以便套叠。
R_FOOT = 65.0        # 底部圈足外半径(圆弧起点)
Z_KNEE = ZB + 12.0   # 外壁圆弧到达满半径的高度 -> 以上竖直
outer_fillet = np.array([
    [R_FOOT,        ZB],
    [R_FOOT + 9.0,  ZB + 1.5],
    [R_FOOT + 18.0, ZB + 4.0],
    [R_OUT - 4.0,   ZB + 7.0],
    [R_OUT - 1.0,   ZB + 9.0],
    [R_OUT,         Z_KNEE],
], dtype=np.float64)
inner_fillet = np.array([
    [R_IN,                Z_KNEE + 2.0],
    [R_IN - 2.0,          ZB + 8.0],
    [R_IN - 10.0,         ZB + BASE_THICK + 3.0],
    [R_FOOT - WALL - 2.0, ZB + BASE_THICK],
    [0.0,                 ZB + BASE_THICK],   # 井底封腔
], dtype=np.float64)
profile = np.vstack([
    [[0.0, ZB]],          # 外底中心
    outer_fillet,         # 圆弧圈足 -> 满半径
    [[R_OUT, ZT]],        # >>> 竖直外壁(夹取带) <<<
    [[R_IN, ZT]],         # 内缘沿口
    inner_fillet,         # 竖直内壁 -> 圆弧 -> 井底
]).astype(np.float64)

plate = trimesh.creation.revolve(profile, sections=SECTIONS)   # 绕 Z 回转成实体
plate.apply_translation(-plate.bounds.mean(axis=0))            # 居中
plate.apply_scale(0.001)                                       # mm -> m
plate.export(out)
print(f"saved {out}  extents(mm)={np.round(plate.extents*1000,2)}  "
      f"verts={len(plate.vertices)} faces={len(plate.faces)} watertight={plate.is_watertight}")
