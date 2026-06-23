"""生成一个小球(弹珠/小球), 用于"将红杯中的球倒入绿杯"任务。
   球需小到能落进杯里、多个并存、并能从杯口倒出。杯口内径 ~Ø73mm, 杯底内径 ~Ø51mm,
   故取 Ø16mm 小球(半径 8mm): 数个可同存于杯内, 倒杯时顺畅滑出。
   icosphere 给均匀三角面, tet 质量稳定; 居中到原点。
用法: python scripts/asset_tools/build_ball.py [out.obj]   (默认 /tmp/ball_assets/BALL.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ball_assets/BALL.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

R = 8.0          # 球半径 (mm) —— Ø16mm 小球
SUBDIV = 3       # icosphere 细分: 642 顶点/1280 面, 球面平滑、tet 均匀

ball = trimesh.creation.icosphere(subdivisions=SUBDIV, radius=R)
ball.apply_translation(-ball.bounds.mean(axis=0))   # 居中到原点
ball.apply_scale(0.001)                             # mm -> m
ball.export(out)
print(f"saved {out}  extents(mm)={np.round(ball.extents*1000,2)}  "
      f"verts={len(ball.vertices)} faces={len(ball.faces)} watertight={ball.is_watertight}")
