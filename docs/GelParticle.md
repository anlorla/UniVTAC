# 无 marker「凝胶 + 力」表征 (gel_particle)

`gel_particle` 是一种**无 marker** 的视触觉图像表征：银白色凝胶底 + 一层细密彩色**粒子涂层**，粒子随接触**流动**（切向来自 marker 运动、法向来自压痕），另加一层很淡的接触阴影。分辨率与 marker 版一致：**(T, 240, 320, 3) uint8**。

配方（渲染流程，见 `envs/sensors/tactile.py` 的 `get_gel_particle_image` 与离线脚本）：

1. **银底**：由 Taxim 光学图 `rgb` 的亮度重建 `g0 = clip(172 + (lum−mean)·1.1, 128, 214)`，冷色微调 + 灰噪声。
2. **粒子涂层**：全画幅细密彩色斑点（1px，饱和 120–210），alpha 合成。
3. **流动（关键）**：涂层被一个位移场平流 `I(x)=coating(x−D(x))`，`D` = **切向 + 法向**两部分：
   - 切向：`marker` 逐点位移（curr−rest，去刚体），归一化卷积成稠密场 → 剪切/拖拽的流纹；
   - 法向：`depth` 压痕梯度 `−∇(indent)·bulge`（`indent = 静止基线 − depth`）→ 压头压入时把粒子**向外推开**（凝胶隆起）。缺了这项，纯按压（如 grasp_chip，切向仅约 5px）粒子几乎不动、和接触割裂。
4. **接触阴影**：`out ·= 1 − 0.10·norm(depth)`，很淡。

---

## 获取方式一：采集时在环生成（live）

在任务采集配置里加一个 flag，采集时直接产出 `gel_particle`（在环 mesh-bound 版）：

```yaml
sensor_type: gsmini
gel_particle: true          # ← 采集时额外输出无 marker 凝胶图（见 scripts/collect_data.py）
observations:
  tactile:
  - rgb_marker              # 你原有的 marker 版
  - depth
```

分辨率天然和 marker 版一致（同一 `tactile_img_res` 320×240）。示例配置：`task_config/lc_marker_gp.yml`。

---

## 获取方式二：对已采数据离线补 (推荐, 不用重采)

对**已经采好**的 hdf5，用 `scripts/asset_tools/add_gel_particle_offline.py` 离线算出 `gel_particle` 写回文件——**无需重跑仿真**。

**前置要求**（每个 gel 都要有）：`rgb`（干净 Taxim 图）、`marker`（T,2,64,2）、`depth`（T,H,W）。三者是采集默认就存的。

```bash
conda activate UniVTAC   # 需 numpy / opencv / h5py（不需要 isaacsim/GPU）

# 1) 先预览观感（6 阶段拼图，不写文件）:
python scripts/asset_tools/add_gel_particle_offline.py IN.hdf5 --preview /tmp/gp.png

# 2) 原地写入(覆盖已有 gel_particle):
python scripts/asset_tools/add_gel_particle_offline.py IN.hdf5 --inplace

# 3) 或写到副本(不动原文件):
python scripts/asset_tools/add_gel_particle_offline.py IN.hdf5 --out OUT.hdf5
```

- `--gels auto`（默认）：自动处理 `tactile/` 下**所有** gel，**含双臂的 `left/right_tactile` + `_b` 四个**（不会漏）。也可 `--gels left_tactile,right_tactile` 指定。
- `--bulge 0.5`（默认）：法向压入流动强度。想让按压时粒子流动更明显 → 调大；太多 → 调小；`0` = 只有切向。
- 写入的 `tactile/<gel>/gel_particle` 为 `(T,240,320,3) uint8`，gzip 压缩，和在环格式一致。

**批量整份数据集**（对一个采集目录下所有 hdf5 原地补）：

```bash
for h in $(find data/<task>/<collect>/hdf5 -name '*.hdf5'); do
    python scripts/asset_tools/add_gel_particle_offline.py "$h" --inplace
done
```

> ⚠️ `--inplace` 会**覆盖**已有的 `gel_particle`（例如旧版本）。想保留旧的先用 `--out`。

---

## 常见问题

- **数据里已经有 `gel_particle` 了，还要重跑吗？** 要看是哪一版。旧版（无法向流动耦合）在 grasp 这类纯按压任务里粒子几乎不动；跑本脚本会用当前带 `--bulge` 流动的版本**覆盖**掉。
- **只有 `rgb_marker` 没有干净 `rgb`？** 银底需要干净 `rgb`（无黑点）。若只有 `rgb_marker`，marker 黑点会渗进底色，建议采集时保留 `rgb`。
- **想要「数值力」而不是图像？** 那是另一套（`force_field` / `marker_force`，见 `docs/ForceField.md`），需要采集时存 `vertex_force`；`gel_particle` 是**图像**表征，不含标定力。
