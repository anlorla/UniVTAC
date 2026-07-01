# 触觉力表征:采集与离线计算 (Tactile Force Field)

本文说明如何在采集时**只存无损的逐顶点接触力**,采集后**离线**算出稠密的 `64×48×3` 力场表征(`force_field`)。理念类似相机的 RAW → 冲洗:贵的仿真只跑一次,栅格表征随时离线重算。

---

## 1. 力表征有哪些

在环仿真里(`envs/sensors/tactile.py`),每个 gel 可输出:

| obs key | 形状 | 说明 | 是否插值 |
|---|---|---|---|
| `contact_force` | `(T, N_v, 3)` | 逐顶点接触力,**世界系**(UIPC 求解器原始输出,稀疏) | 否(原始) |
| `vertex_force` | `(T, N_v, 3)` | 逐顶点接触力,**传感器系**(xy=剪切, z=法向)。**采集就存这个** | 否(原始) |
| `force_field` | `(T, H, W, 3)` | 稠密力场(默认 `48×64`),把逐顶点力**重心插值**到规则栅格,传感器系 | 是 |
| `marker_force` | `(T, 63, 3)` | 9×7 marker 栅格的力,传感器系 | 是 |

- `N_v` = gel 网格顶点数:粗网格 **169**,密网格(方案 B)**615**。
- 采集时**建议只存 `vertex_force`**(无损、比 `force_field` 还小),`force_field` 离线算。

---

## 2. 采集(存 vertex_force)

用 `task_config/collect_force.yml`(或自建),关键是 `tactile` 里放 `vertex_force`、**不放** `force_field`/`*_img`:

```yaml
sensor_type: gsmini
dense_gelpad: false        # 网格密度: false=粗169(快,mesh级); true=密615(方案B,~1.8x保真,~2.6x慢)
force_field_grid: [64, 48] # 力场栅格 (W,H): 64x48 冗余大; 建议 ~mesh级如 [16,12](真信息一样,更好学)
observations:
  tactile:
    - rgb_marker           # gel 视触觉图
    - depth
    - vertex_force         # ← 逐顶点接触力(传感器系, 无损)
```

**两个权衡开关**(见第 5 节):
- `dense_gelpad`:网格密度 = 真实分辨率(粗 169 / 密 615)。
- `force_field_grid`:输出栅格 `(W,H)`。只影响"表示形状",不增真信息;也是离线重建的默认栅格。可留 `[64,48]` 但离线随时能换(`--grid`)。

跑采集(节点 `.56` 用二进制 Isaac,需先 source):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate UniVTAC
source ~/yifan/isaacsim/setup_conda_env.sh
export HEADLESS=1
python scripts/collect_data.py <task_name> collect_force --episode_num 50 --gpu 0
```

### task_name 在哪 / 采集逻辑改哪里

- **`<task_name>` = `envs/` 下的任务文件名(去掉 `.py`)**。例如 `lift_can` ↔ `envs/lift_can.py`,`insert_hole` ↔ `envs/insert_hole.py`。当前可选:`collect, grasp_classify, grasp_chip, insert_HDMI, insert_hole, insert_tube, insert_USB, lift_bottle, lift_can, pull_out_key, put_bottle_in_shelf, pour_ball, phone_socket_replug, dual_*`(`dual_*` 为双臂)。
- **命令第 2 个参数是 `task_config/` 下的 yaml 名**(去掉 `.yml`)。所以 `... lift_can collect_force ...` = 用 `envs/lift_can.py` 的任务逻辑 + `task_config/collect_force.yml` 的观测/开关配置。
- **要改"采集时某个任务怎么动/存什么",就改对应的 `envs/<task_name>.py`**(动作序列、抓取判定、actor 布置等),观测/力表征相关的逻辑在 `envs/sensors/tactile.py`,基类流程(存盘、渲染、force_field_meta 落盘)在 `envs/_base_task.py`。

采集会在运行目录写:
```
<save_dir>/<task>/collect_force/
├── hdf5/<seed>.hdf5          # 每条轨迹, 含 tactile/<gel>/vertex_force
├── ff_meta_left_tactile.npz  # 栅格↔网格 的重心绑定(每次运行自动写一次)
└── ff_meta_right_tactile.npz
```

> `ff_meta_<gel>.npz` 是离线重建 `force_field` 必需的绑定(网格固定,和 seed 无关)。**别删。**

---

## 3. 离线计算 force_field(不需要 GPU / Isaac)

采集完,用纯 numpy/h5py 的脚本把 `vertex_force` → `force_field` 写回 hdf5:

```bash
# 处理整个运行目录(会递归找所有 .hdf5):
python scripts/asset_tools/contact_force_to_field.py <save_dir>/<task>/collect_force
# 或单个文件:
python scripts/asset_tools/contact_force_to_field.py path/to/0.hdf5
# 换栅格分辨率(不用重跑仿真, 从参考表面重建绑定):
python scripts/asset_tools/contact_force_to_field.py path/to/0.hdf5 --grid 16 12
# 可选: --out-key <名字>(默认 force_field), --meta-dir <ff_meta所在目录>
```

完成后每个 hdf5 的 `tactile/<gel>/` 下会多出 `force_field`,形状 `(T, H, W, 3)`(默认 `(T,48,64,3)`;`--grid 16 12` 则 `(T,12,16,3)`)。**同一份 `vertex_force` 可离线出任意栅格**,想换分辨率随时重跑,不用重新采集。

**它和在环算的完全一致**(逐元素误差 ~1e-10,已验证)。原理:`force_field = 逐顶点力 的重心插值到 64×48 栅格`;因为存的是传感器系的 `vertex_force`,离线只需插值、无需再旋转(旋转与加权和可交换)。

---

## 4. 读取结果

```python
import h5py, numpy as np
f = h5py.File("0.hdf5", "r")
ff = f["tactile/left_tactile/force_field"][:]   # (T, 48, 64, 3)  float32
# 语义: 最后一维 = (fx, fy, fz) 传感器系; fx,fy=平面剪切, fz=法向
vf = f["tactile/left_tactile/vertex_force"][:]  # (T, N_v, 3) 原始逐顶点力(想自己插值/换栅格时用)
```

- **48 行 × 64 列 × 3**(H×W×3)。想要 `(64,48,3)` 就 `ff.transpose(0,2,1,3)`。
- 单位是 UIPC 接触梯度的力单位(要标定成牛顿需乘一个比例,视材料/尺度而定)。

---

## 5. 保真度与网格选择(重要)

有**两个独立开关**,别混:

**(a) `dense_gelpad` = 网格密度 = 真实分辨率(唯一真正增信息的杠杆)**

| 网格 | `dense_gelpad` | 峰值真实接触顶点 | 每步开销 |
|---|---|---|---|
| 粗 169 | `false` | ~40 | 1× |
| 密 615(方案 B) | `true` | ~74(≈1.8×) | ~2.6× |

**(b) `force_field_grid` = 输出栅格 `(W,H)` = 表示形状(不增真信息)**

`force_field` **始终是逐顶点力的重心插值**,给定网格,插值已是最优重建。`64×48 = 3072` 格远密于网格顶点(密网格有效分辨率也才 ~9–14),所以 **64×48 大部分是上采样/零,对学习是冗余**。

**建议:**
- **要更真** → 开 `dense_gelpad: true`(网格密度),不是调大栅格。
- **喂网络学习** → `force_field_grid` 用 **~mesh 级(如 `[16,12]`)**:真信息一样、冗余小、更好学、更省;要 64×48 的形状,训练时或离线 `--grid` 再放大。**并且一定对 force 三通道逐通道标准化**(值 ~1e-4,量级本身无所谓)。
- 想"填满" 64×48 到无损需网格≈栅格(~3000 顶面顶点),代价极高、不现实。详见桌面 PDF(映射与保真度分析)。
- 密网格需先生成资产:`python scripts/asset_tools/make_dense_gelpad.py`(一次性)。密网格的表面提取已按 gel 局部系修正(PATCH-F);否则夹爪姿态一变 force_field 会是零。

---

## 6. 可视化(可选)

`force_field` 的可视化可离线从数值渲染(绿→红彩色箭头 quiver,或 fx/fy/fz 热力图);采集时也可临时加 `force_field_img` obs 直接存图,但**正式采集不建议存图**(比数值大 ~16×,零额外信息)。

---

## 附:数据键速查

```
tactile/<gel>/vertex_force    (T, N_v, 3)   逐顶点力, 传感器系  ← 采集存这个
tactile/<gel>/contact_force   (T, N_v, 3)   逐顶点力, 世界系(可选)
tactile/<gel>/force_field     (T, H, W, 3)  稠密力场(离线生成或在环存, 默认 48x64)
<run>/ff_meta_<gel>.npz       栅格↔网格绑定(离线重建必需)
```
