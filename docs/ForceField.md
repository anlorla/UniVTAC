# 触觉力表征:采集与离线计算 (Tactile Force Field)

本文说明如何在采集时**只存无损的逐顶点接触力**,采集后**离线**算出稠密的 `64×48×3` 力场表征(`force_field`)。理念类似相机的 RAW → 冲洗:贵的仿真只跑一次,栅格表征随时离线重算。

---

## 1. 力表征有哪些

在环仿真里(`envs/sensors/tactile.py`),每个 gel 可输出:

| obs key | 形状 | 说明 | 是否插值 |
|---|---|---|---|
| `contact_force` | `(T, N_v, 3)` | 逐顶点接触力,**世界系**(UIPC 求解器原始输出,稀疏) | 否(原始) |
| `vertex_force` | `(T, N_v, 3)` | 逐顶点接触力,**传感器系**(xy=剪切, z=法向)。**采集就存这个** | 否(原始) |
| `force_field` | `(T, 48, 64, 3)` | 稠密 `64×48×3` 力场,把逐顶点力**重心插值**到规则栅格,传感器系 | 是 |
| `marker_force` | `(T, 63, 3)` | 9×7 marker 栅格的力,传感器系 | 是 |

- `N_v` = gel 网格顶点数:粗网格 **169**,密网格(方案 B)**615**。
- 采集时**建议只存 `vertex_force`**(无损、比 `force_field` 还小),`force_field` 离线算。

---

## 2. 采集(存 vertex_force)

用 `task_config/collect_force.yml`(或自建),关键是 `tactile` 里放 `vertex_force`、**不放** `force_field`/`*_img`:

```yaml
sensor_type: gsmini
dense_gelpad: false        # false=粗网格169; true=密网格615(方案 B, 更高保真, 约2.6x慢)
observations:
  tactile:
    - rgb_marker           # gel 视触觉图
    - depth
    - vertex_force         # ← 逐顶点接触力(传感器系, 无损)
```

跑采集(节点 `.56` 用二进制 Isaac,需先 source):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate UniVTAC
source ~/yifan/isaacsim/setup_conda_env.sh
export HEADLESS=1
python scripts/collect_data.py <task> collect_force --episode_num 50 --gpu 0
```

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
# 可选: --out-key <名字>(默认写成 force_field), --meta-dir <ff_meta所在目录>
```

完成后每个 hdf5 的 `tactile/<gel>/` 下会多出 `force_field`,形状 `(T, 48, 64, 3)`。

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

`force_field` 那张 64×48 **始终是插值**,保真度只取决于**网格密度**:

| 网格 | `dense_gelpad` | 峰值真实接触顶点 | 每步开销 | force_field |
|---|---|---|---|---|
| 粗 169 | `false` | ~40 | 1× | 正确,mesh 级分辨率 |
| 密 615(方案 B) | `true` | ~74(≈1.8×) | ~2.6× | 正确,更高保真 |

- 64×48 = 3072 格,远密于网格顶点,所以**大部分是上采样**;密网格(615)提供约 2× 真实样本,让同一张 64×48 更实。
- 想"填满"64×48 到无损需要网格≈栅格(约 3000 顶面顶点),代价极高、不现实。详见 `docs/`(映射与保真度分析)/桌面 PDF。
- 密网格需先生成资产:`python scripts/asset_tools/make_dense_gelpad.py`(一次性,见脚本头注释)。

---

## 6. 可视化(可选)

`force_field` 的可视化可离线从数值渲染(绿→红彩色箭头 quiver,或 fx/fy/fz 热力图);采集时也可临时加 `force_field_img` obs 直接存图,但**正式采集不建议存图**(比数值大 ~16×,零额外信息)。

---

## 附:数据键速查

```
tactile/<gel>/vertex_force    (T, N_v, 3)   逐顶点力, 传感器系  ← 采集存这个
tactile/<gel>/contact_force   (T, N_v, 3)   逐顶点力, 世界系(可选)
tactile/<gel>/force_field     (T, 48,64,3)  稠密力场(离线生成或在环存)
<run>/ff_meta_<gel>.npz       栅格↔网格绑定(离线重建必需)
```
