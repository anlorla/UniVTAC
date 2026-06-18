# 新增 4090 测试节点：配置文档

把一台新 4090 配成和 **150/56 一致**的 UniVTAC 测试（Isaac 仿真）节点。
代码用本仓（已对齐 150/56）；环境按 `ENV_LOCK.md` 锁版本。

> 角色提醒：测试节点 = 跑 Isaac 仿真的本地机，**不能走 CFS**（Isaac 是本机二进制）。
> 训练节点才是"挂 CFS 加卡零配置"。本文只讲测试节点。

---

## 0. 前置
- **Ubuntu 22.04**（glibc 2.35）。⚠️ 强烈建议别用 20.04——178 上 20.04/glibc 2.31 逼着源码编译
  torch_scatter 等一堆 wheel，很痛。新机直接上 22.04 省一整天。
- RTX 4090（24G），NVIDIA 驱动 ≥535（580 实测可）。
- 能联网 clone GitHub / 装 pip（isaacsim 走 pypi.nvidia.com）。

## 1. 拉代码
```bash
git clone git@github.com:anlorla/UniVTAC.git ~/UniVTAC   # 已对齐 150/56
cd ~/UniVTAC
```
> 这份 fork 已经把上游 byml 和 150/56 分叉的 8 个文件（`_base_task.py` 的相机/触觉布局、
> `robot/*`、`sensors/tactile.py` 等）对齐回节点版，所以 clone 下来行为就和 150/56 一样。

## 2. 装环境（一键 + 自检）
```bash
bash setup.sh            # 见 setup.sh 顶部"配置区"可改 ENV_NAME/WORK 等
```
`setup.sh` 干的事（对应 `ENV_LOCK.md`）：conda env(py3.10) → torch 2.7.0+cu128 →
isaacsim 4.5.0.0(pip) → IsaacLab v2.1.1(editable) → curobo → TacEx(libuipc) → 其余 PyPI。

纯 PyPI 部分全自动；下面三样是"难点"，装不顺照这里手动来。

### 2a. ⚠️ TacEx / libuipc（最容易卡的一步）
- TacEx 必须从**本仓 `third_party/TacEx`** 装（含 UniVTAC 专属改动，**别装公共版**）。
- 它依赖 libuipc（C++），要本机编译出 `pyuipc`（150 上是
  `third_party/TacEx/source/tacex_uipc/build/python`，pyuipc 0.9.0）。
- 装法以 `third_party/TacEx/README` 为准；setup.sh 会先试 `third_party/TacEx/install.sh`，
  失败再退化成逐个 `pip -e source/*`。libuipc 编译报错时检查：cmake、CUDA toolkit、驱动。
- 535 驱动跑 CUDA 12.4 编译的 libuipc 实测可行（minor compat）。
- 若 `third_party/TacEx` 是空/不全：本仓把 TacEx 资产**直接入库（无 Git LFS）**，正常 clone 就有；
  若缺，检查 clone 是否完整。

### 2b. Isaac Sim / Isaac Lab
- 22.04 上 **isaacsim 直接 pip 装**（`isaacsim[all]==4.5.0.0`，源 `https://pypi.nvidia.com`），
  不需要 binary 版（那是 20.04 的无奈之举）。
- IsaacLab 锁在 **v2.1.1**（commit `90b79bb2d`）。`./isaaclab.sh -i` 会 editable 装 `source/*` 五个扩展。
- 验证：`python -c "import isaaclab, isaacsim; print(isaacsim.__version__)"` → 4.5.0.0。

### 2c. torch_scatter
特殊 wheel（pt28cu128，配 torch 2.7 用，compat 可行）：
`pip install https://data.pyg.org/whl/torch-2.8.0+cu128/torch_scatter-2.1.2+pt28cu128-cp310-cp310-linux_x86_64.whl`

## 3. 验收（必须跑通）
```bash
conda activate UniVTAC
python -c "import torch,isaaclab,tacex_uipc; print(torch.__version__)"   # 2.7.0+cu128
bash collect_data.sh insert_hole demo 0 0 0 1     # 采 1 局；出 hdf5/视频 = 仿真 OK
```
若 TacEx demo 报 `UsdGeom.Mesh(NoneType)` ——那是树自带 ball_rolling demo 的问题，
不影响真实采集/评测，用上面 `collect_data.sh` 验收即可。

## 4. 接入评测（让这台机当第 5、6… 个 eval 槽）
评测是 **闭环 server/client**：策略 server 在 baidu GPU，本机跑 Isaac client。
1. 隧道：本机 ←(Mac `-R`)— Mac —(Mac `-L`)→ baidu serve（4090 不能直连 baidu，走 Mac 双跳）。
2. 起 client：
```bash
cd ~/UniVTAC
TASK=insert_tube PROMPT="Insert the tube into the slot" PORT=29541 TAG=mytest \
  bash eval/run_eval.sh        # 默认 --start_seed 100 --total_num 20（held-out 20 局）
```
3. EE 用 `eval/eval_render_cl.py`、joint 用 `eval/eval_joint_cl.py`；纯数字用 `eval/eval_official_ee.py`。
4. 整套多机分发逻辑（槽/端口轮换/断点续/贪婪队列）见 214 的 `EVAL_DISTRIBUTION_DESIGN.md`。

> ⚠️ held-out：`--start_seed` 要大于该任务采集用过的最大 seed（查 `data/<task>/.../suc_map.txt`），
> 否则评测 seed 和训练集重叠。单 4090 跑 2 个 Isaac ≈ 20-22G，偏紧；重任务(grasp)只跑 1 个。

## 5. 和 150/56 保持一致（以后）
- 代码：以本 fork 为准；150/56 有更新 → 同步到 fork → 各节点 `git pull`。
- 环境：以 `ENV_LOCK.md` / `requirements.lock` 为准；版本漂了照它对齐（"对齐要实查 freeze，别信旧文档"）。
- 别把 lewm（另一个项目）、旧实验脚本混进来——本 fork 走剃刀，只留评测/采集要用的。
