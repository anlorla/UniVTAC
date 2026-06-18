# UniVTAC 测试节点环境锁（从 .150 实抓，2026-06-18）

新 4090 测试节点要和 **150/56 完全一致**，按此版本装。`requirements.lock` 是 .150 上
conda 环境 `UniVTAC` 的完整 `pip freeze`（274 个包，参考真相——文档会过时，以这份为准）。

一键装见 `setup.sh`；逐步走查 + 踩坑见 `docs/NODE_SETUP.md`。

## 系统层（硬前置）
| 项 | 值 | 说明 |
|---|---|---|
| OS | **Ubuntu 22.04.5 LTS**（glibc 2.35） | ⚠️ **别用 20.04**：glibc 2.31 会逼你源码编译一堆 wheel（torch_scatter 等），178 上吃过大亏 |
| GPU / 驱动 | RTX 4090 / NVIDIA **580.x**（≥535 即可，minor compat 跑 CUDA 12.x） | |
| ffmpeg | 4.4.2（`sudo apt install ffmpeg`） | apt 层 pip freeze 覆盖不到，单独装 |
| 编译链 | build-essential / cmake / git | TacEx(libuipc) 要编译 |

## conda env `UniVTAC`，Python **3.10.20**

### 核心硬约束（错一个就不跑）
| 包 | 版本 | 装法 |
|---|---|---|
| torch | **2.7.0+cu128** | pip，index cu128 |
| torchvision | 0.22.0+cu128 | pip，index cu128 |
| numpy | **1.26.4** | pip |
| isaacsim | **4.5.0.0**（含全部 `isaacsim-*` 扩展） | pip（22.04 可直接 pip 装，不用 binary） |
| IsaacLab | **v2.1.1**（commit `90b79bb2d`，editable `source/*` 五个扩展） | git clone + `./isaaclab.sh -i` |
| nvidia_curobo | 0.7.8（NVlabs/curobo `@d64c4b0`，editable） | git clone + `pip -e` |
| warp-lang | 1.12.1 | pip |
| pyuipc | 0.9.0（**本地 build**：`third_party/TacEx/source/tacex_uipc/build/python`） | 编译 libuipc |
| tacex / tacex_assets / tacex_uipc / tacex_tasks | 0.1.0（**本仓 `third_party/TacEx`**，editable） | `pip -e source/*` |
| torch_scatter | 2.1.2（pt28cu128 wheel） | 特殊 wheel，见 setup.sh |
| transformers 5.5.4 ｜ trimesh 4.11.5 ｜ opencv-python 4.11.0.86 ｜ scipy 1.15.3 | | pip |

### `requirements.lock` 里这几行是机器本地路径，**不能直接 `pip install -r`**，由 setup.sh 单独装：
- `-e git+https://github.com/isaac-sim/IsaacLab@90b79bb2d...`（×5：isaaclab / _assets / _mimic / _rl / _tasks）
- `-e git+https://github.com/NVlabs/curobo.git@d64c4b0...#egg=nvidia_curobo`
- `pyuipc @ file:///.../third_party/TacEx/source/tacex_uipc/build/python`（本地 build）
- `-e /tmp/openpi-client`（评测远程推理客户端，要跑 `eval/` 才需要）

## 验收
```bash
conda activate UniVTAC && source <isaacsim>/setup_conda_env.sh   # 若 binary；pip 装则无需
python -c "import torch,isaaclab,tacex_uipc; print(torch.__version__)"   # 2.7.0+cu128
bash collect_data.sh insert_hole demo 0 0 0 1   # 跑通 1 局采集 = 环境 OK
```
