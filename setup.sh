#!/usr/bin/env bash
# =============================================================================
# UniVTAC 4090 测试节点一键装 —— 对齐 .150/.56。从本仓根目录跑：bash setup.sh
# 版本真相见 ENV_LOCK.md / requirements.lock；逐步走查+踩坑见 docs/NODE_SETUP.md。
# 本脚本是 best-effort：纯 PyPI 部分全自动；isaacsim/IsaacLab/TacEx(libuipc) 这种
# 重依赖装好后会自检，失败给出明确手动指引（这几样本来就难，别指望 100% 无人值守）。
# =============================================================================
set -euo pipefail

# ============================ 配置区（按需改）================================
ENV_NAME=${ENV_NAME:-UniVTAC}
PY_VER=${PY_VER:-3.10}
WORK=${WORK:-$HOME/CodeSpace}                 # IsaacLab / curobo clone 到这
ISAACLAB_REF=${ISAACLAB_REF:-90b79bb2d44feb8d833f260f2bf37da3487180ba}   # v2.1.1
CUROBO_REF=${CUROBO_REF:-d64c4b005459db10c5dd867d8b30a87d5bda9bdb}
SKIP_APT=${SKIP_APT:-0}
# ============================================================================
REPO=$(cd "$(dirname "$0")" && pwd)
say(){ echo -e "\n\033[1;36m[setup]\033[0m $*"; }
mkdir -p "$WORK"

say "[0/8] 前置检查（应为 Ubuntu 22.04 / glibc 2.35 / 驱动≥535）"
echo "  OS: $(lsb_release -ds 2>/dev/null || echo '?') | glibc $(ldd --version|head -1|grep -oE '[0-9]+\.[0-9]+$' || echo '?') | 驱动 $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null|head -1 || echo none)"
case "$(lsb_release -rs 2>/dev/null)" in 22.04) :;; *) echo "  ⚠️ 不是 22.04，glibc 不匹配可能要源码编译 wheel（见 docs/NODE_SETUP.md）";; esac

if [ "$SKIP_APT" != 1 ]; then
  say "[1/8] apt 依赖"; sudo apt-get update -qq && sudo apt-get install -y ffmpeg build-essential git cmake
fi

say "[2/8] conda env: $ENV_NAME (python $PY_VER)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python="$PY_VER" 2>/dev/null || echo "  env 已存在，复用"
conda activate "$ENV_NAME"

say "[3/8] torch 2.7.0+cu128"
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128

say "[4/8] Isaac Sim 4.5.0.0 (pip)"
pip install 'isaacsim[all,extscache]==4.5.0.0' --extra-index-url https://pypi.nvidia.com

say "[5/8] Isaac Lab v2.1.1 (@$ISAACLAB_REF)"
[ -d "$WORK/IsaacLab/.git" ] || git clone https://github.com/isaac-sim/IsaacLab "$WORK/IsaacLab"
git -C "$WORK/IsaacLab" fetch -q origin && git -C "$WORK/IsaacLab" checkout -q "$ISAACLAB_REF"
( cd "$WORK/IsaacLab" && ./isaaclab.sh -i )    # editable 装 source/* 五个扩展

say "[6/8] curobo (NVlabs @$CUROBO_REF)"
[ -d "$WORK/curobo/.git" ] || git clone https://github.com/NVlabs/curobo "$WORK/curobo"
git -C "$WORK/curobo" fetch -q origin && git -C "$WORK/curobo" checkout -q "$CUROBO_REF"
pip install -e "$WORK/curobo" --no-build-isolation

say "[7/8] TacEx（本仓 third_party，含 libuipc/pyuipc 编译）"
if [ -d "$REPO/third_party/TacEx" ]; then
  ( cd "$REPO/third_party/TacEx" && bash install.sh ) 2>/dev/null \
    || pip install -e "$REPO/third_party/TacEx/source/tacex_assets" \
                   -e "$REPO/third_party/TacEx/source/tacex" \
                   -e "$REPO/third_party/TacEx/source/tacex_uipc" \
                   -e "$REPO/third_party/TacEx/source/tacex_tasks" \
    || echo "  ⚠️ TacEx/libuipc 需手动 build，见 third_party/TacEx/README + docs/NODE_SETUP.md"
else
  echo "  ⚠️ 没有 third_party/TacEx（可能 LFS/子模块没拉全），见 docs/NODE_SETUP.md"
fi

say "[8/8] 其余 PyPI 依赖 + torch_scatter（特殊 wheel）"
pip install numpy==1.26.4 warp-lang==1.12.1 transformers==5.5.4 trimesh==4.11.5 opencv-python==4.11.0.86 scipy==1.15.3
pip install "https://data.pyg.org/whl/torch-2.8.0+cu128/torch_scatter-2.1.2+pt28cu128-cp310-cp310-linux_x86_64.whl" || echo "  torch_scatter 装失败可源码编译（见 docs）"

say "完成。自检："
python - <<'PY' || echo "❌ import 有问题，对照 ENV_LOCK.md 排查"
import torch, isaaclab, tacex_uipc
print("  torch", torch.__version__, "| cuda", torch.version.cuda)
print("  ✅ torch/isaaclab/tacex_uipc 都能 import")
PY
echo "  最终验收：bash collect_data.sh insert_hole demo 0 0 0 1  （跑通 1 局 = 环境 OK）"
echo "  评测：见 eval/run_eval.sh + docs/NODE_SETUP.md（隧道 + server/client）"
