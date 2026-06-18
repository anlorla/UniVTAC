#!/bin/bash
# 178 评测驱动。用法: TASK=grasp_classify PROMPT="Grasp the object" PORT=29541 TAG=notac_gc_0613 bash run_eval.sh
set +u
TASK=${TASK:?}; PROMPT=${PROMPT:?}; PORT=${PORT:-29541}; TAG=${TAG:?}
OUT=~/yifan/eval/results/$TAG
mkdir -p $OUT
source ~/miniconda3/etc/profile.d/conda.sh && conda activate UniVTAC && source ~/isaacsim/setup_conda_env.sh 2>/dev/null
# 等 server ready(隧道通+端口响应)
for i in $(seq 1 60); do timeout 3 bash -c "echo > /dev/tcp/${EVAL_HOST:-localhost}/$PORT" 2>/dev/null && break; sleep 10; done
{ cd ~/yifan/CodeSpace/UniVTAC 2>/dev/null || cd ~/CodeSpace/UniVTAC; }
cd "$(cd "$(dirname "$0")/.." && pwd)"; python eval/eval_official_ee.py $TASK demo \
  --vta_host ${EVAL_HOST:-localhost} --vta_port $PORT --prompt "$PROMPT" \
  --start_seed 100 --total_num 20 2>&1 | tee $OUT/eval.log
echo EVAL_DONE_$TAG >> $OUT/eval.log
