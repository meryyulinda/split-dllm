#!/usr/bin/env bash
# Exp21 edge side: runs the edge program against the four cloud programs (ports 6001-6004).
#   On vast.ai (after setup_exp21_edge.sh and with the SSH tunnel open):
#     cd split-dllm/dasr1 && TAG=vast_3060_tw nohup bash experiments/run_exp21_edge.sh > exp21_edge.log 2>&1 &
#   On the lab server (check that the two programs give the single-process answers):
#     TAG=lab_a6000 GPU=2 nohup bash experiments/run_exp21_edge.sh > exp21_edge_lab.log 2>&1 &
#   SMOKE=1 ...   2 questions, to check that everything connects
# Output: outputs/exp21_deploy_<TAG>[_smoke]/
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
TAG="${TAG:-vast}"
N="${N:-10}"
SLOWDOWN="${SLOWDOWN:-1}"
if [ "${SMOKE:-0}" = "1" ]; then N=2; TAG="${TAG}_smoke"; fi
python - <<PY
import runpy, sys
import configs.experiment_config as c
c.EXP21_TAG = "$TAG"
c.EXP21_N_QUESTIONS = $N
c.EXP21_EDGE_SLOWDOWN = float("$SLOWDOWN")
c.EXP21_EDGE_DEVICE = "cuda:0"
sys.argv = ["experiments/exp21_edge.py"]
runpy.run_path("experiments/exp21_edge.py", run_name="__main__")
PY
