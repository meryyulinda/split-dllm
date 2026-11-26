#!/usr/bin/env bash
# Exp20: breadth -- answer quality on GSM8K (100 problems, 3 seeds) and HumanEval (164, pass@1)
# for both backends, on ONE GPU (no timing, so the GPU may be shared with other jobs; it only
# needs ~20 GB of free memory). One (backend, task) per output folder, so jobs can run in
# parallel on different GPUs:
#   nohup bash experiments/run_exp20.sh > exp20.log 2>&1 &                       # all, one GPU, ~10 h
#   GPU=1 BACKENDS=dllm_cache TASKS=gsm8k     nohup bash experiments/run_exp20.sh > exp20_a.log 2>&1 &
#   GPU=2 BACKENDS=dllm_cache TASKS=humaneval nohup bash experiments/run_exp20.sh > exp20_b.log 2>&1 &
#   GPU=0 BACKENDS=fastdllm                   nohup bash experiments/run_exp20.sh > exp20_c.log 2>&1 &
#   SMOKE=1 bash experiments/run_exp20.sh > exp20_smoke.log 2>&1                 # ~10 min
# Approximate run times on an idle A6000: dLLM-Cache GSM8K ~5 h, dLLM-Cache HumanEval ~2.5 h,
# Fast-dLLM (both tasks) ~2 h. Waits by itself (every 5 min) until the GPU has MIN_FREE_MIB free.
# HumanEval executes the model's code against the unit tests (separate process, 10 s, 2 GiB).
# Outputs: outputs/exp20_breadth_<backend>_dream_<task>[_smoke]/
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
BACKENDS="${BACKENDS:-dllm_cache fastdllm}"
TASKS="${TASKS:-gsm8k humaneval}"
SMOKE="${SMOKE:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
echo "GPU $GPU | backends: $BACKENDS | tasks: $TASKS"

if [ ! -f ../../human-eval/data/HumanEval.jsonl.gz ] && [[ " $TASKS " == *" humaneval "* ]]; then
  echo "cloning HumanEval to ../../human-eval"
  git clone --depth 1 https://github.com/openai/human-eval.git ../../human-eval
fi

while true; do   # wait until the GPU has enough free memory
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "$free" -ge "$MIN_FREE_MIB" ]; then echo "$(date '+%F %T') GPU $GPU: ${free} MiB free, starting."; break; fi
  echo "$(date '+%F %T') GPU $GPU: only ${free} MiB free; checking again in ${WAIT_SECONDS} s ..."
  sleep "$WAIT_SECONDS"
done

if [ "$SMOKE" = "1" ]; then
  SIZE='EXP20_N_GSM8K = 3
EXP20_N_HUMANEVAL = 3
EXP20_SEEDS = {"gsm8k": [42], "humaneval": [42]}
EXP20_NEW_TOKENS = {"gsm8k": 64, "humaneval": 128}'
  SUFFIX="_smoke"
else
  SIZE='EXP20_N_GSM8K = 100
EXP20_N_HUMANEVAL = None
EXP20_SEEDS = {"gsm8k": [42, 43, 44], "humaneval": [42]}
EXP20_NEW_TOKENS = {"gsm8k": 256, "humaneval": 256}'
  SUFFIX=""
fi

for backend in $BACKENDS; do
  for task in $TASKS; do
    echo; echo "=== $backend, $task  ($(date '+%F %T'))"
    python - "EXP20_BACKEND = '$backend'
EXP20_TASKS = ['$task']
EXP20_OUTPUT_TAG = '_${task}${SUFFIX}'
EXP20_DEVICE = 'cuda:0'
EXP20_SPLIT_LAYER = 4
EXP20_RESUME = True
EXP20_HUMANEVAL = '../../human-eval/data/HumanEval.jsonl.gz'
$SIZE" <<'PY'
import runpy
import sys

import configs.experiment_config as c

exec(sys.argv[1], vars(c))
sys.argv = ["experiments/exp20_breadth.py"]
runpy.run_path("experiments/exp20_breadth.py", run_name="__main__")
PY
  done
done
echo; echo "Done ($(date '+%F %T'))."
