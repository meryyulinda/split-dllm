#!/usr/bin/env bash
# Exp13: Fast-dLLM (DualCache + parallel decoding) on the two-GPU split, three workloads
# back to back on otherwise idle GPUs:
#   1. Dream, GSM8K (30)        2. Dream, long prompts (12)        3. LLaDA, GSM8K (30)
# Run from dasr1/:
#   SMOKE=1 bash experiments/run_exp13.sh                       # 2 questions, 64 tokens (~5 min)
#   nohup bash experiments/run_exp13.sh > exp13.log 2>&1 &      # full run
# Nothing in configs/experiment_config.py needs editing; FASTDLLM_REPO must point to a
# Fast-dLLM checkout (default ../../Fast-dLLM). Per-process GPU activity is logged.
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2          # cuda:0 = GPU 1 (cloud), cuda:1 = GPU 2 (edge)
SMOKE="${SMOKE:-0}"

for gpu in 1 2; do
  max_util=0
  for _ in 1 2 3 4 5; do
    u=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    [ "$u" -gt "$max_util" ] && max_util=$u
    sleep 1
  done
  echo "GPU $gpu: max utilization over 5 s = ${max_util}%"
  if [ "$max_util" -gt 10 ]; then
    echo "GPU $gpu is busy right now; try again later."; nvidia-smi -i "$gpu"; exit 1
  fi
done

nvidia-smi pmon -s um -d 5 > gpu_pmon_exp13.log 2>&1 &
PMON_PID=$!
trap 'kill $PMON_PID 2>/dev/null || true' EXIT

run() {  # $1 = label, $2 = Python assignments that override config values
  echo; echo "=== $1  ($(date '+%F %T'))"
  python - "$2" <<'PY'
import runpy
import sys

import configs.experiment_config as c

exec(sys.argv[1], vars(c))
print({k: getattr(c, k) for k in ("EXP13_MODEL", "EXP13_DATASET", "EXP13_N_QUESTIONS",
                                  "EXP13_NEW_TOKENS", "EXP13_OUTPUT_TAG")}, flush=True)
sys.argv = ["experiments/exp13_fastdllm_split.py"]
runpy.run_path("experiments/exp13_fastdllm_split.py", run_name="__main__")
PY
}

if [ "$SMOKE" = "1" ]; then
  SIZE_G='EXP13_N_QUESTIONS = 2
EXP13_NEW_TOKENS = 64
EXP13_OUTPUT_TAG = "_smoke"'
  SIZE_A="$SIZE_G"
else
  SIZE_G='EXP13_N_QUESTIONS = 30
EXP13_NEW_TOKENS = 256
EXP13_OUTPUT_TAG = ""'
  SIZE_A='EXP13_N_QUESTIONS = 12
EXP13_NEW_TOKENS = 128
EXP13_OUTPUT_TAG = ""'
fi

COMMON='
EXP13_BLOCK_LENGTH = 32
EXP13_THRESHOLDS = [0.9, 0.8, 0.7]
EXP13_SPLIT_LAYER = 4
EXP13_EDGE_DEVICE = "cuda:1"
EXP13_CLOUD_DEVICE = "cuda:0"
EXP13_ANCHOR_INTERVAL = 32
EXP13_RESUME = True
'

run "1/3 Dream, GSM8K" "$COMMON
EXP13_MODEL = \"dream\"
EXP13_DATASET = \"gsm8k\"
$SIZE_G"
run "2/3 Dream, long prompts" "$COMMON
EXP13_MODEL = \"dream\"
EXP13_DATASET = \"authored\"
$SIZE_A"
run "3/3 LLaDA, GSM8K" "$COMMON
EXP13_MODEL = \"llada\"
EXP13_DATASET = \"gsm8k\"
$SIZE_G"

echo; echo "Done ($(date '+%F %T'))."
