#!/usr/bin/env bash
# Exp14: block-start row reuse on top of Fast-dLLM (DualCache + parallel decoding, gamma 0.9),
# same three workloads as Exp13:
#   1. Dream, GSM8K (30)        2. Dream, long prompts (12)        3. LLaDA, GSM8K (30)
# Run from dasr1/:
#   SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp14.sh > exp14_smoke.log 2>&1   # 2 questions, 64 tokens
#   nohup bash experiments/run_exp14.sh > exp14.log 2>&1 &                            # full run, idle GPUs
# Outputs: outputs/exp13_<model>_<dataset>_k4_t<tokens>_b32_reuse[_smoke]/
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1          # cuda:0 = GPU 1 (cloud), cuda:1 = GPU 2 (edge)
SMOKE="${SMOKE:-0}"
SKIP_IDLE_CHECK="${SKIP_IDLE_CHECK:-0}"

if [ "$SKIP_IDLE_CHECK" != "1" ]; then
  for gpu in 0 1; do
    max_util=0
    for _ in 1 2 3 4 5; do
      u=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
      [ "$u" -gt "$max_util" ] && max_util=$u
      sleep 1
    done
    echo "GPU $gpu: max utilization over 5 s = ${max_util}%"
    if [ "$max_util" -gt 10 ]; then
      echo "GPU $gpu is busy right now; try again later (or SKIP_IDLE_CHECK=1 for a smoke test)."
      nvidia-smi -i "$gpu"; exit 1
    fi
  done
fi

nvidia-smi pmon -s um -d 5 > "gpu_pmon_exp14$([ "$SMOKE" = "1" ] && echo _smoke).log" 2>&1 &
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
                                  "EXP13_NEW_TOKENS", "EXP13_OUTPUT_TAG",
                                  "EXP13_VARIANT_NAMES")}, flush=True)
sys.argv = ["experiments/exp13_fastdllm_split.py"]
runpy.run_path("experiments/exp13_fastdllm_split.py", run_name="__main__")
PY
}

if [ "$SMOKE" = "1" ]; then
  SIZE_G='EXP13_N_QUESTIONS = 2
EXP13_NEW_TOKENS = 64
EXP13_OUTPUT_TAG = "_reuse_smoke"'
  SIZE_A="$SIZE_G"
else
  SIZE_G='EXP13_N_QUESTIONS = 30
EXP13_NEW_TOKENS = 256
EXP13_OUTPUT_TAG = "_reuse"'
  SIZE_A='EXP13_N_QUESTIONS = 12
EXP13_NEW_TOKENS = 128
EXP13_OUTPUT_TAG = "_reuse"'
fi

# F0 is the quality reference the script requires; K2/K3 are the Exp13 baselines
# (K2 = cache_ref, run before the R* variants in the same process).
COMMON='
EXP13_BLOCK_LENGTH = 32
EXP13_THRESHOLDS = [0.9]
EXP13_SPLIT_LAYER = 4
EXP13_EDGE_DEVICE = "cuda:1"
EXP13_CLOUD_DEVICE = "cuda:0"
EXP13_ANCHOR_INTERVAL = 32
EXP13_RESUME = True
EXP14_REFRESH_EVERY = 4
EXP13_VARIANT_NAMES = ["F0_nocache_fixed", "K2_dual_thr0.9", "K3_dual_thr0.9_int8",
                       "R1_dual_thr0.9_reuse", "R2_dual_thr0.9_reuse_r4",
                       "R3_dual_thr0.9_reuse_int8"]
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
