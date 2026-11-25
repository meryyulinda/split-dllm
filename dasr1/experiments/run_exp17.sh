#!/usr/bin/env bash
# Exp17: offloading -- cloud GPU time and energy per answer, split vs cloud-only,
# for split layers 2, 4, 8, 14, 20 (of 28). Dream on 10 GSM8K problems, no network.
#   1. dLLM-Cache backend: DASR + dLLM-Cache (bf16 / INT8) vs cloud-only + dLLM-Cache
#   2. Fast-dLLM backend:  DASR + Fast-dLLM (bf16 / INT8) vs cloud-only + Fast-dLLM
# Run from dasr1/:
#   SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp17.sh > exp17_smoke.log 2>&1   # ~5 min
#   nohup bash experiments/run_exp17.sh > exp17.log 2>&1 &                           # ~40 min, idle GPUs
#   GPUS=0,1 ...   other GPUs (first = cloud, second = edge)
# Timing and energy matter: the GPUs must be idle (checked below, logged by nvidia-smi pmon).
# Outputs: outputs/exp17_offloading_{dllm_cache,fastdllm}_dream_gsm8k_t<tokens>[_smoke]/
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPUS="${GPUS:-1,2}"
export CUDA_VISIBLE_DEVICES="$GPUS"      # cuda:0 = first GPU (cloud), cuda:1 = second GPU (edge)
echo "cloud = GPU ${GPUS%,*}, edge = GPU ${GPUS#*,}"
SMOKE="${SMOKE:-0}"
SKIP_IDLE_CHECK="${SKIP_IDLE_CHECK:-0}"

if [ "$SKIP_IDLE_CHECK" != "1" ]; then
  for gpu in ${GPUS//,/ }; do
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

nvidia-smi pmon -s um -d 5 -o T > "gpu_pmon_exp17$([ "$SMOKE" = "1" ] && echo _smoke).log" 2>&1 &
PMON_PID=$!
trap 'kill $PMON_PID 2>/dev/null || true' EXIT

run() {  # $1 = label, $2 = Python assignments that override config values
  echo; echo "=== $1  ($(date '+%F %T'))"
  python - "$2" <<'PY'
import runpy
import sys

import configs.experiment_config as c

exec(sys.argv[1], vars(c))
print({k: getattr(c, k) for k in ("EXP17_BACKEND", "EXP17_N_QUESTIONS", "EXP17_NEW_TOKENS",
                                  "EXP17_SPLIT_LAYERS", "EXP17_VARIANT_NAMES",
                                  "EXP17_OUTPUT_TAG")}, flush=True)
sys.argv = ["experiments/exp17_offloading.py"]
runpy.run_path("experiments/exp17_offloading.py", run_name="__main__")
PY
}

COMMON='
EXP17_DATASET = "gsm8k"
EXP17_EDGE_DEVICE = "cuda:1"
EXP17_CLOUD_DEVICE = "cuda:0"
EXP17_ANCHOR_INTERVAL = 32
EXP17_RTT_MS = 20
EXP17_BANDWIDTHS_MBPS = [10, 100, 1000]
EXP17_EDGE_SLOWDOWN = 3.0
EXP17_RESUME = True
'

if [ "$SMOKE" = "1" ]; then
  SIZE='EXP17_N_QUESTIONS = 2
EXP17_NEW_TOKENS = 64
EXP17_STEPS = 64
EXP17_SPLIT_LAYERS = [4, 14]
EXP17_OUTPUT_TAG = "_smoke"'
else
  SIZE='EXP17_N_QUESTIONS = 10
EXP17_NEW_TOKENS = 256
EXP17_STEPS = 256
EXP17_SPLIT_LAYERS = [2, 4, 8, 14, 20]
EXP17_OUTPUT_TAG = ""'
fi

run "1/2 dLLM-Cache backend" "$COMMON
$SIZE
EXP17_BACKEND = \"dllm_cache\"
EXP17_VARIANT_NAMES = [\"C2_cache_aware\", \"C4_cache_aware_int8\"]"

run "2/2 Fast-dLLM backend" "$COMMON
$SIZE
EXP17_BACKEND = \"fastdllm\"
EXP17_THRESHOLD = 0.9
EXP17_BLOCK_LENGTH = 32
EXP17_VARIANT_NAMES = [\"K2_dual_thr0.9\", \"K3_dual_thr0.9_int8\"]"

echo; echo "Done ($(date '+%F %T'))."
