#!/usr/bin/env bash
# Exp16: the split over a real shaped link (relay processes) vs the latency model.
# Two workloads, Dream on GSM8K, split after layer 4:
#   1. dLLM-Cache backend: static split (bf16), DASR + dLLM-Cache (bf16 / INT8)
#   2. Fast-dLLM backend:  DASR + Fast-dLLM (bf16 / INT8)
# Links (bandwidth Mbit/s, RTT ms, questions): 1000/20/10, 100/20/10, 10/20/(2 | 10).
# Each question also runs once without network ("local") as the reference.
# Run from dasr1/:
#   SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp16.sh > exp16_smoke.log 2>&1   # ~10 min
#   nohup bash experiments/run_exp16.sh > exp16.log 2>&1 &                           # ~2 h, idle GPUs
#   GPUS=0,1 ...   other GPUs (first = cloud, second = edge)
# Timing matters: the GPUs must be idle (checked below, logged by nvidia-smi pmon).
# The relays are CPU processes on the same server; no sudo needed.
# Outputs: outputs/exp16_{dllm_cache,fastdllm}_dream_gsm8k_k4_t<tokens>[_smoke]/
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

nvidia-smi pmon -s um -d 5 -o T > "gpu_pmon_exp16$([ "$SMOKE" = "1" ] && echo _smoke).log" 2>&1 &
PMON_PID=$!
trap 'kill $PMON_PID 2>/dev/null || true' EXIT

run() {  # $1 = label, $2 = Python assignments that override config values
  echo; echo "=== $1  ($(date '+%F %T'))"
  python - "$2" <<'PY'
import runpy
import sys

import configs.experiment_config as c

exec(sys.argv[1], vars(c))
print({k: getattr(c, k) for k in ("EXP16_BACKEND", "EXP16_NEW_TOKENS", "EXP16_LINKS",
                                  "EXP16_VARIANT_NAMES", "EXP16_OUTPUT_TAG")}, flush=True)
sys.argv = ["experiments/exp16_real_network.py"]
runpy.run_path("experiments/exp16_real_network.py", run_name="__main__")
PY
}

COMMON='
EXP16_DATASET = "gsm8k"
EXP16_SPLIT_LAYER = 4
EXP16_EDGE_DEVICE = "cuda:1"
EXP16_CLOUD_DEVICE = "cuda:0"
EXP16_ANCHOR_INTERVAL = 32
EXP16_EDGE_SLOWDOWN = 1.0
EXP16_TCPIP_HEADER_BYTES = 40
EXP16_RESUME = True
'

if [ "$SMOKE" = "1" ]; then
  SIZE='EXP16_NEW_TOKENS = 64
EXP16_STEPS = 64
EXP16_OUTPUT_TAG = "_smoke"'
  LINKS_C='EXP16_LINKS = [(1000, 20, 2), (100, 20, 1)]'
  LINKS_F="$LINKS_C"
else
  SIZE='EXP16_NEW_TOKENS = 256
EXP16_STEPS = 256
EXP16_OUTPUT_TAG = ""'
  LINKS_C='EXP16_LINKS = [(1000, 20, 10), (100, 20, 10), (10, 20, 2)]'
  LINKS_F='EXP16_LINKS = [(1000, 20, 10), (100, 20, 10), (10, 20, 10)]'
fi

run "1/2 dLLM-Cache backend" "$COMMON
$SIZE
$LINKS_C
EXP16_BACKEND = \"dllm_cache\"
EXP16_VARIANT_NAMES = [\"S0_split_full\", \"C2_cache_aware\", \"C4_cache_aware_int8\"]"

run "2/2 Fast-dLLM backend" "$COMMON
$SIZE
$LINKS_F
EXP16_BACKEND = \"fastdllm\"
EXP16_THRESHOLD = 0.9
EXP16_BLOCK_LENGTH = 32
EXP16_VARIANT_NAMES = [\"K2_dual_thr0.9\", \"K3_dual_thr0.9_int8\"]"

echo; echo "Done ($(date '+%F %T'))."
