#!/usr/bin/env bash
# Exp18: choosing the codec (and Fast-dLLM's gamma) while the bandwidth changes, over a real
# TCP link through the network emulator replaying bandwidth traces. Dream, GSM8K, split
# after layer 4, RTT 20 ms, edge 3x slower.
#   1. Fast-dLLM backend:  step trace (100 <-> 10 Mbit/s), our phone's uplink trace (RTT 9.5 ms),
#                          T-Mobile and Verizon LTE uplink traces (Mahimahi), 10 questions each, 5 policies
#   2. dLLM-Cache backend: step trace and phone trace, 5 questions, 4 policies (the Mahimahi
#                          LTE traces are too slow for it: one answer would take 5-10 minutes)
# Starts by itself once both GPUs are idle (checks every 5 min), so start it and log out:
#   nohup bash experiments/run_exp18.sh > exp18.log 2>&1 &                            # ~3 h after start
#   SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp18.sh > exp18_smoke.log 2>&1    # ~10 min
#   GPUS=0,1 ...   other GPUs (first = cloud, second = edge)
# Needs the Mahimahi traces: cloned to ../../mahimahi (next to split-dllm) if missing.
# Outputs: outputs/exp18_controller_{fastdllm,dllm_cache}_dream_gsm8k_t<tokens>[_smoke]/
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPUS="${GPUS:-1,2}"
export CUDA_VISIBLE_DEVICES="$GPUS"      # cuda:0 = first GPU (cloud), cuda:1 = second GPU (edge)
echo "cloud = GPU ${GPUS%,*}, edge = GPU ${GPUS#*,}"
SMOKE="${SMOKE:-0}"
SKIP_IDLE_CHECK="${SKIP_IDLE_CHECK:-0}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

if [ ! -f ../../mahimahi/traces/TMobile-LTE-short.up ]; then
  echo "cloning the Mahimahi traces to ../../mahimahi"
  git clone --depth 1 https://github.com/ravinet/mahimahi.git ../../mahimahi
fi

if [ "$SKIP_IDLE_CHECK" != "1" ]; then   # wait until every GPU in $GPUS is idle
  while true; do
    busy=""
    for gpu in ${GPUS//,/ }; do
      max_util=0
      for _ in 1 2 3 4 5; do
        u=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
        [ "$u" -gt "$max_util" ] && max_util=$u
        sleep 1
      done
      [ "$max_util" -gt 10 ] && busy="$busy GPU $gpu (${max_util}%)"
    done
    if [ -z "$busy" ]; then echo "$(date '+%F %T') GPUs $GPUS idle, starting."; break; fi
    echo "$(date '+%F %T') busy:$busy; checking again in ${WAIT_SECONDS} s ..."
    sleep "$WAIT_SECONDS"
  done
fi

nvidia-smi pmon -s um -d 5 -o T > "gpu_pmon_exp18$([ "$SMOKE" = "1" ] && echo _smoke).log" 2>&1 &
PMON_PID=$!
trap 'kill $PMON_PID 2>/dev/null || true' EXIT

run() {  # $1 = label, $2 = Python assignments that override config values
  echo; echo "=== $1  ($(date '+%F %T'))"
  python - "$2" <<'PY'
import runpy
import sys

import configs.experiment_config as c

exec(sys.argv[1], vars(c))
print({k: getattr(c, k) for k in ("EXP18_BACKEND", "EXP18_NEW_TOKENS", "EXP18_TRACES",
                                  "EXP18_POLICY_NAMES", "EXP18_OUTPUT_TAG")}, flush=True)
sys.argv = ["experiments/exp18_live_controller.py"]
runpy.run_path("experiments/exp18_live_controller.py", run_name="__main__")
PY
}

COMMON='
EXP18_DATASET = "gsm8k"
EXP18_SPLIT_LAYER = 4
EXP18_EDGE_DEVICE = "cuda:1"
EXP18_CLOUD_DEVICE = "cuda:0"
EXP18_ANCHOR_INTERVAL = 32
EXP18_RTT_MS = 20
EXP18_EDGE_SLOWDOWN = 3.0
EXP18_LAMBDA = 0.10
EXP18_MAHIMAHI_DIR = "../../mahimahi/traces"
EXP18_RESUME = True
'

if [ "$SMOKE" = "1" ]; then
  SIZE='EXP18_NEW_TOKENS = 64
EXP18_STEPS = 64
EXP18_OUTPUT_TAG = "_smoke"'
  TR_F='EXP18_TRACES = [{"name": "step_100_10", "step": (100, 10, 20), "n": 2},
                {"name": "tmobile_lte_up", "mahimahi": "TMobile-LTE-short.up", "n": 1}]'
  TR_C='EXP18_TRACES = [{"name": "step_100_10", "step": (100, 10, 20), "n": 1}]'
else
  SIZE='EXP18_NEW_TOKENS = 256
EXP18_STEPS = 256
EXP18_OUTPUT_TAG = ""'
  TR_F='EXP18_TRACES = [{"name": "step_100_10", "step": (100, 10, 20), "n": 10},
                {"name": "phone_lte_lab", "csv": "outputs/phone_traces/lte_lab.csv", "rtt_ms": 9.5, "n": 10},
                {"name": "tmobile_lte_up", "mahimahi": "TMobile-LTE-short.up", "n": 10},
                {"name": "verizon_lte_up", "mahimahi": "Verizon-LTE-short.up", "n": 10}]'
  TR_C='EXP18_TRACES = [{"name": "step_100_10", "step": (100, 10, 20), "n": 5},
                {"name": "phone_lte_lab", "csv": "outputs/phone_traces/lte_lab.csv", "rtt_ms": 9.5, "n": 5}]'
fi

run "1/2 Fast-dLLM backend" "$COMMON
$SIZE
$TR_F
EXP18_BACKEND = \"fastdllm\"
EXP18_THRESHOLD = 0.9
EXP18_GAMMA_LOW = 0.7
EXP18_BLOCK_LENGTH = 32
EXP18_POLICY_NAMES = [\"bf16\", \"int8\", \"codec_per_request\", \"codec_per_step\", \"codec_gamma_per_step\"]"

run "2/2 dLLM-Cache backend" "$COMMON
$SIZE
$TR_C
EXP18_BACKEND = \"dllm_cache\"
EXP18_POLICY_NAMES = [\"bf16\", \"int8\", \"codec_per_request\", \"codec_per_step\"]"

echo; echo "Done ($(date '+%F %T'))."
