#!/usr/bin/env bash
# Exp19: stronger baselines -- plain quantization of the boundary rows, as split-inference
# systems commonly use (per-row INT8 and group-wise INT4, no delta, no anchors), on the
# static split and on DASR's rows, for both backends. Same workloads as Exp11 / Exp13:
#   dLLM-Cache backend (Exp11 script): Dream GSM8K (30), Dream long prompts (12), LLaDA GSM8K (30)
#   Fast-dLLM backend (Exp13 script):  the same three workloads
# Only the new variants plus each script's reference (S0 / F0) and K2 run; the INT8-delta
# variants (S2, C2, C4, K3) are deterministic and come from the existing outputs.
# Starts by itself once both GPUs are idle (checks every 5 min), so start it and log out:
#   nohup bash experiments/run_exp19.sh > exp19.log 2>&1 &                            # ~2.5 h after start
#   SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp19.sh > exp19_smoke.log 2>&1    # ~10 min
#   GPUS=0,1 ...   other GPUs (first = cloud, second = edge)
# Outputs: outputs/exp11_*_baselines[_smoke]/ and outputs/exp13_*_baselines[_smoke]/
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPUS="${GPUS:-1,2}"
export CUDA_VISIBLE_DEVICES="$GPUS"      # cuda:0 = first GPU (cloud), cuda:1 = second GPU (edge)
echo "cloud = GPU ${GPUS%,*}, edge = GPU ${GPUS#*,}"
SMOKE="${SMOKE:-0}"
SKIP_IDLE_CHECK="${SKIP_IDLE_CHECK:-0}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

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

nvidia-smi pmon -s um -d 5 -o T > "gpu_pmon_exp19$([ "$SMOKE" = "1" ] && echo _smoke).log" 2>&1 &
PMON_PID=$!
trap 'kill $PMON_PID 2>/dev/null || true' EXIT

py() {  # $1 = label, $2 = script, $3 = Python assignments that override config values
  echo; echo "=== $1  ($(date '+%F %T'))"
  python - "$2" "$3" <<'PY'
import runpy
import sys

import configs.experiment_config as c

script, overrides = sys.argv[1], sys.argv[2]
exec(overrides, vars(c))
print({k: v for k, v in vars(c).items() if k.startswith(("EXP11_", "EXP13_"))
       and k.split("_", 1)[1] in {"MODEL", "DATASET", "N_QUESTIONS", "NEW_TOKENS", "STEPS",
                                  "VARIANT_NAMES", "OUTPUT_TAG"}}, flush=True)
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
PY
}

if [ "$SMOKE" = "1" ]; then
  TAG="_baselines_smoke"; NG=2; NA=2; TG=64; TA=64
else
  TAG="_baselines"; NG=30; NA=12; TG=256; TA=128
fi

E11='
EXP11_SPLIT_LAYER = 4
EXP11_EDGE_DEVICE = "cuda:1"
EXP11_CLOUD_DEVICE = "cuda:0"
EXP11_ANCHOR_INTERVAL = 32
EXP11_RESUME = True
EXP11_OVERWRITE = False
EXP11_OUTPUT_TAG = "'$TAG'"
EXP11_VARIANT_NAMES = ["S0_split_full", "S3_split_int8abs", "S4_split_int4",
                       "C5_cache_aware_int8abs", "C6_cache_aware_int4"]
'
E13='
EXP13_BLOCK_LENGTH = 32
EXP13_THRESHOLDS = [0.9]
EXP13_SPLIT_LAYER = 4
EXP13_EDGE_DEVICE = "cuda:1"
EXP13_CLOUD_DEVICE = "cuda:0"
EXP13_ANCHOR_INTERVAL = 32
EXP13_RESUME = True
EXP13_OUTPUT_TAG = "'$TAG'"
EXP13_VARIANT_NAMES = ["F0_nocache_fixed", "K2_dual_thr0.9", "K4_dual_thr0.9_int8abs",
                       "K5_dual_thr0.9_int4"]
'

py "1/6 dLLM-Cache, Dream GSM8K" experiments/exp11_dasr_split.py "$E11
EXP11_MODEL = 'dream'
EXP11_DATASET = 'gsm8k'
EXP11_N_QUESTIONS = $NG
EXP11_NEW_TOKENS = $TG
EXP11_STEPS = $TG"
py "2/6 dLLM-Cache, Dream long prompts" experiments/exp11_dasr_split.py "$E11
EXP11_MODEL = 'dream'
EXP11_DATASET = 'authored'
EXP11_N_QUESTIONS = $NA
EXP11_NEW_TOKENS = $TA
EXP11_STEPS = $TA"
py "3/6 dLLM-Cache, LLaDA GSM8K" experiments/exp11_dasr_split.py "$E11
EXP11_MODEL = 'llada'
EXP11_DATASET = 'gsm8k'
EXP11_N_QUESTIONS = $NG
EXP11_NEW_TOKENS = $TG
EXP11_STEPS = $TG"
py "4/6 Fast-dLLM, Dream GSM8K" experiments/exp13_fastdllm_split.py "$E13
EXP13_MODEL = 'dream'
EXP13_DATASET = 'gsm8k'
EXP13_N_QUESTIONS = $NG
EXP13_NEW_TOKENS = $TG"
py "5/6 Fast-dLLM, Dream long prompts" experiments/exp13_fastdllm_split.py "$E13
EXP13_MODEL = 'dream'
EXP13_DATASET = 'authored'
EXP13_N_QUESTIONS = $NA
EXP13_NEW_TOKENS = $TA"
py "6/6 Fast-dLLM, LLaDA GSM8K" experiments/exp13_fastdllm_split.py "$E13
EXP13_MODEL = 'llada'
EXP13_DATASET = 'gsm8k'
EXP13_N_QUESTIONS = $NG
EXP13_NEW_TOKENS = $TG"

echo; echo "=== emulated latency for the Exp11 folders  ($(date '+%F %T'))"
for d in outputs/exp11_*"$TAG"; do
  python experiments/exp11b_network_emulation.py "$d" > /dev/null && echo "  $d"
done
echo; echo "Done ($(date '+%F %T'))."
