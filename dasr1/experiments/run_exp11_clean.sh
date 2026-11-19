#!/usr/bin/env bash
# Clean rerun of the three Exp11 runs that shared the edge GPU with another job:
#   1. Dream add-on, GSM8K        -> outputs/exp11_gsm8k_k4_t256_T0.1_addon_clean/
#   2. Dream add-on, long prompts -> outputs/exp11_authored_k4_t128_T0.1_addon_clean/
#   3. LLaDA split, GSM8K         -> outputs/exp11_llada_gsm8k_k4_t256_T0_clean/
# Run from dasr1/:   nohup bash run_exp11_clean.sh > exp11_clean.log 2>&1 &
# Nothing in configs/experiment_config.py needs editing: each run overrides the
# EXP11_* values it needs in memory. Existing output folders are not touched.
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2          # cuda:0 = GPU 1 (cloud), cuda:1 = GPU 2 (edge)

# Other users' processes may sit on GPUs 1/2 holding memory; that is harmless as long
# as they stay idle. Stop only if a GPU is actually busy (utilization > 10%) now.
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

# Record per-process GPU activity every 5 s during the whole run, so we can check
# afterwards whether another process computed on GPU 1/2 while we measured.
nvidia-smi pmon -s um -d 5 > gpu_pmon_exp11_clean.log 2>&1 &
PMON_PID=$!
trap 'kill $PMON_PID 2>/dev/null || true' EXIT

run() {  # $1 = label, $2 = Python assignments that override config values
  echo; echo "=== $1  ($(date '+%F %T'))"
  python - "$2" <<'PY'
import runpy
import sys

import configs.experiment_config as c

exec(sys.argv[1], vars(c))  # override EXP11_* in the already-loaded config module
print({k: getattr(c, k) for k in ("EXP11_MODEL", "EXP11_DATASET", "EXP11_NEW_TOKENS",
                                  "EXP11_OUTPUT_TAG")}, flush=True)
sys.argv = ["experiments/exp11_dasr_split.py"]
runpy.run_path("experiments/exp11_dasr_split.py", run_name="__main__")
PY
}

COMMON='
EXP11_N_QUESTIONS = 30
EXP11_SPLIT_LAYER = 4
EXP11_EDGE_DEVICE = "cuda:1"
EXP11_CLOUD_DEVICE = "cuda:0"
EXP11_ANCHOR_INTERVAL = 32
EXP11_IDLE_SECONDS = 5.0
EXP11_RESUME = True
EXP11_OVERWRITE = False
'
DREAM='
EXP11_MODEL = "dream"
EXP11_CACHE_POLICY = ("fixed", 25, 2, 0.25)
EXP11_TEMPERATURE = 0.1
EXP11_TOP_P = 0.9
EXP11_OUTPUT_TAG = "_addon_clean"
EXP11_VARIANT_NAMES = ["S0_split_full", "S1_split_exact_delta", "S2_split_int8",
                       "C2_cache_aware", "C4_cache_aware_int8", "A1_adaptive_cache_aware"]
'
LLADA='
EXP11_MODEL = "llada"
EXP11_LLADA_CACHE_POLICY = ("fixed", 50, 7, 0.25)
EXP11_LLADA_TEMPERATURE = 0.0
EXP12_BLOCK_LENGTH = 8
EXP11_OUTPUT_TAG = "_clean"
EXP11_VARIANT_NAMES = ["S0_split_full", "S1_split_exact_delta", "S2_split_int8", "C1_cache_full",
                       "C2_cache_aware", "C3_cache_aware_exact", "C4_cache_aware_int8",
                       "A1_adaptive_cache_aware"]
'

run "1/3 Dream add-on, GSM8K" "$COMMON$DREAM
EXP11_DATASET = \"gsm8k\"
EXP11_NEW_TOKENS = 256
EXP11_STEPS = 256
"
run "2/3 Dream add-on, long prompts" "$COMMON$DREAM
EXP11_DATASET = \"authored\"
EXP11_NEW_TOKENS = 128
EXP11_STEPS = 128
"
run "3/3 LLaDA split, GSM8K" "$COMMON$LLADA
EXP11_DATASET = \"gsm8k\"
EXP11_NEW_TOKENS = 256
EXP11_STEPS = 256
"

echo; echo "=== network emulation for the LLaDA run"
python experiments/exp11b_network_emulation.py outputs/exp11_llada_gsm8k_k4_t256_T0_clean
echo; echo "Done ($(date '+%F %T'))."