#!/usr/bin/env bash
# Exp23: the public long-prompt task, HotpotQA (distractor setting: 10 Wikipedia paragraphs
# per question in the prompt), on Dream with both backends, with the variants of the main
# table and the Fast-dLLM table (and the plain INT8 / INT4 baselines of Exp19):
#   dLLM-Cache backend (Exp11 script): S0 S1 S2 S3 S4, C1 C2 C4 C5 C6, A1
#   Fast-dLLM backend  (Exp13 script): F0 F1 K1, K2 (gamma 0.9 and 0.7), K3 K4 K5
# 30 questions, 128 new tokens (as the authored long prompts), split after layer 4.
# Downloads the dev set once from Hugging Face (hotpotqa/hotpot_qa) and keeps only the
# first 100 questions (data/hotpotqa/dev_distractor_first100.jsonl).
# Runs in the background and waits until both GPUs are idle (checked every 5 min):
#   nohup bash experiments/run_exp23.sh > exp23.log 2>&1 &
#   SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp23.sh > exp23_smoke.log 2>&1   # ~10 min
#   GPUS=1,2 ...   (first = cloud, second = edge)
# Re-running resumes (finished runs are kept).
# Outputs: outputs/exp11_hotpotqa_k4_t128_T0.1[_smoke]/, outputs/exp13_dream_hotpotqa_k4_t128_b32[_smoke]/
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPUS="${GPUS:-1,2}"
export CUDA_VISIBLE_DEVICES="$GPUS"
SMOKE="${SMOKE:-0}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

DATA=data/hotpotqa/dev_distractor_first100.jsonl
if [ ! -s "$DATA" ]; then
  echo "$(date '+%F %T') downloading the HotpotQA dev set (distractor) from Hugging Face ..."
  python -c "from src.hotpotqa import prepare_from_hub; print(prepare_from_hub('$DATA', 100), 'questions kept')"
fi
python - <<'PY' || echo "(prompt-length check skipped)"
import statistics, transformers
from src.hotpotqa import load_hotpotqa
tok = transformers.AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
n = [len(tok.apply_chat_template([{"role": "user", "content": c["prompt"]}], add_generation_prompt=True))
     for c in load_hotpotqa("data/hotpotqa/dev_distractor_first100.jsonl", n=30)]
print(f"HotpotQA prompts (first 30): mean {statistics.mean(n):.0f} tokens, min {min(n)}, max {max(n)}", flush=True)
PY

if [ "${SKIP_IDLE_CHECK:-0}" != "1" ]; then
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

nvidia-smi pmon -s um -d 5 -o T > "gpu_pmon_exp23$([ "$SMOKE" = "1" ] && echo _smoke).log" 2>&1 &
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
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
PY
}

if [ "$SMOKE" = "1" ]; then TAG="_smoke"; N=2; T=64; else TAG=""; N=30; T=128; fi

py "1/2 dLLM-Cache backend, Dream, HotpotQA" experiments/exp11_dasr_split.py "
EXP11_MODEL = 'dream'
EXP11_DATASET = 'hotpotqa'
EXP11_N_QUESTIONS = $N
EXP11_NEW_TOKENS = $T
EXP11_STEPS = $T
EXP11_SPLIT_LAYER = 4
EXP11_EDGE_DEVICE = 'cuda:1'
EXP11_CLOUD_DEVICE = 'cuda:0'
EXP11_CACHE_POLICY = ('fixed', 25, 2, 0.25)
EXP11_TEMPERATURE = 0.1
EXP11_TOP_P = 0.9
EXP11_ANCHOR_INTERVAL = 32
EXP11_IDLE_SECONDS = 5.0
EXP11_RESUME = True
EXP11_OVERWRITE = False
EXP11_OUTPUT_TAG = '$TAG'
EXP11_VARIANT_NAMES = ['S0_split_full', 'S1_split_exact_delta', 'S2_split_int8',
                       'S3_split_int8abs', 'S4_split_int4', 'C1_cache_full', 'C2_cache_aware',
                       'C4_cache_aware_int8', 'C5_cache_aware_int8abs', 'C6_cache_aware_int4',
                       'A1_adaptive_cache_aware']"
python experiments/exp11b_network_emulation.py "outputs/exp11_hotpotqa_k4_t${T}_T0.1${TAG}" > /dev/null \
  && echo "emulated latency written for outputs/exp11_hotpotqa_k4_t${T}_T0.1${TAG}"

py "2/2 Fast-dLLM backend, Dream, HotpotQA" experiments/exp13_fastdllm_split.py "
EXP13_MODEL = 'dream'
EXP13_DATASET = 'hotpotqa'
EXP13_N_QUESTIONS = $N
EXP13_NEW_TOKENS = $T
EXP13_BLOCK_LENGTH = 32
EXP13_THRESHOLDS = [0.9, 0.7]
EXP13_SPLIT_LAYER = 4
EXP13_EDGE_DEVICE = 'cuda:1'
EXP13_CLOUD_DEVICE = 'cuda:0'
EXP13_ANCHOR_INTERVAL = 32
EXP13_RESUME = True
EXP13_OUTPUT_TAG = '$TAG'
EXP13_VARIANT_NAMES = ['F0_nocache_fixed', 'F1_nocache_thr0.9', 'K1_dual_fixed',
                       'K2_dual_thr0.9', 'K2_dual_thr0.7', 'K3_dual_thr0.9_int8',
                       'K4_dual_thr0.9_int8abs', 'K5_dual_thr0.9_int4']"
echo; echo "Exp23 completed ($(date '+%F %T'))."
