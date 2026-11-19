#!/usr/bin/env bash
# Rerun the block-caching experiments Exp8F / Exp8G (RQ3 motivation) under Dream's
# official sampler (entropy, T=0.1, top_p=0.9), like run_rq_entropy.sh did for RQ1-RQ5.
# Run from dasr1/:   bash experiments/run_exp8_entropy.sh
#
# 1. In configs/experiment_config.py set RERUN_ENTROPY = True (block at the end).
# 2. Afterwards set RERUN_ENTROPY = False again.
# Results go to outputs/exp8f_entropy/ and outputs/exp8g_entropy/; the original
# origin-sampler folders outputs/exp8f, outputs/exp8g are untouched.
# Both scripts use one GPU.
set -euo pipefail

SINGLE_GPU="${SINGLE_GPU:-1}"

python - <<'PY'
import sys
sys.path.insert(0, ".")
import configs.experiment_config as c
if not c.OUTPUT_SUFFIX or c.ALGORITHM != "entropy":
    sys.exit("Set RERUN_ENTROPY = True in configs/experiment_config.py first "
             "(otherwise outputs/exp8f and outputs/exp8g would be overwritten).")
print(f"sampler={c.ALGORITHM} T={c.TEMPERATURE} top_p={c.TOP_P} "
      f"-> outputs/exp8f{c.OUTPUT_SUFFIX}, outputs/exp8g{c.OUTPUT_SUFFIX}")
PY

run1() { echo; echo "=== $1"; CUDA_VISIBLE_DEVICES="$SINGLE_GPU" python "experiments/$1.py"; }

run1 exp8f_block_cache_scaling       # latency of block-residual caching (timed B-C-B)
run1 exp8g_block_cache_robustness    # fidelity: 3 tasks x 3 seeds x 2 groups x 2 intervals

echo; echo "Done. Now set RERUN_ENTROPY = False in configs/experiment_config.py."
