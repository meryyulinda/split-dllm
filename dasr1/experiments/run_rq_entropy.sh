#!/usr/bin/env bash
# Rerun the RQ1-RQ5 Dream experiments under Dream's official sampler
# (entropy, T=0.1, top_p=0.9). Run from dasr1/:   bash run_rq_entropy.sh
#
# 1. In configs/experiment_config.py set RERUN_ENTROPY = True (block at the end).
# 2. Pick free GPUs below. Single-GPU scripts use SINGLE_GPU. Exp4A/4B/5A/5B use the two
#    GPUs named by EDGE_DEVICE / CLOUD_DEVICE in the config, numbered as seen
#    through TWO_GPUS (with TWO_GPUS="1,2": cuda:0 = physical 1, cuda:1 = physical 2).
# 3. Afterwards set RERUN_ENTROPY = False again.
# Results go to outputs/exp*_entropy/; the original origin results are untouched.
set -euo pipefail

SINGLE_GPU="${SINGLE_GPU:-1}"
TWO_GPUS="${TWO_GPUS:-1,2}"

TWO_GPUS="$TWO_GPUS" python - <<'PY'
import os, sys
sys.path.insert(0, ".")
import configs.experiment_config as c
if not c.OUTPUT_SUFFIX or c.ALGORITHM != "entropy":
    sys.exit("Set RERUN_ENTROPY = True in configs/experiment_config.py first "
             "(otherwise the origin results in outputs/exp* would be overwritten).")
n_visible = len([g for g in os.environ["TWO_GPUS"].split(",") if g.strip()])
for name in ("EDGE_DEVICE", "CLOUD_DEVICE"):
    idx = int(str(getattr(c, name)).split(":")[-1])
    if idx >= n_visible:
        sys.exit(f"{name} = {getattr(c, name)!r}, but TWO_GPUS={os.environ['TWO_GPUS']} only "
                 f"exposes cuda:0..cuda:{n_visible - 1}. Set {name} accordingly in the config.")
print(f"sampler={c.ALGORITHM} T={c.TEMPERATURE} top_p={c.TOP_P} -> outputs/exp*{c.OUTPUT_SUFFIX}")
print(f"Exp4A/4B/5A/5B devices: edge={c.EDGE_DEVICE} cloud={c.CLOUD_DEVICE} (within TWO_GPUS)")
PY

run1() { echo; echo "=== $1"; CUDA_VISIBLE_DEVICES="$SINGLE_GPU" python "experiments/$1.py"; }
run2() { echo; echo "=== $1"; CUDA_VISIBLE_DEVICES="$TWO_GPUS" python "experiments/$1.py"; }
cpu()  { echo; echo "=== $1"; python "experiments/$1.py"; }

# RQ1: token changes and hidden-state drift
run1 exp1_token_changes
cpu  exp1_plot
run1 exp2_hidden_drift
cpu  exp2c_token_vs_hidden
# RQ2: two-GPU static split and exact deltas
run2 exp4a_two_gpu_static_split
run2 exp4b_synchronized_delta
# RQ3: row-level selective reuse (motivation for feature caching)
run2 exp5a_selective_reuse
run2 exp5b_selective_compute_estimate
# RQ4: adaptive split scheduling (analytical, from Exp5B tables)
cpu  exp6_adaptive_scheduler
# RQ5: Dream cross-task drift (LLaDA's Exp7C already uses its confidence sampler)
run1 exp7d_dream_cross_task_hidden_drift

echo; echo "Done. Now set RERUN_ENTROPY = False in configs/experiment_config.py."