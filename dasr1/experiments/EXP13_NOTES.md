# Exp13: Fast-dLLM on the two-GPU split

## What it measures

Fast-dLLM (NVlabs) adds two things to a dLLM:

* **DualCache (block-wise KV cache).** At the start of each block of 32 answer tokens the model runs once over the whole sequence and keeps every layer's keys/values; inside the block only the block's 32 rows are computed. For the split this means the edge sends **32 rows per step instead of the whole sequence** (the paper's row rule with the block as the recomputed set).
* **Parallel decoding.** Inside a block, every masked position with confidence ≥ γ is committed in the same step (at least one). Fewer steps = **fewer network round trips**, the floor that keeps any split behind cloud-only.

| Variant | Cache | Decoding | Boundary |
|---|---|---|---|
| `F0_nocache_fixed` | none | 1 token / step | all rows, bf16 (reference) |
| `F1_nocache_thr0.9` | none | parallel, γ = 0.9 | all rows, bf16 |
| `K1_dual_fixed` | DualCache | 1 token / step | block rows (all rows at block start) |
| `K2_dual_thr<γ>` | DualCache | parallel, γ ∈ {0.9, 0.8, 0.7} | block rows |
| `K3_dual_thr0.9_int8` | DualCache | parallel, γ = 0.9 | block rows, INT8 delta |

Quality is compared with F0 (paired flips, McNemar). Steps, bytes and per-side GPU time are measured per step; the emulation of Exp11b is applied at the end (`emulated_summary.csv`; `cloud_only` = F0 computation, `cloud_only_cache` = K2 with γ = 0.9).

## Setup

```bash
cd split-dllm/dasr1
git clone https://github.com/NVlabs/Fast-dLLM.git ../../Fast-dLLM
# configs/experiment_config.py:  FASTDLLM_REPO = "../../Fast-dLLM"
```

The code loads Fast-dLLM's own Dream / LLaDA model files (they add the cache arguments). Fast-dLLM pins `transformers==4.49.0`; try the `dream` environment (4.46.2) first. If loading fails with an import error, create a copy of the environment with `pip install transformers==4.49.0`.

## Run

```bash
SMOKE=1 bash experiments/run_exp13.sh                    # 2 questions, 64 tokens, ~5-10 min
nohup bash experiments/run_exp13.sh > exp13.log 2>&1 &   # full: Dream GSM8K, Dream long, LLaDA GSM8K
```

The script checks that GPUs 1 and 2 are idle, logs per-process GPU activity to `gpu_pmon_exp13.log`, and needs no config edits. Smoke results go to `outputs/exp13_*_smoke/` (delete afterwards).

## What to check in the smoke test

* every variant prints `ok=…, steps=…, up=… MB` without errors;
* `F0` takes 64 steps; `K1` also 64; `F1`/`K2` far fewer (parallel decoding);
* `K1`/`K2` send much less per step than `F0` (32 rows inside a block, all rows only at block starts);
* the answers of `K2_dual_thr0.9` look sensible (DualCache is approximate, so they need not equal F0).

## Outputs

`outputs/exp13_<model>_<dataset>_k4_t<tokens>_b32/`: `runs.csv`, `steps.csv` (one row per step: `kind` = full / block, rows, bytes, times), `summary_by_variant.csv`, `emulated_summary.csv`, `metadata.json`.
