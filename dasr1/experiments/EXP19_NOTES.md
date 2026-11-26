# Exp19: stronger baselines (plain quantization of the boundary rows)

## Question
Reviewers will ask how DASR compares with the activation compression that split-inference
systems already use, not only with our own static split. The common baseline quantizes the
boundary activations themselves at every step, with no delta and no reference.

## Codecs (`src/split_runtime.py`, `quantize_abs`)
| Codec | What it does | Bytes per row (d = 3584) |
|---|---|---|
| `int8_abs` | per-row absmax INT8 with an fp16 scale | d + 2 = 3586 (about half of bf16) |
| `int4_group` | absmax INT4 in groups of 128 values, with fp16 scales | d/2 + 2·d/128 = 1848 (about a quarter) |

Both are applied:
- to the static split (S3, S4);
- to DASR's rows: dLLM-Cache backend (C5, C6) and Fast-dLLM backend (K4, K5).

They are compared with DASR's INT8 delta (S2, C4, K3, from the existing outputs) and bf16.

## Run
```
nohup bash experiments/run_exp19.sh > exp19.log 2>&1 &        # waits for idle GPUs, ~2.5 h
SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp19.sh > exp19_smoke.log 2>&1
```
- Runs the Exp11 and Exp13 scripts with `_baselines` output folders, on Dream GSM8K (30),
  Dream long prompts (12) and LLaDA GSM8K (30).
- Then adds the emulated latency for the Exp11 folders.

## What to read
- `summary_by_variant.csv`: accuracy, regressions vs S0 / F0, uplink MB.
- `emulated_summary.csv`: latency.
- Exp13's `same_as_cache_ref`: answers identical to DASR + Fast-dLLM (bf16).

The question is whether plain INT4 reaches INT8-delta's bytes or lower without losing
answers, and what INT8 without delta costs in accuracy compared with INT8 delta.
