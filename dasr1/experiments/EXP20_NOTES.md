# Exp20: breadth (more problems, seeds, and HumanEval)

## Question
Do the quality results hold beyond 30 GSM8K problems and one seed, and on a second kind of
task?

## Set-up
- **Tasks:**
  - GSM8K: first 100 test problems, seeds 42, 43 and 44 for dLLM-Cache's sampling.
  - HumanEval: all 164 problems, pass@1 with one sample, the code executed against the unit
    tests (`src/humaneval.py`).
- **One GPU.** bf16 rows are exact at every split layer (Exp11, Exp13C, Exp17), so answers
  do not need the second GPU. The split hooks stay attached with edge = cloud, which applies
  the row rule and the INT8 codec on one GPU. No timing is measured, so the GPU may be
  shared with other jobs; it needs about 20 GB of free memory.
- **Configurations:**
  - dLLM-Cache backend: uncached (reference), DASR + dLLM-Cache (bf16), (INT8), adaptive
    refresh.
  - Fast-dLLM backend: one token per step (reference), DASR + Fast-dLLM (bf16), (INT8),
    γ = 0.7.
- 256 tokens for both tasks. Some HumanEval answers may be cut off, which affects every
  configuration alike.

## Run (each backend and task in its own folder, so jobs can run in parallel)
```
nohup bash experiments/run_exp20.sh > exp20.log 2>&1 &                         # everything on GPU 1, ~10 h
GPU=2 BACKENDS=fastdllm nohup bash experiments/run_exp20.sh > exp20_f.log 2>&1 &   # e.g. split across GPUs
SMOKE=1 bash experiments/run_exp20.sh > exp20_smoke.log 2>&1
```
Outputs: `outputs/exp20_breadth_<backend>_dream_<task>/` (`runs.csv`, `summary.csv`).

`summary.csv` reports, per configuration:
- accuracy (mean, and min/max over seeds);
- answers lost and gained against the reference, pooled over seeds;
- the McNemar p;
- answers identical to the reference.
