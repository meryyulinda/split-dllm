# Exp8F/8G entropy rerun / Exp11c / Exp11 add-on / Exp12 (LLaDA): what to run

All commands from `dasr1/`. Nothing below touches the finished Exp11 folders.

## 0. Config block (paste at the END of `configs/experiment_config.py`)

```python
# ---- Exp11 add-on: compressed static split (S2) + peak memory + energy ----
EXP11_OUTPUT_TAG = "_addon"                   # -> outputs/exp11_..._addon/
EXP11_VARIANT_NAMES = ["S0_split_full", "S1_split_exact_delta", "S2_split_int8",
                       "C2_cache_aware", "C4_cache_aware_int8", "A1_adaptive_cache_aware"]
EXP11_IDLE_SECONDS = 5.0                      # idle power measured once, for *_active_j
# EXP11_OVERWRITE = False                     # never set True unless you mean it

# ---- LLaDA (RQ5) ----
LLADA_MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
EXP12_NEW_TOKENS = 256                        # smoke test: 64
EXP12_STEPS = 256                             # smoke test: 64
EXP12_BLOCK_LENGTH = 8                        # official LLaDA GSM8K setting
EXP12_TEMPERATURE = 0.0
EXP12A_N_QUESTIONS = 100                      # smoke test: 2
EXP12A_RESUME = True
EXP11_MODEL = "dream"                         # "llada" for the LLaDA split prototype
EXP11_LLADA_CACHE_POLICY = ("fixed", 50, 7, 0.25)   # official dLLM-Cache LLaDA GSM8K
EXP11_LLADA_TEMPERATURE = 0.0
```

Energy needs `pip install nvidia-ml-py` (module `pynvml`). Without it the run still
works and the energy columns stay empty.

## 0b. Exp8F / Exp8G block caching under the entropy sampler (one GPU)

Same switch as the RQ1-RQ5 rerun: set `RERUN_ENTROPY = True`, then

```bash
SINGLE_GPU=1 nohup bash experiments/run_exp8_entropy.sh > exp8_entropy.log 2>&1 &
```

and set `RERUN_ENTROPY = False` afterwards. Results: `outputs/exp8f_entropy/`,
`outputs/exp8g_entropy/` (originals untouched). For the paper, Exp8F with
`EXP8F_TASK_KEYS = ["reasoning"]` (the current config) is enough; Exp8G gives the
"x of 36 runs keep the reference trajectory" number. Do not run this at the same
time as Exp11 on the same GPU (Exp8F timings would be disturbed).

## 1. Exp11 add-on (Dream, ~1 h GSM8K + ~25 min long prompts)

```bash
# GSM8K: EXP11_DATASET="gsm8k", EXP11_NEW_TOKENS=EXP11_STEPS=256, EXP11_N_QUESTIONS=30
CUDA_VISIBLE_DEVICES=1,2 nohup python experiments/exp11_dasr_split.py > exp11_addon_gsm8k.log 2>&1 &
# long prompts: EXP11_DATASET="authored", EXP11_NEW_TOKENS=EXP11_STEPS=128
CUDA_VISIBLE_DEVICES=1,2 nohup python experiments/exp11_dasr_split.py > exp11_addon_authored.log 2>&1 &
```

Run them one after the other (both use the same two GPUs). The script refuses to
write into a folder whose `runs.csv` holds a different variant set, so a forgotten
`EXP11_OUTPUT_TAG` stops with a message instead of overwriting the finished runs.

Check: `S0_split_full` in the add-on must repeat the original S0 answers (same seed);
`exp11c` prints this ("add-on S0 repeats the original S0 text on 100% ...").

## 2. LLaDA feature-cache quality on GSM8K (Exp12A, one GPU)

Run in the environment in which Exp7C (LLaDA) ran.

```bash
# smoke test first: EXP12A_N_QUESTIONS=2, EXP12_NEW_TOKENS=EXP12_STEPS=64
CUDA_VISIBLE_DEVICES=1 python experiments/exp12a_llada_gsm8k_cache.py
# full: EXP12A_N_QUESTIONS=100, 256/256  (~6 generations per question)
CUDA_VISIBLE_DEVICES=1 nohup python experiments/exp12a_llada_gsm8k_cache.py > exp12a.log 2>&1 &
```

Output `outputs/exp12a_llada_T0/` (same files as Exp10D). Read first:
`F_full` must have 0 regressions and (almost) all runs `exact_final` = the hook
itself does not change LLaDA. Then accuracy / McNemar p / row work of
`F_k100r7`, `F_k50r7`, `F_k25r2`, `AD_frontier`.

## 3. LLaDA split prototype (Exp11 with EXP11_MODEL = "llada", two GPUs)

```python
EXP11_MODEL = "llada"
EXP11_DATASET = "gsm8k"; EXP11_N_QUESTIONS = 30
EXP11_NEW_TOKENS = 256; EXP11_STEPS = 256
EXP11_OUTPUT_TAG = ""            # new folder anyway: outputs/exp11_llada_gsm8k_k4_t256_T0/
# remove EXP11_VARIANT_NAMES (or list the variants you want); the default set is
# S0, S1, C1, C2, C3, C4, A1. Add "S2_split_int8" for the compressed split.
```

```bash
# smoke test: EXP11_N_QUESTIONS=2, 64/64
CUDA_VISIBLE_DEVICES=1,2 nohup python experiments/exp11_dasr_split.py > exp11_llada.log 2>&1 &
python experiments/exp11b_network_emulation.py outputs/exp11_llada_gsm8k_k4_t256_T0
```

Must hold: `exact_vs_ref = runs` for S1 (vs S0), C2 and C3 (vs C1). LLaDA has no
logit shift, so cache-aware transmission needs no prompt-tail rows.

## 4. Tables and figures (no GPU)

```bash
python experiments/exp11c_tables_figures.py
```

Picks up the add-on folders (S2, memory, energy) and the LLaDA folder
(`exp11_llada_gsm8k_k4_t256_T0`) automatically when they exist; the `\hl{...}`
pending note in Table XI shrinks accordingly. Copy `outputs/exp11c/*.tex` and
`fig_*.pdf` into the paper (`tables/`, `figs/`).
