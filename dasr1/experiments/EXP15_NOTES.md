# Exp15: boundary leakage (privacy) and Exp13C: split exactness of Fast-dLLM

## Exp15: how much of the prompt do the boundary activations reveal?
The main reason to split is privacy: the raw prompt stays on the device. But the edge sends the boundary rows of every prompt token at least once (first step / first block start). Exp15 measures, for splits after layers 1, 2, 4, 8, 12, 16, 20, 24, how many tokens of the private part of a prompt an observer of the link recovers:

| Attack | Needs | Idea |
|---|---|---|
| `embed_nn` | public weights | nearest input-embedding row to the boundary row |
| `lm_lens` | public weights | model's own final norm + LM head on the boundary row (same or next token) |
| `ridge` | public weights + offline training | linear map boundary row -> input embedding, fitted on GSM8K train questions, nearest embedding |

Tested on held-out prompts: GSM8K test 0..99 (private part = the question) and the 12 authored cases (private part = document + question). Chat template and fixed instructions are not counted.

```bash
# one GPU, busy GPUs are fine (no timing); ~10-20 min for Dream
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python experiments/exp15_boundary_leakage.py
```
Outputs `outputs/exp15_leakage_dream/`: `summary.csv` (top-1 / top-5 / digits recovery per layer and attack), `examples.md` (recovered text), `metadata.json`. LLaDA: set `EXP15_MODEL = "llada"` in `configs/experiment_config.py`. Needs the Hugging Face `datasets` package and Hub access for GSM8K train (falls back to local test questions 100..199).

## Exp13C: does splitting Fast-dLLM change its output?
Runs K2 (DualCache + gamma 0.9) on one GPU, then split after layer 4 with bf16 rows, and compares the generated tokens prompt by prompt. Identical on every prompt = the Fast-dLLM tier's transmission is exact, like C2 vs C1 for dLLM-Cache.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 python experiments/exp13c_split_exactness.py   # ~5 min, Dream GSM8K 30
```
Optional in the config: `EXP13C_VARIANTS = ["F0", "K1", "K2"]`, `EXP13_DATASET = "authored"`, `EXP13_MODEL = "llada"`.
