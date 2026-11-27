# Exp23: public long-prompt task (HotpotQA)

## Question
The long-prompt results so far use 12 authored cases. Reviewers will ask for a public
benchmark. HotpotQA in its distractor setting gives every question 10 Wikipedia paragraphs
(2 with the answer, 8 retrieved distractors), so the prompt dominates the sequence, as in the
authored cases, with public questions and gold answers.

## Data (`src/hotpotqa.py`)
- Official dev set (Yang et al., EMNLP 2018, CC BY-SA 4.0), distractor setting, downloaded
  once by `run_exp23.sh` from the Hugging Face copy `hotpotqa/hotpot_qa` (the authors' server
  curtis.ml.cmu.edu did not resolve from the lab); the first 100 questions are kept in
  `data/hotpotqa/dev_distractor_first100.jsonl`.
- Prompt: the 10 paragraphs in their official order ("Paragraph i: title" + text), the
  question, and "Reply with the short answer only".
- Score: `ok` = the gold answer appears in the reply after HotpotQA's official answer
  normalization (lower case, no punctuation or articles); for yes/no questions the reply's first
  yes or no must match. The official token F1 of the reply's first line is also returned.
- 30 questions, 128 new tokens (128 steps for dLLM-Cache), as the authored long prompts.

## Run
```
nohup bash experiments/run_exp23.sh > exp23.log 2>&1 &          # waits for idle GPUs 1, 2
SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp23.sh > exp23_smoke.log 2>&1   # ~10 min
```
- dLLM-Cache backend (Exp11 script): S0-S4, C1, C2, C4-C6, A1; then the emulated latency.
- Fast-dLLM backend (Exp13 script): F0, F1, K1, K2 (gamma 0.9 and 0.7), K3-K5.
- The log first prints the prompt lengths in tokens.

## What to read
`outputs/exp11_hotpotqa_k4_t128_T0.1/` and `outputs/exp13_dream_hotpotqa_k4_t128_b32/`:
`summary_by_variant.csv` (accuracy, uplink MB, regressions), `emulated_summary.csv` (latency).
These become a fourth workload column of the main table and the Fast-dLLM table.
