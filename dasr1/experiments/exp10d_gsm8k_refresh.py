"""Exp10D: feature-cache refresh policies on GSM8K (real benchmark, math).

Run from dasr1/:  python experiments/exp10d_gsm8k_refresh.py

Why: Exp10C's conclusions ("factual/instruction are safe, math is sensitive",
"adaptive vs fixed") rest on only 3 math questions, one of which has an exact
tie (margin 0.0) in the baseline and breaks under EVERY cache setting. Tuning
rules on that would be tuning to noise. GSM8K is also the benchmark the
dLLM-Cache paper reports for Dream, so numbers are comparable.

Setting: Dream-v0-Instruct-7B, alg="entropy", prompt-tail fix on, 256 tokens /
256 steps, zero-shot chat prompt asking for the final answer after "####".
Default sampling = Dream's official GSM8K eval (eval_instruct/eval.sh):
temperature 0.1, top_p 0.9. With temperature 0 Dream fills the answer from the
end with EOS first and often collapses to an empty answer (Exp10D, T=0: 5/10
baseline answers empty). With T > 0 every seed in EXP10D_SEEDS is run, and the
baseline and the cached run of a question share the same seed.

Variants:
  F_full            (1, 1, 1.0) through the cache hook: everything recomputed every
                    step. NOISE FLOOR. Its regressions are numerical differences of
                    the hook's attention path, not caching. Subtract them mentally
                    from every other variant.
  F_k100r7          official demo setting (Exp10A-C)
  F_k25r2           official dLLM-Cache GSM8K setting for Dream
  F_k5r1            conservative (official mmlu_pro)
  AD_frontier       rule-based adaptive (Exp10C)
  AD_frontier_tight rule-based adaptive, tighter (Exp10C)

Read summary_by_variant.csv:
  cache_acc vs baseline_acc   accuracy with / without cache
  regressions, improvements   paired flips (baseline right -> cache wrong, and back)
  mcnemar_p                   exact McNemar test on those flips; p < 0.05 means the
                              accuracy change is unlikely to be chance
  mean_row_work_frac          compute proxy (fraction of token rows recomputed)
A good refresh policy: regressions close to F_full's, at the lowest work.

Cost: ~7 generations per question (~1.5 min on an RTX A6000) -> 100 questions
~2.5 h. Do a quick run first (EXP10D_N_QUESTIONS = 10). The run is resumable:
with EXP10D_RESUME = True a restart skips finished questions.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp10_common import ROOT, cfg, run_grid  # noqa: E402  (sets sys.path for dasr1/)

DATA = Path(getattr(cfg, "EXP10D_DATA", ROOT / "data" / "gsm8k" / "test_first200.jsonl"))
N = int(getattr(cfg, "EXP10D_N_QUESTIONS", 100))
START = int(getattr(cfg, "EXP10D_START", 0))
INSTRUCTION = getattr(
    cfg, "EXP10D_INSTRUCTION",
    "Please reason step by step, and put your final answer after '####'.")

NUM = r"-?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def to_number(text):
    if text is None:
        return None
    cleaned = text.replace("$", "").replace(",", "").replace(" ", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_answer(text):
    """Number after the last '####'; otherwise the last number in the text."""
    text = str(text or "")
    tagged = re.findall(r"####\s*(" + NUM + ")", text)
    if tagged:
        return to_number(tagged[-1])
    numbers = re.findall(NUM, text)
    return to_number(numbers[-1]) if numbers else None


def gsm8k_score(case, text):
    pred = extract_answer(text)
    gold = float(case["gold_answer"])
    return {"ok": pred is not None and abs(pred - gold) < 1e-6, "extracted": pred}


def load_cases(n=None, start=None, data=None):
    n = N if n is None else int(n)
    start = START if start is None else int(start)
    data = Path(data) if data is not None else DATA
    if not data.is_file():
        raise FileNotFoundError(f"GSM8K file not found: {data}")
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    selected = rows[start:start + n]
    if not selected:
        raise ValueError(f"No questions selected (start={start}, n={n}, file has {len(rows)})")
    cases = []
    for offset, row in enumerate(selected):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        cases.append({
            "case_id": f"gsm8k_{start + offset:04d}", "doc_id": "gsm8k", "task": "math",
            "question": row["question"], "gold_answer": gold,
            "prompt": row["question"].strip() + "\n" + INSTRUCTION,
        })
    return cases


def v(name, policy):
    return {"name": name, "alg": "entropy", "prompt_tail_rows": 1, "policy": policy}


DEFAULT_VARIANTS = [
    v("F_full", ("fixed", 1, 1, 1.0)),
    v("F_k100r7", ("fixed", 100, 7, 0.25)),
    v("F_k25r2", ("fixed", 25, 2, 0.25)),
    v("F_k5r1", ("fixed", 5, 1, 0.25)),
    v("AD_frontier", ("adaptive", {"gen_content_commits": 2, "gen_max_age": 4,
                                   "prompt_content_commits": 16, "prompt_max_age": 32,
                                   "eos_frontier_window": 8, "transfer_ratio": 0.25})),
    v("AD_frontier_tight", ("adaptive", {"gen_content_commits": 1, "gen_max_age": 2,
                                         "prompt_content_commits": 8, "prompt_max_age": 16,
                                         "eos_frontier_window": 8, "transfer_ratio": 0.25})),
]


TEMPERATURE = float(getattr(cfg, "EXP10D_TEMPERATURE", 0.1))
TOP_P = getattr(cfg, "EXP10D_TOP_P", 0.9)
SEEDS = [int(x) for x in getattr(cfg, "EXP10D_SEEDS", [getattr(cfg, "SEED", 42)])]


def main():
    run_grid(
        experiment="Exp10D GSM8K refresh policies",
        variants=list(getattr(cfg, "EXP10D_VARIANTS", DEFAULT_VARIANTS)),
        out_dir=ROOT / "outputs" / f"exp10d_T{TEMPERATURE:g}",
        seeds=SEEDS,
        new_tokens=int(getattr(cfg, "EXP10D_NEW_TOKENS", 256)),
        steps=int(getattr(cfg, "EXP10D_STEPS", 256)),
        temperature=TEMPERATURE,
        top_p=TOP_P if TEMPERATURE > 0 else None,
        cases=load_cases(),
        score_fn=gsm8k_score,
        warmup_runs=int(getattr(cfg, "EXP10D_WARMUP_RUNS", 1)),
        run_timing=bool(getattr(cfg, "EXP10D_RUN_TIMING", False)),
        timing_repeats=int(getattr(cfg, "EXP10D_TIMING_REPEATS", 2)),
        resume=bool(getattr(cfg, "EXP10D_RESUME", True)),
    )


if __name__ == "__main__":
    main()
