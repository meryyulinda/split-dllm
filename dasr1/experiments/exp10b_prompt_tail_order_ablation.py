"""Exp10B: why does official dLLM-Cache break answers on Dream? (root-cause test)

Run from dasr1/:  python experiments/exp10b_prompt_tail_order_ablation.py

Same 12 authored cases as Exp10A, same cache schedule (Kp=100, Kr=7, rho=0.25).
Two hypotheses, crossed 2 x 2:

H1  Prompt-tail off-by-one. Dream predicts response token 0 from
    the LAST PROMPT ROW (logit shift), but the official hook keeps that row in
    the prompt cache (refreshed every Kp=100 steps). Fix: prompt_tail_rows=1.
    Evidence in Exp10A: wrong FIRST token only ("1 reorder threshold ... is 140",
    "The3 November 2026"), Kr 7/3/1 identical, only Kp=1 helps.
H2  alg="origin" commits tokens in RANDOM order, i.e. also low-margin tokens, so
    small cache errors flip argmax. Official dLLM-Cache Dream evals use
    alg="entropy". Fix: alg="entropy".

Expected if H1 holds: A1 >> A0 on first_token_equal / exact_final, and the
divergences that remain in A1 no longer sit at position 0 (div_includes_pos0).
Expected if H2 holds: B0 >> A0, and remaining divergences in A* happen at
low baseline commit margin (div_base_commit_margin in runs.csv).

Note: entropy + temperature 0 is deterministic, so it runs only the first seed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp10_common import ROOT, cfg, run_grid  # noqa: E402  (sets sys.path for dasr1/)

DEMO = ("fixed", 100, 7, 0.25)  # official demo_Dream.py / Exp10A setting

DEFAULT_VARIANTS = [
    {"name": "A0_origin_official", "alg": "origin", "prompt_tail_rows": 0, "policy": DEMO},
    {"name": "A1_origin_tailfix", "alg": "origin", "prompt_tail_rows": 1, "policy": DEMO},
    {"name": "B0_entropy_official", "alg": "entropy", "prompt_tail_rows": 0, "policy": DEMO},
    {"name": "B1_entropy_tailfix", "alg": "entropy", "prompt_tail_rows": 1, "policy": DEMO},
]


def main():
    run_grid(
        experiment="Exp10B prompt-tail x decoding-order ablation",
        variants=list(getattr(cfg, "EXP10B_VARIANTS", DEFAULT_VARIANTS)),
        out_dir=ROOT / "outputs" / "exp10b",
        seeds=list(getattr(cfg, "EXP10B_SEEDS", [42, 43, 44])),
        new_tokens=int(getattr(cfg, "EXP10B_NEW_TOKENS", 128)),
        steps=int(getattr(cfg, "EXP10B_STEPS", 128)),
        temperature=getattr(cfg, "EXP10B_TEMPERATURE", 0.0),
        top_p=getattr(cfg, "EXP10B_TOP_P", None),
        top_k=getattr(cfg, "EXP10B_TOP_K", None),
        case_ids=list(getattr(cfg, "EXP10B_CASE_IDS", [])),
        warmup_runs=int(getattr(cfg, "EXP10B_WARMUP_RUNS", 1)),
        run_timing=bool(getattr(cfg, "EXP10B_RUN_TIMING", False)),
        timing_repeats=int(getattr(cfg, "EXP10B_TIMING_REPEATS", 2)),
    )


if __name__ == "__main__":
    main()
