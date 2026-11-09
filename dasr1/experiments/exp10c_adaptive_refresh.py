"""Exp10C: quality vs work for fixed schedules, and a rule-based adaptive refresh.

Run from dasr1/:  python experiments/exp10c_adaptive_refresh.py
Uses alg="entropy" + prompt-tail fix (B1 in Exp10B, the best setting there).

Exp10B finding: at (Kp=100, Kr=7) the cache does only ~7% of the row work and
still breaks 2/10 correct answers (math). Remaining divergences pile up at the
OLDEST step of the response-cache window and often are early EOS commits that
cut the answer short.

Two questions:
 Q1  Quality vs work: how much refresh is needed before regressions vanish?
     Fixed schedules from aggressive to conservative, incl. official per-task
     settings (gsm8k 25/2, humaneval 50/1, mmlu_pro 5/1). F_k100r1 isolates the
     response cache (answer refreshed every step, prompt almost never).
 Q2  Can ONE rule-based policy reach the quality of the conservative schedules
     at lower work, without per-task tuning?

Adaptive rule (src/refresh_rules.AdaptivePolicy): refresh the response features
after G triggers or Ag steps, the prompt after P triggers or Ap steps.
Trigger = committed content token, or an EOS committed near the answer end
(within W positions of the rightmost content token; it decides the length).
The signal is token-level and free in the sampler (no extra forward pass), so
it can run on the edge side of a split deployment.

Approximate row work (prompt ~700, answer 128): k100r7 ~7%, k25r2 ~14%,
k100r1 ~17%, k50r1 ~18%, k5r1 ~33%. Read summary_by_variant(_task).csv:
cache_ok vs baseline_ok, regressions, exact_final, mean_row_work_frac.
Check every regression manually in runs.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp10_common import ROOT, cfg, run_grid  # noqa: E402  (sets sys.path for dasr1/)

ALG = getattr(cfg, "EXP10C_ALG", "entropy")
B = int(getattr(cfg, "EXP10C_PROMPT_TAIL_ROWS", 1))


def v(name, policy):
    return {"name": name, "alg": ALG, "prompt_tail_rows": B, "policy": policy}


DEFAULT_VARIANTS = [
    # Q1: fixed schedules, aggressive -> conservative
    v("F_k100r7", ("fixed", 100, 7, 0.25)),   # official demo (Exp10A/B)
    v("F_k25r2", ("fixed", 25, 2, 0.25)),     # official gsm8k
    v("F_k100r1", ("fixed", 100, 1, 0.25)),   # response refreshed every step
    v("F_k50r1", ("fixed", 50, 1, 0.25)),     # official humaneval
    v("F_k5r1", ("fixed", 5, 1, 0.25)),       # official mmlu_pro
    # Q2: rule-based adaptive (same rule for every task)
    v("AD_frontier", ("adaptive", {"gen_content_commits": 2, "gen_max_age": 4,
                                   "prompt_content_commits": 16, "prompt_max_age": 32,
                                   "eos_frontier_window": 8, "transfer_ratio": 0.25})),
    v("AD_frontier_tight", ("adaptive", {"gen_content_commits": 1, "gen_max_age": 2,
                                         "prompt_content_commits": 8, "prompt_max_age": 16,
                                         "eos_frontier_window": 8, "transfer_ratio": 0.25})),
]


def main():
    run_grid(
        experiment="Exp10C rule-based adaptive refresh",
        variants=list(getattr(cfg, "EXP10C_VARIANTS", DEFAULT_VARIANTS)),
        out_dir=ROOT / "outputs" / "exp10c",
        seeds=list(getattr(cfg, "EXP10C_SEEDS", [42, 43, 44])),
        new_tokens=int(getattr(cfg, "EXP10C_NEW_TOKENS", 128)),
        steps=int(getattr(cfg, "EXP10C_STEPS", 128)),
        temperature=getattr(cfg, "EXP10C_TEMPERATURE", 0.0),
        top_p=getattr(cfg, "EXP10C_TOP_P", None),
        top_k=getattr(cfg, "EXP10C_TOP_K", None),
        case_ids=list(getattr(cfg, "EXP10C_CASE_IDS", [])),
        warmup_runs=int(getattr(cfg, "EXP10C_WARMUP_RUNS", 1)),
        run_timing=bool(getattr(cfg, "EXP10C_RUN_TIMING", False)),
        timing_repeats=int(getattr(cfg, "EXP10C_TIMING_REPEATS", 2)),
    )


if __name__ == "__main__":
    main()
