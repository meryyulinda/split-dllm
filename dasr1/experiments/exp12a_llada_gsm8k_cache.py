"""Exp12A: dLLM-Cache feature caching on LLaDA-8B-Instruct, GSM8K quality (RQ5).

Run from dasr1/:  python experiments/exp12a_llada_gsm8k_cache.py

LLaDA counterpart of Exp10D (Dream). Same GSM8K questions, same zero-shot prompt,
same answer extraction, same paired analysis (baseline vs cached run of the same
question, exact McNemar test, F_full noise floor). Differences, all following the
official LLaDA / dLLM-Cache GSM8K setting:
  * sampler: LLaDA low-confidence remasking, semi-autoregressive blocks of
    EXP12_BLOCK_LENGTH = 8 tokens, 256 tokens / 256 steps, temperature 0
    (deterministic, so one seed is enough)
  * official dLLM-Cache LLaDA hook (cache_hook_LLaDA on model.transformer.blocks)
  * no prompt-tail correction: LLaDA has no logit shift (prompt_tail_rows = 0)

Variants:
  F_full       (1, 1, 1.0) through the hook = NOISE FLOOR (must be ~identical)
  F_k100r7     official demo setting (demo_LLaDA.py)
  F_k50r7      official LLaDA GSM8K setting (scripts/run_LLaDA_gsm8k_Instruct.sh)
  F_k25r2      official Dream GSM8K setting (what Dream uses in Exp10D / Exp11)
  AD_frontier  rule-based adaptive refresh, same parameters as Dream (Exp10D)

Output: outputs/exp12a_llada_T<T>/ with the same files as Exp10D
(baselines.csv, runs.csv, summary_by_variant.csv, commits.jsonl, metadata.json).

Environment: run in the environment in which Exp7C (LLaDA) already ran. Cost:
6 generations per question; do a smoke test first (EXP12A_N_QUESTIONS = 2,
EXP12_NEW_TOKENS = EXP12_STEPS = 64). Resumable (EXP12A_RESUME = True).
"""
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp10_common import ROOT, cfg, run_grid  # noqa: E402  (sets sys.path for dasr1/)
from exp9b_official_feature_cache_diagnostics import import_official  # noqa: E402
from exp10d_gsm8k_refresh import gsm8k_score, load_cases  # noqa: E402
from src.llada_cache_ext import (  # noqa: E402
    LLADA_MODEL_NAME, import_official_llada, llada_generate, llada_prepare_input,
    load_llada_model,
)

MODEL_NAME = getattr(cfg, "LLADA_MODEL_NAME", LLADA_MODEL_NAME)
N = int(getattr(cfg, "EXP12A_N_QUESTIONS", 100))
START = int(getattr(cfg, "EXP12A_START", 0))
NEW_TOKENS = int(getattr(cfg, "EXP12_NEW_TOKENS", 256))
STEPS = int(getattr(cfg, "EXP12_STEPS", 256))
BLOCK_LENGTH = int(getattr(cfg, "EXP12_BLOCK_LENGTH", 8))
TEMPERATURE = float(getattr(cfg, "EXP12_TEMPERATURE", 0.0))
SEEDS = [int(x) for x in getattr(cfg, "EXP12A_SEEDS", [getattr(cfg, "SEED", 42)])]


def v(name, policy):
    return {"name": name, "alg": "low_confidence", "prompt_tail_rows": 0,
            "force_prompt_tail": False, "policy": policy}


ADAPTIVE = ("adaptive", {"gen_content_commits": 2, "gen_max_age": 4,
                         "prompt_content_commits": 16, "prompt_max_age": 32,
                         "eos_frontier_window": 8, "transfer_ratio": 0.25})
DEFAULT_VARIANTS = [
    v("F_full", ("fixed", 1, 1, 1.0)),
    v("F_k100r7", ("fixed", 100, 7, 0.25)),
    v("F_k50r7", ("fixed", 50, 7, 0.25)),
    v("F_k25r2", ("fixed", 25, 2, 0.25)),
    v("AD_frontier", ADAPTIVE),
]


def llada_backend():
    def api():
        repo, _dream_api = import_official()  # puts the official repo on sys.path
        return repo, import_official_llada(repo)

    return {
        "name": MODEL_NAME,
        "algs": {"low_confidence", "random"},
        "api": api,
        "load": lambda: load_llada_model(MODEL_NAME),
        "prepare": llada_prepare_input,
        "generate": partial(llada_generate, block_length=BLOCK_LENGTH),
    }


def main():
    run_grid(
        experiment="Exp12A LLaDA GSM8K feature cache",
        variants=list(getattr(cfg, "EXP12A_VARIANTS", DEFAULT_VARIANTS)),
        out_dir=ROOT / "outputs" / f"exp12a_llada_T{TEMPERATURE:g}",
        seeds=SEEDS,
        new_tokens=NEW_TOKENS,
        steps=STEPS,
        temperature=TEMPERATURE,
        cases=load_cases(n=N, start=START),
        score_fn=gsm8k_score,
        warmup_runs=int(getattr(cfg, "EXP12A_WARMUP_RUNS", 1)),
        run_timing=bool(getattr(cfg, "EXP12A_RUN_TIMING", False)),
        timing_repeats=int(getattr(cfg, "EXP12A_TIMING_REPEATS", 2)),
        resume=bool(getattr(cfg, "EXP12A_RESUME", True)),
        backend=llada_backend(),
    )


if __name__ == "__main__":
    main()
