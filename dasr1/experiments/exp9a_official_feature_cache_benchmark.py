"""Exp9A: official dLLM-Cache feature caching vs vanilla Dream.

Purpose
-------
Reproduce a genuine FEATURE-CACHING baseline before implementing it in a
split edge/cloud runtime. This uses the official dLLM-Cache implementation,
not the residual-block cache implemented for Exp8B-8E.

Run this from dasr1/ with a Dream-compatible environment:
    python experiments/exp9a_official_feature_cache_benchmark.py

Requirements
------------
1. Clone https://github.com/maomaocun/dLLM-cache to the directory specified
   by EXP9A_OFFICIAL_CACHE_REPO in configs/experiment_config.py.
2. Use the exact generation settings of the project's experiment_config.py.
3. Ensure the official package's dependencies are present in the environment.
   Do not run its unpinned requirements installer blindly in the working Dream
   environment; use a separate environment if versions conflict.

Benchmarks actual *single-GPU full generation time* (including trajectory
instrumentation) using interleaved baseline/cache/baseline trials, and compares
all denoising token states and final answers. Cache installation/reset is
recorded separately from generation time. It does not measure edge-cloud
network, and it does not claim token-level exactness when trajectories differ.

Official reference:
  Liu et al., dLLM-Cache, ICML 2026.
  https://github.com/maomaocun/dLLM-cache/blob/main/demo_Dream.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import transformers

from configs.experiment_config import (
    MODEL_NAME, SEED, MAX_NEW_TOKENS, STEPS,
    TEMPERATURE, TOP_P, TOP_K, ALGORITHM, PROMPT, RQ5_PROMPTS,
    EXP9A_OFFICIAL_CACHE_REPO,
    EXP9A_PROMPT_INTERVAL_STEPS, EXP9A_GEN_INTERVAL_STEPS,
    EXP9A_TRANSFER_RATIOS, EXP9A_TASK_KEYS, EXP9A_SEEDS,
    EXP9A_REPEATS, EXP9A_WARMUP_RUNS,
)
from src.model_loader import load_dream_model
from src.generation import generate_with_token_history
from src.instrumentation import create_token_history_hook
from src.utils import set_seed

OUTPUT_DIR = Path(PROJECT_ROOT) / "outputs" / "exp9a"


def load_official_cache():
    """Import only the APIs shown in the official Dream demo; fail fast."""
    candidate = Path(os.path.expandvars(os.path.expanduser(EXP9A_OFFICIAL_CACHE_REPO)))
    if not candidate.is_absolute():
        candidate = Path(PROJECT_ROOT) / candidate
    candidate = candidate.resolve()
    if not (candidate / "dllm_cache" / "cache" / "Cache.py").is_file():
        raise FileNotFoundError(
            f"Official cache implementation not found in: {candidate}"
        )
    sys.path.insert(0, str(candidate))
    try:
        from dllm_cache.cache import dLLMCache, dLLMCacheConfig
        from dllm_cache.hooks import register_cache_Dream, logout_cache_Dream
    except ImportError as exc:
        raise RuntimeError(
            "Could not import official dLLM-Cache. Check its dependencies and "
            "your Dream environment. The original Dream model loader must "
            "still work in this environment."
        ) from exc
    return candidate, dLLMCache, dLLMCacheConfig, register_cache_Dream, logout_cache_Dream


def sync(device):
    torch.cuda.synchronize(device)


def trajectory_sha256(states):
    digest = hashlib.sha256()
    for state in states:
        digest.update(state.numpy().tobytes())
    return digest.hexdigest()


def evaluate(reference, other):
    if reference["prompt_length"] != other["prompt_length"]:
        raise RuntimeError("Prompt length differs between compared runs")
    a, b = reference["states"], other["states"]
    if len(a) != len(b):
        raise RuntimeError("Token history length differs between runs")
    p = reference["prompt_length"]
    matches = [bool(torch.equal(x, y)) for x, y in zip(a, b)]
    ratios = [float((x[p:] == y[p:]).float().mean().item()) for x, y in zip(a, b)]
    first_diff = next((i for i, ok in enumerate(matches) if not ok), None)
    return {
        "exact_full_trajectory": bool(all(matches)),
        "exact_final_ids": bool(matches[-1]),
        "exact_final_text": reference["text"] == other["text"],
        "exact_state_count_including_initial": int(sum(matches)),
        "exact_transition_states": int(sum(matches[1:])),
        "first_divergent_state": first_diff,
        "mean_generated_state_token_agreement": float(statistics.mean(ratios[1:])),
        "final_generated_token_agreement": float(ratios[-1]),
    }


def record_generation(model, tokenizer, prompt, seed, device,
                      cache_api=None, prompt_len_for_cache=None,
                      prompt_interval=None, gen_interval=None, transfer_ratio=None):
    """Run with the same existing Dream generation + history hook as Exp8E.

    Official cache setup (new_instance/register/reset) happens before timing
    inference, as in their demo. It is measured separately and reported.
    Logout is guaranteed even if generation raises.
    """
    set_seed(int(seed))
    history_hook, token_history = create_token_history_hook()
    install_ms = 0.0
    cache_enabled = cache_api is not None

    if cache_enabled:
        if prompt_len_for_cache is None:
            raise ValueError("Need baseline-derived prompt length for cache reset")
        dLLMCache, dLLMCacheConfig, register_cache_Dream, logout_cache_Dream = cache_api
        sync(device)
        t0 = time.perf_counter()
        # Official Dream demo's exact public API (not a reimplementation).
        dLLMCache.new_instance(**asdict(dLLMCacheConfig(
            prompt_interval_steps=int(prompt_interval),
            gen_interval_steps=int(gen_interval),
            transfer_ratio=float(transfer_ratio),
        )))
        register_cache_Dream(model, "model.layers")
        try:
            dLLMCache().reset_cache(int(prompt_len_for_cache))
            sync(device)
        except Exception:
            logout_cache_Dream(model, "model.layers")
            raise
        install_ms = (time.perf_counter() - t0) * 1000

    try:
        sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            generate_with_token_history(
                model=model, tokenizer=tokenizer, prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS, steps=STEPS,
                temperature=TEMPERATURE, history_hook=history_hook,
                top_p=TOP_P, top_k=TOP_K, alg=ALGORITHM,
            )
        sync(device)
        generation_ms = (time.perf_counter() - start) * 1000.0
    finally:
        if cache_enabled:
            # Restore original Dream forwards before any baseline-after run.
            logout_cache_Dream(model, "model.layers")

    states = []
    for state in token_history:
        if not isinstance(state, torch.Tensor) or state.ndim != 1:
            raise RuntimeError("Expected 1D token states from Dream history hook")
        states.append(state.detach().cpu().long().clone())
    if len(states) != STEPS + 1:
        raise RuntimeError(f"Expected {STEPS+1} states; got {len(states)}")
    p = states[0].numel() - MAX_NEW_TOKENS
    if p <= 0 or any(len(s) != len(states[0]) for s in states):
        raise RuntimeError("Unexpected Dream sequence/prompt lengths")
    if cache_enabled and p != prompt_len_for_cache:
        raise RuntimeError("Actual prompt length differs from cache reset length")

    return {
        "states": states,
        "prompt_length": p,
        "text": tokenizer.decode(states[-1][p:], skip_special_tokens=True),
        "trajectory_sha256": trajectory_sha256(states),
        "generation_ms": generation_ms,
        "cache_install_reset_ms": install_ms,
    }


def summarize(rows, group_cols):
    df = pd.DataFrame(rows)
    df = df[df["baseline_pair_stable"]]
    if df.empty:
        return pd.DataFrame()
    aggregated = []
    for key, grp in df.groupby(group_cols, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_cols, key))
        record.update({
            "trials": int(len(grp)),
            "distinct_baseline_trajectories": int(grp["baseline_trajectory_sha256"].nunique()),
            "median_baseline_generation_ms": float(grp["paired_baseline_ms"].median()),
            "median_cached_generation_ms": float(grp["cached_generation_ms"].median()),
            "median_generation_saving_pct": float(grp["generation_saving_pct"].median()),
            "median_generation_speedup": float(grp["generation_speedup"].median()),
            "median_cache_install_reset_ms": float(grp["cache_install_reset_ms"].median()),
            "median_request_including_setup_saving_pct": float(
                grp["request_including_setup_saving_pct"].median()
            ),
            "exact_trajectory_trials": int(grp["exact_full_trajectory"].sum()),
            "exact_final_ids_trials": int(grp["exact_final_ids"].sum()),
            "exact_final_text_trials": int(grp["exact_final_text"].sum()),
            "mean_final_generated_token_agreement": float(
                grp["final_generated_token_agreement"].mean()
            ),
        })
        aggregated.append(record)
    return pd.DataFrame(aggregated)


def save_partial(records, texts):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "feature_cache_runs.csv", index=False)
    pd.DataFrame(texts).to_csv(OUTPUT_DIR / "generated_texts.csv", index=False)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Exp9A requires CUDA GPU")
    if "Dream" not in MODEL_NAME:
        raise RuntimeError("Exp9A uses the official Dream cache adapter only")
    if MAX_NEW_TOKENS != STEPS:
        print("NOTE: Non-1:1 generation schedules have not been validated for this experiment")

    candidate, dLLMCache, dLLMCacheConfig, register_cache_Dream, logout_cache_Dream = (
        load_official_cache()
    )
    cache_api = (dLLMCache, dLLMCacheConfig, register_cache_Dream, logout_cache_Dream)
    tasks = list(EXP9A_TASK_KEYS)
    seeds = list(EXP9A_SEEDS)
    cache_configs = [
        (int(pi), int(gi), float(rho))
        for pi in EXP9A_PROMPT_INTERVAL_STEPS
        for gi in EXP9A_GEN_INTERVAL_STEPS
        for rho in EXP9A_TRANSFER_RATIOS
    ]
    if not tasks or not seeds or not cache_configs:
        raise ValueError("Exp9A task/seed/cache configuration cannot be empty")
    if int(EXP9A_REPEATS) < 1 or int(EXP9A_WARMUP_RUNS) < 0:
        raise ValueError("Invalid repeats/warmups")
    for pi, gi, rho in cache_configs:
        if pi < 1 or gi < 1 or not (0 < rho <= 1):
            raise ValueError(f"Invalid official cache settings: {(pi,gi,rho)}")

    print("=" * 78)
    print("EXP9A: OFFICIAL dLLM-CACHE FEATURE CACHING (DREAM, SINGLE GPU)")
    print("=" * 78)
    print(f"Checkpoint: {MODEL_NAME}")
    print(f"Official repo: {candidate}")
    print(f"Generation: {MAX_NEW_TOKENS} tokens / {STEPS} steps | alg={ALGORITHM}")
    print(f"Temperature={TEMPERATURE} | top_p={TOP_P} | top_k={TOP_K}")
    print(f"Tasks={tasks}, seeds={seeds}, repeats={EXP9A_REPEATS}")
    print(f"Cache configs (prompt_interval, gen_interval, transfer_ratio): {cache_configs}")
    print("Cache setup time measured separately from generation time")

    model, tokenizer = load_dream_model(MODEL_NAME)
    model.eval()
    device = next(model.parameters()).device
    print(f"GPU={torch.cuda.get_device_name(device)} | device={device}")

    # Preserve normal generation config; official demo uses the normal Dream
    # model. Baseline and cached execution share the same model/settings.
    print(f"HF use_cache={getattr(model.config, 'use_cache', None)}")
    warm_prompt = PROMPT if tasks[0] == "default" else RQ5_PROMPTS[tasks[0]]
    for _ in range(EXP9A_WARMUP_RUNS):
        b = record_generation(model, tokenizer, warm_prompt, seeds[0], device)
        pi, gi, rho = cache_configs[0]
        record_generation(
            model, tokenizer, warm_prompt, seeds[0], device, cache_api,
            prompt_len_for_cache=b["prompt_length"], prompt_interval=pi,
            gen_interval=gi, transfer_ratio=rho,
        )
    print("Warmup completed")

    run_rows, texts = [], []
    planned = len(tasks) * len(seeds) * len(cache_configs) * int(EXP9A_REPEATS)
    trial_id = 0

    for task in tasks:
        prompt = PROMPT if task == "default" else RQ5_PROMPTS[task]
        for seed in seeds:
            for repeat in range(int(EXP9A_REPEATS)):
                for pi, gi, rho in cache_configs:
                    trial_id += 1
                    before = record_generation(model, tokenizer, prompt, seed, device)
                    cached = record_generation(
                        model, tokenizer, prompt, seed, device, cache_api,
                        prompt_len_for_cache=before["prompt_length"],
                        prompt_interval=pi, gen_interval=gi, transfer_ratio=rho,
                    )
                    after = record_generation(model, tokenizer, prompt, seed, device)

                    control = evaluate(before, after)
                    fidelity = evaluate(before, cached)
                    paired_ms = (before["generation_ms"] + after["generation_ms"]) / 2.0
                    cached_ms = cached["generation_ms"]
                    setup_ms = cached["cache_install_reset_ms"]
                    row = {
                        "trial_id": trial_id, "task": task, "seed": int(seed),
                        "repeat": repeat, "prompt": prompt,
                        "prompt_interval_steps": pi,
                        "gen_interval_steps": gi,
                        "transfer_ratio": rho,
                        "prompt_length": before["prompt_length"],
                        "baseline_before_ms": before["generation_ms"],
                        "baseline_after_ms": after["generation_ms"],
                        "paired_baseline_ms": paired_ms,
                        "cached_generation_ms": cached_ms,
                        "cache_install_reset_ms": setup_ms,
                        "cached_request_including_setup_ms": cached_ms + setup_ms,
                        "generation_saving_pct": 100 * (1 - cached_ms / paired_ms),
                        "generation_speedup": paired_ms / cached_ms,
                        "request_including_setup_saving_pct": 100 * (
                            1 - (cached_ms + setup_ms) / paired_ms
                        ),
                        "baseline_pair_stable": control["exact_full_trajectory"],
                        "baseline_trajectory_sha256": before["trajectory_sha256"],
                        **fidelity,
                    }
                    run_rows.append(row)
                    texts.append({
                        "trial_id": trial_id, "task": task, "seed": int(seed),
                        "prompt_interval_steps": pi, "gen_interval_steps": gi,
                        "transfer_ratio": rho, "baseline_text": before["text"],
                        "cache_text": cached["text"],
                        "baseline_after_text": after["text"],
                    })
                    save_partial(run_rows, texts)
                    tag = "EXACT" if fidelity["exact_full_trajectory"] else (
                        f"DIFF@state{fidelity['first_divergent_state']}"
                    )
                    print(
                        f"[{trial_id:03d}/{planned:03d}] {task:<13} seed={seed} "
                        f"pi={pi} gi={gi} rho={rho:g} | "
                        f"B={paired_ms:.1f}ms C={cached_ms:.1f}ms "
                        f"save={row['generation_saving_pct']:+.2f}% | "
                        f"{tag} | control={'OK' if row['baseline_pair_stable'] else 'UNSTABLE'}",
                        flush=True,
                    )

    for group_cols, fn in [
        (["prompt_interval_steps", "gen_interval_steps", "transfer_ratio"], "feature_cache_summary.csv"),
        (["task", "prompt_interval_steps", "gen_interval_steps", "transfer_ratio"], "feature_cache_by_task.csv"),
        (["task", "seed", "prompt_interval_steps", "gen_interval_steps", "transfer_ratio"], "feature_cache_by_task_seed.csv"),
    ]:
        summarize(run_rows, group_cols).to_csv(OUTPUT_DIR / fn, index=False)

    metadata = {
        "experiment": "Exp9A official dLLM-Cache baseline reproduction (not split inference)",
        "upstream_source": "https://github.com/maomaocun/dLLM-cache",
        "upstream_repo_path": str(candidate),
        "method": "Official cache APIs: dLLMCacheConfig/new_instance, register_cache_Dream, reset_cache, logout_cache_Dream",
        "settings": {
            "model": MODEL_NAME, "max_new_tokens": MAX_NEW_TOKENS, "steps": STEPS,
            "algorithm": ALGORITHM, "temperature": TEMPERATURE, "top_p": TOP_P,
            "top_k": TOP_K, "tasks": tasks, "seeds": seeds, "repeats": EXP9A_REPEATS,
            "cache_configs": cache_configs,
        },
        "gpu": torch.cuda.get_device_name(device), "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "latency_definition": "CUDA-synchronized full-generation time with identical trajectory capture; cache registration/reset measured separately; report both times",
        "limitations": "One GPU, not two-GPU split. Fidelity measured on observed token trajectories; approximate feature caching may change logits, decoding decisions or final quality. No wire bytes measured.",
    }
    with (OUTPUT_DIR / "experiment_metadata.json").open("w", encoding="utf-8") as out:
        json.dump(metadata, out, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 78)
    print("OFFICIAL FEATURE CACHE RESULTS (STABLE B-C-B PAIRS ONLY)")
    print("=" * 78)
    for fn in ["feature_cache_summary.csv", "feature_cache_by_task.csv"]:
        print("\n", fn)
        df = pd.read_csv(OUTPUT_DIR / fn)
        if df.empty:
            print("No stable pairs")
        else:
            cols = ["task"] if "task" in df else []
            cols += ["prompt_interval_steps", "gen_interval_steps", "transfer_ratio",
                     "trials", "median_generation_saving_pct", "exact_trajectory_trials",
                     "exact_final_ids_trials"]
            print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not all(r["baseline_pair_stable"] for r in run_rows):
        print("WARNING: Some baseline B-C-B control trajectories differed.")
    if not all(r["exact_full_trajectory"] for r in run_rows):
        print("WARNING: Cache changed some token trajectories. Inspect texts and task correctness.")
    print("\nSaved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
