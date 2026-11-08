"""Exp8E: cross-task/seed robustness of measured multi-block residual caching.

Run from dasr1/ with the Dream environment:
    python experiments/exp8e_multiblock_cache_robustness.py

Add EXP8E_* parameters to configs/experiment_config.py (see companion snippet).

Building on Exp8D, this experiment:
- truly bypasses block.forward() on reuse calls (no oracle output substitution),
- caches the total residual R_l = H_l(out) - H_l(in) for each selected block,
- compares baseline/cache/baseline sequentially for every trial (B-C-B),
- measures full-generation wall-clock GPU-synchronized time,
- compares all token states and final decoded text against local baseline,
- tests multiple prompts, seeds, and refresh intervals, including k=8;
- records fidelity failures instead of assuming cache preserves generation;
- saves checkpoints after each paired trial (so partial data survive interruptions).

Not KV caching. Not edge-cloud split. No performance claims until hardware runs.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import json
import hashlib
import platform
import statistics
import time
from contextlib import ExitStack
from dataclasses import dataclass

import pandas as pd
import torch
import transformers

from configs.experiment_config import (
    MODEL_NAME, SEED, MAX_NEW_TOKENS, STEPS, TEMPERATURE,
    TOP_P, TOP_K, ALGORITHM, PROMPT, DRIFT_EPSILON,
    RQ5_PROMPTS,
    EXP8E_LAYER_GROUPS, EXP8E_REFRESH_INTERVALS,
    EXP8E_TASK_KEYS, EXP8E_SEEDS, EXP8E_REPEATS, EXP8E_WARMUP_RUNS,
)
from src.model_loader import load_dream_model
from src.generation import generate_with_token_history
from src.instrumentation import create_token_history_hook
from src.utils import set_seed

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "exp8e")


@dataclass
class CacheCounter:
    layer: int
    total: int = 0
    executed: int = 0
    reused: int = 0


class PeriodicBlockResidualCache:
    """Patch one Dream block. On skip, original block.forward is NOT called.

    Cache is allocated per experimental generation and reset between trials.
    When an earlier block is also cached, a later block's residual is defined
    relative to its *current (possibly approximated)* input, as required.
    """

    def __init__(self, model, layer_number, interval):
        self.layer_number = int(layer_number)
        self.interval = int(interval)
        self.layer = model.model.layers[self.layer_number - 1]
        self._original = None
        self._cached_residual_fp32 = None
        self._dtype = None
        self._tuple_extras = None
        self._returns_tuple = None
        self.counter = CacheCounter(layer=self.layer_number)
        self._last_refresh_call = None

    @staticmethod
    def _get_input(args, kwargs):
        x = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise RuntimeError("Expected Dream decoder block input [B, N, D]")
        return x

    def _forward(self, *args, **kwargs):
        if kwargs.get("use_cache", False) or kwargs.get("past_key_value") is not None:
            raise RuntimeError("Exp8E requires HF use_cache=False and no past_key_value")
        if kwargs.get("output_attentions", False):
            raise RuntimeError("Exp8E requires output_attentions=False")

        x = self._get_input(args, kwargs)
        step = self.counter.total
        self.counter.total += 1
        needs_refresh = (
            self._cached_residual_fp32 is None
            or (step - self._last_refresh_call) >= self.interval
        )
        if not needs_refresh:
            if (self._cached_residual_fp32.shape != x.shape
                    or self._cached_residual_fp32.device != x.device):
                raise RuntimeError("Cached residual shape/device mismatch")
            # Main optimization: original block.forward is never called here.
            approx_y = (x.float() + self._cached_residual_fp32).to(self._dtype)
            self.counter.reused += 1
            if self._returns_tuple:
                return (approx_y, *self._tuple_extras)
            return approx_y

        # Calculate the residual against the actual input, not an old input.
        input_fp32 = x.detach().float().clone()
        y_with_aux = self._original(*args, **kwargs)
        y = y_with_aux[0] if isinstance(y_with_aux, tuple) else y_with_aux
        if not isinstance(y, torch.Tensor) or y.shape != x.shape:
            raise RuntimeError("Unexpected Dream block output; expected [B,N,D]")
        extras = y_with_aux[1:] if isinstance(y_with_aux, tuple) else ()
        if any(item is not None for item in extras):
            raise RuntimeError(
                "Dream block returned auxiliary outputs despite disabling HF "
                "KV caching and attention outputs; refusing to reuse stale state"
            )
        self._returns_tuple = isinstance(y_with_aux, tuple)
        self._tuple_extras = extras
        self._dtype = y.dtype
        self._cached_residual_fp32 = y.detach().float().sub(input_fp32)
        self._last_refresh_call = step
        self.counter.executed += 1
        return y_with_aux

    def __enter__(self):
        if self._original is not None:
            raise RuntimeError("Block cache installed twice")
        self._original = self.layer.forward
        self.layer.forward = self._forward
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._original is not None:
            self.layer.forward = self._original
            self._original = None
        return False

    def stats(self):
        c = self.counter
        if c.total != c.executed + c.reused:
            raise RuntimeError(f"Inconsistent block counts for layer {c.layer}")
        return {"layer": c.layer, "total": c.total,
                "executed": c.executed, "reused": c.reused}


def disable_hf_kv_cache(model):
    """Use the same Dream forward contract for baseline and cached runs."""
    for name, owner in (("model", model), ("model.model", model.model)):
        cfg = getattr(owner, "config", None)
        if cfg is None:
            raise RuntimeError(f"Missing {name}.config")
        cfg.use_cache = False
        cfg.output_attentions = False
    cfg = getattr(model, "generation_config", None)
    if cfg is not None:
        cfg.use_cache = False
        cfg.output_attentions = False
    assert not model.config.use_cache
    assert not model.model.config.use_cache


def run_generation(model, tokenizer, device, prompt, seed, layers=(), interval=None):
    """A real Dream trajectory, with all per-run caches reset and timings synced."""
    set_seed(seed)
    hook, token_history = create_token_history_hook()
    caches = []
    if layers:
        if not isinstance(interval, int) or interval < 2:
            raise ValueError("Periodic refresh interval must be >= 2")
        caches = [PeriodicBlockResidualCache(model, layer, interval)
                  for layer in layers]

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        with ExitStack() as stack:
            for cache in caches:
                stack.enter_context(cache)
            generate_with_token_history(
                model=model, tokenizer=tokenizer, prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS, steps=STEPS,
                temperature=TEMPERATURE, history_hook=hook,
                top_p=TOP_P, top_k=TOP_K, alg=ALGORITHM,
            )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    states = []
    for state in token_history:
        if not isinstance(state, torch.Tensor) or state.ndim != 1:
            raise RuntimeError("Dream history hook returned a non-1D token state")
        states.append(state.detach().cpu().long().clone())
    if len(states) != STEPS + 1:
        raise RuntimeError(
            f"Expected {STEPS + 1} states; got {len(states)}. "
            "Check instrumentation / generation setting."
        )
    seq_len = states[0].numel()
    prompt_len = seq_len - MAX_NEW_TOKENS
    if prompt_len <= 0 or any(s.numel() != seq_len for s in states):
        raise RuntimeError("Invalid/inconsistent trajectory length")

    counters = [c.stats() for c in caches]
    for item in counters:
        if item["total"] != STEPS:
            raise RuntimeError(
                f"Layer {item['layer']} invoked {item['total']} times instead of "
                f"{STEPS}; caching schedule may be misaligned"
            )
    final_tokens = states[-1][prompt_len:]
    return {
        "states": states,
        "prompt_len": prompt_len,
        "final_text": tokenizer.decode(final_tokens, skip_special_tokens=True),
        "elapsed_ms": elapsed_ms,
        "counters": counters,
    }


def trajectory_fingerprint(result):
    """Hash all observed token states to detect identical seeds/trajectories."""
    digest = hashlib.sha256()
    for state in result["states"]:
        digest.update(state.numpy().tobytes())
    return digest.hexdigest()


def compare_trajectory(reference, trial):
    """Check every generated token state, not just final answer string."""
    if reference["prompt_len"] != trial["prompt_len"]:
        raise RuntimeError("Prompt length changed between baseline/cache")
    a, b = reference["states"], trial["states"]
    if len(a) != len(b):
        raise RuntimeError("Different trajectory length")
    p = reference["prompt_len"]
    exact = [bool(torch.equal(x, y)) for x, y in zip(a, b)]
    ratios = [float((x[p:] == y[p:]).float().mean()) for x, y in zip(a, b)]
    return {
        "final_ids_equal": exact[-1],
        "final_text_equal": reference["final_text"] == trial["final_text"],
        "exact_full_trajectory": all(exact),
        "exact_states_including_initial": int(sum(exact)),
        "exact_transition_states": int(sum(exact[1:])),
        "mean_generated_token_agreement": float(statistics.mean(ratios)),
        "final_generated_token_agreement": ratios[-1],
        "first_divergent_state": next((i for i, value in enumerate(exact) if not value), None),
        "per_state": [
            {"state": i, "exact_state": match,
             "generated_token_agreement": agreement}
            for i, (match, agreement) in enumerate(zip(exact, ratios))
        ],
    }


def validate_setup(model):
    """Validate all experiment parameters before running expensive trials."""
    num_layers = len(model.model.layers)
    groups = [tuple(g) for g in EXP8E_LAYER_GROUPS]
    intervals = list(EXP8E_REFRESH_INTERVALS)
    tasks = list(EXP8E_TASK_KEYS)
    seeds = list(EXP8E_SEEDS)

    if not groups or not intervals or not tasks or not seeds:
        raise ValueError("EXP8E groups, intervals, tasks, and seeds must all be nonempty")
    if len(set(groups)) != len(groups):
        raise ValueError("EXP8E_LAYER_GROUPS contains duplicate groups")
    for group in groups:
        if not group or sorted(set(group)) != list(group):
            raise ValueError(f"Layer group must be unique, increasing and nonempty: {group}")
        if any(type(l) is not int or l < 1 or l > num_layers for l in group):
            raise ValueError(f"Invalid 1-based Dream block ID in group {group}; valid 1..{num_layers}")
    if len(set(intervals)) != len(intervals) or any(type(k) is not int or k < 2 for k in intervals):
        raise ValueError("EXP8E_REFRESH_INTERVALS must be distinct integers >= 2")
    if len(set(tasks)) != len(tasks):
        raise ValueError("EXP8E_TASK_KEYS contains duplicate task keys")
    for task in tasks:
        if task != "default" and task not in RQ5_PROMPTS:
            raise ValueError(f"Task {task!r} missing in RQ5_PROMPTS")
    if len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("EXP8E_SEEDS must be distinct nonnegative integers")
    if type(EXP8E_REPEATS) is not int or EXP8E_REPEATS < 1:
        raise ValueError("EXP8E_REPEATS must be >= 1")
    if type(EXP8E_WARMUP_RUNS) is not int or EXP8E_WARMUP_RUNS < 0:
        raise ValueError("EXP8E_WARMUP_RUNS must be >= 0")
    if float(DRIFT_EPSILON) <= 0:
        raise ValueError("DRIFT_EPSILON must be positive")
    return groups, intervals, tasks, seeds


def label_group(group):
    return "L" + "_L".join(str(layer) for layer in group)


def save_trial_checkpoint(rows, trajectory_rows, counter_rows, text_rows):
    """Store intermediate results outside timed generation; rerun starts fresh."""
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "robustness_runs.csv"), index=False)
    pd.DataFrame(trajectory_rows).to_csv(os.path.join(OUTPUT_DIR, "trajectory_fidelity.csv"), index=False)
    pd.DataFrame(counter_rows).to_csv(os.path.join(OUTPUT_DIR, "block_execution_counts.csv"), index=False)
    pd.DataFrame(text_rows).to_csv(os.path.join(OUTPUT_DIR, "generated_texts.csv"), index=False)


def group_summary(df, keys):
    """Summarize only B-C-B pairs whose before/after token trajectories agree."""
    df = df[df["baseline_before_after_exact"]].copy()
    if df.empty:
        return pd.DataFrame()
    return (df.groupby(keys, as_index=False)
            .agg(
                trials=("trial_id", "size"),
                task_seed_trajectories=("task_seed", "nunique"),
                distinct_baseline_trajectories=("baseline_trajectory_sha256", "nunique"),
                median_paired_baseline_ms=("paired_baseline_ms", "median"),
                median_cached_ms=("cached_e2e_ms", "median"),
                median_paired_saving_pct=("paired_saving_pct", "median"),
                mean_paired_saving_pct=("paired_saving_pct", "mean"),
                min_paired_saving_pct=("paired_saving_pct", "min"),
                max_paired_saving_pct=("paired_saving_pct", "max"),
                median_paired_speedup=("paired_speedup", "median"),
                mean_skipped_calls=("skipped_block_calls", "mean"),
                mean_skipped_fraction=("skipped_fraction", "mean"),
                exact_trajectory_trials=("exact_full_trajectory", "sum"),
                exact_final_ids_trials=("final_ids_equal", "sum"),
                min_exact_transition_states=("exact_transition_states", "min"),
                mean_exact_transition_states=("exact_transition_states", "mean"),
                min_final_token_agreement=("final_generated_token_agreement", "min"),
                mean_final_token_agreement=("final_generated_token_agreement", "mean"),
            ).sort_values(keys))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Exp8E requires a CUDA GPU")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 78)
    print("EXP8E: MULTI-BLOCK RESIDUAL CACHING ROBUSTNESS (DREAM, SINGLE GPU)")
    print("=" * 78)
    print(f"Model: {MODEL_NAME} | algorithm={ALGORITHM} | top_p={TOP_P} | top_k={TOP_K}")
    print(f"Generation: {MAX_NEW_TOKENS} tokens / {STEPS} steps")

    set_seed(SEED)
    model, tokenizer = load_dream_model(MODEL_NAME)
    model.eval()
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Model loaded on {device}; expected CUDA")

    groups, intervals, tasks, seeds = validate_setup(model)
    configs = [(group, k) for group in groups for k in intervals]
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Block groups: {groups}")
    print(f"Refresh k: {intervals}")
    print(f"Tasks: {tasks} | seeds: {seeds} | repeats: {EXP8E_REPEATS}")
    planned = len(tasks) * len(seeds) * len(configs) * EXP8E_REPEATS
    print(f"Paired B-C-B trials planned: {planned}")

    # Same sanity check as validated Exp8D: compare Dream's original
    # generation with the HF KV-cache-off control used for all trials.
    first_prompt = PROMPT if tasks[0] == "default" else RQ5_PROMPTS[tasks[0]]
    print("Sanity check: disable HF use_cache for both baseline and cached runs...")
    original_ref = run_generation(model, tokenizer, device, first_prompt, seeds[0])
    disable_hf_kv_cache(model)
    off_ref = run_generation(model, tokenizer, device, first_prompt, seeds[0])
    hf_cache_toggle_preserves_trajectory = compare_trajectory(
        original_ref, off_ref
    )["exact_full_trajectory"]
    print(f"  KV-cache toggle preserves token trajectory: {hf_cache_toggle_preserves_trajectory}")
    if not hf_cache_toggle_preserves_trajectory:
        print("  WARNING: HF KV-cache setting changed baseline tokens. Interpret results relative to HF-cache-off baseline.")

    for _ in range(EXP8E_WARMUP_RUNS):
        run_generation(model, tokenizer, device, first_prompt, seeds[0])
        run_generation(model, tokenizer, device, first_prompt, seeds[0],
                       layers=groups[0], interval=intervals[0])

    rows, trajectory_rows, block_rows, text_rows = [], [], [], []
    trial_id = 0

    for task_index, task in enumerate(tasks):
        prompt = PROMPT if task == "default" else RQ5_PROMPTS[task]
        for seed in seeds:
            for repeat in range(EXP8E_REPEATS):
                # Rotate and reverse order across task/seed/repeat to mitigate
                # scheduling / thermal confounds. This does not change seeds
                # used during generation: run_generation resets them each call.
                shift = (task_index + seed + repeat) % len(configs)
                ordered = configs[shift:] + configs[:shift]
                if repeat % 2:
                    ordered = list(reversed(ordered))

                for group, k in ordered:
                    trial_id += 1
                    label = label_group(group)
                    before = run_generation(model, tokenizer, device, prompt, seed)
                    cached = run_generation(model, tokenizer, device, prompt, seed,
                                            layers=group, interval=k)
                    after = run_generation(model, tokenizer, device, prompt, seed)

                    control = compare_trajectory(before, after)
                    fidelity = compare_trajectory(before, cached)
                    baseline_stable = control["exact_full_trajectory"]
                    paired_baseline_ms = (before["elapsed_ms"] + after["elapsed_ms"]) / 2.0
                    cached_ms = cached["elapsed_ms"]
                    saving_pct = 100.0 * (1.0 - cached_ms / paired_baseline_ms)
                    speedup = paired_baseline_ms / cached_ms

                    counters = cached["counters"]
                    skipped = sum(c["reused"] for c in counters)
                    executed = sum(c["executed"] for c in counters)
                    expected = len(group) * STEPS
                    if skipped + executed != expected:
                        raise RuntimeError("Block skip accounting mismatch")

                    expected_executed_per_block = (STEPS + k - 1) // k
                    for c in counters:
                        if c["executed"] != expected_executed_per_block:
                            raise RuntimeError(
                                f"Layer {c['layer']}: got {c['executed']} refreshes, "
                                f"expected {expected_executed_per_block} for k={k}"
                            )

                    run_key = {"trial_id": trial_id, "task": task, "seed": seed,
                               "repeat": repeat, "task_seed": f"{task}:{seed}",
                               "block_group": label, "num_cached_blocks": len(group),
                               "refresh_interval": k}
                    rows.append({
                        **run_key, "prompt": prompt,
                        "baseline_before_ms": before["elapsed_ms"],
                        "cached_e2e_ms": cached_ms,
                        "baseline_after_ms": after["elapsed_ms"],
                        "paired_baseline_ms": paired_baseline_ms,
                        "baseline_trajectory_sha256": trajectory_fingerprint(before),
                        "paired_saving_pct": saving_pct,
                        "paired_speedup": speedup,
                        "baseline_before_after_exact": baseline_stable,
                        "total_block_calls": expected,
                        "executed_block_calls": executed,
                        "skipped_block_calls": skipped,
                        "skipped_fraction": skipped / expected,
                        **{key: value for key, value in fidelity.items() if key != "per_state"},
                    })
                    for item in fidelity["per_state"]:
                        trajectory_rows.append({**run_key, **item})
                    for item in counters:
                        block_rows.append({**run_key, **item})
                    text_rows.append({
                        **run_key,
                        "reference_text": before["final_text"],
                        "cache_text": cached["final_text"],
                        "baseline_after_text": after["final_text"],
                    })
                    # Persistence is intentionally outside the timed region.
                    save_trial_checkpoint(rows, trajectory_rows, block_rows, text_rows)
                    status = "EXACT" if fidelity["exact_full_trajectory"] else (
                        f"DIFF@state{fidelity['first_divergent_state']}"
                    )
                    control_status = "OK" if baseline_stable else "UNSTABLE"
                    print(
                        f"[{trial_id:03d}/{planned:03d}] {task:<13} seed={seed:<3d} "
                        f"{label:<12} k={k:<2d} "
                        f"B={paired_baseline_ms:8.2f}ms C={cached_ms:8.2f}ms "
                        f"save={saving_pct:+6.2f}% "
                        f"skip={skipped}/{expected} "
                        f"trajectory={status} control={control_status}",
                        flush=True,
                    )

    run_df = pd.DataFrame(rows)
    by_task_seed = group_summary(
        run_df, ["task", "seed", "block_group", "num_cached_blocks", "refresh_interval"]
    )
    by_task = group_summary(
        run_df, ["task", "block_group", "num_cached_blocks", "refresh_interval"]
    )
    overall = group_summary(
        run_df, ["block_group", "num_cached_blocks", "refresh_interval"]
    )
    by_task_seed.to_csv(os.path.join(OUTPUT_DIR, "robustness_by_task_seed.csv"), index=False)
    by_task.to_csv(os.path.join(OUTPUT_DIR, "robustness_by_task.csv"), index=False)
    overall.to_csv(os.path.join(OUTPUT_DIR, "robustness_summary.csv"), index=False)

    metadata = {
        "experiment": "Exp8E cross-task/seed/refresh robustness for Dream block-residual caching",
        "model": MODEL_NAME,
        "max_new_tokens": MAX_NEW_TOKENS, "steps": STEPS,
        "algorithm": ALGORITHM, "temperature": TEMPERATURE,
        "top_p": TOP_P, "top_k": TOP_K,
        "block_groups_1_based": [list(g) for g in groups],
        "refresh_intervals": intervals, "tasks": tasks,
        "prompts": {task: PROMPT if task == "default" else RQ5_PROMPTS[task] for task in tasks},
        "seeds": seeds, "repeats_per_task_seed_config": EXP8E_REPEATS,
        "warmup_runs": EXP8E_WARMUP_RUNS,
        "num_pairs": len(run_df),
        "num_stable_baseline_pairs": int(run_df["baseline_before_after_exact"].sum()),
        "num_exact_trajectory_pairs": int(run_df["exact_full_trajectory"].sum()),
        "hf_kv_cache_toggle_preserves_trajectory_first_task_seed": hf_cache_toggle_preserves_trajectory,
        "hf_use_cache": bool(model.config.use_cache),
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "benchmark": "Synchronized B-C-B baseline/cache/baseline full generation per trial; "
                     "paired baseline=(before+after)/2; saving=1-cached/paired_baseline. "
                     "Baseline token trajectories must match within pair for aggregate summaries.",
        "cache_method": "Each patched Dream block stores FP32 residual output-input at refresh calls. "
                        "At reuse, original block.forward is bypassed and current_input + cached_residual returned; "
                        "all selected blocks use the same periodic k. Caches reset each generation.",
        "limitations": "One model, single GPU; generated token trajectories evaluated, not probability/logit equivalence; "
                       "no edge-cloud network; tested tasks/seeds only; latency includes generation + instrumentation "
                       "and excludes report generation; no proof of quality preservation beyond tested trajectories.",
    }
    with open(os.path.join(OUTPUT_DIR, "experiment_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print("EXP8E: POOLED ROBUSTNESS SUMMARY (STABLE BASELINE PAIRS ONLY)")
    print("=" * 78)
    if overall.empty:
        print("No stable baseline pairs; inspect robustness_runs.csv")
    else:
        cols = ["block_group", "refresh_interval", "trials", "task_seed_trajectories",
                "distinct_baseline_trajectories",
                "median_paired_saving_pct", "min_paired_saving_pct", "median_paired_speedup",
                "exact_trajectory_trials", "exact_final_ids_trials", "min_exact_transition_states"]
        print(overall[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nBY TASK:")
    if not by_task.empty:
        cols = ["task", "block_group", "refresh_interval", "trials", "median_paired_saving_pct",
                "exact_trajectory_trials", "min_exact_transition_states"]
        print(by_task[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSaved to:", OUTPUT_DIR)
    print("NOTE: Exact token-trajectory agreement does NOT imply exactly equal logits or hidden states.")
    print("NOTE: Results measure single-GPU generation E2E, NOT split edge-cloud network latency.")
    distinct_by_task = run_df.groupby("task")["baseline_trajectory_sha256"].nunique()
    print("\nDistinct observed token trajectories per task (across seeds):")
    print(distinct_by_task.to_string())
    if (distinct_by_task < len(seeds)).any():
        print("NOTE: Some seeds produced identical token trajectories; "
              "multiple seed values do not necessarily mean independent generation outcomes.")
    if not run_df["baseline_before_after_exact"].all():
        print("WARNING: Some baseline pairs were unstable; excluded from aggregate summaries.")
    if not run_df["exact_full_trajectory"].all():
        print("WARNING: Some approximate-cache runs changed denoising token trajectories.")


if __name__ == "__main__":
    main()
