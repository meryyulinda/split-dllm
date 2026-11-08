"""Exp8D: measured multi-block residual reuse during real Dream generation.

Run from dasr1/ with the Dream environment:
    python experiments/exp8d_multi_block_residual_cache.py

Add EXP8D_* parameters to configs/experiment_config.py (see companion snippet).

This experiment:
- truly bypasses block.forward() on reuse calls (no oracle output substitution),
- caches the total residual R_l = H_l(out) - H_l(in) for each selected block,
- compares baseline/cache/baseline sequentially for every trial (B-C-B),
- measures full-generation wall-clock GPU-synchronized time,
- compares all token states and final decoded text against local baseline,
- supports multiple prompts and seeds without assuming quality remains exact.

Not KV caching. Not edge-cloud split. No performance claims until hardware runs.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import json
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
    EXP8D_LAYER_GROUPS, EXP8D_REFRESH_INTERVALS,
    EXP8D_TASK_KEYS, EXP8D_SEEDS, EXP8D_REPEATS, EXP8D_WARMUP_RUNS,
)
from src.model_loader import load_dream_model
from src.generation import generate_with_token_history
from src.instrumentation import create_token_history_hook
from src.utils import set_seed

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "exp8d")


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
            raise RuntimeError("Exp8D requires HF use_cache=False and no past_key_value")
        if kwargs.get("output_attentions", False):
            raise RuntimeError("Exp8D requires output_attentions=False")

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
    n_layers = len(model.model.layers)
    groups = [tuple(g) for g in EXP8D_LAYER_GROUPS]
    intervals = list(EXP8D_REFRESH_INTERVALS)
    tasks = list(EXP8D_TASK_KEYS)
    seeds = list(EXP8D_SEEDS)
    if not groups or not intervals or not tasks or not seeds:
        raise ValueError("EXP8D layer groups/intervals/tasks/seeds must be non-empty")
    for group in groups:
        if not group or len(set(group)) != len(group):
            raise ValueError(f"Invalid or duplicate block group: {group}")
        if sorted(group) != list(group):
            raise ValueError(f"Keep layer group in increasing order: {group}")
        if any(not isinstance(l, int) or l < 1 or l > n_layers for l in group):
            raise ValueError(f"Blocks must be 1-based layer IDs in [1,{n_layers}]")
    if len(set(groups)) != len(groups):
        raise ValueError("EXP8D_LAYER_GROUPS contains duplicate groups")
    if any(not isinstance(k, int) or k < 2 for k in intervals):
        raise ValueError("EXP8D_REFRESH_INTERVALS must be integers >= 2")
    for task in tasks:
        if task != "default" and task not in RQ5_PROMPTS:
            raise ValueError(f"Unknown task key {task!r}; use 'default' or RQ5_PROMPTS keys")
    if not all(isinstance(s, int) for s in seeds):
        raise ValueError("EXP8D_SEEDS must be integers")
    if not isinstance(EXP8D_REPEATS, int) or EXP8D_REPEATS < 1:
        raise ValueError("EXP8D_REPEATS must be >= 1")
    if not isinstance(EXP8D_WARMUP_RUNS, int) or EXP8D_WARMUP_RUNS < 0:
        raise ValueError("EXP8D_WARMUP_RUNS must be >= 0")
    if not isinstance(DRIFT_EPSILON, float) or DRIFT_EPSILON <= 0:
        raise ValueError("Invalid DRIFT_EPSILON")
    return groups, intervals, tasks, seeds


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Exp8D requires a CUDA device")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 78)
    print("EXP8D: MULTI-BLOCK RESIDUAL CACHING (MEASURED SINGLE-GPU INFERENCE)")
    print("=" * 78)
    print(f"Model: {MODEL_NAME} | alg={ALGORITHM}, TOP_P={TOP_P}, TOP_K={TOP_K}")
    print(f"Generation: {MAX_NEW_TOKENS} generated tokens / {STEPS} steps; default seed={SEED}")

    set_seed(SEED)
    model, tokenizer = load_dream_model(MODEL_NAME)
    model.eval()
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Loaded Dream on {device}; CUDA required")

    # Ensure full-model reference under original HF settings is equal to the
    # baseline with HF auxiliary KV caching disabled (one prompt and seed).
    tasks = list(EXP8D_TASK_KEYS)
    first_prompt = PROMPT if tasks[0] == "default" else RQ5_PROMPTS[tasks[0]]
    print("Sanity check: HF KV-cache toggle...")
    original_reference = run_generation(model, tokenizer, device, first_prompt, SEED)
    disable_hf_kv_cache(model)
    off_reference = run_generation(model, tokenizer, device, first_prompt, SEED)
    hf_toggle_exact = compare_trajectory(original_reference, off_reference)["exact_full_trajectory"]
    print(f"  HF cache toggle preserves token trajectory: {hf_toggle_exact}")
    if not hf_toggle_exact:
        print("  WARNING: Changed HF forward config affects baseline generation; "
              "Exp8D controls are still internally matched.")

    groups, intervals, tasks, seeds = validate_setup(model)
    configs = [(group, k) for group in groups for k in intervals]
    print(f"GPU: {torch.cuda.get_device_name(device)} | block groups: {groups}")
    print(f"Refresh intervals: {intervals} | tasks: {tasks} | seeds: {seeds}")
    print(f"Paired trials per config: {EXP8D_REPEATS} | warmup generations: {EXP8D_WARMUP_RUNS}")
    print("Per trial: baseline_before -> multi_block_cache -> baseline_after")

    for _ in range(EXP8D_WARMUP_RUNS):
        run_generation(model, tokenizer, device, first_prompt, seeds[0])

    rows = []
    state_rows = []
    per_block_rows = []
    text_rows = []
    trial_id = 0
    for task in tasks:
        prompt = PROMPT if task == "default" else RQ5_PROMPTS[task]
        for seed in seeds:
            for repeat in range(EXP8D_REPEATS):
                # Alternate config order each repeat to limit systematic
                # scheduling/thermal advantage for any single configuration.
                ordered = configs if repeat % 2 == 0 else list(reversed(configs))
                for group, k in ordered:
                    trial_id += 1
                    label = "L" + "_L".join(str(v) for v in group)
                    before = run_generation(model, tokenizer, device, prompt, seed)
                    candidate = run_generation(
                        model, tokenizer, device, prompt, seed,
                        layers=group, interval=k,
                    )
                    after = run_generation(model, tokenizer, device, prompt, seed)

                    before_after = compare_trajectory(before, after)
                    fidelity = compare_trajectory(before, candidate)
                    ctrl_ok = before_after["exact_full_trajectory"]
                    baseline_ms = (before["elapsed_ms"] + after["elapsed_ms"]) / 2
                    cached_ms = candidate["elapsed_ms"]
                    speedup = baseline_ms / cached_ms
                    saving = (1 - cached_ms / baseline_ms) * 100
                    counters = candidate["counters"]
                    total_skipped = sum(item["reused"] for item in counters)
                    total_executed = sum(item["executed"] for item in counters)
                    assert total_skipped + total_executed == STEPS * len(group)

                    rows.append({
                        "trial_id": trial_id, "task": task, "seed": seed,
                        "repeat": repeat, "block_group": label,
                        "num_cached_blocks": len(group), "refresh_interval": k,
                        "baseline_before_ms": before["elapsed_ms"],
                        "cached_e2e_ms": cached_ms,
                        "baseline_after_ms": after["elapsed_ms"],
                        "paired_baseline_ms": baseline_ms,
                        "paired_speedup": speedup,
                        "paired_saving_pct": saving,
                        "baseline_before_after_exact": ctrl_ok,
                        "total_block_calls": STEPS * len(group),
                        "executed_block_calls": total_executed,
                        "skipped_block_calls": total_skipped,
                        "skipped_fraction": total_skipped / (STEPS * len(group)),
                        **{key: val for key, val in fidelity.items() if key != "per_state"},
                    })
                    for item in fidelity["per_state"]:
                        state_rows.append({"trial_id": trial_id, "task": task,
                                           "seed": seed, "block_group": label,
                                           "refresh_interval": k, **item})
                    for item in counters:
                        per_block_rows.append({"trial_id": trial_id, "task": task,
                                               "seed": seed, "block_group": label,
                                               "refresh_interval": k, **item})
                    text_rows.append({
                        "trial_id": trial_id, "task": task, "seed": seed,
                        "block_group": label, "refresh_interval": k,
                        "reference_text": before["final_text"],
                        "cache_text": candidate["final_text"],
                    })
                    status = "EXACT" if fidelity["exact_full_trajectory"] else "DIFF"
                    print(
                        f"[{trial_id:03d}] {task:<12} seed={seed} {label:<13} "
                        f"k={k} | B={baseline_ms:8.2f}ms C={cached_ms:8.2f}ms "
                        f"save={saving:+6.2f}% skip={total_skipped:3d}/"
                        f"{STEPS*len(group):3d} trajectory={status} "
                        f"baseline_stable={ctrl_ok}",
                        flush=True,
                    )

    run_df = pd.DataFrame(rows)
    # Use paired ratio per trial, not ratio of independently pooled medians.
    summary = (run_df.groupby(["task", "block_group", "num_cached_blocks", "refresh_interval"],
                              as_index=False)
               .agg(
                   trials=("trial_id", "size"),
                   median_paired_baseline_ms=("paired_baseline_ms", "median"),
                   median_cached_ms=("cached_e2e_ms", "median"),
                   median_paired_saving_pct=("paired_saving_pct", "median"),
                   mean_paired_saving_pct=("paired_saving_pct", "mean"),
                   median_paired_speedup=("paired_speedup", "median"),
                   mean_skipped_calls=("skipped_block_calls", "mean"),
                   mean_skipped_fraction=("skipped_fraction", "mean"),
                   exact_final_ids_trials=("final_ids_equal", "sum"),
                   exact_trajectory_trials=("exact_full_trajectory", "sum"),
                   baseline_stable_trials=("baseline_before_after_exact", "sum"),
                   mean_generated_token_agreement=("mean_generated_token_agreement", "mean"),
                   mean_exact_transition_states=("exact_transition_states", "mean"),
               ))
    summary = summary.sort_values(["task", "num_cached_blocks", "block_group", "refresh_interval"])

    output_paths = {
        "runs": os.path.join(OUTPUT_DIR, "multi_block_runs.csv"),
        "summary": os.path.join(OUTPUT_DIR, "multi_block_summary.csv"),
        "fidelity": os.path.join(OUTPUT_DIR, "multi_block_trajectory_fidelity.csv"),
        "counters": os.path.join(OUTPUT_DIR, "per_block_execution_counts.csv"),
        "texts": os.path.join(OUTPUT_DIR, "generated_texts.csv"),
        "metadata": os.path.join(OUTPUT_DIR, "experiment_metadata.json"),
    }
    run_df.to_csv(output_paths["runs"], index=False)
    summary.to_csv(output_paths["summary"], index=False)
    pd.DataFrame(state_rows).to_csv(output_paths["fidelity"], index=False)
    pd.DataFrame(per_block_rows).to_csv(output_paths["counters"], index=False)
    pd.DataFrame(text_rows).to_csv(output_paths["texts"], index=False)

    metadata = {
        "experiment": "Exp8D multi-block residual caching with periodic refresh",
        "model": MODEL_NAME, "default_seed": SEED, "task_keys": tasks, "seeds": seeds,
        "block_groups_1_based": [list(group) for group in groups],
        "refresh_intervals": intervals, "repeats": EXP8D_REPEATS,
        "max_new_tokens": MAX_NEW_TOKENS, "steps": STEPS,
        "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
        "algorithm": ALGORITHM, "prompt_for_default_task": PROMPT,
        "gpu": torch.cuda.get_device_name(device), "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hf_use_cache": bool(model.config.use_cache),
        "hf_cache_toggle_preserves_baseline_trajectory": hf_toggle_exact,
        "benchmark_method": (
            "Warmup; for every task, seed, repeat and config, synchronized "
            "B-C-B (baseline-cache-baseline) generation calls. "
            "Latency includes token-history hook and cache logic, excludes offline CSV processing. "
            "Paired baseline is average(before, after); report median paired saving."
        ),
        "cache_method": (
            "For each selected layer, refresh runs original decoder block and stores "
            "FP32 residual output-input. Reuse bypasses block.forward() and computes "
            "current_input + cached_residual. Earlier cached layers may affect the "
            "inputs to later refreshed layers. Built-in HF KV cache disabled for both modes."
        ),
        "caveats": (
            "Actual single-GPU whole-generation timings, not edge-cloud measured. "
            "Approximate internal activations despite exact token trajectory in some runs. "
            "Exact sequence agreement over tested prompts/seeds is not a general quality guarantee. "
            "CUDA overhead, synchronization, sampling, and system load affect latency."
        ),
    }
    with open(output_paths["metadata"], "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print("EXP8D SUMMARY — PAIRED MEDIAN SAVINGS AND FULL-TRAJECTORY FIDELITY")
    print("=" * 78)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not run_df["baseline_before_after_exact"].all():
        print("WARNING: Some baseline-before/after token trajectories differ. "
              "Check reproducibility before interpreting cache fidelity.")
    print("\nSaved output files:")
    for f in output_paths.values():
        print("  ", f)
    print("\nNOTE: No edge-cloud communication measured; cache-policy gains are "
          "single-GPU results only.")


if __name__ == "__main__":
    main()
