"""Exp8F — controlled scale/workload test of Dream block-residual caching.

Run from dasr1/:
    python experiments/exp8f_block_cache_scaling.py

Design:
- Same synthetic long-prompt construction and direct Dream generation API as Exp9B.
- Four workloads: short/long prompt, 64/64 and 256/256 generation.
- Nested 3-/6-/8-layer groups; periodic refresh k=2 and k=4.
- Timed B-C-B with output_history=False and CUDA sync, NO trajectory hooks.
- Separate untimed output_history=True runs to evaluate token trajectories.
- Exact counts of actually skipped Dream decoder block forwards.
- Manual answer-quality review CSV; no invented quality scores.

Important: Block-residual caching is approximate, not exact. This experiment
uses HF use_cache=False for BOTH baseline and cached runs because skipped
Dream layers must not return stale HF KV data. Exp9B's official dLLM-Cache
run reported HF use_cache=True; these are not automatically matched environments.
This is single GPU only; communication time is NOT measured.
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
from contextlib import ExitStack
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import torch
import transformers

import configs.experiment_config as ec
from configs.experiment_config import (
    MODEL_NAME, SEED, TEMPERATURE, TOP_P, TOP_K, ALGORITHM, PROMPT, RQ5_PROMPTS,
)
from src.model_loader import load_dream_model
from src.utils import set_seed


def setting(name, default):
    return getattr(ec, name, default)


# Defaults deliberately start with one task/seed/repeat to keep the first run bounded.
# Add EXP8F_* overrides to configs/experiment_config.py (companion snippet).
TASKS = list(setting("EXP8F_TASK_KEYS", ["reasoning"]))
SEEDS = list(setting("EXP8F_SEEDS", [SEED]))
GENERATION_CASES = [tuple(map(int, c)) for c in setting(
    "EXP8F_GENERATION_CASES", [(64, 64), (256, 256)]
)]
PROMPT_MODES = list(setting("EXP8F_PROMPT_MODES", ["short", "long"]))
LONG_PROMPT_TARGET_TOKENS = int(setting(
    "EXP8F_LONG_PROMPT_TARGET_TOKENS",
    setting("EXP9B_LONG_PROMPT_TARGET_TOKENS", 1024),
))
LAYER_GROUPS = [tuple(map(int, g)) for g in setting("EXP8F_LAYER_GROUPS", [
    [8, 20, 24],
    [4, 8, 12, 16, 20, 24],
    [4, 8, 12, 16, 18, 20, 22, 24],
])]
REFRESH_INTERVALS = [int(k) for k in setting("EXP8F_REFRESH_INTERVALS", [2, 4])]
REPEATS = int(setting("EXP8F_REPEATS", 1))
WARMUP_RUNS = int(setting("EXP8F_WARMUP_RUNS", 1))
RUN_FIDELITY = bool(setting("EXP8F_RUN_FIDELITY", True))
VALIDATE_HF_TOGGLE = bool(setting("EXP8F_VALIDATE_HF_TOGGLE", True))
OUTPUT_DIR = Path(PROJECT_ROOT) / "outputs" / "exp8f"


def validate_configuration():
    if not torch.cuda.is_available():
        raise RuntimeError("Exp8F needs CUDA for synchronized GPU timing")
    if "Dream" not in MODEL_NAME:
        raise ValueError("Exp8F is specifically for the Dream model API")
    if not TASKS or not SEEDS or not GENERATION_CASES or not PROMPT_MODES:
        raise ValueError("Tasks, seeds, generation cases, and modes must be nonempty")
    if not LAYER_GROUPS or not REFRESH_INTERVALS:
        raise ValueError("At least one layer group and interval are needed")
    if REPEATS < 1 or WARMUP_RUNS < 0 or LONG_PROMPT_TARGET_TOKENS < 1:
        raise ValueError("Invalid repeats/warmups/target prompt tokens")
    for task in TASKS:
        if task != "default" and task not in RQ5_PROMPTS:
            raise KeyError(f"Task {task!r} not in RQ5_PROMPTS")
    for prompt_mode in PROMPT_MODES:
        if prompt_mode not in ("short", "long"):
            raise ValueError(f"Unknown prompt mode: {prompt_mode!r}")
    for n, steps in GENERATION_CASES:
        if n < 1 or steps < 1:
            raise ValueError(f"Invalid generation case {(n, steps)}")
    if any(k < 2 for k in REFRESH_INTERVALS):
        raise ValueError("Periodic refresh k must be >= 2")
    if len(set(LAYER_GROUPS)) != len(LAYER_GROUPS):
        raise ValueError("Duplicate layer group")
    for group in LAYER_GROUPS:
        if not group or sorted(set(group)) != list(group):
            raise ValueError(f"Each group must be nonempty unique sorted layers: {group}")
    if len(set(SEEDS)) != len(SEEDS):
        raise ValueError("Duplicate seed")
    if any(not isinstance(s, int) for s in SEEDS):
        raise ValueError("Seeds must be ints")


def disable_hf_decoder_cache(model):
    """Ensure old-KV outputs cannot leak across approximated block calls."""
    for owner_name, owner in (("model", model), ("model.model", model.model)):
        conf = getattr(owner, "config", None)
        if conf is None:
            raise RuntimeError(f"{owner_name}.config is missing")
        conf.use_cache = False
        conf.output_attentions = False
    gen_conf = getattr(model, "generation_config", None)
    if gen_conf is not None:
        gen_conf.use_cache = False
        gen_conf.output_attentions = False
    if model.config.use_cache or model.model.config.use_cache:
        raise RuntimeError("Could not disable built-in HF KV caching")


class BlockResidualCache:
    """Caches R_l = block(x)-x. Reuse returns x_current + R_old.

    On reuse, original block.forward is NOT called. Per-block residual is
    refreshed from the actual, possibly approximated upstream input.
    """
    def __init__(self, model, layer_id, k):
        self.layer_id = int(layer_id)  # one-based
        self.k = int(k)
        self.module = model.model.layers[self.layer_id - 1]
        self._original = None
        self._residual_f32 = None
        self._last_refresh = -1
        self._returns_tuple = None
        self._aux = ()
        self.calls = 0
        self.executed = 0
        self.skipped = 0

    def _patched_forward(self, *args, **kwargs):
        if kwargs.get("use_cache", False) or kwargs.get("past_key_value") is not None:
            raise RuntimeError("Exp8F does not support HF use_cache/past_key_value")
        if kwargs.get("output_attentions", False):
            raise RuntimeError("Exp8F requires output_attentions=False")

        x = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise RuntimeError("Expected Dream block input tensor [batch, seq, hidden]")
        index = self.calls
        self.calls += 1
        refresh = self._residual_f32 is None or index - self._last_refresh >= self.k

        if not refresh:
            if self._residual_f32.shape != x.shape or self._residual_f32.device != x.device:
                raise RuntimeError("Cached residual shape/device mismatch")
            # True compute skipping: DO NOT invoke self._original here.
            y = (x.float() + self._residual_f32).to(dtype=x.dtype)
            self.skipped += 1
            return (y, *self._aux) if self._returns_tuple else y

        original_input_f32 = x.detach().float().clone()
        output = self._original(*args, **kwargs)
        y = output[0] if isinstance(output, tuple) else output
        if not isinstance(y, torch.Tensor) or y.shape != x.shape:
            raise RuntimeError("Unexpected Dream block output shape/type")
        aux = output[1:] if isinstance(output, tuple) else ()
        if any(a is not None for a in aux):
            raise RuntimeError(
                "Block returned non-None auxiliary cache/attention output. "
                "Exp8F refuses to reuse stale auxiliary states."
            )
        self._returns_tuple = isinstance(output, tuple)
        self._aux = aux
        self._residual_f32 = y.detach().float().sub(original_input_f32)
        self._last_refresh = index
        self.executed += 1
        return output

    def __enter__(self):
        if self._original is not None:
            raise RuntimeError("Duplicate block patch")
        self._original = self.module.forward
        self.module.forward = self._patched_forward
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self._original is not None:
            self.module.forward = self._original
            self._original = None
        return False

    def stats(self):
        if self.calls != self.executed + self.skipped:
            raise RuntimeError("Inconsistent executed/skipped counts")
        return {"layer": self.layer_id, "calls": self.calls,
                "executed": self.executed, "skipped": self.skipped}


def prompt_token_length(tokenizer, prompt):
    # EXACT same length and long-prompt formatting logic as Exp9B.
    return len(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        tokenize=True,
    ))


def build_prompt(tokenizer, task, mode):
    original = PROMPT if task == "default" else RQ5_PROMPTS[task]
    if mode == "short":
        return original
    if mode != "long":
        raise ValueError(mode)
    # EXACT same synthetic scaling stress-test background as Exp9B.
    notes = []
    block = (
        "This is a synthetic background reference note for a controlled "
        "long-context throughput experiment. It is not evidence for the "
        "user's final question. It contains logistics terminology, "
        "warehouse indices, equipment maintenance schedules, and "
        "unrelated status descriptions."
    )
    for idx in range(1, 240):
        notes.append(f"Reference entry {idx:03d}: {block}")
        candidate = (
            "The following reference appendix is irrelevant to the final task. "
            "Ignore it when forming the answer.\n\n" +
            "\n".join(notes) +
            "\n\nFinal user task (answer this):\n" + original
        )
        if prompt_token_length(tokenizer, candidate) >= LONG_PROMPT_TARGET_TOKENS:
            return candidate
    raise RuntimeError("Synthetic prompt could not reach requested token length")


def prepare_input(tokenizer, prompt, device):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    mask = torch.ones_like(ids) if mask is None else mask.to(device)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise RuntimeError("Expected single-sequence batch")
    return ids, mask


def generate(model, ids, mask, n, steps, seed, device,
             group=(), k=None, capture_trajectory=False):
    """Direct Dream inference. Timed runs use capture_trajectory=False.

    Cache install happens before timer; per-call wrapper overhead is included.
    One new cache object per generation prevents state leakage between trials.
    """
    set_seed(int(seed))
    caches = []
    if group:
        if k is None or k < 2:
            raise ValueError("For caching specify k>=2")
        caches = [BlockResidualCache(model, layer_id, k) for layer_id in group]

    with ExitStack() as stack:
        for cache in caches:
            stack.enter_context(cache)
        set_seed(int(seed))
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            result = model.diffusion_generate(
                ids, attention_mask=mask,
                max_new_tokens=int(n), steps=int(steps),
                temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                alg=ALGORITHM, output_history=bool(capture_trajectory),
                return_dict_in_generate=True,
            )
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not hasattr(result, "sequences"):
            raise RuntimeError("Dream diffusion_generate did not return .sequences")
        generated_ids = result.sequences[0, ids.shape[1]:].detach().cpu().long().clone()
        history = None
        if capture_trajectory:
            raw = getattr(result, "history", None)
            if raw is None or len(raw) != int(steps):
                raise RuntimeError(
                    f"Expected Dream history length {steps}; got "
                    f"{None if raw is None else len(raw)}"
                )
            history = [x[0].detach().cpu().long().clone() for x in raw]
            if not torch.equal(history[-1][ids.shape[1]:], generated_ids):
                raise RuntimeError("Final history state does not match .sequences")
        counters = [cache.stats() for cache in caches]
        for c in counters:
            if c["calls"] != int(steps):
                raise RuntimeError(
                    f"Layer {c['layer']} called {c['calls']} times, expected {steps}"
                )
        return {"ms": elapsed_ms, "generated_ids": generated_ids,
                "history": history, "counters": counters}


def compare_histories(base, cached, prompt_tokens):
    left, right = base["history"], cached["history"]
    if left is None or right is None or len(left) != len(right):
        raise RuntimeError("Missing/mismatched full history")
    exact = [bool(torch.equal(a, b)) for a, b in zip(left, right)]
    generated_match = [float((a[prompt_tokens:] == b[prompt_tokens:]).float().mean())
                       for a, b in zip(left, right)]
    return {
        "exact_trajectory": bool(all(exact)),
        "exact_transition_states": int(sum(exact)),
        "first_divergent_step": next((i + 1 for i, ok in enumerate(exact) if not ok), None),
        "mean_generated_token_agreement": statistics.mean(generated_match),
        "final_generated_token_agreement": generated_match[-1],
        "exact_final_ids": bool(torch.equal(base["generated_ids"], cached["generated_ids"])),
    }


def trajectory_hash(history):
    hasher = hashlib.sha256()
    for tokens in history:
        hasher.update(tokens.numpy().tobytes())
    return hasher.hexdigest()


def save_csv(records, filename):
    if records:
        pd.DataFrame(records).to_csv(OUTPUT_DIR / filename, index=False)


def print_summary(timing_rows, fidelity_rows):
    if timing_rows:
        df = pd.DataFrame(timing_rows)
        stable = df[df["baseline_pair_stable"]].copy()
        if len(stable):
            group_cols = ["prompt_mode", "new_tokens", "steps",
                          "block_group", "num_cached_blocks", "refresh_interval"]
            summary = stable.groupby(group_cols, dropna=False, sort=True).agg(
                trials=("trial_id", "size"),
                median_prompt_tokens=("prompt_tokens", "median"),
                median_paired_baseline_ms=("paired_baseline_ms", "median"),
                median_cached_ms=("cache_ms", "median"),
                median_paired_saving_pct=("saving_pct", "median"),
                min_paired_saving_pct=("saving_pct", "min"),
                mean_skipped_calls=("skipped_calls", "mean"),
                mean_skipped_fraction=("skipped_fraction", "mean"),
                exact_timed_final_trials=("exact_timed_final", "sum"),
            ).reset_index()
            summary.to_csv(OUTPUT_DIR / "timing_summary_by_workload_group.csv", index=False)
            if fidelity_rows:
                f = pd.DataFrame(fidelity_rows)
                f_summary = f.groupby(group_cols, dropna=False, sort=True).agg(
                    fidelity_cases=("trial_id", "size"),
                    exact_trajectory_cases=("exact_trajectory", "sum"),
                    exact_final_ids_cases=("exact_final_ids", "sum"),
                    baseline_history_control_cases=("baseline_history_control", "sum"),
                ).reset_index()
                f_summary.to_csv(OUTPUT_DIR / "fidelity_summary_by_workload_group.csv", index=False)
                summary = summary.merge(f_summary, on=group_cols, how="left")
            summary.to_csv(OUTPUT_DIR / "combined_summary.csv", index=False)
            print("\n" + "=" * 82)
            print("EXP8F SUMMARY (STABLE B-C-B BASELINES ONLY)")
            print("=" * 82)
            display_cols = ["prompt_mode", "new_tokens", "steps", "num_cached_blocks",
                            "block_group", "refresh_interval", "trials",
                            "median_paired_saving_pct", "exact_trajectory_cases",
                            "fidelity_cases"]
            display_cols = [c for c in display_cols if c in summary]
            print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        else:
            print("No stable B-C-B baseline pairs; examine timing_runs.csv")
    if fidelity_rows:
        print("\nFidelity (independent untimed runs):")
        f = pd.DataFrame(fidelity_rows)
        print(f.groupby(["prompt_mode", "new_tokens", "steps", "num_cached_blocks",
                         "refresh_interval"], dropna=False).agg(
            cases=("trial_id", "size"), exact=("exact_trajectory", "sum")
        ).reset_index().to_string(index=False))


def main():
    validate_configuration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model, tokenizer = load_dream_model(MODEL_NAME)
    model.eval()
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Expected Dream model.model.layers")
    n_layers = len(model.model.layers)
    if any(any(l < 1 or l > n_layers for l in g) for g in LAYER_GROUPS):
        raise ValueError(f"Layer groups not compatible with {n_layers} layers")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Dream loaded on {device}; CUDA required")

    print("=" * 82)
    print("EXP8F: BLOCK RESIDUAL CACHING SCALING (DREAM, SINGLE GPU)")
    print("=" * 82)
    print(f"Model={MODEL_NAME} | GPU={torch.cuda.get_device_name(device)} | layers={n_layers}")
    print(f"Tasks={TASKS} seeds={SEEDS}; generation={GENERATION_CASES}, prompt={PROMPT_MODES}")
    print(f"Groups={LAYER_GROUPS}; k={REFRESH_INTERVALS}; repeats={REPEATS}")
    print("TIMING: direct diffusion_generate(output_history=False), CUDA-synced B-C-B")
    print("FIDELITY: separate untimed output_history=True, with baseline-history control")
    print("Exp9B difference: HF use_cache=True there; Exp8F must use False for safe skip")

    if VALIDATE_HF_TOGGLE:
        # Check original baseline and HF-off baseline on one short 64/64 workload.
        prompt = build_prompt(tokenizer, TASKS[0], "short")
        ids0, attn0 = prepare_input(tokenizer, prompt, device)
        print("Sanity check: original HF decoder-cache setting versus HF-off...", flush=True)
        original = generate(model, ids0, attn0, 64, 64, SEEDS[0], device,
                            capture_trajectory=True)
        disable_hf_decoder_cache(model)
        off = generate(model, ids0, attn0, 64, 64, SEEDS[0], device,
                       capture_trajectory=True)
        toggle_comparison = compare_histories(original, off, ids0.shape[1])
        print("  HF-toggle exact token trajectory:", toggle_comparison["exact_trajectory"],
              flush=True)
    else:
        disable_hf_decoder_cache(model)
        toggle_comparison = None

    # Warmup baseline and one cached config for each generation length.
    first_prompt = build_prompt(tokenizer, TASKS[0], "short")
    warm_ids, warm_mask = prepare_input(tokenizer, first_prompt, device)
    if WARMUP_RUNS:
        for n, steps in GENERATION_CASES:
            for _ in range(WARMUP_RUNS):
                generate(model, warm_ids, warm_mask, n, steps, SEEDS[0], device)
                generate(model, warm_ids, warm_mask, n, steps, SEEDS[0], device,
                         group=LAYER_GROUPS[0], k=REFRESH_INTERVALS[0])
        print("Warmup done", flush=True)

    timing_rows, fidelity_rows, text_rows, count_rows = [], [], [], []
    trial_id = 0
    total_trials = len(TASKS) * len(SEEDS) * len(PROMPT_MODES) * \
        len(GENERATION_CASES) * len(LAYER_GROUPS) * len(REFRESH_INTERVALS) * REPEATS

    # Cache the UNTIMED baseline history per task/workload/seed to reduce total GPU work.
    # Every config's cached history still gets its own reset/seeded generation.
    for mode in PROMPT_MODES:
        for n, steps in GENERATION_CASES:
            for task in TASKS:
                prompt = build_prompt(tokenizer, task, mode)
                ids, mask = prepare_input(tokenizer, prompt, device)
                prompt_tokens = int(ids.shape[1])
                print(f"\nWORKLOAD mode={mode} {n}/{steps} task={task} prompt_tokens={prompt_tokens}",
                      flush=True)
                for seed in SEEDS:
                    baseline_history = None
                    history_control_exact = None
                    if RUN_FIDELITY:
                        baseline_history = generate(model, ids, mask, n, steps, seed, device,
                                                    capture_trajectory=True)
                        baseline_control = generate(model, ids, mask, n, steps, seed, device,
                                                    capture_trajectory=True)
                        history_control_exact = compare_histories(
                            baseline_history, baseline_control, prompt_tokens
                        )["exact_trajectory"]
                        print(f"  Untimed baseline history stable={history_control_exact}",
                              flush=True)

                    configs = [(group, k) for group in LAYER_GROUPS
                               for k in REFRESH_INTERVALS]
                    for repeat in range(REPEATS):
                        configs_this_repeat = configs if repeat % 2 == 0 else list(reversed(configs))
                        for group, k in configs_this_repeat:
                            trial_id += 1
                            group_label = "L" + "_L".join(map(str, group))
                            before = generate(model, ids, mask, n, steps, seed, device)
                            cached = generate(model, ids, mask, n, steps, seed, device,
                                              group=group, k=k)
                            after = generate(model, ids, mask, n, steps, seed, device)
                            baseline_stable = torch.equal(
                                before["generated_ids"], after["generated_ids"]
                            )
                            pair_ms = (before["ms"] + after["ms"]) / 2.0
                            saving = (1.0 - cached["ms"] / pair_ms) * 100.0
                            skip_calls = sum(c["skipped"] for c in cached["counters"])
                            executed_calls = sum(c["executed"] for c in cached["counters"])
                            possible = int(steps) * len(group)
                            if skip_calls + executed_calls != possible:
                                raise RuntimeError("Block accounting mismatch")

                            row = {
                                "trial_id": trial_id, "task": task, "prompt_mode": mode,
                                "new_tokens": int(n), "steps": int(steps),
                                "prompt_tokens": prompt_tokens, "seed": seed, "repeat": repeat,
                                "block_group": group_label, "num_cached_blocks": len(group),
                                "refresh_interval": k, "hf_use_cache": False,
                                "baseline_before_ms": before["ms"], "cache_ms": cached["ms"],
                                "baseline_after_ms": after["ms"],
                                "paired_baseline_ms": pair_ms, "saving_pct": saving,
                                "paired_speedup": pair_ms / cached["ms"],
                                "baseline_pair_stable": baseline_stable,
                                "exact_timed_final": torch.equal(before["generated_ids"],
                                                                 cached["generated_ids"]),
                                "skipped_calls": skip_calls,
                                "executed_calls": executed_calls,
                                "skipped_fraction": skip_calls / possible,
                            }
                            timing_rows.append(row)
                            save_csv(timing_rows, "timing_runs.csv")
                            for c in cached["counters"]:
                                count_rows.append({"trial_id": trial_id, "task": task,
                                                   "prompt_mode": mode, "new_tokens": n,
                                                   "steps": steps, "group": group_label,
                                                   "refresh_interval": k, **c})
                            save_csv(count_rows, "per_layer_execution_counts.csv")
                            print(
                                f"[{trial_id:03d}/{total_trials:03d}] {task:12s} {mode:5s} "
                                f"{n:3d}/{steps:3d} {group_label:28s} k={k} "
                                f"B={pair_ms:.1f}ms C={cached['ms']:.1f}ms "
                                f"saving={saving:+.2f}% skipped={skip_calls}/{possible} "
                                f"timed_final={'EXACT' if row['exact_timed_final'] else 'DIFF'} "
                                f"control={'OK' if baseline_stable else 'UNSTABLE'}",
                                flush=True,
                            )

                            if RUN_FIDELITY and repeat == 0:
                                f_cached = generate(
                                    model, ids, mask, n, steps, seed, device,
                                    group=group, k=k, capture_trajectory=True,
                                )
                                comparison = compare_histories(
                                    baseline_history, f_cached, prompt_tokens
                                )
                                # Ensures history collection does not change final tokens.
                                timed_cached_matches_history = torch.equal(
                                    cached["generated_ids"], f_cached["generated_ids"]
                                )
                                timed_base_matches_history = torch.equal(
                                    before["generated_ids"], baseline_history["generated_ids"]
                                )
                                fidelity_rows.append({
                                    "trial_id": trial_id, "task": task, "prompt_mode": mode,
                                    "new_tokens": n, "steps": steps,
                                    "prompt_tokens": prompt_tokens, "seed": seed,
                                    "block_group": group_label, "num_cached_blocks": len(group),
                                    "refresh_interval": k, **comparison,
                                    "baseline_history_control": history_control_exact,
                                    "baseline_timed_vs_history_final": timed_base_matches_history,
                                    "cache_timed_vs_history_final": timed_cached_matches_history,
                                    "baseline_trajectory_sha256": trajectory_hash(
                                        baseline_history["history"]
                                    ),
                                })
                                save_csv(fidelity_rows, "fidelity_runs.csv")
                                base_text = tokenizer.decode(
                                    baseline_history["generated_ids"], skip_special_tokens=True
                                )
                                cached_text = tokenizer.decode(
                                    f_cached["generated_ids"], skip_special_tokens=True
                                )
                                text_rows.append({
                                    "trial_id": trial_id, "task": task, "prompt_mode": mode,
                                    "seed": seed, "new_tokens": n, "steps": steps,
                                    "block_group": group_label, "refresh_interval": k,
                                    "final_text_exact": base_text == cached_text,
                                    "character_similarity_not_quality": SequenceMatcher(
                                        None, base_text, cached_text
                                    ).ratio(),
                                    "baseline_text": base_text, "cached_text": cached_text,
                                    "baseline_answer_correct_human": "",
                                    "cache_answer_correct_human": "",
                                    "quality_review_notes": "",
                                })
                                save_csv(text_rows, "answer_quality_review.csv")
                                label = "EXACT" if comparison["exact_trajectory"] else \
                                    f"DIFF@step{comparison['first_divergent_step']}"
                                print(f"    fidelity (UNTIMED): {label}; "
                                      f"history-control={history_control_exact}; "
                                      f"timed-vs-history={timed_base_matches_history}/"
                                      f"{timed_cached_matches_history}", flush=True)

    print_summary(timing_rows, fidelity_rows)
    metadata = {
        "experiment": "Exp8F controlled Dream block-residual caching scaling",
        "model": MODEL_NAME, "tasks": TASKS, "seeds": SEEDS,
        "generation_cases": GENERATION_CASES, "prompt_modes": PROMPT_MODES,
        "long_prompt_target_tokens": LONG_PROMPT_TARGET_TOKENS,
        "layer_groups": LAYER_GROUPS, "refresh_intervals": REFRESH_INTERVALS,
        "repeats": REPEATS, "warmups": WARMUP_RUNS,
        "hf_use_cache": False, "hf_cache_toggle_check": toggle_comparison,
        "generation_params": {
            "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
            "alg": ALGORITHM,
        },
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "timing": "Direct diffusion_generate, output_history=False, CUDA synced, "
                  "separate B-C-B, no history hooks or profiler. Patch installation "
                  "excluded; per-call residual compute and skip overhead included.",
        "fidelity": "Separate output_history=True runs; history contains steps 1..N; "
                    "baseline history re-used for each group/interval and checked "
                    "with second baseline-history run. Exact trajectories are token "
                    "states, NOT logits or intermediate features.",
        "comparison_warning": "Exp9B uses HF use_cache=True but Exp8F uses False. "
                              "Cross-experiment raw latency comparison is not "
                              "strictly controlled; compare per-run paired savings "
                              "and explicitly disclose different forward settings.",
        "quality_warning": "Synthetic long prompt is not real long-context benchmark. "
                           "Manual quality review fields are NOT automatically scored. "
                           "Single-GPU only; no edge-cloud communication measured.",
    }
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("\nSaved outputs to:", OUTPUT_DIR, flush=True)
    print("WARNING: Percent saving is E2E generation time. Quality still requires review.")


if __name__ == "__main__":
    main()
