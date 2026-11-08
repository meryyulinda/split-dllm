"""Exp8C: measured single-block residual caching with actual decoder-layer skips.

Model: Dream-org/Dream-v0-Instruct-7B
Run:   python experiments/exp8c_actual_block_residual_cache.py
Config: add EXP8C_* variables to configs/experiment_config.py.

Refresh: execute decoder_layer.forward, cache R = H_out - H_in.
Reuse:   bypass decoder_layer.forward, return (H_in + R_cached,) in Dream's
         one-element tuple output contract (with built-in KV/attention output
         disabled for baseline AND cached runs).

This is NOT KV caching, and is not yet edge-cloud split inference.
"""
import os
import sys

# Execute from dasr1/: python experiments/exp8c_actual_block_residual_cache.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd
import torch
import transformers

from configs.experiment_config import (
    MODEL_NAME, SEED, MAX_NEW_TOKENS, STEPS, TEMPERATURE,
    TOP_P, TOP_K, ALGORITHM, PROMPT, DRIFT_EPSILON,
    EXP8C_LAYERS, EXP8C_REFRESH_INTERVALS,
    EXP8C_ADAPTIVE_INPUT_DRIFT_THRESHOLDS, EXP8C_ADAPTIVE_MAX_AGE,
    EXP8C_REPEATS, EXP8C_WARMUP_RUNS,
)
from src.model_loader import load_dream_model
from src.generation import generate_with_token_history
from src.instrumentation import create_token_history_hook
from src.utils import set_seed

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "exp8c")


@dataclass(frozen=True)
class CachePolicy:
    mode: str  # "periodic" or "input_drift"
    interval: Optional[int] = None
    drift_threshold: Optional[float] = None
    max_age: Optional[int] = None

    @property
    def label(self):
        if self.mode == "periodic":
            return f"periodic_k{self.interval}"
        return f"input_drift_tau{self.drift_threshold:g}_maxage{self.max_age}"


class ActualSingleBlockResidualCache:
    """Patch one Dream layer.forward to *actually skip* its original computation.

    Refresh: run original block and cache R = H_out - H_in (FP32).
    Reuse:   return H_in(current) + R(last refresh), without calling original.

    This mechanism caches one transformer block's total residual update,
    NOT the whole prefix activation and NOT an attention K/V cache.
    """

    def __init__(self, model, layer_number: int, policy: CachePolicy, eps: float):
        self.model = model
        self.layer_number = int(layer_number)
        self.policy = policy
        self.eps = float(eps)
        self.layer = model.model.layers[self.layer_number - 1]
        self.original_forward = None
        self.cache_residual = None
        self.cache_input = None
        self.output_dtype = None
        self.output_is_tuple = None
        self.output_tuple_size = None
        self.output_tuple_extras = None
        self.total_calls = 0
        self.executed_calls = 0
        self.reused_calls = 0
        self.last_refresh_call = None
        self.refresh_input_drift = []
        self.all_probe_drifts = []
        self.skip_confirmations = 0

    @staticmethod
    def _hidden_from_output(output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(
                "Unexpected Dream block output. Expected Tensor [B,N,D] "
                "or tuple with a Tensor in its first position."
            )
        return hidden

    @staticmethod
    def _input_from_call(args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("Expected Dream block input [B,N,D]")
        return hidden

    def _input_drift(self, current):
        # This scalar GPU -> CPU read synchronizes the stream; its overhead
        # is included in end-to-end timings for adaptive policy.
        old = self.cache_input
        if old is None or old.shape != current.shape:
            return None
        x = current.detach().float()
        y = old.float()
        return float((
            torch.linalg.vector_norm(x - y) /
            (torch.linalg.vector_norm(x) + self.eps)
        ).item())

    def _pack_reused_output(self, hidden):
        if self.output_is_tuple:
            # Non-None auxiliary outputs (e.g. attention weights, KV cache)
            # cannot safely be replayed as if they came from current input.
            if self.output_tuple_extras and any(v is not None for v in self.output_tuple_extras):
                raise RuntimeError(
                    "Cannot skip block: last refresh returned non-None "
                    "auxiliary tensors/cache. Disable them or adapt wrapper."
                )
            return (hidden, *self.output_tuple_extras)
        return hidden

    def _patched_forward(self, *args, **kwargs):
        # DreamBaseModel.forward passes these keyword arguments to decoder layers.
        # Skipping a block while leaving a mutable Hugging Face KV cache active
        # would silently desynchronize that cache. Do not allow it in this test.
        if kwargs.get("use_cache", False) or kwargs.get("past_key_value") is not None:
            raise RuntimeError(
                "Exp8C block-residual reuse requires use_cache=False and "
                "past_key_value=None. Call _disable_hf_decoder_cache(model) "
                "before running baseline or cached generation."
            )
        if kwargs.get("output_attentions", False):
            raise RuntimeError(
                "Exp8C block-residual reuse requires output_attentions=False "
                "because skipped blocks cannot produce fresh attention weights."
            )
        current_input = self._input_from_call(args, kwargs)
        call_index = self.total_calls
        self.total_calls += 1

        if self.cache_residual is not None:
            if (
                self.cache_residual.shape != current_input.shape
                or self.cache_residual.device != current_input.device
            ):
                raise RuntimeError("Cached residual and current input shapes/devices differ")

        refresh = self.cache_residual is None
        drift = None
        if not refresh:
            age = call_index - self.last_refresh_call
            if self.policy.mode == "periodic":
                refresh = age >= self.policy.interval
            elif self.policy.mode == "input_drift":
                drift = self._input_drift(current_input)
                self.all_probe_drifts.append(drift)
                refresh = (
                    drift is None
                    or drift > self.policy.drift_threshold
                    or age >= self.policy.max_age
                )
            else:
                raise ValueError(f"Unknown cache policy: {self.policy.mode}")

        if refresh:
            # Copy before forward to handle any in-place changes defensively.
            input_fp32 = current_input.detach().float().clone()
            output = self.original_forward(*args, **kwargs)
            hidden = self._hidden_from_output(output)
            if hidden.shape != current_input.shape:
                raise RuntimeError("Block must preserve [B,N,D] for residual cache")
            if isinstance(output, tuple):
                extras = output[1:]
                if any(v is not None for v in extras):
                    raise RuntimeError(
                        "Block produced auxiliary outputs even though KV caching "
                        "and attention-output collection should be disabled. "
                        "Check Dream model configuration / explicit forward flags."
                    )
                self.output_tuple_extras = extras
                self.output_is_tuple = True
            else:
                self.output_tuple_extras = None
                self.output_is_tuple = False
            self.output_dtype = hidden.dtype
            self.cache_residual = hidden.detach().float().sub(input_fp32)
            self.cache_input = current_input.detach().clone()
            self.executed_calls += 1
            self.last_refresh_call = call_index
            if drift is not None:
                self.refresh_input_drift.append(drift)
            return output

        approx_hidden = (current_input.float() + self.cache_residual).to(self.output_dtype)
        self.reused_calls += 1
        self.skip_confirmations += 1  # original_forward was NOT called on this path
        return self._pack_reused_output(approx_hidden)

    def __enter__(self):
        if self.original_forward is not None:
            raise RuntimeError("Cache patch already installed")
        self.original_forward = self.layer.forward
        self.layer.forward = self._patched_forward
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_forward is not None:
            self.layer.forward = self.original_forward
            self.original_forward = None
        return False

    def stats(self):
        if self.total_calls != self.executed_calls + self.reused_calls:
            raise RuntimeError("Inconsistent block execution counters")
        return {
            "total_block_calls": self.total_calls,
            "executed_block_calls": self.executed_calls,
            "reused_block_calls": self.reused_calls,
            "skip_confirmations": self.skip_confirmations,
            "reuse_fraction": self.reused_calls / max(1, self.total_calls),
            "mean_adaptive_input_drift": (
                statistics.mean(self.all_probe_drifts)
                if self.all_probe_drifts else None
            ),
            "max_adaptive_input_drift": (
                max(self.all_probe_drifts) if self.all_probe_drifts else None
            ),
        }


def _disable_hf_decoder_cache(model):
    """Disable *Hugging Face* KV/attention outputs for this experiment.

    Dream's diffusion sampler recomputes the entire sequence on each step.
    When config.use_cache=True, DreamDecoderLayer returns a non-None
    past_key_value (a mutable DynamicCache object). An approximated/skipped
    decoder layer must not leave that object in an inconsistent state.

    Apply this to the baseline AND cache runs, so their forward contracts
    are identical. This does NOT disable the Exp8C block-residual cache.
    """
    for owner_name, owner in (("model", model), ("model.model", model.model)):
        cfg = getattr(owner, "config", None)
        if cfg is None:
            raise RuntimeError(f"{owner_name} has no config; cannot disable KV caching")
        cfg.use_cache = False
        cfg.output_attentions = False

    generation_cfg = getattr(model, "generation_config", None)
    if generation_cfg is not None:
        # The Dream 'origin' sampler calls self(...) directly, but keep this
        # consistent in case the local generation wrapper forwards it.
        generation_cfg.use_cache = False
        generation_cfg.output_attentions = False

    if model.config.use_cache or model.model.config.use_cache:
        raise RuntimeError("Could not disable Hugging Face decoder KV caching")
    print("HF use_cache=False; output_attentions=False (baseline and Exp8C)")


def _run_generation(model, tokenizer, device, cache=None):
    """Time one *actual* diffusion_generate trajectory with token history enabled."""
    set_seed(SEED)
    history_hook, history = create_token_history_hook()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        if cache is None:
            result = generate_with_token_history(
                model=model, tokenizer=tokenizer, prompt=PROMPT,
                max_new_tokens=MAX_NEW_TOKENS, steps=STEPS,
                temperature=TEMPERATURE, history_hook=history_hook,
                top_p=TOP_P, top_k=TOP_K, alg=ALGORITHM,
            )
            counters = None
        else:
            with cache:
                result = generate_with_token_history(
                    model=model, tokenizer=tokenizer, prompt=PROMPT,
                    max_new_tokens=MAX_NEW_TOKENS, steps=STEPS,
                    temperature=TEMPERATURE, history_hook=history_hook,
                    top_p=TOP_P, top_k=TOP_K, alg=ALGORITHM,
                )
            counters = cache.stats()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    states = []
    for state in history:
        if not isinstance(state, torch.Tensor) or state.ndim != 1:
            raise RuntimeError("Expected 1D token states from history_hook")
        states.append(state.detach().cpu().long().clone())
    if len(states) != STEPS + 1:
        raise RuntimeError(
            f"Expected {STEPS + 1} token states, received {len(states)}. "
            "Check Dream history instrumentation."
        )
    sequence_length = states[0].numel()
    prompt_length = sequence_length - MAX_NEW_TOKENS
    if prompt_length <= 0:
        raise RuntimeError("Invalid prompt length inferred from token trajectory")
    if any(s.numel() != sequence_length for s in states):
        raise RuntimeError("Inconsistent token trajectory lengths")
    if counters is not None and counters["total_block_calls"] != STEPS:
        raise RuntimeError(
            f"Expected one block call per denoising step ({STEPS}), "
            f"observed {counters['total_block_calls']}. "
            "Caching cadence is not aligned; inspect generation.forward."
        )
    # Use *actual final token state*, avoiding assumptions about helper's
    # returned dict or whether decoded output includes the prompt.
    final_tokens = states[-1][prompt_length:]
    final_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
    return {
        "states": states,
        "prompt_length": prompt_length,
        "elapsed_ms": elapsed_ms,
        "final_text": final_text,
        "counters": counters,
    }


def _compare_trajectories(reference, candidate):
    base = reference["states"]
    trial = candidate["states"]
    if reference["prompt_length"] != candidate["prompt_length"]:
        raise RuntimeError("Reference/trial prompt length mismatch")
    prompt_len = reference["prompt_length"]
    same_state = [torch.equal(a, b) for a, b in zip(base, trial)]
    token_matches = [float((a[prompt_len:] == b[prompt_len:]).float().mean()) for a, b in zip(base, trial)]
    first_divergent = next((i for i, identical in enumerate(same_state) if not identical), None)
    return {
        "exact_states_including_initial": int(sum(same_state)),
        "exact_transitions_output_states": int(sum(same_state[1:])),
        "mean_generated_token_match_all_states": float(statistics.mean(token_matches)),
        "final_generated_token_match_ratio": token_matches[-1],
        "final_generated_ids_equal": bool(same_state[-1]),
        "final_text_equal": bool(reference["final_text"] == candidate["final_text"]),
        "first_divergent_state": first_divergent,
        "per_state": [
            {"state": t, "state_ids_equal": bool(is_same), "generated_token_match": match}
            for t, (is_same, match) in enumerate(zip(same_state, token_matches))
        ],
    }


def _validate_settings(model):
    n_layers = len(model.model.layers)
    layers = list(EXP8C_LAYERS)
    intervals = list(EXP8C_REFRESH_INTERVALS)
    thresholds = list(EXP8C_ADAPTIVE_INPUT_DRIFT_THRESHOLDS)
    if not layers or any(not isinstance(l, int) or not (1 <= l <= n_layers) for l in layers):
        raise RuntimeError(f"EXP8C_LAYERS must be in [1, {n_layers}]")
    if len(layers) != len(set(layers)):
        raise RuntimeError("EXP8C_LAYERS contains duplicate layers")
    if any(not isinstance(k, int) or k < 2 for k in intervals):
        raise RuntimeError("Refresh intervals must be integers >= 2")
    if any(not (0 <= t < 10) for t in thresholds):
        raise RuntimeError("Invalid adaptive drift thresholds")
    if thresholds and (not isinstance(EXP8C_ADAPTIVE_MAX_AGE, int) or EXP8C_ADAPTIVE_MAX_AGE < 2):
        raise RuntimeError("EXP8C_ADAPTIVE_MAX_AGE must be >= 2")
    if not (isinstance(EXP8C_REPEATS, int) and EXP8C_REPEATS >= 1):
        raise RuntimeError("EXP8C_REPEATS must be >= 1")
    if not (isinstance(EXP8C_WARMUP_RUNS, int) and EXP8C_WARMUP_RUNS >= 0):
        raise RuntimeError("EXP8C_WARMUP_RUNS must be >= 0")
    return layers, intervals, thresholds


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp8C requires CUDA")

    print("=" * 78)
    print("EXPERIMENT 8C: ACTUAL SINGLE-BLOCK RESIDUAL CACHE + REFRESH (DREAM)")
    print("=" * 78)
    print(f"Model: {MODEL_NAME} | Prompt: {PROMPT}")
    print(f"Generation: {MAX_NEW_TOKENS} tokens / {STEPS} steps | alg: {ALGORITHM}")
    print(f"TOP_P: {TOP_P} | TOP_K: {TOP_K} | seed: {SEED}")

    set_seed(SEED)
    model, tokenizer = load_dream_model(MODEL_NAME)
    model.eval()
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Model is on {device}, expected CUDA")

    # Sanity control: confirm that disabling the built-in Hugging Face KV
    # cache does not change the baseline token trajectory for this checkpoint.
    # This check is untimed and uses the *same* generation helper and seed.
    print("\nSanity check: reference trajectory before disabling HF KV cache...")
    original_config_reference = _run_generation(model, tokenizer, device)
    _disable_hf_decoder_cache(model)
    disabled_config_reference = _run_generation(model, tokenizer, device)
    hf_cache_toggle_exact = all(
        torch.equal(a, b)
        for a, b in zip(
            original_config_reference["states"],
            disabled_config_reference["states"],
        )
    )
    print(f"HF KV-cache-off baseline matches original trajectory: {hf_cache_toggle_exact}")
    if not hf_cache_toggle_exact:
        print("WARNING: Disabling HF KV cache changes the baseline trajectory. "
              "Do not compare fidelity/latency directly with older experiments; "
              "Exp8C still compares baseline and cache modes under identical settings.")
    layers, intervals, thresholds = _validate_settings(model)
    policies = (
        [CachePolicy("periodic", interval=k) for k in intervals]
        + [CachePolicy("input_drift", drift_threshold=t, max_age=EXP8C_ADAPTIVE_MAX_AGE)
           for t in thresholds]
    )
    if not policies:
        raise RuntimeError("Configure at least one Exp8C refresh policy")
    print(f"Device: {device} | Layers: {layers}")
    print(f"Policies: {[p.label for p in policies]} | repeats: {EXP8C_REPEATS}")

    print("\nWarmup: baseline Dream generation (not timed)...")
    for _ in range(EXP8C_WARMUP_RUNS):
        _run_generation(model, tokenizer, device)

    runs = []
    transition_rows = []
    generated_rows = []
    baseline_samples = []
    reference = None
    baseline_deterministic = True

    def add_baseline(repeat, position):
        nonlocal reference, baseline_deterministic
        run = _run_generation(model, tokenizer, device)
        if reference is None:
            reference = run
        else:
            if not all(torch.equal(a, b) for a, b in zip(reference["states"], run["states"])):
                baseline_deterministic = False
        baseline_samples.append(run["elapsed_ms"])
        runs.append({
            "mode": "baseline", "policy": "full_recompute", "layer": None,
            "refresh_interval": None, "input_drift_threshold": None,
            "max_age": None, "repeat": repeat, "position": position,
            "e2e_ms": run["elapsed_ms"], "total_block_calls": None,
            "executed_block_calls": None, "reused_block_calls": None,
            "reuse_fraction": 0.0,
            "final_generated_ids_equal": True, "final_text_equal": True,
            "exact_transitions_output_states": STEPS,
            "mean_generated_token_match_all_states": 1.0,
            "first_divergent_state": None,
            "mean_adaptive_input_drift": None,
        })
        print(f"  baseline {position} rep {repeat}: {run['elapsed_ms']:.3f} ms")

    print("\nBaseline benchmark (before cached runs)...")
    for i in range(EXP8C_REPEATS):
        add_baseline(i, "before")

    generated_rows.append({"mode": "baseline", "policy": "full_recompute", "layer": None,
                           "repeat": 0, "final_text": reference["final_text"]})

    print("\nRunning cache policies...")
    for layer in layers:
        for policy in policies:
            for repeat in range(EXP8C_REPEATS):
                cache = ActualSingleBlockResidualCache(
                    model=model, layer_number=layer, policy=policy, eps=DRIFT_EPSILON
                )
                trial = _run_generation(model, tokenizer, device, cache)
                fidelity = _compare_trajectories(reference, trial)
                stats = trial["counters"]
                runs.append({
                    "mode": "block_residual_cache", "policy": policy.label,
                    "layer": layer, "refresh_interval": policy.interval,
                    "input_drift_threshold": policy.drift_threshold,
                    "max_age": policy.max_age,
                    "repeat": repeat, "position": "cache",
                    "e2e_ms": trial["elapsed_ms"],
                    **stats,
                    **{k: v for k, v in fidelity.items() if k != "per_state"},
                })
                generated_rows.append({
                    "mode": "block_residual_cache", "policy": policy.label,
                    "layer": layer, "repeat": repeat,
                    "final_text": trial["final_text"],
                })
                for item in fidelity["per_state"]:
                    transition_rows.append({
                        "layer": layer, "policy": policy.label, "repeat": repeat,
                        **item,
                    })
                print(
                    f"  L{layer:02d} {policy.label:<32} rep {repeat} | "
                    f"{trial['elapsed_ms']:8.2f} ms | "
                    f"skip={stats['reused_block_calls']:2d}/{stats['total_block_calls']} | "
                    f"final_ids_equal={fidelity['final_generated_ids_equal']} | "
                    f"exact_states={fidelity['exact_transitions_output_states']}/{STEPS}"
                )

    print("\nBaseline benchmark (after cached runs)...")
    for i in range(EXP8C_REPEATS):
        add_baseline(i, "after")

    baseline_median = statistics.median(baseline_samples)
    run_df = pd.DataFrame(runs)
    run_df["baseline_median_ms"] = baseline_median
    run_df["speedup_vs_baseline"] = baseline_median / run_df["e2e_ms"]
    run_df["e2e_saving_pct"] = (1 - run_df["e2e_ms"] / baseline_median) * 100

    grouped = (run_df[run_df["mode"] == "block_residual_cache"]
               .groupby(["layer", "policy"], as_index=False)
               .agg(
                   repeats=("repeat", "count"),
                   median_e2e_ms=("e2e_ms", "median"),
                   mean_e2e_ms=("e2e_ms", "mean"),
                   min_e2e_ms=("e2e_ms", "min"),
                   reused_block_calls=("reused_block_calls", "mean"),
                   executed_block_calls=("executed_block_calls", "mean"),
                   exact_final_ids_runs=("final_generated_ids_equal", "sum"),
                   exact_final_text_runs=("final_text_equal", "sum"),
                   mean_exact_states=("exact_transitions_output_states", "mean"),
                   mean_state_token_match=("mean_generated_token_match_all_states", "mean"),
               ))
    grouped["baseline_median_ms"] = baseline_median
    grouped["speedup_vs_baseline_median"] = baseline_median / grouped["median_e2e_ms"]
    grouped["e2e_saving_pct"] = (1 - grouped["median_e2e_ms"] / baseline_median) * 100
    grouped = grouped.sort_values(["layer", "policy"])

    files = {
        "runs": os.path.join(OUTPUT_DIR, "actual_cache_runs.csv"),
        "summary": os.path.join(OUTPUT_DIR, "actual_cache_summary.csv"),
        "trajectory": os.path.join(OUTPUT_DIR, "cache_trajectory_fidelity.csv"),
        "text": os.path.join(OUTPUT_DIR, "generated_texts.csv"),
        "metadata": os.path.join(OUTPUT_DIR, "experiment_metadata.json"),
    }
    run_df.to_csv(files["runs"], index=False)
    grouped.to_csv(files["summary"], index=False)
    pd.DataFrame(transition_rows).to_csv(files["trajectory"], index=False)
    pd.DataFrame(generated_rows).to_csv(files["text"], index=False)

    metadata = {
        "experiment": "8C actual single-block residual skip with periodic/adaptive refresh",
        "research_stage": "single-GPU actual inference microbenchmark (NOT edge-cloud)",
        "model": MODEL_NAME, "prompt": PROMPT, "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS, "steps": STEPS,
        "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
        "algorithm": ALGORITHM, "layers_1_based": layers,
        "policies": [asdict(p) for p in policies],
        "repeats": EXP8C_REPEATS, "warmup_runs": EXP8C_WARMUP_RUNS,
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "baseline_median_ms": baseline_median,
        "baseline_reproduced_exactly": baseline_deterministic,
        "hf_cache_toggle_baseline_exact": hf_cache_toggle_exact,
        "hf_use_cache": bool(model.config.use_cache),
        "hf_output_attentions": bool(model.config.output_attentions),
        "kv_cache_note": (
            "Built-in Hugging Face KV caching disabled for both baseline "
            "and caching runs; Exp8C residual cache remains enabled. "
            "The Dream sampler recomputes the full sequence per denoising step."
        ),
        "cache_mechanism": (
            "Refresh: original block.forward, cache FP32 residual H_out-H_in; "
            "reuse: skip original block.forward and return H_in(current) + cached residual"
        ),
        "quality_limitations": (
            "Generated-token exact agreement and trajectory match are measured, "
            "but no semantic benchmark is run. Any mismatch does NOT prove bad quality; "
            "an exact match is a stronger correctness control. Adaptive input drift "
            "requires GPU->CPU synchronization and is included in latency."
        ),
        "systems_limitations": (
            "Only a single transformer block is cached at a time; edge/cloud split, "
            "boundary transfer, real network costs, and multi-block caching are NOT tested."
        ),
    }
    with open(files["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print("EXP8C SUMMARY: ACTUAL SINGLE-BLOCK SKIP, MEDIAN E2E GENERATION LATENCY")
    print("=" * 78)
    print(f"Baseline median: {baseline_median:.2f} ms | baseline deterministic: {baseline_deterministic}")
    print(grouped.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:")
    for path in files.values():
        print("  ", path)
    print(
        "\nCAVEAT: Measured single-GPU whole-generation time, not split edge-cloud E2E. "
        "This experiment tests one-block skipping; it does not yet demonstrate "
        "multi-block acceleration or semantic quality preservation."
    )
    if not baseline_deterministic:
        print("WARNING: baseline runs were not deterministic; treat quality matches cautiously.")
    if not hf_cache_toggle_exact:
        print("WARNING: Disabling Hugging Face KV caching altered the prior-default trajectory. "
              "Exp8C internal comparisons remain controlled, but historical comparisons do not.")


if __name__ == "__main__":
    main()
