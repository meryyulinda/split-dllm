"""Exp9B — official dLLM-Cache diagnostic reproduction for Dream.

Run from dasr1/: python experiments/exp9b_official_feature_cache_diagnostics.py

Unlike Exp9A, the TIMED B-C-B generations use model.diffusion_generate()
DIRECTLY with output_history=False, no token-history hooks, no profiler and
no module instrumentation. Full denoising trajectory and module-work probes
are separately run OUTSIDE latency measurements.

This is not an official paper benchmark/LongBench reproduction, nor a
split-edge-cloud experiment. Long prompts are synthetic diagnostic contexts.
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
from contextlib import contextmanager
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import torch
import transformers

from configs.experiment_config import (
    MODEL_NAME, SEED, TEMPERATURE, TOP_P, TOP_K, ALGORITHM, PROMPT, RQ5_PROMPTS,
    EXP9A_OFFICIAL_CACHE_REPO,
)
from src.model_loader import load_dream_model
from src.utils import set_seed

# Optional Exp9B-only parameters. All defaults are overridable in the shared
# configs/experiment_config.py; no changes to older EXP9A or EXP8* scripts.
import configs.experiment_config as cfg


def setting(name, fallback):
    return getattr(cfg, name, fallback)


OFFICIAL_REPO = setting("EXP9B_OFFICIAL_CACHE_REPO", EXP9A_OFFICIAL_CACHE_REPO)
TASKS = list(setting("EXP9B_TASK_KEYS", ["factual_qa", "reasoning", "instruction"]))
SEEDS = list(setting("EXP9B_SEEDS", [SEED]))
GENERATION_CASES = [tuple(map(int, case)) for case in setting(
    "EXP9B_GENERATION_CASES", [(64, 64), (256, 256)]
)]
PROMPT_MODES = list(setting("EXP9B_PROMPT_MODES", ["short", "long"]))
LONG_PROMPT_TARGET_TOKENS = int(setting("EXP9B_LONG_PROMPT_TARGET_TOKENS", 1024))
CACHE_CONFIGS = [tuple(v) for v in setting("EXP9B_CACHE_CONFIGS", [(100, 7, 0.25)])]
REPEATS = int(setting("EXP9B_REPEATS", 1))
WARMUP_RUNS = int(setting("EXP9B_WARMUP_RUNS", 1))
RUN_MODULE_PROBE = bool(setting("EXP9B_RUN_MODULE_PROBE", True))
PROBE_TASK_KEY = setting("EXP9B_PROBE_TASK_KEY", "reasoning")
PROFILE_ONE_CASE = bool(setting("EXP9B_PROFILE_ONE_CASE", False))
OUTPUT_DIR = Path(PROJECT_ROOT) / "outputs" / "exp9b"


def validate_config():
    if not torch.cuda.is_available():
        raise RuntimeError("Exp9B requires a CUDA-capable GPU")
    if "Dream" not in MODEL_NAME:
        raise ValueError("Exp9B only supports Dream with the official Dream cache adapter")
    if not TASKS or not SEEDS or not GENERATION_CASES or not PROMPT_MODES or not CACHE_CONFIGS:
        raise ValueError("Task/seed/workload/cache configuration cannot be empty")
    if REPEATS < 1 or WARMUP_RUNS < 0:
        raise ValueError("Invalid repeats/warmups")
    for n, steps in GENERATION_CASES:
        if n <= 0 or steps <= 0:
            raise ValueError("Generation lengths and steps must be positive")
    for p in PROMPT_MODES:
        if p not in {"short", "long"}:
            raise ValueError(f"Unknown prompt mode: {p}")
    for pi, gi, rho in CACHE_CONFIGS:
        if int(pi) < 1 or int(gi) < 1 or not (0 < float(rho) <= 1):
            raise ValueError(f"Invalid official cache config: {(pi, gi, rho)}")
    for task in TASKS:
        if task != "default" and task not in RQ5_PROMPTS:
            raise KeyError(f"Task {task!r} missing from RQ5_PROMPTS")


def import_official():
    repo = Path(os.path.expandvars(os.path.expanduser(str(OFFICIAL_REPO))))
    if not repo.is_absolute():
        repo = Path(PROJECT_ROOT) / repo
    repo = repo.resolve()
    expected = [repo / "dllm_cache" / "cache" / "Cache.py",
                repo / "dllm_cache" / "cache.py"]
    if not any(p.is_file() for p in expected):
        raise FileNotFoundError(
            "Official dLLM-Cache checkout missing. Expected one of:\n" +
            "\n".join(map(str, expected)) +
            "\nEXP9B_OFFICIAL_CACHE_REPO must be the repo ROOT, not dllm_cache/"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dllm_cache.cache import dLLMCache, dLLMCacheConfig
    from dllm_cache.hooks import register_cache_Dream, logout_cache_Dream
    return repo, (dLLMCache, dLLMCacheConfig, register_cache_Dream, logout_cache_Dream)


def synchronize(device):
    torch.cuda.synchronize(device)


def prompt_token_length(tokenizer, prompt):
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
    # Synthetic length-scaling stress test, NOT genuine LongBench documents.
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
    raise RuntimeError("Long prompt could not reach desired token count")


def prepare_input(tokenizer, prompt, device):
    # Matches the public Dream example and official dLLM-Cache Dream demo.
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(ids)
    else:
        mask = mask.to(device)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise RuntimeError("Expected single batch [1, prompt_len]")
    return ids, mask


@contextmanager
def official_cache(model, api, input_len, cache_cfg, device):
    # Cache registration/reset is deliberately NOT part of timed generation.
    dLLMCache, dLLMCacheConfig, register, logout = api
    pi, gi, rho = cache_cfg
    synchronize(device)
    start = time.perf_counter()
    dLLMCache.new_instance(**asdict(dLLMCacheConfig(
        prompt_interval_steps=int(pi),
        gen_interval_steps=int(gi),
        transfer_ratio=float(rho),
    )))
    registered = False
    try:
        register(model, "model.layers")
        registered = True
        dLLMCache().reset_cache(int(input_len))
        synchronize(device)
        setup_ms = (time.perf_counter() - start) * 1000.0
        yield setup_ms
    finally:
        if registered:
            logout(model, "model.layers")


def dream_generate(model, ids, mask, new_tokens, steps, capture_trajectory):
    with torch.inference_mode():
        out = model.diffusion_generate(
            ids, attention_mask=mask,
            max_new_tokens=int(new_tokens), steps=int(steps),
            temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            alg=ALGORITHM, output_history=bool(capture_trajectory),
            return_dict_in_generate=True,
        )
    if not hasattr(out, "sequences"):
        raise RuntimeError("Dream diffusion_generate did not return sequences")
    return out


def run_generation(model, ids, mask, n, steps, seed, device,
                   cache_api=None, cache_cfg=None, capture_trajectory=False,
                   probe=None, use_profiler=False):
    """One generation. History/probes/profiler must be OFF for timing runs."""
    set_seed(int(seed))
    cache_context = official_cache(model, cache_api, ids.shape[1], cache_cfg, device) \
        if cache_api is not None else None
    setup_ms = 0.0
    if cache_context is not None:
        setup_ms = cache_context.__enter__()
    try:
        if probe is not None:
            probe.attach(model)
        try:
            # Seed immediately before generation, AFTER cache registration/reset,
            # so potential initialization RNG does not skew baseline vs cache.
            set_seed(int(seed))
            # Sync excludes queued work from previous trials.
            synchronize(device)
            start = time.perf_counter()
            if use_profiler:
                import torch.profiler as prof
                with prof.profile(activities=[prof.ProfilerActivity.CPU,
                                              prof.ProfilerActivity.CUDA],
                                  record_shapes=False, profile_memory=False) as profiler:
                    out = dream_generate(model, ids, mask, n, steps, capture_trajectory)
                    synchronize(device)
            else:
                profiler = None
                out = dream_generate(model, ids, mask, n, steps, capture_trajectory)
                synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        finally:
            if probe is not None:
                probe.detach()
        # Copy AFTER timing, so copies and decode do not bias the reported time.
        generated_ids = out.sequences[0, ids.shape[1]:].detach().cpu().long().clone()
        history = None
        if capture_trajectory:
            raw = getattr(out, "history", None)
            if raw is None or len(raw) != int(steps):
                raise RuntimeError(
                    f"Expected Dream history of {steps} states; got "
                    f"{None if raw is None else len(raw)}"
                )
            history = [h[0].detach().cpu().long().clone() for h in raw]
            if not torch.equal(history[-1][ids.shape[1]:], generated_ids):
                raise RuntimeError("Last history state does not match final output")
        return {
            "ms": elapsed_ms,
            "cache_setup_ms": float(setup_ms),
            "generated_ids": generated_ids,
            "history": history,
            "profiler": profiler,
        }
    finally:
        if cache_context is not None:
            cache_context.__exit__(None, None, None)


def sha256_history(history):
    digest = hashlib.sha256()
    for state in history:
        digest.update(state.numpy().tobytes())
    return digest.hexdigest()


def compare_history(a, b, prompt_len):
    x, y = a["history"], b["history"]
    if x is None or y is None or len(x) != len(y):
        raise RuntimeError("Both runs must have equally long histories")
    same = [torch.equal(s, t) for s, t in zip(x, y)]
    agreement = [float((s[prompt_len:] == t[prompt_len:]).float().mean())
                 for s, t in zip(x, y)]
    return {
        "exact_trajectory": bool(all(same)),
        "exact_transition_states": int(sum(same)),
        "first_divergent_step": next((i + 1 for i, ok in enumerate(same) if not ok), None),
        "mean_generated_token_agreement": float(statistics.mean(agreement)),
        "final_generated_token_agreement": float(agreement[-1]),
        "exact_final_ids": torch.equal(a["generated_ids"], b["generated_ids"]),
    }


class ModuleWorkProbe:
    """Counts Python module invocations and input rows; NOT FLOPs or CUDA time.

    Never active during timed B-C-B runs. Official cached kernels might have
    different work even when these module invocation counts do not change.
    """
    def __init__(self):
        self.handles = []
        self.data = {}

    def attach(self, model):
        if self.handles:
            raise RuntimeError("Probe already attached")
        for layer_idx, layer in enumerate(model.model.layers, 1):
            for component in ("self_attn", "mlp"):
                module = getattr(layer, component, None)
                if module is None:
                    continue
                key = (layer_idx, component)
                self.data[key] = {"calls": 0, "input_tokens": 0}
                def hook(mod, args, k=key):
                    row = self.data[k]
                    row["calls"] += 1
                    if args and isinstance(args[0], torch.Tensor):
                        tensor = args[0]
                        if tensor.ndim >= 3:
                            row["input_tokens"] += int(tensor.shape[0] * tensor.shape[-2])
                self.handles.append(module.register_forward_pre_hook(hook))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def totals(self):
        grouped = {}
        for (layer, component), item in self.data.items():
            row = grouped.setdefault(component, {"calls": 0, "input_tokens": 0, "observed_layers": 0})
            row["calls"] += item["calls"]
            row["input_tokens"] += item["input_tokens"]
            row["observed_layers"] += 1
        return grouped


def profile_rows(profiler, label):
    if profiler is None:
        return []
    result = []
    stats = sorted(profiler.key_averages(),
                   key=lambda x: float(getattr(x, "self_device_time_total", 0) or
                                       getattr(x, "self_cuda_time_total", 0)),
                   reverse=True)
    for item in stats[:25]:
        result.append({
            "run_label": label,
            "op_name": item.key,
            "calls": int(item.count),
            "self_device_us": float(getattr(item, "self_device_time_total", 0) or
                                    getattr(item, "self_cuda_time_total", 0)),
            "self_cpu_us": float(item.self_cpu_time_total),
        })
    return result


def to_csv_safely(records, filename):
    if records:
        pd.DataFrame(records).to_csv(OUTPUT_DIR / filename, index=False)


def summarize(timing):
    df = pd.DataFrame(timing)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    stable = df[df["baseline_pair_stable"]].copy()
    if stable.empty:
        return pd.DataFrame(), pd.DataFrame()
    grouping = ["task", "prompt_mode", "new_tokens", "steps",
                "prompt_interval_steps", "gen_interval_steps", "transfer_ratio"]
    summary = stable.groupby(grouping, dropna=False, sort=True).agg(
        trials=("trial_id", "size"),
        median_prompt_tokens=("prompt_tokens", "median"),
        median_paired_baseline_ms=("paired_baseline_ms", "median"),
        median_cached_ms=("cache_ms", "median"),
        median_paired_saving_pct=("saving_pct", "median"),
        min_paired_saving_pct=("saving_pct", "min"),
        median_cache_setup_ms=("cache_setup_ms", "median"),
    ).reset_index()
    by_workload = stable.groupby(["prompt_mode", "new_tokens", "steps"],
                                 dropna=False, sort=True).agg(
        trials=("trial_id", "size"),
        median_paired_saving_pct=("saving_pct", "median"),
        median_prompt_tokens=("prompt_tokens", "median"),
    ).reset_index()
    return summary, by_workload


def main():
    validate_config()
    repo, api = import_official()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_dream_model(MODEL_NAME)
    model.eval()
    device = next(model.parameters()).device
    print("=" * 78)
    print("EXP9B: OFFICIAL FEATURE CACHE DIAGNOSTIC (DREAM, SINGLE GPU)")
    print("=" * 78)
    print(f"Model: {MODEL_NAME} | GPU: {torch.cuda.get_device_name(device)}")
    print(f"Official repo: {repo}")
    print(f"Generation cases: {GENERATION_CASES} | modes: {PROMPT_MODES}")
    print(f"Tasks: {TASKS} | seeds: {SEEDS} | repeats: {REPEATS}")
    print(f"Cache configs (pi,gi,rho): {CACHE_CONFIGS}")
    print("TIMED RUNS: direct diffusion_generate(output_history=False); no history/hooks/profiler")
    print("Diagnostics: separate history runs; optional separate module probes/profiler")

    timing_rows, fidelity_rows, text_rows = [], [], []
    probe_rows, kernel_rows = [], []
    trial_id = 0
    total_trials = len(TASKS) * len(PROMPT_MODES) * len(GENERATION_CASES) * \
                   len(SEEDS) * len(CACHE_CONFIGS) * REPEATS
    examined_probe_cases = set()
    did_profile = False

    # One warmup for short & 256-token generation to avoid first-call artifacts.
    if WARMUP_RUNS:
        wp = PROMPT if "default" in TASKS else RQ5_PROMPTS[TASKS[0]]
        ids_w, attn_w = prepare_input(tokenizer, wp, device)
        for n, steps in GENERATION_CASES:
            for _ in range(WARMUP_RUNS):
                run_generation(model, ids_w, attn_w, n, steps, int(SEEDS[0]), device)
                run_generation(model, ids_w, attn_w, n, steps, int(SEEDS[0]), device,
                               cache_api=api, cache_cfg=CACHE_CONFIGS[0])
        print("Warmup completed", flush=True)

    for prompt_mode in PROMPT_MODES:
        for n, steps in GENERATION_CASES:
            for task in TASKS:
                prompt = build_prompt(tokenizer, task, prompt_mode)
                ids, attn = prepare_input(tokenizer, prompt, device)
                p_len = int(ids.shape[1])
                print(f"\nWORKLOAD {prompt_mode} {n}/{steps} task={task} prompt_tokens={p_len}",
                      flush=True)
                for seed in SEEDS:
                    for conf in CACHE_CONFIGS:
                        pi, gi, rho = int(conf[0]), int(conf[1]), float(conf[2])
                        for repeat in range(REPEATS):
                            trial_id += 1
                            before = run_generation(model, ids, attn, n, steps, seed, device)
                            cache = run_generation(model, ids, attn, n, steps, seed, device,
                                                   cache_api=api, cache_cfg=(pi, gi, rho))
                            after = run_generation(model, ids, attn, n, steps, seed, device)
                            paired = (before["ms"] + after["ms"]) / 2
                            stable = torch.equal(before["generated_ids"], after["generated_ids"])
                            row = {
                                "trial_id": trial_id, "task": task,
                                "prompt_mode": prompt_mode, "seed": int(seed),
                                "repeat": repeat, "prompt_tokens": p_len,
                                "new_tokens": int(n), "steps": int(steps),
                                "prompt_interval_steps": pi,
                                "gen_interval_steps": gi, "transfer_ratio": rho,
                                "baseline_before_ms": before["ms"],
                                "baseline_after_ms": after["ms"],
                                "paired_baseline_ms": paired,
                                "cache_ms": cache["ms"],
                                "cache_setup_ms": cache["cache_setup_ms"],
                                "saving_pct": 100 * (1 - cache["ms"] / paired),
                                "speedup": paired / cache["ms"],
                                "including_setup_saving_pct": 100 * (
                                    1 - (cache["ms"] + cache["cache_setup_ms"]) / paired
                                ),
                                "baseline_pair_stable": stable,
                                "exact_final_ids_timed": torch.equal(
                                    before["generated_ids"], cache["generated_ids"]
                                ),
                            }
                            timing_rows.append(row)
                            to_csv_safely(timing_rows, "timing_runs.csv")
                            print(f"[{trial_id:03d}/{total_trials:03d}] "
                                  f"B={paired:.1f}ms C={cache['ms']:.1f}ms "
                                  f"saving={row['saving_pct']:+.2f}% "
                                  f"exact_final={row['exact_final_ids_timed']} "
                                  f"control={'OK' if stable else 'UNSTABLE'}", flush=True)

                            # Capture full trajectories in a SEPARATE untimed pass.
                            if repeat == 0:
                                f_before = run_generation(model, ids, attn, n, steps, seed,
                                                          device, capture_trajectory=True)
                                f_cached = run_generation(model, ids, attn, n, steps, seed,
                                                          device, cache_api=api,
                                                          cache_cfg=(pi, gi, rho),
                                                          capture_trajectory=True)
                                f_after = run_generation(model, ids, attn, n, steps, seed,
                                                         device, capture_trajectory=True)
                                traj = compare_history(f_before, f_cached, p_len)
                                control = compare_history(f_before, f_after, p_len)
                                # Ensure output_history=True does not change base final IDs.
                                same_mode_base = torch.equal(before["generated_ids"],
                                                             f_before["generated_ids"])
                                same_mode_cache = torch.equal(cache["generated_ids"],
                                                              f_cached["generated_ids"])
                                fidelity_rows.append({
                                    **{k: row[k] for k in ("trial_id", "task", "prompt_mode",
                                       "seed", "prompt_tokens", "new_tokens", "steps",
                                       "prompt_interval_steps", "gen_interval_steps",
                                       "transfer_ratio")},
                                    **traj,
                                    "baseline_history_control_exact": control["exact_trajectory"],
                                    "baseline_timing_vs_history_final_equal": same_mode_base,
                                    "cache_timing_vs_history_final_equal": same_mode_cache,
                                    "baseline_trajectory_sha256": sha256_history(f_before["history"]),
                                })
                                baseline_text = tokenizer.decode(
                                    f_before["generated_ids"], skip_special_tokens=True)
                                cached_text = tokenizer.decode(
                                    f_cached["generated_ids"], skip_special_tokens=True)
                                text_rows.append({
                                    "trial_id": trial_id, "task": task, "prompt_mode": prompt_mode,
                                    "seed": seed, "new_tokens": n, "steps": steps,
                                    "exact_text": baseline_text == cached_text,
                                    "character_similarity_not_quality": SequenceMatcher(
                                        None, baseline_text, cached_text
                                    ).ratio(),
                                    "baseline_text": baseline_text,
                                    "cache_text": cached_text,
                                    # Fill in manually: task correctness CANNOT be inferred
                                    # from token similarity alone.
                                    "baseline_answer_correct_human": "",
                                    "cache_answer_correct_human": "",
                                    "quality_review_notes": "",
                                })
                                to_csv_safely(fidelity_rows, "fidelity_runs.csv")
                                to_csv_safely(text_rows, "answer_quality_review.csv")
                                tag = "EXACT" if traj["exact_trajectory"] else \
                                      f"DIFF@step{traj['first_divergent_step']}"
                                print(f"    fidelity (UNTIMED): {tag}; "
                                      f"history-control={control['exact_trajectory']}; "
                                      f"timed-vs-history="
                                      f"{same_mode_base}/{same_mode_cache}", flush=True)

                                # Hooked submodule counts: only on one task per workload,
                                # one seed, and one cache configuration; never timed.
                                probe_key = (prompt_mode, n, steps)
                                if (RUN_MODULE_PROBE and task == PROBE_TASK_KEY and
                                        seed == SEEDS[0] and conf == CACHE_CONFIGS[0] and
                                        probe_key not in examined_probe_cases):
                                    examined_probe_cases.add(probe_key)
                                    for label in ("baseline", "official_cache"):
                                        probe = ModuleWorkProbe()
                                        run_generation(
                                            model, ids, attn, n, steps, seed, device,
                                            cache_api=(api if label == "official_cache" else None),
                                            cache_cfg=(pi, gi, rho), probe=probe,
                                        )
                                        for component, item in probe.totals().items():
                                            probe_rows.append({
                                                "run_label": label, "task": task,
                                                "prompt_mode": prompt_mode, "new_tokens": n,
                                                "steps": steps, "prompt_tokens": p_len,
                                                "component": component, **item,
                                            })
                                    to_csv_safely(probe_rows, "module_work_probe.csv")
                                    print("    module-work probe recorded (UNTIMED)", flush=True)

                                # Optional costly CUDA profiler: exactly one B/C pair,
                                # separate from every timed/trajectory measurement.
                                if PROFILE_ONE_CASE and not did_profile and n == 64 \
                                        and prompt_mode == "short":
                                    did_profile = True
                                    for label in ("baseline", "official_cache"):
                                        prof = run_generation(
                                            model, ids, attn, n, steps, seed, device,
                                            cache_api=api if label == "official_cache" else None,
                                            cache_cfg=(pi, gi, rho), use_profiler=True,
                                        )["profiler"]
                                        kernel_rows += profile_rows(prof, label)
                                    to_csv_safely(kernel_rows, "torch_profiler_ops.csv")
                                    print("    torch profiler recorded (UNTIMED)", flush=True)

    summary, by_workload = summarize(timing_rows)
    if not summary.empty:
        summary.to_csv(OUTPUT_DIR / "timing_summary_by_task.csv", index=False)
    if not by_workload.empty:
        by_workload.to_csv(OUTPUT_DIR / "timing_summary_by_workload.csv", index=False)
    if fidelity_rows:
        df = pd.DataFrame(fidelity_rows)
        df.groupby(["prompt_mode", "new_tokens", "steps"], sort=True).agg(
            cases=("trial_id", "size"),
            exact_trajectory_cases=("exact_trajectory", "sum"),
            exact_final_ids_cases=("exact_final_ids", "sum"),
            history_control_cases=("baseline_history_control_exact", "sum"),
        ).reset_index().to_csv(OUTPUT_DIR / "fidelity_summary_by_workload.csv", index=False)

    metadata = {
        "experiment": "Exp9B official Dream cache diagnostic, not a published benchmark",
        "official_repo": str(repo),
        "official_source": "https://github.com/maomaocun/dLLM-cache",
        "model": MODEL_NAME,
        "generation": {
            "algorithm": ALGORITHM, "temperature": TEMPERATURE,
            "top_p": TOP_P, "top_k": TOP_K,
        },
        "cases": GENERATION_CASES, "prompt_modes": PROMPT_MODES,
        "synthetic_long_prompt_target_tokens": LONG_PROMPT_TARGET_TOKENS,
        "tasks": TASKS, "seeds": SEEDS, "repeats": REPEATS,
        "cache_configs": CACHE_CONFIGS,
        "torch": torch.__version__, "transformers": getattr(transformers, "__version__", "unknown"),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(device),
        "hf_use_cache": getattr(model.config, "use_cache", None),
        "timing": "Direct diffusion_generate, output_history=False, no history hook/probe/profiler. CUDA sync before and after. Setup excluded; measured separately.",
        "fidelity": "Separate diffusion_generate calls with output_history=True; Dream output.history covers denoising steps 1..STEPS.",
        "probes": "Submodule forward-pre-hook invocation and input token count; not FLOPs or kernel count; excluded from timing.",
        "warning": "Synthetic long prompts are not official LongBench or real industrial tasks; no automatic semantic correctness scoring. This is single-GPU, no network transfer measured.",
    }
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("\n" + "=" * 78)
    print("EXP9B DIAGNOSTIC RESULTS")
    print("=" * 78)
    if not by_workload.empty:
        print(by_workload.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if fidelity_rows:
        f = pd.read_csv(OUTPUT_DIR / "fidelity_summary_by_workload.csv")
        print("\nFidelity by workload:")
        print(f.to_string(index=False))
    if probe_rows:
        print("\nModule probe (not FLOPs, not included in timings):")
        print(pd.DataFrame(probe_rows).to_string(index=False))
    print("\nOutput directory:", OUTPUT_DIR)
    print("Review answer_quality_review.csv manually before making quality claims.")


if __name__ == "__main__":
    main()
