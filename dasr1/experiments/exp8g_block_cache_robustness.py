
"""Exp8G: Dream multi-block residual-cache robustness.

Run from dasr1/:
    python experiments/exp8g_block_cache_robustness.py

Fidelity first; optional B-C-B timing via EXP8G_RUN_TIMING.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import hashlib
import json
import time
from contextlib import ExitStack
from difflib import SequenceMatcher

import pandas as pd
import torch

import configs.experiment_config as cfg
from src.model_loader import load_dream_model
from src.utils import set_seed


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = cfg.MODEL_NAME

TASKS = list(
    getattr(
        cfg,
        "EXP8G_TASK_KEYS",
        ["factual_qa", "reasoning", "instruction"],
    )
)

SEEDS = list(
    getattr(
        cfg,
        "EXP8G_SEEDS",
        [cfg.SEED, cfg.SEED + 1, cfg.SEED + 2],
    )
)

GROUPS = [
    tuple(x)
    for x in getattr(
        cfg,
        "EXP8G_LAYER_GROUPS",
        [
            [8, 20, 24],
            [4, 8, 12, 16, 18, 20, 22, 24],
        ],
    )
]

INTERVALS = list(
    getattr(cfg, "EXP8G_REFRESH_INTERVALS", [2, 4])
)

N_TOKENS = int(
    getattr(cfg, "EXP8G_NEW_TOKENS", 64)
)

STEPS = int(
    getattr(cfg, "EXP8G_STEPS", 64)
)

RUN_TIMING = bool(
    getattr(cfg, "EXP8G_RUN_TIMING", False)
)

TIMING_REPEATS = int(
    getattr(cfg, "EXP8G_TIMING_REPEATS", 2)
)

WARMUPS = int(
    getattr(cfg, "EXP8G_WARMUP_RUNS", 1)
)

CHECK_HF_TOGGLE = bool(
    getattr(cfg, "EXP8G_VALIDATE_HF_TOGGLE", True)
)

OUT = ROOT / "outputs" / "exp8g"


# ============================================================
# 2. UTILITIES
# ============================================================

def save(rows, filename):
    if rows:
        pd.DataFrame(rows).to_csv(
            OUT / filename,
            index=False,
        )


def disable_hf_kv(model):
    """
    Disable built-in HF KV caching.

    Prevents stale HF KV states during approximate
    block-residual reuse.
    """
    for obj in (model, model.model):
        obj.config.use_cache = False
        obj.config.output_attentions = False

    gen_cfg = getattr(model, "generation_config", None)

    if gen_cfg is not None:
        gen_cfg.use_cache = False
        gen_cfg.output_attentions = False


# ============================================================
# 3. BLOCK RESIDUAL CACHE
# ============================================================

class ResidualBlockCache:
    """
    Cache the residual of one transformer block.

    Refresh:
        R_t = block(x_t) - x_t

    Reuse:
        y_t = x_t + R_cached

    On reuse, original block.forward is NOT executed.
    """

    def __init__(self, model, layer, refresh_interval):
        self.layer = int(layer)  # one-based
        self.k = int(refresh_interval)

        self.module = model.model.layers[self.layer - 1]

        self.original = None
        self.residual = None

        self.refresh_at = -1
        self.returns_tuple = None
        self.extra = ()

        self.calls = 0
        self.executed = 0
        self.skipped = 0

    def forward(self, *args, **kwargs):

        if (
            kwargs.get("use_cache", False)
            or kwargs.get("past_key_value") is not None
        ):
            raise RuntimeError(
                "Exp8G requires HF use_cache=False"
            )

        if kwargs.get("output_attentions", False):
            raise RuntimeError(
                "Exp8G requires output_attentions=False"
            )

        x = (
            args[0]
            if args
            else kwargs.get("hidden_states")
        )

        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise RuntimeError(
                "Unexpected Dream block input"
            )

        i = self.calls
        self.calls += 1

        refresh = (
            self.residual is None
            or i - self.refresh_at >= self.k
        )

        # -------------------------------
        # REFRESH: execute original block
        # -------------------------------
        if refresh:

            x_before = x.detach().float().clone()

            output = self.original(*args, **kwargs)

            y = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            if (
                not isinstance(y, torch.Tensor)
                or y.shape != x.shape
            ):
                raise RuntimeError(
                    "Unexpected Dream block output"
                )

            extra = (
                output[1:]
                if isinstance(output, tuple)
                else ()
            )

            if any(v is not None for v in extra):
                raise RuntimeError(
                    "Non-None attention/KV output; "
                    "refuse cache reuse"
                )

            self.residual = (
                y.detach().float() - x_before
            )

            self.returns_tuple = isinstance(
                output, tuple
            )

            self.extra = extra
            self.refresh_at = i
            self.executed += 1

            return output

        # -------------------------------
        # REUSE: skip original block
        # -------------------------------
        if (
            self.residual.shape != x.shape
            or self.residual.device != x.device
        ):
            raise RuntimeError(
                "Residual shape/device mismatch"
            )

        self.skipped += 1

        y = (
            x.float() + self.residual
        ).to(dtype=x.dtype)

        if self.returns_tuple:
            return (y, *self.extra)

        return y

    def __enter__(self):
        self.original = self.module.forward
        self.module.forward = self.forward
        return self

    def __exit__(self, *_):
        self.module.forward = self.original
        self.original = None
        return False

    def counts(self):
        assert self.calls == self.executed + self.skipped

        return {
            "layer": self.layer,
            "calls": self.calls,
            "executed": self.executed,
            "skipped": self.skipped,
        }


# ============================================================
# 4. PROMPT PREPARATION
# ============================================================

def prepare(tokenizer, task, device):

    prompt = cfg.RQ5_PROMPTS[task]

    encoded = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    ids = encoded["input_ids"].to(device)

    mask = encoded.get("attention_mask")

    if mask is None:
        mask = torch.ones_like(ids)
    else:
        mask = mask.to(device)

    return ids, mask


# ============================================================
# 5. GENERATION
# ============================================================

def generate(
    model,
    ids,
    mask,
    seed,
    group=(),
    k=None,
    history=False,
):
    """
    Direct Dream generation.

    Cache objects are created anew every generation.
    Therefore, no cache state leaks between trials.

    When history=False, output_history instrumentation
    is disabled for E2E latency measurement.
    """

    set_seed(int(seed))

    device = ids.device

    caches = [
        ResidualBlockCache(model, layer, k)
        for layer in group
    ]

    with ExitStack() as stack:

        for cache in caches:
            stack.enter_context(cache)

        torch.cuda.synchronize(device)
        start = time.perf_counter()

        with torch.inference_mode():

            result = model.diffusion_generate(
                ids,
                attention_mask=mask,
                max_new_tokens=N_TOKENS,
                steps=STEPS,
                temperature=cfg.TEMPERATURE,
                top_p=cfg.TOP_P,
                top_k=cfg.TOP_K,
                alg=cfg.ALGORITHM,
                output_history=history,
                return_dict_in_generate=True,
            )

        torch.cuda.synchronize(device)

        ms = (
            time.perf_counter() - start
        ) * 1000

        generated = (
            result.sequences[0, ids.shape[1]:]
            .detach()
            .cpu()
            .clone()
        )

        states = None

        if history:

            raw = getattr(result, "history", None)

            if raw is None or len(raw) != STEPS:
                raise RuntimeError(
                    "Unexpected Dream output_history length"
                )

            states = [
                h[0].detach().cpu().clone()
                for h in raw
            ]

            if not torch.equal(
                states[-1][ids.shape[1]:],
                generated,
            ):
                raise RuntimeError(
                    "History final state differs "
                    "from sequences"
                )

        counts = [
            cache.counts()
            for cache in caches
        ]

        if any(
            c["calls"] != STEPS
            for c in counts
        ):
            raise RuntimeError(
                "Expected one forward per block "
                "per denoising step"
            )

    return {
        "ms": ms,
        "ids": generated,
        "states": states,
        "counts": counts,
    }


# ============================================================
# 6. TRAJECTORY FIDELITY
# ============================================================

def compare(base, cached, prompt_len):

    if (
        base["states"] is None
        or cached["states"] is None
    ):
        raise RuntimeError(
            "History required for fidelity"
        )

    a = base["states"]
    b = cached["states"]

    if len(a) != len(b):
        raise RuntimeError(
            "Different number of states"
        )

    exact_states = [
        torch.equal(x, y)
        for x, y in zip(a, b)
    ]

    token_agreements = [
        (
            x[prompt_len:] == y[prompt_len:]
        ).float().mean().item()
        for x, y in zip(a, b)
    ]

    return {
        "exact_trajectory": all(exact_states),

        "exact_transition_states": sum(
            exact_states
        ),

        "first_divergent_step": next(
            (
                i + 1
                for i, ok in enumerate(exact_states)
                if not ok
            ),
            None,
        ),

        "mean_token_agreement": (
            sum(token_agreements)
            / len(token_agreements)
        ),

        "final_token_agreement": (
            token_agreements[-1]
        ),

        "exact_final_ids": torch.equal(
            base["ids"],
            cached["ids"],
        ),
    }


def fingerprint(states):

    h = hashlib.sha256()

    for state in states:
        h.update(
            state.numpy().tobytes()
        )

    return h.hexdigest()


# ============================================================
# 7. RESULTS SUMMARY
# ============================================================

def summarize(fidelity, baselines, timing):

    if fidelity:

        df = pd.DataFrame(fidelity)

        cols = [
            "block_group",
            "num_cached_blocks",
            "refresh_interval",
        ]

        summary = (
            df.groupby(cols, sort=True)
            .agg(
                cases=("case_id", "size"),

                exact_trajectory_cases=(
                    "exact_trajectory",
                    "sum",
                ),

                exact_final_ids_cases=(
                    "exact_final_ids",
                    "sum",
                ),

                min_exact_transition_states=(
                    "exact_transition_states",
                    "min",
                ),

                mean_token_agreement=(
                    "mean_token_agreement",
                    "mean",
                ),
            )
            .reset_index()
        )

        summary["exact_trajectory_rate"] = (
            summary["exact_trajectory_cases"]
            / summary["cases"]
        )

        summary.to_csv(
            OUT / "fidelity_summary.csv",
            index=False,
        )

        by_task = (
            df.groupby(
                ["task", *cols],
                sort=True,
            )
            .agg(
                cases=("case_id", "size"),

                exact_trajectory_cases=(
                    "exact_trajectory",
                    "sum",
                ),

                exact_final_ids_cases=(
                    "exact_final_ids",
                    "sum",
                ),
            )
            .reset_index()
        )

        by_task.to_csv(
            OUT / "fidelity_by_task.csv",
            index=False,
        )

        print(
            "\nFIDELITY SUMMARY:\n"
            + summary.to_string(index=False),
            flush=True,
        )

        print(
            "\nBY TASK:\n"
            + by_task.to_string(index=False),
            flush=True,
        )

    if baselines:

        df = pd.DataFrame(baselines)

        diversity = (
            df.groupby("task", sort=True)
            .agg(
                seeds_tested=(
                    "seed",
                    "nunique",
                ),

                distinct_trajectories=(
                    "trajectory_sha256",
                    "nunique",
                ),

                stable_baselines=(
                    "baseline_stable",
                    "sum",
                ),
            )
            .reset_index()
        )

        diversity.to_csv(
            OUT / "baseline_seed_diversity.csv",
            index=False,
        )

    if timing:

        df = pd.DataFrame(timing)

        df = df[
            df.baseline_pair_stable
        ]

        if not df.empty:

            summary = (
                df.groupby(
                    [
                        "block_group",
                        "num_cached_blocks",
                        "refresh_interval",
                    ],
                    sort=True,
                )
                .agg(
                    trials=(
                        "trial_id",
                        "size",
                    ),

                    median_paired_saving_pct=(
                        "saving_pct",
                        "median",
                    ),

                    mean_paired_saving_pct=(
                        "saving_pct",
                        "mean",
                    ),

                    min_paired_saving_pct=(
                        "saving_pct",
                        "min",
                    ),

                    exact_final_ids_trials=(
                        "exact_timed_final",
                        "sum",
                    ),
                )
                .reset_index()
            )

            summary.to_csv(
                OUT / "timing_summary.csv",
                index=False,
            )

            print(
                "\nTIMING SUMMARY:\n"
                + summary.to_string(index=False),
                flush=True,
            )


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    for task in TASKS:
        if task not in cfg.RQ5_PROMPTS:
            raise KeyError(
                f"Missing RQ5_PROMPTS[{task!r}]"
            )

    if (
        not GROUPS
        or not INTERVALS
        or any(k < 2 for k in INTERVALS)
    ):
        raise ValueError(
            "Invalid block groups/refresh intervals"
        )

    set_seed(cfg.SEED)

    model, tokenizer = load_dream_model(
        MODEL_NAME
    )

    model.eval()

    device = next(
        model.parameters()
    ).device

    n_layers = len(
        model.model.layers
    )

    if any(
        (
            not group
            or len(set(group)) != len(group)
            or any(
                layer < 1 or layer > n_layers
                for layer in group
            )
        )
        for group in GROUPS
    ):
        raise ValueError(
            "Invalid block layer indices"
        )

    print("=" * 75)
    print(
        "EXP8G: BLOCK CACHE ROBUSTNESS "
        "| Dream | single GPU"
    )
    print("=" * 75)

    print(
        "Tasks", TASKS,
        "Seeds", SEEDS,
        "Groups", GROUPS,
        "k", INTERVALS,
    )

    print(
        f"{N_TOKENS} generated tokens / "
        f"{STEPS} steps; timing={RUN_TIMING}"
    )

    # --------------------------------------
    # HF KV cache sanity check
    # --------------------------------------

    ids0, mask0 = prepare(
        tokenizer,
        TASKS[0],
        device,
    )

    if CHECK_HF_TOGGLE:

        before = generate(
            model,
            ids0,
            mask0,
            SEEDS[0],
            history=True,
        )

        disable_hf_kv(model)

        after = generate(
            model,
            ids0,
            mask0,
            SEEDS[0],
            history=True,
        )

        toggle_result = compare(
            before,
            after,
            ids0.shape[1],
        )

        if not toggle_result["exact_trajectory"]:
            raise RuntimeError(
                "HF cache toggle changed "
                "reference trajectory"
            )

        print(
            "HF KV-toggle sanity check: PASS"
        )

    else:
        disable_hf_kv(model)

    # --------------------------------------
    # Warmup
    # --------------------------------------

    for _ in range(WARMUPS):

        generate(
            model,
            ids0,
            mask0,
            SEEDS[0],
        )

        generate(
            model,
            ids0,
            mask0,
            SEEDS[0],
            group=GROUPS[0],
            k=INTERVALS[0],
        )

    # --------------------------------------
    # Experiment loops
    # --------------------------------------

    fidelity = []
    baselines = []
    quality = []
    block_counts = []
    timing = []

    total = (
        len(TASKS)
        * len(SEEDS)
        * len(GROUPS)
        * len(INTERVALS)
    )

    for task in TASKS:

        ids, mask = prepare(
            tokenizer,
            task,
            device,
        )

        prompt_len = ids.shape[1]

        for seed in SEEDS:

            # Baseline trajectory
            baseline = generate(
                model,
                ids,
                mask,
                seed,
                history=True,
            )

            # Independent baseline control
            control = generate(
                model,
                ids,
                mask,
                seed,
                history=True,
            )

            stable = compare(
                baseline,
                control,
                prompt_len,
            )["exact_trajectory"]

            if not stable:
                raise RuntimeError(
                    f"Non-deterministic baseline "
                    f"for task={task}, seed={seed}"
                )

            sha = fingerprint(
                baseline["states"]
            )

            baselines.append({
                "task": task,
                "seed": seed,
                "trajectory_sha256": sha,
                "baseline_stable": True,
            })

            save(
                baselines,
                "baseline_trajectories.csv",
            )

            text_base = tokenizer.decode(
                baseline["ids"],
                skip_special_tokens=True,
            )

            # Compare all cache configurations
            for group in GROUPS:

                label = (
                    "L"
                    + "_L".join(
                        map(str, group)
                    )
                )

                for k in INTERVALS:

                    cached = generate(
                        model,
                        ids,
                        mask,
                        seed,
                        group=group,
                        k=k,
                        history=True,
                    )

                    result = compare(
                        baseline,
                        cached,
                        prompt_len,
                    )

                    skipped = sum(
                        c["skipped"]
                        for c in cached["counts"]
                    )

                    possible = (
                        len(group) * STEPS
                    )

                    expected = (
                        len(group)
                        * (
                            STEPS
                            - (STEPS + k - 1) // k
                        )
                    )

                    if skipped != expected:
                        raise RuntimeError(
                            f"Skip-count mismatch: "
                            f"{skipped} != {expected}"
                        )

                    case_id = len(fidelity) + 1

                    fidelity.append({
                        "case_id": case_id,
                        "task": task,
                        "seed": seed,
                        "prompt_tokens": prompt_len,
                        "block_group": label,
                        "num_cached_blocks": len(group),
                        "refresh_interval": k,
                        "skipped_calls": skipped,
                        "possible_calls": possible,
                        "baseline_sha256": sha,
                        **result,
                    })

                    save(
                        fidelity,
                        "fidelity_runs.csv",
                    )

                    # ----------------------
                    # Generated text review
                    # ----------------------

                    text_cached = tokenizer.decode(
                        cached["ids"],
                        skip_special_tokens=True,
                    )

                    quality.append({
                        "case_id": case_id,
                        "task": task,
                        "seed": seed,
                        "block_group": label,
                        "k": k,

                        "exact_final_ids": (
                            result["exact_final_ids"]
                        ),

                        "final_text_exact": (
                            text_base == text_cached
                        ),

                        "character_similarity_not_quality": (
                            SequenceMatcher(
                                None,
                                text_base,
                                text_cached,
                            ).ratio()
                        ),

                        "baseline_text": text_base,
                        "cached_text": text_cached,

                        "baseline_answer_correct_human": "",
                        "cache_answer_correct_human": "",
                        "notes": "",
                    })

                    save(
                        quality,
                        "answer_quality_review.csv",
                    )

                    # ----------------------
                    # Block execution counts
                    # ----------------------

                    for counter in cached["counts"]:

                        block_counts.append({
                            "case_id": case_id,
                            "task": task,
                            "seed": seed,
                            "block_group": label,
                            "k": k,
                            **counter,
                        })

                    save(
                        block_counts,
                        "per_layer_execution_counts.csv",
                    )

                    # ----------------------
                    # Console progress
                    # ----------------------

                    if result["exact_trajectory"]:
                        status = "EXACT"
                    else:
                        step = result[
                            "first_divergent_step"
                        ]
                        status = f"DIFF@step{step}"

                    final_status = (
                        "EXACT"
                        if result["exact_final_ids"]
                        else "DIFF"
                    )

                    print(
                        f"[{case_id:02d}/{total:02d}] "
                        f"{task:12s} "
                        f"seed={seed} "
                        f"{label:29s} "
                        f"k={k} "
                        f"{status:13s} "
                        f"final={final_status} "
                        f"skipped={skipped}/{possible}",
                        flush=True,
                    )

                    # ----------------------
                    # Optional latency test
                    # ----------------------

                    if RUN_TIMING:

                        for repeat in range(
                            TIMING_REPEATS
                        ):

                            b1 = generate(
                                model,
                                ids,
                                mask,
                                seed,
                            )

                            c = generate(
                                model,
                                ids,
                                mask,
                                seed,
                                group=group,
                                k=k,
                            )

                            b2 = generate(
                                model,
                                ids,
                                mask,
                                seed,
                            )

                            paired = (
                                b1["ms"] + b2["ms"]
                            ) / 2

                            trial_id = (
                                len(timing) + 1
                            )

                            timing.append({
                                "trial_id": trial_id,
                                "case_id": case_id,
                                "repeat": repeat,
                                "task": task,
                                "seed": seed,
                                "block_group": label,
                                "num_cached_blocks": (
                                    len(group)
                                ),
                                "refresh_interval": k,

                                "baseline_before_ms": (
                                    b1["ms"]
                                ),
                                "baseline_after_ms": (
                                    b2["ms"]
                                ),
                                "paired_baseline_ms": paired,
                                "cached_ms": c["ms"],

                                "saving_pct": (
                                    100
                                    * (
                                        1
                                        - c["ms"] / paired
                                    )
                                ),

                                "speedup": (
                                    paired / c["ms"]
                                ),

                                "baseline_pair_stable": (
                                    torch.equal(
                                        b1["ids"],
                                        b2["ids"],
                                    )
                                ),

                                "exact_timed_final": (
                                    torch.equal(
                                        b1["ids"],
                                        c["ids"],
                                    )
                                ),

                                "timed_baseline_matches_history": (
                                    torch.equal(
                                        b1["ids"],
                                        baseline["ids"],
                                    )
                                ),

                                "timed_cached_matches_history": (
                                    torch.equal(
                                        c["ids"],
                                        cached["ids"],
                                    )
                                ),
                            })

                            save(
                                timing,
                                "timing_runs.csv",
                            )

                            saving = timing[-1][
                                "saving_pct"
                            ]

                            print(
                                f"  Timing rep {repeat}: "
                                f"saving={saving:+.2f}%",
                                flush=True,
                            )

            # Save summaries after each task/seed
            summarize(
                fidelity,
                baselines,
                timing,
            )

    # --------------------------------------
    # Final metadata
    # --------------------------------------

    metadata = {
        "model": MODEL_NAME,
        "tasks": TASKS,
        "seeds": SEEDS,
        "groups": GROUPS,
        "intervals": INTERVALS,
        "new_tokens": N_TOKENS,
        "steps": STEPS,
        "run_timing": RUN_TIMING,
        "hf_cache_disabled": True,

        "note": (
            "Approximate residual reuse. "
            "Exact tokens do not imply "
            "exact logits or equivalent "
            "semantic quality."
        ),
    }

    (
        OUT / "experiment_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(
        "\nDONE. Results saved to",
        OUT,
    )


if __name__ == "__main__":
    main()
