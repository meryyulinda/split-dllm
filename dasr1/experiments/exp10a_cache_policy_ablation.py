
"""Exp10A cache-policy ablation.

Run from dasr1/:
    python experiments/exp10a_cache_policy_ablation.py

Requires:
    exp9b_official_feature_cache_diagnostics.py
    exp10a_official_feature_cache_quality.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import configs.experiment_config as cfg
from src.model_loader import load_dream_model

from exp9b_official_feature_cache_diagnostics import (
    import_official,
    prepare_input,
    run_generation,
    compare_history,
)
from exp10a_official_feature_cache_quality import (
    make_cases,
    advisory_score,
)

TARGETS = list(getattr(cfg, "EXP10A_ABLATION_TARGETS", [
    ("clinic_math", 42),
    ("sensors_instruction", 42),
    ("transit_factual", 44),
]))

POLICIES = [
    tuple(v) for v in getattr(
        cfg, "EXP10A_ABLATION_POLICIES", [
            (100, 7, 0.25),
            (100, 3, 0.25),
            (100, 1, 0.25),
            (100, 7, 0.50),
            (1, 7, 0.25),
            (1, 1, 1.00),
        ]
    )
]

N_TOKENS = int(getattr(cfg, "EXP10A_NEW_TOKENS", 128))
STEPS = int(getattr(cfg, "EXP10A_STEPS", 128))

OUT = ROOT / "outputs" / "exp10a" / "policy_ablation"


def inspect_output(tokenizer, result):
    ids = result["generated_ids"].tolist()

    eos_id = tokenizer.eos_token_id
    mask_id = getattr(tokenizer, "mask_token_id", None)

    special = set(tokenizer.all_special_ids or [])

    if eos_id is None:
        raise RuntimeError("EOS token ID is missing")

    first_eos = next(
        (i for i, token in enumerate(ids)
         if token == eos_id),
        None,
    )

    return {
        "text": tokenizer.decode(
            ids, skip_special_tokens=True
        ),
        "visible_tokens": sum(
            token not in special for token in ids
        ),
        "eos_count": sum(
            token == eos_id for token in ids
        ),
        "mask_count": (
            sum(token == mask_id for token in ids)
            if mask_id is not None else None
        ),
        "first_eos": first_eos,
        "first_24_ids": ids[:24],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    cases = {
        c["case_id"]: c for c in make_cases()
    }

    repo, cache_api = import_official()

    model, tokenizer = load_dream_model(cfg.MODEL_NAME)
    model.eval()

    device = next(model.parameters()).device

    OUT.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": cfg.MODEL_NAME,
        "official_cache_repo": str(repo),
        "targets": TARGETS,
        "policies": POLICIES,
        "new_tokens": N_TOKENS,
        "steps": STEPS,
        "temperature": cfg.TEMPERATURE,
        "top_p": cfg.TOP_P,
        "top_k": cfg.TOP_K,
        "algorithm": cfg.ALGORITHM,
    }

    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    rows = []

    for case_id, seed in TARGETS:

        case = cases[case_id]

        input_ids, attention_mask = prepare_input(
            tokenizer,
            case["prompt"],
            device,
        )

        prompt_len = input_ids.shape[1]

        # Full recomputation reference
        baseline = run_generation(
            model,
            input_ids,
            attention_mask,
            N_TOKENS,
            STEPS,
            seed,
            device,
            capture_trajectory=True,
        )

        baseline_info = inspect_output(
            tokenizer, baseline
        )

        baseline_score = advisory_score(
            case, baseline_info["text"]
        )

        print(
            f"\nCASE={case_id} seed={seed}",
            flush=True,
        )

        print(
            "BASELINE:",
            repr(baseline_info["text"]),
            flush=True,
        )

        for policy in POLICIES:

            cached = run_generation(
                model,
                input_ids,
                attention_mask,
                N_TOKENS,
                STEPS,
                seed,
                device,
                cache_api=cache_api,
                cache_cfg=policy,
                capture_trajectory=True,
            )

            cache_info = inspect_output(
                tokenizer, cached
            )

            score = advisory_score(
                case, cache_info["text"]
            )

            fidelity = compare_history(
                baseline, cached, prompt_len
            )

            row = {
                "case_id": case_id,
                "task": case["task"],
                "seed": seed,
                "prompt_tokens": prompt_len,

                "prompt_interval": policy[0],
                "gen_interval": policy[1],
                "transfer_ratio": policy[2],

                "gold_answer": case["gold_answer"],

                "baseline_text": baseline_info["text"],
                "cache_text": cache_info["text"],

                "baseline_gold_advisory":
                    baseline_score["advisory_gold_match"],

                "cache_gold_advisory":
                    score["advisory_gold_match"],

                "baseline_instruction_format":
                    baseline_score["instruction_format_pass"],

                "cache_instruction_format":
                    score["instruction_format_pass"],

                "baseline_numeric_advisory":
                    baseline_score["advisory_extracted_answer"],

                "cache_numeric_advisory":
                    score["advisory_extracted_answer"],

                "baseline_visible_tokens":
                    baseline_info["visible_tokens"],

                "cache_visible_tokens":
                    cache_info["visible_tokens"],

                "baseline_eos_count":
                    baseline_info["eos_count"],

                "cache_eos_count":
                    cache_info["eos_count"],

                "cache_mask_count":
                    cache_info["mask_count"],

                "cache_first_eos":
                    cache_info["first_eos"],

                "cache_first_24_ids":
                    str(cache_info["first_24_ids"]),

                **fidelity,
            }

            rows.append(row)

            pd.DataFrame(rows).to_csv(
                OUT / "policy_ablation_results.csv",
                index=False,
            )

            print(
                f"  policy={policy}"
                f" exact={fidelity['exact_final_ids']}"
                f" diff={fidelity['first_divergent_step']}"
                f" eos={cache_info['eos_count']}"
                f" gold={score['advisory_gold_match']}"
                f" text={cache_info['text'][:100]!r}",
                flush=True,
            )

        # Check that hooks were removed and baseline
        # generation remains unchanged.
        baseline_after = run_generation(
            model,
            input_ids,
            attention_mask,
            N_TOKENS,
            STEPS,
            seed,
            device,
            capture_trajectory=True,
        )

        stable = compare_history(
            baseline, baseline_after, prompt_len
        )["exact_trajectory"]

        for row in rows:
            if (
                row["case_id"] == case_id
                and row["seed"] == seed
            ):
                row["baseline_stable_after_policies"] = stable

        pd.DataFrame(rows).to_csv(
            OUT / "policy_ablation_results.csv",
            index=False,
        )

        print("Baseline stable after caching:", stable)

    print("\nCompleted. Output:", OUT)


if __name__ == "__main__":
    main()
