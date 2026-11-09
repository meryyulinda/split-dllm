
"""Exp10A diagnostic: inspect anomalously short cached answers.

Run from dasr1/:
    python experiments/exp10a_diagnostic.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import pandas as pd
import torch

import configs.experiment_config as cfg
from src.model_loader import load_dream_model
from exp9b_official_feature_cache_diagnostics import (
    import_official,
    prepare_input,
    run_generation,
)
from exp10a_official_feature_cache_quality import make_cases


# Selected failures from Exp10A.
TARGETS = getattr(cfg, "EXP10A_DIAG_CASES", [
    ("clinic_math", 42),
    ("sensors_instruction", 42),
    ("transit_factual", 44),
])

NEW_TOKENS = int(getattr(cfg, "EXP10A_NEW_TOKENS", 128))
STEPS = int(getattr(cfg, "EXP10A_STEPS", 128))
CACHE_CFG = tuple(
    getattr(cfg, "EXP10A_CACHE_CONFIG", (100, 7, 0.25))
)

OUT = ROOT / "outputs" / "exp10a" / "diagnostic"


def collect_special_ids(tokenizer, model):
    special = set(
        int(v) for v in (tokenizer.all_special_ids or [])
    )
    by_kind = {}

    for kind in ("mask", "eos", "pad", "bos"):
        values = set()

        for obj in (
            tokenizer,
            getattr(model, "generation_config", None),
            getattr(model, "config", None),
        ):
            value = getattr(
                obj, f"{kind}_token_id", None
            )
            if isinstance(value, int):
                values.add(value)

        by_kind[kind] = sorted(values)
        special.update(values)

    return special, by_kind


def diagnose(tokenizer, output, prompt_len,
             special, by_kind):

    ids = output["generated_ids"].tolist()

    pieces = tokenizer.convert_ids_to_tokens(ids)

    visible = tokenizer.decode(
        ids, skip_special_tokens=True
    )
    raw = tokenizer.decode(
        ids, skip_special_tokens=False
    )

    kind_counts = {
        k: sum(v in set(vals) for v in ids)
        for k, vals in by_kind.items()
    }

    first_eos = next(
        (i for i, x in enumerate(ids)
         if x in by_kind["eos"]),
        None,
    )

    history_masks = []
    hist = output["history"]

    if hist:
        indexes = sorted(set((
            0,
            len(hist) // 2,
            len(hist) - 1,
        )))

        for i in indexes:
            generated = hist[i][prompt_len:].tolist()

            history_masks.append({
                "step": i + 1,
                "mask_count": sum(
                    x in by_kind["mask"]
                    for x in generated
                ),
                "non_special_count": sum(
                    x not in special
                    for x in generated
                ),
            })

    return {
        "visible_text": visible,
        "raw_decode_including_special_tokens": raw,
        "all_generated_token_ids": ids,
        "last_24_token_pieces": pieces[-24:],
        "num_generated_slots": len(ids),
        "num_non_special_tokens": sum(
            x not in special for x in ids
        ),
        "special_token_counts": kind_counts,
        "first_eos_position_zero_based": first_eos,
        "history_probe": history_masks,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "NVIDIA CUDA GPU required"
        )

    cases = {
        c["case_id"]: c
        for c in make_cases()
    }

    repo, api = import_official()

    model, tokenizer = load_dream_model(
        cfg.MODEL_NAME
    )
    model.eval()

    device = next(model.parameters()).device

    special, by_kind = collect_special_ids(
        tokenizer, model
    )

    OUT.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": cfg.MODEL_NAME,
        "official_cache_repo": str(repo),
        "cache_config": CACHE_CFG,
        "new_tokens": NEW_TOKENS,
        "steps": STEPS,
        "targets": TARGETS,
        "special_ids": sorted(special),
        "special_ids_by_kind": by_kind,
        "temperature": cfg.TEMPERATURE,
        "top_p": cfg.TOP_P,
        "top_k": cfg.TOP_K,
        "algorithm": cfg.ALGORITHM,
    }

    (OUT / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    summary = []
    details = []

    # History vs no-history comparisons.
    # Final baseline detects cache-hook cleanup issues.
    variants = [
        ("B_history", False, True),
        ("B_plain", False, False),
        ("C_history", True, True),
        ("C_plain", True, False),
        ("B_after_cache", False, False),
    ]

    for case_id, seed in TARGETS:
        if case_id not in cases:
            raise KeyError(
                f"Unknown case_id: {case_id}"
            )

        case = cases[case_id]

        ids, mask = prepare_input(
            tokenizer,
            case["prompt"],
            device,
        )

        plen = ids.shape[1]

        outputs = {}
        diagnostics = {}

        print(
            f"\n{case_id}, seed={seed}, "
            f"prompt_tokens={plen}",
            flush=True,
        )

        for name, cached, history in variants:

            outputs[name] = run_generation(
                model,
                ids,
                mask,
                NEW_TOKENS,
                STEPS,
                seed,
                device,
                cache_api=api if cached else None,
                cache_cfg=CACHE_CFG if cached else None,
                capture_trajectory=history,
            )

            diagnostics[name] = diagnose(
                tokenizer,
                outputs[name],
                plen,
                special,
                by_kind,
            )

            d = diagnostics[name]

            details.append({
                "case_id": case_id,
                "seed": seed,
                "variant": name,
                **d,
            })

            print(
                f"  {name:13s} "
                f"text={d['visible_text'][:95]!r} "
                f"non-special={d['num_non_special_tokens']} "
                f"mask={d['special_token_counts']['mask']} "
                f"eos={d['special_token_counts']['eos']}",
                flush=True,
            )

        def equal(a, b):
            return torch.equal(
                outputs[a]["generated_ids"],
                outputs[b]["generated_ids"],
            )

        row = {
            "case_id": case_id,
            "task": case["task"],
            "seed": seed,
            "gold": case["gold_answer"],
            "prompt_tokens": plen,

            "baseline_history_vs_plain_equal":
                equal("B_history", "B_plain"),

            "cache_history_vs_plain_equal":
                equal("C_history", "C_plain"),

            "baseline_stable_after_cache":
                equal("B_plain", "B_after_cache"),

            "baseline_vs_cache_equal":
                equal("B_history", "C_history"),

            "baseline_text":
                diagnostics["B_history"]["visible_text"],

            "cache_text":
                diagnostics["C_history"]["visible_text"],

            "baseline_visible_tokens":
                diagnostics["B_history"]["num_non_special_tokens"],

            "cache_visible_tokens":
                diagnostics["C_history"]["num_non_special_tokens"],

            "cache_mask_count":
                diagnostics["C_history"]["special_token_counts"]["mask"],

            "cache_eos_count":
                diagnostics["C_history"]["special_token_counts"]["eos"],

            "cache_first_eos_position":
                diagnostics["C_history"]["first_eos_position_zero_based"],
        }

        summary.append(row)

        pd.DataFrame(summary).to_csv(
            OUT / "diagnostic_summary.csv",
            index=False,
        )

        with (OUT / "raw_tokens.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for d in details:
                f.write(
                    json.dumps(d, ensure_ascii=False) + "\n"
                )

        print(
            "  baseline history=plain:",
            row["baseline_history_vs_plain_equal"],
            "| cache history=plain:",
            row["cache_history_vs_plain_equal"],
            "| baseline after cache stable:",
            row["baseline_stable_after_cache"],
            flush=True,
        )

    print("\nDiagnostic completed.")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
