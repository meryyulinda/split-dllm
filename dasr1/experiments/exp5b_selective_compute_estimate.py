import os
import sys
import gc

import torch
import pandas as pd


sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


from configs.experiment_config import (
    MODEL_NAME,
    SEED,
    MAX_NEW_TOKENS,
    STEPS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    ALGORITHM,
    PROMPT,
    SPLIT_LAYERS,
    EDGE_DEVICE,
    CLOUD_DEVICE,
    WARMUP,
    REPEATS,
    SELECTIVE_TIMING_THRESHOLDS,
)

from src.model_loader import (
    load_dream_model,
)

from src.generation import (
    generate_with_token_history,
)

from src.instrumentation import (
    create_token_history_hook,
)

from src.split_model import (
    configure_two_gpu_split,
    two_gpu_split_forward,
)

from src.utils import (
    set_seed,
)


def main():

    print("=" * 70)
    print(
        "EXPERIMENT 5B: "
        "IDEAL SELECTIVE-COMPUTE ESTIMATE"
    )
    print("=" * 70)

    set_seed(SEED)

    # =====================================================
    # 1. LOAD EXP5A RESULTS
    # =====================================================

    exp5a = pd.read_csv(
        "outputs/exp5a/"
        "selective_reuse_per_transition.csv"
    )

    exp5a = exp5a[
        exp5a["threshold"].isin(
            SELECTIVE_TIMING_THRESHOLDS
        )
    ].copy()

    print(
        "\nThresholds:"
        f" {SELECTIVE_TIMING_THRESHOLDS}"
    )

    # =====================================================
    # 2. LOAD MODEL
    # =====================================================

    model, tokenizer = load_dream_model(
        MODEL_NAME
    )

    # =====================================================
    # 3. CAPTURE SAME TRAJECTORY
    # =====================================================

    hook, history = (
        create_token_history_hook()
    )

    _ = generate_with_token_history(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        steps=STEPS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        alg=ALGORITHM,
        history_hook=hook,
    )

    print(
        f"\nCaptured states: {len(history)}"
    )

    print(
        f"Transitions: {len(history) - 1}"
    )

    assert len(history) == STEPS + 1

    # =====================================================
    # 4. MEASURE FULL CLOUD COMPUTE
    # =====================================================

    timing_records = []

    for split_layer in SPLIT_LAYERS:

        print("\n" + "=" * 70)
        print(
            f"SPLIT LAYER {split_layer}"
        )
        print("=" * 70)

        cloud_rotary_emb = (
            configure_two_gpu_split(
                model=model,
                split_layer=split_layer,
                edge_device=EDGE_DEVICE,
                cloud_device=CLOUD_DEVICE,
            )
        )

        # ---------------------------------------------
        # Warm-up once per split.
        #
        # Warm-up is not recorded.
        # ---------------------------------------------

        warmup_input = (
            history[0]
            .unsqueeze(0)
            .to(EDGE_DEVICE)
        )

        for _ in range(WARMUP):

            _ = two_gpu_split_forward(
                model=model,
                input_ids=warmup_input,
                split_layer=split_layer,
                cloud_rotary_emb=
                    cloud_rotary_emb,
                edge_device=EDGE_DEVICE,
                cloud_device=CLOUD_DEVICE,
                profile=False,
            )

        # ---------------------------------------------
        # Transitions 1..64
        #
        # Exp5A transition t corresponds to
        # history[t].
        # ---------------------------------------------

        for transition in range(
            1,
            len(history)
        ):

            input_ids = (
                history[transition]
                .unsqueeze(0)
                .to(EDGE_DEVICE)
            )

            stage2_times = []
            e2e_times = []

            for _ in range(REPEATS):

                result = (
                    two_gpu_split_forward(
                        model=model,
                        input_ids=input_ids,
                        split_layer=split_layer,
                        cloud_rotary_emb=
                            cloud_rotary_emb,
                        edge_device=EDGE_DEVICE,
                        cloud_device=CLOUD_DEVICE,
                        profile=True,
                    )
                )

                stage2_times.append(
                    result["stage2_ms"]
                )

                e2e_times.append(
                    result[
                        "measured_e2e_ms"
                    ]
                )

            avg_stage2_ms = (
                sum(stage2_times)
                / len(stage2_times)
            )

            avg_e2e_ms = (
                sum(e2e_times)
                / len(e2e_times)
            )

            timing_records.append({
                "split_layer":
                    split_layer,

                "transition":
                    transition,

                "full_stage2_ms":
                    avg_stage2_ms,

                "full_e2e_ms":
                    avg_e2e_ms,
            })

        del cloud_rotary_emb

        gc.collect()

        torch.cuda.empty_cache()

    timing_df = pd.DataFrame(
        timing_records
    )

    # =====================================================
    # 5. MERGE MEASURED TIME WITH EXP5A ACTIVE SET
    # =====================================================

    merged = pd.merge(
        exp5a,
        timing_df,
        on=[
            "split_layer",
            "transition",
        ],
        how="inner",
    )

    # =====================================================
    # 6. DERIVED IDEAL SELECTIVE COMPUTE
    # =====================================================
    #
    # IMPORTANT:
    #
    # These are NOT measured sparse GPU times.
    #
    # Assumption:
    # cloud compute scales proportionally with the
    # fraction of active rows.
    #
    # Example:
    # active_ratio = 0.4
    # full_stage2 = 20 ms
    #
    # ideal selective = 8 ms
    #
    # =====================================================

    merged[
        "ideal_selective_stage2_ms"
    ] = (
        merged["active_ratio"]
        * merged["full_stage2_ms"]
    )

    merged[
        "ideal_stage2_saving_ms"
    ] = (
        merged["full_stage2_ms"]
        -
        merged[
            "ideal_selective_stage2_ms"
        ]
    )

    merged[
        "ideal_stage2_reduction_pct"
    ] = (
        merged["reuse_ratio"]
        * 100.0
    )

    # ---------------------------------------------
    # Optional contextual estimate:
    #
    # Keep all measured non-cloud work unchanged
    # and replace only Stage 2 by its ideal
    # proportional selective estimate.
    #
    # Again: derived, NOT measured.
    # ---------------------------------------------

    merged[
        "ideal_e2e_no_overhead_ms"
    ] = (
        merged["full_e2e_ms"]
        -
        merged["full_stage2_ms"]
        +
        merged[
            "ideal_selective_stage2_ms"
        ]
    )

    merged[
        "ideal_e2e_reduction_pct"
    ] = (
        (
            merged["full_e2e_ms"]
            -
            merged[
                "ideal_e2e_no_overhead_ms"
            ]
        )
        /
        merged["full_e2e_ms"]
        * 100.0
    )

    # =====================================================
    # 7. SAVE PER-TRANSITION RESULTS
    # =====================================================

    os.makedirs(
        "outputs/exp5b",
        exist_ok=True,
    )

    merged.to_csv(
        "outputs/exp5b/"
        "selective_compute_per_transition.csv",
        index=False,
    )

    # =====================================================
    # 8. SUMMARY
    # =====================================================

    summary = (
        merged
        .groupby(
            [
                "split_layer",
                "threshold",
            ]
        )
        .agg(
            avg_active_ratio=(
                "active_ratio",
                "mean",
            ),

            avg_reuse_ratio=(
                "reuse_ratio",
                "mean",
            ),

            measured_full_stage2_ms=(
                "full_stage2_ms",
                "mean",
            ),

            ideal_selective_stage2_ms=(
                "ideal_selective_stage2_ms",
                "mean",
            ),

            ideal_stage2_saving_ms=(
                "ideal_stage2_saving_ms",
                "mean",
            ),

            ideal_stage2_reduction_pct=(
                "ideal_stage2_reduction_pct",
                "mean",
            ),

            measured_full_e2e_ms=(
                "full_e2e_ms",
                "mean",
            ),

            ideal_e2e_no_overhead_ms=(
                "ideal_e2e_no_overhead_ms",
                "mean",
            ),

            ideal_e2e_reduction_pct=(
                "ideal_e2e_reduction_pct",
                "mean",
            ),

            mean_logit_error=(
                "mean_logit_error",
                "mean",
            ),

            avg_argmax_match_ratio=(
                "argmax_match_ratio",
                "mean",
            ),

            exact_transition_ratio=(
                "all_argmax_same",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        "outputs/exp5b/"
        "selective_compute_summary.csv",
        index=False,
    )

    # =====================================================
    # 9. PRINT
    # =====================================================

    print(
        "\n=== EXPERIMENT 5B SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print("\nSaved:")

    print(
        "outputs/exp5b/"
        "selective_compute_per_transition.csv"
    )

    print(
        "outputs/exp5b/"
        "selective_compute_summary.csv"
    )

    print(
        "\nNOTE:"
        "\n'measured_full_*' = actual measured GPU latency"
        "\n'ideal_*' = derived estimate under proportional"
        " active-row scaling; NOT measured sparse execution."
    )


if __name__ == "__main__":
    main()