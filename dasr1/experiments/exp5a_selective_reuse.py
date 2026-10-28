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
    SELECTIVE_THRESHOLDS,
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

from src.selective_recompute import (
    build_selective_boundary,
    cloud_suffix_from_boundary,
)

from src.utils import (
    set_seed,
)


def main():

    print("=" * 70)
    print(
        "EXPERIMENT 5A: "
        "THRESHOLDED SELECTIVE BOUNDARY REUSE"
    )
    print("=" * 70)

    set_seed(SEED)

    # =====================================================
    # 1. LOAD MODEL
    # =====================================================

    model, tokenizer = load_dream_model(
        MODEL_NAME
    )

    # =====================================================
    # 2. CAPTURE FULL DENOISING TRAJECTORY
    # =====================================================

    hook, history = (
        create_token_history_hook()
    )

    result = generate_with_token_history(
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
        f"\nCaptured states: "
        f"{len(history)}"
    )

    print(
        f"Transitions: "
        f"{len(history) - 1}"
    )

    assert len(history) == STEPS + 1, (
        f"Expected {STEPS + 1} states, "
        f"but captured {len(history)}."
    )

    # =====================================================
    # 3. RUN SELECTIVE-REUSE SWEEP
    # =====================================================

    records = []

    for split_layer in SPLIT_LAYERS:

        print("\n" + "=" * 70)
        print(
            f"SPLIT LAYER {split_layer}"
        )
        print("=" * 70)

        # ---------------------------------------------
        # Physically partition model across two GPUs.
        #
        # This placement is setup cost and is NOT part
        # of the experiment measurement.
        # ---------------------------------------------

        cloud_rotary_emb = (
            configure_two_gpu_split(
                model=model,
                split_layer=split_layer,
                edge_device=EDGE_DEVICE,
                cloud_device=CLOUD_DEVICE,
            )
        )

        previous_exact_boundary = None

        # =================================================
        # State trajectory
        # =================================================

        for state_idx, state in enumerate(
            history
        ):

            input_ids = (
                state
                .unsqueeze(0)
                .to(EDGE_DEVICE)
            )

            # ---------------------------------------------
            # Exact physical split reference
            # ---------------------------------------------

            exact_result = (
                two_gpu_split_forward(
                    model=model,
                    input_ids=input_ids,
                    split_layer=split_layer,
                    cloud_rotary_emb=
                        cloud_rotary_emb,
                    edge_device=EDGE_DEVICE,
                    cloud_device=CLOUD_DEVICE,
                    profile=False,
                )
            )

            current_exact_boundary = (
                exact_result["boundary"]
                .detach()
            )

            current_exact_logits = (
                exact_result["logits"]
                .detach()
            )

            # ---------------------------------------------
            # State 0 is initialization only.
            #
            # No previous boundary exists, so there is no
            # selective-reuse transition to evaluate.
            # ---------------------------------------------

            if previous_exact_boundary is None:

                previous_exact_boundary = (
                    current_exact_boundary.clone()
                )

                continue

            # Human-readable:
            # transition 1 = initial -> first state
            transition = state_idx

            # =================================================
            # Threshold sweep
            # =================================================

            for threshold in (
                SELECTIVE_THRESHOLDS
            ):

                selective = (
                    build_selective_boundary(
                        current_boundary=
                            current_exact_boundary,
                        previous_boundary=
                            previous_exact_boundary,
                        threshold=threshold,
                    )
                )

                approximate_boundary = (
                    selective["boundary"]
                )

                # -----------------------------------------
                # Run cloud suffix using selectively
                # refreshed boundary.
                # -----------------------------------------

                approximate_logits = (
                    cloud_suffix_from_boundary(
                        model=model,
                        boundary_cloud=
                            approximate_boundary,
                        split_layer=split_layer,
                        cloud_rotary_emb=
                            cloud_rotary_emb,
                        cloud_device=CLOUD_DEVICE,
                    )
                )

                # =========================================
                # Boundary error
                # =========================================

                boundary_error = torch.abs(
                    approximate_boundary.float()
                    -
                    current_exact_boundary.float()
                )

                mean_boundary_error = (
                    boundary_error
                    .mean()
                    .item()
                )

                max_boundary_error = (
                    boundary_error
                    .max()
                    .item()
                )

                # =========================================
                # Logit error
                # =========================================

                logit_error = torch.abs(
                    approximate_logits.float()
                    -
                    current_exact_logits.float()
                )

                mean_logit_error = (
                    logit_error
                    .mean()
                    .item()
                )

                max_logit_error = (
                    logit_error
                    .max()
                    .item()
                )

                # =========================================
                # Argmax preservation
                # =========================================

                exact_argmax = (
                    current_exact_logits
                    .argmax(
                        dim=-1
                    )
                )

                approximate_argmax = (
                    approximate_logits
                    .argmax(
                        dim=-1
                    )
                )

                argmax_equal = (
                    exact_argmax
                    ==
                    approximate_argmax
                )

                argmax_match_ratio = (
                    argmax_equal
                    .float()
                    .mean()
                    .item()
                )

                all_argmax_same = bool(
                    torch.all(
                        argmax_equal
                    ).item()
                )

                # =========================================
                # Save result
                # =========================================

                records.append({
                    "split_layer":
                        split_layer,

                    "transition":
                        transition,

                    "threshold":
                        threshold,

                    "active_ratio":
                        selective[
                            "active_ratio"
                        ],

                    "reuse_ratio":
                        selective[
                            "reuse_ratio"
                        ],

                    "mean_boundary_error":
                        mean_boundary_error,

                    "max_boundary_error":
                        max_boundary_error,

                    "mean_logit_error":
                        mean_logit_error,

                    "max_logit_error":
                        max_logit_error,

                    "argmax_match_ratio":
                        argmax_match_ratio,

                    "all_argmax_same":
                        all_argmax_same,
                })

            # ---------------------------------------------
            # IMPORTANT:
            #
            # Exp5A is ONE-STEP oracle reuse.
            #
            # The next transition always starts from the
            # previous EXACT boundary rather than the
            # approximate boundary.
            #
            # This isolates the effect of the threshold
            # without accumulated approximation error.
            # ---------------------------------------------

            previous_exact_boundary = (
                current_exact_boundary.clone()
            )

        # ---------------------------------------------
        # Cleanup per split
        # ---------------------------------------------

        del cloud_rotary_emb

        gc.collect()

        torch.cuda.empty_cache()

    # =====================================================
    # 4. SAVE PER-TRANSITION RESULTS
    # =====================================================

    df = pd.DataFrame(
        records
    )

    os.makedirs(
        "outputs/exp5a",
        exist_ok=True,
    )

    df.to_csv(
        "outputs/exp5a/"
        "selective_reuse_per_transition.csv",
        index=False,
    )

    # =====================================================
    # 5. SUMMARY
    # =====================================================

    summary = (
        df
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

            mean_boundary_error=(
                "mean_boundary_error",
                "mean",
            ),

            max_boundary_error=(
                "max_boundary_error",
                "max",
            ),

            mean_logit_error=(
                "mean_logit_error",
                "mean",
            ),

            max_logit_error=(
                "max_logit_error",
                "max",
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
        "outputs/exp5a/"
        "selective_reuse_summary.csv",
        index=False,
    )

    # =====================================================
    # 6. TERMINAL OUTPUT
    # =====================================================

    print(
        "\n=== EXPERIMENT 5A SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nSaved:"
    )

    print(
        "outputs/exp5a/"
        "selective_reuse_per_transition.csv"
    )

    print(
        "outputs/exp5a/"
        "selective_reuse_summary.csv"
    )


if __name__ == "__main__":
    main()