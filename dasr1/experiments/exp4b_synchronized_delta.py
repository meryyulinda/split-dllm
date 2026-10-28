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
    WARMUP,
    REPEATS,
    EDGE_DEVICE,
    CLOUD_DEVICE,
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
    two_gpu_delta_split_forward,
)

from src.utils import (
    set_seed,
)


def main():

    print("=" * 60)
    print(
        "EXPERIMENT 4B: "
        "SYNCHRONIZED DELTA TRANSMISSION"
    )
    print("=" * 60)

    # -------------------------------------------------
    # 0. Verify GPUs
    # -------------------------------------------------

    gpu_count = torch.cuda.device_count()

    print(
        f"Available GPUs: {gpu_count}"
    )

    if gpu_count < 2:
        raise RuntimeError(
            "Experiment 4B requires "
            "at least two GPUs."
        )

    print(
        f"EDGE_DEVICE : {EDGE_DEVICE}"
    )

    print(
        f"CLOUD_DEVICE: {CLOUD_DEVICE}"
    )

    for gpu_idx in range(gpu_count):

        print(
            f"GPU {gpu_idx}: "
            f"{torch.cuda.get_device_name(gpu_idx)}"
        )

    # -------------------------------------------------
    # 1. Load full Dream model
    #
    # Model is initially intact on its default GPU.
    # We need this before physical partitioning to:
    #
    #   A. generate the official diffusion trajectory
    #   B. capture full-model reference predictions
    # -------------------------------------------------

    set_seed(SEED)

    model, tokenizer = load_dream_model(
        MODEL_NAME
    )

    model.eval()

    # -------------------------------------------------
    # 2. Capture official Dream trajectory
    # -------------------------------------------------

    hook, history = (
        create_token_history_hook()
    )

    set_seed(SEED)

    _ = generate_with_token_history(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        steps=STEPS,
        temperature=TEMPERATURE,
        history_hook=hook,
        top_p=TOP_P,
        top_k=TOP_K,
        alg=ALGORITHM,
    )

    print(
        f"\nCaptured states: "
        f"{len(history)}"
    )

    # -------------------------------------------------
    # 3. Capture full-model reference predictions
    #
    # This happens BEFORE physical model partitioning.
    # -------------------------------------------------

    print(
        "\nCapturing full-model "
        "reference predictions..."
    )

    reference_argmax = []

    reference_device = (
        next(model.parameters()).device
    )

    for state in history:

        input_ids = (
            state
            .unsqueeze(0)
            .to(reference_device)
        )

        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.bool,
            device=reference_device,
        )

        with torch.no_grad():

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        reference_argmax.append(
            output.logits
            .argmax(dim=-1)
            .cpu()
        )

    # Release the last full-model output.
    del output

    gc.collect()
    torch.cuda.empty_cache()

    print(
        "Reference predictions captured."
    )

    # -------------------------------------------------
    # 4. Sweep physical split locations
    # -------------------------------------------------

    records = []

    for split_layer in SPLIT_LAYERS:

        print(
            "\n" + "=" * 60
        )

        print(
            f"DELTA SPLIT LAYER "
            f"{split_layer}"
        )

        print(
            "=" * 60
        )

        # ---------------------------------------------
        # Physically place model layers.
        #
        # Placement cost is setup cost and therefore
        # NOT included in inference latency.
        # ---------------------------------------------

        cloud_rotary_emb = (
            configure_two_gpu_split(
                model=model,
                split_layer=split_layer,
                edge_device=EDGE_DEVICE,
                cloud_device=CLOUD_DEVICE,
            )
        )

        # ---------------------------------------------
        # Warmup
        #
        # Use the ordinary full-boundary split here.
        # This warms kernels without changing the
        # synchronized delta state.
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

        torch.cuda.synchronize(
            EDGE_DEVICE
        )

        torch.cuda.synchronize(
            CLOUD_DEVICE
        )

        # ---------------------------------------------
        # Synchronized state
        #
        # Edge and cloud each keep their copy of the
        # previous boundary.
        #
        # For the first state both are None, therefore
        # a full initial synchronization is required.
        # ---------------------------------------------

        prev_boundary_edge = None
        prev_boundary_cloud = None

        # ---------------------------------------------
        # Replay diffusion trajectory
        # ---------------------------------------------

        for state_idx, state in enumerate(
            history
        ):

            input_ids = (
                state
                .unsqueeze(0)
                .to(EDGE_DEVICE)
            )

            # -----------------------------------------
            # Timing arrays
            # -----------------------------------------

            stage1_times = []
            encode_times = []
            transfer_times = []
            reconstruction_times = []
            stage2_times = []

            component_sum_times = []
            e2e_times = []

            last_output = None

            # -----------------------------------------
            # Repeat SAME transition several times.
            #
            # IMPORTANT:
            # prev_boundary_edge / cloud are NOT
            # updated inside this loop.
            #
            # Example:
            #
            #   current state = 11
            #   previous state = 10
            #
            # Every repeat measures:
            #
            #   state 10 -> state 11
            # -----------------------------------------

            for _ in range(REPEATS):

                output = (
                    two_gpu_delta_split_forward(
                        model=model,
                        input_ids=input_ids,
                        split_layer=
                            split_layer,

                        cloud_rotary_emb=
                            cloud_rotary_emb,

                        prev_boundary_edge=
                            prev_boundary_edge,

                        prev_boundary_cloud=
                            prev_boundary_cloud,

                        edge_device=
                            EDGE_DEVICE,

                        cloud_device=
                            CLOUD_DEVICE,

                        profile=True,
                    )
                )

                stage1_times.append(
                    output["stage1_ms"]
                )

                encode_times.append(
                    output["encode_ms"]
                )

                transfer_times.append(
                    output["transfer_ms"]
                )

                reconstruction_times.append(
                    output[
                        "reconstruction_ms"
                    ]
                )

                stage2_times.append(
                    output["stage2_ms"]
                )

                component_sum_times.append(
                    output[
                        "component_sum_ms"
                    ]
                )

                e2e_times.append(
                    output[
                        "measured_e2e_ms"
                    ]
                )

                last_output = output

            # -----------------------------------------
            # Average repeated timings
            # -----------------------------------------

            avg_stage1 = (
                sum(stage1_times)
                / len(stage1_times)
            )

            avg_encode = (
                sum(encode_times)
                / len(encode_times)
            )

            avg_transfer = (
                sum(transfer_times)
                / len(transfer_times)
            )

            avg_reconstruction = (
                sum(reconstruction_times)
                / len(reconstruction_times)
            )

            avg_stage2 = (
                sum(stage2_times)
                / len(stage2_times)
            )

            avg_component_sum = (
                sum(component_sum_times)
                / len(component_sum_times)
            )

            avg_e2e = (
                sum(e2e_times)
                / len(e2e_times)
            )

            # -----------------------------------------
            # Boundary reconstruction exactness
            #
            # Do this OUTSIDE timed inference.
            # -----------------------------------------

            torch.cuda.synchronize(
                EDGE_DEVICE
            )

            torch.cuda.synchronize(
                CLOUD_DEVICE
            )

            boundary_exact = torch.equal(
                last_output[
                    "boundary_edge"
                ].cpu(),

                last_output[
                    "boundary_cloud"
                ].cpu(),
            )

            # -----------------------------------------
            # Output correctness
            # -----------------------------------------

            delta_argmax = (
                last_output["logits"]
                .argmax(dim=-1)
                .cpu()
            )

            same_argmax = torch.equal(
                reference_argmax[state_idx],
                delta_argmax,
            )

            # -----------------------------------------
            # Communication statistics
            # -----------------------------------------

            full_boundary_bytes = (
                last_output[
                    "full_boundary_bytes"
                ]
            )

            delta_bytes = (
                last_output[
                    "delta_bytes"
                ]
            )

            value_bytes = (
                last_output[
                    "value_bytes"
                ]
            )

            index_bytes = (
                last_output[
                    "index_bytes"
                ]
            )

            # Per-state wire reduction.
            #
            # Can be slightly negative during initial
            # synchronization if row indices are also
            # transmitted.
            wire_reduction = (
                1.0
                - (
                    delta_bytes
                    / full_boundary_bytes
                )
            )

            orchestration_overhead_ms = (
                avg_e2e
                - avg_component_sum
            )

            # -----------------------------------------
            # Save current state result
            # -----------------------------------------

            records.append({
                "state":
                    state_idx,

                "split_layer":
                    split_layer,

                "initial_sync":
                    last_output[
                        "initial_sync"
                    ],

                "changed_rows":
                    last_output[
                        "changed_rows"
                    ],

                "total_rows":
                    last_output[
                        "total_rows"
                    ],

                "changed_row_ratio":
                    last_output[
                        "changed_row_ratio"
                    ],

                "full_boundary_bytes":
                    full_boundary_bytes,

                "delta_bytes":
                    delta_bytes,

                "value_bytes":
                    value_bytes,

                "index_bytes":
                    index_bytes,

                "wire_reduction":
                    wire_reduction,

                "boundary_exact":
                    boundary_exact,

                "same_argmax":
                    same_argmax,

                "stage1_ms":
                    avg_stage1,

                "encode_ms":
                    avg_encode,

                "transfer_ms":
                    avg_transfer,

                "reconstruction_ms":
                    avg_reconstruction,

                "stage2_ms":
                    avg_stage2,

                "component_sum_ms":
                    avg_component_sum,

                "measured_e2e_ms":
                    avg_e2e,

                "orchestration_overhead_ms":
                    orchestration_overhead_ms,
            })

            # =========================================
            # ADVANCE SYNCHRONIZED STATE
            #
            # IMPORTANT:
            # This happens ONCE, only after every
            # timing repeat for this transition.
            # =========================================

            prev_boundary_edge = (
                last_output[
                    "boundary_edge"
                ]
            )

            prev_boundary_cloud = (
                last_output[
                    "boundary_cloud"
                ]
            )

        print(
            f"Finished split "
            f"{split_layer}"
        )

        # ---------------------------------------------
        # Release temporary split-specific objects
        # ---------------------------------------------

        del cloud_rotary_emb
        del prev_boundary_edge
        del prev_boundary_cloud

        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # 5. Save detailed per-state results
    # -------------------------------------------------

    df = pd.DataFrame(
        records
    )

    os.makedirs(
        "outputs/exp4b",
        exist_ok=True,
    )

    df.to_csv(
        "outputs/exp4b/"
        "synchronized_delta_per_state.csv",
        index=False,
    )

    # -------------------------------------------------
    # 6. Aggregate by split layer
    # -------------------------------------------------

    summary = (
        df
        .groupby("split_layer")
        .agg(
            states=(
                "state",
                "count",
            ),

            avg_changed_row_ratio=(
                "changed_row_ratio",
                "mean",
            ),

            total_full_bytes=(
                "full_boundary_bytes",
                "sum",
            ),

            total_delta_bytes=(
                "delta_bytes",
                "sum",
            ),

            total_value_bytes=(
                "value_bytes",
                "sum",
            ),

            total_index_bytes=(
                "index_bytes",
                "sum",
            ),

            stage1_ms=(
                "stage1_ms",
                "mean",
            ),

            encode_ms=(
                "encode_ms",
                "mean",
            ),

            transfer_ms=(
                "transfer_ms",
                "mean",
            ),

            reconstruction_ms=(
                "reconstruction_ms",
                "mean",
            ),

            stage2_ms=(
                "stage2_ms",
                "mean",
            ),

            component_sum_ms=(
                "component_sum_ms",
                "mean",
            ),

            measured_e2e_ms=(
                "measured_e2e_ms",
                "mean",
            ),

            orchestration_overhead_ms=(
                "orchestration_overhead_ms",
                "mean",
            ),

            all_boundary_exact=(
                "boundary_exact",
                "all",
            ),

            all_argmax_same=(
                "same_argmax",
                "all",
            ),
        )
        .reset_index()
    )

    # -------------------------------------------------
    # 7. Derived communication metrics
    # -------------------------------------------------

    summary["full_boundary_mb"] = (
        summary["total_full_bytes"]
        / 1024**2
    )

    summary["delta_mb"] = (
        summary["total_delta_bytes"]
        / 1024**2
    )

    summary["wire_reduction"] = (
        1.0
        - (
            summary["total_delta_bytes"]
            / summary["total_full_bytes"]
        )
    )

    summary["wire_reduction_pct"] = (
        summary["wire_reduction"]
        * 100.0
    )

    # -------------------------------------------------
    # 8. Steady-state communication reduction
    #
    # Excludes state 0 because state 0 is the initial
    # synchronization rather than a delta transition.
    # -------------------------------------------------

    steady_df = df[
        ~df["initial_sync"]
    ]

    steady_summary = (
        steady_df
        .groupby("split_layer")
        .agg(
            steady_full_bytes=(
                "full_boundary_bytes",
                "sum",
            ),

            steady_delta_bytes=(
                "delta_bytes",
                "sum",
            ),

            steady_changed_row_ratio=(
                "changed_row_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    steady_summary[
        "steady_wire_reduction_pct"
    ] = (
        1.0
        - (
            steady_summary[
                "steady_delta_bytes"
            ]
            / steady_summary[
                "steady_full_bytes"
            ]
        )
    ) * 100.0

    summary = summary.merge(
        steady_summary[
            [
                "split_layer",
                "steady_changed_row_ratio",
                "steady_wire_reduction_pct",
            ]
        ],
        on="split_layer",
        how="left",
    )

    # -------------------------------------------------
    # 9. Save summary
    # -------------------------------------------------

    summary.to_csv(
        "outputs/exp4b/"
        "synchronized_delta_summary.csv",
        index=False,
    )

    # -------------------------------------------------
    # 10. Print compact summary
    # -------------------------------------------------

    print(
        "\n=== EXPERIMENT 4B SUMMARY ==="
    )

    print(
        summary[
            [
                "split_layer",
                "avg_changed_row_ratio",
                "steady_changed_row_ratio",
                "full_boundary_mb",
                "delta_mb",
                "wire_reduction_pct",
                "steady_wire_reduction_pct",
                "encode_ms",
                "transfer_ms",
                "reconstruction_ms",
                "measured_e2e_ms",
                "all_boundary_exact",
                "all_argmax_same",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nDetailed results:"
    )

    print(
        "outputs/exp4b/"
        "synchronized_delta_per_state.csv"
    )

    print(
        "\nSummary:"
    )

    print(
        "outputs/exp4b/"
        "synchronized_delta_summary.csv"
    )


if __name__ == "__main__":
    main()