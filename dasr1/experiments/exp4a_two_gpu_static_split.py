import os
import sys
import torch
import pandas as pd
import gc

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
)

from src.utils import (
    set_seed,
)


def main():

    print("=" * 60)
    print(
        "EXPERIMENT 4A: "
        "PHYSICAL TWO-GPU STATIC SPLIT"
    )
    print("=" * 60)

    # -------------------------------------------------
    # 0. Verify two GPUs
    # -------------------------------------------------

    gpu_count = torch.cuda.device_count()

    print(
        f"Available GPUs: {gpu_count}"
    )

    if gpu_count < 2:
        raise RuntimeError(
            "Experiment 4A requires "
            "at least two GPUs."
        )

    for gpu_idx in range(gpu_count):

        print(
            f"GPU {gpu_idx}: "
            f"{torch.cuda.get_device_name(gpu_idx)}"
        )

    # -------------------------------------------------
    # 1. Load full Dream model initially on GPU 0
    #
    # We first need the intact model to:
    #   A. generate the diffusion trajectory
    #   B. capture full-model reference predictions
    #
    # Physical partitioning happens afterward.
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
    # Store argmax only to keep CPU memory small.
    # -------------------------------------------------

    print(
        "\nCapturing full-model "
        "reference predictions..."
    )

    reference_argmax = []

    reference_device = next(model.parameters()).device
    
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
            f"PHYSICAL SPLIT LAYER "
            f"{split_layer}"
        )

        print(
            "=" * 60
        )

        # ---------------------------------------------
        # Move model modules to their physical GPUs.
        #
        # This placement cost is NOT included in
        # inference timing.
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
        # Replay all diffusion states
        # ---------------------------------------------

        for state_idx, state in enumerate(
            history
        ):

            input_ids = (
                state
                .unsqueeze(0)
                .to(EDGE_DEVICE)
            )

            stage1_times = []
            transfer_times = []
            stage2_times = []

            component_sum_times = []
            e2e_times = []

            last_output = None

            # -----------------------------------------
            # Repeated timing
            # -----------------------------------------

            for _ in range(REPEATS):

                output = (
                    two_gpu_split_forward(
                        model=model,
                        input_ids=input_ids,
                        split_layer=
                            split_layer,
                        cloud_rotary_emb=
                            cloud_rotary_emb,
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

                transfer_times.append(
                    output["transfer_ms"]
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
            # Average repeated measurements
            # -----------------------------------------

            avg_stage1 = (
                sum(stage1_times)
                / len(stage1_times)
            )

            avg_transfer = (
                sum(transfer_times)
                / len(transfer_times)
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
            # Correctness sanity check
            # -----------------------------------------

            two_gpu_argmax = (
                last_output["logits"]
                .argmax(dim=-1)
                .cpu()
            )

            same_argmax = torch.equal(
                reference_argmax[state_idx],
                two_gpu_argmax,
            )

            # -----------------------------------------
            # Boundary communication volume
            # -----------------------------------------

            boundary_bytes = (
                last_output[
                    "boundary_bytes"
                ]
            )

            boundary_mb = (
                boundary_bytes
                / 1024**2
            )

            # -----------------------------------------
            # Runtime overhead not explained by
            # Stage1 + transfer + Stage2
            # -----------------------------------------

            orchestration_overhead_ms = (
                avg_e2e
                - avg_component_sum
            )

            records.append({
                "state":
                    state_idx,

                "split_layer":
                    split_layer,

                "stage1_ms":
                    avg_stage1,

                "transfer_ms":
                    avg_transfer,

                "stage2_ms":
                    avg_stage2,

                "component_sum_ms":
                    avg_component_sum,

                "measured_e2e_ms":
                    avg_e2e,

                "orchestration_overhead_ms":
                    orchestration_overhead_ms,

                "boundary_bytes":
                    boundary_bytes,

                "boundary_mb":
                    boundary_mb,

                "same_argmax":
                    same_argmax,
            })

        print(
            f"Finished split "
            f"{split_layer}"
        )

        del cloud_rotary_emb
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # 5. Save per-state results
    # -------------------------------------------------

    df = pd.DataFrame(
        records
    )

    os.makedirs(
        "outputs/exp4a",
        exist_ok=True,
    )

    df.to_csv(
        "outputs/exp4a/"
        "two_gpu_static_per_state.csv",
        index=False,
    )

    # -------------------------------------------------
    # 6. Aggregate by split layer
    # -------------------------------------------------

    summary = (
        df
        .groupby("split_layer")
        .agg(
            stage1_ms=(
                "stage1_ms",
                "mean",
            ),

            transfer_ms=(
                "transfer_ms",
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

            boundary_mb=(
                "boundary_mb",
                "mean",
            ),

            all_argmax_same=(
                "same_argmax",
                "all",
            ),
        )
        .reset_index()
    )

    # Full boundary transmitted once per state.
    summary["total_boundary_mb"] = (
        summary["boundary_mb"]
        * len(history)
    )

    # -------------------------------------------------
    # 7. Save summary
    # -------------------------------------------------

    summary.to_csv(
        "outputs/exp4a/"
        "two_gpu_static_summary.csv",
        index=False,
    )

    # -------------------------------------------------
    # 8. Print summary
    # -------------------------------------------------

    print(
        "\n=== EXPERIMENT 4A SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        "outputs/exp4a/"
        "two_gpu_static_summary.csv"
    )


if __name__ == "__main__":
    main()