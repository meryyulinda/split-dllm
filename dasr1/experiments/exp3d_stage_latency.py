import os
import sys
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
    manual_split_forward,
)

from src.utils import (
    set_seed,
    boundary_size_mb,
)


def main():

    print("=" * 60)
    print(
        "EXPERIMENT 3D: "
        "STAGE-WISE LATENCY PROFILING"
    )
    print("=" * 60)

    # ---------------------------------------------
    # 1. Load model
    # ---------------------------------------------

    set_seed(SEED)

    model, tokenizer = load_dream_model(
        MODEL_NAME
    )

    model.eval()

    # ---------------------------------------------
    # 2. Capture diffusion trajectory
    # ---------------------------------------------

    hook, history = create_token_history_hook()

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
        f"\nCaptured states: {len(history)}"
    )

    records = []

    # ---------------------------------------------
    # 3. Sweep split layers
    # ---------------------------------------------

    for split_layer in SPLIT_LAYERS:

        print("\n" + "=" * 60)
        print(
            f"SPLIT LAYER {split_layer}"
        )
        print("=" * 60)

        # -----------------------------------------
        # Warmup using first state
        # -----------------------------------------

        state = history[0]

        input_ids = (
            state
            .unsqueeze(0)
            .to("cuda")
        )

        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.bool,
            device="cuda",
        )

        for _ in range(WARMUP):

            with torch.no_grad():

                _ = manual_split_forward(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    split_layer=split_layer,
                    profile=False,
                )

        torch.cuda.synchronize()

        # -----------------------------------------
        # Replay trajectory
        # -----------------------------------------

        for state_idx, state in enumerate(history):

            input_ids = (
                state
                .unsqueeze(0)
                .to("cuda")
            )

            attention_mask = torch.ones_like(
                input_ids,
                dtype=torch.bool,
                device="cuda",
            )

            stage1_times = []
            stage2_times = []

            last_output = None

            # -------------------------------------
            # Repeat measurement
            # -------------------------------------

            for _ in range(REPEATS):

                with torch.no_grad():

                    output = manual_split_forward(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        split_layer=split_layer,
                        profile=True,
                    )

                stage1_times.append(output["stage1_ms"])

                stage2_times.append(output["stage2_ms"])

                last_output = output

            # -------------------------------------
            # Average
            # -------------------------------------

            avg_stage1 = (
                sum(stage1_times)
                / len(stage1_times)
            )

            avg_stage2 = (
                sum(stage2_times)
                / len(stage2_times)
            )

            total_compute = (
                avg_stage1
                + avg_stage2
            )

            boundary = (last_output["boundary"])

            boundary_mb = (boundary_size_mb(boundary))

            records.append({
                "state":
                    state_idx,

                "split_layer":
                    split_layer,

                "sequence_length":
                    input_ids.shape[1],

                "stage1_ms":
                    avg_stage1,

                "stage2_ms":
                    avg_stage2,

                "total_compute_ms":
                    total_compute,

                "stage1_fraction":
                    avg_stage1
                    / total_compute,

                "stage2_fraction":
                    avg_stage2
                    / total_compute,

                "boundary_mb":
                    boundary_mb,
            })

    # ---------------------------------------------
    # 4. Save detailed results
    # ---------------------------------------------

    df = pd.DataFrame(records)

    os.makedirs(
        "outputs/exp3d",
        exist_ok=True,
    )

    df.to_csv(
        "outputs/exp3d/"
        "stage_latency_per_state.csv",
        index=False,
    )

    # ---------------------------------------------
    # 5. Aggregate by split layer
    # ---------------------------------------------

    summary = (
        df
        .groupby("split_layer")
        .agg(
            stage1_ms=(
                "stage1_ms",
                "mean",
            ),
            stage2_ms=(
                "stage2_ms",
                "mean",
            ),
            total_compute_ms=(
                "total_compute_ms",
                "mean",
            ),
            stage1_fraction=(
                "stage1_fraction",
                "mean",
            ),
            stage2_fraction=(
                "stage2_fraction",
                "mean",
            ),
            boundary_mb=(
                "boundary_mb",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        "outputs/exp3d/"
        "stage_latency_summary.csv",
        index=False,
    )

    print(
        "\n=== EXPERIMENT 3D SUMMARY ==="
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
        "outputs/exp3d/"
        "stage_latency_summary.csv"
    )


if __name__ == "__main__":
    main()