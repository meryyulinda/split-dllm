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
    cuda_time,
    boundary_size_mb,
)


def main():

    print("=" * 60)
    print(
        "EXPERIMENT 3C: "
        "STATIC SPLIT LAYER SWEEP"
    )
    print("=" * 60)

    # -------------------------------------------------
    # 1. Load model
    # -------------------------------------------------

    set_seed(SEED)

    model, tokenizer = load_dream_model(MODEL_NAME)

    model.eval()

    # -------------------------------------------------
    # 2. Generate SAME Dream diffusion trajectory
    #    as Exp3B
    # -------------------------------------------------

    hook, history = (
        create_token_history_hook()
    )

    set_seed(SEED)

    generation_result = (
        generate_with_token_history(
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
    )

    print(
        f"\nCaptured states: "
        f"{len(history)}"
    )

    print(
        f"Split layers: "
        f"{SPLIT_LAYERS}"
    )

    # -------------------------------------------------
    # 3. Sweep split layers
    # -------------------------------------------------

    records = []

    for split_layer in SPLIT_LAYERS:

        print(
            "\n" + "=" * 60
        )

        print(
            f"SPLIT LAYER "
            f"{split_layer}"
        )

        print(
            "=" * 60
        )

        state_records = []

        # ---------------------------------------------
        # Replay all diffusion states
        # ---------------------------------------------

        for state_idx, state in enumerate(
            history
        ):

            input_ids = (
                state
                .unsqueeze(0)
                .to("cuda")
            )

            attention_mask = (
                torch.ones_like(
                    input_ids,
                    dtype=torch.bool,
                    device="cuda",
                )
            )

            # -----------------------------------------
            # Warmup
            #
            # only needed once per split layer
            # -----------------------------------------

            if state_idx == 0:

                for _ in range(WARMUP):

                    with torch.no_grad():

                        _ = model(
                            input_ids=
                                input_ids,
                            attention_mask=
                                attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )

                        _ = manual_split_forward(
                            model=model,
                            input_ids=
                                input_ids,
                            attention_mask=
                                attention_mask,
                            split_layer=
                                split_layer,
                        )

                torch.cuda.synchronize()

            # -----------------------------------------
            # Full model latency
            # -----------------------------------------

            full_times = []

            for _ in range(REPEATS):

                with torch.no_grad():

                    full_output, elapsed = (
                        cuda_time(
                            lambda: model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                return_dict=True,
                            )
                        )
                    )

                full_times.append(
                    elapsed
                )

            avg_full_latency = (
                sum(full_times)
                / len(full_times)
            )

            full_logits = (
                full_output.logits
            )

            # -----------------------------------------
            # Manual split latency
            # -----------------------------------------

            split_times = []

            for _ in range(REPEATS):

                with torch.no_grad():

                    split_output, elapsed = (
                        cuda_time(
                            lambda:
                                manual_split_forward(
                                    model=model,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    split_layer=split_layer,
                                )
                        )
                    )

                split_times.append(
                    elapsed
                )

            avg_split_latency = (
                sum(split_times)
                / len(split_times)
            )

            split_logits = (
                split_output["logits"]
            )

            boundary = (
                split_output["boundary"]
            )

            # -----------------------------------------
            # Correctness
            # -----------------------------------------

            diff = (full_logits - split_logits).abs()

            max_abs_diff = (
                diff.max().item()
            )

            mean_abs_diff = (
                diff.mean().item()
            )

            allclose = (
                torch.allclose(
                    full_logits,
                    split_logits,
                    atol=1e-5,
                    rtol=1e-5,
                )
            )

            same_argmax = (
                torch.equal(
                    full_logits.argmax(dim=-1),
                    split_logits.argmax(dim=-1),
                )
            )

            # -----------------------------------------
            # Boundary size
            # -----------------------------------------

            boundary_mb = (
                boundary_size_mb(boundary)
            )

            state_records.append({
                "state":
                    state_idx,

                "split_layer":
                    split_layer,

                "sequence_length":
                    input_ids.shape[1],

                "boundary_rows":
                    boundary.shape[1],

                "boundary_dim":
                    boundary.shape[2],

                "boundary_mb":
                    boundary_mb,

                "full_latency_ms":
                    avg_full_latency,

                "split_latency_ms":
                    avg_split_latency,

                "max_abs_diff":
                    max_abs_diff,

                "mean_abs_diff":
                    mean_abs_diff,

                "allclose":
                    allclose,

                "same_argmax":
                    same_argmax,
            })

        # -------------------------------------------------
        # 4. Aggregate current split layer
        # -------------------------------------------------

        split_df = pd.DataFrame(
            state_records
        )

        all_states_allclose = (
            split_df["allclose"].all()
        )

        all_argmax_same = (
            split_df["same_argmax"].all()
        )

        avg_boundary_mb = (
            split_df["boundary_mb"].mean()
        )

        total_boundary_mb = (
            split_df["boundary_mb"].sum()
        )

        avg_full_latency = (
            split_df["full_latency_ms"].mean()
        )

        avg_split_latency = (
            split_df["split_latency_ms"].mean()
        )

        max_diff_overall = (
            split_df["max_abs_diff"].max()
        )

        mean_diff_overall = (
            split_df["mean_abs_diff"].mean()
        )

        records.append({
            "split_layer":
                split_layer,

            "states":
                len(split_df),

            "allclose":
                all_states_allclose,

            "same_argmax":
                all_argmax_same,

            "max_abs_diff":
                max_diff_overall,

            "mean_abs_diff":
                mean_diff_overall,

            "avg_full_latency_ms":
                avg_full_latency,

            "avg_split_latency_ms":
                avg_split_latency,

            "avg_boundary_mb":
                avg_boundary_mb,

            "total_boundary_mb":
                total_boundary_mb,
        })

        print(
            f"All states allclose   : "
            f"{all_states_allclose}"
        )

        print(
            f"All argmax identical  : "
            f"{all_argmax_same}"
        )

        print(
            f"Avg full latency      : "
            f"{avg_full_latency:.3f} ms"
        )

        print(
            f"Avg split latency     : "
            f"{avg_split_latency:.3f} ms"
        )

        print(
            f"Avg boundary/state    : "
            f"{avg_boundary_mb:.4f} MB"
        )

        print(
            f"Total boundary traffic: "
            f"{total_boundary_mb:.4f} MB"
        )

    # -------------------------------------------------
    # 5. Final summary
    # -------------------------------------------------

    df = pd.DataFrame(records)

    os.makedirs(
        "outputs/exp3c",
        exist_ok=True,
    )

    df.to_csv(
        "outputs/exp3c/"
        "split_sweep.csv",
        index=False,
    )

    print(
        "\n=== EXPERIMENT 3C SUMMARY ==="
    )

    print(
        df[
            [
                "split_layer",
                "allclose",
                "same_argmax",
                "avg_full_latency_ms",
                "avg_split_latency_ms",
                "avg_boundary_mb",
                "total_boundary_mb",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nFull results saved to:"
    )

    print(
        "outputs/exp3c/"
        "split_sweep.csv"
    )


if __name__ == "__main__":
    main()