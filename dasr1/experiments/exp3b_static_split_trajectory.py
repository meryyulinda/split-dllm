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
    SPLIT_LAYER,
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
)


def main():

    print("=" * 60)
    print(
        "EXPERIMENT 3B: "
        "STATIC SPLIT ACROSS DIFFUSION TRAJECTORY"
    )
    print("=" * 60)

    # -------------------------------------------------
    # 1. Load model
    # -------------------------------------------------

    set_seed(SEED)

    model, tokenizer = load_dream_model(MODEL_NAME)

    # -------------------------------------------------
    # 2. Generate official Dream trajectory
    # -------------------------------------------------

    hook, history = create_token_history_hook()

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

    # -------------------------------------------------
    # 3. Replay each state through:
    #
    #    A. normal full model
    #    B. manual static split
    # -------------------------------------------------

    records = []

    total_boundary_bytes = 0

    for state_idx, state in enumerate(history):

        # history is stored on CPU,
        # so move back to GPU
        input_ids = (
            state
            .unsqueeze(0)
            .to("cuda")
        )

        # All positions in the captured diffusion state
        # are valid sequence positions.
        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.bool,
            device="cuda",
        )

        # ---------------------------------------------
        # Full model
        # ---------------------------------------------

        with torch.no_grad():

            full_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        full_logits = full_output.logits

        # ---------------------------------------------
        # Static split
        # ---------------------------------------------

        split_output = manual_split_forward(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            split_layer=SPLIT_LAYER,
        )

        split_logits = split_output["logits"]

        boundary = split_output["boundary"]

        # ---------------------------------------------
        # Compare
        # ---------------------------------------------

        diff = (full_logits - split_logits).abs()

        max_abs_diff = (
            diff.max().item()
        )

        mean_abs_diff = (
            diff.mean().item()
        )

        allclose = torch.allclose(
            full_logits,
            split_logits,
            atol=1e-5,
            rtol=1e-5,
        )

        same_argmax = torch.equal(
            full_logits.argmax(dim=-1),
            split_logits.argmax(dim=-1),
        )

        # ---------------------------------------------
        # Boundary size
        # ---------------------------------------------

        boundary_bytes = (
            boundary.numel()
            * boundary.element_size()
        )

        total_boundary_bytes += (
            boundary_bytes
        )

        records.append({
            "state": state_idx,
            "sequence_length":
                input_ids.shape[1],
            "boundary_rows":
                boundary.shape[1],
            "boundary_dim":
                boundary.shape[2],
            "boundary_bytes":
                boundary_bytes,
            "boundary_mb":
                boundary_bytes / 1024**2,
            "max_abs_diff":
                max_abs_diff,
            "mean_abs_diff":
                mean_abs_diff,
            "allclose":
                allclose,
            "same_argmax":
                same_argmax,
        })

        print(
            f"State {state_idx:02d} | "
            f"allclose={allclose} | "
            f"max_diff={max_abs_diff:.8f}"
        )

    # -------------------------------------------------
    # 4. Create DataFrame
    # -------------------------------------------------

    df = pd.DataFrame(
        records
    )

    os.makedirs(
        "outputs/exp3b",
        exist_ok=True,
    )

    df.to_csv(
        "outputs/exp3b/"
        "static_split_trajectory.csv",
        index=False,
    )

    # -------------------------------------------------
    # 5. Summary
    # -------------------------------------------------

    all_states_allclose = (
        df["allclose"].all()
    )

    all_argmax_same = (
        df["same_argmax"].all()
    )

    total_boundary_mb = (
        total_boundary_bytes
        / 1024**2
    )

    average_boundary_mb = (
        df["boundary_mb"].mean()
    )

    max_diff_overall = (
        df["max_abs_diff"].max()
    )

    mean_diff_overall = (
        df["mean_abs_diff"].mean()
    )

    print(
        "\n=== EXPERIMENT 3B SUMMARY ==="
    )

    print(
        f"States checked        : "
        f"{len(df)}"
    )

    print(
        f"Split layer           : "
        f"{SPLIT_LAYER}"
    )

    print(
        f"All states allclose   : "
        f"{all_states_allclose}"
    )

    print(
        f"All argmax identical  : "
        f"{all_argmax_same}"
    )

    print(
        f"Max absolute diff     : "
        f"{max_diff_overall:.10f}"
    )

    print(
        f"Mean absolute diff    : "
        f"{mean_diff_overall:.10f}"
    )

    print(
        f"Avg boundary/state    : "
        f"{average_boundary_mb:.4f} MB"
    )

    print(
        f"Total boundary traffic: "
        f"{total_boundary_mb:.4f} MB"
    )

    print(
        "\nFull results saved to:"
    )

    print(
        "outputs/exp3b/"
        "static_split_trajectory.csv"
    )


if __name__ == "__main__":
    main()