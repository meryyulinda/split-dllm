import os
import sys
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
)

from src.model_loader import load_dream_model

from src.generation import (
    generate_with_token_history,
)

from src.instrumentation import (
    create_token_history_hook,
    count_token_changes,
    count_masks,
    changed_indices,
)

from src.utils import set_seed


def main():

    set_seed(SEED)

    model, tokenizer = load_dream_model(
        MODEL_NAME
    )

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

    # --------------------------------
    # Prompt / generated-region boundary
    # --------------------------------

    prompt_length = (
        result["inputs"]["input_ids"]
        .shape[1]
    )

    print(
        f"Prompt length: "
        f"{prompt_length}"
    )

    print(
        f"Generated region length: "
        f"{MAX_NEW_TOKENS}"
    )

    mask_token_id = (
        model.config.mask_token_id
    )

    rows = []

    # --------------------------------
    # Compare consecutive diffusion states
    # --------------------------------

    for transition in range(
        1,
        len(history)
    ):

        previous = history[
            transition - 1
        ]

        current = history[
            transition
        ]

        # Only analyze generated tokens,
        # excluding the fixed prompt.
        previous_generated = (
            previous[prompt_length:]
        )

        current_generated = (
            current[prompt_length:]
        )

        (
            changed_mask,
            changed_count,
            changed_ratio,
        ) = count_token_changes(
            previous_generated,
            current_generated,
        )

        masks_remaining = count_masks(
            current_generated,
            mask_token_id,
        )

        indices = changed_indices(
            changed_mask
        )

        rows.append({
            "transition": transition,
            "changed_tokens": changed_count,
            "changed_ratio": changed_ratio,
            "masks_remaining": masks_remaining,
            "changed_indices": str(indices),
        })

    df = pd.DataFrame(rows)

    # --------------------------------
    # Output
    # --------------------------------

    print("\nGenerated output:")
    print(result["text"])

    print(
        "\n=== EXPERIMENT 1 SUMMARY ==="
    )

    print(
        f"Recorded transitions: "
        f"{len(df)}"
    )

    print(
        f"Generated token rows: "
        f"{MAX_NEW_TOKENS}"
    )

    print(
        "\nLast transition:"
    )

    print(
        df[
            [
                "transition",
                "changed_tokens",
                "changed_ratio",
                "masks_remaining",
            ]
        ]
        .tail(1)
        .to_string(
            index=False
        )
    )

    # --------------------------------
    # Save CSV
    # --------------------------------

    os.makedirs(
        "outputs/exp1",
        exist_ok=True,
    )

    df.to_csv(
        "outputs/exp1/token_changes.csv",
        index=False,
    )

    print(
        "\nFull per-transition data saved to:"
    )

    print(
        "outputs/exp1/token_changes.csv"
    )


if __name__ == "__main__":
    main()