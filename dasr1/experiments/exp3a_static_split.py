import os
import sys
import torch


sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


from configs.experiment_config import (
    MODEL_NAME,
    PROMPT,
    SEED,
    SPLIT_LAYER,
)

from src.model_loader import (
    load_dream_model,
)

from src.generation import (
    prepare_prompt,
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
        "EXPERIMENT 3A: STATIC SPLIT CORRECTNESS"
    )
    print("=" * 60)

    set_seed(SEED)

    model, tokenizer = (
        load_dream_model(MODEL_NAME)
    )

    inputs = prepare_prompt(
        tokenizer,
        PROMPT,
    )

    input_ids = inputs["input_ids"]

    attention_mask = inputs.get("attention_mask").bool()

    # --------------------------------
    # 1. ORIGINAL FULL MODEL
    # --------------------------------

    with torch.no_grad():

        full_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    full_logits = (full_output.logits)

    # --------------------------------
    # 2. MANUAL STATIC SPLIT
    # --------------------------------

    split_output = (
        manual_split_forward(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            split_layer=SPLIT_LAYER,
        )
    )

    split_logits = (split_output["logits"])

    boundary = (split_output["boundary"])

    # --------------------------------
    # 3. Compare results
    # --------------------------------

    absolute_diff = (full_logits - split_logits).abs()

    max_abs_diff = (
        absolute_diff
        .max()
        .item()
    )

    mean_abs_diff = (
        absolute_diff
        .mean()
        .item()
    )

    same_argmax = torch.equal(
        full_logits.argmax(dim=-1),
        split_logits.argmax(dim=-1),
    )

    allclose = torch.allclose(
        full_logits,
        split_logits,
        atol=1e-5,
        rtol=1e-5,
    )

    # --------------------------------
    # 4. Boundary information
    # --------------------------------

    boundary_bytes = (
        boundary.numel()
        * boundary.element_size()
    )

    boundary_mb = (
        boundary_bytes
        / 1024**2
    )

    # --------------------------------
    # 5. Print
    # --------------------------------

    print(
        f"\nSplit layer: "
        f"{SPLIT_LAYER}"
    )

    print(
        f"Total layers: "
        f"{len(model.model.layers)}"
    )

    print(
        f"\nBoundary shape: "
        f"{boundary.shape}"
    )

    print(
        f"Boundary dtype: "
        f"{boundary.dtype}"
    )

    print(
        f"Boundary size: "
        f"{boundary_mb:.4f} MB"
    )

    print(
        "\n=== CORRECTNESS ==="
    )

    print(
        f"Max absolute difference: "
        f"{max_abs_diff:.10f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_abs_diff:.10f}"
    )

    print(
        f"torch.allclose: "
        f"{allclose}"
    )

    print(
        f"Same argmax tokens: "
        f"{same_argmax}"
    )


if __name__ == "__main__":
    main()