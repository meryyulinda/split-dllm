import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

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
    SELECTED_LAYERS,
    DRIFT_EPSILON,
)

from src.model_loader import (load_dream_model,)

from src.generation import (prepare_prompt,)

from src.hidden_state_tracker import (HiddenDriftTracker)

from src.utils import (set_seed,)

def main():

    print("=" * 60)
    print(
        "EXPERIMENT 2: "
        "CROSS-STEP HIDDEN-STATE DRIFT"
    )
    print("=" * 60)

    set_seed(SEED)

    # --------------------------------
    # 1. Load model
    # --------------------------------

    model, tokenizer = (load_dream_model(MODEL_NAME))

    # --------------------------------
    # 2. Prepare prompt
    # --------------------------------

    inputs = prepare_prompt(
        tokenizer,
        PROMPT,
    )

    prompt_length = (inputs["input_ids"].shape[1])

    print(f"Prompt tokens: {prompt_length}")

    print(
        f"Selected layers: "
        f"{SELECTED_LAYERS}"
    )

    # --------------------------------
    # 3. Create tracker
    # --------------------------------

    tracker = HiddenDriftTracker(
        selected_layers=SELECTED_LAYERS,
        prompt_length=prompt_length,
        eps=DRIFT_EPSILON,
    )

    tracker.register(model)

    # --------------------------------
    # 4. Run ORIGINAL Dream generation
    # --------------------------------

    try:

        output = model.diffusion_generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=MAX_NEW_TOKENS,
            steps=STEPS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            alg=ALGORITHM,
            return_dict_in_generate=True,
        )

    finally:

        # Very important:
        # always remove hooks afterward.
        tracker.remove()

    # --------------------------------
    # 5. Decode result
    # --------------------------------

    sequence = output.sequences[0]

    generated_text = tokenizer.decode(
        sequence,
        skip_special_tokens=True,
    )

    # --------------------------------
    # 6. Create DataFrame
    # --------------------------------

    df = pd.DataFrame(tracker.records)

    os.makedirs(
        "outputs/exp2",
        exist_ok=True,
    )

    df.to_csv(
        "outputs/exp2/hidden_drift.csv",
        index=False,
    )

    # --------------------------------------
    # 7. Create Heatmap Plot of Hidden Drift
    # --------------------------------------

    heatmap_data = df.pivot(
        index="layer",
        columns="transition",
        values="mean_drift",
    )


    plt.figure(
        figsize=(12, 5)
    )

    plt.imshow(
        heatmap_data.values,
        aspect="auto",
        origin="lower",
    )

    plt.colorbar(
        label="Mean normalized hidden-state drift"
    )

    plt.yticks(
        range(
            len(heatmap_data.index)
        ),
        heatmap_data.index,
    )

    plt.xlabel(
        "Denoising transition"
    )

    plt.ylabel(
        "Transformer layer"
    )

    plt.title(
        "Dream Hidden-State Drift "
        "Across Denoising Steps"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/exp2/"
        "hidden_drift_heatmap.png",
        dpi=200,
    )

    plt.show()

    # --------------------------------
    # 8. Compact terminal output
    # --------------------------------

    print("\nGenerated output:")
    print(generated_text)

    print("\n=== EXPERIMENT 2 SUMMARY ===")

    print(
        f"Recorded transitions: "
        f"{df['transition'].nunique()}"
    )

    print(
        f"Layers analyzed: "
        f"{len(SELECTED_LAYERS)}"
    )

    summary = (
        df
        .groupby("layer")
        .agg(
            # EXPERIMENT 2A: HIDDEN-STATE DRIFT MEASUREMENT + HEATMAP across Layers & Transitions
            mean_drift=(
                "mean_drift",
                "mean",
            ),
            median_drift=(
                "median_drift",
                "median",
            ),
            max_drift=(
                "max_drift",
                "max",
            ),
            # EXPERIMENT 2B: ACTIVE-ROW RATIO BY THRESHOLD
            active_1e_4=(
                "active_ratio_1e-4",
                "mean"
            ),
            active_5e_4=(
                "active_ratio_5e-4",
                "mean"
            ),
            active_1e_3=(
                "active_ratio_1e-3",
                "mean"
            ),
            active_5e_3=(
                "active_ratio_5e-3",
                "mean"
            ),
            active_1e_2=(
                "active_ratio_1e-2",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    # Check active ratio across transitions in a layer (sample: layer 12)
    print("\n=== ACTVE RATIO DIAGNOSTIC ===")

    diagnostic = df[df["layer"]==12][
        [
        "transition",
        "mean_drift",
        "median_drift",
        "active_ratio_1e-3",
        "active_ratio_1e-2",
        ]]

    print(diagnostic.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))

    print(diagnostic["active_ratio_1e-3"]
    .value_counts()
    .sort_index()
    )


    print(
        "\nFull per-step data saved to:"
    )

    print(
        "outputs/exp2/"
        "hidden_drift.csv"
    )


if __name__ == "__main__":
    main()
