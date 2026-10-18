import os
import pandas as pd


def main():

    # --------------------------------
    # 1. Load Experiment 1 results
    # --------------------------------

    token_df = pd.read_csv(
        "outputs/exp1/token_changes.csv"
    )

    # --------------------------------
    # 2. Load Experiment 2 results
    # --------------------------------

    hidden_df = pd.read_csv(
        "outputs/exp2/hidden_drift.csv"
    )

    # --------------------------------
    # 3. Rename step -> transition
    # --------------------------------

    token_df = token_df.rename(
        columns={
            "step": "transition"
        }
    )

    # --------------------------------
    # 4. Keep relevant columns
    # --------------------------------

    token_df = token_df[
        [
            "transition",
            "changed_tokens",
            "changed_ratio",
        ]
    ]

    hidden_df = hidden_df[
        [
            "transition",
            "layer",
            "mean_drift",
            "median_drift",
            "active_ratio_1e-3",
        ]
    ]

    # --------------------------------
    # 5. Merge by transition
    # --------------------------------

    merged = pd.merge(
        hidden_df,
        token_df,
        on="transition",
        how="inner",
    )

    # --------------------------------
    # 6. Reorder columns
    # --------------------------------

    merged = merged[
        [
            "transition",
            "layer",
            "changed_tokens",
            "changed_ratio",
            "mean_drift",
            "median_drift",
            "active_ratio_1e-3",
        ]
    ]

    # --------------------------------
    # 7. Save full result
    # --------------------------------

    os.makedirs(
        "outputs/exp2c",
        exist_ok=True,
    )

    merged.to_csv(
        "outputs/exp2c/"
        "token_vs_hidden.csv",
        index=False,
    )

    # --------------------------------
    # 8. Print compact summary
    # --------------------------------

    print(
        "\n=== EXPERIMENT 2C SUMMARY ==="
    )

    print(
        f"Transitions matched: "
        f"{merged['transition'].nunique()}"
    )

    print(
        f"Layers matched: "
        f"{merged['layer'].nunique()}"
    )

    # --------------------------------
    # 9. Focus on one layer first
    # --------------------------------

    target_layer = 12

    layer_df = merged[
        merged["layer"] == target_layer
    ].copy()

    print(
        f"\nLayer {target_layer}"
    )

    print(
        layer_df[
            [
                "transition",
                "changed_tokens",
                "changed_ratio",
                "active_ratio_1e-3",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    # Correlation between changed-token state and hidden drift
    correlation = (
        layer_df[
            [
                "changed_ratio",
                "active_ratio_1e-3",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    print(
        f"\nCorrelation "
        f"(changed token ratio vs active hidden ratio): "
        f"{correlation:.6f}"
    )

    # Mean of hidden active ratio when token changes occur
    changed_cases = layer_df[
        layer_df["changed_tokens"] > 0
    ]

    unchanged_cases = layer_df[
        layer_df["changed_tokens"] == 0
    ]

    print(
        "\nWhen token changes occur:"
    )

    print(
        f"Transitions: "
        f"{len(changed_cases)}"
    )

    print(
        f"Mean active hidden ratio: "
        f"{changed_cases['active_ratio_1e-3'].mean():.6f}"
    )

    print(
        "\nWhen no token changes occur:"
    )

    print(
        f"Transitions: "
        f"{len(unchanged_cases)}"
    )

    print(
        f"Mean active hidden ratio: "
        f"{unchanged_cases['active_ratio_1e-3'].mean():.6f}"
    )


if __name__ == "__main__":
    main()