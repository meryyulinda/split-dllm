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
    SCHEDULER_THRESHOLD,
    SCHEDULER_SWITCH_OVERHEADS_MS,
)


def compute_optimal_schedule_dp(
    df,
    switch_overhead_ms,
):
    """
    Dynamic-programming scheduler.

    Objective:
        minimize total trajectory cost

    Cost for transition t at split s:
        execution_cost[t, s]
        + switching penalty if previous split != s

    execution_cost uses:
        ideal_e2e_no_overhead_ms

    Therefore, this remains an analytical scheduler
    based on Exp5B's idealized selective-compute estimate.
    """

    # =====================================================
    # 1. PREPARE TRANSITIONS AND SPLIT OPTIONS
    # =====================================================

    transitions = sorted(
        df["transition"].unique()
    )

    split_layers = sorted(
        df["split_layer"].unique()
    )

    # ---------------------------------------------
    # Cost lookup:
    # execution_cost[(transition, split)]
    # ---------------------------------------------

    execution_cost = {}

    for _, row in df.iterrows():

        key = (
            int(row["transition"]),
            int(row["split_layer"]),
        )

        execution_cost[key] = float(
            row[
                "ideal_e2e_no_overhead_ms"
            ]
        )

    # =====================================================
    # 2. DP TABLES
    # =====================================================
    #
    # dp[t][s]
    #   minimum cumulative cost up to transition t
    #   if transition t uses split s
    #
    # parent[t][s]
    #   previous split that gives that minimum
    # =====================================================

    dp = {}
    parent = {}

    # =====================================================
    # 3. INITIAL TRANSITION
    # =====================================================

    first_transition = transitions[0]

    dp[first_transition] = {}
    parent[first_transition] = {}

    for split_layer in split_layers:

        dp[first_transition][
            split_layer
        ] = execution_cost[
            (
                first_transition,
                split_layer,
            )
        ]

        # No previous split exists.
        parent[first_transition][
            split_layer
        ] = None

    # =====================================================
    # 4. DP RECURRENCE
    # =====================================================

    for idx in range(
        1,
        len(transitions)
    ):

        transition = transitions[idx]

        previous_transition = (
            transitions[idx - 1]
        )

        dp[transition] = {}
        parent[transition] = {}

        for current_split in split_layers:

            current_execution = (
                execution_cost[
                    (
                        transition,
                        current_split,
                    )
                ]
            )

            best_total = None
            best_previous_split = None

            # -----------------------------------------
            # Try every possible previous split.
            # -----------------------------------------

            for previous_split in split_layers:

                previous_cost = (
                    dp[
                        previous_transition
                    ][
                        previous_split
                    ]
                )

                if (
                    previous_split
                    != current_split
                ):
                    switching_cost = (
                        switch_overhead_ms
                    )

                else:
                    switching_cost = 0.0

                candidate_total = (
                    previous_cost
                    + switching_cost
                    + current_execution
                )

                if (
                    best_total is None
                    or
                    candidate_total
                    < best_total
                ):
                    best_total = (
                        candidate_total
                    )

                    best_previous_split = (
                        previous_split
                    )

            dp[transition][
                current_split
            ] = best_total

            parent[transition][
                current_split
            ] = best_previous_split

    # =====================================================
    # 5. FIND BEST FINAL SPLIT
    # =====================================================

    last_transition = transitions[-1]

    best_final_split = min(
        split_layers,
        key=lambda split:
            dp[last_transition][split],
    )

    optimal_total_ms = (
        dp[
            last_transition
        ][
            best_final_split
        ]
    )

    # =====================================================
    # 6. BACKTRACK OPTIMAL PATH
    # =====================================================

    selected_splits = {}

    current_split = (
        best_final_split
    )

    for idx in range(
        len(transitions) - 1,
        -1,
        -1,
    ):

        transition = transitions[idx]

        selected_splits[
            transition
        ] = current_split

        previous_split = (
            parent[
                transition
            ][
                current_split
            ]
        )

        current_split = (
            previous_split
        )

    # =====================================================
    # 7. BUILD SCHEDULE TABLE
    # =====================================================

    schedule_records = []

    previous_split = None

    total_execution_ms = 0.0
    total_switching_ms = 0.0
    num_switches = 0

    for transition in transitions:

        selected_split = (
            selected_splits[
                transition
            ]
        )

        execution_ms = (
            execution_cost[
                (
                    transition,
                    selected_split,
                )
            ]
        )

        if (
            previous_split is not None
            and
            selected_split
            != previous_split
        ):
            switching_ms = (
                switch_overhead_ms
            )

            num_switches += 1

        else:
            switching_ms = 0.0

        transition_total_ms = (
            execution_ms
            + switching_ms
        )

        total_execution_ms += (
            execution_ms
        )

        total_switching_ms += (
            switching_ms
        )

        schedule_records.append({
            "transition":
                transition,

            "selected_split":
                selected_split,

            "execution_ms":
                execution_ms,

            "switching_ms":
                switching_ms,

            "transition_total_ms":
                transition_total_ms,
        })

        previous_split = (
            selected_split
        )

    schedule_df = pd.DataFrame(
        schedule_records
    )

    # ---------------------------------------------
    # Sanity check:
    # reconstructed cost should equal DP optimum.
    # ---------------------------------------------

    reconstructed_total = (
        total_execution_ms
        + total_switching_ms
    )

    assert abs(
        reconstructed_total
        - optimal_total_ms
    ) < 1e-6, (
        "DP backtracking cost does not "
        "match optimal total."
    )

    return {
        "schedule":
            schedule_df,

        "total_execution_ms":
            total_execution_ms,

        "total_switching_ms":
            total_switching_ms,

        "total_ms":
            reconstructed_total,

        "num_switches":
            num_switches,
    }


def main():

    print("=" * 70)
    print(
        "EXPERIMENT 6: "
        "DYNAMIC-PROGRAMMING ADAPTIVE "
        "SPLIT SCHEDULING"
    )
    print("=" * 70)

    # =====================================================
    # 1. LOAD EXP5B RESULTS
    # =====================================================

    df = pd.read_csv(
        "outputs/exp5b/"
        "selective_compute_per_transition.csv"
    )

    # ---------------------------------------------
    # Use only the conservative threshold
    # that preserved exact outputs.
    # ---------------------------------------------

    df = df[
        df["threshold"]
        == SCHEDULER_THRESHOLD
    ].copy()

    print(
        f"\nScheduler threshold: "
        f"{SCHEDULER_THRESHOLD}"
    )

    print(
        f"Transitions: "
        f"{df['transition'].nunique()}"
    )

    print(
        f"Split layers: "
        f"{sorted(df['split_layer'].unique())}"
    )

    # =====================================================
    # 2. STATIC SPLIT BASELINES
    # =====================================================
    #
    # Fair comparison:
    #
    # Static:
    #   same threshold
    #   same idealized selective execution model
    #   one fixed split across all transitions
    #
    # Adaptive:
    #   same threshold
    #   same execution model
    #   split may change between transitions
    #   switching overhead included
    # =====================================================

    static_results = (
        df
        .groupby("split_layer")
        .agg(
            total_ms=(
                "ideal_e2e_no_overhead_ms",
                "sum",
            ),

            avg_ms=(
                "ideal_e2e_no_overhead_ms",
                "mean",
            ),
        )
        .reset_index()
    )

    best_static_row = (
        static_results
        .sort_values(
            "total_ms"
        )
        .iloc[0]
    )

    best_static_split = int(
        best_static_row[
            "split_layer"
        ]
    )

    best_static_total_ms = float(
        best_static_row[
            "total_ms"
        ]
    )

    print(
        "\n=== STATIC SPLIT RESULTS ==="
    )

    print(
        static_results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        f"\nBest static split: "
        f"{best_static_split}"
    )

    print(
        f"Best static total: "
        f"{best_static_total_ms:.6f} ms"
    )

    # =====================================================
    # 3. RUN DP SCHEDULER FOR EACH SWITCHING COST
    # =====================================================

    os.makedirs(
        "outputs/exp6",
        exist_ok=True,
    )

    summary_records = []

    for switch_overhead_ms in (
        SCHEDULER_SWITCH_OVERHEADS_MS
    ):

        result = (
            compute_optimal_schedule_dp(
                df=df,
                switch_overhead_ms=
                    switch_overhead_ms,
            )
        )

        adaptive_total_ms = (
            result["total_ms"]
        )

        improvement_ms = (
            best_static_total_ms
            -
            adaptive_total_ms
        )

        improvement_pct = (
            improvement_ms
            /
            best_static_total_ms
            *
            100.0
        )

        adaptive_better = (
            adaptive_total_ms
            <
            best_static_total_ms
        )

        summary_records.append({
            "switch_overhead_ms":
                switch_overhead_ms,

            "best_static_split":
                best_static_split,

            "best_static_total_ms":
                best_static_total_ms,

            "adaptive_execution_ms":
                result[
                    "total_execution_ms"
                ],

            "adaptive_switching_ms":
                result[
                    "total_switching_ms"
                ],

            "adaptive_total_ms":
                adaptive_total_ms,

            "num_switches":
                result[
                    "num_switches"
                ],

            "improvement_ms":
                improvement_ms,

            "improvement_pct":
                improvement_pct,

            "adaptive_better":
                adaptive_better,
        })

        # -----------------------------------------
        # Save optimal schedule for this
        # switching-overhead assumption.
        # -----------------------------------------

        overhead_name = (
            str(
                switch_overhead_ms
            )
            .replace(
                ".",
                "p",
            )
        )

        result[
            "schedule"
        ].to_csv(
            "outputs/exp6/"
            f"optimal_schedule_"
            f"{overhead_name}ms.csv",
            index=False,
        )

    # =====================================================
    # 4. SAVE SUMMARY
    # =====================================================

    summary = pd.DataFrame(
        summary_records
    )

    summary.to_csv(
        "outputs/exp6/"
        "adaptive_scheduler_summary.csv",
        index=False,
    )

    print(
        "\n=== DP ADAPTIVE SCHEDULER SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # =====================================================
    # 5. BREAK-EVEN ANALYSIS
    # =====================================================

    beneficial = summary[
        summary[
            "adaptive_better"
        ]
        == True
    ]

    if len(beneficial) > 0:

        max_tested_beneficial = (
            beneficial[
                "switch_overhead_ms"
            ]
            .max()
        )

        print(
            "\nLargest tested switching "
            "overhead where adaptive "
            "scheduling remains better "
            "than the best static split:"
        )

        print(
            f"{max_tested_beneficial:.6f} ms"
        )

    else:

        print(
            "\nAdaptive scheduling does not "
            "outperform the best static split "
            "for any tested switching overhead."
        )

    # =====================================================
    # 6. SPLIT-SELECTION DISTRIBUTION
    # =====================================================

    print(
        "\n=== SELECTED SPLIT DISTRIBUTION ==="
    )

    for switch_overhead_ms in (
        SCHEDULER_SWITCH_OVERHEADS_MS
    ):

        overhead_name = (
            str(
                switch_overhead_ms
            )
            .replace(
                ".",
                "p",
            )
        )

        schedule = pd.read_csv(
            "outputs/exp6/"
            f"optimal_schedule_"
            f"{overhead_name}ms.csv"
        )

        counts = (
            schedule[
                "selected_split"
            ]
            .value_counts()
            .sort_index()
        )

        print(
            f"\nSwitch overhead "
            f"{switch_overhead_ms} ms:"
        )

        print(
            counts.to_string()
        )

    # =====================================================
    # 7. IMPORTANT INTERPRETATION NOTE
    # =====================================================

    print(
        "\nNOTE:"
    )

    print(
        "Adaptive execution costs use "
        "Exp5B's idealized selective-compute "
        "latency estimates."
    )

    print(
        "The scheduler is globally optimized "
        "with dynamic programming."
    )

    print(
        "Switching overhead is analytically "
        "swept and is not a measured model-"
        "migration latency."
    )


if __name__ == "__main__":
    main()