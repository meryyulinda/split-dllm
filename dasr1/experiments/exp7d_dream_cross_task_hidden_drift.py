import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import random
import numpy as np
import torch
import pandas as pd

from configs.experiment_config import (
    MODEL_NAME,
    SEED,
    MAX_NEW_TOKENS,
    STEPS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    ALGORITHM,
    DRIFT_EPSILON,
    RQ5_PROMPTS,
    RQ5_LAYERS,
)

from src.model_loader import load_dream_model

from src.generation import (
    prepare_prompt,
    generate_with_token_history,
)

from src.instrumentation import (
    create_token_history_hook,
)

from src.hidden_state_tracker import (
    HiddenDriftTracker,
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "exp7d",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD DREAM ONCE
# ============================================================

print("Loading Dream...")

model, tokenizer = load_dream_model(
    model_name=MODEL_NAME
)

model.eval()

MODEL_DEVICE = next(
    model.parameters()
).device

print("Model device:", MODEL_DEVICE)


# ============================================================
# RESULT CONTAINERS
# ============================================================

all_merged_dfs = []
all_summary_dfs = []
all_token_records = []


# ============================================================
# CROSS-TASK LOOP
# ============================================================

for task_name, prompt in RQ5_PROMPTS.items():

    # Reset RNG before every task.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print()
    print("=" * 70)
    print(f"TASK: {task_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # PREPARE PROMPT
    # --------------------------------------------------------

    inputs = prepare_prompt(
        tokenizer,
        prompt,
    )

    prompt_len = inputs[
        "input_ids"
    ].shape[1]

    print("Prompt      :", prompt)
    print("Prompt len  :", prompt_len)
    print("Gen length  :", MAX_NEW_TOKENS)
    print("Steps       :", STEPS)
    print("Algorithm   :", ALGORITHM)
    print("Temperature :", TEMPERATURE)
    print("Top-p       :", TOP_P)
    print("Top-k       :", TOP_K)

    # --------------------------------------------------------
    # TOKEN TRAJECTORY HOOK
    # --------------------------------------------------------

    history_hook, token_history = (
        create_token_history_hook()
    )

    # --------------------------------------------------------
    # HIDDEN TRACKER
    # --------------------------------------------------------

    tracker = HiddenDriftTracker(
        selected_layers=RQ5_LAYERS,
        prompt_length=prompt_len,
        eps=DRIFT_EPSILON,
    )

    tracker.register(model)

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print("Running generation...")

    generation_result = (
        generate_with_token_history(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            steps=STEPS,
            temperature=TEMPERATURE,
            history_hook=history_hook,
            top_p=TOP_P,
            top_k=TOP_K,
            alg=ALGORITHM,
        )
    )

    # --------------------------------------------------------
    # TOKEN TRAJECTORY CHECK
    # --------------------------------------------------------

    print(
        "Token states:",
        len(token_history),
    )

    if len(token_history) != STEPS + 1:

        tracker.remove()

        raise RuntimeError(
            f"{task_name}: expected "
            f"{STEPS + 1} token states, "
            f"got {len(token_history)}."
        )

    # --------------------------------------------------------
    # FINAL SEQUENCE
    # --------------------------------------------------------

    final_sequence = (
        generation_result[
            "sequence"
        ]
        .unsqueeze(0)
        .to(MODEL_DEVICE)
    )

    # --------------------------------------------------------
    # FINAL ANALYSIS-ONLY FORWARD
    # --------------------------------------------------------

    final_attention_mask = torch.ones(
        final_sequence.shape,
        dtype=torch.bool,
        device=final_sequence.device,
    )

    print(
        "Running final analysis-only forward..."
    )

    with torch.no_grad():

        _ = model(
            input_ids=final_sequence,
            attention_mask=final_attention_mask,
        )

    tracker.remove()

    # --------------------------------------------------------
    # HIDDEN TRAJECTORY CHECK
    # --------------------------------------------------------

    print()
    print("Trajectory check")

    print(
        "Token states      :",
        len(token_history),
    )

    print(
        "Token transitions :",
        len(token_history) - 1,
    )

    for layer in RQ5_LAYERS:

        num_calls = tracker.call_counts[
            layer
        ]

        print(
            f"Layer {layer:2d} hidden states:",
            num_calls,
        )

        assert num_calls == STEPS + 1, (
            f"Layer {layer}: expected "
            f"{STEPS + 1} hidden states, "
            f"got {num_calls}"
        )

    task_hidden_df = pd.DataFrame(
        tracker.records
    )

    hidden_transition_count = (
        task_hidden_df[
            "transition"
        ].nunique()
    )

    print(
        "Hidden transitions:",
        hidden_transition_count,
    )

    assert hidden_transition_count == STEPS, (
        f"{task_name}: expected "
        f"{STEPS} hidden transitions, "
        f"got {hidden_transition_count}"
    )

    # --------------------------------------------------------
    # TOKEN CHANGE ANALYSIS
    # --------------------------------------------------------

    token_records = []

    for transition in range(
        1,
        len(token_history),
    ):

        previous = token_history[
            transition - 1
        ]

        current = token_history[
            transition
        ]

        previous_gen = previous[
            prompt_len:
            prompt_len + MAX_NEW_TOKENS
        ]

        current_gen = current[
            prompt_len:
            prompt_len + MAX_NEW_TOKENS
        ]

        changed_mask = (
            previous_gen
            != current_gen
        )

        changed_tokens = (
            changed_mask
            .sum()
            .item()
        )

        total_tokens = (
            changed_mask
            .numel()
        )

        changed_ratio = (
            changed_tokens
            / total_tokens
        )

        record = {
            "model":
                "Dream-v0-Instruct-7B",

            "task":
                task_name,

            "transition":
                transition,

            "changed_tokens":
                changed_tokens,

            "changed_ratio":
                changed_ratio,
        }

        token_records.append(
            record
        )

        all_token_records.append(
            record
        )

    token_df = pd.DataFrame(
        token_records
    )

    # --------------------------------------------------------
    # ADD MODEL/TASK TO HIDDEN RESULTS
    # --------------------------------------------------------

    task_hidden_df[
        "model"
    ] = "Dream-v0-Instruct-7B"

    task_hidden_df[
        "task"
    ] = task_name

    # --------------------------------------------------------
    # MERGE TOKEN + HIDDEN
    # --------------------------------------------------------

    merged = task_hidden_df.merge(
        token_df[
            [
                "transition",
                "changed_tokens",
                "changed_ratio",
            ]
        ],
        on="transition",
        how="inner",
    )

    matched_transitions = (
        merged[
            "transition"
        ].nunique()
    )

    print(
        "Matched transitions:",
        matched_transitions,
    )

    assert matched_transitions == STEPS, (
        f"{task_name}: expected "
        f"{STEPS} matched transitions"
    )

    all_merged_dfs.append(
        merged
    )

    # --------------------------------------------------------
    # TASK SUMMARY
    # --------------------------------------------------------

    task_summary = (
        merged
        .groupby(
            [
                "model",
                "task",
                "layer",
            ],
            as_index=False,
        )
        .agg(
            mean_changed_ratio=(
                "changed_ratio",
                "mean",
            ),

            mean_drift=(
                "mean_drift",
                "mean",
            ),

            median_drift=(
                "median_drift",
                "mean",
            ),

            mean_active_ratio_1e3=(
                "active_ratio_1e-3",
                "mean",
            ),

            min_active_ratio_1e3=(
                "active_ratio_1e-3",
                "min",
            ),

            max_active_ratio_1e3=(
                "active_ratio_1e-3",
                "max",
            ),
        )
    )

    all_summary_dfs.append(
        task_summary
    )

    # --------------------------------------------------------
    # FINAL OUTPUT TEXT
    # --------------------------------------------------------

    print()
    print("Final output:")
    print(
        generation_result[
            "text"
        ]
    )

    # --------------------------------------------------------
    # TASK SUMMARY
    # --------------------------------------------------------

    print()
    print(
        f"{task_name.upper()} SUMMARY"
    )

    print(
        task_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # LAYER 12 PATTERN
    # --------------------------------------------------------

    layer12 = merged[
        merged["layer"] == 12
    ].copy()

    token_change_transitions = (
        layer12[
            "changed_tokens"
        ]
        > 0
    ).sum()

    unchanged_transitions = (
        layer12[
            "changed_tokens"
        ]
        == 0
    ).sum()

    fully_active = (
        layer12[
            "active_ratio_1e-3"
        ]
        == 1.0
    ).sum()

    changed_only = layer12[
        layer12[
            "changed_tokens"
        ]
        > 0
    ]

    unchanged_only = layer12[
        layer12[
            "changed_tokens"
        ]
        == 0
    ]

    if len(changed_only) > 0:

        mean_active_when_changed = (
            changed_only[
                "active_ratio_1e-3"
            ].mean()
        )

    else:

        mean_active_when_changed = float(
            "nan"
        )

    if len(unchanged_only) > 0:

        mean_active_when_unchanged = (
            unchanged_only[
                "active_ratio_1e-3"
            ].mean()
        )

    else:

        mean_active_when_unchanged = float(
            "nan"
        )

    print()
    print("Layer 12 pattern")

    print(
        "Mean changed-token ratio:",
        layer12[
            "changed_ratio"
        ].mean(),
    )

    print(
        "Mean active hidden ratio:",
        layer12[
            "active_ratio_1e-3"
        ].mean(),
    )

    print(
        "Min active hidden ratio:",
        layer12[
            "active_ratio_1e-3"
        ].min(),
    )

    print(
        "Max active hidden ratio:",
        layer12[
            "active_ratio_1e-3"
        ].max(),
    )

    print(
        "Transitions with token changes:",
        token_change_transitions,
        "/",
        len(layer12),
    )

    print(
        "Fully unchanged transitions:",
        unchanged_transitions,
        "/",
        len(layer12),
    )

    print(
        "Fully active hidden transitions:",
        fully_active,
        "/",
        len(layer12),
    )

    print(
        "Mean active ratio when tokens changed:",
        mean_active_when_changed,
    )

    print(
        "Mean active ratio when tokens unchanged:",
        mean_active_when_unchanged,
    )

    del tracker
    del generation_result
    del final_sequence
    del token_history

    torch.cuda.empty_cache()


# ============================================================
# COMBINE RESULTS
# ============================================================

all_df = pd.concat(
    all_merged_dfs,
    ignore_index=True,
)

summary_df = pd.concat(
    all_summary_dfs,
    ignore_index=True,
)

all_token_df = pd.DataFrame(
    all_token_records
)


# ============================================================
# SAVE
# ============================================================

per_transition_path = os.path.join(
    OUTPUT_DIR,
    "dream_rq5_hidden_drift_per_transition.csv",
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "dream_rq5_hidden_drift_summary.csv",
)

token_path = os.path.join(
    OUTPUT_DIR,
    "dream_rq5_token_changes.csv",
)

all_df.to_csv(
    per_transition_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)

all_token_df.to_csv(
    token_path,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("RQ5 DREAM CROSS-TASK SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(per_transition_path)
print(summary_path)
print(token_path)