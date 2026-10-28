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

from transformers import AutoModel, AutoTokenizer

from configs.experiment_config import (
    SEED,
    STEPS,
    MAX_NEW_TOKENS,
    RQ5_PROMPTS,
    RQ5_LAYERS,
    RQ5_THRESHOLD,
)

from src.llada_generation import generate_with_history

from src.llada_hidden_state_tracker import (
    LLaDAHiddenStateTracker,
    normalized_row_drift,
)


# ============================================================
# MODEL-SPECIFIC CONFIG
# ============================================================

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"

DEVICE = "cuda"

GEN_LENGTH = MAX_NEW_TOKENS
BLOCK_LENGTH = MAX_NEW_TOKENS

LAYERS = RQ5_LAYERS
DRIFT_THRESHOLD = RQ5_THRESHOLD

LLADA_TEMPERATURE = 0.0
LLADA_CFG_SCALE = 0.0
LLADA_REMASKING = "low_confidence"


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
    "exp7c",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("Loading LLaDA...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(DEVICE).eval()

print("Model loaded.")


# ============================================================
# RESULT CONTAINERS
# ============================================================

all_records = []
task_summaries = []


# ============================================================
# LOOP OVER RQ5 TASKS
# ============================================================

for task_name, prompt in RQ5_PROMPTS.items():

    # Reset RNG before each task.
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

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    encoded = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids = encoded[
        "input_ids"
    ].to(DEVICE)

    attention_mask = encoded[
        "attention_mask"
    ].to(DEVICE)

    prompt_len = input_ids.shape[1]

    print("Prompt      :", prompt)
    print("Prompt len  :", prompt_len)
    print("Gen length  :", GEN_LENGTH)
    print("Steps       :", STEPS)

    # --------------------------------------------------------
    # REGISTER TRACKER
    # --------------------------------------------------------

    tracker = LLaDAHiddenStateTracker(
        model=model,
        layer_indices=LAYERS,
    )

    tracker.register()

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print("Running generation...")

    with torch.no_grad():

        output, token_history, transition_history = (
            generate_with_history(
                model=model,
                prompt=input_ids,
                attention_mask=attention_mask,
                steps=STEPS,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                temperature=LLADA_TEMPERATURE,
                cfg_scale=LLADA_CFG_SCALE,
                remasking=LLADA_REMASKING,
            )
        )

    # --------------------------------------------------------
    # FINAL ANALYSIS-ONLY FORWARD
    # --------------------------------------------------------

    final_attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones(
                (
                    attention_mask.shape[0],
                    GEN_LENGTH,
                ),
                dtype=attention_mask.dtype,
                device=DEVICE,
            ),
        ],
        dim=-1,
    )

    print(
        "Running final analysis-only forward..."
    )

    with torch.no_grad():

        _ = model(
            output,
            attention_mask=final_attention_mask,
        )

    tracker.remove()

    # --------------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------------

    print()
    print("Trajectory check")

    print(
        "Token states      :",
        len(token_history),
    )

    print(
        "Token transitions :",
        len(transition_history),
    )

    assert len(token_history) == STEPS + 1

    assert len(transition_history) == STEPS

    for layer_idx in LAYERS:

        num_hidden_states = tracker.num_states(
            layer_idx
        )

        print(
            f"Layer {layer_idx:2d} hidden states:",
            num_hidden_states,
        )

        assert num_hidden_states == STEPS + 1, (
            f"Layer {layer_idx}: "
            f"expected {STEPS + 1} hidden states, "
            f"got {num_hidden_states}"
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    generated_ids = output[
        :,
        prompt_len:
    ]

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    print()
    print("Final output:")
    print(generated_text)

    # --------------------------------------------------------
    # HIDDEN DRIFT ANALYSIS
    # --------------------------------------------------------

    task_records = []

    for layer_idx in LAYERS:

        history = tracker.hidden_history[
            layer_idx
        ]

        for transition_idx in range(
            1,
            len(history),
        ):

            previous = history[
                transition_idx - 1
            ]

            current = history[
                transition_idx
            ]

            # Generated region only.
            previous_gen = previous[
                :,
                prompt_len:
                prompt_len + GEN_LENGTH,
                :,
            ]

            current_gen = current[
                :,
                prompt_len:
                prompt_len + GEN_LENGTH,
                :,
            ]

            row_drift = normalized_row_drift(
                current=current_gen,
                previous=previous_gen,
            )

            active_mask = (
                row_drift
                > DRIFT_THRESHOLD
            )

            active_rows = (
                active_mask
                .sum()
                .item()
            )

            total_rows = (
                active_mask
                .numel()
            )

            active_ratio = (
                active_rows
                / total_rows
            )

            token_record = transition_history[
                transition_idx - 1
            ]

            record = {
                "model":
                    "LLaDA-8B-Instruct",

                "task":
                    task_name,

                "layer":
                    layer_idx,

                "transition":
                    transition_idx,

                "changed_tokens":
                    token_record[
                        "changed_tokens"
                    ],

                "changed_ratio":
                    token_record[
                        "changed_ratio"
                    ],

                "mean_drift":
                    row_drift
                    .mean()
                    .item(),

                "median_drift":
                    row_drift
                    .median()
                    .item(),

                "max_drift":
                    row_drift
                    .max()
                    .item(),

                "active_rows":
                    active_rows,

                "total_rows":
                    total_rows,

                "active_ratio_1e-3":
                    active_ratio,
            }

            task_records.append(
                record
            )

            all_records.append(
                record
            )

    # --------------------------------------------------------
    # TASK SUMMARY
    # --------------------------------------------------------

    task_df = pd.DataFrame(
        task_records
    )

    task_summary = (
        task_df
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

    task_summaries.append(
        task_summary
    )

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

    layer12 = task_df[
        task_df["layer"] == 12
    ].copy()

    fully_active = (
        layer12[
            "active_ratio_1e-3"
        ]
        == 1.0
    ).sum()

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
        "Fully active transitions:",
        fully_active,
        "/",
        len(layer12),
    )

    del output
    del token_history

    torch.cuda.empty_cache()


# ============================================================
# COMBINE ALL TASKS
# ============================================================

all_df = pd.DataFrame(
    all_records
)

summary_df = pd.concat(
    task_summaries,
    ignore_index=True,
)


# ============================================================
# SAVE
# ============================================================

per_transition_path = os.path.join(
    OUTPUT_DIR,
    "llada_rq5_hidden_drift_per_transition.csv",
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "llada_rq5_hidden_drift_summary.csv",
)

all_df.to_csv(
    per_transition_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("RQ5 LLaDA CROSS-TASK SUMMARY")
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