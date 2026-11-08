import os
import sys

# Allow running with: python experiments/exp8a_block_temporal_drift.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import json

import numpy as np
import pandas as pd
import torch

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
    SCHEDULER_THRESHOLD,
)
from src.model_loader import load_dream_model
from src.generation import generate_with_token_history
from src.instrumentation import create_token_history_hook
from src.utils import set_seed


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "exp8a")


class BlockTemporalDriftProfiler:
    """Profile consecutive whole-block outputs; no caching or model modification.

    Layers are numbered starting at 1, as in existing Dream experiments.
    A transition t compares H_l(state t-1) with H_l(state t).
    """

    def __init__(self, selected_layers, prompt_length, eps, row_threshold):
        self.selected_layers = list(selected_layers)
        self.prompt_length = int(prompt_length)
        self.eps = float(eps)
        self.row_threshold = float(row_threshold)
        self.previous = {}
        self.records = []
        self.handles = []
        self.state_index = None
        self.call_counts = {layer: 0 for layer in self.selected_layers}

    def _relative_frobenius(self, current, previous):
        # Frobenius norm = L2 norm of the flattened (tokens x features) matrix.
        return (
            torch.linalg.vector_norm(current - previous)
            / (torch.linalg.vector_norm(current) + self.eps)
        )

    def _create_hook(self, layer_number):
        def hook(module, inputs, output):
            if self.state_index is None:
                raise RuntimeError("state_index must be set before each forward")

            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError(
                    f"Unexpected output from Dream layer {layer_number}: "
                    f"{type(hidden)} / {getattr(hidden, 'shape', None)}"
                )
            if hidden.size(0) != 1:
                raise RuntimeError("Exp8A currently expects batch size 1")

            # Keep previous feature matrix on the same GPU for cheap comparison.
            # float32 storage avoids accumulating bfloat16 rounding errors in norms.
            current = hidden[0].detach().to(dtype=torch.float32).clone()
            if not (0 < self.prompt_length < current.size(0)):
                raise RuntimeError(
                    f"Invalid prompt_length={self.prompt_length}, "
                    f"sequence_length={current.size(0)}"
                )

            if layer_number in self.previous:
                previous = self.previous[layer_number]
                if previous.shape != current.shape:
                    raise RuntimeError(
                        f"Changed shape at layer {layer_number}: "
                        f"{tuple(previous.shape)} -> {tuple(current.shape)}"
                    )

                current_prompt = current[: self.prompt_length]
                previous_prompt = previous[: self.prompt_length]
                current_gen = current[self.prompt_length :]
                previous_gen = previous[self.prompt_length :]

                full_delta = current - previous
                gen_delta = current_gen - previous_gen
                gen_row_drift = (
                    torch.linalg.vector_norm(gen_delta, dim=-1)
                    / (torch.linalg.vector_norm(current_gen, dim=-1) + self.eps)
                )

                # These quantities describe feature similarity, NOT safe cache reuse.
                full_drift = self._relative_frobenius(current, previous)
                prompt_drift = self._relative_frobenius(
                    current_prompt, previous_prompt
                )
                generated_drift = self._relative_frobenius(
                    current_gen, previous_gen
                )
                full_cosine = torch.nn.functional.cosine_similarity(
                    current.flatten(), previous.flatten(), dim=0, eps=self.eps
                )
                active_row_ratio = (
                    (gen_row_drift > self.row_threshold).float().mean()
                )

                metrics = torch.stack(
                    [
                        full_drift,
                        prompt_drift,
                        generated_drift,
                        full_cosine,
                        full_delta.abs().max(),
                        active_row_ratio,
                    ]
                ).cpu().tolist()

                self.records.append(
                    {
                        "layer": layer_number,
                        "transition": self.state_index,
                        "full_block_drift": metrics[0],
                        "prompt_drift": metrics[1],
                        "generated_drift": metrics[2],
                        "full_block_cosine": metrics[3],
                        "max_abs_diff": metrics[4],
                        "generated_active_row_ratio": metrics[5],
                    }
                )

            self.previous[layer_number] = current
            self.call_counts[layer_number] += 1

        return hook

    def register(self, model):
        for layer_number in self.selected_layers:
            # Dream uses 0-based model.model.layers, but experiment layer labels
            # are 1-based, matching the project's HiddenDriftTracker.
            layer = model.model.layers[layer_number - 1]
            self.handles.append(
                layer.register_forward_hook(self._create_hook(layer_number))
            )

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _as_1d_cpu_long(state):
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"Expected a token tensor, got {type(state)}")
    if state.ndim != 1:
        raise ValueError(f"Expected 1D history state, got {tuple(state.shape)}")
    return state.detach().cpu().to(dtype=torch.long)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 72)
    print("EXPERIMENT 8A: TEMPORAL WHOLE-BLOCK DRIFT PROFILING (DREAM)")
    print("=" * 72)
    print(f"Model          : {MODEL_NAME}")
    print(f"Prompt         : {PROMPT}")
    print(f"Generation     : {MAX_NEW_TOKENS} tokens / {STEPS} steps")
    print(f"Algorithm      : {ALGORITHM}")
    print(f"TOP_P / TOP_K  : {TOP_P} / {TOP_K}")
    print(f"Selected layers: {SELECTED_LAYERS} (1-based)")
    print(f"Row threshold  : {SCHEDULER_THRESHOLD} (diagnostic only)")

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")

    set_seed(SEED)
    model, tokenizer = load_dream_model(MODEL_NAME)
    model.eval()

    # First collect the OFFICIAL generation trajectory using the existing hook.
    # Do not attach block hooks until generation is finished.
    history_hook, token_history = create_token_history_hook()
    set_seed(SEED)
    with torch.inference_mode():
        _ = generate_with_token_history(
            model=model,
            tokenizer=tokenizer,
            prompt=PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            steps=STEPS,
            temperature=TEMPERATURE,
            history_hook=history_hook,
            top_p=TOP_P,
            top_k=TOP_K,
            alg=ALGORITHM,
        )

    states = [_as_1d_cpu_long(x) for x in token_history]
    expected_states = STEPS + 1
    if len(states) != expected_states:
        raise RuntimeError(
            f"Expected {expected_states} token states (initial + {STEPS} updates), "
            f"got {len(states)}. Check generation instrumentation."
        )

    full_length = states[0].numel()
    prompt_length = full_length - MAX_NEW_TOKENS
    if prompt_length <= 0:
        raise RuntimeError("Cannot infer prompt length from token history")
    if any(x.numel() != full_length for x in states):
        raise RuntimeError("All token states must have the same sequence length")

    token_change_records = []
    for t in range(1, len(states)):
        changed = int(
            (states[t][prompt_length:] != states[t - 1][prompt_length:])
            .sum()
            .item()
        )
        token_change_records.append(
            {
                "transition": t,
                "changed_token_count": changed,
                "changed_token_ratio": changed / MAX_NEW_TOKENS,
                "token_changed": bool(changed > 0),
            }
        )
    token_df = pd.DataFrame(token_change_records)
    print(f"Captured states     : {len(states)}")
    print(f"Consecutive changes : {len(states) - 1}")
    print(f"Prompt length       : {prompt_length}")
    print(f"Generated positions : {MAX_NEW_TOKENS}")
    print(f"Token-changing transitions: {int(token_df['token_changed'].sum())}")
    print(f"Unchanged transitions     : {int((~token_df['token_changed']).sum())}")

    # Replay each of the 65 token states through the unsplit full model.
    # This yields 64 aligned hidden-state transitions. This is profiling-only;
    # none of these extra forwards should be included in any latency result.
    profiler = BlockTemporalDriftProfiler(
        selected_layers=SELECTED_LAYERS,
        prompt_length=prompt_length,
        eps=DRIFT_EPSILON,
        row_threshold=SCHEDULER_THRESHOLD,
    )
    profiler.register(model)
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            for state_index, state in enumerate(states):
                profiler.state_index = state_index
                input_ids = state.unsqueeze(0).to(device)
                attention_mask = torch.ones_like(
                    input_ids, dtype=torch.bool, device=device
                )
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                if state_index % 10 == 0 or state_index == len(states) - 1:
                    print(f"Replayed state {state_index:>2}/{STEPS}")
    finally:
        profiler.remove()

    for layer, count in profiler.call_counts.items():
        if count != expected_states:
            raise RuntimeError(
                f"Layer {layer} had {count} forwards, expected {expected_states}"
            )

    block_df = pd.DataFrame(profiler.records)
    if len(block_df) != STEPS * len(SELECTED_LAYERS):
        raise RuntimeError(
            f"Expected {STEPS * len(SELECTED_LAYERS)} drift rows, "
            f"got {len(block_df)}"
        )
    per_transition = block_df.merge(
        token_df, on="transition", how="left", validate="many_to_one"
    ).sort_values(["transition", "layer"])

    summary = (
        per_transition.groupby("layer", as_index=False)
        .agg(
            transitions=("transition", "count"),
            mean_full_block_drift=("full_block_drift", "mean"),
            median_full_block_drift=("full_block_drift", "median"),
            p90_full_block_drift=(
                "full_block_drift", lambda x: float(np.percentile(x, 90))
            ),
            mean_prompt_drift=("prompt_drift", "mean"),
            mean_generated_drift=("generated_drift", "mean"),
            mean_full_block_cosine=("full_block_cosine", "mean"),
            mean_generated_active_row_ratio=(
                "generated_active_row_ratio", "mean"
            ),
        )
        .sort_values("layer")
    )

    # Separate summaries expose how a layer behaves with and without changes
    # in the visible generated token IDs (without implying reuse safety).
    by_token_change = (
        per_transition.groupby(["layer", "token_changed"], as_index=False)
        .agg(
            transitions=("transition", "count"),
            mean_full_block_drift=("full_block_drift", "mean"),
            mean_generated_drift=("generated_drift", "mean"),
            mean_generated_active_row_ratio=(
                "generated_active_row_ratio", "mean"
            ),
        )
        .sort_values(["layer", "token_changed"])
    )

    per_transition_path = os.path.join(
        OUTPUT_DIR, "block_drift_per_transition.csv"
    )
    summary_path = os.path.join(OUTPUT_DIR, "block_drift_summary.csv")
    by_change_path = os.path.join(OUTPUT_DIR, "block_drift_by_token_change.csv")
    metadata_path = os.path.join(OUTPUT_DIR, "experiment_metadata.json")

    per_transition.to_csv(per_transition_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_token_change.to_csv(by_change_path, index=False)

    metadata = {
        "experiment": "8A: whole-block temporal drift (descriptive only)",
        "model": MODEL_NAME,
        "seed": SEED,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "algorithm": ALGORITHM,
        "selected_layers_1_based": list(SELECTED_LAYERS),
        "prompt_length": prompt_length,
        "captured_states": len(states),
        "analyzed_transitions": STEPS,
        "drift_epsilon": DRIFT_EPSILON,
        "diagnostic_row_threshold": SCHEDULER_THRESHOLD,
        "gpu": torch.cuda.get_device_name(device),
        "warning": (
            "Drift profiling is not an actual caching implementation, "
            "a cache correctness test, or an E2E latency measurement."
        ),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\n" + "=" * 72)
    print("WHOLE-BLOCK TEMPORAL DRIFT SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nBY TOKEN-CHANGE CONDITION")
    print(by_token_change.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved:")
    for p in (per_transition_path, summary_path, by_change_path, metadata_path):
        print("  ", p)
    print(
        "\nNOTE: This experiment does NOT skip any block. "
        "Use Exp8B oracle interventions to test cache correctness."
    )


if __name__ == "__main__":
    main()
