import os
import sys

# Run from dasr1/: python experiments/exp8b_oracle_block_residual_reuse.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import json
import math

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
)
from src.model_loader import load_dream_model
from src.generation import generate_with_token_history
from src.instrumentation import create_token_history_hook
from src.utils import set_seed

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "exp8b")


class CachedBlockResidualInjector:
    """One-step oracle test of a *single-block residual update* cache.

    A transformer block maps H_in -> H_out. Store R = H_out - H_in
    from the previous exact denoising state, then approximate the current
    block's output as H_in(current) + R(previous).

    Crucially, the current upstream activation is retained, unlike the
    original Exp8B whole-output replacement that always reproduced the
    previous state's logits for every candidate layer.

    The forward hook intercepts output *after computing the block*; this
    evaluates fidelity ONLY, not the cost/speedup of actually skipping it.
    """

    def __init__(self, selected_layers):
        self.selected_layers = list(selected_layers)
        self.handles = []
        self.mode = "idle"
        self.target_layer = None
        self.current_outputs = {}
        self.current_residuals = {}
        self.previous_residuals = {}
        self.injected_count = 0
        self.injection_relative_error = None

    @staticmethod
    def hidden_from_output(output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("Expected layer output tensor [B, N, D] or tuple thereof")
        return hidden

    def _make_hook(self, layer_number):
        def hook(_module, inputs, output):
            current_input = inputs[0]
            current_output = self.hidden_from_output(output)
            if current_input.shape != current_output.shape:
                raise RuntimeError("Residual cache requires same-shaped block input/output")

            if self.mode == "capture":
                self.current_outputs[layer_number] = current_output.detach().clone()
                # FP32 residual is essential to avoid loss from bf16 subtraction.
                self.current_residuals[layer_number] = (
                    current_output.detach().float() - current_input.detach().float()
                ).clone()
                return None

            if self.mode == "reuse" and self.target_layer == layer_number:
                old_residual = self.previous_residuals[layer_number]
                if old_residual.shape != current_input.shape or old_residual.device != current_input.device:
                    raise RuntimeError(f"Cache shape/device mismatch at layer {layer_number}")
                approx = (
                    current_input.float() + old_residual
                ).to(dtype=current_output.dtype)
                self.injection_relative_error = float((
                    torch.linalg.vector_norm(approx.float() - current_output.float()) /
                    (torch.linalg.vector_norm(current_output.float()) + DRIFT_EPSILON)
                ).item())
                self.injected_count += 1
                if isinstance(output, tuple):
                    return (approx, *output[1:])
                return approx

            return None
        return hook

    def register(self, model):
        for layer_number in self.selected_layers:
            layer = model.model.layers[layer_number - 1]  # 1-based layer labels
            self.handles.append(layer.register_forward_hook(self._make_hook(layer_number)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.mode = "idle"

    def begin_capture(self):
        self.mode = "capture"
        self.target_layer = None
        self.current_outputs = {}
        self.current_residuals = {}

    def capture_done(self):
        missing = set(self.selected_layers) - set(self.current_residuals)
        if missing:
            raise RuntimeError(f"Missing baseline layer activations: {missing}")
        self.mode = "idle"
        return self.current_outputs, self.current_residuals

    def begin_reuse(self, layer_number, previous_residuals):
        self.mode = "reuse"
        self.target_layer = layer_number
        self.previous_residuals = previous_residuals
        self.injected_count = 0
        self.injection_relative_error = None

    def reuse_done(self):
        if self.injected_count != 1:
            raise RuntimeError(
                f"Block injection was invoked {self.injected_count} times; expected exactly one"
            )
        error = self.injection_relative_error
        self.mode = "idle"
        self.target_layer = None
        return error


def _as_1d_cpu_long(state):
    if not isinstance(state, torch.Tensor) or state.ndim != 1:
        raise TypeError("Each token-history state must be a 1D tensor")
    return state.detach().cpu().to(dtype=torch.long)


def _relative_l2(current, previous, eps):
    current = current.detach().float()
    previous = previous.detach().float()
    return float((
        torch.linalg.vector_norm(current - previous) /
        (torch.linalg.vector_norm(current) + eps)
    ).item())


def _mask_token_id(tokenizer, model):
    for obj in (tokenizer, getattr(model, "config", None)):
        if obj is not None:
            value = getattr(obj, "mask_token_id", None)
            if isinstance(value, int) and value >= 0:
                return value
    return None


def _compute_metrics(reference_gen, approximate_gen, masked_positions, eps):
    if reference_gen.shape != approximate_gen.shape:
        raise RuntimeError("Reference/reuse generated-logit shapes differ")

    reference_gen = reference_gen.float()
    approximate_gen = approximate_gen.float()
    delta = approximate_gen - reference_gen
    same_argmax = reference_gen.argmax(-1).eq(approximate_gen.argmax(-1))

    output = {
        "argmax_match_ratio_generated": float(same_argmax.float().mean().item()),
        "argmax_mismatch_count_generated": int((~same_argmax).sum().item()),
        "all_generated_argmax_match": bool(same_argmax.all().item()),
        "mean_abs_logit_diff_generated": float(delta.abs().mean().item()),
        "relative_l2_logit_error_generated": float((
            torch.linalg.vector_norm(delta) /
            (torch.linalg.vector_norm(reference_gen) + eps)
        ).item()),
        "masked_positions": int(masked_positions.sum().item()) if masked_positions is not None else None,
        "argmax_match_ratio_masked": math.nan,
        "all_masked_argmax_match": None,
    }
    if masked_positions is not None and bool(masked_positions.any().item()):
        subset = same_argmax[masked_positions]
        output["argmax_match_ratio_masked"] = float(subset.float().mean().item())
        output["all_masked_argmax_match"] = bool(subset.all().item())
    return output


def _one_forward(model, ids):
    output = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids, dtype=torch.bool),
        use_cache=False,
        return_dict=True,
    )
    return output.logits


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp8B residual reuse requires a CUDA GPU")
    if not SELECTED_LAYERS:
        raise RuntimeError("SELECTED_LAYERS must not be empty")

    print("=" * 78)
    print("EXP8B (CORRECTED): ONE-STEP ORACLE SINGLE-BLOCK RESIDUAL REUSE (DREAM)")
    print("=" * 78)
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}")
    print(f"Generation: {MAX_NEW_TOKENS} tokens / {STEPS} steps")
    print(f"Algorithm: {ALGORITHM} | TOP_P / TOP_K: {TOP_P} / {TOP_K}")
    print(f"Candidate layers: {SELECTED_LAYERS} (1-based)")

    set_seed(SEED)
    model, tokenizer = load_dream_model(MODEL_NAME)
    model.eval()
    device = next(model.parameters()).device
    mask_token_id = _mask_token_id(tokenizer, model)

    history_hook, token_history = create_token_history_hook()
    set_seed(SEED)
    with torch.inference_mode():
        _ = generate_with_token_history(
            model=model, tokenizer=tokenizer, prompt=PROMPT,
            max_new_tokens=MAX_NEW_TOKENS, steps=STEPS,
            temperature=TEMPERATURE, history_hook=history_hook,
            top_p=TOP_P, top_k=TOP_K, alg=ALGORITHM,
        )

    states = [_as_1d_cpu_long(s) for s in token_history]
    if len(states) != STEPS + 1:
        raise RuntimeError(f"Expected {STEPS + 1} states, got {len(states)}")
    seq_len = states[0].numel()
    prompt_length = seq_len - MAX_NEW_TOKENS
    if prompt_length <= 0 or any(s.numel() != seq_len for s in states):
        raise RuntimeError("Inconsistent state shapes/prompt length")
    if any(not torch.equal(s[:prompt_length], states[0][:prompt_length]) for s in states):
        raise RuntimeError("Prompt tokens unexpectedly changed")
    n_changed_by_step = [
        int((states[t][prompt_length:] != states[t-1][prompt_length:]).sum().item())
        for t in range(1, len(states))
    ]
    print(f"CUDA device: {device}")
    print(f"Captured states: {len(states)} | transitions: {len(states)-1}")
    print(f"Changed: {sum(x>0 for x in n_changed_by_step)} | unchanged: {sum(x==0 for x in n_changed_by_step)}")

    injector = CachedBlockResidualInjector(SELECTED_LAYERS)
    injector.register(model)
    records = []
    try:
        with torch.inference_mode():
            injector.begin_capture()
            _ = _one_forward(model, states[0].unsqueeze(0).to(device))
            previous_outputs, previous_residuals = injector.capture_done()

            for t in range(1, len(states)):
                ids = states[t].unsqueeze(0).to(device)
                injector.begin_capture()
                full_logits = _one_forward(model, ids)
                current_outputs, current_residuals = injector.capture_done()
                ref_generated = full_logits[0, prompt_length:, :]
                masks = ids[0, prompt_length:].eq(mask_token_id) if mask_token_id is not None else None
                n_changed = n_changed_by_step[t - 1]

                for layer in SELECTED_LAYERS:
                    output_drift = _relative_l2(current_outputs[layer], previous_outputs[layer], DRIFT_EPSILON)
                    residual_drift = _relative_l2(current_residuals[layer], previous_residuals[layer], DRIFT_EPSILON)
                    injector.begin_reuse(layer, previous_residuals)
                    try:
                        reused_logits = _one_forward(model, ids)
                    finally:
                        injection_error = injector.reuse_done()

                    metrics = _compute_metrics(
                        ref_generated, reused_logits[0, prompt_length:, :], masks, DRIFT_EPSILON
                    )
                    records.append({
                        "transition": t,
                        "layer": layer,
                        "token_changed": bool(n_changed > 0),
                        "changed_token_count": n_changed,
                        "changed_token_ratio": n_changed / MAX_NEW_TOKENS,
                        "whole_output_drift": output_drift,
                        "block_residual_drift": residual_drift,
                        "injected_block_relative_error": injection_error,
                        **metrics,
                    })

                    if n_changed == 0 and not metrics["all_generated_argmax_match"]:
                        raise RuntimeError(
                            f"Identical-state control failed at step={t}, layer={layer}"
                        )
                    del reused_logits

                previous_outputs = current_outputs
                previous_residuals = current_residuals
                del full_logits
                if t % 10 == 0 or t == STEPS:
                    print(f"Evaluated transition {t}/{STEPS}")
    finally:
        injector.remove()

    df = pd.DataFrame(records).sort_values(["layer", "transition"])
    if len(df) != STEPS * len(SELECTED_LAYERS):
        raise RuntimeError("Wrong number of records")
    agg = {
        "transitions": ("transition", "count"),
        "mean_whole_output_drift": ("whole_output_drift", "mean"),
        "mean_block_residual_drift": ("block_residual_drift", "mean"),
        "mean_injected_block_error": ("injected_block_relative_error", "mean"),
        "mean_argmax_match_generated": ("argmax_match_ratio_generated", "mean"),
        "exact_argmax_transitions": ("all_generated_argmax_match", "sum"),
        "mean_logit_relative_error": ("relative_l2_logit_error_generated", "mean"),
        "mean_masked_argmax_match": ("argmax_match_ratio_masked", "mean"),
    }
    summary = df.groupby("layer", as_index=False).agg(**agg).sort_values("layer")
    by_change = df.groupby(["layer", "token_changed"], as_index=False).agg(**agg)

    files = {
        "per_transition": os.path.join(OUTPUT_DIR, "block_residual_reuse_per_transition.csv"),
        "summary": os.path.join(OUTPUT_DIR, "block_residual_reuse_summary.csv"),
        "by_condition": os.path.join(OUTPUT_DIR, "block_residual_reuse_by_token_change.csv"),
        "metadata": os.path.join(OUTPUT_DIR, "experiment_metadata.json"),
    }
    df.to_csv(files["per_transition"], index=False)
    summary.to_csv(files["summary"], index=False)
    by_change.to_csv(files["by_condition"], index=False)

    with open(files["metadata"], "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Exp8B corrected - one-step oracle cached single-block residual",
            "model": MODEL_NAME,
            "prompt": PROMPT,
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "steps": STEPS,
            "algorithm": ALGORITHM,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "layers_1_based": list(SELECTED_LAYERS),
            "mask_token_id": mask_token_id,
            "token_states": len(states),
            "transitions": len(states) - 1,
            "prompt_length": prompt_length,
            "mechanism": "H_out_approx(t) = H_in_exact(t) + [H_out_exact(t-1) - H_in_exact(t-1)]",
            "caveat": (
                "This is an oracle one-step teacher-forced fidelity test of a cached block residual update. "
                "The target block is still fully computed before its output is overwritten; "
                "no real block skipping or latency reduction occurs. "
                "An end-to-end free-running trajectory validation is not yet performed. "
                "The earlier Exp8B whole-output replacement always collapsed to stale previous-state logits "
                "regardless of injected layer; this method retains current upstream inputs."
            ),
        }, f, indent=2)

    print("\n" + "=" * 78)
    print("CORRECTED BLOCK-RESIDUAL REUSE: ALL TRANSITIONS")
    print("=" * 78)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nBY TOKEN-CHANGE CONDITION")
    print(by_change.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nSaved:")
    for p in files.values():
        print(" ", p)
    print("\nOne-step fidelity only: the hook still runs the original block. No E2E speedup claimed.")


if __name__ == "__main__":
    main()
