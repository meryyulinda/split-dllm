import os
import sys
import pandas as pd
import torch

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from configs.experiment_config import(
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

from src.generation import (generate_baseline, generate_with_token_history,)

from src.instrumentation import (
    create_token_history_hook,
    count_token_changes,
    count_masks,
    changed_indices,
)

from src.utils import set_seed

def main():

    set_seed(SEED)

    model, tokenizer = load_dream_model(MODEL_NAME)

    hook, history = (create_token_history_hook())

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

    mask_token_id = (model.config.mask_token_id if model.config.mask_token_id is not None else tokenizer.mask_token_id)

    rows = []

    for step in range(1, len(history)):
        previous = history[step-1]
        current = history[step]

        (changed_mask, changed_count, changed_ratio,) = count_token_changes(previous, current,)

        masks_remaining = count_masks(current, mask_token_id,)

        indices = changed_indices(changed_mask)

        rows.append({
            "step":step,
            "changed_tokens": changed_count,
            "changed_ratio": changed_ratio,
            "masks_remaining": masks_remaining,
            "changed_indices": str(indices),
        })

        df = pd.DataFrame(rows)

        os.makedirs("outputs/exp1", exist_ok=True)
        
        df.to_csv("outputs/exp1/token_changes.csv", index=False)

    print("\nGenerated output:")
    print(result["text"])

    print("\nFinal token change statistics:")
    print(df[["step", "changed_tokens", "changed_ratio", "masks_remaining",]].to_string(index=False))

        

if __name__ == "__main__":
    main()