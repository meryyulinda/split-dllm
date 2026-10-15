import os
import sys
import json

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

from src.generation import generate_baseline

from src.utils import (set_seed, print_gpu_info)

def main():

    print("=" * 60)
    print("EXPERIMENT 0: BASELINE DREAM")
    print("=" * 60)

    print_gpu_info()

    set_seed(SEED)

    model, tokenizer = load_dream_model(MODEL_NAME)

    result = generate_baseline(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        steps=STEPS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        alg=ALGORITHM,
    )

    print("\nPROMPT")
    print(PROMPT)

    print("\nOUTPUT")
    print(result["text"])

    os.makedirs("outputs/exp0", exist_ok=True)

    metadata = {
        "model": MODEL_NAME,
        "seed": SEED,
        "prompt": PROMPT,
        "steps": STEPS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "algorithm": ALGORITHM,
        "output": result["text"],
    }

    with open("outputs/exp0/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()