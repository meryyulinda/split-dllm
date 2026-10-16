import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from configs.experiment_config import *

from src.model_loader import (
    load_dream_model
)

from src.generation import (
    generate_baseline,
    generate_with_token_history,
)

from src.instrumentation import (
    create_token_history_hook,
)

from src.utils import set_seed


model, tokenizer = load_dream_model(
    MODEL_NAME
)


set_seed(SEED)

baseline = generate_baseline(
    model,
    tokenizer,
    PROMPT,
    MAX_NEW_TOKENS,
    STEPS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    ALGORITHM,
)


set_seed(SEED)

hook, history = (
    create_token_history_hook()
)

instrumented = (
    generate_with_token_history(
        model,
        tokenizer,
        PROMPT,
        MAX_NEW_TOKENS,
        STEPS,
        TEMPERATURE,
        hook,
        TOP_P,
        TOP_K,
        ALGORITHM,
    )
)


same = (
    baseline["sequence"]
    .cpu()
    .equal(
        instrumented["sequence"]
        .cpu()
    )
)

print(
    "Baseline == instrumented:",
    same
)

print(
    "Captured states:",
    len(history)
)