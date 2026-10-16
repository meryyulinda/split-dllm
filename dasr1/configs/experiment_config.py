MODEL_NAME = "Dream-org/Dream-v0-Instruct-7B"

SEED = 42

MAX_NEW_TOKENS = 64

# Start small while debugging.
STEPS = 64

TEMPERATURE = 0.0

TOP_P = None
TOP_K = None

ALGORITHM = "origin"

PROMPT = "What is 15 multiplied by 12? Explain briefly."

SELECTED_LAYERS = [
    4,
    8,
    12,
    16,
    20,
    24,
    28,
]

DRIFT_EPSILON = 1e-8