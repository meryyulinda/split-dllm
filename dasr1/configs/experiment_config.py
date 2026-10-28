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

SPLIT_LAYER = 12

SPLIT_LAYERS = [4, 8, 12, 16, 20, 24]

WARMUP = 3
REPEATS = 10

EDGE_DEVICE = "cuda:2"
CLOUD_DEVICE = "cuda:0"

SELECTIVE_THRESHOLDS = [
    1e-4,
    5e-4,
    1e-3,
    5e-3,
    1e-2,
]

SELECTIVE_TIMING_THRESHOLDS = [
    1e-3,   # exact / conservative reference
    5e-3,   # intermediate
    1e-2,   # aggressive
]

SCHEDULER_THRESHOLD = 1e-3

SCHEDULER_SWITCH_OVERHEADS_MS = [
    0.0,
    0.1,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    2.0,
    5.0,
    10.0,
]

RQ5_SPLIT_LAYERS = [
    4,    # early
    12,   # middle
    24,   # late
]

RQ5_THRESHOLD = 1e-3

RQ5_PROMPTS = {
    "factual_qa":
        "What is the capital of Australia?",

    "reasoning":
         "What is 15 multiplied by 12? Explain briefly.",

    "instruction":
        "Explain in three short sentences why regular backups are important."
}

RQ5_LAYERS = [4, 8, 12, 16, 20, 24, 28]