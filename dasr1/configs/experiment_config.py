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

# Experiment 8C
EXP8C_LAYERS = [8, 20, 24]
EXP8C_REFRESH_INTERVALS = [2, 4]
EXP8C_ADAPTIVE_INPUT_DRIFT_THRESHOLDS = [0.10]
EXP8C_ADAPTIVE_MAX_AGE = 4
EXP8C_REPEATS = 3
EXP8C_WARMUP_RUNS = 1

# Exp8D — measured multi-block residual caching (Dream)
# Block numbers are 1-based, matching prior Exp8C.
EXP8D_LAYER_GROUPS = [[20], [8, 20], [20, 24], [8, 20, 24]]
EXP8D_REFRESH_INTERVALS = [2, 4]
EXP8D_TASK_KEYS = ["reasoning"]  # Later: ["factual_qa", "reasoning", "instruction"]
EXP8D_SEEDS = [SEED]              # Later: [SEED, SEED + 1]
EXP8D_REPEATS = 5
EXP8D_WARMUP_RUNS = 1

# Experiment 8E — robustness of measured multi-block residual caching
EXP8E_LAYER_GROUPS = [
    [8, 20, 24],   # Best Exp8D group. Add [20] or [8, 20] for extra ablations.
]
EXP8E_REFRESH_INTERVALS = [2, 4, 8]
EXP8E_TASK_KEYS = ["factual_qa", "reasoning", "instruction"]
EXP8E_SEEDS = [SEED, SEED + 1, SEED + 2]
EXP8E_REPEATS = 2
EXP8E_WARMUP_RUNS = 1


# Exp8F — append to dasr1/configs/experiment_config.py
# Uses the same workload and synthetic long-prompt builder as Exp9B.
# Start with reasoning only; expand to other tasks AFTER validating mechanism.
EXP8F_GENERATION_CASES = [(64, 64), (256, 256)]
EXP8F_PROMPT_MODES = ["short", "long"]
EXP8F_LONG_PROMPT_TARGET_TOKENS = 1024
EXP8F_TASK_KEYS = ["reasoning"]  # later: ["factual_qa", "reasoning", "instruction"]
EXP8F_SEEDS = [SEED]
EXP8F_LAYER_GROUPS = [
    [8, 20, 24],                        # 3 blocks; Exp8E reference
    [4, 8, 12, 16, 20, 24],            # 6 blocks; nested
    [4, 8, 12, 16, 18, 20, 22, 24],    # 8 blocks; nested
]
EXP8F_REFRESH_INTERVALS = [2, 4]
EXP8F_REPEATS = 1
EXP8F_WARMUP_RUNS = 1
EXP8F_RUN_FIDELITY = True
EXP8F_VALIDATE_HF_TOGGLE = True


# Exp8G
# Reuses existing MODEL_NAME, SEED, RQ5_PROMPTS, MAX_NEW_TOKENS, STEPS,
# TEMPERATURE, TOP_P, TOP_K, ALGORITHM.
EXP8G_TASK_KEYS = ["factual_qa", "reasoning", "instruction"]
EXP8G_SEEDS = [SEED, SEED + 1, SEED + 2]
EXP8G_LAYER_GROUPS = [
    [8, 20, 24],
    [4, 8, 12, 16, 18, 20, 22, 24],
]
EXP8G_REFRESH_INTERVALS = [2, 4]
EXP8G_NEW_TOKENS = 64
EXP8G_STEPS = 64
EXP8G_PROMPT_MODES = ["short"]
EXP8G_WARMUP_RUNS = 1
EXP8G_VALIDATE_HF_TOGGLE = True

# Fidelity-first initial run (~36 cached generations + baseline controls).
# Once fidelity is reviewed, set to True for separate CUDA-synced B-C-B timing.
EXP8G_RUN_TIMING = False
EXP8G_TIMING_REPEATS = 2



# Exp9A
EXP9A_OFFICIAL_CACHE_REPO = "../../dLLM-cache-official"

EXP9A_PROMPT_INTERVAL_STEPS = [100]
EXP9A_GEN_INTERVAL_STEPS = [7]
EXP9A_TRANSFER_RATIOS = [0.25]

EXP9A_TASK_KEYS = [
    "factual_qa",
    "reasoning",
    "instruction",
]
EXP9A_SEEDS = [SEED, SEED + 1, SEED + 2]
EXP9A_REPEATS = 2
EXP9A_WARMUP_RUNS = 1

# Experiment 9B
EXP9B_GENERATION_CASES = [(64, 64), (256, 256)]

EXP9B_PROMPT_MODES = ["short", "long"]
EXP9B_LONG_PROMPT_TARGET_TOKENS = 1024

EXP9B_TASK_KEYS = [
    "factual_qa",
    "reasoning",
    "instruction",
]

EXP9B_SEEDS = [SEED]
EXP9B_REPEATS = 1
EXP9B_WARMUP_RUNS = 1

EXP9B_CACHE_CONFIGS = [(100, 7, 0.25)]

EXP9B_RUN_MODULE_PROBE = True
EXP9B_PROBE_TASK_KEY = "reasoning"
EXP9B_PROFILE_ONE_CASE = False

# Exp10A: Official Feature Cache Quality Robustness

EXP10A_SEEDS = [42, 43, 44]
EXP10A_NEW_TOKENS = 128
EXP10A_STEPS = 128

EXP10A_CACHE_CONFIG = (100, 7, 0.25)

EXP10A_WARMUP_RUNS = 1
EXP10A_RUN_TIMING = False
EXP10A_TIMING_REPEATS = 2

EXP10A_CASE_IDS = []  # Empty = run all 12 questions

# Exp10B: root-cause ablation (prompt-tail off-by-one x decoding order)
# Variants are defined in experiments/exp10b_prompt_tail_order_ablation.py;
# override with EXP10B_VARIANTS = [...] if needed.
EXP10B_SEEDS = [42, 43, 44]       # entropy + T=0 is deterministic -> uses seed 42 only
EXP10B_NEW_TOKENS = 128
EXP10B_STEPS = 128
EXP10B_TEMPERATURE = 0.0
EXP10B_CASE_IDS = []    # empty = all 12 Exp10A cases
EXP10B_WARMUP_RUNS = 1
EXP10B_RUN_TIMING = False

# Exp10C: rule-based adaptive refresh (run after Exp10B)
EXP10C_ALG = "entropy"            # set to "origin" if Exp10B does not support H2
EXP10C_PROMPT_TAIL_ROWS = 1          # set to 0 if Exp10B does not support H1
EXP10C_SEEDS = [42, 43, 44]
EXP10C_NEW_TOKENS = 128           # also try 256 / 256 (official eval length)
EXP10C_STEPS = 128
EXP10C_TEMPERATURE = 0.0
EXP10C_CASE_IDS = []
EXP10C_WARMUP_RUNS = 1
EXP10C_RUN_TIMING = False         # enable with a small EXP10C_CASE_IDS subset
EXP10C_TIMING_REPEATS = 2

# Exp10D: refresh policies on GSM8K (first 200 test problems in data/gsm8k/)
# Variants are defined in experiments/exp10d_gsm8k_refresh.py.
EXP10D_N_QUESTIONS = 100           # quick check first (~15 min); then 100 (~2.5 h)
EXP10D_START = 0
EXP10D_NEW_TOKENS = 256           # official dLLM-Cache GSM8K length for Dream
EXP10D_STEPS = 256
EXP10D_RESUME = True              # restart skips finished questions
EXP10D_WARMUP_RUNS = 1
EXP10D_RUN_TIMING = False

# Exp11: DASR prototype on two GPUs (edge = EXP11_EDGE_DEVICE, cloud = EXP11_CLOUD_DEVICE)
# Smoke test first: EXP11_N_QUESTIONS = 2, EXP11_NEW_TOKENS = EXP11_STEPS = 64 (~3 min).
# Output folder: outputs/exp11_<dataset>_k<split>_t<tokens>/
EXP11_DATASET = "gsm8k"           # "gsm8k" (256/256) or "authored" (Exp10A docs, 128/128)
EXP11_N_QUESTIONS = 2             # smoke test; then 30
EXP11_NEW_TOKENS = 64             # smoke test; then 256 (gsm8k) / 128 (authored)
EXP11_STEPS = 64
EXP11_SPLIT_LAYER = 4             # edge runs layers 1..4 (best static split in RQ4)
EXP11_EDGE_DEVICE = "cuda:1"    # = physical GPU 2 when launched with CUDA_VISIBLE_DEVICES=1,2
EXP11_CLOUD_DEVICE = "cuda:0"   # = physical GPU 1
EXP11_CACHE_POLICY = ("fixed", 25, 2, 0.25)   # official dLLM-Cache GSM8K setting
EXP11_ADAPTIVE_POLICY = ("adaptive", {"gen_content_commits": 2, "gen_max_age": 4,
                        "prompt_content_commits": 16, "prompt_max_age": 32,
                        "eos_frontier_window": 8, "transfer_ratio": 0.25})
EXP11_ANCHOR_INTERVAL = 32        # INT8 codec: bf16 anchor at least every 32 steps
EXP11_RESUME = True

# Exp11b: network emulation grid (no GPU)
EXP11B_BANDWIDTHS_MBPS = [1, 5, 10, 25, 50, 100, 1000]
EXP11B_RTTS_MS = [5, 20, 50, 100]
EXP11B_EDGE_SLOWDOWNS = [1.0, 3.0]  # 3.0 ~ RTX 3060-class edge vs the measured A6000
EXP10D_TEMPERATURE = 0.1          # Dream's official GSM8K eval: T=0.1, top_p=0.9 (T=0 collapses to EOS)
EXP10D_TOP_P = 0.9
EXP10D_SEEDS = [42]               # with T > 0, every seed is a separate paired run
EXP11_TEMPERATURE = 0.1           # Dream's official GSM8K sampling (T=0 collapses to EOS)
EXP11_TOP_P = 0.9

