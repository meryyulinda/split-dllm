# Split dLLM Inference for Edge–Cloud Systems

This repository contains an experimental prototype for studying **split inference, communication reduction, selective reuse, and adaptive split scheduling in diffusion large language models (dLLMs)**.

The current implementation focuses primarily on **Dream-v0-Instruct-7B** and uses **LLaDA-8B-Instruct** for cross-model and cross-task validation.

The project is organized around five research questions:

- **RQ1 — Hidden-state dynamics:** How do token changes and hidden-state drift vary across denoising steps and model layers?
- **RQ2 — Exact synchronized deltas:** Can synchronized deltas reduce wire bytes and latency without changing outputs in the exact tier?
- **RQ3 — Selective recomputation:** At what active-set or refresh thresholds can selective reuse reduce computation while preserving output quality?
- **RQ4 — Adaptive scheduling:** When does adaptive split-point scheduling outperform a static split after switching overhead is included?
- **RQ5 — Transferability:** Do the conclusions transfer across multiple public dLLMs and multiple task types?

---

## 1. Project Overview

Diffusion LLMs generate text through repeated denoising steps rather than standard left-to-right autoregressive decoding. This creates temporal redundancy across consecutive denoising states and motivates several possible optimizations for edge–cloud inference:

1. **Split inference** — execute early layers on an edge GPU and later layers on a cloud GPU.
2. **Exact delta communication** — send only changed boundary rows instead of retransmitting the full split boundary.
3. **Selective hidden-state reuse** — reuse previous hidden rows when their normalized drift is below a threshold.
4. **Adaptive split scheduling** — change the split layer across denoising transitions when doing so lowers total cost.

A major empirical finding of this project is that **sparse token updates do not necessarily imply sparse hidden-state updates**. Once at least one token changes, bidirectional attention can propagate that change to nearly every generated hidden row.

---

## 2. Models

### 2.1 Dream

Primary model:

```text
Dream-org/Dream-v0-Instruct-7B
```

Main experiment configuration:

```text
SEED = 42
MAX_NEW_TOKENS = 64
STEPS = 64
TEMPERATURE = 0.0
TOP_P = None
TOP_K = None
ALGORITHM = "origin"
```

Representative hidden layers:

```text
[4, 8, 12, 16, 20, 24, 28]
```

Representative split layers:

```text
[4, 8, 12, 16, 20, 24]
```

Current physical two-GPU configuration:

```text
EDGE_DEVICE = "cuda:2"
CLOUD_DEVICE = "cuda:0"
```

### 2.2 LLaDA

Cross-model validation model:

```text
GSAI-ML/LLaDA-8B-Instruct
```

For RQ5, LLaDA is evaluated using the same:

```text
64 generated positions
64 denoising steps
```

with one diffusion block and `low_confidence` remasking.

---

## 3. Installation

Dream and LLaDA currently use **separate Python environments** because their compatible `transformers` versions differ.

### 3.1 Dream environment

A working Dream environment used during development:

```text
Python       3.11.x
PyTorch      2.5.1+cu124
Transformers 4.46.2
```

Example:

```bash
conda create -n dream-split python=3.11 -y
conda activate dream-split

pip install torch==2.5.1
pip install transformers==4.46.2
pip install accelerate pandas numpy sentencepiece
```

Use the PyTorch/CUDA build that matches the target server.

### 3.2 LLaDA environment

LLaDA is tested in a separate environment:

```bash
conda create -n llada-rq5 python=3.11 -y
conda activate llada-rq5

pip install torch==2.5.1
pip install transformers==4.38.2
pip install accelerate pandas numpy sentencepiece
```

Clone the official LLaDA repository when its generation implementation is needed:

```bash
git clone https://github.com/ML-GSAI/LLaDA.git
```

### 3.3 Model download

Models are downloaded from Hugging Face on first use.

Example:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Dream-org/Dream-v0-Instruct-7B",
    trust_remote_code=True,
)
```

and:

```python
tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct",
    trust_remote_code=True,
)
```

---

## 4. File Structure

```text
dasr1/
├── configs/
│   └── experiment_config.py
│
├── src/
│   ├── model_loader.py
│   ├── generation.py
│   ├── instrumentation.py
│   ├── hidden_state_tracker.py
│   ├── split_model.py
│   ├── selective_recompute.py
│   ├── llada_generation.py
│   ├── llada_hidden_state_tracker.py
│   └── utils.py
│
├── experiments/
│   ├── exp0_baseline.py
│   ├── exp1_plot.py
│   ├── exp1_token_changes.py
│   ├── validate_exp1.py
│   ├── exp2_hidden_drift.py
│   ├── exp2c_token_vs_hidden.py
│   ├── exp3a_static_split.py
│   ├── exp3b_static_split_trajectory.py
│   ├── exp3c_split_sweep.py
│   ├── exp3d_stage_latency.py
│   ├── exp4a_two_gpu_static_split.py
│   ├── exp4b_synchronized_delta.py
│   ├── exp5a_selective_reuse.py
│   ├── exp5b_selective_compute_estimate.py
│   ├── exp6_adaptive_scheduler.py
│   ├── exp7a_llada_token_history.py
│   ├── exp7b_inspect_llada.py
│   ├── exp7c_llada_hidden_drift.py
│   └── exp7d_dream_cross_task_hidden_drift.py
│
├── outputs/
│   ├── exp0/
│   ├── exp1/
│   ├── exp2/
│   ├── exp3/
│   ├── exp4/
│   ├── exp5/
│   ├── exp6/
│   ├── exp7c/
│   └── exp7d/
│
└── README.md
```

### 4.1 Experiment files and roles

| File | Purpose | Related RQ |
|---|---|---|
| `exp0_baseline.py` | Baseline Dream generation and runtime/output validation | Setup |
| `exp1_token_changes.py` | Capture token trajectory and measure token changes across denoising transitions | RQ1 |
| `exp1_plot.py` | Plot token-change behavior | RQ1 |
| `validate_exp1.py` | Validate trajectory/state-count assumptions | RQ1 validation |
| `exp2_hidden_drift.py` | Capture hidden states and compute row-wise normalized drift across selected layers | RQ1 |
| `exp2c_token_vs_hidden.py` | Compare token sparsity with hidden-state activity | RQ1 |
| `exp3a_static_split.py` | Validate logical static split correctness at a representative split | RQ2 foundation |
| `exp3b_static_split_trajectory.py` | Validate split correctness across the denoising trajectory | RQ2 foundation |
| `exp3c_split_sweep.py` | Sweep candidate split layers | RQ2 foundation |
| `exp3d_stage_latency.py` | Measure Stage-1 and Stage-2 latency fractions | RQ2 foundation |
| `exp4a_two_gpu_static_split.py` | Physical two-GPU full-boundary split inference | RQ2 |
| `exp4b_synchronized_delta.py` | Exact synchronized-delta communication | RQ2 |
| `exp5a_selective_reuse.py` | Semantic/oracle selective boundary reuse under multiple thresholds | RQ3 |
| `exp5b_selective_compute_estimate.py` | Derive idealized selective-compute latency from measured dense timing | RQ3 |
| `exp6_adaptive_scheduler.py` | Dynamic-programming split scheduler with switching-overhead sweep | RQ4 |
| `exp7a_llada_token_history.py` | Validate LLaDA token trajectory and state-count convention | RQ5 |
| `exp7b_inspect_llada.py` | Inspect LLaDA model structure | RQ5 setup |
| `exp7c_llada_hidden_drift.py` | Cross-task LLaDA hidden-state validation | RQ5 |
| `exp7d_dream_cross_task_hidden_drift.py` | Cross-task Dream hidden-state validation using the same RQ5 prompts | RQ5 |

---

## 5. Important Source Files

### `configs/experiment_config.py`

Central configuration for shared experiment parameters.

Important values include:

```text
MODEL_NAME
SEED
MAX_NEW_TOKENS
STEPS
TEMPERATURE
TOP_P
TOP_K
ALGORITHM

SELECTED_LAYERS
SPLIT_LAYER
SPLIT_LAYERS

EDGE_DEVICE
CLOUD_DEVICE

SELECTIVE_THRESHOLDS
SELECTIVE_TIMING_THRESHOLDS

SCHEDULER_THRESHOLD
SCHEDULER_SWITCH_OVERHEADS_MS

RQ5_SPLIT_LAYERS
RQ5_THRESHOLD
RQ5_PROMPTS
RQ5_LAYERS
```

Experiment scripts should import shared values from this file instead of hardcoding them.

### `src/model_loader.py`

Contains model-loading helpers, including:

```python
load_dream_model(model_name)
```

### `src/generation.py`

Contains Dream prompt preparation and generation helpers.

Examples:

```text
prepare_prompt(...)
generate_baseline(...)
generate_with_token_history(...)
```

### `src/instrumentation.py`

Contains token-history instrumentation for capturing the Dream denoising trajectory.

Canonical Dream trajectory convention:

```text
state 0  = initial pre-denoising state
state 1  = after first denoising update
...
state 64 = final state
```

Therefore:

```text
65 token states
64 consecutive token transitions
```

### `src/hidden_state_tracker.py`

Captures selected Dream transformer-layer outputs and computes row-wise normalized L2 drift:

```text
||h_t - h_(t-1)||_2
--------------------
    ||h_t||_2 + eps
```

It also reports active-row ratios for:

```text
1e-4
5e-4
1e-3
5e-3
1e-2
```

### `src/split_model.py`

Implements split-inference helpers.

Current functionality includes:

- logical same-GPU split forward
- physical two-GPU model placement
- full-boundary two-GPU inference
- exact synchronized-delta inference
- Stage-1 / transfer / Stage-2 timing
- boundary-byte accounting

### `src/selective_recompute.py`

Implements semantic/oracle selective boundary reuse.

Inactive rows reuse the previous exact boundary while active rows use the current exact boundary.

This is currently used for correctness and analytical feasibility rather than actual sparse GPU execution.

### `src/llada_generation.py`

LLaDA generation extended with token-history capture.

For the current `64 steps / 64 generated positions` setting:

```text
65 states
64 transitions
```

### `src/llada_hidden_state_tracker.py`

Captures selected LLaDA transformer-block outputs and computes normalized row-wise hidden-state drift.

---

## 6. Experiment Script Convention

Every experiment script under `experiments/` should add the project root before importing project modules:

```python
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
```

Run experiments from the project root:

```bash
python experiments/<experiment_name>.py
```

---

## 7. Experimental Workflow

### RQ1 — Token Changes and Hidden-State Drift

1. Generate 64 tokens with 64 denoising steps.
2. Capture the initial token state and every denoising update.
3. Obtain:

```text
65 token states
64 token transitions
```

4. Hook selected hidden layers.
5. Dream performs 64 generation forwards, corresponding to hidden representations of states 0–63.
6. Perform one additional **analysis-only final forward** on state 64.
7. Compute normalized row-wise hidden drift.
8. Compare token-change ratio against active hidden-row ratio.

The final analysis-only forward must **not** be included in latency measurements.

### RQ2 — Static Split and Exact Delta Communication

The model is partitioned into:

```text
Stage 1: edge GPU
boundary transfer
Stage 2: cloud GPU
```

Candidate splits:

```text
4, 8, 12, 16, 20, 24
```

Correctness checks include:

```text
torch.allclose
same argmax
```

Exact synchronized delta works as follows:

```text
first state:
    full boundary synchronization

later states:
    changed row indices
    +
    changed row values
```

The cloud stores the previous exact boundary and patches changed rows.

### RQ3 — Selective Reuse

For each boundary row:

```text
if normalized_drift > threshold:
    use current row
else:
    reuse previous row
```

Thresholds:

```text
1e-4
5e-4
1e-3
5e-3
1e-2
```

The current implementation is an oracle/semantic prototype because the current boundary is still fully computed before reuse is applied.

Therefore selective timing is derived using:

```text
ideal_selective_stage2
    = active_ratio × measured_full_stage2
```

This is an idealized upper-bound estimate, not measured sparse-kernel execution.

### RQ4 — Adaptive Split Scheduling

RQ4 evaluates whether changing the split layer across transitions can outperform a fixed split.

The scheduler uses the per-transition selective-reuse cost model from RQ3.

Dynamic programming:

```text
DP[t, s]
    = execution_cost[t, s]
    + min(
        DP[t-1, previous_split]
        + switching_cost
      )
```

The scheduler currently adapts **only the split location**.

The selective-reuse strategy and threshold remain fixed.

Full-boundary, exact-delta, and selective execution are not jointly selected because:

- full-boundary E2E is measured
- exact-delta E2E is measured
- selective-reuse E2E is currently idealized

Jointly optimizing them would mix measured and estimated costs.

### RQ5 — Cross-Model and Cross-Task Validation

Current shared prompts:

```python
RQ5_PROMPTS = {
    "factual_qa":
        "What is the capital of Australia?",

    "reasoning":
        "What is 15 multiplied by 12? Explain briefly.",

    "instruction":
        "Explain in three short sentences why regular backups are important."
}
```

Models:

```text
Dream-v0-Instruct-7B
LLaDA-8B-Instruct
```

Current RQ5 validation covers:

- token-change sparsity
- hidden-state propagation
- task dependence
- model dependence

Full cross-model validation of RQ2–RQ4 is still pending.

---

## 8. Main Experimental Results

### 8.1 RQ1 — Dream Hidden-State Dynamics

For the original Dream reasoning workload:

```text
64 transitions
44 transitions with token changes
20 fully unchanged transitions
```

At Layer 12 with threshold `1e-3`:

```text
token change present:
    active hidden ratio = 1.0

no token change:
    active hidden ratio = 0.0
```

Layer-wise averages:

| Layer | Mean Drift | Median Drift | Mean Active Ratio |
|---:|---:|---:|---:|
| 4  | 0.0903 | 0.0287 | 0.6875 |
| 8  | 0.3243 | 0.0925 | 0.6875 |
| 12 | 0.4012 | 0.1268 | 0.6875 |
| 16 | 0.4082 | 0.1529 | 0.6875 |
| 20 | 0.4011 | 0.1814 | 0.6875 |
| 24 | 0.3408 | 0.1859 | 0.6875 |
| 28 | 0.2938 | 0.2164 | 0.6875 |

Main result:

> Sparse token changes do not translate into sparse hidden-state changes. When at least one generated token changes, hidden-state drift propagates broadly across generated positions.

### 8.2 RQ2 — Static Split

Across tested splits:

```text
allclose = True
same argmax = True
```

Average boundary size:

```text
~0.6631 MB/state
```

Across 65 states:

```text
~43.10 MB
```

Latency follows the expected split trade-off:

```text
early split:
    shorter Stage 1
    longer Stage 2

late split:
    longer Stage 1
    shorter Stage 2
```

### 8.3 RQ2 — Exact Synchronized Delta

Dream trajectory:

```text
Full-boundary communication:
    ~43.10 MB

Exact-delta communication:
    ~29.87 MB
```

Reduction:

```text
~30.69% including initial full synchronization
~31.17% excluding initial full synchronization
```

Correctness:

```text
boundary exact
final argmax unchanged
```

However, exact delta does not reduce E2E latency in the current same-server two-GPU setup because delta encoding overhead is larger than the saved transfer time.

Measured full-boundary transfer is already approximately:

```text
0.067–0.070 ms/state
```

### 8.4 RQ3 — Selective Reuse

At:

```text
tau = 1e-3
```

Dream obtains approximately:

```text
active ratio = 68.75%
reuse ratio = 31.25%
```

However, the 31.25% reuse comes mainly from:

```text
20 / 64 fully unchanged transitions
```

not from fine-grained row reuse inside active transitions.

For the 44 transitions with token changes:

```text
nearly/all generated rows are active
```

At higher thresholds:

```text
tau = 5e-3:
    almost no additional reuse

tau = 1e-2:
    more reuse at some splits
    but output fidelity begins to decrease
```

### 8.5 RQ3 — Idealized Selective Timing

Because no sparse CUDA/Triton kernel is implemented yet, selective compute time is estimated using proportional scaling.

At `tau = 1e-3`:

| Split | Ideal Stage-2 Reduction | Ideal E2E Reduction |
|---:|---:|---:|
| 4  | 31.25% | 25.84% |
| 8  | 31.25% | 20.98% |
| 12 | 31.25% | 16.46% |
| 16 | 31.25% | 12.20% |
| 20 | 31.25% | 8.71% |
| 24 | 31.25% | 4.99% |

The same percentage Stage-2 reuse produces less E2E benefit at deeper splits because the cloud suffix is shorter.

### 8.6 RQ4 — Adaptive Split Scheduling

Best static split under the current selective-reuse cost model:

```text
Layer 4
Total estimated latency: ~1477.97 ms
Average latency: ~23.09 ms/transition
```

Adaptive scheduler:

| Switch Cost | Adaptive Total | Switches | Improvement |
|---:|---:|---:|---:|
| 0.0 ms | 1474.13 ms | 8 | 0.260% |
| 0.1 ms | 1474.93 ms | 8 | 0.206% |
| 0.5 ms | 1477.41 ms | 4 | 0.038% |
| 0.6 ms | 1477.81 ms | 2 | 0.011% |
| >= 0.7 ms | 1477.97 ms | 0 | 0.000% |

Main result:

> Adaptive split scheduling provides only marginal improvement for this workload because split layer 4 is already optimal for nearly the entire trajectory.

### 8.7 RQ5 — Dream Cross-Task Validation

For all three evaluated Dream tasks:

```text
44 / 64 transitions contain token changes
20 / 64 transitions are fully unchanged
```

For factual QA and reasoning:

```text
token change present:
    active hidden ratio = 1.0

token unchanged:
    active hidden ratio = 0.0
```

Instruction following is almost identical:

```text
mean active ratio when tokens change ≈ 0.9993
```

The hidden-propagation result is therefore stable across the tested Dream tasks.

### 8.8 RQ5 — LLaDA Cross-Task Validation

With `64 steps / 64 generated positions`, LLaDA changes exactly one generated token per transition:

```text
changed-token ratio = 1 / 64 = 0.015625
```

Across factual QA, reasoning, and instruction-following tasks:

```text
mean active hidden ratio = 1.0
minimum active ratio = 1.0
maximum active ratio = 1.0
```

At Layer 12:

```text
64 / 64 transitions fully active
```

Main cross-model result:

> Even when only one generated token changes, hidden-state changes propagate across essentially the entire generated region.

---

## 9. Current Interpretation

The project currently supports the following conclusions:

1. Token updates can be sparse.
2. Hidden-state updates are often dense once any token changes.
3. Exact deltas can reduce communication volume while preserving exact output.
4. Communication savings do not automatically become latency savings.
5. Safe fine-grained selective reuse is limited during active transitions.
6. Fully unchanged transitions can provide useful reuse, but availability depends on the model and trajectory.
7. Adaptive split scheduling provides little benefit when one split dominates most transitions.

The strongest general result is:

> **Sparse token updates are not a reliable proxy for sparse hidden-state computation in diffusion LLMs.**

---

## 10. Existing Issues and Limitations

### 10.1 No sparse CUDA/Triton selective kernel

The current selective-reuse implementation does not physically skip inactive rows in GPU kernels.

Standard PyTorch kernels still execute dense tensor operations.

Therefore:

```text
selective correctness/reuse:
    measured

selective sparse runtime:
    not measured

selective timing:
    idealized/derived
```

### 10.2 Bidirectional attention limits naive row-selective execution

Active query rows can still depend on all key/value positions.

Therefore, simply slicing active rows through a standard transformer block is not guaranteed to reproduce full-model computation.

### 10.3 Same-server two-GPU setup is not a realistic edge–cloud network

The physical transfer latency is very small.

Future work should evaluate:

- constrained bandwidth
- higher network latency
- PCIe
- Ethernet
- Wi-Fi
- WAN-style links
- realistic edge hardware

### 10.4 Adaptive switching overhead is hypothetical

Runtime model repartitioning/migration is not implemented.

RQ4 therefore sweeps assumed switching overhead rather than measuring actual repartitioning cost.

### 10.5 RQ4 adapts only the split layer

The current scheduler does not jointly select:

```text
full boundary
exact delta
selective reuse
```

because full-boundary and delta latency are measured while selective-reuse latency is still estimated.

### 10.6 LLaDA RQ2–RQ4 validation is incomplete

LLaDA currently validates:

- token-change behavior
- hidden-state propagation
- cross-task consistency

Still pending:

- physical split inference
- exact-delta communication
- selective-reuse cost analysis
- adaptive split scheduling

Representative future RQ5 splits:

```text
4, 12, 24
```

### 10.7 Limited task and prompt coverage

RQ5 currently uses three representative prompts:

```text
factual QA
short arithmetic reasoning
instruction following
```

More prompts or benchmark datasets are needed for stronger statistical generalization.

### 10.8 Fixed generation setting

Most experiments use:

```text
64 generated tokens
64 denoising steps
```

Future experiments should evaluate different sequence lengths and denoising schedules.

---

## 11. Recommended Experiment Order

```text
1.  exp0_baseline.py
    Baseline Dream generation and environment sanity check.

2.  exp1_token_changes.py
    Capture the 65 token states and 64 token transitions.

3.  validate_exp1.py
    Validate trajectory-count assumptions.

4.  exp1_plot.py
    Visualize token-change behavior.

5.  exp2_hidden_drift.py
    Measure hidden-state drift across selected layers.

6.  exp2c_token_vs_hidden.py
    Compare token sparsity against hidden-state activity.

7.  exp3a_static_split.py
    Validate one logical static split.

8.  exp3b_static_split_trajectory.py
    Validate split correctness across the full trajectory.

9.  exp3c_split_sweep.py
    Sweep candidate split points.

10. exp3d_stage_latency.py
    Profile Stage-1 and Stage-2 latency.

11. exp4a_two_gpu_static_split.py
    Run physical two-GPU full-boundary split inference.

12. exp4b_synchronized_delta.py
    Run exact synchronized-delta communication.

13. exp5a_selective_reuse.py
    Evaluate selective reuse and fidelity across thresholds.

14. exp5b_selective_compute_estimate.py
    Derive idealized selective-compute latency.

15. exp6_adaptive_scheduler.py
    Evaluate adaptive split scheduling.

16. exp7a_llada_token_history.py
    Validate LLaDA token trajectory.

17. exp7b_inspect_llada.py
    Inspect LLaDA layer structure.

18. exp7c_llada_hidden_drift.py
    Run LLaDA cross-task hidden-drift validation.

19. exp7d_dream_cross_task_hidden_drift.py
    Run Dream cross-task validation with the same RQ5 prompts.
```

Notes:

- Run Dream experiments in the Dream-compatible environment.
- Run LLaDA experiments in the LLaDA-compatible environment.
- Shared parameters should be imported from `configs/experiment_config.py`.
- Avoid hardcoding shared experiment settings inside experiment scripts.

---

## 12. Current Project Status

| RQ | Status | Notes |
|---|---|---|
| RQ1 | Completed | Dream + cross-model/cross-task hidden-propagation validation |
| RQ2 | Completed for Dream | LLaDA validation pending |
| RQ3 | Prototype completed for Dream | Sparse CUDA/Triton runtime and LLaDA validation pending |
| RQ4 | Analytical scheduler completed for Dream | LLaDA validation and measured switching cost pending |
| RQ5 | Partially completed | Dream + LLaDA hidden propagation validated; full RQ2–RQ4 transferability pending |

---

## 13. Reproducibility Notes

### Dream

Verify the active environment:

```bash
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available())"
```

Working development configuration:

```text
Python       3.11.x
PyTorch      2.5.1+cu124
Transformers 4.46.2
CUDA         available
```

### LLaDA

Use a separate LLaDA-compatible environment.

Do not run Dream from the LLaDA environment if its `transformers` version is too old for Dream's dynamic model code.

---

## 14. Main Takeaway

The central empirical result of the project is:

> **Sparse token updates are not a reliable proxy for sparse hidden-state computation in diffusion LLMs.**

Across Dream and LLaDA, local token changes can propagate broadly through hidden representations.

Therefore, reuse-aware edge–cloud dLLM inference should reason directly about hidden-state dynamics, communication costs, and scheduling overhead rather than assuming that sparse token updates automatically imply sparse reusable computation.
