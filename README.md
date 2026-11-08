# Diffusion-Aware Split Inference for dLLMs

This repository investigates **efficient edge–cloud inference for diffusion large language models (dLLMs)** through split inference, delta communication, temporal caching, and adaptive scheduling.

**Models:**
- Dream-v0-Instruct-7B (primary model)
- LLaDA-8B-Instruct (cross-model validation)

**Current progress:** Experiments 0–9B, including split inference, communication optimization, block caching, and feature caching.

## 1. Research Questions and Progress

| RQ | Focus | Status | Key Findings |
|---|---|---|---|
| RQ1 | Token changes and hidden-state drift across denoising steps and layers | Completed | Sparse token changes induce dense hidden-state updates. |
| RQ2 | Exact synchronized-delta communication | Partially answered | 30.69% fewer boundary bytes with unchanged outputs, but no E2E speedup on the current local two-GPU setup. |
| RQ3 | Selective recomputation and temporal caching | Partially answered | Row-level reuse is limited. Block and feature caching reduce actual computation but may alter generation quality. |
| RQ4 | Adaptive scheduling | Partially answered | Analytical split scheduling yields at most 0.26% improvement. Adaptive cache-mode selection is not yet implemented. |
| RQ5 | Cross-model and cross-task generalization | Partially answered | Dense hidden-state propagation is observed in Dream and LLaDA; cross-model split/caching validation remains pending. |

## 2. Experiments and Key Results

### Exp0–7D: Split Inference and Initial RQs

| Experiments | Focus | Main Findings |
|---|---|---|
| Exp0–2C | Baseline, token changes, hidden-state drift | Dream has 44/64 changing transitions; hidden-state updates are broadly dense when tokens change. |
| Exp3A–3D | Static split correctness and latency | Tested split layers preserve baseline outputs. |
| Exp4A–4B | Physical two-GPU split and exact deltas | 30.69% communication reduction with exact outputs, but no measured E2E improvement. |
| Exp5A–5B | Selective hidden-state reuse | ~31.25% reuse at threshold 1e-3, mostly from unchanged transitions; compute savings remain idealized. |
| Exp6 | Adaptive split scheduling | Maximum 0.26% estimated improvement at zero switching overhead. |
| Exp7A–7D | Dream/LLaDA cross-task validation | Both models show widespread hidden-state changes despite sparse token updates. |

### Exp8A–8G: Temporal Block-Residual Caching

Block caching reuses cached transformer-block residuals across denoising steps, skipping selected block executions between refreshes.

| Experiments | Focus | Main Findings |
|---|---|---|
| Exp8A–8B | Block drift and one-step residual reuse | Residual reuse can preserve many token predictions despite substantial whole-block drift. |
| Exp8C–8D | Actual single-/multi-block caching | Up to 7.42% measured E2E latency saving with three cached blocks. |
| Exp8E | Cross-task and cross-seed robustness | Up to 8.57% latency saving; 32/54 exact trajectories. |
| Exp8F | Scaling to 3, 6, and 8 cached blocks | Up to 19.53% measured latency saving; 24/24 exact trajectories in one reasoning task/seed. |
| Exp8G | Cross-task/seed fidelity and quality | 22/36 exact trajectories; some approximations introduce incorrect or degraded answers. |

**Finding:** Block caching can provide meaningful computation savings, but fixed caching policies are not consistently quality-preserving.

### Exp9A–9B: Official dLLM-Cache Feature Caching

Feature caching reuses prompt features and selectively updates response features across denoising steps.

| Experiment | Focus | Main Findings |
|---|---|---|
| Exp9A | Short prompts, 64 tokens / 64 steps | ~0% latency saving; 8/18 exact trajectories. |
| Exp9B | Short/long prompts, 64/256 tokens and steps | Up to 72.69% measured latency saving on synthetic long-context workloads; 8/12 exact final outputs. |

**Exp9B measured E2E latency reduction:**

| Input prompt | Generation | Time Saving |
|---|---|---:|
| Short (~32 tokens) | 64 tokens / 64 steps | -0.36% |
| Short (~32 tokens) | 256 tokens / 256 steps | 39.62% |
| Long (~1,056 tokens) | 64 tokens / 64 steps | 72.45% |
| Long (~1,056 tokens) | 256 tokens / 256 steps | 72.69% |

**Finding:** Feature caching is especially effective on larger workloads. Preliminary output inspection found generally acceptable answers, but quality robustness still requires broader validation.

**Measurement limitation:** Exp8–9 report single-GPU generation latency, not integrated edge–cloud inference latency. Block and feature caching also use different HF KV-cache settings and have not yet been benchmarked under fully matched conditions.

---

## 3. Quick Reproduction Guide

### Prerequisites

- Linux with a **CUDA-enabled NVIDIA GPU** and sufficient VRAM for 7B–8B models.
- Conda and Git.
- **Two GPUs required** for physical split inference experiments (Exp4A–4B).

### Setup

Clone the repository:

```bash
git clone https://github.com/meryyulinda/split-dllm.git
cd split-dllm/dasr1
```

**Dream is the main environment**, used for almost all experiments (Exp0–6, Exp7D, Exp8A–9B).

```bash
conda create -n dream python=3.11 -y
conda activate dream
pip install -r requirements_dream.txt
```

A separate **LLaDA environment** is required only for Exp7A–7C (RQ5 cross-model experiments):

```bash
conda create -n llada python=3.11 -y
conda activate llada
pip install -r requirements_llada.txt
```

| Environment | Experiments |
|---|---|
| **`dream` (main)** | Exp0–6, Exp7D, Exp8A–8G, Exp9A–9B |
| `llada` | Exp7A–7C only |

Verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Models and External Dependencies

Model checkpoints are automatically downloaded from Hugging Face on first use:

- [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)
- [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)

For Exp9A–9B, clone the official dLLM-Cache repository separately:

```bash
git clone https://github.com/maomaocun/dLLM-cache.git ../../dLLM-cache-official
```

Set the repository path in `configs/experiment_config.py`:

```python
EXP9A_OFFICIAL_CACHE_REPO = "../../dLLM-cache-official"
```

### Run Experiments

Configure model parameters and GPU devices in `configs/experiment_config.py`. Run all scripts from the `dasr1/` directory.

**The following commands are examples only**, illustrating how to run experiments under each environment. To reproduce other experiments, replace the script name with the corresponding Python file in `experiments/`.

**Dream experiments (main environment):**

```bash
conda activate dream

# Example: RQ1 hidden-state drift
python experiments/exp2_hidden_drift.py

# Example: RQ2 exact delta (requires two GPUs)
python experiments/exp4b_synchronized_delta.py

# Example: RQ5 Dream cross-task validation
python experiments/exp7d_dream_cross_task_hidden_drift.py

# Example: block caching
python experiments/exp8g_block_cache_robustness.py

# Example: feature caching
python experiments/exp9b_official_feature_cache_diagnostics.py
```

**LLaDA experiments (switch environment for Exp7A–7C only):**

```bash
conda activate llada

python experiments/exp7a_llada_token_history.py
python experiments/exp7b_inspect_llada.py
python experiments/exp7c_llada_hidden_drift.py
```

Switch back to the main environment afterward:

```bash
conda activate dream
```

**General command for any experiment:**

```bash
python experiments/<experiment_script>.py
```

Results are saved under `outputs/exp*/`.

---

## 4. Next Steps

1. **Quality validation:** Evaluate feature and block caching on more diverse prompts, tasks, and seeds using correctness and answer-quality metrics.
2. **Adaptive cache-mode scheduling:** Explore a lightweight scheduler selecting full recomputation, block caching, or feature caching based on workload, estimated latency, and quality risk.
3. **Two-GPU integration:** Combine caching with physical split inference and full/exact-delta boundary transmission.
4. **Joint performance evaluation:** Measure computation time, communication bytes, transmission latency, cache overhead, switching overhead, and E2E latency under varied bandwidth conditions.
5. **Cross-model validation:** Extend integrated caching and communication optimizations to LLaDA.

## 5. Current Research Direction

The long-term objective is a **diffusion-aware split inference runtime** that jointly considers:

- Split location between edge and cloud
- Boundary communication policy (full or exact delta)
- Temporal computation reuse (block caching, feature caching, or full recomputation)
- Quality constraints and scheduling overhead

**Adaptive cache-mode selection and its integration into physical two-GPU split inference remain future work.**