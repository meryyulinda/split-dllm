# DASR: Diffusion-Aware Split Computing for dLLMs

Code and results for running a **diffusion large language model (dLLM) split between an edge device and a cloud server**, without retraining. The edge runs the first layers, the cloud runs the rest and the sampler, and the hidden state at the split has to cross the network at every denoising step.

**Idea.** dLLM feature caching ([dLLM-Cache](https://github.com/maomaocun/dLLM-cache)) is run on both sides of the split. Between cache refreshes the cloud takes everything it needs about the prompt from its own cache, so the prompt rows of the hidden state only have to be sent at refresh steps. The answer is identical to sending everything. An optional INT8 codec saves more bytes at a small accuracy cost.

**Models:** Dream-v0-Instruct-7B and LLaDA-8B-Instruct. **Hardware:** two NVIDIA RTX A6000 GPUs, one acting as the edge and one as the cloud; network delays are emulated from the measured traffic.

## Main results

GSM8K and long-prompt workloads, split after layer 4, emulated at 100 Mbit/s and 20 ms round-trip time, edge assumed 3× slower than an A6000.

| Workload | Method | Accuracy | Data sent per answer | Latency | GPU energy |
|---|---|---:|---:|---:|---:|
| Dream, GSM8K | Conventional split | 83% | 641 MB | 78 s | 6.7 kJ |
| | DASR (exact) | 80% | 479 MB | 62 s | 5.3 kJ |
| | DASR (INT8) | 77% | 253 MB | 44 s | 5.3 kJ |
| Dream, long prompts | Conventional split | 100% | 793 MB | 90 s | 7.3 kJ |
| | DASR (exact) | 100% | 150 MB | 25 s | 2.9 kJ |
| LLaDA, GSM8K | Conventional split | 60% | 722 MB | 88 s | 7.7 kJ |
| | DASR (exact) | 60% | 541 MB | 65 s | 4.7 kJ |

* The exact variant reproduces the output of the cached model token for token on every test prompt; accuracy differences come from caching itself and are within sampling noise.
* Savings are largest when the prompt is long relative to the answer.
* Running the whole model in the cloud is still faster (18–21 s), because a split needs one network round trip per denoising step. DASR makes splitting much cheaper when it is needed, e.g. to keep the raw prompt on the device.

Other findings: a single newly generated token changes almost every hidden row, so sending or reusing only "unchanged" rows saves very little (under 2%); and Dream's released cache adapter has an off-by-one prompt/answer boundary, which we fix.

## Repository layout

```
dasr1/
  configs/experiment_config.py   all settings (models, GPUs, experiment parameters)
  src/                           split runtime, codecs, cache wrappers, refresh rules
  experiments/                   one script per experiment (exp0 ... exp12a)
  outputs/                       results, one folder per experiment run
  data/gsm8k/                    first 200 GSM8K test problems
```

The experiments that the results above come from:

| Script | Purpose | Results |
|---|---|---|
| `exp1`–`exp7d` | How hidden states change across denoising steps; static split; row reuse | `outputs/exp*_entropy`, `exp7c` |
| `exp8f`, `exp8g` | Block caching (skipping layers), for comparison | `outputs/exp8f_entropy`, `exp8g_entropy` |
| `exp10d_gsm8k_refresh.py` | Answer quality of cache refresh policies, Dream, 100 GSM8K problems | `outputs/exp10d_T0.1` |
| `exp12a_llada_gsm8k_cache.py` | Same for LLaDA | `outputs/exp12a_llada_T0` |
| `exp11_dasr_split.py` | Two-GPU split prototype (all methods in the table) | `outputs/exp11_*_clean` |
| `exp11b_network_emulation.py` | Adds network delay for a range of links | `emulated_summary.csv` in each folder |
| `exp11c_tables_figures.py` | Final tables and figures | `outputs/exp11c` |

Folders without `_entropy` or `_clean` are earlier runs (older sampler, or GPUs shared with other jobs) kept for reference only.

## Reproducing

```bash
git clone https://github.com/meryyulinda/split-dllm.git
cd split-dllm/dasr1
conda create -n dream python=3.11 -y && conda activate dream
pip install -r requirements_dream.txt nvidia-ml-py
git clone https://github.com/maomaocun/dLLM-cache.git ../../dLLM-cache-official
```

Models download automatically from Hugging Face. Run everything from `dasr1/`; settings are in `configs/experiment_config.py`.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# answer quality of the cache (one GPU)
CUDA_VISIBLE_DEVICES=1 python experiments/exp10d_gsm8k_refresh.py
CUDA_VISIBLE_DEVICES=1 python experiments/exp12a_llada_gsm8k_cache.py

# two-GPU prototype: Dream (GSM8K, long prompts) and LLaDA, about 3 h
bash experiments/run_exp11_clean.sh

# tables and figures (no GPU)
python experiments/exp11c_tables_figures.py
```

With `CUDA_VISIBLE_DEVICES=1,2`, GPU 1 is the cloud and GPU 2 the edge. Timing and energy are only valid if no other process is computing on these GPUs; `run_exp11_clean.sh` checks this before it starts.

## Notes

* Dream uses its official GSM8K sampling (entropy order, temperature 0.1, top-p 0.9); at temperature 0 it often produces empty answers. LLaDA uses its official sampler at temperature 0.
* Latency is emulated (measured GPU time + measured bytes + a link model), not measured on a real network or edge device.
* Splitting keeps the raw prompt on the device, but the transmitted hidden states were not evaluated for information leakage.
* `requirements_llada.txt` is only needed for the early LLaDA scripts `exp7a`–`exp7c`.

## Next Steps

1. **Block-wise KV caching and parallel decoding (Fast-dLLM):** the row rule then shrinks the transmitted rows to the current block, and parallel decoding reduces the number of round trips.
2. **Cost-aware refresh rule** that weighs prompt and answer refreshes by their share of the sequence.
3. **Live codec controller** that switches between exact and INT8 transmission from the measured bandwidth.
4. **Real edge device and network**, including edge energy.
5. **More benchmarks** (e.g. MMLU, HumanEval, LongBench) and more seeds.