# Exp17: offloading

## Question
The paper now motivates splitting by offloading computation from the cloud. How much
cloud GPU time and energy does a split actually save per answer? And what does the
edge pay for it, in time, memory and latency, as the split moves deeper?

The boundary rows have the same size at every layer, so the bytes sent do not depend
on the split layer k. Only the work divides differently between the two sides.

## Set-up
- Dream, 10 GSM8K problems, 256 tokens. Two GPUs, no network (as in Exp11 / Exp13).
- **Cloud-only reference:** the whole model on the cloud GPU, with the same cache and
  decoding. The split hooks stay attached with edge = cloud, so it is timed with the
  same CUDA events as the split runs: cloud GPU time = all layers + sampler.
- **Split after layer k** = 2, 4, 8, 14, 20 (of 28):
  - cloud GPU time = cloud layers + sampler (CUDA events);
  - edge GPU time = edge layers + codec.
- **Energy:** NVML energy per GPU, above idle power.
- **Peak memory** per side.
- **Latency:** the latency equation at 10 / 100 / 1000 Mbit/s, RTT 20 ms, edge 3x
  slower. It is compared with cloud-only = measured time + one RTT. (Exp16: the
  equation is 3-10% optimistic on a real link.)

| Backend | Variants (paper names) | Compared with |
|---|---|---|
| dLLM-Cache (T=0.1, top-p 0.9, Kp=25, Kr=2) | DASR + dLLM-Cache (bf16), (INT8) | Cloud-only + dLLM-Cache |
| Fast-dLLM (greedy, n_block = 32, γ = 0.9) | DASR + Fast-dLLM (bf16), (INT8) | Cloud-only + Fast-dLLM |

## Run
```
SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp17.sh > exp17_smoke.log 2>&1   # ~5 min
nohup bash experiments/run_exp17.sh > exp17.log 2>&1 &                           # ~40 min, idle GPUs
```
Outputs: `outputs/exp17_offloading_{dllm_cache,fastdllm}_dream_gsm8k_t256/`
(`runs.csv`, `summary.csv`, `metadata.json`).

## What to read in summary.csv (one row per split layer and variant)
- **`cloud_saving`:** share of the cloud GPU time per answer taken off the cloud.
  Expect roughly k/28 minus the sampler share; the sampler always runs on the cloud.
- **`answers_per_cloud_gpu_ratio`:** how many times more answers one cloud GPU can serve
  per hour than with cloud-only. This assumes the cloud serves answers one after
  another without batching across users; a real server would batch.
- **`cloud_energy_saving`:** the same for the cloud GPU's energy above idle.
- **What the edge pays:** `edge_s`, `edge_peak_gib`, and `latency_ratio_<B>mbps` (split
  latency / cloud-only latency).
- **Checks:**
  - `same_as_cloud` should equal `runs` for the bf16 variants at every k. This
    generalizes Exp13C to every split layer and to dLLM-Cache.
  - `acc` vs `cloud_only_acc`.

- **Energy:** if the log warns that the idle power is above 150 W, another job was
  using the GPU (the smoke test on 25 Nov measured 284 W on the cloud GPU), and the
  energy and timing of that run are not valid.

## Limits
- One cloud GPU serving one user at a time. Capacity with batching is not measured.
- The edge is an A6000; its latency is scaled by 3 in the equation, its energy is not.
