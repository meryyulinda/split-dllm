# Exp16: the split over a real link

## Question
Every latency number in the paper comes from the latency model (Exp11b):

    time per answer = sum over steps of (edge + encode + bytes / bandwidth + RTT + cloud + sampler)

applied to measured compute and bytes. Exp16 runs the split over a real, shaped network
link and checks whether the model predicts the measured time per answer.

## Set-up
- Same split as Exp11 / Exp13: Dream, GSM8K, split after layer 4, edge on one GPU, cloud
  on another.
- **New:** every message between the halves goes as bytes over TCP through a relay
  process (`src/net_relay.py`, one process per direction, no sudo):
  - uplink: the boundary rows;
  - downlink: the committed tokens.
- The relay imposes a fixed bandwidth (plus 40 B of TCP/IP headers per 1460-B packet)
  and a one-way delay of RTT/2. It queues without loss, i.e. the link the model assumes.
  A bandwidth trace can replace the fixed rate (`{"mbps": None, "rtt_ms": 20, "n": 10,
  "trace": "path.csv"}` in `EXP16_LINKS`, CSV columns `time_s,mbps`).
- The cloud computes only from the rows it decoded from the received bytes, and the
  edge only from the tokens it received (`src/net_wire.py`). The answers therefore also
  test the wire format: `same_as_local` must be true for every run.
- The two halves still run in one Python process (same model code and samplers as
  Exp11 / Exp13); only the link is real. The edge is the same A6000 as the cloud
  (`EXP16_EDGE_SLOWDOWN = 1`). A slower edge can be emulated by sleeping (s - 1) x the
  measured edge time per step.

## Configurations
| Backend | Variant | Paper name |
|---|---|---|
| dLLM-Cache (T=0.1, top-p 0.9) | S0_split_full | Static split (bf16) |
| | C2_cache_aware | DASR + dLLM-Cache (bf16) |
| | C4_cache_aware_int8 | DASR + dLLM-Cache (INT8) |
| Fast-dLLM (greedy, n_block = 32, γ = 0.9) | K2_dual_thr0.9 | DASR + Fast-dLLM (bf16) |
| | K3_dual_thr0.9_int8 | DASR + Fast-dLLM (INT8) |

Optional: S2_split_int8 (Static split (INT8)) and A1_adaptive_cache_aware (adaptive
refresh), via `EXP16_VARIANT_NAMES`.

Links (bandwidth Mbit/s, RTT ms, questions):
- dLLM-Cache: 1000/20/10, 100/20/10, 10/20/2. At 10 Mbit/s one static-split answer
  takes about 9 minutes, so only 2 questions there.
- Fast-dLLM: 1000/20/10, 100/20/10, 10/20/10.
- Every question also runs once without network ("local").

## Run
```
SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp16.sh > exp16_smoke.log 2>&1   # ~10 min
nohup bash experiments/run_exp16.sh > exp16.log 2>&1 &                           # ~2 h, idle GPUs
```
Outputs: `outputs/exp16_dllm_cache_dream_gsm8k_k4_t256/` and
`outputs/exp16_fastdllm_dream_gsm8k_k4_t256/` (`runs.csv`, `steps.csv`, `summary.csv`,
`metadata.json`).

## What to read in summary.csv
- **Checks:** `same_as_local`, `edge_has_answer` and `bytes_match` should all equal
  `runs`. `ping_ms` should be close to `rtt_ms`.
- **`ratio_measured_predicted`:** measured / model, with the model applied to the same
  run's compute and bytes. Within about 1.05 means the model holds.
- **`ratio_measured_predicted_local`:** the same against the run without network, which
  is what the paper reports.
- **`wire_overhead_ms_per_step`:** time per step the model leaves out (GPU-to-CPU
  copies, framing, sockets, the relay's TCP/IP headers).
  - If it is large at 1000 Mbit/s, the paper should add it as a per-step constant.
  - If it is small, the model is validated as it is.

## Limits (to state if reported)
- The relay is a fixed-rate, no-loss link. It does not show TCP congestion behaviour on
  lossy or variable wireless links (Mahimahi or tc/netem on a machine we control, or
  trace replay here).
- The edge is an A6000, not an edge device.
- Both halves share one process and one CPU, so Python overheads on the two sides
  are not separate.
