# Exp18: choosing the codec while the bandwidth changes

## Question (RQ4, adaptation)
Today the codec (bf16 or INT8 delta) is chosen once per request, by the backend
selector. On a link whose bandwidth changes during an answer, is it worth choosing it
at every step? And, with Fast-dLLM, is it also worth choosing the decoding threshold γ
at every step?

## Set-up
- The real TCP link of Exp16 (network emulator), but the emulator now replays a
  **bandwidth trace**:
  - **step:** 100 Mbit/s for 10 s, then 10 Mbit/s for 10 s, repeating;
  - **T-Mobile LTE / Verizon LTE uplink:** Mahimahi traces in 100-ms bins, averaging
    13.3 and 5.9 Mbit/s with outages. The downlink replays the matching `.down` trace.
- The emulator integrates the rate over the trace (an outage really stalls the link).
- Each question starts at its own position in the trace, 13 s apart. That position is
  the same for every policy, so the policies of one question see the same bandwidth.
- Dream, GSM8K, split after layer 4, RTT 20 ms, edge 3x slower (sleep emulation, as in
  the paper's latency numbers).

## Policies (`src/net_controller.py`)
| Policy | What it does |
|---|---|
| `bf16` | bf16 rows at every step (exact tier) |
| `int8` | INT8 deltas at every step (approximate tier) |
| `codec_per_request` | bf16 or INT8 chosen once, after the first two steps (today's design) |
| `codec_per_step` | bf16 or INT8 chosen at every step |
| `codec_gamma_per_step` | (Fast-dLLM) + γ = 0.9 or 0.7 chosen at every step |

**Rule:**
- The controller measures the uplink throughput B (bytes over transfer time) and the
  time per step outside the link c, both smoothed.
- **Codec:** INT8 when it would shorten the step by more than λ = 10%:
  (V_bf16 − V_INT8)/B > λ · (c + RTT + V_bf16/B). The saving depends on the step's
  rows, so large messages (block-start or prompt-refresh steps) may go as INT8 while
  small ones stay bf16.
- **γ:** 0.7 while the link takes more than half of the step time, otherwise 0.9.
- Both sides always keep the INT8 reference, so the codec can change at any step.

Mechanics: links take `encoding="adaptive"` plus `link.codec_ctl`. `kv_generate` takes
`threshold=ctl.gamma` (a function); the γ used at each step is returned in `gammas`.
Steps record the codec actually sent (`codec` column).

## Workloads
- **Fast-dLLM:** step, T-Mobile LTE and Verizon LTE; 10 questions each; 5 policies.
- **dLLM-Cache:** step only; 5 questions; 4 policies. On LTE one answer would take
  5–10 minutes.

## Run
```
nohup bash experiments/run_exp18.sh > exp18.log 2>&1 &                           # waits for idle GPUs, ~2 h
SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp18.sh > exp18_smoke.log 2>&1   # ~10 min
```
The script clones the Mahimahi traces to `../../mahimahi` if they are missing.
Outputs: `outputs/exp18_controller_{fastdllm,dllm_cache}_dream_gsm8k_t256/`
(`runs.csv`, `steps.csv`, `summary.csv`, `traces/`, `metadata.json`).

## What to read in summary.csv (per trace and policy)
- `measured_s`, `speedup_vs_bf16`, `time_vs_int8`: latency.
- `same_as_bf16`: exact answers.
- `acc`: accuracy.
- `int8_step_share`: share of steps sent as INT8.
- `mean_gamma`, `mean_steps`, `up_mb`, `switches`.
- **Checks:** `edge_has_answer` equals `runs`, and `same_as_bf16` equals `runs` for `bf16`.

Read it as a trade-off: how close does each policy get to the `int8` latency, and how
many answers stay exact?

## Limits
- The link has no loss, and TCP congestion control does not react to the trace,
  because the emulator queues without dropping.
- The controller reads the cloud's receive time directly; edge and cloud share one clock
  in the prototype. A real deployment would have the cloud report its receive time.
