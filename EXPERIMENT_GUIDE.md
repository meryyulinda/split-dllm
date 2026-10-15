# Navigating the code and running the research experiments

Start with `run_experiments.py` in the workspace root. It names the experiment stages and delegates to the existing package. Keep it beside `dasr/`. It is a Python entry point, not a duplicated copy of the model/runtime. In Colab, the notebook extracts `dasr/` and provides equivalent `python -m dasr ...` commands.

The current saved results are toy-model protocol checks. In particular, `dasr/results/network_estimates.csv` is an analytical sweep derived from toy logs, not a LLaDA or measured-network result. Do not use these numbers in the paper.

## 1. Read files in this order

| File | What it controls | What you usually do with it |
| --- | --- | --- |
| `run_experiments.py` | Named experiment stages | Run `check`, `codecs`, `ffn`, `drift`, `cloud`, `analyze`, etc. |
| `dasr/__main__.py` | CLI arguments, data loading, query sweeps, summaries, profiling and validation | Read `benchmark()` first; use `parser()` to find supported flags |
| `dasr/configs/colab_pilot.json` | Communication baseline configurations | Change codec, drift threshold and refresh interval; keep first configuration dense/exact |
| `dasr/configs/ffn_ablation.json` | Compute-reuse ablations | Change active fraction and refresh interval |
| `dasr/runtime.py` | Denoising and split execution | Read `RunConfig`, `run_query()`, then `CloudEndpoint.exchange()` |
| `dasr/torch_backend.py` | Actual LLaDA/Dream weights, prefix/suffix blocks and FFN reuse | Read `TorchModel.prefix()`, `suffix()`, `proposals()`, then `RowCache.forward()` |
| `dasr/codec.py` | Boundary encoding, cache versions, quantization, reconstruction | Read `BoundaryEncoder.encode()` and `BoundaryCache.apply()` |
| `dasr/transport.py` | Framed persistent TCP and application-byte serialization | Needed for physical two-host experiments; local runs use the same message representation |
| `dasr/scheduler.py` | Table-driven action selection | Read `Action.cost()` and `Scheduler.choose()` after obtaining calibration measurements |
| `dasr/metrics.py` | Numeric answer scoring, bootstrap CIs, environment capture | Inspect before interpreting quality or uncertainty |
| `dasr/scripts/external_baselines.py` | Author Fast-dLLM v1 and dLLM-Cache runners | Run separately in compatible author environments |
| `dasr/scripts/network_sweep.py` | Analytical bandwidth/RTT post-processing | Produces modeled latency, not measured network latency |
| `dasr/scripts/colab_support.py` | GPU/RAM inspection, split sizing and pinned download helpers | Used by notebook setup |
| `dasr/scripts/build_notebook.py` | Embeds package source in the notebook | Regenerate only after editing source; ordinary experiments do not need it |
| `dasr/toy.py` | Small bidirectional semantic reference | Debugging only |
| `dasr/tests/test_protocol.py` | Protocol, split, TCP and scheduler checks | Run before real checkpoints |
| `dasr/__init__.py`, `scripts/__init__.py`, `tests/__init__.py` | Package declarations | No experiment settings to change |
| `requirements-colab.txt` | Dependency compatibility target | Install in the Colab environment |
| `README.md`, `PAPER_REVIEW.md`, `SOURCES.md`, `IMPLEMENTATION_STATUS.md`, `paper_corrections.tex` | Commands, manuscript corrections, source links and implementation limits | Use when writing/revising the paper |

Execution flow:

```text
run_experiments.py → __main__.benchmark()
  → TorchModel: load checkpoint once for the chosen split
  → run_query(): execute every denoising step
      → prefix on edge
      → encode boundary and build request
      → local CloudEndpoint or TCP cloud
      → reconstruct boundary, execute suffix and commit tokens
      → return next IDs and verify cache acknowledgment
  → configuration.jsonl + manifest.json + summary.json
  → analyze(): summary.csv and plots
```

`--transport local` emulates a split on one host. `--transport tcp` uses separate processes/hosts. Neither codec selection nor a local CPU prefix creates a physical edge device.

## 2. Keep three comparison groups separate

### Placement baselines

- Cloud-only: `--split 0 --codec full`. Denoising stays in cloud; one request and final response.
- Edge-only: `--split L --codec full`. All layers/head execute at the edge; no communication. For LLaDA-8B, verify the checkpoint's `L` rather than assuming it for other models.
- Static split/full: `0 < split < L`, `full_dense`. Both parts compute densely.

Choose feasible placement from actual memory checks. If full GPU cloud-only does not fit, report infeasible/OOM; silently adding CPU weight offload changes the baseline. A CPU-only edge is an emulated/CPU edge, not the paper's RTX 3060.

### Communication comparisons: `colab_pilot.json`

| Configuration | Prefix/suffix computation | Boundary mechanism | Correctness tier |
| --- | --- | --- | --- |
| `full_dense` | Dense | Native tensor each step | Exact transport reference |
| `zlib_dense` | Dense | Lossless compression of full tensor | Exact transport |
| `xor_dense` | Dense | Native-bit XOR against synchronized reference, anchors/fallback | Exact transport |
| `int8_dense` | Dense | Full row-wise INT8 tensor | Approximate |
| `sparse_native` | Dense | Selected native row replacements | Approximate because omitted rows may drift |
| `sparse_int8` | Dense | Selected INT8 row replacements | Approximate |

All six use the same loaded split and sampler. Thus changes in bytes/latency isolate communication mechanisms. `sparse_native` uses the selected native dtype: FP16, BF16 or FP32. Do not label it FP16 if your run used BF16.

### Compute reuse: `ffn_ablation.json`

| Configuration | Compute policy | Communication | Interpretation |
| --- | --- | --- | --- |
| `full_dense` | Dense | Full | Reference |
| `ffn_50pct` | Approximate FFN updates with initial 50% row target | Full | FFN computation ablation |
| `ffn_25pct` | Approximate FFN updates with initial 25% row target | Full | More aggressive FFN ablation |
| `ffn_sparse_int8` | Approximate FFN updates | Sparse INT8 | Combined compute/communication prototype |

Attention remains dense. Mandatory changed rows, periodic refreshes and audits can increase actual work beyond the initial percentage. These are not reproductions of Fast-dLLM, dLLM-Cache or sparse-attention kernels.

Use the paper labels `DASR-static: FFN-only prototype` and `DASR-adaptive: fixed-split codec policy` if reporting current functionality. A full adaptive layer-placement system is not implemented.

## 3. Freeze the experimental contract

For every paired comparison fix model/checkpoint commit, native dtype, tokenizer/chat template, example IDs, prompt/few-shot setting, batch size, generation length, steps, block length, commit rule and seeds. Also fix devices and the latency/byte measurement scope. Sweep only the intended factor.

Use disjoint calibration and evaluation IDs. The supplied GSM8K command prepares a zero-shot numeric pilot, not the author five-shot evaluation. Thirty queries is a systems pilot; use a larger held-out set for a tight quality margin. Length 32 is a functional smoke setting. Truncated reasoning answers cannot establish benchmark quality.

In Colab, use the notebook's pinned `CHECKPOINT`, `MODEL_REVISION`, `DTYPE`, `SPLIT` and completed native-validation report. In terminal examples below uppercase tokens are values you must substitute.

## 4. Run the stages

Run from the workspace root, which contains `run_experiments.py` and `dasr/`. In Colab use a notebook `!python ...` cell or `subprocess.run`. Running `dasr/__main__.py` directly as a file can break relative imports; use `python -m dasr` or the new entry point.

First check the protocol:

```bash
python run_experiments.py check
python run_experiments.py codecs --backend toy --split 2 --gen-length 8 --steps 8 --block-length 8 --output results/debug
python run_experiments.py analyze --input results/debug
```

Prepare data and validate native model/sampler parity before real comparisons:

```bash
python run_experiments.py prepare-data --revision DATASET_COMMIT --limit 30 --seed 7 --output data/eval.jsonl
python run_experiments.py validate --backend llada --checkpoint CHECKPOINT --revision MODEL_COMMIT --dtype DTYPE --edge-device VALIDATION_DEVICE --cloud-device VALIDATION_DEVICE --official-repo author_code/LLaDA --gen-length 8 --steps 4 --block-length 8 --output results/native_validation.json
```

`VALIDATION_DEVICE` must be the same device for both sides and must fit the complete native model. A CPU validation needs enough RAM. Validation with eight tokens checks implementation parity, not task quality.

Then run the communication study and the FFN study with identical evaluation parameters:

```bash
python run_experiments.py codecs --backend llada --checkpoint CHECKPOINT --revision MODEL_COMMIT --dtype DTYPE --split K --edge-device cpu --cloud-device cuda:0 --data data/eval.jsonl --gen-length 256 --steps 256 --block-length 32 --warmup 1 --validation-report results/native_validation.json --output results/llada_codecs_kK
python run_experiments.py ffn --backend llada --checkpoint CHECKPOINT --revision MODEL_COMMIT --dtype DTYPE --split K --edge-device cpu --cloud-device cuda:0 --data data/eval.jsonl --gen-length 256 --steps 256 --block-length 32 --warmup 1 --validation-report results/native_validation.json --output results/llada_ffn_kK
python run_experiments.py analyze --input results/llada_codecs_kK
python run_experiments.py analyze --input results/llada_ffn_kK
```

These longer lengths are an example study setting, not a promise of Colab memory/runtime feasibility. First run the shorter three-query notebook smoke test and recheck memory, particularly before FFN caching.

Cloud-only uses the same checkpoint/data/sampler flags but `cloud` forces split zero:

```bash
python run_experiments.py cloud --backend llada --checkpoint CHECKPOINT --revision MODEL_COMMIT --dtype DTYPE --cloud-device cuda:0 --data data/eval.jsonl --gen-length 256 --steps 256 --block-length 32 --output results/llada_cloud
```

Use `bench --split L --codec full` for edge-only, and set the edge device explicitly. Cross-folder native/full output comparison is still required: the automatic sweep agreement check is relative to the first configuration in that sweep, not automatically to cloud-only or author outputs.

## 5. Author-cache baselines

`external_baselines.py` supports LLaDA standalone cloud-only author code. It does not automatically combine their caches with our split protocol or support both models' author baselines. Run in fresh compatible environments; imported modules may conflict otherwise.

```bash
python -m dasr.scripts.external_baselines --method fast_v1 --repo author_code/Fast-dLLM --checkpoint CHECKPOINT --revision MODEL_COMMIT --dtype DTYPE --data data/eval.jsonl --gen-length 256 --steps 256 --block-length 32 --limit 30 --output results/fast_v1_cache.jsonl
python -m dasr.scripts.external_baselines --method dllm_cache --repo author_code/dLLM-cache --checkpoint CHECKPOINT --revision MODEL_COMMIT --dtype DTYPE --data data/eval.jsonl --gen-length 256 --steps 256 --block-length 32 --limit 30 --output results/dllm_cache.jsonl
```

Match these lengths to the actual DASR study. First compare each author cache to a native cloud-only run on the same GPU. That answers whether the cache helps its backbone. Then compare deployment policies within a separate plot/table that includes communication and emulated edge costs. The author runner times model/sampler inference; the DASR runner includes protocol/serialization overhead. Align timing boundaries before calculating direct speedups across the two runners. Author logs do not contain split bytes, edge compute or our step schema, and `dasr analyze` does not automatically ingest their format.

Fast-dLLM's default runner omits the confidence threshold. Adding `--threshold 0.9` changes decoding and belongs in a secondary sampler study; log actual NFE. Verify the author sampler/model path and prompt parity even without the threshold. dLLM-Cache refresh/transfer settings are explicit runner flags. Author runners were source-inspected but not executed against large checkpoints here.

## 6. Answer the RQs

| RQ | Run | Fields/evidence | What to conclude |
| --- | --- | --- | --- |
| RQ1 token changes/drift | `drift` at several feasible `--split K`, with full dense computation | `changed_ratio`, `boundary_row_drift`, layer K and step | How much temporal redundancy exists and where; aggregate all queries and separate prompt/masked/changed positions |
| RQ2 exact byte/latency savings | `codecs`: full vs zlib vs XOR | Application bytes, measured/modeled latency, native boundary equality and paired output IDs | Whether lossless transport saves bytes and whether codec/copy overhead defeats latency savings |
| RQ3 selective work/quality | `ffn`: full vs 50%/25% FFN vs combined codec | `edge_ms`, `cloud_ms`, executed FFN rows, task-score loss, audits, memory | Whether the implemented FFN subset actually improves time at a predefined quality margin; no sparse-attention claim |
| RQ4 adaptation | Calibration `codecs` runs → `profile` → `adaptive` on held-out IDs | Selected actions, predicted vs measured stage/step costs, anchor/switch trace, quality | Fixed-split codec-policy benefit; `replay` is only placement modeling |
| RQ5 transfer | Repeat validation/studies using Dream and additional standardized tasks | Per-model/task scores, bytes and latency | Transfer only after both models and multiple tasks are actually evaluated |

RQ1 command:

```bash
python run_experiments.py drift --backend llada --checkpoint CHECKPOINT --revision MODEL_COMMIT --dtype DTYPE --split K --data data/eval.jsonl --gen-length 256 --steps 256 --block-length 32 --output results/drift_kK
```

The built-in drift plot shows the first query. The code logs every query; aggregate them yourself for paper figures. Step zero initializes caches and reports changed ratio one by convention; exclude it from claims about between-step token changes. These instrumentation runs must stay separate from timing runs.

RQ4 commands:

```bash
python run_experiments.py profile --input results/calibration_kK --switch-ms MEASURED_SWITCH_MS --output results/profile.json
python run_experiments.py adaptive --backend llada --checkpoint CHECKPOINT --revision MODEL_COMMIT --dtype DTYPE --split K --data data/heldout.jsonl --gen-length 256 --steps 256 --block-length 32 --adaptive-profile results/profile.json --network-trace dasr/configs/network_trace.json --memory-budget-bytes EDGE_BUDGET_BYTES --risk-budget 0.01 --output results/adaptive_kK
```

Compare adaptive to the strongest feasible static codec chosen using calibration, on the same held-out data/network trace; full transmission alone is not a sufficient static comparator. The current sweep automatically adds a full reference, not every static comparator. Run the calibration-selected static policies separately. Switch estimates appear in scheduler predictions, not an actually executed live layer migration; full placement/switch claims need more implementation. Cloud-only currently models a fixed link setting, so a varying-link comparison needs an explicitly chosen request-level network convention.

## 7. Understand result files

Each ordinary benchmark output directory contains:

- `manifest.json`: checkpoint revision, environment, dataset hash, CLI settings, deployment and byte scope. Inspect this first.
- `CONFIGURATION.jsonl`: one line per query with `id`, `text`/`output_ids`, `numeric_score`, `reference_agreement`, `totals`, `steps`, and configuration.
- `summary.json`: means and bootstrap CIs for score, latency, bytes and agreement.
- `summary.csv`: compact means, generated by `analyze`.
- `tradeoffs.png`: token-agreement diagnostics versus modeled latency/bytes.
- `quality_latency.png`: numeric pilot score versus modeled latency, if scores exist.
- `*_active.png`: first-query changed/active fractions.
- `*_drift.png`: first-query drift, only if instrumentation exists.

The `publication_protocol_ready` field is a limited bookkeeping check, not publication certification. It checks query count, real backend, supplied validation and instrumentation flags; it cannot certify dataset quality, matching baselines, hardware scope or global noninferiority.

## 8. Fill evaluation-table columns from measured evidence

| Paper column | Use this field/calculation | Current limitation |
| --- | --- | --- |
| Model / Method | Manifest checkpoint and configuration name | Qualify FFN-only/adaptive-codec prototype and emulated hardware |
| Quality | `100 × summary.numeric_score.mean`; scale CI similarly for percentages | Numeric zero-shot GSM8K pilot only; agreement is not quality |
| Edge / Cloud GFLOPs | Measured profiler counts divided by `1e9` | `edge_flops` and `cloud_flops` are null; FFN row counts are not FLOPs |
| Wire MB | Physical captured link bytes divided by `1e6` | Current `application_bytes` is application frames; rename column `Application MB` for Colab |
| Boundary MB | Sum `steps.boundary_packet_bytes / 1e6` | Optional separate payload diagnostic; excludes response/control traffic |
| Cache hit | Executed FFN reuse proxy `1 - totals.ffn_selected_rows / totals.ffn_total_rows` | Names and scope must say FFN row reuse; audit executions are additional work and are not included in this proxy |
| Boundary row reuse | `1 - boundary_active_ratio` for sparse codecs | Do not use this for XOR compression, which encodes every row |
| Latency ms | `totals.latency_ms`, summarized across matched queries | Single-host measurements exclude actual network when no sleep is injected |
| Emulated E2E latency ms | `totals.modeled_latency_ms` | Modeled RTT/bandwidth; cannot label measured physical E2E |
| Throughput | `sum(gen_length) / sum(latency_ms / 1000)` across completed serial queries | This is allocated output positions/s; inspect masks/truncation and use consistent accepted-output definition |
| Emulated throughput | Same ratio using modeled latency | Match its timing scope to the latency column |
| Edge GiB | Edge GPU peak allocated bytes divided by `2**30` | CPU prefix has no edge GPU peak; whole-process RSS is not an isolated edge peak |
| Cloud GiB | Cloud GPU peak allocated bytes divided by `2**30` | GPU allocator peak, not all device memory |
| Energy J | Instrumented device energy | `energy_j` is null; Colab energy cannot be inferred from latency |

Choose one timing definition for each table, label it in the caption, and use the same definition for throughput/speedups. Do not average per-query speedups as though they equal a ratio of total runtimes. For serial benchmark throughput use the ratio of total generated positions to total time; the average of per-query tokens/s is a different statistic.

For a same-split communication result, compute byte reduction from matched queries:

```text
byte_reduction = 1 - sum(candidate application bytes) / sum(full application bytes)
speedup = sum(full latency) / sum(candidate latency)
quality_change_pp = 100 × (candidate mean score - full mean score)
```

Bootstrap paired query differences/ratios when claiming a comparative CI. The current summaries bootstrap each configuration's mean separately; they are not automatically CIs for the difference. A sparse codec can send more bytes than full transmission if nearly every row changes; report that outcome rather than assuming sparsity wins.

Use `--`/`unmeasured` for missing instrumentation, and `infeasible/OOM` for a deployment that cannot fit. Do not use zero for either. Keep toy runs, source-screened but unexecuted baselines, analytical placement replay and physical results in separately identified rows.

## 9. Minimum order for the first study

1. Protocol tests and a toy debug run.
2. Colab memory plan, pinned checkpoint, native adapter/official sampler gate.
3. Three-query LLaDA codec smoke test with short outputs.
4. Freeze a feasible longer-output evaluation setting and task protocol.
5. Same-split dense communication comparisons for RQ2.
6. Separate drift instrumentation across splits for RQ1.
7. FFN-cache ablations with paired quality checks for the restricted RQ3.
8. Native cloud and author-cache baselines where feasible, with timing boundaries aligned.
9. Disjoint calibration, strongest static comparator and adaptive-codec evaluation for restricted RQ4.
10. Dream and standardized additional tasks for RQ5; physical hardware/instrumentation for remaining table columns.
