# Diffusion-Aware Split Runtime experiments

For the file-by-file reading order, baseline comparisons, RQ experiments and evaluation-table mapping, start with [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md). The workspace-level [run_experiments.py](../run_experiments.py) provides readable stage names such as `check`, `codecs`, `ffn`, `drift` and `analyze` while using this package.

Open `../Untitled-1.ipynb` in Colab and run its cells in order. It embeds this source package, installs a compatibility environment, chooses a memory-feasible CPU-edge/GPU-cloud split, downloads a pinned LLaDA checkpoint when you run the download cell, and provides native-forward/official-sampler validation and codec sweeps. Your original empty notebook is preserved as `../Untitled-1.original.ipynb`.

The implementation supports exact full/zlib/XOR transmission, approximate sparse/native and row-wise INT8 transmission, approximate FFN row caching, persistent TCP partition execution, query/stage logs, bootstrap CIs, plotting, measured profile tables, fixed-split codec adaptation, and offline placement replay. Read [the paper review](PAPER_REVIEW.md), [source screen](SOURCES.md), and [implementation status](IMPLEMENTATION_STATUS.md) before interpreting results.

## Local protocol checks

From the workspace parent of `dasr`:

```bash
python -m unittest discover -s dasr/tests -v
python -m dasr bench --backend toy --split 2 --sweep dasr/configs/colab_pilot.json --gen-length 8 --steps 8 --block-length 8 --trace-drift --output dasr/results/toy_smoke
python -m dasr analyze --input dasr/results/toy_smoke
```

Toy data only verifies systems semantics. Its tokens are not language and its timing/byte numbers must not enter the paper's pretrained-model result tables.

## LLaDA pilot outside the notebook

Use CUDA PyTorch >=2.5 and `dasr/requirements-colab.txt`; retain the matching CUDA build. Resolve the model revision and author repository commit and record them. The exact checkpoint version is passed as `REVISION` below; substitute the value rather than using a mutable `main` in final experiments.

```bash
python -m dasr prepare-gsm8k --revision DATASET_REVISION --limit 30 --seed 7 --output data/eval.jsonl
python -m dasr bench --backend llada --checkpoint CHECKPOINT_PATH --revision REVISION --dtype bfloat16 --split 12 --edge-device cpu --cloud-device cuda:0 --data data/eval.jsonl --gen-length 32 --steps 16 --block-length 32 --sweep dasr/configs/colab_pilot.json --output results/llada_pilot
python -m dasr analyze --input results/llada_pilot
```

Split 12 is an example, not a universally feasible optimum. The notebook computes a conservative plan for its actual allocation. On a T4 use FP16 consistently for all comparisons; on suitable newer GPUs use BF16. `k=0` is cloud-only, `k=L` is edge-only. A CPU edge and GPU cloud on one host is emulation. Do not identify Colab as the proposed RTX 3060/RTX 6000 physical testbed.

The native adapter/sampler gate needs the complete checkpoint on a single sufficiently provisioned GPU or CPU. It fails instead of silently accepting differences:

```bash
python -m dasr validate-model --backend llada --checkpoint CHECKPOINT_PATH --revision REVISION --dtype bfloat16 --edge-device cpu --cloud-device cpu --official-repo author_code/LLaDA --gen-length 8 --steps 4 --block-length 8 --output results/native_validation.json
```

After that succeeds, pass `--validation-report results/native_validation.json` to `bench` using the same checkpoint revision and dtype. This is an initial adapter/sampler check; heterogeneous execution still needs paired exact-tier output checks. GPU execution and these real checkpoint checks were not performed in the authoring workspace.

For Dream select `--backend dream --checkpoint Dream-org/Dream-v0-Instruct-7B`; native generation uses `maskgit_plus`, temperature 0, `alg_temp=0`, epsilon .001 and full-canvas output (`block-length = gen-length`). The adapter performs the model's required shifted logit interpretation. Run its own `validate-model` gate before comparing results.

## Inputs and evaluation

JSONL rows contain a unique `id`, a user `prompt`, optional `answer`, `task`, and `scorer`. `numeric` is the only integrated task scorer. Any other scorer leaves task score unavailable. Generation is batch one, deterministic, temperature zero, no classifier-free guidance. Each codec sweep begins with an exact dense reference.

```json
{"id":"question-001","task":"gsm8k","prompt":"How many pencils are in 3 boxes of 4? End with #### and the answer.","answer":"12","scorer":"numeric"}
```

The built-in GSM8K preparation is a fixed-ID zero-shot numeric pilot. It is not interchangeable with author five-shot scores. For standardized results, use the model's official evaluation prompts, lengths and benchmark scorer. Thirty timed unique queries is a systems minimum; use larger held-out task sets for small quality margins. Calibration questions must have disjoint IDs from evaluation questions.

Warm-up excludes model load and tokenizer download. Model load time and weight migration are not included in query compute timing. JSONL logs include every stage, codec mode, selected fractions, anchors, codec error, query output IDs, memory scope and measured/modelled latency. Energy/FLOPs are null where unmeasured. GPU peaks are allocator measurements; CPU RSS is whole-process resident memory, not an isolated edge GPU peak. Codec checksums and sampler responses include control/telemetry bytes. TCP/IP headers, connection setup and physical retransmissions require external packet accounting.

## Selective computation and RQ1

```bash
python -m dasr bench --backend llada --checkpoint CHECKPOINT_PATH --revision REVISION --split 12 --dtype bfloat16 --data data/eval.jsonl --sweep dasr/configs/ffn_ablation.json --output results/ffn
python -m dasr bench --backend llada --checkpoint CHECKPOINT_PATH --revision REVISION --split 12 --dtype bfloat16 --data data/eval.jsonl --codec full --trace-drift --limit 3 --output results/drift_k12
```

The compute ablation skips actual token-local FFN work while attention stays dense. It uses mandatory changed rows plus random fill, per-operator random audits, full operator fallback and periodic dense refresh. It does not implement the paper's attention-aware active-set expansion or a fused selective-attention kernel. Audits are local operator diagnostics, not a full dense branch or global quality guarantee. Cache storage can be substantial, especially LLaDA's intermediate FFN projection; recheck memory headroom before enabling it.

Run drift traces at several feasible splits, aggregate over queries, and keep instrumentation separate from latency runs. Boundary drift detection is post-prefix and saves communication only. The two selected-row ratios refer to different mechanisms: `boundary_active_ratio` counts transmitted rows; `ffn_active_ratio` counts the input active set for FFN reuse. Guard-triggered FFN work and audit refreshes may exceed that initial set.

## Networks and adaptation

```bash
python -m dasr.scripts.network_sweep --input results/llada_pilot/xor_dense.jsonl --output results/network_estimates.csv
python -m dasr profile-table --input results/calibration_k12 --switch-ms MEASURED_CODEC_SWITCH_MS --output results/profile.json
python -m dasr bench --backend llada --checkpoint CHECKPOINT_PATH --revision REVISION --split 12 --dtype bfloat16 --data data/heldout.jsonl --adaptive-profile results/profile.json --network-trace dasr/configs/network_trace.json --memory-budget-bytes EDGE_MEMORY_BUDGET --risk-budget 0.01 --output results/adaptive_codec
python -m dasr replay-scheduler --profile results/profile_multisplit.json --trace dasr/configs/network_trace.json --memory-budget EDGE_MEMORY_BUDGET --risk-budget 0.01 --output results/placement_replay.json
```

Replace uppercase placeholders with measured values. The supplied network trace contains scenario settings, not experimental observations. `network_sweep` adds an analytical symmetric-link serialization cost and one RTT per split exchange (one per entire cloud-only request). `--sleep-network` optionally injects that delay into single-host wall time. Neither produces physical link evidence.

`profile-table` uses observed stages, bytes, memory and paired calibration task-score loss. Approximate configurations without scores are rejected, not assigned an invented risk. The upper bootstrap loss interval is an empirical calibration statistic, not a formal safety bound. Live scheduling is limited to codecs/refresh intervals at the loaded split and current compute policy. It logs selected actions and predicted cost. Full split adaptation is trace replay; to demonstrate RQ4 for migrating weights, implement and measure residency, transfer and reset costs on the deployed system. Profiles should be stage-conditioned for the eventual paper rather than relying only on these aggregate pilot means.

## Physical edge/cloud TCP

Start the cloud partition on the cloud host, then the edge client on the edge host using the same checkpoint revision, native dtype and split:

```bash
# Cloud host (trusted research link)
python -m dasr serve --backend llada --checkpoint CHECKPOINT_PATH --revision REVISION --split 12 --dtype bfloat16 --cloud-device cuda:0 --host 0.0.0.0 --port 8765
# Edge host
python -m dasr bench --backend llada --checkpoint CHECKPOINT_PATH --revision REVISION --split 12 --dtype bfloat16 --edge-device cuda:0 --cloud-device cuda:0 --transport tcp --host CLOUD_HOST --port 8765 --data data/eval.jsonl --sweep dasr/configs/colab_pilot.json --output results/physical_split
```

The edge process streams only prefix parameters and the cloud process only suffix/head parameters; the author checkpoint archive may still need to exist at both endpoints. Meta placeholders for unowned layers are never executed. The server is serial and batch one. Use a trusted tunnel or protected test network for this plain TCP research protocol; it is not a production service or privacy-preserving design. Checksums detect corruption/reference mismatches. A rejected patch is resent as an anchor, with retry bytes counted. A lost response after token commitment aborts/restarts the query because blindly replaying sampling could advance it twice.

Use both hosts' traffic controls/counters when making physical bandwidth/RTT claims. Asymmetric uplink/downlink modeling and TCP congestion effects need a physical sweep. Do not double-count analytical sleep on top of a traffic-shaped real link.

## Required author baselines

Run each in a fresh process with its author dependencies. Keep author sources in their repositories; record `git rev-parse HEAD`. The thin runner imports author cache/sampler code rather than pretending our FFN cache reproduces it.

```bash
git clone https://github.com/NVlabs/Fast-dLLM.git author_code/Fast-dLLM
git clone https://github.com/maomaocun/dLLM-cache.git author_code/dLLM-cache
python -m dasr.scripts.external_baselines --method fast_v1 --repo author_code/Fast-dLLM --checkpoint CHECKPOINT_PATH --revision REVISION --data data/eval.jsonl --dtype bfloat16 --gen-length 256 --steps 256 --block-length 32 --output results/fast_v1_cache.jsonl
python -m dasr.scripts.external_baselines --method dllm_cache --repo author_code/dLLM-cache --checkpoint CHECKPOINT_PATH --revision REVISION --data data/eval.jsonl --dtype bfloat16 --gen-length 256 --steps 256 --block-length 32 --output results/dllm_cache.jsonl
```

For Fast-dLLM, adding `--threshold .9` is a **sampler-change secondary experiment**; the default runs prefix caching without that threshold. Actual NFE is logged when the author sampler exposes it. dLLM-Cache uses its author feature-cache hooks and sampler, with explicit prompt/generation refresh and transfer settings. Verify author sampler parity/prompt conventions before using these as a matched comparison. These external runners have not been executed against real weights here. Full GPU baselines may be infeasible on a small Colab GPU; do not silently offload/quantize them and present the resulting latency as the original baseline. Automatic composition of either author cache with the split runtime is not implemented.

## Regenerate or export

After changing package sources, regenerate the embedded notebook with:

```bash
python -m dasr.scripts.build_notebook
```

The notebook contains an export cell for scripts/results/manifests. It excludes downloaded checkpoints. The local folder contains toy verification artifacts only. Read the manifest of every result directory before labeling a figure or filling a paper table.
