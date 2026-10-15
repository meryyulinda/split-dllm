# Review of the supplied LaTeX excerpt

The proposal is implementable as a staged systems study. Communication savings are plausible but are not established measurements. General exact reuse of unchanged token rows after bidirectional attention is impossible without additional invariance assumptions. The supplied draft already recognizes this; preserve that distinction in every result table.

The implementation in this folder establishes a concrete protocol and measurement path. It does not certify the full research hypothesis, optimized sparse attention, physical edge energy, or a model-agnostic implementation. See `IMPLEMENTATION_STATUS.md` for the exact scope.

## Corrections to make before experiments

1. **Floating-point numerical deltas are not generally lossless.** Encoding `H - H_old` losslessly does not make the inverse floating-point addition exact. Cancellation and rounding can change the recovered bits. Keep XOR of native byte representations as the exact delta mechanism, with entropy coding and fallback to anchors. A test in this package demonstrates failure of subtraction/addition inversion. Treat numerical residual quantization as approximate.

2. **The proposed error-feedback equation can double-count the previous quantization error.** When the sender already uses the receiver's reconstructed reference, `H_t - Hhat_(t-1)` includes the previous reconstruction error. Adding `e_(t-1)` again can overshoot. For a constant scalar target 1, previous reconstruction .9 and error .1, the stated equation asks to transmit .2, pushing reconstruction to 1.1. Use `u_t = H_t - Hhat_(t-1)`, with receiver addition and sender-side wire reconstruction, or use increments relative to the true previous tensor plus a separate accumulator. Do not combine both formulations. This implementation sends selected absolute row replacements; each next residual targets the reconstructed cache, so omitted/quantized error is naturally retained without an extra accumulator.

3. **Specify where sampling and the LM head execute.** Returning an `n × vocab_size` logit tensor can dwarf the uplink activation. The implemented split executes the LM head and deterministic token commitment in the cloud and returns compact next-token IDs. It sends current IDs/control state with each split request. Cloud-only keeps its entire denoising loop in cloud and sends one initial request plus one final response. Edge-only incurs no network traffic. These topologies must be stated; no split-activation privacy claim follows from sending IDs.

4. **Replace the one-way latency equation with both directions and explicit overhead.** For serial, batch-one split steps use

   `edge_compute + D2H + encode + uplink_bytes/B_up + RTT + decode + H2D + cloud_compute_and_sampler + downlink_bytes/B_down + edge_apply + switching`.

   RTT appears once per sequential exchange; additional retry round trips must be counted. Avoid adding a full RTT independently for uplink and downlink. Define bytes/s versus bits/s. The runtime assumes symmetric bandwidth for its analytical model and records application framing/telemetry bytes; these are not physical TCP/IP wire bytes. Use packet capture or interface counters for actual link bytes and retransmissions. Analytical network estimates and injected sleep are emulation, not traffic-controlled hardware measurements.

5. **Exact transport is different from exact model execution.** A bitwise boundary reconstruction test only proves the codec recovered the tensor the prefix actually produced. If the prefix reused approximate FFNs, that tensor can already differ from dense inference. Cross-device kernel choices, dtype conversion, TF32, nondeterministic attention, and reduction order can also change logits. Gate adapter and sampler parity, then test paired output identity for the actual deployment. Mathematical equivalence does not imply bitwise equality across heterogeneous GPUs.

6. **Drift-based selection has an observation cost.** Computing the dense prefix and then testing every boundary row saves communication, not prefix computation. Propagating an active set through Transformer layers requires deciding updates before paying for them. Dense attention maps cost time and quadratic storage; they cannot be assumed free. The included compute ablation skips selected FFN rows while retaining dense attention and is labeled approximate. It must include gather/scatter, audits and cache memory in measured latency.

7. **Random audits of a local operator do not certify the whole hidden state.** Recomputing an FFN on a drifted cached input cannot reveal all errors caused earlier in the network. A dense reference branch at occasional checkpoints is needed for full hidden/logit KL audits. The FFN audit implemented here checks local cache error and can trigger a full refresh of that operator; it is not a global quality bound. Anchor refresh of communication alone must not be described as full model-state recovery.

8. **An active-row cap can conflict with mandatory seeds and guards.** The changed-token set may exceed a 5% budget. Define fallback: keep mandatory rows, expand the budget or refresh densely. The runtime never drops mandatory changed rows merely to meet a percentage. The boundary codec's exact drift selection and the FFN update set are separate quantities, not interchangeable cache-hit rates.

9. **Training-free still allows calibration, but calibrated risk is empirical.** A table fitted on calibration questions is not a formal bound on test-set quality. Split calibration and evaluation IDs. Select thresholds using paired task-score loss confidence intervals, then evaluate once on held-out questions. Thirty latency queries alone are insufficient to certify a small quality loss on hard reasoning tasks. A one-sided noninferiority interval and a prespecified margin are better than claiming preservation from a nonsignificant difference.

10. **Adaptive split movement needs real memory and weight-transfer costs.** A prefix for one split may be CPU/GPU resident while another is not. Count the union of resident candidate weights, embedding/head placement, caches, temporary tensors, and switch/reset cost. The supplied live runtime adapts codecs at a fixed loaded split. Full layer-placement adaptation is represented only by measured-table trace replay; replay is not evidence of a deployed migrating runtime. A future migration implementation must also measure fallback/recovery behavior.

11. **Do not conflate sampler improvements with deployment improvements.** Hold steps, block length, generation length, commit rule, dtype, prompts and checkpoint fixed for the main split experiment. Run Fast-dLLM v1 caching alone before its parallel-decoding combination. Current Fast-dLLM v2 uses fine-tuning and is not a same-weight training-free baseline. FlashDLM's guidance requires an auxiliary AR model, so its total resource use belongs in a separate comparison. [Fast-dLLM repository](https://github.com/NVlabs/Fast-dLLM), [FlashDLM](https://openreview.net/pdf/0c6df3fff98929b22a3a729f467b6ab6951e4a42.pdf).

12. **Narrow the first reproducible experiment.** On Colab, start with LLaDA, batch one, deterministic low-confidence commitment and no classifier-free guidance. Use a CPU prefix/GPU suffix and clearly label the edge as emulated. Dream needs its own adapter and sampler parity check; replacing it with LLaDA's sampler is invalid. A full native-precision 8B checkpoint plus activations can exceed a 16 GB GPU. Report cloud/edge-only configurations as infeasible if they do not fit. Weight quantization changes the model arithmetic and requires a separately matched quantized reference.

13. **Separate system metrics from task metrics.** In a masked full-canvas generator, committed positions can be noncontiguous, so AR time-to-first-token and TPOT are not automatically meaningful. Primary metrics should be completed-response latency, effective output tokens/s, NFE, and time until a defined stable/contiguous output prefix. Token agreement/edit distance are diagnostics, not task quality. GSM8K numeric parsing is not mathematical-expression equivalence. MMLU likelihood scoring and generation-based choice scoring are different protocols. HumanEval/MBPP require an execution evaluator and correctly sampled pass@k; no code is executed by this pilot scorer.

14. **Memory, FLOPs and energy need measurement boundaries.** Count weights as well as activation and per-layer feature caches. PyTorch allocator peaks cover GPU tensors, not complete device memory; CPU RSS is process memory, not isolated edge peak memory. Skipped FFN row counts are not measured FLOPs. Use a separate operator profiler/analytic operation model with kernel coverage stated. Colab does not provide trustworthy edge-device energy instrumentation. Record unavailable FLOPs/energy as null, not zero or power × GPU-time guesses. Physical energy needs idle-subtracted GPU power integration and/or external device meters, with synchronization and sampling accuracy stated.

15. **The novelty claim must stay corpus-bounded.** The screen found recent serving, edge and codec work beyond the initial backbone papers: [Sangam](https://arxiv.org/abs/2607.04206), the [edge DLM survey](https://arxiv.org/abs/2609.04778), and [edge block-diffusion hardware](https://arxiv.org/abs/2609.01084). RAC is a strong codec comparator, but its local-cloud-local causal layout differs from this experiment. Do not report our simple sparse replacement codec as a reproduction of RAC. [RAC](https://arxiv.org/abs/2608.04991).

16. **Constrain LaCache exactness to its demonstrated invariant region.** Its detailed method restricts exact pre-attention and attention-tile memoization to the first layer; deeper hidden-state caches are approximate. Its FP8 component is also separate from exact caching. Do not cite it as proof of arbitrary deep token-row invariance. Its different reported caching-only versus combined numbers should be reproduced rather than imported as a split-runtime speedup. [LaCache method and limitations](https://arxiv.org/html/2607.16339v1).

## RQ-to-experiment mapping

| RQ | Concrete measurement | Requirement still needed for a paper-level claim |
| --- | --- | --- |
| RQ1 drift | `--trace-drift` at independent candidate splits; per-step changed-token and per-row drift logs | Add layer/prompt/masked-region aggregation over task-diverse queries; instrumentation runs separate from timed runs |
| RQ2 exact communication | Full vs zlib vs XOR, same sampler; CRC/version, native-bit and generated-ID checks | Native pretrained parity and actual deployment output identity; physical traffic accounting if claiming wire bytes |
| RQ3 selective compute | FFN-only row cache, active fraction/refresh/audit sweep | Task-quality CIs and wall-clock gain; general selective attention requires kernel work not delivered here |
| RQ4 adaptation | Measured profile generation; live fixed-split codec scheduler; split-table replay | Stage-conditioned profiles, prediction-error reporting, and an actual migrating split runtime before full placement claims |
| RQ5 transfer | Explicit LLaDA/Dream adapters and separate validation command | Run both models and standardized math, choice, code and long-context evaluators; a single GSM8K pilot cannot answer RQ5 |

## LaTeX/template discrepancies

- Two latency terms contain `widehat` without a leading backslash (`encode` and `cloud`); fix to `\widehat`.
- The main table declares `llcccccccccc` (12 columns) but the header lists 11. Use `llccccccccc` or add the missing column deliberately.
- Replace the IEEE example acknowledgments, dates, publication number, running headers and generic keywords with manuscript-specific text.
- This attachment starts at `\begin{document}` and contains no preamble, macro definitions, bibliography or `\end{document}`. That may be an excerpt rather than the complete Overleaf project. Verify definitions for `\method`, `\mask`, `\todoresult`, booktabs/graphics packages, all bibliography keys, and the architecture asset. Compilation cannot be established from this excerpt alone.
- Clarify whether anchors increment a version within a query or start a new query epoch. The draft says both monotonically increasing and resets version. Use `(query_id, boundary_id, version)` and reset only for an explicit new query/epoch.
- Keep numerical result tables blank until experiments run. Toy outputs, modeled network curves and trace replay must not populate physical-hardware result rows.

## Recommended first study

Run protocol tests, native adapter/sampler validation, a short functional LLaDA sweep, then a 30+ query systems pilot. Use disjoint task-diverse calibration/evaluation IDs and longer standardized benchmark outputs for quality. Validate exact communication first. Only then enable FFN caching and adaptation. If bytes fall but encoding/copying raises latency, report that result directly; it is a useful answer to RQ2.
