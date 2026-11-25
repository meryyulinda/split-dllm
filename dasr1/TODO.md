# To do

## Now
- [x] **Exp13: Fast-dLLM on the split** (DualCache + parallel decoding): `outputs/exp13_*_b32/`. Split ~10x faster than the static split, 2.0-2.4x cloud-only, accuracy unchanged.
- [x] **Exp14 / Exp14b: block-start row reuse** (`EXP14_NOTES.md`, `EXP14B_NOTES.md`; `outputs/exp13_*_reuse/`, `_reuse2/`): truncates long answers even when only prompt rows are reused -> negative result, removed from the paper. Final Fast-dLLM configuration: DualCache + gamma 0.9 + INT8.
- [x] **Exp13C: split exactness of Fast-dLLM** (`outputs/exp13c_exactness_dream_gsm8k_k4_t256/`): 30/30 identical to single-GPU Fast-dLLM -> in the paper (Fast-dLLM rows are exact tier).
- [x] **Exp15: boundary leakage** (`outputs/exp15_leakage_dream/`, preliminary): at layer 4, 78-89% of private prompt tokens and ~100% of numbers recoverable; a deeper split only stops the untrained attack (linear inverter still 41-77% at layer 24). Decision (25 Nov): privacy is not a motivation of the paper; it appears only as a limitation (one sentence in Threats) and as future research, without new citations. Split layer stays at 4. Optional, only if privacy becomes a topic: rerun with `pip install datasets` (1,800 training prompts) and on LLaDA.
- [x] Paper terms confirmed (25 Nov): dLLM-Cache backend / Fast-dLLM backend; exact / approximate tier = what the edge sends; backend selector, row selector, step schedule, block-start step, in-block step; readable configuration names (no codes); no "transmission"; block length n_block.
- [ ] Paper: **verify the claims about other caches** in the related-work paragraph ("why dLLM-Cache and Fast-dLLM") against the papers: Sparse-dLLM, d2Cache and Elastic-Cache select tokens per layer from attention or drift statistics; dKV-Cache is in the key/value-caching family; FlashDLM needs an auxiliary autoregressive model. These come from Claude's reading, not from checking the papers.
- [ ] Architecture figure (teammate redraw): brief `architecture_brief_for_teammate.md`, all labels confirmed, split layer 4.
- [x] **Exp16: real network** (`EXP16_NOTES.md`; `outputs/exp16_{dllm_cache,fastdllm}_dream_gsm8k_k4_t256/`): split over real TCP through a network emulator (relay processes; 10/100/1000 Mbit/s, RTT 20 ms, s = 1). All 116 networked answers identical to the runs without network. Measured time = latency model on the same run +2.8-3.9% (max 4.0%), explained to within 0.4% by 2.5-3.1 ms per step (host copies, sockets) + 2.7% TCP/IP header bytes. Vs the runs without network (what the paper's latencies use): +2.7-9.7%, because compute of the cached configurations is up to 11% slower in the networked runs (cause not checked). Ranking unchanged. In the paper: RQ2 paragraph, Threats, future work.
- [x] Paper term confirmed (25 Nov): "network emulator" for the relay (common term, not defined).
- [ ] **Exp17: offloading** (`EXP17_NOTES.md`, `run_exp17.sh`): code ready. Cloud GPU time and energy per answer, split vs cloud-only (same cache and decoding), for split layers 2/4/8/14/20, both backends (bf16 / INT8), with what the edge pays (time, memory, latency ratio). Next: smoke test, then the full run on idle GPUs (~40 min). Then: does the offloading argument hold at k=4, or should the paper pick a deeper split? Paper terms for the new quantities (e.g. "cloud GPU time per answer", "offloaded share") to be confirmed with Mery before writing.

## For a strong SIGCOMM submission
- [ ] **(a) Show when splitting is the right choice.** Cloud-only is faster in every result, and privacy is no longer an argument (Exp15). Quantify the offloading argument instead: cloud GPU-seconds and energy per answer for the split vs cloud-only, and how many concurrent users one cloud GPU serves when the edge runs the first layers (Exp17 = per-answer cost vs split layer; a multi-user run with batching is still open).
- [ ] **(b) Real network instead of the analytical model.** Exp16 done (fixed-rate network emulator, same server). Still open: Wi-Fi/LTE trace replay (the relay already takes a trace CSV), lossy links with `tc`/netem or Mahimahi on a machine we control, overlap of transfer with computation, and a real edge device (Jetson Orin or laptop GPU) instead of a 3x-slowed A6000.
- [ ] **(c) Live network-aware controller.** Switch the codec (exact / INT8) and Fast-dLLM's threshold γ online from the measured bandwidth; evaluate on bandwidth traces.
- [ ] **(d) Stronger baselines.** Existing split-inference compression, e.g. RAC-style residual quantization and SmoothQuant-style activation quantization, not only our own static split.
- [ ] **(e) Breadth.** More tasks (HumanEval, MMLU, LongBench), several seeds, larger samples than 30 + 12 prompts.
- [ ] **(f) Later: privacy follow-up.** Run the cloud part under GPU confidential computing (H100 CC; keeps outputs exact) or design step-aware noise for dLLM boundaries; measure per-step overhead.

## Paper format for SIGCOMM
- [ ] ACM conference template, ≤ 12 pages body (references/appendices extra), letter size, 10 pt.
- [ ] Double-blind: no names, affiliations, acknowledgments or repo links; neutral file names.
- [ ] Ethics statement in the main body (desk reject without one).
- [ ] Abstract ≤ 200 words; submit the .bbl file; disclose generative-AI use.
- [ ] Deadline: SIGCOMM 2027 CFP not out yet (2026 cycle: abstract Jan 30, paper Feb 6).

## Smaller items
- [ ] Paper: choose the adaptive-refresh wording (original / condensed / middle); remove the `geometry` line; add `cobbe2021gsm8k` to the bib.
- [ ] Repo: replace README with the short version; delete `exp11c_save_emulation.patch`, `untitled.ipynb`, `try_llada.py`, old logs.
- [ ] Optional: confirm that the `llada` and `dream` environments give identical LLaDA text (Exp12A, 3 questions).
- [ ] Later: cost-aware refresh rule (weigh prompt/answer refreshes by their share).
