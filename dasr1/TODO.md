# To do

## Now
- [x] **Exp13: Fast-dLLM on the split** (DualCache + parallel decoding): full run done (`outputs/exp13_*_b32/`). Split ~10x faster than the static split, ~2x cloud-only, accuracy unchanged.
- [x] **Exp14: block-start row reuse** (`EXP14_NOTES.md`, `EXP14B_NOTES.md`): full run done (`outputs/exp13_*_reuse/`); bytes -40-51% vs K2, long prompts lose 3/12 (answers stop early); long-prompt/LLaDA timings disturbed.
- [ ] **Exp14b**: clean rerun on GPUs 1,2 + prompt-only reuse (R4, R5), `experiments/run_exp14b.sh`.
- [x] Paper draft v4 (25 Nov): five RQs with Fast-dLLM in RQ2-RQ4, "block" defined, preliminary experiments removed, no appendix, architecture figure without equations, why dLLM-Cache + Fast-dLLM.
- [ ] Paper: fill the Exp14b placeholders (`\hl{--}` in the Fast-dLLM table: R1/R3 latency, all R4 numbers; R4 sentence in RQ3).
- [ ] Paper: **verify the claims about other caches** in the new related-work paragraph ("why dLLM-Cache and Fast-dLLM") against the papers: that Sparse-dLLM, d²Cache and Elastic-Cache select the tokens to update per layer from attention or drift statistics (so the choice is made inside the cloud's layers), that dKV-Cache belongs to the key/value-caching family, and that FlashDLM needs an auxiliary autoregressive model. These come from Claude's reading, not from checking the papers.
- [ ] **Leakage check for the privacy argument**: how much of the prompt can be recovered from the layer-4 boundary states (train a small inverter), and at deeper splits.

## For a strong SIGCOMM submission
- [ ] **(a) Show when splitting is the right choice.** Cloud-only is faster in every current result, so quantify a regime where splitting wins: cloud GPU time / cost per answer with several concurrent users (cloud throughput), keeping the raw prompt on the device (plus a basic look at what the boundary activations leak), and cutting the per-step round-trip floor with parallel decoding (Exp13).
- [ ] **(b) Real network instead of the analytical model.** Run the prototype over a shaped link (no sudo on the shared server, so a user-space relay that delays and rate-limits the edge-cloud socket, replaying Wi-Fi/LTE traces; `tc`/netem or Mahimahi only on a machine we control), measure real end-to-end latency, overlap transfer with computation; ideally a real edge device (Jetson Orin or laptop GPU) instead of a 3x-slowed A6000.
- [ ] **(c) Live network-aware controller.** Switch the codec (exact / INT8) and Fast-dLLM's threshold γ online from the measured bandwidth; evaluate on bandwidth traces.
- [ ] **(d) Stronger baselines.** Existing split-inference compression, e.g. RAC-style residual quantization and SmoothQuant-style activation quantization, not only our own static split.
- [ ] **(e) Breadth.** More tasks (HumanEval, MMLU, LongBench), several seeds, larger samples than 30 + 12 prompts.

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
