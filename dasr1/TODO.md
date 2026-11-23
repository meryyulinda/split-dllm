# To do

## Now
- [ ] **Exp13: Fast-dLLM on the split** (DualCache + parallel decoding): smoke test, full run, add results to the paper (RQ2 bytes, latency vs. round trips; new γ trade-off).

## For a strong SIGCOMM submission
- [ ] **(a) Show when splitting is the right choice.** Cloud-only is faster in every current result, so quantify a regime where splitting wins: cloud GPU time / cost per answer with several concurrent users (cloud throughput), keeping the raw prompt on the device (plus a basic look at what the boundary activations leak), and cutting the per-step round-trip floor with parallel decoding (Exp13).
- [ ] **(b) Real network instead of the analytical model.** Run the prototype over a shaped link (Linux `tc`/netem or Mahimahi with Wi-Fi/LTE traces), measure real end-to-end latency, overlap transfer with computation; ideally a real edge device (Jetson Orin or laptop GPU) instead of a 3x-slowed A6000.
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
