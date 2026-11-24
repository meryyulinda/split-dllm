# Exp14b: clean rerun of Exp14 + prompt-only row reuse

See EXP14_NOTES.md for the idea and variants R1-R3.
## Exp14 first full run (outputs/exp13_*_reuse/)

* Row reuse works as intended for bytes: R1 cuts uplink vs K2 by 40-51%, R3 (+INT8) by 66-68%.
* Accuracy: no significant change on GSM8K (Dream, LLaDA), but on the long prompts R1/R2 lose 3 of 12 answers vs K2. The lost answers stop early (mean answer length 207 vs 234 characters): the reused rows of the future, still-masked blocks keep the cloud's "short answer" state from earlier blocks.
* R2 (refresh every 2nd block start) does not fix the long prompts.
* Timings of the long-prompt and LLaDA workloads were disturbed (a llama-server on GPU 0, the cloud GPU, was active ~35% of the time); Dream GSM8K is clean.

Exp14b (`experiments/run_exp14b.sh`, outputs `..._reuse2/`) reruns on GPUs 1,2 and adds

| Variant | What it is |
|---|---|
| `R4_dual_thr0.9_reuse_prompt` | reuse prompt rows only; every answer row is sent at block start |
| `R5_dual_thr0.9_reuse_prompt_int8` | R4 + INT8 delta for the rows that are sent |

If R4 keeps K2's accuracy, prompt rows are safe to reuse and the stale future answer rows caused the losses. Texts, steps and bytes of F0/K2/K3/R1/R3 are deterministic and are reproduced; only the timings change.

```bash
SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp14b.sh > exp14b_smoke.log 2>&1
nohup bash experiments/run_exp14b.sh > exp14b.log 2>&1 &
```
