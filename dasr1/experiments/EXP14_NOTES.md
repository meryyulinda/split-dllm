# Exp14: block-start row reuse on top of Fast-dLLM

## Idea

In Exp13 (Fast-dLLM on the split), the **block-start step** of every block still sends every row (prompt + whole answer), and these steps are 55–77% of the remaining uplink. Between two block starts, however, only the tokens of the block that was just finished have changed.

Row reuse: at a block-start step the edge still computes every row (it needs them for its own cache), but sends only

* rows whose token changed since the cloud last received that row (in practice: the block just finished),
* rows the cloud has never received,
* the rows that predict the current block (for Dream one row earlier, because of the logit shift).

The cloud fills every other row from the copy it already holds. This is approximate in the same way as DualCache: the reused rows are up to one block out of date. `R2` bounds that by sending all rows at every 4th block start. In-block steps are unchanged (they send the 32 block rows).

| Variant | What it is |
|---|---|
| `F0_nocache_fixed` | quality reference (required by the script) |
| `K2_dual_thr0.9` | Exp13 baseline: DualCache + parallel decoding |
| `K3_dual_thr0.9_int8` | K2 + INT8 delta codec |
| `R1_dual_thr0.9_reuse` | K2 + row reuse at every block start |
| `R2_dual_thr0.9_reuse_r4` | R1, all rows at every 4th block start |
| `R3_dual_thr0.9_reuse_int8` | R1 + INT8 delta for the rows that are sent |

Code: `reuse` / `refresh_every` in `KVBlockLink` (`src/fastdllm_split.py`), variants in `experiments/exp13_fastdllm_split.py`. Only the transmitted bytes change; edge and cloud compute the same rows as K2.

## Run (from `dasr1/`)

```bash
SMOKE=1 SKIP_IDLE_CHECK=1 bash experiments/run_exp14.sh > exp14_smoke.log 2>&1   # 2 questions, 64 tokens
nohup bash experiments/run_exp14.sh > exp14.log 2>&1 &                            # full run, idle GPUs
```

## What to check in the smoke test

* `R1`–`R3` send fewer MB than `K2` / `K3` (column `mean_block_start_mb`);
* `R*` answers are readable; `=K2` marks answers identical to K2;
* steps of `R*` are close to K2's (same decoding).

## Outputs

`outputs/exp13_<model>_<dataset>_k4_t<tokens>_b32_reuse/`: same files as Exp13. `steps.csv` has `rows_reused`; `summary_by_variant.csv` adds `mean_block_start_mb` and the comparison with K2 (`same_as_cache_ref`, `regressions_vs_cache_ref`, `improvements_vs_cache_ref`, `mcnemar_p_vs_cache_ref`).
