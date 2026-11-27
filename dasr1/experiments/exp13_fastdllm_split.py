"""Exp13: Fast-dLLM (block KV cache + parallel decoding) on the two-GPU split.

Run from dasr1/:  CUDA_VISIBLE_DEVICES=1,2 python experiments/exp13_fastdllm_split.py

Question: how much do Fast-dLLM's two mechanisms change the cost of splitting?
  * DualCache: inside a block only the block's rows are computed, so the edge
    sends `block_length` rows per step instead of the whole sequence (the paper's
    row rule S_t = R_t u L_t with the block as R_t).
  * Parallel decoding (confidence threshold gamma): fewer denoising steps, hence
    fewer network round trips -- the floor that keeps any split behind cloud-only.

Variants (all on the same split, same prompts and seeds; blocks of EXP13_BLOCK_LENGTH):
  F0_nocache_fixed       no cache, 1 token per step, full boundary  (reference)
  F1_nocache_thr<g>      no cache, parallel decoding (gamma = first threshold)
  K1_dual_fixed          DualCache, 1 token per step
  K2_dual_thr<g>         DualCache + parallel decoding, one variant per threshold
  K3_dual_thr<g>_int8    K2 (first threshold) with the INT8 delta codec
  K4_dual_thr<g>_int8abs / K5_dual_thr<g>_int4   Exp19 baselines: K2 with plain per-row
                         INT8 / group-wise INT4 quantization of the rows (no delta)
Exp14, block-start row reuse (same script, run_exp14.sh selects these):
  R1_dual_thr<g>_reuse       K2 + row reuse at every block start
  R2_dual_thr<g>_reuse_r<m>  R1 with all rows sent at every m-th block start
  R3_dual_thr<g>_reuse_int8  R1 + INT8 delta for the rows that are sent
  R4_dual_thr<g>_reuse_prompt       reuse prompt rows only (every answer row is sent)
  R5_dual_thr<g>_reuse_prompt_int8  R4 + INT8 delta for the rows that are sent
  Row reuse is also compared with K2 (cache_ref): same_as_cache_ref, flips, McNemar.
Quality is compared with F0 (paired flips, exact McNemar test). Timing and bytes
are measured per step; exp11b's emulation is applied at the end. Every variant also
gets its own cloud-only reference (same cache and decoding, all layers on the cloud
GPU, one round trip), so a split is always compared with cloud-only under the same
policy: rows "cloud_only:<variant>" and column overhead_vs_own_cloud.
Sampler: greedy for both models (Fast-dLLM's evaluation setting).

Outputs: outputs/exp13_<model>_<dataset>_k<split>_t<tokens>_b<block><tag>/
  runs.csv, steps.csv, summary_by_variant.csv, emulated_summary.csv, metadata.json

Needs a Fast-dLLM checkout (FASTDLLM_REPO, default ../../Fast-dLLM) because the
DualCache path uses Fast-dLLM's modified Dream / LLaDA model code.
"""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from exp10_common import ROOT, cfg, default_score, describe, mcnemar_exact_p  # noqa: E402
import exp11b_network_emulation as emu  # noqa: E402
from src.fastdllm_split import (  # noqa: E402
    DREAM_MODEL_NAME, LLADA_MODEL_NAME, KVBlockLink, kv_generate, load_fastdllm_model,
)
from src.gpu_meter import EnergyMeter, measure_idle_w  # noqa: E402
from src.split_runtime import place_split  # noqa: E402


def setting(name, default):
    return getattr(cfg, name, default)


MODEL = str(setting("EXP13_MODEL", "dream")).lower()
if MODEL not in {"dream", "llada"}:
    raise ValueError(f"EXP13_MODEL must be 'dream' or 'llada', not {MODEL!r}")
DATASET = setting("EXP13_DATASET", "gsm8k")
N = int(setting("EXP13_N_QUESTIONS", 30))
NEW_TOKENS = int(setting("EXP13_NEW_TOKENS", {"gsm8k": 256, "authored": 128, "hotpotqa": 128}[DATASET]))
BLOCK = int(setting("EXP13_BLOCK_LENGTH", 32))
THRESHOLDS = [float(g) for g in setting("EXP13_THRESHOLDS", [0.9])]
SPLIT = int(setting("EXP13_SPLIT_LAYER", 4))
EDGE = setting("EXP13_EDGE_DEVICE", "cuda:1")
CLOUD = setting("EXP13_CLOUD_DEVICE", "cuda:0")
if MODEL == "dream":
    # Fast-dLLM's own Dream setting: greedy, no top-p. The threshold is compared with
    # the token's probability *after* temperature / top-p, so T=0.1 + top-p 0.9 pushes
    # almost every position above 0.9 and parallel decoding commits garbage.
    TEMPERATURE = float(setting("EXP13_DREAM_TEMPERATURE", 0.0))
    TOP_P = setting("EXP13_DREAM_TOP_P", None)
    MODEL_NAME = setting("MODEL_NAME", DREAM_MODEL_NAME)
else:
    TEMPERATURE = float(setting("EXP13_LLADA_TEMPERATURE", 0.0))
    TOP_P = None
    MODEL_NAME = setting("LLADA_MODEL_NAME", LLADA_MODEL_NAME)
ANCHOR = int(setting("EXP13_ANCHOR_INTERVAL", 32))
REUSE_REFRESH = int(setting("EXP14_REFRESH_EVERY", 4))
SEED = int(setting("SEED", 42))
REPO = setting("FASTDLLM_REPO", "../../Fast-dLLM")
RESUME = bool(setting("EXP13_RESUME", True))
OUTPUT_TAG = str(setting("EXP13_OUTPUT_TAG", ""))
IDLE_SECONDS = float(setting("EXP13_IDLE_SECONDS", 5.0))
OUT = ROOT / "outputs" / f"exp13_{MODEL}_{DATASET}_k{SPLIT}_t{NEW_TOKENS}_b{BLOCK}{OUTPUT_TAG}"
GIB = 2 ** 30
REFERENCE = "F0_nocache_fixed"


def V(name, cache, decode, threshold=None, encoding="bf16", reuse=False, refresh_every=0,
      reuse_scope="all"):
    return {"name": name, "cache": cache, "decode": decode, "threshold": threshold,
            "encoding": encoding, "reuse": reuse, "refresh_every": refresh_every,
            "reuse_scope": reuse_scope}


def gname(g):
    return f"{g:g}"


g0 = THRESHOLDS[0]
ALL_VARIANTS = [
    V(REFERENCE, "none", "fixed"),
    V(f"F1_nocache_thr{gname(g0)}", "none", "threshold", g0),
    V("K1_dual_fixed", "dual", "fixed"),
    *[V(f"K2_dual_thr{gname(g)}", "dual", "threshold", g) for g in THRESHOLDS],
    V(f"K3_dual_thr{gname(g0)}_int8", "dual", "threshold", g0, "int8_delta"),
    # Exp19 baselines: plain quantization of the rows (no delta, no anchors)
    V(f"K4_dual_thr{gname(g0)}_int8abs", "dual", "threshold", g0, "int8_abs"),
    V(f"K5_dual_thr{gname(g0)}_int4", "dual", "threshold", g0, "int4_group"),
    V(f"R1_dual_thr{gname(g0)}_reuse", "dual", "threshold", g0, reuse=True),
    V(f"R2_dual_thr{gname(g0)}_reuse_r{REUSE_REFRESH}", "dual", "threshold", g0, reuse=True,
      refresh_every=REUSE_REFRESH),
    V(f"R3_dual_thr{gname(g0)}_reuse_int8", "dual", "threshold", g0, "int8_delta", reuse=True),
    V(f"R4_dual_thr{gname(g0)}_reuse_prompt", "dual", "threshold", g0, reuse=True,
      reuse_scope="prompt"),
    V(f"R5_dual_thr{gname(g0)}_reuse_prompt_int8", "dual", "threshold", g0, "int8_delta",
      reuse=True, reuse_scope="prompt"),
]
CACHE_REF = f"K2_dual_thr{gname(g0)}"


def load_cases():
    if DATASET == "gsm8k":
        from exp10d_gsm8k_refresh import gsm8k_score, load_cases as gsm8k_cases
        return gsm8k_cases(n=N, start=0), gsm8k_score
    if DATASET == "authored":
        from exp10a_official_feature_cache_quality import make_cases
        return make_cases()[:N], default_score
    if DATASET == "hotpotqa":
        from src.hotpotqa import hotpot_score, load_hotpotqa
        path = Path(getattr(cfg, "HOTPOTQA_DATA", ROOT / "data" / "hotpotqa" / "dev_distractor_first100.jsonl"))
        return load_hotpotqa(path, n=N), hotpot_score
    raise ValueError(f"EXP13_DATASET must be 'gsm8k', 'authored' or 'hotpotqa', not {DATASET!r}")


def prepare(tokenizer, prompt, device):
    if MODEL == "llada":
        from src.llada_cache_ext import llada_prepare_input
        return llada_prepare_input(tokenizer, prompt, device)
    from exp9b_official_feature_cache_diagnostics import prepare_input
    return prepare_input(tokenizer, prompt, device)


def check_output_folder(names):
    runs_csv = OUT / "runs.csv"
    if not runs_csv.exists():
        return
    prev = set(pd.read_csv(runs_csv, usecols=["variant"])["variant"])
    if prev != set(names):
        raise SystemExit(f"{runs_csv} holds variants {sorted(prev)}, but this run asks for "
                         f"{sorted(names)}. Set EXP13_OUTPUT_TAG to write to a new folder.")
    if not RESUME:
        raise SystemExit(f"{runs_csv} exists and EXP13_RESUME=False; set another EXP13_OUTPUT_TAG.")


def run_summary(steps):
    df = pd.DataFrame(steps)
    return {
        "n_steps": len(df), "tokens_per_step": NEW_TOKENS / max(len(df), 1),
        "block_start_steps": int((df["kind"] == "full").sum()),
        "edge_ms": df["edge_ms"].sum(), "encode_ms": df["encode_ms"].sum(),
        "cloud_ms": df["cloud_ms"].sum(), "sample_ms": df["sample_ms"].sum(),
        "up_mb": df["up_bytes"].sum() / 1e6, "full_mb": df["full_bytes"].sum() / 1e6,
        "down_kb": df["down_bytes"].sum() / 1e3, "mean_rows_sent": df["rows_sent"].mean(),
        "block_start_mb": df.loc[df["kind"] == "full", "up_bytes"].sum() / 1e6,
        "rows_reused": int(df["rows_reused"].sum()) if "rows_reused" in df else 0,
    }


def summarize(runs):
    df = pd.DataFrame(runs)
    for col in ("ok", "ref_ok", "same_as_ref", "cache_ref_ok", "same_as_cache_ref"):
        if col in df:
            df[col] = df[col].map(lambda v: str(v).strip().lower() == "true")
    df["regression"] = df["ref_ok"] & ~df["ok"]
    df["improvement"] = ~df["ref_ok"] & df["ok"]
    ref_up = df[df["variant"] == REFERENCE].set_index("case_id")["up_mb"]
    df["up_reduction_vs_ref"] = 1 - df["up_mb"] / df["case_id"].map(ref_up)
    cols = dict(
        runs=("case_id", "size"), acc=("ok", "mean"), ref_acc=("ref_ok", "mean"),
        regressions=("regression", "sum"), improvements=("improvement", "sum"),
        same_as_ref=("same_as_ref", "sum"), mean_steps=("n_steps", "mean"),
        mean_tokens_per_step=("tokens_per_step", "mean"), mean_up_mb=("up_mb", "mean"),
        up_reduction_vs_ref=("up_reduction_vs_ref", "mean"),
        mean_edge_ms=("edge_ms", "mean"), mean_cloud_ms=("cloud_ms", "mean"),
        mean_sample_ms=("sample_ms", "mean"),
    )
    for c in ("edge_peak_gib", "cloud_peak_gib", "edge_energy_j", "cloud_energy_j", "wall_s"):
        if c in df:
            cols[f"mean_{c}"] = (c, "mean")
    if "block_start_mb" in df:
        cols["mean_block_start_mb"] = ("block_start_mb", "mean")
    if "cache_ref_ok" in df:  # flips against K2 (DualCache, first threshold, no reuse)
        df["reg_vs_cache_ref"] = df["cache_ref_ok"] & ~df["ok"]
        df["imp_vs_cache_ref"] = ~df["cache_ref_ok"] & df["ok"]
        cols.update(same_as_cache_ref=("same_as_cache_ref", "sum"),
                    regressions_vs_cache_ref=("reg_vs_cache_ref", "sum"),
                    improvements_vs_cache_ref=("imp_vs_cache_ref", "sum"))
    out = df.groupby("variant", sort=False).agg(**cols).reset_index()
    out["mcnemar_p"] = [mcnemar_exact_p(r, i) for r, i in zip(out["regressions"],
                                                                out["improvements"])]
    if "regressions_vs_cache_ref" in out:
        out["mcnemar_p_vs_cache_ref"] = [
            mcnemar_exact_p(r, i) for r, i in zip(out["regressions_vs_cache_ref"],
                                                   out["improvements_vs_cache_ref"])]
    return out


def main():
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Exp13 needs two GPUs (edge + cloud)")
    names = list(setting("EXP13_VARIANT_NAMES", [v["name"] for v in ALL_VARIANTS]))
    by_name = {v["name"]: v for v in ALL_VARIANTS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown EXP13_VARIANT_NAMES {unknown}; known: {list(by_name)}")
    if REFERENCE not in names:
        raise ValueError(f"{REFERENCE} must be in EXP13_VARIANT_NAMES (quality reference)")
    variants = [v for v in ALL_VARIANTS if v["name"] in names]  # K2 runs before R*

    cases, score_fn = load_cases()
    check_output_folder(names)
    OUT.mkdir(parents=True, exist_ok=True)
    runs, done = [], set()
    if RESUME and (OUT / "runs.csv").exists():
        prev = pd.read_csv(OUT / "runs.csv", keep_default_na=False).to_dict("records")
        have = {}
        for r in prev:
            have.setdefault(r["case_id"], set()).add(r["variant"])
        done = {c for c, vs in have.items() if set(names) <= vs}
        runs = [r for r in prev if r["case_id"] in done]
        if (OUT / "steps.csv").exists():
            st = pd.read_csv(OUT / "steps.csv")
            st[st["case_id"].isin(done)].to_csv(OUT / "steps.csv", index=False)
        print(f"resume: {len(done)} finished cases kept", flush=True)
    elif (OUT / "steps.csv").exists():
        (OUT / "steps.csv").unlink()

    model, tokenizer = load_fastdllm_model(MODEL, MODEL_NAME, REPO)
    edge, cloud = place_split(model, SPLIT, EDGE, CLOUD)
    link = KVBlockLink(model, SPLIT, edge, cloud)
    meter = EnergyMeter({"edge": edge, "cloud": cloud})
    idle_w = measure_idle_w(meter, (edge, cloud), IDLE_SECONDS)
    if idle_w:
        print("idle power: " + ", ".join(f"{k}={v:.1f} W" for k, v in idle_w.items()), flush=True)
    print(f"Exp13 | {MODEL} | {DATASET} n={len(cases)} | split after layer {SPLIT} | "
          f"{NEW_TOKENS} tokens, blocks of {BLOCK} | thresholds {THRESHOLDS} | "
          f"T={TEMPERATURE} top_p={TOP_P}", flush=True)

    def run(v, ids):
        for d in (edge, cloud):
            torch.cuda.synchronize(d)
            torch.cuda.reset_peak_memory_stats(d)
        e0, t0 = (meter.read_j() if meter.handles else {}), time.perf_counter()
        res = kv_generate(model, MODEL, tokenizer, ids, NEW_TOKENS, BLOCK, SEED,
                          cache=v["cache"], decode=v["decode"], threshold=v["threshold"],
                          temperature=TEMPERATURE, top_p=TOP_P, link=link,
                          link_cfg={"encoding": v["encoding"], "anchor_interval": ANCHOR,
                                    "reuse": v["reuse"], "refresh_every": v["refresh_every"],
                                    "reuse_scope": v["reuse_scope"]},
                          device=cloud)
        for d in (edge, cloud):
            torch.cuda.synchronize(d)
        wall = time.perf_counter() - t0
        e1 = meter.read_j() if meter.handles else {}
        res["resources"] = {
            "wall_s": wall,
            "edge_peak_gib": torch.cuda.max_memory_allocated(edge) / GIB,
            "cloud_peak_gib": torch.cuda.max_memory_allocated(cloud) / GIB,
        }
        for side in ("edge", "cloud"):
            if side in e0:
                res["resources"][f"{side}_energy_j"] = e1[side] - e0[side]
        return res

    ids_w, _ = prepare(tokenizer, cases[0]["prompt"], cloud)
    for v in (by_name[REFERENCE], by_name[CACHE_REF] if CACHE_REF in by_name else variants[-1]):
        run(v, ids_w)
    print("warmup done", flush=True)

    for case in cases:
        if case["case_id"] in done:
            continue
        ids, _ = prepare(tokenizer, case["prompt"], cloud)
        plen = int(ids.shape[1])
        print(f"\n{case['case_id']} | prompt tokens={plen}", flush=True)
        results, case_rows, step_rows = {}, [], []
        for v in variants:
            res = run(v, ids)
            results[v["name"]] = res
            text = describe(tokenizer, res["generated_ids"])["text"]
            score = score_fn(case, text)
            ref = results.get(REFERENCE)
            cref = results.get(CACHE_REF)
            row = {
                "variant": v["name"], "case_id": case["case_id"], "task": case["task"],
                "prompt_tokens": plen, "cache": v["cache"], "decode": v["decode"],
                "threshold": v["threshold"], "encoding": v["encoding"],
                "gold_answer": case["gold_answer"], "text": text,
                "extracted": score.get("extracted"), "ok": bool(score["ok"]),
                "same_as_ref": bool(ref is not None and torch.equal(ref["generated_ids"],
                                                                    res["generated_ids"])),
                "same_as_cache_ref": bool(cref is not None and torch.equal(
                    cref["generated_ids"], res["generated_ids"])),
                **run_summary(res["split_steps"]), **res["resources"],
            }
            case_rows.append(row)
            for st in res["split_steps"]:
                step_rows.append({"case_id": case["case_id"], "variant": v["name"], **st})
            print(f"  {v['name']:<24s} ok={row['ok']!s:<5} steps={row['n_steps']:4d} "
                  f"up={row['up_mb']:7.2f} MB edge={row['edge_ms']:6.0f} ms "
                  f"cloud={row['cloud_ms']:6.0f} ms "
                  f"{'=K2 ' if v['reuse'] and row['same_as_cache_ref'] else ''}"
                  f"{text[-50:]!r}", flush=True)
        ref_ok = next(r["ok"] for r in case_rows if r["variant"] == REFERENCE)
        cref_ok = next((r["ok"] for r in case_rows if r["variant"] == CACHE_REF), None)
        for r in case_rows:
            r["ref_ok"] = ref_ok
            if cref_ok is not None:
                r["cache_ref_ok"] = cref_ok
        runs.extend(case_rows)
        pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)
        steps_path = OUT / "steps.csv"
        pd.DataFrame(step_rows).to_csv(steps_path, mode="a", index=False,
                                       header=not steps_path.exists())

    link.remove()
    summary = summarize(runs)
    summary.to_csv(OUT / "summary_by_variant.csv", index=False)
    steps = pd.read_csv(OUT / "steps.csv")
    em, em_summary = emu.emulate(steps, pd.DataFrame(runs).rename(columns={"ok": "cache_ok"}),
                                 reference=REFERENCE,
                                 cache_ref=CACHE_REF if CACHE_REF in names else REFERENCE,
                                 own_cloud=True)
    em.to_csv(OUT / "emulated_runs.csv", index=False)
    em_summary.to_csv(OUT / "emulated_summary.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "experiment": "Exp13 Fast-dLLM (DualCache + parallel decoding) on the two-GPU split",
        "model": MODEL_NAME, "arch": MODEL, "fastdllm_repo": str(REPO), "dataset": DATASET,
        "n_cases": len(cases), "split_layer_1based": SPLIT, "new_tokens": NEW_TOKENS,
        "block_length": BLOCK, "thresholds": THRESHOLDS, "temperature": TEMPERATURE,
        "top_p": TOP_P, "seed": SEED, "anchor_interval": ANCHOR, "variants": variants,
        "reference": REFERENCE, "cloud_only_cache_ref": CACHE_REF, "idle_power_w": idle_w,
        "edge_gpu": torch.cuda.get_device_name(edge), "cloud_gpu": torch.cuda.get_device_name(cloud),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
    }, indent=2, default=str), encoding="utf-8")
    print("\nExp13 completed. Output:", OUT)
    show = ["variant", "runs", "acc", "regressions", "improvements", "mcnemar_p",
            "mean_steps", "mean_up_mb", "up_reduction_vs_ref", "mean_edge_ms", "mean_cloud_ms"]
    if any(v["reuse"] for v in variants) and "same_as_cache_ref" in summary:
        show += ["mean_block_start_mb", "same_as_cache_ref", "regressions_vs_cache_ref",
                 "improvements_vs_cache_ref", "mcnemar_p_vs_cache_ref"]
    print(summary[show].to_string(index=False))
    view = em_summary[(em_summary.rtt_ms == 20) & (em_summary.edge_slowdown == 3.0)]
    split = view[~view.variant.str.startswith("cloud_only")]
    print("\nEmulated split latency (s), RTT 20 ms, edge 3x slower:")
    print(split.pivot(index="variant", columns="bandwidth_mbps",
                      values="mean_latency_s").round(1).to_string())
    own = view[view.variant.str.startswith("cloud_only:")].groupby("variant")["mean_latency_s"].first()
    print("\nCloud-only with the same cache and decoding (s):")
    print(own.rename(lambda v: v.split(":", 1)[1]).round(2).to_string())


if __name__ == "__main__":
    main()
