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
Quality is compared with F0 (paired flips, exact McNemar test). Timing and bytes
are measured per step; exp11b's emulation is applied at the end (reference F0,
cloud-only with cache = the first K2 variant).

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
NEW_TOKENS = int(setting("EXP13_NEW_TOKENS", {"gsm8k": 256, "authored": 128}[DATASET]))
BLOCK = int(setting("EXP13_BLOCK_LENGTH", 32))
THRESHOLDS = [float(g) for g in setting("EXP13_THRESHOLDS", [0.9])]
SPLIT = int(setting("EXP13_SPLIT_LAYER", 4))
EDGE = setting("EXP13_EDGE_DEVICE", "cuda:1")
CLOUD = setting("EXP13_CLOUD_DEVICE", "cuda:0")
if MODEL == "dream":
    TEMPERATURE = float(setting("EXP13_DREAM_TEMPERATURE", 0.1))
    TOP_P = setting("EXP13_DREAM_TOP_P", 0.9)
    MODEL_NAME = setting("MODEL_NAME", DREAM_MODEL_NAME)
else:
    TEMPERATURE = float(setting("EXP13_LLADA_TEMPERATURE", 0.0))
    TOP_P = None
    MODEL_NAME = setting("LLADA_MODEL_NAME", LLADA_MODEL_NAME)
ANCHOR = int(setting("EXP13_ANCHOR_INTERVAL", 32))
SEED = int(setting("SEED", 42))
REPO = setting("FASTDLLM_REPO", "../../Fast-dLLM")
RESUME = bool(setting("EXP13_RESUME", True))
OUTPUT_TAG = str(setting("EXP13_OUTPUT_TAG", ""))
IDLE_SECONDS = float(setting("EXP13_IDLE_SECONDS", 5.0))
OUT = ROOT / "outputs" / f"exp13_{MODEL}_{DATASET}_k{SPLIT}_t{NEW_TOKENS}_b{BLOCK}{OUTPUT_TAG}"
GIB = 2 ** 30
REFERENCE = "F0_nocache_fixed"


def V(name, cache, decode, threshold=None, encoding="bf16"):
    return {"name": name, "cache": cache, "decode": decode, "threshold": threshold,
            "encoding": encoding}


def gname(g):
    return f"{g:g}"


g0 = THRESHOLDS[0]
ALL_VARIANTS = [
    V(REFERENCE, "none", "fixed"),
    V(f"F1_nocache_thr{gname(g0)}", "none", "threshold", g0),
    V("K1_dual_fixed", "dual", "fixed"),
    *[V(f"K2_dual_thr{gname(g)}", "dual", "threshold", g) for g in THRESHOLDS],
    V(f"K3_dual_thr{gname(g0)}_int8", "dual", "threshold", g0, "int8_delta"),
]
CACHE_REF = f"K2_dual_thr{gname(g0)}"


def load_cases():
    if DATASET == "gsm8k":
        from exp10d_gsm8k_refresh import gsm8k_score, load_cases as gsm8k_cases
        return gsm8k_cases(n=N, start=0), gsm8k_score
    if DATASET == "authored":
        from exp10a_official_feature_cache_quality import make_cases
        return make_cases()[:N], default_score
    raise ValueError(f"EXP13_DATASET must be 'gsm8k' or 'authored', not {DATASET!r}")


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
    }


def summarize(runs):
    df = pd.DataFrame(runs)
    for col in ("ok", "ref_ok", "same_as_ref"):
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
    out = df.groupby("variant", sort=False).agg(**cols).reset_index()
    out["mcnemar_p"] = [mcnemar_exact_p(r, i) for r, i in zip(out["regressions"],
                                                                out["improvements"])]
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
    variants = [by_name[n] for n in names]

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
                          link_cfg={"encoding": v["encoding"], "anchor_interval": ANCHOR},
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
            row = {
                "variant": v["name"], "case_id": case["case_id"], "task": case["task"],
                "prompt_tokens": plen, "cache": v["cache"], "decode": v["decode"],
                "threshold": v["threshold"], "encoding": v["encoding"],
                "gold_answer": case["gold_answer"], "text": text,
                "extracted": score.get("extracted"), "ok": bool(score["ok"]),
                "same_as_ref": bool(ref is not None and torch.equal(ref["generated_ids"],
                                                                    res["generated_ids"])),
                **run_summary(res["split_steps"]), **res["resources"],
            }
            case_rows.append(row)
            for st in res["split_steps"]:
                step_rows.append({"case_id": case["case_id"], "variant": v["name"], **st})
            print(f"  {v['name']:<24s} ok={row['ok']!s:<5} steps={row['n_steps']:4d} "
                  f"up={row['up_mb']:7.2f} MB edge={row['edge_ms']:6.0f} ms "
                  f"cloud={row['cloud_ms']:6.0f} ms {text[-50:]!r}", flush=True)
        ref_ok = next(r["ok"] for r in case_rows if r["variant"] == REFERENCE)
        for r in case_rows:
            r["ref_ok"] = ref_ok
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
                                 cache_ref=CACHE_REF if CACHE_REF in names else REFERENCE)
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
    print(summary[["variant", "runs", "acc", "regressions", "improvements", "mcnemar_p",
                   "mean_steps", "mean_up_mb", "up_reduction_vs_ref", "mean_edge_ms",
                   "mean_cloud_ms"]].to_string(index=False))
    view = em_summary[(em_summary.rtt_ms == 20) & (em_summary.edge_slowdown == 3.0)]
    print("\nEmulated latency (s), RTT 20 ms, edge 3x slower:")
    print(view.pivot(index="variant", columns="bandwidth_mbps",
                     values="mean_latency_s").round(1).to_string())


if __name__ == "__main__":
    main()
