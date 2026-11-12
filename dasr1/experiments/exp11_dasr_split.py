"""Exp11: DASR prototype on two GPUs (simulated edge + cloud), closed-loop generation.

Run from dasr1/:  python experiments/exp11_dasr_split.py
Then:             python experiments/exp11b_network_emulation.py

Dream's own sampler runs unchanged; src/split_runtime.py places layers 1..k on the
EDGE GPU and layers k+1..L + LM head + sampler on the CLOUD GPU, and records per
step: edge / cloud / sampler compute (CUDA events) and uplink / downlink bytes.

Variants (entropy, Dream's official sampling T=0.1 / top_p=0.9, one seed shared by
all variants of a question; prompt-tail fix whenever the cache is on):
  S0_split_full            no cache, full boundary every step  (= static split/full;
                           output identical to cloud-only up to cross-GPU numerics)
  S1_split_exact_delta     no cache, rows that changed bitwise          (exact; vs S0)
  C1_cache_full            feature cache on both sides, full boundary
  C2_cache_aware           + prompt rows only at prompt refreshes       (exact; vs C1)
  C3_cache_aware_exact     + only bitwise-changed rows                  (exact; vs C1)
  C4_cache_aware_int8      + INT8 deltas, bf16 anchors                  (approximate)
  A1_adaptive_cache_aware  C2 with the rule-based adaptive refresh

Checks to read first in summary_by_variant.csv:
  exact_vs_ref = runs   for S1, C2, C3  -> the exact tier really is exact
  regressions vs S0                       -> what caching / INT8 cost in answers
  mean_up_mb, up_reduction_vs_S0          -> communication
  edge_ms / cloud_ms totals               -> computation per side
"""
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from exp10_common import (  # noqa: E402  (also puts dasr1/ on sys.path)
    ROOT, cfg, as_bool, default_score, describe, mcnemar_exact_p,
)
from exp9b_official_feature_cache_diagnostics import (  # noqa: E402
    import_official, prepare_input, compare_history,
)
from exp10a_official_feature_cache_quality import make_cases  # noqa: E402
from src.feature_cache_ext import generate  # noqa: E402
from src.model_loader import load_dream_model  # noqa: E402
from src.refresh_rules import make_policy  # noqa: E402
from src.split_runtime import SplitLink, place_split  # noqa: E402

DATASET = getattr(cfg, "EXP11_DATASET", "gsm8k")  # "gsm8k" or "authored"
N = int(getattr(cfg, "EXP11_N_QUESTIONS", 20))
SPLIT = int(getattr(cfg, "EXP11_SPLIT_LAYER", 4))
EDGE = getattr(cfg, "EXP11_EDGE_DEVICE", getattr(cfg, "EDGE_DEVICE", "cuda:1"))
CLOUD = getattr(cfg, "EXP11_CLOUD_DEVICE", getattr(cfg, "CLOUD_DEVICE", "cuda:0"))
DEFAULT_LEN = {"gsm8k": 256, "authored": 128}[DATASET]
NEW_TOKENS = int(getattr(cfg, "EXP11_NEW_TOKENS", DEFAULT_LEN))
STEPS = int(getattr(cfg, "EXP11_STEPS", DEFAULT_LEN))
POLICY = tuple(getattr(cfg, "EXP11_CACHE_POLICY", ("fixed", 25, 2, 0.25)))
ADAPTIVE = getattr(cfg, "EXP11_ADAPTIVE_POLICY", (
    "adaptive", {"gen_content_commits": 1, "gen_max_age": 2, "prompt_content_commits": 8,
                 "prompt_max_age": 16, "eos_frontier_window": 8, "transfer_ratio": 0.25}))
ANCHOR = int(getattr(cfg, "EXP11_ANCHOR_INTERVAL", 32))
RESUME = bool(getattr(cfg, "EXP11_RESUME", True))
SEED = int(getattr(cfg, "SEED", 42))
TEMPERATURE = float(getattr(cfg, "EXP11_TEMPERATURE", 0.1))
TOP_P = getattr(cfg, "EXP11_TOP_P", 0.9) if TEMPERATURE > 0 else None
OUT = ROOT / "outputs" / f"exp11_{DATASET}_k{SPLIT}_t{NEW_TOKENS}_T{TEMPERATURE:g}"


def V(name, policy, rows, encoding, same_as=None):
    return {"name": name, "policy": policy, "rows": rows, "encoding": encoding,
            "same_as": same_as}


DEFAULT_VARIANTS = [
    V("S0_split_full", None, "all", "bf16"),
    V("S1_split_exact_delta", None, "all", "exact_delta", same_as="S0_split_full"),
    V("C1_cache_full", POLICY, "all", "bf16"),
    V("C2_cache_aware", POLICY, "cache_aware", "bf16", same_as="C1_cache_full"),
    V("C3_cache_aware_exact", POLICY, "cache_aware", "exact_delta", same_as="C1_cache_full"),
    V("C4_cache_aware_int8", POLICY, "cache_aware", "int8_delta"),
    V("A1_adaptive_cache_aware", ADAPTIVE, "cache_aware", "bf16"),
]
REFERENCE = "S0_split_full"


def load():
    if DATASET == "gsm8k":
        from exp10d_gsm8k_refresh import gsm8k_score, load_cases
        return load_cases(n=N, start=0), gsm8k_score
    if DATASET == "authored":
        return make_cases(), default_score
    raise ValueError(f"EXP11_DATASET must be 'gsm8k' or 'authored', not {DATASET!r}")


def run_summary(steps):
    df = pd.DataFrame(steps)
    return {
        "steps": len(df),
        "edge_ms": df["edge_ms"].sum(), "encode_ms": df["encode_ms"].sum(),
        "cloud_ms": df["cloud_ms"].sum(), "sample_ms": df["sample_ms"].sum(),
        "up_mb": df["up_bytes"].sum() / 1e6, "full_mb": df["full_bytes"].sum() / 1e6,
        "down_kb": df["down_bytes"].sum() / 1e3,
        "mean_rows_sent": df["rows_sent"].mean(),
        "prompt_sent_steps": int(df["prompt_sent"].sum()),
        "anchor_steps": int(df["anchor"].sum()),
    }


def summarize(runs):
    df = pd.DataFrame(runs)
    for col in ("cache_ok", "ref_ok", "exact_vs_ref", "exact_vs_S0"):
        df[col] = df[col].map(as_bool)
    df["regression"] = df["ref_ok"] & ~df["cache_ok"]
    df["improvement"] = ~df["ref_ok"] & df["cache_ok"]
    s0_up = df[df["variant"] == REFERENCE].set_index("case_id")["up_mb"]
    df["up_reduction_vs_S0"] = 1 - df["up_mb"] / df["case_id"].map(s0_up)
    out = df.groupby("variant", sort=False).agg(
        runs=("case_id", "size"), acc=("cache_ok", "mean"), ref_acc=("ref_ok", "mean"),
        regressions=("regression", "sum"), improvements=("improvement", "sum"),
        exact_vs_ref=("exact_vs_ref", "sum"), exact_vs_S0=("exact_vs_S0", "sum"),
        mean_up_mb=("up_mb", "mean"), mean_full_mb=("full_mb", "mean"),
        up_reduction_vs_S0=("up_reduction_vs_S0", "mean"),
        mean_edge_ms=("edge_ms", "mean"), mean_encode_ms=("encode_ms", "mean"),
        mean_cloud_ms=("cloud_ms", "mean"), mean_sample_ms=("sample_ms", "mean"),
        mean_prompt_sent_steps=("prompt_sent_steps", "mean"),
    ).reset_index()
    out["mcnemar_p"] = [mcnemar_exact_p(r, i) for r, i in
                        zip(out["regressions"], out["improvements"])]
    return out


def main():
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Exp11 needs two GPUs (edge + cloud)")
    if torch.device(EDGE) == torch.device(CLOUD):
        raise ValueError("EXP11_EDGE_DEVICE and EXP11_CLOUD_DEVICE must differ")
    variants = list(getattr(cfg, "EXP11_VARIANTS", DEFAULT_VARIANTS))
    names = [v["name"] for v in variants]
    if REFERENCE not in names or len(set(names)) != len(names):
        raise ValueError(f"Variants need unique names incl. {REFERENCE}")
    for v in variants:
        if v["policy"] is not None:
            make_policy(v["policy"])
        if v.get("same_as") and v["same_as"] not in names:
            raise ValueError(f"{v['name']}: same_as={v['same_as']} is not a variant")

    cases, score_fn = load()
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

    repo, api = import_official()
    model, tokenizer = load_dream_model(cfg.MODEL_NAME)
    model.eval()
    edge, cloud = place_split(model, SPLIT, EDGE, CLOUD)
    link = SplitLink(model, SPLIT, edge, cloud)
    print(f"Exp11 | {DATASET} n={len(cases)} | split after layer {SPLIT} | edge={edge} "
          f"cloud={cloud} | {NEW_TOKENS} tok / {STEPS} steps | cache {POLICY}", flush=True)

    def run(case, v, ids, mask):
        return generate(
            model, tokenizer, ids, mask, NEW_TOKENS, STEPS, SEED, alg="entropy",
            temperature=TEMPERATURE, top_p=TOP_P, cache_api=api if v["policy"] is not None else None,
            policy=v["policy"], prompt_tail_rows=1 if v["policy"] is not None else 0,
            capture_trajectory=True, log_confidence=False, link=link,
            link_cfg={"rows": v["rows"], "encoding": v["encoding"],
                      "anchor_interval": ANCHOR})

    ids_w, mask_w = prepare_input(tokenizer, cases[0]["prompt"], cloud)
    for v in (variants[0], next((x for x in variants if x["rows"] == "cache_aware"),
                                variants[0])):
        run(cases[0], v, ids_w, mask_w)
    print("warmup done", flush=True)

    for case in cases:
        if case["case_id"] in done:
            continue
        ids, mask = prepare_input(tokenizer, case["prompt"], cloud)
        plen = int(ids.shape[1])
        print(f"\n{case['case_id']} | prompt tokens={plen}", flush=True)
        results, case_rows, step_rows = {}, [], []
        for v in variants:
            res = run(case, v, ids, mask)
            results[v["name"]] = res
            text = describe(tokenizer, res["generated_ids"])["text"]
            score = score_fn(case, text)
            ref = results.get(v["same_as"]) if v.get("same_as") else None
            s0 = results[REFERENCE] if REFERENCE in results else None
            row = {
                "variant": v["name"], "case_id": case["case_id"], "task": case["task"],
                "prompt_tokens": plen, "policy": make_policy(v["policy"]).label()
                if v["policy"] is not None else "none",
                "rows": v["rows"], "encoding": v["encoding"],
                "gold_answer": case["gold_answer"], "text": text,
                "extracted": score.get("extracted"), "cache_ok": bool(score["ok"]),
                "exact_vs_ref": (torch.equal(ref["generated_ids"], res["generated_ids"])
                                 and compare_history(ref, res, plen)["exact_trajectory"])
                if ref is not None else "",
                "exact_vs_S0": torch.equal(s0["generated_ids"], res["generated_ids"])
                if s0 is not None else "",
                **run_summary(res["split_steps"]),
                "refresh_reasons": json.dumps(res["refresh"]["refresh_reasons"])
                if res["refresh"] else "",
            }
            case_rows.append(row)
            for st in res["split_steps"]:
                step_rows.append({"case_id": case["case_id"], "variant": v["name"], **st})
            tag = ("" if ref is None else
                   (" EXACT" if row["exact_vs_ref"] else " NOT-EXACT(!)"))
            print(f"  {v['name']:<26s} ok={row['cache_ok']!s:<5} up={row['up_mb']:7.2f} MB "
                  f"edge={row['edge_ms']:7.0f} ms cloud={row['cloud_ms']:7.0f} ms{tag} "
                  f"{text[-50:]!r}", flush=True)
        ref_ok = next(r["cache_ok"] for r in case_rows if r["variant"] == REFERENCE)
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
    (OUT / "metadata.json").write_text(json.dumps({
        "experiment": "Exp11 DASR two-GPU split prototype (closed loop)",
        "model": cfg.MODEL_NAME, "official_repo": str(repo), "dataset": DATASET,
        "n_cases": len(cases), "split_layer_1based": SPLIT, "edge": str(edge),
        "cloud": str(cloud), "edge_gpu": torch.cuda.get_device_name(edge),
        "cloud_gpu": torch.cuda.get_device_name(cloud), "new_tokens": NEW_TOKENS,
        "steps": STEPS, "alg": "entropy", "temperature": TEMPERATURE, "top_p": TOP_P,
        "seed": SEED,
        "cache_policy": POLICY, "adaptive_policy": ADAPTIVE, "anchor_interval": ANCHOR,
        "variants": variants, "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "notes": "Bytes from the wire format in src/split_runtime.py; compute from CUDA "
                 "events per GPU; network latency added by exp11b (emulation).",
    }, indent=2, default=str), encoding="utf-8")
    print("\nExp11 completed. Output:", OUT)
    print(summary[["variant", "runs", "acc", "regressions", "exact_vs_ref", "mean_up_mb",
                   "up_reduction_vs_S0", "mean_edge_ms", "mean_cloud_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
