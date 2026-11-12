"""Shared grid runner for Exp10B / 10C / 10D (official dLLM-Cache, Dream, single GPU).

Cases default to the 12 authored Exp10A cases; pass ``cases=`` and
``score_fn=`` to evaluate another set (Exp10D: GSM8K).

A *variant* is a dict:
    {"name": "B1_entropy_tailfix",
     "alg": "entropy",                       # Dream remasking order
     "prompt_tail_rows": 1,                     # 0 = official split at prompt_len
     "force_prompt_tail": True,                 # prompt-tail rows always in partial update
     "policy": ("fixed", 100, 7, 0.25)}      # or ("adaptive", {...})

Every cached run is compared with a full-recompute baseline that uses the SAME
alg / seed / temperature. Outputs (in out_dir):
    baselines.csv        one row per (case, alg, seed)
    runs.csv             one row per cached run (fidelity + quality + refresh plan)
    summary_by_variant.csv, summary_by_variant_task.csv
    commits.jsonl        per-run commit log (step, pos, token, prob, margin)
    timing_runs.csv      only if timing is enabled (B-C-B, no history/logging)
    metadata.json
"""
from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
import transformers

import configs.experiment_config as cfg
from src.model_loader import load_dream_model
from src.feature_cache_ext import generate, divergence_detail
from src.refresh_rules import make_policy
from exp9b_official_feature_cache_diagnostics import (
    import_official, prepare_input, compare_history,
)
from exp10a_official_feature_cache_quality import make_cases, advisory_score


def quality_ok(case, score):
    """Advisory only: gold match (factual/math) or 3-item format (instruction)."""
    if case["task"] == "instruction":
        return score["instruction_format_pass"]
    return score["advisory_gold_match"]


def default_score(case, text):
    s = advisory_score(case, text)
    return {"ok": quality_ok(case, s), "extracted": s["advisory_extracted_answer"]}


def mcnemar_exact_p(regressions, improvements):
    """Two-sided exact McNemar test on the discordant pairs (binomial, p=0.5).
    Small p => cached accuracy really differs from the baseline."""
    n = int(regressions) + int(improvements)
    if n == 0:
        return 1.0
    k = min(int(regressions), int(improvements))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def seeds_for(alg, seeds, temperature):
    # Confidence-based orders with temperature 0 use no randomness at all:
    # extra seeds would only repeat identical runs.
    if alg != "origin" and not temperature:
        return list(seeds[:1])
    return list(seeds)


def row_work_fraction(refresh, prompt_split, total_len, ratio):
    """Proxy for per-layer token-row work vs full recompute (ignores layer 0,
    attention's quadratic term and the cheap V projection used for selection)."""
    if not refresh or not refresh["decided_steps"]:
        return None
    P, G = prompt_split, total_len - prompt_split
    gen_steps, prompt_steps = set(refresh["gen_refresh_steps"]), set(refresh["prompt_refresh_steps"])
    work = 0.0
    for s in range(1, refresh["decided_steps"] + 1):
        g, p = s in gen_steps, s in prompt_steps
        rows = (P if p else 0) + (G if g else ratio * G)
        work += rows / (P + G)
    return work / refresh["decided_steps"]


def describe(tokenizer, ids):
    eos = tokenizer.eos_token_id
    lst = ids.tolist()
    return {"text": tokenizer.decode(ids, skip_special_tokens=True),
            "eos_count": sum(t == eos for t in lst),
            "first_token": lst[0] if lst else None,
            "first_token_str": tokenizer.convert_ids_to_tokens(lst[:1])[0] if lst else None}


def as_bool(v):
    """True/False from bools, numbers and CSV strings ("True", "False", "")."""
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    if v is None or v != v:  # None / NaN
        return False
    return bool(v)


def summarize(runs):
    df = pd.DataFrame(runs)
    for col in ("exact_final_ids", "exact_trajectory", "first_token_equal", "baseline_ok",
                "cache_ok", "quality_regression", "quality_improvement", "div_includes_pos0"):
        df[col] = df[col].map(as_bool)
    for col in ("first_divergent_step", "gen_refresh_frac", "prompt_refresh_frac",
                "row_work_frac"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    agg = dict(
        runs=("case_id", "size"),
        exact_final=("exact_final_ids", "sum"),
        exact_trajectory=("exact_trajectory", "sum"),
        first_token_equal=("first_token_equal", "sum"),
        baseline_ok=("baseline_ok", "sum"),
        cache_ok=("cache_ok", "sum"),
        regressions=("quality_regression", "sum"),
        improvements=("quality_improvement", "sum"),
        diverged_at_pos0=("div_includes_pos0", "sum"),
        median_first_divergent_step=("first_divergent_step", "median"),
        mean_gen_refresh_frac=("gen_refresh_frac", "mean"),
        mean_prompt_refresh_frac=("prompt_refresh_frac", "mean"),
        mean_row_work_frac=("row_work_frac", "mean"),
    )
    by_variant = df.groupby(["variant", "alg", "prompt_tail_rows", "policy"],
                            sort=False).agg(**agg).reset_index()
    by_task = df.groupby(["variant", "task"], sort=False).agg(**agg).reset_index()
    for side in ("baseline", "cache"):
        col = f"{side}_extracted"
        if col in df:
            df[f"{side}_no_answer"] = pd.to_numeric(df[col], errors="coerce").isna()
            for table, keys in ((by_variant, ["variant", "alg", "prompt_tail_rows", "policy"]),
                                (by_task, ["variant", "task"])):
                counts = df.groupby(keys, sort=False)[f"{side}_no_answer"].sum()
                table[f"{side}_no_answer"] = counts.values
    for table in (by_variant, by_task):
        table["baseline_acc"] = table["baseline_ok"] / table["runs"]
        table["cache_acc"] = table["cache_ok"] / table["runs"]
        table["mcnemar_p"] = [mcnemar_exact_p(r, i) for r, i in
                              zip(table["regressions"], table["improvements"])]
    return by_variant, by_task


def run_grid(*, experiment, variants, out_dir, seeds, new_tokens, steps, temperature,
             top_p=None, top_k=None, alg_temp=0.0, case_ids=(), warmup_runs=1,
             run_timing=False, timing_repeats=2, cases=None, score_fn=None,
             resume=False):
    """resume=True keeps finished cases from an earlier, interrupted run in
    out_dir (a case counts as finished when every variant has a row)."""
    if not torch.cuda.is_available():
        raise RuntimeError(f"{experiment} requires a CUDA GPU")
    if "Dream" not in cfg.MODEL_NAME:
        raise RuntimeError(f"{experiment} uses the official Dream cache hook only")
    # Accept the old key names from earlier configs ("boundary_rows" / "force_boundary").
    renamed = {"boundary_rows": "prompt_tail_rows", "force_boundary": "force_prompt_tail"}
    variants = [{renamed.get(k, k): val for k, val in v.items()} for v in variants]
    for v in variants:
        make_policy(v["policy"])  # validate early
        if v["alg"] not in {"origin", "entropy", "maskgit_plus", "topk_margin"}:
            raise ValueError(f"Unknown alg in {v}")
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise ValueError("Variant names must be unique")

    score_fn = score_fn or default_score
    cases = list(cases) if cases is not None else make_cases()
    if case_ids:
        unknown = set(case_ids) - {c["case_id"] for c in cases}
        if unknown:
            raise KeyError(f"Unknown case IDs: {sorted(unknown)}")
        cases = [c for c in cases if c["case_id"] in case_ids]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commits_path = out_dir / "commits.jsonl"
    base_rows, runs, timing_rows = [], [], []
    done = set()
    if resume and (out_dir / "runs.csv").exists() and (out_dir / "baselines.csv").exists():
        prev_runs = pd.read_csv(out_dir / "runs.csv", keep_default_na=False).to_dict("records")
        prev_base = pd.read_csv(out_dir / "baselines.csv", keep_default_na=False).to_dict("records")
        want = {v["name"] for v in variants}
        have = {}
        for r in prev_runs:
            have.setdefault(r["case_id"], set()).add(r["variant"])
        done = {c for c, vs in have.items() if want <= vs}
        runs = [r for r in prev_runs if r["case_id"] in done]
        base_rows = [r for r in prev_base if r["case_id"] in done]
        print(f"resume: {len(done)} finished cases kept", flush=True)
    else:
        commits_path.write_text("", encoding="utf-8")

    repo, api = import_official()
    model, tokenizer = load_dream_model(cfg.MODEL_NAME)
    model.eval()
    device = next(model.parameters()).device
    gen_kw = dict(temperature=temperature, top_p=top_p, top_k=top_k, alg_temp=alg_temp)
    algs = list(dict.fromkeys(v["alg"] for v in variants))

    print(f"{experiment} | {len(cases)} cases | algs {algs} | seeds {list(seeds)} | "
          f"{new_tokens} tokens / {steps} steps | T={temperature}", flush=True)
    for v in variants:
        print(f"  variant {v['name']}: alg={v['alg']} B={v.get('prompt_tail_rows', 0)} "
              f"{make_policy(v['policy']).label()}", flush=True)

    if warmup_runs:
        ids_w, mask_w = prepare_input(tokenizer, cases[0]["prompt"], device)
        v0 = variants[0]
        for _ in range(warmup_runs):
            generate(model, tokenizer, ids_w, mask_w, new_tokens, steps, seeds[0],
                     alg=v0["alg"], capture_trajectory=False, observe=False, **gen_kw)
            generate(model, tokenizer, ids_w, mask_w, new_tokens, steps, seeds[0],
                     alg=v0["alg"], cache_api=api, policy=v0["policy"],
                     prompt_tail_rows=v0.get("prompt_tail_rows", 0),
                     force_prompt_tail=v0.get("force_prompt_tail", True),
                     capture_trajectory=False, **gen_kw)
        print("warmup done", flush=True)

    def log_commits(**record):
        with commits_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    for case in cases:
        if case["case_id"] in done:
            continue
        ids, mask = prepare_input(tokenizer, case["prompt"], device)
        plen = int(ids.shape[1])
        print(f"\n{case['case_id']} | prompt tokens={plen}", flush=True)
        for alg in algs:
            for seed in seeds_for(alg, seeds, temperature):
                base = generate(model, tokenizer, ids, mask, new_tokens, steps, seed,
                                alg=alg, **gen_kw)
                b = describe(tokenizer, base["generated_ids"])
                b_score = score_fn(case, b["text"])
                b_ok = b_score["ok"]
                pos0 = next((c for c in base["commits"] if c["pos"] == 0), None)
                base_rows.append({
                    "case_id": case["case_id"], "task": case["task"], "alg": alg,
                    "seed": int(seed), "prompt_tokens": plen,
                    "gold_answer": case["gold_answer"], "baseline_ok": b_ok,
                    "baseline_extracted": b_score.get("extracted"),
                    "baseline_text": b["text"], "baseline_eos": b["eos_count"],
                    "pos0_commit_step": pos0["step"] if pos0 else None,
                    "pos0_commit_prob": pos0["prob"] if pos0 else None,
                    "pos0_commit_margin": pos0["margin"] if pos0 else None,
                })
                pd.DataFrame(base_rows).to_csv(out_dir / "baselines.csv", index=False)
                log_commits(case_id=case["case_id"], alg=alg, seed=int(seed),
                            variant="baseline", commits=base["commits"])
                print(f"  [{alg} seed={seed}] BASE ok={b_ok} pos0@{pos0['step'] if pos0 else None}"
                      f" {b['text'][:70]!r}", flush=True)

                for v in (v for v in variants if v["alg"] == alg):
                    policy = make_policy(v["policy"])
                    B = int(v.get("prompt_tail_rows", 0))
                    cached = generate(model, tokenizer, ids, mask, new_tokens, steps, seed,
                                      alg=alg, cache_api=api, policy=policy,
                                      prompt_tail_rows=B,
                                      force_prompt_tail=v.get("force_prompt_tail", True),
                                      **gen_kw)
                    c = describe(tokenizer, cached["generated_ids"])
                    c_score = score_fn(case, c["text"])
                    c_ok = c_score["ok"]
                    fid = compare_history(base, cached, plen)
                    div = divergence_detail(base, cached, plen)
                    refresh = cached["refresh"]
                    row = {
                        "variant": v["name"], "alg": alg, "prompt_tail_rows": B,
                        "policy": policy.label(), "case_id": case["case_id"],
                        "task": case["task"], "seed": int(seed), "prompt_tokens": plen,
                        "gold_answer": case["gold_answer"],
                        "baseline_text": b["text"], "cache_text": c["text"],
                        "baseline_first_token": b["first_token_str"],
                        "cache_first_token": c["first_token_str"],
                        "baseline_eos": b["eos_count"], "cache_eos": c["eos_count"],
                        "baseline_ok": b_ok, "cache_ok": c_ok,
                        "baseline_extracted": b_score.get("extracted"),
                        "cache_extracted": c_score.get("extracted"),
                        "quality_regression": bool(b_ok) and not bool(c_ok),
                        "quality_improvement": (not bool(b_ok)) and bool(c_ok),
                        **fid, **div,
                        "gen_refresh_frac": refresh["gen_refresh_frac"],
                        "prompt_refresh_frac": refresh["prompt_refresh_frac"],
                        "row_work_frac": row_work_fraction(
                            refresh, plen - B, plen + int(new_tokens), policy.transfer_ratio),
                        "content_commits": refresh["content_commits"],
                        "frontier_eos_commits": refresh["frontier_eos_commits"],
                        "refresh_reasons": json.dumps(refresh["refresh_reasons"]),
                        "gen_refresh_steps": json.dumps(refresh["gen_refresh_steps"]),
                        "prompt_refresh_steps": json.dumps(refresh["prompt_refresh_steps"]),
                    }
                    runs.append(row)
                    pd.DataFrame(runs).to_csv(out_dir / "runs.csv", index=False)
                    log_commits(case_id=case["case_id"], alg=alg, seed=int(seed),
                                variant=v["name"], commits=cached["commits"])
                    tag = "EXACT" if fid["exact_trajectory"] else (
                        f"DIFF@{fid['first_divergent_step']} pos={div['div_first_pos']}")
                    work = row["row_work_frac"]
                    print(f"    {v['name']:<28s} {tag:<18s} ok={c_ok} "
                          f"work={'n/a' if work is None else f'{work:.2f}'} "
                          f"{c['text'][:60]!r}", flush=True)

        if run_timing:
            seed = seeds[0]
            for v in variants:
                for rep in range(int(timing_repeats)):
                    common = dict(alg=v["alg"], capture_trajectory=False, **gen_kw)
                    b1 = generate(model, tokenizer, ids, mask, new_tokens, steps, seed,
                                  observe=False, **common)
                    cc = generate(model, tokenizer, ids, mask, new_tokens, steps, seed,
                                  cache_api=api, policy=v["policy"],
                                  prompt_tail_rows=v.get("prompt_tail_rows", 0),
                                  force_prompt_tail=v.get("force_prompt_tail", True),
                                  observe=False, log_confidence=False, **common)
                    b2 = generate(model, tokenizer, ids, mask, new_tokens, steps, seed,
                                  observe=False, **common)
                    paired = (b1["ms"] + b2["ms"]) / 2
                    timing_rows.append({
                        "variant": v["name"], "case_id": case["case_id"],
                        "task": case["task"], "seed": int(seed), "repeat": rep,
                        "prompt_tokens": plen,
                        "baseline_pair_stable": torch.equal(b1["generated_ids"], b2["generated_ids"]),
                        "paired_baseline_ms": paired, "cache_ms": cc["ms"],
                        "saving_pct": 100 * (1 - cc["ms"] / paired),
                        "gen_refresh_frac": cc["refresh"]["gen_refresh_frac"],
                        "prompt_refresh_frac": cc["refresh"]["prompt_refresh_frac"],
                    })
                    pd.DataFrame(timing_rows).to_csv(out_dir / "timing_runs.csv", index=False)
            print("  timing recorded", flush=True)

    by_variant, by_task = summarize(runs)
    by_variant.to_csv(out_dir / "summary_by_variant.csv", index=False)
    by_task.to_csv(out_dir / "summary_by_variant_task.csv", index=False)
    if timing_rows:
        td = pd.DataFrame(timing_rows)
        td[td["baseline_pair_stable"]].groupby("variant", sort=False).agg(
            trials=("saving_pct", "size"),
            median_saving_pct=("saving_pct", "median"),
            min_saving_pct=("saving_pct", "min"),
        ).reset_index().to_csv(out_dir / "timing_summary.csv", index=False)

    metadata = {
        "experiment": experiment, "model": cfg.MODEL_NAME, "official_repo": str(repo),
        "variants": [{**v, "policy_label": make_policy(v["policy"]).label()} for v in variants],
        "cases": [c["case_id"] for c in cases], "seeds": list(seeds),
        "seeds_used_per_alg": {a: seeds_for(a, seeds, temperature) for a in algs},
        "new_tokens": new_tokens, "steps": steps, "temperature": temperature,
        "top_p": top_p, "top_k": top_k, "alg_temp": alg_temp, "timing": run_timing,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "cuda_device": torch.cuda.get_device_name(device),
        "notes": ("quality is advisory (gold substring / last number / 3-item format); "
                  "review runs.csv manually. row_work_frac is a row-count proxy, not FLOPs. "
                  "Single GPU, no edge-cloud transfer."),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str),
                                           encoding="utf-8")
    print(f"\n{experiment} completed. Output: {out_dir}")
    print(by_variant[["variant", "runs", "exact_final", "baseline_ok", "cache_ok",
                      "regressions", "improvements", "mcnemar_p",
                      "mean_row_work_frac"]].to_string(index=False))
    return by_variant
