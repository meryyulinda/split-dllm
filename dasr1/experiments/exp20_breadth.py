"""Exp20: breadth -- answer quality on more problems, more seeds and a second task.

Run from dasr1/ (see run_exp20.sh; one GPU, timing not measured, so the GPU may be shared):
  CUDA_VISIBLE_DEVICES=1 python experiments/exp20_breadth.py

Question (RQ3 / RQ5): do the quality results hold beyond 30 GSM8K problems and one seed?
  GSM8K      first EXP20_N_GSM8K test problems (default 100), seeds EXP20_SEEDS["gsm8k"]
  HumanEval  all 164 problems, pass@1 with one sample, executed against the unit tests
             (src/humaneval.py)

The answers do not depend on the link for bf16 rows (exact, Exp11 / Exp13C / Exp17), so
every configuration runs on ONE GPU: the split hooks stay attached with edge = cloud,
which applies the same row rule and codec (INT8 deltas included) without a second GPU.

Configurations (paper names in brackets):
  dllm_cache  (Dream sampling T=0.1, top-p 0.9, entropy order; Kp=25, Kr=2)
    uncached          [no cache, reference]
    dllm_cache        [DASR + dLLM-Cache (bf16)] = the cache itself (bf16 rows are exact)
    dllm_cache_int8   [DASR + dLLM-Cache (INT8)]
    adaptive          [DASR + dLLM-Cache, adaptive refresh (bf16)]
  fastdllm    (greedy, blocks of 32; seeds do not matter)
    one_token_per_step   [no cache, one token per step, reference]
    fastdllm             [DASR + Fast-dLLM (bf16)], gamma 0.9
    fastdllm_int8        [DASR + Fast-dLLM (INT8)]
    fastdllm_gamma0.7    [DASR + Fast-dLLM (bf16), gamma = 0.7]

Reported per (task, configuration): accuracy per seed and mean, answers lost / gained
against the reference (pooled over seeds), exact McNemar p, answers identical to the
reference.

Outputs: outputs/exp20_breadth_<backend>_dream<tag>/ runs.csv, summary.csv, metadata.json
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

from exp10_common import ROOT, cfg, describe, mcnemar_exact_p  # noqa: E402
from src.split_runtime import place_split  # noqa: E402


def setting(name, default):
    return getattr(cfg, name, default)


BACKEND = str(setting("EXP20_BACKEND", "dllm_cache")).lower()
if BACKEND not in {"dllm_cache", "fastdllm"}:
    raise ValueError(f"EXP20_BACKEND must be 'dllm_cache' or 'fastdllm', not {BACKEND!r}")
DEVICE = setting("EXP20_DEVICE", "cuda:0")
TASKS = list(setting("EXP20_TASKS", ["gsm8k", "humaneval"]))
N_GSM8K = int(setting("EXP20_N_GSM8K", 100))
N_HUMANEVAL = setting("EXP20_N_HUMANEVAL", None)          # None = all 164
HUMANEVAL_PATH = setting("EXP20_HUMANEVAL", "../../human-eval/data/HumanEval.jsonl.gz")
SEEDS = dict(setting("EXP20_SEEDS", {"gsm8k": [42, 43, 44], "humaneval": [42]}))
NEW_TOKENS = dict(setting("EXP20_NEW_TOKENS", {"gsm8k": 256, "humaneval": 256}))
SPLIT = int(setting("EXP20_SPLIT_LAYER", 4))
ANCHOR = int(setting("EXP20_ANCHOR_INTERVAL", 32))
POLICY = tuple(setting("EXP11_CACHE_POLICY", ("fixed", 25, 2, 0.25)))
ADAPTIVE = setting("EXP11_ADAPTIVE_POLICY", (
    "adaptive", {"gen_content_commits": 2, "gen_max_age": 4, "prompt_content_commits": 8,
                 "prompt_max_age": 16, "eos_frontier_window": 8, "transfer_ratio": 0.25}))
DREAM_T = float(setting("EXP20_DREAM_TEMPERATURE", 0.1))
DREAM_TOP_P = setting("EXP20_DREAM_TOP_P", 0.9)
BLOCK = int(setting("EXP20_BLOCK_LENGTH", 32))
REPO = setting("FASTDLLM_REPO", "../../Fast-dLLM")
RESUME = bool(setting("EXP20_RESUME", True))
OUTPUT_TAG = str(setting("EXP20_OUTPUT_TAG", ""))
OUT = ROOT / "outputs" / f"exp20_breadth_{BACKEND}_dream{OUTPUT_TAG}"

CONFIGS = {
    "dllm_cache": {
        "uncached": dict(paper="No cache (reference)", policy=None, rows="all", encoding="bf16"),
        "dllm_cache": dict(paper="DASR + dLLM-Cache (bf16)", policy=POLICY, rows="cache_aware",
                           encoding="bf16"),
        "dllm_cache_int8": dict(paper="DASR + dLLM-Cache (INT8)", policy=POLICY,
                                rows="cache_aware", encoding="int8_delta"),
        "adaptive": dict(paper="DASR + dLLM-Cache, adaptive refresh (bf16)", policy=ADAPTIVE,
                         rows="cache_aware", encoding="bf16"),
    },
    "fastdllm": {
        "one_token_per_step": dict(paper="No cache, one token per step (reference)",
                                   cache="none", decode="fixed", threshold=None,
                                   encoding="bf16"),
        "fastdllm": dict(paper="DASR + Fast-dLLM (bf16)", cache="dual", decode="threshold",
                         threshold=0.9, encoding="bf16"),
        "fastdllm_int8": dict(paper="DASR + Fast-dLLM (INT8)", cache="dual", decode="threshold",
                              threshold=0.9, encoding="int8_delta"),
        "fastdllm_gamma0.7": dict(paper="DASR + Fast-dLLM (bf16), gamma = 0.7", cache="dual",
                                  decode="threshold", threshold=0.7, encoding="bf16"),
    },
}[BACKEND]
REFERENCE = {"dllm_cache": "uncached", "fastdllm": "one_token_per_step"}[BACKEND]


def load_task(task):
    if task == "gsm8k":
        from exp10d_gsm8k_refresh import gsm8k_score, load_cases
        return load_cases(n=N_GSM8K, start=0), gsm8k_score
    if task == "humaneval":
        from src.humaneval import humaneval_score, load_humaneval
        path = Path(HUMANEVAL_PATH)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; clone https://github.com/openai/human-eval "
                                    f"next to split-dllm or set EXP20_HUMANEVAL")
        return load_humaneval(path, N_HUMANEVAL), humaneval_score
    raise ValueError(f"unknown task {task!r}")


def build_backend():
    dev = torch.device(DEVICE)
    if BACKEND == "fastdllm":
        from src.fastdllm_split import DREAM_MODEL_NAME, KVBlockLink, kv_generate, \
            load_fastdllm_model
        model, tokenizer = load_fastdllm_model("dream", setting("MODEL_NAME", DREAM_MODEL_NAME),
                                               REPO, device=dev)
        place_split(model, SPLIT, dev, dev)           # hooks only: edge = cloud
        link = KVBlockLink(model, SPLIT, dev, dev)

        def generate(c, ids, mask, n_tokens, seed):
            return kv_generate(model, "dream", tokenizer, ids, n_tokens, BLOCK, seed,
                               cache=c["cache"], decode=c["decode"], threshold=c["threshold"],
                               temperature=0.0, top_p=None, link=link,
                               link_cfg={"encoding": c["encoding"], "anchor_interval": ANCHOR},
                               device=dev)
        return tokenizer, link, generate

    from exp9b_official_feature_cache_diagnostics import import_official
    from src.feature_cache_ext import generate as dream_generate
    from src.model_loader import load_dream_model
    from src.split_runtime import SplitLink
    _, api = import_official()
    model, tokenizer = load_dream_model(cfg.MODEL_NAME)
    model = model.to(dev).eval()
    place_split(model, SPLIT, dev, dev)               # hooks only: edge = cloud
    link = SplitLink(model, SPLIT, dev, dev)
    top_p = DREAM_TOP_P if DREAM_T > 0 else None

    def generate(c, ids, mask, n_tokens, seed):
        cached = c["policy"] is not None
        return dream_generate(model, tokenizer, ids, mask, n_tokens, n_tokens, seed,
                              alg="entropy", temperature=DREAM_T, top_p=top_p,
                              cache_api=api if cached else None, policy=c["policy"],
                              prompt_tail_rows=1 if cached else 0, capture_trajectory=False,
                              log_confidence=False, link=link,
                              link_cfg={"rows": c["rows"], "encoding": c["encoding"],
                                        "anchor_interval": ANCHOR})
    return tokenizer, link, generate


def prepare(tokenizer, prompt):
    from exp9b_official_feature_cache_diagnostics import prepare_input
    return prepare_input(tokenizer, prompt, torch.device(DEVICE))


def summarize(runs):
    df = pd.DataFrame(runs)
    for c in ("ok", "same_as_ref"):
        df[c] = df[c].map(lambda v: str(v).strip().lower() == "true")
    ref = df[df["config"] == REFERENCE].set_index(["task", "seed", "case_id"])["ok"]
    df["ref_ok"] = [ref.get(k) for k in zip(df["task"], df["seed"], df["case_id"])]
    df = df[df["ref_ok"].notna()].copy()
    df["ref_ok"] = df["ref_ok"].astype(bool)
    df["lost"] = df["ref_ok"] & ~df["ok"]
    df["gained"] = ~df["ref_ok"] & df["ok"]
    per_seed = df.groupby(["task", "config", "seed"])["ok"].mean().unstack("seed")
    out = df.groupby(["task", "config"], sort=False).agg(
        paper_name=("paper_name", "first"), runs=("case_id", "size"),
        seeds=("seed", "nunique"), acc=("ok", "mean"), lost=("lost", "sum"),
        gained=("gained", "sum"), identical=("same_as_ref", "sum")).reset_index()
    out["acc_seed_min"] = [per_seed.loc[(t, c)].min() for t, c in zip(out["task"], out["config"])]
    out["acc_seed_max"] = [per_seed.loc[(t, c)].max() for t, c in zip(out["task"], out["config"])]
    out["mcnemar_p"] = [mcnemar_exact_p(l, g) for l, g in zip(out["lost"], out["gained"])]
    return out


def main():
    names = list(setting("EXP20_CONFIG_NAMES", list(CONFIGS)))
    bad = [n for n in names if n not in CONFIGS]
    if bad:
        raise ValueError(f"unknown EXP20_CONFIG_NAMES {bad}; known: {list(CONFIGS)}")
    names = [REFERENCE] + [n for n in names if n != REFERENCE]   # reference first
    OUT.mkdir(parents=True, exist_ok=True)
    runs, have = [], set()
    if RESUME and (OUT / "runs.csv").exists():
        runs = pd.read_csv(OUT / "runs.csv").to_dict("records")
        have = {(r["task"], int(r["seed"]), r["case_id"]) for r in runs}
        print(f"resume: {len(have)} finished (task, seed, question) triples kept", flush=True)
    tokenizer, link, generate = build_backend()
    print(f"Exp20 | {BACKEND} | one GPU ({DEVICE}) | tasks {TASKS} | configs {names}", flush=True)
    t_start = time.time()
    for task in TASKS:
        cases, score_fn = load_task(task)
        seeds = SEEDS.get(task, [42]) if BACKEND == "dllm_cache" else SEEDS.get(task, [42])[:1]
        n_tok = int(NEW_TOKENS.get(task, 256))
        for seed in seeds:
            todo = [c for c in cases if (task, int(seed), c["case_id"]) not in have]
            print(f"\n=== {task}, seed {seed}: {len(todo)} of {len(cases)} problems to run  "
                  f"({time.strftime('%F %T')})", flush=True)
            for case in todo:
                ids, mask = prepare(tokenizer, case["prompt"])
                rows, ref_ids = [], None
                for name in names:
                    c = CONFIGS[name]
                    res = generate(c, ids, mask, n_tok, int(seed))
                    gen = res["generated_ids"]
                    if name == REFERENCE:
                        ref_ids = gen
                    text = describe(tokenizer, gen)["text"]
                    score = score_fn(case, text)
                    rows.append({"task": task, "seed": int(seed), "case_id": case["case_id"],
                                 "config": name, "paper_name": c["paper"], "ok": bool(score["ok"]),
                                 "extracted": score.get("extracted"),
                                 "same_as_ref": bool(torch.equal(ref_ids, gen)),
                                 "n_steps": len(res.get("split_steps") or []), "text": text})
                runs.extend(rows)
                pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)
                print(f"  {case['case_id']:<14s} " + " ".join(
                    f"{r['config']}={'Y' if r['ok'] else '-'}" for r in rows), flush=True)
    link.remove()
    summary = summarize(runs)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "experiment": "Exp20 breadth: quality on GSM8K (more problems, seeds) and HumanEval",
        "backend": BACKEND, "device": DEVICE, "tasks": TASKS, "n_gsm8k": N_GSM8K,
        "n_humaneval": N_HUMANEVAL or 164, "seeds": SEEDS, "new_tokens": NEW_TOKENS,
        "configs": {n: CONFIGS[n] for n in names}, "reference": REFERENCE,
        "split_layer_1based": SPLIT, "single_gpu": "edge = cloud (hooks only)",
        "dream_temperature": DREAM_T if BACKEND == "dllm_cache" else 0.0,
        "block_length": BLOCK if BACKEND == "fastdllm" else None,
        "hours": (time.time() - t_start) / 3600,
        "gpu": torch.cuda.get_device_name(torch.device(DEVICE)), "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
    }, indent=2, default=str), encoding="utf-8")
    print("\nExp20 completed. Output:", OUT)
    print(summary.drop(columns=["paper_name"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
