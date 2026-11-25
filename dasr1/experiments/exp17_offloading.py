"""Exp17: how much cloud work does a split take off the cloud GPU?

Run from dasr1/ (see run_exp17.sh):
  CUDA_VISIBLE_DEVICES=1,2 python experiments/exp17_offloading.py

The paper motivates splitting by offloading computation from the cloud. This
experiment measures that benefit per answer, and how it trades against latency
and edge memory as the split moves deeper (the boundary rows have the same size
at every layer, so the bytes sent do not depend on the split layer k):

  1. cloud-only: the whole model on the cloud GPU, same cache and decoding
     (no link). Cloud GPU time = generation time; NVML energy of the cloud GPU.
  2. for every split layer k in EXP17_SPLIT_LAYERS: DASR on two GPUs (no network,
     as Exp11 / Exp13). Cloud GPU time = cloud layers + sampler (CUDA events),
     edge GPU time = edge layers + codec; NVML energy and peak memory per side.

Backends and variants (paper names in brackets):
  dllm_cache  (Dream sampling, T=0.1, top-p 0.9, entropy order, Kp=25, Kr=2)
    C2_cache_aware        [DASR + dLLM-Cache (bf16)]  vs cloud-only + dLLM-Cache
    C4_cache_aware_int8   [DASR + dLLM-Cache (INT8)]
  fastdllm    (greedy, DualCache, blocks of 32, gamma 0.9)
    K2_dual_thr0.9        [DASR + Fast-dLLM (bf16)]   vs cloud-only + Fast-dLLM
    K3_dual_thr0.9_int8   [DASR + Fast-dLLM (INT8)]

Per (k, variant) summary.csv reports
  cloud_s / cloud_only_s       cloud GPU seconds per answer, split vs cloud-only
  cloud_saving                 1 - cloud_s / cloud_only_s   (work taken off the cloud)
  answers_per_cloud_gpu_ratio  cloud_only_s / cloud_s  (answers one cloud GPU can serve
                               per hour, split vs cloud-only, if it serves users one
                               after another without batching)
  cloud_active_j, cloud_only_active_j, cloud_energy_saving  (above idle power)
  edge_s, edge_peak_gib        what the edge pays
  same_as_cloud                identical answer to cloud-only (bf16 rows: expected all)
  latency_<B>mbps_s, latency_ratio_<B>mbps   Eq. latency (RTT, edge slowdown s) and
                               its ratio to cloud-only (measured time + one RTT)

Outputs: outputs/exp17_offloading_<backend>_dream_<dataset>_t<tokens><tag>/
  runs.csv, summary.csv, metadata.json
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

from exp10_common import ROOT, cfg, describe  # noqa: E402
import exp11b_network_emulation as emu  # noqa: E402
from src.gpu_meter import EnergyMeter, measure_idle_w  # noqa: E402
from src.split_runtime import model_parts, place_split  # noqa: E402


def setting(name, default):
    return getattr(cfg, name, default)


BACKEND = str(setting("EXP17_BACKEND", "dllm_cache")).lower()
if BACKEND not in {"dllm_cache", "fastdllm"}:
    raise ValueError(f"EXP17_BACKEND must be 'dllm_cache' or 'fastdllm', not {BACKEND!r}")
DATASET = setting("EXP17_DATASET", "gsm8k")
N = int(setting("EXP17_N_QUESTIONS", 10))
NEW_TOKENS = int(setting("EXP17_NEW_TOKENS", 256))
STEPS = int(setting("EXP17_STEPS", NEW_TOKENS))          # dLLM-Cache backend only
SPLITS = [int(k) for k in setting("EXP17_SPLIT_LAYERS", [2, 4, 8, 14, 20])]
EDGE = setting("EXP17_EDGE_DEVICE", "cuda:1")
CLOUD = setting("EXP17_CLOUD_DEVICE", "cuda:0")
ANCHOR = int(setting("EXP17_ANCHOR_INTERVAL", 32))
SEED = int(setting("SEED", 42))
RTT_MS = float(setting("EXP17_RTT_MS", 20.0))
MBPS = [float(b) for b in setting("EXP17_BANDWIDTHS_MBPS", [10, 100, 1000])]
EDGE_SLOWDOWN = float(setting("EXP17_EDGE_SLOWDOWN", 3.0))
IDLE_SECONDS = float(setting("EXP17_IDLE_SECONDS", 5.0))
RESUME = bool(setting("EXP17_RESUME", True))
OUTPUT_TAG = str(setting("EXP17_OUTPUT_TAG", ""))
POLICY = tuple(setting("EXP11_CACHE_POLICY", ("fixed", 25, 2, 0.25)))
DREAM_T = float(setting("EXP17_DREAM_TEMPERATURE", 0.1))
DREAM_TOP_P = setting("EXP17_DREAM_TOP_P", 0.9)
BLOCK = int(setting("EXP17_BLOCK_LENGTH", 32))
THRESHOLD = float(setting("EXP17_THRESHOLD", 0.9))
REPO = setting("FASTDLLM_REPO", "../../Fast-dLLM")
GIB = 2 ** 30

OUT = ROOT / "outputs" / f"exp17_offloading_{BACKEND}_dream_{DATASET}_t{NEW_TOKENS}{OUTPUT_TAG}"
g = f"{THRESHOLD:g}"
VARIANTS = {
    "C2_cache_aware": dict(paper="DASR + dLLM-Cache (bf16)", rows="cache_aware",
                           encoding="bf16"),
    "C4_cache_aware_int8": dict(paper="DASR + dLLM-Cache (INT8)", rows="cache_aware",
                                encoding="int8_delta"),
    f"K2_dual_thr{g}": dict(paper="DASR + Fast-dLLM (bf16)", encoding="bf16"),
    f"K3_dual_thr{g}_int8": dict(paper="DASR + Fast-dLLM (INT8)", encoding="int8_delta"),
}
DEFAULT_NAMES = {"dllm_cache": ["C2_cache_aware", "C4_cache_aware_int8"],
                 "fastdllm": [f"K2_dual_thr{g}", f"K3_dual_thr{g}_int8"]}
CLOUD_ONLY = {"dllm_cache": "Cloud-only + dLLM-Cache", "fastdllm": "Cloud-only + Fast-dLLM"}


def load_cases():
    if DATASET == "gsm8k":
        from exp10d_gsm8k_refresh import gsm8k_score, load_cases as gsm8k_cases
        return gsm8k_cases(n=N, start=0), gsm8k_score
    if DATASET == "authored":
        from exp10_common import default_score
        from exp10a_official_feature_cache_quality import make_cases
        return make_cases()[:N], default_score
    raise ValueError(f"EXP17_DATASET must be 'gsm8k' or 'authored', not {DATASET!r}")


def build_backend():
    """Model on the cloud GPU + generate(v, ids, mask, link) for this backend."""
    if BACKEND == "fastdllm":
        from src.fastdllm_split import DREAM_MODEL_NAME, KVBlockLink, kv_generate, \
            load_fastdllm_model
        model, tokenizer = load_fastdllm_model(
            "dream", setting("MODEL_NAME", DREAM_MODEL_NAME), REPO, device=CLOUD)

        def generate(v, ids, mask, link):
            return kv_generate(model, "dream", tokenizer, ids, NEW_TOKENS, BLOCK, SEED,
                               cache="dual", decode="threshold", threshold=THRESHOLD,
                               temperature=0.0, top_p=None, link=link,
                               link_cfg={"encoding": v["encoding"], "anchor_interval": ANCHOR}
                               if link is not None else None,
                               device=torch.device(CLOUD))
        return model, tokenizer, KVBlockLink, generate

    from exp9b_official_feature_cache_diagnostics import import_official
    from src.feature_cache_ext import generate as dream_generate
    from src.model_loader import load_dream_model
    from src.split_runtime import SplitLink
    _, api = import_official()
    model, tokenizer = load_dream_model(cfg.MODEL_NAME)
    model = model.to(CLOUD).eval()                        # whole model on the cloud GPU
    top_p = DREAM_TOP_P if DREAM_T > 0 else None

    def generate(v, ids, mask, link):
        return dream_generate(model, tokenizer, ids, mask, NEW_TOKENS, STEPS, SEED,
                              alg="entropy", temperature=DREAM_T, top_p=top_p,
                              cache_api=api, policy=POLICY, prompt_tail_rows=1,
                              capture_trajectory=False, log_confidence=False, link=link,
                              link_cfg={"rows": v["rows"], "encoding": v["encoding"],
                                        "anchor_interval": ANCHOR} if link is not None else None)
    return model, tokenizer, SplitLink, generate


def prepare(tokenizer, prompt, device):
    from exp9b_official_feature_cache_diagnostics import prepare_input
    return prepare_input(tokenizer, prompt, device)


def summarize(runs):
    df = pd.DataFrame(runs)
    for c in ("ok", "same_as_cloud"):
        df[c] = df[c].map(lambda v: str(v).strip().lower() == "true")
    co = df[df["config"] == "cloud_only"].set_index("case_id")
    sp = df[df["config"] != "cloud_only"].copy()
    if sp.empty:
        return sp
    for col in ("cloud_s", "cloud_active_j", "wall_s"):
        if col in co:
            sp[f"cloud_only_{col}"] = sp["case_id"].map(co[col])
    sp["cloud_saving"] = 1 - sp["cloud_s"] / sp["cloud_only_cloud_s"]
    if "cloud_active_j" in sp:
        sp["cloud_energy_saving"] = 1 - sp["cloud_active_j"] / sp["cloud_only_cloud_active_j"]
    agg = dict(paper_name=("paper_name", "first"), runs=("case_id", "size"),
               acc=("ok", "mean"), same_as_cloud=("same_as_cloud", "sum"),
               edge_layer_share=("edge_layer_share", "first"),
               cloud_s=("cloud_s", "mean"), cloud_only_s=("cloud_only_cloud_s", "mean"),
               cloud_saving=("cloud_saving", "mean"), edge_s=("edge_s", "mean"),
               cloud_active_j=("cloud_active_j", "mean"),
               cloud_only_active_j=("cloud_only_cloud_active_j", "mean"),
               cloud_energy_saving=("cloud_energy_saving", "mean"),
               edge_active_j=("edge_active_j", "mean"),
               edge_peak_gib=("edge_peak_gib", "mean"), cloud_peak_gib=("cloud_peak_gib", "mean"),
               mean_steps=("n_steps", "mean"), up_mb=("up_mb", "mean"))
    for b in MBPS:
        sp[f"latency_ratio_{b:g}mbps"] = sp[f"latency_{b:g}mbps_s"] / sp["cloud_only_latency_s"]
        agg[f"latency_{b:g}mbps_s"] = (f"latency_{b:g}mbps_s", "mean")
        agg[f"latency_ratio_{b:g}mbps"] = (f"latency_ratio_{b:g}mbps", "mean")
    agg["cloud_only_latency_s"] = ("cloud_only_latency_s", "mean")
    agg = {k: v for k, v in agg.items() if v[0] in sp.columns}   # energy needs NVML
    out = sp.groupby(["split_layer", "variant"], sort=True).agg(**agg).reset_index()
    out["answers_per_cloud_gpu_ratio"] = out["cloud_only_s"] / out["cloud_s"]
    co_acc = df[df["config"] == "cloud_only"]["ok"].mean()
    out["cloud_only_acc"] = co_acc
    return out


def main():
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Exp17 needs two GPUs (edge + cloud)")
    names = list(setting("EXP17_VARIANT_NAMES", DEFAULT_NAMES[BACKEND]))
    fast = {n for n in VARIANTS if n.startswith("K")}
    bad = [n for n in names if n not in VARIANTS or (n in fast) != (BACKEND == "fastdllm")]
    if bad:
        raise ValueError(f"variants {bad} do not exist for backend {BACKEND!r}")
    variants = [dict(VARIANTS[n], name=n) for n in names]
    cases, score_fn = load_cases()
    OUT.mkdir(parents=True, exist_ok=True)
    runs, have = [], set()
    if RESUME and (OUT / "runs.csv").exists():
        runs = pd.read_csv(OUT / "runs.csv").to_dict("records")
        have = {(r["config"], r["case_id"]) for r in runs}
        print(f"resume: {len(runs)} finished runs kept", flush=True)

    model, tokenizer, LinkCls, generate = build_backend()
    edge, cloud = torch.device(EDGE), torch.device(CLOUD)
    n_layers = len(model_parts(model).layers)
    meter = EnergyMeter({"edge": edge, "cloud": cloud})
    idle_w = measure_idle_w(meter, (edge, cloud), IDLE_SECONDS)
    if idle_w:
        print("idle power: " + ", ".join(f"{k}={v:.1f} W" for k, v in idle_w.items()), flush=True)
    print(f"Exp17 | {BACKEND} | {DATASET} n={len(cases)} | {NEW_TOKENS} tokens | splits {SPLITS} "
          f"of {n_layers} layers | variants {names}", flush=True)

    def run(v, case, link):
        ids, mask = prepare(tokenizer, case["prompt"], cloud)
        for d in (edge, cloud):
            torch.cuda.synchronize(d)
            torch.cuda.reset_peak_memory_stats(d)
        e0, t0 = (meter.read_j() if meter.handles else {}), time.perf_counter()
        res = generate(v, ids, mask, link)
        for d in (edge, cloud):
            torch.cuda.synchronize(d)
        wall = time.perf_counter() - t0
        e1 = meter.read_j() if meter.handles else {}
        r = {"wall_s": wall, "prompt_tokens": int(ids.shape[1]),
             "edge_peak_gib": torch.cuda.max_memory_allocated(edge) / GIB,
             "cloud_peak_gib": torch.cuda.max_memory_allocated(cloud) / GIB}
        for side in ("edge", "cloud"):
            if side in e0:
                joules = e1[side] - e0[side]
                r[f"{side}_energy_j"] = joules
                r[f"{side}_active_j"] = joules - idle_w.get(side, 0.0) * wall
        return res, r

    def record(config, k, v, case, res, r, ref_ids):
        text = describe(tokenizer, res["generated_ids"])["text"]
        row = {"config": config, "split_layer": k, "case_id": case["case_id"],
               "variant": v["name"] if v else "cloud_only",
               "paper_name": v["paper"] if v else CLOUD_ONLY[BACKEND],
               "edge_layer_share": k / n_layers if k else 0.0,
               "text": text, "extracted": score_fn(case, text).get("extracted"),
               "ok": bool(score_fn(case, text)["ok"]),
               "same_as_cloud": bool(ref_ids is None or torch.equal(ref_ids,
                                                                    res["generated_ids"])),
               **r}
        steps = res.get("split_steps")
        if steps:  # split run: per-side GPU time from CUDA events, Eq. latency
            df = pd.DataFrame(steps)
            row.update(n_steps=len(df), up_mb=df["up_bytes"].sum() / 1e6,
                       edge_s=(df["edge_ms"].sum() + df["encode_ms"].sum()) / 1e3,
                       cloud_s=(df["cloud_ms"].sum() + df["sample_ms"].sum()) / 1e3)
            df["case_id"], df["variant"] = 0, "x"
            terms = emu.per_run_terms(df)
            for b in MBPS:
                row[f"latency_{b:g}mbps_s"] = float(
                    emu.latency_s(terms, b, RTT_MS, EDGE_SLOWDOWN).iloc[0])
        else:      # cloud-only: the cloud GPU is busy for the whole generation
            row.update(n_steps=res.get("n_steps"), up_mb=0.0, edge_s=0.0,
                       cloud_s=res["ms"] / 1e3)
            row["cloud_only_latency_s"] = res["ms"] / 1e3 + RTT_MS / 1e3
        return row

    def save(rows):
        nonlocal runs
        keys = {(r["config"], r["case_id"]) for r in rows}
        runs = [r for r in runs if (r["config"], r["case_id"]) not in keys] + rows
        pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)

    # 1) cloud-only: whole model on the cloud GPU (references for answers and cost)
    ref_ids = {}
    cv = {"name": "cloud_only", "encoding": "bf16", "rows": "all"}
    run(cv, cases[0], None)  # warm-up
    print(f"\n=== cloud-only ({time.strftime('%F %T')})", flush=True)
    for case in cases:
        res, r = run(cv, case, None)
        ref_ids[case["case_id"]] = res["generated_ids"]
        if ("cloud_only", case["case_id"]) not in have:
            row = record("cloud_only", 0, None, case, res, r, None)
            save([row])
            print(f"  {case['case_id']} ok={row['ok']!s:<5} cloud={row['cloud_s']:6.1f} s "
                  f"energy={row.get('cloud_active_j', float('nan')):7.0f} J", flush=True)
    # cloud-only latency for every split row (same case)
    co_lat = {r["case_id"]: r["cloud_only_latency_s"] for r in runs if r["config"] == "cloud_only"}

    # 2) split after layer k
    for k in SPLITS:
        if not 1 <= k < n_layers:
            raise ValueError(f"split layer {k} not in [1, {n_layers - 1}]")
        todo = [(c, v) for c in cases for v in variants
                if (f"k{k}:{v['name']}", c["case_id"]) not in have]
        if not todo:
            continue
        place_split(model, k, EDGE, CLOUD)
        link = LinkCls(model, k, edge, cloud)
        run(variants[0], cases[0], link)  # warm-up for this placement
        print(f"\n=== split after layer {k} ({time.strftime('%F %T')})", flush=True)
        for case, v in todo:
            res, r = run(v, case, link)
            row = record(f"k{k}:{v['name']}", k, v, case, res, r, ref_ids[case["case_id"]])
            row["cloud_only_latency_s"] = co_lat[case["case_id"]]
            save([row])
            print(f"  {case['case_id']} {v['name']:<22s} ok={row['ok']!s:<5} "
                  f"same={row['same_as_cloud']!s:<5} cloud={row['cloud_s']:6.1f} s "
                  f"edge={row['edge_s']:5.1f} s edge_mem={row['edge_peak_gib']:4.1f} GiB "
                  f"lat100={row.get('latency_100mbps_s', float('nan')):6.1f} s", flush=True)
        link.remove()

    summary = summarize(runs)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "experiment": "Exp17 offloading: cloud GPU time and energy per answer vs split layer",
        "backend": BACKEND, "dataset": DATASET, "n_cases": len(cases),
        "new_tokens": NEW_TOKENS, "steps": STEPS, "split_layers_1based": SPLITS,
        "n_layers": n_layers, "variants": variants, "cache_policy": POLICY,
        "dream_temperature": DREAM_T if BACKEND == "dllm_cache" else 0.0,
        "block_length": BLOCK if BACKEND == "fastdllm" else None,
        "threshold": THRESHOLD if BACKEND == "fastdllm" else None,
        "latency_model": {"rtt_ms": RTT_MS, "bandwidths_mbps": MBPS,
                          "edge_slowdown": EDGE_SLOWDOWN},
        "idle_power_w": idle_w, "anchor_interval": ANCHOR, "seed": SEED,
        "notes": "cloud-only cloud_s = generation time with the whole model on the cloud GPU; "
                 "split cloud_s = cloud layers + sampler (CUDA events); no network in this run",
        "edge_gpu": torch.cuda.get_device_name(edge), "cloud_gpu": torch.cuda.get_device_name(cloud),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
    }, indent=2, default=str), encoding="utf-8")
    print("\nExp17 completed. Output:", OUT)
    if not summary.empty:
        show = ["split_layer", "variant", "runs", "acc", "same_as_cloud", "cloud_s", "cloud_only_s",
                "cloud_saving", "answers_per_cloud_gpu_ratio", "cloud_energy_saving", "edge_s",
                "edge_peak_gib", "latency_100mbps_s", "latency_ratio_100mbps"]
        print(summary[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
