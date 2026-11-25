"""Exp16: the split over a real (shaped) network link instead of the latency model.

Run from dasr1/ (see run_exp16.sh):
  CUDA_VISIBLE_DEVICES=1,2 python experiments/exp16_real_network.py

Question: does the analytical latency model of Exp11b, which all latency numbers in
the paper come from, predict the time per answer on a real link?

  step latency = edge + encode + bytes / bandwidth + RTT + cloud + sampler

Set-up: the edge half (embedding + layers 1..k) and the cloud half (layers k+1..L,
LM head, sampler) run on two GPUs as in Exp11 / Exp13, but every boundary message
and every downlink of committed tokens now goes as bytes over TCP through a relay
process (src/net_relay.py) that imposes a bandwidth and a one-way delay of RTT/2.
The cloud reads only the rows decoded from the received bytes and the edge only the
tokens it received (src/net_wire.py), so the answers also test the wire format.

For each workload and variant (paper names in brackets):
  dLLM-Cache backend  (Dream sampling, T=0.1, top-p 0.9, entropy order, Kp=25, Kr=2)
    S0_split_full            [Static split (bf16)]
    S2_split_int8            [Static split (INT8)]                optional
    C2_cache_aware           [DASR + dLLM-Cache (bf16)]
    C4_cache_aware_int8      [DASR + dLLM-Cache (INT8)]
    A1_adaptive_cache_aware  [DASR + dLLM-Cache, adaptive refresh (bf16)]  optional
  Fast-dLLM backend   (greedy, DualCache, blocks of 32, gamma 0.9)
    K2_dual_thr0.9           [DASR + Fast-dLLM (bf16)]
    K3_dual_thr0.9_int8      [DASR + Fast-dLLM (INT8)]
the script runs
  1. "local": no network (as Exp11 / Exp13), the reference for answers and for the
     prediction the paper uses (equation applied to this run);
  2. every link of EXP16_LINKS (bandwidth, RTT, number of questions): the real time
     per answer, wall clock from the first step until the edge holds the answer.

Per run it reports measured_s, predicted_s (equation on this run's own compute and
bytes), predicted_local_s (equation on the local run = what the paper reports),
their ratios, same_as_local (identical answer), the real ping RTT, and the
per-step wire overhead the model ignores (GPU<->CPU copies, framing, sockets).

Outputs: outputs/exp16_<backend>_dream_<dataset>_k<split>_t<tokens><tag>/
  runs.csv, steps.csv, summary.csv, metadata.json
"""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from exp10_common import ROOT, cfg, describe  # noqa: E402
import exp11b_network_emulation as emu  # noqa: E402
from src.net_wire import Link, Wire  # noqa: E402
from src.split_runtime import place_split  # noqa: E402


def setting(name, default):
    return getattr(cfg, name, default)


BACKEND = str(setting("EXP16_BACKEND", "dllm_cache")).lower()
if BACKEND not in {"dllm_cache", "fastdllm"}:
    raise ValueError(f"EXP16_BACKEND must be 'dllm_cache' or 'fastdllm', not {BACKEND!r}")
DATASET = setting("EXP16_DATASET", "gsm8k")
NEW_TOKENS = int(setting("EXP16_NEW_TOKENS", 256))
STEPS = int(setting("EXP16_STEPS", NEW_TOKENS))          # dLLM-Cache backend only
SPLIT = int(setting("EXP16_SPLIT_LAYER", 4))
EDGE = setting("EXP16_EDGE_DEVICE", "cuda:1")
CLOUD = setting("EXP16_CLOUD_DEVICE", "cuda:0")
ANCHOR = int(setting("EXP16_ANCHOR_INTERVAL", 32))
SEED = int(setting("SEED", 42))
EDGE_SLOWDOWN = float(setting("EXP16_EDGE_SLOWDOWN", 1.0))
HEADER_BYTES = int(setting("EXP16_TCPIP_HEADER_BYTES", 40))   # per 1460-B segment; 0 = off
# (bandwidth Mbit/s, RTT ms, number of questions); a dict may add "trace": csv path
LINKS = list(setting("EXP16_LINKS", [(1000, 20, 10), (100, 20, 10), (10, 20, 2)]))
RESUME = bool(setting("EXP16_RESUME", True))
OUTPUT_TAG = str(setting("EXP16_OUTPUT_TAG", ""))
# dLLM-Cache backend (Exp11 settings)
POLICY = tuple(setting("EXP11_CACHE_POLICY", ("fixed", 25, 2, 0.25)))
ADAPTIVE = setting("EXP11_ADAPTIVE_POLICY", (
    "adaptive", {"gen_content_commits": 2, "gen_max_age": 4, "prompt_content_commits": 8,
                 "prompt_max_age": 16, "eos_frontier_window": 8, "transfer_ratio": 0.25}))
DREAM_T = float(setting("EXP16_DREAM_TEMPERATURE", 0.1))
DREAM_TOP_P = setting("EXP16_DREAM_TOP_P", 0.9)
# Fast-dLLM backend (Exp13 settings)
BLOCK = int(setting("EXP16_BLOCK_LENGTH", 32))
THRESHOLD = float(setting("EXP16_THRESHOLD", 0.9))
REPO = setting("FASTDLLM_REPO", "../../Fast-dLLM")

OUT = ROOT / "outputs" / f"exp16_{BACKEND}_dream_{DATASET}_k{SPLIT}_t{NEW_TOKENS}{OUTPUT_TAG}"
g = f"{THRESHOLD:g}"
VARIANTS = {
    "S0_split_full": dict(paper="Static split (bf16)", policy=None, rows="all",
                          encoding="bf16"),
    "S2_split_int8": dict(paper="Static split (INT8)", policy=None, rows="all",
                          encoding="int8_delta"),
    "C2_cache_aware": dict(paper="DASR + dLLM-Cache (bf16)", policy=POLICY,
                           rows="cache_aware", encoding="bf16"),
    "C4_cache_aware_int8": dict(paper="DASR + dLLM-Cache (INT8)", policy=POLICY,
                                rows="cache_aware", encoding="int8_delta"),
    "A1_adaptive_cache_aware": dict(paper="DASR + dLLM-Cache, adaptive refresh (bf16)",
                                    policy=ADAPTIVE, rows="cache_aware", encoding="bf16"),
    f"K2_dual_thr{g}": dict(paper="DASR + Fast-dLLM (bf16)", encoding="bf16"),
    f"K3_dual_thr{g}_int8": dict(paper="DASR + Fast-dLLM (INT8)", encoding="int8_delta"),
}
DEFAULT_NAMES = {"dllm_cache": ["S0_split_full", "C2_cache_aware", "C4_cache_aware_int8"],
                 "fastdllm": [f"K2_dual_thr{g}", f"K3_dual_thr{g}_int8"]}


def link_specs():
    specs = []
    for item in LINKS:
        if isinstance(item, dict):
            spec = {"mbps": item.get("mbps"), "rtt_ms": float(item["rtt_ms"]),
                    "n": int(item["n"]), "trace": item.get("trace")}
        else:
            spec = {"mbps": float(item[0]), "rtt_ms": float(item[1]), "n": int(item[2]),
                    "trace": None}
        if spec["trace"]:  # model prediction with the trace's time-average rate
            tr = pd.read_csv(spec["trace"])
            dt = np.diff(tr["time_s"].to_numpy(), append=tr["time_s"].iloc[-1])
            spec["mbps_pred"] = float((tr["mbps"] * dt).sum() / max(dt.sum(), 1e-9))
            spec["label"] = f"trace_{Path(spec['trace']).stem}_rtt{spec['rtt_ms']:g}"
        else:
            spec["mbps_pred"] = spec["mbps"]
            spec["label"] = f"{spec['mbps']:g}mbps_rtt{spec['rtt_ms']:g}"
        specs.append(spec)
    return specs


def load_cases(n):
    if DATASET == "gsm8k":
        from exp10d_gsm8k_refresh import gsm8k_score, load_cases as gsm8k_cases
        return gsm8k_cases(n=n, start=0), gsm8k_score
    if DATASET == "authored":
        from exp10_common import default_score
        from exp10a_official_feature_cache_quality import make_cases
        return make_cases()[:n], default_score
    raise ValueError(f"EXP16_DATASET must be 'gsm8k' or 'authored', not {DATASET!r}")


def build_backend():
    """Load the model, place the split and return (generate(v, ids, mask, link), link, ...)."""
    if BACKEND == "fastdllm":
        from src.fastdllm_split import DREAM_MODEL_NAME, KVBlockLink, kv_generate, \
            load_fastdllm_model
        model, tokenizer = load_fastdllm_model("dream", setting("MODEL_NAME", DREAM_MODEL_NAME),
                                               REPO)
        edge, cloud = place_split(model, SPLIT, EDGE, CLOUD)
        link = KVBlockLink(model, SPLIT, edge, cloud)

        def generate(v, ids, mask):
            return kv_generate(model, "dream", tokenizer, ids, NEW_TOKENS, BLOCK, SEED,
                               cache="dual", decode="threshold", threshold=THRESHOLD,
                               temperature=0.0, top_p=None, link=link,
                               link_cfg={"encoding": v["encoding"], "anchor_interval": ANCHOR},
                               device=cloud)
    else:
        from exp9b_official_feature_cache_diagnostics import import_official
        from src.feature_cache_ext import generate as dream_generate
        from src.model_loader import load_dream_model
        from src.split_runtime import SplitLink
        _, api = import_official()
        model, tokenizer = load_dream_model(cfg.MODEL_NAME)
        model.eval()
        edge, cloud = place_split(model, SPLIT, EDGE, CLOUD)
        link = SplitLink(model, SPLIT, edge, cloud)
        top_p = DREAM_TOP_P if DREAM_T > 0 else None

        def generate(v, ids, mask):
            cached = v["policy"] is not None
            return dream_generate(model, tokenizer, ids, mask, NEW_TOKENS, STEPS, SEED,
                                  alg="entropy", temperature=DREAM_T, top_p=top_p,
                                  cache_api=api if cached else None, policy=v["policy"],
                                  prompt_tail_rows=1 if cached else 0,
                                  capture_trajectory=False, log_confidence=False, link=link,
                                  link_cfg={"rows": v["rows"], "encoding": v["encoding"],
                                            "anchor_interval": ANCHOR})
    return model, tokenizer, edge, cloud, link, generate


def prepare(tokenizer, prompt, device):
    from exp9b_official_feature_cache_diagnostics import prepare_input
    return prepare_input(tokenizer, prompt, device)


def step_terms(steps):
    df = pd.DataFrame(steps)
    df["case_id"], df["variant"] = 0, "x"
    return emu.per_run_terms(df)


def predict_s(steps, mbps, rtt_ms):
    return float(emu.latency_s(step_terms(steps), mbps, rtt_ms, EDGE_SLOWDOWN).iloc[0])


def done_keys():
    if not (RESUME and (OUT / "runs.csv").exists()):
        return [], set()
    prev = pd.read_csv(OUT / "runs.csv").to_dict("records")
    return prev, {(r["link"], r["case_id"], r["variant"]) for r in prev}


def summarize(runs):
    df = pd.DataFrame(runs)
    for c in ("ok", "same_as_local", "edge_has_answer", "bytes_match"):
        if c in df:
            df[c] = df[c].map(lambda v: str(v).strip().lower() == "true")
    net = df[df["link"] != "local"].copy()
    if net.empty:
        return net
    for c in ("measured_s", "predicted_s", "predicted_local_s", "wire_overhead_ms_per_step",
              "ping_ms", "n_steps", "up_mb", "bandwidth_mbps", "rtt_ms"):
        net[c] = pd.to_numeric(net[c], errors="coerce")
    net["abs_err"] = (net["measured_s"] / net["predicted_s"] - 1).abs()
    net["abs_err_local"] = (net["measured_s"] / net["predicted_local_s"] - 1).abs()
    out = net.groupby(["link", "variant"], sort=False).agg(
        paper_name=("paper_name", "first"), runs=("case_id", "size"), acc=("ok", "mean"),
        same_as_local=("same_as_local", "sum"), edge_has_answer=("edge_has_answer", "sum"),
        bytes_match=("bytes_match", "sum"), mbps=("bandwidth_mbps", "first"),
        rtt_ms=("rtt_ms", "first"), ping_ms=("ping_ms", "first"),
        mean_steps=("n_steps", "mean"), mean_up_mb=("up_mb", "mean"),
        measured_s=("measured_s", "mean"), predicted_s=("predicted_s", "mean"),
        predicted_local_s=("predicted_local_s", "mean"),
        mean_abs_err=("abs_err", "mean"), max_abs_err=("abs_err", "max"),
        mean_abs_err_local=("abs_err_local", "mean"),
        wire_overhead_ms_per_step=("wire_overhead_ms_per_step", "mean"),
    ).reset_index()
    out["ratio_measured_predicted"] = out["measured_s"] / out["predicted_s"]
    out["ratio_measured_predicted_local"] = out["measured_s"] / out["predicted_local_s"]
    return out


def main():
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Exp16 needs two GPUs (edge + cloud)")
    names = list(setting("EXP16_VARIANT_NAMES", DEFAULT_NAMES[BACKEND]))
    fast = {n for n in VARIANTS if n.startswith("K")}
    bad = [n for n in names if n not in VARIANTS or (n in fast) != (BACKEND == "fastdllm")]
    if bad:
        raise ValueError(f"variants {bad} do not exist for backend {BACKEND!r}")
    variants = [dict(VARIANTS[n], name=n) for n in names]
    specs = link_specs()
    n_max = max(s["n"] for s in specs)
    cases, score_fn = load_cases(n_max)
    OUT.mkdir(parents=True, exist_ok=True)
    runs, have = done_keys()
    if have:
        print(f"resume: {len(have)} finished runs kept", flush=True)

    model, tokenizer, edge, cloud, link, generate = build_backend()
    print(f"Exp16 | {BACKEND} | {DATASET} | split after layer {SPLIT} | {NEW_TOKENS} tokens | "
          f"variants {names} | links {[s['label'] for s in specs]} | edge slowdown "
          f"{EDGE_SLOWDOWN:g}", flush=True)

    def run(v, case, wire):
        ids, mask = prepare(tokenizer, case["prompt"], cloud)
        link.wire = wire
        for d in (edge, cloud):
            torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        res = generate(v, ids, mask)
        fin = {}
        if wire is not None:  # last step's commits to the edge: the answer is delivered
            full = torch.cat([ids.to(cloud), res["generated_ids"].to(cloud).unsqueeze(0)], 1)
            fin = wire.final(full)
        measured = time.perf_counter() - t0
        link.wire = None
        return ids, res, measured, fin

    # warm-up (kernels, caches), not recorded
    for v in variants:
        run(v, cases[0], None)
    print("warmup done", flush=True)

    local = {}  # (case_id, variant) -> (generated ids, steps) of the run without network
    plan = [({"label": "local", "mbps": None, "mbps_pred": None, "rtt_ms": 0.0,
              "n": n_max, "trace": None}, None)] + [(s, s) for s in specs]
    steps_path = OUT / "steps.csv"
    for spec, net in plan:
        todo = [c for c in cases[:spec["n"]]
                if not all((spec["label"], c["case_id"], v["name"]) in have for v in variants)]
        if not todo:
            continue
        print(f"\n=== link {spec['label']} ({len(todo)} questions)  "
              f"({time.strftime('%F %T')})", flush=True)
        wl, wire, ping = None, None, float("nan")
        if net is not None:
            wl = Link(mbps=net["mbps"], rtt_ms=net["rtt_ms"], trace=net["trace"],
                      header_bytes=HEADER_BYTES)
            ping = float(np.median(wl.ping(7)))
            wire = Wire(wl, edge, cloud, edge_slowdown=EDGE_SLOWDOWN)
            print(f"real RTT through the relay: {ping:.2f} ms (set {net['rtt_ms']:g} ms)",
                  flush=True)
        try:
            for case in todo:
                rows, step_rows = [], []
                for v in variants:
                    key = (case["case_id"], v["name"])
                    if net is not None and key not in local:  # local run needed first
                        _, lres, _, _ = run(v, case, None)
                        local[key] = (lres["generated_ids"], lres["split_steps"])
                    ids, res, measured, fin = run(v, case, wire)
                    steps = res["split_steps"]
                    if net is None:
                        local[key] = (res["generated_ids"], steps)
                    text = describe(tokenizer, res["generated_ids"])["text"]
                    score = score_fn(case, text)
                    df = pd.DataFrame(steps)
                    row = {
                        "link": spec["label"], "case_id": case["case_id"], "variant": v["name"],
                        "paper_name": v["paper"], "encoding": v["encoding"],
                        "bandwidth_mbps": spec["mbps"], "rtt_ms": spec["rtt_ms"],
                        "trace": spec["trace"] or "", "edge_slowdown": EDGE_SLOWDOWN,
                        "ping_ms": ping, "prompt_tokens": int(ids.shape[1]),
                        "n_steps": len(df), "up_mb": df["up_bytes"].sum() / 1e6,
                        "down_kb": df["down_bytes"].sum() / 1e3,
                        "edge_ms": df["edge_ms"].sum(), "encode_ms": df["encode_ms"].sum(),
                        "cloud_ms": df["cloud_ms"].sum(), "sample_ms": df["sample_ms"].sum(),
                        "measured_s": measured, "text": text,
                        "extracted": score.get("extracted"), "ok": bool(score["ok"]),
                        "same_as_local": bool(torch.equal(local[key][0], res["generated_ids"])),
                    }
                    if net is not None:
                        mb = spec["mbps_pred"]
                        row["predicted_s"] = predict_s(steps, mb, spec["rtt_ms"])
                        row["predicted_local_s"] = predict_s(local[key][1], mb, spec["rtt_ms"])
                        row["wire_up_mb"] = df["wire_up_bytes"].sum() / 1e6
                        row["bytes_match"] = bool(df["wire_up_bytes"].sum() == df["up_bytes"].sum())
                        row["up_wire_ms"] = df["up_wire_ms"].sum()
                        row["down_wire_ms"] = df["down_wire_ms"].sum() + fin["final_down_ms"]
                        row["edge_sleep_ms"] = df["edge_sleep_ms"].sum()
                        model_net_ms = ((df["up_bytes"].sum() + df["down_bytes"].sum())
                                        * 8 / (mb * 1e3) + len(df) * spec["rtt_ms"])
                        row["wire_overhead_ms_per_step"] = (
                            (row["up_wire_ms"] + row["down_wire_ms"] - model_net_ms) / len(df))
                        row.update(fin)
                    rows.append(row)
                    for st in steps:
                        step_rows.append({"link": spec["label"], "case_id": case["case_id"],
                                          "variant": v["name"], **st})
                    pred = (f" predicted={row['predicted_s']:6.1f} s (local {row['predicted_local_s']:6.1f})"
                            if net is not None else "")
                    print(f"  {case['case_id']} {v['name']:<24s} ok={row['ok']!s:<5} "
                          f"same={row['same_as_local']!s:<5} steps={row['n_steps']:4d} "
                          f"up={row['up_mb']:7.2f} MB measured={measured:6.1f} s{pred}",
                          flush=True)
                runs = [r for r in runs if (r["link"], r["case_id"]) !=
                        (spec["label"], case["case_id"])] + rows
                pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)
                pd.DataFrame(step_rows).to_csv(steps_path, mode="a", index=False,
                                               header=not steps_path.exists())
        finally:
            if wl is not None:
                wl.close()

    link.remove()
    summary = summarize(runs)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "experiment": "Exp16 split over a real shaped link vs the latency model",
        "backend": BACKEND, "model": setting("MODEL_NAME", "Dream-org/Dream-v0-Instruct-7B"),
        "dataset": DATASET, "new_tokens": NEW_TOKENS, "steps": STEPS,
        "split_layer_1based": SPLIT, "variants": variants, "links": specs,
        "edge_slowdown": EDGE_SLOWDOWN, "tcpip_header_bytes_per_1460B": HEADER_BYTES,
        "anchor_interval": ANCHOR, "seed": SEED,
        "dream_temperature": DREAM_T if BACKEND == "dllm_cache" else 0.0,
        "block_length": BLOCK if BACKEND == "fastdllm" else None,
        "threshold": THRESHOLD if BACKEND == "fastdllm" else None,
        "relay": "src/net_relay.py: one process per direction, fixed rate + delay, FIFO, "
                 "no loss; halves in one process, messages over TCP through the relays",
        "edge_gpu": torch.cuda.get_device_name(edge), "cloud_gpu": torch.cuda.get_device_name(cloud),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
    }, indent=2, default=str), encoding="utf-8")
    print("\nExp16 completed. Output:", OUT)
    if not summary.empty:
        show = ["link", "variant", "runs", "same_as_local", "edge_has_answer", "bytes_match",
                "ping_ms", "measured_s", "predicted_s", "predicted_local_s",
                "ratio_measured_predicted", "mean_abs_err", "wire_overhead_ms_per_step"]
        print(summary[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
