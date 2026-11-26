"""Exp18: choosing the codec (and Fast-dLLM's gamma) while the bandwidth changes.

Run from dasr1/ (see run_exp18.sh):
  CUDA_VISIBLE_DEVICES=1,2 python experiments/exp18_live_controller.py

Question (RQ4, adaptation): today the codec is chosen once per request. On a link
whose bandwidth changes during an answer, is it worth choosing it at every step,
and, with Fast-dLLM, also the decoding threshold gamma?

Set-up: as Exp16 (real TCP link through the network emulator, src/net_relay.py),
but the emulator replays a bandwidth trace; the edge is slowed down s times
(EXP18_EDGE_SLOWDOWN, default 3, as in the paper). Traces (EXP18_TRACES):
  step      synthetic square wave, 100 Mbit/s for 10 s, then 10 Mbit/s for 10 s
  LTE       Mahimahi cellular uplink traces (T-Mobile and Verizon LTE, "short"),
            converted to 100-ms bins (src/net_traces.py); the downlink replays the
            matching .down trace
Every question starts at its own position in the trace (13 s apart), the same for
all policies, so the policies of a question see the same bandwidth.

Policies (src/net_controller.py):
  bf16                   bf16 rows at every step (exact tier)
  int8                   INT8 deltas at every step (approximate tier)
  codec_per_request      bf16 or INT8 chosen once, after the first two steps
  codec_per_step         bf16 or INT8 chosen at every step
  codec_gamma_per_step   (Fast-dLLM) + gamma 0.9 or 0.7 chosen at every step
Rule: INT8 when it would shorten the step by more than EXP18_LAMBDA (10%), from the
measured throughput and compute time; gamma 0.7 while the link takes more than half
of the step time.

Reported per (trace, policy): measured time per answer, accuracy, answers identical
to bf16 (= exact), share of steps sent as INT8, steps, uplink, switches.

Outputs: outputs/exp18_controller_<backend>_dream_<dataset>_t<tokens><tag>/
  runs.csv, steps.csv, summary.csv, traces/*.csv, metadata.json
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
from src.net_controller import CodecController  # noqa: E402
from src.net_traces import mahimahi_to_csv, step_csv, trace_stats  # noqa: E402
from src.net_wire import Link, Wire  # noqa: E402
from src.split_runtime import place_split  # noqa: E402


def setting(name, default):
    return getattr(cfg, name, default)


BACKEND = str(setting("EXP18_BACKEND", "fastdllm")).lower()
if BACKEND not in {"dllm_cache", "fastdllm"}:
    raise ValueError(f"EXP18_BACKEND must be 'dllm_cache' or 'fastdllm', not {BACKEND!r}")
DATASET = setting("EXP18_DATASET", "gsm8k")
NEW_TOKENS = int(setting("EXP18_NEW_TOKENS", 256))
STEPS = int(setting("EXP18_STEPS", NEW_TOKENS))
SPLIT = int(setting("EXP18_SPLIT_LAYER", 4))
EDGE = setting("EXP18_EDGE_DEVICE", "cuda:1")
CLOUD = setting("EXP18_CLOUD_DEVICE", "cuda:0")
ANCHOR = int(setting("EXP18_ANCHOR_INTERVAL", 32))
SEED = int(setting("SEED", 42))
RTT_MS = float(setting("EXP18_RTT_MS", 20.0))
EDGE_SLOWDOWN = float(setting("EXP18_EDGE_SLOWDOWN", 3.0))
HEADER_BYTES = int(setting("EXP18_TCPIP_HEADER_BYTES", 40))
LAMBDA = float(setting("EXP18_LAMBDA", 0.10))
HYSTERESIS = float(setting("EXP18_HYSTERESIS", 1.0))
NET_SHARE = float(setting("EXP18_NET_SHARE", 0.5))
GAMMA_HIGH = float(setting("EXP18_THRESHOLD", 0.9))
GAMMA_LOW = float(setting("EXP18_GAMMA_LOW", 0.7))
OFFSET_STEP_S = float(setting("EXP18_TRACE_OFFSET_STEP_S", 13.0))
MAHIMAHI_DIR = Path(setting("EXP18_MAHIMAHI_DIR", "../../mahimahi/traces"))
TRACES = list(setting("EXP18_TRACES", [
    {"name": "step_100_10", "step": (100, 10, 20), "n": 10},
    {"name": "tmobile_lte_up", "mahimahi": "TMobile-LTE-short.up", "n": 10},
    {"name": "verizon_lte_up", "mahimahi": "Verizon-LTE-short.up", "n": 10},
]))
RESUME = bool(setting("EXP18_RESUME", True))
OUTPUT_TAG = str(setting("EXP18_OUTPUT_TAG", ""))
POLICY = tuple(setting("EXP11_CACHE_POLICY", ("fixed", 25, 2, 0.25)))
DREAM_T = float(setting("EXP18_DREAM_TEMPERATURE", 0.1))
DREAM_TOP_P = setting("EXP18_DREAM_TOP_P", 0.9)
BLOCK = int(setting("EXP18_BLOCK_LENGTH", 32))
REPO = setting("FASTDLLM_REPO", "../../Fast-dLLM")

OUT = ROOT / "outputs" / f"exp18_controller_{BACKEND}_dream_{DATASET}_t{NEW_TOKENS}{OUTPUT_TAG}"
POLICIES = {
    "bf16": dict(encoding="bf16"),
    "int8": dict(encoding="int8_delta"),
    "codec_per_request": dict(encoding="adaptive", mode="per_request"),
    "codec_per_step": dict(encoding="adaptive", mode="per_step"),
    "codec_gamma_per_step": dict(encoding="adaptive", mode="per_step", adapt_gamma=True),
}
DEFAULT_NAMES = {"dllm_cache": ["bf16", "int8", "codec_per_request", "codec_per_step"],
                 "fastdllm": ["bf16", "int8", "codec_per_request", "codec_per_step",
                              "codec_gamma_per_step"]}


def load_cases(n):
    if DATASET == "gsm8k":
        from exp10d_gsm8k_refresh import gsm8k_score, load_cases as gsm8k_cases
        return gsm8k_cases(n=n, start=0), gsm8k_score
    if DATASET == "authored":
        from exp10_common import default_score
        from exp10a_official_feature_cache_quality import make_cases
        return make_cases()[:n], default_score
    raise ValueError(f"EXP18_DATASET must be 'gsm8k' or 'authored', not {DATASET!r}")


def prepare_traces():
    out = []
    for t in TRACES:
        dst = OUT / "traces" / f"{t['name']}.csv"
        if "step" in t:
            hi, lo, period = t["step"]
            step_csv(dst, hi, lo, period)
        elif "mahimahi" in t:
            src = Path(t["mahimahi"])
            if not src.is_absolute():
                src = (ROOT / MAHIMAHI_DIR / src).resolve()
            if not src.is_file():
                raise FileNotFoundError(f"Mahimahi trace {src} not found; clone "
                                        f"https://github.com/ravinet/mahimahi next to split-dllm "
                                        f"or set EXP18_MAHIMAHI_DIR")
            mahimahi_to_csv(src, dst, bin_ms=int(t.get("bin_ms", 100)))
            down = src.with_suffix(".down")   # matching downlink trace, if any
            if src.suffix == ".up" and down.is_file():
                ddst = OUT / "traces" / f"{t['name']}_down.csv"
                mahimahi_to_csv(down, ddst, bin_ms=int(t.get("bin_ms", 100)))
                t = {**t, "down_csv": str(ddst)}
        elif "csv" in t:
            dst = Path(t["csv"])
        else:
            raise ValueError(f"trace {t} needs 'step', 'mahimahi' or 'csv'")
        out.append({**t, "csv": str(dst), **trace_stats(dst)})
    return out


def build_backend():
    if BACKEND == "fastdllm":
        from src.fastdllm_split import DREAM_MODEL_NAME, KVBlockLink, kv_generate, \
            load_fastdllm_model
        model, tokenizer = load_fastdllm_model("dream", setting("MODEL_NAME", DREAM_MODEL_NAME),
                                               REPO)
        edge, cloud = place_split(model, SPLIT, EDGE, CLOUD)
        link = KVBlockLink(model, SPLIT, edge, cloud)

        def generate(p, ids, mask, ctl):
            thr = ctl.gamma if (ctl is not None and p.get("adapt_gamma")) else GAMMA_HIGH
            return kv_generate(model, "dream", tokenizer, ids, NEW_TOKENS, BLOCK, SEED,
                               cache="dual", decode="threshold", threshold=thr,
                               temperature=0.0, top_p=None, link=link,
                               link_cfg={"encoding": p["encoding"], "anchor_interval": ANCHOR},
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

        def generate(p, ids, mask, ctl):
            return dream_generate(model, tokenizer, ids, mask, NEW_TOKENS, STEPS, SEED,
                                  alg="entropy", temperature=DREAM_T, top_p=top_p,
                                  cache_api=api, policy=POLICY, prompt_tail_rows=1,
                                  capture_trajectory=False, log_confidence=False, link=link,
                                  link_cfg={"rows": "cache_aware", "encoding": p["encoding"],
                                            "anchor_interval": ANCHOR})
    return tokenizer, edge, cloud, link, generate


def prepare(tokenizer, prompt, device):
    from exp9b_official_feature_cache_diagnostics import prepare_input
    return prepare_input(tokenizer, prompt, device)


def summarize(runs):
    df = pd.DataFrame(runs)
    for c in ("ok", "same_as_bf16", "edge_has_answer"):
        df[c] = df[c].map(lambda v: str(v).strip().lower() == "true")
    bf = df[df["policy"] == "bf16"].set_index(["trace", "case_id"])["measured_s"]
    i8 = df[df["policy"] == "int8"].set_index(["trace", "case_id"])["measured_s"]
    key = list(zip(df["trace"], df["case_id"]))
    df["speedup_vs_bf16"] = [bf.get(k, np.nan) for k in key] / df["measured_s"]
    df["time_vs_int8"] = df["measured_s"] / [i8.get(k, np.nan) for k in key]
    out = df.groupby(["trace", "policy"], sort=False).agg(
        runs=("case_id", "size"), acc=("ok", "mean"), same_as_bf16=("same_as_bf16", "sum"),
        edge_has_answer=("edge_has_answer", "sum"), measured_s=("measured_s", "mean"),
        speedup_vs_bf16=("speedup_vs_bf16", "mean"), time_vs_int8=("time_vs_int8", "mean"),
        int8_step_share=("int8_step_share", "mean"), mean_gamma=("mean_gamma", "mean"),
        mean_steps=("n_steps", "mean"), up_mb=("up_mb", "mean"),
        switches=("ctl_switches", "mean"), trace_mean_mbps=("trace_mean_mbps", "first"),
    ).reset_index()
    return out


def main():
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Exp18 needs two GPUs (edge + cloud)")
    names = list(setting("EXP18_POLICY_NAMES", DEFAULT_NAMES[BACKEND]))
    bad = [n for n in names if n not in POLICIES
           or (n == "codec_gamma_per_step" and BACKEND != "fastdllm")]
    if bad:
        raise ValueError(f"policies {bad} do not exist for backend {BACKEND!r}")
    if "bf16" not in names:
        raise ValueError("'bf16' must be among the policies (reference for exactness)")
    names = ["bf16"] + [n for n in names if n != "bf16"]      # bf16 first: reference
    OUT.mkdir(parents=True, exist_ok=True)
    traces = prepare_traces()
    n_max = max(int(t["n"]) for t in traces)
    cases, score_fn = load_cases(n_max)
    runs, have = [], set()
    if RESUME and (OUT / "runs.csv").exists():
        runs = pd.read_csv(OUT / "runs.csv").to_dict("records")
        have = {(r["trace"], r["case_id"]) for r in runs}
        print(f"resume: {len(have)} finished (trace, question) pairs kept", flush=True)

    tokenizer, edge, cloud, link, generate = build_backend()
    print(f"Exp18 | {BACKEND} | {DATASET} | split after layer {SPLIT} | {NEW_TOKENS} tokens | "
          f"RTT {RTT_MS:g} ms | edge slowdown {EDGE_SLOWDOWN:g} | policies {names}", flush=True)
    for t in traces:
        print(f"  trace {t['name']}: mean {t['mean_mbps']:.1f} Mbit/s (p10 {t['p10_mbps']:.1f}, "
              f"p90 {t['p90_mbps']:.1f}), period {t['period_s']:.0f} s, {t['n']} questions",
              flush=True)

    def run(p, case, trace, offset):
        ids, mask = prepare(tokenizer, case["prompt"], cloud)
        wl = Link(rtt_ms=RTT_MS, trace=trace["csv"], trace_offset=offset,
                  header_bytes=HEADER_BYTES, down_trace=trace.get("down_csv"))
        wire = Wire(wl, edge, cloud, edge_slowdown=EDGE_SLOWDOWN)
        ctl = None
        if p["encoding"] == "adaptive":
            ctl = CodecController(mode=p["mode"], rtt_ms=RTT_MS, lam=LAMBDA,
                                  hysteresis=HYSTERESIS, adapt_gamma=p.get("adapt_gamma", False),
                                  gamma_high=GAMMA_HIGH, gamma_low=GAMMA_LOW,
                                  net_share=NET_SHARE)
        link.wire, link.codec_ctl = wire, ctl
        try:
            for d in (edge, cloud):
                torch.cuda.synchronize(d)
            t0 = time.perf_counter()
            res = generate(p, ids, mask, ctl)
            full = torch.cat([ids.to(cloud), res["generated_ids"].to(cloud).unsqueeze(0)], 1)
            fin = wire.final(full)
            measured = time.perf_counter() - t0
        finally:
            link.wire, link.codec_ctl = None, None
            wl.close()
        return res, measured, fin, ctl

    # warm-up without network (kernels, caches); not recorded
    warm = Link(mbps=1000, rtt_ms=0.0)
    link.wire = Wire(warm, edge, cloud)
    ids_w, mask_w = prepare(tokenizer, cases[0]["prompt"], cloud)
    generate(POLICIES["bf16"], ids_w, mask_w, None)
    link.wire = None
    warm.close()
    print("warmup done", flush=True)

    steps_path = OUT / "steps.csv"
    for trace in traces:
        todo = [(i, c) for i, c in enumerate(cases[:int(trace["n"])])
                if (trace["name"], c["case_id"]) not in have]
        if not todo:
            continue
        print(f"\n=== trace {trace['name']} ({len(todo)} questions)  ({time.strftime('%F %T')})",
              flush=True)
        for i, case in todo:
            offset = (i * OFFSET_STEP_S) % trace["period_s"]
            rows, step_rows, ref_ids = [], [], None
            for name in names:
                p = POLICIES[name]
                res, measured, fin, ctl = run(p, case, trace, offset)
                if name == "bf16":
                    ref_ids = res["generated_ids"]
                df = pd.DataFrame(res["split_steps"])
                text = describe(tokenizer, res["generated_ids"])["text"]
                score = score_fn(case, text)
                gam = res.get("gammas") or []
                row = {
                    "trace": trace["name"], "case_id": case["case_id"], "policy": name,
                    "trace_offset_s": offset, "trace_mean_mbps": trace["mean_mbps"],
                    "rtt_ms": RTT_MS, "edge_slowdown": EDGE_SLOWDOWN,
                    "measured_s": measured, "n_steps": len(df),
                    "up_mb": df["up_bytes"].sum() / 1e6,
                    "int8_step_share": float((df["codec"] == "int8_delta").mean())
                    if "codec" in df else 0.0,
                    "mean_gamma": float(np.mean(gam)) if gam else np.nan,
                    "text": text, "extracted": score.get("extracted"), "ok": bool(score["ok"]),
                    "same_as_bf16": bool(torch.equal(ref_ids, res["generated_ids"])),
                    **fin, **(ctl.summary() if ctl is not None else {"ctl_switches": 0}),
                }
                rows.append(row)
                for st in res["split_steps"]:
                    step_rows.append({"trace": trace["name"], "case_id": case["case_id"],
                                      "policy": name, **st})
                print(f"  {case['case_id']} {name:<22s} ok={row['ok']!s:<5} "
                      f"same={row['same_as_bf16']!s:<5} steps={row['n_steps']:4d} "
                      f"up={row['up_mb']:7.1f} MB int8={row['int8_step_share']:.2f} "
                      f"switches={row['ctl_switches']:3d} time={measured:7.1f} s", flush=True)
            runs = [r for r in runs if (r["trace"], r["case_id"]) !=
                    (trace["name"], case["case_id"])] + rows
            pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)
            pd.DataFrame(step_rows).to_csv(steps_path, mode="a", index=False,
                                           header=not steps_path.exists())

    link.remove()
    summary = summarize(runs)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "experiment": "Exp18 codec / gamma controller on bandwidth traces over a real link",
        "backend": BACKEND, "dataset": DATASET, "new_tokens": NEW_TOKENS, "steps": STEPS,
        "split_layer_1based": SPLIT, "policies": {n: POLICIES[n] for n in names},
        "traces": traces, "rtt_ms": RTT_MS, "edge_slowdown": EDGE_SLOWDOWN,
        "controller": {"lambda": LAMBDA, "hysteresis": HYSTERESIS, "net_share": NET_SHARE,
                       "gamma_high": GAMMA_HIGH, "gamma_low": GAMMA_LOW},
        "trace_offset_step_s": OFFSET_STEP_S, "tcpip_header_bytes_per_1460B": HEADER_BYTES,
        "anchor_interval": ANCHOR, "seed": SEED,
        "dream_temperature": DREAM_T if BACKEND == "dllm_cache" else 0.0,
        "block_length": BLOCK if BACKEND == "fastdllm" else None,
        "edge_gpu": torch.cuda.get_device_name(edge), "cloud_gpu": torch.cuda.get_device_name(cloud),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
    }, indent=2, default=str), encoding="utf-8")
    print("\nExp18 completed. Output:", OUT)
    if not summary.empty:
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
