"""Exp11b: trace-driven edge-cloud network emulation (no GPU needed).

Run from dasr1/ after Exp11:  python experiments/exp11b_network_emulation.py [exp11 output dir]

For every Exp11 run and every network setting, replays the MEASURED per-step
compute and bytes from steps.csv:

  step latency = slow * (edge_ms + encode_ms)          edge GPU (scaled)
               + up_bytes / bandwidth                  uplink serialization
               + RTT                                   one round trip per step
               + cloud_ms + sample_ms                  cloud GPU
               + down_bytes / bandwidth                downlink (committed tokens)

`slow` emulates a weaker edge GPU than the one measured (1 = same GPU). The
RTT term is the diffusion-specific floor: one round trip per denoising step.

Cloud-only references use the same measurements (both halves on one GPU, one
round trip and the prompt upload): cloud_only = S0 compute, cloud_only_cache =
C1 compute (same cache policy). The intra-server GPU copy time is not used.

Outputs (next to the Exp11 results):
  emulated_runs.csv     one row per (run, network setting)
  emulated_summary.csv  per (variant, setting): mean latency with bootstrap 95% CI,
                        saving vs S0 (static split/full) and overhead vs cloud-only
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import configs.experiment_config as cfg
except Exception:  # pragma: no cover - allows running the script stand-alone
    cfg = None


def setting(name, default):
    return getattr(cfg, name, default) if cfg is not None else default


BANDWIDTHS_MBPS = list(setting("EXP11B_BANDWIDTHS_MBPS", [1, 5, 10, 25, 50, 100, 1000]))
RTTS_MS = list(setting("EXP11B_RTTS_MS", [5, 20, 50, 100]))
EDGE_SLOWDOWNS = list(setting("EXP11B_EDGE_SLOWDOWNS", [1.0, 3.0]))
BOOTSTRAP = int(setting("EXP11B_BOOTSTRAP", 2000))
REFERENCE, CACHE_REF = "S0_split_full", "C1_cache_full"


def per_run_terms(steps):
    """Aggregate the per-step trace into the sums the latency model needs."""
    g = steps.groupby(["case_id", "variant"], sort=False)
    return g.agg(
        n_steps=("step", "size"),
        edge_ms=("edge_ms", "sum"), encode_ms=("encode_ms", "sum"),
        cloud_ms=("cloud_ms", "sum"), sample_ms=("sample_ms", "sum"),
        up_bytes=("up_bytes", "sum"), down_bytes=("down_bytes", "sum"),
    ).reset_index()


def latency_s(t, mbps, rtt_ms, slow):
    bytes_per_ms = mbps * 1e6 / 8 / 1000
    return (slow * (t["edge_ms"] + t["encode_ms"])
            + (t["up_bytes"] + t["down_bytes"]) / bytes_per_ms
            + t["n_steps"] * rtt_ms
            + t["cloud_ms"] + t["sample_ms"]) / 1000


def cloud_only_s(t, rtt_ms):
    """All layers + sampler on the cloud GPU, one round trip (prompt upload ignored)."""
    return (t["edge_ms"] + t["cloud_ms"] + t["sample_ms"] + rtt_ms) / 1000


def bootstrap_ci(values, reps, rng):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or reps <= 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def emulate(steps, runs=None, bandwidths=BANDWIDTHS_MBPS, rtts=RTTS_MS,
            slowdowns=EDGE_SLOWDOWNS, reps=BOOTSTRAP, seed=0):
    terms = per_run_terms(steps)
    quality = {}
    if runs is not None and "cache_ok" in runs:
        ok = runs["cache_ok"].map(lambda v: str(v).strip().lower() == "true")
        quality = dict(zip(zip(runs["case_id"], runs["variant"]), ok))
    rows = []
    for mbps in bandwidths:
        for rtt in rtts:
            for slow in slowdowns:
                lat = latency_s(terms, mbps, rtt, slow)
                for (_, t), value in zip(terms.iterrows(), lat):
                    rows.append({"case_id": t["case_id"], "variant": t["variant"],
                                 "bandwidth_mbps": mbps, "rtt_ms": rtt, "edge_slowdown": slow,
                                 "latency_s": value,
                                 "ok": quality.get((t["case_id"], t["variant"]))})
                for ref, label in ((REFERENCE, "cloud_only"), (CACHE_REF, "cloud_only_cache")):
                    sub = terms[terms["variant"] == ref]
                    for (_, t), value in zip(sub.iterrows(), cloud_only_s(sub, rtt)):
                        rows.append({"case_id": t["case_id"], "variant": label,
                                     "bandwidth_mbps": mbps, "rtt_ms": rtt,
                                     "edge_slowdown": slow, "latency_s": value,
                                     "ok": quality.get((t["case_id"], ref))})
    em = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    keys = ["bandwidth_mbps", "rtt_ms", "edge_slowdown"]
    out = []
    for key, grp in em.groupby(keys + ["variant"], sort=False):
        lo, hi = bootstrap_ci(grp["latency_s"], reps, rng)
        out.append(dict(zip(keys + ["variant"], key), mean_latency_s=grp["latency_s"].mean(),
                        ci95_low=lo, ci95_high=hi, n=len(grp),
                        acc=grp["ok"].mean() if grp["ok"].notna().any() else None))
    summary = pd.DataFrame(out)
    base = summary[summary["variant"] == REFERENCE].set_index(keys)["mean_latency_s"]
    cloud = summary[summary["variant"] == "cloud_only"].set_index(keys)["mean_latency_s"]
    idx = pd.MultiIndex.from_frame(summary[keys])
    summary["saving_vs_S0"] = 1 - summary["mean_latency_s"].values / base.reindex(idx).values
    summary["overhead_vs_cloud_only"] = (summary["mean_latency_s"].values
                                         / cloud.reindex(idx).values - 1)
    return em, summary


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out_dir is None:
        cands = sorted((ROOT / "outputs").glob("exp11_*"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit("No outputs/exp11_* directory found; run Exp11 first")
        out_dir = cands[-1]
    steps = pd.read_csv(out_dir / "steps.csv")
    runs = pd.read_csv(out_dir / "runs.csv") if (out_dir / "runs.csv").exists() else None
    em, summary = emulate(steps, runs)
    em.to_csv(out_dir / "emulated_runs.csv", index=False)
    summary.to_csv(out_dir / "emulated_summary.csv", index=False)
    rtt = 20 if 20 in RTTS_MS else RTTS_MS[0]
    slow = 3.0 if 3.0 in EDGE_SLOWDOWNS else EDGE_SLOWDOWNS[0]
    view = summary[(summary["rtt_ms"] == rtt) & (summary["edge_slowdown"] == slow)]
    print(f"Mean latency (s), RTT {rtt} ms, edge {slow}x slower than measured GPU  [{out_dir}]")
    print(view.pivot(index="variant", columns="bandwidth_mbps",
                     values="mean_latency_s").round(1).to_string())
    print("\nSaving vs S0_split_full:")
    print(view.pivot(index="variant", columns="bandwidth_mbps",
                     values="saving_vs_S0").round(3).to_string())


if __name__ == "__main__":
    main()
