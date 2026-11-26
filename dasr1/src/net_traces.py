"""Bandwidth traces for the network emulator (src/net_relay.py), CSV time_s,mbps.

  mahimahi_to_csv(src, dst, bin_ms=100)
      Mahimahi trace (one line per 1500-byte delivery opportunity, in ms; e.g. the
      cellular traces in https://github.com/ravinet/mahimahi/tree/master/traces)
      -> piecewise-constant bandwidth per bin. The last row closes the period.
  step_csv(dst, high_mbps, low_mbps, period_s)
      synthetic square wave: high for period_s / 2, then low.
  trace_stats(csv) -> mean / p10 / p90 Mbit/s over the period.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

MTU_BYTES = 1500


def mahimahi_to_csv(src, dst, bin_ms=100):
    ts = np.loadtxt(src, dtype=np.int64, ndmin=1)
    if ts.size == 0:
        raise ValueError(f"empty Mahimahi trace {src}")
    n_bins = int(ts.max() // bin_ms) + 1
    counts = np.bincount((ts // bin_ms).astype(np.int64), minlength=n_bins)
    mbps = counts * MTU_BYTES * 8 / (bin_ms / 1000) / 1e6
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "mbps"])
        for i, m in enumerate(mbps):
            w.writerow([f"{i * bin_ms / 1000:.3f}", f"{m:.4f}"])
        w.writerow([f"{n_bins * bin_ms / 1000:.3f}", f"{mbps[0]:.4f}"])   # period end
    return dst


def step_csv(dst, high_mbps=100.0, low_mbps=10.0, period_s=20.0):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "mbps"])
        w.writerow([0.0, high_mbps])
        w.writerow([period_s / 2, low_mbps])
        w.writerow([period_s, high_mbps])
    return dst


def trace_stats(path):
    rows = list(csv.DictReader(open(path, newline="")))
    t = np.array([float(r["time_s"]) for r in rows])
    m = np.array([float(r["mbps"]) for r in rows])
    dt = np.diff(t)
    vals = m[:-1]
    mean = float((vals * dt).sum() / dt.sum())
    rep = np.repeat(vals, np.maximum(1, np.round(dt / dt.min()).astype(int)))
    return {"period_s": float(t[-1]), "mean_mbps": mean,
            "p10_mbps": float(np.percentile(rep, 10)), "p90_mbps": float(np.percentile(rep, 90))}
