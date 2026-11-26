"""How fast could a phone run DASR's edge part? (runs on the PHONE, numpy only)

Times the matrix work of Dream-7B's first k=4 layers (hidden 3584, MLP 18944, 28 query
heads and 4 key/value heads of 128) for m rows per denoising step, in float32 with numpy
on the phone's CPU. To fit in a phone's memory it reuses one layer's weights in chunks
(same arithmetic, ~0.2 GB instead of 3.7 GB).

  pkg install python python-numpy        (Termux)
  python phone_edge_bench.py             (prints a table and one JSON line to send back)

Reference: on the A6000 the edge part takes about 8 ms per step at ~350 rows (Exp11,
static split) and the paper assumes an edge 3x slower. The ratio phone / A6000 is the
measured edge slowdown s of a phone CPU (no NPU or GPU runtime, so an upper bound).
"""
import json
import os
import platform
import time

import numpy as np

D, FF, KV, HEADS, HD = 3584, 18944, 512, 28, 128
LAYERS, CHUNKS = 4, 4


def layer_time(m, ctx, rng, w, reps):
    x = rng.standard_normal((m, D), dtype=np.float32)
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        q = x @ w["q"]
        k = x @ w["kv"]
        v = x @ w["kv"]
        # attention of m queries over ctx keys (GQA: 4 KV heads shared by 28 heads)
        kc = rng.standard_normal((ctx, KV), dtype=np.float32) if ctx > m else k
        qh = q.reshape(m, HEADS, HD).transpose(1, 0, 2)
        kh = np.repeat(kc.reshape(-1, 4, HD).transpose(1, 0, 2), HEADS // 4, axis=0)
        s = qh @ kh.transpose(0, 2, 1)
        a = (s @ kh).transpose(1, 0, 2).reshape(m, D)
        o = a @ w["o"]
        acc = np.zeros((m, D), dtype=np.float32)
        for _c in range(CHUNKS):                  # MLP in 4 chunks of the 18944 columns
            g = o @ w["gate"]
            u = o @ w["up"]
            acc += (g * u) @ w["down"]
        best = min(best, time.perf_counter() - t)
        _ = (v, acc)
    return best


def main():
    rng = np.random.default_rng(0)
    c = FF // CHUNKS
    w = {"q": rng.standard_normal((D, D), dtype=np.float32) * 0.01,
         "kv": rng.standard_normal((D, KV), dtype=np.float32) * 0.01,
         "o": rng.standard_normal((D, D), dtype=np.float32) * 0.01,
         "gate": rng.standard_normal((D, c), dtype=np.float32) * 0.01,
         "up": rng.standard_normal((D, c), dtype=np.float32) * 0.01,
         "down": rng.standard_normal((c, D), dtype=np.float32) * 0.01}
    layer_time(8, 8, rng, w, 1)                   # warm-up
    cases = [(32, 360, "Fast-dLLM in-block step (32 rows, 360 context)"),
             (128, 128, "128 rows"),
             (256, 256, "dLLM-Cache answer rows (~256)"),
             (360, 360, "full boundary, GSM8K (~360 rows)")]
    res = {}
    print(f"{platform.machine()} | {os.cpu_count()} cores | numpy {np.__version__}")
    for m, ctx, label in cases:
        ms = layer_time(m, ctx, rng, w, reps=3) * LAYERS * 1e3
        res[m] = ms
        print(f"  {label:<48s} {ms:9.1f} ms per step (4 layers)")
    s = res[360] / 8.0
    print(f"\nedge slowdown vs the A6000 at ~360 rows (8 ms): about {s:.0f}x (the paper assumes 3x)")
    print("JSON " + json.dumps({"machine": platform.machine(), "cores": os.cpu_count(),
                                "ms_per_step": res, "slowdown_vs_a6000_360rows": s}))


if __name__ == "__main__":
    main()
